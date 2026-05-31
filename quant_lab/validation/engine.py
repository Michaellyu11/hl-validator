"""Validation engine — orchestrates the kill-test pipeline."""
from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dsr import deflated_sharpe, DSRResult
from .kill_tests import (
    Event, SignalResult,
    baseline_by_horizon, benchmark_sensitivity, jensen_alpha_by_benchmark,
    jensen_alpha_by_regime,
    leave_one_out, outlier_contribution,
    sign_stability_across_horizons,
    capital_sensitivity,
    evaluate_pre_commit,
)
from .spec import (
    HypothesisSpec, read_trial_counter, increment_trial_counter,
    DEFAULT_COUNTER_PATH,
)
from .explanation import generate_explanation, generate_operator_section


DOCS_DIR = Path(os.environ.get("HL_VALIDATOR_DOCS_DIR", "docs/validations"))


@dataclass
class ValidationReport:
    spec: HypothesisSpec
    signal_notes: list[str]
    trial_counter_before: int
    trial_counter_after: int
    baseline: dict[str, Any]
    benchmark_sensitivity: dict[str, Any]
    jensen_alpha: dict[str, Any]
    jensen_alpha_by_regime: dict[str, Any]
    leave_one_out: dict[str, Any]
    outlier_contribution: dict[str, Any]
    sign_stability: dict[str, Any]
    dsr_per_horizon: dict[str, dict[str, Any]]
    pre_commit_results: list[dict[str, Any]]
    verdict: str  # "SURVIVOR" / "KILL" / "UNDERPOWERED" / "WAITING-ON-DATA"
    verdict_tentative: bool  # tentative_pending_oos flag from spec
    n_events: int
    horizons: list[int]
    capital_sensitivity: dict[str, Any] = field(default_factory=dict)
    run_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spec"] = self.spec.to_yaml_dict()
        return d


# Public note: regime-classification price DB is configurable. Set
# HL_VALIDATOR_DAILY_DB to your own daily-close DuckDB; the data layer is
# private and not shipped in this public repo.
_REGIME_ETH_DEFAULT_DB = os.environ.get(
    "HL_VALIDATOR_DAILY_DB",
    "data/extended_universe_daily/daily.duckdb",
)
_REGIME_LOOKBACK_DAYS = 30


def _populate_regime_metrics(events: list, spec: Any) -> None:
    """Sprint 9: populate Event.regime_metric = r_ETH_30d for any event
    whose signal didn't pre-fill it. Uses ETH daily closes from
    `spec.signal_config["daily_archive_db"]` if present, otherwise the
    canonical extended_universe_daily DB.

    Silently no-ops if (a) no events need population, (b) ETH closes
    cannot be loaded (missing DB / missing ETH coin / I/O error), or
    (c) any individual event's lookback close is unavailable. Errors
    are NEVER raised — regime analysis is diagnostic, not gating.
    """
    needs = [e for e in events if e.regime_metric is None]
    if not needs:
        return
    # Resolve daily-DB path
    db_path = None
    cfg = getattr(spec, "signal_config", {}) or {}
    if isinstance(cfg, dict):
        db_path = cfg.get("daily_archive_db")
    if not db_path:
        db_path = _REGIME_ETH_DEFAULT_DB
    try:
        import duckdb  # local import — engine otherwise has no duckdb dep
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = conn.execute(
                "SELECT date, close FROM daily_close WHERE coin = 'ETH'"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return  # silent fallback — regime analysis is best-effort
    eth_close = {d.strftime("%Y-%m-%d"): float(c) for d, c in rows}
    if not eth_close:
        return
    from datetime import timedelta as _td
    for ev in needs:
        # entry_at is required on all Events; use its date as anchor
        if getattr(ev, "entry_at", None) is None:
            continue
        d = ev.entry_at.date()
        s_entry = d.strftime("%Y-%m-%d")
        s_back = (d - _td(days=_REGIME_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        c_entry = eth_close.get(s_entry)
        c_back = eth_close.get(s_back)
        if c_entry is None or c_back is None or c_back == 0:
            continue
        ev.regime_metric = (c_entry / c_back) - 1.0


def _verdict_for(n_events: int, pre_commit_results: list[dict],
                 horizons: list[int], baseline: dict,
                 min_n_for_test: int = 30) -> str:
    """Assign one of the 4 standardized verdicts."""
    if n_events == 0:
        return "WAITING-ON-DATA"
    max_n = max(
        baseline.get(f"{H}h", {}).get("n", 0) or 0 for H in horizons
    )
    if max_n < min_n_for_test:
        return "UNDERPOWERED"
    # If any pre-commit criterion is UNPARSED or FAIL -> KILL
    failed = [r for r in pre_commit_results
              if r["verdict"] in ("FAIL", "UNPARSED")]
    if failed:
        return "KILL"
    return "SURVIVOR"


def _compute_sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    sd = statistics.stdev(values)
    if sd == 0:
        return 0.0
    return statistics.fmean(values) / sd


def run(spec: HypothesisSpec,
        counter_path: Path = DEFAULT_COUNTER_PATH,
        force_unlocked: bool = False,
        dry_run: bool = False) -> ValidationReport:
    """Run a hypothesis through the full kill-test pipeline.

    `dry_run=True` skips the trial counter increment (useful for the very
    first run when iterating on the signal function). Default is False —
    every real run consumes a trial.
    """
    if not force_unlocked:
        spec.ensure_locked()
    if spec.factor_type == "cross_sectional":
        return _run_cross_sectional(spec, counter_path, dry_run)

    # 1) Trial counter
    counter_state = read_trial_counter(counter_path)
    trial_counter_before = counter_state.get("trial_count", 0)

    # 2) Load + call signal function
    fn = spec.load_signal_function()
    sig_result: SignalResult = fn(spec.signal_config)
    if not isinstance(sig_result, SignalResult):
        raise TypeError(
            f"signal_function {spec.signal_module}:{spec.signal_function} "
            f"returned {type(sig_result).__name__}, expected SignalResult"
        )
    events = sig_result.events
    horizons = spec.forward_horizons

    # 3) Kill-test battery
    baseline = baseline_by_horizon(events, horizons)
    benchmark = benchmark_sensitivity(events, horizons, spec.benchmarks)
    # Sprint 4b-2: Jensen α battery — OLS r_strat = α + β·r_bench + ε
    # per benchmark. The Jensen α gate (criterion
    # 'jensen_alpha_t_<bench>_at_<H>h > x') reads this. The β=1 excess
    # gate (vs_<bench>_t_at_<H>h) stays computed alongside for
    # comparability + transparency. See
    # docs/product/engine/JENSEN_ALPHA_GATE_2026_05_30.md.
    # Sprint 5: pass primary_attribution_key so per-entity Jensen α
    # breakdown is computed for signals that already populate the
    # attribution dict (e.g. whale_address for WhaleFollow). Signals
    # that explicitly set Event.attribution_key win over the dict
    # fallback.
    jensen = jensen_alpha_by_benchmark(
        events, horizons, spec.benchmarks,
        attribution_key_field=spec.primary_attribution_key,
    )
    # Sprint 9: regime-conditional Jensen α (diagnostic).
    # Populates event.regime_metric = r_ETH_30d when absent.
    # See docs/product/engine/REGIME_CONDITIONAL_VERDICT_2026_05_30.md.
    _populate_regime_metrics(events, spec)
    jensen_regime = (
        jensen_alpha_by_regime(events, horizons, benchmark="ETH")
        if "ETH" in spec.benchmarks
        else {"skipped": "ETH not in spec.benchmarks"}
    )
    loo = leave_one_out(events, horizons, spec.primary_attribution_key)
    outlier = outlier_contribution(events, horizons)
    sign_stab = sign_stability_across_horizons(events, horizons)
    cap_sens = capital_sensitivity(
        events, horizons, spec.capital_levels
    )

    # 4) Deflated Sharpe — per horizon, against the TRIAL COUNTER BEFORE
    # this hypothesis is added.
    dsr_per_horizon: dict = {}
    n_trials_for_dsr = max(trial_counter_before, 1)
    for H in horizons:
        vals = [e.forward_returns[H] for e in events
                if H in e.forward_returns
                and e.forward_returns[H] is not None]
        if len(vals) < 2:
            dsr_per_horizon[f"{H}h"] = {
                "untestable": True,
                "n_obs": len(vals),
            }
            continue
        sr = _compute_sharpe(vals)
        result = deflated_sharpe(
            sr_observed=sr,
            n_obs=len(vals),
            n_trials=n_trials_for_dsr,
            note=f"H={H}h n_events={len(vals)} "
                 f"n_trials_at_test={n_trials_for_dsr}",
        )
        dsr_per_horizon[f"{H}h"] = result.to_dict()

    # 5) Pre-commit evaluation
    dsr_passes_any_horizon = any(
        d.get("passes_gate", False)
        for d in dsr_per_horizon.values()
        if isinstance(d, dict) and not d.get("untestable")
    )
    pre_commit_results = evaluate_pre_commit(
        spec.pre_commit_kill_criteria,
        baseline=baseline,
        benchmark=benchmark,
        loo=loo,
        outlier=outlier,
        sign_stab=sign_stab,
        dsr_passes_gate=dsr_passes_any_horizon,
        horizons=horizons,
        jensen=jensen,
    )

    # 6) Assign verdict
    verdict = _verdict_for(
        n_events=len(events),
        pre_commit_results=pre_commit_results,
        horizons=horizons,
        baseline=baseline,
    )

    # 7) Increment trial counter (unless dry-run or WAITING-ON-DATA)
    if dry_run or verdict == "WAITING-ON-DATA":
        trial_counter_after = trial_counter_before
    else:
        state = increment_trial_counter(
            spec.hypothesis_id,
            spec.trial_counter_increment,
            verdict,
            counter_path,
        )
        trial_counter_after = state["trial_count"]

    return ValidationReport(
        spec=spec,
        signal_notes=sig_result.notes,
        trial_counter_before=trial_counter_before,
        trial_counter_after=trial_counter_after,
        baseline=baseline,
        benchmark_sensitivity=benchmark,
        jensen_alpha=jensen,
        jensen_alpha_by_regime=jensen_regime,
        leave_one_out=loo,
        outlier_contribution=outlier,
        sign_stability=sign_stab,
        capital_sensitivity=cap_sens,
        dsr_per_horizon=dsr_per_horizon,
        pre_commit_results=pre_commit_results,
        verdict=verdict,
        verdict_tentative=spec.tentative_pending_oos,
        n_events=len(events),
        horizons=horizons,
    )


def _run_cross_sectional(
    spec: HypothesisSpec, counter_path: Path, dry_run: bool
) -> ValidationReport:
    """Dispatch for factor_type=cross_sectional. Reuses the same
    ValidationReport shape so the verdict doc template still works.
    cross-sectional stats are stuffed into `report.baseline` and
    `report.leave_one_out` as analogous structures."""
    from .cross_sectional import (
        CrossSectionalResult, evaluate_cross_sectional,
        eval_cross_sectional_criterion,
    )
    counter_state = read_trial_counter(counter_path)
    tc_before = counter_state.get("trial_count", 0)

    fn = spec.load_signal_function()
    cs_result = fn(spec.signal_config)
    if not isinstance(cs_result, CrossSectionalResult):
        raise TypeError(
            f"cross_sectional signal must return CrossSectionalResult, "
            f"got {type(cs_result).__name__}"
        )
    horizons = spec.forward_horizons
    cs_stats = evaluate_cross_sectional(cs_result, horizons)

    # Coerce into the report's baseline + LOO shape so the existing
    # template can render meaningful tables.
    baseline: dict = {}
    loo: dict = {}
    for H in horizons:
        ph = cs_stats["per_horizon"][f"{H}h"]
        ic = ph["ic"]
        baseline[f"{H}h"] = {
            "label": f"ic_{H}h",
            "n": ic.get("n", 0),
            "mean": ic.get("mean", 0.0),
            "median": ic.get("mean", 0.0),
            "std": ic.get("std", 0.0),
            "pct_positive": ic.get("pct_positive", 0.0),
            "t_stat": ic.get("t_stat"),
            "ci95": None,
        }
        loo[f"{H}h"] = {
            "full_summary": baseline[f"{H}h"],
            "n_groups": ph["loo_by_coin"].get("n_coins", 0),
            "loo_rows_top10": ph["loo_by_coin"].get("top5_loo", []),
            "max_abs_delta_t": ph["loo_by_coin"].get(
                "max_abs_delta_t", 0
            ),
            "fragile": ph["loo_by_coin"].get("fragile", False),
        }
    sign_stab = {
        "means_by_horizon": {
            f"{H}h": baseline[f"{H}h"]["mean"] for H in horizons
        },
        "all_same_sign": (len({
            (1 if baseline[f"{H}h"]["mean"] > 0
             else -1 if baseline[f"{H}h"]["mean"] < 0 else 0)
            for H in horizons
        }) == 1),
        "all_positive": all(baseline[f"{H}h"]["mean"] > 0
                            for H in horizons),
        "all_negative": all(baseline[f"{H}h"]["mean"] < 0
                            for H in horizons),
    }
    sign_stab["fragile_horizon_flip"] = not sign_stab["all_same_sign"]

    # DSR on IC time series at each horizon
    dsr_per_horizon: dict = {}
    n_trials_for_dsr = max(tc_before, 1)
    for H in horizons:
        s = baseline[f"{H}h"]
        n = s["n"]
        sr = s["mean"] / s["std"] if s["std"] > 0 else 0
        if n < 2 or s["std"] == 0:
            dsr_per_horizon[f"{H}h"] = {
                "untestable": True, "n_obs": n,
            }
            continue
        from .dsr import deflated_sharpe
        d = deflated_sharpe(sr, n, n_trials_for_dsr,
                            note=f"cross_sectional IC, H={H}")
        dsr_per_horizon[f"{H}h"] = d.to_dict()

    # Pre-commit: try main grammar, then cross-sectional grammar
    pc_results: list[dict] = []
    for c in spec.pre_commit_kill_criteria:
        # Main grammar first (covers ic_ via baseline if mapped)
        from .kill_tests import _eval_criterion
        v, r = _eval_criterion(c, baseline, {}, loo, {}, sign_stab,
                                any(d.get("passes_gate", False)
                                    for d in dsr_per_horizon.values()
                                    if isinstance(d, dict)
                                    and not d.get("untestable")),
                                horizons)
        if v == "UNPARSED":
            v, r = eval_cross_sectional_criterion(c, cs_stats)
        pc_results.append({"criterion": c, "verdict": v, "reason": r})

    verdict = _verdict_for(
        n_events=cs_stats["n_obs_total"],
        pre_commit_results=pc_results,
        horizons=horizons,
        baseline=baseline,
    )
    if dry_run or verdict == "WAITING-ON-DATA":
        tc_after = tc_before
    else:
        st = increment_trial_counter(
            spec.hypothesis_id, spec.trial_counter_increment,
            verdict, counter_path,
        )
        tc_after = st["trial_count"]

    return ValidationReport(
        spec=spec,
        signal_notes=cs_result.notes + [
            f"cross-sectional N_obs={cs_stats['n_obs_total']:,} "
            f"coins={cs_stats['n_coins']} "
            f"timestamps={cs_stats['n_timestamps']:,}"
        ],
        trial_counter_before=tc_before,
        trial_counter_after=tc_after,
        baseline=baseline,
        benchmark_sensitivity={"skipped": "n/a for cross-sectional"},
        jensen_alpha={"skipped": "n/a for cross-sectional"},
        jensen_alpha_by_regime={"skipped": "n/a for cross-sectional"},
        leave_one_out=loo,
        outlier_contribution={
            f"{H}h": {"n": baseline[f"{H}h"]["n"],
                      "top1_contribution_pct_of_total_abs": 0.0,
                      "fragile": False}
            for H in horizons
        },
        sign_stability=sign_stab,
        dsr_per_horizon=dsr_per_horizon,
        pre_commit_results=pc_results,
        verdict=verdict,
        verdict_tentative=spec.tentative_pending_oos,
        n_events=cs_stats["n_obs_total"],
        horizons=horizons,
        capital_sensitivity={
            "skipped": "n/a for cross-sectional (no Event.capital_net_returns)"
        },
    )


def render_verdict_doc(report: ValidationReport) -> str:
    """Render the markdown verdict doc using the bundled template."""
    template_path = (Path(__file__).parent / "templates" / "verdict.md")
    template = template_path.read_text()
    spec = report.spec

    def fmt_pct(x):
        return f"{x * 100:+.3f}%" if isinstance(x, (int, float)) else str(x)

    def fmt_t(x):
        return f"{x:+.2f}" if isinstance(x, (int, float)) else str(x)

    # Build baseline table rows
    baseline_rows = []
    for H in report.horizons:
        s = report.baseline.get(f"{H}h", {})
        if s.get("n", 0) == 0:
            baseline_rows.append(
                f"| {H}h | 0 | — | — | — | — |"
            )
            continue
        ci = s.get("ci95")
        ci_str = (f"({ci[0]*100:+.3f}%, {ci[1]*100:+.3f}%)" if ci
                  else "—")
        baseline_rows.append(
            f"| {H}h | {s['n']:,} | {fmt_pct(s['mean'])} | "
            f"{fmt_pct(s.get('median', 0))} | {fmt_t(s.get('t_stat'))} | "
            f"{ci_str} |"
        )

    # Benchmark rows
    bench_rows = []
    for bench, per_h in (report.benchmark_sensitivity.items()
                          if isinstance(report.benchmark_sensitivity,
                                        dict) else []):
        if not isinstance(per_h, dict) or "skipped" in per_h:
            continue
        for H in report.horizons:
            s = per_h.get(f"{H}h", {})
            if s.get("n", 0) == 0:
                continue
            ci = s.get("ci95")
            ci_str = (f"({ci[0]*100:+.3f}%, {ci[1]*100:+.3f}%)" if ci
                      else "—")
            bench_rows.append(
                f"| {bench} | {H}h | {s['n']:,} | "
                f"{fmt_pct(s['mean'])} | {fmt_t(s.get('t_stat'))} | "
                f"{ci_str} |"
            )

    # Jensen α rows (Sprint 4b-2) — β-free benchmark sensitivity
    jensen_rows = []
    ja = getattr(report, "jensen_alpha", {}) or {}
    if isinstance(ja, dict) and "skipped" not in ja:
        for bench, per_h in ja.items():
            # Sprint 5: '_per_key' is a sibling dict, not a benchmark
            if bench == "_per_key" or not isinstance(per_h, dict):
                continue
            for H in report.horizons:
                s = per_h.get(f"{H}h", {})
                if not isinstance(s, dict) or "alpha" not in s:
                    continue
                jensen_rows.append(
                    f"| {bench} | {H}h | {s.get('n', 0):,} | "
                    f"{fmt_pct(s.get('alpha') or 0)} | "
                    f"{fmt_t(s.get('alpha_t_hc0'))} | "
                    f"{(s.get('beta') or 0):+.4f} | "
                    f"{fmt_t(s.get('beta_t_hc0_vs_1'))} | "
                    f"{(s.get('r2') or 0)*100:.1f}% |"
                )

    # Sprint 5: per-attribution-key Jensen α breakdown.
    # Truncate display at 30 rows to keep verdict markdown readable
    # (cross-sectional pipelines can produce 100+ rows; full data
    # still flows through the API).
    JENSEN_PER_KEY_DISPLAY_CAP = 30
    jensen_per_key_rows = []
    per_key = ja.get("_per_key") if isinstance(ja, dict) else None
    if isinstance(per_key, dict):
        for key, per_bench in per_key.items():
            if not isinstance(per_bench, dict):
                continue
            for bench, per_h in per_bench.items():
                if not isinstance(per_h, dict):
                    continue
                for H in report.horizons:
                    s = per_h.get(f"{H}h", {})
                    if not isinstance(s, dict) or "alpha" not in s:
                        continue
                    # Display key short (last 8 chars for 42-char hex
                    # addresses, leave alone if shorter)
                    key_disp = key if len(key) <= 16 else "…" + key[-12:]
                    jensen_per_key_rows.append(
                        f"| `{key_disp}` | {bench} | {H}h | "
                        f"{s.get('n', 0):,} | "
                        f"{fmt_pct(s.get('alpha') or 0)} | "
                        f"{fmt_t(s.get('alpha_t_hc0'))} | "
                        f"{(s.get('beta') or 0):+.4f} | "
                        f"{fmt_t(s.get('beta_t_hc0_vs_1'))} | "
                        f"{(s.get('r2') or 0)*100:.1f}% |"
                    )
    n_per_key_total = len(jensen_per_key_rows)
    jensen_per_key_truncated_note = ""
    if n_per_key_total > JENSEN_PER_KEY_DISPLAY_CAP:
        jensen_per_key_rows = jensen_per_key_rows[:JENSEN_PER_KEY_DISPLAY_CAP]
        jensen_per_key_truncated_note = (
            f"\n_… {n_per_key_total - JENSEN_PER_KEY_DISPLAY_CAP} "
            f"additional rows truncated; full breakdown available "
            f"via API field `jensen_alpha.per_key`._"
        )

    # Sprint 9: regime-conditional Jensen α (diagnostic) rows.
    regime_rows = []
    jbr = getattr(report, "jensen_alpha_by_regime", {}) or {}
    if isinstance(jbr, dict) and "skipped" not in jbr:
        # Stable Q1..Q5 ordering
        for q_key in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            bucket = jbr.get(q_key, {})
            if not isinstance(bucket, dict):
                continue
            label = bucket.get("label", "")
            rng = bucket.get("range", [None, None])
            lo = rng[0] if isinstance(rng, list) else None
            hi = rng[1] if isinstance(rng, list) else None
            lo_str = f"{lo*100:+.0f}%" if lo is not None else "−∞"
            hi_str = f"{hi*100:+.0f}%" if hi is not None else "+∞"
            range_str = f"({lo_str}, {hi_str}]"
            n_total = bucket.get("n_total", 0)
            per_h = bucket.get("per_horizon", {})
            for H in report.horizons:
                s = per_h.get(f"{H}h", {})
                if isinstance(s, dict) and "skipped" in s:
                    regime_rows.append(
                        f"| {q_key} {label} | {range_str} | {H}h | "
                        f"{s.get('n', 0):,} | _{s['skipped']}_ |  |  |"
                    )
                    continue
                if isinstance(s, dict) and "alpha" in s:
                    regime_rows.append(
                        f"| {q_key} {label} | {range_str} | {H}h | "
                        f"{s.get('n', 0):,} | "
                        f"{fmt_pct(s.get('alpha') or 0)} | "
                        f"{fmt_t(s.get('alpha_t_hc0'))} | "
                        f"{(s.get('r2') or 0)*100:.1f}% |"
                    )

    # LOO rows
    loo_rows = []
    if isinstance(report.leave_one_out, dict):
        for H in report.horizons:
            d = report.leave_one_out.get(f"{H}h", {})
            if isinstance(d, dict) and "skipped" not in d:
                loo_rows.append(
                    f"| {H}h | {d.get('n_groups', 0)} | "
                    f"{d.get('max_abs_delta_t', 0):.2f} | "
                    f"{'⚠ YES' if d.get('fragile') else 'no'} |"
                )

    # Outlier rows
    outlier_rows = []
    for H in report.horizons:
        o = report.outlier_contribution.get(f"{H}h", {})
        if o.get("n", 0) == 0:
            continue
        outlier_rows.append(
            f"| {H}h | {o['n']:,} | "
            f"{o.get('top1_contribution_pct_of_total_abs', 0):.1f}% | "
            f"{'⚠ YES' if o.get('fragile') else 'no'} |"
        )

    # DSR rows
    dsr_rows = []
    for H in report.horizons:
        d = report.dsr_per_horizon.get(f"{H}h", {})
        if d.get("untestable"):
            dsr_rows.append(f"| {H}h | — | — | — | untestable (n<2) |")
            continue
        dsr_rows.append(
            f"| {H}h | {d.get('sr_observed', 0):+.3f} | "
            f"{d.get('expected_max_sr', 0):+.3f} | "
            f"{d.get('dsr_sharpe_units', 0):+.3f} | "
            f"{'PASS' if d.get('passes_gate') else 'FAIL'} |"
        )

    # Capital sensitivity rows (v0.2)
    cap_rows = []
    cs = report.capital_sensitivity
    if isinstance(cs, dict) and "per_capital" in cs:
        focal = cs.get("focal_horizon")
        for cap_label, by_h in cs["per_capital"].items():
            s = by_h.get(focal, {}) if focal else {}
            if not s or s.get("n", 0) == 0:
                continue
            cap_rows.append(
                f"| {cap_label} | {focal} | {s['n']:,} | "
                f"{fmt_pct(s['mean'])} | {fmt_t(s.get('t_stat'))} | "
                f"{s.get('pct_positive', 0):.0f}% |"
            )
        be = cs.get("first_non_positive_capital_usd")
        lp = cs.get("last_positive_capital_usd")
        if be is not None and lp is not None:
            be_str = (f"net positive up to ${lp:,.0f}; "
                      f"flips negative at ${be:,.0f}")
        elif lp is not None:
            be_str = (f"stayed net positive through highest "
                      f"(${lp:,.0f}); no breakeven crossing observed")
        else:
            be_str = "never net positive at any tested capital level"
        cap_rows.append(f"| _summary_ | _{focal}_ | — | — | — | {be_str} |")

    # Pre-commit
    pc_rows = []
    for r in report.pre_commit_results:
        sym = ("✅" if r["verdict"] == "PASS"
               else ("❌" if r["verdict"] == "FAIL" else "⚠"))
        pc_rows.append(
            f"| {sym} {r['verdict']} | `{r['criterion']}` | "
            f"{r['reason']} |"
        )

    # Verdict banner
    if report.verdict == "SURVIVOR":
        verdict_banner = (
            "🟢 **SURVIVOR**" + (
                " ⚠ **TENTATIVE — PENDING OUT-OF-SAMPLE**"
                if report.verdict_tentative else ""
            )
        )
    elif report.verdict == "KILL":
        verdict_banner = "🔴 **KILL** — pre-commit criteria failed"
    elif report.verdict == "UNDERPOWERED":
        verdict_banner = (
            "🟡 **UNDERPOWERED** — sample below threshold (n<30)"
        )
    else:
        verdict_banner = (
            "⚪ **WAITING-ON-DATA** — signal returned 0 events"
        )

    # Render
    return template.format(
        hypothesis_id=spec.hypothesis_id,
        name=spec.name,
        run_at=report.run_at,
        verdict_banner=verdict_banner,
        n_events=report.n_events,
        spec_locked_at=spec.locked_at or "(not locked)",
        spec_tentative=("YES" if spec.tentative_pending_oos else "no"),
        trial_counter_before=report.trial_counter_before,
        trial_counter_after=report.trial_counter_after,
        signal_notes="\n".join(
            f"- {n}" for n in report.signal_notes
        ) or "_(none)_",
        mechanism=spec.mechanism,
        data_provenance=spec.data_provenance_notes,
        baseline_rows="\n".join(baseline_rows) or "| (none) |",
        bench_rows="\n".join(bench_rows) or "| (no benchmarks computed) |",
        jensen_rows="\n".join(jensen_rows) or (
            "| (no Jensen α computed — set spec.benchmarks) |"
        ),
        jensen_per_key_rows=("\n".join(jensen_per_key_rows) or (
            "| (no per-key breakdown — signal didn't populate "
            "Event.attribution_key) |"
        )) + jensen_per_key_truncated_note,
        regime_rows="\n".join(regime_rows) or (
            "| (regime analysis skipped — ETH not in benchmarks "
            "or daily-close DB unavailable) |"
        ),
        loo_rows="\n".join(loo_rows) or "| (LOO not applicable) |",
        outlier_rows="\n".join(outlier_rows) or "| (none) |",
        dsr_rows="\n".join(dsr_rows) or "| (none) |",
        cap_rows="\n".join(cap_rows) or "| (capital sizing not enabled) |",
        pc_rows="\n".join(pc_rows) or "| (no criteria) |",
        plain_english_explanation=generate_explanation(report),
        operator_section=generate_operator_section(report),
    )


def write_verdict_doc(report: ValidationReport,
                      docs_dir: Path = DOCS_DIR) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    ts = report.run_at.replace(":", "-").replace("+", "_")
    out_path = (docs_dir
                / f"{report.spec.hypothesis_id}_{ts[:16]}Z.md")
    out_path.write_text(render_verdict_doc(report))
    return out_path
