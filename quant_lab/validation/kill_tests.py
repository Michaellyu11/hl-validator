"""Standardized kill-test battery — refactored from
scripts/backtest/large_scale_screen.py + unlock_killtest.py.

Each kill test consumes a list[Event] + horizon list, returns a structured
dict. The engine aggregates these into the verdict.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# --- core Event/SignalResult contract used by all signal functions ---------


@dataclass
class Event:
    """One observation/trade for the hypothesis.

    The signal function returns a list of these. The engine computes
    benchmark sensitivity, LOO, outlier check, etc. from this contract.
    """
    entry_at: datetime
    attribution: dict[str, str]
    trade_direction: int  # +1 long, -1 short
    # horizon_hours -> signed net trade return (already cost-adjusted by
    # the signal function — keeps cost assumptions explicit per spec)
    forward_returns: dict[int, float]
    # Optional pre-computed excess vs benchmarks:
    # benchmark_name -> horizon_hours -> excess return
    benchmark_excess: dict[str, dict[int, float]] = field(
        default_factory=dict
    )
    # v0.2 capital sizing: optional pre-computed net return at each
    # capital level. If present + spec has capital_levels, engine runs
    # the capital sensitivity battery.
    # capital_usd -> horizon -> net_return
    capital_net_returns: dict[float, dict[int, float]] = field(
        default_factory=dict
    )
    metadata: dict = field(default_factory=dict)
    # Sprint 5: explicit attribution key for per-entity Jensen α
    # breakdown. When set, jensen_alpha_by_benchmark also returns a
    # per-key OLS battery. When None, the engine falls back to
    # attribution[spec.primary_attribution_key] (same source LOO
    # uses) so existing signals work without modification.
    attribution_key: str | None = None
    # Sprint 9: regime metric at entry-time (e.g. r_ETH_30d).
    # Signals MAY populate it; if None, engine.run() computes
    # r_ETH_30d from extended_universe_daily ETH closes via
    # event.entry_at.date(). Used by `jensen_alpha_by_regime`
    # to produce a diagnostic per-regime-bucket OLS breakdown.
    regime_metric: float | None = None


@dataclass
class SignalResult:
    events: list[Event]
    notes: list[str] = field(default_factory=list)


# --- stats helpers ---------------------------------------------------------


def _t_stat_ci95(values: list[float]) -> tuple[float | None, tuple | None]:
    n = len(values)
    if n < 2:
        return None, None
    mu = statistics.fmean(values)
    sd = statistics.stdev(values)
    if sd == 0:
        return None, (mu, mu)
    se = sd / math.sqrt(n)
    return mu / se, (mu - 1.96 * se, mu + 1.96 * se)


def _summary(vals: list[float], label: str = "") -> dict:
    n = len(vals)
    if n == 0:
        return {"label": label, "n": 0}
    t, ci = _t_stat_ci95(vals)
    return {
        "label": label,
        "n": n,
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.stdev(vals) if n > 1 else 0.0,
        "pct_positive": sum(1 for v in vals if v > 0) / n * 100,
        "t_stat": t,
        "ci95": ci,
    }


# --- kill tests ------------------------------------------------------------


def baseline_by_horizon(events: list[Event], horizons: list[int]
                        ) -> dict[str, dict]:
    """Raw aggregate per horizon."""
    out: dict[str, dict] = {}
    for H in horizons:
        vals = [
            e.forward_returns[H] for e in events
            if H in e.forward_returns and e.forward_returns[H] is not None
        ]
        out[f"{H}h"] = _summary(vals, f"baseline_{H}h")
    return out


def benchmark_sensitivity(
    events: list[Event], horizons: list[int], benchmarks: list[str]
) -> dict:
    """Excess return vs each named benchmark. Only includes benchmarks
    that the signal function actually populated in Event.benchmark_excess."""
    if not benchmarks:
        return {"skipped": "no benchmarks declared in spec"}
    out: dict = {}
    for bench in benchmarks:
        per_horizon: dict[str, dict] = {}
        for H in horizons:
            vals = [
                e.benchmark_excess.get(bench, {}).get(H)
                for e in events
            ]
            vals = [v for v in vals if v is not None]
            per_horizon[f"{H}h"] = _summary(vals, f"vs_{bench}_{H}h")
        out[bench] = per_horizon
    return out


def _ols_simple_hc0(
    y: list[float], x: list[float]
) -> dict | None:
    """Closed-form OLS for y = α + β·x + ε with HC0 (White, 1980) robust
    SE. Pure stdlib (matches the rest of kill_tests.py — no numpy).

    Returns dict {alpha, alpha_se_cls, alpha_se_hc0, alpha_t_hc0,
    beta, beta_se_cls, beta_se_hc0, beta_t_hc0, beta_t_hc0_vs_1,
    n, r2}, or None if degenerate (n<3, Sxx=0, residual_ss=0).

    Used by jensen_alpha_by_benchmark — exposed at module level so
    other research scripts can use the same canonical implementation.
    """
    n = len(y)
    if n < 3 or len(x) != n:
        return None
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    sxx = sum((xi - x_mean) ** 2 for xi in x)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    if sxx <= 0:
        return None
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean
    # Residuals
    resid = [yi - alpha - beta * xi for xi, yi in zip(x, y)]
    rss = sum(e * e for e in resid)
    tss = sum((yi - y_mean) ** 2 for yi in y)
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    # Classical: σ² = RSS/(n-2)
    sigma2 = rss / (n - 2) if n > 2 else None
    beta_se_cls = math.sqrt(sigma2 / sxx) if sigma2 is not None else None
    alpha_se_cls = (
        math.sqrt(sigma2 * (1.0 / n + x_mean ** 2 / sxx))
        if sigma2 is not None else None
    )
    # HC0 sandwich for 2-param OLS in re-centered form Z=[1, x-x̄]:
    #   (Z'Z)⁻¹ = diag(1/n, 1/Sxx)
    #   Z' diag(e²) Z = [[Σe², Σ(x-x̄)·e²], [·, Σ(x-x̄)²·e²]]
    #   Var = (Z'Z)⁻¹ · meat · (Z'Z)⁻¹
    # → var(α_centered) = Σe² / n²
    #   var(β) = Σ((x-x̄)²·e²) / Sxx²
    # To go from centered intercept to raw α use:
    #   α_raw = α_centered - β·x̄
    #   var(α_raw) = var(α_centered) + x̄² · var(β)
    #                - 2·x̄·cov(α_centered, β)
    # cov(α_centered, β) = Σ((x-x̄)·e²) · 1/(n·Sxx)
    s_e2 = sum(e * e for e in resid)
    s_xe2 = sum((xi - x_mean) * e * e for xi, e in zip(x, resid))
    s_x2e2 = sum((xi - x_mean) ** 2 * e * e for xi, e in zip(x, resid))
    var_alpha_c = s_e2 / (n * n)
    var_beta_hc0 = s_x2e2 / (sxx * sxx)
    cov_ac_b = s_xe2 / (n * sxx)
    var_alpha_hc0 = (
        var_alpha_c + x_mean ** 2 * var_beta_hc0
        - 2.0 * x_mean * cov_ac_b
    )
    alpha_se_hc0 = math.sqrt(var_alpha_hc0) if var_alpha_hc0 > 0 else None
    beta_se_hc0 = math.sqrt(var_beta_hc0) if var_beta_hc0 > 0 else None
    alpha_t_hc0 = (alpha / alpha_se_hc0) if alpha_se_hc0 else None
    beta_t_hc0 = (beta / beta_se_hc0) if beta_se_hc0 else None
    beta_t_hc0_vs_1 = (
        (beta - 1.0) / beta_se_hc0 if beta_se_hc0 else None
    )
    return {
        "n": n,
        "alpha": alpha,
        "alpha_se_cls": alpha_se_cls,
        "alpha_se_hc0": alpha_se_hc0,
        "alpha_t_hc0": alpha_t_hc0,
        "beta": beta,
        "beta_se_cls": beta_se_cls,
        "beta_se_hc0": beta_se_hc0,
        "beta_t_hc0": beta_t_hc0,
        "beta_t_hc0_vs_1": beta_t_hc0_vs_1,
        "r2": r2,
    }


def jensen_alpha_by_benchmark(
    events: list[Event], horizons: list[int], benchmarks: list[str],
    attribution_key_field: str | None = None,
    min_n_per_key: int = 30,
) -> dict:
    """Per-benchmark Jensen α via OLS r_strat = α + β·r_bench + ε.

    Recovers direction-adjusted r_bench algebraically:
        trade_dir · bench_ret = signed - benchmark_excess[bench]
    so this works with the existing Event contract — signal functions
    do NOT need updating.

    Sprint 5: if any event has `attribution_key` set (or falls back
    to `attribution[attribution_key_field]`), the output also
    contains a per-key OLS breakdown for entities with n_events ≥
    `min_n_per_key`. Pure diagnostic — kill-gate still reads cohort.

    Returns:
      Cohort-only mode (default):
        { bench: { "<H>h": <ols dict> } }   ← Sprint 4b-2 shape
      Per-key mode (when keys present, additional sibling block):
        { bench: { "<H>h": <ols dict> },
          "_per_key": {
            "<key>": { bench: { "<H>h": <ols dict> } },
            ...
          }
        }
      The leading underscore on `_per_key` keeps it lexicographically
      distinct from benchmark names without forcing a wrapping change.

    The Jensen α gate (criterion 'jensen_alpha_t_<bench>_at_<H>h > x')
    only reads the cohort path — `_per_key` is informational.
    """
    if not benchmarks:
        return {"skipped": "no benchmarks declared in spec"}

    def _key_for(ev: Event) -> str | None:
        if ev.attribution_key is not None:
            return ev.attribution_key
        if attribution_key_field is not None:
            return ev.attribution.get(attribution_key_field)
        return None

    def _fit(bench: str, H: int, subset: list[Event]) -> dict:
        ys: list[float] = []
        xs: list[float] = []
        for ev in subset:
            r_strat = ev.forward_returns.get(H)
            excess = ev.benchmark_excess.get(bench, {}).get(H)
            if r_strat is None or excess is None:
                continue
            # signed - excess = trade_dir · bench_ret (direction-adj)
            r_bench_dir = r_strat - excess
            ys.append(r_strat)
            xs.append(r_bench_dir)
        res = _ols_simple_hc0(ys, xs)
        return res if res is not None else {
            "n": len(ys), "skipped": "degenerate (n<3 or Sxx=0)",
        }

    out: dict = {}
    # Cohort path — Sprint 4b-2 shape, unchanged
    for bench in benchmarks:
        out[bench] = {f"{H}h": _fit(bench, H, events) for H in horizons}

    # Per-key path (Sprint 5) — only computed when any event has a key
    by_key: dict[str, list[Event]] = {}
    for ev in events:
        k = _key_for(ev)
        if k is None:
            continue
        by_key.setdefault(k, []).append(ev)
    if by_key:
        per_key_out: dict[str, dict] = {}
        for key, subset in by_key.items():
            if len(subset) < min_n_per_key:
                continue
            per_key_out[key] = {
                bench: {f"{H}h": _fit(bench, H, subset) for H in horizons}
                for bench in benchmarks
            }
        if per_key_out:
            out["_per_key"] = per_key_out
    return out


# Sprint 9: regime bucket cutoffs are FIXED (absolute), not data-
# driven quantiles, so the labels (deep bear / bull / etc.) stay
# meaningful regardless of where the strategy actually trades.
# A strategy with all events in Q1 (deep bear) is exactly what
# WhaleFollow v3 showed in Sprint 8.
REGIME_BUCKETS: list[tuple[str, str, float | None, float | None]] = [
    # (key, label, lower_excl, upper_incl) — lower/upper None = unbounded
    ("Q1", "deep_bear",     None, -0.10),
    ("Q2", "moderate_bear", -0.10, -0.05),
    ("Q3", "mild_bear",     -0.05,  0.00),
    ("Q4", "mild_bull",      0.00,  0.05),
    ("Q5", "strong_bull",    0.05,  None),
]


def _regime_bucket(r: float) -> str:
    """Map an r_ETH_30d value to its Q1..Q5 bucket key."""
    for key, _label, lo, hi in REGIME_BUCKETS:
        lo_ok = (lo is None) or (r > lo)
        hi_ok = (hi is None) or (r <= hi)
        if lo_ok and hi_ok:
            return key
    return "Q3"  # fallback should never hit


def jensen_alpha_by_regime(
    events: list[Event], horizons: list[int], benchmark: str = "ETH",
    min_n_per_bucket: int = 30,
) -> dict:
    """Sprint 9: Jensen α conditional on regime bucket.

    For each (regime bucket, horizon) combo, fits the same OLS as
    `jensen_alpha_by_benchmark` on the subset of events whose
    `regime_metric` falls in that bucket. Returns a stable shape
    keyed by bucket key (Q1..Q5).

    DIAGNOSTIC only — the engine kill-gate continues to read cohort
    Jensen α. This battery surfaces "α conditional on regime" so
    users can decide deployment (regime gate at runtime).

    Returns:
      { "<Qn>": {
          "label": "deep_bear",
          "range": [-0.30, -0.10],  (or [None, x] / [x, None])
          "per_horizon": { "<H>h": <ols dict OR insufficient marker> },
          "n_total": int,
        }, ... }
    """
    # Pre-bucket all events with a populated regime_metric
    by_bucket: dict[str, list[Event]] = {}
    for ev in events:
        if ev.regime_metric is None:
            continue
        by_bucket.setdefault(_regime_bucket(ev.regime_metric), []).append(ev)

    out: dict = {}
    for key, label, lo, hi in REGIME_BUCKETS:
        subset = by_bucket.get(key, [])
        per_h: dict = {}
        for H in horizons:
            ys: list[float] = []
            xs: list[float] = []
            for ev in subset:
                r_strat = ev.forward_returns.get(H)
                excess = ev.benchmark_excess.get(benchmark, {}).get(H)
                if r_strat is None or excess is None:
                    continue
                ys.append(r_strat)
                xs.append(r_strat - excess)
            if len(ys) < min_n_per_bucket:
                per_h[f"{H}h"] = {
                    "n": len(ys),
                    "skipped": (
                        f"INSUFFICIENT (n={len(ys)} < {min_n_per_bucket})"
                    ),
                }
                continue
            res = _ols_simple_hc0(ys, xs)
            per_h[f"{H}h"] = res if res is not None else {
                "n": len(ys), "skipped": "degenerate (Sxx=0)",
            }
        out[key] = {
            "label": label,
            "range": [lo, hi],
            "per_horizon": per_h,
            "n_total": len(subset),
        }
    return out


def leave_one_out(
    events: list[Event], horizons: list[int], group_key: str,
) -> dict:
    """For each unique attribution[group_key], drop ALL its events and
    recompute mean+t for the remainder at each horizon. Flag the LOO
    leaver that moves the result the most.
    """
    groups: dict[str, list[Event]] = {}
    for e in events:
        g = e.attribution.get(group_key, "_unknown")
        groups.setdefault(g, []).append(e)
    if len(groups) < 2:
        return {"skipped": (f"only {len(groups)} group(s) — LOO undefined"),
                "n_groups": len(groups)}
    per_horizon: dict[str, dict] = {}
    for H in horizons:
        full = [e.forward_returns[H] for e in events
                if H in e.forward_returns and e.forward_returns[H] is not None]
        full_summary = _summary(full, f"full_{H}h")
        rows = []
        for g, g_events in groups.items():
            keep = [e.forward_returns[H] for e in events
                    if e.attribution.get(group_key) != g
                    and H in e.forward_returns
                    and e.forward_returns[H] is not None]
            s = _summary(keep, f"drop_{g}")
            rows.append({
                "group_dropped": g,
                "n_dropped": len(g_events),
                "n_kept": s["n"],
                "mean_after_drop": s.get("mean"),
                "t_after_drop": s.get("t_stat"),
                "pct_positive_after_drop": s.get("pct_positive"),
            })
        # Find LOO with largest |t_after_drop - full_t|
        full_t = full_summary.get("t_stat") or 0.0
        for r in rows:
            r["delta_t_vs_full"] = (
                (r["t_after_drop"] or 0.0) - full_t
            )
        rows.sort(key=lambda r: -abs(r["delta_t_vs_full"]))
        per_horizon[f"{H}h"] = {
            "full_summary": full_summary,
            "n_groups": len(groups),
            "loo_rows_top10": rows[:10],
            "max_abs_delta_t": (
                abs(rows[0]["delta_t_vs_full"]) if rows else 0.0
            ),
            "fragile": (
                abs(rows[0]["delta_t_vs_full"]) > abs(full_t) * 0.5
                if rows and full_t else False
            ),
        }
    return per_horizon


def outlier_contribution(
    events: list[Event], horizons: list[int]
) -> dict:
    """For each horizon, what fraction of the SUM-OF-RETURNS comes from
    the top-|return| event? A signal that depends on one outlier > 30%
    is fragile."""
    out: dict[str, dict] = {}
    for H in horizons:
        vals = [e.forward_returns[H] for e in events
                if H in e.forward_returns and e.forward_returns[H] is not None]
        if not vals:
            out[f"{H}h"] = {"n": 0}
            continue
        total_abs = sum(abs(v) for v in vals)
        max_v = max(vals, key=abs)
        contrib = abs(max_v) / total_abs if total_abs > 0 else 0.0
        out[f"{H}h"] = {
            "n": len(vals),
            "max_abs_value": max_v,
            "top1_contribution_pct_of_total_abs": contrib * 100,
            "fragile": contrib > 0.30,
        }
    return out


def capital_sensitivity(
    events: list[Event], horizons: list[int], capital_levels: list[float]
) -> dict:
    """For each (capital_usd, horizon), aggregate the net trade returns
    that the signal function pre-computed at that capital level. Find
    the smallest capital at which the focal horizon t-stat crosses zero
    (breakeven). v0.2.
    """
    if not capital_levels:
        return {"skipped": "no capital_levels in spec"}
    if not any(e.capital_net_returns for e in events):
        return {"skipped": ("no Event.capital_net_returns populated; "
                            "signal fn must opt in")}
    per_capital: dict = {}
    H_focus = horizons[-1] if horizons else None
    breakeven_cap = None
    last_pos_cap = None
    for c in capital_levels:
        by_horizon: dict = {}
        for H in horizons:
            vals = [e.capital_net_returns.get(c, {}).get(H)
                    for e in events]
            vals = [v for v in vals if v is not None]
            by_horizon[f"{H}h"] = _summary(vals, f"cap_${c:.0f}_{H}h")
        per_capital[f"${c:,.0f}"] = by_horizon
        if H_focus is not None:
            t = by_horizon.get(f"{H_focus}h", {}).get("t_stat")
            mu = by_horizon.get(f"{H_focus}h", {}).get("mean")
            if mu is not None:
                if mu > 0 and (t is not None and t > 0):
                    last_pos_cap = c
                elif last_pos_cap is not None and breakeven_cap is None:
                    breakeven_cap = c
    return {
        "per_capital": per_capital,
        "focal_horizon": (f"{H_focus}h" if H_focus is not None else None),
        "last_positive_capital_usd": last_pos_cap,
        "first_non_positive_capital_usd": breakeven_cap,
    }


def sign_stability_across_horizons(
    events: list[Event], horizons: list[int]
) -> dict:
    """Sign-stability checks across forward horizons. Returns three
    flags (audit A6 fix):
      - all_same_sign:   all means same sign (positive OR negative)
      - all_positive:    every horizon's mean > 0
      - all_negative:    every horizon's mean < 0
    Horizon-shopping risk = means flip sign across horizons.
    """
    means: dict[int, float] = {}
    for H in horizons:
        vals = [e.forward_returns[H] for e in events
                if H in e.forward_returns and e.forward_returns[H] is not None]
        if vals:
            means[H] = statistics.fmean(vals)
    if len(means) < 2:
        return {"skipped": "need >=2 horizons with data"}
    signs = [1 if m > 0 else (-1 if m < 0 else 0) for m in means.values()]
    all_same_sign = len(set(signs)) == 1
    all_positive = all(m > 0 for m in means.values())
    all_negative = all(m < 0 for m in means.values())
    return {
        "means_by_horizon": {f"{H}h": m for H, m in means.items()},
        "all_same_sign": all_same_sign,
        "all_positive": all_positive,
        "all_negative": all_negative,
        "fragile_horizon_flip": not all_same_sign,
    }


# --- pre-commit gate evaluation --------------------------------------------


def evaluate_pre_commit(
    criteria: list[str],
    baseline: dict[str, dict],
    benchmark: dict,
    loo: dict,
    outlier: dict,
    sign_stab: dict,
    dsr_passes_gate: bool,
    horizons: list[int],
    jensen: dict | None = None,
) -> list[dict]:
    """A LIGHT pre-commit evaluator. Each criterion is a SHORT string;
    the engine parses a small grammar to decide PASS/FAIL.

    Supported criterion forms (the MVP — extend as needed):
      "baseline_t_at_<H>h > <x>"       e.g. "baseline_t_at_24h > 2.0"
      "baseline_t_at_<H>h < <x>"
      "baseline_t_at_<H>h_abs > <x>"
      "vs_<bench>_t_at_<H>h > <x>"     e.g. "vs_BTC_t_at_24h > 2.0"
      "vs_<bench>_t_at_<H>h_abs > <x>"
      "loo_fragile_<H>h == false"      LOO does not flip the verdict
      "outlier_fragile_<H>h == false"  top1 < 30% contribution
      "sign_stability == true"          same sign across all horizons
      "dsr_passes_gate == true"         DSR > 0
      "n_at_<H>h >= <x>"                sample size at horizon

    For any criterion the parser doesn't understand, returns "UNPARSED"
    and counts as a FAIL (conservative).
    """
    rows: list[dict] = []
    for c in criteria:
        verdict, reason = _eval_criterion(c, baseline, benchmark,
                                          loo, outlier, sign_stab,
                                          dsr_passes_gate, horizons,
                                          jensen=jensen)
        rows.append({"criterion": c, "verdict": verdict, "reason": reason})
    return rows


def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _eval_criterion(c: str, baseline, benchmark, loo, outlier, sign_stab,
                    dsr_passes_gate, horizons,
                    jensen: dict | None = None) -> tuple[str, str]:
    c = c.strip()
    try:
        # jensen_alpha_t_<bench>_at_<H>h[_abs] <op> <val>
        # Reads ols["alpha_t_hc0"] from jensen_alpha_by_benchmark output.
        # Sprint 4b-2: replaces β=1 vs_<bench>_t_at gate. See
        # docs/product/engine/JENSEN_ALPHA_GATE_2026_05_30.md.
        if c.startswith("jensen_alpha_t_"):
            if not isinstance(jensen, dict):
                return "FAIL", "jensen battery not run"
            rest = c[len("jensen_alpha_t_"):]
            bench, rest2 = rest.split("_at_", 1)
            parts = rest2.split(None, 2)
            h_part, op, thresh = parts[0], parts[1], parts[2]
            abs_mode = h_part.endswith("_abs")
            if abs_mode:
                h_part = h_part[:-len("_abs")]
            t = _get(jensen, bench, h_part, "alpha_t_hc0", default=None)
            if t is None:
                return ("FAIL",
                        f"no jensen α at {bench}/{h_part} "
                        f"(degenerate or missing data)")
            val = abs(t) if abs_mode else t
            ok = _compare(val, op, float(thresh))
            return ("PASS" if ok else "FAIL",
                    f"jensen α vs {bench} {h_part} "
                    f"t_hc0={val:+.3f} {op} {thresh}: {ok}")
        # Patterns of form "<key> <op> <value>"
        if c.startswith("baseline_t_at_"):
            rest = c[len("baseline_t_at_"):]
            # e.g. "24h_abs > 2.0" or "24h > 2.0"
            parts = rest.split(None, 2)
            h_part, op, thresh = parts[0], parts[1], parts[2]
            abs_mode = h_part.endswith("_abs")
            if abs_mode:
                h_part = h_part[:-len("_abs")]
            t = _get(baseline, h_part, "t_stat", default=None)
            if t is None:
                return "FAIL", f"no t_stat at {h_part}"
            val = abs(t) if abs_mode else t
            ok = _compare(val, op, float(thresh))
            return ("PASS" if ok else "FAIL",
                    f"|t|={val:.3f} at {h_part} {op} {thresh}: {ok}")
        if c.startswith("vs_"):
            # vs_<bench>_t_at_<H>h[_abs] <op> <val>
            rest = c[len("vs_"):]
            bench, rest2 = rest.split("_t_at_", 1)
            parts = rest2.split(None, 2)
            h_part, op, thresh = parts[0], parts[1], parts[2]
            abs_mode = h_part.endswith("_abs")
            if abs_mode:
                h_part = h_part[:-len("_abs")]
            t = _get(benchmark, bench, h_part, "t_stat", default=None)
            if t is None:
                return "FAIL", f"no excess t at vs_{bench} {h_part}"
            val = abs(t) if abs_mode else t
            ok = _compare(val, op, float(thresh))
            return ("PASS" if ok else "FAIL",
                    f"vs_{bench} {h_part} |t|={val:.3f} {op} {thresh}: {ok}")
        if c.startswith("loo_fragile_"):
            rest = c[len("loo_fragile_"):]
            h_part, op, val = rest.split(None, 2)
            fragile = _get(loo, h_part, "fragile", default=None)
            if fragile is None:
                return "FAIL", f"no LOO data at {h_part}"
            expected = (val.lower() == "true")
            ok = (fragile == expected) if op == "==" else None
            return ("PASS" if ok else "FAIL",
                    f"loo_fragile_{h_part}={fragile}; required {op} {val}")
        if c.startswith("outlier_fragile_"):
            rest = c[len("outlier_fragile_"):]
            h_part, op, val = rest.split(None, 2)
            fragile = _get(outlier, h_part, "fragile", default=None)
            if fragile is None:
                return "FAIL", f"no outlier data at {h_part}"
            expected = (val.lower() == "true")
            ok = (fragile == expected) if op == "==" else None
            return ("PASS" if ok else "FAIL",
                    f"outlier_fragile_{h_part}={fragile}; "
                    f"required {op} {val}")
        if c.startswith("sign_stability_positive"):
            parts = c.split(None, 2)
            _, op, val = parts
            actual = sign_stab.get("all_positive")
            expected = (val.lower() == "true")
            ok = (actual == expected) if op == "==" else None
            return ("PASS" if ok else "FAIL",
                    f"sign_stability_positive all_positive={actual}; "
                    f"required {op} {val}")
        if c.startswith("sign_stability_negative"):
            parts = c.split(None, 2)
            _, op, val = parts
            actual = sign_stab.get("all_negative")
            expected = (val.lower() == "true")
            ok = (actual == expected) if op == "==" else None
            return ("PASS" if ok else "FAIL",
                    f"sign_stability_negative all_negative={actual}; "
                    f"required {op} {val}")
        if c.startswith("sign_stability"):
            parts = c.split(None, 2)
            _, op, val = parts
            actual = sign_stab.get("all_same_sign")
            expected = (val.lower() == "true")
            ok = (actual == expected) if op == "==" else None
            return ("PASS" if ok else "FAIL",
                    f"sign_stability all_same_sign={actual}; "
                    f"required {op} {val}")
        if c.startswith("dsr_passes_gate"):
            parts = c.split(None, 2)
            _, op, val = parts
            expected = (val.lower() == "true")
            ok = (dsr_passes_gate == expected) if op == "==" else None
            return ("PASS" if ok else "FAIL",
                    f"dsr_passes_gate={dsr_passes_gate}; "
                    f"required {op} {val}")
        if c.startswith("n_at_"):
            rest = c[len("n_at_"):]
            parts = rest.split(None, 2)
            h_part, op, val = parts
            n = _get(baseline, h_part, "n", default=0)
            ok = _compare(n, op, float(val))
            return ("PASS" if ok else "FAIL",
                    f"n at {h_part}={n}; required {op} {val}")
        return "UNPARSED", f"no parser for '{c}'"
    except Exception as e:
        return "FAIL", f"exception while parsing '{c}': {e}"


def _compare(a: float, op: str, b: float) -> bool:
    return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b,
            "==": a == b, "!=": a != b}.get(op, False)
