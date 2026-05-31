"""Sprint 16: generate a natural-language explanation of the verdict.

Rule-based (no LLM) — reproducible, no external API, no hallucination
risk. Templates branch on verdict status (SURVIVOR/KILL/UNDERPOWERED/
WAITING-ON-DATA); each cites the SPECIFIC numbers that drove the
outcome.

Goal (per Sprint 16 dispatch): an LLM agent or human user reading
the verdict markdown should understand "why KILL" without manually
mapping "DSR=0.3" → "70% likely overfitted".

Constraint: ≤ 200 words per explanation (dispatch hard stop).
"""
from __future__ import annotations

import re
from typing import Any

# Regex shared across reason-string parsing. Mirrors patterns the
# hl_validator engine_adapter/verdict_parser.py uses, so we stay
# consistent with the API layer.
_PAT_T_STAT = re.compile(
    r"\|t\|=([-+]?\d+(?:\.\d+)?)"
    r"\s+at\s+\w+\s+[><=!]+\s+([-+]?\d+(?:\.\d+)?)"
)
_PAT_JENSEN_T = re.compile(
    r"t_hc0=([-+]?\d+(?:\.\d+)?)\s+[><=!]+\s+([-+]?\d+(?:\.\d+)?)"
)
_PAT_COUNT = re.compile(
    r"=(\d+(?:\.\d+)?)\s*;\s*required\s+(?:>=|<=|>|<|==|!=)\s+"
    r"(\d+(?:\.\d+)?)"
)


def _parse_reason_numbers(reason: str) -> tuple[float | None, float | None]:
    """Extract (observed, threshold) from a pre_commit reason string."""
    if not reason:
        return None, None
    for pat in (_PAT_JENSEN_T, _PAT_T_STAT, _PAT_COUNT):
        m = pat.search(reason)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:
                continue
    return None, None


def _criterion_label(criterion: str) -> str:
    """Translate a criterion key into a short English phrase."""
    c = criterion.strip()
    # Pre-commit gate names → human English
    if c.startswith("n_at_"):
        return "sample size"
    if c.startswith("baseline_t_at_"):
        return "baseline t-statistic"
    if c.startswith("jensen_alpha_t_BTC_at_"):
        return "Jensen α vs BTC"
    if c.startswith("jensen_alpha_t_ETH_at_"):
        return "Jensen α vs ETH"
    if c.startswith("vs_BTC_t_at_"):
        return "β=1 excess vs BTC (legacy gate)"
    if c.startswith("vs_ETH_t_at_"):
        return "β=1 excess vs ETH (legacy gate)"
    if c.startswith("sign_stability"):
        return "sign stability across horizons"
    if c.startswith("loo_fragile_"):
        return "leave-one-out fragility"
    if c.startswith("outlier_fragile_"):
        return "outlier dominance"
    if c.startswith("dsr_passes_gate"):
        return "Deflated Sharpe Ratio"
    return c


def _explain_failed_gate(
    criterion: str, reason: str,
) -> tuple[str, str]:
    """Return (one-line WHY, one-line WHAT-IT-MEANS) for a failed
    criterion. Pulls specific numbers from reason via regex."""
    obs, thresh = _parse_reason_numbers(reason)

    if criterion.startswith("n_at_"):
        n = int(obs) if obs is not None else None
        n_min = int(thresh) if thresh is not None else 30
        return (
            f"Only {n} events were observed; the gate requires at "
            f"least {n_min}.",
            "Statistical tests are unreliable below that sample "
            "floor — verdict is unenforceable on this little data.",
        )

    if criterion.startswith("baseline_t_at_"):
        return (
            f"Baseline t-statistic is {obs:+.2f}, "
            f"below the threshold of {thresh:+.2f} (95% confidence).",
            "Cannot reject the null hypothesis 'this strategy has "
            "no edge' — mean return is indistinguishable from random.",
        )

    if criterion.startswith("jensen_alpha_t_BTC_at_"):
        return (
            f"Jensen α vs BTC has t-stat {obs:+.2f}, below the "
            f"threshold of {thresh:+.2f}.",
            "After OLS-adjusting for the strategy's BTC beta, there "
            "is no statistically significant residual alpha.",
        )

    if criterion.startswith("jensen_alpha_t_ETH_at_"):
        return (
            f"Jensen α vs ETH has t-stat {obs:+.2f}, below the "
            f"threshold of {thresh:+.2f}.",
            "After OLS-adjusting for the strategy's ETH beta, there "
            "is no statistically significant residual alpha.",
        )

    if criterion.startswith("sign_stability"):
        return (
            "Direction of returns flips across the 1d / 3d / 7d "
            "horizons.",
            "Pattern is horizon-specific (e.g. wins at 3d, loses at "
            "1d and 7d) — does not generalize.",
        )

    if criterion.startswith("loo_fragile_"):
        return (
            "Leave-one-out removed a single entity (whale / coin / "
            "cohort) and the t-stat changed materially.",
            "The signal depends heavily on one contributor — "
            "fragile, not robust to entity dropout.",
        )

    if criterion.startswith("outlier_fragile_"):
        return (
            "A single event accounts for > 30% of the sum of |return|.",
            "Strategy's apparent edge is concentrated in one outlier "
            "trade, not a general pattern.",
        )

    if criterion.startswith("dsr_passes_gate"):
        return (
            "Deflated Sharpe Ratio (Bailey & López de Prado) "
            "failed to clear the multi-test corrected floor.",
            "After accounting for all prior trials searched, this "
            "Sharpe is statistically indistinguishable from a lucky "
            "outcome under the null.",
        )

    # Generic fallback
    return (
        f"`{criterion}` was not satisfied (observed "
        f"{obs if obs is not None else '?'} vs threshold "
        f"{thresh if thresh is not None else '?'}).",
        "Refer to the per-criterion table above for full context.",
    )


def _scan_per_key_skew(
    jensen_alpha: dict[str, Any] | None, n_events: int,
) -> str | None:
    """If a single attribution key carries > 50% of total events, return
    a one-sentence callout. Else None."""
    if not isinstance(jensen_alpha, dict):
        return None
    per_key = jensen_alpha.get("_per_key")
    if not isinstance(per_key, dict) or not per_key or n_events <= 0:
        return None
    # Per-key OLS dicts always have at least one horizon entry; the
    # n is shared across horizons (same events), so any horizon works.
    max_key, max_n = None, 0
    for k, bench_block in per_key.items():
        if not isinstance(bench_block, dict):
            continue
        for bench, per_h in bench_block.items():
            if not isinstance(per_h, dict):
                continue
            for h_key, ols in per_h.items():
                if isinstance(ols, dict) and "n" in ols:
                    try:
                        n = int(ols["n"])
                    except (TypeError, ValueError):
                        continue
                    if n > max_n:
                        max_n, max_key = n, k
                    break
            break
    if not max_key or max_n <= 0:
        return None
    share = max_n / n_events * 100
    if share > 50:
        short = max_key if len(max_key) <= 14 else f"…{max_key[-12:]}"
        return (
            f"Note: one entity ({short}) carries {share:.0f}% of all "
            f"events — the cohort effectively is that entity."
        )
    return None


def _scan_regime_concentration(
    jensen_regime: dict[str, Any] | None,
) -> str | None:
    """If events concentrate in 1-2 regime buckets, flag it.
    Particularly: if only Q1 (deep_bear) or only Q5 (strong_bull),
    the strategy is regime-conditional."""
    if not isinstance(jensen_regime, dict) or "skipped" in jensen_regime:
        return None
    populated = []
    for q_key in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        bucket = jensen_regime.get(q_key, {})
        if not isinstance(bucket, dict):
            continue
        n = bucket.get("n_total", 0)
        if n and n >= 30:
            populated.append((q_key, bucket.get("label", q_key), n))
    if not populated:
        return None
    if len(populated) == 1:
        q, label, n = populated[0]
        return (
            f"Note: all {n:,} events fell in the **{label}** regime "
            f"bucket ({q}); strategy is regime-conditional — α only "
            f"exists in this slice of the market."
        )
    return None


def _explain_dsr(report: Any) -> str | None:
    """If DSR is the failing gate, surface the exact value."""
    dsr = getattr(report, "dsr_per_horizon", {}) or {}
    best = None
    for h, d in dsr.items():
        if not isinstance(d, dict) or d.get("untestable"):
            continue
        val = d.get("dsr_sharpe_units")
        if val is None:
            continue
        if best is None or val > best[1]:
            best = (h, val)
    if not best:
        return None
    h, val = best
    return (
        f"Best DSR across horizons: {val:+.3f} at {h} (a positive value "
        f"means Sharpe exceeds the multi-test floor; this strategy did "
        f"NOT clear it)."
    )


# ───────────────────────────────────────────────────────────────────────
# Sprint 22 — user-facing tier + machine-generated next-action.
#
# Both are a PRESENTATION layer over what the engine already computed in
# Sprint 16's failure analysis. They invent no new statistic (HARD STOP:
# "Tier logic requires a statistic the engine doesn't already compute").
# ───────────────────────────────────────────────────────────────────────

USER_TIERS = ("KILL_CONFIRMED", "INCONCLUSIVE", "ALIVE_FOR_NOW")

# A KILL is "confirmed" when at least one STRUCTURAL gate fired — those
# say "the edge is not in the data". A KILL whose only failures are
# fragility / sample-size gates says "this sample can't be trusted",
# which is closer to INCONCLUSIVE than a decisive kill.
_STRUCTURAL_GATE_PREFIXES = (
    "baseline_t_at_",      # mean return indistinguishable from 0
    "jensen_alpha_t_",     # no residual α after β-adjustment
    "vs_BTC_t_at_",        # legacy β=1 excess gates — still structural
    "vs_ETH_t_at_",
    "sign_stability",      # direction flips across horizons
    "dsr_passes_gate",     # Sharpe below multi-test floor
)
# These say "can't trust THIS sample", not "no edge" → not decisive.
_FRAGILITY_GATE_PREFIXES = ("loo_fragile_", "outlier_fragile_", "n_at_")


def _failing_gates(report: Any) -> list[dict]:
    """Failing pre-commit gates, matching the engine's kill definition
    (_verdict_for treats FAIL and UNPARSED as kills)."""
    pre_commit = getattr(report, "pre_commit_results", []) or []
    return [r for r in pre_commit
            if r.get("verdict") in ("FAIL", "UNPARSED")]


def map_user_tier(report: Any) -> str:
    """Coarse 3-tier USER-facing level over the engine's EXISTING verdict
    + gate results (Sprint 22, D1). Pure function — no new statistics.

    Returns one of USER_TIERS:
      KILL_CONFIRMED — a structural gate fired decisively; dead in the data.
      INCONCLUSIVE   — underpowered / too little data / only fragility gates.
      ALIVE_FOR_NOW  — nothing fired to kill it (NOT an endorsement).
    """
    verdict = getattr(report, "verdict", "?")

    # SURVIVOR = "not yet killed", explicitly NOT proof it works.
    if verdict == "SURVIVOR":
        return "ALIVE_FOR_NOW"

    # Zero / sub-floor data — can't judge.
    if verdict in ("UNDERPOWERED", "WAITING-ON-DATA"):
        return "INCONCLUSIVE"

    if verdict == "KILL":
        # Invariant: a KILL has max_n >= 30 (else _verdict_for returns
        # UNDERPOWERED), so D1's "n>=30" qualifier is already satisfied.
        structural = [
            r for r in _failing_gates(report)
            if any(r.get("criterion", "").startswith(p)
                   for p in _STRUCTURAL_GATE_PREFIXES)
        ]
        return "KILL_CONFIRMED" if structural else "INCONCLUSIVE"

    # Unknown verdict string — never fake confidence.
    return "INCONCLUSIVE"


def generate_next_action(report: Any) -> str:
    """Machine-generated "what I'd do next" string (Sprint 22, D2).

    Rule-based, ≤ 2 sentences, every clause tied to an ACTUAL engine
    signal — never generic "try harder" advice (HARD STOP). Reuses the
    same primary-failure + per-key-skew analysis as generate_explanation.
    """
    verdict = getattr(report, "verdict", "?")
    n_events = int(getattr(report, "n_events", 0))

    # 1. No data / sub-floor data.
    if verdict == "WAITING-ON-DATA" or n_events == 0:
        return (
            "Signal emitted zero events in the test window — the trigger "
            "never fired or an upstream stream is stale. Verify the data "
            "source is fresh (list_available_data), or broaden the trigger."
        )
    if verdict == "UNDERPOWERED":
        return (
            f"Signal fires too rarely — only {n_events} events, under the "
            f"30-event floor. Move to a higher-frequency timeframe, broaden "
            f"the trigger, or pool correlated assets to reach the floor."
        )

    # 2. SURVIVOR — the live risk is in-sample / PIT, not a fired gate.
    if verdict == "SURVIVOR":
        if getattr(report, "verdict_tentative", False):
            return (
                "No gate fired to kill it, but selection ran on the same "
                "window being tested — that's 'not dead yet', not proof. "
                "Confirm on a strict post-locked-at OOS window before trusting it."
            )
        return (
            "No gate fired to kill it, but that is not proof it works. "
            "Confirm on out-of-sample data before deploying real capital."
        )

    # 3. KILL (or unknown-but-failed). Pick the single most actionable
    #    lever, keyed to a real signal.
    failures = _failing_gates(report)
    jensen_alpha = getattr(report, "jensen_alpha", {}) or {}
    skew = _scan_per_key_skew(jensen_alpha, n_events)

    # 3a. One entity dominating is the single most concrete lever — its
    #     removal can flip every other gate, so it leads when present.
    if skew:
        return (
            "One entity drives most of the events — the result is about "
            "that entity, not a general rule. Re-test with it excluded, or "
            "treat it as a single-name bet rather than a strategy."
        )

    crit = failures[0].get("criterion", "") if failures else ""

    if crit.startswith("sign_stability"):
        return (
            "Direction flips across horizons — likely horizon-specific "
            "fitting. Test whether the sign holds on a held-out window "
            "before trusting any single horizon."
        )
    if crit.startswith("dsr_passes_gate"):
        return (
            "Sharpe doesn't clear the multi-test floor for the number of "
            "strategies already tried. Re-test only with a strong prior — "
            "every retest raises the bar."
        )
    if (crit.startswith("jensen_alpha_t_")
            or crit.startswith("vs_BTC_t_at_")
            or crit.startswith("vs_ETH_t_at_")):
        return (
            "Apparent returns are explained by passive benchmark exposure — "
            "no residual α after β-adjustment. Drop it, or find a variant "
            "whose edge survives subtracting a position-sized buy-and-hold."
        )
    if crit.startswith("baseline_t_at_"):
        return (
            "Mean return is statistically indistinguishable from zero. "
            "Extend the backtest window or add a regime filter — or accept "
            "there is no edge here."
        )
    if crit.startswith("n_at_"):
        return (
            "Too few events at the tested horizon to judge. Broaden the "
            "trigger or extend the window to clear the 30-event floor."
        )
    if crit.startswith("loo_fragile_") or crit.startswith("outlier_fragile_"):
        return (
            "The result hangs on a single entity or one outlier trade, not "
            "a general pattern. Re-test with that contributor removed before "
            "trusting the signal."
        )

    # Defensive fallback — still names the specific gate (never generic).
    if crit:
        return (
            f"Failed the `{crit}` gate (see the per-criterion table for the "
            f"exact value). Address that specific gate or treat the strategy "
            f"as dead in this data."
        )
    return (
        "Verdict is KILL but no specific gate is marked FAIL — usually the "
        "min-events floor was hit first. Broaden the trigger or extend the "
        "window so the per-gate tests can run."
    )


def generate_explanation(report: Any) -> str:
    """Render a Plain-English Explanation markdown section.

    Returns a multi-line string starting with the section header. Caller
    embeds in the verdict markdown template.
    """
    verdict = getattr(report, "verdict", "?")
    n_events = int(getattr(report, "n_events", 0))
    pre_commit = getattr(report, "pre_commit_results", []) or []
    jensen_alpha = getattr(report, "jensen_alpha", {}) or {}
    jensen_regime = getattr(report, "jensen_alpha_by_regime", {}) or {}

    if verdict in ("SURVIVOR",):
        # Strongest pass: find the criterion with the largest margin
        passes = [r for r in pre_commit if r.get("verdict") == "PASS"]
        strongest_line = "All eight pre-commit gates passed."
        if passes:
            # Use n_events as a rough "strongest signal" proxy
            strongest_line = (
                f"All {len(passes)} pre-commit gates passed across "
                f"{n_events:,} events."
            )
        lines = [
            "## Plain-English explanation (Sprint 16)",
            "",
            "**Verdict**: this strategy passed all validation gates.",
            "",
            strongest_line,
        ]
        skew = _scan_per_key_skew(jensen_alpha, n_events)
        if skew:
            lines.append("")
            lines.append(skew)
        regime = _scan_regime_concentration(jensen_regime)
        if regime:
            lines.append("")
            lines.append(regime)
        lines.append("")
        lines.append(
            "**What this means**: recommend OOS validation on a strict "
            "post-locked-at window before deploying with real capital. "
            "In-sample passes are necessary but not sufficient."
        )
        return "\n".join(lines)

    if verdict == "UNDERPOWERED":
        floor = 30
        lines = [
            "## Plain-English explanation (Sprint 16)",
            "",
            "**Verdict pending**: insufficient data.",
            "",
            f"Only {n_events:,} events were observed; the engine needs "
            f"at least {floor} to compute reliable statistics.",
            "",
            "**To complete validation**: either broaden the trigger "
            "(more permissive thresholds), extend the backtest window, "
            "or wait for more live data to accumulate.",
        ]
        return "\n".join(lines)

    if verdict == "WAITING-ON-DATA":
        lines = [
            "## Plain-English explanation (Sprint 16)",
            "",
            "**Verdict pending**: zero events were emitted.",
            "",
            "The signal function returned no events for this spec. "
            "Either an upstream data source is missing/stale, or the "
            "trigger conditions never fired in the test window.",
            "",
            "**Next step**: inspect signal_notes (above) to see what "
            "the signal reported; verify the relevant data stream is "
            "fresh via list_available_data.",
        ]
        return "\n".join(lines)

    # KILL — most common case; needs careful gate-by-gate breakdown
    failures = [r for r in pre_commit if r.get("verdict") == "FAIL"]
    if not failures:
        # KILL but no FAIL recorded (edge case — e.g. min_n_for_test)
        return (
            "## Plain-English explanation (Sprint 16)\n\n"
            "**Verdict**: KILL.\n\n"
            "Verdict was assigned but no specific pre-commit criterion "
            "is marked FAIL — this usually means n_events < min_n_for_"
            "test floor was hit before per-gate evaluation. See sample "
            "size / signal_notes above."
        )

    # Primary failure = first failing gate (criteria are evaluated
    # in spec order; the first one to fail is typically the canonical
    # reason).
    primary = failures[0]
    why, what = _explain_failed_gate(
        primary.get("criterion", ""), primary.get("reason", ""),
    )

    lines = [
        "## Plain-English explanation (Sprint 16)",
        "",
        "**Verdict**: this strategy was rejected.",
        "",
        f"**Primary reason** ({_criterion_label(primary.get('criterion', ''))}): "
        f"{why} {what}",
    ]

    # Secondary failures (after the first)
    if len(failures) > 1:
        sec_phrases = []
        for f in failures[1:]:
            why_s, _what_s = _explain_failed_gate(
                f.get("criterion", ""), f.get("reason", ""),
            )
            sec_phrases.append(
                f"{_criterion_label(f.get('criterion', ''))} also failed: "
                f"{why_s.lower().rstrip('.')}"
            )
        if sec_phrases:
            lines.append("")
            lines.append("**Secondary signals**: " + "; ".join(sec_phrases) + ".")

    # Per-key concentration callout
    skew = _scan_per_key_skew(jensen_alpha, n_events)
    if skew:
        lines.append("")
        lines.append(skew)

    # Per-regime concentration callout
    regime = _scan_regime_concentration(jensen_regime)
    if regime:
        lines.append("")
        lines.append(regime)

    # Closing actionable line
    lines.append("")
    if primary.get("criterion", "").startswith("dsr_passes"):
        lines.append(
            "**What this means**: this strategy is statistically "
            "indistinguishable from a lucky outcome given the number of "
            "prior strategies tested. To clear DSR, you'd need a higher "
            "Sharpe OR fewer prior trials in the counter."
        )
    elif primary.get("criterion", "").startswith("n_at_"):
        lines.append(
            "**What this means**: not enough data to test rigorously. "
            "Either broaden the trigger or extend the backtest window."
        )
    elif primary.get("criterion", "").startswith("sign_stability"):
        lines.append(
            "**What this means**: the strategy works at one horizon but "
            "not adjacent ones — typically a sign of horizon-specific "
            "fitting. Real edges generally survive across nearby horizons."
        )
    elif primary.get("criterion", "").startswith("jensen_alpha"):
        lines.append(
            "**What this means**: the strategy's apparent returns can be "
            "explained by passive exposure to the benchmark. Once you "
            "subtract a position-sized buy-and-hold of that benchmark, "
            "no edge remains."
        )
    else:
        lines.append(
            "**What this means**: if deployed live as-is, expect "
            "performance to be no better than the gate suggests. See "
            "the per-criterion table above for specifics."
        )

    return "\n".join(lines)


# One-line gloss per tier — the "what it means for YOU" framing from D1.
_TIER_GLOSS = {
    "KILL_CONFIRMED": (
        "This strategy is dead in the data. Move on."
    ),
    "INCONCLUSIVE": (
        "Not enough data to judge. Come back when you have more events "
        "or cleaner data."
    ),
    "ALIVE_FOR_NOW": (
        "No evidence to kill it — but that's not proof it works. It's "
        "\"not dead yet\", pending OOS confirmation."
    ),
}


def generate_operator_section(report: Any) -> str:
    """Sprint 22 — render the "For the operator" markdown block that pairs
    the coarse user tier with the machine-generated next-action. Embedded
    in the verdict template right after the Sprint 16 plain-English block.
    """
    tier = map_user_tier(report)
    action = generate_next_action(report)
    return "\n".join([
        "## For the operator (Sprint 22)",
        "",
        f"**Level**: `{tier}` — {_TIER_GLOSS.get(tier, '')}",
        "",
        f"**What I'd do next**: {action}",
    ])
