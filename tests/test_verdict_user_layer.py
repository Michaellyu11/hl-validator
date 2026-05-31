"""Sprint 22: tests for the user-facing tier + machine next-action.

Two pure functions over the EXISTING ValidationReport fields:
  - map_user_tier(report) -> KILL_CONFIRMED | INCONCLUSIVE | ALIVE_FOR_NOW
  - generate_next_action(report) -> str  (rule-based, signal-tied)

Plus the operator-section renderer that pairs them. No new statistics;
this is a presentation layer (D1/D2). Backward compat (D4): the Sprint 16
plain-English explanation is unchanged — covered by test_explanation.py,
re-asserted here for the shared report shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quant_lab.validation.explanation import (
    USER_TIERS,
    generate_explanation,
    generate_next_action,
    generate_operator_section,
    map_user_tier,
)


@dataclass
class _FakeReport:
    """Same minimal ValidationReport stand-in as test_explanation.py."""
    verdict: str = "KILL"
    n_events: int = 0
    verdict_tentative: bool = False
    pre_commit_results: list[dict] = field(default_factory=list)
    jensen_alpha: dict[str, Any] = field(default_factory=dict)
    jensen_alpha_by_regime: dict[str, Any] = field(default_factory=dict)
    dsr_per_horizon: dict[str, Any] = field(default_factory=dict)


def _gate(criterion: str, verdict: str = "FAIL", reason: str = "") -> dict:
    return {"criterion": criterion, "verdict": verdict, "reason": reason}


# ── tier mapping: one decisive case per tier (D1) ─────────────────────


def test_tier_kill_confirmed_on_structural_gate() -> None:
    """A structural gate (sign-flip) firing at n>=30 → KILL_CONFIRMED."""
    report = _FakeReport(
        verdict="KILL", n_events=1348,
        pre_commit_results=[
            _gate("n_at_3h >= 30", "PASS"),
            _gate("baseline_t_at_3h > 2.0", "PASS"),
            _gate("sign_stability_positive == true", "FAIL",
                  "sign_stability_positive all_positive=False"),
        ],
    )
    assert map_user_tier(report) == "KILL_CONFIRMED"


def test_tier_inconclusive_on_underpowered() -> None:
    assert map_user_tier(_FakeReport(verdict="UNDERPOWERED", n_events=12)) \
        == "INCONCLUSIVE"


def test_tier_inconclusive_on_waiting_on_data() -> None:
    assert map_user_tier(_FakeReport(verdict="WAITING-ON-DATA", n_events=0)) \
        == "INCONCLUSIVE"


def test_tier_inconclusive_when_kill_is_only_fragility() -> None:
    """A KILL whose ONLY failing gates are fragility/sample (not structural)
    is 'can't trust this sample', not a decisive kill → INCONCLUSIVE."""
    report = _FakeReport(
        verdict="KILL", n_events=80,
        pre_commit_results=[
            _gate("loo_fragile_3h == false", "FAIL", "loo fragile"),
            _gate("outlier_fragile_3h == false", "FAIL", "outlier dominates"),
        ],
    )
    assert map_user_tier(report) == "INCONCLUSIVE"


def test_tier_alive_for_now_on_survivor() -> None:
    """SURVIVOR is 'not yet killed', never an endorsement → ALIVE_FOR_NOW."""
    report = _FakeReport(
        verdict="SURVIVOR", n_events=500,
        pre_commit_results=[_gate("n_at_3h >= 30", "PASS")],
    )
    assert map_user_tier(report) == "ALIVE_FOR_NOW"


def test_tier_always_in_enum() -> None:
    for v in ("KILL", "SURVIVOR", "UNDERPOWERED", "WAITING-ON-DATA", "???"):
        assert map_user_tier(_FakeReport(verdict=v)) in USER_TIERS


# ── next-action: one per branch, each tied to a real signal (D2) ──────


def test_next_action_underpowered_cites_event_count() -> None:
    action = generate_next_action(_FakeReport(verdict="UNDERPOWERED",
                                              n_events=13))
    assert "13" in action
    assert "30-event floor" in action
    # higher-frequency / pool — the dispatch's example levers
    assert "higher-frequency" in action or "pool" in action


def test_next_action_sign_flip_cites_horizon_fitting() -> None:
    report = _FakeReport(
        verdict="KILL", n_events=1348,
        pre_commit_results=[
            _gate("sign_stability_positive == true", "FAIL",
                  "sign_stability_positive all_positive=False"),
        ],
    )
    action = generate_next_action(report)
    assert "flips across horizons" in action
    assert "held-out" in action


def test_next_action_per_key_skew_recommends_exclusion() -> None:
    """One entity > 50% of events → 'single-name bet' framing, and it
    LEADS over the gate-specific message because removing it can flip
    every other gate."""
    report = _FakeReport(
        verdict="KILL", n_events=1000,
        pre_commit_results=[
            _gate("jensen_alpha_t_ETH_at_3h > 1.5", "FAIL",
                  "jensen α vs ETH 3h t_hc0=-11.9 > 1.5: False"),
        ],
        jensen_alpha={
            "_per_key": {
                "0xdominant": {"ETH": {"3h": {"n": 700}}},
                "0xminor": {"ETH": {"3h": {"n": 300}}},
            },
        },
    )
    action = generate_next_action(report)
    assert "One entity drives" in action
    assert "single-name bet" in action


def test_next_action_survivor_flags_oos_not_endorsement() -> None:
    report = _FakeReport(verdict="SURVIVOR", n_events=500,
                         verdict_tentative=True)
    action = generate_next_action(report)
    assert "not proof" in action or "not dead yet" in action
    assert "OOS" in action or "out-of-sample" in action or "post-locked-at" in action


def test_next_action_jensen_cites_benchmark_exposure() -> None:
    report = _FakeReport(
        verdict="KILL", n_events=300,
        pre_commit_results=[
            _gate("jensen_alpha_t_BTC_at_24h > 1.5", "FAIL",
                  "jensen α vs BTC 24h t_hc0=+0.5 > 1.5: False"),
        ],
    )
    action = generate_next_action(report)
    assert "benchmark exposure" in action
    assert "buy-and-hold" in action


def test_next_action_never_empty_and_short() -> None:
    """Every branch returns a non-empty ≤2-sentence string (D2 hard stop:
    no generic advice → always tied, always present)."""
    cases = [
        _FakeReport(verdict="UNDERPOWERED", n_events=5),
        _FakeReport(verdict="WAITING-ON-DATA", n_events=0),
        _FakeReport(verdict="SURVIVOR", n_events=400),
        _FakeReport(verdict="KILL", n_events=100, pre_commit_results=[
            _gate("dsr_passes_gate == true", "FAIL",
                  "dsr_passes_gate=False; required == true")]),
        _FakeReport(verdict="KILL", n_events=100, pre_commit_results=[
            _gate("baseline_t_at_3h > 2.0", "FAIL",
                  "|t|=0.3 at 3h > 2.0: False")]),
    ]
    for r in cases:
        a = generate_next_action(r)
        assert a, f"empty action for {r.verdict}"
        # ≤ 2 sentences (allow a trailing period)
        assert a.count(". ") <= 2, f"too long ({r.verdict}): {a}"


# ── operator section render ───────────────────────────────────────────


def test_operator_section_pairs_tier_and_action() -> None:
    report = _FakeReport(
        verdict="KILL", n_events=1348,
        pre_commit_results=[
            _gate("sign_stability_positive == true", "FAIL",
                  "sign_stability_positive all_positive=False"),
        ],
    )
    md = generate_operator_section(report)
    assert "## For the operator" in md
    assert "`KILL_CONFIRMED`" in md
    assert "dead in the data" in md
    assert "What I'd do next" in md
    assert "flips across horizons" in md


# ── backward compat (D4): Sprint 16 explanation unchanged ─────────────


def test_plain_english_block_still_renders() -> None:
    """The Sprint 16 block is independent of the Sprint 22 additions."""
    report = _FakeReport(
        verdict="KILL", n_events=1348,
        pre_commit_results=[
            _gate("sign_stability_positive == true", "FAIL",
                  "sign_stability_positive all_positive=False"),
        ],
    )
    text = generate_explanation(report)
    assert "Plain-English explanation" in text
    assert "rejected" in text.lower()
    # The Sprint 22 helpers do NOT leak into the Sprint 16 block.
    assert "For the operator" not in text
