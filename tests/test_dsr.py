"""Tests for the Deflated Sharpe Ratio (public engine excerpt).

Covers the public surface of quant_lab/validation/dsr.py:
  - expected_max_sharpe(n_trials): monotone increasing, 0 at the n<=1 floor
  - deflated_sharpe(...): the floor subtraction + gate logic
This test imports ONLY the public dsr module (scipy is the one external dep).
"""
from __future__ import annotations

import math

from quant_lab.validation.dsr import (
    DSRResult,
    deflated_sharpe,
    expected_max_sharpe,
)


# ── expected_max_sharpe: the multiple-testing floor ───────────────────


def test_expected_max_sharpe_zero_at_floor():
    # n<=1 means "no multiple testing" → no inflation to correct.
    assert expected_max_sharpe(0) == 0.0
    assert expected_max_sharpe(1) == 0.0


def test_expected_max_sharpe_monotone_increasing():
    # The whole point: more trials searched ⇒ a higher bar.
    xs = [2, 10, 50, 100, 376, 1000]
    vals = [expected_max_sharpe(n) for n in xs]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    # Sanity anchors (std-normal units), loose bounds.
    assert 1.5 < expected_max_sharpe(50) < 2.6
    assert 2.5 < expected_max_sharpe(376) < 3.3


# ── deflated_sharpe: floor subtraction + gate ─────────────────────────


def test_deflated_sharpe_untestable_below_two_obs():
    r = deflated_sharpe(sr_observed=1.0, n_obs=1, n_trials=10)
    assert isinstance(r, DSRResult)
    assert r.passes_gate is False
    assert math.isnan(r.dsr_sharpe_units)


def test_deflated_sharpe_strong_signal_passes():
    # A large Sharpe over many obs at a modest trial count should clear 0.
    r = deflated_sharpe(sr_observed=0.5, n_obs=500, n_trials=20)
    assert r.dsr_sharpe_units > 0
    assert r.passes_gate is True
    # gate is exactly dsr_sharpe > 0
    assert r.passes_gate == (r.dsr_sharpe_units > 0)


def test_deflated_sharpe_floor_rises_with_trials():
    # Same observed Sharpe, more trials ⇒ lower (or equal) deflated Sharpe.
    lo = deflated_sharpe(sr_observed=0.3, n_obs=300, n_trials=10)
    hi = deflated_sharpe(sr_observed=0.3, n_obs=300, n_trials=5000)
    assert hi.expected_max_sr > lo.expected_max_sr
    assert hi.dsr_sharpe_units < lo.dsr_sharpe_units


def test_deflated_sharpe_negative_signal_fails():
    r = deflated_sharpe(sr_observed=-0.1, n_obs=300, n_trials=10)
    assert r.dsr_sharpe_units < 0
    assert r.passes_gate is False


def test_dsr_result_to_dict_roundtrip():
    r = deflated_sharpe(sr_observed=0.4, n_obs=200, n_trials=50)
    d = r.to_dict()
    assert d["n_trials"] == 50
    assert d["n_obs"] == 200
    assert d["passes_gate"] == r.passes_gate
    assert "dsr_sharpe_units" in d and "expected_max_sr" in d
