"""Deflated Sharpe Ratio — refactored from scripts/dsr_pbo_retro/dsr_pbo.py.

Kept the two functions the engine needs (`expected_max_sharpe`,
`deflated_sharpe`) and the `DSRResult` dataclass. PBO/CSCV is left in the
original module — the engine doesn't need it today.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

EULER_MASCHERONI = 0.5772156649015328606065120900824024310421


def expected_max_sharpe(n_trials: int) -> float:
    """E[SR_max | N trials, true SR = 0], standard-normal units.
    Mertens approximation per Bailey & López de Prado 2014."""
    if n_trials <= 1:
        return 0.0
    g = EULER_MASCHERONI
    inv_phi_1 = stats.norm.ppf(1 - 1.0 / n_trials)
    inv_phi_2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return (1 - g) * inv_phi_1 + g * inv_phi_2


@dataclass
class DSRResult:
    sr_observed: float
    n_obs: int
    n_trials: int
    skew: float
    kurt: float
    expected_max_sr: float
    dsr_sharpe_units: float
    dsr_probability: float
    passes_gate: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "sr_observed": self.sr_observed,
            "n_obs": self.n_obs,
            "n_trials": self.n_trials,
            "skew": self.skew,
            "kurt": self.kurt,
            "expected_max_sr": self.expected_max_sr,
            "dsr_sharpe_units": self.dsr_sharpe_units,
            "dsr_probability": self.dsr_probability,
            "passes_gate": self.passes_gate,
            "note": self.note,
        }


def deflated_sharpe(
    sr_observed: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    note: str = "",
) -> DSRResult:
    """Bailey & López de Prado 2014 Deflated Sharpe.

    `sr_observed` is per-period Sharpe (mean/std of returns); n_obs is the
    number of return observations; n_trials is the cumulative trial counter
    (the binding multi-test correction).
    """
    if n_obs < 2:
        return DSRResult(
            sr_observed=sr_observed, n_obs=n_obs, n_trials=n_trials,
            skew=skew, kurt=kurt,
            expected_max_sr=float("nan"),
            dsr_sharpe_units=float("nan"),
            dsr_probability=float("nan"),
            passes_gate=False,
            note=note + " | DSR-untestable: n_obs < 2",
        )
    e_max_std = expected_max_sharpe(n_trials)
    se_sr = 1.0 / math.sqrt(n_obs - 1)
    e_max_sr = e_max_std * se_sr
    dsr_sharpe = sr_observed - e_max_sr
    sigma_sr = math.sqrt(
        max(1e-12,
            1.0 - skew * sr_observed + (kurt - 1.0) / 4.0 * sr_observed**2)
    )
    z = (sr_observed - e_max_sr) * math.sqrt(n_obs - 1) / sigma_sr
    dsr_prob = float(stats.norm.cdf(z))
    return DSRResult(
        sr_observed=sr_observed, n_obs=n_obs, n_trials=n_trials,
        skew=skew, kurt=kurt,
        expected_max_sr=e_max_sr,
        dsr_sharpe_units=dsr_sharpe,
        dsr_probability=dsr_prob,
        passes_gate=bool(dsr_sharpe > 0.0),
        note=note,
    )
