# Deflated Sharpe Ratio (the multiple-testing correction)

A Sharpe ratio computed after searching many strategies is biased upward:
the best of N random strategies looks good by luck alone. The Deflated
Sharpe Ratio (Bailey & López de Prado, 2014) corrects for this by comparing
the observed Sharpe against the Sharpe you would *expect from the best of N
independent random trials under the null*.

## The math (`dsr.py`)

Expected maximum Sharpe under H₀ (true Sharpe = 0), in standard-normal
units, via the Mertens approximation:

```
E[SR_max | N] = (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))
```

where γ is the Euler–Mascheroni constant and N is the cumulative trial
count. `E[SR_max]` is **monotonically increasing in N** — the more
strategies you've tried, the higher the bar the next one must clear.

The deflated Sharpe subtracts a per-observation-scaled floor:

```
DSR = sr_observed − E[SR_max] / √(n_obs − 1)
gate passes ⇔ DSR > 0
```

It also reports a probability (the CDF of the deflated, skew/kurtosis-
adjusted z-statistic).

## The trial count is the lever

Because the floor rises with N, **what you count as a "trial" matters**.
Counting unrelated strategy families into one N over-penalizes; counting
nothing under-corrects. Two principles in this engine:

1. **N is the search you actually did** — a running count of hypotheses
   tested, not a single hypothesis's re-runs.

2. **The count is per-user, not global.** In a multi-tenant setting, one
   user's trial-burning must not raise another user's floor. The counter is
   keyed by caller identity; a brand-new caller starts at a low floor
   because they have not yet burned any multiple-testing budget.

## Why it's the binding gate so often

For a genuinely marginal strategy, the DSR floor is what separates "looked
good in one backtest" from "survives the fact that you tried many things."
A strategy with a real Sharpe of ~2 can still fail DSR if the prior search
was large — and that is the correct, humbling answer.
