# Example verdict — "WhaleFollow v3" (a real KILL)

A real verdict from dogfooding the engine on a copy-trading hypothesis:
*follow a set of high-performing on-chain wallets ("whales") into their
positions.* It's included because it shows the engine doing its job —
killing a hypothesis the author wanted to work, and saying exactly why.
(On-chain wallet addresses are public; shown abbreviated here.)

## The hypothesis

After two earlier versions were killed, v3 tightened whale selection to the
strictest rule yet: keep a whale only if its benchmark-adjusted (Jensen) α
vs ETH is **positive at all of the 1-, 3-, and 7-day horizons** and
significant at one. That selected **2 of 6 whales**, giving **1,348 events**.

## Verdict: 🔴 KILL

Binding gate: **`sign_stability_positive == false`**.

What v3 *passed* before it died:

| Gate | Result |
|---|---|
| `n_at_3h >= 30` | ✅ 1,348 events |
| `baseline_t_at_3h > 2.0` | ✅ t = 19.23 |
| `jensen_alpha_t_BTC_at_3h > 1.5` | ✅ t = +8.17 |
| `jensen_alpha_t_ETH_at_3h > 1.5` | ✅ t = +2.16 |
| `loo_fragile_3h == false` | ✅ |
| `outlier_fragile_3h == false` | ✅ |
| `dsr_passes_gate == true` | ✅ |
| **`sign_stability_positive == true`** | ❌ **FAIL** |

It cleared seven gates — including the multiple-testing-corrected DSR floor
— and died on the eighth.

## Why it died (plain English)

The selected whales' edge **flips sign across horizons**: positive at one
holding period, negative at adjacent ones. That is the signature of
horizon-specific fitting, not a generalizable edge. Two compounding tells:

- **In-sample selection.** The whale subset was chosen on the same window
  it was tested on. Tightening the rule until the numbers look good, on the
  same data, manufactures a horizon-specific pattern — which is exactly what
  sign-stability caught.
- **Single-entity concentration.** One wallet (`…517d1`) accounted for ~86%
  of the events, so the "cohort edge" was effectively one wallet's bet.

## The point

The author iterated this hypothesis through four versions and it never
survived. That's the engine working as intended: a strategy that *almost*
passes is still killed, and the verdict tells you which gate and why — so
the rare survivor means something.
