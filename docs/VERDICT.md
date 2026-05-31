# Verdict

Every run produces an engine verdict and a user-facing layer on top of it
(`explanation.py`), so that a reader who didn't write the engine knows what
the result means and what to do next.

## Engine verdicts

- **SURVIVOR** — cleared every pre-commit kill criterion. Rare by design;
  still provisional if selection was in-sample (tentative pending OOS).
- **KILL** — failed ≥1 gate. The explanation names the binding gate.
- **UNDERPOWERED** — fewer than the minimum events (default 30) to test
  reliably. Returned *before* the per-gate evaluation, independent of the
  DSR floor.
- **WAITING-ON-DATA** — the signal emitted zero events.

## The user layer

Three additive fields make the verdict readable to a non-author:

### 1. `user_level` — a coarse 3-tier read
- **KILL_CONFIRMED** — a *structural* gate fired decisively (baseline-t,
  Jensen-α, sign-stability, or DSR). The edge is not in the data.
- **INCONCLUSIVE** — underpowered, or a KILL whose only failures are
  fragility/sample gates (leave-one-out, outlier, per-horizon n). "Can't
  trust this sample yet," not "decisively dead."
- **ALIVE_FOR_NOW** — nothing fired to kill it. **Explicitly not an
  endorsement** — it is "not dead yet," pending out-of-sample confirmation.

### 2. `what_id_do_next` — a rule-based next action
≤ 2 sentences, every clause tied to an actual engine signal — never generic
advice. Examples:
- sign-flip across horizons → "Direction flips by horizon — likely
  horizon-specific fitting. Test sign-stability on a held-out window."
- one entity drives > 50% of events → "The result is about that entity, not
  a general rule. Re-test excluding it, or treat it as a single-name bet."
- underpowered → "Fires too rarely to reach the event floor. Try a higher-
  frequency timeframe, broaden the trigger, or pool correlated assets."

### 3. Plain-English explanation
A short paragraph that names the binding gate and what it means in words —
so "KILL: sign_stability_positive == false" becomes "wins at one horizon,
loses at the adjacent ones; that's horizon-specific fitting, not edge."

## Design intent

The verdict is meant to be **honest first, flattering never**. A strategy
that fails is told exactly why; a strategy that passes is told its pass is
provisional. The user layer exists so that honesty is also *legible*.
