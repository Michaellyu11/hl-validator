# Design

hl-validator turns a strategy hypothesis into a **verdict** by running it
through a fixed battery of kill-tests and a multiple-testing-corrected
significance bar. The design goal is to make a false "edge" fail loudly,
not to help a strategy pass.

## Pipeline

1. **Signal → events.** A hypothesis's signal function emits a list of
   events, each with point-in-time-safe forward returns at the configured
   horizons, benchmark-relative returns, and an attribution key (which
   coin / wallet / entity drove it).

2. **Kill-test battery** (`kill_tests.py`), per horizon:
   - **baseline** — mean forward return + t-stat.
   - **benchmark sensitivity** — return net of a position-sized benchmark.
   - **sign stability** — does the sign of the edge hold across horizons?
     (a horizon-specific flip is a classic overfitting tell.)
   - **leave-one-out** — does removing one attribution entity collapse the
     t-stat? (fragility / single-name concentration.)
   - **outlier contribution** — does one event dominate the result?

3. **Deflated Sharpe** (`dsr.py`) — the multiple-testing correction. See
   `DSR.md`.

4. **Pre-commit kill criteria** — a small, explicit grammar of gates
   (e.g. `baseline_t_at_24h > 2.0`, `sign_stability_positive == true`,
   `dsr_passes_gate == true`), committed *before* the run. The verdict is
   the conjunction: any gate fails ⇒ the hypothesis is killed.

5. **Verdict + explanation** (`engine.py`, `explanation.py`) — one of
   `SURVIVOR` / `KILL` / `UNDERPOWERED` / `WAITING-ON-DATA`, plus a
   plain-English explanation, a coarse user tier, and a machine-generated
   next action. See `VERDICT.md`.

## Why "default to KILL"

The engine assumes a hypothesis is noise until it clears every gate. The
verdicts are designed to be unflattering: a strategy that "almost passes"
is killed, and the explanation says which gate it died on and why. The
point is to be the skeptic you can't argue with — so that the rare
SURVIVOR means something.

## Point-in-time discipline

Forward returns are computed entry-at-next-bar-close, exit at +H. Selection
that runs on the same window as the test is flagged in-sample (tentative
pending out-of-sample confirmation), because in-sample selection is the
fastest way to manufacture a fake edge.
