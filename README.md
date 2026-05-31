# hl-validator

A strict, multiple-testing-corrected validation engine for trading-strategy
hypotheses. It exists to do one thing well: **tell you when a backtested
"edge" is not real** — before you risk capital on it.

This public repository is the **engine core** (the part worth auditing
before you trust a verdict). The data-collection pipeline, the strategy
specs, the API/MCP server, and the operator's own test history are private.

## What's here

- `quant_lab/validation/` — the kill-test engine:
  - `dsr.py` — Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
  - `kill_tests.py` — baseline / benchmark / sign-stability / leave-one-out
    / outlier-contribution batteries
  - `engine.py` — orchestration: run the batteries, apply pre-committed
    kill criteria, emit a verdict
  - `explanation.py` — plain-English verdict + user tier + next action
  - `spec.py` — hypothesis spec + trial-counter persistence
- `tests/` — unit tests for the DSR math and the verdict user-layer
- `docs/` — `DESIGN.md`, `DSR.md`, `VERDICT.md`, and one real dogfood
  verdict (`example_verdict_whalefollow_v3.md`)

## The one-line philosophy

Every hypothesis is judged against a multiple-testing-corrected bar that
rises with the number of strategies already tried, and against
pre-committed kill criteria. **The default outcome is KILL.** Passing is
supposed to be hard.

## Run the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Status

> **TODO (origin story):** this README is a stub. The full origin story —
> why this engine exists, what it has killed, and the track record behind
> it — goes here. Written by the founder, not auto-generated.

## A note on completeness

The DSR, kill-test, explanation, and spec modules are self-contained and
their tests run standalone. `engine.py` imports cleanly, but one code path
(the cross-sectional pipeline) lazily imports a module that lives in the
private layer and is not shipped here — so `engine.py` is published as an
**auditable reference** of the orchestration logic, not a turn-key runnable
system. See `MANIFEST.md` for the exact public/private boundary.

## License

MIT — see `LICENSE`.
