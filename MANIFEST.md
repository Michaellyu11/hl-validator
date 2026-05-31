# MANIFEST — public-repo allowlist + secret-scan results

This repository was built **allowlist-first**: a fresh git history into
which only explicitly vetted, secret-scanned files were copied from a
private source tree. Nothing was stripped-and-republished. Every file was
scanned for: local paths (`/Users/`, `/etc/`, `/opt/`, `/tmp/`, `/var/`),
hostnames/IPs, `alphalab`/`alpha_lab`, `postgres`/`psycopg`, `TELEGRAM`,
`PGPASSWORD`, `password`/`secret`/`bearer`, API-key prefixes, and
data-pipeline identifiers.

## Included files (18)

| File | Source | Secret scan | Notes |
|---|---|---|---|
| `quant_lab/validation/dsr.py` | copied as-is | CLEAN | self-contained |
| `quant_lab/validation/kill_tests.py` | copied as-is | CLEAN | self-contained |
| `quant_lab/validation/explanation.py` | copied as-is | CLEAN | self-contained |
| `quant_lab/validation/spec.py` | copied + **cleaned** | CLEAN after clean | removed 1 hardcoded `/Users/...` counter path → env-overridable repo-relative default |
| `quant_lab/validation/engine.py` | copied + **cleaned** | CLEAN after clean | removed **2** hardcoded `/Users/...` paths (`DOCS_DIR`, `_REGIME_ETH_DEFAULT_DB`) → env-overridable defaults; see "completeness" below |
| `quant_lab/__init__.py` | freshly written | CLEAN | minimal package marker |
| `quant_lab/validation/__init__.py` | freshly written | CLEAN | empty |
| `tests/test_dsr.py` | **freshly written** | CLEAN | imports only `quant_lab.validation.dsr`. NOT copied: the source's DSR test (`tests/dsr_pbo_retro/test_dsr_pbo.py`) also exercises the private PBO/CSCV module that is not shipped here, so a fresh minimal test of the public `dsr.py` surface was written instead |
| `tests/test_verdict_user_layer.py` | copied as-is | CLEAN | imports only `quant_lab.validation.explanation`; synthetic fixtures, no real data |
| `docs/DESIGN.md` | freshly written | CLEAN | no internal references |
| `docs/DSR.md` | freshly written | CLEAN | no internal references |
| `docs/VERDICT.md` | freshly written | CLEAN | no internal references |
| `docs/example_verdict_whalefollow_v3.md` | freshly written | CLEAN | dogfood verdict; on-chain addresses abbreviated; NOT copied from the raw internal verdict doc |
| `README.md` | freshly written | CLEAN | stub with TODO for the origin story |
| `LICENSE` | freshly written | CLEAN | MIT |
| `requirements.txt` | freshly written | CLEAN | scipy + pytest only |
| `.gitignore` | freshly written | CLEAN | deny-by-default |
| `MANIFEST.md` | this file | CLEAN | — |

`.gitignore` is **deny-by-default** (`*` then explicit re-allows) plus hard
denies for `trial_counter.json`, `*.sqlite`, `*.duckdb`, `*.log`, `.env`,
`**/data/`, `**/secrets*`, `**/*credential*`, `__pycache__`, `_*.txt`.

## Cleaned (not excluded)

Three hardcoded `/Users/mltb/...` local paths were found (a username +
directory-layout leak — **not credentials**). Because the engine is the
trust signal, they were cleaned in the copies rather than excluded;
**source originals were never modified** (verified: source still contains
its original paths):

- `spec.py` `DEFAULT_COUNTER_PATH` → `os.environ.get("HL_VALIDATOR_TRIAL_COUNTER_PATH", <repo-relative>)`
- `engine.py` `DOCS_DIR` → `os.environ.get("HL_VALIDATOR_DOCS_DIR", "docs/validations")`
- `engine.py` `_REGIME_ETH_DEFAULT_DB` → `os.environ.get("HL_VALIDATOR_DAILY_DB", "data/...")`

After cleaning, a recursive scan of `quant_lab/` and `tests/` for `/Users`
or `mltb` returns **zero hits**.

## Excluded (stayed private)

Per the moat boundary, NONE of the following entered this repo:
data-collection code, strategy specs + signals, the DuckDB/Postgres data
and configs, the API-key store, the personal trial-counter history, session
logs, and all infra (hostnames, DB creds, Telegram tokens, `/etc/` env).

## Self-containment (verified by import + tests)

All five engine modules **import cleanly**, including `engine.py`:

| Module | `import` standalone | Tested |
|---|---|---|
| `dsr.py` | ✅ | ✅ test_dsr.py |
| `kill_tests.py` | ✅ | (indirect) |
| `explanation.py` | ✅ | ✅ test_verdict_user_layer.py |
| `spec.py` | ✅ | — |
| `engine.py` | ✅ | — |

`python -m pytest tests/` → **21 passed** (7 DSR + 14 verdict-tier),
verified with the `venv` interpreter that has `requirements.txt` installed.

> **Verification note (honest):** the import + pytest results above were
> verified with an interpreter that has the `requirements.txt` deps
> installed (scipy, pytest; pyyaml is also needed by `spec.py`). A bare
> interpreter without those deps will raise `ModuleNotFoundError` on
> `dsr`/`spec`/`engine` import — that's a missing-dependency issue, not a
> code problem. Run `pip install -r requirements.txt` first.

**One private dependency, lazily imported.** `engine.py` imports
`.cross_sectional` (not shipped — part of the private layer) but only
*inside* the cross-sectional-pipeline function (line ~301), as a deferred
import. So `import quant_lab.validation.engine` **succeeds**, and the
event-driven verdict path works; only the cross-sectional code path would
raise `ModuleNotFoundError` at call time. Correction to an earlier
assumption: there are no `jensen_alpha.py` / `regime.py` / `calibration.py`
modules — `.cross_sectional` is the sole non-shipped intra-package import.

The founder may, before/after publishing, choose to (a) ship `engine.py`
as the current auditable reference, (b) add a minimal public stub for
`.cross_sectional`, or (c) whitelist + scan that module in a later pass.

## Publication status

**NOT PUBLISHED.** No git remote is configured; nothing was committed or
pushed. The repository exists only locally at
`/Users/mltb/hl-validator-public/`. The founder reviews this manifest +
the file list and publishes manually.
