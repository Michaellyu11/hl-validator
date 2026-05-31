"""Hypothesis spec — YAML format + locking semantics.

A spec captures the PRE-COMMITTED parameters of a hypothesis test, so
that the verdict cannot be retro-fitted to a passing reading. Once
`locked_at` is set, any change should produce a NEW hypothesis_id.

Schema (YAML keys, all required unless noted):
    hypothesis_id: str  (e.g. "H-C2-REVERSED")
    name: str           (one-line human name)
    mechanism: str      (multi-line paragraph: why we expect this to work)
    data_source: dict   (free-form description: tables, files, time ranges)
    signal_module: str  (Python module path, e.g.
                         "hypotheses.signals.hc2_reversed")
    signal_function: str  (function name in that module)
    signal_config: dict   (kwargs passed to the signal function — held in
                            spec so the test is reproducible)
    forward_horizons: list[int]  (forward hours, e.g. [1, 4, 24])
    primary_attribution_key: str  (for LOO: which Event.attribution key
                                    is the LOO group, e.g. "coin")
    benchmarks: list[str]         (e.g. ["BTC", "ETH", "alt_basket"] —
                                    informational; signal function decides
                                    whether to compute excess. If list is
                                    empty, engine reports raw only.)
    pre_commit_kill_criteria: list[str]  (each criterion = one line)
    trial_counter_increment: int          (default 1)
    tentative_pending_oos: bool           (true if signal was generated
                                            from the same data being tested
                                            — verdict is provisional)
    data_provenance_notes: str            (PIT discipline, look-ahead
                                            checks, known biases)
    created_at: str                        (ISO 8601)
    locked_at: str | null                  (set when spec is finalized)
"""
from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml


REQUIRED_KEYS = {
    "hypothesis_id", "name", "mechanism", "data_source",
    "signal_module", "signal_function", "signal_config",
    "forward_horizons", "primary_attribution_key", "benchmarks",
    "pre_commit_kill_criteria", "trial_counter_increment",
    "tentative_pending_oos", "data_provenance_notes", "created_at",
}

# Optional v0.2 fields (factor_type / capital_levels / venue / re_run_when)
_DEFAULTS_V0_2 = {
    "factor_type": "event_driven",   # or "cross_sectional"
    "capital_levels": [],             # list[float] USD
    "venue": None,                    # for slippage model
    "re_run_when": None,              # automation (Task D)
}


@dataclass
class HypothesisSpec:
    hypothesis_id: str
    name: str
    mechanism: str
    data_source: dict
    signal_module: str
    signal_function: str
    signal_config: dict
    forward_horizons: list[int]
    primary_attribution_key: str
    benchmarks: list[str]
    pre_commit_kill_criteria: list[str]
    trial_counter_increment: int
    tentative_pending_oos: bool
    data_provenance_notes: str
    created_at: str
    locked_at: str | None = None
    # v0.2 optional fields
    factor_type: str = "event_driven"
    capital_levels: list[float] = field(default_factory=list)
    venue: str | None = None
    re_run_when: dict | None = None
    source_path: Path | None = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HypothesisSpec":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Spec at {path} missing required keys: {sorted(missing)}"
            )
        # Coerce locked_at = None vs missing
        data.setdefault("locked_at", None)
        spec = cls(
            hypothesis_id=data["hypothesis_id"],
            name=data["name"],
            mechanism=data["mechanism"],
            data_source=data["data_source"],
            signal_module=data["signal_module"],
            signal_function=data["signal_function"],
            signal_config=data["signal_config"] or {},
            forward_horizons=list(data["forward_horizons"]),
            primary_attribution_key=data["primary_attribution_key"],
            benchmarks=list(data["benchmarks"] or []),
            pre_commit_kill_criteria=list(data["pre_commit_kill_criteria"]),
            trial_counter_increment=int(data["trial_counter_increment"]),
            tentative_pending_oos=bool(data["tentative_pending_oos"]),
            data_provenance_notes=data["data_provenance_notes"],
            created_at=data["created_at"],
            locked_at=data["locked_at"],
            # v0.2 optional fields
            factor_type=data.get("factor_type",
                                 _DEFAULTS_V0_2["factor_type"]),
            capital_levels=list(data.get("capital_levels") or []),
            venue=data.get("venue"),
            re_run_when=data.get("re_run_when"),
        )
        spec.source_path = path
        return spec

    def ensure_locked(self) -> None:
        """Raise if spec is not yet locked. The engine refuses to run on
        unlocked specs to enforce the no-retrofit discipline."""
        if not self.locked_at:
            raise ValueError(
                f"Spec {self.hypothesis_id} at {self.source_path} is "
                "NOT LOCKED. Set `locked_at: <ISO timestamp>` in the YAML "
                "to certify the spec is final. Editing a locked spec "
                "requires a NEW hypothesis_id."
            )

    def load_signal_function(self) -> Callable[[dict], Any]:
        """Import the spec's signal function. Module path is resolved
        from the parent of hypotheses/specs/ so signal modules live at
        hypotheses/signals/."""
        # Make hypotheses dir importable
        if self.source_path is not None:
            project_root = self.source_path.parents[2]
            import sys
            sp = str(project_root)
            if sp not in sys.path:
                sys.path.insert(0, sp)
        mod = importlib.import_module(self.signal_module)
        fn = getattr(mod, self.signal_function, None)
        if fn is None:
            raise ValueError(
                f"signal_function '{self.signal_function}' not found in "
                f"module '{self.signal_module}'"
            )
        return fn

    def to_yaml_dict(self) -> dict:
        d = asdict(self)
        d.pop("source_path", None)
        return d


# --- trial counter ----------------------------------------------------------

# Public note: trial-counter location is configurable. Override with the
# HL_VALIDATOR_TRIAL_COUNTER_PATH env var (the API layer scopes this
# per-user); the default is repo-relative.
DEFAULT_COUNTER_PATH = Path(
    os.environ.get(
        "HL_VALIDATOR_TRIAL_COUNTER_PATH",
        str(Path(__file__).resolve().parents[2] / "trial_counter.json"),
    )
)


def read_trial_counter(path: Path = DEFAULT_COUNTER_PATH) -> dict:
    if not path.exists():
        return {"trial_count": 0, "history": [],
                "note": "newly initialized"}
    with open(path) as f:
        return json.load(f)


def write_trial_counter(state: dict,
                        path: Path = DEFAULT_COUNTER_PATH) -> None:
    """Atomic write: temp file in same dir, then os.replace.
    POSIX guarantees os.replace is atomic on the same filesystem,
    closing the partial-write window (audit A7)."""
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def increment_trial_counter(hypothesis_id: str, increment: int,
                            verdict: str,
                            path: Path = DEFAULT_COUNTER_PATH) -> dict:
    state = read_trial_counter(path)
    state["trial_count"] = int(state.get("trial_count", 0)) + increment
    state.setdefault("history", []).append({
        "hypothesis_id": hypothesis_id,
        "increment": increment,
        "verdict": verdict,
        "applied_at": datetime.now(timezone.utc).isoformat(),
    })
    write_trial_counter(state, path)
    return state
