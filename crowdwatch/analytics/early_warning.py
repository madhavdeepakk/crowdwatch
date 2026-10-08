"""A learned early-warning model.

The rule in ``forecast.py`` extrapolates one number, the count. The tracker
knows more than that: how many people are walking towards a zone right now,
how fast the doors are letting people in, whether the people already there
are settling or passing through. This model combines those signals into one
probability: *how likely is this zone to reach capacity within the next two
minutes?*

It is a gradient-boosted tree ensemble trained on simulated crowd build-ups
with randomised timing, size and location (see ``eval/train.py``), exported to
ONNX and run with the same runtime as the person detector.

Honest status: the evidence is mixed. On the final held-out simulations it
warned of more events than the hand-built rule (80% against 57%) with a
similar share of false warnings; on the validation runs used while building it
the rule was slightly ahead on balance (docs/evaluation.md has the numbers).
So the simpler rule is the default and this model is opt-in
(``alerts.forecast_method: learned``). It was trained on 60 simulated
afternoons, which is little, and it has never seen a real crowd, so its
probability is a ranking of concern, not a calibrated forecast for a building.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

# Everything is scaled by the zone's capacity so one model fits zones of any size.
FEATURES = [
    "ratio",          # people now / capacity
    "slope15",        # change per minute over the last 15 s, as a share of capacity
    "slope45",        # ... over 45 s
    "slope120",       # ... over 120 s
    "churn",          # arrivals plus departures per minute / capacity
    "approaching",    # people outside the zone walking towards it / capacity
    "leaving",        # people inside it walking out / capacity
    "damped",         # the rule-based damped forecast two minutes ahead / capacity
    "speed",          # median walking speed in the zone, m/s
    "dwell",          # mean time people here have stayed, minutes
    "door_in",        # venue-wide door entries per minute / capacity
    "door_net",       # entries minus exits per minute / capacity
    "venue",          # everyone in view / capacity
]
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "early_warning.onnx"


class EarlyWarningModel:
    def __init__(self, path: Path = DEFAULT_MODEL):
        import onnxruntime as ort

        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta["features"] != FEATURES:
            raise ValueError("the early-warning model was trained on a different feature set")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.threshold: float = float(meta["threshold"])
        self.lead_s: float = float(meta["lead_s"])
        self.meta = meta

    def predict(self, features: np.ndarray) -> float:
        """Probability that the zone reaches capacity within ``lead_s`` seconds."""
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.session.run(None, {self.input_name: x})[1][0, 1])

    def predict_many(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURES))
        return self.session.run(None, {self.input_name: x})[1][:, 1]


_cache: dict[str, Optional[EarlyWarningModel]] = {}


def load_model(path: Optional[str] = None) -> Optional[EarlyWarningModel]:
    """The packaged model, or None if it is missing (the rule is used instead)."""
    key = str(path or DEFAULT_MODEL)
    if key not in _cache:
        target = Path(key)
        _cache[key] = EarlyWarningModel(target) if target.exists() else None
    return _cache[key]
