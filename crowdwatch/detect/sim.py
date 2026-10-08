"""A stand-in detector for the simulator.

The simulator draws stylised people that no real detector was trained on, so
detections are produced from the simulator's ground truth and then deliberately
spoiled the way a real detector's output is spoiled:

* people are missed, and missed more often when others stand close to them
  (occlusion is the main reason camera counts run low in a crowd);
* someone hidden in a crowd tends to stay hidden for a second or more, not
  for a single frame, so the misses cannot simply be averaged away;
* half-hidden people come back with low confidence;
* boxes jitter by a pixel or two;
* now and then a detection appears where nobody is.

The rest of the pipeline never sees the ground truth, only these detections.
"""

from __future__ import annotations

import numpy as np

from ..sources.base import Frame
from .base import Detections


class SimDetector:
    BOX_PX = 18.0
    BASE_MISS = 0.04
    MISS_PER_NEIGHBOUR = 0.07
    MAX_MISS = 0.45
    JITTER_PX = 1.4
    FALSE_PER_FRAME = 0.15
    HIDE_PER_S = 0.012            # chance per second of being lost for a while ...
    HIDE_PER_S_NEIGHBOUR = 0.035  # ... which grows with each close neighbour
    HIDE_MEAN_S = 1.8

    def __init__(self, frame_size: tuple[int, int], seed: int = 0):
        self.width, self.height = frame_size
        self.rng = np.random.default_rng(seed + 1000)
        self._hidden_until: dict[int, float] = {}
        self._last_t: float | None = None

    def detect(self, frame: Frame) -> Detections:
        truth = frame.truth
        if truth is None:
            raise RuntimeError("the simulated detector only works with the simulator source")
        rng = self.rng
        n = len(truth.ids)
        dt = 0.1 if self._last_t is None else max(frame.t - self._last_t, 0.0)
        self._last_t = frame.t

        # longer occlusions: a person drops out for a spell, then comes back
        start = rng.random(n) < (self.HIDE_PER_S + self.HIDE_PER_S_NEIGHBOUR * truth.crowding) * dt
        for pid in truth.ids[start].tolist():
            self._hidden_until.setdefault(pid, frame.t + float(rng.exponential(self.HIDE_MEAN_S)))
        self._hidden_until = {p: u for p, u in self._hidden_until.items() if u > frame.t}
        hidden = np.fromiter((p in self._hidden_until for p in truth.ids.tolist()), dtype=bool, count=n)

        miss = np.clip(self.BASE_MISS + self.MISS_PER_NEIGHBOUR * truth.crowding, 0, self.MAX_MISS)
        seen = (rng.random(n) >= miss) & ~hidden
        centres = truth.positions[seen] + rng.normal(0, self.JITTER_PX, size=(int(seen.sum()), 2))
        crowd = truth.crowding[seen]
        half = self.BOX_PX / 2 * (1 + rng.normal(0, 0.05, size=(len(centres), 1)))
        boxes = np.hstack([centres - half, centres + half])
        # confidence falls as a person gets more hidden
        scores = np.clip(rng.normal(0.86, 0.06, size=len(centres)) - 0.12 * crowd
                         + rng.normal(0, 0.05, size=len(centres)), 0.12, 0.98)

        k = int(rng.poisson(self.FALSE_PER_FRAME))
        if k:
            ghosts = np.column_stack([rng.uniform(0, self.width, k), rng.uniform(0, self.height, k)])
            boxes = np.vstack([boxes, np.hstack([ghosts - self.BOX_PX / 2, ghosts + self.BOX_PX / 2])])
            scores = np.concatenate([scores, rng.uniform(0.3, 0.7, k)])
        return Detections(boxes.astype(np.float32), scores.astype(np.float32))
