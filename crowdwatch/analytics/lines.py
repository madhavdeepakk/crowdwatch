"""Directional people counters for doorways and corridors.

A counting line is a segment a->b drawn across a walkway. Stand at ``a`` and
look towards ``b``: someone who walks across from your left to your right is
counted "in", the other way is "out".

People hovering on the line would otherwise be counted again and again, so a
person only changes side once they are a small margin past the line.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..config import LineConfig
from ..geometry import projection_along_segment, signed_distance_to_line


class LineCounter:
    RATE_WINDOW_S = 30.0

    def rates_per_min(self) -> tuple[float, float]:
        """People a minute coming in and going out, over the last half minute."""
        scale = 60.0 / self.RATE_WINDOW_S
        ins = sum(1 for _, d in self._recent if d > 0)
        return ins * scale, (len(self._recent) - ins) * scale

    def __init__(self, cfg: LineConfig, a_px: np.ndarray, b_px: np.ndarray, margin_px: float):
        self.cfg = cfg
        self.a = np.asarray(a_px, dtype=float)
        self.b = np.asarray(b_px, dtype=float)
        self.margin = margin_px
        self.count_in = 0
        self.count_out = 0
        self._side: dict[int, int] = {}
        self._last_seen: dict[int, float] = {}
        self._recent: deque[tuple[float, int]] = deque()     # (time, +1 in / -1 out)

    def update(self, t: float, ids: np.ndarray, points: np.ndarray) -> None:
        if len(ids):
            dist = signed_distance_to_line(points, self.a, self.b)
            along = projection_along_segment(points, self.a, self.b)
            for pid, d, s in zip(ids.tolist(), dist.tolist(), along.tolist()):
                self._last_seen[pid] = t
                if abs(d) < self.margin:
                    continue
                side = 1 if d > 0 else -1
                previous = self._side.get(pid)
                self._side[pid] = side
                if previous is None or previous == side:
                    continue
                if -0.05 <= s <= 1.05:        # crossed within the segment, not around its end
                    if previous > 0:
                        self.count_in += 1
                        self._recent.append((t, 1))
                    else:
                        self.count_out += 1
                        self._recent.append((t, -1))
        while self._recent and t - self._recent[0][0] > self.RATE_WINDOW_S:
            self._recent.popleft()
        if len(self._last_seen) > 4 * max(len(ids), 50):
            for pid in [p for p, seen in self._last_seen.items() if t - seen > 5.0]:
                self._last_seen.pop(pid, None)
                self._side.pop(pid, None)

    def to_dict(self) -> dict:
        return {"id": self.cfg.id, "name": self.cfg.name,
                "in": self.count_in, "out": self.count_out,
                "net": self.count_in - self.count_out}
