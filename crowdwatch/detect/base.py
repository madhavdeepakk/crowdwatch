"""What every person detector returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..sources.base import Frame


@dataclass
class Detections:
    boxes: np.ndarray     # (N, 4) xyxy in pixels
    scores: np.ndarray    # (N,) confidence 0..1

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32))


class Detector(Protocol):
    def detect(self, frame: Frame) -> Detections: ...
