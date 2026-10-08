"""What every video source hands to the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Truth:
    """Ground truth, only available from the simulator."""

    ids: np.ndarray          # (N,)
    positions: np.ndarray    # (N, 2) pixels
    crowding: np.ndarray     # (N,) number of close neighbours, drives the occlusion model


@dataclass
class Frame:
    t: float                         # seconds on the source's own clock
    image: Optional[np.ndarray]      # BGR, or None when running headless
    truth: Optional[Truth] = None


class Source(Protocol):
    size: tuple[int, int]            # (width, height)
    live: bool                       # a camera rather than a recording or simulation
    speed: float                     # playback speed relative to real time

    def read(self) -> Optional[Frame]: ...
    def close(self) -> None: ...
