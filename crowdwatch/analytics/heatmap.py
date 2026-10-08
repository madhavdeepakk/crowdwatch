"""A slowly fading map of where people have been standing."""

from __future__ import annotations

import math

import numpy as np


class Heatmap:
    def __init__(self, frame_size: tuple[int, int], cell_px: int = 16, tau_s: float = 20.0):
        self.width, self.height = frame_size
        self.cell = cell_px
        self.tau = tau_s
        self.grid = np.zeros((max(1, self.height // cell_px), max(1, self.width // cell_px)), dtype=np.float32)

    def update(self, dt: float, points: np.ndarray) -> None:
        if dt <= 0:
            return
        self.grid *= math.exp(-dt / self.tau)
        if len(points) == 0:
            return
        gx = np.clip((points[:, 0] / self.cell).astype(int), 0, self.grid.shape[1] - 1)
        gy = np.clip((points[:, 1] / self.cell).astype(int), 0, self.grid.shape[0] - 1)
        np.add.at(self.grid, (gy, gx), dt)

    def normalised(self) -> np.ndarray:
        """Values in 0..1, where 1 means a cell that has been occupied continuously."""
        return np.clip(self.grid / self.tau, 0.0, 1.0)
