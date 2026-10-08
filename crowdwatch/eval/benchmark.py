"""Counting accuracy on real images with hand-labelled ground truth.

The simulator tests the logic; this tests the eyes. It runs a counter over a
public crowd-counting benchmark and reports the standard metrics:

* MAE, the mean absolute error in people per image;
* RMSE, which punishes large misses;
* bias, which shows whether the counter runs low or high.

The loader understands the ShanghaiTech layout (``images/IMG_n.jpg`` beside
``ground_truth/GT_IMG_n.mat``). The dataset is not bundled: it is free for
research use and you need your own copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class Sample:
    image: Path
    count: int
    points: np.ndarray      # (N, 2) head positions, x then y


def load_shanghaitech(split_dir: str | Path, limit: Optional[int] = None) -> list[Sample]:
    """Read one split, e.g. ``.../part_B_final/test_data``."""
    import scipy.io

    split_dir = Path(split_dir)
    images = sorted((split_dir / "images").glob("*.jpg"),
                    key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not images:
        raise FileNotFoundError(f"no images found under {split_dir / 'images'}")
    samples = []
    for image in images[:limit]:
        truth = split_dir / "ground_truth" / f"GT_{image.stem}.mat"
        points = scipy.io.loadmat(truth)["image_info"][0, 0][0, 0][0]
        samples.append(Sample(image, len(points), np.asarray(points, dtype=float).reshape(-1, 2)))
    return samples


@dataclass
class Score:
    name: str
    n: int
    mae: float
    rmse: float
    bias: float
    by_size: dict[str, tuple[int, float, float]]   # bucket -> (images, MAE, bias)

    def row(self) -> str:
        return f"| {self.name} | {self.mae:.1f} | {self.rmse:.1f} | {self.bias:+.1f} |"


BUCKETS = (("up to 50 people", 0, 50), ("51 to 150", 51, 150), ("more than 150", 151, 10**9))


def score(name: str, predicted: np.ndarray, truth: np.ndarray) -> Score:
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = predicted - truth
    by_size = {}
    for label, lo, hi in BUCKETS:
        mask = (truth >= lo) & (truth <= hi)
        if mask.any():
            by_size[label] = (int(mask.sum()), float(np.abs(err[mask]).mean()), float(err[mask].mean()))
    return Score(name, len(err), float(np.abs(err).mean()), float(np.sqrt((err**2).mean())),
                 float(err.mean()), by_size)


def run_counter(samples: list[Sample], count_image: Callable[[np.ndarray], float],
                progress: bool = False) -> np.ndarray:
    import cv2

    out = []
    for i, sample in enumerate(samples):
        image = cv2.imread(str(sample.image))
        out.append(float(count_image(image)))
        if progress and (i + 1) % 25 == 0:
            print(f"\r  {i + 1}/{len(samples)} images", end="", flush=True)
    if progress:
        print()
    return np.asarray(out)
