"""Small, dependency-light geometry helpers used across the pipeline."""

from __future__ import annotations

import numpy as np


def polygon_area(polygon: np.ndarray) -> float:
    """Area of a simple polygon (shoelace formula). Units follow the input."""
    p = np.asarray(polygon, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting. Returns a boolean mask of shape (N,)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float)
    if len(pts) == 0 or len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    hits = straddles & (x < x_cross)
    return (hits.sum(axis=1) % 2).astype(bool)


def signed_distance_to_line(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed perpendicular distance from each point to the infinite line a->b.

    Positive values are on the left of the direction of travel a->b in image
    coordinates (y pointing down), negative on the right.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    length = float(np.hypot(*d)) or 1.0
    return ((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / length


def projection_along_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Position of each point's projection on segment a->b, 0 at a and 1 at b."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    denom = float(d @ d) or 1.0
    return ((pts - a) @ d) / denom


class Homography:
    """Maps image pixels to floor coordinates in metres.

    Built from four or more point pairs: where a spot appears in the image and
    where that spot is on the floor plan. With it, densities come out in people
    per square metre and speeds in metres per second.
    """

    def __init__(self, image_points: np.ndarray, floor_points: np.ndarray):
        import cv2

        src = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(floor_points, dtype=np.float64).reshape(-1, 2)
        if len(src) < 4 or len(src) != len(dst):
            raise ValueError("calibration needs at least four matching image/floor point pairs")
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError("calibration points are degenerate (are three of them in a line?)")
        self.matrix = matrix

    def to_floor(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2))
        homog = np.hstack([pts, np.ones((len(pts), 1))]) @ self.matrix.T
        w = homog[:, 2:3]
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        return homog[:, :2] / w
