"""Multi-person tracker in the style of ByteTrack, written from scratch.

Why track at all when we only want a head count? Because per-frame detections
flicker: people vanish for a frame or two behind each other and reappear.
Following each person through time gives a steady count, lets us bridge short
occlusions, and yields the things a plain detector cannot: direction of travel
(for entrance counters), speed (for congestion) and dwell time.

Two ideas from ByteTrack (Zhang et al., ECCV 2022) are kept:

* association happens in two rounds, first with confident detections and then
  with the low-confidence ones, which are usually real but half-hidden people;
* a track is only born from a confident detection.

The motion model is a constant-velocity Kalman filter over (cx, cy, w, h).
Because the four dimensions are independent, the 8-state filter decomposes
into four 2-state filters, which lets every track be updated in a handful of
vectorised numpy operations instead of a Python loop over matrix products.
Time is handled in seconds, so the frame rate may vary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TrackerConfig
from .assignment import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = (a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i][None, :] for i in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.maximum(union, 1e-9)


def _match(iou: np.ndarray, min_iou: float) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one assignment, keeping only pairs with IoU >= min_iou."""
    if iou.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    rows, cols = linear_sum_assignment(-iou)
    keep = iou[rows, cols] >= min_iou
    return rows[keep], cols[keep]


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes, dtype=float)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def _cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    out = np.empty_like(state, dtype=float)
    w = np.maximum(state[:, 2], 1.0)
    h = np.maximum(state[:, 3], 1.0)
    out[:, 0] = state[:, 0] - w / 2
    out[:, 1] = state[:, 1] - h / 2
    out[:, 2] = state[:, 0] + w / 2
    out[:, 3] = state[:, 1] + h / 2
    return out


@dataclass
class Tracks:
    """The people currently being followed. All arrays share the same length."""

    ids: np.ndarray        # (N,) int, stable identity
    boxes: np.ndarray      # (N, 4) xyxy in pixels
    velocity: np.ndarray   # (N, 2) centre velocity in pixels per second
    scores: np.ndarray     # (N,) last detection confidence
    age_s: np.ndarray      # (N,) seconds since the track was born
    seen: np.ndarray       # (N,) bool, matched to a detection in this frame

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def empty() -> "Tracks":
        return Tracks(
            np.zeros(0, dtype=int), np.zeros((0, 4)), np.zeros((0, 2)),
            np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool),
        )


class ByteTracker:
    # Process noise: people change speed by up to about this many body
    # sizes per second squared. Measurement noise is a fraction of box size.
    ACCEL = 2.0
    MEAS = 0.08

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self._next_id = 1
        self._t: float | None = None
        self._pos = np.zeros((0, 4))   # cx, cy, w, h
        self._vel = np.zeros((0, 4))
        self._p00 = np.zeros((0, 4))
        self._p01 = np.zeros((0, 4))
        self._p11 = np.zeros((0, 4))
        self._ids = np.zeros(0, dtype=int)
        self._hits = np.zeros(0, dtype=int)
        self._confirmed = np.zeros(0, dtype=bool)
        self._born = np.zeros(0)
        self._last_seen = np.zeros(0)
        self._score = np.zeros(0)

    # ------------------------------------------------------------------ api

    def reset(self) -> None:
        self.__init__(self.cfg)

    def update(self, boxes: np.ndarray, scores: np.ndarray, t: float) -> Tracks:
        """Advance to time ``t`` (seconds) with this frame's detections."""
        cfg = self.cfg
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        self._predict(dt)

        n = len(self._ids)
        predicted = _cxcywh_to_xyxy(self._pos)
        high = np.flatnonzero(scores >= cfg.high_thresh)
        low = np.flatnonzero((scores >= cfg.low_thresh) & (scores < cfg.high_thresh))
        matched_track = np.full(n, -1, dtype=int)

        # Round 1: confident detections against every live track.
        rows, cols = _match(iou_matrix(predicted, boxes[high]), cfg.match_iou)
        matched_track[rows] = high[cols]
        used_high = np.zeros(len(high), dtype=bool)
        used_high[cols] = True

        # Round 2: low-confidence detections against tracks still unmatched
        # that were seen recently. These are mostly partly hidden people.
        open_tracks = np.flatnonzero(
            (matched_track < 0) & self._confirmed & (t - self._last_seen <= cfg.coast_s + dt + 1e-6)
        )
        if len(open_tracks) and len(low):
            rows, cols = _match(iou_matrix(predicted[open_tracks], boxes[low]), cfg.match_iou_low)
            matched_track[open_tracks[rows]] = low[cols]

        hit = matched_track >= 0
        if hit.any():
            self._correct(np.flatnonzero(hit), _xyxy_to_cxcywh(boxes[matched_track[hit]]))
            self._hits[hit] += 1
            self._last_seen[hit] = t
            self._score[hit] = scores[matched_track[hit]]
            self._confirmed |= self._hits >= cfg.min_hits

        # Drop tentative tracks that missed a frame, and tracks lost too long.
        keep = np.where(self._confirmed, t - self._last_seen <= cfg.max_age_s, hit)
        self._filter(keep)
        hit = hit[keep]

        # Round 3: start new tracks from confident detections nobody claimed.
        fresh = high[~used_high]
        fresh = fresh[scores[fresh] >= cfg.new_thresh]
        if len(fresh):
            self._spawn(boxes[fresh], scores[fresh], t)
            hit = np.concatenate([hit, np.ones(len(fresh), dtype=bool)])

        visible = self._confirmed & (t - self._last_seen <= cfg.coast_s + 1e-6)
        return Tracks(
            ids=self._ids[visible].copy(),
            boxes=_cxcywh_to_xyxy(self._pos[visible]),
            velocity=self._vel[visible, :2].copy(),
            scores=self._score[visible].copy(),
            age_s=t - self._born[visible],
            seen=hit[visible],
        )

    # ------------------------------------------------------------ internals

    def _predict(self, dt: float) -> None:
        if dt <= 0 or len(self._ids) == 0:
            return
        size = np.maximum(self._pos[:, 3:4], 8.0)
        sigma2 = (self.ACCEL * size) ** 2
        self._pos += self._vel * dt
        self._p00 += dt * (2 * self._p01 + dt * self._p11) + sigma2 * dt**4 / 4
        self._p01 += dt * self._p11 + sigma2 * dt**3 / 2
        self._p11 += sigma2 * dt**2
        # width and height should stay positive while a track coasts
        self._pos[:, 2:] = np.maximum(self._pos[:, 2:], 2.0)

    def _correct(self, idx: np.ndarray, z: np.ndarray) -> None:
        size = np.maximum(z[:, 3:4], 8.0)
        r = np.maximum(self.MEAS * size, 1.0) ** 2
        p00, p01, p11 = self._p00[idx], self._p01[idx], self._p11[idx]
        s = p00 + r
        k0, k1 = p00 / s, p01 / s
        y = z - self._pos[idx]
        self._pos[idx] += k0 * y
        self._vel[idx] += k1 * y
        self._p00[idx] = (1 - k0) * p00
        self._p01[idx] = (1 - k0) * p01
        self._p11[idx] = p11 - k1 * p01

    def _spawn(self, boxes: np.ndarray, scores: np.ndarray, t: float) -> None:
        z = _xyxy_to_cxcywh(boxes)
        m = len(z)
        size = np.maximum(z[:, 3:4], 8.0)
        self._pos = np.vstack([self._pos, z])
        self._vel = np.vstack([self._vel, np.zeros((m, 4))])
        self._p00 = np.vstack([self._p00, np.broadcast_to((0.1 * size) ** 2, (m, 4))])
        self._p01 = np.vstack([self._p01, np.zeros((m, 4))])
        self._p11 = np.vstack([self._p11, np.broadcast_to((1.5 * size) ** 2, (m, 4))])
        self._ids = np.concatenate([self._ids, np.arange(self._next_id, self._next_id + m)])
        self._next_id += m
        self._hits = np.concatenate([self._hits, np.ones(m, dtype=int)])
        self._confirmed = np.concatenate([self._confirmed, np.full(m, self.cfg.min_hits <= 1)])
        self._born = np.concatenate([self._born, np.full(m, t)])
        self._last_seen = np.concatenate([self._last_seen, np.full(m, t)])
        self._score = np.concatenate([self._score, scores])

    def _filter(self, keep: np.ndarray) -> None:
        if keep.all():
            return
        for name in ("_pos", "_vel", "_p00", "_p01", "_p11", "_ids", "_hits",
                     "_confirmed", "_born", "_last_seen", "_score"):
            setattr(self, name, getattr(self, name)[keep])
