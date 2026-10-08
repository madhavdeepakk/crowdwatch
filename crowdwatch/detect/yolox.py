"""Person detection with YOLOX, run through ONNX Runtime.

YOLOX (Ge et al., 2021) is used because its weights are Apache-2.0 licensed
and published as ready-made ONNX files, so the whole system installs without
PyTorch and runs on a plain CPU. Only the "person" class is kept.

Small, far-away people are the hard case in crowd footage. With
``detector.tiles`` set to, say, ``[2, 2]`` the frame is also cut into
overlapping tiles that are each detected at full model resolution, and the
results are merged. It costs one extra model run per tile.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import DetectorConfig
from ..sources.base import Frame
from .base import Detections

_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
MODELS = {
    "yolox_tiny": (f"{_RELEASE}/yolox_tiny.onnx",
                   "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
    "yolox_s": (f"{_RELEASE}/yolox_s.onnx",
                "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
}
PERSON_CLASS = 0


def model_dir() -> Path:
    return Path(os.environ.get("CROWDWATCH_MODELS", Path.home() / ".cache" / "crowdwatch" / "models"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(name: str, quiet: bool = False) -> Path:
    """Return the path of a model file, downloading and verifying it on first use."""
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}'. Choose from: {', '.join(MODELS)}")
    url, sha = MODELS[name]
    path = model_dir() / f"{name}.onnx"
    if path.exists() and _sha256(path) == sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"Downloading the {name} person detector (one time) ...", file=sys.stderr)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download {url} ({exc}). Download it yourself and set "
            f"detector.model_path in the config, or place it at {path}"
        ) from exc
    if _sha256(tmp) != sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"the downloaded {name} model failed its checksum; please try again")
    tmp.replace(path)
    return path


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns the indices of the boxes to keep."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=int)


def decode(raw: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    """Turn YOLOX's grid-relative output (N, 5 + classes) into pixel boxes, in place."""
    h, w = input_hw
    grids, strides = [], []
    for stride in (8, 16, 32):
        gh, gw = h // stride, w // stride
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grids.append(np.stack([xv, yv], axis=2).reshape(-1, 2))
        strides.append(np.full((gh * gw, 1), stride))
    grid = np.concatenate(grids)
    stride = np.concatenate(strides)
    raw[:, :2] = (raw[:, :2] + grid) * stride
    raw[:, 2:4] = np.exp(raw[:, 2:4]) * stride
    return raw


class YoloxDetector:
    def __init__(self, cfg: DetectorConfig, min_score: float = 0.1):
        import onnxruntime as ort

        path = Path(cfg.model_path) if cfg.model_path else ensure_model(cfg.model)
        if not path.exists():
            raise FileNotFoundError(f"detector model not found: {path}")
        options = ort.SessionOptions()
        if cfg.threads:
            options.intra_op_num_threads = cfg.threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        self.input_hw = (int(meta.shape[2]), int(meta.shape[3]))
        self.min_score = min_score
        self.nms_iou = cfg.nms
        self.tiles = tuple(cfg.tiles)

    # ------------------------------------------------------------------

    def _run(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect people in one image. Boxes come back in that image's pixels."""
        import cv2

        ih, iw = self.input_hw
        h, w = image.shape[:2]
        ratio = min(ih / h, iw / w)
        resized = cv2.resize(image, (max(1, int(w * ratio)), max(1, int(h * ratio))),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)

        raw = self.session.run(None, {self.input_name: blob})[0][0]
        raw = decode(raw, self.input_hw)
        scores = raw[:, 4] * raw[:, 5 + PERSON_CLASS]
        keep = scores >= self.min_score
        raw, scores = raw[keep], scores[keep]
        boxes = np.empty((len(raw), 4), dtype=np.float32)
        boxes[:, 0] = raw[:, 0] - raw[:, 2] / 2
        boxes[:, 1] = raw[:, 1] - raw[:, 3] / 2
        boxes[:, 2] = raw[:, 0] + raw[:, 2] / 2
        boxes[:, 3] = raw[:, 1] + raw[:, 3] / 2
        boxes /= ratio
        return boxes, scores.astype(np.float32)

    def detect_image(self, image: np.ndarray) -> Detections:
        h, w = image.shape[:2]
        boxes, scores = self._run(image)
        cols, rows = self.tiles
        if cols * rows > 1:
            all_boxes, all_scores = [boxes], [scores]
            overlap = 0.2
            tw, th = w / (cols - (cols - 1) * overlap), h / (rows - (rows - 1) * overlap)
            for r in range(rows):
                for c in range(cols):
                    x0 = int(round(c * tw * (1 - overlap)))
                    y0 = int(round(r * th * (1 - overlap)))
                    x1, y1 = min(w, int(round(x0 + tw))), min(h, int(round(y0 + th)))
                    b, s = self._run(image[y0:y1, x0:x1])
                    # a box cut off by a tile edge is a partial person; the
                    # neighbouring tile or the full frame has the whole one
                    inner = ((b[:, 0] > 2) | (x0 == 0)) & ((b[:, 1] > 2) | (y0 == 0)) \
                        & ((b[:, 2] < x1 - x0 - 2) | (x1 == w)) & ((b[:, 3] < y1 - y0 - 2) | (y1 == h))
                    b, s = b[inner], s[inner]
                    b[:, [0, 2]] += x0
                    b[:, [1, 3]] += y0
                    all_boxes.append(b)
                    all_scores.append(s)
            boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        keep = nms(boxes, scores, self.nms_iou)
        return Detections(boxes[keep], scores[keep])

    def detect(self, frame: Frame) -> Detections:
        if frame.image is None:
            return Detections.empty()
        return self.detect_image(frame.image)


def build_detector(cfg: DetectorConfig, frame_size: tuple[int, int], seed: int = 0,
                   min_score: float = 0.1):
    if cfg.backend == "sim":
        from .sim import SimDetector

        return SimDetector(frame_size, seed)
    return YoloxDetector(cfg, min_score=min_score)
