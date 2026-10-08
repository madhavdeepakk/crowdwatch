"""Counting packed crowds with a density map.

A box detector has to find each person, and in a tight crowd most people are
mostly hidden. A density-map network takes a different route: for every patch
of the picture it estimates how many people are there, as a fraction, and the
count for any region is simply the sum over that region. Nobody has to be
separated from their neighbours, so it keeps working where detection gives up.

The network used here is DM-Count (Wang et al., NeurIPS 2020; MIT licence), a
VGG-19 trunk with a small regression head, run through ONNX Runtime. Its
published weights are PyTorch files, so the first use converts them to ONNX,
which needs PyTorch installed once (``pip install torch``). After that only
ONNX Runtime is used.

It is slow on a CPU (a second or more per frame), which is fine: it runs every
couple of seconds beside the tracker, not on every frame. See ``fuse``.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from .yolox import model_dir

_RELEASE = "https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1"
WEIGHTS = {
    # trained on UCF-QNRF: varied scenes, the better choice for a camera it has never seen
    "dmcount_qnrf": f"{_RELEASE}/DM-Count_QNRF.pth",
    # trained on ShanghaiTech Part B street scenes
    "dmcount_shb": f"{_RELEASE}/DM-Count_SHB.pth",
}
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_VGG19 = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]


def export_dmcount(weights: Path, onnx_path: Path) -> None:
    """Convert DM-Count's PyTorch weights to an ONNX file that accepts any image size."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "converting the density model needs PyTorch once: run `pip install torch`, "
            "then try again"
        ) from exc

    layers, channels = [], 3
    for v in _VGG19:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(channels, v, 3, padding=1), nn.ReLU(inplace=True)]
            channels = v

    class DMCount(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*layers)
            self.reg_layer = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
            self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

        def forward(self, x):
            x = self.features(x)
            x = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
            return self.density_layer(self.reg_layer(x))

    net = DMCount()
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    if "model" in state and not hasattr(state["model"], "shape"):
        state = state["model"]                      # some checkpoints wrap the weights
    state = {k.removeprefix("module."): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 384, 512)
    torch.onnx.export(net, dummy, str(onnx_path), input_names=["image"], output_names=["density"],
                      dynamic_axes={"image": {2: "height", 3: "width"}, "density": {2: "h", 3: "w"}},
                      opset_version=17, dynamo=False)


def ensure_density_model(name: str = "dmcount_qnrf", quiet: bool = False) -> Path:
    """Path of the ONNX density model, downloading and converting it on first use."""
    if name not in WEIGHTS:
        raise ValueError(f"unknown density model '{name}'. Choose from: {', '.join(WEIGHTS)}")
    onnx_path = model_dir() / f"{name}.onnx"
    if onnx_path.exists():
        return onnx_path
    weights = model_dir() / f"{name}.pth"
    if not weights.exists():
        if not quiet:
            print(f"Downloading the {name} density model (86 MB, one time) ...", file=sys.stderr)
        weights.parent.mkdir(parents=True, exist_ok=True)
        tmp = weights.with_suffix(".part")
        try:
            urllib.request.urlretrieve(WEIGHTS[name], tmp)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"could not download {WEIGHTS[name]} ({exc})") from exc
        tmp.replace(weights)
    if not quiet:
        print("Converting it to ONNX ...", file=sys.stderr)
    export_dmcount(weights, onnx_path)
    return onnx_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DensityCounter:
    def __init__(self, model: str = "dmcount_qnrf", model_path: Optional[str] = None,
                 max_side: int = 1024, threads: int = 0):
        import onnxruntime as ort

        path = Path(model_path) if model_path else ensure_density_model(model)
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.max_side = max_side

    def density_map(self, image: np.ndarray) -> np.ndarray:
        """People per cell, as an array one eighth the size of the (resized) image.

        The values sum to the estimated number of people in the picture.
        """
        import cv2

        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        nw, nh = max(32, int(round(w * scale / 16)) * 16), max(32, int(round(h * scale / 16)) * 16)
        rgb = cv2.cvtColor(cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        blob = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        return self.session.run(None, {"image": blob})[0][0, 0]

    def count(self, image: np.ndarray) -> float:
        return float(self.density_map(image).sum())


def fuse(tracked: float, density: Optional[float], min_people: float = 15.0,
         switch_ratio: float = 1.2) -> float:
    """Combine the two counts: the tracker in sparse scenes, the density map in packed ones.

    Tracking is exact when people are apart and it is the only source of
    direction, speed and dwell time, so it stays in charge by default. Its
    failure is one-sided: in a dense crowd it misses people, it does not invent
    them. So the density estimate takes over only when it is clearly higher
    and there are enough people for a density map to be reliable.
    """
    if density is None:
        return tracked
    if density >= min_people and density >= tracked * switch_ratio:
        return float(density)
    return tracked


def zone_counts(density: np.ndarray, polygons_norm: list[np.ndarray]) -> list[float]:
    """Sum a density map inside each zone. Polygons are in 0..1 picture coordinates."""
    import cv2

    h, w = density.shape
    out = []
    for polygon in polygons_norm:
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = (np.asarray(polygon, dtype=float) * [w, h]).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        out.append(float(density[mask.astype(bool)].sum()))
    return out
