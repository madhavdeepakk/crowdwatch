"""Ready-made configurations."""

from __future__ import annotations

from typing import Optional, Union

from .config import AppConfig
from .sources.simulator import demo_layout


def demo_config(scenario: str = "surge", seed: int = 7, time_scale: float = 4.0,
                storage_path: Optional[str] = "data/crowdwatch.db") -> AppConfig:
    """The simulated mall: four zones, three doorways, a calibrated ceiling camera."""
    layout = demo_layout()
    return AppConfig.model_validate({
        "name": "Riverside Mall (simulated)",
        "source": {"type": "simulator", "scenario": scenario, "seed": seed, "time_scale": time_scale},
        "processing": {"fps": 10, "anchor": "center"},
        "detector": {"backend": "sim"},
        "calibration": layout["calibration"],
        "zones": layout["zones"],
        "lines": layout["lines"],
        "privacy": {"blur_people": False},
        "storage": {"path": storage_path},
    })


def quick_video_config(uri: Union[str, int], capacity: int = 30, model: str = "yolox_s",
                       storage_path: Optional[str] = "data/crowdwatch.db") -> AppConfig:
    """Watch a whole camera view as a single zone. Zones can be redrawn in the dashboard."""
    return AppConfig.model_validate({
        "name": "Camera 1",
        "source": {"type": "video", "uri": uri, "loop": True},
        "detector": {"backend": "yolox", "model": model},
        "zones": [{
            "id": "whole_view", "name": "Whole view", "capacity": capacity,
            "polygon": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }],
        "storage": {"path": storage_path},
    })
