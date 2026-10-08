import numpy as np
import pytest

from crowdwatch.config import AlertConfig, AppConfig


@pytest.fixture
def alert_cfg() -> AlertConfig:
    return AlertConfig()


@pytest.fixture
def one_zone_cfg() -> AppConfig:
    """A 100 x 100 pixel frame whose left half is a zone that holds five people."""
    return AppConfig.model_validate({
        "processing": {"fps": 10, "anchor": "center"},
        "detector": {"backend": "sim"},
        "zones": [{"id": "left", "name": "Left half", "capacity": 5,
                   "polygon": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]}],
        "lines": [{"id": "gate", "name": "Gate", "a": (0.5, 1.0), "b": (0.5, 0.0)}],
        "storage": {"path": None},
    })


def boxes_at(points, size=6.0):
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.hstack([pts - size / 2, pts + size / 2])
