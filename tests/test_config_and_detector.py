import numpy as np
import pytest
from pydantic import ValidationError

from crowdwatch.config import AppConfig, load_config, save_config
from crowdwatch.detect.yolox import MODELS, decode, model_dir, nms
from crowdwatch.presets import demo_config, quick_video_config


def test_demo_config_round_trips_through_yaml(tmp_path):
    cfg = demo_config("gradual", 5, storage_path=None)
    path = tmp_path / "demo.yaml"
    save_config(cfg, path)
    again = load_config(path)
    assert again.zones == cfg.zones and again.lines == cfg.lines
    assert again.source.scenario == "gradual"


def test_missing_config_file_has_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("patch, message", [
    ({"zones": [{"id": "a", "name": "A", "capacity": 5, "polygon": [(0, 0), (1, 0)]}]}, "at least 3"),
    ({"zones": [{"id": "a", "name": "A", "capacity": 5, "polygon": [(0, 0), (9, 0), (1, 1)]}]}, "normalised"),
    ({"zones": [{"id": "Bad Id", "name": "A", "capacity": 5, "polygon": [(0, 0), (1, 0), (1, 1)]}]}, "pattern"),
    ({"zones": [{"id": "a", "name": "A", "capacity": 5, "polygon": [(0, 0), (1, 0), (1, 1)]}] * 2}, "duplicate zone"),
    ({"alerts": {"warning_ratio": 1.2}}, "busy < warning < critical"),
    ({"source": {"type": "video"}}, "source.uri is required"),
    ({"surprise": 1}, "Extra inputs"),
])
def test_bad_configs_are_rejected(patch, message):
    with pytest.raises(ValidationError, match=message):
        AppConfig.model_validate(patch)


def test_quick_video_config_watches_the_whole_view():
    cfg = quick_video_config("lobby.mp4", capacity=25, storage_path=None)
    assert cfg.source.uri == "lobby.mp4" and cfg.zones[0].capacity == 25
    assert cfg.privacy.blur_people


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.6, 0.9, 0.8])
    assert sorted(nms(boxes, scores, 0.5).tolist()) == [1, 2]
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).size == 0


def test_decode_places_boxes_on_the_right_grid_cell():
    raw = np.zeros((8400, 85), dtype=np.float32)       # 80x80 + 40x40 + 20x20 cells
    out = decode(raw.copy(), (640, 640))
    assert out[0, :4].tolist() == [0, 0, 8, 8]          # first cell of the stride-8 grid
    assert out[81, :2].tolist() == [8, 8]               # second row, second column
    assert out[6400, 2:4].tolist() == [16, 16]          # first cell of the stride-16 grid
    assert out[-1, :2].tolist() == [19 * 32, 19 * 32]


@pytest.mark.skipif(not (model_dir() / "yolox_tiny.onnx").exists(), reason="detector model not downloaded")
def test_yolox_runs_and_returns_sane_boxes():
    from crowdwatch.config import DetectorConfig
    from crowdwatch.detect.yolox import YoloxDetector

    detector = YoloxDetector(DetectorConfig(model="yolox_tiny"))
    image = np.full((360, 640, 3), 120, dtype=np.uint8)
    dets = detector.detect_image(image)
    assert dets.boxes.shape[1] == 4 and len(dets.boxes) == len(dets.scores)
    assert (dets.scores >= 0.1).all()
    assert set(MODELS) == {"yolox_tiny", "yolox_s"}


def test_shipped_example_configs_are_valid():
    from pathlib import Path

    configs = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))
    assert len(configs) >= 2
    for path in configs:
        cfg = load_config(path)
        assert cfg.zones, path.name
