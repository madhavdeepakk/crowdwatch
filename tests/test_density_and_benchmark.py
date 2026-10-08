import numpy as np
import pytest

from crowdwatch.detect.density import fuse, zone_counts
from crowdwatch.engine import Engine
from crowdwatch.eval.benchmark import load_shanghaitech, run_counter, score

from .conftest import boxes_at


def test_fuse_trusts_tracking_when_sparse_and_density_when_packed():
    assert fuse(8, None) == 8
    assert fuse(8, 11) == 8                 # too few people for a density map to overrule
    assert fuse(40, 44) == 40               # the two agree: keep the tracker
    assert fuse(40, 95) == 95               # a packed crowd the detector cannot separate
    assert fuse(40, 20) == 40               # the density map never pulls the count down


def test_zone_counts_sum_the_density_inside_each_outline():
    density = np.zeros((40, 60), dtype=np.float32)
    density[:, :30] = 100 / (40 * 30)       # 100 people spread over the left half
    density[:, 30:] = 20 / (40 * 30)        # 20 over the right half
    left = [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]
    right = [(0.5, 0), (1, 0), (1, 1), (0.5, 1)]
    counts = zone_counts(density, [left, right])
    assert counts[0] == pytest.approx(100, rel=0.06)
    assert counts[1] == pytest.approx(20, rel=0.08)


def test_density_estimate_takes_over_a_packed_zone(one_zone_cfg):
    one_zone_cfg.zones[0].capacity = 50
    engine = Engine(one_zone_cfg, (100, 100))
    people = [(8 + 10 * (i % 4), 10 + 12 * (i // 4)) for i in range(16)]     # the tracker sees 16
    for i in range(30):
        boxes = boxes_at(people, 8)
        snap = engine.step(i * 0.1, boxes, np.full(len(boxes), 0.9))
    assert snap["zones"][0]["method"] == "tracking" and snap["zones"][0]["count"] == 16
    for i in range(30, 120):
        engine.set_density(i * 0.1, {"left": 46.0}, total=50.0)
        boxes = boxes_at(people, 8)
        snap = engine.step(i * 0.1, boxes, np.full(len(boxes), 0.9))
    zone = snap["zones"][0]
    assert zone["method"] == "density" and zone["count"] == 46
    assert zone["raw"] == 16 and zone["density_count"] == 46
    assert zone["level_name"] == "warning"          # 92% of 50: the tracker alone would say "normal"
    assert snap["people"] == 50


def test_stale_density_estimates_are_dropped(one_zone_cfg):
    one_zone_cfg.zones[0].capacity = 50
    engine = Engine(one_zone_cfg, (100, 100))
    engine.set_density(0.0, {"left": 46.0})
    people = [(10, 10), (20, 20)]
    snap = None
    for i in range(150):
        boxes = boxes_at(people, 8)
        snap = engine.step(i * 0.1, boxes, np.full(2, 0.9))
    assert snap["zones"][0]["method"] == "tracking" and snap["zones"][0]["density_count"] is None


def test_people_walking_towards_a_zone_are_counted_as_approaching(one_zone_cfg):
    engine = Engine(one_zone_cfg, (100, 100))
    snap = None
    for i in range(25):
        heading_in = (90 - 2.0 * i, 30)          # right half, walking left towards the zone
        heading_away = (60 + 1.0 * i, 70)        # right half, walking further right
        boxes = boxes_at([heading_in, heading_away], 8)
        snap = engine.step(i * 0.1, boxes, np.full(2, 0.9))
        if heading_in[0] < 72:
            break
    assert snap["zones"][0]["approaching"] == 1


def test_score_reports_error_bias_and_size_buckets():
    truth = np.array([10, 40, 100, 200, 400])
    s = score("x", truth * 0.5, truth)
    assert s.mae == pytest.approx(75) and s.bias == pytest.approx(-75)
    assert s.rmse > s.mae
    assert s.by_size["up to 50 people"][0] == 2 and s.by_size["more than 150"][0] == 2
    assert s.row().startswith("| x | 75.0 |")


def test_shanghaitech_loader_reads_images_and_head_counts(tmp_path):
    cv2 = pytest.importorskip("cv2")
    scipy_io = pytest.importorskip("scipy.io")
    (tmp_path / "images").mkdir()
    (tmp_path / "ground_truth").mkdir()
    for n, heads in ((2, 5), (10, 3), (1, 7)):
        cv2.imwrite(str(tmp_path / "images" / f"IMG_{n}.jpg"), np.zeros((32, 48, 3), np.uint8))
        inner = np.zeros((1, 1), dtype=[("location", "O"), ("number", "O")])     # the dataset's own layout
        inner[0, 0] = (np.random.rand(heads, 2) * 30, np.array([[heads]]))
        info = np.empty((1, 1), dtype=object)
        info[0, 0] = inner
        scipy_io.savemat(str(tmp_path / "ground_truth" / f"GT_IMG_{n}.mat"), {"image_info": info})
    samples = load_shanghaitech(tmp_path)
    assert [s.image.name for s in samples] == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg"]     # numeric order
    assert [s.count for s in samples] == [7, 5, 3]
    counts = run_counter(samples, lambda image: image.shape[1])
    assert counts.tolist() == [48, 48, 48]
    with pytest.raises(FileNotFoundError):
        load_shanghaitech(tmp_path / "missing")
