import numpy as np
import pytest

from crowdwatch.config import AppConfig, ZoneConfig
from crowdwatch.engine import ConfigError, Engine

from .conftest import boxes_at


def step(engine, t, points):
    boxes = boxes_at(points, size=8)
    return engine.step(t, boxes, np.full(len(boxes), 0.9))


def test_counts_only_people_inside_the_zone(one_zone_cfg):
    engine = Engine(one_zone_cfg, (100, 100))
    inside = [(10, 20), (20, 60), (40, 80)]
    outside = [(70, 30), (90, 90)]
    for i in range(40):
        snap = step(engine, i * 0.1, inside + outside)
    zone = snap["zones"][0]
    assert snap["people"] == 5
    assert zone["raw"] == 3 and zone["count"] == 3
    assert zone["ratio"] == pytest.approx(0.6, abs=0.02)
    assert zone["level_name"] == "busy"


def test_overcrowding_raises_one_critical_alert(one_zone_cfg):
    engine = Engine(one_zone_cfg, (100, 100))
    crowd = [(8 + 12 * (i % 3), 10 + 14 * (i // 3)) for i in range(6)]     # six people, limit five
    for i in range(80):
        snap = step(engine, i * 0.1, crowd)
    assert snap["zones"][0]["level_name"] == "critical"
    assert snap["status"]["level"] == 3
    alerts = [a for a in snap["alerts"] if a["kind"] == "overcrowding"]
    assert len(alerts) == 1 and alerts[0]["level"] == "critical"
    assert "over capacity: 6 of 5" in snap["status"]["headline"]


def test_door_counter_counts_each_direction_once(one_zone_cfg):
    engine = Engine(one_zone_cfg, (100, 100))
    t = 0.0
    # the gate runs up the middle of the frame; walking left to right is "in"
    for x in np.linspace(20, 80, 40):
        step(engine, t, [(x, 50)]); t += 0.1
    for x in np.linspace(80, 20, 40):
        snap = step(engine, t, [(x, 50)]); t += 0.1
    assert snap["lines"][0] == {"id": "gate", "name": "Gate", "in": 1, "out": 1, "net": 0}


def test_loitering_on_the_line_is_not_counted_repeatedly(one_zone_cfg):
    engine = Engine(one_zone_cfg, (100, 100))
    rng = np.random.default_rng(0)
    for i in range(200):
        snap = step(engine, i * 0.1, [(50 + rng.normal(0, 0.4), 50)])
    assert snap["lines"][0]["in"] + snap["lines"][0]["out"] == 0


def test_capacity_from_floor_area_and_calibration():
    cfg = AppConfig.model_validate({
        "calibration": {"image_points": [(0, 0), (1, 0), (1, 1), (0, 1)],
                        "floor_points": [(0, 0), (20, 0), (20, 10), (0, 10)]},
        "default_max_density": 2.0,
        "zones": [
            {"id": "half", "name": "Half", "polygon": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]},
            {"id": "tight", "name": "Tight", "polygon": [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
             "max_density": 0.5},
            {"id": "given", "name": "Given", "polygon": [(0, 0), (1, 0), (1, 1)], "area_m2": 30},
        ],
        "storage": {"path": None},
    })
    engine = Engine(cfg, (200, 100))
    capacities = {z.cfg.id: (z.capacity, z.area_m2) for z in engine.zones}
    assert capacities["half"] == (200, pytest.approx(100))     # 100 m2 at 2 people per m2
    assert capacities["tight"][0] == 50
    assert capacities["given"] == (60, 30)


def test_zone_without_capacity_or_area_is_rejected():
    cfg = AppConfig.model_validate({
        "zones": [{"id": "z", "name": "Z", "polygon": [(0, 0), (1, 0), (1, 1)]}],
        "storage": {"path": None},
    })
    with pytest.raises(ConfigError, match="needs a capacity"):
        Engine(cfg, (100, 100))


def test_speed_in_metres_per_second_with_calibration():
    cfg = AppConfig.model_validate({
        "processing": {"anchor": "center"},
        "calibration": {"image_points": [(0, 0), (1, 0), (1, 1), (0, 1)],
                        "floor_points": [(0, 0), (40, 0), (40, 20), (0, 20)]},
        "zones": [{"id": "all", "name": "All", "capacity": 10,
                   "polygon": [(0, 0), (1, 0), (1, 1), (0, 1)]}],
        "storage": {"path": None},
    })
    engine = Engine(cfg, (400, 200))            # 10 pixels per metre
    for i in range(60):
        snap = step(engine, i * 0.1, [(50 + 1.2 * i, 100)])     # 12 px/s = 1.2 m/s
    assert snap["zones"][0]["speed_mps"] == pytest.approx(1.2, abs=0.15)
    assert snap["zones"][0]["area_m2"] == pytest.approx(800)


def test_replacing_zones_keeps_unchanged_ones_and_closes_orphan_alerts(one_zone_cfg):
    engine = Engine(one_zone_cfg, (100, 100))
    crowd = [(8 + 12 * (i % 3), 10 + 14 * (i // 3)) for i in range(6)]
    for i in range(80):
        step(engine, i * 0.1, crowd)
    kept = engine.zones[0]
    new = ZoneConfig(id="right", name="Right half", capacity=3,
                     polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)])
    engine.set_zones([one_zone_cfg.zones[0], new])
    assert engine.zones[0] is kept and len(engine.zones) == 2
    engine.set_zones([new])
    assert engine.alerts.active == []
    assert engine.alerts.history[0].outcome == "zone removed"


def test_history_is_recorded_once_a_second(one_zone_cfg):
    engine = Engine(one_zone_cfg, (100, 100))
    for i in range(125):
        step(engine, i * 0.1, [(10, 10)])
    rows = engine.history("left", 600)
    assert 9 <= len(rows) <= 11
    assert engine.history("nope", 600) == []
