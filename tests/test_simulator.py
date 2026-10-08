import numpy as np
import pytest

from crowdwatch.detect.sim import SimDetector
from crowdwatch.engine import Engine
from crowdwatch.presets import demo_config
from crowdwatch.sources.simulator import FRAME_H, FRAME_W, SCENARIOS, ZONES_M, SimulatorSource


def run(scenario, seed, seconds, render=False):
    cfg = demo_config(scenario, seed, storage_path=None)
    src = SimulatorSource(scenario, seed=seed, render=render)
    det = SimDetector(src.size, seed)
    engine = Engine(cfg, src.size)
    log, snap, errors = [], None, []
    engine.alerts.subscribe(lambda e: log.append((engine.t, e.type, e.alert.kind, e.alert.level)))
    while src.world.t < seconds:
        frame = src.read()
        dets = det.detect(frame)
        snap = engine.step(frame.t, dets.boxes, dets.scores)
        truth = src.true_counts()
        errors.append([z["count_smooth"] - truth[z["id"]] for z in snap["zones"]])
    return snap, log, np.abs(np.asarray(errors)).mean(), src


def test_same_seed_gives_the_same_crowd():
    a = SimulatorSource("surge", seed=3, render=False)
    b = SimulatorSource("surge", seed=3, render=False)
    for _ in range(100):
        fa, fb = a.read(), b.read()
    assert np.array_equal(fa.truth.positions, fb.truth.positions)
    c = SimulatorSource("surge", seed=4, render=False)
    for _ in range(100):
        fc = c.read()
    assert fc.truth.positions.shape != fa.truth.positions.shape or not np.allclose(
        fc.truth.positions, fa.truth.positions)


def test_frames_are_rendered_at_the_declared_size():
    src = SimulatorSource("surge", seed=1, render=True)
    frame = src.read()
    assert frame.image.shape == (FRAME_H, FRAME_W, 3)
    assert src.size == (FRAME_W, FRAME_H)
    assert (frame.truth.positions[:, 0] <= FRAME_W).all()


def test_unknown_scenario_is_rejected_with_the_choices():
    with pytest.raises(ValueError, match="surge"):
        SimulatorSource("stampede")


def test_scenario_ends_and_every_scenario_is_listed():
    assert set(SCENARIOS) == {"surge", "gradual", "busy", "wave"}
    src = SimulatorSource("wave", seed=1, render=False)
    src.world.t = SCENARIOS["wave"].duration_s
    assert src.read() is None and src.finished


def test_counts_track_ground_truth_closely():
    _, _, mean_abs_error, _ = run("surge", 11, 120)
    assert mean_abs_error < 1.5          # people, averaged over all zones and frames


def test_surge_is_predicted_then_flagged_critical():
    snap, log, _, src = run("surge", 11, 300)
    atrium = next(z for z in snap["zones"] if z["id"] == "atrium")
    assert src.true_counts()["atrium"] >= ZONES_M["atrium"][2]
    assert atrium["level_name"] == "critical"
    critical_at = next(t for t, kind, what, level in log if what == "overcrowding" and level == "critical")
    predicted_at = next(t for t, kind, what, _ in log if what == "predicted" and kind == "opened")
    assert predicted_at < critical_at - 20       # warned well before it happened
    assert 90 < predicted_at                     # and not before the surge began


def test_quiet_period_raises_nothing():
    _, log, _, _ = run("surge", 11, 85)          # the surge starts at 90 s
    assert log == []
