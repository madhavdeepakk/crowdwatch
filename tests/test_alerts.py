from crowdwatch.analytics.alerts import (BUSY, CRITICAL, NORMAL, WARNING, AlertManager,
                                         LevelTracker, ZoneReading, describe)
from crowdwatch.config import AlertConfig


def reading(count, level, capacity=50, eta=None, trend=0.0, congested=False, approaching=0.0):
    return ZoneReading("z", "Atrium", count, capacity, count / capacity, level,
                       eta_s=eta, trend_per_min=trend, congested=congested,
                       predicted=eta is not None, approaching=approaching)


def drive(tracker, ratios, dt=0.1, t0=0.0):
    levels = []
    for i, r in enumerate(ratios):
        levels.append(tracker.update(t0 + i * dt, r))
    return levels


def test_level_rises_only_after_the_hold_time():
    tracker = LevelTracker(AlertConfig(raise_after_s=3))
    levels = drive(tracker, [1.05] * 50)
    assert levels[0] == NORMAL and levels[25] == NORMAL
    assert levels[-1] == CRITICAL
    assert levels.index(CRITICAL) == 30          # exactly three seconds in


def test_a_brief_spike_is_ignored():
    tracker = LevelTracker(AlertConfig(raise_after_s=3))
    levels = drive(tracker, [0.3] * 10 + [1.2] * 20 + [0.3] * 30)
    assert set(levels) == {NORMAL}


def test_count_hovering_on_the_limit_does_not_chatter():
    tracker = LevelTracker(AlertConfig())
    drive(tracker, [1.05] * 60)
    assert tracker.level == CRITICAL
    wobble = [0.97, 1.02, 0.98, 1.01, 0.96, 1.03] * 60       # 36 s either side of 100%
    levels = drive(tracker, wobble, t0=6.0)
    assert set(levels) == {CRITICAL}


def test_level_falls_after_the_clear_time_and_margin():
    cfg = AlertConfig(clear_after_s=15, hysteresis=0.05)
    tracker = LevelTracker(cfg)
    drive(tracker, [1.1] * 60)
    levels = drive(tracker, [0.85] * 200, t0=6.0)
    assert levels[100] == CRITICAL            # ten seconds in: still held
    assert levels[-1] == WARNING


def test_levels_step_through_busy_and_warning():
    tracker = LevelTracker(AlertConfig(raise_after_s=1))
    assert drive(tracker, [0.65] * 20)[-1] == BUSY
    assert drive(tracker, [0.85] * 20, t0=2)[-1] == WARNING


def test_one_alert_per_episode_with_escalation_and_resolution():
    manager = AlertManager(AlertConfig(), clock=lambda: 1000.0)
    seen = []
    manager.subscribe(lambda e: seen.append(e.type))
    manager.update(0, reading(20, NORMAL))
    manager.update(1, reading(42, WARNING))
    manager.update(2, reading(44, WARNING))
    manager.update(3, reading(52, CRITICAL))
    manager.update(4, reading(55, CRITICAL))
    manager.update(5, reading(43, WARNING))
    manager.update(6, reading(20, NORMAL))
    assert seen == ["opened", "escalated", "downgraded", "resolved"]
    assert len(manager.history) == 1
    alert = manager.history[0]
    assert alert.peak_count == 55 and not alert.active
    assert alert.outcome == "back to a safe level"


def test_escalation_reopens_an_acknowledged_alert():
    manager = AlertManager(AlertConfig())
    manager.update(0, reading(42, WARNING))
    alert = manager.acknowledge(manager.active[0].id)
    assert alert.acknowledged
    manager.update(1, reading(52, CRITICAL))
    assert not manager.active[0].acknowledged
    assert manager.acknowledge(999) is None


def test_prediction_needs_confirmation_and_ends_when_capacity_is_reached():
    cfg = AlertConfig(forecast_confirm_s=4)
    manager = AlertManager(cfg)
    assert manager.update(0, reading(30, BUSY, eta=60, trend=12)) == []
    assert manager.update(2, reading(31, BUSY, eta=58, trend=12)) == []
    events = manager.update(4.1, reading(33, BUSY, eta=50, trend=12))
    assert [e.type for e in events] == ["opened"] and events[0].alert.kind == "predicted"
    events = manager.update(30, reading(51, CRITICAL, eta=None))
    kinds = {(e.alert.kind, e.type) for e in events}
    assert ("overcrowding", "opened") in kinds and ("predicted", "resolved") in kinds
    predicted = next(a for a in manager.history if a.kind == "predicted")
    assert predicted.outcome == "the zone reached capacity"


def test_prediction_that_fizzles_closes_quietly():
    cfg = AlertConfig(forecast_confirm_s=0, forecast_clear_s=30)
    manager = AlertManager(cfg)
    manager.update(0, reading(30, BUSY, eta=60, trend=12))
    assert len(manager.active) == 1
    manager.update(10, reading(30, BUSY, eta=None))
    assert len(manager.active) == 1               # give it time
    manager.update(41, reading(29, BUSY, eta=None))
    assert manager.active == []
    assert manager.history[0].outcome == "the rise levelled off"


def test_messages_read_like_sentences():
    text = describe(reading(52, CRITICAL, trend=8), "overcrowding")
    assert text == "Atrium is over capacity: 52 of 50 (104%). Still rising by 8 a minute."
    text = describe(reading(42, WARNING, congested=True), "overcrowding")
    assert text.endswith("The crowd has almost stopped moving.")
    text = describe(reading(30, BUSY, eta=95, trend=12, approaching=9), "predicted")
    assert text == ("Atrium is filling up: 30 of 50 now, rising by 12 a minute, "
                    "9 more heading this way. Full in about 1.5 minutes.")
    reading_without_eta = reading(30, BUSY, eta=95, trend=0.2)
    reading_without_eta.eta_s = None
    assert describe(reading_without_eta, "predicted") == (
        "Atrium is filling up: 30 of 50 now. Likely to fill within 2 minutes.")


def test_closing_everything_notifies_listeners():
    manager = AlertManager(AlertConfig())
    manager.update(0, reading(52, CRITICAL))
    seen = []
    manager.subscribe(lambda e: seen.append((e.type, e.alert.outcome)))
    manager.close_all(5, "scenario restarted")
    assert seen == [("resolved", "scenario restarted")] and manager.active == []
