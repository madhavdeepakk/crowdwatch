import numpy as np
import pytest

from crowdwatch.analytics.early_warning import DEFAULT_MODEL, FEATURES, load_model
from crowdwatch.config import AlertConfig
from crowdwatch.eval.harness import WarningScore, intervals, replay_predictions, true_events
from crowdwatch.eval.train import label
from crowdwatch.eval.harness import ZoneLog
from crowdwatch.sources.simulator import random_scenario


def test_random_scenarios_are_repeatable_and_varied():
    a, b = random_scenario(5), random_scenario(5)
    assert a.title == b.title and a.duration_s == b.duration_s and len(a.phases) == len(b.phases)
    kinds = {random_scenario(s).title.split("(")[1] for s in range(40)}
    assert len(kinds) >= 4          # quiet, surge, gradual, wave, mixed
    assert all(480 <= random_scenario(s).duration_s <= 660 for s in range(10))


def test_true_events_need_ten_seconds_and_bridge_short_dips():
    t = np.arange(0, 60, 0.1)
    truth = np.full(len(t), 5)
    truth[(t >= 5) & (t < 9)] = 12              # four seconds: a blip, not an event
    truth[(t >= 20) & (t < 28)] = 12
    truth[(t >= 30) & (t < 40)] = 12            # joined to the stretch before it (2 s dip)
    events = true_events(truth, 10, t)
    assert len(events) == 1
    assert events[0][0] == pytest.approx(20, abs=0.11) and events[0][1] == pytest.approx(40, abs=0.11)
    assert len(intervals(truth >= 10, t)) == 3


def test_replayed_warnings_respect_confirmation_time():
    alerts = AlertConfig(forecast_confirm_s=4, forecast_clear_s=30)
    t = np.arange(0, 100, 0.1)
    level = np.zeros(len(t), dtype=int)
    flicker = (t >= 10) & (t < 12)               # two seconds: too short to confirm
    steady = (t >= 40) & (t < 60)
    assert replay_predictions(alerts, t, level, flicker) == []
    opened = replay_predictions(alerts, t, level, steady)
    assert len(opened) == 1 and opened[0] == pytest.approx(44, abs=0.2)


def test_warning_score_counts_leads_and_false_alerts():
    score = WarningScore()
    score.add(opened=[50.0, 400.0], events=[(110.0, 170.0)])
    assert score.alerts == 2 and score.false == 1
    assert score.events_warned == 1 and score.leads == [60.0]


def test_labels_mark_the_two_minutes_before_an_event_and_skip_the_event_itself():
    t = np.arange(0, 400, 0.1)
    truth = np.where((t >= 250) & (t < 300), 12, 5)
    log = ZoneLog("z", 10, t, truth, np.zeros(len(t), dtype=int),
                  np.zeros((len(t), len(FEATURES)), np.float32), np.full(len(t), np.nan),
                  np.zeros(len(t), dtype=bool))
    rows = label(log, every=10)
    seconds_before = 250 - 130                  # usable seconds before the event: 30 s warm-up to 250 s
    assert rows.y.sum() == pytest.approx(120, abs=2)
    assert len(rows.y) == pytest.approx(seconds_before + 100 + 100, abs=4)


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="the early-warning model has not been trained")
def test_packaged_model_ranks_an_obvious_build_up_above_a_quiet_zone():
    model = load_model()
    assert model is not None and 0 < model.threshold < 1 and model.lead_s == 120

    def row(**values):
        return np.array([values.get(name, 0.0) for name in FEATURES], dtype=np.float32)

    quiet = row(ratio=0.3, churn=1.5, damped=0.3, speed=0.5, dwell=0.8, door_in=0.5, venue=2.0)
    filling = row(ratio=0.8, slope15=0.5, slope45=0.45, slope120=0.3, churn=2.5, approaching=0.25,
                  damped=1.2, speed=0.2, dwell=0.6, door_in=1.2, door_net=0.6, venue=2.5)
    p_quiet, p_filling = model.predict(quiet), model.predict(filling)
    assert p_quiet < 0.2 and p_filling > p_quiet + 0.4
    assert np.allclose(model.predict_many(np.stack([quiet, filling])), [p_quiet, p_filling], atol=1e-5)
