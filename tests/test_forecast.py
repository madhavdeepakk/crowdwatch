import pytest

from crowdwatch.analytics.forecast import TrendForecaster


def feed(forecaster, fn, seconds, dt=0.1):
    for i in range(int(seconds / dt)):
        forecaster.add(i * dt, fn(i * dt))


def test_no_trend_until_there_is_enough_history():
    f = TrendForecaster(window_s=45)
    feed(f, lambda t: 10, 3)
    assert f.trend() is None


def test_slope_and_level_of_a_steady_rise():
    f = TrendForecaster(window_s=45)
    feed(f, lambda t: 10 + 0.5 * t, 60)        # +30 people a minute
    trend = f.trend()
    assert trend.per_minute == pytest.approx(30, rel=0.02)
    assert trend.level == pytest.approx(10 + 0.5 * 59.5, abs=1.0)
    assert f.is_rising_clearly(trend)


def test_forecast_is_damped_not_a_straight_line():
    f = TrendForecaster(window_s=45, damping_s=90)
    feed(f, lambda t: 0.5 * t, 60)
    trend = f.trend()
    straight = trend.level + trend.slope * 120
    assert trend.level < trend.predict(120) < straight
    # in the limit the rise is capped at slope * damping
    assert trend.predict(1e6) == pytest.approx(trend.level + trend.slope * 90)


def test_eta_matches_the_damped_curve():
    f = TrendForecaster(window_s=45, damping_s=90)
    feed(f, lambda t: 0.5 * t, 60)
    trend = f.trend()
    target = trend.level + 20
    eta = trend.eta(target)
    assert eta is not None and trend.predict(eta) == pytest.approx(target, abs=1e-6)
    assert trend.eta(trend.level - 1) == 0.0
    assert trend.eta(trend.level + 1000) is None        # the trend fades before getting there


def test_flat_or_falling_counts_never_predict_a_fill():
    f = TrendForecaster(window_s=45)
    feed(f, lambda t: 40 - 0.2 * t, 60)
    trend = f.trend()
    assert trend.eta(60) is None
    assert not f.is_rising_clearly(trend)


def test_a_rise_must_beat_ordinary_coming_and_going():
    f = TrendForecaster(window_s=45)
    feed(f, lambda t: 20 + 0.1 * t, 60)        # +6 a minute
    trend = f.trend()
    assert f.is_rising_clearly(trend, churn_per_s=0.0)
    # with two people a second entering or leaving, +6 a minute is just noise
    assert not f.is_rising_clearly(trend, churn_per_s=2.0)
    assert f.rise_z(trend, 2.0) < f.rise_z(trend, 0.2)
