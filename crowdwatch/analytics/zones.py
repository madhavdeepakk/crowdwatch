"""Per-zone occupancy, density, dwell time, movement and early warning."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import AlertConfig, ZoneConfig
from .alerts import CRITICAL, LEVEL_NAMES, WARNING, LevelTracker, ZoneReading
from .early_warning import FEATURES, EarlyWarningModel
from .forecast import Trend, TrendForecaster

DENSITY_STALE_S = 10.0      # ignore a density estimate older than this


def ema_alpha(dt: float, tau: float) -> float:
    """Smoothing factor for an exponential moving average with time constant tau."""
    if dt <= 0:
        return 0.0
    return 1.0 - math.exp(-dt / max(tau, 1e-6))


@dataclass
class VenueContext:
    """Things that are the same for every zone at one instant."""

    people: int = 0
    door_in_per_min: float = 0.0
    door_out_per_min: float = 0.0


@dataclass
class ZoneMetrics:
    id: str
    name: str
    count: float
    raw: int
    capacity: int
    ratio: float
    level: int
    area_m2: Optional[float]
    density: Optional[float]
    trend_per_min: float
    forecast_count: Optional[float]
    eta_s: Optional[float]
    speed_mps: Optional[float]
    dwell_s: Optional[float]
    congested: bool
    risk: int
    approaching: float = 0.0
    leaving: float = 0.0
    predicted: bool = False                 # the early warning in use says "about to fill"
    probability: Optional[float] = None     # the learned model's estimate, when it is in use
    rule_predicted: bool = False            # what the damped-trend rule alone would say
    density_count: Optional[float] = None   # the density map's estimate, when that model is on
    method: str = "tracking"                # which count is in charge: "tracking" or "density"
    features: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        def r(value, digits=2):
            return None if value is None else round(float(value), digits)

        return {
            "id": self.id, "name": self.name,
            "count": int(round(self.count)), "count_smooth": r(self.count), "raw": self.raw,
            "capacity": self.capacity, "ratio": r(self.ratio, 3),
            "level": self.level, "level_name": LEVEL_NAMES[self.level],
            "area_m2": r(self.area_m2, 1), "density": r(self.density),
            "trend_per_min": r(self.trend_per_min, 1),
            "forecast_count": r(self.forecast_count, 1), "eta_s": r(self.eta_s, 0),
            "speed_mps": r(self.speed_mps), "dwell_s": r(self.dwell_s, 0),
            "congested": self.congested, "risk": self.risk,
            "approaching": int(round(self.approaching)), "leaving": int(round(self.leaving)),
            "predicted": self.predicted, "probability": r(self.probability),
            "density_count": r(self.density_count, 1), "method": self.method,
        }

    def reading(self) -> ZoneReading:
        return ZoneReading(
            zone_id=self.id, zone_name=self.name, count=self.count, capacity=self.capacity,
            ratio=self.ratio, level=self.level, eta_s=self.eta_s,
            trend_per_min=self.trend_per_min, congested=self.congested,
            predicted=self.predicted, probability=self.probability, approaching=self.approaching,
        )


class ZoneState:
    """Everything remembered about one zone between frames."""

    DWELL_GRACE_S = 1.5     # a person may flicker out of a zone this long and keep their clock
    SETTLE_S = 2.0          # the tracker needs a moment before its first counts mean anything
    MODEL_WARMUP_S = 30.0   # the learned model needs some history before it is consulted
    MODEL_EVERY_S = 0.5
    HISTORY_S = 3600

    def __init__(self, cfg: ZoneConfig, polygon_px: np.ndarray, capacity: int,
                 area_m2: Optional[float], alerts: AlertConfig,
                 model: Optional[EarlyWarningModel] = None):
        self.cfg = cfg
        self.polygon_px = polygon_px
        self.capacity = capacity
        self.area_m2 = area_m2
        self.alerts = alerts
        self.model = model if alerts.forecast_method == "learned" else None
        self.levels = LevelTracker(alerts)
        self.forecaster = TrendForecaster(alerts.forecast_window_s, damping_s=alerts.forecast_damping_s)
        self.count: Optional[float] = None
        self.speed: Optional[float] = None
        self.approaching = 0.0
        self.leaving = 0.0
        self._entered: dict[int, float] = {}
        self._last_inside: dict[int, float] = {}
        self._churn: deque[float] = deque()      # times at which someone entered or left
        self.history: deque[tuple[float, float, float, Optional[float]]] = deque(maxlen=self.HISTORY_S)
        self._last_history_t = -1e9
        self._first_t: Optional[float] = None
        self._probability: Optional[float] = None
        self._last_model_t = -1e9
        self._feature_cache: Optional[np.ndarray] = None
        self._last_feature_t = -1e9
        self.density_count: Optional[float] = None
        self._density_t = -1e9
        self.fusion: tuple[float, float] = (15.0, 1.2)      # min people, switch ratio

    def set_density(self, t: float, count: float) -> None:
        """Take a fresh estimate from the density-map model."""
        self.density_count = count if self.density_count is None else 0.5 * (self.density_count + count)
        self._density_t = t

    def update(self, t: float, dt: float, ids: np.ndarray, speeds: Optional[np.ndarray],
               approaching: int = 0, leaving: int = 0,
               venue: Optional[VenueContext] = None) -> ZoneMetrics:
        a = self.alerts
        venue = venue or VenueContext()
        raw = int(len(ids))
        tracked = raw
        method = "tracking"
        density_count = self.density_count if t - self._density_t <= DENSITY_STALE_S else None
        if density_count is not None:
            from ..detect.density import fuse

            fused = fuse(raw, density_count, *self.fusion)
            if fused != raw:
                raw, method = int(round(fused)), "density"

        # Tracks take a few frames to be confirmed, so the very first counts
        # climb from zero. Do not mistake that for a crowd arriving.
        if self._first_t is None:
            self._first_t = t
        settling = t - self._first_t < self.SETTLE_S

        # Smoothed count: detections flicker, so average over a few seconds.
        if self.count is None or settling:
            self.count = float(raw)
        else:
            self.count += ema_alpha(dt, a.smoothing_s) * (raw - self.count)

        # People on their way in and on their way out, lightly smoothed.
        k = ema_alpha(dt, 3.0) if not settling else 1.0
        self.approaching += k * (approaching - self.approaching)
        self.leaving += k * (leaving - self.leaving)

        # Dwell time: how long the people here have been here.
        for pid in ids.tolist():
            if pid not in self._entered:
                self._entered[pid] = t
                self._churn.append(t)
            self._last_inside[pid] = t
        for pid in [p for p, seen in self._last_inside.items() if t - seen > self.DWELL_GRACE_S]:
            self._last_inside.pop(pid, None)
            self._entered.pop(pid, None)
            self._churn.append(t)
        while self._churn and t - self._churn[0] > a.forecast_window_s:
            self._churn.popleft()
        churn_per_s = len(self._churn) / a.forecast_window_s
        dwell = float(np.mean([t - self._entered[p] for p in ids.tolist()])) if tracked else None

        # Movement: the median walking speed of the people in the zone.
        if speeds is not None and tracked:
            median = float(np.median(speeds))
            self.speed = median if self.speed is None else self.speed + ema_alpha(dt, 2.0) * (median - self.speed)
        elif tracked == 0:
            self.speed = None

        if not settling:
            self.forecaster.add(t, raw)
        trend: Optional[Trend] = self.forecaster.trend()

        ratio = self.count / self.capacity
        level = self.levels.update(t, ratio)
        limit = self.capacity * a.critical_ratio

        # The rule: a damped straight line through the last 45 seconds.
        eta = forecast_count = None
        trend_per_min = 0.0
        rule_predicted = False
        if trend is not None:
            trend_per_min = trend.per_minute
            forecast_count = trend.predict(a.forecast_lead_s)
            eta = trend.eta(limit) if trend.slope > 0 else None
            rule_predicted = bool(
                level < CRITICAL
                and ratio >= a.forecast_min_ratio
                and self.forecaster.is_rising_clearly(trend, churn_per_s, a.forecast_min_z)
                and not self._streaming_through()
                and forecast_count >= limit
            )

        # The learned model, which also sees who is on their way.
        if self._feature_cache is None or t - self._last_feature_t >= self.MODEL_EVERY_S:
            self._feature_cache = self._features(ratio, trend, churn_per_s, forecast_count, dwell, venue)
            self._last_feature_t = t
        features = self._feature_cache
        predicted, probability = rule_predicted, None
        if self.model is not None and trend is not None and t - self._first_t >= self.MODEL_WARMUP_S:
            if t - self._last_model_t >= self.MODEL_EVERY_S:
                self._probability = self.model.predict(features)
                self._last_model_t = t
            probability = self._probability
            threshold = a.forecast_threshold or self.model.threshold
            # the model must be confident and the trend itself must reach capacity
            predicted = bool(level < CRITICAL and ratio >= a.forecast_min_ratio
                             and probability >= threshold and forecast_count >= limit)
        if not predicted or (eta is not None and eta > 2 * a.forecast_lead_s):
            eta = None

        # Standing still is normal in a food court and alarming in a corridor,
        # so only zones marked as walkways can be "congested".
        congested = bool(
            self.cfg.walkway and level >= WARNING
            and self.speed is not None and self.speed < a.stall_speed_mps
        )
        density = self.count / self.area_m2 if self.area_m2 else None
        risk = self._risk(ratio, forecast_count, congested)

        if not settling and t - self._last_history_t >= 1.0:
            self.history.append((round(t, 2), round(self.count, 2), round(ratio, 4),
                                 None if forecast_count is None else round(forecast_count, 1)))
            self._last_history_t = t

        return ZoneMetrics(
            id=self.cfg.id, name=self.cfg.name, count=self.count, raw=tracked,
            density_count=density_count, method=method,
            capacity=self.capacity, ratio=ratio, level=level, area_m2=self.area_m2,
            density=density, trend_per_min=trend_per_min, forecast_count=forecast_count,
            eta_s=eta, speed_mps=self.speed, dwell_s=dwell, congested=congested, risk=risk,
            approaching=self.approaching, leaving=self.leaving, predicted=predicted,
            probability=probability, rule_predicted=rule_predicted, features=features,
        )

    def _features(self, ratio: float, trend: Optional[Trend], churn_per_s: float,
                  forecast_count: Optional[float], dwell: Optional[float],
                  venue: VenueContext) -> np.ndarray:
        """The model's inputs, in the order given by ``early_warning.FEATURES``."""
        c = float(self.capacity)

        def slope(window: float) -> float:
            fit = self.forecaster.trend(window)
            return 0.0 if fit is None else fit.per_minute / c

        values = {
            "ratio": ratio,
            "slope15": slope(15.0),
            "slope45": 0.0 if trend is None else trend.per_minute / c,
            "slope120": slope(120.0),
            "churn": churn_per_s * 60.0 / c,
            "approaching": self.approaching / c,
            "leaving": self.leaving / c,
            "damped": ratio if forecast_count is None else forecast_count / c,
            "speed": 0.0 if self.speed is None else min(self.speed, 3.0),
            "dwell": 0.0 if dwell is None else min(dwell / 60.0, 10.0),
            "door_in": venue.door_in_per_min / c,
            "door_net": (venue.door_in_per_min - venue.door_out_per_min) / c,
            "venue": venue.people / c,
        }
        return np.array([values[name] for name in FEATURES], dtype=np.float32)

    def _streaming_through(self) -> bool:
        limit = self.alerts.forecast_max_speed_mps
        return limit is not None and self.speed is not None and self.speed > limit

    def _risk(self, ratio: float, forecast_count: Optional[float], congested: bool) -> int:
        """A 0-100 score: mostly how full the zone is, partly where it is heading.

        risk = 65% current occupancy + 25% forecast occupancy + 10% stalled crowd,
        with occupancy capped at 125% of capacity.
        """
        cap = 1.25
        now = min(ratio, cap) / cap
        ahead = now
        if forecast_count is not None:
            ahead = min(max(forecast_count / self.capacity, 0.0), cap) / cap
        score = 0.65 * now + 0.25 * ahead + 0.10 * (1.0 if congested else 0.0)
        return int(round(100 * min(score, 1.0)))
