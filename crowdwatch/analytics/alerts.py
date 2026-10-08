"""Alert levels with hysteresis, and the alert log built on top of them.

A raw threshold on a noisy count chatters: 59, 61, 60, 62 around a limit of 60
would fire and clear an alarm four times in a second, and operators learn to
ignore alarms that do that. Two guards keep the alert state calm:

* a zone must stay above a threshold for ``raise_after_s`` before its level
  goes up, and stay below it (by a margin) for ``clear_after_s`` before it
  comes down;
* one alert is opened per episode and then escalated, downgraded or resolved,
  instead of a fresh alert on every change.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from ..config import AlertConfig

NORMAL, BUSY, WARNING, CRITICAL = 0, 1, 2, 3
LEVEL_NAMES = {NORMAL: "normal", BUSY: "busy", WARNING: "warning", CRITICAL: "critical"}


class LevelTracker:
    """Turns a noisy occupancy ratio into a steady level for one zone."""

    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self.level = NORMAL
        self._pending: Optional[int] = None
        self._pending_since = 0.0

    def _instant_level(self, ratio: float) -> int:
        c = self.cfg
        thresholds = (c.busy_ratio, c.warning_ratio, c.critical_ratio)
        target = sum(ratio >= th for th in thresholds)
        if target < self.level:
            # only step down once clearly below the threshold we are sitting on
            while target < self.level and ratio >= thresholds[target] - c.hysteresis:
                target += 1
        return target

    def update(self, t: float, ratio: float) -> int:
        target = self._instant_level(ratio)
        if target == self.level:
            self._pending = None
            return self.level
        rising = target > self.level
        if self._pending is None or (self._pending > self.level) != rising:
            self._pending, self._pending_since = target, t
        else:
            # hold the most conservative level seen while waiting
            self._pending = min(self._pending, target) if rising else max(self._pending, target)
        wait = self.cfg.raise_after_s if rising else self.cfg.clear_after_s
        if t - self._pending_since >= wait:
            self.level = self._pending
            self._pending = None
        return self.level


@dataclass
class Alert:
    id: int
    zone_id: str
    zone_name: str
    kind: str                  # "overcrowding" or "predicted"
    level: str                 # "warning", "critical" or "predicted"
    message: str
    started_t: float           # source time, seconds
    started_wall: float        # wall clock, epoch seconds
    peak_count: float = 0.0
    peak_ratio: float = 0.0
    ended_t: Optional[float] = None
    ended_wall: Optional[float] = None
    acknowledged: bool = False
    outcome: Optional[str] = None   # how it ended, for the log

    @property
    def active(self) -> bool:
        return self.ended_t is None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["active"] = self.active
        return data


@dataclass
class ZoneReading:
    """What the alert logic needs to know about one zone at one instant."""

    zone_id: str
    zone_name: str
    count: float
    capacity: int
    ratio: float
    level: int
    eta_s: Optional[float] = None       # estimated seconds until capacity, when one can be given
    trend_per_min: float = 0.0
    congested: bool = False
    predicted: bool = False             # the early warning says this zone is about to fill
    probability: Optional[float] = None
    approaching: float = 0.0            # people walking towards the zone


def _fmt_eta(seconds: float) -> str:
    if seconds < 15:
        return "a few seconds"
    if seconds < 50:
        return f"about {max(10, int(round(seconds / 10) * 10))} seconds"
    minutes = seconds / 60
    if minutes < 1.75:
        return "about 1 minute" if minutes < 1.25 else "about 1.5 minutes"
    return f"about {int(round(minutes))} minutes"


def describe(reading: ZoneReading, kind: str) -> str:
    n, cap = int(round(reading.count)), reading.capacity
    pct = int(round(reading.ratio * 100))
    if kind == "predicted":
        if n >= cap:
            return f"{reading.zone_name} is filling up and has just reached capacity: {n} of {cap}."
        text = f"{reading.zone_name} is filling up: {n} of {cap} now"
        if reading.trend_per_min >= 1:
            text += f", rising by {reading.trend_per_min:.0f} a minute"
        heading = int(round(reading.approaching))
        if heading >= 3:
            text += f", {heading} more heading this way"
        if reading.eta_s:
            return f"{text}. Full in {_fmt_eta(reading.eta_s)}."
        return f"{text}. Likely to fill within 2 minutes."

    if reading.level >= CRITICAL:
        text = f"{reading.zone_name} is over capacity: {n} of {cap} ({pct}%)."
    else:
        text = f"{reading.zone_name} is nearly full: {n} of {cap} ({pct}%)."
    if reading.congested:
        text += " The crowd has almost stopped moving."
    elif reading.trend_per_min >= 3:
        text += f" Still rising by {reading.trend_per_min:.0f} a minute."
    return text


@dataclass
class AlertEvent:
    type: str      # opened, escalated, downgraded, resolved
    alert: Alert


class AlertManager:
    """Keeps at most one overcrowding alert and one early warning per zone."""

    def __init__(self, cfg: AlertConfig, next_id: int = 1,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self._next_id = next_id
        self._clock = clock
        self._open: dict[tuple[str, str], Alert] = {}
        self._predict_since: dict[str, Optional[float]] = {}
        self._predict_clear_since: dict[str, Optional[float]] = {}
        self.history: list[Alert] = []
        self._listeners: list[Callable[[AlertEvent], None]] = []

    def subscribe(self, listener: Callable[[AlertEvent], None]) -> None:
        self._listeners.append(listener)

    @property
    def active(self) -> list[Alert]:
        return list(self._open.values())

    def find(self, alert_id: int) -> Optional[Alert]:
        return next((a for a in self.history if a.id == alert_id), None)

    def acknowledge(self, alert_id: int) -> Optional[Alert]:
        alert = self.find(alert_id)
        if alert is not None:
            alert.acknowledged = True
        return alert

    # ------------------------------------------------------------------

    def update(self, t: float, reading: ZoneReading) -> list[AlertEvent]:
        events: list[AlertEvent] = []
        events += self._update_overcrowding(t, reading)
        events += self._update_prediction(t, reading)
        self._notify(events)
        return events

    def _notify(self, events: list[AlertEvent]) -> None:
        for event in events:
            for listener in self._listeners:
                listener(event)

    def close_all(self, t: float, outcome: str = "monitoring stopped") -> None:
        self._notify([AlertEvent("resolved", self._close(key, t, outcome)) for key in list(self._open)])

    def close_missing(self, zone_ids: set[str], t: float) -> None:
        """Close alerts that belong to zones which no longer exist."""
        self._notify([AlertEvent("resolved", self._close(key, t, "zone removed"))
                      for key in list(self._open) if key[0] not in zone_ids])

    # ------------------------------------------------------------------

    def _open_alert(self, t: float, reading: ZoneReading, kind: str, level: str) -> Alert:
        alert = Alert(
            id=self._next_id, zone_id=reading.zone_id, zone_name=reading.zone_name,
            kind=kind, level=level, message=describe(reading, kind),
            started_t=t, started_wall=self._clock(),
            peak_count=reading.count, peak_ratio=reading.ratio,
        )
        self._next_id += 1
        self._open[(reading.zone_id, kind)] = alert
        self.history.append(alert)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        return alert

    def _close(self, key: tuple[str, str], t: float, outcome: str) -> Alert:
        alert = self._open.pop(key)
        alert.ended_t, alert.ended_wall, alert.outcome = t, self._clock(), outcome
        return alert

    def _update_overcrowding(self, t: float, r: ZoneReading) -> list[AlertEvent]:
        key = (r.zone_id, "overcrowding")
        alert = self._open.get(key)
        wanted = LEVEL_NAMES[r.level] if r.level >= WARNING else None
        if alert is None:
            if wanted is None:
                return []
            return [AlertEvent("opened", self._open_alert(t, r, "overcrowding", wanted))]
        if r.count > alert.peak_count:
            alert.peak_count, alert.peak_ratio = r.count, r.ratio
        if wanted is None:
            self._close(key, t, "back to a safe level")
            return [AlertEvent("resolved", alert)]
        alert.message = describe(r, "overcrowding")
        if wanted != alert.level:
            kind = "escalated" if wanted == "critical" else "downgraded"
            alert.level = wanted
            if kind == "escalated":
                alert.acknowledged = False      # a worse situation needs a fresh look
            return [AlertEvent(kind, alert)]
        return []

    def _update_prediction(self, t: float, r: ZoneReading) -> list[AlertEvent]:
        key = (r.zone_id, "predicted")
        alert = self._open.get(key)
        rising = r.predicted and r.level < CRITICAL
        if alert is None:
            if not rising:
                self._predict_since[r.zone_id] = None
                return []
            since = self._predict_since.get(r.zone_id)
            if since is None:
                self._predict_since[r.zone_id] = since = t
            if t - since < self.cfg.forecast_confirm_s:
                return []
            self._predict_since[r.zone_id] = None
            self._predict_clear_since[r.zone_id] = None
            return [AlertEvent("opened", self._open_alert(t, r, "predicted", "predicted"))]

        if r.count > alert.peak_count:
            alert.peak_count, alert.peak_ratio = r.count, r.ratio
        if r.level >= CRITICAL:
            self._close(key, t, "the zone reached capacity")
            return [AlertEvent("resolved", alert)]
        if rising:
            alert.message = describe(r, "predicted")
            self._predict_clear_since[r.zone_id] = None
            return []
        since = self._predict_clear_since.get(r.zone_id)
        if since is None:
            self._predict_clear_since[r.zone_id] = since = t
        if t - since >= self.cfg.forecast_clear_s:
            self._close(key, t, "the rise levelled off")
            return [AlertEvent("resolved", alert)]
        return []
