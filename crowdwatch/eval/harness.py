"""Measure the system against the simulator's ground truth.

The question is not "does it raise an alarm when a zone is full", which any
threshold does. It is: how early does it warn, how often does it cry wolf, and
how much does it chatter, compared with the obvious baseline of counting the
detections in each frame and comparing with the limit.

Definitions used throughout:

* An *overcrowding event* is a stretch where the true number of people in a
  zone is at or above its capacity for at least ``MIN_EVENT_S`` seconds
  (dips shorter than ``MERGE_GAP_S`` do not split an event).
* The *naive baseline* raises its alarm on every frame in which the number of
  confident detections inside the zone is at or above capacity.
* The *smoothed baseline* does the same on a two-second moving average of
  that number, which is the first fix most people reach for.
* An alarm is *false* if it does not overlap any event (with a small tolerance).
* An early warning is *false* if no event starts within ``LOOKBACK_S`` of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from ..analytics.alerts import AlertManager, ZoneReading
from ..config import AlertConfig, AppConfig
from ..detect.sim import SimDetector
from ..engine import Engine
from ..geometry import points_in_polygon
from ..presets import demo_config
from ..sources.simulator import Scenario, SimulatorSource

MIN_EVENT_S = 10.0
MERGE_GAP_S = 5.0
TOLERANCE_S = 5.0
LOOKBACK_S = 240.0       # an early warning counts for an event that starts within this long


def intervals(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    """Runs of True in ``mask`` as (start, end) times."""
    out, start = [], None
    for flag, ti in zip(mask.tolist(), t.tolist()):
        if flag and start is None:
            start = ti
        elif not flag and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, float(t[-1])))
    return out


def true_events(truth: np.ndarray, capacity: int, t: np.ndarray) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in intervals(truth >= capacity, t):
        if merged and start - merged[-1][1] < MERGE_GAP_S:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= MIN_EVENT_S]


def overlaps(a: tuple[float, float], b: tuple[float, float], tol: float = TOLERANCE_S) -> bool:
    return a[0] <= b[1] + tol and b[0] <= a[1] + tol


def replay_predictions(alerts: AlertConfig, t: np.ndarray, level: np.ndarray,
                       predicted: np.ndarray) -> list[float]:
    """When would early-warning alerts open, given a per-frame yes/no signal?

    Runs the real alert logic (confirmation and clearing times included), so
    any signal can be compared on equal terms without re-running the video.
    """
    manager = AlertManager(alerts)
    opened = []
    for ti, lv, flag in zip(t.tolist(), level.tolist(), predicted.tolist()):
        reading = ZoneReading("z", "z", 0.0, 1, 0.0, int(lv), predicted=bool(flag))
        for event in manager._update_prediction(ti, reading):
            if event.type == "opened":
                opened.append(ti)
    return opened


@dataclass
class WarningScore:
    alerts: int = 0
    false: int = 0
    events_warned: int = 0
    leads: list[float] = field(default_factory=list)

    def add(self, opened: list[float], events: list[tuple[float, float]]) -> None:
        self.alerts += len(opened)
        for p in opened:
            if not any(-TOLERANCE_S <= ev[0] - p <= LOOKBACK_S or ev[0] <= p <= ev[1] for ev in events):
                self.false += 1
        for ev in events:
            before = [p for p in opened if 0 <= ev[0] - p <= LOOKBACK_S]
            if before:
                self.events_warned += 1
                self.leads.append(ev[0] - min(before))

    def merge(self, other: "WarningScore") -> None:
        self.alerts += other.alerts
        self.false += other.false
        self.events_warned += other.events_warned
        self.leads += other.leads


@dataclass
class ZoneLog:
    """Per-frame record of one zone, kept for training and for replays."""

    zone_id: str
    capacity: int
    t: np.ndarray
    truth: np.ndarray
    level: np.ndarray
    features: np.ndarray        # (frames, n_features)
    probability: np.ndarray     # NaN where the model was not consulted
    rule_predicted: np.ndarray


@dataclass
class RunResult:
    scenario: str
    seed: int
    duration_s: float
    events: int = 0
    naive_detected: int = 0
    naive_alarms: int = 0
    naive_false: int = 0
    naive_delays: list[float] = field(default_factory=list)
    smooth_detected: int = 0
    smooth_alarms: int = 0
    smooth_false: int = 0
    smooth_delays: list[float] = field(default_factory=list)
    ours_detected: int = 0
    ours_alarms: int = 0
    ours_false: int = 0
    ours_delays: list[float] = field(default_factory=list)
    # early warning: as run, the rule on its own, and the rule or the "nearly full" level
    warning: WarningScore = field(default_factory=WarningScore)
    warning_rule: WarningScore = field(default_factory=WarningScore)
    warning_any: WarningScore = field(default_factory=WarningScore)
    abs_err_naive: float = 0.0
    abs_err_ours: float = 0.0
    err_samples: int = 0
    door_true: int = 0
    door_counted: int = 0
    logs: Optional[list[ZoneLog]] = field(default=None, repr=False)


def run_one(scenario: Union[str, Scenario], seed: int, cfg: Optional[AppConfig] = None,
            capacity_scale: Optional[dict[str, float]] = None, keep_logs: bool = False) -> RunResult:
    name = scenario.name if isinstance(scenario, Scenario) else scenario
    cfg = cfg or demo_config("surge", seed, storage_path=None)
    if capacity_scale:
        for zone in cfg.zones:
            zone.capacity = max(5, int(round(zone.capacity * capacity_scale.get(zone.id, 1.0))))
    src = SimulatorSource(scenario, seed=seed, fps=cfg.processing.fps, render=False)
    detector = SimDetector(src.size, seed)
    engine = Engine(cfg, src.size)
    zone_ids = [z.cfg.id for z in engine.zones]
    caps = {z.cfg.id: z.capacity for z in engine.zones}
    polys = {z.cfg.id: z.polygon_px for z in engine.zones}

    alert_log: list[tuple] = []
    engine.alerts.subscribe(
        lambda e: alert_log.append((engine.t, e.alert.zone_id, e.alert.kind, e.type, e.alert.level))
    )

    times: list[float] = []
    truth = {z: [] for z in zone_ids}
    naive = {z: [] for z in zone_ids}
    ours = {z: [] for z in zone_ids}
    level = {z: [] for z in zone_ids}
    feats = {z: [] for z in zone_ids}
    prob = {z: [] for z in zone_ids}
    rule = {z: [] for z in zone_ids}
    while True:
        frame = src.read()
        if frame is None:
            break
        dets = detector.detect(frame)
        engine.step(frame.t, dets.boxes, dets.scores)
        times.append(frame.t)
        true_now = src.true_counts()
        confident = dets.boxes[dets.scores >= 0.5]
        centres = (confident[:, :2] + confident[:, 2:]) / 2
        for m in engine.last_metrics:
            truth[m.id].append(true_now[m.id])
            naive[m.id].append(int(points_in_polygon(centres, polys[m.id]).sum()))
            ours[m.id].append(m.count)
            level[m.id].append(m.level)
            feats[m.id].append(m.features)
            prob[m.id].append(np.nan if m.probability is None else m.probability)
            rule[m.id].append(m.rule_predicted)

    t = np.asarray(times)
    result = RunResult(name, seed, float(t[-1]))
    if keep_logs:
        result.logs = []
    for zid in zone_ids:
        tr = np.asarray(truth[zid])
        nv = np.asarray(naive[zid])
        lv = np.asarray(level[zid])
        events = true_events(tr, caps[zid], t)
        result.events += len(events)
        result.abs_err_naive += float(np.abs(nv - tr).sum())
        result.abs_err_ours += float(np.abs(np.asarray(ours[zid]) - tr).sum())
        result.err_samples += len(tr)

        window = max(1, int(round(2.0 * cfg.processing.fps)))
        averaged = np.convolve(nv, np.ones(window) / window, mode="full")[: len(nv)]
        for label, alarm_mask in (("naive", nv >= caps[zid]), ("smooth", averaged >= caps[zid]),
                                  ("ours", lv >= 3)):
            alarms = intervals(alarm_mask, t)
            setattr(result, f"{label}_alarms", getattr(result, f"{label}_alarms") + len(alarms))
            false = [a for a in alarms if not any(overlaps(a, ev) for ev in events)]
            setattr(result, f"{label}_false", getattr(result, f"{label}_false") + len(false))
            for ev in events:
                hits = [a for a in alarms if overlaps(a, ev)]
                if hits:
                    setattr(result, f"{label}_detected", getattr(result, f"{label}_detected") + 1)
                    delay = min(a[0] for a in hits) - ev[0]
                    # An alarm still on from an earlier event has no delay to
                    # speak of; leave it out rather than flatter the average.
                    if delay >= -TOLERANCE_S:
                        getattr(result, f"{label}_delays").append(delay)

        predicted_at = [a[0] for a in alert_log if a[1] == zid and a[2] == "predicted" and a[3] == "opened"]
        warned_at = [a[0] for a in alert_log if a[1] == zid and a[2] == "overcrowding" and a[3] == "opened"]
        rule_at = replay_predictions(cfg.alerts, t, lv, np.asarray(rule[zid]))
        result.warning.add(predicted_at, events)
        result.warning_rule.add(rule_at, events)
        any_score = WarningScore()
        any_score.add(sorted(rule_at + warned_at), events)
        result.warning_any.merge(any_score)

        if keep_logs:
            result.logs.append(ZoneLog(zid, caps[zid], t, tr, lv, np.asarray(feats[zid]),
                                       np.asarray(prob[zid]), np.asarray(rule[zid])))

    result.door_true = sum(src.world.entered.values()) + sum(src.world.exited.values())
    result.door_counted = sum(line.count_in + line.count_out for line in engine.lines)
    return result


def _stats(values) -> Optional[dict]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    return {"median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)), "n": int(len(values))}


def warning_summary(score: WarningScore, events: int) -> dict:
    return {
        "alerts": score.alerts, "false": score.false,
        "precision": (score.alerts - score.false) / score.alerts if score.alerts else None,
        "events_warned": score.events_warned,
        "recall": score.events_warned / events if events else None,
        "lead": _stats(score.leads),
    }


def summarise(results: list[RunResult]) -> dict:
    def total(name):
        return sum(getattr(r, name) for r in results)

    def pooled(name):
        return [v for r in results for v in getattr(r, name)]

    hours = total("duration_s") / 3600.0
    events = total("events")
    samples = max(total("err_samples"), 1)

    def system(name):
        return {
            "detected": total(f"{name}_detected"), "alarms": total(f"{name}_alarms"),
            "false_alarms": total(f"{name}_false"),
            "false_per_hour": total(f"{name}_false") / hours,
            "alarms_per_event": (total(f"{name}_alarms") - total(f"{name}_false")) / max(events, 1),
            "delay": _stats(pooled(f"{name}_delays")),
        }

    def warnings(name):
        merged = WarningScore()
        for r in results:
            merged.merge(getattr(r, name))
        return warning_summary(merged, events)

    return {
        "runs": len(results), "simulated_hours": hours, "events": events,
        "naive": {**system("naive"), "count_mae": total("abs_err_naive") / samples},
        "smooth": system("smooth"),
        "ours": {**system("ours"), "count_mae": total("abs_err_ours") / samples},
        "early_warning": warnings("warning"),
        "early_warning_rule": warnings("warning_rule"),
        "early_warning_any": warnings("warning_any"),
        "doors": {"true": total("door_true"), "counted": total("door_counted")},
    }
