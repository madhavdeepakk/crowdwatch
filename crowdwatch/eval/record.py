"""Record runs of the real system for the browser demo.

The live demo page is a replay: it draws what the simulator, tracker, zone
analytics and alert logic actually produced, second by second. Nothing about
the system is re-implemented in the browser, so the demo cannot drift from the
real behaviour. This module makes those recordings.

Each scenario becomes one JSON file holding, once per second of video time,
the metrics for every zone, the door counters, the active alerts, and the
position of every person (packed as 16-bit integers, base64 encoded).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from ..detect.sim import SimDetector
from ..engine import Engine
from ..presets import demo_config
from ..sources.simulator import PX_PER_M, SCENARIOS, WORLD_H, WORLD_W, SimulatorSource

KEY_S = 1.0          # one keyframe per second of video time
MARGIN_M = 2.0       # people are recorded a little beyond the frame so they can walk in and out
UNITS_PER_M = 500    # positions are stored to 2 mm


def _r(value, digits=2):
    return None if value is None else round(float(value), digits)


def record(scenario: str, seed: int = 7) -> dict:
    cfg = demo_config(scenario, seed, storage_path=None)
    src = SimulatorSource(scenario, seed=seed, fps=cfg.processing.fps, render=False)
    detector = SimDetector(src.size, seed)
    engine = Engine(cfg, src.size)
    every = int(round(KEY_S * cfg.processing.fps))
    fields = ("count", "ratio", "level", "trend", "fc", "eta", "speed", "dwell", "appr", "risk", "pred", "cong", "dens")
    zones = [{f: [] for f in fields} for _ in engine.zones]
    doors = [{"in": [], "out": []} for _ in engine.lines]
    times, people, alerts, packed = [], [], {}, []

    frame_no = 0
    while True:
        frame = src.read()
        if frame is None:
            break
        dets = detector.detect(frame)
        snap = engine.step(frame.t, dets.boxes, dets.scores)
        frame_no += 1
        if frame_no % every:
            continue
        k = len(times)
        times.append(round(frame.t, 1))
        people.append(snap["people"])
        for store, m in zip(zones, engine.last_metrics):
            store["count"].append(int(round(m.count)))
            store["ratio"].append(_r(m.ratio, 3))
            store["level"].append(m.level)
            store["trend"].append(_r(m.trend_per_min, 1))
            store["fc"].append(_r(m.forecast_count, 1))
            store["eta"].append(_r(m.eta_s, 0))
            store["speed"].append(_r(m.speed_mps, 2))
            store["dwell"].append(_r(m.dwell_s, 0))
            store["appr"].append(int(round(m.approaching)))
            store["risk"].append(m.risk)
            store["pred"].append(int(m.predicted))
            store["cong"].append(int(m.congested))
            store["dens"].append(_r(m.density, 2))
        for store, line in zip(doors, engine.lines):
            store["in"].append(line.count_in)
            store["out"].append(line.count_out)
        if snap["alerts"]:
            alerts[str(k)] = [{key: a[key] for key in ("id", "zone_id", "kind", "level", "message", "started_t")}
                              for a in snap["alerts"]]

        # Everyone on the floor, and whether the tracker currently has them.
        world = src.world
        pos = world.pos
        keep = ((pos[:, 0] > -1.5) & (pos[:, 0] < WORLD_W + 1.5) & (pos[:, 1] > -1.5) & (pos[:, 1] < WORLD_H + 1.5))
        ids, pos = world.ids[keep], pos[keep]
        tracked = np.zeros(len(ids), dtype=bool)
        if len(engine.anchors) and len(ids):
            gap = np.linalg.norm(pos[:, None, :] - engine.anchors[None, :, :] / PX_PER_M, axis=2)
            tracked = gap.min(axis=1) < 0.5
        row = np.empty((len(ids), 3), dtype=np.uint16)
        row[:, 0] = (ids & 0x7FFF) | (tracked.astype(np.uint16) << 15)
        row[:, 1] = np.round((pos[:, 0] + MARGIN_M) * UNITS_PER_M)
        row[:, 2] = np.round((pos[:, 1] + MARGIN_M) * UNITS_PER_M)
        packed.append(np.concatenate([np.array([len(ids)], dtype=np.uint16), row.ravel()]))

    engine.alerts.close_all(engine.t, "the recording ended")
    sc = SCENARIOS[scenario]
    return {
        "name": sc.name, "title": sc.title, "description": sc.description,
        "duration": times[-1], "key_s": KEY_S, "seed": seed,
        "frame": {"width": src.size[0], "height": src.size[1]},
        "world": {"w": WORLD_W, "h": WORLD_H, "margin": MARGIN_M, "units": UNITS_PER_M},
        "thresholds": {"warning": cfg.alerts.warning_ratio, "critical": cfg.alerts.critical_ratio,
                       "lead_s": cfg.alerts.forecast_lead_s},
        "zones": [{"id": z.cfg.id, "name": z.cfg.name, "capacity": z.capacity,
                   "polygon": [list(p) for p in z.cfg.polygon], "walkway": z.cfg.walkway} for z in engine.zones],
        "lines": [{"id": line.cfg.id, "name": line.cfg.name, "a": list(line.cfg.a), "b": list(line.cfg.b)}
                  for line in engine.lines],
        "t": times, "people": people, "z": zones, "doors": doors, "alerts": alerts,
        "log": [{"id": a.id, "zone_id": a.zone_id, "zone_name": a.zone_name, "kind": a.kind, "level": a.level,
                 "started_t": round(a.started_t, 1), "ended_t": round(a.ended_t, 1),
                 "peak_count": round(a.peak_count), "peak_ratio": _r(a.peak_ratio, 3), "outcome": a.outcome}
                for a in engine.alerts.history],
        "pos": base64.b64encode(np.concatenate(packed).astype("<u2").tobytes()).decode("ascii"),
    }


def record_all(out_dir: str | Path, seed: int = 7) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in SCENARIOS:
        data = record(name, seed)
        path = out / f"{name}.json"
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        paths.append(path)
    return paths
