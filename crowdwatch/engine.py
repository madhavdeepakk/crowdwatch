"""The analysis engine: detections in, a full picture of the venue out.

The engine has no idea where frames come from or where results go. It is fed
person boxes with a timestamp and returns a snapshot dictionary. That keeps it
deterministic (it runs on the video's clock, not the wall clock), which is
what makes the evaluation and the tests possible.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .analytics.alerts import CRITICAL, LEVEL_NAMES, AlertManager
from .analytics.heatmap import Heatmap
from .analytics.lines import LineCounter
from .analytics.early_warning import load_model
from .analytics.zones import VenueContext, ZoneMetrics, ZoneState
from .config import AppConfig, ZoneConfig
from .geometry import Homography, points_in_polygon, polygon_area
from .tracking.bytetrack import ByteTracker, Tracks

PERSON_HEIGHT_M = 1.7
# "Heading for a zone" means: keep walking as now and you are inside it at one
# of these moments (seconds ahead).
LOOK_AHEAD_S = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
WALKING_MPS = 0.4


class ConfigError(ValueError):
    pass


class Engine:
    def __init__(self, cfg: AppConfig, frame_size: tuple[int, int], next_alert_id: int = 1):
        self.cfg = cfg
        self.width, self.height = frame_size
        self._scale = np.array([self.width, self.height], dtype=float)
        self.tracker = ByteTracker(cfg.tracker)
        self.heatmap = Heatmap(frame_size)
        self.model = None
        if cfg.alerts.forecast_method == "learned":
            self.model = load_model(cfg.alerts.forecast_model_path)
        self.alerts = AlertManager(cfg.alerts, next_id=next_alert_id)
        self.homography: Optional[Homography] = None
        if cfg.calibration is not None:
            self.homography = Homography(
                np.asarray(cfg.calibration.image_points) * self._scale,
                np.asarray(cfg.calibration.floor_points),
            )
        self.t: Optional[float] = None
        self.zones: list[ZoneState] = []
        self.set_zones(cfg.zones)
        margin = 0.008 * float(np.hypot(self.width, self.height))
        self.lines = [
            LineCounter(line, np.asarray(line.a) * self._scale, np.asarray(line.b) * self._scale, margin)
            for line in cfg.lines
        ]
        self.tracks: Tracks = Tracks.empty()
        self.last_metrics: list[ZoneMetrics] = []
        self._density_total: Optional[tuple[float, float]] = None
        self.anchors = np.zeros((0, 2))
        self._fps = 0.0
        self._last_wall = None

    # ------------------------------------------------------------- zones

    def _build_zone(self, zone: ZoneConfig) -> ZoneState:
        polygon_px = np.asarray(zone.polygon, dtype=float) * self._scale
        area = zone.area_m2
        if area is None and self.homography is not None:
            area = polygon_area(self.homography.to_floor(polygon_px))
        capacity = zone.capacity
        if capacity is None:
            if area is None:
                raise ConfigError(
                    f"zone '{zone.id}' needs a capacity, or an area_m2 (or a camera calibration) "
                    "so a capacity can be worked out from its floor area"
                )
            density = zone.max_density or self.cfg.default_max_density
            capacity = max(1, int(round(area * density)))
        state = ZoneState(zone, polygon_px, capacity, area, self.cfg.alerts, self.model)
        state.fusion = (self.cfg.density.min_people, self.cfg.density.switch_ratio)
        return state

    def set_zones(self, zones: list[ZoneConfig]) -> None:
        """Replace the watched zones. Zones whose definition is unchanged keep their history."""
        previous = {z.cfg.id: z for z in self.zones}
        rebuilt = []
        for zone in zones:
            old = previous.get(zone.id)
            if old is not None and old.cfg == zone:
                rebuilt.append(old)
            else:
                rebuilt.append(self._build_zone(zone))
        self.alerts.close_missing({z.cfg.id for z in rebuilt}, self.t or 0.0)
        self.zones = rebuilt
        self.cfg.zones = list(zones)

    def set_density(self, t: float, zone_counts: dict[str, float], total: Optional[float] = None) -> None:
        """Hand in fresh per-zone estimates from the density-map model."""
        for zone in self.zones:
            if zone.cfg.id in zone_counts:
                zone.set_density(t, zone_counts[zone.cfg.id])
        if total is not None:
            self._density_total = (t, total)

    # -------------------------------------------------------------- step

    def step(self, t: float, boxes: np.ndarray, scores: np.ndarray) -> dict:
        dt = 0.0 if self.t is None else max(t - self.t, 0.0)
        self.t = t
        tracks = self.tracker.update(boxes, scores, t)
        self.tracks = tracks

        if self.cfg.processing.anchor == "foot":
            anchors = np.column_stack([(tracks.boxes[:, 0] + tracks.boxes[:, 2]) / 2, tracks.boxes[:, 3]])
        else:
            anchors = np.column_stack([(tracks.boxes[:, 0] + tracks.boxes[:, 2]) / 2,
                                       (tracks.boxes[:, 1] + tracks.boxes[:, 3]) / 2])
        self.anchors = anchors
        speeds = self._speeds(tracks, anchors)

        for line in self.lines:
            line.update(t, tracks.ids, anchors)
        rates = [line.rates_per_min() for line in self.lines]
        venue = VenueContext(people=len(tracks),
                             door_in_per_min=sum(r[0] for r in rates),
                             door_out_per_min=sum(r[1] for r in rates))

        # Where will everyone be a few seconds from now if they keep walking?
        n = len(tracks)
        walking = speeds > WALKING_MPS if n else np.zeros(0, dtype=bool)
        future = (anchors[:, None, :] + tracks.velocity[:, None, :] * LOOK_AHEAD_S[None, :, None]).reshape(-1, 2)

        metrics: list[ZoneMetrics] = []
        self.last_metrics = metrics
        for zone in self.zones:
            inside = points_in_polygon(anchors, zone.polygon_px)
            approaching = leaving = 0
            if n:
                later = points_in_polygon(future, zone.polygon_px).reshape(n, len(LOOK_AHEAD_S))
                approaching = int((~inside & walking & later.any(axis=1)).sum())
                leaving = int((inside & walking & ~later[:, 1]).sum())
            m = zone.update(t, dt, tracks.ids[inside], speeds[inside] if n else None,
                            approaching=approaching, leaving=leaving, venue=venue)
            metrics.append(m)
            self.alerts.update(t, m.reading())

        self.heatmap.update(dt, anchors)

        wall = time.perf_counter()
        if self._last_wall is not None and wall > self._last_wall:
            self._fps += 0.1 * (1.0 / (wall - self._last_wall) - self._fps)
        self._last_wall = wall
        return self._snapshot(t, metrics)

    def _speeds(self, tracks: Tracks, anchors: np.ndarray) -> np.ndarray:
        """Walking speed of each person in metres per second.

        With a calibrated camera the pixel velocity is projected onto the
        floor. Without one, a person's own height in the image is used as the
        ruler, which is rough but self-corrects for distance from the camera.
        """
        if len(tracks) == 0:
            return np.zeros(0)
        if self.homography is not None:
            step = 0.1
            here = self.homography.to_floor(anchors)
            there = self.homography.to_floor(anchors + tracks.velocity * step)
            return np.linalg.norm(there - here, axis=1) / step
        height = np.maximum(tracks.boxes[:, 3] - tracks.boxes[:, 1], 1.0)
        return np.linalg.norm(tracks.velocity, axis=1) / height * PERSON_HEIGHT_M

    # ---------------------------------------------------------- snapshot

    def _snapshot(self, t: float, metrics: list[ZoneMetrics]) -> dict:
        active = sorted(
            self.alerts.active,
            key=lambda a: ({"critical": 0, "warning": 1, "predicted": 2}[a.level], -a.started_t),
        )
        worst = max((m.level for m in metrics), default=0)
        if active:
            headline = active[0].message
        elif not metrics:
            headline = "No zones yet. Draw one on the live view to start counting."
        else:
            headline = "All zones are within their limits."
        people = float(len(self.tracks))
        if self._density_total is not None and t - self._density_total[0] <= 10.0:
            from .detect.density import fuse

            people = fuse(people, self._density_total[1], self.cfg.density.min_people,
                          self.cfg.density.switch_ratio)
        return {
            "t": round(t, 2),
            "fps": round(self._fps, 1),
            "people": int(round(people)),
            "density_model": self.cfg.density.enabled,
            "calibrated": self.homography is not None,
            "early_warning": "learned" if self.model is not None else "rule",
            "status": {
                "level": worst,
                "level_name": LEVEL_NAMES[worst],
                "predicted": any(a.level == "predicted" for a in active) and worst < CRITICAL,
                "headline": headline,
            },
            "zones": [m.to_dict() for m in metrics],
            "lines": [line.to_dict() for line in self.lines],
            "alerts": [a.to_dict() for a in active],
        }

    def history(self, zone_id: str, seconds: float) -> list[tuple]:
        zone = next((z for z in self.zones if z.cfg.id == zone_id), None)
        if zone is None or self.t is None:
            return []
        start = self.t - seconds
        return [row for row in zone.history if row[0] >= start]
