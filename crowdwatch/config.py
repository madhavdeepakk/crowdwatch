"""Configuration schema.

Everything about a deployment lives in one YAML file: where the video comes
from, which areas to watch and how many people each can hold, and when to
raise an alert. Zone and line coordinates are normalised (0 to 1 across the
frame) so a config keeps working if the camera resolution changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Point = tuple[float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(_Model):
    type: Literal["simulator", "video"] = "simulator"
    # video: a file path, an RTSP/HTTP URL, or a webcam index such as 0
    uri: Optional[Union[int, str]] = None
    loop: bool = True
    # simulator
    scenario: str = "surge"
    seed: int = 7
    time_scale: float = Field(4.0, gt=0, le=60)

    @model_validator(mode="after")
    def _video_needs_uri(self):
        if self.type == "video" and self.uri is None:
            raise ValueError("source.uri is required when source.type is 'video'")
        return self


class ProcessingConfig(_Model):
    fps: float = Field(10.0, gt=0, le=60)
    # which point of a person's box stands for where they are on the floor
    anchor: Literal["foot", "center"] = "foot"


class DetectorConfig(_Model):
    backend: Literal["yolox", "sim"] = "yolox"
    model: Literal["yolox_tiny", "yolox_s"] = "yolox_s"
    model_path: Optional[str] = None
    nms: float = Field(0.5, gt=0, lt=1)
    # split the frame into overlapping tiles so small, far-away people are found
    tiles: tuple[int, int] = (1, 1)
    threads: int = Field(0, ge=0)


class DensityConfig(_Model):
    """Optional second opinion for packed crowds, where box detectors run low."""

    enabled: bool = False
    model: Literal["dmcount_qnrf", "dmcount_shb"] = "dmcount_qnrf"
    model_path: Optional[str] = None
    every_s: float = Field(2.0, gt=0)          # how often the density map is refreshed
    max_side: int = Field(1024, ge=256, le=2048)
    # the density estimate takes over once it is at least this many people ...
    min_people: float = Field(15.0, ge=0)
    # ... and this many times the tracked count
    switch_ratio: float = Field(1.2, ge=1.0)


class TrackerConfig(_Model):
    high_thresh: float = Field(0.5, gt=0, lt=1)
    low_thresh: float = Field(0.1, gt=0, lt=1)
    new_thresh: float = Field(0.6, gt=0, lt=1)
    match_iou: float = Field(0.2, gt=0, lt=1)
    match_iou_low: float = Field(0.4, gt=0, lt=1)
    min_hits: int = Field(3, ge=1)
    max_age_s: float = Field(2.0, gt=0)
    coast_s: float = Field(1.2, ge=0)


class CalibrationConfig(_Model):
    image_points: list[Point] = Field(min_length=4)
    floor_points: list[Point] = Field(min_length=4)

    @model_validator(mode="after")
    def _same_length(self):
        if len(self.image_points) != len(self.floor_points):
            raise ValueError("calibration.image_points and floor_points must pair up one to one")
        return self


class ZoneConfig(_Model):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z0-9_\-]+$")
    name: str = Field(min_length=1, max_length=60)
    polygon: list[Point] = Field(min_length=3)
    capacity: Optional[int] = Field(None, gt=0)
    area_m2: Optional[float] = Field(None, gt=0)
    max_density: Optional[float] = Field(None, gt=0)
    # People are meant to keep moving here (a corridor, a doorway). A dense
    # crowd that has stopped is then flagged as congestion.
    walkway: bool = False

    @field_validator("polygon")
    @classmethod
    def _normalised(cls, pts):
        for x, y in pts:
            if not (-0.01 <= x <= 1.01 and -0.01 <= y <= 1.01):
                raise ValueError("zone polygon points must be normalised to the 0..1 range")
        return pts


class LineConfig(_Model):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z0-9_\-]+$")
    name: str = Field(min_length=1, max_length=60)
    a: Point
    b: Point


class AlertConfig(_Model):
    busy_ratio: float = Field(0.6, gt=0)
    warning_ratio: float = Field(0.8, gt=0)
    critical_ratio: float = Field(1.0, gt=0)
    hysteresis: float = Field(0.05, ge=0, lt=0.5)
    raise_after_s: float = Field(3.0, ge=0)
    clear_after_s: float = Field(15.0, ge=0)
    smoothing_s: float = Field(2.0, gt=0)
    # Early warning. "rule" is the damped trend on its own. "learned" adds the trained
    # model, which caught more build-ups in testing but is not clearly better
    # overall (see docs/evaluation.md), so it is opt-in.
    forecast_method: Literal["rule", "learned"] = "rule"
    forecast_model_path: Optional[str] = None
    forecast_threshold: Optional[float] = Field(None, gt=0, lt=1)   # None: the model's own
    forecast_window_s: float = Field(45.0, ge=10)
    forecast_lead_s: float = Field(120.0, ge=10)
    forecast_damping_s: float = Field(90.0, ge=10)
    forecast_min_ratio: float = Field(0.4, ge=0)
    forecast_min_z: float = Field(1.5, ge=0)
    # A crowd streaming through at walking pace is not piling up. Only predict
    # overcrowding while the people in the zone are slower than this.
    forecast_max_speed_mps: Optional[float] = Field(0.7, gt=0)
    forecast_confirm_s: float = Field(4.0, ge=0)
    forecast_clear_s: float = Field(30.0, ge=0)
    # slow, packed crowds are the dangerous kind
    stall_speed_mps: float = Field(0.3, ge=0)
    webhook_url: Optional[str] = None

    @model_validator(mode="after")
    def _ordered(self):
        if not (self.busy_ratio < self.warning_ratio < self.critical_ratio):
            raise ValueError("alert ratios must satisfy busy < warning < critical")
        return self


class PrivacyConfig(_Model):
    blur_people: bool = True


class StorageConfig(_Model):
    path: Optional[str] = "data/crowdwatch.db"
    sample_every_s: float = Field(2.0, gt=0)


class AppConfig(_Model):
    name: str = "CrowdWatch"
    source: SourceConfig = SourceConfig()
    processing: ProcessingConfig = ProcessingConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    density: DensityConfig = DensityConfig()
    calibration: Optional[CalibrationConfig] = None
    # used to turn a zone's floor area into a capacity when none is given
    default_max_density: float = Field(2.0, gt=0)
    zones: list[ZoneConfig] = []
    lines: list[LineConfig] = []
    alerts: AlertConfig = AlertConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    storage: StorageConfig = StorageConfig()

    @model_validator(mode="after")
    def _unique_ids(self):
        for label, items in (("zone", self.zones), ("line", self.lines)):
            ids = [item.id for item in items]
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ValueError(f"duplicate {label} id(s): {', '.join(sorted(dupes))}")
        return self


def load_config(path: Union[str, Path]) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)


def save_config(cfg: AppConfig, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(mode="json", exclude_none=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
