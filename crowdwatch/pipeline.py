"""Runs source -> detector -> engine in a background thread and holds the latest result."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .analytics.alerts import AlertEvent
from .config import AppConfig, ZoneConfig, save_config
from .detect.yolox import build_detector
from .engine import Engine
from .render import ViewOptions, annotate
from .sources.simulator import SCENARIOS, SimulatorSource
from .sources.video import VideoSource
from .storage import Storage

log = logging.getLogger("crowdwatch")


def build_source(cfg: AppConfig, render: bool = True):
    s = cfg.source
    if s.type == "simulator":
        return SimulatorSource(s.scenario, seed=s.seed, fps=cfg.processing.fps,
                               time_scale=s.time_scale, loop=s.loop, render=render)
    return VideoSource(s.uri, fps=cfg.processing.fps, loop=s.loop)


class Webhook:
    """Posts alert changes as JSON to a URL, off the analysis thread."""

    def __init__(self, url: str):
        self.url = url
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        threading.Thread(target=self._run, daemon=True, name="webhook").start()

    def send(self, payload: dict) -> None:
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            log.warning("webhook queue is full; dropping an alert notification")

    def _run(self) -> None:
        while True:
            payload = self._queue.get()
            body = json.dumps(payload).encode()
            request = urllib.request.Request(self.url, data=body,
                                             headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(request, timeout=5).close()
            except Exception as exc:  # a dead webhook must never stop the monitoring
                log.warning("webhook delivery failed: %s", exc)


class DensityWorker:
    """Runs the slow density-map model beside the fast loop, on the newest frame only."""

    def __init__(self, cfg: AppConfig, on_result):
        from .detect.density import DensityCounter

        self.counter = DensityCounter(cfg.density.model, cfg.density.model_path, cfg.density.max_side)
        self.every_s = cfg.density.every_s
        self._on_result = on_result
        self._pending = None
        self._last_t = -1e9
        self._wake = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="density").start()

    def offer(self, t: float, image, zones: list) -> None:
        if t - self._last_t >= self.every_s and self._pending is None:
            self._last_t = t
            self._pending = (t, image, zones)
            self._wake.set()

    def estimate(self, image, zones: list) -> tuple[dict, float]:
        from .detect.density import zone_counts

        density = self.counter.density_map(image)
        counts = zone_counts(density, [z.polygon for z in zones])
        return {z.id: c for z, c in zip(zones, counts)}, float(density.sum())

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            job, self._pending = self._pending, None
            if job is None:
                continue
            t, image, zones = job
            try:
                counts, total = self.estimate(image, zones)
                self._on_result(t, counts, total)
            except Exception:
                log.exception("density model failed on a frame")

    def close(self) -> None:
        self._stop.set()
        self._wake.set()


class Pipeline:
    STREAM_FPS = 15.0
    JPEG_QUALITY = 80

    def __init__(self, cfg: AppConfig, config_path: Optional[str] = None):
        self.cfg = cfg
        self.config_path = config_path
        self.view = ViewOptions(blur=cfg.privacy.blur_people)
        self.storage = Storage(cfg.storage.path)
        self.webhook = Webhook(cfg.alerts.webhook_url) if cfg.alerts.webhook_url else None
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._jpeg: Optional[bytes] = None
        self._frame_id = 0
        self._snapshot: dict = {}
        self._pending_zones: Optional[list[ZoneConfig]] = None
        self._pending_restart: Optional[str] = None
        self.error: Optional[str] = None
        self.density: Optional[DensityWorker] = None
        if cfg.density.enabled and cfg.source.type == "video":
            self.density = DensityWorker(cfg, self._on_density)
        self._start_run()

    def _on_density(self, t: float, counts: dict, total: float) -> None:
        with self._lock:
            self.engine.set_density(t, counts, total)

    # ----------------------------------------------------------- lifecycle

    def _start_run(self) -> None:
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.source = build_source(self.cfg)
        self.detector = build_detector(self.cfg.detector, self.source.size,
                                       seed=self.cfg.source.seed,
                                       min_score=self.cfg.tracker.low_thresh)
        self.engine = Engine(self.cfg, self.source.size, next_alert_id=self.storage.next_alert_id())
        self.engine.alerts.subscribe(self._on_alert)
        self._last_sample_t = -1e9
        self._ended = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pipeline")
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)
        self.source.close()
        if self.density is not None:
            self.density.close()
        self.storage.close()

    # ------------------------------------------------------------- control

    def request_zones(self, zones: list[ZoneConfig]) -> None:
        """Validate new zones now, apply them on the analysis thread."""
        probe = Engine(self.cfg.model_copy(update={"zones": list(zones)}, deep=True), self.source.size)
        del probe
        with self._lock:
            self._pending_zones = zones

    def request_restart(self, scenario: Optional[str] = None) -> None:
        if self.cfg.source.type != "simulator":
            raise ValueError("only the simulator can be restarted with a scenario")
        if scenario is not None and scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario '{scenario}'")
        with self._lock:
            self._pending_restart = scenario or self.cfg.source.scenario

    def acknowledge(self, alert_id: int) -> bool:
        with self._lock:
            alert = self.engine.alerts.acknowledge(alert_id)
            if alert is None:
                return False
            self.storage.save_alert(self.run_id, alert)
            return True

    # ---------------------------------------------------------------- loop

    def _on_alert(self, event: AlertEvent) -> None:
        self.storage.save_alert(self.run_id, event.alert)
        log.info("alert %s: %s", event.type, event.alert.message)
        if self.webhook:
            self.webhook.send({"event": event.type, "site": self.cfg.name, **event.alert.to_dict()})

    def _apply_pending(self) -> None:
        with self._lock:
            zones, self._pending_zones = self._pending_zones, None
            restart, self._pending_restart = self._pending_restart, None
        if zones is not None:
            with self._lock:
                self.engine.set_zones(zones)
            if self.config_path:
                save_config(self.cfg, self.config_path)
        if restart is not None:
            with self._lock:
                self.engine.alerts.close_all(self.engine.t or 0.0, "scenario restarted")
                self.cfg.source.scenario = restart
                self.source.close()
                self._start_run()

    def _loop(self) -> None:
        next_due = time.perf_counter()
        last_render = 0.0
        while self._running.is_set():
            try:
                self._apply_pending()
                frame = self.source.read()
                if frame is None:
                    if self.cfg.source.type == "simulator" and self.cfg.source.loop:
                        with self._lock:
                            self.engine.alerts.close_all(self.engine.t or 0.0, "scenario restarted")
                            self.source.close()
                            self._start_run()
                        continue
                    self._ended = True
                    self._publish_end()
                    time.sleep(0.25)
                    continue
                if getattr(self.source, "rewound", False):
                    self.source.rewound = False
                    self.engine.tracker.reset()

                if frame.image is None and frame.truth is None:
                    detections = build_empty()
                else:
                    detections = self.detector.detect(frame)
                    if self.density is not None and frame.image is not None:
                        self.density.offer(frame.t, frame.image, list(self.cfg.zones))
                with self._lock:
                    snapshot = self.engine.step(frame.t, detections.boxes, detections.scores)

                now = time.perf_counter()
                jpeg = None
                if frame.image is not None and now - last_render >= 1.0 / self.STREAM_FPS:
                    import cv2

                    last_render = now
                    image = annotate(frame.image, self.engine, snapshot, self.view,
                                     compact_markers=self.cfg.source.type == "simulator")
                    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
                    jpeg = buf.tobytes() if ok else None

                self._decorate(snapshot)
                with self._lock:
                    self._snapshot = snapshot
                    if jpeg is not None:
                        self._jpeg = jpeg
                        self._frame_id += 1
                if frame.t - self._last_sample_t >= self.cfg.storage.sample_every_s:
                    self._last_sample_t = frame.t
                    self.storage.add_samples(self.run_id, time.time(), frame.t, snapshot["zones"])
                self.error = None

                # pace recordings and simulations; live cameras pace themselves
                if not self.source.live:
                    next_due += (1.0 / self.cfg.processing.fps) / max(self.source.speed, 1e-6)
                    delay = next_due - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_due = time.perf_counter()      # running behind: do not try to catch up
            except Exception as exc:  # keep the service up and say what went wrong
                log.exception("pipeline error")
                self.error = str(exc)
                time.sleep(1.0)

    def _decorate(self, snapshot: dict) -> None:
        s = self.cfg.source
        t = snapshot["t"]
        snapshot["name"] = self.cfg.name
        snapshot["run_id"] = self.run_id
        snapshot["ended"] = False
        snapshot["frame"] = {"width": self.source.size[0], "height": self.source.size[1]}
        snapshot["view"] = {"markers": self.view.markers, "heatmap": self.view.heatmap, "blur": self.view.blur}
        if self.source.live:
            snapshot["clock"] = datetime.now().strftime("%H:%M:%S")
        else:
            snapshot["clock"] = f"{int(t // 60):02d}:{int(t % 60):02d}"
        source = {"type": s.type, "live": self.source.live, "speed": self.source.speed}
        if s.type == "simulator":
            sc = SCENARIOS[s.scenario]
            source.update(scenario=sc.name, title=sc.title, description=sc.description,
                          duration_s=sc.duration_s)
        else:
            source["label"] = Path(str(s.uri)).name if not self.source.live else "Live camera"
        snapshot["source"] = source

    def _publish_end(self) -> None:
        with self._lock:
            if self._snapshot:
                self._snapshot = {**self._snapshot, "ended": True}

    # ------------------------------------------------------------- readers

    def snapshot(self) -> dict:
        with self._lock:
            snap = dict(self._snapshot)
        snap["error"] = self.error
        return snap

    def jpeg(self) -> tuple[int, Optional[bytes]]:
        with self._lock:
            return self._frame_id, self._jpeg

    def history(self, zone_id: str, seconds: float) -> list[tuple]:
        with self._lock:
            return list(self.engine.history(zone_id, seconds))

    def alert_log(self, limit: int = 50) -> list[dict]:
        return self.storage.recent_alerts(limit)


def build_empty():
    from .detect.base import Detections

    return Detections.empty()
