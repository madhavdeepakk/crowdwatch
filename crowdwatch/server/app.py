"""HTTP and WebSocket API, and the dashboard it serves."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from ..config import ZoneConfig
from ..engine import ConfigError
from ..pipeline import Pipeline
from ..sources.simulator import SCENARIOS

STATIC = Path(__file__).parent / "static"


class ZonesBody(BaseModel):
    zones: list[ZoneConfig]


class ViewBody(BaseModel):
    markers: Optional[bool] = None
    heatmap: Optional[bool] = None
    blur: Optional[bool] = None


class ScenarioBody(BaseModel):
    name: Optional[str] = None


def create_app(pipeline: Pipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        pipeline.start()
        yield
        pipeline.stop()

    app = FastAPI(title="CrowdWatch", version="1.0.0", docs_url="/api/docs", redoc_url=None,
                  lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------- reads

    @app.get("/api/state")
    async def state():
        """The latest snapshot: every zone's count, level, forecast and the active alerts."""
        return pipeline.snapshot()

    @app.get("/api/config")
    async def config():
        cfg = pipeline.cfg
        capacities = {z.cfg.id: z.capacity for z in pipeline.engine.zones}
        body = {
            "name": cfg.name,
            "source": cfg.source.type,
            "zones": [{**z.model_dump(mode="json"), "capacity": capacities.get(z.id, z.capacity)}
                      for z in cfg.zones],
            "lines": [line.model_dump(mode="json") for line in cfg.lines],
            "thresholds": {"busy": cfg.alerts.busy_ratio, "warning": cfg.alerts.warning_ratio,
                           "critical": cfg.alerts.critical_ratio,
                           "forecast_lead_s": cfg.alerts.forecast_lead_s},
            "scenarios": [],
        }
        if cfg.source.type == "simulator":
            body["scenarios"] = [{"name": s.name, "title": s.title, "description": s.description}
                                 for s in SCENARIOS.values()]
        return body

    @app.get("/api/history")
    async def history(seconds: float = 600.0):
        """Recent counts per zone as [video_time_s, people, occupancy, forecast] rows."""
        seconds = max(10.0, min(seconds, 3600.0))
        return {"zones": {z.id: pipeline.history(z.id, seconds) for z in pipeline.cfg.zones}}

    @app.get("/api/alerts")
    async def alerts(limit: int = 30):
        return {"alerts": pipeline.alert_log(max(1, min(limit, 200)))}

    @app.get("/api/export.csv")
    async def export_csv(all_runs: bool = False):
        text = pipeline.storage.export_csv(None if all_runs else pipeline.run_id)
        return PlainTextResponse(text, media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="crowdwatch-counts.csv"'})

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus text format, for Grafana dashboards and alerting."""
        s = pipeline.snapshot()
        esc = lambda v: str(v).replace("\\", "\\\\").replace('"', '\\"')
        lines = []

        def gauge(name, text, rows, kind="gauge"):
            lines.append(f"# HELP crowdwatch_{name} {text}")
            lines.append(f"# TYPE crowdwatch_{name} {kind}")
            for labels, value in rows:
                label = ",".join(f'{k}="{esc(v)}"' for k, v in labels.items())
                lines.append(f"crowdwatch_{name}{{{label}}} {value}" if label else f"crowdwatch_{name} {value}")

        zones = s.get("zones", [])
        gauge("people_in_view", "People currently tracked in the picture.", [({}, s.get("people", 0))])
        gauge("pipeline_fps", "Frames analysed per second.", [({}, s.get("fps", 0))])
        gauge("zone_people", "People counted in the zone.", [({"zone": z["id"]}, z["count_smooth"]) for z in zones])
        gauge("zone_capacity", "People the zone can safely hold.", [({"zone": z["id"]}, z["capacity"]) for z in zones])
        gauge("zone_occupancy_ratio", "People divided by capacity.", [({"zone": z["id"]}, z["ratio"]) for z in zones])
        gauge("zone_level", "0 normal, 1 busy, 2 nearly full, 3 over capacity.", [({"zone": z["id"]}, z["level"]) for z in zones])
        gauge("zone_approaching", "People walking towards the zone.", [({"zone": z["id"]}, z["approaching"]) for z in zones])
        gauge("zone_fill_probability", "Estimated chance the zone fills within two minutes.",
              [({"zone": z["id"]}, z["probability"]) for z in zones if z.get("probability") is not None])
        gauge("door_crossings_total", "People counted across a door line.",
              [({"door": d["id"], "direction": k}, d[k]) for d in s.get("lines", []) for k in ("in", "out")],
              kind="counter")
        active = s.get("alerts", [])
        gauge("active_alerts", "Alerts currently open.",
              [({"level": lv}, sum(1 for a in active if a["level"] == lv)) for lv in ("predicted", "warning", "critical")])
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------ writes

    @app.post("/api/alerts/{alert_id}/ack")
    async def acknowledge(alert_id: int):
        if not pipeline.acknowledge(alert_id):
            raise HTTPException(404, "No alert with that number in the current run.")
        return {"acknowledged": alert_id}

    @app.put("/api/zones")
    async def put_zones(body: ZonesBody):
        ids = [z.id for z in body.zones]
        if len(set(ids)) != len(ids):
            raise HTTPException(422, "Two zones share the same id. Give each zone its own name.")
        try:
            pipeline.request_zones(body.zones)
        except (ConfigError, ValidationError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"saved": len(body.zones)}

    @app.post("/api/view")
    async def set_view(body: ViewBody):
        for key, value in body.model_dump(exclude_none=True).items():
            setattr(pipeline.view, key, value)
        v = pipeline.view
        return {"markers": v.markers, "heatmap": v.heatmap, "blur": v.blur}

    @app.post("/api/scenario")
    async def restart_scenario(body: ScenarioBody):
        try:
            pipeline.request_restart(body.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"restarting": body.name or pipeline.cfg.source.scenario}

    # ------------------------------------------------------------- video

    @app.get("/api/frame.jpg")
    async def frame():
        _, jpeg = pipeline.jpeg()
        if jpeg is None:
            raise HTTPException(503, "No video frame yet.")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/stream.mjpg")
    async def stream():
        async def frames():
            last = -1
            while True:
                frame_id, jpeg = pipeline.jpeg()
                if jpeg is not None and frame_id != last:
                    last = frame_id
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.03)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def live(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_json(pipeline.snapshot())
                await asyncio.sleep(0.25)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app
