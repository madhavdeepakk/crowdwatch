import time

import pytest
from fastapi.testclient import TestClient

from crowdwatch.pipeline import Pipeline
from crowdwatch.presets import demo_config
from crowdwatch.server.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("data") / "test.db"
    cfg = demo_config("surge", seed=5, time_scale=20, storage_path=str(db))
    pipeline = Pipeline(cfg)
    with TestClient(create_app(pipeline)) as c:
        deadline = time.time() + 20
        while time.time() < deadline and not c.get("/api/state").json().get("zones"):
            time.sleep(0.1)
        c.pipeline = pipeline
        yield c


def test_state_has_every_zone_and_a_headline(client):
    state = client.get("/api/state").json()
    assert [z["id"] for z in state["zones"]] == ["west_hall", "atrium", "food_court", "south_concourse"]
    assert state["status"]["headline"]
    assert state["source"]["scenario"] == "surge" and state["source"]["speed"] == 20
    assert state["frame"] == {"width": 1280, "height": 768}
    assert state["error"] is None


def test_dashboard_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "CrowdWatch" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/fonts/overpass-latin-wght-normal.woff2").status_code == 200


def test_config_lists_zones_lines_and_scenarios(client):
    cfg = client.get("/api/config").json()
    assert len(cfg["zones"]) == 4 and len(cfg["lines"]) == 3
    assert {s["name"] for s in cfg["scenarios"]} == {"surge", "gradual", "busy", "wave"}
    assert cfg["thresholds"]["critical"] == 1.0


def test_a_video_frame_is_a_jpeg(client):
    deadline = time.time() + 10
    res = client.get("/api/frame.jpg")
    while res.status_code == 503 and time.time() < deadline:
        time.sleep(0.2)
        res = client.get("/api/frame.jpg")
    assert res.status_code == 200 and res.content[:2] == b"\xff\xd8"


def test_history_and_csv_export(client):
    time.sleep(1.0)
    history = client.get("/api/history?seconds=600").json()["zones"]
    assert set(history) == {"west_hall", "atrium", "food_court", "south_concourse"}
    assert len(history["atrium"]) > 3 and len(history["atrium"][0]) == 4
    csv = client.get("/api/export.csv")
    assert csv.headers["content-type"].startswith("text/csv")
    assert csv.text.splitlines()[0].startswith("run_id,wall_time,video_time_s,zone,people")
    assert len(csv.text.splitlines()) > 4


def test_view_options_can_be_switched(client):
    assert client.post("/api/view", json={"heatmap": True}).json()["heatmap"] is True
    assert client.post("/api/view", json={"heatmap": False, "markers": False}).json() == {
        "markers": False, "heatmap": False, "blur": False}
    client.post("/api/view", json={"markers": True})


def test_websocket_pushes_snapshots(client):
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        second = ws.receive_json()
    assert "zones" in first and second["t"] >= first["t"]


def test_zones_can_be_replaced_and_bad_ones_are_refused(client):
    zones = client.get("/api/config").json()["zones"]
    extra = {"id": "kiosk", "name": "Kiosk", "capacity": 8,
             "polygon": [[0.7, 0.6], [0.9, 0.6], [0.9, 0.8], [0.7, 0.8]]}
    assert client.put("/api/zones", json={"zones": zones + [extra]}).json() == {"saved": 5}
    deadline = time.time() + 10
    while time.time() < deadline and len(client.get("/api/state").json()["zones"]) != 5:
        time.sleep(0.1)
    assert client.get("/api/state").json()["zones"][-1]["name"] == "Kiosk"

    bad = {"id": "kiosk", "name": "Twin", "capacity": 8, "polygon": extra["polygon"]}
    assert client.put("/api/zones", json={"zones": zones + [extra, bad]}).status_code == 422
    no_capacity = {"id": "open", "name": "Open", "polygon": [[0, 0], [5, 0], [1, 1]]}
    assert client.put("/api/zones", json={"zones": [no_capacity]}).status_code == 422
    assert client.put("/api/zones", json={"zones": zones}).status_code == 200


def test_unknown_alert_and_scenario_are_reported(client):
    assert client.post("/api/alerts/99999/ack").status_code == 404
    res = client.post("/api/scenario", json={"name": "stampede"})
    assert res.status_code == 400 and "unknown scenario" in res.json()["detail"]


def test_surge_alert_can_be_acknowledged_and_is_logged(client):
    # at 20x speed the surge at 1:30 of video time arrives within a few seconds
    deadline = time.time() + 60
    alerts = []
    while time.time() < deadline and not alerts:
        alerts = client.get("/api/state").json()["alerts"]
        time.sleep(0.2)
    assert alerts, "no alert was raised during the surge"
    alert_id = alerts[0]["id"]
    assert client.post(f"/api/alerts/{alert_id}/ack").json() == {"acknowledged": alert_id}
    logged = client.get("/api/alerts").json()["alerts"]
    assert any(a["id"] == alert_id and a["acknowledged"] for a in logged)


def test_scenario_restart_starts_a_new_run(client):
    before = client.get("/api/state").json()["run_id"]
    time.sleep(1.1)
    assert client.post("/api/scenario", json={"name": "busy"}).status_code == 200
    deadline = time.time() + 20
    state = client.get("/api/state").json()
    while time.time() < deadline and state["run_id"] == before:
        time.sleep(0.2)
        state = client.get("/api/state").json()
    assert state["run_id"] != before and state["source"]["scenario"] == "busy"
    assert state["t"] < 60


def test_prometheus_metrics_are_exposed(client):
    res = client.get("/metrics")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/plain")
    text = res.text
    assert "# TYPE crowdwatch_zone_people gauge" in text
    assert 'crowdwatch_zone_capacity{zone="atrium"} 60' in text
    assert "# TYPE crowdwatch_door_crossings_total counter" in text
    assert 'crowdwatch_active_alerts{level="critical"}' in text
