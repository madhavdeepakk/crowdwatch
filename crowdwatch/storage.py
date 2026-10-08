"""A small SQLite log of counts and alerts.

Only numbers are stored: no video and no images of people.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .analytics.alerts import Alert

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    run_id TEXT NOT NULL, wall REAL NOT NULL, t REAL NOT NULL, zone_id TEXT NOT NULL,
    count REAL NOT NULL, capacity INTEGER NOT NULL, ratio REAL NOT NULL,
    level INTEGER NOT NULL, forecast REAL, risk INTEGER
);
CREATE INDEX IF NOT EXISTS samples_by_zone ON samples (zone_id, wall);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, zone_id TEXT NOT NULL, zone_name TEXT NOT NULL,
    kind TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL,
    started_t REAL NOT NULL, started_wall REAL NOT NULL, ended_t REAL, ended_wall REAL,
    peak_count REAL, peak_ratio REAL, acknowledged INTEGER NOT NULL DEFAULT 0, outcome TEXT
);
"""


class Storage:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()
        if path is None:
            self._db = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._db.executescript(_SCHEMA)
            # alerts left open by a previous run that was stopped mid-alert
            self._db.execute(
                "UPDATE alerts SET ended_t = started_t, ended_wall = started_wall, "
                "outcome = 'monitoring stopped' WHERE ended_t IS NULL"
            )
            self._db.commit()

    def next_alert_id(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM alerts").fetchone()
        return int(row[0])

    def add_samples(self, run_id: str, wall: float, t: float, zones: list[dict]) -> None:
        rows = [(run_id, wall, t, z["id"], z["count_smooth"], z["capacity"], z["ratio"],
                 z["level"], z["forecast_count"], z["risk"]) for z in zones]
        with self._lock:
            self._db.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            self._db.commit()

    def save_alert(self, run_id: str, alert: Alert) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (alert.id, run_id, alert.zone_id, alert.zone_name, alert.kind, alert.level,
                 alert.message, alert.started_t, alert.started_wall, alert.ended_t,
                 alert.ended_wall, alert.peak_count, alert.peak_ratio,
                 int(alert.acknowledged), alert.outcome),
            )
            self._db.commit()

    def recent_alerts(self, limit: int = 50) -> list[dict]:
        with self._lock:
            cur = self._db.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
            names = [c[0] for c in cur.description]
            rows = cur.fetchall()
        out = []
        for row in rows:
            item = dict(zip(names, row))
            item["acknowledged"] = bool(item["acknowledged"])
            item["active"] = item["ended_t"] is None
            out.append(item)
        return out

    def export_csv(self, run_id: Optional[str] = None) -> str:
        query = "SELECT run_id, wall, t, zone_id, count, capacity, ratio, level, forecast, risk FROM samples"
        args: tuple = ()
        if run_id:
            query += " WHERE run_id = ?"
            args = (run_id,)
        with self._lock:
            rows = self._db.execute(query + " ORDER BY wall", args).fetchall()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["run_id", "wall_time", "video_time_s", "zone", "people", "capacity",
                         "occupancy", "level", "forecast_people", "risk"])
        writer.writerows(rows)
        return buf.getvalue()

    def close(self) -> None:
        with self._lock:
            self._db.close()
