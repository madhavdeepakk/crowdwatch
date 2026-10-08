"""CrowdWatch as a Model Context Protocol (MCP) server.

MCP is the open standard AI assistants use to call external tools. With this
server connected, an assistant such as Claude can answer questions like
"which zone is busiest right now?", "has the atrium been over capacity this
afternoon?" or "acknowledge the food court alert" by calling the tools below
against a running CrowdWatch instance.

It is a thin, read-mostly layer over CrowdWatch's HTTP API, so it can run on a
different machine from the cameras:

    crowdwatch demo                                  # in one terminal
    crowdwatch mcp --url http://localhost:8000       # the MCP server (stdio)

To use it from Claude Desktop, add this to its MCP configuration:

    {"mcpServers": {"crowdwatch": {"command": "crowdwatch",
                                   "args": ["mcp", "--url", "http://localhost:8000"]}}}
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional


class Api:
    """Minimal client for the CrowdWatch HTTP API."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def call(self, path: str, method: str = "GET", body: Optional[dict] = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail", exc.reason)
            except Exception:
                detail = exc.reason
            raise _tool_error(f"CrowdWatch refused the request: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise _tool_error(
                f"CrowdWatch is not reachable at {self.base}. Is it running? ({exc})"
            ) from exc


def _zone_summary(z: dict) -> dict:
    out = {
        "id": z["id"], "name": z["name"], "people": z["count"], "capacity": z["capacity"],
        "occupancy_percent": round(z["ratio"] * 100), "level": z["level_name"],
        "trend_people_per_minute": z["trend_per_min"],
        "people_heading_this_way": z.get("approaching", 0),
        "early_warning": z.get("predicted", False),
    }
    if z.get("probability") is not None:
        out["chance_of_filling_within_2_min"] = z["probability"]
    if z.get("eta_s") is not None:
        out["estimated_seconds_until_full"] = z["eta_s"]
    if z.get("density") is not None:
        out["people_per_square_metre"] = z["density"]
    return out


def _tool_error(message: str) -> Exception:
    """An error whose message is passed on to the assistant, so it can correct itself."""
    for module in ("mcp.server.mcpserver.exceptions", "mcp.server.fastmcp.exceptions"):
        try:
            return __import__(module, fromlist=["ToolError"]).ToolError(message)
        except (ImportError, AttributeError):
            continue
    return ValueError(message)


def _find_zone(zones: list[dict], wanted: str) -> dict:
    key = wanted.strip().lower()
    for z in zones:
        if key in (z["id"].lower(), z["name"].lower()):
            return z
    partial = [z for z in zones if key in z["name"].lower() or key in z["id"].lower()]
    if len(partial) == 1:
        return partial[0]
    names = ", ".join(z["name"] for z in zones)
    raise _tool_error(f"No zone called '{wanted}'. The zones are: {names}.")


def build_server(base_url: str = "http://localhost:8000"):
    try:
        from mcp.server.mcpserver import MCPServer as Server      # mcp 2.x
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server          # mcp 1.x

    api = Api(base_url)
    server = Server(
        "crowdwatch",
        instructions=(
            "Live crowd monitoring for a venue. Zones are named areas with a safe capacity. "
            "Levels are normal, busy (60% full), warning (80%) and critical (at or over "
            "capacity). Call get_live_status first for an overview."
        ),
    )

    @server.tool()
    def get_live_status() -> dict:
        """Current picture of the whole venue: overall status and headline, how many people
        are in view, and for every zone its head count, capacity, occupancy, alert level,
        trend and early-warning state. Start here."""
        s = api.call("/api/state")
        return {
            "site": s.get("name"), "time": s.get("clock"), "people_in_view": s["people"],
            "overall_level": s["status"]["level_name"], "headline": s["status"]["headline"],
            "zones": [_zone_summary(z) for z in s["zones"]],
            "active_alerts": len(s["alerts"]),
        }

    @server.tool()
    def get_zone(zone: str) -> dict:
        """Everything known about one zone right now. `zone` is its name or id, for example
        "Atrium". Includes walking speed, average stay, and the risk score (0 to 100)."""
        s = api.call("/api/state")
        z = _find_zone(s["zones"], zone)
        detail = _zone_summary(z)
        detail.update(walking_speed_m_per_s=z["speed_mps"], average_stay_seconds=z["dwell_s"],
                      crowd_stalled=z["congested"], risk_score=z["risk"])
        detail["alerts"] = [a["message"] for a in s["alerts"] if a["zone_id"] == z["id"]]
        return detail

    @server.tool()
    def list_alerts(active_only: bool = False, limit: int = 20) -> list[dict]:
        """Alerts, newest first. Each has an id, the zone, kind ("overcrowding" or
        "predicted" for an early warning), level, the message shown to operators, whether
        it is still active, whether someone acknowledged it, and the peak head count."""
        if active_only:
            rows = api.call("/api/state")["alerts"]
        else:
            rows = api.call(f"/api/alerts?limit={max(1, min(int(limit), 200))}")["alerts"]
        return [{
            "id": a["id"], "zone": a["zone_name"], "kind": a["kind"], "level": a["level"],
            "message": a["message"], "active": a["active"], "acknowledged": a["acknowledged"],
            "peak_people": round(a["peak_count"]), "peak_occupancy_percent": round(a["peak_ratio"] * 100),
            "started_at_video_time_s": a["started_t"], "ended_at_video_time_s": a["ended_t"],
            "how_it_ended": a.get("outcome"),
        } for a in rows[:limit]]

    @server.tool()
    def get_zone_history(zone: str, minutes: float = 10) -> dict:
        """How a zone's head count changed over the last `minutes` (up to 60): lowest,
        highest and average count, time spent at or over capacity, and a sample every 30
        seconds. Use it for questions about the past, such as peaks or how long a zone
        was full."""
        s = api.call("/api/state")
        z = _find_zone(s["zones"], zone)
        seconds = max(30.0, min(float(minutes), 60.0) * 60.0)
        rows = api.call(f"/api/history?seconds={seconds}")["zones"].get(z["id"], [])
        if not rows:
            return {"zone": z["name"], "note": "No history recorded yet."}
        counts = [r[1] for r in rows]
        over = sum(1 for r in rows if r[2] >= 1.0)
        return {
            "zone": z["name"], "capacity": z["capacity"], "seconds_covered": round(rows[-1][0] - rows[0][0]),
            "lowest": round(min(counts)), "highest": round(max(counts)),
            "average": round(sum(counts) / len(counts), 1),
            "seconds_at_or_over_capacity": over,
            "samples": [{"video_time_s": r[0], "people": round(r[1])} for r in rows[::30]],
        }

    @server.tool()
    def get_door_counts() -> list[dict]:
        """People counted in and out at each door or counting line since monitoring began."""
        return [{"door": d["name"], "in": d["in"], "out": d["out"], "net": d["net"]}
                for d in api.call("/api/state")["lines"]]

    @server.tool()
    def acknowledge_alert(alert_id: int) -> str:
        """Mark an active alert as seen by an operator. This changes what the control-room
        dashboard shows, so only do it when the person has asked for it. Get the id from
        list_alerts."""
        api.call(f"/api/alerts/{int(alert_id)}/ack", method="POST", body={})
        return f"Alert {alert_id} is now marked as acknowledged."

    return server


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CrowdWatch MCP server (stdio)")
    parser.add_argument("--url", default="http://localhost:8000", help="address of a running CrowdWatch")
    args = parser.parse_args(argv)
    try:
        server = build_server(args.url)
    except ImportError:
        raise SystemExit('The MCP server needs the "mcp" package: pip install "crowdwatch[mcp]"')
    server.run()


if __name__ == "__main__":
    main()
