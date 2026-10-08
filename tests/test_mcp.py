"""The MCP server, driven the way an AI assistant drives it: over stdio, by tool name."""

import asyncio
import json
import socket
import sys
import threading
import time

import pytest

mcp = pytest.importorskip("mcp")

from crowdwatch.pipeline import Pipeline  # noqa: E402
from crowdwatch.presets import demo_config  # noqa: E402
from crowdwatch.server.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def live_url():
    import urllib.request

    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    pipeline = Pipeline(demo_config("surge", seed=5, time_scale=20, storage_path=None))
    server = uvicorn.Server(uvicorn.Config(create_app(pipeline), host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if json.load(urllib.request.urlopen(url + "/api/state", timeout=1)).get("zones"):
                break
        except Exception:
            pass
        time.sleep(0.2)
    yield url
    server.should_exit = True
    thread.join(timeout=10)


def _payload(result):
    """A tool result as plain Python, whichever shape the SDK version returns."""
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured is not None:
        return structured.get("result", structured) if isinstance(structured, dict) else structured
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def _session(url, script):
    params = mcp.StdioServerParameters(command=sys.executable,
                                       args=["-m", "crowdwatch.mcp_server", "--url", url])
    if hasattr(mcp, "Client"):                                   # mcp 2.x
        async with mcp.Client(params) as client:
            return await script(client)
    async with mcp.stdio_client(params) as (read, write):        # mcp 1.x
        async with mcp.ClientSession(read, write) as session:
            await session.initialize()
            return await script(session)


def test_an_assistant_can_list_and_call_the_tools(live_url):
    async def script(client):
        tools = await client.list_tools()
        names = {t.name for t in tools.tools}
        status = _payload(await client.call_tool("get_live_status", {}))
        zone = _payload(await client.call_tool("get_zone", {"zone": "atrium"}))
        doors = _payload(await client.call_tool("get_door_counts", {}))
        history = _payload(await client.call_tool("get_zone_history", {"zone": "Food court", "minutes": 5}))
        alerts = _payload(await client.call_tool("list_alerts", {"active_only": True}))
        missing = await client.call_tool("get_zone", {"zone": "car park"})
        return names, status, zone, doors, history, alerts, missing

    names, status, zone, doors, history, alerts, missing = asyncio.run(_session(live_url, script))
    assert names == {"get_live_status", "get_zone", "list_alerts", "get_zone_history",
                     "get_door_counts", "acknowledge_alert"}
    assert status["site"].startswith("Riverside Mall") and len(status["zones"]) == 4
    assert status["zones"][1]["name"] == "Atrium" and status["zones"][1]["capacity"] == 60
    assert zone["name"] == "Atrium" and 0 <= zone["risk_score"] <= 100
    assert {d["door"] for d in doors} == {"West doors", "East doors", "South doors"}
    assert history["zone"] == "Food court"
    assert isinstance(alerts, list)
    assert missing.is_error if hasattr(missing, "is_error") else missing.isError
    assert "The zones are" in missing.content[0].text
