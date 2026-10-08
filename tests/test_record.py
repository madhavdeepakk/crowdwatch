import base64
import json

import numpy as np

from crowdwatch.eval.record import MARGIN_M, UNITS_PER_M, record
from crowdwatch.sources.simulator import SCENARIOS, WORLD_H, WORLD_W, Scenario


def test_recording_holds_one_keyframe_a_second_with_everyone_on_the_floor(monkeypatch):
    monkeypatch.setitem(SCENARIOS, "short", Scenario(
        name="short", title="Short test run", description="Twenty quiet seconds.", duration_s=20))
    data = record("short", seed=3)
    json.dumps(data)                                    # plain JSON, nothing numpy left in it

    n = len(data["t"])
    assert n == 20 and data["t"][0] == 1.0 and data["duration"] == 20.0
    assert [z["id"] for z in data["zones"]] == ["west_hall", "atrium", "food_court", "south_concourse"]
    assert len(data["lines"]) == 3 and len(data["doors"]) == 3
    for zone in data["z"]:
        assert all(len(column) == n for column in zone.values())
        assert all(0 <= level <= 3 for level in zone["level"])
    assert len(data["people"]) == n and max(data["people"]) > 20

    # the packed positions decode to one block per keyframe, inside the recorded area
    packed = np.frombuffer(base64.b64decode(data["pos"]), dtype="<u2")
    cursor, blocks = 0, 0
    while cursor < len(packed):
        count = int(packed[cursor])
        rows = packed[cursor + 1: cursor + 1 + 3 * count].reshape(count, 3)
        x = rows[:, 1] / UNITS_PER_M - MARGIN_M
        y = rows[:, 2] / UNITS_PER_M - MARGIN_M
        assert (x > -1.6).all() and (x < WORLD_W + 1.6).all() and (y > -1.6).all() and (y < WORLD_H + 1.6).all()
        ids = rows[:, 0] & 0x7FFF
        assert len(set(ids.tolist())) == count          # nobody appears twice in a keyframe
        cursor += 1 + 3 * count
        blocks += 1
    assert blocks == n and cursor == len(packed)
    # by the end of the run the tracker has nearly everyone
    tracked = (rows[:, 0] >> 15).mean()
    assert tracked > 0.8
