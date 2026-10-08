"""A small crowd simulator: a mall floor seen from a ceiling camera.

It exists for two reasons. First, the whole system can be demonstrated with
nothing plugged in. Second, and more important, it knows exactly how many
people are in each zone at every instant, which no real video does. That
ground truth is what the evaluation measures the detector against.

People arrive through three entrances, visit one to three places (the atrium,
the food court, shop fronts), linger, and leave. They steer around each other
and slow down in a crowd. A scenario is a timetable that changes how many
people arrive and where they want to go.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import Frame, Truth

WORLD_W, WORLD_H = 40.0, 24.0          # metres
PX_PER_M = 32
FRAME_W, FRAME_H = int(WORLD_W * PX_PER_M), int(WORLD_H * PX_PER_M)
WALK_Y = (3.3, 20.7)                   # the concourse between the two rows of shops
PERSON_RADIUS = 0.23

# name -> (spawn point outside the frame, first waypoint just inside)
ENTRANCES = {
    "west": ((-1.0, 12.0), (1.6, 12.0)),
    "east": ((41.0, 16.5), (38.4, 16.5)),
    "south": ((20.0, 25.0), (20.0, 20.0)),
}
ENTRANCE_SPREAD = {"west": (0.0, 2.2), "east": (0.0, 1.8), "south": (1.8, 0.0)}

# places people go, as rectangles (x1, y1, x2, y2) they stand inside
POIS = {
    "atrium": (15.0, 7.5, 25.0, 15.5),
    "food_court": (30.5, 4.6, 38.4, 11.2),
    "west_hall": (2.5, 8.5, 7.5, 15.5),
    "shops_north": (2.5, 3.6, 27.0, 4.8),
    "shops_south": (2.5, 19.2, 14.0, 20.4),
    "shops_east": (26.5, 19.2, 37.5, 20.4),
    "concourse": (16.0, 17.8, 24.0, 20.2),
}

# zones the demo watches, in metres; converted to normalised polygons below
ZONES_M = {
    "west_hall": ("West hall", (1.0, 7.0, 9.0, 17.0), 40),
    "atrium": ("Atrium", (14.0, 6.5, 26.0, 16.5), 60),
    "food_court": ("Food court", (29.5, 3.8, 39.2, 12.0), 45),
    "south_concourse": ("South concourse", (15.0, 17.2, 25.0, 20.8), 24),
}
LINES_M = {
    "west_doors": ("West doors", (2.4, 15.5), (2.4, 8.5)),
    "east_doors": ("East doors", (37.6, 13.6), (37.6, 19.4)),
    "south_doors": ("South doors", (23.0, 20.6), (17.0, 20.6)),
}

_BASE_WEIGHTS = {"atrium": 0.22, "food_court": 0.18, "west_hall": 0.10, "shops_north": 0.22,
                 "shops_south": 0.12, "shops_east": 0.12, "concourse": 0.04}
_BASE_DWELL = {"atrium": 55.0, "food_court": 110.0, "west_hall": 45.0, "shops_north": 30.0,
               "shops_south": 30.0, "shops_east": 30.0, "concourse": 25.0}
_BASE_ENTRANCES = {"west": 0.45, "east": 0.30, "south": 0.25}


@dataclass
class Phase:
    """A stretch of time during which the crowd behaves differently."""

    start: float
    end: float
    arrivals_per_min: Optional[float] = None
    weights: Optional[dict] = None          # overrides for where people want to go
    dwell: Optional[dict] = None            # overrides for how long they stay
    entrances: Optional[dict] = None
    stops: Optional[tuple[int, int]] = None  # min/max number of places visited
    through: Optional[tuple[str, str]] = None  # walk straight from one entrance to another


@dataclass
class Scenario:
    name: str
    title: str
    description: str
    duration_s: float
    arrivals_per_min: float = 26.0
    initial_people: int = 55
    phases: list[Phase] = field(default_factory=list)


SCENARIOS: dict[str, Scenario] = {
    "surge": Scenario(
        name="surge", title="Flash sale in the atrium",
        description="A promotion starts in the atrium at 1:30. Arrivals nearly double and most "
                    "newcomers head straight for it. It winds down after 4:30.",
        duration_s=540,
        phases=[
            Phase(90, 270, arrivals_per_min=48, weights={"atrium": 1.6}, dwell={"atrium": 150.0},
                  stops=(1, 2)),
            Phase(270, 540, arrivals_per_min=18, weights={"atrium": 0.08}, dwell={"atrium": 25.0}),
        ],
    ),
    "gradual": Scenario(
        name="gradual", title="Lunch rush in the food court",
        description="The food court fills slowly over several minutes as lunchtime builds, "
                    "then empties again.",
        duration_s=660,
        phases=[
            Phase(60, 420, arrivals_per_min=30, weights={"food_court": 0.34}, dwell={"food_court": 210.0}),
            Phase(420, 660, arrivals_per_min=20, weights={"food_court": 0.08}, dwell={"food_court": 40.0}),
        ],
    ),
    "busy": Scenario(
        name="busy", title="A busy afternoon",
        description="A steady, busy afternoon with the atrium around two thirds full. Ordinary "
                    "comings and goings occasionally push it to its limit for a short while.",
        duration_s=600, initial_people=95,
        phases=[
            Phase(0, 600, arrivals_per_min=34, weights={"atrium": 0.36}, dwell={"atrium": 105.0}),
        ],
    ),
    "wave": Scenario(
        name="wave", title="A crowd passes through",
        description="A large group arrives at once through the west doors and walks straight "
                    "through to the east exit. Zones spike for a few seconds, then clear.",
        duration_s=420,
        phases=[
            Phase(120, 134, arrivals_per_min=300, through=("west", "east")),
            Phase(260, 272, arrivals_per_min=280, through=("east", "west")),
        ],
    ),
}


def random_scenario(seed: int) -> Scenario:
    """A made-up afternoon: maybe quiet, maybe a surge, a slow build-up or a passing crowd.

    Used to train and test the early-warning model on situations nobody
    scripted by hand, with the timing, size and location all drawn at random.
    """
    rng = np.random.default_rng(100_000 + seed)
    u = rng.uniform
    kind = str(rng.choice(["quiet", "surge", "gradual", "wave", "mixed"], p=[0.22, 0.30, 0.22, 0.10, 0.16]))
    base = float(u(18, 34))
    duration = float(u(480, 660))
    doors = list(ENTRANCES)
    phases: list[Phase] = []

    def target() -> str:
        return str(rng.choice(["atrium", "food_court", "west_hall", "concourse"], p=[0.4, 0.3, 0.2, 0.1]))

    def surge(start: float) -> float:
        poi, length = target(), float(u(80, 220))
        phases.append(Phase(start, start + length, arrivals_per_min=base * float(u(1.3, 2.6)),
                            weights={poi: float(u(0.7, 2.4))}, dwell={poi: float(u(90, 240))},
                            stops=(1, int(rng.integers(1, 3)))))
        phases.append(Phase(start + length, duration, arrivals_per_min=base * float(u(0.6, 1.0)),
                            weights={poi: 0.08}, dwell={poi: float(u(20, 50))}))
        return start + length

    def gradual(start: float) -> float:
        poi, length = target(), float(u(220, 400))
        phases.append(Phase(start, start + length, arrivals_per_min=base * float(u(1.05, 1.5)),
                            weights={poi: float(u(0.3, 0.9))}, dwell={poi: float(u(150, 300))}))
        phases.append(Phase(start + length, duration, arrivals_per_min=base * float(u(0.6, 0.9)),
                            weights={poi: 0.08}, dwell={poi: float(u(30, 60))}))
        return start + length

    def wave(start: float) -> float:
        a, b = rng.choice(len(doors), size=2, replace=False)
        length = float(u(8, 16))
        phases.append(Phase(start, start + length, arrivals_per_min=float(u(180, 330)),
                            through=(doors[a], doors[b])))
        return start + length

    if kind == "quiet":
        poi = target()
        phases.append(Phase(0, duration, arrivals_per_min=base * float(u(0.9, 1.4)),
                            weights={poi: float(u(0.15, 0.5))}, dwell={poi: float(u(60, 140))}))
    elif kind == "surge":
        surge(float(u(40, 280)))
    elif kind == "gradual":
        gradual(float(u(30, 150)))
    elif kind == "wave":
        end = wave(float(u(60, 200)))
        if rng.random() < 0.6:
            wave(end + float(u(60, 200)))
    else:
        end = wave(float(u(40, 120))) if rng.random() < 0.5 else surge(float(u(40, 120)))
        (surge if rng.random() < 0.6 else gradual)(min(end + float(u(30, 120)), duration - 150))
    return Scenario(name=f"random-{seed}", title=f"Random afternoon {seed} ({kind})",
                    description=f"A randomly generated {kind} scenario.", duration_s=duration,
                    arrivals_per_min=base, initial_people=int(u(40, 90)), phases=phases)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


class World:
    """The simulated crowd. Positions are in metres, time in seconds."""

    WARMUP_S = 150.0

    def __init__(self, scenario: Scenario, seed: int = 7):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self._next_id = 1
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.target = np.zeros((0, 2))
        self.v0 = np.zeros(0)
        self.ids = np.zeros(0, dtype=int)
        self.state = np.zeros(0, dtype=int)        # 0 walking, 1 standing, 2 leaving
        self.until = np.zeros(0)                   # when a standing person moves on
        self.shade = np.zeros(0, dtype=int)
        self.heading = np.zeros((0, 2))
        self.plans: list[list] = []                # remaining steps for each person
        self.neighbours = np.zeros(0, dtype=int)
        self.entered = {door: 0 for door in ENTRANCES}   # true door counts, for evaluation
        self.exited = {door: 0 for door in ENTRANCES}
        self._seed_initial(scenario.initial_people)
        # Let the crowd settle before the clock starts, so a run opens on a
        # mall in full swing rather than on people frozen in place.
        self.t = -self.WARMUP_S
        while self.t < -1e-9:
            self.step(0.2)
        self.t = 0.0
        self.entered = {door: 0 for door in ENTRANCES}
        self.exited = {door: 0 for door in ENTRANCES}

    # ----------------------------------------------------------- timetable

    def _phase_values(self):
        s = self.scenario
        rate, weights, dwell = s.arrivals_per_min, dict(_BASE_WEIGHTS), dict(_BASE_DWELL)
        entrances, stops, through = dict(_BASE_ENTRANCES), (1, 3), None
        for ph in s.phases:
            if ph.start <= self.t < ph.end:
                rate = ph.arrivals_per_min if ph.arrivals_per_min is not None else rate
                weights.update(ph.weights or {})
                dwell.update(ph.dwell or {})
                entrances.update(ph.entrances or {})
                stops = ph.stops or stops
                through = ph.through or through
        return rate, weights, dwell, entrances, stops, through

    # ------------------------------------------------------------- people

    def _spot(self, poi: str) -> np.ndarray:
        x1, y1, x2, y2 = POIS[poi]
        return np.array([self.rng.uniform(x1, x2), self.rng.uniform(y1, y2)])

    def _door(self, name: str, inside: bool) -> np.ndarray:
        outside_pt, inside_pt = ENTRANCES[name]
        sx, sy = ENTRANCE_SPREAD[name]
        base = np.array(inside_pt if inside else outside_pt)
        return base + np.array([self.rng.uniform(-sx, sx), self.rng.uniform(-sy, sy)])

    def _pick(self, table: dict) -> str:
        names = list(table)
        p = np.array([max(table[n], 0.0) for n in names])
        return names[self.rng.choice(len(names), p=p / p.sum())]

    def _plan_visit(self, weights, dwell, stops, exit_door: Optional[str] = None) -> list:
        plan = []
        for _ in range(int(self.rng.integers(stops[0], stops[1] + 1))):
            poi = self._pick(weights)
            stay = float(self.rng.gamma(4.0, dwell[poi] / 4.0))
            plan.append(("visit", self._spot(poi), stay))
        door = exit_door or self._pick(_BASE_ENTRANCES)
        plan.append(("walk", self._door(door, inside=True)))
        plan.append(("leave", self._door(door, inside=False), door))
        return plan

    def _add(self, pos, plan, state=0, until=0.0):
        self.pos = np.vstack([self.pos, pos])
        self.vel = np.vstack([self.vel, np.zeros(2)])
        self.heading = np.vstack([self.heading, self.rng.normal(size=2)])
        self.target = np.vstack([self.target, pos])
        self.v0 = np.append(self.v0, float(np.clip(self.rng.normal(1.25, 0.18), 0.8, 1.7)))
        self.ids = np.append(self.ids, self._next_id)
        self._next_id += 1
        self.state = np.append(self.state, state)
        self.until = np.append(self.until, until)
        self.shade = np.append(self.shade, int(self.rng.integers(0, len(_CLOTHES))))
        self.neighbours = np.append(self.neighbours, 0)
        self.plans.append(plan)
        if state == 0:
            self._advance(len(self.ids) - 1)

    def _seed_initial(self, n: int) -> None:
        """Start with the mall already occupied rather than empty."""
        _, weights, dwell, _, stops, _ = self._phase_values()
        for _ in range(n):
            plan = self._plan_visit(weights, dwell, stops)
            _, spot, stay = plan.pop(0)
            self._add(spot, plan, state=1, until=float(self.rng.uniform(0, stay)))

    def _spawn(self, dt: float) -> None:
        rate, weights, dwell, entrances, stops, through = self._phase_values()
        for _ in range(int(self.rng.poisson(rate / 60.0 * dt))):
            if through is not None:
                door, exit_door = through
                plan = [("walk", self._door(door, inside=True)),
                        ("walk", self._door(exit_door, inside=True)),
                        ("leave", self._door(exit_door, inside=False), exit_door)]
            else:
                door = self._pick(entrances)
                plan = [("walk", self._door(door, inside=True))] + self._plan_visit(weights, dwell, stops)
            self._add(self._door(door, inside=False), plan)
            self.entered[door] += 1

    def _advance(self, i: int) -> None:
        """Send person ``i`` towards the next step of their plan."""
        if not self.plans[i]:
            self.state[i] = 3                        # done, will be removed
            return
        step = self.plans[i][0]
        self.target[i] = step[1]
        self.state[i] = 2 if step[0] == "leave" else 0

    # --------------------------------------------------------------- step

    def step(self, dt: float) -> None:
        self.t += dt
        self._spawn(dt)
        n = len(self.ids)
        if n == 0:
            return

        to_target = self.target - self.pos
        dist = np.linalg.norm(to_target, axis=1)

        # Who is close to whom. Used for steering, for slowing down in a
        # crowd, and by the detector model to decide who is hidden.
        delta = self.pos[:, None, :] - self.pos[None, :, :]
        gap = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(gap, np.inf)
        self.neighbours = (gap < 0.55).sum(axis=1)
        around = (gap < 1.1).sum(axis=1)

        # Arrivals at a waypoint, a standing spot (or as near as the crowd allows), or a door.
        moving = self.state != 1
        arrived = moving & ((dist < 0.4) | ((dist < 1.6) & (around >= 5) & (self.state == 0)))
        for i in np.flatnonzero(arrived):
            step = self.plans[i].pop(0)
            if step[0] == "visit":
                self.state[i] = 1
                self.until[i] = self.t + step[2]
                self.target[i] = self.pos[i]
            else:
                if step[0] == "leave":
                    self.exited[step[2]] += 1
                self._advance(i)
        for i in np.flatnonzero((self.state == 1) & (self.until <= self.t)):
            self._advance(i)

        to_target = self.target - self.pos
        dist = np.linalg.norm(to_target, axis=1)
        standing = self.state == 1
        desired = _unit(to_target) * self.v0[:, None]
        desired[standing] = to_target[standing] * 0.6
        slow = np.clip(1.0 - 0.11 * (around - 2), 0.3, 1.0)
        desired[~standing] *= slow[~standing, None]

        # Push apart people who are closer than a comfortable 0.7 m.
        reach = 0.7
        push = np.clip(reach - gap, 0.0, None) / reach
        repulsion = (delta / np.maximum(gap, 1e-6)[:, :, None] * push[:, :, None]).sum(axis=1) * 1.6

        velocity = desired + repulsion
        speed = np.linalg.norm(velocity, axis=1)
        velocity *= np.minimum(1.0, 1.9 / np.maximum(speed, 1e-9))[:, None]
        self.vel = 0.55 * self.vel + 0.45 * velocity
        self.pos = self.pos + self.vel * dt

        # Keep people on the concourse unless they are passing through a doorway.
        inside_x = (self.pos[:, 0] > 0.6) & (self.pos[:, 0] < WORLD_W - 0.6)
        at_south_door = (np.abs(self.pos[:, 0] - 20.0) < 2.4) & (self.pos[:, 1] > WALK_Y[1] - 0.2)
        confine = inside_x & ~at_south_door
        self.pos[confine, 1] = np.clip(self.pos[confine, 1], WALK_Y[0], WALK_Y[1])

        walking = np.linalg.norm(self.vel, axis=1) > 0.25
        self.heading[walking] = 0.7 * self.heading[walking] + 0.3 * self.vel[walking]

        gone = self.state == 3
        if gone.any():
            keep = ~gone
            for name in ("pos", "vel", "target", "v0", "ids", "state", "until", "shade",
                         "heading", "neighbours"):
                setattr(self, name, getattr(self, name)[keep])
            self.plans = [p for p, k in zip(self.plans, keep) if k]

    # -------------------------------------------------------------- truth

    def visible(self) -> np.ndarray:
        p = self.pos
        return (p[:, 0] > 0.2) & (p[:, 0] < WORLD_W - 0.2) & (p[:, 1] > 0.2) & (p[:, 1] < WORLD_H - 0.2)

    def truth(self) -> Truth:
        v = self.visible()
        return Truth(ids=self.ids[v].copy(), positions=self.pos[v] * PX_PER_M,
                     crowding=self.neighbours[v].copy())

    def count_in(self, rect_m: tuple[float, float, float, float]) -> int:
        x1, y1, x2, y2 = rect_m
        p = self.pos
        return int(((p[:, 0] >= x1) & (p[:, 0] <= x2) & (p[:, 1] >= y1) & (p[:, 1] <= y2)).sum())


# ------------------------------------------------------------------ drawing

_CLOTHES = [  # BGR
    (92, 74, 58), (140, 98, 62), (70, 86, 150), (96, 128, 88), (150, 140, 120),
    (60, 60, 70), (120, 96, 150), (88, 132, 170), (170, 150, 96), (110, 110, 118),
]
_SHOPS_NORTH = ["Bookshop", "Pharmacy", "Shoes", "Electronics", "Toys", "Fashion"]
_SHOPS_SOUTH = ["Bakery", "Opticians", "Gifts", "", "Sports", "Home", "Phones"]


def draw_floor() -> np.ndarray:
    import cv2

    s = PX_PER_M
    img = np.full((FRAME_H, FRAME_W, 3), (226, 229, 232), dtype=np.uint8)
    # floor tiles
    for x in range(0, FRAME_W, 2 * s):
        cv2.line(img, (x, 0), (x, FRAME_H), (218, 221, 225), 1)
    for y in range(0, FRAME_H, 2 * s):
        cv2.line(img, (0, y), (FRAME_W, y), (218, 221, 225), 1)

    def block(x1, y1, x2, y2, label=""):
        p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
        cv2.rectangle(img, p1, p2, (176, 170, 164), -1)
        cv2.rectangle(img, p1, p2, (150, 144, 138), 1)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.putText(img, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (106, 100, 96), 1, cv2.LINE_AA)

    edges = np.linspace(0, WORLD_W, len(_SHOPS_NORTH) + 1)
    for name, x1, x2 in zip(_SHOPS_NORTH, edges[:-1], edges[1:]):
        block(x1 + 0.08, 0, x2 - 0.08, 3.0, name)
    edges = [0, 6, 12, 17.5, 22.5, 28, 34, 40]
    for name, x1, x2 in zip(_SHOPS_SOUTH, edges[:-1], edges[1:]):
        if name:
            block(x1 + 0.08, 21.0, x2 - 0.08, 24.0, name)

    # atrium feature, food court tables, benches
    cv2.circle(img, (int(20 * s), int(11.5 * s)), int(4.6 * s), (214, 216, 219), -1)
    cv2.circle(img, (int(20 * s), int(11.5 * s)), int(4.6 * s), (200, 202, 206), 2, cv2.LINE_AA)
    cv2.circle(img, (int(20 * s), int(11.5 * s)), int(0.9 * s), (168, 178, 170), -1, cv2.LINE_AA)
    for tx in np.arange(31.0, 38.6, 1.9):
        for ty in np.arange(5.0, 11.4, 1.6):
            c = (int(tx * s), int(ty * s))
            cv2.rectangle(img, (c[0] - 9, c[1] - 9), (c[0] + 9, c[1] + 9), (196, 190, 180), -1)
    for by in (9.5, 14.5):
        cv2.rectangle(img, (int(3.5 * s), int(by * s) - 6), (int(6.5 * s), int(by * s) + 6), (170, 160, 150), -1)

    # door mats
    cv2.rectangle(img, (0, int(9.6 * s)), (int(0.5 * s), int(14.4 * s)), (120, 126, 134), -1)
    cv2.rectangle(img, (FRAME_W - int(0.5 * s), int(14.5 * s)), (FRAME_W, int(18.5 * s)), (120, 126, 134), -1)
    cv2.rectangle(img, (int(17.6 * s), FRAME_H - int(0.5 * s)), (int(22.4 * s), FRAME_H), (120, 126, 134), -1)
    return img


def draw_people(img: np.ndarray, world: World) -> None:
    import cv2

    s = PX_PER_M
    order = np.argsort(world.pos[:, 1])
    for i in order:
        x, y = world.pos[i]
        if not (-1 < x < WORLD_W + 1 and -1 < y < WORLD_H + 1):
            continue
        c = (int(round(x * s)), int(round(y * s)))
        hx, hy = world.heading[i]
        angle = float(np.degrees(np.arctan2(hy, hx))) + 90.0
        cv2.ellipse(img, (c[0] + 2, c[1] + 3), (9, 6), angle, 0, 360, (196, 199, 203), -1, cv2.LINE_AA)
        cv2.ellipse(img, c, (9, 5), angle, 0, 360, _CLOTHES[world.shade[i]], -1, cv2.LINE_AA)
        cv2.circle(img, c, 4, (58, 52, 50) if world.shade[i] % 3 else (96, 120, 150), -1, cv2.LINE_AA)


# ------------------------------------------------------------------- source

def demo_layout() -> dict:
    """Zones, counting lines and calibration for the simulated mall, as config data."""
    zones = []
    for zid, (name, (x1, y1, x2, y2), capacity) in ZONES_M.items():
        poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        zones.append({
            "id": zid, "name": name, "capacity": capacity,
            "walkway": zid in ("west_hall", "south_concourse"),
            "polygon": [(round(x / WORLD_W, 4), round(y / WORLD_H, 4)) for x, y in poly],
        })
    lines = [
        {"id": lid, "name": name,
         "a": (round(a[0] / WORLD_W, 4), round(a[1] / WORLD_H, 4)),
         "b": (round(b[0] / WORLD_W, 4), round(b[1] / WORLD_H, 4))}
        for lid, (name, a, b) in LINES_M.items()
    ]
    calibration = {
        "image_points": [(0, 0), (1, 0), (1, 1), (0, 1)],
        "floor_points": [(0, 0), (WORLD_W, 0), (WORLD_W, WORLD_H), (0, WORLD_H)],
    }
    return {"zones": zones, "lines": lines, "calibration": calibration}


class SimulatorSource:
    live = False

    def __init__(self, scenario: "str | Scenario" = "surge", seed: int = 7, fps: float = 10.0,
                 time_scale: float = 4.0, loop: bool = True, render: bool = True):
        if isinstance(scenario, Scenario):
            self.scenario = scenario
        elif scenario in SCENARIOS:
            self.scenario = SCENARIOS[scenario]
        else:
            raise ValueError(f"unknown scenario '{scenario}'. Choose from: {', '.join(SCENARIOS)}")
        self.seed = seed
        self.dt = 1.0 / fps
        self.speed = time_scale
        self.loop = loop
        self.render = render
        self.size = (FRAME_W, FRAME_H)
        self.world = World(self.scenario, seed)
        self._floor = draw_floor() if render else None
        self.finished = False

    def read(self) -> Optional[Frame]:
        if self.world.t >= self.scenario.duration_s:
            self.finished = True
            return None
        self.world.step(self.dt)
        image = None
        if self.render:
            image = self._floor.copy()
            draw_people(image, self.world)
        return Frame(t=self.world.t, image=image, truth=self.world.truth())

    def true_counts(self) -> dict[str, int]:
        return {zid: self.world.count_in(rect) for zid, (_, rect, _) in ZONES_M.items()}

    def close(self) -> None:
        pass
