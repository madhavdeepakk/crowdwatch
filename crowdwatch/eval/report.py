"""Run the evaluation over many simulated afternoons and write up the result."""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

from ..sources.simulator import SCENARIOS, random_scenario
from .harness import LOOKBACK_S, MIN_EVENT_S, RunResult, run_one, summarise

# Which random seeds were used for what. The final report uses seeds that
# nothing was tuned, trained or validated on:
#   1-4, 101-110        the hand-written scenarios, used while tuning thresholds
#   1000-1059           random afternoons the early-warning model was trained on
#   2000-2023           random afternoons used to choose its alert threshold
#   201 onward          hand-written scenarios, final report
#   3000 onward         random afternoons, final report
REPORT_SEED_START = 201
RANDOM_SEED_START = 3000


def _job(args: tuple[str, int]) -> RunResult:
    from ..presets import demo_config

    kind, seed = args
    # Run with the learned model switched on so its warnings are produced end to
    # end. The rule's warnings are replayed from the same run, and the alarms
    # themselves do not depend on the choice.
    cfg = demo_config("surge", seed, storage_path=None)
    cfg.alerts.forecast_method = "learned"
    return run_one(random_scenario(seed) if kind == "random" else kind, seed, cfg)


def evaluate(seeds: int = 10, random_runs: int = 20, workers: int = 2) -> tuple[dict, list[RunResult]]:
    jobs = [(name, REPORT_SEED_START + i) for name in SCENARIOS for i in range(seeds)]
    jobs += [("random", RANDOM_SEED_START + i) for i in range(random_runs)]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(_job, jobs)
    else:
        results = [_job(job) for job in jobs]
    summary = summarise(results)
    groups = {SCENARIOS[name].title: [r for r in results if r.scenario == name] for name in SCENARIOS}
    groups["Random afternoons"] = [r for r in results if r.scenario.startswith("random-")]
    summary["by_scenario"] = {title: summarise(rs) for title, rs in groups.items() if rs}
    summary["seeds"] = {"scripted": [REPORT_SEED_START, REPORT_SEED_START + seeds - 1],
                        "random": [RANDOM_SEED_START, RANDOM_SEED_START + random_runs - 1]}
    return summary, results


def _delay(stats) -> str:
    return "n/a" if not stats else f"{stats['median']:.1f} s"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _warning_row(label: str, w: dict, events: int, prediction: bool = True) -> str:
    lead = w["lead"]
    lead_text = "n/a" if not lead else f"{lead['median']:.0f} s ({lead['p10']:.0f} to {lead['p90']:.0f})"
    row = f"| {label} | {w['events_warned']} of {events} ({_pct(w['recall'])}) | {lead_text} | "
    if not prediction:
        return row + "n/a | n/a |"
    false = _pct(1 - w["precision"]) if w["precision"] is not None else "n/a"
    return row + f"{w['alerts']} | {w['false']} ({false}) |"


def to_markdown(summary: dict) -> str:
    n, o, s = summary["naive"], summary["ours"], summary["smooth"]
    ev = summary["events"]
    doors = summary["doors"]
    door_err = 100 * (doors["counted"] - doors["true"]) / max(doors["true"], 1)
    seeds = summary["seeds"]
    rows = [
        ("Overcrowding events caught", f"{n['detected']} of {ev}", f"{s['detected']} of {ev}", f"{o['detected']} of {ev}"),
        ("Separate alarms raised", str(n["alarms"]), str(s["alarms"]), str(o["alarms"])),
        ("Alarms per real event", f"{n['alarms_per_event']:.1f}", f"{s['alarms_per_event']:.1f}", f"{o['alarms_per_event']:.1f}"),
        ("False alarms", f"{n['false_alarms']} ({n['false_per_hour']:.1f} an hour)",
         f"{s['false_alarms']} ({s['false_per_hour']:.1f} an hour)",
         f"{o['false_alarms']} ({o['false_per_hour']:.1f} an hour)"),
        ("Delay after a zone truly fills (median)", _delay(n["delay"]), _delay(s["delay"]), _delay(o["delay"])),
    ]
    lines = [
        "# Evaluation on simulated crowds",
        "",
        f"{summary['runs']} simulated runs ({summary['simulated_hours']:.1f} hours of mall time): the four "
        f"hand-written scenarios with random seeds {seeds['scripted'][0]} to {seeds['scripted'][1]}, plus "
        f"{seeds['random'][1] - seeds['random'][0] + 1} randomly generated afternoons (seeds "
        f"{seeds['random'][0]} to {seeds['random'][1]}). None of these seeds was used to tune a "
        "threshold or to train or validate the early-warning model.",
        "",
        "**These numbers come from the built-in simulator, not from real camera footage.** They show "
        "that the tracking, counting, forecasting and alert logic behave well under a stated model of "
        "detector noise (missed people, lasting occlusions, jitter, false detections). The "
        "early-warning model was also trained on simulated crowds, so its score here says how well "
        "it generalises to new simulated situations, not to a real building. For accuracy on real "
        "images see [benchmark.md](benchmark.md).",
        "",
        "## Alarms",
        "",
        f"An overcrowding event is a zone at or above capacity for at least {MIN_EVENT_S:.0f} seconds. "
        f"There were {ev} of them.",
        "",
        "| | Per-frame threshold | Threshold on 2 s average | CrowdWatch |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows],
        "",
        "## Early warning",
        "",
        "How many events were announced in advance, how far ahead (median, with the middle 80% "
        f"in brackets), and how many warnings were not followed by an event within "
        f"{LOOKBACK_S / 60:.0f} minutes.",
        "",
        "| Method | Events warned of | Lead time | Warnings raised | Of which false |",
        "|---|---|---|---|---|",
        _warning_row("Damped-trend rule (the default)", summary["early_warning_rule"], ev),
        _warning_row("Learned model (opt-in)", summary["early_warning"], ev),
        _warning_row("Rule, or the 'nearly full' level, whichever comes first",
                     summary["early_warning_any"], ev, prediction=False),
        "",
        "The last row is what an operator actually gets by default: the forecast, backed up by the "
        "'nearly full' level at 80% of capacity. That level is a statement of fact rather than a "
        "prediction, so it has no false-warning figure.",
        "",
        "Both forecasts raise a lot of warnings that come to nothing. Most of the false ones are in "
        "the two scenarios built to provoke them: a zone that hovers just under its limit, and a "
        "crowd that walks straight through.",
        "",
        "## Counting",
        "",
        f"- Mean error of the zone count: {o['count_mae']:.2f} people (per-frame detections alone: {n['count_mae']:.2f}).",
        f"- Door counters: {doors['counted']} crossings counted against {doors['true']} true "
        f"({door_err:+.1f}%).",
        "",
        "## By scenario",
        "",
        "| Scenario | Runs | Events | Per-frame alarms | CrowdWatch alarms (false) | Rule warnings (false) | Model warnings (false) |",
        "|---|---|---|---|---|---|---|",
    ]
    for title, sub in summary["by_scenario"].items():
        lines.append(
            f"| {title} | {sub['runs']} | {sub['events']} | {sub['naive']['alarms']} | "
            f"{sub['ours']['alarms']} ({sub['ours']['false_alarms']}) | "
            f"{sub['early_warning_rule']['alerts']} ({sub['early_warning_rule']['false']}) | "
            f"{sub['early_warning']['alerts']} ({sub['early_warning']['false']}) |"
        )
    lines += ["", "Reproduce with `crowdwatch evaluate`.", ""]
    return "\n".join(lines)


def write_report(summary: dict, results: list[RunResult], out_dir: str | Path = "docs") -> Path:
    from dataclasses import asdict

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.md").write_text(to_markdown(summary), encoding="utf-8")
    runs = []
    for r in results:
        row = asdict(r)
        row.pop("logs", None)
        runs.append(row)
    (out / "evaluation.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=1),
                                         encoding="utf-8")
    return out / "evaluation.md"
