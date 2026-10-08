"""Command line entry point: ``crowdwatch <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path


def _serve(cfg, config_path, host: str, port: int, open_browser: bool) -> None:
    import uvicorn

    from .pipeline import Pipeline
    from .server.app import create_app

    pipeline = Pipeline(cfg, config_path=config_path)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print(f"\n  CrowdWatch is running: {url}\n  Press Ctrl+C to stop.\n")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(pipeline), host=host, port=port, log_level="warning")


def cmd_demo(args) -> None:
    from .presets import demo_config

    cfg = demo_config(args.scenario, args.seed, args.speed)
    _serve(cfg, None, args.host, args.port, not args.no_browser)


def cmd_run(args) -> None:
    from .config import load_config

    _serve(load_config(args.config), args.config, args.host, args.port, not args.no_browser)


def cmd_video(args) -> None:
    from .config import load_config, save_config
    from .presets import quick_video_config

    stem = Path(str(args.source)).stem if not str(args.source).isdigit() else f"camera{args.source}"
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40] or "camera"
    path = Path(args.config or f"crowdwatch-{stem}.yaml")
    if path.exists():
        cfg = load_config(path)
        print(f"  Using the saved setup in {path}")
    else:
        cfg = quick_video_config(args.source, capacity=args.capacity, model=args.model)
        cfg.density.enabled = args.dense
        save_config(cfg, path)
        print(f"  Created {path}. Zones you draw in the dashboard are saved there.")
    _serve(cfg, str(path), args.host, args.port, not args.no_browser)


def cmd_analyze(args) -> None:
    """Process a recording from start to finish with no dashboard."""
    import csv

    import cv2

    from .config import load_config
    from .detect.yolox import build_detector
    from .engine import Engine
    from .pipeline import build_source
    from .presets import quick_video_config
    from .render import ViewOptions, annotate

    target = Path(args.source)
    if target.suffix.lower() in (".yaml", ".yml"):
        cfg = load_config(target)
    else:
        cfg = quick_video_config(args.source, capacity=args.capacity, model=args.model)
    cfg.source.loop = False
    cfg.storage.path = None
    if getattr(args, "dense", False):
        cfg.density.enabled = True
    source = build_source(cfg, render=bool(args.out))
    density = None
    if cfg.density.enabled and cfg.source.type == "video":
        from .pipeline import DensityWorker

        density = DensityWorker(cfg, lambda *a: None)
    next_density = 0.0
    detector = build_detector(cfg.detector, source.size, seed=cfg.source.seed,
                              min_score=cfg.tracker.low_thresh)
    engine = Engine(cfg, source.size)
    events = []
    engine.alerts.subscribe(lambda e: events.append((engine.t, e.type, e.alert.message)))
    view = ViewOptions(blur=cfg.privacy.blur_people, zones=True)
    writer = None
    rows, peaks, frames, next_row = [], {}, 0, 0.0
    while True:
        frame = source.read()
        if frame is None:
            break
        dets = detector.detect(frame)
        if density is not None and frame.image is not None and frame.t >= next_density:
            next_density = frame.t + cfg.density.every_s
            counts, total = density.estimate(frame.image, list(cfg.zones))
            engine.set_density(frame.t, counts, total)
        snap = engine.step(frame.t, dets.boxes, dets.scores)
        frames += 1
        for z in snap["zones"]:
            peaks[z["name"]] = max(peaks.get(z["name"], 0), z["count"])
        if frame.t >= next_row:
            next_row = frame.t + 1.0
            rows += [(round(frame.t, 1), z["id"], z["count"], z["capacity"], z["ratio"], z["level_name"])
                     for z in snap["zones"]]
        if args.out and frame.image is not None:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cfg.processing.fps, source.size)
            writer.write(annotate(frame.image, engine, snap, view,
                                  compact_markers=cfg.source.type == "simulator"))
        if frames % 50 == 0:
            print(f"\r  {frame.t:7.1f} s of video analysed", end="", file=sys.stderr)
    print(file=sys.stderr)
    if density is not None:
        density.close()
    if writer is not None:
        writer.release()
        print(f"Annotated video: {args.out}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["video_time_s", "zone", "people", "capacity", "occupancy", "level"])
            w.writerows(rows)
        print(f"Counts: {args.csv}")
    print(f"Frames analysed: {frames}")
    for name, peak in peaks.items():
        print(f"  Peak in {name}: {peak} people")
    print(f"Alert changes: {len(events)}")
    for t, kind, message in events:
        print(f"  {int(t // 60):02d}:{int(t % 60):02d}  {kind:10s} {message}")


def cmd_evaluate(args) -> None:
    from .eval.report import evaluate, to_markdown, write_report

    print(f"Running {args.seeds} seeds of each scenario and {args.random} random afternoons. "
          "This takes several minutes ...", file=sys.stderr)
    summary, results = evaluate(args.seeds, args.random, args.workers)
    path = write_report(summary, results, args.out)
    print(to_markdown(summary))
    print(f"Written to {path}", file=sys.stderr)


def cmd_benchmark(args) -> None:
    """Counting error on a labelled image set (ShanghaiTech layout)."""
    import numpy as np

    from .config import DetectorConfig
    from .detect.density import DensityCounter, fuse
    from .detect.yolox import YoloxDetector
    from .eval.benchmark import load_shanghaitech, run_counter, score

    samples = load_shanghaitech(args.dataset, limit=args.limit)[:: args.every]
    truth = np.array([s.count for s in samples])
    print(f"{len(samples)} images, {truth.mean():.0f} people each on average", file=sys.stderr)
    detector = YoloxDetector(DetectorConfig(model=args.model, tiles=tuple(args.tiles)))
    print("Person detector ...", file=sys.stderr)
    detected = run_counter(samples, lambda im: float((detector.detect_image(im).scores >= args.score).sum()), True)
    rows = [score(f"Detector only ({args.model}, tiles {args.tiles[0]}x{args.tiles[1]})", detected, truth)]
    if not args.no_density:
        counter = DensityCounter(args.density)
        print("Density map ...", file=sys.stderr)
        dense = run_counter(samples, counter.count, True)
        rows.append(score(f"Density map only ({args.density})", dense, truth))
        rows.append(score("Hybrid (what the live system uses)",
                          np.array([fuse(a, b) for a, b in zip(detected, dense)]), truth))
    print("| Counter | MAE | RMSE | Bias |\n|---|---|---|---|")
    for row in rows:
        print(row.row())


def cmd_record(args) -> None:
    from .eval.record import record_all

    for path in record_all(args.out, args.seed):
        print(f"{path}  ({path.stat().st_size / 1024:.0f} KB)")


def cmd_train(args) -> None:
    from .eval.train import train

    train(args.train, args.val, args.workers)


def cmd_mcp(args) -> None:
    from .mcp_server import main as mcp_main

    mcp_main(["--url", args.url])


def cmd_fetch(args) -> None:
    if args.model.startswith("dmcount"):
        from .detect.density import ensure_density_model

        print(ensure_density_model(args.model))
        return
    from .detect.yolox import ensure_model

    print(ensure_model(args.model))


def cmd_scenarios(_args) -> None:
    from .sources.simulator import SCENARIOS

    for s in SCENARIOS.values():
        print(f"{s.name:8s} {s.title}\n         {s.description}\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="crowdwatch",
        description="Real-time overcrowding detection for malls and public places.")
    sub = parser.add_subparsers(dest="command", required=True)

    def serving(p):
        p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--no-browser", action="store_true", help="do not open the dashboard automatically")

    p = sub.add_parser("demo", help="run the simulated mall, no camera needed")
    p.add_argument("--scenario", default="surge", help="surge, gradual, busy or wave")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--speed", type=float, default=4.0, help="simulation speed (default 4x)")
    serving(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("video", help="watch a video file, webcam (0) or camera stream URL")
    p.add_argument("source", help="file path, webcam number or rtsp:// address")
    p.add_argument("--capacity", type=int, default=30, help="people the view can safely hold")
    p.add_argument("--model", default="yolox_s", choices=["yolox_s", "yolox_tiny"])
    p.add_argument("--dense", action="store_true",
                   help="also run the density-map model, for packed crowds (needs a one-time conversion)")
    p.add_argument("--config", help="where to keep this camera's setup (default: crowdwatch-<name>.yaml)")
    serving(p)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("run", help="run from a config file")
    p.add_argument("config")
    serving(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="process a recording without the dashboard")
    p.add_argument("source", help="a video file, or a config file (.yaml)")
    p.add_argument("--out", help="write an annotated video here (.mp4)")
    p.add_argument("--csv", help="write per-second counts here (.csv)")
    p.add_argument("--dense", action="store_true", help="also run the density-map model")
    p.add_argument("--capacity", type=int, default=30)
    p.add_argument("--model", default="yolox_s", choices=["yolox_s", "yolox_tiny"])
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("evaluate", help="measure alerts and counts against simulated ground truth")
    p.add_argument("--seeds", type=int, default=10, help="runs of each hand-written scenario")
    p.add_argument("--random", type=int, default=20, help="randomly generated afternoons")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="docs")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark", help="measure counting error on labelled crowd images")
    p.add_argument("dataset", help="a ShanghaiTech split folder, e.g. part_B_final/test_data")
    p.add_argument("--model", default="yolox_s", choices=["yolox_s", "yolox_tiny"])
    p.add_argument("--tiles", type=int, nargs=2, default=[1, 1], metavar=("COLS", "ROWS"))
    p.add_argument("--score", type=float, default=0.2, help="detection confidence to count from")
    p.add_argument("--density", default="dmcount_qnrf", choices=["dmcount_qnrf", "dmcount_shb"])
    p.add_argument("--no-density", action="store_true")
    p.add_argument("--limit", type=int, help="use only the first N images")
    p.add_argument("--every", type=int, default=1, help="use every Nth image")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("record-demo", help="record each scenario for the browser demo page")
    p.add_argument("--out", default="docs/demo-data")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("train-warning", help="retrain the early-warning model on simulated crowds")
    p.add_argument("--train", type=int, default=60, help="number of simulated afternoons to learn from")
    p.add_argument("--val", type=int, default=24, help="number held back to choose the alert threshold")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("mcp", help="let AI assistants query a running CrowdWatch (MCP server)")
    p.add_argument("--url", default="http://localhost:8000", help="address of the running dashboard")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("fetch-model", help="download the person detector ahead of time")
    p.add_argument("model", nargs="?", default="yolox_s",
                   choices=["yolox_s", "yolox_tiny", "dmcount_qnrf", "dmcount_shb"])
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("scenarios", help="list the simulator scenarios")
    p.set_defaults(func=cmd_scenarios)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        args.func(args)
    except KeyboardInterrupt:
        pass
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        sys.exit(f"crowdwatch: {exc}")


if __name__ == "__main__":
    main()
