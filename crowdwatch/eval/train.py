"""Train the early-warning model on simulated crowds.

The data comes from randomly generated afternoons (``random_scenario``): the
timing, size and place of every build-up differ from run to run, and so do the
zone capacities. The hand-written demo scenarios are never trained on, so they
stay a fair test.

For every zone and every second the question is the same: *does an
overcrowding event start here within the next two minutes?* The inputs are
only things the live system can see (see ``early_warning.FEATURES``); the
answer comes from the simulator's ground truth.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..analytics.early_warning import DEFAULT_MODEL, FEATURES
from ..config import AlertConfig
from ..presets import demo_config
from ..sources.simulator import ZONES_M, random_scenario
from .harness import (WarningScore, ZoneLog, intervals, replay_predictions, true_events,
                      warning_summary)

LEAD_S = 120.0
WARMUP_S = 30.0
TRAIN_SEEDS = 1000
VAL_SEEDS = 2000


def _collect(seed: int) -> list[ZoneLog]:
    from .harness import run_one

    rng = np.random.default_rng(seed)
    scale = {zid: float(rng.uniform(0.75, 1.35)) for zid in ZONES_M}
    cfg = demo_config("surge", seed, storage_path=None)
    cfg.alerts.forecast_method = "rule"            # never let an older model shape its own training data
    return run_one(random_scenario(seed), seed, cfg, capacity_scale=scale, keep_logs=True).logs


def collect(seeds: list[int], workers: int = 2) -> list[ZoneLog]:
    if workers > 1:
        with Pool(workers) as pool:
            nested = pool.map(_collect, seeds)
    else:
        nested = [_collect(s) for s in seeds]
    return [log for logs in nested for log in logs]


@dataclass
class Labelled:
    x: np.ndarray
    y: np.ndarray


def label(log: ZoneLog, every: int = 10) -> Labelled:
    """One training row per second: will an event start within LEAD_S?"""
    events = true_events(log.truth, log.capacity, log.t)
    spikes = intervals(log.truth >= log.capacity, log.t)       # includes over-capacity blips too short to count
    t = log.t
    y = np.zeros(len(t), dtype=bool)
    usable = (log.level < 3) & (t >= WARMUP_S)
    for start, end in events:
        y |= (t < start) & (start - t <= LEAD_S)
        usable &= ~((t >= start) & (t <= end))
    for start, end in spikes:
        if not any(start >= ev[0] - 1e-6 and end <= ev[1] + 1e-6 for ev in events):
            # a near miss: neither a clean yes nor a clean no, so leave it out of training
            usable &= ~((t >= start - LEAD_S) & (t <= end))
    idx = np.flatnonzero(usable)[::every]
    return Labelled(log.features[idx], y[idx])


def alert_scores(logs: list[ZoneLog], signal: list[np.ndarray], alerts: AlertConfig) -> dict:
    """Score a per-frame yes/no signal at the level that matters: alerts raised."""
    score, n_events = WarningScore(), 0
    for log, flags in zip(logs, signal):
        events = true_events(log.truth, log.capacity, log.t)
        n_events += len(events)
        score.add(replay_predictions(alerts, log.t, log.level, flags), events)
    return warning_summary(score, n_events)


def decision_signals(logs: list[ZoneLog], probabilities: list[np.ndarray], threshold: float,
                     alerts: AlertConfig) -> list[np.ndarray]:
    """The per-frame yes/no the live system computes in "learned" mode."""
    ratio_i, damped_i = FEATURES.index("ratio"), FEATURES.index("damped")
    return [(p >= threshold) & (log.features[:, ratio_i] >= alerts.forecast_min_ratio)
            & (log.features[:, damped_i] >= alerts.critical_ratio) & (log.t >= WARMUP_S)
            for log, p in zip(logs, probabilities)]


def f1(summary: dict) -> float:
    p, r = summary["precision"] or 0.0, summary["recall"] or 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def choose_threshold(logs: list[ZoneLog], probabilities: list[np.ndarray],
                     alerts: AlertConfig) -> tuple[dict, list[dict], dict]:
    """Pick the alert threshold with the best F1 at the level of alerts raised.

    Returns the chosen row, the whole curve, and the rule's score on the same runs.
    """
    curve = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98):
        s = alert_scores(logs, decision_signals(logs, probabilities, threshold, alerts), alerts)
        curve.append({"threshold": threshold, "f1": f1(s), **s})
    rule = alert_scores(logs, [log.rule_predicted for log in logs], alerts)
    rule["f1"] = f1(rule)
    best = max(curve, key=lambda c: (round(c["f1"], 3), (c["lead"] or {}).get("median", 0)))
    return best, curve, rule


def train(n_train: int = 60, n_val: int = 24, workers: int = 2, out: Path = DEFAULT_MODEL,
          verbose: bool = True) -> dict:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score

    say = print if verbose else (lambda *a, **k: None)
    started = time.time()
    say(f"Simulating {n_train} training and {n_val} validation afternoons ...")
    train_logs = collect(list(range(TRAIN_SEEDS, TRAIN_SEEDS + n_train)), workers)
    val_logs = collect(list(range(VAL_SEEDS, VAL_SEEDS + n_val)), workers)
    say(f"  done in {time.time() - started:.0f} s")

    tr = [label(log) for log in train_logs]
    va = [label(log) for log in val_logs]
    x_train, y_train = np.concatenate([d.x for d in tr]), np.concatenate([d.y for d in tr])
    x_val, y_val = np.concatenate([d.x for d in va]), np.concatenate([d.y for d in va])
    say(f"  {len(y_train)} training rows ({y_train.mean():.1%} positive), {len(y_val)} validation rows")

    model = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.06,
                                       subsample=0.7, min_samples_leaf=40, random_state=0)
    weights = np.where(y_train, (~y_train).sum() / max(y_train.sum(), 1), 1.0)
    model.fit(x_train, y_train, sample_weight=weights)
    p_val = model.predict_proba(x_val)[:, 1]
    auc, ap = roc_auc_score(y_val, p_val), average_precision_score(y_val, p_val)
    say(f"  validation ROC AUC {auc:.3f}, average precision {ap:.3f}")

    # Choose the alert threshold on validation, at the level of alerts raised.
    alerts = AlertConfig()
    probs = [model.predict_proba(log.features)[:, 1] for log in val_logs]
    best, curve, rule = choose_threshold(val_logs, probs, alerts)
    say(f"  rule on validation:    precision {rule['precision']:.2f}, recall {rule['recall']:.2f}, F1 {rule['f1']:.2f}")
    say(f"  model at {best['threshold']:.2f}: precision {best['precision']:.2f}, recall {best['recall']:.2f}, F1 {best['f1']:.2f}")
    if best["f1"] <= rule["f1"]:
        say("  the model does not beat the rule on validation; keep alerts.forecast_method on 'rule'")

    importance = permutation_importance(model, x_val, y_val, scoring="average_precision",
                                        n_repeats=3, random_state=0)
    ranked = sorted(zip(FEATURES, importance.importances_mean.tolist()), key=lambda kv: -kv[1])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx_model = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
                                 options={id(model): {"zipmap": False}}, target_opset=15)
    out.write_bytes(onnx_model.SerializeToString())
    meta = {
        "features": FEATURES, "threshold": best["threshold"], "lead_s": LEAD_S,
        "model": "GradientBoostingClassifier, 250 trees of depth 3",
        "trained_on": f"{n_train} random simulated afternoons (seeds {TRAIN_SEEDS}-{TRAIN_SEEDS + n_train - 1})",
        "validated_on": f"{n_val} more (seeds {VAL_SEEDS}-{VAL_SEEDS + n_val - 1})",
        "rows": {"train": int(len(y_train)), "validation": int(len(y_val))},
        "validation": {"roc_auc": float(auc), "average_precision": float(ap),
                       "rule": rule, "chosen": best, "curve": curve},
        "importance": ranked,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    say(f"  saved {out} ({out.stat().st_size / 1024:.0f} KB) in {time.time() - started:.0f} s total")
    return meta
