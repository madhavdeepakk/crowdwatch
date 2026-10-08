# Evaluation on simulated crowds

60 simulated runs (9.3 hours of mall time): the four hand-written scenarios with random seeds 201 to 210, plus 20 randomly generated afternoons (seeds 3000 to 3019). None of these seeds was used to tune a threshold or to train or validate the early-warning model.

**These numbers come from the built-in simulator, not from real camera footage.** They show that the tracking, counting, forecasting and alert logic behave well under a stated model of detector noise (missed people, lasting occlusions, jitter, false detections). The early-warning model was also trained on simulated crowds, so its score here says how well it generalises to new simulated situations, not to a real building. For accuracy on real images see [benchmark.md](benchmark.md).

## Alarms

An overcrowding event is a zone at or above capacity for at least 10 seconds. There were 51 of them.

| | Per-frame threshold | Threshold on 2 s average | CrowdWatch |
|---|---|---|---|
| Overcrowding events caught | 49 of 51 | 44 of 51 | 47 of 51 |
| Separate alarms raised | 2920 | 204 | 47 |
| Alarms per real event | 56.6 | 4.0 | 0.9 |
| False alarms | 31 (3.3 an hour) | 1 (0.1 an hour) | 2 (0.2 an hour) |
| Delay after a zone truly fills (median) | 1.8 s | 8.8 s | 7.3 s |

## Early warning

How many events were announced in advance, how far ahead (median, with the middle 80% in brackets), and how many warnings were not followed by an event within 4 minutes.

| Method | Events warned of | Lead time | Warnings raised | Of which false |
|---|---|---|---|---|
| Damped-trend rule (the default) | 29 of 51 (57%) | 51 s (21 to 110) | 64 | 27 (42%) |
| Learned model (opt-in) | 41 of 51 (80%) | 40 s (13 to 181) | 90 | 42 (47%) |
| Rule, or the 'nearly full' level, whichever comes first | 44 of 51 (86%) | 56 s (29 to 206) | n/a | n/a |

The last row is what an operator actually gets by default: the forecast, backed up by the 'nearly full' level at 80% of capacity. That level is a statement of fact rather than a prediction, so it has no false-warning figure.

Both forecasts raise a lot of warnings that come to nothing. Most of the false ones are in the two scenarios built to provoke them: a zone that hovers just under its limit, and a crowd that walks straight through.

## Counting

- Mean error of the zone count: 0.82 people (per-frame detections alone: 1.35).
- Door counters: 35377 crossings counted against 36978 true (-4.3%).

## By scenario

| Scenario | Runs | Events | Per-frame alarms | CrowdWatch alarms (false) | Rule warnings (false) | Model warnings (false) |
|---|---|---|---|---|---|---|
| Flash sale in the atrium | 10 | 11 | 398 | 10 (0) | 15 (3) | 13 (3) |
| Lunch rush in the food court | 10 | 11 | 1284 | 10 (0) | 10 (2) | 15 (0) |
| A busy afternoon | 10 | 6 | 269 | 4 (0) | 7 (7) | 20 (15) |
| A crowd passes through | 10 | 5 | 89 | 7 (2) | 13 (9) | 13 (13) |
| Random afternoons | 20 | 18 | 880 | 16 (0) | 19 (6) | 29 (11) |

Reproduce with `crowdwatch evaluate`.
