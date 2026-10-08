# Counting accuracy on real images

The simulator tests the logic. This page tests the eyes: how many people does each counter
find in real photographs whose crowds were counted by hand?

## Dataset

ShanghaiTech Part B (Zhang et al., "Single-Image Crowd Counting via Multi-Column Convolutional
Neural Network", CVPR 2016), test split: 316 street
scenes from Shanghai with every head marked, between 9 and 539 people per image (124 on
average). It is a standard benchmark in crowd counting. The dataset is not bundled with this
project; it is free for research use.

Nothing here was trained on it. The person detector was trained on COCO and the density model
on UCF-QNRF, so every number below is for a model looking at this kind of scene for the first
time, which is the honest stand-in for pointing the system at a new camera.

## Results

MAE is the mean absolute error in people per image, RMSE punishes large misses, and bias shows
whether the counter runs low (negative) or high. The last three columns are the MAE for images
with few, some and many people.

### All 316 test images (detector only)

| Counter | MAE | RMSE | Bias | Up to 50 people | 51 to 150 | More than 150 |
|---|---|---|---|---|---|---|
| Detector only, YOLOX-s | 74.8 | 114.2 | -73.8 | 5.9 | 39.2 | 177.1 |
| Detector only, YOLOX-s with 2x2 tiling | 49.9 | 85.4 | -40.0 | 14.6 | 17.7 | 125.6 |

### Every third test image (106 images), all counters

The density model takes several seconds per image on the two-core machine this was run on, so
it was run on every third test image rather than all 316. The detector rows are repeated on the
same 106 images so the comparison is like for like.

| Counter | MAE | RMSE | Bias | Up to 50 people | 51 to 150 | More than 150 |
|---|---|---|---|---|---|---|
| Detector only, YOLOX-s | 70.8 | 104.2 | -69.8 | 6.7 | 43.1 | 165.1 |
| Detector only, YOLOX-s with 2x2 tiling | 45.2 | 75.6 | -35.6 | 15.2 | 18.9 | 114.5 |
| Density map only (DM-Count, QNRF weights) | 13.4 | 26.3 | -9.7 | 1.7 | 8.3 | 30.6 |
| Detector at the live tracker's confidence (0.5) | 100.3 | 130.9 | -100.3 | 18.8 | 73.1 | 205.5 |
| Hybrid, default settings | 13.4 | 26.3 | -9.7 | 1.8 | 8.3 | 30.6 |

## What this shows

- **A box detector alone is not enough for a dense crowd.** YOLOX finds people one at a time,
  and in a packed scene most of them are small or hidden. It is close on sparse images (about 6
  people out on images of up to 50) and badly low on dense ones (177 out on images of more than
  150). At the confidence the live tracker uses, it is lower still.
- **Tiling helps the detector** by showing it distant people at higher resolution: the error
  drops by a third. It also makes it over-count in sparse scenes, and it costs four more model
  runs per frame.
- **The density map is far closer** in every size band, and the hybrid matches it: on these
  images the density estimate takes over in all but the emptiest scenes. The hybrid exists for
  the opposite case, a quiet floor, where the tracker is exact and also provides direction,
  speed and dwell time.
- **It still runs low on the densest images** (30 out on images of more than 150 people, about
  13% under). The
  DM-Count paper reports an MAE of 7.4 on this benchmark with weights trained on its own
  training split; the 13.4 here is with weights that have never seen it. Fine-tuning on footage
  from your own cameras is the way to close that gap.

## How the settings were chosen

- The detector's counting threshold (confidence 0.1) was chosen on 100 images of the *training*
  split and then applied to the test split unchanged.
- The hybrid rule uses its default settings (the density estimate takes over when it is at
  least 15 people and at least 1.2 times the tracked count). These were tried against 34
  training images, where every setting gave nearly the same error, and not adjusted on the
  test images.

## Reproduce

```bash
pip install torch          # once, to convert the density model
crowdwatch benchmark path/to/part_B_final/test_data --score 0.1
crowdwatch benchmark path/to/part_B_final/test_data --score 0.1 --tiles 2 2 --every 3
```
