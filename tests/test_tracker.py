import numpy as np

from crowdwatch.tracking.bytetrack import ByteTracker, iou_matrix

from .conftest import boxes_at


def run(tracker, frames, dt=0.1, score=0.9):
    out = None
    for i, pts in enumerate(frames):
        boxes = boxes_at(pts, size=20)
        out = tracker.update(boxes, np.full(len(boxes), score), i * dt)
    return out


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10.0]])
    b = np.array([[0, 0, 10, 10.0], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert iou_matrix(a, b)[0].tolist() == [1.0, 1 / 3, 0.0]
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_track_is_confirmed_after_min_hits_and_keeps_its_id():
    tracker = ByteTracker()
    seen_ids = []
    for i in range(20):
        tracks = tracker.update(boxes_at([(50 + 3 * i, 50)], 20), np.array([0.9]), i * 0.1)
        seen_ids.append(tracks.ids.tolist())
    assert seen_ids[0] == [] and seen_ids[1] == []          # not yet confirmed
    assert all(ids == seen_ids[5] for ids in seen_ids[2:])   # one stable identity
    assert len(seen_ids[5]) == 1


def test_velocity_estimate_is_in_pixels_per_second():
    tracker = ByteTracker()
    tracks = run(tracker, [[(50 + 4 * i, 100)] for i in range(40)])
    assert np.allclose(tracks.velocity[0], [40, 0], atol=4)


def test_two_people_crossing_keep_separate_ids():
    tracker = ByteTracker()
    ids_start = ids_end = None
    for i in range(60):
        a = (40 + 4 * i, 100)           # left to right
        b = (280 - 4 * i, 130)          # right to left, one lane over
        tracks = tracker.update(boxes_at([a, b], 20), np.array([0.9, 0.9]), i * 0.1)
        if i == 5:
            ids_start = dict(zip(tracks.ids.tolist(), tracks.boxes[:, 0].tolist()))
        ids_end = dict(zip(tracks.ids.tolist(), tracks.boxes[:, 0].tolist()))
    assert set(ids_start) == set(ids_end) and len(ids_end) == 2
    left_first = min(ids_start, key=ids_start.get)
    assert ids_end[left_first] == max(ids_end.values())      # the same person is now on the right


def test_short_occlusion_is_bridged():
    tracker = ByteTracker()
    counts, ids = [], set()
    for i in range(40):
        hidden = 15 <= i < 22                                  # 0.7 s without a detection
        pts = [] if hidden else [(50 + 2 * i, 80)]
        tracks = tracker.update(boxes_at(pts, 20), np.full(len(pts), 0.9), i * 0.1)
        counts.append(len(tracks))
        ids.update(tracks.ids.tolist())
    assert counts[18] == 1          # still counted while hidden
    assert len(ids) == 1            # and the same identity afterwards


def test_long_absence_ends_the_track():
    tracker = ByteTracker()
    run(tracker, [[(50, 50)]] * 10)
    tracks = tracker.update(np.zeros((0, 4)), np.zeros(0), 0.9 + 3.0)
    assert len(tracks) == 0


def test_low_confidence_detections_keep_a_track_alive_but_never_start_one():
    tracker = ByteTracker()
    for i in range(10):
        tracks = tracker.update(boxes_at([(60, 60)], 20), np.array([0.3]), i * 0.1)
    assert len(tracks) == 0                                   # a weak detection alone is ignored
    tracker = ByteTracker()
    for i in range(30):
        score = 0.9 if i < 5 else 0.3                         # the person becomes half hidden
        tracks = tracker.update(boxes_at([(60, 60)], 20), np.array([score]), i * 0.1)
    assert len(tracks) == 1 and tracks.seen[0]


def test_reset_forgets_everything():
    tracker = ByteTracker()
    run(tracker, [[(50, 50)]] * 10)
    tracker.reset()
    assert len(tracker.update(np.zeros((0, 4)), np.zeros(0), 5.0)) == 0
