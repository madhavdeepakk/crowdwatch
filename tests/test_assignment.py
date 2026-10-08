import numpy as np
import pytest

from crowdwatch.tracking.assignment import linear_sum_assignment


def test_small_case_by_hand():
    cost = np.array([[4.0, 1.0, 3.0],
                     [2.0, 0.0, 5.0],
                     [3.0, 2.0, 2.0]])
    rows, cols = linear_sum_assignment(cost)
    assert rows.tolist() == [0, 1, 2] and cols.tolist() == [1, 0, 2]      # 1 + 2 + 2 = 5
    assert cost[rows, cols].sum() == 5


def test_maximize_and_empty_and_bad_input():
    iou = np.array([[0.9, 0.1], [0.8, 0.7]])
    rows, cols = linear_sum_assignment(iou, maximize=True)
    assert cols.tolist() == [0, 1]                    # 0.9 + 0.7 beats 0.1 + 0.8
    rows, cols = linear_sum_assignment(np.zeros((0, 3)))
    assert rows.size == 0 and cols.size == 0
    with pytest.raises(ValueError):
        linear_sum_assignment(np.array([[1.0, np.inf]]))
    with pytest.raises(ValueError):
        linear_sum_assignment(np.zeros(3))


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (7, 3), (12, 12), (40, 55), (55, 40)])
def test_each_side_is_used_once_and_the_smaller_side_fully(shape):
    rng = np.random.default_rng(sum(shape))
    cost = rng.random(shape)
    rows, cols = linear_sum_assignment(cost)
    k = min(shape)
    assert len(rows) == len(cols) == k
    assert len(set(rows.tolist())) == k and len(set(cols.tolist())) == k
    assert rows.tolist() == sorted(rows.tolist())


def test_matches_brute_force_on_small_problems():
    from itertools import permutations

    rng = np.random.default_rng(1)
    for _ in range(40):
        n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        cost = rng.random((n, m)).round(2)
        rows, cols = linear_sum_assignment(cost)
        if n <= m:
            best = min(sum(cost[i, p[i]] for i in range(n)) for p in permutations(range(m), n))
        else:
            best = min(sum(cost[p[j], j] for j in range(m)) for p in permutations(range(n), m))
        assert cost[rows, cols].sum() == pytest.approx(best)


def test_agrees_with_scipy_when_it_is_available():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(2)
    for shape in [(30, 30), (80, 120), (120, 80), (200, 200)]:
        cost = rng.random(shape)
        cost[rng.random(shape) < 0.7] = 0.0            # sparse, with ties, like IoU matrices
        ours = cost[linear_sum_assignment(cost, maximize=True)].sum()
        theirs = cost[scipy_optimize.linear_sum_assignment(cost, maximize=True)].sum()
        assert ours == pytest.approx(theirs)
