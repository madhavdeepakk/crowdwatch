"""Optimal one-to-one matching (the assignment problem), in plain numpy.

The tracker has to pair this frame's detections with the people it is already
following so that the total overlap is as large as possible. That is the
classic assignment problem, solved exactly here with the Hungarian algorithm
(the shortest-augmenting-path form, O(n^2 m) in the worst case and close to
O(n m) on tracking problems, where most people have one obvious match).

It is written out rather than taken from SciPy on purpose. It keeps the core
of the project free of large compiled dependencies, which matters on locked
down machines: Windows application-control policies can refuse to load
SciPy's unsigned binaries, and a people counter should not fail to start
because of a linear-programming module it never uses.
"""

from __future__ import annotations

import numpy as np


def _solve_rows_le_cols(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost assignment for an (n, m) cost matrix with n <= m.

    Returns, for each row, the column it is assigned to.
    """
    n, m = cost.shape
    u = np.zeros(n + 1)                    # row potentials (1-based)
    v = np.zeros(m + 1)                    # column potentials (1-based)
    match = np.zeros(m + 1, dtype=int)     # match[j] = row assigned to column j, 0 = none
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        slack = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = match[j0]
            free = ~used[1:]
            reduced = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (reduced < slack[1:])
            slack[1:][better] = reduced[better]
            way[1:][better] = j0
            candidates = np.where(free, slack[1:], np.inf)
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            u[match[used]] += delta
            v[used] -= delta
            slack[~used] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:                           # flip the augmenting path
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
    columns = np.zeros(n, dtype=int)
    assigned = np.flatnonzero(match[1:])
    columns[match[1:][assigned] - 1] = assigned
    return columns


def linear_sum_assignment(cost: np.ndarray, maximize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of an optimal assignment, like ``scipy.optimize.linear_sum_assignment``.

    Every row is assigned when there are at least as many columns as rows,
    otherwise every column is. Rows come back in increasing order.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("expected a 2-D cost matrix")
    if cost.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if not np.isfinite(cost).all():
        raise ValueError("the cost matrix must be finite")
    if maximize:
        cost = -cost
    n, m = cost.shape
    if n <= m:
        return np.arange(n), _solve_rows_le_cols(cost)
    rows = _solve_rows_le_cols(cost.T)      # here each column picks a row
    order = np.argsort(rows)
    return rows[order], np.arange(m)[order]
