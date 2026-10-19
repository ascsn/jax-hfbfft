"""
Paths on a potential-energy surface tabulated on a grid of collective
coordinates (e.g. the energies of a multi-dimensional run_constrained_scan):

- minimum_energy_path: the path between two points whose highest energy is
  lowest (the saddle, i.e. the barrier, is that maximum);
- least_action_path: the WKB path that minimises the action
  S = Int sqrt(2 (E - E0) dq.M.dq) / hbar, with the inertia tensor M of
  cranking_inertia (hbar^2 MeV^-1 per unit of q^2), and its penetrability
  P = 1 / (1 + exp(2 S)).

Both are Dijkstra searches over grid neighbours (diagonals included); points
where E is NaN (unconverged) are impassable.
"""

import heapq
import itertools

import numpy as np


def _search(E, start, end, key):
    """Dijkstra over grid neighbours: key(acc, a, b) is the accumulated cost at b via a
    (key(None, start, start) at the start). Returns (path, cost at end)."""
    E = np.asarray(E, dtype=float)
    steps = [s for s in itertools.product((-1, 0, 1), repeat=E.ndim) if any(s)]
    start, end = tuple(start), tuple(end)
    best, prev = {start: key(None, start, start)}, {}
    heap = [(best[start], start)]
    while heap:
        acc, a = heapq.heappop(heap)
        if a == end:
            break
        if acc > best[a]:
            continue
        for s in steps:
            b = tuple(np.add(a, s))
            if any(i < 0 or i >= n for i, n in zip(b, E.shape)) or np.isnan(E[b]):
                continue
            new = key(acc, a, b)
            if b not in best or new < best[b]:
                best[b], prev[b] = new, a
                heapq.heappush(heap, (new, b))
    if end not in best:
        raise ValueError("end is not reachable from start")
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1], best[end]


def minimum_energy_path(E, start, end) -> dict:
    """
    The path from index start to index end over which the highest energy is lowest
    (ties broken by the number of steps). Returns dict(path, energies, saddle,
    barrier): the grid indices, E along them, the index of the highest point, and
    its height above E[start].
    """
    E = np.asarray(E, dtype=float)
    path, _ = _search(E, start, end,
                      lambda acc, a, b: (E[b], 0) if acc is None else (max(acc[0], E[b]), acc[1] + 1))
    energies = np.array([E[p] for p in path])
    k = int(np.argmax(energies))
    return dict(path=path, energies=energies, saddle=path[k], barrier=float(energies[k] - E[tuple(start)]))


def least_action_path(E, coords, start, end, mass=None, E0=None) -> dict:
    """
    The least-action path from start to end (grid indices). coords: one 1D array of
    coordinate values per axis; mass: the inertia tensor on the grid, shape
    E.shape + (d, d) (default the unit tensor); E0: the energy the action is
    measured from (default E[start]). Returns dict(path, energies, action,
    penetrability), action in units of hbar.
    """
    E = np.asarray(E, dtype=float)
    d = E.ndim
    q = [np.asarray(c, dtype=float) for c in coords]
    M = np.broadcast_to(np.eye(d), E.shape + (d, d)) if mass is None else np.asarray(mass, dtype=float)
    E0 = E[tuple(start)] if E0 is None else E0

    def step(acc, a, b):
        if acc is None:
            return 0.0
        dq = np.array([q[i][b[i]] - q[i][a[i]] for i in range(d)])
        p = 0.5 * sum(np.sqrt(2.0 * max(E[c] - E0, 0.0)) for c in (a, b))      # trapezoid
        return acc + p * np.sqrt(dq @ (0.5 * (M[a] + M[b])) @ dq)

    path, S = _search(E, start, end, step)
    return dict(path=path, energies=np.array([E[p] for p in path]), action=float(S),
                penetrability=float(1.0 / (1.0 + np.exp(2.0 * S))))
