"""Benchmark problem generators: QUBO, MWIS, and a small industrial example."""

from __future__ import annotations

import numpy as np

from .common.types import BinaryQuadraticHamiltonian


# --------------------------------------------------------------------------- #
# QUBO
# --------------------------------------------------------------------------- #
def synthetic_qubo(n: int, density: float = 0.5, seed: int = 0,
                   value_range: tuple[float, float] = (-2.0, 2.0)) -> BinaryQuadraticHamiltonian:
    """Random symmetric QUBO with given density of nonzero off-diagonal entries."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                v = rng.uniform(*value_range)
                Q[i, j] = v
                Q[j, i] = v
    np.fill_diagonal(Q, rng.uniform(*value_range, size=n))
    h = BinaryQuadraticHamiltonian.from_qubo(Q)
    h.metadata = {"type": "qubo", "n": n, "density": density, "seed": seed}
    return h


# --------------------------------------------------------------------------- #
# MWIS
# --------------------------------------------------------------------------- #
def mwis_from_graph(weights: np.ndarray, edges: list[tuple[int, int]],
                    U: float | None = None) -> BinaryQuadraticHamiltonian:
    """Build the MWIS penalty Hamiltonian E_U = -w.x + U sum x_i x_j."""
    weights = np.asarray(weights, dtype=float).ravel()
    if U is None:
        U = float(weights.max()) + 1.0
    n = weights.shape[0]
    linear = -weights.copy()
    quadratic = np.zeros((n, n))
    for i, j in edges:
        i, j = (i, j) if i < j else (j, i)
        quadratic[i, j] = U
        quadratic[j, i] = U
    h = BinaryQuadraticHamiltonian(linear=linear, quadratic=quadratic, constant=0.0,
                                   metadata={"type": "mwis", "U": U,
                                             "weights": weights.tolist(),
                                             "edges": [tuple(e) for e in edges]})
    return h


def mwis_path(n: int, weights: np.ndarray | None = None, seed: int = 0,
              U: float | None = None) -> tuple[BinaryQuadraticHamiltonian, dict]:
    weights = weights if weights is not None else np.random.default_rng(seed).uniform(1, 5, size=n)
    edges = [(i, i + 1) for i in range(n - 1)]
    h = mwis_from_graph(weights, edges, U=U)
    return h, {"weights": weights.tolist(), "edges": edges}


def mwis_cycle(n: int, seed: int = 0, U: float | None = None):
    weights = np.random.default_rng(seed).uniform(1, 5, size=n)
    edges = [(i, (i + 1) % n) for i in range(n)]
    return mwis_from_graph(weights, edges, U=U), {"weights": weights.tolist(), "edges": edges}


def mwis_grid(rows: int, cols: int, seed: int = 0, U: float | None = None):
    n = rows * cols
    weights = np.random.default_rng(seed).uniform(1, 5, size=n)
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1))
            if r + 1 < rows:
                edges.append((i, i + cols))
    return mwis_from_graph(weights, edges, U=U), {"weights": weights.tolist(), "edges": edges}


def mwis_random_geometric(n: int, radius: float = 0.7, seed: int = 0, U: float | None = None):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n, 2))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) < radius:
                edges.append((i, j))
    weights = rng.uniform(1, 5, size=n)
    return mwis_from_graph(weights, edges, U=U), {"weights": weights.tolist(), "edges": edges}


def mwis_erdos_renyi(n: int, p: float = 0.4, seed: int = 0, U: float | None = None):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    weights = rng.uniform(1, 5, size=n)
    return mwis_from_graph(weights, edges, U=U), {"weights": weights.tolist(), "edges": edges}


# --------------------------------------------------------------------------- #
# Small industrial example: shift assignment (scheduling)
# --------------------------------------------------------------------------- #
def shift_assignment(n_workers: int = 4, n_shifts: int = 3, seed: int = 0):
    """Small scheduling QUBO: assign workers to shifts minimizing conflicts.

    Variables x_{w,s} = 1 if worker w assigned to shift s.  Constraints:
      - each worker assigned to exactly one shift (one-hot)
      - minimize a soft cost (random preferences).

    The one-hot penalty is P * (sum_s x_{w,s} - 1)^2 which expands to:
      P * (sum_s x_{w,s}^2 + 2*sum_{s<s'} x_{w,s} x_{w,s'} - 2*sum_s x_{w,s} + 1)

    Since x^2 = x for binary variables, the diagonal contribution is:
      P * (1 - 2) = -P  per variable (from the -2*sum term)
    The off-diagonal contribution is:
      2*P  per pair (from the 2*sum_{s<s'} term)
    The constant P is dropped (doesn't affect optimization).

    Encoded as a QUBO small enough for exact enumeration.
    """
    rng = np.random.default_rng(seed)
    n = n_workers * n_shifts
    Q = np.zeros((n, n))
    P = 5.0
    for w in range(n_workers):
        idxs = [w * n_shifts + s for s in range(n_shifts)]
        # One-hot penalty: P * (sum - 1)^2
        for a in idxs:
            Q[a, a] += -2 * P  # from -2*P*sum term
        for a in idxs:
            for b in idxs:
                if a != b:
                    Q[a, b] += P  # from P*sum_{s<s'} term (symmetric, so each pair gets P)
        # Soft preference cost on diagonal
        for a in idxs:
            Q[a, a] += rng.uniform(-1, 1)
    h = BinaryQuadraticHamiltonian.from_qubo(Q)
    h.metadata = {"type": "scheduling", "n_workers": n_workers,
                  "n_shifts": n_shifts, "seed": seed}
    return h


__all__ = [
    "synthetic_qubo", "mwis_from_graph", "mwis_path", "mwis_cycle", "mwis_grid",
    "mwis_random_geometric", "mwis_erdos_renyi", "shift_assignment",
]
