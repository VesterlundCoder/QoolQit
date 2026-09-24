"""Pareto dominance and frontier computation for multi-objective representation search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray, maximize: Sequence[bool] | None = None) -> bool:
    """Return True if vector a Pareto-dominates b (a is no worse everywhere, strictly somewhere).

    `maximize[i]` True means objective i is maximized.  Defaults to all minimization.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if maximize is None:
        maximize = [False] * a.shape[0]
    maximize = np.asarray(maximize, dtype=bool)
    a_adj = a.copy()
    b_adj = b.copy()
    a_adj[maximize] = -a_adj[maximize]
    b_adj[maximize] = -b_adj[maximize]
    return bool(np.all(a_adj <= b_adj + 1e-12) and np.any(a_adj < b_adj - 1e-12))


def pareto_front(
    points: np.ndarray, maximize: Sequence[bool] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frontier_points, frontier_indices) for a set of objective vectors.

    `points` is (m, d).  Lower is better unless `maximize[i]` is True.
    """
    points = np.asarray(points, dtype=float)
    m = points.shape[0]
    if maximize is None:
        maximize = [False] * points.shape[1]
    is_front = np.ones(m, dtype=bool)
    for i in range(m):
        if not is_front[i]:
            continue
        for j in range(m):
            if i == j or not is_front[j]:
                continue
            if dominates(points[j], points[i], maximize):
                is_front[i] = False
                break
    idx = np.where(is_front)[0]
    return points[idx], idx


@dataclass
class ParetoResult:
    points: np.ndarray
    indices: np.ndarray
    labels: list[str]


def label_pareto(points: np.ndarray, names: list[str], maximize: Sequence[bool] | None = None) -> dict:
    """Label notable points on the Pareto frontier (best per objective)."""
    points = np.asarray(points, dtype=float)
    if maximize is None:
        maximize = [False] * points.shape[1]
    maximize = np.asarray(maximize, dtype=bool)
    front, idx = pareto_front(points, maximize)
    labels: dict[int, str] = {}
    for d, name in enumerate(names):
        col = front[:, d]
        if maximize[d]:
            best = np.argmax(col)
        else:
            best = np.argmin(col)
        labels[int(idx[best])] = name
    return {"front": front, "indices": idx, "labels": labels}


__all__ = ["dominates", "pareto_front", "ParetoResult", "label_pareto"]
