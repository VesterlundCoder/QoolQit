"""Representation proposer interface.

The project must remain fully functional without WestQuant AI.  The proposers
here are deterministic (random, grid, adaptive-heuristic).  An optional
WestQuantProposer may use an AI model but is never required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .candidates import RepresentationCandidate
from .embedders import candidate_grid


class RepresentationProposer(ABC):
    """Abstract proposer of representation candidates."""

    @abstractmethod
    def propose(self, problem, history, n_candidates: int, budget: int) -> list[RepresentationCandidate]:
        ...


class RandomProposer(RepresentationProposer):
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, problem, history, n_candidates: int, budget: int) -> list[RepresentationCandidate]:
        rng = np.random.default_rng(self.seed)
        seeds = rng.integers(0, 10_000, size=max(n_candidates, 3)).tolist()
        return candidate_grid(embedders=["interaction", "spring", "blade"],
                               seeds=seeds[:max(1, n_candidates // 3)])


class GridProposer(RepresentationProposer):
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, problem, history, n_candidates: int, budget: int) -> list[RepresentationCandidate]:
        return candidate_grid(seeds=list(range(self.seed, self.seed + max(1, n_candidates // 6))))


class AdaptiveHeuristicProposer(RepresentationProposer):
    """Propose more candidates from embedder families that performed well historically."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, problem, history, n_candidates: int, budget: int) -> list[RepresentationCandidate]:
        if not history:
            return candidate_grid(seeds=list(range(self.seed, self.seed + 3)))
        # rank embedders by mean frobenius error (lower better)
        from collections import defaultdict
        sums = defaultdict(list)
        for rec in history:
            sums[rec.get("embedder", "interaction")].append(rec.get("frobenius_error", 1.0))
        ranked = sorted(sums.items(), key=lambda kv: np.mean(kv[1]))
        best_embedders = [k for k, _ in ranked[:2]]
        seeds = list(range(self.seed, self.seed + max(1, n_candidates // 3)))
        return candidate_grid(embedders=best_embedders, seeds=seeds)


class WestQuantProposer(RepresentationProposer):
    """Optional AI proposer.  Falls back to AdaptiveHeuristicProposer if no model.

    This demonstrates that the search API can support AI-in-the-loop
    representation design, but the project remains fully functional without it.
    """

    def __init__(self, model=None, seed: int = 0) -> None:
        self.model = model
        self._fallback = AdaptiveHeuristicProposer(seed=seed)

    def propose(self, problem, history, n_candidates: int, budget: int) -> list[RepresentationCandidate]:
        if self.model is None:
            return self._fallback.propose(problem, history, n_candidates, budget)
        # If a real model is provided, it would suggest configs here.
        return self._fallback.propose(problem, history, n_candidates, budget)


__all__ = [
    "RepresentationProposer", "RandomProposer", "GridProposer",
    "AdaptiveHeuristicProposer", "WestQuantProposer",
]
