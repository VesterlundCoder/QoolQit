"""Scheduler result container with Pareto front and labeled bests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..common.pareto import pareto_front, label_pareto
from .scoring import CandidateScore


@dataclass
class SchedulerResult:
    """Result of a representation search."""
    scores: list[CandidateScore]
    problem: Any
    objective_names: list[str] = field(default_factory=list)
    maximize: list[bool] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def _vectors(self) -> np.ndarray:
        names = self.objective_names or [
            "frobenius_error", "ground_state_probability", "duration",
        ]
        return np.array([s.objective_vector(names) for s in self.scores]), names

    def pareto_front(self):
        vecs, names = self._vectors()
        maximize = self.maximize or [False] * vecs.shape[1]
        return pareto_front(vecs, maximize=maximize)

    def best(self, objective: str = "ground_state_probability") -> CandidateScore | None:
        """Best candidate by a single objective (maximize for probabilities)."""
        if not self.scores:
            return None
        maximize_objs = {"ground_state_probability", "rank_correlation",
                        "edge_preservation", "robust_mean_p_opt",
                        "min_spacing_margin", "approximation_ratio"}
        vals = [getattr(s, objective) for s in self.scores]
        valid = [(v, s) for v, s in zip(vals, self.scores) if v is not None]
        if not valid:
            return None
        if objective in maximize_objs:
            return max(valid, key=lambda t: t[0])[1]
        return min(valid, key=lambda t: t[0])[1]

    def history(self) -> list[dict]:
        out = []
        for s in self.scores:
            out.append({
                "candidate_id": s.candidate_id,
                "frobenius_error": s.frobenius_error,
                "rank_correlation": s.rank_correlation,
                "compilation_success": s.compilation_success,
                "duration": s.duration,
                "ground_state_probability": s.ground_state_probability,
                "robust_mean_p_opt": s.robust_mean_p_opt,
            })
        return out

    def labels(self) -> dict:
        """Label notable candidates on the Pareto frontier."""
        vecs, names = self._vectors()
        maximize = self.maximize or [False] * vecs.shape[1]
        res = label_pareto(vecs, names, maximize=maximize)
        labels = {}
        for idx, name in res["labels"].items():
            labels[name] = self.scores[idx].candidate_id
        return labels


__all__ = ["SchedulerResult"]
