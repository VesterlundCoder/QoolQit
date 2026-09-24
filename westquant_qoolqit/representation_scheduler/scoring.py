"""Scoring of representation candidates at multiple cost levels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..common.types import BinaryQuadraticHamiltonian
from ..common.metrics import (
    interaction_frobenius_error, max_abs_interaction_error,
    interaction_rank_correlation, edge_preservation, geometry_metrics,
    logical_fidelity, LogicalFidelity, GeometryMetrics,
)
from ..common.qoolqit_adapter import (
    EmbeddingResult, CompilationOutcome, EmulationOutcome,
)


@dataclass
class CandidateScore:
    """Multi-dimensional score vector for one candidate."""
    candidate_id: str
    frobenius_error: float | None = None
    max_abs_error: float | None = None
    rank_correlation: float | None = None
    edge_preservation: float | None = None
    geometry: GeometryMetrics | None = None
    logical_fidelity: LogicalFidelity | None = None
    compilation_success: bool | None = None
    compilation_failure_reason: str | None = None
    duration: float | None = None
    min_spacing_margin: float | None = None
    max_radial_extent: float | None = None
    ground_state_probability: float | None = None
    expected_objective: float | None = None
    best_sampled_objective: float | None = None
    approximation_ratio: float | None = None
    robust_mean_p_opt: float | None = None
    robust_std_p_opt: float | None = None
    robust_q05_p_opt: float | None = None
    robustness_drop: float | None = None
    embedding: EmbeddingResult | None = None
    compilation: CompilationOutcome | None = None
    emulation: EmulationOutcome | None = None
    metadata: dict = field(default_factory=dict)

    def objective_vector(self, names: list[str]) -> np.ndarray:
        m = {
            "frobenius_error": self.frobenius_error,
            "max_abs_error": self.max_abs_error,
            "rank_correlation": self.rank_correlation,
            "edge_preservation": self.edge_preservation,
            "compilation_success": (0.0 if self.compilation_success else 1.0),
            "duration": self.duration,
            "min_spacing_margin": self.min_spacing_margin,
            "ground_state_probability": self.ground_state_probability,
            "robust_mean_p_opt": self.robust_mean_p_opt,
        }
        return np.array([m.get(n, 0.0) if m.get(n) is not None else 0.0 for n in names])


def score_cheap(h: BinaryQuadraticHamiltonian, embedding: EmbeddingResult) -> dict:
    """Stage 0: cheap geometric + interaction metrics (no compilation/emulation)."""
    return {
        "frobenius_error": interaction_frobenius_error(embedding.target_matrix,
                                                        embedding.realized_interactions),
        "max_abs_error": max_abs_interaction_error(embedding.target_matrix,
                                                   embedding.realized_interactions),
        "rank_correlation": interaction_rank_correlation(embedding.target_matrix,
                                                         embedding.realized_interactions),
        "edge_preservation": edge_preservation(embedding.target_matrix,
                                                embedding.realized_interactions),
        "geometry": geometry_metrics(embedding.coords),
    }


__all__ = ["CandidateScore", "score_cheap"]
