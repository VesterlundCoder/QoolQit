"""WestQuant Hamiltonian Representation Explorer (Project B)."""

from .transforms import (
    TransformResult, positive_scale, variable_permutation, bit_complement,
    mwis_penalty, mwis_penalty_safe_threshold, mwis_penalty_family,
    ControlRepresentation, control_representations,
)
from .realizability import PhysicalRealizabilityReport, classify_realizability
from .scoring import LandscapeMetrics, landscape_metrics
from .result import HamiltonianCandidate
from .generator import generate_hamiltonians
from .explorer import HamiltonianRepresentationExplorer, ExplorerResult

__all__ = [
    "TransformResult", "positive_scale", "variable_permutation", "bit_complement",
    "mwis_penalty", "mwis_penalty_safe_threshold", "mwis_penalty_family",
    "ControlRepresentation", "control_representations",
    "PhysicalRealizabilityReport", "classify_realizability",
    "LandscapeMetrics", "landscape_metrics",
    "HamiltonianCandidate", "generate_hamiltonians",
    "HamiltonianRepresentationExplorer", "ExplorerResult",
]
