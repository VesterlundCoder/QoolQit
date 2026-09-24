"""WestQuant Representation Scheduler (Project A)."""

from .candidates import RepresentationCandidate
from .embedders import candidate_grid
from .scoring import CandidateScore, score_cheap
from .successive_halving import HalvingConfig, halve_indices
from .robustness import perturb_coordinates, robustness_stats, robustness_drop
from .proposer import (
    RepresentationProposer, RandomProposer, GridProposer,
    AdaptiveHeuristicProposer, WestQuantProposer,
)
from .result import SchedulerResult
from .scheduler import RepresentationScheduler

__all__ = [
    "RepresentationCandidate", "candidate_grid",
    "CandidateScore", "score_cheap",
    "HalvingConfig", "halve_indices",
    "perturb_coordinates", "robustness_stats", "robustness_drop",
    "RepresentationProposer", "RandomProposer", "GridProposer",
    "AdaptiveHeuristicProposer", "WestQuantProposer",
    "SchedulerResult", "RepresentationScheduler",
]
