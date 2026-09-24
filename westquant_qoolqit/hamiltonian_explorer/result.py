"""Hamiltonian candidate data model with full provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..common.types import BinaryQuadraticHamiltonian
from ..common.equivalence import EquivalenceClass, EquivalenceReport
from .realizability import PhysicalRealizabilityReport
from .scoring import LandscapeMetrics


@dataclass
class HamiltonianCandidate:
    """A single Hamiltonian representation of an underlying problem."""
    id: str
    hamiltonian: BinaryQuadraticHamiltonian
    transform_name: str
    transform_parameters: dict
    equivalence_class: EquivalenceClass
    forward_state_map: Callable | None = None
    inverse_state_map: Callable | None = None
    proof_metadata: dict = field(default_factory=dict)
    verification: EquivalenceReport | None = None
    realizability: PhysicalRealizabilityReport | None = None
    landscape: LandscapeMetrics | None = None
    parent_id: str | None = None

    def to_record(self) -> dict:
        """Flatten to a machine-readable record (no callables)."""
        return {
            "id": self.id,
            "transform": self.transform_name,
            "transform_parameters": self.transform_parameters,
            "equivalence": self.equivalence_class.value,
            "parent_id": self.parent_id,
            "proof_metadata": self.proof_metadata,
            "verification": (self.verification.__dict__ if self.verification else None),
            "realizability": (self.realizability.to_dict() if self.realizability else None),
            "landscape": (self.landscape.__dict__ if self.landscape else None),
            "constant": self.hamiltonian.constant,
            "linear": self.hamiltonian.linear.tolist(),
            "quadratic": self.hamiltonian.quadratic.tolist(),
        }


__all__ = ["HamiltonianCandidate"]
