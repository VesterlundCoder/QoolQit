"""Hamiltonian Representation Explorer (Project B).

Searches over mathematically valid Hamiltonian representations of one
underlying optimization problem, verifies equivalence, classifies physical
realizability, and measures landscape differences.

    P -> H(P) = {H1, H2, ..., Hm}  (same problem, different representations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..common.types import BinaryQuadraticHamiltonian
from ..common.equivalence import EquivalenceClass
from .generator import generate_hamiltonians
from .result import HamiltonianCandidate
from .realizability import classify_realizability
from .scoring import landscape_metrics


@dataclass
class ExplorerResult:
    """Result of a Hamiltonian representation exploration."""
    candidates: list[HamiltonianCandidate]
    problem: BinaryQuadraticHamiltonian
    metadata: dict = field(default_factory=dict)

    def equivalent(self) -> list[HamiltonianCandidate]:
        """Candidates classified as exact or ground-state equivalent."""
        return [c for c in self.candidates
                if c.equivalence_class in (EquivalenceClass.EXACT_EQUIVALENT,
                                            EquivalenceClass.GROUND_STATE_EQUIVALENT)]

    def rydberg_native(self) -> list[HamiltonianCandidate]:
        """Candidates whose pair interactions are all native-Rydberg (>=0)."""
        return [c for c in self.candidates
                if c.realizability is not None and c.realizability.native_pair_interactions]

    def by_transform(self, name: str) -> list[HamiltonianCandidate]:
        return [c for c in self.candidates if c.transform_name == name]

    def table(self) -> list[dict]:
        """Compact table of candidate properties."""
        rows = []
        for c in self.candidates:
            lm = c.landscape
            rows.append({
                "id": c.id,
                "transform": c.transform_name,
                "equivalence": c.equivalence_class.value,
                "native_rydberg": c.realizability.native_pair_interactions if c.realizability else None,
                "dynamic_range": c.realizability.dynamic_range if c.realizability else None,
                "gap": lm.classical_gap if lm else None,
                "gs_degeneracy": lm.ground_state_degeneracy if lm else None,
                "gs_energy": lm.ground_state_energy if lm else None,
            })
        return rows


class HamiltonianRepresentationExplorer:
    """Explore multiple Hamiltonian representations of one problem.

    Example:
        explorer = HamiltonianRepresentationExplorer(
            transforms=["positive_scale", "bit_complement", "mwis_penalty"],
            verify=True, seed=42,
        )
        result = explorer.explore(problem, mwis_info={"weights": w, "edges": E})
    """

    def __init__(
        self,
        transforms: list[str] | None = None,
        verify: bool = True,
        seed: int = 42,
    ) -> None:
        self.transforms = transforms or ["positive_scale", "bit_complement", "mwis_penalty"]
        self.verify = verify
        self.seed = seed

    def explore(
        self,
        problem: BinaryQuadraticHamiltonian,
        mwis_info: dict | None = None,
    ) -> ExplorerResult:
        candidates = generate_hamiltonians(
            problem, transforms=self.transforms, verify=self.verify,
            mwis_info=mwis_info, seed=self.seed,
        )
        return ExplorerResult(candidates=candidates, problem=problem,
                              metadata={"seed": self.seed, "n_candidates": len(candidates)})


__all__ = ["HamiltonianRepresentationExplorer", "ExplorerResult"]
