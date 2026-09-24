"""Physical realizability classifier for the Rydberg Analog Model.

Before sending a Hamiltonian into expensive embedding/emulation, classify its
compatibility with the native neutral-atom model.  Native Rydberg interactions
are repulsive/positive (J_ij = 1/r^6 > 0), so a Hamiltonian with negative pair
interactions is not directly native-realizable without extra machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common.types import BinaryQuadraticHamiltonian


@dataclass
class PhysicalRealizabilityReport:
    native_pair_interactions: bool        # all pair interactions >= 0
    requires_dmm: bool                    # has non-trivial linear terms
    requires_rescaling: bool              # dynamic range large
    dynamic_range: float                  # max/min positive interaction
    n_negative_interactions: int
    n_positive_interactions: int
    warnings: list[str] = field(default_factory=list)
    realizable: bool = False

    def to_dict(self) -> dict:
        return {
            "native_pair_interactions": self.native_pair_interactions,
            "requires_dmm": self.requires_dmm,
            "requires_rescaling": self.requires_rescaling,
            "dynamic_range": self.dynamic_range,
            "n_negative_interactions": self.n_negative_interactions,
            "n_positive_interactions": self.n_positive_interactions,
            "warnings": self.warnings,
            "realizable": self.realizable,
        }


def classify_realizability(h: BinaryQuadraticHamiltonian) -> PhysicalRealizabilityReport:
    """Classify a Hamiltonian's compatibility with the native Rydberg model."""
    J = h.interaction_matrix()
    iu = np.triu_indices_from(J, k=1)
    pairs = J[iu]
    n_neg = int(np.sum(pairs < -1e-12))
    n_pos = int(np.sum(pairs > 1e-12))
    pos = pairs[pairs > 1e-12]
    dynamic_range = float(pos.max() / pos.min()) if pos.size and pos.min() > 0 else float("inf")
    native = n_neg == 0
    requires_dmm = bool(np.any(np.abs(h.linear) > 1e-12))
    requires_rescaling = dynamic_range > 1e3 if np.isfinite(dynamic_range) else True
    warnings = []
    if n_neg > 0:
        warnings.append(
            f"{n_neg} negative pair interactions: not directly native-Rydberg-realizable "
            "(Rydberg interactions are repulsive/positive). May require bit-complement or DMM."
        )
    if np.isinf(dynamic_range) and n_pos > 0:
        warnings.append("Infinite/undefined dynamic range (zero or near-zero positive interactions).")
    if dynamic_range > 1e4:
        warnings.append(f"Very large dynamic range ({dynamic_range:.2e}); embedding will be hard.")
    realizable = native and (not np.isinf(dynamic_range) or n_pos == 0)
    return PhysicalRealizabilityReport(
        native_pair_interactions=native,
        requires_dmm=requires_dmm,
        requires_rescaling=requires_rescaling,
        dynamic_range=dynamic_range,
        n_negative_interactions=n_neg,
        n_positive_interactions=n_pos,
        warnings=warnings,
        realizable=realizable,
    )


__all__ = ["PhysicalRealizabilityReport", "classify_realizability"]
