"""Hamiltonian landscape metrics: exact spectra, gaps, near-optimal density."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.types import BinaryQuadraticHamiltonian
from ..common.exact_solver import solve_exact


@dataclass
class LandscapeMetrics:
    ground_state_energy: float
    ground_state_degeneracy: int
    first_excitation: float
    classical_gap: float
    spectral_width: float
    energy_variance: float
    n_near_optimal: int
    near_optimal_density: float
    n: int


def landscape_metrics(h: BinaryQuadraticHamiltonian, near_tol: float = 1e-6) -> LandscapeMetrics:
    """Compute exact classical spectrum metrics for small systems."""
    sol = solve_exact(h, near_tol=near_tol)
    e = sol.energies
    var = float(np.var(e)) if e.size else 0.0
    return LandscapeMetrics(
        ground_state_energy=sol.optimum_energy,
        ground_state_degeneracy=int(sol.optimum_states.shape[0]),
        first_excitation=sol.first_excitation,
        classical_gap=sol.gap,
        spectral_width=sol.spectral_width,
        energy_variance=var,
        n_near_optimal=sol.n_near_optimal,
        near_optimal_density=sol.near_optimal_fraction,
        n=sol.n,
    )


__all__ = ["LandscapeMetrics", "landscape_metrics"]
