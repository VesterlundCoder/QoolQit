"""Exact classical solver for small binary-quadratic problems.

For n <= 22 we enumerate all 2^n bitstrings and compute the exact optimum,
ground-state set, spectrum and near-optimal states.  This is the ground truth
against which all quantum/emulator results are compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import BinaryQuadraticHamiltonian


@dataclass
class ExactSolution:
    """Exact classical solution of a BinaryQuadraticHamiltonian."""

    hamiltonian: BinaryQuadraticHamiltonian
    energies: np.ndarray  # (2^n,) energy of every bitstring
    optimum_energy: float
    optimum_states: np.ndarray  # (k, n) bitstrings achieving the optimum
    optimum_indices: np.ndarray  # indices into the 2^n enumeration
    first_excitation: float  # E1 - E0 (inf if degenerate-only or n<2)
    gap: float  # alias for first_excitation
    spectral_width: float
    n_near_optimal: int  # states within `near_tol` of optimum
    near_optimal_fraction: float
    sorted_energies: np.ndarray  # energies sorted ascending
    n: int
    exhaustive: bool = True
    metadata: dict = field(default_factory=dict)


def solve_exact(h: BinaryQuadraticHamiltonian, near_tol: float = 1e-6) -> ExactSolution:
    """Exhaustively solve a small BinaryQuadraticHamiltonian."""
    n = h.n
    if n > 22:
        raise ValueError(f"Exhaustive solve infeasible for n={n} (>22).")
    energies = h.energies_all()
    e0 = float(energies.min())
    opt_idx = np.where(np.isclose(energies, e0, atol=1e-9, rtol=0.0))[0]
    bits = ((opt_idx[:, None] >> np.arange(n)) & 1).astype(np.int8)
    sorted_e = np.sort(energies)
    # first excitation = smallest energy strictly above e0
    above = sorted_e[sorted_e > e0 + 1e-9]
    gap = float(above.min() - e0) if above.size else float("inf")
    near = int(np.sum(energies <= e0 + near_tol))
    return ExactSolution(
        hamiltonian=h,
        energies=energies,
        optimum_energy=e0,
        optimum_states=bits,
        optimum_indices=opt_idx,
        first_excitation=gap,
        gap=gap,
        spectral_width=float(energies.max() - energies.min()),
        n_near_optimal=near,
        near_optimal_fraction=near / len(energies),
        sorted_energies=sorted_e,
        n=n,
    )


def mwis_exact(weights: np.ndarray, edges: list[tuple[int, int]]) -> ExactSolution:
    """Build the MWIS penalty Hamiltonian with a safe penalty and solve exactly.

    E_U(x) = -sum w_i x_i + U sum_{(i,j) in E} x_i x_j, with U > max weight
    guarantees the independent-set constraint.  We use U = max(w)+1 (plus a
    margin) as a safe default and return the exact solution of that encoding.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    n = weights.shape[0]
    U = float(weights.max()) + 1.0
    linear = -weights.copy()
    quadratic = np.zeros((n, n))
    for i, j in edges:
        i, j = (i, j) if i < j else (j, i)
        quadratic[i, j] = U
        quadratic[j, i] = U
    h = BinaryQuadraticHamiltonian(linear=linear, quadratic=quadratic, constant=0.0,
                                   metadata={"type": "mwis", "U": U, "weights": weights.tolist(),
                                             "edges": [tuple(e) for e in edges]})
    return solve_exact(h)


__all__ = ["ExactSolution", "solve_exact", "mwis_exact"]
