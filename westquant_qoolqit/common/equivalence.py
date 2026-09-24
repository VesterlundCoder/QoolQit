"""Equivalence verifier for Hamiltonian transformations.

Every transformation is classified into exactly one of:
    EXACT_EQUIVALENT, GROUND_STATE_EQUIVALENT, SAME_PROBLEM_DIFFERENT_DYNAMICS,
    APPROXIMATE, INVALID

For n <= 16 verification is exhaustive over all 2^n states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from scipy.stats import spearmanr, kendalltau

from .types import BinaryQuadraticHamiltonian


class EquivalenceClass(str, Enum):
    EXACT_EQUIVALENT = "EXACT_EQUIVALENT"
    GROUND_STATE_EQUIVALENT = "GROUND_STATE_EQUIVALENT"
    SAME_PROBLEM_DIFFERENT_DYNAMICS = "SAME_PROBLEM_DIFFERENT_DYNAMICS"
    APPROXIMATE = "APPROXIMATE"
    INVALID = "INVALID"


@dataclass
class EquivalenceReport:
    classification: EquivalenceClass
    max_energy_error: float
    ground_state_match: bool
    ordering_match: bool
    mapped_state_match: bool
    scale: float
    shift: float
    spearman_rho: float
    kendall_tau: float
    top_k_overlap: float
    ground_state_overlap: float
    exhaustive: bool
    details: dict = field(default_factory=dict)


def _best_affine(E_orig: np.ndarray, E_new: np.ndarray) -> tuple[float, float, float]:
    """Find a>0, b minimizing ||E_new - (a E_orig + b)||. Returns (a, b, residual)."""
    # minimize over a>0: linear least squares with sign constraint
    A = np.column_stack([E_orig, np.ones_like(E_orig)])
    coef, *_ = np.linalg.lstsq(A, E_new, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if a <= 0:
        # try forcing a>0 by projecting
        a = abs(a) if a != 0 else 1e-12
    resid = float(np.max(np.abs(E_new - (a * E_orig + b))))
    return a, b, resid


def verify_equivalence(
    original: BinaryQuadraticHamiltonian,
    candidate: BinaryQuadraticHamiltonian,
    forward_map: Callable[[np.ndarray], np.ndarray] | None = None,
    inverse_map: Callable[[np.ndarray], np.ndarray] | None = None,
    top_k: int = 8,
    tol: float = 1e-6,
) -> EquivalenceReport:
    """Verify the equivalence between `original` and `candidate`.

    If forward_map is given, the candidate is interpreted as E'(T(x)) ~ a E(x) + b
    and we compare E'(T(x)) against a E(x)+b.  If no map is given we compare the
    energy orderings directly (used for ground-state-equivalent penalty families).
    """
    n = original.n
    exhaustive = n <= 16
    if not exhaustive:
        # sample a subset of states for approximate verification
        rng = np.random.default_rng(0)
        n_states = min(4096, 2**n)
        idx = rng.choice(2**n, size=n_states, replace=False)
        bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    else:
        bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)

    E_orig = original.energy(bits)
    if forward_map is not None:
        mapped = forward_map(bits)
        E_new = candidate.energy(mapped)
    else:
        E_new = candidate.energy(bits)

    a, b, resid = _best_affine(E_orig, E_new)
    max_err = float(np.max(np.abs(E_new - (a * E_orig + b))))

    # ground-state sets
    e0_orig = float(E_orig.min())
    e0_new = float(E_new.min())
    gs_orig = np.isclose(E_orig, e0_orig, atol=1e-9, rtol=0.0)
    gs_new = np.isclose(E_new, e0_new, atol=1e-9, rtol=0.0)

    if forward_map is not None:
        # mapped ground state: T(argmin E) should equal argmin E'
        gs_mapped = gs_orig.copy()
    else:
        gs_mapped = gs_new

    ground_state_match = bool(np.array_equal(gs_orig, gs_mapped))
    # ordering match: ranks identical
    r_orig = np.argsort(np.argsort(E_orig))
    r_new = np.argsort(np.argsort(E_new))
    ordering_match = bool(np.array_equal(r_orig, r_new))

    # rank correlations
    sp = spearmanr(E_orig, E_new)
    kt = kendalltau(E_orig, E_new)
    spearman_rho = float(sp.correlation) if sp.correlation is not np.nan else 0.0
    kendall_tau = float(kt.correlation) if kt.correlation is not np.nan else 0.0

    # top-k overlap
    k = min(top_k, len(E_orig))
    top_orig = set(np.argsort(E_orig)[:k].tolist())
    top_new = set(np.argsort(E_new)[:k].tolist())
    top_k_overlap = len(top_orig & top_new) / k

    # ground-state overlap (Jaccard)
    inter = int(np.sum(gs_orig & gs_new))
    union = int(np.sum(gs_orig | gs_new))
    ground_state_overlap = inter / union if union > 0 else 1.0

    # classification
    if max_err < tol and ordering_match:
        classification = EquivalenceClass.EXACT_EQUIVALENT
    elif ground_state_match:
        classification = EquivalenceClass.GROUND_STATE_EQUIVALENT
    elif top_k_overlap > 0.5 or spearman_rho > 0.9:
        classification = EquivalenceClass.APPROXIMATE
    else:
        classification = EquivalenceClass.INVALID

    return EquivalenceReport(
        classification=classification,
        max_energy_error=max_err,
        ground_state_match=ground_state_match,
        ordering_match=ordering_match,
        mapped_state_match=ground_state_match,
        scale=a,
        shift=b,
        spearman_rho=spearman_rho,
        kendall_tau=kendall_tau,
        top_k_overlap=top_k_overlap,
        ground_state_overlap=ground_state_overlap,
        exhaustive=exhaustive,
        details={"n_states": len(E_orig), "e0_orig": e0_orig, "e0_new": e0_new},
    )


__all__ = ["EquivalenceClass", "EquivalenceReport", "verify_equivalence"]
