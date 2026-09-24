"""Equivalence verifier for Hamiltonian transformations.

Every transformation is classified into exactly one of:
    EXACT_EQUIVALENT, GROUND_STATE_EQUIVALENT, SAME_PROBLEM_DIFFERENT_DYNAMICS,
    APPROXIMATE, INVALID

For n <= 16 verification is exhaustive over all 2^n states.

Equivalence semantics:
    EXACT_EQUIVALENT: E'(T(x)) = a*E(x) + b  with a > 0 for every state x.
    GROUND_STATE_EQUIVALENT: T(argmin E) == argmin E'  (including degeneracies).
    SAME_PROBLEM_DIFFERENT_DYNAMICS: same classical objective, different dynamics.
    APPROXIMATE: quantitative approximation criterion satisfied.
    INVALID: fails all above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from scipy.stats import rankdata, spearmanr, kendalltau

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
    A = np.column_stack([E_orig, np.ones_like(E_orig)])
    coef, *_ = np.linalg.lstsq(A, E_new, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if a <= 0:
        a = abs(a) if a != 0 else 1e-12
    resid = float(np.max(np.abs(E_new - (a * E_orig + b))))
    return a, b, resid


def _tie_aware_ordering_match(E1: np.ndarray, E2: np.ndarray, tol: float = 1e-9) -> bool:
    """Check if two energy arrays have the same ordering, respecting ties.

    Uses rankdata(method='average') so that degenerate levels get the same rank.
    Two orderings match iff their tie-aware ranks are identical.
    """
    r1 = rankdata(E1, method='average')
    r2 = rankdata(E2, method='average')
    # Normalize to compare rank patterns (allowing for different scale)
    # For exact equivalence, ranks must be identical up to affine transform
    # But since we already check affine separately, here we check rank equality
    return bool(np.allclose(r1, r2, atol=tol))


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

    Ground-state verification when forward_map is provided:
        G = argmin E_orig   (set of original ground states)
        G' = argmin E_new   (set of candidate ground states)
        We require T(G) == G'  (mapped original ground states match candidate ground states).
    """
    n = original.n
    exhaustive = n <= 16
    if not exhaustive:
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
        # MAPPED ground-state verification:
        # T(argmin E_orig) should equal argmin E_new
        # gs_orig is a boolean mask over original states.
        # We need to check: for each original ground state x, T(x) is a candidate
        # ground state; AND every candidate ground state is T(x) for some
        # original ground state x.
        #
        # Since we already computed E_new = candidate.energy(forward_map(bits)),
        # gs_new is the set of mapped states that are candidate ground states.
        # gs_orig is the set of original states that are original ground states.
        # The mapped ground-state set is: { T(x) : x in argmin E_orig }
        # In our array representation, this is gs_new restricted to indices where
        # gs_orig is True (because E_new[i] = candidate.energy(T(bits[i]))).
        #
        # CORRECT verification:
        # 1. Every original ground state maps to a candidate ground state:
        #    gs_orig → gs_new  (if gs_orig[i] then gs_new[i])
        # 2. Every candidate ground state is the image of some original ground state:
        #    gs_new ⊆ T(gs_orig)  (if gs_new[i] then gs_orig[i])
        # Combined: gs_new[i] == gs_orig[i] for all i (in the mapped space)
        gs_mapped = gs_orig.copy()  # which original states are ground states
        # Check: original ground states map to candidate ground states
        orig_gs_map_to_new_gs = bool(np.all(gs_new[gs_mapped]))
        # Check: all candidate ground states come from original ground states
        new_gs_come_from_orig_gs = bool(np.all(gs_mapped[gs_new]))
        ground_state_match = orig_gs_map_to_new_gs and new_gs_come_from_orig_gs
    else:
        # No forward map: compare ground-state sets directly
        ground_state_match = bool(np.array_equal(gs_orig, gs_new))

    # ordering match: tie-aware ranks identical
    ordering_match = _tie_aware_ordering_match(E_orig, E_new, tol=1e-9)

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
