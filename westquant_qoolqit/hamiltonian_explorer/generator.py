"""Hamiltonian candidate generator: produces a family of representations for one problem."""

from __future__ import annotations

import numpy as np

from ..common.types import BinaryQuadraticHamiltonian
from ..common.equivalence import EquivalenceClass, verify_equivalence
from .transforms import (
    TransformResult, positive_scale, variable_permutation, bit_complement,
    mwis_penalty, mwis_penalty_family, mwis_penalty_safe_threshold,
)
from .realizability import classify_realizability
from .scoring import landscape_metrics
from .result import HamiltonianCandidate


def _make_candidate(tr: TransformResult, cid: str, original: BinaryQuadraticHamiltonian,
                    verify: bool = True) -> HamiltonianCandidate:
    verification = None
    final_class = tr.equivalence_class
    if verify and tr.equivalence_class.value in ("EXACT_EQUIVALENT", "GROUND_STATE_EQUIVALENT"):
        verification = verify_equivalence(original, tr.hamiltonian,
                                          forward_map=tr.forward_map)
        # Verification is authoritative: if the declared class doesn't match
        # the verified class, downgrade to INVALID.
        declared = tr.equivalence_class
        verified_class = verification.classification
        if declared == EquivalenceClass.EXACT_EQUIVALENT and verified_class != EquivalenceClass.EXACT_EQUIVALENT:
            final_class = EquivalenceClass.INVALID
        elif declared == EquivalenceClass.GROUND_STATE_EQUIVALENT and verified_class not in (
                EquivalenceClass.EXACT_EQUIVALENT, EquivalenceClass.GROUND_STATE_EQUIVALENT):
            final_class = EquivalenceClass.INVALID
    return HamiltonianCandidate(
        id=cid, hamiltonian=tr.hamiltonian, transform_name=tr.transform_name,
        transform_parameters=tr.transform_parameters, equivalence_class=final_class,
        forward_state_map=tr.forward_map, inverse_state_map=tr.inverse_map,
        proof_metadata=tr.proof_metadata, verification=verification,
        realizability=classify_realizability(tr.hamiltonian),
        landscape=landscape_metrics(tr.hamiltonian),
        parent_id=tr.parent_id,
    )


def generate_hamiltonians(
    problem: BinaryQuadraticHamiltonian,
    transforms: list[str] | None = None,
    verify: bool = True,
    mwis_info: dict | None = None,
    seed: int = 42,
) -> list[HamiltonianCandidate]:
    """Generate a family of Hamiltonian representations for one problem.

    Args:
        problem: the canonical problem Hamiltonian.
        transforms: list of transform family names to apply.
        mwis_info: if the problem is MWIS, dict with 'weights' and 'edges' to
            generate the penalty family with several U values.
    """
    transforms = transforms or ["positive_scale", "permutation", "bit_complement"]
    candidates: list[HamiltonianCandidate] = []
    rng = np.random.default_rng(seed)

    # baseline (identity)
    candidates.append(_make_candidate(
        TransformResult(hamiltonian=problem.copy(), transform_name="identity",
                        transform_parameters={}, equivalence_class=__import__(
                            "westquant_qoolqit.common.equivalence", fromlist=["EquivalenceClass"]
                        ).EquivalenceClass.EXACT_EQUIVALENT,
                        forward_map=lambda x: x, inverse_map=lambda y: y,
                        proof_metadata={"relation": "identity"}),
        cid="H_identity", original=problem, verify=verify))

    if "positive_scale" in transforms:
        for a in [0.5, 2.0, 5.0]:
            tr = positive_scale(problem, a=a, b=0.0)
            candidates.append(_make_candidate(tr, f"H_scale_{a}", problem, verify))

    if "permutation" in transforms:
        for k in range(2):
            perm = rng.permutation(problem.n).tolist()
            tr = variable_permutation(problem, perm)
            candidates.append(_make_candidate(tr, f"H_perm_{k}", problem, verify))

    if "bit_complement" in transforms:
        # complement a random subset
        S = sorted(set(rng.choice(problem.n, size=max(1, problem.n // 2), replace=False).tolist()))
        tr = bit_complement(problem, S)
        candidates.append(_make_candidate(tr, "H_complement", problem, verify))

    if "mwis_penalty" in transforms and mwis_info is not None:
        weights = np.asarray(mwis_info["weights"], dtype=float)
        edges = mwis_info["edges"]
        wmax = mwis_penalty_safe_threshold(weights)
        # safe U values strictly above the threshold
        U_values = [wmax * f for f in (1.25, 1.5, 2.0, 3.0, 5.0)]
        for tr in mwis_penalty_family(weights, edges, U_values):
            cid = f"H_mwis_U{tr.transform_parameters['U']:.3f}"
            candidates.append(_make_candidate(tr, cid, problem, verify))

    return candidates


__all__ = ["generate_hamiltonians"]
