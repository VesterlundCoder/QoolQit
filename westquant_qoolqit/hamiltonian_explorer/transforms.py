"""Hamiltonian transformations for the Representation Explorer.

Each transform maps a canonical BinaryQuadraticHamiltonian to another
mathematically-related Hamiltonian, with explicit forward/inverse state maps
and an equivalence class.  All algebra is done on the canonical polynomial
representation (not brittle matrix tricks).

Transform families:
  1. positive_scale   -- E'(x) = a E(x) + b          (EXACT_EQUIVALENT)
  2. permutation       -- x' = P x                    (EXACT_EQUIVALENT)
  3. bit_complement    -- x_i = 1-y_i on subset S    (EXACT_EQUIVALENT)
  4. mwis_penalty      -- E_U = -w.x + U sum x_i x_j (GROUND_STATE_EQUIVALENT)
  5. control_decomp    -- same Hamiltonian, different control decomposition
                          (SAME_PROBLEM_DIFFERENT_DYNAMICS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..common.types import BinaryQuadraticHamiltonian
from ..common.equivalence import EquivalenceClass


@dataclass
class TransformResult:
    """The output of a single Hamiltonian transform."""
    hamiltonian: BinaryQuadraticHamiltonian
    transform_name: str
    transform_parameters: dict
    equivalence_class: EquivalenceClass
    forward_map: Callable[[np.ndarray], np.ndarray] | None
    inverse_map: Callable[[np.ndarray], np.ndarray] | None
    proof_metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


# --------------------------------------------------------------------------- #
# Family 1: positive energy scaling  E'(x) = a E(x) + b, a > 0
# --------------------------------------------------------------------------- #
def positive_scale(h: BinaryQuadraticHamiltonian, a: float, b: float = 0.0) -> TransformResult:
    if a <= 0:
        raise ValueError("Scale a must be positive.")
    new = BinaryQuadraticHamiltonian(
        linear=a * h.linear,
        quadratic=a * h.quadratic,
        constant=a * h.constant + b,
        metadata=dict(h.metadata),
    )
    new.metadata["transform"] = "positive_scale"
    new.metadata["scale"] = a
    new.metadata["shift"] = b
    return TransformResult(
        hamiltonian=new, transform_name="positive_scale",
        transform_parameters={"a": a, "b": b},
        equivalence_class=EquivalenceClass.EXACT_EQUIVALENT,
        forward_map=lambda x: x, inverse_map=lambda y: y,
        proof_metadata={"relation": "E'(x)=a E(x)+b", "a": a, "b": b},
    )


# --------------------------------------------------------------------------- #
# Family 2: variable permutation  x' = P x
# --------------------------------------------------------------------------- #
def variable_permutation(h: BinaryQuadraticHamiltonian, perm: list[int] | np.ndarray) -> TransformResult:
    perm = np.asarray(perm, dtype=int)
    n = h.n
    if sorted(perm.tolist()) != list(range(n)):
        raise ValueError("perm must be a permutation of 0..n-1.")
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0
    # x = P x'  =>  E'(x') = E(P x')
    new_linear = P.T @ h.linear
    new_quad = P.T @ h.quadratic @ P
    new = BinaryQuadraticHamiltonian(linear=new_linear, quadratic=new_quad,
                                     constant=h.constant, metadata=dict(h.metadata))
    new.metadata["transform"] = "permutation"
    new.metadata["perm"] = perm.tolist()
    return TransformResult(
        hamiltonian=new, transform_name="permutation",
        transform_parameters={"perm": perm.tolist()},
        equivalence_class=EquivalenceClass.EXACT_EQUIVALENT,
        forward_map=lambda x, P=P: (np.asarray(x) @ P),       # x' = x P  (x row vector)
        inverse_map=lambda y, P=P: (np.asarray(y) @ P.T),     # x  = x' P^T
        proof_metadata={"relation": "E'(x')=E(P x')", "perm": perm.tolist()},
    )


# --------------------------------------------------------------------------- #
# Family 3: binary complement on subset S  x_i = 1 - y_i for i in S
# --------------------------------------------------------------------------- #
def bit_complement(h: BinaryQuadraticHamiltonian, S: list[int] | set[int]) -> TransformResult:
    """x_i = 1 - y_i for i in S, x_i = y_i otherwise.  Exact bijective transform.

    Computed algebraically from the canonical polynomial:
      E(x) = c + sum h_i x_i + sum_{i<j} J_ij x_i x_j
    Substitute x_i = 1 - y_i for i in S and expand.
    """
    S = set(int(i) for i in S)
    n = h.n
    lin = h.linear.copy()
    quad = h.quadratic.copy()
    const = h.constant

    # linear term: h_i x_i -> h_i (1 - y_i) = h_i - h_i y_i  for i in S
    for i in S:
        const += lin[i]
        lin[i] = -lin[i]

    # pair term J_ij x_i x_j:
    #   both in S:  J_ij (1-y_i)(1-y_j) = J_ij(1 - y_i - y_j + y_i y_j)
    #   i in S:     J_ij (1-y_i) y_j = J_ij(y_j - y_i y_j)
    #   j in S:     J_ij y_i (1-y_j) = J_ij(y_i - y_i y_j)
    #   neither:    J_ij y_i y_j  (unchanged)
    for i in range(n):
        for j in range(i + 1, n):
            J = quad[i, j]
            if J == 0.0:
                continue
            iS = i in S
            jS = j in S
            if iS and jS:
                const += J
                lin[i] -= J
                lin[j] -= J
                # quad[i,j] stays +J (the y_i y_j term)
            elif iS:
                lin[j] += J
                quad[i, j] = -J
                quad[j, i] = -J
            elif jS:
                lin[i] += J
                quad[i, j] = -J
                quad[j, i] = -J
            # neither: unchanged

    new = BinaryQuadraticHamiltonian(linear=lin, quadratic=quad, constant=const,
                                     metadata=dict(h.metadata))
    new.metadata["transform"] = "bit_complement"
    new.metadata["S"] = sorted(S)

    S_list = sorted(S)
    def forward(x, S_list=S_list):
        x = np.asarray(x).copy().astype(float)
        x[..., S_list] = 1.0 - x[..., S_list]
        return x

    def inverse(y, S_list=S_list):
        y = np.asarray(y).copy().astype(float)
        y[..., S_list] = 1.0 - y[..., S_list]
        return y

    return TransformResult(
        hamiltonian=new, transform_name="bit_complement",
        transform_parameters={"S": sorted(S)},
        equivalence_class=EquivalenceClass.EXACT_EQUIVALENT,
        forward_map=forward, inverse_map=inverse,
        proof_metadata={"relation": "E'(T(x))=E(x)", "S": sorted(S)},
    )


# --------------------------------------------------------------------------- #
# Family 4: MWIS penalty representations  E_U = -w.x + U sum_{(i,j) in E} x_i x_j
# --------------------------------------------------------------------------- #
def mwis_penalty(
    weights: np.ndarray,
    edges: list[tuple[int, int]],
    U: float,
    parent_id: str | None = None,
) -> TransformResult:
    """Construct an MWIS penalty Hamiltonian for a given penalty strength U.

    E_U(x) = -sum w_i x_i + U sum_{(i,j) in E} x_i x_j.
    For U sufficiently large (U > max weight) the ground state is the MWIS.
    Different valid U values encode the same optimum but different landscapes.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    n = weights.shape[0]
    wmax = float(weights.max())
    linear = -weights.copy()
    quadratic = np.zeros((n, n))
    for i, j in edges:
        i, j = (i, j) if i < j else (j, i)
        quadratic[i, j] = U
        quadratic[j, i] = U
    h = BinaryQuadraticHamiltonian(
        linear=linear, quadratic=quadratic, constant=0.0,
        metadata={"type": "mwis_penalty", "U": U, "weights": weights.tolist(),
                  "edges": [tuple(e) for e in edges], "wmax": wmax},
    )
    # ground-state equivalence holds when U > wmax (sufficient condition for
    # positive weights: any independent set beats any set with an edge)
    gs_equiv = U > wmax
    return TransformResult(
        hamiltonian=h, transform_name="mwis_penalty",
        transform_parameters={"U": U, "weights": weights.tolist(),
                              "edges": [tuple(e) for e in edges]},
        equivalence_class=(EquivalenceClass.GROUND_STATE_EQUIVALENT if gs_equiv
                           else EquivalenceClass.APPROXIMATE),
        forward_map=None, inverse_map=None, parent_id=parent_id,
        proof_metadata={"sufficient_condition": "U > max(weights)", "U": U,
                        "wmax": wmax, "gs_equivalent": gs_equiv},
    )


def mwis_penalty_safe_threshold(weights: np.ndarray) -> float:
    """A safe sufficient penalty threshold: U must exceed the max node weight."""
    return float(np.asarray(weights).max())


def mwis_penalty_family(
    weights: np.ndarray, edges: list[tuple[int, int]],
    U_values: list[float],
) -> list[TransformResult]:
    """Generate a family of MWIS penalty Hamiltonians for several U values."""
    return [mwis_penalty(weights, edges, U) for U in U_values]


# --------------------------------------------------------------------------- #
# Family 5: control decomposition (same Hamiltonian, different drive)
# --------------------------------------------------------------------------- #
@dataclass
class ControlRepresentation:
    """A control decomposition of the same endpoint Hamiltonian."""
    name: str
    amplitude_schedule: str   # 'linear', 'blackman', 'piecewise'
    detuning_schedule: str
    duration: float
    parameters: dict
    equivalence_class: EquivalenceClass = EquivalenceClass.SAME_PROBLEM_DIFFERENT_DYNAMICS


def control_representations(
    durations: list[float] | None = None,
    schedules: list[str] | None = None,
) -> list[ControlRepresentation]:
    """Generate alternative control decompositions for the same Hamiltonian.

    These reach the same endpoint Hamiltonian via different adiabatic paths.
    Classified as SAME_PROBLEM_DIFFERENT_DYNAMICS.
    """
    durations = durations or [2.0, 4.0, 6.0]
    schedules = schedules or ["linear", "blackman", "piecewise"]
    out = []
    for dur in durations:
        for sch in schedules:
            out.append(ControlRepresentation(
                name=f"ctrl_{sch}_d{dur}", amplitude_schedule=sch,
                detuning_schedule="linear_ramp", duration=dur,
                parameters={"amp_max": 1.5, "det_max": 5.0},
            ))
    return out


__all__ = [
    "TransformResult", "positive_scale", "variable_permutation", "bit_complement",
    "mwis_penalty", "mwis_penalty_safe_threshold", "mwis_penalty_family",
    "ControlRepresentation", "control_representations",
]
