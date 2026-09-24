"""Algebra tests: QUBO canonicalization, round trips, scaling, permutation, bit-complement."""

import numpy as np
import pytest

from westquant_qoolqit.common import BinaryQuadraticHamiltonian, solve_exact
from westquant_qoolqit.hamiltonian_explorer import (
    positive_scale, variable_permutation, bit_complement, mwis_penalty,
)
from westquant_qoolqit.common.equivalence import verify_equivalence, EquivalenceClass


def test_qubo_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        n = rng.integers(3, 7)
        Q = rng.uniform(-2, 2, size=(n, n))
        Q = (Q + Q.T) / 2
        h = BinaryQuadraticHamiltonian.from_qubo(Q)
        Q2 = h.to_qubo()
        assert np.allclose(Q, Q2), "QUBO round trip failed"


def test_qubo_energy_consistency():
    Q = np.array([[1.0, 2.0, 0.0], [2.0, -1.0, 3.0], [0.0, 3.0, 0.5]])
    h = BinaryQuadraticHamiltonian.from_qubo(Q)
    x = np.array([1, 0, 1])
    assert np.isclose(float(x @ Q @ x), float(h.energy(x)[0]))


def test_pair_counted_once():
    Q = np.array([[0.0, 3.0], [3.0, 0.0]])
    h = BinaryQuadraticHamiltonian.from_qubo(Q)
    # x^T Q x with x=[1,1] = 6; canonical pair J=6
    assert np.isclose(h.quadratic[0, 1], 6.0)
    assert np.isclose(float(h.energy([1, 1])[0]), 6.0)


def test_scaling_exact_equivalent():
    h = BinaryQuadraticHamiltonian.from_qubo(np.array([[1.0, 2.0], [2.0, -1.0]]))
    tr = positive_scale(h, a=2.5, b=1.3)
    rep = verify_equivalence(h, tr.hamiltonian, forward_map=tr.forward_map)
    assert rep.classification == EquivalenceClass.EXACT_EQUIVALENT
    assert rep.max_energy_error < 1e-9


def test_permutation_exact_equivalent():
    h = BinaryQuadraticHamiltonian.from_qubo(np.array([[1.0, 2.0, 0.0],
                                                        [2.0, -1.0, 3.0],
                                                        [0.0, 3.0, 0.5]]))
    tr = variable_permutation(h, perm=[2, 0, 1])
    rep = verify_equivalence(h, tr.hamiltonian, forward_map=tr.forward_map)
    assert rep.classification == EquivalenceClass.EXACT_EQUIVALENT
    assert rep.max_energy_error < 1e-9


def test_bit_complement_exact_equivalent():
    h = BinaryQuadraticHamiltonian.from_qubo(np.array([[1.0, 2.0, 0.0],
                                                        [2.0, -1.0, 3.0],
                                                        [0.0, 3.0, 0.5]]))
    for S in [[0], [1, 2], [0, 1, 2]]:
        tr = bit_complement(h, S=S)
        rep = verify_equivalence(h, tr.hamiltonian, forward_map=tr.forward_map)
        assert rep.classification == EquivalenceClass.EXACT_EQUIVALENT, f"S={S}"
        assert rep.max_energy_error < 1e-9


def test_constant_shift():
    h = BinaryQuadraticHamiltonian.from_qubo(np.array([[1.0, 2.0], [2.0, -1.0]]))
    tr = positive_scale(h, a=1.0, b=5.0)
    rep = verify_equivalence(h, tr.hamiltonian, forward_map=tr.forward_map)
    assert rep.classification == EquivalenceClass.EXACT_EQUIVALENT


def test_mwis_penalty_ground_state_equivalent():
    weights = np.array([2.0, 3.0, 1.0, 4.0, 2.0])
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    base = mwis_penalty(weights, edges, U=5.0).hamiltonian
    base_sol = solve_exact(base)
    for U in [6.0, 8.0, 12.0]:
        tr = mwis_penalty(weights, edges, U=U)
        assert tr.equivalence_class == EquivalenceClass.GROUND_STATE_EQUIVALENT
        s = solve_exact(tr.hamiltonian)
        assert np.array_equal(np.sort(s.optimum_indices), np.sort(base_sol.optimum_indices))
