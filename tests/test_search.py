"""Tests for equivalence verification, Pareto, robustness, and search reproducibility."""

import numpy as np
import pytest

from westquant_qoolqit.common import (
    BinaryQuadraticHamiltonian, solve_exact, verify_equivalence, EquivalenceClass,
    pareto_front, dominates,
)
from westquant_qoolqit.representation_scheduler import (
    candidate_grid, perturb_coordinates, robustness_stats, robustness_drop,
)
from westquant_qoolqit.hamiltonian_explorer import (
    HamiltonianRepresentationExplorer, classify_realizability, bit_complement,
)


def test_equivalence_invalid():
    h1 = BinaryQuadraticHamiltonian(linear=[1, 0], quadratic=np.array([[0, 1], [1, 0]]))
    h2 = BinaryQuadraticHamiltonian(linear=[0, 5], quadratic=np.array([[0, 0], [0, 0]]))
    rep = verify_equivalence(h1, h2)
    assert rep.classification in (EquivalenceClass.INVALID, EquivalenceClass.APPROXIMATE)


def test_pareto_dominance():
    a = np.array([1.0, 1.0])
    b = np.array([2.0, 2.0])
    assert dominates(a, b)
    assert not dominates(b, a)


def test_pareto_front():
    pts = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [2, 5]])
    front, idx = pareto_front(pts)
    # the five diagonal points are all non-dominated; (2,5) is dominated by (1,5)
    assert len(front) == 5
    assert 5 not in idx  # (2,5) at index 5 is dominated and excluded


def test_pareto_with_maximize():
    pts = np.array([[0.1, 0.5], [0.2, 0.9], [0.3, 0.4]])
    front, idx = pareto_front(pts, maximize=[False, True])
    assert len(front) >= 1


def test_robustness_perturbation():
    coords = np.zeros((4, 2))
    noisy = perturb_coordinates(coords, sigma=0.1, seed=0)
    assert not np.allclose(coords, noisy)
    stats = robustness_stats([0.5, 0.6, 0.4, 0.55])
    assert 0.4 < stats["mean"] < 0.6
    assert robustness_drop(0.6, 0.5) == pytest.approx(0.1)


def test_candidate_grid_reproducible():
    c1 = candidate_grid(seeds=[0, 1])
    c2 = candidate_grid(seeds=[0, 1])
    assert [c.id for c in c1] == [c.id for c in c2]


def test_realizability_classifier():
    # native positive interactions
    h = BinaryQuadraticHamiltonian(linear=[1, 1], quadratic=np.array([[0, 2], [2, 0]]))
    rz = classify_realizability(h)
    assert rz.native_pair_interactions
    # negative interactions after bit complement
    from westquant_qoolqit.hamiltonian_explorer import bit_complement
    tr = bit_complement(h, S=[0])
    rz2 = classify_realizability(tr.hamiltonian)
    assert not rz2.native_pair_interactions


def test_explorer_reproducible():
    h = BinaryQuadraticHamiltonian.from_qubo(np.array([[1.0, 2.0], [2.0, -1.0]]))
    e1 = HamiltonianRepresentationExplorer(seed=42)
    e2 = HamiltonianRepresentationExplorer(seed=42)
    r1 = e1.explore(h)
    r2 = e2.explore(h)
    assert [c.id for c in r1.candidates] == [c.id for c in r2.candidates]


def test_exhaustive_verification():
    rng = np.random.default_rng(1)
    for _ in range(5):
        n = rng.integers(3, 6)
        Q = rng.uniform(-2, 2, size=(n, n))
        Q = (Q + Q.T) / 2
        h = BinaryQuadraticHamiltonian.from_qubo(Q)
        tr = bit_complement(h, S=[0, 1])
        rep = verify_equivalence(h, tr.hamiltonian, forward_map=tr.forward_map)
        assert rep.exhaustive
        assert rep.classification == EquivalenceClass.EXACT_EQUIVALENT
