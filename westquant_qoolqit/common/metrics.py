"""Cheap and dynamic metrics for representation evaluation.

Cheap metrics (geometry + interaction fidelity + logical fidelity) require no
emulation.  Dynamic metrics require emulator output and are computed here from
bitstring samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from .types import BinaryQuadraticHamiltonian


# --------------------------------------------------------------------------- #
# Cheap embedding / interaction metrics
# --------------------------------------------------------------------------- #
def interaction_frobenius_error(target: np.ndarray, realized: np.ndarray, eps: float = 1e-12) -> float:
    """Relative Frobenius error ||U - J(R)||_F / (||U||_F + eps)."""
    t = np.asarray(target, dtype=float)
    r = np.asarray(realized, dtype=float)
    return float(np.linalg.norm(t - r) / (np.linalg.norm(t) + eps))


def max_abs_interaction_error(target: np.ndarray, realized: np.ndarray) -> float:
    t = np.asarray(target, dtype=float)
    r = np.asarray(realized, dtype=float)
    iu = np.triu_indices_from(t, k=1)
    return float(np.max(np.abs(t[iu] - r[iu]))) if iu[0].size else 0.0


def interaction_rank_correlation(target: np.ndarray, realized: np.ndarray) -> float:
    """Spearman correlation between desired and realized pair interactions."""
    t = np.asarray(target, dtype=float)
    r = np.asarray(realized, dtype=float)
    iu = np.triu_indices_from(t, k=1)
    if iu[0].size < 2:
        return 1.0
    sp = spearmanr(t[iu], r[iu])
    return float(sp.correlation) if sp.correlation is not np.nan else 0.0


def edge_preservation(target: np.ndarray, realized: np.ndarray, threshold: float | None = None) -> float:
    """Fraction of strong target interactions preserved as strong realized ones."""
    t = np.asarray(target, dtype=float)
    r = np.asarray(realized, dtype=float)
    iu = np.triu_indices_from(t, k=1)
    if iu[0].size == 0:
        return 1.0
    if threshold is None:
        threshold = np.median(np.abs(t[iu])) if np.any(t[iu] != 0) else 0.0
    strong_t = np.abs(t[iu]) >= threshold
    strong_r = np.abs(r[iu]) >= threshold
    if strong_t.sum() == 0:
        return 1.0
    return float(np.mean(strong_t & strong_r))


@dataclass
class GeometryMetrics:
    min_distance: float
    max_distance: float
    radius: float
    diameter: float
    bbox_area: float
    pairwise_distance_mean: float
    pairwise_distance_std: float


def geometry_metrics(coords: np.ndarray) -> GeometryMetrics:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    from scipy.spatial.distance import pdist
    d = pdist(coords)
    radial = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    bbox = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
    return GeometryMetrics(
        min_distance=float(d.min()) if d.size else 0.0,
        max_distance=float(d.max()) if d.size else 0.0,
        radius=float(radial.max()) if radial.size else 0.0,
        diameter=float(d.max()) if d.size else 0.0,
        bbox_area=float(bbox),
        pairwise_distance_mean=float(d.mean()) if d.size else 0.0,
        pairwise_distance_std=float(d.std()) if d.size else 0.0,
    )


# --------------------------------------------------------------------------- #
# Logical objective fidelity
# --------------------------------------------------------------------------- #
@dataclass
class LogicalFidelity:
    energy_rank_correlation: float
    ground_state_agreement: bool
    top_k_agreement: float
    energy_distortion_residual: float
    affine_scale: float
    affine_shift: float


def logical_fidelity(
    logical: BinaryQuadraticHamiltonian,
    realized: BinaryQuadraticHamiltonian,
    top_k: int = 8,
) -> LogicalFidelity:
    """Compare the logical objective E_logical(x) with the realized E_realized(x).

    Both must share the same variable space (same n).  For small n we enumerate
    all bitstrings; otherwise we sample.
    """
    n = logical.n
    assert realized.n == n
    if n <= 18:
        bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(2**n, size=min(4096, 2**n), replace=False)
        bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    E_l = logical.energy(bits)
    E_r = realized.energy(bits)
    sp = spearmanr(E_l, E_r)
    rho = float(sp.correlation) if sp.correlation is not np.nan else 0.0
    e0l = float(E_l.min())
    e0r = float(E_r.min())
    gs_l = np.isclose(E_l, e0l, atol=1e-9, rtol=0.0)
    gs_r = np.isclose(E_r, e0r, atol=1e-9, rtol=0.0)
    gs_agree = bool(np.array_equal(gs_l, gs_r))
    k = min(top_k, len(E_l))
    top_l = set(np.argsort(E_l)[:k].tolist())
    top_r = set(np.argsort(E_r)[:k].tolist())
    topk = len(top_l & top_r) / k
    A = np.column_stack([E_l, np.ones_like(E_l)])
    coef, *_ = np.linalg.lstsq(A, E_r, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = float(np.max(np.abs(E_r - (a * E_l + b))))
    return LogicalFidelity(
        energy_rank_correlation=rho,
        ground_state_agreement=gs_agree,
        top_k_agreement=topk,
        energy_distortion_residual=resid,
        affine_scale=a,
        affine_shift=b,
    )


# --------------------------------------------------------------------------- #
# Dynamic / emulator metrics
# --------------------------------------------------------------------------- #
@dataclass
class DynamicMetrics:
    ground_state_probability: float
    probability_any_optimal: float
    expected_objective: float
    best_sampled_objective: float
    top_k_solution_probability: float
    approximation_ratio: float
    entropy: float
    n_shots: int


def dynamic_metrics(
    bitstrings: np.ndarray | Sequence,
    logical: BinaryQuadraticHamiltonian,
    optimum_energy: float,
    optimum_states: set | None = None,
    top_k: int = 8,
) -> DynamicMetrics:
    """Compute dynamic metrics from sampled bitstrings against known truth."""
    bs = np.asarray(bitstrings, dtype=int)
    if bs.ndim == 1:
        bs = bs.reshape(1, -1)
    n_shots = bs.shape[0]
    energies = logical.energy(bs.astype(float))
    # optimum states as a set of integer indices for exact membership
    if optimum_states is None:
        # recompute from logical (small n)
        sol_bits = ((np.arange(2**logical.n)[:, None] >> np.arange(logical.n)) & 1).astype(int)
        sol_e = logical.energy(sol_bits.astype(float))
        e0 = float(sol_e.min())
        opt_idx = np.where(np.isclose(sol_e, e0, atol=1e-9, rtol=0.0))[0]
        optimum_states = set(int(i) for i in opt_idx)
    # membership: convert each sampled bitstring to its integer index
    idx = bs @ (1 << np.arange(bs.shape[1]))
    p_opt = float(np.mean([int(int(i) in optimum_states) for i in idx]))
    p_gs = p_opt  # ground-state probability == probability of any optimal
    exp_obj = float(np.mean(energies))
    best_obj = float(energies.min())
    # approximation ratio: best/|optimum| style (higher is better, capped)
    denom = abs(optimum_energy) if abs(optimum_energy) > 1e-12 else 1.0
    approx = float(best_obj / denom) if optimum_energy < 0 else float(optimum_energy / best_obj if best_obj != 0 else 0.0)
    # top-k solution probability
    k = min(top_k, len(optimum_states))
    top_set = set(list(optimum_states)[:k])
    p_topk = float(np.mean([int(int(i) in top_set) for i in idx]))
    # entropy of output distribution
    from collections import Counter
    cnt = Counter(idx.tolist())
    probs = np.array(list(cnt.values()), dtype=float) / n_shots
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return DynamicMetrics(
        ground_state_probability=p_gs,
        probability_any_optimal=p_opt,
        expected_objective=exp_obj,
        best_sampled_objective=best_obj,
        top_k_solution_probability=p_topk,
        approximation_ratio=approx,
        entropy=entropy,
        n_shots=n_shots,
    )


__all__ = [
    "interaction_frobenius_error", "max_abs_interaction_error",
    "interaction_rank_correlation", "edge_preservation",
    "GeometryMetrics", "geometry_metrics",
    "LogicalFidelity", "logical_fidelity",
    "DynamicMetrics", "dynamic_metrics",
]
