"""Candidate generation from QoolQit embedders.

Generates a grid of RepresentationCandidate specifications covering the
available QoolQit embedding methods (interaction, spring, blade) and their
hyperparameters.  Each candidate is reproducible from its config + seed.

Key design decisions:
- InteractionEmbedder uses a hardcoded rng(1) when x0=None.  To get genuine
  restarts, we MUST pass a concrete x0 array generated from the caller's seed.
- Different Monte Carlo samples from the same embedder with the same config
  are NOT different representations — they are stochastic replicates.
- The grid explicitly distinguishes embedding family, configuration, and
  initialization/restart.
"""

from __future__ import annotations

import numpy as np

from .candidates import RepresentationCandidate


def candidate_grid(
    embedders: list[str] | None = None,
    seeds: list[int] | None = None,
    n_interaction_restarts: int = 3,
    spring_iterations: list[int] | None = None,
    blade_dimensions: list[tuple[int, ...]] | None = None,
    n_qubits: int = 10,
) -> list[RepresentationCandidate]:
    """Build a grid of representation candidate specifications.

    Arguments:
        embedders: which embedder families to include.
        seeds: seeds for stochastic embedders.
        n_interaction_restarts: number of restarts for InteractionEmbedder
            (each with a different x0 generated from the seed).
        spring_iterations: iteration counts for SpringLayoutEmbedder.
        blade_dimensions: dimension configs for Blade.
        n_qubits: number of qubits (used to size x0 for InteractionEmbedder).
    """
    embedders = embedders or ["interaction", "spring", "blade"]
    seeds = seeds or [0, 1, 2]
    spring_iterations = spring_iterations or [50, 200]
    blade_dimensions = blade_dimensions or [(5, 4, 3, 2), (3, 2)]
    cands: list[RepresentationCandidate] = []

    if "interaction" in embedders:
        for seed in seeds:
            for r in range(n_interaction_restarts):
                # Generate x0 from seed+restart so different restarts give
                # different initial positions → genuinely different optima.
                rng = np.random.default_rng(seed * 1000 + r)
                x0 = rng.random(n_qubits * 2).tolist()
                cands.append(RepresentationCandidate(
                    id=f"R_int_s{seed}_r{r}", embedder_name="interaction",
                    embedder_config={"method": "Nelder-Mead", "maxiter": 5000,
                                     "tol": 1e-8, "x0": x0},
                    seed=seed, metadata={"restart": r, "embedding_family": "interaction"}))

    if "spring" in embedders:
        for seed in seeds:
            for it in spring_iterations:
                cands.append(RepresentationCandidate(
                    id=f"R_spr_s{seed}_it{it}", embedder_name="spring",
                    embedder_config={"iterations": it, "threshold": 1e-4, "seed": seed},
                    seed=seed, metadata={"iterations": it, "embedding_family": "spring"}))

    if "blade" in embedders:
        for seed in seeds:
            for dims in blade_dimensions:
                cands.append(RepresentationCandidate(
                    id=f"R_bld_s{seed}_d{len(dims)}", embedder_name="blade",
                    embedder_config={"dimensions": list(dims), "steps_per_round": 100,
                                     "pca": True, "starting_positions": None},
                    seed=seed, metadata={"dimensions": list(dims), "embedding_family": "blade"}))
    return cands


def candidate_grid_stratified(
    embedders: list[str] | None = None,
    n_per_family: int = 3,
    seed: int = 42,
    n_qubits: int = 10,
) -> list[RepresentationCandidate]:
    """Build a grid with exactly ``n_per_family`` candidates per embedder family.

    This ensures the factorial experiment has genuine diversity across embedding
    families rather than collapsing to a single family.

    Each candidate gets a unique seed derived from the base seed.
    """
    embedders = embedders or ["interaction", "spring", "blade"]
    cands: list[RepresentationCandidate] = []
    cid = 0

    for family in embedders:
        for i in range(n_per_family):
            s = seed + cid
            if family == "interaction":
                rng = np.random.default_rng(s * 1000 + i)
                x0 = rng.random(n_qubits * 2).tolist()
                cands.append(RepresentationCandidate(
                    id=f"R_int_s{s}_r{i}", embedder_name="interaction",
                    embedder_config={"method": "Nelder-Mead", "maxiter": 5000,
                                     "tol": 1e-8, "x0": x0},
                    seed=s, metadata={"restart": i, "embedding_family": "interaction"}))
            elif family == "spring":
                iters = [50, 200, 500][i % 3]
                cands.append(RepresentationCandidate(
                    id=f"R_spr_s{s}_it{iters}", embedder_name="spring",
                    embedder_config={"iterations": iters, "threshold": 1e-4, "seed": s},
                    seed=s, metadata={"iterations": iters, "embedding_family": "spring"}))
            elif family == "blade":
                dims_options = [(5, 4, 3, 2), (3, 2), (4, 3, 2)]
                dims = dims_options[i % len(dims_options)]
                cands.append(RepresentationCandidate(
                    id=f"R_bld_s{s}_d{len(dims)}", embedder_name="blade",
                    embedder_config={"dimensions": list(dims), "steps_per_round": 100,
                                     "pca": True, "starting_positions": None},
                    seed=s, metadata={"dimensions": list(dims), "embedding_family": "blade"}))
            cid += 1
    return cands


__all__ = ["candidate_grid", "candidate_grid_stratified"]
