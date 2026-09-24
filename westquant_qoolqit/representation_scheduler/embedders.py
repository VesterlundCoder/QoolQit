"""Candidate generation from QoolQit embedders.

Generates a grid of RepresentationCandidate specifications covering the
available QoolQit embedding methods (interaction, spring, blade) and their
hyperparameters.  Each candidate is reproducible from its config + seed.
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
) -> list[RepresentationCandidate]:
    """Build a grid of representation candidate specifications."""
    embedders = embedders or ["interaction", "spring", "blade"]
    seeds = seeds or [0, 1, 2]
    spring_iterations = spring_iterations or [50, 200]
    blade_dimensions = blade_dimensions or [(5, 4, 3, 2), (3, 2)]
    cands: list[RepresentationCandidate] = []
    cid = 0

    if "interaction" in embedders:
        for seed in seeds:
            for r in range(n_interaction_restarts):
                rng = np.random.default_rng(seed * 100 + r)
                x0 = rng.random(20).tolist()  # will be reshaped per problem n
                cands.append(RepresentationCandidate(
                    id=f"R_int_s{seed}_r{r}", embedder_name="interaction",
                    embedder_config={"method": "Nelder-Mead", "maxiter": 5000,
                                     "tol": 1e-8, "x0": None},
                    seed=seed, metadata={"restart": r}))

    if "spring" in embedders:
        for seed in seeds:
            for it in spring_iterations:
                cands.append(RepresentationCandidate(
                    id=f"R_spr_s{seed}_it{it}", embedder_name="spring",
                    embedder_config={"iterations": it, "threshold": 1e-4, "seed": seed},
                    seed=seed))

    if "blade" in embedders:
        for seed in seeds:
            for dims in blade_dimensions:
                cands.append(RepresentationCandidate(
                    id=f"R_bld_s{seed}_d{len(dims)}", embedder_name="blade",
                    embedder_config={"dimensions": list(dims), "steps_per_round": 100,
                                     "pca": True, "starting_positions": None},
                    seed=seed, metadata={"dimensions": list(dims)}))
    return cands


__all__ = ["candidate_grid"]
