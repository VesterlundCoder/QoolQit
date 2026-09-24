"""Two-way ANOVA and effect decomposition for factorial experiments.

Implements the experimental model:

    Y_{hrk} = mu + alpha_h + beta_r + (alpha*beta)_{hr} + epsilon_{hrk}

where:
    h = Hamiltonian representation
    r = embedding representation
    k = stochastic replicate

Returns sums of squares, eta-squared effect sizes, and degrees of freedom.

This is a proper factorial analysis that separates:
- Hamiltonian main effect (H)
- Embedding main effect (R)
- H × R interaction
- Residual / sampling variation

Unlike the old var(row_means) + var(col_means) approach, this correctly
accounts for the interaction term and uses replicates to estimate residual
variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EffectDecomposition:
    """Result of a two-way ANOVA effect decomposition."""
    ss_H: float          # Sum of squares for Hamiltonian main effect
    ss_R: float          # Sum of squares for embedding main effect
    ss_HxR: float        # Sum of squares for H × R interaction
    ss_residual: float   # Residual sum of squares (within-cell variation)
    ss_total: float      # Total sum of squares
    df_H: int            # Degrees of freedom for H
    df_R: int            # Degrees of freedom for R
    df_HxR: int          # Degrees of freedom for H × R
    df_residual: int     # Degrees of freedom for residual
    eta2_H: float        # Effect size (eta-squared) for H
    eta2_R: float        # Effect size (eta-squared) for R
    eta2_HxR: float      # Effect size (eta-squared) for H × R
    eta2_residual: float # Effect size (eta-squared) for residual
    n_cells: int         # Total number of (H, R) cells
    n_replicates: int    # Replicates per cell (min)
    grand_mean: float
    details: dict = field(default_factory=dict)


def two_way_anova(
    Y: np.ndarray,
    n_H: int,
    n_R: int,
    n_replicates: int,
) -> EffectDecomposition:
    """Perform a two-way ANOVA on a balanced factorial design.

    Arguments:
        Y: array of shape (n_H, n_R, n_replicates) containing observations.
           NaN values are treated as missing and excluded from that cell.
        n_H: number of Hamiltonian levels.
        n_R: number of embedding levels.
        n_replicates: number of replicates per cell.

    Returns:
        EffectDecomposition with sums of squares, degrees of freedom, and
        eta-squared effect sizes.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.shape != (n_H, n_R, n_replicates):
        # Try to reshape
        Y = Y.reshape(n_H, n_R, n_replicates)

    # Replace NaN with cell mean (or grand mean if entire cell is NaN)
    grand_mean = np.nanmean(Y)
    for h in range(n_H):
        for r in range(n_R):
            cell = Y[h, r, :]
            mask = ~np.isnan(cell)
            if not np.any(mask):
                Y[h, r, :] = grand_mean
            elif not np.all(mask):
                cell_mean = np.nanmean(cell)
                Y[h, r, ~mask] = cell_mean

    # Grand mean
    mu = float(Y.mean())

    # Cell means
    cell_means = Y.mean(axis=2)  # (n_H, n_R)
    row_means = cell_means.mean(axis=1)  # (n_H,)
    col_means = cell_means.mean(axis=0)  # (n_R,)

    # Sums of squares
    ss_total = float(np.sum((Y - mu) ** 2))
    ss_H = float(n_R * n_replicates * np.sum((row_means - mu) ** 2))
    ss_R = float(n_H * n_replicates * np.sum((col_means - mu) ** 2))
    ss_HxR = float(n_replicates * np.sum(
        (cell_means - row_means[:, None] - col_means[None, :] + mu) ** 2))
    ss_residual = float(np.sum((Y - cell_means[:, :, None]) ** 2))

    # Degrees of freedom
    df_H = n_H - 1
    df_R = n_R - 1
    df_HxR = (n_H - 1) * (n_R - 1)
    df_residual = n_H * n_R * (n_replicates - 1)

    # Eta-squared
    eta2_H = ss_H / ss_total if ss_total > 0 else 0.0
    eta2_R = ss_R / ss_total if ss_total > 0 else 0.0
    eta2_HxR = ss_HxR / ss_total if ss_total > 0 else 0.0
    eta2_residual = ss_residual / ss_total if ss_total > 0 else 0.0

    return EffectDecomposition(
        ss_H=ss_H, ss_R=ss_R, ss_HxR=ss_HxR, ss_residual=ss_residual,
        ss_total=ss_total,
        df_H=df_H, df_R=df_R, df_HxR=df_HxR, df_residual=df_residual,
        eta2_H=eta2_H, eta2_R=eta2_R, eta2_HxR=eta2_HxR, eta2_residual=eta2_residual,
        n_cells=n_H * n_R, n_replicates=n_replicates,
        grand_mean=mu,
    )


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Arguments:
        p: observed proportion.
        n: number of trials.
        z: z-score (default 1.96 for 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


__all__ = ["EffectDecomposition", "two_way_anova", "wilson_ci"]
