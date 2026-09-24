"""Robustness evaluation under geometric coordinate perturbation.

Because J_ij ~ 1/r_ij^6, small geometric deviations can produce amplified
interaction errors.  We perturb atomic coordinates with Gaussian noise at
several sigma levels and measure the distribution of a score S.
"""

from __future__ import annotations

import numpy as np


def perturb_coordinates(coords: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add Gaussian noise N(0, sigma^2 I) to coordinates."""
    rng = np.random.default_rng(seed)
    return coords + rng.normal(0.0, sigma, size=coords.shape)


def robustness_stats(samples: np.ndarray) -> dict:
    """Compute E[S], Std[S], Q_0.05[S] for a set of score samples."""
    s = np.asarray(samples, dtype=float)
    return {
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "q05": float(np.quantile(s, 0.05)),
        "worst": float(np.min(s)),
    }


def robustness_drop(nominal: float, mean_noisy: float) -> float:
    """Drop in performance from nominal to mean noisy (>=0 is a degradation)."""
    return float(nominal - mean_noisy)


__all__ = ["perturb_coordinates", "robustness_stats", "robustness_drop"]
