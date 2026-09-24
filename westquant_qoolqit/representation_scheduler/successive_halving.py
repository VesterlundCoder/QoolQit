"""Successive-halving staged allocation for representation search.

Avoids emulating hundreds of candidates expensively by staged filtering:
  Stage 0: many candidates  -> cheap geometric/interaction scoring
  Stage 1: ~20% kept          -> logical fidelity + hardware feasibility
  Stage 2: ~20 kept          -> compile
  Stage 3: ~5 kept           -> emulate
  Stage 4: top candidates    -> robustness + deeper emulation
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HalvingConfig:
    stage0_keep: float = 1.0     # fraction kept after cheap scoring
    stage1_keep: int = 20       # absolute number kept after logical fidelity
    stage2_keep: int = 20       # absolute number kept for compilation
    stage3_keep: int = 5        # absolute number kept for emulation
    stage4_keep: int = 3        # absolute number for robustness


def halve_indices(scores: list[float], keep: int | float, minimize: bool = True) -> list[int]:
    """Return indices of the top `keep` candidates by score.

    If keep is a float in (0,1), it is a fraction.  If int, an absolute count.
    """
    n = len(scores)
    if isinstance(keep, float) and 0 < keep <= 1:
        k = max(1, int(round(keep * n)))
    else:
        k = max(1, int(keep))
    k = min(k, n)
    order = sorted(range(n), key=lambda i: scores[i], reverse=not minimize)
    return order[:k]


__all__ = ["HalvingConfig", "halve_indices"]
