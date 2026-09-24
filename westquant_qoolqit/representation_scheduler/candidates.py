"""Representation candidate model for the Representation Scheduler (Project A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RepresentationCandidate:
    """A single physical representation candidate (embedder + config + seed)."""
    id: str
    embedder_name: str
    embedder_config: dict
    seed: int
    coordinates: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "embedder": self.embedder_name,
            "embedder_config": self.embedder_config,
            "seed": self.seed,
            "coordinates": (self.coordinates.tolist() if self.coordinates is not None else None),
            "metadata": self.metadata,
        }


__all__ = ["RepresentationCandidate"]
