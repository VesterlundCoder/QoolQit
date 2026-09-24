"""Reproducibility manifest: record software versions, seeds, git commit."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class EnvironmentManifest:
    qoolqit_version: str
    qoolqit_git_commit: str
    pulser_version: str
    python_version: str
    numpy_version: str
    scipy_version: str
    networkx_version: str
    torch_version: str
    platform: str
    random_seed: int | None
    timestamp: str
    machine: str = field(default_factory=platform.node)

    def to_dict(self) -> dict:
        return asdict(self)


def capture_environment(seed: int | None = None) -> EnvironmentManifest:
    """Capture the current environment for reproducibility."""
    import qoolqit
    try:
        import pulser
        pulser_v = pulser.__version__
    except Exception:
        pulser_v = "n/a"
    try:
        import numpy as np
        numpy_v = np.__version__
    except Exception:
        numpy_v = "n/a"
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "n/a"
    try:
        import networkx
        nx_v = networkx.__version__
    except Exception:
        nx_v = "n/a"
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = "n/a"
    return EnvironmentManifest(
        qoolqit_version=qoolqit.__version__,
        qoolqit_git_commit="installed-pypi",
        pulser_version=pulser_v,
        python_version=sys.version.split()[0],
        numpy_version=numpy_v,
        scipy_version=scipy_v,
        networkx_version=nx_v,
        torch_version=torch_v,
        platform=platform.platform(),
        random_seed=seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def hash_config(config: dict) -> str:
    """Stable SHA256 of a JSON-serializable config dict."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


__all__ = ["EnvironmentManifest", "capture_environment", "hash_config"]
