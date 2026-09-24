"""WestQuant Representation Stack for QoolQit.

Two composable open-source contributions for analog neutral-atom quantum
programming, built on Pasqal's QoolQit framework:

  * WestQuant Representation Scheduler (Project A) -- search over physical
    embeddings/realizations for a fixed logical Hamiltonian.
  * WestQuant Hamiltonian Representation Explorer (Project B) -- search over
    mathematically valid Hamiltonian representations of the same problem.

Verified against qoolqit 1.4.0.
"""

from .common import (
    BinaryQuadraticHamiltonian,
    QOOLQIT_VERSION,
    capture_environment,
)

__version__ = "0.1.0"

__all__ = [
    "BinaryQuadraticHamiltonian",
    "QOOLQIT_VERSION",
    "capture_environment",
    "__version__",
]
