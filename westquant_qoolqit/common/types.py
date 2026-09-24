"""Canonical binary-quadratic problem representation.

Internally every quadratic binary objective is represented canonically as

    E(x) = c + sum_i h_i x_i + sum_{i<j} J_{ij} x_i x_j,   x_i in {0,1}

The off-diagonal QUBO coefficients are counted *once* (upper-triangular only),
never twice.  All converters make this convention explicit and are covered by
round-trip tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class BinaryQuadraticHamiltonian:
    """Canonical binary-quadratic objective E(x) = c + h.x + x^T J x (upper-tri).

    Attributes:
        linear: 1D array of length n with the h_i coefficients.
        quadratic: 2D symmetric array of shape (n, n).  Only the upper-triangular
            entries (i < j) are used as the J_{ij} coefficients; the diagonal is
            ignored (there are no x_i^2 terms because x_i in {0,1}).  The matrix
            is stored symmetric for convenience but the *canonical* value of a
            pair interaction is the single upper-triangular entry.
        constant: scalar offset c.
        metadata: free-form provenance metadata.
    """

    linear: np.ndarray
    quadratic: np.ndarray
    constant: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=float).ravel()
        q = np.asarray(self.quadratic, dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("quadratic must be a square 2D array.")
        if q.shape[0] != self.linear.shape[0]:
            raise ValueError("quadratic shape does not match linear length.")
        # Symmetrize defensively; canonical value is the upper-triangular half.
        self.quadratic = np.triu(q, 1)
        self.quadratic = self.quadratic + self.quadratic.T
        self.constant = float(self.constant)

    @property
    def n(self) -> int:
        """Number of binary variables."""
        return self.linear.shape[0]

    def energy(self, x: npt.NDArray[np.float64] | list[int]) -> float:
        """Evaluate E(x) for one or several bitstrings (each row a bitstring)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        # pair term counted once: 0.5 x^T (J+J^T) x with J upper-tri => x^T J_sym x
        # but J_sym has both halves, so use 0.5
        pair = 0.5 * np.einsum("bi,ij,bj->b", x, self.quadratic, x)
        lin = x @ self.linear
        return (pair + lin + self.constant).ravel()

    def energies_all(self) -> npt.NDArray[np.float64]:
        """Energies of all 2^n bitstrings, in lexicographic order (bit 0 = x_0)."""
        if self.n > 22:
            raise ValueError(f"Exhaustive enumeration infeasible for n={self.n}.")
        bits = ((np.arange(2**self.n)[:, None] >> np.arange(self.n)) & 1).astype(float)
        return self.energy(bits)

    def copy(self) -> "BinaryQuadraticHamiltonian":
        return BinaryQuadraticHamiltonian(
            linear=self.linear.copy(),
            quadratic=self.quadratic.copy(),
            constant=self.constant,
            metadata=dict(self.metadata),
        )

    # ------------------------------------------------------------------ #
    # Converters
    # ------------------------------------------------------------------ #
    @classmethod
    def from_qubo(cls, Q: npt.NDArray[np.float64]) -> "BinaryQuadraticHamiltonian":
        """Build from a QUBO matrix Q where E(x) = x^T Q x (x in {0,1}).

        Standard convention: E(x) = x^T Q x = sum_i Q[i,i] x_i + sum_{i<j}
        (Q[i,j]+Q[j,i]) x_i x_j.  The canonical pair coefficient (counted once)
        is therefore J_{ij} = Q[i,j] + Q[j,i] for i<j.  This is unambiguous and
        works for both symmetric and asymmetric Q.
        """
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("QUBO matrix must be square 2D.")
        linear = np.diag(Q).copy()
        # pair coefficient counted once = sum of both halves
        pair_upper = np.triu(Q, 1)
        pair_lower = np.tril(Q, -1).T
        pair = pair_upper + pair_lower  # J_{ij} for i<j
        quadratic = np.zeros_like(Q)
        for i in range(Q.shape[0]):
            for j in range(i + 1, Q.shape[0]):
                quadratic[i, j] = pair[i, j]
                quadratic[j, i] = pair[i, j]
        return cls(linear=linear, quadratic=quadratic, constant=0.0)

    def to_qubo(self) -> npt.NDArray[np.float64]:
        """Return the symmetric QUBO matrix Q with E(x) = x^T Q x.

        With J_{ij} the canonical pair coefficient (counted once), the symmetric
        matrix satisfying x^T Q x = E(x) has Q[i,j] = Q[j,i] = J_{ij}/2 for i<j.
        """
        Q = np.zeros((self.n, self.n))
        np.fill_diagonal(Q, self.linear)
        Q = Q + 0.5 * np.triu(self.quadratic, 1)
        Q = Q + 0.5 * np.tril(self.quadratic, -1)
        return Q

    @classmethod
    def from_ising(
        cls,
        h: npt.NDArray[np.float64],
        J: dict[tuple[int, int], float],
        offset: float = 0.0,
    ) -> "BinaryQuadraticHamiltonian":
        """Build from Ising variables z_i in {-1,+1} with H = sum h_i z_i + sum J_ij z_i z_j.

        Uses the standard mapping x_i = (1 + z_i)/2.
        """
        h = np.asarray(h, dtype=float).ravel()
        n = h.shape[0]
        linear = np.zeros(n)
        quadratic = np.zeros((n, n))
        constant = offset
        # z_i = 2 x_i - 1
        # h_i z_i = h_i (2 x_i - 1) = 2 h_i x_i - h_i
        linear += 2.0 * h
        constant -= float(np.sum(h))
        for (i, j), val in J.items():
            i, j = (i, j) if i < j else (j, i)
            # J_ij z_i z_j = J_ij (2x_i-1)(2x_j-1) = 4 J_ij x_i x_j - 2 J_ij x_i - 2 J_ij x_j + J_ij
            quadratic[i, j] += 4.0 * val
            quadratic[j, i] += 4.0 * val
            linear[i] -= 2.0 * val
            linear[j] -= 2.0 * val
            constant += val
        return cls(linear=linear, quadratic=quadratic, constant=constant)

    def to_ising(self) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
        """Convert to Ising (z in {-1,+1}). Returns (h, J, offset)."""
        # x_i = (1+z_i)/2.  E = c + sum h_i x_i + sum_{i<j} J_ij x_i x_j
        # h_i x_i = h_i/2 + h_i/2 z_i
        # J_ij x_i x_j = J_ij/4 (1+z_i)(1+z_j) = J_ij/4 + J_ij/4 z_i + J_ij/4 z_j + J_ij/4 z_i z_j
        h = np.zeros(self.n)
        J: dict[tuple[int, int], float] = {}
        offset = self.constant
        h += 0.5 * self.linear
        offset += 0.5 * float(np.sum(self.linear))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                val = self.quadratic[i, j]
                if val == 0.0:
                    continue
                offset += 0.25 * val
                h[i] += 0.25 * val
                h[j] += 0.25 * val
                J[(i, j)] = 0.25 * val
        return h, J, offset

    def interaction_matrix(self) -> npt.NDArray[np.float64]:
        """Symmetric matrix of pair interactions (pair counted once, 0 diagonal).

        This is the *target* interaction matrix for Rydberg embedding: the
        off-diagonal entries are the J_{ij} pair coefficients.
        """
        return self.quadratic.copy()

    def __repr__(self) -> str:
        return (
            f"BinaryQuadraticHamiltonian(n={self.n}, constant={self.constant:.4g}, "
            f"linear_range=[{self.linear.min():.4g},{self.linear.max():.4g}])"
        )


__all__ = ["BinaryQuadraticHamiltonian"]
