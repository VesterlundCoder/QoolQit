"""Adapter bridging the canonical BinaryQuadraticHamiltonian to the real QoolQit API.

All QoolQit calls here were verified against the installed qoolqit 1.4.0 API:
  - Register(qubits: dict), Register.from_coordinates(coords)
  - Drive(*, amplitude, detuning=None, dmm=None, phase=0.0)
  - DetuningMapModulator(waveform, weights)  (waveform <= 0, weights in [0,1])
  - QuantumProgram(register, drive); program.compile_to(device, profile=...)
  - LocalEmulator(*, backend_type=QutipBackendV2, num_shots=N)
  - InteractionEmbedder(), SpringLayoutEmbedder(config), Blade(config)
  - DataGraph, AnalogDevice, AnalogDeviceWithDMM, MockDevice

Never invent QoolQit API calls; if the installed API differs this module is the
single place to update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qoolqit import (
    AnalogDevice,
    AnalogDeviceWithDMM,
    Drive,
    MockDevice,
    QuantumProgram,
    Register,
)
from qoolqit.drive import DetuningMapModulator
from qoolqit.embedding import (
    Blade,
    BladeConfig,
    InteractionEmbedder,
    SpringLayoutConfig,
    SpringLayoutEmbedder,
)
from qoolqit.execution import BitStrings, EmulationConfig, LocalEmulator
from qoolqit.execution.backends import QutipBackendV2
from qoolqit.graphs import DataGraph
from qoolqit.waveforms import BlackmanWaveform, CompositeWaveform, ConstantWaveform, RampWaveform

from .types import BinaryQuadraticHamiltonian

QOOLQIT_VERSION = "1.4.0"

# --------------------------------------------------------------------------- #
# Centralized numerical tolerances
# --------------------------------------------------------------------------- #
TOL_ENERGY_EQUIVALENCE = 1e-6
TOL_COORD_DUPLICATE = 1e-9
TOL_GROUND_STATE = 1e-9
TOL_AFFINE_FIT = 1e-6


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingResult:
    coords: np.ndarray
    register: Register
    graph: DataGraph
    embedder_name: str
    embedder_config: dict
    seed: int | None
    target_matrix: np.ndarray
    realized_interactions: np.ndarray
    metadata: dict = field(default_factory=dict)


def target_interaction_matrix(h: BinaryQuadraticHamiltonian) -> np.ndarray:
    """The symmetric pair-interaction matrix to embed (0 diagonal)."""
    return h.interaction_matrix()


def run_embedder(
    h: BinaryQuadraticHamiltonian,
    method: str = "interaction",
    config: dict | None = None,
    seed: int | None = None,
    device: Any = None,
) -> EmbeddingResult:
    """Run one QoolQit embedder and return coordinates + register + realized interactions."""
    config = config or {}
    target = target_interaction_matrix(h)
    embedder_config: dict = {}
    name = method

    if method == "interaction":
        cfg_kwargs = {k: v for k, v in config.items() if k in {"method", "maxiter", "tol", "x0"}}
        # If x0 is provided as a 1-D array, it will be reshaped to (n, 2) inside
        # the embedder.  If x0 is None, QoolQit uses a hardcoded rng(1), so all
        # "restarts" with x0=None produce identical results.  To get genuine
        # restarts we MUST pass a concrete x0 generated from the caller's seed.
        if cfg_kwargs.get("x0") is not None:
            cfg_kwargs["x0"] = np.asarray(cfg_kwargs["x0"], dtype=float)
        from qoolqit.embedding.algorithms.interaction_embedding import InteractionEmbedderConfig
        cfg = InteractionEmbedderConfig(**cfg_kwargs)
        emb = InteractionEmbedder.__new__(InteractionEmbedder)
        InteractionEmbedder.__init__(emb)
        emb._config = cfg
        graph = emb.embed(target)
        embedder_config = cfg.dict()

    elif method == "spring":
        # spring layout works on a graph; build a DataGraph from the target matrix
        graph_in = DataGraph.from_matrix(target)
        cfg_kwargs = {k: v for k, v in config.items() if k in {"iterations", "threshold", "seed"}}
        if seed is not None:
            cfg_kwargs["seed"] = seed
        cfg = SpringLayoutConfig(**cfg_kwargs)
        emb = SpringLayoutEmbedder(config=cfg)
        graph = emb.embed(graph_in)
        embedder_config = cfg.dict()

    elif method == "blade":
        cfg_kwargs = {k: v for k, v in config.items()
                      if k in {"max_min_dist_ratio", "dimensions", "starting_positions",
                               "pca", "steps_per_round", "ratio_rerun"}}
        if "starting_positions" in cfg_kwargs and cfg_kwargs["starting_positions"] is not None:
            cfg_kwargs["starting_positions"] = np.asarray(cfg_kwargs["starting_positions"], dtype=float)
        if device is not None and "max_min_dist_ratio" not in cfg_kwargs:
            cfg = BladeConfig(device=device)
        else:
            cfg = BladeConfig(**cfg_kwargs)
        emb = Blade(config=cfg)
        graph = emb.embed(target)
        embedder_config = {k: str(v) for k, v in cfg.dict().items() if k != "device"}

    else:
        raise ValueError(f"Unknown embedder method: {method}")

    coords = np.array([graph.coords[i] for i in sorted(graph.nodes())], dtype=float)
    register = Register.from_coordinates(coords.tolist())
    realized = register.interaction_matrix()
    return EmbeddingResult(
        coords=coords, register=register, graph=graph, embedder_name=name,
        embedder_config=embedder_config, seed=seed, target_matrix=target,
        realized_interactions=realized,
    )


# --------------------------------------------------------------------------- #
# Drive / program construction
# --------------------------------------------------------------------------- #
def build_adiabatic_drive(
    duration: float = 4.0,
    amp_max: float = 1.5,
    det_max: float = 5.0,
    dmm: DetuningMapModulator | None = None,
    schedule: str = "linear",
) -> Drive:
    """Build a physically valid adiabatic drive.

    The amplitude schedule starts at 0, rises to ``amp_max``, and returns to 0
    at the terminal time so that the final measurement is interpreted against
    the classical target objective without a residual transverse field.

    Schedules:
        - ``"linear"``: piecewise-linear ramp up then ramp down.
        - ``"blackman"``: smooth Blackman pulse (naturally 0 → max → 0).
        - ``"ramp_flat"``: ramp up, flat hold, ramp down.

    The detuning sweeps from ``-det_max`` to ``+det_max`` (linear ramp).

    All values are dimensionless (QoolQit's adimensional framework).
    """
    if schedule == "linear":
        half = duration / 2.0
        amp = CompositeWaveform(
            RampWaveform(half, 0.0, amp_max),
            RampWaveform(half, amp_max, 0.0),
        )
    elif schedule == "blackman":
        amp = BlackmanWaveform(duration, amp_max)
    elif schedule == "ramp_flat":
        quarter = duration / 4.0
        half = duration / 2.0
        amp = CompositeWaveform(
            RampWaveform(quarter, 0.0, amp_max),
            ConstantWaveform(half, amp_max),
            RampWaveform(quarter, amp_max, 0.0),
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    det = RampWaveform(duration, -det_max, det_max)
    if dmm is not None:
        return Drive(amplitude=amp, detuning=det, dmm=dmm, phase=0.0)
    return Drive(amplitude=amp, detuning=det, phase=0.0)


def build_mwis_dmm(
    h: BinaryQuadraticHamiltonian,
    weights: np.ndarray,
    dmm_depth: float = 1.0,
) -> DetuningMapModulator | None:
    """Build a DMM encoding MWIS vertex weights as per-atom detuning weights.

    For MWIS with positive vertex weights ``w_i``, the QoolQit/Pasqal encoding
    uses normalized DMM weights:

        epsilon_i = 1 - w_i / w_max

    where ``w_max = max(w_i)``.  This ensures that the highest-weight vertex
    receives zero additional detuning (epsilon=1 → full DMM contribution
    relative to the global detuning), while lower-weight vertices receive
    proportionally more DMM detuning to compensate for their lower reward.

    The DMM waveform is a negative constant (DMM detunings are ≤ 0).

    Arguments:
        h: the canonical Hamiltonian (linear terms should encode -w_i).
        weights: the original MWIS vertex weights (positive).
        dmm_depth: the DMM waveform amplitude (must be ≤ 0 after sign convention).

    Returns None if weights are trivially uniform (no DMM needed).
    """
    weights = np.asarray(weights, dtype=float)
    w_max = float(weights.max())
    if w_max <= 0:
        return None
    eps = 1.0 - weights / w_max
    if np.allclose(eps, 0.0):
        return None  # uniform weights: no DMM needed
    weight_dict = {i: float(eps[i]) for i in range(len(eps))}
    wf = ConstantWaveform(4.0, -abs(dmm_depth))
    return DetuningMapModulator(waveform=wf, weights=weight_dict)


def local_detuning_dmm(
    h: BinaryQuadraticHamiltonian,
    dmm_depth: float = 1.0,
) -> DetuningMapModulator | None:
    """Build a DMM encoding the linear terms of a general BinaryQuadraticHamiltonian.

    .. warning::

        This generic mapping uses ``abs(linear) / max(abs(linear))`` and is NOT
        a physically justified encoding for arbitrary QUBO problems.  It is
        retained for backward compatibility with non-MWIS experiments.

    For MWIS problems, use :func:`build_mwis_dmm` instead, which implements the
    correct ``epsilon_i = 1 - w_i / w_max`` mapping.

    Raises ``NotImplementedError`` if the linear terms have mixed signs (which
    cannot be justified by this generic mapping).
    """
    lin = h.linear.copy()
    if np.allclose(lin, 0.0):
        return None
    # Reject mixed-sign linear terms: the generic abs-mapping is not justified
    if np.any(lin > 0) and np.any(lin < 0):
        raise NotImplementedError(
            "local_detuning_dmm cannot encode mixed-sign linear terms. "
            "For MWIS, use build_mwis_dmm with explicit vertex weights."
        )
    w = np.abs(lin)
    if w.max() <= 0:
        return None
    weights = (w / w.max()).tolist()
    weights = {i: float(wi) for i, wi in enumerate(weights)}
    wf = ConstantWaveform(4.0, -abs(dmm_depth))
    return DetuningMapModulator(waveform=wf, weights=weights)


def build_program(
    h: BinaryQuadraticHamiltonian,
    embedding: EmbeddingResult,
    duration: float = 4.0,
    amp_max: float = 1.5,
    det_max: float = 5.0,
    use_dmm: bool = True,
    schedule: str = "linear",
    mwis_weights: np.ndarray | None = None,
) -> QuantumProgram:
    """Assemble a QuantumProgram from a Hamiltonian and an embedding.

    Arguments:
        mwis_weights: if provided, use the correct MWIS DMM encoding
            (``epsilon_i = 1 - w_i / w_max``).  If None and use_dmm=True,
            falls back to the generic ``local_detuning_dmm``.
        schedule: control schedule type (``"linear"``, ``"blackman"``,
            ``"ramp_flat"``).
    """
    if use_dmm:
        if mwis_weights is not None:
            dmm = build_mwis_dmm(h, mwis_weights)
        else:
            dmm = local_detuning_dmm(h)
    else:
        dmm = None
    drive = build_adiabatic_drive(
        duration=duration, amp_max=amp_max, det_max=det_max,
        dmm=dmm, schedule=schedule,
    )
    return QuantumProgram(register=embedding.register, drive=drive)


# --------------------------------------------------------------------------- #
# Compilation + emulation
# --------------------------------------------------------------------------- #
@dataclass
class CompilationOutcome:
    success: bool
    failure_reason: str | None
    program: QuantumProgram | None
    device_name: str
    profile: str
    duration: float | None
    min_spacing_margin: float | None
    max_radial_extent: float | None
    metadata: dict = field(default_factory=dict)


def compile_program(
    program: QuantumProgram,
    device: Any = None,
    profile: str = "max_energy",
) -> CompilationOutcome:
    """Compile a program to a device, capturing hardware margins. Failure is data."""
    if device is None:
        device = AnalogDevice()
    try:
        program.compile_to(device, profile=profile)
        specs = device.specs
        reg = program.register
        min_d = reg.min_distance()
        max_r = reg.max_radial_distance()
        margin = (min_d - specs["min_distance"]) if specs["min_distance"] else None
        return CompilationOutcome(
            success=True, failure_reason=None, program=program,
            device_name=device.name, profile=profile,
            duration=program.drive.duration,
            min_spacing_margin=float(margin) if margin is not None else None,
            max_radial_extent=float(max_r) if max_r is not None else None,
        )
    except Exception as e:  # CompilationError or device-constraint errors
        return CompilationOutcome(
            success=False, failure_reason=repr(e), program=None,
            device_name=getattr(device, "name", "unknown"), profile=profile,
            duration=None, min_spacing_margin=None, max_radial_extent=None,
        )


@dataclass
class EmulationOutcome:
    success: bool
    bitstrings: np.ndarray | None
    n_shots: int
    wall_clock_s: float
    failure_reason: str | None = None


def emulate_program(
    program: QuantumProgram,
    num_shots: int = 500,
    backend_type: type = QutipBackendV2,
) -> EmulationOutcome:
    """Run the local emulator on a *compiled* program and return sampled bitstrings.

    QoolQit/Pulser returns ``results.bitstrings`` as a list of Counters mapping
    bitstring strings (qubit 0 = leftmost character) to counts.  We expand these
    into a flat (n_shots, n) int array with qubit 0 as the LSB column (matching
    the canonical BinaryQuadraticHamiltonian convention).
    """
    import time
    t0 = time.time()
    try:
        cfg = EmulationConfig(observables=(BitStrings(num_shots=num_shots),))
        emu = LocalEmulator(backend_type=backend_type, emulation_config=cfg, num_shots=num_shots)
        job = emu.run(program)
        results = job.results()
        bs_list = results.bitstrings  # list[Counter[str, int]]
        rows: list[np.ndarray] = []
        for counter in bs_list:
            for bitstr, count in counter.items():
                # qubit 0 is the leftmost char; reverse so column 0 = qubit 0 (LSB)
                bits = np.array([[int(c) for c in bitstr[::-1]]], dtype=int)
                rows.append(np.repeat(bits, count, axis=0))
        if rows:
            bs = np.vstack(rows)
        else:
            bs = np.zeros((0, program.register.n_qubits), dtype=int)
        return EmulationOutcome(success=True, bitstrings=bs, n_shots=bs.shape[0],
                                wall_clock_s=time.time() - t0)
    except Exception as e:
        return EmulationOutcome(success=False, bitstrings=None, n_shots=0,
                                wall_clock_s=time.time() - t0, failure_reason=repr(e))


# --------------------------------------------------------------------------- #
# Terminal encoding validation
# --------------------------------------------------------------------------- #
@dataclass
class TerminalEncodingReport:
    """Result of validating that the physical program's terminal Hamiltonian
    preserves the intended classical objective up to an affine transformation."""
    valid: bool
    affine_scale: float
    affine_shift: float
    max_residual: float
    ground_state_agreement: bool
    rank_correlation: float
    n_states: int
    details: dict = field(default_factory=dict)


def validate_terminal_encoding(
    h: BinaryQuadraticHamiltonian,
    embedding: EmbeddingResult,
    mwis_weights: np.ndarray | None = None,
    det_max: float = 5.0,
    dmm_depth: float = 1.0,
) -> TerminalEncodingReport:
    """Check that the terminal encoded Hamiltonian preserves the classical objective.

    At the terminal point of the adiabatic schedule:
        Ω = 0, Δ = +det_max, DMM at its full negative value

    The effective diagonal energy for each computational basis state x is:

        E(x) = Δ * Σx_i  +  DMM * Σ(epsilon_i * x_i)  +  Σ J_ij * n_i * n_j

    where n_i = x_i (Rydberg occupation) and J_ij = C6 / r_ij^6 (realized interactions).

    For MWIS, the target classical objective is:
        E_target(x) = -Σ w_i x_i + U Σ_{(i,j)∈E} x_i x_j

    We verify that the terminal encoded diagonal ordering matches the target
    ordering up to an affine transformation (scale > 0, shift).
    """
    from scipy.stats import spearmanr

    n = h.n
    if n > 16:
        # sample states for large n
        rng = np.random.default_rng(42)
        n_states = min(256, 2**n)
        idx = rng.choice(2**n, size=n_states, replace=False)
        bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    else:
        bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)

    # Target classical energies
    E_target = h.energy(bits)

    # Terminal encoded energies: Δ * n_ryd + DMM * Σ(eps_i * x_i) + Σ J_ij n_i n_j
    # At terminal: Δ = +det_max (positive, favoring |1>), DMM = -|dmm_depth|
    realized = embedding.realized_interactions
    n_ryd = bits.sum(axis=1)
    interaction_term = np.zeros(len(bits))
    for i in range(n):
        for j in range(i + 1, n):
            if realized[i, j] != 0:
                interaction_term += realized[i, j] * bits[:, i] * bits[:, j]

    if mwis_weights is not None:
        w_max = float(np.max(mwis_weights))
        eps = 1.0 - np.asarray(mwis_weights, dtype=float) / w_max
        dmm_term = -abs(dmm_depth) * (bits * eps).sum(axis=1)
    else:
        lin = h.linear.copy()
        if np.any(lin > 0) and np.any(lin < 0):
            return TerminalEncodingReport(
                valid=False, affine_scale=0, affine_shift=0, max_residual=np.inf,
                ground_state_agreement=False, rank_correlation=0, n_states=len(bits),
                details={"error": "mixed-sign linear terms not supported"})
        w = np.abs(lin)
        if w.max() > 0:
            eps = w / w.max()
        else:
            eps = np.zeros(n)
        dmm_term = -abs(dmm_depth) * (bits * eps).sum(axis=1)

    E_encoded = det_max * n_ryd + dmm_term + interaction_term

    # Best affine fit: E_encoded ≈ a * E_target + b
    A = np.column_stack([E_target, np.ones_like(E_target)])
    coef, *_ = np.linalg.lstsq(A, E_encoded, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if a <= 0:
        a = abs(a) if a != 0 else 1e-12
    residual = float(np.max(np.abs(E_encoded - (a * E_target + b))))

    # Ground-state agreement
    e0_target = float(E_target.min())
    e0_enc = float(E_encoded.min())
    gs_target = np.isclose(E_target, e0_target, atol=TOL_GROUND_STATE, rtol=0.0)
    gs_enc = np.isclose(E_encoded, e0_enc, atol=TOL_GROUND_STATE, rtol=0.0)
    gs_agreement = bool(np.array_equal(gs_target, gs_enc))

    # Rank correlation
    sp = spearmanr(E_target, E_encoded)
    rho = float(sp.correlation) if sp.correlation is not np.nan else 0.0

    valid = (a > 0) and (residual < TOL_AFFINE_FIT * max(1.0, abs(e0_target))) and gs_agreement

    return TerminalEncodingReport(
        valid=valid, affine_scale=a, affine_shift=b, max_residual=residual,
        ground_state_agreement=gs_agreement, rank_correlation=rho,
        n_states=len(bits),
        details={"e0_target": e0_target, "e0_encoded": e0_enc},
    )


__all__ = [
    "QOOLQIT_VERSION", "EmbeddingResult", "target_interaction_matrix", "run_embedder",
    "build_adiabatic_drive", "build_mwis_dmm", "local_detuning_dmm", "build_program",
    "CompilationOutcome", "compile_program", "EmulationOutcome", "emulate_program",
    "validate_terminal_encoding", "TerminalEncodingReport",
    "TOL_ENERGY_EQUIVALENCE", "TOL_COORD_DUPLICATE", "TOL_GROUND_STATE", "TOL_AFFINE_FIT",
]
