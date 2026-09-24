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
from qoolqit.waveforms import ConstantWaveform, RampWaveform

from .types import BinaryQuadraticHamiltonian

QOOLQIT_VERSION = "1.4.0"


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
        if "x0" in cfg_kwargs and cfg_kwargs["x0"] is not None:
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
) -> Drive:
    """Build a standard adiabatic drive: ramp amplitude up, sweep detuning from negative to positive.

    All values are dimensionless (QoolQit's adimensional framework).
    """
    amp = RampWaveform(duration, 0.0, amp_max)
    det = RampWaveform(duration, -det_max, det_max)
    if dmm is not None:
        return Drive(amplitude=amp, detuning=det, dmm=dmm, phase=0.0)
    return Drive(amplitude=amp, detuning=det, phase=0.0)


def local_detuning_dmm(
    h: BinaryQuadraticHamiltonian,
    dmm_depth: float = 1.0,
) -> DetuningMapModulator | None:
    """Build a DMM encoding the linear terms h_i as per-atom detuning weights.

    The DMM applies a *negative* detuning weighted by epsilon_i in [0,1].  We map
    the linear coefficients to weights proportional to their magnitude.  Returns
    None if there are no meaningful linear terms (so no DMM is needed).
    """
    lin = h.linear.copy()
    if np.allclose(lin, 0.0):
        return None
    # weights in [0,1]; map magnitude. Negative linear terms (rewards) get full weight.
    w = np.abs(lin)
    if w.max() <= 0:
        return None
    weights = (w / w.max()).tolist()
    weights = {i: float(wi) for i, wi in enumerate(weights)}
    # negative waveform
    wf = ConstantWaveform(4.0, -dmm_depth)
    return DetuningMapModulator(waveform=wf, weights=weights)


def build_program(
    h: BinaryQuadraticHamiltonian,
    embedding: EmbeddingResult,
    duration: float = 4.0,
    amp_max: float = 1.5,
    det_max: float = 5.0,
    use_dmm: bool = True,
) -> QuantumProgram:
    """Assemble a QuantumProgram from a Hamiltonian and an embedding."""
    dmm = local_detuning_dmm(h) if use_dmm else None
    drive = build_adiabatic_drive(duration=duration, amp_max=amp_max, det_max=det_max, dmm=dmm)
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


__all__ = [
    "QOOLQIT_VERSION", "EmbeddingResult", "target_interaction_matrix", "run_embedder",
    "build_adiabatic_drive", "local_detuning_dmm", "build_program",
    "CompilationOutcome", "compile_program", "EmulationOutcome", "emulate_program",
]
