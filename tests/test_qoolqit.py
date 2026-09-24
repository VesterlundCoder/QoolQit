"""QoolQit integration tests (require qoolqit installed)."""

import numpy as np
import pytest

qoolqit = pytest.importorskip("qoolqit")

from westquant_qoolqit.common import (
    BinaryQuadraticHamiltonian, QOOLQIT_VERSION, run_embedder, build_program,
    compile_program, emulate_program, interaction_frobenius_error,
)
from qoolqit import AnalogDeviceWithDMM, MockDevice


def test_qoolqit_version_recorded():
    assert QOOLQIT_VERSION == "1.4.0"


def test_interaction_embedder_runs():
    h = BinaryQuadraticHamiltonian(linear=[-1, -2, -3, -1],
                                   quadratic=np.array([[0, 4, 0, 0],
                                                       [4, 0, 4, 0],
                                                       [0, 4, 0, 4],
                                                       [0, 0, 4, 0]]))
    emb = run_embedder(h, method="interaction")
    assert emb.coords.shape == (4, 2)
    assert emb.realized_interactions.shape == (4, 4)
    err = interaction_frobenius_error(emb.target_matrix, emb.realized_interactions)
    assert err < 0.5


def test_compile_to_dmm_device():
    h = BinaryQuadraticHamiltonian(linear=[-1, -2, -3, -1],
                                   quadratic=np.array([[0, 4, 0, 0],
                                                       [4, 0, 4, 0],
                                                       [0, 4, 0, 4],
                                                       [0, 0, 4, 0]]))
    emb = run_embedder(h, method="interaction")
    prog = build_program(h, emb, use_dmm=True)
    comp = compile_program(prog, device=AnalogDeviceWithDMM(), profile="max_energy")
    assert comp.success


def test_emulation_returns_bitstrings():
    h = BinaryQuadraticHamiltonian(linear=[-1, -2, -3, -1],
                                   quadratic=np.array([[0, 4, 0, 0],
                                                       [4, 0, 4, 0],
                                                       [0, 4, 0, 4],
                                                       [0, 0, 4, 0]]))
    emb = run_embedder(h, method="interaction")
    prog = build_program(h, emb, use_dmm=True)
    comp = compile_program(prog, device=AnalogDeviceWithDMM(), profile="max_energy")
    if not comp.success:
        pytest.skip("compilation failed on this environment")
    emu = emulate_program(comp.program, num_shots=50)
    assert emu.success
    assert emu.bitstrings.shape[1] == 4
    assert emu.bitstrings.shape[0] == 50


def test_mock_device_compiles():
    h = BinaryQuadraticHamiltonian(linear=[-1, -2], quadratic=np.array([[0, 3], [3, 0]]))
    emb = run_embedder(h, method="interaction")
    prog = build_program(h, emb, use_dmm=False)
    comp = compile_program(prog, device=MockDevice(), profile="default")
    assert comp.success
