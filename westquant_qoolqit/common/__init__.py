"""WestQuant QoolQit common layer: canonical types, solvers, metrics, adapter."""

from .types import BinaryQuadraticHamiltonian
from .exact_solver import ExactSolution, solve_exact, mwis_exact
from .equivalence import EquivalenceClass, EquivalenceReport, verify_equivalence
from .metrics import (
    GeometryMetrics, LogicalFidelity, DynamicMetrics,
    interaction_frobenius_error, max_abs_interaction_error,
    interaction_rank_correlation, edge_preservation, geometry_metrics,
    logical_fidelity, dynamic_metrics,
)
from .pareto import dominates, pareto_front, label_pareto
from .reproducibility import EnvironmentManifest, capture_environment, hash_config
from .serialization import write_jsonl, read_jsonl
from .qoolqit_adapter import (
    QOOLQIT_VERSION, EmbeddingResult, run_embedder, build_program,
    compile_program, emulate_program, build_adiabatic_drive,
    build_mwis_dmm, local_detuning_dmm,
    validate_terminal_encoding, TerminalEncodingReport,
)
from .anova import EffectDecomposition, two_way_anova, wilson_ci

__all__ = [
    "BinaryQuadraticHamiltonian", "ExactSolution", "solve_exact", "mwis_exact",
    "EquivalenceClass", "EquivalenceReport", "verify_equivalence",
    "GeometryMetrics", "LogicalFidelity", "DynamicMetrics",
    "interaction_frobenius_error", "max_abs_interaction_error",
    "interaction_rank_correlation", "edge_preservation", "geometry_metrics",
    "logical_fidelity", "dynamic_metrics",
    "dominates", "pareto_front", "label_pareto",
    "EnvironmentManifest", "capture_environment", "hash_config",
    "write_jsonl", "read_jsonl",
    "QOOLQIT_VERSION", "EmbeddingResult", "run_embedder", "build_program",
    "compile_program", "emulate_program", "build_adiabatic_drive",
    "build_mwis_dmm", "local_detuning_dmm",
    "validate_terminal_encoding", "TerminalEncodingReport",
    "EffectDecomposition", "two_way_anova", "wilson_ci",
]
