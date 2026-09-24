"""Combined hierarchical representation search.

    P -> {H_i} -> {R_ij} -> {Programs_ijk} -> Pareto frontier

This is the closest implementation to the full WestQuant representation-search
philosophy: it searches *both* the logical Hamiltonian representation (Project B)
and the physical embedding (Project A) in a nested fashion, using successive
halving at each level to keep expensive emulation cheap.

The flagship experiment is a factorial matrix:

              R_1   R_2   R_3
    H_{U_1}    .     .     .
    H_{U_2}    .     .     .
    H_{U_3}    .     .     .

asking: how much variation comes from H, and how much from R?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qoolqit import AnalogDeviceWithDMM

from .common.types import BinaryQuadraticHamiltonian
from .common.exact_solver import solve_exact
from .common.qoolqit_adapter import (
    run_embedder, build_program, compile_program, emulate_program,
)
from .common.metrics import (
    interaction_frobenius_error, logical_fidelity, dynamic_metrics,
)
from .common.serialization import write_jsonl
from .hamiltonian_explorer import HamiltonianRepresentationExplorer
from .representation_scheduler import (
    RepresentationScheduler, HalvingConfig, candidate_grid,
)
from .representation_scheduler.successive_halving import halve_indices


@dataclass
class FactorialCell:
    """One cell of the H x R factorial matrix."""
    hamiltonian_id: str
    embedder: str
    seed: int
    equivalence: str
    frobenius_error: float | None
    logical_fidelity_rho: float | None
    compilation_success: bool | None
    duration: float | None
    ground_state_probability: float | None
    expected_objective: float | None
    robustness: float | None


@dataclass
class CombinedResult:
    """Result of a combined hierarchical search."""
    cells: list[FactorialCell]
    hamiltonian_ids: list[str]
    embedders: list[str]
    problem: BinaryQuadraticHamiltonian
    records: list[dict] = field(default_factory=list)

    def matrix(self, metric: str = "ground_state_probability") -> np.ndarray:
        """Return the H x R matrix for one metric (NaN where unavailable)."""
        H = sorted(set(c.hamiltonian_id for c in self.cells),
                   key=lambda s: self.hamiltonian_ids.index(s) if s in self.hamiltonian_ids else 0)
        R = sorted(set((c.embedder, c.seed) for c in self.cells), key=lambda t: str(t))
        mat = np.full((len(H), len(R)), np.nan)
        hidx = {h: i for i, h in enumerate(H)}
        ridx = {r: i for i, r in enumerate(R)}
        for c in self.cells:
            val = getattr(c, metric, None)
            if val is not None:
                mat[hidx[c.hamiltonian_id], ridx[(c.embedder, c.seed)]] = val
        return mat

    def variance_decomposition(self, metric: str = "ground_state_probability") -> dict:
        """Decompose total variance into H (Hamiltonian) and R (embedding) components."""
        mat = self.matrix(metric)
        mask = ~np.isnan(mat)
        if mask.sum() < 4:
            return {"H_variance": 0.0, "R_variance": 0.0, "residual": 0.0,
                    "frac_H": 0.0, "frac_R": 0.0}
        # fill NaN with column means for a rough decomposition
        col_means = np.nanmean(mat, axis=0)
        row_means = np.nanmean(mat, axis=1)
        grand = np.nanmean(mat)
        H_var = np.nanvar(row_means)
        R_var = np.nanvar(col_means)
        total = np.nanvar(mat)
        return {
            "H_variance": float(H_var),
            "R_variance": float(R_var),
            "total_variance": float(total),
            "frac_H": float(H_var / total) if total > 0 else 0.0,
            "frac_R": float(R_var / total) if total > 0 else 0.0,
        }


def combined_search(
    problem: BinaryQuadraticHamiltonian,
    mwis_info: dict | None = None,
    n_hamiltonians: int = 3,
    n_embedders: int = 3,
    embedder_methods: list[str] | None = None,
    num_shots: int = 300,
    device: Any = None,
    seed: int = 42,
    run_emulation: bool = True,
    run_robustness: bool = True,
    robustness_samples: int = 3,
) -> CombinedResult:
    """Run the combined H x R factorial search.

    For MWIS problems, generates `n_hamiltonians` penalty Hamiltonians (varying U)
    and `n_embedders` embeddings per Hamiltonian, forming a factorial matrix.
    """
    device = device or AnalogDeviceWithDMM()
    embedder_methods = embedder_methods or ["interaction", "spring", "blade"]

    # Level 0: generate Hamiltonian representations
    explorer = HamiltonianRepresentationExplorer(
        transforms=["mwis_penalty"] if mwis_info else ["positive_scale"],
        verify=True, seed=seed,
    )
    exp = explorer.explore(problem, mwis_info=mwis_info)
    # keep ground-state-equivalent, native-Rydberg Hamiltonians
    h_candidates = [c for c in exp.candidates
                    if c.equivalence_class.value in ("EXACT_EQUIVALENT",
                                                      "GROUND_STATE_EQUIVALENT")]
    if mwis_info:
        h_candidates = [c for c in h_candidates if c.transform_name == "mwis_penalty"]
    # limit to n_hamiltonians, spread across U values
    if len(h_candidates) > n_hamiltonians:
        step = max(1, len(h_candidates) // n_hamiltonians)
        h_candidates = h_candidates[::step][:n_hamiltonians]
    h_candidates = h_candidates[:n_hamiltonians]
    if not h_candidates:
        h_candidates = [exp.candidates[0]]

    # exact optima per Hamiltonian
    optima = {}
    for c in h_candidates:
        sol = solve_exact(c.hamiltonian) if c.hamiltonian.n <= 22 else None
        optima[c.id] = (set(int(i) for i in sol.optimum_indices) if sol else None,
                        sol.optimum_energy if sol else None)

    # Level 1-5: for each Hamiltonian, run a small embedding search
    cells: list[FactorialCell] = []
    records: list[dict] = []
    seeds = list(range(seed, seed + n_embedders))
    cands = candidate_grid(embedders=embedder_methods, seeds=seeds)
    # keep one candidate per (method, seed)
    seen = set()
    embedder_cands = []
    for c in cands:
        key = (c.embedder_name, c.seed)
        if key in seen:
            continue
        seen.add(key)
        embedder_cands.append(c)
        if len(embedder_cands) >= n_embedders:
            break

    for hc in h_candidates:
        for ec in embedder_cands:
            cell = FactorialCell(
                hamiltonian_id=hc.id, embedder=ec.embedder_name, seed=ec.seed,
                equivalence=hc.equivalence_class.value,
                frobenius_error=None, logical_fidelity_rho=None,
                compilation_success=None, duration=None,
                ground_state_probability=None, expected_objective=None,
                robustness=None,
            )
            rec = {"hamiltonian_id": hc.id, "embedder": ec.embedder_name, "seed": ec.seed,
                   "equivalence": hc.equivalence_class.value, "transform": hc.transform_name,
                   "U": hc.transform_parameters.get("U")}
            try:
                emb = run_embedder(hc.hamiltonian, method=ec.embedder_name,
                                   config=ec.embedder_config, seed=ec.seed, device=device)
                cell.frobenius_error = interaction_frobenius_error(
                    emb.target_matrix, emb.realized_interactions)
                rec["embedding_error"] = cell.frobenius_error
                realized = hc.hamiltonian.copy()
                realized.quadratic = emb.realized_interactions
                lf = logical_fidelity(hc.hamiltonian, realized)
                cell.logical_fidelity_rho = lf.energy_rank_correlation
                rec["logical_fidelity_rho"] = lf.energy_rank_correlation
                # compile + emulate
                comp = None
                for use_dmm, dev, prof in [(True, device, "max_energy"),
                                           (False, device, "max_energy")]:
                    prog = build_program(hc.hamiltonian, emb, use_dmm=use_dmm)
                    comp = compile_program(prog, device=dev, profile=prof)
                    if comp.success:
                        break
                cell.compilation_success = comp.success if comp else False
                rec["compilation_success"] = cell.compilation_success
                if comp and comp.success and run_emulation:
                    cell.duration = comp.duration
                    rec["physical_duration"] = comp.duration
                    emu = emulate_program(comp.program, num_shots=num_shots)
                    if emu.success and emu.bitstrings is not None:
                        opt_states, opt_energy = optima[hc.id]
                        if opt_states is not None:
                            dm = dynamic_metrics(emu.bitstrings, hc.hamiltonian,
                                                 opt_energy, opt_states)
                            cell.ground_state_probability = dm.ground_state_probability
                            cell.expected_objective = dm.expected_objective
                            rec["solution_probability"] = dm.ground_state_probability
                            rec["expected_objective"] = dm.expected_objective
                # robustness (cheap proxy: rank correlation under perturbation)
                if run_robustness and emb is not None:
                    from .representation_scheduler.robustness import (
                        perturb_coordinates, robustness_stats)
                    from qoolqit import Register
                    from .common.metrics import interaction_rank_correlation
                    rcs = []
                    for r in range(robustness_samples):
                        noisy = perturb_coordinates(emb.coords, 0.02, seed=seed + r)
                        reg = Register.from_coordinates(noisy.tolist())
                        rcs.append(interaction_rank_correlation(emb.target_matrix,
                                                                reg.interaction_matrix()))
                    cell.robustness = robustness_stats(rcs)["mean"]
                    rec["robustness"] = cell.robustness
            except Exception as e:
                rec["error"] = repr(e)
            cells.append(cell)
            records.append(rec)
    return CombinedResult(cells=cells, hamiltonian_ids=[c.id for c in h_candidates],
                           embedders=embedder_methods, problem=problem,
                           records=records)


__all__ = ["combined_search", "CombinedResult", "FactorialCell"]
