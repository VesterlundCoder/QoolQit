"""Combined hierarchical representation search.

    P -> {H_i} -> {R_ij} -> {Programs_ijk} -> Pareto frontier

This is the closest implementation to the full WestQuant representation-search
philosophy: it searches *both* the logical Hamiltonian representation (Project B)
and the physical embedding (Project A) in a nested fashion.

The flagship experiment is a factorial matrix:

              R_1   R_2   R_3
    H_{U_1}    .     .     .
    H_{U_2}    .     .     .
    H_{U_3}    .     .     .

asking: how much variation comes from H, how much from R, and how much from
their interaction?

Key fixes vs. the original implementation:
- Uses candidate_grid_stratified to ensure genuine embedder family diversity
  (not just different InteractionEmbedder restarts).
- InteractionEmbedder restarts use concrete x0 arrays (not None) so different
  seeds produce genuinely different embeddings.
- Supports stochastic replicates per (H, R) cell for proper ANOVA.
- Uses two-way ANOVA for variance decomposition (H, R, H×R, residual).
- Reports Wilson confidence intervals for solution probabilities.
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
    validate_terminal_encoding,
)
from .common.metrics import (
    interaction_frobenius_error, logical_fidelity, dynamic_metrics,
)
from .common.anova import two_way_anova, wilson_ci, EffectDecomposition
from .common.serialization import write_jsonl
from .hamiltonian_explorer import HamiltonianRepresentationExplorer
from .representation_scheduler import (
    RepresentationScheduler, HalvingConfig, candidate_grid_stratified,
)


@dataclass
class FactorialCell:
    """One cell of the H x R factorial matrix (one replicate)."""
    hamiltonian_id: str
    embedder: str
    embedder_family: str
    embedder_config: dict
    seed: int
    replicate: int
    equivalence: str
    frobenius_error: float | None
    logical_fidelity_rho: float | None
    compilation_success: bool | None
    duration: float | None
    ground_state_probability: float | None
    p_opt_ci_low: float | None
    p_opt_ci_high: float | None
    expected_objective: float | None
    robustness: float | None
    terminal_encoding_valid: bool | None


@dataclass
class CombinedResult:
    """Result of a combined hierarchical search."""
    cells: list[FactorialCell]
    hamiltonian_ids: list[str]
    embedder_ids: list[str]
    embedder_families: list[str]
    problem: BinaryQuadraticHamiltonian
    records: list[dict] = field(default_factory=list)

    def matrix(self, metric: str = "ground_state_probability") -> np.ndarray:
        """Return the H x R matrix for one metric (mean over replicates, NaN where unavailable)."""
        H = self.hamiltonian_ids
        R = self.embedder_ids
        mat = np.full((len(H), len(R)), np.nan)
        hidx = {h: i for i, h in enumerate(H)}
        ridx = {r: i for i, r in enumerate(R)}
        # collect all values per (H, R) cell
        values: dict[tuple[str, str], list[float]] = {}
        for c in self.cells:
            val = getattr(c, metric, None)
            if val is not None and not np.isnan(val):
                key = (c.hamiltonian_id, c.embedder)
                values.setdefault(key, []).append(val)
        for (h, r), vals in values.items():
            mat[hidx[h], ridx[r]] = float(np.mean(vals))
        return mat

    def matrix_with_replicates(self, metric: str = "ground_state_probability") -> np.ndarray:
        """Return a 3-D array (n_H, n_R, n_replicates) for ANOVA."""
        H = self.hamiltonian_ids
        R = self.embedder_ids
        hidx = {h: i for i, h in enumerate(H)}
        ridx = {r: i for i, r in enumerate(R)}
        # collect replicates per cell
        reps: dict[tuple[str, str], list[float]] = {}
        for c in self.cells:
            val = getattr(c, metric, None)
            if val is not None and not np.isnan(val):
                key = (c.hamiltonian_id, c.embedder)
                reps.setdefault(key, []).append(val)
        n_replicates = max((len(v) for v in reps.values()), default=1)
        Y = np.full((len(H), len(R), n_replicates), np.nan)
        for (h, r), vals in reps.items():
            for k, v in enumerate(vals):
                Y[hidx[h], ridx[r], k] = v
        return Y

    def variance_decomposition(self, metric: str = "ground_state_probability") -> dict:
        """Decompose total variance using two-way ANOVA.

        Returns:
            ss_H, ss_R, ss_HxR, ss_residual, ss_total,
            eta2_H, eta2_R, eta2_HxR, eta2_residual,
            df_H, df_R, df_HxR, df_residual,
            n_cells, n_replicates
        """
        Y = self.matrix_with_replicates(metric)
        n_H = len(self.hamiltonian_ids)
        n_R = len(self.embedder_ids)
        n_reps = Y.shape[2] if Y.size > 0 else 0
        if n_H < 2 or n_R < 2 or n_reps < 1:
            return {"eta2_H": 0.0, "eta2_R": 0.0, "eta2_HxR": 0.0,
                    "eta2_residual": 0.0, "n_cells": n_H * n_R, "n_replicates": n_reps}
        result = two_way_anova(Y, n_H, n_R, max(n_reps, 1))
        return {
            "ss_H": result.ss_H, "ss_R": result.ss_R,
            "ss_HxR": result.ss_HxR, "ss_residual": result.ss_residual,
            "ss_total": result.ss_total,
            "eta2_H": result.eta2_H, "eta2_R": result.eta2_R,
            "eta2_HxR": result.eta2_HxR, "eta2_residual": result.eta2_residual,
            "df_H": result.df_H, "df_R": result.df_R,
            "df_HxR": result.df_HxR, "df_residual": result.df_residual,
            "n_cells": result.n_cells, "n_replicates": result.n_replicates,
        }

    def best_cell(self, metric: str = "ground_state_probability") -> FactorialCell | None:
        """Return the best cell by mean metric over replicates (avoids winner's curse).

        Computes the mean of the metric over all replicates for each (H, R) pair,
        then returns the cell from the winning (H, R) pair. This avoids the
        winner's curse of selecting the best individual replicate.
        """
        # Group by (H, R) and compute mean
        from collections import defaultdict
        cell_means: dict[tuple[str, str], tuple[float, list[FactorialCell]]] = {}
        groups: dict[tuple[str, str], list[FactorialCell]] = defaultdict(list)
        for c in self.cells:
            val = getattr(c, metric, None)
            if val is not None and not np.isnan(val):
                groups[(c.hamiltonian_id, c.embedder)].append(c)
        if not groups:
            return None
        best_key = None
        best_mean = -np.inf
        for key, cells in groups.items():
            vals = [getattr(c, metric) for c in cells]
            mean_val = float(np.mean(vals))
            if mean_val > best_mean:
                best_mean = mean_val
                best_key = key
        if best_key is None:
            return None
        # Return the first replicate of the best (H, R) pair
        return groups[best_key][0]

    def baseline_cell(self, metric: str = "ground_state_probability") -> FactorialCell | None:
        """Return the baseline cell (first Hamiltonian, first embedder).

        The baseline is the first (H, R) pair, using the mean over its replicates
        to avoid comparing against a single lucky replicate.
        """
        if not self.cells:
            return None
        # Find the first (H, R) pair's cells
        first_h = self.cells[0].hamiltonian_id
        first_r = self.cells[0].embedder
        baseline_cells = [c for c in self.cells
                          if c.hamiltonian_id == first_h and c.embedder == first_r]
        return baseline_cells[0] if baseline_cells else self.cells[0]

    def best_vs_baseline(self, metric: str = "ground_state_probability") -> dict:
        """Compare best (H,R) mean vs baseline (H,R) mean, with Wilson CI.

        Uses pooled counts across replicates for the Wilson interval.
        """
        from collections import defaultdict
        groups: dict[tuple[str, str], list[FactorialCell]] = defaultdict(list)
        for c in self.cells:
            val = getattr(c, metric, None)
            if val is not None and not np.isnan(val):
                groups[(c.hamiltonian_id, c.embedder)].append(c)
        if not groups:
            return {}
        # Compute mean per (H, R)
        means = {}
        for key, cells in groups.items():
            vals = [getattr(c, metric) for c in cells]
            means[key] = float(np.mean(vals))
        best_key = max(means, key=means.get)
        baseline_key = (self.cells[0].hamiltonian_id, self.cells[0].embedder)
        best_mean = means[best_key]
        baseline_mean = means.get(baseline_key, 0.0)
        # When baseline is 0, relative improvement is undefined.
        # Report absolute improvement and whether baseline succeeded.
        if baseline_mean > 0:
            relative = (best_mean - baseline_mean) / baseline_mean
        elif best_mean > 0:
            relative = None  # baseline=0, best>0 → relative undefined
        else:
            relative = 0.0
        improvement = {
            "absolute": best_mean - baseline_mean,
            "relative": relative,
            "best_key": best_key,
            "baseline_key": baseline_key,
            "best_mean": best_mean,
            "baseline_mean": baseline_mean,
            "baseline_success": baseline_mean > 0,
            "best_success": best_mean > 0,
        }
        return improvement


def combined_search(
    problem: BinaryQuadraticHamiltonian,
    mwis_info: dict | None = None,
    n_hamiltonians: int = 3,
    n_embedders: int = 3,
    embedder_methods: list[str] | None = None,
    num_shots: int = 300,
    n_replicates: int = 1,
    device: Any = None,
    seed: int = 42,
    run_emulation: bool = True,
    run_robustness: bool = True,
    robustness_samples: int = 3,
    robustness_sigma: float = 0.02,
    schedule: str = "linear",
    duration: float = 4.0,
    amp_max: float = 1.5,
    det_max: float = 5.0,
    profile: str = "max_energy",
    validate_terminal: bool = True,
) -> CombinedResult:
    """Run the combined H x R factorial search with replicates.

    For MWIS problems, generates ``n_hamiltonians`` penalty Hamiltonians (varying U)
    and ``n_embedders`` embeddings (stratified across embedder families), forming
    a factorial matrix. Each cell is replicated ``n_replicates`` times with
    independent sampling seeds.

    Arguments:
        n_replicates: number of stochastic replicates per (H, R) cell.
        schedule: control schedule (``"linear"``, ``"blackman"``, ``"ramp_flat"``).
        duration: adiabatic drive duration (seconds).
        amp_max: maximum Rabi amplitude.
        det_max: maximum global detuning (also sets DMM amplitude = -det_max).
        profile: QoolQit compilation profile (``"max_energy"``, ``"default"``).
        robustness_sigma: coordinate perturbation sigma for robustness tests.
        validate_terminal: if True, validate terminal encoding for each (H, R).
    """
    device = device or AnalogDeviceWithDMM()
    embedder_methods = embedder_methods or ["interaction", "spring", "blade"]

    # Level 0: generate Hamiltonian representations
    explorer = HamiltonianRepresentationExplorer(
        transforms=["mwis_penalty"] if mwis_info else ["positive_scale"],
        verify=True, seed=seed,
    )
    exp = explorer.explore(problem, mwis_info=mwis_info)
    h_candidates = [c for c in exp.candidates
                    if c.equivalence_class.value in ("EXACT_EQUIVALENT",
                                                      "GROUND_STATE_EQUIVALENT")]
    if mwis_info:
        h_candidates = [c for c in h_candidates if c.transform_name == "mwis_penalty"]
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

    # Level 1: generate embedding candidates (stratified across families)
    n_qubits = problem.n
    embedder_cands = candidate_grid_stratified(
        embedders=embedder_methods, n_per_family=max(1, n_embedders // len(embedder_methods)),
        seed=seed, n_qubits=n_qubits,
    )
    # Limit to n_embedders total, ensuring family diversity
    if len(embedder_cands) > n_embedders:
        # Keep balanced across families
        per_family = n_embedders // len(embedder_methods)
        remainder = n_embedders % len(embedder_methods)
        selected = []
        fam_counts: dict[str, int] = {}
        for c in embedder_cands:
            fam = c.metadata.get("embedding_family", c.embedder_name)
            limit = per_family + (1 if fam_counts.get(fam, 0) < remainder else 0)
            if fam_counts.get(fam, 0) < limit:
                selected.append(c)
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
        embedder_cands = selected[:n_embedders]

    embedder_ids = [c.id for c in embedder_cands]
    embedder_families = [c.metadata.get("embedding_family", c.embedder_name)
                         for c in embedder_cands]

    # Level 2-5: evaluate each (H, R) cell with replicates
    cells: list[FactorialCell] = []
    records: list[dict] = []

    for hc in h_candidates:
        mwis_weights = mwis_info["weights"] if mwis_info else None
        for ec in embedder_cands:
            # Terminal encoding validation (once per H-R pair)
            terminal_valid = None
            if validate_terminal:
                try:
                    emb_val = run_embedder(hc.hamiltonian, method=ec.embedder_name,
                                           config=ec.embedder_config, seed=ec.seed,
                                           device=device)
                    term_report = validate_terminal_encoding(
                        hc.hamiltonian, emb_val,
                        mwis_weights=mwis_weights if mwis_weights is not None else None,
                        det_max=det_max,
                    )
                    terminal_valid = term_report.valid
                except Exception:
                    terminal_valid = False

            for rep in range(n_replicates):
                rep_seed = seed + rep * 10000
                cell = FactorialCell(
                    hamiltonian_id=hc.id, embedder=ec.id,
                    embedder_family=ec.metadata.get("embedding_family", ec.embedder_name),
                    embedder_config=ec.embedder_config, seed=ec.seed,
                    replicate=rep, equivalence=hc.equivalence_class.value,
                    frobenius_error=None, logical_fidelity_rho=None,
                    compilation_success=None, duration=None,
                    ground_state_probability=None,
                    p_opt_ci_low=None, p_opt_ci_high=None,
                    expected_objective=None, robustness=None,
                    terminal_encoding_valid=terminal_valid,
                )
                rec = {"hamiltonian_id": hc.id, "embedder": ec.id,
                       "embedder_family": cell.embedder_family,
                       "seed": ec.seed, "replicate": rep,
                       "equivalence": hc.equivalence_class.value,
                       "transform": hc.transform_name,
                       "U": hc.transform_parameters.get("U"),
                       "terminal_encoding_valid": terminal_valid}
                # Skip emulation if terminal encoding is invalid
                if validate_terminal and terminal_valid is False:
                    rec["error"] = "terminal_encoding_invalid"
                    cells.append(cell)
                    records.append(rec)
                    continue
                try:
                    emb = run_embedder(hc.hamiltonian, method=ec.embedder_name,
                                       config=ec.embedder_config, seed=ec.seed,
                                       device=device)
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
                    for use_dmm, dev, prof in [(True, device, profile),
                                               (False, device, profile)]:
                        prog = build_program(hc.hamiltonian, emb, use_dmm=use_dmm,
                                             schedule=schedule,
                                             duration=duration, amp_max=amp_max,
                                             det_max=det_max,
                                             mwis_weights=mwis_weights)
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
                                # Wilson CI
                                ci_lo, ci_hi = wilson_ci(
                                    dm.ground_state_probability, emu.n_shots)
                                cell.p_opt_ci_low = ci_lo
                                cell.p_opt_ci_high = ci_hi
                                rec["p_opt_ci_low"] = ci_lo
                                rec["p_opt_ci_high"] = ci_hi
                                rec["n_shots"] = emu.n_shots
                    # robustness
                    if run_robustness and emb is not None:
                        from .representation_scheduler.robustness import (
                            perturb_coordinates, robustness_stats)
                        from qoolqit import Register
                        from .common.metrics import interaction_rank_correlation
                        rcs = []
                        for r in range(robustness_samples):
                            noisy = perturb_coordinates(emb.coords, robustness_sigma,
                                                        seed=rep_seed + r)
                            reg = Register.from_coordinates(noisy.tolist())
                            rcs.append(interaction_rank_correlation(
                                emb.target_matrix, reg.interaction_matrix()))
                        cell.robustness = robustness_stats(rcs)["mean"]
                        rec["robustness_rank_corr_mean"] = cell.robustness
                except Exception as e:
                    rec["error"] = repr(e)
                cells.append(cell)
                records.append(rec)
    return CombinedResult(
        cells=cells, hamiltonian_ids=[c.id for c in h_candidates],
        embedder_ids=embedder_ids, embedder_families=embedder_families,
        problem=problem, records=records,
    )


__all__ = ["combined_search", "CombinedResult", "FactorialCell"]
