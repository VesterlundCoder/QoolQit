"""WestQuant Representation Scheduler (Project A).

For a fixed logical Hamiltonian, searches over physical embeddings and hardware
realizations to find QoolQit programs that are better than a single default
embedding, across multiple objectives (interaction fidelity, logical fidelity,
compilation, duration, solution quality, robustness).

Uses successive halving to avoid expensive emulation of all candidates.
"""

from __future__ import annotations

import numpy as np

from qoolqit import AnalogDevice, AnalogDeviceWithDMM, MockDevice

from ..common.types import BinaryQuadraticHamiltonian
from ..common.exact_solver import solve_exact
from ..common.qoolqit_adapter import (
    run_embedder, build_program, compile_program, emulate_program,
)
from ..common.metrics import logical_fidelity, dynamic_metrics
from .candidates import RepresentationCandidate
from .embedders import candidate_grid
from .scoring import CandidateScore, score_cheap
from .successive_halving import HalvingConfig, halve_indices
from .robustness import perturb_coordinates, robustness_stats, robustness_drop
from .proposer import RepresentationProposer, GridProposer
from .result import SchedulerResult


class RepresentationScheduler:
    """Automated representation search over QoolQit embeddings.

    Example:
        scheduler = RepresentationScheduler(
            embedders=["interaction", "spring", "blade"], budget=200, seed=42)
        result = scheduler.search(problem, device=device)
        result.best(); result.pareto_front()
    """

    def __init__(
        self,
        embedders: list[str] | None = None,
        budget: int = 200,
        seed: int = 42,
        objectives: list[str] | None = None,
        strategy: str = "successive_halving",
        proposer: RepresentationProposer | None = None,
        num_shots: int = 500,
        device=None,
        halving: HalvingConfig | None = None,
    ) -> None:
        self.embedders = embedders or ["interaction", "spring", "blade"]
        self.budget = budget
        self.seed = seed
        self.objectives = objectives or [
            "frobenius_error", "compilation_success", "duration",
            "ground_state_probability", "robust_interaction_rank_mean",
        ]
        self.strategy = strategy
        self.proposer = proposer or GridProposer(seed=seed, embedders=self.embedders)
        self.num_shots = num_shots
        self.device = device
        self.halving = halving or HalvingConfig()

    def search(
        self,
        problem: BinaryQuadraticHamiltonian,
        device=None,
        num_shots: int | None = None,
        run_emulation: bool = True,
        run_robustness: bool = True,
        robustness_sigma: float = 0.02,
        robustness_samples: int = 5,
    ) -> SchedulerResult:
        device = device or self.device or AnalogDevice()
        num_shots = num_shots or self.num_shots
        exact = solve_exact(problem) if problem.n <= 22 else None
        opt_states = (set(int(i) for i in exact.optimum_indices) if exact else None)
        opt_energy = exact.optimum_energy if exact else None

        # ---- Stage 0: generate candidates + cheap scoring ----
        candidates = self.proposer.propose(problem, history=[], n_candidates=self.budget,
                                            budget=self.budget)
        scores: list[CandidateScore] = []
        for cand in candidates:
            try:
                emb = run_embedder(problem, method=cand.embedder_name,
                                   config=cand.embedder_config, seed=cand.seed,
                                   device=device)
                cheap = score_cheap(problem, emb)
                s = CandidateScore(candidate_id=cand.id, embedding=emb, **cheap)
                s.metadata["embedder"] = cand.embedder_name
                scores.append(s)
                cand.coordinates = emb.coords
            except Exception as e:
                scores.append(CandidateScore(candidate_id=cand.id,
                                              metadata={"embedder": cand.embedder_name,
                                                        "error": repr(e)}))

        # ---- Stage 1: logical fidelity on top fraction ----
        frob = [s.frobenius_error if s.frobenius_error is not None else 1e9 for s in scores]
        keep1 = halve_indices(frob, self.halving.stage0_keep, minimize=True)
        for i in keep1:
            s = scores[i]
            if s.embedding is None:
                continue
            # realized Hamiltonian from the realized interactions + linear terms
            realized = problem.copy()
            realized.quadratic = s.embedding.realized_interactions
            s.logical_fidelity = logical_fidelity(problem, realized)

        # ---- Stage 2: compile top candidates ----
        # DMM requires a DMM-capable device; pick the right device automatically.
        dmm_device = device
        if not getattr(device, "name", "").endswith("WithDMM"):
            # AnalogDevice lacks DMM channels; use the DMM-capable variant.
            dmm_device = AnalogDeviceWithDMM()
        lf = [s.logical_fidelity.energy_rank_correlation if s.logical_fidelity else -1e9
              for s in scores]
        keep2 = halve_indices(lf, self.halving.stage2_keep, minimize=False)
        for i in keep2:
            s = scores[i]
            if s.embedding is None:
                continue
            comp = None
            # try DMM + max_energy first, then fall back to no-DMM + max_energy
            for use_dmm, dev, prof in [(True, dmm_device, "max_energy"),
                                       (False, device, "max_energy"),
                                       (True, dmm_device, "default")]:
                try:
                    prog = build_program(problem, s.embedding, use_dmm=use_dmm)
                    comp = compile_program(prog, device=dev, profile=prof)
                    if comp.success:
                        break
                except Exception as e:
                    s.compilation_failure_reason = repr(e)
            if comp is not None:
                s.compilation = comp
                s.compilation_success = comp.success
                s.compilation_failure_reason = comp.failure_reason
                s.duration = comp.duration
                s.min_spacing_margin = comp.min_spacing_margin
                s.max_radial_extent = comp.max_radial_extent

        # ---- Stage 3: emulate compiled candidates ----
        if run_emulation:
            comp_ok = [i for i in keep2 if scores[i].compilation_success]
            keep3 = comp_ok[:self.halving.stage3_keep]
            for i in keep3:
                s = scores[i]
                if s.compilation is None or s.compilation.program is None:
                    continue
                emu = emulate_program(s.compilation.program, num_shots=num_shots)
                s.emulation = emu
                if emu.success and emu.bitstrings is not None and opt_states is not None:
                    dm = dynamic_metrics(emu.bitstrings, problem, opt_energy, opt_states)
                    s.ground_state_probability = dm.ground_state_probability
                    s.expected_objective = dm.expected_objective
                    s.best_sampled_objective = dm.best_sampled_objective
                    s.approximation_ratio = dm.approximation_ratio

        # ---- Stage 4: robustness on top candidates ----
        if run_robustness and run_emulation:
            emu_ok = [i for i in keep3 if scores[i].ground_state_probability is not None]
            keep4 = emu_ok[:self.halving.stage4_keep]
            for i in keep4:
                s = scores[i]
                if s.embedding is None:
                    continue
                p_opts = []
                for r in range(robustness_samples):
                    noisy = perturb_coordinates(s.embedding.coords, robustness_sigma,
                                                seed=self.seed + r)
                    # recompute realized interactions under perturbation
                    from qoolqit import Register
                    reg = Register.from_coordinates(noisy.tolist())
                    realized = reg.interaction_matrix()
                    # cheap proxy: rank correlation as a stand-in score
                    from ..common.metrics import interaction_rank_correlation
                    rc = interaction_rank_correlation(s.embedding.target_matrix, realized)
                    p_opts.append(rc)
                stats = robustness_stats(p_opts)
                # These are rank correlations, NOT solution probabilities
                s.robust_interaction_rank_mean = stats["mean"]
                s.robust_interaction_rank_std = stats["std"]
                s.robust_interaction_rank_q05 = stats["q05"]
                nominal = (s.rank_correlation if s.rank_correlation is not None else 0.0)
                s.robustness_drop = robustness_drop(nominal, stats["mean"])

        maximize = [n in {"ground_state_probability", "rank_correlation",
                          "edge_preservation", "robust_interaction_rank_mean",
                          "robust_mean_p_opt", "min_spacing_margin",
                          "approximation_ratio"} for n in self.objectives]
        return SchedulerResult(
            scores=scores, problem=problem, objective_names=self.objectives,
            maximize=maximize, metadata={"seed": self.seed, "budget": self.budget,
                                          "device": device.name, "n_candidates": len(scores)})


__all__ = ["RepresentationScheduler"]
