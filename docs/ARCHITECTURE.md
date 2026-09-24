# Architecture

## Canonical pipeline

```
Optimization Problem (P)
        |
        v
Hamiltonian Representations (H_i)
  - positive_scale, bit_complement, mwis_penalty, ...
  - equivalence verification (EXACT, GROUND_STATE, etc.)
  - physical realizability classification
        |
        v
Physical Embeddings (R_ij)
  - InteractionEmbedder, SpringLayoutEmbedder, Blade
  - candidate_grid_stratified for family diversity
  - successive halving across 5 stages
        |
        v
QoolQit Compilation + Emulation (Q_ijk)
  - compile_to(device)
  - LocalEmulator with QutipBackendV2
  - 1000 shots per emulation
        |
        v
Verification & Metrics
  - terminal encoding validity
  - compilation success
  - end-to-end success S = F × p_opt
  - two-way ANOVA (H, R, H×R, residual)
        |
        v
Pareto Selection
  - multi-objective: p_opt, robustness, duration
        |
        v
RepGraph (planned)
  - graph of representation transformations
  - physical realizability
  - compilation outcomes
  - evaluation results
        |
        v
WQT20 (planned)
  - learned search policy over P → H → R → Q
```

## Module structure

```
westquant_qoolqit/
  common/
    types.py              BinaryQuadraticHamiltonian, ProblemSpec
    exact_solver.py       exhaustive classical solver
    equivalence.py        5-class equivalence taxonomy + verification
    metrics.py            interaction fidelity, logical fidelity, dynamic metrics
    pareto.py             Pareto front selection
    qoolqit_adapter.py    QoolQit API wrapper (embed, compile, emulate)
    reproducibility.py    environment capture
    serialization.py      JSONL output
    anova.py              two-way ANOVA, Wilson CI
  hamiltonian_explorer/
    transforms.py         positive_scale, bit_complement, mwis_penalty
    realizability.py      native Rydberg classification
    scoring.py            landscape metrics (gap, degeneracy)
    result.py             ExplorerResult dataclass
    generator.py          candidate generation
    explorer.py           HamiltonianRepresentationExplorer
  representation_scheduler/
    candidates.py         candidate_grid_stratified
    embedders.py          embedder family implementations
    scoring.py            EmbedderScore dataclass
    successive_halving.py 5-stage halving
    robustness.py         coordinate perturbation
    proposer.py           candidate proposal
    result.py             SchedulerResult dataclass
    scheduler.py          RepresentationScheduler
  combined.py             combined_search + CombinedResult + ANOVA
  benchmarks.py           MWIS problem generators
  run_benchmarks.py       CLI entry point
  generate_notebooks.py   notebook generation
  generate_slides.py      slide deck generation (HTML + PDF)
  validate_submission.py  submission validator
```

## Data flow

```
experiments/flagship_v3.yaml (frozen config)
        |
        v
run_benchmarks --mode flagship
        |
        v
results/runs/flagship_v3/
  config.yaml              frozen config copy
  environment.json         software versions + seed
  raw/cells.jsonl          per-cell records (self-describing)
  processed/summary.json   per-problem statistics
  processed/report_metrics.json  aggregate metrics (single source of truth)
  figures/                 generated figures
  passports/               representation passports
```

## Key design principles

1. **Verification is authoritative**: declared equivalence is never trusted
   without exhaustive verification.
2. **No imputation**: missing H×R cells are represented as S=0 (feasibility
   failure), not filled with the grand mean.
3. **Feasibility is separate from performance**: conditional p_opt is
   reported only for feasible cells.
4. **Reproducibility by construction**: config is frozen, environment is
   captured, seeds are explicit.
5. **Telemetry off by default**: no network calls during benchmark execution.

## Future extensions (planned, not implemented)

- **WQIR**: WestQuant Intermediate Representation (see `rfcs/RFC-0001-WQIR.md`)
- **RepGraph**: Representation Graph (see `rfcs/RFC-0002-REPGRAPH.md`)
- **Transformation Registry** (see `rfcs/RFC-0003-TRANSFORMATION-REGISTRY.md`)
- **WQT20**: Learned representation search policy
- **westquant-qiskit**: Qiskit transpiler plugin
