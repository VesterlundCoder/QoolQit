# WestQuant Representation Stack for QoolQit

**Verified against QoolQit 1.4.0** · Python 3.10.12 · macOS / Linux · MIT License

> A mathematical optimization problem does not necessarily determine a unique
> useful quantum representation. **Representation itself can be treated as an
> optimization variable.**

This repository contains two composable, open-source contributions for the
[Pasqal QoolQit Contest](https://www.pasqal.com/resources/the-qoolqit-contest-is-live/),
built on the real installed QoolQit 1.4.0 API:

| Project | What it searches | Main API |
|---------|------------------|----------|
| **A — Representation Scheduler** | Physical embeddings / hardware realizations for a fixed Hamiltonian | `RepresentationScheduler.search(problem)` |
| **B — Hamiltonian Representation Explorer** | Mathematically valid Hamiltonian representations of one problem | `HamiltonianRepresentationExplorer.explore(problem)` |

Both are independently useful and composable into a **Representation Stack**
that searches both layers simultaneously via a factorial experiment.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Repository structure](#repository-structure)
3. [The scientific thesis](#the-scientific-thesis)
4. [Project A — Representation Scheduler](#project-a--representation-scheduler)
5. [Project B — Hamiltonian Representation Explorer](#project-b--hamiltonian-representation-explorer)
6. [Combined hierarchical search](#combined-hierarchical-search)
7. [Equivalence taxonomy](#equivalence-taxonomy)
8. [Key results](#key-results)
9. [Benchmarks & figures](#benchmarks--figures)
10. [Contest notebooks](#contest-notebooks)
11. [Slide decks](#slide-decks)
12. [Test suite](#test-suite)
13. [Reproducibility](#reproducibility)
14. [Design principles](#design-principles)
15. [Contest submission](#contest-submission)

---

## Quick start

```bash
git clone https://github.com/VesterlundCoder/QoolQit.git
cd QoolQit
pip install -e ".[test]"       # installs westquant_qoolqit + deps (qoolqit>=1.4.0)

# verify
pytest tests/                    # 59 tests pass, 0 skipped
python -c "import qoolqit; print('QoolQit', qoolqit.__version__)"

# run the smoke benchmark (fast, ~10s)
python -m westquant_qoolqit.run_benchmarks --mode smoke

# run the flagship benchmark (official results, ~9 min)
python -m westquant_qoolqit.run_benchmarks --mode flagship

# validate the submission
python -m westquant_qoolqit.validate_submission

# open the combined notebook
jupyter notebook notebooks/westquant_representation_stack_combined.ipynb
```

---

## Repository structure

```
QoolQit/
├── README.md                         ← this file
├── pyproject.toml                     ← package metadata + deps
├── .gitignore
│
├── westquant_qoolqit/                 ← the Python package
│   ├── __init__.py
│   ├── common/                        ← shared foundation (both projects)
│   │   ├── types.py                   ← BinaryQuadraticHamiltonian + QUBO/Ising converters
│   │   ├── exact_solver.py            ← exhaustive classical solver (ground truth)
│   │   ├── equivalence.py             ← 5-class equivalence verifier
│   │   ├── metrics.py                 ← interaction fidelity, logical fidelity, dynamic metrics
│   │   ├── pareto.py                  ← Pareto dominance + frontier
│   │   ├── qoolqit_adapter.py         ← bridge to real QoolQit 1.4.0 API
│   │   ├── reproducibility.py         ← environment manifest
│   │   └── serialization.py           ← JSONL I/O
│   │
│   ├── hamiltonian_explorer/          ← Project B
│   │   ├── transforms.py              ← scaling, permutation, bit-complement, MWIS penalty, control decomp
│   │   ├── realizability.py           ← native-Rydberg compatibility classifier
│   │   ├── scoring.py                 ← landscape metrics (gap, degeneracy, spectrum)
│   │   ├── result.py                  ← HamiltonianCandidate data model
│   │   ├── generator.py               ← candidate generation
│   │   └── explorer.py                 ← HamiltonianRepresentationExplorer (main API)
│   │
│   ├── representation_scheduler/      ← Project A
│   │   ├── candidates.py              ← RepresentationCandidate model
│   │   ├── embedders.py               ← candidate grid from QoolQit embedders
│   │   ├── scoring.py                 ← multi-stage scoring
│   │   ├── successive_halving.py      ← staged allocation
│   │   ├── robustness.py              ← coordinate perturbation analysis
│   │   ├── proposer.py                ← deterministic + optional AI proposers
│   │   ├── result.py                  ← SchedulerResult + Pareto
│   │   └── scheduler.py               ← RepresentationScheduler (main API)
│   │
│   ├── combined.py                    ← P → H → R → Q factorial search + variance decomposition
│   ├── benchmarks.py                  ← QUBO, MWIS, scheduling problem generators
│   ├── run_benchmarks.py              ← generates figures + manifest
│   ├── generate_notebooks.py          ← regenerates the 3 contest notebooks
│   └── generate_pdf_slides.py         ← regenerates the 3 PDF slide decks
│
├── notebooks/                         ← 3 contest notebooks
│   ├── westquant_representation_stack_combined.ipynb   ← flagship (run this first)
│   ├── project_A_representation_scheduler.ipynb
│   └── project_B_hamiltonian_explorer.ipynb
│
├── tests/                             ← 59 pytest tests
│   ├── test_algebra.py                ← QUBO round trips, scaling, permutation, bit-complement
│   ├── test_search.py                 ← equivalence, Pareto, robustness, reproducibility
│   └── test_qoolqit.py                ← QoolQit integration (embedding, compilation, emulation)
│
├── results/                           ← benchmark outputs
│   ├── raw/benchmark_cells.jsonl      ← per-cell records
│   ├── processed/benchmark_summary.json
│   ├── figures/                       ← PNG figures
│   │   ├── factorial_heatmap.png
│   │   ├── variance_decomposition.png
│   │   └── pareto_front.png
│   └── manifests/environment.json     ← software versions + seed
│
└── contest/                           ← slide decks (HTML + PDF)
    ├── project_A/slides/
    ├── project_B/slides/
    └── combined/slides/
```

---

## The scientific thesis

Quantum algorithm design typically performs **one fixed translation** from an
optimization problem to hardware. We argue this leaves performance on the
table: the same problem admits multiple valid representations, and the choice
of representation materially affects the final solution quality.

The decomposition is:

```
P  →  H_i  →  R_ij  →  Q_ijk
```

- **P**: optimization problem
- **H_i**: alternative Hamiltonian representations (Project B)
- **R_ij**: embeddings / layouts / control representations (Project A)
- **Q_ijk**: compiled programs and execution outcomes (QoolQit)

The pipeline:

```
Optimization Problem P
        │
        ▼
Hamiltonian Representation Explorer  →  {H_1, H_2, …, H_m}
        │
        ▼
Representation Scheduler             →  {R_11, R_12, …, R_mn}
        │
        ▼
QoolQit compilation + local emulation →  {Q_111, Q_112, …}
        │
        ▼
Pareto selection
```

---

## Project A — Representation Scheduler

For a fixed logical Hamiltonian, searches over QoolQit embeddings and hardware
realizations to find programs that outperform a single default embedding
across multiple objectives.

**QoolQit embedders used** (all from the real 1.4.0 API):
- `InteractionEmbedder` — optimizes coordinates so `1/r^6` approximates a target matrix
- `SpringLayoutEmbedder` — NetworkX spring layout with Rydberg-weight adaptation
- `Blade` — BLaDE embedding algorithm for interaction matrices / QUBOs

**Successive halving** avoids expensive emulation of all candidates:

| Stage | What happens | Cost |
|-------|-------------|------|
| 0 | Cheap geometric + interaction metrics | milliseconds |
| 1 | Logical fidelity (rank correlation, ground-state agreement) | milliseconds |
| 2 | Compilation to device | seconds |
| 3 | Local QutipBackendV2 emulation | seconds–minutes |
| 4 | Robustness under coordinate perturbation | seconds |

**Usage:**

```python
from westquant_qoolqit.representation_scheduler import RepresentationScheduler, HalvingConfig
from westquant_qoolqit.benchmarks import mwis_path
from qoolqit import AnalogDeviceWithDMM

problem, info = mwis_path(n=5, seed=42)
scheduler = RepresentationScheduler(
    embedders=["interaction", "spring", "blade"],
    budget=12, seed=42, num_shots=200,
)
result = scheduler.search(problem, device=AnalogDeviceWithDMM())
best = result.best("ground_state_probability")
```

---

## Project B — Hamiltonian Representation Explorer

Generates multiple mathematically valid Hamiltonian representations of one
problem, verifies equivalence by exhaustive state-by-state comparison, and
classifies each into exactly one of five equivalence classes.

**Transform families:**

| Family | Operation | Equivalence |
|--------|-----------|-------------|
| `positive_scale` | `E'(x) = a·E(x) + b`, `a > 0` | EXACT_EQUIVALENT |
| `variable_permutation` | `x' = P·x` | EXACT_EQUIVALENT |
| `bit_complement` | `x_i = 1 - y_i` on subset S | EXACT_EQUIVALENT |
| `mwis_penalty` | `E_U = -w·x + U·Σ x_i x_j` | GROUND_STATE_EQUIVALENT (if U > wmax) |
| `control_decomposition` | Same Hamiltonian, different drive | SAME_PROBLEM_DIFFERENT_DYNAMICS |

**Physical realizability classifier** flags whether a Hamiltonian's pair
interactions are native-Rydberg (all `J ≥ 0`, since Rydberg interactions are
repulsive). Bit-complement can flip signs, making a Hamiltonian non-native —
this is caught before expensive embedding.

**Usage:**

```python
from westquant_qoolqit.hamiltonian_explorer import HamiltonianRepresentationExplorer
from westquant_qoolqit.benchmarks import mwis_path

problem, info = mwis_path(n=5, seed=42)
explorer = HamiltonianRepresentationExplorer(
    transforms=["positive_scale", "bit_complement", "mwis_penalty"],
    verify=True, seed=42,
)
result = explorer.explore(problem, mwis_info=info)
for c in result.candidates:
    print(c.id, c.equivalence_class.value, c.realizability.native_pair_interactions)
```

---

## Combined hierarchical search

The flagship experiment: a factorial matrix of Hamiltonian representations
(varying penalty strength U) × embedding candidates.

```
              R_1   R_2   R_3
    H_{U_1}    ●     ●     ●
    H_{U_2}    ●     ●     ●
    H_{U_3}    ●     ●     ●
```

Each cell is compiled and emulated with QoolQit. The variance decomposition
then answers: **how much variation comes from H, and how much from R?**

**Usage:**

```python
from westquant_qoolqit.combined import combined_search
from westquant_qoolqit.benchmarks import mwis_path

problem, info = mwis_path(n=5, seed=42)
result = combined_search(problem, mwis_info=info,
                          n_hamiltonians=3, n_embedders=3, num_shots=200)
print(result.matrix("ground_state_probability"))
print(result.variance_decomposition("ground_state_probability"))
```

---

## Equivalence taxonomy

Every Hamiltonian transformation receives **exactly one** classification:

| Class | Definition |
|-------|-----------|
| `EXACT_EQUIVALENT` | `E'(T(x)) = a·E(x) + b` for every state `x` (verified exhaustively for n ≤ 16) |
| `GROUND_STATE_EQUIVALENT` | Full spectrum need not match, but the mapped optimum set is identical |
| `SAME_PROBLEM_DIFFERENT_DYNAMICS` | Same optimization problem, different quantum dynamics |
| `APPROXIMATE` | Approximate relationship, labeled as such |
| `INVALID` | Transformation cannot be justified |

No Hamiltonian is called equivalent without mathematical proof or
computational verification.

---

## Key results

**Flagship experiment (v3):** 6 MWIS problems × 5 Hamiltonian representations × 9 embedding representations × 3 replicates = 270 unique H×R cells × 3 replicates = 810 observations, 1000 shots per emulation.

The analysis uses an **end-to-end success metric** S = F × p_opt, where F = 1[terminal valid AND compilable]. This gives a balanced factorial design: infeasible cells contribute S = 0, so ANOVA is mathematically valid without imputation.

Source: `results/runs/flagship_v3/processed/report_metrics.json`

| Metric | Value |
|--------|-------|
| Problems | 6 (path, cycle, grid, geometric, Erdős-Rényi) |
| Hamiltonian representations per problem | 5 (MWIS penalty, varying U) |
| Embedding representations per problem | 9 (3 interaction + 3 spring + 3 blade) |
| Replicates per cell | 3 |
| Unique H×R cells | 270 |
| Total observations | 810 |
| Feasible H×R cells (terminal valid + compilable) | 309/810 |
| **η²(H) — Hamiltonian main effect** | **6.8%** (median) |
| **η²(R) — Embedding main effect** | **19.0%** (median) |
| **η²(H×R) — Interaction effect** | **48.2%** (median) |
| Fraction of problems improved | **100%** (6/6) |
| Problems with feasible baseline | 1/6 |
| Problems with feasible best | **6/6** |
| QoolQit version | 1.4.0 |

**Finding:** The H×R interaction is the dominant effect (median ~33%), directly
motivating joint representation search. Neither the Hamiltonian alone nor the
embedding alone determines performance — it is their *interaction* that matters.

**Feasibility rescue:** For 5 of 6 problems, the preselected fixed baseline
(first H, first R) fails terminal ground-state preservation — it cannot be
physically realized correctly on the device model. Representation search
identifies physically valid alternatives for **all 6 problems**. This is a
stronger result than a percentage improvement: the search enables solutions
that the default representation cannot reach at all.

Per-problem breakdown (source: `results/runs/flagship_v3/processed/summary.json`):

| Problem | η²(H) | η²(R) | η²(H×R) | Baseline feasible | Best feasible |
|---------|-------|-------|---------|-------------------|---------------|
| path_n5 | 7.2% | 67.8% | 20.5% | Yes | Yes |
| path_n6 | 4.9% | 10.5% | 52.0% | No | Yes |
| cycle_n6 | 6.0% | 23.3% | 63.7% | No | Yes |
| grid_2x3 | 7.2% | 23.8% | 45.2% | No | Yes |
| geometric_n6 | 8.5% | 14.4% | 44.5% | No | Yes |
| erdos_n6 | 6.5% | 14.6% | 51.1% | No | Yes |

**Note on improvement metric:** Relative improvement is undefined when the
baseline is infeasible (not emulated). We report feasibility rescue: the
search finds representations that the preselected baseline cannot achieve.

**Note on baseline:** The "preselected fixed baseline" is the first
Hamiltonian representation × first embedding candidate. It is not the QoolQit
default InteractionEmbedder (which uses its own internal seed).

---

## Benchmarks & figures

Run `python -m westquant_qoolqit.run_benchmarks --mode smoke` (fast, ~10s) or
`python -m westquant_qoolqit.run_benchmarks --mode flagship` (official, ~9 min).

Results are stored under `results/runs/<experiment_id>/`:

| Output | Description |
|--------|-------------|
| `results/runs/flagship_v3/processed/report_metrics.json` | **Single source of truth** — aggregate metrics |
| `results/runs/flagship_v3/processed/summary.json` | Per-problem statistics |
| `results/runs/flagship_v3/raw/cells.jsonl` | Per-cell records (JSONL, includes problem_id) |
| `results/runs/flagship_v3/figures/factorial_heatmap.png` | H × R matrix of solution probabilities |
| `results/runs/flagship_v3/figures/variance_decomposition.png` | Bar chart: H, R, H×R effect sizes |
| `results/runs/flagship_v3/figures/baseline_comparison.png` | Search improvement over baseline |
| `results/runs/flagship_v3/environment.json` | Software versions + seed |
| `results/runs/flagship_v3/config.yaml` | Frozen experiment config |

`results/latest.txt` points to the most recent experiment.

**Benchmark problem generators** (`westquant_qoolqit.benchmarks`):
- `synthetic_qubo(n, density, seed)` — random QUBO
- `mwis_path(n, seed)` — MWIS on a path graph
- `mwis_cycle(n, seed)` — MWIS on a cycle
- `mwis_grid(rows, cols, seed)` — MWIS on a grid
- `mwis_random_geometric(n, radius, seed)` — MWIS on a random geometric graph
- `mwis_erdos_renyi(n, p, seed)` — MWIS on an Erdős–Rényi graph
- `shift_assignment(n_workers, n_shifts, seed)` — small scheduling QUBO

---

## Contest notebooks

All notebooks state the QoolQit version explicitly and run end-to-end.

| Notebook | Description |
|----------|-------------|
| `notebooks/westquant_representation_stack_combined.ipynb` | **Flagship** — both projects + factorial experiment + variance decomposition |
| `notebooks/project_A_representation_scheduler.ipynb` | Project A only — embedding search with successive halving |
| `notebooks/project_B_hamiltonian_explorer.ipynb` | Project B only — Hamiltonian transforms + equivalence verification |

Regenerate with: `python -m westquant_qoolqit.generate_notebooks`

---

## Slide decks

Two-slide PDF decks for each variant (Slide 1: overview/results; Slide 2:
QoolQit experience):

| Deck | Path |
|------|------|
| Project A | `contest/project_A/slides/slides.pdf` |
| Project B | `contest/project_B/slides/slides.pdf` |
| Combined | `contest/combined/slides/slides.pdf` |

Regenerate with: `python -m westquant_qoolqit.generate_pdf_slides`

---

## Test suite

59 pytest tests (59 passed, 0 skipped) covering algebra, equivalence, search,
Pareto, robustness, real QoolQit integration, and submission-grade regression
tests:

```bash
pytest tests/ -v
```

| File | Tests | Coverage |
|------|-------|----------|
| `test_algebra.py` | 8 | QUBO round trips, pair counting, scaling, permutation, bit-complement, MWIS penalty |
| `test_search.py` | 9 | Equivalence invalid detection, Pareto dominance/front, robustness, candidate grid reproducibility, realizability, explorer reproducibility, exhaustive verification |
| `test_qoolqit.py` | 5 | QoolQit version, interaction embedder, DMM compilation, emulation bitstrings, MockDevice |
| `test_regression.py` | 35 | Terminal drive, MWIS DMM, forward-map verification, embedder diversity, terminal encoding, two-way ANOVA, Wilson CI, degenerate equivalence, adversarial equivalence, scheduler embedders, reproducibility, declared vs verified, QUBO edge cases |

---

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full clean-room reproduction
guide. Every experiment records:
- Software versions (`results/runs/<id>/environment.json`)
- Random seeds (passed to all generators and embedders)
- Per-cell records (`results/runs/<id>/raw/cells.jsonl`)
- The exact QoolQit version (`1.4.0`) stated in every notebook
- Frozen experiment configs (`experiments/smoke_v1.yaml`, `experiments/flagship_v3.yaml`)
- Pinned dependencies (`requirements-contest.txt`)

```python
from westquant_qoolqit.common import capture_environment
env = capture_environment(seed=42)
print(env.qoolqit_version, env.numpy_version, env.python_version)
```

---

## Design principles

1. **No false equivalence claims** — every classification is backed by exhaustive verification or a documented sufficient condition.
2. **No meaningless proxy metrics** — interaction fidelity, logical fidelity, and solution probability are physically meaningful.
3. **AI is optional** — deterministic, random, and heuristic search work without any proprietary infrastructure. The `WestQuantProposer` falls back to `AdaptiveHeuristicProposer` when no model is provided.
4. **Reproducibility** — environment manifest, seeds, and JSONL records for every experiment.
5. **Real QoolQit API** — built against the installed qoolqit 1.4.0, not an assumed interface. The `qoolqit_adapter.py` module is the single place to update if the API changes.

---

## Contest submission

Each variant includes:
- One runnable QoolQit notebook (stating QoolQit version 1.4.0)
- One two-slide PDF deck
- Submission email: `qoolqit.contest@pasqal.com`
- Deadline: October 18, 2026, 23:59 CEST

The combined notebook is the recommended single submission; the two project
notebooks are provided for separate submission if preferred.

---

## License

MIT
