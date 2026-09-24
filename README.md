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
pip install -e .                 # installs westquant_qoolqit + deps (qoolqit>=1.4.0)

# verify
pytest tests/                    # 22 tests pass
python -c "import qoolqit; print('QoolQit', qoolqit.__version__)"

# run the flagship benchmark (generates figures + manifest)
python -m westquant_qoolqit.run_benchmarks

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
├── tests/                             ← 22 pytest tests
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

**Benchmark:** MWIS path graph, n=5, weights ~ Uniform(1, 5), seed=42

| Metric | Value |
|--------|-------|
| Hamiltonian representations generated & verified | 10 |
| Exact transforms verified as `EXACT_EQUIVALENT` (max error) | < 1e-14 |
| MWIS penalty candidates verified as `GROUND_STATE_EQUIVALENT` | 5 (all U > wmax) |
| Factorial cells (H × R) | 9 |
| Local QoolQit emulations | 9 (QutipBackendV2) |
| **Variance from Hamiltonian choice (H)** | **~62%** |
| **Variance from embedding choice (R)** | **~10%** |
| Best solution probability | 11% |
| Worst solution probability | 2% |
| Best / worst ratio | >5× |

**Finding:** Both representation layers contribute meaningfully. The
Hamiltonian choice dominates for this instance, but the embedding still
contributes a 5× spread in solution probability within a single Hamiltonian.

---

## Benchmarks & figures

Run `python -m westquant_qoolqit.run_benchmarks` to regenerate:

| Output | Description |
|--------|-------------|
| `results/figures/factorial_heatmap.png` | H × R matrix of solution probabilities |
| `results/figures/variance_decomposition.png` | Bar chart: H vs R variance fractions |
| `results/figures/pareto_front.png` | Pareto scatter (interaction error vs solution probability) |
| `results/raw/benchmark_cells.jsonl` | Per-cell records (JSONL) |
| `results/processed/benchmark_summary.json` | Summary statistics |
| `results/manifests/environment.json` | Software versions + seed |

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

22 pytest tests covering algebra, equivalence, search, Pareto, robustness,
and real QoolQit integration:

```bash
pytest tests/ -v
```

| File | Tests | Coverage |
|------|-------|----------|
| `test_algebra.py` | 8 | QUBO round trips, pair counting, scaling, permutation, bit-complement, MWIS penalty |
| `test_search.py` | 9 | Equivalence invalid detection, Pareto dominance/front, robustness, candidate grid reproducibility, realizability, explorer reproducibility, exhaustive verification |
| `test_qoolqit.py` | 5 | QoolQit version, interaction embedder, DMM compilation, emulation bitstrings, MockDevice |

---

## Reproducibility

Every experiment records:
- Software versions (`results/manifests/environment.json`)
- Random seeds (passed to all generators and embedders)
- Per-cell records (`results/raw/benchmark_cells.jsonl`)
- The exact QoolQit version (`1.4.0`) stated in every notebook

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
