# Reproducibility

## Quick reproduction (smoke mode)

```bash
git clone https://github.com/VesterlundCoder/QoolQit.git
cd QoolQit
git checkout qoolqit-contest-v1.1

python -m venv .venv
source .venv/bin/activate

pip install -r requirements-contest.txt
pip install -e .

pytest -v
python -m westquant_qoolqit.run_benchmarks --mode smoke
python -m westquant_qoolqit.validate_submission
```

Expected runtime: ~10 seconds for the smoke benchmark.

## Full reproduction (flagship mode)

```bash
python -m westquant_qoolqit.run_benchmarks --mode flagship
```

This resolves to `experiments/flagship_v3.yaml` and runs:
6 problems × 5 Hamiltonians × 9 embeddings × 3 replicates with
1000 shots per emulation. Expected runtime: ~7 minutes.

## Custom experiment

```bash
python -m westquant_qoolqit.run_benchmarks --config experiments/flagship_v3.yaml
```

## Result locations

Results are stored under `results/runs/<experiment_id>/`:

```
results/runs/flagship_v3/
    config.yaml              — frozen experiment config
    environment.json         — software versions + seed
    raw/cells.jsonl          — per-cell records (includes problem_id, feasible, end_to_end_success)
    processed/summary.json   — per-problem statistics
    processed/report_metrics.json — aggregate metrics (single source of truth)
    figures/                 — generated figures
    passports/               — representation passports (best per problem)
```

`results/latest.txt` points to the most recent experiment.

## Environment

Official results were generated with:

| Package | Version |
|---------|---------|
| Python | 3.10.12 |
| QoolQit | 1.4.0 |
| Pulser | 1.9.1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| NetworkX | 3.4.2 |
| QuTiP | 5.2.3 |
| Matplotlib | 3.10.9 |
| Platform | macOS-26.4-arm64 |

## Validation

```bash
python -m westquant_qoolqit.validate_submission
```

Checks that all required files exist, QoolQit version matches, no NaNs in
metrics, raw records are self-describing, tests pass, and the combined
notebook executes top-to-bottom.

## Clean-room reproduction checklist

1. Create a fresh virtual environment
2. Clone the exact submission tag: `git checkout qoolqit-contest-v1.1`
3. Install only `requirements-contest.txt` + `pip install -e .`
4. Run `pytest -v` — all tests must pass (59 passed, 0 skipped)
5. Run `python -m westquant_qoolqit.run_benchmarks --mode smoke`
6. Run `python -m westquant_qoolqit.validate_submission`
7. Execute `notebooks/westquant_representation_stack_combined.ipynb` top to bottom

Everything must pass without manual intervention.
