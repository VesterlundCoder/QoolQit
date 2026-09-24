# Benchmark Card

## Purpose

This benchmark evaluates whether representation choice (Hamiltonian and
physical embedding) is a significant source of variance in quantum
optimization outcomes. It is not a benchmark of absolute quantum
performance — it is a benchmark of *representation sensitivity*.

## Problem family

**Maximum Weighted Independent Set (MWIS)** on six small graph instances:

| Problem ID | Graph type | Nodes | Seed |
|------------|------------|-------|------|
| path_n5_seed42 | Path | 5 | 42 |
| path_n6_seed1 | Path | 6 | 1 |
| cycle_n6_seed42 | Cycle | 6 | 42 |
| grid_2x3_seed42 | Grid (2×3) | 6 | 42 |
| geometric_n6_seed1 | Random geometric | 6 | 1 |
| erdos_n6_seed2 | Erdős-Rényi | 6 | 2 |

## Factors

| Factor | Levels | Description |
|--------|--------|-------------|
| Hamiltonian (H) | 5 | MWIS penalty family, varying penalty strength U |
| Embedding (R) | 9 | 3 families (interaction, spring, blade) × 3 per family |
| Replicate | 3 | Stochastic replicates per H×R cell |

Total: 6 problems × 5 H × 9 R × 3 replicates = 810 observations.
Unique H×R factor cells: 270.

## Emulation

| Parameter | Value |
|-----------|-------|
| Simulator | QoolQit LocalEmulator (QutipBackendV2) |
| Shots per emulation | 1000 |
| Device | AnalogDeviceWithDMM |
| Compilation profile | max_energy |
| Control schedule | Linear |
| Duration | 4.0 μs |
| Amplitude max | 1.5 |
| Detuning max | 5.0 |

## Feasibility definition

A representation (H, R) is **feasible** if and only if:
1. Terminal encoding is valid (ground-state preservation under the
   forward map).
2. Compilation succeeds (QoolQit can compile the program to the device).

Infeasible representations are excluded from emulation. They contribute
`S = 0` to the end-to-end success metric.

## Primary metric: end-to-end success

```
S_hrk = F_hrk × p_opt,hrk
```

where `F_hrk = 1[terminal valid AND compilable]` and `p_opt` is the
ground-state probability from emulation.

This metric gives a balanced factorial design for ANOVA: infeasible cells
contribute 0, not NaN, so no imputation is needed.

## Secondary metric: conditional p_opt

Ground-state probability for feasible cells only. This is the performance
*conditional on* the representation being physically realizable.

## Baseline

The **preselected fixed baseline** is the first Hamiltonian representation
× first embedding candidate. It is not the QoolQit default
`InteractionEmbedder` (which uses its own internal seed).

## Primary results

| Metric | Value |
|--------|-------|
| η²(H) — Hamiltonian main effect | 6.8% (median) |
| η²(R) — Embedding main effect | 19.0% (median) |
| η²(H×R) — Interaction effect | 48.2% (median) |
| Feasible factor cells | 103/270 |
| Feasible observations | 309/810 |
| Problems with feasible baseline | 1/6 |
| Problems with feasible best | 6/6 |

Source: `results/runs/flagship_v3/processed/report_metrics.json`

## Statistical method

Two-way ANOVA with H, R, H×R, and residual terms on the end-to-end success
metric. Effect sizes reported as η² (fraction of total variance explained).

## Known limitations

- **Local emulation only**: results are from a classical emulator, not a
  quantum processor.
- **Small problem sizes**: 5–6 nodes. Scaling behavior is not evaluated.
- **Single problem family**: MWIS only. Generalization to other problem
  families is not established by this benchmark.
- **Single device model**: AnalogDeviceWithDMM. Other device models may
  yield different feasibility patterns.

## What this benchmark does NOT establish

- Absolute quantum advantage over classical methods.
- Performance on problem sizes relevant to quantum advantage.
- Generalization beyond MWIS.
- Performance on real quantum hardware.

## Reproduction

```bash
python -m westquant_qoolqit.run_benchmarks --mode flagship
```

Expected runtime: ~7 minutes.
