# Ecosystem

## WestQuant Open — Artifact #001

This repository is the first public reference implementation of
representation search. It is designated **WestQuant Open Artifact #001**.

The core idea:

> Search the representation, not just the parameters.

A mathematical optimization problem does not determine a unique useful
quantum representation. The Hamiltonian representation, physical embedding,
and control realization should themselves be treated as optimization
variables.

## Lineage

```
Artifact #001 (this repo)
  QoolQit Representation Stack
  - Hamiltonian Explorer
  - Representation Scheduler
  - Combined factorial experiment
  - End-to-end success metric
  - Feasibility rescue
        |
        v
WQIR (planned)
  WestQuant Intermediate Representation
  - canonical problem representation
  - decouples logic from physics
        |
        v
RepGraph (planned)
  Representation Graph
  - transformation registry
  - physical realizability
  - compilation outcomes
  - evaluation results
        |
        v
WQT5M (planned)
  Learned Representation Scheduling
  - predicts which representations to evaluate
  - trains on RepGraph traces
        |
        v
WQT20 (planned)
  Representation Search Policy
  - learned policy over P → H → R → Q
  - "search the representation, not just the parameters"
```

All items below Artifact #001 are **planned**, not implemented. See
`ROADMAP.md` for status.

## This repository's role

This repository demonstrates that:

1. Representation choice is a significant source of variance (H×R
   interaction = 48.2% median η²).
2. Representation search can rescue physical feasibility (baseline
   feasible 1/6, search finds feasible 6/6).
3. The problem → representation → compilation → evaluation pipeline is
   composable and reproducible.

It does not implement WQIR, RepGraph, or WQT20. It provides the
experimental foundation on which those systems will build.

## External dependencies

| Dependency | Role | Required |
|------------|------|----------|
| [QoolQit](https://github.com/pasqal-io/qoolqit) | Quantum compilation + emulation | Yes |
| [Pulser](https://github.com/pasqal-io/pulser) | Pulse-level simulation (via QoolQit) | Yes |
| [QuTiP](https://qutip.org/) | Quantum toolbox (via QoolQit) | Yes |
| NumPy | Numerical | Yes |
| SciPy | Statistical | Yes |
| NetworkX | Graph problems | Yes |
| Matplotlib | Visualization | Yes |

## Planned integrations

- **westquant-qiskit**: A Qiskit transpiler stage exposing WestQuant
  representation search as a third-party plugin.
- **Hugging Face**: A WestQuant Open organization for models, datasets,
  and Spaces.
- **Zenodo**: DOI assignment for each release.
- **OpenSSF**: Scorecard and Best Practices badge.

These are documented in `ROADMAP.md` and are not part of the current
release.
