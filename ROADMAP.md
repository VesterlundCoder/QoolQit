# Roadmap

This roadmap describes the planned evolution of the WestQuant representation
search ecosystem. Items here are **planned**, not implemented, unless
explicitly marked otherwise.

## Current state (Artifact #001)

- [x] Hamiltonian Representation Explorer (Project B)
- [x] Representation Scheduler (Project A)
- [x] Combined hierarchical search with factorial experiment
- [x] End-to-end success metric and balanced ANOVA
- [x] Feasibility rescue framing
- [x] `flagship_v3` benchmark (6 problems, 270 H×R cells, 810 observations)
- [x] Reproducibility infrastructure (config, environment, validator)
- [x] Contest notebooks and slide decks

## Planned

### WQIR — WestQuant Intermediate Representation

A canonical intermediate representation for optimization problems that
decouples the logical problem from its Hamiltonian and physical
representations. A draft schema is provided in `examples/wqir/`.

Status: **Draft**. See `rfcs/RFC-0001-WQIR.md`.

### RepGraph — Representation Graph

A graph structure that records representation transformations, physical
realizability, compilation outcomes, and evaluation results. Enables
search over the representation space itself.

Status: **Draft**. See `rfcs/RFC-0002-REPGRAPH.md` and `examples/repgraph/`.

### Transformation Registry

A canonical registry of transformation identities and verification
metadata. Each transformation records its mathematical properties,
applicability conditions, and verification method.

Status: **Draft**. See `rfcs/RFC-0003-TRANSFORMATION-REGISTRY.md`.

### WQT5M — Learned Representation Scheduling

A learned model that schedules representation search over an
already-defined representation space. Trains on RepGraph traces to
predict which representations are worth evaluating.

Status: **Planned**. Not started.

### WQT20 — Representation Search Policy

A learned search policy over the full representation hierarchy
(P → H → R → Q). This is the long-term goal: search the representation,
not just the parameters, using a learned policy.

Status: **Planned**. Not started.

### westquant-qiskit

A Qiskit transpiler stage that exposes WestQuant representation search
as a third-party plugin. Would allow Qiskit users to benefit from
representation search without changing their workflow.

Status: **Planned**. Not started.

### Hugging Face Organization

A WestQuant Hugging Face organization for hosting models, datasets,
and Spaces related to representation search.

**Namespace: `WestQuantStudio`** (active)

Purpose:
- WestQuant models
- Representation-search datasets
- RepGraph datasets
- Evaluation sets
- Interactive demos and Spaces

Status: **Active**. Account confirmed and connected.

### GitHub Organization

**Namespace: `WestQuantOpen`** (planned)

The permanent home for WestQuant Open source artifacts. Future repos:
- `WestQuantOpen/representation-stack` (this artifact, migrated)
- `WestQuantOpen/wqir`
- `WestQuantOpen/repgraph`
- `WestQuantOpen/transformation-registry`
- `WestQuantOpen/westquant-qiskit`
- `WestQuantOpen/wqt-training`
- `WestQuantOpen/westquant-evals`

Status: **Planned**. Not yet created.

### Zenodo DOI

A Zenodo DOI for permanent archival of each release.

Status: **Planned**. Requires GitHub-Zenodo integration setup.

### OpenSSF Scorecard

OpenSSF Scorecard and Best Practices badge for open-source security posture.

Status: **Planned**. Requires repository settings configuration.

## Not planned

- A QPU execution backend (this is a simulation-only artifact).
- A proprietary WestQuant AI service (the stack is fully open-source).
- Telemetry or usage tracking (off by default, see `PRIVACY.md`).
