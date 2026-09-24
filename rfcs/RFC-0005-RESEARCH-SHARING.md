# RFC-0005: Research Sharing

- Status: Draft
- Created: 2025-01-01
- Author: David Vesterlund

## Summary

This RFC proposes guidelines for sharing research artifacts within the
WestQuant Open ecosystem: benchmark results, RepGraph traces, and learned
models used or produced during representation search. The goal is to make
representation-search experiments reproducible, comparable, and composable
across contributors and institutions.

## Motivation

Representation search is inherently empirical: which representation works best
depends on the problem, the hardware target, and the classical optimizer. If
results are shared only as prose or plots, the community cannot verify them,
build on them, or integrate them into new searches. Standardized sharing of
traces and models lets a result from one contributor become a prior or a
baseline for another, accelerating collective progress on the core thesis of
searching the representation.

## Proposal

The guidelines define three shareable artifact types and their formats:

- Benchmark results: structured records keyed by WQIR problem identity (see
  RFC-0001), including metrics, hardware target, optimizer config, and
  environment metadata.
- RepGraph traces: serialized graphs (see RFC-0002) capturing the full
  transformation lineage and evaluation outcomes for one or more problems.
- Learned models: any model that proposes or scores representations, stored
  with its input/output contract and the transformation registry version (see
  RFC-0003) it was trained against.

Each artifact carries provenance: author, creation date, software version, and
the registry version it depends on. Artifacts are versioned; updates produce
new versions rather than overwriting. The guidelines recommend, but do not
mandate, a canonical storage location and a content hash for integrity.

## Open questions

- What is the canonical storage and discovery mechanism for shared artifacts?
- How should artifacts cite private or unreleased problem instances?
- What licensing terms should apply to shared models and traces?
- How should compatibility be enforced when the transformation registry or WQIR
  schema evolves?
