# RFC-0002: Representation Graph (RepGraph)

- Status: Draft
- Created: 2025-01-01
- Author: David Vesterlund

## Summary

This RFC proposes the Representation Graph (RepGraph), a graph structure that
records the full lineage of representation transformations applied to a logical
problem. Each node is a representation (logical, Hamiltonian, physical, or
evaluated) and each edge is a transformation with its outcome metadata. RepGraph
is the central artifact that makes representation search reproducible,
inspectable, and shareable across the WestQuant Open ecosystem.

## Motivation

Representation search produces many candidate representations for a single
problem. Without a structured record of which transformations were tried, what
they produced, and how they performed, the search is neither reproducible nor
auditable. A researcher or downstream tool needs to answer: "Which
representation gave the best result, and what exact sequence of transformations
led to it?" RepGraph answers this by making the search history a first-class,
queryable object rather than an ephemeral log.

## Proposal

RepGraph is a directed acyclic graph where:

- Nodes are representations derived from a WQIR root (see RFC-0001), including
  Hamiltonian forms, register/pulse specifications, and evaluated outcomes.
- Edges are transformations (see RFC-0003) annotated with applicability checks,
  verification status, and the parameters used.
- Leaf nodes carry evaluation results (energy, approximation ratio, feasibility,
  hardware cost) keyed to a shared benchmark protocol.

RepGraph is serializable and designed to be shared as part of research bundles
(see RFC-0005). A reference RepGraph example is maintained at `examples/repgraph/`
and should be updated whenever the node/edge schema changes.

## Open questions

- Should RepGraph be stored per-problem or as a global, cross-problem graph?
- What is the canonical serialization format (JSON, Parquet, or a custom
  binary)?
- How should failed or infeasible transformations be represented — as edges with
  error metadata, or as pruned nodes?
- What query API is needed to support common search and analysis tasks?
