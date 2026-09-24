# RFC-0004: Objective API

- Status: Draft
- Created: 2025-01-01
- Author: David Vesterlund

## Summary

This RFC proposes a unified Objective API that lets different problem types —
MWIS, QUBO, scheduling, and others — be expressed through a common interface so
that representation search can operate uniformly across them. The API abstracts
over problem-specific construction logic while preserving the information WQIR
and the transformation pipeline need to produce and evaluate representations.

## Motivation

Each problem type currently brings its own construction code, its own
Hamiltonian assembly, and its own evaluation routine. This makes it difficult
to run the same representation-search workflow across problem types, and it
duplicates effort across the ecosystem. A shared objective interface lets a
single search harness target any supported problem type, and lets new problems
be added by implementing the interface rather than by rewriting the search
machinery. This directly supports the thesis of searching the representation
independent of the problem's surface syntax.

## Proposal

The Objective API defines a common interface implemented by each problem type:

- A method to emit a WQIR description of the logical problem (see RFC-0001).
- A method to evaluate a candidate solution against the original problem,
  returning standard metrics (objective value, feasibility, approximation
  ratio where defined).
- A method to declare which transformation families the problem supports, so
  the search harness can select valid edges from the Transformation Registry
  (see RFC-0003).

The API is intentionally narrow: it does not expose Hamiltonian or register
construction, which remain the responsibility of registered transformations.
This keeps problem-type code focused on problem semantics while the search
harness remains problem-agnostic.

## Open questions

- Which metrics are mandatory for all problem types, and which are optional?
- How should the API expose problem-specific constraints that affect
  transformation applicability?
- Should the API support incremental/warm-start evaluation across
  representations?
- What is the boundary between the Objective API and the WQIR schema — are they
  redundant, or complementary?
