# RFC-0001: WestQuant Intermediate Representation (WQIR)

- Status: Draft
- Created: 2025-01-01
- Author: David Vesterlund

## Summary

This RFC proposes the WestQuant Intermediate Representation (WQIR), a canonical
intermediate representation for optimization problems that decouples the logical
problem definition from its Hamiltonian and physical (register/pulse)
representations. WQIR serves as the stable substrate upon which representation
search is performed, enabling the same logical problem to be mapped to many
candidate physical representations without losing the original problem
semantics.

## Motivation

In quantum optimization, the choice of representation — the Hamiltonian, the
register geometry, the embedding, the pulse schedule — often has a larger impact
on solution quality than the choice of classical optimizer or its
hyperparameters. Today these concerns are tangled together inside
problem-specific code, making it hard to compare representations fairly or to
search over them systematically. The core thesis of WestQuant Open Artifact #001
is "search the representation, not just the parameters." WQIR provides the
common anchor point that makes such a search well-defined: a logical problem is
expressed once, and many representations can be derived, evaluated, and compared
against it.

## Proposal

WQIR is a structured, serializable description of a logical optimization
problem. It specifies:

- The problem type (e.g. MWIS, QUBO, scheduling) and its logical variables.
- The objective terms and constraints in a type-agnostic form.
- Metadata required for evaluation (graph adjacency, weights, deadlines).
- A stable problem identity used to key RepGraph traces and benchmarks.

WQIR does **not** specify a Hamiltonian, a register, or a pulse schedule. Those
are produced by transformations registered separately (see RFC-0003) and
recorded as edges in the Representation Graph (see RFC-0002). A reference WQIR
example is maintained at `examples/wqir/` and should be kept in sync with any
changes to the schema defined here.

## Open questions

- Should WQIR be a typed algebraic data structure or a schema-validated
  document format (e.g. JSON Schema)?
- How should problem identity be computed — by structural hash, by author
  declaration, or both?
- What is the minimal set of problem types WQIR must support at v0.2?
- How should WQIR represent constraints that are not naturally penalty-based?
