# RFC-0003: Transformation Registry

- Status: Draft
- Created: 2025-01-01
- Author: David Vesterlund

## Summary

This RFC proposes a canonical Transformation Registry: a structured catalog of
the transformations that convert one representation into another within the
representation search pipeline. Each entry records the transformation's
mathematical properties, applicability conditions, and verification method, so
that transformations can be composed, audited, and trusted automatically.

## Motivation

Representation search depends on a library of transformations — MWIS to QUBO,
QUBO to Ising, Ising to register/pulse, and many others. If these
transformations are ad hoc functions scattered across the codebase, there is no
way to guarantee that a given edge in a RepGraph trace is correct, applicable, or
reproducible. A registry makes each transformation a first-class, documented
unit with declared semantics, enabling automated composition and verification
rather than manual trust.

## Proposal

The registry is a collection of transformation entries. Each entry specifies:

- A stable identity (name, version) and the input/output representation types.
- Mathematical properties (e.g. exact vs. approximate, cost-preserving,
  symmetry-preserving).
- Applicability conditions (what input representations it accepts, and any
  preconditions on the logical problem).
- Verification method (analytic check, round-trip test, bounded-error test, or
  manual) and the verification status of the current implementation.

Transformations are referenced by identity from RepGraph edges (see RFC-0002),
so a trace is fully determined by the WQIR root plus the ordered transformation
identities and their parameters. New transformations are added by registering an
entry; existing entries are versioned rather than silently modified.

## Open questions

- What is the minimal set of transformations required for v0.2?
- How should approximate transformations declare and bound their error?
- Should the registry be a static catalog or a runtime-discoverable plugin
  system?
- Who owns the verification process, and how is verification status surfaced to
  users?
