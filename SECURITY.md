# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.1.x   | Yes       |
| 1.0.x   | No (legacy) |

## Reporting a vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email david@vesterlundcoder.com with a description of the vulnerability.
3. Include steps to reproduce if possible.
4. We will acknowledge within 48 hours and work toward a fix.

## Security scope

This repository is a research artifact. It does not:

- Collect telemetry or usage data.
- Make network requests during benchmark execution.
- Store credentials or secrets.
- Require authentication.

Dependencies are pinned in `requirements-contest.txt` and `pyproject.toml`.

## Dependency security

We recommend installing in a virtual environment and reviewing dependencies
before use. The primary dependencies are:

- `qoolqit` (Pasqal quantum compiler)
- `numpy`, `scipy`, `networkx` (numerical)
- `matplotlib` (visualization)
- `pulser`, `qutip` (via QoolQit)

No dependency is expected to make network calls during benchmark execution.
