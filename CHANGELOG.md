# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-09-24

### Added
- Apache-2.0 license file.
- `CITATION.cff` for software citation.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `PRIVACY.md`.
- `ROADMAP.md` documenting WQIR, RepGraph, WQT20 as planned.
- `AUTHORS.md`.
- `docs/BENCHMARK_CARD.md`, `docs/ARCHITECTURE.md`, `docs/ECOSYSTEM.md`.
- GitHub issue templates (bug, feature, new-transform, benchmark-result).
- GitHub pull-request template.
- `CODEOWNERS`.
- WQIR draft example (`examples/wqir/`).
- RepGraph trace example (`examples/repgraph/`).
- Representation passports (`results/runs/flagship_v3/passports/`).
- RFCs directory (`rfcs/`).
- End-to-end success metric `S = F * p_opt` for balanced ANOVA.
- Feasibility, conditional p_opt, and end-to-end success matrices in notebook.
- `experiment_id`, `feasible`, `end_to_end_success`, `failure_stage` in raw records.
- Clean metric names: `n_factor_cells`, `n_feasible_factor_cells`, `n_observations`.
- Hardened validator with 16 checks including stale-claim grep.
- Unified slide generator (HTML + PDF from `report_metrics.json`).
- "Search the representation, not just the parameters." tagline.
- WestQuant Open Artifact #001 designation.
- Canonical architecture diagram in README.

### Changed
- License from MIT to Apache-2.0.
- Combined notebook now runs full flagship (810 observations, ~7 min).
- Combined notebook uses `end_to_end_success` for ANOVA.
- Combined notebook passes `mwis_weights` to scheduler.
- Slide generators unified into `generate_slides.py`.
- README rewritten with v3 headline numbers and feasibility rescue framing.
- "default representation" → "preselected fixed baseline" throughout.
- Validator checks `flagship_v3.yaml` instead of `flagship_v1.yaml`.

### Removed
- `generate_pdf_slides.py` (merged into `generate_slides.py`).
- Stale claims: 62%, 10%, 33%, `frac_H`, `frac_R`, `robust_mean_p_opt`.
- Grand-mean imputation for missing H×R cells.

## [1.0.0] — 2024-12-01

### Added
- Initial WestQuant Representation Stack for QoolQit 1.4.0.
- Project A: Representation Scheduler with successive halving.
- Project B: Hamiltonian Representation Explorer with equivalence verification.
- Combined hierarchical search with factorial experiment.
- MWIS benchmark problem generators (path, cycle, grid, geometric, Erdős-Rényi).
- Two-way ANOVA variance decomposition.
- Smoke and flagship benchmark modes.
- Submission validator.
- Contest notebooks (combined, Project A, Project B).
- Slide decks (HTML + PDF).
