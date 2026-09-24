# Contributing to WestQuant Representation Stack

Thank you for your interest in contributing. This repository is both a
QoolQit contest artifact and WestQuant Open Artifact #001, so we value
scientific rigor, reproducibility, and clear documentation.

## Ways to contribute

- **Bug reports**: open an issue using the bug template.
- **Feature requests**: open an issue using the feature template.
- **New transforms**: open an issue using the new-transform template.
- **Benchmark results**: open an issue using the benchmark-result template.
- **Documentation improvements**: PRs welcome.
- **Code fixes**: PRs welcome, please include tests.

## Development setup

```bash
git clone https://github.com/VesterlundCoder/QoolQit.git
cd QoolQit
pip install -e ".[test]"
pytest -v
```

## Adding a new Hamiltonian transform

1. Implement the transform in `westquant_qoolqit/hamiltonian_explorer/transforms.py`.
2. Add an equivalence class if needed in `westquant_qoolqit/common/equivalence.py`.
3. Add a test in `tests/test_algebra.py` that verifies equivalence.
4. Update the explorer to accept the new transform name.
5. Run `pytest -v` — all tests must pass.

## Adding a new embedder family

1. Implement the embedder in `westquant_qoolqit/representation_scheduler/embedders.py`.
2. Add it to the candidate grid in `westquant_qoolqit/representation_scheduler/candidates.py`.
3. Add a test in `tests/test_search.py`.
4. Run `pytest -v` — all tests must pass.

## Pull request process

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused.
3. Ensure `pytest -v` passes with no failures.
4. Ensure `python -m westquant_qoolqit.validate_submission` passes.
5. Update documentation if your change affects the API or results.
6. Open a PR using the pull-request template.
7. Do not re-run the flagship benchmark without coordinating — official
   results must remain frozen at `flagship_v3`.

## Scientific integrity

- Never claim equivalence without verification.
- Never report results from invalid terminal encodings.
- Never impute missing H×R cells with the grand mean.
- Always distinguish conditional p_opt from end-to-end success.
- Always label WQT20 and future systems as planned, not implemented.

## License

By contributing, you agree that your contributions will be licensed under
the Apache-2.0 license.
