# Pull Request Template

## Summary

Brief description of what this PR changes and why.

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] New transform or embedder

## Checklist

- [ ] `pytest -v` passes with no failures
- [ ] `python -m westquant_qoolqit.validate_submission` passes
- [ ] I have not re-run the flagship benchmark (official results are frozen at `flagship_v3`)
- [ ] I have not introduced stale claims (62%, `frac_H`, `flagship_v1`, etc.)
- [ ] I have used "preselected fixed baseline" (not "default representation")
- [ ] I have distinguished conditional p_opt from end-to-end success
- [ ] I have labeled any future systems (WQT20, WQIR, RepGraph) as planned
- [ ] I have updated documentation if the API or results changed
- [ ] I have added tests for new functionality

## Scientific integrity

- [ ] I have not claimed equivalence without verification
- [ ] I have not reported results from invalid terminal encodings
- [ ] I have not imputed missing H×R cells with the grand mean
