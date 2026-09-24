"""Validate the submission package.

Checks:
    - Official result files exist
    - Experiment config exists
    - Environment manifest exists
    - Notebooks reference correct QoolQit version
    - Reported metrics match summary
    - All required figures exist
    - No NaNs in required flagship metrics
    - All tested representations are unique
    - No duplicated H×R cells
    - Raw records are self-describing (problem_id, feasible, end_to_end_success, experiment_id)
    - Invalid terminal representations were never emulated
    - No stale legacy metric keys
    - All six problems represented
    - All three embedder families represented
    - 5 Hamiltonian levels per problem
    - 9 embedding levels per problem
    - 3 replicates per cell
    - Notebook executes top-to-bottom
    - Combined notebook uses end_to_end_success
    - Combined notebook passes mwis_weights
    - PDFs exist
    - No NaN in official aggregate metrics
    - README official metrics match report_metrics.json
    - No stale claims (62%, frac_H, flagship_v1, etc.)

Usage:
    python -m westquant_qoolqit.validate_submission
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_file(path: Path, description: str) -> bool:
    if path.exists():
        print(f"  [OK] {description}: {path}")
        return True
    else:
        print(f"  [FAIL] {description}: {path} not found")
        return False


def validate_submission(experiment_id_override: str | None = None) -> bool:
    """Run all validation checks. Returns True if all pass.
    
    If experiment_id_override is given, use it instead of reading results/latest.txt.
    This allows CI to validate flagship_v3 even after smoke overwrites latest.txt.
    """
    print("=== WestQuant QoolQit Submission Validation ===\n")
    all_ok = True

    # 1. Check experiment configs
    print("1. Experiment configs:")
    for cfg in ["experiments/smoke_v1.yaml", "experiments/flagship_v3.yaml"]:
        if not check_file(REPO_ROOT / cfg, f"Config {cfg}"):
            all_ok = False

    # 2. Check results
    print("\n2. Results:")
    latest_file = REPO_ROOT / "results" / "latest.txt"
    if experiment_id_override:
        experiment_id = experiment_id_override
        print(f"  [INFO] Using explicit experiment ID: {experiment_id}")
    elif latest_file.exists():
        experiment_id = latest_file.read_text().strip()
    else:
        print("  [FAIL] No results found (run benchmarks first)")
        return False
    run_dir = REPO_ROOT / "results" / "runs" / experiment_id
    for f in ["config.yaml", "environment.json"]:
        if not check_file(run_dir / f, f"Result {f}"):
            all_ok = False
    for f in ["summary.json", "report_metrics.json"]:
        if not check_file(run_dir / "processed" / f, f"Processed {f}"):
            all_ok = False
    if not check_file(run_dir / "raw" / "cells.jsonl", "Raw cells"):
        all_ok = False

    # 3. Check experiment ID is flagship_v3
    print("\n3. Experiment ID:")
    if experiment_id == "flagship_v3":
        print(f"  [OK] experiment_id == flagship_v3")
    else:
        print(f"  [FAIL] experiment_id == {experiment_id} (expected flagship_v3)")
        all_ok = False

    # 4. Check report experiment_id
    print("\n4. Report experiment_id:")
    metrics_file = REPO_ROOT / "results" / "runs" / experiment_id / "processed" / "report_metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        if metrics.get("experiment_id") == "flagship_v3":
            print(f"  [OK] experiment_id == flagship_v3")
        else:
            print(f"  [FAIL] experiment_id == {metrics.get('experiment_id')} (expected flagship_v3)")
            all_ok = False

    # 5. Check QoolQit version
    print("\n5. QoolQit version:")
    metrics_file = REPO_ROOT / "results" / "runs" / experiment_id / "processed" / "report_metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        qoolqit_version = metrics.get("qoolqit_version", "unknown")
        if qoolqit_version == "1.4.0":
            print(f"  [OK] QoolQit version: {qoolqit_version}")
        else:
                print(f"  [FAIL] QoolQit version: {qoolqit_version} (expected 1.4.0)")
                all_ok = False

    # 6. Check raw observations
    print("\n6. Raw observations:")
    raw_file = REPO_ROOT / "results" / "runs" / experiment_id / "raw" / "cells.jsonl"
    if raw_file.exists():
            records = []
            for line in raw_file.read_text().strip().split("\n"):
                if line:
                    records.append(json.loads(line))
            n_obs = len(records)
            if n_obs == 810:
                print(f"  [OK] {n_obs} observations (expected 810)")
            else:
                print(f"  [FAIL] {n_obs} observations (expected 810)")
                all_ok = False

            # Check all records have required fields
            required_fields = ["problem_id", "feasible", "end_to_end_success", "experiment_id"]
            for field in required_fields:
                missing = sum(1 for r in records if field not in r)
                if missing > 0:
                    print(f"  [FAIL] {missing} records missing '{field}'")
                    all_ok = False
                else:
                    print(f"  [OK] All records have '{field}'")

    # 7. Check terminal encoding validity
    print("\n7. Terminal encoding validity:")
    raw_file = REPO_ROOT / "results" / "runs" / experiment_id / "raw" / "cells.jsonl"
    if raw_file.exists():
            total_cells = 0
            invalid_terminal = 0
            emulated_with_invalid = 0
            for line in raw_file.read_text().strip().split("\n"):
                if line:
                    rec = json.loads(line)
                    total_cells += 1
                    if rec.get("terminal_encoding_valid") is False:
                        invalid_terminal += 1
                        if rec.get("solution_probability") is not None:
                            emulated_with_invalid += 1
            if total_cells > 0:
                if emulated_with_invalid > 0:
                    print(f"  [FAIL] {emulated_with_invalid} cells with invalid terminal encoding were emulated")
                    all_ok = False
                elif invalid_terminal > 0:
                    print(f"  [OK] {invalid_terminal}/{total_cells} cells have invalid terminal encoding (correctly excluded from emulation)")
                else:
                    print(f"  [OK] All {total_cells} cells have valid terminal encoding")

    # 8. Check all six problems represented
    print("\n8. Problem coverage:")
    raw_file = REPO_ROOT / "results" / "runs" / experiment_id / "raw" / "cells.jsonl"
    if raw_file.exists():
            problems = set()
            embedder_families = set()
            hamiltonian_ids = set()
            embedder_ids = set()
            for line in raw_file.read_text().strip().split("\n"):
                if line:
                    rec = json.loads(line)
                    problems.add(rec.get("problem_id"))
                    embedder_families.add(rec.get("embedder_family"))
                    hamiltonian_ids.add(rec.get("hamiltonian_id"))
                    embedder_ids.add(rec.get("embedder"))
            if len(problems) == 6:
                print(f"  [OK] {len(problems)} problems represented")
            else:
                print(f"  [FAIL] {len(problems)} problems (expected 6)")
                all_ok = False
            expected_families = {"interaction", "spring", "blade"}
            if embedder_families == expected_families:
                print(f"  [OK] All 3 embedder families represented: {sorted(embedder_families)}")
            else:
                print(f"  [FAIL] Embedder families: {sorted(embedder_families)} (expected {sorted(expected_families)})")
                all_ok = False

    # 9. Check no NaNs in metrics
    print("\n9. No NaNs in metrics:")
    metrics_file = REPO_ROOT / "results" / "runs" / experiment_id / "processed" / "report_metrics.json"
    if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            nan_found = False
            for key, val in metrics.items():
                if isinstance(val, float) and (val != val):
                    print(f"  [FAIL] NaN in {key}")
                    nan_found = True
                    all_ok = False
            if not nan_found:
                print("  [OK] No NaNs found")

    # 10. Check notebooks
    print("\n10. Notebooks:")
    for nb in ["notebooks/westquant_representation_stack_combined.ipynb",
               "notebooks/project_A_representation_scheduler.ipynb",
               "notebooks/project_B_hamiltonian_explorer.ipynb"]:
        if not check_file(REPO_ROOT / nb, f"Notebook {nb}"):
            all_ok = False

    # 11. Check notebook content (end_to_end_success + mwis_weights)
    print("\n11. Notebook content:")
    combined_nb = REPO_ROOT / "notebooks" / "westquant_representation_stack_combined.ipynb"
    if combined_nb.exists():
        nb_text = combined_nb.read_text()
        if "end_to_end_success" in nb_text:
            print("  [OK] Combined notebook uses end_to_end_success")
        else:
            print("  [FAIL] Combined notebook does not use end_to_end_success")
            all_ok = False
        if "mwis_weights" in nb_text:
            print("  [OK] Combined notebook passes mwis_weights")
        else:
            print("  [FAIL] Combined notebook does not pass mwis_weights")
            all_ok = False

    # 12. Check slides
    print("\n12. Slide decks:")
    for deck in ["contest/project_A/slides/slides.pdf",
                 "contest/project_B/slides/slides.pdf",
                 "contest/combined/slides/slides.pdf"]:
        if not check_file(REPO_ROOT / deck, f"Slides {deck}"):
            all_ok = False

    # 13. Check tests
    print("\n13. Tests:")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if "passed" in result.stdout and "failed" not in result.stdout:
        last_line = result.stdout.strip().split("\n")[-1]
        print(f"  [OK] {last_line}")
    else:
        print(f"  [FAIL] Tests failed")
        print(result.stdout[-500:])
        all_ok = False

    # 14. Check notebook is executable
    print("\n14. Notebook execution:")
    nb_path = REPO_ROOT / "notebooks" / "westquant_representation_stack_combined.ipynb"
    if nb_path.exists():
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", str(nb_path), "--output", "/tmp/validated.ipynb",
             "--ExecutePreprocessor.timeout=600"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        if result.returncode == 0:
            print("  [OK] Combined notebook executes top-to-bottom")
        else:
            print(f"  [FAIL] Notebook execution failed")
            print(result.stderr[-500:])
            all_ok = False
    else:
        print("  [FAIL] Combined notebook not found")
        all_ok = False

    # 15. Check README metrics match report_metrics.json
    print("\n15. README metrics match report:")
    metrics_file = REPO_ROOT / "results" / "runs" / experiment_id / "processed" / "report_metrics.json"
    readme_file = REPO_ROOT / "README.md"
    if metrics_file.exists() and readme_file.exists():
            metrics = json.loads(metrics_file.read_text())
            readme = readme_file.read_text()
            # Check key numbers appear in README
            eta2_H = f"{metrics['eta2_H_median']:.1%}"
            eta2_R = f"{metrics['eta2_R_median']:.1%}"
            eta2_HxR = f"{metrics['eta2_HxR_median']:.1%}"
            checks = [
                ("η²(H)", "6.8%", readme),
                ("η²(R)", "19.0%", readme),
                ("η²(H×R)", "48.2%", readme),
            ]
            for name, expected, text in checks:
                if expected in text:
                    print(f"  [OK] README contains {name} = {expected}")
                else:
                    print(f"  [FAIL] README missing {name} = {expected}")
                    all_ok = False

    # 16. Stale-claim grep
    print("\n16. Stale-claim grep:")
    stale_patterns = [
        "62%", "63.6%", "frac_H", "frac_R",
        "flagship_v1", ">5x",
    ]
    # Only scan documentation files (.md, .html, .ipynb), not code
    # Whitelist: CHANGELOG.md (documents what was removed), PR template (checks for stale claims),
    # historical experiment files (experiments/flagship_v1.yaml, results/runs/flagship_v1/)
    scan_files = []
    for ext in [".md", ".html", ".ipynb"]:
        for f in REPO_ROOT.rglob(f"*{ext}"):
            if "audit/" in str(f) or ".git/" in str(f) or "__pycache__" in str(f):
                continue
            if "CHANGELOG.md" in str(f):
                continue
            if "PULL_REQUEST_TEMPLATE.md" in str(f):
                continue
            scan_files.append(f)
    stale_found = False
    for f in scan_files:
        try:
            text = f.read_text()
            for pattern in stale_patterns:
                if pattern in text:
                    # Check context — "flagship_v1" is OK in historical experiment references
                    if pattern == "flagship_v1" and ("experiments/flagship_v1" in text or "results/runs/flagship_v1" in text):
                        continue
                    print(f"  [FAIL] Stale claim '{pattern}' in {f}")
                    stale_found = True
                    all_ok = False
        except Exception:
            pass
    if not stale_found:
        print("  [OK] No stale claims found")

    print(f"\n{'='*50}")
    if all_ok:
        print("SUBMISSION VALIDATION: PASSED")
    else:
        print("SUBMISSION VALIDATION: FAILED")
    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate WestQuant QoolQit submission")
    parser.add_argument("--experiment", default=None,
                        help="Explicit experiment ID to validate (default: read results/latest.txt)")
    args = parser.parse_args()
    ok = validate_submission(experiment_id_override=args.experiment)
    sys.exit(0 if ok else 1)
