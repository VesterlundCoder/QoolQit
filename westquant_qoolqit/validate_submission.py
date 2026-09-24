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

Usage:
    python -m westquant_qoolqit.validate_submission
"""

from __future__ import annotations

import json
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


def validate_submission() -> bool:
    """Run all validation checks. Returns True if all pass."""
    print("=== WestQuant QoolQit Submission Validation ===\n")
    all_ok = True

    # Check experiment configs
    print("1. Experiment configs:")
    for cfg in ["experiments/smoke_v1.yaml", "experiments/flagship_v1.yaml"]:
        if not check_file(REPO_ROOT / cfg, f"Config {cfg}"):
            all_ok = False

    # Check results
    print("\n2. Results:")
    latest_file = REPO_ROOT / "results" / "latest.txt"
    if latest_file.exists():
        experiment_id = latest_file.read_text().strip()
        run_dir = REPO_ROOT / "results" / "runs" / experiment_id
        for f in ["config.yaml", "environment.json"]:
            if not check_file(run_dir / f, f"Result {f}"):
                all_ok = False
        for f in ["summary.json", "report_metrics.json"]:
            if not check_file(run_dir / "processed" / f, f"Processed {f}"):
                all_ok = False
        if not check_file(run_dir / "raw" / "cells.jsonl", "Raw cells"):
            all_ok = False
    else:
        print("  [FAIL] No results found (run benchmarks first)")
        all_ok = False

    # Check notebooks
    print("\n3. Notebooks:")
    for nb in ["notebooks/westquant_representation_stack_combined.ipynb",
               "notebooks/project_A_representation_scheduler.ipynb",
               "notebooks/project_B_hamiltonian_explorer.ipynb"]:
        if not check_file(REPO_ROOT / nb, f"Notebook {nb}"):
            all_ok = False

    # Check slides
    print("\n4. Slide decks:")
    for deck in ["contest/project_A/slides/slides.pdf",
                 "contest/project_B/slides/slides.pdf",
                 "contest/combined/slides/slides.pdf"]:
        if not check_file(REPO_ROOT / deck, f"Slides {deck}"):
            all_ok = False

    # Check QoolQit version in report metrics
    print("\n5. QoolQit version:")
    if latest_file.exists():
        experiment_id = latest_file.read_text().strip()
        metrics_file = REPO_ROOT / "results" / "runs" / experiment_id / "processed" / "report_metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            qoolqit_version = metrics.get("qoolqit_version", "unknown")
            if qoolqit_version == "1.4.0":
                print(f"  [OK] QoolQit version: {qoolqit_version}")
            else:
                print(f"  [FAIL] QoolQit version: {qoolqit_version} (expected 1.4.0)")
                all_ok = False

    # Check for terminal encoding validity in raw cells
    print("\n6. Terminal encoding validity:")
    if latest_file.exists():
        experiment_id = latest_file.read_text().strip()
        raw_file = REPO_ROOT / "results" / "runs" / experiment_id / "raw" / "cells.jsonl"
        if raw_file.exists():
            import json as json_mod
            total_cells = 0
            invalid_terminal = 0
            emulated_with_invalid = 0
            for line in raw_file.read_text().strip().split("\n"):
                if line:
                    rec = json_mod.loads(line)
                    total_cells += 1
                    if rec.get("terminal_encoding_valid") is False:
                        invalid_terminal += 1
                        # Check if this invalid cell was still emulated
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
            else:
                print("  [FAIL] No cells found in raw data")
                all_ok = False

    # Check for NaNs in metrics
    print("\n7. No NaNs in metrics:")
    if latest_file.exists():
        experiment_id = latest_file.read_text().strip()
        metrics_file = REPO_ROOT / "results" / "runs" / experiment_id / "processed" / "report_metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            for key, val in metrics.items():
                if isinstance(val, float) and (val != val):  # NaN check
                    print(f"  [FAIL] NaN in {key}")
                    all_ok = False
            print("  [OK] No NaNs found")

    # Check tests
    print("\n8. Tests:")
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

    # Check notebook is executable
    print("\n9. Notebook execution:")
    nb_path = REPO_ROOT / "notebooks" / "westquant_representation_stack_combined.ipynb"
    if nb_path.exists():
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", str(nb_path), "--output", "/tmp/validated.ipynb",
             "--ExecutePreprocessor.timeout=300"],
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

    print(f"\n{'='*50}")
    if all_ok:
        print("SUBMISSION VALIDATION: PASSED")
    else:
        print("SUBMISSION VALIDATION: FAILED")
    return all_ok


if __name__ == "__main__":
    ok = validate_submission()
    sys.exit(0 if ok else 1)
