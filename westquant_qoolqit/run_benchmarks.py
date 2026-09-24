"""Run benchmark experiments with smoke and flagship modes.

Usage:
    python -m westquant_qoolqit.run_benchmarks --mode smoke
    python -m westquant_qoolqit.run_benchmarks --mode flagship
    python -m westquant_qoolqit.run_benchmarks --config experiments/flagship_v1.yaml

Outputs are stored under results/runs/<experiment_id>/ with:
    config.yaml          — frozen experiment config
    environment.json      — software versions + seed
    raw/cells.jsonl       — per-cell records
    processed/summary.json — aggregate statistics
    processed/report_metrics.json — single source of truth for documentation
    figures/              — generated figures
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .common.reproducibility import capture_environment
from .common.serialization import write_jsonl
from .combined import combined_search
from .benchmarks import (
    mwis_path, mwis_cycle, mwis_grid, mwis_random_geometric, mwis_erdos_renyi,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = REPO_ROOT / "results"


def load_problem(problem_spec: dict):
    """Load a problem from its spec dict."""
    ptype = problem_spec["type"]
    if ptype == "mwis_path":
        return mwis_path(n=problem_spec["n"], seed=problem_spec["seed"])
    elif ptype == "mwis_cycle":
        return mwis_cycle(n=problem_spec["n"], seed=problem_spec["seed"])
    elif ptype == "mwis_grid":
        return mwis_grid(rows=problem_spec["rows"], cols=problem_spec["cols"],
                         seed=problem_spec["seed"])
    elif ptype == "mwis_random_geometric":
        return mwis_random_geometric(n=problem_spec["n"], radius=problem_spec.get("radius", 0.7),
                                     seed=problem_spec["seed"])
    elif ptype == "mwis_erdos_renyi":
        return mwis_erdos_renyi(n=problem_spec["n"], p=problem_spec.get("p", 0.4),
                                seed=problem_spec["seed"])
    else:
        raise ValueError(f"Unknown problem type: {ptype}")


def run_experiment(config: dict, experiment_id: str) -> dict:
    """Run the full benchmark experiment from a config dict."""
    t0 = time.time()
    problems = config["problems"]
    n_h = config["hamiltonian"]["n_representations"]
    n_per_family = config["embedding"]["n_per_family"]
    embedder_families = config["embedding"]["families"]
    num_shots = config["execution"]["num_shots"]
    n_replicates = config["execution"]["n_replicates"]
    schedule = config["control"]["schedule"]
    duration = config["control"].get("duration", 4.0)
    amp_max = config["control"].get("amp_max", 1.5)
    det_max = config["control"].get("det_max", 5.0)
    profile = config["execution"].get("profile", "max_energy")
    device_name = config["execution"].get("device", "AnalogDeviceWithDMM")
    robustness_enabled = config.get("robustness", {}).get("enabled", False)
    robustness_samples = config.get("robustness", {}).get("samples", 0)
    robustness_sigma = config.get("robustness", {}).get("sigma", 0.02)
    base_seed = config["seeds"]["base"]

    # Verify QoolQit version matches config
    import qoolqit
    actual_version = qoolqit.__version__
    expected_version = config.get("qoolqit_version", actual_version)
    if actual_version != expected_version:
        raise RuntimeError(
            f"QoolQit version mismatch: config says {expected_version}, "
            f"but installed version is {actual_version}")

    # Select device from config
    from qoolqit import AnalogDevice, AnalogDeviceWithDMM
    device = AnalogDeviceWithDMM() if device_name == "AnalogDeviceWithDMM" else AnalogDevice()

    all_results = []
    all_records = []

    for pspec in problems:
        print(f"  Running {pspec['id']}...")
        h, info = load_problem(pspec)
        result = combined_search(
            h, mwis_info=info,
            n_hamiltonians=n_h,
            n_embedders=n_per_family * len(embedder_families),
            embedder_methods=embedder_families,
            num_shots=num_shots,
            n_replicates=n_replicates,
            seed=base_seed,
            device=device,
            run_emulation=True,
            run_robustness=robustness_enabled,
            robustness_samples=robustness_samples,
            robustness_sigma=robustness_sigma,
            schedule=schedule,
            duration=duration,
            amp_max=amp_max,
            det_max=det_max,
            profile=profile,
        )
        # Collect per-problem stats using end-to-end success metric
        mat = result.matrix("ground_state_probability")
        vd = result.variance_decomposition("end_to_end_success")
        imp = result.best_vs_baseline("end_to_end_success")
        problem_result = {
            "problem_id": pspec["id"],
            "n_cells": len(result.cells),
            "hamiltonian_ids": result.hamiltonian_ids,
            "embedder_ids": result.embedder_ids,
            "embedder_families": result.embedder_families,
            "variance_decomposition": vd,
            "improvement": imp if imp else None,
        }
        all_results.append(problem_result)
        for rec in result.records:
            rec["problem_id"] = pspec["id"]
        all_records.extend(result.records)

    elapsed = time.time() - t0

    # Aggregate across problems
    absolute_improvements = [r["improvement"]["absolute"] for r in all_results
                             if r.get("improvement")]
    baseline_feasible_count = sum(1 for r in all_results
                                  if r.get("improvement") and r["improvement"].get("baseline_feasible"))
    best_feasible_count = sum(1 for r in all_results
                              if r.get("improvement") and r["improvement"].get("best_feasible"))
    relative_improvements = [r["improvement"]["relative"] for r in all_results
                             if r.get("improvement") and r["improvement"].get("relative") is not None]
    eta2_H_values = [r["variance_decomposition"].get("eta2_H", 0) for r in all_results]
    eta2_R_values = [r["variance_decomposition"].get("eta2_R", 0) for r in all_results]
    eta2_HxR_values = [r["variance_decomposition"].get("eta2_HxR", 0) for r in all_results]
    n_feasible_values = [r["variance_decomposition"].get("n_feasible_cells", 0) for r in all_results]

    # Fraction improved: best > baseline (including baseline infeasible, best feasible)
    n_improved = sum(1 for r in all_results
                     if r.get("improvement") and r["improvement"]["absolute"] > 0)

    report_metrics = {
        "experiment_id": experiment_id,
        "n_problems": len(all_results),
        "n_hamiltonian_representations": n_h,
        "n_embedding_representations": n_per_family * len(embedder_families),
        "n_cells_total": sum(r["n_cells"] for r in all_results),
        "n_replicates": n_replicates,
        "n_feasible_cells_total": sum(n_feasible_values),
        "eta2_H_median": float(np.median(eta2_H_values)) if eta2_H_values else 0.0,
        "eta2_R_median": float(np.median(eta2_R_values)) if eta2_R_values else 0.0,
        "eta2_HxR_median": float(np.median(eta2_HxR_values)) if eta2_HxR_values else 0.0,
        "median_absolute_improvement": float(np.median(absolute_improvements)) if absolute_improvements else 0.0,
        "median_relative_improvement": float(np.median(relative_improvements)) if relative_improvements else None,
        "fraction_improved": n_improved / len(all_results) if all_results else 0.0,
        "n_problems_baseline_feasible": baseline_feasible_count,
        "n_problems_best_feasible": best_feasible_count,
        "elapsed_seconds": elapsed,
        "qoolqit_version": actual_version,
    }

    return {
        "report_metrics": report_metrics,
        "per_problem": all_results,
        "records": all_records,
        "elapsed_seconds": elapsed,
    }


def save_results(experiment_data: dict, config: dict, experiment_id: str):
    """Save results under results/runs/<experiment_id>/."""
    run_dir = RESULTS_DIR / "runs" / experiment_id
    raw_dir = run_dir / "raw"
    processed_dir = run_dir / "processed"
    figures_dir = run_dir / "figures"
    for d in [run_dir, raw_dir, processed_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save config
    import yaml
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save environment
    env = capture_environment(seed=config["seeds"]["base"])
    env_dict = {
        "qoolqit_version": env.qoolqit_version,
        "python_version": env.python_version,
        "numpy_version": env.numpy_version,
        "scipy_version": env.scipy_version,
        "networkx_version": env.networkx_version,
        "platform": env.platform,
        "seed": config["seeds"]["base"],
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(run_dir / "environment.json", "w") as f:
        json.dump(env_dict, f, indent=2)

    # Save raw records
    write_jsonl(experiment_data["records"], str(raw_dir / "cells.jsonl"))

    # Save processed results
    with open(processed_dir / "summary.json", "w") as f:
        json.dump(experiment_data["per_problem"], f, indent=2)

    with open(processed_dir / "report_metrics.json", "w") as f:
        json.dump(experiment_data["report_metrics"], f, indent=2)

    # Update latest pointer
    latest_file = RESULTS_DIR / "latest.txt"
    latest_file.write_text(experiment_id)

    print(f"\nResults saved to: {run_dir}")
    print(f"Report metrics: {processed_dir / 'report_metrics.json'}")
    return run_dir


def generate_figures(experiment_data: dict, run_dir: Path):
    """Generate flagship figures from experiment data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = run_dir / "figures"

    # Figure: H×R heatmap for first problem
    per_problem = experiment_data["per_problem"]
    if per_problem:
        pp = per_problem[0]
        # Reconstruct matrix from records, using MEAN over replicates
        h_ids = pp["hamiltonian_ids"]
        r_ids = pp["embedder_ids"]
        mat = np.full((len(h_ids), len(r_ids)), np.nan)
        hidx = {h: i for i, h in enumerate(h_ids)}
        ridx = {r: i for i, r in enumerate(r_ids)}
        # Collect all values per (H, R) cell
        cell_values: dict[tuple[str, str], list[float]] = {}
        for rec in experiment_data["records"]:
            if rec.get("problem_id", pp["problem_id"]) == pp["problem_id"]:
                if "solution_probability" in rec:
                    h = rec["hamiltonian_id"]
                    r = rec["embedder"]
                    if h in hidx and r in ridx:
                        cell_values.setdefault((h, r), []).append(rec["solution_probability"])
        # Compute mean per cell
        for (h, r), vals in cell_values.items():
            mat[hidx[h], ridx[r]] = float(np.mean(vals))
        if not np.all(np.isnan(mat)):
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(mat, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(r_ids)))
            ax.set_yticks(range(len(h_ids)))
            ax.set_xticklabels(r_ids, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(h_ids, fontsize=8)
            ax.set_xlabel("Embedding representation")
            ax.set_ylabel("Hamiltonian representation")
            ax.set_title(f"H × R solution probability: {pp['problem_id']}")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=9)
            fig.colorbar(im, ax=ax, label="p(optimum)")
            plt.tight_layout()
            plt.savefig(figures_dir / "factorial_heatmap.png", dpi=150)
            plt.close()

    # Figure: Variance decomposition
    rm = experiment_data["report_metrics"]
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["H", "R", "H×R"]
    values = [rm["eta2_H_median"], rm["eta2_R_median"], rm["eta2_HxR_median"]]
    colors = ["#1a237e", "#0d47a1", "#1565c0"]
    ax.bar(categories, values, color=colors)
    ax.set_ylabel("η² (effect size)")
    ax.set_title("Variance decomposition (median across problems)")
    ax.set_ylim(0, max(0.6, max(values) * 1.1))
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(figures_dir / "variance_decomposition.png", dpi=150)
    plt.close()

    # Figure: Baseline vs best
    improvements = []
    labels = []
    for pp in per_problem:
        if pp.get("improvement") and pp["improvement"].get("absolute") is not None:
            improvements.append(pp["improvement"]["absolute"])
            labels.append(pp["problem_id"])
    if improvements:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        ax.bar(x, improvements, color=["#2e7d32" if v > 0 else "#c62828" for v in improvements])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Relative improvement")
        ax.set_title("Search improvement over baseline")
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(figures_dir / "baseline_comparison.png", dpi=150)
        plt.close()

    print(f"Figures saved to: {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run WestQuant QoolQit benchmarks")
    parser.add_argument("--mode", choices=["smoke", "flagship"], default="smoke",
                        help="Benchmark mode (smoke=fast CI, flagship=official results)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment config YAML")
    args = parser.parse_args()

    import yaml
    if args.config:
        config_path = Path(args.config)
    else:
        # --mode flagship → flagship_v3.yaml (current official version)
        # --mode smoke → smoke_v1.yaml
        config_name = f"{args.mode}_v3.yaml" if args.mode == "flagship" else f"{args.mode}_v1.yaml"
        config_path = EXPERIMENTS_DIR / config_name

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_id = config["experiment_id"]
    print(f"=== WestQuant QoolQit Benchmark: {experiment_id} ===")
    print(f"Mode: {args.mode}")
    print(f"Problems: {len(config['problems'])}")
    print(f"Hamiltonians per problem: {config['hamiltonian']['n_representations']}")
    print(f"Embedders per problem: {config['embedding']['n_per_family'] * len(config['embedding']['families'])}")
    print(f"Replicates: {config['execution']['n_replicates']}")
    print(f"Shots: {config['execution']['num_shots']}")
    print()

    experiment_data = run_experiment(config, experiment_id)
    run_dir = save_results(experiment_data, config, experiment_id)
    generate_figures(experiment_data, run_dir)

    rm = experiment_data["report_metrics"]
    print(f"\n=== Results ===")
    print(f"Problems: {rm['n_problems']}")
    print(f"Total cells: {rm['n_cells_total']}")
    print(f"η²(H)  median: {rm['eta2_H_median']:.1%}")
    print(f"η²(R)  median: {rm['eta2_R_median']:.1%}")
    print(f"η²(H×R) median: {rm['eta2_HxR_median']:.1%}")
    print(f"Median absolute improvement: {rm['median_absolute_improvement']:.4f}")
    rel = rm.get('median_relative_improvement')
    if rel is not None:
        print(f"Median relative improvement: {rel:.1%}")
    else:
        print(f"Median relative improvement: N/A (baseline infeasible for some problems)")
    print(f"Fraction improved: {rm['fraction_improved']:.1%}")
    print(f"Problems with feasible baseline: {rm['n_problems_baseline_feasible']}/{rm['n_problems']}")
    print(f"Problems with feasible best: {rm['n_problems_best_feasible']}/{rm['n_problems']}")
    print(f"Feasible H×R cells: {rm['n_feasible_cells_total']}/{rm['n_cells_total']}")
    print(f"Elapsed: {rm['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
