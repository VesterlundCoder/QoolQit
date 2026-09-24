"""Benchmark runner: generates the factorial experiment, figures, and manifest.

Run from the repo root:
    python -m westquant_qoolqit.run_benchmarks

Produces:
    results/raw/benchmark_cells.jsonl
    results/processed/benchmark_summary.json
    results/figures/factorial_heatmap.png
    results/figures/variance_decomposition.png
    results/figures/pareto_front.png
    results/manifests/environment.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import capture_environment, write_jsonl, solve_exact
from .combined import combined_search
from .benchmarks import mwis_path, mwis_grid, mwis_random_geometric, synthetic_qubo


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def run_mwis_factorial(n: int = 4, num_shots: int = 200, seed: int = 42) -> dict:
    """Run the flagship MWIS H x R factorial experiment."""
    h, info = mwis_path(n=n, seed=seed)
    res = combined_search(
        h, mwis_info=info, n_hamiltonians=3, n_embedders=3,
        num_shots=num_shots, seed=seed, run_emulation=True, run_robustness=True,
        robustness_samples=3,
    )
    return {"result": res, "problem": h, "info": info}


def plot_factorial_heatmap(res, metric: str, title: str, path: Path) -> None:
    mat = res.matrix(metric)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels([f"R{i+1}" for i in range(mat.shape[1])])
    ax.set_yticklabels(res.hamiltonian_ids)
    ax.set_xlabel("Embedding candidate")
    ax.set_ylabel("Hamiltonian representation")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variance_decomposition(res, path: Path) -> None:
    metrics = ["ground_state_probability", "frobenius_error", "logical_fidelity_rho"]
    fracs = []
    for m in metrics:
        vd = res.variance_decomposition(m)
        fracs.append((m, vd.get("frac_H", 0), vd.get("frac_R", 0)))
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    w = 0.35
    h_fracs = [f[1] for f in fracs]
    r_fracs = [f[2] for f in fracs]
    ax.bar(x - w/2, h_fracs, w, label="Hamiltonian (H)", color="#2196F3")
    ax.bar(x + w/2, r_fracs, w, label="Embedding (R)", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=15)
    ax.set_ylabel("Fraction of total variance")
    ax.set_title("Variance decomposition: H vs R")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pareto(res, path: Path) -> None:
    """Pareto scatter: frobenius error (x, lower better) vs solution probability (y, higher)."""
    xs, ys, ids = [], [], []
    for c in res.cells:
        if c.frobenius_error is not None and c.ground_state_probability is not None:
            xs.append(c.frobenius_error)
            ys.append(c.ground_state_probability)
            ids.append(f"{c.hamiltonian_id}\n{c.embedder}")
    if not xs:
        return
    xs = np.array(xs); ys = np.array(ys)
    # simple pareto front (min x, max y)
    keep = np.ones(len(xs), dtype=bool)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i != j and xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                keep[i] = False
                break
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs[~keep], ys[~keep], c="gray", alpha=0.5, label="dominated")
    ax.scatter(xs[keep], ys[keep], c="red", s=80, label="Pareto front", zorder=5)
    for i in np.where(keep)[0]:
        ax.annotate(ids[i], (xs[i], ys[i]), fontsize=7, alpha=0.7)
    ax.set_xlabel("Interaction Frobenius error (lower is better)")
    ax.set_ylabel("Solution probability (higher is better)")
    ax.set_title("Pareto front across H x R representations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "raw").mkdir(exist_ok=True)
    (RESULTS / "processed").mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    (RESULTS / "manifests").mkdir(exist_ok=True)

    # environment manifest
    env = capture_environment(seed=42)
    (RESULTS / "manifests" / "environment.json").write_text(json.dumps(env.to_dict(), indent=2))

    # flagship MWIS factorial
    print("Running MWIS path(n=4) factorial...")
    out = run_mwis_factorial(n=4, num_shots=200, seed=42)
    res = out["result"]
    write_jsonl(res.records, RESULTS / "raw" / "benchmark_cells.jsonl")

    # summary
    summary = {
        "problem": "MWIS path n=4",
        "n_cells": len(res.cells),
        "hamiltonian_ids": res.hamiltonian_ids,
        "variance_decomposition_p_opt": res.variance_decomposition("ground_state_probability"),
        "variance_decomposition_frob": res.variance_decomposition("frobenius_error"),
        "best_cell_p_opt": max(res.cells, key=lambda c: c.ground_state_probability or -1).hamiltonian_id
                            if any(c.ground_state_probability for c in res.cells) else None,
    }
    (RESULTS / "processed" / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # figures
    plot_factorial_heatmap(res, "ground_state_probability",
                           "Solution probability: H x R factorial",
                           RESULTS / "figures" / "factorial_heatmap.png")
    plot_variance_decomposition(res, RESULTS / "figures" / "variance_decomposition.png")
    plot_pareto(res, RESULTS / "figures" / "pareto_front.png")
    print(f"Benchmark complete. Results in {RESULTS}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
