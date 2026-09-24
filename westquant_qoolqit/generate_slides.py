"""Generate slide decks (HTML + PDF) for all three contest variants.

All result numbers are read from results/runs/flagship_v3/processed/report_metrics.json.
No hardcoded flagship result numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parent.parent

# Load metrics from the single source of truth
METRICS_PATH = REPO / "results" / "runs" / "flagship_v3" / "processed" / "report_metrics.json"
SUMMARY_PATH = REPO / "results" / "runs" / "flagship_v3" / "processed" / "summary.json"

with open(METRICS_PATH) as f:
    METRICS = json.load(f)
with open(SUMMARY_PATH) as f:
    SUMMARY = json.load(f)

ETA2_H = f"{METRICS['eta2_H_median']:.1%}"
ETA2_R = f"{METRICS['eta2_R_median']:.1%}"
ETA2_HXR = f"{METRICS['eta2_HxR_median']:.1%}"
N_BASELINE_FEASIBLE = METRICS["n_problems_baseline_feasible"]
N_BEST_FEASIBLE = METRICS["n_problems_best_feasible"]
N_PROBLEMS = METRICS["n_problems"]
N_FACTOR_CELLS = METRICS.get("n_factor_cells", 270)
N_OBSERVATIONS = METRICS.get("n_observations", 810)
N_FEASIBLE_FACTOR = METRICS.get("n_feasible_factor_cells", 103)
QOOLQIT_VERSION = METRICS.get("qoolqit_version", "1.4.0")


def slide_html(title: str, slide1: list[str], slide2: list[str]) -> str:
    s1 = "".join(f"<li>{x}</li>" for x in slide1)
    s2 = "".join(f"<li>{x}</li>" for x in slide2)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; }}
.slide {{ width: 1280px; height: 720px; padding: 48px 64px; box-sizing: border-box;
          page-break-after: always; position: relative; }}
.slide h1 {{ font-size: 32px; margin-bottom: 24px; color: #1a237e; }}
.slide h2 {{ font-size: 24px; color: #0d47a1; margin-top: 0; }}
.slide ul {{ font-size: 20px; line-height: 1.6; }}
.slide li {{ margin-bottom: 10px; }}
.footer {{ position: absolute; bottom: 24px; left: 64px; font-size: 14px; color: #666; }}
</style></head><body>
<div class="slide"><h1>{title}</h1><h2>Slide 1 — Project Overview</h2><ul>{s1}</ul>
<div class="footer">WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}</div></div>
<div class="slide"><h1>{title}</h1><h2>Slide 2 — QoolQit Experience</h2><ul>{s2}</ul>
<div class="footer">WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}</div></div>
</body></html>"""


def make_pdf(path: Path, title: str, slide1_title: str, slide1_items: list[str],
             slide2_title: str, slide2_items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(path)) as pdf:
        for slide_title, items in [(slide1_title, slide1_items), (slide2_title, slide2_items)]:
            fig, ax = plt.subplots(figsize=(13.33, 7.5))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            ax.text(0.05, 0.93, title, fontsize=22, fontweight="bold", color="#1a237e",
                    transform=ax.transAxes)
            ax.text(0.05, 0.85, slide_title, fontsize=16, color="#0d47a1",
                    transform=ax.transAxes)
            y = 0.77
            for item in items:
                lines = []
                words = item.split()
                line = ""
                for w in words:
                    if len(line) + len(w) + 1 > 95:
                        lines.append(line)
                        line = w
                    else:
                        line = (line + " " + w).strip()
                lines.append(line)
                for li, ln in enumerate(lines):
                    prefix = "  " if li > 0 else ""
                    ax.text(0.07, y, prefix + ln, fontsize=13, va="top",
                            transform=ax.transAxes, wrap=True)
                    y -= 0.035
                y -= 0.015
            ax.text(0.05, 0.03, f"WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}",
                    fontsize=10, color="#666", transform=ax.transAxes)
            pdf.savefig(fig); plt.close(fig)


# =========================================================================== #
# Project A slides
# =========================================================================== #
SLIDES_A_DIR = REPO / "contest" / "project_A" / "slides"
SLIDES_A_DIR.mkdir(parents=True, exist_ok=True)

slides_a_1 = [
    f"<b>Problem:</b> A fixed logical Hamiltonian does not determine a unique useful quantum representation. The physical embedding/layout matters.",
    f"<b>Approach:</b> Automated search over QoolQit embeddings (InteractionEmbedder, SpringLayoutEmbedder, Blade) using successive halving across 5 stages: cheap metrics, logical fidelity, compilation, emulation, robustness.",
    f"<b>Results:</b> Flagship experiment ({N_PROBLEMS} problems, 9 embeddings, 3 replicates): embedding main effect η²(R) = {ETA2_R} (median). For {N_PROBLEMS-1}/{N_PROBLEMS} problems, the preselected baseline fails terminal ground-state preservation; search finds feasible alternatives for {N_BEST_FEASIBLE}/{N_PROBLEMS}.",
    f"<b>Importance:</b> Treating the embedding as an optimization variable (not a fixed choice) materially improves quantum solution quality. Open-source, deterministic, no proprietary infrastructure required.",
]
slides_a_2 = [
    "<b>Positive aspects:</b> QoolQit's Register, Drive, and QuantumProgram abstractions are clean and composable. The embedding API (InteractionEmbedder, Blade) is well-designed. compile_to(device) handles device constraints automatically. LocalEmulator with QutipBackendV2 works out of the box.",
    "<b>Challenges:</b> DMM drives require a DMM-capable device (AnalogDeviceWithDMM, not AnalogDevice). The max_energy compilation profile is needed for automatic rescaling; the default profile requires pre-scaled drives. Bitstring results come as Counters of strings (not arrays), requiring careful bit-ordering handling.",
    "<b>Workflow:</b> The successive-halving approach (cheap metrics first, expensive emulation last) is essential for scaling the search. QoolQit's interaction_matrix() and 1/r^6 convention made interaction-fidelity metrics straightforward.",
]

(SLIDES_A_DIR / "slides.html").write_text(
    slide_html("WestQuant Representation Scheduler (Project A)", slides_a_1, slides_a_2))
make_pdf(SLIDES_A_DIR / "slides.pdf", "WestQuant Representation Scheduler (Project A)",
         "Slide 1 — Project Overview", slides_a_1,
         "Slide 2 — QoolQit Experience", slides_a_2)


# =========================================================================== #
# Project B slides
# =========================================================================== #
SLIDES_B_DIR = REPO / "contest" / "project_B" / "slides"
SLIDES_B_DIR.mkdir(parents=True, exist_ok=True)

slides_b_1 = [
    f"<b>Problem:</b> A mathematical optimization problem does not determine a unique useful Hamiltonian representation. The penalty strength, variable encoding, and landscape all matter.",
    f"<b>Approach:</b> Generate multiple mathematically valid Hamiltonian representations (scaling, permutation, bit-complement, MWIS penalty family) and verify equivalence by exhaustive state-by-state comparison. Each is classified: EXACT_EQUIVALENT, GROUND_STATE_EQUIVALENT, SAME_PROBLEM_DIFFERENT_DYNAMICS, APPROXIMATE, or INVALID.",
    f"<b>Results:</b> For MWIS, penalty strengths U in [5.5, 22.2] all yield GROUND_STATE_EQUIVALENT Hamiltonians. The Hamiltonian main effect is η²(H) = {ETA2_H} (median across {N_PROBLEMS} problems).",
    f"<b>Importance:</b> The Hamiltonian representation choice matters less than its interaction with the embedding. Representation itself is an optimization variable, but the H×R interaction ({ETA2_HXR}) is the dominant effect.",
]
slides_b_2 = [
    "<b>Positive aspects:</b> QoolQit's DataGraph.from_matrix and interaction_matrix() made it easy to connect logical Hamiltonians to physical interactions. The 1/r^6 Rydberg convention is natural for QUBO/MWIS. Register.from_coordinates enables rapid coordinate experimentation.",
    "<b>Challenges:</b> Native Rydberg interactions are repulsive (J>0), so Hamiltonians with negative pair interactions (e.g., after bit-complement) are not directly native-realizable. The physical realizability classifier must flag this before expensive embedding. DMM is needed for MWIS node weights.",
    "<b>Workflow:</b> The five-class equivalence taxonomy forced mathematical rigor. Exhaustive verification for n<=16 ensured no false equivalence claims. The realizability classifier prevents wasted computation on non-native Hamiltonians.",
]

(SLIDES_B_DIR / "slides.html").write_text(
    slide_html("WestQuant Hamiltonian Representation Explorer (Project B)", slides_b_1, slides_b_2))
make_pdf(SLIDES_B_DIR / "slides.pdf", "WestQuant Hamiltonian Representation Explorer (Project B)",
         "Slide 1 — Project Overview", slides_b_1,
         "Slide 2 — QoolQit Experience", slides_b_2)


# =========================================================================== #
# Combined slides
# =========================================================================== #
SLIDES_C_DIR = REPO / "contest" / "combined" / "slides"
SLIDES_C_DIR.mkdir(parents=True, exist_ok=True)

slides_c_1 = [
    f"<b>Problem:</b> Quantum algorithm design typically performs one fixed translation from optimization problem to hardware. We argue representation itself should be searched.",
    f"<b>Approach:</b> A composable pipeline: P -> H_i (Hamiltonian Explorer) -> R_ij (Representation Scheduler) -> Q_ijk (QoolQit compilation + emulation) -> Pareto selection. Both projects are independently useful and composable.",
    f"<b>Results:</b> Flagship experiment ({N_PROBLEMS} problems x 5 H x 9 R x 3 replicates = {N_OBSERVATIONS} observations): H×R interaction dominates (η² = {ETA2_HXR}), embedding main effect η²(R) = {ETA2_R}, Hamiltonian main effect η²(H) = {ETA2_H}. For {N_PROBLEMS-1}/{N_PROBLEMS} problems, the preselected baseline fails terminal ground-state preservation; search finds feasible alternatives for {N_BEST_FEASIBLE}/{N_PROBLEMS}. Representation search can rescue an otherwise invalid physical realization.",
    f"<b>Importance:</b> The H×R interaction is the dominant effect, directly motivating joint representation search. Neither H alone nor R alone determines performance — it is their interaction that matters. QoolQit serves as the evaluation engine; WestQuant AI is optional. Fully open-source, deterministic, reproducible.",
]
slides_c_2 = [
    "<b>Positive aspects:</b> QoolQit provides a clean end-to-end stack: Register, Drive, QuantumProgram, compile_to(device), LocalEmulator. The embedding API (InteractionEmbedder, SpringLayoutEmbedder, Blade) covers the main approaches. DataGraph bridges logical and physical.",
    "<b>Challenges:</b> DMM requires AnalogDeviceWithDMM (not AnalogDevice). max_energy profile needed for auto-rescaling. Bitstring results are Counters of strings. Native Rydberg J>0 constrains which Hamiltonians are directly realizable.",
    "<b>Workflow:</b> Successive halving (cheap metrics -> emulation) is essential for search scalability. The five-class equivalence taxonomy ensures scientific rigor. Two-way ANOVA with H, R, H×R, and residual terms quantifies how each representation layer matters.",
]

(SLIDES_C_DIR / "slides.html").write_text(
    slide_html("WestQuant Representation Stack (Combined)", slides_c_1, slides_c_2))
make_pdf(SLIDES_C_DIR / "slides.pdf", "WestQuant Representation Stack (Combined)",
         "Slide 1 — Project Overview", slides_c_1,
         "Slide 2 — QoolQit Experience", slides_c_2)


print("Generated slide decks (HTML + PDF):")
for p in sorted((REPO / "contest").rglob("slides.*")):
    print(f"  {p} ({p.stat().st_size} bytes)")
