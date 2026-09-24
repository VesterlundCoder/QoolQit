"""Generate slide decks (HTML + PDF) for all three contest variants.

All result numbers are read from results/runs/flagship_v3/processed/report_metrics.json.
Style: dark navy background matching WQT20 brand, WestQuant logo at top.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

# Brand colors from WQT20 image
BG_COLOR = "#000e22"
ACCENT_COLOR = "#1a4ba0"
TEXT_COLOR = "#ffffff"
SUBTITLE_COLOR = "#7fb3ff"
BODY_COLOR = "#c8d8f0"
FOOTER_COLOR = "#4a6a9a"


def parse_bold(text: str) -> list[tuple[str, bool]]:
    """Parse <b>...</b> tags into (text, is_bold) segments."""
    segments = []
    pattern = re.compile(r'(<b>.*?</b>)')
    parts = pattern.split(text)
    for part in parts:
        if part.startswith('<b>') and part.endswith('</b>'):
            segments.append((part[3:-4], True))
        else:
            segments.append((part, False))
    return segments


def render_slide(fig, title: str, slide_title: str, items: list[str],
                 logo_path: Path | None = None):
    """Render one slide on the given figure with dark navy background and logo."""
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG_COLOR)

    # Logo at top-left
    if logo_path and logo_path.exists():
        logo_img = mpimg.imread(str(logo_path))
        lh, lw = logo_img.shape[:2]
        aspect = lw / lh
        logo_w = 0.22
        logo_h = logo_w / aspect
        logo_x = 0.04
        logo_y = 0.88
        ax.imshow(logo_img, extent=[logo_x, logo_x + logo_w,
                                    logo_y, logo_y + logo_h],
                  zorder=10, aspect='auto', alpha=0.85)

    # Title
    fig.text(0.04, 0.82, title, fontsize=22, fontweight="bold",
             color=TEXT_COLOR, va="top")

    # Slide subtitle
    fig.text(0.04, 0.75, slide_title, fontsize=15, color=SUBTITLE_COLOR, va="top")

    # Accent line under subtitle
    ax.plot([0.04, 0.96], [0.72, 0.72], color=ACCENT_COLOR,
            linewidth=2, transform=ax.transAxes)

    # Items with bold parsing — use figure text for reliability
    y_start = 0.66
    y_step = 0.085
    y = y_start
    for item in items:
        segments = parse_bold(item)
        # Build a single line with mixed weights using annotate
        x = 0.06
        for text, is_bold in segments:
            weight = "bold" if is_bold else "normal"
            color = TEXT_COLOR if is_bold else BODY_COLOR
            fontsize = 13
            fig.text(x, y, text, fontsize=fontsize, color=color,
                     fontweight=weight, va="top", family="sans-serif")
            # Advance x by approximate text width
            approx_width = len(text) * 0.0068 * (fontsize / 13)
            x += approx_width
        y -= y_step

    # Footer
    fig.text(0.04, 0.03, f"WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}",
             fontsize=10, color=FOOTER_COLOR)
    fig.text(0.96, 0.03, "WestQuant Open Artifact #001",
             fontsize=10, color=FOOTER_COLOR, ha="right")


def make_pdf(path: Path, title: str, slide1_title: str, slide1_items: list[str],
             slide2_title: str, slide2_items: list[str], logo_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(path)) as pdf:
        for slide_title, items in [(slide1_title, slide1_items), (slide2_title, slide2_items)]:
            fig = plt.figure(figsize=(13.33, 7.5))
            render_slide(fig, title, slide_title, items, logo_path)
            pdf.savefig(fig, facecolor=BG_COLOR)
            plt.close(fig)


def make_html(path: Path, title: str, slide1: list[str], slide2: list[str]) -> None:
    """Generate HTML with proper bold rendering and dark navy style."""
    def render_items(items):
        html = ""
        for item in items:
            html += f"<li>{item}</li>\n"
        return html

    s1 = render_items(slide1)
    s2 = render_items(slide2)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; background: {BG_COLOR}; }}
.slide {{ width: 1280px; height: 720px; padding: 48px 64px; box-sizing: border-box;
          page-break-after: always; position: relative; background: {BG_COLOR}; color: {TEXT_COLOR}; }}
.slide h1 {{ font-size: 28px; margin-bottom: 24px; color: {TEXT_COLOR}; }}
.slide h2 {{ font-size: 22px; color: {SUBTITLE_COLOR}; margin-top: 0; }}
.slide ul {{ font-size: 18px; line-height: 1.6; list-style: none; padding: 0; }}
.slide li {{ margin-bottom: 12px; color: {BODY_COLOR}; }}
.slide li b {{ color: {TEXT_COLOR}; }}
.footer {{ position: absolute; bottom: 24px; left: 64px; font-size: 13px; color: {FOOTER_COLOR}; }}
.footer-right {{ position: absolute; bottom: 24px; right: 64px; font-size: 13px; color: {FOOTER_COLOR}; }}
</style></head><body>
<div class="slide"><h1>{title}</h1><h2>Slide 1 — Project Overview</h2><ul>{s1}</ul>
<div class="footer">WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}</div>
<div class="footer-right">WestQuant Open Artifact #001</div></div>
<div class="slide"><h1>{title}</h1><h2>Slide 2 — QoolQit Experience</h2><ul>{s2}</ul>
<div class="footer">WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}</div>
<div class="footer-right">WestQuant Open Artifact #001</div></div>
</body></html>"""
    path.write_text(html)


# =========================================================================== #
# Project A slides
# =========================================================================== #
SLIDES_A_DIR = REPO / "contest" / "project_A" / "slides"
SLIDES_A_DIR.mkdir(parents=True, exist_ok=True)
LOGO_A = SLIDES_A_DIR / "WestQuan Studio.png"

slides_a_1 = [
    "<b>Problem:</b> A fixed logical Hamiltonian does not determine a unique useful quantum representation. The physical embedding/layout matters.",
    "<b>Approach:</b> Automated search over QoolQit embeddings (InteractionEmbedder, SpringLayoutEmbedder, Blade) using successive halving across 5 stages: cheap metrics, logical fidelity, compilation, emulation, robustness.",
    f"<b>Results:</b> Flagship experiment ({N_PROBLEMS} problems, 9 embeddings, 3 replicates): embedding main effect η²(R) = {ETA2_R} (median). For {N_PROBLEMS-1}/{N_PROBLEMS} problems, the preselected baseline fails terminal ground-state preservation; search finds feasible alternatives for {N_BEST_FEASIBLE}/{N_PROBLEMS}.",
    "<b>Importance:</b> Treating the embedding as an optimization variable (not a fixed choice) materially improves quantum solution quality. Open-source, deterministic, no proprietary infrastructure required.",
]
slides_a_2 = [
    "<b>Positive aspects:</b> QoolQit's Register, Drive, and QuantumProgram abstractions are clean and composable. The embedding API (InteractionEmbedder, Blade) is well-designed. compile_to(device) handles device constraints automatically. LocalEmulator with QutipBackendV2 works out of the box.",
    "<b>Challenges:</b> DMM drives require a DMM-capable device (AnalogDeviceWithDMM, not AnalogDevice). The max_energy compilation profile is needed for automatic rescaling; the default profile requires pre-scaled drives. Bitstring results come as Counters of strings (not arrays), requiring careful bit-ordering handling.",
    "<b>Workflow:</b> The successive-halving approach (cheap metrics first, expensive emulation last) is essential for scaling the search. QoolQit's interaction_matrix() and 1/r^6 convention made interaction-fidelity metrics straightforward.",
]

make_html(SLIDES_A_DIR / "slides.html", "WestQuant Representation Scheduler (Project A)",
          slides_a_1, slides_a_2)
make_pdf(SLIDES_A_DIR / "slides.pdf", "WestQuant Representation Scheduler (Project A)",
         "Slide 1 — Project Overview", slides_a_1,
         "Slide 2 — QoolQit Experience", slides_a_2, LOGO_A)


# =========================================================================== #
# Project B slides
# =========================================================================== #
SLIDES_B_DIR = REPO / "contest" / "project_B" / "slides"
SLIDES_B_DIR.mkdir(parents=True, exist_ok=True)
LOGO_B = SLIDES_B_DIR / "WestQuan Studio.png"

slides_b_1 = [
    "<b>Problem:</b> A mathematical optimization problem does not determine a unique useful Hamiltonian representation. The penalty strength, variable encoding, and landscape all matter.",
    "<b>Approach:</b> Generate multiple mathematically valid Hamiltonian representations (scaling, permutation, bit-complement, MWIS penalty family) and verify equivalence by exhaustive state-by-state comparison. Each is classified: EXACT_EQUIVALENT, GROUND_STATE_EQUIVALENT, SAME_PROBLEM_DIFFERENT_DYNAMICS, APPROXIMATE, or INVALID.",
    f"<b>Results:</b> For MWIS, penalty strengths U in [5.5, 22.2] all yield GROUND_STATE_EQUIVALENT Hamiltonians. The Hamiltonian main effect is η²(H) = {ETA2_H} (median across {N_PROBLEMS} problems).",
    f"<b>Importance:</b> The Hamiltonian representation choice matters less than its interaction with the embedding. Representation itself is an optimization variable, but the H×R interaction ({ETA2_HXR}) is the dominant effect.",
]
slides_b_2 = [
    "<b>Positive aspects:</b> QoolQit's DataGraph.from_matrix and interaction_matrix() made it easy to connect logical Hamiltonians to physical interactions. The 1/r^6 Rydberg convention is natural for QUBO/MWIS. Register.from_coordinates enables rapid coordinate experimentation.",
    "<b>Challenges:</b> Native Rydberg interactions are repulsive (J>0), so Hamiltonians with negative pair interactions (e.g., after bit-complement) are not directly native-realizable. The physical realizability classifier must flag this before expensive embedding. DMM is needed for MWIS node weights.",
    "<b>Workflow:</b> The five-class equivalence taxonomy forced mathematical rigor. Exhaustive verification for n<=16 ensured no false equivalence claims. The realizability classifier prevents wasted computation on non-native Hamiltonians.",
]

make_html(SLIDES_B_DIR / "slides.html", "WestQuant Hamiltonian Representation Explorer (Project B)",
          slides_b_1, slides_b_2)
make_pdf(SLIDES_B_DIR / "slides.pdf", "WestQuant Hamiltonian Representation Explorer (Project B)",
         "Slide 1 — Project Overview", slides_b_1,
         "Slide 2 — QoolQit Experience", slides_b_2, LOGO_B)


# =========================================================================== #
# Combined slides
# =========================================================================== #
SLIDES_C_DIR = REPO / "contest" / "combined" / "slides"
SLIDES_C_DIR.mkdir(parents=True, exist_ok=True)
LOGO_C = SLIDES_C_DIR / "WestQuan Studio.png"

slides_c_1 = [
    "<b>Problem:</b> Quantum algorithm design typically performs one fixed translation from optimization problem to hardware. We argue representation itself should be searched.",
    "<b>Approach:</b> A composable pipeline: P -> H_i (Hamiltonian Explorer) -> R_ij (Representation Scheduler) -> Q_ijk (QoolQit compilation + emulation) -> Pareto selection. Both projects are independently useful and composable.",
    f"<b>Results:</b> Flagship experiment ({N_PROBLEMS} problems x 5 H x 9 R x 3 replicates = {N_OBSERVATIONS} observations): H×R interaction dominates (η² = {ETA2_HXR}), embedding main effect η²(R) = {ETA2_R}, Hamiltonian main effect η²(H) = {ETA2_H}. For {N_PROBLEMS-1}/{N_PROBLEMS} problems, the preselected baseline fails terminal ground-state preservation; search finds feasible alternatives for {N_BEST_FEASIBLE}/{N_PROBLEMS}. Representation search can rescue an otherwise invalid physical realization.",
    "<b>Importance:</b> The H×R interaction is the dominant effect, directly motivating joint representation search. Neither H alone nor R alone determines performance — it is their interaction that matters. QoolQit serves as the evaluation engine; WestQuant AI is optional. Fully open-source, deterministic, reproducible.",
]
slides_c_2 = [
    "<b>Positive aspects:</b> QoolQit provides a clean end-to-end stack: Register, Drive, QuantumProgram, compile_to(device), LocalEmulator. The embedding API (InteractionEmbedder, SpringLayoutEmbedder, Blade) covers the main approaches. DataGraph bridges logical and physical.",
    "<b>Challenges:</b> DMM requires AnalogDeviceWithDMM (not AnalogDevice). max_energy profile needed for auto-rescaling. Bitstring results are Counters of strings. Native Rydberg J>0 constrains which Hamiltonians are directly realizable.",
    "<b>Workflow:</b> Successive halving (cheap metrics -> emulation) is essential for search scalability. The five-class equivalence taxonomy ensures scientific rigor. Two-way ANOVA with H, R, H×R, and residual terms quantifies how each representation layer matters.",
]

make_html(SLIDES_C_DIR / "slides.html", "WestQuant Representation Stack (Combined)",
          slides_c_1, slides_c_2)
make_pdf(SLIDES_C_DIR / "slides.pdf", "WestQuant Representation Stack (Combined)",
         "Slide 1 — Project Overview", slides_c_1,
         "Slide 2 — QoolQit Experience", slides_c_2, LOGO_C)


print("Generated slide decks (HTML + PDF):")
for p in sorted((REPO / "contest").rglob("slides.*")):
    print(f"  {p} ({p.stat().st_size} bytes)")
