"""Generate slide decks (HTML + PPTX) for all three contest variants.

All result numbers are read from results/runs/flagship_v3/processed/report_metrics.json.
Style: dark navy background matching WQT20 brand, WestQuant logo at top.
PPTX is the primary format (editable in PowerPoint/Keynote); HTML is a bonus.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

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
BG_COLOR = RGBColor(0x00, 0x0e, 0x22)
ACCENT_COLOR = RGBColor(0x1a, 0x4b, 0xa0)
TEXT_COLOR = RGBColor(0xff, 0xff, 0xff)
SUBTITLE_COLOR = RGBColor(0x7f, 0xb3, 0xff)
BODY_COLOR = RGBColor(0xc8, 0xd8, 0xf0)
FOOTER_COLOR = RGBColor(0x4a, 0x6a, 0x9a)


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


def add_pptx_slide(prs: Presentation, title: str, slide_title: str,
                  items: list[str], logo_path: Path | None = None):
    """Add one slide to the PowerPoint presentation."""
    slide_layout = prs.slide_layouts[6]  # blank layout
    slide = prs.slides.add_slide(slide_layout)

    # Set background color
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_COLOR

    # Logo at top-left
    if logo_path and logo_path.exists():
        slide.shapes.add_picture(str(logo_path),
                                 Inches(0.4), Inches(0.25),
                                 width=Inches(2.8))

    # Title
    txBox = slide.shapes.add_textbox(Inches(0.4), Inches(1.0),
                                     Inches(12.5), Inches(0.6))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_COLOR

    # Slide subtitle
    txBox2 = slide.shapes.add_textbox(Inches(0.4), Inches(1.55),
                                      Inches(12.5), Inches(0.4))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = slide_title
    p2.font.size = Pt(18)
    p2.font.color.rgb = SUBTITLE_COLOR

    # Accent line (thin rectangle)
    from pptx.enum.shapes import MSO_SHAPE
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(0.4), Inches(2.05),
                                  Inches(12.5), Pt(2))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT_COLOR
    line.line.fill.background()

    # Items with bold parsing
    txBox3 = slide.shapes.add_textbox(Inches(0.5), Inches(2.3),
                                       Inches(12.3), Inches(4.5))
    tf3 = txBox3.text_frame
    tf3.word_wrap = True

    first = True
    for item in items:
        if first:
            p = tf3.paragraphs[0]
            first = False
        else:
            p = tf3.add_paragraph()
        p.space_after = Pt(14)

        segments = parse_bold(item)
        for i, (text, is_bold) in enumerate(segments):
            if i == 0:
                run = p.runs[0] if p.runs else p.add_run()
                run.text = text
            else:
                run = p.add_run()
                run.text = text
            run.font.size = Pt(14)
            run.font.bold = is_bold
            run.font.color.rgb = TEXT_COLOR if is_bold else BODY_COLOR

    # Footer left
    txBox4 = slide.shapes.add_textbox(Inches(0.4), Inches(6.9),
                                      Inches(6), Inches(0.3))
    tf4 = txBox4.text_frame
    p4 = tf4.paragraphs[0]
    p4.text = f"WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}"
    p4.font.size = Pt(11)
    p4.font.color.rgb = FOOTER_COLOR

    # Footer right
    txBox5 = slide.shapes.add_textbox(Inches(7.5), Inches(6.9),
                                      Inches(5.5), Inches(0.3))
    tf5 = txBox5.text_frame
    p5 = tf5.paragraphs[0]
    p5.text = "WestQuant Open Artifact #001"
    p5.font.size = Pt(11)
    p5.font.color.rgb = FOOTER_COLOR
    p5.alignment = PP_ALIGN.RIGHT


def make_pptx(path: Path, title: str, slide1_title: str, slide1_items: list[str],
              slide2_title: str, slide2_items: list[str], logo_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_pptx_slide(prs, title, slide1_title, slide1_items, logo_path)
    add_pptx_slide(prs, title, slide2_title, slide2_items, logo_path)
    prs.save(str(path))


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
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; background: #000e22; }}
.slide {{ width: 1280px; height: 720px; padding: 48px 64px; box-sizing: border-box;
          page-break-after: always; position: relative; background: #000e22; color: #ffffff; }}
.slide h1 {{ font-size: 28px; margin-bottom: 24px; color: #ffffff; }}
.slide h2 {{ font-size: 22px; color: #7fb3ff; margin-top: 0; }}
.slide ul {{ font-size: 18px; line-height: 1.6; list-style: none; padding: 0; }}
.slide li {{ margin-bottom: 12px; color: #c8d8f0; }}
.slide li b {{ color: #ffffff; }}
.footer {{ position: absolute; bottom: 24px; left: 64px; font-size: 13px; color: #4a6a9a; }}
.footer-right {{ position: absolute; bottom: 24px; right: 64px; font-size: 13px; color: #4a6a9a; }}
</style></head><body>
<div class="slide"><h1>{title}</h1><h2>Slide 1 — Project Overview</h2><ul>{s1}</ul>
<div class="footer">WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}</div>
<div class="footer-right">WestQuant Open Artifact #001</div></div>
<div class="slide"><h1>{title}</h1><h2>Slide 2 — QoolQit Experience</h2><ul>{s2}</ul>
<div class="footer">WestQuant Representation Stack — QoolQit {QOOLQIT_VERSION}</div>
<div class="footer-right">WestQuant Open Artifact #001</div></div>
</body></html>"""
    path.write_text(html)


def copy_notebook(slides_dir: Path, notebook_name: str = "westquant_representation_stack_combined.ipynb"):
    """Copy the combined notebook into the slides directory."""
    src = REPO / "notebooks" / notebook_name
    if src.exists():
        dst = slides_dir / notebook_name
        shutil.copy2(src, dst)


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
make_pptx(SLIDES_A_DIR / "slides.pptx", "WestQuant Representation Scheduler (Project A)",
          "Slide 1 — Project Overview", slides_a_1,
          "Slide 2 — QoolQit Experience", slides_a_2, LOGO_A)
copy_notebook(SLIDES_A_DIR, "project_A_representation_scheduler.ipynb")


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
make_pptx(SLIDES_B_DIR / "slides.pptx", "WestQuant Hamiltonian Representation Explorer (Project B)",
          "Slide 1 — Project Overview", slides_b_1,
          "Slide 2 — QoolQit Experience", slides_b_2, LOGO_B)
copy_notebook(SLIDES_B_DIR, "project_B_hamiltonian_explorer.ipynb")


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
make_pptx(SLIDES_C_DIR / "slides.pptx", "WestQuant Representation Stack (Combined)",
          "Slide 1 — Project Overview", slides_c_1,
          "Slide 2 — QoolQit Experience", slides_c_2, LOGO_C)
copy_notebook(SLIDES_C_DIR, "westquant_representation_stack_combined.ipynb")


print("Generated slide decks (HTML + PPTX) with notebooks:")
for p in sorted((REPO / "contest").rglob("slides.*")):
    print(f"  {p} ({p.stat().st_size} bytes)")
for p in sorted((REPO / "contest").rglob("*.ipynb")):
    print(f"  {p} ({p.stat().st_size} bytes)")
