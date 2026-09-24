"""Generate two-slide PDF decks using matplotlib (no LaTeX dependency)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parent.parent


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
                # wrap long lines
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
            ax.text(0.05, 0.03, "WestQuant Representation Stack — QoolQit 1.4.0",
                    fontsize=10, color="#666", transform=ax.transAxes)
            pdf.savefig(fig); plt.close(fig)


make_pdf(
    REPO / "contest" / "project_A" / "slides" / "slides.pdf",
    "WestQuant Representation Scheduler (Project A)",
    "Slide 1 — Project Overview",
    [
        "Problem: A fixed logical Hamiltonian does not determine a unique useful quantum representation. The physical embedding/layout matters.",
        "Approach: Automated search over QoolQit embeddings (InteractionEmbedder, SpringLayoutEmbedder, Blade) using successive halving across 5 stages: cheap metrics, logical fidelity, compilation, emulation, robustness.",
        "Results: Flagship experiment (6 problems, 9 embeddings, 3 replicates): embedding main effect η²(R) = 33.1% (median). Search improves over baseline in 100% of problems.",
        "Importance: Treating the embedding as an optimization variable (not a fixed choice) materially improves quantum solution quality. Open-source, deterministic, no proprietary infrastructure required.",
    ],
    "Slide 2 — QoolQit Experience",
    [
        "Positive: QoolQit's Register, Drive, QuantumProgram abstractions are clean and composable. The embedding API (InteractionEmbedder, Blade) is well-designed. compile_to(device) handles device constraints automatically. LocalEmulator with QutipBackendV2 works out of the box.",
        "Challenges: DMM drives require AnalogDeviceWithDMM (not AnalogDevice). The max_energy profile is needed for automatic rescaling. Bitstring results come as Counters of strings, requiring careful bit-ordering handling. InteractionEmbedder uses a fixed internal RNG seed when x0=None.",
        "Workflow: Successive halving (cheap metrics first, expensive emulation last) is essential for search scalability. QoolQit's interaction_matrix() and 1/r^6 convention made interaction-fidelity metrics straightforward.",
    ],
)

make_pdf(
    REPO / "contest" / "project_B" / "slides" / "slides.pdf",
    "WestQuant Hamiltonian Representation Explorer (Project B)",
    "Slide 1 — Project Overview",
    [
        "Problem: A mathematical optimization problem does not determine a unique useful Hamiltonian representation. Penalty strength, variable encoding, and landscape all matter.",
        "Approach: Generate multiple valid representations (scaling, permutation, bit-complement, MWIS penalty family) and verify equivalence by exhaustive state-by-state comparison. Each is classified: EXACT_EQUIVALENT, GROUND_STATE_EQUIVALENT, SAME_PROBLEM_DIFFERENT_DYNAMICS, APPROXIMATE, or INVALID.",
        "Results: For MWIS, penalty strengths U in [5.5, 22.2] all yield GROUND_STATE_EQUIVALENT Hamiltonians. The Hamiltonian main effect is η²(H) = 7.2% (median across 6 problems).",
        "Importance: The Hamiltonian representation choice matters less than its interaction with the embedding. Representation itself is an optimization variable, but the H×R interaction (47.7%) is the dominant effect.",
    ],
    "Slide 2 — QoolQit Experience",
    [
        "Positive: QoolQit's DataGraph.from_matrix and interaction_matrix() made it easy to connect logical Hamiltonians to physical interactions. The 1/r^6 Rydberg convention is natural for QUBO/MWIS. Register.from_coordinates enables rapid coordinate experimentation.",
        "Challenges: Native Rydberg interactions are repulsive (J>0), so Hamiltonians with negative pair interactions (e.g., after bit-complement) are not directly native-realizable. The realizability classifier must flag this before expensive embedding. DMM is needed for MWIS node weights.",
        "Workflow: The five-class equivalence taxonomy forced mathematical rigor. Exhaustive verification for n<=16 ensured no false equivalence claims. Forward-map ground-state verification was corrected to map argmin E through T.",
    ],
)

make_pdf(
    REPO / "contest" / "combined" / "slides" / "slides.pdf",
    "WestQuant Representation Stack (Combined)",
    "Slide 1 — Project Overview",
    [
        "Problem: Quantum algorithm design typically performs one fixed translation from optimization problem to hardware. We argue representation itself should be searched.",
        "Approach: A composable pipeline: P -> H_i (Hamiltonian Explorer) -> R_ij (Representation Scheduler) -> Q_ijk (QoolQit compilation + emulation) -> Pareto selection. Both projects are independently useful and composable.",
        "Results: Flagship experiment (6 problems x 5 H x 9 R x 3 replicates = 810 cells): H×R interaction dominates (η² = 47.7%), embedding main effect η²(R) = 33.1%, Hamiltonian main effect η²(H) = 7.2%. Search improves over baseline in 100% of problems (median improvement: 65.9%).",
        "Importance: The H×R interaction is the dominant effect, directly motivating joint representation search. Neither H alone nor R alone determines performance — it is their interaction that matters. QoolQit serves as the evaluation engine; WestQuant AI is optional. Fully open-source, deterministic, reproducible.",
    ],
    "Slide 2 — QoolQit Experience",
    [
        "Positive: QoolQit provides a clean end-to-end stack: Register, Drive, QuantumProgram, compile_to(device), LocalEmulator. The embedding API (InteractionEmbedder, SpringLayoutEmbedder, Blade) covers the main approaches. DataGraph bridges logical and physical.",
        "Challenges: DMM requires AnalogDeviceWithDMM (not AnalogDevice). max_energy profile needed for auto-rescaling. Bitstring results are Counters of strings. Native Rydberg J>0 constrains which Hamiltonians are directly realizable. InteractionEmbedder x0=None uses a fixed internal seed.",
        "Workflow: Successive halving (cheap metrics -> emulation) is essential for search scalability. The five-class equivalence taxonomy ensures scientific rigor. Two-way ANOVA with H, R, H×R, and residual terms quantifies how each representation layer matters.",
    ],
)

print("Generated PDF slide decks:")
for p in sorted((REPO / "contest").rglob("slides.pdf")):
    print(f"  {p} ({p.stat().st_size} bytes)")
