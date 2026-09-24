"""Generate two-slide decks (as HTML/PDF-ready) for both project variants."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SLIDES_A = REPO / "contest" / "project_A" / "slides"
SLIDES_B = REPO / "contest" / "project_B" / "slides"
SLIDES_A.mkdir(parents=True, exist_ok=True)
SLIDES_B.mkdir(parents=True, exist_ok=True)


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
<div class="footer">WestQuant Representation Stack — QoolQit 1.4.0</div></div>
<div class="slide"><h1>{title}</h1><h2>Slide 2 — QoolQit Experience</h2><ul>{s2}</ul>
<div class="footer">WestQuant Representation Stack — QoolQit 1.4.0</div></div>
</body></html>"""


# Project A slides
slides_a = slide_html(
    "WestQuant Representation Scheduler (Project A)",
    [
        "<b>Problem:</b> A fixed logical Hamiltonian does not determine a unique useful quantum representation. The physical embedding/layout matters.",
        "<b>Approach:</b> Automated search over QoolQit embeddings (InteractionEmbedder, SpringLayoutEmbedder, Blade) using successive halving across 5 stages: cheap metrics, logical fidelity, compilation, emulation, robustness.",
        "<b>Results:</b> On a 5-node MWIS instance, different embeddings of the same Hamiltonian yield solution probabilities ranging from 2% to 11%. The best embedding outperforms the default by >2x.",
        "<b>Importance:</b> Treating the embedding as an optimization variable (not a fixed choice) materially improves quantum solution quality. Open-source, deterministic, no proprietary infrastructure required.",
    ],
    [
        "<b>Positive aspects:</b> QoolQit's Register, Drive, and QuantumProgram abstractions are clean and composable. The embedding API (InteractionEmbedder, Blade) is well-designed. compile_to(device) handles device constraints automatically. LocalEmulator with QutipBackendV2 works out of the box.",
        "<b>Challenges:</b> DMM drives require a DMM-capable device (AnalogDeviceWithDMM, not AnalogDevice). The max_energy compilation profile is needed for automatic rescaling; the default profile requires pre-scaled drives. Bitstring results come as Counters of strings (not arrays), requiring careful bit-ordering handling.",
        "<b>Workflow:</b> The successive-halving approach (cheap metrics first, expensive emulation last) is essential for scaling the search. QoolQit's interaction_matrix() and 1/r^6 convention made interaction-fidelity metrics straightforward.",
    ],
)
(SLIDES_A / "slides.html").write_text(slides_a)

# Project B slides
slides_b = slide_html(
    "WestQuant Hamiltonian Representation Explorer (Project B)",
    [
        "<b>Problem:</b> A mathematical optimization problem does not determine a unique useful Hamiltonian representation. The penalty strength, variable encoding, and landscape all matter.",
        "<b>Approach:</b> Generate multiple mathematically valid Hamiltonian representations (scaling, permutation, bit-complement, MWIS penalty family) and verify equivalence by exhaustive state-by-state comparison. Each is classified: EXACT_EQUIVALENT, GROUND_STATE_EQUIVALENT, SAME_PROBLEM_DIFFERENT_DYNAMICS, APPROXIMATE, or INVALID.",
        "<b>Results:</b> For MWIS, penalty strengths U in [5.5, 8.9] all yield GROUND_STATE_EQUIVALENT Hamiltonians with the same optimum but different gaps (1.0 to 6.9) and degeneracies. Bit-complement flips interaction signs, making a Hamiltonian non-native-Rydberg.",
        "<b>Importance:</b> The Hamiltonian representation choice drives 62% of solution-probability variance in our factorial experiment, vs 10% from embedding. Representation itself is an optimization variable.",
    ],
    [
        "<b>Positive aspects:</b> QoolQit's DataGraph.from_matrix and interaction_matrix() made it easy to connect logical Hamiltonians to physical interactions. The 1/r^6 Rydberg convention is natural for QUBO/MWIS. Register.from_coordinates enables rapid coordinate experimentation.",
        "<b>Challenges:</b> Native Rydberg interactions are repulsive (J>0), so Hamiltonians with negative pair interactions (e.g., after bit-complement) are not directly native-realizable. The physical realizability classifier must flag this before expensive embedding. DMM is needed for MWIS node weights.",
        "<b>Workflow:</b> The five-class equivalence taxonomy forced mathematical rigor. Exhaustive verification for n<=16 ensured no false equivalence claims. The realizability classifier prevents wasted computation on non-native Hamiltonians.",
    ],
)
(SLIDES_B / "slides.html").write_text(slides_b)

# Combined slides
SLIDES_C = REPO / "contest" / "combined" / "slides"
SLIDES_C.mkdir(parents=True, exist_ok=True)
slides_c = slide_html(
    "WestQuant Representation Stack (Combined)",
    [
        "<b>Problem:</b> Quantum algorithm design typically performs one fixed translation from optimization problem to hardware. We argue representation itself should be searched.",
        "<b>Approach:</b> A composable pipeline: P -> H_i (Hamiltonian Explorer) -> R_ij (Representation Scheduler) -> Q_ijk (QoolQit compilation + emulation) -> Pareto selection. Both projects are independently useful and composable.",
        "<b>Results:</b> Factorial experiment (3 Hamiltonians x 3 embeddings = 9 cells) on MWIS: Hamiltonian choice drives 62% of variance, embedding 10%. The best (H, R) combination outperforms the worst by >2x in solution probability.",
        "<b>Importance:</b> Representation can be treated as an optimization variable. QoolQit serves as the evaluation engine; WestQuant AI is optional. Fully open-source, deterministic, reproducible.",
    ],
    [
        "<b>Positive aspects:</b> QoolQit provides a clean end-to-end stack: Register, Drive, QuantumProgram, compile_to(device), LocalEmulator. The embedding API (InteractionEmbedder, SpringLayoutEmbedder, Blade) covers the main approaches. DataGraph bridges logical and physical.",
        "<b>Challenges:</b> DMM requires AnalogDeviceWithDMM (not AnalogDevice). max_energy profile needed for auto-rescaling. Bitstring results are Counters of strings. Native Rydberg J>0 constrains which Hamiltonians are directly realizable.",
        "<b>Workflow:</b> Successive halving (cheap metrics -> emulation) is essential for search scalability. The five-class equivalence taxonomy ensures scientific rigor. Variance decomposition quantifies how much each representation layer matters.",
    ],
)
(SLIDES_C / "slides.html").write_text(slides_c)

print("Generated slide decks:")
for p in [SLIDES_A / "slides.html", SLIDES_B / "slides.html", SLIDES_C / "slides.html"]:
    print(f"  {p} ({p.stat().st_size} bytes)")
