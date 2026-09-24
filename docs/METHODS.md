# Methods

This document describes the mathematical foundations, algorithms, and experimental
design of the WestQuant Representation Stack. It is suitable as the foundation for
a future paper.

## 1. Canonical Hamiltonian convention

The canonical representation is a **BinaryQuadraticHamiltonian**:

$$E(\mathbf{x}) = \sum_i h_i x_i + \sum_{i<j} J_{ij} x_i x_j + c$$

where $x_i \in \{0, 1\}$, $h_i$ are linear coefficients, $J_{ij}$ are pairwise
couplings, and $c$ is a constant offset.

## 2. QUBO/Ising conversion

**QUBO convention**: $E(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x}$

For a symmetric QUBO matrix $Q$, the pair coefficient $J_{ij}$ in the canonical
Hamiltonian is $Q_{ij} + Q_{ji}$ (summed, not averaged), so that
$E(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x}$ holds exactly.

**Ising convention**: $E(\mathbf{z}) = \sum_i h_i z_i + \sum_{i<j} J_{ij} z_i z_j + c$
where $z_i \in \{-1, +1\}$.

Conversion: $z_i = 2x_i - 1$.

## 3. Equivalence classes

Every Hamiltonian transformation is classified into exactly one of five classes:

### EXACT_EQUIVALENT

$$E'(T(\mathbf{x})) = a \cdot E(\mathbf{x}) + b, \quad a > 0$$

for every enumerated state $\mathbf{x}$. For $n \leq 16$, verification is
exhaustive over all $2^n$ states. For larger $n$, sampled validation is used
and clearly marked as non-exhaustive.

### GROUND_STATE_EQUIVALENT

The mapped ground-state sets must match exactly:

$$T(\arg\min E) = \arg\min E'$$

including degeneracies. The full spectrum need not match.

### SAME_PROBLEM_DIFFERENT_DYNAMICS

The classical objective and target ground states are the same, but the
physical evolution differs (e.g., different control schedules).

### APPROXIMATE

A quantitative approximation criterion is satisfied (e.g., Spearman
correlation > 0.9 or top-k overlap > 0.5). The criterion is stored with
the result.

### INVALID

The transformation fails all above requirements.

## 4. Representation search

The decomposition is:

$$P \to H_i \to R_{ij} \to Q_{ijk}$$

- **$P$**: optimization problem
- **$H_i$**: alternative Hamiltonian representations (Project B)
- **$R_{ij}$**: embeddings / layouts / control representations (Project A)
- **$Q_{ijk}$**: compiled programs and execution outcomes

### Hamiltonian transforms

| Transform | Operation | Expected class |
|-----------|-----------|----------------|
| Positive scaling | $E' = aE + b$, $a > 0$ | EXACT_EQUIVALENT |
| Variable permutation | $x' = Px$ | EXACT_EQUIVALENT |
| Bit complement | $x_i = 1 - y_i$ on $S$ | EXACT_EQUIVALENT |
| MWIS penalty | $E_U = -\mathbf{w} \cdot \mathbf{x} + U \sum x_i x_j$ | GROUND_STATE_EQUIVALENT (if $U > w_{\max}$) |
| Control decomposition | Same $H$, different drive | SAME_PROBLEM_DIFFERENT_DYNAMICS |

### Embedding families

- **InteractionEmbedder**: optimizes coordinates so $1/r^6$ approximates a target matrix
- **SpringLayoutEmbedder**: NetworkX spring layout with Rydberg-weight adaptation
- **Blade**: BLaDE embedding algorithm for interaction matrices / QUBOs

## 5. QoolQit program construction

### Adiabatic drive

The amplitude schedule starts at 0, rises to $\Omega_{\max}$, and returns to 0:

$$\Omega(t): 0 \to \Omega_{\max} \to 0$$

The detuning sweeps from $-\Delta_{\max}$ to $+\Delta_{\max}$:

$$\Delta(t): -\Delta_{\max} \to +\Delta_{\max}$$

This ensures the terminal transverse field vanishes, so the final measurement
is interpreted against the classical target objective.

### MWIS DMM encoding

For MWIS with positive vertex weights $w_i$:

$$\epsilon_i = 1 - \frac{w_i}{w_{\max}}$$

The DMM waveform is a negative constant (DMM detunings are $\leq 0$).

## 6. Experimental design

### Factorial design

The flagship experiment is a balanced factorial:

$$Y_{hrk} = \mu + \alpha_h + \beta_r + (\alpha\beta)_{hr} + \epsilon_{hrk}$$

where $h$ = Hamiltonian, $r$ = embedding, $k$ = replicate.

### Two-way ANOVA

Sums of squares are decomposed:

- $SS_H$: Hamiltonian main effect
- $SS_R$: Embedding main effect
- $SS_{H \times R}$: interaction
- $SS_{\text{residual}}$: within-cell variation

Effect sizes: $\eta^2 = SS / SS_{\text{total}}$.

### Uncertainty

Wilson score confidence intervals for binomial proportions:

$$\frac{p + z^2/(2n)}{1 + z^2/n} \pm \frac{z\sqrt{p(1-p)/n + z^2/(4n^2)}}{1 + z^2/n}$$

## 7. Limitations

- Small-$n$ exact benchmarks ($n \leq 10$) due to QuTiP runtime
- Local emulation rather than hardware execution
- Limited representation families (3 embedders, MWIS penalty family)
- MWIS-focused empirical evaluation
- Finite-shot uncertainty
- QoolQit 1.4.0 dependency (API may change in future versions)
- Terminal encoding validation does not always preserve MWIS ground state
  with default physical parameters (requires parameter tuning)
