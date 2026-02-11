# Day 248: Spectral Theorem for Bounded Self-Adjoint Operators

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning I** | 2 hours | Spectral Measures and Projection-Valued Measures |
| **Morning II** | 2 hours | The Spectral Theorem (General Case) |
| **Afternoon** | 2 hours | Resolution of the Identity and Applications |
| **Evening** | 2 hours | Computational Lab: Spectral Measures |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** spectral measures and projection-valued measures (PVM)
2. **State and understand** the spectral theorem for bounded self-adjoint operators
3. **Construct** the spectral family (resolution of the identity) for an operator
4. **Interpret** the spectral integral $A = \int \lambda \, dE_\lambda$
5. **Connect** spectral projections to quantum measurement collapse
6. **Apply** the spectral theorem to compute operator properties

## Core Content

### 1. Motivation: Beyond Compact Operators

For compact self-adjoint operators, we had eigenvalue decompositions:
$$A = \sum_n \lambda_n |e_n\rangle\langle e_n|$$

But many important operators are **not compact**:
- Multiplication operators: $(Mf)(x) = xf(x)$ on $L^2[0,1]$
- Position operator: $(\hat{x}\psi)(x) = x\psi(x)$
- Identity operator on infinite-dimensional spaces

These operators have **continuous spectrum**—no eigenvectors in the usual sense!

**Key Insight**: Replace discrete sums with integrals over the spectrum:
$$A = \int_{\sigma(A)} \lambda \, dE_\lambda$$

### 2. Projection-Valued Measures

**Definition (Projection-Valued Measure)**

Let $(\Omega, \mathcal{F})$ be a measurable space and $\mathcal{H}$ a Hilbert space. A **projection-valued measure** (PVM) is a map $E: \mathcal{F} \to \mathcal{B}(\mathcal{H})$ such that:

1. **Projections**: Each $E(\Delta)$ is an orthogonal projection
2. **Normalized**: $E(\emptyset) = 0$ and $E(\Omega) = I$
3. **$\sigma$-additivity**: For disjoint $\{\Delta_n\}$:
$$E\left(\bigcup_n \Delta_n\right) = \sum_n E(\Delta_n)$$
(convergence in strong operator topology)
4. **Multiplicative**: $E(\Delta_1 \cap \Delta_2) = E(\Delta_1)E(\Delta_2)$

**Remark**: The multiplicative property follows from the others: projections that commute and sum to a projection must multiply.

**Definition (Spectral Measure)**

For $A$ self-adjoint, a **spectral measure** for $A$ is a PVM $E$ on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$ such that:
$$\boxed{A = \int_{-\infty}^{\infty} \lambda \, dE_\lambda}$$

### 3. The Spectral Family

**Definition (Spectral Family / Resolution of the Identity)**

A **spectral family** is a family $\{E_\lambda\}_{\lambda \in \mathbb{R}}$ of orthogonal projections satisfying:

1. **Monotonicity**: $\lambda < \mu \implies E_\lambda \leq E_\mu$ (i.e., $E_\lambda E_\mu = E_\lambda$)
2. **Right-continuity**: $E_\lambda = \lim_{\mu \to \lambda^+} E_\mu$ (in strong operator topology)
3. **Limits at infinity**: $\lim_{\lambda \to -\infty} E_\lambda = 0$, $\lim_{\lambda \to +\infty} E_\lambda = I$

**Connection to PVM**: Given spectral family $\{E_\lambda\}$, define:
$$E((a, b]) = E_b - E_a$$

This extends to a PVM on Borel sets.

### 4. The Spectral Theorem

**Theorem (Spectral Theorem for Bounded Self-Adjoint Operators)**

Let $A$ be a bounded self-adjoint operator on a Hilbert space $\mathcal{H}$. Then there exists a unique spectral measure $E$ supported on $\sigma(A) \subseteq [m, M]$ (where $m = \inf\sigma(A)$, $M = \sup\sigma(A)$) such that:

$$\boxed{A = \int_m^M \lambda \, dE_\lambda}$$

In terms of the spectral family:
$$\langle Ax, y \rangle = \int_m^M \lambda \, d\langle E_\lambda x, y \rangle$$

**Properties of the Spectral Measure**:

1. **Support is the spectrum**: $\text{supp}(E) = \sigma(A)$
2. **Point spectrum**: $\lambda \in \sigma_p(A) \iff E(\{\lambda\}) \neq 0$
   - In this case, $E(\{\lambda\})$ projects onto the eigenspace
3. **Continuous spectrum**: $\lambda \in \sigma_c(A) \iff E(\{\lambda\}) = 0$ but $E((a,b)) \neq 0$ for all $(a,b) \ni \lambda$
4. **Norm bound**: $\|A\| = \max\{|m|, |M|\}$

### 5. Constructing the Spectral Measure

**Method 1: Via Resolvent (Stieltjes Inversion)**

For $\lambda \in \mathbb{R}$:
$$\langle E_\lambda x, x \rangle = \lim_{\epsilon \to 0^+} \frac{1}{\pi} \int_{-\infty}^\lambda \text{Im}\langle R_{\mu + i\epsilon}(A)x, x \rangle \, d\mu$$

**Method 2: Via Polynomial Approximation**

For continuous $f$ on $\sigma(A)$:
$$\langle f(A)x, y \rangle = \int_{\sigma(A)} f(\lambda) \, d\langle E_\lambda x, y \rangle$$

Start with polynomials $p(A)$, extend by density.

**Method 3: Via Gelfand Transform (Abstract)**

Use the C*-algebra isomorphism $C(\sigma(A)) \cong C^*(A, I)$.

### 6. Integration with Respect to Spectral Measures

**Definition (Spectral Integral)**

For bounded Borel function $f: \sigma(A) \to \mathbb{C}$:
$$\boxed{f(A) = \int_{\sigma(A)} f(\lambda) \, dE_\lambda}$$

**Properties**:
1. **Linearity**: $(\alpha f + \beta g)(A) = \alpha f(A) + \beta g(A)$
2. **Multiplicativity**: $(fg)(A) = f(A)g(A)$
3. **Adjoint**: $\overline{f}(A) = f(A)^*$
4. **Norm**: $\|f(A)\| = \sup_{\lambda \in \sigma(A)} |f(\lambda)|$

**Key Examples**:
- $f(\lambda) = \lambda^n \implies f(A) = A^n$
- $f(\lambda) = e^{i\lambda t} \implies f(A) = e^{iAt}$
- $f(\lambda) = \chi_{[a,b]}(\lambda) \implies f(A) = E([a,b]) = E_b - E_a$

### 7. Spectral Decomposition Examples

**Example 1: Multiplication Operator**

$(Mf)(x) = xf(x)$ on $L^2[0,1]$.

The spectral family is:
$$(E_\lambda f)(x) = \chi_{[0, \min(\lambda, 1)]}(x) f(x)$$

Verification:
$$\int_0^1 \lambda \, dE_\lambda f = \int_0^1 \lambda \chi_{[0,\lambda]}(x) dE_\lambda \cdot f = xf(x) = Mf$$

**Example 2: Diagonal Matrix**

$D = \text{diag}(d_1, d_2, \ldots, d_n)$

Spectral measure:
$$E(\Delta) = \sum_{i: d_i \in \Delta} |e_i\rangle\langle e_i|$$

Spectral decomposition:
$$D = \sum_{i=1}^n d_i |e_i\rangle\langle e_i| = \int \lambda \, dE_\lambda$$

The integral reduces to a sum over atoms of the measure.

### 8. Quantum Mechanics Connection

The spectral theorem is the mathematical backbone of quantum measurement theory.

**Observable Operators and Measurement**

| Mathematical Concept | Quantum Interpretation |
|---------------------|------------------------|
| Spectral measure $E$ | Determines measurement statistics |
| $E(\Delta)$ for Borel set $\Delta$ | Projector for "outcome in $\Delta$" |
| $\langle\psi|E(\Delta)|\psi\rangle$ | Probability of outcome in $\Delta$ |
| $E(\{a\})$ for point spectrum | Projector onto eigenspace |

**Born Rule (General Form)**

For observable $A$ with spectral measure $E$, measuring on state $|\psi\rangle$:
$$\boxed{P(A \in \Delta) = \langle\psi|E(\Delta)|\psi\rangle = \|E(\Delta)\psi\|^2}$$

**Example: Position Measurement**

Position operator $\hat{x}$ on $L^2(\mathbb{R})$: $(\hat{x}\psi)(x) = x\psi(x)$

Spectral measure: $(E_x \psi)(y) = \chi_{(-\infty, x]}(y)\psi(y)$

Probability of finding particle in $[a, b]$:
$$P(a \leq x \leq b) = \langle\psi|(E_b - E_a)|\psi\rangle = \int_a^b |\psi(x)|^2 \, dx$$

This is exactly the Born rule!

**Post-Measurement State**

After measuring $A$ with outcome in $\Delta$:
$$|\psi\rangle \mapsto \frac{E(\Delta)|\psi\rangle}{\|E(\Delta)|\psi\|\|}$$

**Expectation Values**

$$\langle A \rangle_\psi = \langle\psi|A|\psi\rangle = \int_{\sigma(A)} \lambda \, d\langle E_\lambda\psi, \psi\rangle = \int_{\sigma(A)} \lambda \, d\|E_\lambda\psi\|^2$$

**Variance**

$$\text{Var}(A)_\psi = \langle A^2 \rangle - \langle A \rangle^2 = \int_{\sigma(A)} (\lambda - \langle A \rangle)^2 \, d\|E_\lambda\psi\|^2$$

---

## Worked Examples

### Example 1: Spectral Family for a 2x2 Matrix

**Problem**: Find the spectral family for $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$.

**Solution**:

**Step 1**: Find eigenvalues and eigenvectors.

$\det(A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda-1)(\lambda-3) = 0$

Eigenvalues: $\lambda_1 = 1$, $\lambda_2 = 3$.

Eigenvectors:
- $\lambda_1 = 1$: $(A - I)v = 0 \Rightarrow v_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$
- $\lambda_2 = 3$: $(A - 3I)v = 0 \Rightarrow v_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$

**Step 2**: Construct spectral projections.

$$P_1 = |v_1\rangle\langle v_1| = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$

$$P_2 = |v_2\rangle\langle v_2| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**Step 3**: Write the spectral family.

$$E_\lambda = \begin{cases}
0 & \lambda < 1 \\
P_1 & 1 \leq \lambda < 3 \\
P_1 + P_2 = I & \lambda \geq 3
\end{cases}$$

**Step 4**: Verify the spectral integral.

$$A = \int_{-\infty}^{\infty} \lambda \, dE_\lambda = 1 \cdot (E_1 - E_{1^-}) + 3 \cdot (E_3 - E_{3^-})$$
$$= 1 \cdot P_1 + 3 \cdot P_2 = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} + \frac{3}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

### Example 2: Spectral Measure of Multiplication Operator

**Problem**: For $(Mf)(x) = xf(x)$ on $L^2[0,1]$, find the spectral measure and verify $\sigma(M) = [0,1]$.

**Solution**:

**Step 1**: Propose the spectral family.

For $\lambda \in \mathbb{R}$, define:
$$(E_\lambda f)(x) = \chi_{(-\infty, \lambda] \cap [0,1]}(x) f(x) = \chi_{[0, \min(1, \max(0,\lambda))]}(x) f(x)$$

So:
- $E_\lambda = 0$ for $\lambda < 0$
- $(E_\lambda f)(x) = \chi_{[0,\lambda]}(x)f(x)$ for $0 \leq \lambda \leq 1$
- $E_\lambda = I$ for $\lambda > 1$

**Step 2**: Verify properties of spectral family.

- **Projections**: $(E_\lambda)^2 = E_\lambda$ and $E_\lambda^* = E_\lambda$ ✓
- **Monotonicity**: $\lambda < \mu \Rightarrow E_\lambda E_\mu = E_\lambda$ ✓
- **Right-continuity**: Holds pointwise ✓
- **Limits**: $E_{-\infty} = 0$, $E_{+\infty} = I$ ✓

**Step 3**: Verify the spectral integral.

$$\left(\int_0^1 \lambda \, dE_\lambda f\right)(x) = \int_0^1 \lambda \cdot \delta(x - \lambda) f(x) \, d\lambda = xf(x) = (Mf)(x)$$

More rigorously, for $g = \int_0^1 \lambda \, dE_\lambda f$:
$$\langle g, h \rangle = \int_0^1 \lambda \, d\langle E_\lambda f, h\rangle = \int_0^1 \lambda \int_0^\lambda f(x)\overline{h(x)}\, dx \, \text{(Stieltjes)}$$
$$= \int_0^1 xf(x)\overline{h(x)}\, dx = \langle Mf, h \rangle$$

**Step 4**: Identify the spectrum.

$\sigma(M) = \text{supp}(E) = [0,1]$

Since $E(\{\lambda\}) = 0$ for all $\lambda$ (characteristic functions of single points are null in $L^2$), the entire spectrum is **continuous**: $\sigma_c(M) = [0,1]$.

### Example 3: Quantum Measurement Probabilities

**Problem**: A spin-1/2 particle is in state $|\psi\rangle = \frac{1}{\sqrt{3}}|{\uparrow}\rangle + \sqrt{\frac{2}{3}}|{\downarrow}\rangle$. Using the spectral measure of $S_z$, find the probability distribution for measuring $S_z$.

**Solution**:

**Step 1**: Write $S_z$ in spectral form.

$$S_z = \frac{\hbar}{2}(|{\uparrow}\rangle\langle{\uparrow}| - |{\downarrow}\rangle\langle{\downarrow}|)$$

Spectral measure:
- $E(\{+\hbar/2\}) = |{\uparrow}\rangle\langle{\uparrow}|$
- $E(\{-\hbar/2\}) = |{\downarrow}\rangle\langle{\downarrow}|$

**Step 2**: Compute probabilities using Born rule.

$$P(S_z = +\hbar/2) = \langle\psi|E(\{+\hbar/2\})|\psi\rangle = |\langle{\uparrow}|\psi\rangle|^2 = \left|\frac{1}{\sqrt{3}}\right|^2 = \frac{1}{3}$$

$$P(S_z = -\hbar/2) = \langle\psi|E(\{-\hbar/2\})|\psi\rangle = |\langle{\downarrow}|\psi\rangle|^2 = \frac{2}{3}$$

**Step 3**: Compute expectation value.

$$\langle S_z \rangle = \int \lambda \, d\langle E_\lambda\psi, \psi\rangle = \frac{\hbar}{2} \cdot \frac{1}{3} + \left(-\frac{\hbar}{2}\right) \cdot \frac{2}{3} = -\frac{\hbar}{6}$$

Alternatively: $\langle S_z \rangle = \langle\psi|S_z|\psi\rangle = -\frac{\hbar}{6}$

**Step 4**: Post-measurement states.

If outcome is $+\hbar/2$: $|\psi\rangle \to |{\uparrow}\rangle$

If outcome is $-\hbar/2$: $|\psi\rangle \to |{\downarrow}\rangle$

---

## Practice Problems

### Level 1: Direct Application

1. **3x3 Matrix**: Find the spectral family for $A = \text{diag}(1, 2, 2)$.

2. **Projection Operators**: Show that for projection $P$, the spectral measure is $E(\{0\}) = I - P$ and $E(\{1\}) = P$.

3. **Spectral Integral**: If $A$ has spectral family $\{E_\lambda\}$, express $A^2$ as a spectral integral.

### Level 2: Intermediate

4. **Position-Momentum Commutator**: Using spectral representations, explain why $[\hat{x}, \hat{p}] = i\hbar$ cannot have both operators bounded.

5. **Spectral Measure of a Shift**: For the bilateral shift $U$ on $\ell^2(\mathbb{Z})$, $U$ is unitary with $\sigma(U) = \mathbb{T}$ (unit circle). Describe its spectral measure on $\mathbb{T}$.

6. **Stieltjes Inversion**: Use the formula $\langle E_\lambda x, x \rangle = \lim_{\epsilon \to 0} \frac{1}{\pi}\int_{-\infty}^\lambda \text{Im}\langle R_{\mu+i\epsilon}x, x\rangle d\mu$ to find the spectral measure of $M$ on $L^2[0,1]$.

### Level 3: Challenging

7. **Spectral Theorem Uniqueness**: Prove that the spectral measure in the spectral theorem is unique.

8. **Joint Spectral Measure**: For commuting self-adjoint $A$ and $B$, define a joint spectral measure $E_{A,B}$ on $\mathbb{R}^2$.

9. **Continuous Spectrum Characterization**: Prove that $\lambda$ is in the continuous spectrum of self-adjoint $A$ if and only if $E(\{\lambda\}) = 0$ but $E((a,b)) \neq 0$ for all intervals $(a,b)$ containing $\lambda$.

---

## Computational Lab: Spectral Measures

```python
"""
Day 248 Computational Lab: Spectral Theorem for Bounded Self-Adjoint
Exploring spectral measures and projection-valued measures
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import quad
from typing import Tuple, Callable, List

np.random.seed(42)

# =============================================================================
# Part 1: Spectral Family Construction for Matrices
# =============================================================================

print("="*70)
print("PART 1: SPECTRAL FAMILY FOR MATRICES")
print("="*70)

def compute_spectral_family(A: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Compute the spectral family {E_lambda} for a self-adjoint matrix.

    Returns:
        eigenvalues: sorted eigenvalues
        projections: list of spectral projections P_i = |e_i><e_i|
        E_cumulative: E_lambda = sum_{lambda_i <= lambda} P_i
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Sort eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute spectral projections
    n = A.shape[0]
    projections = []
    for i in range(n):
        e_i = eigenvectors[:, i:i+1]
        P_i = e_i @ e_i.conj().T
        projections.append(P_i)

    return eigenvalues, projections, eigenvectors

# Example: 4x4 self-adjoint matrix
A = np.array([[4, 1, 0, 0],
              [1, 3, 1, 0],
              [0, 1, 2, 1],
              [0, 0, 1, 1]], dtype=float)

eigenvalues, projections, eigenvectors = compute_spectral_family(A)

print(f"\nMatrix A:\n{A}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nSpectral projections:")
for i, (ev, P) in enumerate(zip(eigenvalues, projections)):
    print(f"\nlambda_{i} = {ev:.4f}")
    print(f"P_{i} = |e_{i}><e_{i}| =\n{np.round(P, 4)}")

# Verify: A = sum lambda_i P_i
A_reconstructed = sum(ev * P for ev, P in zip(eigenvalues, projections))
print(f"\nReconstruction A = sum lambda_i P_i:")
print(f"||A - A_reconstructed|| = {np.linalg.norm(A - A_reconstructed):.2e}")

# Verify: P_i are projections and orthogonal
print("\nVerification of projection properties:")
for i, P in enumerate(projections):
    print(f"P_{i}^2 = P_{i}: {np.allclose(P @ P, P)}, "
          f"P_{i}^* = P_{i}: {np.allclose(P, P.conj().T)}")

print("\nOrthogonality P_i P_j = 0 for i != j:")
for i in range(len(projections)):
    for j in range(i+1, len(projections)):
        prod = projections[i] @ projections[j]
        print(f"||P_{i} P_{j}|| = {np.linalg.norm(prod):.2e}")

# =============================================================================
# Part 2: Spectral Family Visualization
# =============================================================================

print("\n" + "="*70)
print("PART 2: VISUALIZING THE SPECTRAL FAMILY")
print("="*70)

def evaluate_spectral_family(eigenvalues: np.ndarray,
                             projections: List[np.ndarray],
                             lambda_val: float) -> np.ndarray:
    """Evaluate E_lambda = sum_{lambda_i <= lambda} P_i."""
    n = projections[0].shape[0]
    E = np.zeros((n, n), dtype=complex)
    for ev, P in zip(eigenvalues, projections):
        if ev <= lambda_val:
            E += P
    return E

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot spectral projections as heatmaps
for i, (ev, P) in enumerate(zip(eigenvalues[:4], projections[:4])):
    ax = axes[i // 2, i % 2]
    im = ax.imshow(np.real(P), cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title(f'$P_{i+1}$ for $\\lambda_{i+1} = {ev:.3f}$')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('spectral_projections_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: spectral_projections_heatmap.png")

# Plot the spectral family E_lambda for different lambda values
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

lambda_test = [eigenvalues[0] - 0.5, eigenvalues[1] + 0.1,
               eigenvalues[2] + 0.1, eigenvalues[3] + 0.5]

for ax, lam in zip(axes, lambda_test):
    E_lam = evaluate_spectral_family(eigenvalues, projections, lam)
    im = ax.imshow(np.real(E_lam), cmap='RdBu', vmin=-0.5, vmax=1.5)
    rank = int(np.round(np.trace(E_lam)))
    ax.set_title(f'$E_{{\\lambda}}$ for $\\lambda = {lam:.2f}$\nrank = {rank}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('spectral_family_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: spectral_family_evolution.png")

# =============================================================================
# Part 3: Spectral Measure for Multiplication Operator
# =============================================================================

print("\n" + "="*70)
print("PART 3: DISCRETIZED MULTIPLICATION OPERATOR")
print("="*70)

def discretized_multiplication(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize (Mf)(x) = xf(x) on L^2[0,1].
    Returns matrix and grid points.
    """
    x = np.linspace(0.5/n, 1 - 0.5/n, n)
    M = np.diag(x)
    return M, x

n = 50
M, x_grid = discretized_multiplication(n)
eigenvalues_M, projections_M, eigenvectors_M = compute_spectral_family(M)

print(f"Multiplication operator M on {n} grid points")
print(f"Eigenvalues (first 10): {eigenvalues_M[:10]}")
print(f"Expected: {x_grid[:10]}")

# The spectral measure is essentially Lebesgue measure on [0,1]
# E([a,b]) corresponds to multiplying by indicator function

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot: eigenvalue distribution
ax1 = axes[0]
ax1.hist(eigenvalues_M, bins=20, density=True, alpha=0.7, edgecolor='black')
ax1.axhline(y=1, color='r', linestyle='--', label='Uniform density')
ax1.set_xlabel('$\\lambda$')
ax1.set_ylabel('Density')
ax1.set_title('Eigenvalue Distribution of M\n(approaches uniform on [0,1])')
ax1.legend()

# Plot: cumulative distribution (spectral measure)
ax2 = axes[1]
lambdas = np.linspace(-0.1, 1.1, 100)
cumulative = [np.sum(eigenvalues_M <= lam) / n for lam in lambdas]
ax2.plot(lambdas, cumulative, 'b-', linewidth=2)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='$\\lambda$ (theory)')
ax2.set_xlabel('$\\lambda$')
ax2.set_ylabel('$\\langle E_\\lambda \\mathbf{1}, \\mathbf{1}\\rangle / n$')
ax2.set_title('Spectral Distribution Function')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot: eigenfunctions (should be delta-like)
ax3 = axes[2]
for i in [0, n//4, n//2, 3*n//4, n-1]:
    ax3.plot(x_grid, np.abs(eigenvectors_M[:, i])**2 * n,
             label=f'$|e_{{{i}}}|^2$, $\\lambda$={eigenvalues_M[i]:.2f}')
ax3.set_xlabel('x')
ax3.set_ylabel('$|e_i(x)|^2 \\cdot n$')
ax3.set_title('Eigenfunctions (approximate delta functions)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiplication_operator_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: multiplication_operator_spectrum.png")

# =============================================================================
# Part 4: Spectral Integral Computation
# =============================================================================

print("\n" + "="*70)
print("PART 4: COMPUTING f(A) VIA SPECTRAL INTEGRAL")
print("="*70)

def spectral_integral(eigenvalues: np.ndarray,
                      projections: List[np.ndarray],
                      f: Callable) -> np.ndarray:
    """
    Compute f(A) = int f(lambda) dE_lambda = sum f(lambda_i) P_i.
    """
    n = projections[0].shape[0]
    result = np.zeros((n, n), dtype=complex)
    for ev, P in zip(eigenvalues, projections):
        result += f(ev) * P
    return result

# Test various functions
A_test = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues_test, projections_test, _ = compute_spectral_family(A_test)

print(f"\nTest matrix A:\n{A_test}")
print(f"Eigenvalues: {eigenvalues_test}")

# Function 1: f(x) = x^2
print("\n--- f(lambda) = lambda^2 ---")
A_squared_spectral = spectral_integral(eigenvalues_test, projections_test, lambda x: x**2)
A_squared_direct = A_test @ A_test
print(f"Via spectral integral:\n{np.real(A_squared_spectral)}")
print(f"Direct A^2:\n{A_squared_direct}")
print(f"Match: {np.allclose(A_squared_spectral, A_squared_direct)}")

# Function 2: f(x) = exp(x)
print("\n--- f(lambda) = exp(lambda) ---")
exp_A_spectral = spectral_integral(eigenvalues_test, projections_test, np.exp)
exp_A_direct = linalg.expm(A_test)
print(f"Via spectral integral:\n{np.real(exp_A_spectral)}")
print(f"Via matrix exponential:\n{exp_A_direct}")
print(f"Match: {np.allclose(exp_A_spectral, exp_A_direct)}")

# Function 3: f(x) = sqrt(x) (only for positive eigenvalues)
print("\n--- f(lambda) = sqrt(lambda) ---")
sqrt_A_spectral = spectral_integral(eigenvalues_test, projections_test, np.sqrt)
sqrt_A_direct = linalg.sqrtm(A_test)
print(f"Via spectral integral:\n{np.real(sqrt_A_spectral)}")
print(f"Via matrix sqrt:\n{np.real(sqrt_A_direct)}")
print(f"Match: {np.allclose(sqrt_A_spectral, sqrt_A_direct)}")

# Function 4: Indicator function (spectral projection)
print("\n--- f(lambda) = chi_{[1.5, 2.5]}(lambda) ---")
chi_proj = spectral_integral(eigenvalues_test, projections_test,
                             lambda x: 1.0 if 1.5 <= x <= 2.5 else 0.0)
print(f"E([1.5, 2.5]):\n{np.real(chi_proj)}")
print(f"This is projection onto eigenspace of lambda=1 (only eigenvalue in range)")

# =============================================================================
# Part 5: Quantum Measurement Simulation
# =============================================================================

print("\n" + "="*70)
print("PART 5: QUANTUM MEASUREMENT WITH SPECTRAL PROJECTIONS")
print("="*70)

def simulate_measurement(A: np.ndarray, psi: np.ndarray,
                         num_measurements: int = 10000) -> dict:
    """
    Simulate measurements of observable A on state psi.

    Returns:
        Dictionary with theoretical and simulated statistics
    """
    # Get spectral decomposition
    eigenvalues, projections, eigenvectors = compute_spectral_family(A)

    # Normalize state
    psi = psi / np.linalg.norm(psi)

    # Compute theoretical probabilities using Born rule
    probs = np.array([np.real(psi.conj().T @ P @ psi).item()
                      for P in projections])

    # Simulate measurements
    outcomes = np.random.choice(eigenvalues, size=num_measurements, p=probs)

    # Statistics
    theoretical_mean = np.sum(eigenvalues * probs)
    simulated_mean = np.mean(outcomes)
    theoretical_var = np.sum(eigenvalues**2 * probs) - theoretical_mean**2
    simulated_var = np.var(outcomes)

    return {
        'eigenvalues': eigenvalues,
        'probabilities': probs,
        'outcomes': outcomes,
        'theoretical_mean': theoretical_mean,
        'simulated_mean': simulated_mean,
        'theoretical_var': theoretical_var,
        'simulated_var': simulated_var
    }

# Spin-1 system: Jz measurement
Jz = np.diag([1, 0, -1]).astype(float)

# Superposition state
psi = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)

results = simulate_measurement(Jz, psi, num_measurements=10000)

print("\nSpin-1 Jz Measurement Simulation:")
print(f"Initial state: (|+1> + |0> + |-1>)/sqrt(3)")
print(f"\nTheoretical probabilities:")
for ev, p in zip(results['eigenvalues'], results['probabilities']):
    print(f"  P(Jz = {ev:+.0f}) = {p:.4f}")

print(f"\nStatistics (10000 measurements):")
print(f"  Theoretical <Jz> = {results['theoretical_mean']:.4f}")
print(f"  Simulated <Jz>   = {results['simulated_mean']:.4f}")
print(f"  Theoretical Var  = {results['theoretical_var']:.4f}")
print(f"  Simulated Var    = {results['simulated_var']:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of measurement outcomes
ax1 = axes[0]
for i, ev in enumerate(results['eigenvalues']):
    count = np.sum(results['outcomes'] == ev)
    ax1.bar(ev, count/len(results['outcomes']), width=0.3,
            label=f'm={ev:.0f}', alpha=0.7)
    ax1.axhline(y=results['probabilities'][i], xmin=(ev+2)/4, xmax=(ev+2.3)/4,
                color='red', linewidth=2)
ax1.set_xlabel('Measurement outcome (Jz)')
ax1.set_ylabel('Frequency')
ax1.set_title('Measurement Statistics\n(bars=simulated, lines=theoretical)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Time evolution of measurement statistics
ax2 = axes[1]
n_measurements = np.arange(1, len(results['outcomes'])+1)
running_mean = np.cumsum(results['outcomes']) / n_measurements
ax2.plot(n_measurements, running_mean, 'b-', alpha=0.7)
ax2.axhline(y=results['theoretical_mean'], color='r', linestyle='--',
            label=f'Theory: {results["theoretical_mean"]:.4f}')
ax2.set_xlabel('Number of measurements')
ax2.set_ylabel('Running mean')
ax2.set_title('Convergence of Sample Mean to Expectation')
ax2.legend()
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quantum_measurement_simulation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: quantum_measurement_simulation.png")

# =============================================================================
# Part 6: Spectral Resolution Visualization
# =============================================================================

print("\n" + "="*70)
print("PART 6: RESOLUTION OF THE IDENTITY")
print("="*70)

# Create a larger self-adjoint matrix
n = 20
H = np.random.randn(n, n)
H = (H + H.T) / 2  # Make symmetric

eigenvalues_H, projections_H, _ = compute_spectral_family(H)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Spectral family E_lambda as step function
ax1 = axes[0]
lambda_range = np.linspace(eigenvalues_H[0] - 1, eigenvalues_H[-1] + 1, 500)

# Compute rank of E_lambda (which equals trace for projections)
ranks = []
for lam in lambda_range:
    E = evaluate_spectral_family(eigenvalues_H, projections_H, lam)
    ranks.append(np.real(np.trace(E)))

ax1.plot(lambda_range, ranks, 'b-', linewidth=2)
ax1.scatter(eigenvalues_H, range(1, n+1), color='red', s=50, zorder=5,
            label='Eigenvalues')
ax1.set_xlabel('$\\lambda$')
ax1.set_ylabel('rank($E_\\lambda$)')
ax1.set_title('Spectral Family: rank($E_\\lambda$) = #{eigenvalues $\\leq \\lambda$}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Spectral measure dE_lambda
ax2 = axes[1]
ax2.stem(eigenvalues_H, np.ones(n), basefmt=" ")
ax2.set_xlabel('$\\lambda$')
ax2.set_ylabel('Spectral measure')
ax2.set_title('Spectral Measure (point masses at eigenvalues)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resolution_of_identity.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: resolution_of_identity.png")

# Verify resolution of identity: sum P_i = I
identity_check = sum(projections_H)
print(f"\nResolution of identity: sum P_i = I")
print(f"||sum P_i - I|| = {np.linalg.norm(identity_check - np.eye(n)):.2e}")

print("\n" + "="*70)
print("LAB COMPLETE")
print("="*70)
print("""
Key takeaways:
1. Spectral family E_lambda is a step function increasing at eigenvalues
2. Spectral projections P_i are orthogonal and sum to identity
3. f(A) = sum f(lambda_i) P_i computes any function of A
4. Quantum measurement probabilities come from spectral projections
5. The Born rule P(lambda) = ||P_lambda psi||^2 determines statistics
6. Spectral theorem generalizes matrix diagonalization to operators
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Spectral integral | $A = \int_{\sigma(A)} \lambda \, dE_\lambda$ |
| Spectral family | $E_\lambda = \sum_{\lambda_n \leq \lambda} P_n$ |
| Projection formula | $E(\Delta) = E_b - E_a$ for $\Delta = (a, b]$ |
| Born rule | $P(A \in \Delta) = \|E(\Delta)\psi\|^2$ |
| Expectation | $\langle A \rangle = \int \lambda \, d\|E_\lambda\psi\|^2$ |
| Spectral function | $f(A) = \int f(\lambda) \, dE_\lambda$ |

### Main Takeaways

1. **Spectral measures generalize eigenvector decomposition**: The spectral theorem holds for all bounded self-adjoint operators, not just compact ones.

2. **Projection-valued measures**: The spectral measure $E$ assigns orthogonal projections to Borel sets.

3. **Integration replaces summation**: $A = \int \lambda \, dE_\lambda$ generalizes $A = \sum \lambda_n P_n$.

4. **Continuous spectrum**: When there are no eigenvalues, the spectral measure is "continuous" (no atoms).

5. **Quantum measurement**: Spectral projections implement the measurement postulate and Born rule.

6. **Functional calculus preview**: $f(A) = \int f(\lambda) \, dE_\lambda$ allows defining any function of $A$.

---

## Daily Checklist

- [ ] I can define projection-valued measures and their properties
- [ ] I understand the spectral family (resolution of the identity)
- [ ] I can state the spectral theorem for bounded self-adjoint operators
- [ ] I can construct spectral families for matrices and multiplication operators
- [ ] I understand how spectral projections relate to quantum measurement
- [ ] I can compute expectation values using spectral integrals
- [ ] I can explain the difference between point and continuous spectrum
- [ ] I completed the computational lab

---

## Preview: Day 249

Tomorrow we develop the **functional calculus** in detail: given $A$ self-adjoint and function $f$, we construct $f(A)$ rigorously:
$$f(A) = \int_{\sigma(A)} f(\lambda) \, dE_\lambda$$

This allows us to define $e^{iAt}$, $\sqrt{A}$, $\log(A)$, and any reasonable function of an operator—essential for quantum mechanics where we need things like $e^{-iHt/\hbar}$ for time evolution.

---

*"The spectral theorem is to operators what the fundamental theorem of algebra is to polynomials—it tells us that everything can be decomposed into simple pieces."*
— Tosio Kato
