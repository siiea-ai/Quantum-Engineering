# Day 229: Compactness in Metric Spaces

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning I** | 2 hours | Sequential compactness and total boundedness |
| **Morning II** | 1.5 hours | Compactness in ℝⁿ: Heine-Borel theorem |
| **Afternoon I** | 2 hours | Arzelà-Ascoli theorem for function spaces |
| **Afternoon II** | 1.5 hours | Worked examples and practice problems |
| **Evening** | 1 hour | Computational lab: compact sets and approximation |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** sequential compactness and total boundedness
2. **Prove** the equivalence of compactness conditions in metric spaces
3. **Apply** the Heine-Borel theorem in ℝⁿ
4. **State and use** the Arzelà-Ascoli theorem
5. **Understand** compact operators and their spectral properties
6. **Connect** compactness to approximation methods in quantum mechanics

---

## 1. Compactness: Three Equivalent Definitions

### Definition 1: Sequential Compactness

A metric space (X, d) is **sequentially compact** if every sequence in X has a convergent subsequence:

$$\boxed{\forall (x_n) \subseteq X, \exists (x_{n_k}) \text{ and } x \in X : x_{n_k} \to x}$$

### Definition 2: Cover Compactness

A metric space (X, d) is **compact** if every open cover has a finite subcover:

$$\boxed{X = \bigcup_{\alpha \in A} U_\alpha \text{ (open)} \implies X = \bigcup_{i=1}^{n} U_{\alpha_i} \text{ for some finite } \{\alpha_1, ..., \alpha_n\}}$$

### Definition 3: Total Boundedness + Completeness

A metric space is **totally bounded** if for every ε > 0, X can be covered by finitely many ε-balls:

$$\forall \varepsilon > 0, \exists x_1, ..., x_n \in X : X = \bigcup_{i=1}^{n} B(x_i, \varepsilon)$$

**Theorem (Equivalence in Metric Spaces):**
For a metric space (X, d), the following are equivalent:
1. X is sequentially compact
2. X is compact (cover definition)
3. X is complete and totally bounded

---

## 2. Properties of Compact Sets

### Basic Properties

**Theorem 1:** Every compact subset of a metric space is closed and bounded.

**Proof:**
*Bounded:* Fix any point p. The balls B(p, n) for n ∈ ℕ cover X. By compactness, finitely many suffice, so X ⊆ B(p, N) for some N.

*Closed:* Let xₙ → x with xₙ ∈ K (compact). By sequential compactness, some subsequence converges in K. But the full sequence converges to x, so x is the unique limit, hence x ∈ K. ∎

**Theorem 2:** Closed subsets of compact sets are compact.

**Theorem 3:** Continuous images of compact sets are compact.

**Theorem 4:** A continuous function on a compact set attains its maximum and minimum.

### Compactness and Uniform Continuity

**Theorem:** A continuous function f: K → Y from a compact metric space K to a metric space Y is uniformly continuous.

**Proof:** Suppose not. Then ∃ε > 0 and sequences (xₙ), (yₙ) with d(xₙ, yₙ) < 1/n but d(f(xₙ), f(yₙ)) ≥ ε.

By compactness, xₙₖ → x for some subsequence. Then yₙₖ → x as well.

By continuity: f(xₙₖ) → f(x) and f(yₙₖ) → f(x).

So d(f(xₙₖ), f(yₙₖ)) → 0, contradicting d(f(xₙₖ), f(yₙₖ)) ≥ ε. ∎

---

## 3. The Heine-Borel Theorem

### Statement

**Theorem (Heine-Borel):** A subset K ⊆ ℝⁿ is compact if and only if K is closed and bounded.

### Why This Fails in Infinite Dimensions

In infinite-dimensional spaces, closed and bounded does NOT imply compact!

**Example:** The closed unit ball in l²:
$$B = \{x \in \ell^2 : \|x\|_2 \leq 1\}$$

Consider the sequence eₙ = (0, ..., 0, 1, 0, ...) (1 in n-th position). This sequence:
- Is in B (each ||eₙ|| = 1)
- Has no convergent subsequence (||eₘ - eₙ|| = √2 for m ≠ n)

So B is closed and bounded but NOT compact.

### Totally Bounded in ℝⁿ ⟺ Bounded

In ℝⁿ, a set is totally bounded iff it's bounded. Combined with the fact that closed subsets of complete spaces are complete:

Closed + Bounded (in ℝⁿ) → Complete + Totally Bounded → Compact

---

## 4. The Arzelà-Ascoli Theorem

### The Problem

What does compactness look like in function spaces? The Arzelà-Ascoli theorem answers this for C[a, b].

### Definitions

Let F ⊆ C[a, b].

**Uniform Boundedness:** F is **uniformly bounded** if there exists M > 0 such that:
$$\sup_{f \in F} \|f\|_\infty \leq M$$

**Equicontinuity:** F is **equicontinuous** if for every ε > 0, there exists δ > 0 such that:
$$|x - y| < \delta \implies |f(x) - f(y)| < \varepsilon \quad \forall f \in F$$

The same δ works for ALL functions in F.

### Theorem (Arzelà-Ascoli)

A subset F ⊆ C[a, b] (with the sup norm) is relatively compact (its closure is compact) if and only if F is:
1. **Uniformly bounded**, and
2. **Equicontinuous**

### Proof Sketch

**(⟹)** If F is relatively compact:
- Uniform boundedness: The continuous function f ↦ ||f||_∞ attains its max on the compact closure.
- Equicontinuity: By total boundedness, cover F by finitely many ε/3-balls. The functions in these balls form a finite set, which is equicontinuous.

**(⟸)** If F is uniformly bounded and equicontinuous:
- Show F is totally bounded using a diagonal argument
- Completeness of C[a,b] then gives compactness

---

## 5. Compact Operators

### Definition

Let X and Y be Banach spaces. A linear operator T: X → Y is **compact** if the image of the unit ball has compact closure:

$$\boxed{\overline{T(B_X)} \text{ is compact in } Y}$$

where $B_X = \{x \in X : ||x|| \leq 1\}$.

Equivalently: T maps bounded sequences to sequences with convergent subsequences.

### Examples of Compact Operators

**Example 1: Integral Operators**

On L²[a, b], define:
$$(Kf)(x) = \int_a^b k(x, y) f(y) \, dy$$

If k ∈ L²([a,b] × [a,b]), then K is compact (Hilbert-Schmidt operator).

**Example 2: Multiplication by Decay**

On l², define T(x₁, x₂, x₃, ...) = (x₁, x₂/2, x₃/3, ...).

T is compact because Tₙ → T where Tₙ has finite rank.

**Example 3: The Identity is NOT Compact**

On infinite-dimensional spaces, I: X → X is never compact (the unit ball isn't compact).

### Properties of Compact Operators

1. Compact + Bounded (always satisfied)
2. Sum of compact operators is compact
3. Composition: (compact) ∘ (bounded) and (bounded) ∘ (compact) are compact
4. Limits of compact operators (in operator norm) are compact
5. Compact operators form a closed ideal in B(X, Y)

---

## Quantum Mechanics Connection: Compact Operators and Spectra

### The Spectral Theorem for Compact Operators

**Theorem:** Let T be a compact self-adjoint operator on a Hilbert space H. Then:

1. T has at most countably many eigenvalues
2. Non-zero eigenvalues have finite multiplicity
3. The only accumulation point of eigenvalues is 0
4. H has an orthonormal basis of eigenvectors

### Physical Interpretation

Many quantum mechanical operators are compact:

**Example: Bound States**

The resolvent (H - zI)⁻¹ for a Hamiltonian H with bound states is often compact. This explains:
- Discrete energy levels
- Finite degeneracy (usually)
- Eigenstates form a complete basis

**Example: Integral Kernels**

The Green's function G(x, x') defines a compact integral operator:
$$(Gf)(x) = \int G(x, x') f(x') dx'$$

### Approximation by Finite-Rank Operators

A key feature of compact operators: they can be approximated by finite-rank operators!

If T is compact and self-adjoint:
$$T = \sum_{n=1}^{\infty} \lambda_n |e_n\rangle\langle e_n|$$

The partial sums (finite rank) converge to T in operator norm.

This is why numerical methods work: truncate to finite dimensions.

---

## Worked Examples

### Example 1: Compactness in ℝ²

**Problem:** Determine which of the following subsets of ℝ² are compact:
(a) {(x, y) : x² + y² ≤ 1}
(b) {(x, y) : x² + y² < 1}
(c) {(x, y) : x² + y² = 1}
(d) {(x, 0) : x ∈ ℝ}

**Solution:**

(a) **Compact.** Closed (includes boundary) and bounded (contained in square [-1,1]²).

(b) **Not compact.** Bounded but not closed. The sequence (1 - 1/n, 0) → (1, 0) which is not in the set.

(c) **Compact.** Closed (complement of open disk union exterior) and bounded.

(d) **Not compact.** Closed but unbounded (extends to infinity).

### Example 2: Arzelà-Ascoli Application

**Problem:** Let F = {fₙ : fₙ(x) = sin(nx)/n, n ∈ ℕ} ⊆ C[0, 2π]. Is F relatively compact?

**Solution:**

**Check uniform boundedness:**
$$\|f_n\|_\infty = \sup_x \left|\frac{\sin(nx)}{n}\right| \leq \frac{1}{n} \leq 1$$

Yes, uniformly bounded by 1.

**Check equicontinuity:**
$$|f_n(x) - f_n(y)| = \left|\frac{\sin(nx) - \sin(ny)}{n}\right| \leq \frac{|nx - ny|}{n} = |x - y|$$

So δ = ε works for all n. Equicontinuous!

**Conclusion:** By Arzelà-Ascoli, F is relatively compact. ∎

### Example 3: Non-Compact Operator

**Problem:** Show that the identity operator on l² is not compact.

**Solution:**

Consider the unit ball B = {x : ||x||₂ ≤ 1}.

I(B) = B itself. We need to show B is not compact.

Consider the sequence eₙ = (0, ..., 0, 1, 0, ...) (1 in n-th position).
- Each eₙ ∈ B
- ||eₘ - eₙ||² = 2 for m ≠ n

No subsequence can be Cauchy, hence no convergent subsequence.

Therefore B is not compact, so I is not compact. ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Heine-Borel:** Is [0, 1) × [0, 1] compact in ℝ²? Explain.

2. **Bounded but not compact:** Give an example of a bounded sequence in C[0, 1] (sup norm) with no convergent subsequence.

3. **Equicontinuity:** Is the family {fₙ(x) = xⁿ, n ∈ ℕ} equicontinuous on [0, 1]?

### Level 2: Intermediate

4. **Total boundedness:** Show that the set {x ∈ l² : ∑n|xₙ|² ≤ 1 and |xₙ| ≤ 1/n} is totally bounded in l².

5. **Compact operator:** Prove that T: l² → l² defined by T(x₁, x₂, ...) = (0, x₁, x₂/2, x₃/3, ...) is compact.

6. **Arzelà-Ascoli failure:** Find a uniformly bounded family in C[0, 1] that is NOT equicontinuous.

### Level 3: Challenging

7. **Compactness characterization:** Prove that a metric space X is compact iff every continuous real-valued function on X is bounded and attains its bounds.

8. **Compact self-adjoint:** Let T be compact and self-adjoint on a Hilbert space. Prove that if T has no zero eigenvalue, then T⁻¹ exists but is unbounded.

9. **Rellich-Kondrachov:** In what sense is the embedding H¹(Ω) ↪ L²(Ω) compact for bounded Ω ⊆ ℝⁿ? (State the theorem and explain its significance.)

---

## Computational Lab: Compactness and Approximation

```python
"""
Day 229: Compactness - Computational Lab
Exploring compact sets, Arzelà-Ascoli, and compact operators
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# =============================================================================
# Part 1: Non-Compactness of Unit Ball in Infinite Dimensions
# =============================================================================

def unit_ball_not_compact():
    """Demonstrate that the unit ball in l^2 is not compact."""

    # Simulate the sequence e_n in l^2 (truncated to first 50 components)
    dim = 50
    n_points = 30

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Visualize e_n vectors (first few components)
    ax1 = axes[0]
    for n in range(1, min(11, n_points)):
        e_n = np.zeros(dim)
        e_n[n-1] = 1
        ax1.plot(range(1, 16), e_n[:15], 'o-', markersize=8,
                 alpha=0.7, label=f'$e_{n}$' if n <= 5 else '')

    ax1.set_xlabel('Component index', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Standard Basis Vectors in $\\ell^2$ (first 15 components)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Distance matrix
    ax2 = axes[1]
    distances = np.full((n_points, n_points), np.sqrt(2))
    np.fill_diagonal(distances, 0)

    im = ax2.imshow(distances, cmap='viridis')
    plt.colorbar(im, ax=ax2, label='$\\|e_m - e_n\\|_2$')
    ax2.set_xlabel('m', fontsize=12)
    ax2.set_ylabel('n', fontsize=12)
    ax2.set_title(f'Distance Matrix: All off-diagonal = $\\sqrt{{2}}$', fontsize=14)

    plt.tight_layout()
    plt.savefig('unit_ball_not_compact.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Unit ball in l² is NOT compact:")
    print(f"  All ||e_m - e_n|| = √2 ≈ {np.sqrt(2):.4f} for m ≠ n")
    print("  No subsequence is Cauchy, hence no convergent subsequence")

# =============================================================================
# Part 2: Arzelà-Ascoli - Equicontinuity
# =============================================================================

def arzela_ascoli_demo():
    """Demonstrate equicontinuity and its failure."""

    x = np.linspace(0, 1, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Equicontinuous family f_n(x) = sin(nx)/n
    ax1 = axes[0]
    for n in [1, 2, 3, 5, 10, 20]:
        f_n = np.sin(n * np.pi * x) / n
        ax1.plot(x, f_n, linewidth=2, label=f'n = {n}')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('$f_n(x) = \\sin(n\\pi x)/n$', fontsize=12)
    ax1.set_title('Equicontinuous Family (Relatively Compact)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Non-equicontinuous family g_n(x) = sin(nx) (bounded but not equicontinuous)
    ax2 = axes[1]
    for n in [1, 2, 5, 10, 20]:
        g_n = np.sin(n * np.pi * x)
        ax2.plot(x, g_n, linewidth=2, label=f'n = {n}')

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('$g_n(x) = \\sin(n\\pi x)$', fontsize=12)
    ax2.set_title('NOT Equicontinuous (Not Relatively Compact)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('arzela_ascoli.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nArzelà-Ascoli Comparison:")
    print("-" * 50)
    print("Left: f_n(x) = sin(nπx)/n")
    print("  - Uniformly bounded: |f_n| ≤ 1/n ≤ 1")
    print("  - Equicontinuous: |f_n(x) - f_n(y)| ≤ |x - y|")
    print("  → RELATIVELY COMPACT by Arzelà-Ascoli")
    print()
    print("Right: g_n(x) = sin(nπx)")
    print("  - Uniformly bounded: |g_n| ≤ 1")
    print("  - NOT equicontinuous: oscillations get faster")
    print("  → NOT relatively compact")

# =============================================================================
# Part 3: Compact Operators - Spectral Properties
# =============================================================================

def compact_operator_spectrum():
    """Demonstrate spectral properties of compact operators."""

    # Create a compact operator: integral operator with kernel
    N = 100
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Kernel k(x, y) = min(x, y) (Green's function for -d²/dx² with Dirichlet BC)
    K = np.minimum(x[:, None], x[None, :]) * dx

    # Make it symmetric
    K = (K + K.T) / 2

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Sort by magnitude

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Eigenvalues
    ax1 = axes[0]
    ax1.semilogy(range(1, len(eigenvalues) + 1), eigenvalues, 'bo', markersize=4)
    ax1.set_xlabel('Eigenvalue index n', fontsize=12)
    ax1.set_ylabel('$|\\lambda_n|$', fontsize=12)
    ax1.set_title('Eigenvalues of Compact Integral Operator', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Theoretical: λ_n ∝ 1/n² for this kernel
    n_theory = np.arange(1, N + 1)
    lambda_theory = eigenvalues[0] * (1 / n_theory**2)
    ax1.semilogy(n_theory, lambda_theory, 'r--', linewidth=2,
                 label=r'Theory: $\lambda_n \propto 1/n^2$')
    ax1.legend()

    # Right: Finite-rank approximation error
    ax2 = axes[1]

    # SVD for low-rank approximation
    U, S, Vt = svd(K)

    ranks = range(1, 51)
    errors = []

    for r in ranks:
        K_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        error = np.linalg.norm(K - K_approx, 'fro')
        errors.append(error)

    ax2.semilogy(ranks, errors, 'go-', markersize=4, linewidth=1)
    ax2.set_xlabel('Rank of approximation', fontsize=12)
    ax2.set_ylabel('Frobenius norm error', fontsize=12)
    ax2.set_title('Approximation by Finite-Rank Operators', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('compact_operator_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nCompact Operator Spectral Properties:")
    print("-" * 50)
    print(f"Largest 5 eigenvalues: {eigenvalues[:5]}")
    print(f"Eigenvalues → 0 (only accumulation point)")
    print(f"Rank-10 approximation error: {errors[9]:.6f}")
    print(f"Rank-30 approximation error: {errors[29]:.6f}")

# =============================================================================
# Part 4: Totally Bounded Sets
# =============================================================================

def total_boundedness_demo():
    """Visualize total boundedness vs just bounded."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Totally bounded (compact) set in R^2 - closed disk
    ax1 = axes[0]

    # The set
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.fill(np.cos(theta), np.sin(theta), alpha=0.2, color='blue')
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)

    # ε-balls covering
    epsilon = 0.4
    # Grid of centers
    n_grid = int(np.ceil(2 / epsilon))
    for i in range(-n_grid, n_grid + 1):
        for j in range(-n_grid, n_grid + 1):
            cx, cy = i * epsilon * 0.7, j * epsilon * 0.7
            if cx**2 + cy**2 <= (1 + epsilon)**2:
                circle = plt.Circle((cx, cy), epsilon, fill=False,
                                   color='red', linestyle='--', alpha=0.5)
                ax1.add_patch(circle)
                ax1.plot(cx, cy, 'r.', markersize=5)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title(f'Totally Bounded: Finitely many ε-balls (ε={epsilon})', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Right: Not totally bounded in l^2 - unit sphere
    ax2 = axes[1]

    # Project the standard basis vectors onto first 2 coordinates (all at origin except...)
    # Instead, visualize the concept differently
    # Show that for ε < √2, we need infinitely many balls

    ax2.text(0.5, 0.7, 'Unit sphere in $\\ell^2$:', fontsize=14,
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.55, '$S = \\{x \\in \\ell^2 : \\|x\\| = 1\\}$', fontsize=14,
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.4, 'For $\\varepsilon < \\sqrt{2}$:', fontsize=12,
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.25, 'Need infinitely many balls!', fontsize=14,
             ha='center', transform=ax2.transAxes, color='red')
    ax2.text(0.5, 0.1, '(Each $e_n$ needs its own ball)', fontsize=11,
             ha='center', transform=ax2.transAxes)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('NOT Totally Bounded: Infinite dimensions', fontsize=14)

    plt.tight_layout()
    plt.savefig('total_boundedness.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 5: Compact Operator Approximation in Quantum Mechanics
# =============================================================================

def quantum_compact_operator():
    """Approximate a quantum mechanical compact operator."""

    # Harmonic oscillator resolvent (compact for E not an eigenvalue)
    # Eigenvalues: E_n = (n + 1/2)ℏω, eigenfunctions: |n⟩

    # We'll work in the energy eigenbasis
    N = 50  # Number of basis states
    hbar_omega = 1.0  # Set ℏω = 1

    E = -0.3  # Not an eigenvalue (they're at 0.5, 1.5, 2.5, ...)

    # Resolvent (H - EI)^{-1} in energy eigenbasis
    # Diagonal with entries 1/(E_n - E)
    E_n = (np.arange(N) + 0.5) * hbar_omega
    resolvent_diag = 1 / (E_n - E)

    # As a matrix
    R = np.diag(resolvent_diag)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Singular values of resolvent
    ax1 = axes[0]
    singular_values = np.abs(resolvent_diag)  # For diagonal, these are |diagonal elements|
    singular_values_sorted = np.sort(singular_values)[::-1]

    ax1.semilogy(range(1, N + 1), singular_values_sorted, 'bo-', markersize=4)
    ax1.set_xlabel('Index n', fontsize=12)
    ax1.set_ylabel('Singular value', fontsize=12)
    ax1.set_title('Singular Values of $(H - EI)^{-1}$', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Right: Approximation error vs rank
    ax2 = axes[1]

    ranks = range(1, N)
    errors = []

    for r in ranks:
        # Keep only r largest singular values
        approx_diag = np.zeros(N)
        top_r_indices = np.argsort(np.abs(resolvent_diag))[::-1][:r]
        approx_diag[top_r_indices] = resolvent_diag[top_r_indices]

        R_approx = np.diag(approx_diag)
        error = np.linalg.norm(R - R_approx, 2)  # Operator norm
        errors.append(error)

    ax2.semilogy(list(ranks), errors, 'go-', markersize=4, linewidth=1)
    ax2.set_xlabel('Rank of approximation', fontsize=12)
    ax2.set_ylabel('Operator norm error', fontsize=12)
    ax2.set_title('Resolvent Approximation Error', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantum_compact.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nQuantum Mechanical Compact Operator:")
    print("-" * 50)
    print(f"Resolvent (H - EI)^(-1) with E = {E}")
    print(f"Harmonic oscillator eigenvalues: E_n = (n + 1/2)ℏω")
    print(f"Top 5 singular values: {singular_values_sorted[:5]}")
    print(f"Rank-5 approximation error: {errors[4]:.6f}")
    print(f"Rank-20 approximation error: {errors[19]:.6f}")

# =============================================================================
# Run All Visualizations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 229: Compactness - Computational Lab")
    print("=" * 60)

    print("\n1. Unit ball in l² is not compact...")
    unit_ball_not_compact()

    print("\n2. Arzelà-Ascoli: equicontinuity...")
    arzela_ascoli_demo()

    print("\n3. Compact operator spectrum...")
    compact_operator_spectrum()

    print("\n4. Total boundedness visualization...")
    total_boundedness_demo()

    print("\n5. Quantum mechanical compact operator...")
    quantum_compact_operator()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas and Concepts

| Concept | Definition/Property |
|---------|---------------------|
| Sequential compactness | Every sequence has a convergent subsequence |
| Cover compactness | Every open cover has a finite subcover |
| Total boundedness | Finitely many ε-balls cover X for all ε > 0 |
| Equivalence | Compact ⟺ Complete + Totally bounded (in metric spaces) |
| Heine-Borel | In ℝⁿ: Compact ⟺ Closed + Bounded |
| Equicontinuity | Same δ works for all f ∈ F |
| Arzelà-Ascoli | Relatively compact ⟺ Uniformly bounded + Equicontinuous |
| Compact operator | $\overline{T(B_X)}$ is compact |

### Compactness Comparison

| Property | ℝⁿ | Infinite-dim |
|----------|-----|--------------|
| Closed + Bounded ⟹ Compact | Yes | No |
| Unit ball compact | Yes | No |
| Bounded ⟺ Totally bounded | Yes | No |

### Main Takeaways

1. **Compactness generalizes finiteness** to infinite settings
2. **In ℝⁿ, Heine-Borel** gives a simple criterion
3. **In function spaces, Arzelà-Ascoli** is the key theorem
4. **Compact operators** have nice spectral properties (discrete spectrum)
5. **Quantum mechanics** relies on compact operators for discrete spectra

---

## Daily Checklist

- [ ] I can define sequential compactness and total boundedness
- [ ] I understand why closed+bounded ≠ compact in infinite dimensions
- [ ] I can apply the Arzelà-Ascoli theorem
- [ ] I know what compact operators are and their spectral properties
- [ ] I understand the connection to quantum mechanical operators
- [ ] I completed the computational lab

---

## Preview: Day 230

Tomorrow we study the **completion of metric spaces**:
- How to "fill in the gaps" in incomplete spaces
- Construction via Cauchy sequences
- The completion of ℚ is ℝ
- Completing pre-Hilbert spaces to get Hilbert spaces
