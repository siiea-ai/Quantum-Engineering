# Day 227: Completeness and Cauchy Sequences

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning I** | 2 hours | Cauchy sequences and the completeness definition |
| **Morning II** | 1.5 hours | Complete spaces: ℝ, l^p, L^p |
| **Afternoon I** | 2 hours | Proving completeness and Banach spaces |
| **Afternoon II** | 1.5 hours | Worked examples and practice problems |
| **Evening** | 1 hour | Computational lab: approximating limits |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** Cauchy sequences and characterize their relationship to convergence
2. **Prove** that ℝⁿ, l^p, and C[a,b] are complete metric spaces
3. **Understand** why completeness is essential for quantum mechanics
4. **Apply** the completeness criterion to verify convergence
5. **Distinguish** between complete and incomplete spaces
6. **Connect** Banach spaces to operator theory in quantum mechanics

---

## 1. Cauchy Sequences

### The Fundamental Idea

In an incomplete space, a sequence can "try to converge" but have no limit point to converge to. Cauchy sequences capture the idea of "trying to converge" without reference to the limit.

### Definition: Cauchy Sequence

A sequence (xₙ) in a metric space (X, d) is **Cauchy** if:

$$\boxed{\forall \varepsilon > 0, \exists N \in \mathbb{N} : m, n \geq N \implies d(x_m, x_n) < \varepsilon}$$

In words: the terms get arbitrarily close to each other eventually.

### Key Properties

**Theorem 1:** Every convergent sequence is Cauchy.

**Proof:** Let xₙ → x. Given ε > 0, choose N such that n ≥ N implies d(xₙ, x) < ε/2. For m, n ≥ N:
$$d(x_m, x_n) \leq d(x_m, x) + d(x, x_n) < \varepsilon/2 + \varepsilon/2 = \varepsilon$$
∎

**Theorem 2:** Every Cauchy sequence is bounded.

**Proof:** Let (xₙ) be Cauchy. Choose N such that m, n ≥ N implies d(xₘ, xₙ) < 1. Then all terms with n ≥ N lie in B(x_N, 1). The finite set {x₁, ..., x_{N-1}} is bounded, so the entire sequence is bounded. ∎

**Warning:** The converse of Theorem 1 is FALSE in general! A Cauchy sequence need not converge if the limit point is "missing" from the space.

---

## 2. Complete Metric Spaces

### Definition

A metric space (X, d) is **complete** if every Cauchy sequence in X converges to a point in X.

$$\boxed{\text{Complete: } (x_n) \text{ Cauchy} \implies \exists x \in X : x_n \to x}$$

### Examples of Complete Spaces

1. **ℝ with |·|**: The Completeness Axiom of ℝ
2. **ℝⁿ with any p-norm**: Product of complete spaces
3. **l^p for 1 ≤ p ≤ ∞**: Sequence spaces
4. **L^p(X, μ)**: The Riesz-Fischer theorem
5. **C[a,b] with ||·||_∞**: Uniform limit of continuous functions is continuous

### Examples of Incomplete Spaces

1. **ℚ with |·|**: The sequence 1, 1.4, 1.41, 1.414, ... is Cauchy but converges to √2 ∉ ℚ

2. **(0, 1) with |·|**: The sequence 1/n is Cauchy but converges to 0 ∉ (0, 1)

3. **C[a,b] with L² norm**: Consider a sequence of continuous functions approaching a step function

---

## 3. Completeness of Key Spaces

### Theorem: ℝⁿ is Complete

**Proof:** Let (x^{(k)}) be a Cauchy sequence in ℝⁿ, where x^{(k)} = (x₁^{(k)}, ..., xₙ^{(k)}).

For each coordinate i, the sequence (xᵢ^{(k)})_{k=1}^∞ is Cauchy in ℝ:
$$|x_i^{(m)} - x_i^{(n)}| \leq \|x^{(m)} - x^{(n)}\|_2 < \varepsilon$$

By completeness of ℝ, each coordinate converges: xᵢ^{(k)} → xᵢ.

Define x = (x₁, ..., xₙ). Then x^{(k)} → x in ℝⁿ:
$$\|x^{(k)} - x\|_2 = \sqrt{\sum_i |x_i^{(k)} - x_i|^2} \to 0$$
∎

### Theorem: l^p is Complete (1 ≤ p < ∞)

**Proof Sketch:** Let (x^{(k)}) be Cauchy in l^p, where x^{(k)} = (x_n^{(k)})_{n=1}^∞.

1. **Coordinate convergence:** For each n, (x_n^{(k)})_k is Cauchy in ℂ, so x_n^{(k)} → x_n for some x_n ∈ ℂ.

2. **Define the candidate limit:** x = (x₁, x₂, x₃, ...).

3. **Show x ∈ l^p:** For any N and k large enough:
$$\sum_{n=1}^{N} |x_n|^p = \lim_{m \to \infty} \sum_{n=1}^{N} |x_n^{(m)}|^p \leq \limsup_{m} \|x^{(m)}\|_p^p < \infty$$

4. **Show x^{(k)} → x in l^p:** Given ε > 0, choose K such that k, l ≥ K implies ||x^{(k)} - x^{(l)}||_p < ε. Taking l → ∞:
$$\|x^{(k)} - x\|_p \leq \varepsilon$$
∎

### Theorem: C[a,b] with ||·||_∞ is Complete

**Proof:** This is the classical result that the uniform limit of continuous functions is continuous.

Let (fₖ) be Cauchy in (C[a,b], ||·||_∞).

1. **Pointwise convergence:** For each x ∈ [a,b], (fₖ(x)) is Cauchy in ℝ (since |fₖ(x) - fₗ(x)| ≤ ||fₖ - fₗ||_∞). Define f(x) = lim fₖ(x).

2. **Uniform convergence:** Given ε > 0, choose K such that k, l ≥ K implies ||fₖ - fₗ||_∞ < ε. Then for all x:
$$|f_k(x) - f(x)| = \lim_{l \to \infty} |f_k(x) - f_l(x)| \leq \varepsilon$$
So ||fₖ - f||_∞ ≤ ε for k ≥ K.

3. **Continuity of f:** Let x₀ ∈ [a,b]. For any ε > 0:
$$|f(x) - f(x_0)| \leq |f(x) - f_k(x)| + |f_k(x) - f_k(x_0)| + |f_k(x_0) - f(x_0)|$$
$$< \varepsilon/3 + \varepsilon/3 + \varepsilon/3 = \varepsilon$$
for k large and x close to x₀ (using continuity of fₖ).
∎

---

## 4. Banach Spaces

### Definition

A **Banach space** is a complete normed vector space.

That is, (X, ||·||) is a Banach space if:
1. X is a vector space
2. ||·||: X → [0, ∞) is a norm
3. X is complete with respect to the metric d(x, y) = ||x - y||

### Examples of Banach Spaces

| Space | Norm | Banach? |
|-------|------|---------|
| ℝⁿ | ||x||_p for any p ∈ [1, ∞] | Yes |
| l^p | ||x||_p = (∑\|xₙ\|^p)^{1/p} | Yes |
| L^p(X, μ) | ||f||_p | Yes (Riesz-Fischer) |
| C[a,b] | ||f||_∞ | Yes |
| C[a,b] | ||f||_2 | **No!** |

### The L² Space: A Hilbert Space

L²(X, μ) is not just a Banach space—it's a **Hilbert space** (complete inner product space):

$$\langle f, g \rangle = \int_X f(x) \overline{g(x)} \, d\mu(x)$$

$$\|f\|_2 = \sqrt{\langle f, f \rangle}$$

**Theorem (Riesz-Fischer):** L²(X, μ) is complete.

This is the mathematical foundation of quantum mechanics!

---

## Quantum Mechanics Connection: Why Completeness Matters

### The State Space Must Be Complete

In quantum mechanics, the state space is L²(ℝ³). Completeness is not optional—it's essential:

**Physical Requirement 1: Limits of Physical States Are Physical**

If a sequence of normalized states ψₙ approaches some function ψ (in the sense that ||ψₙ - ψ||₂ → 0), then ψ must also be a valid quantum state.

**Physical Requirement 2: Time Evolution**

The Schrödinger equation generates a continuous family of states ψ(t). Completeness ensures:
$$\psi(t) = \lim_{\Delta t \to 0} \psi(t + \Delta t) \in L^2$$

**Physical Requirement 3: Spectral Decomposition**

Expressing a state in an energy eigenbasis:
$$|\psi\rangle = \sum_{n=0}^{\infty} c_n |E_n\rangle$$

This infinite sum is really a limit of partial sums. Completeness guarantees the limit exists in L².

### Example: The Particle in a Box

The energy eigenstates of a particle in [0, L] are:
$$\phi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)$$

Any state can be written as:
$$\psi(x) = \sum_{n=1}^{\infty} c_n \phi_n(x)$$

This series converges in L² (Parseval's theorem), and completeness guarantees the limit is in L².

### Incomplete Spaces Break QM

If we worked in C[0, L] with the L² norm (incomplete!), we could have:
- A sequence of smooth wavefunctions
- Approaching a discontinuous step function
- Which is NOT in C[0, L]

The "limit state" would be undefined, breaking the theory.

---

## Worked Examples

### Example 1: A Non-Convergent Cauchy Sequence

**Problem:** Show that in (ℚ, |·|), the sequence defined by:
$$x_1 = 1, \quad x_{n+1} = \frac{x_n}{2} + \frac{1}{x_n}$$
is Cauchy but does not converge in ℚ.

**Solution:**

This is the Babylonian method for computing √2. One can show:
- x_{n+1}² - 2 = (x_n² - 2)²/(4x_n²) (error squares each step)
- |x_{n+1} - √2| ≤ |x_n - √2|²/(2√2) (quadratic convergence)

The sequence satisfies:
$$|x_m - x_n| < \varepsilon \text{ for } m, n \geq N$$

But lim x_n = √2 ∉ ℚ.

This demonstrates ℚ is incomplete. ∎

### Example 2: Completeness of C[0,1] with Supremum Norm

**Problem:** Show that the sequence fₙ(x) = xⁿ is NOT Cauchy in (C[0,1], ||·||_∞).

**Solution:**

For m > n:
$$\|f_m - f_n\|_\infty = \sup_{x \in [0,1]} |x^m - x^n| = \sup_{x \in [0,1]} x^n|x^{m-n} - 1|$$

At x = (1/2)^{1/n}:
$$f_n(x) = 1/2, \quad f_m(x) = (1/2)^{m/n} \to 0 \text{ as } m \to \infty$$

So ||fₘ - fₙ||_∞ ≥ |1/2 - (1/2)^{m/n}| → 1/2 as m → ∞.

The sequence is NOT Cauchy (terms don't get arbitrarily close). ∎

### Example 3: A Cauchy Sequence in l²

**Problem:** Show that x^{(k)} = (1, 1/2, 1/3, ..., 1/k, 0, 0, ...) is Cauchy in l².

**Solution:**

For m > k:
$$\|x^{(m)} - x^{(k)}\|_2^2 = \sum_{n=k+1}^{m} \frac{1}{n^2}$$

As k → ∞:
$$\sum_{n=k+1}^{\infty} \frac{1}{n^2} \to 0$$
(tail of convergent series)

Given ε > 0, choose K such that ∑_{n>K} 1/n² < ε². Then for m > k ≥ K:
$$\|x^{(m)} - x^{(k)}\|_2 < \varepsilon$$

The sequence converges to x = (1, 1/2, 1/3, ...) ∈ l² since ∑ 1/n² = π²/6 < ∞. ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Cauchy check:** Is the sequence xₙ = (-1)ⁿ Cauchy in ℝ? Is it convergent?

2. **Completeness of subsets:** Is the closed interval [0, 1] complete as a subspace of ℝ? Is the open interval (0, 1)?

3. **l^∞ sequence:** Show that eₙ = (0, ..., 0, 1, 0, ...) (1 in n-th position) is NOT Cauchy in l^∞.

### Level 2: Intermediate

4. **Completing ℚ:** Explain why the completion of ℚ (with the usual metric) is ℝ.

5. **Cauchy in function space:** In (C[0,1], ||·||_∞), define fₙ(x) = sin(nx)/n. Is this sequence Cauchy? Does it converge?

6. **Closed subspaces:** Prove that a closed subspace of a complete metric space is complete.

### Level 3: Challenging

7. **C[0,1] incomplete in L² norm:** Find a sequence of continuous functions on [0,1] that is Cauchy in the L² norm but whose L² limit is not continuous.

8. **Absolutely convergent series:** In a Banach space X, prove that if ∑ ||xₙ|| < ∞, then ∑ xₙ converges.

9. **Nested spheres:** In a complete metric space, if Bₙ are closed balls with B₁ ⊇ B₂ ⊇ ... and radius(Bₙ) → 0, prove ∩ Bₙ is a single point.

---

## Computational Lab: Approximating Limits

```python
"""
Day 227: Completeness and Cauchy Sequences - Computational Lab
Exploring Cauchy sequences and completeness numerically
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# =============================================================================
# Part 1: Babylonian Method - Cauchy in Q, Convergent in R
# =============================================================================

def babylonian_method():
    """Demonstrate the Babylonian method for sqrt(2)."""

    # Babylonian iteration: x_{n+1} = (x_n + 2/x_n) / 2
    x = [1.0]
    for _ in range(10):
        x.append((x[-1] + 2/x[-1]) / 2)

    x = np.array(x)
    errors = np.abs(x - np.sqrt(2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sequence values
    ax1 = axes[0]
    ax1.plot(range(len(x)), x, 'bo-', markersize=10, linewidth=2)
    ax1.axhline(y=np.sqrt(2), color='r', linestyle='--', linewidth=2,
                label=f'$\\sqrt{{2}} \\approx {np.sqrt(2):.10f}$')
    ax1.set_xlabel('Iteration n', fontsize=12)
    ax1.set_ylabel('$x_n$', fontsize=12)
    ax1.set_title('Babylonian Method: $x_{n+1} = (x_n + 2/x_n)/2$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: convergence rate (log scale)
    ax2 = axes[1]
    ax2.semilogy(range(len(errors)), errors, 'go-', markersize=10, linewidth=2)
    ax2.set_xlabel('Iteration n', fontsize=12)
    ax2.set_ylabel('$|x_n - \\sqrt{2}|$', fontsize=12)
    ax2.set_title('Quadratic Convergence (Error)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('babylonian_sqrt2.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Babylonian Method for sqrt(2):")
    print("-" * 50)
    for i, (xi, ei) in enumerate(zip(x, errors)):
        print(f"x_{i} = {xi:.15f}, error = {ei:.2e}")

# =============================================================================
# Part 2: Cauchy vs Convergent Sequences
# =============================================================================

def cauchy_examples():
    """Compare Cauchy and non-Cauchy sequences."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Cauchy and convergent: 1/n in R
    ax1 = axes[0, 0]
    n = np.arange(1, 51)
    seq1 = 1/n

    ax1.plot(n, seq1, 'bo-', markersize=6, linewidth=1)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Limit = 0')
    ax1.set_xlabel('n', fontsize=12)
    ax1.set_ylabel('$x_n = 1/n$', fontsize=12)
    ax1.set_title('Cauchy & Convergent in $\\mathbb{R}$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Cauchy but not convergent in (0,1): 1/n
    ax2 = axes[0, 1]
    ax2.plot(n, seq1, 'ro-', markersize=6, linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2,
                label='Limit 0 $\\notin (0,1)$')
    ax2.axhspan(-0.1, 0, alpha=0.3, color='gray', label='Outside (0,1)')
    ax2.set_xlabel('n', fontsize=12)
    ax2.set_ylabel('$x_n = 1/n$', fontsize=12)
    ax2.set_title('Cauchy but NOT Convergent in $(0,1)$', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    # (c) Not Cauchy: (-1)^n
    ax3 = axes[1, 0]
    seq3 = (-1)**n

    ax3.plot(n, seq3, 'go-', markersize=6, linewidth=1)
    ax3.axhline(y=1, color='b', linestyle=':', alpha=0.5)
    ax3.axhline(y=-1, color='b', linestyle=':', alpha=0.5)
    ax3.set_xlabel('n', fontsize=12)
    ax3.set_ylabel('$x_n = (-1)^n$', fontsize=12)
    ax3.set_title('NOT Cauchy: Terms stay distance 2 apart', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # (d) Cauchy distance plot
    ax4 = axes[1, 1]

    # For sequence 1/n, plot d(x_m, x_n) as heatmap
    N = 30
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distances[i, j] = abs(1/(i+1) - 1/(j+1))

    im = ax4.imshow(distances, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax4, label='$|x_m - x_n|$')
    ax4.set_xlabel('m', fontsize=12)
    ax4.set_ylabel('n', fontsize=12)
    ax4.set_title('Cauchy: $|x_m - x_n| \\to 0$ as $m,n \\to \\infty$', fontsize=14)

    plt.tight_layout()
    plt.savefig('cauchy_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 3: Incompleteness of C[0,1] with L^2 Norm
# =============================================================================

def incomplete_C01_L2():
    """Show C[0,1] is incomplete in L^2 norm."""

    # Sequence of continuous functions approaching step function
    def f_n(x, n):
        """Smooth approximation to step at x=0.5."""
        return 1 / (1 + np.exp(-n * (x - 0.5)))

    x = np.linspace(0, 1, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: the sequence
    ax1 = axes[0]

    for n in [2, 5, 10, 20, 50]:
        ax1.plot(x, f_n(x, n), label=f'n = {n}', linewidth=2)

    # The discontinuous limit
    step = np.where(x < 0.5, 0, 1)
    ax1.plot(x, step, 'k--', linewidth=3, label='Limit (step)')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('$f_n(x)$', fontsize=12)
    ax1.set_title('Continuous $f_n \\to$ Discontinuous Step', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Cauchy in L^2
    ax2 = axes[1]

    n_values = np.arange(1, 51)
    L2_distances = []

    for n in n_values:
        # Compute ||f_n - step||_L^2
        diff_sq = (f_n(x, n) - step)**2
        L2_dist = np.sqrt(np.trapz(diff_sq, x))
        L2_distances.append(L2_dist)

    ax2.semilogy(n_values, L2_distances, 'b-', linewidth=2)
    ax2.set_xlabel('n', fontsize=12)
    ax2.set_ylabel('$\\|f_n - \\mathrm{step}\\|_{L^2}$', fontsize=12)
    ax2.set_title('$L^2$ Distance to Step Function', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('incomplete_C01.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("C[0,1] with L² norm is INCOMPLETE:")
    print("The limit of continuous functions need not be continuous!")

# =============================================================================
# Part 4: l^2 Completeness Demonstration
# =============================================================================

def l2_completeness():
    """Demonstrate completeness of l^2 with a Cauchy sequence."""

    # Sequence: x^(k) = (1, 1/2, 1/3, ..., 1/k, 0, 0, ...)
    # Limit: x = (1, 1/2, 1/3, ...)

    max_k = 50
    max_n = 100  # Show first 100 components

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Partial sequences and limit
    ax1 = axes[0]

    n = np.arange(1, max_n + 1)
    limit = 1 / n

    for k in [5, 10, 20, 50]:
        x_k = np.where(n <= k, 1/n, 0)
        ax1.plot(n, x_k, 'o-', markersize=3, alpha=0.7, label=f'$x^{{({k})}}$')

    ax1.plot(n, limit, 'k-', linewidth=2, label='Limit $x$')
    ax1.set_xlabel('Component n', fontsize=12)
    ax1.set_ylabel('$x_n$', fontsize=12)
    ax1.set_title('Cauchy Sequence in $\\ell^2$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 50)

    # Right: Convergence to limit
    ax2 = axes[1]

    k_values = np.arange(1, max_k + 1)
    distances = []

    for k in k_values:
        # ||x^(k) - x||_2^2 = sum_{n>k} 1/n^2
        d_sq = np.sum(1/np.arange(k+1, 10000)**2)  # Approximate tail sum
        distances.append(np.sqrt(d_sq))

    ax2.semilogy(k_values, distances, 'b-', linewidth=2)
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('$\\|x^{(k)} - x\\|_{\\ell^2}$', fontsize=12)
    ax2.set_title('Convergence Rate in $\\ell^2$', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('l2_completeness.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Limit x = (1, 1/2, 1/3, ...) is in l² since sum(1/n²) = π²/6 ≈ {np.pi**2/6:.4f}")

# =============================================================================
# Part 5: Quantum State Expansion Convergence
# =============================================================================

def quantum_state_expansion():
    """Show convergence of quantum state expansion in L^2."""

    # Particle in a box: expand a Gaussian in energy eigenstates
    L = 1.0  # Box length

    def psi_target(x):
        """Target state: Gaussian centered in box."""
        return np.exp(-20 * (x - 0.5)**2)

    def phi_n(x, n):
        """Energy eigenstate n."""
        return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

    def compute_coefficient(n):
        """Compute expansion coefficient c_n = <phi_n | psi>."""
        result, _ = quad(lambda x: phi_n(x, n) * psi_target(x), 0, L)
        return result

    x = np.linspace(0, L, 1000)
    target = psi_target(x)

    # Normalize target
    norm = np.sqrt(np.trapz(target**2, x))
    target = target / norm

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Partial sums
    ax1 = axes[0]
    ax1.plot(x, target, 'k-', linewidth=3, label='Target $\\psi(x)$')

    n_terms_list = [1, 3, 5, 10, 20]
    for n_max in n_terms_list:
        coeffs = [compute_coefficient(n) / norm for n in range(1, n_max + 1)]
        partial_sum = np.zeros_like(x)
        for n, c in enumerate(coeffs, 1):
            partial_sum += c * phi_n(x, n)

        ax1.plot(x, partial_sum, '--', linewidth=1.5, alpha=0.8,
                label=f'{n_max} terms')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('$\\psi(x)$', fontsize=12)
    ax1.set_title('Partial Sums in Energy Eigenbasis', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: L^2 error vs number of terms
    ax2 = axes[1]

    n_max_values = range(1, 51)
    L2_errors = []

    for n_max in n_max_values:
        coeffs = [compute_coefficient(n) / norm for n in range(1, n_max + 1)]
        partial_sum = np.zeros_like(x)
        for n, c in enumerate(coeffs, 1):
            partial_sum += c * phi_n(x, n)

        error = np.sqrt(np.trapz((partial_sum - target)**2, x))
        L2_errors.append(error)

    ax2.semilogy(list(n_max_values), L2_errors, 'b-', linewidth=2)
    ax2.set_xlabel('Number of terms', fontsize=12)
    ax2.set_ylabel('$L^2$ error', fontsize=12)
    ax2.set_title('Convergence of Eigenfunction Expansion', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantum_expansion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Completeness of L² guarantees this expansion converges!")

# =============================================================================
# Run All Visualizations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 227: Completeness and Cauchy Sequences - Lab")
    print("=" * 60)

    print("\n1. Babylonian method (Cauchy in Q)...")
    babylonian_method()

    print("\n2. Cauchy vs non-Cauchy sequences...")
    cauchy_examples()

    print("\n3. C[0,1] is incomplete in L² norm...")
    incomplete_C01_L2()

    print("\n4. l² completeness demonstration...")
    l2_completeness()

    print("\n5. Quantum state expansion convergence...")
    quantum_state_expansion()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula/Definition |
|---------|---------------------|
| Cauchy sequence | $\forall \varepsilon > 0, \exists N: m,n \geq N \Rightarrow d(x_m, x_n) < \varepsilon$ |
| Complete space | Every Cauchy sequence converges in X |
| Banach space | Complete normed vector space |
| Hilbert space | Complete inner product space |
| l^p norm | $\|x\|_p = \left(\sum_n |x_n|^p\right)^{1/p}$ |
| L^p norm | $\|f\|_p = \left(\int |f|^p d\mu\right)^{1/p}$ |

### Complete Spaces Checklist

| Space | Metric/Norm | Complete? |
|-------|-------------|-----------|
| ℝ, ℝⁿ | Any p-norm | Yes |
| ℚ | \|·\| | No |
| l^p (1≤p≤∞) | l^p norm | Yes |
| L^p(X, μ) | L^p norm | Yes |
| C[a,b] | Sup norm | Yes |
| C[a,b] | L² norm | No |

### Main Takeaways

1. **Cauchy sequences** capture "trying to converge" without knowing the limit
2. **Completeness** means every Cauchy sequence has its limit in the space
3. **L² is complete** (Riesz-Fischer)—essential for quantum mechanics
4. **Banach spaces** are the natural setting for operator theory
5. **Quantum states** are limits of partial sums—completeness makes this well-defined

---

## Daily Checklist

- [ ] I can define Cauchy sequences and verify the Cauchy property
- [ ] I understand why convergent ⟹ Cauchy but not conversely in general
- [ ] I can prove completeness of specific metric spaces
- [ ] I know examples of incomplete spaces (ℚ, C[a,b] with L² norm)
- [ ] I understand why L² completeness is essential for quantum mechanics
- [ ] I completed the computational lab

---

## Preview: Day 228

Tomorrow we explore the **Banach Fixed-Point Theorem**, one of the most powerful results in analysis. We'll see:
- Contraction mappings and their properties
- The proof of existence and uniqueness of fixed points
- Applications to differential equations and iterative methods
- Quantum mechanical applications: self-consistent field theory
