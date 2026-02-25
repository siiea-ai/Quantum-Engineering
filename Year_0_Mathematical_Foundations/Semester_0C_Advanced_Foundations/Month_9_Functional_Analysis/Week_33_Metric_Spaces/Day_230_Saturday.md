# Day 230: Completion of Metric Spaces

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning I** | 2 hours | The completion construction via Cauchy sequences |
| **Morning II** | 1.5 hours | Universal property and uniqueness |
| **Afternoon I** | 2 hours | Examples: ℚ → ℝ, pre-Hilbert → Hilbert |
| **Afternoon II** | 1.5 hours | Worked examples and practice problems |
| **Evening** | 1 hour | Computational lab: building completions |

## Learning Objectives

By the end of today, you will be able to:

1. **Construct** the completion of any metric space using Cauchy sequences
2. **Prove** the universal property characterizing completions
3. **Understand** isometric embeddings and dense subsets
4. **Apply** completion to concrete examples (ℚ → ℝ, polynomials → L²)
5. **Explain** why step functions complete to L² functions
6. **Connect** completions to the construction of quantum state spaces

---

## 1. The Problem: Incomplete Spaces

### Why Completion Matters

Some naturally arising metric spaces are incomplete:
- ℚ with |·|: rationals don't contain √2
- C[a,b] with L² norm: limits can be discontinuous
- Polynomials with L² norm: limits need not be polynomials

We need a systematic way to "fill in the gaps."

### The Goal

Given an incomplete metric space (X, d), construct a complete metric space (X̃, d̃) such that:
1. X embeds into X̃ (we can identify X with a subset of X̃)
2. X is dense in X̃ (every point of X̃ is a limit of points from X)
3. X̃ is complete (every Cauchy sequence converges)

---

## 2. Construction via Cauchy Sequences

### Step 1: Define the Set of Cauchy Sequences

Let C(X) be the set of all Cauchy sequences in (X, d):

$$C(X) = \{(x_n) : (x_n) \text{ is Cauchy in } X\}$$

### Step 2: Define an Equivalence Relation

Two Cauchy sequences are **equivalent** if they "try to converge to the same limit":

$$\boxed{(x_n) \sim (y_n) \iff \lim_{n \to \infty} d(x_n, y_n) = 0}$$

**Claim:** This is an equivalence relation.

**Proof:**
- *Reflexive:* d(xₙ, xₙ) = 0 → 0 ✓
- *Symmetric:* d(xₙ, yₙ) = d(yₙ, xₙ) ✓
- *Transitive:* d(xₙ, zₙ) ≤ d(xₙ, yₙ) + d(yₙ, zₙ) → 0 + 0 = 0 ✓

### Step 3: Define the Completion

$$\boxed{\tilde{X} = C(X) / \sim = \{[(x_n)] : (x_n) \text{ Cauchy}\}}$$

where [(xₙ)] denotes the equivalence class of (xₙ).

### Step 4: Define the Metric

For equivalence classes [(xₙ)] and [(yₙ)]:

$$\boxed{\tilde{d}([(x_n)], [(y_n)]) = \lim_{n \to \infty} d(x_n, y_n)}$$

**Claim:** This limit exists and is well-defined.

**Proof of existence:** The sequence (d(xₙ, yₙ)) is Cauchy in ℝ:
$$|d(x_m, y_m) - d(x_n, y_n)| \leq d(x_m, x_n) + d(y_m, y_n) \to 0$$

**Proof of well-definedness:** If (xₙ) ~ (x'ₙ) and (yₙ) ~ (y'ₙ):
$$|d(x_n, y_n) - d(x'_n, y'_n)| \leq d(x_n, x'_n) + d(y_n, y'_n) \to 0$$

### Step 5: Verify d̃ is a Metric

**(M1) Positivity:**
- d̃ ≥ 0: limit of non-negative numbers
- d̃ = 0 ⟺ d(xₙ, yₙ) → 0 ⟺ (xₙ) ~ (yₙ) ⟺ same equivalence class

**(M2) Symmetry:** d(xₙ, yₙ) = d(yₙ, xₙ)

**(M3) Triangle Inequality:** d(xₙ, zₙ) ≤ d(xₙ, yₙ) + d(yₙ, zₙ), take limit.

---

## 3. Properties of the Completion

### The Embedding

Define i: X → X̃ by:
$$i(x) = [(x, x, x, ...)] = \text{constant sequence at } x$$

**Claim:** i is an **isometric embedding**: d̃(i(x), i(y)) = d(x, y).

**Proof:** d̃(i(x), i(y)) = lim d(x, y) = d(x, y). ∎

We identify X with i(X) ⊆ X̃.

### Density

**Theorem:** X is dense in X̃.

**Proof:** Let [(xₙ)] ∈ X̃. We show i(X) comes arbitrarily close to [(xₙ)].

For any ε > 0, since (xₙ) is Cauchy, ∃N: n ≥ N ⟹ d(xₙ, xₘ) < ε for m ≥ N.

Consider i(xₙ) = [(xₙ, xₙ, xₙ, ...)]. Then:
$$\tilde{d}(i(x_N), [(x_n)]) = \lim_{n \to \infty} d(x_N, x_n) < \varepsilon$$

So every point in X̃ is within ε of X for any ε > 0. ∎

### Completeness

**Theorem:** X̃ is complete.

**Proof:** Let ((xₙ^{(k)}))_k be a Cauchy sequence in X̃ (each xₙ^{(k)} represents a Cauchy sequence in X).

Since X is dense in X̃, for each k, choose yₖ ∈ X with d̃(yₖ, [(xₙ^{(k)})]) < 1/k.

The sequence (yₖ) is Cauchy in X:
$$d(y_k, y_l) = \tilde{d}(i(y_k), i(y_l)) \leq \tilde{d}(i(y_k), [(x_n^{(k)})]) + \tilde{d}([(x_n^{(k)})], [(x_n^{(l)})]) + \tilde{d}([(x_n^{(l)})], i(y_l))$$

As k, l → ∞, this goes to 0.

So [(yₖ)] ∈ X̃, and one can verify [(xₙ^{(k)})] → [(yₖ)]. ∎

---

## 4. The Universal Property

### Statement

**Theorem (Universal Property of Completion):**

Let (X̃, d̃) be the completion of (X, d) with embedding i: X → X̃. If (Y, d_Y) is any complete metric space and f: X → Y is uniformly continuous, then there exists a unique uniformly continuous extension f̃: X̃ → Y such that f̃ ∘ i = f.

$$\begin{CD}
X @>i>> \tilde{X} \\
@VfVV @VV\exists ! \tilde{f}V \\
Y @= Y
\end{CD}$$

### Proof Sketch

**Existence:** For [(xₙ)] ∈ X̃, the sequence (f(xₙ)) is Cauchy in Y (by uniform continuity). Since Y is complete, it converges. Define f̃([(xₙ)]) = lim f(xₙ).

**Uniqueness:** Any continuous extension must agree on the dense subset X, hence everywhere.

### Corollary: Uniqueness of Completion

**Theorem:** The completion is unique up to isometric isomorphism.

If (X̃₁, i₁) and (X̃₂, i₂) are both completions of X, there's a unique isometry φ: X̃₁ → X̃₂ with φ ∘ i₁ = i₂.

---

## 5. Examples of Completions

### Example 1: ℚ → ℝ

The completion of (ℚ, |·|) is (ℝ, |·|).

**Construction:**
- Cauchy sequences in ℚ (like 1, 1.4, 1.41, 1.414, ...)
- Equivalent sequences define the same real number
- ℚ embeds as the constant sequences

**Every real number** is an equivalence class of Cauchy sequences of rationals!

This is one of the standard constructions of ℝ from ℚ.

### Example 2: Polynomials → L²

Let P[a, b] be the polynomials on [a, b] with the L² norm:

$$\|p\|_2 = \sqrt{\int_a^b |p(x)|^2 \, dx}$$

The completion of P[a, b] is L²[a, b].

**Why?** The Weierstrass approximation theorem says polynomials are dense in C[a, b] (sup norm), hence also dense in L². The completion adds all L² functions.

### Example 3: Step Functions → L²

Let S[a, b] be the step functions (piecewise constant). The completion is again L²[a, b].

**This is how L² is often constructed:**
1. Start with step functions
2. Complete in the L² norm
3. Obtain all square-integrable functions

### Example 4: Pre-Hilbert → Hilbert Space

A **pre-Hilbert space** is an inner product space that may not be complete.

**Theorem:** The completion of any pre-Hilbert space is a Hilbert space.

The inner product extends by:
$$\langle [(x_n)], [(y_n)] \rangle = \lim_{n \to \infty} \langle x_n, y_n \rangle$$

---

## Quantum Mechanics Connection: Constructing State Spaces

### The Algebraic vs Analytic Approach

In quantum mechanics, we can start with:
- **Simple states** (e.g., polynomial wavefunctions, step functions)
- **Complete** to get the full state space L²

This ensures:
1. All physically reasonable limits exist
2. The mathematical structure is well-behaved
3. Spectral theory applies

### Example: Constructing L²(ℝ)

**Step 1:** Start with smooth, compactly supported functions C_c^∞(ℝ).

**Step 2:** Define the L² inner product:
$$\langle f, g \rangle = \int_{-\infty}^{\infty} f(x)\overline{g(x)} \, dx$$

**Step 3:** Complete to get L²(ℝ).

The resulting space contains:
- All smooth functions with finite L² norm
- Many non-smooth functions (e.g., step functions)
- Functions defined only almost everywhere

### Physical Interpretation

The completion process adds:
- **Limit states** of physical processes
- **Idealizations** (delta-function-like objects, approached as limits)
- **Mathematical convenience** (Fourier transforms, spectral decomposition)

### The Rigged Hilbert Space

For quantum mechanics, we actually use a "rigged" Hilbert space:

$$\Phi \subset \mathcal{H} \subset \Phi'$$

Where:
- Φ = "nice" functions (Schwartz space, C_c^∞)
- H = L² (completion of Φ)
- Φ' = distributions (generalized functions, including δ)

The completion sits in the middle of this structure.

---

## Worked Examples

### Example 1: Completing (0, 1)

**Problem:** Find the completion of the open interval (0, 1) with the usual metric.

**Solution:**

The completion is the closed interval [0, 1].

**Why?**
- Cauchy sequences in (0, 1) that "try to converge to 0" (like 1/n) become the equivalence class representing 0
- Similarly for sequences approaching 1
- Sequences converging to x ∈ (0, 1) stay in (0, 1)

The embedding i: (0, 1) → [0, 1] is the inclusion map. ∎

### Example 2: Completing a Discrete Space

**Problem:** Let X = {1/n : n ∈ ℕ} with the usual metric. Find the completion.

**Solution:**

The completion is X̃ = X ∪ {0} = {0} ∪ {1/n : n ∈ ℕ}.

**Why?**
- Cauchy sequences in X that don't converge in X must approach 0
- The only "missing" limit point is 0

The completion adds the single point 0. ∎

### Example 3: The Completion is Already Complete

**Problem:** What is the completion of ℝ?

**Solution:**

Since ℝ is already complete, its completion is (isometrically isomorphic to) ℝ itself.

Every Cauchy sequence in ℝ converges to some r ∈ ℝ, so the equivalence class is just the constant sequence at r.

**General principle:** X complete ⟹ X̃ ≅ X. ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Simple completion:** What is the completion of ℤ with the usual metric?

2. **Embedding:** Verify that the map i: ℚ → ℝ given by i(q) = q is an isometry.

3. **Density:** Show that ℚ is dense in ℝ (every real is a limit of rationals).

### Level 2: Intermediate

4. **Completion of C[0,1] in L¹:** The space C[0,1] with the L¹ norm is incomplete. What is its completion?

5. **Preserving structure:** If X is a normed vector space (incomplete), show its completion X̃ is also a normed vector space with the extended norm.

6. **Completion of finite sets:** Let X be a finite metric space. Prove X is complete and hence X̃ ≅ X.

### Level 3: Challenging

7. **Product completion:** If X and Y are metric spaces with completions X̃ and Ỹ, what is the completion of X × Y?

8. **Completion and subspaces:** If Y ⊆ X is a subspace, describe the relationship between Ỹ and X̃. Is Ỹ always a subspace of X̃?

9. **Non-Archimedean completion:** The p-adic metric on ℚ is defined by |x|_p = p^{-v_p(x)} where v_p(x) is the largest power of p dividing x. Find the completion of (ℚ, |·|_p). (This gives the p-adic numbers ℚ_p!)

---

## Computational Lab: Building Completions

```python
"""
Day 230: Completion of Metric Spaces - Computational Lab
Exploring the construction and properties of metric space completions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# =============================================================================
# Part 1: Visualizing Cauchy Sequences and Equivalence Classes
# =============================================================================

def cauchy_equivalence_classes():
    """Visualize Cauchy sequences that are equivalent (converge to same limit)."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n = np.arange(1, 51)

    # Left: Two equivalent Cauchy sequences (both → √2)
    ax1 = axes[0]

    # Sequence 1: Babylonian method
    x1 = [1.0]
    for _ in range(49):
        x1.append((x1[-1] + 2/x1[-1]) / 2)
    x1 = np.array(x1)

    # Sequence 2: Different approach to √2
    x2 = [1.5]
    for _ in range(49):
        # Newton on x^2 - 2 = 0, starting from 1.5
        x2.append(x2[-1] - (x2[-1]**2 - 2) / (2 * x2[-1]))
    x2 = np.array(x2)

    ax1.plot(n, x1, 'b-', linewidth=2, label='Babylonian from 1.0')
    ax1.plot(n, x2, 'r--', linewidth=2, label='Newton from 1.5')
    ax1.axhline(y=np.sqrt(2), color='k', linestyle=':', linewidth=2,
                label=f'$\\sqrt{{2}} \\approx {np.sqrt(2):.6f}$')

    ax1.set_xlabel('n', fontsize=12)
    ax1.set_ylabel('$x_n$', fontsize=12)
    ax1.set_title('Equivalent Cauchy Sequences (same class in $\\tilde{\\mathbb{Q}}$)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Distance between them → 0
    ax2 = axes[1]
    distances = np.abs(x1 - x2)

    ax2.semilogy(n, distances + 1e-20, 'g-', linewidth=2)
    ax2.set_xlabel('n', fontsize=12)
    ax2.set_ylabel('$|x_n^{(1)} - x_n^{(2)}|$', fontsize=12)
    ax2.set_title('Distance Between Equivalent Sequences → 0', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cauchy_equivalence.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Equivalent Cauchy sequences define the same element in the completion.")
    print(f"Final distance: |x₅₀⁽¹⁾ - x₅₀⁽²⁾| = {distances[-1]:.2e}")

# =============================================================================
# Part 2: Completing (0, 1) to [0, 1]
# =============================================================================

def complete_open_interval():
    """Demonstrate the completion of (0,1) to [0,1]."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: The incomplete space (0, 1)
    ax1 = axes[0]
    x = np.linspace(0.01, 0.99, 100)
    ax1.plot(x, np.zeros_like(x), 'b-', linewidth=4)
    ax1.plot(0, 0, 'wo', markersize=15, markeredgecolor='blue', markeredgewidth=2)
    ax1.plot(1, 0, 'wo', markersize=15, markeredgecolor='blue', markeredgewidth=2)

    # Cauchy sequence approaching 0
    seq_to_0 = 1 / np.arange(2, 12)
    ax1.plot(seq_to_0, np.zeros_like(seq_to_0), 'ro', markersize=8)
    for i, xi in enumerate(seq_to_0[:5]):
        ax1.annotate(f'1/{i+2}', (xi, 0.02), ha='center', fontsize=9)

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 0.1)
    ax1.set_title('Incomplete: $(0, 1)$', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.axhline(y=0, color='k', linewidth=0.5)

    # Right: The complete space [0, 1]
    ax2 = axes[1]
    x = np.linspace(0, 1, 100)
    ax2.plot(x, np.zeros_like(x), 'b-', linewidth=4)
    ax2.plot(0, 0, 'bo', markersize=15)  # Filled - now included
    ax2.plot(1, 0, 'bo', markersize=15)  # Filled - now included

    # The sequences now converge
    ax2.annotate('Added!', (0, 0.03), ha='center', fontsize=10, color='red')
    ax2.annotate('Added!', (1, 0.03), ha='center', fontsize=10, color='red')

    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 0.1)
    ax2.set_title('Completion: $[0, 1]$', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.axhline(y=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('complete_interval.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Completion of (0, 1) adds the endpoints 0 and 1.")

# =============================================================================
# Part 3: Completing Step Functions to L^2
# =============================================================================

def step_to_L2_completion():
    """Show how step functions complete to L^2 functions."""

    x = np.linspace(0, 1, 1000)

    # Target: a non-step L^2 function
    target = np.sin(np.pi * x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Approximating sin(πx) by step functions
    ax1 = axes[0]
    ax1.plot(x, target, 'k-', linewidth=3, label='$\\sin(\\pi x)$ (target)')

    for n_steps in [2, 4, 8, 16]:
        # Create step function approximation
        step_x = np.linspace(0, 1, n_steps + 1)
        step_y = np.sin(np.pi * (step_x[:-1] + step_x[1:]) / 2)  # Midpoint values

        # Plot step function
        for i in range(n_steps):
            ax1.hlines(step_y[i], step_x[i], step_x[i+1],
                      colors=plt.cm.viridis(n_steps/20), linewidth=2, alpha=0.7)
            if i > 0:
                ax1.vlines(step_x[i], step_y[i-1], step_y[i],
                          colors=plt.cm.viridis(n_steps/20), linewidth=1, alpha=0.5)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Step Function Approximations', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: L^2 error decreasing
    ax2 = axes[1]

    n_values = [2, 4, 8, 16, 32, 64, 128, 256]
    L2_errors = []

    for n_steps in n_values:
        step_x = np.linspace(0, 1, n_steps + 1)
        step_y = np.sin(np.pi * (step_x[:-1] + step_x[1:]) / 2)

        # Compute L^2 error
        error_sq = 0
        for i in range(n_steps):
            # Integrate (sin(πx) - step_y[i])^2 on [step_x[i], step_x[i+1]]
            def integrand(t):
                return (np.sin(np.pi * t) - step_y[i])**2
            val, _ = quad(integrand, step_x[i], step_x[i+1])
            error_sq += val

        L2_errors.append(np.sqrt(error_sq))

    ax2.loglog(n_values, L2_errors, 'bo-', markersize=8, linewidth=2)
    ax2.set_xlabel('Number of steps', fontsize=12)
    ax2.set_ylabel('$L^2$ error', fontsize=12)
    ax2.set_title('Convergence in $L^2$ Norm', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Theoretical: error ~ 1/n for step functions
    ax2.loglog(n_values, [L2_errors[0] * n_values[0] / n for n in n_values],
               'r--', linewidth=2, label='$O(1/n)$ reference')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('step_to_L2.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Step functions are dense in L^2:")
    print("Every L^2 function is a limit of step functions!")

# =============================================================================
# Part 4: Polynomial Approximation (Weierstrass)
# =============================================================================

def polynomial_completion():
    """Polynomials completing to continuous functions (Weierstrass)."""

    x = np.linspace(0, 1, 1000)

    # Target: |x - 0.5| (continuous but not smooth)
    target = np.abs(x - 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Polynomial approximations
    ax1 = axes[0]
    ax1.plot(x, target, 'k-', linewidth=3, label='$|x - 0.5|$')

    # Bernstein polynomials approximate
    def bernstein_approx(f, n, x):
        """Bernstein polynomial approximation of degree n."""
        result = np.zeros_like(x)
        for k in range(n + 1):
            # Binomial coefficient
            binom = np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))
            # Bernstein basis polynomial
            basis = binom * x**k * (1 - x)**(n - k)
            # Function value at k/n
            result += f(k / n) * basis
        return result

    f = lambda t: np.abs(t - 0.5)

    for n in [5, 10, 20, 50]:
        approx = bernstein_approx(f, n, x)
        ax1.plot(x, approx, '--', linewidth=1.5, alpha=0.8, label=f'n = {n}')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Bernstein Polynomial Approximations', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Sup norm convergence
    ax2 = axes[1]

    n_values = [5, 10, 20, 50, 100, 200]
    sup_errors = []

    for n in n_values:
        approx = bernstein_approx(f, n, x)
        sup_errors.append(np.max(np.abs(target - approx)))

    ax2.loglog(n_values, sup_errors, 'go-', markersize=8, linewidth=2)
    ax2.set_xlabel('Polynomial degree n', fontsize=12)
    ax2.set_ylabel('Sup norm error', fontsize=12)
    ax2.set_title('Convergence in Sup Norm', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('polynomial_completion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nWeierstrass Approximation Theorem:")
    print("Polynomials are dense in C[0,1] with sup norm.")
    print("(But completion adds NO new functions - C[0,1] is already complete!)")

# =============================================================================
# Part 5: Completion Metric Calculation
# =============================================================================

def completion_metric():
    """Numerically compute the completion metric."""

    # Two Cauchy sequences in Q approaching different limits
    # Sequence 1: approaches √2
    # Sequence 2: approaches √3

    n_terms = 50
    n = np.arange(1, n_terms + 1)

    # Sequence 1: √2 via Newton
    x = [1.0]
    for _ in range(n_terms - 1):
        x.append((x[-1] + 2/x[-1]) / 2)
    x = np.array(x)

    # Sequence 2: √3 via Newton
    y = [2.0]
    for _ in range(n_terms - 1):
        y.append((y[-1] + 3/y[-1]) / 2)
    y = np.array(y)

    # Compute d(x_n, y_n) for each n
    distances = np.abs(x - y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: The two sequences
    ax1 = axes[0]
    ax1.plot(n, x, 'b-', linewidth=2, label=f'$(x_n) \\to \\sqrt{{2}} \\approx {np.sqrt(2):.4f}$')
    ax1.plot(n, y, 'r-', linewidth=2, label=f'$(y_n) \\to \\sqrt{{3}} \\approx {np.sqrt(3):.4f}$')
    ax1.axhline(y=np.sqrt(2), color='b', linestyle='--', alpha=0.5)
    ax1.axhline(y=np.sqrt(3), color='r', linestyle='--', alpha=0.5)

    ax1.set_xlabel('n', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Two Cauchy Sequences in $\\mathbb{Q}$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: The completion metric converging
    ax2 = axes[1]

    # d_tilde = lim d(x_n, y_n)
    true_distance = np.sqrt(3) - np.sqrt(2)

    ax2.plot(n, distances, 'g-', linewidth=2, label='$d(x_n, y_n)$')
    ax2.axhline(y=true_distance, color='k', linestyle='--', linewidth=2,
                label=f'$\\tilde{{d}} = \\sqrt{{3}} - \\sqrt{{2}} \\approx {true_distance:.4f}$')

    ax2.set_xlabel('n', fontsize=12)
    ax2.set_ylabel('Distance', fontsize=12)
    ax2.set_title('Completion Metric: $\\tilde{d} = \\lim d(x_n, y_n)$', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('completion_metric.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nCompletion metric computation:")
    print(f"  lim d(x_n, y_n) = {distances[-1]:.10f}")
    print(f"  √3 - √2 = {true_distance:.10f}")
    print(f"  Error: {abs(distances[-1] - true_distance):.2e}")

# =============================================================================
# Run All Visualizations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 230: Completion of Metric Spaces - Lab")
    print("=" * 60)

    print("\n1. Cauchy equivalence classes...")
    cauchy_equivalence_classes()

    print("\n2. Completing (0,1) to [0,1]...")
    complete_open_interval()

    print("\n3. Step functions → L²...")
    step_to_L2_completion()

    print("\n4. Polynomial approximation...")
    polynomial_completion()

    print("\n5. Computing the completion metric...")
    completion_metric()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas and Concepts

| Concept | Definition/Formula |
|---------|---------------------|
| Cauchy sequences | $C(X) = \{(x_n) : (x_n) \text{ is Cauchy}\}$ |
| Equivalence | $(x_n) \sim (y_n) \iff \lim d(x_n, y_n) = 0$ |
| Completion | $\tilde{X} = C(X) / \sim$ |
| Completion metric | $\tilde{d}([(x_n)], [(y_n)]) = \lim d(x_n, y_n)$ |
| Embedding | $i(x) = [(x, x, x, ...)]$ |
| Density | $X$ is dense in $\tilde{X}$ |

### Key Properties of Completion

| Property | Statement |
|----------|-----------|
| Isometric embedding | $\tilde{d}(i(x), i(y)) = d(x, y)$ |
| Density | Every point in $\tilde{X}$ is a limit of points from $X$ |
| Completeness | $\tilde{X}$ is complete |
| Universal property | Uniformly continuous maps extend uniquely |
| Uniqueness | Completion is unique up to isometry |

### Important Completions

| Space | Completion |
|-------|------------|
| ℚ | ℝ |
| (0, 1) | [0, 1] |
| Polynomials (L² norm) | L² |
| Step functions (L² norm) | L² |
| Pre-Hilbert space | Hilbert space |

### Main Takeaways

1. **Completion fills gaps** by adding limits of Cauchy sequences
2. **The construction** uses equivalence classes of Cauchy sequences
3. **The original space** embeds isometrically and densely
4. **The completion is unique** up to isometric isomorphism
5. **Quantum mechanics** uses completions to construct L²

---

## Daily Checklist

- [ ] I can construct the completion using Cauchy sequences
- [ ] I understand the equivalence relation on Cauchy sequences
- [ ] I can verify the completion metric is well-defined
- [ ] I know the universal property of completions
- [ ] I can apply this to concrete examples (ℚ → ℝ, polynomials → L²)
- [ ] I completed the computational lab

---

## Preview: Day 231

Tomorrow is the **Week 33 Review**. We'll:
- Synthesize all concepts from the week
- Work through comprehensive problems
- Practice proofs involving metric spaces
- Connect everything to quantum mechanics applications
