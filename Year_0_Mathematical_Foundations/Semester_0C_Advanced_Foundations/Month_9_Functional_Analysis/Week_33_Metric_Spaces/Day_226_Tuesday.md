# Day 226: Convergence and Continuity in Metric Spaces

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning I** | 2 hours | Sequential convergence and limits |
| **Morning II** | 1.5 hours | Open sets, closed sets, and neighborhoods |
| **Afternoon I** | 2 hours | Continuity: sequential and ε-δ characterizations |
| **Afternoon II** | 1.5 hours | Worked examples and practice problems |
| **Evening** | 1 hour | Computational lab: visualizing convergence |

## Learning Objectives

By the end of today, you will be able to:

1. **Define and verify** convergence of sequences in metric spaces
2. **Characterize** open balls, open sets, and closed sets topologically
3. **Prove** continuity using both sequential and ε-δ definitions
4. **Understand** the relationship between metric and topological structures
5. **Apply** continuity to linear operators on normed spaces
6. **Connect** convergence concepts to quantum mechanical state evolution

---

## 1. Convergence in Metric Spaces

### Definition: Convergent Sequence

Let (X, d) be a metric space. A sequence (xₙ) in X **converges** to x ∈ X if:

$$\boxed{\lim_{n \to \infty} d(x_n, x) = 0}$$

Equivalently, for every ε > 0, there exists N ∈ ℕ such that:

$$n \geq N \implies d(x_n, x) < \varepsilon$$

We write xₙ → x or lim xₙ = x.

### Uniqueness of Limits

**Theorem:** In a metric space, limits are unique.

**Proof:** Suppose xₙ → x and xₙ → y. By the triangle inequality:

$$d(x, y) \leq d(x, x_n) + d(x_n, y)$$

Given ε > 0, choose N such that n ≥ N implies d(xₙ, x) < ε/2 and d(xₙ, y) < ε/2. Then:

$$d(x, y) < \varepsilon/2 + \varepsilon/2 = \varepsilon$$

Since ε > 0 was arbitrary, d(x, y) = 0, so x = y. ∎

### Examples of Convergence

**Example 1: ℝ with Euclidean metric**

The sequence xₙ = 1/n converges to 0:
$$d(x_n, 0) = |1/n - 0| = 1/n \to 0$$

**Example 2: l² convergence**

The sequence of vectors eₙ = (0, 0, ..., 1, 0, ...) (1 in n-th position) does NOT converge in l². For m ≠ n:
$$d(e_m, e_n) = \sqrt{|1|^2 + |1|^2} = \sqrt{2}$$

The sequence doesn't even satisfy the Cauchy criterion.

**Example 3: L² convergence**

Consider fₙ(x) = sin(nx)/√π on [0, 2π]. These converge to 0 in L²:
$$\|f_n\|_2^2 = \frac{1}{\pi}\int_0^{2\pi} \sin^2(nx)\,dx = \frac{1}{\pi} \cdot \pi = 1$$

Wait—the norm is constant! They do NOT converge to 0. Instead:
$$d(f_n, 0) = 1 \text{ for all } n$$

This is a non-convergent bounded sequence.

---

## 2. Open and Closed Sets

### Open Balls

**Definition:** The **open ball** centered at x with radius r > 0 is:

$$\boxed{B(x, r) = B_r(x) = \{y \in X : d(x, y) < r\}}$$

The **closed ball** is:
$$\overline{B}(x, r) = \{y \in X : d(x, y) \leq r\}$$

### Open Sets

**Definition:** A set U ⊆ X is **open** if for every x ∈ U, there exists ε > 0 such that B(x, ε) ⊆ U.

**Properties of Open Sets:**
1. ∅ and X are open
2. Arbitrary unions of open sets are open
3. Finite intersections of open sets are open

**Theorem:** Open balls are open sets.

**Proof:** Let y ∈ B(x, r). Set ε = r - d(x, y) > 0. For any z ∈ B(y, ε):
$$d(x, z) \leq d(x, y) + d(y, z) < d(x, y) + \varepsilon = r$$
Thus B(y, ε) ⊆ B(x, r). ∎

### Closed Sets

**Definition:** A set F ⊆ X is **closed** if its complement X \ F is open.

**Equivalent characterization:** F is closed iff F contains all its limit points.

**Theorem:** F is closed ⟺ whenever (xₙ) ⊆ F and xₙ → x, then x ∈ F.

### Interior, Closure, and Boundary

For any set A ⊆ X:

- **Interior:** int(A) = largest open set contained in A
- **Closure:** cl(A) = Ā = smallest closed set containing A
- **Boundary:** ∂A = cl(A) \ int(A)

$$\boxed{\bar{A} = \{x \in X : d(x, A) = 0\} = \{x \in X : B(x, \varepsilon) \cap A \neq \emptyset \text{ for all } \varepsilon > 0\}}$$

---

## 3. Continuity

### Sequential Continuity

**Definition:** A function f: (X, d_X) → (Y, d_Y) is **continuous at x₀** if:

$$\boxed{x_n \to x_0 \implies f(x_n) \to f(x_0)}$$

f is **continuous** if it's continuous at every point.

### The ε-δ Definition

**Theorem:** f is continuous at x₀ iff for every ε > 0, there exists δ > 0 such that:

$$d_X(x, x_0) < \delta \implies d_Y(f(x), f(x_0)) < \varepsilon$$

### Topological Characterization

**Theorem:** f: X → Y is continuous iff the preimage of every open set is open:

$$V \text{ open in } Y \implies f^{-1}(V) \text{ open in } X$$

Equivalently: preimages of closed sets are closed.

### Lipschitz Continuity

**Definition:** f: X → Y is **Lipschitz continuous** with constant L ≥ 0 if:

$$\boxed{d_Y(f(x), f(y)) \leq L \cdot d_X(x, y) \text{ for all } x, y \in X}$$

If L < 1, f is called a **contraction** (crucial for Day 228).

**Lipschitz ⟹ uniformly continuous ⟹ continuous**

---

## 4. Bounded Linear Operators

### Continuity of Linear Maps

**Theorem:** For a linear map T: X → Y between normed spaces (which are metric spaces via d(x,y) = ||x-y||), the following are equivalent:

1. T is continuous
2. T is continuous at 0
3. T is bounded: there exists C ≥ 0 such that ||Tx|| ≤ C||x|| for all x
4. T is Lipschitz continuous

**Proof of (2) ⟺ (3):**

(3) ⟹ (2): If xₙ → 0, then ||Txₙ|| ≤ C||xₙ|| → 0, so Txₙ → 0.

(2) ⟹ (3): Suppose T is continuous at 0 but not bounded. Then for each n, there exists xₙ with ||Txₙ|| > n||xₙ||. Set yₙ = xₙ/(n||xₙ||). Then ||yₙ|| = 1/n → 0, so yₙ → 0. But:
$$\|Ty_n\| = \frac{\|Tx_n\|}{n\|x_n\|} > 1$$
So Tyₙ ↛ 0, contradicting continuity at 0. ∎

### The Operator Norm

For bounded linear T: X → Y:

$$\boxed{\|T\| = \sup_{\|x\| \leq 1} \|Tx\| = \sup_{x \neq 0} \frac{\|Tx\|}{\|x\|}}$$

This makes the space B(X, Y) of bounded linear operators a normed (hence metric) space.

---

## Quantum Mechanics Connection: Convergence of States

### Strong Convergence in L²

In quantum mechanics, a sequence of states (ψₙ) converges to ψ in L² means:

$$\|\psi_n - \psi\|_2 = \sqrt{\int |\psi_n(x) - \psi(x)|^2 \, dx} \to 0$$

This is the natural convergence for quantum states.

### Physical Interpretation

**What does L² convergence mean physically?**

If ||ψₙ - ψ||₂ → 0, then for any observable A with bounded operator:
$$|\langle \psi_n | A | \psi_n \rangle - \langle \psi | A | \psi \rangle| \to 0$$

The expectation values of observables converge!

### Weak vs Strong Convergence

**Strong convergence:** ψₙ → ψ means ||ψₙ - ψ|| → 0

**Weak convergence:** ψₙ ⇀ ψ means ⟨φ|ψₙ⟩ → ⟨φ|ψ⟩ for all φ ∈ L²

Strong implies weak, but not conversely. The orthonormal basis eₙ converges weakly to 0 (Riemann-Lebesgue lemma) but not strongly.

### Continuity of Quantum Evolution

The time evolution operator U(t) = e^{-iHt/ℏ} is:
- **Unitary**: ||U(t)ψ|| = ||ψ|| (preserves norms)
- **Strongly continuous**: ψ(t) → ψ(t₀) as t → t₀

This means small changes in time give small changes in the quantum state.

---

## Worked Examples

### Example 1: Proving Continuity of the Norm

**Problem:** Show that the norm function ||·||: X → ℝ is continuous on any normed space X.

**Solution:**

We need to show: xₙ → x ⟹ ||xₙ|| → ||x||.

By the reverse triangle inequality:
$$\big| \|x_n\| - \|x\| \big| \leq \|x_n - x\|$$

If xₙ → x, then ||xₙ - x|| → 0, so |||xₙ|| - ||x||| → 0.

Therefore ||·|| is continuous. In fact, it's Lipschitz with constant 1. ∎

### Example 2: Continuity of Matrix Multiplication

**Problem:** Show that matrix multiplication A ↦ AB is continuous on M_n(ℂ) with the operator norm.

**Solution:**

Let A, A' ∈ M_n(ℂ) and B ∈ M_n(ℂ) fixed. We show A ↦ AB is Lipschitz:

$$\|AB - A'B\| = \|(A - A')B\| \leq \|A - A'\| \cdot \|B\|$$

So the map is Lipschitz with constant ||B||, hence continuous.

Similarly, B ↦ AB is Lipschitz with constant ||A||. ∎

### Example 3: The Evaluation Map

**Problem:** On C[0,1] with the supremum norm, show that the evaluation map eₓ: f ↦ f(x) is continuous for each fixed x ∈ [0,1].

**Solution:**

For any f, g ∈ C[0,1]:

$$|e_x(f) - e_x(g)| = |f(x) - g(x)| \leq \sup_{t \in [0,1]} |f(t) - g(t)| = \|f - g\|_\infty$$

So eₓ is Lipschitz with constant 1, hence continuous.

**Note:** With the L² metric, eₓ is NOT continuous! Small L² distance doesn't imply pointwise closeness. ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Convergence in discrete metric:** Let X have the discrete metric. Characterize all convergent sequences in X.

2. **Open ball in l^∞:** Describe the open ball B(0, 1) in l^∞. Is the sequence (1, 1, 1, ...) in this ball?

3. **Continuity check:** Is f(x) = |x| continuous on ℝ with the usual metric? Prove it.

### Level 2: Intermediate

4. **Closure computation:** In ℝ with the usual metric, find the closure of:
   - (a) ℚ (rationals)
   - (b) (0, 1)
   - (c) {1/n : n ∈ ℕ}

5. **Continuous extension:** Let f: ℚ → ℝ be given by f(x) = x². Show that f has a unique continuous extension to ℝ.

6. **Operator continuity:** Let T: l² → l² be defined by T(x₁, x₂, x₃, ...) = (x₁, x₂/2, x₃/3, ...). Show T is bounded and find ||T||.

### Level 3: Challenging

7. **Nowhere continuous function:** Construct a function f: ℝ → ℝ that is nowhere continuous using the indicator function of ℚ. Prove it's discontinuous at every point.

8. **Equivalent metrics:** Two metrics d₁ and d₂ on X are equivalent if they define the same open sets. Prove that d₁ and d₂ are equivalent iff they define the same convergent sequences.

9. **Homeomorphism:** Show that (0, 1) and ℝ are homeomorphic (there exists a continuous bijection with continuous inverse). Find an explicit homeomorphism.

---

## Computational Lab: Visualizing Convergence

```python
"""
Day 226: Convergence and Continuity - Visualization Lab
Exploring convergence in different metrics and continuity
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from scipy.integrate import quad

# =============================================================================
# Part 1: Visualizing Open Balls in Different Metrics
# =============================================================================

def visualize_open_balls():
    """Compare open balls in different metrics on R^2."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    center = np.array([0, 0])
    radius = 1

    # L^1 (taxicab) - diamond
    ax1 = axes[0]
    diamond = plt.Polygon([
        [radius, 0], [0, radius], [-radius, 0], [0, -radius]
    ], fill=True, alpha=0.3, color='blue', edgecolor='blue', linewidth=2)
    ax1.add_patch(diamond)
    ax1.plot(0, 0, 'ko', markersize=8)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(r'$B_1(0, 1)$ in $d_1$ (taxicab)', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # L^2 (Euclidean) - circle
    ax2 = axes[1]
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax2.fill(x, y, alpha=0.3, color='green')
    ax2.plot(x, y, 'g-', linewidth=2)
    ax2.plot(0, 0, 'ko', markersize=8)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(r'$B_1(0, 1)$ in $d_2$ (Euclidean)', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # L^infinity (Chebyshev) - square
    ax3 = axes[2]
    square = plt.Rectangle((-radius, -radius), 2*radius, 2*radius,
                           fill=True, alpha=0.3, color='red',
                           edgecolor='red', linewidth=2)
    ax3.add_patch(square)
    ax3.plot(0, 0, 'ko', markersize=8)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title(r'$B_1(0, 1)$ in $d_\infty$ (Chebyshev)', fontsize=14)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    plt.suptitle('Open Balls in Different Metrics on $\\mathbb{R}^2$', fontsize=16)
    plt.tight_layout()
    plt.savefig('open_balls.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 2: Convergence of Sequences
# =============================================================================

def visualize_convergence():
    """Visualize different types of convergence."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Pointwise convergence: f_n(x) = x^n on [0,1]
    ax1 = axes[0, 0]
    x = np.linspace(0, 1, 1000)

    for n in [1, 2, 5, 10, 50]:
        ax1.plot(x, x**n, label=f'n = {n}', linewidth=2)

    # Limit function
    limit = np.where(x < 1, 0, 1)
    ax1.plot(x, limit, 'k--', linewidth=3, label='Limit')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('$f_n(x) = x^n$', fontsize=12)
    ax1.set_title('Pointwise but NOT Uniform Convergence', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Uniform convergence: f_n(x) = sin(x)/n
    ax2 = axes[0, 1]
    x = np.linspace(0, 2*np.pi, 1000)

    for n in [1, 2, 5, 10]:
        ax2.plot(x, np.sin(x)/n, label=f'n = {n}', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, label='Limit = 0')

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('$f_n(x) = \\sin(x)/n$', fontsize=12)
    ax2.set_title('Uniform Convergence to 0', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (c) L^2 but not pointwise: characteristic functions moving to infinity
    ax3 = axes[1, 0]
    x = np.linspace(0, 20, 2000)

    for n in [1, 3, 5, 10, 15]:
        fn = np.where((n <= x) & (x < n + 1), 1, 0)
        ax3.fill_between(x, 0, fn, alpha=0.4, label=f'$\\chi_{{[{n},{n+1}]}}$')

    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('$f_n(x)$', fontsize=12)
    ax3.set_title('$L^2$ Convergence to 0 (not pointwise)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 20)

    # (d) Sequence convergence in R^2
    ax4 = axes[1, 1]

    # Spiral sequence converging to origin
    n_points = 50
    n = np.arange(1, n_points + 1)
    theta = n * 0.5
    r = 1 / n

    x_seq = r * np.cos(theta)
    y_seq = r * np.sin(theta)

    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    ax4.scatter(x_seq, y_seq, c=colors, s=50, zorder=5)
    ax4.plot(x_seq, y_seq, 'gray', alpha=0.3, linewidth=1)
    ax4.plot(0, 0, 'r*', markersize=20, label='Limit (0, 0)')

    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('y', fontsize=12)
    ax4.set_title('Spiral Sequence Converging to Origin', fontsize=14)
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_types.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 3: ε-δ Definition of Continuity
# =============================================================================

def visualize_epsilon_delta():
    """Visualize the ε-δ definition of continuity."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Continuous function: f(x) = x^2
    ax1 = axes[0]
    x = np.linspace(-2, 2, 1000)
    f = lambda t: t**2

    x0 = 1
    epsilon = 0.3
    # For f(x) = x^2 near x0=1, |f(x) - f(x0)| = |x^2 - 1| = |x-1||x+1|
    # If |x - 1| < δ and |x| < 2 (say), then |x^2 - 1| < 3δ
    # So δ = ε/3 works
    delta = epsilon / 3

    ax1.plot(x, f(x), 'b-', linewidth=2, label='$f(x) = x^2$')
    ax1.axhline(y=f(x0), color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=f(x0) + epsilon, color='green', linestyle='--', label=f'$f(x_0) + \\varepsilon$')
    ax1.axhline(y=f(x0) - epsilon, color='green', linestyle='--', label=f'$f(x_0) - \\varepsilon$')

    ax1.axvline(x=x0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=x0 + delta, color='red', linestyle='--', label=f'$x_0 + \\delta$')
    ax1.axvline(x=x0 - delta, color='red', linestyle='--', label=f'$x_0 - \\delta$')

    # Shade the δ-neighborhood in domain
    ax1.axvspan(x0 - delta, x0 + delta, alpha=0.2, color='red')
    # Shade the ε-neighborhood in range
    ax1.axhspan(f(x0) - epsilon, f(x0) + epsilon, alpha=0.2, color='green')

    ax1.plot(x0, f(x0), 'ko', markersize=10)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title(f'Continuous: $\\varepsilon = {epsilon}$, $\\delta = {delta:.3f}$', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 4)

    # Discontinuous function: step function
    ax2 = axes[1]

    step = lambda t: np.where(t < 1, 0, 1)
    x = np.linspace(-0.5, 2.5, 1000)

    ax2.plot(x, step(x), 'b-', linewidth=2, label='Step function')
    ax2.plot(1, 0, 'bo', markersize=10)  # Open circle at discontinuity
    ax2.plot(1, 1, 'bo', markersize=10, fillstyle='none')

    x0 = 1
    epsilon = 0.3

    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1 + epsilon, color='green', linestyle='--')
    ax2.axhline(y=1 - epsilon, color='green', linestyle='--')

    # No δ works!
    ax2.axvspan(x0 - 0.3, x0 + 0.3, alpha=0.2, color='red',
                label='Any $\\delta$-neighborhood')

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f(x)', fontsize=12)
    ax2.set_title('Discontinuous: No $\\delta$ works for $\\varepsilon < 1$', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2)

    plt.tight_layout()
    plt.savefig('epsilon_delta.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 4: Operator Continuity - Quantum Example
# =============================================================================

def visualize_operator_continuity():
    """Demonstrate continuity of bounded linear operators."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: The multiplication operator on l^2 truncated
    ax1 = axes[0]

    # Operator T: (x_1, x_2, ...) -> (x_1, x_2/2, x_3/3, ...)
    def T_op(x, n_terms=20):
        """Apply the operator T(x_n) = x_n/n."""
        result = np.zeros_like(x)
        for i in range(len(x)):
            result[i] = x[i] / (i + 1)
        return result

    # Test vectors
    n_terms = 20
    indices = np.arange(1, n_terms + 1)

    # Vector 1: exponentially decaying
    x1 = 2.0 ** (-indices)
    Tx1 = T_op(x1)

    # Vector 2: slowly decaying
    x2 = 1.0 / indices
    Tx2 = T_op(x2)

    width = 0.35
    ax1.bar(indices - width/2, np.abs(x1), width, label='$|x_n|$', color='blue', alpha=0.7)
    ax1.bar(indices + width/2, np.abs(Tx1), width, label='$|Tx_n| = |x_n|/n$', color='red', alpha=0.7)

    ax1.set_xlabel('Index n', fontsize=12)
    ax1.set_ylabel('Magnitude', fontsize=12)
    ax1.set_title('Bounded Operator $T: x_n \\mapsto x_n/n$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # Compute norms
    norm_x1 = np.sqrt(np.sum(np.abs(x1)**2))
    norm_Tx1 = np.sqrt(np.sum(np.abs(Tx1)**2))
    print(f"||x|| = {norm_x1:.6f}, ||Tx|| = {norm_Tx1:.6f}, ratio = {norm_Tx1/norm_x1:.6f}")

    # Right: Continuity - small change in input gives small change in output
    ax2 = axes[1]

    # Create many random unit vectors and compute ||Tx|| / ||x||
    np.random.seed(42)
    n_samples = 1000
    ratios = []

    for _ in range(n_samples):
        # Random unit vector in first 50 components
        x = np.random.randn(50)
        x = x / np.linalg.norm(x)
        Tx = T_op(x)
        ratios.append(np.linalg.norm(Tx))

    ax2.hist(ratios, bins=50, density=True, alpha=0.7, color='steelblue',
             edgecolor='black')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=2,
                label=f'$\\|T\\| = 1$')
    ax2.axvline(x=np.max(ratios), color='green', linestyle='--', linewidth=2,
                label=f'Max observed = {np.max(ratios):.4f}')

    ax2.set_xlabel('$\\|Tx\\|$ for unit vectors $x$', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of $\\|Tx\\|$ for Random Unit Vectors', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('operator_continuity.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 5: Interior, Closure, Boundary Visualization
# =============================================================================

def visualize_closure_interior():
    """Visualize interior, closure, and boundary of sets."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Set: open disk minus a point
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1

    # Original set: open ball B(0,1) minus the origin
    ax1 = axes[0]
    circle = plt.Circle((0, 0), r, fill=True, alpha=0.3, color='blue',
                         edgecolor='blue', linewidth=2, linestyle='--')
    ax1.add_patch(circle)
    ax1.plot(0, 0, 'ro', markersize=10, label='Removed point (0,0)')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Set $A = B(0,1) \\setminus \\{0\\}$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Interior: same as original (origin was never in the interior)
    ax2 = axes[1]
    circle_int = plt.Circle((0, 0), r, fill=True, alpha=0.3, color='green',
                             edgecolor='green', linewidth=2, linestyle='--')
    ax2.add_patch(circle_int)
    ax2.plot(0, 0, 'go', markersize=10, fillstyle='none', markeredgewidth=2,
             label='Origin in interior')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('Interior: $\\mathrm{int}(A) = B(0,1) \\setminus \\{0\\}$', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Closure: closed ball (includes boundary and origin)
    ax3 = axes[2]
    circle_cl = plt.Circle((0, 0), r, fill=True, alpha=0.3, color='red',
                           edgecolor='red', linewidth=3)
    ax3.add_patch(circle_cl)
    ax3.plot(0, 0, 'r*', markersize=15, label='Origin now included')
    # Mark boundary
    x_bd = r * np.cos(theta)
    y_bd = r * np.sin(theta)
    ax3.plot(x_bd, y_bd, 'k-', linewidth=3, label='Boundary $\\partial A$')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_title('Closure: $\\bar{A} = \\overline{B}(0,1)$', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Interior, Closure, and Boundary', fontsize=16)
    plt.tight_layout()
    plt.savefig('closure_interior.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Run All Visualizations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 226: Convergence and Continuity - Visualization Lab")
    print("=" * 60)

    print("\n1. Open balls in different metrics...")
    visualize_open_balls()

    print("\n2. Types of convergence...")
    visualize_convergence()

    print("\n3. ε-δ definition of continuity...")
    visualize_epsilon_delta()

    print("\n4. Operator continuity...")
    visualize_operator_continuity()

    print("\n5. Interior, closure, and boundary...")
    visualize_closure_interior()

    print("\n" + "=" * 60)
    print("Lab complete! Images saved.")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula/Definition |
|---------|---------------------|
| Convergence | $x_n \to x$ iff $d(x_n, x) \to 0$ |
| Open ball | $B(x, r) = \{y : d(x,y) < r\}$ |
| Open set | Contains a ball around each point |
| Closed set | Complement of open set; contains all limits |
| Closure | $\bar{A} = \{x : d(x, A) = 0\}$ |
| Continuity (seq) | $x_n \to x \Rightarrow f(x_n) \to f(x)$ |
| Continuity (ε-δ) | $\forall \varepsilon > 0, \exists \delta > 0: d(x,x_0) < \delta \Rightarrow d(f(x), f(x_0)) < \varepsilon$ |
| Lipschitz | $d(f(x), f(y)) \leq L \cdot d(x, y)$ |
| Operator norm | $\|T\| = \sup_{\|x\| \leq 1} \|Tx\|$ |

### Main Takeaways

1. **Convergence generalizes** from ℝ to any metric space via the distance function
2. **Open balls generate** the topology (open sets, closed sets)
3. **Continuity has equivalent formulations**: sequential, ε-δ, topological
4. **Bounded linear operators** are automatically continuous
5. **Strong L² convergence** ensures expectation values converge in QM

---

## Daily Checklist

- [ ] I can verify convergence of sequences in metric spaces
- [ ] I understand open balls, open sets, and closed sets
- [ ] I can prove continuity using ε-δ arguments
- [ ] I know the relationship between boundedness and continuity for linear operators
- [ ] I understand the physical meaning of L² convergence in QM
- [ ] I completed the computational lab

---

## Preview: Day 227

Tomorrow we tackle **completeness and Cauchy sequences**. We'll explore:
- The Cauchy criterion for convergence
- Complete metric spaces (where every Cauchy sequence converges)
- Why completeness is essential for L² as the quantum state space
- Banach spaces (complete normed spaces)
