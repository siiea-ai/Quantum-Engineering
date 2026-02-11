# Day 225: Metric Spaces — Definitions and Examples

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning I** | 2 hours | Metric space axioms and basic properties |
| **Morning II** | 1.5 hours | Examples: ℝⁿ, sequence spaces l^p |
| **Afternoon I** | 2 hours | Function spaces: C[a,b] and L^p |
| **Afternoon II** | 1.5 hours | Worked examples and practice problems |
| **Evening** | 1 hour | Computational lab: visualizing metrics |

## Learning Objectives

By the end of today, you will be able to:

1. **State and verify** the three axioms defining a metric space
2. **Compute distances** in ℝⁿ with Euclidean, taxicab, and supremum metrics
3. **Define and work with** sequence spaces l^p for p ∈ [1, ∞]
4. **Understand** the function spaces C[a,b] and L^p with their natural metrics
5. **Prove** that a given distance function is (or is not) a valid metric
6. **Connect** metric space structure to quantum mechanical state spaces

---

## 1. Metric Space Axioms

### Definition: Metric Space

A **metric space** is an ordered pair (X, d) where X is a non-empty set and d: X × X → ℝ is a function called a **metric** (or **distance function**) satisfying:

$$\boxed{\begin{aligned}
&\textbf{(M1) Positivity:} \quad d(x, y) \geq 0 \text{ for all } x, y \in X \\
&\phantom{\textbf{(M1) Positivity:}} \quad d(x, y) = 0 \iff x = y \\
&\textbf{(M2) Symmetry:} \quad d(x, y) = d(y, x) \text{ for all } x, y \in X \\
&\textbf{(M3) Triangle Inequality:} \quad d(x, z) \leq d(x, y) + d(y, z) \text{ for all } x, y, z \in X
\end{aligned}}$$

### Intuition Behind the Axioms

- **Positivity**: Distance is never negative; zero distance means identical points
- **Symmetry**: The distance from A to B equals the distance from B to A
- **Triangle Inequality**: The direct path is never longer than any detour

### Remark on Pseudometrics

If we relax (M1) to allow d(x, y) = 0 for x ≠ y, we get a **pseudometric**. This arises naturally in L^p spaces where functions differing on measure-zero sets have zero distance.

---

## 2. The Euclidean Spaces ℝⁿ

### The Standard Euclidean Metric

For x = (x₁, ..., xₙ) and y = (y₁, ..., yₙ) in ℝⁿ:

$$\boxed{d_2(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} = \|x - y\|_2}$$

**Verification of Axioms:**

- **(M1)**: Square root of sum of squares is ≥ 0. Equals zero iff all (xᵢ - yᵢ)² = 0, iff x = y.
- **(M2)**: (xᵢ - yᵢ)² = (yᵢ - xᵢ)², so symmetric.
- **(M3)**: Follows from the Cauchy-Schwarz inequality (proven below).

### Triangle Inequality Proof for ℝⁿ

We use the Minkowski inequality. For any vectors u, v ∈ ℝⁿ:

$$\|u + v\|_2 \leq \|u\|_2 + \|v\|_2$$

Setting u = x - y and v = y - z:

$$\|x - z\|_2 = \|(x - y) + (y - z)\|_2 \leq \|x - y\|_2 + \|y - z\|_2$$

Thus d₂(x, z) ≤ d₂(x, y) + d₂(y, z). ∎

### Other Metrics on ℝⁿ

**The p-metrics** (for p ≥ 1):

$$d_p(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}$$

**Special cases:**

| p | Name | Formula | Geometry |
|---|------|---------|----------|
| 1 | Taxicab/Manhattan | ∑\|xᵢ - yᵢ\| | Diamond-shaped balls |
| 2 | Euclidean | √(∑\|xᵢ - yᵢ\|²) | Spherical balls |
| ∞ | Chebyshev/Supremum | max\|xᵢ - yᵢ\| | Cubic balls |

$$d_\infty(x, y) = \max_{1 \leq i \leq n} |x_i - y_i|$$

**Theorem (Equivalence of Metrics on ℝⁿ):** For 1 ≤ p ≤ q ≤ ∞:

$$d_q(x, y) \leq d_p(x, y) \leq n^{1/p - 1/q} d_q(x, y)$$

All p-metrics on ℝⁿ are **equivalent**: they define the same convergent sequences and open sets.

---

## 3. Sequence Spaces l^p

### Definition: l^p Spaces

For 1 ≤ p < ∞, define:

$$\ell^p = \left\{ (x_n)_{n=1}^{\infty} : x_n \in \mathbb{C}, \sum_{n=1}^{\infty} |x_n|^p < \infty \right\}$$

With metric:

$$\boxed{d_p(x, y) = \left( \sum_{n=1}^{\infty} |x_n - y_n|^p \right)^{1/p}}$$

For p = ∞:

$$\ell^\infty = \left\{ (x_n)_{n=1}^{\infty} : \sup_{n} |x_n| < \infty \right\}$$

$$d_\infty(x, y) = \sup_{n \geq 1} |x_n - y_n|$$

### The Hilbert Space l²

The space l² is particularly important because it's a **Hilbert space**:

$$\ell^2 = \left\{ (x_n) : \sum_{n=1}^{\infty} |x_n|^2 < \infty \right\}$$

The metric comes from the inner product:

$$\langle x, y \rangle = \sum_{n=1}^{\infty} x_n \overline{y_n}$$

$$d(x, y) = \|x - y\|_2 = \sqrt{\langle x - y, x - y \rangle}$$

### Quantum Connection: Discrete Basis Expansion

In quantum mechanics, a state |ψ⟩ can be expanded in a discrete orthonormal basis {|n⟩}:

$$|\psi\rangle = \sum_{n=0}^{\infty} c_n |n\rangle$$

The coefficients (c₀, c₁, c₂, ...) form an element of l² with:

$$\|\psi\|^2 = \sum_{n=0}^{\infty} |c_n|^2 = 1$$

The metric distance between states:

$$d(|\psi\rangle, |\phi\rangle) = \sqrt{\sum_{n=0}^{\infty} |c_n - d_n|^2}$$

---

## 4. Function Spaces: C[a,b]

### The Space of Continuous Functions

$$C[a,b] = \{ f: [a,b] \to \mathbb{R} \text{ (or } \mathbb{C}) : f \text{ is continuous} \}$$

### The Supremum (Uniform) Metric

$$\boxed{d_\infty(f, g) = \sup_{x \in [a,b]} |f(x) - g(x)| = \|f - g\|_\infty}$$

**Verification:**

- **(M1)**: |f(x) - g(x)| ≥ 0, and sup = 0 iff f(x) = g(x) for all x.
- **(M2)**: |f(x) - g(x)| = |g(x) - f(x)|.
- **(M3)**: |f(x) - h(x)| ≤ |f(x) - g(x)| + |g(x) - h(x)|, taking sup over x.

### The Integral Metrics on C[a,b]

We can also define L^p-type metrics:

$$d_p(f, g) = \left( \int_a^b |f(x) - g(x)|^p \, dx \right)^{1/p}$$

For p = 2:

$$d_2(f, g) = \sqrt{\int_a^b |f(x) - g(x)|^2 \, dx}$$

**Important:** These give different topologies on C[a,b]!
- Uniform convergence (d_∞): fₙ → f uniformly ⟺ d_∞(fₙ, f) → 0
- L²-convergence (d₂): fₙ → f in L² ⟺ d₂(fₙ, f) → 0

Uniform convergence implies L²-convergence, but not conversely.

---

## 5. The L^p Spaces

### Definition: L^p Spaces

For a measure space (X, μ) and 1 ≤ p < ∞:

$$L^p(X, \mu) = \left\{ f: X \to \mathbb{C} \text{ measurable} : \int_X |f|^p \, d\mu < \infty \right\} / \sim$$

where f ~ g if f = g almost everywhere (μ-a.e.).

**The L^p metric:**

$$\boxed{d_p(f, g) = \|f - g\|_p = \left( \int_X |f - g|^p \, d\mu \right)^{1/p}}$$

### The Quantum Mechanics Space: L²(ℝⁿ)

The state space of quantum mechanics is L²(ℝⁿ, dx):

$$L^2(\mathbb{R}^n) = \left\{ \psi: \mathbb{R}^n \to \mathbb{C} : \int_{\mathbb{R}^n} |\psi(x)|^2 \, dx < \infty \right\}$$

**Physical interpretation:**

- |ψ(x)|² is the probability density
- ∫|ψ|² = 1 for normalized states
- d(ψ, φ) measures "how different" two quantum states are

### Key Inequalities for L^p Spaces

**Hölder's Inequality:** For 1/p + 1/q = 1:

$$\int |fg| \, d\mu \leq \|f\|_p \|g\|_q$$

**Minkowski's Inequality:** (Triangle inequality for L^p)

$$\|f + g\|_p \leq \|f\|_p + \|g\|_p$$

---

## Quantum Mechanics Connection: Distance Between States

In quantum mechanics, the L² metric has profound physical meaning.

### Fidelity and Distance

For normalized states ψ, φ ∈ L²:

$$d(\psi, \phi) = \|\psi - \phi\|_2 = \sqrt{\langle \psi - \phi | \psi - \phi \rangle}$$

Expanding:

$$d(\psi, \phi)^2 = \|\psi\|^2 + \|\phi\|^2 - 2\text{Re}\langle \psi | \phi \rangle = 2(1 - \text{Re}\langle \psi | \phi \rangle)$$

The quantity |⟨ψ|φ⟩|² is the **fidelity** or **overlap** between states.

### Orthogonal States are Maximally Distant

If ⟨ψ|φ⟩ = 0 (orthogonal states):

$$d(\psi, \phi) = \sqrt{2}$$

This is the maximum distance between normalized states.

### Physical Interpretation

- **d = 0**: States are identical (same physical state)
- **d = √2**: States are perfectly distinguishable (orthogonal)
- **0 < d < √2**: Partial overlap, quantum interference effects

---

## Worked Examples

### Example 1: Verifying the Discrete Metric

**Problem:** Show that for any non-empty set X, the **discrete metric**

$$d(x, y) = \begin{cases} 0 & \text{if } x = y \\ 1 & \text{if } x \neq y \end{cases}$$

is indeed a metric.

**Solution:**

**(M1) Positivity:**
- d(x, y) ∈ {0, 1} ⊂ [0, ∞) ✓
- d(x, y) = 0 ⟺ x = y by definition ✓

**(M2) Symmetry:**
- If x = y: d(x, y) = 0 = d(y, x) ✓
- If x ≠ y: d(x, y) = 1 = d(y, x) ✓

**(M3) Triangle Inequality:**
For any x, y, z ∈ X:
- If x = z: d(x, z) = 0 ≤ d(x, y) + d(y, z) ✓
- If x ≠ z: We need d(x, z) = 1 ≤ d(x, y) + d(y, z)
  - Either x ≠ y or y ≠ z (or both), so d(x,y) + d(y,z) ≥ 1 ✓

Therefore (X, d) is a metric space. ∎

### Example 2: The l² Distance Between Quantum States

**Problem:** The hydrogen atom eigenstates in the energy basis are |n⟩ for n = 1, 2, 3, ....
Consider the states:
- |ψ⟩ = (1/√2)|1⟩ + (1/2)|2⟩ + (1/2)|3⟩
- |φ⟩ = (1/√2)|1⟩ + (1/√2)|2⟩

Find d(|ψ⟩, |φ⟩) in the l² metric.

**Solution:**

The coefficient sequences are:
- ψ → (1/√2, 1/2, 1/2, 0, 0, ...)
- φ → (1/√2, 1/√2, 0, 0, 0, ...)

The difference:
- ψ - φ → (0, 1/2 - 1/√2, 1/2, 0, 0, ...)

Computing:

$$d(\psi, \phi)^2 = |0|^2 + |1/2 - 1/\sqrt{2}|^2 + |1/2|^2$$

$$= 0 + (1/2 - 1/\sqrt{2})^2 + 1/4$$

$$= (1/4 - 1/\sqrt{2} + 1/2) + 1/4$$

$$= 1 - \frac{1}{\sqrt{2}} \approx 1 - 0.707 = 0.293$$

$$\boxed{d(\psi, \phi) = \sqrt{1 - \frac{1}{\sqrt{2}}} \approx 0.541}$$

### Example 3: Supremum vs L² Metric

**Problem:** For f(x) = x and g(x) = x² on [0, 1], compute d_∞(f, g) and d₂(f, g).

**Solution:**

**Supremum metric:**

$$d_\infty(f, g) = \sup_{x \in [0,1]} |x - x^2| = \sup_{x \in [0,1]} x(1-x)$$

Taking derivative: d/dx[x(1-x)] = 1 - 2x = 0 at x = 1/2.

$$d_\infty(f, g) = \frac{1}{2} \cdot \frac{1}{2} = \boxed{\frac{1}{4}}$$

**L² metric:**

$$d_2(f, g)^2 = \int_0^1 (x - x^2)^2 \, dx = \int_0^1 (x^2 - 2x^3 + x^4) \, dx$$

$$= \left[ \frac{x^3}{3} - \frac{x^4}{2} + \frac{x^5}{5} \right]_0^1 = \frac{1}{3} - \frac{1}{2} + \frac{1}{5}$$

$$= \frac{10 - 15 + 6}{30} = \frac{1}{30}$$

$$\boxed{d_2(f, g) = \sqrt{\frac{1}{30}} \approx 0.183}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Taxicab metric:** For x = (1, 2, 3) and y = (4, 0, 1) in ℝ³, compute d₁(x, y), d₂(x, y), and d_∞(x, y).

2. **Discrete metric:** In the discrete metric on ℝ, what is the open ball B(0, 1/2)? What is B(0, 2)?

3. **l² norm:** Is the sequence xₙ = 1/n in l²? Is xₙ = 1/√n in l²?

### Level 2: Intermediate

4. **Not a metric:** Show that d(x, y) = (x - y)² is NOT a metric on ℝ. Which axiom fails?

5. **French railway metric:** On ℝ², define:
   $$d(x, y) = \begin{cases} \|x - y\|_2 & \text{if } x, y, 0 \text{ are collinear} \\ \|x\|_2 + \|y\|_2 & \text{otherwise} \end{cases}$$
   Verify this is a metric. (All trains go through Paris = origin!)

6. **Quantum fidelity:** Two qubit states have |⟨ψ|φ⟩|² = 0.75. Find d(ψ, φ).

### Level 3: Challenging

7. **Hölder continuity metric:** For 0 < α ≤ 1, define on C[0,1]:
   $$d_\alpha(f, g) = \|f - g\|_\infty + \sup_{x \neq y} \frac{|f(x) - g(x) - f(y) + g(y)|}{|x - y|^\alpha}$$
   Verify this is a metric.

8. **Ultrametric spaces:** A metric d is an **ultrametric** if d(x, z) ≤ max{d(x,y), d(y,z)}. Show that in an ultrametric space, every triangle is isoceles with the unique side being the shortest.

9. **L^p interpolation:** Prove that if f ∈ L^p ∩ L^r with 1 ≤ p < q < r ≤ ∞, then f ∈ L^q and:
   $$\|f\|_q \leq \|f\|_p^\theta \|f\|_r^{1-\theta}$$
   where 1/q = θ/p + (1-θ)/r.

---

## Computational Lab: Visualizing Metrics

```python
"""
Day 225: Metric Spaces - Visualization Lab
Exploring different metrics on R^2 and function spaces
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

# =============================================================================
# Part 1: Unit Balls in Different Metrics on R^2
# =============================================================================

def draw_unit_balls():
    """Visualize unit balls (circles) for different p-norms in R^2."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    theta = np.linspace(0, 2*np.pi, 1000)

    p_values = [1, 2, np.inf]
    titles = ['p = 1 (Taxicab)', 'p = 2 (Euclidean)', r'p = $\infty$ (Chebyshev)']

    for ax, p, title in zip(axes, p_values, titles):
        if p == 1:
            # Diamond: |x| + |y| = 1
            # Parametrize as: when theta in [0, pi/2]: x = 1-t, y = t for t in [0,1]
            x = np.cos(theta)
            y = np.sin(theta)
            r = 1 / (np.abs(x) + np.abs(y) + 1e-10)
            x, y = r * x, r * y
        elif p == 2:
            # Circle: x^2 + y^2 = 1
            x = np.cos(theta)
            y = np.sin(theta)
        else:  # p = infinity
            # Square: max(|x|, |y|) = 1
            x = np.cos(theta)
            y = np.sin(theta)
            r = 1 / np.maximum(np.abs(x), np.abs(y))
            x, y = r * x, r * y

        ax.fill(x, y, alpha=0.3, color='blue')
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.suptitle('Unit Balls in Different Metrics on $\\mathbb{R}^2$', fontsize=16)
    plt.tight_layout()
    plt.savefig('unit_balls.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 2: Distance Functions on Function Spaces
# =============================================================================

def compare_function_metrics():
    """Compare L^2 and L^infinity metrics on C[0,1]."""

    x = np.linspace(0, 1, 1000)

    # Define two functions
    f = x
    g = x**2

    # Compute metrics
    d_inf = np.max(np.abs(f - g))
    d_2 = np.sqrt(np.trapz((f - g)**2, x))
    d_1 = np.trapz(np.abs(f - g), x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: the functions
    ax1 = axes[0]
    ax1.plot(x, f, 'b-', linewidth=2, label='$f(x) = x$')
    ax1.plot(x, g, 'r-', linewidth=2, label='$g(x) = x^2$')
    ax1.fill_between(x, f, g, alpha=0.3, color='green', label='|f - g|')

    # Mark maximum difference
    idx_max = np.argmax(np.abs(f - g))
    ax1.plot([x[idx_max], x[idx_max]], [f[idx_max], g[idx_max]],
             'k-', linewidth=3, label=f'Max diff = {d_inf:.4f}')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Functions and Their Difference', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: comparison of metrics
    ax2 = axes[1]
    metrics = ['$d_1$ (L¹)', '$d_2$ (L²)', '$d_\\infty$ (sup)']
    values = [d_1, d_2, d_inf]
    colors = ['steelblue', 'darkorange', 'forestgreen']

    bars = ax2.bar(metrics, values, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Distance', fontsize=12)
    ax2.set_title('Comparison of Metrics on C[0,1]', fontsize=14)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)

    ax2.set_ylim(0, max(values) * 1.2)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('function_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Metrics between f(x)=x and g(x)=x² on [0,1]:")
    print(f"  L¹ metric (d₁):  {d_1:.6f}")
    print(f"  L² metric (d₂):  {d_2:.6f}")
    print(f"  Sup metric (d∞): {d_inf:.6f}")

# =============================================================================
# Part 3: l^2 Distance Between Quantum States
# =============================================================================

def quantum_state_distance():
    """Visualize l^2 distance between discrete quantum states."""

    # Number of basis states to consider
    n_basis = 10

    # Define two quantum states (coefficients in energy basis)
    # State 1: Ground state heavy superposition
    psi = np.zeros(n_basis, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[1] = 1/2
    psi[2] = 1/2

    # State 2: First excited state heavy
    phi = np.zeros(n_basis, dtype=complex)
    phi[0] = 1/np.sqrt(2)
    phi[1] = 1/np.sqrt(2)

    # Compute distance
    diff = psi - phi
    distance = np.sqrt(np.sum(np.abs(diff)**2))

    # Compute overlap (fidelity)
    overlap = np.abs(np.sum(psi.conj() * phi))**2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    n = np.arange(n_basis)
    width = 0.35

    # Left: |psi|^2
    ax1 = axes[0]
    ax1.bar(n, np.abs(psi)**2, color='blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Basis state |n⟩', fontsize=12)
    ax1.set_ylabel('$|c_n|^2$', fontsize=12)
    ax1.set_title('State $|\\psi\\rangle$ probability distribution', fontsize=14)
    ax1.set_xticks(n)
    ax1.grid(True, alpha=0.3, axis='y')

    # Middle: |phi|^2
    ax2 = axes[1]
    ax2.bar(n, np.abs(phi)**2, color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Basis state |n⟩', fontsize=12)
    ax2.set_ylabel('$|d_n|^2$', fontsize=12)
    ax2.set_title('State $|\\phi\\rangle$ probability distribution', fontsize=14)
    ax2.set_xticks(n)
    ax2.grid(True, alpha=0.3, axis='y')

    # Right: Comparison
    ax3 = axes[2]
    ax3.bar(n - width/2, np.abs(psi)**2, width, label='$|\\psi\\rangle$',
            color='blue', alpha=0.7, edgecolor='black')
    ax3.bar(n + width/2, np.abs(phi)**2, width, label='$|\\phi\\rangle$',
            color='red', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Basis state |n⟩', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title(f'Comparison: d = {distance:.4f}, Fidelity = {overlap:.4f}', fontsize=14)
    ax3.set_xticks(n)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('quantum_state_distance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nQuantum State Metrics:")
    print(f"  l² distance d(ψ,φ) = {distance:.6f}")
    print(f"  Fidelity |⟨ψ|φ⟩|² = {overlap:.6f}")
    print(f"  Theoretical: d² = 2(1 - Re⟨ψ|φ⟩) = {2*(1 - np.real(np.sum(psi.conj()*phi))):.6f}")

# =============================================================================
# Part 4: The French Railway Metric
# =============================================================================

def french_railway_metric():
    """Visualize the French railway (SNCF) metric - all paths go through origin."""

    def d_sncf(p1, p2):
        """French railway metric: all trains go through Paris (origin)."""
        # Check if points are collinear with origin
        x1, y1 = p1
        x2, y2 = p2

        # Cross product to check collinearity
        cross = x1 * y2 - x2 * y1

        if np.abs(cross) < 1e-10:  # Collinear with origin
            return np.sqrt((x2-x1)**2 + (y2-y1)**2)
        else:  # Must go through origin
            return np.sqrt(x1**2 + y1**2) + np.sqrt(x2**2 + y2**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Example 1: Non-collinear points
    ax1 = axes[0]
    A = np.array([2, 1])
    B = np.array([-1, 2])
    O = np.array([0, 0])

    d_eucl = np.sqrt(np.sum((A - B)**2))
    d_sncf_val = d_sncf(A, B)

    # Plot points
    ax1.plot(*A, 'bo', markersize=12, label=f'A = ({A[0]}, {A[1]})')
    ax1.plot(*B, 'ro', markersize=12, label=f'B = ({B[0]}, {B[1]})')
    ax1.plot(0, 0, 'ko', markersize=12, label='Paris (origin)')

    # Direct path (Euclidean)
    ax1.plot([A[0], B[0]], [A[1], B[1]], 'g--', linewidth=2,
             label=f'Direct: d₂ = {d_eucl:.2f}')

    # SNCF path (through origin)
    ax1.plot([A[0], 0], [A[1], 0], 'b-', linewidth=3)
    ax1.plot([0, B[0]], [0, B[1]], 'r-', linewidth=3)
    ax1.annotate('', xy=O, xytext=A,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.annotate('', xy=B, xytext=O,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax1.set_xlim(-3, 4)
    ax1.set_ylim(-1, 4)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.legend(loc='upper right')
    ax1.set_title(f'Non-collinear: SNCF distance = {d_sncf_val:.2f}', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Example 2: Collinear points
    ax2 = axes[1]
    C = np.array([1, 2])
    D = np.array([2, 4])  # Collinear with origin

    d_eucl2 = np.sqrt(np.sum((C - D)**2))
    d_sncf_val2 = d_sncf(C, D)

    ax2.plot(*C, 'bo', markersize=12, label=f'C = ({C[0]}, {C[1]})')
    ax2.plot(*D, 'go', markersize=12, label=f'D = ({D[0]}, {D[1]})')
    ax2.plot(0, 0, 'ko', markersize=12, label='Paris (origin)')

    # Direct path (same as SNCF for collinear)
    ax2.plot([C[0], D[0]], [C[1], D[1]], 'purple', linewidth=3,
             label=f'Direct = SNCF = {d_sncf_val2:.2f}')

    # Line through origin
    t = np.linspace(-0.5, 3, 100)
    ax2.plot(t, 2*t, 'k--', alpha=0.3, label='Line through origin')

    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 6)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.legend(loc='upper left')
    ax2.set_title(f'Collinear: SNCF = Euclidean = {d_sncf_val2:.2f}', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.suptitle('The French Railway (SNCF) Metric', fontsize=16)
    plt.tight_layout()
    plt.savefig('sncf_metric.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Run All Visualizations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 225: Metric Spaces - Visualization Lab")
    print("=" * 60)

    print("\n1. Drawing unit balls for different p-norms...")
    draw_unit_balls()

    print("\n2. Comparing function space metrics...")
    compare_function_metrics()

    print("\n3. Quantum state distance in l²...")
    quantum_state_distance()

    print("\n4. The French Railway metric...")
    french_railway_metric()

    print("\n" + "=" * 60)
    print("Lab complete! Images saved.")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Metric axioms | d(x,y) ≥ 0, d(x,y)=0⟺x=y; d(x,y)=d(y,x); d(x,z)≤d(x,y)+d(y,z) |
| Euclidean metric | $d_2(x,y) = \sqrt{\sum_i \|x_i - y_i\|^2}$ |
| l^p metric | $d_p(x,y) = \left(\sum_n \|x_n - y_n\|^p\right)^{1/p}$ |
| Supremum metric | $d_\infty(f,g) = \sup_x \|f(x) - g(x)\|$ |
| L^p metric | $d_p(f,g) = \left(\int \|f-g\|^p d\mu\right)^{1/p}$ |
| Discrete metric | d(x,y) = 0 if x=y, 1 if x≠y |
| QM state distance | $d(\psi,\phi)^2 = 2(1 - \text{Re}\langle\psi\|\phi\rangle)$ |

### Main Takeaways

1. **Metric spaces generalize distance** from ℝⁿ to abstract settings
2. **Three axioms suffice**: positivity, symmetry, triangle inequality
3. **Many equivalent metrics** can exist on the same set (e.g., all p-norms on ℝⁿ)
4. **Function spaces** (C[a,b], L^p) are natural metric spaces
5. **L² is the QM state space** with distance measuring state distinguishability

---

## Daily Checklist

- [ ] I can state and verify all three metric axioms
- [ ] I can compute distances in ℝⁿ with different p-metrics
- [ ] I understand l^p spaces and their metrics
- [ ] I can work with supremum and L² metrics on function spaces
- [ ] I can explain why L² distance matters for quantum states
- [ ] I completed the computational lab

---

## Preview: Day 226

Tomorrow we explore **convergence and continuity** in metric spaces. We'll see how:
- Sequences converge when distances go to zero
- Open balls define the topology
- Continuity has an elegant ε-δ reformulation
- These concepts directly apply to quantum mechanical operators
