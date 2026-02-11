# Day 228: The Banach Fixed-Point Theorem

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning I** | 2 hours | Contraction mappings and the fixed-point theorem |
| **Morning II** | 1.5 hours | Proof of the Banach fixed-point theorem |
| **Afternoon I** | 2 hours | Applications: ODEs, integral equations |
| **Afternoon II** | 1.5 hours | Worked examples and practice problems |
| **Evening** | 1 hour | Computational lab: fixed-point iteration |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** contraction mappings and verify the contraction property
2. **State and prove** the Banach fixed-point theorem
3. **Apply** fixed-point iteration to solve equations
4. **Understand** convergence rates and error bounds
5. **Use** the theorem to prove existence/uniqueness for differential equations
6. **Connect** fixed-point methods to self-consistent field calculations in QM

---

## 1. Contraction Mappings

### Definition

Let (X, d) be a metric space. A function T: X → X is a **contraction** (or **contraction mapping**) if there exists a constant 0 ≤ k < 1 such that:

$$\boxed{d(Tx, Ty) \leq k \cdot d(x, y) \quad \text{for all } x, y \in X}$$

The constant k is called the **contraction constant** or **Lipschitz constant**.

### Key Properties

**Theorem:** Every contraction is continuous.

**Proof:** Contractions are Lipschitz with constant k < 1, hence uniformly continuous. ∎

**Intuition:** A contraction "shrinks" distances. If you have two points, their images under T are strictly closer together than the original points (by a factor of at least k).

### Examples

**Example 1:** T(x) = x/2 on ℝ is a contraction with k = 1/2.
$$|T(x) - T(y)| = |x/2 - y/2| = \frac{1}{2}|x - y|$$

**Example 2:** T(x) = cos(x) on [0, 1] is a contraction.
$$|T(x) - T(y)| = |\cos x - \cos y| \leq \sup_{z \in [0,1]} |\sin z| \cdot |x - y| \leq \sin(1) |x - y|$$
Since sin(1) ≈ 0.841 < 1, this is a contraction.

**Example 3:** T(x) = 2x is NOT a contraction (k = 2 > 1).

---

## 2. The Banach Fixed-Point Theorem

### Statement

**Theorem (Banach Fixed-Point / Contraction Mapping Theorem):**

Let (X, d) be a **complete** metric space and T: X → X a **contraction** with constant k < 1. Then:

1. **Existence:** T has a fixed point x* ∈ X (i.e., T(x*) = x*)
2. **Uniqueness:** The fixed point is unique
3. **Convergence:** For any x₀ ∈ X, the sequence xₙ₊₁ = T(xₙ) converges to x*
4. **Error estimate:**
$$\boxed{d(x_n, x^*) \leq \frac{k^n}{1-k} d(x_0, x_1)}$$

### The Complete Proof

**Uniqueness:** Suppose x* and y* are both fixed points. Then:
$$d(x^*, y^*) = d(Tx^*, Ty^*) \leq k \cdot d(x^*, y^*)$$
Since k < 1, this implies d(x*, y*) = 0, so x* = y*. ∎

**Existence and Convergence:**

Start with any x₀ ∈ X and define xₙ₊₁ = T(xₙ).

*Step 1: The sequence (xₙ) is Cauchy.*

First, observe:
$$d(x_{n+1}, x_n) = d(Tx_n, Tx_{n-1}) \leq k \cdot d(x_n, x_{n-1})$$

By induction:
$$d(x_{n+1}, x_n) \leq k^n d(x_1, x_0)$$

For m > n:
$$d(x_m, x_n) \leq d(x_m, x_{m-1}) + d(x_{m-1}, x_{m-2}) + \cdots + d(x_{n+1}, x_n)$$
$$\leq (k^{m-1} + k^{m-2} + \cdots + k^n) d(x_1, x_0)$$
$$= k^n \frac{1 - k^{m-n}}{1-k} d(x_1, x_0)$$
$$\leq \frac{k^n}{1-k} d(x_1, x_0)$$

As n → ∞, kⁿ → 0, so (xₙ) is Cauchy.

*Step 2: The limit exists and is a fixed point.*

Since X is complete, xₙ → x* for some x* ∈ X.

Since T is continuous:
$$T(x^*) = T(\lim x_n) = \lim T(x_n) = \lim x_{n+1} = x^*$$

So x* is a fixed point. ∎

### Geometric Interpretation

The iteration xₙ₊₁ = T(xₙ) is like repeatedly applying T. The contraction property ensures:
- Each iteration brings us closer to the fixed point
- The "errors" decrease geometrically: O(kⁿ)

---

## 3. Error Estimates and Convergence Rate

### A Priori Estimate

Before running the iteration:
$$\boxed{d(x_n, x^*) \leq \frac{k^n}{1-k} d(x_0, x_1)}$$

### A Posteriori Estimate

After computing xₙ and xₙ₊₁:
$$\boxed{d(x_n, x^*) \leq \frac{k}{1-k} d(x_n, x_{n-1})}$$

### Convergence Rate

The convergence is **linear** with rate k:
$$d(x_{n+1}, x^*) \leq k \cdot d(x_n, x^*)$$

Smaller k means faster convergence. When k is close to 0, convergence is very fast.

---

## 4. Applications

### Application 1: Solving Equations

To solve f(x) = 0, rewrite as x = g(x) for some g, then apply fixed-point iteration.

**Example:** Solve x³ + x - 1 = 0.

Rewrite as x = (1 - x³)/1 = 1 - x³... but this isn't a contraction on a suitable interval.

Better: x = (1 - x)/x² isn't great either.

Try: x³ = 1 - x, so x = (1 - x)^{1/3}. Define g(x) = (1 - x)^{1/3}.

On [0, 1]: |g'(x)| = |-(1/3)(1-x)^{-2/3}|. At x = 0: |g'(0)| = 1/3 < 1 ✓

So g is a contraction near the root, and iteration converges.

### Application 2: Integral Equations

Consider the Fredholm integral equation:
$$f(x) = g(x) + \lambda \int_a^b K(x, y) f(y) \, dy$$

Define the operator T on C[a, b]:
$$(Tf)(x) = g(x) + \lambda \int_a^b K(x, y) f(y) \, dy$$

**Theorem:** If |λ| · sup_{x} ∫|K(x,y)|dy < 1, then T is a contraction and the integral equation has a unique solution.

### Application 3: Picard's Theorem (ODE Existence)

Consider the initial value problem:
$$\frac{dy}{dx} = f(x, y), \quad y(x_0) = y_0$$

Convert to integral equation:
$$y(x) = y_0 + \int_{x_0}^{x} f(t, y(t)) \, dt$$

Define the Picard operator:
$$(Ty)(x) = y_0 + \int_{x_0}^{x} f(t, y(t)) \, dt$$

**Theorem (Picard-Lindelöf):** If f is Lipschitz in y:
$$|f(x, y_1) - f(x, y_2)| \leq L|y_1 - y_2|$$
then on a sufficiently small interval [x₀ - h, x₀ + h], T is a contraction and the ODE has a unique local solution.

---

## Quantum Mechanics Connection: Self-Consistent Field Theory

### The Hartree-Fock Method

In quantum mechanics, many-electron systems are solved using **self-consistent field (SCF)** methods. The idea:

1. Guess initial orbitals φ₁⁽⁰⁾, φ₂⁽⁰⁾, ...
2. Compute the effective potential from these orbitals
3. Solve for new orbitals φ₁⁽¹⁾, φ₂⁽¹⁾, ...
4. Repeat until convergence

This is exactly fixed-point iteration! The "operator" maps orbitals to new orbitals.

### The SCF Iteration

For a two-electron system (simplified):
$$\left[-\frac{\hbar^2}{2m}\nabla^2 + V_{ext}(r) + V_{ee}[\phi](r)\right]\phi(r) = E\phi(r)$$

The electron-electron term V_{ee}[φ] depends on φ itself:
$$V_{ee}[\phi](r) = \int \frac{|\phi(r')|^2}{|r - r'|} d^3r'$$

Define T(φ) = "solve Schrödinger equation with potential V_{ext} + V_{ee}[φ]".

If T is a contraction (or can be made into one via mixing), the SCF iteration converges to the self-consistent solution.

### Mixing for Convergence

Often, direct iteration T(φₙ) = φₙ₊₁ doesn't converge. A common fix is **linear mixing**:
$$\phi_{n+1} = (1 - \alpha)\phi_n + \alpha \cdot T(\phi_n)$$

Choosing α appropriately can make the mixed iteration a contraction even when T isn't.

### Density Functional Theory (DFT)

In DFT, one iterates on the electron density ρ(r):
$$\rho_{n+1} = F(\rho_n)$$

where F involves solving Kohn-Sham equations. The Banach fixed-point theorem (and its generalizations) provide the theoretical foundation for convergence of these calculations.

---

## Worked Examples

### Example 1: Fixed-Point Iteration for √2

**Problem:** Use fixed-point iteration to find √2 by solving x² = 2.

**Solution:**

Rewrite as x = g(x) for various choices:

**Attempt 1:** x = 2/x

This has fixed points at ±√2, but |g'(x)| = 2/x² = 1 at x = √2. Not a contraction!

**Attempt 2:** x = (x + 2/x)/2 (Newton's method variant)

Let g(x) = (x + 2/x)/2. Then:
$$g'(x) = \frac{1}{2}\left(1 - \frac{2}{x^2}\right)$$

At x = √2: g'(√2) = (1 - 1)/2 = 0! This is superlinear convergence.

Near √2, |g'(x)| < 1, so it's a local contraction.

**Iteration:**
- x₀ = 1
- x₁ = (1 + 2)/2 = 1.5
- x₂ = (1.5 + 2/1.5)/2 = 1.4166...
- x₃ = 1.41421568...
- x₄ = 1.41421356... ≈ √2

$$\boxed{\sqrt{2} \approx 1.41421356}$$

### Example 2: Solving an Integral Equation

**Problem:** Solve f(x) = x + (1/4)∫₀¹ xy f(y) dy on C[0, 1].

**Solution:**

Define (Tf)(x) = x + (1/4) x ∫₀¹ y f(y) dy.

Let c = ∫₀¹ y f(y) dy. Then (Tf)(x) = x(1 + c/4).

**Check contraction:**
$$\|(Tf - Tg)\|_\infty = \frac{1}{4}\|x\|_\infty \left|\int_0^1 y(f(y) - g(y))dy\right|$$
$$\leq \frac{1}{4} \cdot 1 \cdot \int_0^1 y \, dy \cdot \|f - g\|_\infty = \frac{1}{4} \cdot \frac{1}{2} \|f - g\|_\infty = \frac{1}{8}\|f - g\|_\infty$$

So T is a contraction with k = 1/8.

**Find fixed point:** f* = Tf* means f*(x) = x + (c*/4)x where c* = ∫₀¹ y f*(y) dy.

So f*(x) = x(1 + c*/4). Then:
$$c^* = \int_0^1 y \cdot y(1 + c^*/4) dy = (1 + c^*/4) \int_0^1 y^2 dy = (1 + c^*/4) \cdot \frac{1}{3}$$

Solving: c* = 1/3 + c*/12, so 11c*/12 = 1/3, giving c* = 4/11.

$$\boxed{f^*(x) = x\left(1 + \frac{1}{11}\right) = \frac{12x}{11}}$$

### Example 3: Picard Iteration for dy/dx = y

**Problem:** Solve dy/dx = y with y(0) = 1 using Picard iteration.

**Solution:**

The integral form is:
$$y(x) = 1 + \int_0^x y(t) \, dt$$

**Iteration:**
- y₀(x) = 1
- y₁(x) = 1 + ∫₀ˣ 1 dt = 1 + x
- y₂(x) = 1 + ∫₀ˣ (1 + t) dt = 1 + x + x²/2
- y₃(x) = 1 + ∫₀ˣ (1 + t + t²/2) dt = 1 + x + x²/2 + x³/6

By induction:
$$y_n(x) = \sum_{k=0}^{n} \frac{x^k}{k!}$$

Taking n → ∞:
$$\boxed{y(x) = \sum_{k=0}^{\infty} \frac{x^k}{k!} = e^x}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Contraction check:** Is T(x) = (1/3)cos(x) + 1 a contraction on [0, 2]? Find the contraction constant.

2. **Fixed-point iteration:** Starting from x₀ = 0, compute x₁, x₂, x₃ for T(x) = cos(x). Does it converge?

3. **Error estimate:** If T is a contraction with k = 0.5 and d(x₀, x₁) = 1, bound the distance d(x₅, x*).

### Level 2: Intermediate

4. **Root finding:** Solve x = e⁻ˣ using fixed-point iteration. Show convergence and find the root to 4 decimal places.

5. **Integral equation:** Solve f(x) = 1 + (1/3)∫₀¹ f(y) dy on C[0, 1] using the Banach fixed-point theorem.

6. **ODE existence:** For dy/dx = x² + y² with y(0) = 0, show the Picard operator is a contraction on a small interval and find y₁(x), y₂(x).

### Level 3: Challenging

7. **Nonlinear system:** Solve the system:
   - x = (1/3)sin(y) + 1
   - y = (1/4)cos(x)

   Show it's a contraction on an appropriate set and find the fixed point.

8. **Partial contraction:** Suppose T: X → X satisfies d(T²x, T²y) ≤ k·d(x, y) for k < 1 (T² is a contraction but T may not be). Prove T has a unique fixed point.

9. **Parameter dependence:** For the equation x = λsin(x) with λ ∈ (0, 1), analyze how the fixed point x*(λ) depends on λ.

---

## Computational Lab: Fixed-Point Iteration

```python
"""
Day 228: Banach Fixed-Point Theorem - Computational Lab
Exploring fixed-point iteration and applications
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint

# =============================================================================
# Part 1: Basic Fixed-Point Iteration
# =============================================================================

def fixed_point_iteration_basic():
    """Demonstrate basic fixed-point iteration for cos(x) = x."""

    def g(x):
        return np.cos(x)

    # Iteration
    x0 = 0.5
    n_iter = 20
    x_values = [x0]

    for _ in range(n_iter):
        x_values.append(g(x_values[-1]))

    x_values = np.array(x_values)

    # The fixed point (Dottie number)
    x_star = 0.7390851332151607

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Cobweb diagram
    ax1 = axes[0]
    x = np.linspace(0, 1.5, 1000)
    ax1.plot(x, g(x), 'b-', linewidth=2, label='$g(x) = \\cos(x)$')
    ax1.plot(x, x, 'k--', linewidth=1, label='$y = x$')
    ax1.plot(x_star, x_star, 'r*', markersize=15, label=f'Fixed point ≈ {x_star:.4f}')

    # Cobweb
    for i in range(min(10, len(x_values)-1)):
        xi = x_values[i]
        yi = g(xi)
        # Vertical line to curve
        ax1.plot([xi, xi], [xi, yi], 'g-', linewidth=1, alpha=0.7)
        # Horizontal line to diagonal
        ax1.plot([xi, yi], [yi, yi], 'g-', linewidth=1, alpha=0.7)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Cobweb Diagram: $x_{n+1} = \\cos(x_n)$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 1.5)
    ax1.set_aspect('equal')

    # Right: Convergence
    ax2 = axes[1]
    errors = np.abs(x_values - x_star)
    ax2.semilogy(range(len(errors)), errors, 'bo-', markersize=6, linewidth=1)

    # Theoretical convergence: |g'(x*)| = sin(x*) ≈ 0.6736
    k = np.sin(x_star)
    theoretical = errors[0] * k**np.arange(len(errors))
    ax2.semilogy(range(len(errors)), theoretical, 'r--', linewidth=2,
                 label=f'Theoretical: $k = \\sin(x^*) \\approx {k:.4f}$')

    ax2.set_xlabel('Iteration n', fontsize=12)
    ax2.set_ylabel('$|x_n - x^*|$', fontsize=12)
    ax2.set_title('Convergence Rate', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fixed_point_basic.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Fixed point of cos(x) = x: x* ≈ {x_star}")
    print(f"Contraction constant: k = sin(x*) ≈ {k:.6f}")

# =============================================================================
# Part 2: Newton's Method as Fixed-Point Iteration
# =============================================================================

def newton_fixed_point():
    """Compare regular fixed-point with Newton's method for sqrt(2)."""

    # Method 1: x = 2/x (not a contraction at sqrt(2))
    def g1(x):
        return 2/x

    # Method 2: Newton for x^2 - 2 = 0 -> x = (x + 2/x)/2
    def g2(x):
        return (x + 2/x) / 2

    x_star = np.sqrt(2)
    x0 = 1.5
    n_iter = 10

    # Run both methods
    x1_values = [x0]
    x2_values = [x0]

    for _ in range(n_iter):
        x1_values.append(g1(x1_values[-1]))
        x2_values.append(g2(x2_values[-1]))

    x1_values = np.array(x1_values)
    x2_values = np.array(x2_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Iteration values
    ax1 = axes[0]
    ax1.plot(range(len(x1_values)), x1_values, 'ro-', markersize=8,
             linewidth=1, label='$g_1(x) = 2/x$ (oscillates)')
    ax1.plot(range(len(x2_values)), x2_values, 'bo-', markersize=8,
             linewidth=1, label='$g_2(x) = (x + 2/x)/2$ (Newton)')
    ax1.axhline(y=x_star, color='k', linestyle='--', label=f'$\\sqrt{{2}} \\approx {x_star:.6f}$')

    ax1.set_xlabel('Iteration n', fontsize=12)
    ax1.set_ylabel('$x_n$', fontsize=12)
    ax1.set_title('Two Methods for $\\sqrt{2}$', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Errors (log scale)
    ax2 = axes[1]
    err1 = np.abs(x1_values - x_star)
    err2 = np.abs(x2_values - x_star)

    ax2.semilogy(range(len(err1)), err1, 'ro-', markersize=8,
                 linewidth=1, label='$g_1$ (constant error)')
    ax2.semilogy(range(len(err2)), err2 + 1e-20, 'bo-', markersize=8,
                 linewidth=1, label='$g_2$ (quadratic convergence)')

    ax2.set_xlabel('Iteration n', fontsize=12)
    ax2.set_ylabel('$|x_n - \\sqrt{2}|$', fontsize=12)
    ax2.set_title('Convergence Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('newton_vs_naive.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nMethod 1 (x = 2/x): Oscillates between x and 2/x")
    print("Method 2 (Newton): Quadratic convergence")
    print(f"\nNewton iterations:")
    for i, x in enumerate(x2_values[:8]):
        print(f"  x_{i} = {x:.15f}")

# =============================================================================
# Part 3: Picard Iteration for ODE
# =============================================================================

def picard_iteration():
    """Demonstrate Picard iteration for dy/dx = y, y(0) = 1."""

    # Exact solution
    def exact(x):
        return np.exp(x)

    # Picard iterates: y_n(x) = sum_{k=0}^n x^k/k!
    def picard_n(x, n):
        return sum(x**k / np.math.factorial(k) for k in range(n+1))

    x = np.linspace(0, 2, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Picard iterates
    ax1 = axes[0]
    ax1.plot(x, exact(x), 'k-', linewidth=3, label='$e^x$ (exact)')

    for n in [0, 1, 2, 3, 5, 10]:
        y_n = np.array([picard_n(xi, n) for xi in x])
        ax1.plot(x, y_n, '--', linewidth=1.5, label=f'$y_{n}(x)$')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title("Picard Iteration: $y' = y$, $y(0) = 1$", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 8)

    # Right: Error at x = 1
    ax2 = axes[1]
    n_values = range(0, 20)
    errors = [abs(picard_n(1, n) - np.e) for n in n_values]

    ax2.semilogy(list(n_values), errors, 'bo-', markersize=8, linewidth=1)
    ax2.set_xlabel('Number of Picard iterations', fontsize=12)
    ax2.set_ylabel('$|y_n(1) - e|$', fontsize=12)
    ax2.set_title('Convergence at $x = 1$', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('picard_iteration.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Picard iteration for dy/dx = y, y(0) = 1:")
    print("-" * 40)
    for n in range(8):
        print(f"y_{n}(x) = " + " + ".join([f"x^{k}/{k}!" for k in range(n+1)]))

# =============================================================================
# Part 4: Self-Consistent Field (SCF) Iteration
# =============================================================================

def scf_iteration():
    """Demonstrate SCF-like iteration for a simplified quantum problem."""

    # Simplified 1D problem: solve Schrödinger with self-interaction
    # [-d²/dx² + V_ext + λ|ψ|²]ψ = Eψ on [0, π]
    # V_ext = 0, boundary ψ(0) = ψ(π) = 0

    # We'll use a finite difference approach
    N = 100
    L = np.pi
    dx = L / (N + 1)
    x = np.linspace(dx, L - dx, N)

    # Kinetic energy matrix (finite difference)
    T = (-2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)) / dx**2
    T = -T  # Make it positive (we have -d²/dx²)

    # External potential (harmonic-like for demo)
    V_ext = 0.5 * (x - L/2)**2

    # Nonlinearity strength
    lam = 2.0

    def scf_step(psi_in):
        """One SCF step: solve with current potential."""
        # Effective potential
        V_eff = V_ext + lam * np.abs(psi_in)**2

        # Hamiltonian
        H = T + np.diag(V_eff)

        # Solve eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # Ground state
        psi_out = eigenvectors[:, 0]

        # Normalize
        psi_out = psi_out / np.sqrt(np.trapz(psi_out**2, x))

        # Make sure sign is consistent
        if np.sum(psi_out) < 0:
            psi_out = -psi_out

        return psi_out, eigenvalues[0]

    # Initial guess: ground state of particle in a box
    psi = np.sin(np.pi * x / L)
    psi = psi / np.sqrt(np.trapz(psi**2, x))

    # Run SCF
    n_iter = 50
    mixing = 0.3  # Linear mixing for stability
    energies = []
    psi_history = [psi.copy()]

    for i in range(n_iter):
        psi_new, E = scf_step(psi)
        energies.append(E)

        # Mixing
        psi = (1 - mixing) * psi + mixing * psi_new
        psi = psi / np.sqrt(np.trapz(psi**2, x))

        psi_history.append(psi.copy())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Wave function evolution
    ax1 = axes[0]
    for i in [0, 1, 2, 5, 10, -1]:
        label = f'Iter {i}' if i >= 0 else 'Final'
        ax1.plot(x, psi_history[i], linewidth=2, alpha=0.7, label=label)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('$\\psi(x)$', fontsize=12)
    ax1.set_title('SCF Iteration: Wave Function', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Energy convergence
    ax2 = axes[1]
    ax2.plot(range(len(energies)), energies, 'bo-', markersize=4, linewidth=1)
    ax2.axhline(y=energies[-1], color='r', linestyle='--',
                label=f'Converged E ≈ {energies[-1]:.4f}')
    ax2.set_xlabel('SCF Iteration', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Energy Convergence', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scf_iteration.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSCF converged energy: E = {energies[-1]:.6f}")
    print(f"Linear mixing parameter: α = {mixing}")

# =============================================================================
# Part 5: Contraction Constant Visualization
# =============================================================================

def contraction_visualization():
    """Visualize how contraction constant affects convergence."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Define contractions with different k
    def make_contraction(k):
        return lambda x: k * x

    k_values = [0.2, 0.5, 0.8, 0.95]
    x0 = 1.0
    n_iter = 30

    # Left: Iteration values
    ax1 = axes[0]
    for k in k_values:
        g = make_contraction(k)
        x_values = [x0]
        for _ in range(n_iter):
            x_values.append(g(x_values[-1]))
        ax1.plot(range(len(x_values)), x_values, 'o-', markersize=4,
                 linewidth=1, label=f'k = {k}')

    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Fixed point = 0')
    ax1.set_xlabel('Iteration n', fontsize=12)
    ax1.set_ylabel('$x_n$', fontsize=12)
    ax1.set_title('Effect of Contraction Constant', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Error bound
    ax2 = axes[1]
    n = np.arange(0, 20)

    for k in k_values:
        # Error bound: |x_n - x*| <= k^n / (1-k) * |x_0 - x_1|
        # With g(x) = kx, x* = 0, |x_0 - x_1| = |1 - k| = 1 - k
        bound = k**n / (1 - k) * (1 - k)  # = k^n
        actual = k**n * x0  # Actual error since x* = 0

        ax2.semilogy(n, bound, 'o-', markersize=4, linewidth=1, label=f'k = {k}')

    ax2.set_xlabel('Iteration n', fontsize=12)
    ax2.set_ylabel('Error bound $k^n$', fontsize=12)
    ax2.set_title('Geometric Convergence Rate', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('contraction_constant.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nEffect of contraction constant k:")
    print("-" * 40)
    for k in k_values:
        n_for_0001 = int(np.ceil(np.log(0.001) / np.log(k)))
        print(f"k = {k}: Need ~{n_for_0001} iterations for error < 0.001")

# =============================================================================
# Run All Visualizations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 228: Banach Fixed-Point Theorem - Lab")
    print("=" * 60)

    print("\n1. Basic fixed-point iteration (cos(x) = x)...")
    fixed_point_iteration_basic()

    print("\n2. Newton's method as fixed-point...")
    newton_fixed_point()

    print("\n3. Picard iteration for ODEs...")
    picard_iteration()

    print("\n4. Self-consistent field iteration...")
    scf_iteration()

    print("\n5. Effect of contraction constant...")
    contraction_visualization()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Contraction | $d(Tx, Ty) \leq k \cdot d(x, y)$ with $k < 1$ |
| Fixed point | $T(x^*) = x^*$ |
| Iteration | $x_{n+1} = T(x_n)$ |
| A priori error | $d(x_n, x^*) \leq \frac{k^n}{1-k} d(x_0, x_1)$ |
| A posteriori error | $d(x_n, x^*) \leq \frac{k}{1-k} d(x_n, x_{n-1})$ |
| Picard operator | $(Ty)(x) = y_0 + \int_{x_0}^x f(t, y(t)) dt$ |

### The Banach Fixed-Point Theorem

**Requirements:**
1. Complete metric space (X, d)
2. Contraction mapping T: X → X with constant k < 1

**Conclusions:**
1. Unique fixed point x* exists
2. xₙ₊₁ = T(xₙ) converges to x* from any starting point
3. Convergence is geometric with rate k

### Main Takeaways

1. **Contractions shrink distances** — this forces convergence
2. **Completeness is essential** — limits must exist in the space
3. **The theorem gives both existence and an algorithm**
4. **Applications everywhere**: ODEs, integral equations, optimization
5. **SCF methods in QM** are fixed-point iterations with mixing

---

## Daily Checklist

- [ ] I can verify that a function is a contraction
- [ ] I can state and prove the Banach fixed-point theorem
- [ ] I understand error estimates and convergence rates
- [ ] I can apply fixed-point iteration to solve equations
- [ ] I understand the connection to self-consistent field methods
- [ ] I completed the computational lab

---

## Preview: Day 229

Tomorrow we explore **compactness** in metric spaces:
- Definitions: sequential compactness, total boundedness
- The Arzelà-Ascoli theorem for function spaces
- Compact operators in quantum mechanics
- Why compact sets are "almost finite-dimensional"
