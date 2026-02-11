# Day 233: Inner Product Spaces

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Inner products, Cauchy-Schwarz, induced norms |
| Afternoon | 3 hours | Problems: Polarization identity, parallelogram law |
| Evening | 2 hours | Computational lab: Geometry of inner product spaces |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** an inner product and verify the inner product axioms
2. **Prove** the Cauchy-Schwarz inequality and understand its geometric meaning
3. **Show** that every inner product induces a norm
4. **State and apply** the polarization identity
5. **Prove** the parallelogram law and understand why it characterizes inner product spaces
6. **Connect** inner products to quantum mechanical probability amplitudes

---

## 1. Core Content: Inner Product Spaces

### 1.1 Definition of Inner Product

An **inner product** on a vector space $$V$$ over $$\mathbb{C}$$ is a function $$\langle\cdot,\cdot\rangle: V \times V \to \mathbb{C}$$ satisfying:

$$\boxed{\begin{aligned}
&\text{(IP1) Conjugate Symmetry:} && \langle x, y\rangle = \overline{\langle y, x\rangle} \\
&\text{(IP2) Linearity (1st arg):} && \langle\alpha x + \beta y, z\rangle = \alpha\langle x, z\rangle + \beta\langle y, z\rangle \\
&\text{(IP3) Positive Definiteness:} && \langle x, x\rangle \geq 0, \text{ and } \langle x, x\rangle = 0 \Leftrightarrow x = 0
\end{aligned}}$$

**Convention Note**: We use the "physics convention" where the inner product is linear in the **first** argument and conjugate-linear in the second. Mathematicians often use the opposite convention.

From (IP1) and (IP2), we get **conjugate-linearity in the second argument**:
$$\langle x, \alpha y + \beta z\rangle = \bar{\alpha}\langle x, y\rangle + \bar{\beta}\langle x, z\rangle$$

An **inner product space** (or **pre-Hilbert space**) is a vector space equipped with an inner product.

### 1.2 Real Inner Products

For a real vector space ($$\mathbb{F} = \mathbb{R}$$), the conjugate symmetry becomes ordinary symmetry:
$$\langle x, y\rangle = \langle y, x\rangle$$

and the inner product is bilinear (linear in both arguments).

### 1.3 Fundamental Examples

**Example 1: Standard Inner Product on $$\mathbb{C}^n$$**

$$\langle x, y\rangle = \sum_{i=1}^n x_i \overline{y_i} = x^* y$$

where $$x^* = \bar{x}^T$$ is the conjugate transpose.

**Example 2: Inner Product on $$\ell^2$$**

For sequences $$x = (x_n)$$ and $$y = (y_n)$$ in $$\ell^2$$:
$$\langle x, y\rangle = \sum_{n=1}^\infty x_n \overline{y_n}$$

This sum converges by the Cauchy-Schwarz inequality (which we'll prove shortly).

**Example 3: Inner Product on $$L^2[a,b]$$**

For functions in $$L^2[a,b]$$:
$$\langle f, g\rangle = \int_a^b f(x) \overline{g(x)} \, dx$$

**Example 4: Weighted Inner Product**

Given a positive weight function $$w(x) > 0$$:
$$\langle f, g\rangle_w = \int_a^b f(x) \overline{g(x)} w(x) \, dx$$

---

## 2. The Cauchy-Schwarz Inequality

### 2.1 Statement and Proof

**Theorem (Cauchy-Schwarz Inequality)**:
For any $$x, y$$ in an inner product space:
$$\boxed{|\langle x, y\rangle| \leq \|x\| \|y\|}$$

where $$\|x\| = \sqrt{\langle x, x\rangle}$$.

Equality holds if and only if $$x$$ and $$y$$ are linearly dependent.

**Proof**:

If $$y = 0$$, both sides are zero and the inequality holds trivially.

Assume $$y \neq 0$$. For any $$\lambda \in \mathbb{C}$$:
$$0 \leq \langle x - \lambda y, x - \lambda y\rangle = \|x\|^2 - \lambda\langle y, x\rangle - \bar{\lambda}\langle x, y\rangle + |\lambda|^2 \|y\|^2$$

Choose $$\lambda = \frac{\langle x, y\rangle}{\|y\|^2}$$ (this minimizes the expression):

$$0 \leq \|x\|^2 - \frac{\langle x, y\rangle \overline{\langle x, y\rangle}}{\|y\|^2} - \frac{\overline{\langle x, y\rangle}\langle x, y\rangle}{\|y\|^2} + \frac{|\langle x, y\rangle|^2}{\|y\|^4}\|y\|^2$$

$$0 \leq \|x\|^2 - \frac{|\langle x, y\rangle|^2}{\|y\|^2} - \frac{|\langle x, y\rangle|^2}{\|y\|^2} + \frac{|\langle x, y\rangle|^2}{\|y\|^2}$$

$$0 \leq \|x\|^2 - \frac{|\langle x, y\rangle|^2}{\|y\|^2}$$

Therefore:
$$|\langle x, y\rangle|^2 \leq \|x\|^2 \|y\|^2$$

Taking square roots gives the result. $$\square$$

### 2.2 Geometric Interpretation

In real inner product spaces, we can define the **angle** between vectors:
$$\cos\theta = \frac{\langle x, y\rangle}{\|x\| \|y\|}$$

Cauchy-Schwarz guarantees $$|\cos\theta| \leq 1$$, so $$\theta$$ is well-defined.

**Orthogonality**: $$x \perp y$$ ($$x$$ is orthogonal to $$y$$) if $$\langle x, y\rangle = 0$$.

---

## 3. Induced Norm and Triangle Inequality

### 3.1 Every Inner Product Induces a Norm

**Theorem**: If $$\langle\cdot,\cdot\rangle$$ is an inner product on $$V$$, then
$$\|x\| = \sqrt{\langle x, x\rangle}$$
defines a norm on $$V$$.

**Proof**:

**(N1) Positivity**: $$\|x\| = \sqrt{\langle x, x\rangle} \geq 0$$ by (IP3), and $$\|x\| = 0 \Leftrightarrow \langle x, x\rangle = 0 \Leftrightarrow x = 0$$. $$\checkmark$$

**(N2) Homogeneity**:
$$\|\alpha x\| = \sqrt{\langle\alpha x, \alpha x\rangle} = \sqrt{\alpha \bar{\alpha} \langle x, x\rangle} = \sqrt{|\alpha|^2}\sqrt{\langle x, x\rangle} = |\alpha| \|x\| \checkmark$$

**(N3) Triangle Inequality**:
$$\begin{aligned}
\|x + y\|^2 &= \langle x+y, x+y\rangle \\
&= \langle x, x\rangle + \langle x, y\rangle + \langle y, x\rangle + \langle y, y\rangle \\
&= \|x\|^2 + 2\text{Re}\langle x, y\rangle + \|y\|^2 \\
&\leq \|x\|^2 + 2|\langle x, y\rangle| + \|y\|^2 \\
&\leq \|x\|^2 + 2\|x\|\|y\| + \|y\|^2 \quad \text{(Cauchy-Schwarz)} \\
&= (\|x\| + \|y\|)^2
\end{aligned}$$

Taking square roots: $$\|x + y\| \leq \|x\| + \|y\|$$. $$\square$$

---

## 4. The Polarization Identity

### 4.1 Recovering the Inner Product from the Norm

The **polarization identity** shows that the inner product is completely determined by its induced norm.

**Theorem (Polarization Identity)**: In a complex inner product space:
$$\boxed{\langle x, y\rangle = \frac{1}{4}\left(\|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2\right)}$$

In a real inner product space:
$$\boxed{\langle x, y\rangle = \frac{1}{4}\left(\|x+y\|^2 - \|x-y\|^2\right)}$$

**Proof (Complex Case)**:

Compute each term:
$$\|x + y\|^2 = \|x\|^2 + \|y\|^2 + \langle x, y\rangle + \langle y, x\rangle = \|x\|^2 + \|y\|^2 + 2\text{Re}\langle x, y\rangle$$

$$\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2\text{Re}\langle x, y\rangle$$

$$\|x + iy\|^2 = \|x\|^2 + \|y\|^2 + 2\text{Re}\langle x, iy\rangle = \|x\|^2 + \|y\|^2 + 2\text{Re}(-i\langle x, y\rangle)$$
$$= \|x\|^2 + \|y\|^2 + 2\text{Im}\langle x, y\rangle$$

$$\|x - iy\|^2 = \|x\|^2 + \|y\|^2 - 2\text{Im}\langle x, y\rangle$$

Therefore:
$$\|x+y\|^2 - \|x-y\|^2 = 4\text{Re}\langle x, y\rangle$$
$$\|x+iy\|^2 - \|x-iy\|^2 = 4\text{Im}\langle x, y\rangle$$

Combining:
$$\frac{1}{4}(\|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2) = \text{Re}\langle x, y\rangle + i\text{Im}\langle x, y\rangle = \langle x, y\rangle$$ $$\square$$

---

## 5. The Parallelogram Law

### 5.1 Statement and Proof

**Theorem (Parallelogram Law)**: In any inner product space:
$$\boxed{\|x + y\|^2 + \|x - y\|^2 = 2(\|x\|^2 + \|y\|^2)}$$

**Proof**:
$$\|x + y\|^2 + \|x - y\|^2 = (\|x\|^2 + \|y\|^2 + 2\text{Re}\langle x, y\rangle) + (\|x\|^2 + \|y\|^2 - 2\text{Re}\langle x, y\rangle)$$
$$= 2\|x\|^2 + 2\|y\|^2$$ $$\square$$

### 5.2 Geometric Meaning

In a parallelogram with sides $$x$$ and $$y$$:
- Diagonals are $$x + y$$ and $$x - y$$
- The law states: sum of squares of diagonals = twice sum of squares of sides

### 5.3 Characterization Theorem

**Theorem (Jordan-von Neumann)**: A norm on a vector space comes from an inner product if and only if it satisfies the parallelogram law.

This is remarkable: the parallelogram law is both necessary AND sufficient for a norm to be induced by an inner product!

**Counterexample**: The $$\ell^1$$ norm does NOT satisfy the parallelogram law.

Take $$x = (1, 0)$$ and $$y = (0, 1)$$ in $$\ell^1$$:
- $$\|x + y\|_1 = \|(1, 1)\|_1 = 2$$
- $$\|x - y\|_1 = \|(1, -1)\|_1 = 2$$
- $$\|x\|_1 = 1$$, $$\|y\|_1 = 1$$

Check: $$2^2 + 2^2 = 8 \neq 4 = 2(1^2 + 1^2)$$

Therefore, $$\ell^1$$ is NOT an inner product space.

---

## 6. Quantum Mechanics Connection

### 6.1 Inner Products as Probability Amplitudes

In quantum mechanics, the inner product $$\langle\phi|\psi\rangle$$ is the **probability amplitude** for transitioning from state $$|\psi\rangle$$ to state $$|\phi\rangle$$:

$$P(\phi|\psi) = |\langle\phi|\psi\rangle|^2$$

The Cauchy-Schwarz inequality immediately gives:
$$P(\phi|\psi) = |\langle\phi|\psi\rangle|^2 \leq \|\phi\|^2 \|\psi\|^2 = 1$$

for normalized states, ensuring probabilities are at most 1.

### 6.2 Orthogonality as Distinguishability

Two states $$|\phi\rangle$$ and $$|\psi\rangle$$ are **perfectly distinguishable** (can be distinguished with certainty by some measurement) if and only if:
$$\langle\phi|\psi\rangle = 0$$

This is orthogonality in the Hilbert space sense.

### 6.3 The Born Rule Connection

The normalization condition $$\langle\psi|\psi\rangle = 1$$ ensures total probability equals 1.

For a state $$|\psi\rangle = \sum_n c_n |n\rangle$$ expanded in an orthonormal basis:
$$\langle\psi|\psi\rangle = \sum_n |c_n|^2 = 1$$

Each $$|c_n|^2$$ is the probability of measuring outcome $$n$$.

### 6.4 Dirac Notation Review

| Mathematical Notation | Dirac Notation |
|----------------------|----------------|
| $$\psi \in \mathcal{H}$$ | $$\|\psi\rangle$$ (ket) |
| $$\phi^* \in \mathcal{H}^*$$ | $$\langle\phi\|$$ (bra) |
| $$\langle\phi, \psi\rangle$$ | $$\langle\phi\|\psi\rangle$$ (bracket) |
| $$\|\psi\|^2$$ | $$\langle\psi\|\psi\rangle$$ |

---

## 7. Worked Examples

### Example 1: Verifying an Inner Product

**Problem**: Verify that $$\langle f, g\rangle = \int_0^{2\pi} f(x)\overline{g(x)} \, dx$$ defines an inner product on $$C[0, 2\pi]$$.

**Solution**:

**(IP1) Conjugate Symmetry**:
$$\overline{\langle g, f\rangle} = \overline{\int_0^{2\pi} g(x)\overline{f(x)} \, dx} = \int_0^{2\pi} \overline{g(x)} f(x) \, dx = \int_0^{2\pi} f(x)\overline{g(x)} \, dx = \langle f, g\rangle \checkmark$$

**(IP2) Linearity in First Argument**:
$$\langle\alpha f + \beta g, h\rangle = \int_0^{2\pi} (\alpha f(x) + \beta g(x))\overline{h(x)} \, dx$$
$$= \alpha\int_0^{2\pi} f(x)\overline{h(x)} \, dx + \beta\int_0^{2\pi} g(x)\overline{h(x)} \, dx = \alpha\langle f, h\rangle + \beta\langle g, h\rangle \checkmark$$

**(IP3) Positive Definiteness**:
$$\langle f, f\rangle = \int_0^{2\pi} |f(x)|^2 \, dx \geq 0$$

If $$\langle f, f\rangle = 0$$, then $$\int_0^{2\pi} |f(x)|^2 \, dx = 0$$.

Since $$|f|^2$$ is continuous and non-negative, this implies $$f(x) = 0$$ for all $$x$$. $$\checkmark$$

Therefore, $$\langle\cdot,\cdot\rangle$$ is an inner product on $$C[0, 2\pi]$$. $$\square$$

---

### Example 2: Using Cauchy-Schwarz

**Problem**: Prove that for any $$a_1, \ldots, a_n, b_1, \ldots, b_n \in \mathbb{R}$$:
$$\left(\sum_{i=1}^n a_i b_i\right)^2 \leq \left(\sum_{i=1}^n a_i^2\right)\left(\sum_{i=1}^n b_i^2\right)$$

**Solution**:

Consider $$\mathbb{R}^n$$ with the standard inner product $$\langle x, y\rangle = \sum_i x_i y_i$$.

Let $$x = (a_1, \ldots, a_n)$$ and $$y = (b_1, \ldots, b_n)$$.

By Cauchy-Schwarz:
$$|\langle x, y\rangle| \leq \|x\| \|y\|$$

$$\left|\sum_{i=1}^n a_i b_i\right| \leq \sqrt{\sum_{i=1}^n a_i^2} \cdot \sqrt{\sum_{i=1}^n b_i^2}$$

Squaring both sides:
$$\left(\sum_{i=1}^n a_i b_i\right)^2 \leq \left(\sum_{i=1}^n a_i^2\right)\left(\sum_{i=1}^n b_i^2\right)$$ $$\square$$

---

### Example 3: Polarization Identity Application

**Problem**: Let $$\|x\| = 2$$, $$\|y\| = 3$$, and $$\|x + y\| = 4$$ in a real inner product space. Find $$\langle x, y\rangle$$.

**Solution**:

In a real inner product space:
$$\|x + y\|^2 = \|x\|^2 + 2\langle x, y\rangle + \|y\|^2$$

Substituting:
$$16 = 4 + 2\langle x, y\rangle + 9$$

$$2\langle x, y\rangle = 16 - 13 = 3$$

$$\langle x, y\rangle = \frac{3}{2}$$

**Verification**: Using the polarization identity:
$$\langle x, y\rangle = \frac{1}{4}(\|x+y\|^2 - \|x-y\|^2)$$

We need $$\|x - y\|^2 = \|x\|^2 - 2\langle x, y\rangle + \|y\|^2 = 4 - 3 + 9 = 10$$.

So $$\|x - y\| = \sqrt{10}$$.

Check: $$\frac{1}{4}(16 - 10) = \frac{6}{4} = \frac{3}{2}$$ $$\checkmark$$ $$\square$$

---

## 8. Practice Problems

### Level 1: Direct Application

1. Verify that $$\langle x, y\rangle = \sum_{n=1}^\infty x_n \overline{y_n}$$ is an inner product on $$\ell^2$$.

2. For $$x = (1, 2, 3)$$ and $$y = (2, -1, 1)$$ in $$\mathbb{R}^3$$ with the standard inner product:
   - Compute $$\langle x, y\rangle$$
   - Find $$\|x\|$$ and $$\|y\|$$
   - Verify Cauchy-Schwarz directly
   - Find the angle between $$x$$ and $$y$$

3. Check whether the parallelogram law holds for $$x = (1, 0)$$ and $$y = (0, 1)$$ in:
   - $$(\mathbb{R}^2, \|\cdot\|_2)$$
   - $$(\mathbb{R}^2, \|\cdot\|_1)$$
   - $$(\mathbb{R}^2, \|\cdot\|_\infty)$$

### Level 2: Intermediate

4. Let $$\langle f, g\rangle = \int_{-1}^1 f(x)\overline{g(x)}(1-x^2) \, dx$$. Show this is a weighted inner product on $$C[-1,1]$$.

5. Prove that in any inner product space:
   $$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2$$
   implies
   $$\|x + y\|^2 - \|x - y\|^2 = 4\text{Re}\langle x, y\rangle$$

6. **Quantum Problem**: Two qubits are in states:
   $$|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |\phi\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

   Show they are orthogonal and interpret this physically.

### Level 3: Challenging

7. Prove that if $$\|x\| = \|y\|$$ and $$x \perp y$$, then $$\|x + y\| = \|x - y\|$$.

8. **(Jordan-von Neumann)**: Given a norm satisfying the parallelogram law, define
   $$\langle x, y\rangle = \frac{1}{4}(\|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2)$$
   Prove this is an inner product. (Show it satisfies all three axioms.)

9. Let $$V$$ be a finite-dimensional inner product space. Prove that every linear functional $$f: V \to \mathbb{C}$$ can be written as $$f(x) = \langle x, y\rangle$$ for some unique $$y \in V$$.

---

## 9. Computational Lab: Geometry of Inner Product Spaces

```python
"""
Day 233 Computational Lab: Inner Product Spaces
Exploring geometry, Cauchy-Schwarz, and polarization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ============================================================
# Part 1: Verifying Cauchy-Schwarz Inequality
# ============================================================

def verify_cauchy_schwarz():
    """
    Numerically verify Cauchy-Schwarz for random vectors.
    """
    np.random.seed(42)
    n_trials = 1000
    dimensions = [2, 10, 100]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, dim in zip(axes, dimensions):
        # Generate random complex vectors
        x = np.random.randn(n_trials, dim) + 1j * np.random.randn(n_trials, dim)
        y = np.random.randn(n_trials, dim) + 1j * np.random.randn(n_trials, dim)

        # Compute |<x,y>| and ||x|| ||y||
        inner_prods = np.abs(np.sum(x * np.conj(y), axis=1))
        norm_prods = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)

        # Plot
        ax.scatter(norm_prods, inner_prods, alpha=0.3, s=10)
        max_val = np.max(norm_prods) * 1.1
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='$|⟨x,y⟩| = \|x\|\|y\|$')
        ax.set_xlabel(r'$\|x\| \|y\|$')
        ax.set_ylabel(r'$|⟨x,y⟩|$')
        ax.set_title(f'Cauchy-Schwarz in $\\mathbb{{C}}^{{{dim}}}$')
        ax.legend()
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

        # Count violations (should be zero!)
        violations = np.sum(inner_prods > norm_prods + 1e-10)
        print(f"Dimension {dim}: {violations} violations out of {n_trials}")

    plt.tight_layout()
    plt.savefig('cauchy_schwarz_verification.png', dpi=150)
    plt.show()

# ============================================================
# Part 2: Polarization Identity Verification
# ============================================================

def verify_polarization():
    """
    Verify the polarization identity numerically.
    """
    np.random.seed(123)

    # Random complex vectors
    x = np.array([1+2j, 3-1j, 2+1j])
    y = np.array([2-1j, 1+1j, -1+2j])

    # Direct inner product
    inner_direct = np.sum(x * np.conj(y))

    # Polarization formula
    def norm_sq(v):
        return np.sum(np.abs(v)**2)

    term1 = norm_sq(x + y)
    term2 = norm_sq(x - y)
    term3 = norm_sq(x + 1j*y)
    term4 = norm_sq(x - 1j*y)

    inner_polar = (term1 - term2 + 1j*term3 - 1j*term4) / 4

    print("Polarization Identity Verification")
    print("=" * 40)
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"\nDirect computation: ⟨x,y⟩ = {inner_direct:.6f}")
    print(f"Polarization formula: {inner_polar:.6f}")
    print(f"Difference: {abs(inner_direct - inner_polar):.2e}")

# ============================================================
# Part 3: Parallelogram Law and Norm Characterization
# ============================================================

def check_parallelogram_law():
    """
    Check which norms satisfy the parallelogram law.
    """
    np.random.seed(456)
    n_trials = 100

    def p_norm(v, p):
        if p == np.inf:
            return np.max(np.abs(v))
        return np.sum(np.abs(v)**p)**(1/p)

    p_values = [1, 1.5, 2, 3, 4, np.inf]
    violations = []

    print("\nParallelogram Law Check")
    print("=" * 50)
    print("Law: ||x+y||^2 + ||x-y||^2 = 2(||x||^2 + ||y||^2)")
    print()

    for p in p_values:
        max_error = 0
        for _ in range(n_trials):
            x = np.random.randn(5) + 1j * np.random.randn(5)
            y = np.random.randn(5) + 1j * np.random.randn(5)

            lhs = p_norm(x+y, p)**2 + p_norm(x-y, p)**2
            rhs = 2 * (p_norm(x, p)**2 + p_norm(y, p)**2)

            error = abs(lhs - rhs) / rhs
            max_error = max(max_error, error)

        violations.append(max_error)
        p_str = '∞' if p == np.inf else str(p)
        status = "SATISFIES" if max_error < 1e-10 else "VIOLATES"
        print(f"p = {p_str:>4}: Max relative error = {max_error:.2e} [{status}]")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    p_labels = ['1', '1.5', '2', '3', '4', '∞']
    colors = ['red' if v > 1e-10 else 'green' for v in violations]
    bars = ax.bar(p_labels, violations, color=colors, alpha=0.7)
    ax.axhline(y=1e-10, color='gray', linestyle='--', label='Tolerance')
    ax.set_yscale('log')
    ax.set_xlabel('$p$-norm')
    ax.set_ylabel('Max Relative Error')
    ax.set_title('Parallelogram Law Violations by $p$-Norm\n(Only $p=2$ satisfies the law)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('parallelogram_law.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: Quantum States and Inner Products
# ============================================================

def quantum_inner_products():
    """
    Demonstrate inner products for quantum states.
    """
    # Computational basis
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)

    # Superposition states
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    ket_minus = (ket_0 - ket_1) / np.sqrt(2)

    # Circular polarization states
    ket_R = (ket_0 + 1j*ket_1) / np.sqrt(2)
    ket_L = (ket_0 - 1j*ket_1) / np.sqrt(2)

    states = {
        '|0⟩': ket_0,
        '|1⟩': ket_1,
        '|+⟩': ket_plus,
        '|-⟩': ket_minus,
        '|R⟩': ket_R,
        '|L⟩': ket_L
    }

    print("\nQuantum State Inner Products")
    print("=" * 60)

    # Compute inner product matrix
    names = list(states.keys())
    n = len(names)
    inner_matrix = np.zeros((n, n), dtype=complex)

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            inner_matrix[i, j] = np.vdot(states[name2], states[name1])  # ⟨name1|name2⟩

    # Display as probability matrix
    prob_matrix = np.abs(inner_matrix)**2

    print("\nProbability Matrix |⟨ψ|φ⟩|²:")
    print("(Probability of measuring state ψ given system is in state φ)")
    print()

    # Print header
    print("        ", end="")
    for name in names:
        print(f"{name:>8}", end="")
    print()

    for i, name1 in enumerate(names):
        print(f"{name1:>8}", end="")
        for j in range(n):
            print(f"{prob_matrix[i,j]:>8.3f}", end="")
        print()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Inner product magnitudes
    im1 = axes[0].imshow(np.abs(inner_matrix), cmap='Blues', vmin=0, vmax=1)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(names)
    axes[0].set_yticklabels(names)
    axes[0].set_title(r'$|⟨\psi|\phi⟩|$ (Amplitude Magnitudes)')
    plt.colorbar(im1, ax=axes[0])

    # Annotate
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f'{np.abs(inner_matrix[i,j]):.2f}',
                        ha='center', va='center', fontsize=9)

    # Probabilities
    im2 = axes[1].imshow(prob_matrix, cmap='Oranges', vmin=0, vmax=1)
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(names)
    axes[1].set_yticklabels(names)
    axes[1].set_title(r'$|⟨\psi|\phi⟩|^2$ (Transition Probabilities)')
    plt.colorbar(im2, ax=axes[1])

    # Annotate
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f'{prob_matrix[i,j]:.2f}',
                        ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('quantum_inner_products.png', dpi=150)
    plt.show()

# ============================================================
# Part 5: Function Space Inner Product
# ============================================================

def function_space_inner_product():
    """
    Demonstrate inner products in L^2[0, 2π].
    """
    x = np.linspace(0, 2*np.pi, 1000)

    # Define some functions
    f1 = np.sin(x)
    f2 = np.cos(x)
    f3 = np.sin(2*x)
    f4 = np.ones_like(x) / np.sqrt(2*np.pi)  # Normalized constant

    functions = {
        r'$\sin(x)$': f1,
        r'$\cos(x)$': f2,
        r'$\sin(2x)$': f3,
        r'$1/\sqrt{2\pi}$': f4
    }

    # Compute inner products (using numerical integration)
    print("\nFunction Space Inner Products on L²[0, 2π]")
    print("=" * 50)

    names = list(functions.keys())
    n = len(names)
    inner_matrix = np.zeros((n, n))

    for i, (name1, f_i) in enumerate(functions.items()):
        for j, (name2, f_j) in enumerate(functions.items()):
            # ⟨f_i, f_j⟩ = ∫ f_i(x) f_j(x) dx
            inner_matrix[i, j] = np.trapz(f_i * f_j, x)

    # Display
    print("\nInner Product Matrix ⟨f, g⟩:")
    print()
    print("                    ", end="")
    for name in names:
        print(f"{name:>15}", end="")
    print()

    for i, name1 in enumerate(names):
        print(f"{name1:>20}", end="")
        for j in range(n):
            print(f"{inner_matrix[i,j]:>15.4f}", end="")
        print()

    # Note: sin and cos are orthogonal, sin(x) and sin(2x) are orthogonal
    print("\nKey observations:")
    print("- ⟨sin(x), cos(x)⟩ ≈ 0  (orthogonal)")
    print("- ⟨sin(x), sin(2x)⟩ ≈ 0  (orthogonal)")
    print("- ⟨sin(x), sin(x)⟩ = π  (norm squared)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, f in functions.items():
        ax.plot(x, f, linewidth=2, label=name)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Functions in $L^2[0, 2\\pi]$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('function_space.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 233: Inner Product Spaces - Computational Lab")
    print("=" * 60)

    print("\n1. Verifying Cauchy-Schwarz inequality...")
    verify_cauchy_schwarz()

    print("\n2. Verifying polarization identity...")
    verify_polarization()

    print("\n3. Checking parallelogram law for different norms...")
    check_parallelogram_law()

    print("\n4. Quantum state inner products...")
    quantum_inner_products()

    print("\n5. Function space inner products...")
    function_space_inner_product()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## 10. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Inner Product** | Sesquilinear form satisfying conjugate symmetry, linearity, positive definiteness |
| **Inner Product Space** | Vector space equipped with an inner product (pre-Hilbert space) |
| **Induced Norm** | $$\|x\| = \sqrt{\langle x, x\rangle}$$ |
| **Orthogonality** | $$x \perp y \Leftrightarrow \langle x, y\rangle = 0$$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{Cauchy-Schwarz:} && |\langle x, y\rangle| \leq \|x\| \|y\| \\
&\text{Parallelogram Law:} && \|x+y\|^2 + \|x-y\|^2 = 2\|x\|^2 + 2\|y\|^2 \\
&\text{Polarization (Complex):} && \langle x,y\rangle = \tfrac{1}{4}(\|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2) \\
&\text{Polarization (Real):} && \langle x,y\rangle = \tfrac{1}{4}(\|x+y\|^2 - \|x-y\|^2)
\end{aligned}}$$

### Key Insights

1. **Inner products add geometry** to vector spaces (angles, orthogonality)
2. **Cauchy-Schwarz** is the most important inequality in analysis
3. **Polarization identity** shows norm determines inner product completely
4. **Parallelogram law characterizes** which norms come from inner products
5. **Quantum amplitudes** are inner products; **probabilities** are their squares

---

## 11. Daily Checklist

- [ ] I can define an inner product and verify the three axioms
- [ ] I can prove the Cauchy-Schwarz inequality
- [ ] I understand why every inner product induces a norm
- [ ] I can use the polarization identity to recover the inner product from the norm
- [ ] I know the parallelogram law and why it characterizes inner product spaces
- [ ] I can connect inner products to quantum probability amplitudes
- [ ] I completed the computational lab exercises

---

## 12. Preview: Day 234

Tomorrow we define **Hilbert spaces** as complete inner product spaces and study the canonical example $$L^2$$. We'll prove the Riesz-Fischer theorem ($$L^2$$ is complete) and understand why $$L^2$$ is the natural home for quantum mechanical wave functions. The connection $$\int |\psi(x)|^2 dx = 1$$ will become rigorous.

---

*"The Cauchy-Schwarz inequality is the most useful and most important inequality in all of mathematics."* — J. Michael Steele
