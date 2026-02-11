# Day 232: Normed Spaces and Banach Spaces

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Norms, normed spaces, Banach spaces |
| Afternoon | 3 hours | Problems: $$\ell^p$$ spaces, completeness proofs |
| Evening | 2 hours | Computational lab: Visualizing norms and convergence |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** a norm and verify the norm axioms for given examples
2. **Explain** why normed spaces generalize the concept of length to infinite dimensions
3. **Prove** that every normed space is a metric space
4. **Define** Banach spaces and explain the significance of completeness
5. **Work with** the $$\ell^p$$ sequence spaces, especially $$\ell^1$$, $$\ell^2$$, and $$\ell^\infty$$
6. **Connect** normed spaces to the structure of quantum state spaces

---

## 1. Core Content: Normed Spaces

### 1.1 Definition of a Norm

A **norm** on a vector space $$V$$ over $$\mathbb{F}$$ (where $$\mathbb{F} = \mathbb{R}$$ or $$\mathbb{C}$$) is a function $$\|\cdot\|: V \to [0,\infty)$$ satisfying:

$$\boxed{\begin{aligned}
&\text{(N1) Positivity:} && \|x\| \geq 0, \text{ and } \|x\| = 0 \Leftrightarrow x = 0 \\
&\text{(N2) Homogeneity:} && \|\alpha x\| = |\alpha| \|x\| \text{ for all } \alpha \in \mathbb{F} \\
&\text{(N3) Triangle Inequality:} && \|x + y\| \leq \|x\| + \|y\|
\end{aligned}}$$

A **normed space** (or **normed vector space**) is a pair $$(V, \|\cdot\|)$$ where $$V$$ is a vector space and $$\|\cdot\|$$ is a norm on $$V$$.

### 1.2 Fundamental Examples

**Example 1: Euclidean Norm on $$\mathbb{R}^n$$**

$$\|x\|_2 = \sqrt{\sum_{i=1}^n |x_i|^2}$$

This is the familiar "length" of a vector.

**Example 2: $$p$$-Norms on $$\mathbb{R}^n$$**

For $$1 \leq p < \infty$$:
$$\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

For $$p = \infty$$:
$$\|x\|_\infty = \max_{1 \leq i \leq n} |x_i|$$

**Example 3: Supremum Norm on $$C[a,b]$$**

For continuous functions on $$[a,b]$$:
$$\|f\|_\infty = \sup_{x \in [a,b]} |f(x)| = \max_{x \in [a,b]} |f(x)|$$

### 1.3 Every Normed Space is a Metric Space

**Theorem**: If $$(V, \|\cdot\|)$$ is a normed space, then $$d(x,y) = \|x - y\|$$ defines a metric on $$V$$.

**Proof**:
We verify the metric axioms:

1. **(M1)** $$d(x,y) = \|x-y\| \geq 0$$ by (N1), and $$d(x,y) = 0 \Leftrightarrow \|x-y\| = 0 \Leftrightarrow x - y = 0 \Leftrightarrow x = y$$.

2. **(M2)** $$d(x,y) = \|x-y\| = \|(-1)(y-x)\| = |-1|\|y-x\| = \|y-x\| = d(y,x)$$ by (N2).

3. **(M3)** $$d(x,z) = \|x-z\| = \|(x-y) + (y-z)\| \leq \|x-y\| + \|y-z\| = d(x,y) + d(y,z)$$ by (N3). $$\square$$

**Important**: The converse is false! Not every metric comes from a norm. A metric $$d$$ comes from a norm if and only if it satisfies:
- Translation invariance: $$d(x+z, y+z) = d(x,y)$$
- Absolute homogeneity: $$d(\alpha x, \alpha y) = |\alpha| d(x,y)$$

---

## 2. Banach Spaces

### 2.1 Definition

A **Banach space** is a complete normed space—that is, a normed space in which every Cauchy sequence converges.

**Definition (Cauchy Sequence in Normed Space)**: A sequence $$(x_n)$$ in a normed space is **Cauchy** if:
$$\forall \varepsilon > 0, \exists N \in \mathbb{N}: m, n \geq N \Rightarrow \|x_n - x_m\| < \varepsilon$$

**Definition (Completeness)**: A normed space is **complete** if every Cauchy sequence converges to an element of the space.

$$\boxed{\text{Banach Space} = \text{Complete Normed Space}}$$

### 2.2 Why Completeness Matters

Completeness ensures that:
1. Limits of "reasonable" sequences stay in the space
2. Infinite series $$\sum_{n=1}^\infty x_n$$ can converge
3. Fixed point theorems apply
4. The space is "closed under limits"

**Physical Intuition**: In quantum mechanics, we need limits of states to remain valid states. If we approximate a physical state by a sequence of simpler states, the limit must exist in our state space.

### 2.3 The $$\ell^p$$ Sequence Spaces

For $$1 \leq p < \infty$$, define:
$$\ell^p = \left\{(x_n)_{n=1}^\infty : \sum_{n=1}^\infty |x_n|^p < \infty\right\}$$

with norm:
$$\|(x_n)\|_p = \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p}$$

For $$p = \infty$$:
$$\ell^\infty = \left\{(x_n)_{n=1}^\infty : \sup_n |x_n| < \infty\right\}$$

with norm:
$$\|(x_n)\|_\infty = \sup_{n \geq 1} |x_n|$$

**Theorem**: For $$1 \leq p \leq \infty$$, $$\ell^p$$ is a Banach space.

### 2.4 Key Inequality: Minkowski's Inequality

The triangle inequality for $$\ell^p$$ is **Minkowski's inequality**:

$$\boxed{\|x + y\|_p \leq \|x\|_p + \|y\|_p}$$

For $$1 < p < \infty$$, this follows from **Hölder's inequality**:

$$\boxed{\sum_{n=1}^\infty |x_n y_n| \leq \|x\|_p \|y\|_q \quad \text{where } \frac{1}{p} + \frac{1}{q} = 1}$$

---

## 3. Quantum Mechanics Connection

### 3.1 Why Quantum Mechanics Needs Banach/Hilbert Spaces

In quantum mechanics, the state of a system is described by a vector $$|\psi\rangle$$ in a Hilbert space (a special type of Banach space). The key requirements are:

1. **Superposition**: States can be added ($$|\psi\rangle + |\phi\rangle$$) and scaled ($$\alpha|\psi\rangle$$) — this requires a **vector space**

2. **Probability Interpretation**: $$|\langle\phi|\psi\rangle|^2$$ gives probabilities — this requires an **inner product**, hence a **norm**

3. **Physical Limits**: Sequences of approximating states must converge to valid states — this requires **completeness**

### 3.2 The Space $$\ell^2$$ in Quantum Mechanics

The space $$\ell^2$$ appears naturally when we have a countable orthonormal basis $$\{|n\rangle\}_{n=0}^\infty$$ (like energy eigenstates of the harmonic oscillator):

$$|\psi\rangle = \sum_{n=0}^\infty c_n |n\rangle$$

The normalization condition becomes:
$$\langle\psi|\psi\rangle = \sum_{n=0}^\infty |c_n|^2 = 1$$

This means the coefficient sequence $$(c_n) \in \ell^2$$ with $$\|(c_n)\|_2 = 1$$.

### 3.3 Completeness and Physical States

**Why must the state space be complete?**

Consider approximating a state $$|\psi\rangle$$ by partial sums:
$$|\psi_N\rangle = \sum_{n=0}^N c_n |n\rangle$$

As $$N \to \infty$$, $$(|\psi_N\rangle)$$ is Cauchy. Completeness guarantees the limit $$|\psi\rangle$$ exists in the space—it's a valid physical state, not just a formal expression.

---

## 4. Worked Examples

### Example 1: Verifying Norm Axioms

**Problem**: Show that $$\|x\|_1 = \sum_{n=1}^\infty |x_n|$$ defines a norm on $$\ell^1$$.

**Solution**:

**(N1) Positivity**:
- $$\|x\|_1 = \sum |x_n| \geq 0$$ since each $$|x_n| \geq 0$$
- If $$\|x\|_1 = 0$$, then $$\sum |x_n| = 0$$, which (for non-negative terms) implies $$|x_n| = 0$$ for all $$n$$, so $$x = 0$$
- Conversely, if $$x = 0$$, then $$\|x\|_1 = \sum 0 = 0$$ $$\checkmark$$

**(N2) Homogeneity**:
$$\|\alpha x\|_1 = \sum_{n=1}^\infty |\alpha x_n| = \sum_{n=1}^\infty |\alpha||x_n| = |\alpha| \sum_{n=1}^\infty |x_n| = |\alpha| \|x\|_1 \checkmark$$

**(N3) Triangle Inequality**:
$$\|x + y\|_1 = \sum_{n=1}^\infty |x_n + y_n| \leq \sum_{n=1}^\infty (|x_n| + |y_n|) = \sum_{n=1}^\infty |x_n| + \sum_{n=1}^\infty |y_n| = \|x\|_1 + \|y\|_1 \checkmark$$

Therefore, $$\|\cdot\|_1$$ is a norm on $$\ell^1$$. $$\square$$

---

### Example 2: Proving $$\ell^2$$ is Complete

**Problem**: Prove that $$\ell^2$$ is a Banach space.

**Solution**:

Let $$(x^{(k)})_{k=1}^\infty$$ be a Cauchy sequence in $$\ell^2$$, where each $$x^{(k)} = (x_n^{(k)})_{n=1}^\infty$$.

**Step 1**: For each fixed $$n$$, the sequence $$(x_n^{(k)})_{k=1}^\infty$$ is Cauchy in $$\mathbb{C}$$.

Since $$(x^{(k)})$$ is Cauchy in $$\ell^2$$:
$$|x_n^{(k)} - x_n^{(m)}|^2 \leq \sum_{j=1}^\infty |x_j^{(k)} - x_j^{(m)}|^2 = \|x^{(k)} - x^{(m)}\|_2^2 \to 0$$

So $$(x_n^{(k)})_k$$ is Cauchy in $$\mathbb{C}$$, hence converges. Define:
$$x_n = \lim_{k \to \infty} x_n^{(k)}$$

**Step 2**: Show $$x = (x_n) \in \ell^2$$.

For any $$N$$ and any $$\varepsilon > 0$$, choose $$K$$ such that $$\|x^{(k)} - x^{(m)}\|_2 < \varepsilon$$ for $$k, m \geq K$$. Then:
$$\sum_{n=1}^N |x_n^{(k)} - x_n^{(m)}|^2 \leq \varepsilon^2$$

Taking $$m \to \infty$$:
$$\sum_{n=1}^N |x_n^{(k)} - x_n|^2 \leq \varepsilon^2$$

This holds for all $$N$$, so:
$$\|x^{(k)} - x\|_2 \leq \varepsilon$$

Thus $$x^{(k)} - x \in \ell^2$$, and since $$\ell^2$$ is a vector space, $$x = x^{(k)} - (x^{(k)} - x) \in \ell^2$$.

**Step 3**: Show $$x^{(k)} \to x$$ in $$\ell^2$$.

From Step 2, for $$k \geq K$$: $$\|x^{(k)} - x\|_2 \leq \varepsilon$$, so $$x^{(k)} \to x$$. $$\square$$

---

### Example 3: Non-Example — $$C[0,1]$$ with $$L^1$$ Norm

**Problem**: Show that $$C[0,1]$$ (continuous functions on $$[0,1]$$) with the $$L^1$$ norm is NOT complete.

**Solution**:

Define:
$$f_n(x) = \begin{cases} 0 & x \in [0, \frac{1}{2} - \frac{1}{n}] \\ n(x - \frac{1}{2} + \frac{1}{n}) & x \in [\frac{1}{2} - \frac{1}{n}, \frac{1}{2}] \\ 1 & x \in [\frac{1}{2}, 1] \end{cases}$$

Each $$f_n$$ is continuous. The sequence $$(f_n)$$ is Cauchy in the $$L^1$$ norm:
$$\|f_n - f_m\|_1 = \int_0^1 |f_n(x) - f_m(x)| dx \to 0$$

However, the pointwise limit is:
$$f(x) = \begin{cases} 0 & x < \frac{1}{2} \\ 1 & x \geq \frac{1}{2} \end{cases}$$

This $$f$$ is **not continuous**, so $$(f_n)$$ has no limit in $$C[0,1]$$.

Therefore, $$(C[0,1], \|\cdot\|_1)$$ is not complete, hence not a Banach space. $$\square$$

---

## 5. Practice Problems

### Level 1: Direct Application

1. **Verify** that $$\|(x_n)\|_\infty = \sup_n |x_n|$$ satisfies the norm axioms.

2. For $$x = (1, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \ldots) = (2^{-n+1})_{n=1}^\infty$$, compute:
   - $$\|x\|_1$$
   - $$\|x\|_2$$
   - $$\|x\|_\infty$$

3. Show that $$\ell^1 \subsetneq \ell^2 \subsetneq \ell^\infty$$ (proper inclusions).

### Level 2: Intermediate

4. Prove that if $$1 \leq p \leq q \leq \infty$$, then $$\ell^p \subseteq \ell^q$$ and $$\|x\|_q \leq \|x\|_p$$.

5. Let $$c_0 = \{(x_n) \in \ell^\infty : \lim_{n\to\infty} x_n = 0\}$$. Prove that $$c_0$$ is a closed subspace of $$\ell^\infty$$, hence a Banach space.

6. **Quantum Connection**: A quantum harmonic oscillator state has energy eigenstate coefficients $$c_n = \frac{1}{\sqrt{n!}}$$.
   - Is $$(c_n) \in \ell^1$$?
   - Is $$(c_n) \in \ell^2$$?
   - What does this mean physically?

### Level 3: Challenging

7. Prove that no norm on an infinite-dimensional vector space can make it both complete and locally compact. (Hint: Use Riesz's lemma.)

8. Show that $$\ell^p$$ and $$\ell^q$$ ($$p \neq q$$) are not isometrically isomorphic (as normed spaces).

9. Let $$X$$ be a normed space and $$Y$$ a Banach space. Prove that the space of bounded linear operators $$B(X,Y)$$ with the operator norm is a Banach space.

---

## 6. Computational Lab: Visualizing Norms and Convergence

```python
"""
Day 232 Computational Lab: Normed Spaces and Banach Spaces
Visualizing different norms and convergence in ell^p spaces
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Part 1: Visualizing Unit Balls in Different Norms (2D)
# ============================================================

def plot_unit_balls_2d():
    """
    Plot unit balls {x: ||x||_p <= 1} for various p-norms in R^2.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    theta = np.linspace(0, 2*np.pi, 1000)

    # Different p values to visualize
    p_values = [1, 1.5, 2, 3, 5, np.inf]
    colors = plt.cm.viridis(np.linspace(0, 1, len(p_values)))

    for p, color in zip(p_values, colors):
        if p == np.inf:
            # Unit ball for sup norm is a square
            x = np.array([1, 1, -1, -1, 1])
            y = np.array([1, -1, -1, 1, 1])
            label = r'$p = \infty$'
        else:
            # Parametric form: x = r*cos(t), y = r*sin(t)
            # ||x||_p = 1 means |cos(t)|^p + |sin(t)|^p = r^p = 1
            # So r = 1 / (|cos(t)|^p + |sin(t)|^p)^(1/p)
            r = 1 / (np.abs(np.cos(theta))**p + np.abs(np.sin(theta))**p)**(1/p)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            label = f'$p = {p}$'

        ax.plot(x, y, color=color, linewidth=2, label=label)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(r'Unit Balls in $\mathbb{R}^2$ for Different $p$-Norms', fontsize=14)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()
    plt.savefig('unit_balls_2d.png', dpi=150)
    plt.show()

# ============================================================
# Part 2: Convergence in ell^p Spaces
# ============================================================

def demonstrate_ellp_convergence():
    """
    Demonstrate convergence of a Cauchy sequence in ell^2.
    """
    # Define a sequence x^(k) that converges to x = (1/n)
    def x_limit(N):
        return np.array([1/n for n in range(1, N+1)])

    def x_k(k, N):
        """Sequence x^(k) = (1/1, 1/2, ..., 1/k, 0, 0, ...)"""
        return np.array([1/n if n <= k else 0 for n in range(1, N+1)])

    N_terms = 100  # Number of terms to consider
    k_values = range(1, 51)

    # Compute ||x^(k) - x||_p for p = 1, 2, infinity
    errors_1 = []
    errors_2 = []
    errors_inf = []

    x_lim = x_limit(N_terms)

    for k in k_values:
        diff = x_k(k, N_terms) - x_lim
        errors_1.append(np.sum(np.abs(diff)))  # ell^1 norm
        errors_2.append(np.sqrt(np.sum(diff**2)))  # ell^2 norm
        errors_inf.append(np.max(np.abs(diff)))  # ell^infty norm

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(k_values, errors_1, 'b-o', label=r'$\|x^{(k)} - x\|_1$', markersize=3)
    ax.semilogy(k_values, errors_2, 'r-s', label=r'$\|x^{(k)} - x\|_2$', markersize=3)
    ax.semilogy(k_values, errors_inf, 'g-^', label=r'$\|x^{(k)} - x\|_\infty$', markersize=3)

    ax.set_xlabel('$k$', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title(r'Convergence in $\ell^p$ Spaces', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ellp_convergence.png', dpi=150)
    plt.show()

# ============================================================
# Part 3: Quantum Harmonic Oscillator Coefficients
# ============================================================

def quantum_harmonic_oscillator():
    """
    Analyze the ell^p membership of quantum harmonic oscillator
    coherent state coefficients.
    """
    # Coherent state |alpha> has coefficients c_n = e^{-|alpha|^2/2} * alpha^n / sqrt(n!)
    alpha = 2.0  # Coherent state parameter

    n_max = 30
    n_values = np.arange(n_max)

    # Compute coefficients (using log to avoid overflow)
    log_cn = -np.abs(alpha)**2/2 + n_values * np.log(np.abs(alpha)) - 0.5 * np.array([
        sum(np.log(np.arange(1, n+1))) if n > 0 else 0 for n in n_values
    ])
    cn = np.exp(log_cn)

    # Compute partial sums for different norms
    cumsum_1 = np.cumsum(np.abs(cn))
    cumsum_2 = np.sqrt(np.cumsum(np.abs(cn)**2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot coefficients
    axes[0].bar(n_values, cn, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('$n$')
    axes[0].set_ylabel(r'$|c_n|$')
    axes[0].set_title(f'Coherent State Coefficients ($\\alpha = {alpha}$)')

    # Plot partial sums for ell^1
    axes[1].plot(n_values, cumsum_1, 'b-o', markersize=4)
    axes[1].axhline(y=np.sum(np.abs(cn)), color='r', linestyle='--',
                     label=f'Limit = {np.sum(np.abs(cn)):.4f}')
    axes[1].set_xlabel('$N$')
    axes[1].set_ylabel(r'$\sum_{n=0}^{N} |c_n|$')
    axes[1].set_title(r'$\ell^1$ Partial Sums')
    axes[1].legend()

    # Plot partial sums for ell^2
    axes[2].plot(n_values, cumsum_2, 'r-s', markersize=4)
    axes[2].axhline(y=1.0, color='g', linestyle='--', label='Normalization = 1')
    axes[2].set_xlabel('$N$')
    axes[2].set_ylabel(r'$\sqrt{\sum_{n=0}^{N} |c_n|^2}$')
    axes[2].set_title(r'$\ell^2$ Partial Sums')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('coherent_state_norms.png', dpi=150)
    plt.show()

    print(f"Coherent state |α={alpha}>:")
    print(f"  ||c||_1 = {np.sum(np.abs(cn)):.6f}")
    print(f"  ||c||_2 = {np.sqrt(np.sum(np.abs(cn)**2)):.6f}")
    print(f"  Verification: ||c||_2 should equal 1 (normalization)")

# ============================================================
# Part 4: Demonstrating Non-Completeness of C[0,1] with L^1 norm
# ============================================================

def demonstrate_incompleteness():
    """
    Visualize the Cauchy sequence in C[0,1] that has no continuous limit.
    """
    def f_n(x, n):
        """Step function approximation"""
        result = np.zeros_like(x)
        mask1 = x <= 0.5 - 1/n
        mask2 = (x > 0.5 - 1/n) & (x <= 0.5)
        mask3 = x > 0.5

        result[mask1] = 0
        result[mask2] = n * (x[mask2] - 0.5 + 1/n)
        result[mask3] = 1
        return result

    x = np.linspace(0, 1, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot the sequence
    n_values = [2, 5, 10, 20, 50]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(n_values)))

    for n, color in zip(n_values, colors):
        axes[0].plot(x, f_n(x, n), color=color, label=f'$f_{{{n}}}$', linewidth=1.5)

    # Plot the discontinuous limit
    step = np.where(x < 0.5, 0, 1)
    axes[0].plot(x, step, 'k--', linewidth=2, label='Limit (discontinuous)')

    axes[0].axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$f_n(x)$')
    axes[0].set_title('Cauchy Sequence in $(C[0,1], \|\cdot\|_1)$ with No Continuous Limit')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Plot L^1 distances ||f_n - f_m||_1
    n_range = range(2, 51)
    distances = []
    for n in n_range:
        # ||f_n - f_50||_1 as approximation
        diff = np.abs(f_n(x, n) - f_n(x, 50))
        distances.append(np.trapz(diff, x))

    axes[1].semilogy(list(n_range), distances, 'b-o', markersize=3)
    axes[1].set_xlabel('$n$')
    axes[1].set_ylabel(r'$\|f_n - f_{50}\|_1$')
    axes[1].set_title('The Sequence is Cauchy (distances decrease)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('incompleteness_demo.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 232: Normed Spaces and Banach Spaces - Computational Lab")
    print("=" * 60)

    print("\n1. Plotting unit balls for different p-norms...")
    plot_unit_balls_2d()

    print("\n2. Demonstrating convergence in ell^p spaces...")
    demonstrate_ellp_convergence()

    print("\n3. Quantum harmonic oscillator coherent state analysis...")
    quantum_harmonic_oscillator()

    print("\n4. Demonstrating non-completeness of C[0,1] with L^1 norm...")
    demonstrate_incompleteness()

    print("\n" + "=" * 60)
    print("Lab complete! Check the generated plots.")
    print("=" * 60)
```

---

## 7. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Norm** | Function $$\|\cdot\|: V \to [0,\infty)$$ satisfying positivity, homogeneity, triangle inequality |
| **Normed Space** | Vector space equipped with a norm |
| **Banach Space** | Complete normed space |
| **$$\ell^p$$ Space** | Sequences $$(x_n)$$ with $$\sum |x_n|^p < \infty$$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{p-Norm:} && \|x\|_p = \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p} \\
&\text{Sup Norm:} && \|x\|_\infty = \sup_n |x_n| \\
&\text{Hölder:} && \sum |x_n y_n| \leq \|x\|_p \|y\|_q, \quad \tfrac{1}{p} + \tfrac{1}{q} = 1 \\
&\text{Minkowski:} && \|x + y\|_p \leq \|x\|_p + \|y\|_p
\end{aligned}}$$

### Key Insights

1. **Norms generalize length** to infinite-dimensional spaces
2. **Completeness is essential** — Cauchy sequences must converge within the space
3. **$$\ell^p$$ spaces are Banach spaces** for all $$1 \leq p \leq \infty$$
4. **Not all normed spaces are complete** — $$C[0,1]$$ with $$L^1$$ norm is a counterexample
5. **Quantum states live in $$\ell^2$$** — square-summable coefficients ensure normalization

---

## 8. Daily Checklist

- [ ] I can state and verify the three norm axioms
- [ ] I understand why every normed space is a metric space
- [ ] I can define a Banach space and explain why completeness matters
- [ ] I can work with $$\ell^1$$, $$\ell^2$$, and $$\ell^\infty$$ spaces
- [ ] I understand Hölder's and Minkowski's inequalities
- [ ] I can explain why quantum mechanics requires complete spaces
- [ ] I completed the computational lab exercises

---

## 9. Preview: Day 233

Tomorrow we introduce **inner product spaces**, which add geometric structure (angles, orthogonality) to normed spaces. We'll prove the **Cauchy-Schwarz inequality** and discover the **polarization identity** that recovers the inner product from the norm. This additional structure is precisely what makes Hilbert spaces so powerful for quantum mechanics.

---

*"The notion of completeness for normed spaces plays a fundamental role in analysis. In a Banach space, one can take limits freely, knowing they will remain in the space."* — Stefan Banach
