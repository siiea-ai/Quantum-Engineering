# Day 234: Hilbert Spaces and L²

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Hilbert space definition, L² construction |
| Afternoon | 3 hours | Problems: Completeness proofs, L² examples |
| Evening | 2 hours | Computational lab: Wave functions in L² |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** a Hilbert space as a complete inner product space
2. **Explain** the construction of $$L^2$$ via Lebesgue integration
3. **State** the Riesz-Fischer theorem and understand its significance
4. **Work with** concrete examples in $$L^2[a,b]$$ and $$\ell^2$$
5. **Prove** that closed subspaces of Hilbert spaces are Hilbert spaces
6. **Connect** $$L^2$$ to quantum wave functions and the Born rule

---

## 1. Core Content: Hilbert Spaces

### 1.1 Definition of Hilbert Space

**Definition**: A **Hilbert space** is a complete inner product space.

$$\boxed{\text{Hilbert Space} = \text{Complete Inner Product Space} = \text{Banach Space with Inner Product}}$$

More precisely, $$\mathcal{H}$$ is a Hilbert space if:
1. $$\mathcal{H}$$ is a vector space over $$\mathbb{C}$$ (or $$\mathbb{R}$$)
2. $$\mathcal{H}$$ has an inner product $$\langle\cdot,\cdot\rangle$$
3. $$\mathcal{H}$$ is complete with respect to the induced norm $$\|x\| = \sqrt{\langle x, x\rangle}$$

### 1.2 Hierarchy of Spaces

$$\text{Vector Space} \supset \text{Normed Space} \supset \text{Inner Product Space}$$

$$\text{Banach Space} = \text{Complete Normed Space}$$

$$\text{Hilbert Space} = \text{Complete Inner Product Space} = \text{Banach Space} \cap \text{Inner Product Space}$$

**Key Point**: Not every Banach space is a Hilbert space. The parallelogram law distinguishes them:
- $$\ell^2$$ is a Hilbert space (satisfies parallelogram law)
- $$\ell^1$$ is a Banach space but NOT a Hilbert space (violates parallelogram law)

### 1.3 Basic Examples

**Example 1: $$\mathbb{C}^n$$** (Finite-dimensional)

The space $$\mathbb{C}^n$$ with inner product $$\langle x, y\rangle = \sum_{i=1}^n x_i \overline{y_i}$$ is a Hilbert space.

*Completeness*: Every finite-dimensional normed space is complete.

**Example 2: $$\ell^2$$** (Sequence space)

$$\ell^2 = \left\{(x_n)_{n=1}^\infty : \sum_{n=1}^\infty |x_n|^2 < \infty\right\}$$

with inner product $$\langle x, y\rangle = \sum_{n=1}^\infty x_n \overline{y_n}$$.

*Completeness*: Proved on Day 232.

**Example 3: $$L^2[a,b]$$** (Function space) — Today's main focus!

---

## 2. The Space L²

### 2.1 Motivation: Why Not Just C[a,b]?

Consider the space $$C[a,b]$$ of continuous functions with the $$L^2$$ norm:
$$\|f\|_2 = \sqrt{\int_a^b |f(x)|^2 \, dx}$$

**Problem**: $$C[a,b]$$ is NOT complete with this norm!

We saw on Day 232 that a sequence of continuous functions can converge (in the $$L^2$$ norm) to a discontinuous function. The limit doesn't stay in the space.

**Solution**: Enlarge the space to include all "square-integrable" functions, using the Lebesgue integral.

### 2.2 The Lebesgue Integral (Overview)

The **Lebesgue integral** extends the Riemann integral to handle:
- More general functions (measurable functions)
- Better convergence theorems
- Functions that may be discontinuous on sets of measure zero

**Key Idea**: Instead of partitioning the domain, partition the range and ask "where does $$f$$ take values in each interval?"

For our purposes, the key facts are:
1. If $$f$$ is Riemann integrable, its Lebesgue integral equals its Riemann integral
2. The Lebesgue integral handles limits much better (dominated convergence theorem)
3. Two functions that differ only on a set of measure zero have the same integral

### 2.3 Definition of L²

**Definition**: The space $$L^2[a,b]$$ consists of equivalence classes of Lebesgue measurable functions $$f: [a,b] \to \mathbb{C}$$ such that:
$$\int_a^b |f(x)|^2 \, dx < \infty$$

Two functions $$f$$ and $$g$$ are **equivalent** ($$f \sim g$$) if $$f = g$$ almost everywhere (a.e.), meaning they differ only on a set of measure zero.

The inner product is:
$$\boxed{\langle f, g\rangle = \int_a^b f(x) \overline{g(x)} \, dx}$$

The induced norm is:
$$\|f\|_2 = \sqrt{\int_a^b |f(x)|^2 \, dx}$$

### 2.4 Why Equivalence Classes?

Without equivalence classes, we would have $$\|f\| = 0$$ but $$f \neq 0$$. For example, the function that equals 1 at a single point and 0 elsewhere has $$\|f\|_2 = 0$$.

By identifying functions that are equal a.e., we ensure:
$$\|f\|_2 = 0 \Leftrightarrow f = 0 \text{ (in } L^2)$$

---

## 3. The Riesz-Fischer Theorem

### 3.1 Statement

**Theorem (Riesz-Fischer)**: The space $$L^2[a,b]$$ is complete. Hence, $$L^2[a,b]$$ is a Hilbert space.

$$\boxed{L^2[a,b] \text{ is a Hilbert space}}$$

### 3.2 Proof Sketch

Let $$(f_n)$$ be a Cauchy sequence in $$L^2[a,b]$$.

**Step 1**: Extract a rapidly convergent subsequence.

Since $$(f_n)$$ is Cauchy, for each $$k$$ we can find $$n_k$$ such that:
$$\|f_{n_{k+1}} - f_{n_k}\|_2 < 2^{-k}$$

**Step 2**: Define the limit function.

Set $$g_N = f_{n_1} + \sum_{k=1}^{N-1} (f_{n_{k+1}} - f_{n_k})$$.

By the triangle inequality:
$$\sum_{k=1}^\infty \|f_{n_{k+1}} - f_{n_k}\|_2 < \sum_{k=1}^\infty 2^{-k} = 1 < \infty$$

The series $$\sum_k |f_{n_{k+1}} - f_{n_k}|$$ converges a.e., so:
$$f(x) = \lim_{N \to \infty} g_N(x) = f_{n_1}(x) + \sum_{k=1}^\infty (f_{n_{k+1}}(x) - f_{n_k}(x))$$
exists for almost every $$x$$.

**Step 3**: Show $$f \in L^2$$ and $$f_n \to f$$ in $$L^2$$.

By Fatou's lemma and dominated convergence, one shows that $$f \in L^2$$ and the original sequence converges to $$f$$ in the $$L^2$$ norm.

The full proof requires careful use of measure theory, but the key insight is that $$L^2$$ "automatically completes" the space of continuous functions.

### 3.3 Significance

The Riesz-Fischer theorem tells us:
1. $$L^2$$ is the "correct" space for square-integrable functions
2. Limits of Cauchy sequences stay in $$L^2$$
3. We can do analysis (take limits, sum series) in $$L^2$$ without leaving the space

---

## 4. Important Properties of L²

### 4.1 Dense Subsets

**Theorem**: The following are dense in $$L^2[a,b]$$:
1. Continuous functions $$C[a,b]$$
2. Step functions
3. Polynomials (by Weierstrass approximation)
4. Trigonometric polynomials (on $$[0, 2\pi]$$)

This means: any $$L^2$$ function can be approximated arbitrarily well by continuous functions, polynomials, etc.

### 4.2 Closed Subspaces

**Theorem**: A closed subspace of a Hilbert space is itself a Hilbert space.

**Proof**: Let $$M \subseteq \mathcal{H}$$ be a closed subspace. Then $$M$$ inherits the inner product from $$\mathcal{H}$$. If $$(x_n)$$ is Cauchy in $$M$$, it's Cauchy in $$\mathcal{H}$$, so $$x_n \to x \in \mathcal{H}$$. Since $$M$$ is closed, $$x \in M$$. $$\square$$

### 4.3 L² on Other Domains

The construction extends to:
- $$L^2(\mathbb{R})$$: square-integrable functions on all of $$\mathbb{R}$$
- $$L^2(\mathbb{R}^n)$$: functions of multiple variables
- $$L^2(X, \mu)$$: functions on any measure space $$(X, \mu)$$

All are Hilbert spaces with the appropriate inner product.

---

## 5. Quantum Mechanics Connection

### 5.1 Wave Functions Live in L²

In quantum mechanics, a particle's state is described by a **wave function** $$\psi(x)$$. The physical interpretation requires:

$$\int_{-\infty}^\infty |\psi(x)|^2 \, dx = 1$$

This says $$\psi \in L^2(\mathbb{R})$$ with $$\|\psi\|_2 = 1$$.

The quantity $$|\psi(x)|^2$$ is the **probability density** for finding the particle at position $$x$$.

### 5.2 The Born Rule

**Born Rule**: The probability of finding a particle in the interval $$[a,b]$$ is:
$$P(a \leq x \leq b) = \int_a^b |\psi(x)|^2 \, dx$$

This integral makes sense precisely because $$\psi \in L^2$$.

### 5.3 Normalization and the Hilbert Space Structure

The normalization condition $$\|\psi\|_2 = 1$$ defines the **unit sphere** in $$L^2$$:
$$S = \{\psi \in L^2(\mathbb{R}) : \|\psi\|_2 = 1\}$$

Physical states are rays in Hilbert space (equivalence classes $$\psi \sim e^{i\theta}\psi$$).

### 5.4 Inner Products as Transition Amplitudes

For states $$\psi, \phi \in L^2$$:
$$\langle\phi|\psi\rangle = \int_{-\infty}^\infty \phi^*(x)\psi(x) \, dx$$

This is the **probability amplitude** for transitioning from $$\psi$$ to $$\phi$$, with probability:
$$P(\psi \to \phi) = |\langle\phi|\psi\rangle|^2$$

### 5.5 Orthogonality in Quantum Mechanics

Two states are **orthogonal** if $$\langle\phi|\psi\rangle = 0$$. This means:
- The states are perfectly distinguishable
- Measuring in a basis containing $$\phi$$ gives zero probability of finding $$\psi$$

**Example**: The harmonic oscillator energy eigenstates $$\{\psi_n\}$$ satisfy:
$$\langle\psi_m|\psi_n\rangle = \int_{-\infty}^\infty \psi_m^*(x)\psi_n(x) \, dx = \delta_{mn}$$

They form an orthonormal set in $$L^2(\mathbb{R})$$.

---

## 6. Worked Examples

### Example 1: Verifying a Function is in L²

**Problem**: Show that $$f(x) = \frac{1}{\sqrt{x}}$$ is in $$L^2[1, \infty)$$ but not in $$L^2[0, 1]$$.

**Solution**:

**On $$[1, \infty)$$**:
$$\int_1^\infty |f(x)|^2 \, dx = \int_1^\infty \frac{1}{x} \, dx = [\ln x]_1^\infty = \infty$$

Wait, this diverges! Let me reconsider. Actually $$f(x) = x^{-1/2}$$, so $$|f(x)|^2 = x^{-1}$$.

The integral $$\int_1^\infty x^{-1} dx$$ diverges, so $$f \notin L^2[1, \infty)$$.

Let's try $$f(x) = \frac{1}{x}$$ instead:
$$\int_1^\infty \frac{1}{x^2} \, dx = \left[-\frac{1}{x}\right]_1^\infty = 0 - (-1) = 1 < \infty$$

So $$g(x) = 1/x$$ is in $$L^2[1, \infty)$$.

**On $$[0, 1]$$ for $$f(x) = x^{-1/2}$$**:
$$\int_0^1 |f(x)|^2 \, dx = \int_0^1 \frac{1}{x} \, dx = [\ln x]_0^1 = 0 - (-\infty) = \infty$$

So $$f(x) = x^{-1/2} \notin L^2[0, 1]$$.

**Corrected Problem**: Show $$f(x) = x^{-1/4}$$ is in $$L^2[0,1]$$ but $$g(x) = x^{-3/4}$$ is not.

$$\int_0^1 x^{-1/2} \, dx = [2x^{1/2}]_0^1 = 2 < \infty$$ $$\checkmark$$

$$\int_0^1 x^{-3/2} \, dx = \left[-2x^{-1/2}\right]_0^1 = -2 - (-\infty) = \infty$$ $$\times$$

So $$x^{-1/4} \in L^2[0,1]$$ but $$x^{-3/4} \notin L^2[0,1]$$. $$\square$$

---

### Example 2: Computing L² Inner Products

**Problem**: Compute $$\langle f, g\rangle$$ in $$L^2[0, 2\pi]$$ where $$f(x) = \sin(x)$$ and $$g(x) = \sin(2x)$$.

**Solution**:

$$\langle f, g\rangle = \int_0^{2\pi} \sin(x) \sin(2x) \, dx$$

Use the product-to-sum formula: $$\sin A \sin B = \frac{1}{2}[\cos(A-B) - \cos(A+B)]$$

$$= \frac{1}{2}\int_0^{2\pi} [\cos(-x) - \cos(3x)] \, dx = \frac{1}{2}\int_0^{2\pi} [\cos(x) - \cos(3x)] \, dx$$

$$= \frac{1}{2}\left[\sin(x) - \frac{\sin(3x)}{3}\right]_0^{2\pi} = \frac{1}{2}[(0 - 0) - (0 - 0)] = 0$$

Therefore $$\sin(x) \perp \sin(2x)$$ in $$L^2[0, 2\pi]$$. $$\square$$

---

### Example 3: A Quantum Gaussian Wave Packet

**Problem**: The Gaussian wave function $$\psi(x) = \left(\frac{1}{\pi\sigma^2}\right)^{1/4} e^{-x^2/(2\sigma^2)}$$ is in $$L^2(\mathbb{R})$$ with $$\|\psi\|_2 = 1$$. Verify the normalization.

**Solution**:

$$\|\psi\|_2^2 = \int_{-\infty}^\infty |\psi(x)|^2 \, dx = \sqrt{\frac{1}{\pi\sigma^2}} \int_{-\infty}^\infty e^{-x^2/\sigma^2} \, dx$$

Let $$u = x/\sigma$$, so $$dx = \sigma \, du$$:

$$= \sqrt{\frac{1}{\pi\sigma^2}} \cdot \sigma \int_{-\infty}^\infty e^{-u^2} \, du = \frac{1}{\sqrt{\pi}} \cdot \sqrt{\pi} = 1$$

(using $$\int_{-\infty}^\infty e^{-u^2} du = \sqrt{\pi}$$)

Therefore $$\|\psi\|_2 = 1$$, confirming the normalization. $$\square$$

---

## 7. Practice Problems

### Level 1: Direct Application

1. Verify that $$f(x) = e^{-|x|}$$ is in $$L^2(\mathbb{R})$$. Compute $$\|f\|_2$$.

2. Compute the inner product $$\langle e^{inx}, e^{imx}\rangle$$ in $$L^2[0, 2\pi]$$ for integers $$m, n$$.

3. Show that the constant function $$f(x) = 1$$ is in $$L^2[0, 1]$$ but not in $$L^2(\mathbb{R})$$.

### Level 2: Intermediate

4. For $$f, g \in L^2[a,b]$$, prove that $$fg \in L^1[a,b]$$ (Hint: $$2|fg| \leq |f|^2 + |g|^2$$).

5. Prove that if $$f_n \to f$$ in $$L^2$$ and $$g_n \to g$$ in $$L^2$$, then $$\langle f_n, g_n\rangle \to \langle f, g\rangle$$.

6. **Quantum Problem**: The hydrogen atom ground state wave function is $$\psi_{100}(r) = \frac{1}{\sqrt{\pi}a_0^{3/2}} e^{-r/a_0}$$ (in spherical coordinates with $$dV = 4\pi r^2 dr$$).
   Verify that $$\int_0^\infty |\psi_{100}|^2 4\pi r^2 \, dr = 1$$.

### Level 3: Challenging

7. Prove that $$L^2[0, 1]$$ is separable (has a countable dense subset). Hint: Consider polynomials with rational coefficients.

8. Show that $$L^2[0, 1]$$ and $$L^2[0, 2]$$ are isometrically isomorphic as Hilbert spaces.

9. Prove that if $$M$$ is a closed subspace of a Hilbert space $$\mathcal{H}$$, then every $$x \in \mathcal{H}$$ can be uniquely written as $$x = y + z$$ where $$y \in M$$ and $$z \perp M$$.

---

## 8. Computational Lab: Wave Functions in L²

```python
"""
Day 234 Computational Lab: Hilbert Spaces and L²
Working with quantum wave functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import hermite

# ============================================================
# Part 1: Gaussian Wave Packets
# ============================================================

def gaussian_wave_packets():
    """
    Explore Gaussian wave packets in L^2(R).
    """
    x = np.linspace(-10, 10, 1000)

    def gaussian(x, sigma, x0=0):
        """Normalized Gaussian wave function."""
        norm = (1 / (np.pi * sigma**2))**0.25
        return norm * np.exp(-(x - x0)**2 / (2 * sigma**2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Different widths
    sigmas = [0.5, 1.0, 2.0]
    colors = ['blue', 'green', 'red']

    ax = axes[0, 0]
    for sigma, color in zip(sigmas, colors):
        psi = gaussian(x, sigma)
        ax.plot(x, psi, color=color, linewidth=2, label=f'$\\sigma = {sigma}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\psi(x)$')
    ax.set_title('Gaussian Wave Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Probability densities
    ax = axes[0, 1]
    for sigma, color in zip(sigmas, colors):
        psi = gaussian(x, sigma)
        ax.plot(x, np.abs(psi)**2, color=color, linewidth=2, label=f'$\\sigma = {sigma}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\\psi(x)|^2$')
    ax.set_title('Probability Densities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Verify normalization
    ax = axes[1, 0]
    sigmas_test = np.linspace(0.1, 5, 50)
    norms = []
    for sigma in sigmas_test:
        psi = gaussian(x, sigma)
        norm_sq = np.trapz(np.abs(psi)**2, x)
        norms.append(np.sqrt(norm_sq))

    ax.plot(sigmas_test, norms, 'b-', linewidth=2)
    ax.axhline(y=1, color='r', linestyle='--', label='Expected = 1')
    ax.set_xlabel('$\\sigma$')
    ax.set_ylabel('$\\|\\psi\\|_2$')
    ax.set_title('Normalization Verification')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inner products of displaced Gaussians
    ax = axes[1, 1]
    displacements = np.linspace(0, 10, 100)
    sigma = 1.0
    overlaps = []

    psi_0 = gaussian(x, sigma, x0=0)
    for d in displacements:
        psi_d = gaussian(x, sigma, x0=d)
        overlap = np.trapz(psi_0 * np.conj(psi_d), x)
        overlaps.append(np.abs(overlap)**2)

    ax.plot(displacements, overlaps, 'b-', linewidth=2)
    ax.set_xlabel('Displacement $d$')
    ax.set_ylabel('$|\\langle\\psi_0|\\psi_d\\rangle|^2$')
    ax.set_title('Overlap of Displaced Gaussians')
    ax.grid(True, alpha=0.3)

    # Theory: |<psi_0|psi_d>|^2 = exp(-d^2/(4*sigma^2))
    theory = np.exp(-displacements**2 / (4 * sigma**2))
    ax.plot(displacements, theory, 'r--', linewidth=2, label='Theory')
    ax.legend()

    plt.tight_layout()
    plt.savefig('gaussian_wave_packets.png', dpi=150)
    plt.show()

# ============================================================
# Part 2: Harmonic Oscillator Eigenstates
# ============================================================

def harmonic_oscillator_eigenstates():
    """
    Study the quantum harmonic oscillator eigenstates in L^2(R).
    """
    x = np.linspace(-6, 6, 1000)

    def psi_n(x, n):
        """
        Normalized harmonic oscillator eigenstate.
        psi_n(x) = (1/(sqrt(2^n n! sqrt(pi)))) * H_n(x) * exp(-x^2/2)
        """
        Hn = hermite(n)  # Physicist's Hermite polynomial
        norm = 1 / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
        return norm * Hn(x) * np.exp(-x**2 / 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot first several eigenstates
    ax = axes[0, 0]
    for n in range(5):
        psi = psi_n(x, n)
        ax.plot(x, psi + n, linewidth=2, label=f'$n = {n}$')
        ax.axhline(y=n, color='gray', linewidth=0.5, linestyle='--')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\psi_n(x) + n$ (offset for clarity)')
    ax.set_title('Harmonic Oscillator Eigenstates')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Probability densities
    ax = axes[0, 1]
    for n in range(5):
        psi = psi_n(x, n)
        ax.plot(x, np.abs(psi)**2 + n*0.3, linewidth=2, label=f'$n = {n}$')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\\psi_n(x)|^2$ (offset)')
    ax.set_title('Probability Densities')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Orthonormality matrix
    ax = axes[1, 0]
    n_max = 6
    overlap_matrix = np.zeros((n_max, n_max))

    for i in range(n_max):
        for j in range(n_max):
            psi_i = psi_n(x, i)
            psi_j = psi_n(x, j)
            overlap_matrix[i, j] = np.trapz(psi_i * psi_j, x)

    im = ax.imshow(np.abs(overlap_matrix), cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('$n$')
    ax.set_ylabel('$m$')
    ax.set_title('$|\\langle\\psi_m|\\psi_n\\rangle|$ (Orthonormality)')
    ax.set_xticks(range(n_max))
    ax.set_yticks(range(n_max))
    plt.colorbar(im, ax=ax)

    # Annotate
    for i in range(n_max):
        for j in range(n_max):
            ax.text(j, i, f'{overlap_matrix[i,j]:.2f}',
                   ha='center', va='center', fontsize=8)

    # Superposition state
    ax = axes[1, 1]

    # Create a superposition: |ψ> = (|0> + |1> + |2>) / sqrt(3)
    psi_super = (psi_n(x, 0) + psi_n(x, 1) + psi_n(x, 2)) / np.sqrt(3)

    ax.plot(x, psi_super, 'b-', linewidth=2, label=r'$\psi(x) = \frac{1}{\sqrt{3}}(\psi_0 + \psi_1 + \psi_2)$')
    ax.fill_between(x, 0, np.abs(psi_super)**2, alpha=0.3, label=r'$|\psi(x)|^2$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\psi(x)$')
    ax.set_title('Superposition State')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Verify normalization
    norm_sq = np.trapz(np.abs(psi_super)**2, x)
    print(f"Superposition state norm: ||ψ||₂ = {np.sqrt(norm_sq):.6f}")

    plt.tight_layout()
    plt.savefig('harmonic_oscillator.png', dpi=150)
    plt.show()

# ============================================================
# Part 3: L² Convergence vs Pointwise Convergence
# ============================================================

def convergence_comparison():
    """
    Illustrate that L² convergence does NOT imply pointwise convergence.
    """
    x = np.linspace(0, 1, 1000)

    def f_n(x, n):
        """
        Function that is n on [0, 1/n] and 0 elsewhere.
        ||f_n||_2 = sqrt(n^2 * 1/n) = sqrt(n) -> infinity
        """
        return np.where(x <= 1/n, n, 0)

    def g_n(x, n):
        """
        Function that is 1 on [k/n, (k+1)/n] for some k depending on n.
        This converges to 0 in L^2 but not pointwise anywhere!
        """
        # "Typewriter sequence" - cycles through subintervals
        k = n % int(np.sqrt(n) + 1) if n > 0 else 0
        m = int(np.sqrt(n)) + 1
        left = k / m
        right = (k + 1) / m
        return np.where((x >= left) & (x < right), 1, 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sequence that converges pointwise to 0 but not in L^2
    ax = axes[0, 0]
    for n in [1, 2, 5, 10]:
        y = f_n(x, n)
        ax.plot(x, y, linewidth=2, label=f'$n = {n}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f_n(x)$')
    ax.set_title('$f_n$: Converges pointwise to 0, but $\\|f_n\\|_2 \\to \\infty$')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # L^2 norms
    ax = axes[0, 1]
    n_values = range(1, 51)
    norms_f = [np.sqrt(np.trapz(f_n(x, n)**2, x)) for n in n_values]
    ax.plot(n_values, norms_f, 'b-o', markersize=3)
    ax.set_xlabel('$n$')
    ax.set_ylabel('$\\|f_n\\|_2$')
    ax.set_title('$L^2$ Norms of $f_n$ (diverging)')
    ax.grid(True, alpha=0.3)

    # Sequence that converges in L^2 to 0 but not pointwise anywhere
    ax = axes[1, 0]
    for n in [1, 4, 9, 16]:
        y = g_n(x, n)
        ax.plot(x, y + n*0.1, linewidth=2, label=f'$n = {n}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$g_n(x)$ (offset)')
    ax.set_title('$g_n$: Converges in $L^2$ to 0, but not pointwise')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # L^2 norms of g_n
    ax = axes[1, 1]
    n_values = range(1, 101)
    norms_g = []
    for n in n_values:
        m = int(np.sqrt(n)) + 1
        # ||g_n||_2^2 = 1/m -> 0
        norms_g.append(1 / np.sqrt(m))

    ax.plot(n_values, norms_g, 'r-o', markersize=2)
    ax.set_xlabel('$n$')
    ax.set_ylabel('$\\|g_n\\|_2$')
    ax.set_title('$L^2$ Norms of $g_n$ (converging to 0)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: Fourier Series and L² Approximation
# ============================================================

def fourier_L2_approximation():
    """
    Approximate a function in L² using Fourier partial sums.
    """
    x = np.linspace(0, 2*np.pi, 1000)

    # Target function: square wave
    def square_wave(x):
        return np.where(np.sin(x) >= 0, 1, -1)

    # Fourier partial sum
    def fourier_partial_sum(x, N):
        """Fourier series of square wave: (4/pi) * sum_{k odd} sin(kx)/k"""
        result = np.zeros_like(x)
        for k in range(1, N+1, 2):  # odd k only
            result += np.sin(k * x) / k
        return (4 / np.pi) * result

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Show approximations
    ax = axes[0, 0]
    f = square_wave(x)
    ax.plot(x, f, 'k-', linewidth=2, label='Square wave')
    for N in [1, 5, 15, 51]:
        approx = fourier_partial_sum(x, N)
        ax.plot(x, approx, linewidth=1.5, label=f'$N = {N}$', alpha=0.7)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Fourier Approximation of Square Wave')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # L^2 error
    ax = axes[0, 1]
    N_values = range(1, 102, 2)
    L2_errors = []
    for N in N_values:
        approx = fourier_partial_sum(x, N)
        error = np.sqrt(np.trapz((f - approx)**2, x))
        L2_errors.append(error)

    ax.semilogy(N_values, L2_errors, 'b-o', markersize=3)
    ax.set_xlabel('$N$')
    ax.set_ylabel('$\\|f - S_N\\|_2$')
    ax.set_title('$L^2$ Approximation Error')
    ax.grid(True, alpha=0.3)

    # Pointwise error (Gibbs phenomenon)
    ax = axes[1, 0]
    N = 51
    approx = fourier_partial_sum(x, N)
    ax.plot(x, np.abs(f - approx), 'r-', linewidth=1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|f(x) - S_N(x)|$')
    ax.set_title(f'Pointwise Error for $N = {N}$ (Gibbs Phenomenon)')
    ax.grid(True, alpha=0.3)

    # Parseval's identity verification
    ax = axes[1, 1]
    # ||f||_2^2 = integral of |f|^2 = 2*pi (since f = ±1)
    norm_f_sq = np.trapz(f**2, x)

    # Sum of |c_k|^2: coefficients are 2/(k*pi) for odd k
    N_max = 500
    sum_ck_sq = 0
    partial_sums = []
    for k in range(1, N_max+1, 2):
        sum_ck_sq += 2 * (2 / (k * np.pi))**2  # factor of 2 for ±k
        partial_sums.append(sum_ck_sq)

    k_values = list(range(1, N_max+1, 2))
    ax.plot(k_values, partial_sums, 'b-', linewidth=2, label=r'$\sum |c_k|^2$')
    ax.axhline(y=norm_f_sq / (2*np.pi), color='r', linestyle='--',
               label=f'$\\|f\\|_2^2 / (2\\pi) = {norm_f_sq/(2*np.pi):.4f}$')
    ax.set_xlabel('$N$ (odd terms)')
    ax.set_ylabel('Partial sum')
    ax.set_title("Parseval's Identity Verification")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fourier_L2.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 234: Hilbert Spaces and L² - Computational Lab")
    print("=" * 60)

    print("\n1. Gaussian wave packets...")
    gaussian_wave_packets()

    print("\n2. Harmonic oscillator eigenstates...")
    harmonic_oscillator_eigenstates()

    print("\n3. Convergence comparison (L² vs pointwise)...")
    convergence_comparison()

    print("\n4. Fourier series and L² approximation...")
    fourier_L2_approximation()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## 9. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Hilbert Space** | Complete inner product space |
| **$$L^2[a,b]$$** | Square-integrable functions (mod a.e. equality) |
| **Square-integrable** | $$\int |f|^2 \, dx < \infty$$ |

### Key Theorems

$$\boxed{\begin{aligned}
&\text{Riesz-Fischer:} && L^2 \text{ is complete, hence a Hilbert space} \\
&\text{Dense Subsets:} && C[a,b], \text{polynomials, trig functions are dense in } L^2[a,b] \\
&\text{Closed Subspace:} && \text{Closed subspace of Hilbert space is Hilbert space}
\end{aligned}}$$

### Quantum Mechanics Connection

| $$L^2$$ Concept | Quantum Interpretation |
|----------------|----------------------|
| $$\psi \in L^2(\mathbb{R})$$ | Valid wave function |
| $$\|\psi\|_2 = 1$$ | Normalization |
| $$\|\psi(x)\|^2$$ | Probability density |
| $$\langle\phi\|\psi\rangle$$ | Transition amplitude |
| $$\|\langle\phi\|\psi\rangle\|^2$$ | Transition probability |

---

## 10. Daily Checklist

- [ ] I can define a Hilbert space
- [ ] I understand the construction of $$L^2$$ and why equivalence classes are necessary
- [ ] I can state the Riesz-Fischer theorem
- [ ] I can compute inner products in $$L^2[a,b]$$
- [ ] I understand why quantum wave functions must be in $$L^2$$
- [ ] I can connect the Born rule to the $$L^2$$ structure
- [ ] I completed the computational lab exercises

---

## 11. Preview: Day 235

Tomorrow we study **orthonormal sets** and the **Gram-Schmidt process**. We'll learn how to construct orthonormal sets from arbitrary linearly independent sets, prove **Bessel's inequality**, and see how orthonormal sets relate to quantum mechanical bases of states.

---

*"The completeness of $$L^2$$ is not merely a technical convenience—it is essential for the mathematical consistency of quantum mechanics."* — John von Neumann
