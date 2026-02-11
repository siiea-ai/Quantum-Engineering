# Day 237: Separable Hilbert Spaces

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Separability, classification theorem |
| Afternoon | 3 hours | Problems: Isomorphisms, examples |
| Evening | 2 hours | Computational lab: Hilbert space isomorphisms |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** separability for Hilbert spaces
2. **Prove** that a Hilbert space is separable iff it has a countable orthonormal basis
3. **State** the classification theorem for separable Hilbert spaces
4. **Construct** explicit isomorphisms between $$L^2$$ and $$\ell^2$$
5. **Understand** the Riesz representation theorem for Hilbert spaces
6. **Connect** separability to the structure of quantum mechanics

---

## 1. Core Content: Separability

### 1.1 Definition of Separability

**Definition**: A metric space $$(X, d)$$ is **separable** if it contains a countable dense subset.

For Hilbert spaces, this means: $$\mathcal{H}$$ is separable if there exists a countable set $$\{x_n\}_{n=1}^\infty \subset \mathcal{H}$$ such that $$\overline{\{x_n\}} = \mathcal{H}$$.

**Equivalently**: Every element of $$\mathcal{H}$$ can be approximated arbitrarily well by elements of the countable set.

### 1.2 Examples

**Separable Hilbert Spaces**:
- $$\mathbb{C}^n$$ (finite-dimensional, automatically separable)
- $$\ell^2$$ (the set of sequences with finitely many non-zero rational entries is dense)
- $$L^2[a,b]$$ (continuous functions, hence polynomials with rational coefficients, are dense)
- $$L^2(\mathbb{R})$$

**Non-separable Hilbert Space**:
- $$\ell^2(I)$$ where $$I$$ is uncountable (e.g., $$I = \mathbb{R}$$)

### 1.3 Characterization Theorem

**Theorem**: A Hilbert space $$\mathcal{H}$$ is separable if and only if it has a countable (or finite) orthonormal basis.

**Proof**:

$$(\Leftarrow)$$ Suppose $$\{e_n\}_{n=1}^\infty$$ is a countable orthonormal basis.

Define $$D = \{\sum_{k=1}^N q_k e_k : N \in \mathbb{N}, q_k \in \mathbb{Q} + i\mathbb{Q}\}$$.

This is countable (countable union of countable sets). We claim $$D$$ is dense.

For any $$x \in \mathcal{H}$$ and $$\varepsilon > 0$$:
- By Parseval, $$x = \sum_n c_n e_n$$ where $$c_n = \langle e_n, x\rangle$$
- Choose $$N$$ large enough that $$\|\sum_{n > N} c_n e_n\| < \varepsilon/2$$
- Approximate each $$c_k$$ by $$q_k \in \mathbb{Q} + i\mathbb{Q}$$ with $$|c_k - q_k| < \varepsilon/(2\sqrt{N})$$

Then $$\|\sum_{k=1}^N q_k e_k - x\| < \varepsilon$$. So $$D$$ is dense.

$$(\Rightarrow)$$ Suppose $$\mathcal{H}$$ is separable with dense subset $$D = \{d_n\}$$.

Claim: Any orthonormal set in $$\mathcal{H}$$ is at most countable.

Let $$\{e_\alpha\}_{\alpha \in A}$$ be orthonormal. For $$\alpha \neq \beta$$:
$$\|e_\alpha - e_\beta\|^2 = \|e_\alpha\|^2 + \|e_\beta\|^2 - 2\text{Re}\langle e_\alpha, e_\beta\rangle = 2$$

So the balls $$B(e_\alpha, 1/2)$$ are disjoint. Each must contain some $$d_n$$ (density). This gives an injection $$A \to \mathbb{N}$$, so $$A$$ is countable.

Therefore, any orthonormal basis (which exists by Zorn) is countable. $$\square$$

---

## 2. The Classification Theorem

### 2.1 Statement

**Theorem (Classification of Separable Hilbert Spaces)**:

1. Every finite-dimensional Hilbert space of dimension $$n$$ is isometrically isomorphic to $$\mathbb{C}^n$$.

2. Every infinite-dimensional separable Hilbert space is isometrically isomorphic to $$\ell^2$$.

$$\boxed{\text{All infinite-dimensional separable Hilbert spaces are isomorphic to } \ell^2}$$

### 2.2 Proof

Let $$\mathcal{H}$$ be an infinite-dimensional separable Hilbert space with orthonormal basis $$\{e_n\}_{n=1}^\infty$$.

Define $$U: \mathcal{H} \to \ell^2$$ by:
$$U(x) = (\langle e_1, x\rangle, \langle e_2, x\rangle, \langle e_3, x\rangle, \ldots)$$

**Well-defined**: By Bessel, $$\sum_n |\langle e_n, x\rangle|^2 \leq \|x\|^2 < \infty$$, so $$U(x) \in \ell^2$$.

**Linear**: Clear from linearity of inner product.

**Isometry**: By Parseval,
$$\|U(x)\|_{\ell^2}^2 = \sum_n |\langle e_n, x\rangle|^2 = \|x\|^2$$

**Surjective**: Given $$(c_n) \in \ell^2$$, define $$x = \sum_n c_n e_n$$ (converges by completeness). Then:
$$\langle e_k, x\rangle = \sum_n c_n \langle e_k, e_n\rangle = c_k$$

So $$U(x) = (c_n)$$.

Therefore $$U$$ is a surjective isometry, hence an isometric isomorphism. $$\square$$

### 2.3 Significance

This theorem says:
- There is essentially **only one** infinite-dimensional separable Hilbert space (up to isomorphism)
- $$L^2[0,1]$$, $$L^2(\mathbb{R})$$, and $$\ell^2$$ are "the same" space!
- The specific realization (functions vs. sequences) is a matter of convenience

---

## 3. The Riesz Representation Theorem

### 3.1 Statement

**Theorem (Riesz Representation for Hilbert Spaces)**: Let $$\mathcal{H}$$ be a Hilbert space and $$f: \mathcal{H} \to \mathbb{C}$$ a continuous (bounded) linear functional. Then there exists a unique $$y \in \mathcal{H}$$ such that:

$$\boxed{f(x) = \langle y, x\rangle \quad \forall x \in \mathcal{H}}$$

Moreover, $$\|f\| = \|y\|$$.

### 3.2 Proof

**Existence**:

If $$f = 0$$, take $$y = 0$$.

Otherwise, let $$N = \ker(f) = \{x : f(x) = 0\}$$. This is a closed proper subspace.

Take any $$z \notin N$$ and let $$z_0 = z - P_N z$$ where $$P_N$$ is the orthogonal projection onto $$N$$. Then $$z_0 \perp N$$ and $$z_0 \neq 0$$.

For any $$x \in \mathcal{H}$$:
$$f(x) z_0 - f(z_0) x \in N$$
(verify: $$f(f(x)z_0 - f(z_0)x) = f(x)f(z_0) - f(z_0)f(x) = 0$$)

Since $$z_0 \perp N$$:
$$0 = \langle z_0, f(x)z_0 - f(z_0)x\rangle = f(x)\|z_0\|^2 - f(z_0)\langle z_0, x\rangle$$

So:
$$f(x) = \frac{f(z_0)}{\|z_0\|^2}\langle z_0, x\rangle = \langle y, x\rangle$$

where $$y = \frac{\overline{f(z_0)}}{\|z_0\|^2} z_0$$.

**Uniqueness**: If $$\langle y, x\rangle = \langle y', x\rangle$$ for all $$x$$, then $$\langle y - y', x\rangle = 0$$ for all $$x$$. Taking $$x = y - y'$$ gives $$\|y - y'\|^2 = 0$$, so $$y = y'$$.

**Norm equality**: $$\|f\| = \sup_{\|x\|=1} |f(x)| = \sup_{\|x\|=1} |\langle y, x\rangle| \leq \|y\|$$ by Cauchy-Schwarz. Equality achieved at $$x = y/\|y\|$$. $$\square$$

### 3.3 Interpretation

The Riesz representation theorem says:
- Every continuous linear functional on $$\mathcal{H}$$ is given by taking an inner product
- The dual space $$\mathcal{H}^*$$ is isometrically isomorphic to $$\mathcal{H}$$ itself
- Hilbert spaces are **self-dual**

---

## 4. Quantum Mechanics Connection

### 4.1 The State Space of Quantum Mechanics

In quantum mechanics, the state space is a **separable Hilbert space**. This is a fundamental postulate.

**Why separable?**
- Physical observables have at most countably many distinct eigenvalues (discrete spectrum)
- Separability allows us to work with countable bases
- All "reasonable" function spaces ($$L^2$$) are separable

### 4.2 Bra-Ket Duality and Riesz

Dirac's bra-ket notation encodes Riesz representation:

| Ket $$\|\psi\rangle$$ | Element of $$\mathcal{H}$$ |
| Bra $$\langle\phi\|$$ | Element of $$\mathcal{H}^* \cong \mathcal{H}$$ |
| $$\langle\phi\|\psi\rangle$$ | Inner product / functional evaluation |

The Riesz theorem justifies identifying $$\langle\phi|$$ with $$|\phi\rangle$$ (up to conjugation).

### 4.3 Position and Momentum Representations

The isomorphism $$L^2(\mathbb{R}) \cong \ell^2$$ manifests in physics as the equivalence of:
- **Position representation**: Wave functions $$\psi(x)$$
- **Energy representation**: Coefficients $$c_n$$ in eigenstate expansion

The Fourier transform gives another isomorphism:
- **Position representation**: $$\psi(x)$$
- **Momentum representation**: $$\tilde{\psi}(p)$$

All are "the same" Hilbert space in different clothes!

### 4.4 Dimension of Quantum State Space

For a quantum system with Hilbert space $$\mathcal{H}$$:

| System | Hilbert Space | Dimension |
|--------|---------------|-----------|
| Qubit | $$\mathbb{C}^2$$ | 2 |
| Spin-$$j$$ particle | $$\mathbb{C}^{2j+1}$$ | $$2j+1$$ |
| Harmonic oscillator | $$L^2(\mathbb{R})$$ | $$\aleph_0$$ (countably infinite) |
| Free particle | $$L^2(\mathbb{R}^3)$$ | $$\aleph_0$$ |

All infinite-dimensional cases are isomorphic to $$\ell^2$$!

---

## 5. Non-Separable Hilbert Spaces

### 5.1 When They Arise

Non-separable Hilbert spaces have uncountable orthonormal bases. They appear in:
- Quantum field theory (some formulations)
- Continuous tensor products
- Abstract mathematical constructions

### 5.2 Example: $$\ell^2(\mathbb{R})$$

$$\ell^2(\mathbb{R}) = \{f: \mathbb{R} \to \mathbb{C} : \sum_{t \in \mathbb{R}} |f(t)|^2 < \infty\}$$

Here, the "sum" means only countably many terms are non-zero. The standard basis $$\{e_t\}_{t \in \mathbb{R}}$$ (where $$e_t(s) = \delta_{ts}$$) is uncountable.

### 5.3 Classification

**Theorem**: Two Hilbert spaces are isometrically isomorphic iff they have the same **dimension** (cardinality of any orthonormal basis).

For separable spaces: dimension is either finite $$n$$ or countably infinite $$\aleph_0$$.

---

## 6. Worked Examples

### Example 1: Explicit Isomorphism $$L^2[0, 2\pi] \to \ell^2$$

**Problem**: Construct an explicit isometric isomorphism from $$L^2[0, 2\pi]$$ to $$\ell^2$$.

**Solution**:

The Fourier basis $$\{e_n(x) = e^{inx}/\sqrt{2\pi}\}_{n \in \mathbb{Z}}$$ is an orthonormal basis for $$L^2[0, 2\pi]$$.

We can relabel: let $$f_1 = e_0, f_2 = e_1, f_3 = e_{-1}, f_4 = e_2, f_5 = e_{-2}, \ldots$$ to get $$\{f_k\}_{k=1}^\infty$$.

Define $$U: L^2[0, 2\pi] \to \ell^2$$ by:
$$U(g) = (\langle f_1, g\rangle, \langle f_2, g\rangle, \langle f_3, g\rangle, \ldots)$$

Explicitly:
$$U(g) = \left(\frac{1}{\sqrt{2\pi}}\int_0^{2\pi} g(x)dx, \frac{1}{\sqrt{2\pi}}\int_0^{2\pi} g(x)e^{-ix}dx, \frac{1}{\sqrt{2\pi}}\int_0^{2\pi} g(x)e^{ix}dx, \ldots\right)$$

The inverse is:
$$U^{-1}((c_k)) = \sum_{k=1}^\infty c_k f_k(x)$$

This is an isometric isomorphism by the classification theorem. $$\square$$

---

### Example 2: Riesz Representation

**Problem**: Find the element $$y \in L^2[0, 1]$$ representing the functional $$f(g) = \int_0^1 x^2 g(x) dx$$.

**Solution**:

The functional is:
$$f(g) = \int_0^1 x^2 g(x) dx = \int_0^1 \overline{x^2} g(x) dx = \langle x^2, g\rangle$$

(since $$x^2$$ is real, $$\overline{x^2} = x^2$$)

By Riesz representation, $$y(x) = x^2$$.

**Verification**: $$f(g) = \langle y, g\rangle = \int_0^1 x^2 \cdot g(x) dx$$ $$\checkmark$$

**Norm check**:
$$\|f\| = \|y\|_{L^2} = \sqrt{\int_0^1 x^4 dx} = \sqrt{\frac{1}{5}} = \frac{1}{\sqrt{5}}$$ $$\square$$

---

### Example 3: Showing a Space is Separable

**Problem**: Prove that $$L^2(\mathbb{R})$$ is separable.

**Solution**:

**Step 1**: Show step functions with rational heights on intervals with rational endpoints are dense.

Let $$S$$ be the set of functions of the form $$\sum_{k=1}^N q_k \chi_{[a_k, b_k)}$$ where $$q_k \in \mathbb{Q} + i\mathbb{Q}$$ and $$a_k, b_k \in \mathbb{Q}$$.

$$S$$ is countable.

**Step 2**: Any $$f \in L^2(\mathbb{R})$$ can be approximated.

By density of compactly supported continuous functions in $$L^2$$, and density of step functions in continuous functions (in $$L^2$$ norm), we can approximate $$f$$ by step functions.

Rational approximation to step functions shows $$S$$ is dense.

**Conclusion**: $$L^2(\mathbb{R})$$ has a countable dense subset, so it's separable. $$\square$$

---

## 7. Practice Problems

### Level 1: Direct Application

1. Verify that the map $$U: \ell^2 \to \ell^2$$ defined by $$U((c_1, c_2, c_3, \ldots)) = (0, c_1, c_2, \ldots)$$ (right shift) is an isometry but not surjective.

2. Find the Riesz representer for the functional $$f(g) = g(1/2)$$ on... wait, this isn't bounded on $$L^2$$! Explain why.

3. Show that $$\ell^1$$ is separable but $$\ell^\infty$$ is not.

### Level 2: Intermediate

4. Prove that if $$\{e_n\}$$ and $$\{f_n\}$$ are orthonormal bases for $$\mathcal{H}$$, then there exists a unitary $$U: \mathcal{H} \to \mathcal{H}$$ with $$Ue_n = f_n$$.

5. Let $$M$$ be a closed subspace of a separable Hilbert space. Prove $$M$$ and $$M^\perp$$ are both separable.

6. **Quantum Problem**: The position and momentum operators on $$L^2(\mathbb{R})$$ have continuous spectra. How does this relate to the classification theorem?

### Level 3: Challenging

7. Prove that the dual of $$\ell^1$$ is $$\ell^\infty$$ but $$\ell^1 \not\cong \ell^\infty$$. Why doesn't this contradict Riesz representation?

8. Show that $$L^2[0, 1]$$ and $$L^2[0, 2]$$ are isometrically isomorphic by constructing an explicit isomorphism.

9. **(Gram Matrix)**: Given vectors $$v_1, \ldots, v_n$$ in a Hilbert space, the Gram matrix is $$G_{ij} = \langle v_i, v_j\rangle$$. Prove the vectors are linearly independent iff $$\det(G) \neq 0$$.

---

## 8. Computational Lab: Hilbert Space Isomorphisms

```python
"""
Day 237 Computational Lab: Separable Hilbert Spaces and Isomorphisms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import hermite

# ============================================================
# Part 1: Explicit Isomorphism L^2[0, 2π] ↔ ℓ^2
# ============================================================

def demonstrate_L2_ell2_isomorphism():
    """
    Demonstrate the isomorphism between L^2[0, 2π] and ℓ^2.
    """
    x = np.linspace(0, 2*np.pi, 1000)

    # A function in L^2[0, 2π]
    def f(x):
        return np.sin(x) + 0.5*np.sin(2*x) + 0.25*np.sin(3*x)

    f_vals = f(x)

    # Map to ℓ^2: compute Fourier coefficients
    N_max = 20
    coeffs = []
    for n in range(-N_max, N_max + 1):
        c_n = np.trapz(f_vals * np.exp(-1j * n * x), x) / (2 * np.pi)
        coeffs.append(c_n)

    # The ℓ^2 representation (reindexed to 1, 2, 3, ...)
    ell2_rep = np.array(coeffs)

    print("Isomorphism L²[0, 2π] ↔ ℓ²")
    print("=" * 60)

    # Verify isometry: ||f||_{L^2} = ||c||_{ℓ^2}
    norm_L2 = np.sqrt(np.trapz(np.abs(f_vals)**2, x) / (2*np.pi))
    norm_ell2 = np.linalg.norm(ell2_rep)

    print(f"||f||_{'{L^2}'} = {norm_L2:.10f}")
    print(f"||c||_{'{ℓ^2}'} = {norm_ell2:.10f}")
    print(f"Difference: {abs(norm_L2 - norm_ell2):.2e}")

    # Reconstruct f from ℓ^2 coefficients
    f_reconstructed = np.zeros_like(x, dtype=complex)
    for idx, n in enumerate(range(-N_max, N_max + 1)):
        f_reconstructed += ell2_rep[idx] * np.exp(1j * n * x)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original function
    ax = axes[0, 0]
    ax.plot(x, f_vals, 'b-', linewidth=2, label='Original $f(x)$')
    ax.plot(x, np.real(f_reconstructed), 'r--', linewidth=1.5,
           label='Reconstructed')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Function in $L^2[0, 2\\pi]$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ℓ^2 coefficients
    ax = axes[0, 1]
    n_vals = range(-N_max, N_max + 1)
    ax.stem(n_vals, np.abs(ell2_rep), linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$|c_n|$')
    ax.set_title('$\\ell^2$ Representation (Fourier Coefficients)')
    ax.grid(True, alpha=0.3)

    # Demonstrate inverse map
    ax = axes[1, 0]

    # Start with a sequence in ℓ^2
    ell2_input = np.zeros(2*N_max + 1, dtype=complex)
    ell2_input[N_max + 1] = 1.0  # c_1 = 1 (sin(x)/i + cos(x))/2
    ell2_input[N_max - 1] = 1.0  # c_{-1} = 1

    # Map to L^2
    g_vals = np.zeros_like(x, dtype=complex)
    for idx, n in enumerate(range(-N_max, N_max + 1)):
        g_vals += ell2_input[idx] * np.exp(1j * n * x)

    ax.plot(x, np.real(g_vals), 'g-', linewidth=2, label='Re(g)')
    ax.plot(x, np.imag(g_vals), 'm--', linewidth=2, label='Im(g)')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$g(x)$')
    ax.set_title('Inverse Map: $\\ell^2 \\to L^2$ (c_1 = c_{-1} = 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Norm preservation
    ax = axes[1, 1]
    norms_L2 = []
    norms_ell2 = []

    test_functions = [
        ('sin(x)', np.sin(x)),
        ('cos(x)', np.cos(x)),
        ('x', x),
        ('exp(-x/2)', np.exp(-x/2))
    ]

    for name, func in test_functions:
        norm_L2 = np.sqrt(np.trapz(np.abs(func)**2, x) / (2*np.pi))

        coeffs = []
        for n in range(-50, 51):
            c_n = np.trapz(func * np.exp(-1j * n * x), x) / (2 * np.pi)
            coeffs.append(c_n)
        norm_ell2 = np.linalg.norm(coeffs)

        norms_L2.append(norm_L2)
        norms_ell2.append(norm_ell2)

    names = [name for name, _ in test_functions]
    x_pos = np.arange(len(names))
    width = 0.35

    ax.bar(x_pos - width/2, norms_L2, width, label='$\\|f\\|_{L^2}$', alpha=0.7)
    ax.bar(x_pos + width/2, norms_ell2, width, label='$\\|c\\|_{\\ell^2}$', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_ylabel('Norm')
    ax.set_title('Isometry Verification: $\\|f\\|_{L^2} = \\|U(f)\\|_{\\ell^2}$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('isomorphism_demo.png', dpi=150)
    plt.show()

# ============================================================
# Part 2: Riesz Representation Theorem
# ============================================================

def demonstrate_riesz():
    """
    Demonstrate the Riesz representation theorem.
    """
    x = np.linspace(0, 1, 1000)

    print("\nRiesz Representation Theorem Demonstration")
    print("=" * 60)

    # Functional: f(g) = ∫₀¹ y(x) g(x) dx for various y

    test_representers = [
        ('y(x) = x', x),
        ('y(x) = x^2', x**2),
        ('y(x) = sin(πx)', np.sin(np.pi*x)),
        ('y(x) = 1', np.ones_like(x))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, y) in enumerate(test_representers):
        ax = axes[idx]

        # The functional is f(g) = <y, g>
        # Test on some functions
        test_g = [
            ('g = 1', np.ones_like(x)),
            ('g = x', x),
            ('g = x^2', x**2),
            ('g = sin(πx)', np.sin(np.pi*x))
        ]

        results = []
        for g_name, g in test_g:
            # Compute f(g) = ∫ y(x) g(x) dx
            f_g = np.trapz(y * g, x)
            results.append((g_name, f_g))

        # Display
        ax.bar([r[0] for r in results], [r[1] for r in results],
              color='steelblue', alpha=0.7)
        ax.set_ylabel('$f(g) = \\langle y, g \\rangle$')
        ax.set_title(f'Functional represented by {name}')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3)

        # Compute norm of functional
        norm_y = np.sqrt(np.trapz(y**2, x))
        ax.annotate(f'$\\|y\\|_{{L^2}} = {norm_y:.4f}$',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        print(f"\n{name}:")
        print(f"  ||y||_{'{L^2}'} = {norm_y:.6f}")
        print(f"  This is also ||f|| (operator norm)")

    plt.tight_layout()
    plt.savefig('riesz_representation.png', dpi=150)
    plt.show()

# ============================================================
# Part 3: Separability Visualization
# ============================================================

def visualize_separability():
    """
    Visualize what separability means for Hilbert spaces.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Part 1: Dense subset approximation in ℓ^2
    ax = axes[0, 0]

    # Target: a sequence in ℓ^2
    target = np.array([1/n for n in range(1, 21)])

    # Rational approximations
    approx_levels = [1, 2, 3, 4]  # decimal places
    colors = plt.cm.viridis(np.linspace(0, 1, len(approx_levels)))

    for level, color in zip(approx_levels, colors):
        approx = np.round(target, level)
        error = np.linalg.norm(target - approx)
        ax.plot(range(1, 21), approx, 'o-', color=color, markersize=4,
               label=f'{level} decimals (error = {error:.4f})')

    ax.plot(range(1, 21), target, 'k--', linewidth=2, label='Target (1/n)')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$x_n$')
    ax.set_title('Dense Subset Approximation in $\\ell^2$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Part 2: Polynomial approximation in L^2
    ax = axes[0, 1]
    x = np.linspace(-1, 1, 1000)

    # Target function
    target_func = np.abs(x)

    # Polynomial approximations (use Chebyshev)
    from numpy.polynomial import chebyshev

    for deg in [2, 4, 8, 16]:
        # Least squares fit
        coeffs = np.polynomial.chebyshev.chebfit(x, target_func, deg)
        approx_func = np.polynomial.chebyshev.chebval(x, coeffs)
        error = np.sqrt(np.trapz((target_func - approx_func)**2, x))
        ax.plot(x, approx_func, linewidth=1.5,
               label=f'deg {deg} (L² error = {error:.4f})', alpha=0.7)

    ax.plot(x, target_func, 'k--', linewidth=2, label='$|x|$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Polynomial Approximation in $L^2[-1, 1]$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Part 3: Orthonormal basis completeness
    ax = axes[1, 0]

    # Show how orthonormal expansion captures more of the signal
    x = np.linspace(0, 2*np.pi, 500)
    target_func = np.sign(np.sin(x))  # Square wave

    N_values = [1, 3, 9, 27]
    for N in N_values:
        approx = np.zeros_like(x)
        for n in range(1, N+1, 2):  # Odd terms only
            approx += (4/(n*np.pi)) * np.sin(n*x)

        # L^2 error
        error = np.sqrt(np.trapz((target_func - approx)**2, x) / (2*np.pi))
        ax.plot(x, approx, linewidth=1.5,
               label=f'N = {N} (L² error = {error:.4f})', alpha=0.7)

    ax.plot(x, target_func, 'k--', linewidth=2, label='Square wave')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Fourier Series: ONB Expansion Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Part 4: Classification theorem visualization
    ax = axes[1, 1]

    # Show that L^2 and ℓ^2 have the same "shape"
    # by plotting similar decay patterns

    # In ℓ^2: geometric decay
    n = np.arange(1, 31)
    ell2_decay = 0.9**n

    # In L^2: Fourier coefficients of a smooth function
    x = np.linspace(0, 2*np.pi, 1000)
    f = np.exp(np.cos(x))  # Smooth periodic function
    L2_decay = []
    for k in range(1, 31):
        c_k = np.abs(np.trapz(f * np.exp(-1j * k * x), x) / (2*np.pi))
        L2_decay.append(c_k)

    ax.semilogy(n, ell2_decay, 'b-o', markersize=4, label='$\\ell^2$: $(0.9)^n$')
    ax.semilogy(n, L2_decay, 'r-s', markersize=4, label='$L^2$: Fourier of $e^{\\cos x}$')
    ax.set_xlabel('$n$')
    ax.set_ylabel('Magnitude')
    ax.set_title('Classification: $L^2$ and $\\ell^2$ Are Isomorphic')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('separability.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: Quantum State Representations
# ============================================================

def quantum_representations():
    """
    Show how the same quantum state looks in different representations.
    """
    x = np.linspace(-10, 10, 1000)

    # Harmonic oscillator eigenstate (energy representation)
    def psi_n(x, n):
        Hn = hermite(n)
        norm = 1 / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
        return norm * Hn(x) * np.exp(-x**2 / 2)

    # Coherent state (semiclassical)
    alpha = 2.0  # Complex amplitude
    n_max = 20

    # Energy representation: c_n = e^{-|α|²/2} α^n / sqrt(n!)
    coeffs_energy = []
    for n in range(n_max):
        c_n = np.exp(-abs(alpha)**2/2) * alpha**n / np.sqrt(np.math.factorial(n))
        coeffs_energy.append(c_n)

    # Position representation: sum c_n ψ_n(x)
    psi_position = np.zeros_like(x, dtype=complex)
    for n, c_n in enumerate(coeffs_energy):
        psi_position += c_n * psi_n(x, n)

    # Momentum representation (Fourier transform)
    dp = 2 * np.pi / (x[-1] - x[0])
    p = np.fft.fftfreq(len(x), d=(x[1]-x[0])) * 2 * np.pi
    psi_momentum = np.fft.fft(psi_position) * (x[1] - x[0]) / np.sqrt(2*np.pi)

    # Sort for plotting
    p_sorted = np.fft.fftshift(p)
    psi_momentum_sorted = np.fft.fftshift(psi_momentum)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Energy representation
    ax = axes[0, 0]
    ax.bar(range(n_max), np.abs(coeffs_energy)**2, color='steelblue', alpha=0.7)
    ax.set_xlabel('Energy level $n$')
    ax.set_ylabel('$|c_n|^2$')
    ax.set_title('Energy Representation (Coefficient Basis)')
    ax.grid(True, alpha=0.3)

    # Position representation
    ax = axes[0, 1]
    ax.fill_between(x, np.abs(psi_position)**2, alpha=0.5, color='green')
    ax.plot(x, np.abs(psi_position)**2, 'g-', linewidth=2)
    ax.set_xlabel('Position $x$')
    ax.set_ylabel('$|\\psi(x)|^2$')
    ax.set_title('Position Representation ($L^2$ of x)')
    ax.set_xlim(-6, 10)
    ax.grid(True, alpha=0.3)

    # Momentum representation
    ax = axes[1, 0]
    mask = np.abs(p_sorted) < 5
    ax.fill_between(p_sorted[mask], np.abs(psi_momentum_sorted[mask])**2, alpha=0.5, color='red')
    ax.plot(p_sorted[mask], np.abs(psi_momentum_sorted[mask])**2, 'r-', linewidth=2)
    ax.set_xlabel('Momentum $p$')
    ax.set_ylabel('$|\\tilde{\\psi}(p)|^2$')
    ax.set_title('Momentum Representation ($L^2$ of p)')
    ax.grid(True, alpha=0.3)

    # All three norms should be equal (isometry)
    ax = axes[1, 1]
    norm_energy = np.sum(np.abs(coeffs_energy)**2)
    norm_position = np.trapz(np.abs(psi_position)**2, x)
    norm_momentum = np.trapz(np.abs(psi_momentum_sorted)**2, p_sorted)

    norms = [norm_energy, norm_position, norm_momentum / (2*np.pi)]
    labels = ['Energy\n$\\sum|c_n|^2$', 'Position\n$\\int|\\psi(x)|^2dx$',
              'Momentum\n$\\int|\\tilde{\\psi}(p)|^2dp/(2\\pi)$']

    bars = ax.bar(labels, norms, color=['steelblue', 'green', 'red'], alpha=0.7)
    ax.axhline(y=1, color='k', linestyle='--', label='Expected = 1')
    ax.set_ylabel('Norm squared')
    ax.set_title('All Representations Have Same Norm (Isometry)')
    ax.legend()
    ax.set_ylim(0, 1.2)

    for bar, norm in zip(bars, norms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{norm:.4f}', ha='center', fontsize=10)

    plt.suptitle(f'Coherent State $|\\alpha = {alpha}\\rangle$ in Three Representations',
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('quantum_representations.png', dpi=150)
    plt.show()

    print("\nQuantum State Representations")
    print("=" * 60)
    print(f"State: Coherent state |α = {alpha}⟩")
    print(f"\nNorm in different representations:")
    print(f"  Energy:   {norm_energy:.10f}")
    print(f"  Position: {norm_position:.10f}")
    print(f"  Momentum: {norm_momentum/(2*np.pi):.10f}")
    print("\nAll equal to 1 (normalized state) - demonstrating isometry!")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 237: Separable Hilbert Spaces - Computational Lab")
    print("=" * 60)

    print("\n1. Demonstrating L² ↔ ℓ² isomorphism...")
    demonstrate_L2_ell2_isomorphism()

    print("\n2. Demonstrating Riesz representation theorem...")
    demonstrate_riesz()

    print("\n3. Visualizing separability...")
    visualize_separability()

    print("\n4. Quantum state in different representations...")
    quantum_representations()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## 9. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Separable** | Has countable dense subset |
| **Dimension** | Cardinality of any orthonormal basis |
| **Isometric Isomorphism** | Bijective linear map preserving norm |

### Key Theorems

$$\boxed{\begin{aligned}
&\text{Characterization:} && \mathcal{H} \text{ separable} \Leftrightarrow \text{countable ONB exists} \\[5pt]
&\text{Classification:} && \text{All sep. inf-dim Hilbert spaces} \cong \ell^2 \\[5pt]
&\text{Riesz Representation:} && \forall f \in \mathcal{H}^*, \exists! y: f(x) = \langle y, x\rangle
\end{aligned}}$$

### Key Insights

1. **Separability = countable ONB** — equivalent conditions
2. **Only one infinite-dimensional separable Hilbert space** — all isomorphic to $$\ell^2$$
3. **$$L^2 \cong \ell^2$$** — functions and sequences are "the same"
4. **Hilbert spaces are self-dual** — Riesz representation
5. **Quantum state spaces are separable** — countable energy levels suffice

---

## 10. Daily Checklist

- [ ] I can define separability for Hilbert spaces
- [ ] I understand the characterization theorem (separable ⟺ countable ONB)
- [ ] I can state the classification theorem for separable Hilbert spaces
- [ ] I can construct explicit isomorphisms $$L^2 \to \ell^2$$
- [ ] I understand the Riesz representation theorem
- [ ] I can relate these concepts to quantum mechanics
- [ ] I completed the computational lab exercises

---

## 11. Preview: Day 238

Tomorrow's review will synthesize all the concepts from Week 34: normed spaces, Banach spaces, inner products, Hilbert spaces, orthonormal bases, Parseval's identity, and separability. We'll work through comprehensive problems that tie everything together and solidify the connection to quantum mechanics.

---

*"The remarkable fact that all infinite-dimensional separable Hilbert spaces are isomorphic means that quantum mechanics, in all its guises, takes place in essentially the same mathematical arena: the space ℓ²."*
