# Day 244: Compact Operators

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Definition, characterizations, properties |
| Afternoon | 3 hours | Problems: Hilbert-Schmidt and trace class operators |
| Evening | 2 hours | Computational lab: Singular value decomposition |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** compact operators and explain their significance
2. **Characterize** compact operators as limits of finite-rank operators
3. **Prove** basic properties: compactness preserved under limits, products, adjoints
4. **Work with** Hilbert-Schmidt operators and their inner product structure
5. **Introduce** trace class operators and the trace functional
6. **Connect** compact operators to quantum mechanics (density matrices, Green's functions)

---

## 1. Core Content: Compact Operators

### 1.1 Definition and Motivation

**Definition**: A bounded linear operator $K: \mathcal{H} \to \mathcal{K}$ is **compact** if it maps the unit ball to a relatively compact set. That is, for any bounded sequence $(x_n)$ in $\mathcal{H}$, the sequence $(Kx_n)$ has a convergent subsequence.

Equivalently, $K$ is compact if $\overline{K(B_1)}$ is compact, where $B_1 = \{x : \|x\| \leq 1\}$.

**Notation**: The set of compact operators from $\mathcal{H}$ to $\mathcal{K}$ is denoted $\mathcal{K}(\mathcal{H}, \mathcal{K})$ or $\mathcal{K}(\mathcal{H})$ when $\mathcal{H} = \mathcal{K}$.

**Motivation**: Compact operators are the "nearly finite-dimensional" operators. They inherit many nice properties of matrices:
- Discrete spectrum (except possibly 0)
- Eigenspaces are finite-dimensional
- Spectral theorem applies

### 1.2 Finite-Rank Operators

**Definition**: An operator $F: \mathcal{H} \to \mathcal{K}$ has **finite rank** if $\dim(\text{ran}(F)) < \infty$.

Every finite-rank operator can be written as:
$$Fx = \sum_{j=1}^n \langle f_j, x \rangle g_j$$

for some $f_j \in \mathcal{H}$, $g_j \in \mathcal{K}$.

In Dirac notation: $F = \sum_{j=1}^n |g_j\rangle\langle f_j|$.

**Theorem**: Every finite-rank operator is compact.

**Proof**: The image of the unit ball under $F$ lies in $\text{span}\{g_1, \ldots, g_n\}$, a finite-dimensional space. Bounded sets in finite dimensions are relatively compact. $\square$

### 1.3 Characterization Theorem

**Theorem**: For a bounded operator $K$, the following are equivalent:

1. $K$ is compact
2. $K$ is the norm limit of finite-rank operators
3. For every orthonormal sequence $(e_n)$, $Ke_n \to 0$
4. $K$ maps weakly convergent sequences to norm convergent sequences

**Proof sketch**:

**(2) $\Rightarrow$ (1)**: If $K = \lim_n F_n$ with $F_n$ finite-rank, and $(x_j)$ is bounded, then $(F_n x_j)_j$ has a convergent subsequence (finite rank). A diagonalization argument gives a subsequence $(x_{j_k})$ such that $(K x_{j_k})$ converges.

**(1) $\Rightarrow$ (3)**: Let $(e_n)$ be orthonormal, hence $e_n \to 0$ weakly. If $(Ke_n)$ doesn't converge to 0, some subsequence satisfies $\|Ke_{n_k}\| \geq \varepsilon > 0$. By compactness, a further subsequence $(Ke_{n_{k_l}})$ converges to some $y \neq 0$. But $e_{n_{k_l}} \to 0$ weakly, and compact operators map weakly convergent to norm convergent, so $Ke_{n_{k_l}} \to K(0) = 0$. Contradiction. $\square$

---

## 2. Properties of Compact Operators

### 2.1 Algebraic Properties

**Theorem**: Let $K, L$ be compact and $A, B$ bounded. Then:

1. $K + L$ is compact
2. $\alpha K$ is compact for any scalar $\alpha$
3. $AK$ and $KB$ are compact
4. $K^\dagger$ is compact

**Corollary**: $\mathcal{K}(\mathcal{H})$ is a **two-sided ideal** in $\mathcal{B}(\mathcal{H})$.

**Proof of (4)**: We use: $K$ compact $\Leftrightarrow$ $Ke_n \to 0$ for any ONS $(e_n)$.

Let $(e_n)$ be orthonormal. We need $K^\dagger e_n \to 0$.

$$\|K^\dagger e_n\|^2 = \langle K^\dagger e_n, K^\dagger e_n \rangle = \langle KK^\dagger e_n, e_n \rangle$$

Since $KK^\dagger$ is compact (product of compact and bounded), and $e_n \to 0$ weakly, we have $KK^\dagger e_n \to 0$ in norm.

Thus $|\langle KK^\dagger e_n, e_n \rangle| \leq \|KK^\dagger e_n\| \|e_n\| = \|KK^\dagger e_n\| \to 0$. $\square$

### 2.2 Spectral Properties

**Theorem (Spectral Properties of Compact Operators)**:

Let $K$ be a compact operator on a Hilbert space $\mathcal{H}$.

1. If $\lambda \neq 0$ is in $\sigma(K)$, then $\lambda$ is an eigenvalue
2. Each nonzero eigenvalue has finite multiplicity
3. Nonzero eigenvalues have no accumulation point except possibly 0
4. If $\mathcal{H}$ is infinite-dimensional, then $0 \in \sigma(K)$

**Remark**: This is a special case of the Riesz-Schauder theorem. For self-adjoint compact operators, the spectral theorem gives even more (orthonormal eigenbasis).

---

## 3. Hilbert-Schmidt Operators

### 3.1 Definition

**Definition**: A bounded operator $A: \mathcal{H} \to \mathcal{K}$ is **Hilbert-Schmidt** if for some (hence any) orthonormal basis $\{e_n\}$ of $\mathcal{H}$:

$$\boxed{\|A\|_{\text{HS}}^2 = \sum_n \|Ae_n\|^2 < \infty}$$

The quantity $\|A\|_{\text{HS}}$ is the **Hilbert-Schmidt norm**.

**Theorem**: The Hilbert-Schmidt norm is independent of the choice of orthonormal basis.

**Proof**: For another ONB $\{f_m\}$:
$$\sum_n \|Ae_n\|^2 = \sum_n \sum_m |\langle Ae_n, f_m\rangle|^2 = \sum_m \sum_n |\langle e_n, A^\dagger f_m\rangle|^2 = \sum_m \|A^\dagger f_m\|^2$$

So $\|A\|_{\text{HS}} = \|A^\dagger\|_{\text{HS}}$. The result follows by symmetry. $\square$

### 3.2 Properties of Hilbert-Schmidt Operators

**Theorem**:
1. Every Hilbert-Schmidt operator is compact
2. $\|A\| \leq \|A\|_{\text{HS}}$
3. $\|A\|_{\text{HS}} = \|A^\dagger\|_{\text{HS}}$
4. $\|AB\|_{\text{HS}} \leq \|A\| \|B\|_{\text{HS}}$ and $\|AB\|_{\text{HS}} \leq \|A\|_{\text{HS}} \|B\|$
5. The Hilbert-Schmidt operators form a two-sided ideal

### 3.3 Inner Product Structure

**Theorem**: The space of Hilbert-Schmidt operators $\mathcal{L}^2(\mathcal{H})$ is a Hilbert space with inner product:

$$\boxed{\langle A, B \rangle_{\text{HS}} = \sum_n \langle Ae_n, Be_n \rangle = \text{Tr}(A^\dagger B)}$$

**Important**: $\mathcal{L}^2(\mathcal{H})$ is itself a Hilbert space!

### 3.4 Integral Operators as Hilbert-Schmidt

**Theorem**: An integral operator $K: L^2[a,b] \to L^2[a,b]$ defined by:
$$(Kf)(x) = \int_a^b k(x,y) f(y) \, dy$$

is Hilbert-Schmidt if and only if $k \in L^2([a,b] \times [a,b])$, and:

$$\|K\|_{\text{HS}} = \|k\|_{L^2} = \left(\int_a^b \int_a^b |k(x,y)|^2 \, dx \, dy\right)^{1/2}$$

---

## 4. Trace Class Operators

### 4.1 Definition

**Definition**: A bounded operator $A$ is **trace class** if:

$$\|A\|_1 = \text{Tr}(|A|) = \sum_n \langle |A|e_n, e_n \rangle < \infty$$

where $|A| = \sqrt{A^\dagger A}$.

**Theorem**: Trace class $\subset$ Hilbert-Schmidt $\subset$ Compact.

### 4.2 The Trace

For a trace class operator $A$, the **trace** is:

$$\boxed{\text{Tr}(A) = \sum_n \langle Ae_n, e_n \rangle}$$

This is independent of the orthonormal basis and satisfies:
- $\text{Tr}(A + B) = \text{Tr}(A) + \text{Tr}(B)$
- $\text{Tr}(\alpha A) = \alpha \text{Tr}(A)$
- $\text{Tr}(AB) = \text{Tr}(BA)$ (cyclic property)
- $|\text{Tr}(A)| \leq \|A\|_1$

### 4.3 Singular Value Decomposition

**Theorem (SVD for Compact Operators)**: Every compact operator $K$ can be written as:

$$Kx = \sum_n s_n \langle f_n, x \rangle g_n$$

where:
- $(s_n)$ are the **singular values** (non-negative, decreasing to 0)
- $(f_n)$ is an ONB of $(\ker K)^\perp$
- $(g_n)$ is an ONB of $\overline{\text{ran}(K)}$
- $Kf_n = s_n g_n$ and $K^\dagger g_n = s_n f_n$

**Connection to norms**:
- $\|K\| = s_1$ (operator norm = largest singular value)
- $\|K\|_{\text{HS}}^2 = \sum_n s_n^2$
- $\|K\|_1 = \sum_n s_n$ (trace norm)

---

## 5. Quantum Mechanics Connection

### 5.1 Density Matrices as Trace Class

In quantum mechanics, **density matrices** (density operators) are:
- Self-adjoint: $\rho = \rho^\dagger$
- Positive: $\rho \geq 0$
- Trace one: $\text{Tr}(\rho) = 1$

**Theorem**: Every density matrix is trace class.

For a pure state: $\rho = |\psi\rangle\langle\psi|$ (rank-1 projection).

For a mixed state: $\rho = \sum_n p_n |\psi_n\rangle\langle\psi_n|$ with $p_n \geq 0$, $\sum p_n = 1$.

### 5.2 Expectation Values

For an observable $A$ and density matrix $\rho$:

$$\langle A \rangle = \text{Tr}(\rho A)$$

If $A$ is bounded and $\rho$ is trace class, this is well-defined.

### 5.3 Green's Functions and Resolvents

The **Green's function** (or propagator) in quantum mechanics:

$$G(E) = (E - H)^{-1}$$

For suitable $E$, this is often compact (especially for potentials that vanish at infinity).

### 5.4 The Thermal Density Matrix

At temperature $T$, the thermal state is:

$$\rho_\beta = \frac{e^{-\beta H}}{Z}, \quad Z = \text{Tr}(e^{-\beta H})$$

where $\beta = 1/(k_B T)$.

If $H$ has compact resolvent, $e^{-\beta H}$ is trace class for $\beta > 0$.

---

## 6. Worked Examples

### Example 1: The Volterra Operator is Compact

**Problem**: Show that the Volterra operator $V: L^2[0,1] \to L^2[0,1]$ defined by $(Vf)(x) = \int_0^x f(t) \, dt$ is compact.

**Solution**:

**Method 1: Hilbert-Schmidt**

The kernel is $k(x,t) = \mathbf{1}_{t \leq x}$.

$$\|V\|_{\text{HS}}^2 = \int_0^1 \int_0^1 |k(x,t)|^2 \, dt \, dx = \int_0^1 \int_0^x 1 \, dt \, dx = \int_0^1 x \, dx = \frac{1}{2} < \infty$$

So $V$ is Hilbert-Schmidt, hence compact.

**Method 2: Explicit singular values**

One can show the singular values of $V$ are $s_n = \frac{1}{(n-1/2)\pi}$, which satisfy $\sum s_n^2 < \infty$ (Hilbert-Schmidt) but $\sum s_n = \infty$ (not trace class). $\square$

---

### Example 2: Hilbert-Schmidt Norm Calculation

**Problem**: Find the Hilbert-Schmidt norm of the operator $A: \ell^2 \to \ell^2$ defined by $(Ax)_n = x_n / n$.

**Solution**:

The standard basis $\{e_n\}$ is an ONB. We have $Ae_n = \frac{1}{n}e_n$.

$$\|A\|_{\text{HS}}^2 = \sum_{n=1}^\infty \|Ae_n\|^2 = \sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$$

$$\boxed{\|A\|_{\text{HS}} = \frac{\pi}{\sqrt{6}}}$$

Since this converges, $A$ is Hilbert-Schmidt (hence compact).

For comparison: $\|A\| = 1$ (largest diagonal entry), $\|A\|_1 = \sum 1/n = \infty$ (not trace class). $\square$

---

### Example 3: Thermal Density Matrix

**Problem**: Find the density matrix for a quantum harmonic oscillator at temperature $T$.

**Solution**:

The Hamiltonian is $H = \hbar\omega(N + 1/2)$ where $N|n\rangle = n|n\rangle$.

$$e^{-\beta H} = e^{-\beta\hbar\omega/2} \sum_{n=0}^\infty e^{-n\beta\hbar\omega} |n\rangle\langle n|$$

The partition function:
$$Z = \text{Tr}(e^{-\beta H}) = e^{-\beta\hbar\omega/2} \sum_{n=0}^\infty e^{-n\beta\hbar\omega} = \frac{e^{-\beta\hbar\omega/2}}{1 - e^{-\beta\hbar\omega}}$$

The density matrix:
$$\rho_\beta = \frac{e^{-\beta H}}{Z} = (1 - e^{-\beta\hbar\omega}) \sum_{n=0}^\infty e^{-n\beta\hbar\omega} |n\rangle\langle n|$$

Each $p_n = (1 - e^{-\beta\hbar\omega}) e^{-n\beta\hbar\omega}$ is a geometric distribution (Bose-Einstein statistics).

**Verify trace class**: $\text{Tr}(\rho_\beta) = \sum_n p_n = 1$ ✓

**At high temperature** ($\beta \to 0$): All states equally populated.
**At low temperature** ($\beta \to \infty$): Ground state dominates, $\rho \to |0\rangle\langle 0|$. $\square$

---

## 7. Practice Problems

### Level 1: Direct Application

1. Show that the projection $P_n$ onto $\text{span}\{e_1, \ldots, e_n\}$ in $\ell^2$ is finite-rank (hence compact). What is $\|P_n\|_{\text{HS}}$?

2. Prove that if $K$ is compact and $A$ is bounded, then $AK$ and $KA$ are compact.

3. Find the Hilbert-Schmidt norm of the integral operator with kernel $k(x,y) = xy$ on $L^2[0,1]$.

### Level 2: Intermediate

4. **Prove**: The Hilbert-Schmidt operators form a two-sided ideal in $\mathcal{B}(\mathcal{H})$.

5. Let $K$ be a compact self-adjoint operator with eigenvalues $\lambda_n \to 0$. Show:
   - $K$ is Hilbert-Schmidt $\Leftrightarrow \sum_n \lambda_n^2 < \infty$
   - $K$ is trace class $\Leftrightarrow \sum_n |\lambda_n| < \infty$

6. **Quantum Connection**: A density matrix has eigenvalues $p_n = 1/2^n$ for $n = 1, 2, 3, \ldots$. Verify it's trace class and compute $\text{Tr}(\rho)$ and $\text{Tr}(\rho^2)$.

### Level 3: Challenging

7. **Prove**: An operator $K$ on $\ell^2$ given by infinite matrix $(k_{ij})$ is Hilbert-Schmidt if and only if $\sum_{i,j} |k_{ij}|^2 < \infty$, with $\|K\|_{\text{HS}}^2 = \sum_{i,j}|k_{ij}|^2$.

8. **Prove**: The identity operator $I$ on an infinite-dimensional Hilbert space is not compact. (Hint: Consider an orthonormal sequence.)

9. **Research problem**: Investigate the Schatten $p$-classes $\mathcal{L}^p(\mathcal{H})$ defined by $\sum_n s_n^p < \infty$ where $s_n$ are singular values. Show $\mathcal{L}^1 = $ trace class, $\mathcal{L}^2 = $ Hilbert-Schmidt, $\mathcal{L}^\infty = $ compact.

---

## 8. Computational Lab: Compact Operators and SVD

```python
"""
Day 244 Computational Lab: Compact Operators
Singular value decomposition, Hilbert-Schmidt norm, and trace class
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, norm
from scipy.integrate import dblquad

# ============================================================
# Part 1: Singular Value Decomposition
# ============================================================

def svd_demonstration():
    """
    Demonstrate SVD for finite-dimensional operators (matrices).
    """
    print("=" * 60)
    print("Part 1: Singular Value Decomposition")
    print("=" * 60)

    np.random.seed(42)

    # Create a low-rank matrix (approximately compact behavior)
    m, n = 50, 30
    rank = 5

    # A = U @ S @ V^T where S has only `rank` nonzero singular values
    U_true = np.linalg.qr(np.random.randn(m, rank))[0]
    V_true = np.linalg.qr(np.random.randn(n, rank))[0]
    s_true = np.array([10, 5, 2, 1, 0.5])

    A = U_true @ np.diag(s_true) @ V_true.T

    # Add small noise
    A_noisy = A + 0.01 * np.random.randn(m, n)

    # Compute SVD
    U, s, Vt = svd(A_noisy, full_matrices=False)

    print(f"Matrix shape: {A_noisy.shape}")
    print(f"Singular values (first 10): {np.round(s[:10], 4)}")
    print(f"True singular values: {s_true}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot singular values
    axes[0].semilogy(s, 'b-o', markersize=4)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Singular value')
    axes[0].set_title('Singular Value Decay')
    axes[0].grid(True, alpha=0.3)

    # Different norms
    op_norm = s[0]  # ||A|| = s_1
    hs_norm = np.sqrt(np.sum(s**2))  # ||A||_HS = sqrt(sum s_n^2)
    tr_norm = np.sum(s)  # ||A||_1 = sum s_n

    axes[1].bar(['||A||', '||A||_HS', '||A||_1'], [op_norm, hs_norm, tr_norm],
               color=['blue', 'green', 'red'], alpha=0.7)
    axes[1].set_ylabel('Norm value')
    axes[1].set_title('Different Operator Norms')

    # Low-rank approximation error
    ranks = range(1, min(m, n) + 1)
    errors = []
    for r in ranks:
        A_r = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
        errors.append(norm(A_noisy - A_r, ord=2))

    axes[2].semilogy(ranks, errors, 'r-o', markersize=3)
    axes[2].set_xlabel('Rank $r$')
    axes[2].set_ylabel('$||A - A_r||$')
    axes[2].set_title('Low-Rank Approximation Error')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('svd_demo.png', dpi=150)
    plt.show()

    print(f"\nNorms:")
    print(f"  Operator norm ||A|| = s_1 = {op_norm:.4f}")
    print(f"  Hilbert-Schmidt ||A||_HS = {hs_norm:.4f}")
    print(f"  Trace norm ||A||_1 = {tr_norm:.4f}")

# ============================================================
# Part 2: Integral Operators and Hilbert-Schmidt Norm
# ============================================================

def integral_operator_hs():
    """
    Compute Hilbert-Schmidt norm of integral operators.
    """
    print("\n" + "=" * 60)
    print("Part 2: Integral Operators as Hilbert-Schmidt")
    print("=" * 60)

    # Discretize the Volterra operator (Vf)(x) = int_0^x f(t) dt
    N = 100
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Kernel k(x,t) = 1 if t <= x, 0 otherwise
    K_volterra = np.tril(np.ones((N, N))) * dx

    # Kernel k(x,y) = xy
    X, Y = np.meshgrid(x, x)
    K_xy = X * Y * dx

    # Kernel k(x,y) = exp(-|x-y|)
    K_exp = np.exp(-np.abs(X - Y)) * dx

    operators = [
        (K_volterra, "Volterra: $k(x,t) = \mathbf{1}_{t \leq x}$"),
        (K_xy, "Rank-1: $k(x,y) = xy$"),
        (K_exp, "Exponential: $k(x,y) = e^{-|x-y|}$")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (K, title) in enumerate(operators):
        # Compute SVD
        U, s, Vt = svd(K)

        # Hilbert-Schmidt norm (should equal L^2 norm of kernel)
        hs_norm_svd = np.sqrt(np.sum(s**2))
        hs_norm_frob = norm(K, 'fro')

        print(f"\n{title}:")
        print(f"  ||K||_HS (from SVD) = {hs_norm_svd:.6f}")
        print(f"  ||K||_F (Frobenius) = {hs_norm_frob:.6f}")
        print(f"  First 5 singular values: {np.round(s[:5], 4)}")

        # Plot kernel
        im = axes[0, i].imshow(K / dx, extent=[0, 1, 1, 0], cmap='viridis')
        axes[0, i].set_title(f'{title}\n$||K||_{{HS}} = {hs_norm_svd:.3f}$')
        axes[0, i].set_xlabel('y')
        axes[0, i].set_ylabel('x')
        plt.colorbar(im, ax=axes[0, i])

        # Plot singular values
        axes[1, i].semilogy(s[:50], 'b-o', markersize=3)
        axes[1, i].set_xlabel('Index')
        axes[1, i].set_ylabel('$s_n$')
        axes[1, i].set_title('Singular Value Decay')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('integral_operators_hs.png', dpi=150)
    plt.show()

# ============================================================
# Part 3: Trace Class and Density Matrices
# ============================================================

def trace_class_demo():
    """
    Demonstrate trace class operators and density matrices.
    """
    print("\n" + "=" * 60)
    print("Part 3: Trace Class Operators")
    print("=" * 60)

    # Example 1: Density matrix for thermal state
    # H = diag(0, 1, 2, ..., N-1) (harmonic oscillator energies)
    N = 20
    beta = 1.0  # Inverse temperature

    energies = np.arange(N)
    exp_factors = np.exp(-beta * energies)
    Z = np.sum(exp_factors)

    # Density matrix (diagonal)
    rho = np.diag(exp_factors / Z)

    print("Thermal density matrix (harmonic oscillator, β=1):")
    print(f"  First 5 diagonal elements: {np.round(np.diag(rho)[:5], 4)}")
    print(f"  Tr(ρ) = {np.trace(rho):.6f}")
    print(f"  Tr(ρ²) = {np.trace(rho @ rho):.6f} (purity)")

    # Trace norm = sum of eigenvalues (for positive operator)
    eigenvalues = np.linalg.eigvalsh(rho)
    tr_norm = np.sum(np.abs(eigenvalues))
    print(f"  ||ρ||_1 = {tr_norm:.6f}")

    # Example 2: Pure state density matrix
    psi = np.random.randn(N) + 1j * np.random.randn(N)
    psi = psi / np.linalg.norm(psi)
    rho_pure = np.outer(psi, psi.conj())

    print("\nPure state density matrix:")
    print(f"  Tr(ρ) = {np.trace(rho_pure):.6f}")
    print(f"  Tr(ρ²) = {np.trace(rho_pure @ rho_pure):.6f} (should be 1)")
    print(f"  ρ² = ρ: {np.allclose(rho_pure @ rho_pure, rho_pure)}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Thermal state populations
    axes[0].bar(range(N), np.diag(rho), color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Energy level $n$')
    axes[0].set_ylabel('$p_n$')
    axes[0].set_title(f'Thermal State ($\\beta = {beta}$)')

    # Density matrix visualization
    im = axes[1].imshow(np.abs(rho), cmap='Blues')
    axes[1].set_title('$|\\rho_{mn}|$ (Thermal)')
    plt.colorbar(im, ax=axes[1])

    im = axes[2].imshow(np.abs(rho_pure), cmap='Reds')
    axes[2].set_title('$|\\rho_{mn}|$ (Pure State)')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig('density_matrices.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: Compactness Check
# ============================================================

def compactness_check():
    """
    Demonstrate that identity is not compact via orthonormal sequences.
    """
    print("\n" + "=" * 60)
    print("Part 4: Compactness Criterion")
    print("=" * 60)

    N = 50  # Truncated dimension

    # For a compact operator K, Ke_n -> 0 for any ONS (e_n)
    # For identity, Ie_n = e_n does NOT converge to 0

    # Standard basis
    e = [np.zeros(N) for _ in range(N)]
    for n in range(N):
        e[n][n] = 1

    # Identity operator
    I = np.eye(N)

    # Compact operator: K with eigenvalues 1/n
    K = np.diag([1/(n+1) for n in range(N)])

    # Check ||operator @ e_n||
    I_norms = [np.linalg.norm(I @ e[n]) for n in range(N)]
    K_norms = [np.linalg.norm(K @ e[n]) for n in range(N)]

    print("||Ae_n|| for orthonormal basis {e_n}:")
    print(f"  Identity: {I_norms[:10]} (constant = 1, NOT compact)")
    print(f"  K=diag(1/n): {[f'{x:.3f}' for x in K_norms[:10]]} (→ 0, compact)")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(N), I_norms, 'b-o', label='Identity $I$', markersize=4)
    ax.plot(range(N), K_norms, 'r-s', label='$K = \\text{diag}(1/n)$', markersize=4)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$||Ae_n||$')
    ax.set_title('Compactness Criterion: $||Ke_n|| \\to 0$ for Compact $K$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('compactness_check.png', dpi=150)
    plt.show()

# ============================================================
# Part 5: Schatten p-Norms
# ============================================================

def schatten_norms():
    """
    Compare different Schatten p-norms.
    """
    print("\n" + "=" * 60)
    print("Part 5: Schatten p-Norms")
    print("=" * 60)

    np.random.seed(123)

    # Create operators with different singular value decays
    N = 50

    # Fast decay: s_n = 1/n^2
    s_fast = 1 / (np.arange(1, N+1)**2)

    # Slow decay: s_n = 1/n
    s_slow = 1 / np.arange(1, N+1)

    # Constant: s_n = 1 (like identity truncated)
    s_const = np.ones(N)

    operators = [
        (s_fast, "Fast decay: $s_n = 1/n^2$"),
        (s_slow, "Slow decay: $s_n = 1/n$"),
        (s_const, "Constant: $s_n = 1$")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot singular values
    for s, label in operators:
        axes[0].semilogy(s, '-o', label=label, markersize=3)

    axes[0].set_xlabel('Index $n$')
    axes[0].set_ylabel('$s_n$')
    axes[0].set_title('Singular Value Decay')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Compute Schatten norms
    p_values = [1, 2, 4, np.inf]

    print("\nSchatten p-norms:")
    print(f"{'Operator':<30} | " + " | ".join([f'p={p}' for p in p_values]))
    print("-" * 70)

    for s, label in operators:
        norms = []
        for p in p_values:
            if p == np.inf:
                norm_p = np.max(s)  # Operator norm
            else:
                norm_p = np.sum(s**p)**(1/p)
            norms.append(norm_p)

        print(f"{label:<30} | " + " | ".join([f'{n:.4f}' for n in norms]))

        # Note: trace class (p=1), HS (p=2)

    # Bar chart of norms
    x = np.arange(len(operators))
    width = 0.2

    for i, p in enumerate(p_values):
        norms = []
        for s, _ in operators:
            if p == np.inf:
                norms.append(np.max(s))
            else:
                norms.append(np.sum(s**p)**(1/p))

        axes[1].bar(x + i*width, norms, width, label=f'$p={p}$' if p != np.inf else '$p=\\infty$')

    axes[1].set_xlabel('Operator')
    axes[1].set_ylabel('$||\\cdot||_p$')
    axes[1].set_title('Schatten $p$-Norms')
    axes[1].set_xticks(x + 1.5*width)
    axes[1].set_xticklabels(['$1/n^2$', '$1/n$', 'Constant'])
    axes[1].legend()
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('schatten_norms.png', dpi=150)
    plt.show()

    print("\nInterpretation:")
    print("  - Trace class (p=1): Need sum |s_n| < ∞ (fast decay required)")
    print("  - Hilbert-Schmidt (p=2): Need sum |s_n|² < ∞ (slower decay OK)")
    print("  - Compact (p=∞): Just need s_n → 0 (any decay to 0)")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 244: Compact Operators - Computational Lab")
    print("=" * 60)

    svd_demonstration()
    integral_operator_hs()
    trace_class_demo()
    compactness_check()
    schatten_norms()

    print("\n" + "=" * 60)
    print("Lab complete! Key takeaways:")
    print("  1. Compact operators = limits of finite-rank operators")
    print("  2. SVD gives K = Σ s_n |g_n⟩⟨f_n|")
    print("  3. ||K|| = s_1, ||K||_HS = √(Σs_n²), ||K||_1 = Σs_n")
    print("  4. Trace class ⊂ Hilbert-Schmidt ⊂ Compact")
    print("  5. Density matrices are trace class with Tr(ρ) = 1")
    print("=" * 60)
```

---

## 9. Summary

### Key Definitions

| Concept | Definition | Condition |
|---------|------------|-----------|
| **Compact** | Maps bounded sets to relatively compact | $Ke_n \to 0$ for ONS |
| **Finite-rank** | $\dim(\text{ran}(F)) < \infty$ | Trivially compact |
| **Hilbert-Schmidt** | $\sum_n \|Ae_n\|^2 < \infty$ | $\sum s_n^2 < \infty$ |
| **Trace class** | $\text{Tr}(|A|) < \infty$ | $\sum s_n < \infty$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{SVD:} && K = \sum_n s_n |g_n\rangle\langle f_n| \\
&\text{Operator norm:} && \|K\| = s_1 \\
&\text{Hilbert-Schmidt:} && \|K\|_{\text{HS}}^2 = \sum_n s_n^2 = \text{Tr}(K^\dagger K) \\
&\text{Trace norm:} && \|K\|_1 = \sum_n s_n = \text{Tr}(|K|) \\
&\text{Trace:} && \text{Tr}(A) = \sum_n \langle e_n, Ae_n\rangle
\end{aligned}}$$

### Hierarchy

$$\text{Finite-rank} \subset \text{Trace class} \subset \text{Hilbert-Schmidt} \subset \text{Compact} \subset \text{Bounded}$$

### Key Insights

1. **Compact = nearly finite-dimensional**: Limits of finite-rank operators
2. **Singular values characterize everything**: $s_n$ determine all norms
3. **Density matrices are trace class**: $\rho \geq 0$, $\text{Tr}(\rho) = 1$
4. **Hilbert-Schmidt is a Hilbert space**: Inner product $\langle A, B\rangle = \text{Tr}(A^\dagger B)$
5. **Spectral theory applies**: Compact self-adjoint operators have orthonormal eigenbasis

---

## 10. Daily Checklist

- [ ] I can define compact operators and their characterizations
- [ ] I can prove finite-rank operators are compact
- [ ] I understand the spectral properties of compact operators
- [ ] I can compute Hilbert-Schmidt norms
- [ ] I understand trace class operators and the trace
- [ ] I can work with the singular value decomposition
- [ ] I can connect to density matrices in QM
- [ ] I completed the computational lab exercises

---

## 11. Preview: Day 245

Tomorrow is our Week 35 review, where we'll:
- Synthesize all operator concepts into a coherent framework
- Work through comprehensive problem sets
- Connect the mathematical theory to quantum mechanics applications
- Prepare for Week 36's spectral theory

---

*"Compact operators are the infinite-dimensional analogues of matrices. They bring the computational and theoretical tools of linear algebra into the realm of functional analysis."* — Barry Simon
