# Day 235: Orthonormal Sets and Gram-Schmidt

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Orthonormal sets, Gram-Schmidt algorithm |
| Afternoon | 3 hours | Problems: Bessel's inequality, orthogonal projections |
| Evening | 2 hours | Computational lab: Implementing Gram-Schmidt |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** orthogonal and orthonormal sets in Hilbert spaces
2. **Apply** the Gram-Schmidt orthonormalization process
3. **Prove** Bessel's inequality
4. **Compute** Fourier coefficients and orthogonal projections
5. **Understand** the relationship between linear independence and orthonormality
6. **Connect** orthonormal sets to complete sets of quantum states

---

## 1. Core Content: Orthonormal Sets

### 1.1 Definitions

**Definition (Orthogonal Set)**: A set $$S \subseteq \mathcal{H}$$ is **orthogonal** if every pair of distinct elements is orthogonal:
$$x, y \in S, \, x \neq y \Rightarrow \langle x, y\rangle = 0$$

**Definition (Orthonormal Set)**: A set $$S \subseteq \mathcal{H}$$ is **orthonormal** if it is orthogonal and every element has norm 1:
$$\langle e_i, e_j\rangle = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

**Notation**: We typically denote orthonormal sets by $$\{e_n\}$$ or $$\{e_i\}_{i \in I}$$ where $$I$$ is an index set.

### 1.2 Key Properties

**Theorem**: An orthonormal set is linearly independent.

**Proof**: Suppose $$\sum_{i=1}^n \alpha_i e_i = 0$$ for scalars $$\alpha_i$$. Taking the inner product with $$e_j$$:
$$0 = \left\langle e_j, \sum_{i=1}^n \alpha_i e_i \right\rangle = \sum_{i=1}^n \bar{\alpha}_i \langle e_j, e_i\rangle = \sum_{i=1}^n \bar{\alpha}_i \delta_{ji} = \bar{\alpha}_j$$

So $$\alpha_j = 0$$ for all $$j$$. The set is linearly independent. $$\square$$

**Theorem (Pythagorean Theorem)**: If $$\{e_1, \ldots, e_n\}$$ is orthonormal, then:
$$\boxed{\left\|\sum_{i=1}^n \alpha_i e_i\right\|^2 = \sum_{i=1}^n |\alpha_i|^2}$$

**Proof**:
$$\left\|\sum_i \alpha_i e_i\right\|^2 = \left\langle \sum_i \alpha_i e_i, \sum_j \alpha_j e_j \right\rangle = \sum_i \sum_j \alpha_i \bar{\alpha}_j \langle e_i, e_j\rangle = \sum_i \sum_j \alpha_i \bar{\alpha}_j \delta_{ij} = \sum_i |\alpha_i|^2$$ $$\square$$

---

## 2. Fourier Coefficients and Projections

### 2.1 Fourier Coefficients

**Definition**: Given an orthonormal set $$\{e_n\}$$ and a vector $$x \in \mathcal{H}$$, the **Fourier coefficients** of $$x$$ with respect to $$\{e_n\}$$ are:
$$\boxed{c_n = \langle e_n, x\rangle}$$

(Note: With our convention, $$c_n = \langle e_n, x\rangle$$ extracts the component of $$x$$ along $$e_n$$.)

### 2.2 Orthogonal Projection

**Definition**: The **orthogonal projection** of $$x$$ onto the subspace spanned by $$\{e_1, \ldots, e_n\}$$ is:
$$\boxed{P_n x = \sum_{k=1}^n \langle e_k, x\rangle e_k = \sum_{k=1}^n c_k e_k}$$

**Theorem (Best Approximation)**: The projection $$P_n x$$ is the best approximation to $$x$$ in $$\text{span}\{e_1, \ldots, e_n\}$$:
$$\|x - P_n x\| = \min_{y \in \text{span}\{e_1,\ldots,e_n\}} \|x - y\|$$

**Proof**: Let $$y = \sum_{k=1}^n \alpha_k e_k$$ be any element of the span. Then:
$$\|x - y\|^2 = \|x - P_n x + P_n x - y\|^2$$

Note that $$x - P_n x \perp \text{span}\{e_k\}$$ (we'll verify this), so:
$$= \|x - P_n x\|^2 + \|P_n x - y\|^2 \geq \|x - P_n x\|^2$$

with equality iff $$y = P_n x$$. $$\square$$

### 2.3 Verification of Orthogonality

To verify $$x - P_n x \perp e_j$$:
$$\langle e_j, x - P_n x\rangle = \langle e_j, x\rangle - \langle e_j, P_n x\rangle = \langle e_j, x\rangle - \sum_{k=1}^n c_k \langle e_j, e_k\rangle$$
$$= c_j - c_j = 0$$ $$\checkmark$$

---

## 3. Bessel's Inequality

### 3.1 Statement and Proof

**Theorem (Bessel's Inequality)**: For any orthonormal set $$\{e_n\}_{n=1}^\infty$$ and any $$x \in \mathcal{H}$$:
$$\boxed{\sum_{n=1}^\infty |\langle e_n, x\rangle|^2 \leq \|x\|^2}$$

Equivalently: $$\sum_{n=1}^\infty |c_n|^2 \leq \|x\|^2$$

**Proof**: For any finite $$N$$:
$$0 \leq \left\|x - \sum_{n=1}^N c_n e_n\right\|^2 = \|x\|^2 - 2\text{Re}\left\langle x, \sum_n c_n e_n\right\rangle + \left\|\sum_n c_n e_n\right\|^2$$

$$= \|x\|^2 - 2\text{Re}\sum_n \bar{c}_n \langle x, e_n\rangle + \sum_n |c_n|^2$$

$$= \|x\|^2 - 2\text{Re}\sum_n |c_n|^2 + \sum_n |c_n|^2 = \|x\|^2 - \sum_n |c_n|^2$$

So $$\sum_{n=1}^N |c_n|^2 \leq \|x\|^2$$ for all $$N$$. Taking $$N \to \infty$$:
$$\sum_{n=1}^\infty |c_n|^2 \leq \|x\|^2$$ $$\square$$

### 3.2 Interpretation

Bessel's inequality says:
- The sum of squared Fourier coefficients is bounded by the total "energy" $$\|x\|^2$$
- The coefficients $$(c_n)$$ form a sequence in $$\ell^2$$
- The orthonormal set may not "capture" all of $$x$$ (equality only for complete bases)

---

## 4. The Gram-Schmidt Process

### 4.1 The Algorithm

**Goal**: Given a linearly independent set $$\{x_1, x_2, x_3, \ldots\}$$, construct an orthonormal set $$\{e_1, e_2, e_3, \ldots\}$$ with the same span.

**Gram-Schmidt Algorithm**:

$$\boxed{\begin{aligned}
&\text{Step 1:} && e_1 = \frac{x_1}{\|x_1\|} \\[10pt]
&\text{Step 2:} && u_2 = x_2 - \langle e_1, x_2\rangle e_1, \quad e_2 = \frac{u_2}{\|u_2\|} \\[10pt]
&\text{Step n:} && u_n = x_n - \sum_{k=1}^{n-1} \langle e_k, x_n\rangle e_k, \quad e_n = \frac{u_n}{\|u_n\|}
\end{aligned}}$$

**Key Property**: For each $$n$$:
$$\text{span}\{e_1, \ldots, e_n\} = \text{span}\{x_1, \ldots, x_n\}$$

### 4.2 Proof of Correctness

**Claim**: The $$\{e_n\}$$ produced by Gram-Schmidt are orthonormal.

**Proof by induction**:

*Base case*: $$e_1 = x_1/\|x_1\|$$ has $$\|e_1\| = 1$$. $$\checkmark$$

*Inductive step*: Assume $$\{e_1, \ldots, e_{n-1}\}$$ is orthonormal. We must show:
1. $$e_n \perp e_k$$ for $$k < n$$
2. $$\|e_n\| = 1$$
3. $$u_n \neq 0$$ (so the normalization is valid)

For (1): $$\langle e_k, u_n\rangle = \langle e_k, x_n\rangle - \sum_{j=1}^{n-1} \langle e_j, x_n\rangle \langle e_k, e_j\rangle = \langle e_k, x_n\rangle - \langle e_k, x_n\rangle = 0$$ $$\checkmark$$

For (2): $$e_n = u_n/\|u_n\|$$ by construction. $$\checkmark$$

For (3): If $$u_n = 0$$, then $$x_n = \sum_{k=1}^{n-1} \langle e_k, x_n\rangle e_k \in \text{span}\{e_1, \ldots, e_{n-1}\} = \text{span}\{x_1, \ldots, x_{n-1}\}$$, contradicting linear independence. $$\checkmark$$ $$\square$$

### 4.3 Matrix Formulation

In finite dimensions with $$A = [x_1 | x_2 | \cdots | x_n]$$, Gram-Schmidt produces the **QR decomposition**:
$$A = QR$$
where $$Q = [e_1 | e_2 | \cdots | e_n]$$ has orthonormal columns and $$R$$ is upper triangular.

---

## 5. Classical Examples

### 5.1 Legendre Polynomials

Apply Gram-Schmidt to $$\{1, x, x^2, x^3, \ldots\}$$ in $$L^2[-1, 1]$$:

$$\begin{aligned}
P_0(x) &= 1 \\
P_1(x) &= x \\
P_2(x) &= \frac{1}{2}(3x^2 - 1) \\
P_3(x) &= \frac{1}{2}(5x^3 - 3x)
\end{aligned}$$

These are the **Legendre polynomials** (unnormalized). They satisfy:
$$\int_{-1}^1 P_m(x) P_n(x) \, dx = \frac{2}{2n+1} \delta_{mn}$$

### 5.2 Hermite Polynomials

Apply Gram-Schmidt to $$\{1, x, x^2, \ldots\}$$ in $$L^2(\mathbb{R})$$ with weight $$w(x) = e^{-x^2}$$:

This produces the **Hermite polynomials**, which appear in the quantum harmonic oscillator.

### 5.3 Fourier Basis

The set $$\left\{\frac{1}{\sqrt{2\pi}}, \frac{\cos(nx)}{\sqrt{\pi}}, \frac{\sin(nx)}{\sqrt{\pi}}\right\}_{n=1}^\infty$$ is orthonormal in $$L^2[0, 2\pi]$$.

Alternatively, $$\left\{\frac{e^{inx}}{\sqrt{2\pi}}\right\}_{n \in \mathbb{Z}}$$ is orthonormal in $$L^2[0, 2\pi]$$.

---

## 6. Quantum Mechanics Connection

### 6.1 Complete Sets of Commuting Observables

In quantum mechanics, an orthonormal set $$\{|n\rangle\}$$ corresponds to the eigenstates of some observable. If the set is complete (spans the whole Hilbert space), these are **all possible measurement outcomes**.

**Example**: The energy eigenstates $$\{|E_n\rangle\}$$ of the harmonic oscillator.

### 6.2 Expansion in Eigenstates

Any state $$|\psi\rangle$$ can be expanded:
$$|\psi\rangle = \sum_n c_n |n\rangle, \quad c_n = \langle n|\psi\rangle$$

The Fourier coefficients $$c_n$$ are **probability amplitudes**:
$$P(n) = |c_n|^2 = |\langle n|\psi\rangle|^2$$

### 6.3 Bessel's Inequality as Probability Bound

Bessel's inequality becomes:
$$\sum_n |c_n|^2 = \sum_n P(n) \leq \|\psi\|^2 = 1$$

This is consistent with probability theory: the sum of probabilities is at most 1 (equality for complete bases).

### 6.4 Gram-Schmidt in Quantum Mechanics

Gram-Schmidt is used to:
1. Construct orthonormal bases from linearly independent states
2. Build orthonormal states from non-orthogonal trial wave functions
3. Implement the Löwdin orthogonalization in molecular orbital theory

---

## 7. Worked Examples

### Example 1: Gram-Schmidt in $$\mathbb{R}^3$$

**Problem**: Apply Gram-Schmidt to $$x_1 = (1, 1, 0)$$, $$x_2 = (1, 0, 1)$$, $$x_3 = (0, 1, 1)$$.

**Solution**:

**Step 1**:
$$\|x_1\| = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2}$$
$$e_1 = \frac{x_1}{\|x_1\|} = \frac{1}{\sqrt{2}}(1, 1, 0)$$

**Step 2**:
$$\langle e_1, x_2\rangle = \frac{1}{\sqrt{2}}(1 \cdot 1 + 1 \cdot 0 + 0 \cdot 1) = \frac{1}{\sqrt{2}}$$

$$u_2 = x_2 - \langle e_1, x_2\rangle e_1 = (1, 0, 1) - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1, 1, 0) = (1, 0, 1) - \frac{1}{2}(1, 1, 0)$$
$$= \left(\frac{1}{2}, -\frac{1}{2}, 1\right)$$

$$\|u_2\| = \sqrt{\frac{1}{4} + \frac{1}{4} + 1} = \sqrt{\frac{3}{2}} = \frac{\sqrt{6}}{2}$$

$$e_2 = \frac{u_2}{\|u_2\|} = \frac{2}{\sqrt{6}}\left(\frac{1}{2}, -\frac{1}{2}, 1\right) = \frac{1}{\sqrt{6}}(1, -1, 2)$$

**Step 3**:
$$\langle e_1, x_3\rangle = \frac{1}{\sqrt{2}}(0 + 1 + 0) = \frac{1}{\sqrt{2}}$$
$$\langle e_2, x_3\rangle = \frac{1}{\sqrt{6}}(0 - 1 + 2) = \frac{1}{\sqrt{6}}$$

$$u_3 = x_3 - \langle e_1, x_3\rangle e_1 - \langle e_2, x_3\rangle e_2$$
$$= (0, 1, 1) - \frac{1}{2}(1, 1, 0) - \frac{1}{6}(1, -1, 2)$$
$$= (0, 1, 1) - \left(\frac{1}{2}, \frac{1}{2}, 0\right) - \left(\frac{1}{6}, -\frac{1}{6}, \frac{1}{3}\right)$$
$$= \left(-\frac{2}{3}, \frac{2}{3}, \frac{2}{3}\right)$$

$$\|u_3\| = \frac{2}{3}\sqrt{3} = \frac{2\sqrt{3}}{3}$$

$$e_3 = \frac{u_3}{\|u_3\|} = \frac{1}{\sqrt{3}}(-1, 1, 1)$$

**Result**:
$$e_1 = \frac{1}{\sqrt{2}}(1, 1, 0), \quad e_2 = \frac{1}{\sqrt{6}}(1, -1, 2), \quad e_3 = \frac{1}{\sqrt{3}}(-1, 1, 1)$$ $$\square$$

---

### Example 2: Gram-Schmidt for Polynomials

**Problem**: Apply Gram-Schmidt to $$\{1, x, x^2\}$$ in $$L^2[-1, 1]$$ to find the first three (unnormalized) Legendre polynomials.

**Solution**:

**Step 1**: Take $$p_0(x) = 1$$.

$$\|1\|^2 = \int_{-1}^1 1 \, dx = 2$$

**Step 2**:
$$\langle 1, x\rangle = \int_{-1}^1 x \, dx = 0$$ (odd function on symmetric interval)

So $$u_1 = x - 0 \cdot 1 = x$$.

$$\|x\|^2 = \int_{-1}^1 x^2 \, dx = \frac{2}{3}$$

Take $$p_1(x) = x$$ (unnormalized).

**Step 3**:
$$\langle 1, x^2\rangle = \int_{-1}^1 x^2 \, dx = \frac{2}{3}$$
$$\langle x, x^2\rangle = \int_{-1}^1 x^3 \, dx = 0$$

$$u_2 = x^2 - \frac{\langle 1, x^2\rangle}{\|1\|^2} \cdot 1 - \frac{\langle x, x^2\rangle}{\|x\|^2} \cdot x = x^2 - \frac{2/3}{2} = x^2 - \frac{1}{3}$$

This is proportional to $$P_2(x) = \frac{1}{2}(3x^2 - 1) = \frac{3}{2}(x^2 - \frac{1}{3})$$.

**Result**:
$$P_0(x) = 1, \quad P_1(x) = x, \quad P_2(x) = \frac{1}{2}(3x^2 - 1)$$ $$\square$$

---

### Example 3: Bessel's Inequality Application

**Problem**: For $$f(x) = x$$ on $$[0, 2\pi]$$ and the orthonormal set $$e_n(x) = \frac{e^{inx}}{\sqrt{2\pi}}$$, compute $$\sum_{n=-N}^N |c_n|^2$$ and verify Bessel's inequality.

**Solution**:

$$c_n = \langle e_n, f\rangle = \frac{1}{\sqrt{2\pi}}\int_0^{2\pi} e^{-inx} \cdot x \, dx$$

For $$n \neq 0$$, integrate by parts:
$$\int_0^{2\pi} x e^{-inx} \, dx = \left[-\frac{x e^{-inx}}{in}\right]_0^{2\pi} + \frac{1}{in}\int_0^{2\pi} e^{-inx} \, dx$$
$$= -\frac{2\pi e^{-2\pi in}}{in} + \frac{1}{in}\left[-\frac{e^{-inx}}{in}\right]_0^{2\pi} = -\frac{2\pi}{in} + 0 = \frac{2\pi i}{n}$$

So $$c_n = \frac{1}{\sqrt{2\pi}} \cdot \frac{2\pi i}{n} = \sqrt{2\pi} \cdot \frac{i}{n}$$ for $$n \neq 0$$.

For $$n = 0$$: $$c_0 = \frac{1}{\sqrt{2\pi}}\int_0^{2\pi} x \, dx = \frac{2\pi^2}{\sqrt{2\pi}} = \pi\sqrt{2\pi}$$.

$$|c_n|^2 = 2\pi \cdot \frac{1}{n^2}$$ for $$n \neq 0$$, and $$|c_0|^2 = 2\pi^3$$.

$$\sum_{n=-N, n\neq 0}^N |c_n|^2 = 2 \cdot 2\pi \sum_{n=1}^N \frac{1}{n^2} = 4\pi \sum_{n=1}^N \frac{1}{n^2}$$

As $$N \to \infty$$: $$4\pi \cdot \frac{\pi^2}{6} = \frac{2\pi^3}{3}$$.

Including $$c_0$$: Total $$= 2\pi^3 + \frac{2\pi^3}{3} = \frac{8\pi^3}{3}$$.

**Verify with** $$\|f\|^2$$:
$$\|f\|^2 = \int_0^{2\pi} x^2 \, dx = \frac{(2\pi)^3}{3} = \frac{8\pi^3}{3}$$

We have equality! This is because $$\{e_n\}_{n \in \mathbb{Z}}$$ is complete in $$L^2[0, 2\pi]$$. $$\square$$

---

## 8. Practice Problems

### Level 1: Direct Application

1. Apply Gram-Schmidt to $$\{(1, 0, 1), (1, 1, 0), (0, 1, 1)\}$$ in $$\mathbb{R}^3$$.

2. Verify that $$\{e^{inx}/\sqrt{2\pi}\}_{n \in \mathbb{Z}}$$ is orthonormal in $$L^2[0, 2\pi]$$.

3. Compute the Fourier coefficients of $$f(x) = \cos(x)$$ with respect to $$\{e^{inx}/\sqrt{2\pi}\}$$.

### Level 2: Intermediate

4. Apply Gram-Schmidt to $$\{1, x, x^2, x^3\}$$ in $$L^2[0, 1]$$ to find the first four orthonormal polynomials.

5. Prove that if $$\{e_n\}$$ is orthonormal and $$x = \sum_n c_n e_n$$ (convergent in norm), then $$c_n = \langle e_n, x\rangle$$.

6. **Quantum Problem**: A particle in a box has energy eigenstates $$\psi_n(x) = \sqrt{2/L}\sin(n\pi x/L)$$ for $$n = 1, 2, 3, \ldots$$. Verify these are orthonormal in $$L^2[0, L]$$.

### Level 3: Challenging

7. Prove that the Gram-Schmidt process is "stable" in the sense that if $$\{x_n\}$$ is "almost" linearly dependent, the algorithm will detect this ($$u_n$$ will be "small").

8. Show that the Fourier expansion $$f = \sum_n c_n e_n$$ converges in $$L^2$$ if and only if $$\sum_n |c_n|^2 < \infty$$. (This is Bessel's inequality becoming equality for complete bases.)

9. **(Löwdin Orthogonalization)**: Given an overlap matrix $$S_{ij} = \langle x_i, x_j\rangle$$, show that $$e_i = \sum_j (S^{-1/2})_{ij} x_j$$ defines an orthonormal set with the same span.

---

## 9. Computational Lab: Implementing Gram-Schmidt

```python
"""
Day 235 Computational Lab: Gram-Schmidt Orthonormalization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import legendre

# ============================================================
# Part 1: Classical Gram-Schmidt in R^n
# ============================================================

def classical_gram_schmidt(vectors):
    """
    Classical Gram-Schmidt orthonormalization.

    Parameters:
        vectors: list of numpy arrays (the x_i)

    Returns:
        orthonormal: list of orthonormal vectors (the e_i)
    """
    orthonormal = []

    for x in vectors:
        # Subtract projections onto previous vectors
        u = x.copy().astype(float)
        for e in orthonormal:
            u = u - np.vdot(e, x) * e  # u = x - <e, x>e

        # Normalize
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-10:
            print("Warning: Nearly dependent vector detected!")
            continue
        e = u / norm_u
        orthonormal.append(e)

    return orthonormal

def test_gram_schmidt_Rn():
    """Test Gram-Schmidt on vectors in R^3."""
    print("Gram-Schmidt in R³")
    print("=" * 40)

    x1 = np.array([1, 1, 0])
    x2 = np.array([1, 0, 1])
    x3 = np.array([0, 1, 1])

    vectors = [x1, x2, x3]
    orthonormal = classical_gram_schmidt(vectors)

    print("\nOriginal vectors:")
    for i, x in enumerate(vectors):
        print(f"  x_{i+1} = {x}")

    print("\nOrthonormal vectors:")
    for i, e in enumerate(orthonormal):
        print(f"  e_{i+1} = [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}]")

    print("\nVerification (inner products):")
    n = len(orthonormal)
    for i in range(n):
        for j in range(n):
            ip = np.vdot(orthonormal[i], orthonormal[j])
            print(f"  <e_{i+1}, e_{j+1}> = {ip:.6f}")

# ============================================================
# Part 2: Modified Gram-Schmidt (More Stable)
# ============================================================

def modified_gram_schmidt(vectors):
    """
    Modified Gram-Schmidt - more numerically stable.

    Instead of subtracting all projections at once,
    we subtract them one at a time, updating u as we go.
    """
    n = len(vectors)
    Q = np.array(vectors, dtype=float).T  # columns are vectors

    for i in range(n):
        # Normalize column i
        Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])

        # Orthogonalize remaining columns against column i
        for j in range(i+1, n):
            Q[:, j] = Q[:, j] - np.dot(Q[:, i], Q[:, j]) * Q[:, i]

    return [Q[:, i] for i in range(n)]

def compare_stability():
    """Compare stability of classical vs modified Gram-Schmidt."""
    print("\nStability Comparison")
    print("=" * 40)

    # Nearly dependent vectors (Hilbert matrix-like)
    n = 10
    vectors = [np.array([1/(i+j+1) for j in range(n)]) for i in range(n)]

    # Classical
    try:
        Q_classical = classical_gram_schmidt(vectors)
        # Check orthonormality
        Q_mat = np.column_stack(Q_classical)
        error_classical = np.linalg.norm(Q_mat.T @ Q_mat - np.eye(n))
    except:
        error_classical = float('inf')

    # Modified
    Q_modified = modified_gram_schmidt(vectors)
    Q_mat = np.column_stack(Q_modified)
    error_modified = np.linalg.norm(Q_mat.T @ Q_mat - np.eye(n))

    print(f"Classical Gram-Schmidt orthonormality error: {error_classical:.6e}")
    print(f"Modified Gram-Schmidt orthonormality error: {error_modified:.6e}")

# ============================================================
# Part 3: Gram-Schmidt for Polynomials (Legendre)
# ============================================================

def gram_schmidt_polynomials(n_polys, interval=(-1, 1)):
    """
    Apply Gram-Schmidt to {1, x, x^2, ...} in L^2[a, b].
    """
    a, b = interval
    x = np.linspace(a, b, 1000)

    def inner_product(f, g):
        """L^2 inner product."""
        return np.trapz(f * g, x)

    def normalize(f):
        """Normalize in L^2."""
        return f / np.sqrt(inner_product(f, f))

    # Monomials
    monomials = [x**k for k in range(n_polys)]

    # Gram-Schmidt
    orthonormal = []
    for mono in monomials:
        u = mono.copy()
        for e in orthonormal:
            u = u - inner_product(e, mono) * e
        orthonormal.append(normalize(u))

    return orthonormal, x

def plot_orthonormal_polynomials():
    """Plot orthonormal polynomials from Gram-Schmidt."""
    orthonormal, x = gram_schmidt_polynomials(5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot polynomials
    ax = axes[0]
    for n, p in enumerate(orthonormal):
        ax.plot(x, p, linewidth=2, label=f'$p_{n}(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p_n(x)$')
    ax.set_title('Orthonormal Polynomials on $[-1, 1]$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Verify orthonormality
    ax = axes[1]
    n_polys = len(orthonormal)
    inner_matrix = np.zeros((n_polys, n_polys))
    for i in range(n_polys):
        for j in range(n_polys):
            inner_matrix[i, j] = np.trapz(orthonormal[i] * orthonormal[j], x)

    im = ax.imshow(inner_matrix, cmap='RdBu', vmin=-0.5, vmax=1.5)
    ax.set_xlabel('$n$')
    ax.set_ylabel('$m$')
    ax.set_title('$\\langle p_m, p_n \\rangle$ (should be $\\delta_{mn}$)')
    ax.set_xticks(range(n_polys))
    ax.set_yticks(range(n_polys))
    plt.colorbar(im, ax=ax)

    # Annotate
    for i in range(n_polys):
        for j in range(n_polys):
            ax.text(j, i, f'{inner_matrix[i,j]:.2f}',
                   ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('orthonormal_polynomials.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: Fourier Coefficients and Bessel's Inequality
# ============================================================

def demonstrate_bessel():
    """Demonstrate Bessel's inequality and convergence."""
    x = np.linspace(0, 2*np.pi, 1000)

    # Target function
    def f(x):
        return x  # Simple: f(x) = x

    # Fourier basis: e_n(x) = e^{inx} / sqrt(2*pi)
    def fourier_coeff(f_vals, n):
        """Compute c_n = <e_n, f>."""
        e_n = np.exp(-1j * n * x) / np.sqrt(2 * np.pi)
        return np.trapz(f_vals * e_n, x)

    f_vals = f(x)
    f_norm_sq = np.trapz(np.abs(f_vals)**2, x)

    # Compute Fourier coefficients
    N_max = 50
    n_values = np.arange(-N_max, N_max + 1)
    coeffs = np.array([fourier_coeff(f_vals, n) for n in n_values])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot |c_n|^2
    ax = axes[0, 0]
    ax.stem(n_values, np.abs(coeffs)**2, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$|c_n|^2$')
    ax.set_title('Squared Fourier Coefficients')
    ax.grid(True, alpha=0.3)

    # Partial sums of |c_n|^2 (Bessel)
    ax = axes[0, 1]
    cumsum = np.cumsum(np.abs(coeffs)**2)
    ax.plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2)
    ax.axhline(y=f_norm_sq, color='r', linestyle='--',
               label=f'$\\|f\\|^2 = {f_norm_sq:.4f}$')
    ax.set_xlabel('Number of terms')
    ax.set_ylabel('$\\sum |c_n|^2$')
    ax.set_title("Bessel's Inequality (converging to equality)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reconstruction with partial sums
    ax = axes[1, 0]
    ax.plot(x, f_vals, 'k-', linewidth=2, label='$f(x) = x$')

    for N in [1, 5, 10, 25]:
        reconstruction = np.zeros_like(x, dtype=complex)
        for n in range(-N, N+1):
            c_n = fourier_coeff(f_vals, n)
            e_n = np.exp(1j * n * x) / np.sqrt(2 * np.pi)
            reconstruction += c_n * e_n
        ax.plot(x, np.real(reconstruction), linewidth=1.5,
               label=f'$N = {N}$', alpha=0.7)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Fourier Reconstruction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # L^2 error vs N
    ax = axes[1, 1]
    N_values = range(1, 51)
    errors = []
    for N in N_values:
        reconstruction = np.zeros_like(x, dtype=complex)
        for n in range(-N, N+1):
            c_n = fourier_coeff(f_vals, n)
            e_n = np.exp(1j * n * x) / np.sqrt(2 * np.pi)
            reconstruction += c_n * e_n
        error = np.sqrt(np.trapz(np.abs(f_vals - reconstruction)**2, x))
        errors.append(error)

    ax.semilogy(N_values, errors, 'b-o', markersize=3)
    ax.set_xlabel('$N$')
    ax.set_ylabel('$\\|f - S_N\\|_2$')
    ax.set_title('$L^2$ Error of Fourier Approximation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bessel_inequality.png', dpi=150)
    plt.show()

# ============================================================
# Part 5: Quantum States and Orthonormal Expansions
# ============================================================

def quantum_orthonormal_expansion():
    """
    Demonstrate orthonormal expansion of a quantum state.
    """
    x = np.linspace(-10, 10, 1000)

    # Basis: Harmonic oscillator eigenstates
    from scipy.special import hermite

    def psi_n(x, n):
        """Normalized harmonic oscillator eigenstate."""
        Hn = hermite(n)
        norm = 1 / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
        return norm * Hn(x) * np.exp(-x**2 / 2)

    # Target state: displaced Gaussian
    x0 = 2.0  # displacement
    sigma = 1.0
    target = (1 / (np.pi * sigma**2))**0.25 * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Compute Fourier coefficients
    n_max = 20
    coeffs = []
    for n in range(n_max):
        c_n = np.trapz(psi_n(x, n) * target, x)
        coeffs.append(c_n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot coefficients
    ax = axes[0, 0]
    ax.bar(range(n_max), np.abs(coeffs)**2, color='steelblue', alpha=0.7)
    ax.set_xlabel('$n$')
    ax.set_ylabel('$|c_n|^2$')
    ax.set_title('Probability Distribution over Energy Eigenstates')
    ax.grid(True, alpha=0.3)

    # Verify sum of probabilities (Bessel)
    total_prob = sum(np.abs(c)**2 for c in coeffs)
    print(f"\nQuantum state expansion:")
    print(f"  Sum of |c_n|^2 = {total_prob:.6f} (should approach 1)")
    print(f"  ||target||^2 = {np.trapz(np.abs(target)**2, x):.6f}")

    # Reconstruction
    ax = axes[0, 1]
    ax.plot(x, target, 'k-', linewidth=2, label='Target state')

    for N in [1, 5, 10, 20]:
        reconstruction = sum(coeffs[n] * psi_n(x, n) for n in range(N))
        ax.plot(x, np.real(reconstruction), linewidth=1.5,
               label=f'$N = {N}$', alpha=0.7)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\psi(x)$')
    ax.set_title('Reconstructing State from Eigenstates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative probability
    ax = axes[1, 0]
    cumsum = np.cumsum(np.abs(coeffs)**2)
    ax.plot(range(1, n_max+1), cumsum, 'b-o', markersize=4)
    ax.axhline(y=1, color='r', linestyle='--', label='Unity')
    ax.set_xlabel('Number of eigenstates')
    ax.set_ylabel('$\\sum_{n=0}^{N-1} |c_n|^2$')
    ax.set_title('Cumulative Probability (Bessel → Parseval)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase space representation (coefficients as complex numbers)
    ax = axes[1, 1]
    for n in range(min(10, n_max)):
        c = coeffs[n]
        ax.arrow(0, 0, np.real(c), np.imag(c), head_width=0.02,
                head_length=0.01, fc=plt.cm.viridis(n/10), ec=plt.cm.viridis(n/10))
        ax.annotate(f'$c_{n}$', (np.real(c), np.imag(c)), fontsize=8)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Re($c_n$)')
    ax.set_ylabel('Im($c_n$)')
    ax.set_title('Fourier Coefficients in Complex Plane')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantum_expansion.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 235: Gram-Schmidt Orthonormalization - Computational Lab")
    print("=" * 60)

    print("\n1. Testing Gram-Schmidt in R³...")
    test_gram_schmidt_Rn()

    print("\n2. Comparing stability of classical vs modified GS...")
    compare_stability()

    print("\n3. Plotting orthonormal polynomials...")
    plot_orthonormal_polynomials()

    print("\n4. Demonstrating Bessel's inequality...")
    demonstrate_bessel()

    print("\n5. Quantum state orthonormal expansion...")
    quantum_orthonormal_expansion()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## 10. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Orthonormal Set** | $$\langle e_i, e_j\rangle = \delta_{ij}$$ |
| **Fourier Coefficient** | $$c_n = \langle e_n, x\rangle$$ |
| **Orthogonal Projection** | $$P_n x = \sum_{k=1}^n c_k e_k$$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{Pythagorean Theorem:} && \left\|\sum_i \alpha_i e_i\right\|^2 = \sum_i |\alpha_i|^2 \\[5pt]
&\text{Bessel's Inequality:} && \sum_n |\langle e_n, x\rangle|^2 \leq \|x\|^2 \\[5pt]
&\text{Gram-Schmidt:} && u_n = x_n - \sum_{k=1}^{n-1} \langle e_k, x_n\rangle e_k, \quad e_n = \frac{u_n}{\|u_n\|}
\end{aligned}}$$

### Key Insights

1. **Orthonormal sets are linearly independent** — orthogonality implies independence
2. **Gram-Schmidt constructs orthonormal sets** — algorithmic orthonormalization
3. **Bessel's inequality bounds Fourier coefficients** — sum of squares ≤ total norm squared
4. **Projections give best approximations** — minimizes distance in the subspace
5. **Quantum states expand in orthonormal bases** — coefficients give probability amplitudes

---

## 11. Daily Checklist

- [ ] I can define orthogonal and orthonormal sets
- [ ] I can apply the Gram-Schmidt process
- [ ] I can prove Bessel's inequality
- [ ] I understand Fourier coefficients and orthogonal projections
- [ ] I can connect orthonormal sets to quantum mechanical bases
- [ ] I completed the computational lab exercises

---

## 12. Preview: Day 236

Tomorrow we study **orthonormal bases** and **Parseval's identity**. An orthonormal basis is a complete orthonormal set—one that spans the entire Hilbert space. Parseval's identity states that Bessel's inequality becomes equality for complete bases, which has profound implications for quantum mechanics: the sum of all probabilities equals 1.

---

*"The Gram-Schmidt process is the bridge between linear independence and orthonormality—transforming algebraic structure into geometric structure."*
