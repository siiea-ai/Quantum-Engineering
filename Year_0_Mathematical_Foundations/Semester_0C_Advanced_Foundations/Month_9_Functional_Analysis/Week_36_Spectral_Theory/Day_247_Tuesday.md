# Day 247: Spectral Theorem for Compact Self-Adjoint Operators

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning I** | 2 hours | Hilbert-Schmidt Theorem |
| **Morning II** | 2 hours | Proof of the Spectral Theorem |
| **Afternoon** | 2 hours | Applications and Examples |
| **Evening** | 2 hours | Computational Lab: Eigenvalue Decomposition |

## Learning Objectives

By the end of today, you will be able to:

1. **State and prove** the spectral theorem for compact self-adjoint operators
2. **Apply** the Hilbert-Schmidt theorem to characterize eigenvalue sequences
3. **Construct** the spectral decomposition $A = \sum \lambda_n |e_n\rangle\langle e_n|$
4. **Derive** trace formulas and trace-class operator properties
5. **Connect** the spectral decomposition to quantum state expansions
6. **Compute** eigenvalue decompositions numerically and verify convergence

## Core Content

### 1. Statement of the Spectral Theorem

**Theorem (Spectral Theorem for Compact Self-Adjoint Operators)**

Let $A: \mathcal{H} \to \mathcal{H}$ be a compact self-adjoint operator on a separable Hilbert space. Then:

1. $\sigma(A) \setminus \{0\}$ consists entirely of eigenvalues (point spectrum)
2. Each nonzero eigenvalue has finite multiplicity
3. If $\sigma(A)$ is infinite, the eigenvalues form a sequence $\lambda_n \to 0$
4. There exists an orthonormal basis $\{e_n\}$ of $\mathcal{H}$ consisting of eigenvectors
5. $A$ has the spectral representation:

$$\boxed{A = \sum_{n=1}^{\infty} \lambda_n |e_n\rangle\langle e_n| = \sum_{n=1}^{\infty} \lambda_n \langle e_n, \cdot \rangle e_n}$$

where the sum converges in operator norm.

**Remark on Notation**

The **Dirac notation** $|e_n\rangle\langle e_n|$ represents the **projection operator** onto the span of $e_n$:
$$(|e_n\rangle\langle e_n|)(x) = \langle e_n, x \rangle e_n$$

In matrix terms, if $e_n$ is a column vector, then $|e_n\rangle\langle e_n| = e_n e_n^\dagger$.

### 2. Key Lemmas for the Proof

**Lemma 1 (Compact Self-Adjoint Has an Eigenvalue)**

If $A$ is compact and self-adjoint with $A \neq 0$, then either $\|A\|$ or $-\|A\|$ is an eigenvalue.

*Proof:* By the characterization of operator norm for self-adjoint operators:
$$\|A\| = \sup_{\|x\|=1} |\langle Ax, x \rangle|$$

There exists a sequence $\{x_n\}$ with $\|x_n\| = 1$ such that $|\langle Ax_n, x_n \rangle| \to \|A\|$.

By passing to a subsequence, assume $\langle Ax_n, x_n \rangle \to \lambda$ where $\lambda = \pm\|A\|$.

Since $A$ is compact, $\{Ax_n\}$ has a convergent subsequence. Compute:
$$\|Ax_n - \lambda x_n\|^2 = \|Ax_n\|^2 - 2\lambda\langle Ax_n, x_n\rangle + \lambda^2\|x_n\|^2$$
$$\leq \|A\|^2 - 2\lambda\langle Ax_n, x_n\rangle + \lambda^2 \to 0$$

So $Ax_n - \lambda x_n \to 0$. By compactness, extract $Ax_{n_k} \to y$. Then $x_{n_k} \to y/\lambda$ and $Ay = \lambda y$. $\square$

**Lemma 2 (Eigenspaces are Finite-Dimensional)**

For $\lambda \neq 0$, the eigenspace $\ker(A - \lambda I)$ is finite-dimensional.

*Proof:* Suppose $\dim \ker(A - \lambda I) = \infty$. Take orthonormal $\{e_n\}_{n=1}^\infty$ in this space.

Then $Ae_n = \lambda e_n$ and $\|Ae_n - Ae_m\|^2 = |\lambda|^2 \|e_n - e_m\|^2 = 2|\lambda|^2$.

The sequence $\{Ae_n\}$ has no convergent subsequence, contradicting compactness. $\square$

**Lemma 3 (Only Accumulation Point is Zero)**

If $A$ has infinitely many distinct eigenvalues, then $0$ is their only accumulation point.

*Proof:* Suppose $\lambda_n \to \lambda \neq 0$ with $Ae_n = \lambda_n e_n$ and $\{e_n\}$ orthonormal.

Then $\|Ae_n - Ae_m\|^2 = |\lambda_n|^2 + |\lambda_m|^2 \geq |\lambda|^2 > 0$ for large $n, m$.

Again, no convergent subsequence exists, contradicting compactness. $\square$

### 3. Proof of the Spectral Theorem

**Proof of the Spectral Theorem**

**Step 1: Construct eigenvalues and eigenvectors inductively.**

By Lemma 1, $A$ has an eigenvalue $\lambda_1$ with $|\lambda_1| = \|A\|$. Choose normalized eigenvector $e_1$.

Define $\mathcal{H}_1 = (\text{span}\{e_1\})^\perp$. Since $A$ is self-adjoint:
$$x \perp e_1 \implies \langle Ax, e_1 \rangle = \langle x, Ae_1 \rangle = \lambda_1\langle x, e_1 \rangle = 0$$

So $A(\mathcal{H}_1) \subseteq \mathcal{H}_1$. The restriction $A|_{\mathcal{H}_1}$ is compact and self-adjoint.

If $A|_{\mathcal{H}_1} \neq 0$, find eigenvalue $\lambda_2$ with $|\lambda_2| = \|A|_{\mathcal{H}_1}\| \leq |\lambda_1|$.

**Step 2: Continue inductively.**

Proceeding inductively, we obtain:
- Eigenvalues $\lambda_1, \lambda_2, \ldots$ with $|\lambda_1| \geq |\lambda_2| \geq \cdots$
- Orthonormal eigenvectors $e_1, e_2, \ldots$

The process terminates (finite spectrum) or continues indefinitely. In the latter case, $|\lambda_n| \to 0$ by Lemma 3.

**Step 3: The eigenvectors span $\mathcal{H}$.**

Let $\mathcal{M} = \overline{\text{span}\{e_n\}}$ and $\mathcal{M}^\perp$ its orthogonal complement.

By construction, $A(\mathcal{M}^\perp) \subseteq \mathcal{M}^\perp$. If $\mathcal{M}^\perp \neq \{0\}$, then $A|_{\mathcal{M}^\perp}$ has an eigenvalue, but we've exhausted all nonzero eigenvalues.

Therefore $A|_{\mathcal{M}^\perp} = 0$, so $\mathcal{M}^\perp \subseteq \ker(A)$.

If $0$ is an eigenvalue, include an orthonormal basis for $\ker(A)$ in $\{e_n\}$.

**Step 4: Verify operator norm convergence.**

Define partial sums $A_N = \sum_{n=1}^N \lambda_n |e_n\rangle\langle e_n|$.

For $x \in \mathcal{H}$:
$$\|(A - A_N)x\|^2 = \left\|\sum_{n>N} \lambda_n \langle e_n, x \rangle e_n\right\|^2 = \sum_{n>N} |\lambda_n|^2 |\langle e_n, x \rangle|^2$$
$$\leq |\lambda_{N+1}|^2 \sum_{n>N} |\langle e_n, x \rangle|^2 \leq |\lambda_{N+1}|^2 \|x\|^2$$

So $\|A - A_N\| \leq |\lambda_{N+1}| \to 0$. $\square$

### 4. Consequences and Generalizations

**Corollary (Eigenvalue Ordering)**

For compact self-adjoint $A$, arrange eigenvalues as:
$$|\lambda_1| \geq |\lambda_2| \geq \cdots$$
(repeated according to multiplicity). Then:
$$|\lambda_n| = \|A|_{(\text{span}\{e_1,\ldots,e_{n-1}\})^\perp}\|$$

This is the **min-max characterization** (Courant-Fischer).

**Corollary (Rayleigh Quotient)**

$$\lambda_1 = \max_{\|x\|=1} \langle Ax, x \rangle, \quad \lambda_{\min} = \min_{\|x\|=1} \langle Ax, x \rangle$$

**Definition (Trace-Class Operators)**

$A$ is **trace-class** if $\sum_n |\lambda_n| < \infty$. The trace is:
$$\boxed{\text{tr}(A) = \sum_{n=1}^\infty \lambda_n}$$

**Definition (Hilbert-Schmidt Operators)**

$A$ is **Hilbert-Schmidt** if $\sum_n |\lambda_n|^2 < \infty$. Equivalently:
$$\|A\|_{HS}^2 = \sum_{i,j} |\langle Ae_i, e_j \rangle|^2 < \infty$$

### 5. Examples of Spectral Decomposition

**Example 1: Integral Operators**

Consider the Fredholm integral operator on $L^2[0,1]$:
$$(Kf)(x) = \int_0^1 k(x,y) f(y)\, dy$$

If $k(x,y) = k(y,x)$ (symmetric kernel) and $k \in L^2([0,1]^2)$, then $K$ is compact and self-adjoint.

The spectral theorem gives:
$$k(x,y) = \sum_{n=1}^\infty \lambda_n e_n(x) \overline{e_n(y)}$$

This is **Mercer's theorem** (under continuity assumptions).

**Example 2: Heat Kernel on $[0, \pi]$**

The operator $(Af)(x) = \int_0^\pi G(x,y) f(y)\, dy$ with Green's function:
$$G(x,y) = \begin{cases} x(π-y)/π & x \leq y \\ y(π-x)/π & x > y \end{cases}$$

has eigenfunctions $e_n(x) = \sqrt{2/\pi} \sin(nx)$ with eigenvalues $\lambda_n = 1/n^2$.

### 6. Quantum Mechanics Connection

The spectral theorem is the mathematical foundation for quantum mechanics' postulates about measurement.

**Observable Operators**

In quantum mechanics, observables (energy, position, momentum, spin) are represented by self-adjoint operators. For a compact self-adjoint observable $A$:

$$\boxed{A = \sum_n E_n |n\rangle\langle n|}$$

where:
- $E_n$ = possible measurement outcomes (eigenvalues)
- $|n\rangle$ = states with definite value $E_n$ (eigenstates)
- $|n\rangle\langle n|$ = projection onto eigenspace (measurement projector)

**Born Rule for Discrete Spectrum**

If system is in state $|\psi\rangle$ and we measure $A$:

$$\boxed{P(E_n) = |\langle n|\psi\rangle|^2}$$

**Post-Measurement State**

After measuring $E_n$, the state collapses to:
$$|\psi\rangle \mapsto \frac{|n\rangle\langle n|\psi\rangle}{\sqrt{\langle\psi|n\rangle\langle n|\psi\rangle}} = |n\rangle$$

**Expectation Value**

$$\boxed{\langle A \rangle_\psi = \langle\psi|A|\psi\rangle = \sum_n E_n |\langle n|\psi\rangle|^2}$$

**Example: Spin-1/2 System**

The $z$-component of spin is:
$$S_z = \frac{\hbar}{2}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \frac{\hbar}{2}(|{\uparrow}\rangle\langle{\uparrow}| - |{\downarrow}\rangle\langle{\downarrow}|)$$

Eigenvalues: $\pm\hbar/2$. These are the only possible measurement outcomes.

For state $|\psi\rangle = \alpha|{\uparrow}\rangle + \beta|{\downarrow}\rangle$:
- $P(+\hbar/2) = |\alpha|^2$
- $P(-\hbar/2) = |\beta|^2$

**Example: Quantum Harmonic Oscillator**

The Hamiltonian $H = \frac{p^2}{2m} + \frac{m\omega^2 x^2}{2}$ has spectral decomposition:
$$H = \sum_{n=0}^\infty \hbar\omega\left(n + \frac{1}{2}\right)|n\rangle\langle n|$$

Energy levels are quantized: $E_n = \hbar\omega(n + 1/2)$.

---

## Worked Examples

### Example 1: Spectral Decomposition of a Kernel Operator

**Problem**: Find the spectral decomposition of the integral operator on $L^2[-1,1]$:
$$(Kf)(x) = \int_{-1}^1 xy \cdot f(y)\, dy$$

**Solution**:

**Step 1**: Identify the kernel and check properties.

The kernel is $k(x,y) = xy$, which is:
- Symmetric: $k(x,y) = k(y,x)$
- Continuous and bounded
- Separable: $k(x,y) = x \cdot y$

So $K$ is compact and self-adjoint.

**Step 2**: The separable kernel implies finite rank.

Note that $Kf = x \cdot \int_{-1}^1 y f(y)\, dy = x \cdot \langle y, f \rangle_{L^2}$.

So $\text{Range}(K) = \text{span}\{x\}$, and $\text{rank}(K) = 1$.

**Step 3**: Find the eigenvalues.

For $Kf = \lambda f$ with $\lambda \neq 0$:
$$x \cdot \langle y, f \rangle = \lambda f(x)$$

This means $f(x) = cx$ for some constant $c$. Then:
$$\langle y, cy \rangle = c \int_{-1}^1 y^2\, dy = \frac{2c}{3}$$

So $x \cdot \frac{2c}{3} = \lambda \cdot cx$, giving $\lambda = \frac{2}{3}$.

**Step 4**: Find the normalized eigenfunction.

Eigenvector: $e_1(x) = cx$ where $\|cx\|^2 = c^2 \cdot \frac{2}{3} = 1$, so $c = \sqrt{3/2}$.

$$e_1(x) = \sqrt{\frac{3}{2}} x$$

**Step 5**: Write the spectral decomposition.

$$\boxed{K = \frac{2}{3}|e_1\rangle\langle e_1| = \frac{2}{3} \cdot \frac{3}{2} xy = xy}$$

where $(|e_1\rangle\langle e_1|f)(x) = \langle e_1, f\rangle e_1(x) = \frac{3}{2}x \int_{-1}^1 y f(y)\, dy$.

Verification: The kernel $k(x,y) = \frac{2}{3} \cdot e_1(x)e_1(y) = \frac{2}{3} \cdot \frac{3}{2}xy = xy$. ✓

### Example 2: Hilbert-Schmidt Norm and Trace

**Problem**: For the Volterra-like operator on $L^2[0,1]$:
$$(Af)(x) = \int_0^1 \min(x,y) f(y)\, dy$$

Show $A$ is Hilbert-Schmidt and find $\|A\|_{HS}$.

**Solution**:

**Step 1**: Verify the kernel is in $L^2([0,1]^2)$.

$$\|k\|_{L^2}^2 = \int_0^1 \int_0^1 [\min(x,y)]^2\, dx\, dy$$

Split the integral:
$$= \int_0^1 \int_0^x y^2\, dy\, dx + \int_0^1 \int_x^1 x^2\, dy\, dx$$
$$= \int_0^1 \frac{x^3}{3}\, dx + \int_0^1 x^2(1-x)\, dx = \frac{1}{12} + \frac{1}{12} = \frac{1}{6}$$

**Step 2**: Hilbert-Schmidt norm.

$$\boxed{\|A\|_{HS} = \|k\|_{L^2} = \frac{1}{\sqrt{6}}}$$

**Step 3**: Eigenvalue decomposition (sketch).

The eigenfunctions satisfy:
$$\int_0^1 \min(x,y) e(y)\, dy = \lambda e(x)$$

Differentiating twice: $e''(x) = -\frac{1}{\lambda} e(x)$

With boundary conditions $e(0) = 0$ and $e'(1) = 0$:

$$e_n(x) = \sqrt{2}\sin\left(\frac{(2n-1)\pi x}{2}\right), \quad \lambda_n = \frac{4}{(2n-1)^2\pi^2}$$

**Verification**: $\sum_n \lambda_n^2 = \frac{16}{\pi^4}\sum_{n=1}^\infty \frac{1}{(2n-1)^4} = \frac{16}{\pi^4} \cdot \frac{\pi^4}{96} = \frac{1}{6}$. ✓

### Example 3: Eigenvalues of a Finite-Dimensional Projection

**Problem**: Let $P$ be the orthogonal projection onto a $k$-dimensional subspace of $\mathcal{H}$. Find the spectral decomposition of $P$.

**Solution**:

**Step 1**: Identify eigenvalues.

$P^2 = P$ and $P = P^*$. For eigenvector $Pv = \lambda v$:
$$P^2 v = \lambda Pv = \lambda^2 v = Pv = \lambda v$$

So $\lambda^2 = \lambda$, meaning $\lambda \in \{0, 1\}$.

**Step 2**: Identify eigenspaces.

- $\lambda = 1$: $Pv = v$ means $v \in \text{Range}(P)$. Eigenspace has dimension $k$.
- $\lambda = 0$: $Pv = 0$ means $v \in \ker(P) = (\text{Range}(P))^\perp$. Eigenspace has dimension $\dim(\mathcal{H}) - k$.

**Step 3**: Write spectral decomposition.

Let $\{e_1, \ldots, e_k\}$ be an orthonormal basis for $\text{Range}(P)$.

$$\boxed{P = \sum_{i=1}^k |e_i\rangle\langle e_i| = \sum_{i=1}^k \langle e_i, \cdot \rangle e_i}$$

This shows $P$ projects onto $\text{span}\{e_1, \ldots, e_k\}$. The trace is $\text{tr}(P) = k$.

---

## Practice Problems

### Level 1: Direct Application

1. **Diagonal Operator**: On $\ell^2$, let $(Dx)_n = \frac{x_n}{n}$. Find all eigenvalues and eigenvectors. Write the spectral decomposition.

2. **Finite-Rank Operator**: For $K: L^2[0,1] \to L^2[0,1]$ with kernel $k(x,y) = \sin(\pi x)\sin(\pi y)$, find the spectral decomposition.

3. **Trace Calculation**: If $A = \sum_{n=1}^\infty \frac{1}{n^2}|e_n\rangle\langle e_n|$ for orthonormal $\{e_n\}$, compute $\text{tr}(A)$ and $\|A\|_{HS}$.

### Level 2: Intermediate

4. **Kernel Eigenvalue Problem**: Find eigenvalues and eigenfunctions of:
$$(Af)(x) = \int_0^{\pi} \sin(x)\sin(y) f(y)\, dy$$

5. **Product of Projections**: If $P$ and $Q$ are orthogonal projections and $PQ = QP$, prove $PQ$ is also a projection and find its spectral decomposition in terms of those of $P$ and $Q$.

6. **Variational Characterization**: Prove the min-max formula:
$$\lambda_k = \max_{\dim V = k} \min_{x \in V, \|x\|=1} \langle Ax, x\rangle$$

### Level 3: Challenging

7. **Weyl's Inequality**: If $A$ and $B$ are compact self-adjoint with eigenvalues $\{\alpha_n\}$ and $\{\beta_n\}$ (decreasing), prove:
$$|\alpha_n - \beta_n| \leq \|A - B\|$$

8. **Lidskii's Trace Theorem**: Prove that for trace-class $A$, $\text{tr}(A) = \sum_n \lambda_n$ where $\lambda_n$ are eigenvalues (with multiplicity).

9. **Approximation Numbers**: Define $s_n(A) = \inf\{\|A - F\| : \text{rank}(F) < n\}$. Prove $s_n(A) = |\lambda_n(A)|$ for compact self-adjoint $A$.

---

## Computational Lab: Eigenvalue Decomposition

```python
"""
Day 247 Computational Lab: Spectral Theorem for Compact Self-Adjoint
Exploring eigenvalue decomposition and its quantum mechanical interpretation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import quad, dblquad
from typing import Tuple, Callable, List

np.random.seed(42)

# =============================================================================
# Part 1: Spectral Decomposition Verification
# =============================================================================

print("="*70)
print("PART 1: SPECTRAL DECOMPOSITION VERIFICATION")
print("="*70)

def create_self_adjoint_matrix(n: int) -> np.ndarray:
    """Create a random self-adjoint (Hermitian) matrix."""
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return (A + A.conj().T) / 2

def verify_spectral_decomposition(A: np.ndarray) -> None:
    """Verify A = sum lambda_n |e_n><e_n|."""
    n = A.shape[0]

    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Reconstruct A from spectral decomposition
    A_reconstructed = np.zeros_like(A)
    for i in range(n):
        e_i = eigenvectors[:, i:i+1]  # Column vector
        A_reconstructed += eigenvalues[i] * (e_i @ e_i.conj().T)

    # Check reconstruction
    error = np.linalg.norm(A - A_reconstructed)
    print(f"\nMatrix size: {n}x{n}")
    print(f"Eigenvalues (sorted): {np.sort(eigenvalues)[:5]}...")
    print(f"Reconstruction error ||A - A_reconstructed||: {error:.2e}")
    print(f"Reconstruction successful: {error < 1e-10}")

    # Verify eigenvectors are orthonormal
    orthonormality = np.linalg.norm(eigenvectors.conj().T @ eigenvectors - np.eye(n))
    print(f"Orthonormality check ||V^H V - I||: {orthonormality:.2e}")

# Test with different sizes
for n in [5, 20, 100]:
    A = create_self_adjoint_matrix(n)
    verify_spectral_decomposition(A)

# =============================================================================
# Part 2: Compact Operator Eigenvalue Decay
# =============================================================================

print("\n" + "="*70)
print("PART 2: EIGENVALUE DECAY FOR COMPACT OPERATORS")
print("="*70)

def discretize_integral_operator(kernel_func: Callable, n: int,
                                  a: float = 0, b: float = 1) -> np.ndarray:
    """
    Discretize an integral operator (Kf)(x) = int_a^b k(x,y)f(y)dy.
    Uses midpoint quadrature.
    """
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(x[i], x[j]) * h
    return K, x

# Example kernels
def kernel_smooth(x, y):
    """Smooth kernel - fast eigenvalue decay."""
    return np.exp(-(x-y)**2)

def kernel_rough(x, y):
    """Less smooth kernel - slower decay."""
    return np.minimum(x, y)

def kernel_separable(x, y):
    """Separable kernel - finite rank."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

kernels = [
    (kernel_smooth, "Gaussian kernel\n$k(x,y)=e^{-(x-y)^2}$"),
    (kernel_rough, "Min kernel\n$k(x,y)=\\min(x,y)$"),
    (kernel_separable, "Separable\n$k(x,y)=\\sin(\\pi x)\\sin(\\pi y)$")
]

for ax, (kernel, title) in zip(axes, kernels):
    K, _ = discretize_integral_operator(kernel, 200)

    # Make symmetric (self-adjoint)
    K = (K + K.T) / 2

    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Sort descending

    ax.semilogy(range(1, 51), eigenvalues[:50], 'bo-', markersize=4)
    ax.set_xlabel('Eigenvalue index n')
    ax.set_ylabel('$|\\lambda_n|$ (log scale)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eigenvalue_decay.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: eigenvalue_decay.png")

# =============================================================================
# Part 3: Trace and Hilbert-Schmidt Norm
# =============================================================================

print("\n" + "="*70)
print("PART 3: TRACE AND HILBERT-SCHMIDT NORM")
print("="*70)

def compute_spectral_quantities(K: np.ndarray) -> dict:
    """Compute trace, HS norm, and operator norm from eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(K)

    # Trace (sum of eigenvalues)
    trace_spectral = np.sum(eigenvalues)
    trace_matrix = np.trace(K)

    # Hilbert-Schmidt norm (sqrt of sum of |lambda|^2)
    hs_norm_spectral = np.sqrt(np.sum(np.abs(eigenvalues)**2))
    hs_norm_frobenius = np.linalg.norm(K, 'fro')

    # Operator norm (largest |eigenvalue|)
    op_norm_spectral = np.max(np.abs(eigenvalues))
    op_norm_matrix = np.linalg.norm(K, 2)

    print(f"\nTrace:")
    print(f"  From eigenvalues: {trace_spectral:.6f}")
    print(f"  From matrix trace: {trace_matrix:.6f}")
    print(f"  Match: {np.isclose(trace_spectral, trace_matrix)}")

    print(f"\nHilbert-Schmidt norm:")
    print(f"  From eigenvalues: {hs_norm_spectral:.6f}")
    print(f"  Frobenius norm: {hs_norm_frobenius:.6f}")
    print(f"  Match: {np.isclose(hs_norm_spectral, hs_norm_frobenius)}")

    print(f"\nOperator norm:")
    print(f"  From eigenvalues: {op_norm_spectral:.6f}")
    print(f"  2-norm: {op_norm_matrix:.6f}")
    print(f"  Match: {np.isclose(op_norm_spectral, op_norm_matrix)}")

    return {
        'eigenvalues': eigenvalues,
        'trace': trace_spectral,
        'hs_norm': hs_norm_spectral,
        'op_norm': op_norm_spectral
    }

# Test on min kernel
K_min, _ = discretize_integral_operator(kernel_rough, 100)
K_min = (K_min + K_min.T) / 2
result = compute_spectral_quantities(K_min)

# Theoretical values for min kernel
print("\nTheoretical values for min(x,y) kernel on [0,1]:")
print(f"  Eigenvalues: lambda_n = 4/((2n-1)^2 pi^2)")
n_theory = np.arange(1, 101)
lambda_theory = 4 / ((2*n_theory - 1)**2 * np.pi**2)
print(f"  Trace = sum lambda_n = 1/6 = {1/6:.6f}")
print(f"  Numerical trace: {result['trace']:.6f}")

# =============================================================================
# Part 4: Quantum Mechanics - Spin System
# =============================================================================

print("\n" + "="*70)
print("PART 4: QUANTUM MECHANICS - SPIN-1 SYSTEM")
print("="*70)

# Spin-1 matrices (angular momentum j=1)
hbar = 1  # Natural units

Jz = hbar * np.diag([1, 0, -1]).astype(complex)
Jplus = hbar * np.sqrt(2) * np.array([[0, 1, 0],
                                       [0, 0, 1],
                                       [0, 0, 0]], dtype=complex)
Jminus = Jplus.conj().T
Jx = (Jplus + Jminus) / 2
Jy = (Jplus - Jminus) / (2j)

print("Spin-1 Angular Momentum Operators:")
print(f"\nJz = \n{Jz}")

# Spectral decomposition of Jz
eigenvalues_Jz, eigenvectors_Jz = np.linalg.eigh(Jz)
print(f"\nEigenvalues of Jz: {eigenvalues_Jz} (in units of hbar)")

# Label eigenstates
labels = ['|1,-1>', '|1,0>', '|1,+1>']
print("\nSpectral decomposition Jz = sum_m m|1,m><1,m|:")
for i, (eig, label) in enumerate(zip(eigenvalues_Jz, labels)):
    print(f"  m = {eig:.0f}: eigenstate {label}")

# Measurement simulation
print("\n--- Measurement Simulation ---")

# Initial state: superposition
psi = np.array([1, 1, 1], dtype=complex)
psi = psi / np.linalg.norm(psi)
print(f"Initial state |psi> = (|+1> + |0> + |-1>)/sqrt(3)")

# Probabilities
probs = np.abs(eigenvectors_Jz.conj().T @ psi)**2
print("\nMeasurement probabilities for Jz:")
for i, (m, p) in enumerate(zip(eigenvalues_Jz, probs)):
    print(f"  P(m={m:.0f}) = {p:.4f}")

# Expectation value
expectation = np.real(psi.conj().T @ Jz @ psi)
print(f"\nExpectation value <Jz> = {expectation:.4f}")
print(f"Sum check: sum_m m*P(m) = {np.sum(eigenvalues_Jz * probs):.4f}")

# =============================================================================
# Part 5: Visualization of Spectral Projections
# =============================================================================

print("\n" + "="*70)
print("PART 5: SPECTRAL PROJECTIONS VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Create a 2x2 self-adjoint matrix with nice eigenvalues
A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eigh(A)

print(f"Matrix A = [[2,1],[1,2]]")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Plot 1: Matrix action visualization
ax1 = axes[0, 0]
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])
ellipse = A @ circle

ax1.plot(circle[0], circle[1], 'b-', label='Unit circle')
ax1.plot(ellipse[0], ellipse[1], 'r-', linewidth=2, label='Image under A')
for i in range(2):
    ev = eigenvectors[:, i]
    ax1.arrow(0, 0, ev[0], ev[1], head_width=0.1, head_length=0.1,
              fc='green', ec='green')
    ax1.arrow(0, 0, eigenvalues[i]*ev[0], eigenvalues[i]*ev[1],
              head_width=0.1, head_length=0.1, fc='purple', ec='purple')
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_aspect('equal')
ax1.legend()
ax1.set_title('Matrix Action: Circle → Ellipse\nGreen=eigenvector, Purple=stretched')
ax1.grid(True, alpha=0.3)

# Plot 2: Spectral projections
ax2 = axes[0, 1]
P1 = np.outer(eigenvectors[:, 0], eigenvectors[:, 0])
P2 = np.outer(eigenvectors[:, 1], eigenvectors[:, 1])

# Show projection operators
ax2.text(0.1, 0.8, f'$P_1 = |e_1\\rangle\\langle e_1| = $\n{P1}', fontsize=10,
         transform=ax2.transAxes, family='monospace')
ax2.text(0.1, 0.5, f'$P_2 = |e_2\\rangle\\langle e_2| = $\n{P2}', fontsize=10,
         transform=ax2.transAxes, family='monospace')
ax2.text(0.1, 0.2, f'$A = \\lambda_1 P_1 + \\lambda_2 P_2 = $\n{eigenvalues[0]:.1f}*P1 + {eigenvalues[1]:.1f}*P2',
         fontsize=10, transform=ax2.transAxes)
ax2.text(0.1, 0.05, f'Verification: {np.allclose(A, eigenvalues[0]*P1 + eigenvalues[1]*P2)}',
         fontsize=10, transform=ax2.transAxes, color='green')
ax2.axis('off')
ax2.set_title('Spectral Decomposition')

# Plot 3: Eigenfunction visualization for integral operator
ax3 = axes[1, 0]
n_grid = 100
K, x = discretize_integral_operator(kernel_rough, n_grid)
K = (K + K.T) / 2
eigs, vecs = np.linalg.eigh(K)
idx = np.argsort(np.abs(eigs))[::-1]
eigs = eigs[idx]
vecs = vecs[:, idx]

for i in range(4):
    # Normalize for visualization
    ef = vecs[:, i] * np.sqrt(n_grid)  # Account for discretization
    ax3.plot(x, ef, label=f'$e_{i+1}$, $\\lambda_{i+1}$={eigs[i]:.4f}')

ax3.set_xlabel('x')
ax3.set_ylabel('Eigenfunction value')
ax3.set_title('Eigenfunctions of min(x,y) kernel operator')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Convergence of partial sums
ax4 = axes[1, 1]
errors = []
ns = range(1, 51)
for n_terms in ns:
    A_partial = sum(eigs[i] * np.outer(vecs[:, i], vecs[:, i])
                    for i in range(n_terms))
    error = np.linalg.norm(K - A_partial)
    errors.append(error)

ax4.semilogy(ns, errors, 'b-', linewidth=2)
ax4.set_xlabel('Number of terms in spectral sum')
ax4.set_ylabel('$||K - K_N||$ (log scale)')
ax4.set_title('Convergence of Spectral Decomposition')
ax4.grid(True, alpha=0.3)

# Also plot the theoretical bound |lambda_{N+1}|
ax4.semilogy(ns, np.abs(eigs[1:51]), 'r--', label='$|\\lambda_{N+1}|$ bound')
ax4.legend()

plt.tight_layout()
plt.savefig('spectral_projections.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: spectral_projections.png")

# =============================================================================
# Part 6: Quantum State Expansion
# =============================================================================

print("\n" + "="*70)
print("PART 6: QUANTUM STATE EXPANSION IN EIGENBASIS")
print("="*70)

# Simulate harmonic oscillator ground state in position basis
# and expand in energy eigenbasis

# Create harmonic oscillator Hamiltonian (discretized)
n = 100
dx = 0.1
x = (np.arange(n) - n//2) * dx

# Kinetic energy (finite difference)
T = np.diag(np.ones(n) * 2) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
T = T / (2 * dx**2)

# Potential energy
V = np.diag(0.5 * x**2)

# Hamiltonian
H = T + V

# Diagonalize
E, psi = np.linalg.eigh(H)

print("Harmonic Oscillator Hamiltonian:")
print(f"First 10 energy levels: {E[:10]}")
print(f"Theoretical: E_n = n + 0.5 for n = 0,1,2,...")
print(f"Ground state energy: {E[0]:.4f} (theory: 0.5)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot energy eigenfunctions
ax1 = axes[0]
for i in range(5):
    # Normalize and shift for visualization
    wave = psi[:, i] / dx**0.5  # Normalize for continuous approximation
    ax1.plot(x, wave + E[i], label=f'n={i}, E={E[i]:.2f}')
    ax1.axhline(y=E[i], color='gray', linestyle=':', alpha=0.5)

ax1.set_xlabel('x')
ax1.set_ylabel('$\\psi_n(x)$ + $E_n$')
ax1.set_title('Harmonic Oscillator Energy Eigenfunctions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Create a Gaussian initial state and expand in eigenbasis
sigma = 0.5
x0 = 1.0  # Offset from equilibrium
initial_state = np.exp(-(x - x0)**2 / (2*sigma**2))
initial_state = initial_state / np.linalg.norm(initial_state)

# Expand in energy eigenbasis
coefficients = psi.T @ initial_state

ax2 = axes[1]
ax2.bar(range(20), np.abs(coefficients[:20])**2, color='blue', alpha=0.7)
ax2.set_xlabel('Energy level n')
ax2.set_ylabel('$|c_n|^2$')
ax2.set_title(f'Expansion of Gaussian (centered at x={x0})\n'
              f'in energy eigenbasis: $|\\psi\\rangle = \\sum_n c_n |n\\rangle$')
ax2.grid(True, alpha=0.3)

# Add text with key information
ax2.text(0.6, 0.9, f'$\\sum_n |c_n|^2$ = {np.sum(np.abs(coefficients)**2):.6f}',
         transform=ax2.transAxes, fontsize=11)
ax2.text(0.6, 0.8, f'$\\langle H \\rangle = \\sum_n E_n |c_n|^2$ = {np.sum(E * np.abs(coefficients)**2):.4f}',
         transform=ax2.transAxes, fontsize=11)

plt.tight_layout()
plt.savefig('quantum_state_expansion.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: quantum_state_expansion.png")

print("\n" + "="*70)
print("LAB COMPLETE")
print("="*70)
print("""
Key takeaways:
1. Spectral decomposition A = sum lambda_n |e_n><e_n| is exact for self-adjoint
2. Compact operators have eigenvalues that decay to zero
3. Trace = sum of eigenvalues, HS norm = sqrt(sum |lambda|^2)
4. In QM, spectral decomposition gives measurement outcomes and probabilities
5. Any state can be expanded in the eigenbasis of an observable
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Spectral decomposition | $A = \sum_n \lambda_n \|e_n\rangle\langle e_n\|$ |
| Orthogonality | $\langle e_n, e_m \rangle = \delta_{nm}$ |
| Completeness | $\sum_n \|e_n\rangle\langle e_n\| = I$ |
| Trace | $\text{tr}(A) = \sum_n \lambda_n$ |
| Hilbert-Schmidt norm | $\|A\|_{HS}^2 = \sum_n |\lambda_n|^2$ |
| Operator norm | $\|A\| = \max_n |\lambda_n|$ |
| Expectation value | $\langle A \rangle_\psi = \sum_n \lambda_n |\langle e_n|\psi\rangle|^2$ |

### Main Takeaways

1. **Compact self-adjoint operators diagonalize**: The spectral theorem provides an orthonormal basis of eigenvectors.

2. **Eigenvalues decay to zero**: For infinite-dimensional compact operators, the only accumulation point is zero.

3. **Convergence is in operator norm**: $\|A - A_N\| \to 0$ where $A_N$ is the $N$-term partial sum.

4. **Mercer's theorem for kernels**: Symmetric $L^2$ kernels have eigenfunction expansions.

5. **Quantum mechanics foundation**: The spectral theorem justifies the measurement postulates of quantum mechanics.

6. **Trace-class and Hilbert-Schmidt**: These operator classes are characterized by eigenvalue summability.

---

## Daily Checklist

- [ ] I can state the spectral theorem for compact self-adjoint operators
- [ ] I understand why compact operators have only point spectrum (except possibly 0)
- [ ] I can prove that eigenvalues of compact operators accumulate only at 0
- [ ] I can write the spectral decomposition in Dirac notation
- [ ] I can compute traces and Hilbert-Schmidt norms from eigenvalues
- [ ] I understand how the spectral theorem connects to quantum measurement
- [ ] I can expand quantum states in an energy eigenbasis
- [ ] I completed the computational lab

---

## Preview: Day 248

Tomorrow we extend the spectral theorem to **all bounded self-adjoint operators**, not just compact ones. This requires introducing **spectral measures** and **projection-valued measures**:
$$A = \int_{\sigma(A)} \lambda \, dE_\lambda$$

This generalization handles continuous spectrum and is essential for position and momentum operators in quantum mechanics.

---

*"The spectral theorem is the key theorem in the mathematics of quantum mechanics. It allows us to make sense of operators that don't have eigenvectors in the usual sense."*
— Barry Simon
