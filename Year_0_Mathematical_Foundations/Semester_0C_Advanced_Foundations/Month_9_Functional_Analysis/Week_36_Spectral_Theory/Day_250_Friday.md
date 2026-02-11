# Day 250: Unbounded Operators

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning I** | 2 hours | Unbounded Operators and Domains |
| **Morning II** | 2 hours | Closed and Closable Operators |
| **Afternoon** | 2 hours | Self-Adjointness Criteria |
| **Evening** | 2 hours | Computational Lab: Position and Momentum |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** unbounded operators with proper specification of domains
2. **Distinguish** between closed, closable, and non-closable operators
3. **Differentiate** symmetric, essentially self-adjoint, and self-adjoint operators
4. **Apply** von Neumann's criterion and other self-adjointness tests
5. **Analyze** the position and momentum operators in quantum mechanics
6. **Understand** the spectral theorem for unbounded self-adjoint operators

## Core Content

### 1. Why Unbounded Operators?

The most important operators in quantum mechanics are **unbounded**:

| Operator | Definition | Domain Issue |
|----------|------------|--------------|
| Position $\hat{x}$ | $(\hat{x}\psi)(x) = x\psi(x)$ | Not all $\psi \in L^2$ satisfy $x\psi \in L^2$ |
| Momentum $\hat{p}$ | $(\hat{p}\psi)(x) = -i\hbar\frac{d\psi}{dx}$ | Needs differentiable $\psi$ |
| Hamiltonian $H$ | $H = -\frac{\hbar^2}{2m}\nabla^2 + V$ | Needs twice-differentiable $\psi$ |

**Theorem (Hellinger-Toeplitz)**

If $A$ is a symmetric operator defined on all of $\mathcal{H}$, then $A$ is bounded.

*Contrapositive*: Unbounded operators cannot be defined everywhere—they require **restricted domains**.

### 2. Unbounded Operators: Basic Definitions

**Definition (Unbounded Operator)**

An **unbounded operator** on Hilbert space $\mathcal{H}$ is a linear map:
$$A: D(A) \to \mathcal{H}$$
where $D(A) \subseteq \mathcal{H}$ is a linear subspace called the **domain** of $A$.

We typically require $D(A)$ to be **dense**: $\overline{D(A)} = \mathcal{H}$.

**Convention**: When we write "operator," we mean the pair $(A, D(A))$.

**Definition (Extension and Restriction)**

- $B$ **extends** $A$ (written $A \subseteq B$) if $D(A) \subseteq D(B)$ and $Bx = Ax$ for all $x \in D(A)$.
- $A$ **restricts** to $B$ if $B \subseteq A$.

**Definition (Graph of an Operator)**

The **graph** of $A$ is:
$$\Gamma(A) = \{(x, Ax) : x \in D(A)\} \subseteq \mathcal{H} \times \mathcal{H}$$

This is a linear subspace of the Hilbert space $\mathcal{H} \times \mathcal{H}$ with inner product:
$$\langle (x_1, y_1), (x_2, y_2) \rangle = \langle x_1, x_2 \rangle + \langle y_1, y_2 \rangle$$

### 3. Closed and Closable Operators

**Definition (Closed Operator)**

$A$ is **closed** if its graph $\Gamma(A)$ is closed in $\mathcal{H} \times \mathcal{H}$.

Equivalently: If $x_n \in D(A)$, $x_n \to x$, and $Ax_n \to y$, then $x \in D(A)$ and $Ax = y$.

**Definition (Closable Operator)**

$A$ is **closable** if the closure of its graph is itself a graph.

Equivalently: If $x_n \in D(A)$, $x_n \to 0$, and $Ax_n \to y$, then $y = 0$.

**Definition (Closure)**

If $A$ is closable, its **closure** $\overline{A}$ is the operator with graph $\overline{\Gamma(A)}$.

**Proposition (Closed Operator Criteria)**

1. Bounded operators on closed domains are closed.
2. $A$ is closed iff $D(A)$ is complete under the **graph norm**:
$$\|x\|_A = \sqrt{\|x\|^2 + \|Ax\|^2}$$

**Example (Non-Closable Operator)**

On $L^2[0,1]$, define $D(A) = C[0,1]$ and $(Af)(x) = f(0)$ (constant function).

Take $f_n(x) = (1-nx)^+$. Then $f_n \to 0$ in $L^2$ but $Af_n = f_n(0) = 1$.

So $f_n \to 0$ and $Af_n \to 1 \neq 0$. Not closable.

### 4. Adjoint of Unbounded Operators

**Definition (Adjoint)**

For densely defined $A$, the **adjoint** $A^*$ has:
- **Domain**: $D(A^*) = \{y \in \mathcal{H} : x \mapsto \langle Ax, y \rangle \text{ is bounded on } D(A)\}$
- **Action**: For $y \in D(A^*)$, $A^*y$ is the unique element such that:
$$\boxed{\langle Ax, y \rangle = \langle x, A^*y \rangle \quad \forall x \in D(A)}$$

**Proposition (Properties of Adjoint)**

1. $A^*$ is always closed.
2. $A$ is closable iff $D(A^*)$ is dense.
3. If $A$ is closable, then $\overline{A} = A^{**}$.
4. $A \subseteq B \implies B^* \subseteq A^*$.

### 5. Symmetric and Self-Adjoint Operators

**Definition (Symmetric Operator)**

$A$ is **symmetric** (or Hermitian) if:
$$\langle Ax, y \rangle = \langle x, Ay \rangle \quad \forall x, y \in D(A)$$

Equivalently: $A \subseteq A^*$.

**Definition (Self-Adjoint Operator)**

$A$ is **self-adjoint** if $A = A^*$, meaning:
1. $A$ is symmetric
2. $D(A) = D(A^*)$

**Definition (Essentially Self-Adjoint)**

$A$ is **essentially self-adjoint** if $\overline{A}$ is self-adjoint.

**The Crucial Distinction**:

| Type | Condition | Physical Status |
|------|-----------|-----------------|
| Symmetric | $A \subseteq A^*$ | Not enough for QM |
| Essentially self-adjoint | $\overline{A} = \overline{A}^*$ | Acceptable (unique extension) |
| Self-adjoint | $A = A^*$ | Required for observables |

**Warning**: A symmetric operator may have many self-adjoint extensions, one, or none!

### 6. Von Neumann's Criterion

**Theorem (Von Neumann's First Criterion)**

A symmetric operator $A$ is essentially self-adjoint iff:
$$\boxed{\ker(A^* - iI) = \ker(A^* + iI) = \{0\}}$$

**Theorem (Von Neumann's Second Criterion / Deficiency Indices)**

Define the **deficiency subspaces**:
$$\mathcal{N}_+ = \ker(A^* - iI), \quad \mathcal{N}_- = \ker(A^* + iI)$$

and **deficiency indices**:
$$n_+ = \dim \mathcal{N}_+, \quad n_- = \dim \mathcal{N}_-$$

Then:
- $A$ is essentially self-adjoint iff $n_+ = n_- = 0$
- $A$ has self-adjoint extensions iff $n_+ = n_-$
- If $n_+ = n_- = n > 0$, there are infinitely many self-adjoint extensions, parametrized by $U(n)$

### 7. Other Self-Adjointness Criteria

**Theorem (Kato-Rellich Theorem)**

If $A$ is self-adjoint and $B$ is symmetric with $D(A) \subseteq D(B)$ and:
$$\|Bx\| \leq a\|Ax\| + b\|x\| \quad \forall x \in D(A)$$
for some $a < 1$ and $b \geq 0$, then $A + B$ is self-adjoint on $D(A)$.

**Theorem (Nelson's Criterion)**

If $A$ is symmetric and there exists a dense set $\mathcal{D} \subseteq D(A^\infty)$ such that:
1. $A(\mathcal{D}) \subseteq \mathcal{D}$
2. $A|_\mathcal{D}$ is essentially self-adjoint

Then $A$ is essentially self-adjoint.

**Theorem (KLMN Theorem)**

$A$ is essentially self-adjoint on $D(A)$ iff:
$$\text{Range}(A + iI) \text{ and } \text{Range}(A - iI) \text{ are dense}$$

### 8. The Spectral Theorem for Unbounded Operators

**Theorem (Spectral Theorem - Unbounded Case)**

If $A$ is self-adjoint (possibly unbounded), there exists a unique projection-valued measure $E$ on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$ such that:

$$\boxed{A = \int_{-\infty}^{\infty} \lambda \, dE_\lambda}$$

with domain:
$$D(A) = \left\{x \in \mathcal{H} : \int_{-\infty}^{\infty} \lambda^2 \, d\|E_\lambda x\|^2 < \infty\right\}$$

The spectrum $\sigma(A) = \text{supp}(E)$ may be unbounded.

### 9. Quantum Mechanics: Position and Momentum

**Position Operator**

On $\mathcal{H} = L^2(\mathbb{R})$:
$$(\hat{x}\psi)(x) = x\psi(x)$$

**Domain**: $D(\hat{x}) = \{\psi \in L^2(\mathbb{R}) : x\psi(x) \in L^2(\mathbb{R})\}$

**Properties**:
- Self-adjoint (not just symmetric)
- $\sigma(\hat{x}) = \mathbb{R}$ (continuous spectrum)
- Spectral family: $(E_a\psi)(x) = \chi_{(-\infty, a]}(x)\psi(x)$

**Momentum Operator**

On $\mathcal{H} = L^2(\mathbb{R})$:
$$(\hat{p}\psi)(x) = -i\hbar\frac{d\psi}{dx}$$

**Domain**: $D(\hat{p}) = H^1(\mathbb{R}) = \{\psi \in L^2 : \psi' \in L^2\}$ (Sobolev space)

More precisely: $\psi$ absolutely continuous with $\psi' \in L^2$.

**Properties**:
- Self-adjoint
- $\sigma(\hat{p}) = \mathbb{R}$ (continuous spectrum)
- In momentum space: $(\hat{p}\tilde{\psi})(p) = p\tilde{\psi}(p)$ (multiplication!)

**The Canonical Commutation Relation**

$$\boxed{[\hat{x}, \hat{p}] = i\hbar I}$$

This only holds on a suitable domain (e.g., Schwartz space $\mathcal{S}(\mathbb{R})$).

**Important**: This relation implies both $\hat{x}$ and $\hat{p}$ must be unbounded (no bounded operators satisfy this).

---

## Worked Examples

### Example 1: Self-Adjointness of the Momentum Operator

**Problem**: Show that $\hat{p} = -i\frac{d}{dx}$ on $L^2[0,1]$ with domain $D(\hat{p}) = \{\psi \in H^1[0,1] : \psi(0) = \psi(1)\}$ (periodic boundary conditions) is self-adjoint.

**Solution**:

**Step 1**: Verify symmetric.

For $\psi, \phi \in D(\hat{p})$:
$$\langle \hat{p}\psi, \phi \rangle = \int_0^1 \left(-i\psi'(x)\right)\overline{\phi(x)}\, dx$$

Integrate by parts:
$$= -i[\psi\bar{\phi}]_0^1 + i\int_0^1 \psi(x)\overline{\phi'(x)}\, dx$$

With periodic boundary conditions, $[\psi\bar{\phi}]_0^1 = \psi(1)\overline{\phi(1)} - \psi(0)\overline{\phi(0)} = 0$.

So:
$$\langle \hat{p}\psi, \phi \rangle = \int_0^1 \psi(x)\overline{(-i\phi'(x))}\, dx = \langle \psi, \hat{p}\phi \rangle$$

$\hat{p}$ is symmetric. ✓

**Step 2**: Find $D(\hat{p}^*)$.

$\phi \in D(\hat{p}^*)$ iff the map $\psi \mapsto \langle \hat{p}\psi, \phi \rangle$ is bounded.

By the same integration by parts, this requires $\phi' \in L^2$, i.e., $\phi \in H^1[0,1]$.

But we also need the boundary term to vanish for all $\psi \in D(\hat{p})$, which gives no additional constraint since $\psi(0) = \psi(1)$.

**Wait**: We need $[\psi\bar{\phi}]_0^1 = 0$ for all periodic $\psi$. Testing with specific $\psi$ shows we need $\phi(0) = \phi(1)$.

**Step 3**: Conclude.

$D(\hat{p}^*) = \{\phi \in H^1[0,1] : \phi(0) = \phi(1)\} = D(\hat{p})$

Therefore $\hat{p} = \hat{p}^*$, so $\hat{p}$ is self-adjoint. ✓

### Example 2: Deficiency Indices for Momentum on Half-Line

**Problem**: For $\hat{p} = -i\frac{d}{dx}$ on $L^2[0, \infty)$ with $D(\hat{p}) = C_0^\infty(0, \infty)$ (smooth functions with compact support away from 0), find the deficiency indices.

**Solution**:

**Step 1**: Find $\hat{p}^*$.

For $\phi \in D(\hat{p}^*)$, we need $\psi \mapsto \langle \hat{p}\psi, \phi \rangle$ bounded for $\psi \in C_0^\infty(0,\infty)$.

$\hat{p}^* = -i\frac{d}{dx}$ on $D(\hat{p}^*) = H^1[0,\infty)$ (no boundary conditions needed since test functions vanish near 0).

**Step 2**: Find deficiency subspaces.

Solve $\hat{p}^*\phi = i\phi$:
$$-i\phi' = i\phi \implies \phi' = -\phi \implies \phi(x) = ce^{-x}$$

Is $e^{-x} \in L^2[0,\infty)$? Yes! So $\mathcal{N}_+ = \text{span}\{e^{-x}\}$, $n_+ = 1$.

Solve $\hat{p}^*\phi = -i\phi$:
$$-i\phi' = -i\phi \implies \phi' = \phi \implies \phi(x) = ce^{x}$$

Is $e^{x} \in L^2[0,\infty)$? No! So $\mathcal{N}_- = \{0\}$, $n_- = 0$.

**Step 3**: Conclusion.

$$\boxed{n_+ = 1, \quad n_- = 0}$$

Since $n_+ \neq n_-$, $\hat{p}$ has **no** self-adjoint extensions on $L^2[0,\infty)$.

**Physical interpretation**: Momentum is not observable for a particle confined to a half-line with a hard wall at $x = 0$.

### Example 3: Harmonic Oscillator is Self-Adjoint

**Problem**: Show that the harmonic oscillator Hamiltonian $H = -\frac{d^2}{dx^2} + x^2$ on $L^2(\mathbb{R})$ is essentially self-adjoint on $C_0^\infty(\mathbb{R})$.

**Solution**:

**Step 1**: Use Nelson's criterion.

The Hermite functions $h_n(x) = H_n(x)e^{-x^2/2}$ form an orthonormal basis of $L^2(\mathbb{R})$ and satisfy:
- $h_n \in \mathcal{S}(\mathbb{R}) \subset C_0^\infty(\mathbb{R})$ in the closure sense
- $Hh_n = (2n+1)h_n$

So $H$ acts on a dense set of eigenfunctions.

**Step 2**: Apply von Neumann's criterion.

Need to show $(H^* \pm i)$ has trivial kernel.

Suppose $(H^* - i)\psi = 0$, i.e., $(-\psi'' + x^2\psi) = i\psi$.

This is the differential equation $\psi'' = (x^2 - i)\psi$.

Any $L^2$ solution must decay at $\pm\infty$. Analysis shows no such solution exists (the WKB approximation gives growing solutions at infinity).

Similarly for $(H^* + i)\psi = 0$.

**Step 3**: Conclusion.

$n_+ = n_- = 0$, so $H$ is essentially self-adjoint on $C_0^\infty(\mathbb{R})$.

Its unique self-adjoint extension has domain $D(H) = \{\psi \in H^2(\mathbb{R}) : x^2\psi \in L^2(\mathbb{R})\}$.

---

## Practice Problems

### Level 1: Direct Application

1. **Closed Operator**: Show that the differentiation operator $A = \frac{d}{dx}$ on $D(A) = H^1[0,1]$ is closed.

2. **Graph Norm**: For $A = -\frac{d^2}{dx^2}$ on $D(A) = H^2[0,1] \cap H_0^1[0,1]$, write out the graph norm $\|f\|_A$.

3. **Symmetric Check**: Verify that $\hat{x} = x$ on $D(\hat{x}) = \{f \in L^2(\mathbb{R}) : xf \in L^2\}$ is symmetric.

### Level 2: Intermediate

4. **Adjoint Computation**: Find $D(A^*)$ for $A = i\frac{d}{dx}$ on $D(A) = \{f \in H^1[0,1] : f(0) = 0\}$.

5. **Self-Adjoint Extension**: The operator $A = -\frac{d^2}{dx^2}$ on $D(A) = C_0^\infty(0,1)$ has deficiency indices $(2, 2)$. Describe the self-adjoint extensions.

6. **Kato-Rellich**: Using Kato-Rellich, show $H = -\frac{d^2}{dx^2} + V(x)$ is self-adjoint on $H^2(\mathbb{R})$ if $V \in L^\infty(\mathbb{R})$.

### Level 3: Challenging

7. **Deficiency Indices**: Compute the deficiency indices for $A = -\frac{d^2}{dx^2}$ on $L^2[0,\infty)$ with $D(A) = C_0^\infty(0,\infty)$.

8. **Non-Closable**: Construct an explicit example of a densely defined non-closable operator.

9. **Weyl's Limit Point/Circle**: For $-\frac{d^2}{dx^2} + q(x)$ on $[0,\infty)$, explain how Weyl's theory determines self-adjoint extensions.

---

## Computational Lab: Position and Momentum Operators

```python
"""
Day 250 Computational Lab: Unbounded Operators
Position, momentum, and the harmonic oscillator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.special import hermite
from scipy.integrate import quad
from typing import Tuple, Callable

np.random.seed(42)

# =============================================================================
# Part 1: Discretizing the Position Operator
# =============================================================================

print("="*70)
print("PART 1: POSITION OPERATOR")
print("="*70)

def position_operator_matrix(n: int, L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create matrix representation of position operator on L^2[-L, L].
    Uses n grid points with spacing dx = 2L/n.
    """
    x = np.linspace(-L, L, n, endpoint=False) + L/n
    X = np.diag(x)
    return X, x

# Create position operator
n = 100
L = 5.0
X, x = position_operator_matrix(n, L)

print(f"Position operator on L^2[{-L}, {L}] with {n} grid points")
print(f"Matrix is diagonal with entries x_i")

# Eigenvalues of position operator (should just be the positions)
eigenvalues_X = np.linalg.eigvalsh(X)
print(f"\nEigenvalues of X (first 5): {eigenvalues_X[:5]}")
print(f"Grid points (first 5): {x[:5]}")
print(f"Match: {np.allclose(np.sort(eigenvalues_X), np.sort(x))}")

# =============================================================================
# Part 2: Discretizing the Momentum Operator
# =============================================================================

print("\n" + "="*70)
print("PART 2: MOMENTUM OPERATOR")
print("="*70)

def momentum_operator_matrix(n: int, L: float, hbar: float = 1.0) -> np.ndarray:
    """
    Create matrix representation of momentum operator p = -i*hbar*d/dx.
    Uses periodic boundary conditions and finite differences.
    """
    dx = 2*L / n

    # Central difference: (f(x+dx) - f(x-dx)) / (2*dx)
    # This is an anti-symmetric matrix (representing -i*d/dx)
    P = np.zeros((n, n), dtype=complex)
    for i in range(n):
        P[i, (i+1) % n] = 1 / (2*dx)
        P[i, (i-1) % n] = -1 / (2*dx)

    P = -1j * hbar * P
    return P

P = momentum_operator_matrix(n, L)

print(f"Momentum operator -i*d/dx (periodic BC)")
print(f"Matrix structure: tri-diagonal with wrap-around")

# Check that P is anti-Hermitian (so iP is Hermitian)
print(f"\nP is anti-Hermitian: {np.allclose(P, -P.conj().T)}")
print(f"Therefore -iP is Hermitian (self-adjoint)")

# Eigenvalues of P (should be discrete momenta for periodic BC)
eigenvalues_P = np.linalg.eigvalsh(1j * P)  # eigenvalues of -iP * i = P
print(f"\nExpected eigenvalues: 2*pi*n / (2L) for integer n")
print(f"p_1 expected: {np.pi / L:.4f}")
print(f"Computed eigenvalues (sorted by magnitude):")
sorted_eigs = np.sort(np.abs(eigenvalues_P))
print(f"  {sorted_eigs[:5]}")

# =============================================================================
# Part 3: Canonical Commutation Relation
# =============================================================================

print("\n" + "="*70)
print("PART 3: CANONICAL COMMUTATION RELATION [X, P] = i*hbar")
print("="*70)

hbar = 1.0
commutator_XP = X @ P - P @ X

# The commutator should be approximately i*hbar*I
expected_commutator = 1j * hbar * np.eye(n)

# Check (note: discretization introduces errors)
error = np.linalg.norm(commutator_XP - expected_commutator) / np.linalg.norm(expected_commutator)
print(f"Relative error in [X, P] = i*hbar*I: {error:.4f}")

# The commutator in finite dimensions can't exactly equal i*I (trace argument)
print(f"\nTrace([X, P]) = {np.trace(commutator_XP):.4f}")
print(f"Trace(i*hbar*I) = {n * 1j * hbar:.4f}")
print("Note: tr([A, B]) = 0 for finite matrices, so exact CCR is impossible!")

# Better approximation: check on smooth test functions
# [X, P]f ≈ i*hbar*f for smooth f
test_function = np.exp(-x**2)  # Gaussian
test_function = test_function / np.linalg.norm(test_function)

XP_f = X @ P @ test_function
PX_f = P @ X @ test_function
commutator_applied = XP_f - PX_f
expected_applied = 1j * hbar * test_function

print(f"\nApplied to Gaussian:")
print(f"||[X,P]f - i*hbar*f|| / ||f|| = {np.linalg.norm(commutator_applied - expected_applied):.4f}")

# =============================================================================
# Part 4: Harmonic Oscillator
# =============================================================================

print("\n" + "="*70)
print("PART 4: QUANTUM HARMONIC OSCILLATOR")
print("="*70)

def harmonic_oscillator_hamiltonian(n: int, L: float,
                                    m: float = 1.0, omega: float = 1.0,
                                    hbar: float = 1.0) -> np.ndarray:
    """
    Create Hamiltonian H = p^2/(2m) + m*omega^2*x^2/2 on L^2[-L, L].
    """
    X, x = position_operator_matrix(n, L)
    P = momentum_operator_matrix(n, L, hbar)

    # Kinetic energy: p^2/(2m)
    # Use P^2 carefully (or finite difference for -d^2/dx^2)
    dx = 2*L / n
    kinetic = np.zeros((n, n), dtype=complex)
    for i in range(n):
        kinetic[i, i] = -2
        kinetic[i, (i+1) % n] = 1
        kinetic[i, (i-1) % n] = 1
    kinetic = -hbar**2 / (2*m * dx**2) * kinetic

    # Potential energy: m*omega^2*x^2/2
    potential = 0.5 * m * omega**2 * np.diag(x**2)

    H = kinetic + potential
    return H, x

# Create Hamiltonian
n_ho = 200
L_ho = 10.0
H, x_ho = harmonic_oscillator_hamiltonian(n_ho, L_ho)

# Make Hermitian (numerical errors)
H = (H + H.conj().T) / 2

# Compute eigenvalues and eigenvectors
eigenvalues_H, eigenvectors_H = np.linalg.eigh(H)

print(f"Harmonic oscillator: H = p^2/2 + x^2/2")
print(f"\nFirst 10 eigenvalues:")
print(f"  Computed: {eigenvalues_H[:10]}")
print(f"  Theory (n + 1/2): {np.arange(10) + 0.5}")
print(f"  Relative errors: {np.abs(eigenvalues_H[:10] - (np.arange(10) + 0.5)) / (np.arange(10) + 0.5)}")

# =============================================================================
# Part 5: Visualizing Harmonic Oscillator Eigenstates
# =============================================================================

print("\n" + "="*70)
print("PART 5: HARMONIC OSCILLATOR EIGENSTATES")
print("="*70)

# Analytic Hermite function solutions
def hermite_function(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute the n-th Hermite function (normalized).
    psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) * H_n(x) * exp(-x^2/2)
    """
    Hn = hermite(n)
    normalization = 1 / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
    return normalization * Hn(x) * np.exp(-x**2 / 2)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, ax in enumerate(axes.flat):
    if idx >= 6:
        break

    # Numerical eigenstate
    psi_numerical = eigenvectors_H[:, idx]
    # Normalize and fix sign
    psi_numerical = psi_numerical / np.sqrt(np.sum(np.abs(psi_numerical)**2) * (2*L_ho/n_ho))
    if psi_numerical[n_ho//2] < 0:
        psi_numerical = -psi_numerical

    # Analytic eigenstate
    psi_analytic = hermite_function(idx, x_ho)

    ax.plot(x_ho, np.real(psi_numerical), 'b-', linewidth=2, label='Numerical')
    ax.plot(x_ho, psi_analytic, 'r--', linewidth=2, label='Analytic')
    ax.fill_between(x_ho, 0, 0.1 * x_ho**2, alpha=0.2, color='gray', label='V(x)')

    ax.set_xlabel('x')
    ax.set_ylabel('$\\psi_n(x)$')
    ax.set_title(f'n = {idx}, E = {eigenvalues_H[idx]:.4f}')
    ax.set_xlim([-5, 5])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('harmonic_oscillator_eigenstates.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: harmonic_oscillator_eigenstates.png")

# =============================================================================
# Part 6: Domain Issues - Unboundedness
# =============================================================================

print("\n" + "="*70)
print("PART 6: DEMONSTRATING UNBOUNDEDNESS")
print("="*70)

def demonstrate_unboundedness():
    """Show that X and P are unbounded by exhibiting vectors with large <X^2> or <P^2>."""

    # For position: use states localized far from origin
    # For momentum: use highly oscillatory states

    n = 500
    L = 20.0
    X, x = position_operator_matrix(n, L)
    P = momentum_operator_matrix(n, L)

    # States with increasing position uncertainty
    print("Position operator is unbounded:")
    print("-" * 40)
    for x0 in [2, 5, 10, 15]:
        # Gaussian centered at x0
        psi = np.exp(-(x - x0)**2)
        psi = psi / np.linalg.norm(psi)

        X_expectation = np.real(psi.conj() @ X @ psi)
        X2_expectation = np.real(psi.conj() @ X @ X @ psi)

        print(f"  x0 = {x0:3d}: <X> = {X_expectation:7.3f}, <X^2> = {X2_expectation:10.3f}")

    # States with increasing momentum
    print("\nMomentum operator is unbounded:")
    print("-" * 40)
    for k in [1, 5, 10, 20]:
        # Plane wave with wavenumber k, localized by Gaussian
        psi = np.exp(1j * k * x) * np.exp(-x**2 / 4)
        psi = psi / np.linalg.norm(psi)

        P_expectation = np.real(psi.conj() @ P @ psi)
        P2_expectation = np.real(psi.conj() @ P @ P @ psi)

        print(f"  k = {k:3d}: <P> = {P_expectation:7.3f}, <P^2> = {P2_expectation:10.3f}")

demonstrate_unboundedness()

# =============================================================================
# Part 7: Uncertainty Principle
# =============================================================================

print("\n" + "="*70)
print("PART 7: HEISENBERG UNCERTAINTY PRINCIPLE")
print("="*70)

def compute_uncertainties(psi: np.ndarray, X: np.ndarray, P: np.ndarray) -> Tuple[float, float]:
    """Compute Delta_X and Delta_P for state psi."""
    psi = psi / np.linalg.norm(psi)

    X_mean = np.real(psi.conj() @ X @ psi)
    X2_mean = np.real(psi.conj() @ X @ X @ psi)
    Delta_X = np.sqrt(X2_mean - X_mean**2)

    P_mean = np.real(psi.conj() @ P @ psi)
    P2_mean = np.real(psi.conj() @ P @ P @ psi)
    Delta_P = np.sqrt(max(0, P2_mean - P_mean**2))

    return Delta_X, Delta_P

n = 300
L = 15.0
X, x = position_operator_matrix(n, L)
P = momentum_operator_matrix(n, L)

print("Testing Delta_X * Delta_P >= hbar/2:")
print("-" * 50)

# Different states
states = []

# 1. Gaussian (minimum uncertainty)
for sigma in [0.5, 1.0, 2.0]:
    psi = np.exp(-x**2 / (2*sigma**2))
    states.append((psi, f"Gaussian (sigma={sigma})"))

# 2. Ground state of harmonic oscillator (minimum uncertainty)
psi_ho = np.exp(-x**2 / 2)
states.append((psi_ho, "HO ground state"))

# 3. Superposition (increased uncertainty)
psi_super = np.exp(-(x-2)**2) + np.exp(-(x+2)**2)
states.append((psi_super, "Superposition"))

for psi, name in states:
    Delta_X, Delta_P = compute_uncertainties(psi, X, P)
    product = Delta_X * Delta_P
    print(f"  {name:25s}: Delta_X = {Delta_X:.4f}, Delta_P = {Delta_P:.4f}, product = {product:.4f} (>= 0.5)")

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))

sigmas = np.linspace(0.3, 3.0, 50)
Delta_X_list = []
Delta_P_list = []

for sigma in sigmas:
    psi = np.exp(-x**2 / (2*sigma**2))
    Delta_X, Delta_P = compute_uncertainties(psi, X, P)
    Delta_X_list.append(Delta_X)
    Delta_P_list.append(Delta_P)

ax.plot(Delta_X_list, Delta_P_list, 'bo-', markersize=3, label='Gaussian states')

# Uncertainty bound
Delta_X_theory = np.linspace(0.3, 3.0, 100)
Delta_P_bound = 0.5 / Delta_X_theory
ax.plot(Delta_X_theory, Delta_P_bound, 'r-', linewidth=2,
        label='Uncertainty bound $\\Delta_X \\Delta_P = \\hbar/2$')

ax.set_xlabel('$\\Delta X$')
ax.set_ylabel('$\\Delta P$')
ax.set_title('Heisenberg Uncertainty Principle')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 3.5])
ax.set_ylim([0, 2])

plt.tight_layout()
plt.savefig('uncertainty_principle.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: uncertainty_principle.png")

# =============================================================================
# Part 8: Self-Adjointness Check
# =============================================================================

print("\n" + "="*70)
print("PART 8: VERIFYING SELF-ADJOINTNESS")
print("="*70)

def check_self_adjoint(A: np.ndarray, name: str) -> None:
    """Check if matrix A is self-adjoint (Hermitian)."""
    is_hermitian = np.allclose(A, A.conj().T)
    eigenvalues = np.linalg.eigvalsh(A)
    all_real = np.allclose(eigenvalues.imag, 0)

    print(f"\n{name}:")
    print(f"  Hermitian (A = A^*): {is_hermitian}")
    print(f"  All eigenvalues real: {all_real}")
    print(f"  Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

check_self_adjoint(X, "Position operator X")
check_self_adjoint((P + P.conj().T)/2, "Momentum operator P (symmetrized)")
check_self_adjoint(H, "Harmonic oscillator H")

print("\n" + "="*70)
print("LAB COMPLETE")
print("="*70)
print("""
Key takeaways:
1. Position X and momentum P are unbounded - <X^2> and <P^2> can be arbitrarily large
2. The canonical commutation relation [X, P] = i*hbar holds approximately
3. In finite dimensions, exact CCR is impossible (trace argument)
4. Harmonic oscillator eigenvalues: E_n = (n + 1/2) * hbar * omega
5. Gaussian states saturate the uncertainty bound Delta_X * Delta_P = hbar/2
6. Self-adjointness requires careful attention to domains
""")
```

---

## Summary

### Key Formulas

| Concept | Formula/Definition |
|---------|-------------------|
| Unbounded operator | $A: D(A) \to \mathcal{H}$ with $D(A)$ dense |
| Graph | $\Gamma(A) = \{(x, Ax) : x \in D(A)\}$ |
| Graph norm | $\|x\|_A = \sqrt{\|x\|^2 + \|Ax\|^2}$ |
| Adjoint domain | $D(A^*) = \{y : x \mapsto \langle Ax, y\rangle \text{ bounded}\}$ |
| Symmetric | $A \subseteq A^*$ |
| Self-adjoint | $A = A^*$ |
| Deficiency indices | $n_\pm = \dim\ker(A^* \mp iI)$ |
| Von Neumann criterion | $A$ ess. self-adjoint iff $n_+ = n_- = 0$ |

### Main Takeaways

1. **Unbounded operators require domains**: The pair $(A, D(A))$ defines the operator.

2. **Closed vs. closable**: Closed means the graph is closed; closable means the closure is a graph.

3. **Symmetric ≠ self-adjoint**: Symmetric means $A \subseteq A^*$; self-adjoint means $A = A^*$.

4. **Deficiency indices classify extensions**: $(n_+, n_-)$ determines if/how many self-adjoint extensions exist.

5. **Position and momentum are self-adjoint**: With proper domains, $\hat{x}$ and $\hat{p}$ are self-adjoint on $L^2(\mathbb{R})$.

6. **Spectral theorem extends**: Self-adjoint unbounded operators have spectral measures on $\mathbb{R}$ (possibly unbounded).

---

## Daily Checklist

- [ ] I can define unbounded operators with their domains
- [ ] I understand the difference between closed and closable operators
- [ ] I can distinguish symmetric, essentially self-adjoint, and self-adjoint
- [ ] I can apply von Neumann's criterion to check essential self-adjointness
- [ ] I understand why position and momentum are unbounded
- [ ] I can compute deficiency indices for simple operators
- [ ] I understand the spectral theorem for unbounded operators
- [ ] I completed the computational lab

---

## Preview: Day 251

Tomorrow we prove **Stone's Theorem**, which establishes a beautiful correspondence:
$$\text{Self-adjoint operators } A \quad \longleftrightarrow \quad \text{Strongly continuous unitary groups } U(t) = e^{-iAt}$$

This theorem explains why the Hamiltonian generates time evolution in quantum mechanics and gives the mathematical foundation for the Schrödinger equation.

---

*"The treatment of unbounded operators requires considerably more care than the treatment of bounded operators; the problem of specifying the domain is one of the most subtle aspects of the theory."*
— Michael Reed & Barry Simon
