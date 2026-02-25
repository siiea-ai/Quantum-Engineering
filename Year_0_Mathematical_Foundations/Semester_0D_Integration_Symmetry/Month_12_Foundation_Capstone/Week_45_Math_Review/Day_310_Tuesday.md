# Day 310: Linear Algebra Synthesis — From Vectors to Operators

## Overview

**Month 12, Week 45, Day 2 — Tuesday**

Today we synthesize linear algebra from basic vectors to abstract operators. This progression — vectors → matrices → abstract operators — directly parallels quantum mechanics, where states are vectors, observables are operators, and measurements are eigenvalue problems.

## Learning Objectives

1. Review vector spaces and their properties
2. Synthesize matrix theory and eigenvalue problems
3. Connect to abstract operator theory
4. Prepare for quantum mechanical formalism

---

## 1. Vector Spaces

### Definition

A vector space $V$ over field $\mathbb{F}$ satisfies:
- Closure under addition and scalar multiplication
- Associativity, commutativity of addition
- Existence of zero vector and additive inverses
- Distributivity of scalar multiplication

### Key Examples

| Space | Elements | Field |
|-------|----------|-------|
| $\mathbb{R}^n$ | n-tuples | $\mathbb{R}$ |
| $\mathbb{C}^n$ | Complex n-tuples | $\mathbb{C}$ |
| $\mathcal{P}_n$ | Polynomials degree ≤ n | $\mathbb{R}$ or $\mathbb{C}$ |
| $L^2[a,b]$ | Square-integrable functions | $\mathbb{C}$ |
| $\ell^2$ | Square-summable sequences | $\mathbb{C}$ |

### Linear Independence and Basis

Vectors $\{v_1, ..., v_n\}$ are **linearly independent** if:
$$c_1 v_1 + ... + c_n v_n = 0 \implies c_i = 0 \; \forall i$$

A **basis** is a maximal linearly independent set. Dimension = number of basis vectors.

---

## 2. Inner Product Spaces

### Definition

An inner product $\langle \cdot | \cdot \rangle$ satisfies:
- $\langle u | v \rangle = \overline{\langle v | u \rangle}$ (conjugate symmetry)
- $\langle u | \alpha v + \beta w \rangle = \alpha\langle u | v \rangle + \beta\langle u | w \rangle$ (linearity in second argument)
- $\langle v | v \rangle \geq 0$, equality iff $v = 0$ (positive definiteness)

### Norm and Distance

$$\|v\| = \sqrt{\langle v | v \rangle}$$
$$d(u, v) = \|u - v\|$$

### Key Inequalities

**Cauchy-Schwarz:** $|\langle u | v \rangle| \leq \|u\| \|v\|$

**Triangle:** $\|u + v\| \leq \|u\| + \|v\|$

### Orthogonality

$$u \perp v \iff \langle u | v \rangle = 0$$

**Orthonormal basis:** $\langle e_i | e_j \rangle = \delta_{ij}$

---

## 3. Matrix Theory

### Matrices as Linear Transformations

$$T: V \to W, \quad T(v) = Av$$

Matrix elements: $A_{ij} = \langle e_i | T | e_j \rangle$

### Special Matrices

| Type | Definition | Properties |
|------|------------|------------|
| Hermitian | $A^\dagger = A$ | Real eigenvalues, orthogonal eigenvectors |
| Unitary | $U^\dagger U = I$ | Preserves inner product |
| Normal | $[A, A^\dagger] = 0$ | Diagonalizable by unitary |
| Positive | $\langle v|A|v\rangle \geq 0$ | Non-negative eigenvalues |
| Projection | $P^2 = P = P^\dagger$ | Eigenvalues 0 or 1 |

### Matrix Operations

$$\text{Trace:} \quad \text{Tr}(A) = \sum_i A_{ii}$$
$$\text{Determinant:} \quad \det(AB) = \det(A)\det(B)$$
$$(AB)^\dagger = B^\dagger A^\dagger$$

---

## 4. Eigenvalue Problems

### Definition

$$A|v\rangle = \lambda|v\rangle$$

- $\lambda$: eigenvalue
- $|v\rangle$: eigenvector

### Characteristic Polynomial

$$p(\lambda) = \det(A - \lambda I)$$

Eigenvalues are roots of $p(\lambda) = 0$.

### Spectral Theorem

For Hermitian $A$:
1. All eigenvalues are real
2. Eigenvectors for different eigenvalues are orthogonal
3. $A = \sum_i \lambda_i |i\rangle\langle i|$ (spectral decomposition)

### Diagonalization

If $A$ has $n$ linearly independent eigenvectors:
$$A = PDP^{-1}$$

where $D = \text{diag}(\lambda_1, ..., \lambda_n)$ and $P = [v_1 | ... | v_n]$.

For Hermitian: $P$ is unitary, $A = UDU^\dagger$.

---

## 5. Abstract Operators

### Linear Operators

$$T(\alpha u + \beta v) = \alpha T(u) + \beta T(v)$$

### Adjoint

$$\langle u | T v \rangle = \langle T^\dagger u | v \rangle$$

### Commutators

$$[A, B] = AB - BA$$

**Key property:** $[A, B]^\dagger = [B^\dagger, A^\dagger]$

### Functions of Operators

$$f(A) = \sum_i f(\lambda_i)|i\rangle\langle i|$$

For matrix exponential:
$$e^A = \sum_{n=0}^{\infty}\frac{A^n}{n!}$$

---

## 6. Quantum Mechanics Preview

### States as Vectors

$$|\psi\rangle \in \mathcal{H}$$

### Observables as Operators

$$\hat{A}|\psi\rangle = a|\psi\rangle \implies \text{measurement outcome } a$$

### Probability Amplitudes

$$P(a) = |\langle a | \psi \rangle|^2$$

### The Postulates (Preview)

1. States are vectors in Hilbert space
2. Observables are Hermitian operators
3. Measurements yield eigenvalues
4. Time evolution is unitary

---

## 7. Computational Lab

```python
"""
Day 310: Linear Algebra Synthesis
"""

import numpy as np
from scipy.linalg import expm, logm, eigh
import matplotlib.pyplot as plt

def linear_algebra_fundamentals():
    """Demonstrate core linear algebra concepts."""
    print("=" * 50)
    print("LINEAR ALGEBRA FUNDAMENTALS")
    print("=" * 50)

    # Eigenvalue problem
    A = np.array([[2, 1], [1, 2]], dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    print("\n1. Eigenvalue Problem")
    print(f"   A = \n{A}")
    print(f"   Eigenvalues: {eigenvalues}")
    print(f"   Eigenvectors:\n{eigenvectors}")

    # Verify A = V D V^†
    D = np.diag(eigenvalues)
    V = eigenvectors
    A_reconstructed = V @ D @ V.conj().T
    print(f"\n   A = VDV†: {np.allclose(A, A_reconstructed)}")

    # Spectral decomposition
    print("\n2. Spectral Decomposition")
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        projector = np.outer(v, v.conj())
        print(f"   λ_{i} = {lam:.4f}")
        print(f"   |{i}⟩⟨{i}| = \n{projector.round(4)}")


def hermitian_properties():
    """Demonstrate Hermitian matrix properties."""
    print("\n" + "=" * 50)
    print("HERMITIAN MATRIX PROPERTIES")
    print("=" * 50)

    # Random Hermitian matrix
    n = 3
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = (H + H.conj().T) / 2  # Make Hermitian

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    print("\n1. Eigenvalues are real:")
    print(f"   λ = {eigenvalues}")
    print(f"   All real: {np.allclose(eigenvalues.imag, 0)}")

    print("\n2. Eigenvectors are orthonormal:")
    overlap = eigenvectors.conj().T @ eigenvectors
    print(f"   V†V = \n{overlap.round(4)}")
    print(f"   Is identity: {np.allclose(overlap, np.eye(n))}")


def unitary_properties():
    """Demonstrate unitary matrix properties."""
    print("\n" + "=" * 50)
    print("UNITARY MATRIX PROPERTIES")
    print("=" * 50)

    # Create unitary from Hermitian
    H = np.array([[1, 1j], [-1j, 2]])
    U = expm(1j * H)

    print("\n1. U = exp(iH):")
    print(f"   U = \n{U.round(4)}")

    print("\n2. U†U = I:")
    product = U.conj().T @ U
    print(f"   U†U = \n{product.round(4)}")
    print(f"   Is unitary: {np.allclose(product, np.eye(2))}")

    print("\n3. |det(U)| = 1:")
    print(f"   |det(U)| = {np.abs(np.linalg.det(U)):.6f}")

    print("\n4. Preserves inner product:")
    v = np.array([1, 0])
    w = np.array([0, 1])
    inner_before = v.conj() @ w
    inner_after = (U @ v).conj() @ (U @ w)
    print(f"   ⟨v|w⟩ = {inner_before}")
    print(f"   ⟨Uv|Uw⟩ = {inner_after.round(6)}")


def commutator_relations():
    """Explore commutator algebra."""
    print("\n" + "=" * 50)
    print("COMMUTATOR RELATIONS")
    print("=" * 50)

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    print("\n1. Pauli matrix commutators:")
    print(f"   [σx, σy] = 2i σz: {np.allclose(sigma_x @ sigma_y - sigma_y @ sigma_x, 2j * sigma_z)}")
    print(f"   [σy, σz] = 2i σx: {np.allclose(sigma_y @ sigma_z - sigma_z @ sigma_y, 2j * sigma_x)}")
    print(f"   [σz, σx] = 2i σy: {np.allclose(sigma_z @ sigma_x - sigma_x @ sigma_z, 2j * sigma_y)}")

    print("\n2. Anticommutators:")
    print(f"   {σx, σy} = 0: {np.allclose(sigma_x @ sigma_y + sigma_y @ sigma_x, 0)}")
    print(f"   {σx, σx} = 2I: {np.allclose(sigma_x @ sigma_x + sigma_x @ sigma_x, 2 * np.eye(2))}")


def functions_of_matrices():
    """Compute functions of matrices."""
    print("\n" + "=" * 50)
    print("FUNCTIONS OF MATRICES")
    print("=" * 50)

    A = np.array([[1, 2], [0, 3]])

    print("\n1. Matrix exponential e^A:")
    exp_A = expm(A)
    print(f"   e^A = \n{exp_A.round(4)}")

    # Verify via eigenvalues for diagonalizable matrix
    eigenvalues, eigenvectors = np.linalg.eig(A)
    D_exp = np.diag(np.exp(eigenvalues))
    exp_A_eigen = eigenvectors @ D_exp @ np.linalg.inv(eigenvectors)
    print(f"\n   Via eigenvalues:\n{exp_A_eigen.round(4)}")
    print(f"   Match: {np.allclose(exp_A, exp_A_eigen)}")

    # Derivative of e^{tA}
    print("\n2. d/dt e^{tA} = A e^{tA}:")
    t = 0.5
    dt = 1e-6
    deriv_numerical = (expm((t + dt) * A) - expm(t * A)) / dt
    deriv_analytical = A @ expm(t * A)
    print(f"   Numerical: \n{deriv_numerical.round(4)}")
    print(f"   Analytical:\n{deriv_analytical.round(4)}")
    print(f"   Match: {np.allclose(deriv_numerical, deriv_analytical, rtol=1e-4)}")


def projection_operators():
    """Demonstrate projection operators."""
    print("\n" + "=" * 50)
    print("PROJECTION OPERATORS")
    print("=" * 50)

    # Eigenvector projection
    v = np.array([1, 1]) / np.sqrt(2)
    P = np.outer(v, v)

    print("\n1. Projection onto v = (1,1)/√2:")
    print(f"   P = |v⟩⟨v| = \n{P.round(4)}")
    print(f"   P² = P: {np.allclose(P @ P, P)}")
    print(f"   P† = P: {np.allclose(P.conj().T, P)}")
    print(f"   Tr(P) = 1: {np.trace(P):.4f}")

    # Orthogonal complement
    Q = np.eye(2) - P
    print(f"\n2. Orthogonal complement Q = I - P:")
    print(f"   Q = \n{Q.round(4)}")
    print(f"   PQ = 0: {np.allclose(P @ Q, 0)}")


# Main execution
if __name__ == "__main__":
    linear_algebra_fundamentals()
    hermitian_properties()
    unitary_properties()
    commutator_relations()
    functions_of_matrices()
    projection_operators()
```

---

## 8. Practice Problems

### Problem 1: Eigenvalues

Find the eigenvalues and eigenvectors of:
$$A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

### Problem 2: Spectral Decomposition

Write the spectral decomposition of the Hamiltonian $H = \omega\sigma_z$.

### Problem 3: Unitary Evolution

If $U(t) = e^{-iHt/\hbar}$ with $H = \begin{pmatrix} E_1 & 0 \\ 0 & E_2 \end{pmatrix}$, find $U(t)$ explicitly.

### Problem 4: Commutators

Show that $[A, BC] = [A, B]C + B[A, C]$.

---

## Summary

### The Linear Algebra → QM Dictionary

| Linear Algebra | Quantum Mechanics |
|----------------|-------------------|
| Vector | State |
| Inner product | Probability amplitude |
| Hermitian matrix | Observable |
| Eigenvalue | Measurement outcome |
| Eigenvector | Definite-value state |
| Unitary matrix | Time evolution |
| Projection | Measurement |

---

## Preview: Day 311

Tomorrow: **Differential Equations Synthesis** — from ODEs to PDEs, eigenvalue problems to special functions.
