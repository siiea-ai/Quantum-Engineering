# Day 340: Hermitian and Unitary Operators — The Pillars of Quantum Mechanics

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Hermitian and Unitary Operators |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 340, you will be able to:

1. Define and compute the adjoint (Hermitian conjugate) of operators
2. Identify Hermitian operators and prove their eigenvalues are real
3. Prove that eigenvectors of Hermitian operators with distinct eigenvalues are orthogonal
4. Define unitary operators and verify they preserve inner products
5. Show that eigenvalues of unitary operators have unit modulus
6. Connect Hermitian operators to observables and unitary operators to quantum evolution

---

## Core Content

### 1. The Adjoint Operator

The **adjoint** (or **Hermitian conjugate**) of an operator Â is denoted Â† and defined by:

$$\boxed{⟨φ|\hat{A}^†|ψ⟩ = ⟨ψ|\hat{A}|φ⟩^*}$$

This is the fundamental definition. Equivalently, for all |φ⟩, |ψ⟩:

$$⟨\hat{A}^†φ|ψ⟩ = ⟨φ|\hat{A}ψ⟩$$

**Matrix Representation:**
If Â has matrix elements $A_{mn} = ⟨m|\hat{A}|n⟩$ in some orthonormal basis, then:

$$\boxed{(A^†)_{mn} = A_{nm}^* = ⟨n|\hat{A}|m⟩^*}$$

The adjoint is the **conjugate transpose** (transpose + complex conjugate).

---

### 2. Properties of the Adjoint

For operators Â, B̂ and scalar α ∈ ℂ:

| Property | Formula |
|----------|---------|
| Involution | $(\hat{A}^†)^† = \hat{A}$ |
| Antilinearity | $(α\hat{A})^† = α^*\hat{A}^†$ |
| Sum | $(\hat{A} + \hat{B})^† = \hat{A}^† + \hat{B}^†$ |
| Product | $(\hat{A}\hat{B})^† = \hat{B}^†\hat{A}^†$ (order reverses!) |
| Outer product | $(|ψ⟩⟨φ|)^† = |φ⟩⟨ψ|$ |

**Proof of Product Rule:**

$$⟨φ|(\hat{A}\hat{B})^†|ψ⟩ = ⟨ψ|\hat{A}\hat{B}|φ⟩^* = (⟨ψ|\hat{A})(⟨\hat{B}|φ⟩))^*$$

Let |χ⟩ = B̂|φ⟩, then:
$$= ⟨ψ|\hat{A}|χ⟩^* = ⟨χ|\hat{A}^†|ψ⟩ = ⟨φ|\hat{B}^†\hat{A}^†|ψ⟩$$

Therefore $(\hat{A}\hat{B})^† = \hat{B}^†\hat{A}^†$. ∎

---

### 3. Hermitian (Self-Adjoint) Operators

An operator Ĥ is **Hermitian** (or **self-adjoint**) if:

$$\boxed{\hat{H}^† = \hat{H}}$$

Equivalently:
$$⟨φ|\hat{H}|ψ⟩ = ⟨ψ|\hat{H}|φ⟩^* = ⟨\hat{H}φ|ψ⟩$$

**Matrix Criterion:**
A matrix H is Hermitian if $H_{mn} = H_{nm}^*$, i.e., H = H†.

**Examples of Hermitian Operators:**

| Operator | Symbol | Matrix (2D example) |
|----------|--------|---------------------|
| Position | x̂ | (Continuous spectrum) |
| Momentum | p̂ | (Continuous spectrum) |
| Hamiltonian | Ĥ | $\begin{pmatrix} E_0 & 0 \\ 0 & E_1 \end{pmatrix}$ |
| Pauli-X | σ̂ₓ | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ |
| Pauli-Y | σ̂ᵧ | $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ |
| Pauli-Z | σ̂ᵤ | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ |

---

### 4. Fundamental Theorem: Eigenvalues of Hermitian Operators are Real

**Theorem:** If Ĥ is Hermitian, all its eigenvalues are real.

**Proof:**

Let Ĥ|λ⟩ = λ|λ⟩ where |λ⟩ ≠ 0.

Compute ⟨λ|Ĥ|λ⟩ two ways:

**Method 1:** Act Ĥ to the right:
$$⟨λ|\hat{H}|λ⟩ = ⟨λ|λ|λ⟩ = λ⟨λ|λ⟩$$

**Method 2:** Use Hermiticity (Ĥ† = Ĥ):
$$⟨λ|\hat{H}|λ⟩ = ⟨\hat{H}λ|λ⟩ = ⟨λλ|λ⟩ = λ^*⟨λ|λ⟩$$

Since ⟨λ|λ⟩ > 0 for |λ⟩ ≠ 0:
$$λ = λ^*$$

$$\boxed{\text{Therefore } λ ∈ ℝ}$$ ∎

---

### 5. Fundamental Theorem: Orthogonality of Eigenvectors

**Theorem:** Eigenvectors of a Hermitian operator corresponding to distinct eigenvalues are orthogonal.

**Proof:**

Let Ĥ|λ₁⟩ = λ₁|λ₁⟩ and Ĥ|λ₂⟩ = λ₂|λ₂⟩ with λ₁ ≠ λ₂.

Compute ⟨λ₁|Ĥ|λ₂⟩ two ways:

**Method 1:** Act Ĥ to the right:
$$⟨λ_1|\hat{H}|λ_2⟩ = λ_2⟨λ_1|λ_2⟩$$

**Method 2:** Use Hermiticity and reality of λ₁:
$$⟨λ_1|\hat{H}|λ_2⟩ = ⟨\hat{H}λ_1|λ_2⟩ = λ_1^*⟨λ_1|λ_2⟩ = λ_1⟨λ_1|λ_2⟩$$

Equating:
$$(λ_2 - λ_1)⟨λ_1|λ_2⟩ = 0$$

Since λ₁ ≠ λ₂:

$$\boxed{⟨λ_1|λ_2⟩ = 0}$$ ∎

---

### 6. The Spectral Theorem for Hermitian Operators

**Spectral Theorem:** Every Hermitian operator Ĥ on a finite-dimensional Hilbert space has:
1. A complete set of orthonormal eigenvectors {|λₙ⟩}
2. Real eigenvalues {λₙ}
3. Spectral decomposition:

$$\boxed{\hat{H} = \sum_n λ_n |λ_n⟩⟨λ_n|}$$

This is profoundly important: the operator equals a weighted sum of projection operators onto its eigenstates.

**Completeness:**
$$\sum_n |λ_n⟩⟨λ_n| = \hat{I}$$

---

### 7. Unitary Operators

An operator Û is **unitary** if:

$$\boxed{\hat{U}^†\hat{U} = \hat{U}\hat{U}^† = \hat{I}}$$

Equivalently: $\hat{U}^† = \hat{U}^{-1}$

**Key Properties:**

| Property | Meaning |
|----------|---------|
| Invertible | Û⁻¹ = Û† always exists |
| Preserves inner products | ⟨Ûφ\|Ûψ⟩ = ⟨φ\|ψ⟩ |
| Preserves norms | \|\|Û\|ψ⟩\|\| = \|\|\|ψ⟩\|\| |
| \|det(U)\| = 1 | Determinant has unit modulus |

---

### 8. Unitary Operators Preserve Inner Products

**Theorem:** If Û is unitary, then ⟨Ûφ|Ûψ⟩ = ⟨φ|ψ⟩.

**Proof:**

$$⟨\hat{U}φ|\hat{U}ψ⟩ = ⟨φ|\hat{U}^†\hat{U}|ψ⟩ = ⟨φ|\hat{I}|ψ⟩ = ⟨φ|ψ⟩$$ ∎

**Physical Interpretation:** Unitary transformations preserve:
- Normalization (probability conservation)
- Orthogonality (distinguishability)
- All geometric relationships in Hilbert space

---

### 9. Eigenvalues of Unitary Operators

**Theorem:** The eigenvalues of a unitary operator have unit modulus (|λ| = 1).

**Proof:**

Let Û|λ⟩ = λ|λ⟩ with |λ⟩ normalized.

$$⟨λ|λ⟩ = ⟨\hat{U}λ|\hat{U}λ⟩ = ⟨λ|\hat{U}^†\hat{U}|λ⟩ = ⟨λ|λ⟩ \quad ✓$$

But also:
$$⟨\hat{U}λ|\hat{U}λ⟩ = ⟨λλ|λλ⟩ = |λ|^2⟨λ|λ⟩$$

Since ⟨λ|λ⟩ = 1:

$$\boxed{|λ|^2 = 1 \implies |λ| = 1}$$

Therefore λ = e^{iθ} for some θ ∈ ℝ. ∎

---

### 10. Normal Operators

An operator N̂ is **normal** if it commutes with its adjoint:

$$\boxed{[\hat{N}, \hat{N}^†] = \hat{N}\hat{N}^† - \hat{N}^†\hat{N} = 0}$$

**Important:** Both Hermitian and unitary operators are normal:
- Hermitian: Ĥ†Ĥ = ĤĤ = Ĥ² = ĤĤ† ✓
- Unitary: Û†Û = Î = ÛÛ† ✓

**Significance:** Normal operators have spectral decompositions with orthonormal eigenvectors. They are precisely the "diagonalizable" operators in quantum mechanics.

---

### 11. Physical Significance

#### Hermitian Operators = Observables

Every physical observable in quantum mechanics is represented by a Hermitian operator:

| Observable | Operator | Why Hermitian? |
|------------|----------|----------------|
| Energy | Ĥ | Measurement gives real E |
| Position | x̂ | Measurement gives real x |
| Momentum | p̂ | Measurement gives real p |
| Angular momentum | L̂, Ŝ | Measurement gives real values |

**Measurement Postulate:** When measuring observable Â on state |ψ⟩:
- Possible outcomes: eigenvalues of Â (real numbers!)
- Probability of outcome a: $P(a) = |⟨a|ψ⟩|^2$
- State after measurement: |a⟩ (eigenstate collapse)

#### Unitary Operators = Time Evolution and Symmetries

Unitary operators represent:
1. **Time evolution:** |ψ(t)⟩ = Û(t)|ψ(0)⟩ where Û(t) = e^{-iĤt/ℏ}
2. **Symmetry transformations:** Rotations, translations, reflections
3. **Change of basis:** From one orthonormal basis to another

**Why unitary for evolution?** Probability must be conserved:
$$⟨ψ(t)|ψ(t)⟩ = ⟨ψ(0)|\hat{U}^†\hat{U}|ψ(0)⟩ = ⟨ψ(0)|ψ(0)⟩ = 1$$

---

## Quantum Computing Connection

### Hermitian Operators: Quantum Measurements

In quantum computing, measurements are described by Hermitian operators:

**Pauli Measurements:**
- σ̂ᵤ measurement: Measures in {|0⟩, |1⟩} basis
- σ̂ₓ measurement: Measures in {|+⟩, |-⟩} basis
- σ̂ᵧ measurement: Measures in {|+i⟩, |-i⟩} basis

**Expectation Value:** For qubit |ψ⟩ = α|0⟩ + β|1⟩:
$$⟨σ̂_z⟩ = ⟨ψ|σ̂_z|ψ⟩ = |α|^2 - |β|^2$$

### Unitary Operators: Quantum Gates

Every quantum gate is a unitary operator:

| Gate | Matrix | Action |
|------|--------|--------|
| Pauli-X | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip: \|0⟩ ↔ \|1⟩ |
| Pauli-Z | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Phase flip: \|1⟩ → -\|1⟩ |
| Hadamard | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | Creates superposition |
| Phase | $\begin{pmatrix} 1 & 0 \\ 0 & e^{iφ} \end{pmatrix}$ | Adds relative phase |
| CNOT | $\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$ | Entangling gate |

**Universal Gate Sets:**
{H, T, CNOT} can approximate any unitary to arbitrary precision (Solovay-Kitaev theorem).

### The Connection

$$\boxed{\text{Hermitian} \xleftrightarrow{\text{exponentiation}} \text{Unitary}}$$

For Hermitian Ĥ:
$$\hat{U} = e^{i\hat{H}θ}$$
is unitary for any real θ.

**Example:** The Pauli matrices are Hermitian. Their exponentials:
$$R_x(θ) = e^{-iσ_x θ/2}, \quad R_y(θ) = e^{-iσ_y θ/2}, \quad R_z(θ) = e^{-iσ_z θ/2}$$
are rotation gates (unitary).

---

## Worked Examples

### Example 1: Verifying a Matrix is Hermitian

**Problem:** Verify that σ̂ᵧ = $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ is Hermitian.

**Solution:**

Compute the adjoint (conjugate transpose):

$$σ_y^† = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}^† = \begin{pmatrix} 0^* & i^* \\ (-i)^* & 0^* \end{pmatrix} = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} = σ_y$$

Since σ̂ᵧ† = σ̂ᵧ, it is **Hermitian**. ✓

**Find the eigenvalues:**
$$\det(σ_y - λI) = \det\begin{pmatrix} -λ & -i \\ i & -λ \end{pmatrix} = λ^2 - i(-i) = λ^2 - 1 = 0$$

$$λ = ±1$$

The eigenvalues are **real**, as guaranteed by Hermiticity. ∎

---

### Example 2: Verifying a Matrix is Unitary

**Problem:** Verify that the Hadamard gate H = $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ is unitary.

**Solution:**

First, compute H†:
$$H^† = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}^† = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = H$$

(H is real symmetric, so H† = H)

Now compute H†H:
$$H^†H = H^2 = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 1+1 & 1-1 \\ 1-1 & 1+1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I$$

Since H†H = I, the Hadamard gate is **unitary**. ✓

**Note:** The Hadamard gate is both Hermitian (H† = H) and unitary (H†H = I). This means H² = I; it is its own inverse! ∎

---

### Example 3: Eigenvalue Problem for a Hermitian Operator

**Problem:** Find the eigenvalues and normalized eigenvectors of the Hermitian matrix:
$$A = \begin{pmatrix} 1 & 2i \\ -2i & 1 \end{pmatrix}$$

**Solution:**

**Step 1: Verify Hermiticity**
$$A^† = \begin{pmatrix} 1 & (-2i)^* \\ (2i)^* & 1 \end{pmatrix} = \begin{pmatrix} 1 & 2i \\ -2i & 1 \end{pmatrix} = A \quad ✓$$

**Step 2: Find eigenvalues**
$$\det(A - λI) = \det\begin{pmatrix} 1-λ & 2i \\ -2i & 1-λ \end{pmatrix} = (1-λ)^2 - (2i)(-2i)$$
$$= (1-λ)^2 - 4 = 0$$
$$(1-λ) = ±2$$
$$λ_1 = 3, \quad λ_2 = -1$$

Both eigenvalues are **real**. ✓

**Step 3: Find eigenvectors**

For λ₁ = 3:
$$(A - 3I)|v_1⟩ = 0$$
$$\begin{pmatrix} -2 & 2i \\ -2i & -2 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0$$

From row 1: $-2v_1 + 2iv_2 = 0 \implies v_1 = iv_2$

Choose $v_2 = 1$: $|v_1⟩ = \begin{pmatrix} i \\ 1 \end{pmatrix}$

Normalized: $|λ_1⟩ = \frac{1}{\sqrt{2}}\begin{pmatrix} i \\ 1 \end{pmatrix}$

For λ₂ = -1:
$$(A + I)|v_2⟩ = 0$$
$$\begin{pmatrix} 2 & 2i \\ -2i & 2 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0$$

From row 1: $2v_1 + 2iv_2 = 0 \implies v_1 = -iv_2$

Choose $v_2 = 1$: $|v_2⟩ = \begin{pmatrix} -i \\ 1 \end{pmatrix}$

Normalized: $|λ_2⟩ = \frac{1}{\sqrt{2}}\begin{pmatrix} -i \\ 1 \end{pmatrix}$

**Step 4: Verify orthogonality**
$$⟨λ_1|λ_2⟩ = \frac{1}{2}\begin{pmatrix} -i & 1 \end{pmatrix}\begin{pmatrix} -i \\ 1 \end{pmatrix} = \frac{1}{2}((-i)(-i) + 1) = \frac{1}{2}(-1 + 1) = 0 \quad ✓$$

**Results:**
$$\boxed{λ_1 = 3: |λ_1⟩ = \frac{1}{\sqrt{2}}\begin{pmatrix} i \\ 1 \end{pmatrix}, \quad λ_2 = -1: |λ_2⟩ = \frac{1}{\sqrt{2}}\begin{pmatrix} -i \\ 1 \end{pmatrix}}$$ ∎

---

## Practice Problems

### Level 1: Direct Application

1. Verify that the Pauli-Z matrix σ̂ᵤ = $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ is both Hermitian and unitary.

2. Show that the matrix $A = \begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix}$ is NOT Hermitian.

3. If Â is Hermitian, prove that Â² is also Hermitian.

### Level 2: Intermediate

4. Prove that if Û is unitary, then |det(Û)| = 1.
   *Hint: Use det(ÂB̂) = det(Â)det(B̂) and det(Â†) = det(Â)*.*

5. Show that the product of two unitary operators is unitary.

6. For the rotation operator $R_z(θ) = \begin{pmatrix} e^{-iθ/2} & 0 \\ 0 & e^{iθ/2} \end{pmatrix}$:
   - Verify it is unitary
   - Find its eigenvalues and verify |λ| = 1
   - Find the value of θ for which R_z(θ) = σ̂ᵤ

### Level 3: Challenging

7. Prove: If Û is unitary and Ĥ is Hermitian, then ÛĤÛ† is Hermitian.
   *Physical interpretation: This is a "rotated" observable.*

8. Show that for any Hermitian operator Ĥ, the operator Û = e^{iĤ} is unitary.
   *Hint: Use the fact that f(Ĥ)† = f*(Ĥ) for analytic f when Ĥ is Hermitian.*

9. **Research Problem:** The position operator x̂ on L²(ℝ) is Hermitian but unbounded. Explain why this complicates the spectral theorem and how physicists handle this (generalized eigenvectors, rigged Hilbert spaces).

---

## Computational Lab

```python
"""
Day 340 Computational Lab: Hermitian and Unitary Operators
Quantum Mechanics Core - Year 1
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

print("=" * 70)
print("Day 340: Hermitian and Unitary Operators")
print("=" * 70)

# =============================================================================
# Part 1: Adjoint (Hermitian Conjugate) Operations
# =============================================================================

print("\n--- Part 1: Adjoint Operations ---\n")

# Define a general complex matrix
A = np.array([[1 + 2j, 3 - 1j],
              [4j, 2 + 1j]], dtype=complex)

print("Matrix A:")
print(A)

# Compute adjoint (conjugate transpose)
A_dagger = A.conj().T
print("\nA† (Adjoint):")
print(A_dagger)

# Verify (AB)† = B†A†
B = np.array([[2, 1j],
              [-1j, 3]], dtype=complex)

AB = A @ B
AB_dagger = AB.conj().T
B_dagger_A_dagger = B.conj().T @ A.conj().T

print("\nVerifying (AB)† = B†A†:")
print(f"(AB)† =\n{AB_dagger}")
print(f"B†A† =\n{B_dagger_A_dagger}")
print(f"Equal: {np.allclose(AB_dagger, B_dagger_A_dagger)}")

# =============================================================================
# Part 2: Hermitian Operator Properties
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Hermitian Operator Properties")
print("=" * 70)

def is_hermitian(M, tol=1e-10):
    """Check if matrix is Hermitian."""
    return np.allclose(M, M.conj().T, atol=tol)

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

print("\nPauli Matrices - Hermiticity Check:")
print(f"σₓ is Hermitian: {is_hermitian(sigma_x)}")
print(f"σᵧ is Hermitian: {is_hermitian(sigma_y)}")
print(f"σᵤ is Hermitian: {is_hermitian(sigma_z)}")

# Custom Hermitian matrix from Example 3
H_custom = np.array([[1, 2j], [-2j, 1]], dtype=complex)
print(f"\nCustom H = [[1, 2i], [-2i, 1]] is Hermitian: {is_hermitian(H_custom)}")

# Eigenvalue decomposition
eigenvalues, eigenvectors = LA.eigh(H_custom)  # eigh for Hermitian matrices

print("\n--- Eigenvalue Analysis ---")
print(f"Eigenvalues: {eigenvalues}")
print(f"All eigenvalues real: {np.allclose(eigenvalues.imag, 0)}")

print("\nEigenvectors (columns):")
print(eigenvectors)

# Verify orthogonality
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]
inner_product = np.vdot(v1, v2)
print(f"\n⟨v₁|v₂⟩ = {inner_product:.6f}")
print(f"Orthogonal: {np.isclose(inner_product, 0)}")

# Verify spectral decomposition: H = λ₁|v₁⟩⟨v₁| + λ₂|v₂⟩⟨v₂|
P1 = np.outer(v1, v1.conj())
P2 = np.outer(v2, v2.conj())
H_reconstructed = eigenvalues[0] * P1 + eigenvalues[1] * P2

print("\nSpectral Decomposition H = Σ λₙ|vₙ⟩⟨vₙ|:")
print(f"Reconstructed H:\n{H_reconstructed}")
print(f"Original H:\n{H_custom}")
print(f"Match: {np.allclose(H_reconstructed, H_custom)}")

# =============================================================================
# Part 3: Unitary Operator Properties
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Unitary Operator Properties")
print("=" * 70)

def is_unitary(U, tol=1e-10):
    """Check if matrix is unitary."""
    n = U.shape[0]
    return np.allclose(U @ U.conj().T, np.eye(n), atol=tol)

# Hadamard gate
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase gate
S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)

# T gate
T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Rotation gates
def Rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def Ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def Rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

print("\nQuantum Gates - Unitarity Check:")
print(f"Hadamard H is unitary: {is_unitary(H_gate)}")
print(f"Phase S is unitary: {is_unitary(S_gate)}")
print(f"T gate is unitary: {is_unitary(T_gate)}")
print(f"Rₓ(π/4) is unitary: {is_unitary(Rx(np.pi/4))}")
print(f"Rᵧ(π/3) is unitary: {is_unitary(Ry(np.pi/3))}")
print(f"Rᵤ(π/2) is unitary: {is_unitary(Rz(np.pi/2))}")

# Verify eigenvalues have unit modulus
print("\n--- Unitary Eigenvalue Analysis ---")
U_test = Rz(np.pi/3)
eig_U, _ = LA.eig(U_test)
print(f"Rᵤ(π/3) eigenvalues: {eig_U}")
print(f"Moduli: {np.abs(eig_U)}")
print(f"All |λ| = 1: {np.allclose(np.abs(eig_U), 1)}")

# Verify inner product preservation
print("\n--- Inner Product Preservation ---")
psi = np.array([[1 + 1j], [2 - 1j]], dtype=complex)
phi = np.array([[1], [1j]], dtype=complex)

# Normalize
psi = psi / LA.norm(psi)
phi = phi / LA.norm(phi)

inner_before = np.vdot(psi, phi)
U_psi = H_gate @ psi
U_phi = H_gate @ phi
inner_after = np.vdot(U_psi, U_phi)

print(f"⟨φ|ψ⟩ before = {inner_before[0]:.6f}")
print(f"⟨Hφ|Hψ⟩ after = {inner_after[0]:.6f}")
print(f"Inner product preserved: {np.isclose(inner_before, inner_after)}")

# =============================================================================
# Part 4: Connection Between Hermitian and Unitary
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Hermitian-Unitary Connection")
print("=" * 70)

# For Hermitian H, exp(iH) is unitary
from scipy.linalg import expm

print("\nVerifying: If H is Hermitian, then U = exp(iH) is unitary")

# Use σ_y as Hermitian generator
theta = np.pi / 4
U_from_H = expm(1j * theta * sigma_y)

print(f"\nH = σᵧ (Hermitian: {is_hermitian(sigma_y)})")
print(f"U = exp(iσᵧ·π/4):\n{U_from_H}")
print(f"U is unitary: {is_unitary(U_from_H)}")

# This should equal Ry(2*theta) up to global phase
print(f"\nCompare with Rᵧ(π/2):")
print(f"Rᵧ(π/2) =\n{Ry(2*theta)}")

# =============================================================================
# Part 5: Visualization of Eigenvalue Structure
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Eigenvalue Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Hermitian eigenvalues (on real line)
ax1 = axes[0]
hermitian_matrices = {
    'σₓ': sigma_x,
    'σᵧ': sigma_y,
    'σᵤ': sigma_z,
    'H_custom': H_custom
}

colors = ['red', 'green', 'blue', 'purple']
y_offset = 0
for (name, H_mat), color in zip(hermitian_matrices.items(), colors):
    eigs, _ = LA.eigh(H_mat)
    ax1.scatter(eigs, [y_offset] * len(eigs), c=color, s=200, label=name, zorder=5)
    ax1.axhline(y=y_offset, color=color, alpha=0.3, linewidth=1)
    y_offset += 1

ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('Eigenvalue (Real)', fontsize=12)
ax1.set_ylabel('Operator', fontsize=12)
ax1.set_yticks(range(len(hermitian_matrices)))
ax1.set_yticklabels(hermitian_matrices.keys())
ax1.set_title('Hermitian Operators: Real Eigenvalues', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-4, 4)

# Plot 2: Unitary eigenvalues (on unit circle)
ax2 = axes[1]
theta_circle = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3, label='Unit circle')

unitary_matrices = {
    'H': H_gate,
    'S': S_gate,
    'T': T_gate,
    'Rᵤ(π/3)': Rz(np.pi/3),
    'Rₓ(π/4)': Rx(np.pi/4)
}

markers = ['o', 's', '^', 'D', 'v']
for (name, U_mat), marker in zip(unitary_matrices.items(), markers):
    eigs, _ = LA.eig(U_mat)
    ax2.scatter(eigs.real, eigs.imag, s=150, marker=marker, label=name, zorder=5)

ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax2.set_xlabel('Real Part', fontsize=12)
ax2.set_ylabel('Imaginary Part', fontsize=12)
ax2.set_title('Unitary Operators: Eigenvalues on Unit Circle', fontsize=14)
ax2.set_aspect('equal')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('day_340_eigenvalue_structure.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_340_eigenvalue_structure.png'")

# =============================================================================
# Part 6: Summary Statistics
# =============================================================================

print("\n" + "=" * 70)
print("Summary: Key Properties Verified")
print("=" * 70)

print("""
HERMITIAN OPERATORS (Ĥ† = Ĥ):
  ✓ Eigenvalues are real
  ✓ Eigenvectors with distinct eigenvalues are orthogonal
  ✓ Spectral decomposition: Ĥ = Σ λₙ|vₙ⟩⟨vₙ|
  ✓ Physical meaning: OBSERVABLES

UNITARY OPERATORS (Û†Û = I):
  ✓ Eigenvalues have |λ| = 1
  ✓ Inner products are preserved: ⟨Ûφ|Ûψ⟩ = ⟨φ|ψ⟩
  ✓ Norms are preserved (probability conservation)
  ✓ Physical meaning: TIME EVOLUTION and QUANTUM GATES

CONNECTION:
  ✓ If Ĥ is Hermitian, then Û = exp(iĤθ) is unitary
  ✓ Quantum gates are generated by Hermitian operators
""")

print("=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Adjoint definition | $⟨φ|\hat{A}^†|ψ⟩ = ⟨ψ|\hat{A}|φ⟩^*$ |
| Matrix adjoint | $(A^†)_{mn} = A_{nm}^*$ |
| Hermitian condition | $\hat{H}^† = \hat{H}$ |
| Hermitian eigenvalues | $λ ∈ ℝ$ (always real) |
| Eigenvector orthogonality | $⟨λ_1|λ_2⟩ = 0$ for $λ_1 ≠ λ_2$ |
| Spectral theorem | $\hat{H} = \sum_n λ_n |λ_n⟩⟨λ_n|$ |
| Unitary condition | $\hat{U}^†\hat{U} = \hat{U}\hat{U}^† = \hat{I}$ |
| Unitary eigenvalues | $|λ| = 1$ (unit modulus) |
| Inner product preservation | $⟨\hat{U}φ|\hat{U}ψ⟩ = ⟨φ|ψ⟩$ |
| Hermitian → Unitary | $\hat{U} = e^{i\hat{H}θ}$ |

### Physical Significance

$$\boxed{\text{Hermitian operators} \leftrightarrow \text{Physical observables}}$$
$$\boxed{\text{Unitary operators} \leftrightarrow \text{Time evolution / Gates}}$$

### Main Takeaways

1. **Adjoint = conjugate transpose** for matrices; defined abstractly via inner products
2. **Hermitian operators have real eigenvalues** — essential for measurable quantities
3. **Eigenvectors of Hermitian operators are orthogonal** — enables basis expansion
4. **Unitary operators preserve inner products** — probability conservation
5. **Eigenvalues of unitary operators lie on the unit circle** — |λ| = 1
6. **Exponentiating a Hermitian gives a unitary** — generators of evolution

---

## Daily Checklist

- [ ] Read Shankar Chapter 1.8 (Hermitian and Unitary Operators)
- [ ] Read Sakurai Chapter 1.4 (Matrix Representations, Eigenvalue Problem)
- [ ] Prove the product rule: (ÂB̂)† = B̂†Â†
- [ ] Verify Hermiticity and find eigenvalues for all three Pauli matrices
- [ ] Verify the Hadamard gate is unitary and find its eigenvalues
- [ ] Complete Level 1-2 practice problems
- [ ] Run the computational lab and study the eigenvalue plots
- [ ] Write down the spectral theorem from memory

---

## Preview: Day 341

Tomorrow we dive deeper into **eigenvalue problems**: degenerate eigenspaces, simultaneous diagonalization of commuting operators, and the physical interpretation of compatible observables. We'll see why [Â, B̂] = 0 is so important in quantum mechanics.

---

*"The Hermitian operators are Nature's way of ensuring that measurements yield real numbers. The unitary operators are Nature's way of ensuring that probability is conserved."*

---

**References:**
- Shankar, R. *Principles of Quantum Mechanics*, Chapter 1.8
- Sakurai, J.J. *Modern Quantum Mechanics*, Chapter 1.4
- Nielsen & Chuang, *Quantum Computation and Quantum Information*, Chapter 2

---

**Next:** [Day_341_Friday.md](Day_341_Friday.md) — Eigenvalue Problems
