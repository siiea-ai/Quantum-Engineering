# Day 506: Properties and Trace

## Overview

**Day 506** | Week 73, Day 2 | Year 1, Month 19 | Fundamental Properties of Density Matrices

Today we establish the three defining properties of density matrices and develop the trace formalism for computing expectation values and probabilities.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Properties and proofs |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Trace calculations |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. State and prove the three defining properties of density matrices
2. Show that any operator satisfying these properties is a valid density matrix
3. Use the cyclic property of trace for calculations
4. Compute expectation values using Tr(ρA)
5. Calculate measurement probabilities using Tr(Πρ)
6. Derive the eigenvalue constraints from the properties

---

## Core Content

### The Three Defining Properties

A valid density matrix ρ must satisfy:

$$\boxed{\text{1. Hermitian: } \rho^\dagger = \rho}$$

$$\boxed{\text{2. Positive semidefinite: } \rho \geq 0 \text{ (all eigenvalues } \geq 0\text{)}}$$

$$\boxed{\text{3. Normalized: } \text{Tr}(\rho) = 1}$$

**Theorem:** An operator is a valid density matrix if and only if it satisfies these three properties.

### Property 1: Hermiticity

**Definition:** ρ is Hermitian if ρ† = ρ, meaning ρᵢⱼ = ρⱼᵢ*.

**Physical meaning:** Hermitian operators have real eigenvalues, corresponding to real probabilities.

**Proof for pure states:** For ρ = |ψ⟩⟨ψ|:
$$\rho^\dagger = (|\psi\rangle\langle\psi|)^\dagger = |\psi\rangle\langle\psi| = \rho$$

**Proof for mixed states:** If each term pᵢ|ψᵢ⟩⟨ψᵢ| is Hermitian and pᵢ are real, the sum is Hermitian.

### Property 2: Positive Semidefiniteness

**Definition:** ρ ≥ 0 means ⟨φ|ρ|φ⟩ ≥ 0 for all |φ⟩.

**Equivalent conditions:**
1. All eigenvalues of ρ are non-negative
2. ρ can be written as ρ = A†A for some operator A
3. ⟨φ|ρ|φ⟩ ≥ 0 for all |φ⟩

**Physical meaning:** Probabilities must be non-negative.

**Proof for pure states:** For ρ = |ψ⟩⟨ψ|:
$$\langle\phi|\rho|\phi\rangle = \langle\phi|\psi\rangle\langle\psi|\phi\rangle = |\langle\phi|\psi\rangle|^2 \geq 0$$

**Proof for mixed states:** If each pᵢ ≥ 0 and |ψᵢ⟩⟨ψᵢ| ≥ 0:
$$\langle\phi|\rho|\phi\rangle = \sum_i p_i |\langle\phi|\psi_i\rangle|^2 \geq 0$$

### Property 3: Normalization

**Definition:** Tr(ρ) = 1.

**Physical meaning:** Total probability equals 1.

**Proof for pure states:**
$$\text{Tr}(|\psi\rangle\langle\psi|) = \sum_i \langle i|\psi\rangle\langle\psi|i\rangle = \langle\psi|\left(\sum_i |i\rangle\langle i|\right)|\psi\rangle = \langle\psi|\psi\rangle = 1$$

**Proof for mixed states:**
$$\text{Tr}(\rho) = \sum_i p_i \text{Tr}(|\psi_i\rangle\langle\psi_i|) = \sum_i p_i \cdot 1 = 1$$

### Properties of the Trace

The trace operation has special properties crucial for quantum mechanics:

**Cyclic property:**
$$\text{Tr}(ABC) = \text{Tr}(BCA) = \text{Tr}(CAB)$$

**Linearity:**
$$\text{Tr}(\alpha A + \beta B) = \alpha \text{Tr}(A) + \beta \text{Tr}(B)$$

**Outer product:**
$$\text{Tr}(|\psi\rangle\langle\phi|) = \langle\phi|\psi\rangle$$

**Basis independence:** Tr(A) = Σᵢ⟨eᵢ|A|eᵢ⟩ for any orthonormal basis.

### Trace Formula for Expectation Values

For an observable A and state ρ:

$$\boxed{\langle A \rangle = \text{Tr}(\rho A) = \text{Tr}(A\rho)}$$

**Derivation for pure state:** If ρ = |ψ⟩⟨ψ|:
$$\text{Tr}(\rho A) = \text{Tr}(|\psi\rangle\langle\psi|A) = \langle\psi|A|\psi\rangle = \langle A \rangle$$

**For mixed state:**
$$\text{Tr}(\rho A) = \sum_i p_i \text{Tr}(|\psi_i\rangle\langle\psi_i|A) = \sum_i p_i \langle\psi_i|A|\psi_i\rangle$$

This is the weighted average of expectation values—exactly what we expect for a statistical mixture.

### Measurement Probabilities

For a projective measurement with projector Πₘ = |m⟩⟨m|:

$$\boxed{p(m) = \text{Tr}(\Pi_m \rho) = \text{Tr}(\rho \Pi_m)}$$

**Verification:**
$$\text{Tr}(\Pi_m \rho) = \text{Tr}(|m\rangle\langle m|\rho) = \langle m|\rho|m\rangle = \rho_{mm}$$

The probability equals the diagonal element in the measurement basis.

### Eigenvalue Constraints

From the three properties, we can derive constraints on eigenvalues.

Let ρ = Σᵢ λᵢ|eᵢ⟩⟨eᵢ| be the spectral decomposition.

1. **From Hermiticity:** λᵢ are real
2. **From positive semidefiniteness:** λᵢ ≥ 0
3. **From normalization:** Σᵢ λᵢ = 1

Therefore, eigenvalues form a **probability distribution** over orthonormal states.

---

## Quantum Computing Connection

### State Tomography

In quantum computing, we reconstruct ρ by measuring expectation values:

For a qubit: ρ = ½(I + ⟨X⟩X + ⟨Y⟩Y + ⟨Z⟩Z)

We measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩ using Tr(ρX), Tr(ρY), Tr(ρZ).

### Noise Characterization

The trace formulas help quantify gate errors:
- **Fidelity:** F = Tr(ρ_ideal ρ_actual)
- **Error rate:** ε = 1 - F

### Quantum Channels

Trace preservation ensures probability conservation in noisy evolution.

---

## Worked Examples

### Example 1: Computing Expectation Values

**Problem:** For ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1|, compute ⟨Z⟩ and ⟨X⟩.

**Solution:**

$$\rho = \begin{pmatrix} 3/4 & 0 \\ 0 & 1/4 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \quad X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

For ⟨Z⟩:
$$\langle Z \rangle = \text{Tr}(\rho Z) = \text{Tr}\begin{pmatrix} 3/4 & 0 \\ 0 & -1/4 \end{pmatrix} = \frac{3}{4} - \frac{1}{4} = \frac{1}{2}$$

For ⟨X⟩:
$$\rho X = \begin{pmatrix} 3/4 & 0 \\ 0 & 1/4 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 3/4 \\ 1/4 & 0 \end{pmatrix}$$

$$\langle X \rangle = \text{Tr}(\rho X) = 0 + 0 = 0$$

### Example 2: Measurement Probabilities

**Problem:** A qubit is in state ρ = ½|+⟩⟨+| + ½|−⟩⟨−|. Find p(0) and p(1) when measuring in the computational basis.

**Solution:**

First, compute ρ:
$$\rho = \frac{1}{2} \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} + \frac{1}{2} \cdot \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

Projectors: Π₀ = |0⟩⟨0|, Π₁ = |1⟩⟨1|

$$p(0) = \text{Tr}(\Pi_0 \rho) = \text{Tr}\left(\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \cdot \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\right) = \text{Tr}\begin{pmatrix} 1/2 & 0 \\ 0 & 0 \end{pmatrix} = \frac{1}{2}$$

Similarly, p(1) = ½.

### Example 3: Verifying Properties

**Problem:** Verify that ρ = (begin matrix) 0.6, 0.2-0.1i; 0.2+0.1i, 0.4 (end matrix) is a valid density matrix.

**Solution:**

**Hermiticity:** ρ₁₂ = 0.2 - 0.1i, ρ₂₁* = (0.2 + 0.1i)* = 0.2 - 0.1i ✓

**Trace:** Tr(ρ) = 0.6 + 0.4 = 1 ✓

**Positive semidefinite:** Find eigenvalues.
$$\det(\rho - \lambda I) = (0.6-\lambda)(0.4-\lambda) - |0.2-0.1i|^2 = 0$$
$$\lambda^2 - \lambda + 0.24 - 0.05 = \lambda^2 - \lambda + 0.19 = 0$$
$$\lambda = \frac{1 \pm \sqrt{1-0.76}}{2} = \frac{1 \pm 0.49}{2}$$

λ₁ = 0.745, λ₂ = 0.255. Both positive ✓

ρ is a valid density matrix.

---

## Practice Problems

### Direct Application

**Problem 1:** Compute ⟨Y⟩ for ρ = |+⟩⟨+|.

**Problem 2:** For ρ = ⅔|0⟩⟨0| + ⅓|+⟩⟨+|, find p(+) when measuring in the {|+⟩, |−⟩} basis.

**Problem 3:** Verify Tr(ABC) = Tr(CAB) for 2×2 matrices of your choice.

### Intermediate

**Problem 4:** Prove that if ρ is a density matrix and U is unitary, then UρU† is also a density matrix.

**Problem 5:** Show that eigenvalues of a density matrix must satisfy 0 ≤ λᵢ ≤ 1.

**Problem 6:** For a qutrit (3-level system), what is the minimum purity Tr(ρ²)?

### Challenging

**Problem 7:** Prove: Tr(ρA) is real if A is Hermitian and ρ is a density matrix.

**Problem 8:** Show that Tr(ρ²) ≤ 1 with equality iff ρ is pure.

**Problem 9:** Given that ⟨X⟩ = 0.5, ⟨Y⟩ = 0, ⟨Z⟩ = 0.5, reconstruct ρ and verify it's a valid density matrix.

---

## Computational Lab

```python
"""
Day 506: Properties and Trace
Verifying density matrix properties and trace calculations
"""

import numpy as np
import matplotlib.pyplot as plt

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def is_hermitian(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is Hermitian"""
    return np.allclose(A, A.conj().T, atol=tol)

def is_positive_semidefinite(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is positive semidefinite"""
    eigenvalues = np.linalg.eigvalsh(A)
    return all(eigenvalues >= -tol)

def is_normalized(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if trace equals 1"""
    return np.isclose(np.trace(A), 1, atol=tol)

def is_valid_density_matrix(rho: np.ndarray) -> bool:
    """Check all three properties"""
    return is_hermitian(rho) and is_positive_semidefinite(rho) and is_normalized(rho)

def expectation_value(rho: np.ndarray, A: np.ndarray) -> complex:
    """Compute ⟨A⟩ = Tr(ρA)"""
    return np.trace(rho @ A)

def measurement_probability(rho: np.ndarray, projector: np.ndarray) -> float:
    """Compute p = Tr(Πρ)"""
    return np.trace(projector @ rho).real

print("=" * 60)
print("DENSITY MATRIX PROPERTIES VERIFICATION")
print("=" * 60)

# Example 1: Valid density matrix
rho1 = np.array([[0.75, 0], [0, 0.25]], dtype=complex)
print(f"\nρ₁ = diag(0.75, 0.25)")
print(f"Hermitian: {is_hermitian(rho1)}")
print(f"Positive semidefinite: {is_positive_semidefinite(rho1)}")
print(f"Normalized: {is_normalized(rho1)}")
print(f"Valid density matrix: {is_valid_density_matrix(rho1)}")

# Example 2: Matrix with off-diagonal elements
rho2 = np.array([[0.6, 0.2-0.1j], [0.2+0.1j, 0.4]], dtype=complex)
print(f"\nρ₂ with off-diagonal coherences")
print(f"Hermitian: {is_hermitian(rho2)}")
print(f"Positive semidefinite: {is_positive_semidefinite(rho2)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(rho2)}")
print(f"Normalized: {is_normalized(rho2)}")
print(f"Valid density matrix: {is_valid_density_matrix(rho2)}")

# Example 3: Invalid matrix (not positive)
rho_invalid = np.array([[1.5, 0], [0, -0.5]], dtype=complex)
print(f"\nρ_invalid = diag(1.5, -0.5)")
print(f"Hermitian: {is_hermitian(rho_invalid)}")
print(f"Positive semidefinite: {is_positive_semidefinite(rho_invalid)}")
print(f"Normalized: {is_normalized(rho_invalid)}")
print(f"Valid density matrix: {is_valid_density_matrix(rho_invalid)}")

print("\n" + "=" * 60)
print("TRACE CALCULATIONS")
print("=" * 60)

# Verify cyclic property
A = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
B = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
C = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)

tr_ABC = np.trace(A @ B @ C)
tr_BCA = np.trace(B @ C @ A)
tr_CAB = np.trace(C @ A @ B)

print(f"\nCyclic property verification:")
print(f"Tr(ABC) = {tr_ABC:.6f}")
print(f"Tr(BCA) = {tr_BCA:.6f}")
print(f"Tr(CAB) = {tr_CAB:.6f}")
print(f"All equal: {np.allclose(tr_ABC, tr_BCA) and np.allclose(tr_BCA, tr_CAB)}")

print("\n" + "=" * 60)
print("EXPECTATION VALUES")
print("=" * 60)

# Mixed state: 3/4 |0⟩ + 1/4 |1⟩
rho = np.array([[0.75, 0], [0, 0.25]], dtype=complex)

exp_X = expectation_value(rho, X)
exp_Y = expectation_value(rho, Y)
exp_Z = expectation_value(rho, Z)

print(f"\nFor ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1|:")
print(f"⟨X⟩ = {exp_X.real:.4f}")
print(f"⟨Y⟩ = {exp_Y.real:.4f}")
print(f"⟨Z⟩ = {exp_Z.real:.4f}")

# Verify: ⟨Z⟩ should be 3/4 - 1/4 = 1/2
print(f"\nExpected ⟨Z⟩ = 3/4 × (+1) + 1/4 × (-1) = 0.5 ✓")

print("\n" + "=" * 60)
print("MEASUREMENT PROBABILITIES")
print("=" * 60)

# Projectors
Pi_0 = np.array([[1, 0], [0, 0]], dtype=complex)
Pi_1 = np.array([[0, 0], [0, 1]], dtype=complex)
Pi_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
Pi_minus = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)

# State: |+⟩
rho_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)

print(f"\nFor |+⟩ state:")
print(f"p(0) in computational basis: {measurement_probability(rho_plus, Pi_0):.4f}")
print(f"p(1) in computational basis: {measurement_probability(rho_plus, Pi_1):.4f}")
print(f"p(+) in ±basis: {measurement_probability(rho_plus, Pi_plus):.4f}")
print(f"p(-) in ±basis: {measurement_probability(rho_plus, Pi_minus):.4f}")

# Visualization: Eigenvalue constraints
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Valid eigenvalue region for 2D
ax = axes[0]
lambda1 = np.linspace(0, 1, 100)
lambda2 = 1 - lambda1

valid = (lambda2 >= 0) & (lambda2 <= 1)
ax.fill_between(lambda1[valid], 0, lambda2[valid], alpha=0.3, color='blue')
ax.plot(lambda1[valid], lambda2[valid], 'b-', lw=2, label='Tr(ρ) = 1')
ax.plot([0, 1], [0, 0], 'b-', lw=2)
ax.plot([0, 0], [0, 1], 'b-', lw=2)
ax.scatter([1, 0, 0.5], [0, 1, 0.5], c=['red', 'red', 'green'],
           s=100, zorder=5, label='Pure/Mixed states')
ax.set_xlabel('λ₁')
ax.set_ylabel('λ₂')
ax.set_title('Valid Eigenvalue Region (2D)')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.grid(True, alpha=0.3)
ax.legend()
ax.annotate('Pure', (1, 0), xytext=(0.85, 0.15), fontsize=10)
ax.annotate('Pure', (0, 1), xytext=(0.1, 0.85), fontsize=10)
ax.annotate('Maximally\nmixed', (0.5, 0.5), xytext=(0.55, 0.6), fontsize=10)

# Right plot: Purity as function of eigenvalue split
ax = axes[1]
p = np.linspace(0, 1, 100)  # probability for first eigenvalue
purity = p**2 + (1-p)**2

ax.plot(p, purity, 'b-', lw=2)
ax.fill_between(p, 0.5, purity, alpha=0.3, color='blue')
ax.axhline(1, color='red', ls='--', label='Pure state (γ=1)')
ax.axhline(0.5, color='green', ls='--', label='Maximally mixed (γ=1/d)')
ax.set_xlabel('Eigenvalue λ₁ (with λ₂ = 1-λ₁)')
ax.set_ylabel('Purity γ = Tr(ρ²)')
ax.set_title('Purity vs Eigenvalue Distribution')
ax.set_xlim(0, 1)
ax.set_ylim(0.4, 1.1)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('density_matrix_properties.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Day 506 Complete: Properties and Trace")
print("=" * 60)
```

---

## Summary

### Three Defining Properties

| Property | Mathematical Statement | Physical Meaning |
|----------|----------------------|------------------|
| Hermitian | ρ† = ρ | Real probabilities |
| Positive semidefinite | ρ ≥ 0 | Non-negative probabilities |
| Normalized | Tr(ρ) = 1 | Total probability = 1 |

### Key Trace Formulas

| Formula | Application |
|---------|-------------|
| ⟨A⟩ = Tr(ρA) | Expectation values |
| p(m) = Tr(Πₘρ) | Measurement probabilities |
| Tr(ABC) = Tr(CAB) | Cyclic property |

### Eigenvalue Constraints

- Eigenvalues λᵢ are real, non-negative
- Sum to 1: Σλᵢ = 1
- Form a probability distribution

---

## Daily Checklist

- [ ] I can state the three defining properties of density matrices
- [ ] I can prove each property for pure and mixed states
- [ ] I can compute expectation values using Tr(ρA)
- [ ] I can calculate measurement probabilities using Tr(Πρ)
- [ ] I understand the eigenvalue constraints
- [ ] I can verify if a given matrix is a valid density matrix

---

## Preview: Day 507

Tomorrow we focus on **expectation values and measurement statistics**, exploring:
- Complete derivation of the trace formula for observables
- Variance calculations: ⟨A²⟩ - ⟨A⟩²
- Post-measurement states
- The measurement update rule for density matrices

---

*Next: Day 507 — Expectation Values*
