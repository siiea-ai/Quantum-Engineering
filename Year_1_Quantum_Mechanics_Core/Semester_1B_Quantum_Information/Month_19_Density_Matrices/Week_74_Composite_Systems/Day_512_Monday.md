# Day 512: Tensor Products Revisited

## Overview

**Day 512** | Week 74, Day 1 | Year 1, Month 19 | Composite Quantum Systems

Today we revisit tensor products with a focus on density matrices for composite systems. Understanding tensor product structure is essential for describing multi-qubit quantum computers and entangled states.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Tensor product theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. Construct the tensor product Hilbert space for composite systems
2. Express product states and general states in tensor product notation
3. Write density matrices for composite systems
4. Compute tensor products of operators
5. Identify product vs entangled states
6. Apply tensor product formalism to multi-qubit systems

---

## Core Content

### Tensor Product of Hilbert Spaces

For two quantum systems A and B with Hilbert spaces ℋ_A and ℋ_B:

$$\boxed{\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B}$$

**Dimension:**
$$\dim(\mathcal{H}_{AB}) = \dim(\mathcal{H}_A) \times \dim(\mathcal{H}_B) = d_A \cdot d_B$$

### Basis for Composite Space

If {|i⟩_A} is a basis for ℋ_A and {|j⟩_B} is a basis for ℋ_B:

$$\{|i\rangle_A \otimes |j\rangle_B\} = \{|i,j\rangle\} = \{|ij\rangle\}$$

forms a basis for ℋ_AB.

**Example: Two qubits**
$$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$

### Product States

A **product state** (separable) has the form:
$$|\psi\rangle_{AB} = |\phi\rangle_A \otimes |\chi\rangle_B$$

**Example:**
$$|+\rangle \otimes |0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

### General States

Most states are NOT product states:
$$|\psi\rangle_{AB} = \sum_{i,j} c_{ij} |i\rangle_A \otimes |j\rangle_B$$

**Entangled state example (Bell state):**
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

This cannot be written as |φ⟩_A ⊗ |χ⟩_B for any states |φ⟩, |χ⟩.

### Tensor Product of Operators

For operators A on ℋ_A and B on ℋ_B:

$$(A \otimes B)|i,j\rangle = (A|i\rangle) \otimes (B|j\rangle)$$

**Matrix representation:** The Kronecker product
$$(A \otimes B)_{(ik),(jl)} = A_{ij} B_{kl}$$

### Local Operations

Operations on subsystem A only:
$$A \otimes I_B$$

**Example:** X gate on first qubit
$$X \otimes I = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$$

### Density Matrices for Composite Systems

**Product state density matrix:**
$$\rho_{AB} = \rho_A \otimes \rho_B$$

**General composite density matrix:**
$$\rho_{AB} = \sum_{ijkl} \rho_{ij,kl} |i\rangle\langle k|_A \otimes |j\rangle\langle l|_B$$

### Expectation Values

For observable A ⊗ B:
$$\langle A \otimes B \rangle = \text{Tr}(\rho_{AB}(A \otimes B))$$

For product states:
$$\langle A \otimes B \rangle = \langle A \rangle_A \cdot \langle B \rangle_B$$

---

## Quantum Computing Connection

### Multi-Qubit Systems

An n-qubit system lives in:
$$\mathcal{H} = (\mathbb{C}^2)^{\otimes n} = \underbrace{\mathbb{C}^2 \otimes \mathbb{C}^2 \otimes \cdots \otimes \mathbb{C}^2}_{n \text{ times}}$$

Dimension: 2ⁿ (exponential!)

### Two-Qubit Gates

**CNOT:**
$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$

**CZ:**
$$\text{CZ} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes Z$$

### Creating Entanglement

Starting from |00⟩:
1. Apply H to first qubit: (H⊗I)|00⟩ = |+0⟩ = ½(|00⟩ + |10⟩)
2. Apply CNOT: |Φ⁺⟩ = ½(|00⟩ + |11⟩)

---

## Worked Examples

### Example 1: Tensor Product of States

**Problem:** Compute |+⟩ ⊗ |−⟩ in the computational basis.

**Solution:**

$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

$$|+\rangle \otimes |-\rangle = \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle - |1\rangle)$$
$$= \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$$

### Example 2: Tensor Product of Operators

**Problem:** Compute Z ⊗ X.

**Solution:**

$$Z \otimes X = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \otimes \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

Using Kronecker product:
$$= \begin{pmatrix} 1 \cdot X & 0 \cdot X \\ 0 \cdot X & -1 \cdot X \end{pmatrix} = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & -1 \\ 0 & 0 & -1 & 0 \end{pmatrix}$$

### Example 3: Product State Density Matrix

**Problem:** Write the density matrix for |+⟩⟨+| ⊗ |0⟩⟨0|.

**Solution:**

$$|+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}, \quad |0\rangle\langle 0| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

$$\rho = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

---

## Practice Problems

### Direct Application

**Problem 1:** Compute |0⟩ ⊗ |1⟩ ⊗ |+⟩ for a 3-qubit system.

**Problem 2:** Calculate (H ⊗ H)|00⟩.

**Problem 3:** Write the 4×4 matrix for I ⊗ Z.

### Intermediate

**Problem 4:** Show that (A⊗B)(C⊗D) = (AC)⊗(BD).

**Problem 5:** Verify that CNOT|+0⟩ = |Φ⁺⟩.

**Problem 6:** For ρ = |+⟩⟨+| ⊗ |+⟩⟨+|, compute ⟨Z⊗Z⟩.

### Challenging

**Problem 7:** Prove that |Φ⁺⟩ cannot be written as a product state.

**Problem 8:** Show that Tr(ρ_A ⊗ ρ_B) = Tr(ρ_A)·Tr(ρ_B).

**Problem 9:** For n qubits, how many real parameters specify a general pure state?

---

## Computational Lab

```python
"""
Day 512: Tensor Products Revisited
Composite quantum systems
"""

import numpy as np
import matplotlib.pyplot as plt

def tensor(A, B):
    """Compute tensor product A ⊗ B (Kronecker product)"""
    return np.kron(A, B)

# Standard states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# CNOT
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

print("=" * 60)
print("TENSOR PRODUCTS OF STATES")
print("=" * 60)

# Two-qubit computational basis
ket_00 = tensor(ket_0, ket_0)
ket_01 = tensor(ket_0, ket_1)
ket_10 = tensor(ket_1, ket_0)
ket_11 = tensor(ket_1, ket_1)

print("\nComputational basis states:")
print(f"|00⟩ = {ket_00.flatten()}")
print(f"|01⟩ = {ket_01.flatten()}")
print(f"|10⟩ = {ket_10.flatten()}")
print(f"|11⟩ = {ket_11.flatten()}")

# Product state
psi_product = tensor(ket_plus, ket_0)
print(f"\n|+⟩ ⊗ |0⟩ = {psi_product.flatten()}")

# Bell state
phi_plus = (ket_00 + ket_11) / np.sqrt(2)
print(f"\n|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 = {phi_plus.flatten()}")

print("\n" + "=" * 60)
print("TENSOR PRODUCTS OF OPERATORS")
print("=" * 60)

# Local operations
X_I = tensor(X, I)
I_X = tensor(I, X)
Z_Z = tensor(Z, Z)

print("\nX ⊗ I:")
print(X_I)

print("\nZ ⊗ Z:")
print(Z_Z)

print("\n" + "=" * 60)
print("CREATING ENTANGLEMENT")
print("=" * 60)

# Start with |00⟩
psi = ket_00
print(f"Initial |00⟩: {psi.flatten()}")

# Apply H ⊗ I
H_I = tensor(H, I)
psi = H_I @ psi
print(f"After H⊗I: {psi.flatten()}")

# Apply CNOT
psi = CNOT @ psi
print(f"After CNOT: {psi.flatten()}")
print("This is |Φ⁺⟩!")

# Verify it's a Bell state
print(f"\nMatch with |Φ⁺⟩: {np.allclose(psi, phi_plus)}")

print("\n" + "=" * 60)
print("EXPECTATION VALUES")
print("=" * 60)

def density(psi):
    return psi @ psi.conj().T

# For product state
rho_product = density(psi_product)
exp_ZZ = np.trace(rho_product @ Z_Z).real
print(f"\n|+0⟩: ⟨Z⊗Z⟩ = {exp_ZZ:.4f}")

# For Bell state
rho_bell = density(phi_plus)
exp_ZZ_bell = np.trace(rho_bell @ Z_Z).real
print(f"|Φ⁺⟩: ⟨Z⊗Z⟩ = {exp_ZZ_bell:.4f}")

# Correlation
exp_Z_A = np.trace(rho_bell @ tensor(Z, I)).real
exp_Z_B = np.trace(rho_bell @ tensor(I, Z)).real
print(f"|Φ⁺⟩: ⟨Z⊗I⟩ = {exp_Z_A:.4f}, ⟨I⊗Z⟩ = {exp_Z_B:.4f}")
print(f"Correlation: ⟨ZZ⟩ - ⟨Z⟩⟨Z⟩ = {exp_ZZ_bell - exp_Z_A*exp_Z_B:.4f}")

print("\n" + "=" * 60)
print("Day 512 Complete: Tensor Products")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| ℋ_AB = ℋ_A ⊗ ℋ_B | Composite Hilbert space |
| dim(ℋ_AB) = d_A · d_B | Dimension multiplies |
| (A⊗B)\|i,j⟩ = (A\|i⟩)⊗(B\|j⟩) | Operator action |
| (A⊗B)(C⊗D) = (AC)⊗(BD) | Mixed product rule |

### Key Concepts

- **Product states** can be factored: |ψ⟩_AB = |φ⟩_A ⊗ |χ⟩_B
- **Entangled states** cannot be factored
- **Local operations** act as A ⊗ I or I ⊗ B
- **Multi-qubit systems** have dimension 2ⁿ

---

## Daily Checklist

- [ ] I can construct tensor products of states
- [ ] I can compute Kronecker products of matrices
- [ ] I understand product vs entangled states
- [ ] I can apply local operations
- [ ] I can compute expectation values for composite systems

---

## Preview: Day 513

Tomorrow we introduce the **partial trace**—the key operation for describing subsystems of composite quantum systems. This connects to entanglement: tracing out part of an entangled system gives a mixed state!

---

*Next: Day 513 — Partial Trace*
