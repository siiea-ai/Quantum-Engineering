# Day 402: Pauli Matrices

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Pauli matrices |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 402, you will be able to:

1. Write the Pauli matrices σₓ, σᵧ, σᵤ from memory
2. Verify their fundamental algebraic properties
3. Express spin operators in terms of Pauli matrices
4. Recognize Pauli matrices as quantum gates (X, Y, Z)
5. Use Pauli matrices to represent any 2×2 Hermitian matrix

---

## Core Content

### 1. The Pauli Matrices

$$\boxed{\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}}$$

Also denoted σ₁, σ₂, σ₃ or X, Y, Z in quantum computing.

### 2. Basic Properties

**Hermitian:** σᵢ† = σᵢ

**Unitary:** σᵢ†σᵢ = I

**Traceless:** Tr(σᵢ) = 0

**Determinant:** det(σᵢ) = -1

**Eigenvalues:** ±1 for all three matrices

### 3. Algebraic Relations

**Square to identity:**
$$\boxed{\sigma_x^2 = \sigma_y^2 = \sigma_z^2 = I}$$

**Anticommutation:**
$$\boxed{\{\sigma_i, \sigma_j\} = \sigma_i\sigma_j + \sigma_j\sigma_i = 2\delta_{ij}I}$$

**Commutation:**
$$\boxed{[\sigma_i, \sigma_j] = 2i\varepsilon_{ijk}\sigma_k}$$

**Product formula:**
$$\boxed{\sigma_i\sigma_j = \delta_{ij}I + i\varepsilon_{ijk}\sigma_k}$$

### 4. Spin Operators

The spin angular momentum operators are:
$$\boxed{\hat{S}_i = \frac{\hbar}{2}\sigma_i}$$

The commutation relations follow:
$$[\hat{S}_i, \hat{S}_j] = \frac{\hbar^2}{4}[\sigma_i, \sigma_j] = \frac{\hbar^2}{4}(2i\varepsilon_{ijk}\sigma_k) = i\hbar\varepsilon_{ijk}\hat{S}_k$$

Which matches the general angular momentum algebra!

### 5. Completeness: Any 2×2 Matrix

Any 2×2 Hermitian matrix can be written:
$$\boxed{H = h_0 I + h_1\sigma_x + h_2\sigma_y + h_3\sigma_z = h_0 I + \mathbf{h}\cdot\boldsymbol{\sigma}}$$

where h₀, h₁, h₂, h₃ are real and **σ** = (σₓ, σᵧ, σᵤ).

### 6. Eigenstates of Pauli Matrices

**σᵤ eigenstates:**
$$\sigma_z|\uparrow\rangle = +|\uparrow\rangle, \quad \sigma_z|\downarrow\rangle = -|\downarrow\rangle$$

**σₓ eigenstates:**
$$|+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad |-\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

$$\sigma_x|+\rangle = +|+\rangle, \quad \sigma_x|-\rangle = -|-\rangle$$

**σᵧ eigenstates:**
$$|+i\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}, \quad |-i\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix}$$

---

## Quantum Computing Connection

| Pauli Matrix | Quantum Gate | Action |
|--------------|--------------|--------|
| σₓ = X | Pauli-X (NOT) | Bit flip: \|0⟩↔\|1⟩ |
| σᵧ = Y | Pauli-Y | Bit + phase flip |
| σᵤ = Z | Pauli-Z | Phase flip: \|1⟩→-\|1⟩ |

**Pauli group:** {±I, ±iI, ±X, ±iX, ±Y, ±iY, ±Z, ±iZ}

This group is fundamental to quantum error correction!

---

## Worked Examples

### Example 1: Verify σₓσᵧ = iσᵤ

**Solution:**
$$\sigma_x\sigma_y = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} = \begin{pmatrix} i & 0 \\ 0 & -i \end{pmatrix} = i\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = i\sigma_z \checkmark$$

### Example 2: Find σₓ Eigenstates

**Problem:** Find the eigenstates of σₓ.

**Solution:**
$$\sigma_x|\psi\rangle = \lambda|\psi\rangle$$

Characteristic equation: det(σₓ - λI) = -λ² + 1 = 0 → λ = ±1

For λ = +1:
$$(σ_x - I)|\psi\rangle = 0 \Rightarrow \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} a \\ b \end{pmatrix} = 0$$

This gives a = b, so |+⟩ = (1,1)ᵀ/√2

For λ = -1: a = -b, so |-⟩ = (1,-1)ᵀ/√2

### Example 3: Express Matrix Using Paulis

**Problem:** Write H = [[2, 1-i], [1+i, 0]] in terms of I, σₓ, σᵧ, σᵤ.

**Solution:**
Using the formulas:
- h₀ = Tr(H)/2 = (2+0)/2 = 1
- h₁ = Tr(Hσₓ)/2 = Tr([[0,2], [2+2i,1-i]])/2 = (0+1-i)/2... (continue calculation)

---

## Practice Problems

### Direct Application

1. Verify σᵧσᵤ = iσₓ and σᵤσₓ = iσᵧ.

2. Calculate σₓσᵧσᵤ.

3. Show that Tr(σᵢσⱼ) = 2δᵢⱼ.

### Intermediate

4. Prove {σᵢ, σⱼ} = 2δᵢⱼI using the product formula.

5. Find the eigenstates of σᵧ.

6. Express the matrix [[1, 2], [2, 3]] in terms of {I, σₓ, σᵧ, σᵤ}.

### Challenging

7. Show that e^{iθσᵢ} = cos(θ)I + i sin(θ)σᵢ.

8. Prove that det(a₀I + **a**·**σ**) = a₀² - |**a**|².

---

## Computational Lab

```python
"""
Day 402 Computational Lab: Pauli Matrices
"""

import numpy as np

# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

pauli = {'I': I, 'X': sigma_x, 'Y': sigma_y, 'Z': sigma_z}

def verify_properties():
    """Verify basic Pauli matrix properties."""
    print("Verifying Pauli Matrix Properties")
    print("=" * 50)

    for name, sigma in [('X', sigma_x), ('Y', sigma_y), ('Z', sigma_z)]:
        print(f"\nσ_{name}:")
        print(f"  Hermitian: {np.allclose(sigma, sigma.conj().T)}")
        print(f"  Trace: {np.trace(sigma):.4f}")
        print(f"  Det: {np.linalg.det(sigma):.4f}")
        print(f"  σ² = I: {np.allclose(sigma @ sigma, I)}")

        # Eigenvalues
        eigs = np.linalg.eigvalsh(sigma)
        print(f"  Eigenvalues: {eigs}")

def verify_algebra():
    """Verify Pauli algebra relations."""
    print("\nVerifying Algebraic Relations")
    print("=" * 50)

    # Products
    print("\nProducts σᵢσⱼ = δᵢⱼI + iεᵢⱼₖσₖ:")
    print(f"  σₓσᵧ = iσᵤ: {np.allclose(sigma_x @ sigma_y, 1j * sigma_z)}")
    print(f"  σᵧσᵤ = iσₓ: {np.allclose(sigma_y @ sigma_z, 1j * sigma_x)}")
    print(f"  σᵤσₓ = iσᵧ: {np.allclose(sigma_z @ sigma_x, 1j * sigma_y)}")

    # Anticommutators
    print("\nAnticommutators {σᵢ, σⱼ} = 2δᵢⱼI:")
    for (n1, s1), (n2, s2) in [
        (('X', sigma_x), ('Y', sigma_y)),
        (('X', sigma_x), ('Z', sigma_z)),
        (('Y', sigma_y), ('Z', sigma_z)),
        (('X', sigma_x), ('X', sigma_x)),
    ]:
        anticomm = s1 @ s2 + s2 @ s1
        expected = 2*I if n1 == n2 else np.zeros((2,2))
        print(f"  {{σ_{n1}, σ_{n2}}} = {2 if n1==n2 else 0}I: {np.allclose(anticomm, expected)}")

    # Commutators
    print("\nCommutators [σᵢ, σⱼ] = 2iεᵢⱼₖσₖ:")
    print(f"  [σₓ, σᵧ] = 2iσᵤ: {np.allclose(sigma_x @ sigma_y - sigma_y @ sigma_x, 2j * sigma_z)}")

def find_eigenstates():
    """Find and display eigenstates of all Pauli matrices."""
    print("\nEigenstates of Pauli Matrices")
    print("=" * 50)

    for name, sigma in [('X', sigma_x), ('Y', sigma_y), ('Z', sigma_z)]:
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)

        print(f"\nσ_{name}:")
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            print(f"  λ = {val:+.0f}: |{'+' if val > 0 else '-'}⟩ = {vec}")

def pauli_decomposition(H):
    """Decompose 2x2 Hermitian matrix into Pauli basis."""
    h0 = np.trace(H) / 2
    h1 = np.trace(H @ sigma_x) / 2
    h2 = np.trace(H @ sigma_y) / 2
    h3 = np.trace(H @ sigma_z) / 2

    return np.real(h0), np.real(h1), np.real(h2), np.real(h3)

def demonstrate_decomposition():
    """Demonstrate Pauli decomposition."""
    print("\nPauli Decomposition Example")
    print("=" * 50)

    # Example matrix
    H = np.array([[2, 1], [1, 0]], dtype=complex)
    print(f"Matrix H =\n{H}")

    h0, h1, h2, h3 = pauli_decomposition(H)
    print(f"\nDecomposition: H = {h0:.2f}I + {h1:.2f}σₓ + {h2:.2f}σᵧ + {h3:.2f}σᵤ")

    # Verify
    H_reconstructed = h0*I + h1*sigma_x + h2*sigma_y + h3*sigma_z
    print(f"Reconstruction matches: {np.allclose(H, H_reconstructed)}")

if __name__ == "__main__":
    print("Day 402: Pauli Matrices")
    print("=" * 50)

    verify_properties()
    verify_algebra()
    find_eigenstates()
    demonstrate_decomposition()

    print("\nLab complete!")
```

---

## Summary

| Property | Formula |
|----------|---------|
| Pauli matrices | σₓ, σᵧ, σᵤ (2×2 matrices) |
| Square | σᵢ² = I |
| Product | σᵢσⱼ = δᵢⱼI + iεᵢⱼₖσₖ |
| Commutator | [σᵢ, σⱼ] = 2iεᵢⱼₖσₖ |
| Anticommutator | {σᵢ, σⱼ} = 2δᵢⱼI |
| Spin operators | Ŝᵢ = (ℏ/2)σᵢ |

---

## Daily Checklist

- [ ] I can write all three Pauli matrices
- [ ] I know σᵢ² = I
- [ ] I can verify the product formula
- [ ] I understand the connection to quantum gates
- [ ] I completed the computational lab

---

## Preview: Day 403

Tomorrow we visualize spin states using the **Bloch sphere**—the most intuitive representation of qubit states.

---

**Next:** [Day_403_Thursday.md](Day_403_Thursday.md) — Bloch Sphere
