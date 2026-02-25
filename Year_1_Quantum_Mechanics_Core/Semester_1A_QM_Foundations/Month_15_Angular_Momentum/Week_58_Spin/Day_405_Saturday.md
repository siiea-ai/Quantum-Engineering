# Day 405: Higher Spin Representations

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Spin-1 and beyond |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 405, you will be able to:

1. Construct spin matrices for s = 1, 3/2, 2
2. Verify the angular momentum algebra for higher spins
3. Calculate eigenvalues and eigenstates
4. Understand the (2s+1)-dimensional representations of SU(2)
5. Connect higher spin to quantum computing (qutrits)

---

## Core Content

### 1. General Spin-s Particles

For spin quantum number s (integer or half-integer):

- **Dimension:** 2s + 1
- **States:** |s, m⟩ where m = -s, -s+1, ..., s-1, s
- **Eigenvalues:**
  - Ŝ² |s,m⟩ = ℏ²s(s+1)|s,m⟩
  - Ŝᵤ|s,m⟩ = ℏm|s,m⟩

### 2. Spin-1 (Triplet)

Three states: |1,1⟩, |1,0⟩, |1,-1⟩

**Spin-1 matrices:**

$$S_z = \hbar\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

$$S_+ = \hbar\sqrt{2}\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}, \quad S_- = \hbar\sqrt{2}\begin{pmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$$

$$S_x = \frac{\hbar}{\sqrt{2}}\begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}, \quad S_y = \frac{\hbar}{\sqrt{2}i}\begin{pmatrix} 0 & 1 & 0 \\ -1 & 0 & 1 \\ 0 & -1 & 0 \end{pmatrix}$$

### 3. Spin-3/2

Four states: |3/2, 3/2⟩, |3/2, 1/2⟩, |3/2, -1/2⟩, |3/2, -3/2⟩

**4×4 matrices** constructed using ladder operators.

### 4. General Construction

For any spin s, the matrix elements are:

$$\langle s,m'|\hat{S}_z|s,m\rangle = \hbar m\,\delta_{m'm}$$

$$\langle s,m'|\hat{S}_+|s,m\rangle = \hbar\sqrt{s(s+1)-m(m+1)}\,\delta_{m',m+1}$$

$$\langle s,m'|\hat{S}_-|s,m\rangle = \hbar\sqrt{s(s+1)-m(m-1)}\,\delta_{m',m-1}$$

### 5. SU(2) Representations

The spin-s representation is the (2s+1)-dimensional **irreducible representation** of the group SU(2).

| s | Dimension | Name | Example |
|---|-----------|------|---------|
| 0 | 1 | Scalar | π⁰ |
| 1/2 | 2 | Spinor | Electron |
| 1 | 3 | Vector | Photon (spin) |
| 3/2 | 4 | — | Δ baryon |
| 2 | 5 | Tensor | Graviton |

---

## Quantum Computing Connection

| Spin | QC System | Dimension |
|------|-----------|-----------|
| s = 1/2 | Qubit | 2 |
| s = 1 | **Qutrit** | 3 |
| s = 3/2 | Ququart | 4 |
| General s | Qudit | 2s+1 |

**Qutrits** (3-level systems) offer advantages for certain quantum algorithms and error correction schemes.

---

## Worked Examples

### Example 1: Verify [Sₓ, Sᵧ] = iℏSᵤ for Spin-1

**Solution:**
Using the spin-1 matrices:

$$[S_x, S_y] = S_x S_y - S_y S_x$$

After matrix multiplication (computation shown in lab):
$$[S_x, S_y] = i\hbar S_z \checkmark$$

### Example 2: Spin-1 Eigenvalue of Sₓ

**Problem:** Find the eigenvalues of Sₓ for spin-1.

**Solution:**
$$S_x = \frac{\hbar}{\sqrt{2}}\begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

Characteristic equation: det(Sₓ - λI) = 0

$$-\lambda^3 + \frac{\hbar^2}{2}(2\lambda) = 0 \Rightarrow \lambda(-\lambda^2 + \hbar^2) = 0$$

Eigenvalues: λ = 0, ±ℏ

Same as Sᵤ eigenvalues, as expected by symmetry!

### Example 3: Constructing Spin-3/2 Matrices

Using Ŝ±|s,m⟩ = ℏ√[s(s+1)-m(m±1)]|s,m±1⟩:

For s = 3/2:
- Ŝ₊|3/2,1/2⟩ = ℏ√[15/4 - 1/2·3/2]|3/2,3/2⟩ = ℏ√3|3/2,3/2⟩
- Ŝ₊|3/2,-1/2⟩ = ℏ√[15/4 - (-1/2)(1/2)]|3/2,1/2⟩ = ℏ·2|3/2,1/2⟩

---

## Practice Problems

### Direct Application

1. Write the Sᵤ matrix for spin-3/2.

2. Verify Ŝ²|1,0⟩ = 2ℏ²|1,0⟩ using the spin-1 matrices.

3. What are the eigenvalues of Sₓ for spin-3/2?

### Intermediate

4. Calculate [Sᵤ, S₊] for spin-1 and verify it equals ℏS₊.

5. Find the Sᵤ = 0 eigenstate of Sₓ for spin-1.

6. Construct the S₊ matrix for spin-2.

### Challenging

7. Prove that Tr(Sᵢ) = 0 for any spin s.

8. Show that the spin-s representation matrices satisfy the same commutation relations as spin-1/2.

---

## Computational Lab

```python
"""
Day 405 Computational Lab: Higher Spin Representations
"""

import numpy as np
from scipy.linalg import eigh

def construct_spin_matrices(s):
    """
    Construct Sx, Sy, Sz matrices for spin s.
    Basis: |s,s>, |s,s-1>, ..., |s,-s>
    """
    dim = int(2*s + 1)
    m_values = np.arange(s, -s-1, -1)

    # S_z diagonal
    Sz = np.diag(m_values).astype(complex)

    # S_+ raising operator
    Splus = np.zeros((dim, dim), dtype=complex)
    for i in range(dim-1):
        m = m_values[i+1]  # m value being raised
        Splus[i, i+1] = np.sqrt(s*(s+1) - m*(m+1))

    # S_- lowering operator
    Sminus = Splus.T.conj()

    # S_x and S_y
    Sx = (Splus + Sminus) / 2
    Sy = (Splus - Sminus) / (2j)

    # S^2
    S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz

    return Sx, Sy, Sz, Splus, Sminus, S2

def verify_commutation_relations(s):
    """Verify [Si, Sj] = i*epsilon_ijk*Sk for spin s."""
    Sx, Sy, Sz, _, _, _ = construct_spin_matrices(s)

    print(f"\nSpin s = {s} (dimension {int(2*s+1)}):")
    print("-" * 40)

    # [Sx, Sy] = i*Sz
    comm_xy = Sx @ Sy - Sy @ Sx
    print(f"  [Sx, Sy] = iSz: {np.allclose(comm_xy, 1j*Sz)}")

    # [Sy, Sz] = i*Sx
    comm_yz = Sy @ Sz - Sz @ Sy
    print(f"  [Sy, Sz] = iSx: {np.allclose(comm_yz, 1j*Sx)}")

    # [Sz, Sx] = i*Sy
    comm_zx = Sz @ Sx - Sx @ Sz
    print(f"  [Sz, Sx] = iSy: {np.allclose(comm_zx, 1j*Sy)}")

def display_matrices(s):
    """Display spin matrices for given s."""
    Sx, Sy, Sz, Splus, Sminus, S2 = construct_spin_matrices(s)

    print(f"\nSpin-{s} Matrices (ℏ = 1):")
    print("=" * 50)

    print("\nS_z =")
    print(np.real(Sz))

    print("\nS_x =")
    print(np.real(Sx))

    print("\nS_y =")
    print(np.real(Sy * 1j))  # Show imaginary part

    # Eigenvalues
    print(f"\nEigenvalues of S_z: {np.diag(np.real(Sz))}")
    eigs_x = np.linalg.eigvalsh(Sx)
    print(f"Eigenvalues of S_x: {eigs_x}")

    # S^2 eigenvalue
    s2_eigenvalue = s * (s + 1)
    print(f"\nS² eigenvalue: s(s+1) = {s2_eigenvalue}")
    print(f"Computed S² diagonal: {np.diag(np.real(S2))}")

def compare_dimensions():
    """Show how dimension grows with spin."""
    import matplotlib.pyplot as plt

    spins = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    dims = [int(2*s + 1) for s in spins]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar([str(s) for s in spins], dims, color='steelblue', edgecolor='black')
    ax.set_xlabel('Spin quantum number s', fontsize=12)
    ax.set_ylabel('Hilbert space dimension (2s+1)', fontsize=12)
    ax.set_title('Dimension of Spin-s Hilbert Space', fontsize=14)

    for i, (s, d) in enumerate(zip(spins, dims)):
        ax.annotate(f'{d}', (i, d), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=11)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('spin_dimensions.png', dpi=150)
    plt.show()

def qutrit_example():
    """Demonstrate qutrit (spin-1) as quantum system."""
    print("\nQutrit (Spin-1) Example")
    print("=" * 50)

    Sx, Sy, Sz, _, _, _ = construct_spin_matrices(1)

    # Qutrit states
    ket_plus1 = np.array([1, 0, 0], dtype=complex)
    ket_0 = np.array([0, 1, 0], dtype=complex)
    ket_minus1 = np.array([0, 0, 1], dtype=complex)

    # Superposition
    ket_super = (ket_plus1 + ket_0 + ket_minus1) / np.sqrt(3)

    print("Equal superposition: (|+1⟩ + |0⟩ + |-1⟩)/√3")
    print(f"  ⟨Sz⟩ = {np.real(np.vdot(ket_super, Sz @ ket_super)):.4f}")
    print(f"  ⟨Sx⟩ = {np.real(np.vdot(ket_super, Sx @ ket_super)):.4f}")

    # Probabilities
    print(f"  P(Sz=+1) = {np.abs(np.vdot(ket_plus1, ket_super))**2:.4f}")
    print(f"  P(Sz=0) = {np.abs(np.vdot(ket_0, ket_super))**2:.4f}")
    print(f"  P(Sz=-1) = {np.abs(np.vdot(ket_minus1, ket_super))**2:.4f}")

if __name__ == "__main__":
    print("Day 405: Higher Spin Representations")
    print("=" * 50)

    # Display matrices for spin-1
    display_matrices(1)

    # Verify commutation relations for various spins
    for s in [0.5, 1, 1.5, 2]:
        verify_commutation_relations(s)

    # Compare dimensions
    print("\nPlotting dimension comparison...")
    compare_dimensions()

    # Qutrit example
    qutrit_example()

    print("\nLab complete!")
```

---

## Summary

| Spin s | Dimension | Physical Example |
|--------|-----------|------------------|
| 0 | 1 | Higgs boson |
| 1/2 | 2 | Electron, quark |
| 1 | 3 | Photon, W/Z bosons |
| 3/2 | 4 | Delta baryon |
| 2 | 5 | Graviton |

**Key:** All spins satisfy the same angular momentum algebra: [Ŝᵢ, Ŝⱼ] = iℏεᵢⱼₖŜₖ

---

## Daily Checklist

- [ ] I can construct spin matrices for any s
- [ ] I verified commutation relations hold
- [ ] I understand the dimension formula 2s+1
- [ ] I know physical examples of different spins
- [ ] I completed the computational lab

---

## Preview: Day 406

Tomorrow we conclude Week 58 with a comprehensive review and a Qiskit lab connecting spin-1/2 physics to qubit operations.

---

**Next:** [Day_406_Sunday.md](Day_406_Sunday.md) — Week Review & Qiskit Lab
