# Day 300: Representations of SU(2) — All Spins

## Overview

**Month 11, Week 43, Day 6 — Saturday**

Today we classify all irreducible representations of SU(2). Unlike SO(3), which only has integer-spin representations, SU(2) has representations for every half-integer spin $j = 0, 1/2, 1, 3/2, 2, ...$. This explains why nature has both integer-spin bosons and half-integer-spin fermions.

## Learning Objectives

1. Classify all irreps of SU(2)
2. Construct representations using ladder operators
3. Connect to quantum mechanical spin
4. Master the Clebsch-Gordan coefficients

---

## 1. Classification of Irreps

### The Main Result

For each $j = 0, \frac{1}{2}, 1, \frac{3}{2}, 2, \ldots$, there is a unique $(2j+1)$-dimensional irreducible representation $D^{(j)}$ of SU(2).

| $j$ | Dimension | Physical Name |
|-----|-----------|---------------|
| 0 | 1 | Scalar |
| 1/2 | 2 | Spinor |
| 1 | 3 | Vector |
| 3/2 | 4 | Spin-3/2 |
| 2 | 5 | Tensor |

### Why Both Integer and Half-Integer?

- **SO(3):** Only $j = 0, 1, 2, \ldots$ (single-valued functions on sphere)
- **SU(2):** All $j = 0, \frac{1}{2}, 1, \frac{3}{2}, \ldots$ (double cover allows half-integers)

---

## 2. Construction Using Ladder Operators

### The Algebra

$$[J_z, J_\pm] = \pm J_\pm$$
$$[J_+, J_-] = 2J_z$$
$$\mathbf{J}^2 = J_z^2 + \frac{1}{2}(J_+J_- + J_-J_+)$$

### Building the Representation

Start with highest weight state $|j, j\rangle$:
$$J_+|j, j\rangle = 0$$

Apply $J_-$ repeatedly:
$$|j, m\rangle = N_m J_-^{j-m}|j, j\rangle$$

for $m = j, j-1, \ldots, -j$.

### Normalization

$$J_\pm|j, m\rangle = \sqrt{j(j+1) - m(m\pm 1)}|j, m\pm 1\rangle$$

---

## 3. Matrix Representations

### Spin-1/2 ($j = 1/2$)

$$J_z = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \quad
J_+ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad
J_- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$$

### Spin-1 ($j = 1$)

$$J_z = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}, \quad
J_+ = \sqrt{2}\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

---

## 4. Tensor Products and Clebsch-Gordan

### Tensor Product Decomposition

$$D^{(j_1)} \otimes D^{(j_2)} = \bigoplus_{j=|j_1-j_2|}^{j_1+j_2} D^{(j)}$$

**Example:** $\frac{1}{2} \otimes \frac{1}{2} = 0 \oplus 1$

$$2 \times 2 = 1 + 3 \quad \checkmark$$

### Clebsch-Gordan Coefficients

$$|j, m\rangle = \sum_{m_1, m_2} C^{jm}_{j_1 m_1; j_2 m_2} |j_1, m_1\rangle \otimes |j_2, m_2\rangle$$

---

## 5. Physical Applications

### Electron Spin

The electron has $j = 1/2$:
$$|\uparrow\rangle = |{\textstyle\frac{1}{2}}, +{\textstyle\frac{1}{2}}\rangle, \quad |\downarrow\rangle = |{\textstyle\frac{1}{2}}, -{\textstyle\frac{1}{2}}\rangle$$

### Two-Electron Coupling

$$|{\textstyle\frac{1}{2}}\rangle \otimes |{\textstyle\frac{1}{2}}\rangle = |0, 0\rangle_\text{singlet} \oplus |1, m\rangle_\text{triplet}$$

Singlet: $\frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$

Triplet: $|\uparrow\uparrow\rangle, \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle), |\downarrow\downarrow\rangle$

---

## 6. Computational Lab

```python
"""
Day 300: Representations of SU(2)
"""

import numpy as np

def spin_matrices(j):
    """Generate J_x, J_y, J_z for spin j."""
    dim = int(2*j + 1)
    m = np.arange(j, -j-1, -1)

    Jz = np.diag(m)

    # J+ and J-
    Jp = np.zeros((dim, dim))
    Jm = np.zeros((dim, dim))

    for i in range(dim - 1):
        Jp[i, i+1] = np.sqrt(j*(j+1) - m[i]*(m[i]-1))
        Jm[i+1, i] = np.sqrt(j*(j+1) - m[i+1]*(m[i+1]+1))

    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2j)

    return Jx, Jy.imag, Jz

# Verify dimensions
for j in [0, 0.5, 1, 1.5, 2]:
    dim = int(2*j + 1)
    print(f"j = {j}: dimension = {dim}")

# Check Casimir
j = 1
Jx, Jy, Jz = spin_matrices(j)
J2 = Jx@Jx + Jy@Jy + Jz@Jz
print(f"\nJ² eigenvalues for j=1: {np.diag(J2).round(4)}")
print(f"Expected: j(j+1) = {j*(j+1)}")
```

---

## Summary

### Irreducible Representations of SU(2)

$$\boxed{D^{(j)}: \quad j = 0, \frac{1}{2}, 1, \frac{3}{2}, 2, \ldots \quad \dim = 2j + 1}$$

### Key Formulas

$$\mathbf{J}^2|j, m\rangle = j(j+1)|j, m\rangle$$
$$J_z|j, m\rangle = m|j, m\rangle$$
$$J_\pm|j, m\rangle = \sqrt{j(j+1) - m(m\pm 1)}|j, m\pm 1\rangle$$

---

## Preview: Day 301

Tomorrow: **Week 43 Review** — Lie groups and algebras synthesis.
