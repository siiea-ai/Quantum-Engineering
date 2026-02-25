# Day 298: The Lie Algebra so(3) and Angular Momentum

## Overview

**Month 11, Week 43, Day 4 — Thursday**

Today we study **so(3)** in depth, connecting it to angular momentum in quantum mechanics. The commutation relations of angular momentum operators are precisely the structure of so(3), making this the most important Lie algebra for physics.

## Learning Objectives

1. Master the so(3) commutation relations
2. Connect to quantum angular momentum operators
3. Understand the Casimir operator
4. Derive eigenvalue spectra using Lie algebra methods

---

## 1. The Lie Algebra so(3)

### Definition

$$\mathfrak{so}(3) = \{X \in M_3(\mathbb{R}) : X^T = -X\}$$

A general element:
$$X = \begin{pmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{pmatrix} = \omega_x J_x + \omega_y J_y + \omega_z J_z$$

### The Fundamental Commutation Relations

$$\boxed{[J_i, J_j] = \epsilon_{ijk} J_k}$$

Explicitly:
- $[J_x, J_y] = J_z$
- $[J_y, J_z] = J_x$
- $[J_z, J_x] = J_y$

---

## 2. Quantum Angular Momentum

### The Operators

In quantum mechanics, we use Hermitian operators $L_i = -i\hbar J_i$:

$$\boxed{[L_i, L_j] = i\hbar \epsilon_{ijk} L_k}$$

Explicitly:
- $[L_x, L_y] = i\hbar L_z$
- $[L_y, L_z] = i\hbar L_x$
- $[L_z, L_x] = i\hbar L_y$

### Position Space Representation

$$L_x = -i\hbar(y\partial_z - z\partial_y)$$
$$L_y = -i\hbar(z\partial_x - x\partial_z)$$
$$L_z = -i\hbar(x\partial_y - y\partial_x)$$

In spherical coordinates:
$$L_z = -i\hbar \frac{\partial}{\partial\phi}$$

---

## 3. The Casimir Operator

### Definition

$$\mathbf{L}^2 = L_x^2 + L_y^2 + L_z^2$$

### Key Property

$$[\mathbf{L}^2, L_i] = 0 \quad \text{for all } i$$

The Casimir commutes with all generators!

### Eigenvalues

$$\mathbf{L}^2 |l, m\rangle = \hbar^2 l(l+1) |l, m\rangle$$
$$L_z |l, m\rangle = \hbar m |l, m\rangle$$

where $l = 0, 1, 2, \ldots$ and $m = -l, -l+1, \ldots, l-1, l$.

---

## 4. Ladder Operators

### Definition

$$L_\pm = L_x \pm i L_y$$

### Commutation Relations

$$[L_z, L_\pm] = \pm\hbar L_\pm$$
$$[L_+, L_-] = 2\hbar L_z$$

### Action on States

$$L_\pm |l, m\rangle = \hbar\sqrt{l(l+1) - m(m\pm 1)} |l, m\pm 1\rangle$$

---

## 5. Representations of so(3)

### Irreducible Representations

For each $l = 0, 1, 2, \ldots$, there is a $(2l+1)$-dimensional irrep.

| $l$ | Dimension | Name |
|-----|-----------|------|
| 0 | 1 | Scalar |
| 1 | 3 | Vector |
| 2 | 5 | Tensor |

### Matrix Representations

For $l = 1$:
$$L_z = \hbar\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}, \quad
L_+ = \hbar\sqrt{2}\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

---

## 6. Computational Lab

```python
"""
Day 298: so(3) and Angular Momentum
"""

import numpy as np

def angular_momentum_matrices(l):
    """Generate L_x, L_y, L_z matrices for spin l."""
    dim = int(2*l + 1)
    m_vals = np.arange(l, -l-1, -1)

    Lz = np.diag(m_vals)

    Lp = np.zeros((dim, dim))
    Lm = np.zeros((dim, dim))

    for i in range(dim - 1):
        m = m_vals[i + 1]
        Lp[i, i+1] = np.sqrt(l*(l+1) - m*(m+1))
        Lm[i+1, i] = np.sqrt(l*(l+1) - m*(m-1))

    Lx = (Lp + Lm) / 2
    Ly = (Lp - Lm) / (2j)

    return Lx, Ly, Lz

# For l = 1
Lx, Ly, Lz = angular_momentum_matrices(1)
print("L_z (l=1):\n", Lz)

# Verify commutation
comm_xy = Lx @ Ly - Ly @ Lx
print("\n[Lx, Ly] = i*Lz:", np.allclose(comm_xy, 1j * Lz))

# Casimir
L2 = Lx @ Lx + Ly @ Ly + Lz @ Lz
print("\nL² eigenvalues:", np.diag(L2))  # Should all be l(l+1) = 2
```

---

## Summary

$$\boxed{[L_i, L_j] = i\hbar \epsilon_{ijk} L_k}$$

| Operator | Eigenvalue |
|----------|------------|
| $\mathbf{L}^2$ | $\hbar^2 l(l+1)$ |
| $L_z$ | $\hbar m$ |
| $L_\pm$ | Raises/lowers $m$ by 1 |

---

## Preview: Day 299

Tomorrow: **SU(2) and its relationship to SO(3)** — the double cover and spinors.
