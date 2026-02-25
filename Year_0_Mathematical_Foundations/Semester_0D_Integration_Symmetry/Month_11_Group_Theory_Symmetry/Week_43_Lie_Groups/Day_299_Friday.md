# Day 299: SU(2) and Its Relationship to SO(3)

## Overview

**Month 11, Week 43, Day 5 — Friday**

Today we study **SU(2)** and its deep connection to SO(3). Although both groups have dimension 3 and the same Lie algebra, SU(2) is the **universal cover** of SO(3). This explains why spin-1/2 particles exist and why fermions need a $4\pi$ rotation to return to their original state.

## Learning Objectives

1. Define and parameterize SU(2)
2. Understand the double cover $SU(2) \to SO(3)$
3. Connect to spinors and spin-1/2 particles
4. Master the Pauli matrices

---

## 1. The Group SU(2)

### Definition

$$SU(2) = \{U \in M_2(\mathbb{C}) : U^\dagger U = I, \det(U) = 1\}$$

### Explicit Parameterization

Every $U \in SU(2)$ has the form:
$$U = \begin{pmatrix} \alpha & -\bar{\beta} \\ \beta & \bar{\alpha} \end{pmatrix}, \quad |\alpha|^2 + |\beta|^2 = 1$$

This is the **3-sphere** $S^3$ embedded in $\mathbb{C}^2 \cong \mathbb{R}^4$.

### Properties

- **Dimension:** 3 (same as SO(3))
- **Topology:** $S^3$ (simply connected!)
- **Lie algebra:** $\mathfrak{su}(2) \cong \mathfrak{so}(3)$

---

## 2. The Pauli Matrices

### Definition

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### Properties

$$\sigma_i^2 = I, \quad \sigma_i \sigma_j = i\epsilon_{ijk}\sigma_k + \delta_{ij}I$$

$$[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k$$

### The Lie Algebra su(2)

$\mathfrak{su}(2)$ = traceless skew-Hermitian matrices.

Basis: $\{-i\sigma_x/2, -i\sigma_y/2, -i\sigma_z/2\}$

Commutation: $[T_i, T_j] = i\epsilon_{ijk}T_k$ where $T_i = \sigma_i/2$.

---

## 3. The Double Cover $SU(2) \to SO(3)$

### The Homomorphism

For $U \in SU(2)$, define $\phi(U): \mathbb{R}^3 \to \mathbb{R}^3$ by:
$$\phi(U)(\vec{v}) = U(\vec{v} \cdot \vec{\sigma})U^\dagger$$

where $\vec{v} \cdot \vec{\sigma} = v_x\sigma_x + v_y\sigma_y + v_z\sigma_z$.

**Theorem:** $\phi(U) \in SO(3)$ and $\ker(\phi) = \{I, -I\}$.

### The Key Relation

$$\boxed{SO(3) \cong SU(2)/\{\pm I\}}$$

Both $U$ and $-U$ give the same rotation!

### Rotation Matrix

For rotation by angle $\theta$ about axis $\hat{n}$:
$$U = \cos(\theta/2)I - i\sin(\theta/2)(\hat{n}\cdot\vec{\sigma})$$

Note the $\theta/2$: a $2\pi$ rotation gives $U = -I \neq I$.

---

## 4. Spinors and Spin-1/2

### Physical Implication

Electrons are spin-1/2 particles: they transform under SU(2), not SO(3)!

A $2\pi$ rotation: $|\psi\rangle \mapsto -|\psi\rangle$

A $4\pi$ rotation: $|\psi\rangle \mapsto +|\psi\rangle$

### The Spin-1/2 Representation

The defining representation of SU(2) on $\mathbb{C}^2$ describes spin-1/2.

Spin operators: $S_i = \frac{\hbar}{2}\sigma_i$

$$[S_i, S_j] = i\hbar\epsilon_{ijk}S_k$$

---

## 5. Computational Lab

```python
"""
Day 299: SU(2) and SO(3)
"""

import numpy as np

# Pauli matrices
sigma = [
    np.array([[0, 1], [1, 0]], dtype=complex),      # σ_x
    np.array([[0, -1j], [1j, 0]], dtype=complex),   # σ_y
    np.array([[1, 0], [0, -1]], dtype=complex),     # σ_z
]

def su2_rotation(axis, angle):
    """SU(2) matrix for rotation."""
    axis = np.array(axis) / np.linalg.norm(axis)
    n_dot_sigma = sum(axis[i] * sigma[i] for i in range(3))
    return np.cos(angle/2)*np.eye(2) - 1j*np.sin(angle/2)*n_dot_sigma

def su2_to_so3(U):
    """Map SU(2) → SO(3)."""
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.5 * np.trace(sigma[i] @ U @ sigma[j] @ U.conj().T).real
    return R

# Verify double cover
U_2pi = su2_rotation([0, 0, 1], 2*np.pi)
print("U(2π) = -I:", np.allclose(U_2pi, -np.eye(2)))

U_4pi = su2_rotation([0, 0, 1], 4*np.pi)
print("U(4π) = +I:", np.allclose(U_4pi, np.eye(2)))

# Both U and -U give same rotation
U = su2_rotation([1, 0, 0], np.pi/3)
R1 = su2_to_so3(U)
R2 = su2_to_so3(-U)
print("R(U) = R(-U):", np.allclose(R1, R2))
```

---

## Summary

$$\boxed{SU(2) \xrightarrow{2:1} SO(3)}$$

| Property | SU(2) | SO(3) |
|----------|-------|-------|
| Dimension | 3 | 3 |
| Topology | $S^3$ | $\mathbb{RP}^3$ |
| Simply connected | Yes | No |
| Fundamental rep | $\mathbb{C}^2$ (spinors) | $\mathbb{R}^3$ (vectors) |

---

## Preview: Day 300

Tomorrow: **Representations of SU(2)** — all spin values j = 0, 1/2, 1, 3/2, ...
