# Day 296: The Rotation Group SO(3)

## Overview

**Month 11, Week 43, Day 2 — Tuesday**

Today we study **SO(3)**, the group of rotations in three dimensions. This is the most important Lie group for physics—it describes angular momentum, molecular symmetry, and the transformation of quantum states under spatial rotations. We explore various parameterizations, the geometry of SO(3), and its connection to angular momentum operators.

## Learning Objectives

By the end of today, you will be able to:
1. Parameterize SO(3) using axis-angle and Euler angles
2. Derive Rodrigues' rotation formula
3. Understand the topology of SO(3)
4. Connect rotations to angular momentum operators
5. Compute rotation matrices from different parameterizations

---

## 1. SO(3): The Special Orthogonal Group

### Definition

$$SO(3) = \{R \in GL_3(\mathbb{R}) : R^T R = I, \det(R) = 1\}$$

**Properties:**
- Preserves lengths and angles
- Preserves orientation (right-handed → right-handed)
- Dimension: 3 (three rotation angles)
- Connected but NOT simply connected

### Rotation Generators

The generators of infinitesimal rotations:
$$J_x = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}, \quad
J_y = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix}, \quad
J_z = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

These satisfy the **so(3) commutation relations:**
$$[J_i, J_j] = \epsilon_{ijk} J_k$$

---

## 2. Parameterizations of SO(3)

### Axis-Angle Parameterization

Any rotation is specified by:
- **Axis:** unit vector $\hat{n} = (n_x, n_y, n_z)$
- **Angle:** $\theta \in [0, \pi]$

**Rodrigues' Formula:**
$$R(\hat{n}, \theta) = I + \sin\theta \, [\hat{n}]_\times + (1-\cos\theta) \, [\hat{n}]_\times^2$$

where $[\hat{n}]_\times = \begin{pmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{pmatrix}$ is the skew-symmetric matrix.

### Euler Angles (ZYZ Convention)

$$R(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_z(\gamma)$$

- $\alpha \in [0, 2\pi)$: first rotation about z
- $\beta \in [0, \pi]$: rotation about new y
- $\gamma \in [0, 2\pi)$: rotation about new z

**Note:** Euler angles have singularities at $\beta = 0, \pi$ (gimbal lock).

### Quaternion Representation

Unit quaternions $q = (q_0, q_1, q_2, q_3)$ with $|q| = 1$ represent rotations:
$$R(q) = \begin{pmatrix}
1-2(q_2^2+q_3^2) & 2(q_1q_2-q_0q_3) & 2(q_1q_3+q_0q_2) \\
2(q_1q_2+q_0q_3) & 1-2(q_1^2+q_3^2) & 2(q_2q_3-q_0q_1) \\
2(q_1q_3-q_0q_2) & 2(q_2q_3+q_0q_1) & 1-2(q_1^2+q_2^2)
\end{pmatrix}$$

---

## 3. Topology of SO(3)

### SO(3) is NOT Simply Connected

A path from $R(\hat{z}, 0)$ to $R(\hat{z}, 2\pi)$ cannot be continuously contracted to a point!

This is related to the "belt trick" or "Dirac string trick."

### Universal Cover: SU(2)

The group SU(2) (unit quaternions) covers SO(3) with a 2:1 map:
$$\pi: SU(2) \to SO(3)$$

Both $q$ and $-q$ map to the same rotation.

This is why spin-1/2 particles need $4\pi$ rotation to return to original state!

---

## 4. Angular Momentum Connection

### Quantum Mechanical Angular Momentum

The angular momentum operators satisfy:
$$[L_i, L_j] = i\hbar \epsilon_{ijk} L_k$$

This is the Lie algebra so(3), scaled by $i\hbar$!

Rotations act on quantum states via:
$$|\psi\rangle \mapsto e^{-i\theta \hat{n} \cdot \vec{L}/\hbar}|\psi\rangle$$

### Spherical Harmonics

Spherical harmonics $Y_\ell^m(\theta, \phi)$ form the $2\ell+1$ dimensional representation of SO(3).

---

## 5. Computational Lab

```python
"""
Day 296: The Rotation Group SO(3)
"""

import numpy as np
from scipy.spatial.transform import Rotation

def rodrigues(axis, angle):
    """Rodrigues' rotation formula."""
    axis = np.array(axis) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

def euler_to_rotation(alpha, beta, gamma):
    """ZYZ Euler angles to rotation matrix."""
    return Rotation.from_euler('ZYZ', [alpha, beta, gamma]).as_matrix()

# Demonstrations
if __name__ == "__main__":
    # Compare parameterizations
    axis = [1, 1, 1]
    angle = np.pi/3

    R_rodrigues = rodrigues(axis, angle)
    R_scipy = Rotation.from_rotvec(angle * np.array(axis)/np.linalg.norm(axis)).as_matrix()

    print("Rodrigues vs SciPy match:", np.allclose(R_rodrigues, R_scipy))

    # SO(3) is 3-dimensional
    print(f"dim(SO(3)) = 3: axis (2 params) + angle (1 param)")
```

---

## 6. Summary

| Parameterization | Parameters | Singularities |
|-----------------|------------|---------------|
| Axis-angle | $\hat{n}, \theta$ | $\theta = 0$ |
| Euler (ZYZ) | $\alpha, \beta, \gamma$ | $\beta = 0, \pi$ |
| Quaternion | $q_0, q_1, q_2, q_3$ | None (double cover) |

**Key Result:** $SO(3) \cong \mathbb{RP}^3$ (real projective space)

---

## Preview: Day 297

Tomorrow: **Lie Algebras** — the tangent space structure and commutator bracket.
