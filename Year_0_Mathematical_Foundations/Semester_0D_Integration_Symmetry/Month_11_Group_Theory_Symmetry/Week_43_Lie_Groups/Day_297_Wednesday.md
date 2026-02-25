# Day 297: Lie Algebras — Infinitesimal Symmetries

## Overview

**Month 11, Week 43, Day 3 — Wednesday**

Today we study **Lie algebras**, the linearization of Lie groups at the identity. The Lie algebra captures the infinitesimal structure of a Lie group and is often easier to work with. In quantum mechanics, Lie algebra elements become observables—the generators of the corresponding symmetry transformations.

## Learning Objectives

1. Define Lie algebras abstractly and for matrix groups
2. Compute the Lie bracket (commutator)
3. Find the Lie algebra of standard matrix groups
4. Understand structure constants
5. Connect Lie algebra generators to quantum observables

---

## 1. Definition of Lie Algebra

### Abstract Definition

A **Lie algebra** $\mathfrak{g}$ over field $\mathbb{F}$ is a vector space with a bilinear operation $[\cdot, \cdot]: \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}$ (the **Lie bracket**) satisfying:

1. **Antisymmetry:** $[X, Y] = -[Y, X]$
2. **Jacobi Identity:** $[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0$

### Matrix Lie Algebras

For matrix Lie groups, the Lie algebra consists of matrices $X$ such that $e^{tX} \in G$ for all $t$, with Lie bracket:
$$[X, Y] = XY - YX$$

---

## 2. Lie Algebras of Matrix Groups

| Lie Group | Lie Algebra | Condition | Dimension |
|-----------|-------------|-----------|-----------|
| $GL_n$ | $\mathfrak{gl}_n$ | all matrices | $n^2$ |
| $SL_n$ | $\mathfrak{sl}_n$ | $\text{Tr}(X) = 0$ | $n^2-1$ |
| $O(n)$ | $\mathfrak{o}(n)$ | $X^T = -X$ (skew-symmetric) | $n(n-1)/2$ |
| $SO(n)$ | $\mathfrak{so}(n)$ | $X^T = -X$ | $n(n-1)/2$ |
| $U(n)$ | $\mathfrak{u}(n)$ | $X^\dagger = -X$ (skew-Hermitian) | $n^2$ |
| $SU(n)$ | $\mathfrak{su}(n)$ | $X^\dagger = -X$, $\text{Tr}(X) = 0$ | $n^2-1$ |

---

## 3. The Lie Algebra so(3)

### Basis Elements

$$J_x = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}, \quad
J_y = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix}, \quad
J_z = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

### Commutation Relations

$$\boxed{[J_i, J_j] = \epsilon_{ijk} J_k}$$

These define so(3) completely!

---

## 4. Structure Constants

For a basis $\{T_a\}$ of $\mathfrak{g}$, the **structure constants** $f_{ab}^c$ are defined by:
$$[T_a, T_b] = \sum_c f_{ab}^c T_c$$

For so(3): $f_{ij}^k = \epsilon_{ijk}$

Structure constants satisfy:
- Antisymmetry: $f_{ab}^c = -f_{ba}^c$
- Jacobi: $f_{ab}^d f_{dc}^e + f_{bc}^d f_{da}^e + f_{ca}^d f_{db}^e = 0$

---

## 5. The Exponential Map

The exponential map connects the Lie algebra to the Lie group:
$$\exp: \mathfrak{g} \to G, \quad X \mapsto e^X = \sum_{n=0}^\infty \frac{X^n}{n!}$$

**Properties:**
- $\exp(0) = I$
- $\exp(tX)$ is a one-parameter subgroup
- $\frac{d}{dt}\exp(tX)|_{t=0} = X$

---

## 6. Quantum Mechanics Connection

### Generators = Observables

In QM, the Lie algebra generators (multiplied by $i\hbar$) become Hermitian observables:

| Generator | Observable | Conservation |
|-----------|------------|--------------|
| Rotation $J_i$ | Angular momentum $L_i$ | Rotational symmetry |
| Translation | Momentum $p$ | Translational symmetry |
| Time evolution | Hamiltonian $H$ | Time symmetry |

### The Master Formula

$$[L_i, L_j] = i\hbar \epsilon_{ijk} L_k$$

This is so(3) with the factor $i\hbar$!

---

## 7. Computational Lab

```python
"""
Day 297: Lie Algebras
"""

import numpy as np

# so(3) generators
J = [
    np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),  # J_x
    np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float),  # J_y
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float),  # J_z
]

def commutator(A, B):
    return A @ B - B @ A

# Verify [J_x, J_y] = J_z
print("[J_x, J_y] = J_z:", np.allclose(commutator(J[0], J[1]), J[2]))
print("[J_y, J_z] = J_x:", np.allclose(commutator(J[1], J[2]), J[0]))
print("[J_z, J_x] = J_y:", np.allclose(commutator(J[2], J[0]), J[1]))

# Jacobi identity
jacobi = commutator(J[0], commutator(J[1], J[2])) + \
         commutator(J[1], commutator(J[2], J[0])) + \
         commutator(J[2], commutator(J[0], J[1]))
print("Jacobi identity satisfied:", np.allclose(jacobi, 0))
```

---

## Summary

$$\boxed{\text{Lie algebra } \mathfrak{g} = T_e G \text{ with bracket } [X,Y] = XY - YX}$$

| Concept | Formula |
|---------|---------|
| Bracket | $[X,Y] = XY - YX$ |
| Structure constants | $[T_a, T_b] = f_{ab}^c T_c$ |
| Exponential | $e^X \in G$ for $X \in \mathfrak{g}$ |
| so(3) | $[J_i, J_j] = \epsilon_{ijk} J_k$ |

---

## Preview: Day 298

Tomorrow: **The Lie Algebra so(3)** in depth — connection to angular momentum.
