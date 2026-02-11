# Week 43: Lie Groups and Lie Algebras (Days 295-301)

## Overview

Week 43 introduces **Lie groups** and **Lie algebras**, the mathematical framework for continuous symmetries. While finite groups describe discrete symmetries like reflections, Lie groups describe continuous symmetries like rotations. The Lie algebra captures the infinitesimal structure—the "derivatives" of group elements near the identity. This is where group theory connects most directly to quantum mechanics, where observables are generators of symmetry transformations.

## Learning Objectives

By the end of Week 43, you will be able to:

1. Define Lie groups and identify common examples (SO(n), SU(n), etc.)
2. Compute the Lie algebra from a Lie group
3. Work with the exponential map connecting algebra to group
4. Understand the relationship between SO(3) and SU(2)
5. Connect Lie algebra generators to quantum observables
6. Compute commutation relations and structure constants

## Daily Schedule

| Day | Topic | Key Concepts | QM Connection |
|-----|-------|--------------|---------------|
| 295 | Introduction to Lie Groups | Continuous groups, matrix Lie groups | Continuous symmetries |
| 296 | The Rotation Group SO(3) | Parameterizations, Euler angles | Spatial rotations |
| 297 | Lie Algebras | Tangent space, commutators | Infinitesimal generators |
| 298 | The Lie Algebra so(3) | Angular momentum generators | L_x, L_y, L_z |
| 299 | SU(2) and SO(3) | Double cover, spinors | Spin-1/2 |
| 300 | Representations of SU(2) | Spin representations, ladders | All spins j |
| 301 | Week Review | Synthesis, applications | Full integration |

## Key Concepts

### Lie Groups

A **Lie group** is a group that is also a smooth manifold, with smooth group operations.

**Matrix Lie groups:** Closed subgroups of $GL_n(\mathbb{C})$:
- $GL_n$: General linear (all invertible)
- $SL_n$: Special linear ($\det = 1$)
- $O(n)$: Orthogonal ($A^T A = I$)
- $SO(n)$: Special orthogonal ($\det = 1$)
- $U(n)$: Unitary ($A^\dagger A = I$)
- $SU(n)$: Special unitary ($\det = 1$)

### Lie Algebras

The **Lie algebra** $\mathfrak{g}$ of a Lie group $G$ is the tangent space at identity with the commutator bracket:

$$[X, Y] = XY - YX$$

### Exponential Map

$$\exp: \mathfrak{g} \to G, \quad X \mapsto e^X = \sum_{n=0}^\infty \frac{X^n}{n!}$$

This connects infinitesimal generators to finite transformations.

### Angular Momentum Relations

$$[L_i, L_j] = i\hbar \epsilon_{ijk} L_k$$

This is the Lie algebra $\mathfrak{so}(3)$!

## Python Libraries

```python
import numpy as np
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation
```

## Resources

### Primary Texts
- Hall, *Lie Groups, Lie Algebras, and Representations*
- Stillwell, *Naive Lie Theory*
- Kirillov, *Introduction to Lie Groups and Lie Algebras*

### Physics-Oriented
- Georgi, *Lie Algebras in Particle Physics*
- Jones, *Groups, Representations and Physics*

## Week Progress

| Day | File | Status |
|-----|------|--------|
| 295 | Day_295_Monday.md | 🔄 In Progress |
| 296 | Day_296_Tuesday.md | ⬜ Not Started |
| 297 | Day_297_Wednesday.md | ⬜ Not Started |
| 298 | Day_298_Thursday.md | ⬜ Not Started |
| 299 | Day_299_Friday.md | ⬜ Not Started |
| 300 | Day_300_Saturday.md | ⬜ Not Started |
| 301 | Day_301_Sunday.md | ⬜ Not Started |

**Week Progress:** 0/7 days (0%)
