# Week 57: Orbital Angular Momentum

## Overview

This week introduces one of the most important concepts in quantum mechanics: **orbital angular momentum**. We transition from the classical definition $\mathbf{L} = \mathbf{r} \times \mathbf{p}$ to its quantum mechanical operator formulation, discover the fundamental commutation relations that define angular momentum algebra, and derive the quantized eigenvalue spectrum that explains atomic orbital structure.

**Prerequisites:** Quantum mechanics basics (Year 1 Weeks 1-56), classical mechanics angular momentum, complex analysis, spherical coordinates

**Primary References:**
- Shankar, *Principles of Quantum Mechanics*, Chapter 12
- Sakurai & Napolitano, *Modern Quantum Mechanics*, Chapter 3.5-3.6
- Griffiths, *Introduction to Quantum Mechanics*, Chapter 4

---

## Daily Schedule

| Day | Topic | Key Concepts | Hours |
|-----|-------|--------------|-------|
| **393 (Mon)** | Classical to Quantum Angular Momentum | $\hat{L} = -i\hbar(\mathbf{r} \times \nabla)$, component operators | 7 |
| **394 (Tue)** | Commutation Relations | $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$, uncertainty | 7 |
| **395 (Wed)** | Ladder Operators | $\hat{L}_\pm = \hat{L}_x \pm i\hat{L}_y$, raising/lowering | 7 |
| **396 (Thu)** | Eigenvalue Spectrum | $\ell(\ell+1)\hbar^2$, $m\hbar$, quantization | 7 |
| **397 (Fri)** | Spherical Harmonics I | $Y_\ell^m(\theta,\phi)$, explicit forms | 7 |
| **398 (Sat)** | Spherical Harmonics II | Legendre polynomials, addition theorem | 7 |
| **399 (Sun)** | Week Review & Lab | Atomic orbitals, comprehensive lab | 7 |

**Total Study Time:** 49 hours

---

## Learning Objectives

By the end of this week, you will be able to:

1. **Derive** the quantum angular momentum operators from the classical expression using canonical quantization
2. **Prove** the fundamental commutation relations $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$
3. **Construct** ladder operators and demonstrate their action on angular momentum eigenstates
4. **Derive** the complete eigenvalue spectrum of $\hat{L}^2$ and $\hat{L}_z$
5. **Express** spherical harmonics explicitly and explain their physical significance
6. **Visualize** angular momentum eigenfunctions and connect them to atomic orbitals
7. **Apply** angular momentum algebra to quantum computing gate operations

---

## Key Formulas

### Angular Momentum Operators

$$\boxed{\hat{L}_x = \hat{y}\hat{p}_z - \hat{z}\hat{p}_y = -i\hbar\left(y\frac{\partial}{\partial z} - z\frac{\partial}{\partial y}\right)}$$

$$\boxed{\hat{L}_y = \hat{z}\hat{p}_x - \hat{x}\hat{p}_z = -i\hbar\left(z\frac{\partial}{\partial x} - x\frac{\partial}{\partial z}\right)}$$

$$\boxed{\hat{L}_z = \hat{x}\hat{p}_y - \hat{y}\hat{p}_x = -i\hbar\left(x\frac{\partial}{\partial y} - y\frac{\partial}{\partial x}\right) = -i\hbar\frac{\partial}{\partial\phi}}$$

$$\boxed{\hat{L}^2 = \hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2 = -\hbar^2\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]}$$

### Commutation Relations

$$\boxed{[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k}$$

$$\boxed{[\hat{L}^2, \hat{L}_i] = 0 \quad \text{for all } i}$$

### Ladder Operators

$$\boxed{\hat{L}_\pm = \hat{L}_x \pm i\hat{L}_y}$$

$$\boxed{[\hat{L}_z, \hat{L}_\pm] = \pm\hbar\hat{L}_\pm}$$

$$\boxed{\hat{L}^2 = \hat{L}_\mp\hat{L}_\pm + \hat{L}_z^2 \pm \hbar\hat{L}_z}$$

$$\boxed{\hat{L}_\pm|\ell, m\rangle = \hbar\sqrt{\ell(\ell+1) - m(m\pm 1)}|\ell, m\pm 1\rangle}$$

### Eigenvalues

$$\boxed{\hat{L}^2|\ell, m\rangle = \hbar^2\ell(\ell+1)|\ell, m\rangle, \quad \ell = 0, 1, 2, 3, \ldots}$$

$$\boxed{\hat{L}_z|\ell, m\rangle = \hbar m|\ell, m\rangle, \quad m = -\ell, -\ell+1, \ldots, \ell-1, \ell}$$

### Spherical Harmonics

$$\boxed{Y_\ell^m(\theta, \phi) = (-1)^m\sqrt{\frac{(2\ell+1)}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}}P_\ell^m(\cos\theta)e^{im\phi}}$$

**Orthonormality:**
$$\boxed{\int_0^{2\pi}\int_0^\pi Y_{\ell'}^{m'*}(\theta,\phi)Y_\ell^m(\theta,\phi)\sin\theta\,d\theta\,d\phi = \delta_{\ell\ell'}\delta_{mm'}}$$

**Completeness:**
$$\boxed{\sum_{\ell=0}^\infty\sum_{m=-\ell}^{\ell}Y_\ell^{m*}(\theta',\phi')Y_\ell^m(\theta,\phi) = \delta(\cos\theta - \cos\theta')\delta(\phi - \phi')}$$

---

## Explicit Spherical Harmonics

| $\ell$ | $m$ | $Y_\ell^m(\theta,\phi)$ | Orbital |
|--------|-----|-------------------------|---------|
| 0 | 0 | $\frac{1}{\sqrt{4\pi}}$ | s |
| 1 | 0 | $\sqrt{\frac{3}{4\pi}}\cos\theta$ | p$_z$ |
| 1 | $\pm 1$ | $\mp\sqrt{\frac{3}{8\pi}}\sin\theta\,e^{\pm i\phi}$ | p$_\pm$ |
| 2 | 0 | $\sqrt{\frac{5}{16\pi}}(3\cos^2\theta - 1)$ | d$_{z^2}$ |
| 2 | $\pm 1$ | $\mp\sqrt{\frac{15}{8\pi}}\sin\theta\cos\theta\,e^{\pm i\phi}$ | d$_{\pm 1}$ |
| 2 | $\pm 2$ | $\sqrt{\frac{15}{32\pi}}\sin^2\theta\,e^{\pm 2i\phi}$ | d$_{\pm 2}$ |

---

## Quantum Computing Connections

Angular momentum is foundational to quantum computing:

1. **Qubit as Spin-1/2:** The two-level qubit system is mathematically equivalent to spin-1/2 angular momentum
2. **Rotation Gates:** Pauli gates $X$, $Y$, $Z$ generate rotations: $R_n(\theta) = e^{-i\theta\hat{n}\cdot\boldsymbol{\sigma}/2}$
3. **Bloch Sphere:** Represents qubit states using angular momentum language
4. **SU(2) Algebra:** The commutation relations $[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k$ mirror angular momentum

---

## Computational Tools This Week

```python
# Required packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, lpmv
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy.physics.quantum import *
```

---

## Assessment Milestones

- [ ] Derive all three angular momentum commutation relations from $[\hat{x}, \hat{p}_x] = i\hbar$
- [ ] Prove $[\hat{L}^2, \hat{L}_z] = 0$ algebraically
- [ ] Show ladder operator action: $\hat{L}_+|1, 0\rangle = \sqrt{2}\hbar|1, 1\rangle$
- [ ] Verify orthonormality of $Y_1^0$, $Y_1^1$, $Y_1^{-1}$
- [ ] Generate 3D visualizations of spherical harmonics for $\ell = 0, 1, 2, 3$
- [ ] Complete comprehensive computational lab

---

## Preview: Week 58 (Spin Angular Momentum)

Next week introduces **spin**, a purely quantum mechanical form of angular momentum with no classical analog. We will discover that particles can have half-integer angular momentum ($s = 1/2$), leading to:
- Pauli matrices and spinors
- The Stern-Gerlach experiment
- Spin-statistics connection
- Quantum entanglement via spin states

---

*Week 57 of Year 1: Quantum Mechanics Core*
*QSE Self-Study Curriculum - Orbital Angular Momentum*
