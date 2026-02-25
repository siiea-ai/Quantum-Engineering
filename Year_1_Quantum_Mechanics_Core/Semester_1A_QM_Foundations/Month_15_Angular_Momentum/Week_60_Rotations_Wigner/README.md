# Week 60: Rotations and Wigner D-Matrices

## Month 15: Angular Momentum | Days 414-420

### Week Overview

This week culminates Month 15 by developing the full theory of rotations in quantum mechanics. We explore how rotation operators act on quantum states, introduce Euler angles as the standard parameterization, and derive the Wigner D-matrices that encode how angular momentum states transform under rotations. The powerful Wigner-Eckart theorem reveals how matrix elements of tensor operators factorize into geometric (Clebsch-Gordan) and dynamical (reduced matrix element) parts. This leads to selection rules that govern atomic transitions and spectroscopy. The week concludes with a comprehensive Month 15 capstone connecting angular momentum to quantum computing through the SU(2)-qubit correspondence.

**Primary References:**
- Shankar, *Principles of Quantum Mechanics*, Chapters 12-13
- Sakurai & Napolitano, *Modern Quantum Mechanics*, Sections 3.3-3.4
- Varshalovich et al., *Quantum Theory of Angular Momentum*

---

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **414 (Mon)** | Rotation Operators | Classical SO(3), quantum $$\hat{R}(\hat{n},\theta) = e^{-i\theta\hat{n}\cdot\hat{\mathbf{J}}/\hbar}$$, infinitesimal rotations |
| **415 (Tue)** | Euler Angles | Convention $$R(\alpha,\beta,\gamma)$$, physical interpretation, gimbal lock |
| **416 (Wed)** | Wigner D-Matrices | $$D^j_{m'm}(\alpha,\beta,\gamma)$$, small d-matrices, orthogonality |
| **417 (Thu)** | Wigner-Eckart Theorem | Tensor operators, reduced matrix elements, factorization theorem |
| **418 (Fri)** | Selection Rules | Electric/magnetic dipole, quadrupole, parity, polarization rules |
| **419 (Sat)** | Tensor Operators | Spherical tensors, irreducible representations, 6j/9j symbols |
| **420 (Sun)** | Month 15 Capstone | Comprehensive review, SU(2) ↔ qubits, capstone project |

---

## Learning Objectives

By the end of this week, you will be able to:

1. **Construct rotation operators** using the exponential map $$\hat{R} = e^{-i\theta\hat{n}\cdot\hat{\mathbf{J}}/\hbar}$$
2. **Parameterize rotations** using Euler angles and understand their geometric meaning
3. **Calculate Wigner D-matrices** and apply them to rotate angular momentum states
4. **Apply the Wigner-Eckart theorem** to evaluate matrix elements of tensor operators
5. **Derive selection rules** for electromagnetic transitions in atoms
6. **Manipulate tensor operators** and understand their transformation properties
7. **Connect SU(2) to quantum computing** through single-qubit gate representations

---

## Key Formulas

### Rotation Operators

$$\boxed{\hat{R}(\hat{n},\theta) = e^{-i\theta\hat{n}\cdot\hat{\mathbf{J}}/\hbar} = \sum_{k=0}^{\infty} \frac{(-i\theta/\hbar)^k}{k!}(\hat{n}\cdot\hat{\mathbf{J}})^k}$$

**Infinitesimal rotation:**
$$\hat{R}(\hat{n},d\theta) = \hat{I} - \frac{i}{\hbar}d\theta\,\hat{n}\cdot\hat{\mathbf{J}} + O(d\theta^2)$$

**Non-commutativity:**
$$[\hat{R}_x(\theta_1), \hat{R}_y(\theta_2)] \neq 0$$

### Euler Angle Decomposition

$$\boxed{R(\alpha,\beta,\gamma) = R_z(\alpha)R_y(\beta)R_z(\gamma)}$$

**Parameter ranges:**
- $$0 \leq \alpha < 2\pi$$
- $$0 \leq \beta \leq \pi$$
- $$0 \leq \gamma < 2\pi$$

### Wigner D-Matrices

$$\boxed{D^j_{m'm}(\alpha,\beta,\gamma) = \langle j,m'|\hat{R}(\alpha,\beta,\gamma)|j,m\rangle = e^{-im'\alpha}d^j_{m'm}(\beta)e^{-im\gamma}}$$

**Small d-matrix (Wigner formula):**
$$d^j_{m'm}(\beta) = \sum_k \frac{(-1)^{k-m+m'}\sqrt{(j+m)!(j-m)!(j+m')!(j-m')!}}{(j+m-k)!(j-m'-k)!k!(k-m+m')!}\left(\cos\frac{\beta}{2}\right)^{2j-2k+m-m'}\left(\sin\frac{\beta}{2}\right)^{2k-m+m'}$$

**Orthogonality:**
$$\int d\Omega\, D^{j_1}_{m_1'm_1}(\alpha,\beta,\gamma)^* D^{j_2}_{m_2'm_2}(\alpha,\beta,\gamma) = \frac{8\pi^2}{2j_1+1}\delta_{j_1j_2}\delta_{m_1'm_2'}\delta_{m_1m_2}$$

### Wigner-Eckart Theorem

$$\boxed{\langle j',m'|T^{(k)}_q|j,m\rangle = \langle j,m;k,q|j',m'\rangle \frac{\langle j'||T^{(k)}||j\rangle}{\sqrt{2j'+1}}}$$

**Reduced matrix element** $$\langle j'||T^{(k)}||j\rangle$$ is independent of $$m, m', q$$.

### Selection Rules

| Transition Type | $$\Delta J$$ | $$\Delta m$$ | Parity |
|-----------------|-------------|-------------|--------|
| Electric dipole (E1) | $$0, \pm 1$$ (not $$0 \to 0$$) | $$0, \pm 1$$ | Changes |
| Magnetic dipole (M1) | $$0, \pm 1$$ (not $$0 \to 0$$) | $$0, \pm 1$$ | No change |
| Electric quadrupole (E2) | $$0, \pm 1, \pm 2$$ (not $$0 \to 0, 1$$) | $$0, \pm 1, \pm 2$$ | No change |

### SU(2) and Quantum Computing

**Spin-1/2 rotation:**
$$\hat{R}(\hat{n},\theta) = \cos\frac{\theta}{2}\hat{I} - i\sin\frac{\theta}{2}(\hat{n}\cdot\boldsymbol{\sigma})$$

**Single-qubit gates as SU(2):**
$$\boxed{U(\theta,\phi,\lambda) = \begin{pmatrix} \cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2} \end{pmatrix}}$$

---

## Prerequisites

- Week 57: Orbital angular momentum ($$\hat{L}^2, \hat{L}_z$$, spherical harmonics)
- Week 58: Spin (Pauli matrices, spinor transformations)
- Week 59: Addition of angular momentum (Clebsch-Gordan coefficients)
- Matrix exponentials and group theory basics

---

## Computational Tools

This week's Python implementations use:
```python
import numpy as np
from scipy.linalg import expm
from scipy.special import factorial
import matplotlib.pyplot as plt
from sympy.physics.quantum.spin import Rotation, WignerD
from sympy.physics.quantum.cg import CG
```

---

## Assessment Criteria

- [ ] Derive rotation operators from angular momentum generators
- [ ] Convert between axis-angle and Euler angle representations
- [ ] Calculate Wigner D-matrix elements for $$j = 1/2, 1, 3/2$$
- [ ] Apply Wigner-Eckart theorem to evaluate tensor matrix elements
- [ ] Predict allowed transitions using selection rules
- [ ] Construct and manipulate spherical tensor operators
- [ ] Explain SU(2) $$\leftrightarrow$$ single-qubit gate correspondence

---

## Week Summary

This week bridges abstract group theory with practical quantum mechanics. Rotation operators form the Lie group SU(2), which is the universal cover of SO(3)—this is why spin-1/2 particles acquire a sign flip under $$2\pi$$ rotation. Wigner D-matrices provide the explicit representation of rotations on angular momentum states and are essential for calculating transition amplitudes. The Wigner-Eckart theorem is one of the most powerful results in quantum mechanics, reducing complex matrix element calculations to products of known Clebsch-Gordan coefficients and a single reduced matrix element. Selection rules derived from this theorem govern all of atomic spectroscopy and are foundational to understanding light-matter interaction. The connection to quantum computing through SU(2) shows that every single-qubit gate is a rotation of the Bloch sphere—a beautiful unification of abstract mathematics with quantum technology.

---

*Week 60 of 312 | Month 15 of 72 | Year 1: Quantum Mechanics Core*
