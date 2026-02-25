# Week 44: Angular Momentum and Clebsch-Gordan Coefficients

## Overview

**Month 11, Week 44 (Days 302-308)**

This week applies the SU(2) representation theory developed in Week 43 to the quantum mechanics of angular momentum. We connect the abstract mathematical framework to physical observables: orbital angular momentum, spin, and their coupling. The Clebsch-Gordan coefficients provide the key to understanding composite quantum systems.

## Learning Objectives

By the end of this week, you will be able to:

1. Derive angular momentum operators from rotation generators
2. Solve the eigenvalue problem for $\mathbf{L}^2$ and $L_z$
3. Understand spherical harmonics as orbital angular momentum eigenfunctions
4. Work with spin angular momentum and Pauli matrices
5. Add angular momenta using tensor products
6. Calculate Clebsch-Gordan coefficients
7. Apply selection rules to atomic transitions

## Daily Schedule

| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 302 | Monday | Angular Momentum in QM | Operators, commutation relations, eigenvalue problem |
| 303 | Tuesday | Spherical Harmonics | $Y_\ell^m(\theta, \phi)$, orbital angular momentum eigenfunctions |
| 304 | Wednesday | Spin Angular Momentum | Spin-1/2, Pauli matrices, Stern-Gerlach experiment |
| 305 | Thursday | Addition of Angular Momenta | Tensor products, coupled/uncoupled bases |
| 306 | Friday | Clebsch-Gordan Coefficients | Calculation methods, tables, recursion relations |
| 307 | Saturday | Applications | Selection rules, atomic spectra, Wigner-Eckart theorem |
| 308 | Sunday | Month 11 Review | Complete group theory synthesis |

## Key Formulas

### Angular Momentum Algebra
$$[J_i, J_j] = i\hbar \epsilon_{ijk} J_k$$
$$[\mathbf{J}^2, J_i] = 0$$

### Eigenvalue Equations
$$\mathbf{J}^2 |j, m\rangle = \hbar^2 j(j+1) |j, m\rangle$$
$$J_z |j, m\rangle = \hbar m |j, m\rangle$$

### Ladder Operators
$$J_\pm = J_x \pm i J_y$$
$$J_\pm |j, m\rangle = \hbar\sqrt{j(j+1) - m(m \pm 1)} |j, m \pm 1\rangle$$

### Spherical Harmonics
$$Y_\ell^m(\theta, \phi) = (-1)^m \sqrt{\frac{2\ell + 1}{4\pi} \frac{(\ell - m)!}{(\ell + m)!}} P_\ell^m(\cos\theta) e^{im\phi}$$

### Clebsch-Gordan Decomposition
$$|j_1, j_2; j, m\rangle = \sum_{m_1, m_2} C_{j_1 m_1; j_2 m_2}^{jm} |j_1, m_1\rangle |j_2, m_2\rangle$$

### Angular Momentum Addition
$$j_1 \otimes j_2 = |j_1 - j_2| \oplus |j_1 - j_2| + 1 \oplus \cdots \oplus (j_1 + j_2)$$

## Physical Applications

### Atomic Physics
- Electron orbital states: $\ell = 0$ (s), $\ell = 1$ (p), $\ell = 2$ (d)
- Fine structure: spin-orbit coupling $\mathbf{L} \cdot \mathbf{S}$
- Selection rules: $\Delta \ell = \pm 1$, $\Delta m = 0, \pm 1$

### Particle Physics
- Spin states of fundamental particles
- Isospin multiplets
- Angular momentum in scattering

### Quantum Information
- Qubit as spin-1/2 system
- Two-qubit states and entanglement
- Angular momentum in quantum gates

## Computational Focus

This week emphasizes numerical methods for:
- Computing spherical harmonics
- Building spin matrices for arbitrary j
- Calculating Clebsch-Gordan coefficients
- Visualizing angular momentum states

## Prerequisites

- Week 43: Lie Groups (especially SU(2) representations)
- Week 42: Representation Theory
- Month 5: Complex analysis (spherical coordinates)
- Basic quantum mechanics concepts

## Resources

### Primary Texts
- Sakurai, *Modern Quantum Mechanics*, Chapter 3
- Griffiths, *Introduction to Quantum Mechanics*, Chapter 4
- Zettili, *Quantum Mechanics*, Chapters 5-7

### Mathematical References
- Tinkham, *Group Theory and Quantum Mechanics*, Chapter 5
- Varshalovich et al., *Quantum Theory of Angular Momentum*

### Computational Resources
- SymPy: `sympy.physics.quantum.spin`
- SciPy: `scipy.special.sph_harm`

## Progress Tracking

- [ ] Day 302: Angular Momentum in QM
- [ ] Day 303: Spherical Harmonics
- [ ] Day 304: Spin Angular Momentum
- [ ] Day 305: Addition of Angular Momenta
- [ ] Day 306: Clebsch-Gordan Coefficients
- [ ] Day 307: Applications
- [ ] Day 308: Month 11 Review

## Connection to Future Topics

Week 44 prepares for:
- **Month 12:** Capstone integration of all Year 0 concepts
- **Year 1:** Quantum mechanics formalism (Hilbert space, observables)
- **Year 2:** Atomic and molecular physics
- **Year 3:** Quantum field theory (Lorentz group representations)
