# Month 38: QM Foundations Review II - Angular Momentum, Spin, and Perturbation Theory

## Overview

**Duration:** 28 days (Days 1037-1064)
**Weeks:** 149-152
**Theme:** Angular momentum, spin physics, and perturbation methods for qualifying exam mastery

---

## Status: COMPLETE

| Week | Days | Topic | Status | Progress |
|------|------|-------|--------|----------|
| **149** | 1037-1043 | Orbital Angular Momentum | Complete | 7/7 |
| **150** | 1044-1050 | Spin & Magnetic Interactions | Complete | 7/7 |
| **151** | 1051-1057 | Angular Momentum Addition | Complete | 7/7 |
| **152** | 1058-1064 | Perturbation Theory | Complete | 7/7 |

**Total Progress:** 28/28 days (100%)

---

## Month Learning Arc

This month covers three fundamental pillars of quantum mechanics that appear extensively on PhD qualifying exams: angular momentum theory, spin physics, and perturbation methods. These topics interconnect deeply and form the basis for understanding atomic structure, selection rules, and quantum computing.

### Week-by-Week Progression

```
Week 149: Orbital Angular Momentum
├── Angular momentum algebra and commutation relations
├── Eigenvalue problem for L² and Lz
├── Spherical harmonics as eigenfunctions
├── Central potential problems and hydrogen atom
└── Qualifying exam problem techniques

Week 150: Spin & Magnetic Interactions
├── Intrinsic angular momentum and spin-1/2
├── Pauli matrices and spinor algebra
├── Stern-Gerlach experiment and measurement
├── Spin precession in magnetic fields
└── Magnetic moments and coupling

Week 151: Angular Momentum Addition
├── Tensor product spaces for composite systems
├── Clebsch-Gordan coefficients and derivation
├── Spin-orbit coupling in atoms
├── Selection rules for transitions
└── Term symbols and atomic spectroscopy

Week 152: Perturbation Theory
├── Non-degenerate time-independent perturbation
├── Degenerate perturbation theory
├── Time-dependent perturbation and transitions
├── Fermi's golden rule applications
└── Adiabatic theorem and Berry phase
```

---

## Core Equations to Master

### Angular Momentum Algebra

$$\boxed{[L_i, L_j] = i\hbar\epsilon_{ijk}L_k}$$

$$\boxed{L^2|l,m\rangle = \hbar^2 l(l+1)|l,m\rangle}$$

$$\boxed{L_z|l,m\rangle = \hbar m|l,m\rangle}$$

$$\boxed{L_{\pm}|l,m\rangle = \hbar\sqrt{l(l+1)-m(m\pm 1)}|l,m\pm 1\rangle}$$

### Spherical Harmonics

$$Y_l^m(\theta,\phi) = (-1)^m\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\phi}$$

### Spin-1/2 Operators

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

$$\boxed{\mathbf{S} = \frac{\hbar}{2}\boldsymbol{\sigma}}$$

### Clebsch-Gordan Coefficients

$$|j,m\rangle = \sum_{m_1,m_2} \langle j_1,m_1;j_2,m_2|j,m\rangle |j_1,m_1\rangle|j_2,m_2\rangle$$

### Perturbation Theory

**First-Order Energy Correction:**
$$\boxed{E_n^{(1)} = \langle n^{(0)}|H'|n^{(0)}\rangle}$$

**Second-Order Energy Correction:**
$$\boxed{E_n^{(2)} = \sum_{k\neq n}\frac{|\langle k^{(0)}|H'|n^{(0)}\rangle|^2}{E_n^{(0)}-E_k^{(0)}}}$$

**Fermi's Golden Rule:**
$$\boxed{\Gamma_{i\to f} = \frac{2\pi}{\hbar}|\langle f|H'|i\rangle|^2\rho(E_f)}$$

---

## Qualifying Exam Focus Areas

### High-Frequency Problem Types

Based on analysis of Yale, MIT, Caltech, and CUNY qualifying exams:

| Topic | Problem Type | Frequency |
|-------|-------------|-----------|
| Angular Momentum | Eigenvalue calculations | Very High |
| Spherical Harmonics | Orbital shapes, selection rules | High |
| Spin-1/2 | Measurement outcomes, rotations | Very High |
| Clebsch-Gordan | Coefficient calculations | High |
| Fine Structure | Spin-orbit coupling energy | Medium-High |
| Perturbation Theory | Energy corrections | Very High |
| Stark/Zeeman Effect | Applied field problems | High |
| Fermi's Golden Rule | Transition rates | Medium-High |
| Berry Phase | Geometric phase calculation | Medium |

### Common Oral Exam Questions

1. "Derive the angular momentum commutation relations from the position-momentum commutators."
2. "Explain why spin-1/2 particles require 4π rotation to return to their original state."
3. "Walk me through how you would calculate the fine structure of hydrogen."
4. "What happens when you apply degenerate perturbation theory? Give an example."
5. "Explain Fermi's golden rule and its assumptions."

---

## Weekly Schedule Template

### Daily Structure (7.5 hours)

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Problem solving (exam-style) |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Topic review, derivations |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Oral practice, flash cards |

### Weekly Distribution

- **Day 1-2:** Review guide study, key derivations
- **Day 3-4:** Problem set work (timed, exam conditions)
- **Day 5:** Solutions review, gap identification
- **Day 6:** Oral practice, explaining concepts aloud
- **Day 7:** Self-assessment, integration with previous weeks

---

## Week Content Overview

### Week 149: Orbital Angular Momentum

**Key Topics:**
- Angular momentum operators and commutation relations
- Ladder operators and eigenvalue spectrum
- Spherical harmonics: derivation, properties, visualization
- Central potential and separation of variables
- Hydrogen atom quantum numbers and degeneracy

**Sample Qualifying Problem:**
> Calculate the expectation value of $L_x$ for a hydrogen atom in the state $|2,1,1\rangle$. Then find the probability of measuring $L_x = +\hbar$.

### Week 150: Spin & Magnetic Interactions

**Key Topics:**
- Spin-1/2 formalism and Pauli matrices
- Spinor states and rotations
- Stern-Gerlach apparatus analysis
- Spin precession (Larmor frequency)
- Magnetic dipole moment and coupling

**Sample Qualifying Problem:**
> A spin-1/2 particle is prepared in state $|\uparrow_z\rangle$ and placed in a magnetic field $\mathbf{B} = B_0(\cos\omega t\, \hat{x} + \sin\omega t\, \hat{y})$. Find the probability of measuring spin-down along z at time t.

### Week 151: Angular Momentum Addition

**Key Topics:**
- Tensor product of angular momentum states
- Clebsch-Gordan coefficient derivation
- Recursion relations and tables
- Spin-orbit coupling: $\mathbf{L}\cdot\mathbf{S}$ operator
- Fine structure energy splitting

**Sample Qualifying Problem:**
> Couple $j_1 = 1$ and $j_2 = 1/2$. Find all Clebsch-Gordan coefficients and express $|j=3/2, m=1/2\rangle$ in terms of uncoupled basis states.

### Week 152: Perturbation Theory

**Key Topics:**
- Non-degenerate perturbation: first and second order
- Degenerate perturbation: diagonalization in degenerate subspace
- Time-dependent perturbation theory
- Fermi's golden rule and transition rates
- Adiabatic theorem and Berry phase

**Sample Qualifying Problem:**
> Consider a hydrogen atom in a uniform electric field $\mathbf{E} = E_0\hat{z}$. Calculate the first-order Stark effect splitting of the $n=2$ level. Which states mix?

---

## Key References

### Primary Textbooks
- **Shankar**, *Principles of Quantum Mechanics*, Chapters 12-17
- **Sakurai**, *Modern Quantum Mechanics*, Chapters 3-5
- **Griffiths**, *Introduction to Quantum Mechanics*, Chapters 4, 7

### Problem Collections
- [Yale Physics Qualifying Exams](https://physics.yale.edu/academics/graduate-studies/graduate-student-handbook/qualifying-exam-past-exams)
- [CUNY Graduate Center QM Solutions](https://www.gc.cuny.edu/sites/default/files/2022-06/SOLUTIONS-QUANTUM-MECHANICS-AUG-2020-THRU-JUNE-2022-DM-FINAL.pdf)
- *Problems and Solutions on Quantum Mechanics* (World Scientific)
- [MIT 8.06 Perturbation Theory Notes](https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/)

### Supplementary Materials
- [Physics LibreTexts: Fine Structure of Hydrogen](https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Introductory_Quantum_Mechanics_(Fitzpatrick)/11:_Time-Independent_Perturbation_Theory/11.08:_Fine_Structure_of_Hydrogen)
- [Spherical Harmonics](https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Introductory_Quantum_Mechanics_(Fitzpatrick)/07:_Orbital_Angular_Momentum/7.06:_Spherical_Harmonics)
- [Cambridge Perturbation Theory](https://www.damtp.cam.ac.uk/user/dbs26/PQM/chap8.pdf)

---

## Assessment Checkpoints

### Mid-Month Check (Day 1050)
- [ ] Can derive angular momentum commutation relations
- [ ] Can solve hydrogen atom eigenvalue problems
- [ ] Can calculate spin measurement probabilities
- [ ] Can use Clebsch-Gordan coefficients

### End-of-Month Assessment (Day 1064)
- [ ] Written exam: 10 problems, 3 hours (target: 80%)
- [ ] Oral exam: 30 minutes, 5 questions
- [ ] All problem sets completed
- [ ] Self-assessment checklists complete

---

## Connection to Other Topics

### From Month 37 (QM Foundations I)
| Previous Topic | This Month's Extension |
|----------------|----------------------|
| Hilbert space operators | Angular momentum operators |
| Eigenvalue problems | L², Lz eigenvalues |
| 1D bound states | 3D central potentials |

### To Month 39 (QM Foundations III)
| This Month's Foundation | Future Application |
|------------------------|-------------------|
| Angular momentum addition | Many-body systems |
| Perturbation theory | Scattering theory |
| Selection rules | Transition matrix elements |

### Quantum Computing Connections
| QM Topic | QC Application |
|----------|----------------|
| Spin-1/2 | Qubit states |
| Pauli matrices | Single-qubit gates |
| Angular momentum addition | Multi-qubit states |
| Time-dependent perturbation | Gate errors |

---

*"Angular momentum is the heart of quantum mechanics. Master it, and the rest follows."*
— Common PhD advisor wisdom

---

**Created:** February 9, 2026
**Status:** COMPLETE
**Progress:** 28/28 days (100%)
