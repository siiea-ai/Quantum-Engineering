# Semester 1A: Quantum Mechanics Foundations

## Overview

**Semester 1A** | Months 13-18 | Days 337-504 | Quantum Mechanics Core

This semester establishes the theoretical foundations of quantum mechanics at graduate level. Starting from the postulates of quantum mechanics and building through increasingly sophisticated applications, this corresponds to MIT's 8.04/8.05 sequence and Harvard's Physics 143a.

---

## Status: ✅ COMPLETE

| Month | Topic | Weeks | Days | Status |
|-------|-------|-------|------|--------|
| **13** | Postulates & Mathematical Framework | 49-52 | 337-364 | ✅ Complete |
| **14** | One-Dimensional Systems | 53-56 | 365-392 | ✅ Complete |
| **15** | Angular Momentum & Spin | 57-60 | 393-420 | ✅ Complete |
| **16** | Three-Dimensional Problems | 61-64 | 421-448 | ✅ Complete |
| **17** | Perturbation Theory & Approximations | 65-68 | 449-476 | ✅ Complete |
| **18** | Identical Particles & Many-Body | 69-72 | 477-504 | ✅ Complete |

**Total:** 168/168 days (100%) | 24 weeks | 6 months

---

## Primary Texts

| Text | Author(s) | Role |
|------|-----------|------|
| **Principles of Quantum Mechanics** | R. Shankar (2nd ed.) | Primary textbook |
| **Modern Quantum Mechanics** | Sakurai & Napolitano (3rd ed.) | Advanced problems |
| **Introduction to Quantum Mechanics** | D.J. Griffiths (3rd ed.) | Reference |

---

## Learning Objectives

Upon completing Semester 1A, students will be able to:

### Foundations
1. State and apply the postulates of quantum mechanics
2. Work fluently with Dirac notation and Hilbert spaces
3. Calculate expectation values, uncertainties, and commutators
4. Solve the time-dependent and time-independent Schrödinger equation

### Applications
5. Analyze bound state problems (wells, harmonic oscillator)
6. Understand tunneling and scattering phenomena
7. Master angular momentum algebra (orbital and spin)
8. Solve the hydrogen atom and understand fine structure

### Advanced Topics
9. Apply perturbation theory (time-independent and time-dependent)
10. Use variational and WKB methods
11. Understand identical particles and exchange symmetry
12. Work with second quantization formalism

---

## Month Summaries

### Month 13: Postulates & Mathematical Framework (Days 337-364)

**Primary Reference:** Shankar Chapters 1, 4

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 49 | 337-343 | Hilbert Space Formalism | Dirac notation, inner products, complete bases |
| 50 | 344-350 | Observables & Measurement | Hermitian operators, eigenvalues, measurement postulate |
| 51 | 351-357 | Uncertainty & Commutators | Heisenberg uncertainty, generalized uncertainty relations |
| 52 | 358-364 | Time Evolution | Schrödinger equation, propagators, pictures |

### Month 14: One-Dimensional Systems (Days 365-392)

**Primary Reference:** Shankar Chapters 5-7

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 53 | 365-371 | Free Particle & Wave Packets | Plane waves, Gaussian packets, dispersion |
| 54 | 372-378 | Bound States - Wells | Infinite/finite wells, energy quantization |
| 55 | 379-385 | Quantum Harmonic Oscillator | Ladder operators, coherent states |
| 56 | 386-392 | Tunneling & Barriers | Transmission coefficients, alpha decay |

### Month 15: Angular Momentum & Spin (Days 393-420)

**Primary Reference:** Shankar Chapters 12-14

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 57 | 393-399 | Orbital Angular Momentum | L², Lz, spherical harmonics |
| 58 | 400-406 | Spin Angular Momentum | Spin-1/2, Pauli matrices, spinors |
| 59 | 407-413 | Addition of Angular Momenta | Clebsch-Gordan coefficients |
| 60 | 414-420 | Rotations & Wigner D-Matrices | SO(3)/SU(2), rotation operators |

### Month 16: Three-Dimensional Problems (Days 421-448)

**Primary Reference:** Shankar Chapters 10, 13

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 61 | 421-427 | Central Potentials | Radial equation, effective potential |
| 62 | 428-434 | Hydrogen Atom | Laguerre polynomials, energy levels |
| 63 | 435-441 | Fine Structure | Relativistic corrections, spin-orbit coupling |
| 64 | 442-448 | Hyperfine & External Fields | Nuclear effects, Zeeman/Stark effects |

### Month 17: Perturbation Theory & Approximations (Days 449-476)

**Primary Reference:** Shankar Chapter 17

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 65 | 449-455 | Time-Independent Perturbation Theory | Non-degenerate, degenerate cases |
| 66 | 456-462 | Variational Method | Ground state bounds, trial wave functions |
| 67 | 463-469 | WKB Approximation | Classical limit, connection formulas |
| 68 | 470-476 | Time-Dependent Perturbation Theory | Fermi's golden rule, transitions |

### Month 18: Identical Particles & Many-Body (Days 477-504)

**Primary Reference:** Shankar Chapter 10

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 69 | 477-483 | Identical Particles | Symmetrization, fermions/bosons |
| 70 | 484-490 | Second Quantization | Creation/annihilation operators |
| 71 | 491-497 | Many-Body Systems | Hartree-Fock, correlation |
| 72 | 498-504 | Scattering Theory & Capstone | Cross sections, partial waves |

---

## Computational Labs

### Physics Simulations (QuTiP/NumPy)
- Wave packet dynamics and dispersion
- Quantum harmonic oscillator eigenstates
- Angular momentum coupling visualization
- Hydrogen atom orbital plotting
- Perturbation theory numerical verification
- Scattering phase shift analysis

---

## Key Formulas Reference

### Postulates
$$\hat{H}|\psi\rangle = i\hbar\frac{\partial}{\partial t}|\psi\rangle$$

### Uncertainty Relations
$$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$$

### Angular Momentum
$$\hat{L}^2|l,m\rangle = \hbar^2 l(l+1)|l,m\rangle, \quad \hat{L}_z|l,m\rangle = \hbar m|l,m\rangle$$

### Ladder Operators
$$\hat{L}_{\pm}|l,m\rangle = \hbar\sqrt{l(l+1)-m(m\pm 1)}|l,m\pm 1\rangle$$

### Hydrogen Energy Levels
$$E_n = -\frac{13.6 \text{ eV}}{n^2}$$

---

## Prerequisites

Completion of Year 0 required, specifically:
- Linear algebra (eigenvalues, operators)
- Complex analysis (wave function manipulation)
- Differential equations (Schrödinger equation)
- Classical mechanics (Hamiltonian formulation)
- Functional analysis (Hilbert spaces)
- Group theory (angular momentum algebra)

---

## Directory Structure

```
Semester_1A_QM_Foundations/
├── README.md                            # This file
├── Month_13_Postulates_Framework/
│   ├── README.md
│   ├── Week_49_Hilbert_Space_Formalism/
│   ├── Week_50_Observables_Measurement/
│   ├── Week_51_Uncertainty_Commutators/
│   └── Week_52_Time_Evolution/
├── Month_14_One_Dimensional/
│   ├── README.md
│   ├── Week_53_Free_Particle_Wave_Packets/
│   ├── Week_54_Bound_States_Wells/
│   ├── Week_55_Quantum_Harmonic_Oscillator/
│   └── Week_56_Tunneling_Barriers/
├── Month_15_Angular_Momentum/
│   ├── README.md
│   ├── Week_57_Orbital_Angular_Momentum/
│   ├── Week_58_Spin_Angular_Momentum/
│   ├── Week_59_Addition_Angular_Momenta/
│   └── Week_60_Rotations_Wigner/
├── Month_16_Three_Dimensional/
│   ├── README.md
│   ├── Week_61_Central_Potentials/
│   ├── Week_62_Hydrogen_Atom/
│   ├── Week_63_Fine_Structure/
│   └── Week_64_Hyperfine_External_Fields/
├── Month_17_Perturbation_Theory/
│   ├── README.md
│   ├── Week_65_Time_Independent_Perturbation/
│   ├── Week_66_Variational_Method/
│   ├── Week_67_WKB_Approximation/
│   └── Week_68_Time_Dependent_Perturbation/
└── Month_18_Many_Body/
    ├── README.md
    ├── Week_69_Identical_Particles/
    ├── Week_70_Second_Quantization/
    ├── Week_71_Many_Body_Systems/
    └── Week_72_Scattering_Theory/
```

---

## References

### Primary Texts
- Shankar, R. (2011). *Principles of Quantum Mechanics* (2nd Edition). Springer.
- Sakurai, J. J., & Napolitano, J. (2017). *Modern Quantum Mechanics* (3rd Edition). Cambridge.

### Supplementary
- Griffiths, D. J., & Schroeter, D. F. (2018). *Introduction to Quantum Mechanics* (3rd Edition). Cambridge.
- Cohen-Tannoudji, C., Diu, B., & Laloë, F. (2019). *Quantum Mechanics* (Volumes 1-3). Wiley.

### Online Resources
- MIT OCW 8.04/8.05: https://ocw.mit.edu
- Barton Zwiebach's Lectures: https://ocw.mit.edu/8-04
- David Tong's QM Notes: http://www.damtp.cam.ac.uk/user/tong/quantum.html

---

**Start Date:** Day 337
**End Date:** Day 504
**Duration:** 168 days (6 months)
**Status:** ✅ COMPLETE

---

*Next: Semester 1B — Quantum Information Foundations*
