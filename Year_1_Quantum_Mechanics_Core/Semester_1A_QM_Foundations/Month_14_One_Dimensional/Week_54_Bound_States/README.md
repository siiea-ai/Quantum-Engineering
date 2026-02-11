# Week 54: Bound States - Infinite and Finite Square Wells

## Overview

**Days 372-378 | Month 14: One-Dimensional Systems | Semester 1A**

This week provides a comprehensive treatment of bound state problems in one-dimensional quantum mechanics. We begin with the infinite square well (particle in a box) - the quintessential exactly solvable quantum system - and progress to the more realistic finite square well that introduces the profound phenomenon of quantum tunneling into classically forbidden regions.

These systems form the foundation for understanding:
- Energy quantization in confined systems
- Quantum confinement effects in nanostructures
- Semiconductor quantum wells and quantum dots
- Qubit implementations in superconducting circuits

---

## Weekly Schedule

| Day | Date | Topic | Focus Area |
|-----|------|-------|------------|
| **372** | Monday | Infinite Square Well Setup | Boundary conditions, quantization |
| **373** | Tuesday | ISW Eigenfunctions | Orthonormality, completeness |
| **374** | Wednesday | ISW Dynamics | Time evolution, quantum revivals |
| **375** | Thursday | Finite Square Well | Transcendental equations |
| **376** | Friday | FSW Bound States | Penetration depth, parity |
| **377** | Saturday | FSW Matching Conditions | Continuity, shooting method |
| **378** | Sunday | Week Review & Comprehensive Lab | Integration, assessment |

---

## Learning Objectives

By the end of this week, you will be able to:

### Infinite Square Well
1. Derive energy eigenvalues $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$ from boundary conditions
2. Construct normalized eigenfunctions $\psi_n(x) = \sqrt{2/L}\sin(n\pi x/L)$
3. Verify orthonormality and completeness of the energy eigenbasis
4. Calculate quantum revival times and understand wave packet dynamics
5. Compute expectation values and uncertainties for arbitrary states

### Finite Square Well
6. Set up and solve the transcendental eigenvalue equations
7. Explain wave function penetration into classically forbidden regions
8. Distinguish even and odd parity solutions
9. Relate well depth to the number of bound states
10. Apply boundary matching conditions systematically
11. Implement numerical shooting methods for eigenvalue problems

---

## Key Formulas

### Infinite Square Well

| Quantity | Formula |
|----------|---------|
| **Potential** | $V(x) = 0$ for $0 < x < L$, $V(x) = \infty$ otherwise |
| **Energy Eigenvalues** | $$\boxed{E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}, \quad n = 1, 2, 3, \ldots}$$ |
| **Eigenfunctions** | $$\boxed{\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)}$$ |
| **Wave Number** | $k_n = \frac{n\pi}{L}$ |
| **Ground State Energy** | $E_1 = \frac{\pi^2\hbar^2}{2mL^2}$ |
| **Energy Ratio** | $E_n = n^2 E_1$ |
| **Orthonormality** | $\langle\psi_m|\psi_n\rangle = \delta_{mn}$ |
| **Completeness** | $\sum_{n=1}^{\infty} |\psi_n\rangle\langle\psi_n| = \mathbb{1}$ |
| **Revival Time** | $$\boxed{T_{\text{rev}} = \frac{4mL^2}{\pi\hbar}}$$ |

### Finite Square Well

| Quantity | Formula |
|----------|---------|
| **Potential** | $V(x) = -V_0$ for $|x| < a$, $V(x) = 0$ otherwise |
| **Inside Well** | $\psi'' = -k^2\psi$, where $k^2 = \frac{2m(E + V_0)}{\hbar^2}$ |
| **Outside Well** | $\psi'' = \kappa^2\psi$, where $\kappa^2 = \frac{-2mE}{\hbar^2}$ |
| **Constraint** | $k^2 + \kappa^2 = \frac{2mV_0}{\hbar^2}$ |
| **Even Parity** | $$\boxed{k\tan(ka) = \kappa}$$ |
| **Odd Parity** | $$\boxed{-k\cot(ka) = \kappa}$$ |
| **Penetration Depth** | $$\boxed{\delta = \frac{1}{\kappa} = \frac{\hbar}{\sqrt{2m|E|}}}$$ |
| **Dimensionless Parameter** | $z_0 = \frac{a}{\hbar}\sqrt{2mV_0}$ |
| **Number of Bound States** | $N \approx \left\lfloor \frac{z_0}{\pi/2} \right\rfloor + 1$ |

---

## Physical Insights

### Quantum vs Classical Confinement

| Aspect | Classical Particle | Quantum Particle |
|--------|-------------------|------------------|
| Energy | Continuous | Quantized ($E_n \propto n^2$) |
| Minimum energy | $E = 0$ allowed | $E_1 > 0$ (zero-point energy) |
| Position | Definite | Probability distribution |
| Momentum | Definite | Superposition of $\pm\hbar k_n$ |
| At boundaries | Hard reflection | Probability vanishes smoothly |
| Beyond barriers | Impossible | Exponential penetration |

### Zero-Point Energy

The ground state energy $E_1 > 0$ is a direct consequence of the uncertainty principle:
$$\Delta x \sim L \implies \Delta p \gtrsim \frac{\hbar}{L} \implies E \gtrsim \frac{(\Delta p)^2}{2m} \sim \frac{\hbar^2}{2mL^2}$$

### Quantum Confinement Scaling

For a particle in a box of size $L$:
- Energy scales as $E \propto L^{-2}$ (smaller box = higher energy)
- Quantum effects dominate when $L \lesssim \lambda_{\text{dB}}$ (de Broglie wavelength)
- This explains color changes in semiconductor quantum dots with size

---

## Quantum Computing Connections

### Semiconductor Quantum Dots
- Electrons confined in ~10 nm regions behave as particles in 3D boxes
- Discrete energy levels enable qubit implementations
- Spin states in quantum dots are leading qubit candidates

### Superconducting Qubits
- Transmon qubits approximate anharmonic oscillators derived from finite well physics
- Energy level spacing engineered via junction parameters
- Two lowest levels form computational qubit states

### Gate-Defined Quantum Dots
- Electrostatic gates create tunable potential wells
- Single electron transistors for charge sensing
- Exchange coupling between adjacent dots for two-qubit gates

---

## Required Background

### From Year 0
- Linear algebra: eigenvalue problems, orthonormal bases (Month 4)
- Differential equations: boundary value problems (Month 3)
- Fourier series and completeness (Month 5)
- Complex analysis: transcendental equations (Month 7)

### From Earlier Year 1
- Schrodinger equation (Week 49)
- Probability interpretation (Week 50)
- Operators and observables (Week 51)
- Free particle solutions (Week 53)

---

## Primary References

### Textbooks
- **Shankar**, *Principles of Quantum Mechanics*, Chapter 5: Simple Problems in One Dimension
- **Griffiths**, *Introduction to Quantum Mechanics*, Chapter 2: Time-Independent Schrodinger Equation
- **Cohen-Tannoudji**, *Quantum Mechanics*, Complement $H_I$: Particle in a One-Dimensional Square Well

### Online Resources
- MIT 8.04 Lecture Notes on Bound States
- Physics LibreTexts: Particle in a Box
- Hyperphysics: Finite Square Well

---

## Computational Tools

This week uses Python with:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import eigh_tridiagonal
import matplotlib.animation as animation
```

### Key Numerical Methods
- **Graphical solution**: Finding roots of transcendental equations
- **Shooting method**: Numerical eigenvalue determination
- **Matrix diagonalization**: Discrete approximation to continuum problem
- **Wave packet animation**: Visualizing quantum dynamics

---

## Assessment

### Problem Sets
Each day includes tiered problems:
- **Level 1**: Direct application (verify formulas, simple calculations)
- **Level 2**: Intermediate (derivations, multi-step problems)
- **Level 3**: Challenging (conceptual, numerical, research-connected)

### Computational Labs
Daily Python exercises culminating in Sunday's comprehensive lab:
- Shooting method eigenvalue solver
- Wave function visualization with penetration
- Comparison of finite and infinite well limits
- Quantum revival animations

### Self-Assessment Checklist
Track mastery of:
- [ ] ISW energy quantization derivation
- [ ] Eigenfunction orthonormality proof
- [ ] Time evolution and revival calculation
- [ ] FSW transcendental equation setup
- [ ] Boundary matching conditions
- [ ] Penetration depth interpretation
- [ ] Shooting method implementation

---

## Preview of Week 55

**Week 55: Quantum Harmonic Oscillator** introduces the most important exactly solvable system in all of quantum mechanics. We will develop both:
- **Analytic solution**: Hermite polynomials and Gaussian ground state
- **Algebraic solution**: Ladder operators $\hat{a}$ and $\hat{a}^\dagger$

The harmonic oscillator appears everywhere: molecular vibrations, phonons in solids, electromagnetic field modes, and forms the basis for understanding coherent states and quantum optics.

---

*Week 54 of QSE Self-Study Curriculum*
*Month 14: One-Dimensional Systems*
*Year 1: Quantum Mechanics Core*
