# Month 14: One-Dimensional Systems

## Overview

**Duration:** 28 days (Days 365-392)
**Position:** Year 1, Semester 1A, Month 2
**Theme:** Solving the Schrödinger Equation in One Dimension

This month applies the quantum formalism to concrete physical systems. We solve the time-independent Schrödinger equation for progressively more complex potentials, developing both analytical techniques and physical intuition for quantum behavior.

---

## STATUS: ✅ COMPLETE

| Week | Days | Topic | Status | Progress |
|------|------|-------|--------|----------|
| **Week 53** | 365-371 | Free Particle & Wave Packets | ✅ COMPLETE | 7/7 |
| **Week 54** | 372-378 | Bound States: Wells | ✅ COMPLETE | 7/7 |
| **Week 55** | 379-385 | Quantum Harmonic Oscillator | ✅ COMPLETE | 7/7 |
| **Week 56** | 386-392 | Tunneling & Barriers | ✅ COMPLETE | 7/7 |
| **Total** | 365-392 | One-Dimensional Systems | ✅ **COMPLETE** | **28/28** |

---

## Learning Objectives

By the end of Month 14, you will be able to:

1. **Solve** the free particle Schrödinger equation and interpret plane wave solutions
2. **Construct** and evolve Gaussian wave packets
3. **Derive** quantized energy levels for infinite and finite square wells
4. **Master** the algebraic (ladder operator) method for the harmonic oscillator
5. **Calculate** transmission and reflection coefficients for potential barriers
6. **Explain** quantum tunneling and its applications
7. **Connect** 1D quantum systems to real-world applications

---

## Weekly Breakdown

### Week 53: Free Particle & Wave Packets (Days 365-371)

| Day | Topic | Key Content |
|-----|-------|-------------|
| 365 | Free Particle TISE | ψ(x) = Ae^{ikx}, E = ℏ²k²/2m |
| 366 | Plane Waves & Normalization | Box normalization, delta normalization |
| 367 | Wave Packets I | Superposition of plane waves |
| 368 | Gaussian Wave Packets | Optimal uncertainty product |
| 369 | Wave Packet Dynamics | Group velocity, dispersion |
| 370 | Position & Momentum Space | Fourier transforms in QM |
| 371 | Week Review & Lab | Numerical wave packet evolution |

**Key Results:**
- Free particle: $E = \frac{\hbar^2 k^2}{2m}$, continuous spectrum
- Gaussian packet: $\Delta x \cdot \Delta p = \frac{\hbar}{2}$ (minimum uncertainty)
- Group velocity: $v_g = \frac{\partial \omega}{\partial k} = \frac{\hbar k}{m} = \frac{p}{m}$

---

### Week 54: Bound States — Infinite & Finite Wells (Days 372-378)

| Day | Topic | Key Content |
|-----|-------|-------------|
| 372 | Infinite Square Well | Standing waves, E_n = n²π²ℏ²/2mL² |
| 373 | ISW Eigenfunctions | ψ_n(x) = √(2/L) sin(nπx/L) |
| 374 | ISW Dynamics | Time evolution, revivals |
| 375 | Finite Square Well | Transcendental equations |
| 376 | FSW Bound States | Exponential tails, penetration depth |
| 377 | FSW Wave Functions | Matching conditions, parity |
| 378 | Week Review & Lab | Shooting method for bound states |

**Key Results:**
- Infinite well: $E_n = \frac{n^2 \pi^2 \hbar^2}{2mL^2}$, n = 1,2,3,...
- Finite well: Fewer bound states, wave function penetrates into classically forbidden region
- Penetration depth: $\kappa^{-1} = \frac{\hbar}{\sqrt{2m(V_0 - E)}}$

---

### Week 55: Quantum Harmonic Oscillator (Days 379-385)

| Day | Topic | Key Content |
|-----|-------|-------------|
| 379 | QHO Setup | V = ½mω²x², dimensionless variables |
| 380 | Ladder Operators | â = (mωx̂ + ip̂)/√(2mωℏ) |
| 381 | Number States | Ĥ = ℏω(N̂ + ½), \|n⟩ states |
| 382 | QHO Wave Functions | Hermite polynomials, ψ_n(x) |
| 383 | Coherent States | \|α⟩ = e^{-\|α\|²/2} Σ (α^n/√n!)\|n⟩ |
| 384 | QHO in Phase Space | Wigner function, classical limit |
| 385 | Week Review & Lab | Coherent state dynamics |

**Key Results:**
- Energy levels: $E_n = \hbar\omega(n + \frac{1}{2})$, equally spaced
- Ladder operators: $\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$, $\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$
- Zero-point energy: $E_0 = \frac{\hbar\omega}{2}$ (vacuum fluctuations)
- Coherent states: Minimum uncertainty, follow classical trajectory

---

### Week 56: Tunneling & Barriers (Days 386-392)

| Day | Topic | Key Content |
|-----|-------|-------------|
| 386 | Step Potential | Reflection and transmission |
| 387 | Rectangular Barrier | T, R coefficients |
| 388 | Tunneling Probability | WKB approximation preview |
| 389 | Alpha Decay | Gamow model, nuclear physics |
| 390 | Scanning Tunneling Microscope | Exponential sensitivity |
| 391 | Tunnel Diodes & Josephson | Solid-state applications |
| 392 | Month Review & Capstone | Integration and assessment |

**Key Results:**
- Tunneling probability: $T \approx e^{-2\kappa L}$ for rectangular barrier
- Gamow factor: $G = \frac{2}{\hbar}\int_a^b \sqrt{2m(V(x)-E)} dx$
- STM resolution: Exponential dependence enables atomic-scale imaging

---

## Primary References

### Textbooks
- **Shankar** "Principles of Quantum Mechanics" Ch. 5-7 (Primary)
- **Sakurai** "Modern Quantum Mechanics" Ch. 2 (Supplementary)
- **Griffiths** "Introduction to QM" Ch. 2 (Reference)

### Key Sections
| Topic | Shankar | Sakurai | Griffiths |
|-------|---------|---------|-----------|
| Free Particle | §5.1 | §2.4 | §2.4 |
| Square Wells | §5.2 | §2.4 | §2.2 |
| Harmonic Oscillator | Ch. 7 | §2.3 | §2.3 |
| Tunneling | §5.4 | §2.4 | §2.5 |

### Video Resources
- MIT OCW 8.04 Lectures 6-10 (Zwiebach)
- Feynman Lectures Vol. III, Ch. 7-9
- Physics LibreTexts Quantum Mechanics

---

## Computational Labs

| Week | Lab Topic | Tools |
|------|-----------|-------|
| 53 | Wave packet evolution | NumPy, Matplotlib |
| 54 | Bound state shooting method | SciPy |
| 55 | Coherent state animation | QuTiP |
| 56 | Tunneling probability calculation | NumPy |

---

## Quantum Computing Connections

| Classical QM Concept | Quantum Computing Application |
|---------------------|------------------------------|
| Two-level well | Qubit as effective two-level system |
| Harmonic oscillator | Bosonic qubits, continuous-variable QC |
| Tunneling | Quantum annealing, optimization |
| Coherent states | Quantum optics, photonic qubits |

---

## Assessment Milestones

### Week 53 Checkpoint
- [ ] Derive free particle dispersion relation
- [ ] Calculate Gaussian wave packet spread
- [ ] Implement numerical wave packet evolution

### Week 54 Checkpoint
- [ ] Derive infinite well energy levels from boundary conditions
- [ ] Solve transcendental equation for finite well graphically
- [ ] Implement shooting method for bound states

### Week 55 Checkpoint
- [ ] Prove [â, â†] = 1 from [x̂, p̂] = iℏ
- [ ] Derive ground state wave function from âψ₀ = 0
- [ ] Construct and animate coherent states

### Week 56 Checkpoint
- [ ] Calculate transmission coefficient for rectangular barrier
- [ ] Apply Gamow model to α-decay half-lives
- [ ] Explain STM working principle

---

## Directory Structure

```
Month_14_One_Dimensional/
├── README.md                              # This file
├── Week_53_Free_Particle_Waves/
│   ├── README.md
│   └── Day_365-371_*.md                   # 7 day files
├── Week_54_Bound_States/
│   ├── README.md
│   └── Day_372-378_*.md                   # 7 day files
├── Week_55_Harmonic_Oscillator/
│   ├── README.md
│   └── Day_379-385_*.md                   # 7 day files
└── Week_56_Tunneling/
    ├── README.md
    └── Day_386-392_*.md                   # 7 day files
```

---

## Prerequisites from Month 13

| Month 13 Topic | Month 14 Application |
|----------------|---------------------|
| Hilbert space | State space for bound states |
| Operators | Hamiltonian, ladder operators |
| Eigenvalue problems | Energy quantization |
| Time evolution | Wave packet dynamics |
| Position/momentum | Wave function representations |

---

## Preview: Month 15

Next month covers **Angular Momentum & Spin**, where we:
- Solve the angular part of 3D problems
- Discover quantization of angular momentum
- Introduce spin as intrinsic angular momentum
- Master addition of angular momenta

---

*"The quantum harmonic oscillator is to quantum mechanics what the quadratic function is to calculus — the simplest nontrivial example that teaches you everything."*

---

**Created:** February 2, 2026
**Status:** In Progress
