# Week 53: Free Particle & Wave Packets (Days 365-371)

## Overview

This week marks a pivotal transition in our quantum mechanics journey. We move from the abstract formalism of Hilbert spaces, operators, and the postulates to our first concrete physical system: the **free particle**. Though deceptively simple—a particle with no forces acting on it—this system reveals profound aspects of quantum mechanics including continuous spectra, wave packet dynamics, and the deep connection between position and momentum representations.

The free particle serves as the foundation for understanding scattering theory, tunneling phenomena, and the behavior of matter waves in free space. The wave packet formalism developed here underlies quantum computing's understanding of qubit evolution and quantum communication protocols.

**Primary Reference:** Shankar, *Principles of Quantum Mechanics*, Chapter 5

---

## Weekly Schedule

| Day | Topic | Focus Areas |
|-----|-------|-------------|
| 365 (Mon) | Free Particle TISE | Time-independent Schrödinger equation, plane wave solutions, dispersion relation |
| 366 (Tue) | Plane Waves & Normalization | Box normalization, delta-function normalization, momentum eigenstates |
| 367 (Wed) | Wave Packets I | Superposition principle, general wave packets, Fourier analysis |
| 368 (Thu) | Gaussian Wave Packets | Gaussian states, minimum uncertainty, special properties |
| 369 (Fri) | Wave Packet Dynamics | Time evolution, phase/group velocity, dispersion and spreading |
| 370 (Sat) | Position & Momentum Space | Fourier transforms, operator representations, dual descriptions |
| 371 (Sun) | Week Review & Lab | Synthesis, comprehensive lab, practice problems |

---

## Learning Objectives

By the end of this week, you will be able to:

1. **Solve the free particle Schrödinger equation** and interpret plane wave solutions
2. **Apply normalization conventions** (box and delta-function) to continuous spectrum states
3. **Construct wave packets** from superpositions of plane waves
4. **Analyze Gaussian wave packets** and their minimum uncertainty property
5. **Calculate group and phase velocities** and explain wave packet spreading
6. **Transform between position and momentum representations** using Fourier analysis
7. **Simulate wave packet dynamics** numerically and interpret the results
8. **Connect classical and quantum descriptions** through the correspondence principle

---

## Key Formulas

### Free Particle Schrödinger Equation

$$\boxed{-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi}$$

### Plane Wave Solutions

$$\boxed{\psi_k(x) = Ae^{ikx}, \quad k = \pm\frac{\sqrt{2mE}}{\hbar}}$$

### Dispersion Relation

$$\boxed{E = \frac{\hbar^2 k^2}{2m} = \frac{p^2}{2m}}$$

### General Wave Packet

$$\boxed{\psi(x,t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \phi(k)e^{i(kx - \omega(k)t)}dk}$$

### Gaussian Wave Packet (t=0)

$$\boxed{\psi(x,0) = \left(\frac{1}{2\pi\sigma^2}\right)^{1/4}e^{ik_0 x}e^{-x^2/4\sigma^2}}$$

### Group and Phase Velocity

$$\boxed{v_g = \frac{d\omega}{dk} = \frac{\hbar k}{m} = \frac{p}{m} = v_{\text{classical}}}$$

$$\boxed{v_p = \frac{\omega}{k} = \frac{\hbar k}{2m} = \frac{v_{\text{classical}}}{2}}$$

### Fourier Transform Pairs

$$\boxed{\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}\psi(x)dx}$$

$$\boxed{\psi(x) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{ipx/\hbar}\phi(p)dp}$$

### Minimum Uncertainty (Gaussian)

$$\boxed{\Delta x \cdot \Delta p = \frac{\hbar}{2}}$$

---

## Mathematical Prerequisites

- **Fourier analysis:** Transforms, convolutions, Parseval's theorem
- **Complex analysis:** Gaussian integrals, contour integration basics
- **Linear algebra:** Continuous bases, delta function normalization
- **Calculus:** Improper integrals, limiting procedures

---

## Physical Concepts

### Why the Free Particle Matters

1. **Scattering theory foundation:** Incident and transmitted waves are free particle states
2. **Tunneling phenomena:** Free particle regions before/after barriers
3. **Matter wave propagation:** De Broglie waves in free space
4. **Quantum communication:** Photon wave packets in quantum key distribution

### Classical Correspondence

The free particle beautifully illustrates the correspondence principle:
- Group velocity equals classical velocity
- Wave packet center follows classical trajectory
- Spreading reveals purely quantum behavior

---

## Quantum Computing Connections

| Classical QM Concept | Quantum Computing Application |
|---------------------|------------------------------|
| Plane waves | Momentum eigenstates in continuous-variable QC |
| Wave packets | Photon pulse shaping in quantum communication |
| Fourier transform | Quantum Fourier Transform algorithm |
| Group velocity | Signal propagation in quantum networks |
| Dispersion | Pulse distortion in quantum channels |

---

## Computational Tools

This week's Python labs use:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
```

Key numerical techniques:
- Fast Fourier Transform (FFT) for momentum space
- Numerical integration for wave packet construction
- Animation for time evolution visualization
- Gaussian fitting and analysis

---

## Reading List

### Primary
- Shankar, Chapter 5: "Simple Problems in One Dimension" (Sections 5.1-5.2)

### Supplementary
- Griffiths, Chapter 2: "Time-Independent Schrödinger Equation" (Section 2.4)
- Cohen-Tannoudji, Chapter I: "Waves and Particles"
- Sakurai, Chapter 2: "Quantum Dynamics" (Wave packets section)

### Online Resources
- MIT OCW 8.04: Lecture notes on wave packets
- Physics LibreTexts: Wave packet dynamics
- arXiv: Recent papers on ultrafast wave packet control

---

## Assessment Criteria

### Conceptual Understanding
- [ ] Explain why free particle has continuous spectrum
- [ ] Distinguish phase velocity from group velocity
- [ ] Interpret wave packet spreading physically
- [ ] Connect Fourier transforms to uncertainty principle

### Mathematical Skills
- [ ] Solve TISE for free particle
- [ ] Apply normalization conventions correctly
- [ ] Perform Fourier transforms of Gaussian packets
- [ ] Calculate expectation values in both representations

### Computational Proficiency
- [ ] Implement FFT-based momentum space calculations
- [ ] Animate wave packet evolution
- [ ] Verify uncertainty relations numerically
- [ ] Analyze dispersion effects quantitatively

---

## Week Preview

**Week 54** will introduce the first potential: the infinite square well (particle in a box). We'll see how boundary conditions lead to quantization—discrete energy levels replacing the continuous spectrum of the free particle. This sets the stage for understanding atomic orbitals and quantum confinement in semiconductors.

---

*"The wave packet is the quantum mechanical compromise between the particle picture and the wave picture—it moves like a particle but spreads like a wave."*
