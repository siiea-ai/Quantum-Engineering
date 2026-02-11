# Week 72: Scattering Theory

## Overview

**Week 72** | Days 498-504 | Quantum Scattering and Semester 1A Capstone

This final week of Semester 1A covers scattering theory—how quantum particles interact with potentials and each other. Scattering experiments are the primary tool for probing the microscopic world, from Rutherford's discovery of the nucleus to modern particle physics at CERN. We conclude with a comprehensive review of all Semester 1A material.

---

## Daily Schedule

| Day | Topic | Focus |
|-----|-------|-------|
| **498** | Scattering Formalism | Cross sections, scattering amplitude |
| **499** | Born Approximation | Weak potential scattering, Fourier transform |
| **500** | Partial Wave Analysis | Phase shifts, S-matrix |
| **501** | Low-Energy Scattering | Scattering length, effective range |
| **502** | Resonances | Breit-Wigner formula, width and lifetime |
| **503** | Optical Theorem | Unitarity, total cross section |
| **504** | Semester 1A Capstone | Comprehensive review |

---

## Key Concepts

### The Scattering Problem

A beam of particles incident on a target scatters into various angles. We seek:
- **Differential cross section** $d\sigma/d\Omega$: probability per solid angle
- **Total cross section** $\sigma_{tot}$: total scattering probability
- **Scattering amplitude** $f(\theta, \phi)$: quantum-mechanical observable

### Asymptotic Wave Function

Far from the scatterer, the wave function has the form:

$$\psi(\mathbf{r}) \xrightarrow{r \to \infty} e^{ikz} + f(\theta) \frac{e^{ikr}}{r}$$

- Incident plane wave + outgoing spherical wave
- $f(\theta)$ encodes all scattering information

### Central Relation

$$\boxed{\frac{d\sigma}{d\Omega} = |f(\theta)|^2}$$

---

## Key Formulas

### Born Approximation
$$f^{(1)}(\theta) = -\frac{m}{2\pi\hbar^2}\int V(\mathbf{r})e^{i\mathbf{q}\cdot\mathbf{r}}d^3r$$

where $\mathbf{q} = \mathbf{k} - \mathbf{k}'$ is the momentum transfer.

### Partial Wave Expansion
$$f(\theta) = \sum_{\ell=0}^{\infty}(2\ell + 1)f_\ell P_\ell(\cos\theta)$$
$$f_\ell = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k} = \frac{1}{k\cot\delta_\ell - ik}$$

### Optical Theorem
$$\sigma_{tot} = \frac{4\pi}{k}\text{Im}[f(0)]$$

### Breit-Wigner Resonance
$$f_\ell \approx \frac{\Gamma/2}{E_R - E - i\Gamma/2}$$

---

## Physical Applications

### Nuclear Physics
- Neutron-proton scattering
- Nuclear resonances
- Compound nucleus formation

### Atomic Physics
- Electron-atom collisions
- Ramsauer-Townsend effect
- Photoionization

### Particle Physics
- Cross section measurements at colliders
- Resonance searches (Higgs, Z, W)
- Partial wave analysis of decays

### Condensed Matter
- Impurity scattering in metals
- Ultracold atomic collisions
- Feshbach resonances

---

## Quantum Computing Connections

### Quantum Simulation
- Scattering on quantum computers
- Phase estimation for scattering matrices
- Real-time dynamics of collisions

### Cross Section Calculations
- Quantum algorithms for nuclear physics
- Lattice QCD scattering amplitudes

---

## References

### Primary
- **Griffiths** Ch. 11: Scattering
- **Sakurai** Ch. 7: Scattering Theory
- **Shankar** Ch. 19: Scattering Theory

### Supplementary
- **Taylor** "Scattering Theory"
- **Newton** "Scattering Theory of Waves and Particles"
- Cambridge [Scattering Notes](https://www.damtp.cam.ac.uk/user/tong/aqm/aqmten.pdf)

---

## Week 72 Checklist

- [ ] Day 498: Understand scattering formalism and cross sections
- [ ] Day 499: Master Born approximation calculations
- [ ] Day 500: Compute phase shifts and partial wave expansions
- [ ] Day 501: Apply low-energy scattering concepts
- [ ] Day 502: Analyze resonances using Breit-Wigner
- [ ] Day 503: Derive and apply optical theorem
- [ ] Day 504: Complete Semester 1A comprehensive review

---

**Upon completing Week 72, you will have finished Semester 1A: Foundations of Quantum Mechanics!**

Congratulations on this major milestone in your quantum mechanics journey!
