# Week 55: Quantum Harmonic Oscillator

## Overview

**Days:** 379-385 (7 days)
**Position:** Year 1, Month 14, Week 3
**Theme:** The Most Important Exactly Solvable Problem in Quantum Mechanics

The quantum harmonic oscillator (QHO) is ubiquitous in physics. It describes vibrations of molecules, electromagnetic field modes, phonons in solids, and serves as the foundation for quantum field theory. This week, we master both the algebraic (ladder operator) and analytic (Hermite polynomial) approaches.

---

## Learning Objectives

By the end of Week 55, you will be able to:

1. **Set up** the QHO Hamiltonian and introduce dimensionless variables
2. **Define** ladder operators and prove their fundamental commutation relations
3. **Construct** the complete set of number states |n⟩ algebraically
4. **Derive** wave functions using Hermite polynomials
5. **Build** coherent states as eigenstates of the annihilation operator
6. **Visualize** quantum states in phase space via the Wigner function
7. **Connect** the QHO to quantum optics and bosonic quantum computing

---

## Daily Schedule

| Day | Date | Topic | Key Content | Shankar | Sakurai |
|-----|------|-------|-------------|---------|---------|
| **379** | Mon | QHO Setup & Motivation | V = ½mω²x², dimensionless ξ | §7.1-7.2 | §2.3 |
| **380** | Tue | Ladder Operators | â, â†, [â, â†] = 1 | §7.3 | §2.3 |
| **381** | Wed | Number States \|n⟩ | E_n = ℏω(n+½), Fock space | §7.3 | §2.3 |
| **382** | Thu | QHO Wave Functions | Hermite polynomials, ψ_n(x) | §7.4 | §2.3 |
| **383** | Fri | Coherent States | \|α⟩, displacement operator | §7.5 | — |
| **384** | Sat | QHO in Phase Space | Wigner function, classical limit | §7.5 | — |
| **385** | Sun | Week Review & Lab | Comprehensive lab with QuTiP | — | — |

---

## Key Concepts

### 1. Harmonic Oscillator Hamiltonian

$$\boxed{\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2}$$

Dimensionless position: $\xi = \sqrt{\frac{m\omega}{\hbar}}x$

### 2. Ladder Operators

**Annihilation operator:**
$$\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} + \frac{i\hat{p}}{m\omega}\right)$$

**Creation operator:**
$$\hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} - \frac{i\hat{p}}{m\omega}\right)$$

**Fundamental commutator:**
$$\boxed{[\hat{a}, \hat{a}^\dagger] = 1}$$

### 3. Hamiltonian in Terms of Ladder Operators

$$\boxed{\hat{H} = \hbar\omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right) = \hbar\omega\left(\hat{N} + \frac{1}{2}\right)}$$

where $\hat{N} = \hat{a}^\dagger\hat{a}$ is the **number operator**.

### 4. Number States (Fock States)

**Energy eigenvalues:**
$$\boxed{E_n = \hbar\omega\left(n + \frac{1}{2}\right)}, \quad n = 0, 1, 2, \ldots$$

**Ladder actions:**
$$\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$$
$$\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$$

**State construction:**
$$|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$$

### 5. Ground State Wave Function

$$\boxed{\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} e^{-m\omega x^2/2\hbar}}$$

**Zero-point energy:** $E_0 = \frac{\hbar\omega}{2}$ (vacuum fluctuations)

### 6. Excited State Wave Functions

$$\psi_n(x) = \frac{1}{\sqrt{2^n n!}} H_n(\xi) \psi_0(x)$$

where $H_n(\xi)$ are Hermite polynomials: $H_0 = 1$, $H_1 = 2\xi$, $H_2 = 4\xi^2 - 2$, ...

### 7. Coherent States

**Definition:** Eigenstates of $\hat{a}$:
$$\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$$

**Expansion:**
$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^{\infty} \frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

**Minimum uncertainty:** $\Delta x \cdot \Delta p = \frac{\hbar}{2}$

**Time evolution:**
$$|\alpha(t)\rangle = |e^{-i\omega t}\alpha\rangle$$

### 8. Wigner Function

$$W(x, p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty} \psi^*\left(x+y\right)\psi\left(x-y\right)e^{2ipy/\hbar} dy$$

Phase space quasi-probability distribution connecting quantum and classical mechanics.

---

## Essential Formulas Summary

| Quantity | Formula |
|----------|---------|
| Energy levels | $E_n = \hbar\omega(n + \frac{1}{2})$ |
| Level spacing | $\Delta E = \hbar\omega$ (constant) |
| Ground state energy | $E_0 = \frac{\hbar\omega}{2}$ |
| Position from ladder | $\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$ |
| Momentum from ladder | $\hat{p} = i\sqrt{\frac{m\omega\hbar}{2}}(\hat{a}^\dagger - \hat{a})$ |
| Number operator | $\hat{N} = \hat{a}^\dagger\hat{a}$ |
| Commutator | $[\hat{a}, \hat{a}^\dagger] = 1$ |
| Coherent state mean | $\langle\alpha|\hat{N}|\alpha\rangle = |\alpha|^2$ |

---

## Physical Applications

### Molecular Vibrations
- Diatomic molecules near equilibrium: V(r) ≈ ½k(r - r₀)²
- Infrared spectroscopy probes vibrational transitions

### Quantum Electrodynamics
- Each mode of EM field is a quantum harmonic oscillator
- Photon number states are Fock states |n⟩
- Laser light is described by coherent states |α⟩

### Solid State Physics
- Phonons (lattice vibrations) are QHO excitations
- Specific heat of solids (Einstein model)

### Quantum Computing
- **Bosonic qubits:** Information encoded in oscillator states
- **Cavity QED:** Superconducting circuits with resonators
- **Cat states:** Superpositions of coherent states

---

## Connections to Year 0 Mathematics

| Year 0 Topic | Week 55 Application |
|--------------|---------------------|
| Hermite polynomials (Month 3) | Wave function solutions |
| Gaussian integrals (Month 1) | Normalization, expectation values |
| Commutator algebra (Month 4) | [â, â†] = 1 derivation |
| Fourier analysis (Month 3) | Momentum space wave functions |
| Phase space (Month 6) | Wigner function, classical limit |

---

## Quantum Computing Deep Dive

### Bosonic Quantum Computing

The harmonic oscillator provides an alternative to qubit-based quantum computing:

| Qubit Encoding | QHO Realization |
|---------------|-----------------|
| Computational basis | Fock states \|n⟩ |
| Cat qubit | \|α⟩ + \|-α⟩ |
| GKP qubit | Grid states in phase space |
| Binomial codes | Superpositions of Fock states |

### Key Advantages
- Infinite-dimensional Hilbert space
- Hardware-efficient error correction
- Natural interface with photonic systems

---

## Computational Lab Overview

### Tools
- **NumPy/SciPy:** Matrix representations, eigensolvers
- **QuTiP:** Quantum Toolbox in Python for oscillator states
- **Matplotlib:** Wave function and Wigner function visualization

### Lab Projects (Day 385)
1. Build ladder operator matrices
2. Construct Fock states and verify orthonormality
3. Animate coherent state time evolution
4. Plot Wigner functions for various states

---

## Self-Assessment Checklist

After Week 55, you should be able to:

- [ ] Derive the QHO energy spectrum using ladder operators
- [ ] Prove [â, â†] = 1 from canonical commutation relation
- [ ] Construct |n⟩ states from |0⟩ using creation operator
- [ ] Calculate ⟨x⟩, ⟨p⟩, ⟨x²⟩, ⟨p²⟩ for any Fock state
- [ ] Write the first few Hermite polynomials
- [ ] Verify that coherent states are minimum uncertainty states
- [ ] Explain why coherent states are "most classical"
- [ ] Sketch Wigner functions for Fock and coherent states

---

## Preview: Week 56

Next week covers **Tunneling & Barriers**, where we:
- Analyze scattering from step and rectangular potentials
- Calculate transmission and reflection coefficients
- Explore quantum tunneling and its applications
- Study the WKB approximation

---

## References

### Primary Texts
- Shankar, "Principles of Quantum Mechanics," Chapter 7
- Sakurai, "Modern Quantum Mechanics," Section 2.3

### Supplementary
- Griffiths, "Introduction to Quantum Mechanics," Section 2.3
- Cohen-Tannoudji, "Quantum Mechanics," Chapter V
- Gerry & Knight, "Introductory Quantum Optics," Chapters 2-3

### Online Resources
- MIT OCW 8.04 Lecture 9: Harmonic Oscillator
- Physics LibreTexts: Quantum Harmonic Oscillator
- QuTiP Documentation: Quantum States

---

*"The harmonic oscillator... has an importance that can hardly be overestimated. Any smooth potential can be approximated near a minimum by a harmonic potential."*
— R. Shankar, Principles of Quantum Mechanics

---

**Next:** [Day_379_Monday.md](Day_379_Monday.md) — QHO Setup & Motivation
