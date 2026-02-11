# Month 15: Angular Momentum & Spin

## Overview

**Duration:** 28 days (Days 393-420)
**Position:** Year 1, Semester 1A, Month 3
**Theme:** Rotational Symmetry and Intrinsic Angular Momentum

This month explores one of the most beautiful structures in quantum mechanics: the theory of angular momentum. We discover how rotational symmetry leads to quantization, introduce spin as a purely quantum property, and master the addition of angular momenta—essential for understanding atomic structure and quantum computing.

---

## STATUS: ✅ COMPLETE

| Week | Days | Topic | Status | Progress |
|------|------|-------|--------|----------|
| **Week 57** | 393-399 | Orbital Angular Momentum | ✅ COMPLETE | 7/7 |
| **Week 58** | 400-406 | Spin Angular Momentum | ✅ COMPLETE | 7/7 |
| **Week 59** | 407-413 | Addition of Angular Momenta | ✅ COMPLETE | 7/7 |
| **Week 60** | 414-420 | Rotations & Wigner D-Matrices | ✅ COMPLETE | 7/7 |
| **Total** | 393-420 | Angular Momentum & Spin | ✅ **COMPLETE** | **28/28** |

---

## Learning Objectives

By the end of Month 15, you will be able to:

1. **Derive** angular momentum commutation relations from rotational symmetry
2. **Solve** the angular momentum eigenvalue problem algebraically
3. **Construct** spherical harmonics Y_l^m(θ,φ)
4. **Master** spin-1/2 algebra and Pauli matrices
5. **Add** angular momenta using Clebsch-Gordan coefficients
6. **Apply** Wigner-Eckart theorem to matrix elements
7. **Connect** angular momentum to qubit operations

---

## Weekly Breakdown

### Week 57: Orbital Angular Momentum (Days 393-399)

| Day | Topic | Key Content |
|-----|-------|-------------|
| 393 | Classical to Quantum | L = r × p → L̂ = -iℏ(r × ∇) |
| 394 | Commutation Relations | [L̂ᵢ, L̂ⱼ] = iℏε_ijk L̂ₖ, [L̂², L̂ᵢ] = 0 |
| 395 | Ladder Operators | L̂± = L̂ₓ ± iL̂ᵧ, raising/lowering |
| 396 | Eigenvalue Spectrum | L² eigenvalues: ℏ²l(l+1), Lᵤ: ℏm |
| 397 | Spherical Harmonics I | Y_l^m(θ,φ), orthonormality |
| 398 | Spherical Harmonics II | Addition theorem, recursion |
| 399 | Week Review & Lab | Visualization and applications |

**Key Results:**
- Quantization: l = 0, 1, 2, ... and m = -l, ..., +l
- Spherical harmonics: $Y_l^m(\theta,\phi) = N_{lm} P_l^m(\cos\theta) e^{im\phi}$
- Ladder action: $\hat{L}_\pm|l,m\rangle = \hbar\sqrt{l(l+1)-m(m\pm 1)}|l,m\pm 1\rangle$

---

### Week 58: Spin Angular Momentum (Days 400-406)

| Day | Topic | Key Content |
|-----|-------|-------------|
| 400 | Stern-Gerlach Experiment | Discovery of spin |
| 401 | Spin-1/2 Formalism | |↑⟩, |↓⟩ basis, 2D Hilbert space |
| 402 | Pauli Matrices | σₓ, σᵧ, σᵤ properties and algebra |
| 403 | Spin States & Bloch Sphere | |n̂⟩ = cos(θ/2)|↑⟩ + e^{iφ}sin(θ/2)|↓⟩ |
| 404 | Spin Dynamics | Precession in magnetic field |
| 405 | Higher Spin | Spin-1, spin-3/2 representations |
| 406 | Week Review & Lab | Qiskit: Qubit as spin-1/2 |

**Key Results:**
- Pauli matrices: σₓ², σᵧ², σᵤ² = I, {σᵢ, σⱼ} = 2δᵢⱼI
- Spin-1/2: S = ℏ/2 σ, eigenvalues ±ℏ/2
- Bloch sphere: Complete representation of qubit states

---

### Week 59: Addition of Angular Momenta (Days 407-413)

| Day | Topic | Key Content |
|-----|-------|-------------|
| 407 | Two Angular Momenta | Ĵ = Ĵ₁ + Ĵ₂, tensor product space |
| 408 | Coupled vs Uncoupled | |j₁,m₁;j₂,m₂⟩ vs |j,m;j₁,j₂⟩ |
| 409 | Clebsch-Gordan Coefficients | ⟨j₁,m₁;j₂,m₂|j,m⟩, triangle rule |
| 410 | Spin-Orbit Coupling | j = l ± 1/2, atomic fine structure |
| 411 | Two Spin-1/2 Addition | Singlet |0,0⟩ and triplet |1,m⟩ |
| 412 | Wigner 3j Symbols | Symmetry properties |
| 413 | Week Review & Lab | CG coefficient calculations |

**Key Results:**
- Triangle rule: |j₁-j₂| ≤ j ≤ j₁+j₂
- Two spin-1/2: Singlet (antisymmetric) + Triplet (symmetric)
- Total magnetic quantum number: m = m₁ + m₂

---

### Week 60: Rotations & Wigner D-Matrices (Days 414-420)

| Day | Topic | Key Content |
|-----|-------|-------------|
| 414 | Rotation Operators | R̂(n̂,θ) = e^{-iθn̂·Ĵ/ℏ} |
| 415 | Euler Angles | R(α,β,γ) = Rᵤ(α)Rᵧ(β)Rᵤ(γ) |
| 416 | Wigner D-Matrices | D^j_{m'm}(α,β,γ) = ⟨j,m'|R̂|j,m⟩ |
| 417 | Wigner-Eckart Theorem | Reduced matrix elements |
| 418 | Selection Rules | ΔJ, Δm rules for transitions |
| 419 | Tensor Operators | Irreducible tensor operators |
| 420 | Month Review & Capstone | Integration and assessment |

**Key Results:**
- SU(2) double cover of SO(3): Spin-1/2 rotates by 4π
- D-matrices: $D^j_{m'm}(\alpha,\beta,\gamma) = e^{-im'\alpha} d^j_{m'm}(\beta) e^{-im\gamma}$
- Wigner-Eckart: Separates geometry from dynamics

---

## Primary References

### Textbooks

- **Shankar** "Principles of Quantum Mechanics" Ch. 12-15 (Primary)
- **Sakurai** "Modern Quantum Mechanics" Ch. 3 (Supplementary)
- **Griffiths** "Introduction to QM" Ch. 4 (Reference)

### Key Sections

| Topic | Shankar | Sakurai | Griffiths |
|-------|---------|---------|-----------|
| Orbital L | Ch. 12 | §3.5-3.6 | §4.3 |
| Spin | Ch. 14 | §3.1-3.2 | §4.4 |
| Addition | Ch. 15 | §3.7-3.8 | §4.4 |
| Rotations | Ch. 12-13 | §3.3-3.4 | — |

### Video Resources

- MIT OCW 8.05 Lectures 16-22 (Zwiebach)
- Feynman Lectures Vol. III, Ch. 18
- Leonard Susskind "Quantum Mechanics" (Stanford)

---

## Computational Labs

| Week | Lab Topic | Tools |
|------|-----------|-------|
| 57 | Spherical harmonics visualization | NumPy, Matplotlib |
| 58 | Bloch sphere animation | Qiskit, QuTiP |
| 59 | Clebsch-Gordan calculator | SymPy |
| 60 | Wigner D-matrix computation | SciPy |

---

## Quantum Computing Connections

| Angular Momentum Concept | Quantum Computing Application |
|-------------------------|------------------------------|
| Spin-1/2 | Single qubit |
| Pauli matrices | X, Y, Z gates |
| Bloch sphere | State visualization |
| SU(2) rotations | Single-qubit gates |
| Two-spin addition | Two-qubit entanglement |
| Singlet state | Bell state |0,0⟩ ↔ |Ψ⁻⟩ |

---

## Assessment Milestones

### Week 57 Checkpoint

- [ ] Derive [Lₓ, Lᵧ] = iℏLᵤ from [x̂, p̂ₓ] = iℏ
- [ ] Prove L² eigenvalues are ℏ²l(l+1)
- [ ] Plot spherical harmonics for l = 0,1,2

### Week 58 Checkpoint

- [ ] Verify Pauli matrix algebra: σᵢσⱼ = δᵢⱼI + iεᵢⱼₖσₖ
- [ ] Calculate spin expectation values on Bloch sphere
- [ ] Implement qubit rotations in Qiskit

### Week 59 Checkpoint

- [ ] Add two spin-1/2 to get singlet and triplet
- [ ] Calculate Clebsch-Gordan coefficients for j₁=1, j₂=1/2
- [ ] Apply to hydrogen fine structure

### Week 60 Checkpoint

- [ ] Compute D¹(α,β,γ) matrix elements
- [ ] Apply Wigner-Eckart to dipole transitions
- [ ] Derive selection rules ΔJ = 0, ±1

---

## Directory Structure

```
Month_15_Angular_Momentum/
├── README.md                              # This file
├── Week_57_Orbital_Angular_Momentum/
│   ├── README.md
│   └── Day_393-399_*.md                   # 7 day files
├── Week_58_Spin/
│   ├── README.md
│   └── Day_400-406_*.md                   # 7 day files
├── Week_59_Addition_Angular_Momentum/
│   ├── README.md
│   └── Day_407-413_*.md                   # 7 day files
└── Week_60_Rotations_Wigner/
    ├── README.md
    └── Day_414-420_*.md                   # 7 day files
```

---

## Prerequisites from Months 13-14

| Previous Topic | Month 15 Application |
|----------------|---------------------|
| Hilbert space | Angular momentum Hilbert space |
| Ladder operators | L±, J± operators |
| Eigenvalue problems | Angular momentum quantization |
| QHO algebra | Parallel structure for J algebra |
| Commutators | Angular momentum commutation |

---

## Preview: Month 16

Next month covers **Three-Dimensional Problems**, where we:
- Solve the radial Schrödinger equation
- Master the hydrogen atom completely
- Introduce fine and hyperfine structure
- Apply perturbation theory to real atoms

---

*"The spin of the electron is the most important discovery in physics of the 20th century."*
— Wolfgang Pauli

---

**Created:** February 2, 2026
**Status:** In Progress
