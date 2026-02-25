# Week 58: Spin Angular Momentum

## Overview

**Days:** 400-406 (7 days)
**Theme:** Intrinsic Angular Momentum and the Birth of the Qubit
**Position:** Year 1, Month 15, Week 2

This week introduces spin—the purely quantum mechanical angular momentum that has no classical analog. From the historical Stern-Gerlach experiment through the complete mathematical formalism of spin-1/2, we build the foundation for understanding qubits in quantum computing.

---

## STATUS: 🟢 IN PROGRESS

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 400 | Monday | Stern-Gerlach Experiment | ⬜ NOT STARTED |
| 401 | Tuesday | Spin-1/2 Formalism | ⬜ NOT STARTED |
| 402 | Wednesday | Pauli Matrices | ⬜ NOT STARTED |
| 403 | Thursday | Bloch Sphere | ⬜ NOT STARTED |
| 404 | Friday | Spin Dynamics | ⬜ NOT STARTED |
| 405 | Saturday | Higher Spin | ⬜ NOT STARTED |
| 406 | Sunday | Week Review & Qiskit Lab | ⬜ NOT STARTED |

---

## Learning Objectives

By the end of Week 58, you will be able to:

1. **Explain** the Stern-Gerlach experiment and why it demonstrates spatial quantization
2. **Work** fluently with spin-1/2 states in the two-dimensional Hilbert space
3. **Apply** Pauli matrix algebra to calculate spin observables
4. **Visualize** arbitrary spin states on the Bloch sphere
5. **Derive** spin precession in magnetic fields (Larmor precession)
6. **Construct** spin matrices for arbitrary spin s
7. **Connect** spin-1/2 formalism directly to qubit operations in quantum computing

---

## Daily Schedule

### Day 400 (Monday): Stern-Gerlach Experiment
- Historical context: 1922 experiment by Stern and Gerlach
- Experimental setup: Inhomogeneous magnetic field
- Key observation: Silver atoms split into exactly TWO spots
- Spatial quantization → intrinsic angular momentum
- Sequential Stern-Gerlach: Quantum measurement foundations
- **Lab:** Simulate beam splitting

### Day 401 (Tuesday): Spin-1/2 Formalism
- Two-dimensional Hilbert space for spin
- Basis states: |↑⟩ = |+⟩ = |0⟩ and |↓⟩ = |-⟩ = |1⟩
- General spinor: |χ⟩ = α|↑⟩ + β|↓⟩ with |α|² + |β|² = 1
- Eigenvalues: S² → ℏ²s(s+1) = (3/4)ℏ², Sᵤ → ±ℏ/2
- Spin operators in matrix form
- **Lab:** Spinor calculations

### Day 402 (Wednesday): Pauli Matrices
- Explicit forms of σₓ, σᵧ, σᵤ
- Key properties: σᵢ² = I, Tr(σᵢ) = 0, det(σᵢ) = -1
- Fundamental algebra: σᵢσⱼ = δᵢⱼI + iεᵢⱼₖσₖ
- Completeness: Any 2×2 Hermitian matrix as I, σₓ, σᵧ, σᵤ combination
- Spin operators: Ŝ = (ℏ/2)σ⃗
- **Lab:** Pauli matrix algebra

### Day 403 (Thursday): Bloch Sphere
- Parameterization: |n̂⟩ = cos(θ/2)|↑⟩ + e^{iφ}sin(θ/2)|↓⟩
- Bloch vector: ⟨σ⃗⟩ = (sin θ cos φ, sin θ sin φ, cos θ)
- Eigenstates of n̂·σ for arbitrary direction
- Density matrix: ρ = (I + r⃗·σ⃗)/2
- Pure vs mixed states: |r⃗| = 1 vs |r⃗| < 1
- **Lab:** Interactive Bloch sphere visualization

### Day 404 (Friday): Spin Dynamics
- Magnetic moment: μ⃗ = -gₛ(e/2m)Ŝ = γŜ
- Hamiltonian: Ĥ = -μ⃗·B⃗ = (gₛeℏ/4m)σ⃗·B⃗
- Larmor precession: ωL = gₛeB/2m
- Time evolution on Bloch sphere
- Rabi oscillations in oscillating field
- **Lab:** Precession animation

### Day 405 (Saturday): Higher Spin
- Spin-1: Three states |1,+1⟩, |1,0⟩, |1,-1⟩
- General spin-s: (2s+1)-dimensional representation
- Explicit spin matrices for s = 1, 3/2
- Construction via ladder operators
- SU(2) representation theory
- **Lab:** Construct spin-s matrices algorithmically

### Day 406 (Sunday): Week Review & Qiskit Lab
- Comprehensive review of spin formalism
- Qubit = spin-1/2 isomorphism
- Full Qiskit lab: Single-qubit gates as spin rotations
- X, Y, Z gates as σₓ, σᵧ, σᵤ
- Hadamard, S, T gates on Bloch sphere
- Practice problems and assessment
- Preview: Week 59 Addition of Angular Momenta

---

## Key Formulas

### Spin-1/2 Fundamentals

| Quantity | Formula |
|----------|---------|
| Eigenvalues of S² | $$\hbar^2 s(s+1) = \frac{3\hbar^2}{4}$$ |
| Eigenvalues of Sᵤ | $$m_s\hbar = \pm\frac{\hbar}{2}$$ |
| General spinor | $$\|\chi\rangle = \alpha\|↑\rangle + \beta\|↓\rangle$$ |
| Normalization | $$\|\alpha\|^2 + \|\beta\|^2 = 1$$ |

### Pauli Matrices

$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

### Pauli Algebra

| Property | Formula |
|----------|---------|
| Square | $$\sigma_i^2 = I$$ |
| Trace | $$\text{Tr}(\sigma_i) = 0$$ |
| Determinant | $$\det(\sigma_i) = -1$$ |
| Product | $$\sigma_i\sigma_j = \delta_{ij}I + i\varepsilon_{ijk}\sigma_k$$ |
| Anticommutator | $$\{\sigma_i, \sigma_j\} = 2\delta_{ij}I$$ |
| Commutator | $$[\sigma_i, \sigma_j] = 2i\varepsilon_{ijk}\sigma_k$$ |

### Bloch Sphere

| Quantity | Formula |
|----------|---------|
| State parameterization | $$\|n̂\rangle = \cos\frac{\theta}{2}\|↑\rangle + e^{i\phi}\sin\frac{\theta}{2}\|↓\rangle$$ |
| Bloch vector | $$\vec{r} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$$ |
| Density matrix | $$\rho = \frac{1}{2}(I + \vec{r}\cdot\vec{\sigma})$$ |
| Eigenstate of n̂·σ | $$(\hat{n}\cdot\vec{\sigma})\|n̂\rangle = \|n̂\rangle$$ |

### Spin Dynamics

| Quantity | Formula |
|----------|---------|
| Magnetic moment | $$\vec{\mu} = -g_s\frac{e}{2m}\vec{S} = \gamma\vec{S}$$ |
| Spin Hamiltonian | $$\hat{H} = -\vec{\mu}\cdot\vec{B} = \frac{g_s e\hbar}{4m}\vec{\sigma}\cdot\vec{B}$$ |
| Larmor frequency | $$\omega_L = \frac{g_s eB}{2m} = \gamma B$$ |
| Precession period | $$T = \frac{2\pi}{\omega_L}$$ |

---

## Quantum Computing Connections

This week establishes the direct connection between spin-1/2 physics and quantum computing:

| Spin Physics | Quantum Computing |
|--------------|-------------------|
| Spin-1/2 state \|χ⟩ | Qubit state \|ψ⟩ |
| \|↑⟩, \|↓⟩ basis | \|0⟩, \|1⟩ computational basis |
| Pauli matrices σₓ, σᵧ, σᵤ | X, Y, Z gates |
| Bloch sphere | Qubit state visualization |
| Spin rotation e^{-iθn̂·σ/2} | Single-qubit gate U(θ,φ,λ) |
| Larmor precession | Z-rotation (phase gate) |
| Rabi oscillation | X, Y rotations |

---

## Primary References

### Textbooks

- **Shankar** "Principles of Quantum Mechanics" Ch. 14
- **Sakurai** "Modern Quantum Mechanics" Ch. 3.1-3.2
- **Griffiths** "Introduction to QM" Ch. 4.4
- **Nielsen & Chuang** "Quantum Computation" Ch. 1.2

### Historical Papers

- Gerlach, W. & Stern, O. (1922) "Der experimentelle Nachweis der Richtungsquantelung im Magnetfeld"
- Pauli, W. (1927) "Zur Quantenmechanik des magnetischen Elektrons"

### Online Resources

- MIT OCW 8.05 Lectures 17-19 (Zwiebach)
- Feynman Lectures Vol. III, Ch. 6, 10
- IBM Qiskit Textbook: Single Qubit Gates

---

## Computational Labs

| Day | Lab Focus | Tools |
|-----|-----------|-------|
| 400 | Stern-Gerlach simulation | NumPy, Matplotlib |
| 401 | Spinor calculations | NumPy |
| 402 | Pauli algebra verification | NumPy, SymPy |
| 403 | Bloch sphere visualization | Matplotlib 3D, QuTiP |
| 404 | Precession animation | Matplotlib animation |
| 405 | Spin-s matrix construction | NumPy, SymPy |
| 406 | Qiskit single-qubit gates | Qiskit |

---

## Prerequisites

From Week 57 (Orbital Angular Momentum):
- Angular momentum commutation relations
- Ladder operators L̂±
- Eigenvalue problem J² and Jᵤ
- Quantum numbers j and m

From Months 13-14:
- Hilbert space structure
- Matrix mechanics
- Eigenvalue problems

---

## Assessment Checklist

### Conceptual Understanding
- [ ] Explain why Stern-Gerlach shows exactly 2 spots for silver atoms
- [ ] Distinguish spin from orbital angular momentum
- [ ] Describe what the Bloch sphere represents physically

### Mathematical Skills
- [ ] Verify Pauli matrix algebra: σᵢσⱼ = δᵢⱼI + iεᵢⱼₖσₖ
- [ ] Calculate expectation values ⟨σₓ⟩, ⟨σᵧ⟩, ⟨σᵤ⟩ for any spinor
- [ ] Find eigenspinors of σₓ, σᵧ and verify eigenvalues ±1

### Computational Proficiency
- [ ] Implement Bloch sphere visualization
- [ ] Simulate Larmor precession
- [ ] Execute single-qubit gates in Qiskit

---

## Preview: Week 59

Next week we tackle **Addition of Angular Momenta**:
- Combining two angular momenta: Ĵ = Ĵ₁ + Ĵ₂
- Coupled vs uncoupled bases
- Clebsch-Gordan coefficients
- Spin-orbit coupling
- Two spin-1/2 particles: Singlet and triplet states
- Foundation for multi-qubit entanglement

---

*"I could not find any difference between the particle with spin and a tiny magnet."*
— Wolfgang Pauli

---

**Created:** February 2, 2026
**Status:** In Progress
**Dependencies:** Week 57 (Orbital Angular Momentum)
