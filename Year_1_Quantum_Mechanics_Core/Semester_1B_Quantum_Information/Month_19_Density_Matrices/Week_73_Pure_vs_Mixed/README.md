# Week 73: Pure vs Mixed States

## Month 19: Density Matrices | Semester 1B: Quantum Information

---

## Week Overview

This week introduces one of the most fundamental distinctions in quantum mechanics: the difference between **pure states** and **mixed states**. While pure states represent complete quantum information described by state vectors, mixed states describe systems with classical uncertainty—arising from incomplete knowledge, thermal ensembles, or entanglement with external systems. The density matrix formalism provides a unified framework for handling both cases, making it essential for quantum information science.

### Why This Matters for Quantum Computing

In real quantum computers, qubits are never perfectly isolated. Decoherence transforms pure quantum states into mixed states, fundamentally limiting quantum computation. Understanding the density matrix formalism is crucial for:
- Characterizing noise in quantum systems
- Designing error correction protocols
- Analyzing quantum channels
- Understanding entanglement and quantum correlations

---

## Learning Objectives for the Week

By the end of Week 73, you will be able to:

1. **Define and construct** density operators for both pure and mixed quantum states
2. **Verify** the three essential properties: Hermiticity, positivity, and unit trace
3. **Calculate** expectation values and measurement probabilities using the trace formula
4. **Quantify** state purity using the purity measure and distinguish pure from mixed states
5. **Visualize** single-qubit mixed states on the Bloch ball and relate purity to radius
6. **Compute** trace distance and fidelity to quantify distinguishability between quantum states
7. **Apply** density matrix techniques to realistic quantum computing scenarios

---

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **505 (Mon)** | Density Operator Definition | Pure states as projectors, ensemble interpretation, ρ = Σᵢ pᵢ\|ψᵢ⟩⟨ψᵢ\| |
| **506 (Tue)** | Properties and Trace | Hermiticity, positivity, normalization, trace operations |
| **507 (Wed)** | Expectation Values | ⟨A⟩ = Tr(ρA), measurement statistics, POVM connection |
| **508 (Thu)** | Purity and Mixedness | Purity γ = Tr(ρ²), von Neumann entropy, maximally mixed states |
| **509 (Fri)** | Bloch Sphere Mixed States | Bloch ball representation, ρ = (I + r⃗·σ⃗)/2, purity as radius |
| **510 (Sat)** | Distinguishing States | Trace distance D(ρ,σ), fidelity F(ρ,σ), operational meaning |
| **511 (Sun)** | Week Review | Integration, comprehensive problems, preparation for dynamics |

---

## Key Formulas

### Density Operator Construction

$$\rho_{\text{pure}} = |\psi\rangle\langle\psi|$$

$$\rho_{\text{mixed}} = \sum_i p_i |\psi_i\rangle\langle\psi_i|, \quad \sum_i p_i = 1, \quad p_i \geq 0$$

### Essential Properties

$$\rho^\dagger = \rho \quad \text{(Hermiticity)}$$

$$\langle\phi|\rho|\phi\rangle \geq 0 \, \forall |\phi\rangle \quad \text{(Positivity)}$$

$$\text{Tr}(\rho) = 1 \quad \text{(Normalization)}$$

### Expectation Values

$$\langle A \rangle = \text{Tr}(\rho A)$$

$$P(\text{outcome } m) = \text{Tr}(\rho \Pi_m)$$

### Purity Measure

$$\gamma = \text{Tr}(\rho^2), \quad \frac{1}{d} \leq \gamma \leq 1$$

- $\gamma = 1$: Pure state
- $\gamma = 1/d$: Maximally mixed state (dimension $d$)

### Bloch Ball Representation (Single Qubit)

$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma}) = \frac{1}{2}\begin{pmatrix} 1 + r_z & r_x - ir_y \\ r_x + ir_y & 1 - r_z \end{pmatrix}$$

$$|\vec{r}| \leq 1, \quad \gamma = \frac{1 + |\vec{r}|^2}{2}$$

### Distance Measures

$$D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma| = \frac{1}{2}\sum_i |\lambda_i|$$

$$F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

---

## Prerequisites

Before starting this week, ensure familiarity with:
- Hilbert space structure and Dirac notation (Year 0)
- Linear operators, eigenvalues, and spectral decomposition
- Outer products and projectors
- Basic probability theory
- Pauli matrices and single-qubit states (Week 72)

---

## Primary References

1. **Nielsen & Chuang**, *Quantum Computation and Quantum Information*, Chapter 2
2. **Preskill**, *Ph219 Lecture Notes*, Chapters 2-3
3. **Wilde**, *Quantum Information Theory*, Chapter 4
4. **Bengtsson & Życzkowski**, *Geometry of Quantum States*

---

## Computational Tools

This week's labs use:
- **NumPy**: Matrix operations and eigenvalue computations
- **Qiskit**: Quantum circuit simulation and state tomography
- **Matplotlib**: Bloch sphere and purity visualizations
- **SciPy**: Matrix functions (square roots, logarithms)

---

## Assessment Checkpoints

### Conceptual Understanding
- [ ] Can explain why mixed states arise in quantum mechanics
- [ ] Understands difference between classical and quantum uncertainty
- [ ] Can interpret density matrix elements physically

### Mathematical Proficiency
- [ ] Can construct density matrices from state vectors or ensembles
- [ ] Can verify all three defining properties
- [ ] Can compute expectation values using the trace formula
- [ ] Can calculate purity and interpret the result

### Computational Skills
- [ ] Can implement density matrix operations in Python
- [ ] Can visualize states on the Bloch ball
- [ ] Can compute distance measures numerically

---

## Connection to Future Topics

This week's material directly prepares you for:
- **Week 74**: Partial Trace and Reduced Density Matrices
- **Week 75**: Quantum Channels and Completely Positive Maps
- **Week 76**: Quantum Entanglement Measures
- **Month 20**: Open Quantum Systems and Decoherence

---

## Historical Note

The density matrix was introduced independently by John von Neumann (1927) and Lev Landau (1927) to handle statistical mixtures in quantum mechanics. Von Neumann's work established the mathematical foundations, while Landau applied it to quantum statistical mechanics. Today, the density operator formalism is indispensable for quantum information science, where the distinction between pure and mixed states has profound operational consequences for quantum computing and communication.

---

*"The density matrix is the quantum analog of a probability distribution, but with the crucial addition of quantum coherences that make interference possible."* — John Preskill
