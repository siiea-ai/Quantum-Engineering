# Week 17: Hermitian and Unitary Operators (Days 113-119)

## Overview
This week covers the essential operator classes for quantum mechanics: adjoint, Hermitian (self-adjoint), unitary, and normal operators, culminating in the powerful Spectral Theorem.

## Topics Covered

### Day 113 (Monday): Adjoint Operators
- Definition via inner product: ⟨Av, w⟩ = ⟨v, A*w⟩
- Matrix representation: A* = Ā^T
- Properties: (AB)* = B*A*, (A*)* = A
- Connection to transpose and complex conjugate

### Day 114 (Tuesday): Hermitian (Self-Adjoint) Operators
- Definition: A = A*
- Real eigenvalues theorem
- Orthogonality of eigenvectors
- Physical observables in QM
- Examples: Pauli matrices, Hamiltonians

### Day 115 (Wednesday): Unitary Operators
- Definition: U*U = UU* = I
- Preservation of inner products and norms
- Eigenvalues on unit circle
- QM: Quantum gates, time evolution operators
- Examples: Hadamard, rotation gates

### Day 116 (Thursday): The Spectral Theorem
- Statement for Hermitian operators
- Proof outline
- Spectral decomposition: A = ΣλᵢPᵢ
- Functions of operators: f(A) = Σf(λᵢ)Pᵢ
- QM: Measurement postulate connection

### Day 117 (Friday): Normal Operators and Applications
- Definition: AA* = A*A
- Spectral theorem for normal operators
- Simultaneous diagonalization: [A,B] = 0
- Positive operators
- Polar decomposition
- Compatible observables, CSCO

### Day 118 (Saturday): Computational Lab
- Quantum operator library implementation
- Bloch sphere visualization
- Quantum circuit simulation
- Decoherence channels
- State tomography basics

### Day 119 (Sunday): Week Review
- Comprehensive concept review
- Full problem set
- Self-assessment
- QM connection synthesis

## Key Theorems
1. **Spectral Theorem (Hermitian):** A = A* implies orthonormal eigenbasis with real eigenvalues
2. **Spectral Theorem (Normal):** Normal ⟺ unitarily diagonalizable
3. **Simultaneous Diagonalization:** [A,B] = 0 ⟺ common eigenbasis exists

## Quantum Mechanics Connections
- **Observables:** Hermitian operators (real measurement outcomes)
- **Time Evolution:** U(t) = e^(-iHt/ℏ) (unitary)
- **Measurement:** Spectral decomposition → probabilities
- **Compatibility:** [A,B] = 0 → simultaneous measurement possible

## Prerequisites
- Week 16: Inner Product Spaces
- Week 15: Eigenvalues and Eigenvectors

## Textbook References
- Axler, "Linear Algebra Done Right," Chapters 7.A-7.D
- Shankar, "Principles of Quantum Mechanics," Chapter 1
- Sakurai, "Modern Quantum Mechanics," Chapter 1

## Study Time
- Days 113-117: 7 hours/day (35 hours)
- Day 118: 7.5 hours (Saturday lab)
- Day 119: 4 hours (Sunday review)
- **Total: 46.5 hours**
