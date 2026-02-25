# Week 18: Advanced Linear Algebra — SVD, Tensor Products, Density Matrices (Days 120-126)

## Overview
This week covers advanced linear algebra topics essential for quantum information theory: Singular Value Decomposition, tensor products for composite systems, and the density matrix formalism for mixed quantum states.

## Topics Covered

### Day 120 (Monday): Singular Value Decomposition
- SVD theorem: A = UΣV*
- Singular values and vectors
- Geometric interpretation
- Connection to eigendecomposition
- Low-rank approximation

### Day 121 (Tuesday): SVD Applications
- Moore-Penrose pseudoinverse
- Least squares via SVD
- Principal Component Analysis (PCA)
- Polar decomposition
- Schmidt decomposition for quantum states

### Day 122 (Wednesday): Tensor Products
- Definition of tensor product spaces
- Kronecker product
- Computational basis for multi-qubit systems
- Product states vs entangled states
- Multi-qubit quantum gates

### Day 123 (Thursday): Composite Quantum Systems
- Partial trace operation
- Reduced density matrices
- Entanglement detection via reduced states
- Entanglement entropy
- Schmidt decomposition revisited

### Day 124 (Friday): Density Matrices
- Pure vs mixed states
- Density matrix properties
- Bloch sphere for mixed states
- Expectation values and measurements
- Quantum channels (Kraus operators)

### Day 125 (Saturday): Computational Lab
- SVD-based quantum state analysis
- Tensor product simulations
- Density matrix operations
- Lindblad master equation
- Quantum error correction basics

### Day 126 (Sunday): Week Review
- Comprehensive concept review
- Integration problems
- Self-assessment
- Week 19 preparation

## Key Theorems and Results

1. **SVD Theorem:** Every m×n matrix A = UΣV* with U, V unitary, Σ diagonal
2. **Eckart-Young:** Best rank-k approximation is truncated SVD
3. **Schmidt Decomposition:** |ψ⟩ = Σλᵢ|aᵢ⟩|bᵢ⟩ (SVD of coefficient matrix)
4. **Entanglement Criterion:** Pure |ψ⟩_AB entangled ⟺ ρ_A mixed

## Quantum Mechanics Connections

| Linear Algebra | Quantum Application |
|----------------|---------------------|
| SVD | Schmidt decomposition, entanglement |
| Tensor product | Composite systems, multi-qubit states |
| Partial trace | Subsystem description |
| Density matrix | Mixed states, decoherence |
| Quantum channel | Noise, error, measurement |

## Prerequisites
- Week 17: Hermitian and Unitary Operators
- Week 16: Inner Product Spaces

## Textbook References
- Strang, "Introduction to Linear Algebra," Chapter 7 (SVD)
- Nielsen & Chuang, Sections 2.1.7, 2.4 (Tensor products, Density matrices)
- Preskill Lecture Notes, Chapters 2-3

## Study Time
- Days 120-124: 7 hours/day (35 hours)
- Day 125: 7.5 hours (Saturday lab)
- Day 126: 4 hours (Sunday review)
- **Total: 46.5 hours**

## Key Equations

**SVD:** A = UΣV*

**Tensor Product:** (A⊗B)(C⊗D) = (AC)⊗(BD)

**Partial Trace:** ρ_A = tr_B(ρ_AB)

**Entanglement Entropy:** E = -Σλᵢ²log₂(λᵢ²)

**Quantum Channel:** ℰ(ρ) = Σₖ Kₖ ρ Kₖ†
