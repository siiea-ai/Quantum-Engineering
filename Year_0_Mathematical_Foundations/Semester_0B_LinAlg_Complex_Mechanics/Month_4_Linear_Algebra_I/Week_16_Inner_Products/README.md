# Week 16: Inner Product Spaces (Days 106-112)

## Overview

This week completes Month 4 by introducing inner product spaces—the mathematical foundation for quantum mechanics. Inner products provide the geometric concepts of length, angle, and orthogonality that are essential for understanding quantum states and measurements.

## Learning Objectives

By the end of this week, you will be able to:

1. Define and work with inner products on real and complex vector spaces
2. Apply the Cauchy-Schwarz and triangle inequalities
3. Understand orthogonality and orthogonal complements
4. Execute the Gram-Schmidt orthogonalization process
5. Compute orthogonal projections and solve least squares problems
6. Connect all concepts to quantum mechanical probability and measurement

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 106 (Mon) | Inner Products | Definition, axioms, Dirac notation, ⟨u\|v⟩ |
| 107 (Tue) | Norms & Inequalities | Cauchy-Schwarz, triangle inequality, normalization |
| 108 (Wed) | Orthogonality | Orthogonal vectors, complements, direct sum |
| 109 (Thu) | Gram-Schmidt | Orthogonalization algorithm, QR decomposition |
| 110 (Fri) | Orthonormal Bases | Parseval, Bessel, best approximation, least squares |
| 111 (Sat) | Computational Lab | Python implementations, quantum simulations |
| 112 (Sun) | Review | Comprehensive assessment, Month 4 wrap-up |

## Key Quantum Connections

- **Inner product ⟨φ|ψ⟩**: Probability amplitude
- **|⟨φ|ψ⟩|²**: Probability of measuring |φ⟩ given state |ψ⟩
- **Normalization ⟨ψ|ψ⟩ = 1**: Physical states have total probability 1
- **Orthogonality ⟨φ|ψ⟩ = 0**: Perfectly distinguishable states
- **Orthonormal basis**: Valid measurement basis
- **Completeness Σᵢ|i⟩⟨i| = I**: Probability conservation

## Required Reading

### Primary Texts
- Axler, "Linear Algebra Done Right" (4th ed): Sections 6.A, 6.B, 6.C
- Shankar, "Principles of Quantum Mechanics": Chapter 1.3-1.4

### Supplementary
- Strang, "Introduction to Linear Algebra": Chapter 4
- MIT OCW 18.06: Lectures 15, 17

## Key Theorems

1. **Cauchy-Schwarz Inequality**: |⟨u|v⟩| ≤ ‖u‖·‖v‖
2. **Triangle Inequality**: ‖u+v‖ ≤ ‖u‖ + ‖v‖
3. **Parseval's Identity**: ‖v‖² = Σᵢ|⟨eᵢ|v⟩|²
4. **Best Approximation**: Projection minimizes distance
5. **Completeness**: Σᵢ|eᵢ⟩⟨eᵢ| = I

## Computational Tools

- NumPy: `np.vdot()` for complex inner products
- NumPy: `np.linalg.qr()` for QR decomposition
- NumPy: `np.linalg.lstsq()` for least squares
- SciPy: `scipy.integrate.quad()` for function space inner products

## Assessment

- Daily problem sets
- Computational lab exercises
- Week 16 comprehensive review (Day 112)
- Month 4 mastery assessment

## Prerequisites

- Week 13: Vector Spaces
- Week 14: Linear Transformations
- Week 15: Eigenvalues and Eigenvectors

## Connection to Future Topics

Week 16 prepares you for:
- **Week 17**: Hermitian and unitary operators (spectral theorem)
- **Week 18**: Tensor products and density matrices
- **Quantum Information**: Fidelity, distinguishability, tomography

---

*"The inner product is the heart of quantum mechanics."*
