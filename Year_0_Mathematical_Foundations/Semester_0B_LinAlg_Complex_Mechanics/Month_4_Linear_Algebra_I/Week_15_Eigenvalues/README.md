# Week 15: Eigenvalues and Eigenvectors (Days 99-105)

## Overview

This week introduces the eigenvalue problem—one of the most important concepts in linear algebra and the mathematical backbone of quantum mechanics. Eigenvalues and eigenvectors reveal the intrinsic structure of linear operators, showing which directions are preserved under transformation. In quantum mechanics, eigenvalues are the only possible measurement outcomes, and eigenvectors are the corresponding observable states.

## Learning Objectives

By the end of this week, you will be able to:

1. Define eigenvalues and eigenvectors and verify them for a given operator
2. Compute the characteristic polynomial and find eigenvalues
3. Determine eigenspaces and their dimensions (geometric multiplicity)
4. Diagonalize matrices when possible and identify obstructions
5. Apply the spectral decomposition for diagonalizable operators
6. Connect eigenvalue theory to quantum measurement and observables

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 99 (Mon) | The Eigenvalue Problem | Definition, eigenvalue equation Av = λv, examples |
| 100 (Tue) | Characteristic Polynomial | det(A - λI) = 0, finding eigenvalues, algebraic multiplicity |
| 101 (Wed) | Eigenspaces | Null space of (A - λI), geometric multiplicity, basis for eigenspaces |
| 102 (Thu) | Diagonalization | When matrices are diagonalizable, P⁻¹AP = D, similarity |
| 103 (Fri) | Spectral Decomposition | A = Σλᵢ|vᵢ⟩⟨vᵢ|, projection operators, spectral theorem preview |
| 104 (Sat) | Computational Lab | Python: numpy.linalg.eig(), visualization, applications |
| 105 (Sun) | Review | Comprehensive assessment, problem-solving mastery |

## Key Quantum Connections

- **Eigenvalues λ**: The only possible measurement outcomes for observable Ô
- **Eigenvectors |λ⟩**: States with definite measurement values
- **Eigenvalue equation Ô|λ⟩ = λ|λ⟩**: Fundamental equation of quantum measurement
- **Spectral decomposition Ô = Σλᵢ|λᵢ⟩⟨λᵢ|**: Operator expressed via its measurement basis
- **Degeneracy**: Multiple independent states with same eigenvalue
- **Complete set of eigenvectors**: Orthonormal measurement basis

### Quantum Examples
1. **Energy eigenstates**: Ĥ|E⟩ = E|E⟩ (stationary states)
2. **Spin measurements**: σᵤ|↑⟩ = +|↑⟩, σᵤ|↓⟩ = -|↓⟩
3. **Position/momentum**: Continuous eigenvalue spectrum
4. **Pauli matrices**: Eigenvalues ±1 for all σₓ, σᵧ, σᵤ

## Required Reading

### Primary Texts
- Axler, "Linear Algebra Done Right" (4th ed): Sections 5.A, 5.B, 5.C
- Shankar, "Principles of Quantum Mechanics": Chapter 1.4 (Eigenvalue Problem)

### Supplementary
- Strang, "Introduction to Linear Algebra": Chapter 6
- MIT OCW 18.06: Lectures 21-22 (Eigenvalues and Diagonalization)

## Key Theorems

1. **Eigenvalue Existence**: Every operator on a complex vector space has at least one eigenvalue
2. **Linear Independence**: Eigenvectors corresponding to distinct eigenvalues are linearly independent
3. **Characteristic Polynomial**: Eigenvalues are roots of det(A - λI) = 0
4. **Diagonalizability Criterion**: A is diagonalizable iff it has n linearly independent eigenvectors
5. **Spectral Theorem (Preview)**: Normal operators have orthonormal eigenbases

## Key Formulas

| Concept | Formula |
|---------|---------|
| Eigenvalue equation | Av = λv (v ≠ 0) |
| Characteristic polynomial | p(λ) = det(A - λI) |
| Eigenspace | E_λ = ker(A - λI) |
| Diagonalization | A = PDP⁻¹ where D = diag(λ₁,...,λₙ) |
| Spectral decomposition | A = Σᵢ λᵢPᵢ where Pᵢ = |vᵢ⟩⟨vᵢ| |
| Trace | tr(A) = Σλᵢ |
| Determinant | det(A) = Πλᵢ |

## Computational Tools

- NumPy: `np.linalg.eig()` for eigenvalues and eigenvectors
- NumPy: `np.linalg.eigh()` for Hermitian matrices (more stable)
- SciPy: `scipy.linalg.eig()` for generalized eigenvalue problems
- SymPy: `Matrix.eigenvals()`, `Matrix.eigenvects()` for symbolic computation

## Assessment

- Daily problem sets from Axler Chapter 5
- Computational lab exercises (Day 104)
- Week 15 comprehensive review (Day 105)
- Self-assessment checklist

## Prerequisites

- Week 13: Vector Spaces (subspaces, basis, dimension)
- Week 14: Linear Transformations (matrices, kernel, range)

## Connection to Future Topics

Week 15 prepares you for:
- **Week 16**: Inner product spaces (orthonormality of eigenvectors)
- **Month 5**: Hermitian operators and the full spectral theorem
- **Quantum Mechanics**: Measurement postulates, observable operators
- **Quantum Computing**: Diagonalization of quantum gates

## Week 15 Checklist

- [ ] Day 99: Eigenvalue definition and simple examples
- [ ] Day 100: Characteristic polynomial and finding eigenvalues
- [ ] Day 101: Eigenspaces and multiplicities
- [ ] Day 102: Diagonalization conditions and procedure
- [ ] Day 103: Spectral decomposition
- [ ] Day 104: Computational lab complete
- [ ] Day 105: Review and self-assessment
- [ ] All Axler Chapter 5 exercises attempted
- [ ] Can diagonalize 3×3 matrices by hand
- [ ] Understand quantum measurement connection

---

*"The eigenvalue problem is the heart of quantum mechanics—eigenvalues tell us what we can measure, and eigenvectors tell us what states give definite results."*
