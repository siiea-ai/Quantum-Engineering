# Week 49: Hilbert Space Formalism

## Overview

**Days:** 337-343 (7 days)
**Position:** Year 1, Month 13, Week 1
**Theme:** The Mathematical Arena of Quantum Mechanics

This week establishes Hilbert space as the mathematical setting for quantum mechanics. We develop Dirac notation, the language that unifies wave mechanics and matrix mechanics into a powerful abstract formalism.

---

## Learning Objectives

By the end of Week 49, you will be able to:

1. Define and work with abstract complex vector spaces
2. Use Dirac bra-ket notation fluently
3. Represent states as vectors and expand in orthonormal bases
4. Apply operators to quantum states
5. Distinguish Hermitian, unitary, and projection operators
6. Solve eigenvalue problems for quantum observables
7. Handle continuous spectra with generalized eigenvectors

---

## Daily Schedule

| Day | Date | Topic | Shankar | Sakurai |
|-----|------|-------|---------|---------|
| **337** | Mon | Complex Vector Spaces | Ch. 1.1-1.3 | Ch. 1.1 |
| **338** | Tue | Dirac Notation | Ch. 1.4-1.5 | Ch. 1.2 |
| **339** | Wed | Operators in Hilbert Space | Ch. 1.6-1.7 | Ch. 1.3 |
| **340** | Thu | Hermitian and Unitary Operators | Ch. 1.8 | Ch. 1.4 |
| **341** | Fri | Eigenvalue Problems | Ch. 1.9 | Ch. 1.4 |
| **342** | Sat | Continuous Spectra | Ch. 1.10-1.11 | Ch. 1.6 |
| **343** | Sun | Week Review & Lab | — | — |

---

## Key Concepts

### 1. Hilbert Space

A complete inner product space over ℂ:
- **Vector space:** Linear combinations of states
- **Inner product:** ⟨φ|ψ⟩ with properties (conjugate symmetry, linearity, positivity)
- **Completeness:** Cauchy sequences converge
- **Separable:** Countable orthonormal basis exists

### 2. Dirac Notation

| Symbol | Name | Meaning |
|--------|------|---------|
| |ψ⟩ | Ket | State vector (column vector) |
| ⟨φ| | Bra | Dual vector (row vector) |
| ⟨φ\|ψ⟩ | Bracket | Inner product (complex number) |
| |ψ⟩⟨φ| | Outer product | Operator (matrix) |
| ⟨φ\|Â\|ψ⟩ | Matrix element | ⟨φ|·(Â|ψ⟩) |

### 3. Basis Expansion

For orthonormal basis {|n⟩}:

$$|ψ⟩ = \sum_n c_n |n⟩ \quad \text{where} \quad c_n = ⟨n|ψ⟩$$

**Completeness relation:**
$$\hat{I} = \sum_n |n⟩⟨n|$$

### 4. Operators

**Linear:** Â(α|ψ⟩ + β|φ⟩) = αÂ|ψ⟩ + βÂ|φ⟩

**Hermitian (self-adjoint):** Â† = Â
- Eigenvalues are real
- Eigenvectors for different eigenvalues are orthogonal
- Represents observables

**Unitary:** Û†Û = ÛÛ† = Î
- Preserves inner products
- Eigenvalues have |λ| = 1
- Represents symmetry transformations

### 5. Eigenvalue Equation

$$Â|a⟩ = a|a⟩$$

- |a⟩ is an eigenstate (eigenvector)
- a is the eigenvalue
- For Hermitian Â: eigenvalues real, eigenstates form complete basis

---

## Essential Formulas

### Inner Product Properties
$$⟨φ|ψ⟩ = ⟨ψ|φ⟩^*$$
$$⟨φ|αψ_1 + βψ_2⟩ = α⟨φ|ψ_1⟩ + β⟨φ|ψ_2⟩$$
$$⟨ψ|ψ⟩ ≥ 0, \quad = 0 \text{ iff } |ψ⟩ = 0$$

### Normalization
$$⟨ψ|ψ⟩ = 1$$
$$\sum_n |c_n|^2 = 1$$

### Adjoint Properties
$$(Â†)† = Â$$
$$(ÂB̂)† = B̂†Â†$$
$$(α Â)† = α^* Â†$$
$$⟨φ|Â†|ψ⟩ = ⟨ψ|Â|φ⟩^*$$

### Position-Momentum Eigenstates
$$⟨x|x'⟩ = δ(x - x')$$
$$⟨p|p'⟩ = δ(p - p')$$
$$⟨x|p⟩ = \frac{1}{\sqrt{2πℏ}} e^{ipx/ℏ}$$

---

## Connections to Year 0

| Year 0 Topic | Week 49 Application |
|--------------|---------------------|
| Linear Algebra (Month 4) | Abstract vector spaces, linear operators |
| Hermitian/Unitary (Month 5) | Physical significance in QM |
| Functional Analysis (Month 9) | Hilbert spaces, unbounded operators |
| Complex Analysis (Month 7) | Complex amplitudes, inner products |

---

## Quantum Computing Connection

Hilbert space formalism directly underlies quantum computing:

| QM Concept | Quantum Computing |
|------------|-------------------|
| 2D Hilbert space | Single qubit |
| Basis states |0⟩, |1⟩ | Computational basis |
| Superposition | α|0⟩ + β|1⟩ |
| Unitary operators | Quantum gates |
| Tensor product | Multi-qubit systems |

---

## Problem Set Topics

1. Prove properties of inner products
2. Expand states in given bases
3. Compute matrix elements ⟨φ|Â|ψ⟩
4. Verify operator identities
5. Solve eigenvalue problems
6. Prove completeness relations
7. Transform between bases

---

## Computational Lab (Day 343)

**Topics:**
- Implement Hilbert space operations in NumPy
- Matrix representations of operators
- Eigenvalue decomposition
- Visualization of quantum states

```python
import numpy as np

# Pauli matrices (2D Hilbert space)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(sigma_z)
```

---

## Self-Assessment Checklist

After Week 49, you should be able to:

- [ ] Define a Hilbert space and its properties
- [ ] Convert between ket notation and wave function
- [ ] Expand arbitrary states in orthonormal bases
- [ ] Compute inner products and matrix elements
- [ ] Identify Hermitian, unitary, and projection operators
- [ ] Solve eigenvalue equations
- [ ] Apply completeness relations
- [ ] Work with continuous bases (position, momentum)

---

## Preview: Week 50

With Hilbert space established, Week 50 develops the measurement postulate: how observables are represented, how measurement outcomes are predicted, and what happens to quantum states upon measurement.

---

**Next:** [Day_337_Monday.md](Day_337_Monday.md) — Complex Vector Spaces
