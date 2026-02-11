# Week 158: Composite Systems

## Overview

**Days:** 1100-1106
**Theme:** Tensor Products, Reduced Density Matrices, Schmidt Decomposition, Purification

This week focuses on the mathematical framework for describing composite quantum systems. Understanding tensor products, reduced density matrices, and Schmidt decomposition is essential for quantum information theory, entanglement analysis, and qualifying examination success.

## Daily Breakdown

### Day 1100 (Monday): Tensor Products of Hilbert Spaces
- Definition of tensor product space $$\mathcal{H}_A \otimes \mathcal{H}_B$$
- Computational basis: $$|ij\rangle = |i\rangle_A \otimes |j\rangle_B$$
- Dimension of composite space: $$d_A \cdot d_B$$
- Operators on composite systems

### Day 1101 (Tuesday): Tensor Product of Operators
- Kronecker product representation
- Local vs. global operators
- Identity: $$(A \otimes B)(C \otimes D) = AC \otimes BD$$
- Partial transpose operation

### Day 1102 (Wednesday): Reduced Density Matrices
- Partial trace definition and computation
- Physical interpretation: local observations
- Product states vs. entangled states
- Examples with Bell states

### Day 1103 (Thursday): Schmidt Decomposition Theorem
- Statement and proof of Schmidt decomposition
- Schmidt coefficients and Schmidt rank
- Uniqueness (up to phase)
- Connection to singular value decomposition

### Day 1104 (Friday): Applications of Schmidt Decomposition
- Entanglement detection via Schmidt rank
- Computing reduced density matrices
- Schmidt number as entanglement witness
- Bipartite pure state classification

### Day 1105 (Saturday): Purification
- Every mixed state has a purification
- Construction of purifications
- Non-uniqueness: unitary freedom on ancilla
- Applications in quantum information

### Day 1106 (Sunday): Review and Problem Session
- Comprehensive problem solving
- Oral exam practice
- Self-assessment and gap identification

## Key Formulas

### Tensor Product Properties

$$\boxed{|i\rangle_A \otimes |j\rangle_B \equiv |ij\rangle \equiv |i,j\rangle}$$

$$\boxed{(A \otimes B)(|a\rangle \otimes |b\rangle) = (A|a\rangle) \otimes (B|b\rangle)}$$

$$\boxed{(A \otimes B)^\dagger = A^\dagger \otimes B^\dagger}$$

### Kronecker Product (Matrix Form)

For $$A = (a_{ij})$$ of size $$m \times n$$ and $$B$$ of size $$p \times q$$:

$$\boxed{A \otimes B = \begin{pmatrix} a_{11}B & a_{12}B & \cdots \\ a_{21}B & a_{22}B & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}}$$

### Partial Trace

$$\boxed{\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B)\rho_{AB}(I_A \otimes |j\rangle_B)}$$

For operators: $$\boxed{\text{Tr}_B(A \otimes B) = A \cdot \text{Tr}(B)}$$

### Schmidt Decomposition

For any bipartite pure state $$|\psi\rangle_{AB}$$ with $$d_A \leq d_B$$:

$$\boxed{|\psi\rangle_{AB} = \sum_{i=1}^{r} \lambda_i |a_i\rangle_A |b_i\rangle_B}$$

where:
- $$\lambda_i > 0$$ are Schmidt coefficients with $$\sum_i \lambda_i^2 = 1$$
- $$\{|a_i\rangle\}$$ and $$\{|b_i\rangle\}$$ are orthonormal
- $$r \leq \min(d_A, d_B)$$ is the Schmidt rank

### Reduced States from Schmidt Decomposition

$$\boxed{\rho_A = \sum_i \lambda_i^2 |a_i\rangle\langle a_i|, \quad \rho_B = \sum_i \lambda_i^2 |b_i\rangle\langle b_i|}$$

### Purification

For $$\rho_A = \sum_i p_i |i\rangle\langle i|$$:

$$\boxed{|\Psi\rangle_{AB} = \sum_i \sqrt{p_i}|i\rangle_A|i\rangle_B}$$

is a purification satisfying $$\rho_A = \text{Tr}_B(|\Psi\rangle\langle\Psi|)$$.

## Learning Objectives

By the end of this week, you should be able to:

1. Construct tensor product states and operators for composite systems
2. Compute partial traces for arbitrary bipartite density matrices
3. Apply the Schmidt decomposition to analyze bipartite pure states
4. Determine Schmidt rank and coefficients from a given state
5. Construct purifications for mixed states
6. Explain the connection between Schmidt decomposition and SVD
7. Use these tools to characterize entanglement

## Files in This Week

| File | Description |
|------|-------------|
| `README.md` | This overview document |
| `Review_Guide.md` | Comprehensive theoretical review |
| `Problem_Set.md` | 30 qualifying exam-style problems |
| `Problem_Solutions.md` | Detailed worked solutions |
| `Oral_Practice.md` | Oral exam questions and answers |
| `Self_Assessment.md` | Diagnostic checklist and self-test |

## Prerequisites

Before starting this week, ensure mastery of:
- Density matrices (Week 157)
- Linear algebra: eigenvalues, SVD, orthonormal bases
- Dirac notation for multiple qubits

## References

1. Nielsen & Chuang, Section 2.4: The postulates of quantum mechanics
2. Preskill Notes, Chapter 2: Foundations I
3. [MIT Schmidt Decomposition Notes](https://www.rle.mit.edu/cua_pub/8.422/Reading%20Material/NOTES-schmidt-decomposition-and-epr-from-nielsen-and-chuang-p109.pdf)
4. [Caltech Ph219 Chapter 2](https://www.preskill.caltech.edu/ph219/chap2_13.pdf)
