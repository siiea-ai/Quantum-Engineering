# Week 157: Density Matrices

## Overview

**Days:** 1093-1099
**Theme:** Pure and Mixed States, Bloch Representation, Trace Operations

This week provides a comprehensive review of density matrix formalism, the fundamental mathematical framework for describing quantum states in the most general sense. Understanding density matrices is essential for quantum information theory, open quantum systems, and qualifying examination success.

## Daily Breakdown

### Day 1093 (Monday): Pure States and Density Operators
- State vectors and outer products
- Density matrix definition: $$\rho = |\psi\rangle\langle\psi|$$
- Properties: Hermiticity, unit trace, positivity
- Purity: $$\text{Tr}(\rho^2) = 1$$ for pure states

### Day 1094 (Tuesday): Mixed States and Statistical Ensembles
- Statistical mixtures vs. quantum superposition
- Ensemble interpretation: $$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$
- Convexity of the set of density matrices
- Extremal points as pure states

### Day 1095 (Wednesday): Bloch Sphere for Single Qubits
- Bloch representation: $$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$$
- Bloch vector constraints: $$|\vec{r}| \leq 1$$
- Pure states on surface, mixed states inside
- Geometric interpretation of quantum operations

### Day 1096 (Thursday): Bloch Vector Calculations
- Computing Bloch vectors from density matrices
- Expectation values and Bloch components
- Effect of unitary operations on Bloch vector
- Measurement probabilities from Bloch representation

### Day 1097 (Friday): Trace Operations and Properties
- Trace as inner product: $$\text{Tr}(A^\dagger B)$$
- Cyclic property and invariance under basis change
- Trace distance: $$D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma|$$
- Fidelity: $$F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

### Day 1098 (Saturday): Partial Trace Introduction
- Motivation: Describing subsystems of composite systems
- Partial trace definition and computation
- Examples: Two-qubit systems
- Connection to reduced density matrices

### Day 1099 (Sunday): Review and Problem Session
- Comprehensive problem solving
- Oral exam practice
- Self-assessment and gap identification

## Key Formulas

### Density Matrix Properties

$$\boxed{\rho = \rho^\dagger, \quad \text{Tr}(\rho) = 1, \quad \rho \geq 0}$$

### Purity

$$\boxed{\gamma = \text{Tr}(\rho^2), \quad \frac{1}{d} \leq \gamma \leq 1}$$

### Bloch Representation (Qubit)

$$\boxed{\rho = \frac{1}{2}(I + r_x \sigma_x + r_y \sigma_y + r_z \sigma_z)}$$

### Bloch Vector from Density Matrix

$$\boxed{r_i = \text{Tr}(\rho \sigma_i), \quad i \in \{x, y, z\}}$$

### Von Neumann Entropy

$$\boxed{S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i}$$

### Trace Distance

$$\boxed{D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma| = \frac{1}{2}\sum_i |\lambda_i|}$$

### Fidelity

$$\boxed{F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2}$$

For pure state $$|\psi\rangle$$: $$F(\rho, |\psi\rangle\langle\psi|) = \langle\psi|\rho|\psi\rangle$$

### Partial Trace

$$\boxed{\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B) \rho_{AB} (I_A \otimes |j\rangle_B)}$$

## Learning Objectives

By the end of this week, you should be able to:

1. Construct density matrices for pure and mixed states from physical descriptions
2. Verify all three defining properties (Hermiticity, trace, positivity)
3. Convert between density matrix and Bloch vector representations
4. Calculate purity and von Neumann entropy
5. Compute trace distance and fidelity between states
6. Perform partial trace calculations for two-qubit systems
7. Explain the physical meaning of density matrices in oral exam format

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

Before starting this week, ensure familiarity with:
- Linear algebra: Hermitian matrices, eigenvalues, trace
- Dirac notation: bras, kets, inner products
- Single qubit: $$|0\rangle$$, $$|1\rangle$$, Pauli matrices
- Probability theory: expectation values, statistical mixtures

## References

1. Nielsen & Chuang, Section 2.4: The density operator
2. Preskill Notes, Chapter 2: Foundations I
3. Wilde, Chapter 3: Quantum states and channels
4. [CMU Quantum Channels Notes](https://quantum.phys.cmu.edu/QCQI/qitd412.pdf)
