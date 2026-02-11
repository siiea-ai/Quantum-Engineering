# Week 35: Operators on Hilbert Spaces

## Overview

Week 35 marks a crucial transition from studying the **spaces** of quantum mechanics to studying the **operators** that act on them. If Hilbert spaces are the stage where quantum mechanics unfolds, then operators are the actors performing the physics. This week, we develop the theory of bounded linear operators, laying the groundwork for understanding observables, symmetries, and dynamics in quantum systems.

The progression this week builds systematically from fundamental definitions to powerful structural results:
- **Days 239-240**: Establish bounded operators, their norm, and the Banach algebra B(H)
- **Days 241-242**: Introduce adjoint operators and special classes (self-adjoint, unitary, normal)
- **Days 243-244**: Study projection operators and compact operators
- **Day 245**: Consolidate understanding through comprehensive review

## Why Operators Matter for Quantum Mechanics

In the Dirac-von Neumann formulation of quantum mechanics, every physical quantity corresponds to an operator:

| Operator Concept | Quantum Mechanics Interpretation |
|-----------------|----------------------------------|
| Bounded operator $A$ | Transformation of states |
| Operator norm $\|A\|$ | Maximum amplification factor |
| Adjoint $A^\dagger$ | Hermitian conjugate |
| Self-adjoint $A = A^\dagger$ | Observable (Hermitian operator) |
| Unitary $U^\dagger U = I$ | Symmetry transformation, time evolution |
| Projection $P^2 = P = P^\dagger$ | Measurement, state collapse |
| Compact operator | Finite-rank approximation, trace class |
| Spectrum $\sigma(A)$ | Possible measurement outcomes |

## Weekly Learning Objectives

By the end of Week 35, you will be able to:

1. **Define and analyze** bounded linear operators and the operator norm
2. **Work with** the Banach algebra $\mathcal{B}(\mathcal{H})$ of bounded operators
3. **Compute** adjoint operators and verify their properties
4. **Classify** operators as self-adjoint, unitary, normal, or none
5. **Apply** projection operators to decompose Hilbert spaces
6. **Characterize** compact operators and their approximation properties
7. **Connect** all concepts to quantum mechanical observables and dynamics

## Daily Topics

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 239 | Bounded Linear Operators | Definition, examples, boundedness ↔ continuity |
| 240 | Operator Norm and B(H) | $\|A\| = \sup \|Ax\|/\|x\|$, Banach algebra structure |
| 241 | Adjoint Operators | $\langle Ax, y\rangle = \langle x, A^\dagger y\rangle$, properties |
| 242 | Self-Adjoint and Unitary Operators | Observables, symmetries, normal operators |
| 243 | Projections and Orthogonal Decomposition | $P^2 = P = P^\dagger$, direct sum decomposition |
| 244 | Compact Operators | Finite-rank limits, Hilbert-Schmidt operators |
| 245 | Week Review | Synthesis, problem solving, quantum applications |

## Key Theorems This Week

1. **BLT Theorem**: Bounded ⟺ Continuous for linear operators
2. **B(H) is Banach**: Space of bounded operators is complete in operator norm
3. **Adjoint Properties**: $(AB)^\dagger = B^\dagger A^\dagger$, $\|A^\dagger A\| = \|A\|^2$
4. **Spectral Properties**: Self-adjoint operators have real spectrum
5. **Projection Theorem**: Every closed subspace has unique orthogonal projection
6. **Approximation Theorem**: Compact operators are limits of finite-rank operators

## Key Formulas

### Bounded Operators
$$\|A\| = \sup_{\|x\|=1} \|Ax\| = \sup_{x \neq 0} \frac{\|Ax\|}{\|x\|}$$

### Adjoint
$$\langle Ax, y \rangle = \langle x, A^\dagger y \rangle \quad \forall x, y \in \mathcal{H}$$

### Special Operators
$$\begin{aligned}
\text{Self-adjoint:} & \quad A = A^\dagger \\
\text{Unitary:} & \quad U^\dagger U = UU^\dagger = I \\
\text{Normal:} & \quad AA^\dagger = A^\dagger A \\
\text{Projection:} & \quad P^2 = P = P^\dagger
\end{aligned}$$

### Operator Inequalities
$$\|AB\| \leq \|A\| \|B\|, \quad \|A^\dagger\| = \|A\|, \quad \|A^\dagger A\| = \|A\|^2$$

## Prerequisites

- Hilbert spaces and inner products (Week 34)
- Orthonormal bases and completeness
- Linear maps between vector spaces
- Complex conjugation and sesquilinearity

## Resources

### Primary Texts
- Kreyszig, *Introductory Functional Analysis with Applications*, Chapters 3-4
- Reed & Simon, *Methods of Mathematical Physics I*, Chapters 2, 6
- Conway, *A Course in Functional Analysis*, Chapters 2-3

### Supplementary Materials
- MIT OCW 18.102 (Operators on Hilbert Spaces)
- Teschl, *Mathematical Methods in Quantum Mechanics*, Chapter 2
- Hall, *Quantum Theory for Mathematicians*, Chapters 5-7

### Quantum Mechanics Connections
- Sakurai, *Modern Quantum Mechanics*, Chapter 1
- Cohen-Tannoudji, *Quantum Mechanics*, Chapters II-III
- Nielsen & Chuang, *Quantum Computation*, Chapter 2 (operator formalism)

## Study Tips

1. **Think in matrices first**: Every concept for infinite dimensions has a finite-dimensional matrix analogue
2. **Use $\ell^2$ as your sandbox**: Operators on $\ell^2$ are infinite matrices—concrete and computable
3. **The adjoint is the key**: Most important properties flow from the adjoint operation
4. **Physical intuition helps**: Self-adjoint = observable, unitary = symmetry, projection = measurement

## Weekly Schedule

- **Morning Sessions** (3 hours): Theory, definitions, proofs
- **Afternoon Sessions** (3 hours): Problem solving, examples
- **Evening Sessions** (2 hours): Computational labs and quantum connections

Total: **56 hours** of focused study

---

## The Big Picture

This week answers a fundamental question: **What mathematical objects represent physical quantities in quantum mechanics?**

The answer: **Operators on Hilbert space.**

More precisely:
- Observables (position, momentum, energy, spin) are **self-adjoint operators**
- Symmetry transformations are **unitary operators**
- Measurement outcomes correspond to **eigenvalues**
- State collapse involves **projection operators**
- Time evolution is generated by **the Hamiltonian operator**

Understanding operators is understanding how quantum systems evolve, how we measure them, and what symmetries they possess.

---

*"The Hilbert space formalism represents physical quantities as operators. This abstract approach, far from being merely mathematical convenience, captures the essential non-commutativity of quantum mechanical observables."* — John von Neumann
