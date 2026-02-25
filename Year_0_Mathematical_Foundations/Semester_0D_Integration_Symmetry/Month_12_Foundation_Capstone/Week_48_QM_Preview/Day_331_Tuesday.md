# Day 331: Mathematical Structure of Quantum Mechanics

## Overview

**Month 12, Week 48, Day 2 — Tuesday**

Today we preview the mathematical framework of quantum mechanics: Hilbert spaces as state spaces, operators as observables, and eigenvalue problems as measurement theory.

## Learning Objectives

1. Understand states as Hilbert space vectors
2. Connect observables to operators
3. See how Year 0 math maps to QM
4. Prepare for rigorous treatment in Year 1

---

## 1. The State Space

### States as Vectors

Quantum states are vectors $|\psi\rangle$ in a Hilbert space $\mathcal{H}$.

**Properties:**
- Normalized: $\langle\psi|\psi\rangle = 1$
- Superposition: $|\psi\rangle = \sum_n c_n |n\rangle$
- Complex amplitudes: $c_n \in \mathbb{C}$

### Year 0 Connection

| Year 0 | Quantum Mechanics |
|--------|-------------------|
| $\mathbb{C}^n$ | Finite-dimensional systems |
| $L^2(\mathbb{R})$ | Position wavefunctions |
| Inner product | Probability amplitudes |

---

## 2. Observables as Operators

### The Correspondence

$$\boxed{\text{Observable} \leftrightarrow \text{Hermitian Operator}}$$

**Properties of Hermitian operators:**
- Real eigenvalues (measurement outcomes)
- Orthogonal eigenvectors (definite-value states)
- Spectral decomposition

### Key Operators

| Observable | Operator | Eigenvalues |
|------------|----------|-------------|
| Position | $\hat{x}$ | $x \in \mathbb{R}$ |
| Momentum | $\hat{p} = -i\hbar\nabla$ | $p \in \mathbb{R}$ |
| Energy | $\hat{H}$ | $E_n$ |
| Angular momentum | $\hat{L}_z$ | $m\hbar$ |

---

## 3. The Eigenvalue Problem

### Physical Interpretation

$$\hat{A}|\psi\rangle = a|\psi\rangle$$

- $|\psi\rangle$: State with definite value $a$ for observable $A$
- $a$: Possible measurement outcome

### Year 0 Connection

This is exactly the eigenvalue problem from linear algebra!

---

## 4. Commutators and Uncertainty

### The Commutator

$$[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$$

### Uncertainty Relation

$$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$$

### The Canonical Commutator

$$[\hat{x}, \hat{p}] = i\hbar$$

This is the quantum version of $\{x, p\} = 1$ from Hamiltonian mechanics!

---

## 5. Preview: Key QM Equations

### Schrödinger Equation

$$i\hbar\frac{\partial}{\partial t}|\psi\rangle = \hat{H}|\psi\rangle$$

### Time-Independent

$$\hat{H}|\psi_n\rangle = E_n|\psi_n\rangle$$

This is a Sturm-Liouville eigenvalue problem!

---

## Summary

### The Mathematical Framework

$$\boxed{\text{States} \to \text{Hilbert Space}}$$
$$\boxed{\text{Observables} \to \text{Hermitian Operators}}$$
$$\boxed{\text{Measurements} \to \text{Eigenvalues}}$$

---

## Preview: Day 332

Tomorrow: **The Postulates of Quantum Mechanics** — the axioms of the theory.
