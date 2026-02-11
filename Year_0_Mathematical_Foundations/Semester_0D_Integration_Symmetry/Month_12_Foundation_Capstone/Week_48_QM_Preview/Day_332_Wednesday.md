# Day 332: The Postulates of Quantum Mechanics

## Overview

**Month 12, Week 48, Day 3 — Wednesday**

Today we state the postulates of quantum mechanics — the axioms from which all of quantum theory follows.

## Learning Objectives

1. State all postulates precisely
2. Understand their mathematical content
3. See how they explain quantum phenomena
4. Prepare for rigorous Year 1 treatment

---

## The Postulates

### Postulate 1: States

**Every physical system is associated with a Hilbert space $\mathcal{H}$. The state of the system is completely described by a normalized vector $|\psi\rangle \in \mathcal{H}$.**

$$\langle\psi|\psi\rangle = 1$$

### Postulate 2: Observables

**Every physical observable corresponds to a Hermitian (self-adjoint) operator $\hat{A}$ acting on $\mathcal{H}$.**

$$\hat{A} = \hat{A}^\dagger$$

### Postulate 3: Measurement

**When measuring observable $\hat{A}$ on state $|\psi\rangle$:**
- The only possible outcomes are eigenvalues $a$ of $\hat{A}$
- Probability of outcome $a$: $P(a) = |\langle a|\psi\rangle|^2$
- After measurement yielding $a$: state collapses to $|a\rangle$

### Postulate 4: Time Evolution

**The time evolution of a closed system is governed by the Schrödinger equation:**

$$i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

Equivalently: $|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle$

### Postulate 5: Composite Systems

**The Hilbert space of a composite system is the tensor product of component spaces:**

$$\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B$$

---

## Consequences

### Superposition

From Postulate 1: any linear combination of states is a valid state.

### Uncertainty

From Postulate 3: non-commuting observables cannot have simultaneous definite values.

### Entanglement

From Postulate 5: composite systems can be in states that cannot be written as products.

### Wave Function Collapse

From Postulate 3: measurement fundamentally changes the state.

---

## Summary

### The Quantum Framework

$$\boxed{\text{States} + \text{Operators} + \text{Measurement} + \text{Evolution} = \text{Quantum Mechanics}}$$

---

## Preview: Day 333

Tomorrow: **Simple Systems Preview** — particle in a box, harmonic oscillator, hydrogen atom.
