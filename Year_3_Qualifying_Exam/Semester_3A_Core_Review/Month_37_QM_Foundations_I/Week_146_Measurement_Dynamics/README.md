# Week 146: Measurement and Dynamics

## Overview

Week 146 focuses on two pillars of quantum mechanics: measurement theory and quantum dynamics. Understanding how measurements collapse states and how states evolve in time is essential for every area of quantum physics. These topics are heavily tested on qualifying exams, both as standalone problems and in combination with other material.

**Days:** 1016-1022
**Total Study Time:** ~50 hours
**Focus:** Measurement postulate, state collapse, Schrödinger equation, unitary evolution, propagators

---

## Learning Objectives

By the end of this week, you will be able to:

1. State and apply all five postulates of quantum mechanics
2. Calculate probabilities and post-measurement states
3. Solve the time-independent Schrödinger equation
4. Compute time evolution using the propagator
5. Work in both Schrödinger and Heisenberg pictures
6. Apply Ehrenfest's theorem to relate quantum and classical motion
7. Connect symmetries to conservation laws

---

## Daily Schedule

| Day | Focus | Activities |
|-----|-------|------------|
| 1016 (Mon) | Measurement Postulate | Review Guide sections 1-2; Problems 1-6 |
| 1017 (Tue) | Expectation Values & Collapse | Review Guide sections 3-4; Problems 7-12 |
| 1018 (Wed) | Schrödinger Equation | Review Guide section 5; Problems 13-17 |
| 1019 (Thu) | Time Evolution & Propagator | Review Guide sections 6-7; Problems 18-22 |
| 1020 (Fri) | Pictures & Ehrenfest | Review Guide sections 8-9; Problems 23-26 |
| 1021 (Sat) | Timed Problem Practice | Complete remaining problems under exam conditions |
| 1022 (Sun) | Oral Practice & Assessment | Oral questions; Self-assessment |

---

## Core Concepts

### 1. Postulates of Quantum Mechanics
- State space postulate
- Observable postulate
- Measurement postulate
- Collapse postulate
- Evolution postulate

### 2. Measurement Theory
- Probabilities from Born rule
- Post-measurement states
- Repeated measurements
- Compatible vs. incompatible observables

### 3. Time Evolution
- Time-dependent Schrödinger equation
- Stationary states
- Time-evolution operator
- Propagator (Green's function)

### 4. Pictures of Quantum Mechanics
- Schrödinger picture
- Heisenberg picture
- Interaction picture

### 5. Conservation Laws
- Ehrenfest's theorem
- Symmetries and conserved quantities

---

## Key Equations

### Measurement
$$P(a_n) = |\langle a_n | \psi \rangle|^2$$

$$|\psi\rangle \xrightarrow{\text{measure } a_n} |a_n\rangle$$

### Time Evolution
$$i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

$$|\psi(t)\rangle = \hat{U}(t)|\psi(0)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle$$

### Propagator
$$K(x,t;x',0) = \langle x | e^{-i\hat{H}t/\hbar} | x' \rangle$$

$$\psi(x,t) = \int K(x,t;x',0)\psi(x',0) \, dx'$$

### Ehrenfest's Theorem
$$\frac{d}{dt}\langle \hat{A} \rangle = \frac{i}{\hbar}\langle [\hat{H}, \hat{A}] \rangle + \left\langle \frac{\partial \hat{A}}{\partial t} \right\rangle$$

---

## Files in This Week

| File | Description |
|------|-------------|
| [Review_Guide.md](Review_Guide.md) | Comprehensive topic review (~3000 words) |
| [Problem_Set.md](Problem_Set.md) | 26 qualifying exam problems |
| [Problem_Solutions.md](Problem_Solutions.md) | Detailed solutions |
| [Oral_Practice.md](Oral_Practice.md) | 15 oral exam questions with frameworks |
| [Self_Assessment.md](Self_Assessment.md) | Progress checklist and mastery criteria |

---

## Study Tips for This Week

1. **Master the postulates** — They're the axioms of QM
2. **Practice propagator calculations** — Common on exams
3. **Compare Schrödinger and Heisenberg** — Know when each is useful
4. **Connect to classical mechanics** — Ehrenfest shows the correspondence
5. **Work with specific systems** — Apply to oscillator, wells, etc.

---

## Common Exam Mistakes to Avoid

1. Forgetting normalization after collapse
2. Confusing probability with probability amplitude
3. Sign errors in time evolution ($$-i$$ vs. $$+i$$)
4. Incorrect order of limits (time vs. space)
5. Forgetting the partial time derivative in Ehrenfest

---

## Connections to Other Topics

| This Week's Topic | Connection |
|-------------------|------------|
| Measurement | Foundation for quantum computing, decoherence |
| Time evolution | Essential for all dynamics problems |
| Propagator | Key to path integrals, scattering |
| Ehrenfest | Connects to classical limit |

---

## Resources

### Primary Texts
- Shankar, Chapters 4, 6
- Sakurai, Chapter 2
- Griffiths, Chapter 1, 3

### Supplementary
- MIT 8.04 Lecture Notes (Time Evolution)
- Cohen-Tannoudji, Chapter III

---

## Next Week Preview

**Week 147: One-Dimensional Systems**

Applying the formalism to concrete problems:
- Infinite and finite square wells
- Quantum harmonic oscillator
- Delta function potentials
- Wave packet dynamics

---

*Week 146 of the QSE Self-Study Curriculum*
