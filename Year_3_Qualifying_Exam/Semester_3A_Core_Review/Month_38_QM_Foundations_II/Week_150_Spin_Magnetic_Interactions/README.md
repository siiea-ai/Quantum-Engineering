# Week 150: Spin and Magnetic Interactions

## Overview

**Days:** 1044-1050 (7 days)
**Theme:** Complete mastery of spin-1/2 formalism and magnetic field interactions for PhD qualifying exams
**Focus:** Pauli matrices, spinor algebra, Stern-Gerlach, spin precession, magnetic moments

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Derive** properties of Pauli matrices and spinor representations
2. **Calculate** measurement probabilities for arbitrary spin-1/2 states
3. **Analyze** Stern-Gerlach experiments with multiple stages
4. **Solve** spin precession problems in time-varying magnetic fields
5. **Apply** spin formalism to magnetic resonance phenomena
6. **Connect** spin physics to quantum computing qubit operations

---

## Daily Schedule

### Day 1044 (Monday): Spin-1/2 Formalism

**Focus:** Intrinsic angular momentum and spin operators

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Sections 1-2 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 1-6 |
| Evening | 7:00-8:30 | Oral Practice: Spin basics |

**Key Topics:**
- Intrinsic angular momentum vs orbital
- Spin-1/2 states $|\uparrow\rangle$, $|\downarrow\rangle$
- Pauli matrices and their properties

### Day 1045 (Tuesday): Pauli Matrix Algebra

**Focus:** Properties, identities, and rotations

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Sections 3-4 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 7-12 |
| Evening | 7:00-8:30 | Matrix manipulation practice |

**Key Topics:**
- Pauli matrix algebra and identities
- Spinor rotations
- General spin-1/2 state on Bloch sphere

### Day 1046 (Wednesday): Stern-Gerlach Experiment

**Focus:** Sequential measurements and state preparation

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 5 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 13-17 |
| Evening | 7:00-8:30 | Visualization and diagrams |

**Key Topics:**
- Single Stern-Gerlach apparatus
- Sequential measurements
- Quantum eraser concepts

### Day 1047 (Thursday): Spin Precession

**Focus:** Time evolution in magnetic fields

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 6 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 18-22 |
| Evening | 7:00-8:30 | Larmor precession problems |

**Key Topics:**
- Hamiltonian in magnetic field
- Larmor frequency
- Rabi oscillations

### Day 1048 (Friday): Magnetic Moments and NMR

**Focus:** Applications to magnetic resonance

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 7 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 23-26 |
| Evening | 7:00-8:30 | NMR/ESR physics |

**Key Topics:**
- Magnetic dipole moment
- Gyromagnetic ratio
- Resonance conditions

### Day 1049 (Saturday): Problem Solving Session

**Focus:** Exam-style problem solving under timed conditions

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Timed problems (6 problems, 3 hours) |
| Afternoon | 2:00-5:00 | Solution review and gap analysis |
| Evening | 7:00-8:30 | Oral practice |

### Day 1050 (Sunday): Integration and Assessment

**Focus:** Self-assessment and week review

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Remaining problems |
| Afternoon | 2:00-5:00 | Self-assessment checklist |
| Evening | 7:00-8:30 | Preview Week 151 |

---

## Key Equations

### Pauli Matrices

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### Spin Operators

$$\mathbf{S} = \frac{\hbar}{2}\boldsymbol{\sigma}$$

$$S^2 = \frac{3\hbar^2}{4}I, \quad S_z|\pm\rangle = \pm\frac{\hbar}{2}|\pm\rangle$$

### General Spin State

$$|\psi\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + e^{i\phi}\sin\frac{\theta}{2}|\downarrow\rangle$$

### Pauli Matrix Identities

$$\sigma_i\sigma_j = \delta_{ij}I + i\epsilon_{ijk}\sigma_k$$

$$(\boldsymbol{\sigma}\cdot\mathbf{a})(\boldsymbol{\sigma}\cdot\mathbf{b}) = \mathbf{a}\cdot\mathbf{b}\,I + i\boldsymbol{\sigma}\cdot(\mathbf{a}\times\mathbf{b})$$

### Magnetic Hamiltonian

$$H = -\boldsymbol{\mu}\cdot\mathbf{B} = -\gamma\mathbf{S}\cdot\mathbf{B} = -\frac{\gamma\hbar}{2}\boldsymbol{\sigma}\cdot\mathbf{B}$$

### Larmor Frequency

$$\omega_L = \gamma B = \frac{g_s e B}{2m_e}$$

---

## Connection to Quantum Computing

| Spin Concept | Qubit Equivalent |
|--------------|------------------|
| $|\uparrow\rangle$, $|\downarrow\rangle$ | $|0\rangle$, $|1\rangle$ |
| Pauli matrices | Pauli gates X, Y, Z |
| Spin rotation | Single-qubit gate |
| Bloch sphere | State visualization |
| Larmor precession | Z rotation |
| Rabi oscillation | X rotation |

---

## Week Files

| File | Purpose |
|------|---------|
| `Review_Guide.md` | Comprehensive topic review (2500+ words) |
| `Problem_Set.md` | 28 qualifying exam problems |
| `Problem_Solutions.md` | Detailed step-by-step solutions |
| `Oral_Practice.md` | Oral exam questions and frameworks |
| `Self_Assessment.md` | Progress tracking checklist |

---

## Resources

### Primary References
- Shankar, Chapter 14
- Sakurai, Chapter 1 & 3
- Griffiths, Chapter 4.4

### Supplementary
- Feynman Lectures Vol. III, Chapters 5-6
- Nielsen & Chuang, Chapter 1 (qubit connection)

---

**Created:** February 9, 2026
**Week:** 150 of 192
**Progress:** 0/7 days
