# Week 149: Orbital Angular Momentum

## Overview

**Days:** 1037-1043 (7 days)
**Theme:** Complete mastery of orbital angular momentum for PhD qualifying exams
**Focus:** Commutation relations, eigenvalue problems, spherical harmonics, hydrogen atom

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Derive** the angular momentum commutation relations from first principles
2. **Prove** the eigenvalue spectrum of $L^2$ and $L_z$ using ladder operators
3. **Calculate** spherical harmonics for specific $l,m$ values
4. **Solve** the hydrogen atom eigenvalue problem completely
5. **Apply** angular momentum techniques to central potential problems
6. **Explain** orbital angular momentum concepts in oral exam format

---

## Daily Schedule

### Day 1037 (Monday): Angular Momentum Algebra

**Focus:** Commutation relations and operator algebra

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Sections 1-2 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 1-6 |
| Evening | 7:00-8:30 | Oral Practice: Derivations |

**Key Topics:**
- Definition of orbital angular momentum: $\mathbf{L} = \mathbf{r} \times \mathbf{p}$
- Commutation relations: $[L_i, L_j] = i\hbar\epsilon_{ijk}L_k$
- $[L^2, L_z] = 0$ and simultaneous eigenstates

### Day 1038 (Tuesday): Eigenvalue Problem

**Focus:** Ladder operators and eigenvalue spectrum

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Sections 3-4 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 7-12 |
| Evening | 7:00-8:30 | Derivation practice |

**Key Topics:**
- Ladder operators: $L_{\pm} = L_x \pm iL_y$
- Eigenvalue spectrum derivation
- Action of $L_{\pm}$ on $|l,m\rangle$ states

### Day 1039 (Wednesday): Spherical Harmonics

**Focus:** Explicit form and properties of $Y_l^m(\theta,\phi)$

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 5 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 13-17 |
| Evening | 7:00-8:30 | Visualization and computation |

**Key Topics:**
- Differential equation for spherical harmonics
- Legendre polynomials and associated Legendre functions
- Orthonormality and completeness relations

### Day 1040 (Thursday): Central Potentials

**Focus:** Separation of variables and radial equation

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 6 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 18-21 |
| Evening | 7:00-8:30 | 3D problem techniques |

**Key Topics:**
- Separation of Schrödinger equation in spherical coordinates
- Effective radial potential
- General properties of central force problems

### Day 1041 (Friday): Hydrogen Atom

**Focus:** Complete solution of the hydrogen atom

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 7 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 22-25 |
| Evening | 7:00-8:30 | Energy level calculations |

**Key Topics:**
- Coulomb potential: $V(r) = -e^2/(4\pi\epsilon_0 r)$
- Bound state energies: $E_n = -13.6\text{ eV}/n^2$
- Radial wave functions and degeneracy

### Day 1042 (Saturday): Problem Solving Session

**Focus:** Exam-style problem solving under timed conditions

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Timed problems (6 problems, 3 hours) |
| Afternoon | 2:00-5:00 | Solution review and gap analysis |
| Evening | 7:00-8:30 | Oral practice |

### Day 1043 (Sunday): Integration and Assessment

**Focus:** Self-assessment and week review

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Remaining problems from set |
| Afternoon | 2:00-5:00 | Self-assessment checklist |
| Evening | 7:00-8:30 | Preview Week 150 |

---

## Week Files

| File | Purpose |
|------|---------|
| `Review_Guide.md` | Comprehensive topic review (2000+ words) |
| `Problem_Set.md` | 25 qualifying exam problems |
| `Problem_Solutions.md` | Detailed step-by-step solutions |
| `Oral_Practice.md` | Oral exam questions and frameworks |
| `Self_Assessment.md` | Progress tracking checklist |

---

## Key Equations

### Fundamental Commutation Relations

$$[L_x, L_y] = i\hbar L_z, \quad [L_y, L_z] = i\hbar L_x, \quad [L_z, L_x] = i\hbar L_y$$

$$[L^2, L_i] = 0 \quad \text{for } i = x,y,z$$

### Eigenvalue Equations

$$L^2|l,m\rangle = \hbar^2 l(l+1)|l,m\rangle$$

$$L_z|l,m\rangle = \hbar m|l,m\rangle$$

where $l = 0, 1, 2, \ldots$ and $m = -l, -l+1, \ldots, l-1, l$

### Ladder Operators

$$L_{\pm}|l,m\rangle = \hbar\sqrt{l(l+1)-m(m\pm 1)}|l,m\pm 1\rangle$$

$$L_{\pm}|l,\pm l\rangle = 0$$

### Spherical Harmonics

$$Y_l^m(\theta,\phi) = (-1)^m\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\phi}$$

**Low-order examples:**

$$Y_0^0 = \frac{1}{\sqrt{4\pi}}$$

$$Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta$$

$$Y_1^{\pm 1} = \mp\sqrt{\frac{3}{8\pi}}\sin\theta\, e^{\pm i\phi}$$

### Hydrogen Atom

$$E_n = -\frac{13.6\text{ eV}}{n^2} = -\frac{m_e e^4}{2(4\pi\epsilon_0)^2\hbar^2 n^2}$$

$$a_0 = \frac{4\pi\epsilon_0\hbar^2}{m_e e^2} \approx 0.529\text{ \AA}$$

---

## Common Exam Mistakes to Avoid

1. **Forgetting the $\hbar^2$ in $L^2$ eigenvalue:** $L^2|l,m\rangle = \hbar^2 l(l+1)|l,m\rangle$, not $l(l+1)$

2. **Confusing $l$ and $m$:** $l$ determines the magnitude, $m$ determines the z-component

3. **Wrong normalization for ladder operators:** The factor is $\sqrt{l(l+1)-m(m\pm 1)}$

4. **Forgetting degeneracy:** The $n$-th level of hydrogen has $n^2$ degenerate states

5. **Sign errors in spherical harmonics:** Watch the $(-1)^m$ phase convention

---

## Resources

### Primary References
- Shankar, Chapter 12
- Sakurai, Chapter 3
- Griffiths, Chapter 4

### Problem Sources
- Yale qualifying exams
- MIT 8.04/8.05 problem sets
- *Problems and Solutions on Quantum Mechanics*

---

**Created:** February 9, 2026
**Week:** 149 of 192
**Progress:** 0/7 days
