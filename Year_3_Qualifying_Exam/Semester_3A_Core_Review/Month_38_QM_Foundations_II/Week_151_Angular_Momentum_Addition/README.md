# Week 151: Angular Momentum Addition

## Overview

**Days:** 1051-1057 (7 days)
**Theme:** Mastery of angular momentum coupling and Clebsch-Gordan coefficients for PhD qualifying exams
**Focus:** Tensor products, Clebsch-Gordan coefficients, spin-orbit coupling, selection rules

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Construct** coupled angular momentum states from uncoupled basis
2. **Calculate** Clebsch-Gordan coefficients using recursion relations
3. **Apply** angular momentum coupling to spin-orbit problems
4. **Derive** selection rules from angular momentum conservation
5. **Interpret** spectroscopic term symbols and atomic multiplets
6. **Solve** qualifying exam problems on angular momentum addition

---

## Daily Schedule

### Day 1051 (Monday): Tensor Product Spaces

**Focus:** Combining angular momentum systems

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Sections 1-2 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 1-5 |
| Evening | 7:00-8:30 | Oral Practice: Tensor products |

**Key Topics:**
- Tensor product of Hilbert spaces
- Uncoupled vs coupled basis
- Allowed values of total angular momentum

### Day 1052 (Tuesday): Clebsch-Gordan Coefficients I

**Focus:** Definition, properties, and simple cases

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 3 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 6-10 |
| Evening | 7:00-8:30 | CG coefficient tables |

**Key Topics:**
- Definition of CG coefficients
- Orthogonality relations
- Simple cases: $j_2 = 1/2$

### Day 1053 (Wednesday): Clebsch-Gordan Coefficients II

**Focus:** Recursion relations and calculations

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 4 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 11-15 |
| Evening | 7:00-8:30 | Calculation practice |

**Key Topics:**
- Recursion relations
- Calculation techniques
- Symmetry properties

### Day 1054 (Thursday): Spin-Orbit Coupling

**Focus:** Coupling of orbital and spin angular momentum

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 5 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 16-20 |
| Evening | 7:00-8:30 | Fine structure problems |

**Key Topics:**
- $\mathbf{L}\cdot\mathbf{S}$ operator
- Fine structure energy
- Good quantum numbers

### Day 1055 (Friday): Selection Rules and Spectroscopy

**Focus:** Transition rules and term symbols

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Sections 6-7 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 21-25 |
| Evening | 7:00-8:30 | Spectroscopy practice |

**Key Topics:**
- Electric dipole selection rules
- Term symbol notation
- Multiplet structure

### Day 1056 (Saturday): Problem Solving Session

**Focus:** Exam-style timed problems

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Timed problems (6 problems, 3 hours) |
| Afternoon | 2:00-5:00 | Solution review |
| Evening | 7:00-8:30 | Oral practice |

### Day 1057 (Sunday): Integration and Assessment

**Focus:** Self-assessment and consolidation

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Remaining problems |
| Afternoon | 2:00-5:00 | Self-assessment |
| Evening | 7:00-8:30 | Preview Week 152 |

---

## Key Equations

### Angular Momentum Addition

For two angular momenta $\mathbf{J}_1$ and $\mathbf{J}_2$, the total is $\mathbf{J} = \mathbf{J}_1 + \mathbf{J}_2$.

**Allowed values of $j$:**
$$j = |j_1 - j_2|, |j_1 - j_2| + 1, \ldots, j_1 + j_2 - 1, j_1 + j_2$$

### Clebsch-Gordan Expansion

$$|j,m\rangle = \sum_{m_1,m_2} \langle j_1,m_1;j_2,m_2|j,m\rangle |j_1,m_1\rangle|j_2,m_2\rangle$$

**Inverse:**
$$|j_1,m_1\rangle|j_2,m_2\rangle = \sum_{j} \langle j_1,m_1;j_2,m_2|j,m\rangle |j,m\rangle$$

### Selection Rules

$$m = m_1 + m_2$$

$$|j_1 - j_2| \leq j \leq j_1 + j_2$$

### Orthogonality

$$\sum_{m_1,m_2} \langle j_1,m_1;j_2,m_2|j,m\rangle\langle j_1,m_1;j_2,m_2|j',m'\rangle = \delta_{jj'}\delta_{mm'}$$

### Spin-Orbit Coupling

$$H_{SO} = \xi(r)\mathbf{L}\cdot\mathbf{S}$$

$$\mathbf{L}\cdot\mathbf{S} = \frac{1}{2}(J^2 - L^2 - S^2)$$

$$\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$$

### Fine Structure Energy

$$E_{nlj} = E_n^{(0)} + \frac{\hbar^2}{2}\langle\xi(r)\rangle[j(j+1) - l(l+1) - s(s+1)]$$

---

## Common CG Coefficients

### Adding $j_2 = 1/2$

For $j_1 \otimes \frac{1}{2}$, the allowed $j$ values are $j = j_1 \pm \frac{1}{2}$:

$$|j_1 + \frac{1}{2}, m\rangle = \sqrt{\frac{j_1 + m + \frac{1}{2}}{2j_1 + 1}}|j_1, m-\frac{1}{2}\rangle|\frac{1}{2}, +\frac{1}{2}\rangle + \sqrt{\frac{j_1 - m + \frac{1}{2}}{2j_1 + 1}}|j_1, m+\frac{1}{2}\rangle|\frac{1}{2}, -\frac{1}{2}\rangle$$

$$|j_1 - \frac{1}{2}, m\rangle = -\sqrt{\frac{j_1 - m + \frac{1}{2}}{2j_1 + 1}}|j_1, m-\frac{1}{2}\rangle|\frac{1}{2}, +\frac{1}{2}\rangle + \sqrt{\frac{j_1 + m + \frac{1}{2}}{2j_1 + 1}}|j_1, m+\frac{1}{2}\rangle|\frac{1}{2}, -\frac{1}{2}\rangle$$

### Adding $j_2 = 1$

For $j_1 = 1$ and $j_2 = 1$: $j \in \{0, 1, 2\}$

Key coefficients available in standard tables.

---

## Term Symbol Notation

$$^{2S+1}L_J$$

where:
- $S$ = total spin
- $L$ = total orbital ($S, P, D, F, \ldots$ for $L = 0, 1, 2, 3, \ldots$)
- $J$ = total angular momentum
- $2S+1$ = spin multiplicity

**Example:** $^2P_{3/2}$ means $S = 1/2$, $L = 1$, $J = 3/2$

---

## Week Files

| File | Purpose |
|------|---------|
| `Review_Guide.md` | Comprehensive topic review (2500+ words) |
| `Problem_Set.md` | 27 qualifying exam problems |
| `Problem_Solutions.md` | Detailed solutions with CG calculations |
| `Oral_Practice.md` | Oral exam questions |
| `Self_Assessment.md` | Progress checklist |

---

## Resources

### Primary References
- Shankar, Chapter 15
- Sakurai, Chapter 3.8
- Griffiths, Section 4.4

### CG Coefficient Tables
- PDG tables
- Condon-Shortley convention

---

**Created:** February 9, 2026
**Week:** 151 of 192
**Progress:** 0/7 days
