# Week 152: Perturbation Theory

## Overview

**Days:** 1058-1064 (7 days)
**Theme:** Complete mastery of perturbation methods for PhD qualifying exams
**Focus:** Non-degenerate, degenerate, time-dependent perturbation, Fermi's golden rule, adiabatic theorem, Berry phase

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Apply** non-degenerate perturbation theory to first and second order
2. **Solve** degenerate perturbation problems by diagonalization
3. **Calculate** transition probabilities using time-dependent perturbation
4. **Use** Fermi's golden rule for transition rates to continua
5. **Explain** the adiabatic theorem and calculate Berry phases
6. **Recognize** when each perturbation method is appropriate

---

## Daily Schedule

### Day 1058 (Monday): Non-Degenerate Perturbation Theory

**Focus:** First and second order corrections

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Sections 1-2 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 1-6 |
| Evening | 7:00-8:30 | Oral Practice |

**Key Topics:**
- Perturbation expansion setup
- First-order energy and state corrections
- Second-order energy correction
- Convergence conditions

### Day 1059 (Tuesday): Degenerate Perturbation Theory

**Focus:** Handling degenerate subspaces

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 3 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 7-12 |
| Evening | 7:00-8:30 | Stark effect problems |

**Key Topics:**
- Why non-degenerate theory fails
- Diagonalization in degenerate subspace
- Finding good quantum numbers
- Stark effect in hydrogen

### Day 1060 (Wednesday): Time-Dependent Perturbation I

**Focus:** Transition amplitudes and probabilities

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 4 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 13-17 |
| Evening | 7:00-8:30 | Sinusoidal perturbation |

**Key Topics:**
- Interaction picture
- Transition amplitude formula
- Sinusoidal perturbations
- Resonance conditions

### Day 1061 (Thursday): Time-Dependent Perturbation II

**Focus:** Fermi's golden rule

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Section 5 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 18-22 |
| Evening | 7:00-8:30 | Transition rate problems |

**Key Topics:**
- Fermi's golden rule derivation
- Density of states
- Applications: decay rates, absorption

### Day 1062 (Friday): Adiabatic Theorem and Berry Phase

**Focus:** Geometric phases

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Review Guide: Sections 6-7 |
| Afternoon | 2:00-5:00 | Problem Set: Problems 23-27 |
| Evening | 7:00-8:30 | Berry phase examples |

**Key Topics:**
- Adiabatic theorem statement
- Geometric vs dynamical phase
- Berry phase calculation
- Aharonov-Bohm effect connection

### Day 1063 (Saturday): Problem Solving Session

**Focus:** Timed exam-style problems

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Timed problems (3 hours) |
| Afternoon | 2:00-5:00 | Solution review |
| Evening | 7:00-8:30 | Oral practice |

### Day 1064 (Sunday): Month Integration

**Focus:** Month 38 review and assessment

| Block | Time | Activity |
|-------|------|----------|
| Morning | 9:00-12:00 | Month comprehensive review |
| Afternoon | 2:00-5:00 | Self-assessment |
| Evening | 7:00-8:30 | Prepare for Month 39 |

---

## Key Equations

### Non-Degenerate Perturbation

**First-order energy:**
$$\boxed{E_n^{(1)} = \langle n^{(0)}|H'|n^{(0)}\rangle}$$

**First-order state:**
$$|n^{(1)}\rangle = \sum_{k\neq n}\frac{\langle k^{(0)}|H'|n^{(0)}\rangle}{E_n^{(0)} - E_k^{(0)}}|k^{(0)}\rangle$$

**Second-order energy:**
$$\boxed{E_n^{(2)} = \sum_{k\neq n}\frac{|\langle k^{(0)}|H'|n^{(0)}\rangle|^2}{E_n^{(0)} - E_k^{(0)}}}$$

### Degenerate Perturbation

Diagonalize $H'$ in the degenerate subspace:
$$\det(H'_{ij} - E^{(1)}\delta_{ij}) = 0$$

where $H'_{ij} = \langle i^{(0)}|H'|j^{(0)}\rangle$ for degenerate states $i, j$.

### Time-Dependent Perturbation

**Transition amplitude:**
$$c_f(t) = -\frac{i}{\hbar}\int_0^t dt'\, \langle f|H'(t')|i\rangle e^{i\omega_{fi}t'}$$

where $\omega_{fi} = (E_f - E_i)/\hbar$.

### Fermi's Golden Rule

$$\boxed{\Gamma_{i\to f} = \frac{2\pi}{\hbar}|\langle f|H'|i\rangle|^2\rho(E_f)}$$

where $\rho(E_f)$ is the density of final states at energy $E_f$.

### Adiabatic Theorem

For slowly varying $H(t)$:
$$|\psi(t)\rangle \approx e^{i\gamma_n(t)}e^{-i\theta_n(t)}|n(t)\rangle$$

where:
- $\theta_n(t) = \frac{1}{\hbar}\int_0^t E_n(t')dt'$ (dynamical phase)
- $\gamma_n(t) = i\int_0^t \langle n(t')|\dot{n}(t')\rangle dt'$ (geometric phase)

### Berry Phase

$$\boxed{\gamma_n = i\oint \langle n(\mathbf{R})|\nabla_{\mathbf{R}}|n(\mathbf{R})\rangle \cdot d\mathbf{R}}$$

For a spin-1/2 in a rotating magnetic field:
$$\gamma = -\frac{\Omega}{2}$$

where $\Omega$ is the solid angle enclosed.

---

## Common Applications

| Problem Type | Method | Key Formula |
|--------------|--------|-------------|
| Energy shifts (non-degen) | Non-degen PT | $E^{(1)} = \langle H'\rangle$ |
| Lifting degeneracy | Degen PT | Diagonalize $H'$ |
| Stark effect (linear) | Degen PT | $n=2$ hydrogen |
| Zeeman effect | Non-degen or degen | Depends on field strength |
| Absorption spectrum | Time-dep PT | Fermi's golden rule |
| Decay rates | Fermi's golden rule | $\Gamma \propto |M|^2\rho$ |
| Spin in rotating B | Adiabatic + Berry | $\gamma = -\Omega/2$ |

---

## Week Files

| File | Purpose |
|------|---------|
| `Review_Guide.md` | Comprehensive review (3000+ words) |
| `Problem_Set.md` | 30 qualifying exam problems |
| `Problem_Solutions.md` | Detailed solutions |
| `Oral_Practice.md` | Oral exam preparation |
| `Self_Assessment.md` | Progress tracking |

---

## Resources

### Primary References
- Shankar, Chapter 17
- Sakurai, Chapter 5
- Griffiths, Chapters 7-9

### MIT OCW
- [8.06 Quantum Physics III](https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/)

---

**Created:** February 9, 2026
**Week:** 152 of 192
**Progress:** 0/7 days
