# Day 322: Physics Integration Exam

## Overview

**Month 12, Week 46, Day 7 — Sunday**

Today tests mastery of all Year 0 physics through a comprehensive exam integrating Newtonian, Lagrangian, and Hamiltonian mechanics with symmetry principles.

## Exam Instructions

- Time: 4 hours
- Show all work
- Connect to quantum mechanics where relevant

---

## Part A: Newtonian Mechanics (20 points)

### Problem A1 (10 points)

A particle of mass $m$ moves under a central force $F(r) = -kr^n$, where $k > 0$.

(a) For what values of $n$ are circular orbits stable?
(b) Find the period of small radial oscillations about a circular orbit.

### Problem A2 (10 points)

Two masses $m_1$ and $m_2$ interact via gravity. Transform to center-of-mass and relative coordinates and reduce to an effective one-body problem.

---

## Part B: Lagrangian Mechanics (25 points)

### Problem B1 (15 points)

A bead of mass $m$ slides on a frictionless wire bent into a helix: $x = a\cos\phi$, $y = a\sin\phi$, $z = b\phi$.

(a) Write the Lagrangian using $\phi$ as the generalized coordinate
(b) Find the equation of motion
(c) Identify any conserved quantities

### Problem B2 (10 points)

Prove Noether's theorem: if $L$ is invariant under $q \to q + \epsilon\eta(q, t)$, then:
$$Q = \frac{\partial L}{\partial \dot{q}}\eta$$
is conserved.

---

## Part C: Hamiltonian Mechanics (25 points)

### Problem C1 (15 points)

For the 2D harmonic oscillator:
$$H = \frac{1}{2m}(p_x^2 + p_y^2) + \frac{m\omega^2}{2}(x^2 + y^2)$$

(a) Compute $\{L_z, H\}$ and interpret
(b) Find the generating function for transformation to action-angle variables
(c) Show that in action-angle variables, the motion is linear in time

### Problem C2 (10 points)

Show that canonical transformations preserve Poisson brackets:
$$\{Q, P\}_{q,p} = 1$$

---

## Part D: Quantum Preview (30 points)

### Problem D1 (15 points)

For the harmonic oscillator, compare classical and quantum:
(a) Classical: Write Hamilton's equations and solve
(b) Quantum: Write the commutator $[\hat{a}, \hat{a}^\dagger] = 1$ and find energy levels
(c) Show correspondence in the classical limit

### Problem D2 (15 points)

The classical angular momentum satisfies $\{L_i, L_j\} = \epsilon_{ijk}L_k$.
(a) Write the quantum commutation relations
(b) Find $[\hat{L}_z, \hat{L}^2]$
(c) Explain why angular momentum is quantized

---

## Solutions Outline

### A1 Solution

Effective potential: $V_{eff} = V(r) + L^2/(2mr^2)$

Stability requires $V_{eff}''(r_0) > 0$ at circular orbit.

### B1 Solution

$L = \frac{1}{2}m(a^2 + b^2)\dot{\phi}^2 - mgb\phi$

$\phi$ not cyclic (gravity), but total energy conserved.

### C1 Solution

$\{L_z, H\} = 0$ (rotational symmetry), so $L_z$ conserved.

### D1 Solution

Classical: $x(t) = A\cos(\omega t + \phi)$

Quantum: $E_n = \hbar\omega(n + 1/2)$

Correspondence: For large $n$, spacing $\hbar\omega$ small compared to $E_n$.

---

## Self-Assessment

Score yourself and identify areas for review before Year 1.

---

## Preview: Week 47

Next week: **Computational Capstone Project** — integrating all Year 0 skills.
