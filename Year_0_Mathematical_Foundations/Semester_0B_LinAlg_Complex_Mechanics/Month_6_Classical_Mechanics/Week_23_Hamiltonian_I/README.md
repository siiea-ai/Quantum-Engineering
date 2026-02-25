# Week 23: Hamiltonian Mechanics I — Foundations

## Overview

**Month 6, Week 3 | Days 155-161 | Classical Mechanics**

This week establishes the foundations of **Hamiltonian Mechanics** — the elegant reformulation of classical mechanics that serves as the bridge to quantum mechanics. We develop the geometric and algebraic structures (phase space, symplectic geometry, Poisson brackets) that underpin all of modern theoretical physics.

---

## Learning Goals

By the end of this week, you will be able to:

1. **Construct Hamiltonians** using the Legendre transform from Lagrangians
2. **Solve Hamilton's equations** for standard mechanical systems
3. **Analyze phase space** geometry, fixed points, and separatrices
4. **Compute Poisson brackets** and use them for time evolution and conservation
5. **Apply Noether's theorem** to connect symmetries and conservation laws
6. **Implement symplectic integrators** that preserve phase space structure
7. **Explain the classical-quantum correspondence** via the Dirac prescription

---

## Daily Schedule

| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| **155** | Monday | Legendre Transform | H = Σpᵢq̇ᵢ - L, convex duality, canonical momenta |
| **156** | Tuesday | Hamilton's Equations | q̇ = ∂H/∂p, ṗ = -∂H/∂q, symplectic form ż = J∇H |
| **157** | Wednesday | Phase Space | 2n-dimensional manifold, trajectories, fixed points, separatrices |
| **158** | Thursday | Poisson Brackets | {f,g} definition, Jacobi identity, df/dt = {f,H} |
| **159** | Friday | Constants of Motion | {f,H} = 0, Noether's theorem, integrable systems |
| **160** | Saturday | Computational Lab | Symplectic integrators, Poincaré sections, chaos |
| **161** | Sunday | Week Review | Problem sets, self-assessment, Week 24 preview |

---

## Core Content

### The Hamiltonian Framework

The transition from Lagrangian to Hamiltonian mechanics:

$$L(q, \dot{q}, t) \xrightarrow{\text{Legendre}} H(q, p, t)$$

**Hamiltonian:**
$$H = \sum_i p_i \dot{q}_i - L, \quad p_i = \frac{\partial L}{\partial \dot{q}_i}$$

**Hamilton's Equations:**
$$\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}$$

### Phase Space Geometry

Phase space is the 2n-dimensional manifold of (q₁, ..., qₙ, p₁, ..., pₙ) equipped with the symplectic structure ω = Σdpᵢ∧dqᵢ.

**Key Properties:**
- Trajectories cannot cross (determinism)
- Phase space volume is preserved (Liouville's theorem)
- Fixed points are centers or saddles (for Hamiltonian systems)

### Poisson Brackets

The fundamental algebraic structure of Hamiltonian mechanics:

$$\{f, g\} = \sum_i \left(\frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i}\right)$$

**Time Evolution:**
$$\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t}$$

**Conservation:** {f, H} = 0 ↔ f is conserved

### Classical-Quantum Bridge

The Dirac correspondence connects classical and quantum mechanics:

$$\{A, B\}_{\text{classical}} \longleftrightarrow \frac{1}{i\hbar}[\hat{A}, \hat{B}]_{\text{quantum}}$$

This maps:
- {q, p} = 1 → [q̂, p̂] = iℏ (canonical commutation relations)
- Poisson algebra → Heisenberg algebra
- Classical observables → Quantum operators

---

## Key Equations Summary

| Concept | Formula |
|---------|---------|
| Hamiltonian | H = Σpᵢq̇ᵢ - L |
| Hamilton's equations | q̇ = ∂H/∂p, ṗ = -∂H/∂q |
| Symplectic form | ż = J∇H |
| Poisson bracket | {f,g} = Σ(∂f/∂q·∂g/∂p - ∂f/∂p·∂g/∂q) |
| Time evolution | df/dt = {f,H} + ∂f/∂t |
| Conservation | {f,H} = 0 |
| Fundamental brackets | {qᵢ,pⱼ} = δᵢⱼ |
| Angular momentum | {Lᵢ,Lⱼ} = εᵢⱼₖLₖ |

---

## Quantum Mechanics Connections

This week establishes critical connections to quantum mechanics:

| Classical | Quantum | Significance |
|-----------|---------|--------------|
| Phase space point | State vector \|ψ⟩ | States |
| Observable f(q,p) | Operator f̂ | Observables |
| Poisson bracket {,} | Commutator [,]/(iℏ) | Algebra |
| {f,H} = 0 | [f̂,Ĥ] = 0 | Conservation/good quantum numbers |
| Hamilton's equations | Heisenberg equations | Dynamics |
| Phase space distribution | Wigner function | Quasi-probability |

---

## Computational Skills

This week develops essential programming skills:

1. **Symbolic Computing:** Poisson brackets with SymPy
2. **Phase Portraits:** Streamplots and energy contours
3. **Numerical Integration:** Comparing integrators
4. **Symplectic Methods:** Störmer-Verlet algorithm
5. **Chaos Analysis:** Poincaré sections

---

## Prerequisites

- Month 6, Weeks 21-22: Lagrangian mechanics
- Differential equations (characteristic equations)
- Linear algebra (eigenvalues, matrices)
- Multivariable calculus (partial derivatives)

---

## Resources

### Primary Texts
- Goldstein, *Classical Mechanics*, Chapters 8-10
- Landau & Lifshitz, *Mechanics*, Chapters 7-8
- Arnold, *Mathematical Methods of Classical Mechanics*

### Supplementary
- David Tong, Cambridge Lecture Notes on Classical Dynamics
- MIT OCW 8.09 Classical Mechanics III
- Shankar, *Principles of Quantum Mechanics* (for correspondence)

---

## Assessment

### Weekly Problem Sets
- **Set A:** Conceptual questions (Day 161)
- **Set B:** Calculations (Day 161)

### Computational Projects
- Phase space visualizer
- Symplectic integrator comparison
- Poincaré section generator

### Self-Assessment
- Checklist in Day 161
- Identify areas needing review before Week 24

---

## Next Week Preview

**Week 24: Hamiltonian Mechanics II** covers advanced topics:
- Canonical transformations
- Liouville's theorem (full treatment)
- Action-angle variables
- Hamilton-Jacobi equation
- Introduction to chaos
- Year 0 completion!

---

## Study Tips

1. **Visualize:** Phase space is geometric — draw pictures!
2. **Compute:** Practice Poisson brackets until automatic
3. **Connect:** Always ask "What's the quantum analog?"
4. **Code:** Implement methods to deepen understanding
5. **Review:** This material is foundational for all of Year 1

---

*Week 23 of Month 6 | Year 0: Mathematical & Physical Foundations*
*QSE Self-Study Curriculum*
