# Week 36: Spectral Theory

## Overview

This week represents the culmination of our functional analysis journey, where we develop the spectral theory that forms the mathematical backbone of quantum mechanics. Spectral theory provides the rigorous framework for understanding measurements, observables, and time evolution in quantum systems. The spectral theorem—often called the most important theorem in quantum mechanics—tells us that self-adjoint operators can be "diagonalized" in a generalized sense, with their spectrum corresponding to possible measurement outcomes.

## Week Objectives

By the end of this week, you will be able to:

1. **Classify spectra** of linear operators into point, continuous, and residual components
2. **Prove and apply** the spectral theorem for compact and bounded self-adjoint operators
3. **Construct functional calculus** to define functions of operators via spectral measures
4. **Analyze unbounded operators** with proper attention to domains and self-adjointness criteria
5. **Derive Stone's theorem** connecting self-adjoint generators to unitary groups
6. **Apply spectral theory** to quantum mechanical observables and time evolution

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **Day 246** (Mon) | Spectrum of an Operator | Point/continuous/residual spectrum, resolvent, spectral radius |
| **Day 247** (Tue) | Spectral Theorem: Compact Case | Hilbert-Schmidt theorem, eigenvalue decomposition, nuclear operators |
| **Day 248** (Wed) | Spectral Theorem: Bounded Case | Spectral measure, projection-valued measure, resolution of identity |
| **Day 249** (Thu) | Functional Calculus | f(A) via spectral integral, continuous/Borel functional calculus |
| **Day 250** (Fri) | Unbounded Operators | Domains, closed/closable operators, self-adjointness criteria |
| **Day 251** (Sat) | Stone's Theorem | One-parameter unitary groups, generators, Schrödinger equation |
| **Day 252** (Sun) | Month 9 Review | Comprehensive synthesis of functional analysis |

## Mathematical Prerequisites

- Hilbert space theory (Week 33)
- Bounded linear operators (Week 34)
- Compact operators (Week 35)
- Complex analysis (contour integration)
- Measure theory fundamentals

## Key Theorems This Week

### Spectral Theorem (Compact Self-Adjoint)
For compact self-adjoint $A$ on Hilbert space $\mathcal{H}$:
$$A = \sum_{n=1}^{\infty} \lambda_n |e_n\rangle\langle e_n|$$
where $\{\lambda_n\}$ are eigenvalues (real, $\lambda_n \to 0$) and $\{e_n\}$ orthonormal eigenvectors.

### Spectral Theorem (Bounded Self-Adjoint)
For bounded self-adjoint $A$:
$$A = \int_{\sigma(A)} \lambda \, dE_\lambda$$
where $\{E_\lambda\}$ is the spectral family (projection-valued measure).

### Functional Calculus
For Borel function $f$ on $\sigma(A)$:
$$f(A) = \int_{\sigma(A)} f(\lambda) \, dE_\lambda$$

### Stone's Theorem
$\{U(t)\}_{t \in \mathbb{R}}$ is a strongly continuous one-parameter unitary group if and only if:
$$U(t) = e^{-iAt}$$
for some self-adjoint operator $A$ (the generator).

## Quantum Mechanics Connections

| Mathematical Concept | Quantum Mechanical Meaning |
|---------------------|---------------------------|
| Spectrum $\sigma(A)$ | Possible measurement outcomes |
| Eigenvalues (point spectrum) | Discrete energy levels, quantization |
| Continuous spectrum | Scattering states, free particle momenta |
| Spectral projections $E_\lambda$ | Measurement collapse operators |
| $f(A)$ via functional calculus | Functions of observables |
| Unbounded operators | Position $\hat{x}$, momentum $\hat{p}$ |
| Stone's theorem | Schrödinger equation: $i\hbar\frac{d}{dt}|\psi\rangle = H|\psi\rangle$ |

## Computational Focus

- Numerical eigenvalue computation for large matrices
- Power method and inverse iteration
- Spectral decomposition visualization
- Verification of spectral theorem
- Time evolution via matrix exponentials
- Unbounded operator approximations

## Study Tips

1. **Visualize spectra**: Draw the complex plane and mark point/continuous/residual regions
2. **Start with finite dimensions**: Understand spectral theory for matrices first
3. **Connect to physics**: Every theorem has a physical interpretation in QM
4. **Mind the domains**: Unbounded operators require careful domain specification
5. **Use symmetry**: Self-adjointness simplifies everything dramatically

## Resources

### Primary Texts
- Reed & Simon, *Methods of Modern Mathematical Physics*, Vol. I, Chapters 7-8
- Conway, *A Course in Functional Analysis*, Chapters 9-10
- Kreyszig, *Introductory Functional Analysis*, Chapters 7-9

### Supplementary
- Hall, *Quantum Theory for Mathematicians*, Chapters 9-10
- Teschl, *Mathematical Methods in Quantum Mechanics*, Chapters 2-4
- Blank, Exner, Havlíček, *Hilbert Space Operators in Quantum Physics*

### Online Resources
- MIT OCW 18.102 Functional Analysis
- Stanford Math 205A/B notes on spectral theory
- Physics LibreTexts: Spectral Theory for Physicists

## Assessment Goals

By Sunday's review, you should be able to:

- [ ] Compute spectra and resolvents for specific operators
- [ ] State and prove the spectral theorem (compact case)
- [ ] Construct spectral measures for bounded self-adjoint operators
- [ ] Apply functional calculus to compute $f(A)$
- [ ] Determine domains and check self-adjointness for unbounded operators
- [ ] Derive the Schrödinger equation from Stone's theorem
- [ ] Explain the physical significance of every mathematical concept

## Week 36 in Context

```
Week 33: Hilbert Spaces ──┐
Week 34: Bounded Operators ├──► Week 36: Spectral Theory ──► Quantum Mechanics
Week 35: Compact Operators ─┘                                  Applications
```

This week synthesizes all previous functional analysis into the spectral theory that quantum mechanics requires. Every concept—Hilbert spaces, operators, compactness—comes together in the spectral theorem and its applications to physics.

---

*"The spectral theorem is the fundamental result of Hilbert space theory. It provides the mathematical framework for the uncertainty principle, the measurement postulate, and all of quantum mechanics."*
— Michael Reed & Barry Simon
