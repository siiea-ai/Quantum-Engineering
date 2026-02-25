# Week 26: Contour Integration

## Overview

**Days:** 176-182 (7 days)
**Status:** ✅ COMPLETE
**Focus:** The most powerful computational tool in complex analysis

This week develops contour integration from line integrals through Cauchy's theorems to applications in evaluating real integrals. These techniques are essential for quantum mechanics, where propagators, Green's functions, and scattering amplitudes are computed via contour methods.

---

## Daily Schedule

| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 176 | Monday | Complex Line Integrals | Parametrization, path dependence, ML inequality |
| 177 | Tuesday | Cauchy's Integral Theorem | Simply connected domains, winding number |
| 178 | Wednesday | Cauchy's Integral Formula | Values and derivatives from boundary, Liouville |
| 179 | Thursday | Real Integrals I | Semicircular contours, Jordan's lemma, Fourier |
| 180 | Friday | Real Integrals II | Branch cuts, keyhole contours, Dirichlet integral |
| 181 | Saturday | Computational Lab | Numerical verification, physics applications |
| 182 | Sunday | Week Review | Synthesis, problem sets, self-assessment |

---

## Learning Objectives

By the end of Week 26, you will be able to:

1. **Line Integrals:** Compute complex line integrals via parametrization
2. **Cauchy's Theorem:** Apply to show integrals vanish for analytic functions
3. **Cauchy's Formula:** Extract function values and derivatives from boundary data
4. **Real Integrals:** Evaluate improper integrals using semicircular contours
5. **Jordan's Lemma:** Handle Fourier-type integrals with exponential factors
6. **Branch Cuts:** Navigate multi-valued functions with keyhole contours
7. **Trigonometric Integrals:** Convert to contour integrals via $z = e^{i\theta}$
8. **Physics:** Compute Green's functions and understand dispersion relations

---

## Key Formulas

### Complex Line Integrals
| Formula | Description |
|---------|-------------|
| $\int_C f(z) \, dz = \int_a^b f(z(t)) z'(t) \, dt$ | Parametric definition |
| $\left\|\int_C f(z) \, dz\right\| \leq ML$ | ML inequality |
| $\oint (z-z_0)^n dz = 2\pi i \delta_{n,-1}$ | Fundamental integral |

### Cauchy's Theorems
| Formula | Description |
|---------|-------------|
| $\oint_C f(z) \, dz = 0$ | Cauchy's theorem |
| $f(z_0) = \frac{1}{2\pi i}\oint \frac{f(z)}{z-z_0} dz$ | Integral formula |
| $f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint \frac{f(z)}{(z-z_0)^{n+1}} dz$ | Derivatives |

### Real Integral Techniques
| Formula | Description |
|---------|-------------|
| $\int_{-\infty}^\infty \frac{P(x)}{Q(x)} dx = 2\pi i \sum_{\text{UHP}} \text{Res}$ | Rational functions |
| $\int_0^\infty \frac{x^{\alpha-1}}{1+x} dx = \frac{\pi}{\sin(\pi\alpha)}$ | Keyhole contour |
| $\int_0^\infty \frac{\sin x}{x} dx = \frac{\pi}{2}$ | Dirichlet integral |

---

## Quantum Mechanics Connections

| Complex Analysis Topic | Quantum Mechanics Application |
|------------------------|------------------------------|
| Contour deformation | Wick rotation (real ↔ imaginary time) |
| Cauchy's theorem | Topological phases (Berry, Aharonov-Bohm) |
| Cauchy's formula | Spectral decomposition |
| Semicircular contours | Propagator computation |
| Branch cuts | Continuous spectrum contributions |
| Kramers-Kronig | Causality and response functions |

---

## Textbook References

**Primary:**
- Brown & Churchill, *Complex Variables and Applications*, Chapters 4-7
- Ahlfors, *Complex Analysis*, Chapters 4-5

**Physics Applications:**
- Arfken & Weber, *Mathematical Methods for Physicists*, Chapter 6
- Byron & Fuller, *Mathematics of Classical and Quantum Physics*, Chapter 6

---

## Prerequisites

**Required from Week 25:**
- Analytic functions and Cauchy-Riemann equations
- Harmonic functions
- Conformal mappings

**Required from earlier months:**
- Multivariable calculus (line integrals, Green's theorem)
- Complex numbers and exponentials

---

## Completion Checklist

- [x] Day 176: Complex Line Integrals
- [x] Day 177: Cauchy's Integral Theorem
- [x] Day 178: Cauchy's Integral Formula
- [x] Day 179: Applications to Real Integrals I
- [x] Day 180: Applications to Real Integrals II
- [x] Day 181: Computational Lab
- [x] Day 182: Week Review

---

## Key Insights

1. **Cauchy's theorem is topological:** The integral depends only on which singularities are enclosed, not on the contour's shape.

2. **Analytic functions are rigid:** Boundary values completely determine interior values (Cauchy's formula).

3. **Real integrals become tractable:** Extending to ℂ and using residues transforms "impossible" integrals into algebraic problems.

4. **Branch cuts encode multi-valuedness:** Keyhole contours navigate around branch points systematically.

5. **Physics uses all these tools:** Propagators, Green's functions, and dispersion relations are contour integrals.

---

## Preview: Week 27

Next week completes contour integration with **Laurent Series and the Residue Theorem**:

- Laurent series for functions with singularities
- Classification: removable, poles, essential
- Residue computation techniques
- The residue theorem (unification of all contour methods)
- Advanced physics applications

---

*"The integral calculus was enriched by Cauchy with a wholly new method — contour integration."*
— Felix Klein

---

**Week 26 Complete!**
