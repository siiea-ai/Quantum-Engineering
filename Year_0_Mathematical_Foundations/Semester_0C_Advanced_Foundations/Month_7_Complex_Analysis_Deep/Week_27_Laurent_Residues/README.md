# Week 27: Laurent Series and Residues

## Overview

**Days:** 183-189 (7 days)
**Status:** ✅ COMPLETE
**Focus:** The residue theorem — unifying all contour integration techniques

This week develops the residue theorem, the most powerful tool in complex analysis. Starting from Laurent series, we classify singularities, develop residue computation techniques, and prove the theorem that underlies all contour integration applications.

---

## Daily Schedule

| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 183 | Monday | Laurent Series | Expansion in annuli, principal part, coefficients |
| 184 | Tuesday | Singularity Classification | Removable, poles, essential; Casorati-Weierstrass |
| 185 | Wednesday | Residue Computation | Simple/higher poles, L'Hôpital, residue at infinity |
| 186 | Thursday | The Residue Theorem | Statement, proof, argument principle |
| 187 | Friday | Advanced Applications | Series summation, Mittag-Leffler, definite integrals |
| 188 | Saturday | Computational Lab | Numerical methods, visualizations, physics |
| 189 | Sunday | Week Review | Synthesis, problem sets, self-assessment |

---

## Learning Objectives

By the end of Week 27, you will be able to:

1. **Laurent Series:** Construct expansions in different annuli
2. **Classification:** Identify removable singularities, poles, and essential singularities
3. **Residue Computation:** Calculate residues using multiple techniques
4. **Residue Theorem:** Apply to evaluate contour integrals
5. **Argument Principle:** Count zeros and poles inside contours
6. **Applications:** Sum series and evaluate integrals systematically
7. **Physics:** Compute scattering amplitudes and Green's functions

---

## Key Formulas

### Laurent Series
| Formula | Description |
|---------|-------------|
| $f(z) = \sum_{n=-\infty}^{\infty} a_n(z-z_0)^n$ | Laurent expansion |
| $a_n = \frac{1}{2\pi i}\oint \frac{f(z)}{(z-z_0)^{n+1}} dz$ | Coefficient formula |
| $a_{-1} = \text{Res}_{z=z_0} f(z)$ | Residue is coefficient of $(z-z_0)^{-1}$ |

### Residue Computation
| Formula | Description |
|---------|-------------|
| $\text{Res} = \lim_{z \to z_0}(z-z_0)f(z)$ | Simple pole |
| $\text{Res} = \frac{1}{(m-1)!}\lim_{z \to z_0}\frac{d^{m-1}}{dz^{m-1}}[(z-z_0)^m f(z)]$ | Order $m$ pole |
| $\text{Res}_{z=z_0}\frac{P(z)}{Q(z)} = \frac{P(z_0)}{Q'(z_0)}$ | Simple pole of ratio |

### The Residue Theorem
| Formula | Description |
|---------|-------------|
| $\oint_C f(z) dz = 2\pi i \sum_k \text{Res}_{z=z_k} f(z)$ | Residue theorem |
| $\frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)} dz = N - P$ | Argument principle |

---

## Singularity Classification

| Type | Principal Part | Behavior | Example |
|------|---------------|----------|---------|
| Removable | None | Bounded | $\sin z / z$ at 0 |
| Pole (order $m$) | Finite | $\|f\| \to \infty$ | $1/z^m$ at 0 |
| Essential | Infinite | Chaotic | $e^{1/z}$ at 0 |

---

## Quantum Mechanics Connections

| Complex Analysis | Quantum Mechanics |
|-----------------|-------------------|
| Pole locations | Bound state energies |
| Pole residues | Wave function normalization |
| Essential singularities | Continuous spectrum |
| Argument principle | Levinson's theorem |
| Series summation | Casimir effect |

---

## Textbook References

**Primary:**
- Brown & Churchill, *Complex Variables*, Chapters 5-7
- Ahlfors, *Complex Analysis*, Chapters 4-5

**Applications:**
- Arfken & Weber, Chapter 7 (Residue Theory)
- Shankar, *Quantum Mechanics*, Chapter 19 (Scattering)

---

## Completion Checklist

- [x] Day 183: Laurent Series
- [x] Day 184: Singularity Classification
- [x] Day 185: Residue Computation Techniques
- [x] Day 186: The Residue Theorem
- [x] Day 187: Advanced Applications
- [x] Day 188: Computational Lab
- [x] Day 189: Week Review

---

## Key Insights

1. **Laurent series generalize Taylor series** to functions with singularities.

2. **The residue is the key quantity** — it's the coefficient $a_{-1}$ that determines contour integrals.

3. **The residue theorem unifies** all contour integration techniques from Week 26.

4. **In physics, poles encode discrete states** while branch cuts encode continua.

5. **The argument principle** provides a topological way to count zeros and poles.

---

## Preview: Week 28

Next week applies complex analysis to physics problems:
- Green's functions and propagators
- Dispersion relations and causality
- Scattering theory applications
- Asymptotic expansions
- Comprehensive Month 7 review

---

*"The theory of residues is perhaps the most delicate and at the same time the most useful tool in the whole field of analysis."*
— E.T. Whittaker

---

**Week 27 Complete!**
