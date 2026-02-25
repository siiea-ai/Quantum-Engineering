# Day 461: Ritz Method and Systematic Improvement

## Overview
**Day 461** | Year 1, Month 17, Week 66 | Converging to the Exact Answer

Today we explore how systematic basis expansion converges to exact results.

---

## Core Content

### The Ritz Method

Increase basis size systematically: N → N+1 → N+2 → ...

**Key property:** Adding more basis functions can only lower (improve) the energy.

### Completeness

If {φ_n} is complete: lim_{N→∞} E_N = E_exact

### Practical Considerations

- **Diminishing returns:** First few functions matter most
- **Numerical stability:** Basis must be linearly independent
- **Computational cost:** Matrix diagonalization scales as N³

### Hylleraas Variational Calculation

For helium, using functions with explicit r₁₂ dependence:
$$\psi = \sum_{lmn} c_{lmn} r_1^l r_2^m r_{12}^n e^{-Z(r_1+r_2)}$$

Achieved 99.9999% accuracy!

### Connection to Hartree-Fock

Modern quantum chemistry uses:
- Slater determinants for antisymmetry
- Optimized Gaussian basis sets
- Configuration interaction for correlation

---

## Practice Problems

1. Show E(N+1) ≤ E(N) for Ritz method.
2. What happens if basis functions are nearly linearly dependent?
3. Estimate the basis size needed for 1% accuracy in helium.

---

**Next:** [Day_462_Sunday.md](Day_462_Sunday.md) — Week 66 Review
