# Day 457: Trial Wavefunctions

## Overview
**Day 457** | Year 1, Month 17, Week 66 | Designing Effective Trial States

Today we learn strategies for choosing trial wavefunctions that capture essential physics.

---

## Core Content

### Principles for Good Trial Functions

1. **Correct boundary conditions** (essential!)
2. **Correct symmetry** (even/odd, angular momentum)
3. **Correct asymptotic behavior** (e^{-κr} for bound states)
4. **Physical intuition** (cusp conditions, nodes)
5. **Computational tractability** (analytic integrals preferred)

### Common Trial Function Forms

**Exponential:** ψ = e^{-αr} (hydrogen-like)
**Gaussian:** ψ = e^{-αr²} (oscillator-like)
**Polynomial × exponential:** ψ = (1 + βr)e^{-αr}
**Linear combination:** ψ = Σ c_n φ_n (basis expansion)

### Multiple Parameters

More parameters → lower (better) energy bound
$$E(\alpha, \beta, ...) \geq E_0$$

Optimize all simultaneously: ∇E = 0

### The Cusp Condition

For Coulomb potential at nucleus:
$$\left.\frac{1}{\psi}\frac{\partial\psi}{\partial r}\right|_{r=0} = -\frac{Z}{a_0}$$

---

## Worked Example

**Problem:** Hydrogen ground state with ψ = e^{-αr}.

**Solution:**
$$E(\alpha) = \frac{\hbar^2\alpha^2}{2m} - \frac{e^2\alpha}{4\pi\varepsilon_0}$$

∂E/∂α = 0 → α = 1/a₀

$$E_{min} = -\frac{e^2}{8\pi\varepsilon_0 a_0} = -13.6 \text{ eV (exact!)}$$

---

## Practice Problems

1. Try ψ = r^n e^{-αr} for hydrogen and find optimal n.
2. For particle in box, compare ψ = x(L-x) with exact ground state.
3. Why do Gaussians give worse bounds than exponentials for Coulomb?

---

**Next:** [Day_458_Wednesday.md](Day_458_Wednesday.md) — Helium Atom
