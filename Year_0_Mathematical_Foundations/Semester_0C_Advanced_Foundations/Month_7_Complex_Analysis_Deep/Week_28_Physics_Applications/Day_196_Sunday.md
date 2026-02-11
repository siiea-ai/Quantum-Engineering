# Day 196: Month 7 Review — Complex Analysis Deep

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 11:00 AM | 2 hours | Concept Synthesis |
| Late Morning | 11:00 AM - 12:30 PM | 1.5 hours | Problem Set A |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Problem Set B |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Self-Assessment & Preview |

**Total Study Time: 6.5 hours**

---

## Month 7 Concept Map

```
                        COMPLEX ANALYSIS DEEP
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   FOUNDATIONS            TECHNIQUES             APPLICATIONS
   (Week 25)              (Weeks 26-27)          (Week 28)
        │                      │                      │
   ┌────┴────┐           ┌─────┴─────┐          ┌────┴────┐
   │         │           │           │          │         │
Complex   Cauchy-     Contour    Laurent    Green's   Scattering
Functions Riemann    Integration  Series   Functions    Theory
   │         │           │           │          │         │
Branch   Harmonic    Cauchy's    Residue   Dispersion  S-matrix
 Cuts    Functions   Theorems    Theorem   Relations    Poles
   │         │           │           │          │         │
Multi-   Conformal  Real       Singularity  Kramers-  Levinson
valued   Mappings   Integrals  Classification Kronig   Theorem
                        │           │
                     Jordan's   Special
                      Lemma    Functions
                               (Γ, ζ)
```

---

## Key Formulas Summary

### Week 25: Foundations

| Formula | Description |
|---------|-------------|
| $e^{i\theta} = \cos\theta + i\sin\theta$ | Euler's formula |
| $u_x = v_y$, $u_y = -v_x$ | Cauchy-Riemann |
| $\nabla^2 u = 0$ | Harmonic functions |
| $T(z) = \frac{az+b}{cz+d}$ | Möbius transformation |

### Week 26: Contour Integration

| Formula | Description |
|---------|-------------|
| $\oint_C f(z)dz = 0$ | Cauchy's theorem |
| $f(z_0) = \frac{1}{2\pi i}\oint \frac{f(z)}{z-z_0}dz$ | Cauchy's formula |
| $\|f^{(n)}(z_0)\| \leq \frac{n!M}{R^n}$ | Cauchy's inequality |
| $\int_{-\infty}^\infty \frac{P}{Q}dx = 2\pi i\sum_{\text{UHP}}\text{Res}$ | Real integrals |

### Week 27: Residues

| Formula | Description |
|---------|-------------|
| $f(z) = \sum_{n=-\infty}^{\infty}a_n(z-z_0)^n$ | Laurent series |
| $\text{Res}_{z=z_0} = a_{-1}$ | Residue definition |
| $\oint_C f dz = 2\pi i\sum_k \text{Res}_{z_k}$ | Residue theorem |
| Casorati-Weierstrass | Essential singularities |

### Week 28: Physics Applications

| Formula | Description |
|---------|-------------|
| $G(E) = (E - H + i\varepsilon)^{-1}$ | Green's function |
| $\chi'(\omega) = \frac{1}{\pi}\mathcal{P}\int\frac{\chi''(\omega')}{\omega'-\omega}d\omega'$ | Kramers-Kronig |
| $\delta(0) - \delta(\infty) = n_B\pi$ | Levinson's theorem |
| $\Gamma(z)\Gamma(1-z) = \pi/\sin(\pi z)$ | Reflection formula |

---

## Problem Set A: Core Techniques

### A1: Cauchy-Riemann
Show that $f(z) = z^2\bar{z}$ is nowhere analytic.

### A2: Contour Integration
Evaluate $\oint_{|z|=2} \frac{z^2 + 1}{z(z-1)^2} dz$.

### A3: Laurent Series
Find the Laurent series of $\frac{1}{z(z-2)}$ valid for $0 < |z| < 2$.

### A4: Residue Computation
Find the residue of $\frac{e^z}{(z-\pi i)^3}$ at $z = \pi i$.

### A5: Real Integrals
Evaluate $\int_{-\infty}^{\infty} \frac{x^2}{(x^2+1)(x^2+4)} dx$.

---

## Problem Set B: Applications

### B1: Green's Function
For the 1D free particle, show that $G(x,x';E+i\varepsilon)$ gives outgoing waves.

### B2: Kramers-Kronig
Given $\chi''(\omega) = \frac{\gamma\omega}{(\omega_0^2-\omega^2)^2 + \gamma^2\omega^2}$, find $\chi'(0)$.

### B3: Scattering
A potential has S-matrix $S(k) = \frac{k-i\kappa}{k+i\kappa}$. Find the bound state energy.

### B4: Special Functions
Evaluate $\int_0^\infty x^{1/2}e^{-x}dx$ using Gamma functions.

### B5: Saddle Point
Use Stirling to approximate $\binom{100}{50}$.

### B6: Comprehensive
The dielectric function of a metal is $\varepsilon(\omega) = 1 - \frac{\omega_p^2}{\omega^2 + i\gamma\omega}$.
(a) Find poles and verify they're in the lower half-plane.
(b) Apply Kramers-Kronig to relate $\varepsilon_1$ and $\varepsilon_2$.
(c) Find the sum rule $\int_0^\infty \omega\varepsilon_2(\omega)d\omega$.

---

## Solutions to Selected Problems

### Solution A2

Singularities: $z = 0$ (simple pole), $z = 1$ (double pole).

**Residue at $z = 0$:**
$$\text{Res}_{z=0} = \lim_{z\to 0} z \cdot \frac{z^2+1}{z(z-1)^2} = \frac{1}{1} = 1$$

**Residue at $z = 1$:**
$$\text{Res}_{z=1} = \lim_{z\to 1}\frac{d}{dz}\left[(z-1)^2 \cdot \frac{z^2+1}{z(z-1)^2}\right] = \lim_{z\to 1}\frac{d}{dz}\frac{z^2+1}{z}$$
$$= \lim_{z\to 1}\frac{2z \cdot z - (z^2+1)}{z^2} = \frac{2-2}{1} = 0$$

Wait, let me recalculate:
$$\frac{d}{dz}\left(\frac{z^2+1}{z}\right) = \frac{2z^2 - z^2 - 1}{z^2} = \frac{z^2-1}{z^2}$$

At $z = 1$: $\frac{1-1}{1} = 0$

$$\oint = 2\pi i(1 + 0) = 2\pi i$$

### Solution A5

Partial fractions:
$$\frac{x^2}{(x^2+1)(x^2+4)} = \frac{A}{x^2+1} + \frac{B}{x^2+4}$$

$x^2 = A(x^2+4) + B(x^2+1)$

Setting $x^2 = -1$: $-1 = 3A \Rightarrow A = -1/3$
Setting $x^2 = -4$: $-4 = -3B \Rightarrow B = 4/3$

$$\int_{-\infty}^\infty = -\frac{1}{3}\pi + \frac{4}{3}\cdot\frac{\pi}{2} = -\frac{\pi}{3} + \frac{2\pi}{3} = \frac{\pi}{3}$$

### Solution B4

$$\int_0^\infty x^{1/2}e^{-x}dx = \Gamma(3/2) = \frac{1}{2}\Gamma(1/2) = \frac{\sqrt{\pi}}{2}$$

---

## Self-Assessment Checklist

### Foundations (Week 25)
- [ ] I can verify Cauchy-Riemann equations
- [ ] I understand multi-valued functions and branch cuts
- [ ] I can apply the maximum principle
- [ ] I can construct conformal mappings

### Contour Integration (Week 26)
- [ ] I can parametrize contours and compute integrals
- [ ] I understand Cauchy's theorem and formula
- [ ] I can evaluate real integrals via residues
- [ ] I can apply Jordan's lemma

### Residues (Week 27)
- [ ] I can construct Laurent series
- [ ] I can classify singularities
- [ ] I can compute residues by various methods
- [ ] I can apply the residue theorem

### Applications (Week 28)
- [ ] I understand Green's functions and their poles
- [ ] I can derive Kramers-Kronig relations
- [ ] I can analyze S-matrix poles for bound states
- [ ] I can apply saddle point methods

---

## Month 7 Key Insights

### 1. Analyticity is Powerful
Complex differentiability implies:
- Infinite differentiability
- Harmonic components
- Cauchy's theorems
- Path-independent integrals

### 2. Residues Simplify Everything
The residue theorem reduces contour integrals to algebra:
$$\oint f dz = 2\pi i \sum \text{(residues)}$$

### 3. Physics Lives in Complex Plane
- Bound states = poles on imaginary axis
- Resonances = poles in lower half-plane
- Causality = analyticity in upper half-plane

### 4. Dispersion Relations are Exact
Kramers-Kronig connects absorption and dispersion through analyticity — no approximations!

---

## Preview: Month 8 — Electromagnetism

Next month covers classical electromagnetism, the foundation for quantum electrodynamics:

| Week | Topic | QM Connection |
|------|-------|---------------|
| 29 | Electrostatics | Coulomb potential, hydrogen atom |
| 30 | Magnetostatics | Magnetic moment, spin |
| 31 | Maxwell's Equations | Photons, quantization |
| 32 | Special Relativity | Covariant QM, Dirac equation |

**Required reading:** Griffiths, *Introduction to Electrodynamics*

---

## Month 7 Complete! 🎉

You have mastered deep complex analysis:

✅ **Week 25:** Analytic functions, Cauchy-Riemann, conformal mappings
✅ **Week 26:** Contour integration, Cauchy's theorems, real integrals
✅ **Week 27:** Laurent series, residue theorem, singularity classification
✅ **Week 28:** Green's functions, dispersion relations, scattering theory

**These tools are essential for quantum mechanics and quantum field theory!**

---

*"Complex analysis is perhaps the most elegant branch of mathematics."*
— Walter Rudin

---

**Next:** Month 8 — Electromagnetism (Days 197-224)
