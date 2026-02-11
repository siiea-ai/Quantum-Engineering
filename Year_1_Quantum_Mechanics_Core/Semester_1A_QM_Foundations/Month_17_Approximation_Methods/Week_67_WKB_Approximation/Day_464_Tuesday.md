# Day 464: WKB Wavefunctions

## Overview
**Day 464** | Year 1, Month 17, Week 67 | Solutions in Allowed and Forbidden Regions

---

## Core Content

### Classically Allowed Region (E > V)

$$\boxed{\psi(x) = \frac{A}{\sqrt{p(x)}}\exp\left(\frac{i}{\hbar}\int^x p(x')\,dx'\right) + \frac{B}{\sqrt{p(x)}}\exp\left(-\frac{i}{\hbar}\int^x p(x')\,dx'\right)}$$

where p(x) = √(2m(E-V(x))) is the local classical momentum.

### Classically Forbidden Region (E < V)

$$\psi(x) = \frac{C}{\sqrt{\kappa(x)}}\exp\left(-\frac{1}{\hbar}\int^x \kappa(x')\,dx'\right) + \frac{D}{\sqrt{\kappa(x)}}\exp\left(+\frac{1}{\hbar}\int^x \kappa(x')\,dx'\right)$$

where κ(x) = √(2m(V(x)-E)) is the decay constant.

### The 1/√p Factor

Ensures probability conservation: |ψ|²v = const (flux continuity)

Larger momentum → faster motion → less time spent → lower probability density

### Phase Accumulation

Total phase in allowed region:
$$\phi = \frac{1}{\hbar}\int_{x_1}^{x_2} p(x)\,dx$$

---

**Next:** [Day_465_Wednesday.md](Day_465_Wednesday.md) — Connection Formulas
