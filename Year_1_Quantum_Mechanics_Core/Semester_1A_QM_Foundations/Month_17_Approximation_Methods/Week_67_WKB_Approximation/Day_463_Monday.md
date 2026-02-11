# Day 463: The Semiclassical Limit

## Overview
**Day 463** | Year 1, Month 17, Week 67 | When ℏ Is Small

Today we develop the WKB approximation starting from the semiclassical limit where quantum corrections are small.

---

## Core Content

### The Idea

When the potential varies slowly compared to the de Broglie wavelength:
$$\lambda(x) = \frac{h}{p(x)} = \frac{h}{\sqrt{2m(E-V(x))}}$$

The wavefunction looks locally like a plane wave with slowly varying wavelength.

### Formal Expansion

Write ψ = exp(iS/ℏ) and expand S in powers of ℏ:
$$S = S_0 + \hbar S_1 + \hbar^2 S_2 + ...$$

### Leading Order (Classical)

$$\frac{1}{2m}\left(\frac{\partial S_0}{\partial x}\right)^2 + V(x) = E$$

This is the **Hamilton-Jacobi equation**! S₀' = ±p(x) where p = √(2m(E-V)).

### Validity Condition

WKB valid when:
$$\left|\frac{d\lambda}{dx}\right| \ll 1$$

Breaks down at **turning points** where V(x) = E.

---

## Practice Problems

1. Find λ(x) for a linear potential V = Fx.
2. Where does WKB break down for the harmonic oscillator?
3. Show that λ'/λ = p'/p.

---

**Next:** [Day_464_Tuesday.md](Day_464_Tuesday.md) — WKB Wavefunctions
