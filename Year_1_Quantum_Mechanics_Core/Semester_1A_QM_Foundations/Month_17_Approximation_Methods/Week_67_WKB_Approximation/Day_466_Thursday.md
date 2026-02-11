# Day 466: Bound State Quantization

## Overview
**Day 466** | Year 1, Month 17, Week 67 | Bohr-Sommerfeld Condition

---

## Core Content

### Derivation

For bound state between turning points x₁ and x₂:
- Wave must be normalizable (decay at both ends)
- Connection formulas at both turning points
- Single-valuedness requires constructive interference

### Bohr-Sommerfeld Quantization

$$\boxed{\oint p(x)\,dx = \left(n + \frac{1}{2}\right)h}$$

or equivalently:
$$\int_{x_1}^{x_2} p(x)\,dx = \left(n + \frac{1}{2}\right)\frac{\pi\hbar}{2} \times 2 = \left(n + \frac{1}{2}\right)\pi\hbar$$

### The Half-Integer

The 1/2 comes from π/4 at each turning point (total π/2)!

This is a quantum correction to the old Bohr-Sommerfeld rule.

### Example: Harmonic Oscillator

$$\oint p\,dx = 2\int_{-x_0}^{x_0}\sqrt{2m(E - \frac{1}{2}m\omega^2x^2)}\,dx = \frac{2\pi E}{\omega}$$

Quantization: E = (n + 1/2)ℏω — **Exact!**

### Example: Particle in Box [0, L]

p = √(2mE) = const

$$\int_0^L p\,dx = pL = \left(n + \frac{1}{2}\right)\pi\hbar$$

$$E_n = \frac{\pi^2\hbar^2(n+1/2)^2}{2mL^2}$$

Compare exact: E_n = π²ℏ²n²/(2mL²) — Good for large n!

---

**Next:** [Day_467_Friday.md](Day_467_Friday.md) — Tunneling Rates
