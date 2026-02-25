# Day 467: Tunneling Rates

## Overview
**Day 467** | Year 1, Month 17, Week 67 | The Gamow Factor

---

## Core Content

### WKB Tunneling Formula

Transmission probability through barrier from x₁ to x₂:
$$\boxed{T \approx \exp\left(-\frac{2}{\hbar}\int_{x_1}^{x_2}\sqrt{2m(V(x)-E)}\,dx\right)}$$

This is the **Gamow factor**.

### Physical Interpretation

- Exponential suppression in barrier
- Wider/taller barrier → smaller transmission
- Sensitive to barrier parameters

### Alpha Decay

Nuclear alpha particle tunneling through Coulomb barrier:
$$V(r) = \frac{2Ze^2}{r} \quad (r > R_{\text{nucleus}})$$

$$T = \exp\left(-\frac{2}{\hbar}\int_R^{r_0}\sqrt{2m_\alpha\left(\frac{2Ze^2}{r}-E\right)}\,dr\right)$$

Gamow's calculation explained nuclear lifetimes varying from 10⁻⁷ s to 10¹⁰ years!

### Decay Rate

$$\Gamma = \frac{\hbar}{\tau} = f \cdot T$$

where f is attempt frequency (how often particle hits barrier).

---

## Practice Problems

1. Calculate T for rectangular barrier of width a, height V₀.
2. How does T depend on particle mass?
3. Estimate alpha decay lifetime for ²³⁸U.

---

**Next:** [Day_468_Saturday.md](Day_468_Saturday.md) — Above-Barrier Reflection
