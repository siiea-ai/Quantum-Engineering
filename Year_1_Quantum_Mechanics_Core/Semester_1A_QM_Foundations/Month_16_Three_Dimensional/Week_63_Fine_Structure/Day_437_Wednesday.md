# Day 437: Darwin Term

## Overview
**Day 437** | Year 1, Month 16, Week 63 | The Contact Interaction

Today we derive the Darwin term — a relativistic correction that affects only s-orbitals, arising from the "zitterbewegung" of the electron.

---

## Learning Objectives

1. Understand the physical origin of the Darwin term
2. Derive the Darwin Hamiltonian
3. See why it only affects l = 0 states
4. Calculate the energy correction
5. Connect to the Dirac equation

---

## Core Content

### Physical Origin

In the Dirac equation, the electron undergoes rapid oscillations ("zitterbewegung") on the scale of the Compton wavelength λ_C = ℏ/(m_e c).

This "smears" the electron over a small region, sampling the potential at slightly different positions.

### The Darwin Hamiltonian

$$\boxed{\hat{H}'_D = \frac{\pi\hbar^2 e^2}{2m_e^2 c^2}\delta^3(\mathbf{r})}$$

This is a **contact interaction** — nonzero only at r = 0.

### Why Only s-Orbitals?

The delta function picks out ψ(0). But:
- ψ_{nlm}(0) ∝ R_{nl}(0) × Y_l^m(0)
- R_{nl}(0) = 0 for l > 0 (centrifugal barrier)
- Only l = 0 states have ψ(0) ≠ 0

### First-Order Correction

$$E^{(1)}_D = \frac{\pi\hbar^2 e^2}{2m_e^2 c^2}|\psi_{n00}(0)|^2$$

Using |ψ_{n00}(0)|² = 1/(πn³a₀³):

$$\boxed{E^{(1)}_D = \frac{E_n^2}{2m_e c^2} \cdot \frac{4n}{l+1/2} \quad (l=0 \text{ only})}$$

### Combining with Spin-Orbit

For l > 0: only spin-orbit contributes
For l = 0: Darwin term replaces spin-orbit (which would diverge)

Together they give the same j-dependent formula!

---

## Practice Problems

1. Calculate |ψ₁₀₀(0)|² and E^{(1)}_D for 1s.
2. Why doesn't the Darwin term appear for p-orbitals?
3. Show that the Darwin term for l=0 matches the SO formula with j = 1/2.

---

## Summary

| Quantity | Formula |
|----------|---------|
| Darwin H' | (πℏ²e²/2m²c²)δ³(r) |
| Affects | Only l = 0 states |
| E^{(1)}_D | (E_n²/2m_ec²)(4n/(l+1/2)) for l=0 |

---

**Next:** [Day_438_Thursday.md](Day_438_Thursday.md) — Total Fine Structure
