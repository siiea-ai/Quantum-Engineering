# Day 435: Relativistic Kinetic Energy Correction

## Overview
**Day 435** | Year 1, Month 16, Week 63 | The p⁴ Term

Today we derive the relativistic correction to kinetic energy, the first of three fine structure contributions.

---

## Learning Objectives

1. Understand the relativistic origin of kinetic corrections
2. Derive the p⁴ perturbation Hamiltonian
3. Calculate first-order energy correction
4. See why this lifts l-degeneracy

---

## Core Content

### Relativistic Kinetic Energy

Full relativistic:
$$E = \sqrt{p^2c^2 + m^2c^4} \approx mc^2 + \frac{p^2}{2m} - \frac{p^4}{8m^3c^2} + ...$$

The perturbation:
$$\boxed{\hat{H}'_{\text{rel}} = -\frac{\hat{p}^4}{8m_e^3 c^2}}$$

### First-Order Correction

$$E^{(1)}_{\text{rel}} = \langle nlm | \hat{H}'_{\text{rel}} | nlm \rangle$$

Using p² = 2m_e(E - V):
$$E^{(1)}_{\text{rel}} = -\frac{1}{2m_e c^2}\left[E_n^2 + 2E_n\langle V \rangle + \langle V^2 \rangle\right]$$

### Result

$$\boxed{E^{(1)}_{\text{rel}} = -\frac{E_n^2}{2m_e c^2}\left[\frac{4n}{l+1/2} - 3\right]}$$

In terms of α:
$$E^{(1)}_{\text{rel}} = E_n \frac{\alpha^2}{n^2}\left[\frac{n}{l+1/2} - \frac{3}{4}\right]$$

### Key Observation

This depends on l, not just n — **breaks the l-degeneracy**!

---

## Practice Problems

1. Calculate E^{(1)}_rel for the 2s state.
2. Compare the magnitude to the unperturbed energy.
3. Why doesn't this depend on m?

---

## Summary

| Quantity | Formula |
|----------|---------|
| Perturbation | H' = -p⁴/(8m³c²) |
| Order | O(α²) relative to E_n |
| l-dependence | Yes (lifts degeneracy) |

---

**Next:** [Day_436_Tuesday.md](Day_436_Tuesday.md) — Spin-Orbit Coupling
