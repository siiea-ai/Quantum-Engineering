# Day 458: Helium Atom Ground State

## Overview
**Day 458** | Year 1, Month 17, Week 66 | Two-Electron Atoms

Today we apply the variational method to helium, the simplest multi-electron atom.

---

## Core Content

### Helium Hamiltonian

$$H = -\frac{\hbar^2}{2m}(\nabla_1^2 + \nabla_2^2) - \frac{2e^2}{r_1} - \frac{2e^2}{r_2} + \frac{e^2}{r_{12}}$$

The electron-electron repulsion e²/r₁₂ makes this unsolvable exactly!

### Trial Function: Independent Electrons

$$\psi(r_1, r_2) = e^{-Z_{\text{eff}}(r_1 + r_2)/a_0}$$

with Z_eff as variational parameter.

### Physical Interpretation

Z_eff < 2: electrons "screen" each other
Each electron sees effective nuclear charge Z_eff

### Variational Calculation

$$E(Z_{\text{eff}}) = \left(Z_{\text{eff}}^2 - \frac{27}{8}Z_{\text{eff}}\right) E_R$$

Minimize: ∂E/∂Z_eff = 0 → Z_eff = 27/16 = 1.69

$$E_{min} = -\left(\frac{27}{16}\right)^2 E_R = -2.85 E_R = -77.5 \text{ eV}$$

Experimental: -79.0 eV (98% accuracy!)

### Better Trial Functions

Adding r₁₂ dependence improves result:
$$\psi = (1 + c \cdot r_{12})e^{-Z_{\text{eff}}(r_1+r_2)/a_0}$$

Can achieve 99.99% accuracy with enough parameters.

---

## Practice Problems

1. Calculate the electron-electron repulsion energy for the simple trial function.
2. What is the ionization energy prediction?
3. Why is Z_eff < 2?

---

**Next:** [Day_459_Thursday.md](Day_459_Thursday.md) — Hydrogen Molecule Ion
