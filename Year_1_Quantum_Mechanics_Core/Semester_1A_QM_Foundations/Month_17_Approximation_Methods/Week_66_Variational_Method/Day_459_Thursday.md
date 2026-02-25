# Day 459: Hydrogen Molecule Ion H₂⁺

## Overview
**Day 459** | Year 1, Month 17, Week 66 | The Simplest Molecule

Today we apply variational methods to H₂⁺, demonstrating chemical bonding.

---

## Core Content

### The System

- Two protons at distance R
- One electron
- H = kinetic + V(proton A) + V(proton B) + proton-proton repulsion

### LCAO Trial Function

Linear Combination of Atomic Orbitals:
$$\psi_\pm = \frac{1}{\sqrt{2(1\pm S)}}(\phi_A \pm \phi_B)$$

where φ_A, φ_B are 1s orbitals centered on each proton, and S = ⟨φ_A|φ_B⟩ is the overlap.

### Bonding and Antibonding

- ψ₊ (symmetric): **Bonding orbital** — electron between nuclei
- ψ₋ (antisymmetric): **Antibonding orbital** — node between nuclei

### Potential Energy Curves

E₊(R): Has minimum at R_eq ≈ 1.06 Å → **stable molecule**
E₋(R): Purely repulsive → **unstable**

### Results

| Property | Theory | Experiment |
|----------|--------|------------|
| R_eq | 1.32 Å | 1.06 Å |
| D_e | 1.77 eV | 2.79 eV |

Simple LCAO is only qualitative; better with optimized exponents.

---

## Practice Problems

1. Calculate the overlap integral S for 1s orbitals.
2. Why does the bonding orbital have lower energy?
3. Optimize the orbital exponent variationally.

---

**Next:** [Day_460_Friday.md](Day_460_Friday.md) — Linear Variational Method
