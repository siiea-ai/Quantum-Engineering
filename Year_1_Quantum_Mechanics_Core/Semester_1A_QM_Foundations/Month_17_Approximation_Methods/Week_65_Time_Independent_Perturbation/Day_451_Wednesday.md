# Day 451: Degenerate Perturbation Theory I

## Overview
**Day 451** | Year 1, Month 17, Week 65 | When States Have Equal Energy

Today we handle the case of degenerate unperturbed states, where the standard perturbation formulas fail due to vanishing energy denominators.

---

## Learning Objectives

By the end of today, you will be able to:
1. Identify when degeneracy is a problem
2. Set up the secular equation
3. Diagonalize H' in the degenerate subspace
4. Find the lifted energies to first order
5. Determine the "good" zeroth-order states
6. Apply to 2-fold degeneracy

---

## Core Content

### The Problem

If E_n^(0) = E_m^(0) for some m ≠ n:
$$|n^{(1)}\rangle = \sum_{m \neq n} \frac{\langle m|H'|n\rangle}{E_n^{(0)} - E_m^{(0)}}|m\rangle \rightarrow \text{DIVERGES!}$$

### The Solution

Within the degenerate subspace, we must diagonalize H' to find the "good" basis.

### Setup for g-fold Degeneracy

States |n_1^(0)⟩, |n_2^(0)⟩, ..., |n_g^(0)⟩ all have energy E^(0).

**Any linear combination is also an eigenstate of H₀:**
$$|n^{(0)}\rangle = \sum_{i=1}^{g} c_i |n_i^{(0)}\rangle$$

### The Secular Equation

The correct zeroth-order state satisfies:
$$\det\left(H'_{ij} - E^{(1)}\delta_{ij}\right) = 0$$

where H'_{ij} = ⟨n_i^(0)|H'|n_j^(0)⟩.

### First-Order Energies

The g solutions E^(1)_α (α = 1, ..., g) are the eigenvalues of the g×g matrix H' restricted to the degenerate subspace.

$$\boxed{E_\alpha^{(1)} = \text{eigenvalue of } H'|_{\text{degenerate subspace}}}$$

### Good Zeroth-Order States

The "good" basis states |α^(0)⟩ are eigenvectors of H' within the degenerate subspace:
$$H'|_{\text{deg}}|\alpha^{(0)}\rangle = E_\alpha^{(1)}|\alpha^{(0)}\rangle$$

### Two-Fold Degeneracy

For 2 degenerate states |1⟩, |2⟩:
$$H' = \begin{pmatrix} W_{11} & W_{12} \\ W_{21} & W_{22} \end{pmatrix}$$

Eigenvalues:
$$E^{(1)}_\pm = \frac{W_{11} + W_{22}}{2} \pm \sqrt{\left(\frac{W_{11}-W_{22}}{2}\right)^2 + |W_{12}|^2}$$

### Degeneracy Lifting

If W_{12} ≠ 0: degeneracy is lifted (avoided crossing)
If W_{12} = 0: may remain degenerate

---

## Quantum Computing Connection

### Avoided Level Crossings

In quantum systems:
- **Adiabatic quantum computing:** Follow ground state through avoided crossings
- **Qubit spectroscopy:** Identify resonances
- **Landau-Zener transitions:** Diabatic vs adiabatic evolution

---

## Worked Example

**Problem:** Two-level system with degenerate energies E₁ = E₂ = E, perturbed by H' = V(|1⟩⟨2| + |2⟩⟨1|).

**Solution:**
$$H' = V\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

Eigenvalues: det(H' - E^(1)I) = (E^(1))² - V² = 0

$$E^{(1)}_\pm = \pm V$$

Total energies: E + V and E - V (degeneracy lifted by 2V)

Good states: |±⟩ = (|1⟩ ± |2⟩)/√2

---

## Practice Problems

1. Find E^(1) for 3-fold degeneracy with H' diagonal.
2. Show the "good" states for spin-orbit coupling in hydrogen.
3. When does 2-fold degeneracy remain unlifted?

---

## Summary

| Step | Action |
|------|--------|
| 1 | Identify degenerate subspace |
| 2 | Compute H' matrix elements within subspace |
| 3 | Diagonalize to find E^(1) |
| 4 | Eigenvectors give "good" basis |

---

**Next:** [Day_452_Thursday.md](Day_452_Thursday.md) — Degenerate Perturbation Theory II
