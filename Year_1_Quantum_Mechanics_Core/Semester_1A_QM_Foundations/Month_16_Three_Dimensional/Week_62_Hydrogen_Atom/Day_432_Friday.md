# Day 432: Degeneracy and Symmetry

## Overview
**Day 432** | Year 1, Month 16, Week 62 | The Hidden SO(4) Symmetry

Today we explore why hydrogen has "accidental" degeneracy — energy depending only on n, not l — revealing a hidden symmetry beyond rotation.

---

## Learning Objectives

By the end of today, you will be able to:
1. Calculate the degeneracy g_n = n² (or 2n² with spin)
2. Explain why different l values are degenerate
3. Understand the Runge-Lenz vector
4. Recognize the hidden SO(4) symmetry
5. Compare with other central potentials
6. Connect to supersymmetry concepts

---

## Core Content

### Degeneracy Count

For principal quantum number n:
- l = 0, 1, 2, ..., n-1
- For each l: m = -l, ..., +l (2l+1 states)

Total:
$$g_n = \sum_{l=0}^{n-1}(2l+1) = n^2$$

With electron spin:
$$g_n^{\text{spin}} = 2n^2$$

### Comparison of Degeneracies

| System | Degeneracy | Symmetry |
|--------|------------|----------|
| 3D harmonic oscillator | (N+1)(N+2)/2 | SU(3) |
| Hydrogen | n² | SO(4) |
| General V(r) | 2l+1 | SO(3) only |

### The Runge-Lenz Vector

Classically, for the Kepler problem:
$$\mathbf{A} = \mathbf{p} \times \mathbf{L} - mke^2\hat{\mathbf{r}}$$

This vector is conserved and points along the ellipse's major axis!

### Quantum Runge-Lenz

$$\hat{\mathbf{A}} = \frac{1}{2}(\hat{\mathbf{p}} \times \hat{\mathbf{L}} - \hat{\mathbf{L}} \times \hat{\mathbf{p}}) - \frac{me^2\hat{\mathbf{r}}}{r}$$

Properties:
- [Â, Ĥ] = 0 (commutes with Hamiltonian)
- [L̂ᵢ, Âⱼ] = iℏε_{ijk}Â_k (vector under rotations)

### SO(4) Symmetry

Define:
$$\hat{\mathbf{J}}_\pm = \frac{1}{2}(\hat{\mathbf{L}} \pm \hat{\mathbf{A}}/\sqrt{-2mE})$$

These satisfy:
$$[J_+^i, J_+^j] = i\hbar\varepsilon^{ijk}J_+^k$$
$$[J_-^i, J_-^j] = i\hbar\varepsilon^{ijk}J_-^k$$
$$[J_+^i, J_-^j] = 0$$

This is SO(4) ≅ SU(2) × SU(2)!

### Consequence

Both J₊² and J₋² have the same eigenvalue j(j+1) with j = (n-1)/2.

Dimension of representation: (2j+1)² = n²

---

## Quantum Computing Connection

### Symmetry-Adapted Circuits

Understanding symmetries enables:
- Efficient VQE ansätze
- Reduced Hilbert space dimension
- Symmetry-preserving error correction

---

## Practice Problems

1. Calculate g₄ by explicit counting.
2. Why does breaking the 1/r potential (e.g., screening) lift the l-degeneracy?
3. Show that [L², A] = 0.

---

## Summary

| Degeneracy | Formula | Origin |
|------------|---------|--------|
| Rotational | 2l+1 | SO(3) |
| Hydrogen | n² | SO(4) via Runge-Lenz |
| With spin | 2n² | SO(4) × SU(2) |

---

**Next:** [Day_433_Saturday.md](Day_433_Saturday.md) — Expectation Values
