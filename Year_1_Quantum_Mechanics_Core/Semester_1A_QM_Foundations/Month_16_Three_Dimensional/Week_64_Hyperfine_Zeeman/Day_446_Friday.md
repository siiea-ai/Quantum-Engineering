# Day 446: Stark Effect

## Overview
**Day 446** | Year 1, Month 16, Week 64 | Atoms in Electric Fields

Today we study the Stark effect — how atoms respond to external electric fields.

---

## Learning Objectives

1. Derive the electric field perturbation
2. Understand why linear Stark requires degeneracy
3. Calculate quadratic Stark shift
4. Apply to hydrogen and multi-electron atoms
5. Connect to polarizability

---

## Core Content

### Electric Field Perturbation

$$\hat{H}' = -\mathbf{d} \cdot \mathbf{E} = eEz = eEr\cos\theta$$

This is an **odd parity** operator.

### No Linear Stark for Non-Degenerate States

For states with definite parity:
$$\langle n,l,m|z|n,l,m\rangle = 0$$

(z has odd parity, |ψ|² has even parity)

### Hydrogen: Linear Stark for n > 1

States with same n but different l are degenerate. The 2s-2p mixing gives:
$$\Delta E_{\text{linear}} = \pm 3eEa_0$$

### Quadratic Stark Effect (Dominant for Ground State)

Second-order perturbation:
$$\boxed{\Delta E^{(2)} = -\frac{1}{2}\alpha E^2}$$

where α is the **polarizability**.

### Hydrogen Ground State Polarizability

$$\alpha_{1s} = \frac{9}{2}a_0^3 (4\pi\varepsilon_0)$$

---

## Quantum Computing Connection

Stark effect enables:
- **Electric field qubit control**
- **DC Stark tuning** of transition frequencies
- **Rydberg atom interactions** (large polarizability)

---

## Practice Problems

1. Why is the linear Stark effect zero for hydrogen 1s?
2. Calculate the energy shift for H 1s in E = 10⁶ V/m.
3. Which hydrogen states mix under an electric field?

---

## Summary

| Effect | Formula | Applies to |
|--------|---------|------------|
| Linear Stark | ΔE ∝ E | Degenerate states |
| Quadratic Stark | ΔE = -αE²/2 | All states |

---

**Next:** [Day_447_Saturday.md](Day_447_Saturday.md) — Atomic Qubits
