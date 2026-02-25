# Day 450: Non-Degenerate Perturbation Theory II

## Overview
**Day 450** | Year 1, Month 17, Week 65 | Second-Order Corrections

Today we derive second-order energy corrections and understand when first-order results vanish, making second-order the leading contribution.

---

## Learning Objectives

By the end of today, you will be able to:
1. Derive second-order energy corrections
2. Understand virtual state contributions
3. Apply to problems where E^(1) = 0
4. Calculate second-order wavefunction corrections
5. Assess convergence of the series
6. Recognize energy level repulsion

---

## Core Content

### Second-Order Energy

From the O(λ²) equation:
$$\boxed{E_n^{(2)} = \sum_{m \neq n} \frac{|\langle m^{(0)} | H' | n^{(0)} \rangle|^2}{E_n^{(0)} - E_m^{(0)}}}$$

### Properties of E^(2)

1. **Always real** (sum of squared magnitudes)
2. **Ground state:** E₀^(2) < 0 always (all denominators negative)
3. **Level repulsion:** States pushed apart by perturbation
4. **Sum rule:** Related to completeness

### Virtual Transitions

E^(2) represents "virtual transitions" to intermediate states:
- Amplitude to go n → m: ⟨m|H'|n⟩
- Energy cost: E_n - E_m
- Return amplitude: implicit
- Sum over all possible intermediate states

### When First Order Vanishes

If H' has odd parity and |n⟩ has definite parity:
$$E_n^{(1)} = \langle n | H' | n \rangle = 0$$

Then second order is the leading correction!

### Second-Order Wavefunction

$$|n^{(2)}\rangle = \sum_{m \neq n}\sum_{k \neq n} \frac{\langle m|H'|k\rangle\langle k|H'|n\rangle}{(E_n - E_m)(E_n - E_k)}|m\rangle - \frac{1}{2}\sum_{m \neq n}\frac{|\langle m|H'|n\rangle|^2}{(E_n - E_m)^2}|n\rangle$$

(The second term maintains normalization.)

### Convergence Criterion

The series converges when:
$$\left|\frac{\langle m|H'|n\rangle}{E_n - E_m}\right| \ll 1 \quad \text{for all } m$$

---

## Quantum Computing Connection

### Effective Hamiltonians

Second-order PT generates effective interactions:
- **Virtual photon exchange** → effective qubit-qubit coupling
- **Dispersive readout** in circuit QED
- **Lamb shift** in atomic qubits

---

## Worked Examples

### Example 1: Ground State of Perturbed Oscillator

**Problem:** Find E₀^(2) for H' = αx on harmonic oscillator.

**Solution:**
E₀^(1) = α⟨0|x|0⟩ = 0 (x has odd parity)

$$E_0^{(2)} = \sum_{m \neq 0} \frac{|\langle m|αx|0\rangle|^2}{E_0 - E_m}$$

Only m = 1 contributes: ⟨1|x|0⟩ = √(ℏ/2mω)

$$E_0^{(2)} = \frac{\alpha^2 \hbar/(2m\omega)}{(1/2)\hbar\omega - (3/2)\hbar\omega} = -\frac{\alpha^2}{2m\omega^2}$$

$$\boxed{E_0^{(2)} = -\frac{\alpha^2}{2m\omega^2}}$$

### Example 2: Quadratic Stark Effect

**Problem:** Hydrogen ground state in electric field E: H' = eEz.

**Solution:**
E_1s^(1) = ⟨1s|eEz|1s⟩ = 0 (z has odd parity, 1s has even)

$$E_{1s}^{(2)} = \sum_{nlm \neq 1s} \frac{|eE\langle nlm|z|1s\rangle|^2}{E_{1s} - E_{nlm}}$$

This gives:
$$\boxed{E_{1s}^{(2)} = -\frac{9}{4}a_0^3 E^2 = -\frac{1}{2}\alpha E^2}$$

where α = (9/2)a₀³ is the polarizability.

---

## Practice Problems

1. Calculate E₁^(2) for harmonic oscillator with H' = αx.
2. Show that the ground state energy always decreases to second order.
3. Find E^(2) for particle in box with H' = V₀cos(πx/L).

---

## Summary

| Quantity | Formula |
|----------|---------|
| E^(2) | Σ_{m≠n} \|⟨m\|H'\|n⟩\|²/(E_n - E_m) |
| Ground state | E₀^(2) < 0 always |
| Level repulsion | Nearby states push apart |

---

**Next:** [Day_451_Wednesday.md](Day_451_Wednesday.md) — Degenerate Perturbation Theory I
