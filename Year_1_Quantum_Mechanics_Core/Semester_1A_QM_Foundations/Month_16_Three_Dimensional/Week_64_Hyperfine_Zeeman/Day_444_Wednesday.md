# Day 444: Weak-Field Zeeman Effect

## Overview
**Day 444** | Year 1, Month 16, Week 64 | Atoms in Magnetic Fields

Today we study how atomic energy levels split in an external magnetic field — the Zeeman effect.

---

## Learning Objectives

1. Derive the magnetic interaction Hamiltonian
2. Calculate energy shifts in weak fields
3. Understand normal vs anomalous Zeeman effect
4. Apply the Landé g-factor
5. Connect to spectroscopy and qubit control

---

## Core Content

### Magnetic Interaction

$$\hat{H}_B = -\boldsymbol{\mu} \cdot \mathbf{B} = \frac{\mu_B}{\hbar}(\hat{\mathbf{L}} + g_e\hat{\mathbf{S}}) \cdot \mathbf{B}$$

With B = B_z ẑ:
$$\hat{H}_B = \frac{\mu_B B}{\hbar}(\hat{L}_z + g_e\hat{S}_z)$$

### Weak Field Limit

When μ_B B << fine structure splitting, J is still good:

$$\boxed{E_B = g_J \mu_B B m_J}$$

### Landé g-Factor

$$\boxed{g_J = 1 + \frac{J(J+1) + S(S+1) - L(L+1)}{2J(J+1)}}$$

Special cases:
- l = 0: g_J = 2 (pure spin)
- s = 0: g_J = 1 (pure orbital)

### Anomalous Zeeman Effect

For g_J ≠ 1, unequal spacings appear:
- Selection rule: Δm_J = 0, ±1
- Different transitions have different frequencies

### Normal Zeeman (Historical)

Only for singlet states (S = 0, g_J = 1):
- Three equally spaced lines
- Spacing = μ_B B

---

## Quantum Computing Connection

Zeeman effect enables:
- **Magnetic field tuning** of qubit frequencies
- **State-selective addressing**
- **Qubit isolation** via field gradients

---

## Practice Problems

1. Calculate g_J for the ²P₃/₂ state.
2. How many Zeeman sublevels does ²D₅/₂ have?
3. Sketch the Zeeman splitting of the 2P → 1S transition.

---

## Summary

| Quantity | Formula |
|----------|---------|
| Energy shift | E_B = g_J μ_B B m_J |
| Landé g-factor | 1 + [J(J+1)+S(S+1)-L(L+1)]/[2J(J+1)] |
| Bohr magneton | μ_B = 9.27×10⁻²⁴ J/T |

---

**Next:** [Day_445_Thursday.md](Day_445_Thursday.md) — Strong-Field Zeeman Effect
