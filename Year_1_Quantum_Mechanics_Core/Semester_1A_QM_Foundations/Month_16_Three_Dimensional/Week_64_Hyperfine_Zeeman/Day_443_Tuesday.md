# Day 443: Hyperfine Interaction

## Overview
**Day 443** | Year 1, Month 16, Week 64 | The 21 cm Line

Today we derive the hyperfine splitting and the famous 21 cm hydrogen line used in radio astronomy.

---

## Learning Objectives

1. Derive the hyperfine Hamiltonian
2. Calculate the 1s hyperfine splitting
3. Understand the 21 cm transition
4. Connect to astrophysics applications
5. Compare F = 0 and F = 1 states

---

## Core Content

### Hyperfine Hamiltonian

$$\hat{H}_{\text{HF}} = -\boldsymbol{\mu}_p \cdot \mathbf{B}_e(0)$$

For s-orbitals (contact interaction):
$$\boxed{\hat{H}_{\text{HF}} = \frac{2\mu_0}{3}g_e g_p \mu_B \mu_N |\psi(0)|^2 \hat{\mathbf{I}} \cdot \hat{\mathbf{J}}}$$

### I·J Eigenvalue

Using F = I + J:
$$\mathbf{I}\cdot\mathbf{J} = \frac{1}{2}(F^2 - I^2 - J^2)$$

Eigenvalue: (ℏ²/2)[F(F+1) - I(I+1) - J(J+1)]

### Ground State Splitting

For 1s (J = I = 1/2):
- F = 1 (triplet): ⟨I·J⟩ = +ℏ²/4
- F = 0 (singlet): ⟨I·J⟩ = -3ℏ²/4

$$\boxed{\Delta E_{\text{HF}} = \frac{4}{3}g_e g_p \alpha^2 \frac{m_e}{m_p} E_1 \approx 5.9 \times 10^{-6} \text{ eV}}$$

### The 21 cm Line

$$\nu = \frac{\Delta E}{h} = 1420.4 \text{ MHz}$$
$$\lambda = \frac{c}{\nu} = 21.1 \text{ cm}$$

### Astrophysical Importance

- Maps neutral hydrogen in galaxies
- Doppler shift → velocity measurements
- Foundation of radio astronomy
- Spin temperature of interstellar medium

---

## Quantum Computing Connection

Hyperfine levels provide:
- **Long-lived qubits** (ground state hyperfine)
- **Microwave control** frequencies
- **Atomic clock** transitions

---

## Practice Problems

1. Calculate the 21 cm line frequency from ΔE_HF.
2. What is the spontaneous emission lifetime for this transition?
3. Why is hyperfine structure largest for s-orbitals?

---

## Summary

| Quantity | Value |
|----------|-------|
| Hyperfine splitting (1s) | 5.9 μeV |
| Frequency | 1420 MHz |
| Wavelength | 21 cm |

---

**Next:** [Day_444_Wednesday.md](Day_444_Wednesday.md) — Weak-Field Zeeman Effect
