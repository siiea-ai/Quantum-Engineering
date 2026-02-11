# Day 433: Expectation Values

## Overview
**Day 433** | Year 1, Month 16, Week 62 | Atomic Properties

Today we calculate key expectation values for hydrogen, including ⟨r⟩, ⟨1/r⟩, and verify the virial theorem.

---

## Learning Objectives

By the end of today, you will be able to:
1. Calculate ⟨r^k⟩ for hydrogen states
2. Apply the virial theorem
3. Derive relations between kinetic and potential energy
4. Understand the Hellmann-Feynman theorem
5. Calculate atomic polarizability
6. Apply to spectroscopic measurements

---

## Core Content

### Key Expectation Values

$$\boxed{\langle r \rangle_{nl} = \frac{a_0}{2}[3n^2 - l(l+1)]}$$

$$\boxed{\langle r^2 \rangle_{nl} = \frac{a_0^2 n^2}{2}[5n^2 + 1 - 3l(l+1)]}$$

$$\boxed{\langle \frac{1}{r} \rangle_{nl} = \frac{1}{n^2 a_0}}$$

$$\boxed{\langle \frac{1}{r^2} \rangle_{nl} = \frac{1}{n^3 a_0^2 (l+1/2)}}$$

### The Virial Theorem

For V ∝ r^n:
$$\langle T \rangle = \frac{n}{2}\langle V \rangle$$

For Coulomb (n = -1):
$$\boxed{\langle T \rangle = -\frac{1}{2}\langle V \rangle = -E}$$
$$\boxed{\langle V \rangle = 2E}$$

### Verification for Ground State

For 1s (n=1, l=0):
- E₁ = -13.6 eV
- ⟨T⟩ = +13.6 eV
- ⟨V⟩ = -27.2 eV
- ⟨V⟩ + ⟨T⟩ = E₁ ✓

### Uncertainty Relations

$$\Delta r \cdot \Delta p \geq \frac{\hbar}{2}$$

For 1s: Δr ≈ a₀, Δp ≈ ℏ/a₀

---

## Worked Examples

### Example 1: Mean Radius

**Problem:** Calculate ⟨r⟩ for the 2p state.

**Solution:**
$$\langle r \rangle_{21} = \frac{a_0}{2}[3(4) - 2] = 5a_0$$

### Example 2: Potential Energy

**Problem:** Find ⟨V⟩ for the 1s state.

**Solution:**
$$\langle V \rangle = -e^2\langle 1/r \rangle = -\frac{e^2}{a_0} = -27.2 \text{ eV}$$

---

## Summary

| Quantity | Formula |
|----------|---------|
| ⟨r⟩ | (a₀/2)[3n² - l(l+1)] |
| ⟨1/r⟩ | 1/(n²a₀) |
| ⟨T⟩ | -E (virial) |
| ⟨V⟩ | 2E (virial) |

---

**Next:** [Day_434_Sunday.md](Day_434_Sunday.md) — Week 62 Review
