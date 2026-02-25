# Day 410: Spin-Orbit Coupling

## Overview
**Day 410** | Year 1, Month 15, Week 59 | Atomic Fine Structure

Spin-orbit coupling combines orbital (L) and spin (S) angular momentum into total angular momentum J = L + S. This causes the fine structure splitting in atomic spectra.

---

## Core Content

### The Spin-Orbit Hamiltonian
$$\hat{H}_{SO} = \frac{1}{2m^2c^2}\frac{1}{r}\frac{dV}{dr}\hat{\mathbf{L}}\cdot\hat{\mathbf{S}}$$

For hydrogen (Coulomb potential):
$$\hat{H}_{SO} = \frac{e^2}{8\pi\epsilon_0 m^2 c^2}\frac{1}{r^3}\hat{\mathbf{L}}\cdot\hat{\mathbf{S}}$$

### Total Angular Momentum
$$\hat{\mathbf{J}} = \hat{\mathbf{L}} + \hat{\mathbf{S}}$$

Using Ĵ² = L̂² + Ŝ² + 2**L̂**·**Ŝ**:
$$\hat{\mathbf{L}}\cdot\hat{\mathbf{S}} = \frac{1}{2}(\hat{J}^2 - \hat{L}^2 - \hat{S}^2) = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$$

### Allowed j Values
For electron (s = 1/2) with orbital angular momentum l:
- j = l + 1/2 (spin parallel to L)
- j = l - 1/2 (spin antiparallel to L)

Exception: l = 0 gives only j = 1/2

### Energy Shift
$$\Delta E_{SO} = \frac{\hbar^2}{2}\langle\mathbf{L}\cdot\mathbf{S}\rangle \cdot \xi_{nl}$$

where ξ_{nl} depends on the radial wave function.

---

## Key Results

| State | l | j | Spectroscopic Notation |
|-------|---|---|----------------------|
| n=2 | l=0 | 1/2 | 2S₁/₂ |
| n=2 | l=1 | 1/2, 3/2 | 2P₁/₂, 2P₃/₂ |

**Fine structure:** The 2P level splits into 2P₁/₂ and 2P₃/₂.

---

## Quantum Computing Connection
The fine structure underlies atomic clock transitions and trapped-ion qubits.

---

## Practice Problems
1. Calculate j values for an electron with l = 2.
2. Find ⟨**L**·**S**⟩ for the 2P₃/₂ state.
3. Which state has lower energy: 2P₁/₂ or 2P₃/₂?

---

**Next:** [Day_411_Friday.md](Day_411_Friday.md) — Two Spin-1/2 Addition
