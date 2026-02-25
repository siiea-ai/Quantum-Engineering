# Day 454: Stark Effect

## Overview
**Day 454** | Year 1, Month 17, Week 65 | Atoms in Electric Fields

Today we apply perturbation theory to atoms in electric fields, demonstrating both linear (degenerate) and quadratic (non-degenerate) Stark effects.

---

## Learning Objectives

By the end of today, you will be able to:
1. Set up the Stark effect perturbation
2. Apply quadratic Stark to ground state (non-degenerate)
3. Apply linear Stark to excited states (degenerate)
4. Calculate atomic polarizability
5. Understand the selection rules
6. Compare linear vs quadratic regimes

---

## Core Content

### The Perturbation

$$H' = -\mathbf{d}\cdot\mathbf{E} = eEz = eEr\cos\theta$$

### Ground State: Quadratic Stark

|1s⟩ is non-degenerate, and E^(1) = 0 (parity):
$$E_{1s}^{(2)} = -\frac{1}{2}\alpha E^2$$

where α = (9/2)a₀³(4πε₀) is the polarizability.

### Excited States: Linear Stark (n = 2)

The 4 states 2s, 2p₀, 2p₊, 2p₋ are degenerate.

Selection rules: ⟨l,m|z|l',m'⟩ ≠ 0 only if:
- Δl = ±1
- Δm = 0

**Only 2s ↔ 2p₀ couple!**

$$H' = \begin{pmatrix} 0 & W \\ W^* & 0 \end{pmatrix}$$

where W = ⟨2s|eEz|2p₀⟩ = -3eEa₀.

### Linear Stark Result

$$E^{(1)}_\pm = \pm 3eEa_0$$

The 2s-2p degeneracy is lifted linearly in E!

### Good States

$$|\pm\rangle = \frac{1}{\sqrt{2}}(|2s\rangle \pm |2p_0\rangle)$$

These have permanent electric dipole moments ∓3ea₀.

---

## Practice Problems

1. Calculate the polarizability of hydrogen 2s state.
2. Why is the linear Stark effect zero for ground states?
3. Sketch the n = 2 energy levels vs electric field.

---

## Summary

| Stark Type | Condition | Energy Shift |
|------------|-----------|--------------|
| Quadratic | Non-degenerate | ΔE = -αE²/2 |
| Linear | Degenerate | ΔE = ±pE |

---

**Next:** [Day_455_Sunday.md](Day_455_Sunday.md) — Week 65 Review
