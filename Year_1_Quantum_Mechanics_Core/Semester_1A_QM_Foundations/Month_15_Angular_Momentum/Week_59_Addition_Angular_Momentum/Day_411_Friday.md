# Day 411: Two Spin-1/2 Addition — Singlet and Triplet

## Overview
**Day 411** | Year 1, Month 15, Week 59 | Entanglement from Angular Momentum

Adding two spin-1/2 particles produces the singlet (j=0) and triplet (j=1) states—directly connected to Bell states and entanglement.

---

## Core Content

### The Addition: 1/2 ⊗ 1/2 = 0 ⊕ 1

Two spin-1/2 particles: dimension = 2 × 2 = 4

Triangle rule: |1/2 - 1/2| ≤ j ≤ 1/2 + 1/2 → j = 0, 1

- **Triplet (j=1):** 3 states (symmetric)
- **Singlet (j=0):** 1 state (antisymmetric)

### Triplet States (j = 1)

$$|1,+1\rangle = |{\uparrow\uparrow}\rangle$$

$$|1,0\rangle = \frac{1}{\sqrt{2}}(|{\uparrow\downarrow}\rangle + |{\downarrow\uparrow}\rangle)$$

$$|1,-1\rangle = |{\downarrow\downarrow}\rangle$$

All symmetric under particle exchange!

### Singlet State (j = 0)

$$\boxed{|0,0\rangle = \frac{1}{\sqrt{2}}(|{\uparrow\downarrow}\rangle - |{\downarrow\uparrow}\rangle)}$$

**Antisymmetric** under particle exchange: |0,0⟩ → -|0,0⟩

### Connection to Bell States

| Angular Momentum | Bell State |
|-----------------|------------|
| \|0,0⟩ (singlet) | \|Ψ⁻⟩ = (\|01⟩ - \|10⟩)/√2 |
| \|1,0⟩ (triplet) | \|Ψ⁺⟩ = (\|01⟩ + \|10⟩)/√2 |
| \|1,+1⟩ | \|Φ⁺⟩ (rotated) |

The singlet is **maximally entangled** and rotationally invariant!

### Singlet Properties

For the singlet state:
- ⟨Ŝ₁ᵤ⟩ = ⟨Ŝ₂ᵤ⟩ = 0
- ⟨Ŝ₁ᵤŜ₂ᵤ⟩ = -ℏ²/4
- **Perfect anticorrelation:** If particle 1 is spin-up, particle 2 is spin-down

---

## Computational Lab

```python
"""Day 411: Two Spin-1/2 Addition"""
import numpy as np

# Basis: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩
triplet_11 = np.array([1, 0, 0, 0])
triplet_10 = np.array([0, 1, 1, 0]) / np.sqrt(2)
triplet_1m1 = np.array([0, 0, 0, 1])
singlet_00 = np.array([0, 1, -1, 0]) / np.sqrt(2)

# Verify orthogonality
print(f"⟨triplet|singlet⟩ = {np.dot(triplet_10, singlet_00):.4f}")
print(f"⟨singlet|singlet⟩ = {np.dot(singlet_00, singlet_00):.4f}")
```

---

## Practice Problems
1. Verify |1,0⟩ and |0,0⟩ are orthonormal.
2. Show the singlet is antisymmetric under exchange.
3. Calculate ⟨Ŝ₁·Ŝ₂⟩ for the singlet.

---

**Next:** [Day_412_Saturday.md](Day_412_Saturday.md) — Wigner 3j Symbols
