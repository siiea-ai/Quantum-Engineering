# Day 524: Optimal Measurements

## Overview
**Day 524** | Week 75, Day 6 | Year 1, Month 19 | Minimum Error State Discrimination

Today we study optimal measurement strategies, including the Holevo-Helstrom theorem for binary state discrimination.

---

## Learning Objectives
1. State the Holevo-Helstrom theorem
2. Derive the minimum error probability
3. Construct optimal measurements for binary discrimination
4. Understand the role of trace distance
5. Apply optimization to multiple-state scenarios

---

## Core Content

### Binary State Discrimination

Given states ρ₀, ρ₁ with prior probabilities p₀, p₁:

**Minimum error probability:**
$$\boxed{P_{error}^{min} = \frac{1}{2}\left(1 - D(p_0\rho_0, p_1\rho_1)\right)}$$

where D is trace distance: D(A,B) = ½Tr|A-B|

### Helstrom Measurement

The optimal POVM has two elements:
- E₀ = projector onto positive eigenspace of p₀ρ₀ - p₁ρ₁
- E₁ = I - E₀

### Special Case: Pure States with Equal Priors

For |ψ₀⟩, |ψ₁⟩ with p₀ = p₁ = ½:
$$P_{error}^{min} = \frac{1}{2}(1 - \sqrt{1-|\langle\psi_0|\psi_1\rangle|^2})$$

### Pretty Good Measurement (PGM)

For multiple states: Eₘ ∝ ρ⁻¹/² pₘρₘ ρ⁻¹/²

Not always optimal but often nearly optimal.

---

## Computational Lab

```python
"""Day 524: Optimal Measurements"""
import numpy as np

def min_error_probability(rho0, rho1, p0=0.5, p1=0.5):
    """Compute minimum error probability (Helstrom)"""
    diff = p0 * rho0 - p1 * rho1
    trace_dist = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff)))
    return 0.5 * (1 - trace_dist)

# Example: |0⟩ vs |+⟩
rho0 = np.array([[1,0],[0,0]], dtype=complex)
rho_plus = 0.5 * np.array([[1,1],[1,1]], dtype=complex)

p_err = min_error_probability(rho0, rho_plus)
print(f"Minimum error probability: {p_err:.4f}")

# Compare to random guessing
print(f"Random guessing: 0.5000")
print(f"Improvement: {(0.5 - p_err)/0.5*100:.1f}%")
```

---

## Summary
- Helstrom measurement minimizes error for binary discrimination
- Trace distance determines distinguishability
- Optimal measurements often require POVMs

---
*Next: Day 525 — Week Review*
