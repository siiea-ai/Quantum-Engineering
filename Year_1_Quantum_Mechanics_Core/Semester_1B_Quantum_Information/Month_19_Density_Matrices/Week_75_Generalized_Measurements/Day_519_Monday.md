# Day 519: Projective Measurements Review

## Overview
**Day 519** | Week 75, Day 1 | Year 1, Month 19 | von Neumann Measurements

Today we review projective (von Neumann) measurements as preparation for the more general POVM framework.

---

## Learning Objectives
1. State the measurement postulate for projective measurements
2. Apply the spectral theorem to construct measurement operators
3. Calculate measurement probabilities and post-measurement states
4. Understand limitations of projective measurements
5. Recognize when projective measurements are insufficient

---

## Core Content

### The Measurement Postulate (Projective)

For observable A with spectral decomposition A = Σₘ aₘ Πₘ:

**Probability:** p(m) = Tr(Πₘρ)

**Post-measurement state:** ρ → Πₘ ρ Πₘ / p(m)

### Key Properties
- Πₘ are orthogonal projectors: Πₘ Πₙ = δₘₙ Πₘ
- Completeness: Σₘ Πₘ = I
- Idempotent: Πₘ² = Πₘ
- Hermitian: Πₘ† = Πₘ

### Limitations
1. Cannot distinguish non-orthogonal states with certainty
2. Number of outcomes ≤ dimension
3. Not always optimal for state discrimination

---

## Computational Lab

```python
"""Day 519: Projective Measurements"""
import numpy as np

# Projective measurement in computational basis
Pi_0 = np.array([[1,0],[0,0]], dtype=complex)
Pi_1 = np.array([[0,0],[0,1]], dtype=complex)

# Example: measure |+⟩
rho_plus = 0.5 * np.array([[1,1],[1,1]], dtype=complex)

p_0 = np.trace(Pi_0 @ rho_plus).real
p_1 = np.trace(Pi_1 @ rho_plus).real
print(f"p(0) = {p_0:.4f}, p(1) = {p_1:.4f}")
```

---

## Summary
- Projective measurements are orthogonal, repeatable, and limited
- Spectral theorem provides the mathematical foundation
- Tomorrow: generalize beyond projective measurements

---
*Next: Day 520 — POVM Introduction*
