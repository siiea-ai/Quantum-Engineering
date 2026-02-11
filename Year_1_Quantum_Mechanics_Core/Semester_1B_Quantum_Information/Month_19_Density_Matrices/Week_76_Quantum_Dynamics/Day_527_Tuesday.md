# Day 527: Quantum Operations

## Overview
**Day 527** | Week 76, Day 2 | Year 1, Month 19 | Maps on Density Matrices

Today we study quantum operations—the most general transformations of quantum states.

---

## Learning Objectives
1. Define quantum operations (quantum channels)
2. State the physical requirements for valid operations
3. Understand linearity, trace preservation, and positivity
4. Distinguish unitary from non-unitary evolution
5. Connect to open system dynamics

---

## Core Content

### Quantum Operation Definition

A **quantum operation** (channel) is a map E: ρ → E(ρ) satisfying:

1. **Linearity:** E(αρ + βσ) = αE(ρ) + βE(σ)
2. **Trace preservation:** Tr(E(ρ)) = Tr(ρ)
3. **Complete positivity:** (E ⊗ I)(ρ) ≥ 0 for all ρ ≥ 0

### Why Complete Positivity?

Mere positivity (E(ρ) ≥ 0 for ρ ≥ 0) is not enough!

If system A interacts with reference R:
- E ⊗ I must preserve positivity of joint states
- This requires **complete** positivity

### Examples of Quantum Operations

**Unitary channel:** E(ρ) = UρU†

**Measurement channel:** E(ρ) = Σₘ ΠₘρΠₘ

**Partial trace:** E(ρ_AB) = Tr_B(ρ_AB)

**Depolarizing:** E(ρ) = (1-p)ρ + pI/d

---

## Computational Lab

```python
"""Day 527: Quantum Operations"""
import numpy as np

def is_trace_preserving(channel_func, d=2):
    """Check if channel preserves trace"""
    for _ in range(10):
        rho = np.random.rand(d, d) + 1j*np.random.rand(d, d)
        rho = rho @ rho.conj().T
        rho /= np.trace(rho)
        rho_out = channel_func(rho)
        if not np.isclose(np.trace(rho_out), 1):
            return False
    return True

def depolarizing(rho, p):
    d = rho.shape[0]
    return (1-p) * rho + p * np.eye(d) / d

# Verify
print(f"Depolarizing is trace-preserving: {is_trace_preserving(lambda r: depolarizing(r, 0.3))}")
```

---

## Summary
- Quantum operations are linear, trace-preserving, completely positive maps
- Complete positivity ensures consistency with entangled states
- Foundation for noise modeling in quantum computing

---
*Next: Day 528 — Completely Positive Maps*
