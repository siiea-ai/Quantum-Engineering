# Day 528: Completely Positive Maps

## Overview
**Day 528** | Week 76, Day 3 | Year 1, Month 19 | The Choi-Jamiolkowski Isomorphism

Today we study complete positivity in depth, including the powerful Choi matrix representation.

---

## Learning Objectives
1. Define complete positivity rigorously
2. Construct and interpret the Choi matrix
3. Apply the Choi-Jamiolkowski isomorphism
4. Test for complete positivity using Choi matrix
5. Understand why CP is necessary for physical operations

---

## Core Content

### Complete Positivity

A map E is **completely positive (CP)** if:
$$(E \otimes I_n)(\rho) \geq 0$$
for all n and all ρ ≥ 0 on ℋ ⊗ ℋₙ.

### Choi Matrix

The **Choi matrix** of E is:
$$\boxed{J(\mathcal{E}) = (\mathcal{E} \otimes I)(|\Phi^+\rangle\langle\Phi^+|)}$$

where |Φ⁺⟩ = Σᵢ|ii⟩/√d is the maximally entangled state.

### Choi's Theorem

$$\mathcal{E} \text{ is CP} \Leftrightarrow J(\mathcal{E}) \geq 0$$

**To test:** Compute Choi matrix, check if positive semidefinite.

### Properties

| Map Property | Choi Matrix Condition |
|--------------|----------------------|
| CP | J(E) ≥ 0 |
| TP | Tr_output(J(E)) = I |
| CPTP | J(E) ≥ 0 and Tr_out(J(E)) = I |
| Unitary | J(E) is rank-1 |

---

## Computational Lab

```python
"""Day 528: Completely Positive Maps"""
import numpy as np

def choi_matrix(channel_func, d=2):
    """Compute Choi matrix for a channel"""
    # Maximally entangled state
    phi_plus = np.zeros((d**2, d**2), dtype=complex)
    for i in range(d):
        for j in range(d):
            phi_plus[i*d+i, j*d+j] = 1/d

    # Apply E ⊗ I
    J = np.zeros((d**2, d**2), dtype=complex)
    for i in range(d):
        for j in range(d):
            # Input: |i⟩⟨j| on first system
            input_state = np.zeros((d, d), dtype=complex)
            input_state[i, j] = 1
            output = channel_func(input_state)
            for k in range(d):
                for l in range(d):
                    J[k*d+i, l*d+j] = output[k, l]
    return J / d

def is_cp(channel_func, d=2):
    J = choi_matrix(channel_func, d)
    eigs = np.linalg.eigvalsh(J)
    return np.min(eigs) >= -1e-10

# Test depolarizing channel
def depol(rho, p=0.5):
    return (1-p)*rho + p*np.eye(2)/2

print(f"Depolarizing is CP: {is_cp(depol)}")
```

---

## Summary
- CP is tested via positive semidefiniteness of Choi matrix
- Choi-Jamiolkowski provides bijection: channels ↔ states
- Essential for characterizing quantum noise

---
*Next: Day 529 — Kraus Representation*
