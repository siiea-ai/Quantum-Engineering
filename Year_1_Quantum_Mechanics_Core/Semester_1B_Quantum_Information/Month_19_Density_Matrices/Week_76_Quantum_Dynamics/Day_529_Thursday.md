# Day 529: Kraus Representation

## Overview
**Day 529** | Week 76, Day 4 | Year 1, Month 19 | Operator-Sum Representation

Today we prove the Kraus representation theorem—every CPTP map has an operator-sum form.

---

## Learning Objectives
1. State the Kraus representation theorem
2. Derive Kraus operators from Choi matrix
3. Understand non-uniqueness of Kraus operators
4. Apply to physical examples
5. Connect Kraus to Stinespring dilation

---

## Core Content

### Kraus Representation Theorem

**Theorem:** Every CPTP map E can be written as:

$$\boxed{\mathcal{E}(\rho) = \sum_{k=1}^{r} K_k \rho K_k^\dagger}$$

where:
- {Kₖ} are **Kraus operators**
- Trace preservation: Σₖ Kₖ†Kₖ = I
- r ≤ d² (at most d² operators needed)

### Derivation from Choi

If J(E) = Σᵢ λᵢ|vᵢ⟩⟨vᵢ| (spectral decomposition), then:
$$K_i = \sqrt{d \lambda_i} \cdot \text{reshape}(|v_i\rangle)$$

### Non-Uniqueness

If {Kₖ} are Kraus operators for E, so are:
$$K_k' = \sum_j U_{kj} K_j$$

for any unitary U (with appropriate dimensions).

### Physical Interpretation

Kraus form arises from:
1. System-environment interaction: U_SE
2. Tracing out environment: Tr_E(U_SE ρ⊗|0⟩⟨0| U_SE†)
3. Kₖ = ⟨k|U_SE|0⟩

---

## Computational Lab

```python
"""Day 529: Kraus Representation"""
import numpy as np

def kraus_to_choi(kraus_ops, d=2):
    """Convert Kraus operators to Choi matrix"""
    J = np.zeros((d**2, d**2), dtype=complex)
    for K in kraus_ops:
        vec_K = K.flatten()
        J += np.outer(vec_K, vec_K.conj())
    return J

def apply_kraus(rho, kraus_ops):
    """Apply channel via Kraus representation"""
    result = np.zeros_like(rho)
    for K in kraus_ops:
        result += K @ rho @ K.conj().T
    return result

# Example: Bit-flip channel
p = 0.1
K0 = np.sqrt(1-p) * np.eye(2, dtype=complex)
K1 = np.sqrt(p) * np.array([[0,1],[1,0]], dtype=complex)

# Verify trace preservation
print(f"Trace preservation: {np.allclose(K0.conj().T@K0 + K1.conj().T@K1, np.eye(2))}")

# Apply to |+⟩
rho_plus = 0.5 * np.array([[1,1],[1,1]], dtype=complex)
rho_out = apply_kraus(rho_plus, [K0, K1])
print(f"Output state:\n{rho_out}")
```

---

## Summary
- Every CPTP map has Kraus form: E(ρ) = Σₖ KₖρKₖ†
- Kraus operators encode system-environment interaction
- Non-unique but all physically equivalent

---
*Next: Day 530 — Important Channels*
