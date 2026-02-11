# Day 522: Neumark's Theorem

## Overview
**Day 522** | Week 75, Day 4 | Year 1, Month 19 | Dilating POVMs to Projective Measurements

Today we prove Neumark's theorem: every POVM can be realized as a projective measurement on a larger system.

---

## Learning Objectives
1. State Neumark's theorem precisely
2. Construct the dilation for given POVMs
3. Understand the physical meaning of dilation
4. Build the isometry V from POVM elements
5. Apply the theorem to specific examples

---

## Core Content

### Neumark's Theorem

**Theorem:** For any POVM {Eₘ} on ℋ, there exists:
- An ancilla space ℋ_A
- An isometry V: ℋ → ℋ ⊗ ℋ_A
- Projective measurement {Πₘ} on ℋ ⊗ ℋ_A

such that:
$$\boxed{E_m = V^\dagger \Pi_m V}$$

### Construction

Write Eₘ = Aₘ†Aₘ (positive square root).

Define V|ψ⟩ = Σₘ (Aₘ|ψ⟩) ⊗ |m⟩

Then Πₘ = I ⊗ |m⟩⟨m| gives the dilation.

### Physical Interpretation

1. Attach ancilla in state |0⟩
2. Apply unitary coupling system to ancilla
3. Perform projective measurement on ancilla
4. Result: POVM on original system

### Example: Three-Outcome Qubit POVM

For POVM with Eₘ = (2/3)|φₘ⟩⟨φₘ|:
- Ancilla dimension: 3 (qutrit)
- Isometry: V|ψ⟩ = Σₘ √(2/3)⟨φₘ|ψ⟩ |φₘ⟩|m⟩

---

## Computational Lab

```python
"""Day 522: Neumark's Theorem"""
import numpy as np

def neumark_dilation(povm_elements):
    """Construct Neumark dilation for POVM"""
    n_outcomes = len(povm_elements)
    d = povm_elements[0].shape[0]

    # Compute A_m = sqrt(E_m)
    A_list = []
    for E in povm_elements:
        evals, evecs = np.linalg.eigh(E)
        A = evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.conj().T
        A_list.append(A)

    # Build isometry V
    V = np.zeros((d * n_outcomes, d), dtype=complex)
    for m, A in enumerate(A_list):
        V[m*d:(m+1)*d, :] = A

    return V, A_list

# Example
E1 = 0.5 * np.array([[1,0],[0,0]], dtype=complex)
E2 = 0.5 * np.array([[0,0],[0,1]], dtype=complex)
E3 = 0.5 * np.eye(2, dtype=complex)  # Simplified
V, _ = neumark_dilation([E1, E2])
print(f"Isometry V shape: {V.shape}")
print(f"V†V = I: {np.allclose(V.conj().T @ V, np.eye(2))}")
```

---

## Summary
- Every POVM = projective measurement on larger space
- Neumark dilation provides constructive proof
- Deep connection between POVMs and ancilla systems

---
*Next: Day 523 — Measurement Implementations*
