# Day 521: POVM Examples

## Overview
**Day 521** | Week 75, Day 3 | Year 1, Month 19 | Practical POVM Constructions

Today we study concrete POVM examples including unambiguous state discrimination and SIC-POVMs.

---

## Learning Objectives
1. Construct unambiguous state discrimination POVMs
2. Understand the trine POVM for qubits
3. Define symmetric informationally complete (SIC) POVMs
4. Calculate success probabilities for discrimination
5. Apply POVMs to practical scenarios

---

## Core Content

### Unambiguous State Discrimination

For distinguishing |ψ₀⟩ and |ψ₁⟩ with no errors (but possible inconclusive):

**POVM elements:**
- E₀: conclusively identifies |ψ₀⟩
- E₁: conclusively identifies |ψ₁⟩
- E?: inconclusive

**Condition:** E₀|ψ₁⟩ = 0 and E₁|ψ₀⟩ = 0

**Optimal success probability:**
$$p_{success} = 1 - |\langle\psi_0|\psi_1\rangle|$$

### Trine POVM

Three symmetric elements for qubit:
$$E_k = \frac{2}{3}|\phi_k\rangle\langle\phi_k|, \quad k = 0, 1, 2$$

where |φₖ⟩ are equally spaced on Bloch sphere equator.

### SIC-POVM

**Symmetric Informationally Complete:**
- d² elements in d dimensions
- |⟨ψᵢ|ψⱼ⟩|² = 1/(d+1) for i ≠ j
- Optimal for state tomography

---

## Computational Lab

```python
"""Day 521: POVM Examples"""
import numpy as np

# Unambiguous discrimination of |0⟩ and |+⟩
# |+⟩ = (|0⟩ + |1⟩)/√2, overlap = 1/√2

ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# POVM for unambiguous discrimination
# E_0 detects |0⟩ (orthogonal to |+⟩ → proportional to |−⟩⟨−|)
# E_1 detects |+⟩ (orthogonal to |0⟩ → proportional to |1⟩⟨1|)
overlap = abs(np.dot(ket_0.conj(), ket_plus))
p_success = 1 - overlap
print(f"Optimal success probability: {p_success:.4f}")
```

---

## Summary
- Unambiguous discrimination trades certainty for incompleteness
- SIC-POVMs are optimal for tomography
- Many practical applications require non-projective measurements

---
*Next: Day 522 — Neumark's Theorem*
