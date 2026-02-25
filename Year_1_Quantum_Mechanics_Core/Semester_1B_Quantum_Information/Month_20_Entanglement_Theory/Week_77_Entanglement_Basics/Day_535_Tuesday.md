# Day 535: Entanglement Detection

## Overview
**Day 535** | Week 77, Day 3 | Year 1, Month 20 | Witnesses and Detection Methods

Today we study practical methods to detect entanglement, focusing on entanglement witnesses—observables that can certify entanglement through expectation values.

---

## Learning Objectives
1. Define entanglement witnesses mathematically
2. Construct witnesses for specific entangled states
3. Understand the geometric interpretation
4. Apply the range criterion for separability
5. Design experimentally measurable witnesses
6. Connect witnesses to Bell inequalities

---

## Core Content

### The Separability Problem

Given ρ, determine: Is ρ separable or entangled?

**Bad news:** This problem is NP-hard in general!

**Good news:** We have necessary conditions that are easy to check.

### Entanglement Witnesses

An **entanglement witness** is a Hermitian operator W such that:

$$\boxed{\text{Tr}(W\rho_{sep}) \geq 0 \text{ for all separable } \rho_{sep}}$$
$$\boxed{\text{Tr}(W\rho_{ent}) < 0 \text{ for some entangled } \rho_{ent}}$$

If $\text{Tr}(W\rho) < 0$, we **certify** that ρ is entangled.

### Geometric Interpretation

```
              Tr(Wρ) = 0 (hyperplane)
                    /
                   /
    ┌─────────────/─────────────┐
    │            /              │
    │  SEP     /    ENTANGLED   │
    │  (convex)/                │
    │         /                 │
    │        / ← witness detects│
    │       /     this region   │
    └──────/────────────────────┘
```

The separable states form a convex set. A witness W defines a hyperplane that:
- Has all separable states on one side ($\text{Tr}(W\rho) \geq 0$)
- Has some entangled states on the other side

### Constructing Witnesses

**Method 1: From Projectors**

For a pure entangled state $|\psi\rangle$:
$$W = \alpha I - |\psi\rangle\langle\psi|$$

Choose α = max separable overlap:
$$\alpha = \max_{|\phi\rangle \text{ product}} |\langle\phi|\psi\rangle|^2$$

**For Bell state** $|\Phi^+\rangle$: $\alpha = 1/2$ (max product overlap)
$$W_{\Phi^+} = \frac{1}{2}I - |\Phi^+\rangle\langle\Phi^+|$$

**Method 2: From Partial Transpose**

If $\rho^{T_B}$ has negative eigenvalue $-\lambda$ with eigenvector $|v\rangle$:
$$W = |v\rangle\langle v|^{T_B}$$

### Optimal Witnesses

A witness W is **optimal** if there is no other witness W' with:
$$\{ρ : \text{Tr}(W'\rho) < 0\} \supsetneq \{ρ : \text{Tr}(W\rho) < 0\}$$

Optimal witnesses are "tangent" to the separable set.

### Range Criterion

**Theorem:** If ρ is separable, then there exist product vectors $|a_i\rangle|b_i\rangle$ such that:
1. Range(ρ) is spanned by {$|a_i\rangle|b_i\rangle$}
2. Range($\rho^{T_B}$) is spanned by {$|a_i\rangle|b_i^*\rangle$}

**Corollary:** If Range(ρ) contains no product vectors, ρ is entangled.

### Experimentally Measurable Witnesses

For practical detection, decompose W into local observables:
$$W = \sum_{ij} c_{ij} A_i \otimes B_j$$

where $A_i$, $B_j$ are single-qubit observables (Paulis).

**Example for Bell state witness:**
$$W_{\Phi^+} = \frac{1}{4}(I \otimes I - X \otimes X + Y \otimes Y - Z \otimes Z)$$

Each term $A \otimes B$ requires joint measurement statistics.

### Witness Families

**Swap witness:**
$$W_{swap} = \frac{1}{2}(I - F)$$
where F is the swap operator: $F|ab\rangle = |ba\rangle$

Detects states with $\text{Tr}(F\rho) > 1$.

---

## Worked Examples

### Example 1: Bell State Witness
Verify that $W = \frac{1}{2}I - |\Phi^+\rangle\langle\Phi^+|$ detects $|\Phi^+\rangle$.

**Solution:**
$$\text{Tr}(W|\Phi^+\rangle\langle\Phi^+|) = \frac{1}{2}\text{Tr}(|\Phi^+\rangle\langle\Phi^+|) - \text{Tr}(|\Phi^+\rangle\langle\Phi^+||\Phi^+\rangle\langle\Phi^+|)$$
$$= \frac{1}{2}(1) - 1 = -\frac{1}{2} < 0$$

The witness detects the entanglement! ∎

### Example 2: Witness on Separable State
Show that $\text{Tr}(W|00\rangle\langle 00|) \geq 0$ for the same witness.

**Solution:**
$$|\langle 00|\Phi^+\rangle|^2 = |\frac{1}{\sqrt{2}}|^2 = \frac{1}{2}$$

$$\text{Tr}(W|00\rangle\langle 00|) = \frac{1}{2} - \frac{1}{2} = 0$$

The witness gives non-negative value for this product state. ∎

### Example 3: Werner State Detection
For which p does the witness detect $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)I/4$?

**Solution:**
Using $W = \frac{1}{2}I - |\Psi^-\rangle\langle\Psi^-|$:

$$\text{Tr}(W\rho_W) = \frac{1}{2} - p \cdot 1 - (1-p) \cdot \frac{1}{4}$$
$$= \frac{1}{2} - p - \frac{1-p}{4} = \frac{2 - 4p - 1 + p}{4} = \frac{1 - 3p}{4}$$

This is negative when $p > 1/3$.

The witness detects entanglement for $p > 1/3$. ∎

---

## Practice Problems

### Problem 1: GHZ Witness
Construct an entanglement witness for the GHZ state $|GHZ\rangle = (|000\rangle + |111\rangle)/\sqrt{2}$.

### Problem 2: Separable Verification
Show that any witness W satisfies $\text{Tr}(W) \geq 0$ for trace-1 constraints.

### Problem 3: Witness Optimization
Given witness $W_1$ for $|\Phi^+\rangle$ and $W_2$ for $|\Psi^+\rangle$, construct a witness that detects both.

---

## Computational Lab

```python
"""Day 535: Entanglement Detection"""
import numpy as np
from scipy.linalg import eigvalsh

# Bell states
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

def projector(psi):
    """Create projector |ψ⟩⟨ψ|"""
    return np.outer(psi, psi.conj())

def entanglement_witness(target_state, alpha=0.5):
    """Create witness W = αI - |ψ⟩⟨ψ|"""
    d = len(target_state)
    return alpha * np.eye(d) - projector(target_state)

def witness_value(W, rho):
    """Compute Tr(Wρ)"""
    return np.trace(W @ rho).real

def partial_transpose_B(rho, dim_A=2, dim_B=2):
    """Compute partial transpose over B"""
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_TB = rho_reshaped.transpose(0, 3, 2, 1)
    return rho_TB.reshape(dim_A * dim_B, dim_A * dim_B)

def witness_from_ppt(rho):
    """Construct witness from PPT negative eigenvector"""
    rho_TB = partial_transpose_B(rho)
    eigenvalues, eigenvectors = np.linalg.eigh(rho_TB)

    min_idx = np.argmin(eigenvalues)
    if eigenvalues[min_idx] >= -1e-10:
        return None, eigenvalues[min_idx]

    v = eigenvectors[:, min_idx]
    W_TB = np.outer(v, v.conj())
    W = partial_transpose_B(W_TB)  # Transpose back
    return W, eigenvalues[min_idx]

# Test witness on various states
print("=== Entanglement Witness Testing ===\n")

# Create witness for |Φ⁺⟩
W_phi = entanglement_witness(phi_plus, alpha=0.5)
print("Witness for |Φ⁺⟩: W = 0.5*I - |Φ⁺⟩⟨Φ⁺|")

# Test on Bell state
rho_bell = projector(phi_plus)
val = witness_value(W_phi, rho_bell)
print(f"Tr(W·|Φ⁺⟩⟨Φ⁺|) = {val:.4f} {'< 0 → ENTANGLED' if val < 0 else '>= 0'}")

# Test on product state |00⟩
rho_00 = projector(np.array([1, 0, 0, 0]))
val = witness_value(W_phi, rho_00)
print(f"Tr(W·|00⟩⟨00|) = {val:.4f} {'< 0 → ENTANGLED' if val < 0 else '>= 0'}")

# Test on maximally mixed state
rho_mixed = np.eye(4) / 4
val = witness_value(W_phi, rho_mixed)
print(f"Tr(W·I/4) = {val:.4f} {'< 0 → ENTANGLED' if val < 0 else '>= 0'}")

# Werner state analysis
print("\n=== Werner State Analysis ===")
print("ρ_W(p) = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4")

W_psi = entanglement_witness(psi_minus, alpha=0.5)
rho_psi = projector(psi_minus)

for p in [0.0, 0.25, 0.33, 0.34, 0.5, 0.75, 1.0]:
    rho_werner = p * rho_psi + (1-p) * np.eye(4) / 4
    val = witness_value(W_psi, rho_werner)
    status = "ENTANGLED" if val < -1e-10 else "not detected"
    print(f"p = {p:.2f}: Tr(W·ρ) = {val:+.4f} → {status}")

# PPT-based witness
print("\n=== PPT-Based Witness Construction ===")
rho_bell = projector(phi_plus)
W_ppt, min_eig = witness_from_ppt(rho_bell)
if W_ppt is not None:
    print(f"Min eigenvalue of ρ^TB: {min_eig:.4f}")
    val = witness_value(W_ppt, rho_bell)
    print(f"PPT witness value: {val:.4f}")

# Pauli decomposition of witness
print("\n=== Pauli Decomposition ===")
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

paulis = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
pauli_labels = ['I', 'X', 'Y', 'Z']

W = W_phi
print("Decomposition of W_Φ⁺ into Pauli basis:")
for p1 in pauli_labels:
    for p2 in pauli_labels:
        coeff = np.trace(np.kron(paulis[p1], paulis[p2]) @ W).real / 4
        if abs(coeff) > 1e-10:
            print(f"  {coeff:+.4f} * {p1}⊗{p2}")
```

**Expected Output:**
```
=== Entanglement Witness Testing ===

Witness for |Φ⁺⟩: W = 0.5*I - |Φ⁺⟩⟨Φ⁺|
Tr(W·|Φ⁺⟩⟨Φ⁺|) = -0.5000 < 0 → ENTANGLED
Tr(W·|00⟩⟨00|) = 0.0000 >= 0
Tr(W·I/4) = 0.2500 >= 0

=== Werner State Analysis ===
ρ_W(p) = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4
p = 0.00: Tr(W·ρ) = +0.2500 → not detected
p = 0.25: Tr(W·ρ) = +0.0625 → not detected
p = 0.33: Tr(W·ρ) = +0.0025 → not detected
p = 0.34: Tr(W·ρ) = -0.0100 → ENTANGLED
p = 0.50: Tr(W·ρ) = -0.1250 → ENTANGLED
p = 0.75: Tr(W·ρ) = -0.3125 → ENTANGLED
p = 1.00: Tr(W·ρ) = -0.5000 → ENTANGLED
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Witness condition | $\text{Tr}(W\rho_{sep}) \geq 0$, $\text{Tr}(W\rho_{ent}) < 0$ |
| Projector witness | $W = \alpha I - \|\psi\rangle\langle\psi\|$ |
| Max product overlap | $\alpha = \max_{\|φ\rangle \text{ product}} \|\langle φ\|\psi\rangle\|^2$ |
| Bell state α | $\alpha = 1/2$ |

### Key Takeaways
1. **Witnesses** are observables that certify entanglement
2. **Geometric picture**: hyperplane separating SEP from some ENT
3. **Construction** from projectors or PPT negative eigenvectors
4. **Experimental**: decompose into local Pauli measurements
5. **Limitation**: no single witness detects all entangled states

---

## Daily Checklist

- [ ] I understand what an entanglement witness is
- [ ] I can construct witnesses for specific states
- [ ] I understand the geometric interpretation
- [ ] I can check if a witness detects a given state
- [ ] I understand the Pauli decomposition for experiments

---

*Next: Day 536 — PPT Criterion Deep Dive*
