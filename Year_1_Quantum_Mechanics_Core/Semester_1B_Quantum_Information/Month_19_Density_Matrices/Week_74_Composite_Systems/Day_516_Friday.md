# Day 516: Entanglement Detection

## Overview

**Day 516** | Week 74, Day 5 | Year 1, Month 19 | Separability Criteria

Today we study how to determine whether a given quantum state is entangled or separable—a fundamental problem in quantum information theory.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Separability criteria |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. Define separable and entangled states for mixed states
2. Apply the PPT criterion for entanglement detection
3. Understand entanglement witnesses
4. Compute the partial transpose
5. Apply the reduction criterion
6. Recognize the difficulty of the separability problem

---

## Core Content

### Separable vs Entangled Mixed States

**Separable state:** Can be written as a convex combination of product states:
$$\boxed{\rho_{sep} = \sum_i p_i \rho_i^A \otimes \rho_i^B, \quad p_i \geq 0, \sum_i p_i = 1}$$

**Entangled state:** Cannot be written in this form.

**Key difference from pure states:** For mixed states, not entangled ≠ product state.

### The Partial Transpose

For a bipartite state ρ_AB, the **partial transpose** (with respect to B) is:
$$(\rho_{AB})^{T_B} = (I_A \otimes T)(\rho_{AB})$$

In the computational basis:
$$\langle ij|(\rho)^{T_B}|kl\rangle = \langle il|\rho|kj\rangle$$

### PPT Criterion (Peres-Horodecki)

**Theorem:** If ρ is separable, then ρ^(T_B) ≥ 0 (positive semidefinite).

**Contrapositive:** If ρ^(T_B) has a negative eigenvalue, ρ is entangled.

$$\boxed{\rho^{T_B} \not\geq 0 \Rightarrow \rho \text{ is entangled}}$$

**For 2×2 and 2×3 systems:** PPT is necessary AND sufficient for separability.

**For larger systems:** PPT is necessary but not sufficient (PPT entangled states exist).

### Computing Partial Transpose

For a 4×4 matrix in basis |00⟩, |01⟩, |10⟩, |11⟩:
$$\rho^{T_B} = \begin{pmatrix} \rho_{00,00} & \rho_{01,00} & \rho_{00,10} & \rho_{01,10} \\ \rho_{00,01} & \rho_{01,01} & \rho_{00,11} & \rho_{01,11} \\ \rho_{10,00} & \rho_{11,00} & \rho_{10,10} & \rho_{11,10} \\ \rho_{10,01} & \rho_{11,01} & \rho_{10,11} & \rho_{11,11} \end{pmatrix}$$

### Entanglement Witnesses

An **entanglement witness** W is an observable such that:
- Tr(Wρ_sep) ≥ 0 for all separable states
- Tr(Wρ_ent) < 0 for some entangled state

**Example:** For Bell state |Φ⁺⟩:
$$W = \frac{1}{2}I - |\Phi^+\rangle\langle\Phi^+|$$

Any state with ⟨W⟩ < 0 is entangled.

### Other Criteria

**Reduction criterion:**
$$\rho_A \otimes I - \rho_{AB} \geq 0 \text{ for separable states}$$

**CCNR criterion:** Based on realignment of the density matrix.

---

## Quantum Computing Connection

### Detecting Entanglement in Experiments

To verify entanglement in a quantum computer:
1. Prepare the state
2. Measure correlations (Bell inequalities) or
3. Perform state tomography and check PPT

### Entanglement as Resource

Entanglement is a **resource** for:
- Quantum teleportation
- Superdense coding
- Quantum key distribution
- Quantum error correction

Detecting it is crucial for verifying quantum protocols.

---

## Worked Examples

### Example 1: PPT Criterion for Bell State

**Problem:** Check if |Φ⁺⟩⟨Φ⁺| is entangled using PPT.

**Solution:**

$$\rho = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

Partial transpose:
$$\rho^{T_B} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Eigenvalues of ρ^(T_B): {½, ½, ½, -½}

There's a negative eigenvalue (-½), so **the state is entangled**.

### Example 2: Separable State Check

**Problem:** Is ρ = ½|00⟩⟨00| + ½|11⟩⟨11| entangled?

**Solution:**

This is clearly separable: ρ = ½(|0⟩⟨0| ⊗ |0⟩⟨0|) + ½(|1⟩⟨1| ⊗ |1⟩⟨1|).

Check PPT:
$$\rho = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}, \quad \rho^{T_B} = \rho$$

All eigenvalues ≥ 0, consistent with separability.

### Example 3: Werner State

**Problem:** For what values of p is the Werner state ρ = p|Φ⁺⟩⟨Φ⁺| + (1-p)I/4 entangled?

**Solution:**

The partial transpose is:
$$\rho^{T_B} = p \cdot \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} + (1-p)\frac{I}{4}$$

Minimum eigenvalue: (1-p)/4 - p/2 = (1-3p)/4

Negative when p > 1/3.

**Result:** Werner state is entangled for p > 1/3.

---

## Practice Problems

### Direct Application

**Problem 1:** Compute the partial transpose of |Ψ⁺⟩⟨Ψ⁺| where |Ψ⁺⟩ = (|01⟩+|10⟩)/√2.

**Problem 2:** Is ρ = |+⟩⟨+| ⊗ |0⟩⟨0| entangled?

**Problem 3:** Find the entanglement witness that detects |Ψ⁺⟩.

### Intermediate

**Problem 4:** Show that partial transpose preserves trace.

**Problem 5:** Prove that product states satisfy PPT.

**Problem 6:** For what p is ρ = p|Ψ⁺⟩⟨Ψ⁺| + (1-p)|00⟩⟨00| entangled?

### Challenging

**Problem 7:** Show that PPT implies separability for 2×2 systems (Horodecki theorem).

**Problem 8:** Construct a PPT entangled state for a 3×3 system.

**Problem 9:** Prove the reduction criterion: separable ⟹ ρ_A ⊗ I - ρ ≥ 0.

---

## Computational Lab

```python
"""
Day 516: Entanglement Detection
PPT criterion and entanglement witnesses
"""

import numpy as np
import matplotlib.pyplot as plt

def partial_transpose_B(rho, dim_A, dim_B):
    """Compute partial transpose over system B"""
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    return rho_reshaped.transpose(0, 3, 2, 1).reshape(dim_A*dim_B, dim_A*dim_B)

def is_ppt(rho, dim_A, dim_B):
    """Check if state satisfies PPT criterion"""
    rho_pt = partial_transpose_B(rho, dim_A, dim_B)
    min_eig = np.min(np.linalg.eigvalsh(rho_pt))
    return min_eig >= -1e-10, min_eig

def density(psi):
    return np.outer(psi, psi.conj())

# Bell states
ket_00 = np.array([1, 0, 0, 0], dtype=complex)
ket_01 = np.array([0, 1, 0, 0], dtype=complex)
ket_10 = np.array([0, 0, 1, 0], dtype=complex)
ket_11 = np.array([0, 0, 0, 1], dtype=complex)

phi_plus = (ket_00 + ket_11) / np.sqrt(2)
psi_plus = (ket_01 + ket_10) / np.sqrt(2)

print("=" * 60)
print("PPT CRITERION EXAMPLES")
print("=" * 60)

# Test various states
states = [
    ("Bell |Φ⁺⟩", density(phi_plus)),
    ("Bell |Ψ⁺⟩", density(psi_plus)),
    ("Product |00⟩", density(ket_00)),
    ("Classical mix", 0.5*density(ket_00) + 0.5*density(ket_11)),
]

for name, rho in states:
    ppt, min_eig = is_ppt(rho, 2, 2)
    status = "Separable (PPT)" if ppt else "Entangled (NPT)"
    print(f"\n{name}:")
    print(f"  Min eigenvalue of ρ^TB: {min_eig:.4f}")
    print(f"  Status: {status}")

print("\n" + "=" * 60)
print("WERNER STATE ANALYSIS")
print("=" * 60)

p_vals = np.linspace(0, 1, 100)
min_eigs = []

for p in p_vals:
    rho_werner = p * density(phi_plus) + (1-p) * np.eye(4)/4
    _, min_eig = is_ppt(rho_werner, 2, 2)
    min_eigs.append(min_eig)

plt.figure(figsize=(10, 5))
plt.plot(p_vals, min_eigs, 'b-', lw=2)
plt.axhline(0, color='r', ls='--', label='PPT boundary')
plt.axvline(1/3, color='g', ls='--', label='p = 1/3')
plt.fill_between(p_vals, -0.3, 0, where=np.array(min_eigs)<0,
                 alpha=0.3, color='red', label='Entangled region')
plt.xlabel('p (Bell state weight)')
plt.ylabel('Min eigenvalue of ρ^TB')
plt.title('Werner State: PPT Criterion')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.3, 0.3)
plt.savefig('ppt_criterion.png', dpi=150, bbox_inches='tight')
plt.show()

# Find threshold
threshold_idx = np.where(np.array(min_eigs) < 0)[0][0]
print(f"\nWerner state becomes entangled at p ≈ {p_vals[threshold_idx]:.3f}")
print(f"Theoretical threshold: p = 1/3 ≈ {1/3:.3f}")

print("\n" + "=" * 60)
print("ENTANGLEMENT WITNESS")
print("=" * 60)

# Witness for |Φ⁺⟩
W = 0.5 * np.eye(4) - density(phi_plus)
print(f"\nWitness W = I/2 - |Φ⁺⟩⟨Φ⁺|")

# Test on various states
test_states = [
    ("Bell |Φ⁺⟩", density(phi_plus)),
    ("Bell |Ψ⁺⟩", density(psi_plus)),
    ("Product |+0⟩", np.kron(density(np.array([1,1])/np.sqrt(2)), density(np.array([1,0])))),
    ("Max mixed", np.eye(4)/4),
]

for name, rho in test_states:
    witness_val = np.trace(W @ rho).real
    detected = "ENTANGLED" if witness_val < 0 else "not detected"
    print(f"{name}: Tr(Wρ) = {witness_val:.4f} → {detected}")

print("\n" + "=" * 60)
print("Day 516 Complete: Entanglement Detection")
print("=" * 60)
```

---

## Summary

### Key Criteria

| Criterion | Formula | Completeness |
|-----------|---------|--------------|
| PPT | ρ^(T_B) ≥ 0 | Necessary; sufficient for 2×2, 2×3 |
| Witness | Tr(Wρ) < 0 | Depends on witness |
| Reduction | ρ_A⊗I - ρ ≥ 0 | Weaker than PPT |

### Key Results

- **PPT criterion** detects many entangled states via negative partial transpose eigenvalues
- **Entanglement witnesses** are observables that can detect specific entangled states
- **Werner state threshold:** p > 1/3 for entanglement

---

## Daily Checklist

- [ ] I can compute partial transposes
- [ ] I can apply the PPT criterion
- [ ] I understand entanglement witnesses
- [ ] I know when PPT is sufficient for separability

---

## Preview: Day 517

Tomorrow we study **purification**—how any mixed state can be viewed as part of a larger pure state.

---

*Next: Day 517 — Purification*
