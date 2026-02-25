# Day 513: Partial Trace

## Overview

**Day 513** | Week 74, Day 2 | Year 1, Month 19 | Tracing Out Subsystems

Today we introduce the partial trace—the mathematical operation that describes what happens when we "ignore" or "trace out" part of a composite quantum system. This is the key tool connecting global states to local descriptions.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Partial trace theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. Define the partial trace operation mathematically
2. Compute partial traces for two-qubit systems
3. Understand the physical interpretation of tracing out
4. Verify that partial trace gives a valid density matrix
5. Connect partial trace to entanglement (pure → mixed)
6. Apply partial trace to compute local observables

---

## Core Content

### Definition of Partial Trace

For a bipartite system ρ_AB on ℋ_A ⊗ ℋ_B, the **partial trace over B** gives:

$$\boxed{\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B)\rho_{AB}(I_A \otimes |j\rangle_B)}$$

where {|j⟩_B} is any orthonormal basis for ℋ_B.

**Alternative notation:**
$$\rho_A = \sum_j \langle j|_B \rho_{AB} |j\rangle_B$$

### Partial Trace of Product States

For ρ_AB = ρ_A ⊗ ρ_B:
$$\text{Tr}_B(\rho_A \otimes \rho_B) = \rho_A \cdot \text{Tr}(\rho_B) = \rho_A$$

The partial trace "removes" the traced-out system.

### Partial Trace of Pure Entangled States

For |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:
$$\rho_{AB} = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|)$$

Tracing over B:
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

**Key insight:** The Bell state is pure globally but **maximally mixed locally**!

### Why Partial Trace?

**Physical interpretation:** If we only have access to subsystem A:
- We cannot perform measurements on B
- All our predictions come from ρ_A = Tr_B(ρ_AB)
- ρ_A contains all information needed for local predictions

**Theorem:** For any observable A_A acting only on system A:
$$\langle A_A \rangle = \text{Tr}(\rho_{AB}(A_A \otimes I_B)) = \text{Tr}(\rho_A A_A)$$

### Computation Rules

**Rule 1:** Trace is linear
$$\text{Tr}_B(\alpha\rho + \beta\sigma) = \alpha\text{Tr}_B(\rho) + \beta\text{Tr}_B(\sigma)$$

**Rule 2:** For |i⟩⟨k|_A ⊗ |j⟩⟨l|_B:
$$\text{Tr}_B(|i\rangle\langle k|_A \otimes |j\rangle\langle l|_B) = |i\rangle\langle k|_A \cdot \langle l|j\rangle = |i\rangle\langle k|_A \cdot \delta_{jl}$$

**Rule 3:** The partial trace gives a valid density matrix (preserves positivity and normalization).

### Matrix Computation

For a 4×4 density matrix in the basis |00⟩, |01⟩, |10⟩, |11⟩:
$$\rho_{AB} = \begin{pmatrix} \rho_{00,00} & \rho_{00,01} & \rho_{00,10} & \rho_{00,11} \\ \rho_{01,00} & \rho_{01,01} & \rho_{01,10} & \rho_{01,11} \\ \rho_{10,00} & \rho_{10,01} & \rho_{10,10} & \rho_{10,11} \\ \rho_{11,00} & \rho_{11,01} & \rho_{11,10} & \rho_{11,11} \end{pmatrix}$$

The partial trace over B:
$$\rho_A = \begin{pmatrix} \rho_{00,00} + \rho_{01,01} & \rho_{00,10} + \rho_{01,11} \\ \rho_{10,00} + \rho_{11,01} & \rho_{10,10} + \rho_{11,11} \end{pmatrix}$$

---

## Quantum Computing Connection

### Entanglement Creates Mixedness

When qubits are entangled with environment:
1. Global state |ψ⟩_SE is pure
2. System state ρ_S = Tr_E(|ψ⟩⟨ψ|) is mixed
3. This is **decoherence**!

### Quantum Channels

Many quantum channels can be modeled as:
1. System interacts with environment
2. Trace out environment
3. Result: system undergoes non-unitary evolution

### Multi-Qubit Operations

To see the effect on one qubit after a two-qubit gate:
$$\rho_1' = \text{Tr}_2(U \rho_{12} U^\dagger)$$

---

## Worked Examples

### Example 1: Bell State Partial Trace

**Problem:** Compute Tr_B(|Ψ⁺⟩⟨Ψ⁺|) where |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2.

**Solution:**

$$|\Psi^+\rangle\langle\Psi^+| = \frac{1}{2}(|01\rangle\langle 01| + |01\rangle\langle 10| + |10\rangle\langle 01| + |10\rangle\langle 10|)$$

Apply Tr_B to each term:
- Tr_B(|01⟩⟨01|) = |0⟩⟨0|·⟨1|1⟩ = |0⟩⟨0|
- Tr_B(|01⟩⟨10|) = |0⟩⟨1|·⟨0|1⟩ = 0
- Tr_B(|10⟩⟨01|) = |1⟩⟨0|·⟨1|0⟩ = 0
- Tr_B(|10⟩⟨10|) = |1⟩⟨1|·⟨0|0⟩ = |1⟩⟨1|

$$\rho_A = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

### Example 2: Product State Partial Trace

**Problem:** Compute Tr_B(|+⟩⟨+| ⊗ |0⟩⟨0|).

**Solution:**

$$\text{Tr}_B(|+\rangle\langle +| \otimes |0\rangle\langle 0|) = |+\rangle\langle +| \cdot \text{Tr}(|0\rangle\langle 0|) = |+\rangle\langle +| \cdot 1 = |+\rangle\langle +|$$

For product states, partial trace just removes the other system.

### Example 3: General Two-Qubit State

**Problem:** For |ψ⟩ = (|00⟩ + |01⟩ + |10⟩)/√3, find ρ_A.

**Solution:**

$$\rho_{AB} = \frac{1}{3}\begin{pmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Using the partial trace formula:
$$\rho_A = \begin{pmatrix} \rho_{00,00} + \rho_{01,01} & \rho_{00,10} + \rho_{01,11} \\ \rho_{10,00} + \rho_{11,01} & \rho_{10,10} + \rho_{11,11} \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$

---

## Practice Problems

### Direct Application

**Problem 1:** Compute Tr_B(|00⟩⟨00|).

**Problem 2:** For ρ = ½|00⟩⟨00| + ½|11⟩⟨11|, find Tr_B(ρ).

**Problem 3:** Verify Tr(Tr_B(ρ_AB)) = Tr(ρ_AB) = 1.

### Intermediate

**Problem 4:** Show that Tr_B(ρ_A ⊗ ρ_B) = ρ_A for any ρ_B.

**Problem 5:** For |GHZ⟩ = (|000⟩ + |111⟩)/√2, find Tr_C(|GHZ⟩⟨GHZ|).

**Problem 6:** Prove: ⟨A ⊗ I⟩_ρAB = ⟨A⟩_Tr_B(ρAB).

### Challenging

**Problem 7:** Show that partial trace preserves positive semidefiniteness.

**Problem 8:** For a bipartite pure state |ψ⟩_AB, prove Tr(ρ_A²) = Tr(ρ_B²).

**Problem 9:** Derive the partial trace formula from the definition using a general basis.

---

## Computational Lab

```python
"""
Day 513: Partial Trace
Computing reduced density matrices
"""

import numpy as np

def partial_trace_B(rho_AB, dim_A, dim_B):
    """
    Compute partial trace over system B.
    rho_AB: density matrix of composite system
    dim_A, dim_B: dimensions of subsystems
    """
    rho_AB = np.array(rho_AB).reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho_AB, axis1=1, axis2=3)

def partial_trace_A(rho_AB, dim_A, dim_B):
    """Compute partial trace over system A."""
    rho_AB = np.array(rho_AB).reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho_AB, axis1=0, axis2=2)

# Standard states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)

# Bell states
ket_00 = np.kron(ket_0, ket_0)
ket_01 = np.kron(ket_0, ket_1)
ket_10 = np.kron(ket_1, ket_0)
ket_11 = np.kron(ket_1, ket_1)

phi_plus = (ket_00 + ket_11) / np.sqrt(2)
psi_plus = (ket_01 + ket_10) / np.sqrt(2)

def density(psi):
    return np.outer(psi, psi.conj())

print("=" * 60)
print("PARTIAL TRACE EXAMPLES")
print("=" * 60)

# Example 1: Bell state |Φ⁺⟩
print("\n--- Bell state |Φ⁺⟩ ---")
rho_phi = density(phi_plus)
rho_A = partial_trace_B(rho_phi, 2, 2)
rho_B = partial_trace_A(rho_phi, 2, 2)

print(f"Global state ρ_AB (pure):")
print(f"  Purity: {np.trace(rho_phi @ rho_phi).real:.4f}")

print(f"\nReduced state ρ_A = Tr_B(ρ_AB):")
print(rho_A)
print(f"  Purity: {np.trace(rho_A @ rho_A).real:.4f}")

print(f"\nReduced state ρ_B = Tr_A(ρ_AB):")
print(rho_B)
print(f"  Purity: {np.trace(rho_B @ rho_B).real:.4f}")

# Example 2: Product state
print("\n--- Product state |+⟩|0⟩ ---")
psi_product = np.kron(ket_plus, ket_0)
rho_product = density(psi_product)
rho_A_prod = partial_trace_B(rho_product, 2, 2)
rho_B_prod = partial_trace_A(rho_product, 2, 2)

print(f"ρ_A = Tr_B(|+⟩⟨+| ⊗ |0⟩⟨0|):")
print(rho_A_prod)
print(f"  Purity: {np.trace(rho_A_prod @ rho_A_prod).real:.4f}")

print(f"\nρ_B = Tr_A(|+⟩⟨+| ⊗ |0⟩⟨0|):")
print(rho_B_prod)
print(f"  Purity: {np.trace(rho_B_prod @ rho_B_prod).real:.4f}")

# Example 3: Mixed global state
print("\n--- Mixed global state ---")
rho_mixed_AB = 0.5 * density(ket_00) + 0.5 * density(ket_11)
rho_A_mixed = partial_trace_B(rho_mixed_AB, 2, 2)

print(f"ρ_AB = ½|00⟩⟨00| + ½|11⟩⟨11|")
print(f"ρ_A = Tr_B(ρ_AB):")
print(rho_A_mixed)

# Verify expectation value formula
print("\n" + "=" * 60)
print("EXPECTATION VALUE VERIFICATION")
print("=" * 60)

Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# For Bell state
Z_I = np.kron(Z, I)
exp_Z_direct = np.trace(rho_phi @ Z_I).real
exp_Z_reduced = np.trace(rho_A @ Z).real

print(f"\nFor |Φ⁺⟩:")
print(f"⟨Z⊗I⟩ from ρ_AB: {exp_Z_direct:.4f}")
print(f"⟨Z⟩ from ρ_A: {exp_Z_reduced:.4f}")
print(f"Match: {np.isclose(exp_Z_direct, exp_Z_reduced)}")

print("\n" + "=" * 60)
print("ENTANGLEMENT AND MIXEDNESS")
print("=" * 60)

# Parameterized state |ψ(θ)⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
import matplotlib.pyplot as plt

theta_vals = np.linspace(0, np.pi/2, 50)
purity_A = []
purity_AB = []

for theta in theta_vals:
    psi = np.cos(theta) * ket_00 + np.sin(theta) * ket_11
    rho_AB = density(psi)
    rho_A = partial_trace_B(rho_AB, 2, 2)
    purity_AB.append(np.trace(rho_AB @ rho_AB).real)
    purity_A.append(np.trace(rho_A @ rho_A).real)

plt.figure(figsize=(10, 5))
plt.plot(theta_vals * 2 / np.pi, purity_AB, 'b-', lw=2, label='Global purity Tr(ρ_AB²)')
plt.plot(theta_vals * 2 / np.pi, purity_A, 'r-', lw=2, label='Local purity Tr(ρ_A²)')
plt.axhline(0.5, color='gray', ls='--', alpha=0.5)
plt.xlabel('θ / (π/2)')
plt.ylabel('Purity')
plt.title('Entanglement Creates Local Mixedness')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('partial_trace_purity.png', dpi=150, bbox_inches='tight')
plt.show()

print("θ = 0: Product state, local purity = 1")
print("θ = π/4: Bell state, local purity = 0.5 (max mixed)")

print("\n" + "=" * 60)
print("Day 513 Complete: Partial Trace")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| ρ_A = Tr_B(ρ_AB) | Partial trace definition |
| Tr_B(\|i⟩⟨k\| ⊗ \|j⟩⟨l\|) = \|i⟩⟨k\| δ_jl | Trace of basis elements |
| ⟨A⊗I⟩_ρAB = ⟨A⟩_ρA | Expectation value rule |

### Key Insights

- Partial trace describes **local** properties of composite systems
- For **product states**: Tr_B(ρ_A ⊗ ρ_B) = ρ_A
- For **entangled pure states**: reduced state is mixed
- **Entanglement ↔ local mixedness** for pure global states

---

## Daily Checklist

- [ ] I can compute partial traces for two-qubit systems
- [ ] I understand why Bell states give maximally mixed reduced states
- [ ] I can verify expectation values using reduced density matrices
- [ ] I understand the physical meaning of partial trace

---

## Preview: Day 514

Tomorrow we'll explore **reduced density matrices** in depth, understanding how they encode all local information and connect to quantum correlations.

---

*Next: Day 514 — Reduced Density Matrices*
