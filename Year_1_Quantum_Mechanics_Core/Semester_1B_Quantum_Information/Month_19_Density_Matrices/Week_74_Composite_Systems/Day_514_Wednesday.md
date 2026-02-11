# Day 514: Reduced Density Matrices

## Overview

**Day 514** | Week 74, Day 3 | Year 1, Month 19 | Local Descriptions of Quantum Systems

Today we deepen our understanding of reduced density matrices—how they capture all locally accessible information and enable us to make predictions about subsystem measurements.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Reduced density matrix theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. Explain why reduced density matrices contain all local information
2. Compute expectation values for local observables
3. Calculate measurement probabilities for subsystem measurements
4. Understand quantum correlations through marginal distributions
5. Distinguish classical from quantum correlations
6. Apply reduced density matrices to multi-qubit systems

---

## Core Content

### The Completeness of Reduced Density Matrices

**Theorem:** For any observable A_A acting only on subsystem A:
$$\langle A_A \rangle_{\rho_{AB}} = \langle A_A \rangle_{\rho_A}$$

where ρ_A = Tr_B(ρ_AB).

**Proof:**
$$\langle A_A \rangle = \text{Tr}_{AB}(\rho_{AB}(A_A \otimes I_B)) = \text{Tr}_A(A_A \cdot \text{Tr}_B(\rho_{AB})) = \text{Tr}_A(\rho_A A_A)$$

**Implication:** If we only measure system A, ρ_A tells us everything.

### Local Measurement Probabilities

For a measurement on A with projectors {Π_m^A}:
$$p(m) = \text{Tr}_{AB}(\rho_{AB}(\Pi_m^A \otimes I_B)) = \text{Tr}_A(\rho_A \Pi_m^A)$$

The reduced density matrix gives correct probabilities for local measurements.

### Marginal Distributions

Consider measuring both A and B in their respective bases.

**Joint distribution:**
$$p(a, b) = \text{Tr}(\rho_{AB}(\Pi_a^A \otimes \Pi_b^B))$$

**Marginal distributions:**
$$p(a) = \sum_b p(a,b) = \text{Tr}_A(\rho_A \Pi_a^A)$$
$$p(b) = \sum_a p(a,b) = \text{Tr}_B(\rho_B \Pi_b^B)$$

**Key insight:** Marginals come from reduced density matrices!

### Classical vs Quantum Correlations

For product states ρ_AB = ρ_A ⊗ ρ_B:
$$p(a,b) = p(a) \cdot p(b)$$

Outcomes are **independent** (no correlations).

For entangled states like |Φ⁺⟩:
$$p(0_A, 0_B) = \frac{1}{2}, \quad p(1_A, 1_B) = \frac{1}{2}$$
$$p(0_A, 1_B) = 0, \quad p(1_A, 0_B) = 0$$

But marginals: p(0_A) = p(1_A) = ½, p(0_B) = p(1_B) = ½

**Perfect correlations** despite uniform marginals!

### Quantum Discord and Correlations

Total correlations in ρ_AB can be quantified by **mutual information**:
$$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$$

where S is von Neumann entropy.

For product states: I(A:B) = 0
For Bell states: I(A:B) = 2 (maximal for qubits)

### Properties of Reduced States

1. **Normalization:** Tr(ρ_A) = 1
2. **Positivity:** ρ_A ≥ 0
3. **Purity bound:** Tr(ρ_A²) ≤ Tr(ρ_AB²)
4. **Entropy bound:** S(ρ_A) ≥ S(ρ_AB) for pure global states

---

## Quantum Computing Connection

### Single-Qubit Tomography in Multi-Qubit Systems

To characterize qubit j in an n-qubit system:
1. Compute ρ_j = Tr_{all except j}(ρ)
2. Measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on qubit j
3. Reconstruct ρ_j

### Error Propagation

Local errors on qubit j only affect ρ_j directly but can spread through entanglement.

### Quantum Channel on Subsystems

If channel E acts on A: ρ_A' = E(ρ_A)

But global state: ρ_AB' = (E ⊗ I)(ρ_AB) may have different correlations.

---

## Worked Examples

### Example 1: Correlations in Bell State

**Problem:** For |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, show that Z measurements are perfectly correlated.

**Solution:**

Joint probabilities:
$$p(0,0) = |\langle 00|\Phi^+\rangle|^2 = \frac{1}{2}$$
$$p(0,1) = |\langle 01|\Phi^+\rangle|^2 = 0$$
$$p(1,0) = |\langle 10|\Phi^+\rangle|^2 = 0$$
$$p(1,1) = |\langle 11|\Phi^+\rangle|^2 = \frac{1}{2}$$

Marginals: p(0_A) = ½, p(1_A) = ½, etc.

**Correlation:** p(same) = 1, p(different) = 0
**Yet:** Each qubit individually is maximally random!

### Example 2: Local State from Three-Qubit System

**Problem:** For |GHZ⟩ = (|000⟩ + |111⟩)/√2, find ρ₁ (first qubit).

**Solution:**

Trace out qubits 2 and 3:
$$\rho_{123} = \frac{1}{2}(|000\rangle\langle 000| + |000\rangle\langle 111| + |111\rangle\langle 000| + |111\rangle\langle 111|)$$

$$\rho_1 = \text{Tr}_{23}(\rho_{123}) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

Each qubit is maximally mixed, despite global pure state.

### Example 3: Mutual Information

**Problem:** Calculate I(A:B) for |Φ⁺⟩.

**Solution:**

- S(ρ_AB) = 0 (pure state)
- S(ρ_A) = S(I/2) = 1 bit
- S(ρ_B) = S(I/2) = 1 bit

$$I(A:B) = 1 + 1 - 0 = 2 \text{ bits}$$

Maximum possible for two qubits!

---

## Practice Problems

### Direct Application

**Problem 1:** For |ψ⟩ = (|00⟩ + |01⟩ + |10⟩)/√3, compute ρ_A and ρ_B.

**Problem 2:** Show that for |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2, X measurements are also perfectly correlated.

**Problem 3:** For a product state |+⟩ ⊗ |0⟩, verify that I(A:B) = 0.

### Intermediate

**Problem 4:** Prove that for any bipartite pure state, S(ρ_A) = S(ρ_B).

**Problem 5:** For ρ_AB = p|Φ⁺⟩⟨Φ⁺| + (1-p)I/4, find ρ_A.

**Problem 6:** Calculate the mutual information for the Werner state in Problem 5.

### Challenging

**Problem 7:** Show that separable states satisfy I(A:B) ≤ min(S(ρ_A), S(ρ_B)).

**Problem 8:** For the W state |W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3, compute ρ₁₂ (two-qubit reduced state).

**Problem 9:** Prove the Araki-Lieb inequality: |S(ρ_A) - S(ρ_B)| ≤ S(ρ_AB).

---

## Computational Lab

```python
"""
Day 514: Reduced Density Matrices
Local descriptions and correlations
"""

import numpy as np
import matplotlib.pyplot as plt

def partial_trace_B(rho_AB, dim_A, dim_B):
    """Partial trace over system B"""
    rho = np.array(rho_AB).reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho, axis1=1, axis2=3)

def partial_trace_A(rho_AB, dim_A, dim_B):
    """Partial trace over system A"""
    rho = np.array(rho_AB).reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho, axis1=0, axis2=2)

def von_neumann_entropy(rho):
    """Compute S(ρ) = -Tr(ρ log₂ ρ)"""
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-10]
    return -np.sum(evals * np.log2(evals))

def mutual_information(rho_AB, dim_A, dim_B):
    """Compute I(A:B) = S(A) + S(B) - S(AB)"""
    rho_A = partial_trace_B(rho_AB, dim_A, dim_B)
    rho_B = partial_trace_A(rho_AB, dim_A, dim_B)
    return von_neumann_entropy(rho_A) + von_neumann_entropy(rho_B) - von_neumann_entropy(rho_AB)

# States
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

ket_00 = np.kron(ket_0, ket_0)
ket_01 = np.kron(ket_0, ket_1)
ket_10 = np.kron(ket_1, ket_0)
ket_11 = np.kron(ket_1, ket_1)

phi_plus = (ket_00 + ket_11) / np.sqrt(2)
psi_plus = (ket_01 + ket_10) / np.sqrt(2)

def density(psi):
    return np.outer(psi, psi.conj())

print("=" * 60)
print("CORRELATIONS IN BELL STATES")
print("=" * 60)

rho_phi = density(phi_plus)

# Joint probabilities for Z measurement
proj_00 = density(ket_00)
proj_01 = density(ket_01)
proj_10 = density(ket_10)
proj_11 = density(ket_11)

p_00 = np.trace(rho_phi @ proj_00).real
p_01 = np.trace(rho_phi @ proj_01).real
p_10 = np.trace(rho_phi @ proj_10).real
p_11 = np.trace(rho_phi @ proj_11).real

print(f"\n|Φ⁺⟩ Z-measurement joint probabilities:")
print(f"p(0,0) = {p_00:.4f}")
print(f"p(0,1) = {p_01:.4f}")
print(f"p(1,0) = {p_10:.4f}")
print(f"p(1,1) = {p_11:.4f}")

# Marginals
p_0A = p_00 + p_01
p_1A = p_10 + p_11
p_0B = p_00 + p_10
p_1B = p_01 + p_11

print(f"\nMarginals:")
print(f"p(0_A) = {p_0A:.4f}, p(1_A) = {p_1A:.4f}")
print(f"p(0_B) = {p_0B:.4f}, p(1_B) = {p_1B:.4f}")

# Verify from reduced density matrices
rho_A = partial_trace_B(rho_phi, 2, 2)
rho_B = partial_trace_A(rho_phi, 2, 2)

p_0A_from_rho = np.trace(rho_A @ density(ket_0)).real
print(f"\nFrom ρ_A: p(0_A) = {p_0A_from_rho:.4f}")

print("\n" + "=" * 60)
print("MUTUAL INFORMATION")
print("=" * 60)

# Bell state
I_phi = mutual_information(rho_phi, 2, 2)
print(f"\n|Φ⁺⟩: I(A:B) = {I_phi:.4f} bits")

# Product state
rho_product = np.kron(density(ket_0 + ket_1)/2, density(ket_0))
I_product = mutual_information(rho_product, 2, 2)
print(f"|+⟩⊗|0⟩: I(A:B) = {I_product:.4f} bits")

# Werner state
p = 0.7
rho_werner = p * rho_phi + (1-p) * np.eye(4)/4
I_werner = mutual_information(rho_werner, 2, 2)
print(f"Werner (p=0.7): I(A:B) = {I_werner:.4f} bits")

# Visualization: Mutual information vs entanglement
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Correlation diagram
ax = axes[0]
ax.bar(['p(0,0)', 'p(0,1)', 'p(1,0)', 'p(1,1)'], [p_00, p_01, p_10, p_11])
ax.set_ylabel('Probability')
ax.set_title('|Φ⁺⟩: Joint Z-measurement')
ax.set_ylim(0, 0.6)

# Plot 2: Mutual information vs mixing
ax = axes[1]
p_vals = np.linspace(0, 1, 50)
I_vals = []
S_A_vals = []

for p in p_vals:
    rho = p * rho_phi + (1-p) * np.eye(4)/4
    I_vals.append(mutual_information(rho, 2, 2))
    rho_A = partial_trace_B(rho, 2, 2)
    S_A_vals.append(von_neumann_entropy(rho_A))

ax.plot(p_vals, I_vals, 'b-', lw=2, label='I(A:B)')
ax.plot(p_vals, S_A_vals, 'r--', lw=2, label='S(ρ_A)')
ax.set_xlabel('p (Bell state weight)')
ax.set_ylabel('Bits')
ax.set_title('Werner State Correlations')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reduced_density_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Day 514 Complete: Reduced Density Matrices")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| ⟨A⊗I⟩_ρAB = ⟨A⟩_ρA | Local observables |
| p(a) = Tr(ρ_A Π_a) | Local measurement probabilities |
| I(A:B) = S(A) + S(B) - S(AB) | Mutual information |

### Key Insights

- Reduced density matrices contain **all local information**
- Entangled states can have **strong correlations** with **uniform marginals**
- Mutual information quantifies **total correlations**
- For pure states: S(ρ_A) = S(ρ_B) (Schmidt decomposition theorem preview)

---

## Daily Checklist

- [ ] I can compute expectation values from reduced states
- [ ] I understand quantum correlations vs marginal distributions
- [ ] I can calculate mutual information
- [ ] I understand the difference between classical and quantum correlations

---

## Preview: Day 515

Tomorrow we introduce the **Schmidt decomposition**—the fundamental theorem characterizing bipartite pure states and their entanglement.

---

*Next: Day 515 — Schmidt Decomposition*
