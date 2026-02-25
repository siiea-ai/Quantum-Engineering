# Day 510: Distinguishing Quantum States

## Overview

**Day 510** | Week 73, Day 6 | Year 1, Month 19 | Trace Distance and Fidelity

Today we study the fundamental measures for quantifying how "close" two quantum states are: trace distance and fidelity. These metrics are essential for quantum error analysis, state preparation verification, and quantum cryptography.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Distance measures theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. Define and compute trace distance between quantum states
2. Define and compute fidelity between quantum states
3. Prove key properties of these measures
4. Relate trace distance to measurement distinguishability
5. Use these metrics for error analysis in quantum computing
6. Apply Uhlmann's theorem for mixed state fidelity

---

## Core Content

### Trace Distance

The **trace distance** between two density matrices ρ and σ is:

$$\boxed{D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma| = \frac{1}{2}\text{Tr}\sqrt{(\rho-\sigma)^\dagger(\rho-\sigma)}}$$

where |A| = √(A†A) is the matrix absolute value.

**For diagonalizable difference:** If ρ - σ has eigenvalues {λᵢ}:
$$D(\rho, \sigma) = \frac{1}{2}\sum_i |\lambda_i|$$

### Properties of Trace Distance

1. **Metric:** D(ρ,σ) ≥ 0 with D(ρ,σ) = 0 ⟺ ρ = σ
2. **Symmetry:** D(ρ,σ) = D(σ,ρ)
3. **Triangle inequality:** D(ρ,τ) ≤ D(ρ,σ) + D(σ,τ)
4. **Bounds:** 0 ≤ D(ρ,σ) ≤ 1
5. **Unitary invariance:** D(UρU†, UσU†) = D(ρ,σ)

### Operational Interpretation

**Helstrom's Theorem:** The maximum probability of correctly distinguishing ρ from σ (given with equal prior probability) is:

$$\boxed{p_{correct}^{max} = \frac{1}{2}(1 + D(\rho, \sigma))}$$

- D = 0: States identical, p = 1/2 (random guessing)
- D = 1: States orthogonal, p = 1 (perfect distinction)

### Fidelity

The **fidelity** between ρ and σ is:

$$\boxed{F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2}$$

**Special cases:**

**Both pure states:** ρ = |ψ⟩⟨ψ|, σ = |φ⟩⟨φ|
$$F(\rho, \sigma) = |\langle\psi|\phi\rangle|^2$$

**One pure state:** σ = |φ⟩⟨φ|
$$F(\rho, |φ\rangle\langle φ|) = \langle φ|\rho|φ\rangle$$

### Properties of Fidelity

1. **Bounds:** 0 ≤ F(ρ,σ) ≤ 1
2. **Identity:** F(ρ,σ) = 1 ⟺ ρ = σ
3. **Symmetry:** F(ρ,σ) = F(σ,ρ)
4. **Unitary invariance:** F(UρU†, UσU†) = F(ρ,σ)
5. **Multiplicativity:** F(ρ₁⊗ρ₂, σ₁⊗σ₂) = F(ρ₁,σ₁)·F(ρ₂,σ₂)

### Relation Between Trace Distance and Fidelity

The Fuchs-van de Graaf inequalities:

$$\boxed{1 - \sqrt{F(\rho,\sigma)} \leq D(\rho,\sigma) \leq \sqrt{1 - F(\rho,\sigma)}}$$

**Interpretation:**
- High fidelity (F ≈ 1) implies low trace distance
- Orthogonal states (F = 0) have maximum trace distance

### Uhlmann's Theorem

For mixed states, fidelity can be computed via purifications:

$$\boxed{F(\rho, \sigma) = \max_{|\psi\rangle, |\phi\rangle} |\langle\psi|\phi\rangle|^2}$$

where the maximum is over all purifications of ρ and σ.

### Trace Distance for Qubits

For qubit states with Bloch vectors r⃗ and s⃗:

$$\boxed{D(\rho, \sigma) = \frac{1}{2}|\vec{r} - \vec{s}|}$$

The trace distance is half the Euclidean distance in the Bloch ball!

### Fidelity for Qubits

For qubit states:

$$F(\rho, \sigma) = \text{Tr}(\rho\sigma) + 2\sqrt{\det(\rho)\det(\sigma)}$$

For pure qubit states, F = |⟨ψ|φ⟩|² = cos²(θ/2), where θ is the angle between Bloch vectors.

---

## Quantum Computing Connection

### Gate Fidelity

The fidelity of a quantum gate U compared to ideal Uᵢdeal:

$$F_{gate} = \frac{1}{d^2}\left|\text{Tr}(U_{ideal}^\dagger U)\right|^2$$

For single qubits, F_gate = ½(1 + |Tr(U†U_ideal)|/2).

### State Preparation Error

If we prepare state ρ instead of target σ:
- **Infidelity:** 1 - F(ρ,σ)
- **Error rate:** Related to trace distance

### Quantum Error Correction

Error detection probability is related to trace distance between error-free and erroneous states.

### Quantum Cryptography

Security proofs often use trace distance to bound the distinguishability between real and ideal protocols.

---

## Worked Examples

### Example 1: Trace Distance for Diagonal States

**Problem:** Compute D(ρ,σ) for ρ = diag(0.8, 0.2) and σ = diag(0.6, 0.4).

**Solution:**

$$\rho - \sigma = \begin{pmatrix} 0.2 & 0 \\ 0 & -0.2 \end{pmatrix}$$

Eigenvalues: λ₁ = 0.2, λ₂ = -0.2

$$D(\rho, \sigma) = \frac{1}{2}(|0.2| + |-0.2|) = \frac{1}{2}(0.4) = 0.2$$

Maximum distinguishing probability: p = ½(1 + 0.2) = 0.6

### Example 2: Fidelity Between Pure States

**Problem:** Find F(|0⟩, |+⟩).

**Solution:**

$$F = |\langle 0|+\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

Alternatively, using Bloch vectors:
- |0⟩: r⃗ = (0, 0, 1)
- |+⟩: s⃗ = (1, 0, 0)
- Angle θ = π/2, so F = cos²(π/4) = ½ ✓

### Example 3: Mixed State Fidelity

**Problem:** Compute F(ρ, I/2) where ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1|.

**Solution:**

For one state maximally mixed:
$$F(\rho, I/2) = \text{Tr}(\rho \cdot I/2) + 2\sqrt{\det(\rho)\det(I/2)}$$

$$= \frac{1}{2} + 2\sqrt{0.75 \times 0.25 \times 0.25}$$
$$= \frac{1}{2} + 2\sqrt{0.046875} = \frac{1}{2} + 0.433 = 0.933$$

---

## Practice Problems

### Direct Application

**Problem 1:** Compute D(|0⟩, |1⟩) and verify D = 1 for orthogonal states.

**Problem 2:** Find F(|+⟩, |−⟩).

**Problem 3:** For ρ with Bloch vector (0.3, 0, 0.4) and σ with (0.1, 0.2, 0.4), compute D(ρ,σ).

### Intermediate

**Problem 4:** Prove that D(ρ,σ) = 0 implies ρ = σ.

**Problem 5:** Show F(ρ,σ) is symmetric: F(ρ,σ) = F(σ,ρ).

**Problem 6:** Verify the Fuchs-van de Graaf bounds for |0⟩ and |+⟩.

### Challenging

**Problem 7:** Prove that trace distance is unitarily invariant.

**Problem 8:** Derive Helstrom's theorem for optimal state discrimination.

**Problem 9:** Show that fidelity is multiplicative under tensor products.

---

## Computational Lab

```python
"""
Day 510: Distinguishing Quantum States
Trace distance and fidelity calculations
"""

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

def trace_distance(rho, sigma):
    """Compute trace distance D(ρ,σ) = ½Tr|ρ-σ|"""
    diff = rho - sigma
    # |A| = sqrt(A†A)
    abs_diff = sqrtm(diff.conj().T @ diff)
    return 0.5 * np.trace(abs_diff).real

def fidelity(rho, sigma):
    """Compute fidelity F(ρ,σ) = (Tr√(√ρ σ √ρ))²"""
    sqrt_rho = sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = sqrtm(inner)
    return (np.trace(sqrt_inner).real)**2

def fidelity_pure(psi, phi):
    """Fidelity between pure states: |⟨ψ|φ⟩|²"""
    return np.abs(psi.conj().T @ phi)**2

def trace_distance_bloch(r1, r2):
    """Trace distance for qubits using Bloch vectors"""
    return 0.5 * np.linalg.norm(np.array(r1) - np.array(r2))

# Standard states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

def density(psi):
    return psi @ psi.conj().T

print("=" * 60)
print("TRACE DISTANCE CALCULATIONS")
print("=" * 60)

# Example 1: Orthogonal states
rho_0 = density(ket_0)
rho_1 = density(ket_1)
D_01 = trace_distance(rho_0, rho_1)
print(f"\nD(|0⟩, |1⟩) = {D_01:.4f}")
print(f"Max distinguish probability: {0.5*(1 + D_01):.4f}")

# Example 2: |0⟩ and |+⟩
rho_plus = density(ket_plus)
D_0plus = trace_distance(rho_0, rho_plus)
print(f"\nD(|0⟩, |+⟩) = {D_0plus:.4f}")
print(f"Max distinguish probability: {0.5*(1 + D_0plus):.4f}")

# Example 3: Mixed states
rho_mixed1 = np.array([[0.8, 0], [0, 0.2]], dtype=complex)
rho_mixed2 = np.array([[0.6, 0], [0, 0.4]], dtype=complex)
D_mixed = trace_distance(rho_mixed1, rho_mixed2)
print(f"\nD(diag(0.8,0.2), diag(0.6,0.4)) = {D_mixed:.4f}")

# Verify using Bloch vectors
r1 = np.array([0, 0, 0.6])  # z = 0.8 - 0.2 = 0.6
r2 = np.array([0, 0, 0.2])  # z = 0.6 - 0.4 = 0.2
D_bloch = trace_distance_bloch(r1, r2)
print(f"Using Bloch vectors: D = {D_bloch:.4f}")

print("\n" + "=" * 60)
print("FIDELITY CALCULATIONS")
print("=" * 60)

# Pure state fidelities
F_01 = fidelity(rho_0, rho_1)
F_0plus = fidelity(rho_0, rho_plus)
F_plusminus = fidelity(rho_plus, density(ket_minus))

print(f"\nF(|0⟩, |1⟩) = {F_01:.4f} (orthogonal)")
print(f"F(|0⟩, |+⟩) = {F_0plus:.4f}")
print(f"F(|+⟩, |−⟩) = {F_plusminus:.4f} (orthogonal)")

# Verify F = |⟨ψ|φ⟩|² for pure states
F_direct = fidelity_pure(ket_0, ket_plus)[0, 0]
print(f"\nDirect |⟨0|+⟩|² = {F_direct:.4f}")

# Mixed state fidelity
F_mixed = fidelity(rho_mixed1, rho_mixed2)
print(f"\nF(diag(0.8,0.2), diag(0.6,0.4)) = {F_mixed:.4f}")

print("\n" + "=" * 60)
print("FUCHS-VAN DE GRAAF BOUNDS")
print("=" * 60)

# Verify bounds for several state pairs
test_pairs = [
    (rho_0, rho_plus, "|0⟩ vs |+⟩"),
    (rho_mixed1, rho_mixed2, "mixed1 vs mixed2"),
    (rho_0, 0.5*np.eye(2, dtype=complex), "|0⟩ vs I/2"),
]

for rho, sigma, name in test_pairs:
    D = trace_distance(rho, sigma)
    F = fidelity(rho, sigma)
    lower = 1 - np.sqrt(F)
    upper = np.sqrt(1 - F)

    print(f"\n{name}:")
    print(f"  D = {D:.4f}, F = {F:.4f}")
    print(f"  Bounds: {lower:.4f} ≤ D ≤ {upper:.4f}")
    print(f"  Satisfied: {lower - 0.001 <= D <= upper + 0.001}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Trace distance as function of state overlap
ax1 = axes[0]
theta_vals = np.linspace(0, np.pi, 100)
D_vals = []
F_vals = []

for theta in theta_vals:
    psi = np.cos(theta/2) * ket_0 + np.sin(theta/2) * ket_1
    rho_psi = density(psi)
    D_vals.append(trace_distance(rho_0, rho_psi))
    F_vals.append(fidelity(rho_0, rho_psi))

ax1.plot(theta_vals/np.pi, D_vals, 'b-', lw=2, label='Trace distance D')
ax1.plot(theta_vals/np.pi, F_vals, 'r-', lw=2, label='Fidelity F')
ax1.set_xlabel('θ/π (angle from |0⟩)')
ax1.set_ylabel('Value')
ax1.set_title('D and F vs State Angle')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Distinguishing probability
ax2 = axes[1]
p_correct = [0.5 * (1 + d) for d in D_vals]
ax2.plot(theta_vals/np.pi, p_correct, 'g-', lw=2)
ax2.axhline(0.5, color='gray', ls='--', label='Random guessing')
ax2.axhline(1.0, color='red', ls='--', label='Perfect distinction')
ax2.fill_between(theta_vals/np.pi, 0.5, p_correct, alpha=0.3, color='green')
ax2.set_xlabel('θ/π')
ax2.set_ylabel('p_correct^max')
ax2.set_title('Maximum Distinguishing Probability')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Fuchs-van de Graaf bounds
ax3 = axes[2]
F_range = np.linspace(0.001, 0.999, 100)
D_lower = 1 - np.sqrt(F_range)
D_upper = np.sqrt(1 - F_range)

ax3.fill_between(F_range, D_lower, D_upper, alpha=0.3, color='blue',
                 label='Allowed region')
ax3.plot(F_range, D_lower, 'b--', lw=1)
ax3.plot(F_range, D_upper, 'b--', lw=1)

# Plot actual (F, D) points
for rho, sigma, name in test_pairs:
    D = trace_distance(rho, sigma)
    F = fidelity(rho, sigma)
    ax3.scatter([F], [D], s=100, zorder=5, label=name)

ax3.set_xlabel('Fidelity F')
ax3.set_ylabel('Trace Distance D')
ax3.set_title('Fuchs-van de Graaf Bounds')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('state_distinguishability.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Day 510 Complete: Distinguishing Quantum States")
print("=" * 60)
```

---

## Summary

### Key Distance Measures

| Measure | Formula | Range | Identity |
|---------|---------|-------|----------|
| Trace distance | D = ½Tr\|ρ-σ\| | [0, 1] | D=0 ⟺ ρ=σ |
| Fidelity | F = (Tr√(√ρσ√ρ))² | [0, 1] | F=1 ⟺ ρ=σ |

### Key Relationships

| Relationship | Formula |
|--------------|---------|
| Fuchs-van de Graaf | 1-√F ≤ D ≤ √(1-F) |
| Distinguishing probability | p_max = ½(1+D) |
| Qubit trace distance | D = ½\|r⃗-s⃗\| |
| Pure state fidelity | F = \|⟨ψ\|φ⟩\|² |

---

## Daily Checklist

- [ ] I can compute trace distance between quantum states
- [ ] I can compute fidelity between quantum states
- [ ] I understand the operational meaning of trace distance
- [ ] I can use Fuchs-van de Graaf bounds
- [ ] I can apply these measures to error analysis

---

## Preview: Day 511

Tomorrow we'll consolidate all Week 73 material in a **comprehensive review**, integrating density matrices, properties, purity, Bloch representation, and distinguishability measures.

---

*Next: Day 511 — Week Review*
