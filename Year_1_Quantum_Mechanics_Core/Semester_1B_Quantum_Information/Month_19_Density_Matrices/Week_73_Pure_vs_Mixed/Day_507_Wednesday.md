# Day 507: Expectation Values and Measurement

## Overview

**Day 507** | Week 73, Day 3 | Year 1, Month 19 | Computing Observables with Density Matrices

Today we develop the complete formalism for computing measurement statistics using density matrices, including expectation values, variances, and post-measurement state updates.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Measurement theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. Derive the trace formula for expectation values from first principles
2. Calculate variances and higher moments using density matrices
3. Apply the measurement update rule (Lüders rule)
4. Compute measurement statistics for general observables
5. Distinguish selective and non-selective measurements
6. Handle repeated measurements on the same system

---

## Core Content

### Expectation Values: Complete Derivation

For a pure state |ψ⟩ with density matrix ρ = |ψ⟩⟨ψ|, the expectation value of observable A is:

$$\langle A \rangle = \langle\psi|A|\psi\rangle$$

Using the trace:
$$\text{Tr}(\rho A) = \text{Tr}(|\psi\rangle\langle\psi|A) = \sum_i \langle i|\psi\rangle\langle\psi|A|i\rangle = \langle\psi|\left(\sum_i A|i\rangle\langle i|\right)|\psi\rangle = \langle\psi|A|\psi\rangle$$

For a mixed state ρ = Σⱼ pⱼ|ψⱼ⟩⟨ψⱼ|:
$$\text{Tr}(\rho A) = \sum_j p_j \text{Tr}(|\psi_j\rangle\langle\psi_j|A) = \sum_j p_j \langle\psi_j|A|\psi_j\rangle$$

This is the **weighted average** of expectation values—exactly what we expect for a classical mixture.

$$\boxed{\langle A \rangle = \text{Tr}(\rho A)}$$

### Variance and Higher Moments

The variance of an observable A:

$$\boxed{(\Delta A)^2 = \langle A^2 \rangle - \langle A \rangle^2 = \text{Tr}(\rho A^2) - [\text{Tr}(\rho A)]^2}$$

More generally, the n-th moment:
$$\langle A^n \rangle = \text{Tr}(\rho A^n)$$

### Measurement Probabilities

For a projective measurement with orthonormal projectors {Πₘ = |m⟩⟨m|}:

$$\boxed{p(m) = \text{Tr}(\Pi_m \rho) = \langle m|\rho|m\rangle}$$

For degenerate eigenvalues with projector Πₘ onto the eigenspace:
$$p(m) = \text{Tr}(\Pi_m \rho)$$

**Verification:** Σₘ p(m) = Tr(Σₘ Πₘ ρ) = Tr(Iρ) = Tr(ρ) = 1 ✓

### Post-Measurement State Update

#### Non-Degenerate Case (Lüders Rule)

If outcome m is obtained:
$$\boxed{\rho \rightarrow \rho' = \frac{\Pi_m \rho \Pi_m}{\text{Tr}(\Pi_m \rho)} = |m\rangle\langle m|}$$

For a non-degenerate measurement, the state collapses to the eigenstate.

#### Degenerate Case

If the measurement has degenerate eigenspace with projector Πₘ:
$$\boxed{\rho \rightarrow \rho' = \frac{\Pi_m \rho \Pi_m}{\text{Tr}(\Pi_m \rho)}}$$

The state is projected onto the eigenspace but retains coherence within it.

### Selective vs Non-Selective Measurements

**Selective measurement:** We know the outcome m, state becomes ρₘ.

**Non-selective measurement:** We don't record the outcome. The final state is the average:
$$\boxed{\rho \rightarrow \rho' = \sum_m \Pi_m \rho \Pi_m}$$

This describes **decoherence** in the measurement basis.

### Example: Measurement in Computational Basis

Initial state: ρ = |+⟩⟨+| = ½(|0⟩⟨0| + |0⟩⟨1| + |1⟩⟨0| + |1⟩⟨1|)

Non-selective Z measurement:
$$\rho' = |0\rangle\langle 0|\rho|0\rangle\langle 0| + |1\rangle\langle 1|\rho|1\rangle\langle 1|$$
$$= \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{1}{2}I$$

The off-diagonal coherences are destroyed—this is **dephasing**.

### Repeated Measurements

**Key insight:** Measuring the same observable twice gives the same result.

After first measurement (outcome m): ρ' = |m⟩⟨m|
Second measurement: p(m) = Tr(Πₘρ') = Tr(|m⟩⟨m|·|m⟩⟨m|) = 1

This is **repeatability** or the **projection postulate**.

---

## Quantum Computing Connection

### Measurement in Quantum Circuits

In quantum computing, measurements are typically in the computational basis:
- Probability p(0) = ρ₀₀ = |α|² (if ρ came from |ψ⟩ = α|0⟩ + β|1⟩)
- Probability p(1) = ρ₁₁ = |β|²

### Mid-Circuit Measurements

Modern quantum computers support mid-circuit measurements:
1. Measure a subset of qubits
2. Condition subsequent operations on the outcome
3. This is essential for quantum error correction

### Measurement Statistics

In practice, we estimate ⟨A⟩ by repeated measurements:
$$\langle A \rangle \approx \frac{1}{N}\sum_{i=1}^N a_i$$

where aᵢ are individual measurement outcomes.

---

## Worked Examples

### Example 1: Complete Measurement Statistics

**Problem:** For ρ = ⅔|+⟩⟨+| + ⅓|0⟩⟨0|, find ⟨Z⟩, ⟨Z²⟩, and ΔZ.

**Solution:**

First, compute ρ:
$$\rho = \frac{2}{3} \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} + \frac{1}{3}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 2/3 & 1/3 \\ 1/3 & 1/3 \end{pmatrix}$$

For ⟨Z⟩:
$$\langle Z \rangle = \text{Tr}(\rho Z) = \text{Tr}\begin{pmatrix} 2/3 & -1/3 \\ 1/3 & -1/3 \end{pmatrix} = \frac{2}{3} - \frac{1}{3} = \frac{1}{3}$$

For ⟨Z²⟩: Since Z² = I,
$$\langle Z^2 \rangle = \text{Tr}(\rho I) = \text{Tr}(\rho) = 1$$

Variance:
$$(\Delta Z)^2 = \langle Z^2 \rangle - \langle Z \rangle^2 = 1 - \frac{1}{9} = \frac{8}{9}$$

$$\Delta Z = \frac{2\sqrt{2}}{3} \approx 0.943$$

### Example 2: State Update After Measurement

**Problem:** Starting from ρ = ½(|0⟩ + |1⟩)(⟨0| + ⟨1|) = |+⟩⟨+|, what is the state after measuring Z and getting outcome +1 (|0⟩)?

**Solution:**

Projector: Π₀ = |0⟩⟨0|

Post-measurement state:
$$\rho' = \frac{\Pi_0 \rho \Pi_0}{\text{Tr}(\Pi_0 \rho)}$$

Compute Π₀ρΠ₀:
$$\Pi_0 \rho \Pi_0 = |0\rangle\langle 0| \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \cdot |0\rangle\langle 0| = \frac{1}{2}|0\rangle\langle 0|$$

Probability:
$$\text{Tr}(\Pi_0 \rho) = \frac{1}{2}$$

Normalized state:
$$\rho' = \frac{(1/2)|0\rangle\langle 0|}{1/2} = |0\rangle\langle 0|$$

### Example 3: Non-Selective Measurement

**Problem:** Apply a non-selective X measurement to ρ = |0⟩⟨0|.

**Solution:**

X eigenstates: |+⟩ (eigenvalue +1), |−⟩ (eigenvalue -1)
Projectors: Π₊ = |+⟩⟨+|, Π₋ = |−⟩⟨−|

$$\rho' = \Pi_+ \rho \Pi_+ + \Pi_- \rho \Pi_-$$

Compute each term:
$$\Pi_+ |0\rangle\langle 0| \Pi_+ = |+\rangle\langle +|0\rangle\langle 0|+\rangle\langle +| = \frac{1}{2}|+\rangle\langle +|$$

$$\Pi_- |0\rangle\langle 0| \Pi_- = |−\rangle\langle −|0\rangle\langle 0|−\rangle\langle −| = \frac{1}{2}|−\rangle\langle −|$$

$$\rho' = \frac{1}{2}|+\rangle\langle +| + \frac{1}{2}|−\rangle\langle −| = \frac{1}{2}I$$

The state becomes maximally mixed!

---

## Practice Problems

### Direct Application

**Problem 1:** For ρ = |1⟩⟨1|, compute ⟨X⟩, ⟨Y⟩, ⟨Z⟩.

**Problem 2:** Find the post-measurement state when measuring ρ = |+⟩⟨+| in the X basis and getting +1.

**Problem 3:** Calculate ΔX for ρ = |0⟩⟨0|.

### Intermediate

**Problem 4:** Show that ⟨[A, B]⟩ = Tr(ρ[A, B]) for any density matrix ρ.

**Problem 5:** For ρ = p|0⟩⟨0| + (1-p)|1⟩⟨1|, find the value of p that minimizes ΔZ.

**Problem 6:** Verify that non-selective measurement preserves trace: Tr(Σₘ Πₘ ρ Πₘ) = 1.

### Challenging

**Problem 7:** Prove that non-selective measurement cannot increase purity.

**Problem 8:** Derive the uncertainty relation ΔA·ΔB ≥ ½|⟨[A,B]⟩| using density matrices.

**Problem 9:** Show that if [ρ, A] = 0, then ΔA = 0 for a pure state ρ.

---

## Computational Lab

```python
"""
Day 507: Expectation Values and Measurement
Computing measurement statistics with density matrices
"""

import numpy as np
import matplotlib.pyplot as plt

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Standard states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

def density_matrix(psi):
    """Create density matrix from state vector"""
    return psi @ psi.conj().T

def expectation(rho, A):
    """Compute ⟨A⟩ = Tr(ρA)"""
    return np.trace(rho @ A).real

def variance(rho, A):
    """Compute (ΔA)² = ⟨A²⟩ - ⟨A⟩²"""
    exp_A = expectation(rho, A)
    exp_A2 = expectation(rho, A @ A)
    return exp_A2 - exp_A**2

def measurement_probability(rho, projector):
    """Compute p = Tr(Πρ)"""
    return np.trace(projector @ rho).real

def post_measurement_state(rho, projector):
    """Apply Lüders rule: ρ → ΠρΠ / Tr(Πρ)"""
    prob = measurement_probability(rho, projector)
    if prob < 1e-10:
        return None  # Zero probability outcome
    return projector @ rho @ projector / prob

def non_selective_measurement(rho, projectors):
    """Apply non-selective measurement: ρ → Σ_m Π_m ρ Π_m"""
    rho_new = np.zeros_like(rho)
    for proj in projectors:
        rho_new += proj @ rho @ proj
    return rho_new

print("=" * 60)
print("EXPECTATION VALUES AND VARIANCE")
print("=" * 60)

# Example: Mixed state
rho = 2/3 * density_matrix(ket_plus) + 1/3 * density_matrix(ket_0)
print(f"\nρ = ⅔|+⟩⟨+| + ⅓|0⟩⟨0|:")
print(f"ρ = \n{rho}")

print(f"\n⟨X⟩ = {expectation(rho, X):.4f}")
print(f"⟨Y⟩ = {expectation(rho, Y):.4f}")
print(f"⟨Z⟩ = {expectation(rho, Z):.4f}")

print(f"\nΔX = {np.sqrt(variance(rho, X)):.4f}")
print(f"ΔY = {np.sqrt(variance(rho, Y)):.4f}")
print(f"ΔZ = {np.sqrt(variance(rho, Z)):.4f}")

# Verify ⟨Z²⟩ = 1 (since Z² = I)
print(f"\n⟨Z²⟩ = {expectation(rho, Z @ Z):.4f} (should be 1)")

print("\n" + "=" * 60)
print("MEASUREMENT PROBABILITIES")
print("=" * 60)

Pi_0 = density_matrix(ket_0)
Pi_1 = density_matrix(ket_1)
Pi_plus = density_matrix(ket_plus)
Pi_minus = density_matrix(ket_minus)

print(f"\nFor ρ = ⅔|+⟩⟨+| + ⅓|0⟩⟨0|:")
print(f"Z measurement: p(0) = {measurement_probability(rho, Pi_0):.4f}")
print(f"Z measurement: p(1) = {measurement_probability(rho, Pi_1):.4f}")
print(f"X measurement: p(+) = {measurement_probability(rho, Pi_plus):.4f}")
print(f"X measurement: p(-) = {measurement_probability(rho, Pi_minus):.4f}")

print("\n" + "=" * 60)
print("POST-MEASUREMENT STATE UPDATE")
print("=" * 60)

rho_plus = density_matrix(ket_plus)
print(f"\nInitial state |+⟩:")
print(f"ρ = \n{rho_plus}")

print(f"\nAfter measuring Z and getting |0⟩:")
rho_after = post_measurement_state(rho_plus, Pi_0)
print(f"ρ' = \n{rho_after}")
print(f"Purity: {np.trace(rho_after @ rho_after).real:.4f}")

print("\n" + "=" * 60)
print("NON-SELECTIVE MEASUREMENT (DECOHERENCE)")
print("=" * 60)

print(f"\nInitial state |+⟩:")
print(f"ρ = \n{rho_plus}")
print(f"Coherence ρ₀₁ = {rho_plus[0,1]:.4f}")

# Non-selective Z measurement
rho_decohered = non_selective_measurement(rho_plus, [Pi_0, Pi_1])
print(f"\nAfter non-selective Z measurement:")
print(f"ρ' = \n{rho_decohered}")
print(f"Coherence ρ'₀₁ = {rho_decohered[0,1]:.4f}")
print(f"Purity before: {np.trace(rho_plus @ rho_plus).real:.4f}")
print(f"Purity after: {np.trace(rho_decohered @ rho_decohered).real:.4f}")

# Visualization: Variance for different mixed states
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Variance vs mixing parameter
p_vals = np.linspace(0, 1, 100)
delta_Z = []
delta_X = []

for p in p_vals:
    rho_p = p * density_matrix(ket_0) + (1-p) * density_matrix(ket_1)
    delta_Z.append(np.sqrt(max(0, variance(rho_p, Z))))
    delta_X.append(np.sqrt(max(0, variance(rho_p, X))))

ax = axes[0]
ax.plot(p_vals, delta_Z, 'b-', lw=2, label='ΔZ')
ax.plot(p_vals, delta_X, 'r-', lw=2, label='ΔX')
ax.set_xlabel('Probability p (for p|0⟩⟨0| + (1-p)|1⟩⟨1|)')
ax.set_ylabel('Standard Deviation')
ax.set_title('Variance vs Mixing Parameter')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Purity change under non-selective measurement
ax = axes[1]
theta_vals = np.linspace(0, np.pi, 100)
purity_before = []
purity_after = []

for theta in theta_vals:
    psi = np.cos(theta/2) * ket_0 + np.sin(theta/2) * ket_1
    rho_psi = density_matrix(psi)
    rho_after = non_selective_measurement(rho_psi, [Pi_0, Pi_1])

    purity_before.append(np.trace(rho_psi @ rho_psi).real)
    purity_after.append(np.trace(rho_after @ rho_after).real)

ax.plot(theta_vals/np.pi, purity_before, 'b-', lw=2, label='Before measurement')
ax.plot(theta_vals/np.pi, purity_after, 'r-', lw=2, label='After non-selective Z')
ax.set_xlabel('θ/π (for state cos(θ/2)|0⟩ + sin(θ/2)|1⟩)')
ax.set_ylabel('Purity Tr(ρ²)')
ax.set_title('Purity Change Under Non-Selective Measurement')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('measurement_statistics.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Day 507 Complete: Expectation Values and Measurement")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Expectation value | ⟨A⟩ = Tr(ρA) |
| n-th moment | ⟨Aⁿ⟩ = Tr(ρAⁿ) |
| Variance | (ΔA)² = Tr(ρA²) - [Tr(ρA)]² |
| Probability | p(m) = Tr(Πₘρ) |
| Post-measurement (selective) | ρ' = ΠₘρΠₘ / Tr(Πₘρ) |
| Post-measurement (non-selective) | ρ' = Σₘ ΠₘρΠₘ |

### Key Concepts

- **Trace formula** gives expectation values as weighted averages
- **Lüders rule** describes state collapse upon measurement
- **Non-selective measurement** causes decoherence
- **Measurement destroys coherence** in the measurement basis

---

## Daily Checklist

- [ ] I can derive the trace formula for expectation values
- [ ] I can compute variances using density matrices
- [ ] I understand selective vs non-selective measurements
- [ ] I can apply the Lüders rule for state update
- [ ] I understand why non-selective measurement causes decoherence

---

## Preview: Day 508

Tomorrow we explore **purity and mixedness**, learning how to quantify the degree of "quantumness" vs "classicality" in a quantum state.

---

*Next: Day 508 — Purity and Mixedness*
