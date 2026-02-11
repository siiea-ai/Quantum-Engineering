# Day 517: Purification

## Overview

**Day 517** | Week 74, Day 6 | Year 1, Month 19 | Mixed States as Partial Views

Today we study purification—the profound insight that every mixed state can be viewed as a reduced state of a pure state on a larger system. This connects mixed states to entanglement and provides a powerful conceptual tool.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Purification theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. State and prove the purification theorem
2. Construct purifications for given mixed states
3. Understand the non-uniqueness of purification
4. Apply the Uhlmann theorem for fidelity
5. Connect purification to the "Church of the Larger Hilbert Space"
6. Use purification in quantum information proofs

---

## Core Content

### The Purification Theorem

**Theorem:** For any density matrix ρ_A on ℋ_A, there exists a pure state |ψ⟩_AR on ℋ_A ⊗ ℋ_R such that:

$$\boxed{\rho_A = \text{Tr}_R(|\psi\rangle_{AR}\langle\psi|)}$$

The ancillary system R is called the **reference** or **purifying** system.

### Construction

If ρ_A = Σᵢ λᵢ|eᵢ⟩⟨eᵢ| (spectral decomposition), then:

$$\boxed{|\psi\rangle_{AR} = \sum_i \sqrt{\lambda_i} |e_i\rangle_A |i\rangle_R}$$

is a purification, where {|i⟩_R} is any orthonormal basis for ℋ_R.

### Verification

$$\text{Tr}_R(|\psi\rangle\langle\psi|) = \sum_{ij} \sqrt{\lambda_i \lambda_j} |e_i\rangle\langle e_j| \cdot \langle j|i\rangle = \sum_i \lambda_i |e_i\rangle\langle e_i| = \rho_A$$

### Minimum Reference Dimension

The reference system needs dimension at least rank(ρ_A).

For a qubit mixed state: dim(ℋ_R) = 2 suffices.

### Non-Uniqueness of Purification

Purifications are **not unique**. If |ψ⟩_AR is a purification of ρ_A, then so is:

$$(I_A \otimes U_R)|\psi\rangle_{AR}$$

for any unitary U_R on the reference.

**Key insight:** All purifications are related by local unitaries on R.

### The Church of the Larger Hilbert Space

**Philosophy:** Any mixed state can be "upgraded" to a pure state by including a reference system.

This viewpoint simplifies many proofs by allowing us to work with pure states only.

### Uhlmann's Theorem

The **fidelity** between mixed states can be computed via purifications:

$$\boxed{F(\rho, \sigma) = \max_{|\psi\rangle, |\phi\rangle} |\langle\psi|\phi\rangle|^2}$$

where the maximum is over all purifications |ψ⟩ of ρ and |φ⟩ of σ.

### Physical Interpretation

1. **Decoherence model:** System A interacts with environment R, creating entanglement. Tracing out R gives the mixed state.

2. **Quantum channels:** Every channel can be modeled as: attach ancilla → unitary → partial trace.

3. **Information perspective:** The "missing information" in ρ_A is stored in correlations with R.

---

## Quantum Computing Connection

### Error Models

Quantum errors can be modeled as:
1. System becomes entangled with environment
2. Environment traced out
3. System state becomes mixed

### Quantum Channels as Purifications

The Stinespring dilation theorem: every quantum channel can be written as:
$$\mathcal{E}(\rho) = \text{Tr}_E(U(\rho \otimes |0\rangle\langle 0|_E)U^\dagger)$$

### Reference Frame Independence

Purifications show that mixed states have intrinsic properties independent of how we "purify" them.

---

## Worked Examples

### Example 1: Purifying a Qubit Mixed State

**Problem:** Find a purification of ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1|.

**Solution:**

Spectral decomposition: ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1|

Purification:
$$|\psi\rangle_{AR} = \sqrt{3/4}|0\rangle_A|0\rangle_R + \sqrt{1/4}|1\rangle_A|1\rangle_R$$
$$= \frac{\sqrt{3}}{2}|00\rangle + \frac{1}{2}|11\rangle$$

Verify: Tr_R(|ψ⟩⟨ψ|) = ¾|0⟩⟨0| + ¼|1⟩⟨1| = ρ ✓

### Example 2: Purifying Maximally Mixed State

**Problem:** Find a purification of ρ = I/2.

**Solution:**

$$\rho = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|$$

Purification:
$$|\psi\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|11\rangle = |\Phi^+\rangle$$

The maximally mixed state is the reduced state of a **maximally entangled** state!

### Example 3: Non-Uniqueness

**Problem:** Find two different purifications of ρ = I/2.

**Solution:**

Purification 1: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
Purification 2: |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2

Both give Tr_R(·) = I/2.

They're related by U_R = X on the reference system.

---

## Practice Problems

### Direct Application

**Problem 1:** Find a purification of ρ = ⅔|+⟩⟨+| + ⅓|−⟩⟨−|.

**Problem 2:** What is the minimum dimension of the reference system to purify a rank-3 qutrit state?

**Problem 3:** Show that |Φ⁻⟩ = (|00⟩ - |11⟩)/√2 also purifies I/2.

### Intermediate

**Problem 4:** Prove that Tr_R(|ψ⟩⟨ψ|) gives a valid density matrix.

**Problem 5:** Show that all purifications of a rank-1 state are product states.

**Problem 6:** For ρ = p|0⟩⟨0| + (1-p)|+⟩⟨+|, find a purification.

### Challenging

**Problem 7:** Prove Uhlmann's theorem: F(ρ,σ) = max over purifications of |⟨ψ|φ⟩|².

**Problem 8:** Show that dim(ℋ_R) ≥ rank(ρ) is necessary for purification.

**Problem 9:** Given purification |ψ⟩_AR, characterize all other purifications of the same ρ_A.

---

## Computational Lab

```python
"""
Day 517: Purification
Constructing and analyzing purifications
"""

import numpy as np
import matplotlib.pyplot as plt

def purify(rho):
    """
    Construct a purification of density matrix rho.
    Returns the purification state vector.
    """
    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # Filter positive eigenvalues
    d = len(eigenvalues)
    rank = np.sum(eigenvalues > 1e-10)

    # Construct purification
    # |ψ⟩ = Σ √λᵢ |eᵢ⟩_A |i⟩_R
    psi = np.zeros(d * d, dtype=complex)

    for i in range(d):
        if eigenvalues[i] > 1e-10:
            for j in range(d):
                # |eᵢ⟩_A component j, |i⟩_R component i
                idx = j * d + i  # Index in tensor product
                psi[idx] = np.sqrt(eigenvalues[i]) * eigenvectors[j, i]

    return psi

def partial_trace_R(psi_AR, dim_A, dim_R):
    """Trace out reference system R"""
    rho_AR = np.outer(psi_AR, psi_AR.conj())
    rho_AR = rho_AR.reshape(dim_A, dim_R, dim_A, dim_R)
    return np.trace(rho_AR, axis1=1, axis2=3)

def density(psi):
    return np.outer(psi, psi.conj())

print("=" * 60)
print("PURIFICATION EXAMPLES")
print("=" * 60)

# Example 1: Simple mixed state
print("\n--- Example 1: ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1| ---")
rho1 = np.array([[0.75, 0], [0, 0.25]], dtype=complex)
print(f"Original ρ:\n{rho1}")

psi1 = purify(rho1)
print(f"\nPurification |ψ⟩ = {psi1}")

rho1_recovered = partial_trace_R(psi1, 2, 2)
print(f"\nTr_R(|ψ⟩⟨ψ|):\n{rho1_recovered}")
print(f"Match: {np.allclose(rho1, rho1_recovered)}")

# Example 2: Maximally mixed state
print("\n--- Example 2: ρ = I/2 (maximally mixed) ---")
rho2 = np.eye(2, dtype=complex) / 2
psi2 = purify(rho2)
print(f"Purification |ψ⟩ = {psi2}")
print("This is the Bell state |Φ⁺⟩!")

rho2_recovered = partial_trace_R(psi2, 2, 2)
print(f"\nTr_R(|ψ⟩⟨ψ|) = I/2: {np.allclose(rho2, rho2_recovered)}")

# Example 3: Non-uniqueness
print("\n--- Example 3: Non-uniqueness of purification ---")

# Original purification
psi_orig = psi2.copy()

# Apply unitary on R (X gate)
X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)
I_X = np.kron(I, X)
psi_new = I_X @ psi_orig

print(f"Original purification: {psi_orig}")
print(f"After I⊗X: {psi_new}")

# Both should give I/2
rho_from_orig = partial_trace_R(psi_orig, 2, 2)
rho_from_new = partial_trace_R(psi_new, 2, 2)

print(f"\nBoth give I/2:")
print(f"From original: {np.allclose(rho_from_orig, rho2)}")
print(f"From new: {np.allclose(rho_from_new, rho2)}")

# Example 4: Varying purity
print("\n--- Example 4: Purification vs Purity ---")

p_vals = np.linspace(0.5, 1, 50)
entanglement_entropy = []

for p in p_vals:
    # Mixed state with varying purity
    rho = np.array([[p, 0], [0, 1-p]], dtype=complex)
    psi = purify(rho)

    # Entanglement of the purification
    rho_A = partial_trace_R(psi, 2, 2)
    evals = np.linalg.eigvalsh(rho_A)
    evals = evals[evals > 1e-10]
    S = -np.sum(evals * np.log2(evals))
    entanglement_entropy.append(S)

plt.figure(figsize=(10, 5))
plt.plot(p_vals, entanglement_entropy, 'b-', lw=2)
plt.xlabel('Largest eigenvalue p (purity increases →)')
plt.ylabel('Entanglement entropy of purification (bits)')
plt.title('More Mixed States Require More Entanglement to Purify')
plt.grid(True, alpha=0.3)
plt.savefig('purification_entanglement.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nKey insight: Maximally mixed states require maximally entangled purifications!")

print("\n" + "=" * 60)
print("Day 517 Complete: Purification")
print("=" * 60)
```

---

## Summary

### Purification Theorem

$$|\psi\rangle_{AR} = \sum_i \sqrt{\lambda_i} |e_i\rangle_A |i\rangle_R$$

purifies ρ_A = Σᵢ λᵢ|eᵢ⟩⟨eᵢ|.

### Key Properties

| Property | Description |
|----------|-------------|
| Existence | Every mixed state has a purification |
| Non-uniqueness | Related by unitaries on reference |
| Min dimension | dim(ℋ_R) ≥ rank(ρ) |
| Max mixed | Purifies to max entangled state |

### Uhlmann's Theorem

$$F(\rho, \sigma) = \max_{purifications} |\langle\psi|\phi\rangle|^2$$

---

## Daily Checklist

- [ ] I can construct purifications from spectral decomposition
- [ ] I understand purification non-uniqueness
- [ ] I can verify purifications by partial trace
- [ ] I understand the connection between mixedness and entanglement

---

## Preview: Day 518

Tomorrow we'll review all of Week 74, integrating tensor products, partial trace, Schmidt decomposition, entanglement detection, and purification.

---

*Next: Day 518 — Week Review*
