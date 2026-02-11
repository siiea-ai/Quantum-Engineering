# Day 515: Schmidt Decomposition

## Overview

**Day 515** | Week 74, Day 4 | Year 1, Month 19 | The Structure of Bipartite Pure States

Today we study the Schmidt decomposition—the fundamental theorem that reveals the structure of bipartite pure states and provides a powerful tool for quantifying entanglement.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Schmidt decomposition theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. State and prove the Schmidt decomposition theorem
2. Compute Schmidt coefficients for bipartite states
3. Determine Schmidt rank and its significance
4. Connect Schmidt decomposition to singular value decomposition
5. Use Schmidt coefficients to quantify entanglement
6. Prove that reduced density matrices have equal spectra

---

## Core Content

### The Schmidt Decomposition Theorem

**Theorem:** Any bipartite pure state |ψ⟩_AB can be written as:

$$\boxed{|\psi\rangle_{AB} = \sum_{i=1}^{r} \sqrt{\lambda_i} |a_i\rangle_A |b_i\rangle_B}$$

where:
- {|aᵢ⟩_A} are orthonormal in ℋ_A
- {|bᵢ⟩_B} are orthonormal in ℋ_B
- λᵢ > 0 are **Schmidt coefficients** with Σλᵢ = 1
- r ≤ min(d_A, d_B) is the **Schmidt rank**

### Proof via SVD

Write |ψ⟩ in product basis:
$$|\psi\rangle = \sum_{ij} c_{ij} |i\rangle_A |j\rangle_B$$

The coefficient matrix C = (c_ij) can be decomposed using SVD:
$$C = U \Sigma V^\dagger$$

where U, V are unitary and Σ = diag(√λ₁, √λ₂, ...).

Define:
- |aᵢ⟩ = Σⱼ U_ji |j⟩
- |bᵢ⟩ = Σⱼ V_ji* |j⟩

Then |ψ⟩ = Σᵢ √λᵢ |aᵢ⟩|bᵢ⟩.

### Schmidt Coefficients and Reduced States

The reduced density matrices have the **same non-zero eigenvalues**:

$$\rho_A = \sum_i \lambda_i |a_i\rangle\langle a_i|$$
$$\rho_B = \sum_i \lambda_i |b_i\rangle\langle b_i|$$

**Key result:** Tr(ρ_A²) = Tr(ρ_B²) = Σᵢ λᵢ²

### Schmidt Rank

The **Schmidt rank** r equals:
- Number of non-zero Schmidt coefficients
- Rank of reduced density matrices
- Minimum dimension needed to represent the state

**Entanglement criterion:**
- r = 1: Product state (not entangled)
- r > 1: Entangled state

### Examples

**Product state |+⟩|0⟩:**
$$|\psi\rangle = |+\rangle|0\rangle$$
Schmidt rank = 1, coefficients: {1}

**Bell state |Φ⁺⟩:**
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}|0\rangle|0\rangle + \frac{1}{\sqrt{2}}|1\rangle|1\rangle$$
Schmidt rank = 2, coefficients: {½, ½}

**Maximally entangled:** All λᵢ = 1/d (for d-dimensional systems)

### Entanglement Measures from Schmidt Decomposition

**Entanglement entropy:**
$$E(\psi) = S(\rho_A) = -\sum_i \lambda_i \log_2 \lambda_i$$

**Schmidt number:** r (the rank)

**Purity:** γ = Σᵢ λᵢ² (lower = more entangled for pure states)

---

## Quantum Computing Connection

### Entanglement in Quantum Algorithms

The power of quantum computing partly comes from entanglement. Schmidt decomposition helps analyze:
- How much entanglement is generated
- Which states can be efficiently simulated classically

### Classical Simulation Limits

States with low Schmidt rank can be efficiently simulated classically. The Schmidt rank grows exponentially for generic quantum evolution.

### Tensor Networks

Schmidt decomposition is the foundation of **matrix product states (MPS)**—a key tool for simulating 1D quantum systems.

---

## Worked Examples

### Example 1: Computing Schmidt Decomposition

**Problem:** Find the Schmidt decomposition of |ψ⟩ = (|00⟩ + |01⟩ + |10⟩)/√3.

**Solution:**

Coefficient matrix:
$$C = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$$

SVD: C = UΣV†

Computing: eigenvalues of C†C:
$$C^\dagger C = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$

Eigenvalues: λ₁ = (3+√5)/6 ≈ 0.873, λ₂ = (3-√5)/6 ≈ 0.127

Schmidt coefficients: √λ₁ ≈ 0.934, √λ₂ ≈ 0.357

Schmidt rank = 2 (entangled state).

### Example 2: Verifying Equal Spectra

**Problem:** For |Φ⁺⟩, verify that ρ_A and ρ_B have the same eigenvalues.

**Solution:**

$$\rho_A = \text{Tr}_B(|\Phi^+\rangle\langle\Phi^+|) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

$$\rho_B = \text{Tr}_A(|\Phi^+\rangle\langle\Phi^+|) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

Both have eigenvalues {½, ½}. ✓

### Example 3: Entanglement Entropy

**Problem:** Calculate the entanglement entropy of |ψ⟩ = √0.9|00⟩ + √0.1|11⟩.

**Solution:**

Schmidt coefficients: λ₁ = 0.9, λ₂ = 0.1

$$E = -0.9\log_2(0.9) - 0.1\log_2(0.1)$$
$$= -0.9(-0.152) - 0.1(-3.322) = 0.137 + 0.332 = 0.469 \text{ bits}$$

Compare to maximally entangled (E = 1 bit).

---

## Practice Problems

### Direct Application

**Problem 1:** Find the Schmidt decomposition of |ψ⟩ = (|00⟩ + |11⟩ + |22⟩)/√3 (qutrit-qutrit).

**Problem 2:** What is the Schmidt rank of |GHZ⟩ = (|000⟩ + |111⟩)/√2 across the A|BC partition?

**Problem 3:** For |ψ⟩ = α|00⟩ + β|11⟩, find the entanglement entropy.

### Intermediate

**Problem 4:** Prove that if |ψ⟩ has Schmidt rank 1, it is a product state.

**Problem 5:** Show that the Schmidt decomposition is unique (up to phases) when all λᵢ are distinct.

**Problem 6:** For the state (|01⟩ + |10⟩)/√2, find the Schmidt basis.

### Challenging

**Problem 7:** Prove that S(ρ_A) = S(ρ_B) for any bipartite pure state.

**Problem 8:** Show that the maximum entanglement entropy for d×d systems is log₂(d).

**Problem 9:** For a random bipartite pure state, what is the expected Schmidt rank?

---

## Computational Lab

```python
"""
Day 515: Schmidt Decomposition
Computing and analyzing bipartite entanglement
"""

import numpy as np
import matplotlib.pyplot as plt

def schmidt_decomposition(psi, dim_A, dim_B):
    """
    Compute Schmidt decomposition of bipartite pure state.
    Returns: (schmidt_coefficients, basis_A, basis_B)
    """
    # Reshape state into coefficient matrix
    C = psi.reshape(dim_A, dim_B)

    # SVD
    U, S, Vh = np.linalg.svd(C, full_matrices=False)

    # Schmidt coefficients are S² (eigenvalues of reduced states)
    lambda_vals = S**2

    # Filter out zeros
    nonzero = lambda_vals > 1e-10
    lambda_vals = lambda_vals[nonzero]
    U = U[:, nonzero]
    Vh = Vh[nonzero, :]

    return lambda_vals, U, Vh.conj()

def entanglement_entropy(lambda_vals):
    """Compute entanglement entropy from Schmidt coefficients"""
    return -np.sum(lambda_vals * np.log2(lambda_vals + 1e-15))

# Standard states
ket_00 = np.array([1, 0, 0, 0], dtype=complex)
ket_01 = np.array([0, 1, 0, 0], dtype=complex)
ket_10 = np.array([0, 0, 1, 0], dtype=complex)
ket_11 = np.array([0, 0, 0, 1], dtype=complex)

phi_plus = (ket_00 + ket_11) / np.sqrt(2)
psi_plus = (ket_01 + ket_10) / np.sqrt(2)

print("=" * 60)
print("SCHMIDT DECOMPOSITION EXAMPLES")
print("=" * 60)

# Example 1: Bell state
print("\n--- Bell state |Φ⁺⟩ ---")
lambdas, U, V = schmidt_decomposition(phi_plus, 2, 2)
print(f"Schmidt coefficients: {lambdas}")
print(f"Schmidt rank: {len(lambdas)}")
print(f"Entanglement entropy: {entanglement_entropy(lambdas):.4f} bits")

# Example 2: Product state
print("\n--- Product state |+⟩|0⟩ ---")
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
ket_0 = np.array([1, 0], dtype=complex)
psi_product = np.kron(ket_plus, ket_0)
lambdas, U, V = schmidt_decomposition(psi_product, 2, 2)
print(f"Schmidt coefficients: {lambdas}")
print(f"Schmidt rank: {len(lambdas)}")
print(f"Entanglement entropy: {entanglement_entropy(lambdas):.4f} bits")

# Example 3: Partially entangled
print("\n--- Partially entangled ---")
psi_partial = np.sqrt(0.9) * ket_00 + np.sqrt(0.1) * ket_11
lambdas, U, V = schmidt_decomposition(psi_partial, 2, 2)
print(f"Schmidt coefficients: {lambdas}")
print(f"Schmidt rank: {len(lambdas)}")
print(f"Entanglement entropy: {entanglement_entropy(lambdas):.4f} bits")

# Example 4: General state
print("\n--- State (|00⟩+|01⟩+|10⟩)/√3 ---")
psi_gen = (ket_00 + ket_01 + ket_10) / np.sqrt(3)
lambdas, U, V = schmidt_decomposition(psi_gen, 2, 2)
print(f"Schmidt coefficients: {lambdas}")
print(f"Schmidt rank: {len(lambdas)}")
print(f"Entanglement entropy: {entanglement_entropy(lambdas):.4f} bits")

# Verify equal spectra of reduced states
print("\n" + "=" * 60)
print("VERIFYING EQUAL SPECTRA")
print("=" * 60)

def partial_trace_B(rho_AB, dim_A, dim_B):
    rho = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho, axis1=1, axis2=3)

def partial_trace_A(rho_AB, dim_A, dim_B):
    rho = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho, axis1=0, axis2=2)

rho_AB = np.outer(psi_gen, psi_gen.conj())
rho_A = partial_trace_B(rho_AB, 2, 2)
rho_B = partial_trace_A(rho_AB, 2, 2)

evals_A = np.sort(np.linalg.eigvalsh(rho_A))[::-1]
evals_B = np.sort(np.linalg.eigvalsh(rho_B))[::-1]

print(f"\nFor |ψ⟩ = (|00⟩+|01⟩+|10⟩)/√3:")
print(f"Eigenvalues of ρ_A: {evals_A}")
print(f"Eigenvalues of ρ_B: {evals_B}")
print(f"Equal (Schmidt theorem): {np.allclose(evals_A, evals_B)}")

# Visualization: Entanglement entropy vs mixing
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Entropy vs Schmidt coefficient
ax = axes[0]
p_vals = np.linspace(0.001, 0.999, 100)
entropy_vals = -p_vals*np.log2(p_vals) - (1-p_vals)*np.log2(1-p_vals)

ax.plot(p_vals, entropy_vals, 'b-', lw=2)
ax.axhline(1, color='r', ls='--', label='Max (Bell state)')
ax.axhline(0, color='g', ls='--', label='Min (product)')
ax.set_xlabel('λ₁ (with λ₂ = 1-λ₁)')
ax.set_ylabel('Entanglement Entropy (bits)')
ax.set_title('Entanglement vs Schmidt Coefficients')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Random state statistics
ax = axes[1]
np.random.seed(42)
n_samples = 1000
entropies = []
ranks = []

for _ in range(n_samples):
    # Random pure state (Haar measure approximation)
    psi_random = np.random.randn(4) + 1j*np.random.randn(4)
    psi_random /= np.linalg.norm(psi_random)
    lambdas, _, _ = schmidt_decomposition(psi_random, 2, 2)
    entropies.append(entanglement_entropy(lambdas))
    ranks.append(len(lambdas))

ax.hist(entropies, bins=30, density=True, alpha=0.7)
ax.axvline(np.mean(entropies), color='r', ls='--', lw=2,
           label=f'Mean = {np.mean(entropies):.3f}')
ax.set_xlabel('Entanglement Entropy (bits)')
ax.set_ylabel('Probability Density')
ax.set_title('Entanglement Distribution (Random States)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('schmidt_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nRandom states: mean entropy = {np.mean(entropies):.3f} bits")
print(f"Random states: all have Schmidt rank 2")

print("\n" + "=" * 60)
print("Day 515 Complete: Schmidt Decomposition")
print("=" * 60)
```

---

## Summary

### Schmidt Decomposition Theorem

$$|\psi\rangle_{AB} = \sum_{i=1}^{r} \sqrt{\lambda_i} |a_i\rangle_A |b_i\rangle_B$$

### Key Properties

| Property | Formula | Significance |
|----------|---------|--------------|
| Schmidt coefficients | λᵢ > 0, Σλᵢ = 1 | Eigenvalues of reduced states |
| Schmidt rank | r = #{λᵢ > 0} | r=1 ⟺ product state |
| Entanglement entropy | E = -Σλᵢ log λᵢ | Quantifies entanglement |
| Equal spectra | spec(ρ_A) = spec(ρ_B) | Fundamental symmetry |

---

## Daily Checklist

- [ ] I can state the Schmidt decomposition theorem
- [ ] I can compute Schmidt coefficients using SVD
- [ ] I understand Schmidt rank as entanglement indicator
- [ ] I can calculate entanglement entropy
- [ ] I understand why reduced states have equal spectra

---

## Preview: Day 516

Tomorrow we'll study **entanglement detection**—how to determine whether a given state is entangled or separable.

---

*Next: Day 516 — Entanglement Detection*
