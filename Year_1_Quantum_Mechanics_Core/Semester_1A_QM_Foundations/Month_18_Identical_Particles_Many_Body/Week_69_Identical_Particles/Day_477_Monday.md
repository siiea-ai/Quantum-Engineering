# Day 477: Permutation Symmetry

## Overview
**Day 477** | Year 1, Month 18, Week 69 | Indistinguishability and Exchange

Today we explore the fundamental concept of identical particles in quantum mechanics—particles that are truly indistinguishable, leading to profound consequences for multi-particle systems.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Indistinguishability and permutations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Exchange operator formalism |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Two-particle system simulation |

---

## Learning Objectives

By the end of today, you will be able to:
1. Explain why identical particles are fundamentally indistinguishable
2. Define and apply the exchange (permutation) operator
3. Find eigenvalues and eigenstates of the exchange operator
4. Understand the symmetrization postulate
5. Connect permutation symmetry to physical observables
6. Contrast classical vs quantum identical particles

---

## Core Content

### Classical vs Quantum Identical Particles

**Classical particles:** Even if two particles are identical, we can (in principle) track their trajectories and distinguish them.

**Quantum particles:** The uncertainty principle prevents trajectory tracking. Identical particles are **fundamentally indistinguishable**.

### The Indistinguishability Postulate

For a system of identical particles, there is **no physical observable** that can distinguish which particle is which.

**Consequence:** All physical observables must be **symmetric** under particle exchange.

### Two-Particle Systems

Consider two identical particles with coordinates $\mathbf{r}_1$ and $\mathbf{r}_2$. The wave function:
$$\Psi(\mathbf{r}_1, \mathbf{r}_2, t)$$

**Question:** What happens under exchange $\mathbf{r}_1 \leftrightarrow \mathbf{r}_2$?

### The Exchange Operator

Define the **exchange (permutation) operator**:
$$\hat{P}_{12}\Psi(\mathbf{r}_1, \mathbf{r}_2) = \Psi(\mathbf{r}_2, \mathbf{r}_1)$$

**Properties:**
1. $\hat{P}_{12}^2 = \mathbb{1}$ (exchanging twice returns to original)
2. $\hat{P}_{12}^\dagger = \hat{P}_{12}$ (Hermitian)
3. $\hat{P}_{12}^\dagger = \hat{P}_{12}^{-1}$ (unitary)

### Eigenvalues of Exchange Operator

From $\hat{P}_{12}^2 = \mathbb{1}$:
$$\hat{P}_{12}^2|\psi\rangle = \lambda^2|\psi\rangle = |\psi\rangle$$

Therefore: $\lambda^2 = 1 \Rightarrow \lambda = \pm 1$

$$\boxed{\text{Eigenvalues of } \hat{P}_{12}: \quad +1 \text{ (symmetric)} \quad -1 \text{ (antisymmetric)}}$$

### Symmetric and Antisymmetric States

**Symmetric eigenstate** (λ = +1):
$$\hat{P}_{12}|\Psi_S\rangle = +|\Psi_S\rangle$$
$$\Psi_S(\mathbf{r}_1, \mathbf{r}_2) = \Psi_S(\mathbf{r}_2, \mathbf{r}_1)$$

**Antisymmetric eigenstate** (λ = -1):
$$\hat{P}_{12}|\Psi_A\rangle = -|\Psi_A\rangle$$
$$\Psi_A(\mathbf{r}_1, \mathbf{r}_2) = -\Psi_A(\mathbf{r}_2, \mathbf{r}_1)$$

### The Symmetrization Postulate

**Fundamental postulate of quantum mechanics:**

The wave function of identical particles must be either:
- **Symmetric** under all particle exchanges → **Bosons**
- **Antisymmetric** under all particle exchanges → **Fermions**

This is **not** a derived result in non-relativistic QM—it's a postulate (though justified by relativistic QFT).

### Commutation with Hamiltonian

For identical particles, the Hamiltonian is symmetric:
$$H(\mathbf{r}_1, \mathbf{r}_2) = H(\mathbf{r}_2, \mathbf{r}_1)$$

Therefore:
$$[\hat{P}_{12}, \hat{H}] = 0$$

**Consequence:** Exchange symmetry is **conserved** in time!

If a state starts symmetric (antisymmetric), it remains symmetric (antisymmetric) forever.

### Constructing Symmetric/Antisymmetric States

Given two single-particle states |α⟩ and |β⟩:

**Product state:**
$$|\alpha\rangle_1|\beta\rangle_2 \equiv |\alpha, \beta\rangle$$

**Symmetric combination:**
$$|\Psi_S\rangle = \frac{1}{\sqrt{2}}(|\alpha, \beta\rangle + |\beta, \alpha\rangle)$$

**Antisymmetric combination:**
$$|\Psi_A\rangle = \frac{1}{\sqrt{2}}(|\alpha, \beta\rangle - |\beta, \alpha\rangle)$$

### Verification

$$\hat{P}_{12}|\Psi_S\rangle = \frac{1}{\sqrt{2}}(|\beta, \alpha\rangle + |\alpha, \beta\rangle) = +|\Psi_S\rangle \checkmark$$

$$\hat{P}_{12}|\Psi_A\rangle = \frac{1}{\sqrt{2}}(|\beta, \alpha\rangle - |\alpha, \beta\rangle) = -|\Psi_A\rangle \checkmark$$

---

## N-Particle Generalization

### Permutation Group S_N

For N particles, the symmetric group $S_N$ contains all N! permutations.

**Example:** S_3 has 3! = 6 elements.

### Exchange Operators

Define $\hat{P}_{ij}$ to exchange particles i and j.

**Composition:** $\hat{P}_{12}\hat{P}_{23} \neq \hat{P}_{23}\hat{P}_{12}$ in general.

### Symmetrization/Antisymmetrization

**Symmetric state (bosons):**
$$|\Psi_S\rangle = \frac{1}{\sqrt{N!}}\sum_{\sigma \in S_N}|\psi_{\sigma(1)}, \psi_{\sigma(2)}, \ldots, \psi_{\sigma(N)}\rangle$$

**Antisymmetric state (fermions):**
$$|\Psi_A\rangle = \frac{1}{\sqrt{N!}}\sum_{\sigma \in S_N}(-1)^{\sigma}|\psi_{\sigma(1)}, \psi_{\sigma(2)}, \ldots, \psi_{\sigma(N)}\rangle$$

where $(-1)^\sigma$ = sign of permutation σ.

---

## Quantum Computing Connection

### Qubit Distinguishability

**Key difference:** Qubits in a quantum computer are **distinguishable**—they're at different physical locations (different transmon, different ion, etc.).

The tensor product structure $|q_1\rangle \otimes |q_2\rangle$ doesn't require symmetrization because the qubits are labeled by their position.

### Fermionic Simulation

When simulating **fermionic systems** (molecules, materials) on a quantum computer:
- Must map fermionic operators to qubit operators
- Jordan-Wigner or Bravyi-Kitaev transformations
- Antisymmetry encoded in transformation rules

### Bosonic Codes

**Bosonic error correction:** Encode qubits in bosonic modes (photons, phonons)
- Cat codes use superpositions of coherent states
- GKP codes use grid states
- Bosonic symmetry affects code structure

---

## Worked Examples

### Example 1: Two Spin-1/2 Particles

**Problem:** Construct the symmetric and antisymmetric spatial wave functions for two particles in states ψ_a(r) and ψ_b(r).

**Solution:**

The spatial wave functions are:

**Symmetric:**
$$\Psi_S(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\psi_a(\mathbf{r}_1)\psi_b(\mathbf{r}_2) + \psi_a(\mathbf{r}_2)\psi_b(\mathbf{r}_1)]$$

**Antisymmetric:**
$$\Psi_A(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\psi_a(\mathbf{r}_1)\psi_b(\mathbf{r}_2) - \psi_a(\mathbf{r}_2)\psi_b(\mathbf{r}_1)]$$

Note: If ψ_a = ψ_b, then $\Psi_A = 0$!

### Example 2: Exchange Symmetry of Observables

**Problem:** Show that ⟨r₁²⟩ = ⟨r₂²⟩ for any symmetric or antisymmetric state.

**Solution:**

For any state with definite exchange symmetry:
$$\langle r_1^2\rangle = \langle\Psi|\hat{r}_1^2|\Psi\rangle$$

Using $\hat{P}_{12}|\Psi\rangle = \pm|\Psi\rangle$ and $\hat{P}_{12}^{-1}\hat{r}_1^2\hat{P}_{12} = \hat{r}_2^2$:

$$\langle r_1^2\rangle = \langle\Psi|\hat{P}_{12}^{-1}\hat{r}_1^2\hat{P}_{12}|\Psi\rangle = \langle\Psi|\hat{r}_2^2|\Psi\rangle = \langle r_2^2\rangle$$

Physical interpretation: You can't tell particle 1 from particle 2!

### Example 3: Three Identical Particles

**Problem:** How many independent exchange operators are there for 3 particles?

**Solution:**

There are $\binom{3}{2} = 3$ pair exchanges: $\hat{P}_{12}$, $\hat{P}_{13}$, $\hat{P}_{23}$.

These generate the full symmetric group S_3 (6 elements):
- Identity: $\mathbb{1}$
- Pair exchanges: $\hat{P}_{12}$, $\hat{P}_{13}$, $\hat{P}_{23}$
- Cyclic permutations: $\hat{P}_{12}\hat{P}_{23}$, $\hat{P}_{23}\hat{P}_{12}$

Only 3 generators needed, but they don't commute.

---

## Practice Problems

### Problem Set 69.1

**Direct Application:**
1. Show that $\hat{P}_{12}$ is both Hermitian and unitary. What property must an operator have to be both?

2. For a wave function $\Psi(x_1, x_2) = x_1^2 x_2$, decompose it into symmetric and antisymmetric parts.

3. Verify that $[\hat{P}_{12}, \hat{H}] = 0$ for $\hat{H} = -\frac{\hbar^2}{2m}(\nabla_1^2 + \nabla_2^2) + V(|\mathbf{r}_1 - \mathbf{r}_2|)$.

**Intermediate:**
4. For 3 particles, show that $\hat{P}_{12}\hat{P}_{23}\hat{P}_{12} = \hat{P}_{23}\hat{P}_{12}\hat{P}_{23}$ (braid relation).

5. How many terms are in the symmetrized wave function for 4 identical bosons in 4 different states?

6. Show that the antisymmetrized wave function for N fermions in the same state is identically zero.

**Challenging:**
7. For N particles, prove that the symmetric and antisymmetric subspaces are orthogonal.

8. Consider two identical particles in a 1D harmonic oscillator. If one is in n=0 and one in n=1, write the symmetric and antisymmetric spatial wave functions explicitly.

9. A system of N identical particles has Hamiltonian $H = \sum_i h(i) + \sum_{i<j} v(i,j)$. Show this commutes with all exchange operators.

---

## Computational Lab

```python
"""
Day 477 Lab: Permutation Symmetry and Exchange Operators
Visualizes symmetric and antisymmetric wave functions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations

# ============================================================
# EXCHANGE OPERATOR DEMONSTRATION
# ============================================================

print("=" * 60)
print("EXCHANGE OPERATOR PROPERTIES")
print("=" * 60)

# Define a simple 2-particle state as a matrix
# |ψ⟩ = Σ c_{ij} |i⟩|j⟩

def exchange_operator_matrix(dim):
    """
    Create the exchange operator P_12 for two particles,
    each in a Hilbert space of dimension 'dim'.

    P_12 |i⟩|j⟩ = |j⟩|i⟩
    """
    total_dim = dim * dim
    P = np.zeros((total_dim, total_dim))

    for i in range(dim):
        for j in range(dim):
            # |i⟩|j⟩ has index i*dim + j
            # |j⟩|i⟩ has index j*dim + i
            old_idx = i * dim + j
            new_idx = j * dim + i
            P[new_idx, old_idx] = 1

    return P

dim = 3  # Each particle in 3-dimensional space
P12 = exchange_operator_matrix(dim)

print(f"\nExchange operator P_12 for dim={dim}:")
print(P12)

# Verify properties
print("\nVerifying P_12 properties:")
print(f"P_12² = I: {np.allclose(P12 @ P12, np.eye(dim*dim))}")
print(f"P_12† = P_12 (Hermitian): {np.allclose(P12, P12.T)}")
print(f"P_12† P_12 = I (Unitary): {np.allclose(P12.T @ P12, np.eye(dim*dim))}")

# Find eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(P12)
print(f"\nEigenvalues of P_12: {np.unique(np.round(eigenvalues, 10))}")
print(f"Number of +1 eigenvalues (symmetric): {np.sum(np.isclose(eigenvalues, 1))}")
print(f"Number of -1 eigenvalues (antisymmetric): {np.sum(np.isclose(eigenvalues, -1))}")

# ============================================================
# SYMMETRIC AND ANTISYMMETRIC WAVE FUNCTIONS
# ============================================================

print("\n" + "=" * 60)
print("SYMMETRIC VS ANTISYMMETRIC WAVE FUNCTIONS")
print("=" * 60)

def psi_a(x, a=1):
    """Single-particle wave function a (Gaussian centered at -a)"""
    return np.exp(-(x + a)**2)

def psi_b(x, a=1):
    """Single-particle wave function b (Gaussian centered at +a)"""
    return np.exp(-(x - a)**2)

# Create grid
x = np.linspace(-4, 4, 100)
X1, X2 = np.meshgrid(x, x)

# Product state (not symmetrized)
psi_product = psi_a(X1) * psi_b(X2)

# Symmetric combination (bosons)
psi_symmetric = (1/np.sqrt(2)) * (psi_a(X1) * psi_b(X2) + psi_a(X2) * psi_b(X1))

# Antisymmetric combination (fermions)
psi_antisymmetric = (1/np.sqrt(2)) * (psi_a(X1) * psi_b(X2) - psi_a(X2) * psi_b(X1))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Product state
im0 = axes[0].contourf(X1, X2, psi_product, levels=50, cmap='RdBu_r')
axes[0].set_xlabel('$x_1$', fontsize=12)
axes[0].set_ylabel('$x_2$', fontsize=12)
axes[0].set_title('Product State (distinguishable)', fontsize=12)
axes[0].set_aspect('equal')
plt.colorbar(im0, ax=axes[0])

# Symmetric (bosons)
im1 = axes[1].contourf(X1, X2, psi_symmetric, levels=50, cmap='RdBu_r')
axes[1].set_xlabel('$x_1$', fontsize=12)
axes[1].set_ylabel('$x_2$', fontsize=12)
axes[1].set_title('Symmetric (Bosons)', fontsize=12)
axes[1].plot(x, x, 'k--', alpha=0.5, label='Exchange line')
axes[1].set_aspect('equal')
plt.colorbar(im1, ax=axes[1])

# Antisymmetric (fermions)
im2 = axes[2].contourf(X1, X2, psi_antisymmetric, levels=50, cmap='RdBu_r')
axes[2].set_xlabel('$x_1$', fontsize=12)
axes[2].set_ylabel('$x_2$', fontsize=12)
axes[2].set_title('Antisymmetric (Fermions)', fontsize=12)
axes[2].plot(x, x, 'k--', alpha=0.5, label='Node at x₁=x₂')
axes[2].set_aspect('equal')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('exchange_symmetry.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PROBABILITY DENSITY COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("PROBABILITY DENSITY |Ψ|²")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

prob_product = np.abs(psi_product)**2
prob_symmetric = np.abs(psi_symmetric)**2
prob_antisymmetric = np.abs(psi_antisymmetric)**2

for ax, prob, title in zip(axes,
                            [prob_product, prob_symmetric, prob_antisymmetric],
                            ['Distinguishable', 'Bosons', 'Fermions']):
    im = ax.contourf(X1, X2, prob, levels=50, cmap='viridis')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'|Ψ|² - {title}', fontsize=12)
    ax.plot(x, x, 'w--', alpha=0.5)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('probability_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# DIAGONAL CROSS-SECTION
# ============================================================

print("\n" + "=" * 60)
print("DIAGONAL SLICE: PROBABILITY ALONG x₁ = x₂")
print("=" * 60)

# Along diagonal x1 = x2
diagonal_indices = np.arange(len(x))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, prob_product[diagonal_indices, diagonal_indices],
        'b-', linewidth=2, label='Distinguishable')
ax.plot(x, prob_symmetric[diagonal_indices, diagonal_indices],
        'g-', linewidth=2, label='Bosons (enhanced)')
ax.plot(x, prob_antisymmetric[diagonal_indices, diagonal_indices],
        'r-', linewidth=2, label='Fermions (zero!)')

ax.set_xlabel('Position (x₁ = x₂)', fontsize=12)
ax.set_ylabel('Probability |Ψ(x, x)|²', fontsize=12)
ax.set_title('Probability for Two Particles at Same Position', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagonal_probability.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nKey observation: Fermions have ZERO probability of being at the same position!")
print("This is a precursor to the Pauli exclusion principle.")

# ============================================================
# N-PARTICLE PERMUTATIONS
# ============================================================

print("\n" + "=" * 60)
print("PERMUTATION GROUP S_N")
print("=" * 60)

def count_permutations(n):
    """Count and display permutation statistics"""
    perms = list(permutations(range(1, n+1)))
    print(f"\nS_{n}: {len(perms)} permutations")

    # Count even and odd permutations
    even_count = 0
    odd_count = 0

    for perm in perms:
        # Count inversions
        inversions = sum(1 for i in range(len(perm))
                        for j in range(i+1, len(perm))
                        if perm[i] > perm[j])
        if inversions % 2 == 0:
            even_count += 1
        else:
            odd_count += 1

    print(f"Even permutations (sign +1): {even_count}")
    print(f"Odd permutations (sign -1): {odd_count}")

    return perms

for n in range(2, 5):
    count_permutations(n)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("""
1. Identical particles are fundamentally indistinguishable
2. Exchange operator P_12 has eigenvalues ±1
3. Symmetric states (λ=+1): Bosons
4. Antisymmetric states (λ=-1): Fermions
5. [P_12, H] = 0 → exchange symmetry is conserved
6. Fermions cannot be at the same position (node at x₁=x₂)
7. Bosons have enhanced probability at same position
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Exchange operator | $\hat{P}_{12}\Psi(\mathbf{r}_1, \mathbf{r}_2) = \Psi(\mathbf{r}_2, \mathbf{r}_1)$ |
| Exchange eigenvalues | $\hat{P}_{12}^2 = \mathbb{1} \Rightarrow \lambda = \pm 1$ |
| Symmetric state | $\Psi_S = \frac{1}{\sqrt{2}}(\psi_a\psi_b + \psi_b\psi_a)$ |
| Antisymmetric state | $\Psi_A = \frac{1}{\sqrt{2}}(\psi_a\psi_b - \psi_b\psi_a)$ |
| Conservation | $[\hat{P}_{12}, \hat{H}] = 0$ |

### Main Takeaways

1. **Identical particles** are truly indistinguishable in QM
2. **Exchange symmetry** constrains wave functions
3. **Bosons** have symmetric, **fermions** antisymmetric wave functions
4. Symmetry is **conserved** under time evolution
5. **Physical observables** must be symmetric under exchange

---

## Daily Checklist

- [ ] I understand why identical particles are indistinguishable
- [ ] I can apply the exchange operator to wave functions
- [ ] I can construct symmetric and antisymmetric combinations
- [ ] I understand the symmetrization postulate
- [ ] I completed the computational lab

---

## Preview: Day 478

Tomorrow we explore **bosons and fermions** in detail—their contrasting properties, statistical behavior, and physical examples.

---

**Next:** [Day_478_Tuesday.md](Day_478_Tuesday.md) — Bosons and Fermions
