# Day 508: Purity and Mixedness

## Overview

**Day 508** | Week 73, Day 4 | Year 1, Month 19 | Quantifying Quantum vs Classical Uncertainty

Today we study purity—the measure that distinguishes pure quantum states from classical statistical mixtures. We'll learn to quantify "how mixed" a state is and understand its physical and information-theoretic significance.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Purity theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. Define and compute the purity Tr(ρ²)
2. Prove bounds on purity: 1/d ≤ γ ≤ 1
3. Identify maximally mixed states and their properties
4. Relate purity to the Bloch vector length (qubits)
5. Connect purity to von Neumann entropy
6. Understand the physical meaning of purity in quantum information

---

## Core Content

### Defining Purity

The **purity** of a density matrix ρ is:

$$\boxed{\gamma = \text{Tr}(\rho^2)}$$

**Properties:**
- Always real and positive
- Bounded: 1/d ≤ γ ≤ 1 (d = dimension)
- γ = 1 ⟺ pure state
- γ = 1/d ⟺ maximally mixed state

### Purity for Pure States

For ρ = |ψ⟩⟨ψ|:
$$\text{Tr}(\rho^2) = \text{Tr}(|\psi\rangle\langle\psi|\psi\rangle\langle\psi|) = \text{Tr}(|\psi\rangle\langle\psi|) = 1$$

**Key insight:** ρ² = ρ for pure states (idempotent), so Tr(ρ²) = Tr(ρ) = 1.

### Purity for Mixed States

For a mixed state with spectral decomposition ρ = Σᵢ λᵢ|eᵢ⟩⟨eᵢ|:
$$\text{Tr}(\rho^2) = \sum_i \lambda_i^2$$

Since Σᵢ λᵢ = 1 and λᵢ ≥ 0:
- Maximum when one λᵢ = 1 (pure state): γ = 1
- Minimum when all λᵢ = 1/d (maximally mixed): γ = d·(1/d)² = 1/d

$$\boxed{\frac{1}{d} \leq \text{Tr}(\rho^2) \leq 1}$$

### The Maximally Mixed State

The **maximally mixed state** in d dimensions:

$$\boxed{\rho_{mm} = \frac{I}{d}}$$

**Properties:**
- All eigenvalues equal: λᵢ = 1/d
- Purity: γ = 1/d (minimum possible)
- Maximum entropy (maximum uncertainty)
- No preferred direction or basis
- Completely random outcomes for any measurement

**For a qubit (d=2):**
$$\rho_{mm} = \frac{1}{2}I = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

Purity: γ = ½

### Linear Entropy

A related measure is the **linear entropy**:

$$\boxed{S_L(\rho) = 1 - \text{Tr}(\rho^2) = 1 - \gamma}$$

**Properties:**
- S_L = 0 for pure states
- S_L = 1 - 1/d for maximally mixed
- Approximates von Neumann entropy for nearly pure states

### Purity and the Bloch Sphere

For a qubit with Bloch representation:
$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma}) = \frac{1}{2}\begin{pmatrix} 1+z & x-iy \\ x+iy & 1-z \end{pmatrix}$$

The purity is:

$$\text{Tr}(\rho^2) = \frac{1}{2}(1 + |\vec{r}|^2)$$

Therefore:
$$\boxed{|\vec{r}|^2 = 2\gamma - 1}$$

**Interpretation:**
- Pure states: |r⃗| = 1 (surface of Bloch sphere)
- Mixed states: |r⃗| < 1 (interior of Bloch ball)
- Maximally mixed: |r⃗| = 0 (center)

### Von Neumann Entropy

The **von Neumann entropy** is the quantum analog of Shannon entropy:

$$\boxed{S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i}$$

**Properties:**
- S(ρ) = 0 for pure states
- S(ρ) = log₂ d for maximally mixed states
- Always non-negative

**Relation to purity:** For nearly pure states:
$$S(\rho) \approx S_L(\rho) = 1 - \gamma$$

### Physical Interpretation

| Purity γ | State Type | Uncertainty | Knowledge |
|----------|-----------|-------------|-----------|
| γ = 1 | Pure | Only quantum | Complete |
| 1/d < γ < 1 | Mixed | Quantum + classical | Partial |
| γ = 1/d | Max mixed | Maximum classical | Minimal |

**Key insight:** Purity measures how much of our uncertainty is classical (ignorance) vs quantum (intrinsic).

---

## Quantum Computing Connection

### Decoherence and Purity

Decoherence causes purity to decrease:
- Initial qubit: γ = 1 (pure)
- After decoherence: γ < 1 (mixed)
- Complete decoherence: γ → ½ (maximally mixed)

### T₁ and T₂ Times

In real qubits:
- **T₁ (amplitude damping):** Energy relaxation
- **T₂ (dephasing):** Coherence decay

Both processes reduce purity.

### Purity as Error Metric

Purity loss indicates errors:
$$\text{Error} \propto 1 - \gamma$$

For a perfect qubit: γ = 1
After noise: γ < 1

### Quantum Volume

IBM's quantum volume metric partly depends on how well systems maintain purity.

---

## Worked Examples

### Example 1: Computing Purity

**Problem:** Find the purity of ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1|.

**Solution:**

Method 1: Direct calculation
$$\rho = \begin{pmatrix} 3/4 & 0 \\ 0 & 1/4 \end{pmatrix}$$

$$\rho^2 = \begin{pmatrix} 9/16 & 0 \\ 0 & 1/16 \end{pmatrix}$$

$$\gamma = \text{Tr}(\rho^2) = \frac{9}{16} + \frac{1}{16} = \frac{10}{16} = \frac{5}{8} = 0.625$$

Method 2: Using eigenvalues
$$\gamma = \lambda_1^2 + \lambda_2^2 = (3/4)^2 + (1/4)^2 = 9/16 + 1/16 = 10/16$$

### Example 2: Bloch Vector and Purity

**Problem:** A qubit has ρ = ½(I + 0.6X + 0.8Z). Find its purity and verify using the Bloch vector.

**Solution:**

The Bloch vector is r⃗ = (0.6, 0, 0.8).
$$|\vec{r}|^2 = 0.36 + 0 + 0.64 = 1.0$$

Using our formula:
$$\gamma = \frac{1 + |\vec{r}|^2}{2} = \frac{1 + 1}{2} = 1$$

This is a pure state! (It lies on the Bloch sphere surface.)

### Example 3: Purity After Partial Dephasing

**Problem:** A pure state |+⟩ undergoes partial dephasing: ρ → (1-p)|+⟩⟨+| + p·I/2. Find the purity as a function of p.

**Solution:**

Initial Bloch vector: r⃗ = (1, 0, 0) (pointing in +x direction)

After dephasing, the Bloch vector shrinks:
$$\vec{r}' = (1-p)\vec{r} = (1-p, 0, 0)$$

Purity:
$$\gamma(p) = \frac{1 + (1-p)^2}{2} = \frac{1 + 1 - 2p + p^2}{2} = 1 - p + \frac{p^2}{2}$$

Check: γ(0) = 1 (pure), γ(1) = ½ (maximally mixed) ✓

---

## Practice Problems

### Direct Application

**Problem 1:** Compute the purity of ρ = ½|+⟩⟨+| + ½|−⟩⟨−|.

**Problem 2:** A qutrit (d=3) is in state ρ = ⅓I. What is its purity?

**Problem 3:** Find the Bloch vector for a qubit with purity γ = 0.75, assuming r⃗ is along the z-axis.

### Intermediate

**Problem 4:** Prove that purity is invariant under unitary transformations: Tr((UρU†)²) = Tr(ρ²).

**Problem 5:** A qubit starts in |0⟩ and undergoes 50% dephasing. Find the final purity.

**Problem 6:** Show that the linear entropy S_L = 1 - γ satisfies 0 ≤ S_L ≤ 1 - 1/d.

### Challenging

**Problem 7:** Prove: Tr(ρ²) = 1 if and only if ρ = |ψ⟩⟨ψ| for some |ψ⟩.

**Problem 8:** For two qubits in state ρ_AB, show that Tr(ρ_A²) ≤ 1 where ρ_A is the reduced state.

**Problem 9:** Derive the relationship between purity and the 2-Rényi entropy: S₂(ρ) = -log₂(Tr(ρ²)).

---

## Computational Lab

```python
"""
Day 508: Purity and Mixedness
Quantifying pure vs mixed states
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def purity(rho):
    """Compute purity γ = Tr(ρ²)"""
    return np.trace(rho @ rho).real

def linear_entropy(rho):
    """Compute linear entropy S_L = 1 - Tr(ρ²)"""
    return 1 - purity(rho)

def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def bloch_vector(rho):
    """Extract Bloch vector from qubit density matrix"""
    rx = np.trace(rho @ X).real
    ry = np.trace(rho @ Y).real
    rz = np.trace(rho @ Z).real
    return np.array([rx, ry, rz])

def density_from_bloch(r):
    """Create density matrix from Bloch vector"""
    return 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

print("=" * 60)
print("PURITY CALCULATIONS")
print("=" * 60)

# Example states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)

rho_pure = ket_plus @ ket_plus.conj().T
rho_mixed = 0.75 * (ket_0 @ ket_0.conj().T) + 0.25 * (ket_1 @ ket_1.conj().T)
rho_max_mixed = 0.5 * I

states = [
    ("Pure |+⟩", rho_pure),
    ("Mixed ¾|0⟩ + ¼|1⟩", rho_mixed),
    ("Maximally mixed I/2", rho_max_mixed)
]

for name, rho in states:
    gamma = purity(rho)
    s_l = linear_entropy(rho)
    s_vn = von_neumann_entropy(rho)
    r = bloch_vector(rho)

    print(f"\n{name}:")
    print(f"  Purity γ = {gamma:.4f}")
    print(f"  Linear entropy S_L = {s_l:.4f}")
    print(f"  von Neumann entropy S = {s_vn:.4f}")
    print(f"  Bloch vector |r| = {np.linalg.norm(r):.4f}")

print("\n" + "=" * 60)
print("PURITY AND BLOCH VECTOR RELATIONSHIP")
print("=" * 60)

# Verify γ = (1 + |r|²)/2
for name, rho in states:
    gamma = purity(rho)
    r = bloch_vector(rho)
    r_norm_sq = np.linalg.norm(r)**2
    gamma_from_bloch = (1 + r_norm_sq) / 2

    print(f"\n{name}:")
    print(f"  Direct: γ = {gamma:.4f}")
    print(f"  From Bloch: γ = (1 + |r|²)/2 = {gamma_from_bloch:.4f}")
    print(f"  Match: {np.isclose(gamma, gamma_from_bloch)}")

# Visualization
fig = plt.figure(figsize=(15, 5))

# 1. Bloch ball with purity
ax1 = fig.add_subplot(131, projection='3d')

# Draw Bloch sphere wireframe
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(x, y, z, alpha=0.1, color='blue')

# Plot states with different purities
purities_plot = [1.0, 0.8, 0.6, 0.5]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(purities_plot)))

for gamma_val, color in zip(purities_plot, colors):
    r_mag = np.sqrt(2*gamma_val - 1) if gamma_val >= 0.5 else 0
    # Plot a few points at this radius
    for theta in np.linspace(0, 2*np.pi, 8):
        x = r_mag * np.cos(theta)
        y = r_mag * np.sin(theta)
        ax1.scatter([x], [y], [0], c=[color], s=50, alpha=0.7)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Bloch Ball: Purity = Distance from Center')

# 2. Purity vs eigenvalue distribution
ax2 = fig.add_subplot(132)

lambda1 = np.linspace(0, 1, 100)
lambda2 = 1 - lambda1
purity_vals = lambda1**2 + lambda2**2

ax2.plot(lambda1, purity_vals, 'b-', lw=2)
ax2.axhline(0.5, color='r', ls='--', label='Maximally mixed (γ=1/d)')
ax2.axhline(1.0, color='g', ls='--', label='Pure state (γ=1)')
ax2.fill_between(lambda1, 0.5, purity_vals, alpha=0.3, color='blue')
ax2.set_xlabel('Eigenvalue λ₁ (with λ₂ = 1-λ₁)')
ax2.set_ylabel('Purity γ = Tr(ρ²)')
ax2.set_title('Purity vs Eigenvalue Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Purity vs von Neumann entropy
ax3 = fig.add_subplot(133)

# Sample mixed states
n_samples = 1000
entropies = []
purities = []

for _ in range(n_samples):
    # Random eigenvalues (probability distribution)
    p = np.random.dirichlet([1, 1])  # Random 2-outcome distribution
    rho_sample = np.diag(p).astype(complex)
    purities.append(purity(rho_sample))
    entropies.append(von_neumann_entropy(rho_sample))

ax3.scatter(purities, entropies, alpha=0.3, s=5)

# Theoretical curve for diagonal qubit states
p_theory = np.linspace(0.001, 0.999, 100)
purity_theory = p_theory**2 + (1-p_theory)**2
entropy_theory = -p_theory*np.log2(p_theory) - (1-p_theory)*np.log2(1-p_theory)

ax3.plot(purity_theory, entropy_theory, 'r-', lw=2, label='Diagonal states')
ax3.set_xlabel('Purity γ')
ax3.set_ylabel('von Neumann Entropy S')
ax3.set_title('Entropy vs Purity')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('purity_mixedness.png', dpi=150, bbox_inches='tight')
plt.show()

# Decoherence simulation
print("\n" + "=" * 60)
print("DECOHERENCE: PURITY DECAY")
print("=" * 60)

# Simulate gradual dephasing
dephasing_levels = np.linspace(0, 1, 50)
purity_decay = []

for p in dephasing_levels:
    # Mix pure |+⟩ with maximally mixed
    rho_decohered = (1-p) * rho_pure + p * rho_max_mixed
    purity_decay.append(purity(rho_decohered))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(dephasing_levels, purity_decay, 'b-', lw=2)
ax.axhline(1.0, color='g', ls='--', label='Pure state')
ax.axhline(0.5, color='r', ls='--', label='Maximally mixed')
ax.set_xlabel('Dephasing parameter p')
ax.set_ylabel('Purity γ')
ax.set_title('Purity Decay Under Dephasing')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('purity_decay.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Day 508 Complete: Purity and Mixedness")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Quantity | Formula | Bounds |
|----------|---------|--------|
| Purity | γ = Tr(ρ²) | 1/d ≤ γ ≤ 1 |
| Linear entropy | S_L = 1 - γ | 0 ≤ S_L ≤ 1-1/d |
| Bloch relation | γ = (1 + \|r⃗\|²)/2 | Qubits only |
| Maximally mixed | ρ = I/d | γ = 1/d |

### Key Concepts

| State | Purity | Bloch vector | Entropy |
|-------|--------|--------------|---------|
| Pure | γ = 1 | \|r⃗\| = 1 | S = 0 |
| Mixed | 1/d < γ < 1 | 0 < \|r⃗\| < 1 | 0 < S < log d |
| Max mixed | γ = 1/d | \|r⃗\| = 0 | S = log d |

---

## Daily Checklist

- [ ] I can compute purity using Tr(ρ²)
- [ ] I understand the bounds 1/d ≤ γ ≤ 1
- [ ] I can relate purity to the Bloch vector for qubits
- [ ] I understand the maximally mixed state
- [ ] I can compute linear and von Neumann entropy
- [ ] I understand purity decay under decoherence

---

## Preview: Day 509

Tomorrow we'll explore the **Bloch sphere representation for mixed states**, understanding how the interior of the Bloch ball represents all possible qubit states.

---

*Next: Day 509 — Bloch Sphere (Mixed States)*
