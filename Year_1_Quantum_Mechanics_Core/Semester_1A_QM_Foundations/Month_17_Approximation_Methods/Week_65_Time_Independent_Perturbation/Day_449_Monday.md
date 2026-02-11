# Day 449: Non-Degenerate Perturbation Theory I

## Overview
**Day 449** | Year 1, Month 17, Week 65 | First-Order Corrections

Today we develop non-degenerate perturbation theory, the workhorse of quantum mechanics approximation methods. When exact solutions aren't available, this systematic approach yields accurate results.

---

## Learning Objectives

By the end of today, you will be able to:
1. Formulate the perturbation expansion
2. Derive first-order energy corrections
3. Calculate first-order wavefunction corrections
4. Understand the validity conditions
5. Apply to simple examples
6. Recognize the importance of matrix elements

---

## Core Content

### The Setup

We have a Hamiltonian:
$$\hat{H} = \hat{H}_0 + \lambda\hat{H}'$$

where:
- H₀: Unperturbed (exactly solvable) Hamiltonian
- H': Perturbation
- λ: Small parameter (0 ≤ λ ≤ 1)

**Known:** H₀|n⁰⟩ = E_n^(0)|n⁰⟩ (complete orthonormal set)

**Goal:** Find eigenvalues and eigenstates of H.

### Perturbation Expansion

Expand in powers of λ:
$$E_n = E_n^{(0)} + \lambda E_n^{(1)} + \lambda^2 E_n^{(2)} + ...$$
$$|n\rangle = |n^{(0)}\rangle + \lambda|n^{(1)}\rangle + \lambda^2|n^{(2)}\rangle + ...$$

### The Schrödinger Equation

$$(H_0 + \lambda H')|n\rangle = E_n|n\rangle$$

Substituting and collecting powers of λ:

**Order λ⁰:**
$$H_0|n^{(0)}\rangle = E_n^{(0)}|n^{(0)}\rangle$$

**Order λ¹:**
$$H_0|n^{(1)}\rangle + H'|n^{(0)}\rangle = E_n^{(0)}|n^{(1)}\rangle + E_n^{(1)}|n^{(0)}\rangle$$

### First-Order Energy

Take inner product with ⟨n⁰|:
$$\langle n^{(0)}|H_0|n^{(1)}\rangle + \langle n^{(0)}|H'|n^{(0)}\rangle = E_n^{(0)}\langle n^{(0)}|n^{(1)}\rangle + E_n^{(1)}$$

Since H₀ is Hermitian: ⟨n⁰|H₀ = E_n^(0)⟨n⁰|

$$\boxed{E_n^{(1)} = \langle n^{(0)} | H' | n^{(0)} \rangle}$$

**The first-order energy correction is the expectation value of the perturbation!**

### First-Order Wavefunction

Take inner product with ⟨m⁰| (m ≠ n):
$$E_m^{(0)}\langle m^{(0)}|n^{(1)}\rangle + \langle m^{(0)}|H'|n^{(0)}\rangle = E_n^{(0)}\langle m^{(0)}|n^{(1)}\rangle$$

Solve for the coefficient:
$$\langle m^{(0)}|n^{(1)}\rangle = \frac{\langle m^{(0)}|H'|n^{(0)}\rangle}{E_n^{(0)} - E_m^{(0)}}$$

Therefore:
$$\boxed{|n^{(1)}\rangle = \sum_{m \neq n} \frac{\langle m^{(0)}|H'|n^{(0)}\rangle}{E_n^{(0)} - E_m^{(0)}}|m^{(0)}\rangle}$$

### Validity Condition

The perturbation expansion is valid when:
$$\left|\frac{\langle m^{(0)}|H'|n^{(0)}\rangle}{E_n^{(0)} - E_m^{(0)}}\right| \ll 1$$

**Key requirement:** Non-degeneracy: E_n^(0) ≠ E_m^(0) for all m ≠ n.

### Normalization

To first order, the perturbed state is:
$$|n\rangle = |n^{(0)}\rangle + \lambda|n^{(1)}\rangle + O(\lambda^2)$$

Normalization: ⟨n|n⟩ = 1 requires ⟨n⁰|n^(1)⟩ = 0 (already satisfied by our construction).

---

## Quantum Computing Connection

### Error Analysis

Perturbation theory is used to analyze:
- **Gate errors:** H = H_ideal + δH
- **Noise effects:** Small unwanted interactions
- **Adiabatic evolution:** Ground state following

### VQE Foundation

The variational quantum eigensolver uses:
- Trial states parameterized by θ
- Energy expectation values E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
- Optimization to find ground state

---

## Worked Examples

### Example 1: Harmonic Oscillator with x⁴ Perturbation

**Problem:** Find E_n^(1) for H' = αx⁴ perturbing the harmonic oscillator.

**Solution:**
Using x = √(ℏ/2mω)(a + a†):
$$x^4 = \left(\frac{\hbar}{2m\omega}\right)^2 (a + a^\dagger)^4$$

For the ground state |0⟩:
$$E_0^{(1)} = \alpha\langle 0|x^4|0\rangle = \alpha\left(\frac{\hbar}{2m\omega}\right)^2 \langle 0|(a + a^\dagger)^4|0\rangle$$

Only terms with equal numbers of a and a† survive:
$$\langle 0|(a + a^\dagger)^4|0\rangle = 3$$

$$\boxed{E_0^{(1)} = \frac{3\alpha\hbar^2}{4m^2\omega^2}}$$

### Example 2: Particle in a Box with Linear Potential

**Problem:** A particle in a box [0, L] with H' = εx. Find E_n^(1).

**Solution:**
Unperturbed states: ψ_n^(0) = √(2/L) sin(nπx/L)

$$E_n^{(1)} = \varepsilon\int_0^L |\psi_n^{(0)}|^2 x\, dx = \frac{2\varepsilon}{L}\int_0^L x\sin^2\left(\frac{n\pi x}{L}\right)dx$$

Using ∫x sin²(αx)dx = x²/4 - x sin(2αx)/(4α) - cos(2αx)/(8α²):

$$\boxed{E_n^{(1)} = \frac{\varepsilon L}{2}}$$

The energy shift is the same for all levels (expected: center of mass shifts by L/2).

---

## Practice Problems

### Direct Application
1. For H' = λx² added to a particle in a box, what is E₁^(1)?
2. Show that E_n^(1) = 0 if H' has odd parity and |n⁰⟩ has definite parity.
3. Calculate E_0^(1) for a harmonic oscillator with H' = αx³.

### Intermediate
4. Find |1^(1)⟩ for the harmonic oscillator with H' = αx.
5. What is the first-order correction to the ground state energy of a hydrogen atom with H' = λr?
6. Show that first-order PT preserves orthogonality to O(λ).

### Challenging
7. Derive the exact eigenvalue equation for H₀ + λH' and show how the perturbation series emerges.
8. Find when the perturbation series diverges for H = -d²/dx² + x² + λx⁴.

---

## Computational Lab

```python
"""
Day 449: Non-Degenerate Perturbation Theory
First-order corrections demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Particle in a box with perturbation
def particle_in_box_perturbation():
    """
    H = -ℏ²/(2m) d²/dx² + V(x) in box [0, L]
    H' = εx (linear perturbation)
    """
    L = 1.0
    N = 100  # Number of basis states
    epsilon = 0.1  # Perturbation strength

    # Unperturbed energies (units where ℏ²/(2m) = 1)
    n = np.arange(1, N+1)
    E0 = (n * np.pi / L)**2

    # First-order correction: E_n^(1) = ε⟨n|x|n⟩
    # For particle in box: ⟨n|x|n⟩ = L/2 (center of box)
    E1_analytic = epsilon * L / 2 * np.ones(N)

    # Build full Hamiltonian matrix
    x_grid = np.linspace(0, L, 1000)
    dx = x_grid[1] - x_grid[0]

    def psi_n(x, n, L):
        return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

    # Compute matrix elements of x
    H_prime = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            psi_i = psi_n(x_grid, i+1, L)
            psi_j = psi_n(x_grid, j+1, L)
            H_prime[i, j] = epsilon * np.trapezoid(psi_i * x_grid * psi_j, x_grid)

    # Full Hamiltonian
    H0 = np.diag(E0)
    H = H0 + H_prime

    # Exact eigenvalues
    E_exact, _ = eigh(H)

    return E0, E1_analytic, E_exact, epsilon

# Run calculation
E0, E1, E_exact, epsilon = particle_in_box_perturbation()

# Compare perturbation theory vs exact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Energy levels
ax = axes[0]
n_show = 10
n = np.arange(1, n_show + 1)

ax.plot(n, E0[:n_show], 'bo-', label='Unperturbed E_n^(0)', markersize=8)
ax.plot(n, E0[:n_show] + E1[:n_show], 'gs-', label='First order E^(0) + E^(1)', markersize=8)
ax.plot(n, E_exact[:n_show], 'r^-', label='Exact', markersize=8)

ax.set_xlabel('n', fontsize=12)
ax.set_ylabel('Energy', fontsize=12)
ax.set_title(f'Particle in Box with Linear Perturbation (ε = {epsilon})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Error analysis
ax = axes[1]
error_1st = np.abs(E_exact[:n_show] - (E0[:n_show] + E1[:n_show]))
relative_error = error_1st / E_exact[:n_show]

ax.semilogy(n, error_1st, 'b-o', label='Absolute error')
ax.semilogy(n, relative_error, 'r-s', label='Relative error')

ax.set_xlabel('n', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_title('First-Order Perturbation Theory Error', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day449_perturbation_theory.png', dpi=150)
plt.show()

print("=== First-Order Perturbation Theory Results ===")
print(f"\nPerturbation strength ε = {epsilon}")
print(f"First-order correction E^(1) = ε⟨x⟩ = {E1[0]:.4f}")
print("\nComparison for first 5 states:")
print("n    E^(0)      E^(0)+E^(1)   Exact       Error")
print("-" * 55)
for i in range(5):
    print(f"{i+1}   {E0[i]:8.4f}   {E0[i]+E1[i]:10.4f}   {E_exact[i]:10.4f}   {error_1st[i]:.2e}")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Energy (1st order) | E_n^(1) = ⟨n⁰|H'|n⁰⟩ |
| State (1st order) | \|n^(1)⟩ = Σ_{m≠n} ⟨m⁰|H'|n⁰⟩/(E_n^(0)-E_m^(0)) \|m⁰⟩ |
| Validity | \|⟨m⁰|H'|n⁰⟩\| << \|E_n^(0) - E_m^(0)\| |

### Key Insights

1. **First-order energy** is the expectation value of the perturbation
2. **First-order state** is a sum over all other states
3. **Energy denominators** must be non-zero (non-degeneracy!)
4. **Convergence** requires perturbation to be "small"
5. **Matrix elements** ⟨m|H'|n⟩ determine everything

---

## Daily Checklist

- [ ] I can derive the first-order energy formula
- [ ] I understand how the wavefunction correction is constructed
- [ ] I can identify validity conditions
- [ ] I can apply to simple examples
- [ ] I recognize when degeneracy is a problem

---

## Preview: Day 450

Tomorrow we extend to second-order perturbation theory, which captures the leading non-trivial correction and reveals the importance of virtual transitions.

---

**Next:** [Day_450_Tuesday.md](Day_450_Tuesday.md) — Non-Degenerate Perturbation Theory II
