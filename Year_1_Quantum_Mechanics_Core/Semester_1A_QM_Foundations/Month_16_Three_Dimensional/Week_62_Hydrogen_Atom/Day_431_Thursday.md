# Day 431: Hydrogen Wavefunctions

## Overview
**Day 431** | Year 1, Month 16, Week 62 | Complete Atomic Orbitals

Today we construct the complete hydrogen wavefunctions by combining radial and angular parts, visualizing the famous atomic orbitals.

---

## Learning Objectives

By the end of today, you will be able to:
1. Write complete wavefunctions ψ_{nlm}(r,θ,φ)
2. Construct real linear combinations (px, py, etc.)
3. Visualize probability densities |ψ|²
4. Interpret orbital shapes physically
5. Calculate probability in regions of space
6. Connect to chemistry and bonding

---

## Core Content

### Complete Wavefunctions

$$\boxed{\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)}$$

### Explicit Formulas

**1s orbital (n=1, l=0, m=0):**
$$\psi_{100} = \frac{1}{\sqrt{\pi}a_0^{3/2}} e^{-r/a_0}$$

**2s orbital (n=2, l=0, m=0):**
$$\psi_{200} = \frac{1}{4\sqrt{2\pi}a_0^{3/2}}\left(2 - \frac{r}{a_0}\right)e^{-r/(2a_0)}$$

**2p orbitals (n=2, l=1):**
$$\psi_{210} = \frac{1}{4\sqrt{2\pi}a_0^{3/2}}\frac{r}{a_0}e^{-r/(2a_0)}\cos\theta$$

$$\psi_{21\pm1} = \mp\frac{1}{8\sqrt{\pi}a_0^{3/2}}\frac{r}{a_0}e^{-r/(2a_0)}\sin\theta\, e^{\pm i\phi}$$

### Real Orbitals

For chemistry, use real linear combinations:

$$p_z = \psi_{210}$$

$$p_x = \frac{1}{\sqrt{2}}(\psi_{21-1} - \psi_{211}) = \frac{1}{4\sqrt{2\pi}a_0^{3/2}}\frac{r}{a_0}e^{-r/(2a_0)}\sin\theta\cos\phi$$

$$p_y = \frac{i}{\sqrt{2}}(\psi_{21-1} + \psi_{211}) = \frac{1}{4\sqrt{2\pi}a_0^{3/2}}\frac{r}{a_0}e^{-r/(2a_0)}\sin\theta\sin\phi$$

### Probability Density

$$|\psi_{nlm}|^2 = |R_{nl}|^2 |Y_l^m|^2$$

For s-orbitals: spherically symmetric
For p-orbitals: dumbbell shapes
For d-orbitals: cloverleaf and dz² shapes

### Probability in Regions

Probability of finding electron at distance r to r+dr:
$$P(r)dr = |R_{nl}(r)|^2 r^2 dr$$

### Most Probable Radius

For 1s: r_mp = a₀ (maximum of r²|R|²)
For 2s: r_mp = 5.24 a₀

---

## Quantum Computing Connection

### Orbital Basis for Quantum Chemistry

Hydrogen orbitals are basis functions for:
- **Molecular orbitals** (LCAO method)
- **VQE ansätze** for molecules
- **Slater-type orbitals** (STO) in quantum chemistry

### Qubit Mapping

Orbital occupation maps to qubits:
- |0⟩ = unoccupied
- |1⟩ = occupied
- Enables quantum simulation of molecular systems

---

## Worked Examples

### Example 1: Normalization Check

**Problem:** Verify ψ₁₀₀ is normalized.

**Solution:**
$$\int|\psi_{100}|^2 d^3r = \frac{1}{\pi a_0^3}\int_0^\infty e^{-2r/a_0} r^2 dr \int d\Omega$$

$$= \frac{4\pi}{\pi a_0^3} \cdot \frac{a_0^3}{4} = 1 \quad \checkmark$$

### Example 2: Probability Inside Bohr Radius

**Problem:** What fraction of 1s electron density is within r < a₀?

**Solution:**
$$P(r < a_0) = \frac{4}{a_0^3}\int_0^{a_0} r^2 e^{-2r/a_0} dr$$

Using integration by parts:
$$= 1 - 5e^{-2} \approx 0.32$$

About 32% of the electron is within one Bohr radius.

---

## Practice Problems

1. Write out ψ₃₀₀ explicitly.
2. Show that ψ₂₀₀ vanishes at r = 2a₀.
3. Find the most probable radius for the 2p orbital.
4. Calculate the probability of finding the electron beyond 2a₀ for 1s.

---

## Computational Lab

```python
"""
Day 431: Hydrogen Wavefunctions and Orbital Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre, factorial
from mpl_toolkits.mplot3d import Axes3D

def hydrogen_wavefunction(n, l, m, r, theta, phi, a0=1):
    """Complete hydrogen wavefunction ψ_nlm"""
    # Radial part
    rho = 2*r/(n*a0)
    norm_r = np.sqrt((2/(n*a0))**3 * factorial(n-l-1)/(2*n*factorial(n+l)**3))
    L = genlaguerre(n-l-1, 2*l+1)(rho)
    R = norm_r * rho**l * np.exp(-rho/2) * L

    # Angular part (scipy uses physics convention)
    Y = sph_harm(m, l, phi, theta)

    return R * Y

# Create 2D cross-section plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Create grid
x = np.linspace(-10, 10, 200)
z = np.linspace(-10, 10, 200)
X, Z = np.meshgrid(x, z)
Y_plane = np.zeros_like(X)

R = np.sqrt(X**2 + Z**2)
THETA = np.arctan2(np.sqrt(X**2), Z)
PHI = np.zeros_like(R)

orbitals = [
    (1, 0, 0, '1s'),
    (2, 0, 0, '2s'),
    (2, 1, 0, '2p_z'),
    (3, 0, 0, '3s'),
    (3, 1, 0, '3p_z'),
    (3, 2, 0, '3d_{z²}'),
]

for ax, (n, l, m, name) in zip(axes.flat, orbitals):
    psi = hydrogen_wavefunction(n, l, m, R, THETA, PHI)
    prob = np.abs(psi)**2

    # Use symmetric log scale for better visualization
    im = ax.contourf(X, Z, prob, levels=50, cmap='hot')
    ax.set_xlabel('x / a₀')
    ax.set_ylabel('z / a₀')
    ax.set_title(f'{name} orbital |ψ|²')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('day431_orbitals.png', dpi=150)
plt.show()

print("=== Hydrogen Orbital Properties ===")
print("\nOrbital   Nodes   Shape")
print("-" * 35)
print("1s        0       Spherical")
print("2s        1       Spherical with node")
print("2p        0       Dumbbell along axis")
print("3s        2       Spherical with 2 nodes")
print("3d        0       Cloverleaf/dz² shapes")
```

---

## Summary

| Orbital | Formula | Shape |
|---------|---------|-------|
| 1s | (1/√π)(1/a₀)^{3/2} e^{-r/a₀} | Sphere |
| 2s | (1/4√2π)(1/a₀)^{3/2}(2-r/a₀)e^{-r/2a₀} | Sphere + node |
| 2p_z | ∝ r e^{-r/2a₀} cos θ | Dumbbell |

---

**Next:** [Day_432_Friday.md](Day_432_Friday.md) — Degeneracy & Symmetry
