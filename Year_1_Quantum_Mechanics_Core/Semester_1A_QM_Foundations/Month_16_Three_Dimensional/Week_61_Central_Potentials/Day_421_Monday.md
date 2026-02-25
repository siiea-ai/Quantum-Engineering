# Day 421: 3D Schrödinger Equation

## Overview
**Day 421** | Year 1, Month 16, Week 61 | From 1D to 3D Quantum Mechanics

Today we extend quantum mechanics to three dimensions, developing the formalism for central potentials that will lead us to the hydrogen atom.

---

## Learning Objectives

By the end of today, you will be able to:
1. Write the 3D time-independent Schrödinger equation
2. Express the Laplacian in spherical coordinates
3. Separate variables for central potentials V(r)
4. Identify the angular equation as the eigenvalue problem for L²
5. Derive the radial equation structure
6. Understand the role of boundary conditions

---

## Core Content

### The 3D Schrödinger Equation

**Time-independent form:**
$$\hat{H}\psi = E\psi$$

$$-\frac{\hbar^2}{2m}\nabla^2\psi(\mathbf{r}) + V(\mathbf{r})\psi(\mathbf{r}) = E\psi(\mathbf{r})$$

### Spherical Coordinates

For central potentials V(r) that depend only on distance from origin:

$$\mathbf{r} = (r, \theta, \phi)$$

where:
- r ∈ [0, ∞): radial distance
- θ ∈ [0, π]: polar angle
- φ ∈ [0, 2π): azimuthal angle

### The Laplacian in Spherical Coordinates

$$\boxed{\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta\frac{\partial}{\partial \theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2}{\partial \phi^2}}$$

This can be written as:
$$\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial}{\partial r}\right) - \frac{\hat{L}^2}{\hbar^2 r^2}$$

where L̂² is the angular momentum squared operator!

### Angular Momentum Connection

Recall from Month 15:
$$\hat{L}^2 = -\hbar^2\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]$$

### Separation of Variables

**Ansatz:**
$$\psi(r, \theta, \phi) = R(r) \cdot Y(\theta, \phi)$$

Substituting into Schrödinger equation and separating:

**Angular equation:**
$$\hat{L}^2 Y(\theta, \phi) = \hbar^2 \lambda Y(\theta, \phi)$$

This is exactly the spherical harmonics equation! With λ = l(l+1):
$$Y(\theta, \phi) = Y_l^m(\theta, \phi)$$

**Radial equation:**
$$\boxed{\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) - \frac{l(l+1)}{r^2}R + \frac{2m}{\hbar^2}[E - V(r)]R = 0}$$

### The Full Wavefunction

$$\boxed{\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)}$$

**Quantum numbers:**
- n: principal quantum number (radial)
- l: orbital angular momentum (l = 0, 1, 2, ...)
- m: magnetic quantum number (m = -l, ..., +l)

### Normalization

$$\int_0^\infty |R_{nl}(r)|^2 r^2 dr = 1$$

$$\int |Y_l^m|^2 d\Omega = 1$$

Combined:
$$\int |\psi_{nlm}|^2 d^3r = \int_0^\infty |R_{nl}|^2 r^2 dr \int |Y_l^m|^2 d\Omega = 1$$

### Probability Density

The radial probability density is:
$$P(r) = |R_{nl}(r)|^2 r^2$$

This gives the probability of finding the particle between r and r + dr (integrated over angles).

---

## Quantum Computing Connection

### Quantum Simulation of 3D Systems

The 3D Schrödinger equation is central to:
- **Variational Quantum Eigensolver (VQE):** Finding ground state energies of molecules
- **Quantum chemistry simulation:** Electronic structure of atoms/molecules
- **Nuclear physics simulation:** Few-body nuclear systems

### Basis Encoding

Spherical harmonics provide a natural basis for encoding angular wavefunctions:
$$|\psi\rangle = \sum_{l,m} c_{lm} |l, m\rangle$$

This is an infinite-dimensional space that must be truncated for quantum computation.

---

## Worked Examples

### Example 1: Verifying Separation

**Problem:** Show that ψ(r,θ,φ) = R(r)Y_l^m(θ,φ) separates the 3D Schrödinger equation.

**Solution:**
Start with:
$$-\frac{\hbar^2}{2m}\nabla^2(RY) + V(r)RY = E \cdot RY$$

Using ∇²:
$$-\frac{\hbar^2}{2m}\left[\frac{Y}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) - \frac{R\hat{L}^2Y}{\hbar^2r^2}\right] + V(r)RY = ERY$$

Since L̂²Y = ℏ²l(l+1)Y:
$$-\frac{\hbar^2}{2m}\left[\frac{Y}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) - \frac{l(l+1)RY}{r^2}\right] + V(r)RY = ERY$$

Dividing by RY gives the radial equation.

### Example 2: Radial Probability

**Problem:** If R₁₀(r) = 2(1/a₀)^{3/2} e^{-r/a₀} for hydrogen, find the most probable radius.

**Solution:**
Radial probability: P(r) = |R|² r² = 4(1/a₀)³ r² e^{-2r/a₀}

Maximum when dP/dr = 0:
$$\frac{d}{dr}[r^2 e^{-2r/a_0}] = 2r e^{-2r/a_0} - \frac{2r^2}{a_0}e^{-2r/a_0} = 0$$

$$2r\left(1 - \frac{r}{a_0}\right) = 0$$

$$\boxed{r_{\text{most probable}} = a_0}$$

The most probable radius for the 1s state equals the Bohr radius!

---

## Practice Problems

### Direct Application
1. Write out the full 3D Schrödinger equation for a free particle (V = 0).
2. What are the allowed values of l and m for n = 3?
3. Calculate the volume element in spherical coordinates.

### Intermediate
4. Show that ∇² = (1/r²)∂/∂r(r²∂/∂r) - L̂²/(ℏ²r²).
5. For the hydrogen 2p state, how many distinct wavefunctions exist?
6. Verify that the radial equation reduces to the 1D case for l = 0.

### Challenging
7. Prove that the 3D Laplacian in spherical coordinates is Hermitian with respect to the measure r² dr dΩ.
8. Derive the commutation relation [∇², L̂²] = 0 directly.

---

## Computational Lab

```python
"""
Day 421: 3D Schrödinger Equation Visualization
Visualizing spherical harmonics and radial probability densities
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

# Physical constants
a0 = 1.0  # Bohr radius (natural units)

def hydrogen_radial(n, l, r):
    """
    Hydrogen radial wavefunction R_nl(r)
    Simplified for small n
    """
    if n == 1 and l == 0:  # 1s
        return 2 * (1/a0)**(3/2) * np.exp(-r/a0)
    elif n == 2 and l == 0:  # 2s
        rho = r / a0
        return (1/(2*np.sqrt(2))) * (1/a0)**(3/2) * (2 - rho) * np.exp(-rho/2)
    elif n == 2 and l == 1:  # 2p
        rho = r / a0
        return (1/(2*np.sqrt(6))) * (1/a0)**(3/2) * rho * np.exp(-rho/2)
    elif n == 3 and l == 0:  # 3s
        rho = 2*r / (3*a0)
        return (2/(81*np.sqrt(3))) * (1/a0)**(3/2) * (27 - 18*rho + 2*rho**2) * np.exp(-rho/2)
    else:
        return np.zeros_like(r)

# Plot radial probability densities
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

r = np.linspace(0.01, 20*a0, 500)

# 1s orbital
ax = axes[0, 0]
R_10 = hydrogen_radial(1, 0, r)
P_10 = np.abs(R_10)**2 * r**2
ax.plot(r/a0, P_10/np.max(P_10), 'b-', linewidth=2)
ax.axvline(x=1, color='r', linestyle='--', label='Bohr radius')
ax.set_xlabel('r/a₀')
ax.set_ylabel('Probability density (normalized)')
ax.set_title('1s: P(r) = |R₁₀|²r²')
ax.legend()
ax.grid(True, alpha=0.3)

# 2s orbital
ax = axes[0, 1]
R_20 = hydrogen_radial(2, 0, r)
P_20 = np.abs(R_20)**2 * r**2
ax.plot(r/a0, P_20/np.max(P_20), 'g-', linewidth=2)
ax.axvline(x=4, color='r', linestyle='--', label='<r> = 4a₀')
ax.set_xlabel('r/a₀')
ax.set_ylabel('Probability density (normalized)')
ax.set_title('2s: P(r) = |R₂₀|²r²')
ax.legend()
ax.grid(True, alpha=0.3)

# 2p orbital
ax = axes[1, 0]
R_21 = hydrogen_radial(2, 1, r)
P_21 = np.abs(R_21)**2 * r**2
ax.plot(r/a0, P_21/np.max(P_21), 'r-', linewidth=2)
ax.set_xlabel('r/a₀')
ax.set_ylabel('Probability density (normalized)')
ax.set_title('2p: P(r) = |R₂₁|²r²')
ax.grid(True, alpha=0.3)

# Comparison of all three
ax = axes[1, 1]
ax.plot(r/a0, P_10/np.max(P_10), 'b-', linewidth=2, label='1s')
ax.plot(r/a0, P_20/np.max(P_20), 'g-', linewidth=2, label='2s')
ax.plot(r/a0, P_21/np.max(P_21), 'r-', linewidth=2, label='2p')
ax.set_xlabel('r/a₀')
ax.set_ylabel('Probability density (normalized)')
ax.set_title('Comparison of Radial Probabilities')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day421_radial_probabilities.png', dpi=150)
plt.show()

# Spherical harmonic visualization
print("\n=== Spherical Harmonics |Y_l^m|² ===")

fig = plt.figure(figsize=(15, 10))

# Create grid for spherical surface
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)

harmonics = [
    (0, 0, 'Y₀⁰ (s orbital)'),
    (1, 0, 'Y₁⁰ (p_z orbital)'),
    (1, 1, 'Y₁¹ (p_x + ip_y)'),
    (2, 0, 'Y₂⁰ (d_{z²} orbital)'),
    (2, 1, 'Y₂¹'),
    (2, 2, 'Y₂² (d_{x²-y²} + id_{xy})')
]

for idx, (l, m, title) in enumerate(harmonics):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

    # Calculate spherical harmonic (absolute value squared)
    Y_lm = sph_harm(m, l, PHI, THETA)
    R = np.abs(Y_lm)**2

    # Convert to Cartesian for plotting
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # Color by sign of real part for visualization
    colors = np.real(Y_lm)
    colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)

    ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu(colors), alpha=0.8)
    ax.set_title(f'{title}\nl={l}, m={m}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Equal aspect ratio
    max_range = np.max([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()])
    ax.set_xlim([-max_range/2, max_range/2])
    ax.set_ylim([-max_range/2, max_range/2])
    ax.set_zlim([-max_range/2, max_range/2])

plt.tight_layout()
plt.savefig('day421_spherical_harmonics.png', dpi=150)
plt.show()

# Quantum numbers and degeneracy
print("\n=== Quantum Number Counting ===")
print("\nFor principal quantum number n:")
print("l can be: 0, 1, 2, ..., n-1")
print("m can be: -l, -l+1, ..., 0, ..., l-1, l")
print()

for n in range(1, 5):
    total_states = 0
    print(f"n = {n}:")
    for l in range(n):
        m_values = list(range(-l, l+1))
        num_m = 2*l + 1
        total_states += num_m
        l_symbol = ['s', 'p', 'd', 'f'][l] if l < 4 else f'l={l}'
        print(f"  l = {l} ({l_symbol}): m = {m_values}, {num_m} states")
    print(f"  Total states for n={n}: {total_states} = n² = {n**2}")
    print()

print("For hydrogen (without spin): degeneracy of level n is n²")
print("With spin: degeneracy is 2n² (two spin states per orbital)")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| 3D Schrödinger | -ℏ²∇²ψ/2m + Vψ = Eψ |
| Laplacian | ∇² = (1/r²)∂_r(r²∂_r) - L̂²/(ℏ²r²) |
| Separation | ψ(r,θ,φ) = R(r)Y_l^m(θ,φ) |
| Angular equation | L̂²Y = ℏ²l(l+1)Y |
| Normalization | ∫|ψ|²d³r = ∫|R|²r²dr × ∫|Y|²dΩ = 1 |
| Radial probability | P(r) = |R(r)|²r² |

### Key Insights

1. **Separation works** because central potentials V(r) have spherical symmetry
2. **Angular part** is solved by spherical harmonics from Month 15
3. **Radial part** becomes a 1D-like equation with effective potential
4. **Quantum numbers** (n, l, m) arise naturally from separation
5. **Normalization** includes r² Jacobian from spherical coordinates

---

## Daily Checklist

- [ ] I can write the 3D Schrödinger equation
- [ ] I understand the spherical coordinate Laplacian
- [ ] I can separate variables for central potentials
- [ ] I recognize the angular momentum connection
- [ ] I know what quantum numbers label 3D states
- [ ] I can normalize a 3D wavefunction

---

## Preview: Day 422

Tomorrow we focus on the **radial equation**, introducing the substitution u = rR that converts it to a more familiar 1D form. We'll identify the centrifugal barrier term and understand its physical significance.

---

**Next:** [Day_422_Tuesday.md](Day_422_Tuesday.md) — The Radial Equation
