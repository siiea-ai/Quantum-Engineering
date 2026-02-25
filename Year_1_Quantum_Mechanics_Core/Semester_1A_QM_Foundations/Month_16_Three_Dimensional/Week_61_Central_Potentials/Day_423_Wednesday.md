# Day 423: Free Particle in 3D

## Overview
**Day 423** | Year 1, Month 16, Week 61 | Spherical Bessel Functions and Plane Waves

Today we solve for the free particle in three dimensions, introducing spherical Bessel functions that will appear throughout atomic and nuclear physics.

---

## Learning Objectives

By the end of today, you will be able to:
1. Solve the radial equation for V(r) = 0
2. Identify spherical Bessel functions of the first and second kind
3. Understand the partial wave expansion of a plane wave
4. Apply asymptotic forms for large arguments
5. Connect to scattering theory foundations
6. Recognize these functions in physics applications

---

## Core Content

### The Free-Particle Radial Equation

With V(r) = 0 and u = rR:
$$-\frac{\hbar^2}{2m}\frac{d^2u}{dr^2} + \frac{\hbar^2 l(l+1)}{2mr^2}u = Eu$$

Defining k² = 2mE/ℏ² (for E > 0):
$$\frac{d^2u}{dr^2} + \left[k^2 - \frac{l(l+1)}{r^2}\right]u = 0$$

### Change of Variable: ρ = kr

Let ρ = kr:
$$\frac{d^2u}{d\rho^2} + \left[1 - \frac{l(l+1)}{\rho^2}\right]u = 0$$

This is the **spherical Bessel equation**.

### Solutions: Spherical Bessel Functions

**Spherical Bessel function of the first kind:**
$$j_l(\rho) = (-\rho)^l \left(\frac{1}{\rho}\frac{d}{d\rho}\right)^l \frac{\sin\rho}{\rho}$$

**Spherical Bessel function of the second kind (Neumann):**
$$n_l(\rho) = -(-\rho)^l \left(\frac{1}{\rho}\frac{d}{d\rho}\right)^l \frac{\cos\rho}{\rho}$$

### Explicit Forms

| l | j_l(ρ) | n_l(ρ) |
|---|--------|--------|
| 0 | sin(ρ)/ρ | -cos(ρ)/ρ |
| 1 | sin(ρ)/ρ² - cos(ρ)/ρ | -cos(ρ)/ρ² - sin(ρ)/ρ |
| 2 | (3/ρ³ - 1/ρ)sin(ρ) - (3/ρ²)cos(ρ) | -(3/ρ³ - 1/ρ)cos(ρ) - (3/ρ²)sin(ρ) |

### Asymptotic Behavior

**As ρ → 0:**
$$j_l(\rho) \approx \frac{\rho^l}{(2l+1)!!}$$
$$n_l(\rho) \approx -\frac{(2l-1)!!}{\rho^{l+1}}$$

Note: (2l+1)!! = 1·3·5·...·(2l+1)

**As ρ → ∞:**
$$j_l(\rho) \approx \frac{1}{\rho}\sin\left(\rho - \frac{l\pi}{2}\right)$$
$$n_l(\rho) \approx -\frac{1}{\rho}\cos\left(\rho - \frac{l\pi}{2}\right)$$

### Spherical Hankel Functions

Linear combinations useful for outgoing/incoming waves:
$$h_l^{(1)}(\rho) = j_l(\rho) + in_l(\rho) \xrightarrow{\rho \to \infty} \frac{e^{i(\rho - l\pi/2)}}{\rho}$$
$$h_l^{(2)}(\rho) = j_l(\rho) - in_l(\rho) \xrightarrow{\rho \to \infty} \frac{e^{-i(\rho - l\pi/2)}}{\rho}$$

### Plane Wave Expansion

A plane wave can be expanded in partial waves:
$$\boxed{e^{ikz} = \sum_{l=0}^{\infty} i^l (2l+1) j_l(kr) P_l(\cos\theta)}$$

This is the **Rayleigh expansion**, fundamental to scattering theory.

### Free Particle Wavefunctions

General solution:
$$\psi_{klm}(r, \theta, \phi) = \left[A_l j_l(kr) + B_l n_l(kr)\right] Y_l^m(\theta, \phi)$$

For states regular at origin: B_l = 0 (n_l diverges as ρ → 0 for l ≥ 0).

---

## Quantum Computing Connection

### Quantum Simulation of Scattering

Spherical Bessel functions appear in:
- **Quantum simulation of nuclear reactions**
- **Electronic scattering in materials**
- **Collision processes on quantum computers**

### Encoding Angular Momentum

The partial wave expansion provides a natural truncation scheme for quantum simulation of scattering processes.

---

## Worked Examples

### Example 1: Verify j₀(ρ)

**Problem:** Show that j₀(ρ) = sin(ρ)/ρ satisfies the l = 0 spherical Bessel equation.

**Solution:**
The equation for l = 0:
$$\frac{d^2u}{d\rho^2} + u = 0 \quad \text{where } u = \rho j_0(\rho) = \sin\rho$$

Check: d²(sin ρ)/dρ² = -sin ρ, so:
$$-\sin\rho + \sin\rho = 0 \quad \checkmark$$

### Example 2: Small Argument Expansion

**Problem:** Find j₁(ρ) for small ρ.

**Solution:**
$$j_1(\rho) = \frac{\sin\rho}{\rho^2} - \frac{\cos\rho}{\rho}$$

Taylor expand:
$$\sin\rho \approx \rho - \frac{\rho^3}{6} + O(\rho^5)$$
$$\cos\rho \approx 1 - \frac{\rho^2}{2} + O(\rho^4)$$

$$j_1(\rho) \approx \frac{1}{\rho^2}\left(\rho - \frac{\rho^3}{6}\right) - \frac{1}{\rho}\left(1 - \frac{\rho^2}{2}\right)$$
$$= \frac{1}{\rho} - \frac{\rho}{6} - \frac{1}{\rho} + \frac{\rho}{2}$$
$$= \frac{\rho}{3}$$

So:
$$\boxed{j_1(\rho) \approx \frac{\rho}{3} = \frac{\rho}{3!!}}$$

### Example 3: Plane Wave Coefficient

**Problem:** What is the l = 0 contribution to the plane wave expansion?

**Solution:**
From the Rayleigh expansion with l = 0:
$$e^{ikz}|_{l=0} = i^0 (2 \cdot 0 + 1) j_0(kr) P_0(\cos\theta) = j_0(kr) = \frac{\sin(kr)}{kr}$$

The s-wave component is isotropic (no angular dependence).

---

## Practice Problems

### Direct Application
1. Verify that j₀(0) = 1 and j_l(0) = 0 for l > 0.
2. Write out j₂(ρ) explicitly.
3. What is the asymptotic phase of j₃(kr)?

### Intermediate
4. Show that n₀(ρ) = -cos(ρ)/ρ by direct calculation.
5. Prove the recurrence relation: (2l+1)j_l = ρ(j_{l-1} + j_{l+1}).
6. Calculate the first two terms of the plane wave expansion.

### Challenging
7. Derive the Rayleigh expansion using generating function methods.
8. Show that ∫₀^∞ j_l(kr)j_l(k'r)r² dr = (π/2k²)δ(k-k').

---

## Computational Lab

```python
"""
Day 423: Spherical Bessel Functions and Plane Wave Expansion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
from scipy.special import legendre

# Plot spherical Bessel functions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

rho = np.linspace(0.01, 20, 1000)

# j_l(ρ) - first kind
ax = axes[0, 0]
for l in range(5):
    jl = spherical_jn(l, rho)
    ax.plot(rho, jl, linewidth=2, label=f'j_{l}(ρ)')

ax.set_xlabel('ρ', fontsize=12)
ax.set_ylabel('j_l(ρ)', fontsize=12)
ax.set_title('Spherical Bessel Functions (First Kind)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)
ax.set_ylim(-0.4, 1.1)
ax.axhline(y=0, color='black', linewidth=0.5)

# n_l(ρ) - second kind (Neumann)
ax = axes[0, 1]
for l in range(5):
    nl = spherical_yn(l, rho)
    ax.plot(rho, nl, linewidth=2, label=f'n_{l}(ρ)')

ax.set_xlabel('ρ', fontsize=12)
ax.set_ylabel('n_l(ρ)', fontsize=12)
ax.set_title('Spherical Bessel Functions (Second Kind/Neumann)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)
ax.set_ylim(-1, 0.5)
ax.axhline(y=0, color='black', linewidth=0.5)

# Small-ρ behavior
ax = axes[1, 0]
rho_small = np.linspace(0.01, 5, 500)

for l in range(4):
    jl = spherical_jn(l, rho_small)
    asymptotic = rho_small**l / np.math.factorial2(2*l + 1)

    ax.plot(rho_small, jl, linewidth=2, label=f'j_{l}(ρ) exact')
    ax.plot(rho_small, asymptotic, '--', linewidth=1.5, label=f'ρ^{l}/{(2*l+1)}!!')

ax.set_xlabel('ρ', fontsize=12)
ax.set_ylabel('j_l(ρ)', fontsize=12)
ax.set_title('Small-ρ Asymptotic Behavior', fontsize=14)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)
ax.set_ylim(-0.1, 1.1)

# Large-ρ behavior
ax = axes[1, 1]
rho_large = np.linspace(10, 30, 500)

for l in [0, 1, 2]:
    jl = spherical_jn(l, rho_large)
    asymptotic = np.sin(rho_large - l*np.pi/2) / rho_large

    ax.plot(rho_large, jl, linewidth=2, label=f'j_{l}(ρ) exact')
    ax.plot(rho_large, asymptotic, '--', linewidth=1.5,
            label=f'sin(ρ-{l}π/2)/ρ')

ax.set_xlabel('ρ', fontsize=12)
ax.set_ylabel('j_l(ρ)', fontsize=12)
ax.set_title('Large-ρ Asymptotic Behavior', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(10, 30)
ax.set_ylim(-0.15, 0.15)
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('day423_spherical_bessel.png', dpi=150)
plt.show()

# Plane wave expansion visualization
print("\n=== Plane Wave Expansion ===")
print("e^{ikz} = Σ i^l (2l+1) j_l(kr) P_l(cos θ)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Parameters
k = 1.0
r = 5.0  # Fixed radius

theta = np.linspace(0, np.pi, 200)
cos_theta = np.cos(theta)

# Exact plane wave at z = r cos(θ)
plane_wave_exact = np.exp(1j * k * r * cos_theta)

# Partial wave expansion
def partial_wave_sum(kr, cos_theta, l_max):
    """Sum partial waves up to l_max"""
    result = np.zeros_like(cos_theta, dtype=complex)
    for l in range(l_max + 1):
        jl = spherical_jn(l, kr)
        Pl = legendre(l)(cos_theta)
        result += (1j)**l * (2*l + 1) * jl * Pl
    return result

# Plot convergence
ax = axes[0, 0]
for l_max in [0, 1, 3, 5, 10]:
    pw_approx = partial_wave_sum(k*r, cos_theta, l_max)
    ax.plot(theta * 180/np.pi, np.real(pw_approx), linewidth=2,
            label=f'l_max = {l_max}')

ax.plot(theta * 180/np.pi, np.real(plane_wave_exact), 'k--', linewidth=2,
        label='Exact', alpha=0.7)
ax.set_xlabel('θ (degrees)', fontsize=12)
ax.set_ylabel('Re[ψ]', fontsize=12)
ax.set_title(f'Partial Wave Expansion Convergence (kr = {k*r})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Angular distribution of each partial wave
ax = axes[0, 1]
for l in range(5):
    Pl = legendre(l)(cos_theta)
    ax.plot(theta * 180/np.pi, Pl, linewidth=2, label=f'P_{l}(cos θ)')

ax.set_xlabel('θ (degrees)', fontsize=12)
ax.set_ylabel('P_l(cos θ)', fontsize=12)
ax.set_title('Legendre Polynomials (Angular Part)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)

# Radial part at different energies
ax = axes[1, 0]
r_range = np.linspace(0.1, 15, 500)

for l in range(4):
    for ki, k in enumerate([0.5, 1.0, 2.0]):
        if ki == 1:  # Only label for k=1
            ax.plot(r_range, spherical_jn(l, k*r_range), linewidth=2,
                    label=f'l = {l}')
        else:
            ax.plot(r_range, spherical_jn(l, k*r_range), linewidth=1, alpha=0.5)

ax.set_xlabel('r (a.u.)', fontsize=12)
ax.set_ylabel('j_l(kr)', fontsize=12)
ax.set_title('Radial Functions at Different k', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 15)
ax.axhline(y=0, color='black', linewidth=0.5)

# 2D visualization of partial waves
ax = axes[1, 1]
# Create polar plot of |e^{ikz}|² and first few partial waves
from matplotlib.colors import Normalize

# Create r-θ grid
r_grid = np.linspace(0.1, 10, 100)
theta_grid = np.linspace(0, 2*np.pi, 100)
R, THETA = np.meshgrid(r_grid, theta_grid)

# Plane wave e^{ikz} where z = r cos(θ)
psi_plane = np.exp(1j * k * R * np.cos(THETA))

# Convert to Cartesian for plotting
X = R * np.cos(THETA)
Y = R * np.sin(THETA)

# Plot real part of plane wave
im = ax.pcolormesh(X, Y, np.real(psi_plane), cmap='RdBu', shading='auto',
                   norm=Normalize(-1.5, 1.5))
plt.colorbar(im, ax=ax, label='Re[ψ]')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(f'Plane Wave e^{{ikz}} (k = {k})', fontsize=14)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('day423_plane_wave_expansion.png', dpi=150)
plt.show()

# Numerical verification
print("\n=== Numerical Verification ===")
print("\nVerifying explicit formula for j_0(ρ) = sin(ρ)/ρ:")
test_rho = 2.5
j0_scipy = spherical_jn(0, test_rho)
j0_formula = np.sin(test_rho) / test_rho
print(f"ρ = {test_rho}: scipy = {j0_scipy:.10f}, sin(ρ)/ρ = {j0_formula:.10f}")
print(f"Difference: {abs(j0_scipy - j0_formula):.2e}")

print("\nVerifying explicit formula for j_1(ρ) = sin(ρ)/ρ² - cos(ρ)/ρ:")
j1_scipy = spherical_jn(1, test_rho)
j1_formula = np.sin(test_rho)/test_rho**2 - np.cos(test_rho)/test_rho
print(f"ρ = {test_rho}: scipy = {j1_scipy:.10f}, formula = {j1_formula:.10f}")
print(f"Difference: {abs(j1_scipy - j1_formula):.2e}")
```

---

## Summary

### Key Formulas

| Function | Definition | Small ρ | Large ρ |
|----------|------------|---------|---------|
| j_l(ρ) | (-ρ)^l (d/ρdρ)^l sin(ρ)/ρ | ρ^l/(2l+1)!! | sin(ρ-lπ/2)/ρ |
| n_l(ρ) | -(-ρ)^l (d/ρdρ)^l cos(ρ)/ρ | -(2l-1)!!/ρ^(l+1) | -cos(ρ-lπ/2)/ρ |
| h_l^(1) | j_l + in_l | — | e^{i(ρ-lπ/2)}/ρ |

**Plane wave expansion:**
$$e^{ikz} = \sum_{l=0}^{\infty} i^l (2l+1) j_l(kr) P_l(\cos\theta)$$

### Key Insights

1. **Spherical Bessel functions** solve the free-particle radial equation
2. **j_l regular** at origin, n_l singular (use j_l for interior solutions)
3. **Plane waves** can be decomposed into angular momentum eigenstates
4. **Partial wave expansion** is essential for scattering theory
5. **Asymptotic forms** give phase shifts and scattering amplitudes

---

## Daily Checklist

- [ ] I can solve the radial equation for a free particle
- [ ] I know the explicit forms of j₀, j₁, n₀, n₁
- [ ] I understand small-ρ and large-ρ asymptotics
- [ ] I can write the plane wave expansion
- [ ] I know when to use j_l vs n_l vs h_l
- [ ] I see the connection to scattering theory

---

## Preview: Day 424

Tomorrow we solve the **infinite spherical well** — a particle confined to r < a. The boundary condition j_l(ka) = 0 quantizes the energy levels.

---

**Next:** [Day_424_Thursday.md](Day_424_Thursday.md) — Infinite Spherical Well
