# Day 422: The Radial Equation

## Overview
**Day 422** | Year 1, Month 16, Week 61 | Effective Potential and Centrifugal Barrier

Today we transform the radial equation into a form analogous to 1D quantum mechanics, revealing the crucial centrifugal barrier that keeps particles away from the origin for l > 0.

---

## Learning Objectives

By the end of today, you will be able to:
1. Transform the radial equation using u = rR
2. Identify the effective potential V_eff(r)
3. Explain the physical origin of the centrifugal barrier
4. Apply boundary conditions at r = 0 and r → ∞
5. Analyze the behavior of solutions near the origin
6. Understand when bound states exist

---

## Core Content

### The Radial Equation (Standard Form)

From yesterday:
$$\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) + \frac{2m}{\hbar^2}[E - V(r)]R - \frac{l(l+1)}{r^2}R = 0$$

This is complicated because of the r² factor.

### The Substitution u(r) = rR(r)

**Key transformation:** Define u(r) = rR(r), so R(r) = u(r)/r.

Computing derivatives:
$$\frac{dR}{dr} = \frac{1}{r}\frac{du}{dr} - \frac{u}{r^2}$$

$$\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) = \frac{1}{r}\frac{d^2u}{dr^2}$$

### The Simplified Radial Equation

$$\boxed{-\frac{\hbar^2}{2m}\frac{d^2u}{dr^2} + V_{\text{eff}}(r)u(r) = Eu(r)}$$

This looks exactly like the 1D Schrödinger equation!

### Effective Potential

$$\boxed{V_{\text{eff}}(r) = V(r) + \frac{\hbar^2 l(l+1)}{2mr^2}}$$

The extra term is the **centrifugal barrier**:
$$V_{\text{centrifugal}} = \frac{\hbar^2 l(l+1)}{2mr^2} = \frac{L^2}{2mr^2}$$

### Physical Interpretation

**Classical analogy:** In classical mechanics, the effective potential for radial motion is:
$$V_{\text{eff}}^{\text{classical}} = V(r) + \frac{L^2}{2mr^2}$$

The centrifugal term represents the "centrifugal force" pushing outward for rotating particles.

**Quantum origin:** Even with l = 0, there's a quantum effect! The l(l+1) factor (not l²) is a quantum correction.

### Behavior at r → 0

**For l = 0:**
- R(r) ∼ constant (s-waves can penetrate to origin)
- u(r) ∼ r

**For l > 0:**
- R(r) ∼ r^l (suppressed at origin)
- u(r) ∼ r^(l+1)

The centrifugal barrier repels particles from the origin!

### Boundary Conditions

**At r = 0:**
$$u(0) = 0$$

This ensures R = u/r is finite. Also, u ∼ r^(l+1) for small r.

**At r → ∞:**
- Bound states: u → 0 (square-integrable)
- Scattering states: u oscillates (continuum)

### Normalization in Terms of u

$$\int_0^\infty |R|^2 r^2 dr = \int_0^\infty |u|^2 dr = 1$$

The u normalization is just like 1D!

### Nodes of Radial Function

The radial function R_nl has:
$$\text{Number of nodes} = n - l - 1$$

(where n is principal quantum number for hydrogen-like atoms)

---

## Quantum Computing Connection

### Quantum Simulation of Effective Potentials

The effective potential concept is crucial for:
- **Molecular simulation:** Effective internuclear potentials
- **VQE ansätze:** Choosing appropriate radial basis functions
- **Quantum optimization:** Understanding potential landscapes

### Classical-Quantum Correspondence

The centrifugal barrier shows where classical and quantum mechanics agree (L²/2mr²) and disagree (l(l+1) vs l²).

---

## Worked Examples

### Example 1: Centrifugal Barrier Height

**Problem:** For an electron with l = 1 at r = a₀, what is the centrifugal potential energy?

**Solution:**
$$V_{\text{centrifugal}} = \frac{\hbar^2 l(l+1)}{2m_e r^2} = \frac{\hbar^2 \cdot 2}{2m_e a_0^2}$$

Using a₀ = ℏ²/(m_e e²) and E_H = e²/a₀ = 27.2 eV:
$$V_{\text{centrifugal}} = \frac{\hbar^2}{m_e a_0^2} = \frac{e^2}{a_0} = E_H = 27.2 \text{ eV}$$

This is the same order as the Coulomb binding energy!

### Example 2: Small-r Behavior

**Problem:** Show that u(r) ∼ r^(l+1) satisfies the radial equation near r = 0 for V(r) finite.

**Solution:**
Near r = 0, the centrifugal term dominates:
$$-\frac{\hbar^2}{2m}\frac{d^2u}{dr^2} + \frac{\hbar^2 l(l+1)}{2mr^2}u \approx 0$$

Try u = r^s:
$$-s(s-1)r^{s-2} + l(l+1)r^{s-2} = 0$$
$$s(s-1) = l(l+1)$$
$$s = l+1 \quad \text{or} \quad s = -l$$

The solution s = -l diverges at origin, so:
$$\boxed{u(r) \sim r^{l+1} \text{ as } r \to 0}$$

### Example 3: Effective Potential for Hydrogen

**Problem:** Sketch V_eff(r) for hydrogen with l = 0, 1, 2.

**Solution:**
$$V_{\text{eff}}(r) = -\frac{e^2}{r} + \frac{\hbar^2 l(l+1)}{2m_e r^2}$$

- l = 0: Pure Coulomb, attractive everywhere, V → -∞ at r = 0
- l = 1: Barrier at small r, minimum at intermediate r
- l = 2: Higher barrier, minimum pushed outward

The minimum occurs at:
$$\frac{d V_{\text{eff}}}{dr} = \frac{e^2}{r^2} - \frac{\hbar^2 l(l+1)}{m_e r^3} = 0$$
$$r_{\text{min}} = \frac{\hbar^2 l(l+1)}{m_e e^2} = l(l+1) a_0$$

---

## Practice Problems

### Direct Application
1. What is V_eff(r) for a free particle (V = 0)?
2. Write u(r) for R₁₀ = 2(1/a₀)^{3/2} e^{-r/a₀}.
3. How many radial nodes does R₃₂ have?

### Intermediate
4. Find the position of the centrifugal barrier maximum for a finite square well.
5. Show that the radial kinetic energy operator is -(ℏ²/2m)d²/dr² when acting on u.
6. For what value of l does the centrifugal energy equal the Coulomb energy at r = a₀?

### Challenging
7. Prove that the transformation u = rR makes the kinetic energy operator Hermitian with respect to the simple inner product ∫u*v dr.
8. Analyze the effective potential for the Yukawa potential V(r) = -V₀ e^{-r/a}/r.

---

## Computational Lab

```python
"""
Day 422: Effective Potential and Centrifugal Barrier
Visualizing how angular momentum creates a barrier
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants (atomic units: ℏ = m_e = e = 1)
a0 = 1.0  # Bohr radius

def coulomb_potential(r):
    """Hydrogen Coulomb potential (atomic units)"""
    return -1.0 / r

def centrifugal_term(r, l):
    """Centrifugal barrier term"""
    return l * (l + 1) / (2 * r**2)

def effective_potential(r, l):
    """Total effective potential for hydrogen"""
    return coulomb_potential(r) + centrifugal_term(r, l)

# Plot effective potentials for different l
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Effective potential for hydrogen
ax = axes[0]
r = np.linspace(0.1, 15, 1000)

colors = ['blue', 'orange', 'green', 'red', 'purple']
for l in range(5):
    V_eff = effective_potential(r, l)
    ax.plot(r, V_eff, color=colors[l], linewidth=2, label=f'l = {l}')

# Add energy levels for n=1,2,3
E_levels = [-0.5, -0.125, -0.0556]  # -1/(2n²) in atomic units
for n, E in enumerate(E_levels, 1):
    ax.axhline(y=E, color='gray', linestyle='--', alpha=0.5)
    ax.text(14, E + 0.02, f'E_{n}', fontsize=10)

ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('V_eff (Hartree)', fontsize=12)
ax.set_title('Effective Potential for Hydrogen', fontsize=14)
ax.set_xlim(0, 15)
ax.set_ylim(-1, 0.5)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)

# Panel 2: Breakdown of effective potential
ax = axes[1]
r = np.linspace(0.2, 10, 500)
l = 2  # Focus on l=2

V_coulomb = coulomb_potential(r)
V_centrifugal = centrifugal_term(r, l)
V_total = effective_potential(r, l)

ax.plot(r, V_coulomb, 'b--', linewidth=2, label='Coulomb: -1/r')
ax.plot(r, V_centrifugal, 'r--', linewidth=2, label=f'Centrifugal: l(l+1)/2r² (l={l})')
ax.plot(r, V_total, 'k-', linewidth=3, label='Total V_eff')

# Find and mark minimum
idx_min = np.argmin(V_total[r > 1])
r_min = r[r > 1][idx_min]
V_min = V_total[r > 1][idx_min]
ax.plot(r_min, V_min, 'go', markersize=10, label=f'Minimum at r = {r_min:.2f} a₀')

ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('V (Hartree)', fontsize=12)
ax.set_title(f'Components of V_eff for l = {l}', fontsize=14)
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 1)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('day422_effective_potential.png', dpi=150)
plt.show()

# Radial wavefunction behavior near origin
print("\n=== Behavior of u(r) near r = 0 ===")
print("u(r) ~ r^(l+1) for small r")
print()

fig, ax = plt.subplots(figsize=(10, 6))

r = np.linspace(0, 3, 500)

for l in range(4):
    # Small-r behavior
    u_small_r = r**(l+1)
    u_small_r = u_small_r / np.max(u_small_r)  # Normalize for display
    ax.plot(r, u_small_r, linewidth=2, label=f'l = {l}: u ~ r^{l+1}')

ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('u(r) (normalized)', fontsize=12)
ax.set_title('Radial Function Behavior Near Origin', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('day422_small_r_behavior.png', dpi=150)
plt.show()

# Numerical solution of radial equation
print("\n=== Numerical Solution of Radial Equation ===")
print("Solving for hydrogen ground state (n=1, l=0)")

from scipy.integrate import odeint

def radial_schrodinger(y, r, E, l):
    """
    Radial Schrödinger equation as first-order system
    y = [u, du/dr]
    d²u/dr² = (2m/ℏ²)[V_eff - E]u = 2[V_eff - E]u (atomic units)
    """
    u, du_dr = y
    if r < 0.01:
        r = 0.01  # Regularize near origin

    V_eff = effective_potential(r, l)
    d2u_dr2 = 2 * (V_eff - E) * u

    return [du_dr, d2u_dr2]

# Shooting method parameters
r_max = 30
r_points = np.linspace(0.001, r_max, 3000)

# Initial conditions: u ~ r^(l+1), so for l=0: u ~ r, du/dr ~ 1
l = 0
u0 = 0.001  # Small starting value
du0 = 1.0   # Derivative

# Solve for several trial energies
fig, ax = plt.subplots(figsize=(10, 6))

E_trials = [-0.6, -0.5, -0.4, -0.3]  # Around E_1 = -0.5 Hartree
colors = ['blue', 'green', 'red', 'purple']

for E, color in zip(E_trials, colors):
    sol = odeint(radial_schrodinger, [u0, du0], r_points, args=(E, l))
    u = sol[:, 0]

    # Normalize for display
    u_norm = u / np.max(np.abs(u[:1000]))  # Normalize using early part

    ax.plot(r_points, u_norm, color=color, linewidth=2, label=f'E = {E} Hartree')

# Plot exact solution
u_exact = r_points * np.exp(-r_points)
u_exact = u_exact / np.max(u_exact)
ax.plot(r_points, u_exact, 'k--', linewidth=2, label='Exact: u ~ r exp(-r)')

ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('u(r) (normalized)', fontsize=12)
ax.set_title('Shooting Method: Finding Bound State Energy', fontsize=14)
ax.set_xlim(0, 20)
ax.set_ylim(-0.5, 1.5)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('day422_shooting_method.png', dpi=150)
plt.show()

print("\nKey observation: Only E = -0.5 Hartree gives a normalizable solution!")
print("This is E_1 = -13.6 eV for hydrogen.")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Substitution | u(r) = rR(r) |
| Radial equation | -ℏ²u''/2m + V_eff u = Eu |
| Effective potential | V_eff = V(r) + ℏ²l(l+1)/(2mr²) |
| Centrifugal barrier | V_cent = L²/(2mr²) |
| Boundary at r = 0 | u(0) = 0, u ~ r^(l+1) |
| Normalization | ∫|u|² dr = 1 |

### Key Insights

1. **u = rR transformation** converts radial equation to 1D form
2. **Centrifugal barrier** keeps particles away from origin for l > 0
3. **Effective potential** combines physical and angular momentum effects
4. **Boundary conditions** determine allowed energies (quantization)
5. **Classical analogy** explains centrifugal term physically

---

## Daily Checklist

- [ ] I can transform R(r) to u(r) = rR
- [ ] I understand the effective potential concept
- [ ] I can explain the centrifugal barrier physically
- [ ] I know the boundary conditions for bound states
- [ ] I can determine small-r behavior for different l
- [ ] I understand how u relates to probability

---

## Preview: Day 423

Tomorrow we solve the simplest 3D case: the **free particle in three dimensions**. This introduces spherical Bessel functions and sets the stage for scattering theory.

---

**Next:** [Day_423_Wednesday.md](Day_423_Wednesday.md) — Free Particle in 3D
