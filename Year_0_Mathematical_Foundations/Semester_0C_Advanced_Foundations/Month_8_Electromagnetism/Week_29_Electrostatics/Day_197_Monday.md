# Day 197: Coulomb's Law and Electric Field

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Coulomb's Law & Superposition |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Electric Field Calculations |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 197, you will be able to:

1. State Coulomb's law and explain its experimental basis
2. Apply the superposition principle for multiple charges
3. Calculate electric fields from discrete charge distributions
4. Compute electric fields from continuous charge distributions
5. Visualize electric field lines and understand their properties
6. Connect to the quantum Coulomb potential in the hydrogen atom

---

## Core Content

### 1. Coulomb's Law: The Foundation

**Historical Context:**
Charles-Augustin de Coulomb (1785) established experimentally that the force between two point charges is:

$$\boxed{\mathbf{F}_{12} = \frac{1}{4\pi\varepsilon_0}\frac{q_1 q_2}{r_{12}^2}\hat{\mathbf{r}}_{12}}$$

where:
- $\varepsilon_0 = 8.854 \times 10^{-12}$ F/m is the permittivity of free space
- $\hat{\mathbf{r}}_{12}$ points from charge 1 to charge 2
- The constant $k = 1/(4\pi\varepsilon_0) \approx 8.99 \times 10^9$ N·m²/C²

**Key Properties:**
1. **Inverse-square law:** Force falls off as $1/r^2$
2. **Central force:** Acts along the line joining charges
3. **Conservative:** Path-independent work
4. **Linear superposition:** Forces add vectorially

### 2. The Electric Field Concept

**Definition:**
The electric field $\mathbf{E}$ at position $\mathbf{r}$ is the force per unit positive test charge:

$$\boxed{\mathbf{E}(\mathbf{r}) = \frac{\mathbf{F}}{q_{\text{test}}} = \frac{1}{4\pi\varepsilon_0}\frac{Q}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\mathbf{r}}}$$

where $\mathbf{r}'$ is the source charge location.

**Physical Interpretation:**
- The field exists throughout space, not just where charges are
- Fields transmit forces at finite speed (speed of light)
- Energy and momentum are stored in the field itself

### 3. Superposition Principle

For $N$ discrete charges at positions $\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N$:

$$\boxed{\mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0}\sum_{i=1}^{N}\frac{q_i}{|\mathbf{r}-\mathbf{r}_i|^2}\hat{\mathbf{r}}_i}$$

where $\hat{\mathbf{r}}_i = (\mathbf{r} - \mathbf{r}_i)/|\mathbf{r} - \mathbf{r}_i|$.

**This is remarkably powerful:** Electric fields simply add. This linearity is fundamental to electrostatics and carries over to quantum mechanics.

### 4. Continuous Charge Distributions

For a continuous distribution with charge density $\rho(\mathbf{r}')$:

$$\boxed{\mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0}\int\frac{\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\boldsymbol{\eta}}\,d^3r'}$$

where $\boldsymbol{\eta} = \mathbf{r} - \mathbf{r}'$ is the separation vector.

**Types of distributions:**

| Type | Density | Element |
|------|---------|---------|
| Volume | $\rho$ (C/m³) | $dq = \rho\,d^3r'$ |
| Surface | $\sigma$ (C/m²) | $dq = \sigma\,da'$ |
| Line | $\lambda$ (C/m) | $dq = \lambda\,dl'$ |

### 5. Electric Field Lines

**Properties of field lines:**
1. Start on positive charges, end on negative charges
2. Tangent gives field direction at each point
3. Density of lines indicates field strength
4. Field lines never cross (field is single-valued)

**Flux interpretation:**
The number of field lines through a surface is proportional to the electric flux:
$$\Phi_E = \int \mathbf{E} \cdot d\mathbf{a}$$

### 6. Important Field Configurations

**Point charge:**
$$\mathbf{E} = \frac{q}{4\pi\varepsilon_0 r^2}\hat{\mathbf{r}}$$

**Electric dipole** (at large $r$):
$$\mathbf{E} \approx \frac{p}{4\pi\varepsilon_0 r^3}(2\cos\theta\,\hat{\mathbf{r}} + \sin\theta\,\hat{\boldsymbol{\theta}})$$

where $p = qd$ is the dipole moment.

**Infinite line charge:**
$$\mathbf{E} = \frac{\lambda}{2\pi\varepsilon_0 s}\hat{\mathbf{s}}$$

where $s$ is perpendicular distance from the line.

**Infinite plane:**
$$\mathbf{E} = \frac{\sigma}{2\varepsilon_0}\hat{\mathbf{n}}$$

independent of distance — remarkable result!

---

## Quantum Mechanics Connection

### The Hydrogen Atom Potential

The electron in hydrogen experiences the Coulomb potential from the proton:

$$V(r) = -\frac{e^2}{4\pi\varepsilon_0 r}$$

This $1/r$ potential is exactly what Coulomb's law predicts. The Schrödinger equation with this potential:

$$\left[-\frac{\hbar^2}{2m}\nabla^2 - \frac{e^2}{4\pi\varepsilon_0 r}\right]\psi = E\psi$$

gives the hydrogen atom energy levels:

$$E_n = -\frac{m e^4}{32\pi^2\varepsilon_0^2\hbar^2}\frac{1}{n^2} = -\frac{13.6\text{ eV}}{n^2}$$

**Key insight:** The $1/r^2$ force law (or equivalently $1/r$ potential) leads to:
- Closed elliptical orbits (classically)
- Discrete energy levels with degeneracy (quantum)
- The "accidental" SO(4) symmetry of hydrogen

### Superposition in Quantum Mechanics

The superposition principle for electric fields has a quantum analog: the superposition of quantum states. Just as:

$$\mathbf{E}_{\text{total}} = \mathbf{E}_1 + \mathbf{E}_2$$

we have in QM:

$$|\psi\rangle = c_1|\psi_1\rangle + c_2|\psi_2\rangle$$

This linearity is fundamental to both theories.

---

## Worked Examples

### Example 1: Electric Dipole Field

**Problem:** Two charges $+q$ at $(0, 0, d/2)$ and $-q$ at $(0, 0, -d/2)$ form a dipole. Find the electric field along the $z$-axis.

**Solution:**
At point $(0, 0, z)$ with $z > d/2$:

$$E_+ = \frac{q}{4\pi\varepsilon_0(z - d/2)^2}$$ (pointing in $+\hat{z}$)

$$E_- = \frac{q}{4\pi\varepsilon_0(z + d/2)^2}$$ (pointing in $-\hat{z}$)

Total field:
$$E_z = \frac{q}{4\pi\varepsilon_0}\left[\frac{1}{(z-d/2)^2} - \frac{1}{(z+d/2)^2}\right]$$

For $z \gg d$, use binomial expansion:
$$\frac{1}{(z \mp d/2)^2} = \frac{1}{z^2}\left(1 \mp \frac{d}{2z}\right)^{-2} \approx \frac{1}{z^2}\left(1 \pm \frac{d}{z}\right)$$

$$E_z \approx \frac{q}{4\pi\varepsilon_0 z^2}\left[\left(1 + \frac{d}{z}\right) - \left(1 - \frac{d}{z}\right)\right] = \frac{2qd}{4\pi\varepsilon_0 z^3}$$

$$\boxed{E_z = \frac{2p}{4\pi\varepsilon_0 z^3}}$$

where $p = qd$ is the dipole moment. The field falls off as $1/r^3$, faster than a monopole.

### Example 2: Uniformly Charged Ring

**Problem:** A ring of radius $R$ carries total charge $Q$ uniformly distributed. Find the electric field on its axis.

**Solution:**
Place the ring in the $xy$-plane centered at origin. Consider a small element $dq = \lambda R\,d\phi$ where $\lambda = Q/(2\pi R)$.

At point $(0, 0, z)$:
- Distance from element: $r = \sqrt{R^2 + z^2}$
- By symmetry, horizontal components cancel
- Only $z$-component survives

$$dE_z = \frac{dq}{4\pi\varepsilon_0(R^2 + z^2)}\cos\theta = \frac{dq}{4\pi\varepsilon_0(R^2 + z^2)}\cdot\frac{z}{\sqrt{R^2 + z^2}}$$

Integrating over the ring:
$$\boxed{E_z = \frac{Qz}{4\pi\varepsilon_0(R^2 + z^2)^{3/2}}}$$

**Special cases:**
- At center ($z = 0$): $E = 0$ (by symmetry)
- Far away ($z \gg R$): $E \approx Q/(4\pi\varepsilon_0 z^2)$ (point charge)
- Maximum at $z = R/\sqrt{2}$

### Example 3: Uniformly Charged Disk

**Problem:** A disk of radius $R$ has surface charge density $\sigma$. Find the field on the axis.

**Solution:**
Build the disk from concentric rings. A ring of radius $r'$ and width $dr'$ has charge $dq = \sigma \cdot 2\pi r'\,dr'$.

From Example 2, this ring contributes:
$$dE_z = \frac{\sigma \cdot 2\pi r'\,dr' \cdot z}{4\pi\varepsilon_0(r'^2 + z^2)^{3/2}}$$

Integrate from $0$ to $R$:
$$E_z = \frac{\sigma z}{2\varepsilon_0}\int_0^R \frac{r'\,dr'}{(r'^2 + z^2)^{3/2}}$$

Let $u = r'^2 + z^2$, $du = 2r'\,dr'$:
$$E_z = \frac{\sigma z}{4\varepsilon_0}\left[-\frac{2}{\sqrt{u}}\right]_{z^2}^{R^2+z^2} = \frac{\sigma}{2\varepsilon_0}\left(1 - \frac{z}{\sqrt{R^2 + z^2}}\right)$$

$$\boxed{E_z = \frac{\sigma}{2\varepsilon_0}\left(1 - \frac{z}{\sqrt{R^2 + z^2}}\right)}$$

**Limit as $R \to \infty$:** $E_z \to \sigma/(2\varepsilon_0)$ — the infinite plane result.

---

## Practice Problems

### Problem 1: Direct Application
Three charges are placed at the corners of an equilateral triangle with side $a$: $+q$ at $(0, 0)$, $+q$ at $(a, 0)$, and $-q$ at $(a/2, a\sqrt{3}/2)$. Find the electric field at the center of the triangle.

**Answer:** $E = \frac{q}{2\pi\varepsilon_0 a^2}$ pointing toward the negative charge.

### Problem 2: Intermediate
A semicircular wire of radius $R$ carries uniform line charge $\lambda$. Find the electric field at the center of the semicircle.

**Hint:** Set up the integral with $dq = \lambda R\,d\theta$.

**Answer:** $E = \frac{\lambda}{2\pi\varepsilon_0 R}$ pointing perpendicular to the diameter.

### Problem 3: Challenging
A spherical shell of radius $R$ carries surface charge density $\sigma(\theta) = \sigma_0\cos\theta$. Find the electric field at the center.

**Hint:** This is equivalent to two displaced hemispheres. Use superposition.

**Answer:** $E = \frac{\sigma_0}{3\varepsilon_0}$ along the axis of symmetry.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
k = 8.99e9  # Coulomb constant (N·m²/C²)

def electric_field_point_charge(q, r_charge, r_field):
    """
    Calculate electric field from a point charge.

    Parameters:
        q: charge (C)
        r_charge: position of charge (array)
        r_field: position where field is evaluated (array)

    Returns:
        Electric field vector (array)
    """
    r_sep = r_field - r_charge
    r_mag = np.linalg.norm(r_sep)
    if r_mag < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    return k * q * r_sep / r_mag**3

def electric_field_distribution(charges, positions, r_field):
    """
    Calculate total electric field from multiple point charges.
    """
    E_total = np.zeros(3)
    for q, r_q in zip(charges, positions):
        E_total += electric_field_point_charge(q, r_q, r_field)
    return E_total

# Create visualization of electric field lines
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Single positive charge ==========
ax1 = axes[0, 0]
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)

q = 1e-9  # 1 nC
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        E = electric_field_point_charge(q, np.array([0, 0, 0]),
                                        np.array([X[i,j], Y[i,j], 0]))
        Ex[i,j] = E[0]
        Ey[i,j] = E[1]

# Normalize for visualization
E_mag = np.sqrt(Ex**2 + Ey**2)
E_mag = np.where(E_mag < 1e-10, 1e-10, E_mag)
Ex_norm = Ex / E_mag
Ey_norm = Ey / E_mag

ax1.quiver(X, Y, Ex_norm, Ey_norm, E_mag, cmap='hot')
ax1.plot(0, 0, 'ro', markersize=15, label='+q')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title('Single Positive Charge')
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ========== Plot 2: Electric dipole ==========
ax2 = axes[0, 1]
charges_dipole = [1e-9, -1e-9]
positions_dipole = [np.array([0, 0.5, 0]), np.array([0, -0.5, 0])]

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        E = electric_field_distribution(charges_dipole, positions_dipole,
                                        np.array([X[i,j], Y[i,j], 0]))
        Ex[i,j] = E[0]
        Ey[i,j] = E[1]

E_mag = np.sqrt(Ex**2 + Ey**2)
E_mag = np.where(E_mag < 1e-10, 1e-10, E_mag)
Ex_norm = Ex / E_mag
Ey_norm = Ey / E_mag

ax2.quiver(X, Y, Ex_norm, Ey_norm, np.log10(E_mag + 1), cmap='viridis')
ax2.plot(0, 0.5, 'ro', markersize=12, label='+q')
ax2.plot(0, -0.5, 'bo', markersize=12, label='-q')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('Electric Dipole')
ax2.set_aspect('equal')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ========== Plot 3: Field along ring axis ==========
ax3 = axes[1, 0]
R = 1.0  # Ring radius
Q = 1e-9  # Total charge

z = np.linspace(-3, 3, 200)
E_ring = Q * z / (4 * np.pi * 8.854e-12 * (R**2 + z**2)**1.5)

# Point charge comparison
E_point = k * Q / (z**2 + 1e-10) * np.sign(z)
E_point = np.where(np.abs(z) < 0.1, np.nan, E_point)

ax3.plot(z, E_ring * 1e9, 'b-', linewidth=2, label='Charged Ring')
ax3.plot(z, np.where(np.abs(z) > R, E_point * 1e9, np.nan), 'r--',
         linewidth=2, label='Point Charge (|z|>R)')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_xlabel('z/R')
ax3.set_ylabel('E (arbitrary units)')
ax3.set_title('Electric Field on Ring Axis')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-3, 3)
ax3.set_ylim(-10, 10)

# ========== Plot 4: Field of charged disk ==========
ax4 = axes[1, 1]
R_disk = 1.0
sigma = 1e-9  # C/m²
eps0 = 8.854e-12

z = np.linspace(0.01, 3, 200)
E_disk = (sigma / (2 * eps0)) * (1 - z / np.sqrt(R_disk**2 + z**2))
E_infinite_plane = sigma / (2 * eps0) * np.ones_like(z)

ax4.plot(z, E_disk / E_infinite_plane, 'b-', linewidth=2,
         label=f'Disk (R = {R_disk} m)')
ax4.axhline(y=1, color='r', linestyle='--', label='Infinite Plane')
ax4.set_xlabel('z (m)')
ax4.set_ylabel('E / E_infinite')
ax4.set_title('Disk Field vs Infinite Plane')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('day_197_coulomb_fields.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 3D Visualization of Dipole Field ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create streamlines-like visualization
theta = np.linspace(0, 2*np.pi, 8)
for t in theta:
    # Start points near positive charge
    x_start = 0.1 * np.cos(t)
    y_start = 0.5 + 0.1 * np.sin(t)

    # Trace field line
    x_line = [x_start]
    y_line = [y_start]
    z_line = [0]

    for _ in range(100):
        r = np.array([x_line[-1], y_line[-1], z_line[-1]])
        E = electric_field_distribution(charges_dipole, positions_dipole, r)
        E_mag = np.linalg.norm(E)
        if E_mag < 1e-6 or np.linalg.norm(r) > 3:
            break
        step = 0.1 * E / E_mag
        x_line.append(x_line[-1] + step[0])
        y_line.append(y_line[-1] + step[1])
        z_line.append(z_line[-1] + step[2])

    ax.plot(x_line, y_line, z_line, 'b-', linewidth=1)

ax.scatter([0], [0.5], [0], c='red', s=100, label='+q')
ax.scatter([0], [-0.5], [0], c='blue', s=100, label='-q')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Electric Dipole Field Lines (3D)')
ax.legend()
plt.savefig('day_197_dipole_3d.png', dpi=150, bbox_inches='tight')
plt.show()

print("Day 197: Coulomb's Law Visualizations Complete")
print("="*50)
print(f"Coulomb constant k = {k:.3e} N·m²/C²")
print(f"ε₀ = {1/(4*np.pi*k):.3e} F/m")
print(f"\nDipole field at z = 10R falls off as 1/z³")
print(f"Ring field maximum at z = R/√2 = {R/np.sqrt(2):.3f} m")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\mathbf{F} = \frac{1}{4\pi\varepsilon_0}\frac{q_1 q_2}{r^2}\hat{\mathbf{r}}$ | Coulomb's law |
| $\mathbf{E} = \mathbf{F}/q_{\text{test}}$ | Electric field definition |
| $\mathbf{E} = \frac{1}{4\pi\varepsilon_0}\sum_i \frac{q_i}{r_i^2}\hat{\mathbf{r}}_i$ | Superposition |
| $\mathbf{E} = \frac{1}{4\pi\varepsilon_0}\int \frac{\rho\,d^3r'}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\boldsymbol{\eta}}$ | Continuous distribution |
| $E_{\text{dipole}} \propto 1/r^3$ | Dipole far-field |

### Main Takeaways

1. **Coulomb's law** is the fundamental law of electrostatics — inverse square, central, conservative
2. **Superposition** makes complex problems tractable by adding contributions
3. **Field lines** visualize direction and strength; never cross
4. **Different geometries** give different distance dependence: point ($1/r^2$), dipole ($1/r^3$), line ($1/r$), plane (constant)
5. **The Coulomb potential** becomes the hydrogen atom's central potential in quantum mechanics

---

## Daily Checklist

- [ ] I can state Coulomb's law and explain each term
- [ ] I can apply superposition for discrete charges
- [ ] I can set up integrals for continuous distributions
- [ ] I can sketch field lines for various configurations
- [ ] I understand the connection to the hydrogen atom

---

## Preview: Day 198

Tomorrow we introduce **Gauss's Law**, which provides a powerful alternative to direct integration for highly symmetric problems. We'll derive fields for spheres, cylinders, and planes with elegant symmetry arguments.

---

*"The inverse square law of electric force has the same form as Newton's law of gravitation. This is no coincidence — both arise from the geometry of three-dimensional space."*

---

**Next:** Day 198 — Gauss's Law
