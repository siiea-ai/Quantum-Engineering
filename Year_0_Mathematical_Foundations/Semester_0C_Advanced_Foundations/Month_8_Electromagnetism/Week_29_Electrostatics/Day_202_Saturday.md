# Day 202: Multipole Expansion

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Monopole, Dipole, Quadrupole |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Multipole Moments & Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 202, you will be able to:

1. Derive the multipole expansion of the potential
2. Calculate monopole, dipole, and quadrupole moments
3. Understand the physical meaning of each multipole
4. Apply multipole expansion to atomic and molecular systems
5. Connect to quantum selection rules for transitions
6. Understand the far-field approximation

---

## Core Content

### 1. The Multipole Expansion

**Goal:** Express the potential of a localized charge distribution at distant points.

For a charge distribution $\rho(\mathbf{r}')$ confined to a region near the origin, the potential at $\mathbf{r}$ (where $r \gg r'$) is:

$$\phi(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0}\int \frac{\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}d^3r'$$

**Key expansion:**
$$\frac{1}{|\mathbf{r}-\mathbf{r}'|} = \sum_{l=0}^{\infty}\frac{r'^l}{r^{l+1}}P_l(\cos\gamma)$$

where $\gamma$ is the angle between $\mathbf{r}$ and $\mathbf{r}'$.

### 2. The First Three Terms

**Monopole ($l = 0$):**
$$\boxed{\phi_0 = \frac{1}{4\pi\varepsilon_0}\frac{Q}{r}}$$
where $Q = \int\rho\,d^3r'$ is the total charge.

**Dipole ($l = 1$):**
$$\boxed{\phi_1 = \frac{1}{4\pi\varepsilon_0}\frac{\mathbf{p}\cdot\hat{\mathbf{r}}}{r^2}}$$
where the **dipole moment** is:
$$\boxed{\mathbf{p} = \int\mathbf{r}'\rho(\mathbf{r}')\,d^3r'}$$

**Quadrupole ($l = 2$):**
$$\boxed{\phi_2 = \frac{1}{4\pi\varepsilon_0}\frac{1}{2r^3}\sum_{i,j}Q_{ij}\hat{r}_i\hat{r}_j}$$
where the **quadrupole tensor** is:
$$\boxed{Q_{ij} = \int(3r'_i r'_j - r'^2\delta_{ij})\rho(\mathbf{r}')\,d^3r'}$$

### 3. General Multipole Expansion

$$\boxed{\phi(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0}\sum_{l=0}^{\infty}\frac{1}{r^{l+1}}\sum_{m=-l}^{l}q_{lm}Y_l^m(\theta,\varphi)}$$

where the **multipole moments** are:
$$q_{lm} = \int r'^l Y_l^{m*}(\theta',\varphi')\rho(\mathbf{r}')\,d^3r'$$

### 4. Distance Dependence

| Multipole | Potential | Field |
|-----------|-----------|-------|
| Monopole | $1/r$ | $1/r^2$ |
| Dipole | $1/r^2$ | $1/r^3$ |
| Quadrupole | $1/r^3$ | $1/r^4$ |
| $2^l$-pole | $1/r^{l+1}$ | $1/r^{l+2}$ |

**Key insight:** At large distances, lower multipoles dominate. If $Q = 0$, the dipole term dominates. If $\mathbf{p} = 0$ too, the quadrupole dominates.

### 5. Electric Dipole Field

For a dipole at the origin pointing in $\hat{\mathbf{z}}$ direction:

$$\boxed{\mathbf{E} = \frac{p}{4\pi\varepsilon_0 r^3}(2\cos\theta\,\hat{\mathbf{r}} + \sin\theta\,\hat{\boldsymbol{\theta}})}$$

In Cartesian coordinates:
$$\mathbf{E} = \frac{1}{4\pi\varepsilon_0}\left[\frac{3(\mathbf{p}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{p}}{r^3}\right]$$

### 6. Energy of a Multipole in External Field

**Monopole:**
$$U_0 = Q\phi_{\text{ext}}(0)$$

**Dipole:**
$$U_1 = -\mathbf{p}\cdot\mathbf{E}_{\text{ext}}(0)$$

**Quadrupole:**
$$U_2 = -\frac{1}{6}\sum_{i,j}Q_{ij}\frac{\partial E_j}{\partial x_i}\bigg|_0$$

**Torque on dipole:**
$$\boldsymbol{\tau} = \mathbf{p} \times \mathbf{E}$$

**Force on dipole:**
$$\mathbf{F} = \nabla(\mathbf{p}\cdot\mathbf{E})$$

---

## Quantum Mechanics Connection

### Atomic Multipole Moments

For an atom with electron density $\rho(\mathbf{r}) = -e|\psi(\mathbf{r})|^2$:

**Dipole moment:**
$$\mathbf{p} = -e\int\mathbf{r}|\psi|^2 d^3r = -e\langle\mathbf{r}\rangle$$

For eigenstates of definite parity, $\langle\mathbf{r}\rangle = 0$, so atoms have no permanent electric dipole moment (unless parity is broken).

### Selection Rules for Transitions

**Electric dipole transitions** ($E1$) dominate, with selection rules:
- $\Delta l = \pm 1$
- $\Delta m = 0, \pm 1$

These come from the matrix element $\langle f|\mathbf{r}|i\rangle$ transforming as a dipole.

**Electric quadrupole transitions** ($E2$) are much weaker:
- $\Delta l = 0, \pm 2$
- $\Delta m = 0, \pm 1, \pm 2$

### Molecular Dipole Moments

**Polar molecules** like H₂O have permanent dipole moments:
- H₂O: $p = 1.85$ D (Debye) = $6.17 \times 10^{-30}$ C·m
- HCl: $p = 1.08$ D
- CO₂: $p = 0$ (symmetric, cancels)

These determine microwave absorption and molecular interactions.

### van der Waals Interactions

**Dipole-dipole interaction:**
$$U \propto -\frac{p_1 p_2}{r^3}$$

**Dipole-induced dipole:**
$$U \propto -\frac{p^2\alpha}{r^6}$$

**Induced dipole-induced dipole (London dispersion):**
$$U \propto -\frac{\alpha_1\alpha_2}{r^6}$$

The $1/r^6$ van der Waals interaction is quantum in origin!

---

## Worked Examples

### Example 1: Linear Quadrupole

**Problem:** Three charges on the $z$-axis: $+q$ at $z = 0$ and $-q/2$ at $z = \pm a$. Find the monopole moment, dipole moment, and quadrupole moment $Q_{zz}$.

**Solution:**

**Monopole:**
$$Q = q - q/2 - q/2 = 0$$

**Dipole:**
$$p_z = \sum q_i z_i = q(0) + (-q/2)(a) + (-q/2)(-a) = 0$$

**Quadrupole:**
$$Q_{zz} = \sum q_i(3z_i^2 - r_i^2) = q(0) - \frac{q}{2}(3a^2 - a^2) - \frac{q}{2}(3a^2 - a^2)$$
$$Q_{zz} = -2qa^2$$

The potential at large $r$ on the $z$-axis:
$$\phi = \frac{Q_{zz}}{4\pi\varepsilon_0}\frac{P_2(\cos\theta)}{r^3} = \frac{Q_{zz}}{4\pi\varepsilon_0 r^3}$$ (for $\theta = 0$)

### Example 2: Physical Dipole Moment

**Problem:** In the water molecule, the O-H bond length is 0.96 Å, and the H-O-H angle is 104.5°. If the partial charges are approximately $\pm 0.33e$ on H and O, estimate the dipole moment.

**Solution:**
Place O at origin. Each H is at angle 52.25° from the bisector.

$x$-component (along bisector):
$$p_x = 2 \times 0.33e \times 0.96 \times 10^{-10} \times \cos(52.25°)$$
$$p_x = 2 \times 0.33 \times 1.6 \times 10^{-19} \times 0.96 \times 10^{-10} \times 0.612$$
$$p_x = 6.2 \times 10^{-30} \text{ C·m} = 1.86 \text{ D}$$

This agrees well with the measured value of 1.85 D.

### Example 3: Quadrupole Field

**Problem:** Find the electric field on the $z$-axis due to a pure quadrupole with $Q_{zz} = Q_0$.

**Solution:**
On the axis ($\theta = 0$), $P_2(1) = 1$:
$$\phi = \frac{Q_0}{4\pi\varepsilon_0}\frac{1}{r^3}$$

$$E_r = -\frac{\partial\phi}{\partial r} = \frac{3Q_0}{4\pi\varepsilon_0 r^4}$$

The field falls off as $1/r^4$.

---

## Practice Problems

### Problem 1: Direct Application
A dipole with moment $p = 10^{-29}$ C·m is located at the origin, pointing in the $+z$ direction. Find:
(a) The potential at $(3, 0, 4)$ m
(b) The electric field at the same point

### Problem 2: Intermediate
Four charges form a square of side $a$ in the $xy$-plane: $+q$ at $(±a/2, 0)$ and $-q$ at $(0, ±a/2)$. Find the leading multipole moment and the potential at large $r$ along the $z$-axis.

**Hint:** This has quadrupole symmetry.

### Problem 3: Challenging
A sphere of radius $R$ has surface charge density $\sigma(\theta) = \sigma_0(3\cos^2\theta - 1)$.
(a) Show this corresponds to a pure quadrupole.
(b) Find the potential outside the sphere.

**Hint:** $3\cos^2\theta - 1 = 2P_2(\cos\theta)$

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
eps0 = 8.854e-12
k = 1/(4*np.pi*eps0)

def dipole_potential(p, r, theta):
    """Electric dipole potential."""
    return k * p * np.cos(theta) / r**2

def dipole_field(p, r, theta):
    """Electric dipole field components (E_r, E_theta)."""
    E_r = 2 * k * p * np.cos(theta) / r**3
    E_theta = k * p * np.sin(theta) / r**3
    return E_r, E_theta

def quadrupole_potential(Q_zz, r, theta):
    """Axial quadrupole potential."""
    P2 = 0.5 * (3*np.cos(theta)**2 - 1)
    return k * Q_zz * P2 / r**3

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Dipole potential contours ==========
ax1 = axes[0, 0]
p = 1e-29  # Dipole moment

# Create grid in Cartesian
x = np.linspace(-2, 2, 100)
z = np.linspace(-2, 2, 100)
X, Z = np.meshgrid(x, z)
R = np.sqrt(X**2 + Z**2)
R = np.where(R < 0.2, 0.2, R)  # Avoid singularity
THETA = np.arctan2(np.abs(X), Z)

# Calculate potential
phi_dipole = k * p * Z / R**3
phi_dipole = np.clip(phi_dipole, -1e-18, 1e-18)

contour = ax1.contourf(X, Z, phi_dipole * 1e18, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax1, label='φ (10⁻¹⁸ V)')

# Draw dipole
ax1.arrow(0, -0.1, 0, 0.2, head_width=0.1, head_length=0.05, fc='black', ec='black')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('z (m)')
ax1.set_title('Electric Dipole Potential')
ax1.set_aspect('equal')

# ========== Plot 2: Quadrupole potential ==========
ax2 = axes[0, 1]
Q_zz = 1e-38  # Quadrupole moment

phi_quad = quadrupole_potential(Q_zz, R, THETA)
phi_quad = np.clip(phi_quad * 1e18, -1, 1)

contour = ax2.contourf(X, Z, phi_quad, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax2, label='φ (arb. units)')

# Draw quadrupole (linear arrangement)
ax2.plot(0, 0.15, 'ro', markersize=10)
ax2.plot(0, -0.15, 'ro', markersize=10)
ax2.plot(0, 0, 'bo', markersize=14)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('z (m)')
ax2.set_title('Linear Quadrupole Potential')
ax2.set_aspect('equal')

# ========== Plot 3: Multipole comparison ==========
ax3 = axes[1, 0]

r = np.linspace(1, 10, 100)
theta = 0  # Along z-axis

# Normalize each to 1 at r=1
phi_monopole = 1/r
phi_dipole_axis = 1/r**2
phi_quadrupole = 1/r**3
phi_octupole = 1/r**4

ax3.semilogy(r, phi_monopole, 'b-', linewidth=2, label='Monopole (1/r)')
ax3.semilogy(r, phi_dipole_axis, 'r-', linewidth=2, label='Dipole (1/r²)')
ax3.semilogy(r, phi_quadrupole, 'g-', linewidth=2, label='Quadrupole (1/r³)')
ax3.semilogy(r, phi_octupole, 'm-', linewidth=2, label='Octupole (1/r⁴)')

ax3.set_xlabel('r (arb. units)')
ax3.set_ylabel('φ/φ(r=1)')
ax3.set_title('Multipole Distance Dependence')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ========== Plot 4: Dipole field lines ==========
ax4 = axes[1, 1]

x = np.linspace(-2, 2, 25)
z = np.linspace(-2, 2, 25)
X, Z = np.meshgrid(x, z)
R = np.sqrt(X**2 + Z**2)
R = np.where(R < 0.2, np.inf, R)

# Field components in Cartesian
# E_x = E_r sin(θ) + E_θ cos(θ)
# E_z = E_r cos(θ) - E_θ sin(θ)
cos_theta = Z / R
sin_theta = X / R

E_r = 2 * k * p * cos_theta / R**3
E_theta = k * p * sin_theta / R**3

Ex = E_r * sin_theta + E_theta * cos_theta
Ez = E_r * cos_theta - E_theta * sin_theta

# Normalize
E_mag = np.sqrt(Ex**2 + Ez**2)
Ex_norm = Ex / E_mag
Ez_norm = Ez / E_mag

ax4.quiver(X, Z, Ex_norm, Ez_norm, np.log10(E_mag + 1e-30), cmap='hot')
ax4.plot(0, 0.05, 'ro', markersize=8)
ax4.plot(0, -0.05, 'bo', markersize=8)
ax4.set_xlabel('x (m)')
ax4.set_ylabel('z (m)')
ax4.set_title('Dipole Electric Field')
ax4.set_xlim(-2, 2)
ax4.set_ylim(-2, 2)
ax4.set_aspect('equal')

plt.tight_layout()
plt.savefig('day_202_multipole.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 3D Visualization of Dipole Field ==========
fig = plt.figure(figsize=(12, 5))

# Dipole field magnitude on a sphere
ax1 = fig.add_subplot(121, projection='3d')

theta = np.linspace(0, np.pi, 50)
phi_angle = np.linspace(0, 2*np.pi, 50)
THETA, PHI = np.meshgrid(theta, phi_angle)

# Field magnitude at fixed r
r_fixed = 1.0
E_r, E_t = dipole_field(1e-29, r_fixed, THETA)
E_mag = np.sqrt(E_r**2 + E_t**2)
E_mag_normalized = E_mag / E_mag.max()

# Plot as colored sphere
X = r_fixed * np.sin(THETA) * np.cos(PHI)
Y = r_fixed * np.sin(THETA) * np.sin(PHI)
Z = r_fixed * np.cos(THETA)

ax1.plot_surface(X, Y, Z, facecolors=plt.cm.hot(E_mag_normalized), alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Dipole Field Magnitude on Sphere')

# Angular dependence
ax2 = fig.add_subplot(122)
theta_1d = np.linspace(0, np.pi, 100)
E_r_1d = 2 * np.cos(theta_1d)
E_t_1d = np.sin(theta_1d)
E_mag_1d = np.sqrt(E_r_1d**2 + E_t_1d**2)

ax2.plot(theta_1d * 180/np.pi, E_r_1d, 'b-', linewidth=2, label='$E_r \\propto 2\\cos\\theta$')
ax2.plot(theta_1d * 180/np.pi, E_t_1d, 'r-', linewidth=2, label='$E_\\theta \\propto \\sin\\theta$')
ax2.plot(theta_1d * 180/np.pi, E_mag_1d, 'k--', linewidth=2, label='$|E|$')
ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.set_xlabel('θ (degrees)')
ax2.set_ylabel('Field component (normalized)')
ax2.set_title('Angular Dependence of Dipole Field')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_202_dipole_angular.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate molecular dipole moments
print("Molecular Dipole Moments")
print("="*50)
Debye = 3.336e-30  # C·m
molecules = {
    'H₂O': 1.85,
    'HCl': 1.08,
    'NH₃': 1.47,
    'CO': 0.11,
    'CO₂': 0,
    'CH₄': 0
}

print(f"{'Molecule':<10} {'Dipole (D)':<12} {'Dipole (C·m)':<15}")
print("-"*40)
for mol, p_D in molecules.items():
    p_SI = p_D * Debye
    print(f"{mol:<10} {p_D:<12.2f} {p_SI:<15.2e}")
```

---

## Summary

### Key Formulas

| Multipole | Moment | Potential |
|-----------|--------|-----------|
| Monopole | $Q = \int\rho\,d^3r$ | $Q/(4\pi\varepsilon_0 r)$ |
| Dipole | $\mathbf{p} = \int\mathbf{r}\rho\,d^3r$ | $\mathbf{p}\cdot\hat{\mathbf{r}}/(4\pi\varepsilon_0 r^2)$ |
| Quadrupole | $Q_{ij} = \int(3r_ir_j - r^2\delta_{ij})\rho\,d^3r$ | $\propto 1/r^3$ |

| Formula | Description |
|---------|-------------|
| $U = -\mathbf{p}\cdot\mathbf{E}$ | Dipole energy |
| $\boldsymbol{\tau} = \mathbf{p}\times\mathbf{E}$ | Torque on dipole |
| $\mathbf{F} = \nabla(\mathbf{p}\cdot\mathbf{E})$ | Force on dipole |

### Main Takeaways

1. **Multipole expansion** organizes potential by distance dependence
2. **Higher multipoles** fall off faster; lowest nonzero term dominates at large $r$
3. **Dipole moment** is the first moment of charge distribution
4. **Selection rules** for QM transitions come from multipole operators
5. **Molecular interactions** depend on multipole moments (van der Waals)

---

## Daily Checklist

- [ ] I can derive the multipole expansion
- [ ] I can calculate monopole, dipole, and quadrupole moments
- [ ] I understand the dipole field formula
- [ ] I can compute energy of multipole in external field
- [ ] I see the connection to quantum selection rules

---

## Preview: Day 203

Tomorrow we complete Week 29 with **dielectrics** — materials that modify the electric field. We'll see how bound charges arise from polarization and review the week's content.

---

*"The multipole expansion is one of the most powerful techniques in physics — it appears everywhere from atoms to antennas."*

---

**Next:** Day 203 — Dielectrics & Week Review
