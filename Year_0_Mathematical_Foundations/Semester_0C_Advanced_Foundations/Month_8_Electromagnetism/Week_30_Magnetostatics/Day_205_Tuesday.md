# Day 205: Biot-Savart Law

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Biot-Savart Law |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Calculating Magnetic Fields |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 205, you will be able to:

1. State the Biot-Savart law and explain its physical meaning
2. Calculate magnetic fields from line currents
3. Find the field of a circular loop on its axis
4. Understand superposition for magnetic fields
5. Compare Biot-Savart to Coulomb's law
6. Connect to magnetic dipole moments

---

## Core Content

### 1. The Biot-Savart Law

**Statement:** The magnetic field $d\mathbf{B}$ produced by a current element $I\,d\boldsymbol{\ell}$ at position $\mathbf{r}'$ is:

$$\boxed{d\mathbf{B} = \frac{\mu_0}{4\pi}\frac{I\,d\boldsymbol{\ell} \times \hat{\boldsymbol{\eta}}}{\eta^2}}$$

where:
- $\mu_0 = 4\pi \times 10^{-7}$ T·m/A is the permeability of free space
- $\boldsymbol{\eta} = \mathbf{r} - \mathbf{r}'$ is the separation vector
- $\hat{\boldsymbol{\eta}} = \boldsymbol{\eta}/|\boldsymbol{\eta}|$

**For a complete circuit:**
$$\boxed{\mathbf{B}(\mathbf{r}) = \frac{\mu_0 I}{4\pi}\oint \frac{d\boldsymbol{\ell}' \times \hat{\boldsymbol{\eta}}}{\eta^2}}$$

### 2. Comparison with Coulomb's Law

| Coulomb | Biot-Savart |
|---------|-------------|
| $d\mathbf{E} = \frac{1}{4\pi\varepsilon_0}\frac{dq\,\hat{\boldsymbol{\eta}}}{\eta^2}$ | $d\mathbf{B} = \frac{\mu_0}{4\pi}\frac{I\,d\boldsymbol{\ell} \times \hat{\boldsymbol{\eta}}}{\eta^2}$ |
| Radial | Perpendicular to both $d\boldsymbol{\ell}$ and $\hat{\boldsymbol{\eta}}$ |
| Source: charge | Source: current |

**Key difference:** Magnetic field lines encircle the current (no beginning or end).

### 3. Field of Infinite Straight Wire

**Setup:** Current $I$ flows along the $z$-axis. Find $\mathbf{B}$ at perpendicular distance $s$.

**Calculation:**
$$B = \frac{\mu_0 I}{4\pi}\int_{-\infty}^{\infty}\frac{s\,dz'}{(s^2 + z'^2)^{3/2}}$$

Using the integral $\int \frac{dz}{(s^2+z^2)^{3/2}} = \frac{z}{s^2\sqrt{s^2+z^2}}$:

$$\boxed{B = \frac{\mu_0 I}{2\pi s}}$$

Direction: Circles the wire (right-hand rule).

### 4. Field of a Circular Loop

**On-axis field** (loop of radius $R$ at origin in $xy$-plane):

$$\boxed{B_z = \frac{\mu_0 I R^2}{2(R^2 + z^2)^{3/2}}}$$

**At center ($z = 0$):**
$$B = \frac{\mu_0 I}{2R}$$

**Far from loop ($z \gg R$):**
$$B \approx \frac{\mu_0 I R^2}{2z^3} = \frac{\mu_0}{4\pi}\frac{2\mu}{z^3}$$

where $\mu = I\pi R^2$ is the magnetic dipole moment.

### 5. Field of a Solenoid

A solenoid has $n$ turns per unit length.

**Inside (long solenoid):**
$$\boxed{B = \mu_0 n I}$$

**Outside:** $B \approx 0$ (field lines return through infinite region)

**At the end:** $B = \frac{1}{2}\mu_0 n I$

### 6. Magnetic Field from Volume Current

For volume current density $\mathbf{J}$:

$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}(\mathbf{r}') \times \hat{\boldsymbol{\eta}}}{\eta^2}d^3r'$$

---

## Quantum Mechanics Connection

### Magnetic Dipole Moment

A current loop has magnetic dipole moment:
$$\boldsymbol{\mu} = IA\hat{\mathbf{n}} = I\pi R^2\hat{\mathbf{n}}$$

For an electron in circular orbit:
$$\mu_{\text{orbital}} = \frac{evr}{2} = \frac{e}{2m}L$$

This gives the **gyromagnetic ratio:**
$$\gamma = \frac{\mu}{L} = \frac{e}{2m}$$

### Bohr Magneton

The quantum of magnetic moment:
$$\mu_B = \frac{e\hbar}{2m_e} = 9.274 \times 10^{-24} \text{ J/T}$$

Electron spin magnetic moment: $\mu_s \approx \mu_B$ (with $g \approx 2$)

### Nuclear Magnetic Moments

Nuclear spins have much smaller moments:
$$\mu_N = \frac{e\hbar}{2m_p} = 5.051 \times 10^{-27} \text{ J/T}$$

This is the basis of NMR and MRI.

---

## Worked Examples

### Example 1: Field of Straight Segment

**Problem:** A wire segment from $(0, 0, -L)$ to $(0, 0, +L)$ carries current $I$. Find $\mathbf{B}$ at point $(s, 0, 0)$.

**Solution:**
$$B = \frac{\mu_0 I}{4\pi s}\left[\sin\theta_2 - \sin\theta_1\right]$$

where $\sin\theta = L/\sqrt{s^2 + L^2}$:

$$B = \frac{\mu_0 I}{4\pi s}\cdot\frac{2L}{\sqrt{s^2 + L^2}}$$

$$\boxed{B = \frac{\mu_0 IL}{2\pi s\sqrt{s^2 + L^2}}}$$

As $L \to \infty$: $B \to \frac{\mu_0 I}{2\pi s}$ ✓

### Example 2: Two Parallel Wires

**Problem:** Two parallel wires separated by distance $d$ carry currents $I_1$ and $I_2$ in the same direction. Find the force per unit length between them.

**Solution:**
Wire 2 is in the field of wire 1:
$$B_1 = \frac{\mu_0 I_1}{2\pi d}$$

Force on wire 2:
$$\frac{F}{L} = I_2 B_1 = \frac{\mu_0 I_1 I_2}{2\pi d}$$

$$\boxed{\frac{F}{L} = \frac{\mu_0 I_1 I_2}{2\pi d}}$$ (attractive for same direction)

This defines the Ampere: $I_1 = I_2 = 1$ A, $d = 1$ m gives $F/L = 2 \times 10^{-7}$ N/m.

### Example 3: Helmholtz Coils

**Problem:** Two identical coaxial loops of radius $R$ are separated by distance $d$. For what $d$ is the field most uniform at the center?

**Solution:**
For maximum uniformity, we want $\frac{d^2B}{dz^2} = 0$ at the midpoint.

This occurs when $d = R$ (Helmholtz configuration).

The field at the center:
$$B = \frac{8\mu_0 I}{5\sqrt{5}R} \approx 0.72\frac{\mu_0 I}{R}$$

---

## Practice Problems

### Problem 1: Direct Application
A circular loop of radius 5 cm carries current 10 A. Find the magnetic field:
(a) At the center
(b) On the axis, 10 cm from the center

**Answers:** (a) $B = 126$ μT; (b) $B = 22.4$ μT

### Problem 2: Intermediate
A square loop of side $a$ carries current $I$. Find the magnetic field at the center.

**Answer:** $B = \frac{2\sqrt{2}\mu_0 I}{\pi a}$

### Problem 3: Challenging
A wire is bent into a semicircle of radius $R$ connected by two straight segments. Current $I$ flows through it. Find the magnetic field at the center of the semicircle.

**Hint:** The straight segments contribute nothing at the center.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
mu0 = 4 * np.pi * 1e-7  # T·m/A

def biot_savart_segment(I, r1, r2, r_field, n_points=100):
    """
    Calculate B field from a current segment using Biot-Savart.

    Parameters:
        I: current (A)
        r1, r2: endpoints of segment (arrays)
        r_field: field point (array)
        n_points: number of integration points

    Returns:
        B field vector (array)
    """
    B = np.zeros(3)
    dl_vec = (r2 - r1) / n_points

    for i in range(n_points):
        r_source = r1 + (i + 0.5) * (r2 - r1) / n_points
        eta = r_field - r_source
        eta_mag = np.linalg.norm(eta)
        if eta_mag > 1e-10:
            dB = (mu0 / (4 * np.pi)) * I * np.cross(dl_vec, eta) / eta_mag**3
            B += dB

    return B

def B_circular_loop_axis(I, R, z):
    """Magnetic field on axis of circular loop."""
    return mu0 * I * R**2 / (2 * (R**2 + z**2)**1.5)

def B_infinite_wire(I, s):
    """Magnetic field of infinite wire at distance s."""
    return mu0 * I / (2 * np.pi * s)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Field of straight wire ==========
ax1 = axes[0, 0]
I = 10  # A

s = np.linspace(0.01, 0.5, 100)
B_wire = B_infinite_wire(I, s)

ax1.plot(s * 100, B_wire * 1e6, 'b-', linewidth=2)
ax1.set_xlabel('Distance s (cm)')
ax1.set_ylabel('B (μT)')
ax1.set_title('Magnetic Field of Infinite Wire (I = 10 A)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 50)

# ========== Plot 2: Field of circular loop on axis ==========
ax2 = axes[0, 1]
R = 0.1  # 10 cm radius
I = 10  # A

z = np.linspace(-0.3, 0.3, 200)
B_loop = B_circular_loop_axis(I, R, z)

# Dipole approximation
mu = I * np.pi * R**2
B_dipole = np.where(np.abs(z) > R,
                     mu0 * 2 * mu / (4 * np.pi * np.abs(z)**3),
                     np.nan)

ax2.plot(z * 100, B_loop * 1e6, 'b-', linewidth=2, label='Exact')
ax2.plot(z * 100, B_dipole * 1e6, 'r--', linewidth=2, label='Dipole approx.')
ax2.axvline(x=R*100, color='g', linestyle=':', alpha=0.5)
ax2.axvline(x=-R*100, color='g', linestyle=':', alpha=0.5)
ax2.set_xlabel('z (cm)')
ax2.set_ylabel('B (μT)')
ax2.set_title('Circular Loop On-Axis Field')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ========== Plot 3: Helmholtz coils ==========
ax3 = axes[1, 0]

R = 0.1  # radius
d = R  # Helmholtz condition

z = np.linspace(-0.15, 0.15, 200)

# Field from each coil
B1 = B_circular_loop_axis(I, R, z - d/2)
B2 = B_circular_loop_axis(I, R, z + d/2)
B_total = B1 + B2

ax3.plot(z * 100, B1 * 1e6, 'b--', alpha=0.5, label='Coil 1')
ax3.plot(z * 100, B2 * 1e6, 'r--', alpha=0.5, label='Coil 2')
ax3.plot(z * 100, B_total * 1e6, 'k-', linewidth=2, label='Total')

# Mark coil positions
ax3.axvline(x=d/2*100, color='gray', linestyle=':', alpha=0.5)
ax3.axvline(x=-d/2*100, color='gray', linestyle=':', alpha=0.5)

ax3.set_xlabel('z (cm)')
ax3.set_ylabel('B (μT)')
ax3.set_title('Helmholtz Coils (d = R)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ========== Plot 4: Field lines of loop ==========
ax4 = axes[1, 1]

# Create field line visualization
y_grid = np.linspace(-0.2, 0.2, 20)
z_grid = np.linspace(-0.2, 0.2, 20)
Y, Z = np.meshgrid(y_grid, z_grid)

By = np.zeros_like(Y)
Bz = np.zeros_like(Z)

# Calculate field from loop using numerical integration
n_phi = 50
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        r_field = np.array([0, Y[i,j], Z[i,j]])
        B = np.zeros(3)

        # Integrate around loop
        for k in range(n_phi):
            phi = 2 * np.pi * k / n_phi
            phi_next = 2 * np.pi * (k + 1) / n_phi

            r1 = np.array([R * np.cos(phi), R * np.sin(phi), 0])
            r2 = np.array([R * np.cos(phi_next), R * np.sin(phi_next), 0])

            B += biot_savart_segment(I, r1, r2, r_field, n_points=1)

        By[i,j] = B[1]
        Bz[i,j] = B[2]

# Normalize
B_mag = np.sqrt(By**2 + Bz**2)
By_norm = By / (B_mag + 1e-10)
Bz_norm = Bz / (B_mag + 1e-10)

ax4.streamplot(Y*100, Z*100, By_norm, Bz_norm, density=1.5, color=np.log10(B_mag + 1e-10),
                cmap='viridis', linewidth=1)

# Draw loop (cross-section)
ax4.plot([-R*100, R*100], [0, 0], 'ro', markersize=10)
ax4.set_xlabel('y (cm)')
ax4.set_ylabel('z (cm)')
ax4.set_title('Magnetic Field Lines of Loop')
ax4.set_aspect('equal')

plt.tight_layout()
plt.savefig('day_205_biot_savart.png', dpi=150, bbox_inches='tight')
plt.show()

# Force between parallel wires
print("\nForce Between Parallel Wires")
print("="*50)
I1 = I2 = 1  # A
d = 1  # m
F_per_L = mu0 * I1 * I2 / (2 * np.pi * d)
print(f"I₁ = I₂ = {I1} A, d = {d} m")
print(f"F/L = {F_per_L:.3e} N/m")
print(f"This defines the Ampere!")

# Magnetic moments
print("\nMagnetic Moments")
print("="*50)
e = 1.602e-19
hbar = 1.055e-34
m_e = 9.109e-31
m_p = 1.673e-27

mu_B = e * hbar / (2 * m_e)
mu_N = e * hbar / (2 * m_p)

print(f"Bohr magneton μ_B = {mu_B:.3e} J/T")
print(f"Nuclear magneton μ_N = {mu_N:.3e} J/T")
print(f"Ratio μ_B/μ_N = {mu_B/mu_N:.0f}")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $d\mathbf{B} = \frac{\mu_0 I}{4\pi}\frac{d\boldsymbol{\ell} \times \hat{\boldsymbol{\eta}}}{\eta^2}$ | Biot-Savart law |
| $B = \frac{\mu_0 I}{2\pi s}$ | Infinite wire |
| $B = \frac{\mu_0 I R^2}{2(R^2+z^2)^{3/2}}$ | Loop on axis |
| $B = \frac{\mu_0 I}{2R}$ | Loop at center |
| $\frac{F}{L} = \frac{\mu_0 I_1 I_2}{2\pi d}$ | Parallel wires |

### Main Takeaways

1. **Biot-Savart law** is the magnetic analog of Coulomb's law
2. **Magnetic field** from current circles the current (no monopoles)
3. **Far-field** of a loop is a dipole field
4. **Solenoid field** is uniform inside: $B = \mu_0 n I$
5. **Bohr magneton** is the quantum unit of magnetic moment

---

## Daily Checklist

- [ ] I can state and apply the Biot-Savart law
- [ ] I can calculate $\mathbf{B}$ for a straight wire
- [ ] I can calculate $\mathbf{B}$ for a circular loop
- [ ] I understand the connection to magnetic dipoles
- [ ] I can compute forces between current-carrying wires

---

## Preview: Day 206

Tomorrow we study **Ampère's law** — a powerful alternative to Biot-Savart for problems with symmetry, analogous to Gauss's law for electrostatics.

---

*"The Biot-Savart law tells us how currents create magnetic fields — the foundation of all electromagnetic technology."*

---

**Next:** Day 206 — Ampère's Law
