# Day 221: Transformation of Electromagnetic Fields

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: How E and B Transform |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications: Moving Charges and Currents |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 221, you will be able to:

1. Derive the transformation laws for electric and magnetic fields
2. Explain why electric and magnetic fields are frame-dependent
3. Apply field transformations to solve problems with moving sources
4. Understand the unification of E and B as a single electromagnetic field
5. Calculate fields from moving charges using the transformation laws
6. Connect field transformations to the electromagnetic field tensor

---

## Core Content

### 1. The Need for Field Transformations

**Question:** If a charge is at rest in frame $S'$, it produces only an electric field. But in frame $S$ where the charge moves, there's a current, so there should be a magnetic field too. How do $\mathbf{E}$ and $\mathbf{B}$ relate between frames?

**Key insight:** Electric and magnetic fields are **not** separate physical entities. They are different aspects of a single electromagnetic field, and their relative magnitudes depend on the observer's reference frame.

### 2. Derivation from the Lorentz Force

The Lorentz force $\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$ must be consistent between frames.

Consider frame $S'$ moving with velocity $\mathbf{V} = V\hat{\mathbf{x}}$ relative to $S$.

**Force transformation:**
The 4-force transforms as:
$$f^{\mu} = \frac{dp^{\mu}}{d\tau}$$

In the instantaneous rest frame of the charge ($S'$ where $\mathbf{v}' = 0$):
$$\mathbf{F}' = q\mathbf{E}'$$

Using the transformation of force and the relationship between $\mathbf{F}$, $\mathbf{E}$, $\mathbf{B}$ in frame $S$, we derive the field transformation laws.

### 3. Field Transformation Equations

For a boost along the $x$-axis with velocity $V$:

**Parallel components (along the boost direction):**
$$\boxed{E'_x = E_x}$$
$$\boxed{B'_x = B_x}$$

**Perpendicular components:**
$$\boxed{E'_y = \gamma(E_y - VB_z)}$$
$$\boxed{E'_z = \gamma(E_z + VB_y)}$$
$$\boxed{B'_y = \gamma\left(B_y + \frac{V}{c^2}E_z\right)}$$
$$\boxed{B'_z = \gamma\left(B_z - \frac{V}{c^2}E_y\right)}$$

**Compact vector form:**
$$\boxed{\mathbf{E}'_{\parallel} = \mathbf{E}_{\parallel}}$$
$$\boxed{\mathbf{E}'_{\perp} = \gamma(\mathbf{E}_{\perp} + \mathbf{V} \times \mathbf{B})}$$
$$\boxed{\mathbf{B}'_{\parallel} = \mathbf{B}_{\parallel}}$$
$$\boxed{\mathbf{B}'_{\perp} = \gamma\left(\mathbf{B}_{\perp} - \frac{\mathbf{V} \times \mathbf{E}}{c^2}\right)}$$

### 4. Physical Interpretation

**Case 1: Pure electric field in $S$ (no magnetic field)**

If $\mathbf{B} = 0$ in $S$:
$$\mathbf{E}'_{\perp} = \gamma\mathbf{E}_{\perp}, \quad \mathbf{B}'_{\perp} = -\gamma\frac{\mathbf{V} \times \mathbf{E}}{c^2}$$

A moving observer sees a magnetic field even though there's none in the original frame!

**Case 2: Pure magnetic field in $S$ (no electric field)**

If $\mathbf{E} = 0$ in $S$:
$$\mathbf{E}'_{\perp} = \gamma\mathbf{V} \times \mathbf{B}, \quad \mathbf{B}'_{\perp} = \gamma\mathbf{B}_{\perp}$$

A moving observer sees an electric field from a pure magnetic field!

**Case 3: Charge at rest in $S'$**

A point charge at rest in $S'$ produces only $\mathbf{E}'$. In $S$ where it moves with velocity $V\hat{\mathbf{x}}$:
$$\mathbf{B} = \gamma\frac{\mathbf{V} \times \mathbf{E}'}{c^2} = \frac{\mathbf{v} \times \mathbf{E}}{c^2}$$

This is the Biot-Savart result for a moving charge!

### 5. Invariants of the Electromagnetic Field

Certain combinations of $\mathbf{E}$ and $\mathbf{B}$ are **Lorentz invariants** (same in all frames):

**Invariant 1:**
$$\boxed{\mathbf{E} \cdot \mathbf{B} = E'_{\parallel}B'_{\parallel} = \text{invariant}}$$

**Invariant 2:**
$$\boxed{E^2 - c^2B^2 = E'^2 - c^2B'^2 = \text{invariant}}$$

Or equivalently: $\mathbf{E}^2 - c^2\mathbf{B}^2$

**Consequences:**
- If $\mathbf{E} \perp \mathbf{B}$ in one frame, they're perpendicular in all frames
- If $|\mathbf{E}| > c|\mathbf{B}|$ in one frame, this holds in all frames
- A pure electric field cannot be transformed into a pure magnetic field (and vice versa) unless $|\mathbf{E}| = c|\mathbf{B}|$ and $\mathbf{E} \perp \mathbf{B}$

### 6. Fields of a Moving Point Charge

A charge $q$ at rest at the origin in $S'$ has:
$$\mathbf{E}' = \frac{q}{4\pi\epsilon_0}\frac{\mathbf{r}'}{r'^3}, \quad \mathbf{B}' = 0$$

Transform to $S$ where the charge moves with velocity $\mathbf{v} = v\hat{\mathbf{x}}$:

Using the inverse transformations and $x' = \gamma(x - vt)$:

$$\boxed{\mathbf{E} = \frac{q}{4\pi\epsilon_0}\frac{\gamma\mathbf{r}}{(\gamma^2(x-vt)^2 + y^2 + z^2)^{3/2}}}$$

At $t = 0$ with the charge at the origin:
$$\mathbf{E} = \frac{q}{4\pi\epsilon_0}\frac{(1-\beta^2)}{(1 - \beta^2\sin^2\theta)^{3/2}}\frac{\hat{\mathbf{r}}}{r^2}$$

where $\theta$ is measured from the direction of motion.

**Key features:**
- Field is **weakened** along the direction of motion ($\theta = 0$): $E \propto (1-\beta^2)$
- Field is **enhanced** perpendicular to motion ($\theta = 90°$): $E \propto \gamma$
- Field lines are **compressed** into a disk perpendicular to motion
- As $v \to c$: field becomes a "pancake" perpendicular to velocity

### 7. Magnetic Field of a Moving Charge

$$\mathbf{B} = \frac{\mathbf{v} \times \mathbf{E}}{c^2} = \frac{\mu_0}{4\pi}\frac{q\mathbf{v} \times \hat{\mathbf{r}}}{r^2}\frac{1-\beta^2}{(1-\beta^2\sin^2\theta)^{3/2}}$$

For $v \ll c$:
$$\mathbf{B} \approx \frac{\mu_0}{4\pi}\frac{q\mathbf{v} \times \hat{\mathbf{r}}}{r^2}$$

This recovers the Biot-Savart law for a point charge!

### 8. The Wire and Moving Observer

**Classic example:** A neutral wire carries current $I$.

In the **lab frame $S$**:
- Positive ions at rest: no contribution to E
- Electrons moving with drift velocity $v_d$: create magnetic field $\mathbf{B}$
- A positive test charge moving parallel to wire at speed $v$ feels force $\mathbf{F} = q\mathbf{v} \times \mathbf{B}$

In the **test charge frame $S'$**:
- Electrons appear closer together (length contraction): negative charge density increases
- Ions appear farther apart (length contraction works opposite way): positive charge density decreases
- Net negative charge density creates electric field pointing toward wire
- Force $\mathbf{F}' = q\mathbf{E}'$ attracts charge to wire

**Same force, different explanation!** In one frame it's magnetic, in the other it's electric.

---

## Quantum Mechanics Connection

### The Electromagnetic Field in QED

In quantum electrodynamics (QED), the electromagnetic field is described by the photon field $A^{\mu}$, which transforms as a 4-vector.

The field tensor $F^{\mu\nu}$ (to be studied tomorrow) contains both $\mathbf{E}$ and $\mathbf{B}$:
$$F^{\mu\nu} = \partial^{\mu}A^{\nu} - \partial^{\nu}A^{\mu}$$

### Gauge Invariance and the Photon

The transformation properties of $\mathbf{E}$ and $\mathbf{B}$ are intimately connected to **gauge invariance** in quantum mechanics:
$$A^{\mu} \to A^{\mu} + \partial^{\mu}\chi$$

This gauge freedom is why the photon is massless and has only two polarization states (not three).

### Spin and Magnetic Moments

The transformation of magnetic fields explains how the electron's magnetic moment $\boldsymbol{\mu} = -g_s\frac{e}{2m}\mathbf{S}$ interacts differently in different frames.

In the electron's rest frame, it sees transformed fields:
$$\mathbf{B}'_{eff} = \gamma\mathbf{B} - \frac{\gamma - 1}{v^2}(\mathbf{v} \cdot \mathbf{B})\mathbf{v} - \gamma\frac{\mathbf{v} \times \mathbf{E}}{c^2}$$

The last term $-\gamma\frac{\mathbf{v} \times \mathbf{E}}{c^2}$ gives the **spin-orbit coupling** in atoms!

### Thomas Precession

When a spinning electron moves in an electric field (as in an atom), its spin precesses due to the relativistic transformation of frames. This **Thomas precession** reduces the spin-orbit coupling by a factor of 2:
$$\Delta E_{so} = \frac{1}{2}\xi(r)\mathbf{L} \cdot \mathbf{S}$$

The factor of 1/2 (compared to naive calculation) comes from proper treatment of field transformations.

---

## Worked Examples

### Example 1: Magnetic Field from Moving Charge

**Problem:** An electron moves with velocity $v = 0.8c$ in the $+x$ direction. At the moment it passes the origin, find the electric and magnetic fields at point $(0, 1 \text{ nm}, 0)$.

**Solution:**

Lorentz factor: $\gamma = 1/\sqrt{1 - 0.64} = 5/3 \approx 1.667$

In the electron's rest frame, only electric field exists:
$$E'_y = \frac{e}{4\pi\epsilon_0 r'^2} = \frac{1.6 \times 10^{-19}}{4\pi \times 8.85 \times 10^{-12} \times (10^{-9})^2} = 1.44 \times 10^{9} \text{ V/m}$$

Transform to lab frame (electron moving in $+x$, field point in $+y$):
$$E_y = \gamma E'_y = 1.667 \times 1.44 \times 10^9 = 2.40 \times 10^9 \text{ V/m}$$

Magnetic field:
$$B_z = \gamma \frac{v}{c^2} E'_y = \frac{v}{c^2} E_y = \frac{0.8 \times 3 \times 10^8}{(3 \times 10^8)^2} \times 2.40 \times 10^9 = 6.4 \text{ T}$$

$$\boxed{\mathbf{E} = 2.4 \times 10^9 \hat{\mathbf{y}} \text{ V/m}, \quad \mathbf{B} = 6.4 \hat{\mathbf{z}} \text{ T}}$$

### Example 2: Frame Where E or B Vanishes

**Problem:** In frame $S$, $\mathbf{E} = E_0\hat{\mathbf{y}}$ and $\mathbf{B} = 2E_0\hat{\mathbf{z}}/c$. Find a frame where the electric field vanishes.

**Solution:**

First, check the invariant: $E^2 - c^2B^2 = E_0^2 - c^2(2E_0/c)^2 = E_0^2 - 4E_0^2 = -3E_0^2 < 0$

Since $c|B| > |E|$, we can find a frame where $\mathbf{E}' = 0$.

From $\mathbf{E}'_{\perp} = \gamma(\mathbf{E}_{\perp} + \mathbf{V} \times \mathbf{B}) = 0$:
$$E_0\hat{\mathbf{y}} + V\hat{\mathbf{x}} \times \frac{2E_0}{c}\hat{\mathbf{z}} = 0$$
$$E_0\hat{\mathbf{y}} - \frac{2E_0V}{c}\hat{\mathbf{y}} = 0$$
$$V = \frac{c}{2}$$

Check: $\gamma = 1/\sqrt{1 - 0.25} = 1.155$

In frame $S'$ moving at $V = c/2$ in the $+x$ direction:
$$\mathbf{E}' = 0$$
$$B'_z = \gamma\left(B_z - \frac{V}{c^2}E_y\right) = 1.155\left(\frac{2E_0}{c} - \frac{0.5c}{c^2}E_0\right) = 1.155 \times \frac{1.5E_0}{c} = \frac{1.73E_0}{c}$$

$$\boxed{V = 0.5c \text{ in } +x \text{ direction}, \quad \mathbf{B}' = \frac{\sqrt{3}E_0}{c}\hat{\mathbf{z}}}$$

### Example 3: Neutral Wire Problem

**Problem:** A long straight wire carries current $I = 10$ A. An electron moves parallel to the wire at distance $r = 1$ mm with velocity $v = 0.1c$ (same direction as current). Calculate the force on the electron in both the lab frame and electron's rest frame.

**Solution:**

**Lab frame:**
Magnetic field at electron: $B = \frac{\mu_0 I}{2\pi r} = \frac{4\pi \times 10^{-7} \times 10}{2\pi \times 10^{-3}} = 2 \times 10^{-3}$ T

Force: $F = evB = 1.6 \times 10^{-19} \times 0.1 \times 3 \times 10^8 \times 2 \times 10^{-3} = 9.6 \times 10^{-15}$ N

Direction: toward the wire (attractive for parallel motion)

**Electron's rest frame:**
In this frame, the wire moves at $v = 0.1c$ in the opposite direction.

Due to length contraction:
- Positive ions (moving backward): spacing contracts by $\gamma$, density increases to $\gamma\lambda_+$
- Conduction electrons (moving backward faster): density changes to $\lambda_-'$

The net effect is a net positive charge density on the wire, creating an electric field:
$$\lambda_{net} = (\gamma - 1)\lambda_0$$

where $\lambda_0$ relates to the current: $I = \lambda_0 v_d$ (drift velocity).

For small $v$: $\gamma - 1 \approx v^2/(2c^2)$

The force remains the same magnitude (as it must, since force is a physical effect):
$$F' \approx 9.6 \times 10^{-15} \text{ N}$$

$$\boxed{F = 9.6 \times 10^{-15} \text{ N toward wire in both frames}}$$

---

## Practice Problems

### Problem 1: Direct Application
In frame $S$, $\mathbf{E} = 3 \times 10^6 \hat{\mathbf{y}}$ V/m and $\mathbf{B} = 0$. Find $\mathbf{E}'$ and $\mathbf{B}'$ in frame $S'$ moving at $v = 0.6c$ in the $+x$ direction.

**Answers:** $\mathbf{E}' = 3.75 \times 10^6 \hat{\mathbf{y}}$ V/m, $\mathbf{B}' = -8.33 \times 10^{-3} \hat{\mathbf{z}}$ T

### Problem 2: Intermediate
Show that the invariant $\mathbf{E} \cdot \mathbf{B}$ is unchanged under the field transformation equations for a boost along $x$.

**Hint:** Calculate $E'_xB'_x + E'_yB'_y + E'_zB'_z$ and simplify.

### Problem 3: Challenging
A proton and an electron are both at rest, separated by distance $d$. An observer moves past at velocity $v$. In the observer's frame:
(a) What are the electric fields of each particle?
(b) What magnetic field does each particle produce?
(c) What is the magnetic force between them?
(d) Compare with the electric force. Why doesn't the magnetic force violate Newton's third law?

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
c = 3e8  # m/s
epsilon_0 = 8.854e-12  # F/m
mu_0 = 4 * np.pi * 1e-7  # H/m
e = 1.602e-19  # C

def gamma(v):
    """Lorentz factor"""
    return 1 / np.sqrt(1 - (v/c)**2)

def transform_fields(E, B, v):
    """
    Transform E and B fields to frame moving with velocity v along x-axis.
    E, B are 3-component arrays [Ex, Ey, Ez]
    Returns E', B' in the new frame
    """
    g = gamma(v)
    beta = v / c

    E_prime = np.zeros(3)
    B_prime = np.zeros(3)

    # Parallel components unchanged
    E_prime[0] = E[0]
    B_prime[0] = B[0]

    # Perpendicular components transform
    E_prime[1] = g * (E[1] - v * B[2])
    E_prime[2] = g * (E[2] + v * B[1])
    B_prime[1] = g * (B[1] + v * E[2] / c**2)
    B_prime[2] = g * (B[2] - v * E[1] / c**2)

    return E_prime, B_prime

def field_of_moving_charge(q, v, r):
    """
    Electric and magnetic field of a charge q moving with velocity v at position r.
    v is along x-axis.
    """
    x, y, z = r
    r_mag = np.sqrt(x**2 + y**2 + z**2)

    if r_mag < 1e-15:
        return np.array([0, 0, 0]), np.array([0, 0, 0])

    beta = v / c
    g = gamma(v)

    # Angle from velocity direction
    sin_theta = np.sqrt(y**2 + z**2) / r_mag
    cos_theta = x / r_mag

    # Field magnitude factor
    factor = (1 - beta**2) / (1 - beta**2 * sin_theta**2)**(3/2)

    # Electric field (radial, modified by relativity)
    E_mag = q * factor / (4 * np.pi * epsilon_0 * r_mag**2)
    E = E_mag * r / r_mag

    # Magnetic field B = (v × E) / c^2
    v_vec = np.array([v, 0, 0])
    B = np.cross(v_vec, E) / c**2

    return E, B

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: E and B field transformation ==========
ax1 = axes[0, 0]

# Original field: E in y direction
E0 = np.array([0, 1e6, 0])  # V/m
B0 = np.array([0, 0, 0])  # T

velocities = np.linspace(0, 0.95*c, 50)
E_y_transformed = []
B_z_transformed = []

for v in velocities:
    E_prime, B_prime = transform_fields(E0, B0, v)
    E_y_transformed.append(E_prime[1] / 1e6)
    B_z_transformed.append(B_prime[2] * 1e3)  # mT

ax1.plot(velocities / c, E_y_transformed, 'b-', linewidth=2, label="$E'_y$ (MV/m)")
ax1.plot(velocities / c, B_z_transformed, 'r-', linewidth=2, label="$B'_z$ (mT)")
ax1.axhline(y=1, color='b', linestyle='--', alpha=0.3)

ax1.set_xlabel('v/c (frame velocity)', fontsize=12)
ax1.set_ylabel('Field magnitude', fontsize=12)
ax1.set_title('Field Transformation: Pure E → E\' and B\'', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# ========== Plot 2: Field lines of moving charge ==========
ax2 = axes[0, 1]

# Create grid
y = np.linspace(-2, 2, 20)
z = np.linspace(-2, 2, 20)
Y, Z = np.meshgrid(y, z)
X = np.zeros_like(Y)

velocities_to_plot = [0, 0.5*c, 0.9*c, 0.99*c]
colors = ['blue', 'green', 'orange', 'red']
labels = ['v=0', 'v=0.5c', 'v=0.9c', 'v=0.99c']

for v, color, label in zip(velocities_to_plot, colors, labels):
    # Field magnitude at fixed distance in different directions
    angles = np.linspace(0, 2*np.pi, 100)
    r = 1  # Fixed distance

    E_mag = []
    for theta in angles:
        pos = np.array([r * np.cos(theta), r * np.sin(theta), 0])
        E, B = field_of_moving_charge(e, v, pos)
        E_mag.append(np.linalg.norm(E) * 4 * np.pi * epsilon_0 / e)  # Normalize

    # Plot as polar
    x_plot = [em * np.cos(theta) for em, theta in zip(E_mag, angles)]
    y_plot = [em * np.sin(theta) for em, theta in zip(E_mag, angles)]
    ax2.plot(x_plot, y_plot, color=color, linewidth=2, label=label)

ax2.set_xlabel('x direction (motion →)', fontsize=12)
ax2.set_ylabel('y direction', fontsize=12)
ax2.set_title('Electric Field Angular Distribution\n(normalized, charge at origin moving in +x)', fontsize=14)
ax2.legend(fontsize=10)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)

# ========== Plot 3: Field invariants ==========
ax3 = axes[1, 0]

# Start with various E and B configurations
# Show that E·B and E² - c²B² are invariant

# Case: E and B at an angle
E0 = np.array([0, 1e6, 0])  # 1 MV/m in y
B0 = np.array([0, 0, 2e-3])  # 2 mT in z

velocities = np.linspace(-0.9*c, 0.9*c, 100)
inv1 = []  # E·B
inv2 = []  # E² - c²B²

for v in velocities:
    E_prime, B_prime = transform_fields(E0, B0, v)
    inv1.append(np.dot(E_prime, B_prime))
    inv2.append(np.dot(E_prime, E_prime) - c**2 * np.dot(B_prime, B_prime))

# Normalize for plotting
inv1 = np.array(inv1) / np.abs(inv1[len(inv1)//2])
inv2 = np.array(inv2) / np.abs(inv2[len(inv2)//2])

ax3.plot(velocities / c, inv1, 'b-', linewidth=2, label='$\\mathbf{E} \\cdot \\mathbf{B}$ (normalized)')
ax3.plot(velocities / c, inv2, 'r-', linewidth=2, label='$E^2 - c^2B^2$ (normalized)')

ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('Frame velocity v/c', fontsize=12)
ax3.set_ylabel('Invariant (normalized)', fontsize=12)
ax3.set_title('Lorentz Invariants of the EM Field', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.5, 1.5)

# ========== Plot 4: Wire and moving charge ==========
ax4 = axes[1, 1]

# Magnetic field from wire
I = 10  # A
r_vals = np.linspace(0.1, 5, 100)  # mm

B_wire = mu_0 * I / (2 * np.pi * r_vals * 1e-3)  # T

# Force on electron moving parallel at different speeds
v_electron = [0.001*c, 0.01*c, 0.1*c]
colors = ['blue', 'green', 'red']

for v, color in zip(v_electron, colors):
    F = e * v * B_wire  # Force in Newtons
    ax4.plot(r_vals, F * 1e15, color=color, linewidth=2,
             label=f'v = {v/c:.3f}c')

ax4.set_xlabel('Distance from wire (mm)', fontsize=12)
ax4.set_ylabel('Force on electron (fN)', fontsize=12)
ax4.set_title('Force on Electron Moving Parallel to Current-Carrying Wire\n(I = 10 A)', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 5)

plt.tight_layout()
plt.savefig('day_221_field_transformation.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Numerical Examples ==========
print("=" * 60)
print("ELECTROMAGNETIC FIELD TRANSFORMATION")
print("=" * 60)

# Example 1: Pure E field
print("\n--- Example 1: Pure Electric Field ---")
E0 = np.array([0, 1e6, 0])  # 1 MV/m in y
B0 = np.array([0, 0, 0])

print(f"Original: E = {E0} V/m, B = {B0} T")

for v_frac in [0.5, 0.8, 0.9, 0.99]:
    v = v_frac * c
    E_prime, B_prime = transform_fields(E0, B0, v)
    print(f"\nv = {v_frac}c:")
    print(f"  E' = [{E_prime[0]:.2e}, {E_prime[1]:.2e}, {E_prime[2]:.2e}] V/m")
    print(f"  B' = [{B_prime[0]:.2e}, {B_prime[1]:.2e}, {B_prime[2]:.2e}] T")
    print(f"  γ = {gamma(v):.3f}")

# Example 2: Crossed E and B fields
print("\n" + "=" * 60)
print("--- Example 2: Crossed E and B Fields ---")
E0 = np.array([0, 1e6, 0])  # 1 MV/m in y
B0 = np.array([0, 0, 5e-3])  # 5 mT in z

print(f"Original: E = {E0[1]:.2e} ŷ V/m, B = {B0[2]:.2e} ẑ T")
print(f"E·B = {np.dot(E0, B0):.2e}")
print(f"E² - c²B² = {np.dot(E0,E0) - c**2*np.dot(B0,B0):.2e}")

# Find frame where E = 0
# Need V such that E_y - V*B_z = 0
V_zero_E = E0[1] / B0[2]
if abs(V_zero_E) < c:
    print(f"\nFrame with E' = 0: v = {V_zero_E/c:.3f}c")
    E_prime, B_prime = transform_fields(E0, B0, V_zero_E)
    print(f"  E' = {E_prime}")
    print(f"  B' = {B_prime}")
else:
    print(f"\nCannot find frame with E' = 0 (would require v = {V_zero_E/c:.3f}c > c)")

# Example 3: Verify invariants
print("\n" + "=" * 60)
print("--- Verification of Lorentz Invariants ---")

E0 = np.array([1e5, 2e6, 3e5])
B0 = np.array([1e-3, 2e-3, 3e-3])

inv1_original = np.dot(E0, B0)
inv2_original = np.dot(E0, E0) - c**2 * np.dot(B0, B0)

print(f"Original frame:")
print(f"  E·B = {inv1_original:.4e}")
print(f"  E² - c²B² = {inv2_original:.4e}")

for v_frac in [0.3, 0.6, 0.9]:
    v = v_frac * c
    E_prime, B_prime = transform_fields(E0, B0, v)
    inv1_new = np.dot(E_prime, B_prime)
    inv2_new = np.dot(E_prime, E_prime) - c**2 * np.dot(B_prime, B_prime)
    print(f"\nv = {v_frac}c:")
    print(f"  E'·B' = {inv1_new:.4e} (diff: {abs(inv1_new - inv1_original)/abs(inv1_original)*100:.2f}%)")
    print(f"  E'² - c²B'² = {inv2_new:.4e} (diff: {abs(inv2_new - inv2_original)/abs(inv2_original)*100:.2f}%)")

print("\n" + "=" * 60)
print("Day 221: Field Transformations Complete")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $E'_{\parallel} = E_{\parallel}$ | Parallel E unchanged |
| $B'_{\parallel} = B_{\parallel}$ | Parallel B unchanged |
| $\mathbf{E}'_{\perp} = \gamma(\mathbf{E}_{\perp} + \mathbf{V} \times \mathbf{B})$ | Perpendicular E transforms |
| $\mathbf{B}'_{\perp} = \gamma(\mathbf{B}_{\perp} - \mathbf{V} \times \mathbf{E}/c^2)$ | Perpendicular B transforms |
| $\mathbf{E} \cdot \mathbf{B} = \text{invariant}$ | First Lorentz invariant |
| $E^2 - c^2B^2 = \text{invariant}$ | Second Lorentz invariant |

### Main Takeaways

1. **E and B are frame-dependent** - what appears as a pure electric field in one frame has a magnetic component in another
2. **Moving charges produce magnetic fields** naturally through field transformation
3. **Lorentz invariants** $\mathbf{E} \cdot \mathbf{B}$ and $E^2 - c^2B^2$ are the same in all frames
4. **Fields of ultrarelativistic charges** are concentrated perpendicular to motion
5. **The wire paradox** shows the same force arises from B in one frame, E in another
6. **Spin-orbit coupling** in atoms comes from field transformations

---

## Daily Checklist

- [ ] I can apply the field transformation equations
- [ ] I understand why E and B mix under Lorentz transformations
- [ ] I can calculate fields from moving charges
- [ ] I know the Lorentz invariants of the electromagnetic field
- [ ] I can explain the neutral wire problem from both frames
- [ ] I understand how this connects to spin-orbit coupling

---

## Preview: Day 222

Tomorrow we introduce the **electromagnetic field tensor $F^{\mu\nu}$** and write Maxwell's equations in **covariant form**. This elegant formulation makes Lorentz invariance manifest and prepares us for quantum electrodynamics.

---

*"The special theory of relativity shows that electric and magnetic fields are different aspects of the same thing, namely the electromagnetic field."*
— Richard Feynman

---

**Next:** Day 222 — Covariant Formulation (Field Tensor)
