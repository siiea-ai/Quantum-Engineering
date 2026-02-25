# Day 201: Method of Images

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Image Charges |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications & Induced Charges |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 201, you will be able to:

1. Explain the principle behind the method of images
2. Find image charges for a point charge near a conducting plane
3. Calculate induced surface charges on conductors
4. Apply the method to spherical conductors
5. Compute forces between charges and conductors
6. Connect to boundary conditions in quantum mechanics

---

## Core Content

### 1. The Principle of Image Charges

**Key insight:** The potential in a region depends only on the charges inside that region and the boundary conditions. If we can find a fictitious charge distribution outside the region that produces the same boundary conditions, it will give the correct potential inside.

**Requirements for image method:**
1. Image charges must be outside the region of interest
2. The total configuration (real + image) must satisfy boundary conditions
3. The solution is unique by the uniqueness theorem

### 2. Point Charge Near Infinite Conducting Plane

**Setup:** Charge $+q$ at distance $d$ above a grounded conducting plane at $z = 0$.

**Image:** Place charge $-q$ at distance $d$ below the plane (at $z = -d$).

**Verification:** At $z = 0$:
$$\phi = \frac{q}{4\pi\varepsilon_0}\left(\frac{1}{\sqrt{x^2+y^2+d^2}} - \frac{1}{\sqrt{x^2+y^2+d^2}}\right) = 0 \checkmark$$

**Potential for $z > 0$:**
$$\boxed{\phi(x,y,z) = \frac{q}{4\pi\varepsilon_0}\left(\frac{1}{\sqrt{x^2+y^2+(z-d)^2}} - \frac{1}{\sqrt{x^2+y^2+(z+d)^2}}\right)}$$

### 3. Induced Surface Charge

The normal component of $\mathbf{E}$ at a conductor surface equals $\sigma/\varepsilon_0$.

$$\sigma = -\varepsilon_0\frac{\partial\phi}{\partial n}\bigg|_{\text{surface}}$$

For the plane at $z = 0$:
$$E_z = -\frac{\partial\phi}{\partial z}\bigg|_{z=0} = \frac{q}{4\pi\varepsilon_0}\cdot\frac{2d}{(x^2+y^2+d^2)^{3/2}}$$

$$\boxed{\sigma(x,y) = -\frac{qd}{2\pi(x^2+y^2+d^2)^{3/2}}}$$

**Total induced charge:**
$$Q_{\text{ind}} = \int\sigma\,dA = -q$$

The induced charge equals the image charge!

### 4. Force on the Charge

The charge $q$ is attracted to its image:

$$\boxed{F = \frac{1}{4\pi\varepsilon_0}\frac{q \cdot (-q)}{(2d)^2} = -\frac{q^2}{16\pi\varepsilon_0 d^2}}$$

The force is attractive (toward the conductor) and falls off as $1/d^2$.

### 5. Energy Considerations

**Work to bring charge from infinity:**
$$W = \int_\infty^d F\,dz' = -\frac{q^2}{16\pi\varepsilon_0}\int_\infty^d \frac{dz'}{z'^2} = -\frac{q^2}{16\pi\varepsilon_0 d}$$

This is **half** what we'd get from the image picture directly. Why? The image charge moves as the real charge moves — the correct energy accounts for this.

### 6. Point Charge Near Conducting Sphere

**Setup:** Charge $q$ at distance $a$ from center of grounded sphere of radius $R$ ($a > R$).

**Image:** Charge $q' = -qR/a$ at distance $a' = R^2/a$ from center (inside sphere).

**Verification:** At any point on the sphere surface:
$$\phi = \frac{1}{4\pi\varepsilon_0}\left(\frac{q}{r_1} + \frac{q'}{r_2}\right)$$

Using geometry, $r_1/r_2 = a/R$ at every point on sphere, so:
$$\phi = \frac{q}{4\pi\varepsilon_0 r_1}\left(1 - \frac{R/a \cdot a/R}{1}\right) = 0 \checkmark$$

**Image charge magnitude and location:**
$$\boxed{q' = -\frac{qR}{a}, \quad a' = \frac{R^2}{a}}$$

### 7. Insulated (Isolated) Sphere

For a sphere with total charge $Q$ (not grounded):

Add a second image charge $Q - q'$ at the center to give total charge $Q$.

**Force on the external charge:**
$$F = \frac{qq'}{4\pi\varepsilon_0(a-a')^2} + \frac{q(Q-q')}{4\pi\varepsilon_0 a^2}$$

---

## Quantum Mechanics Connection

### Boundary Conditions on Wave Functions

In quantum mechanics, wave functions must satisfy boundary conditions at interfaces:

1. **Hard wall:** $\psi = 0$ at boundary (like grounded conductor)
2. **Continuity:** $\psi$ continuous across boundary
3. **Current continuity:** $(1/m^*)\partial\psi/\partial n$ continuous (with effective mass $m^*$)

**Particle in a box:** The condition $\psi(0) = \psi(L) = 0$ is analogous to a grounded conductor boundary condition.

### Image Charges in Dielectric Problems

When a charge is near a dielectric interface:
$$q' = q\frac{\varepsilon_1 - \varepsilon_2}{\varepsilon_1 + \varepsilon_2}$$

This has quantum analogs in scattering at potential steps where reflection coefficients depend on $(k_1 - k_2)/(k_1 + k_2)$.

### Green's Functions

The image method is closely related to **Green's functions**. The potential due to a point charge with given boundary conditions is the Green's function for the Laplacian:

$$G(\mathbf{r}, \mathbf{r}') = \frac{1}{4\pi\varepsilon_0|\mathbf{r}-\mathbf{r}'|} + \text{(image terms)}$$

This connects directly to quantum propagators.

---

## Worked Examples

### Example 1: Charge Between Two Parallel Planes

**Problem:** A charge $q$ is located at $z = d$ between two grounded conducting planes at $z = 0$ and $z = L$. Find the potential.

**Solution:**
Need infinite series of images to satisfy both boundary conditions.

Images for $z = 0$ plane: $-q$ at $z = -d$
But this violates $z = L$ boundary, so add $+q$ at $z = 2L - d$
This violates $z = 0$, so add $-q$ at $z = -2L + d$
Continue...

**Image positions:**
- $z_n = 2nL + d$ (positive charges)
- $z_n = 2nL - d$ (negative charges)

$$\phi = \frac{q}{4\pi\varepsilon_0}\sum_{n=-\infty}^{\infty}\left[\frac{1}{|z - (2nL+d)|} - \frac{1}{|z - (2nL-d)|}\right]$$

This can be summed using Fourier series techniques.

### Example 2: Charge Near Grounded Sphere

**Problem:** A charge $q = 10$ nC is located 30 cm from the center of a grounded conducting sphere of radius 10 cm. Find:
(a) The image charge and its location
(b) The force on $q$
(c) The induced surface charge at the nearest point

**Solution:**

(a) Image charge:
$$q' = -\frac{qR}{a} = -\frac{10 \times 0.1}{0.3} = -3.33 \text{ nC}$$

Location:
$$a' = \frac{R^2}{a} = \frac{0.01}{0.3} = 0.033 \text{ m} = 3.33 \text{ cm from center}$$

(b) Force:
$$F = \frac{qq'}{4\pi\varepsilon_0(a-a')^2} = \frac{8.99 \times 10^9 \times 10^{-8} \times (-3.33 \times 10^{-9})}{(0.267)^2}$$
$$F = -4.2 \times 10^{-6} \text{ N}$$ (attractive)

(c) Surface charge at nearest point ($\theta = 0$):
$$\sigma = -\varepsilon_0 E_n = \frac{1}{4\pi}\left[\frac{q}{(a-R)^2} + \frac{q'}{(R-a')^2}\right]$$

### Example 3: Line Charge Near Cylinder

**Problem:** An infinite line charge $\lambda$ is parallel to and at distance $d$ from the axis of a grounded conducting cylinder of radius $R$.

**Solution:**
By analogy with the sphere, the image is a line charge $-\lambda$ at distance $R^2/d$ from the axis.

The potential at distance $s$ from the line (in the region outside the cylinder):
$$\phi = -\frac{\lambda}{2\pi\varepsilon_0}\ln\frac{s_1}{s_2}$$

where $s_1$ and $s_2$ are distances from real and image lines.

---

## Practice Problems

### Problem 1: Direct Application
A point charge $q$ is at distance $d$ from an infinite grounded conducting plane.
(a) Find the potential at point $(d, 0, d)$.
(b) Find the electric field at this point.

**Answers:** (a) $\phi = \frac{q}{4\pi\varepsilon_0}\left(\frac{1}{d} - \frac{1}{\sqrt{5}d}\right)$

### Problem 2: Intermediate
Two grounded conducting planes meet at a 90° angle. A charge $q$ is placed at position $(a, b)$ from the corner. How many image charges are needed? Find their positions.

**Answer:** 3 images at $(a, -b)$, $(-a, b)$, $(-a, -b)$.

### Problem 3: Challenging
A charge $q$ is at the center of a conducting spherical shell (inner radius $a$, outer radius $b$). The shell has total charge $Q$. Find the potential everywhere.

**Hint:** Use superposition and consider charge distributions on inner and outer surfaces.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# Constants
eps0 = 8.854e-12
k = 1/(4*np.pi*eps0)

def potential_image_plane(x, y, z, q, d):
    """Potential for charge q at height d above grounded plane z=0."""
    # Real charge at (0, 0, d)
    r1 = np.sqrt(x**2 + y**2 + (z-d)**2)
    # Image charge -q at (0, 0, -d)
    r2 = np.sqrt(x**2 + y**2 + (z+d)**2)

    r1 = np.where(r1 < 1e-10, 1e-10, r1)
    r2 = np.where(r2 < 1e-10, 1e-10, r2)

    return k * q * (1/r1 - 1/r2)

def potential_image_sphere(x, y, q, a, R):
    """Potential for charge q at distance a from grounded sphere radius R."""
    # Real charge at (a, 0)
    r1 = np.sqrt((x-a)**2 + y**2)
    # Image charge q' = -qR/a at (R²/a, 0)
    a_prime = R**2 / a
    q_prime = -q * R / a
    r2 = np.sqrt((x-a_prime)**2 + y**2)

    r1 = np.where(r1 < 1e-10, 1e-10, r1)
    r2 = np.where(r2 < 1e-10, 1e-10, r2)

    return k * (q/r1 + q_prime/r2)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Charge near plane - potential ==========
ax1 = axes[0, 0]
d = 1.0
q = 1e-9

x = np.linspace(-3, 3, 100)
z = np.linspace(0.01, 4, 100)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)

phi = potential_image_plane(X, Y, Z, q, d)
phi = np.clip(phi * 1e9, -20, 20)

contour = ax1.contourf(X, Z, phi, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax1, label='φ (nV·m)')

# Draw conductor
ax1.fill_between([-3, 3], [0, 0], [-0.2, -0.2], color='gray', alpha=0.7)
ax1.plot(0, d, 'ro', markersize=12, label=f'+q at z={d}')
ax1.plot(0, -d, 'bo', markersize=10, alpha=0.5, label='Image -q')

ax1.set_xlabel('x (m)')
ax1.set_ylabel('z (m)')
ax1.set_title('Charge Near Grounded Plane')
ax1.legend()
ax1.set_ylim(-0.3, 4)

# ========== Plot 2: Induced surface charge ==========
ax2 = axes[0, 1]
x_surf = np.linspace(-3, 3, 200)
y_surf = 0
z_surf = 0

sigma = -q * d / (2*np.pi * (x_surf**2 + d**2)**1.5)

ax2.plot(x_surf, sigma * 1e9, 'b-', linewidth=2)
ax2.fill_between(x_surf, sigma * 1e9, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('σ (nC/m²)')
ax2.set_title('Induced Surface Charge Density')
ax2.grid(True, alpha=0.3)

# Calculate total induced charge
dx = x_surf[1] - x_surf[0]
# Integrate in 2D (rotate around z-axis)
# Q_ind = ∫σ 2πr dr ≈ -q
print(f"Analytically: Total induced charge = -q")

# ========== Plot 3: Charge near sphere ==========
ax3 = axes[1, 0]
R = 1.0
a = 2.5
q = 1e-9

# Create grid outside sphere
x = np.linspace(-4, 5, 150)
y = np.linspace(-3, 3, 150)
X, Y = np.meshgrid(x, y)

phi = potential_image_sphere(X, Y, q, a, R)
# Set to zero inside sphere
r_from_origin = np.sqrt(X**2 + Y**2)
phi = np.where(r_from_origin > R, phi, 0)
phi = np.clip(phi * 1e9, -20, 20)

contour = ax3.contourf(X, Y, phi, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax3, label='φ (nV·m)')

# Draw sphere
circle = plt.Circle((0, 0), R, fill=True, color='gray', alpha=0.7)
ax3.add_patch(circle)

# Mark charges
ax3.plot(a, 0, 'ro', markersize=12, label=f'+q at x={a}')
a_prime = R**2 / a
ax3.plot(a_prime, 0, 'bo', markersize=10, alpha=0.5, label=f"Image q' at x={a_prime:.2f}")

ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')
ax3.set_title('Charge Near Grounded Sphere')
ax3.legend()
ax3.set_aspect('equal')
ax3.set_xlim(-4, 5)
ax3.set_ylim(-3, 3)

# ========== Plot 4: Force vs distance ==========
ax4 = axes[1, 1]

# Force between charge and plane
d_vals = np.linspace(0.1, 3, 100)
F_plane = -k * q**2 / (4 * d_vals**2)

# Force between charge and sphere (grounded, R=1)
a_vals = np.linspace(1.1, 3, 100)
q_prime = -q * R / a_vals
a_prime_vals = R**2 / a_vals
F_sphere = k * q * q_prime / (a_vals - a_prime_vals)**2

ax4.plot(d_vals, F_plane * 1e9, 'b-', linewidth=2, label='Near plane')
ax4.plot(a_vals, F_sphere * 1e9, 'r-', linewidth=2, label='Near sphere (R=1)')
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.set_xlabel('Distance d or a (m)')
ax4.set_ylabel('Force (nN)')
ax4.set_title('Force on Charge Near Conductor')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_201_image_charges.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Electric field lines for charge near plane ==========
fig, ax = plt.subplots(figsize=(10, 8))

d = 1.0
q = 1e-9

x = np.linspace(-3, 3, 30)
z = np.linspace(0.05, 4, 30)
X, Z = np.meshgrid(x, z)

# Electric field from real charge
r1 = np.sqrt(X**2 + (Z-d)**2)
Ex1 = k * q * X / r1**3
Ez1 = k * q * (Z-d) / r1**3

# Electric field from image charge
r2 = np.sqrt(X**2 + (Z+d)**2)
Ex2 = -k * q * X / r2**3
Ez2 = -k * q * (Z+d) / r2**3

Ex = Ex1 + Ex2
Ez = Ez1 + Ez2

# Normalize for plotting
E_mag = np.sqrt(Ex**2 + Ez**2)
Ex_norm = Ex / E_mag
Ez_norm = Ez / E_mag

# Stream plot
ax.streamplot(X, Z, Ex_norm, Ez_norm, density=2, color=np.log10(E_mag),
               cmap='viridis', linewidth=1)

# Draw conductor
ax.fill_between([-3, 3], [0, 0], [-0.3, -0.3], color='gray', alpha=0.8)
ax.plot(0, d, 'ro', markersize=15, label='+q')

ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
ax.set_title('Electric Field Lines: Charge Near Conducting Plane')
ax.legend()
ax.set_xlim(-3, 3)
ax.set_ylim(-0.3, 4)

plt.tight_layout()
plt.savefig('day_201_field_lines.png', dpi=150, bbox_inches='tight')
plt.show()

print("Day 201: Method of Images Complete")
print("="*50)
print(f"\nImage charge for plane: -q at z = -d")
print(f"Image charge for sphere: q' = -qR/a at r = R²/a")
print(f"\nForce on charge near plane: F = -q²/(16πε₀d²)")
print(f"Total induced charge = -q (equal to image)")
```

---

## Summary

### Key Formulas

| Configuration | Image Charge | Location |
|--------------|--------------|----------|
| Plane at $z=0$, charge at $z=d$ | $-q$ | $z = -d$ |
| Sphere radius $R$, charge at $a$ | $-qR/a$ | $r = R^2/a$ |

| Formula | Description |
|---------|-------------|
| $\sigma = -\varepsilon_0(\partial\phi/\partial n)$ | Induced surface charge |
| $Q_{\text{ind}} = q'$ | Total induced = image charge |
| $F = q^2/(16\pi\varepsilon_0 d^2)$ | Force near plane |

### Main Takeaways

1. **Image charges** are fictitious charges that reproduce boundary conditions
2. **Uniqueness theorem** guarantees the image solution is correct
3. **Induced charge** equals the image charge magnitude
4. **Forces** can be calculated directly from image configurations
5. **Same technique** underlies Green's function methods in quantum mechanics

---

## Daily Checklist

- [ ] I understand why image charges work
- [ ] I can find images for a charge near a plane
- [ ] I can find images for a charge near a sphere
- [ ] I can calculate induced surface charges
- [ ] I can compute forces on charges near conductors

---

## Preview: Day 202

Tomorrow we study **multipole expansion** — a systematic way to approximate potentials at large distances, essential for understanding atomic and molecular interactions.

---

*"The method of images is perhaps the most elegant trick in all of electrostatics."*

---

**Next:** Day 202 — Multipole Expansion
