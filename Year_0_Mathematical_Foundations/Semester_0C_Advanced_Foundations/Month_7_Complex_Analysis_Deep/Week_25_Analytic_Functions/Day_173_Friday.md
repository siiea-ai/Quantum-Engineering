# Day 173: Conformal Mappings — Angle-Preserving Transformations

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Conformal Maps |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of this day, you will be able to:

1. Define conformal mappings and their geometric properties
2. Prove that analytic functions with nonzero derivative are conformal
3. Work with Möbius (linear fractional) transformations
4. Map between standard domains (disk, half-plane, strip)
5. Introduce the Schwarz-Christoffel transformation
6. Apply conformal mapping to physics problems

---

## Core Content

### 1. Definition and Geometric Meaning

**Definition:** A mapping $w = f(z)$ is **conformal** at $z_0$ if it preserves angles between curves.

More precisely: If two smooth curves $\gamma_1$ and $\gamma_2$ intersect at $z_0$ with angle $\alpha$, then their images $f(\gamma_1)$ and $f(\gamma_2)$ intersect at $f(z_0)$ with the same angle $\alpha$.

**Key Theorem:**

$$\boxed{\text{If } f \text{ is analytic with } f'(z_0) \neq 0, \text{ then } f \text{ is conformal at } z_0.}}$$

**Proof Sketch:**

Near $z_0$, the mapping is approximately:
$$f(z) \approx f(z_0) + f'(z_0)(z - z_0)$$

This is a linear map: rotation by $\arg(f'(z_0))$ and scaling by $|f'(z_0)|$. Linear maps with nonzero determinant preserve angles.

**Converse:** In 2D, a smooth orientation-preserving conformal map is analytic.

### 2. Local Behavior of Conformal Maps

At a point where $f'(z_0) \neq 0$:

- **Rotation angle:** $\arg(f'(z_0))$
- **Scale factor:** $|f'(z_0)|$

**Critical Points:** Where $f'(z_0) = 0$:
- Angles are multiplied, not preserved
- If $f'(z_0) = 0$ but $f''(z_0) \neq 0$, angles are doubled

**Example:** $f(z) = z^2$

$f'(z) = 2z$, so $f'(0) = 0$.

At origin: angle between curves is doubled under $z^2$.
Away from origin: conformal (angles preserved).

### 3. Möbius Transformations

**Definition:** A **Möbius (linear fractional) transformation** is:

$$\boxed{T(z) = \frac{az + b}{cz + d}, \quad ad - bc \neq 0}$$

where $a, b, c, d \in \mathbb{C}$.

**Key Properties:**

1. **Always conformal** (except at $z = -d/c$ where it's undefined)

2. **Maps circles/lines to circles/lines** (generalized circles)

3. **Invertible:**
$$T^{-1}(w) = \frac{dw - b}{-cw + a}$$

4. **Group structure:** Composition of Möbius is Möbius

5. **Three points determine the map:** Given $(z_1, z_2, z_3) \mapsto (w_1, w_2, w_3)$, there's a unique Möbius transformation

**Cross-Ratio:** Preserved by Möbius transformations:

$$\boxed{(z_1, z_2; z_3, z_4) = \frac{(z_1 - z_3)(z_2 - z_4)}{(z_1 - z_4)(z_2 - z_3)}}$$

**Special Cases:**

| Type | Form | Description |
|------|------|-------------|
| Translation | $z + b$ | Shift by $b$ |
| Rotation | $e^{i\theta}z$ | Rotate by $\theta$ |
| Dilation | $\lambda z$ | Scale by $\lambda > 0$ |
| Inversion | $1/z$ | Invert through unit circle |

### 4. Standard Mappings

**Unit Disk to Upper Half-Plane:**

$$\boxed{w = i\frac{1 + z}{1 - z}}$$

Maps:
- $|z| = 1 \to$ real axis
- $|z| < 1 \to$ upper half-plane
- $z = 0 \to w = i$
- $z = -1 \to w = 0$
- $z = 1 \to w = \infty$

**Upper Half-Plane to Unit Disk:**

$$w = \frac{z - i}{z + i}$$

**Disk to Disk (Moving Center):**

To map unit disk to itself with $z = a$ going to $w = 0$:

$$\boxed{w = \frac{z - a}{1 - \bar{a}z}, \quad |a| < 1}$$

This is an **automorphism** of the disk.

**Half-Plane to Strip:**

$$w = \log z$$

maps upper half-plane to horizontal strip $0 < \text{Im}(w) < \pi$.

### 5. Schwarz-Christoffel Transformation (Introduction)

Maps the upper half-plane to the interior of a polygon.

**General Form:**

$$\boxed{\frac{dw}{dz} = A \prod_{k=1}^{n} (z - x_k)^{\alpha_k - 1}}$$

where:
- $x_k$ are pre-images of polygon vertices on real axis
- $\alpha_k \pi$ = interior angle at $k$th vertex
- $A$ is a scaling constant

**Example: Half-Plane to Semi-infinite Strip**

For a strip of width $\pi$:
$$w = \log(z)$$

turns upper half-plane into $0 < \text{Im}(w) < \pi$.

### 6. Applications in Physics

**Electrostatics:**

Laplace's equation is invariant under conformal maps. This allows:
1. Solve problem in simple geometry (disk)
2. Conformally map to complex geometry
3. Solution transforms correctly

**Fluid Dynamics:**

For irrotational, incompressible flow:
- Velocity potential $\phi$ is harmonic
- Stream function $\psi$ is harmonic conjugate
- $f = \phi + i\psi$ is analytic (complex potential)

**Joukowsky Transformation:**

$$\boxed{w = z + \frac{1}{z}}$$

- Maps circle to airfoil shape
- Used in aerodynamics to compute flow around wings

### 7. Quantum Mechanics Connection

**2D Quantum Systems:**

Conformal mappings transform the Schrödinger equation:

$$-\frac{\hbar^2}{2m}\nabla^2\psi = E\psi$$

The Laplacian transforms as:
$$\nabla^2_w = \frac{1}{|f'(z)|^2}\nabla^2_z$$

This allows solving problems in complex geometries by:
1. Map to simple domain
2. Solve in simple domain
3. Transform solution back

**Conformal Field Theory (CFT):**

In 2D quantum field theory, conformal symmetry is powerful:
- Exactly solvable models
- Critical phenomena
- String theory
- Condensed matter physics

---

## Worked Examples

### Example 1: Verify Conformality

**Problem:** Show $f(z) = e^z$ is conformal everywhere.

**Solution:**

$f'(z) = e^z \neq 0$ for all $z \in \mathbb{C}$.

Since $f$ is analytic with nonzero derivative, $f$ is conformal everywhere. ✓

At each point:
- Rotation angle: $\arg(e^z) = \text{Im}(z) = y$
- Scale factor: $|e^z| = e^x$

### Example 2: Möbius Transformation

**Problem:** Find the Möbius transformation that maps:
- $0 \to 1$
- $1 \to \infty$
- $\infty \to 0$

**Solution:**

Let $T(z) = \frac{az + b}{cz + d}$.

From $T(0) = 1$: $\frac{b}{d} = 1 \Rightarrow b = d$

From $T(1) = \infty$: $c + d = 0 \Rightarrow c = -d$

From $T(\infty) = 0$: $\frac{a}{c} = 0 \Rightarrow a = 0$

So $T(z) = \frac{d}{-dz + d} = \frac{1}{1 - z}$

**Verification:**
- $T(0) = 1$ ✓
- $T(1) = \infty$ ✓
- $T(\infty) = \lim_{z\to\infty} \frac{1}{1-z} = 0$ ✓

### Example 3: Mapping Disk to Half-Plane

**Problem:** Map the unit disk to the right half-plane $\text{Re}(w) > 0$.

**Solution:**

Start with standard disk-to-upper-half-plane:
$$\zeta = i\frac{1 + z}{1 - z}$$

Rotate by $-90°$ to get right half-plane:
$$w = e^{-i\pi/2} \cdot \zeta = -i \cdot i\frac{1 + z}{1 - z} = \frac{1 + z}{1 - z}$$

**Verification:**
- $z = 0 \to w = 1$ (center to right half-plane) ✓
- $z = 1 \to w = \infty$ ✓
- $z = -1 \to w = 0$ (boundary to boundary) ✓
- $z = i \to w = \frac{1+i}{1-i} = \frac{(1+i)^2}{2} = i$ (on boundary)

---

## Practice Problems

### Level 1: Direct Application

1. Verify that $f(z) = z^3$ is conformal at all $z \neq 0$.

2. Find the image of the unit circle under $w = 2z + 1$.

3. Find the Möbius transformation mapping $0 \to 0$, $1 \to 1$, $\infty \to \infty$.

### Level 2: Intermediate

4. Map the first quadrant ($x > 0$, $y > 0$) to the upper half-plane.

5. Show that the Joukowsky map $w = z + 1/z$ is conformal everywhere except at $z = \pm 1$.

6. Find the automorphism of the unit disk that maps $z = 1/2$ to $w = 0$.

### Level 3: Challenging

7. **Riemann Mapping Theorem:** State the theorem and explain why it's remarkable. What does it imply about the geometry of simply-connected domains?

8. Map the region outside the unit circle to the upper half-plane. (Hint: First invert, then use standard maps.)

9. For a quantum particle in a 2D domain $D$, explain how conformal mapping $f: D \to D'$ transforms the energy eigenvalues. Are they preserved?

---

## Computational Lab

```python
"""
Day 173: Conformal Mappings
Visualizing transformations and their angle-preserving property
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_conformal_map(f, title, ax, xlim=(-2, 2), ylim=(-2, 2)):
    """Plot a conformal mapping showing grid transformation"""
    # Create grid in z-plane
    x = np.linspace(xlim[0], xlim[1], 20)
    y = np.linspace(ylim[0], ylim[1], 20)

    # Plot vertical lines and their images
    for xi in x[::2]:
        z = xi + 1j*y
        w = f(z)
        ax[0].plot(z.real, z.imag, 'b-', linewidth=0.5)
        ax[1].plot(w.real, w.imag, 'b-', linewidth=0.5)

    # Plot horizontal lines and their images
    for yi in y[::2]:
        z = x + 1j*yi
        w = f(z)
        ax[0].plot(z.real, z.imag, 'r-', linewidth=0.5)
        ax[1].plot(w.real, w.imag, 'r-', linewidth=0.5)

    ax[0].set_title(f'{title}: z-plane')
    ax[0].set_xlabel('Re(z)')
    ax[0].set_ylabel('Im(z)')
    ax[0].set_aspect('equal')
    ax[0].grid(True, alpha=0.3)

    ax[1].set_title(f'{title}: w-plane')
    ax[1].set_xlabel('Re(w)')
    ax[1].set_ylabel('Im(w)')
    ax[1].set_aspect('equal')
    ax[1].grid(True, alpha=0.3)

# ========================================
# FIGURE 1: BASIC CONFORMAL MAPS
# ========================================
fig1, axes1 = plt.subplots(2, 4, figsize=(18, 9))

# 1. f(z) = z² (squares the plane)
def f_square(z):
    return z**2

plot_conformal_map(f_square, 'w = z²', axes1[:, 0])

# 2. f(z) = e^z
def f_exp(z):
    return np.exp(z)

plot_conformal_map(f_exp, 'w = e^z', axes1[:, 1], xlim=(-2, 2), ylim=(-np.pi, np.pi))
axes1[1, 1].set_xlim(-1, 8)
axes1[1, 1].set_ylim(-8, 8)

# 3. f(z) = 1/z (inversion)
def f_invert(z):
    return 1/(z + 0.01 + 0.01j)  # Avoid singularity

plot_conformal_map(f_invert, 'w = 1/z', axes1[:, 2])

# 4. Möbius: w = (z-1)/(z+1)
def f_mobius(z):
    return (z - 1)/(z + 1 + 0.01j)

plot_conformal_map(f_mobius, 'w = (z-1)/(z+1)', axes1[:, 3])
axes1[1, 3].set_xlim(-3, 3)
axes1[1, 3].set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('day_173_conformal_maps.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# FIGURE 2: DISK TO HALF-PLANE
# ========================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Unit disk grid
theta = np.linspace(0, 2*np.pi, 100)
r_values = np.linspace(0.1, 0.9, 9)

ax = axes2[0]
# Radial lines
for angle in np.linspace(0, 2*np.pi, 13)[:-1]:
    r = np.linspace(0, 1, 50)
    z = r * np.exp(1j*angle)
    ax.plot(z.real, z.imag, 'b-', linewidth=0.8)

# Concentric circles
for r in r_values:
    z = r * np.exp(1j*theta)
    ax.plot(z.real, z.imag, 'r-', linewidth=0.8)

# Unit circle boundary
ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
ax.set_title('Unit Disk (z-plane)', fontsize=12)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Map to upper half-plane
ax = axes2[1]

def disk_to_halfplane(z):
    return 1j * (1 + z) / (1 - z + 0.001)

# Radial lines map to circles
for angle in np.linspace(0, 2*np.pi, 13)[:-1]:
    r = np.linspace(0, 0.99, 100)
    z = r * np.exp(1j*angle)
    w = disk_to_halfplane(z)
    # Filter out large values
    mask = np.abs(w) < 10
    ax.plot(w.real[mask], w.imag[mask], 'b-', linewidth=0.8)

# Concentric circles map to circles
for r in r_values:
    z = r * np.exp(1j*theta)
    w = disk_to_halfplane(z)
    ax.plot(w.real, w.imag, 'r-', linewidth=0.8)

ax.axhline(y=0, color='k', linewidth=2)  # Real axis = image of unit circle
ax.set_title('Upper Half-Plane (w-plane)', fontsize=12)
ax.set_xlabel('Re(w)')
ax.set_ylabel('Im(w)')
ax.set_xlim(-5, 5)
ax.set_ylim(-0.5, 5)

plt.tight_layout()
plt.savefig('day_173_disk_to_halfplane.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# FIGURE 3: JOUKOWSKY AIRFOIL
# ========================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

def joukowsky(z):
    return z + 1/z

# Circle slightly off-center
ax = axes3[0]
theta = np.linspace(0, 2*np.pi, 500)
R = 1.1  # Radius slightly > 1
center = -0.1 + 0.1j  # Slightly off-center
z_circle = center + R * np.exp(1j*theta)

ax.plot(z_circle.real, z_circle.imag, 'b-', linewidth=2, label='Circle')
ax.plot(center.real, center.imag, 'ro', markersize=8, label='Center')
ax.plot([-1, 1], [0, 0], 'k*', markersize=10)  # Critical points
ax.set_title('z-plane: Circle (off-center)', fontsize=12)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

# Joukowsky transform
ax = axes3[1]
w_airfoil = joukowsky(z_circle)
ax.plot(w_airfoil.real, w_airfoil.imag, 'b-', linewidth=2, label='Airfoil')
ax.fill(w_airfoil.real, w_airfoil.imag, alpha=0.3)
ax.set_title('w-plane: Joukowsky Airfoil', fontsize=12)
ax.set_xlabel('Re(w)')
ax.set_ylabel('Im(w)')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_173_joukowsky.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# ANGLE PRESERVATION DEMONSTRATION
# ========================================
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

ax = axes4[0]
# Two curves meeting at z=1 at 90 degrees
t = np.linspace(-1, 1, 100)
curve1 = 1 + t  # Horizontal line
curve2 = 1 + 1j*t  # Vertical line

ax.plot(curve1.real, curve1.imag, 'b-', linewidth=2, label='Horizontal')
ax.plot(curve2.real, curve2.imag, 'r-', linewidth=2, label='Vertical')
ax.plot(1, 0, 'ko', markersize=10)
ax.annotate('90°', (1.1, 0.15), fontsize=12)
ax.set_title('z-plane: Two curves at 90°', fontsize=11)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

# Under w = z², angle is doubled to 180° at origin (critical point)
# But let's show at a non-critical point
ax = axes4[1]
w1 = curve1**2
w2 = curve2**2

ax.plot(w1.real, w1.imag, 'b-', linewidth=2, label='Image of horizontal')
ax.plot(w2.real, w2.imag, 'r-', linewidth=2, label='Image of vertical')
ax.plot(1, 0, 'ko', markersize=10)  # Image of z=1
ax.annotate('90° preserved!', (1.2, 0.3), fontsize=12)
ax.set_title('w-plane under w=z²: Angles preserved (at z≠0)', fontsize=11)
ax.set_xlabel('Re(w)')
ax.set_ylabel('Im(w)')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 3)
ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('day_173_angles.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("CONFORMAL MAPPINGS - VISUALIZATION COMPLETE")
print("=" * 60)
print("\nKey Maps Demonstrated:")
print("• w = z² - squares the plane")
print("• w = e^z - strips to half-plane")
print("• w = 1/z - inversion")
print("• Möbius transformations - circle/line to circle/line")
print("• Disk ↔ Half-plane standard mapping")
print("• Joukowsky: circle → airfoil")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Conformal condition | $f$ analytic with $f'(z) \neq 0$ |
| Möbius transformation | $T(z) = \frac{az+b}{cz+d}$, $ad-bc \neq 0$ |
| Disk → Half-plane | $w = i\frac{1+z}{1-z}$ |
| Disk automorphism | $w = \frac{z-a}{1-\bar{a}z}$ |
| Joukowsky | $w = z + 1/z$ |
| Cross-ratio | $(z_1,z_2;z_3,z_4) = \frac{(z_1-z_3)(z_2-z_4)}{(z_1-z_4)(z_2-z_3)}$ |

### Main Takeaways

1. **Conformal maps** preserve angles between curves
2. **Analytic + nonzero derivative** = conformal
3. **Möbius transformations** form a group, map circles to circles
4. **Standard mappings** connect disk, half-plane, and strip
5. **Physics applications:** electrostatics, fluid flow, quantum systems

---

## Daily Checklist

- [ ] I understand why analytic functions are conformal (when $f' \neq 0$)
- [ ] I can work with Möbius transformations
- [ ] I can map between disk and half-plane
- [ ] I understand the Joukowsky transformation
- [ ] I can explain applications in physics
- [ ] I completed the computational lab

---

## Preview: Day 174

Tomorrow: **Computational Lab** — Putting it all together!
- Numerical verification of complex analysis theorems
- Visualization projects
- Integration with quantum mechanics
- Building intuition through interactive exploration

---

*"The essence of mathematics lies in its freedom."*
— Georg Cantor
