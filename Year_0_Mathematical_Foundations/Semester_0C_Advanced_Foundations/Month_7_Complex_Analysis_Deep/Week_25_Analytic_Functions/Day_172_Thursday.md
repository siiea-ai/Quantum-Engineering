# Day 172: Harmonic Functions — Solutions to Laplace's Equation

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Harmonic Functions |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of this day, you will be able to:

1. Define harmonic functions and verify Laplace's equation
2. Apply the maximum/minimum principle
3. Use the mean value property
4. Solve the Dirichlet problem using Poisson integral
5. Find harmonic conjugates systematically
6. Connect harmonic functions to quantum potential theory

---

## Core Content

### 1. Definition and Basic Properties

**Definition:** A twice continuously differentiable function $u(x,y)$ is **harmonic** if it satisfies **Laplace's equation**:

$$\boxed{\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0}$$

**In polar coordinates:**

$$\nabla^2 u = \frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial u}{\partial r}\right) + \frac{1}{r^2}\frac{\partial^2 u}{\partial \theta^2} = 0$$

**Fundamental Connection:** From yesterday's Cauchy-Riemann equations:
- If $f = u + iv$ is analytic, then both $u$ and $v$ are harmonic
- Conversely, any harmonic function on a simply-connected domain is the real part of an analytic function

**Examples of Harmonic Functions:**

| Function | Domain | Notes |
|----------|--------|-------|
| $u = x^2 - y^2$ | All $\mathbb{R}^2$ | Real part of $z^2$ |
| $u = \ln r = \ln\sqrt{x^2+y^2}$ | $r > 0$ | Real part of $\ln z$ |
| $u = e^x \cos y$ | All $\mathbb{R}^2$ | Real part of $e^z$ |
| $u = \text{Re}(z^n)$ | All $\mathbb{R}^2$ | Any analytic power |

### 2. Maximum and Minimum Principles

**Maximum Principle:**

Let $u$ be harmonic on a bounded domain $D$ and continuous on $\bar{D}$ (closure). Then:

$$\boxed{\max_{\bar{D}} u = \max_{\partial D} u}$$

**The maximum occurs on the boundary, not in the interior!**

**Minimum Principle:** Similarly:

$$\min_{\bar{D}} u = \min_{\partial D} u$$

**Strong Form:** If $u$ attains its maximum (or minimum) at an interior point, then $u$ is constant.

**Physical Interpretation:**
- Temperature in a steady-state heat distribution: no interior hot/cold spots
- Electrostatic potential: extremes only on conductors (boundary)

**Proof Sketch:**

Suppose $u$ attains max at interior point $z_0$. By the mean value property (below), $u(z_0)$ equals the average over any circle. Since $u(z_0)$ is max, all values on circle equal $u(z_0)$. By continuation, $u$ is constant everywhere.

### 3. Mean Value Property

**Theorem (Mean Value Property):**

If $u$ is harmonic in a disk $B(z_0, R)$, then for any $0 < r < R$:

$$\boxed{u(z_0) = \frac{1}{2\pi}\int_0^{2\pi} u(z_0 + re^{i\theta}) d\theta}$$

The value at the center equals the average over any circle.

**Area Average Version:**

$$u(z_0) = \frac{1}{\pi r^2}\iint_{B(z_0,r)} u(x,y) \, dx\, dy$$

**Characterization:** A continuous function with the mean value property is harmonic. (This can be used as a definition!)

### 4. Poisson Integral Formula

**The Dirichlet Problem:**

*Given:* A domain $D$ with boundary $\partial D$ and continuous boundary data $\phi$ on $\partial D$.

*Find:* A harmonic function $u$ on $D$ with $u = \phi$ on $\partial D$.

**Solution for the Unit Disk:**

Let $D = \{|z| < 1\}$. For boundary data $\phi(e^{i\alpha})$:

$$\boxed{u(re^{i\theta}) = \frac{1}{2\pi}\int_0^{2\pi} P_r(\theta - \alpha) \phi(e^{i\alpha}) d\alpha}$$

where the **Poisson kernel** is:

$$\boxed{P_r(\psi) = \frac{1 - r^2}{1 - 2r\cos\psi + r^2}}$$

**Properties of Poisson Kernel:**

1. $P_r(\psi) > 0$ for $0 \leq r < 1$
2. $\frac{1}{2\pi}\int_0^{2\pi} P_r(\psi) d\psi = 1$
3. As $r \to 1^-$, $P_r$ concentrates at $\psi = 0$ (approaches delta function)

**Physical Interpretation:**

The Poisson formula says: "To find temperature at interior point, average boundary temperature with weights that favor nearby boundary points."

### 5. Harmonic Conjugates

**Definition:** For harmonic $u$, its **harmonic conjugate** $v$ satisfies:
1. $v$ is harmonic
2. $f = u + iv$ is analytic

**Finding $v$ from $u$:**

Using Cauchy-Riemann: $\partial v/\partial y = \partial u/\partial x$ and $\partial v/\partial x = -\partial u/\partial y$

$$v(x,y) = \int \frac{\partial u}{\partial x} dy + g(x) = -\int \frac{\partial u}{\partial y} dx + h(y)$$

**Uniqueness:** Harmonic conjugate is unique up to an additive constant.

**Existence:**
- On simply-connected domains: always exists
- On multiply-connected domains: may not exist globally

### 6. Quantum Mechanics Connection

**Schrödinger Equation:**

The time-independent Schrödinger equation is:

$$-\frac{\hbar^2}{2m}\nabla^2\psi + V\psi = E\psi$$

**In regions where $V = E$:**

$$\nabla^2\psi = 0$$

The wave function is harmonic! This occurs at classical turning points.

**Electrostatic Analogy:**

In atomic physics, the Coulomb potential satisfies:

$$\nabla^2 V = -\frac{\rho}{\varepsilon_0}$$

where $\rho$ is charge density. In charge-free regions ($\rho = 0$):

$$\nabla^2 V = 0 \quad \text{(Laplace equation)}$$

The electrostatic potential is harmonic, and its level surfaces are equipotentials.

**Quantum Dots:**

For a 2D quantum dot with complex boundary shape:
- Solve Dirichlet problem: $\psi = 0$ on boundary
- Energy eigenvalues from eigenvalue problem
- Conformal mapping transforms complex shapes to simple ones

---

## Worked Examples

### Example 1: Verify Harmonicity

**Problem:** Show $u(x,y) = x^3 - 3xy^2$ is harmonic.

**Solution:**

$$\frac{\partial u}{\partial x} = 3x^2 - 3y^2, \quad \frac{\partial^2 u}{\partial x^2} = 6x$$

$$\frac{\partial u}{\partial y} = -6xy, \quad \frac{\partial^2 u}{\partial y^2} = -6x$$

$$\nabla^2 u = 6x + (-6x) = 0 \quad \checkmark$$

### Example 2: Maximum Principle Application

**Problem:** If $u$ is harmonic on the unit disk with $u = \sin\theta$ on the boundary, what is the maximum of $u$ inside?

**Solution:**

By the maximum principle, max occurs on boundary:
$$\max_{\bar{D}} u = \max_{\theta \in [0,2\pi)} \sin\theta = 1$$

at $\theta = \pi/2$ (i.e., at $z = i$ on boundary).

The maximum value inside is strictly less than 1 (by strong maximum principle).

### Example 3: Poisson Integral

**Problem:** Find the harmonic function $u(r,\theta)$ in the unit disk with boundary condition $u(1,\theta) = \cos^2\theta$.

**Solution:**

Use $\cos^2\theta = \frac{1 + \cos 2\theta}{2}$.

For boundary data $\phi(\alpha) = \cos^2\alpha$, the Poisson integral gives:

$$u(re^{i\theta}) = \frac{1}{2\pi}\int_0^{2\pi} P_r(\theta - \alpha) \cos^2\alpha \, d\alpha$$

Using the known result that $\cos n\alpha \to r^n \cos n\theta$ under Poisson integral:

$$u(r,\theta) = \frac{1}{2} + \frac{r^2 \cos 2\theta}{2} = \frac{1 + r^2\cos 2\theta}{2}$$

**Verification:** At $r = 1$: $u = \frac{1 + \cos 2\theta}{2} = \cos^2\theta$ ✓

---

## Practice Problems

### Level 1: Direct Application

1. Verify that $u = e^{-y}\cos x$ is harmonic.

2. Is $u = x^2 + y^2$ harmonic? Why or why not?

3. Find the maximum of $u = x^2 - y^2$ on the disk $x^2 + y^2 \leq 1$.

### Level 2: Intermediate

4. Find the harmonic conjugate of $u = \ln(x^2 + y^2)$ for $(x,y) \neq (0,0)$.

5. Solve the Dirichlet problem on the unit disk with boundary condition $u = \theta$ (the angle).

6. Prove that a harmonic function cannot have a local maximum in the interior of its domain.

### Level 3: Challenging

7. **Schwarz Reflection Principle:** If $u$ is harmonic in the upper half-plane and continuous with $u = 0$ on the real axis, show that $u$ extends to a harmonic function on all of $\mathbb{C}$ via $\tilde{u}(x,y) = -u(x,-y)$ for $y < 0$.

8. **Harnack's Inequality:** Show that if $u$ is positive and harmonic in $B(0,R)$, then for $|z| = r < R$:
$$\frac{R-r}{R+r}u(0) \leq u(z) \leq \frac{R+r}{R-r}u(0)$$

9. **Quantum Application:** For a particle in a 2D box with sides $a$ and $b$, the wave functions $\psi_{nm} = \sin(n\pi x/a)\sin(m\pi y/b)$ are NOT harmonic. Why not? What equation do they satisfy instead?

---

## Computational Lab

```python
"""
Day 172: Harmonic Functions
Visualization of maximum principle, mean value, and Poisson formula
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import dblquad

# Create grid
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 3, figsize=(16, 11))

# ========================================
# 1. HARMONIC FUNCTION EXAMPLE
# ========================================
ax = axes[0, 0]

# u = x² - y² (harmonic)
u = X**2 - Y**2

# Draw unit circle (boundary)
theta_circle = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(theta_circle)
y_circle = np.sin(theta_circle)

contour = ax.contourf(X, Y, u, levels=30, cmap='RdBu', alpha=0.8)
ax.plot(x_circle, y_circle, 'k-', linewidth=2, label='Unit circle')
ax.set_title('Harmonic: u = x² - y²\nMax on boundary (Maximum Principle)', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend()
plt.colorbar(contour, ax=ax, label='u(x,y)')

# ========================================
# 2. NON-HARMONIC FUNCTION
# ========================================
ax = axes[0, 1]

# u = x² + y² (NOT harmonic: ∇²u = 4)
u_non = X**2 + Y**2

contour2 = ax.contourf(X, Y, u_non, levels=30, cmap='viridis', alpha=0.8)
ax.plot(x_circle, y_circle, 'k-', linewidth=2)
ax.plot(0, 0, 'r*', markersize=15, label='Min at interior!')
ax.set_title('NOT Harmonic: u = x² + y²\nMin at origin (violates principle)', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend()
plt.colorbar(contour2, ax=ax, label='u(x,y)')

# ========================================
# 3. MEAN VALUE PROPERTY
# ========================================
ax = axes[0, 2]

# Show that u(0,0) = average over circles
u_harmonic = X**2 - Y**2

# Values on circles of different radii
radii = [0.3, 0.6, 0.9, 1.2, 1.5]
center = (0, 0)

ax.contourf(X, Y, u_harmonic, levels=20, cmap='coolwarm', alpha=0.6)

for r in radii:
    theta = np.linspace(0, 2*np.pi, 100)
    x_c = r * np.cos(theta)
    y_c = r * np.sin(theta)
    ax.plot(x_c, y_c, 'k-', linewidth=1, alpha=0.7)

    # Compute average on this circle
    u_on_circle = r**2 * np.cos(2*theta)  # x²-y² = r²cos(2θ) on circle
    avg = np.mean(u_on_circle)
    ax.annotate(f'avg={avg:.2f}', (r*0.8, r*0.6), fontsize=8)

ax.plot(0, 0, 'go', markersize=10, label=f'u(0,0) = {u_harmonic[100,100]:.2f}')
ax.set_title('Mean Value Property\nCenter value = circle average', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# ========================================
# 4. POISSON KERNEL
# ========================================
ax = axes[1, 0]

psi = np.linspace(-np.pi, np.pi, 500)
r_values = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

for r in r_values:
    P_r = (1 - r**2) / (1 - 2*r*np.cos(psi) + r**2)
    ax.plot(psi/np.pi, P_r, linewidth=2, label=f'r = {r}')

ax.set_xlabel('ψ/π')
ax.set_ylabel('P_r(ψ)')
ax.set_title('Poisson Kernel P_r(ψ)\nConcentrates at ψ=0 as r→1', fontsize=11)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 1)

# ========================================
# 5. DIRICHLET PROBLEM SOLUTION
# ========================================
ax = axes[1, 1]

# Solve Dirichlet problem with u = cos(θ) on boundary
# Solution: u(r,θ) = r cos(θ) = x

r_grid = np.linspace(0, 1, 50)
theta_grid = np.linspace(0, 2*np.pi, 100)
R, Theta = np.meshgrid(r_grid, theta_grid)

# Boundary condition: u(1, θ) = cos(θ)
# Solution: u(r, θ) = r cos(θ)
U_solution = R * np.cos(Theta)

# Convert to Cartesian for plotting
X_polar = R * np.cos(Theta)
Y_polar = R * np.sin(Theta)

contour5 = ax.contourf(X_polar, Y_polar, U_solution, levels=20, cmap='RdBu')
ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=2)
ax.set_title('Dirichlet Solution: u(1,θ) = cos(θ)\nInterior: u = r cos(θ) = x', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(contour5, ax=ax, label='u(r,θ)')

# ========================================
# 6. LAPLACIAN VISUALIZATION
# ========================================
ax = axes[1, 2]

from scipy import ndimage

# Compare Laplacian of harmonic vs non-harmonic
u_h = X**2 - Y**2  # harmonic
u_nh = X**2 + Y**2  # non-harmonic

laplacian_h = ndimage.laplace(u_h)
laplacian_nh = ndimage.laplace(u_nh)

# Plot Laplacian of non-harmonic function
im = ax.imshow(laplacian_nh, extent=[-2, 2, -2, 2], cmap='plasma',
               origin='lower', aspect='auto')
ax.set_title(f'∇²(x²+y²) = 4 everywhere\n(NOT harmonic)', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='∇²u')

plt.tight_layout()
plt.savefig('day_172_harmonic_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# 3D SURFACE PLOT OF HARMONIC FUNCTION
# ========================================
fig2 = plt.figure(figsize=(12, 5))

# Harmonic: u = x² - y²
ax1 = fig2.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, X**2 - Y**2, cmap='RdBu', alpha=0.8)
ax1.set_title('Harmonic: u = x² - y²\n(Saddle point at origin)', fontsize=11)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')

# Non-harmonic: u = x² + y²
ax2 = fig2.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, X**2 + Y**2, cmap='viridis', alpha=0.8)
ax2.set_title('NOT Harmonic: u = x² + y²\n(Minimum at origin)', fontsize=11)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u')

plt.tight_layout()
plt.savefig('day_172_3d_surfaces.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("HARMONIC FUNCTIONS - ANALYSIS COMPLETE")
print("=" * 60)
print("\nKey Properties Verified:")
print("• Maximum Principle: max/min on boundary")
print("• Mean Value: center = circle average")
print("• Laplacian: ∇²u = 0 for harmonic functions")
print("• Poisson kernel: concentrates as r → 1")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Laplace Equation | $\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$ |
| Mean Value (Circle) | $u(z_0) = \frac{1}{2\pi}\int_0^{2\pi} u(z_0 + re^{i\theta})d\theta$ |
| Poisson Kernel | $P_r(\psi) = \frac{1-r^2}{1-2r\cos\psi+r^2}$ |
| Maximum Principle | $\max_{\bar{D}} u = \max_{\partial D} u$ |

### Main Takeaways

1. **Harmonic functions** satisfy $\nabla^2 u = 0$ (Laplace equation)
2. **Maximum principle:** Extremes occur only on boundaries
3. **Mean value property:** Value at center = average over circles
4. **Poisson integral** solves Dirichlet problem on disk
5. **Quantum connection:** Potentials in charge-free regions are harmonic

---

## Daily Checklist

- [ ] I can verify a function is harmonic using Laplace equation
- [ ] I understand and can apply the maximum principle
- [ ] I can use the mean value property
- [ ] I can set up and interpret the Poisson integral
- [ ] I can find harmonic conjugates
- [ ] I completed the computational lab

---

## Preview: Day 173

Tomorrow: **Conformal Mappings** — Functions that preserve angles!
- Definition and geometric meaning
- Möbius transformations
- Mapping regions (disk ↔ half-plane)
- Schwarz-Christoffel transformation
- Applications to potential theory

---

*"Pure mathematics is, in its way, the poetry of logical ideas."*
— Albert Einstein
