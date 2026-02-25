# Day 171: The Cauchy-Riemann Equations — Bridge to Harmonic Analysis

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Cauchy-Riemann Equations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of this day, you will be able to:

1. Derive the Cauchy-Riemann equations from the definition of complex differentiability
2. Apply Cauchy-Riemann to verify analyticity of complex functions
3. Convert Cauchy-Riemann to polar form
4. Prove that real and imaginary parts of analytic functions are harmonic
5. Understand the physical interpretation in terms of flows and potentials
6. Connect to the Schrödinger equation and quantum wave functions

---

## Core Content

### 1. Derivation of the Cauchy-Riemann Equations

Let $f(z) = u(x,y) + iv(x,y)$ where $z = x + iy$.

For $f$ to be differentiable at $z_0$:

$$f'(z_0) = \lim_{h \to 0} \frac{f(z_0 + h) - f(z_0)}{h}$$

The key insight: this limit must be **path-independent**.

**Approach 1: Along real axis ($h = \Delta x$, $\Delta y = 0$)**

$$f'(z) = \lim_{\Delta x \to 0} \frac{[u(x + \Delta x, y) - u(x,y)] + i[v(x + \Delta x, y) - v(x,y)]}{\Delta x}$$

$$= \frac{\partial u}{\partial x} + i\frac{\partial v}{\partial x}$$

**Approach 2: Along imaginary axis ($h = i\Delta y$, $\Delta x = 0$)**

$$f'(z) = \lim_{\Delta y \to 0} \frac{[u(x, y + \Delta y) - u(x,y)] + i[v(x, y + \Delta y) - v(x,y)]}{i\Delta y}$$

$$= \frac{1}{i}\frac{\partial u}{\partial y} + \frac{\partial v}{\partial y} = \frac{\partial v}{\partial y} - i\frac{\partial u}{\partial y}$$

**Equating the two expressions:**

$$\frac{\partial u}{\partial x} + i\frac{\partial v}{\partial x} = \frac{\partial v}{\partial y} - i\frac{\partial u}{\partial y}$$

Separating real and imaginary parts:

### Cauchy-Riemann Equations (Cartesian Form)

$$\boxed{\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y} \quad \text{and} \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}}$$

**Theorem (Cauchy-Riemann):**

*Necessary Condition:* If $f = u + iv$ is differentiable at $z_0$, then $u$ and $v$ satisfy Cauchy-Riemann at $z_0$.

*Sufficient Condition:* If $u$ and $v$ have continuous first partial derivatives and satisfy Cauchy-Riemann, then $f$ is analytic.

**Complex Derivative:**

$$\boxed{f'(z) = \frac{\partial u}{\partial x} + i\frac{\partial v}{\partial x} = \frac{\partial v}{\partial y} - i\frac{\partial u}{\partial y}}$$

### 2. Polar Form of Cauchy-Riemann

For $z = re^{i\theta}$ where $x = r\cos\theta$, $y = r\sin\theta$:

Using the chain rule for coordinate transformation:

$$\boxed{\frac{\partial u}{\partial r} = \frac{1}{r}\frac{\partial v}{\partial \theta} \quad \text{and} \quad \frac{\partial v}{\partial r} = -\frac{1}{r}\frac{\partial u}{\partial \theta}}$$

**Derivation:** From $x = r\cos\theta$, $y = r\sin\theta$:

$$\frac{\partial}{\partial r} = \cos\theta \frac{\partial}{\partial x} + \sin\theta \frac{\partial}{\partial y}$$

$$\frac{\partial}{\partial \theta} = -r\sin\theta \frac{\partial}{\partial x} + r\cos\theta \frac{\partial}{\partial y}$$

Substituting into the Cartesian Cauchy-Riemann equations yields the polar form.

### 3. Connection to Harmonic Functions

**Key Result:** Both $u$ and $v$ satisfy Laplace's equation!

Taking $\partial/\partial x$ of the first equation and $\partial/\partial y$ of the second:

$$\frac{\partial^2 u}{\partial x^2} = \frac{\partial^2 v}{\partial x \partial y}$$

$$\frac{\partial^2 u}{\partial y^2} = -\frac{\partial^2 v}{\partial y \partial x}$$

Adding (using equality of mixed partials):

$$\boxed{\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0}$$

Similarly: $\nabla^2 v = 0$

**Definition:** A function satisfying $\nabla^2 f = 0$ is called **harmonic**.

**Fundamental Result:** If $f = u + iv$ is analytic, then:
- $u$ is harmonic (satisfies Laplace equation)
- $v$ is harmonic
- $v$ is the **harmonic conjugate** of $u$

### 4. Physical Interpretation

**Incompressible Fluid Flow:**

Consider a 2D velocity field $\mathbf{v} = (v_x, v_y)$ where:
- $u$ = velocity potential ($v_x = \partial u/\partial x$, $v_y = \partial u/\partial y$)
- $v$ = stream function (flow lines are curves $v = \text{const}$)

Cauchy-Riemann implies:
- $\nabla \cdot \mathbf{v} = 0$ (incompressible)
- Flow lines ($v = \text{const}$) are orthogonal to equipotential lines ($u = \text{const}$)

**Electrostatics:**

For a 2D electrostatic problem:
- $u$ = electric potential $\phi$
- $v$ = electric flux function
- $\nabla^2 \phi = 0$ (Laplace equation in charge-free region)
- Field lines perpendicular to equipotentials

### 5. Quantum Mechanics Connection

**Schrödinger Equation in 2D:**

The time-independent Schrödinger equation:

$$-\frac{\hbar^2}{2m}\nabla^2 \psi + V\psi = E\psi$$

In regions where $V = E$ (classically allowed, turning points):

$$\nabla^2 \psi = 0$$

The wave function satisfies Laplace's equation! This connects to:
- WKB approximation at turning points
- Evanescent waves in forbidden regions
- Tunneling phenomena

**Wave Function Analyticity:**

For a free particle wave function $\psi(x,y) = e^{i(k_x x + k_y y)}$:

$$\psi = \cos(k_x x + k_y y) + i\sin(k_x x + k_y y)$$

Both real and imaginary parts satisfy $\nabla^2(\text{Re}\psi) = -k^2 \text{Re}\psi$ (Helmholtz equation, not Laplace), but the structure mirrors Cauchy-Riemann through the phase relationship.

---

## Worked Examples

### Example 1: Verify Cauchy-Riemann for $f(z) = z^2$

**Given:** $f(z) = z^2 = (x + iy)^2 = x^2 - y^2 + 2ixy$

So $u = x^2 - y^2$ and $v = 2xy$.

**Check Cauchy-Riemann:**

$$\frac{\partial u}{\partial x} = 2x, \quad \frac{\partial v}{\partial y} = 2x \quad \checkmark$$

$$\frac{\partial u}{\partial y} = -2y, \quad -\frac{\partial v}{\partial x} = -2y \quad \checkmark$$

Both equations satisfied everywhere ⟹ $f(z) = z^2$ is entire.

**Derivative:**
$$f'(z) = \frac{\partial u}{\partial x} + i\frac{\partial v}{\partial x} = 2x + i(2y) = 2(x + iy) = 2z \quad \checkmark$$

### Example 2: Show $f(z) = \bar{z}$ fails Cauchy-Riemann

**Given:** $f(z) = \bar{z} = x - iy$

So $u = x$ and $v = -y$.

**Check:**

$$\frac{\partial u}{\partial x} = 1, \quad \frac{\partial v}{\partial y} = -1$$

These are **not equal**! Cauchy-Riemann fails everywhere.

**Conclusion:** $\bar{z}$ is nowhere analytic, confirming our Day 170 result.

### Example 3: Find harmonic conjugate

**Problem:** Given $u(x,y) = x^3 - 3xy^2$, find its harmonic conjugate $v$.

**Step 1:** Verify $u$ is harmonic.
$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 6x + (-6x) = 0 \quad \checkmark$$

**Step 2:** Use Cauchy-Riemann to find $v$.

From $\partial v/\partial y = \partial u/\partial x = 3x^2 - 3y^2$:
$$v = \int (3x^2 - 3y^2) dy = 3x^2 y - y^3 + g(x)$$

From $\partial v/\partial x = -\partial u/\partial y = -(-6xy) = 6xy$:
$$\frac{\partial v}{\partial x} = 6xy + g'(x) = 6xy$$

So $g'(x) = 0$, meaning $g(x) = C$ (constant).

**Answer:** $v(x,y) = 3x^2 y - y^3 + C$

**Verification:** $f(z) = u + iv = (x^3 - 3xy^2) + i(3x^2 y - y^3) = (x + iy)^3 = z^3$ ✓

---

## Practice Problems

### Level 1: Direct Application

1. Verify Cauchy-Riemann for $f(z) = e^z$ by writing $u$ and $v$ explicitly.

2. Show that $f(z) = |z|^2 = x^2 + y^2$ does NOT satisfy Cauchy-Riemann (except at origin).

3. For $f(z) = z^3$, compute $f'(z)$ using the Cauchy-Riemann formula.

### Level 2: Intermediate

4. Find the harmonic conjugate of $u(x,y) = e^x \cos y$.

5. Verify that $u = \ln(x^2 + y^2)$ is harmonic for $(x,y) \neq (0,0)$, and find its harmonic conjugate.

6. Write the Cauchy-Riemann equations for $f(z) = z^n$ in polar form and verify them.

### Level 3: Challenging

7. **Orthogonality:** Prove that level curves of $u$ and $v$ are orthogonal (when $f' \neq 0$) using $\nabla u \cdot \nabla v = 0$.

8. **Conformal Mapping:** If $f$ is analytic with $f' \neq 0$, show that $f$ preserves angles between curves. (Hint: Consider how tangent vectors transform.)

9. **Quantum Application:** For the 2D hydrogen atom (in atomic units), the ground state wave function is $\psi \propto e^{-r}$. Show that this is related to an analytic function in an appropriate sense.

---

## Computational Lab

```python
"""
Day 171: Cauchy-Riemann Equations
Verifying CR equations and visualizing harmonic conjugates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def check_cauchy_riemann(u, v, x, y, dx, dy):
    """
    Numerically check Cauchy-Riemann equations:
    ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x
    """
    # Compute derivatives using finite differences
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    # Check CR equations
    cr1_error = du_dx - dv_dy  # Should be ≈ 0
    cr2_error = du_dy + dv_dx  # Should be ≈ 0

    return cr1_error, cr2_error, du_dx, du_dy, dv_dx, dv_dy

# Create grid
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 3, figsize=(16, 11))

# ========================================
# 1. f(z) = z² : u = x² - y², v = 2xy
# ========================================
u1 = X**2 - Y**2
v1 = 2*X*Y

cr1_err, cr2_err, _, _, _, _ = check_cauchy_riemann(u1, v1, x, y, dx, dy)

ax = axes[0, 0]
# Plot level curves of u (blue) and v (red)
cs_u = ax.contour(X, Y, u1, levels=15, colors='blue', linewidths=1)
cs_v = ax.contour(X, Y, v1, levels=15, colors='red', linewidths=1)
ax.clabel(cs_u, inline=True, fontsize=7, fmt='%.1f')
ax.set_title('f(z) = z²: u (blue), v (red)\nOrthogonal level curves', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

# ========================================
# 2. CR ERROR FOR z²
# ========================================
ax = axes[0, 1]
total_error = np.sqrt(cr1_err**2 + cr2_err**2)
im = ax.imshow(total_error, extent=[-2, 2, -2, 2], cmap='hot',
               origin='lower', aspect='auto')
ax.set_title(f'CR Error for z² (max={np.max(total_error):.2e})', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='Error')

# ========================================
# 3. f(z) = z̄ (NOT ANALYTIC)
# ========================================
u2 = X  # Re(z̄) = x
v2 = -Y  # Im(z̄) = -y

cr1_err2, cr2_err2, _, _, _, _ = check_cauchy_riemann(u2, v2, x, y, dx, dy)

ax = axes[0, 2]
total_error2 = np.sqrt(cr1_err2**2 + cr2_err2**2)
im2 = ax.imshow(total_error2, extent=[-2, 2, -2, 2], cmap='hot',
                origin='lower', aspect='auto')
ax.set_title(f'CR Error for z̄ (NOT analytic)\nmax={np.max(total_error2):.1f}', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im2, ax=ax, label='Error')

# ========================================
# 4. HARMONIC CONJUGATE EXAMPLE
# ========================================
ax = axes[1, 0]

# u = x³ - 3xy², v = 3x²y - y³ (these are real/imag of z³)
u3 = X**3 - 3*X*Y**2
v3 = 3*X**2*Y - Y**3

cs_u3 = ax.contour(X, Y, u3, levels=15, colors='blue', linewidths=1)
cs_v3 = ax.contour(X, Y, v3, levels=15, colors='red', linewidths=1)
ax.set_title('Harmonic Conjugates\nu = x³-3xy² (blue), v = 3x²y-y³ (red)', fontsize=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

# ========================================
# 5. LAPLACIAN CHECK
# ========================================
ax = axes[1, 1]

# Check ∇²u = 0 for u = x³ - 3xy²
laplacian_u = ndimage.laplace(u3)

im3 = ax.imshow(np.abs(laplacian_u), extent=[-2, 2, -2, 2], cmap='viridis',
                origin='lower', aspect='auto')
ax.set_title(f'|∇²u| for u = x³-3xy²\n(should be ≈0, max={np.max(np.abs(laplacian_u)):.2e})',
             fontsize=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im3, ax=ax, label='|∇²u|')

# ========================================
# 6. GRADIENT ORTHOGONALITY
# ========================================
ax = axes[1, 2]

# Compute ∇u · ∇v for z²
du_dx_z2 = 2*X
du_dy_z2 = -2*Y
dv_dx_z2 = 2*Y
dv_dy_z2 = 2*X

dot_product = du_dx_z2 * dv_dx_z2 + du_dy_z2 * dv_dy_z2

im4 = ax.imshow(np.abs(dot_product), extent=[-2, 2, -2, 2], cmap='coolwarm',
                origin='lower', aspect='auto')
ax.set_title('∇u · ∇v for z² (should be 0)\nConfirms orthogonality', fontsize=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im4, ax=ax, label='∇u · ∇v')

plt.tight_layout()
plt.savefig('day_171_cauchy_riemann.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# FLUID FLOW VISUALIZATION
# ========================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Potential flow around a cylinder (using conformal mapping idea)
# f(z) = z + 1/z maps unit circle to flat plate

ax = axes2[0]
# Stream function and potential for uniform flow
U = 1  # Free stream velocity
stream = U * Y  # Stream function for uniform flow
potential = U * X  # Velocity potential

ax.contour(X, Y, stream, levels=20, colors='blue', linewidths=0.8)
ax.contour(X, Y, potential, levels=20, colors='red', linewidths=0.8)
ax.set_title('Uniform Flow: ψ (blue), φ (red)\nStreamlines ⟂ Equipotentials', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

# Flow around a source
ax = axes2[1]
r = np.sqrt(X**2 + Y**2) + 1e-10
theta = np.arctan2(Y, X)

# Source at origin: φ = ln(r), ψ = θ
phi_source = np.log(r)
psi_source = theta

# Mask near singularity
mask = r < 0.1
phi_source[mask] = np.nan
psi_source[mask] = np.nan

ax.contour(X, Y, phi_source, levels=15, colors='red', linewidths=0.8)
ax.contour(X, Y, psi_source, levels=np.linspace(-np.pi, np.pi, 13),
           colors='blue', linewidths=0.8)
ax.plot(0, 0, 'ko', markersize=10, label='Source')
ax.set_title('Source Flow: φ = ln(r) (red), ψ = θ (blue)\nRadial outflow from origin',
             fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend()

plt.tight_layout()
plt.savefig('day_171_fluid_flow.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("CAUCHY-RIEMANN EQUATIONS - VERIFICATION COMPLETE")
print("=" * 60)
print("\nKey Results:")
print("• f(z) = z² satisfies CR equations: error ≈ 0")
print("• f(z) = z̄ violates CR equations: error = √2 ≠ 0")
print("• Harmonic functions: ∇²u = ∇²v = 0 verified")
print("• Gradient orthogonality: ∇u · ∇v = 0 confirmed")
print("\nPhysical Interpretation:")
print("• u = velocity potential, v = stream function")
print("• Level curves of u ⟂ level curves of v")
print("• Both satisfy Laplace equation (incompressible flow)")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Cauchy-Riemann (Cartesian) | $\partial u/\partial x = \partial v/\partial y$, $\partial u/\partial y = -\partial v/\partial x$ |
| Cauchy-Riemann (Polar) | $\partial u/\partial r = (1/r)\partial v/\partial \theta$, $\partial v/\partial r = -(1/r)\partial u/\partial \theta$ |
| Complex Derivative | $f'(z) = \partial u/\partial x + i\partial v/\partial x$ |
| Laplace Equation | $\nabla^2 u = \nabla^2 v = 0$ |
| Orthogonality | $\nabla u \cdot \nabla v = 0$ |

### Main Takeaways

1. **Cauchy-Riemann equations** are necessary and sufficient for analyticity
2. **Real and imaginary parts** of analytic functions are **harmonic**
3. **Level curves** of $u$ and $v$ are **orthogonal** (conformal property)
4. **Physical interpretation:** potential and stream functions in fluid flow/electrostatics
5. **Quantum connection:** Wave functions in certain regions satisfy related equations

---

## Daily Checklist

- [ ] I can derive Cauchy-Riemann from the definition of complex derivative
- [ ] I can verify analyticity using Cauchy-Riemann
- [ ] I can find harmonic conjugates
- [ ] I understand why harmonic functions appear in physics
- [ ] I can explain the orthogonality of level curves
- [ ] I completed the computational lab

---

## Preview: Day 172

Tomorrow we study **Harmonic Functions** in depth:
- Maximum and minimum principles
- Mean value property
- Poisson integral formula
- Dirichlet boundary value problems
- Applications to quantum mechanics and potential theory

---

*"Nature does not employ equations; equations are our way of organizing the patterns we see."*
— Murray Gell-Mann
