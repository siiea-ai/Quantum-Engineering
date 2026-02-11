# Day 309: Calculus Synthesis — From Limits to Manifolds

## Overview

**Month 12, Week 45, Day 1 — Monday**

Today we synthesize all calculus concepts from Year 0: limits, derivatives, integrals, series, vector calculus, and differential forms. We trace the journey from single-variable beginnings to the powerful tools of multivariable analysis that underpin physics.

## Learning Objectives

1. Review single-variable calculus fundamentals
2. Synthesize multivariable calculus techniques
3. Master vector calculus theorems
4. Connect to physics applications

---

## 1. Single-Variable Calculus Review

### The Derivative

$$f'(x) = \lim_{h \to 0}\frac{f(x+h) - f(x)}{h}$$

**Physical meaning:** Rate of change, velocity, slope of tangent.

### Key Differentiation Rules

| Rule | Formula |
|------|---------|
| Power | $(x^n)' = nx^{n-1}$ |
| Product | $(fg)' = f'g + fg'$ |
| Chain | $(f \circ g)' = f'(g) \cdot g'$ |
| Exponential | $(e^x)' = e^x$ |
| Logarithm | $(\ln x)' = 1/x$ |

### The Integral

$$\int_a^b f(x)dx = F(b) - F(a)$$

where $F'(x) = f(x)$.

**Physical meaning:** Accumulated quantity, area, total change.

### Integration Techniques

1. **Substitution:** $\int f(g(x))g'(x)dx = \int f(u)du$
2. **By parts:** $\int u\,dv = uv - \int v\,du$
3. **Partial fractions:** Decompose rational functions
4. **Trigonometric:** Use identities

---

## 2. Series and Approximations

### Taylor Series

$$f(x) = \sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n$$

**Essential Expansions:**

$$e^x = \sum_{n=0}^{\infty}\frac{x^n}{n!}$$

$$\sin x = \sum_{n=0}^{\infty}\frac{(-1)^n x^{2n+1}}{(2n+1)!}$$

$$\cos x = \sum_{n=0}^{\infty}\frac{(-1)^n x^{2n}}{(2n)!}$$

$$\frac{1}{1-x} = \sum_{n=0}^{\infty}x^n \quad (|x| < 1)$$

### Convergence Tests

1. **Ratio test:** $L = \lim |a_{n+1}/a_n|$; converges if $L < 1$
2. **Root test:** $L = \lim |a_n|^{1/n}$
3. **Integral test:** Compare to $\int_1^\infty f(x)dx$

---

## 3. Multivariable Calculus

### Partial Derivatives

$$\frac{\partial f}{\partial x} = \lim_{h \to 0}\frac{f(x+h, y) - f(x, y)}{h}$$

### The Gradient

$$\nabla f = \frac{\partial f}{\partial x}\hat{i} + \frac{\partial f}{\partial y}\hat{j} + \frac{\partial f}{\partial z}\hat{k}$$

**Properties:**
- Points in direction of steepest increase
- Magnitude = rate of maximum increase
- $\nabla f \perp$ level surfaces

### The Chain Rule (Multivariable)

$$\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt} + \frac{\partial f}{\partial z}\frac{dz}{dt}$$

### Multiple Integrals

$$\iint_R f(x,y)dA = \int_a^b\int_{g_1(x)}^{g_2(x)} f(x,y)dy\,dx$$

**Jacobian for coordinate change:**
$$dA = |J|du\,dv, \quad J = \begin{vmatrix}\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v}\end{vmatrix}$$

---

## 4. Vector Calculus

### The Big Three Operators

$$\text{Gradient:} \quad \nabla f = \sum_i \frac{\partial f}{\partial x_i}\hat{e}_i$$

$$\text{Divergence:} \quad \nabla \cdot \mathbf{F} = \sum_i \frac{\partial F_i}{\partial x_i}$$

$$\text{Curl:} \quad \nabla \times \mathbf{F} = \begin{vmatrix}\hat{i} & \hat{j} & \hat{k} \\ \partial_x & \partial_y & \partial_z \\ F_x & F_y & F_z\end{vmatrix}$$

### The Laplacian

$$\nabla^2 f = \nabla \cdot (\nabla f) = \sum_i \frac{\partial^2 f}{\partial x_i^2}$$

### Fundamental Identities

$$\nabla \times (\nabla f) = 0 \quad \text{(curl of gradient = 0)}$$
$$\nabla \cdot (\nabla \times \mathbf{F}) = 0 \quad \text{(div of curl = 0)}$$

---

## 5. The Great Integration Theorems

### Fundamental Theorem of Calculus

$$\int_a^b f'(x)dx = f(b) - f(a)$$

### Gradient Theorem (Line Integrals)

$$\int_C \nabla f \cdot d\mathbf{r} = f(\mathbf{r}_B) - f(\mathbf{r}_A)$$

### Green's Theorem (2D)

$$\oint_C (P\,dx + Q\,dy) = \iint_R \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right)dA$$

### Stokes' Theorem (3D)

$$\oint_{\partial S} \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{A}$$

### Divergence Theorem (Gauss)

$$\oiint_{\partial V} \mathbf{F} \cdot d\mathbf{A} = \iiint_V (\nabla \cdot \mathbf{F})dV$$

### Unified View

All theorems are instances of the generalized Stokes theorem:
$$\int_{\partial M} \omega = \int_M d\omega$$

---

## 6. Coordinate Systems

### Cylindrical $(r, \phi, z)$

$$\nabla f = \frac{\partial f}{\partial r}\hat{r} + \frac{1}{r}\frac{\partial f}{\partial \phi}\hat{\phi} + \frac{\partial f}{\partial z}\hat{z}$$

$$\nabla^2 f = \frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial f}{\partial r}\right) + \frac{1}{r^2}\frac{\partial^2 f}{\partial \phi^2} + \frac{\partial^2 f}{\partial z^2}$$

### Spherical $(r, \theta, \phi)$

$$\nabla f = \frac{\partial f}{\partial r}\hat{r} + \frac{1}{r}\frac{\partial f}{\partial \theta}\hat{\theta} + \frac{1}{r\sin\theta}\frac{\partial f}{\partial \phi}\hat{\phi}$$

$$\nabla^2 f = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial f}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta\frac{\partial f}{\partial \theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}$$

---

## 7. Physics Applications

### Classical Mechanics

- **Velocity:** $\mathbf{v} = \frac{d\mathbf{r}}{dt}$
- **Acceleration:** $\mathbf{a} = \frac{d\mathbf{v}}{dt}$
- **Work:** $W = \int_C \mathbf{F} \cdot d\mathbf{r}$
- **Conservative forces:** $\mathbf{F} = -\nabla V$

### Electromagnetism

- **Electric field:** $\mathbf{E} = -\nabla \phi$
- **Gauss's law:** $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$
- **Faraday's law:** $\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$

### Quantum Mechanics Preview

- **Probability current:** $\mathbf{j} = \frac{\hbar}{2mi}(\psi^*\nabla\psi - \psi\nabla\psi^*)$
- **Schrödinger equation:** Contains $\nabla^2$
- **Continuity:** $\frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{j} = 0$

---

## 8. Computational Lab

```python
"""
Day 309: Calculus Synthesis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

def taylor_series_demo():
    """Demonstrate Taylor series approximations."""
    x = np.linspace(-3, 3, 200)

    # e^x
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(x, np.exp(x), 'k-', linewidth=2, label='$e^x$')
    for n in [1, 2, 3, 5]:
        taylor = sum(x**k / np.math.factorial(k) for k in range(n+1))
        ax.plot(x, taylor, '--', label=f'Order {n}')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 10)
    ax.legend()
    ax.set_title('Taylor Series for $e^x$')
    ax.grid(True)

    # sin(x)
    ax = axes[0, 1]
    ax.plot(x, np.sin(x), 'k-', linewidth=2, label='$\sin(x)$')
    for n in [1, 3, 5, 7]:
        taylor = sum((-1)**k * x**(2*k+1) / np.math.factorial(2*k+1)
                    for k in range((n+1)//2))
        ax.plot(x, taylor, '--', label=f'Order {n}')
    ax.set_ylim(-2, 2)
    ax.legend()
    ax.set_title('Taylor Series for $\sin(x)$')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('taylor_series.png', dpi=150)
    plt.close()
    print("Saved: taylor_series.png")


def gradient_visualization():
    """Visualize gradient field."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    # Function f(x,y) = x^2 + y^2
    Z = X**2 + Y**2

    # Gradient
    Fx = 2 * X
    Fy = 2 * Y

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.quiver(X, Y, -Fx, -Fy, alpha=0.7, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gradient Field of $f(x,y) = x^2 + y^2$\n(arrows show -∇f)')
    ax.set_aspect('equal')
    plt.savefig('gradient_field.png', dpi=150)
    plt.close()
    print("Saved: gradient_field.png")


def divergence_curl_demo():
    """Visualize divergence and curl."""
    x = np.linspace(-2, 2, 15)
    y = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Source field (positive divergence)
    ax = axes[0]
    Fx, Fy = X, Y
    ax.quiver(X, Y, Fx, Fy)
    ax.set_title('Source Field\n$\\nabla \\cdot \\mathbf{F} > 0$')
    ax.set_aspect('equal')

    # Sink field (negative divergence)
    ax = axes[1]
    Fx, Fy = -X, -Y
    ax.quiver(X, Y, Fx, Fy)
    ax.set_title('Sink Field\n$\\nabla \\cdot \\mathbf{F} < 0$')
    ax.set_aspect('equal')

    # Rotational field (non-zero curl)
    ax = axes[2]
    Fx, Fy = -Y, X
    ax.quiver(X, Y, Fx, Fy)
    ax.set_title('Rotational Field\n$\\nabla \\times \\mathbf{F} \\neq 0$')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('div_curl.png', dpi=150)
    plt.close()
    print("Saved: div_curl.png")


def integration_theorems():
    """Verify integration theorems numerically."""
    print("=" * 50)
    print("INTEGRATION THEOREMS VERIFICATION")
    print("=" * 50)

    # Divergence theorem: sphere
    print("\n1. Divergence Theorem (Gauss)")
    print("-" * 40)

    # F = (x, y, z), div F = 3
    # Surface integral over sphere radius R
    R = 2

    # Volume integral: ∫∫∫ div F dV = 3 * (4/3)πR³
    vol_integral = 3 * (4/3) * np.pi * R**3

    # Surface integral: ∫∫ F·n dA = R * 4πR²
    surf_integral = R * 4 * np.pi * R**2

    print(f"   F = (x, y, z), sphere of radius R = {R}")
    print(f"   Volume integral (∫∫∫ div F dV): {vol_integral:.4f}")
    print(f"   Surface integral (∫∫ F·dA): {surf_integral:.4f}")
    print(f"   Match: {np.isclose(vol_integral, surf_integral)}")

    # Stokes' theorem
    print("\n2. Stokes' Theorem")
    print("-" * 40)

    # F = (-y, x, 0), curl F = (0, 0, 2)
    # Disk of radius R in xy-plane

    # Line integral around circle
    def line_integrand(theta):
        x, y = R * np.cos(theta), R * np.sin(theta)
        Fx, Fy = -y, x
        dx, dy = -R * np.sin(theta), R * np.cos(theta)
        return Fx * dx + Fy * dy

    line_integral, _ = integrate.quad(line_integrand, 0, 2*np.pi)

    # Surface integral of curl
    curl_z = 2
    surf_integral = curl_z * np.pi * R**2

    print(f"   F = (-y, x, 0), disk of radius R = {R}")
    print(f"   Line integral (∮ F·dr): {line_integral:.4f}")
    print(f"   Surface integral (∫∫ curl F·dA): {surf_integral:.4f}")
    print(f"   Match: {np.isclose(line_integral, surf_integral)}")


def coordinate_systems():
    """Demonstrate coordinate system transformations."""
    print("\n" + "=" * 50)
    print("COORDINATE SYSTEMS")
    print("=" * 50)

    # Cartesian to spherical
    x, y, z = 1, 1, 1
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    print(f"\nCartesian (1, 1, 1) → Spherical:")
    print(f"   r = {r:.4f}")
    print(f"   θ = {np.degrees(theta):.4f}°")
    print(f"   φ = {np.degrees(phi):.4f}°")

    # Volume element in spherical
    print(f"\nVolume element: dV = r²sin(θ) dr dθ dφ")

    # Integral of 1 over sphere
    R = 1
    vol, _ = integrate.tplquad(
        lambda phi, theta, r: r**2 * np.sin(theta),
        0, R,  # r
        lambda r: 0, lambda r: np.pi,  # theta
        lambda r, theta: 0, lambda r, theta: 2*np.pi  # phi
    )
    print(f"\n∫∫∫ dV over sphere of radius {R}:")
    print(f"   Numerical: {vol:.6f}")
    print(f"   Analytical: {4/3 * np.pi * R**3:.6f}")


# Main execution
if __name__ == "__main__":
    taylor_series_demo()
    gradient_visualization()
    divergence_curl_demo()
    integration_theorems()
    coordinate_systems()
```

---

## 9. Practice Problems

### Problem 1: Integration

Evaluate $\int_0^1 \int_0^{1-x} e^{x+y} dy\,dx$.

### Problem 2: Vector Calculus

Verify Stokes' theorem for $\mathbf{F} = (y^2, x, z)$ and the hemisphere $z = \sqrt{1-x^2-y^2}$.

### Problem 3: Coordinate Change

Express the Laplacian of $f(r, \theta, \phi) = e^{-r}Y_1^0(\theta, \phi)$ in spherical coordinates.

### Problem 4: Series

Find the radius of convergence of $\sum_{n=1}^{\infty}\frac{n!}{n^n}x^n$.

---

## Summary

### The Calculus Hierarchy

$$\text{Limits} \to \text{Derivatives} \to \text{Integrals} \to \text{Series}$$
$$\downarrow$$
$$\text{Partial Derivatives} \to \text{Multiple Integrals} \to \text{Vector Calculus}$$
$$\downarrow$$
$$\text{Differential Forms} \to \text{Manifolds}$$

### Key Unifying Principle

**The generalized Stokes theorem** connects integration over a region to integration over its boundary:

$$\boxed{\int_{\partial M} \omega = \int_M d\omega}$$

---

## Preview: Day 310

Tomorrow: **Linear Algebra Synthesis** — from vectors to operators, matrices to spectral theory.
