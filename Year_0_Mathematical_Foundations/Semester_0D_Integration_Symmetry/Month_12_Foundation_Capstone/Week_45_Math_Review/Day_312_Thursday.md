# Day 312: Complex Analysis Synthesis

## Overview

**Month 12, Week 45, Day 4 — Thursday**

Today we synthesize complex analysis: analytic functions, contour integration, residues, and their applications to physics. Complex methods are essential for evaluating integrals, solving differential equations, and understanding quantum propagators.

## Learning Objectives

1. Review analytic function properties
2. Master contour integration techniques
3. Apply residue theorem to physics problems
4. Connect to quantum mechanics

---

## 1. Analytic Functions

### Cauchy-Riemann Equations

For $f(z) = u(x,y) + iv(x,y)$ to be analytic:

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

### Key Properties

- Analytic functions are infinitely differentiable
- Real and imaginary parts are harmonic: $\nabla^2 u = \nabla^2 v = 0$
- Zeros are isolated (unless $f \equiv 0$)

---

## 2. Contour Integration

### Cauchy's Integral Theorem

For $f$ analytic inside and on closed contour $C$:
$$\oint_C f(z)dz = 0$$

### Cauchy's Integral Formula

$$f(z_0) = \frac{1}{2\pi i}\oint_C \frac{f(z)}{z-z_0}dz$$

For derivatives:
$$f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint_C \frac{f(z)}{(z-z_0)^{n+1}}dz$$

---

## 3. Residue Theorem

### The Formula

$$\oint_C f(z)dz = 2\pi i \sum_k \text{Res}(f, z_k)$$

### Computing Residues

**Simple pole at $z_0$:**
$$\text{Res}(f, z_0) = \lim_{z \to z_0}(z-z_0)f(z)$$

**Pole of order $n$:**
$$\text{Res}(f, z_0) = \frac{1}{(n-1)!}\lim_{z \to z_0}\frac{d^{n-1}}{dz^{n-1}}[(z-z_0)^n f(z)]$$

---

## 4. Applications to Real Integrals

### Type 1: $\int_0^{2\pi}$ with trigonometric functions

Use $z = e^{i\theta}$, $d\theta = dz/(iz)$

### Type 2: $\int_{-\infty}^{\infty}$ of rational functions

Close contour in upper or lower half-plane

### Type 3: Fourier-type integrals

$$\int_{-\infty}^{\infty} f(x)e^{ikx}dx$$

Use Jordan's lemma

---

## 5. Physics Applications

### Green's Functions

$$G(E) = \frac{1}{E - H + i\epsilon}$$

Poles give energy eigenvalues.

### Propagators in QM

$$K(x_f, t_f; x_i, t_i) = \sum_n \psi_n^*(x_i)\psi_n(x_f)e^{-iE_n(t_f-t_i)/\hbar}$$

---

## 6. Computational Lab

```python
"""
Day 312: Complex Analysis Synthesis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def residue_theorem_demo():
    """Evaluate integrals using residue theorem."""
    print("=" * 50)
    print("RESIDUE THEOREM APPLICATIONS")
    print("=" * 50)

    # Example: ∫_{-∞}^{∞} 1/(x² + 1) dx = π
    # Poles at z = ±i, residue at z=i is 1/(2i)

    result, _ = integrate.quad(lambda x: 1/(x**2 + 1), -100, 100)
    theoretical = np.pi

    print(f"\n∫ 1/(x²+1) dx from -∞ to ∞:")
    print(f"  Numerical: {result:.6f}")
    print(f"  Theoretical (π): {theoretical:.6f}")


def contour_visualization():
    """Visualize contour integration."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw axes
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Draw contour (semicircle in upper half-plane)
    theta = np.linspace(0, np.pi, 100)
    R = 2
    x_arc = R * np.cos(theta)
    y_arc = R * np.sin(theta)
    ax.plot(x_arc, y_arc, 'b-', linewidth=2, label='Arc')
    ax.plot([-R, R], [0, 0], 'b-', linewidth=2, label='Real axis')

    # Mark poles
    ax.plot(0, 1, 'ro', markersize=10, label='Pole at z=i')
    ax.plot(0, -1, 'rx', markersize=10, label='Pole at z=-i (outside)')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Contour for $\\int_{-\\infty}^{\\infty} \\frac{1}{x^2+1}dx$')
    plt.savefig('contour_integration.png', dpi=150)
    plt.close()
    print("Saved: contour_integration.png")


# Main
if __name__ == "__main__":
    residue_theorem_demo()
    contour_visualization()
```

---

## Summary

### Complex Analysis Toolkit

$$\boxed{\oint_C f(z)dz = 2\pi i \sum_k \text{Res}(f, z_k)}$$

### QM Connection

- **Propagators:** Analytic continuation in time
- **Green's functions:** Poles = eigenvalues
- **Path integrals:** Complex action

---

## Preview: Day 313

Tomorrow: **Functional Analysis Synthesis** — Hilbert spaces and operator theory.
