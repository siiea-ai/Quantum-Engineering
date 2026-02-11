# Day 311: Differential Equations Synthesis

## Overview

**Month 12, Week 45, Day 3 — Wednesday**

Today we synthesize differential equations from basic ODEs to the partial differential equations that govern physics. The Schrödinger equation, wave equation, and heat equation all emerge from this framework.

## Learning Objectives

1. Review ODE solution techniques
2. Master eigenvalue problems and Sturm-Liouville theory
3. Connect to PDEs and separation of variables
4. Understand special functions in physics

---

## 1. Ordinary Differential Equations

### First-Order Linear

$$\frac{dy}{dx} + P(x)y = Q(x)$$

**Solution:** $y = e^{-\int P dx}\left[\int Q e^{\int P dx}dx + C\right]$

### Second-Order Linear with Constant Coefficients

$$ay'' + by' + cy = 0$$

**Characteristic equation:** $ar^2 + br + c = 0$

| Roots | Solution |
|-------|----------|
| $r_1 \neq r_2$ real | $y = c_1 e^{r_1 x} + c_2 e^{r_2 x}$ |
| $r_1 = r_2 = r$ | $y = (c_1 + c_2 x)e^{rx}$ |
| $r = \alpha \pm i\beta$ | $y = e^{\alpha x}(c_1\cos\beta x + c_2\sin\beta x)$ |

---

## 2. Eigenvalue Problems

### Standard Form

$$\mathcal{L}y = \lambda y$$

with boundary conditions.

### Sturm-Liouville Problem

$$\frac{d}{dx}\left[p(x)\frac{dy}{dx}\right] + [q(x) + \lambda w(x)]y = 0$$

**Properties:**
1. Eigenvalues are real
2. Eigenfunctions orthogonal with weight $w(x)$
3. Eigenfunctions form complete basis

### Key Examples

| Problem | Eigenfunctions | Eigenvalues |
|---------|----------------|-------------|
| $y'' + \lambda y = 0$, $y(0)=y(L)=0$ | $\sin(n\pi x/L)$ | $(n\pi/L)^2$ |
| Legendre | $P_\ell(\cos\theta)$ | $\ell(\ell+1)$ |
| Hermite | $H_n(x)e^{-x^2/2}$ | $2n+1$ |
| Laguerre | $L_n(x)e^{-x/2}$ | $n$ |

---

## 3. Series Solutions and Special Functions

### Frobenius Method

Near regular singular point $x_0$:
$$y = (x-x_0)^r \sum_{n=0}^{\infty} a_n(x-x_0)^n$$

### Legendre's Equation

$$(1-x^2)y'' - 2xy' + \ell(\ell+1)y = 0$$

Solutions: Legendre polynomials $P_\ell(x)$

$$P_0 = 1, \quad P_1 = x, \quad P_2 = \frac{1}{2}(3x^2-1)$$

### Hermite's Equation

$$y'' - 2xy' + 2ny = 0$$

Solutions: Hermite polynomials $H_n(x)$

$$H_0 = 1, \quad H_1 = 2x, \quad H_2 = 4x^2 - 2$$

### Bessel's Equation

$$x^2y'' + xy' + (x^2-\nu^2)y = 0$$

Solutions: Bessel functions $J_\nu(x)$, $Y_\nu(x)$

---

## 4. Partial Differential Equations

### The Big Three

**Wave Equation:**
$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$

**Heat Equation:**
$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$

**Laplace's Equation:**
$$\nabla^2 u = 0$$

### Separation of Variables

$$u(x,t) = X(x)T(t)$$

Reduces PDE to coupled ODEs.

---

## 5. Green's Functions

### Definition

$$\mathcal{L}G(x,x') = \delta(x-x')$$

### Solution Formula

$$u(x) = \int G(x,x')f(x')dx'$$

---

## 6. Quantum Mechanics Connection

### The Schrödinger Equation

$$i\hbar\frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + V\psi$$

- Time-independent: eigenvalue problem $\hat{H}\psi = E\psi$
- Solutions: wavefunctions with orthogonality
- Boundary conditions: normalizability

---

## 7. Computational Lab

```python
"""
Day 311: Differential Equations Synthesis
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.special import hermite, legendre, jv
import matplotlib.pyplot as plt

def harmonic_oscillator():
    """Solve quantum harmonic oscillator."""
    print("=" * 50)
    print("QUANTUM HARMONIC OSCILLATOR")
    print("=" * 50)

    # Hermite functions (normalized)
    x = np.linspace(-5, 5, 500)

    fig, ax = plt.subplots(figsize=(10, 6))
    for n in range(5):
        Hn = hermite(n)
        psi = Hn(x) * np.exp(-x**2/2) / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
        ax.plot(x, psi + n, label=f'n={n}')
        ax.axhline(y=n, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('ψ_n(x) + n')
    ax.set_title('Quantum Harmonic Oscillator Wavefunctions')
    ax.legend()
    plt.savefig('harmonic_oscillator.png', dpi=150)
    plt.close()
    print("Saved: harmonic_oscillator.png")


def eigenvalue_problem():
    """Solve particle in a box numerically."""
    print("\n" + "=" * 50)
    print("PARTICLE IN A BOX - NUMERICAL")
    print("=" * 50)

    L = 1.0
    N = 100
    dx = L / (N + 1)
    x = np.linspace(dx, L - dx, N)

    # Finite difference Laplacian
    diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    H = (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dx**2
    H = -H  # -d²/dx²

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    print("\nFirst 5 eigenvalues:")
    for n in range(1, 6):
        numerical = eigenvalues[n-1]
        analytical = (n * np.pi / L)**2
        print(f"  n={n}: numerical={numerical:.4f}, analytical={analytical:.4f}")


def special_functions_demo():
    """Demonstrate special functions."""
    x = np.linspace(-1, 1, 200)
    r = np.linspace(0, 10, 200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Legendre polynomials
    ax = axes[0, 0]
    for l in range(5):
        Pl = legendre(l)
        ax.plot(x, Pl(x), label=f'$P_{l}$')
    ax.set_title('Legendre Polynomials')
    ax.legend()
    ax.grid(True)

    # Hermite polynomials
    ax = axes[0, 1]
    x_h = np.linspace(-3, 3, 200)
    for n in range(5):
        Hn = hermite(n)
        ax.plot(x_h, Hn(x_h) * np.exp(-x_h**2/2), label=f'$H_{n}$')
    ax.set_title('Hermite Functions')
    ax.legend()
    ax.grid(True)

    # Bessel functions
    ax = axes[1, 0]
    for n in range(4):
        ax.plot(r, jv(n, r), label=f'$J_{n}$')
    ax.set_title('Bessel Functions')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('special_functions.png', dpi=150)
    plt.close()
    print("Saved: special_functions.png")


# Main
if __name__ == "__main__":
    harmonic_oscillator()
    eigenvalue_problem()
    special_functions_demo()
```

---

## Summary

### ODE → PDE → QM Progression

$$\text{ODEs} \to \text{Eigenvalue Problems} \to \text{Sturm-Liouville} \to \text{QM Eigenstates}$$

### Key Quantum Connection

$$\boxed{\hat{H}\psi_n = E_n\psi_n}$$

The time-independent Schrödinger equation is a Sturm-Liouville eigenvalue problem.

---

## Preview: Day 312

Tomorrow: **Complex Analysis Synthesis** — from analytic functions to contour integrals.
