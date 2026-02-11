# Day 193: Asymptotic Methods and Saddle Points

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Saddle Point Method |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Steepest Descent & WKB |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 193, you will be able to:

1. Apply the saddle point approximation to integrals
2. Find steepest descent paths through saddle points
3. Evaluate integrals with large parameters asymptotically
4. Connect to WKB approximation in quantum mechanics
5. Understand Stokes phenomenon and asymptotic series
6. Apply to partition functions and path integrals

---

## Core Content

### 1. The Problem: Large Parameter Integrals

Many physics problems involve integrals of the form:
$$I(\lambda) = \int_C f(z) e^{\lambda g(z)} dz$$

where $\lambda \gg 1$ is a large parameter.

**Examples:**
- Stirling's formula: $n! = \int_0^\infty t^n e^{-t} dt$ with $n \gg 1$
- Partition function: $Z = \int e^{-\beta H}$ with $\beta \gg 1$
- Path integrals: $K = \int \mathcal{D}\phi \, e^{iS[\phi]/\hbar}$ with $\hbar \ll 1$

### 2. Saddle Point (Stationary Phase) Method

**Key idea:** For large $\lambda$, the integral is dominated by regions where $g'(z_0) = 0$ (saddle points).

**Laplace's method (real version):**
$$\int_a^b f(x) e^{\lambda g(x)} dx \approx f(x_0) e^{\lambda g(x_0)} \sqrt{\frac{2\pi}{\lambda|g''(x_0)|}}$$

where $g'(x_0) = 0$ and $g''(x_0) < 0$.

### 3. Method of Steepest Descent

**For complex integrals:** Deform contour to pass through saddle point along path of **steepest descent**.

At saddle point $z_0$ where $g'(z_0) = 0$:
$$g(z) \approx g(z_0) + \frac{1}{2}g''(z_0)(z-z_0)^2$$

**Steepest descent direction:** $\text{Im}[g(z)]$ = constant, $\text{Re}[g(z)]$ decreasing.

If $g''(z_0) = |g''|e^{i\alpha}$, the steepest descent direction is:
$$\theta = \frac{\pi - \alpha}{2}$$

### 4. The Saddle Point Formula

$$\boxed{\int_C f(z) e^{\lambda g(z)} dz \approx f(z_0) e^{\lambda g(z_0)} \sqrt{\frac{2\pi}{\lambda |g''(z_0)|}} e^{i\phi}}$$

where:
- $z_0$ is the saddle point: $g'(z_0) = 0$
- $\phi$ depends on the direction of passage through saddle
- Valid for $\lambda \to \infty$

### 5. Example: Stirling's Formula

**Problem:** Derive $n! \approx \sqrt{2\pi n}\left(\frac{n}{e}\right)^n$

**Solution:**
$$n! = \int_0^\infty t^n e^{-t} dt = \int_0^\infty e^{n\ln t - t} dt$$

Let $g(t) = \ln t - t/n$. Then $g'(t) = 1/t - 1/n = 0$ gives $t_0 = n$.

$$g(n) = \ln n - 1, \quad g''(n) = -1/n^2$$

$$n! \approx e^{n(\ln n - 1)} \sqrt{\frac{2\pi}{n \cdot 1/n^2}} = n^n e^{-n} \sqrt{2\pi n}$$

This is **Stirling's formula**!

### 6. WKB from Saddle Points

The WKB approximation for $\psi'' + k(x)^2 \psi = 0$:
$$\psi(x) \approx \frac{A}{\sqrt{k(x)}} e^{\pm i\int^x k(x')dx'}$$

**Connection:** The path integral gives:
$$K(x_f, x_i) \approx \sqrt{\frac{m}{2\pi i\hbar t}} e^{iS_{\text{cl}}/\hbar}$$

where $S_{\text{cl}}$ is the classical action — the saddle point of the path integral!

### 7. Stokes Phenomenon

**Problem:** Asymptotic expansions can have different forms in different sectors of the complex plane.

**Example:** The Airy function $\text{Ai}(z)$ has asymptotic:
- For $\arg z = 0$: exponentially decaying
- For $\arg z = 2\pi/3$: oscillatory

The **Stokes lines** are where subdominant terms become comparable.

---

## Worked Examples

### Example 1: Gaussian Integral via Saddle Point

**Problem:** Evaluate $I(\lambda) = \int_{-\infty}^{\infty} e^{-\lambda x^4} dx$ for large $\lambda$.

**Solution:**
This has saddle at $x = 0$, but $g''(0) = 0$ (not a simple saddle).

Better approach: Substitute $u = \lambda^{1/4} x$:
$$I = \lambda^{-1/4} \int_{-\infty}^{\infty} e^{-u^4} du = \lambda^{-1/4} \cdot \frac{\Gamma(1/4)}{2}$$

So $I(\lambda) = \frac{\Gamma(1/4)}{2} \lambda^{-1/4}$ exactly.

### Example 2: Bessel Function Asymptotics

**Problem:** Find $J_0(x)$ for large $x$.

**Solution:**
The integral representation:
$$J_0(x) = \frac{1}{2\pi} \int_0^{2\pi} e^{ix\sin\theta} d\theta$$

Saddle points: $\cos\theta_0 = 0$, so $\theta_0 = \pi/2, 3\pi/2$.

At $\theta = \pi/2$: $\sin\theta = 1$, contribution $\sim e^{ix}$
At $\theta = 3\pi/2$: $\sin\theta = -1$, contribution $\sim e^{-ix}$

Result:
$$J_0(x) \approx \sqrt{\frac{2}{\pi x}} \cos\left(x - \frac{\pi}{4}\right)$$

---

## Practice Problems

**P1.** Use Stirling to show $\binom{2n}{n} \approx \frac{4^n}{\sqrt{\pi n}}$.

**P2.** Find the asymptotic behavior of $\int_0^\infty e^{-x^3 - \lambda x} dx$ for large $\lambda$.

**P3.** Apply saddle point to the Gamma function integral.

**P4.** Show that the semiclassical propagator comes from saddle point approximation.

---

## Computational Lab

```python
import numpy as np
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt

def stirling(n):
    """Stirling's approximation for n!"""
    return np.sqrt(2 * np.pi * n) * (n / np.e)**n

def saddle_point_integral(lambda_val, f, g, g_prime, g_double_prime, z0_guess):
    """
    Estimate integral ∫f(z)e^{λg(z)}dz via saddle point.
    """
    from scipy.optimize import fsolve

    # Find saddle point
    z0 = fsolve(lambda z: g_prime(z[0]), [z0_guess])[0]

    # Saddle point approximation
    prefactor = f(z0)
    exponential = np.exp(lambda_val * g(z0))
    gaussian = np.sqrt(2 * np.pi / (lambda_val * np.abs(g_double_prime(z0))))

    return prefactor * exponential * gaussian, z0

# Test Stirling's formula
n_values = np.arange(1, 50)
exact = factorial(n_values, exact=False)
stirling_approx = stirling(n_values)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].semilogy(n_values, exact, 'b-', linewidth=2, label='Exact n!')
axes[0, 0].semilogy(n_values, stirling_approx, 'r--', linewidth=2, label='Stirling')
axes[0, 0].set_xlabel('n')
axes[0, 0].set_ylabel('n!')
axes[0, 0].set_title("Stirling's Approximation")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Relative error
relative_error = np.abs(stirling_approx - exact) / exact
axes[0, 1].plot(n_values, relative_error * 100, 'g-', linewidth=2)
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('Relative Error (%)')
axes[0, 1].set_title('Error in Stirling Approximation')
axes[0, 1].grid(True, alpha=0.3)

# Saddle point visualization
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# g(z) = z^2 for Gaussian
g = Z**2
axes[1, 0].contour(X, Y, g.real, levels=20, colors='blue', alpha=0.5)
axes[1, 0].contour(X, Y, g.imag, levels=20, colors='red', alpha=0.5)
axes[1, 0].plot([0], [0], 'k*', markersize=15, label='Saddle point')
axes[1, 0].arrow(-2, 0, 4, 0, head_width=0.1, color='green', linewidth=2)
axes[1, 0].set_xlabel('Re(z)')
axes[1, 0].set_ylabel('Im(z)')
axes[1, 0].set_title('Saddle Point for $e^{-z^2}$ (Re=blue, Im=red)')
axes[1, 0].legend()
axes[1, 0].axis('equal')

# Steepest descent for e^{iz^2}
g2 = 1j * Z**2
descent_path = np.linspace(-2, 2, 100) * np.exp(1j * np.pi/4)
axes[1, 1].contourf(X, Y, (1j * Z**2).real, levels=50, cmap='RdBu', alpha=0.7)
axes[1, 1].plot(descent_path.real, descent_path.imag, 'g-', linewidth=3,
                label='Steepest descent')
axes[1, 1].plot([0], [0], 'k*', markersize=15)
axes[1, 1].set_xlabel('Re(z)')
axes[1, 1].set_ylabel('Im(z)')
axes[1, 1].set_title('Steepest Descent for $e^{iz^2}$')
axes[1, 1].legend()
axes[1, 1].axis('equal')

plt.tight_layout()
plt.savefig('saddle_point.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\int f e^{\lambda g} dz \approx f(z_0) e^{\lambda g(z_0)} \sqrt{\frac{2\pi}{\lambda\|g''\|}}$ | Saddle point |
| $n! \approx \sqrt{2\pi n}(n/e)^n$ | Stirling's formula |
| $K \approx \sqrt{\frac{m}{2\pi i\hbar t}} e^{iS_{\text{cl}}/\hbar}$ | Semiclassical propagator |

### Main Takeaways

1. **Saddle point method** evaluates integrals with large parameters
2. **Steepest descent** deforms contours through saddle points
3. **Stirling's formula** is a saddle point result
4. **WKB** is the saddle point of the path integral
5. **Stokes phenomenon** causes discontinuities in asymptotic series

---

## Preview: Day 194

Tomorrow: **Special Functions from Complex Analysis**
- Gamma function and its properties
- Zeta function and analytic continuation
- Hypergeometric functions

---

*"Asymptotic methods extract the essence of physics from complex mathematics."*
