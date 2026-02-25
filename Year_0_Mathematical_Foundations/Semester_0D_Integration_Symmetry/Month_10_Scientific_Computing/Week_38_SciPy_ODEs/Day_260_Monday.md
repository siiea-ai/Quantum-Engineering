# Day 260: Numerical Integration with SciPy

## Overview

**Day 260** | **Week 38** | **Month 10: Scientific Computing**

Today we master numerical integration using `scipy.integrate`. Integration is fundamental to quantum mechanics—normalization, expectation values, and transition amplitudes all require computing integrals. SciPy's adaptive quadrature methods handle the challenging integrals that appear in physics, automatically adjusting their precision to achieve desired accuracy.

**Prerequisites:** Week 37 (NumPy), calculus (Months 1-2)
**Outcome:** Compute single and multi-dimensional integrals with controlled precision

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Quadrature methods, adaptive integration |
| Afternoon | 3 hours | Practice: QM integrals, expectation values |
| Evening | 2 hours | Lab: Complete normalization and overlap calculations |

---

## Learning Objectives

By the end of Day 260, you will be able to:

1. **Use scipy.integrate.quad** for definite integrals with error estimates
2. **Integrate functions with singularities** using appropriate options
3. **Compute multi-dimensional integrals** with dblquad and tplquad
4. **Understand quadrature rules** (Gaussian, adaptive, Romberg)
5. **Calculate quantum mechanical integrals** (normalization, overlap, expectation)
6. **Handle oscillatory integrands** efficiently
7. **Verify numerical accuracy** against analytical results

---

## Core Content

### 1. Basic Integration with quad

```python
import numpy as np
from scipy import integrate

# Basic definite integral: ∫₀¹ x² dx = 1/3
result, error = integrate.quad(lambda x: x**2, 0, 1)
print(f"∫₀¹ x² dx = {result:.10f} ± {error:.2e}")
print(f"Exact: {1/3:.10f}")

# Integrand as a separate function
def gaussian(x):
    return np.exp(-x**2)

result, error = integrate.quad(gaussian, -np.inf, np.inf)
print(f"∫ e^(-x²) dx = {result:.10f} ± {error:.2e}")
print(f"Exact (√π): {np.sqrt(np.pi):.10f}")
```

### 2. Handling Infinite Limits

```python
# Infinite limits work automatically
def lorentzian(x, gamma=1.0):
    return gamma / (np.pi * (x**2 + gamma**2))

# ∫₋∞^∞ Lorentzian = 1
result, _ = integrate.quad(lorentzian, -np.inf, np.inf)
print(f"Lorentzian integral: {result:.10f}")

# Semi-infinite integrals
def exponential_decay(x, tau=1.0):
    return np.exp(-x/tau) / tau

result, _ = integrate.quad(exponential_decay, 0, np.inf)
print(f"Exponential integral: {result:.10f}")
```

### 3. Passing Parameters

```python
# Method 1: args parameter
def harmonic_eigenstate(x, n, alpha=1.0):
    """ψₙ(x) for harmonic oscillator (unnormalized)."""
    from scipy.special import hermite
    Hn = hermite(n)
    return Hn(alpha*x) * np.exp(-alpha**2 * x**2 / 2)

# Integrate with parameters
result, _ = integrate.quad(harmonic_eigenstate, -np.inf, np.inf, args=(0, 1.0))
print(f"∫|ψ₀|dx = {result:.6f}")

# Method 2: Lambda wrapper (more flexible)
n, alpha = 2, 1.0
integrand = lambda x: harmonic_eigenstate(x, n, alpha)**2
result, _ = integrate.quad(integrand, -np.inf, np.inf)
print(f"∫|ψ₂|²dx = {result:.6f}")
```

### 4. Integration Options and Error Control

```python
# Control absolute and relative tolerances
def oscillatory(x):
    return np.sin(100*x) * np.exp(-x)

# Default tolerance
result1, err1 = integrate.quad(oscillatory, 0, 10)

# Higher precision
result2, err2 = integrate.quad(oscillatory, 0, 10, epsabs=1e-12, epsrel=1e-12)

# Increase subdivisions for difficult integrands
result3, err3 = integrate.quad(oscillatory, 0, 10, limit=100)

print(f"Default:    {result1:.10f} ± {err1:.2e}")
print(f"High prec:  {result2:.10f} ± {err2:.2e}")
print(f"More subs:  {result3:.10f} ± {err3:.2e}")
```

### 5. Handling Singularities

```python
# Integrable singularity at x=0: ∫₀¹ x^(-0.5) dx = 2
def singular(x):
    return 1/np.sqrt(x)

# quad handles this automatically
result, _ = integrate.quad(singular, 0, 1)
print(f"∫₀¹ x^(-1/2) dx = {result:.10f} (exact: 2)")

# Specify singular points with 'points' parameter
def piecewise_singular(x):
    if x == 0.5:
        return 0  # Avoid division by zero
    return 1/np.abs(x - 0.5)**0.5

# Tell quad about the singularity location
result, _ = integrate.quad(piecewise_singular, 0, 1, points=[0.5])
print(f"Integral with internal singularity: {result:.6f}")

# Weight functions for common singularities
# quad(f, a, b, weight='cauchy', wvar=c) computes ∫f(x)/(x-c)dx
```

### 6. Multi-dimensional Integration

```python
# Double integral: ∫∫ f(x,y) dy dx
def circular_region(y, x):
    """Integrand for area of unit circle."""
    return 1.0

# ∫₋₁¹ ∫₋√(1-x²)^√(1-x²) dy dx = π
def y_lower(x):
    return -np.sqrt(1 - x**2)

def y_upper(x):
    return np.sqrt(1 - x**2)

result, _ = integrate.dblquad(circular_region, -1, 1, y_lower, y_upper)
print(f"Area of unit circle: {result:.10f} (exact: π = {np.pi:.10f})")

# 2D Gaussian
def gaussian_2d(y, x, sigma_x=1.0, sigma_y=1.0):
    return np.exp(-x**2/(2*sigma_x**2) - y**2/(2*sigma_y**2))

result, _ = integrate.dblquad(gaussian_2d, -5, 5, -5, 5)
print(f"2D Gaussian: {result:.6f} (exact: 2π = {2*np.pi:.6f})")
```

### 7. Triple Integrals

```python
# Volume of unit sphere: ∫∫∫ dV = 4π/3
def sphere_integrand(z, y, x):
    return 1.0

def z_lower(x, y):
    r2 = x**2 + y**2
    if r2 > 1:
        return 0
    return -np.sqrt(1 - r2)

def z_upper(x, y):
    r2 = x**2 + y**2
    if r2 > 1:
        return 0
    return np.sqrt(1 - r2)

result, _ = integrate.tplquad(
    sphere_integrand,
    -1, 1,  # x limits
    lambda x: -np.sqrt(max(0, 1-x**2)), lambda x: np.sqrt(max(0, 1-x**2)),  # y limits
    z_lower, z_upper  # z limits
)
print(f"Sphere volume: {result:.6f} (exact: {4*np.pi/3:.6f})")
```

### 8. Fixed-Point Quadrature

```python
# Gaussian quadrature with fixed points (faster, less flexible)
from scipy.integrate import fixed_quad

def simple_function(x):
    return x**3 + 2*x + 1

# n = number of Gauss points
result, _ = fixed_quad(simple_function, 0, 1, n=5)
print(f"Fixed quad (n=5): {result:.10f}")

# Compare with adaptive
adaptive, _ = integrate.quad(simple_function, 0, 1)
print(f"Adaptive quad: {adaptive:.10f}")
print(f"Exact: {1/4 + 1 + 1:.10f}")

# Romberg integration (Richardson extrapolation)
result = integrate.romberg(simple_function, 0, 1)
print(f"Romberg: {result:.10f}")
```

---

## Quantum Mechanics Connection

### Normalization Integrals

$$\langle\psi|\psi\rangle = \int_{-\infty}^{\infty} |\psi(x)|^2 dx = 1$$

```python
from scipy import integrate
from scipy.special import hermite
from math import factorial, pi

def harmonic_oscillator_normalized(x, n, m=1.0, omega=1.0, hbar=1.0):
    """Normalized harmonic oscillator eigenstate."""
    alpha = np.sqrt(m * omega / hbar)
    Hn = hermite(n)
    norm = (alpha / (np.sqrt(pi) * 2**n * factorial(n)))**0.5
    return norm * Hn(alpha * x) * np.exp(-alpha**2 * x**2 / 2)

# Verify normalization
for n in range(5):
    integrand = lambda x, n=n: harmonic_oscillator_normalized(x, n)**2
    norm_sq, _ = integrate.quad(integrand, -np.inf, np.inf)
    print(f"⟨ψ_{n}|ψ_{n}⟩ = {norm_sq:.10f}")
```

### Overlap Integrals (Orthogonality)

$$\langle\psi_m|\psi_n\rangle = \int_{-\infty}^{\infty} \psi_m^*(x) \psi_n(x) dx = \delta_{mn}$$

```python
def overlap_integral(m, n):
    """Compute ⟨ψ_m|ψ_n⟩."""
    integrand = lambda x: (harmonic_oscillator_normalized(x, m) *
                          harmonic_oscillator_normalized(x, n))
    result, _ = integrate.quad(integrand, -np.inf, np.inf)
    return result

print("Overlap matrix:")
for m in range(4):
    row = [f"{overlap_integral(m, n):+.6f}" for n in range(4)]
    print(f"  {' '.join(row)}")
```

### Expectation Values

$$\langle\hat{x}\rangle = \int_{-\infty}^{\infty} \psi^*(x) \, x \, \psi(x) dx$$

```python
def expectation_x(n):
    """Compute ⟨x⟩ for state n."""
    integrand = lambda x: harmonic_oscillator_normalized(x, n)**2 * x
    result, _ = integrate.quad(integrand, -np.inf, np.inf)
    return result

def expectation_x2(n):
    """Compute ⟨x²⟩ for state n."""
    integrand = lambda x: harmonic_oscillator_normalized(x, n)**2 * x**2
    result, _ = integrate.quad(integrand, -np.inf, np.inf)
    return result

print("Position expectations:")
for n in range(5):
    x_avg = expectation_x(n)
    x2_avg = expectation_x2(n)
    delta_x = np.sqrt(x2_avg - x_avg**2)
    print(f"  n={n}: ⟨x⟩={x_avg:+.6f}, ⟨x²⟩={x2_avg:.6f}, Δx={delta_x:.6f}")
```

### Transition Matrix Elements

$$\langle\psi_m|\hat{x}|\psi_n\rangle = \int \psi_m^*(x) \, x \, \psi_n(x) dx$$

```python
def position_matrix_element(m, n):
    """Compute ⟨m|x|n⟩."""
    integrand = lambda x: (harmonic_oscillator_normalized(x, m) *
                          x *
                          harmonic_oscillator_normalized(x, n))
    result, _ = integrate.quad(integrand, -np.inf, np.inf)
    return result

print("\nPosition matrix elements (selection rules: ⟨m|x|n⟩ ≠ 0 only if |m-n|=1):")
for m in range(4):
    row = [f"{position_matrix_element(m, n):+.4f}" for n in range(4)]
    print(f"  {' '.join(row)}")
```

---

## Worked Examples

### Example 1: Particle in Box Normalization

```python
def particle_in_box(x, n, L=1.0):
    """Particle in box eigenstate."""
    if 0 <= x <= L:
        return np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    return 0.0

# Verify normalization
L = 1.0
for n in [1, 2, 3]:
    norm_sq, err = integrate.quad(
        lambda x: particle_in_box(x, n, L)**2, 0, L
    )
    print(f"n={n}: ⟨ψ|ψ⟩ = {norm_sq:.10f} ± {err:.2e}")
```

### Example 2: Hydrogen Atom Radial Integral

```python
def hydrogen_radial_1s(r, a0=1.0):
    """Hydrogen 1s radial wave function R_{10}(r)."""
    return 2 * (1/a0)**1.5 * np.exp(-r/a0)

def hydrogen_prob_density_1s(r, a0=1.0):
    """Radial probability density |R|²r²."""
    R = hydrogen_radial_1s(r, a0)
    return R**2 * r**2

# Normalization: ∫₀^∞ |R|²r² dr = 1
norm, _ = integrate.quad(hydrogen_prob_density_1s, 0, np.inf)
print(f"1s normalization: {norm:.10f}")

# ⟨r⟩ for 1s state
r_avg, _ = integrate.quad(
    lambda r: hydrogen_prob_density_1s(r) * r, 0, np.inf
)
print(f"⟨r⟩₁ₛ = {r_avg:.6f} a₀ (exact: 1.5 a₀)")

# ⟨r²⟩
r2_avg, _ = integrate.quad(
    lambda r: hydrogen_prob_density_1s(r) * r**2, 0, np.inf
)
print(f"⟨r²⟩₁ₛ = {r2_avg:.6f} a₀² (exact: 3 a₀²)")
```

### Example 3: Tunneling Probability

```python
def tunneling_probability(E, V0, L, m=1.0, hbar=1.0):
    """
    Compute tunneling probability through rectangular barrier.

    T = |t|² where t is transmission coefficient.
    For E < V0: T = 1/(1 + (V0²sinh²(κL))/(4E(V0-E)))
    """
    if E >= V0:
        # Above barrier - classical transmission
        return 1.0

    kappa = np.sqrt(2 * m * (V0 - E)) / hbar
    sinh_term = np.sinh(kappa * L)
    T = 1 / (1 + (V0**2 * sinh_term**2) / (4 * E * (V0 - E)))
    return T

# WKB approximation integral
def wkb_tunneling(E, V_func, x1, x2, m=1.0, hbar=1.0):
    """
    WKB tunneling probability: T ≈ exp(-2∫κ(x)dx)

    where κ(x) = √(2m(V(x)-E))/ℏ in classically forbidden region.
    """
    def kappa(x):
        V = V_func(x)
        if V > E:
            return np.sqrt(2 * m * (V - E)) / hbar
        return 0.0

    integral, _ = integrate.quad(kappa, x1, x2)
    return np.exp(-2 * integral)

# Test with rectangular barrier
V0, L = 2.0, 1.0
V_rect = lambda x: V0 if 0 < x < L else 0

E_values = [0.5, 1.0, 1.5, 1.9]
print("Tunneling probabilities (V0=2, L=1):")
for E in E_values:
    T_exact = tunneling_probability(E, V0, L)
    T_wkb = wkb_tunneling(E, V_rect, 0, L)
    print(f"  E={E}: Exact={T_exact:.6f}, WKB={T_wkb:.6f}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Compute $\int_0^\infty x^3 e^{-x} dx$ using `quad` and verify against the analytical result $\Gamma(4) = 6$.

**Problem 2:** Verify orthonormality of the first 5 particle-in-a-box states by computing all $\langle\psi_m|\psi_n\rangle$.

**Problem 3:** Compute the expectation value $\langle p^2 \rangle$ for the harmonic oscillator ground state using the momentum representation.

### Intermediate

**Problem 4:** Calculate the ground state energy of hydrogen using $E = \langle T \rangle + \langle V \rangle$ with numerical integration of the appropriate integrals.

**Problem 5:** Compute the dipole matrix element $\langle 2p_z | z | 1s \rangle$ for hydrogen (involves angular integral over $Y_{lm}$).

**Problem 6:** Implement numerical integration of the oscillatory integral $\int_0^\infty \sin(kx) e^{-\alpha x} dx$ and compare with the analytical result.

### Challenging

**Problem 7:** Compute the overlap integral between a Gaussian wave packet and harmonic oscillator eigenstates to find expansion coefficients.

**Problem 8:** Calculate the Franck-Condon factors (overlap integrals between vibrational states of different potential curves) for a displaced harmonic oscillator.

**Problem 9:** Implement adaptive integration for the Coulomb integral in a two-electron system (electron-electron repulsion).

---

## Computational Lab

```python
"""
Day 260 Lab: Numerical Integration for Quantum Mechanics
========================================================
"""

import numpy as np
from scipy import integrate
from scipy.special import hermite
from math import factorial, pi
from typing import Callable, Tuple

# ============================================================
# WAVE FUNCTION LIBRARY
# ============================================================

def harmonic_oscillator(x: np.ndarray, n: int, omega: float = 1.0) -> np.ndarray:
    """Normalized harmonic oscillator eigenstate."""
    Hn = hermite(n)
    norm = (omega/pi)**0.25 / np.sqrt(2**n * factorial(n))
    return norm * Hn(np.sqrt(omega)*x) * np.exp(-omega*x**2/2)

def gaussian_wavepacket(x: np.ndarray, x0: float = 0.0,
                        sigma: float = 1.0, k0: float = 0.0) -> np.ndarray:
    """Gaussian wave packet."""
    norm = (2*pi*sigma**2)**(-0.25)
    return norm * np.exp(-(x-x0)**2/(4*sigma**2)) * np.exp(1j*k0*x)

# ============================================================
# INTEGRATION UTILITIES
# ============================================================

def normalize_wavefunction(psi_func: Callable, x_range: Tuple[float, float] = (-np.inf, np.inf)) -> float:
    """Compute normalization constant for a wave function."""
    integrand = lambda x: np.abs(psi_func(x))**2
    norm_sq, _ = integrate.quad(integrand, *x_range)
    return np.sqrt(norm_sq)

def inner_product(psi1_func: Callable, psi2_func: Callable,
                  x_range: Tuple[float, float] = (-np.inf, np.inf)) -> complex:
    """Compute ⟨ψ₁|ψ₂⟩."""
    integrand_real = lambda x: np.real(np.conj(psi1_func(x)) * psi2_func(x))
    integrand_imag = lambda x: np.imag(np.conj(psi1_func(x)) * psi2_func(x))

    real_part, _ = integrate.quad(integrand_real, *x_range)
    imag_part, _ = integrate.quad(integrand_imag, *x_range)

    return real_part + 1j * imag_part

def expectation_value(psi_func: Callable, operator_func: Callable,
                      x_range: Tuple[float, float] = (-np.inf, np.inf)) -> float:
    """Compute ⟨ψ|Ô|ψ⟩ for diagonal operators O(x)."""
    integrand = lambda x: np.abs(psi_func(x))**2 * operator_func(x)
    result, _ = integrate.quad(integrand, *x_range)
    return result

def transition_amplitude(psi1_func: Callable, operator_func: Callable,
                         psi2_func: Callable,
                         x_range: Tuple[float, float] = (-np.inf, np.inf)) -> complex:
    """Compute ⟨ψ₁|Ô|ψ₂⟩."""
    integrand_real = lambda x: np.real(np.conj(psi1_func(x)) * operator_func(x) * psi2_func(x))
    integrand_imag = lambda x: np.imag(np.conj(psi1_func(x)) * operator_func(x) * psi2_func(x))

    real_part, _ = integrate.quad(integrand_real, *x_range)
    imag_part, _ = integrate.quad(integrand_imag, *x_range)

    return real_part + 1j * imag_part

# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Day 260: Numerical Integration for Quantum Mechanics")
    print("=" * 70)

    # 1. Verify normalization
    print("\n1. NORMALIZATION CHECK")
    print("-" * 40)
    for n in range(6):
        psi = lambda x, n=n: harmonic_oscillator(x, n)
        norm = normalize_wavefunction(psi)
        print(f"   |ψ_{n}|: {norm:.10f}")

    # 2. Orthogonality
    print("\n2. ORTHOGONALITY CHECK (first 4 states)")
    print("-" * 40)
    print("   ", end="")
    for n in range(4):
        print(f"    ψ_{n}    ", end="")
    print()

    for m in range(4):
        print(f"ψ_{m} ", end="")
        for n in range(4):
            psi_m = lambda x, m=m: harmonic_oscillator(x, m)
            psi_n = lambda x, n=n: harmonic_oscillator(x, n)
            overlap = inner_product(psi_m, psi_n)
            print(f" {overlap.real:+.6f} ", end="")
        print()

    # 3. Position expectation values
    print("\n3. POSITION EXPECTATION VALUES")
    print("-" * 40)
    print(f"{'n':>3} {'⟨x⟩':>10} {'⟨x²⟩':>10} {'Δx':>10} {'Theory':>10}")

    for n in range(5):
        psi = lambda x, n=n: harmonic_oscillator(x, n)
        x_avg = expectation_value(psi, lambda x: x)
        x2_avg = expectation_value(psi, lambda x: x**2)
        delta_x = np.sqrt(x2_avg - x_avg**2)
        theory_delta_x = np.sqrt(n + 0.5)
        print(f"{n:>3} {x_avg:>+10.6f} {x2_avg:>10.6f} {delta_x:>10.6f} {theory_delta_x:>10.6f}")

    # 4. Transition matrix elements
    print("\n4. POSITION MATRIX ELEMENTS ⟨m|x|n⟩")
    print("-" * 40)
    print("   (Selection rule: non-zero only for |m-n|=1)")

    for m in range(5):
        row = []
        for n in range(5):
            psi_m = lambda x, m=m: harmonic_oscillator(x, m)
            psi_n = lambda x, n=n: harmonic_oscillator(x, n)
            element = transition_amplitude(psi_m, lambda x: x, psi_n)
            row.append(f"{element.real:+.4f}")
        print(f"   {' '.join(row)}")

    # 5. Wave packet expansion
    print("\n5. GAUSSIAN WAVE PACKET EXPANSION")
    print("-" * 40)
    x0, sigma = 2.0, 0.5
    wavepacket = lambda x: gaussian_wavepacket(x, x0=x0, sigma=sigma, k0=0)

    print(f"   Expanding Gaussian (x₀={x0}, σ={sigma}) in HO basis:")
    total_prob = 0
    for n in range(10):
        psi_n = lambda x, n=n: harmonic_oscillator(x, n)
        c_n = inner_product(psi_n, wavepacket)
        prob = np.abs(c_n)**2
        total_prob += prob
        print(f"   |c_{n}|² = {prob:.6f}")

    print(f"   Total probability (sum of first 10): {total_prob:.6f}")

    # 6. Multi-dimensional integral
    print("\n6. TWO-PARTICLE OVERLAP (2D INTEGRAL)")
    print("-" * 40)

    def two_particle_product(x1, x2):
        """Product of two ground states."""
        return harmonic_oscillator(x1, 0) * harmonic_oscillator(x2, 0)

    result, _ = integrate.dblquad(
        lambda x2, x1: two_particle_product(x1, x2)**2,
        -10, 10, -10, 10
    )
    print(f"   ∫∫|ψ₀(x₁)|²|ψ₀(x₂)|² dx₁dx₂ = {result:.6f} (expect 1.0)")

    print("\n" + "=" * 70)
    print("Lab complete! ODE solvers continue on Day 261.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `quad(f, a, b)` | Definite integral | `integrate.quad(f, 0, 1)` |
| `dblquad(f, a, b, g, h)` | Double integral | `integrate.dblquad(f, 0, 1, 0, 1)` |
| `tplquad(...)` | Triple integral | Volume integrals |
| `fixed_quad(f, a, b, n)` | Gaussian quadrature | Fast, fixed precision |
| `romberg(f, a, b)` | Romberg integration | High precision |

### Quantum Integration Patterns

| Integral | Code Pattern |
|----------|--------------|
| $\langle\psi\|\psi\rangle$ | `quad(lambda x: abs(psi(x))**2, -inf, inf)` |
| $\langle\phi\|\psi\rangle$ | `quad(lambda x: conj(phi(x))*psi(x), ...)` |
| $\langle\hat{O}\rangle$ | `quad(lambda x: abs(psi)**2 * O(x), ...)` |
| $\langle m\|\hat{O}\|n\rangle$ | `quad(lambda x: conj(psi_m)*O*psi_n, ...)` |

---

## Daily Checklist

- [ ] Can use `quad` for single integrals with error estimates
- [ ] Handle infinite limits and singularities
- [ ] Compute multi-dimensional integrals with `dblquad`/`tplquad`
- [ ] Verify quantum mechanical normalization
- [ ] Calculate overlap integrals (orthogonality)
- [ ] Compute expectation values numerically
- [ ] Completed all practice problems
- [ ] Ran lab successfully

---

## Preview: Day 261

Tomorrow we tackle **Ordinary Differential Equations** with `scipy.integrate.solve_ivp`. We'll learn:
- Solving initial value problems
- Different ODE solvers (RK45, BDF for stiff equations)
- Time-dependent Schrödinger equation
- Event detection for quantum transitions

This enables simulating quantum dynamics!

---

*"Integration is the inverse of differentiation, but numerical integration is an art in itself."*
