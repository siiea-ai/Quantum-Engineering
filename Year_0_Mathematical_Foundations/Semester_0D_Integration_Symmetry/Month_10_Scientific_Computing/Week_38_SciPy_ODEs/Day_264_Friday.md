# Day 264: Special Functions and FFT

## Overview

**Day 264** | **Week 38** | **Month 10: Scientific Computing**

Today we harness `scipy.special` for the classical functions of mathematical physics and `scipy.fft` for spectral methods. Hermite polynomials, spherical harmonics, and Bessel functions are the building blocks of quantum mechanics. The Fourier transform connects position and momentum representations. These tools enable analytical-quality numerical solutions.

**Prerequisites:** Days 260-263, calculus, complex analysis basics
**Outcome:** Use special functions and FFT for quantum physics computations

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Orthogonal polynomials, special functions |
| Afternoon | 3 hours | Practice: FFT and spectral methods |
| Evening | 2 hours | Lab: Momentum space quantum mechanics |

---

## Learning Objectives

By the end of Day 264, you will be able to:

1. **Use Hermite polynomials** for harmonic oscillator wave functions
2. **Compute spherical harmonics** for angular momentum eigenstates
3. **Apply Bessel functions** to cylindrical and spherical problems
4. **Perform FFT and inverse FFT** for spectral analysis
5. **Transform between position and momentum space**
6. **Implement spectral methods** for solving PDEs
7. **Recognize special function patterns** in quantum mechanics

---

## Core Content

### 1. Hermite Polynomials

The Hermite polynomials $H_n(x)$ appear in harmonic oscillator wave functions:

```python
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Hermite polynomials
x = np.linspace(-4, 4, 200)

# Method 1: scipy.special.hermite returns polynomial object
for n in range(5):
    Hn = special.hermite(n)
    plt.plot(x, Hn(x), label=f'H_{n}(x)')

# Method 2: Direct evaluation at points
H3_values = special.eval_hermite(3, x)

# Physicist's vs Probabilist's convention
# scipy uses physicist's: H_n(x) with H_0=1, H_1=2x, H_2=4x²-2
print("First few Hermite polynomials:")
for n in range(5):
    Hn = special.hermite(n)
    print(f"  H_{n}(0) = {Hn(0):.0f}")
```

### 2. Harmonic Oscillator Wave Functions

$$\psi_n(x) = \frac{1}{\sqrt{2^n n!}} \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} H_n\left(\sqrt{\frac{m\omega}{\hbar}}x\right) e^{-\frac{m\omega x^2}{2\hbar}}$$

```python
from math import factorial, pi

def harmonic_oscillator_wf(x, n, m=1.0, omega=1.0, hbar=1.0):
    """
    Normalized harmonic oscillator eigenstate.
    """
    xi = np.sqrt(m * omega / hbar) * x
    Hn = special.hermite(n)

    prefactor = (m * omega / (pi * hbar))**0.25
    normalization = 1 / np.sqrt(2**n * factorial(n))

    return prefactor * normalization * Hn(xi) * np.exp(-xi**2 / 2)

# Generate eigenstates
x = np.linspace(-6, 6, 500)
for n in range(5):
    psi = harmonic_oscillator_wf(x, n)
    print(f"n={n}: max|ψ| = {np.max(np.abs(psi)):.4f}")

# Verify orthonormality
dx = x[1] - x[0]
for m in range(3):
    for n in range(m, 3):
        psi_m = harmonic_oscillator_wf(x, m)
        psi_n = harmonic_oscillator_wf(x, n)
        overlap = np.sum(psi_m * psi_n) * dx
        print(f"⟨{m}|{n}⟩ = {overlap:.6f}")
```

### 3. Legendre Polynomials and Associated Legendre Functions

```python
# Legendre polynomials P_l(x), x ∈ [-1, 1]
x = np.linspace(-1, 1, 200)

for l in range(5):
    Pl = special.legendre(l)
    print(f"P_{l}(1) = {Pl(1):.0f}, P_{l}(0) = {Pl(0):.4f}")

# Associated Legendre functions P_l^m(x)
l, m = 2, 1
Plm_values = special.lpmv(m, l, x)
print(f"\nP_{l}^{m}(0) = {special.lpmv(m, l, 0):.4f}")

# Normalization for spherical harmonics
# ∫₋₁¹ |P_l^m|² dx = 2(l+m)!/((2l+1)(l-m)!)
from scipy import integrate

def Plm_squared(x, l, m):
    return special.lpmv(m, l, x)**2

integral, _ = integrate.quad(Plm_squared, -1, 1, args=(2, 1))
theory = 2 * factorial(2+1) / ((2*2+1) * factorial(2-1))
print(f"∫|P_2^1|² dx = {integral:.6f} (theory: {theory:.6f})")
```

### 4. Spherical Harmonics

$$Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi}$$

```python
def spherical_harmonic(l, m, theta, phi):
    """
    Spherical harmonic Y_l^m(θ, φ).

    Uses scipy's sph_harm with (m, l, φ, θ) convention.
    """
    return special.sph_harm(m, l, phi, theta)

# Evaluate at specific angles
theta, phi = np.pi/4, np.pi/3

print("Spherical harmonics at θ=π/4, φ=π/3:")
for l in range(3):
    for m in range(-l, l+1):
        Y = spherical_harmonic(l, m, theta, phi)
        print(f"  Y_{l}^{m} = {Y:.4f}")

# Verify orthonormality on sphere
def test_orthonormality(l1, m1, l2, m2, n_theta=50, n_phi=100):
    """Test ∫Y*_l1m1 Y_l2m2 dΩ = δ_l1l2 δ_m1m2"""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)

    Y1 = spherical_harmonic(l1, m1, THETA, PHI)
    Y2 = spherical_harmonic(l2, m2, THETA, PHI)

    integrand = np.conj(Y1) * Y2 * np.sin(THETA)

    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    return np.sum(integrand) * dtheta * dphi

print("\nOrthonormality check:")
for (l1, m1), (l2, m2) in [((1, 0), (1, 0)), ((1, 0), (1, 1)), ((2, 1), (2, 1))]:
    result = test_orthonormality(l1, m1, l2, m2)
    print(f"  ⟨{l1},{m1}|{l2},{m2}⟩ = {result:.4f}")
```

### 5. Bessel Functions

```python
# Bessel functions of the first kind J_n(x)
x = np.linspace(0, 20, 500)

for n in range(4):
    Jn = special.jv(n, x)
    # Find first zero
    zeros = special.jn_zeros(n, 3)
    print(f"J_{n} first 3 zeros: {zeros}")

# Spherical Bessel functions j_l(x) = √(π/2x) J_{l+1/2}(x)
for l in range(4):
    jl = special.spherical_jn(l, x)
    print(f"j_{l}(0) limit = {1 if l==0 else 0}")

# Modified Bessel functions I_n, K_n
In = special.iv(0, x)  # Modified Bessel first kind
Kn = special.kv(0, x[1:])  # Modified Bessel second kind (singular at 0)

# Applications:
# - Cylindrical quantum dots: J_m(k_mn r/R) for radial wave function
# - Spherical wells: j_l(kr) for radial Schrödinger equation
# - Yukawa potential: modified Bessel functions
```

### 6. Fast Fourier Transform (FFT)

```python
from scipy import fft

# Basic FFT
N = 256
x = np.linspace(0, 2*np.pi, N, endpoint=False)
f = np.sin(5*x) + 0.5*np.sin(12*x)

# Forward FFT
F = fft.fft(f)
frequencies = fft.fftfreq(N, d=x[1]-x[0])

print("FFT of sin(5x) + 0.5*sin(12x):")
# Find peaks (non-zero Fourier coefficients)
threshold = 10
peaks = np.abs(F) > threshold
print(f"  Non-zero frequencies: {frequencies[peaks]}")

# Inverse FFT
f_reconstructed = fft.ifft(F)
print(f"  Reconstruction error: {np.max(np.abs(f - f_reconstructed.real)):.2e}")

# Real FFT (more efficient for real signals)
F_real = fft.rfft(f)
print(f"  rfft size: {len(F_real)} vs fft size: {len(F)}")
```

### 7. FFT for Quantum Mechanics

Position to momentum space:
$$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}} \int_{-\infty}^{\infty} \psi(x) e^{-ipx/\hbar} dx$$

```python
def position_to_momentum(psi_x, x, hbar=1.0):
    """
    Transform wave function from position to momentum space.
    """
    N = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0] + dx

    # Momentum grid (FFT convention)
    dp = 2 * np.pi * hbar / L
    p = fft.fftfreq(N, d=dx) * 2 * np.pi * hbar

    # FFT gives ∫ψ(x)e^(-2πikx/L)dx with k=n/L
    # We need ∫ψ(x)e^(-ipx/ℏ)dx, so k = p/(2πℏ)
    psi_p = fft.fft(psi_x) * dx / np.sqrt(2 * np.pi * hbar)

    # Shift to center zero momentum
    p = fft.fftshift(p)
    psi_p = fft.fftshift(psi_p)

    return p, psi_p

def momentum_to_position(psi_p, p, hbar=1.0):
    """
    Transform wave function from momentum to position space.
    """
    N = len(p)
    dp = p[1] - p[0]
    L_p = p[-1] - p[0] + dp

    # Position grid
    dx = 2 * np.pi * hbar / L_p
    x = fft.fftfreq(N, d=dp/(2*np.pi*hbar)) * 2 * np.pi * hbar

    # Unshift
    psi_p_unshifted = fft.ifftshift(psi_p)

    # IFFT
    psi_x = fft.ifft(psi_p_unshifted) * np.sqrt(2 * np.pi * hbar) / dx * N

    x = fft.ifftshift(x)
    psi_x = fft.ifftshift(psi_x)

    return x, psi_x

# Test with Gaussian wave packet
N = 512
L = 20
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

sigma = 1.0
k0 = 5.0
psi_x = (2*np.pi*sigma**2)**(-0.25) * np.exp(-x**2/(4*sigma**2)) * np.exp(1j*k0*x)

p, psi_p = position_to_momentum(psi_x, x)

# Momentum space should also be Gaussian centered at p=k0
print("Gaussian wave packet transformation:")
print(f"  Position width: σ_x = {sigma}")
print(f"  Expected momentum width: σ_p = ℏ/(2σ) = {0.5/sigma}")
print(f"  Center momentum: k_0 = {k0}")

# Find peak in momentum space
peak_idx = np.argmax(np.abs(psi_p))
print(f"  Peak momentum: p = {p[peak_idx]:.4f}")
```

### 8. Spectral Methods for Schrödinger Equation

```python
def split_operator_step(psi, V, dx, dt, hbar=1.0, m=1.0):
    """
    One time step using split-operator FFT method.

    e^(-iHdt/ℏ) ≈ e^(-iVdt/2ℏ) e^(-iTdt/ℏ) e^(-iVdt/2ℏ)
    """
    N = len(psi)

    # Momentum grid
    dp = 2 * np.pi * hbar / (N * dx)
    p = fft.fftfreq(N, d=dx) * 2 * np.pi * hbar

    # Half potential step in position space
    psi = psi * np.exp(-0.5j * V * dt / hbar)

    # Full kinetic step in momentum space
    psi_p = fft.fft(psi)
    T_p = p**2 / (2 * m)
    psi_p = psi_p * np.exp(-1j * T_p * dt / hbar)
    psi = fft.ifft(psi_p)

    # Half potential step in position space
    psi = psi * np.exp(-0.5j * V * dt / hbar)

    return psi

# Test: harmonic oscillator dynamics
N = 256
x = np.linspace(-10, 10, N)
dx = x[1] - x[0]
V = 0.5 * x**2

# Initial: displaced Gaussian
x0 = 3.0
psi = np.exp(-(x - x0)**2 / 2) * np.pi**(-0.25)
psi = psi.astype(complex)

# Evolve
dt = 0.01
n_steps = int(2 * np.pi / dt)  # One classical period

for step in range(n_steps):
    psi = split_operator_step(psi, V, dx, dt)

# Check: should return to initial position after period 2π
x_final = np.sum(np.abs(psi)**2 * x) * dx
print(f"Split-operator evolution (one period):")
print(f"  Initial ⟨x⟩ = {x0}")
print(f"  Final ⟨x⟩ = {x_final:.4f}")
print(f"  Norm: {np.sum(np.abs(psi)**2) * dx:.6f}")
```

---

## Worked Examples

### Example 1: Hydrogen Radial Functions

```python
def hydrogen_radial(r, n, l, a0=1.0):
    """
    Hydrogen radial wave function R_nl(r).

    R_nl(r) = √[(2/na0)³ (n-l-1)!/(2n(n+l)!)] × (2r/na0)^l × e^(-r/na0) × L_{n-l-1}^{2l+1}(2r/na0)
    """
    rho = 2 * r / (n * a0)

    # Normalization
    norm = np.sqrt((2/(n*a0))**3 * factorial(n-l-1) / (2*n*factorial(n+l)**3))

    # Associated Laguerre polynomial
    L = special.genlaguerre(n-l-1, 2*l+1)

    return norm * rho**l * np.exp(-rho/2) * L(rho)

r = np.linspace(0, 30, 500)

print("Hydrogen radial wave functions:")
for n, l in [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]:
    R = hydrogen_radial(r, n, l)
    # Verify normalization: ∫|R|² r² dr = 1
    integrand = R**2 * r**2
    norm = np.trapz(integrand, r)
    print(f"  R_{n}{l}: ∫|R|²r²dr = {norm:.6f}")
```

### Example 2: Angular Momentum Addition

```python
def clebsch_gordan_numerical(j1, m1, j2, m2, j, m):
    """
    Numerical evaluation of Clebsch-Gordan coefficient.

    Uses Wigner 3j symbol relation.
    """
    if m != m1 + m2:
        return 0.0

    if not (abs(j1 - j2) <= j <= j1 + j2):
        return 0.0

    # Convert to Wigner 3j
    from scipy.special import factorial

    # Use recursion relations (simplified implementation)
    # Full implementation would use Racah formula

    # For simple cases, use known values
    if j1 == 0.5 and j2 == 0.5:
        # Two spin-1/2 particles
        if j == 1:  # Triplet
            if m1 == 0.5 and m2 == 0.5:
                return 1.0 if m == 1 else 0.0
            elif m1 == -0.5 and m2 == -0.5:
                return 1.0 if m == -1 else 0.0
            elif m == 0:
                return 1/np.sqrt(2)
        elif j == 0 and m == 0:  # Singlet
            if m1 == 0.5 and m2 == -0.5:
                return 1/np.sqrt(2)
            elif m1 == -0.5 and m2 == 0.5:
                return -1/np.sqrt(2)

    return 0.0  # Not implemented for general case

# Test: two spin-1/2
print("Clebsch-Gordan for j1=j2=1/2:")
for j, m in [(1, 1), (1, 0), (1, -1), (0, 0)]:
    print(f"  |{j},{m}⟩ = ", end="")
    terms = []
    for m1 in [-0.5, 0.5]:
        for m2 in [-0.5, 0.5]:
            c = clebsch_gordan_numerical(0.5, m1, 0.5, m2, j, m)
            if abs(c) > 1e-10:
                terms.append(f"{c:.4f}|{m1},{m2}⟩")
    print(" + ".join(terms) if terms else "0")
```

### Example 3: Quantum Fourier Transform

```python
def quantum_fourier_transform(psi):
    """
    Quantum Fourier Transform of computational basis amplitudes.

    QFT|j⟩ = (1/√N) Σ_k e^(2πijk/N) |k⟩
    """
    N = len(psi)

    # QFT matrix
    j = np.arange(N)
    k = np.arange(N)
    QFT = np.exp(2j * np.pi * np.outer(j, k) / N) / np.sqrt(N)

    return QFT @ psi

def inverse_qft(psi):
    """Inverse QFT."""
    N = len(psi)
    j = np.arange(N)
    k = np.arange(N)
    QFT_dag = np.exp(-2j * np.pi * np.outer(j, k) / N) / np.sqrt(N)
    return QFT_dag @ psi

# Test: QFT of |0⟩ should give equal superposition
N = 8
psi_0 = np.zeros(N, dtype=complex)
psi_0[0] = 1.0

psi_qft = quantum_fourier_transform(psi_0)
print("QFT|0⟩ (should be equal superposition):")
print(f"  Amplitudes: {np.abs(psi_qft)}")
print(f"  All equal: {np.allclose(np.abs(psi_qft), 1/np.sqrt(N))}")

# QFT of |1⟩ should give phases
psi_1 = np.zeros(N, dtype=complex)
psi_1[1] = 1.0
psi_qft_1 = quantum_fourier_transform(psi_1)
print(f"\nQFT|1⟩ phases: {np.angle(psi_qft_1) / np.pi} × π")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Generate and plot the first 5 Hermite polynomials. Verify the recurrence relation $H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)$.

**Problem 2:** Compute $Y_2^1(\pi/3, \pi/4)$ and verify against the explicit formula.

**Problem 3:** Use FFT to find the frequency spectrum of a signal composed of three sine waves with frequencies 5, 12, and 20 Hz.

### Intermediate

**Problem 4:** Implement the radial wave function for a 3D harmonic oscillator and verify normalization.

**Problem 5:** Use the split-operator method to simulate a wave packet tunneling through a rectangular barrier.

**Problem 6:** Compute the momentum-space wave function for the hydrogen 1s state analytically and numerically (using FFT) and compare.

### Challenging

**Problem 7:** Implement a spectral method for solving the 1D Gross-Pitaevskii equation for a BEC.

**Problem 8:** Use spherical Bessel functions to solve the radial Schrödinger equation for a spherical well and find bound states.

**Problem 9:** Implement the discrete Hankel transform for problems with cylindrical symmetry.

---

## Computational Lab

```python
"""
Day 264 Lab: Special Functions and FFT
======================================
"""

import numpy as np
from scipy import special, fft, integrate
from math import factorial, pi

# [Full lab implementation]

if __name__ == "__main__":
    print("=" * 70)
    print("Day 264: Special Functions and FFT")
    print("=" * 70)

    # Demonstrations...

    print("\n" + "=" * 70)
    print("Lab complete! Sparse matrices continue on Day 265.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `special.hermite(n)` | Hermite polynomial | `Hn = special.hermite(3)` |
| `special.sph_harm(m,l,φ,θ)` | Spherical harmonic | `Y = special.sph_harm(1,2,phi,theta)` |
| `special.jv(n, x)` | Bessel function | `J0 = special.jv(0, x)` |
| `fft.fft(x)` | Forward FFT | `X = fft.fft(x)` |
| `fft.ifft(X)` | Inverse FFT | `x = fft.ifft(X)` |
| `fft.fftfreq(N, d)` | Frequency grid | `f = fft.fftfreq(N, dx)` |

### Quantum Applications

| Function | Application |
|----------|-------------|
| Hermite polynomials | Harmonic oscillator eigenstates |
| Spherical harmonics | Angular momentum, atomic orbitals |
| Bessel functions | Cylindrical/spherical problems |
| FFT | Position ↔ momentum transformation |

---

## Daily Checklist

- [ ] Can generate and use Hermite polynomials
- [ ] Understand spherical harmonics and their quantum meaning
- [ ] Know Bessel functions for different geometries
- [ ] Can perform FFT and inverse FFT
- [ ] Implemented position-momentum transformation
- [ ] Used split-operator method for time evolution
- [ ] Completed practice problems
- [ ] Ran lab successfully

---

## Preview: Day 265

Tomorrow we tackle **Sparse Matrices** with `scipy.sparse`. We'll learn:
- Sparse matrix formats (CSR, CSC, COO)
- Efficient storage for large Hamiltonians
- Sparse eigenvalue solvers
- Scaling to million-dimensional Hilbert spaces

Essential for real quantum simulations!

---

*"The Fourier transform reveals the hidden frequencies in any signal, including wave functions."*
