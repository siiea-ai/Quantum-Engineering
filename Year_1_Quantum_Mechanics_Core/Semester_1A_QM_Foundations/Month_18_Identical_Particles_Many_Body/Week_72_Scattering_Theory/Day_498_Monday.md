# Day 498: Scattering Formalism

## Overview

**Day 498 of 2520 | Week 72, Day 1 | Month 18: Identical Particles & Many-Body Physics**

Today we begin our study of quantum scattering theory—the theoretical framework for understanding how particles interact with potentials and each other. Scattering experiments are the primary window into the quantum world, from Rutherford's alpha particle experiments that revealed the atomic nucleus to modern particle physics at the Large Hadron Collider. We establish the fundamental concepts: cross sections, scattering amplitude, and the asymptotic form of the wave function.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Classical vs Quantum Scattering | 60 min |
| 10:00 AM | Cross Section Definitions | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Scattering Amplitude f(θ) | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Asymptotic Wave Function | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Probability Current Analysis | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the setup of a quantum scattering experiment
2. **Define** differential and total cross sections
3. **Derive** the relationship between cross section and scattering amplitude
4. **Write** the asymptotic form of the scattered wave function
5. **Calculate** probability currents for incident and scattered waves
6. **Apply** these concepts to simple scattering scenarios

---

## 1. The Scattering Problem

### Classical vs Quantum Scattering

**Classical picture:** A particle with definite trajectory deflects by angle θ based on impact parameter b.

**Quantum picture:** A wave packet or plane wave interacts with a potential, producing an outgoing spherical wave.

### The Experimental Setup

Consider a beam of particles incident on a target (localized potential):

```
                     Detector at angle θ
                           ●
                          /
                         /
    ═══════════════════●═══════════════════►
    Incident beam      Target              Forward
    (plane wave)       V(r)                direction
                         \
                          \
                           ●
                     Detector at angle -θ
```

**Key assumptions:**
1. Target is localized: $V(\mathbf{r}) \to 0$ as $r \to \infty$
2. Incident beam is effectively a plane wave
3. Scattering is elastic: energy is conserved

### Time-Independent Formulation

For a stationary scattering state at energy $E = \hbar^2k^2/2m$:

$$\left[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r})\right]\psi(\mathbf{r}) = E\psi(\mathbf{r})$$

Rewritten as:

$$(\nabla^2 + k^2)\psi(\mathbf{r}) = \frac{2m}{\hbar^2}V(\mathbf{r})\psi(\mathbf{r})$$

This is an inhomogeneous Helmholtz equation.

---

## 2. Cross Section: The Observable

### Differential Cross Section

The **differential cross section** $d\sigma/d\Omega$ is defined by:

$$\boxed{\frac{d\sigma}{d\Omega} = \frac{\text{Number of particles scattered into } d\Omega \text{ per unit time}}{\text{Incident flux}}}$$

**Physical interpretation:** If I aim N particles per second per unit area at a target, how many scatter into solid angle $d\Omega$ around direction $(\theta, \phi)$?

### Units and Dimensions

$$\left[\frac{d\sigma}{d\Omega}\right] = \frac{[\text{particles/time}]}{[\text{particles/(area·time)}] \cdot [\text{steradian}]} = [\text{area}]$$

Cross section has dimensions of **area**—it represents an "effective target area" for scattering.

**Common units:**
- Atomic physics: $a_0^2$ (Bohr radius squared)
- Nuclear physics: barns (1 b = $10^{-24}$ cm²)
- Particle physics: fb (femtobarns = $10^{-39}$ cm²)

### Total Cross Section

Integrate over all angles:

$$\boxed{\sigma_{tot} = \int \frac{d\sigma}{d\Omega} d\Omega = \int_0^{2\pi}d\phi \int_0^{\pi}\sin\theta \, d\theta \, \frac{d\sigma}{d\Omega}}$$

For azimuthally symmetric scattering (central potential):

$$\sigma_{tot} = 2\pi \int_0^{\pi} \sin\theta \, \frac{d\sigma}{d\Omega} \, d\theta$$

### Physical Example: Hard Sphere

Classical hard sphere of radius R:

$$\frac{d\sigma}{d\Omega} = \frac{R^2}{4} \quad \text{(isotropic)}$$

$$\sigma_{tot} = \pi R^2 \quad \text{(geometric cross section)}$$

---

## 3. The Scattering Amplitude

### Asymptotic Wave Function

Far from the scatterer, the wave function must have the form:

$$\boxed{\psi(\mathbf{r}) \xrightarrow{r \to \infty} A\left[e^{ikz} + f(\theta, \phi)\frac{e^{ikr}}{r}\right]}$$

**Components:**
- $e^{ikz}$: Incident plane wave (traveling in +z direction)
- $f(\theta, \phi)\frac{e^{ikr}}{r}$: Outgoing spherical wave
- $f(\theta, \phi)$: **Scattering amplitude**

### Why This Form?

**Incident wave:** In the asymptotic region, the potential vanishes. Without scattering, we'd have just the incident plane wave.

**Scattered wave:** Must be outgoing (causality) and satisfy the free Schrödinger equation. The general outgoing solution is:

$$\psi_{out} \propto \frac{e^{ikr}}{r}Y_\ell^m(\theta, \phi)$$

The factor $1/r$ ensures the scattered wave carries finite flux to infinity.

### Central Potentials

For spherically symmetric potentials $V(r)$, angular momentum is conserved. The scattering amplitude depends only on the scattering angle:

$$f(\theta, \phi) = f(\theta)$$

We'll assume this unless otherwise stated.

### The Fundamental Relation

The cross section is the modulus squared of the scattering amplitude:

$$\boxed{\frac{d\sigma}{d\Omega} = |f(\theta)|^2}$$

**Proof:** See Section 4 below.

---

## 4. Probability Current Analysis

### Probability Current Definition

The probability current density is:

$$\mathbf{j} = \frac{\hbar}{2mi}\left(\psi^*\nabla\psi - \psi\nabla\psi^*\right) = \frac{\hbar}{m}\text{Im}(\psi^*\nabla\psi)$$

### Incident Current

For the plane wave $\psi_{inc} = e^{ikz}$:

$$\mathbf{j}_{inc} = \frac{\hbar k}{m}\hat{z} = v\hat{z}$$

where $v = \hbar k/m$ is the particle velocity.

**Incident flux:** $|\mathbf{j}_{inc}| = v$ (particles per unit area per unit time, with unit normalization)

### Scattered Current

For the scattered wave $\psi_{sc} = f(\theta)\frac{e^{ikr}}{r}$:

$$\nabla\psi_{sc} = f(\theta)\frac{e^{ikr}}{r}\left(ik - \frac{1}{r}\right)\hat{r} + O(r^{-2})$$

The radial probability current:

$$j_{sc}^{(r)} = \frac{\hbar}{m}\text{Im}\left(f^*\frac{e^{-ikr}}{r} \cdot f\frac{e^{ikr}}{r}\left(ik - \frac{1}{r}\right)\right)$$

$$j_{sc}^{(r)} = \frac{\hbar k}{m}\frac{|f(\theta)|^2}{r^2} = v\frac{|f(\theta)|^2}{r^2}$$

### Scattered Flux Through $d\Omega$

The number of particles passing through area $dA = r^2 d\Omega$ per unit time:

$$dN = j_{sc}^{(r)} \cdot r^2 d\Omega = v|f(\theta)|^2 d\Omega$$

### Deriving the Cross Section Formula

By definition:

$$\frac{d\sigma}{d\Omega} = \frac{dN/d\Omega}{j_{inc}} = \frac{v|f(\theta)|^2}{v} = |f(\theta)|^2$$

$$\boxed{\frac{d\sigma}{d\Omega} = |f(\theta)|^2}$$

---

## 5. Scattering Kinematics

### Momentum Transfer

The incident momentum: $\mathbf{k}_i = k\hat{z}$

The scattered momentum: $\mathbf{k}_f = k\hat{r}$ (elastic scattering, same magnitude)

**Momentum transfer:**

$$\mathbf{q} = \mathbf{k}_i - \mathbf{k}_f$$

Magnitude (using $\cos\theta = \hat{z}\cdot\hat{r}$):

$$q = |\mathbf{q}| = \sqrt{k^2 + k^2 - 2k^2\cos\theta} = 2k\sin\frac{\theta}{2}$$

$$\boxed{q = 2k\sin\frac{\theta}{2}}$$

### Energy-Momentum Relations

For non-relativistic particles:

$$E = \frac{\hbar^2k^2}{2m}, \quad k = \frac{\sqrt{2mE}}{\hbar}$$

At fixed energy, larger scattering angles mean larger momentum transfer.

### De Broglie Wavelength

$$\lambda = \frac{2\pi}{k} = \frac{h}{\sqrt{2mE}}$$

Quantum effects dominate when $\lambda \gtrsim$ range of potential.

---

## 6. Worked Examples

### Example 1: Dimensions Check

**Problem:** Verify that $|f(\theta)|^2$ has dimensions of area.

**Solution:**

From the asymptotic form: $\psi = e^{ikz} + f(\theta)\frac{e^{ikr}}{r}$

The plane wave $e^{ikz}$ is dimensionless.

The spherical wave $\frac{e^{ikr}}{r}$ has dimensions $[r]^{-1} = L^{-1}$.

For dimensional consistency: $[f] = L$ (length).

Therefore: $[|f|^2] = L^2$ (area). ✓

$$\boxed{[f] = \text{length}, \quad [d\sigma/d\Omega] = \text{area}}$$

### Example 2: Isotropic Scattering

**Problem:** A potential produces isotropic scattering with $f(\theta) = f_0$ (constant). If $|f_0| = 2$ fm, calculate the total cross section.

**Solution:**

$$\frac{d\sigma}{d\Omega} = |f_0|^2 = 4 \text{ fm}^2$$

$$\sigma_{tot} = \int |f_0|^2 d\Omega = |f_0|^2 \cdot 4\pi = 4 \times 4\pi = 16\pi \text{ fm}^2$$

$$\boxed{\sigma_{tot} = 16\pi \text{ fm}^2 \approx 50.3 \text{ fm}^2}$$

In barns: $\sigma_{tot} \approx 0.503$ barns.

### Example 3: Forward Peaking

**Problem:** A scattering amplitude is given by $f(\theta) = \frac{f_0}{1 + (\theta/\theta_0)^2}$ for small angles. Describe the angular distribution and estimate the forward peak width.

**Solution:**

$$\frac{d\sigma}{d\Omega} = \frac{|f_0|^2}{(1 + \theta^2/\theta_0^2)^2}$$

At $\theta = 0$: $\frac{d\sigma}{d\Omega}\Big|_{\theta=0} = |f_0|^2$ (maximum)

At $\theta = \theta_0$: $\frac{d\sigma}{d\Omega} = \frac{|f_0|^2}{4}$ (reduced by factor of 4)

**Interpretation:** The scattering is strongly forward-peaked with angular width $\sim\theta_0$. This is characteristic of scattering from an extended object of size $R \sim 1/(k\theta_0)$.

$$\boxed{\text{Forward peak width: } \Delta\theta \sim \theta_0 \sim \frac{1}{kR}}$$

---

## 7. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** The differential cross section for electron-helium scattering at a certain energy is approximately $d\sigma/d\Omega = 2.5 \times 10^{-16}$ cm² at $\theta = 45°$. What is the scattering amplitude magnitude?

**Problem 1.2:** A neutron beam with flux $10^{12}$ neutrons/(cm²·s) strikes a thin target. If $d\sigma/d\Omega = 10$ mb/sr at $90°$, how many neutrons scatter into a detector subtending $0.01$ sr at that angle per second?

**Problem 1.3:** For elastic scattering at $E = 50$ MeV, calculate the de Broglie wavelength for:
(a) Electrons
(b) Protons
(c) Alpha particles

### Level 2: Intermediate

**Problem 2.1:** A scattering amplitude has the form $f(\theta) = a + b\cos\theta$ where $a, b$ are complex constants with $|a| = 1$ fm and $|b| = 2$ fm. If $a$ and $b$ are both real and positive:
(a) Find $d\sigma/d\Omega$ as a function of $\theta$
(b) Calculate the total cross section
(c) Find the angle of maximum cross section

**Problem 2.2:** The Rutherford scattering amplitude is $f(\theta) = -\frac{\eta}{2k\sin^2(\theta/2)}$ where $\eta = Z_1Z_2e^2m/\hbar^2k$. Show that:
(a) $d\sigma/d\Omega = \left(\frac{Z_1Z_2e^2}{4E}\right)^2\frac{1}{\sin^4(\theta/2)}$
(b) The total cross section diverges (explain why physically)

**Problem 2.3:** For a wave function $\psi = e^{ikz} + f(\theta)\frac{e^{ikr}}{r}$, show that the interference term between incident and scattered waves vanishes when integrated over a large sphere.

### Level 3: Challenging

**Problem 3.1:** Consider a superposition of incident waves: $\psi_{inc} = \alpha e^{ikz} + \beta e^{-ikz}$. Write the complete asymptotic wave function including scattering from both directions and show how the cross section is modified.

**Problem 3.2:** Derive the relationship between the scattering amplitude and the S-matrix element for the $\ell = 0$ partial wave. Show that unitarity implies $|S_0| = 1$.

**Problem 3.3:** In the scattering of identical particles, the amplitude receives contributions from both direct and exchange processes. For identical bosons, show that the differential cross section becomes:
$$\frac{d\sigma}{d\Omega} = |f(\theta) + f(\pi - \theta)|^2$$

---

## 8. Computational Lab: Scattering Visualization

```python
"""
Day 498 Computational Lab: Scattering Formalism Visualization
Visualizing scattering wave functions, cross sections, and currents.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# Physical constants (in convenient units)
hbar = 1.0  # Natural units
m = 1.0


def asymptotic_wavefunction(x, z, k, f_theta_func):
    """
    Calculate asymptotic scattering wave function.

    Parameters:
    -----------
    x, z : arrays
        Position coordinates (y=0 plane)
    k : float
        Wave number
    f_theta_func : callable
        Scattering amplitude as function of theta

    Returns:
    --------
    psi : complex array
        Wave function values
    """
    r = np.sqrt(x**2 + z**2)
    # Avoid division by zero
    r = np.where(r < 0.1, 0.1, r)

    # Angle from z-axis
    theta = np.arctan2(np.abs(x), z)

    # Incident plane wave
    psi_inc = np.exp(1j * k * z)

    # Scattered spherical wave
    f_theta = f_theta_func(theta)
    psi_sc = f_theta * np.exp(1j * k * r) / r

    return psi_inc + psi_sc


def plot_scattering_wavefunction():
    """Visualize the total scattering wave function."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    k = 5.0  # Wave number

    # Different scattering amplitudes
    cases = [
        ("Isotropic: f = 1", lambda theta: 1.0),
        ("Forward peaked: f = 5/(1 + 10*theta^2)",
         lambda theta: 5.0 / (1 + 10 * theta**2)),
        ("p-wave: f = 2*cos(theta)",
         lambda theta: 2.0 * np.cos(theta)),
        ("Resonance-like: f = 3*exp(i*pi/4)",
         lambda theta: 3.0 * np.exp(1j * np.pi/4))
    ]

    # Grid for visualization
    x = np.linspace(-5, 5, 200)
    z = np.linspace(-3, 10, 260)
    X, Z = np.meshgrid(x, z)

    for ax, (title, f_func) in zip(axes.flat, cases):
        psi = asymptotic_wavefunction(X, Z, k, f_func)

        # Plot real part
        im = ax.pcolormesh(X, Z, np.real(psi),
                          shading='auto', cmap='RdBu',
                          vmin=-2, vmax=2)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('z (beam direction)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')

        # Mark scattering center
        ax.plot(0, 0, 'ko', markersize=10)

        plt.colorbar(im, ax=ax, label='Re(ψ)')

    plt.suptitle('Scattering Wave Functions (Real Part)\n' +
                 'Incident plane wave + scattered spherical wave',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('scattering_wavefunctions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cross_sections():
    """Plot differential cross sections for various amplitudes."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    theta = np.linspace(0.01, np.pi, 200)

    # Polar plot
    ax1 = plt.subplot(121, projection='polar')

    # Various scattering amplitudes
    amplitudes = {
        'Isotropic (s-wave)': lambda t: np.ones_like(t),
        'p-wave': lambda t: np.cos(t),
        'd-wave': lambda t: 0.5 * (3*np.cos(t)**2 - 1),
        's + p mixture': lambda t: 1 + 1.5*np.cos(t)
    }

    for name, f_func in amplitudes.items():
        f = f_func(theta)
        dsigma = np.abs(f)**2
        ax1.plot(theta, dsigma, label=name, linewidth=2)

    ax1.set_title('Differential Cross Section\n(polar plot)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)

    # Cartesian plot
    ax2 = axes[1]

    for name, f_func in amplitudes.items():
        f = f_func(theta)
        dsigma = np.abs(f)**2
        ax2.plot(np.degrees(theta), dsigma, label=name, linewidth=2)

    ax2.set_xlabel('Scattering angle θ (degrees)', fontsize=12)
    ax2.set_ylabel('dσ/dΩ (arbitrary units)', fontsize=12)
    ax2.set_title('Differential Cross Section vs Angle', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 180)

    plt.tight_layout()
    plt.savefig('differential_cross_sections.png', dpi=150, bbox_inches='tight')
    plt.show()


def calculate_total_cross_section(f_func, num_points=1000):
    """
    Calculate total cross section by numerical integration.

    σ_tot = ∫ |f(θ)|² dΩ = 2π ∫₀^π |f(θ)|² sin(θ) dθ
    """
    theta = np.linspace(0, np.pi, num_points)
    dtheta = theta[1] - theta[0]

    f = f_func(theta)
    integrand = np.abs(f)**2 * np.sin(theta)

    sigma = 2 * np.pi * np.sum(integrand) * dtheta
    return sigma


def plot_probability_currents():
    """Visualize probability current density."""

    fig, ax = plt.subplots(figsize=(12, 8))

    k = 3.0

    # Grid
    x = np.linspace(-4, 4, 20)
    z = np.linspace(-2, 8, 25)
    X, Z = np.meshgrid(x, z)

    # Scattering amplitude (forward-peaked)
    f0 = 2.0
    theta0 = 0.3

    def f_theta(theta):
        return f0 / (1 + (theta/theta0)**2)

    # Calculate current components
    # Incident current: j_inc = (hbar*k/m) z_hat
    j_inc_x = np.zeros_like(X)
    j_inc_z = np.ones_like(Z) * k

    # Scattered current (radial)
    r = np.sqrt(X**2 + Z**2)
    r = np.where(r < 0.5, 0.5, r)  # Avoid singularity
    theta = np.arctan2(np.abs(X), Z)

    f = f_theta(theta)
    j_sc_mag = k * np.abs(f)**2 / r**2

    # Radial components
    j_sc_x = j_sc_mag * X / r
    j_sc_z = j_sc_mag * Z / r

    # Total current
    j_tot_x = j_inc_x + j_sc_x
    j_tot_z = j_inc_z + j_sc_z

    # Normalize for visualization
    j_mag = np.sqrt(j_tot_x**2 + j_tot_z**2)
    j_mag = np.where(j_mag < 0.1, 0.1, j_mag)

    # Plot streamlines
    ax.streamplot(X, Z, j_tot_x, j_tot_z,
                  color=np.log10(j_mag), cmap='viridis',
                  density=1.5, linewidth=1.5)

    # Mark scatterer
    circle = plt.Circle((0, 0), 0.3, color='red', fill=True, alpha=0.7)
    ax.add_patch(circle)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('z (beam direction)', fontsize=12)
    ax.set_title('Probability Current Flow\n' +
                 '(Incident beam + scattered wave)', fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 8)

    plt.tight_layout()
    plt.savefig('probability_currents.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_cross_sections():
    """Compare various scattering scenarios."""

    print("=" * 60)
    print("CROSS SECTION COMPARISON")
    print("=" * 60)

    # Define scattering amplitudes
    cases = {
        'Isotropic (f = 1)': lambda t: np.ones_like(t),
        'Isotropic (f = 2)': lambda t: 2 * np.ones_like(t),
        'p-wave (f = cos θ)': lambda t: np.cos(t),
        'd-wave (f = P_2)': lambda t: 0.5 * (3*np.cos(t)**2 - 1),
        'Forward peaked': lambda t: 3 / (1 + 5*t**2),
        's + p': lambda t: 1 + 2*np.cos(t)
    }

    print("\nTotal Cross Sections:")
    print("-" * 40)

    for name, f_func in cases.items():
        sigma = calculate_total_cross_section(f_func)
        print(f"{name:25s}: σ_tot = {sigma:.4f}")

    # Analytical results for comparison
    print("\n" + "-" * 40)
    print("Analytical results for comparison:")
    print("Isotropic (f = a): σ = 4πa²")
    print("  f = 1: σ = 4π ≈ 12.566")
    print("  f = 2: σ = 16π ≈ 50.265")
    print("p-wave (f = cos θ): σ = 4π/3 ≈ 4.189")


def momentum_transfer_analysis():
    """Analyze momentum transfer in scattering."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # q vs theta for different k
    ax1 = axes[0]
    theta = np.linspace(0, np.pi, 100)

    for k in [1, 2, 5, 10]:
        q = 2 * k * np.sin(theta / 2)
        ax1.plot(np.degrees(theta), q, label=f'k = {k}', linewidth=2)

    ax1.set_xlabel('Scattering angle θ (degrees)', fontsize=12)
    ax1.set_ylabel('Momentum transfer q', fontsize=12)
    ax1.set_title('Momentum Transfer: q = 2k sin(θ/2)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cross section vs q (for Coulomb-like)
    ax2 = axes[1]

    k = 5
    theta = np.linspace(0.1, np.pi, 100)
    q = 2 * k * np.sin(theta / 2)

    # Rutherford-like: dσ/dΩ ∝ 1/sin⁴(θ/2) ∝ 1/q⁴
    dsigma_rutherford = 1 / (q**4)

    # Yukawa-like: dσ/dΩ ∝ 1/(q² + μ²)²
    mu = 1
    dsigma_yukawa = 1 / (q**2 + mu**2)**2

    ax2.semilogy(q, dsigma_rutherford / dsigma_rutherford[0],
                 label='Rutherford (∝ 1/q⁴)', linewidth=2)
    ax2.semilogy(q, dsigma_yukawa / dsigma_yukawa[0],
                 label=f'Yukawa (μ = {mu})', linewidth=2)

    ax2.set_xlabel('Momentum transfer q', fontsize=12)
    ax2.set_ylabel('dσ/dΩ (normalized)', fontsize=12)
    ax2.set_title('Cross Section vs Momentum Transfer', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('momentum_transfer.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_interference():
    """Show interference between incident and scattered waves."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    k = 4.0

    x = np.linspace(-6, 6, 300)
    z = np.linspace(-2, 12, 350)
    X, Z = np.meshgrid(x, z)

    # Incident wave only
    psi_inc = np.exp(1j * k * Z)

    # Scattered wave only (isotropic)
    r = np.sqrt(X**2 + Z**2)
    r = np.where(r < 0.2, 0.2, r)
    f0 = 2.0
    psi_sc = f0 * np.exp(1j * k * r) / r

    # Total
    psi_tot = psi_inc + psi_sc

    waves = [
        ('Incident Wave', np.real(psi_inc)),
        ('Scattered Wave', np.real(psi_sc)),
        ('Total (Interference)', np.real(psi_tot))
    ]

    for ax, (title, psi) in zip(axes, waves):
        im = ax.pcolormesh(X, Z, psi, shading='auto',
                          cmap='RdBu', vmin=-2, vmax=2)
        ax.plot(0, 0, 'ko', markersize=8)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('z', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Wave Interference in Scattering', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('scattering_interference.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Day 498: Scattering Formalism")
    print("=" * 60)

    plot_scattering_wavefunction()
    plot_cross_sections()
    plot_probability_currents()
    compare_cross_sections()
    momentum_transfer_analysis()
    demonstrate_interference()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Cross section | Effective area for scattering; $[σ] = $ area |
| Differential cross section | $d\sigma/d\Omega = $ scattering probability per solid angle |
| Scattering amplitude | $f(\theta)$ in asymptotic wave function |
| Asymptotic form | $\psi \to e^{ikz} + f(\theta)\frac{e^{ikr}}{r}$ |
| Momentum transfer | $q = 2k\sin(\theta/2)$ |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\psi \to e^{ikz} + f(\theta)\frac{e^{ikr}}{r}$$ | Asymptotic wave function |
| $$\frac{d\sigma}{d\Omega} = \|f(\theta)\|^2$$ | Cross section from amplitude |
| $$\sigma_{tot} = \int \|f(\theta)\|^2 d\Omega$$ | Total cross section |
| $$q = 2k\sin\frac{\theta}{2}$$ | Momentum transfer |
| $$\mathbf{j} = \frac{\hbar}{m}\text{Im}(\psi^*\nabla\psi)$$ | Probability current |

---

## 10. Daily Checklist

### Conceptual Understanding
- [ ] I can explain the physical meaning of cross section
- [ ] I understand why $d\sigma/d\Omega = |f(\theta)|^2$
- [ ] I can write the asymptotic wave function form
- [ ] I understand the role of probability currents

### Mathematical Skills
- [ ] I can calculate cross sections from scattering amplitudes
- [ ] I can perform the current density calculation
- [ ] I can compute momentum transfer at various angles
- [ ] I can integrate to find total cross sections

### Computational Skills
- [ ] I visualized scattering wave functions
- [ ] I plotted differential cross sections
- [ ] I computed probability current fields
- [ ] I analyzed momentum transfer dependence

---

## 11. Preview: Day 499

Tomorrow we study the **Born approximation**—a powerful method for calculating scattering amplitudes when the potential is weak:

- First Born approximation formula
- Scattering amplitude as Fourier transform
- Applications to Coulomb and Yukawa potentials
- Validity conditions
- Higher-order corrections

The Born approximation connects scattering directly to the potential's Fourier structure, providing physical insight into why different potentials produce different angular distributions.

---

## References

1. Griffiths, D.J. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Ch. 11.1-11.2.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 7.1-7.2.

3. Shankar, R. (2011). *Principles of Quantum Mechanics*, 2nd ed., Ch. 19.1-19.2.

4. Taylor, J.R. (2006). *Scattering Theory: The Quantum Theory of Nonrelativistic Collisions*.

---

*"The cross section is nature's way of telling us about the structure of the target."*
— Enrico Fermi

---

**Day 498 Complete.** Tomorrow: Born Approximation.
