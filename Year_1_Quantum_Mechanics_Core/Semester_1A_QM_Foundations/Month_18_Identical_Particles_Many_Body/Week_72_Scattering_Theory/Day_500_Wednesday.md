# Day 500: Partial Wave Analysis

## Overview

**Day 500 of 2520 | Week 72, Day 3 | Month 18: Identical Particles & Many-Body Physics**

Today we develop partial wave analysis—a powerful technique that exploits angular momentum conservation in scattering from spherically symmetric potentials. By expanding the scattering amplitude in spherical harmonics, we decompose the problem into independent channels labeled by orbital angular momentum ℓ. Each partial wave contributes through a **phase shift** δℓ, which encodes all the physics of how that angular momentum component is affected by the potential. This formalism is essential for understanding resonances, low-energy scattering, and the S-matrix.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Plane Wave Expansion in Spherical Harmonics | 60 min |
| 10:00 AM | Radial Schrödinger Equation | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Phase Shifts δℓ | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Partial Wave Amplitude and Cross Section | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | The S-Matrix | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Expand** a plane wave in spherical harmonics and Bessel functions
2. **Solve** the radial Schrödinger equation for each partial wave
3. **Define** and calculate phase shifts from asymptotic behavior
4. **Express** the scattering amplitude in terms of phase shifts
5. **Construct** the S-matrix and verify unitarity
6. **Apply** partial wave analysis to simple potentials

---

## 1. Partial Wave Expansion

### Angular Momentum Conservation

For a central potential $V(r)$, angular momentum is conserved:

$$[\hat{H}, \hat{L}^2] = 0, \quad [\hat{H}, \hat{L}_z] = 0$$

Therefore, we can classify scattering states by orbital angular momentum quantum numbers $(\ell, m)$.

### Plane Wave Expansion

The incident plane wave can be expanded in spherical harmonics:

$$e^{ikz} = e^{ikr\cos\theta} = \sum_{\ell=0}^{\infty}(2\ell + 1)i^\ell j_\ell(kr)P_\ell(\cos\theta)$$

where:
- $j_\ell(kr)$ are **spherical Bessel functions**
- $P_\ell(\cos\theta)$ are **Legendre polynomials**

**Key spherical Bessel function properties:**
$$j_0(x) = \frac{\sin x}{x}, \quad j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}$$

$$j_\ell(x) \xrightarrow{x \to \infty} \frac{\sin(x - \ell\pi/2)}{x}$$

### Asymptotic Behavior

For large $kr$:

$$e^{ikz} \xrightarrow{r\to\infty} \sum_{\ell=0}^{\infty}(2\ell + 1)i^\ell \frac{\sin(kr - \ell\pi/2)}{kr}P_\ell(\cos\theta)$$

Using $\sin x = (e^{ix} - e^{-ix})/(2i)$:

$$e^{ikz} \sim \sum_{\ell}(2\ell + 1)P_\ell(\cos\theta)\frac{e^{ikr} - (-1)^\ell e^{-ikr}}{2ikr}$$

This is a sum of incoming ($e^{-ikr}$) and outgoing ($e^{ikr}$) spherical waves.

---

## 2. Scattering Wave Function

### The Full Wave Function

The complete scattering wave function must satisfy:
1. Schrödinger equation with potential V(r)
2. Correct asymptotic form: incident + scattered

$$\psi(\mathbf{r}) = \sum_{\ell=0}^{\infty}(2\ell + 1)i^\ell \frac{u_\ell(r)}{kr}P_\ell(\cos\theta)$$

where $u_\ell(r)$ is the radial wave function for partial wave $\ell$.

### Radial Schrödinger Equation

Substituting into the Schrödinger equation:

$$\boxed{\frac{d^2u_\ell}{dr^2} + \left[k^2 - \frac{\ell(\ell+1)}{r^2} - U(r)\right]u_\ell = 0}$$

where $U(r) = 2mV(r)/\hbar^2$.

**Boundary conditions:**
- $u_\ell(0) = 0$ (wave function finite at origin)
- $u_\ell(r) \sim$ outgoing + incoming waves as $r \to \infty$

### Free Particle Solution

For $V = 0$, the regular solution is:

$$u_\ell^{(0)}(r) = kr \cdot j_\ell(kr)$$

Asymptotically: $u_\ell^{(0)} \sim \sin(kr - \ell\pi/2)$

---

## 3. Phase Shifts

### Definition of Phase Shift

In the presence of a potential, the asymptotic form becomes:

$$\boxed{u_\ell(r) \xrightarrow{r \to \infty} A_\ell \sin(kr - \ell\pi/2 + \delta_\ell)}$$

The **phase shift** $\delta_\ell$ measures how the potential shifts the phase of each partial wave compared to free propagation.

### Physical Interpretation

- $\delta_\ell > 0$: **Attractive** potential pulls the wave inward (positive shift)
- $\delta_\ell < 0$: **Repulsive** potential pushes the wave outward (negative shift)
- $\delta_\ell = n\pi$: Resonance condition (see Day 502)

### Asymptotic Wave Function

The full scattering wave function at large $r$:

$$\psi \sim \sum_\ell (2\ell+1)i^\ell e^{i\delta_\ell}\frac{\sin(kr - \ell\pi/2 + \delta_\ell)}{kr}P_\ell(\cos\theta)$$

Rewriting in terms of incoming and outgoing waves:

$$\psi \sim \frac{1}{2ikr}\sum_\ell (2\ell+1)P_\ell(\cos\theta)\left[e^{2i\delta_\ell}e^{ikr} - (-1)^\ell e^{-ikr}\right]$$

---

## 4. Scattering Amplitude from Phase Shifts

### Extracting f(θ)

Comparing with $\psi \to e^{ikz} + f(\theta)\frac{e^{ikr}}{r}$:

The scattered wave must be the difference between the full wave and the incident wave.

**Result:**

$$\boxed{f(\theta) = \sum_{\ell=0}^{\infty}(2\ell + 1)f_\ell P_\ell(\cos\theta)}$$

where the **partial wave amplitude** is:

$$\boxed{f_\ell = \frac{e^{2i\delta_\ell} - 1}{2ik} = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k}}$$

### Alternative Forms

$$f_\ell = \frac{1}{k\cot\delta_\ell - ik} = \frac{1}{k}e^{i\delta_\ell}\sin\delta_\ell$$

### Differential Cross Section

$$\frac{d\sigma}{d\Omega} = |f(\theta)|^2 = \left|\sum_\ell (2\ell+1)f_\ell P_\ell(\cos\theta)\right|^2$$

### Total Cross Section

Using orthogonality of Legendre polynomials:

$$\int_{-1}^{1}P_\ell(x)P_{\ell'}(x)dx = \frac{2}{2\ell+1}\delta_{\ell\ell'}$$

$$\boxed{\sigma_{tot} = \frac{4\pi}{k^2}\sum_{\ell=0}^{\infty}(2\ell + 1)\sin^2\delta_\ell}$$

### Unitarity Bound

Since $\sin^2\delta_\ell \leq 1$:

$$\sigma_\ell \leq \frac{4\pi(2\ell + 1)}{k^2}$$

Maximum scattering (unitarity limit) when $\delta_\ell = \pi/2 + n\pi$.

---

## 5. The S-Matrix

### Definition

The **scattering matrix** (S-matrix) relates incoming to outgoing partial waves:

$$\boxed{S_\ell = e^{2i\delta_\ell}}$$

### Properties

**Unitarity:** $|S_\ell| = 1$ (probability conservation)

This follows from elastic scattering: no particles are created or destroyed.

**Time reversal:** $S_\ell = S_\ell^*$ for real potentials (only if $\delta_\ell$ real)

### Relation to Partial Wave Amplitude

$$f_\ell = \frac{S_\ell - 1}{2ik}$$

$$S_\ell = 1 + 2ikf_\ell$$

### Physical Meaning

- $S_\ell = 1$: No scattering in channel $\ell$
- $S_\ell = -1$: Maximum scattering ($\delta_\ell = \pi/2$)
- $|S_\ell| < 1$: Absorption (inelastic processes)

### S-Matrix for Multi-Channel Scattering

For multiple open channels, S becomes a matrix $S_{ij}$ with:
- Unitarity: $S^\dagger S = I$
- Cross sections involve $|S_{ij} - \delta_{ij}|^2$

---

## 6. Worked Examples

### Example 1: Hard Sphere Scattering

**Problem:** Calculate the s-wave ($\ell = 0$) phase shift for a hard sphere of radius $a$.

**Solution:**

Boundary condition: $\psi(a) = 0$ (wave function vanishes at sphere surface).

For $\ell = 0$, outside the sphere ($r > a$):
$$u_0(r) = A\sin(kr + \delta_0)$$

Boundary condition $u_0(a) = 0$:
$$\sin(ka + \delta_0) = 0 \implies ka + \delta_0 = n\pi$$

For small $ka$ (low energy), taking $n = 0$:
$$\boxed{\delta_0 = -ka}$$

**Cross section at low energy:**
$$\sigma_{tot} \approx \frac{4\pi}{k^2}\sin^2(-ka) \approx 4\pi a^2$$

Four times the geometric cross section! (Quantum diffraction effect)

### Example 2: Phase Shift Calculation

**Problem:** A potential has phase shifts $\delta_0 = 30°$, $\delta_1 = 15°$, $\delta_2 = 5°$, with higher $\delta_\ell \approx 0$. Calculate $d\sigma/d\Omega$ at $\theta = 90°$.

**Solution:**

At $\theta = 90°$: $\cos\theta = 0$
- $P_0(0) = 1$
- $P_1(0) = 0$
- $P_2(0) = -1/2$

Convert degrees to radians:
- $\delta_0 = \pi/6$, $\sin\delta_0 = 0.5$, $e^{i\delta_0} = 0.866 + 0.5i$
- $\delta_1 = \pi/12$, $\sin\delta_1 = 0.259$
- $\delta_2 = \pi/36$, $\sin\delta_2 = 0.087$

Partial wave amplitudes:
$$f_0 = \frac{e^{i\pi/6}\sin(\pi/6)}{k} = \frac{0.433 + 0.25i}{k}$$
$$f_1 = \frac{e^{i\pi/12}\sin(\pi/12)}{k} = \frac{0.250 + 0.067i}{k}$$
$$f_2 = \frac{e^{i\pi/36}\sin(\pi/36)}{k} = \frac{0.086 + 0.008i}{k}$$

At $\theta = 90°$:
$$f(90°) = f_0 \cdot 1 + 3f_1 \cdot 0 + 5f_2 \cdot(-0.5)$$
$$= \frac{1}{k}[(0.433 + 0.25i) - 2.5(0.086 + 0.008i)]$$
$$= \frac{1}{k}(0.218 + 0.230i)$$

$$\frac{d\sigma}{d\Omega}(90°) = |f(90°)|^2 = \frac{0.218^2 + 0.230^2}{k^2} = \boxed{\frac{0.100}{k^2}}$$

### Example 3: Verifying the Optical Theorem

**Problem:** Show that $\text{Im}[f(0)] = k\sigma_{tot}/(4\pi)$ using partial wave expansion.

**Solution:**

Forward scattering: $\theta = 0$, $P_\ell(1) = 1$ for all $\ell$.

$$f(0) = \sum_\ell (2\ell + 1)f_\ell = \sum_\ell (2\ell + 1)\frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$$

The imaginary part:
$$\text{Im}[f(0)] = \frac{1}{k}\sum_\ell (2\ell + 1)\text{Im}[e^{i\delta_\ell}\sin\delta_\ell]$$
$$= \frac{1}{k}\sum_\ell (2\ell + 1)\sin^2\delta_\ell$$

From the total cross section formula:
$$\sigma_{tot} = \frac{4\pi}{k^2}\sum_\ell (2\ell + 1)\sin^2\delta_\ell$$

Therefore:
$$\boxed{\text{Im}[f(0)] = \frac{k\sigma_{tot}}{4\pi}}$$

This is the **optical theorem**, which we'll study in detail on Day 503.

---

## 7. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** For $\ell = 0$ scattering with $\delta_0 = 45°$, calculate:
(a) The partial wave amplitude $f_0$
(b) The s-wave cross section $\sigma_0$
(c) The S-matrix element $S_0$

**Problem 1.2:** Verify that $j_0(kr) = \sin(kr)/(kr)$ satisfies the radial equation for $\ell = 0$, $V = 0$.

**Problem 1.3:** At what energy (in terms of $ka$) does the hard sphere s-wave phase shift equal $-\pi/4$?

### Level 2: Intermediate

**Problem 2.1:** A scattering amplitude is given by $f(\theta) = f_0 + f_1\cos\theta$. Express this in terms of partial wave amplitudes and find $\sigma_{tot}$.

**Problem 2.2:** For a square well potential $V(r) = -V_0$ for $r < a$, $V = 0$ for $r > a$:
(a) Write the s-wave radial equation inside and outside
(b) Match wave functions and derivatives at $r = a$
(c) Derive the expression for $\delta_0$

**Problem 2.3:** Show that if only s-wave scattering occurs ($\delta_\ell = 0$ for $\ell > 0$), then $d\sigma/d\Omega$ is isotropic.

### Level 3: Challenging

**Problem 3.1:** Derive the partial wave expansion for the Born approximation amplitude and show it gives:
$$\delta_\ell^{Born} = -\frac{mk}{\hbar^2}\int_0^\infty U(r)j_\ell^2(kr)r^2 dr$$

**Problem 3.2:** For two partial waves with $\delta_0 = 90°$ and $\delta_1 = 60°$, calculate the differential cross section at $\theta = 60°$ and show there is interference between s and p waves.

**Problem 3.3:** The Levinson theorem states that $\delta_\ell(0) - \delta_\ell(\infty) = n_\ell\pi$ where $n_\ell$ is the number of bound states with angular momentum $\ell$. For a potential with one s-wave bound state and no others, sketch $\delta_0(k)$ from threshold to high energy.

---

## 8. Computational Lab: Partial Wave Calculations

```python
"""
Day 500 Computational Lab: Partial Wave Analysis
Computing phase shifts and scattering cross sections.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

# Constants (natural units)
hbar = 1.0
m = 1.0


def legendre_p(l, x):
    """Compute Legendre polynomial P_l(x)."""
    if l == 0:
        return np.ones_like(x)
    elif l == 1:
        return x
    else:
        P_prev2 = np.ones_like(x)
        P_prev1 = x
        for n in range(2, l + 1):
            P_curr = ((2*n - 1) * x * P_prev1 - (n - 1) * P_prev2) / n
            P_prev2 = P_prev1
            P_prev1 = P_curr
        return P_curr


class PartialWaveScattering:
    """
    Compute phase shifts and cross sections using partial wave analysis.
    """

    def __init__(self, V_func, l_max=10):
        """
        Initialize with potential function V(r).

        Parameters:
        -----------
        V_func : callable
            Potential V(r)
        l_max : int
            Maximum angular momentum to consider
        """
        self.V = V_func
        self.l_max = l_max

    def radial_equation(self, r, y, l, k):
        """
        Radial Schrödinger equation as first-order system.
        y = [u, u']
        u'' + [k² - l(l+1)/r² - U(r)]u = 0
        """
        u, up = y
        U = 2 * m * self.V(r) / hbar**2

        # Centrifugal term (avoid r=0 singularity)
        if r < 1e-10:
            centrifugal = 0
        else:
            centrifugal = l * (l + 1) / r**2

        upp = (centrifugal + U - k**2) * u
        return [up, upp]

    def compute_phase_shift(self, k, l, r_max=50, r_match=40):
        """
        Compute phase shift δ_l for given k and l.

        Uses numerical integration and matching to asymptotic form.
        """
        # Initial conditions near r=0: u ~ r^(l+1)
        r_start = 0.01
        u_start = r_start**(l + 1)
        up_start = (l + 1) * r_start**l

        # Integrate outward
        r_span = (r_start, r_max)
        r_eval = np.linspace(r_start, r_max, 1000)

        sol = solve_ivp(
            lambda r, y: self.radial_equation(r, y, l, k),
            r_span, [u_start, up_start],
            t_eval=r_eval, method='RK45',
            rtol=1e-8, atol=1e-10
        )

        # Extract solution at matching radius
        idx = np.argmin(np.abs(sol.t - r_match))
        r_m = sol.t[idx]
        u_m = sol.y[0, idx]
        up_m = sol.y[1, idx]

        # Match to j_l and n_l
        j_l = spherical_jn(l, k * r_m)
        n_l = spherical_yn(l, k * r_m)
        jp_l = k * spherical_jn(l, k * r_m, derivative=True)
        np_l = k * spherical_yn(l, k * r_m, derivative=True)

        # tan(δ_l) = (k*j_l' - γ*j_l) / (k*n_l' - γ*n_l)
        # where γ = u'/u at r_match
        gamma = up_m / u_m

        numerator = jp_l * r_m - gamma * j_l * r_m
        denominator = np_l * r_m - gamma * n_l * r_m

        delta = np.arctan2(numerator, denominator)

        return delta

    def compute_all_phase_shifts(self, k):
        """Compute phase shifts for all l up to l_max."""
        phase_shifts = []
        for l in range(self.l_max + 1):
            try:
                delta = self.compute_phase_shift(k, l)
                phase_shifts.append(delta)
            except Exception:
                phase_shifts.append(0.0)
        return np.array(phase_shifts)

    def partial_wave_amplitude(self, delta_l, k, l):
        """Compute partial wave amplitude f_l."""
        return np.exp(1j * delta_l) * np.sin(delta_l) / k

    def scattering_amplitude(self, theta, k, phase_shifts):
        """Compute f(θ) from phase shifts."""
        f = 0j
        cos_theta = np.cos(theta)
        for l, delta in enumerate(phase_shifts):
            f_l = self.partial_wave_amplitude(delta, k, l)
            P_l = legendre_p(l, cos_theta)
            f += (2*l + 1) * f_l * P_l
        return f

    def differential_cross_section(self, theta, k, phase_shifts):
        """Compute dσ/dΩ."""
        f = self.scattering_amplitude(theta, k, phase_shifts)
        return np.abs(f)**2

    def total_cross_section(self, k, phase_shifts):
        """Compute total cross section."""
        sigma = 0
        for l, delta in enumerate(phase_shifts):
            sigma += (2*l + 1) * np.sin(delta)**2
        return 4 * np.pi / k**2 * sigma


def hard_sphere_phase_shift(k, a, l):
    """Analytical phase shift for hard sphere."""
    x = k * a
    j_l = spherical_jn(l, x)
    n_l = spherical_yn(l, x)
    return -np.arctan(j_l / n_l)


def plot_phase_shifts():
    """Analyze phase shifts for various potentials."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Hard sphere
    ax1 = axes[0, 0]
    a = 1.0
    k_values = np.linspace(0.1, 10, 100)

    for l in range(4):
        deltas = [hard_sphere_phase_shift(k, a, l) for k in k_values]
        ax1.plot(k_values * a, np.degrees(deltas), linewidth=2,
                 label=f'$\\ell$ = {l}')

    ax1.set_xlabel('ka', fontsize=12)
    ax1.set_ylabel('Phase shift δ (degrees)', fontsize=12)
    ax1.set_title('Hard Sphere Phase Shifts', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Square well
    ax2 = axes[0, 1]
    V0 = 5.0  # Well depth
    a = 1.0   # Well radius

    def square_well(r):
        return -V0 if r < a else 0

    scatter = PartialWaveScattering(square_well, l_max=3)
    k_values = np.linspace(0.1, 5, 50)

    for l in range(4):
        deltas = []
        for k in k_values:
            try:
                delta = scatter.compute_phase_shift(k, l)
                deltas.append(np.degrees(delta))
            except Exception:
                deltas.append(np.nan)
        ax2.plot(k_values, deltas, 'o-', markersize=3,
                 linewidth=1.5, label=f'$\\ell$ = {l}')

    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Phase shift δ (degrees)', fontsize=12)
    ax2.set_title(f'Square Well Phase Shifts (V₀ = {V0}, a = {a})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Yukawa potential
    ax3 = axes[1, 0]
    V0 = 3.0
    mu = 1.0

    def yukawa(r):
        if r < 0.01:
            return V0 * mu
        return V0 * np.exp(-mu * r) / r

    scatter_yukawa = PartialWaveScattering(yukawa, l_max=3)

    for l in range(4):
        deltas = []
        for k in k_values:
            try:
                delta = scatter_yukawa.compute_phase_shift(k, l)
                deltas.append(np.degrees(delta))
            except Exception:
                deltas.append(np.nan)
        ax3.plot(k_values, deltas, 's-', markersize=3,
                 linewidth=1.5, label=f'$\\ell$ = {l}')

    ax3.set_xlabel('k', fontsize=12)
    ax3.set_ylabel('Phase shift δ (degrees)', fontsize=12)
    ax3.set_title(f'Yukawa Potential (V₀ = {V0}, μ = {mu})', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # S-matrix elements
    ax4 = axes[1, 1]

    # For hard sphere, plot S_l on complex plane
    theta_param = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta_param), np.sin(theta_param), 'k--', alpha=0.5,
             label='Unit circle')

    k = 2.0
    for l in range(4):
        delta = hard_sphere_phase_shift(k, a, l)
        S_l = np.exp(2j * delta)
        ax4.plot(S_l.real, S_l.imag, 'o', markersize=12,
                 label=f'$S_{{{l}}}$ (k={k})')

    ax4.set_xlabel('Re(S)', fontsize=12)
    ax4.set_ylabel('Im(S)', fontsize=12)
    ax4.set_title('S-Matrix Elements (Hard Sphere)', fontsize=12)
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig('phase_shifts_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cross_sections():
    """Plot differential and total cross sections."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Differential cross section for hard sphere
    ax1 = axes[0]
    theta = np.linspace(0.01, np.pi, 200)
    a = 1.0

    for ka in [0.5, 1.0, 2.0, 5.0]:
        k = ka / a

        # Compute phase shifts
        phase_shifts = np.array([hard_sphere_phase_shift(k, a, l)
                                  for l in range(15)])

        # Compute amplitude
        cos_theta = np.cos(theta)
        f = np.zeros_like(theta, dtype=complex)

        for l in range(len(phase_shifts)):
            f_l = np.exp(1j * phase_shifts[l]) * np.sin(phase_shifts[l]) / k
            P_l = legendre_p(l, cos_theta)
            f += (2*l + 1) * f_l * P_l

        dsigma = np.abs(f)**2 / a**2  # Normalize by a²

        ax1.semilogy(np.degrees(theta), dsigma, linewidth=2,
                     label=f'ka = {ka}')

    ax1.set_xlabel('Scattering angle (degrees)', fontsize=12)
    ax1.set_ylabel('dσ/dΩ / a²', fontsize=12)
    ax1.set_title('Hard Sphere Differential Cross Section', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Total cross section vs energy
    ax2 = axes[1]
    ka_values = np.linspace(0.1, 10, 100)

    sigma_numerical = []
    sigma_quantum = []

    for ka in ka_values:
        k = ka / a

        # Compute phase shifts (truncate at l_max ~ ka + 4)
        l_max = int(ka) + 5
        phase_shifts = np.array([hard_sphere_phase_shift(k, a, l)
                                  for l in range(l_max)])

        # Total cross section
        sigma = 4 * np.pi / k**2 * sum((2*l + 1) * np.sin(delta)**2
                                        for l, delta in enumerate(phase_shifts))
        sigma_numerical.append(sigma / (np.pi * a**2))

    ax2.plot(ka_values, sigma_numerical, 'b-', linewidth=2,
             label='Quantum (partial waves)')
    ax2.axhline(y=4, color='r', linestyle='--', linewidth=2,
                label='Low energy limit (4)')
    ax2.axhline(y=2, color='g', linestyle='--', linewidth=2,
                label='High energy limit (2)')

    ax2.set_xlabel('ka', fontsize=12)
    ax2.set_ylabel('σ / πa²', fontsize=12)
    ax2.set_title('Hard Sphere Total Cross Section', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('hard_sphere_cross_section.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_s_wave_dominance():
    """Show s-wave dominance at low energy."""

    print("=" * 60)
    print("S-WAVE DOMINANCE AT LOW ENERGY")
    print("=" * 60)

    a = 1.0
    energies = [0.1, 0.5, 1.0, 2.0, 5.0]

    for ka in energies:
        k = ka / a
        print(f"\nka = {ka}:")
        print("-" * 40)

        sigma_partial = []
        for l in range(6):
            delta = hard_sphere_phase_shift(k, a, l)
            sigma_l = 4 * np.pi / k**2 * (2*l + 1) * np.sin(delta)**2
            sigma_partial.append(sigma_l / (np.pi * a**2))
            print(f"  ℓ = {l}: δ = {np.degrees(delta):8.3f}°, "
                  f"σ_ℓ/πa² = {sigma_partial[-1]:.4f}")

        sigma_total = sum(sigma_partial)
        s_wave_fraction = sigma_partial[0] / sigma_total * 100
        print(f"  Total: σ/πa² = {sigma_total:.4f}, "
              f"s-wave fraction = {s_wave_fraction:.1f}%")


def plot_angular_distribution_decomposition():
    """Show how partial waves combine to give angular distribution."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    theta = np.linspace(0, np.pi, 200)
    cos_theta = np.cos(theta)
    a = 1.0
    ka = 2.0
    k = ka / a

    # Compute phase shifts
    l_max = 6
    phase_shifts = [hard_sphere_phase_shift(k, a, l) for l in range(l_max)]

    # Individual partial wave contributions
    ax1 = axes[0, 0]
    for l in range(4):
        f_l = np.exp(1j * phase_shifts[l]) * np.sin(phase_shifts[l]) / k
        P_l = legendre_p(l, cos_theta)
        contribution = (2*l + 1)**2 * np.abs(f_l)**2 * P_l**2

        ax1.plot(np.degrees(theta), contribution * k**2, linewidth=2,
                 label=f'$\\ell$ = {l}')

    ax1.set_xlabel('θ (degrees)', fontsize=12)
    ax1.set_ylabel('$(2\\ell+1)^2|f_\\ell|^2 P_\\ell^2(\\cos\\theta) \\times k^2$',
                   fontsize=11)
    ax1.set_title('Individual Partial Wave Contributions', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative sum
    ax2 = axes[0, 1]

    for l_cut in [1, 2, 3, 5]:
        f_total = np.zeros_like(theta, dtype=complex)
        for l in range(l_cut):
            f_l = np.exp(1j * phase_shifts[l]) * np.sin(phase_shifts[l]) / k
            P_l = legendre_p(l, cos_theta)
            f_total += (2*l + 1) * f_l * P_l

        dsigma = np.abs(f_total)**2
        ax2.plot(np.degrees(theta), dsigma * k**2, linewidth=2,
                 label=f'$\\ell \\leq$ {l_cut-1}')

    ax2.set_xlabel('θ (degrees)', fontsize=12)
    ax2.set_ylabel('dσ/dΩ × k²', fontsize=12)
    ax2.set_title('Cumulative Partial Wave Sum', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Interference terms
    ax3 = axes[1, 0]

    f_0 = np.exp(1j * phase_shifts[0]) * np.sin(phase_shifts[0]) / k
    f_1 = np.exp(1j * phase_shifts[1]) * np.sin(phase_shifts[1]) / k
    P_0 = legendre_p(0, cos_theta)
    P_1 = legendre_p(1, cos_theta)

    term_00 = np.abs(f_0)**2
    term_11 = 9 * np.abs(f_1)**2 * P_1**2
    term_01 = 2 * np.real(f_0 * np.conj(3 * f_1 * P_1))

    ax3.plot(np.degrees(theta), term_00 * k**2, 'b-', linewidth=2,
             label='s-wave')
    ax3.plot(np.degrees(theta), term_11 * k**2, 'r-', linewidth=2,
             label='p-wave')
    ax3.plot(np.degrees(theta), term_01 * k**2, 'g--', linewidth=2,
             label='Interference')
    ax3.plot(np.degrees(theta), (term_00 + term_11 + term_01) * k**2,
             'k-', linewidth=2, label='Total (s+p)')

    ax3.set_xlabel('θ (degrees)', fontsize=12)
    ax3.set_ylabel('Contribution × k²', fontsize=12)
    ax3.set_title('s-p Wave Interference', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Final full cross section
    ax4 = axes[1, 1]

    f_total = np.zeros_like(theta, dtype=complex)
    for l in range(l_max):
        f_l = np.exp(1j * phase_shifts[l]) * np.sin(phase_shifts[l]) / k
        P_l = legendre_p(l, cos_theta)
        f_total += (2*l + 1) * f_l * P_l

    dsigma = np.abs(f_total)**2

    ax4.plot(np.degrees(theta), dsigma / a**2, 'b-', linewidth=2)
    ax4.fill_between(np.degrees(theta), 0, dsigma / a**2, alpha=0.3)

    ax4.set_xlabel('θ (degrees)', fontsize=12)
    ax4.set_ylabel('dσ/dΩ / a²', fontsize=12)
    ax4.set_title(f'Full Differential Cross Section (ka = {ka})', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('partial_wave_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Day 500: Partial Wave Analysis")
    print("=" * 60)

    plot_phase_shifts()
    plot_cross_sections()
    demonstrate_s_wave_dominance()
    plot_angular_distribution_decomposition()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Partial wave | Scattering component with angular momentum ℓ |
| Phase shift δℓ | Phase accumulated relative to free propagation |
| S-matrix | $S_\ell = e^{2i\delta_\ell}$, relates in/out waves |
| Unitarity | $\|S_\ell\| = 1$ from probability conservation |
| Unitarity limit | Max cross section when $\delta_\ell = \pi/2$ |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$f(\theta) = \sum_\ell (2\ell+1)f_\ell P_\ell(\cos\theta)$$ | Partial wave expansion |
| $$f_\ell = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$$ | Partial wave amplitude |
| $$\sigma_{tot} = \frac{4\pi}{k^2}\sum_\ell (2\ell+1)\sin^2\delta_\ell$$ | Total cross section |
| $$S_\ell = e^{2i\delta_\ell}$$ | S-matrix element |

---

## 10. Daily Checklist

### Conceptual Understanding
- [ ] I can expand a plane wave in spherical harmonics
- [ ] I understand the meaning of phase shifts
- [ ] I see how partial waves combine to give the full amplitude
- [ ] I grasp the physical significance of unitarity

### Mathematical Skills
- [ ] I can calculate phase shifts from boundary conditions
- [ ] I can compute cross sections from phase shifts
- [ ] I can verify the unitarity constraint
- [ ] I can identify when specific partial waves dominate

### Computational Skills
- [ ] I solved the radial equation numerically
- [ ] I extracted phase shifts from wave function matching
- [ ] I visualized partial wave contributions
- [ ] I computed S-matrix elements

---

## 11. Preview: Day 501

Tomorrow we study **low-energy scattering**, where quantum effects are most dramatic:

- s-wave dominance at threshold
- Scattering length a
- Effective range expansion
- Connection to bound states
- Universality in ultracold gases

Low-energy scattering reveals the fundamental length scales of quantum interactions and is crucial for understanding ultracold atomic physics.

---

## References

1. Griffiths, D.J. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Ch. 11.5.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 7.5-7.6.

3. Taylor, J.R. (2006). *Scattering Theory*, Ch. 5-6.

4. Landau, L.D. & Lifshitz, E.M. (1977). *Quantum Mechanics*, Ch. 125-127.

---

*"The partial wave expansion transforms the three-dimensional scattering problem into a sequence of one-dimensional problems, each characterized by a single number: the phase shift."*
— John Taylor

---

**Day 500 Complete.** Tomorrow: Low-Energy Scattering.
