# Day 499: Born Approximation

## Overview

**Day 499 of 2520 | Week 72, Day 2 | Month 18: Identical Particles & Many-Body Physics**

Today we develop the Born approximation—a perturbative approach to calculating scattering amplitudes when the potential is weak. This method reveals a beautiful connection: the scattering amplitude is essentially the Fourier transform of the potential. The Born approximation is one of the most widely used tools in scattering theory, applicable to atomic collisions, nuclear physics, and even quantum electrodynamics.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Integral Equation for Scattering | 60 min |
| 10:00 AM | First Born Approximation | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Born Amplitude as Fourier Transform | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Applications: Coulomb & Yukawa | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Validity & Higher Orders | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Derive** the Lippmann-Schwinger equation for scattering
2. **Apply** the first Born approximation to calculate scattering amplitudes
3. **Recognize** the amplitude as a Fourier transform of the potential
4. **Calculate** Born cross sections for Coulomb and Yukawa potentials
5. **Assess** the validity conditions for the Born approximation
6. **Extend** to second Born approximation conceptually

---

## 1. The Lippmann-Schwinger Equation

### From Schrödinger to Integral Equation

The time-independent Schrödinger equation:

$$(\nabla^2 + k^2)\psi = \frac{2m}{\hbar^2}V(\mathbf{r})\psi \equiv U(\mathbf{r})\psi$$

where $U = 2mV/\hbar^2$.

This is an inhomogeneous Helmholtz equation. Using Green's function methods:

$$\psi(\mathbf{r}) = \phi(\mathbf{r}) + \int G_0(\mathbf{r}, \mathbf{r}')U(\mathbf{r}')\psi(\mathbf{r}')d^3r'$$

where $\phi$ is a solution of the homogeneous equation and $G_0$ is the Green's function.

### The Outgoing Green's Function

For outgoing boundary conditions (scattered waves moving outward):

$$G_0^{(+)}(\mathbf{r}, \mathbf{r}') = -\frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{4\pi|\mathbf{r} - \mathbf{r}'|}$$

This satisfies:

$$(\nabla^2 + k^2)G_0^{(+)} = \delta^3(\mathbf{r} - \mathbf{r}')$$

### The Lippmann-Schwinger Equation

$$\boxed{\psi^{(+)}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} - \frac{m}{2\pi\hbar^2}\int \frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|}V(\mathbf{r}')\psi^{(+)}(\mathbf{r}')d^3r'}$$

**Key features:**
- Incident plane wave: $e^{i\mathbf{k}\cdot\mathbf{r}}$
- Integral over scattered contributions
- Self-consistent: $\psi$ appears on both sides

### Operator Form

In abstract notation:

$$|\psi^{(+)}\rangle = |\phi\rangle + G_0^{(+)}V|\psi^{(+)}\rangle$$

where $G_0^{(+)} = (E - H_0 + i\epsilon)^{-1}$.

---

## 2. The First Born Approximation

### The Perturbative Expansion

The Lippmann-Schwinger equation can be solved iteratively:

$$\psi = \phi + G_0 V \phi + G_0 V G_0 V \phi + G_0 V G_0 V G_0 V \phi + \cdots$$

This is the **Born series**.

### First Born Approximation

Replace $\psi$ inside the integral with the incident wave $\phi = e^{i\mathbf{k}\cdot\mathbf{r}}$:

$$\psi^{(1)}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} - \frac{m}{2\pi\hbar^2}\int \frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|}V(\mathbf{r}')e^{i\mathbf{k}\cdot\mathbf{r}'}d^3r'$$

**Physical meaning:** The potential scatters the incident wave once; multiple scattering is neglected.

### Extracting the Scattering Amplitude

For large $r$ (far from the scatterer), with $\mathbf{r}' \ll r$:

$$|\mathbf{r} - \mathbf{r}'| \approx r - \hat{r}\cdot\mathbf{r}'$$

$$\frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|} \approx \frac{e^{ikr}}{r}e^{-i\mathbf{k}'\cdot\mathbf{r}'}$$

where $\mathbf{k}' = k\hat{r}$ is the outgoing wave vector.

### The Born Scattering Amplitude

$$\boxed{f^{(1)}(\theta, \phi) = -\frac{m}{2\pi\hbar^2}\int V(\mathbf{r}')e^{i(\mathbf{k} - \mathbf{k}')\cdot\mathbf{r}'}d^3r'}$$

Defining the momentum transfer $\mathbf{q} = \mathbf{k} - \mathbf{k}'$:

$$\boxed{f^{(1)}(\mathbf{q}) = -\frac{m}{2\pi\hbar^2}\int V(\mathbf{r})e^{i\mathbf{q}\cdot\mathbf{r}}d^3r = -\frac{m}{2\pi\hbar^2}\tilde{V}(\mathbf{q})}$$

**The Born amplitude is (proportional to) the Fourier transform of the potential!**

---

## 3. Fourier Transform Interpretation

### The Scattering-Fourier Connection

$$f^{(1)}(\mathbf{q}) = -\frac{m}{2\pi\hbar^2}\tilde{V}(\mathbf{q})$$

where $\tilde{V}(\mathbf{q}) = \int V(\mathbf{r})e^{i\mathbf{q}\cdot\mathbf{r}}d^3r$.

### Physical Insight

- **Large momentum transfer** (large angle): probes short-range structure
- **Small momentum transfer** (forward): probes long-range behavior
- **Sharp potential features** → broad angular distribution
- **Extended potential** → forward-peaked scattering

### Momentum Transfer Magnitude

For elastic scattering ($|\mathbf{k}'| = |\mathbf{k}| = k$):

$$q = |\mathbf{q}| = |\mathbf{k} - \mathbf{k}'| = 2k\sin\frac{\theta}{2}$$

| Angle | Momentum Transfer |
|-------|-------------------|
| $\theta = 0°$ | $q = 0$ |
| $\theta = 90°$ | $q = \sqrt{2}k$ |
| $\theta = 180°$ | $q = 2k$ |

### Central Potentials

For spherically symmetric $V(r)$, the Fourier transform depends only on $q$:

$$\tilde{V}(q) = \int_0^\infty 4\pi r^2 V(r)\frac{\sin(qr)}{qr}dr$$

$$\boxed{\tilde{V}(q) = \frac{4\pi}{q}\int_0^\infty rV(r)\sin(qr)dr}$$

---

## 4. Applications

### Coulomb Potential (Rutherford Scattering)

$$V(r) = \frac{Z_1Z_2 e^2}{r}$$

**Problem:** The Fourier transform diverges! Use screened Coulomb:

$$V(r) = \frac{Z_1Z_2 e^2}{r}e^{-\mu r}$$

then take $\mu \to 0$.

$$\tilde{V}(q) = \frac{4\pi Z_1Z_2 e^2}{q^2 + \mu^2}$$

Taking $\mu \to 0$:

$$f^{(1)}(\theta) = -\frac{2mZ_1Z_2 e^2}{\hbar^2 q^2} = -\frac{2mZ_1Z_2 e^2}{\hbar^2 \cdot 4k^2\sin^2(\theta/2)}$$

$$\boxed{f_{Coulomb}(\theta) = -\frac{\eta}{2k\sin^2(\theta/2)}}$$

where $\eta = mZ_1Z_2e^2/\hbar^2 k$ is the Sommerfeld parameter.

### Rutherford Cross Section

$$\boxed{\frac{d\sigma}{d\Omega} = \left(\frac{Z_1Z_2 e^2}{4E}\right)^2\frac{1}{\sin^4(\theta/2)}}$$

where $E = \hbar^2k^2/2m$.

**Remarkable:** This quantum result agrees exactly with the classical Rutherford formula!

### Yukawa Potential

$$V(r) = V_0\frac{e^{-\mu r}}{r}$$

where $\mu = m_\pi c/\hbar$ for nuclear forces (pion range).

Fourier transform:

$$\tilde{V}(q) = \frac{4\pi V_0}{q^2 + \mu^2}$$

Born amplitude:

$$f^{(1)}(\theta) = -\frac{2mV_0}{\hbar^2(q^2 + \mu^2)} = -\frac{2mV_0}{\hbar^2(4k^2\sin^2(\theta/2) + \mu^2)}$$

Cross section:

$$\boxed{\frac{d\sigma}{d\Omega} = \left(\frac{2mV_0}{\hbar^2}\right)^2\frac{1}{(4k^2\sin^2(\theta/2) + \mu^2)^2}}$$

**Features:**
- Finite at all angles (unlike Coulomb)
- Forward-peaked for $k \gg \mu$ (high energy)
- Approaches Rutherford for $\mu \to 0$

### Square Well Potential

$$V(r) = \begin{cases} -V_0 & r < a \\ 0 & r > a \end{cases}$$

Fourier transform:

$$\tilde{V}(q) = -\frac{4\pi V_0}{q^3}(\sin(qa) - qa\cos(qa))$$

Born amplitude:

$$f^{(1)}(\theta) = \frac{2mV_0 a^3}{\hbar^2}\frac{3(\sin(qa) - qa\cos(qa))}{(qa)^3}$$

This is the form factor for a uniform sphere, showing diffraction minima.

---

## 5. Validity of Born Approximation

### When is Born Valid?

The Born approximation assumes the wave function inside the scattering region is approximately the incident plane wave. This requires:

**Condition 1: Weak scattering**

$$|f^{(1)}| \ll \text{range of potential}$$

**Condition 2: Born criterion**

$$\left|\frac{m}{\hbar^2}\int V(\mathbf{r}')\frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|}e^{i\mathbf{k}\cdot\mathbf{r}'}d^3r'\right| \ll 1$$

### Approximate Criteria

For a potential of depth $V_0$ and range $a$:

**Low energy** ($ka \ll 1$):
$$\boxed{\frac{mV_0 a^2}{\hbar^2} \ll 1}$$

**High energy** ($ka \gg 1$):
$$\boxed{\frac{mV_0 a}{\hbar^2 k} \ll 1}$$

### Physical Interpretation

- **Low energy:** Particle spends more time in potential; need $V_0$ small
- **High energy:** Particle moves quickly through; Born improves
- **Generally valid for:** fast particles, weak potentials

### Example Assessment

Consider $V_0 = 10$ MeV, $a = 2$ fm for nucleon scattering:

$$\frac{mV_0 a^2}{\hbar^2} = \frac{(939\text{ MeV}/c^2)(10\text{ MeV})(2\text{ fm})^2}{(197\text{ MeV·fm})^2} \approx 1$$

Born is marginal at low energies but improves at higher energies.

---

## 6. Worked Examples

### Example 1: Gaussian Potential

**Problem:** Calculate the Born differential cross section for:
$$V(r) = V_0 e^{-r^2/a^2}$$

**Solution:**

Fourier transform of Gaussian:
$$\tilde{V}(q) = \int V_0 e^{-r^2/a^2} e^{i\mathbf{q}\cdot\mathbf{r}} d^3r$$

Using $\int e^{-\alpha r^2} e^{i\mathbf{q}\cdot\mathbf{r}} d^3r = \left(\frac{\pi}{\alpha}\right)^{3/2}e^{-q^2/4\alpha}$:

$$\tilde{V}(q) = V_0 (\pi a^2)^{3/2} e^{-q^2 a^2/4}$$

Born amplitude:
$$f^{(1)}(\theta) = -\frac{m V_0 (\pi a^2)^{3/2}}{2\pi\hbar^2} e^{-q^2 a^2/4}$$

With $q = 2k\sin(\theta/2)$:

$$\boxed{\frac{d\sigma}{d\Omega} = \left(\frac{m V_0 \pi^{1/2} a^3}{2\hbar^2}\right)^2 e^{-k^2 a^2 \sin^2(\theta/2)}}$$

**Key feature:** Gaussian falloff with angle; width decreases as $ka$ increases.

### Example 2: Yukawa Total Cross Section

**Problem:** Calculate the total Born cross section for the Yukawa potential.

**Solution:**

$$\frac{d\sigma}{d\Omega} = \left(\frac{2mV_0}{\hbar^2}\right)^2\frac{1}{(q^2 + \mu^2)^2}$$

With $q^2 = 4k^2\sin^2(\theta/2) = 2k^2(1 - \cos\theta)$:

$$\sigma_{tot} = 2\pi \int_0^\pi \frac{d\sigma}{d\Omega}\sin\theta \, d\theta$$

Let $u = 1 - \cos\theta$, so $du = \sin\theta \, d\theta$:

$$\sigma_{tot} = 2\pi\left(\frac{2mV_0}{\hbar^2}\right)^2 \int_0^2 \frac{du}{(2k^2 u + \mu^2)^2}$$

$$= 2\pi\left(\frac{2mV_0}{\hbar^2}\right)^2 \left[-\frac{1}{2k^2(2k^2 u + \mu^2)}\right]_0^2$$

$$\boxed{\sigma_{tot} = \frac{16\pi m^2 V_0^2}{\hbar^4 \mu^2(4k^2 + \mu^2)}}$$

**Limits:**
- $k \to 0$: $\sigma \to 16\pi m^2 V_0^2/(\hbar^4\mu^4)$ (finite)
- $k \to \infty$: $\sigma \to 4\pi m^2 V_0^2/(\hbar^4 k^4)$ (decreases as $1/k^4$)

### Example 3: Form Factor Analysis

**Problem:** A nucleus can be modeled as a uniform sphere of charge density. Show how Born scattering reveals the nuclear radius.

**Solution:**

For uniform density out to radius $R$:
$$V(r) = V_0 \cdot \begin{cases} 1 & r < R \\ 0 & r \geq R \end{cases}$$

The Fourier transform gives:
$$\tilde{V}(q) = V_0 \cdot \frac{4\pi}{q^3}(\sin(qR) - qR\cos(qR))$$

Define the **form factor**:
$$F(q) = \frac{3(\sin(qR) - qR\cos(qR))}{(qR)^3}$$

The cross section:
$$\frac{d\sigma}{d\Omega} = \left(\frac{d\sigma}{d\Omega}\right)_{point} |F(q)|^2$$

**Diffraction minima** occur when $F(q) = 0$:
$$\tan(qR) = qR$$

First minimum: $qR \approx 4.49$, so $q_1 = 4.49/R$.

$$\boxed{\text{Nuclear radius: } R \approx \frac{4.49}{q_1} = \frac{4.49}{2k\sin(\theta_1/2)}}$$

This is how electron scattering determines nuclear sizes!

---

## 7. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the Born scattering amplitude for a delta-function potential $V(\mathbf{r}) = V_0\delta^3(\mathbf{r})$.

**Problem 1.2:** For Rutherford scattering of 5 MeV alpha particles from gold (Z=79), calculate:
(a) $d\sigma/d\Omega$ at $\theta = 30°$
(b) The ratio of cross sections at $90°$ vs $30°$

**Problem 1.3:** A Yukawa potential has $V_0 = -50$ MeV and range $1/\mu = 1.4$ fm. At what energy does the high-energy Born criterion become satisfied?

### Level 2: Intermediate

**Problem 2.1:** For the exponential potential $V(r) = V_0 e^{-r/a}$:
(a) Calculate the Fourier transform
(b) Find the Born differential cross section
(c) Determine the total cross section

**Problem 2.2:** Show that for any central potential with finite range, the forward scattering amplitude in Born approximation is:
$$f(0) = -\frac{m}{2\pi\hbar^2}\int V(r)d^3r$$

**Problem 2.3:** The Coulomb potential gives the exact Rutherford cross section even though Born is not formally valid. Explain this coincidence using the fact that the exact Coulomb wave function is known analytically.

### Level 3: Challenging

**Problem 3.1:** Derive the second Born approximation formula:
$$f^{(2)} = f^{(1)} + \frac{m^2}{4\pi^2\hbar^4}\int\int \frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{|\mathbf{r} - \mathbf{r}'|}V(\mathbf{r})e^{i\mathbf{k}\cdot\mathbf{r}'}V(\mathbf{r}')e^{i\mathbf{k}\cdot\mathbf{r}''}d^3r'd^3r''$$

**Problem 3.2:** For identical bosons scattering via a central potential, the amplitude includes exchange:
$$f_{sym}(\theta) = f(\theta) + f(\pi - \theta)$$
Calculate the Born cross section for two bosons interacting via a Gaussian potential at $\theta = 90°$.

**Problem 3.3:** Prove that if $V(r)$ has a bound state, the Born approximation fails at low energies regardless of how weak the potential is. (Hint: consider the scattering length.)

---

## 8. Computational Lab: Born Approximation Calculations

```python
"""
Day 499 Computational Lab: Born Approximation
Calculating and visualizing Born scattering for various potentials.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import spherical_jn

# Physical constants (natural units: hbar = m = 1)
hbar = 1.0
m = 1.0


def momentum_transfer(k, theta):
    """Calculate momentum transfer q = 2k*sin(theta/2)."""
    return 2 * k * np.sin(theta / 2)


def born_amplitude_numerical(V_func, q, r_max=50, num_points=1000):
    """
    Calculate Born amplitude by numerical integration.

    f(q) = -(m/2π) * (4π/q) ∫₀^∞ r*V(r)*sin(qr) dr
    """
    def integrand(r):
        if q < 1e-10:
            return r * r * V_func(r)  # q→0 limit: ∫r²V(r)dr
        return r * V_func(r) * np.sin(q * r)

    integral, _ = quad(integrand, 0, r_max, limit=200)

    if q < 1e-10:
        # Forward scattering: f(0) = -(m/2π) * 4π * ∫r²V(r)dr
        return -m / (2 * np.pi * hbar**2) * 4 * np.pi * integral
    else:
        return -m / (2 * np.pi * hbar**2) * 4 * np.pi / q * integral


def cross_section_born(V_func, k, theta, r_max=50):
    """Calculate Born differential cross section."""
    q = momentum_transfer(k, theta)
    f = born_amplitude_numerical(V_func, q, r_max)
    return np.abs(f)**2


class Potential:
    """Collection of common potentials."""

    @staticmethod
    def yukawa(r, V0, mu):
        """Yukawa potential: V0 * exp(-mu*r) / r"""
        if r < 1e-10:
            return V0 * mu  # Limit as r→0
        return V0 * np.exp(-mu * r) / r

    @staticmethod
    def gaussian(r, V0, a):
        """Gaussian potential: V0 * exp(-r²/a²)"""
        return V0 * np.exp(-(r/a)**2)

    @staticmethod
    def square_well(r, V0, a):
        """Square well: -V0 for r < a, 0 otherwise"""
        return -V0 if r < a else 0

    @staticmethod
    def exponential(r, V0, a):
        """Exponential potential: V0 * exp(-r/a)"""
        return V0 * np.exp(-r / a)


def analytical_born_yukawa(q, V0, mu):
    """Analytical Born amplitude for Yukawa potential."""
    # f = -(m/2π) * 4πV0/(q² + μ²)
    return -m * V0 * 2 / (hbar**2 * (q**2 + mu**2))


def analytical_born_gaussian(q, V0, a):
    """Analytical Born amplitude for Gaussian potential."""
    # f = -(m/2π) * V0 * (πa²)^(3/2) * exp(-q²a²/4)
    return -m * V0 * (np.pi * a**2)**1.5 * np.exp(-q**2 * a**2 / 4) / (2 * np.pi * hbar**2)


def plot_born_cross_sections():
    """Compare Born cross sections for different potentials."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    k = 2.0  # Wave number
    theta = np.linspace(0.01, np.pi, 100)
    q = momentum_transfer(k, theta)

    # Parameters
    V0 = 1.0
    a = 1.0
    mu = 1.0

    # Yukawa
    ax1 = axes[0, 0]
    f_yukawa = np.array([analytical_born_yukawa(qi, V0, mu) for qi in q])
    dsigma_yukawa = np.abs(f_yukawa)**2

    ax1.semilogy(np.degrees(theta), dsigma_yukawa, 'b-', linewidth=2,
                 label='Yukawa')
    ax1.set_xlabel('Scattering angle (degrees)', fontsize=11)
    ax1.set_ylabel('dσ/dΩ', fontsize=11)
    ax1.set_title('Yukawa Potential', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gaussian
    ax2 = axes[0, 1]
    f_gaussian = np.array([analytical_born_gaussian(qi, V0, a) for qi in q])
    dsigma_gaussian = np.abs(f_gaussian)**2

    ax2.semilogy(np.degrees(theta), dsigma_gaussian, 'r-', linewidth=2,
                 label='Gaussian')
    ax2.set_xlabel('Scattering angle (degrees)', fontsize=11)
    ax2.set_ylabel('dσ/dΩ', fontsize=11)
    ax2.set_title('Gaussian Potential', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Square well (numerical)
    ax3 = axes[1, 0]
    V_well = lambda r: Potential.square_well(r, V0, a)
    dsigma_well = np.array([cross_section_born(V_well, k, t) for t in theta])

    ax3.plot(np.degrees(theta), dsigma_well, 'g-', linewidth=2,
             label='Square well')
    ax3.set_xlabel('Scattering angle (degrees)', fontsize=11)
    ax3.set_ylabel('dσ/dΩ', fontsize=11)
    ax3.set_title('Square Well Potential (Diffraction Pattern)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Comparison at different energies
    ax4 = axes[1, 1]
    energies = [0.5, 1.0, 2.0, 5.0]

    for E in energies:
        k_E = np.sqrt(2 * m * E) / hbar
        theta_plot = np.linspace(0.01, np.pi, 100)
        q_E = momentum_transfer(k_E, theta_plot)
        f_E = np.array([analytical_born_yukawa(qi, V0, mu) for qi in q_E])
        dsigma_E = np.abs(f_E)**2

        ax4.semilogy(np.degrees(theta_plot), dsigma_E,
                     label=f'E = {E}', linewidth=2)

    ax4.set_xlabel('Scattering angle (degrees)', fontsize=11)
    ax4.set_ylabel('dσ/dΩ', fontsize=11)
    ax4.set_title('Yukawa: Energy Dependence', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('born_cross_sections.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_form_factors():
    """Analyze form factors for different charge distributions."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    qR = np.linspace(0.01, 15, 300)

    # Form factors
    def F_uniform(qR):
        """Uniform sphere form factor."""
        return 3 * (np.sin(qR) - qR * np.cos(qR)) / qR**3

    def F_gaussian(qR):
        """Gaussian form factor."""
        return np.exp(-qR**2 / 6)  # RMS radius = sqrt(3/2) * R

    def F_exponential(qR):
        """Exponential charge distribution."""
        return 1 / (1 + qR**2 / 12)**2

    # Plot form factors
    ax1 = axes[0]
    ax1.plot(qR, np.abs(F_uniform(qR))**2, 'b-', linewidth=2, label='Uniform sphere')
    ax1.plot(qR, np.abs(F_gaussian(qR))**2, 'r-', linewidth=2, label='Gaussian')
    ax1.plot(qR, np.abs(F_exponential(qR))**2, 'g-', linewidth=2, label='Exponential')

    ax1.set_xlabel('qR (dimensionless)', fontsize=12)
    ax1.set_ylabel('|F(qR)|²', fontsize=12)
    ax1.set_title('Nuclear Form Factors', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-6, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Electron scattering cross section
    ax2 = axes[1]
    theta = np.linspace(0.1, 3.0, 100)  # radians
    k = 10  # High energy
    R = 1  # Nuclear radius

    q_vals = 2 * k * np.sin(theta / 2)
    qR_vals = q_vals * R

    # Point nucleus (Rutherford-like)
    dsigma_point = 1 / np.sin(theta/2)**4

    # With form factor
    dsigma_uniform = dsigma_point * np.abs(F_uniform(qR_vals))**2

    ax2.semilogy(np.degrees(theta), dsigma_point / dsigma_point[0],
                 'k--', linewidth=2, label='Point nucleus')
    ax2.semilogy(np.degrees(theta), dsigma_uniform / dsigma_point[0],
                 'b-', linewidth=2, label='Uniform sphere')

    ax2.set_xlabel('Scattering angle (degrees)', fontsize=12)
    ax2.set_ylabel('dσ/dΩ (normalized)', fontsize=12)
    ax2.set_title('Electron Scattering: Point vs Extended Nucleus', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('form_factors.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_numerical_analytical():
    """Compare numerical integration with analytical results."""

    print("=" * 60)
    print("NUMERICAL vs ANALYTICAL BORN AMPLITUDES")
    print("=" * 60)

    V0 = 1.0
    mu = 1.0
    a = 1.0

    q_values = [0.5, 1.0, 2.0, 5.0]

    print("\nYukawa Potential:")
    print("-" * 40)
    V_yukawa = lambda r: Potential.yukawa(r, V0, mu)

    for q in q_values:
        f_num = born_amplitude_numerical(V_yukawa, q, r_max=30)
        f_ana = analytical_born_yukawa(q, V0, mu)
        error = abs(f_num - f_ana) / abs(f_ana) * 100
        print(f"q = {q}: Numerical = {f_num:.6f}, "
              f"Analytical = {f_ana:.6f}, Error = {error:.2f}%")

    print("\nGaussian Potential:")
    print("-" * 40)
    V_gauss = lambda r: Potential.gaussian(r, V0, a)

    for q in q_values:
        f_num = born_amplitude_numerical(V_gauss, q)
        f_ana = analytical_born_gaussian(q, V0, a)
        error = abs(f_num - f_ana) / abs(f_ana) * 100 if abs(f_ana) > 1e-10 else 0
        print(f"q = {q}: Numerical = {f_num:.6f}, "
              f"Analytical = {f_ana:.6f}, Error = {error:.2f}%")


def total_cross_section_calculation():
    """Calculate total cross sections."""

    print("\n" + "=" * 60)
    print("TOTAL CROSS SECTIONS (Born Approximation)")
    print("=" * 60)

    V0 = 1.0
    mu = 1.0
    a = 1.0

    k_values = [0.5, 1.0, 2.0, 5.0]

    print("\nYukawa Potential:")
    print("-" * 40)

    for k in k_values:
        # Analytical
        sigma_ana = 16 * np.pi * m**2 * V0**2 / (hbar**4 * mu**2 * (4*k**2 + mu**2))

        # Numerical integration
        def integrand(theta):
            q = momentum_transfer(k, theta)
            f = analytical_born_yukawa(q, V0, mu)
            return 2 * np.pi * np.abs(f)**2 * np.sin(theta)

        sigma_num, _ = quad(integrand, 0, np.pi)

        print(f"k = {k}: σ_ana = {sigma_ana:.6f}, σ_num = {sigma_num:.6f}")


def visualize_momentum_transfer():
    """Visualize the relationship between angle and momentum transfer."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scattering geometry
    ax1 = axes[0]

    # Draw incident and scattered vectors
    k = 1.0
    angles = [30, 60, 90, 120]

    ax1.arrow(0, 0, 0, k, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    ax1.text(0.1, k/2, r"$\mathbf{k}_i$", fontsize=12, color='blue')

    for angle in angles:
        theta_rad = np.radians(angle)
        kf_x = k * np.sin(theta_rad)
        kf_y = k * np.cos(theta_rad)

        ax1.arrow(0, 0, kf_x, kf_y, head_width=0.05, head_length=0.05,
                  fc='red', ec='red', alpha=0.7)

        # Draw q vector
        q_x = -kf_x
        q_y = k - kf_y
        ax1.arrow(kf_x, kf_y, q_x, q_y, head_width=0.03, head_length=0.03,
                  fc='green', ec='green', linestyle='--')

        ax1.text(kf_x + 0.1, kf_y, f'{angle}°', fontsize=10)

    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('$k_x$', fontsize=12)
    ax1.set_ylabel('$k_y$', fontsize=12)
    ax1.set_title('Scattering Geometry\n(blue: incident, red: scattered, green: momentum transfer)',
                  fontsize=11)
    ax1.grid(True, alpha=0.3)

    # q vs theta
    ax2 = axes[1]
    theta = np.linspace(0, np.pi, 100)

    for k in [1, 2, 5, 10]:
        q = 2 * k * np.sin(theta / 2)
        ax2.plot(np.degrees(theta), q, label=f'k = {k}', linewidth=2)

    ax2.set_xlabel('Scattering angle θ (degrees)', fontsize=12)
    ax2.set_ylabel('Momentum transfer q', fontsize=12)
    ax2.set_title('q = 2k sin(θ/2)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('momentum_transfer_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()


def rutherford_analysis():
    """Detailed analysis of Rutherford scattering."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rutherford cross section
    ax1 = axes[0]
    theta = np.linspace(1, 179, 200)  # Avoid 0 and 180
    theta_rad = np.radians(theta)

    # Rutherford formula (normalized)
    dsigma = 1 / np.sin(theta_rad / 2)**4

    ax1.semilogy(theta, dsigma / dsigma[len(dsigma)//2], 'b-', linewidth=2)
    ax1.set_xlabel('Scattering angle (degrees)', fontsize=12)
    ax1.set_ylabel('dσ/dΩ (normalized to 90°)', fontsize=12)
    ax1.set_title('Rutherford Cross Section: dσ/dΩ ∝ 1/sin⁴(θ/2)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 180)

    # Historical: Geiger-Marsden experiment
    ax2 = axes[1]

    # Simulated data (based on actual experiment)
    theta_exp = np.array([15, 22.5, 30, 37.5, 45, 60, 75, 105, 120, 135, 150])
    counts_theory = 1 / np.sin(np.radians(theta_exp)/2)**4
    counts_theory = counts_theory / counts_theory[-1]  # Normalize

    # Add noise
    np.random.seed(42)
    counts_exp = counts_theory * (1 + 0.1 * np.random.randn(len(theta_exp)))

    ax2.semilogy(theta_exp, counts_exp, 'ro', markersize=8, label='Simulated data')
    theta_theory = np.linspace(10, 160, 100)
    theory_curve = 1 / np.sin(np.radians(theta_theory)/2)**4
    theory_curve = theory_curve / theory_curve[-1]
    ax2.semilogy(theta_theory, theory_curve, 'b-', linewidth=2, label='Rutherford formula')

    ax2.set_xlabel('Scattering angle (degrees)', fontsize=12)
    ax2.set_ylabel('Relative counts', fontsize=12)
    ax2.set_title('Geiger-Marsden Experiment (Simulated)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rutherford_scattering.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Day 499: Born Approximation")
    print("=" * 60)

    plot_born_cross_sections()
    analyze_form_factors()
    compare_numerical_analytical()
    total_cross_section_calculation()
    visualize_momentum_transfer()
    rutherford_analysis()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Lippmann-Schwinger | Integral equation for scattering state |
| Born approximation | First-order perturbative scattering |
| Fourier connection | $f(\mathbf{q}) \propto \tilde{V}(\mathbf{q})$ |
| Momentum transfer | $\mathbf{q} = \mathbf{k} - \mathbf{k}'$, $q = 2k\sin(\theta/2)$ |
| Form factor | Modification due to extended target |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$f^{(1)} = -\frac{m}{2\pi\hbar^2}\tilde{V}(\mathbf{q})$$ | Born amplitude |
| $$\tilde{V}(q) = \frac{4\pi}{q}\int_0^\infty rV(r)\sin(qr)dr$$ | Fourier transform (central) |
| $$\frac{d\sigma}{d\Omega}_{Coul} = \left(\frac{Ze^2}{4E}\right)^2\csc^4\frac{\theta}{2}$$ | Rutherford formula |
| $$\frac{d\sigma}{d\Omega}_{Yuk} \propto \frac{1}{(q^2 + \mu^2)^2}$$ | Yukawa cross section |

---

## 10. Daily Checklist

### Conceptual Understanding
- [ ] I can derive the Lippmann-Schwinger equation
- [ ] I understand the Born approximation as first-order perturbation
- [ ] I see how scattering probes the Fourier transform of potential
- [ ] I know when Born approximation is valid

### Mathematical Skills
- [ ] I can calculate Born amplitudes for various potentials
- [ ] I can derive cross sections from amplitudes
- [ ] I can assess validity conditions
- [ ] I can compute total cross sections

### Computational Skills
- [ ] I implemented numerical Born calculations
- [ ] I compared analytical and numerical results
- [ ] I visualized form factors
- [ ] I analyzed Rutherford scattering

---

## 11. Preview: Day 500

Tomorrow we develop **partial wave analysis**—expanding the scattering amplitude in angular momentum components:

- Spherical wave expansion
- Phase shifts δₗ
- Partial wave amplitudes fₗ
- The S-matrix
- Optical theorem preview

Partial wave analysis is essential for understanding resonances, low-energy scattering, and the deep connection between scattering and bound states.

---

## References

1. Griffiths, D.J. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Ch. 11.3-11.4.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 7.2-7.3.

3. Taylor, J.R. (2006). *Scattering Theory*, Ch. 10-11.

4. Hofstadter, R. (1956). *Electron Scattering and Nuclear Structure*, Rev. Mod. Phys.

---

*"The Born approximation transforms the scattering problem into a Fourier analysis problem—nature reveals her structure through the diffraction of matter waves."*
— Max Born

---

**Day 499 Complete.** Tomorrow: Partial Wave Analysis.
