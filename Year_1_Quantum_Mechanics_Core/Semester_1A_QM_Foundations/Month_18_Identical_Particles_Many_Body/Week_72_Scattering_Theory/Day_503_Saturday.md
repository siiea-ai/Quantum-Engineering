# Day 503: Optical Theorem

## Overview

**Day 503 of 2520 | Week 72, Day 6 | Month 18: Identical Particles & Many-Body Physics**

Today we derive and explore the optical theorem—one of the most elegant and powerful results in scattering theory. The optical theorem relates the imaginary part of the forward scattering amplitude to the total cross section. This connection arises from unitarity (probability conservation) and has deep implications for how we understand scattering as interference between incident and scattered waves.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Unitarity and Probability Conservation | 60 min |
| 10:00 AM | Derivation of the Optical Theorem | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Physical Interpretation | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Applications and Extensions | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Connection to Dispersion Relations | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Derive** the optical theorem from S-matrix unitarity
2. **State** the relationship between Im[f(0)] and σ_tot
3. **Explain** the physical interpretation as shadow scattering
4. **Apply** the optical theorem to verify cross section calculations
5. **Connect** to dispersion relations and causality
6. **Understand** extensions to inelastic and multi-channel scattering

---

## 1. Unitarity: The Foundation

### Probability Conservation

In quantum mechanics, total probability must be conserved:

$$\sum_f |S_{fi}|^2 = 1$$

For scattering, this means: incident flux = transmitted flux + scattered flux

### S-Matrix Unitarity

The S-matrix is unitary:

$$\boxed{S^\dagger S = SS^\dagger = I}$$

This is the mathematical statement of probability conservation.

### Elastic Scattering

For a single channel (elastic scattering):

$$|S_\ell|^2 = 1 \implies |e^{2i\delta_\ell}|^2 = 1$$

This is automatically satisfied for real phase shifts.

### Multi-Channel Unitarity

For coupled channels (e.g., elastic + inelastic):

$$\sum_c |S_{ac}|^2 = 1$$

The elastic S-matrix element satisfies $|S_{aa}| \leq 1$.

---

## 2. Derivation of the Optical Theorem

### From Partial Wave Expansion

The total cross section:

$$\sigma_{tot} = \frac{4\pi}{k^2}\sum_\ell (2\ell + 1)\sin^2\delta_\ell$$

The forward scattering amplitude ($\theta = 0$, $P_\ell(1) = 1$):

$$f(0) = \sum_\ell (2\ell + 1)f_\ell = \sum_\ell (2\ell + 1)\frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$$

### Taking the Imaginary Part

$$\text{Im}[f(0)] = \sum_\ell (2\ell + 1)\frac{\text{Im}[e^{i\delta_\ell}\sin\delta_\ell]}{k}$$

$$= \sum_\ell (2\ell + 1)\frac{\sin\delta_\ell \cdot \sin\delta_\ell}{k} = \frac{1}{k}\sum_\ell (2\ell + 1)\sin^2\delta_\ell$$

### The Optical Theorem

Comparing with σ_tot:

$$\boxed{\sigma_{tot} = \frac{4\pi}{k}\text{Im}[f(0)]}$$

or equivalently:

$$\boxed{\text{Im}[f(0)] = \frac{k\sigma_{tot}}{4\pi}}$$

### Alternative Derivation: Direct from Unitarity

Consider the scattering amplitude definition:

$$S = 1 + 2ikf$$

Unitarity $S^\dagger S = 1$:

$$(1 + 2ik f)^\dagger(1 + 2ikf) = 1$$

$$(1 - 2ik f^*)(1 + 2ikf) = 1$$

$$1 + 2ikf - 2ikf^* + 4k^2|f|^2 = 1$$

$$2ik(f - f^*) + 4k^2|f|^2 = 0$$

$$-4k\text{Im}[f] + 4k^2|f|^2 = 0$$

At $\theta = 0$:
$$\text{Im}[f(0)] = k|f(0)|^2 + \frac{k}{4\pi}\int |f(\theta)|^2 d\Omega$$

For pure elastic scattering, this gives the optical theorem.

---

## 3. Physical Interpretation

### Shadow Scattering

The optical theorem has a beautiful physical interpretation:

**The total cross section equals the "shadow" cast by the scatterer.**

### Wave Interference Picture

Consider a plane wave passing a scatterer:
1. Part of the wave continues forward (transmitted)
2. Part scatters in all directions

Behind the scatterer, there's **destructive interference** between:
- The unscattered wave
- The forward-scattered wave

This creates a "shadow" whose area is σ_tot.

### Mathematical Picture

The intensity behind the scatterer at large distance:

$$I = |e^{ikz} + f(0)\frac{e^{ikr}}{r}|^2$$

The reduction in forward intensity (the shadow) is:

$$\Delta I \propto \text{Re}[e^{ikz} \cdot f^*(0)\frac{e^{-ikr}}{r}] \propto \text{Im}[f(0)]$$

This reduction, integrated over the shadow area, equals σ_tot.

### Optical Origin of the Name

In optics, a similar theorem relates forward scattering to extinction:

$$\text{Im}[n] \propto \text{absorption coefficient}$$

where n is the refractive index (related to forward scattering amplitude).

The quantum mechanical result is analogous, hence "optical theorem."

---

## 4. Applications

### Application 1: Consistency Check

The optical theorem provides a powerful consistency check for any scattering calculation.

**Example:** If you calculate $f(\theta)$ by some method, you can verify:

$$\int |f(\theta)|^2 d\Omega \stackrel{?}{=} \frac{4\pi}{k}\text{Im}[f(0)]$$

Any discrepancy indicates an error in the calculation.

### Application 2: Forward Dispersion Relations

The optical theorem connects to dispersion relations:

$$\text{Re}[f(0, E)] = \frac{2E}{\pi}\mathcal{P}\int_0^\infty \frac{\text{Im}[f(0, E')]}{E'^2 - E^2}dE'$$

Using the optical theorem:

$$\text{Re}[f(0, E)] = \frac{E}{2\pi^2}\mathcal{P}\int_0^\infty \frac{k'\sigma_{tot}(E')}{E'^2 - E^2}dE'$$

This relates the real part of forward scattering to total cross sections at all energies.

### Application 3: High-Energy Scattering

At high energies, for diffractive scattering:

$$f(0) \approx ik\sigma_{tot}/4\pi$$

The forward amplitude becomes purely imaginary, and:

$$\frac{d\sigma}{d\Omega}\bigg|_{\theta=0} = |f(0)|^2 = \frac{k^2\sigma_{tot}^2}{16\pi^2}$$

### Application 4: Total Cross Section from Forward Scattering

Experimentally, the optical theorem allows:

1. Measure the forward scattering amplitude (magnitude and phase)
2. Extract Im[f(0)]
3. Deduce σ_tot without counting all scattered particles

This is valuable in high-energy physics where total cross sections are hard to measure directly.

---

## 5. Extensions

### Inelastic Scattering

When inelastic channels are present, the elastic S-matrix satisfies:

$$|S_\ell^{el}| = \eta_\ell \leq 1$$

where $\eta_\ell$ is the **inelasticity parameter**.

The optical theorem generalizes to:

$$\sigma_{tot} = \sigma_{el} + \sigma_{inel} = \frac{4\pi}{k}\text{Im}[f^{el}(0)]$$

### Partial Cross Sections

$$\sigma_{el} = \frac{\pi}{k^2}\sum_\ell (2\ell+1)|1 - S_\ell|^2$$

$$\sigma_{inel} = \frac{\pi}{k^2}\sum_\ell (2\ell+1)(1 - |S_\ell|^2)$$

$$\sigma_{tot} = \sigma_{el} + \sigma_{inel} = \frac{2\pi}{k^2}\sum_\ell (2\ell+1)(1 - \text{Re}[S_\ell])$$

### Generalized Optical Theorem

For transition $a \to b$:

$$\text{Im}[f_{aa}(0)] = \frac{k}{4\pi}\sum_c \int |f_{ac}|^2 d\Omega_c$$

The imaginary part of forward elastic amplitude equals the total rate to all channels.

### Absorptive Scattering

For a perfectly absorbing ("black disk") target:

$$S_\ell = 0 \text{ for } \ell < ka, \quad S_\ell = 1 \text{ for } \ell > ka$$

$$\sigma_{el} = \sigma_{inel} = \pi a^2$$

$$\sigma_{tot} = 2\pi a^2$$

Twice the geometric cross section! This is the **black disk limit**.

---

## 6. Worked Examples

### Example 1: Verifying the Optical Theorem

**Problem:** A potential produces only s-wave scattering with $\delta_0 = 45°$ at wave number k. Verify the optical theorem.

**Solution:**

**Total cross section:**
$$\sigma_{tot} = \frac{4\pi}{k^2}\sin^2(45°) = \frac{4\pi}{k^2} \cdot \frac{1}{2} = \frac{2\pi}{k^2}$$

**Forward amplitude:**
$$f(0) = \sum_\ell (2\ell+1)f_\ell = f_0 = \frac{e^{i\pi/4}\sin(45°)}{k} = \frac{e^{i\pi/4}/\sqrt{2}}{k}$$

$$f(0) = \frac{1}{k}\left(\frac{1}{2} + \frac{i}{2}\right)$$

$$\text{Im}[f(0)] = \frac{1}{2k}$$

**Check optical theorem:**
$$\frac{4\pi}{k}\text{Im}[f(0)] = \frac{4\pi}{k} \cdot \frac{1}{2k} = \frac{2\pi}{k^2} = \sigma_{tot} \checkmark$$

### Example 2: High-Energy Limit

**Problem:** At high energies, a hadron-hadron total cross section is approximately $\sigma_{tot} = 40$ mb. Estimate $|f(0)|$.

**Solution:**

At high energy, $f(0) \approx i|f(0)|$ (purely imaginary).

From optical theorem:
$$|f(0)| = \text{Im}[f(0)] = \frac{k\sigma_{tot}}{4\pi}$$

At high energy, $k \approx E/(\hbar c)$ in natural units.

For $E = 100$ GeV and $\sigma = 40$ mb = $40 \times 10^{-27}$ cm² = $0.4$ fm²:

$$|f(0)| = \frac{(100\text{ GeV}/(0.197\text{ GeV·fm})) \times 0.4\text{ fm}^2}{4\pi}$$

$$|f(0)| = \frac{507 \times 0.4}{4\pi}\text{ fm} \approx \boxed{16\text{ fm}}$$

### Example 3: Black Disk

**Problem:** Show that for a totally absorbing disk of radius R, $\sigma_{tot} = 2\pi R^2$.

**Solution:**

For a black disk, $S_\ell = 0$ for $\ell < kR$ and $S_\ell = 1$ for $\ell > kR$.

$$\sigma_{tot} = \frac{2\pi}{k^2}\sum_{\ell=0}^{kR}(2\ell+1)(1 - \text{Re}[0]) = \frac{2\pi}{k^2}\sum_{\ell=0}^{kR}(2\ell+1)$$

$$= \frac{2\pi}{k^2}(kR)^2 = \boxed{2\pi R^2}$$

The factor of 2 comes from both absorption ($\pi R^2$) and diffractive elastic scattering ($\pi R^2$) around the edge.

---

## 7. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** If the total cross section is 10 barns and $k = 1$ fm⁻¹, what is Im[f(0)]?

**Problem 1.2:** For p-wave scattering ($\ell = 1$) with $\delta_1 = 30°$, verify the optical theorem using the partial wave formula.

**Problem 1.3:** A scatterer produces an angular distribution $d\sigma/d\Omega = a + b\cos\theta$. What constraints does the optical theorem place on the amplitude?

### Level 2: Intermediate

**Problem 2.1:** Derive the generalized optical theorem for the case where inelastic scattering is present, starting from $|S|^2 \leq 1$.

**Problem 2.2:** Show that for a resonance described by Breit-Wigner, the optical theorem is automatically satisfied at all energies.

**Problem 2.3:** Calculate the elastic and inelastic cross sections for a partial wave with $S_\ell = 0.5e^{i\pi/3}$.

### Level 3: Challenging

**Problem 3.1:** Using dispersion relations, show that if $\sigma_{tot}(E)$ is known for all E > 0, then Re[f(0, E)] can be computed. What physical principle underlies this?

**Problem 3.2:** In the Glauber model of high-energy nuclear scattering, the amplitude is:
$$f(\mathbf{q}) = \frac{ik}{2\pi}\int d^2b \, e^{i\mathbf{q}\cdot\mathbf{b}}[1 - e^{i\chi(\mathbf{b})}]$$
where $\chi$ is the eikonal phase. Show this satisfies the optical theorem.

**Problem 3.3:** Derive the Pomeranchuk theorem: at asymptotically high energies, the total cross sections for a particle and its antiparticle become equal.

---

## 8. Computational Lab: Optical Theorem Verification

```python
"""
Day 503 Computational Lab: Optical Theorem
Verifying and visualizing the optical theorem.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import legendre

# Constants
hbar = 1.0
m = 1.0


def partial_wave_amplitude(delta_l, k):
    """Compute partial wave amplitude f_l = exp(i*delta)*sin(delta)/k."""
    return np.exp(1j * delta_l) * np.sin(delta_l) / k


def scattering_amplitude(theta, k, phase_shifts):
    """Compute f(theta) from phase shifts."""
    cos_theta = np.cos(theta)
    f = 0j
    for l, delta in enumerate(phase_shifts):
        f_l = partial_wave_amplitude(delta, k)
        P_l = legendre(l)(cos_theta)
        f += (2*l + 1) * f_l * P_l
    return f


def total_cross_section_direct(k, phase_shifts):
    """Compute sigma_tot from partial wave formula."""
    sigma = 0
    for l, delta in enumerate(phase_shifts):
        sigma += (2*l + 1) * np.sin(delta)**2
    return 4 * np.pi / k**2 * sigma


def total_cross_section_integral(k, phase_shifts):
    """Compute sigma_tot by integrating |f|^2."""
    def integrand(theta):
        f = scattering_amplitude(theta, k, phase_shifts)
        return np.abs(f)**2 * np.sin(theta)

    result, _ = quad(integrand, 0, np.pi)
    return 2 * np.pi * result


def optical_theorem_check(k, phase_shifts):
    """Verify optical theorem: sigma_tot = (4*pi/k) * Im[f(0)]."""
    f_forward = scattering_amplitude(0, k, phase_shifts)
    sigma_optical = 4 * np.pi / k * np.imag(f_forward)
    sigma_direct = total_cross_section_direct(k, phase_shifts)
    return sigma_optical, sigma_direct


def plot_optical_theorem_verification():
    """Verify optical theorem for various phase shift configurations."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    k = 2.0

    # Test case 1: Single partial wave
    ax1 = axes[0, 0]
    delta_values = np.linspace(0, np.pi, 50)
    sigma_optical = []
    sigma_direct = []

    for delta in delta_values:
        phase_shifts = [delta]  # s-wave only
        so, sd = optical_theorem_check(k, phase_shifts)
        sigma_optical.append(so)
        sigma_direct.append(sd)

    ax1.plot(np.degrees(delta_values), sigma_optical, 'b-', linewidth=2,
             label='Optical theorem: 4π/k Im[f(0)]')
    ax1.plot(np.degrees(delta_values), sigma_direct, 'r--', linewidth=2,
             label='Direct: 4π/k² sin²δ')
    ax1.set_xlabel('Phase shift δ₀ (degrees)', fontsize=12)
    ax1.set_ylabel('Total cross section σ', fontsize=12)
    ax1.set_title('s-Wave Only', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test case 2: Multiple partial waves
    ax2 = axes[0, 1]
    configs = [
        ([0.5, 0.3, 0.1], 'δ₀=29°, δ₁=17°, δ₂=6°'),
        ([0.8, 0.4, 0.2, 0.1], 'δ₀=46°, δ₁=23°, δ₂=11°, δ₃=6°'),
        ([1.0, 0.5, 0.0], 'δ₀=57°, δ₁=29°, δ₂=0°'),
        ([np.pi/2, np.pi/4, np.pi/8], 'δ₀=90°, δ₁=45°, δ₂=23°')
    ]

    labels = []
    optical_vals = []
    direct_vals = []
    integral_vals = []

    for i, (deltas, label) in enumerate(configs):
        so, sd = optical_theorem_check(k, deltas)
        si = total_cross_section_integral(k, deltas)
        optical_vals.append(so)
        direct_vals.append(sd)
        integral_vals.append(si)
        labels.append(f'Config {i+1}')

    x = np.arange(len(configs))
    width = 0.25

    ax2.bar(x - width, optical_vals, width, label='Optical theorem')
    ax2.bar(x, direct_vals, width, label='Partial wave sum')
    ax2.bar(x + width, integral_vals, width, label='∫|f|²dΩ')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Total cross section σ', fontsize=12)
    ax2.set_title('Multiple Partial Waves: Three Ways to Compute σ_tot', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Test case 3: Energy dependence
    ax3 = axes[1, 0]
    k_values = np.linspace(0.5, 5, 50)

    # Fixed phase shifts (resonance-like behavior)
    E_R = 2.0
    Gamma = 0.5

    sigma_o = []
    sigma_d = []

    for k in k_values:
        E = k**2 / (2*m)
        delta0 = np.arctan2(Gamma/2, E_R - E)
        phase_shifts = [delta0, 0.1*delta0, 0.01*delta0]  # Predominantly s-wave
        so, sd = optical_theorem_check(k, phase_shifts)
        sigma_o.append(so)
        sigma_d.append(sd)

    ax3.plot(k_values, sigma_o, 'b-', linewidth=2, label='Optical theorem')
    ax3.plot(k_values, sigma_d, 'r--', linewidth=2, label='Direct calculation')
    ax3.set_xlabel('Wave number k', fontsize=12)
    ax3.set_ylabel('Total cross section σ', fontsize=12)
    ax3.set_title('Energy Dependence Near Resonance', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Test case 4: Relative error
    ax4 = axes[1, 1]
    np.random.seed(42)

    n_tests = 100
    errors = []
    l_max_values = []

    for _ in range(n_tests):
        l_max = np.random.randint(1, 8)
        phase_shifts = np.random.uniform(-np.pi/2, np.pi/2, l_max)
        k = np.random.uniform(0.5, 3.0)

        so, sd = optical_theorem_check(k, phase_shifts)
        rel_error = abs(so - sd) / sd if sd > 1e-10 else 0
        errors.append(rel_error)
        l_max_values.append(l_max)

    ax4.scatter(l_max_values, np.array(errors) * 100, alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Number of partial waves', fontsize=12)
    ax4.set_ylabel('Relative error (%)', fontsize=12)
    ax4.set_title('Numerical Verification (100 random tests)', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optical_theorem_verification.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_shadow_scattering():
    """Visualize the shadow interpretation of optical theorem."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    k = 3.0
    a = 1.0  # Scatterer radius

    # Create grid
    x = np.linspace(-4, 8, 300)
    z = np.linspace(-4, 4, 200)
    X, Z = np.meshgrid(x, z)

    # Incident wave
    psi_inc = np.exp(1j * k * X)

    # Scattered wave (simplified forward-peaked)
    R = np.sqrt((X)**2 + Z**2)
    R = np.where(R < 0.3, 0.3, R)  # Avoid singularity

    # Forward scattering amplitude (imaginary for absorber)
    f0 = 1j * k * a**2 / 2  # Absorbing disk approximation

    theta = np.arctan2(np.abs(Z), X)
    # Forward peaked angular distribution
    f_theta = f0 * np.exp(-theta**2 * (k*a)**2 / 4)

    psi_sc = f_theta * np.exp(1j * k * R) / R

    # Total wave
    psi_tot = psi_inc + psi_sc

    # Plot intensity
    ax1 = axes[0]
    intensity = np.abs(psi_tot)**2 - 1  # Deviation from incident

    im = ax1.pcolormesh(X, Z, intensity, shading='auto',
                        cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax1.plot(0, 0, 'ko', markersize=15)  # Scatterer
    ax1.set_xlabel('x (beam direction)', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Intensity Deviation from Incident\n(Shadow behind scatterer)',
                  fontsize=12)
    ax1.set_aspect('equal')
    plt.colorbar(im, ax=ax1, label='|ψ|² - 1')

    # Forward intensity vs x
    ax2 = axes[1]
    x_line = np.linspace(-2, 10, 500)
    z_line = 0

    R_line = np.abs(x_line)
    R_line = np.where(R_line < 0.3, 0.3, R_line)

    psi_inc_line = np.exp(1j * k * x_line)

    # Only consider forward direction
    mask_forward = x_line > 0.5
    f_line = np.zeros_like(x_line, dtype=complex)
    f_line[mask_forward] = f0 * np.exp(1j * k * R_line[mask_forward]) / R_line[mask_forward]

    psi_tot_line = psi_inc_line + f_line
    intensity_line = np.abs(psi_tot_line)**2

    ax2.plot(x_line, np.abs(psi_inc_line)**2, 'b--', linewidth=1.5,
             label='Incident |ψ_inc|²')
    ax2.plot(x_line, intensity_line, 'r-', linewidth=2,
             label='Total |ψ_tot|²')
    ax2.axvline(x=0, color='k', linestyle=':', alpha=0.5)

    ax2.fill_between(x_line[mask_forward],
                     intensity_line[mask_forward],
                     1, alpha=0.3, color='gray', label='Shadow')

    ax2.set_xlabel('x (beam direction)', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title('Forward Intensity (y = 0)\nDestructive interference creates shadow',
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('shadow_scattering.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_inelastic_scattering():
    """Analyze inelastic scattering and modified optical theorem."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k = 2.0

    # Vary inelasticity parameter η from 0 to 1
    eta_values = np.linspace(0.01, 1.0, 100)

    # S = η * exp(2iδ)
    delta = np.pi / 4  # Fixed phase

    sigma_el = []
    sigma_inel = []
    sigma_tot = []
    im_f0 = []

    for eta in eta_values:
        S = eta * np.exp(2j * delta)

        # Cross sections (s-wave only)
        s_el = np.pi / k**2 * np.abs(1 - S)**2
        s_inel = np.pi / k**2 * (1 - np.abs(S)**2)
        s_tot = s_el + s_inel

        sigma_el.append(s_el)
        sigma_inel.append(s_inel)
        sigma_tot.append(s_tot)

        # Forward amplitude: f = (S - 1) / (2ik)
        f = (S - 1) / (2j * k)
        im_f0.append(np.imag(f))

    sigma_el = np.array(sigma_el)
    sigma_inel = np.array(sigma_inel)
    sigma_tot = np.array(sigma_tot)
    im_f0 = np.array(im_f0)

    # Plot cross sections
    ax1 = axes[0]
    ax1.plot(eta_values, sigma_el * k**2 / np.pi, 'b-', linewidth=2,
             label='σ_el')
    ax1.plot(eta_values, sigma_inel * k**2 / np.pi, 'r-', linewidth=2,
             label='σ_inel')
    ax1.plot(eta_values, sigma_tot * k**2 / np.pi, 'k-', linewidth=2,
             label='σ_tot')

    ax1.set_xlabel('Inelasticity η = |S|', fontsize=12)
    ax1.set_ylabel('Cross section × k²/π', fontsize=12)
    ax1.set_title(f'Cross Sections vs Inelasticity (δ = {np.degrees(delta):.0f}°)',
                  fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Special cases
    ax1.axvline(x=1, color='g', linestyle='--', alpha=0.5)
    ax1.text(0.95, 0.1, 'Pure\nelastic', fontsize=10, ha='right')
    ax1.axvline(x=0, color='m', linestyle='--', alpha=0.5)
    ax1.text(0.05, 0.5, 'Black\ndisk', fontsize=10)

    # Verify optical theorem
    ax2 = axes[1]
    sigma_optical = 4 * np.pi / k * im_f0

    ax2.plot(eta_values, sigma_tot, 'b-', linewidth=2,
             label='σ_tot (direct)')
    ax2.plot(eta_values, sigma_optical, 'r--', linewidth=2,
             label='4π/k Im[f(0)]')

    ax2.set_xlabel('Inelasticity η = |S|', fontsize=12)
    ax2.set_ylabel('Cross section', fontsize=12)
    ax2.set_title('Optical Theorem with Inelastic Scattering', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inelastic_optical_theorem.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_dispersion_relation():
    """Show connection between optical theorem and dispersion relations."""

    print("=" * 60)
    print("DISPERSION RELATIONS AND OPTICAL THEOREM")
    print("=" * 60)

    print("""
    The optical theorem connects forward scattering to total cross section:

        Im[f(0, E)] = (k/4π) σ_tot(E)

    Causality (signals can't travel faster than light) requires that
    f(0, E) is analytic in the upper half of the complex E plane.

    This analyticity leads to DISPERSION RELATIONS:

        Re[f(0, E)] = (2E/π) P∫₀^∞ [Im f(0, E')] / (E'² - E²) dE'

    Substituting the optical theorem:

        Re[f(0, E)] = (E/2π²) P∫₀^∞ [k' σ_tot(E')] / (E'² - E²) dE'

    Physical implications:
    1. If σ_tot(E) is known at all energies, Re[f(0)] can be computed
    2. The real and imaginary parts of f(0) are not independent
    3. Measurements of σ_tot(E) constrain the full forward amplitude

    This is related to the Kramers-Kronig relations in optics!
    """)

    # Numerical demonstration
    fig, ax = plt.subplots(figsize=(10, 6))

    # Model total cross section (resonance + background)
    E = np.linspace(0.1, 20, 500)
    E_R = 5.0
    Gamma = 2.0
    sigma_bg = 0.5

    # Breit-Wigner resonance
    sigma_tot = sigma_bg + 10 * (Gamma/2)**2 / ((E - E_R)**2 + (Gamma/2)**2)

    ax.plot(E, sigma_tot, 'b-', linewidth=2)
    ax.axvline(x=E_R, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Energy E', fontsize=12)
    ax.set_ylabel('σ_tot(E)', fontsize=12)
    ax.set_title('Model Total Cross Section\n' +
                 'Dispersion relations connect this to Re[f(0)]', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dispersion_relation_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def black_disk_analysis():
    """Analyze the black disk limit in detail."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Black disk parameters
    R = 2.0
    k = 5.0

    l_max = int(k * R) + 10  # Include partial waves up to l ~ kR
    l_values = np.arange(l_max + 1)

    # S-matrix for black disk
    S = np.where(l_values < k * R, 0, 1)

    # Partial wave cross sections
    sigma_l_el = np.pi / k**2 * (2*l_values + 1) * np.abs(1 - S)**2
    sigma_l_inel = np.pi / k**2 * (2*l_values + 1) * (1 - np.abs(S)**2)

    # Plot partial wave contributions
    ax1 = axes[0]
    ax1.bar(l_values - 0.2, sigma_l_el, 0.4, label='Elastic')
    ax1.bar(l_values + 0.2, sigma_l_inel, 0.4, label='Inelastic')
    ax1.axvline(x=k*R, color='r', linestyle='--',
                label=f'l = kR = {k*R:.1f}')

    ax1.set_xlabel('Angular momentum l', fontsize=12)
    ax1.set_ylabel('σ_l', fontsize=12)
    ax1.set_title('Black Disk: Partial Wave Contributions', fontsize=12)
    ax1.legend()
    ax1.set_xlim(-0.5, l_max + 0.5)
    ax1.grid(True, alpha=0.3)

    # Total cross sections
    sigma_el_tot = np.sum(sigma_l_el)
    sigma_inel_tot = np.sum(sigma_l_inel)
    sigma_tot = sigma_el_tot + sigma_inel_tot

    print("\nBlack Disk Cross Sections:")
    print(f"  R = {R}, k = {k}, kR = {k*R}")
    print(f"  σ_el = {sigma_el_tot:.4f} (theory: πR² = {np.pi*R**2:.4f})")
    print(f"  σ_inel = {sigma_inel_tot:.4f} (theory: πR² = {np.pi*R**2:.4f})")
    print(f"  σ_tot = {sigma_tot:.4f} (theory: 2πR² = {2*np.pi*R**2:.4f})")

    # Angular distribution
    ax2 = axes[1]
    theta = np.linspace(0.001, np.pi/4, 200)

    # Compute f(theta) for black disk
    f_theta = np.zeros_like(theta, dtype=complex)
    for l in range(l_max + 1):
        f_l = (S[l] - 1) / (2j * k)
        P_l = np.zeros_like(theta)
        for i, t in enumerate(theta):
            P_l[i] = legendre(l)(np.cos(t))
        f_theta += (2*l + 1) * f_l * P_l

    dsigma = np.abs(f_theta)**2

    ax2.semilogy(np.degrees(theta), dsigma, 'b-', linewidth=2)
    ax2.set_xlabel('Scattering angle θ (degrees)', fontsize=12)
    ax2.set_ylabel('dσ/dΩ', fontsize=12)
    ax2.set_title('Black Disk: Diffractive Angular Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Mark first diffraction minimum (θ ~ 1/(kR))
    theta_min = np.degrees(1.22 * np.pi / (k * R))
    ax2.axvline(x=theta_min, color='r', linestyle='--', alpha=0.5,
                label=f'First min ~ {theta_min:.1f}°')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('black_disk_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Day 503: Optical Theorem")
    print("=" * 60)

    plot_optical_theorem_verification()
    visualize_shadow_scattering()
    plot_inelastic_scattering()
    demonstrate_dispersion_relation()
    black_disk_analysis()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Optical theorem | σ_tot = (4π/k) Im[f(0)] |
| Unitarity | S†S = 1, probability conservation |
| Shadow scattering | Forward interference creates "shadow" |
| Black disk | σ_tot = 2πR², half elastic, half inelastic |
| Dispersion relations | Re[f] related to Im[f] via integral |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\sigma_{tot} = \frac{4\pi}{k}\text{Im}[f(0)]$$ | Optical theorem |
| $$S^\dagger S = I$$ | S-matrix unitarity |
| $$\sigma_{el} + \sigma_{inel} = \frac{4\pi}{k}\text{Im}[f_{el}(0)]$$ | Generalized optical theorem |
| $$\|S\| \leq 1$$ | Inelasticity constraint |

---

## 10. Daily Checklist

### Conceptual Understanding
- [ ] I can derive the optical theorem from unitarity
- [ ] I understand the shadow scattering interpretation
- [ ] I know how to apply it to consistency checks
- [ ] I understand the extension to inelastic scattering

### Mathematical Skills
- [ ] I can verify the optical theorem for any phase shifts
- [ ] I can calculate cross sections multiple ways
- [ ] I understand the black disk limit
- [ ] I can connect to dispersion relations

### Computational Skills
- [ ] I verified the optical theorem numerically
- [ ] I visualized shadow scattering
- [ ] I analyzed inelastic cross sections
- [ ] I explored the black disk model

---

## 11. Preview: Day 504

Tomorrow is the **Semester 1A Capstone**—a comprehensive review of everything we've learned:

- Key concepts from Months 13-18
- Mathematical framework of quantum mechanics
- Angular momentum and spin
- Approximation methods
- Identical particles and many-body physics
- Scattering theory
- Self-assessment for transition to Semester 1B

Prepare to consolidate your understanding of foundational quantum mechanics!

---

## References

1. Griffiths, D.J. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Ch. 11.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 7.

3. Newton, R.G. (1982). *Scattering Theory of Waves and Particles*, Ch. 8.

4. Goldberger, M.L. & Watson, K.M. (1964). *Collision Theory*, Ch. 3.

---

*"The optical theorem is a direct consequence of the conservation of probability—it tells us that what goes in must come out, one way or another."*
— Roger Newton

---

**Day 503 Complete.** Tomorrow: Semester 1A Capstone Review.
