# Day 502: Resonances

## Overview

**Day 502 of 2520 | Week 72, Day 5 | Month 18: Identical Particles & Many-Body Physics**

Today we explore resonances—spectacular enhancements in scattering cross sections that occur when the projectile energy matches that of a quasi-bound state of the system. Resonances reveal the internal structure of composite systems and are the primary signals sought in particle physics experiments. We develop the Breit-Wigner formula, understand the relationship between width and lifetime, and examine how phase shifts behave near resonance.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Physical Picture of Resonances | 60 min |
| 10:00 AM | Breit-Wigner Formula Derivation | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Width, Lifetime, and Time Delay | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Phase Shift Behavior Near Resonance | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Examples: Nuclear and Particle Physics | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the physical origin of resonances
2. **Derive** the Breit-Wigner resonance formula
3. **Relate** resonance width to lifetime via uncertainty principle
4. **Analyze** phase shift behavior through a resonance
5. **Identify** resonances in cross section data
6. **Apply** resonance concepts to nuclear and particle physics

---

## 1. Physical Origin of Resonances

### What is a Resonance?

A **resonance** occurs when an incident particle can temporarily form a quasi-bound state with the target, before re-emitting. This happens when:

1. The scattering potential supports a quasi-bound level
2. The incident energy matches this level's energy
3. The particle "dwells" in the potential before escaping

### Classical Analogy

Think of pushing a child on a swing:
- Maximum amplitude when push frequency matches natural frequency
- Energy builds up through constructive interference
- **Resonance condition:** driving frequency = natural frequency

### Quantum Picture

In quantum mechanics:
- Quasi-bound states exist above the continuum threshold
- Particles can tunnel out, giving the state a finite lifetime
- Cross section peaks when $E \approx E_R$ (resonance energy)

### Types of Resonances

**Shape resonances:**
- Arise from centrifugal barrier ($\ell > 0$)
- Particle trapped behind barrier, tunnels out
- Example: α-decay

**Feshbach resonances:**
- Arise from coupling to closed channels
- Bound state in one channel couples to continuum in another
- Example: Ultracold atom collisions

---

## 2. The Breit-Wigner Formula

### Derivation from Partial Waves

Near a resonance in partial wave $\ell$, the phase shift varies rapidly. The partial wave amplitude:

$$f_\ell = \frac{1}{k\cot\delta_\ell - ik}$$

Near resonance energy $E_R$, expand:

$$k\cot\delta_\ell \approx \frac{2(E_R - E)}{\Gamma}$$

where $\Gamma$ is the **resonance width**.

### The Breit-Wigner Amplitude

$$\boxed{f_\ell = \frac{\Gamma/2}{(E_R - E) - i\Gamma/2}}$$

### Breit-Wigner Cross Section

$$\sigma_\ell = 4\pi(2\ell + 1)|f_\ell|^2 = \frac{4\pi(2\ell+1)}{k^2}\frac{(\Gamma/2)^2}{(E - E_R)^2 + (\Gamma/2)^2}$$

$$\boxed{\sigma_\ell(E) = \sigma_{max}\frac{(\Gamma/2)^2}{(E - E_R)^2 + (\Gamma/2)^2}}$$

where the peak cross section is:

$$\sigma_{max} = \frac{4\pi(2\ell+1)}{k_R^2}$$

### Properties of Breit-Wigner

| Property | Value |
|----------|-------|
| Peak position | $E = E_R$ |
| Peak height | $\sigma_{max}$ |
| Half-maximum at | $E = E_R \pm \Gamma/2$ |
| Full width at half maximum | $\Gamma$ |

### S-Matrix at Resonance

$$S_\ell = e^{2i\delta_\ell} = \frac{E - E_R + i\Gamma/2}{E - E_R - i\Gamma/2}$$

Note: $|S_\ell| = 1$ (unitary), as required.

At resonance ($E = E_R$): $S_\ell = -1$, so $\delta_\ell = \pi/2 + n\pi$.

---

## 3. Width and Lifetime

### Heisenberg Uncertainty Relation

The resonance width $\Gamma$ and lifetime $\tau$ are related by:

$$\boxed{\tau = \frac{\hbar}{\Gamma}}$$

This is a direct consequence of the energy-time uncertainty relation.

### Physical Interpretation

- **Narrow resonance** ($\Gamma$ small): Long-lived quasi-bound state
- **Broad resonance** ($\Gamma$ large): Short-lived, decays quickly
- **Sharp peak** in cross section corresponds to **long dwell time**

### Typical Values

| System | Width Γ | Lifetime τ |
|--------|---------|------------|
| Atomic line | 10⁻⁸ eV | 10⁻⁸ s |
| Nuclear state | 1 eV | 10⁻¹⁵ s |
| Z boson | 2.5 GeV | 3×10⁻²⁵ s |
| Higgs boson | 4 MeV | 2×10⁻²² s |

### Time Delay in Scattering

The **Wigner time delay** measures how long the particle spends in the scattering region:

$$\tau_{delay} = 2\hbar\frac{d\delta}{dE}$$

At resonance:
$$\tau_{delay} = \frac{2\hbar}{\Gamma} = 2\tau$$

The particle spends twice the lifetime in the scattering region.

---

## 4. Phase Shift Behavior

### Phase Shift Through Resonance

Near a resonance:

$$\tan\delta_\ell = \frac{\Gamma/2}{E_R - E}$$

$$\boxed{\delta_\ell = \arctan\left(\frac{\Gamma/2}{E_R - E}\right) + n\pi}$$

### Key Features

1. **Below resonance** ($E < E_R$): $\delta_\ell$ increases from background value
2. **At resonance** ($E = E_R$): $\delta_\ell = \pi/2$ (modulo π)
3. **Above resonance** ($E > E_R$): $\delta_\ell$ approaches $\pi$ (plus background)

The phase shift increases by **π** as energy passes through the resonance.

### Graphical Behavior

```
δ_ℓ(E)
  |
π |................................●--------
  |                              /
  |                            /
π/2|..........................●...............
  |                        /
  |                      /
  |                    /
0 |●------------------
  |__________________________|_______________|_____ E
                          E_R-Γ/2  E_R  E_R+Γ/2
```

### Levinson Connection

If the resonance corresponds to a quasi-bound state that evolves into a true bound state as the potential deepens, the phase shift behavior follows Levinson's theorem.

---

## 5. Background and Interference

### Resonance Plus Background

In reality, there's often a non-resonant background contribution:

$$f_\ell = f_{bg} + f_{res} = f_{bg} + \frac{\Gamma/2}{E_R - E - i\Gamma/2}$$

### Fano Resonance

When background and resonance interfere, the cross section takes the **Fano form**:

$$\sigma(E) \propto \frac{(q + \epsilon)^2}{1 + \epsilon^2}$$

where:
- $\epsilon = (E - E_R)/(\Gamma/2)$ is the reduced energy
- $q$ is the **Fano parameter** (ratio of resonant to direct amplitude)

### Fano Profile Shapes

| $q$ value | Profile shape |
|-----------|---------------|
| $q \to \infty$ | Symmetric Lorentzian peak |
| $q = 0$ | Symmetric dip (anti-resonance) |
| $q = 1$ | Asymmetric peak-dip |
| $q = -1$ | Asymmetric dip-peak |

### Physical Origin of Asymmetry

The asymmetric Fano profile arises from quantum interference between:
1. Direct scattering (background)
2. Scattering through the resonance

The relative phase between these paths changes sign across the resonance.

---

## 6. Worked Examples

### Example 1: Neutron-Nucleus Resonance

**Problem:** A neutron-nucleus scattering resonance occurs at $E_R = 1$ keV with width $\Gamma = 100$ eV (s-wave). Calculate:
(a) The peak cross section
(b) The resonance lifetime
(c) The cross section at $E = 1.05$ keV

**Solution:**

(a) For s-wave ($\ell = 0$):
$$\sigma_{max} = \frac{4\pi}{k_R^2}$$

At $E_R = 1$ keV for neutrons ($m = 939$ MeV):
$$k_R = \frac{\sqrt{2mE_R}}{\hbar c}\cdot c = \frac{\sqrt{2 \times 939 \times 0.001}}{197}\text{ fm}^{-1} = 6.95 \times 10^{-3}\text{ fm}^{-1}$$

$$\sigma_{max} = \frac{4\pi}{(6.95 \times 10^{-3})^2}\text{ fm}^2 = 2.60 \times 10^{5}\text{ fm}^2 = \boxed{2600\text{ barns}}$$

(b) Lifetime:
$$\tau = \frac{\hbar}{\Gamma} = \frac{6.58 \times 10^{-16}\text{ eV·s}}{100\text{ eV}} = \boxed{6.6 \times 10^{-18}\text{ s}}$$

(c) At $E = 1.05$ keV = $E_R + 50$ eV:
$$\sigma = \sigma_{max}\frac{(\Gamma/2)^2}{(E - E_R)^2 + (\Gamma/2)^2} = 2600 \times \frac{50^2}{50^2 + 50^2}$$
$$\sigma = 2600 \times \frac{1}{2} = \boxed{1300\text{ barns}}$$

### Example 2: Phase Shift Analysis

**Problem:** The s-wave phase shift data shows:

| E (MeV) | δ₀ (degrees) |
|---------|-------------|
| 0.5 | 30 |
| 1.0 | 60 |
| 1.5 | 90 |
| 2.0 | 120 |
| 2.5 | 150 |

Estimate the resonance energy and width.

**Solution:**

At resonance: $\delta_0 = 90°$ → $E_R \approx 1.5$ MeV

From $\tan\delta = \frac{\Gamma/2}{E_R - E}$:

At $E = 1.0$ MeV: $\tan(60°) = \sqrt{3} = \frac{\Gamma/2}{1.5 - 1.0} = \frac{\Gamma/2}{0.5}$

$$\frac{\Gamma}{2} = 0.5\sqrt{3} = 0.866 \text{ MeV}$$

$$\boxed{\Gamma \approx 1.7\text{ MeV}}$$

Check at $E = 2.0$ MeV: $\tan(120°) = -\sqrt{3} = \frac{0.866}{1.5 - 2.0} = \frac{0.866}{-0.5} = -1.73 \approx -\sqrt{3}$ ✓

### Example 3: Z Boson Resonance

**Problem:** The Z boson has $M_Z = 91.2$ GeV and $\Gamma_Z = 2.5$ GeV. In $e^+e^-$ collisions, the cross section for $e^+e^- \to Z \to \mu^+\mu^-$ is given by the Breit-Wigner formula. Estimate the peak cross section.

**Solution:**

At the Z pole, the cross section is:
$$\sigma_{peak} = \frac{12\pi}{M_Z^2}\frac{\Gamma_{ee}\Gamma_{\mu\mu}}{\Gamma_Z^2}$$

where $\Gamma_{ee} = \Gamma_{\mu\mu} \approx 84$ MeV (leptonic width).

$$\sigma_{peak} = \frac{12\pi}{(91.2\text{ GeV})^2}\frac{(0.084\text{ GeV})^2}{(2.5\text{ GeV})^2}$$

Converting to natural units ($\hbar c = 197$ MeV·fm):
$$\sigma_{peak} = 12\pi \times \left(\frac{197\text{ MeV·fm}}{91200\text{ MeV}}\right)^2 \times \frac{0.084^2}{2.5^2}$$

$$\sigma_{peak} \approx 2 \text{ nb} = \boxed{2 \times 10^{-33}\text{ cm}^2}$$

This is the famous "Z peak" measured at LEP.

---

## 7. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** A resonance has $E_R = 5$ MeV and $\Gamma = 0.5$ MeV. Calculate:
(a) The lifetime
(b) The cross section at $E = E_R + \Gamma$
(c) The phase shift at $E = E_R - \Gamma/2$

**Problem 1.2:** At what energies does the Breit-Wigner cross section equal 1/4 of the peak value?

**Problem 1.3:** Two overlapping s-wave resonances have the same width Γ but different energies $E_1$ and $E_2 = E_1 + 2\Gamma$. Sketch the total cross section.

### Level 2: Intermediate

**Problem 2.1:** Derive the Wigner time delay formula $\tau_{delay} = 2\hbar \, d\delta/dE$ from the energy derivative of the phase shift in the Breit-Wigner form.

**Problem 2.2:** A p-wave ($\ell = 1$) resonance at $E_R = 10$ MeV has width $\Gamma = 2$ MeV. Calculate the peak cross section and compare with the unitarity limit.

**Problem 2.3:** The Fano parameter $q = 3$ for a resonance. At what reduced energy $\epsilon$ does the cross section have a minimum?

### Level 3: Challenging

**Problem 3.1:** Derive the expression for the S-matrix near a resonance and verify that $|S| = 1$ (unitarity).

**Problem 3.2:** A potential supports a resonance that becomes a bound state when the potential depth increases by 10%. Estimate how the scattering length changes as the bound state emerges.

**Problem 3.3:** In nuclear physics, compound nucleus resonances are often analyzed using the R-matrix formalism:
$$f_\ell = \frac{\gamma_\ell^2}{E_\lambda - E - i\gamma_\ell^2/k}$$
where $\gamma_\ell$ is the reduced width. Show this reduces to Breit-Wigner form and identify the energy-dependent width.

---

## 8. Computational Lab: Resonance Analysis

```python
"""
Day 502 Computational Lab: Resonances
Analyzing Breit-Wigner resonances and phase shifts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Physical constants
hbar = 6.58e-16  # eV·s


def breit_wigner(E, E_R, Gamma, sigma_max):
    """Breit-Wigner cross section."""
    return sigma_max * (Gamma/2)**2 / ((E - E_R)**2 + (Gamma/2)**2)


def breit_wigner_phase(E, E_R, Gamma, delta_bg=0):
    """Phase shift through a Breit-Wigner resonance."""
    return np.arctan2(Gamma/2, E_R - E) + delta_bg


def fano_profile(epsilon, q):
    """Fano asymmetric profile."""
    return (q + epsilon)**2 / (1 + epsilon**2)


def plot_breit_wigner():
    """Plot Breit-Wigner resonance curves."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Basic Breit-Wigner
    ax1 = axes[0, 0]
    E = np.linspace(0, 10, 500)
    E_R = 5.0

    for Gamma in [0.5, 1.0, 2.0, 4.0]:
        sigma = breit_wigner(E, E_R, Gamma, 1.0)
        ax1.plot(E, sigma, linewidth=2, label=f'Γ = {Gamma}')

    ax1.axvline(x=E_R, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Energy E', fontsize=12)
    ax1.set_ylabel('σ / σ_max', fontsize=12)
    ax1.set_title('Breit-Wigner Cross Section', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Phase shift
    ax2 = axes[0, 1]
    for Gamma in [0.5, 1.0, 2.0]:
        delta = breit_wigner_phase(E, E_R, Gamma)
        ax2.plot(E, np.degrees(delta), linewidth=2, label=f'Γ = {Gamma}')

    ax2.axhline(y=90, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=E_R, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Energy E', fontsize=12)
    ax2.set_ylabel('Phase shift δ (degrees)', fontsize=12)
    ax2.set_title('Phase Shift Through Resonance', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Time delay
    ax3 = axes[1, 0]
    E_fine = np.linspace(3, 7, 500)
    E_R = 5.0
    Gamma = 1.0

    # Wigner time delay: τ = 2ℏ dδ/dE
    delta = breit_wigner_phase(E_fine, E_R, Gamma)
    dE = E_fine[1] - E_fine[0]
    ddelta_dE = np.gradient(delta, dE)
    tau_delay = 2 * hbar * ddelta_dE  # in eV^-1 * eV*s = s

    # Normalize by τ = ℏ/Γ
    tau_lifetime = hbar / Gamma
    ax3.plot(E_fine, tau_delay / tau_lifetime, 'b-', linewidth=2)
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5,
                label='Peak delay = 2τ')
    ax3.axvline(x=E_R, color='k', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Energy E', fontsize=12)
    ax3.set_ylabel('Time delay / τ_lifetime', fontsize=12)
    ax3.set_title('Wigner Time Delay', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # S-matrix trajectory
    ax4 = axes[1, 1]
    E_range = np.linspace(0, 10, 200)
    E_R = 5.0
    Gamma = 1.0

    # S = (E - E_R + iΓ/2) / (E - E_R - iΓ/2)
    S = (E_range - E_R + 1j*Gamma/2) / (E_range - E_R - 1j*Gamma/2)

    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)

    # Plot S-matrix trajectory
    ax4.plot(S.real, S.imag, 'b-', linewidth=2)

    # Mark key points
    S_below = (4 - E_R + 1j*Gamma/2) / (4 - E_R - 1j*Gamma/2)
    S_at = (E_R - E_R + 1j*Gamma/2) / (E_R - E_R - 1j*Gamma/2)
    S_above = (6 - E_R + 1j*Gamma/2) / (6 - E_R - 1j*Gamma/2)

    ax4.plot(S_below.real, S_below.imag, 'go', markersize=10,
             label=f'E < E_R')
    ax4.plot(S_at.real, S_at.imag, 'ro', markersize=12,
             label=f'E = E_R (S = -1)')
    ax4.plot(S_above.real, S_above.imag, 'mo', markersize=10,
             label=f'E > E_R')

    ax4.set_xlabel('Re(S)', fontsize=12)
    ax4.set_ylabel('Im(S)', fontsize=12)
    ax4.set_title('S-Matrix Trajectory Through Resonance', fontsize=12)
    ax4.set_aspect('equal')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig('breit_wigner_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_fano_profiles():
    """Show Fano resonance profiles for different q values."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epsilon = np.linspace(-5, 5, 500)

    # Different q values
    ax1 = axes[0]
    q_values = [-2, -1, 0, 1, 2, 5, 100]

    for q in q_values:
        sigma = fano_profile(epsilon, q)
        if q == 100:
            label = 'q → ∞ (Lorentzian)'
        else:
            label = f'q = {q}'
        ax1.plot(epsilon, sigma / max(sigma.max(), 1), linewidth=2, label=label)

    ax1.set_xlabel('Reduced energy ε = 2(E - E_R)/Γ', fontsize=12)
    ax1.set_ylabel('σ (normalized)', fontsize=12)
    ax1.set_title('Fano Resonance Profiles', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Comparison with experiment-like data
    ax2 = axes[1]

    # Simulate noisy data with Fano profile
    np.random.seed(42)
    q_true = 2.5
    E_R = 10.0
    Gamma = 2.0

    E_data = np.linspace(4, 16, 50)
    epsilon_data = 2 * (E_data - E_R) / Gamma
    sigma_data = fano_profile(epsilon_data, q_true)
    sigma_data *= (1 + 0.1 * np.random.randn(len(E_data)))

    ax2.plot(E_data, sigma_data, 'bo', markersize=5, label='Simulated data')

    # Fit
    def fano_fit(E, E_R, Gamma, q, A):
        eps = 2 * (E - E_R) / Gamma
        return A * fano_profile(eps, q)

    try:
        popt, _ = curve_fit(fano_fit, E_data, sigma_data,
                            p0=[10, 2, 2, 1], maxfev=5000)
        E_fit = np.linspace(4, 16, 200)
        ax2.plot(E_fit, fano_fit(E_fit, *popt), 'r-', linewidth=2,
                 label=f'Fit: $E_R$ = {popt[0]:.2f}, Γ = {popt[1]:.2f}, q = {popt[2]:.2f}')
    except Exception:
        pass

    ax2.set_xlabel('Energy E', fontsize=12)
    ax2.set_ylabel('Cross section σ', fontsize=12)
    ax2.set_title('Fano Profile Fitting', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fano_profiles.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_overlapping_resonances():
    """Show interference between overlapping resonances."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    E = np.linspace(0, 20, 500)

    # Two resonances
    E_R1 = 7.0
    E_R2 = 13.0
    Gamma1 = 2.0
    Gamma2 = 3.0

    # Non-interfering (add cross sections)
    ax1 = axes[0]
    sigma1 = breit_wigner(E, E_R1, Gamma1, 1.0)
    sigma2 = breit_wigner(E, E_R2, Gamma2, 0.8)
    sigma_sum = sigma1 + sigma2

    ax1.plot(E, sigma1, 'b--', linewidth=1.5, label=f'Resonance 1')
    ax1.plot(E, sigma2, 'r--', linewidth=1.5, label=f'Resonance 2')
    ax1.plot(E, sigma_sum, 'k-', linewidth=2, label='Sum (no interference)')

    ax1.set_xlabel('Energy E', fontsize=12)
    ax1.set_ylabel('Cross section σ', fontsize=12)
    ax1.set_title('Non-Interfering Resonances', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Interfering (add amplitudes)
    ax2 = axes[1]

    # Amplitudes (with relative phase)
    f1 = np.sqrt(1.0) * (Gamma1/2) / (E_R1 - E - 1j*Gamma1/2)
    f2 = np.sqrt(0.8) * (Gamma2/2) / (E_R2 - E - 1j*Gamma2/2)

    f_total = f1 + f2
    sigma_coherent = np.abs(f_total)**2

    ax2.plot(E, np.abs(f1)**2, 'b--', linewidth=1.5, label='|f₁|²')
    ax2.plot(E, np.abs(f2)**2, 'r--', linewidth=1.5, label='|f₂|²')
    ax2.plot(E, sigma_coherent, 'k-', linewidth=2, label='|f₁ + f₂|²')

    ax2.set_xlabel('Energy E', fontsize=12)
    ax2.set_ylabel('Cross section σ', fontsize=12)
    ax2.set_title('Interfering Resonances (Coherent)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('overlapping_resonances.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_z_boson():
    """Analyze Z boson resonance line shape."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Z boson parameters
    M_Z = 91.2  # GeV
    Gamma_Z = 2.5  # GeV

    # Energy range (center of mass)
    sqrt_s = np.linspace(80, 100, 200)

    # Breit-Wigner (simplified, ignoring phase space factors)
    sigma_peak = 40  # nb (approximate peak cross section for e+e- -> hadrons)
    sigma = breit_wigner(sqrt_s, M_Z, Gamma_Z, sigma_peak)

    # Simulated LEP data points
    sqrt_s_data = [88, 89, 90, 91, 91.2, 91.5, 92, 93, 94]
    sigma_data = breit_wigner(np.array(sqrt_s_data), M_Z, Gamma_Z, sigma_peak)
    sigma_data *= (1 + 0.05 * np.random.randn(len(sqrt_s_data)))  # Add noise
    errors = 0.05 * sigma_data

    ax.errorbar(sqrt_s_data, sigma_data, yerr=errors, fmt='ko',
                markersize=8, capsize=3, label='Simulated data')
    ax.plot(sqrt_s, sigma, 'b-', linewidth=2, label='Breit-Wigner fit')

    # Mark FWHM
    ax.axhline(y=sigma_peak/2, color='r', linestyle='--', alpha=0.5)
    ax.annotate('', xy=(M_Z - Gamma_Z/2, sigma_peak/2),
                xytext=(M_Z + Gamma_Z/2, sigma_peak/2),
                arrowprops=dict(arrowstyle='<->', color='red'))
    ax.text(M_Z, sigma_peak/2 + 2, f'Γ = {Gamma_Z} GeV', ha='center', fontsize=11)

    ax.set_xlabel(r'$\sqrt{s}$ (GeV)', fontsize=12)
    ax.set_ylabel('σ (nb)', fontsize=12)
    ax.set_title('Z Boson Resonance\n' +
                 f'$M_Z$ = {M_Z} GeV, Γ = {Gamma_Z} GeV', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('z_boson_resonance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nZ Boson Resonance Analysis:")
    print(f"  Mass: M_Z = {M_Z} GeV")
    print(f"  Width: Γ_Z = {Gamma_Z} GeV")
    print(f"  Lifetime: τ = ℏ/Γ = {hbar/Gamma_Z:.2e} s")
    print(f"  (Decay length: c*τ ≈ {3e8 * hbar/Gamma_Z:.2e} m)")


def resonance_from_potential():
    """Demonstrate resonance from a potential with barrier."""

    from scipy.integrate import solve_ivp

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Potential with barrier (e.g., l=1 centrifugal + attractive core)
    l = 2  # d-wave
    V0 = 10  # attractive well
    R = 1.0  # well radius

    def V_eff(r, k=0):
        """Effective potential including centrifugal term."""
        if r < 0.01:
            return 0
        centrifugal = l * (l + 1) / r**2
        if r < R:
            return -V0 + centrifugal
        else:
            return centrifugal

    # Plot potential
    ax1 = axes[0, 0]
    r = np.linspace(0.1, 5, 200)
    V_values = [V_eff(ri) for ri in r]

    ax1.plot(r, V_values, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.fill_between(r, -20, np.minimum(V_values, 0), alpha=0.2,
                     where=np.array(V_values) < 0)
    ax1.set_xlabel('r', fontsize=12)
    ax1.set_ylabel('V_eff(r)', fontsize=12)
    ax1.set_title(f'Effective Potential (ℓ = {l}, V₀ = {V0})', fontsize=12)
    ax1.set_ylim(-15, 10)
    ax1.grid(True, alpha=0.3)

    # Compute phase shifts
    ax2 = axes[0, 1]

    def compute_phase_l(k, l, V0, R, r_max=30):
        """Compute phase shift for angular momentum l."""
        def equation(r, y):
            u, up = y
            if r < 0.01:
                return [up, 0]
            cent = l * (l + 1) / r**2
            V = -V0 if r < R else 0
            return [up, (cent + 2*V - k**2) * u]

        r_start = 0.01
        y0 = [r_start**(l+1), (l+1) * r_start**l]

        sol = solve_ivp(equation, [r_start, r_max], y0,
                        method='RK45', dense_output=True)

        r_m = r_max - 5
        u, up = sol.sol(r_m)

        # Match to spherical Bessel functions
        from scipy.special import spherical_jn, spherical_yn
        j = spherical_jn(l, k * r_m)
        n = spherical_yn(l, k * r_m)
        jp = k * spherical_jn(l, k * r_m, derivative=True)
        np_deriv = k * spherical_yn(l, k * r_m, derivative=True)

        gamma = up / u
        num = jp - gamma * j
        den = np_deriv - gamma * n
        return np.arctan2(num, den)

    k_values = np.linspace(0.1, 3, 100)
    phases = []

    for k in k_values:
        try:
            delta = compute_phase_l(k, l, V0, R)
            phases.append(np.degrees(delta))
        except Exception:
            phases.append(np.nan)

    ax2.plot(k_values, phases, 'b-', linewidth=2)
    ax2.axhline(y=90, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Phase shift δ (degrees)', fontsize=12)
    ax2.set_title(f'd-wave (ℓ={l}) Phase Shift', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Cross section
    ax3 = axes[1, 0]
    phases_rad = np.array(phases) * np.pi / 180
    sigma = 4 * np.pi * (2*l + 1) / k_values**2 * np.sin(phases_rad)**2

    ax3.semilogy(k_values, sigma, 'b-', linewidth=2)
    ax3.set_xlabel('k', fontsize=12)
    ax3.set_ylabel('σ_ℓ', fontsize=12)
    ax3.set_title('d-wave Cross Section', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Energy vs phase derivative (time delay)
    ax4 = axes[1, 1]
    E_values = k_values**2 / 2  # E = k²/2m in natural units
    dE = E_values[1] - E_values[0]
    phases_rad_interp = np.array([p if not np.isnan(p) else 0
                                   for p in phases_rad])
    d_delta_dE = np.gradient(phases_rad_interp, dE)

    ax4.plot(E_values, d_delta_dE, 'b-', linewidth=2)
    ax4.set_xlabel('Energy E', fontsize=12)
    ax4.set_ylabel('dδ/dE (time delay)', fontsize=12)
    ax4.set_title('Time Delay (peaks at resonance)', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('shape_resonance.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Day 502: Resonances")
    print("=" * 60)

    plot_breit_wigner()
    plot_fano_profiles()
    plot_overlapping_resonances()
    analyze_z_boson()
    resonance_from_potential()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Resonance | Peak in cross section at quasi-bound state energy |
| Breit-Wigner | σ ∝ Γ²/[(E-E_R)² + Γ²/4] |
| Width Γ | Related to lifetime: τ = ℏ/Γ |
| Phase shift | Increases by π through resonance |
| Fano profile | Asymmetric shape from interference |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\sigma = \sigma_{max}\frac{(\Gamma/2)^2}{(E-E_R)^2 + (\Gamma/2)^2}$$ | Breit-Wigner |
| $$\tau = \hbar/\Gamma$$ | Lifetime-width relation |
| $$\delta = \arctan\frac{\Gamma/2}{E_R - E}$$ | Phase shift |
| $$S = e^{2i\delta} = -1$$ at resonance | S-matrix |

---

## 10. Daily Checklist

### Conceptual Understanding
- [ ] I understand the physical origin of resonances
- [ ] I can interpret width as inverse lifetime
- [ ] I see how phase shifts behave through resonance
- [ ] I understand Fano interference effects

### Mathematical Skills
- [ ] I can derive the Breit-Wigner formula
- [ ] I can extract resonance parameters from data
- [ ] I can calculate cross sections at various energies
- [ ] I can analyze S-matrix behavior

### Computational Skills
- [ ] I plotted Breit-Wigner curves
- [ ] I visualized phase shift behavior
- [ ] I analyzed the Z boson resonance
- [ ] I explored shape resonances from potentials

---

## 11. Preview: Day 503

Tomorrow we study the **optical theorem**—a fundamental consequence of unitarity:

- Derivation from S-matrix unitarity
- Im[f(0)] and total cross section
- Physical interpretation as shadow scattering
- Applications and experimental verification

The optical theorem connects forward scattering to total absorption and is one of the most elegant results in scattering theory.

---

## References

1. Griffiths, D.J. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Ch. 11.7.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 7.8.

3. Taylor, J.R. (2006). *Scattering Theory*, Ch. 12-13.

4. Fano, U. (1961). *Effects of Configuration Interaction on Intensities and Phase Shifts*, Phys. Rev.

---

*"A resonance is Nature's way of saying that something interesting has almost, but not quite, formed a bound state."*
— John Wheeler

---

**Day 502 Complete.** Tomorrow: Optical Theorem.
