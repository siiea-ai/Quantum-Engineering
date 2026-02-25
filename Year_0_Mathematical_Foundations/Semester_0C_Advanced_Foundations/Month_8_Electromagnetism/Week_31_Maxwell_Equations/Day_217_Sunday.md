# Day 217: Week Review and Integration

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning I | 2 hrs | Comprehensive review of Maxwell's equations |
| Morning II | 2 hrs | Integration problems and synthesis |
| Afternoon | 2 hrs | Historical perspective and modern applications |
| Evening | 2 hrs | Assessment and preparation for next week |

## Learning Objectives

By the end of today, you will be able to:

1. **Demonstrate mastery** of all four Maxwell equations and their applications
2. **Solve integrated problems** combining multiple electromagnetic concepts
3. **Trace the historical development** from Faraday to modern QED
4. **Connect Maxwell's equations** to quantum electrodynamics and photons
5. **Identify modern technologies** based on Maxwell's equations
6. **Self-assess readiness** for radiation and antenna theory

## Week 31 Comprehensive Review

### Maxwell's Equations: Complete Summary

| Equation | Differential Form | Integral Form | Physical Meaning |
|----------|------------------|---------------|------------------|
| Gauss (E) | $\nabla \cdot \vec{E} = \rho/\epsilon_0$ | $\oint \vec{E} \cdot d\vec{A} = Q_{enc}/\epsilon_0$ | Charges source E-fields |
| Gauss (B) | $\nabla \cdot \vec{B} = 0$ | $\oint \vec{B} \cdot d\vec{A} = 0$ | No magnetic monopoles |
| Faraday | $\nabla \times \vec{E} = -\partial\vec{B}/\partial t$ | $\oint \vec{E} \cdot d\vec{l} = -d\Phi_B/dt$ | Changing B creates E |
| Ampère-Maxwell | $\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\epsilon_0\partial\vec{E}/\partial t$ | $\oint \vec{B} \cdot d\vec{l} = \mu_0 I + \mu_0\epsilon_0 d\Phi_E/dt$ | Currents and changing E create B |

### Key Derived Results

**Wave Equation:**
$$\nabla^2 \vec{E} = \mu_0\epsilon_0 \frac{\partial^2 \vec{E}}{\partial t^2}$$

**Speed of Light:**
$$c = \frac{1}{\sqrt{\mu_0\epsilon_0}} = 299,792,458 \text{ m/s}$$

**Poynting Vector:**
$$\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}$$

**Energy Density:**
$$u = \frac{1}{2}\epsilon_0 E^2 + \frac{1}{2\mu_0}B^2$$

**Momentum Density:**
$$\vec{g} = \frac{\vec{S}}{c^2} = \epsilon_0(\vec{E} \times \vec{B})$$

### Wave Properties Summary

| Property | Vacuum | Dielectric | Conductor |
|----------|--------|------------|-----------|
| Speed | $c$ | $c/n$ | Complex |
| Wavelength | $\lambda_0$ | $\lambda_0/n$ | Complex |
| Frequency | $f$ | $f$ | $f$ |
| E/B ratio | $c$ | $c/n$ | Complex |
| Attenuation | None | Minimal | Skin depth $\delta$ |

## Integration Problems

### Problem 1: Complete EM System Analysis

A long solenoid (radius $R = 5$ cm, $n = 1000$ turns/m) has current increasing as $I(t) = I_0 t$ where $I_0 = 100$ A/s.

**(a)** Find the magnetic field inside and the induced electric field everywhere.

**(b)** Calculate the Poynting vector and show energy flows into the solenoid.

**(c)** If the solenoid has resistance $R_{coil} = 2\ \Omega$ and inductance $L = 0.1$ H, find the power balance.

**Solution:**

**(a) Magnetic field:** Inside solenoid ($r < R$):
$$B = \mu_0 n I = \mu_0 n I_0 t = (4\pi \times 10^{-7})(1000)(100)t = 0.126 t \text{ T}$$

**Induced E-field:** Using Faraday's law with circular Amperian loops:

Inside ($r < R$):
$$E(2\pi r) = -\pi r^2 \frac{dB}{dt} = -\pi r^2 \mu_0 n I_0$$
$$E = -\frac{r \mu_0 n I_0}{2} = -\frac{r(0.126)}{2} = -0.063 r \text{ V/m}$$

Outside ($r > R$):
$$E(2\pi r) = -\pi R^2 \mu_0 n I_0$$
$$E = -\frac{R^2 \mu_0 n I_0}{2r} = -\frac{(0.05)^2(0.126)}{2r} = -\frac{1.57 \times 10^{-4}}{r} \text{ V/m}$$

**(b) Poynting vector at $r = R$:**
$$\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}$$

$\vec{E}$ is tangential (azimuthal), $\vec{B}$ is axial, so $\vec{S}$ points **radially inward**.

$$S = \frac{|E||B|}{\mu_0} = \frac{(0.063 \times 0.05)(0.126 t)}{4\pi \times 10^{-7}} = \frac{3.97 \times 10^{-4} t}{4\pi \times 10^{-7}} = 316 t \text{ W/m}^2$$

Power flowing in through cylindrical surface (length $l$):
$$P_{in} = S \cdot 2\pi R l = 316 t \cdot 2\pi(0.05)l = 99.2 t \cdot l \text{ W/m}$$

**(c) Power balance:**

Energy stored in magnetic field:
$$U = \frac{B^2}{2\mu_0} \cdot \pi R^2 l = \frac{(0.126 t)^2}{2(4\pi \times 10^{-7})} \cdot \pi(0.05)^2 l = 49.5 t^2 \cdot l \text{ J/m}$$

Rate of change:
$$\frac{dU}{dt} = 99 t \cdot l \text{ W/m}$$

This matches the Poynting flux (within rounding)! ✓

### Problem 2: EM Wave at Interface

A plane wave with $E_0 = 100$ V/m and $\lambda = 500$ nm travels in air and hits a glass surface ($n = 1.5$) at 45°.

**(a)** Find the refracted angle and wavelength in glass.

**(b)** Calculate the reflectance for both polarizations.

**(c)** Find the Poynting vector magnitudes for incident, reflected, and transmitted waves.

**(d)** Verify energy conservation.

**Solution:**

**(a) Snell's law:**
$$\sin\theta_2 = \frac{n_1}{n_2}\sin\theta_1 = \frac{1}{1.5}\sin(45°) = 0.471$$
$$\theta_2 = 28.1°$$

Wavelength in glass: $\lambda_{glass} = 500/1.5 = 333$ nm

**(b) Fresnel coefficients:**

$\cos\theta_1 = 0.707$, $\cos\theta_2 = 0.882$

s-polarization:
$$r_s = \frac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2} = \frac{0.707 - 1.323}{0.707 + 1.323} = -0.304$$
$$R_s = 0.092 = 9.2\%$$

p-polarization:
$$r_p = \frac{n_2\cos\theta_1 - n_1\cos\theta_2}{n_2\cos\theta_1 + n_1\cos\theta_2} = \frac{1.061 - 0.882}{1.061 + 0.882} = 0.092$$
$$R_p = 0.0085 = 0.85\%$$

**(c) Poynting vectors:**

Incident intensity:
$$I_i = \frac{1}{2}\epsilon_0 c E_0^2 = \frac{1}{2}(8.854 \times 10^{-12})(3 \times 10^8)(100)^2 = 13.3 \text{ W/m}^2$$

For s-polarization:
- Reflected: $I_r = R_s I_i = 1.22$ W/m²
- Transmitted: $I_t = (1 - R_s)I_i = 12.1$ W/m²

**(d) Energy conservation:**

Power per unit area of interface:

Incident: $I_i \cos\theta_1 = 13.3 \times 0.707 = 9.4$ W/m²
Reflected: $I_r \cos\theta_1 = 1.22 \times 0.707 = 0.86$ W/m²
Transmitted: $I_t \cos\theta_2 = 12.1 \times 0.882 = 10.7$ W/m²

Wait—let's recalculate. The transmittance coefficient:
$$T_s = 1 - R_s = 0.908$$

But we need to account for the different beam widths:
$$T_s = \frac{n_2\cos\theta_2}{n_1\cos\theta_1}|t_s|^2$$

$t_s = 1 + r_s = 0.696$

$$T_s = \frac{1.5 \times 0.882}{1 \times 0.707}(0.696)^2 = 1.87 \times 0.484 = 0.905$$

Check: $R_s + T_s = 0.092 + 0.905 = 0.997 \approx 1$ ✓

### Problem 3: Photon-Wave Correspondence

A 1 mW laser beam (wavelength 632.8 nm, HeNe) is focused to a spot of diameter 10 μm.

**(a)** Calculate the intensity and electric field amplitude.

**(b)** How many photons pass through the focal spot per second?

**(c)** What is the radiation pressure on an absorbing surface?

**(d)** Estimate the electric field of a single photon in this focal volume.

**Solution:**

**(a) Intensity:**
$$I = \frac{P}{A} = \frac{10^{-3}}{\pi(5 \times 10^{-6})^2} = \frac{10^{-3}}{7.85 \times 10^{-11}} = 1.27 \times 10^7 \text{ W/m}^2$$

Electric field:
$$E_0 = \sqrt{\frac{2I}{\epsilon_0 c}} = \sqrt{\frac{2 \times 1.27 \times 10^7}{8.854 \times 10^{-12} \times 3 \times 10^8}}$$
$$E_0 = 3.1 \times 10^6 \text{ V/m}$$

**(b) Photon flux:**

Photon energy: $E_{ph} = hc/\lambda = (6.626 \times 10^{-34})(3 \times 10^8)/(632.8 \times 10^{-9}) = 3.14 \times 10^{-19}$ J

Photon rate: $\dot{N} = P/E_{ph} = 10^{-3}/(3.14 \times 10^{-19}) = 3.18 \times 10^{15}$ photons/s

**(c) Radiation pressure:**
$$P_{rad} = \frac{I}{c} = \frac{1.27 \times 10^7}{3 \times 10^8} = 42.3 \text{ mPa}$$

Force on focal spot: $F = P_{rad} \cdot A = 42.3 \times 10^{-3} \times 7.85 \times 10^{-11} = 3.3$ pN

**(d) Single photon field:**

Focal volume: $V \approx \pi r^2 \cdot \lambda = \pi(5 \times 10^{-6})^2(632.8 \times 10^{-9}) = 5 \times 10^{-17}$ m³

Single photon energy density: $u_{1ph} = E_{ph}/V = 3.14 \times 10^{-19}/(5 \times 10^{-17}) = 6.3 \times 10^{-3}$ J/m³

Corresponding electric field:
$$E_{1ph} = \sqrt{\frac{2u_{1ph}}{\epsilon_0}} = \sqrt{\frac{2 \times 6.3 \times 10^{-3}}{8.854 \times 10^{-12}}} = 1.2 \times 10^6 \text{ V/m}$$

This is comparable to the classical field because we have a small focal volume!

## Historical Perspective

### Timeline of Electromagnetic Theory

| Year | Scientist | Contribution |
|------|-----------|--------------|
| 1820 | Oersted | Electric current produces magnetic field |
| 1820 | Ampère | Force law between currents |
| 1831 | Faraday | Electromagnetic induction |
| 1835 | Gauss | Mathematical form of Gauss's law |
| 1855 | Faraday | Field concept and lines of force |
| 1861-65 | Maxwell | Complete electromagnetic theory |
| 1884 | Poynting | Energy flow in EM fields |
| 1887 | Hertz | Experimental confirmation of EM waves |
| 1905 | Einstein | Photoelectric effect (light quanta) |
| 1927 | Dirac | Quantum electrodynamics foundations |
| 1948 | Feynman, Schwinger, Tomonaga | Complete QED |

### Maxwell's Original Equations

Maxwell originally wrote 20 equations in component form! Oliver Heaviside (1880s) reformulated them into the four vector equations we use today.

Maxwell's key insight: Adding the displacement current $\epsilon_0 \partial\vec{E}/\partial t$ to Ampère's law, enabling electromagnetic waves.

### The Unification of Light and Electromagnetism

Before Maxwell:
- Electricity, magnetism, and light were separate phenomena
- Light was thought to be vibrations of a mysterious "luminiferous aether"

After Maxwell:
- Light is electromagnetic radiation
- All EM waves travel at $c$ in vacuum
- No need for aether (confirmed by Michelson-Morley, 1887)

## Modern Applications

### Communications Technology

| Technology | EM Principle | Frequency |
|------------|--------------|-----------|
| AM Radio | Modulated carrier waves | 500-1600 kHz |
| FM Radio | Frequency modulation | 88-108 MHz |
| WiFi | Digital EM pulses | 2.4, 5 GHz |
| 5G Cellular | mmWave | 30-300 GHz |
| Fiber Optics | Total internal reflection | ~200 THz |

### Medical Applications

- **MRI:** Uses RF pulses and gradient B-fields
- **X-ray imaging:** High-frequency EM radiation
- **Radiation therapy:** Targeted EM energy
- **Laser surgery:** Coherent light beams

### Quantum Technologies

- **Lasers:** Coherent photon states
- **Photonic quantum computing:** Single-photon manipulation
- **Quantum cryptography:** Photon polarization states
- **Optical tweezers:** Radiation pressure trapping

## Quantum Electrodynamics Connection

### From Classical to Quantum

| Classical Concept | Quantum Counterpart |
|-------------------|---------------------|
| EM wave | Photon state $|n\rangle$ |
| Intensity | Photon number $\langle n \rangle$ |
| Polarization | Photon spin (helicity) |
| E-field amplitude | Creation/annihilation operators |
| Poynting vector | Photon momentum flux |

### The Photon

Properties of the photon:
- **Mass:** 0 (travels at c)
- **Spin:** 1 (boson)
- **Charge:** 0
- **Energy:** $E = \hbar\omega$
- **Momentum:** $p = \hbar k = E/c$

### Vacuum Fluctuations

Even with no photons present, the quantized field has:
$$\langle 0|\hat{E}^2|0\rangle = \sum_{\vec{k},\lambda} \frac{\hbar\omega_k}{2\epsilon_0 V}$$

Observable consequences:
- Casimir effect (force between conducting plates)
- Lamb shift (hydrogen energy level corrections)
- Spontaneous emission (atoms radiate without stimulation)

## Practice Problems: Comprehensive Review

### Conceptual Questions

1. Why does $\nabla \cdot \vec{B} = 0$ imply there are no magnetic monopoles? What would change if monopoles existed?

2. Explain physically why the displacement current was necessary for Maxwell's equations to be consistent.

3. Why do E and B fields have equal energy density in an electromagnetic wave?

4. How does the skin effect explain why AC power cables don't need to be solid copper?

### Calculation Problems

5. A radio transmitter emits 50 kW at 100 MHz. At 10 km distance:
   (a) What is the intensity?
   (b) What are the E and B field amplitudes?
   (c) How many photons pass through 1 m² per second?

6. Light passes through three polarizers. The first is vertical, the second at 30°, the third at 60°. What fraction of the initial intensity emerges?

7. A 100 mW laser beam (λ = 500 nm) is reflected off a mirror. Calculate the force on the mirror.

8. At what frequency does the skin depth in copper equal the wavelength in vacuum?

### Challenge Problems

9. Derive the wave equation for B from Maxwell's equations, showing all steps.

10. A plane wave in vacuum hits a glass slab of thickness $d$ and index $n$ at normal incidence. Find the fraction of power transmitted through the slab, accounting for multiple reflections.

11. **Quantum:** For a cubic cavity of side $L$ with perfectly conducting walls, the allowed frequencies are $\omega_{lmn} = c\pi\sqrt{l^2 + m^2 + n^2}/L$. Calculate the zero-point energy of the first few modes for $L = 1$ μm.

## Self-Assessment Checklist

### Maxwell's Equations Mastery

- [ ] I can write all four Maxwell equations from memory
- [ ] I can explain the physical meaning of each equation
- [ ] I can derive the wave equation from Maxwell's equations
- [ ] I understand how $c = 1/\sqrt{\mu_0\epsilon_0}$ unifies light and EM

### Electromagnetic Waves

- [ ] I can describe plane wave solutions (E, B, k relationships)
- [ ] I understand polarization states (linear, circular, elliptical)
- [ ] I can calculate intensity, energy density, and momentum
- [ ] I can apply the Poynting vector to energy flow problems

### Waves at Interfaces

- [ ] I can apply Snell's law and find critical angles
- [ ] I understand the Fresnel equations and can calculate reflectance
- [ ] I know Brewster's angle and its applications
- [ ] I understand total internal reflection and evanescent waves

### Waves in Matter

- [ ] I understand how refractive index relates to ε and μ
- [ ] I can calculate skin depth in conductors
- [ ] I understand dispersion and its physical origins
- [ ] I can connect classical EM to quantum optics

### Historical and Conceptual

- [ ] I understand the historical development of EM theory
- [ ] I can identify modern applications of Maxwell's equations
- [ ] I understand how classical EM connects to QED
- [ ] I know the properties of photons

## Computational Lab: Week Integration

```python
"""
Day 217 Computational Lab: Week Review and Integration
Topics: Comprehensive Maxwell equations simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Physical constants
c = 3e8
epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7
h = 6.626e-34
hbar = h / (2 * np.pi)

# =============================================================================
# Part 1: Complete Maxwell Equations Summary Visualization
# =============================================================================

def create_maxwell_summary_visual():
    """Create comprehensive visual summary of Maxwell's equations."""

    fig = plt.figure(figsize=(16, 12))

    # Main title
    fig.suptitle("Maxwell's Equations: The Complete Theory of Classical Electromagnetism",
                fontsize=16, fontweight='bold', y=0.98)

    # Create 2x2 grid for the four equations
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    equations = [
        ("Gauss's Law (Electric)",
         r"$\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}$",
         "Electric charges are sources of E-field",
         "blue"),
        ("Gauss's Law (Magnetic)",
         r"$\nabla \cdot \vec{B} = 0$",
         "No magnetic monopoles exist",
         "green"),
        ("Faraday's Law",
         r"$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$",
         "Changing B-field creates E-field",
         "red"),
        ("Ampère-Maxwell Law",
         r"$\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\epsilon_0\frac{\partial \vec{E}}{\partial t}$",
         "Currents and changing E create B-field",
         "purple")
    ]

    for idx, (name, eq, meaning, color) in enumerate(equations):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Background
        ax.fill([0, 10, 10, 0], [0, 0, 10, 10], alpha=0.1, color=color)

        # Equation name
        ax.text(5, 8.5, name, fontsize=14, fontweight='bold',
               ha='center', va='center', color=color)

        # Equation
        ax.text(5, 5.5, eq, fontsize=18, ha='center', va='center')

        # Physical meaning
        ax.text(5, 2.5, meaning, fontsize=11, ha='center', va='center',
               style='italic')

    plt.savefig('maxwell_summary_visual.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 2: Unified EM Wave Animation
# =============================================================================

def animate_complete_em_wave():
    """Animate EM wave with E, B, S vectors and energy density."""

    fig = plt.figure(figsize=(14, 10))

    # 3D subplot for wave
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Wave parameters
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    omega = k * c
    E_0 = 1.0

    z = np.linspace(0, 2 * wavelength, 100)

    def update(frame):
        t = frame * 0.02

        # Clear all axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()

        # 3D Wave visualization
        E = E_0 * np.sin(k * z - omega * t)
        B = (E_0 / c) * np.sin(k * z - omega * t)

        ax1.plot(z, E, np.zeros_like(z), 'b-', linewidth=2, label='E')
        ax1.plot(z, np.zeros_like(z), B * c, 'r-', linewidth=2, label='cB')

        # Propagation direction arrow
        ax1.quiver(0, 0, 0, 1, 0, 0, color='green', arrow_length_ratio=0.3,
                  linewidth=2)
        ax1.text(0.5, 0, 0, 'k', fontsize=12, color='green')

        ax1.set_xlabel('z')
        ax1.set_ylabel('E (x)')
        ax1.set_zlabel('B (y)')
        ax1.set_title('3D EM Wave')
        ax1.set_xlim(0, 2 * wavelength)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_zlim(-1.5, 1.5)

        # Energy density
        u_E = 0.5 * epsilon_0 * E**2
        u_B = 0.5 * B**2 / mu_0
        u_total = u_E + u_B

        ax2.plot(z, u_E / np.max(u_total), 'b-', linewidth=2, label='u_E')
        ax2.plot(z, u_B / np.max(u_total), 'r--', linewidth=2, label='u_B')
        ax2.plot(z, u_total / np.max(u_total), 'g-', linewidth=2, label='u_total')
        ax2.set_xlabel('z')
        ax2.set_ylabel('Energy Density (normalized)')
        ax2.set_title('Energy Density')
        ax2.legend()
        ax2.set_xlim(0, 2 * wavelength)
        ax2.set_ylim(0, 1.1)

        # Poynting vector
        S = E * B / mu_0
        ax3.fill_between(z, 0, S / np.max(np.abs(S) + 1e-10),
                        where=S > 0, color='green', alpha=0.5)
        ax3.fill_between(z, 0, S / np.max(np.abs(S) + 1e-10),
                        where=S < 0, color='red', alpha=0.5)
        ax3.plot(z, S / np.max(np.abs(S) + 1e-10), 'k-', linewidth=2)
        ax3.set_xlabel('z')
        ax3.set_ylabel('S (normalized)')
        ax3.set_title('Poynting Vector (Energy Flow)')
        ax3.set_xlim(0, 2 * wavelength)
        ax3.set_ylim(-0.1, 1.1)

        # Phase relationships
        phases = np.linspace(0, 4 * np.pi, 200)
        ax4.plot(phases, np.sin(phases), 'b-', linewidth=2, label='E')
        ax4.plot(phases, np.sin(phases), 'r--', linewidth=2, label='B (in phase)')
        ax4.plot(phases, np.sin(phases)**2, 'g-', linewidth=2, label='S ∝ E×B')
        ax4.axvline(x=(omega * t) % (4 * np.pi), color='k', linestyle=':')
        ax4.set_xlabel('Phase (ωt - kz)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Phase Relationships')
        ax4.legend()
        ax4.set_xlim(0, 4 * np.pi)

        return []

    anim = FuncAnimation(fig, update, frames=50, interval=100, blit=False)

    plt.tight_layout()
    plt.savefig('complete_em_wave.png', dpi=150, bbox_inches='tight')
    plt.show()

    return anim

# =============================================================================
# Part 3: Comprehensive Comparison Chart
# =============================================================================

def create_comparison_charts():
    """Create comparison charts for different EM regimes."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========== EM Spectrum ==========
    ax = axes[0, 0]

    spectrum = [
        ('Radio', 1e6, 1e9, 'red'),
        ('Microwave', 1e9, 3e11, 'orange'),
        ('IR', 3e11, 4e14, 'darkred'),
        ('Visible', 4e14, 8e14, 'green'),
        ('UV', 8e14, 1e16, 'violet'),
        ('X-ray', 1e16, 1e19, 'blue'),
        ('Gamma', 1e19, 1e22, 'darkblue')
    ]

    for name, f_min, f_max, color in spectrum:
        ax.barh(0, np.log10(f_max) - np.log10(f_min),
               left=np.log10(f_min), height=0.3, color=color, alpha=0.7,
               edgecolor='black')
        if f_min > 1e12:
            ax.text((np.log10(f_min) + np.log10(f_max))/2, 0, name,
                   fontsize=8, ha='center', va='center', rotation=90)
        else:
            ax.text((np.log10(f_min) + np.log10(f_max))/2, 0, name,
                   fontsize=8, ha='center', va='center')

    ax.set_xlim(5, 23)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('log₁₀(Frequency / Hz)')
    ax.set_yticks([])
    ax.set_title('Electromagnetic Spectrum')

    # ========== Materials Properties ==========
    ax = axes[0, 1]

    materials = {
        'Vacuum': (1.0, 0),
        'Air': (1.0003, 0),
        'Water': (1.33, 0),
        'Glass': (1.5, 0),
        'Diamond': (2.4, 0),
        'Silicon': (3.5, 1e-3),
        'Copper': (0.1, 1e7)
    }

    names = list(materials.keys())
    n_values = [materials[m][0] for m in names]
    sigma_values = [materials[m][1] for m in names]

    x = np.arange(len(names))
    ax.bar(x - 0.2, n_values, 0.4, label='n', color='blue', alpha=0.7)
    ax.bar(x + 0.2, [np.log10(s + 1) for s in sigma_values], 0.4,
          label='log(σ+1)', color='red', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Material Properties (n and σ)')
    ax.legend()

    # ========== Wave-Particle Comparison ==========
    ax = axes[1, 0]

    wavelengths = np.logspace(-12, -3, 100)
    frequency = c / wavelengths
    E_photon = h * frequency / 1.6e-19  # eV

    ax.loglog(wavelengths * 1e9, E_photon, 'b-', linewidth=2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Photon Energy (eV)')
    ax.set_title('Wave-Particle Duality: λ vs E')
    ax.grid(True, alpha=0.3, which='both')

    # Mark key points
    points = [(500, 'Green'), (0.1, 'X-ray'), (1e6, 'Radio')]
    for wl, name in points:
        E = h * c / (wl * 1e-9) / 1.6e-19
        ax.plot(wl, E, 'ro', markersize=8)
        ax.annotate(name, xy=(wl, E), xytext=(wl*2, E*2),
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))

    # ========== Historical Timeline ==========
    ax = axes[1, 1]
    ax.axis('off')

    timeline = [
        (1820, 'Oersted: Current → B'),
        (1831, 'Faraday: Induction'),
        (1865, 'Maxwell: EM Theory'),
        (1887, 'Hertz: EM Waves'),
        (1905, 'Einstein: Photons'),
        (1948, 'QED Complete')
    ]

    for i, (year, event) in enumerate(timeline):
        y = 0.9 - i * 0.15
        ax.text(0.1, y, str(year), fontsize=12, fontweight='bold')
        ax.text(0.25, y, event, fontsize=11)
        ax.plot([0.05, 0.2], [y, y], 'b-', linewidth=2)

    ax.text(0.5, 0.98, 'Historical Development', fontsize=14,
           fontweight='bold', ha='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('comparison_charts.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 4: Problem Solving Summary
# =============================================================================

def demonstrate_problem_solving():
    """Demonstrate key problem-solving techniques from the week."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========== Fresnel at Various Angles ==========
    ax = axes[0, 0]

    n1, n2 = 1.0, 1.5
    theta_i = np.linspace(0, 90, 500)
    theta_i_rad = np.radians(theta_i)

    # Avoid NaN issues
    sin_theta_t = (n1/n2) * np.sin(theta_i_rad)
    sin_theta_t = np.clip(sin_theta_t, -1, 1)
    theta_t_rad = np.arcsin(sin_theta_t)

    r_s = (n1*np.cos(theta_i_rad) - n2*np.cos(theta_t_rad)) / \
          (n1*np.cos(theta_i_rad) + n2*np.cos(theta_t_rad))
    r_p = (n2*np.cos(theta_i_rad) - n1*np.cos(theta_t_rad)) / \
          (n2*np.cos(theta_i_rad) + n1*np.cos(theta_t_rad))

    R_s = np.abs(r_s)**2
    R_p = np.abs(r_p)**2

    ax.plot(theta_i, R_s * 100, 'b-', linewidth=2, label='s-pol')
    ax.plot(theta_i, R_p * 100, 'r-', linewidth=2, label='p-pol')
    ax.plot(theta_i, (R_s + R_p) / 2 * 100, 'g--', linewidth=2, label='Unpolarized')

    theta_B = np.degrees(np.arctan(n2/n1))
    ax.axvline(x=theta_B, color='purple', linestyle=':', label=f'Brewster: {theta_B:.1f}°')

    ax.set_xlabel('Angle of Incidence (degrees)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('Air → Glass Reflectance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== Skin Depth Summary ==========
    ax = axes[0, 1]

    freq = np.logspace(1, 10, 100)
    omega = 2 * np.pi * freq

    conductors = {'Cu': 5.96e7, 'Al': 3.77e7, 'Fe': 1.03e7}

    for name, sigma in conductors.items():
        delta = np.sqrt(2 / (mu_0 * sigma * omega))
        ax.loglog(freq, delta * 1e6, linewidth=2, label=name)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Skin Depth (μm)')
    ax.set_title('Skin Depth vs Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # ========== Radiation Pressure ==========
    ax = axes[1, 0]

    intensity = np.logspace(-2, 8, 100)
    P_absorb = intensity / c
    P_reflect = 2 * intensity / c

    ax.loglog(intensity, P_absorb, 'b-', linewidth=2, label='Absorbing')
    ax.loglog(intensity, P_reflect, 'r-', linewidth=2, label='Reflecting')

    # Reference pressures
    ax.axhline(y=1e5, color='g', linestyle='--', label='1 atm')
    ax.axhline(y=1e-5, color='orange', linestyle='--', label='Sunlight @ Earth')

    ax.set_xlabel('Intensity (W/m²)')
    ax.set_ylabel('Radiation Pressure (Pa)')
    ax.set_title('Radiation Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # ========== Photon Numbers ==========
    ax = axes[1, 1]

    power = np.logspace(-6, 3, 100)  # μW to kW

    for wavelength, color, name in [(400e-9, 'violet', '400 nm'),
                                     (550e-9, 'green', '550 nm'),
                                     (700e-9, 'red', '700 nm')]:
        E_photon = h * c / wavelength
        N_dot = power / E_photon
        ax.loglog(power * 1000, N_dot, color=color, linewidth=2, label=name)

    ax.set_xlabel('Power (mW)')
    ax.set_ylabel('Photons/second')
    ax.set_title('Photon Rate vs Power')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('problem_solving_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Day 217: Week Review and Integration")
    print("="*60)

    print("\n1. Maxwell's Equations Summary Visual")
    create_maxwell_summary_visual()

    print("\n2. Complete EM Wave Animation")
    anim = animate_complete_em_wave()

    print("\n3. Comparison Charts")
    create_comparison_charts()

    print("\n4. Problem Solving Summary")
    demonstrate_problem_solving()

    print("\nWeek 31 Review Complete!")
    print("\nKey Achievements:")
    print("- Mastered all four Maxwell equations")
    print("- Derived electromagnetic wave equation")
    print("- Understood energy and momentum in EM fields")
    print("- Analyzed wave behavior at interfaces")
    print("- Connected classical EM to quantum optics")
```

## Summary

### Week 31 Key Accomplishments

1. **Faraday's Law (Day 211):** Electromagnetic induction, Lenz's law, motional EMF
2. **Displacement Current (Day 212):** Maxwell's correction, Ampère-Maxwell law
3. **Complete Maxwell Equations (Day 213):** All four equations unified, boundary conditions
4. **EM Waves (Day 214):** Wave equation derivation, plane waves, polarization
5. **Energy and Momentum (Day 215):** Poynting vector, radiation pressure
6. **Waves in Matter (Day 216):** Refraction, Fresnel equations, skin effect

### Master Formula Sheet

| Topic | Key Formula |
|-------|-------------|
| Faraday's Law | $\mathcal{E} = -d\Phi_B/dt$ |
| Displacement Current | $J_D = \epsilon_0 \partial E/\partial t$ |
| Wave Equation | $\nabla^2 E = \mu_0\epsilon_0 \partial^2 E/\partial t^2$ |
| Speed of Light | $c = 1/\sqrt{\mu_0\epsilon_0}$ |
| Poynting Vector | $\vec{S} = (1/\mu_0)\vec{E} \times \vec{B}$ |
| Intensity | $I = \frac{1}{2}\epsilon_0 c E_0^2$ |
| Radiation Pressure | $P = I/c$ (absorbing), $P = 2I/c$ (reflecting) |
| Snell's Law | $n_1\sin\theta_1 = n_2\sin\theta_2$ |
| Skin Depth | $\delta = \sqrt{2/(\mu\sigma\omega)}$ |
| Photon Energy | $E = \hbar\omega = hf$ |

## Preview: Week 32

Next week we study **electromagnetic radiation from accelerating charges**:
- Larmor formula for radiated power
- Electric dipole radiation
- Antenna theory
- Retarded potentials
- Synchrotron radiation

This builds directly on Maxwell's equations to explain how electromagnetic waves are generated.

---

*"The theory I propose may therefore be called a theory of the Electromagnetic Field because it has to do with the space in the neighborhood of the electric or magnetic bodies."* — James Clerk Maxwell, 1864

---

## Daily Checklist

- [ ] I can write all four Maxwell equations from memory
- [ ] I can derive the wave equation from Maxwell's equations
- [ ] I understand energy, momentum, and the Poynting vector
- [ ] I can solve problems involving waves at interfaces
- [ ] I understand the connection between classical EM and photons
- [ ] I completed all computational labs this week
- [ ] I am ready for electromagnetic radiation topics

**Congratulations on completing Week 31: Maxwell's Equations!**
