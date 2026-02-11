# Day 216: Electromagnetic Waves in Matter

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning I | 2 hrs | Wave propagation in dielectrics |
| Morning II | 2 hrs | Reflection and refraction (Fresnel equations) |
| Afternoon | 2 hrs | Conductors, skin depth, and dispersion |
| Evening | 2 hrs | Computational lab: Wave behavior at interfaces |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive the wave equation** in linear, homogeneous media
2. **Calculate wave velocity** and refractive index in dielectrics
3. **Apply Snell's law** and the Fresnel equations at interfaces
4. **Analyze wave attenuation** in conductors (skin effect)
5. **Explain dispersion** and its relationship to absorption
6. **Connect classical EM in matter** to quantum optical properties

## Core Content

### 1. Maxwell's Equations in Matter

In a linear, homogeneous, isotropic medium:
$$\vec{D} = \epsilon\vec{E} = \epsilon_r\epsilon_0\vec{E}$$
$$\vec{B} = \mu\vec{H} = \mu_r\mu_0\vec{H}$$

Maxwell's equations become:
$$\nabla \cdot \vec{E} = \frac{\rho_f}{\epsilon}$$
$$\nabla \cdot \vec{B} = 0$$
$$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$$
$$\nabla \times \vec{H} = \vec{J}_f + \frac{\partial \vec{D}}{\partial t}$$

### 2. Wave Equation in Dielectrics

For a **source-free dielectric** ($\rho_f = 0$, $\vec{J}_f = 0$):

Following the same derivation as in vacuum:
$$\nabla^2 \vec{E} = \mu\epsilon\frac{\partial^2 \vec{E}}{\partial t^2}$$

The wave velocity is:
$$\boxed{v = \frac{1}{\sqrt{\mu\epsilon}} = \frac{c}{\sqrt{\mu_r\epsilon_r}}}$$

The **index of refraction**:
$$\boxed{n = \frac{c}{v} = \sqrt{\mu_r\epsilon_r}}$$

For non-magnetic materials ($\mu_r \approx 1$):
$$n \approx \sqrt{\epsilon_r}$$

### 3. Wave Solutions in Dielectrics

Plane wave solutions:
$$\vec{E} = \vec{E}_0 e^{i(\vec{k}\cdot\vec{r} - \omega t)}$$

With:
- Wave number: $k = \omega/v = n\omega/c = nk_0$
- Wavelength in medium: $\lambda = \lambda_0/n$
- Frequency: unchanged ($f$ is same in all media)

The E-B relationship:
$$\frac{E}{B} = v = \frac{c}{n}$$

### 4. Reflection and Refraction at Interfaces

At an interface between media with indices $n_1$ and $n_2$:

**Snell's Law:**
$$\boxed{n_1\sin\theta_1 = n_2\sin\theta_2}$$

This follows from the boundary condition that tangential E and H must be continuous.

**Total Internal Reflection:**
When $n_1 > n_2$, total reflection occurs for angles greater than the critical angle:
$$\sin\theta_c = \frac{n_2}{n_1}$$

### 5. Fresnel Equations

The reflection and transmission coefficients depend on polarization.

**s-polarization (E perpendicular to plane of incidence):**
$$r_s = \frac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2}$$

$$t_s = \frac{2n_1\cos\theta_1}{n_1\cos\theta_1 + n_2\cos\theta_2}$$

**p-polarization (E parallel to plane of incidence):**
$$r_p = \frac{n_2\cos\theta_1 - n_1\cos\theta_2}{n_2\cos\theta_1 + n_1\cos\theta_2}$$

$$t_p = \frac{2n_1\cos\theta_1}{n_2\cos\theta_1 + n_1\cos\theta_2}$$

**Reflectance and Transmittance:**
$$R_s = |r_s|^2, \quad T_s = \frac{n_2\cos\theta_2}{n_1\cos\theta_1}|t_s|^2$$

At **normal incidence** ($\theta_1 = 0$):
$$r = \frac{n_1 - n_2}{n_1 + n_2}, \quad R = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2$$

### 6. Brewster's Angle

At Brewster's angle, p-polarized light has **zero reflection**:
$$\tan\theta_B = \frac{n_2}{n_1}$$

At this angle, reflected and refracted rays are perpendicular.

For air-glass ($n = 1.5$): $\theta_B = 56.3°$

### 7. Waves in Conductors

In a conductor with conductivity $\sigma$, Ohm's law gives $\vec{J}_f = \sigma\vec{E}$.

The wave equation becomes:
$$\nabla^2 \vec{E} = \mu\epsilon\frac{\partial^2 \vec{E}}{\partial t^2} + \mu\sigma\frac{\partial \vec{E}}{\partial t}$$

For a plane wave $\vec{E} = E_0 e^{i(kz - \omega t)}\hat{x}$:
$$k^2 = \mu\epsilon\omega^2 + i\mu\sigma\omega = \mu\omega^2\left(\epsilon + \frac{i\sigma}{\omega}\right)$$

The wave vector is **complex**: $\tilde{k} = k + i\kappa$

$$\vec{E} = E_0 e^{-\kappa z}e^{i(kz - \omega t)}\hat{x}$$

The wave is **attenuated** as it penetrates the conductor.

### 8. Skin Depth

The **skin depth** $\delta$ is the distance over which the wave amplitude decreases by factor $e$:
$$\delta = \frac{1}{\kappa}$$

**Good conductor** ($\sigma \gg \omega\epsilon$):
$$\boxed{\delta = \sqrt{\frac{2}{\mu\sigma\omega}}}$$

Examples at 60 Hz:
- Copper: $\delta \approx 8.5$ mm
- Aluminum: $\delta \approx 11$ mm

At 1 GHz:
- Copper: $\delta \approx 2$ μm

Current flows in a thin "skin" near the surface!

### 9. Dispersion

In real materials, $\epsilon$ depends on frequency: $\epsilon = \epsilon(\omega)$.

This causes **dispersion**—different frequencies travel at different speeds.

**Phase velocity:** $v_p = \omega/k$

**Group velocity:** $v_g = d\omega/dk$ (velocity of wave packets/information)

The **Lorentz model** for dielectrics:
$$\epsilon(\omega) = \epsilon_0\left(1 + \frac{N e^2/m\epsilon_0}{\omega_0^2 - \omega^2 - i\gamma\omega}\right)$$

where $\omega_0$ is the resonance frequency and $\gamma$ is the damping.

### 10. Complex Refractive Index

In absorbing media, we use a complex refractive index:
$$\tilde{n} = n + i\kappa$$

- $n$ = refractive index (phase velocity)
- $\kappa$ = extinction coefficient (absorption)

The absorption coefficient:
$$\alpha = \frac{2\omega\kappa}{c} = \frac{4\pi\kappa}{\lambda_0}$$

Intensity decays as $I = I_0 e^{-\alpha z}$ (Beer's law).

## Quantum Mechanics Connection

### Photons in Dielectrics

In a medium, photons interact with atoms/electrons:

1. **Polarization response:** Photons induce dipoles, which re-radiate
2. **Effective mass:** Photons acquire an effective mass $m^* = \hbar\omega n^2/c^2$
3. **Polaritons:** In some materials, photons hybridize with excitations (phonon-polaritons, exciton-polaritons)

### Quantum Theory of Refractive Index

The refractive index arises from:
$$n^2 - 1 = \frac{N e^2}{\epsilon_0 m}\sum_j \frac{f_j}{\omega_j^2 - \omega^2 - i\gamma_j\omega}$$

where $f_j$ are **oscillator strengths** related to quantum transition probabilities.

### Spontaneous Emission in Media

In a medium with refractive index $n$, spontaneous emission rate is modified:
$$\Gamma = n \cdot \Gamma_0$$

(Purcell effect in its simplest form)

### Evanescent Waves and Tunneling

Total internal reflection produces **evanescent waves**:
$$\vec{E} \propto e^{-\kappa z}e^{i(k_x x - \omega t)}$$

This classical phenomenon has a quantum analog: **photon tunneling** through thin barriers (frustrated total internal reflection).

### Casimir Effect in Dielectrics

The Casimir force between dielectric slabs depends on their permittivities:
$$F \propto \int_0^\infty d\xi \, f[\epsilon_1(i\xi), \epsilon_2(i\xi)]$$

This shows quantum vacuum effects are modified by material properties.

## Worked Examples

### Example 1: Light in Glass

Light of wavelength 600 nm (in vacuum) enters glass ($n = 1.5$) at 30° incidence.

Find: (a) refraction angle, (b) wavelength in glass, (c) reflectance for s-polarization.

**Solution:**

**(a) Refraction angle (Snell's law):**
$$\sin\theta_2 = \frac{n_1}{n_2}\sin\theta_1 = \frac{1}{1.5}\sin(30°) = \frac{1}{3}$$
$$\theta_2 = 19.5°$$

**(b) Wavelength in glass:**
$$\lambda_{glass} = \frac{\lambda_0}{n} = \frac{600}{1.5} = 400 \text{ nm}$$

**(c) Reflectance:**
First find $\cos\theta_2 = \sqrt{1 - (1/3)^2} = 0.943$

$$r_s = \frac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2}$$
$$r_s = \frac{1 \times 0.866 - 1.5 \times 0.943}{1 \times 0.866 + 1.5 \times 0.943} = \frac{0.866 - 1.414}{0.866 + 1.414} = \frac{-0.548}{2.280}$$
$$r_s = -0.240$$

$$\boxed{R_s = |r_s|^2 = 0.058 = 5.8\%}$$

### Example 2: Skin Depth in Copper

Calculate the skin depth in copper ($\sigma = 5.96 \times 10^7$ S/m) at: (a) 60 Hz, (b) 1 MHz, (c) 1 GHz.

**Solution:**

Using $\delta = \sqrt{2/(\mu_0\sigma\omega)}$:

$$\delta = \sqrt{\frac{2}{(4\pi \times 10^{-7})(5.96 \times 10^7)(2\pi f)}}$$

$$\delta = \sqrt{\frac{1}{4\pi^2 \times 10^{-7} \times 5.96 \times 10^7 \times f}} = \frac{0.0661}{\sqrt{f}}$$

**(a) At 60 Hz:**
$$\delta = \frac{0.0661}{\sqrt{60}} = 8.53 \text{ mm}$$

**(b) At 1 MHz:**
$$\delta = \frac{0.0661}{\sqrt{10^6}} = 66.1 \ \mu\text{m}$$

**(c) At 1 GHz:**
$$\delta = \frac{0.0661}{\sqrt{10^9}} = 2.09 \ \mu\text{m}$$

$$\boxed{\delta_{60Hz} = 8.5 \text{ mm}, \quad \delta_{1MHz} = 66\ \mu\text{m}, \quad \delta_{1GHz} = 2.1\ \mu\text{m}}$$

### Example 3: Brewster's Angle and Polarizers

Light reflects off a glass surface ($n = 1.52$) at Brewster's angle.

Find: (a) Brewster's angle, (b) the refraction angle, (c) why this is used in laser cavities.

**Solution:**

**(a) Brewster's angle:**
$$\tan\theta_B = n = 1.52$$
$$\theta_B = \arctan(1.52) = 56.7°$$

**(b) Refraction angle:**
From Snell: $\sin\theta_2 = \sin(56.7°)/1.52 = 0.550$
$$\theta_2 = 33.4°$$

Note: $\theta_B + \theta_2 = 56.7° + 33.4° = 90.1° \approx 90°$ ✓

**(c) Laser cavity application:**

At Brewster's angle, p-polarized light has **zero reflection loss**. Laser windows oriented at Brewster's angle:
- Allow p-polarized light to pass without reflection loss
- Force the laser to emit p-polarized light (s-polarization has losses)
- Eliminate need for anti-reflection coatings

$$\boxed{\theta_B = 56.7°, \quad \text{reflected and refracted rays perpendicular}}$$

## Practice Problems

### Level 1: Direct Application

1. Light passes from water ($n = 1.33$) into diamond ($n = 2.42$) at 45°. Find the refraction angle.

2. Calculate the reflectance at normal incidence for an air-glass interface ($n = 1.5$).

3. What is the critical angle for total internal reflection from glass ($n = 1.5$) to air?

### Level 2: Intermediate

4. Derive Snell's law from the boundary condition that tangential E is continuous across an interface.

5. A 1 GHz wave penetrates aluminum ($\sigma = 3.77 \times 10^7$ S/m). What fraction of the intensity remains at depth 10 μm?

6. Show that at Brewster's angle, the reflected and transmitted rays are perpendicular.

### Level 3: Challenging

7. Derive the Fresnel equations for s-polarization by matching boundary conditions for E and H.

8. A thin film of thickness $d$ and refractive index $n$ is deposited on glass. Find the condition for minimum reflection at normal incidence (anti-reflection coating).

9. **Quantum connection:** The quantum mechanical oscillator strength $f_{12}$ for a transition is related to the electric dipole matrix element by $f_{12} = \frac{2m\omega_{12}}{3\hbar}|\langle 1|r|2\rangle|^2$. Show this has dimensions of 1 and explain its role in the refractive index.

## Computational Lab: EM Waves in Matter

```python
"""
Day 216 Computational Lab: Electromagnetic Waves in Matter
Topics: Dielectrics, reflection/refraction, skin effect, dispersion
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D

# Set up styling
plt.style.use('default')

# Physical constants
c = 3e8
epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7

# =============================================================================
# Part 1: Snell's Law Visualization
# =============================================================================

def visualize_snells_law():
    """Visualize refraction at an interface."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ========== Basic refraction ==========
    ax = axes[0]

    n1, n2 = 1.0, 1.5  # Air to glass

    # Interface
    ax.axhline(y=0, color='black', linewidth=2)
    ax.fill_between([-3, 3], 0, -3, alpha=0.2, color='blue', label=f'Glass (n={n2})')
    ax.fill_between([-3, 3], 0, 3, alpha=0.1, color='white', label=f'Air (n={n1})')

    # Incident angles
    for theta_i_deg in [20, 40, 60]:
        theta_i = np.radians(theta_i_deg)
        theta_t = np.arcsin(n1 * np.sin(theta_i) / n2)

        # Incident ray
        x_inc = [-2 * np.sin(theta_i), 0]
        y_inc = [2 * np.cos(theta_i), 0]
        ax.plot(x_inc, y_inc, 'r-', linewidth=2)
        ax.annotate('', xy=(0, 0), xytext=(x_inc[0], y_inc[0]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

        # Refracted ray
        x_ref = [0, 2 * np.sin(theta_t)]
        y_ref = [0, -2 * np.cos(theta_t)]
        ax.plot(x_ref, y_ref, 'b-', linewidth=2)
        ax.annotate('', xy=(x_ref[1], y_ref[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))

        # Labels
        ax.text(-2.2 * np.sin(theta_i), 2.2 * np.cos(theta_i),
               f'{theta_i_deg}°', fontsize=10, color='red')
        ax.text(2.2 * np.sin(theta_t), -2.2 * np.cos(theta_t),
               f'{np.degrees(theta_t):.1f}°', fontsize=10, color='blue')

    # Normal line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.1, 2.5, 'Normal', fontsize=10, color='gray')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Snell's Law: $n_1\\sin\\theta_1 = n_2\\sin\\theta_2$")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    # ========== Total internal reflection ==========
    ax = axes[1]

    n1, n2 = 1.5, 1.0  # Glass to air (reversed)
    theta_c = np.arcsin(n2 / n1)

    ax.axhline(y=0, color='black', linewidth=2)
    ax.fill_between([-3, 3], 0, 3, alpha=0.2, color='blue', label=f'Glass (n={n1})')
    ax.fill_between([-3, 3], 0, -3, alpha=0.1, color='white', label=f'Air (n={n2})')

    # Incident angles including beyond critical
    for theta_i_deg in [30, 41.8, 60]:
        theta_i = np.radians(theta_i_deg)

        # Incident ray (coming from below)
        x_inc = [2 * np.sin(theta_i), 0]
        y_inc = [2 * np.cos(theta_i), 0]
        ax.plot(x_inc, y_inc, 'r-', linewidth=2)

        if theta_i < theta_c:
            # Refracted ray exists
            theta_t = np.arcsin(n1 * np.sin(theta_i) / n2)
            x_ref = [0, -2 * np.sin(theta_t)]
            y_ref = [0, -2 * np.cos(theta_t)]
            ax.plot(x_ref, y_ref, 'b-', linewidth=2)
        else:
            # Total internal reflection
            x_ref = [0, -2 * np.sin(theta_i)]
            y_ref = [0, 2 * np.cos(theta_i)]
            ax.plot(x_ref, y_ref, 'g-', linewidth=2)

            if theta_i_deg == 60:
                # Show evanescent wave
                x_evan = np.linspace(0, 1, 50)
                y_evan = -0.3 * np.exp(-3 * x_evan)
                ax.plot(-x_evan, y_evan, 'm--', linewidth=1, alpha=0.7)
                ax.text(-0.5, -0.5, 'Evanescent\nwave', fontsize=9, color='purple')

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Critical angle annotation
    ax.annotate(f'Critical angle: {np.degrees(theta_c):.1f}°',
               xy=(0, 0), xytext=(1.5, 1.5),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='orange'))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Total Internal Reflection\n(glass to air)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('snells_law.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 2: Fresnel Equations
# =============================================================================

def plot_fresnel_equations():
    """Plot Fresnel coefficients vs angle."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========== External reflection (air to glass) ==========
    n1, n2 = 1.0, 1.5

    theta_i = np.linspace(0, np.pi/2 - 0.01, 500)
    theta_t = np.arcsin(n1 * np.sin(theta_i) / n2)

    # Fresnel coefficients
    r_s = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
    r_p = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))

    R_s = np.abs(r_s)**2
    R_p = np.abs(r_p)**2

    # Brewster's angle
    theta_B = np.arctan(n2 / n1)

    ax = axes[0, 0]
    ax.plot(np.degrees(theta_i), R_s, 'b-', linewidth=2, label=r'$R_s$ (s-pol)')
    ax.plot(np.degrees(theta_i), R_p, 'r-', linewidth=2, label=r'$R_p$ (p-pol)')
    ax.axvline(x=np.degrees(theta_B), color='g', linestyle='--',
               label=f'Brewster angle = {np.degrees(theta_B):.1f}°')
    ax.set_xlabel('Angle of Incidence (degrees)')
    ax.set_ylabel('Reflectance R')
    ax.set_title(f'External Reflection (n₁={n1} → n₂={n2})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.plot(np.degrees(theta_i), r_s, 'b-', linewidth=2, label=r'$r_s$')
    ax.plot(np.degrees(theta_i), r_p, 'r-', linewidth=2, label=r'$r_p$')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=np.degrees(theta_B), color='g', linestyle='--')
    ax.set_xlabel('Angle of Incidence (degrees)')
    ax.set_ylabel('Amplitude Reflection Coefficient')
    ax.set_title('Amplitude Coefficients\n(note sign change at Brewster)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)

    # ========== Internal reflection (glass to air) ==========
    n1, n2 = 1.5, 1.0
    theta_c = np.arcsin(n2 / n1)

    # Before critical angle
    theta_i_1 = np.linspace(0, theta_c - 0.01, 200)
    theta_t_1 = np.arcsin(n1 * np.sin(theta_i_1) / n2)

    r_s_1 = (n1 * np.cos(theta_i_1) - n2 * np.cos(theta_t_1)) / (n1 * np.cos(theta_i_1) + n2 * np.cos(theta_t_1))
    r_p_1 = (n2 * np.cos(theta_i_1) - n1 * np.cos(theta_t_1)) / (n2 * np.cos(theta_i_1) + n1 * np.cos(theta_t_1))

    R_s_1 = np.abs(r_s_1)**2
    R_p_1 = np.abs(r_p_1)**2

    # After critical angle (total internal reflection)
    theta_i_2 = np.linspace(theta_c + 0.01, np.pi/2 - 0.01, 200)
    R_s_2 = np.ones_like(theta_i_2)  # Total reflection
    R_p_2 = np.ones_like(theta_i_2)

    ax = axes[1, 0]
    ax.plot(np.degrees(theta_i_1), R_s_1, 'b-', linewidth=2, label=r'$R_s$')
    ax.plot(np.degrees(theta_i_1), R_p_1, 'r-', linewidth=2, label=r'$R_p$')
    ax.plot(np.degrees(theta_i_2), R_s_2, 'b-', linewidth=2)
    ax.plot(np.degrees(theta_i_2), R_p_2, 'r-', linewidth=2)
    ax.axvline(x=np.degrees(theta_c), color='purple', linestyle='--',
               label=f'Critical angle = {np.degrees(theta_c):.1f}°')
    ax.fill_between([np.degrees(theta_c), 90], 0, 1, alpha=0.2, color='yellow',
                   label='Total internal reflection')
    ax.set_xlabel('Angle of Incidence (degrees)')
    ax.set_ylabel('Reflectance R')
    ax.set_title(f'Internal Reflection (n₁={n1} → n₂={n2})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.1)

    # ========== Normal incidence vs n ==========
    ax = axes[1, 1]

    n = np.linspace(1, 4, 100)
    R_normal = ((1 - n) / (1 + n))**2

    ax.plot(n, R_normal * 100, 'b-', linewidth=2)
    ax.set_xlabel('Refractive Index n')
    ax.set_ylabel('Reflectance at Normal Incidence (%)')
    ax.set_title('Normal Incidence Reflectance vs n\n' + r'$R = \left(\frac{n-1}{n+1}\right)^2$')
    ax.grid(True, alpha=0.3)

    # Mark common materials
    materials = [(1.5, 'Glass'), (2.4, 'Diamond'), (3.5, 'Silicon')]
    for n_mat, name in materials:
        R_mat = ((1 - n_mat) / (1 + n_mat))**2 * 100
        ax.plot(n_mat, R_mat, 'ro', markersize=8)
        ax.annotate(f'{name}\n({R_mat:.1f}%)', xy=(n_mat, R_mat),
                   xytext=(n_mat + 0.1, R_mat + 3), fontsize=9)

    plt.tight_layout()
    plt.savefig('fresnel_equations.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 3: Skin Depth in Conductors
# =============================================================================

def plot_skin_depth():
    """Analyze skin effect in conductors."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Conductivities
    conductors = {
        'Copper': 5.96e7,
        'Aluminum': 3.77e7,
        'Iron': 1.03e7,
        'Seawater': 4.0
    }

    # ========== Skin depth vs frequency ==========
    ax = axes[0, 0]

    freq = np.logspace(0, 12, 500)
    omega = 2 * np.pi * freq

    for name, sigma in conductors.items():
        delta = np.sqrt(2 / (mu_0 * sigma * omega))
        ax.loglog(freq, delta * 1000, linewidth=2, label=name)  # mm

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Skin Depth (mm)')
    ax.set_title('Skin Depth vs Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Mark common frequencies
    for f, name in [(60, '60 Hz'), (1e6, '1 MHz'), (1e9, '1 GHz')]:
        ax.axvline(x=f, color='gray', linestyle='--', alpha=0.5)
        ax.text(f, 1e3, name, fontsize=9, rotation=90, va='bottom')

    # ========== Field penetration ==========
    ax = axes[0, 1]

    # For copper at different frequencies
    sigma_Cu = 5.96e7
    freqs = [60, 1e3, 1e6, 1e9]

    z = np.linspace(0, 100, 500)  # mm

    for f in freqs:
        omega = 2 * np.pi * f
        delta = np.sqrt(2 / (mu_0 * sigma_Cu * omega)) * 1000  # mm
        E_ratio = np.exp(-z / delta)
        ax.plot(z, E_ratio, linewidth=2, label=f'f = {f:.0e} Hz (δ = {delta:.2g} mm)')

    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel('E/E₀')
    ax.set_title('Field Penetration in Copper')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # ========== AC resistance vs frequency ==========
    ax = axes[1, 0]

    # For a copper wire of radius a = 1 mm
    a = 1e-3  # 1 mm radius
    sigma = 5.96e7
    freq = np.logspace(0, 9, 500)

    # DC resistance per meter
    R_dc = 1 / (sigma * np.pi * a**2)

    # AC resistance (approximate for δ << a)
    omega = 2 * np.pi * freq
    delta = np.sqrt(2 / (mu_0 * sigma * omega))

    # Current flows in annulus of thickness δ
    # For δ << a: R_ac ≈ R_dc * a / (2δ)
    # For δ >> a: R_ac ≈ R_dc
    R_ratio = np.where(delta < a, a / (2 * delta), 1)

    ax.semilogx(freq, R_ratio, 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('R_AC / R_DC')
    ax.set_title(f'AC Resistance Increase (1mm radius Cu wire)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 50)

    # ========== Complex wave vector ==========
    ax = axes[1, 1]

    # In a good conductor
    z = np.linspace(0, 3, 500)  # In units of skin depth
    Re_E = np.exp(-z) * np.cos(z)
    Im_E = np.exp(-z) * np.sin(z)
    envelope = np.exp(-z)

    ax.plot(z, Re_E, 'b-', linewidth=2, label='Re(E)')
    ax.plot(z, envelope, 'r--', linewidth=2, label='Envelope $e^{-z/δ}$')
    ax.plot(z, -envelope, 'r--', linewidth=2)
    ax.fill_between(z, envelope, -envelope, alpha=0.1, color='red')

    ax.set_xlabel('Depth (in units of δ)')
    ax.set_ylabel('E/E₀')
    ax.set_title('Wave in Conductor: Damped Oscillation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('skin_depth.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 4: Dispersion
# =============================================================================

def plot_dispersion():
    """Visualize dispersion and the Lorentz model."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Lorentz model parameters
    omega_0 = 1.0  # Resonance frequency (normalized)
    gamma = 0.1    # Damping
    omega_p = 0.5  # Plasma frequency contribution

    omega = np.linspace(0.01, 2.5, 1000)

    # Complex dielectric function
    epsilon = 1 + omega_p**2 / (omega_0**2 - omega**2 - 1j * gamma * omega)

    n_complex = np.sqrt(epsilon)
    n = np.real(n_complex)
    kappa = np.imag(n_complex)

    # ========== Real and imaginary parts of epsilon ==========
    ax = axes[0, 0]
    ax.plot(omega, np.real(epsilon), 'b-', linewidth=2, label=r"$\epsilon'$ (real)")
    ax.plot(omega, np.imag(epsilon), 'r-', linewidth=2, label=r"$\epsilon''$ (imaginary)")
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=omega_0, color='g', linestyle='--', alpha=0.5, label=r'$\omega_0$ (resonance)')
    ax.set_xlabel(r'$\omega/\omega_0$')
    ax.set_ylabel(r'$\epsilon$')
    ax.set_title('Lorentz Model: Dielectric Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, 5)

    # ========== Refractive index and extinction ==========
    ax = axes[0, 1]
    ax.plot(omega, n, 'b-', linewidth=2, label='n (refractive index)')
    ax.plot(omega, kappa, 'r-', linewidth=2, label=r'$\kappa$ (extinction)')
    ax.axvline(x=omega_0, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\omega/\omega_0$')
    ax.set_ylabel('n, κ')
    ax.set_title('Complex Refractive Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    # ========== Phase and group velocity ==========
    ax = axes[1, 0]

    # Phase velocity: v_p = c/n
    v_p = 1 / n  # In units of c

    # Group velocity: v_g = dω/dk = c / (n + ω dn/dω)
    dn_domega = np.gradient(n, omega)
    v_g = 1 / (n + omega * dn_domega)

    ax.plot(omega, v_p, 'b-', linewidth=2, label=r'$v_p$ (phase velocity)')
    ax.plot(omega, v_g, 'r-', linewidth=2, label=r'$v_g$ (group velocity)')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='c')
    ax.axvline(x=omega_0, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\omega/\omega_0$')
    ax.set_ylabel('Velocity (units of c)')
    ax.set_title('Phase and Group Velocities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    # ========== Absorption coefficient ==========
    ax = axes[1, 1]

    # Absorption coefficient α = 2ωκ/c
    alpha = 2 * omega * kappa  # Normalized

    ax.plot(omega, alpha, 'r-', linewidth=2)
    ax.axvline(x=omega_0, color='g', linestyle='--', alpha=0.5, label=r'$\omega_0$')
    ax.fill_between(omega, 0, alpha, alpha=0.3, color='red')
    ax.set_xlabel(r'$\omega/\omega_0$')
    ax.set_ylabel(r'Absorption $\alpha$ (arb. units)')
    ax.set_title('Absorption Spectrum\n(peaks at resonance)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dispersion.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 5: Prism Dispersion
# =============================================================================

def visualize_prism_dispersion():
    """Visualize dispersion in a prism (rainbow effect)."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Prism vertices
    prism = np.array([[0, 0], [2, 0], [1, np.sqrt(3)], [0, 0]])
    ax.plot(prism[:, 0], prism[:, 1], 'k-', linewidth=2)
    ax.fill(prism[:-1, 0], prism[:-1, 1], alpha=0.2, color='lightblue')

    # Incident white light
    ax.annotate('', xy=(0.5, np.sqrt(3)/2), xytext=(-0.5, np.sqrt(3)/2 + 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=3))
    ax.text(-0.7, np.sqrt(3)/2 + 0.6, 'White light', fontsize=12)

    # Sellmeier equation for BK7 glass (approximate)
    def refractive_index(wavelength_um):
        # Simplified Cauchy equation
        n = 1.5 + 0.004 / wavelength_um**2
        return n

    # Wavelengths and colors (visible spectrum)
    spectrum = [
        (0.38, 'violet'),
        (0.45, 'blue'),
        (0.50, 'cyan'),
        (0.55, 'green'),
        (0.58, 'yellow'),
        (0.62, 'orange'),
        (0.70, 'red')
    ]

    # Entry point and direction
    entry_point = np.array([0.5, np.sqrt(3)/2])
    entry_angle = np.radians(60)  # Angle of incidence

    for wavelength, color in spectrum:
        n = refractive_index(wavelength)

        # First refraction (entering prism)
        theta1 = np.radians(30)  # Angle to prism face normal
        theta2 = np.arcsin(np.sin(theta1) / n)

        # Path through prism (simplified)
        exit_x = 1.5
        exit_y = 0.1

        # Second refraction (exiting prism)
        # Different angles for different colors
        deviation = (n - 1.5) * 20  # Simplified deviation

        # Draw refracted rays
        ax.plot([entry_point[0], exit_x], [entry_point[1], exit_y], color=color, linewidth=2)

        # Exiting rays with dispersion
        exit_length = 1.5
        exit_angle = np.radians(30 + deviation * 100)
        end_x = exit_x + exit_length * np.cos(-exit_angle)
        end_y = exit_y - exit_length * np.sin(-exit_angle)
        ax.plot([exit_x, end_x], [exit_y, end_y], color=color, linewidth=2)

        # Labels
        ax.text(end_x + 0.1, end_y, f'{int(wavelength*1000)} nm', fontsize=9, color=color)

    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Prism Dispersion: White Light → Rainbow\n(shorter wavelengths bend more)',
                fontsize=14)

    plt.tight_layout()
    plt.savefig('prism_dispersion.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Day 216: Electromagnetic Waves in Matter")
    print("="*60)

    print("\n1. Snell's Law Visualization")
    visualize_snells_law()

    print("\n2. Fresnel Equations")
    plot_fresnel_equations()

    print("\n3. Skin Depth in Conductors")
    plot_skin_depth()

    print("\n4. Dispersion and Lorentz Model")
    plot_dispersion()

    print("\n5. Prism Dispersion")
    visualize_prism_dispersion()

    print("\nAll visualizations complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Wave speed in medium | $v = c/n = 1/\sqrt{\mu\epsilon}$ |
| Refractive index | $n = \sqrt{\mu_r\epsilon_r} \approx \sqrt{\epsilon_r}$ |
| Snell's law | $n_1\sin\theta_1 = n_2\sin\theta_2$ |
| Critical angle | $\sin\theta_c = n_2/n_1$ |
| Brewster's angle | $\tan\theta_B = n_2/n_1$ |
| Normal reflectance | $R = [(n_1-n_2)/(n_1+n_2)]^2$ |
| Skin depth | $\delta = \sqrt{2/(\mu\sigma\omega)}$ |
| Complex refractive index | $\tilde{n} = n + i\kappa$ |
| Beer's law | $I = I_0 e^{-\alpha z}$ |

### Main Takeaways

1. **Light slows down** in dielectrics by factor $n$ (refractive index)
2. **Snell's law** follows from boundary conditions on E and B
3. **Total internal reflection** occurs when going from high to low $n$
4. **Brewster's angle** gives zero reflection for p-polarization
5. **In conductors**, waves are attenuated with skin depth $\delta \propto 1/\sqrt{f}$
6. **Dispersion** (frequency-dependent $n$) causes colors to separate

## Daily Checklist

- [ ] I can derive the wave equation in dielectric media
- [ ] I can apply Snell's law and find critical angles
- [ ] I can use Fresnel equations to calculate reflectance
- [ ] I understand Brewster's angle and its applications
- [ ] I can calculate skin depth in conductors
- [ ] I understand dispersion and its origins
- [ ] I completed the computational lab

## Preview: Day 217

Tomorrow we consolidate the week's learning with a comprehensive review of Maxwell's equations and electromagnetic waves. We'll work through integrative problems, explore historical context, and prepare for the upcoming topics in radiation and optics.

---

*"The velocity of transverse undulations in our hypothetical medium... agrees so exactly with the velocity of light... that we can scarcely avoid the inference that light consists in the transverse undulations of the same medium."* — James Clerk Maxwell
