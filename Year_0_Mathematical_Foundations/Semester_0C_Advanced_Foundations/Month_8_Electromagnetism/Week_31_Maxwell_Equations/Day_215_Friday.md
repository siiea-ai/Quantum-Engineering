# Day 215: Energy, Momentum, and the Poynting Vector

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning I | 2 hrs | Energy density and the Poynting vector |
| Morning II | 2 hrs | Momentum of electromagnetic fields and radiation pressure |
| Afternoon | 2 hrs | Problem solving and applications |
| Evening | 2 hrs | Computational lab: Energy flow visualization |

## Learning Objectives

By the end of today, you will be able to:

1. **Calculate energy density** stored in electric and magnetic fields
2. **Apply the Poynting vector** to determine electromagnetic energy flow
3. **Derive the Poynting theorem** (conservation of electromagnetic energy)
4. **Calculate momentum density** and radiation pressure
5. **Analyze practical applications** of radiation pressure (solar sails, optical tweezers)
6. **Connect classical EM energy** to photon energy and quantum fluctuations

## Core Content

### 1. Energy Stored in Electric Fields

The energy stored in an electric field configuration:
$$U_E = \frac{1}{2}\int \epsilon_0 E^2 \, dV$$

The **electric energy density** (energy per unit volume):
$$\boxed{u_E = \frac{1}{2}\epsilon_0 E^2}$$

For a parallel-plate capacitor with field $E = V/d$:
$$U = \frac{1}{2}\epsilon_0 E^2 \cdot Ad = \frac{1}{2}\epsilon_0 \frac{V^2}{d^2} \cdot Ad = \frac{1}{2}\frac{\epsilon_0 A}{d}V^2 = \frac{1}{2}CV^2$$

This confirms our earlier capacitor energy formula.

### 2. Energy Stored in Magnetic Fields

Similarly, the energy stored in a magnetic field:
$$U_B = \frac{1}{2\mu_0}\int B^2 \, dV$$

The **magnetic energy density**:
$$\boxed{u_B = \frac{1}{2\mu_0}B^2}$$

For an inductor with field $B = \mu_0 n I$:
$$U = \frac{1}{2\mu_0}(\mu_0 n I)^2 \cdot Al = \frac{1}{2}\mu_0 n^2 A l \cdot I^2 = \frac{1}{2}LI^2$$

### 3. Total Electromagnetic Energy Density

The total energy density in the electromagnetic field:
$$\boxed{u = u_E + u_B = \frac{1}{2}\epsilon_0 E^2 + \frac{1}{2\mu_0}B^2}$$

**For electromagnetic waves:** Since $E = cB$ and $c = 1/\sqrt{\mu_0\epsilon_0}$:
$$u_E = \frac{1}{2}\epsilon_0 E^2$$
$$u_B = \frac{1}{2\mu_0}\left(\frac{E}{c}\right)^2 = \frac{\mu_0\epsilon_0}{2\mu_0}E^2 = \frac{1}{2}\epsilon_0 E^2$$

$$\boxed{u_E = u_B \text{ (equal partition of energy in EM waves)}}$$

Total energy density:
$$u = \epsilon_0 E^2 = \frac{1}{\mu_0}B^2$$

### 4. The Poynting Vector

Energy flows through electromagnetic fields. The **Poynting vector** describes this energy flux:

$$\boxed{\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}}$$

**Physical meaning:**
- $|\vec{S}|$ = power per unit area (W/m²)
- Direction = direction of energy flow

**For a plane wave:**
$$\vec{E} = E_0\cos(kz - \omega t)\hat{x}$$
$$\vec{B} = \frac{E_0}{c}\cos(kz - \omega t)\hat{y}$$

$$\vec{S} = \frac{1}{\mu_0}E_0\frac{E_0}{c}\cos^2(kz - \omega t)(\hat{x} \times \hat{y})$$

$$\vec{S} = \frac{E_0^2}{\mu_0 c}\cos^2(kz - \omega t)\hat{z} = c\epsilon_0 E_0^2\cos^2(kz - \omega t)\hat{z}$$

### 5. Intensity and Time-Average

The **intensity** (irradiance) is the time-averaged power per unit area:

$$I = \langle S \rangle = \frac{E_0^2}{2\mu_0 c} = \frac{1}{2}c\epsilon_0 E_0^2 = \frac{1}{2}\sqrt{\frac{\epsilon_0}{\mu_0}}E_0^2$$

Using $E_0 = cB_0$:
$$\boxed{I = \frac{1}{2}\frac{c}{\mu_0}B_0^2 = \frac{c}{2\mu_0}B_0^2}$$

The factor of 1/2 comes from $\langle\cos^2\rangle = 1/2$.

### 6. The Poynting Theorem (Energy Conservation)

The Poynting theorem expresses conservation of electromagnetic energy:

$$\boxed{\frac{\partial u}{\partial t} + \nabla \cdot \vec{S} = -\vec{J} \cdot \vec{E}}$$

**Interpretation:**
- $\partial u/\partial t$: Rate of change of energy density
- $\nabla \cdot \vec{S}$: Energy flowing out (per unit volume)
- $-\vec{J} \cdot \vec{E}$: Work done on charges (negative = energy transferred to mechanical form)

**Integral form:**
$$\frac{d}{dt}\int_V u \, dV + \oint_S \vec{S} \cdot d\vec{A} = -\int_V \vec{J} \cdot \vec{E} \, dV$$

Energy stored in V + Energy flowing out through S = Work done on currents

### 7. Momentum of Electromagnetic Fields

EM fields carry **momentum** as well as energy. The momentum density:
$$\boxed{\vec{g} = \frac{\vec{S}}{c^2} = \epsilon_0(\vec{E} \times \vec{B}) = \frac{1}{\mu_0 c^2}\vec{E} \times \vec{B}}$$

For a wave with intensity $I$, the momentum flux (momentum per unit time per unit area):
$$\text{Momentum flux} = \frac{I}{c}$$

The ratio of momentum to energy: $p/U = 1/c$, which matches the photon relation $p = E/c$.

### 8. Radiation Pressure

When EM radiation hits a surface, it exerts **radiation pressure**.

**Perfect absorber:**
$$P_{rad} = \frac{I}{c}$$

**Perfect reflector:**
$$P_{rad} = \frac{2I}{c}$$

(Factor of 2 because momentum reverses direction)

**Example: Sunlight**
Solar intensity at Earth: $I \approx 1400$ W/m²
$$P_{rad} = \frac{1400}{3 \times 10^8} \approx 4.7 \times 10^{-6} \text{ Pa}$$

This is tiny compared to atmospheric pressure (~10⁵ Pa) but significant for:
- Solar sails in space
- Comet tails (pointing away from Sun)
- Stellar formation

### 9. Angular Momentum of Light

Circularly polarized light carries **angular momentum**:
$$L = \pm \frac{U}{\omega}$$

per photon: $l = \pm\hbar$ (spin angular momentum)

This can exert torque on matter and is the basis for:
- Optical tweezers
- Optical spanners
- Angular momentum transfer in quantum optics

## Quantum Mechanics Connection

### Photon Energy and the Poynting Vector

For $N$ photons per unit volume with frequency $\omega$:
- Energy density: $u = N\hbar\omega$
- Momentum density: $g = N\hbar k = N\hbar\omega/c = u/c$

This matches the classical result! The Poynting vector describes the collective flow of photons.

### Quantum Fluctuations of Energy

Even in the vacuum state $|0\rangle$:
$$\langle 0|E^2|0\rangle = \frac{\hbar\omega}{2\epsilon_0 V}$$

per mode. The zero-point energy density is:
$$u_0 = \sum_{modes} \frac{\hbar\omega_k}{2V}$$

This diverges but can be regularized, leading to the **Casimir effect**.

### Photon Number Fluctuations

A coherent state $|\alpha\rangle$ (like a laser beam) has:
- Mean photon number: $\langle n \rangle = |\alpha|^2$
- Fluctuations: $\Delta n = \sqrt{\langle n \rangle}$ (shot noise)

Intensity fluctuations: $\Delta I/I = 1/\sqrt{\langle n \rangle}$

For $N$ photons: relative fluctuation ~ $1/\sqrt{N}$ (classical limit emerges for large $N$)

### Single Photon Energy

For a single photon in a mode:
$$u = \frac{\hbar\omega}{V}$$

The "electric field" of a single photon:
$$E_{single} \sim \sqrt{\frac{\hbar\omega}{\epsilon_0 V}}$$

This is the fundamental field scale in quantum optics.

## Worked Examples

### Example 1: Poynting Vector for a Plane Wave

A plane wave has $E_0 = 500$ V/m. Calculate: (a) $B_0$, (b) intensity, (c) radiation pressure on a mirror.

**Solution:**

**(a) Magnetic field amplitude:**
$$B_0 = \frac{E_0}{c} = \frac{500}{3 \times 10^8} = 1.67 \times 10^{-6} \text{ T}$$

**(b) Intensity:**
$$I = \frac{1}{2}\epsilon_0 c E_0^2 = \frac{1}{2}(8.854 \times 10^{-12})(3 \times 10^8)(500)^2$$
$$I = 332 \text{ W/m}^2$$

**(c) Radiation pressure on mirror:**
$$P = \frac{2I}{c} = \frac{2 \times 332}{3 \times 10^8} = 2.21 \times 10^{-6} \text{ Pa}$$

$$\boxed{P = 2.21\ \mu\text{Pa}}$$

### Example 2: Energy Flow Through a Resistor

A cylindrical resistor (length $L$, radius $a$, resistance $R$) carries current $I$. Show that the Poynting vector accounts for the power dissipation.

**Solution:**

**Fields at the surface:**
- E-field along the wire: $E = V/L = IR/L$
- B-field around the wire (from Ampère): $B = \mu_0 I/(2\pi a)$

**Poynting vector:**
Direction: $\vec{E} \times \vec{B}$ points radially **inward** (energy flows into the resistor!)

Magnitude:
$$S = \frac{EB}{\mu_0} = \frac{IR}{L} \cdot \frac{I}{2\pi a} = \frac{I^2 R}{2\pi a L}$$

**Total power flowing in:**
$$P = S \cdot (2\pi a L) = I^2 R$$

$$\boxed{P = I^2 R}$$

The energy enters through the sides of the wire, not along it! This is true for all conductors.

### Example 3: Solar Sail

A solar sail has area $A = 1000$ m² and total mass $m = 10$ kg. Calculate: (a) the force from sunlight, (b) the acceleration, (c) time to reach 1% of light speed starting from rest.

**Solution:**

Solar intensity: $I = 1400$ W/m²

**(a) Force (assuming perfect reflection):**
$$F = \frac{2I}{c} \cdot A = \frac{2 \times 1400}{3 \times 10^8} \times 1000 = 9.33 \times 10^{-3} \text{ N}$$

**(b) Acceleration:**
$$a = \frac{F}{m} = \frac{9.33 \times 10^{-3}}{10} = 9.33 \times 10^{-4} \text{ m/s}^2$$

**(c) Time to reach 0.01c:**
$$v = at \quad \Rightarrow \quad t = \frac{v}{a} = \frac{0.01 \times 3 \times 10^8}{9.33 \times 10^{-4}}$$
$$t = 3.22 \times 10^9 \text{ s} \approx 102 \text{ years}$$

$$\boxed{t \approx 102 \text{ years}}$$

(This assumes constant solar intensity, which isn't realistic far from the Sun.)

## Practice Problems

### Level 1: Direct Application

1. A laser beam has intensity 10⁵ W/m². Find: (a) the electric field amplitude, (b) the magnetic field amplitude, (c) the energy density.

2. Calculate the radiation pressure exerted by sunlight on a black (absorbing) surface at Earth's orbit.

3. An electromagnetic wave has $\vec{E} = 200\cos(kz - \omega t)\hat{x}$ V/m. Write the expression for $\vec{S}$.

### Level 2: Intermediate

4. Show that for a plane wave, the energy density $u$ and the Poynting magnitude $S$ are related by $S = cu$.

5. A 100 W light bulb radiates uniformly in all directions. At distance $r = 2$ m: (a) What is the intensity? (b) What are $E_0$ and $B_0$? (c) What is the radiation pressure on a small mirror?

6. Derive the Poynting theorem starting from Maxwell's equations.

### Level 3: Challenging

7. A coaxial cable has inner conductor radius $a$ and outer conductor radius $b$. Current $I$ flows in the inner conductor and returns in the outer. Find the Poynting vector and show the power transmitted equals $IV$.

8. A sphere of radius $R$ has uniform polarization $\vec{P} = P_0\hat{z}$. If the polarization oscillates as $\vec{P} = P_0\cos(\omega t)\hat{z}$, find the radiated power (assume $\omega R/c \ll 1$).

9. **Quantum connection:** A laser cavity of volume $V$ contains light at frequency $\omega$ with $N$ photons. Calculate: (a) the energy density, (b) the "classical" electric field amplitude, (c) the number of photons needed for the classical and single-photon fields to be equal.

## Computational Lab: Energy and Momentum Flow

```python
"""
Day 215 Computational Lab: Energy, Momentum, and Poynting Vector
Topics: Energy flow, radiation pressure, Poynting vector visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation

# Set up styling
plt.style.use('default')

# Physical constants
c = 3e8
epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7

# =============================================================================
# Part 1: Energy Density in EM Wave
# =============================================================================

def plot_energy_density():
    """Visualize energy density components in an electromagnetic wave."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Wave parameters
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    E_0 = 100  # V/m
    B_0 = E_0 / c

    # Spatial coordinate
    z = np.linspace(0, 2 * wavelength, 500)
    t = 0  # Snapshot at t=0

    # Fields
    E = E_0 * np.sin(k * z)
    B = B_0 * np.sin(k * z)

    # Energy densities
    u_E = 0.5 * epsilon_0 * E**2
    u_B = 0.5 * B**2 / mu_0
    u_total = u_E + u_B

    # Plot 1: Fields
    ax = axes[0, 0]
    ax.plot(z, E, 'b-', linewidth=2, label=r'$E$ (V/m)')
    ax.plot(z, B * c, 'r-', linewidth=2, label=r'$cB$ (V/m)')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Field Amplitude')
    ax.set_title('Electric and Magnetic Fields')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy densities
    ax = axes[0, 1]
    ax.plot(z, u_E * 1e6, 'b-', linewidth=2, label=r'$u_E = \frac{1}{2}\epsilon_0 E^2$')
    ax.plot(z, u_B * 1e6, 'r--', linewidth=2, label=r'$u_B = \frac{1}{2\mu_0}B^2$')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Energy Density (μJ/m³)')
    ax.set_title('Energy Density Components\n(Note: $u_E = u_B$ at all points!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Total energy density and Poynting vector
    ax = axes[1, 0]
    S = E * B / mu_0  # Poynting vector magnitude

    ax.plot(z, u_total * 1e6, 'g-', linewidth=2, label=r'$u = u_E + u_B$')
    ax.plot(z, S / c * 1e6, 'm--', linewidth=2, label=r'$S/c$ (check: $S = cu$)')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Energy Density (μJ/m³)')
    ax.set_title('Total Energy Density and Poynting Vector')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Intensity calculation
    ax = axes[1, 1]

    # Time average
    intensity = 0.5 * epsilon_0 * c * E_0**2

    ax.axhline(y=intensity, color='r', linewidth=2, label=f'I = {intensity:.2f} W/m²')
    ax.plot(z, c * u_total, 'b-', linewidth=1, alpha=0.5, label='Instantaneous $cu$')

    ax.set_xlabel('z (m)')
    ax.set_ylabel('Power Density (W/m²)')
    ax.set_title(r'Intensity: $I = \langle S \rangle = \frac{1}{2}\epsilon_0 c E_0^2$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2 * intensity)

    plt.tight_layout()
    plt.savefig('energy_density.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"E_0 = {E_0} V/m")
    print(f"B_0 = {B_0:.3e} T")
    print(f"Average intensity = {intensity:.2f} W/m²")
    print(f"Average energy density = {0.5 * epsilon_0 * E_0**2:.3e} J/m³")

# =============================================================================
# Part 2: Poynting Vector Visualization
# =============================================================================

def visualize_poynting_vector():
    """Visualize Poynting vector for various configurations."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ========== Case 1: Plane Wave ==========
    ax = axes[0]

    # Propagation direction with E and B
    ax.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(0.5, 0.55, r'$\vec{S}$ (energy flow)', fontsize=12, ha='center', color='green')

    ax.annotate('', xy=(0.3, 0.7), xytext=(0.3, 0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(0.35, 0.5, r'$\vec{E}$', fontsize=12, color='blue')

    ax.text(0.5, 0.5, '⊙', fontsize=20, ha='center', va='center', color='red')
    ax.text(0.55, 0.4, r'$\vec{B}$', fontsize=12, color='red')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Plane Wave\n' + r'$\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}$')

    # ========== Case 2: Energy into Resistor ==========
    ax = axes[1]

    # Draw resistor
    rect = plt.Rectangle((0.3, 0.3), 0.4, 0.4, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # E field (horizontal, along wire)
    ax.annotate('', xy=(0.65, 0.5), xytext=(0.35, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(0.5, 0.55, r'$\vec{E}$', fontsize=12, ha='center', color='blue')

    # B field (circles around wire - out of page on top)
    ax.text(0.5, 0.8, '⊙', fontsize=16, ha='center', va='center', color='red')
    ax.text(0.5, 0.2, '⊗', fontsize=16, ha='center', va='center', color='red')
    ax.text(0.55, 0.8, r'$\vec{B}$', fontsize=10, color='red')

    # Poynting vectors pointing IN
    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        x_start = 0.5 + 0.35 * np.cos(rad)
        y_start = 0.5 + 0.35 * np.sin(rad)
        dx = -0.1 * np.cos(rad)
        dy = -0.1 * np.sin(rad)
        ax.annotate('', xy=(x_start + dx, y_start + dy), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.text(0.9, 0.5, r'$\vec{S}$', fontsize=12, color='green')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Resistor: Energy Flows IN\nthrough the sides!')

    # ========== Case 3: Radiating Dipole ==========
    ax = axes[2]

    # Draw dipole
    ax.plot([0.5], [0.5], 'ko', markersize=10)
    ax.annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.4),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(0.55, 0.5, r'$\vec{p}$', fontsize=12, color='blue')

    # Poynting vectors radiating outward
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        # Skip along dipole axis (no radiation there)
        if np.abs(np.cos(angle)) < 0.9:
            x_start = 0.5 + 0.15 * np.cos(angle)
            y_start = 0.5 + 0.15 * np.sin(angle)
            # Magnitude varies as sin²θ
            mag = 0.25 * np.sin(angle)**2
            dx = mag * np.cos(angle)
            dy = mag * np.sin(angle)
            ax.annotate('', xy=(x_start + dx, y_start + dy), xytext=(x_start, y_start),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Radiating Dipole\nEnergy flows outward')

    plt.tight_layout()
    plt.savefig('poynting_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 3: Radiation Pressure Applications
# =============================================================================

def plot_radiation_pressure():
    """Analyze radiation pressure and its applications."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========== Solar Sail Performance ==========
    ax = axes[0, 0]

    # Sail parameters
    areas = np.linspace(100, 10000, 100)  # m²
    masses = np.array([1, 5, 10, 50, 100])  # kg

    I_sun = 1400  # W/m² at Earth orbit

    for m in masses:
        F = 2 * I_sun / c * areas  # Perfect reflection
        a = F / m
        ax.semilogy(areas, a * 1000, label=f'm = {m} kg')

    ax.set_xlabel('Sail Area (m²)')
    ax.set_ylabel('Acceleration (mm/s²)')
    ax.set_title('Solar Sail Acceleration vs Area')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== Distance Dependence ==========
    ax = axes[0, 1]

    distances = np.linspace(0.1, 5, 100)  # AU
    I_at_r = I_sun / distances**2  # Inverse square
    P_rad = 2 * I_at_r / c  # Radiation pressure

    ax.semilogy(distances, P_rad * 1e6, 'b-', linewidth=2)
    ax.axvline(x=1, color='g', linestyle='--', label='Earth orbit')
    ax.set_xlabel('Distance from Sun (AU)')
    ax.set_ylabel('Radiation Pressure (μPa)')
    ax.set_title('Radiation Pressure vs Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== Comparison with Gravity ==========
    ax = axes[1, 0]

    # For a spherical particle of radius r and density rho
    r_particle = np.logspace(-7, -4, 100)  # 0.1 μm to 0.1 mm
    rho = 2000  # kg/m³ (typical dust)

    # Radiation force
    A_particle = np.pi * r_particle**2
    F_rad = I_sun / c * A_particle  # Absorbing

    # Gravitational force (at 1 AU)
    M_sun = 2e30
    G = 6.67e-11
    r_orbit = 1.5e11  # 1 AU in meters
    m_particle = (4/3) * np.pi * r_particle**3 * rho
    F_grav = G * M_sun * m_particle / r_orbit**2

    ax.loglog(r_particle * 1e6, F_rad / F_grav, 'b-', linewidth=2)
    ax.axhline(y=1, color='r', linestyle='--', label='F_rad = F_grav')
    ax.set_xlabel('Particle Radius (μm)')
    ax.set_ylabel('F_radiation / F_gravity')
    ax.set_title('Radiation vs Gravity on Dust Particles\n(explains comet tails!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== Optical Tweezers Force ==========
    ax = axes[1, 1]

    # Typical laser parameters
    P_laser = np.linspace(1, 100, 100)  # mW
    w_0 = 1e-6  # Beam waist (1 μm)
    I_laser = P_laser * 1e-3 / (np.pi * w_0**2)  # W/m²

    # Gradient force (approximate)
    # F ~ (n-1) * gradient(I) / c
    # For small particle: F ~ P/c * (particle size / beam size)
    r_p = 0.5e-6  # 0.5 μm particle
    F_trap = I_laser * np.pi * r_p**2 / c * 1e12  # pN

    ax.plot(P_laser, F_trap, 'r-', linewidth=2)
    ax.set_xlabel('Laser Power (mW)')
    ax.set_ylabel('Trapping Force (pN)')
    ax.set_title('Optical Tweezers: Trapping Force')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('radiation_pressure_applications.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 4: Animated Energy Flow
# =============================================================================

def animate_energy_flow():
    """Animate Poynting vector and energy flow in a wave."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Wave parameters
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    omega = k * c
    E_0 = 100  # V/m

    z = np.linspace(0, 2 * wavelength, 200)

    # Left plot: Fields
    line_E, = axes[0].plot([], [], 'b-', linewidth=2, label='E')
    line_B, = axes[0].plot([], [], 'r--', linewidth=2, label='cB')
    axes[0].set_xlim(0, 2 * wavelength)
    axes[0].set_ylim(-1.2 * E_0, 1.2 * E_0)
    axes[0].set_xlabel('z (m)')
    axes[0].set_ylabel('Field (V/m)')
    axes[0].set_title('Electric and Magnetic Fields')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Right plot: Energy flow (Poynting vector)
    quiver = axes[1].quiver([], [], [], [], scale=1)
    axes[1].set_xlim(0, 2 * wavelength)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_xlabel('z (m)')
    axes[1].set_ylabel('y')
    axes[1].set_title('Poynting Vector (Energy Flow)')

    time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes)

    def init():
        line_E.set_data([], [])
        line_B.set_data([], [])
        time_text.set_text('')
        return line_E, line_B, time_text

    def animate(frame):
        t = frame * 2 * np.pi / (omega * 50)  # One period over 50 frames

        E = E_0 * np.sin(k * z - omega * t)
        B = E_0 / c * np.sin(k * z - omega * t)

        line_E.set_data(z, E)
        line_B.set_data(z, c * B)

        # Poynting vector (sampled points)
        z_sample = z[::10]
        E_sample = E[::10]
        S_sample = E_sample**2 / (mu_0 * c)  # Normalized

        axes[1].clear()
        axes[1].quiver(z_sample, np.zeros_like(z_sample),
                      S_sample / np.max(np.abs(S_sample) + 1e-10), np.zeros_like(z_sample),
                      scale=10, color='green', width=0.01)
        axes[1].set_xlim(0, 2 * wavelength)
        axes[1].set_ylim(-0.3, 0.3)
        axes[1].set_xlabel('z (m)')
        axes[1].set_ylabel('y')
        axes[1].set_title('Poynting Vector (Energy Flow →)')

        time_text.set_text(f'ωt = {omega * t:.2f} rad')

        return line_E, line_B, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=50,
                        interval=100, blit=False)

    plt.tight_layout()
    plt.savefig('energy_flow_snapshot.png', dpi=150, bbox_inches='tight')
    plt.show()

    return anim

# =============================================================================
# Part 5: Photon Statistics
# =============================================================================

def analyze_photon_statistics():
    """Compare classical wave energy with photon picture."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Physical constants
    h = 6.626e-34
    hbar = h / (2 * np.pi)

    # ========== Photon Number vs Intensity ==========
    ax = axes[0]

    # For 500 nm light
    wavelength = 500e-9
    f = c / wavelength
    E_photon = h * f

    # Intensity range
    I = np.logspace(-10, 6, 100)  # W/m²

    # Photon flux (photons per m² per second)
    flux = I / E_photon

    ax.loglog(I, flux, 'b-', linewidth=2)
    ax.set_xlabel('Intensity (W/m²)')
    ax.set_ylabel('Photon Flux (photons/m²/s)')
    ax.set_title(f'Photon Flux vs Intensity\n(λ = {wavelength*1e9:.0f} nm, E_photon = {E_photon/1.6e-19:.2f} eV)')
    ax.grid(True, alpha=0.3, which='both')

    # Mark typical intensities
    typical = [(1400, 'Sunlight'), (0.01, 'Moonlight'), (1e-6, 'Starlight')]
    for I_val, name in typical:
        ax.axvline(x=I_val, color='r', linestyle='--', alpha=0.5)
        ax.text(I_val, 1e25, name, rotation=90, va='bottom', fontsize=9)

    # ========== Shot Noise ==========
    ax = axes[1]

    N = np.logspace(0, 10, 100)  # Number of photons
    relative_fluctuation = 1 / np.sqrt(N)

    ax.loglog(N, relative_fluctuation * 100, 'r-', linewidth=2)
    ax.set_xlabel('Number of Photons N')
    ax.set_ylabel('Relative Intensity Fluctuation (%)')
    ax.set_title('Shot Noise: ΔI/I = 1/√N')
    ax.grid(True, alpha=0.3, which='both')

    ax.axhline(y=1, color='g', linestyle='--', label='1% fluctuation')
    ax.axhline(y=0.1, color='b', linestyle='--', label='0.1% fluctuation')
    ax.legend()

    plt.tight_layout()
    plt.savefig('photon_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Photon energy at 500 nm: {E_photon:.3e} J = {E_photon/1.6e-19:.2f} eV")
    print(f"Photons in 1 W beam: {1/E_photon:.3e} per second")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Day 215: Energy, Momentum, and Poynting Vector")
    print("="*60)

    print("\n1. Energy Density in EM Wave")
    plot_energy_density()

    print("\n2. Poynting Vector Examples")
    visualize_poynting_vector()

    print("\n3. Radiation Pressure Applications")
    plot_radiation_pressure()

    print("\n4. Animated Energy Flow")
    anim = animate_energy_flow()

    print("\n5. Photon Statistics")
    analyze_photon_statistics()

    print("\nAll visualizations complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Electric energy density | $u_E = \frac{1}{2}\epsilon_0 E^2$ |
| Magnetic energy density | $u_B = \frac{1}{2\mu_0}B^2$ |
| Poynting vector | $\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}$ |
| Intensity | $I = \frac{1}{2}\epsilon_0 c E_0^2$ |
| Momentum density | $\vec{g} = \vec{S}/c^2 = \epsilon_0(\vec{E} \times \vec{B})$ |
| Radiation pressure (absorbing) | $P = I/c$ |
| Radiation pressure (reflecting) | $P = 2I/c$ |
| Poynting theorem | $\frac{\partial u}{\partial t} + \nabla \cdot \vec{S} = -\vec{J} \cdot \vec{E}$ |

### Main Takeaways

1. **Energy is equally divided** between E and B fields in electromagnetic waves
2. **The Poynting vector** $\vec{S}$ gives direction and magnitude of energy flow
3. **Energy flows into resistors** from the sides, not along the wires
4. **EM waves carry momentum** with density $g = S/c^2$
5. **Radiation pressure** is small but measurable (solar sails, comet tails)
6. **Classical intensity** corresponds to photon flux times photon energy

## Daily Checklist

- [ ] I can calculate energy density in E and B fields
- [ ] I can compute and interpret the Poynting vector
- [ ] I understand the Poynting theorem and energy conservation
- [ ] I can calculate radiation pressure for absorbing and reflecting surfaces
- [ ] I understand practical applications (solar sails, optical tweezers)
- [ ] I can connect wave intensity to photon flux
- [ ] I completed the computational lab

## Preview: Day 216

Tomorrow we explore **electromagnetic waves in matter**—dielectrics, conductors, and plasmas. We'll see how materials affect wave speed, leading to refraction, and how conductors attenuate waves, creating the skin effect. The Fresnel equations will describe reflection and transmission at interfaces.

---

*"Energy cannot be created or destroyed, but it can be transformed from one form to another. Electromagnetic energy flows through space, carrying momentum as it goes."* — Classical wisdom
