# Day 214: Electromagnetic Waves in Vacuum

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning I | 2 hrs | Derivation of the wave equation from Maxwell's equations |
| Morning II | 2 hrs | Plane wave solutions and polarization |
| Afternoon | 2 hrs | Problem solving with EM waves |
| Evening | 2 hrs | Computational lab: Wave visualization and animation |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive the electromagnetic wave equation** from Maxwell's equations
2. **Calculate the speed of light** from fundamental electromagnetic constants
3. **Write plane wave solutions** for E and B fields
4. **Determine the relationship** between E, B, and propagation direction
5. **Describe polarization states** including linear, circular, and elliptical
6. **Connect classical EM waves** to the quantum description of photons

## Core Content

### 1. Derivation of the Wave Equation

Starting from Maxwell's equations in vacuum ($\rho = 0$, $\vec{J} = 0$):
$$\nabla \cdot \vec{E} = 0 \qquad \nabla \cdot \vec{B} = 0$$
$$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t} \qquad \nabla \times \vec{B} = \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}$$

**Step 1:** Take the curl of Faraday's law:
$$\nabla \times (\nabla \times \vec{E}) = -\frac{\partial}{\partial t}(\nabla \times \vec{B})$$

**Step 2:** Use the vector identity:
$$\nabla \times (\nabla \times \vec{E}) = \nabla(\nabla \cdot \vec{E}) - \nabla^2 \vec{E}$$

Since $\nabla \cdot \vec{E} = 0$ in vacuum:
$$-\nabla^2 \vec{E} = -\frac{\partial}{\partial t}(\nabla \times \vec{B})$$

**Step 3:** Substitute Ampère-Maxwell:
$$\nabla^2 \vec{E} = \frac{\partial}{\partial t}\left(\mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}\right)$$

$$\boxed{\nabla^2 \vec{E} = \mu_0 \epsilon_0 \frac{\partial^2 \vec{E}}{\partial t^2}}$$

Similarly for the magnetic field:
$$\boxed{\nabla^2 \vec{B} = \mu_0 \epsilon_0 \frac{\partial^2 \vec{B}}{\partial t^2}}$$

### 2. The Speed of Electromagnetic Waves

The wave equation has the standard form:
$$\nabla^2 \psi = \frac{1}{v^2}\frac{\partial^2 \psi}{\partial t^2}$$

Comparing, the wave speed is:
$$v = \frac{1}{\sqrt{\mu_0 \epsilon_0}}$$

Substituting the values:
$$\mu_0 = 4\pi \times 10^{-7} \text{ H/m}$$
$$\epsilon_0 = 8.854 \times 10^{-12} \text{ F/m}$$

$$\boxed{c = \frac{1}{\sqrt{\mu_0 \epsilon_0}} = 299,792,458 \text{ m/s}}$$

**This is the speed of light!** Maxwell's equations predict that electromagnetic waves travel at $c$.

### 3. Plane Wave Solutions

The simplest solutions are **plane waves**—waves with planar wavefronts.

For a wave propagating in the $+z$ direction:
$$\vec{E} = E_0 \cos(kz - \omega t + \phi)\hat{x}$$
$$\vec{B} = B_0 \cos(kz - \omega t + \phi)\hat{y}$$

Or in complex notation:
$$\tilde{\vec{E}} = \tilde{E}_0 e^{i(kz - \omega t)}\hat{x}$$
$$\tilde{\vec{B}} = \tilde{B}_0 e^{i(kz - \omega t)}\hat{y}$$

**Key parameters:**
- Wave vector magnitude: $k = 2\pi/\lambda$
- Angular frequency: $\omega = 2\pi f$
- **Dispersion relation:** $\omega = ck$, or $f\lambda = c$

### 4. Relationship Between E and B

From Faraday's law applied to our plane wave:
$$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$$

For $\vec{E} = E_0 e^{i(kz-\omega t)}\hat{x}$:
$$\frac{\partial E_x}{\partial z}\hat{y} = ikE_0 e^{i(kz-\omega t)}\hat{y} = -\frac{\partial B_y}{\partial t}\hat{y}$$

$$ikE_0 = i\omega B_0$$

$$\boxed{\frac{E_0}{B_0} = \frac{\omega}{k} = c}$$

**Important relationships:**
$$|\vec{E}| = c|\vec{B}|$$
$$\vec{E} \perp \vec{B} \perp \vec{k}$$
$$\vec{E} \times \vec{B} \parallel \vec{k}$$

The fields are perpendicular to each other and to the direction of propagation. This is a **transverse wave**.

### 5. General Plane Wave

For a plane wave propagating in direction $\hat{k}$:
$$\vec{E}(\vec{r}, t) = \vec{E}_0 e^{i(\vec{k}\cdot\vec{r} - \omega t)}$$
$$\vec{B}(\vec{r}, t) = \frac{1}{c}\hat{k} \times \vec{E}(\vec{r}, t) = \frac{\vec{k} \times \vec{E}}{\omega}$$

The relationship can be written as:
$$\vec{B} = \frac{1}{c}\hat{n} \times \vec{E}$$

where $\hat{n} = \vec{k}/k$ is the unit propagation vector.

### 6. Polarization States

The **polarization** of an EM wave describes the behavior of the electric field vector.

**Linear Polarization:**
The E-field oscillates in a fixed plane.
$$\vec{E} = E_0 \cos(kz - \omega t)\hat{x}$$

**Circular Polarization:**
The E-field rotates in a circle as the wave propagates.

Right-circular (clockwise when looking into beam):
$$\vec{E} = E_0[\cos(kz - \omega t)\hat{x} + \sin(kz - \omega t)\hat{y}]$$

Left-circular (counterclockwise):
$$\vec{E} = E_0[\cos(kz - \omega t)\hat{x} - \sin(kz - \omega t)\hat{y}]$$

In complex notation:
$$\tilde{\vec{E}}_{RCP} = E_0(\hat{x} - i\hat{y})e^{i(kz - \omega t)}$$
$$\tilde{\vec{E}}_{LCP} = E_0(\hat{x} + i\hat{y})e^{i(kz - \omega t)}$$

**Elliptical Polarization:**
The general case—E traces an ellipse.
$$\vec{E} = E_{0x}\cos(kz - \omega t)\hat{x} + E_{0y}\cos(kz - \omega t + \delta)\hat{y}$$

where $\delta$ is the phase difference.

### 7. The Electromagnetic Spectrum

EM waves span an enormous range of frequencies:

| Type | Frequency | Wavelength |
|------|-----------|------------|
| Radio | < 300 MHz | > 1 m |
| Microwave | 300 MHz - 300 GHz | 1 mm - 1 m |
| Infrared | 300 GHz - 400 THz | 750 nm - 1 mm |
| Visible | 400 - 800 THz | 380 - 750 nm |
| Ultraviolet | 800 THz - 30 PHz | 10 - 380 nm |
| X-rays | 30 PHz - 30 EHz | 0.01 - 10 nm |
| Gamma rays | > 30 EHz | < 0.01 nm |

All travel at speed $c$ in vacuum!

### 8. Standing Electromagnetic Waves

Two counter-propagating waves create a standing wave:
$$\vec{E} = E_0[\cos(kz - \omega t) + \cos(kz + \omega t)]\hat{x}$$
$$= 2E_0\cos(kz)\cos(\omega t)\hat{x}$$

The nodes (E = 0) are fixed at $z = (n + 1/2)\pi/k$.

Standing waves are essential for:
- Laser cavities
- Microwave ovens
- Electromagnetic resonators

## Quantum Mechanics Connection

### Photons: The Quanta of Light

Classical electromagnetic waves are quantized into **photons**:

**Energy:**
$$E = \hbar\omega = hf$$

**Momentum:**
$$p = \hbar k = \frac{h}{\lambda} = \frac{E}{c}$$

**Spin:** Photons have spin-1, with two helicity states (±ℏ along $\vec{k}$) corresponding to right/left circular polarization.

### Wave-Particle Duality

- **Wave behavior:** Diffraction, interference (Young's double slit)
- **Particle behavior:** Photoelectric effect, Compton scattering

The wave equation describes the **probability amplitude** for detecting a photon.

### Coherent States

Classical EM waves correspond to **coherent states** $|\alpha\rangle$ with:
- Mean photon number: $\langle n \rangle = |\alpha|^2$
- Poisson distribution of photon numbers
- Minimum uncertainty in phase and number

For a laser beam with power $P$ and frequency $\omega$:
$$\langle n \rangle = \frac{P}{\hbar\omega} \cdot \tau$$

where $\tau$ is the measurement time.

### Quantum Vacuum Fluctuations

Even in vacuum with no photons, the quantized field has zero-point fluctuations:
$$\langle 0|E^2|0 \rangle \neq 0$$

These fluctuations lead to:
- Casimir effect
- Spontaneous emission
- Lamb shift

## Worked Examples

### Example 1: Properties of a Plane Wave

An electromagnetic wave in vacuum has electric field:
$$\vec{E} = 500\cos(1.5 \times 10^{7}z - 4.5 \times 10^{15}t)\hat{x} \text{ V/m}$$

Find: (a) wavelength, (b) frequency, (c) B-field expression.

**Solution:**

Comparing with $\vec{E} = E_0\cos(kz - \omega t)\hat{x}$:
- $E_0 = 500$ V/m
- $k = 1.5 \times 10^7$ rad/m
- $\omega = 4.5 \times 10^{15}$ rad/s

**(a) Wavelength:**
$$\lambda = \frac{2\pi}{k} = \frac{2\pi}{1.5 \times 10^7} = 4.19 \times 10^{-7} \text{ m} = 419 \text{ nm}$$

(This is violet light!)

**(b) Frequency:**
$$f = \frac{\omega}{2\pi} = \frac{4.5 \times 10^{15}}{2\pi} = 7.16 \times 10^{14} \text{ Hz}$$

**(c) B-field:**

Direction: $\hat{k} \times \hat{E} = \hat{z} \times \hat{x} = \hat{y}$

Magnitude: $B_0 = E_0/c = 500/(3 \times 10^8) = 1.67 \times 10^{-6}$ T

$$\boxed{\vec{B} = 1.67 \times 10^{-6}\cos(1.5 \times 10^{7}z - 4.5 \times 10^{15}t)\hat{y} \text{ T}}$$

### Example 2: Circular Polarization

A circularly polarized wave has $E_0 = 100$ V/m and wavelength 600 nm. Write the E and B fields.

**Solution:**

Wave parameters:
$$k = \frac{2\pi}{\lambda} = \frac{2\pi}{600 \times 10^{-9}} = 1.05 \times 10^7 \text{ rad/m}$$
$$\omega = ck = (3 \times 10^8)(1.05 \times 10^7) = 3.14 \times 10^{15} \text{ rad/s}$$

For right-circular polarization:
$$\vec{E} = 100[\cos(kz - \omega t)\hat{x} + \sin(kz - \omega t)\hat{y}]$$

$$\vec{E} = 100[\cos(1.05 \times 10^7 z - 3.14 \times 10^{15} t)\hat{x} + \sin(1.05 \times 10^7 z - 3.14 \times 10^{15} t)\hat{y}] \text{ V/m}$$

For B-field: $\vec{B} = \frac{1}{c}\hat{z} \times \vec{E}$

$$\vec{B} = \frac{100}{c}[\cos(kz - \omega t)(\hat{z} \times \hat{x}) + \sin(kz - \omega t)(\hat{z} \times \hat{y})]$$

$$\boxed{\vec{B} = 3.33 \times 10^{-7}[\cos(kz - \omega t)\hat{y} - \sin(kz - \omega t)\hat{x}] \text{ T}}$$

### Example 3: Photon Properties

Calculate for a 500 nm photon: (a) energy in eV, (b) momentum, (c) number of photons per second in a 1 mW laser beam.

**Solution:**

**(a) Energy:**
$$E = hf = \frac{hc}{\lambda} = \frac{(6.626 \times 10^{-34})(3 \times 10^8)}{500 \times 10^{-9}}$$
$$E = 3.98 \times 10^{-19} \text{ J} = 2.49 \text{ eV}$$

**(b) Momentum:**
$$p = \frac{h}{\lambda} = \frac{6.626 \times 10^{-34}}{500 \times 10^{-9}} = 1.33 \times 10^{-27} \text{ kg·m/s}$$

Or: $p = E/c = 3.98 \times 10^{-19}/(3 \times 10^8) = 1.33 \times 10^{-27}$ kg·m/s ✓

**(c) Photon rate:**
$$\dot{N} = \frac{P}{E} = \frac{10^{-3}}{3.98 \times 10^{-19}} = 2.51 \times 10^{15} \text{ photons/s}$$

$$\boxed{\dot{N} \approx 2.5 \times 10^{15} \text{ photons/s}}$$

## Practice Problems

### Level 1: Direct Application

1. An EM wave has $\lambda = 2$ m. Find: (a) frequency, (b) period, (c) wave number.

2. The E-field amplitude of a wave is 300 V/m. Find the B-field amplitude.

3. Write the expression for a plane wave with $\lambda = 1$ μm traveling in the $-z$ direction, polarized along $\hat{y}$.

### Level 2: Intermediate

4. Show that $\vec{E} = E_0\sin(kx + \omega t)\hat{z}$ and $\vec{B} = B_0\sin(kx + \omega t)\hat{y}$ satisfy Maxwell's equations. What is the direction of propagation?

5. A linearly polarized wave passes through a polarizer oriented at 45° to its polarization. By what factor does the intensity decrease?

6. Two linearly polarized waves with the same amplitude but perpendicular polarizations and a 90° phase difference are superposed. Describe the resulting polarization state.

### Level 3: Challenging

7. Starting from Maxwell's equations, derive the wave equation for $\vec{B}$.

8. An elliptically polarized wave has $E_x = E_0\cos(\omega t)$ and $E_y = 2E_0\sin(\omega t)$ at $z = 0$. Find the ratio of the ellipse's semi-axes and the orientation.

9. **Quantum connection:** A single photon state $|1\rangle$ has electric field expectation $\langle 1|E|1\rangle = 0$ but $\langle 1|E^2|1\rangle \neq 0$. Explain why, and calculate the RMS field in a cavity of volume $V$ for frequency $\omega$.

## Computational Lab: Electromagnetic Wave Visualization

```python
"""
Day 214 Computational Lab: Electromagnetic Waves in Vacuum
Topics: Wave propagation, polarization states, 3D visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# Set up styling
plt.style.use('default')

# Physical constants
c = 3e8  # Speed of light
epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7

# =============================================================================
# Part 1: Verify Wave Equation Solution
# =============================================================================

def verify_wave_equation():
    """Numerically verify that plane wave satisfies the wave equation."""

    print("Verifying Wave Equation Solution")
    print("="*50)

    # Wave parameters
    k = 2 * np.pi  # Wave number (1 m wavelength)
    omega = k * c   # Angular frequency

    # Create spatial and temporal grids
    z = np.linspace(0, 2, 1000)  # 2 meters
    t = 1e-9  # Snapshot time

    # Electric field: E = E_0 * cos(kz - ωt)
    E_0 = 100  # V/m
    E = E_0 * np.cos(k * z - omega * t)

    # Calculate derivatives numerically
    dz = z[1] - z[0]
    d2E_dz2 = np.gradient(np.gradient(E, dz), dz)

    # The wave equation: d²E/dz² = (1/c²) * d²E/dt²
    # For our solution: d²E/dt² = -ω² E
    # So: d²E/dz² should equal -(ω²/c²) E = -k² E
    expected = -k**2 * E

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(z, E, 'b-', linewidth=2, label='E(z)')
    axes[0].set_xlabel('z (m)')
    axes[0].set_ylabel('E (V/m)')
    axes[0].set_title('Electric Field Wave')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z, d2E_dz2, 'r-', linewidth=2, label='∂²E/∂z² (numerical)')
    axes[1].plot(z, expected, 'b--', linewidth=2, label='-k²E (expected)')
    axes[1].set_xlabel('z (m)')
    axes[1].set_ylabel('Second derivative')
    axes[1].set_title('Verification: ∂²E/∂z² = -k²E (Wave Equation)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wave_equation_verification.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Numerical verification
    error = np.max(np.abs(d2E_dz2[100:-100] - expected[100:-100])) / np.max(np.abs(expected))
    print(f"Maximum relative error: {error:.2e}")
    print(f"Wave speed: c = ω/k = {omega/k:.3e} m/s")

# =============================================================================
# Part 2: 3D Plane Wave Visualization
# =============================================================================

def visualize_plane_wave_3d():
    """Create 3D visualization of electromagnetic plane wave."""

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Wave parameters
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    E_0 = 1.0
    B_0 = E_0 / c

    # Spatial points along propagation direction
    z = np.linspace(0, 2 * wavelength, 100)

    # Electric field (along x)
    E_x = E_0 * np.sin(k * z)
    E_y = np.zeros_like(z)
    E_z = np.zeros_like(z)

    # Magnetic field (along y)
    B_x = np.zeros_like(z)
    B_y = (E_0 / c) * np.sin(k * z) * 1e8  # Scaled for visibility
    B_z = np.zeros_like(z)

    # Plot E-field
    ax.plot(z, E_x, E_z, 'b-', linewidth=2, label=r'$\vec{E}$')
    ax.plot(z, np.zeros_like(z), np.zeros_like(z), 'k-', linewidth=0.5)

    # E-field arrows
    for i in range(0, len(z), 10):
        ax.quiver(z[i], 0, 0, 0, E_x[i], 0, color='blue', arrow_length_ratio=0.3)

    # Plot B-field (scaled)
    ax.plot(z, B_x, B_y, 'r-', linewidth=2, label=r'$\vec{B}$ (scaled)')

    # B-field arrows
    for i in range(0, len(z), 10):
        ax.quiver(z[i], 0, 0, 0, 0, B_y[i], color='red', arrow_length_ratio=0.3)

    # Propagation direction
    ax.quiver(0, 0, 0, 0.5, 0, 0, color='green', arrow_length_ratio=0.2, linewidth=3)
    ax.text(0.6, 0, 0, r'$\vec{k}$', fontsize=14, color='green')

    ax.set_xlabel('z (propagation)')
    ax.set_ylabel('x (E-field)')
    ax.set_zlabel('y (B-field)')
    ax.set_title('Electromagnetic Plane Wave\n' + r'$\vec{E} \perp \vec{B} \perp \vec{k}$')
    ax.legend()

    # Set viewing angle
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig('plane_wave_3d.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 3: Polarization States
# =============================================================================

def visualize_polarization():
    """Visualize different polarization states."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Time/phase parameter
    t = np.linspace(0, 2*np.pi, 200)

    # ========== Linear Polarization ==========
    ax = axes[0, 0]
    E_x = np.cos(t)
    E_y = np.zeros_like(t)
    ax.plot(E_x, E_y, 'b-', linewidth=2)
    ax.plot([E_x[-1]], [E_y[-1]], 'bo', markersize=10)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$E_x$')
    ax.set_ylabel(r'$E_y$')
    ax.set_title('Linear Polarization\n(E along x-axis)')
    ax.set_aspect('equal')

    # ========== Linear at 45° ==========
    ax = axes[0, 1]
    E_x = np.cos(t) / np.sqrt(2)
    E_y = np.cos(t) / np.sqrt(2)
    ax.plot(E_x, E_y, 'b-', linewidth=2)
    ax.plot([E_x[-1]], [E_y[-1]], 'bo', markersize=10)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$E_x$')
    ax.set_ylabel(r'$E_y$')
    ax.set_title('Linear Polarization at 45°')
    ax.set_aspect('equal')

    # ========== Right Circular Polarization ==========
    ax = axes[1, 0]
    E_x = np.cos(t)
    E_y = np.sin(t)
    ax.plot(E_x, E_y, 'r-', linewidth=2)
    ax.plot([E_x[-1]], [E_y[-1]], 'ro', markersize=10)

    # Add arrow to show rotation direction
    ax.annotate('', xy=(E_x[10], E_y[10]), xytext=(E_x[0], E_y[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$E_x$')
    ax.set_ylabel(r'$E_y$')
    ax.set_title('Right Circular Polarization\n(counterclockwise rotation)')
    ax.set_aspect('equal')

    # ========== Elliptical Polarization ==========
    ax = axes[1, 1]
    E_x = np.cos(t)
    E_y = 0.5 * np.sin(t + np.pi/4)  # Different amplitude and phase
    ax.plot(E_x, E_y, 'g-', linewidth=2)
    ax.plot([E_x[-1]], [E_y[-1]], 'go', markersize=10)

    # Add arrow to show rotation direction
    ax.annotate('', xy=(E_x[10], E_y[10]), xytext=(E_x[0], E_y[0]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$E_x$')
    ax.set_ylabel(r'$E_y$')
    ax.set_title('Elliptical Polarization\n(general case)')
    ax.set_aspect('equal')

    plt.suptitle('Polarization States of Electromagnetic Waves', fontsize=14,
                fontweight='bold')
    plt.tight_layout()
    plt.savefig('polarization_states.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 4: Animated Wave Propagation
# =============================================================================

def animate_wave_propagation():
    """Animate electromagnetic wave propagation."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Wave parameters
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    omega = k * c
    E_0 = 1.0

    # Spatial domain
    z = np.linspace(0, 3 * wavelength, 500)

    # Initialize plots
    line_E, = axes[0].plot([], [], 'b-', linewidth=2, label=r'$E_x(z)$')
    line_B, = axes[1].plot([], [], 'r-', linewidth=2, label=r'$B_y(z)$')

    axes[0].set_xlim(0, 3 * wavelength)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_ylabel('E-field (normalized)')
    axes[0].set_title('Electric Field')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlim(0, 3 * wavelength)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_xlabel('z (meters)')
    axes[1].set_ylabel('B-field (normalized)')
    axes[1].set_title('Magnetic Field')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes, fontsize=12)

    def init():
        line_E.set_data([], [])
        line_B.set_data([], [])
        time_text.set_text('')
        return line_E, line_B, time_text

    def animate(frame):
        t = frame * (2 * np.pi / omega) / 50  # One period over 50 frames

        E = E_0 * np.sin(k * z - omega * t)
        B = (E_0 / c) * np.sin(k * z - omega * t) * c  # Normalized to same scale

        line_E.set_data(z, E)
        line_B.set_data(z, B)
        time_text.set_text(f'Time: {t*1e9:.3f} ns')

        return line_E, line_B, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=50,
                        interval=100, blit=True)

    plt.tight_layout()
    plt.savefig('wave_propagation_snapshot.png', dpi=150, bbox_inches='tight')
    plt.show()

    return anim

# =============================================================================
# Part 5: Electromagnetic Spectrum
# =============================================================================

def plot_em_spectrum():
    """Visualize the electromagnetic spectrum."""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Spectrum regions (frequency ranges in Hz)
    regions = [
        ('Radio', 1e3, 3e8, 'red'),
        ('Microwave', 3e8, 3e11, 'orange'),
        ('Infrared', 3e11, 4e14, 'darkred'),
        ('Visible', 4e14, 8e14, 'green'),
        ('UV', 8e14, 3e16, 'purple'),
        ('X-ray', 3e16, 3e19, 'blue'),
        ('Gamma', 3e19, 3e22, 'darkblue'),
    ]

    for name, f_min, f_max, color in regions:
        ax.barh(0, np.log10(f_max) - np.log10(f_min),
               left=np.log10(f_min), height=0.5, color=color, alpha=0.7,
               edgecolor='black', linewidth=1)
        ax.text((np.log10(f_min) + np.log10(f_max))/2, 0,
               name, ha='center', va='center', fontsize=10, fontweight='bold')

    # Add frequency scale
    ax.set_xlim(2, 23)
    ax.set_ylim(-1, 1)

    # Custom x-axis
    freq_ticks = [3, 6, 9, 12, 15, 18, 21]
    freq_labels = ['kHz', 'MHz', 'GHz', 'THz', 'PHz', 'EHz', 'ZHz']
    ax.set_xticks(freq_ticks)
    ax.set_xticklabels(freq_labels)
    ax.set_xlabel('Frequency', fontsize=12)

    # Add wavelength scale on top
    ax2 = ax.twiny()
    wave_ticks = [3, 6, 9, 12, 15, 18, 21]
    wave_labels = ['100 km', '100 m', '10 cm', '100 μm', '100 nm', '0.1 nm', '0.1 pm']
    ax2.set_xlim(2, 23)
    ax2.set_xticks(wave_ticks)
    ax2.set_xticklabels(wave_labels)
    ax2.set_xlabel('Wavelength', fontsize=12)

    ax.set_yticks([])
    ax.set_title('The Electromagnetic Spectrum\nAll waves travel at c = 3×10⁸ m/s',
                fontsize=14, fontweight='bold', pad=40)

    # Add some example applications
    annotations = [
        (5, -0.7, 'AM Radio'),
        (8, -0.7, 'WiFi'),
        (12, -0.7, 'Heat'),
        (14.7, -0.7, 'Eyes'),
        (16, -0.7, 'Sun'),
        (18, -0.7, 'Medical'),
        (20, -0.7, 'Nuclear'),
    ]

    for x, y, text in annotations:
        ax.annotate(text, xy=(x, 0.25), xytext=(x, y),
                   fontsize=9, ha='center',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.tight_layout()
    plt.savefig('em_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 6: Photon Energy and Momentum
# =============================================================================

def plot_photon_properties():
    """Plot photon energy and momentum vs wavelength/frequency."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    h = 6.626e-34  # Planck's constant
    eV = 1.602e-19  # eV in Joules

    # Wavelength range
    wavelength = np.logspace(-12, -3, 1000)  # 1 pm to 1 mm

    # Photon energy
    energy_J = h * c / wavelength
    energy_eV = energy_J / eV

    # Photon momentum
    momentum = h / wavelength

    # Plot 1: Energy vs wavelength
    ax = axes[0]
    ax.loglog(wavelength * 1e9, energy_eV, 'b-', linewidth=2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Photon Energy (eV)')
    ax.set_title('Photon Energy: E = hc/λ')
    ax.grid(True, alpha=0.3, which='both')

    # Mark visible range
    ax.axvspan(380, 700, alpha=0.2, color='green', label='Visible')
    ax.axhline(y=1.8, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=3.1, color='violet', linestyle='--', alpha=0.5)
    ax.legend()

    # Plot 2: Momentum vs frequency
    ax = axes[1]
    frequency = c / wavelength
    ax.loglog(frequency, momentum, 'r-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Photon Momentum (kg·m/s)')
    ax.set_title('Photon Momentum: p = h/λ = hf/c')
    ax.grid(True, alpha=0.3, which='both')

    # Mark visible range
    ax.axvspan(4e14, 8e14, alpha=0.2, color='green', label='Visible')
    ax.legend()

    plt.tight_layout()
    plt.savefig('photon_properties.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print some values
    print("\nPhoton Properties for Common Wavelengths:")
    print("-" * 50)
    wavelengths = [500e-9, 1e-6, 1e-9, 1e-12]
    names = ['Green light', 'Infrared', 'X-ray', 'Gamma ray']

    for wl, name in zip(wavelengths, names):
        E = h * c / wl
        p = h / wl
        f = c / wl
        print(f"{name} (λ = {wl*1e9:.3g} nm):")
        print(f"  Energy: {E/eV:.3g} eV")
        print(f"  Momentum: {p:.3e} kg·m/s")
        print(f"  Frequency: {f:.3e} Hz")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Day 214: Electromagnetic Waves in Vacuum")
    print("="*60)

    print("\n1. Verifying Wave Equation Solution")
    verify_wave_equation()

    print("\n2. 3D Plane Wave Visualization")
    visualize_plane_wave_3d()

    print("\n3. Polarization States")
    visualize_polarization()

    print("\n4. Animated Wave Propagation")
    anim = animate_wave_propagation()

    print("\n5. Electromagnetic Spectrum")
    plot_em_spectrum()

    print("\n6. Photon Properties")
    plot_photon_properties()

    print("\nAll visualizations complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Wave equation | $\nabla^2 \vec{E} = \frac{1}{c^2}\frac{\partial^2 \vec{E}}{\partial t^2}$ |
| Speed of light | $c = \frac{1}{\sqrt{\mu_0 \epsilon_0}} = 3 \times 10^8$ m/s |
| Dispersion relation | $\omega = ck$, $f\lambda = c$ |
| E-B relationship | $E = cB$, $\vec{B} = \frac{1}{c}\hat{k} \times \vec{E}$ |
| Plane wave (E) | $\vec{E} = E_0 e^{i(\vec{k}\cdot\vec{r} - \omega t)}\hat{\epsilon}$ |
| Photon energy | $E = \hbar\omega = hf$ |
| Photon momentum | $p = \hbar k = h/\lambda$ |

### Polarization States

| Type | Electric Field |
|------|---------------|
| Linear (x) | $E_0 \cos(kz - \omega t)\hat{x}$ |
| Right circular | $E_0[\cos(kz-\omega t)\hat{x} + \sin(kz-\omega t)\hat{y}]$ |
| Left circular | $E_0[\cos(kz-\omega t)\hat{x} - \sin(kz-\omega t)\hat{y}]$ |

### Main Takeaways

1. **Maxwell's equations predict electromagnetic waves** traveling at $c = 1/\sqrt{\mu_0\epsilon_0}$
2. **E and B are perpendicular** to each other and to the propagation direction (transverse waves)
3. **The E/B ratio equals c** in vacuum
4. **Polarization** describes the E-field vector's behavior—linear, circular, or elliptical
5. **All EM waves** from radio to gamma rays travel at the same speed in vacuum
6. **Photons** are the quantum of EM waves with $E = \hbar\omega$, $p = \hbar k$

## Daily Checklist

- [ ] I can derive the wave equation from Maxwell's equations
- [ ] I can calculate the speed of light from ε₀ and μ₀
- [ ] I can write plane wave solutions for E and B
- [ ] I understand the relationship between E, B, and propagation direction
- [ ] I can describe different polarization states
- [ ] I understand the connection between classical waves and photons
- [ ] I completed the computational lab

## Preview: Day 215

Tomorrow we explore the **energy and momentum carried by electromagnetic waves**. The Poynting vector $\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}$ describes energy flow, and remarkably, light also carries momentum—leading to radiation pressure that can push objects around!

---

*"Light consists of transverse undulations in the same medium that is the cause of electric and magnetic phenomena."* — James Clerk Maxwell, 1862
