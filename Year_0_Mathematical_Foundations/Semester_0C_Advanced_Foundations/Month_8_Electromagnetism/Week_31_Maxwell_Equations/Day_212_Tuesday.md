# Day 212: Displacement Current and Maxwell's Correction

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning I | 2 hrs | The failure of Ampère's law for time-varying fields |
| Morning II | 2 hrs | Maxwell's displacement current concept |
| Afternoon | 2 hrs | Problem solving and applications |
| Evening | 2 hrs | Computational lab: Visualizing displacement current |

## Learning Objectives

By the end of today, you will be able to:

1. **Identify the inconsistency** in Ampère's law when applied to time-varying fields
2. **Derive the displacement current** from charge conservation principles
3. **State the Ampère-Maxwell law** in both differential and integral forms
4. **Calculate displacement current** in capacitors and other configurations
5. **Explain the physical significance** of Maxwell's correction
6. **Connect displacement current** to photon propagation in quantum field theory

## Core Content

### 1. The Problem with Ampère's Law

Ampère's original law states:
$$\oint \vec{B} \cdot d\vec{l} = \mu_0 I_{enc}$$

Or in differential form:
$$\nabla \times \vec{B} = \mu_0 \vec{J}$$

**The Inconsistency:** Consider a charging capacitor with a wire carrying current $I$.

Choose two surfaces bounded by the same Amperian loop:
- **Surface 1:** Flat surface through the wire → $I_{enc} = I$
- **Surface 2:** Bulging surface through the capacitor gap → $I_{enc} = 0$

This gives **two different answers** for the same line integral! Ampère's law is inconsistent.

### 2. Maxwell's Resolution: The Displacement Current

Maxwell realized that something "current-like" must flow between the capacitor plates. He called it the **displacement current**.

**Derivation from Charge Conservation:**

The continuity equation expresses conservation of charge:
$$\nabla \cdot \vec{J} + \frac{\partial \rho}{\partial t} = 0$$

Taking the divergence of Ampère's law:
$$\nabla \cdot (\nabla \times \vec{B}) = \mu_0 \nabla \cdot \vec{J}$$

The left side is always zero (divergence of a curl), but $\nabla \cdot \vec{J} \neq 0$ when charge is accumulating.

**The Fix:** Use Gauss's law $\nabla \cdot \vec{E} = \rho/\epsilon_0$ to substitute:
$$\nabla \cdot \vec{J} = -\frac{\partial \rho}{\partial t} = -\epsilon_0 \frac{\partial}{\partial t}(\nabla \cdot \vec{E}) = -\nabla \cdot \left(\epsilon_0 \frac{\partial \vec{E}}{\partial t}\right)$$

So if we define the **displacement current density**:
$$\boxed{\vec{J}_D = \epsilon_0 \frac{\partial \vec{E}}{\partial t}}$$

Then $\nabla \cdot (\vec{J} + \vec{J}_D) = 0$, and we can write:

### 3. The Ampère-Maxwell Law

$$\boxed{\nabla \times \vec{B} = \mu_0 \vec{J} + \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}}$$

Or equivalently:
$$\nabla \times \vec{B} = \mu_0 (\vec{J} + \vec{J}_D)$$

**Integral form:**
$$\boxed{\oint \vec{B} \cdot d\vec{l} = \mu_0 I_{enc} + \mu_0 \epsilon_0 \frac{d\Phi_E}{dt}}$$

where $\Phi_E = \int \vec{E} \cdot d\vec{A}$ is the electric flux.

### 4. Physical Interpretation

The displacement current is **not** a flow of charges. It represents:

1. **The rate of change of electric field** acts as a source of magnetic field
2. **A changing electric flux** is equivalent to a current for producing $\vec{B}$
3. **This term enables electromagnetic waves** to propagate through vacuum

**Symmetry with Faraday's Law:**
- Changing $\vec{B}$ creates $\vec{E}$ (Faraday)
- Changing $\vec{E}$ creates $\vec{B}$ (Ampère-Maxwell)

This mutual induction is the essence of electromagnetic radiation!

### 5. Displacement Current in a Capacitor

Consider a parallel-plate capacitor with plate area $A$ and separation $d$, being charged by current $I$.

**Between the plates:**
$$E = \frac{\sigma}{\epsilon_0} = \frac{Q}{\epsilon_0 A}$$

$$\frac{\partial E}{\partial t} = \frac{1}{\epsilon_0 A}\frac{dQ}{dt} = \frac{I}{\epsilon_0 A}$$

**Displacement current density:**
$$J_D = \epsilon_0 \frac{\partial E}{\partial t} = \frac{I}{A}$$

**Total displacement current:**
$$I_D = J_D \cdot A = I$$

The displacement current between the plates **equals** the conduction current in the wire. This resolves the ambiguity!

### 6. The Significance of $\mu_0 \epsilon_0$

The coefficient in the displacement current term is $\mu_0 \epsilon_0$. Its units:
$$[\mu_0 \epsilon_0] = \frac{\text{H}}{\text{m}} \cdot \frac{\text{F}}{\text{m}} = \frac{\text{s}^2}{\text{m}^2}$$

This has dimensions of (velocity)$^{-2}$. In fact:
$$c = \frac{1}{\sqrt{\mu_0 \epsilon_0}} = 299,792,458 \text{ m/s}$$

**This is the speed of light!** Maxwell recognized this in 1865, leading him to conclude that light is an electromagnetic wave.

### 7. Maxwell's Equations: Preliminary Form

With the displacement current, we now have all four equations:

| Equation | Differential Form | Physical Meaning |
|----------|------------------|------------------|
| Gauss (E) | $\nabla \cdot \vec{E} = \rho/\epsilon_0$ | Charges source E-fields |
| Gauss (B) | $\nabla \cdot \vec{B} = 0$ | No magnetic monopoles |
| Faraday | $\nabla \times \vec{E} = -\partial \vec{B}/\partial t$ | Changing B creates E |
| Ampère-Maxwell | $\nabla \times \vec{B} = \mu_0 \vec{J} + \mu_0\epsilon_0 \partial \vec{E}/\partial t$ | Currents and changing E create B |

### 8. Electromagnetic Energy and the Poynting Theorem

Maxwell's correction affects energy flow. The **Poynting theorem** states:
$$\frac{\partial u}{\partial t} + \nabla \cdot \vec{S} = -\vec{J} \cdot \vec{E}$$

where:
- $u = \frac{1}{2}\epsilon_0 E^2 + \frac{1}{2\mu_0}B^2$ is energy density
- $\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}$ is the Poynting vector (energy flux)

The displacement current ensures energy is conserved as fields propagate.

## Quantum Mechanics Connection

### Virtual Photons and the Displacement Current

In quantum electrodynamics (QED), the electromagnetic field is quantized into photons. The displacement current can be understood as:

1. **Virtual photon exchange:** The changing E-field between capacitor plates involves virtual photons
2. **Photon propagation:** Displacement current enables "real" photons to travel through vacuum
3. **Field quantization:** $\epsilon_0 \partial E/\partial t$ describes the creation/annihilation of photon states

### The Photon Propagator

In QED, the photon propagator contains the term:
$$D_{\mu\nu}(k) \sim \frac{g_{\mu\nu}}{k^2}$$

This describes how electromagnetic disturbances (photons) propagate. The classical displacement current is the macroscopic manifestation of this quantum process.

### Vacuum Polarization

At the quantum level, the vacuum itself responds to fields:
- Virtual electron-positron pairs briefly appear and disappear
- This modifies $\epsilon_0$ at high field strengths
- The "quantum vacuum" has structure that affects EM propagation

### Coherent States Connection

Classical electromagnetic waves (described by Maxwell's equations with displacement current) correspond to **coherent states** of the photon field:
$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^{\infty} \frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

The displacement current describes the coherent oscillation of many photons.

## Worked Examples

### Example 1: Displacement Current in a Capacitor

A parallel-plate capacitor with circular plates of radius $R = 5$ cm is being charged at a rate $dQ/dt = 2$ A. Find:
(a) The displacement current density between the plates
(b) The magnetic field at radius $r = 3$ cm from the center

**Solution:**

**(a) Displacement current density:**

Electric field between plates: $E = \frac{Q}{\epsilon_0 \pi R^2}$

$$\frac{\partial E}{\partial t} = \frac{1}{\epsilon_0 \pi R^2}\frac{dQ}{dt}$$

$$J_D = \epsilon_0 \frac{\partial E}{\partial t} = \frac{1}{\pi R^2}\frac{dQ}{dt} = \frac{2}{\pi (0.05)^2}$$

$$\boxed{J_D = 254.6 \text{ A/m}^2}$$

**(b) Magnetic field at r = 3 cm:**

Use Ampère-Maxwell law with a circular Amperian loop of radius $r$:
$$\oint \vec{B} \cdot d\vec{l} = B(2\pi r) = \mu_0 I_{D,enc}$$

where $I_{D,enc} = J_D \cdot \pi r^2$ (displacement current through the loop).

$$B = \frac{\mu_0 J_D r}{2} = \frac{\mu_0 r}{2\pi R^2}\frac{dQ}{dt}$$

$$B = \frac{(4\pi \times 10^{-7})(0.03)}{2\pi (0.05)^2}(2) = 4.8 \times 10^{-6} \text{ T}$$

$$\boxed{B = 4.8\ \mu\text{T}}$$

### Example 2: Time-Varying Electric Field

A cylindrical region of radius $R$ has a uniform electric field pointing along the axis, varying as $E = E_0 \sin(\omega t)$.

Find the magnetic field induced inside and outside the cylinder.

**Solution:**

**Inside (r < R):**

The displacement current density is:
$$J_D = \epsilon_0 \frac{\partial E}{\partial t} = \epsilon_0 E_0 \omega \cos(\omega t)$$

Using Ampère-Maxwell law with a circular loop of radius $r$:
$$B(2\pi r) = \mu_0 \epsilon_0 \pi r^2 \frac{\partial E}{\partial t}$$

$$B = \frac{\mu_0 \epsilon_0 r}{2} E_0 \omega \cos(\omega t) = \frac{r E_0 \omega}{2c^2}\cos(\omega t)$$

$$\boxed{B_{in} = \frac{r E_0 \omega}{2c^2}\cos(\omega t) \quad (r < R)}$$

**Outside (r > R):**

$$B(2\pi r) = \mu_0 \epsilon_0 \pi R^2 E_0 \omega \cos(\omega t)$$

$$\boxed{B_{out} = \frac{R^2 E_0 \omega}{2rc^2}\cos(\omega t) \quad (r > R)}$$

Note: The magnetic field is **tangential** (azimuthal), circulating around the E-field axis.

### Example 3: Energy Flow in a Charging Capacitor

For the capacitor in Example 1, find the rate at which energy flows into the region between the plates through a cylindrical surface at radius $R$.

**Solution:**

The electric field between the plates: $E = \frac{Q}{\epsilon_0 \pi R^2}$

The magnetic field at $r = R$:
$$B = \frac{\mu_0 R J_D}{2} = \frac{\mu_0}{2\pi R}\frac{dQ}{dt}$$

The Poynting vector at the cylindrical surface:
$$S = \frac{EB}{\mu_0}$$

Direction: $\vec{E}$ is axial, $\vec{B}$ is azimuthal, so $\vec{S} = \vec{E} \times \vec{B}/\mu_0$ points **radially inward**.

$$S = \frac{1}{\mu_0} \cdot \frac{Q}{\epsilon_0 \pi R^2} \cdot \frac{\mu_0}{2\pi R}\frac{dQ}{dt} = \frac{Q}{2\pi^2 \epsilon_0 R^3}\frac{dQ}{dt}$$

Power flowing through the cylindrical surface (area = $2\pi R d$, where $d$ is plate separation):
$$P = S \cdot 2\pi R d = \frac{Qd}{\pi \epsilon_0 R^2}\frac{dQ}{dt}$$

But $Qd/(\epsilon_0 \pi R^2) = Ed = V$ (voltage across capacitor), so:
$$P = V \frac{dQ}{dt} = VI$$

This equals the power supplied by the charging circuit, confirming energy conservation!

## Practice Problems

### Level 1: Direct Application

1. A capacitor with circular plates of radius 8 cm is charged such that $dE/dt = 10^{12}$ V/(m·s). Calculate the displacement current.

2. Find the magnetic field at the center of a capacitor (r = 0) and at the edge (r = R) during charging at rate $I$.

3. If $E = E_0 e^{-t/\tau}$ in a region of space, find the displacement current density.

### Level 2: Intermediate

4. A parallel-plate capacitor is connected to an AC source: $V = V_0 \cos(\omega t)$. Find expressions for:
   (a) The displacement current as a function of time
   (b) The maximum magnetic field between the plates

5. Between the plates of a capacitor, both conduction current (due to slightly conducting dielectric with conductivity $\sigma$) and displacement current flow. Find the ratio $I_D/I_c$ for a sinusoidal voltage at frequency $f$, given dielectric constant $\kappa$.

6. A coaxial cable has inner radius $a$, outer radius $b$, and carries current $I(t) = I_0 \sin(\omega t)$. Find the displacement current in the dielectric between conductors if $\epsilon = \kappa \epsilon_0$.

### Level 3: Challenging

7. Show that the displacement current accounts for exactly the "missing" current in Ampère's law for a charging capacitor, regardless of the shape of the Amperian surface chosen.

8. A spherical capacitor (inner radius $a$, outer radius $b$) is charged at rate $dQ/dt = I$. Find the magnetic field as a function of position between the shells.

9. **Quantum connection:** The vacuum has quantum fluctuations characterized by the zero-point energy $\frac{1}{2}\hbar\omega$ per mode. Estimate the order of magnitude of the RMS electric field due to vacuum fluctuations in a cubic cavity of side $L$, considering only the lowest frequency mode.

## Computational Lab: Visualizing Displacement Current

```python
"""
Day 212 Computational Lab: Displacement Current Visualization
Topics: Maxwell's correction, fields in capacitors, energy flow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.animation import FuncAnimation

# Set up styling
plt.style.use('default')

# Physical constants
epsilon_0 = 8.854e-12  # F/m
mu_0 = 4 * np.pi * 1e-7  # H/m
c = 1 / np.sqrt(mu_0 * epsilon_0)

# =============================================================================
# Part 1: Ampère's Law Ambiguity
# =============================================================================

def plot_ampere_ambiguity():
    """Visualize the ambiguity in Ampère's law for a charging capacitor."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Surface through wire
    ax = axes[0]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)

    # Draw capacitor plates
    ax.add_patch(Rectangle((-0.1, -1), 0.2, 2, color='gray', alpha=0.5))
    ax.add_patch(Rectangle((0.8, -1), 0.2, 2, color='gray', alpha=0.5))

    # Draw wire
    ax.plot([-2, -0.1], [0, 0], 'b-', linewidth=3, label='Wire (current I)')
    ax.plot([1, 2], [0, 0], 'b-', linewidth=3)

    # Draw Amperian loop
    theta = np.linspace(0, 2*np.pi, 100)
    loop_x = -0.5 + 0.4 * np.cos(theta)
    loop_y = 0.4 * np.sin(theta)
    ax.plot(loop_x, loop_y, 'r-', linewidth=2, label='Amperian loop')

    # Draw flat surface
    ax.fill_between(loop_x, loop_y, alpha=0.3, color='green', label='Surface S₁')

    # Current arrow
    ax.annotate('', xy=(-0.3, 0), xytext=(-0.7, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(-0.5, 0.15, 'I', fontsize=14, color='blue')

    ax.set_title('Surface S₁ through wire: $I_{enc} = I$', fontsize=12)
    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    # Right plot: Surface through gap
    ax = axes[1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)

    # Draw capacitor plates
    ax.add_patch(Rectangle((-0.1, -1), 0.2, 2, color='gray', alpha=0.5))
    ax.add_patch(Rectangle((0.8, -1), 0.2, 2, color='gray', alpha=0.5))

    # Draw wire
    ax.plot([-2, -0.1], [0, 0], 'b-', linewidth=3)
    ax.plot([1, 2], [0, 0], 'b-', linewidth=3)

    # Draw same Amperian loop
    ax.plot(loop_x, loop_y, 'r-', linewidth=2, label='Amperian loop (same)')

    # Draw bulging surface through gap
    # Parametric surface bulging to the right
    theta_surf = np.linspace(-0.4, 0.4, 50)
    x_surf = -0.5 + 0.4 * np.cos(np.linspace(0, 2*np.pi, 100))
    y_surf = 0.4 * np.sin(np.linspace(0, 2*np.pi, 100))

    # Draw bulge
    bulge_y = np.linspace(-0.4, 0.4, 20)
    bulge_x = 0.45 * np.ones_like(bulge_y)
    ax.fill_betweenx(bulge_y, -0.5 + np.sqrt(0.16 - bulge_y**2),
                     bulge_x, alpha=0.3, color='orange', label='Surface S₂')

    # E-field in gap
    for yy in np.linspace(-0.3, 0.3, 4):
        ax.annotate('', xy=(0.7, yy), xytext=(0.3, yy),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(0.5, 0.5, '$\\vec{E}$', fontsize=14, color='red')

    ax.set_title('Surface S₂ through gap: $I_{enc} = 0$ !?', fontsize=12)
    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    plt.suptitle("The Ampère's Law Ambiguity: Same loop, different surfaces, different answers!",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ampere_ambiguity.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 2: B-field Between Capacitor Plates
# =============================================================================

def plot_B_field_capacitor():
    """Plot magnetic field distribution between charging capacitor plates."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parameters
    R = 0.05  # Plate radius (m)
    I = 1.0   # Charging current (A)

    # Radial positions
    r = np.linspace(0, 0.1, 200)

    # B-field: linear inside, 1/r outside
    J_D = I / (np.pi * R**2)  # Displacement current density

    B_inside = mu_0 * J_D * r / 2  # r < R
    B_outside = mu_0 * I / (2 * np.pi * r)  # r > R

    B = np.where(r < R, B_inside, B_outside)

    # Plot 1: B vs r
    axes[0].plot(r * 100, B * 1e6, 'b-', linewidth=2)
    axes[0].axvline(x=R*100, color='r', linestyle='--', label=f'Plate edge R={R*100} cm')
    axes[0].set_xlabel('Radial Distance r (cm)')
    axes[0].set_ylabel('Magnetic Field B (μT)')
    axes[0].set_title(f'B-field Distribution (I = {I} A)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 10)

    # Plot 2: Vector field in the gap
    ax = axes[1]

    # Create grid
    x = np.linspace(-0.08, 0.08, 15)
    y = np.linspace(-0.08, 0.08, 15)
    X, Y = np.meshgrid(x, y)
    r_grid = np.sqrt(X**2 + Y**2)

    # B-field magnitude
    B_mag = np.where(r_grid < R,
                     mu_0 * J_D * r_grid / 2,
                     mu_0 * I / (2 * np.pi * (r_grid + 1e-10)))

    # B-field direction (azimuthal)
    theta = np.arctan2(Y, X)
    Bx = -B_mag * np.sin(theta)
    By = B_mag * np.cos(theta)

    # Normalize for visualization
    B_norm = np.sqrt(Bx**2 + By**2)
    Bx_n = Bx / (B_norm + 1e-15)
    By_n = By / (B_norm + 1e-15)

    # Plot
    quiv = ax.quiver(X*100, Y*100, Bx_n, By_n, B_norm*1e6, cmap='plasma', alpha=0.8)
    plt.colorbar(quiv, ax=ax, label='|B| (μT)')

    # Draw plate outline
    circle = plt.Circle((0, 0), R*100, fill=False, color='green',
                        linewidth=2, linestyle='--', label='Plate edge')
    ax.add_patch(circle)

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title('B-field Vector Pattern in Capacitor Gap')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('B_field_capacitor.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 3: Time Evolution of Displacement Current
# =============================================================================

def animate_displacement_current():
    """Animate the relationship between E-field and displacement current."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Parameters
    omega = 2 * np.pi  # Angular frequency
    E_0 = 1000  # V/m
    t_max = 2.0

    # Time array
    t = np.linspace(0, t_max, 500)

    # E-field and displacement current
    E = E_0 * np.sin(omega * t)
    J_D = epsilon_0 * E_0 * omega * np.cos(omega * t)

    # Plot 1: E vs t
    axes[0, 0].plot(t, E, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Electric Field E (V/m)')
    axes[0, 0].set_title('Time-Varying Electric Field')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, t_max)

    # Plot 2: J_D vs t
    axes[0, 1].plot(t, J_D * 1e9, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Displacement Current Density J_D (nA/m²)')
    axes[0, 1].set_title('Displacement Current Density')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, t_max)

    # Plot 3: Phase relationship
    axes[1, 0].plot(t, E/np.max(E), 'b-', linewidth=2, label='E (normalized)')
    axes[1, 0].plot(t, J_D/np.max(np.abs(J_D)), 'r--', linewidth=2, label='J_D (normalized)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Normalized Value')
    axes[1, 0].set_title('Phase Relationship: J_D leads E by 90°')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, t_max)

    # Plot 4: B-field at capacitor edge
    R = 0.05  # m
    B = mu_0 * epsilon_0 * R * E_0 * omega * np.cos(omega * t) / 2

    axes[1, 1].plot(t, B * 1e9, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Magnetic Field B (nT)')
    axes[1, 1].set_title(f'B-field at Capacitor Edge (R={R*100} cm)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, t_max)

    plt.tight_layout()
    plt.savefig('displacement_current_time.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 4: Energy Flow (Poynting Vector)
# =============================================================================

def plot_poynting_vector_capacitor():
    """Visualize energy flow into a charging capacitor."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Parameters
    R = 0.05  # Plate radius
    d = 0.01  # Plate separation
    I = 1.0   # Charging current

    # Left plot: Schematic with Poynting vectors
    ax = axes[0]

    # Draw capacitor plates
    plate_width = 0.02
    ax.add_patch(Rectangle((-R, -d/2-plate_width), 2*R, plate_width,
                           color='gray', alpha=0.7))
    ax.add_patch(Rectangle((-R, d/2), 2*R, plate_width,
                           color='gray', alpha=0.7))

    # E-field arrows (vertical, between plates)
    for x_pos in np.linspace(-R*0.8, R*0.8, 5):
        ax.annotate('', xy=(x_pos, d/2-0.002), xytext=(x_pos, -d/2+0.002),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.text(0, 0, '$\\vec{E}$', fontsize=14, color='blue', ha='center')

    # B-field indicators (into/out of page at edges)
    ax.plot([-R-0.01], [0], 'go', markersize=15)
    ax.text(-R-0.01, 0, '⊙', fontsize=10, ha='center', va='center')
    ax.text(-R-0.02, 0.015, '$\\vec{B}$', fontsize=12, color='green')

    ax.plot([R+0.01], [0], 'go', markersize=15)
    ax.text(R+0.01, 0, '⊗', fontsize=10, ha='center', va='center')

    # Poynting vectors (radially inward)
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        x_start = (R + 0.025) * np.cos(angle)
        y_start = 0
        dx = -0.015 * np.cos(angle)
        dy = 0
        ax.annotate('', xy=(x_start + dx, y_start), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.text(R + 0.03, 0.015, '$\\vec{S}$', fontsize=14, color='red')
    ax.text(0, -d/2 - 0.03, 'Energy flows IN through sides', fontsize=11,
            ha='center', color='red')

    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.05, 0.05)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title('Energy Flow into Charging Capacitor')
    ax.set_aspect('equal')

    # Right plot: Poynting vector magnitude vs time
    ax = axes[1]

    # Charging with constant current
    t = np.linspace(0, 1, 100)
    Q = I * t  # Charge
    V = Q / (epsilon_0 * np.pi * R**2 / d)  # Voltage (C = ε₀A/d)
    E_field = V / d

    # B at edge
    B = mu_0 * I / (2 * np.pi * R)

    # Poynting vector magnitude at edge
    S = E_field * B / mu_0

    # Power into capacitor
    P_in = S * 2 * np.pi * R * d

    # Energy stored
    U = 0.5 * epsilon_0 * E_field**2 * np.pi * R**2 * d

    ax.plot(t, P_in, 'r-', linewidth=2, label='Power in (Poynting)')
    ax.plot(t, np.gradient(U, t[1]-t[0]), 'b--', linewidth=2, label='dU/dt (energy rate)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Flow vs Energy Storage Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('poynting_capacitor.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 5: Verification of Ampère-Maxwell Law
# =============================================================================

def verify_ampere_maxwell():
    """Numerically verify that displacement current resolves the ambiguity."""

    # Parameters
    R = 0.05  # Plate radius
    I = 1.0   # Conduction current

    # Create various Amperian loop radii
    r_loop = np.linspace(0.01, 0.08, 50)

    # Total current through flat surface (just conduction current, all pass through)
    I_flat = np.where(r_loop > R, I, I)  # All current passes through wire

    # Through bulging surface: no conduction current, only displacement current
    J_D = I / (np.pi * R**2)  # Displacement current density
    I_displacement = np.where(r_loop < R,
                              J_D * np.pi * r_loop**2,  # Inside plates
                              I)  # Outside, full displacement current

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(r_loop * 100, I_flat, 'b-', linewidth=2,
            label='Conduction current (flat surface)')
    ax.plot(r_loop * 100, I_displacement, 'r--', linewidth=2,
            label='Displacement current (bulging surface)')

    ax.axvline(x=R*100, color='green', linestyle=':', linewidth=2,
               label=f'Plate edge (R = {R*100} cm)')

    ax.set_xlabel('Amperian Loop Radius (cm)')
    ax.set_ylabel('Enclosed Current (A)')
    ax.set_title('Resolution of Ampère\'s Law Ambiguity\nDisplacement current = Conduction current at plate edge!')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Total current is same\nfor both surfaces!',
                xy=(R*100, I), xytext=(R*100 + 1, 0.7),
                fontsize=11, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig('ampere_maxwell_verification.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Verification of Ampère-Maxwell Law:")
    print(f"Conduction current in wire: {I:.3f} A")
    print(f"Total displacement current between plates: {J_D * np.pi * R**2:.3f} A")
    print("They are equal, resolving the ambiguity!")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Day 212: Displacement Current Computational Lab")
    print("="*60)

    print("\n1. Ampère's Law Ambiguity Visualization")
    plot_ampere_ambiguity()

    print("\n2. B-field in Charging Capacitor")
    plot_B_field_capacitor()

    print("\n3. Time Evolution of Displacement Current")
    animate_displacement_current()

    print("\n4. Poynting Vector and Energy Flow")
    plot_poynting_vector_capacitor()

    print("\n5. Verification of Ampère-Maxwell Law")
    verify_ampere_maxwell()

    print("\nAll visualizations complete!")
    print(f"\nSpeed of light from ε₀ and μ₀: c = {c:.0f} m/s")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Displacement current density | $\vec{J}_D = \epsilon_0 \frac{\partial \vec{E}}{\partial t}$ |
| Ampère-Maxwell (differential) | $\nabla \times \vec{B} = \mu_0 \vec{J} + \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}$ |
| Ampère-Maxwell (integral) | $\oint \vec{B} \cdot d\vec{l} = \mu_0 I_{enc} + \mu_0 \epsilon_0 \frac{d\Phi_E}{dt}$ |
| Speed of light | $c = \frac{1}{\sqrt{\mu_0 \epsilon_0}}$ |
| B inside capacitor | $B = \frac{\mu_0 J_D r}{2}$ (for $r < R$) |
| Poynting vector | $\vec{S} = \frac{1}{\mu_0}\vec{E} \times \vec{B}$ |

### Main Takeaways

1. **Ampère's law fails** for time-varying fields—different surfaces give different answers
2. **Displacement current** $\vec{J}_D = \epsilon_0 \partial\vec{E}/\partial t$ resolves the inconsistency
3. **Changing E-fields create B-fields**, just as changing B-fields create E-fields
4. **The factor $\mu_0\epsilon_0 = 1/c^2$** reveals that light is an electromagnetic wave
5. **Energy flows into capacitors** from the sides (via Poynting vector), not through the wires!
6. **In QED**, the displacement current describes virtual and real photon dynamics

## Daily Checklist

- [ ] I can explain why Ampère's law fails for time-varying fields
- [ ] I can derive the displacement current from charge conservation
- [ ] I can state the Ampère-Maxwell law in both forms
- [ ] I can calculate displacement current and the induced B-field in a capacitor
- [ ] I understand that $c = 1/\sqrt{\mu_0\epsilon_0}$ implies light is electromagnetic
- [ ] I can explain energy flow using the Poynting vector
- [ ] I completed the computational lab

## Preview: Day 213

Tomorrow we assemble the complete set of Maxwell's equations—the crowning achievement of classical electromagnetism. We'll explore their mathematical structure, symmetry properties, and the remarkable fact that these four equations describe all classical electromagnetic phenomena, from static charges to light waves.

---

*"This velocity is so nearly that of light, that it seems we have strong reason to conclude that light itself is an electromagnetic disturbance."* — James Clerk Maxwell, 1865
