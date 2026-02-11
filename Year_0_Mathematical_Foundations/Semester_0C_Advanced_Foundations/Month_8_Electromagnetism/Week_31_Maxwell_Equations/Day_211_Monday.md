# Day 211: Faraday's Law and Electromagnetic Induction

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning I | 2 hrs | Experimental foundations of electromagnetic induction |
| Morning II | 2 hrs | Mathematical formulation of Faraday's law |
| Afternoon | 2 hrs | Problem solving with motional EMF |
| Evening | 2 hrs | Computational lab: Visualizing induction |

## Learning Objectives

By the end of today, you will be able to:

1. **Describe Faraday's experimental discoveries** and their significance for unifying electricity and magnetism
2. **State Faraday's law** in both integral and differential forms
3. **Apply Lenz's law** to predict the direction of induced currents and EMFs
4. **Calculate motional EMF** for conductors moving through magnetic fields
5. **Derive induced electric fields** from time-varying magnetic fields
6. **Connect electromagnetic induction** to the quantum mechanical Aharonov-Bohm effect

## Core Content

### 1. Historical Introduction: Faraday's Discoveries

In 1831, Michael Faraday discovered that a changing magnetic field produces an electric current. His key experiments showed:

**Experiment 1: Moving magnet**
When a magnet is moved toward or away from a conducting loop, a current flows in the loop, but only while the magnet is moving.

**Experiment 2: Changing current**
When the current in one coil changes, it induces a current in a nearby coil, even with no direct electrical connection.

**Experiment 3: Moving loop**
Moving a conducting loop through a magnetic field region produces a current.

The unifying principle: **A changing magnetic flux through a circuit induces an EMF.**

### 2. Magnetic Flux

The magnetic flux through a surface $S$ is defined as:

$$\boxed{\Phi_B = \int_S \vec{B} \cdot d\vec{A} = \int_S B \cos\theta \, dA}$$

For a uniform field and flat surface:
$$\Phi_B = BA\cos\theta$$

where $\theta$ is the angle between $\vec{B}$ and the surface normal $\hat{n}$.

**Units:** Weber (Wb) = T·m² = V·s

### 3. Faraday's Law: Integral Form

**Faraday's Law:** The induced EMF in a closed loop equals the negative rate of change of magnetic flux through the loop:

$$\boxed{\mathcal{E} = -\frac{d\Phi_B}{dt}}$$

The EMF is defined as the work per unit charge done on a charge going around the loop:
$$\mathcal{E} = \oint \vec{E} \cdot d\vec{l}$$

Therefore, Faraday's law in integral form is:

$$\boxed{\oint \vec{E} \cdot d\vec{l} = -\frac{d\Phi_B}{dt} = -\frac{d}{dt}\int_S \vec{B} \cdot d\vec{A}}$$

### 4. Lenz's Law

The negative sign in Faraday's law embodies **Lenz's Law**:

> *The induced EMF creates a current whose magnetic field opposes the change in flux that produced it.*

This is a consequence of energy conservation: if the induced current enhanced the flux change, we'd get runaway energy production.

**Practical rule:** Point your thumb in the direction of *decreasing* flux; your curled fingers show the direction of induced current.

### 5. Faraday's Law: Differential Form

Applying Stokes' theorem to the integral form:
$$\oint \vec{E} \cdot d\vec{l} = \int_S (\nabla \times \vec{E}) \cdot d\vec{A}$$

Since this must hold for any surface $S$, we obtain:

$$\boxed{\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}}$$

This is the differential form of Faraday's law—one of Maxwell's equations.

**Profound implication:** A time-varying magnetic field creates an electric field that has non-zero curl. This is fundamentally different from electrostatic fields (where $\nabla \times \vec{E} = 0$).

### 6. Two Sources of Induced EMF

The flux $\Phi_B = \int \vec{B} \cdot d\vec{A}$ can change due to:

**Type 1: Changing magnetic field** (transformer EMF)
$$\mathcal{E} = -\int_S \frac{\partial \vec{B}}{\partial t} \cdot d\vec{A}$$

**Type 2: Moving loop** (motional EMF)
$$\mathcal{E} = \oint (\vec{v} \times \vec{B}) \cdot d\vec{l}$$

The general case combines both:
$$\mathcal{E} = -\frac{d\Phi_B}{dt} = \int_S \left(-\frac{\partial \vec{B}}{\partial t}\right) \cdot d\vec{A} + \oint (\vec{v} \times \vec{B}) \cdot d\vec{l}$$

### 7. Motional EMF: Detailed Analysis

Consider a conducting rod of length $L$ moving with velocity $\vec{v}$ perpendicular to a uniform magnetic field $\vec{B}$.

The Lorentz force on a charge $q$ in the rod:
$$\vec{F} = q\vec{v} \times \vec{B}$$

This force separates charges, creating a potential difference:
$$\mathcal{E} = \int_0^L (\vec{v} \times \vec{B}) \cdot d\vec{l} = vBL$$

**For a rectangular loop** of width $L$ with one side moving at velocity $v$:
$$\Phi_B = BLx \quad \Rightarrow \quad \mathcal{E} = -\frac{d\Phi_B}{dt} = -BL\frac{dx}{dt} = -BLv$$

The magnitude is $|BLv|$; the sign (direction) follows from Lenz's law.

### 8. Induced Electric Fields

When $\vec{B}$ changes in time but nothing moves, charges experience a force due to an induced electric field. This field is:

- **Non-conservative:** $\oint \vec{E} \cdot d\vec{l} \neq 0$
- **Solenoidal:** Forms closed loops around regions of changing $\vec{B}$
- **Determined by symmetry:** Use Faraday's law like Ampère's law

**Example: Cylindrical symmetry**

For a solenoid with increasing current, $\vec{B} = B(t)\hat{z}$ inside (radius $R$):

Outside ($r > R$):
$$E(2\pi r) = -\frac{d}{dt}(\pi R^2 B) \quad \Rightarrow \quad E = -\frac{R^2}{2r}\frac{dB}{dt}$$

Inside ($r < R$):
$$E(2\pi r) = -\frac{d}{dt}(\pi r^2 B) \quad \Rightarrow \quad E = -\frac{r}{2}\frac{dB}{dt}$$

The induced electric field circulates around the solenoid axis.

## Quantum Mechanics Connection

### The Aharonov-Bohm Effect

Faraday's law has a profound quantum mechanical extension. In the **Aharonov-Bohm effect** (1959):

1. Electrons travel around a solenoid containing magnetic flux $\Phi_B$
2. The electrons never enter the region where $\vec{B} \neq 0$
3. Yet their interference pattern depends on $\Phi_B$!

The quantum phase acquired by an electron is:
$$\phi = \frac{q}{\hbar}\oint \vec{A} \cdot d\vec{l} = \frac{q\Phi_B}{\hbar}$$

where $\vec{A}$ is the vector potential ($\vec{B} = \nabla \times \vec{A}$).

**Key insight:** In quantum mechanics, the vector potential $\vec{A}$ is more fundamental than $\vec{B}$. The Aharonov-Bohm effect shows that $\vec{A}$ can have physical effects even where $\vec{B} = 0$.

### Connection to Photons

Faraday's law describes how changing magnetic fields create electric fields. In quantum field theory:
- These coupled oscillating E and B fields are **photons**
- The photon is the quantum of electromagnetic induction
- Faraday's law becomes part of the QED Lagrangian

The classical EMF $\mathcal{E}$ corresponds to photon absorption/emission at the quantum level.

## Worked Examples

### Example 1: Rotating Loop in Magnetic Field

A rectangular loop of area $A$ rotates with angular velocity $\omega$ in a uniform field $\vec{B}$.

**Solution:**

The angle between $\vec{B}$ and the loop normal varies as $\theta = \omega t$.

Flux: $\Phi_B = BA\cos(\omega t)$

EMF:
$$\mathcal{E} = -\frac{d\Phi_B}{dt} = BA\omega\sin(\omega t)$$

$$\boxed{\mathcal{E} = \mathcal{E}_0 \sin(\omega t), \quad \mathcal{E}_0 = BA\omega}$$

This is the principle of the **AC generator**. The EMF oscillates sinusoidally with:
- Amplitude $\mathcal{E}_0 = BA\omega$
- Frequency $f = \omega/2\pi$

For $B = 0.5$ T, $A = 0.1$ m², $\omega = 120\pi$ rad/s (60 Hz):
$$\mathcal{E}_0 = (0.5)(0.1)(120\pi) = 18.8 \text{ V}$$

### Example 2: Induced E-field from Solenoid

A long solenoid of radius $R = 3$ cm has $n = 1000$ turns/m. The current increases at $dI/dt = 100$ A/s. Find the induced E-field at $r = 5$ cm from the axis.

**Solution:**

Inside the solenoid: $B = \mu_0 n I$

Rate of change: $\frac{dB}{dt} = \mu_0 n \frac{dI}{dt}$

Since $r > R$, we're outside the solenoid. Using Faraday's law:
$$\oint \vec{E} \cdot d\vec{l} = E(2\pi r) = -\frac{d\Phi_B}{dt} = -\pi R^2 \frac{dB}{dt}$$

$$E = -\frac{R^2}{2r}\mu_0 n \frac{dI}{dt}$$

Substituting values:
$$E = -\frac{(0.03)^2}{2(0.05)}(4\pi \times 10^{-7})(1000)(100)$$
$$E = -1.13 \times 10^{-3} \text{ V/m}$$

The magnitude is $|E| = 1.13$ mV/m. The direction is tangential, opposing the increase in flux.

### Example 3: Sliding Bar on Rails

A conducting bar slides on frictionless rails in a uniform magnetic field $B = 0.8$ T (into the page). The bar has length $L = 0.5$ m, mass $m = 0.2$ kg, and resistance $R = 2\ \Omega$. It starts from rest and slides down a slope at angle $\theta = 30°$.

Find the terminal velocity.

**Solution:**

At velocity $v$, the induced EMF is:
$$\mathcal{E} = BLv$$

Induced current:
$$I = \frac{\mathcal{E}}{R} = \frac{BLv}{R}$$

Magnetic force on the bar (opposing motion):
$$F_{mag} = BIL = \frac{B^2L^2v}{R}$$

At terminal velocity, forces balance:
$$mg\sin\theta = \frac{B^2L^2v_{term}}{R}$$

$$v_{term} = \frac{mgR\sin\theta}{B^2L^2}$$

$$v_{term} = \frac{(0.2)(9.8)(2)\sin(30°)}{(0.8)^2(0.5)^2} = \frac{1.96}{0.16} = 12.25 \text{ m/s}$$

$$\boxed{v_{term} = 12.25 \text{ m/s}}$$

## Practice Problems

### Level 1: Direct Application

1. A circular loop of radius 10 cm is in a magnetic field that decreases uniformly from 0.5 T to 0.1 T in 0.2 s. Calculate the induced EMF.

2. A rectangular loop (20 cm × 30 cm) rotates at 50 Hz in a 0.4 T field. Find the maximum EMF.

3. A solenoid of radius 5 cm and 500 turns/m has current changing at 50 A/s. Find the induced E-field at r = 3 cm.

### Level 2: Intermediate

4. A circular loop of wire lies in the xy-plane. The magnetic field is $\vec{B} = B_0(1 + t^2)\hat{z}$ T. If the loop has radius $R$ and resistance $r$, find the current as a function of time.

5. Two concentric circular loops have radii $a$ and $b$ ($a \ll b$). The outer loop carries current $I = I_0\cos(\omega t)$. Find the EMF induced in the inner loop.

6. A conducting disk of radius $R$ rotates about its axis with angular velocity $\omega$ in a uniform field $\vec{B} = B\hat{z}$. Find the EMF between the center and the rim.

### Level 3: Challenging

7. A rectangular loop moves with constant velocity $v$ toward an infinite wire carrying current $I$. The loop has dimensions $a \times b$ and its near edge is at distance $s$ from the wire. Find the induced EMF as a function of $s$.

8. A toroidal coil has $N$ turns, inner radius $a$, outer radius $b$, and height $h$. The current increases as $I = I_0 t$. Find the induced E-field at a point inside the torus at radius $r$ ($a < r < b$).

9. **Quantum connection:** In an Aharonov-Bohm experiment, electrons (energy 50 keV) pass around a solenoid with flux $\Phi_B$. If the first minimum in the interference pattern occurs when $\Phi_B = \Phi_0$, find $\Phi_0$ in terms of fundamental constants.

## Computational Lab: Visualizing Electromagnetic Induction

```python
"""
Day 211 Computational Lab: Electromagnetic Induction Visualization
Topics: Faraday's law, induced E-fields, flux animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Set up styling
plt.style.use('default')

# =============================================================================
# Part 1: Magnetic Flux Through a Tilting Loop
# =============================================================================

def plot_flux_vs_angle():
    """Visualize flux through a loop at various angles."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Parameters
    B = 1.0  # T
    A = 0.1  # m²

    # Angles
    angles = np.linspace(0, 2*np.pi, 200)
    flux = B * A * np.cos(angles)
    emf = B * A * np.sin(angles)  # For ω = 1 rad/s

    # Plot 1: Flux vs angle
    axes[0].plot(np.degrees(angles), flux * 1000, 'b-', linewidth=2)
    axes[0].set_xlabel('Angle θ (degrees)')
    axes[0].set_ylabel('Magnetic Flux Φ_B (mWb)')
    axes[0].set_title('Magnetic Flux vs Loop Orientation')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 360)

    # Plot 2: EMF vs angle (for rotating loop)
    axes[1].plot(np.degrees(angles), emf * 1000, 'r-', linewidth=2)
    axes[1].set_xlabel('Angle θ (degrees)')
    axes[1].set_ylabel('Induced EMF (mV)')
    axes[1].set_title('EMF in Rotating Loop (ω = 1 rad/s)')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 360)

    # Plot 3: Phase relationship
    axes[2].plot(np.degrees(angles), flux/np.max(np.abs(flux)), 'b-',
                 linewidth=2, label='Flux (normalized)')
    axes[2].plot(np.degrees(angles), emf/np.max(np.abs(emf)), 'r--',
                 linewidth=2, label='EMF (normalized)')
    axes[2].set_xlabel('Angle θ (degrees)')
    axes[2].set_ylabel('Normalized Value')
    axes[2].set_title('Phase Relationship: EMF lags Flux by 90°')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 360)

    plt.tight_layout()
    plt.savefig('flux_angle_relationship.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 2: Induced Electric Field Around a Solenoid
# =============================================================================

def plot_induced_E_field():
    """Visualize induced E-field from time-varying B in solenoid."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parameters
    R = 0.05  # Solenoid radius (m)
    dB_dt = 1.0  # Rate of change of B (T/s)

    # Radial positions
    r = np.linspace(0.001, 0.15, 200)

    # Induced E-field
    E = np.where(r < R,
                 r * dB_dt / 2,  # Inside
                 R**2 * dB_dt / (2 * r))  # Outside

    # Plot 1: E vs r
    axes[0].plot(r * 100, E * 1000, 'b-', linewidth=2)
    axes[0].axvline(x=R*100, color='r', linestyle='--', label=f'Solenoid edge (R={R*100} cm)')
    axes[0].set_xlabel('Radial Distance r (cm)')
    axes[0].set_ylabel('Induced Electric Field |E| (mV/m)')
    axes[0].set_title('Induced E-field vs Distance from Solenoid Axis')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Vector field
    ax = axes[1]

    # Create grid
    x = np.linspace(-0.12, 0.12, 15)
    y = np.linspace(-0.12, 0.12, 15)
    X, Y = np.meshgrid(x, y)
    r_grid = np.sqrt(X**2 + Y**2)

    # E-field magnitude
    E_mag = np.where(r_grid < R,
                     r_grid * dB_dt / 2,
                     R**2 * dB_dt / (2 * r_grid + 1e-10))

    # E-field direction (tangential, counterclockwise for increasing B)
    # If dB/dt > 0 and B points out of page, E circulates clockwise (Lenz's law)
    theta = np.arctan2(Y, X)
    Ex = -E_mag * np.sin(theta)  # Tangential component x
    Ey = E_mag * np.cos(theta)   # Tangential component y

    # Normalize for visualization
    E_norm = np.sqrt(Ex**2 + Ey**2)
    Ex_n = Ex / (E_norm + 1e-10)
    Ey_n = Ey / (E_norm + 1e-10)

    # Plot
    ax.quiver(X*100, Y*100, Ex_n, Ey_n, E_norm*1000, cmap='viridis', alpha=0.8)
    circle = Circle((0, 0), R*100, fill=False, color='red', linewidth=2,
                    linestyle='--', label='Solenoid')
    ax.add_patch(circle)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title('Induced E-field Pattern (B increasing out of page)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('induced_E_field.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 3: AC Generator Animation
# =============================================================================

def animate_generator():
    """Animate a rotating loop in a magnetic field (AC generator)."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parameters
    omega = 2 * np.pi  # Angular frequency (rad/s)
    B = 1.0  # Magnetic field (T)
    A = 0.1  # Area (m²)

    # Time array
    t_max = 2.0  # seconds
    t = np.linspace(0, t_max, 200)

    # Flux and EMF
    flux = B * A * np.cos(omega * t)
    emf = B * A * omega * np.sin(omega * t)

    # Initialize plots
    axes[0].set_xlim(-1.5, 1.5)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_aspect('equal')
    axes[0].set_title('Rotating Loop in B-field')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')

    # Draw B-field arrows (pointing right)
    for y_pos in np.linspace(-1.2, 1.2, 5):
        axes[0].annotate('', xy=(1.2, y_pos), xytext=(-1.2, y_pos),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.3))
    axes[0].text(0, 1.4, '$\\vec{B}$', fontsize=14, ha='center', color='blue')

    # Initialize loop line
    loop_line, = axes[0].plot([], [], 'r-', linewidth=3)
    time_text = axes[0].text(-1.4, -1.4, '', fontsize=12)

    # Flux and EMF plot
    axes[1].set_xlim(0, t_max)
    axes[1].set_ylim(-0.8, 0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Flux and Induced EMF vs Time')
    axes[1].grid(True, alpha=0.3)

    flux_line, = axes[1].plot([], [], 'b-', linewidth=2, label='Flux Φ (Wb)')
    emf_line, = axes[1].plot([], [], 'r-', linewidth=2, label='EMF (V)')
    marker_flux, = axes[1].plot([], [], 'bo', markersize=10)
    marker_emf, = axes[1].plot([], [], 'ro', markersize=10)
    axes[1].legend(loc='upper right')

    def init():
        loop_line.set_data([], [])
        flux_line.set_data([], [])
        emf_line.set_data([], [])
        marker_flux.set_data([], [])
        marker_emf.set_data([], [])
        time_text.set_text('')
        return loop_line, flux_line, emf_line, marker_flux, marker_emf, time_text

    def animate(frame):
        current_t = frame * t_max / 100
        theta = omega * current_t

        # Loop projection (rotating in xz plane, B along x)
        # The loop appears as an ellipse when viewed along y
        width = 1.0 * np.cos(theta)  # Projection
        height = 1.0

        # Draw loop as a line (edge-on view)
        loop_x = [-width/2, width/2]
        loop_y = [-height/2, -height/2]
        loop_line.set_data([loop_x[0], loop_x[1], loop_x[1], loop_x[0], loop_x[0]],
                          [loop_y[0], loop_y[0], -loop_y[0], -loop_y[0], loop_y[0]])

        # Update time traces
        idx = int(frame * len(t) / 100)
        flux_line.set_data(t[:idx], flux[:idx])
        emf_line.set_data(t[:idx], emf[:idx])

        if idx > 0:
            marker_flux.set_data([t[idx-1]], [flux[idx-1]])
            marker_emf.set_data([t[idx-1]], [emf[idx-1]])

        time_text.set_text(f't = {current_t:.2f} s')

        return loop_line, flux_line, emf_line, marker_flux, marker_emf, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=100,
                        interval=50, blit=True)

    plt.tight_layout()
    plt.savefig('generator_snapshot.png', dpi=150, bbox_inches='tight')
    plt.show()

    return anim

# =============================================================================
# Part 4: Motional EMF - Bar Sliding on Rails
# =============================================================================

def simulate_sliding_bar():
    """Simulate a conducting bar sliding on rails with magnetic braking."""

    # Parameters
    B = 0.8  # T
    L = 0.5  # m
    m = 0.2  # kg
    R = 2.0  # Ohms
    theta = 30  # degrees
    g = 9.8  # m/s²

    # Time parameters
    dt = 0.001
    t_max = 5.0

    # Arrays
    t = [0]
    v = [0]
    x = [0]
    I = [0]
    P_dissipated = [0]

    # Simulation
    while t[-1] < t_max:
        # Current velocity
        v_curr = v[-1]

        # Induced EMF and current
        emf = B * L * v_curr
        I_curr = emf / R

        # Forces
        F_gravity = m * g * np.sin(np.radians(theta))
        F_magnetic = B * I_curr * L  # Opposes motion
        F_net = F_gravity - F_magnetic

        # Acceleration and update
        a = F_net / m
        v_new = v_curr + a * dt
        x_new = x[-1] + v_curr * dt

        # Power dissipated
        P = I_curr**2 * R

        # Store
        t.append(t[-1] + dt)
        v.append(v_new)
        x.append(x_new)
        I.append(I_curr)
        P_dissipated.append(P)

    # Convert to arrays
    t = np.array(t)
    v = np.array(v)
    x = np.array(x)
    I = np.array(I)
    P = np.array(P_dissipated)

    # Terminal velocity
    v_terminal = m * g * R * np.sin(np.radians(theta)) / (B**2 * L**2)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Velocity
    axes[0, 0].plot(t, v, 'b-', linewidth=2)
    axes[0, 0].axhline(y=v_terminal, color='r', linestyle='--',
                       label=f'Terminal velocity = {v_terminal:.2f} m/s')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title('Velocity vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Position
    axes[0, 1].plot(t, x, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].set_title('Position vs Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Current
    axes[1, 0].plot(t, I, 'r-', linewidth=2)
    I_terminal = B * L * v_terminal / R
    axes[1, 0].axhline(y=I_terminal, color='k', linestyle='--',
                       label=f'Terminal current = {I_terminal:.3f} A')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Current (A)')
    axes[1, 0].set_title('Induced Current vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Power dissipated
    axes[1, 1].plot(t, P, 'm-', linewidth=2)
    P_terminal = I_terminal**2 * R
    axes[1, 1].axhline(y=P_terminal, color='k', linestyle='--',
                       label=f'Terminal power = {P_terminal:.3f} W')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Power (W)')
    axes[1, 1].set_title('Power Dissipated in Resistance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Magnetic Braking: B={B} T, L={L} m, m={m} kg, R={R} Ω, θ={theta}°',
                fontsize=12)
    plt.tight_layout()
    plt.savefig('sliding_bar_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Terminal velocity: {v_terminal:.3f} m/s")
    print(f"Terminal current: {I_terminal:.4f} A")
    print(f"Terminal power dissipation: {P_terminal:.4f} W")

# =============================================================================
# Part 5: Aharonov-Bohm Phase Visualization
# =============================================================================

def visualize_aharonov_bohm():
    """Visualize the Aharonov-Bohm effect phase."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Physical constants
    hbar = 1.055e-34  # J·s
    e = 1.602e-19     # C
    phi_0 = 2 * np.pi * hbar / e  # Flux quantum (h/e)

    # Flux values
    flux = np.linspace(0, 3 * phi_0, 300)
    phase = e * flux / hbar

    # Interference pattern
    # Two-path interference: I = I_0 * cos²(phase_difference/2)
    intensity = np.cos(phase / 2)**2

    # Plot 1: Phase vs flux
    axes[0].plot(flux / phi_0, phase, 'b-', linewidth=2)
    axes[0].set_xlabel('Magnetic Flux Φ / Φ₀')
    axes[0].set_ylabel('Phase φ (radians)')
    axes[0].set_title('Aharonov-Bohm Phase vs Enclosed Flux')
    axes[0].axvline(x=1, color='r', linestyle='--', alpha=0.5)
    axes[0].axvline(x=2, color='r', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # Add annotations
    axes[0].annotate('Φ₀ = h/e\n= 4.14×10⁻¹⁵ Wb', xy=(1, np.pi),
                     xytext=(1.5, np.pi + 2), fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='red'))

    # Plot 2: Interference intensity vs flux
    axes[1].plot(flux / phi_0, intensity, 'r-', linewidth=2)
    axes[1].fill_between(flux / phi_0, intensity, alpha=0.3)
    axes[1].set_xlabel('Magnetic Flux Φ / Φ₀')
    axes[1].set_ylabel('Interference Intensity (normalized)')
    axes[1].set_title('Aharonov-Bohm Interference Pattern')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)

    # Mark constructive/destructive interference
    axes[1].axvline(x=0, color='g', linestyle=':', alpha=0.7, label='Constructive')
    axes[1].axvline(x=2, color='g', linestyle=':', alpha=0.7)
    axes[1].axvline(x=1, color='purple', linestyle=':', alpha=0.7, label='Destructive')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('aharonov_bohm.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Flux quantum Φ₀ = h/e = {phi_0:.3e} Wb")
    print(f"Phase shift for one flux quantum: 2π radians")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Day 211: Electromagnetic Induction Computational Lab")
    print("="*60)

    print("\n1. Flux and EMF vs Angle Relationship")
    plot_flux_vs_angle()

    print("\n2. Induced Electric Field Around Solenoid")
    plot_induced_E_field()

    print("\n3. AC Generator Animation")
    anim = animate_generator()

    print("\n4. Sliding Bar with Magnetic Braking")
    simulate_sliding_bar()

    print("\n5. Aharonov-Bohm Effect Visualization")
    visualize_aharonov_bohm()

    print("\nAll visualizations complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Magnetic flux | $\Phi_B = \int \vec{B} \cdot d\vec{A}$ |
| Faraday's law (integral) | $\mathcal{E} = -\frac{d\Phi_B}{dt}$ |
| Faraday's law (differential) | $\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$ |
| Motional EMF | $\mathcal{E} = \oint (\vec{v} \times \vec{B}) \cdot d\vec{l}$ |
| Induced E (inside solenoid) | $E = -\frac{r}{2}\frac{dB}{dt}$ |
| Induced E (outside solenoid) | $E = -\frac{R^2}{2r}\frac{dB}{dt}$ |
| AC generator EMF | $\mathcal{E} = BA\omega\sin(\omega t)$ |
| Aharonov-Bohm phase | $\phi = \frac{e\Phi_B}{\hbar}$ |

### Main Takeaways

1. **Faraday's discovery** that changing magnetic flux induces EMF unified electricity and magnetism
2. **Lenz's law** ensures energy conservation—induced currents oppose flux changes
3. **Induced electric fields** from time-varying $\vec{B}$ are non-conservative (closed loops)
4. **Motional EMF** arises from the Lorentz force on charges in moving conductors
5. **The Aharonov-Bohm effect** shows quantum mechanics requires the vector potential $\vec{A}$, not just $\vec{B}$

## Daily Checklist

- [ ] I can state Faraday's law in both integral and differential forms
- [ ] I can apply Lenz's law to predict induced current directions
- [ ] I can calculate motional EMF for moving conductors
- [ ] I can determine induced E-fields from time-varying B-fields
- [ ] I understand the difference between transformer and motional EMF
- [ ] I can explain the Aharonov-Bohm effect and its quantum significance
- [ ] I completed the computational lab visualizations

## Preview: Day 212

Tomorrow we address a profound inconsistency in electromagnetism: Ampère's law fails for time-varying fields! Maxwell's brilliant insight—the **displacement current**—resolves this problem and completes the theoretical framework. This seemingly small correction has enormous consequences: it predicts electromagnetic waves.

---

*"The investigation of the action of currents and magnets has conducted us to the conclusion that the field in which the current flows contains a form of energy, which is everywhere the same in kind."* — James Clerk Maxwell
