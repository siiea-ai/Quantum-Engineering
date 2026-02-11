# Day 271: Animation with Matplotlib

## Schedule Overview
**Date**: Week 39, Day 5 (Friday)
**Duration**: 7 hours
**Theme**: Bringing Quantum Dynamics to Life Through Animation

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | FuncAnimation, wave packet evolution |
| Afternoon | 2.5 hours | Bloch sphere dynamics, multi-panel animations |
| Evening | 1.5 hours | Computational lab: Complete animation toolkit |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Create animations using `FuncAnimation` and `ArtistAnimation`
2. Animate time-evolving wave packets with proper physics
3. Visualize Bloch sphere dynamics for qubit evolution
4. Build multi-panel synchronized animations
5. Export animations to MP4, GIF, and other formats
6. Optimize animations for smooth playback

---

## Core Content

### 1. Animation Fundamentals

Matplotlib provides two animation approaches:
- **FuncAnimation**: Updates plot data each frame via a function
- **ArtistAnimation**: Pre-generates all frames as a list of artists

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib import animation

# Basic FuncAnimation structure
fig, ax = plt.subplots()
line, = ax.plot([], [])  # Empty line object

def init():
    """Initialize animation."""
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1.2, 1.2)
    line.set_data([], [])
    return line,

def update(frame):
    """Update animation for each frame."""
    x = np.linspace(0, 2*np.pi, 200)
    y = np.sin(x + frame/10)
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, frames=100, init_func=init,
                   blit=True, interval=50)
plt.show()
```

### 2. Wave Packet Time Evolution

The time-dependent Schrödinger equation governs wave packet evolution:
$$i\hbar\frac{\partial\psi}{\partial t} = \hat{H}\psi$$

For a free particle:
$$\psi(x, t) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} \phi(k) e^{i(kx - \omega(k)t)} dk$$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_free_wave_packet():
    """Animate a Gaussian wave packet in free space."""

    # Parameters
    x = np.linspace(-20, 20, 500)
    k0 = 3.0  # Initial momentum
    sigma0 = 1.0  # Initial width
    hbar = 1.0
    m = 1.0

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Initialize lines
    line_real, = axes[0].plot([], [], 'b-', linewidth=2, label='Re[ψ]')
    line_imag, = axes[0].plot([], [], 'r-', linewidth=2, label='Im[ψ]')
    line_prob, = axes[1].plot([], [], 'purple', linewidth=2)
    fill = axes[1].fill_between([], [], alpha=0.3, color='purple')

    # Formatting
    axes[0].set_xlim(-20, 20)
    axes[0].set_ylim(-0.8, 0.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlim(-20, 20)
    axes[1].set_ylim(0, 0.5)
    axes[1].set_xlabel('Position x')
    axes[1].set_ylabel('|ψ|²')
    axes[1].grid(True, alpha=0.3)

    time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes)

    def wave_packet(x, t):
        """Time-evolved Gaussian wave packet."""
        # Width spreading
        sigma_t = sigma0 * np.sqrt(1 + (hbar * t / (2 * m * sigma0**2))**2)

        # Normalization
        norm = (2 * np.pi * sigma_t**2)**(-0.25)

        # Classical trajectory
        x0 = hbar * k0 * t / m

        # Wave packet
        gaussian = np.exp(-(x - x0)**2 / (4 * sigma_t**2))

        # Phase
        phase = k0 * (x - x0/2) - hbar * k0**2 * t / (2 * m)

        # Chirp from spreading
        if t > 0:
            phase += (x - x0)**2 * hbar * t / (4 * m * sigma_t**2 * sigma0**2)

        return norm * gaussian * np.exp(1j * phase)

    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        line_prob.set_data([], [])
        time_text.set_text('')
        return line_real, line_imag, line_prob, time_text

    def update(frame):
        t = frame * 0.05
        psi = wave_packet(x, t)

        line_real.set_data(x, psi.real)
        line_imag.set_data(x, psi.imag)
        line_prob.set_data(x, np.abs(psi)**2)

        # Update fill (need to remove and recreate)
        for coll in axes[1].collections:
            coll.remove()
        axes[1].fill_between(x, np.abs(psi)**2, alpha=0.3, color='purple')

        time_text.set_text(f't = {t:.2f}')

        return line_real, line_imag, line_prob, time_text

    ani = FuncAnimation(fig, update, frames=200, init_func=init,
                       blit=False, interval=30)

    plt.suptitle('Free Particle Wave Packet Evolution', fontsize=14)
    plt.tight_layout()

    return fig, ani

fig, ani = animate_free_wave_packet()
# Save animation
# ani.save('wave_packet.mp4', writer='ffmpeg', fps=30, dpi=150)
# ani.save('wave_packet.gif', writer='pillow', fps=30)
plt.show()
```

### 3. Wave Packet in Harmonic Potential

In a harmonic potential, wave packets undergo periodic breathing and sloshing:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm

def animate_harmonic_oscillator():
    """Animate wave packet in harmonic potential."""

    # Grid
    N = 200
    L = 10
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    # Potential
    omega = 1.0
    V = 0.5 * omega**2 * x**2

    # Hamiltonian (finite difference)
    hbar = 1.0
    m = 1.0
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = hbar**2 / (m * dx**2) + V[i]
        if i > 0:
            H[i, i-1] = -hbar**2 / (2 * m * dx**2)
        if i < N-1:
            H[i, i+1] = -hbar**2 / (2 * m * dx**2)

    # Initial state: displaced Gaussian
    x0 = 3.0
    sigma = 0.7
    psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2))
    psi0 = psi0 / np.sqrt(np.trapz(np.abs(psi0)**2, x))

    # Time evolution operator
    dt = 0.02
    U = expm(-1j * H * dt / hbar)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot potential
    V_scaled = V / V.max() * 0.4
    ax.fill_between(x, V_scaled, alpha=0.2, color='gray', label='V(x)')
    ax.plot(x, V_scaled, 'k-', linewidth=1)

    line, = ax.plot([], [], 'b-', linewidth=2, label='|ψ|²')
    fill = None

    ax.set_xlim(-L, L)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel('Position x', fontsize=12)
    ax.set_ylabel('Probability / Energy', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    period_text = ax.text(0.02, 0.88, f'Period T = {2*np.pi/omega:.2f}',
                         transform=ax.transAxes, fontsize=10)

    psi = psi0.copy()

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        nonlocal psi, fill

        t = frame * dt
        prob = np.abs(psi)**2

        line.set_data(x, prob)

        # Update fill
        for coll in ax.collections[1:]:  # Keep potential fill
            coll.remove()
        ax.fill_between(x, prob, alpha=0.4, color='blue')

        time_text.set_text(f't = {t:.2f}, t/T = {t*omega/(2*np.pi):.2f}')

        # Time evolve
        psi = U @ psi

        return line, time_text

    ani = FuncAnimation(fig, update, frames=400, init_func=init,
                       blit=False, interval=20)

    plt.title('Wave Packet in Harmonic Potential', fontsize=14)
    plt.tight_layout()

    return fig, ani

fig, ani = animate_harmonic_oscillator()
plt.show()
```

### 4. Bloch Sphere Animation

Visualize qubit state evolution on the Bloch sphere:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def animate_bloch_sphere():
    """Animate state evolution on the Bloch sphere."""

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere wireframe
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray',
                     alpha=0.3, linewidth=0.5)

    # Axes
    ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'r-', linewidth=1)
    ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'g-', linewidth=1)
    ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'b-', linewidth=1)

    # Labels
    ax.text(1.4, 0, 0, 'x', fontsize=12)
    ax.text(0, 1.4, 0, 'y', fontsize=12)
    ax.text(0, 0, 1.4, '|0⟩', fontsize=12)
    ax.text(0, 0, -1.4, '|1⟩', fontsize=12)

    # Initialize state vector
    state_line, = ax.plot([], [], [], 'purple', linewidth=3)
    state_point, = ax.plot([], [], [], 'o', color='purple', markersize=10)
    trail, = ax.plot([], [], [], '-', color='purple', alpha=0.3, linewidth=1)

    # Trajectory storage
    trajectory = {'x': [], 'y': [], 'z': []}

    # Rabi oscillation parameters
    omega_rabi = 2.0  # Rabi frequency
    omega_0 = 0.0     # Detuning

    def bloch_vector(t):
        """Compute Bloch vector for Rabi oscillation."""
        # Starting from |0⟩, driving field along x-axis
        theta = omega_rabi * t  # Rotation about x-axis
        x = 0
        y = np.sin(theta)
        z = np.cos(theta)
        return x, y, z

    def init():
        state_line.set_data([], [])
        state_line.set_3d_properties([])
        state_point.set_data([], [])
        state_point.set_3d_properties([])
        trail.set_data([], [])
        trail.set_3d_properties([])
        return state_line, state_point, trail

    def update(frame):
        t = frame * 0.02

        x, y, z = bloch_vector(t)

        # Store trajectory
        trajectory['x'].append(x)
        trajectory['y'].append(y)
        trajectory['z'].append(z)

        # Keep only recent trajectory
        max_trail = 100
        for key in trajectory:
            if len(trajectory[key]) > max_trail:
                trajectory[key] = trajectory[key][-max_trail:]

        # Update state vector
        state_line.set_data([0, x], [0, y])
        state_line.set_3d_properties([0, z])
        state_point.set_data([x], [y])
        state_point.set_3d_properties([z])

        # Update trail
        trail.set_data(trajectory['x'], trajectory['y'])
        trail.set_3d_properties(trajectory['z'])

        ax.set_title(f'Rabi Oscillation: t = {t:.2f}, θ = {omega_rabi*t:.2f}',
                    fontsize=14)

        return state_line, state_point, trail

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect([1, 1, 1])

    ani = FuncAnimation(fig, update, frames=200, init_func=init,
                       blit=False, interval=40)

    return fig, ani

fig, ani = animate_bloch_sphere()
plt.show()
```

### 5. Multi-Panel Synchronized Animation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

def animate_quantum_tunneling():
    """Multi-panel animation of quantum tunneling."""

    # Parameters
    N = 300
    L = 30
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]

    # Potential barrier
    barrier_width = 2.0
    barrier_height = 5.0
    V = np.where(np.abs(x) < barrier_width/2, barrier_height, 0)

    # Hamiltonian
    hbar = 1.0
    m = 1.0
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = hbar**2 / (m * dx**2) + V[i]
        if i > 0:
            H[i, i-1] = -hbar**2 / (2 * m * dx**2)
        if i < N-1:
            H[i, i+1] = -hbar**2 / (2 * m * dx**2)

    # Initial wave packet
    x0 = -8
    k0 = 3.0
    sigma = 1.5
    psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi0 = psi0 / np.sqrt(np.trapz(np.abs(psi0)**2, x))

    # Time evolution
    from scipy.linalg import expm
    dt = 0.02
    U = expm(-1j * H * dt / hbar)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

    ax_main = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_trans = fig.add_subplot(gs[2, :])

    # Main plot
    ax_main.fill_between(x, V/barrier_height * 0.4, alpha=0.3, color='gray')
    ax_main.plot(x, V/barrier_height * 0.4, 'k-', linewidth=1)
    line_prob, = ax_main.plot([], [], 'b-', linewidth=2)
    ax_main.set_xlim(-L/2, L/2)
    ax_main.set_ylim(0, 0.6)
    ax_main.set_xlabel('Position x')
    ax_main.set_ylabel('|ψ|²')
    ax_main.set_title('Quantum Tunneling')
    time_text = ax_main.text(0.02, 0.92, '', transform=ax_main.transAxes)

    # Left region probability
    line_left, = ax_left.plot([], [], 'b-', linewidth=2)
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 1.1)
    ax_left.set_xlabel('Time')
    ax_left.set_ylabel('P(x < 0)')
    ax_left.set_title('Reflected')
    ax_left.grid(True, alpha=0.3)

    # Right region probability
    line_right, = ax_right.plot([], [], 'r-', linewidth=2)
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 0.3)
    ax_right.set_xlabel('Time')
    ax_right.set_ylabel('P(x > 0)')
    ax_right.set_title('Transmitted')
    ax_right.grid(True, alpha=0.3)

    # Transmission coefficient
    line_trans, = ax_trans.plot([], [], 'purple', linewidth=2)
    ax_trans.set_xlim(0, 10)
    ax_trans.set_ylim(0, 0.25)
    ax_trans.set_xlabel('Time')
    ax_trans.set_ylabel('T = P(x > barrier)')
    ax_trans.axhline(0.15, color='gray', linestyle='--', alpha=0.5,
                    label='Classical limit')
    ax_trans.legend()
    ax_trans.grid(True, alpha=0.3)

    # Storage for time evolution data
    times = []
    P_left = []
    P_right = []
    P_trans = []

    psi = psi0.copy()

    def init():
        line_prob.set_data([], [])
        line_left.set_data([], [])
        line_right.set_data([], [])
        line_trans.set_data([], [])
        time_text.set_text('')
        return line_prob, line_left, line_right, line_trans, time_text

    def update(frame):
        nonlocal psi

        t = frame * dt
        times.append(t)

        prob = np.abs(psi)**2

        # Calculate regional probabilities
        mask_left = x < -barrier_width/2
        mask_right = x > barrier_width/2

        p_left = np.trapz(prob[mask_left], x[mask_left])
        p_right = np.trapz(prob[mask_right], x[mask_right])

        P_left.append(p_left)
        P_right.append(p_right)
        P_trans.append(p_right)

        # Update plots
        line_prob.set_data(x, prob)

        # Update fill
        for coll in ax_main.collections[1:]:
            coll.remove()
        ax_main.fill_between(x, prob, alpha=0.4, color='blue')

        line_left.set_data(times, P_left)
        line_right.set_data(times, P_right)
        line_trans.set_data(times, P_trans)

        time_text.set_text(f't = {t:.2f}')

        # Time evolve
        psi = U @ psi

        return line_prob, line_left, line_right, line_trans, time_text

    ani = FuncAnimation(fig, update, frames=500, init_func=init,
                       blit=False, interval=20)

    plt.tight_layout()
    return fig, ani

fig, ani = animate_quantum_tunneling()
plt.show()
```

### 6. Saving Animations

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# Create a simple animation for demonstration
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot([], [])
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.2, 1.2)

def update(frame):
    line.set_data(x, np.sin(x + frame/10))
    return line,

ani = FuncAnimation(fig, update, frames=100, blit=True, interval=50)

# Save as MP4 (requires ffmpeg)
"""
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save('animation.mp4', writer=writer, dpi=150)
"""

# Save as GIF (requires pillow)
"""
writer = PillowWriter(fps=30)
ani.save('animation.gif', writer=writer, dpi=100)
"""

# Save as HTML5 video
"""
from matplotlib.animation import HTMLWriter
ani.save('animation.html', writer='html')
"""

# For Jupyter notebooks:
"""
from IPython.display import HTML
HTML(ani.to_jshtml())
"""
```

---

## Quantum Mechanics Connection

### Coherent State Evolution

Coherent states $$|\alpha\rangle$$ are minimum uncertainty states that remain Gaussian under harmonic evolution:

```python
def animate_coherent_state():
    """Animate coherent state in harmonic potential."""

    x = np.linspace(-10, 10, 500)
    omega = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Phase space (left panel)
    ax_phase = axes[0]
    ax_phase.set_xlim(-5, 5)
    ax_phase.set_ylim(-5, 5)
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('p')
    ax_phase.set_title('Phase Space')
    ax_phase.set_aspect('equal')
    ax_phase.grid(True, alpha=0.3)

    # Draw uncertainty circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 * np.cos(theta_circle)
    circle_y = 0.5 * np.sin(theta_circle)

    point, = ax_phase.plot([], [], 'o', color='blue', markersize=15)
    trail, = ax_phase.plot([], [], '-', color='blue', alpha=0.3)
    circle_patch, = ax_phase.plot([], [], '-', color='blue', alpha=0.5)

    # Position space (right panel)
    ax_pos = axes[1]
    ax_pos.set_xlim(-10, 10)
    ax_pos.set_ylim(0, 0.6)
    ax_pos.set_xlabel('x')
    ax_pos.set_ylabel('|ψ|²')
    ax_pos.set_title('Position Space')

    V = 0.5 * x**2
    ax_pos.fill_between(x, V/20, alpha=0.2, color='gray')
    line_prob, = ax_pos.plot([], [], 'b-', linewidth=2)

    # Initial coherent state amplitude
    alpha0 = 3.0

    # Storage
    trail_x = []
    trail_y = []

    def init():
        point.set_data([], [])
        trail.set_data([], [])
        circle_patch.set_data([], [])
        line_prob.set_data([], [])
        return point, trail, circle_patch, line_prob

    def update(frame):
        t = frame * 0.02

        # Coherent state evolution: α(t) = α₀ e^(-iωt)
        alpha_t = alpha0 * np.exp(-1j * omega * t)
        x_mean = alpha_t.real * np.sqrt(2)
        p_mean = alpha_t.imag * np.sqrt(2)

        trail_x.append(x_mean)
        trail_y.append(p_mean)

        # Keep trail limited
        if len(trail_x) > 200:
            trail_x.pop(0)
            trail_y.pop(0)

        point.set_data([x_mean], [p_mean])
        trail.set_data(trail_x, trail_y)
        circle_patch.set_data(x_mean + circle_x, p_mean + circle_y)

        # Position space probability
        sigma = 1/np.sqrt(2)
        prob = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(x - x_mean)**2 / (2*sigma**2))
        line_prob.set_data(x, prob)

        for coll in ax_pos.collections[1:]:
            coll.remove()
        ax_pos.fill_between(x, prob, alpha=0.4, color='blue')

        ax_phase.set_title(f'Phase Space: t = {t:.2f}')

        return point, trail, circle_patch, line_prob

    ani = FuncAnimation(fig, update, frames=400, init_func=init,
                       blit=False, interval=20)

    plt.suptitle('Coherent State in Harmonic Potential', fontsize=14)
    plt.tight_layout()

    return fig, ani
```

### Quantum Revival

In a particle-in-a-box, wave packets exhibit quantum revivals:

```python
def animate_quantum_revival():
    """Animate wave packet revival in infinite square well."""

    # Parameters
    L = 1.0  # Box width
    N = 200
    x = np.linspace(0, L, N)

    # Number of eigenstates to include
    n_max = 50

    # Initial wave packet (Gaussian centered in box)
    x0 = L/2
    sigma = L/20
    k0 = 20 * np.pi / L

    # Expand in eigenstates
    def eigenstate(n, x):
        return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

    def energy(n):
        return (n * np.pi / L)**2 / 2  # ℏ = m = 1

    # Coefficients
    psi0 = np.exp(-(x - x0)**2 / (2*sigma**2)) * np.exp(1j * k0 * x)
    psi0[0] = 0
    psi0[-1] = 0
    psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, x))

    coeffs = []
    for n in range(1, n_max + 1):
        phi_n = eigenstate(n, x)
        c_n = np.trapz(phi_n * psi0, x)
        coeffs.append(c_n)

    # Revival time
    T_revival = 4 * L**2 / np.pi  # T_rev = 4mL²/πℏ

    # Create animation
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axvline(0, color='k', linewidth=3)
    ax.axvline(L, color='k', linewidth=3)
    line, = ax.plot([], [], 'b-', linewidth=2)

    ax.set_xlim(-0.05, L+0.05)
    ax.set_ylim(0, 8)
    ax.set_xlabel('Position x/L')
    ax.set_ylabel('|ψ|²')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    revival_text = ax.text(0.02, 0.88, f'Revival time T = {T_revival:.2f}',
                          transform=ax.transAxes)

    def psi_t(t):
        """Time-evolved wave function."""
        psi = np.zeros_like(x, dtype=complex)
        for n, c_n in enumerate(coeffs, 1):
            E_n = energy(n)
            psi += c_n * eigenstate(n, x) * np.exp(-1j * E_n * t)
        return psi

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        t = frame * T_revival / 200

        psi = psi_t(t)
        prob = np.abs(psi)**2

        line.set_data(x, prob)

        for coll in ax.collections:
            coll.remove()
        ax.fill_between(x, prob, alpha=0.4, color='blue')

        time_text.set_text(f't/T_revival = {t/T_revival:.3f}')

        return line, time_text

    ani = FuncAnimation(fig, update, frames=400, init_func=init,
                       blit=False, interval=30)

    plt.title('Quantum Revival in Infinite Square Well')
    plt.tight_layout()

    return fig, ani
```

---

## Worked Examples

### Example 1: Animated Energy Level Transitions

**Problem**: Animate a quantum system transitioning between energy levels.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_transition():
    """Animate Rabi oscillation between two levels."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Energy levels (left panel)
    ax_levels = axes[0]
    ax_levels.set_xlim(0, 1)
    ax_levels.set_ylim(-0.5, 2)

    # Draw levels
    ax_levels.hlines([0, 1], 0.2, 0.8, colors=['blue', 'red'], linewidths=3)
    ax_levels.text(0.85, 0, '|g⟩', fontsize=14, va='center')
    ax_levels.text(0.85, 1, '|e⟩', fontsize=14, va='center')

    # Population bars
    pop_g = ax_levels.barh(0, 0, height=0.15, left=0.1, color='blue', alpha=0.7)
    pop_e = ax_levels.barh(1, 0, height=0.15, left=0.1, color='red', alpha=0.7)

    ax_levels.set_xticks([])
    ax_levels.set_yticks([])
    ax_levels.set_title('Energy Levels')

    # Time evolution (right panel)
    ax_time = axes[1]
    ax_time.set_xlim(0, 10)
    ax_time.set_ylim(0, 1.1)
    ax_time.set_xlabel('Time (2π/Ω)')
    ax_time.set_ylabel('Population')

    line_g, = ax_time.plot([], [], 'b-', linewidth=2, label='|g⟩')
    line_e, = ax_time.plot([], [], 'r-', linewidth=2, label='|e⟩')
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)

    times = []
    pops_g = []
    pops_e = []

    omega_rabi = 1.0

    def init():
        pop_g[0].set_width(0)
        pop_e[0].set_width(0)
        line_g.set_data([], [])
        line_e.set_data([], [])
        return pop_g, pop_e, line_g, line_e

    def update(frame):
        t = frame * 0.05

        # Rabi oscillation
        P_g = np.cos(omega_rabi * t / 2)**2
        P_e = np.sin(omega_rabi * t / 2)**2

        times.append(t)
        pops_g.append(P_g)
        pops_e.append(P_e)

        # Update bars (scale to fit)
        pop_g[0].set_width(P_g * 0.6)
        pop_e[0].set_width(P_e * 0.6)

        line_g.set_data(times, pops_g)
        line_e.set_data(times, pops_e)

        ax_time.set_title(f't = {t:.2f}: P_g = {P_g:.3f}, P_e = {P_e:.3f}')

        return pop_g, pop_e, line_g, line_e

    ani = FuncAnimation(fig, update, frames=400, init_func=init,
                       blit=False, interval=30)

    plt.suptitle('Two-Level System Rabi Oscillation', fontsize=14)
    plt.tight_layout()

    return fig, ani

fig, ani = animate_transition()
plt.show()
```

### Example 2: Interference Pattern Formation

**Problem**: Animate the build-up of a double-slit interference pattern from individual particle detections.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_interference_buildup():
    """Animate particle-by-particle build-up of interference pattern."""

    # Double slit parameters
    d = 1.0  # Slit separation
    wavelength = 0.1
    k = 2 * np.pi / wavelength

    # Screen
    x_screen = np.linspace(-3, 3, 500)

    # Interference pattern (probability)
    phase_diff = k * d * np.sin(np.arctan(x_screen / 10))
    prob = np.cos(phase_diff / 2)**2
    prob = prob / prob.sum()  # Normalize

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # Particle detection panel
    ax_detect = axes[0]
    ax_detect.set_xlim(-3, 3)
    ax_detect.set_ylim(0, 1)
    ax_detect.set_ylabel('Screen')
    ax_detect.set_title('Particle Detections')
    scatter = ax_detect.scatter([], [], s=2, c='blue', alpha=0.5)

    # Histogram panel
    ax_hist = axes[1]
    ax_hist.set_xlim(-3, 3)
    ax_hist.set_ylim(0, 100)
    ax_hist.set_xlabel('Position')
    ax_hist.set_ylabel('Counts')

    # Theoretical curve
    ax_hist.plot(x_screen, prob * 5000, 'r-', linewidth=2, alpha=0.5,
                label='Theory')
    ax_hist.legend()

    detected_x = []
    detected_y = []

    count_text = ax_detect.text(0.02, 0.95, '', transform=ax_detect.transAxes)

    def init():
        scatter.set_offsets(np.c_[[], []])
        count_text.set_text('')
        return scatter, count_text

    def update(frame):
        # Detect new particles (sample from probability distribution)
        n_new = np.random.poisson(5)  # Average 5 particles per frame

        new_x = np.random.choice(x_screen, size=n_new, p=prob)
        new_y = np.random.uniform(0.1, 0.9, size=n_new)

        detected_x.extend(new_x)
        detected_y.extend(new_y)

        scatter.set_offsets(np.c_[detected_x, detected_y])

        # Update histogram
        ax_hist.clear()
        ax_hist.hist(detected_x, bins=100, range=(-3, 3), alpha=0.7)
        ax_hist.plot(x_screen, prob * len(detected_x) * 6 / 500, 'r-',
                    linewidth=2, alpha=0.7, label='Theory')
        ax_hist.set_xlim(-3, 3)
        ax_hist.set_xlabel('Position')
        ax_hist.set_ylabel('Counts')
        ax_hist.legend()

        count_text.set_text(f'Total particles: {len(detected_x)}')

        return scatter, count_text

    ani = FuncAnimation(fig, update, frames=200, init_func=init,
                       blit=False, interval=50)

    plt.suptitle('Double Slit Experiment: Particle by Particle', fontsize=14)
    plt.tight_layout()

    return fig, ani

fig, ani = animate_interference_buildup()
plt.show()
```

---

## Practice Problems

### Level 1: Direct Application

1. **Simple Harmonic Motion**: Animate a point moving on a circle and its x-projection (simple harmonic motion) side by side.

2. **Traveling Wave**: Animate a sinusoidal wave $$y = A\sin(kx - \omega t)$$ moving to the right.

3. **Standing Wave**: Animate a standing wave with three nodes.

### Level 2: Intermediate

4. **Superposition**: Animate the sum of two waves with slightly different frequencies (beat pattern).

5. **Probability Current**: For a traveling wave packet, animate both $$|\psi|^2$$ and the probability current $$j = \text{Im}(\psi^* \nabla \psi)$$.

6. **Spin Precession**: Animate a spin-1/2 particle precessing in a magnetic field (show the spin vector on the Bloch sphere).

### Level 3: Challenging

7. **Two-Particle System**: Animate correlated motion of two particles (e.g., center of mass and relative coordinates).

8. **Adiabatic Evolution**: Animate a wave function following a slowly moving potential minimum, demonstrating the adiabatic theorem.

9. **Quantum Zeno Effect**: Animate wave packet evolution with and without repeated measurements, showing the difference in dynamics.

---

## Computational Lab

### Project: Complete Quantum Animation Toolkit

```python
"""
Quantum Animation Toolkit
Day 271: Animation with Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.linalg import expm
from scipy.special import hermite
from math import factorial


class QuantumAnimator:
    """
    Comprehensive toolkit for animating quantum mechanical systems.
    """

    def __init__(self, figsize=(12, 8), dpi=100):
        self.figsize = figsize
        self.dpi = dpi

    def animate_wave_packet(self, x, psi0, H, dt=0.02, n_frames=200, interval=30):
        """
        Generic wave packet animation.

        Parameters
        ----------
        x : array
            Position grid
        psi0 : array
            Initial wave function
        H : array
            Hamiltonian matrix
        dt : float
            Time step
        n_frames : int
            Number of animation frames
        interval : int
            Milliseconds between frames
        """
        U = expm(-1j * H * dt)
        psi = psi0.copy()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        line_real, = ax.plot([], [], 'b-', linewidth=1.5, label='Re[ψ]', alpha=0.7)
        line_imag, = ax.plot([], [], 'r-', linewidth=1.5, label='Im[ψ]', alpha=0.7)
        line_prob, = ax.plot([], [], 'purple', linewidth=2, label='|ψ|²')

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-1, 1)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            line_real.set_data([], [])
            line_imag.set_data([], [])
            line_prob.set_data([], [])
            return line_real, line_imag, line_prob, time_text

        def update(frame):
            nonlocal psi
            t = frame * dt

            prob = np.abs(psi)**2
            scale = 0.8 / max(prob.max(), 0.1)

            line_real.set_data(x, psi.real * scale)
            line_imag.set_data(x, psi.imag * scale)
            line_prob.set_data(x, prob * scale)

            time_text.set_text(f't = {t:.2f}')
            psi = U @ psi

            return line_real, line_imag, line_prob, time_text

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                           blit=True, interval=interval)
        return fig, ani

    def animate_harmonic_oscillator(self, n_state=0, x_range=(-6, 6), n_points=500):
        """Animate harmonic oscillator eigenstate with phase rotation."""

        x = np.linspace(*x_range, n_points)

        # Compute eigenstate
        xi = x
        prefactor = (1/np.pi)**0.25
        norm = 1 / np.sqrt(2**n_state * factorial(n_state))
        H_n = hermite(n_state)
        psi_n = prefactor * norm * H_n(xi) * np.exp(-xi**2/2)

        # Energy
        E_n = n_state + 0.5

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Phase space representation
        ax_phase = axes[0]
        ax_phase.set_xlim(-1.5, 1.5)
        ax_phase.set_ylim(-1.5, 1.5)
        ax_phase.set_xlabel('Re[c_n]')
        ax_phase.set_ylabel('Im[c_n]')
        ax_phase.set_aspect('equal')
        ax_phase.grid(True, alpha=0.3)

        circle = plt.Circle((0, 0), 1, fill=False, color='gray')
        ax_phase.add_patch(circle)
        phase_point, = ax_phase.plot([], [], 'o', color='blue', markersize=15)
        phase_line, = ax_phase.plot([], [], '-', color='blue', linewidth=2)

        # Position space
        ax_pos = axes[1]
        ax_pos.set_xlim(*x_range)
        ax_pos.set_ylim(-1, 1)
        ax_pos.set_xlabel('Position x')
        ax_pos.set_ylabel('ψ(x)')
        ax_pos.grid(True, alpha=0.3)

        line_psi, = ax_pos.plot([], [], 'b-', linewidth=2)
        ax_pos.plot(x, psi_n, 'k--', alpha=0.3, label='|ψ_n|')

        def init():
            phase_point.set_data([], [])
            phase_line.set_data([], [])
            line_psi.set_data([], [])
            return phase_point, phase_line, line_psi

        def update(frame):
            t = frame * 0.02
            phase = np.exp(-1j * E_n * t)

            # Phase space
            phase_point.set_data([phase.real], [phase.imag])
            phase_line.set_data([0, phase.real], [0, phase.imag])

            # Position space
            psi_t = psi_n * phase
            line_psi.set_data(x, psi_t.real)

            ax_pos.set_title(f'n={n_state}, t={t:.2f}, E={E_n}ℏω')

            return phase_point, phase_line, line_psi

        ani = FuncAnimation(fig, update, frames=200, init_func=init,
                           blit=True, interval=30)

        plt.suptitle(f'Harmonic Oscillator n={n_state} Phase Evolution', fontsize=14)
        plt.tight_layout()

        return fig, ani

    def animate_two_state_system(self, omega=1.0, delta=0.0, n_frames=300):
        """
        Animate two-level system dynamics.

        Parameters
        ----------
        omega : float
            Rabi frequency
        delta : float
            Detuning
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 2, figure=fig)

        ax_bloch = fig.add_subplot(gs[:, 0], projection='3d')
        ax_pop = fig.add_subplot(gs[0, 1])
        ax_phase = fig.add_subplot(gs[1, 1])

        # Bloch sphere setup
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x_s = np.outer(np.cos(u), np.sin(v))
        y_s = np.outer(np.sin(u), np.sin(v))
        z_s = np.outer(np.ones_like(u), np.cos(v))
        ax_bloch.plot_wireframe(x_s, y_s, z_s, color='lightgray', alpha=0.3)
        ax_bloch.set_xlim(-1.5, 1.5)
        ax_bloch.set_ylim(-1.5, 1.5)
        ax_bloch.set_zlim(-1.5, 1.5)

        state_line, = ax_bloch.plot([], [], [], 'b-', linewidth=3)
        state_point, = ax_bloch.plot([], [], [], 'o', color='blue', markersize=10)
        trail, = ax_bloch.plot([], [], [], '-', color='blue', alpha=0.3)

        # Population plot
        ax_pop.set_xlim(0, 10)
        ax_pop.set_ylim(0, 1.1)
        ax_pop.set_xlabel('Time')
        ax_pop.set_ylabel('Population')
        line_g, = ax_pop.plot([], [], 'b-', label='|g⟩')
        line_e, = ax_pop.plot([], [], 'r-', label='|e⟩')
        ax_pop.legend()
        ax_pop.grid(True, alpha=0.3)

        # Phase plot
        ax_phase.set_xlim(0, 10)
        ax_phase.set_ylim(-np.pi, np.pi)
        ax_phase.set_xlabel('Time')
        ax_phase.set_ylabel('Phase')
        line_phase, = ax_phase.plot([], [], 'g-')
        ax_phase.grid(True, alpha=0.3)

        # Data storage
        times = []
        pops_g = []
        pops_e = []
        phases = []
        trail_x = []
        trail_y = []
        trail_z = []

        # Generalized Rabi frequency
        Omega = np.sqrt(omega**2 + delta**2)

        def bloch_vector(t):
            """Compute Bloch vector for driven two-level system."""
            if Omega == 0:
                return 0, 0, 1

            # Rotation about axis determined by driving
            cos_theta = delta / Omega
            sin_theta = omega / Omega

            # Bloch vector components
            x = sin_theta * np.sin(Omega * t)
            y = sin_theta * (1 - np.cos(Omega * t)) * cos_theta
            z = cos_theta**2 + sin_theta**2 * np.cos(Omega * t)

            return x, y, z

        def init():
            state_line.set_data([], [])
            state_line.set_3d_properties([])
            state_point.set_data([], [])
            state_point.set_3d_properties([])
            trail.set_data([], [])
            trail.set_3d_properties([])
            line_g.set_data([], [])
            line_e.set_data([], [])
            line_phase.set_data([], [])
            return (state_line, state_point, trail, line_g, line_e, line_phase)

        def update(frame):
            t = frame * 0.03

            x, y, z = bloch_vector(t)

            times.append(t)
            pops_g.append((1 + z) / 2)
            pops_e.append((1 - z) / 2)
            phases.append(np.arctan2(y, x))

            trail_x.append(x)
            trail_y.append(y)
            trail_z.append(z)

            # Limit trail
            max_trail = 150
            for lst in [trail_x, trail_y, trail_z]:
                if len(lst) > max_trail:
                    lst.pop(0)

            state_line.set_data([0, x], [0, y])
            state_line.set_3d_properties([0, z])
            state_point.set_data([x], [y])
            state_point.set_3d_properties([z])
            trail.set_data(trail_x, trail_y)
            trail.set_3d_properties(trail_z)

            line_g.set_data(times, pops_g)
            line_e.set_data(times, pops_e)
            line_phase.set_data(times, phases)

            return (state_line, state_point, trail, line_g, line_e, line_phase)

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                           blit=False, interval=30)

        plt.suptitle(f'Two-Level System: Ω={omega}, Δ={delta}', fontsize=14)
        plt.tight_layout()

        return fig, ani


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Animation Toolkit")
    print("Day 271: Animation with Matplotlib")
    print("=" * 60)

    animator = QuantumAnimator(figsize=(12, 6))

    # Demo 1: Harmonic oscillator phase evolution
    print("\n1. Animating harmonic oscillator phase evolution...")
    fig1, ani1 = animator.animate_harmonic_oscillator(n_state=2)
    print("   Created harmonic oscillator animation")

    # Demo 2: Two-state system (Rabi oscillation)
    print("\n2. Animating two-level Rabi oscillation...")
    fig2, ani2 = animator.animate_two_state_system(omega=1.5, delta=0.5)
    print("   Created Rabi oscillation animation")

    # Demo 3: Free wave packet
    print("\n3. Creating free wave packet evolution...")
    N = 200
    L = 20
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]

    # Free particle Hamiltonian
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 1.0 / dx**2
        if i > 0:
            H[i, i-1] = -0.5 / dx**2
        if i < N-1:
            H[i, i+1] = -0.5 / dx**2

    # Initial Gaussian wave packet
    x0, k0, sigma = -5, 3, 1
    psi0 = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*k0*x)
    psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, x))

    fig3, ani3 = animator.animate_wave_packet(x, psi0, H, dt=0.03, n_frames=300)
    fig3.suptitle('Free Wave Packet Evolution', fontsize=14)
    print("   Created free particle animation")

    print("\n" + "=" * 60)
    print("Animations created!")
    print("Close each figure window to see the next animation.")
    print("=" * 60)

    plt.show()
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| `FuncAnimation` | Frame-by-frame animation via update function |
| `ArtistAnimation` | Pre-computed list of frames |
| `blit=True` | Optimize by only redrawing changed elements |
| `interval` | Milliseconds between frames |
| `frames` | Number of frames or iterable |
| `init_func` | Initialize animation state |

### Animation Workflow

```python
# 1. Create figure and initialize artists
fig, ax = plt.subplots()
line, = ax.plot([], [])

# 2. Define init function
def init():
    line.set_data([], [])
    return line,

# 3. Define update function
def update(frame):
    # Compute new data
    line.set_data(x, y_new)
    return line,

# 4. Create animation
ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=50)

# 5. Save or display
ani.save('animation.mp4', writer='ffmpeg', fps=30)
plt.show()
```

---

## Daily Checklist

- [ ] Created basic FuncAnimation
- [ ] Animated wave packet time evolution
- [ ] Visualized Bloch sphere dynamics
- [ ] Built multi-panel synchronized animations
- [ ] Saved animations to MP4/GIF formats
- [ ] Optimized animations with blitting
- [ ] Completed computational lab exercises

---

## Preview of Day 272

Tomorrow we focus on **Publication-Quality Figures**:
- LaTeX rendering in matplotlib
- Journal-specific formatting requirements
- Vector graphics export (PDF, SVG, EPS)
- Multi-panel figure layout optimization
- Creating a reusable style template

These skills ensure your visualizations meet the standards of scientific journals.
