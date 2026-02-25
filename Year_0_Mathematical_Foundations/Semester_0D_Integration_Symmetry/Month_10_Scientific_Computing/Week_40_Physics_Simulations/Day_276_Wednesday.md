# Day 276: Wave Equation Simulations

## Schedule Overview
**Date**: Week 40, Day 3 (Wednesday)
**Duration**: 7 hours
**Theme**: Numerical Solutions to the Classical Wave Equation

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Finite difference methods, stability analysis |
| Afternoon | 2.5 hours | Standing waves, normal modes, boundary conditions |
| Evening | 1.5 hours | Computational lab: 2D wave simulation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Derive finite difference approximations for the wave equation
2. Implement stable numerical schemes (FTCS, Leapfrog)
3. Understand the CFL stability condition
4. Simulate standing waves and normal modes
5. Apply various boundary conditions (fixed, free, periodic)
6. Connect classical waves to quantum wave mechanics

---

## Core Content

### 1. The Wave Equation

The 1D wave equation describes waves on strings, in air, and electromagnetic waves:
$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

where $$c$$ is the wave speed.

### 2. Finite Difference Discretization

Discretize on a grid with spacing $$\Delta x$$ and $$\Delta t$$:

**Second derivative in space:**
$$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{\Delta x^2}$$

**Second derivative in time:**
$$\frac{\partial^2 u}{\partial t^2} \approx \frac{u_i^{n+1} - 2u_i^n + u_i^{n-1}}{\Delta t^2}$$

This gives the **explicit leapfrog scheme**:
$$u_i^{n+1} = 2u_i^n - u_i^{n-1} + \left(\frac{c\Delta t}{\Delta x}\right)^2 (u_{i+1}^n - 2u_i^n + u_{i-1}^n)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class WaveEquation1D:
    """
    Solve 1D wave equation using finite differences.

    ∂²u/∂t² = c² ∂²u/∂x²
    """

    def __init__(self, L=10.0, c=1.0, nx=200, cfl=0.5):
        """
        Initialize wave equation solver.

        Parameters
        ----------
        L : float
            Domain length
        c : float
            Wave speed
        nx : int
            Number of spatial grid points
        cfl : float
            CFL number (must be ≤ 1 for stability)
        """
        self.L = L
        self.c = c
        self.nx = nx
        self.dx = L / (nx - 1)
        self.dt = cfl * self.dx / c  # CFL condition
        self.cfl = cfl

        self.x = np.linspace(0, L, nx)
        self.u = np.zeros(nx)  # Current state
        self.u_prev = np.zeros(nx)  # Previous state
        self.u_next = np.zeros(nx)  # Next state

    def set_initial_condition(self, u0_func, v0_func=None):
        """
        Set initial displacement and velocity.

        u(x, 0) = u0_func(x)
        ∂u/∂t(x, 0) = v0_func(x)
        """
        self.u = u0_func(self.x).copy()

        # First step using initial velocity
        if v0_func is not None:
            v0 = v0_func(self.x)
            # u_prev estimated from: u(t=-dt) ≈ u(0) - v(0)*dt
            self.u_prev = self.u - v0 * self.dt
        else:
            self.u_prev = self.u.copy()

    def apply_boundary(self, bc_type='fixed'):
        """Apply boundary conditions."""
        if bc_type == 'fixed':
            self.u_next[0] = 0
            self.u_next[-1] = 0
        elif bc_type == 'free':
            self.u_next[0] = self.u_next[1]
            self.u_next[-1] = self.u_next[-2]
        elif bc_type == 'periodic':
            self.u_next[0] = self.u_next[-2]
            self.u_next[-1] = self.u_next[1]

    def step(self, bc_type='fixed'):
        """Advance one time step using leapfrog."""
        r2 = self.cfl**2

        # Interior points
        self.u_next[1:-1] = (2 * self.u[1:-1] - self.u_prev[1:-1] +
                            r2 * (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]))

        # Boundary conditions
        self.apply_boundary(bc_type)

        # Cycle arrays
        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()

    def solve(self, t_max, bc_type='fixed', save_every=10):
        """Solve for time t_max."""
        n_steps = int(t_max / self.dt)
        history = [self.u.copy()]
        times = [0]

        for step in range(n_steps):
            self.step(bc_type)
            if step % save_every == 0:
                history.append(self.u.copy())
                times.append((step + 1) * self.dt)

        return np.array(times), np.array(history)

    def animate(self, t_max, bc_type='fixed', interval=20):
        """Create animation of wave evolution."""
        n_steps = int(t_max / self.dt)

        fig, ax = plt.subplots(figsize=(12, 6))
        line, = ax.plot(self.x, self.u, 'b-', linewidth=2)
        ax.set_xlim(0, self.L)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.grid(True, alpha=0.3)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def update(frame):
            for _ in range(5):  # Multiple steps per frame
                self.step(bc_type)
            line.set_ydata(self.u)
            time_text.set_text(f't = {frame * 5 * self.dt:.2f}')
            return line, time_text

        ani = FuncAnimation(fig, update, frames=n_steps//5,
                           blit=True, interval=interval)
        return fig, ani

# Demonstration: Gaussian pulse
wave = WaveEquation1D(L=10, c=1.0, nx=200, cfl=0.9)

# Initial Gaussian pulse
def gaussian_pulse(x):
    return np.exp(-(x - 5)**2 / 0.5)

wave.set_initial_condition(gaussian_pulse)
times, history = wave.solve(t_max=15, bc_type='fixed', save_every=20)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Initial vs final
axes[0, 0].plot(wave.x, history[0], 'b-', label='Initial')
axes[0, 0].plot(wave.x, history[-1], 'r-', label='Final')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('u')
axes[0, 0].legend()
axes[0, 0].set_title('Wave Displacement')

# Space-time diagram
im = axes[0, 1].imshow(history, aspect='auto', origin='lower',
                       extent=[0, wave.L, 0, times[-1]],
                       cmap='RdBu', vmin=-1, vmax=1)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('t')
axes[0, 1].set_title('Space-Time Diagram')
plt.colorbar(im, ax=axes[0, 1])

# Energy conservation
energy = np.sum(history**2, axis=1) * wave.dx
axes[1, 0].plot(times, energy / energy[0])
axes[1, 0].set_xlabel('t')
axes[1, 0].set_ylabel('E(t) / E(0)')
axes[1, 0].set_title('Energy Conservation')
axes[1, 0].set_ylim(0.9, 1.1)

# Snapshots
for i in range(0, len(history), len(history)//5):
    axes[1, 1].plot(wave.x, history[i], label=f't = {times[i]:.1f}')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('u')
axes[1, 1].legend()
axes[1, 1].set_title('Wave Snapshots')

plt.tight_layout()
plt.savefig('wave_equation_1d.png', dpi=150)
plt.show()
```

### 3. Stability Analysis: CFL Condition

The Courant-Friedrichs-Lewy (CFL) condition:
$$\frac{c\Delta t}{\Delta x} \leq 1$$

```python
def demonstrate_cfl_stability():
    """Show the effect of CFL number on stability."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cfl_values = [0.5, 1.0, 1.1]
    titles = ['Stable (CFL=0.5)', 'Marginal (CFL=1.0)', 'Unstable (CFL=1.1)']

    for ax, cfl, title in zip(axes, cfl_values, titles):
        wave = WaveEquation1D(L=10, c=1.0, nx=100, cfl=cfl)
        wave.set_initial_condition(lambda x: np.exp(-(x-5)**2))

        try:
            times, history = wave.solve(t_max=10, save_every=10)
            im = ax.imshow(history, aspect='auto', origin='lower',
                          extent=[0, 10, 0, times[-1]],
                          cmap='RdBu', vmin=-2, vmax=2)
        except:
            ax.text(0.5, 0.5, 'Numerical blowup', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)

        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig('cfl_stability.png', dpi=150)
    plt.show()

demonstrate_cfl_stability()
```

### 4. Standing Waves and Normal Modes

For fixed boundary conditions, normal modes are:
$$u_n(x, t) = \sin\left(\frac{n\pi x}{L}\right)\cos(\omega_n t)$$
where $$\omega_n = \frac{n\pi c}{L}$$

```python
def standing_wave_modes():
    """Demonstrate standing wave normal modes."""

    L = 10.0
    c = 1.0
    x = np.linspace(0, L, 200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mode shapes
    ax1 = axes[0, 0]
    for n in range(1, 5):
        u_n = np.sin(n * np.pi * x / L)
        ax1.plot(x, u_n + 2*(n-1), label=f'n = {n}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Mode shape (offset)')
    ax1.set_title('Normal Mode Shapes')
    ax1.legend()

    # Mode frequencies
    ax2 = axes[0, 1]
    n_modes = np.arange(1, 11)
    omega_n = n_modes * np.pi * c / L
    ax2.stem(n_modes, omega_n, basefmt=' ')
    ax2.set_xlabel('Mode number n')
    ax2.set_ylabel('Frequency ωₙ')
    ax2.set_title('Eigenfrequencies: ωₙ = nπc/L')

    # Simulate superposition
    wave = WaveEquation1D(L=L, c=c, nx=200, cfl=0.9)

    # Initial condition: sum of modes
    def multi_mode_ic(x):
        return 0.5*np.sin(np.pi*x/L) + 0.3*np.sin(2*np.pi*x/L) + 0.2*np.sin(3*np.pi*x/L)

    wave.set_initial_condition(multi_mode_ic)
    times, history = wave.solve(t_max=20, bc_type='fixed', save_every=5)

    # Space-time
    im = axes[1, 0].imshow(history, aspect='auto', origin='lower',
                          extent=[0, L, 0, times[-1]], cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('t')
    axes[1, 0].set_title('Multi-Mode Standing Wave')
    plt.colorbar(im, ax=axes[1, 0])

    # FFT to extract frequencies
    u_center = history[:, len(wave.x)//4]  # Sample at x = L/4
    freqs = np.fft.rfftfreq(len(times), times[1] - times[0])
    spectrum = np.abs(np.fft.rfft(u_center))

    axes[1, 1].plot(freqs * 2 * np.pi, spectrum)
    for n in range(1, 5):
        omega_n = n * np.pi * c / L
        axes[1, 1].axvline(omega_n, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Angular frequency ω')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Frequency Spectrum')
    axes[1, 1].set_xlim(0, 5)

    plt.tight_layout()
    plt.savefig('standing_waves.png', dpi=150)
    plt.show()

standing_wave_modes()
```

### 5. 2D Wave Equation

$$\frac{\partial^2 u}{\partial t^2} = c^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

```python
class WaveEquation2D:
    """2D wave equation solver."""

    def __init__(self, Lx=10.0, Ly=10.0, c=1.0, nx=100, ny=100, cfl=0.5):
        self.Lx, self.Ly = Lx, Ly
        self.c = c
        self.nx, self.ny = nx, ny
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dt = cfl * min(self.dx, self.dy) / (c * np.sqrt(2))

        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.u = np.zeros((ny, nx))
        self.u_prev = np.zeros((ny, nx))
        self.u_next = np.zeros((ny, nx))

        self.rx2 = (c * self.dt / self.dx)**2
        self.ry2 = (c * self.dt / self.dy)**2

    def set_initial_condition(self, u0_func):
        self.u = u0_func(self.X, self.Y)
        self.u_prev = self.u.copy()

    def step(self):
        # Interior update
        self.u_next[1:-1, 1:-1] = (
            2 * self.u[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] +
            self.rx2 * (self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) +
            self.ry2 * (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1])
        )

        # Fixed boundaries
        self.u_next[0, :] = 0
        self.u_next[-1, :] = 0
        self.u_next[:, 0] = 0
        self.u_next[:, -1] = 0

        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()

    def solve(self, t_max, save_every=10):
        n_steps = int(t_max / self.dt)
        history = [self.u.copy()]
        times = [0]

        for step in range(n_steps):
            self.step()
            if step % save_every == 0:
                history.append(self.u.copy())
                times.append((step + 1) * self.dt)

        return np.array(times), np.array(history)

# 2D wave demonstration
wave2d = WaveEquation2D(Lx=10, Ly=10, c=1.0, nx=100, ny=100, cfl=0.5)

# Initial Gaussian
def gaussian_2d(X, Y):
    return np.exp(-((X-5)**2 + (Y-5)**2) / 0.5)

wave2d.set_initial_condition(gaussian_2d)
times, history = wave2d.solve(t_max=10, save_every=20)

# Visualize snapshots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
indices = [0, len(history)//5, 2*len(history)//5, 3*len(history)//5, 4*len(history)//5, -1]

for ax, idx in zip(axes.flatten(), indices):
    im = ax.imshow(history[idx], origin='lower', extent=[0, 10, 0, 10],
                  cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title(f't = {times[idx]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.suptitle('2D Wave Equation: Circular Wave', fontsize=14)
plt.tight_layout()
plt.savefig('wave_equation_2d.png', dpi=150)
plt.show()
```

---

## Quantum Mechanics Connection

The time-dependent Schrödinger equation has the form of a wave equation:
$$i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2} + V\psi$$

Key differences:
1. First-order in time (vs. second-order for classical wave)
2. Complex-valued wave function
3. Potential energy term

The numerical techniques learned here directly apply to quantum simulation in Day 277.

---

## Summary

### Key Equations

$$\boxed{\text{Wave equation: } \frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u}$$

$$\boxed{\text{CFL condition: } \frac{c\Delta t}{\Delta x} \leq 1}$$

$$\boxed{\text{Normal modes: } u_n = \sin\left(\frac{n\pi x}{L}\right), \quad \omega_n = \frac{n\pi c}{L}}$$

---

## Daily Checklist

- [ ] Derived finite difference approximations
- [ ] Implemented leapfrog scheme
- [ ] Verified CFL stability condition
- [ ] Simulated standing waves
- [ ] Solved 2D wave equation
- [ ] Connected to quantum wave mechanics
