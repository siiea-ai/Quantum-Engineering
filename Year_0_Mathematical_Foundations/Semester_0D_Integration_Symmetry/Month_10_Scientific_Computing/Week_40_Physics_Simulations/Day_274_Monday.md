# Day 274: Classical Mechanics Simulations

## Schedule Overview
**Date**: Week 40, Day 1 (Monday)
**Duration**: 7 hours
**Theme**: Numerical Simulation of Classical Mechanical Systems

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Numerical integrators, symplectic methods |
| Afternoon | 2.5 hours | Pendulum dynamics, phase space visualization |
| Evening | 1.5 hours | Computational lab: Planetary motion simulator |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Implement numerical integrators (Euler, RK4, Verlet)
2. Understand symplectic integrators for Hamiltonian systems
3. Simulate pendulum dynamics including chaotic regimes
4. Visualize phase space trajectories
5. Model planetary motion with gravitational interactions
6. Connect classical simulations to quantum analogs

---

## Core Content

### 1. Numerical Integration Methods

Classical mechanics is governed by Newton's second law:
$$m\frac{d^2x}{dt^2} = F(x, \dot{x}, t)$$

Converting to first-order system:
$$\frac{dx}{dt} = v, \quad \frac{dv}{dt} = \frac{F}{m}$$

#### Euler Method (First-Order)

```python
import numpy as np
import matplotlib.pyplot as plt

def euler_step(x, v, dt, force_func, m=1):
    """
    Simple Euler integration step.

    NOT recommended for physics - shown for comparison only.
    """
    a = force_func(x, v) / m
    x_new = x + v * dt
    v_new = v + a * dt
    return x_new, v_new

# Example: Harmonic oscillator
def harmonic_force(x, v, k=1):
    return -k * x

def simulate_euler(x0, v0, dt, n_steps, force_func):
    """Simulate using Euler method."""
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0], v[0] = x0, v0

    for i in range(n_steps - 1):
        x[i+1], v[i+1] = euler_step(x[i], v[i], dt, force_func)

    return x, v
```

#### Velocity Verlet (Symplectic)

The Verlet method preserves energy in Hamiltonian systems:

```python
def verlet_step(x, v, dt, force_func, m=1):
    """
    Velocity Verlet integration - symplectic method.

    Preserves phase space volume and energy (on average).
    """
    a = force_func(x, v) / m
    x_new = x + v * dt + 0.5 * a * dt**2
    a_new = force_func(x_new, v) / m  # Evaluate at new position
    v_new = v + 0.5 * (a + a_new) * dt
    return x_new, v_new

def simulate_verlet(x0, v0, dt, n_steps, force_func, m=1):
    """Simulate using Velocity Verlet."""
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0], v[0] = x0, v0

    for i in range(n_steps - 1):
        x[i+1], v[i+1] = verlet_step(x[i], v[i], dt, force_func, m)

    return x, v
```

#### Runge-Kutta 4th Order (RK4)

```python
def rk4_step(y, t, dt, deriv_func):
    """
    Fourth-order Runge-Kutta step.

    Parameters
    ----------
    y : array
        State vector [x, v]
    deriv_func : callable
        Returns [dx/dt, dv/dt] = [v, a]
    """
    k1 = deriv_func(y, t)
    k2 = deriv_func(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = deriv_func(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = deriv_func(y + dt*k3, t + dt)

    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def simulate_rk4(y0, t_span, dt, deriv_func):
    """Simulate using RK4."""
    t = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    for i in range(n_steps - 1):
        y[i+1] = rk4_step(y[i], t[i], dt, deriv_func)

    return t, y
```

### 2. Energy Conservation Analysis

For conservative systems, energy should be preserved:
$$E = T + V = \frac{1}{2}mv^2 + V(x)$$

```python
def analyze_energy_conservation(x, v, potential_func, m=1):
    """
    Analyze energy conservation during simulation.

    Parameters
    ----------
    x, v : arrays
        Position and velocity time series
    potential_func : callable
        V(x) potential energy function
    """
    kinetic = 0.5 * m * v**2
    potential = potential_func(x)
    total = kinetic + potential

    energy_drift = (total - total[0]) / total[0] * 100  # Percent drift

    return kinetic, potential, total, energy_drift

# Compare methods
def compare_integrators():
    """Compare energy conservation of different integrators."""

    # Harmonic oscillator: V(x) = 0.5*x^2
    def potential(x):
        return 0.5 * x**2

    def force(x, v):
        return -x

    # Initial conditions
    x0, v0 = 1.0, 0.0
    dt = 0.1
    n_steps = 1000
    t = np.arange(n_steps) * dt

    # Euler
    x_euler, v_euler = simulate_euler(x0, v0, dt, n_steps, force)
    _, _, E_euler, drift_euler = analyze_energy_conservation(x_euler, v_euler, potential)

    # Verlet
    x_verlet, v_verlet = simulate_verlet(x0, v0, dt, n_steps, force)
    _, _, E_verlet, drift_verlet = analyze_energy_conservation(x_verlet, v_verlet, potential)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectories
    axes[0, 0].plot(t, x_euler, 'r-', label='Euler', alpha=0.7)
    axes[0, 0].plot(t, x_verlet, 'b-', label='Verlet', alpha=0.7)
    axes[0, 0].plot(t, np.cos(t), 'k--', label='Exact', alpha=0.5)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].legend()
    axes[0, 0].set_title('Position vs Time')

    # Phase space
    axes[0, 1].plot(x_euler, v_euler, 'r-', label='Euler', alpha=0.5)
    axes[0, 1].plot(x_verlet, v_verlet, 'b-', label='Verlet', alpha=0.7)
    theta = np.linspace(0, 2*np.pi, 100)
    axes[0, 1].plot(np.cos(theta), np.sin(theta), 'k--', label='Exact')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('v')
    axes[0, 1].legend()
    axes[0, 1].set_title('Phase Space')
    axes[0, 1].set_aspect('equal')

    # Energy
    axes[1, 0].plot(t, E_euler, 'r-', label='Euler')
    axes[1, 0].plot(t, E_verlet, 'b-', label='Verlet')
    axes[1, 0].axhline(0.5, color='k', linestyle='--', label='Exact')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Total Energy')
    axes[1, 0].legend()
    axes[1, 0].set_title('Energy Conservation')

    # Energy drift
    axes[1, 1].plot(t, drift_euler, 'r-', label=f'Euler (max: {np.abs(drift_euler).max():.1f}%)')
    axes[1, 1].plot(t, drift_verlet, 'b-', label=f'Verlet (max: {np.abs(drift_verlet).max():.1f}%)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Energy Drift (%)')
    axes[1, 1].legend()
    axes[1, 1].set_title('Energy Conservation Error')

    plt.tight_layout()
    plt.savefig('integrator_comparison.png', dpi=150)
    plt.show()

compare_integrators()
```

### 3. Pendulum Dynamics

The simple pendulum equation:
$$\frac{d^2\theta}{dt^2} = -\frac{g}{L}\sin\theta$$

```python
class Pendulum:
    """
    Simple pendulum simulation.

    Demonstrates transition from harmonic oscillation to chaos.
    """

    def __init__(self, L=1.0, g=9.81, damping=0.0):
        self.L = L
        self.g = g
        self.gamma = damping  # Damping coefficient

    def derivatives(self, state, t):
        """Compute [dθ/dt, dω/dt]."""
        theta, omega = state
        dtheta = omega
        domega = -(self.g/self.L) * np.sin(theta) - self.gamma * omega
        return np.array([dtheta, domega])

    def simulate(self, theta0, omega0, t_max, dt):
        """Run simulation using RK4."""
        t = np.arange(0, t_max, dt)
        n = len(t)
        states = np.zeros((n, 2))
        states[0] = [theta0, omega0]

        for i in range(n-1):
            states[i+1] = rk4_step(states[i], t[i], dt, self.derivatives)

        return t, states[:, 0], states[:, 1]

    def energy(self, theta, omega):
        """Calculate total mechanical energy."""
        T = 0.5 * self.L**2 * omega**2  # Kinetic (normalized mass)
        V = self.g * self.L * (1 - np.cos(theta))  # Potential
        return T + V

    def phase_portrait(self, theta_range=(-np.pi, np.pi), omega_max=5, n_trajectories=20):
        """Generate phase portrait."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Sample initial conditions
        theta_init = np.linspace(theta_range[0], theta_range[1], n_trajectories)
        omega_init = np.linspace(-omega_max, omega_max, n_trajectories)

        for theta0 in theta_init[::4]:
            for omega0 in omega_init[::4]:
                t, theta, omega = self.simulate(theta0, omega0, 10, 0.01)
                ax.plot(theta, omega, 'b-', alpha=0.3, linewidth=0.5)

        # Energy contours
        THETA, OMEGA = np.meshgrid(np.linspace(-1.5*np.pi, 1.5*np.pi, 100),
                                   np.linspace(-omega_max, omega_max, 100))
        E = self.energy(THETA, OMEGA)
        E_sep = self.g * self.L * 2  # Separatrix energy

        cs = ax.contour(THETA, OMEGA, E, levels=15, colors='gray', alpha=0.5)
        ax.contour(THETA, OMEGA, E, levels=[E_sep], colors='red', linewidths=2)

        ax.set_xlabel(r'$\theta$ (rad)')
        ax.set_ylabel(r'$\omega$ (rad/s)')
        ax.set_title('Pendulum Phase Portrait (red = separatrix)')
        ax.set_xlim(theta_range[0]*1.5, theta_range[1]*1.5)
        ax.set_ylim(-omega_max, omega_max)

        return fig, ax

# Demonstrate
pendulum = Pendulum(L=1.0, g=10.0)
fig, ax = pendulum.phase_portrait()
plt.savefig('pendulum_phase_portrait.png', dpi=150)
plt.show()
```

### 4. Planetary Motion

Kepler problem with gravitational force:
$$\mathbf{F} = -\frac{GMm}{|\mathbf{r}|^3}\mathbf{r}$$

```python
class PlanetarySystem:
    """
    N-body gravitational simulation.
    """

    def __init__(self, G=1.0):
        self.G = G
        self.bodies = []

    def add_body(self, mass, position, velocity, name=''):
        """Add a body to the system."""
        self.bodies.append({
            'mass': mass,
            'pos': np.array(position, dtype=float),
            'vel': np.array(velocity, dtype=float),
            'name': name
        })

    def compute_accelerations(self, positions, masses):
        """Compute gravitational accelerations for all bodies."""
        n = len(masses)
        acc = np.zeros((n, 3))

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r = np.linalg.norm(r_vec)
                    acc[i] += self.G * masses[j] * r_vec / r**3

        return acc

    def simulate(self, t_max, dt):
        """Run simulation using velocity Verlet."""
        n_bodies = len(self.bodies)
        n_steps = int(t_max / dt)

        # Initialize arrays
        positions = np.zeros((n_steps, n_bodies, 3))
        velocities = np.zeros((n_steps, n_bodies, 3))
        masses = np.array([b['mass'] for b in self.bodies])

        # Initial conditions
        for i, body in enumerate(self.bodies):
            positions[0, i] = body['pos']
            velocities[0, i] = body['vel']

        # Time evolution
        for step in range(n_steps - 1):
            pos = positions[step]
            vel = velocities[step]

            # Velocity Verlet
            acc = self.compute_accelerations(pos, masses)
            pos_new = pos + vel * dt + 0.5 * acc * dt**2
            acc_new = self.compute_accelerations(pos_new, masses)
            vel_new = vel + 0.5 * (acc + acc_new) * dt

            positions[step + 1] = pos_new
            velocities[step + 1] = vel_new

        return np.arange(n_steps) * dt, positions, velocities

    def plot_orbits(self, positions, projection='xy'):
        """Plot orbital trajectories."""
        fig, ax = plt.subplots(figsize=(10, 10))

        idx = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[projection]

        for i, body in enumerate(self.bodies):
            ax.plot(positions[:, i, idx[0]], positions[:, i, idx[1]],
                   label=body['name'], linewidth=0.5)
            ax.scatter(positions[0, i, idx[0]], positions[0, i, idx[1]],
                      s=50, zorder=5)

        ax.set_xlabel(f'{projection[0]} (AU)')
        ax.set_ylabel(f'{projection[1]} (AU)')
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Orbital Trajectories')

        return fig, ax

# Two-body problem (star + planet)
def simulate_kepler():
    """Simulate Kepler problem."""
    system = PlanetarySystem(G=4*np.pi**2)  # G in AU^3 / (M_sun * yr^2)

    # Sun at origin
    system.add_body(1.0, [0, 0, 0], [0, 0, 0], 'Sun')

    # Earth-like planet
    a = 1.0  # Semi-major axis (AU)
    e = 0.5  # Eccentricity
    v_peri = np.sqrt(system.G * 1.0 * (1 + e) / (a * (1 - e)))
    system.add_body(1e-6, [a*(1-e), 0, 0], [0, v_peri, 0], 'Planet')

    # Simulate
    t, pos, vel = system.simulate(t_max=5, dt=0.001)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Orbit
    axes[0].plot(pos[:, 1, 0], pos[:, 1, 1], 'b-', linewidth=0.5)
    axes[0].scatter([0], [0], c='yellow', s=200, label='Sun', edgecolor='orange')
    axes[0].scatter(pos[0, 1, 0], pos[0, 1, 1], c='blue', s=50, label='Start')
    axes[0].set_xlabel('x (AU)')
    axes[0].set_ylabel('y (AU)')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].set_title(f'Elliptical Orbit (e = {e})')

    # Energy conservation
    r = np.linalg.norm(pos[:, 1] - pos[:, 0], axis=1)
    v_mag = np.linalg.norm(vel[:, 1], axis=1)
    E = 0.5 * v_mag**2 - system.G * 1.0 / r

    axes[1].plot(t, E)
    axes[1].set_xlabel('Time (years)')
    axes[1].set_ylabel('Specific Orbital Energy')
    axes[1].set_title('Energy Conservation')

    plt.tight_layout()
    plt.savefig('kepler_orbit.png', dpi=150)
    plt.show()

simulate_kepler()
```

---

## Quantum Mechanics Connection

### Classical-Quantum Correspondence

Classical mechanics emerges from quantum mechanics in the limit $$\hbar \to 0$$:

| Classical | Quantum |
|-----------|---------|
| Trajectory $$x(t)$$ | Wave packet $$|\psi(x,t)|^2$$ |
| Phase space $$(x, p)$$ | Wigner function $$W(x, p)$$ |
| Hamilton's equations | Ehrenfest theorem |
| Chaos (sensitivity to IC) | Quantum scars, level statistics |

```python
def classical_quantum_comparison():
    """
    Compare classical and quantum harmonic oscillator.
    """
    from scipy.special import hermite
    from math import factorial

    omega = 1.0
    x = np.linspace(-6, 6, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Classical: oscillating between turning points
    E_classical = 3  # Classical energy (units of ℏω)
    A = np.sqrt(2 * E_classical)  # Amplitude

    # Classical probability density (time-averaged)
    P_classical = np.zeros_like(x)
    mask = np.abs(x) < A
    P_classical[mask] = 1 / (np.pi * np.sqrt(A**2 - x[mask]**2))
    P_classical /= np.trapz(P_classical, x)  # Normalize

    # Quantum: n = 2 eigenstate (E = 5/2 ℏω ≈ 3)
    n = 2
    xi = x
    psi_n = (1/np.pi)**0.25 / np.sqrt(2**n * factorial(n)) * hermite(n)(xi) * np.exp(-xi**2/2)
    P_quantum = np.abs(psi_n)**2

    # Plot comparison
    axes[0, 0].plot(x, P_classical, 'r-', linewidth=2, label='Classical')
    axes[0, 0].plot(x, P_quantum, 'b-', linewidth=2, label=f'Quantum (n={n})')
    axes[0, 0].axvline(-A, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(A, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Position x')
    axes[0, 0].set_ylabel('Probability density')
    axes[0, 0].legend()
    axes[0, 0].set_title('Classical vs Quantum Probability')

    # High-n quantum approaches classical
    for n_high in [10, 20, 40]:
        psi = (1/np.pi)**0.25 / np.sqrt(2**n_high * factorial(n_high)) * \
              hermite(n_high)(xi) * np.exp(-xi**2/2)
        axes[0, 1].plot(x, np.abs(psi)**2, alpha=0.7, label=f'n={n_high}')

    A_high = np.sqrt(2 * (40 + 0.5))
    P_class_high = np.zeros_like(x)
    mask = np.abs(x) < A_high
    P_class_high[mask] = 1 / (np.pi * np.sqrt(A_high**2 - x[mask]**2))
    P_class_high /= np.trapz(P_class_high, x)
    axes[0, 1].plot(x, P_class_high, 'k--', label='Classical')
    axes[0, 1].set_xlim(-10, 10)
    axes[0, 1].set_xlabel('Position x')
    axes[0, 1].set_ylabel('Probability density')
    axes[0, 1].legend()
    axes[0, 1].set_title('Correspondence Principle: High n')

    # Pendulum: small angle vs large angle
    theta = np.linspace(-0.3, 0.3, 100)
    V_exact = 1 - np.cos(theta)  # Exact potential
    V_harmonic = 0.5 * theta**2  # Harmonic approximation

    axes[1, 0].plot(theta, V_exact, 'b-', linewidth=2, label='Exact: 1 - cos(θ)')
    axes[1, 0].plot(theta, V_harmonic, 'r--', linewidth=2, label='Harmonic: θ²/2')
    axes[1, 0].set_xlabel('θ (rad)')
    axes[1, 0].set_ylabel('V(θ) / mgL')
    axes[1, 0].legend()
    axes[1, 0].set_title('Pendulum: Small Angle Approximation')

    # Large amplitude: potential for full θ range
    theta_large = np.linspace(-np.pi, np.pi, 200)
    V_large = 1 - np.cos(theta_large)
    V_harm_large = 0.5 * theta_large**2

    axes[1, 1].plot(theta_large, V_large, 'b-', linewidth=2, label='Exact')
    axes[1, 1].plot(theta_large, V_harm_large, 'r--', linewidth=2, label='Harmonic')
    axes[1, 1].set_xlabel('θ (rad)')
    axes[1, 1].set_ylabel('V(θ) / mgL')
    axes[1, 1].legend()
    axes[1, 1].set_title('Full Potential Comparison')
    axes[1, 1].set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig('classical_quantum_comparison.png', dpi=150)
    plt.show()

classical_quantum_comparison()
```

---

## Practice Problems

### Level 1: Direct Application

1. **Euler vs RK4**: Simulate a mass on a spring using both Euler and RK4. Compare energy conservation over 100 periods.

2. **Damped Oscillator**: Add damping to the harmonic oscillator. Verify that energy decreases exponentially.

3. **Phase Space**: For a double-well potential $$V(x) = (x^2 - 1)^2$$, plot the phase portrait showing bounded and unbounded trajectories.

### Level 2: Intermediate

4. **Driven Pendulum**: Add a sinusoidal driving force to the damped pendulum. Explore resonance by varying the driving frequency.

5. **Orbital Precession**: Add a small $$1/r^3$$ perturbation to the gravitational potential. Observe and measure orbital precession.

6. **Three-Body Problem**: Set up a restricted three-body problem (two massive bodies + test particle). Find a Lagrange point.

### Level 3: Challenging

7. **Chaotic Pendulum**: Explore the transition to chaos in the driven, damped pendulum by computing Poincaré sections.

8. **Solar System**: Simulate the inner solar system (Sun + 4 planets) for 100 years. Verify Kepler's laws.

9. **Quantum-Classical Border**: Simulate a particle in a double-well potential classically. Compare with quantum tunneling rates.

---

## Computational Lab

```python
"""
Classical Mechanics Simulation Lab
Day 274: Physics Simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


class ClassicalMechanicsLab:
    """Complete classical mechanics simulation toolkit."""

    @staticmethod
    def double_pendulum():
        """Simulate and animate a double pendulum (chaotic system)."""

        # Parameters
        m1, m2 = 1.0, 1.0
        L1, L2 = 1.0, 1.0
        g = 9.81

        def derivatives(t, state):
            theta1, omega1, theta2, omega2 = state

            delta = theta1 - theta2
            denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
            denom2 = (L2 / L1) * denom1

            domega1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                      m2 * g * np.sin(theta2) * np.cos(delta) +
                      m2 * L2 * omega2**2 * np.sin(delta) -
                      (m1 + m2) * g * np.sin(theta1)) / denom1

            domega2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                      (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                      (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                      (m1 + m2) * g * np.sin(theta2)) / denom2

            return [omega1, domega1, omega2, domega2]

        # Solve for two nearby initial conditions (chaos demonstration)
        t_span = (0, 20)
        t_eval = np.linspace(*t_span, 2000)

        ic1 = [np.pi/2, 0, np.pi/2, 0]  # Initial condition 1
        ic2 = [np.pi/2 + 0.001, 0, np.pi/2, 0]  # Slightly perturbed

        sol1 = solve_ivp(derivatives, t_span, ic1, t_eval=t_eval, method='RK45')
        sol2 = solve_ivp(derivatives, t_span, ic2, t_eval=t_eval, method='RK45')

        # Plot trajectory comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # θ1 comparison
        axes[0, 0].plot(sol1.t, sol1.y[0], 'b-', label='IC 1', alpha=0.7)
        axes[0, 0].plot(sol2.t, sol2.y[0], 'r-', label='IC 2', alpha=0.7)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('θ₁')
        axes[0, 0].legend()
        axes[0, 0].set_title('Chaos: Sensitivity to Initial Conditions')

        # θ2 comparison
        axes[0, 1].plot(sol1.t, sol1.y[2], 'b-', alpha=0.7)
        axes[0, 1].plot(sol2.t, sol2.y[2], 'r-', alpha=0.7)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('θ₂')

        # Difference
        diff = np.sqrt((sol1.y[0] - sol2.y[0])**2 + (sol1.y[2] - sol2.y[2])**2)
        axes[1, 0].semilogy(sol1.t, diff)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('|Δθ|')
        axes[1, 0].set_title('Exponential Divergence')

        # Phase space θ1 vs ω1
        axes[1, 1].plot(sol1.y[0], sol1.y[1], 'b-', alpha=0.5, linewidth=0.5)
        axes[1, 1].set_xlabel('θ₁')
        axes[1, 1].set_ylabel('ω₁')
        axes[1, 1].set_title('Phase Space')

        plt.tight_layout()
        plt.savefig('double_pendulum_chaos.png', dpi=150)
        plt.show()

        return sol1, sol2

    @staticmethod
    def create_animation():
        """Create animated pendulum visualization."""

        # Simple pendulum animation
        L = 1.0
        g = 10.0
        theta0 = np.pi * 0.9  # Large amplitude

        def derivatives(t, state):
            theta, omega = state
            return [omega, -(g/L) * np.sin(theta)]

        sol = solve_ivp(derivatives, (0, 10), [theta0, 0],
                       t_eval=np.linspace(0, 10, 500))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 0.5)
        ax.set_aspect('equal')
        ax.set_title('Simple Pendulum')

        line, = ax.plot([], [], 'o-', linewidth=2, markersize=20)
        trail, = ax.plot([], [], '-', alpha=0.3, linewidth=1)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        trail_x, trail_y = [], []

        def init():
            line.set_data([], [])
            trail.set_data([], [])
            return line, trail, time_text

        def update(frame):
            theta = sol.y[0, frame]
            x = L * np.sin(theta)
            y = -L * np.cos(theta)

            line.set_data([0, x], [0, y])

            trail_x.append(x)
            trail_y.append(y)
            if len(trail_x) > 100:
                trail_x.pop(0)
                trail_y.pop(0)
            trail.set_data(trail_x, trail_y)

            time_text.set_text(f't = {sol.t[frame]:.2f} s')

            return line, trail, time_text

        ani = FuncAnimation(fig, update, frames=len(sol.t),
                           init_func=init, blit=True, interval=20)

        # ani.save('pendulum_animation.mp4', writer='ffmpeg', fps=50)
        plt.show()
        return ani


if __name__ == "__main__":
    print("=" * 60)
    print("Classical Mechanics Simulation Lab")
    print("=" * 60)

    lab = ClassicalMechanicsLab()

    print("\n1. Double pendulum chaos demonstration...")
    sol1, sol2 = lab.double_pendulum()

    print("\n2. Creating pendulum animation...")
    # ani = lab.create_animation()

    print("\nLab complete!")
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Euler method | First-order, not energy-conserving |
| Velocity Verlet | Symplectic, preserves phase space volume |
| RK4 | Fourth-order accurate, general purpose |
| Phase space | (x, p) representation of dynamics |
| Symplectic | Volume-preserving in phase space |
| Chaos | Exponential sensitivity to initial conditions |

### Key Formulas

$$\boxed{\text{Euler: } x_{n+1} = x_n + v_n \Delta t, \quad v_{n+1} = v_n + a_n \Delta t}$$

$$\boxed{\text{Verlet: } x_{n+1} = x_n + v_n \Delta t + \frac{1}{2}a_n \Delta t^2}$$

$$\boxed{\text{Energy: } E = \frac{1}{2}mv^2 + V(x)}$$

---

## Daily Checklist

- [ ] Implemented Euler, Verlet, and RK4 integrators
- [ ] Compared energy conservation properties
- [ ] Simulated pendulum with phase space analysis
- [ ] Created planetary orbit simulation
- [ ] Connected to quantum correspondence principle
- [ ] Completed computational lab exercises

---

## Preview of Day 275

Tomorrow: **Electromagnetism Visualizations**
- Electric field visualization from charge distributions
- Magnetic field lines from current elements
- Wave propagation simulation
- Connection to quantum electrodynamics
