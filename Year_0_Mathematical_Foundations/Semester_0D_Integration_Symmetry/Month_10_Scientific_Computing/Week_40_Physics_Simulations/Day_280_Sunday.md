# Day 280: Month 10 Review & Capstone Project

## Schedule Overview
**Date**: Week 40, Day 7 (Sunday) — Month 10 Capstone
**Duration**: 7 hours
**Theme**: Integration of All Scientific Computing Skills

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Month 10 comprehensive review |
| Afternoon | 2.5 hours | Capstone: Complete quantum simulator |
| Evening | 1.5 hours | Self-assessment and Year 0 review preparation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Synthesize all Month 10 scientific computing skills
2. Build a complete quantum mechanics simulation package
3. Apply visualization, numerical methods, and physics together
4. Demonstrate mastery of Python scientific computing stack
5. Prepare for transition to Year 1 Quantum Mechanics

---

## Month 10 Summary

### Week 37: Python & NumPy Foundations
- Python fundamentals, functions, classes
- NumPy arrays, broadcasting, vectorization
- Linear algebra with `np.linalg`
- Random number generation and Monte Carlo

### Week 38: SciPy & Numerical Methods
- Numerical integration (`quad`, `dblquad`)
- ODE solvers (`solve_ivp`)
- Optimization (`minimize`, `root`)
- Special functions (Hermite, Legendre, spherical harmonics)
- Sparse matrices for large systems

### Week 39: Visualization
- Matplotlib fundamentals and advanced plotting
- 3D visualization
- Interactive plots with Plotly
- Animation with `FuncAnimation`
- Publication-quality figures

### Week 40: Physics Simulations
- Classical mechanics (integrators, phase space)
- Electromagnetism visualization
- Wave equation numerics
- Time-dependent Schrödinger equation
- Eigenvalue problems
- Monte Carlo methods

---

## Capstone Project: Complete Quantum Simulator

### Project Specification

Build a comprehensive quantum mechanics simulation package that combines all Month 10 skills.

```python
"""
QuantumSimulator: Complete Quantum Mechanics Simulation Package
Month 10 Capstone Project
Day 280

This package provides:
1. Time-independent solver (eigenvalues/eigenstates)
2. Time-dependent evolution (split-step FFT)
3. Visualization tools (static, animated, interactive)
4. Analysis tools (expectation values, uncertainty)
5. Example potentials and systems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import eigh
from scipy.special import hermite
from math import factorial


class QuantumSimulator:
    """
    Complete 1D quantum mechanics simulator.

    Features:
    - Eigenvalue solver for bound states
    - Time evolution using split-step FFT
    - Built-in visualization methods
    - Expectation value computation
    """

    def __init__(self, x_min=-20, x_max=20, nx=1024, hbar=1.0, m=1.0):
        """
        Initialize quantum simulator.

        Parameters
        ----------
        x_min, x_max : float
            Spatial domain boundaries
        nx : int
            Number of grid points (power of 2 recommended)
        hbar, m : float
            Planck constant and particle mass
        """
        self.hbar = hbar
        self.m = m
        self.nx = nx

        # Spatial grid
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = self.x[1] - self.x[0]
        self.L = x_max - x_min

        # Momentum grid for FFT
        self.k = 2 * np.pi * fftfreq(nx, self.dx)
        self.T_k = (hbar * self.k)**2 / (2 * m)  # Kinetic energy in k-space

        # State
        self.psi = np.zeros(nx, dtype=complex)
        self.V = np.zeros(nx)
        self.t = 0

        # Eigenstates (computed on demand)
        self._eigenvalues = None
        self._eigenstates = None

    # ==================== POTENTIAL DEFINITIONS ====================

    def set_potential(self, V_func):
        """Set potential energy function."""
        self.V = np.array([V_func(xi) for xi in self.x])
        self._eigenvalues = None  # Reset cached eigenstates
        self._eigenstates = None

    @staticmethod
    def harmonic_potential(omega=1.0):
        """Return harmonic oscillator potential function."""
        return lambda x: 0.5 * omega**2 * x**2

    @staticmethod
    def square_well(width=2.0, depth=10.0):
        """Return finite square well potential."""
        return lambda x: 0 if abs(x) < width/2 else depth

    @staticmethod
    def double_well(a=1.0, V0=1.0):
        """Return double well potential V = V0*(x²-a²)²."""
        return lambda x: V0 * (x**2 - a**2)**2

    @staticmethod
    def barrier(width=2.0, height=5.0):
        """Return rectangular barrier."""
        return lambda x: height if abs(x) < width/2 else 0

    # ==================== EIGENVALUE SOLVER ====================

    def solve_eigenstates(self, n_states=10):
        """
        Solve for bound state eigenvalues and eigenstates.

        Uses finite difference matrix diagonalization.
        """
        # Build Hamiltonian matrix
        coeff = -self.hbar**2 / (2 * self.m * self.dx**2)
        H = np.diag(self.V - 2*coeff) + np.diag(coeff*np.ones(self.nx-1), 1) + \
            np.diag(coeff*np.ones(self.nx-1), -1)

        # Diagonalize
        eigenvalues, eigenvectors = eigh(H)

        # Store first n_states
        self._eigenvalues = eigenvalues[:n_states]
        self._eigenstates = eigenvectors[:, :n_states].T

        # Normalize
        for i in range(n_states):
            norm = np.sqrt(np.trapz(np.abs(self._eigenstates[i])**2, self.x))
            self._eigenstates[i] /= norm

        return self._eigenvalues, self._eigenstates

    def get_eigenstate(self, n):
        """Get n-th eigenstate (solving if necessary)."""
        if self._eigenstates is None or n >= len(self._eigenvalues):
            self.solve_eigenstates(n_states=n+1)
        return self._eigenstates[n], self._eigenvalues[n]

    # ==================== WAVE FUNCTION INITIALIZATION ====================

    def set_state(self, psi_func):
        """Set wave function from function."""
        self.psi = psi_func(self.x).astype(complex)
        self._normalize()
        self.t = 0

    def set_gaussian(self, x0=0, sigma=1, k0=0):
        """Set Gaussian wave packet."""
        norm = (2 * np.pi * sigma**2)**(-0.25)
        self.psi = norm * np.exp(-(self.x - x0)**2 / (4*sigma**2)) * np.exp(1j*k0*self.x)
        self._normalize()
        self.t = 0

    def set_eigenstate(self, n):
        """Set state to n-th eigenstate."""
        psi, E = self.get_eigenstate(n)
        self.psi = psi.astype(complex)
        self.t = 0

    def set_superposition(self, coeffs):
        """Set superposition of eigenstates: Σ c_n |n⟩."""
        self.psi = np.zeros(self.nx, dtype=complex)
        for n, c in enumerate(coeffs):
            psi_n, _ = self.get_eigenstate(n)
            self.psi += c * psi_n
        self._normalize()
        self.t = 0

    def _normalize(self):
        """Normalize wave function."""
        norm = np.sqrt(np.trapz(np.abs(self.psi)**2, self.x))
        if norm > 0:
            self.psi /= norm

    # ==================== TIME EVOLUTION ====================

    def evolve_step(self, dt):
        """Single time step using split-step FFT."""
        # Half step in position space (potential)
        self.psi *= np.exp(-0.5j * self.V * dt / self.hbar)

        # Full step in momentum space (kinetic)
        psi_k = fft(self.psi)
        psi_k *= np.exp(-1j * self.T_k * dt / self.hbar)
        self.psi = ifft(psi_k)

        # Half step in position space (potential)
        self.psi *= np.exp(-0.5j * self.V * dt / self.hbar)

        self.t += dt

    def evolve(self, t_total, dt=0.01, save_every=10):
        """
        Evolve for total time t_total.

        Returns times and probability density history.
        """
        n_steps = int(t_total / dt)
        history = [np.abs(self.psi)**2]
        psi_history = [self.psi.copy()]
        times = [self.t]

        for step in range(n_steps):
            self.evolve_step(dt)
            if (step + 1) % save_every == 0:
                history.append(np.abs(self.psi)**2)
                psi_history.append(self.psi.copy())
                times.append(self.t)

        return np.array(times), np.array(history), np.array(psi_history)

    # ==================== OBSERVABLES ====================

    def expectation_value(self, observable='x'):
        """Compute expectation value of observable."""
        prob = np.abs(self.psi)**2

        if observable == 'x':
            return np.trapz(self.x * prob, self.x)
        elif observable == 'x2':
            return np.trapz(self.x**2 * prob, self.x)
        elif observable == 'p':
            psi_k = fft(self.psi) * self.dx / np.sqrt(2*np.pi)
            return np.sum(self.hbar * self.k * np.abs(psi_k)**2).real * 2*np.pi/self.L
        elif observable == 'p2':
            psi_k = fft(self.psi) * self.dx / np.sqrt(2*np.pi)
            return np.sum((self.hbar * self.k)**2 * np.abs(psi_k)**2).real * 2*np.pi/self.L
        elif observable == 'E':
            # Energy = kinetic + potential
            psi_k = fft(self.psi) * self.dx / np.sqrt(2*np.pi)
            T = np.sum(self.T_k * np.abs(psi_k)**2).real * 2*np.pi/self.L
            V_exp = np.trapz(self.V * prob, self.x)
            return T + V_exp
        elif observable == 'V':
            return np.trapz(self.V * prob, self.x)

    def uncertainty(self, var='x'):
        """Compute uncertainty Δx or Δp."""
        if var == 'x':
            x_mean = self.expectation_value('x')
            x2_mean = self.expectation_value('x2')
            return np.sqrt(max(0, x2_mean - x_mean**2))
        elif var == 'p':
            p_mean = self.expectation_value('p')
            p2_mean = self.expectation_value('p2')
            return np.sqrt(max(0, p2_mean - p_mean**2))

    def uncertainty_product(self):
        """Compute Δx·Δp (should be ≥ ℏ/2)."""
        return self.uncertainty('x') * self.uncertainty('p')

    # ==================== VISUALIZATION ====================

    def plot_state(self, show_potential=True, ax=None):
        """Plot current wave function."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        prob = np.abs(self.psi)**2

        if show_potential:
            V_scaled = self.V / max(self.V.max(), 1) * prob.max() * 0.5
            ax.fill_between(self.x, V_scaled, alpha=0.2, color='gray', label='V(x)')

        ax.plot(self.x, prob, 'b-', linewidth=2, label=r'$|\psi|^2$')
        ax.fill_between(self.x, prob, alpha=0.3, color='blue')

        ax.set_xlabel('Position x')
        ax.set_ylabel('Probability density')
        ax.set_title(f't = {self.t:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_eigenstates(self, n_states=5, ax=None):
        """Plot first n eigenstates in potential."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.figure

        E, psi = self.solve_eigenstates(n_states)

        # Plot potential
        ax.plot(self.x, self.V, 'k-', linewidth=2, label='V(x)')

        # Plot eigenstates
        for n in range(n_states):
            if E[n] < self.V.max():
                scale = 0.3 * (E[1] - E[0]) if n_states > 1 else 1
                ax.fill_between(self.x, E[n], E[n] + scale * psi[n]**2 / psi[n].max()**2,
                               alpha=0.5)
                ax.axhline(E[n], color=f'C{n}', linestyle='--', alpha=0.7)
                ax.text(self.x[-1]*0.9, E[n], f'E_{n}={E[n]:.2f}', va='center')

        ax.set_xlabel('Position x')
        ax.set_ylabel('Energy')
        ax.set_title('Eigenstates in Potential')
        ax.set_xlim(self.x[0], self.x[-1])

        return fig, ax

    def animate(self, t_total, dt=0.01, interval=30):
        """Create animation of time evolution."""
        n_steps = int(t_total / dt)

        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], 'b-', linewidth=2)
        fill = ax.fill_between([], [], alpha=0.3, color='blue')

        # Plot potential
        V_max = self.V.max() if self.V.max() > 0 else 1
        V_scaled = self.V / V_max * 0.3
        ax.fill_between(self.x, V_scaled, alpha=0.2, color='gray')

        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel('Position x')
        ax.set_ylabel(r'$|\psi|^2$')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        energy_text = ax.text(0.02, 0.88, '', transform=ax.transAxes)

        # Store initial state
        psi_init = self.psi.copy()
        t_init = self.t

        def init():
            line.set_data([], [])
            return line, time_text, energy_text

        def update(frame):
            for _ in range(5):  # Multiple steps per frame
                self.evolve_step(dt)

            prob = np.abs(self.psi)**2
            line.set_data(self.x, prob)

            time_text.set_text(f't = {self.t:.2f}')
            E = self.expectation_value('E')
            energy_text.set_text(f'⟨E⟩ = {E:.3f}')

            return line, time_text, energy_text

        ani = FuncAnimation(fig, update, frames=n_steps//5,
                           init_func=init, blit=True, interval=interval)

        # Restore initial state after animation creation
        self.psi = psi_init
        self.t = t_init

        return fig, ani

    def plot_phase_space(self, ax=None):
        """Plot Husimi Q-function (phase space representation)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        # Compute Husimi function on grid
        x_grid = np.linspace(self.x.min()/2, self.x.max()/2, 50)
        p_grid = np.linspace(-5, 5, 50)
        Q = np.zeros((len(p_grid), len(x_grid)))

        sigma = 1.0
        for i, x0 in enumerate(x_grid):
            for j, p0 in enumerate(p_grid):
                # Coherent state
                coherent = (2*np.pi*sigma**2)**(-0.25) * \
                          np.exp(-(self.x-x0)**2/(4*sigma**2)) * np.exp(1j*p0*self.x)
                Q[j, i] = np.abs(np.trapz(np.conj(coherent) * self.psi, self.x))**2

        im = ax.imshow(Q, extent=[x_grid[0], x_grid[-1], p_grid[0], p_grid[-1]],
                      origin='lower', aspect='auto', cmap='hot')
        ax.set_xlabel('x')
        ax.set_ylabel('p')
        ax.set_title('Husimi Q-Function (Phase Space)')
        plt.colorbar(im, ax=ax, label='Q(x, p)')

        return fig, ax


# ============================================================
# DEMONSTRATION
# ============================================================

def run_capstone_demo():
    """Complete demonstration of QuantumSimulator capabilities."""

    print("=" * 70)
    print("QUANTUM SIMULATOR: Month 10 Capstone Project")
    print("=" * 70)

    # Initialize simulator
    sim = QuantumSimulator(x_min=-15, x_max=15, nx=1024)

    # Demo 1: Harmonic oscillator eigenstates
    print("\n1. HARMONIC OSCILLATOR EIGENSTATES")
    print("-" * 40)
    sim.set_potential(sim.harmonic_potential(omega=1.0))
    E, psi = sim.solve_eigenstates(n_states=6)

    print("Energy eigenvalues (should be n + 0.5):")
    for n, En in enumerate(E):
        print(f"  E_{n} = {En:.6f} (exact: {n + 0.5:.6f})")

    fig1, ax1 = sim.plot_eigenstates(n_states=5)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-0.5, 6)
    plt.savefig('capstone_eigenstates.png', dpi=150, bbox_inches='tight')
    print("  Saved: capstone_eigenstates.png")

    # Demo 2: Wave packet dynamics
    print("\n2. WAVE PACKET DYNAMICS")
    print("-" * 40)
    sim.set_gaussian(x0=-5, sigma=1, k0=2)

    print(f"Initial state:")
    print(f"  ⟨x⟩ = {sim.expectation_value('x'):.3f}")
    print(f"  Δx = {sim.uncertainty('x'):.3f}")
    print(f"  Δx·Δp = {sim.uncertainty_product():.4f} (≥ 0.5)")

    times, history, _ = sim.evolve(t_total=15, dt=0.01, save_every=50)

    # Space-time plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    im = axes2[0].imshow(history, aspect='auto', origin='lower',
                        extent=[sim.x[0], sim.x[-1], 0, times[-1]],
                        cmap='viridis')
    axes2[0].set_xlabel('Position x')
    axes2[0].set_ylabel('Time t')
    axes2[0].set_title('Wave Packet in Harmonic Potential')
    plt.colorbar(im, ax=axes2[0], label=r'$|\psi|^2$')

    # Expectation values
    sim.set_gaussian(x0=-5, sigma=1, k0=2)
    x_expect = []
    for t in times:
        x_expect.append(sim.expectation_value('x'))
        if t < times[-1]:
            sim.evolve_step((times[1]-times[0]))

    axes2[1].plot(times, x_expect, 'b-', linewidth=2, label='⟨x⟩')
    axes2[1].plot(times, -5*np.cos(times), 'r--', label='Classical')
    axes2[1].set_xlabel('Time t')
    axes2[1].set_ylabel('⟨x⟩')
    axes2[1].set_title('Ehrenfest Theorem')
    axes2[1].legend()

    plt.tight_layout()
    plt.savefig('capstone_dynamics.png', dpi=150, bbox_inches='tight')
    print("  Saved: capstone_dynamics.png")

    # Demo 3: Quantum tunneling
    print("\n3. QUANTUM TUNNELING")
    print("-" * 40)
    sim2 = QuantumSimulator(x_min=-20, x_max=20, nx=1024)
    sim2.set_potential(sim2.barrier(width=2, height=5))
    sim2.set_gaussian(x0=-8, sigma=1.5, k0=2.5)

    print("Barrier parameters: width=2, height=5")
    print(f"Wave packet: k0=2.5, E_kinetic ≈ {0.5 * 2.5**2:.2f}")

    times, history, _ = sim2.evolve(t_total=12, dt=0.005, save_every=100)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    im = ax3.imshow(history, aspect='auto', origin='lower',
                   extent=[sim2.x[0], sim2.x[-1], 0, times[-1]],
                   cmap='viridis', vmax=0.2)
    ax3.axvline(-1, color='white', linestyle='--', alpha=0.7)
    ax3.axvline(1, color='white', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Time t')
    ax3.set_title('Quantum Tunneling Through Barrier')
    plt.colorbar(im, ax=ax3, label=r'$|\psi|^2$')
    plt.savefig('capstone_tunneling.png', dpi=150, bbox_inches='tight')
    print("  Saved: capstone_tunneling.png")

    # Demo 4: Double well
    print("\n4. DOUBLE WELL DYNAMICS")
    print("-" * 40)
    sim3 = QuantumSimulator(x_min=-8, x_max=8, nx=512)
    sim3.set_potential(sim3.double_well(a=1.5, V0=0.5))
    E, psi = sim3.solve_eigenstates(n_states=4)
    print(f"Lowest eigenvalues: {E}")
    print(f"Tunneling splitting: ΔE = {E[1] - E[0]:.6f}")

    fig4, ax4 = sim3.plot_eigenstates(n_states=4)
    ax4.set_xlim(-5, 5)
    plt.savefig('capstone_double_well.png', dpi=150, bbox_inches='tight')
    print("  Saved: capstone_double_well.png")

    print("\n" + "=" * 70)
    print("CAPSTONE DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nFiles generated:")
    print("  - capstone_eigenstates.png")
    print("  - capstone_dynamics.png")
    print("  - capstone_tunneling.png")
    print("  - capstone_double_well.png")

    plt.show()


if __name__ == "__main__":
    run_capstone_demo()
```

---

## Month 10 Assessment

### Skills Checklist

#### Week 37: Python & NumPy
- [ ] Write clean, modular Python code with classes
- [ ] Use NumPy arrays efficiently with broadcasting
- [ ] Apply vectorization for performance
- [ ] Perform linear algebra operations

#### Week 38: SciPy
- [ ] Compute integrals numerically
- [ ] Solve ODEs with `solve_ivp`
- [ ] Find roots and optimize functions
- [ ] Use special functions (Hermite, Legendre)
- [ ] Work with sparse matrices

#### Week 39: Visualization
- [ ] Create publication-quality matplotlib figures
- [ ] Generate 3D visualizations
- [ ] Build interactive Plotly dashboards
- [ ] Animate physical simulations

#### Week 40: Physics Simulations
- [ ] Simulate classical mechanical systems
- [ ] Visualize electromagnetic fields
- [ ] Solve wave equation numerically
- [ ] Implement split-step TDSE solver
- [ ] Find eigenvalues/eigenstates
- [ ] Apply Monte Carlo methods

---

## Transition to Year 1

Month 10 completes Semester 0D and Year 0 (Mathematical Foundations). The skills developed here form the computational backbone for Year 1: Quantum Mechanics Core.

### Key Tools for Year 1
| Year 0 Skill | Year 1 Application |
|--------------|-------------------|
| NumPy arrays | Quantum state vectors |
| `scipy.linalg.eigh` | Hamiltonian diagonalization |
| Split-step FFT | Wave packet dynamics |
| Visualization | Probability densities, orbitals |
| Monte Carlo | Variational methods |

### Upcoming Topics (Year 1 Preview)
- Month 13: Postulates of Quantum Mechanics
- Month 14: One-Dimensional Systems
- Month 15: Three-Dimensional Systems
- Month 16: Angular Momentum
- Month 17: Spin and Identical Particles
- Month 18: Approximation Methods

---

## Summary

### Month 10 Key Achievements

1. **Python Mastery**: Object-oriented programming for scientific computing
2. **NumPy Expertise**: Vectorized numerical operations
3. **SciPy Proficiency**: Full scientific computing toolkit
4. **Visualization Skills**: From basic plots to animations
5. **Physics Simulations**: Classical and quantum systems
6. **Numerical Methods**: Integrators, solvers, Monte Carlo

### Key Equations (Month Summary)

$$\boxed{i\hbar\frac{\partial\psi}{\partial t} = \hat{H}\psi}$$

$$\boxed{\hat{H}\psi_n = E_n\psi_n}$$

$$\boxed{\Delta x \cdot \Delta p \geq \frac{\hbar}{2}}$$

---

## Daily Checklist

- [ ] Reviewed all Week 37-40 concepts
- [ ] Completed capstone quantum simulator
- [ ] Tested all simulator features
- [ ] Generated visualization outputs
- [ ] Self-assessed Month 10 skills
- [ ] Prepared for Year 1 transition

---

## Navigation

- **Previous**: [Day 279: Monte Carlo Methods](Day_279_Saturday.md)
- **Next**: [Month 11: Group Theory & Symmetries](../../Month_11_Group_Theory/README.md)
- **Year Overview**: [Year 0: Mathematical Foundations](../../../README.md)

---

*Congratulations on completing Month 10: Scientific Computing!*
*You now have the computational tools to tackle quantum mechanics numerically.*
