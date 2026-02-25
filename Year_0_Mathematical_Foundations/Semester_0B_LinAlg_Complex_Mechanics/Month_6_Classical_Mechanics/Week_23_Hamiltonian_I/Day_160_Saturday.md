# Day 160: Computational Lab — Hamiltonian Mechanics in Python

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Lab Part 1: Phase Space & Poisson Brackets |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Lab Part 2: Symplectic Integrators |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Lab Part 3: Integrable vs Chaotic Systems |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Implement symbolic Poisson bracket calculations using SymPy
2. Build phase space visualizers for arbitrary 2D Hamiltonian systems
3. Implement symplectic integrators (Störmer-Verlet) and compare with non-symplectic methods
4. Verify conservation laws numerically
5. Visualize the transition from integrable to chaotic dynamics
6. Create Poincaré sections to analyze higher-dimensional systems

---

## Lab Part 1: Phase Space and Poisson Brackets

### Lab 1.1: Symbolic Poisson Bracket Calculator

```python
"""
Lab 1.1: Symbolic Poisson Bracket Calculator
Compute Poisson brackets using SymPy for arbitrary functions.
"""

import sympy as sp
from sympy import symbols, diff, simplify, sqrt, sin, cos, exp, pi
from sympy import Matrix, latex, pprint
from sympy.physics.mechanics import dynamicsymbols
import numpy as np
import matplotlib.pyplot as plt

# Initialize pretty printing
sp.init_printing()

class PoissonBracketCalculator:
    """
    A class for computing Poisson brackets symbolically.

    Supports arbitrary number of degrees of freedom.
    """

    def __init__(self, n_dof, q_names=None, p_names=None):
        """
        Initialize with n degrees of freedom.

        Parameters:
        -----------
        n_dof : int
            Number of degrees of freedom
        q_names : list, optional
            Names for position coordinates
        p_names : list, optional
            Names for momentum coordinates
        """
        self.n_dof = n_dof

        # Create symbolic coordinates
        if q_names is None:
            q_names = [f'q_{i+1}' for i in range(n_dof)]
        if p_names is None:
            p_names = [f'p_{i+1}' for i in range(n_dof)]

        self.q = [symbols(name, real=True) for name in q_names]
        self.p = [symbols(name, real=True) for name in p_names]

        # Combine into phase space coordinates
        self.z = self.q + self.p

    def bracket(self, f, g):
        """
        Compute the Poisson bracket {f, g}.

        {f, g} = Σᵢ (∂f/∂qᵢ ∂g/∂pᵢ - ∂f/∂pᵢ ∂g/∂qᵢ)
        """
        result = 0
        for qi, pi in zip(self.q, self.p):
            result += diff(f, qi) * diff(g, pi) - diff(f, pi) * diff(g, qi)
        return simplify(result)

    def verify_fundamental_brackets(self):
        """Verify the fundamental Poisson brackets."""
        print("=" * 60)
        print("FUNDAMENTAL POISSON BRACKETS")
        print("=" * 60)

        for i, (qi, pi) in enumerate(zip(self.q, self.p)):
            for j, (qj, pj) in enumerate(zip(self.q, self.p)):
                # {qᵢ, qⱼ} = 0
                qq = self.bracket(qi, qj)
                # {pᵢ, pⱼ} = 0
                pp = self.bracket(pi, pj)
                # {qᵢ, pⱼ} = δᵢⱼ
                qp = self.bracket(qi, pj)

                print(f"{{q_{i+1}, q_{j+1}}} = {qq}")
                print(f"{{p_{i+1}, p_{j+1}}} = {pp}")
                print(f"{{q_{i+1}, p_{j+1}}} = {qp}")
                print()

    def verify_jacobi_identity(self, f, g, h):
        """
        Verify the Jacobi identity:
        {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
        """
        term1 = self.bracket(f, self.bracket(g, h))
        term2 = self.bracket(g, self.bracket(h, f))
        term3 = self.bracket(h, self.bracket(f, g))

        result = simplify(term1 + term2 + term3)

        print(f"Jacobi identity check:")
        print(f"  {{f, {{g, h}}}} + {{g, {{h, f}}}} + {{h, {{f, g}}}} = {result}")

        return result == 0

    def time_evolution(self, f, H):
        """
        Compute df/dt = {f, H} (for time-independent f).
        """
        return self.bracket(f, H)

    def is_conserved(self, f, H):
        """
        Check if f is a constant of motion: {f, H} = 0
        """
        bracket = self.bracket(f, H)
        return simplify(bracket) == 0


def demo_poisson_brackets():
    """Demonstrate Poisson bracket calculations."""

    print("\n" + "=" * 60)
    print("LAB 1.1: SYMBOLIC POISSON BRACKET CALCULATOR")
    print("=" * 60)

    # 1D system
    print("\n--- 1D System ---")
    calc1d = PoissonBracketCalculator(1, ['x'], ['p'])
    x, p = calc1d.q[0], calc1d.p[0]
    m, omega, k = symbols('m omega k', positive=True)

    # Simple Harmonic Oscillator
    H_sho = p**2/(2*m) + m*omega**2*x**2/2

    print(f"\nSHO Hamiltonian: H = {H_sho}")
    print(f"  {{x, H}} = {calc1d.bracket(x, H_sho)} = ẋ")
    print(f"  {{p, H}} = {calc1d.bracket(p, H_sho)} = ṗ")

    # Verify {x², p²}
    print(f"\n  {{x², p²}} = {calc1d.bracket(x**2, p**2)}")

    # 3D system for angular momentum
    print("\n--- 3D System (Angular Momentum) ---")
    calc3d = PoissonBracketCalculator(3, ['x', 'y', 'z'], ['p_x', 'p_y', 'p_z'])
    x, y, z = calc3d.q
    px, py, pz = calc3d.p

    # Angular momentum components
    Lx = y*pz - z*py
    Ly = z*px - x*pz
    Lz = x*py - y*px

    print(f"\nAngular Momentum:")
    print(f"  L_x = {Lx}")
    print(f"  L_y = {Ly}")
    print(f"  L_z = {Lz}")

    print(f"\nso(3) Algebra:")
    print(f"  {{L_x, L_y}} = {calc3d.bracket(Lx, Ly)} (should be L_z)")
    print(f"  {{L_y, L_z}} = {calc3d.bracket(Ly, Lz)} (should be L_x)")
    print(f"  {{L_z, L_x}} = {calc3d.bracket(Lz, Lx)} (should be L_y)")

    # Verify {L², Lz} = 0
    L_squared = Lx**2 + Ly**2 + Lz**2
    print(f"\n  {{L², L_z}} = {calc3d.bracket(L_squared, Lz)} (should be 0)")

    # Central force conservation
    r = sqrt(x**2 + y**2 + z**2)
    H_central = (px**2 + py**2 + pz**2)/(2*m) - k/r

    print(f"\nCentral Force: H = p²/(2m) - k/r")
    print(f"  {{L_z, H}} = {simplify(calc3d.bracket(Lz, H_central))} (angular momentum conserved!)")

    # Jacobi identity
    print("\nJacobi Identity Verification:")
    calc3d.verify_jacobi_identity(Lx, Ly, Lz)


# Run the demonstration
demo_poisson_brackets()
```

---

### Lab 1.2: Phase Space Visualizer

```python
"""
Lab 1.2: Phase Space Visualizer
Create interactive phase portraits for 2D Hamiltonian systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

class PhaseSpaceVisualizer:
    """
    Visualize phase space for 2D Hamiltonian systems.
    """

    def __init__(self, H_func, dH_dq_func, dH_dp_func, name="System"):
        """
        Initialize with Hamiltonian and its derivatives.

        Parameters:
        -----------
        H_func : callable
            Hamiltonian H(q, p)
        dH_dq_func : callable
            ∂H/∂q(q, p)
        dH_dp_func : callable
            ∂H/∂p(q, p)
        """
        self.H = H_func
        self.dH_dq = dH_dq_func
        self.dH_dp = dH_dp_func
        self.name = name

    def equations_of_motion(self, state, t):
        """Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q"""
        q, p = state
        dq_dt = self.dH_dp(q, p)
        dp_dt = -self.dH_dq(q, p)
        return [dq_dt, dp_dt]

    def solve_trajectory(self, q0, p0, t_span, n_points=1000):
        """Solve for a single trajectory."""
        t = np.linspace(t_span[0], t_span[1], n_points)
        sol = odeint(self.equations_of_motion, [q0, p0], t)
        return t, sol[:, 0], sol[:, 1]

    def plot_phase_portrait(self, q_range, p_range, n_grid=20,
                           trajectories=None, figsize=(10, 8)):
        """
        Plot phase portrait with vector field and optional trajectories.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create grid for vector field
        q = np.linspace(*q_range, n_grid)
        p = np.linspace(*p_range, n_grid)
        Q, P = np.meshgrid(q, p)

        # Compute vector field
        dQ = np.zeros_like(Q)
        dP = np.zeros_like(P)
        for i in range(n_grid):
            for j in range(n_grid):
                dQ[i, j] = self.dH_dp(Q[i, j], P[i, j])
                dP[i, j] = -self.dH_dq(Q[i, j], P[i, j])

        # Normalize for better visualization
        magnitude = np.sqrt(dQ**2 + dP**2)
        magnitude[magnitude == 0] = 1
        dQ_norm = dQ / magnitude
        dP_norm = dP / magnitude

        # Plot streamlines
        ax.streamplot(Q, P, dQ, dP, density=1.5, color='lightblue',
                     linewidth=0.7, arrowsize=0.8)

        # Plot energy contours
        q_fine = np.linspace(*q_range, 200)
        p_fine = np.linspace(*p_range, 200)
        Q_fine, P_fine = np.meshgrid(q_fine, p_fine)
        H_vals = np.zeros_like(Q_fine)
        for i in range(len(q_fine)):
            for j in range(len(p_fine)):
                H_vals[j, i] = self.H(q_fine[i], p_fine[j])

        contours = ax.contour(Q_fine, P_fine, H_vals, levels=15,
                             colors='darkblue', linewidths=0.8, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

        # Plot specific trajectories
        if trajectories is not None:
            colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(trajectories)))
            for (q0, p0, t_max), color in zip(trajectories, colors):
                t, q_traj, p_traj = self.solve_trajectory(q0, p0, [0, t_max])
                ax.plot(q_traj, p_traj, '-', color=color, lw=2)
                ax.plot(q0, p0, 'o', color=color, markersize=8)

        ax.set_xlabel('q (position)', fontsize=12)
        ax.set_ylabel('p (momentum)', fontsize=12)
        ax.set_title(f'Phase Portrait: {self.name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(q_range)
        ax.set_ylim(p_range)

        return fig, ax

    def verify_energy_conservation(self, q0, p0, t_max, n_points=1000):
        """Verify energy conservation along a trajectory."""
        t, q, p = self.solve_trajectory(q0, p0, [0, t_max], n_points)

        E = np.array([self.H(q[i], p[i]) for i in range(len(t))])
        E_error = (E - E[0]) / abs(E[0])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Phase space trajectory
        axes[0].plot(q, p, 'b-', lw=1.5)
        axes[0].plot(q0, p0, 'go', markersize=10, label='Start')
        axes[0].set_xlabel('q')
        axes[0].set_ylabel('p')
        axes[0].set_title(f'{self.name}: Phase Space Trajectory')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Energy conservation
        axes[1].plot(t, E_error, 'r-', lw=1.5)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Relative Energy Error (E-E₀)/E₀')
        axes[1].set_title(f'Energy Conservation: max error = {np.abs(E_error).max():.2e}')
        axes[1].grid(True, alpha=0.3)

        return fig


def demo_phase_space():
    """Demonstrate phase space visualization for various systems."""

    print("\n" + "=" * 60)
    print("LAB 1.2: PHASE SPACE VISUALIZER")
    print("=" * 60)

    # System 1: Simple Harmonic Oscillator
    omega = 1.0
    sho = PhaseSpaceVisualizer(
        H_func=lambda q, p: p**2/2 + omega**2*q**2/2,
        dH_dq_func=lambda q, p: omega**2*q,
        dH_dp_func=lambda q, p: p,
        name="Simple Harmonic Oscillator"
    )

    fig1, _ = sho.plot_phase_portrait(
        q_range=(-3, 3), p_range=(-3, 3),
        trajectories=[(1, 0, 10), (2, 0, 10), (0.5, 0.5, 10)]
    )
    plt.savefig('phase_sho.png', dpi=150)
    plt.show()

    # System 2: Simple Pendulum
    g, L = 9.8, 1.0
    pendulum = PhaseSpaceVisualizer(
        H_func=lambda th, p: p**2/2 - g*L*np.cos(th),
        dH_dq_func=lambda th, p: g*L*np.sin(th),
        dH_dp_func=lambda th, p: p,
        name="Simple Pendulum"
    )

    fig2, _ = pendulum.plot_phase_portrait(
        q_range=(-np.pi, np.pi), p_range=(-4, 4),
        trajectories=[(0.5, 0, 20), (2.0, 0, 20), (0.1, 3.5, 20)]
    )
    plt.savefig('phase_pendulum.png', dpi=150)
    plt.show()

    # System 3: Double Well Potential
    double_well = PhaseSpaceVisualizer(
        H_func=lambda q, p: p**2/2 + q**4/4 - q**2/2,
        dH_dq_func=lambda q, p: q**3 - q,
        dH_dp_func=lambda q, p: p,
        name="Double Well: V = q⁴/4 - q²/2"
    )

    fig3, _ = double_well.plot_phase_portrait(
        q_range=(-2, 2), p_range=(-1.5, 1.5),
        trajectories=[(-0.8, 0, 20), (0.8, 0, 20), (0, 1.2, 20)]
    )
    plt.savefig('phase_double_well.png', dpi=150)
    plt.show()

    # Verify energy conservation
    fig4 = sho.verify_energy_conservation(1.5, 0, 20)
    plt.savefig('energy_conservation.png', dpi=150)
    plt.show()


demo_phase_space()
```

---

## Lab Part 2: Symplectic Integrators

### Lab 2.1: Comparison of Integration Methods

```python
"""
Lab 2.1: Symplectic Integrators
Compare symplectic (Störmer-Verlet) vs non-symplectic (Euler) integrators.
"""

import numpy as np
import matplotlib.pyplot as plt

class IntegratorComparison:
    """
    Compare different numerical integration methods for Hamiltonian systems.
    """

    def __init__(self, dH_dq, dH_dp, H):
        """
        Initialize with system functions.

        Parameters:
        -----------
        dH_dq : callable
            ∂H/∂q(q, p)
        dH_dp : callable
            ∂H/∂p(q, p)
        H : callable
            Hamiltonian H(q, p)
        """
        self.dH_dq = dH_dq
        self.dH_dp = dH_dp
        self.H = H

    def euler(self, q0, p0, dt, n_steps):
        """
        Forward Euler method (non-symplectic).

        q_{n+1} = q_n + dt * ∂H/∂p(q_n, p_n)
        p_{n+1} = p_n - dt * ∂H/∂q(q_n, p_n)
        """
        q = np.zeros(n_steps + 1)
        p = np.zeros(n_steps + 1)
        q[0], p[0] = q0, p0

        for n in range(n_steps):
            q[n+1] = q[n] + dt * self.dH_dp(q[n], p[n])
            p[n+1] = p[n] - dt * self.dH_dq(q[n], p[n])

        return q, p

    def symplectic_euler(self, q0, p0, dt, n_steps):
        """
        Symplectic Euler (1st order symplectic).

        p_{n+1} = p_n - dt * ∂H/∂q(q_n, p_{n+1})  [implicit in p]
        q_{n+1} = q_n + dt * ∂H/∂p(q_n, p_{n+1})

        For separable H = T(p) + V(q), this becomes explicit:
        p_{n+1} = p_n - dt * dV/dq(q_n)
        q_{n+1} = q_n + dt * dT/dp(p_{n+1})
        """
        q = np.zeros(n_steps + 1)
        p = np.zeros(n_steps + 1)
        q[0], p[0] = q0, p0

        for n in range(n_steps):
            p[n+1] = p[n] - dt * self.dH_dq(q[n], p[n])
            q[n+1] = q[n] + dt * self.dH_dp(q[n], p[n+1])

        return q, p

    def stormer_verlet(self, q0, p0, dt, n_steps):
        """
        Störmer-Verlet / Leapfrog method (2nd order symplectic).

        p_{n+1/2} = p_n - (dt/2) * ∂H/∂q(q_n)
        q_{n+1} = q_n + dt * ∂H/∂p(p_{n+1/2})
        p_{n+1} = p_{n+1/2} - (dt/2) * ∂H/∂q(q_{n+1})
        """
        q = np.zeros(n_steps + 1)
        p = np.zeros(n_steps + 1)
        q[0], p[0] = q0, p0

        for n in range(n_steps):
            # Half step in p
            p_half = p[n] - 0.5 * dt * self.dH_dq(q[n], p[n])
            # Full step in q
            q[n+1] = q[n] + dt * self.dH_dp(q[n], p_half)
            # Half step in p
            p[n+1] = p_half - 0.5 * dt * self.dH_dq(q[n+1], p_half)

        return q, p

    def rk4(self, q0, p0, dt, n_steps):
        """
        4th-order Runge-Kutta (non-symplectic but accurate).
        """
        q = np.zeros(n_steps + 1)
        p = np.zeros(n_steps + 1)
        q[0], p[0] = q0, p0

        def f(state):
            q, p = state
            return np.array([self.dH_dp(q, p), -self.dH_dq(q, p)])

        for n in range(n_steps):
            state = np.array([q[n], p[n]])
            k1 = f(state)
            k2 = f(state + 0.5*dt*k1)
            k3 = f(state + 0.5*dt*k2)
            k4 = f(state + dt*k3)

            new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            q[n+1], p[n+1] = new_state

        return q, p

    def compare_methods(self, q0, p0, dt, n_steps, title=""):
        """Compare all methods."""

        # Run all integrators
        q_euler, p_euler = self.euler(q0, p0, dt, n_steps)
        q_symp, p_symp = self.symplectic_euler(q0, p0, dt, n_steps)
        q_verlet, p_verlet = self.stormer_verlet(q0, p0, dt, n_steps)
        q_rk4, p_rk4 = self.rk4(q0, p0, dt, n_steps)

        # Compute energies
        E0 = self.H(q0, p0)
        E_euler = np.array([self.H(q_euler[i], p_euler[i]) for i in range(n_steps+1)])
        E_symp = np.array([self.H(q_symp[i], p_symp[i]) for i in range(n_steps+1)])
        E_verlet = np.array([self.H(q_verlet[i], p_verlet[i]) for i in range(n_steps+1)])
        E_rk4 = np.array([self.H(q_rk4[i], p_rk4[i]) for i in range(n_steps+1)])

        t = np.arange(n_steps + 1) * dt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Phase space
        ax = axes[0, 0]
        ax.plot(q_euler, p_euler, 'r-', lw=1, alpha=0.7, label='Euler')
        ax.plot(q_symp, p_symp, 'g-', lw=1, alpha=0.7, label='Symplectic Euler')
        ax.plot(q_verlet, p_verlet, 'b-', lw=1, alpha=0.7, label='Störmer-Verlet')
        ax.plot(q_rk4, p_rk4, 'm-', lw=1, alpha=0.7, label='RK4')
        ax.plot(q0, p0, 'ko', markersize=10, label='Start')
        ax.set_xlabel('q', fontsize=12)
        ax.set_ylabel('p', fontsize=12)
        ax.set_title('Phase Space Trajectories', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Energy vs time
        ax = axes[0, 1]
        ax.plot(t, E_euler, 'r-', lw=1, label='Euler')
        ax.plot(t, E_symp, 'g-', lw=1, label='Symplectic Euler')
        ax.plot(t, E_verlet, 'b-', lw=1, label='Störmer-Verlet')
        ax.plot(t, E_rk4, 'm-', lw=1, label='RK4')
        ax.axhline(E0, color='k', linestyle='--', label=f'True E = {E0:.4f}')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title('Energy vs Time', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Relative energy error
        ax = axes[1, 0]
        ax.semilogy(t, np.abs((E_euler - E0)/E0), 'r-', lw=1, label='Euler')
        ax.semilogy(t, np.abs((E_symp - E0)/E0) + 1e-16, 'g-', lw=1, label='Symplectic Euler')
        ax.semilogy(t, np.abs((E_verlet - E0)/E0) + 1e-16, 'b-', lw=1, label='Störmer-Verlet')
        ax.semilogy(t, np.abs((E_rk4 - E0)/E0) + 1e-16, 'm-', lw=1, label='RK4')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('|ΔE/E₀|', fontsize=12)
        ax.set_title('Relative Energy Error (log scale)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase space area (Liouville's theorem test)
        ax = axes[1, 1]

        # Compute area of phase space region using initial circular cloud
        n_test = 100
        theta_test = np.linspace(0, 2*np.pi, n_test)
        r_test = 0.1

        def compute_area(integrator_func):
            areas = []
            for step in range(0, n_steps+1, max(1, n_steps//50)):
                cloud_q = []
                cloud_p = []
                for th in theta_test:
                    q_init = q0 + r_test * np.cos(th)
                    p_init = p0 + r_test * np.sin(th)
                    q_traj, p_traj = integrator_func(q_init, p_init, dt, step)
                    cloud_q.append(q_traj[-1])
                    cloud_p.append(p_traj[-1])
                # Approximate area using shoelace formula
                cloud_q = np.array(cloud_q)
                cloud_p = np.array(cloud_p)
                area = 0.5 * np.abs(np.sum(cloud_q[:-1]*cloud_p[1:] - cloud_q[1:]*cloud_p[:-1]))
                areas.append(area)
            return np.array(areas)

        steps_for_area = np.arange(0, n_steps+1, max(1, n_steps//50))
        t_area = steps_for_area * dt

        area_initial = np.pi * r_test**2

        ax.axhline(1.0, color='k', linestyle='--', label='Initial area')

        area_verlet = compute_area(self.stormer_verlet)
        area_euler = compute_area(self.euler)

        ax.plot(t_area, area_verlet/area_initial, 'b-', lw=2, label='Verlet')
        ax.plot(t_area, area_euler/area_initial, 'r-', lw=2, label='Euler')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Area / Initial Area', fontsize=12)
        ax.set_title("Liouville's Theorem: Area Preservation", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig


def demo_symplectic_integrators():
    """Demonstrate symplectic integrators."""

    print("\n" + "=" * 60)
    print("LAB 2.1: SYMPLECTIC INTEGRATORS")
    print("=" * 60)

    # Simple Harmonic Oscillator
    omega = 1.0

    comp = IntegratorComparison(
        dH_dq=lambda q, p: omega**2 * q,
        dH_dp=lambda q, p: p,
        H=lambda q, p: p**2/2 + omega**2*q**2/2
    )

    # Long-time integration
    fig = comp.compare_methods(
        q0=1.0, p0=0.0, dt=0.1, n_steps=1000,
        title="SHO: dt=0.1, 1000 steps (Long-time behavior)"
    )
    plt.savefig('integrator_comparison.png', dpi=150)
    plt.show()

    print("\nKey Observations:")
    print("  1. Euler: Energy grows unboundedly (numerical instability)")
    print("  2. Symplectic Euler: Energy bounded, oscillates around true value")
    print("  3. Störmer-Verlet: Excellent energy conservation, 2nd order")
    print("  4. RK4: Good short-term, slight drift long-term")
    print("  5. Symplectic methods preserve phase space area (Liouville)")


demo_symplectic_integrators()
```

---

## Lab Part 3: Integrable vs Chaotic Systems

### Lab 3.1: Poincaré Sections

```python
"""
Lab 3.1: Poincaré Sections
Visualize the transition from integrable to chaotic dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def henon_heiles_system():
    """
    Hénon-Heiles system: a classic example of transition to chaos.

    H = (px² + py²)/2 + (x² + y²)/2 + λ(x²y - y³/3)

    Integrable for λ = 0, chaotic for λ ≠ 0 at high energies.
    """

    print("\n" + "=" * 60)
    print("LAB 3.1: HÉNON-HEILES SYSTEM AND POINCARÉ SECTIONS")
    print("=" * 60)

    def henon_heiles_eom(t, state, lam=1.0):
        x, y, px, py = state
        dxdt = px
        dydt = py
        dpxdt = -x - 2*lam*x*y
        dpydt = -y - lam*(x**2 - y**2)
        return [dxdt, dydt, dpxdt, dpydt]

    def hamiltonian(x, y, px, py, lam=1.0):
        return 0.5*(px**2 + py**2) + 0.5*(x**2 + y**2) + lam*(x**2*y - y**3/3)

    # Poincaré section: record (x, px) when y = 0 with py > 0
    def compute_poincare_section(E, n_orbits=20, t_max=500, lam=1.0):
        """Compute Poincaré section for given energy."""
        section_x = []
        section_px = []

        for _ in range(n_orbits):
            # Random initial x with y=0
            x0 = np.random.uniform(-0.5, 0.5)
            px0 = np.random.uniform(-0.5, 0.5)

            # Compute py from energy conservation: py > 0
            remaining = 2*E - px0**2 - x0**2
            if remaining < 0:
                continue
            py0 = np.sqrt(remaining)

            # Verify energy
            if abs(hamiltonian(x0, 0, px0, py0, lam) - E) > 0.01:
                continue

            # Integrate
            sol = solve_ivp(
                lambda t, s: henon_heiles_eom(t, s, lam),
                [0, t_max], [x0, 0, px0, py0],
                dense_output=True, max_step=0.1
            )

            if not sol.success:
                continue

            # Find crossings y=0 with py>0
            t_fine = np.linspace(0, t_max, int(t_max * 100))
            y_vals = sol.sol(t_fine)[1]
            py_vals = sol.sol(t_fine)[3]

            # Find sign changes in y
            crossings = np.where(np.diff(np.sign(y_vals)))[0]

            for idx in crossings:
                if py_vals[idx] > 0:  # Only upward crossings
                    # Linear interpolation for better accuracy
                    t1, t2 = t_fine[idx], t_fine[idx+1]
                    y1, y2 = y_vals[idx], y_vals[idx+1]
                    t_cross = t1 - y1 * (t2 - t1) / (y2 - y1)

                    state_cross = sol.sol(t_cross)
                    section_x.append(state_cross[0])
                    section_px.append(state_cross[2])

        return np.array(section_x), np.array(section_px)

    # Plot Poincaré sections for different energies
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    energies = [1/24, 1/12, 1/8, 1/6, 1/5, 1/4]  # Escape energy is 1/6

    for ax, E in zip(axes.flat, energies):
        print(f"Computing Poincaré section for E = {E:.4f}...")

        x_sec, px_sec = compute_poincare_section(E, n_orbits=30, t_max=300)

        ax.scatter(x_sec, px_sec, s=0.5, c='blue', alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('p_x')
        ax.set_title(f'E = {E:.4f}' + (' (near escape)' if E >= 1/6 else ''))
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Hénon-Heiles: Poincaré Sections (y=0, py>0)\nTransition from Order to Chaos',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('poincare_henon_heiles.png', dpi=150)
    plt.show()

    print("\nInterpretation:")
    print("  • Low energy: Regular orbits → smooth curves (KAM tori)")
    print("  • Higher energy: Islands of stability amid chaotic sea")
    print("  • Near escape energy: Mostly chaotic (scattered points)")


def double_pendulum_chaos():
    """
    Double pendulum: sensitive dependence on initial conditions.
    """

    print("\n" + "=" * 60)
    print("DOUBLE PENDULUM: CHAOS AND SENSITIVITY")
    print("=" * 60)

    def double_pendulum_eom(t, state, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.8):
        th1, th2, p1, p2 = state

        c = np.cos(th1 - th2)
        s = np.sin(th1 - th2)

        denom = m1 + m2*s**2

        th1_dot = (p1 - m2*L1*L2*c*p2/L1**2) / (L1**2 * denom)
        th2_dot = (p2*(m1+m2) - m2*L1*L2*c*p1/L2**2) / (m2*L2**2 * denom)

        # Simplified equations
        p1_dot = -(m1+m2)*g*L1*np.sin(th1) - m2*L1*L2*th1_dot*th2_dot*s
        p2_dot = -m2*g*L2*np.sin(th2) + m2*L1*L2*th1_dot*th2_dot*s

        return [th1_dot, th2_dot, p1_dot, p2_dot]

    # Two nearby initial conditions
    th1_0, th2_0 = 2.0, 2.0
    p1_0, p2_0 = 0.0, 0.0

    delta = 1e-6  # Tiny perturbation

    t_span = [0, 30]
    t_eval = np.linspace(*t_span, 3000)

    sol1 = solve_ivp(double_pendulum_eom, t_span, [th1_0, th2_0, p1_0, p2_0],
                    t_eval=t_eval, method='RK45')
    sol2 = solve_ivp(double_pendulum_eom, t_span, [th1_0 + delta, th2_0, p1_0, p2_0],
                    t_eval=t_eval, method='RK45')

    # Convert to Cartesian for visualization
    L1, L2 = 1.0, 1.0

    def to_cartesian(th1, th2):
        x1 = L1 * np.sin(th1)
        y1 = -L1 * np.cos(th1)
        x2 = x1 + L2 * np.sin(th2)
        y2 = y1 - L2 * np.cos(th2)
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = to_cartesian(sol1.y[0], sol1.y[1])
    x1_2, y1_2, x2_2, y2_2 = to_cartesian(sol2.y[0], sol2.y[1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Trajectories of second bob
    ax = axes[0]
    ax.plot(x2_1, y2_1, 'b-', lw=0.5, alpha=0.7, label=f'θ₁(0) = {th1_0:.6f}')
    ax.plot(x2_2, y2_2, 'r-', lw=0.5, alpha=0.7, label=f'θ₁(0) = {th1_0+delta:.6f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Double Pendulum: Tip Trajectories\n(Different after short time!)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Angle difference over time
    ax = axes[1]
    angle_diff = np.abs(sol1.y[0] - sol2.y[0])
    ax.semilogy(sol1.t, angle_diff)
    ax.set_xlabel('Time')
    ax.set_ylabel('|Δθ₁|')
    ax.set_title(f'Exponential Divergence\nInitial diff: {delta:.0e}')
    ax.grid(True, alpha=0.3)

    # Phase space (θ₁, p₁)
    ax = axes[2]
    ax.plot(sol1.y[0], sol1.y[2], 'b-', lw=0.3, alpha=0.7)
    ax.set_xlabel('θ₁')
    ax.set_ylabel('p₁')
    ax.set_title('Phase Space: Chaotic Trajectory')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Double Pendulum: Chaos and Sensitive Dependence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('double_pendulum_chaos.png', dpi=150)
    plt.show()


# Run all Part 3 demonstrations
henon_heiles_system()
double_pendulum_chaos()
```

---

## Summary

### Key Code Implementations

| Lab | Topic | Files Generated |
|-----|-------|-----------------|
| 1.1 | Poisson Bracket Calculator | (console output) |
| 1.2 | Phase Space Visualizer | `phase_sho.png`, `phase_pendulum.png` |
| 2.1 | Symplectic Integrators | `integrator_comparison.png` |
| 3.1 | Poincaré Sections | `poincare_henon_heiles.png` |

### Computational Skills Acquired

1. **Symbolic Computing:** SymPy for Poisson brackets
2. **Numerical Integration:** scipy.integrate.odeint, solve_ivp
3. **Visualization:** matplotlib streamplots, contours, phase portraits
4. **Symplectic Methods:** Störmer-Verlet implementation
5. **Chaos Analysis:** Poincaré sections, Lyapunov-like divergence

---

## Daily Checklist

- [ ] Implement Poisson bracket calculator
- [ ] Verify fundamental brackets {qᵢ, pⱼ} = δᵢⱼ
- [ ] Build phase space visualizer
- [ ] Compare Euler vs symplectic integrators
- [ ] Observe energy drift in non-symplectic methods
- [ ] Create Poincaré sections
- [ ] Visualize integrable → chaotic transition
- [ ] Save all figures

---

## Preview: Day 161

Tomorrow we conclude Week 23 with a **comprehensive review** of Hamiltonian Mechanics I: Legendre transform, Hamilton's equations, phase space, Poisson brackets, constants of motion, and the bridge to quantum mechanics. We'll solidify your understanding with problem sets and self-assessment.
