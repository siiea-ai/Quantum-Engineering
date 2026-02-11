# Day 167: Computational Lab — Advanced Hamiltonian Mechanics

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Labs 1-3: Symplectic Methods & HJ |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Labs 4-5: Chaos & Visualization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Integration Project |

**Total Study Time: 7 hours**

---

## Lab Overview

This computational lab consolidates all the numerical methods from Week 24:

| Lab | Topic | Key Techniques |
|-----|-------|----------------|
| 1 | Symplectic Integrators | Verlet, Yoshida, energy conservation |
| 2 | Canonical Transformations | Generating functions, verification |
| 3 | Hamilton-Jacobi Equation | Action computation, trajectory extraction |
| 4 | Chaos Analysis | Lyapunov exponents, Poincaré sections |
| 5 | Comprehensive Project | Double pendulum chaos analysis |

---

## Lab 1: Symplectic Integrators

### Objective
Implement and compare symplectic integrators for long-time Hamiltonian simulation.

### Theory Review

Symplectic integrators preserve the symplectic structure of phase space, ensuring:
- Bounded energy error (no drift)
- Preserved phase space volume (Liouville)
- Qualitatively correct long-time behavior

### Code Implementation

```python
"""
Lab 1: Symplectic Integrators for Hamiltonian Systems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class HamiltonianIntegrators:
    """Collection of integrators for Hamiltonian systems."""

    def __init__(self, dVdq, T_func=None, m=1.0):
        """
        Initialize with potential gradient dV/dq.

        Args:
            dVdq: Function returning -∂V/∂q (the force)
            T_func: Kinetic energy function (default: p²/2m)
            m: Mass parameter
        """
        self.dVdq = dVdq
        self.m = m

    def euler(self, q0, p0, dt, n_steps):
        """Standard Euler (non-symplectic)."""
        q, p = np.zeros(n_steps+1), np.zeros(n_steps+1)
        q[0], p[0] = q0, p0

        for i in range(n_steps):
            q[i+1] = q[i] + dt * p[i] / self.m
            p[i+1] = p[i] - dt * self.dVdq(q[i])

        return q, p

    def symplectic_euler(self, q0, p0, dt, n_steps):
        """Symplectic Euler (1st order symplectic)."""
        q, p = np.zeros(n_steps+1), np.zeros(n_steps+1)
        q[0], p[0] = q0, p0

        for i in range(n_steps):
            p[i+1] = p[i] - dt * self.dVdq(q[i])  # Update p first
            q[i+1] = q[i] + dt * p[i+1] / self.m   # Use new p

        return q, p

    def stormer_verlet(self, q0, p0, dt, n_steps):
        """Störmer-Verlet (2nd order symplectic)."""
        q, p = np.zeros(n_steps+1), np.zeros(n_steps+1)
        q[0], p[0] = q0, p0

        for i in range(n_steps):
            p_half = p[i] - 0.5 * dt * self.dVdq(q[i])
            q[i+1] = q[i] + dt * p_half / self.m
            p[i+1] = p_half - 0.5 * dt * self.dVdq(q[i+1])

        return q, p

    def yoshida4(self, q0, p0, dt, n_steps):
        """Yoshida 4th order symplectic integrator."""
        # Yoshida coefficients
        w1 = 1.0 / (2 - 2**(1/3))
        w0 = -2**(1/3) * w1
        c = [w1/2, (w0+w1)/2, (w0+w1)/2, w1/2]
        d = [w1, w0, w1, 0]

        q, p = np.zeros(n_steps+1), np.zeros(n_steps+1)
        q[0], p[0] = q0, p0

        for i in range(n_steps):
            qi, pi = q[i], p[i]
            for j in range(4):
                qi = qi + c[j] * dt * pi / self.m
                pi = pi - d[j] * dt * self.dVdq(qi)
            q[i+1], p[i+1] = qi, pi

        return q, p


def compare_integrators():
    """Compare integrators on the harmonic oscillator."""

    omega = 1.0
    m = 1.0

    # Harmonic oscillator: V = (1/2)mω²q², dV/dq = mω²q
    dVdq = lambda q: m * omega**2 * q

    integrator = HamiltonianIntegrators(dVdq, m=m)

    # Parameters
    q0, p0 = 1.0, 0.0
    E0 = 0.5 * p0**2 / m + 0.5 * m * omega**2 * q0**2
    dt = 0.1
    n_steps = 5000
    t = np.arange(n_steps + 1) * dt

    # Run all integrators
    methods = {
        'Euler': integrator.euler,
        'Symplectic Euler': integrator.symplectic_euler,
        'Störmer-Verlet': integrator.stormer_verlet,
        'Yoshida 4th': integrator.yoshida4
    }

    results = {}
    for name, method in methods.items():
        q, p = method(q0, p0, dt, n_steps)
        E = 0.5 * p**2 / m + 0.5 * m * omega**2 * q**2
        results[name] = {'q': q, 'p': p, 'E': E}

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Phase space
    ax = axes[0, 0]
    for name, data in results.items():
        ax.plot(data['q'], data['p'], alpha=0.7, lw=0.5, label=name)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=2, label='Exact')
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title('Phase Space')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Energy vs time
    ax = axes[0, 1]
    for name, data in results.items():
        ax.plot(t, data['E'], alpha=0.8, lw=0.5, label=name)
    ax.axhline(y=E0, color='k', ls='--', lw=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Conservation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Energy error
    ax = axes[1, 0]
    for name, data in results.items():
        ax.semilogy(t, np.abs(data['E'] - E0) + 1e-16, alpha=0.8, label=name)
    ax.set_xlabel('Time')
    ax.set_ylabel('|E - E₀|')
    ax.set_title('Energy Error (log scale)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Order of convergence
    ax = axes[1, 1]
    dt_values = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    n_period = int(2 * np.pi / dt_values[-1])

    errors = {name: [] for name in methods.keys()}
    for dt_test in dt_values:
        n_test = int(2 * np.pi / dt_test)
        for name, method in methods.items():
            q, p = method(q0, p0, dt_test, n_test)
            E_final = 0.5 * p[-1]**2 / m + 0.5 * m * omega**2 * q[-1]**2
            errors[name].append(abs(E_final - E0))

    colors = plt.cm.tab10(range(len(methods)))
    for (name, err), color in zip(errors.items(), colors):
        ax.loglog(dt_values, err, 'o-', color=color, label=name)

    # Reference slopes
    ax.loglog(dt_values, 0.1 * dt_values, 'k:', alpha=0.5, label='O(dt)')
    ax.loglog(dt_values, 0.01 * dt_values**2, 'k--', alpha=0.5, label='O(dt²)')
    ax.loglog(dt_values, 0.001 * dt_values**4, 'k-.', alpha=0.5, label='O(dt⁴)')

    ax.set_xlabel('dt')
    ax.set_ylabel('Energy error after 1 period')
    ax.set_title('Order of Convergence')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Comparison of Numerical Integrators\nHarmonic Oscillator', fontsize=14)
    plt.tight_layout()
    plt.savefig('integrator_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\nFinal Energy Errors (after t = {:.0f}):".format(n_steps * dt))
    print("=" * 50)
    for name, data in results.items():
        err = abs(data['E'][-1] - E0) / E0 * 100
        print(f"{name:20s}: {err:.6f}%")


if __name__ == "__main__":
    compare_integrators()
```

### Exercises

1. **Modify for the pendulum:** Change dV/dq to sin(q) and observe how integrators perform for the nonlinear pendulum.

2. **Long-time stability:** Run for 10,000 periods. Which methods remain stable?

3. **Step size study:** Plot energy error vs dt on a log-log scale to verify the order of each method.

---

## Lab 2: Canonical Transformations

### Objective
Implement and verify canonical transformations using generating functions.

### Code Implementation

```python
"""
Lab 2: Canonical Transformations
Verification and visualization
"""

import numpy as np
import matplotlib.pyplot as plt

def verify_canonical(Q_func, P_func, n_test=100):
    """
    Verify a transformation is canonical by checking {Q, P} = 1.

    Args:
        Q_func: Function Q(q, p)
        P_func: Function P(q, p)
        n_test: Number of random test points

    Returns:
        Array of Poisson bracket values (should all be ≈ 1)
    """
    eps = 1e-8
    results = []

    np.random.seed(42)
    for _ in range(n_test):
        q = np.random.uniform(-2, 2)
        p = np.random.uniform(-2, 2)

        # Numerical partial derivatives
        dQdq = (Q_func(q + eps, p) - Q_func(q - eps, p)) / (2 * eps)
        dQdp = (Q_func(q, p + eps) - Q_func(q, p - eps)) / (2 * eps)
        dPdq = (P_func(q + eps, p) - P_func(q - eps, p)) / (2 * eps)
        dPdp = (P_func(q, p + eps) - P_func(q, p - eps)) / (2 * eps)

        # Poisson bracket {Q, P} = ∂Q/∂q × ∂P/∂p - ∂Q/∂p × ∂P/∂q
        poisson_bracket = dQdq * dPdp - dQdp * dPdq
        results.append(poisson_bracket)

    return np.array(results)


def generating_function_demo():
    """
    Demonstrate Type-2 generating function: F₂(q, P) = qP + αq³

    From F₂: p = ∂F₂/∂q = P + 3αq²
             Q = ∂F₂/∂P = q

    So: Q = q, P = p - 3αq²
    """
    alpha = 0.5

    # Define the transformation
    Q_func = lambda q, p: q
    P_func = lambda q, p: p - 3 * alpha * q**2

    # Verify canonical
    pb_values = verify_canonical(Q_func, P_func)

    print("Generating Function F₂(q, P) = qP + 0.5q³")
    print("Transformation: Q = q, P = p - 1.5q²")
    print(f"\nPoisson bracket {{Q, P}}:")
    print(f"  Mean: {np.mean(pb_values):.10f}")
    print(f"  Std:  {np.std(pb_values):.2e}")
    print(f"  Should be: 1.0")

    # Visualize the transformation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Grid in (q, p)
    q = np.linspace(-2, 2, 20)
    p = np.linspace(-2, 2, 20)
    Q_grid, P_grid = np.meshgrid(q, p)

    # Transform to (Q, P)
    Q_new = Q_func(Q_grid, P_grid)
    P_new = P_func(Q_grid, P_grid)

    # Plot original grid
    ax = axes[0]
    for i in range(len(q)):
        ax.plot(Q_grid[i, :], P_grid[i, :], 'b-', alpha=0.5)
        ax.plot(Q_grid[:, i], P_grid[:, i], 'r-', alpha=0.5)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title('Original Coordinates (q, p)\nRectangular Grid')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot transformed grid
    ax = axes[1]
    for i in range(len(q)):
        ax.plot(Q_new[i, :], P_new[i, :], 'b-', alpha=0.5)
        ax.plot(Q_new[:, i], P_new[:, i], 'r-', alpha=0.5)
    ax.set_xlabel('Q')
    ax.set_ylabel('P')
    ax.set_title('Transformed Coordinates (Q, P)\nArea Preserved!')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Canonical Transformation from F₂(q, P) = qP + 0.5q³', fontsize=14)
    plt.tight_layout()
    plt.savefig('canonical_transformation.png', dpi=150, bbox_inches='tight')
    plt.show()


def action_angle_transformation():
    """
    Implement the action-angle transformation for harmonic oscillator.

    Transformation:
        q = √(2J/mω) cos(θ)
        p = √(2mωJ) sin(θ)

    Inverse:
        J = (p² + m²ω²q²)/(2mω) = H/ω
        θ = arctan(mωq/p)
    """
    m, omega = 1.0, 1.0

    def qp_to_Jtheta(q, p):
        """Convert (q, p) to (θ, J)."""
        J = (p**2 + m**2 * omega**2 * q**2) / (2 * m * omega)
        theta = np.arctan2(m * omega * q, p)
        return theta, J

    def Jtheta_to_qp(theta, J):
        """Convert (θ, J) to (q, p)."""
        q = np.sqrt(2 * J / (m * omega)) * np.cos(theta)
        p = np.sqrt(2 * m * omega * J) * np.sin(theta)
        return q, p

    # Verify it's canonical
    Q_func = lambda q, p: np.arctan2(m * omega * q, p)  # θ
    P_func = lambda q, p: (p**2 + m**2 * omega**2 * q**2) / (2 * m * omega)  # J

    pb_values = verify_canonical(Q_func, P_func)
    print("\nAction-Angle Transformation for Harmonic Oscillator:")
    print(f"Poisson bracket {{θ, J}} = {np.mean(pb_values):.6f} (should be 1)")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Phase space trajectories
    ax = axes[0]
    J_values = [0.5, 1.0, 2.0, 3.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(J_values)))

    for J, color in zip(J_values, colors):
        theta = np.linspace(0, 2*np.pi, 100)
        q, p = Jtheta_to_qp(theta, J)
        ax.plot(q, p, color=color, lw=2, label=f'J = {J}')

    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title('Phase Space (q, p)\nEllipses of constant J')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Action-angle space
    ax = axes[1]
    for J, color in zip(J_values, colors):
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, np.full_like(theta, J), color=color, lw=3, label=f'J = {J}')

    ax.set_xlabel('θ')
    ax.set_ylabel('J')
    ax.set_title('Action-Angle (θ, J)\nHorizontal lines!')
    ax.legend()
    ax.set_xlim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.grid(True, alpha=0.3)

    plt.suptitle('Action-Angle Variables: Harmonic Oscillator', fontsize=14)
    plt.tight_layout()
    plt.savefig('action_angle_transformation.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    generating_function_demo()
    action_angle_transformation()
```

### Exercises

1. **Exchange transformation:** Verify that Q = p, P = -q is canonical.

2. **Scaling transformation:** For Q = λq, P = p/λ, verify canonicity and find the generating function.

3. **Time-dependent transformation:** Implement a rotating frame transformation and verify it preserves the Poisson brackets.

---

## Lab 3: Hamilton-Jacobi Equation

### Objective
Solve the Hamilton-Jacobi equation numerically and extract trajectories.

### Code Implementation

```python
"""
Lab 3: Hamilton-Jacobi Equation
Numerical solution and trajectory extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d

def characteristic_function_1d(E, V_func, q_range, m=1.0, n_points=500):
    """
    Compute Hamilton's characteristic function W(q) for a 1D system.

    W(q) = ∫ p dq = ∫ √(2m(E - V(q))) dq

    Args:
        E: Total energy
        V_func: Potential energy function V(q)
        q_range: (q_min, q_max) range to compute
        m: Mass
        n_points: Number of grid points

    Returns:
        q_vals, W_vals: Arrays of position and W values
    """
    q_vals = np.linspace(q_range[0], q_range[1], n_points)
    W_vals = np.zeros(n_points)

    for i, q in enumerate(q_vals):
        # Find the classically allowed region
        def integrand(qp):
            kinetic = E - V_func(qp)
            if kinetic > 0:
                return np.sqrt(2 * m * kinetic)
            return 0

        # Integrate from q_range[0] to q
        W, _ = quad(integrand, q_range[0], q, limit=100)
        W_vals[i] = W

    return q_vals, W_vals


def extract_trajectory_from_W(W_func, E, m=1.0, t_max=10, dt=0.01):
    """
    Extract q(t) from the Hamilton-Jacobi solution.

    Using: p = dW/dq and dq/dt = p/m
    """
    # Numerical derivative of W
    dW = lambda q: (W_func(q + 1e-6) - W_func(q - 1e-6)) / (2e-6)

    def equations(t, y):
        q = y[0]
        p = dW(q)  # p = ∂W/∂q
        return [p / m]

    # Initial condition (starting from left turning point)
    q0 = [0.1]  # Small positive value

    sol = solve_ivp(equations, [0, t_max], q0, t_eval=np.arange(0, t_max, dt),
                    method='RK45', max_step=dt/2)

    return sol.t, sol.y[0]


def hamilton_jacobi_demo():
    """Demonstrate HJ solution for harmonic oscillator."""

    m, omega = 1.0, 1.0

    # Potential
    V = lambda q: 0.5 * m * omega**2 * q**2

    # Energy levels to compute
    E_values = [0.5, 1.0, 2.0, 3.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(E_values)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: W(q) for different energies
    ax = axes[0, 0]
    for E, color in zip(E_values, colors):
        q_max = np.sqrt(2 * E / (m * omega**2))
        q_range = (-q_max * 0.99, q_max * 0.99)
        q_vals, W_vals = characteristic_function_1d(E, V, q_range, m)
        ax.plot(q_vals, W_vals, color=color, lw=2, label=f'E = {E}')

    ax.set_xlabel('q')
    ax.set_ylabel('W(q)')
    ax.set_title("Hamilton's Characteristic Function W(q)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: p = dW/dq (should give correct phase space curves)
    ax = axes[0, 1]
    for E, color in zip(E_values, colors):
        q_max = np.sqrt(2 * E / (m * omega**2))
        q_range = (-q_max * 0.99, q_max * 0.99)
        q_vals, W_vals = characteristic_function_1d(E, V, q_range, m)

        # Numerical derivative for p
        p_vals = np.gradient(W_vals, q_vals)
        ax.plot(q_vals, p_vals, color=color, lw=2, label=f'E = {E}')

        # Also plot the lower branch
        ax.plot(q_vals, -p_vals, color=color, lw=2)

    ax.set_xlabel('q')
    ax.set_ylabel('p = dW/dq')
    ax.set_title('Phase Space from HJ Solution')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 3: S(q, t) = W(q) - Et (wavefronts)
    ax = axes[1, 0]
    E = 2.0
    q_max = np.sqrt(2 * E / (m * omega**2))
    q_vals, W_vals = characteristic_function_1d(E, V, (-q_max*0.99, q_max*0.99), m)

    t_values = [0, 0.5, 1.0, 1.5]
    for t, ls in zip(t_values, ['-', '--', '-.', ':']):
        S_vals = W_vals - E * t
        ax.plot(q_vals, S_vals, ls=ls, lw=2, label=f't = {t}')

    ax.set_xlabel('q')
    ax.set_ylabel('S(q, t) = W(q) - Et')
    ax.set_title("Hamilton's Principal Function S(q, t)\nWavefronts at different times")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Trajectory comparison
    ax = axes[1, 1]
    E = 2.0
    q_max = np.sqrt(2 * E / (m * omega**2))

    # Exact solution
    t_exact = np.linspace(0, 4*np.pi, 500)
    q_exact = q_max * np.sin(omega * t_exact)
    ax.plot(t_exact, q_exact, 'k-', lw=2, label='Exact')

    # From direct integration
    def ho_eqs(t, y):
        q, p = y
        return [p/m, -m*omega**2*q]

    sol = solve_ivp(ho_eqs, [0, 4*np.pi], [0, np.sqrt(2*m*E)],
                    t_eval=t_exact, method='RK45')
    ax.plot(sol.t, sol.y[0], 'b--', lw=2, alpha=0.7, label='Direct integration')

    ax.set_xlabel('t')
    ax.set_ylabel('q(t)')
    ax.set_title('Trajectory Extraction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hamilton-Jacobi Equation: Harmonic Oscillator', fontsize=14)
    plt.tight_layout()
    plt.savefig('hamilton_jacobi_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    hamilton_jacobi_demo()
```

### Exercises

1. **Quartic potential:** Solve the HJ equation for V(q) = αq⁴ and compare with numerical integration.

2. **Gravitational field:** For V(q) = mgq, verify the HJ solution gives parabolic motion.

3. **Action integral:** Compute the action J = (1/2π)∮p dq from the HJ solution and verify J = E/ω for the harmonic oscillator.

---

## Lab 4: Chaos Analysis

### Objective
Implement tools for detecting and visualizing chaos in Hamiltonian systems.

### Code Implementation

```python
"""
Lab 4: Chaos Analysis Tools
Lyapunov exponents, Poincaré sections, and phase space visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ChaosAnalyzer:
    """Tools for analyzing chaotic Hamiltonian systems."""

    def __init__(self, equations, jacobian=None):
        """
        Args:
            equations: Function f(t, y) returning dy/dt
            jacobian: Optional Jacobian function J(t, y)
        """
        self.equations = equations
        self.jacobian = jacobian

    def lyapunov_exponent(self, y0, T=100, dt=0.01, eps=1e-9):
        """
        Compute maximal Lyapunov exponent using trajectory divergence.
        """
        n = len(y0)
        y1 = np.array(y0)
        y2 = np.array(y0) + eps * np.array([1] + [0]*(n-1))

        lyap_sum = 0
        n_renorm = 0
        renorm_interval = 1.0

        t = 0
        while t < T:
            # Integrate both trajectories
            sol1 = solve_ivp(self.equations, [t, t + renorm_interval],
                           y1, method='RK45', max_step=dt)
            sol2 = solve_ivp(self.equations, [t, t + renorm_interval],
                           y2, method='RK45', max_step=dt)

            y1 = sol1.y[:, -1]
            y2 = sol2.y[:, -1]

            # Compute separation
            d = np.linalg.norm(y2 - y1)

            # Accumulate
            lyap_sum += np.log(d / eps)

            # Renormalize
            y2 = y1 + eps * (y2 - y1) / d

            n_renorm += 1
            t += renorm_interval

        return lyap_sum / T

    def poincare_section(self, y0, T=1000, dt=0.01, section_var=0,
                         section_value=0, direction=1):
        """
        Compute Poincaré section.

        Args:
            y0: Initial condition
            T: Total integration time
            section_var: Index of variable defining the section
            section_value: Value where section is taken
            direction: +1 for positive crossings, -1 for negative

        Returns:
            section_points: List of (y[i], y[j]) at crossings
        """
        section_points = []

        def event(t, y):
            return y[section_var] - section_value

        event.direction = direction

        sol = solve_ivp(self.equations, [0, T], y0, method='RK45',
                       events=event, max_step=dt, dense_output=True)

        if sol.t_events[0].size > 0:
            for t_cross in sol.t_events[0]:
                y_cross = sol.sol(t_cross)
                # Return non-section variables
                other_vars = [y_cross[i] for i in range(len(y0))
                             if i != section_var]
                section_points.append(other_vars)

        return np.array(section_points)


def standard_map_analysis():
    """Complete analysis of the Chirikov standard map."""

    def standard_map(theta, p, K):
        """One iteration of standard map."""
        p_new = (p + K * np.sin(theta)) % (2 * np.pi)
        theta_new = (theta + p_new) % (2 * np.pi)
        return theta_new, p_new

    def lyapunov_standard_map(K, n_iter=10000):
        """Compute Lyapunov exponent."""
        theta = np.random.random() * 2 * np.pi
        p = np.random.random() * 2 * np.pi

        v = np.array([1.0, 0.0])
        lyap_sum = 0

        for _ in range(n_iter):
            J = np.array([[1, K * np.cos(theta)],
                          [1, 1 + K * np.cos(theta)]])
            v = J @ v
            norm_v = np.linalg.norm(v)
            lyap_sum += np.log(norm_v)
            v = v / norm_v

            p = (p + K * np.sin(theta)) % (2 * np.pi)
            theta = (theta + p) % (2 * np.pi)

        return lyap_sum / n_iter

    # Create comprehensive analysis figure
    fig = plt.figure(figsize=(16, 12))

    # Phase spaces for different K
    K_values = [0.5, 0.9, 0.9716, 2.0]

    for idx, K in enumerate(K_values):
        ax = fig.add_subplot(2, 3, idx + 1)

        np.random.seed(42)
        for _ in range(30):
            theta0 = np.random.random() * 2 * np.pi
            p0 = np.random.random() * 2 * np.pi

            thetas, ps = [theta0], [p0]
            theta, p = theta0, p0

            for _ in range(500):
                theta, p = standard_map(theta, p, K)
                thetas.append(theta)
                ps.append(p)

            ax.plot(thetas, ps, ',', markersize=0.5, alpha=0.5)

        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        ax.set_xlabel('θ')
        ax.set_ylabel('p')
        ax.set_title(f'K = {K}')

    # Lyapunov exponent vs K
    ax = fig.add_subplot(2, 3, 5)
    K_range = np.linspace(0, 5, 50)
    lyap_values = [lyapunov_standard_map(K) for K in K_range]

    ax.plot(K_range, lyap_values, 'b-', lw=2)
    ax.axhline(y=0, color='k', ls='--', alpha=0.5)
    ax.axvline(x=0.9716, color='r', ls='--', alpha=0.5, label='K_c')
    ax.set_xlabel('K')
    ax.set_ylabel('λ')
    ax.set_title('Lyapunov Exponent vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fraction of chaotic orbits
    ax = fig.add_subplot(2, 3, 6)

    def fraction_chaotic(K, n_orbits=50, lyap_threshold=0.01):
        """Estimate fraction of chaotic orbits."""
        n_chaotic = 0
        for _ in range(n_orbits):
            theta0 = np.random.random() * 2 * np.pi
            p0 = np.random.random() * 2 * np.pi

            # Short Lyapunov calculation for this orbit
            theta, p = theta0, p0
            v = np.array([1.0, 0.0])
            lyap_sum = 0

            for i in range(1000):
                J = np.array([[1, K * np.cos(theta)],
                              [1, 1 + K * np.cos(theta)]])
                v = J @ v
                norm_v = np.linalg.norm(v)
                lyap_sum += np.log(norm_v)
                v = v / norm_v
                p = (p + K * np.sin(theta)) % (2 * np.pi)
                theta = (theta + p) % (2 * np.pi)

            lyap = lyap_sum / 1000
            if lyap > lyap_threshold:
                n_chaotic += 1

        return n_chaotic / n_orbits

    K_range2 = np.linspace(0, 3, 30)
    frac_chaotic = [fraction_chaotic(K) for K in K_range2]

    ax.plot(K_range2, frac_chaotic, 'bo-', lw=2)
    ax.axvline(x=0.9716, color='r', ls='--', alpha=0.5, label='K_c')
    ax.set_xlabel('K')
    ax.set_ylabel('Fraction chaotic')
    ax.set_title('Chaos Fraction vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Chirikov Standard Map: Complete Chaos Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('standard_map_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    standard_map_analysis()
```

---

## Lab 5: Comprehensive Project — Double Pendulum

### Objective
Perform a complete dynamical analysis of the double pendulum.

### Code Implementation

```python
"""
Lab 5: Double Pendulum - Complete Chaos Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

class DoublePendulum:
    """Complete analysis of the double pendulum system."""

    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
        self.L1, self.L2 = L1, L2
        self.m1, self.m2 = m1, m2
        self.g = g

    def equations(self, t, state):
        """Equations of motion."""
        theta1, theta2, omega1, omega2 = state

        delta = theta2 - theta1
        den1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * np.cos(delta)**2
        den2 = (self.L2 / self.L1) * den1

        domega1 = (self.m2 * self.L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                   self.m2 * self.g * np.sin(theta2) * np.cos(delta) +
                   self.m2 * self.L2 * omega2**2 * np.sin(delta) -
                   (self.m1 + self.m2) * self.g * np.sin(theta1)) / den1

        domega2 = (-self.m2 * self.L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                   (self.m1 + self.m2) * (self.g * np.sin(theta1) * np.cos(delta) -
                                         self.L1 * omega1**2 * np.sin(delta) -
                                         self.g * np.sin(theta2))) / den2

        return [omega1, omega2, domega1, domega2]

    def energy(self, state):
        """Compute total energy."""
        theta1, theta2, omega1, omega2 = state

        T = 0.5 * (self.m1 + self.m2) * self.L1**2 * omega1**2 + \
            0.5 * self.m2 * self.L2**2 * omega2**2 + \
            self.m2 * self.L1 * self.L2 * omega1 * omega2 * np.cos(theta1 - theta2)

        V = -(self.m1 + self.m2) * self.g * self.L1 * np.cos(theta1) - \
            self.m2 * self.g * self.L2 * np.cos(theta2)

        return T + V

    def lyapunov_exponent(self, state0, T=50, dt=0.01):
        """Compute maximal Lyapunov exponent."""
        eps = 1e-9
        state1 = np.array(state0)
        state2 = np.array(state0) + np.array([eps, 0, 0, 0])

        lyap_sum = 0
        n_renorm = 0
        renorm_interval = 1.0

        t = 0
        while t < T:
            sol1 = solve_ivp(self.equations, [t, t + renorm_interval],
                           state1, method='RK45', max_step=dt)
            sol2 = solve_ivp(self.equations, [t, t + renorm_interval],
                           state2, method='RK45', max_step=dt)

            state1 = sol1.y[:, -1]
            state2 = sol2.y[:, -1]

            d = np.linalg.norm(state2 - state1)
            lyap_sum += np.log(d / eps)
            state2 = state1 + eps * (state2 - state1) / d

            n_renorm += 1
            t += renorm_interval

        return lyap_sum / T

    def poincare_section(self, state0, T=500, dt=0.01):
        """Compute Poincaré section at θ₁ = 0 (downward crossing)."""
        section_points = []

        def event(t, y):
            return y[0]  # θ₁ = 0
        event.direction = -1  # Downward crossing

        sol = solve_ivp(self.equations, [0, T], state0,
                       events=event, method='RK45', max_step=dt)

        if sol.t_events[0].size > 0:
            for i, t_cross in enumerate(sol.t_events[0]):
                y = sol.y_events[0][i]
                section_points.append([y[1], y[3]])  # (θ₂, ω₂)

        return np.array(section_points) if section_points else np.array([[]])


def complete_double_pendulum_analysis():
    """Run complete analysis of the double pendulum."""

    dp = DoublePendulum()

    fig = plt.figure(figsize=(16, 14))

    # 1. Sensitive dependence
    ax = fig.add_subplot(2, 3, 1)
    T = 10
    t_eval = np.linspace(0, T, 2000)
    epsilons = [0, 1e-10, 1e-8]
    colors = ['blue', 'red', 'green']

    for eps, color in zip(epsilons, colors):
        state0 = [np.pi/2 + eps, np.pi/2, 0, 0]
        sol = solve_ivp(dp.equations, [0, T], state0, t_eval=t_eval, method='RK45')
        label = f'ε = {eps:.0e}' if eps > 0 else 'Reference'
        ax.plot(sol.t, sol.y[0], color=color, alpha=0.7, label=label)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ₁ (rad)')
    ax.set_title('Sensitive Dependence on Initial Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Phase space projection
    ax = fig.add_subplot(2, 3, 2)
    state0 = [np.pi/2, np.pi/2, 0, 0]
    sol = solve_ivp(dp.equations, [0, 50], state0, method='RK45', max_step=0.01)
    ax.plot(sol.y[0], sol.y[2], 'b-', lw=0.3, alpha=0.5)
    ax.set_xlabel('θ₁')
    ax.set_ylabel('ω₁')
    ax.set_title('Phase Space (θ₁, ω₁)')
    ax.grid(True, alpha=0.3)

    # 3. Poincaré section
    ax = fig.add_subplot(2, 3, 3)

    # Multiple initial conditions
    np.random.seed(42)
    for i in range(10):
        theta1_0 = np.random.uniform(-0.5, 0.5)
        theta2_0 = np.random.uniform(-np.pi, np.pi)
        omega1_0 = np.random.uniform(-2, 2)
        omega2_0 = np.random.uniform(-2, 2)

        section = dp.poincare_section([theta1_0, theta2_0, omega1_0, omega2_0], T=500)
        if section.size > 2:
            ax.plot(section[:, 0], section[:, 1], '.', markersize=1, alpha=0.5)

    ax.set_xlabel('θ₂')
    ax.set_ylabel('ω₂')
    ax.set_title('Poincaré Section (θ₁ = 0, crossing down)')
    ax.grid(True, alpha=0.3)

    # 4. Lyapunov exponent vs energy
    ax = fig.add_subplot(2, 3, 4)
    theta_range = np.linspace(0.1, np.pi, 15)
    lyap_values = []
    energies = []

    for theta in theta_range:
        state0 = [theta, theta, 0, 0]
        E = dp.energy(state0)
        lyap = dp.lyapunov_exponent(state0, T=30)
        energies.append(E)
        lyap_values.append(lyap)

    ax.plot(energies, lyap_values, 'bo-', lw=2)
    ax.axhline(y=0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Lyapunov exponent (1/s)')
    ax.set_title('Chaos vs Energy')
    ax.grid(True, alpha=0.3)

    # 5. Tip trajectory (chaotic)
    ax = fig.add_subplot(2, 3, 5)
    state0 = [np.pi/2, np.pi/2, 0, 0]
    sol = solve_ivp(dp.equations, [0, 20], state0, method='RK45', max_step=0.01)

    x2 = dp.L1 * np.sin(sol.y[0]) + dp.L2 * np.sin(sol.y[1])
    y2 = -dp.L1 * np.cos(sol.y[0]) - dp.L2 * np.cos(sol.y[1])

    ax.plot(x2, y2, 'b-', lw=0.3, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Tip Trajectory (Chaotic)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 6. Energy conservation
    ax = fig.add_subplot(2, 3, 6)
    E_values = [dp.energy(sol.y[:, i]) for i in range(len(sol.t))]
    E0 = E_values[0]
    ax.plot(sol.t, (np.array(E_values) - E0) / abs(E0) * 100, 'b-', lw=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ΔE/E₀ (%)')
    ax.set_title('Energy Conservation Check')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Double Pendulum: Complete Chaos Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('double_pendulum_complete.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    complete_double_pendulum_analysis()
```

---

## Summary

### Computational Skills Developed

| Lab | Skills |
|-----|--------|
| 1 | Symplectic integration, energy conservation analysis |
| 2 | Canonical transformation verification, action-angle variables |
| 3 | Hamilton-Jacobi numerical solution, trajectory extraction |
| 4 | Lyapunov exponents, Poincaré sections, chaos detection |
| 5 | Complete system analysis, multiple visualization techniques |

### Key Algorithms Implemented

1. **Symplectic Integrators:** Euler, Verlet, Yoshida
2. **Poisson Bracket Verification:** Numerical differentiation
3. **Action Computation:** Phase space integration
4. **Lyapunov Exponents:** Trajectory divergence method
5. **Poincaré Sections:** Event detection in ODE integration

---

## Daily Checklist

- [ ] Implemented all 5 labs
- [ ] Verified symplectic integrators preserve energy
- [ ] Confirmed canonical transformations satisfy {Q, P} = 1
- [ ] Computed action variables numerically
- [ ] Calculated Lyapunov exponents for standard map
- [ ] Generated Poincaré sections showing regular vs chaotic regions
- [ ] Completed double pendulum analysis project

---

## Preview: Day 168

Tomorrow is the **Year 0 Final Review**, consolidating all material from the Mathematical Foundations year:

- Calculus and Differential Equations
- Linear Algebra and Complex Analysis
- Classical Mechanics (Lagrangian and Hamiltonian)
- Connections to Quantum Mechanics

This completes Year 0 and prepares for Year 1: Quantum Mechanics Core.

---

**Day 167 Complete. Next: Year 0 Final Review**
