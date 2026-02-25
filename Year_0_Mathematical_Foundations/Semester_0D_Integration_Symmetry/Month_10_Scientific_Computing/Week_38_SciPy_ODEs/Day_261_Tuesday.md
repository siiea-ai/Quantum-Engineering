# Day 261: Ordinary Differential Equations with SciPy

## Overview

**Day 261** | **Week 38** | **Month 10: Scientific Computing**

Today we solve ordinary differential equations using `scipy.integrate.solve_ivp`. The time-dependent Schrödinger equation, classical mechanics, and quantum dynamics all reduce to systems of ODEs. Mastering these solvers enables simulation of quantum evolution, wave packet dynamics, and coupled systems.

**Prerequisites:** Day 260 (integration), differential equations (Month 3)
**Outcome:** Solve initial value problems for quantum dynamics

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: ODE solvers, RK methods, stiff equations |
| Afternoon | 3 hours | Practice: Quantum dynamics, two-level systems |
| Evening | 2 hours | Lab: Time-dependent Schrödinger equation |

---

## Learning Objectives

By the end of Day 261, you will be able to:

1. **Solve initial value problems** using `solve_ivp`
2. **Choose appropriate solvers** (RK45, RK23, BDF, Radau)
3. **Handle stiff equations** common in quantum systems
4. **Implement event detection** for state transitions
5. **Solve the time-dependent Schrödinger equation** numerically
6. **Simulate two-level (qubit) dynamics** with Rabi oscillations
7. **Verify accuracy** with energy conservation and unitarity

---

## Core Content

### 1. Basic ODE Solving

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ODE: dy/dt = f(t, y)
# Example: exponential decay dy/dt = -λy

def exponential_decay(t, y, lam=1.0):
    return -lam * y

# Solve from t=0 to t=5, initial condition y(0)=1
solution = solve_ivp(
    exponential_decay,
    t_span=[0, 5],
    y0=[1.0],
    args=(1.0,),  # λ = 1
    dense_output=True  # Enable interpolation
)

print(f"Solution status: {solution.message}")
print(f"Time points: {len(solution.t)}")
print(f"Final value: y(5) = {solution.y[0, -1]:.6f}")
print(f"Exact: e^(-5) = {np.exp(-5):.6f}")

# Evaluate at arbitrary times using dense output
t_fine = np.linspace(0, 5, 100)
y_fine = solution.sol(t_fine)
```

### 2. Systems of ODEs

```python
# Harmonic oscillator: d²x/dt² = -ω²x
# Convert to first-order system: y = [x, dx/dt]
# dy/dt = [y[1], -ω²y[0]]

def harmonic_oscillator(t, y, omega=1.0):
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

# Initial conditions: x(0)=1, v(0)=0
solution = solve_ivp(
    harmonic_oscillator,
    t_span=[0, 20],
    y0=[1.0, 0.0],
    args=(1.0,),
    dense_output=True,
    max_step=0.1  # Control step size for smooth output
)

t = np.linspace(0, 20, 500)
y = solution.sol(t)
x, v = y[0], y[1]

# Verify energy conservation: E = ½v² + ½ω²x² = ½
E = 0.5 * v**2 + 0.5 * x**2
print(f"Energy conservation: E_max - E_min = {np.max(E) - np.min(E):.2e}")
```

### 3. Choosing Solvers

```python
# Available methods in solve_ivp:
# - 'RK45': Runge-Kutta 4(5), default, non-stiff
# - 'RK23': Runge-Kutta 2(3), faster but less accurate
# - 'DOP853': High-order (8th), very accurate
# - 'Radau': Implicit Runge-Kutta, stiff equations
# - 'BDF': Backward differentiation, stiff equations
# - 'LSODA': Auto-switch between stiff/non-stiff

# Non-stiff example
sol_rk45 = solve_ivp(harmonic_oscillator, [0, 10], [1, 0], method='RK45')
sol_rk23 = solve_ivp(harmonic_oscillator, [0, 10], [1, 0], method='RK23')
sol_dop853 = solve_ivp(harmonic_oscillator, [0, 10], [1, 0], method='DOP853')

print(f"RK45: {len(sol_rk45.t)} steps")
print(f"RK23: {len(sol_rk23.t)} steps")
print(f"DOP853: {len(sol_dop853.t)} steps")

# Stiff equation example: Van der Pol oscillator with large μ
def vanderpol(t, y, mu=1000):
    x, v = y
    return [v, mu * (1 - x**2) * v - x]

# BDF handles stiff equations much better than RK45
sol_stiff = solve_ivp(vanderpol, [0, 3000], [2, 0],
                      method='BDF', args=(1000,))
print(f"Stiff (BDF): {len(sol_stiff.t)} steps, success={sol_stiff.success}")
```

### 4. Error Control

```python
# Control absolute and relative tolerances
def pendulum(t, y):
    theta, omega = y
    return [omega, -np.sin(theta)]

# Default tolerances
sol_default = solve_ivp(pendulum, [0, 100], [np.pi/4, 0])

# High precision
sol_precise = solve_ivp(pendulum, [0, 100], [np.pi/4, 0],
                        rtol=1e-10, atol=1e-12)

print(f"Default: {len(sol_default.t)} steps")
print(f"Precise: {len(sol_precise.t)} steps")
```

### 5. Event Detection

```python
def bouncing_ball(t, y, g=9.8, restitution=0.9):
    """Free fall with ground at y=0."""
    height, velocity = y
    return [velocity, -g]

def hit_ground(t, y, g=9.8, restitution=0.9):
    """Event: ball hits ground."""
    return y[0]  # height = 0

hit_ground.terminal = True  # Stop integration
hit_ground.direction = -1   # Only falling through zero

# Simulate bounces
y0 = [10.0, 0.0]  # Start at height 10, velocity 0
t_total = 0
bounces = 0
trajectory = []

while bounces < 5:
    sol = solve_ivp(bouncing_ball, [t_total, t_total + 100], y0,
                    events=hit_ground, dense_output=True)

    trajectory.append(sol)
    t_total = sol.t[-1]
    bounces += 1

    # Reverse velocity with energy loss
    y0 = [0.0, -0.9 * sol.y[1, -1]]

print(f"Simulated {bounces} bounces")
```

---

## Quantum Mechanics Connection

### Two-Level System (Qubit) Dynamics

The Schrödinger equation for a two-level system:
$$i\hbar\frac{d|\psi\rangle}{dt} = H|\psi\rangle$$

With $|\psi\rangle = c_0(t)|0\rangle + c_1(t)|1\rangle$:

```python
def two_level_system(t, psi, H, hbar=1.0):
    """
    Time evolution of two-level system.

    psi = [Re(c0), Im(c0), Re(c1), Im(c1)]
    """
    c0 = psi[0] + 1j*psi[1]
    c1 = psi[2] + 1j*psi[3]

    # Schrödinger: i*hbar * dc/dt = H @ c
    c = np.array([c0, c1])
    dc = -1j/hbar * (H @ c)

    return [dc[0].real, dc[0].imag, dc[1].real, dc[1].imag]

# Rabi oscillations: H = ℏ[[0, Ω/2], [Ω/2, 0]]
def rabi_hamiltonian(omega_rabi):
    return np.array([[0, omega_rabi/2],
                     [omega_rabi/2, 0]])

# Start in |0⟩, evolve with Rabi frequency Ω=1
omega_rabi = 1.0
H = rabi_hamiltonian(omega_rabi)

psi0 = [1.0, 0.0, 0.0, 0.0]  # |0⟩ state

sol = solve_ivp(
    two_level_system,
    t_span=[0, 4*np.pi],
    y0=psi0,
    args=(H,),
    dense_output=True,
    max_step=0.1
)

t = np.linspace(0, 4*np.pi, 500)
psi_t = sol.sol(t)
P0 = psi_t[0]**2 + psi_t[1]**2  # |c0|²
P1 = psi_t[2]**2 + psi_t[3]**2  # |c1|²

# Verify: P0 + P1 = 1 (unitarity)
print(f"Unitarity check: max|P0+P1-1| = {np.max(np.abs(P0 + P1 - 1)):.2e}")

# Analytic: P1(t) = sin²(Ωt/2)
P1_analytic = np.sin(omega_rabi * t / 2)**2
print(f"Rabi accuracy: max error = {np.max(np.abs(P1 - P1_analytic)):.2e}")
```

### Time-Dependent Schrödinger Equation (Discretized)

For N-level system: $i\hbar\dot{\psi} = H\psi$

```python
def schrodinger_ivp(t, psi_flat, H, hbar=1.0):
    """
    Schrödinger equation for N-level system.

    psi_flat = [Re(ψ₀), Im(ψ₀), Re(ψ₁), Im(ψ₁), ...]
    """
    N = len(psi_flat) // 2
    psi = psi_flat[::2] + 1j * psi_flat[1::2]

    dpsi = -1j/hbar * (H @ psi)

    result = np.zeros(2*N)
    result[::2] = dpsi.real
    result[1::2] = dpsi.imag
    return result

def solve_schrodinger(H, psi0, t_span, t_eval=None, hbar=1.0):
    """
    Solve time-dependent Schrödinger equation.

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix (N x N)
    psi0 : ndarray
        Initial state (complex N-vector)
    t_span : tuple
        (t_start, t_end)
    t_eval : ndarray, optional
        Times at which to store solution

    Returns
    -------
    t : ndarray
        Time points
    psi : ndarray
        States at each time (N x len(t))
    """
    N = len(psi0)

    # Flatten to real representation
    psi0_flat = np.zeros(2*N)
    psi0_flat[::2] = psi0.real
    psi0_flat[1::2] = psi0.imag

    sol = solve_ivp(
        schrodinger_ivp,
        t_span,
        psi0_flat,
        args=(H, hbar),
        t_eval=t_eval,
        method='DOP853',  # High accuracy for oscillatory problems
        rtol=1e-10,
        atol=1e-12
    )

    # Reconstruct complex wave function
    psi = sol.y[::2] + 1j * sol.y[1::2]

    return sol.t, psi

# Example: 3-level system
H3 = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

psi0 = np.array([1, 0, 0], dtype=complex)  # Start in state 0
t, psi = solve_schrodinger(H3, psi0, (0, 20), t_eval=np.linspace(0, 20, 200))

# Check normalization
norms = np.sum(np.abs(psi)**2, axis=0)
print(f"Normalization preserved: max deviation = {np.max(np.abs(norms - 1)):.2e}")
```

### Driven Qubit (Time-Dependent Hamiltonian)

```python
def driven_qubit(t, psi_flat, omega0, omega_drive, amplitude, hbar=1.0):
    """
    Qubit with time-dependent driving field.

    H(t) = ℏω₀/2 σz + ℏA cos(ωt) σx
    """
    c0 = psi_flat[0] + 1j*psi_flat[1]
    c1 = psi_flat[2] + 1j*psi_flat[3]

    # Time-dependent Hamiltonian
    drive = amplitude * np.cos(omega_drive * t)
    H = np.array([
        [omega0/2, drive],
        [drive, -omega0/2]
    ])

    c = np.array([c0, c1])
    dc = -1j/hbar * (H @ c)

    return [dc[0].real, dc[0].imag, dc[1].real, dc[1].imag]

# On-resonance driving: ω_drive = ω0
omega0 = 10.0
omega_drive = 10.0
amplitude = 1.0

psi0 = [1.0, 0.0, 0.0, 0.0]

sol = solve_ivp(
    driven_qubit,
    [0, 20],
    psi0,
    args=(omega0, omega_drive, amplitude),
    dense_output=True,
    max_step=0.01  # Small steps for oscillatory dynamics
)

t = np.linspace(0, 20, 1000)
psi = sol.sol(t)
P1 = psi[2]**2 + psi[3]**2

print(f"Max excitation probability: {np.max(P1):.4f}")
```

---

## Worked Examples

### Example 1: Quantum Harmonic Oscillator Dynamics

```python
def coherent_state_dynamics():
    """
    Evolve a coherent state in harmonic oscillator potential.

    In Fock basis: |α⟩ = e^{-|α|²/2} Σ (α^n/√n!) |n⟩
    """
    from math import factorial

    N = 20  # Truncation
    alpha = 2.0  # Coherent state parameter
    omega = 1.0

    # Coherent state coefficients
    c0 = np.zeros(N, dtype=complex)
    for n in range(N):
        c0[n] = np.exp(-np.abs(alpha)**2/2) * alpha**n / np.sqrt(factorial(n))

    # Hamiltonian: H = ℏω(n + 1/2)
    H = np.diag([omega * (n + 0.5) for n in range(N)])

    # Evolve
    t_span = (0, 4*np.pi/omega)
    t_eval = np.linspace(*t_span, 200)
    t, psi = solve_schrodinger(H, c0, t_span, t_eval)

    # Compute ⟨n⟩ vs time
    n_op = np.diag(np.arange(N))
    n_expect = np.array([np.real(np.conj(psi[:, i]) @ n_op @ psi[:, i])
                         for i in range(len(t))])

    print(f"Initial ⟨n⟩ = {np.abs(alpha)**2:.4f}")
    print(f"⟨n⟩ at t=2π/ω: {n_expect[100]:.4f}")
    print(f"⟨n⟩ preserved: max deviation = {np.max(np.abs(n_expect - np.abs(alpha)**2)):.2e}")

    return t, psi, n_expect

t, psi, n_expect = coherent_state_dynamics()
```

### Example 2: Landau-Zener Transition

```python
def landau_zener(v, Delta):
    """
    Landau-Zener problem: level crossing with velocity v.

    H(t) = [[vt, Δ], [Δ, -vt]]

    Transition probability: P = 1 - exp(-πΔ²/ℏv)
    """
    def hamiltonian(t, psi_flat):
        c0 = psi_flat[0] + 1j*psi_flat[1]
        c1 = psi_flat[2] + 1j*psi_flat[3]

        H = np.array([[v*t, Delta], [Delta, -v*t]])
        c = np.array([c0, c1])
        dc = -1j * (H @ c)

        return [dc[0].real, dc[0].imag, dc[1].real, dc[1].imag]

    # Start in lower adiabatic state at t = -T (≈ ground state)
    T = 20 / np.sqrt(v)  # Large enough for adiabatic limit
    psi0 = [1.0, 0.0, 0.0, 0.0]

    sol = solve_ivp(hamiltonian, [-T, T], psi0, dense_output=True,
                    rtol=1e-10, atol=1e-12)

    # Final state
    psi_final = sol.y[:, -1]
    P_transition = psi_final[2]**2 + psi_final[3]**2

    # Landau-Zener formula
    P_lz = 1 - np.exp(-np.pi * Delta**2 / v)

    return P_transition, P_lz

# Test for different sweep velocities
print("Landau-Zener transitions (Δ=1):")
print(f"{'v':>8} {'P_num':>10} {'P_LZ':>10} {'Error':>10}")
for v in [0.1, 0.5, 1.0, 2.0, 5.0]:
    P_num, P_lz = landau_zener(v, Delta=1.0)
    print(f"{v:>8.1f} {P_num:>10.6f} {P_lz:>10.6f} {abs(P_num-P_lz):>10.2e}")
```

### Example 3: Spin-Boson Model (Jaynes-Cummings)

```python
def jaynes_cummings(n_photons=5, g=0.1, omega_a=1.0, omega_c=1.0):
    """
    Jaynes-Cummings model: atom + cavity mode.

    H = ω_a σz/2 + ω_c (a†a + 1/2) + g(a†σ- + aσ+)
    """
    N = n_photons + 1  # Photon number states 0 to n_photons
    dim = 2 * N  # 2 atomic states × N photon states

    # Basis: |g,0⟩, |g,1⟩, ..., |g,N-1⟩, |e,0⟩, |e,1⟩, ..., |e,N-1⟩

    H = np.zeros((dim, dim))

    # Atom energy: ω_a σz/2
    for n in range(N):
        H[n, n] = -omega_a/2 + omega_c * n  # ground
        H[N+n, N+n] = omega_a/2 + omega_c * n  # excited

    # Interaction: g(a†σ- + aσ+)
    for n in range(N-1):
        # |e,n⟩ ↔ |g,n+1⟩
        H[n+1, N+n] = g * np.sqrt(n+1)  # a†σ-
        H[N+n, n+1] = g * np.sqrt(n+1)  # aσ+

    return H

# Simulate: start in |e,0⟩ (excited atom, vacuum)
H_jc = jaynes_cummings(n_photons=10, g=0.5, omega_a=1.0, omega_c=1.0)
dim = H_jc.shape[0]
N = dim // 2

psi0 = np.zeros(dim, dtype=complex)
psi0[N] = 1.0  # |e,0⟩

t, psi = solve_schrodinger(H_jc, psi0, (0, 50), t_eval=np.linspace(0, 50, 500))

# Compute atomic excitation probability
P_excited = np.sum(np.abs(psi[N:])**2, axis=0)

print("Jaynes-Cummings model:")
print(f"  Vacuum Rabi frequency: {2*0.5:.2f}")
print(f"  Period: {2*np.pi/(2*0.5):.2f}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Solve the damped harmonic oscillator $\ddot{x} + 2\gamma\dot{x} + \omega_0^2 x = 0$ and plot the trajectory for underdamped, critically damped, and overdamped cases.

**Problem 2:** Implement a Runge-Kutta 4 solver from scratch and compare with `solve_ivp` for the pendulum equation.

**Problem 3:** Solve for Rabi oscillations with detuning: $H = \begin{pmatrix} \Delta/2 & \Omega/2 \\ \Omega/2 & -\Delta/2 \end{pmatrix}$.

### Intermediate

**Problem 4:** Simulate a spin-1/2 particle in a rotating magnetic field and verify the adiabatic theorem.

**Problem 5:** Implement the split-operator method for time evolution and compare accuracy with direct ODE solving.

**Problem 6:** Solve the Gross-Pitaevskii equation $i\hbar\partial_t\psi = (-\hbar^2\nabla^2/2m + g|\psi|^2)\psi$ for a 1D Bose-Einstein condensate.

### Challenging

**Problem 7:** Implement adaptive time-stepping based on error estimates and compare with `solve_ivp`'s built-in adaptation.

**Problem 8:** Solve the Lindblad master equation for a decaying two-level system: $\dot{\rho} = -i[H,\rho] + \gamma(L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\})$.

**Problem 9:** Simulate quantum adiabatic computation: evolve from an easy Hamiltonian to a problem Hamiltonian slowly.

---

## Computational Lab

```python
"""
Day 261 Lab: ODE Solvers for Quantum Dynamics
=============================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable

# [Full lab code with demonstrations...]

if __name__ == "__main__":
    print("=" * 70)
    print("Day 261: ODE Solvers for Quantum Dynamics")
    print("=" * 70)

    # Run demonstrations...
    print("\n" + "=" * 70)
    print("Lab complete! Optimization continues on Day 262.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `solve_ivp(f, t_span, y0)` | Solve IVP | `solve_ivp(f, [0,10], [1,0])` |
| `method='RK45'` | Runge-Kutta 4(5) | Default, non-stiff |
| `method='BDF'` | Backward differentiation | Stiff equations |
| `dense_output=True` | Interpolation | `sol.sol(t_array)` |
| `events=func` | Detect events | Stop at transitions |

### Quantum ODE Patterns

| System | ODE Form |
|--------|----------|
| Schrödinger | $i\hbar\dot{\psi} = H\psi$ → real/imag split |
| Two-level | 4 real ODEs for $c_0, c_1$ |
| N-level | 2N real ODEs |
| Time-dependent H | H(t) in RHS |

---

## Daily Checklist

- [ ] Can solve first-order and systems of ODEs with `solve_ivp`
- [ ] Know when to use different solvers (stiff vs non-stiff)
- [ ] Can control accuracy with tolerances
- [ ] Implemented Schrödinger equation solver
- [ ] Simulated Rabi oscillations
- [ ] Verified unitarity and energy conservation
- [ ] Completed practice problems
- [ ] Ran lab successfully

---

## Preview: Day 262

Tomorrow we tackle **Root Finding and Optimization** with `scipy.optimize`. We'll learn:
- Finding roots of nonlinear equations
- Minimizing energy functionals (variational principle)
- Curve fitting for experimental data
- Constrained optimization

Essential for variational quantum methods!

---

*"Differential equations are the poetry of physics; numerical solvers are the translation."*
