# Day 69: Computational Lab — Second-Order ODEs in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Symbolic & Numerical Solutions |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Oscillation Simulations |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Lab Report |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Solve second-order ODEs symbolically with SymPy
2. Solve second-order ODEs numerically by converting to systems
3. Visualize oscillatory behavior and phase portraits
4. Simulate damped and forced oscillations
5. Analyze resonance computationally

---

## 🖥️ Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from sympy import *

# For nice output
init_printing()
np.set_printoptions(precision=4)
```

---

## 📖 Lab 1: Symbolic Solutions with SymPy

### 1.1 Homogeneous Equations

```python
from sympy import symbols, Function, dsolve, Eq, exp, sin, cos, classify_ode

x = symbols('x')
y = Function('y')

# Example 1: Distinct real roots
# y'' - 5y' + 6y = 0
ode1 = Eq(y(x).diff(x, 2) - 5*y(x).diff(x) + 6*y(x), 0)
sol1 = dsolve(ode1, y(x))
print("y'' - 5y' + 6y = 0:")
print(f"Solution: {sol1}")

# Example 2: Complex roots (oscillation)
# y'' + 4y = 0
ode2 = Eq(y(x).diff(x, 2) + 4*y(x), 0)
sol2 = dsolve(ode2, y(x))
print("\ny'' + 4y = 0:")
print(f"Solution: {sol2}")

# Example 3: Repeated roots
# y'' - 4y' + 4y = 0
ode3 = Eq(y(x).diff(x, 2) - 4*y(x).diff(x) + 4*y(x), 0)
sol3 = dsolve(ode3, y(x))
print("\ny'' - 4y' + 4y = 0:")
print(f"Solution: {sol3}")
```

### 1.2 With Initial Conditions

```python
from sympy import symbols, Function, dsolve, Eq, sin, cos

x = symbols('x')
y = Function('y')

# Solve y'' + 9y = 0, y(0) = 1, y'(0) = 6
ode = Eq(y(x).diff(x, 2) + 9*y(x), 0)

# With initial conditions
sol = dsolve(ode, y(x), ics={y(0): 1, y(x).diff(x).subs(x, 0): 6})
print(f"Solution with ICs: {sol}")

# Simplify and plot
from sympy import lambdify
y_func = lambdify(x, sol.rhs, 'numpy')
x_vals = np.linspace(0, 4*np.pi, 500)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_func(x_vals), 'b-', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Solution to y'' + 9y = 0 with y(0)=1, y'(0)=6")
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.savefig('symbolic_oscillation.png', dpi=150)
plt.show()
```

### 1.3 Nonhomogeneous Equations

```python
from sympy import symbols, Function, dsolve, Eq, exp, cos

x = symbols('x')
y = Function('y')

# Solve y'' + y = cos(x) (resonance case!)
ode = Eq(y(x).diff(x, 2) + y(x), cos(x))
sol = dsolve(ode, y(x))
print(f"y'' + y = cos(x):")
print(f"Solution: {sol}")
# Note the x*sin(x) term - amplitude grows!

# Solve y'' - 4y = exp(3x)
ode2 = Eq(y(x).diff(x, 2) - 4*y(x), exp(3*x))
sol2 = dsolve(ode2, y(x))
print(f"\ny'' - 4y = e^(3x):")
print(f"Solution: {sol2}")
```

---

## 📖 Lab 2: Converting to First-Order Systems

### 2.1 The Key Transformation

Any second-order ODE can be written as a system of first-order ODEs:

$$y'' + p(x)y' + q(x)y = f(x)$$

Let $y_1 = y$ and $y_2 = y'$. Then:

$$\begin{cases} y_1' = y_2 \\ y_2' = f(x) - p(x)y_2 - q(x)y_1 \end{cases}$$

### 2.2 Implementation

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def second_order_to_system(t, Y, p, q, f):
    """
    Convert y'' + p(t)y' + q(t)y = f(t) to system.
    Y = [y, y']
    """
    y, yp = Y
    return [yp, f(t) - p(t)*yp - q(t)*y]

# Example: Simple Harmonic Motion y'' + 4y = 0
def shm_system(t, Y):
    return [Y[1], -4*Y[0]]

# Solve
t_span = (0, 10)
Y0 = [1, 0]  # y(0) = 1, y'(0) = 0
t_eval = np.linspace(0, 10, 500)

sol = solve_ivp(shm_system, t_span, Y0, t_eval=t_eval, dense_output=True)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time series
axes[0].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='y (position)')
axes[0].plot(sol.t, sol.y[1], 'r--', linewidth=2, label="y' (velocity)")
axes[0].set_xlabel('t')
axes[0].set_ylabel('y')
axes[0].set_title("Simple Harmonic Motion: y'' + 4y = 0")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Phase portrait
axes[1].plot(sol.y[0], sol.y[1], 'b-', linewidth=2)
axes[1].set_xlabel('y (position)')
axes[1].set_ylabel("y' (velocity)")
axes[1].set_title('Phase Portrait')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('shm_solution.png', dpi=150)
plt.show()
```

---

## 📖 Lab 3: Damped Oscillations

### 3.1 Three Damping Regimes

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def damped_oscillator(t, Y, gamma, omega0):
    """y'' + 2*gamma*y' + omega0^2*y = 0"""
    y, yp = Y
    return [yp, -2*gamma*yp - omega0**2*y]

omega0 = 2.0  # Natural frequency
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)
Y0 = [1, 0]

# Three regimes
gammas = {
    'Underdamped': 0.5,      # gamma < omega0
    'Critically Damped': 2.0, # gamma = omega0
    'Overdamped': 3.0         # gamma > omega0
}

plt.figure(figsize=(12, 8))

for name, gamma in gammas.items():
    sol = solve_ivp(
        lambda t, Y: damped_oscillator(t, Y, gamma, omega0),
        t_span, Y0, t_eval=t_eval
    )
    plt.plot(sol.t, sol.y[0], linewidth=2, label=f'{name} (γ={gamma})')

plt.xlabel('t', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title(f'Damped Harmonic Oscillator (ω₀ = {omega0})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.savefig('damping_regimes.png', dpi=150)
plt.show()
```

### 3.2 Phase Portraits for Different Damping

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, gamma) in zip(axes, gammas.items()):
    # Multiple initial conditions
    for y0 in [0.5, 1.0, 1.5]:
        for yp0 in [-2, 0, 2]:
            sol = solve_ivp(
                lambda t, Y: damped_oscillator(t, Y, gamma, omega0),
                (0, 15), [y0, yp0], t_eval=np.linspace(0, 15, 500)
            )
            ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel('y')
    ax.set_ylabel("y'")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig('phase_portraits_damping.png', dpi=150)
plt.show()
```

---

## 📖 Lab 4: Forced Oscillations and Resonance

### 4.1 Response to Forcing

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def forced_oscillator(t, Y, gamma, omega0, F0, omega):
    """y'' + 2*gamma*y' + omega0^2*y = F0*cos(omega*t)"""
    y, yp = Y
    return [yp, F0*np.cos(omega*t) - 2*gamma*yp - omega0**2*y]

omega0 = 2.0
gamma = 0.1  # Light damping
F0 = 1.0

# Different driving frequencies
omegas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, omega in zip(axes, omegas):
    sol = solve_ivp(
        lambda t, Y: forced_oscillator(t, Y, gamma, omega0, F0, omega),
        (0, 50), [0, 0], t_eval=np.linspace(0, 50, 2000)
    )
    ax.plot(sol.t, sol.y[0], 'b-', linewidth=1)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.set_title(f'ω = {omega} (ω₀ = {omega0})')
    ax.grid(True, alpha=0.3)

plt.suptitle('Forced Oscillator Response at Different Driving Frequencies', fontsize=14)
plt.tight_layout()
plt.savefig('forced_oscillator.png', dpi=150)
plt.show()
```

### 4.2 Resonance Curve (Amplitude vs Frequency)

```python
def steady_state_amplitude(omega, omega0, gamma, F0):
    """Analytical formula for steady-state amplitude"""
    return F0 / np.sqrt((omega0**2 - omega**2)**2 + (2*gamma*omega)**2)

omega0 = 2.0
F0 = 1.0
omega_range = np.linspace(0.1, 4, 500)

plt.figure(figsize=(12, 8))

for gamma in [0.05, 0.1, 0.2, 0.5, 1.0]:
    A = steady_state_amplitude(omega_range, omega0, gamma, F0)
    plt.plot(omega_range, A, linewidth=2, label=f'γ = {gamma}')

plt.axvline(x=omega0, color='k', linestyle='--', alpha=0.5, label=f'ω₀ = {omega0}')
plt.xlabel('Driving Frequency ω', fontsize=12)
plt.ylabel('Steady-State Amplitude', fontsize=12)
plt.title('Resonance Curves for Different Damping', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 4)
plt.ylim(0, 12)
plt.savefig('resonance_curves.png', dpi=150)
plt.show()
```

### 4.3 Resonance Animation (Undamped Case)

```python
# Resonance: amplitude grows without bound when omega = omega0 and gamma = 0
def undamped_resonance(t, Y, omega0):
    """y'' + omega0^2*y = cos(omega0*t) - Resonance!"""
    y, yp = Y
    return [yp, np.cos(omega0*t) - omega0**2*y]

omega0 = 2.0
sol = solve_ivp(
    lambda t, Y: undamped_resonance(t, Y, omega0),
    (0, 30), [0, 0], t_eval=np.linspace(0, 30, 1000)
)

# Analytical: y_p = t/(2*omega0) * sin(omega0*t)
t_analytical = sol.t
y_envelope = t_analytical / (2*omega0)

plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], 'b-', linewidth=1.5, label='Numerical solution')
plt.plot(t_analytical, y_envelope, 'r--', linewidth=2, label='Envelope: t/(2ω₀)')
plt.plot(t_analytical, -y_envelope, 'r--', linewidth=2)
plt.xlabel('t', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title('Resonance: Undamped Oscillator Driven at Natural Frequency', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('resonance_growth.png', dpi=150)
plt.show()
```

---

## 📖 Lab 5: RLC Circuit Simulation

```python
def rlc_circuit(t, Y, R, L, C, E_func):
    """
    LQ'' + RQ' + Q/C = E(t)
    Y = [Q, I] where I = Q'
    """
    Q, I = Y
    return [I, (E_func(t) - R*I - Q/C) / L]

# Parameters
L = 1.0    # Henry
R = 2.0    # Ohms
C = 0.25   # Farads

# Natural frequency and damping
omega0 = 1/np.sqrt(L*C)  # = 2 rad/s
gamma = R/(2*L)          # = 1

print(f"Natural frequency ω₀ = {omega0:.2f} rad/s")
print(f"Damping coefficient γ = {gamma:.2f}")
print(f"Condition: {'Underdamped' if gamma < omega0 else 'Critically damped' if gamma == omega0 else 'Overdamped'}")

# Different voltage sources
def constant_E(t): return 10
def sine_E(t): return 10*np.sin(2*t)  # At resonance!
def step_E(t): return 10 if t > 1 else 0

t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)
Y0 = [0, 0]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for row, (E_func, name) in enumerate([
    (constant_E, 'Constant 10V'),
    (sine_E, 'Sinusoidal 10sin(2t)'),
    (step_E, 'Step at t=1')
]):
    sol = solve_ivp(
        lambda t, Y: rlc_circuit(t, Y, R, L, C, E_func),
        t_span, Y0, t_eval=t_eval
    )
    
    # Charge
    axes[row, 0].plot(sol.t, sol.y[0], 'b-', linewidth=2)
    axes[row, 0].set_ylabel('Charge Q (C)')
    axes[row, 0].set_title(f'{name}: Charge')
    axes[row, 0].grid(True, alpha=0.3)
    
    # Current
    axes[row, 1].plot(sol.t, sol.y[1], 'r-', linewidth=2)
    axes[row, 1].set_ylabel('Current I (A)')
    axes[row, 1].set_title(f'{name}: Current')
    axes[row, 1].grid(True, alpha=0.3)

axes[2, 0].set_xlabel('Time (s)')
axes[2, 1].set_xlabel('Time (s)')

plt.suptitle(f'RLC Circuit (L={L}H, R={R}Ω, C={C}F)', fontsize=14)
plt.tight_layout()
plt.savefig('rlc_circuit_response.png', dpi=150)
plt.show()
```

---

## 📖 Lab 6: Quantum Harmonic Oscillator

```python
from scipy.special import hermite
from math import factorial
import numpy as np
import matplotlib.pyplot as plt

def quantum_harmonic_wavefunctions(x, n_max, m=1, omega=1, hbar=1):
    """
    Compute quantum harmonic oscillator wavefunctions.
    psi_n(x) = (m*omega/(pi*hbar))^(1/4) * 1/sqrt(2^n * n!) * H_n(xi) * exp(-xi^2/2)
    where xi = sqrt(m*omega/hbar) * x
    """
    alpha = np.sqrt(m * omega / hbar)
    xi = alpha * x
    
    wavefunctions = []
    for n in range(n_max + 1):
        H_n = hermite(n)
        norm = (m*omega/(np.pi*hbar))**0.25 / np.sqrt(2**n * factorial(n))
        psi = norm * H_n(xi) * np.exp(-xi**2 / 2)
        wavefunctions.append(psi)
    
    return wavefunctions

# Parameters
x = np.linspace(-5, 5, 500)
n_max = 5

# Compute wavefunctions
psi = quantum_harmonic_wavefunctions(x, n_max)

# Plot wavefunctions
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Wavefunctions
for n in range(n_max + 1):
    axes[0].plot(x, psi[n] + n, linewidth=2, label=f'n={n}')
    axes[0].axhline(y=n, color='gray', linestyle='--', alpha=0.3)

axes[0].set_xlabel('x')
axes[0].set_ylabel('ψₙ(x) + n (offset for clarity)')
axes[0].set_title('Quantum Harmonic Oscillator Wavefunctions')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Probability densities
for n in range(n_max + 1):
    axes[1].plot(x, np.abs(psi[n])**2, linewidth=2, label=f'n={n}')

axes[1].set_xlabel('x')
axes[1].set_ylabel('|ψₙ(x)|²')
axes[1].set_title('Probability Densities')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quantum_harmonic_oscillator.png', dpi=150)
plt.show()

# Energy levels
print("\nEnergy Levels (E_n = ℏω(n + 1/2)):")
for n in range(n_max + 1):
    E_n = (n + 0.5)  # In units of ℏω
    print(f"  n = {n}: E_{n} = {E_n:.1f} ℏω")
```

---

## ✅ Lab Completion Checklist

- [ ] Solved second-order ODEs symbolically with SymPy
- [ ] Converted second-order to first-order systems
- [ ] Simulated all three damping regimes
- [ ] Created phase portraits
- [ ] Demonstrated resonance behavior
- [ ] Simulated RLC circuits
- [ ] Visualized quantum harmonic oscillator wavefunctions
- [ ] Saved all figures

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Solves a damped harmonic oscillator analytically and numerically, comparing results
2. Creates an interactive resonance demonstration (vary driving frequency)
3. Simulates an RLC circuit and analyzes its frequency response
4. Bonus: Animate the time evolution of quantum harmonic oscillator states

Save as `Week10_SecondOrderODEs_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

---

*"Computation reveals what equations describe—from spring oscillations to quantum states."*
