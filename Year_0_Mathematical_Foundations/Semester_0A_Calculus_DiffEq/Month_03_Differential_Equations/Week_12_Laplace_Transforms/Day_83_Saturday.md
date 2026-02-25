# Day 83: Computational Lab — Laplace Transforms in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Symbolic Laplace Transforms |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications & Visualization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Lab Report |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute Laplace transforms symbolically with SymPy
2. Solve ODEs using the transform method in Python
3. Visualize step and impulse responses
4. Compare transform vs direct numerical solutions
5. Simulate RLC circuits and control systems

---

## 🖥️ Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import *
from sympy.integrals.transforms import laplace_transform, inverse_laplace_transform

# Initialize pretty printing
init_printing()

# Symbols
t, s = symbols('t s', positive=True, real=True)
```

---

## 📖 Lab 1: Symbolic Laplace Transforms

### 1.1 Computing Transforms

```python
from sympy import symbols, exp, sin, cos, laplace_transform, Heaviside

t, s = symbols('t s', positive=True)

# Basic transforms
functions = [
    1,
    t,
    t**2,
    exp(-3*t),
    sin(2*t),
    cos(2*t),
    t*exp(-t),
    exp(-t)*sin(3*t)
]

print("Laplace Transforms:")
print("=" * 50)
for f in functions:
    F, cond, _ = laplace_transform(f, t, s)
    print(f"L{{{f}}} = {F}")
```

### 1.2 Inverse Transforms

```python
from sympy import inverse_laplace_transform, apart

# Define some transforms
F1 = 1/(s*(s+1))
F2 = (s+3)/((s+1)*(s+2))
F3 = (2*s+1)/(s**2 + 4*s + 8)

transforms = [F1, F2, F3]

print("\nInverse Laplace Transforms:")
print("=" * 50)
for F in transforms:
    # Partial fractions first
    F_pf = apart(F, s)
    f = inverse_laplace_transform(F, s, t)
    print(f"L^(-1){{{F}}} = {simplify(f)}")
    print(f"  Partial fractions: {F_pf}")
    print()
```

---

## 📖 Lab 2: Solving ODEs with Laplace Transforms

### 2.1 First-Order ODE

```python
from sympy import Function, dsolve, Eq, symbols, laplace_transform, inverse_laplace_transform

t, s = symbols('t s', positive=True)
Y = symbols('Y')  # Y(s) in Laplace domain

# Solve y' + 2y = e^(-t), y(0) = 1

# Manual Laplace approach
# L{y'} = sY - y(0) = sY - 1
# L{e^(-t)} = 1/(s+1)
# (sY - 1) + 2Y = 1/(s+1)
# Y(s+2) = 1 + 1/(s+1)
# Y = 1/(s+2) + 1/((s+1)(s+2))

Y_s = 1/(s+2) + 1/((s+1)*(s+2))
Y_s_simplified = apart(Y_s, s)
print(f"Y(s) = {Y_s_simplified}")

y_t = inverse_laplace_transform(Y_s, s, t)
print(f"y(t) = {simplify(y_t)}")

# Verify with dsolve
y = Function('y')
ode = Eq(y(t).diff(t) + 2*y(t), exp(-t))
sol = dsolve(ode, y(t), ics={y(0): 1})
print(f"Direct solution: {sol}")
```

### 2.2 Second-Order ODE

```python
from sympy import symbols, Function, Eq, exp, sin, cos, apart, simplify
from sympy.integrals.transforms import inverse_laplace_transform

t, s = symbols('t s', positive=True)

# Solve y'' + 4y = 0, y(0) = 1, y'(0) = 2
# L{y''} = s²Y - s*y(0) - y'(0) = s²Y - s - 2
# s²Y - s - 2 + 4Y = 0
# (s² + 4)Y = s + 2
# Y = (s + 2)/(s² + 4)

Y_s = (s + 2)/(s**2 + 4)
print(f"Y(s) = {Y_s}")

# Decompose
Y_cos = s/(s**2 + 4)  # -> cos(2t)
Y_sin = 2/(s**2 + 4)  # -> sin(2t)

y_t = inverse_laplace_transform(Y_s, s, t)
print(f"y(t) = {y_t}")

# Numerical verification
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def ode_system(t, y):
    return [y[1], -4*y[0]]

sol = solve_ivp(ode_system, [0, 10], [1, 2], t_eval=np.linspace(0, 10, 200))

# Analytical solution
t_vals = np.linspace(0, 10, 200)
y_analytical = np.cos(2*t_vals) + np.sin(2*t_vals)

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Numerical')
plt.plot(t_vals, y_analytical, 'r--', linewidth=2, label='Laplace Solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("y'' + 4y = 0, y(0)=1, y'(0)=2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('laplace_ode_comparison.png', dpi=150)
plt.show()
```

---

## 📖 Lab 3: Step and Impulse Response

### 3.1 Impulse Response (Green's Function)

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# System: y'' + 2y' + 5y = f(t)
# Impulse response: f(t) = δ(t)

# Solution via Laplace: Y = 1/(s² + 2s + 5) = 1/((s+1)² + 4)
# y(t) = (1/2)e^(-t)sin(2t)

def impulse_response(t):
    return 0.5 * np.exp(-t) * np.sin(2*t)

t_vals = np.linspace(0, 6, 500)
h = impulse_response(t_vals)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_vals, h, 'b-', linewidth=2)
plt.xlabel('t')
plt.ylabel('h(t)')
plt.title('Impulse Response: h(t) = (1/2)e^(-t)sin(2t)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)

# Step response (integral of impulse response)
# Or solve y'' + 2y' + 5y = u(t)
def system(t, y):
    return [y[1], 1 - 2*y[1] - 5*y[0]]  # forcing = 1 for t > 0

sol = solve_ivp(system, [0, 6], [0, 0], t_eval=t_vals)

plt.subplot(1, 2, 2)
plt.plot(sol.t, sol.y[0], 'r-', linewidth=2)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Step Response')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.2, color='g', linestyle='--', label='Steady state = 1/5')
plt.legend()

plt.tight_layout()
plt.savefig('impulse_step_response.png', dpi=150)
plt.show()
```

### 3.2 Response to Switched Input

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# y' + y = f(t), where f(t) = 1 for 1 ≤ t < 3, else 0

def forcing(t):
    return 1.0 if 1 <= t < 3 else 0.0

def system(t, y):
    return -y[0] + forcing(t)

sol = solve_ivp(system, [0, 8], [0], t_eval=np.linspace(0, 8, 500), max_step=0.01)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
t_vals = np.linspace(0, 8, 500)
f_vals = [forcing(t) for t in t_vals]
plt.plot(t_vals, f_vals, 'g-', linewidth=2)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Input: Rectangular Pulse')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.5)

plt.subplot(1, 2, 2)
plt.plot(sol.t, sol.y[0], 'b-', linewidth=2)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Response: y' + y = f(t), y(0) = 0")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('switched_input_response.png', dpi=150)
plt.show()
```

---

## 📖 Lab 4: RLC Circuit Simulation

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def rlc_circuit(t, y, R, L, C, E_func):
    """
    LQ'' + RQ' + Q/C = E(t)
    State: y = [Q, I] where I = dQ/dt
    """
    Q, I = y
    dQ = I
    dI = (E_func(t) - R*I - Q/C) / L
    return [dQ, dI]

# Parameters
L = 0.5   # Henry
R = 2.0   # Ohms
C = 0.1   # Farads

# Natural frequency and damping
omega0 = 1/np.sqrt(L*C)
gamma = R/(2*L)
print(f"Natural frequency ω₀ = {omega0:.2f} rad/s")
print(f"Damping coefficient γ = {gamma:.2f}")
print(f"Regime: {'Underdamped' if gamma < omega0 else 'Overdamped'}")

# Different inputs
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)
y0 = [0, 0]

# 1. Step input (DC voltage)
E_step = lambda t: 10.0

# 2. Impulse (approximate)
E_impulse = lambda t: 100.0 if 0 <= t < 0.1 else 0.0

# 3. Sinusoidal at resonance
E_resonance = lambda t: 10*np.sin(omega0*t)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Step response
sol_step = solve_ivp(lambda t, y: rlc_circuit(t, y, R, L, C, E_step),
                     t_span, y0, t_eval=t_eval, max_step=0.01)
axes[0, 0].plot(sol_step.t, sol_step.y[0], 'b-', linewidth=2, label='Charge Q')
axes[0, 0].plot(sol_step.t, sol_step.y[1], 'r--', linewidth=2, label='Current I')
axes[0, 0].axhline(y=C*10, color='g', linestyle=':', label=f'Steady state Q = {C*10}')
axes[0, 0].set_title('Step Response (E = 10V)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Impulse response
sol_impulse = solve_ivp(lambda t, y: rlc_circuit(t, y, R, L, C, E_impulse),
                        t_span, y0, t_eval=t_eval, max_step=0.01)
axes[0, 1].plot(sol_impulse.t, sol_impulse.y[0], 'b-', linewidth=2, label='Charge Q')
axes[0, 1].plot(sol_impulse.t, sol_impulse.y[1], 'r--', linewidth=2, label='Current I')
axes[0, 1].set_title('Impulse Response')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Resonance
sol_res = solve_ivp(lambda t, y: rlc_circuit(t, y, R, L, C, E_resonance),
                    t_span, y0, t_eval=t_eval, max_step=0.01)
axes[1, 0].plot(sol_res.t, sol_res.y[0], 'b-', linewidth=2, label='Charge Q')
axes[1, 0].set_title(f'Driven at Resonance (ω = {omega0:.2f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Frequency response
freqs = np.linspace(0.1, 10, 100)
amplitudes = []
for omega in freqs:
    # Steady-state amplitude formula
    A = 1/np.sqrt((1/(L*C) - omega**2)**2 + (R*omega/L)**2)
    amplitudes.append(A)

axes[1, 1].plot(freqs, amplitudes, 'b-', linewidth=2)
axes[1, 1].axvline(x=omega0, color='r', linestyle='--', label=f'ω₀ = {omega0:.2f}')
axes[1, 1].set_xlabel('Driving Frequency ω')
axes[1, 1].set_ylabel('Amplitude Response')
axes[1, 1].set_title('Frequency Response (Bode Plot)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rlc_circuit_analysis.png', dpi=150)
plt.show()
```

---

## 📖 Lab 5: Transfer Functions

```python
from sympy import symbols, apart, simplify, latex
from sympy.integrals.transforms import inverse_laplace_transform

s, t = symbols('s t', positive=True)

# Transfer function: H(s) = 1/(s² + 2s + 5)
H = 1/(s**2 + 2*s + 5)

print("Transfer Function Analysis")
print("=" * 50)
print(f"H(s) = {H}")

# Poles
poles = solve(s**2 + 2*s + 5, s)
print(f"\nPoles: {poles}")

# Impulse response
h = inverse_laplace_transform(H, s, t)
print(f"\nImpulse response h(t) = {h}")

# Step response = L^{-1}[H(s)/s]
step_response = inverse_laplace_transform(H/s, s, t)
print(f"\nStep response: {simplify(step_response)}")

# Numerical visualization
import numpy as np
import matplotlib.pyplot as plt

t_vals = np.linspace(0, 6, 300)
h_vals = 0.5 * np.exp(-t_vals) * np.sin(2*t_vals)

# Step response (computed)
step_vals = 0.2 * (1 - np.exp(-t_vals) * (np.cos(2*t_vals) + 0.5*np.sin(2*t_vals)))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_vals, h_vals, 'b-', linewidth=2)
plt.xlabel('t')
plt.ylabel('h(t)')
plt.title('Impulse Response')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t_vals, step_vals, 'r-', linewidth=2)
plt.axhline(y=0.2, color='g', linestyle='--', label='Steady state = 1/5')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Step Response')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_function.png', dpi=150)
plt.show()
```

---

## ✅ Lab Completion Checklist

- [ ] Computed symbolic Laplace transforms
- [ ] Solved ODEs using transform method
- [ ] Visualized impulse and step responses
- [ ] Simulated RLC circuit dynamics
- [ ] Analyzed transfer functions
- [ ] Saved all figures

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Solves a mechanical system using Laplace transforms (compare with numerical)
2. Analyzes the frequency response of an RLC circuit
3. Computes the response to a custom piecewise input
4. Bonus: Implement convolution via Laplace transforms

Save as `Week12_Laplace_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

---

*"Computation reveals what transforms describe—from pole locations to time-domain behavior."*
