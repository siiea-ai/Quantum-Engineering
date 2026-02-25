# Day 62: Computational Lab — First-Order ODEs in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Symbolic & Numerical Solutions |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Visualization & Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Lab Report |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Solve ODEs symbolically with SymPy
2. Solve ODEs numerically with SciPy
3. Visualize solutions and direction fields
4. Compare analytical and numerical solutions
5. Implement ODE models for physical applications

---

## 🖥️ Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from sympy import *

# For nice symbolic output
init_printing()
```

---

## 📖 Lab 1: Symbolic Solutions with SymPy

### 1.1 Basic ODE Solving

```python
from sympy import symbols, Function, dsolve, Eq, exp, sin, cos

# Define symbols
x = symbols('x')
y = Function('y')

# Example 1: y' = 2y
ode1 = Eq(y(x).diff(x), 2*y(x))
sol1 = dsolve(ode1, y(x))
print("y' = 2y:")
print(f"Solution: {sol1}")

# Example 2: y' + y = exp(x)
ode2 = Eq(y(x).diff(x) + y(x), exp(x))
sol2 = dsolve(ode2, y(x))
print("\ny' + y = e^x:")
print(f"Solution: {sol2}")

# Example 3: Separable equation y' = xy
ode3 = Eq(y(x).diff(x), x*y(x))
sol3 = dsolve(ode3, y(x))
print("\ny' = xy:")
print(f"Solution: {sol3}")
```

### 1.2 Initial Value Problems

```python
from sympy import symbols, Function, dsolve, Eq, exp, Derivative

x = symbols('x')
y = Function('y')

# Solve y' + 2y = 6, y(0) = 1
ode = Eq(y(x).diff(x) + 2*y(x), 6)

# General solution
general_sol = dsolve(ode, y(x))
print(f"General solution: {general_sol}")

# Apply initial condition
C1 = symbols('C1')
particular_sol = dsolve(ode, y(x), ics={y(0): 1})
print(f"Particular solution with y(0)=1: {particular_sol}")
```

### 1.3 Classifying and Solving Different Types

```python
from sympy import classify_ode

x = symbols('x')
y = Function('y')

# Test different equations
equations = [
    Eq(y(x).diff(x), x*y(x)),           # Separable
    Eq(y(x).diff(x) + y(x)/x, x**2),    # Linear
    Eq(y(x).diff(x), (x + y(x))/(x - y(x))),  # Homogeneous
]

for i, ode in enumerate(equations):
    print(f"\nEquation {i+1}: {ode}")
    print(f"Classification: {classify_ode(ode, y(x))}")
    try:
        sol = dsolve(ode, y(x))
        print(f"Solution: {sol}")
    except:
        print("Solution method not found")
```

---

## 📖 Lab 2: Numerical Solutions with SciPy

### 2.1 Using odeint

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE: dy/dt = f(y, t)
def exponential_growth(y, t, k):
    return k * y

# Parameters
k = 0.5
y0 = 1.0
t = np.linspace(0, 10, 100)

# Solve
y = odeint(exponential_growth, y0, t, args=(k,))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, y, 'b-', linewidth=2, label='Numerical')
plt.plot(t, y0*np.exp(k*t), 'r--', linewidth=2, label='Analytical: $y_0 e^{kt}$')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Exponential Growth: $dy/dt = ky$')
plt.legend()
plt.grid(True)
plt.savefig('exponential_growth.png', dpi=150)
plt.show()
```

### 2.2 Using solve_ivp (Modern API)

```python
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Define ODE: dy/dt = f(t, y) - note argument order!
def logistic(t, y, r, K):
    return r * y * (1 - y/K)

# Parameters
r, K = 0.5, 100
y0 = [10]  # Initial condition as list
t_span = (0, 30)
t_eval = np.linspace(0, 30, 200)

# Solve
sol = solve_ivp(logistic, t_span, y0, args=(r, K), t_eval=t_eval, dense_output=True)

# Analytical solution for comparison
def logistic_analytical(t, y0, r, K):
    return K / (1 + ((K - y0)/y0) * np.exp(-r*t))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Numerical (solve_ivp)')
plt.plot(t_eval, logistic_analytical(t_eval, 10, r, K), 'r--', linewidth=2, label='Analytical')
plt.axhline(y=K, color='g', linestyle=':', label=f'Carrying capacity K={K}')
plt.xlabel('t')
plt.ylabel('P(t)')
plt.title('Logistic Growth')
plt.legend()
plt.grid(True)
plt.savefig('logistic_growth.png', dpi=150)
plt.show()
```

---

## 📖 Lab 3: Direction Fields and Phase Portraits

### 3.1 Direction Field (Slope Field)

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_direction_field(f, x_range, y_range, title="Direction Field"):
    """
    Plot direction field for dy/dx = f(x, y)
    """
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)
    X, Y = np.meshgrid(x, y)
    
    # Compute slopes
    DY = f(X, Y)
    DX = np.ones_like(DY)
    
    # Normalize for uniform arrow length
    N = np.sqrt(DX**2 + DY**2)
    DX, DY = DX/N, DY/N
    
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, DX, DY, color='blue', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt.gca()

# Example: dy/dx = x - y
f = lambda x, y: x - y

ax = plot_direction_field(f, [-3, 3], [-3, 3], r"Direction Field: $\frac{dy}{dx} = x - y$")

# Add some solution curves
from scipy.integrate import odeint
t = np.linspace(-3, 3, 100)
for y0 in [-2, -1, 0, 1, 2]:
    sol = odeint(lambda y, x: x - y, y0, t)
    plt.plot(t, sol, 'r-', linewidth=1.5)

plt.savefig('direction_field.png', dpi=150)
plt.show()
```

### 3.2 Multiple Initial Conditions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def ode_func(t, y):
    return -0.5 * y + np.sin(t)

t_span = (0, 20)
t_eval = np.linspace(0, 20, 500)

plt.figure(figsize=(12, 8))

# Solve for multiple initial conditions
initial_conditions = [-3, -2, -1, 0, 1, 2, 3]
colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))

for y0, color in zip(initial_conditions, colors):
    sol = solve_ivp(ode_func, t_span, [y0], t_eval=t_eval)
    plt.plot(sol.t, sol.y[0], color=color, linewidth=2, label=f'y(0) = {y0}')

plt.xlabel('t', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title(r"Solutions of $y' = -0.5y + \sin(t)$ for various initial conditions", fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig('multiple_ics.png', dpi=150)
plt.show()
```

---

## 📖 Lab 4: Physical Applications

### 4.1 Newton's Law of Cooling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def cooling(t, T, k, T_ambient):
    return -k * (T - T_ambient)

# Parameters
T_ambient = 20  # Room temperature
T0 = 95         # Initial temperature (hot coffee)
k = 0.1         # Cooling constant

t_span = (0, 60)
t_eval = np.linspace(0, 60, 200)

sol = solve_ivp(cooling, t_span, [T0], args=(k, T_ambient), t_eval=t_eval)

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Coffee temperature')
plt.axhline(y=T_ambient, color='r', linestyle='--', label=f'Room temp = {T_ambient}°C')
plt.axhline(y=60, color='g', linestyle=':', alpha=0.7, label='Drinkable (60°C)')

# Find when T = 60
idx = np.argmin(np.abs(sol.y[0] - 60))
plt.plot(sol.t[idx], 60, 'go', markersize=10)
plt.annotate(f't ≈ {sol.t[idx]:.1f} min', (sol.t[idx], 60), 
             textcoords="offset points", xytext=(10, 10), fontsize=10)

plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title("Newton's Law of Cooling: Coffee Cooling")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cooling.png', dpi=150)
plt.show()
```

### 4.2 RC Circuit

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rc_circuit(t, Q, R, C, E_func):
    """dQ/dt = (E(t) - Q/C) / R"""
    return (E_func(t) - Q/C) / R

# Parameters
R = 1000   # Ohms
C = 0.001  # Farads
tau = R * C  # Time constant

# Different input voltages
def constant_voltage(t): return 5
def step_voltage(t): return 5 if t >= 1 else 0
def sine_voltage(t): return 5 * np.sin(2 * np.pi * t / 10)

t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)
Q0 = [0]

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

for ax, E_func, title in zip(axes, 
    [constant_voltage, step_voltage, sine_voltage],
    ['Constant 5V', 'Step from 0 to 5V at t=1', 'Sinusoidal 5sin(2πt/10)']):
    
    sol = solve_ivp(rc_circuit, t_span, Q0, args=(R, C, E_func), t_eval=t_eval)
    
    # Plot charge and current
    Q = sol.y[0]
    I = np.gradient(Q, t_eval)
    V_C = Q / C
    
    ax.plot(sol.t, V_C, 'b-', linewidth=2, label='Capacitor Voltage')
    ax.plot(sol.t, [E_func(t) for t in sol.t], 'r--', linewidth=1, label='Input Voltage')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'RC Circuit Response: {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rc_circuit.png', dpi=150)
plt.show()
```

### 4.3 Mixing Problem

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def mixing(t, x, rate_in, conc_in, rate_out, V0, rate_change):
    """
    dx/dt = rate_in * conc_in - rate_out * x / V(t)
    V(t) = V0 + (rate_in - rate_out) * t
    """
    V = V0 + rate_change * t
    return rate_in * conc_in - rate_out * x / V

# Parameters
V0 = 100          # Initial volume (gallons)
rate_in = 3       # Inflow rate (gal/min)
conc_in = 2       # Concentration of incoming brine (lb/gal)
rate_out = 2      # Outflow rate (gal/min)
rate_change = rate_in - rate_out
x0 = [10]         # Initial salt (lbs)

# Solve until tank is full (200 gallons)
t_max = (200 - V0) / rate_change if rate_change > 0 else 100
t_span = (0, t_max)
t_eval = np.linspace(0, t_max, 200)

sol = solve_ivp(mixing, t_span, x0, 
                args=(rate_in, conc_in, rate_out, V0, rate_change), 
                t_eval=t_eval)

# Calculate volume and concentration over time
V = V0 + rate_change * sol.t
conc = sol.y[0] / V

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

axes[0].plot(sol.t, V, 'b-', linewidth=2)
axes[0].set_ylabel('Volume (gallons)')
axes[0].set_title('Tank Volume')
axes[0].grid(True, alpha=0.3)

axes[1].plot(sol.t, sol.y[0], 'r-', linewidth=2)
axes[1].set_ylabel('Salt (lbs)')
axes[1].set_title('Amount of Salt')
axes[1].grid(True, alpha=0.3)

axes[2].plot(sol.t, conc, 'g-', linewidth=2)
axes[2].axhline(y=conc_in, color='k', linestyle='--', label=f'Input conc = {conc_in} lb/gal')
axes[2].set_xlabel('Time (minutes)')
axes[2].set_ylabel('Concentration (lb/gal)')
axes[2].set_title('Salt Concentration')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mixing_problem.png', dpi=150)
plt.show()

print(f"Final salt amount: {sol.y[0][-1]:.2f} lbs")
print(f"Final concentration: {conc[-1]:.3f} lb/gal")
```

---

## 📖 Lab 5: Comparing Methods

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Compare different numerical methods
def stiff_ode(t, y):
    return -1000 * y + 3000 - 2000 * np.exp(-t)

t_span = (0, 0.1)
y0 = [0]

methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

plt.figure(figsize=(12, 8))

for method in methods:
    try:
        sol = solve_ivp(stiff_ode, t_span, y0, method=method, dense_output=True)
        t_plot = np.linspace(0, 0.1, 200)
        plt.plot(t_plot, sol.sol(t_plot)[0], label=f'{method} ({len(sol.t)} steps)')
    except:
        print(f"Method {method} failed")

plt.xlabel('t')
plt.ylabel('y')
plt.title('Stiff ODE: Comparison of Numerical Methods')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('method_comparison.png', dpi=150)
plt.show()
```

---

## ✅ Lab Completion Checklist

- [ ] Solved ODEs symbolically with SymPy
- [ ] Solved ODEs numerically with odeint and solve_ivp
- [ ] Created direction fields
- [ ] Visualized multiple solution curves
- [ ] Implemented cooling simulation
- [ ] Implemented RC circuit simulation
- [ ] Implemented mixing problem simulation
- [ ] Compared numerical methods
- [ ] Saved all figures

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Solves the logistic equation both symbolically and numerically
2. Creates a direction field for a nonlinear ODE
3. Models a real-world application (choose one):
   - Population dynamics of a species
   - Cooling of a building
   - Drug concentration in bloodstream
4. Compares numerical accuracy for different step sizes

Save as `Week9_FirstOrderODEs_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

---

*"Computation transforms differential equations from abstract theory to practical engineering tools."*
