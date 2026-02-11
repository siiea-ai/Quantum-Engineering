# Day 76: Computational Lab — Systems of ODEs in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Eigenvalue Methods & Phase Portraits |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Lab Report |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute eigenvalues and eigenvectors in Python
2. Solve systems of ODEs numerically
3. Generate phase portraits programmatically
4. Visualize stability and trajectories
5. Simulate coupled oscillators and other applications

---

## 🖥️ Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig
import sympy as sp
from sympy import Matrix, symbols, exp, cos, sin, simplify

np.set_printoptions(precision=4)
```

---

## 📖 Lab 1: Eigenvalue Analysis

### 1.1 Computing Eigenvalues and Eigenvectors

```python
import numpy as np
from scipy.linalg import eig

# Example matrix
A = np.array([[4, -2],
              [1,  1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)

print("Matrix A:")
print(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors (as columns):")
print(eigenvectors)

# Verify: A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A @ v
    lam_v = lam * v
    print(f"\nλ_{i+1} = {lam:.4f}")
    print(f"A @ v_{i+1} = {Av}")
    print(f"λ_{i+1} * v_{i+1} = {lam_v}")
```

### 1.2 Symbolic Computation with SymPy

```python
from sympy import Matrix, symbols, simplify, sqrt

# Define symbolic matrix
A = Matrix([[4, -2],
            [1,  1]])

# Eigenvalues and eigenvectors
eigendata = A.eigenvects()

print("Symbolic eigenvalue analysis:")
for eigenvalue, multiplicity, eigenvectors in eigendata:
    print(f"\nλ = {eigenvalue} (multiplicity {multiplicity})")
    for v in eigenvectors:
        print(f"  Eigenvector: {v.T}")
```

---

## 📖 Lab 2: Solving Systems Numerically

### 2.1 Basic System Solving

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def linear_system(t, x, A):
    """dx/dt = A @ x"""
    return A @ x

# System matrix
A = np.array([[4, -2],
              [1,  1]])

# Initial condition
x0 = [1, 0]

# Solve
t_span = (0, 3)
t_eval = np.linspace(0, 3, 300)

sol = solve_ivp(
    lambda t, x: linear_system(t, x, A),
    t_span, x0, t_eval=t_eval
)

# Plot time series
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='$x_1(t)$')
axes[0].plot(sol.t, sol.y[1], 'r--', linewidth=2, label='$x_2(t)$')
axes[0].set_xlabel('t')
axes[0].set_ylabel('x')
axes[0].set_title('Time Series')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Phase portrait (single trajectory)
axes[1].plot(sol.y[0], sol.y[1], 'b-', linewidth=2)
axes[1].plot(x0[0], x0[1], 'go', markersize=10, label='Start')
axes[1].plot(sol.y[0, -1], sol.y[1, -1], 'ro', markersize=10, label='End')
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title('Phase Space Trajectory')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('system_solution.png', dpi=150)
plt.show()
```

---

## 📖 Lab 3: Phase Portraits

### 3.1 Full Phase Portrait with Direction Field

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def plot_phase_portrait(A, title="Phase Portrait", xlim=(-3, 3), ylim=(-3, 3)):
    """
    Generate complete phase portrait for dx/dt = Ax
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Direction field
    x1 = np.linspace(xlim[0], xlim[1], 20)
    x2 = np.linspace(ylim[0], ylim[1], 20)
    X1, X2 = np.meshgrid(x1, x2)
    
    U = A[0, 0] * X1 + A[0, 1] * X2
    V = A[1, 0] * X1 + A[1, 1] * X2
    
    # Normalize arrows
    N = np.sqrt(U**2 + V**2)
    N[N == 0] = 1
    U, V = U/N, V/N
    
    ax.quiver(X1, X2, U, V, color='lightblue', alpha=0.6)
    
    # Trajectories from various initial conditions
    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 500)
    
    # Grid of initial conditions
    for x10 in np.linspace(xlim[0]*0.9, xlim[1]*0.9, 7):
        for x20 in np.linspace(ylim[0]*0.9, ylim[1]*0.9, 7):
            if abs(x10) < 0.1 and abs(x20) < 0.1:
                continue
            
            # Forward trajectory
            sol = solve_ivp(
                lambda t, x: A @ x,
                t_span, [x10, x20], t_eval=t_eval,
                max_step=0.05
            )
            ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.8, alpha=0.7)
            
            # Backward trajectory
            sol_back = solve_ivp(
                lambda t, x: -A @ x,
                t_span, [x10, x20], t_eval=t_eval,
                max_step=0.05
            )
            ax.plot(sol_back.y[0], sol_back.y[1], 'b-', linewidth=0.8, alpha=0.7)
    
    # Eigenvector directions
    eigenvalues, eigenvectors = np.linalg.eig(A)
    for i in range(2):
        if np.isreal(eigenvalues[i]):
            v = np.real(eigenvectors[:, i])
            scale = min(abs(xlim[0]), abs(xlim[1])) * 0.9
            ax.plot([-scale*v[0], scale*v[0]], [-scale*v[1], scale*v[1]], 
                   'r--', linewidth=2, label=f'λ={eigenvalues[i]:.2f}')
    
    ax.plot(0, 0, 'ko', markersize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return fig, ax

# Examples of different phase portraits
matrices = {
    'Stable Node': np.array([[-2, 0], [0, -1]]),
    'Unstable Node': np.array([[2, 0], [0, 1]]),
    'Saddle Point': np.array([[1, 0], [0, -1]]),
    'Center': np.array([[0, 1], [-1, 0]]),
    'Stable Spiral': np.array([[-0.5, 1], [-1, -0.5]]),
    'Unstable Spiral': np.array([[0.5, 1], [-1, 0.5]])
}

for name, A in matrices.items():
    eigenvalues = np.linalg.eigvals(A)
    fig, ax = plot_phase_portrait(A, f'{name}\nλ = {eigenvalues}')
    plt.savefig(f'phase_{name.replace(" ", "_").lower()}.png', dpi=150)
    plt.show()
```

---

## 📖 Lab 4: Coupled Oscillators

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def coupled_oscillators(t, y, k1, k2, m1, m2):
    """
    Two masses connected by springs
    y = [x1, v1, x2, v2]
    """
    x1, v1, x2, v2 = y
    
    # Accelerations
    a1 = (-k1*x1 + k2*(x2 - x1)) / m1
    a2 = (-k2*(x2 - x1)) / m2
    
    return [v1, a1, v2, a2]

# Parameters
k1, k2 = 3, 2
m1, m2 = 1, 1

# Initial conditions: displace first mass only
y0 = [1, 0, 0, 0]

# Solve
t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)

sol = solve_ivp(
    lambda t, y: coupled_oscillators(t, y, k1, k2, m1, m2),
    t_span, y0, t_eval=t_eval
)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Positions
axes[0, 0].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='$x_1$ (mass 1)')
axes[0, 0].plot(sol.t, sol.y[2], 'r--', linewidth=2, label='$x_2$ (mass 2)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Position')
axes[0, 0].set_title('Positions vs Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Phase space for mass 1
axes[0, 1].plot(sol.y[0], sol.y[1], 'b-', linewidth=1)
axes[0, 1].set_xlabel('$x_1$')
axes[0, 1].set_ylabel('$v_1$')
axes[0, 1].set_title('Phase Space: Mass 1')
axes[0, 1].grid(True, alpha=0.3)

# Configuration space
axes[1, 0].plot(sol.y[0], sol.y[2], 'g-', linewidth=1)
axes[1, 0].set_xlabel('$x_1$')
axes[1, 0].set_ylabel('$x_2$')
axes[1, 0].set_title('Configuration Space')
axes[1, 0].grid(True, alpha=0.3)

# Energy
KE = 0.5 * m1 * sol.y[1]**2 + 0.5 * m2 * sol.y[3]**2
PE = 0.5 * k1 * sol.y[0]**2 + 0.5 * k2 * (sol.y[2] - sol.y[0])**2
E_total = KE + PE

axes[1, 1].plot(sol.t, KE, 'r-', label='Kinetic', alpha=0.7)
axes[1, 1].plot(sol.t, PE, 'b-', label='Potential', alpha=0.7)
axes[1, 1].plot(sol.t, E_total, 'k--', label='Total', linewidth=2)
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Energy')
axes[1, 1].set_title('Energy Conservation')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coupled_oscillators.png', dpi=150)
plt.show()

# Normal mode analysis
print("\nNormal Mode Analysis:")
# The matrix for coupled oscillators
A_system = np.array([
    [0, 1, 0, 0],
    [-(k1+k2)/m1, 0, k2/m1, 0],
    [0, 0, 0, 1],
    [k2/m2, 0, -k2/m2, 0]
])
eigenvalues = np.linalg.eigvals(A_system)
print(f"Eigenvalues: {eigenvalues}")
print(f"Normal frequencies: {np.abs(np.imag(eigenvalues[eigenvalues.imag > 0]))}")
```

---

## 📖 Lab 5: Predator-Prey Model

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lotka_volterra(t, y, a, b, c, d):
    """
    Predator-prey dynamics
    x = prey, y = predator
    dx/dt = ax - bxy
    dy/dt = -cy + dxy
    """
    x, p = y
    return [a*x - b*x*p, -c*p + d*x*p]

# Parameters
a, b, c, d = 1.0, 0.1, 1.5, 0.075

# Multiple initial conditions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for x0, p0 in [(10, 5), (20, 5), (10, 10), (15, 8)]:
    sol = solve_ivp(
        lambda t, y: lotka_volterra(t, y, a, b, c, d),
        (0, 50), [x0, p0], t_eval=np.linspace(0, 50, 1000)
    )
    
    # Time series
    axes[0].plot(sol.t, sol.y[0], '-', label=f'Prey (x₀={x0})')
    axes[0].plot(sol.t, sol.y[1], '--', label=f'Predator (p₀={p0})')
    
    # Phase portrait
    axes[1].plot(sol.y[0], sol.y[1], '-', linewidth=1)
    axes[1].plot(x0, p0, 'o', markersize=8)

axes[0].set_xlabel('Time')
axes[0].set_ylabel('Population')
axes[0].set_title('Lotka-Volterra: Time Series')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(True, alpha=0.3)

# Equilibrium point
x_eq, p_eq = c/d, a/b
axes[1].plot(x_eq, p_eq, 'r*', markersize=15, label=f'Equilibrium ({x_eq:.1f}, {p_eq:.1f})')
axes[1].set_xlabel('Prey (x)')
axes[1].set_ylabel('Predator (y)')
axes[1].set_title('Lotka-Volterra: Phase Portrait')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predator_prey.png', dpi=150)
plt.show()
```

---

## ✅ Lab Completion Checklist

- [ ] Computed eigenvalues/eigenvectors numerically and symbolically
- [ ] Solved systems of ODEs with solve_ivp
- [ ] Generated phase portraits for all equilibrium types
- [ ] Simulated coupled oscillators
- [ ] Implemented predator-prey model
- [ ] Saved all figures

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Analyzes a 3×3 system (find eigenvalues, solve, visualize)
2. Creates an interactive phase portrait (user inputs matrix)
3. Simulates a physical system of your choice
4. Compares numerical and analytical solutions

Save as `Week11_Systems_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

---

*"Computation transforms abstract systems into visible dynamics—watch the mathematics come alive."*
