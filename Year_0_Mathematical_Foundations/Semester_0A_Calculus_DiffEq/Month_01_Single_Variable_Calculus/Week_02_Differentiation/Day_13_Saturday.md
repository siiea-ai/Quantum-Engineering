# Day 13: Computational Lab — Derivatives in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Symbolic Differentiation |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Numerical Methods |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Visualization |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute derivatives symbolically using SymPy
2. Implement numerical differentiation
3. Visualize functions and their derivatives
4. Understand the relationship between numerical and analytical derivatives
5. Apply derivatives to optimization problems

---

## 🖥️ Setup

Ensure your environment has:
```bash
pip install numpy matplotlib sympy scipy
```

---

## 📖 Lab 1: Symbolic Differentiation with SymPy

### 1.1 Basic Symbolic Derivatives

```python
from sympy import *

# Define symbolic variable
x = Symbol('x')

# Define functions
f1 = x**3 - 2*x**2 + 4*x - 7
f2 = sin(x) * exp(x)
f3 = ln(x**2 + 1)
f4 = sqrt(x**2 + 4)

# Compute derivatives
print("=== Symbolic Differentiation ===\n")

print(f"f1(x) = {f1}")
print(f"f1'(x) = {diff(f1, x)}")
print(f"f1''(x) = {diff(f1, x, 2)}\n")

print(f"f2(x) = {f2}")
print(f"f2'(x) = {simplify(diff(f2, x))}\n")

print(f"f3(x) = {f3}")
print(f"f3'(x) = {simplify(diff(f3, x))}\n")

print(f"f4(x) = {f4}")
print(f"f4'(x) = {simplify(diff(f4, x))}")
```

### 1.2 Chain Rule Verification

```python
from sympy import *

x = Symbol('x')

# Composite function: sin(x^2)
outer = sin
inner = x**2

# Manual chain rule
manual = cos(x**2) * 2*x

# SymPy's result
sympy_result = diff(sin(x**2), x)

print("Chain Rule Verification")
print(f"f(x) = sin(x²)")
print(f"Manual: {manual}")
print(f"SymPy:  {sympy_result}")
print(f"Equal:  {simplify(manual - sympy_result) == 0}")
```

### 1.3 Implicit Differentiation

```python
from sympy import *

x, y = symbols('x y')

# Circle: x^2 + y^2 = 25
equation = x**2 + y**2 - 25

# Implicit differentiation
# d/dx(x^2 + y^2) = d/dx(25)
# 2x + 2y(dy/dx) = 0
# dy/dx = -x/y

dydx = -diff(equation, x) / diff(equation, y)
print(f"For x² + y² = 25:")
print(f"dy/dx = {dydx}")

# Folium of Descartes: x^3 + y^3 = 6xy
folium = x**3 + y**3 - 6*x*y
dydx_folium = -diff(folium, x) / diff(folium, y)
print(f"\nFor x³ + y³ = 6xy:")
print(f"dy/dx = {simplify(dydx_folium)}")
```

### 1.4 Higher Derivatives and Taylor Series

```python
from sympy import *

x = Symbol('x')

# Function
f = sin(x)

# Compute derivatives up to order 6
print("Derivatives of sin(x):")
for n in range(7):
    deriv = diff(f, x, n)
    print(f"f^({n})(x) = {deriv}")

# Taylor series
taylor = series(sin(x), x, 0, 10)
print(f"\nTaylor series of sin(x) around x=0:")
print(taylor)
```

---

## 📖 Lab 2: Numerical Differentiation

### 2.1 Forward Difference

```python
import numpy as np

def forward_difference(f, x, h=1e-5):
    """
    Approximate f'(x) using forward difference.
    f'(x) ≈ (f(x+h) - f(x)) / h
    """
    return (f(x + h) - f(x)) / h

# Test with f(x) = x^2 at x = 3
f = lambda x: x**2
x = 3

numerical = forward_difference(f, x)
analytical = 2 * x  # We know f'(x) = 2x

print("Forward Difference Method")
print(f"f(x) = x²")
print(f"At x = {x}:")
print(f"Numerical:  {numerical:.10f}")
print(f"Analytical: {analytical}")
print(f"Error:      {abs(numerical - analytical):.2e}")
```

### 2.2 Central Difference (More Accurate)

```python
import numpy as np

def central_difference(f, x, h=1e-5):
    """
    Approximate f'(x) using central difference.
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    More accurate than forward difference (O(h²) vs O(h))
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# Compare methods
f = lambda x: np.sin(x)
x = np.pi / 4
analytical = np.cos(x)

print("Comparison of Methods at x = π/4 for f(x) = sin(x)")
print(f"Analytical: {analytical:.15f}\n")

for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    fwd = forward_difference(f, x, h)
    ctr = central_difference(f, x, h)
    print(f"h = {h}")
    print(f"  Forward:  {fwd:.15f}, Error: {abs(fwd - analytical):.2e}")
    print(f"  Central:  {ctr:.15f}, Error: {abs(ctr - analytical):.2e}")
```

### 2.3 Second Derivative

```python
import numpy as np

def second_derivative(f, x, h=1e-4):
    """
    Approximate f''(x) using central difference.
    f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    """
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

# Test with f(x) = x^3
f = lambda x: x**3
x = 2
analytical = 6 * x  # f''(x) = 6x

numerical = second_derivative(f, x)
print(f"Second Derivative of f(x) = x³ at x = {x}")
print(f"Numerical:  {numerical:.10f}")
print(f"Analytical: {analytical}")
```

---

## 📖 Lab 3: Visualization

### 3.1 Function and Derivative Plots

```python
import numpy as np
import matplotlib.pyplot as plt

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Function 1: Polynomial
x = np.linspace(-2, 3, 1000)
f1 = x**3 - 3*x
f1_prime = 3*x**2 - 3

axes[0, 0].plot(x, f1, 'b-', label='f(x) = x³ - 3x', linewidth=2)
axes[0, 0].plot(x, f1_prime, 'r--', label="f'(x) = 3x² - 3", linewidth=2)
axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 0].set_title('Polynomial')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(-5, 5)

# Function 2: Trigonometric
x = np.linspace(0, 4*np.pi, 1000)
f2 = np.sin(x)
f2_prime = np.cos(x)

axes[0, 1].plot(x, f2, 'b-', label='f(x) = sin(x)', linewidth=2)
axes[0, 1].plot(x, f2_prime, 'r--', label="f'(x) = cos(x)", linewidth=2)
axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 1].set_title('Trigonometric')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Function 3: Exponential
x = np.linspace(-2, 2, 1000)
f3 = np.exp(x)
f3_prime = np.exp(x)  # Special: derivative equals function!

axes[1, 0].plot(x, f3, 'b-', label='f(x) = eˣ', linewidth=2)
axes[1, 0].plot(x, f3_prime, 'r--', label="f'(x) = eˣ", linewidth=2)
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].set_title('Exponential (derivative = function!)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 8)

# Function 4: Chain rule example
x = np.linspace(-2, 2, 1000)
f4 = np.sin(x**2)
f4_prime = 2*x*np.cos(x**2)

axes[1, 1].plot(x, f4, 'b-', label='f(x) = sin(x²)', linewidth=2)
axes[1, 1].plot(x, f4_prime, 'r--', label="f'(x) = 2x·cos(x²)", linewidth=2)
axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 1].set_title('Chain Rule Example')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('derivatives_visualization.png', dpi=150)
plt.show()
```

### 3.2 Tangent Line Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_tangent_line(f, f_prime, a, x_range, title):
    """
    Plot function with tangent line at x = a.
    """
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = f(x)
    
    # Tangent line: y - f(a) = f'(a)(x - a)
    slope = f_prime(a)
    tangent_y = f(a) + slope * (x - a)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='f(x)', linewidth=2)
    plt.plot(x, tangent_y, 'r--', label=f"Tangent at x={a}", linewidth=2)
    plt.plot(a, f(a), 'ko', markersize=10, label=f'Point ({a}, {f(a):.2f})')
    
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{title}\nTangent slope = {slope:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.savefig(f'tangent_{title.replace(" ", "_")}.png', dpi=150)
    plt.show()

# Example: y = x³ - 2x² + 1 at x = 2
f = lambda x: x**3 - 2*x**2 + 1
f_prime = lambda x: 3*x**2 - 4*x
plot_tangent_line(f, f_prime, 2, (-1, 3), "y = x³ - 2x² + 1")
```

### 3.3 Numerical vs Analytical Derivative

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_derivatives(f, f_prime_analytical, x_range, title):
    """
    Compare numerical and analytical derivatives.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    
    # Analytical
    analytical = f_prime_analytical(x)
    
    # Numerical (central difference)
    h = 0.01
    numerical = (f(x + h) - f(x - h)) / (2 * h)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Compare derivatives
    axes[0].plot(x, analytical, 'b-', label='Analytical', linewidth=2)
    axes[0].plot(x, numerical, 'r--', label='Numerical', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel("f'(x)")
    axes[0].set_title(f'Derivative Comparison: {title}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bottom: Error
    error = np.abs(numerical - analytical)
    axes[1].semilogy(x, error + 1e-16, 'g-', linewidth=2)  # Add small value to avoid log(0)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|Error|')
    axes[1].set_title('Absolute Error (log scale)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{title.replace(" ", "_")}.png', dpi=150)
    plt.show()

# Test with sin(x)
f = lambda x: np.sin(x)
f_prime = lambda x: np.cos(x)
compare_derivatives(f, f_prime, (0, 4*np.pi), "sin(x)")
```

---

## 📖 Lab 4: Optimization Application

### 4.1 Finding Critical Points

```python
from sympy import *

x = Symbol('x')

# Function: f(x) = x³ - 6x² + 9x + 1
f = x**3 - 6*x**2 + 9*x + 1

# First derivative
f_prime = diff(f, x)
print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")

# Find critical points (where f'(x) = 0)
critical_points = solve(f_prime, x)
print(f"Critical points: {critical_points}")

# Second derivative test
f_double_prime = diff(f, x, 2)
print(f"f''(x) = {f_double_prime}")

for cp in critical_points:
    second_deriv_value = f_double_prime.subs(x, cp)
    if second_deriv_value > 0:
        print(f"At x = {cp}: f''({cp}) = {second_deriv_value} > 0 → Local MINIMUM")
    elif second_deriv_value < 0:
        print(f"At x = {cp}: f''({cp}) = {second_deriv_value} < 0 → Local MAXIMUM")
    else:
        print(f"At x = {cp}: f''({cp}) = {second_deriv_value} = 0 → Inconclusive")
```

### 4.2 Optimization Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Function from above
f = lambda x: x**3 - 6*x**2 + 9*x + 1
f_prime = lambda x: 3*x**2 - 12*x + 9

x = np.linspace(-1, 5, 1000)
y = f(x)
y_prime = f_prime(x)

fig, ax = plt.subplots(figsize=(12, 6))

# Plot function
ax.plot(x, y, 'b-', label='f(x) = x³ - 6x² + 9x + 1', linewidth=2)
ax.plot(x, y_prime, 'r--', label="f'(x) = 3x² - 12x + 9", linewidth=2)

# Mark critical points
ax.plot(1, f(1), 'go', markersize=12, label=f'Max at x=1, f(1)={f(1)}')
ax.plot(3, f(3), 'ro', markersize=12, label=f'Min at x=3, f(3)={f(3)}')

ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Finding Local Extrema Using Derivatives', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_example.png', dpi=150)
plt.show()
```

---

## ✅ Lab Completion Checklist

- [ ] Completed Lab 1: Symbolic differentiation
- [ ] Completed Lab 2: Numerical methods
- [ ] Completed Lab 3: Visualizations
- [ ] Completed Lab 4: Optimization
- [ ] Saved all figures
- [ ] Experimented with additional functions

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Symbolically differentiates f(x) = x·e^(-x²) and plots both f and f'
2. Compares forward and central difference for f(x) = tan(x) at x = π/4
3. Finds and classifies all critical points of g(x) = x⁴ - 4x³ + 4x²
4. Creates a visualization showing the tangent lines at multiple points on y = sin(x)

Save as `Week2_Derivatives_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

Day 14 is a lighter day for consolidation before Week 3 (Applications of Derivatives).

---

*"The computer is incredibly fast, accurate, and stupid. Man is unbelievably slow, inaccurate, and brilliant. The marriage of the two is a force beyond calculation."*
— Leo Cherne
