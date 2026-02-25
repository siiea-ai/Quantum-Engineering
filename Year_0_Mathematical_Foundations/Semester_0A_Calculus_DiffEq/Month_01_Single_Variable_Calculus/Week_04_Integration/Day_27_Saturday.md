# Day 27: Computational Lab — Integration in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Symbolic Integration |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Numerical Methods |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Visualization |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute integrals symbolically with SymPy
2. Implement Riemann sums in Python
3. Use scipy.integrate for numerical integration
4. Visualize areas under curves
5. Apply integration to physics problems

---

## 🖥️ Setup

```bash
pip install numpy matplotlib sympy scipy
```

---

## 📖 Lab 1: Symbolic Integration with SymPy

### 1.1 Basic Indefinite Integrals

```python
from sympy import *

x = Symbol('x')

# Basic integrals
print("=== Symbolic Integration ===\n")

integrals = [
    x**3,
    sin(x),
    exp(x),
    1/x,
    x * exp(x**2),
    1/(1 + x**2),
    sqrt(1 - x**2)
]

for f in integrals:
    result = integrate(f, x)
    print(f"∫ {f} dx = {result}")
```

### 1.2 Definite Integrals

```python
from sympy import *

x = Symbol('x')

# Definite integrals
print("\n=== Definite Integrals ===\n")

# ∫₀¹ x² dx
result1 = integrate(x**2, (x, 0, 1))
print(f"∫₀¹ x² dx = {result1}")

# ∫₀^π sin(x) dx
result2 = integrate(sin(x), (x, 0, pi))
print(f"∫₀^π sin(x) dx = {result2}")

# ∫₁^e 1/x dx
result3 = integrate(1/x, (x, 1, E))
print(f"∫₁^e 1/x dx = {result3}")

# ∫₋∞^∞ e^(-x²) dx (Gaussian)
result4 = integrate(exp(-x**2), (x, -oo, oo))
print(f"∫₋∞^∞ e^(-x²) dx = {result4}")
```

### 1.3 Substitution Verification

```python
from sympy import *

x, u = symbols('x u')

# Original integral: ∫ 2x(x²+1)³ dx
f = 2*x*(x**2 + 1)**3

# Direct integration
direct = integrate(f, x)
print(f"Direct: ∫ {f} dx = {simplify(direct)}")

# Verify by differentiation
print(f"Check: d/dx[{simplify(direct)}] = {simplify(diff(direct, x))}")
```

---

## 📖 Lab 2: Riemann Sums Implementation

### 2.1 Riemann Sum Functions

```python
import numpy as np
import matplotlib.pyplot as plt

def left_riemann_sum(f, a, b, n):
    """Left Riemann sum approximation of ∫ₐᵇ f(x)dx"""
    dx = (b - a) / n
    x = np.linspace(a, b - dx, n)  # Left endpoints
    return np.sum(f(x)) * dx

def right_riemann_sum(f, a, b, n):
    """Right Riemann sum approximation of ∫ₐᵇ f(x)dx"""
    dx = (b - a) / n
    x = np.linspace(a + dx, b, n)  # Right endpoints
    return np.sum(f(x)) * dx

def midpoint_riemann_sum(f, a, b, n):
    """Midpoint Riemann sum approximation of ∫ₐᵇ f(x)dx"""
    dx = (b - a) / n
    x = np.linspace(a + dx/2, b - dx/2, n)  # Midpoints
    return np.sum(f(x)) * dx

# Test with f(x) = x²
f = lambda x: x**2
a, b = 0, 2
exact = 8/3  # ∫₀² x² dx = 8/3

print("Riemann Sums for ∫₀² x² dx")
print(f"Exact value: {exact:.10f}")
print("-" * 50)

for n in [4, 10, 50, 100, 1000]:
    left = left_riemann_sum(f, a, b, n)
    right = right_riemann_sum(f, a, b, n)
    mid = midpoint_riemann_sum(f, a, b, n)
    
    print(f"n = {n:4d}:")
    print(f"  Left:     {left:.10f}, Error: {abs(left - exact):.2e}")
    print(f"  Right:    {right:.10f}, Error: {abs(right - exact):.2e}")
    print(f"  Midpoint: {mid:.10f}, Error: {abs(mid - exact):.2e}")
```

### 2.2 Visualizing Riemann Sums

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_riemann(f, a, b, n, method='left'):
    """Visualize Riemann sum approximation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot function
    x_smooth = np.linspace(a, b, 1000)
    ax.plot(x_smooth, f(x_smooth), 'b-', linewidth=2, label='f(x)')
    
    dx = (b - a) / n
    
    if method == 'left':
        x_bars = np.linspace(a, b - dx, n)
        heights = f(x_bars)
        title = 'Left Riemann Sum'
    elif method == 'right':
        x_bars = np.linspace(a + dx, b, n)
        heights = f(x_bars)
        x_bars = x_bars - dx  # Shift for bar positioning
        title = 'Right Riemann Sum'
    elif method == 'midpoint':
        x_bars = np.linspace(a, b - dx, n)
        heights = f(x_bars + dx/2)
        title = 'Midpoint Riemann Sum'
    
    # Draw rectangles
    ax.bar(x_bars, heights, width=dx, align='edge', 
           alpha=0.5, edgecolor='red', linewidth=1.5)
    
    area = np.sum(heights) * dx
    ax.set_title(f'{title}, n={n}\nApproximate area = {area:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'riemann_{method}_n{n}.png', dpi=150)
    plt.show()

# Visualize
f = lambda x: x**2
visualize_riemann(f, 0, 2, 4, 'left')
visualize_riemann(f, 0, 2, 4, 'right')
visualize_riemann(f, 0, 2, 10, 'midpoint')
```

---

## 📖 Lab 3: Numerical Integration with SciPy

### 3.1 Using scipy.integrate.quad

```python
from scipy import integrate
import numpy as np

# quad returns (result, error estimate)

# Example 1: ∫₀² x² dx
result1, error1 = integrate.quad(lambda x: x**2, 0, 2)
print(f"∫₀² x² dx = {result1:.10f} ± {error1:.2e}")

# Example 2: ∫₀^π sin(x) dx
result2, error2 = integrate.quad(np.sin, 0, np.pi)
print(f"∫₀^π sin(x) dx = {result2:.10f} ± {error2:.2e}")

# Example 3: Gaussian integral ∫₋∞^∞ e^(-x²) dx
result3, error3 = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(f"∫₋∞^∞ e^(-x²) dx = {result3:.10f} (√π ≈ {np.sqrt(np.pi):.10f})")

# Example 4: Integral that's hard symbolically
# ∫₀¹ sin(x²) dx (Fresnel-like integral)
result4, error4 = integrate.quad(lambda x: np.sin(x**2), 0, 1)
print(f"∫₀¹ sin(x²) dx = {result4:.10f} ± {error4:.2e}")
```

### 3.2 Comparing Numerical Methods

```python
from scipy import integrate
import numpy as np

def f(x):
    return np.exp(-x**2) * np.cos(x)

a, b = 0, 5

# Different methods
result_quad, _ = integrate.quad(f, a, b)
result_romberg = integrate.romberg(f, a, b)

# Fixed samples
x = np.linspace(a, b, 100)
y = f(x)
result_trapz = integrate.trapezoid(y, x)
result_simpson = integrate.simpson(y, x=x)

print(f"Integration of e^(-x²)cos(x) from 0 to 5:")
print(f"  quad:     {result_quad:.10f}")
print(f"  romberg:  {result_romberg:.10f}")
print(f"  trapezoid:{result_trapz:.10f}")
print(f"  simpson:  {result_simpson:.10f}")
```

---

## 📖 Lab 4: Area Visualization

### 4.1 Area Under a Curve

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def plot_area_under_curve(f, a, b, title="Area Under Curve"):
    """Visualize area under f(x) from a to b."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(a - 0.5, b + 0.5, 1000)
    y = f(x)
    
    # Plot curve
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    
    # Fill area
    x_fill = np.linspace(a, b, 1000)
    y_fill = f(x_fill)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color='blue')
    
    # Calculate area
    area, _ = integrate.quad(f, a, b)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=a, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=b, color='r', linestyle='--', alpha=0.5)
    
    ax.set_title(f'{title}\nArea = {area:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('area_under_curve.png', dpi=150)
    plt.show()
    
    return area

# Example: Area under x² from 0 to 2
area = plot_area_under_curve(lambda x: x**2, 0, 2, "f(x) = x²")
print(f"Computed area: {area:.6f}, Exact: {8/3:.6f}")
```

### 4.2 Area Between Curves

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def area_between_curves(f, g, a, b):
    """Calculate area between f(x) and g(x) from a to b."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(a - 0.5, b + 0.5, 1000)
    
    # Plot curves
    ax.plot(x, f(x), 'b-', linewidth=2, label='f(x)')
    ax.plot(x, g(x), 'r-', linewidth=2, label='g(x)')
    
    # Fill area between
    x_fill = np.linspace(a, b, 1000)
    ax.fill_between(x_fill, f(x_fill), g(x_fill), 
                     alpha=0.3, color='purple', label='Area')
    
    # Calculate area
    area, _ = integrate.quad(lambda x: abs(f(x) - g(x)), a, b)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_title(f'Area Between Curves = {area:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('area_between_curves.png', dpi=150)
    plt.show()
    
    return area

# Area between y = x and y = x² from 0 to 1
f = lambda x: x
g = lambda x: x**2
area = area_between_curves(f, g, 0, 1)
print(f"Area between x and x²: {area:.6f}, Exact: {1/6:.6f}")
```

---

## 📖 Lab 5: Physics Applications

### 5.1 Work Done by a Variable Force

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Spring force: F(x) = kx (Hooke's law)
k = 100  # N/m

def spring_force(x):
    return k * x

# Work to stretch spring from 0 to 0.5 m
x_max = 0.5
work, _ = integrate.quad(spring_force, 0, x_max)

print(f"Work to stretch spring {x_max} m: {work:.2f} J")
print(f"Formula check: (1/2)kx² = {0.5 * k * x_max**2:.2f} J")

# Visualize
x = np.linspace(0, x_max, 100)
plt.figure(figsize=(10, 6))
plt.fill_between(x, spring_force(x), alpha=0.3)
plt.plot(x, spring_force(x), 'b-', linewidth=2)
plt.xlabel('Displacement x (m)')
plt.ylabel('Force F (N)')
plt.title(f'Work Done by Spring = {work:.2f} J')
plt.grid(True, alpha=0.3)
plt.savefig('work_spring.png', dpi=150)
plt.show()
```

### 5.2 Probability from a Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Gaussian probability distribution (normalized)
def gaussian(x, mu=0, sigma=1):
    return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

# Probability that X is between -1 and 1 for standard normal
prob, _ = integrate.quad(gaussian, -1, 1)
print(f"P(-1 < X < 1) for N(0,1): {prob:.4f} (≈ 68.27%)")

prob2, _ = integrate.quad(gaussian, -2, 2)
print(f"P(-2 < X < 2) for N(0,1): {prob2:.4f} (≈ 95.45%)")

# Visualize
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, gaussian(x), 'b-', linewidth=2)
plt.fill_between(x[(x >= -1) & (x <= 1)], 
                  gaussian(x[(x >= -1) & (x <= 1)]), 
                  alpha=0.3, label=f'P(-1 < X < 1) = {prob:.2%}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Standard Normal Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('probability_integral.png', dpi=150)
plt.show()
```

---

## ✅ Lab Completion Checklist

- [ ] Completed Lab 1: Symbolic integration
- [ ] Completed Lab 2: Riemann sums
- [ ] Completed Lab 3: Numerical integration
- [ ] Completed Lab 4: Area visualization
- [ ] Completed Lab 5: Physics applications
- [ ] Saved all figures
- [ ] Created at least one original integration visualization

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Symbolically evaluates ∫₀¹ x·e^(-x) dx and verifies numerically
2. Visualizes convergence of Riemann sums as n increases
3. Computes and visualizes the area between y = sin(x) and y = cos(x) from 0 to π/2
4. Applies integration to calculate work or probability in a scenario of your choice

Save as `Week4_Integration_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

Day 28 completes Month 1 with rest and preparation for Month 2 (Multivariable Calculus).

---

*"Computation is the new microscope—it lets us see what was previously invisible."*
