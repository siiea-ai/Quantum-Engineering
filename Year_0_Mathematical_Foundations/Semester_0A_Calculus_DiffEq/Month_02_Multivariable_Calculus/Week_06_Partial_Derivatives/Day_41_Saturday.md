# Day 41: Computational Lab — Partial Derivatives and Visualization

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | 3D Surface Visualization |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Gradient and Level Curves |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Applications |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Visualize 3D surfaces and contour plots in Python
2. Compute partial derivatives symbolically and numerically
3. Visualize gradient vectors on contour plots
4. Implement directional derivative calculations
5. Visualize tangent planes

---

## 🖥️ Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import *

# For interactive 3D plots (optional)
# %matplotlib notebook  # in Jupyter
```

---

## 📖 Lab 1: 3D Surface Visualization

### 1.1 Basic Surface Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create meshgrid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Define function z = f(x, y)
Z = X**2 + Y**2  # Paraboloid

# Create 3D plot
fig = plt.figure(figsize=(12, 5))

# Surface plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Surface: z = x² + y²')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot (Level Curves)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('surface_and_contour.png', dpi=150)
plt.show()
```

### 1.2 Saddle Surface

```python
# Saddle surface: z = x² - y²
Z_saddle = X**2 - Y**2

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_saddle, cmap='coolwarm', alpha=0.8)
ax1.set_title('Saddle: z = x² - y²')

ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z_saddle, levels=20, cmap='coolwarm')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_title('Saddle Contours')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('saddle_surface.png', dpi=150)
plt.show()
```

### 1.3 Multiple Surfaces for Comparison

```python
fig = plt.figure(figsize=(15, 10))

functions = [
    ('z = x² + y²', lambda x, y: x**2 + y**2),
    ('z = sin(x)cos(y)', lambda x, y: np.sin(x) * np.cos(y)),
    ('z = e^(-(x²+y²))', lambda x, y: np.exp(-(x**2 + y**2))),
    ('z = xy', lambda x, y: x * y),
]

for i, (title, func) in enumerate(functions, 1):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    Z = func(X, Y)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('multiple_surfaces.png', dpi=150)
plt.show()
```

---

## 📖 Lab 2: Symbolic Partial Derivatives

### 2.1 Using SymPy for Partial Derivatives

```python
from sympy import *

x, y, z = symbols('x y z')

# Define function
f = x**2 * y + x * y**3 + exp(x*y)

print("Function: f =", f)
print()

# First partial derivatives
f_x = diff(f, x)
f_y = diff(f, y)

print("∂f/∂x =", f_x)
print("∂f/∂y =", f_y)
print()

# Second partial derivatives
f_xx = diff(f, x, 2)
f_yy = diff(f, y, 2)
f_xy = diff(f, x, y)
f_yx = diff(f, y, x)

print("∂²f/∂x² =", f_xx)
print("∂²f/∂y² =", f_yy)
print("∂²f/∂x∂y =", f_xy)
print("∂²f/∂y∂x =", f_yx)
print()

# Verify mixed partials are equal
print("f_xy = f_yx?", simplify(f_xy - f_yx) == 0)
```

### 2.2 Evaluating at a Point

```python
from sympy import *

x, y = symbols('x y')
f = x**2 + x*y + y**2

f_x = diff(f, x)
f_y = diff(f, y)

# Evaluate at (1, 2)
point = {x: 1, y: 2}

print(f"f(1, 2) = {f.subs(point)}")
print(f"f_x(1, 2) = {f_x.subs(point)}")
print(f"f_y(1, 2) = {f_y.subs(point)}")

# Gradient at (1, 2)
gradient = [f_x.subs(point), f_y.subs(point)]
print(f"∇f(1, 2) = {gradient}")
```

### 2.3 Chain Rule Verification

```python
from sympy import *

x, y, s, t = symbols('x y s t')

# z = x*y, where x = s + t, y = s*t
z = x * y
x_expr = s + t
y_expr = s * t

# Method 1: Substitute then differentiate
z_substituted = z.subs([(x, x_expr), (y, y_expr)])
dz_ds_direct = diff(z_substituted, s)

# Method 2: Chain rule
z_x = diff(z, x)
z_y = diff(z, y)
x_s = diff(x_expr, s)
y_s = diff(y_expr, s)

dz_ds_chain = z_x * x_s + z_y * y_s
dz_ds_chain = dz_ds_chain.subs([(x, x_expr), (y, y_expr)])

print("Direct method: ∂z/∂s =", simplify(dz_ds_direct))
print("Chain rule: ∂z/∂s =", simplify(dz_ds_chain))
print("Equal?", simplify(dz_ds_direct - dz_ds_chain) == 0)
```

---

## 📖 Lab 3: Gradient Visualization

### 3.1 Gradient Field on Contour Plot

```python
import numpy as np
import matplotlib.pyplot as plt

# Define function and its gradient
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return 2*x, 2*y  # (df/dx, df/dy)

# Create grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Compute gradient at each point
U, V = grad_f(X, Y)

# Create fine grid for contours
x_fine = np.linspace(-2, 2, 100)
y_fine = np.linspace(-2, 2, 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
Z_fine = f(X_fine, Y_fine)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Contour plot
contour = ax.contour(X_fine, Y_fine, Z_fine, levels=15, cmap='Blues')
ax.clabel(contour, inline=True, fontsize=8)

# Gradient vectors (quiver plot)
ax.quiver(X, Y, U, V, color='red', alpha=0.7, label='∇f')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gradient Field on Contour Plot\nf(x,y) = x² + y²')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('gradient_field.png', dpi=150)
plt.show()
```

### 3.2 Gradient Shows Perpendicularity to Level Curves

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + 4*y**2  # Elliptic paraboloid

def grad_f(x, y):
    return 2*x, 8*y

# Create grid
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Points to show gradient
points = [(1, 0.5), (-1, 0.5), (0, 1), (1.5, 0.25)]

fig, ax = plt.subplots(figsize=(10, 8))

# Contours
contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# Plot gradient at specific points
for px, py in points:
    gx, gy = grad_f(px, py)
    # Normalize for display
    mag = np.sqrt(gx**2 + gy**2)
    gx_norm, gy_norm = gx/mag * 0.3, gy/mag * 0.3
    
    ax.arrow(px, py, gx_norm, gy_norm, head_width=0.05, head_length=0.03, 
             fc='red', ec='red', linewidth=2)
    ax.plot(px, py, 'ko', markersize=8)

ax.set_title('Gradient Vectors are Perpendicular to Level Curves')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

plt.savefig('gradient_perpendicular.png', dpi=150)
plt.show()
```

---

## 📖 Lab 4: Directional Derivatives

### 4.1 Computing Directional Derivatives

```python
import numpy as np

def f(x, y):
    return x**2 * y - y**3

def grad_f(x, y):
    return (2*x*y, x**2 - 3*y**2)

# Point and direction
point = (2, 1)
v = np.array([3, 4])  # Not necessarily unit vector

# Normalize to get unit vector
u = v / np.linalg.norm(v)
print(f"Direction vector v = {v}")
print(f"Unit vector u = {u}")

# Compute gradient at point
grad = np.array(grad_f(*point))
print(f"\n∇f at {point} = {grad}")

# Directional derivative
D_u_f = np.dot(grad, u)
print(f"\nDirectional derivative D_u f = {D_u_f}")

# Maximum rate of change
max_rate = np.linalg.norm(grad)
print(f"Maximum rate of change = {max_rate}")
print(f"Direction of max increase = {grad / max_rate}")
```

### 4.2 Visualizing Directional Derivative

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

# Point of interest
p = np.array([1.0, 1.0])
grad = grad_f(*p)

# Various directions
angles = np.linspace(0, 2*np.pi, 36)
directions = [(np.cos(a), np.sin(a)) for a in angles]

# Compute directional derivatives
dir_derivs = [np.dot(grad, d) for d in directions]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Polar plot of directional derivative
ax1 = plt.subplot(121, projection='polar')
ax1.plot(angles, dir_derivs, 'b-', linewidth=2)
ax1.fill(angles, dir_derivs, alpha=0.3)
ax1.set_title('Directional Derivative vs Direction\nat (1, 1)')

# Right: Contour with gradient
x = np.linspace(-0.5, 2.5, 100)
y = np.linspace(-0.5, 2.5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax2 = axes[1]
contour = ax2.contour(X, Y, Z, levels=15, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)

# Plot point and gradient
ax2.plot(*p, 'ro', markersize=10, label='Point (1,1)')
ax2.arrow(p[0], p[1], grad[0]*0.2, grad[1]*0.2, 
          head_width=0.05, head_length=0.03, fc='red', ec='red', linewidth=2)
ax2.set_title('Gradient at (1,1)')
ax2.set_aspect('equal')
ax2.legend()

plt.tight_layout()
plt.savefig('directional_derivative.png', dpi=150)
plt.show()
```

---

## 📖 Lab 5: Tangent Plane Visualization

### 5.1 Tangent Plane to a Surface

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**2 + y**2

def f_x(x, y):
    return 2*x

def f_y(x, y):
    return 2*y

def tangent_plane(x, y, a, b):
    """Tangent plane to z = f(x,y) at (a, b, f(a,b))"""
    return f(a, b) + f_x(a, b)*(x - a) + f_y(a, b)*(y - b)

# Point of tangency
a, b = 1, 1

# Create grid
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# Surface
Z_surface = f(X, Y)

# Tangent plane
Z_plane = tangent_plane(X, Y, a, b)

# Plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=0.7)
ax1.plot_surface(X, Y, Z_plane, color='red', alpha=0.5)
ax1.scatter([a], [b], [f(a, b)], color='black', s=100, label='Point of tangency')
ax1.set_title(f'Surface and Tangent Plane at ({a}, {b})')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Close-up view
ax2 = fig.add_subplot(122, projection='3d')
x_close = np.linspace(a-1, a+1, 30)
y_close = np.linspace(b-1, b+1, 30)
X_close, Y_close = np.meshgrid(x_close, y_close)
Z_surf_close = f(X_close, Y_close)
Z_plane_close = tangent_plane(X_close, Y_close, a, b)

ax2.plot_surface(X_close, Y_close, Z_surf_close, cmap='viridis', alpha=0.7)
ax2.plot_surface(X_close, Y_close, Z_plane_close, color='red', alpha=0.5)
ax2.scatter([a], [b], [f(a, b)], color='black', s=100)
ax2.set_title('Close-up View')

plt.tight_layout()
plt.savefig('tangent_plane.png', dpi=150)
plt.show()
```

---

## 📖 Lab 6: Linear Approximation Accuracy

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.sqrt(x**2 + y**2)

def L(x, y, a, b):
    """Linearization at (a, b)"""
    f_val = np.sqrt(a**2 + b**2)
    f_x = a / f_val
    f_y = b / f_val
    return f_val + f_x*(x - a) + f_y*(y - b)

# Point of linearization
a, b = 3, 4  # f(3,4) = 5

# Test points at various distances
distances = np.linspace(0, 1, 50)
errors = []

for d in distances:
    # Points at distance d from (3, 4)
    x_test = a + d
    y_test = b
    
    actual = f(x_test, y_test)
    approx = L(x_test, y_test, a, b)
    error = abs(actual - approx)
    errors.append(error)

plt.figure(figsize=(10, 6))
plt.plot(distances, errors, 'b-', linewidth=2)
plt.xlabel('Distance from linearization point')
plt.ylabel('Absolute error')
plt.title('Linear Approximation Error vs Distance')
plt.grid(True, alpha=0.3)
plt.savefig('linearization_error.png', dpi=150)
plt.show()

print(f"At distance 0.1: error = {errors[5]:.6f}")
print(f"At distance 0.5: error = {errors[25]:.6f}")
print(f"At distance 1.0: error = {errors[49]:.6f}")
```

---

## ✅ Lab Completion Checklist

- [ ] Created 3D surface plots
- [ ] Generated contour plots
- [ ] Computed partial derivatives symbolically
- [ ] Visualized gradient fields
- [ ] Computed directional derivatives
- [ ] Plotted tangent planes
- [ ] Analyzed linearization accuracy
- [ ] Saved all figures

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Visualizes the surface z = sin(x)cos(y) and its contours
2. Computes and displays the gradient field
3. Finds and visualizes the tangent plane at (π/4, π/4)
4. Explores how linear approximation error changes with distance

Save as `Week6_PartialDerivatives_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

Day 42 is a rest day to consolidate Week 6 concepts.

---

*"Visualization is the bridge between abstract mathematics and intuitive understanding."*
