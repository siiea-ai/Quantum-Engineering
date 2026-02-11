# Day 55: Computational Lab — Vector Calculus in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Vector Fields Visualization |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Numerical Integration |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Applications |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Visualize 2D and 3D vector fields
2. Compute divergence and curl numerically
3. Evaluate line integrals numerically
4. Verify Green's and Stokes' Theorems computationally
5. Create publication-quality vector field plots

---

## 🖥️ Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from sympy import *
```

---

## 📖 Lab 1: Visualizing 2D Vector Fields

### 1.1 Basic Vector Field Plot

```python
import numpy as np
import matplotlib.pyplot as plt

# Create grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Define vector field F = <-y, x> (rotation)
U = -Y
V = X

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.quiver(X, Y, U, V, color='blue', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Vector Field: F = ⟨-y, x⟩ (Rotation)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.savefig('rotation_field.png', dpi=150)
plt.show()
```

### 1.2 Multiple Vector Fields Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Field 1: Constant
axes[0,0].quiver(X, Y, np.ones_like(X), np.zeros_like(Y))
axes[0,0].set_title('Constant: F = ⟨1, 0⟩')

# Field 2: Radial (source)
R = np.sqrt(X**2 + Y**2) + 0.1
axes[0,1].quiver(X, Y, X/R, Y/R)
axes[0,1].set_title('Radial: F = r̂ (Source)')

# Field 3: Rotation
axes[1,0].quiver(X, Y, -Y, X)
axes[1,0].set_title('Rotation: F = ⟨-y, x⟩')

# Field 4: Saddle
axes[1,1].quiver(X, Y, X, -Y)
axes[1,1].set_title('Saddle: F = ⟨x, -y⟩')

for ax in axes.flat:
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vector_field_gallery.png', dpi=150)
plt.show()
```

### 1.3 Streamlines

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Vector field
U = -Y
V = X

fig, ax = plt.subplots(figsize=(10, 8))

# Streamlines show flow paths
strm = ax.streamplot(X, Y, U, V, density=1.5, color=np.sqrt(U**2 + V**2), 
                      cmap='viridis', linewidth=1.5)
plt.colorbar(strm.lines, label='Speed')

ax.set_title('Streamlines of F = ⟨-y, x⟩')
ax.set_aspect('equal')

plt.savefig('streamlines.png', dpi=150)
plt.show()
```

---

## 📖 Lab 2: Computing Divergence and Curl

### 2.1 Symbolic Computation

```python
from sympy import *

x, y, z = symbols('x y z')

# Define vector field F = <x*y, y*z, x*z>
P, Q, R = x*y, y*z, x*z

# Divergence
div_F = diff(P, x) + diff(Q, y) + diff(R, z)
print(f"div F = {div_F}")

# Curl
curl_F_x = diff(R, y) - diff(Q, z)
curl_F_y = diff(P, z) - diff(R, x)
curl_F_z = diff(Q, x) - diff(P, y)
print(f"curl F = ⟨{curl_F_x}, {curl_F_y}, {curl_F_z}⟩")
```

### 2.2 Numerical Divergence and Curl

```python
import numpy as np

def numerical_div_2d(Fx, Fy, X, Y, dx, dy):
    """Compute divergence of 2D field numerically."""
    dFx_dx = np.gradient(Fx, dx, axis=1)
    dFy_dy = np.gradient(Fy, dy, axis=0)
    return dFx_dx + dFy_dy

def numerical_curl_2d(Fx, Fy, X, Y, dx, dy):
    """Compute 2D curl (z-component) numerically."""
    dFy_dx = np.gradient(Fy, dx, axis=1)
    dFx_dy = np.gradient(Fx, dy, axis=0)
    return dFy_dx - dFx_dy

# Example: F = <-y, x>
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y)

Fx, Fy = -Y, X

div_F = numerical_div_2d(Fx, Fy, X, Y, dx, dy)
curl_F = numerical_curl_2d(Fx, Fy, X, Y, dx, dy)

print(f"Divergence (should be 0): mean = {np.mean(div_F):.6f}")
print(f"Curl (should be 2): mean = {np.mean(curl_F):.6f}")
```

---

## 📖 Lab 3: Line Integrals

### 3.1 Numerical Line Integral

```python
import numpy as np
from scipy import integrate

def line_integral(F, curve, t_range, n_points=1000):
    """
    Compute ∫_C F · dr numerically.
    F: function (x, y) -> (Fx, Fy)
    curve: function t -> (x, y)
    """
    t = np.linspace(t_range[0], t_range[1], n_points)
    dt = t[1] - t[0]
    
    # Get curve points
    r = np.array([curve(ti) for ti in t])
    
    # Get dr/dt using finite differences
    dr = np.gradient(r, dt, axis=0)
    
    # Evaluate F along curve
    F_vals = np.array([F(r[i,0], r[i,1]) for i in range(len(t))])
    
    # Compute F · dr and integrate
    integrand = np.sum(F_vals * dr, axis=1)
    result = np.trapz(integrand, t)
    
    return result

# Example: F = <y, x>, C is unit circle
def F(x, y):
    return np.array([y, x])

def circle(t):
    return np.array([np.cos(t), np.sin(t)])

result = line_integral(F, circle, [0, 2*np.pi])
print(f"∮_C F · dr (should be 0): {result:.6f}")

# Example: F = <-y, x>/(x²+y²), C is unit circle
def F2(x, y):
    r2 = x**2 + y**2
    return np.array([-y/r2, x/r2])

result2 = line_integral(F2, circle, [0, 2*np.pi])
print(f"∮_C F · dr (should be 2π): {result2:.6f}, 2π = {2*np.pi:.6f}")
```

---

## 📖 Lab 4: Verifying Green's Theorem

```python
import numpy as np
from scipy import integrate

# Verify: ∮_C P dx + Q dy = ∬_D (∂Q/∂x - ∂P/∂y) dA
# For F = <xy, x²> over unit disk

# Line integral (around unit circle)
def line_integrand(t):
    x, y = np.cos(t), np.sin(t)
    dx_dt, dy_dt = -np.sin(t), np.cos(t)
    P, Q = x*y, x**2
    return P * dx_dt + Q * dy_dt

line_result, _ = integrate.quad(line_integrand, 0, 2*np.pi)
print(f"Line integral: {line_result:.6f}")

# Double integral
def double_integrand(y, x):
    # ∂Q/∂x - ∂P/∂y = 2x - x = x
    return x

def y_lower(x):
    return -np.sqrt(1 - x**2)

def y_upper(x):
    return np.sqrt(1 - x**2)

double_result, _ = integrate.dblquad(double_integrand, -1, 1, y_lower, y_upper)
print(f"Double integral: {double_result:.6f}")

print(f"\nGreen's Theorem verified: {np.isclose(line_result, double_result)}")
```

---

## 📖 Lab 5: 3D Vector Fields

### 5.1 3D Quiver Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create 3D grid
x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)
z = np.linspace(-1, 1, 5)
X, Y, Z = np.meshgrid(x, y, z)

# Vector field F = <x, y, z> (radial outward)
U, V, W = X, Y, Z

ax.quiver(X, Y, Z, U, V, W, length=0.2, normalize=True, color='blue', alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Vector Field: F = ⟨x, y, z⟩')

plt.savefig('3d_vector_field.png', dpi=150)
plt.show()
```

---

## 📖 Lab 6: Verifying Divergence Theorem

```python
import numpy as np
from scipy import integrate

# Verify: ∬_S F · dS = ∭_E div F dV
# For F = <x, y, z> over unit sphere

# Surface integral (over unit sphere, using spherical coords)
def surface_integrand(phi, theta):
    # On unit sphere: r = 1, so F = <sin φ cos θ, sin φ sin θ, cos φ>
    # Normal n = <sin φ cos θ, sin φ sin θ, cos φ>
    # F · n = sin²φ cos²θ + sin²φ sin²θ + cos²φ = sin²φ + cos²φ = 1
    # dS = sin φ dφ dθ
    return np.sin(phi)  # F · n * |∂r/∂φ × ∂r/∂θ|

surface_result, _ = integrate.dblquad(surface_integrand, 0, 2*np.pi, 0, np.pi)
print(f"Surface integral (flux): {surface_result:.6f}")
print(f"Expected (4π): {4*np.pi:.6f}")

# Volume integral
def volume_integrand(rho, phi, theta):
    # div F = 3
    # dV = ρ² sin φ dρ dφ dθ
    return 3 * rho**2 * np.sin(phi)

volume_result, _ = integrate.tplquad(volume_integrand, 0, 2*np.pi, 0, np.pi, 0, 1)
print(f"Volume integral: {volume_result:.6f}")
print(f"Expected (4π): {4*np.pi:.6f}")

print(f"\nDivergence Theorem verified: {np.isclose(surface_result, volume_result)}")
```

---

## ✅ Lab Completion Checklist

- [ ] Created 2D vector field plots
- [ ] Generated streamline visualizations
- [ ] Computed divergence and curl numerically
- [ ] Evaluated line integrals numerically
- [ ] Verified Green's Theorem
- [ ] Created 3D vector field visualizations
- [ ] Verified Divergence Theorem
- [ ] Saved all figures

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Visualizes 4 different 2D vector fields with streamlines
2. Computes and plots div and curl for each field
3. Numerically verifies Green's Theorem for a chosen field and region
4. Creates a 3D visualization of an interesting vector field

Save as `Week8_VectorCalculus_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Month 2 Conclusion

---

*"Visualization transforms abstract vector fields into tangible, intuitive understanding."*
