# Day 48: Computational Lab — Multiple Integrals in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Double Integrals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Triple Integrals & Visualization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Applications |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute double integrals numerically with SciPy
2. Compute triple integrals numerically
3. Visualize regions of integration
4. Apply numerical integration to physics problems
5. Verify analytical results computationally

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

## 📖 Lab 1: Double Integrals with SciPy

### 1.1 Basic Double Integration

```python
from scipy import integrate
import numpy as np

# ∬_R (x² + y²) dA over R = [0,1] × [0,1]
def f(y, x):  # Note: scipy.dblquad uses f(y, x) order!
    return x**2 + y**2

# Integrate
result, error = integrate.dblquad(f, 0, 1, 0, 1)
# dblquad(f, x_low, x_high, y_low, y_high)

print(f"∬ (x² + y²) dA = {result:.6f}")
print(f"Analytical: 2/3 = {2/3:.6f}")
```

### 1.2 Variable Limits of Integration

```python
from scipy import integrate
import numpy as np

# ∬_D xy dA where D is bounded by y = x² and y = x
def f(y, x):
    return x * y

def y_lower(x):
    return x**2

def y_upper(x):
    return x

result, error = integrate.dblquad(f, 0, 1, y_lower, y_upper)
print(f"∬_D xy dA = {result:.6f}")
print(f"Analytical: 1/24 = {1/24:.6f}")
```

### 1.3 Polar Coordinates (Manual Conversion)

```python
from scipy import integrate
import numpy as np

# ∬_D (x² + y²) dA where D is unit disk
# In polar: ∫₀^{2π} ∫₀^1 r² · r dr dθ

def f_polar(r, theta):
    return r**3  # r² · r (Jacobian)

result, error = integrate.dblquad(f_polar, 0, 2*np.pi, 0, 1)
print(f"∬ (x² + y²) dA over unit disk = {result:.6f}")
print(f"Analytical: π/2 = {np.pi/2:.6f}")
```

---

## 📖 Lab 2: Visualizing Integration Regions

### 2.1 Type I Region

```python
import numpy as np
import matplotlib.pyplot as plt

# Region between y = x² and y = x
x = np.linspace(0, 1, 100)
y1 = x**2
y2 = x

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y1, 'b-', linewidth=2, label='y = x²')
ax.plot(x, y2, 'r-', linewidth=2, label='y = x')
ax.fill_between(x, y1, y2, alpha=0.3, color='green', label='Region D')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Type I Region: x² ≤ y ≤ x')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.savefig('type1_region.png', dpi=150)
plt.show()
```

### 2.2 Polar Region

```python
import numpy as np
import matplotlib.pyplot as plt

# Polar region: annulus 1 ≤ r ≤ 2
theta = np.linspace(0, 2*np.pi, 100)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Fill the annulus
r_inner = np.ones_like(theta)
r_outer = 2 * np.ones_like(theta)

ax.fill_between(theta, r_inner, r_outer, alpha=0.3, color='blue')
ax.plot(theta, r_inner, 'b-', linewidth=2)
ax.plot(theta, r_outer, 'b-', linewidth=2)

ax.set_title('Annular Region: 1 ≤ r ≤ 2')

plt.savefig('polar_region.png', dpi=150)
plt.show()
```

---

## 📖 Lab 3: Triple Integrals

### 3.1 Triple Integration with tplquad

```python
from scipy import integrate

# ∭_B xyz dV where B = [0,1] × [0,2] × [0,3]
def f(z, y, x):  # Order: innermost to outermost
    return x * y * z

result, error = integrate.tplquad(f, 0, 1, 0, 2, 0, 3)
# tplquad(f, x_low, x_high, y_low, y_high, z_low, z_high)

print(f"∭_B xyz dV = {result:.6f}")
print(f"Analytical: 9/2 = {4.5:.6f}")
```

### 3.2 Variable Limits in 3D

```python
from scipy import integrate

# Volume of tetrahedron: x + y + z ≤ 1, x,y,z ≥ 0
# V = ∭ 1 dV

def f(z, y, x):
    return 1

def z_upper(y, x):
    return 1 - x - y

def y_upper(x):
    return 1 - x

result, error = integrate.tplquad(f, 0, 1, lambda x: 0, y_upper, 
                                   lambda x, y: 0, z_upper)

print(f"Volume of tetrahedron = {result:.6f}")
print(f"Analytical: 1/6 = {1/6:.6f}")
```

---

## 📖 Lab 4: 3D Visualization

### 4.1 Volume Under a Surface

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Visualize z = 4 - x² - y² over unit disk
fig = plt.figure(figsize=(12, 5))

# Surface plot
ax1 = fig.add_subplot(121, projection='3d')
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
Z = 4 - X**2 - Y**2

# Mask outside unit circle
mask = X**2 + Y**2 > 1
Z[mask] = np.nan

ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('z = 4 - x² - y² over unit disk')

# Compute volume
from scipy import integrate

def f(y, x):
    return 4 - x**2 - y**2

def y_lower(x):
    return -np.sqrt(1 - x**2)

def y_upper(x):
    return np.sqrt(1 - x**2)

volume, _ = integrate.dblquad(f, -1, 1, y_lower, y_upper)

ax2 = fig.add_subplot(122)
ax2.text(0.5, 0.5, f'Volume = {volume:.4f}\n(Analytical: 7π/2 ≈ {7*np.pi/2:.4f})', 
         transform=ax2.transAxes, fontsize=16, ha='center', va='center')
ax2.axis('off')

plt.tight_layout()
plt.savefig('volume_visualization.png', dpi=150)
plt.show()
```

### 4.2 Solid Region in 3D

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Vertices
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Faces
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

faces = [
    [vertices[0], vertices[1], vertices[2]],  # bottom
    [vertices[0], vertices[1], vertices[3]],  # front
    [vertices[0], vertices[2], vertices[3]],  # left
    [vertices[1], vertices[2], vertices[3]]   # diagonal
]

collection = Poly3DCollection(faces, alpha=0.5, facecolor='cyan', 
                               edgecolor='blue', linewidth=1)
ax.add_collection3d(collection)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Tetrahedron: x + y + z ≤ 1')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.savefig('tetrahedron.png', dpi=150)
plt.show()
```

---

## 📖 Lab 5: Applications

### 5.1 Center of Mass

```python
import numpy as np
from scipy import integrate

# Center of mass of triangle (0,0), (1,0), (0,1) with ρ = 1
def rho(y, x):
    return 1  # uniform density

def y_upper(x):
    return 1 - x

# Total mass
mass, _ = integrate.dblquad(rho, 0, 1, 0, y_upper)

# Moments
def x_moment(y, x):
    return x

def y_moment(y, x):
    return y

Mx, _ = integrate.dblquad(y_moment, 0, 1, 0, y_upper)
My, _ = integrate.dblquad(x_moment, 0, 1, 0, y_upper)

x_bar = My / mass
y_bar = Mx / mass

print(f"Mass = {mass:.4f}")
print(f"Center of mass: ({x_bar:.4f}, {y_bar:.4f})")
print(f"Analytical: (1/3, 1/3)")
```

### 5.2 Moment of Inertia

```python
import numpy as np
from scipy import integrate

# I_y for rectangle [0,2] × [0,1] with ρ = 1
# I_y = ∬ x² ρ dA

def integrand(y, x):
    return x**2

I_y, _ = integrate.dblquad(integrand, 0, 2, 0, 1)

print(f"I_y = {I_y:.4f}")
print(f"Analytical: 8/3 = {8/3:.4f}")
```

---

## 📖 Lab 6: Symbolic Integration with SymPy

```python
from sympy import *

x, y, z = symbols('x y z')

# Double integral: ∬_D xy dA, D: 0 ≤ x ≤ 1, x² ≤ y ≤ x
result = integrate(x*y, (y, x**2, x), (x, 0, 1))
print(f"∬_D xy dA = {result}")

# Triple integral: ∭_E z dV, E: tetrahedron
result3 = integrate(z, (z, 0, 1-x-y), (y, 0, 1-x), (x, 0, 1))
print(f"∭_E z dV = {result3}")
```

---

## ✅ Lab Completion Checklist

- [ ] Computed double integrals numerically
- [ ] Visualized integration regions
- [ ] Computed triple integrals
- [ ] Created 3D visualizations
- [ ] Applied to center of mass problems
- [ ] Verified results with SymPy

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:
1. Computes the volume under z = sin(x)sin(y) over [0,π] × [0,π]
2. Visualizes the region and surface
3. Computes center of mass of a semicircular disk
4. Compares numerical and analytical results

Save as `Week7_MultipleIntegrals_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

---

*"Computation verifies intuition and reveals errors in calculation."*
