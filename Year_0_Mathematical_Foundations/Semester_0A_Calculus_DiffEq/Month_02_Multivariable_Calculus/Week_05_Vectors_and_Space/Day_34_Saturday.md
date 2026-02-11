# Day 34: Computational Lab — Vectors and 3D Geometry in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Vector Operations |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | 3D Visualization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Applications |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Implement vector operations in NumPy
2. Visualize vectors, lines, and planes in 3D
3. Solve geometric problems computationally
4. Create interactive 3D visualizations

---

## 🖥️ Setup

```bash
pip install numpy matplotlib scipy
```

---

## 📖 Lab 1: Vector Operations with NumPy

### 1.1 Basic Vector Operations

```python
import numpy as np

# Define vectors
a = np.array([3, -2, 5])
b = np.array([1, 4, -2])

print("=== Basic Vector Operations ===\n")
print(f"a = {a}")
print(f"b = {b}")

# Addition and subtraction
print(f"\na + b = {a + b}")
print(f"a - b = {a - b}")

# Scalar multiplication
print(f"\n3a = {3 * a}")
print(f"-2b = {-2 * b}")

# Magnitude (norm)
mag_a = np.linalg.norm(a)
mag_b = np.linalg.norm(b)
print(f"\n|a| = {mag_a:.4f}")
print(f"|b| = {mag_b:.4f}")

# Unit vectors
unit_a = a / mag_a
unit_b = b / mag_b
print(f"\nUnit vector of a: {unit_a}")
print(f"Magnitude check: {np.linalg.norm(unit_a):.6f}")
```

### 1.2 Dot and Cross Products

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("=== Dot and Cross Products ===\n")

# Dot product
dot_product = np.dot(a, b)
print(f"a · b = {dot_product}")

# Alternative: using @ operator
print(f"a @ b = {a @ b}")

# Cross product
cross_product = np.cross(a, b)
print(f"\na × b = {cross_product}")

# Verify perpendicularity
print(f"\n(a × b) · a = {np.dot(cross_product, a)}")
print(f"(a × b) · b = {np.dot(cross_product, b)}")

# Angle between vectors
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)
print(f"\nAngle between a and b: {theta_deg:.2f}°")
```

### 1.3 Projection Functions

```python
import numpy as np

def scalar_projection(b, a):
    """Scalar projection of b onto a"""
    return np.dot(a, b) / np.linalg.norm(a)

def vector_projection(b, a):
    """Vector projection of b onto a"""
    return (np.dot(a, b) / np.dot(a, a)) * a

def orthogonal_decomposition(b, a):
    """Decompose b into components parallel and perpendicular to a"""
    parallel = vector_projection(b, a)
    perpendicular = b - parallel
    return parallel, perpendicular

# Example
a = np.array([1, 1, 0])
b = np.array([3, 1, 2])

print("=== Projections ===\n")
print(f"a = {a}")
print(f"b = {b}")
print(f"\nScalar projection of b onto a: {scalar_projection(b, a):.4f}")
print(f"Vector projection of b onto a: {vector_projection(b, a)}")

parallel, perp = orthogonal_decomposition(b, a)
print(f"\nParallel component: {parallel}")
print(f"Perpendicular component: {perp}")
print(f"Sum (should equal b): {parallel + perp}")
print(f"Dot product (should be 0): {np.dot(parallel, perp):.10f}")
```

---

## 📖 Lab 2: 3D Visualization

### 2.1 Plotting Vectors

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_vectors(vectors, labels=None, colors=None, title="3D Vectors"):
    """Plot 3D vectors from the origin."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is None:
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
    if labels is None:
        labels = [f'v{i}' for i in range(len(vectors))]
    
    origin = np.zeros(3)
    
    for i, v in enumerate(vectors):
        ax.quiver(*origin, *v, color=colors[i % len(colors)], 
                  arrow_length_ratio=0.1, label=labels[i])
    
    # Set equal aspect ratio
    max_val = max(np.max(np.abs(v)) for v in vectors)
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('vectors_3d.png', dpi=150)
    plt.show()

# Example: Plot vectors and their cross product
a = np.array([2, 0, 0])
b = np.array([1, 2, 0])
c = np.cross(a, b)

plot_vectors([a, b, c], ['a', 'b', 'a × b'], ['r', 'g', 'b'], 
             'Vectors and Cross Product')
```

### 2.2 Plotting Lines

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_line_3d(point, direction, t_range=(-2, 2), num_points=100, 
                 ax=None, color='b', label='Line'):
    """Plot a line in 3D: r(t) = point + t * direction"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    t = np.linspace(t_range[0], t_range[1], num_points)
    x = point[0] + t * direction[0]
    y = point[1] + t * direction[1]
    z = point[2] + t * direction[2]
    
    ax.plot(x, y, z, color=color, label=label, linewidth=2)
    ax.scatter(*point, color=color, s=50, marker='o')
    
    return ax

# Plot two lines
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Line 1: through (1, 0, 0) with direction (1, 1, 0)
plot_line_3d(np.array([1, 0, 0]), np.array([1, 1, 0]), 
             ax=ax, color='r', label='Line 1')

# Line 2: through (0, 1, 0) with direction (1, 0, 1)
plot_line_3d(np.array([0, 1, 0]), np.array([1, 0, 1]), 
             ax=ax, color='b', label='Line 2')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Two Lines in 3D Space')
plt.savefig('lines_3d.png', dpi=150)
plt.show()
```

### 2.3 Plotting Planes

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_plane(a, b, c, d, x_range=(-5, 5), y_range=(-5, 5), 
               ax=None, color='blue', alpha=0.5, label='Plane'):
    """Plot plane ax + by + cz = d"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    
    if c != 0:
        Z = (d - a*X - b*Y) / c
        ax.plot_surface(X, Y, Z, alpha=alpha, color=color, label=label)
    else:
        print("Warning: c = 0, plane is vertical")
    
    # Plot normal vector
    point = np.array([0, 0, d/c]) if c != 0 else np.array([d/a, 0, 0])
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal) * 2  # Scale for visibility
    
    ax.quiver(*point, *normal, color='red', arrow_length_ratio=0.2)
    
    return ax

# Plot a plane: 2x + 3y + z = 6
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

plot_plane(2, 3, 1, 6, ax=ax)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Plane: 2x + 3y + z = 6 with Normal Vector')
plt.savefig('plane_3d.png', dpi=150)
plt.show()
```

---

## 📖 Lab 3: Geometric Calculations

### 3.1 Distance Functions

```python
import numpy as np

def distance_point_to_plane(point, a, b, c, d):
    """Distance from point to plane ax + by + cz + d = 0"""
    x, y, z = point
    numerator = abs(a*x + b*y + c*z + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    return numerator / denominator

def distance_point_to_line(point, line_point, direction):
    """Distance from point to line through line_point with given direction"""
    v = point - line_point
    cross = np.cross(v, direction)
    return np.linalg.norm(cross) / np.linalg.norm(direction)

def distance_between_parallel_planes(a, b, c, d1, d2):
    """Distance between parallel planes ax+by+cz=d1 and ax+by+cz=d2"""
    return abs(d2 - d1) / np.sqrt(a**2 + b**2 + c**2)

# Examples
print("=== Distance Calculations ===\n")

# Point to plane
point = np.array([1, 2, 3])
dist = distance_point_to_plane(point, 2, -2, 1, -5)  # 2x - 2y + z - 5 = 0
print(f"Distance from (1,2,3) to plane 2x-2y+z=5: {dist:.4f}")

# Point to line
point = np.array([1, 1, 1])
line_point = np.array([0, 0, 0])
direction = np.array([1, 2, 2])
dist = distance_point_to_line(point, line_point, direction)
print(f"Distance from (1,1,1) to line through origin with dir (1,2,2): {dist:.4f}")

# Between parallel planes
dist = distance_between_parallel_planes(2, 1, -1, 1, 4)
print(f"Distance between planes 2x+y-z=1 and 2x+y-z=4: {dist:.4f}")
```

### 3.2 Line-Plane Intersection

```python
import numpy as np

def line_plane_intersection(line_point, line_dir, plane_normal, plane_d):
    """
    Find intersection of line and plane.
    Line: r(t) = line_point + t * line_dir
    Plane: n · r = d
    """
    # Check if line is parallel to plane
    denom = np.dot(plane_normal, line_dir)
    if abs(denom) < 1e-10:
        return None, None  # Parallel
    
    # Solve for t
    t = (plane_d - np.dot(plane_normal, line_point)) / denom
    
    # Find intersection point
    intersection = line_point + t * line_dir
    
    return intersection, t

# Example
line_point = np.array([1, 0, 2])
line_dir = np.array([2, 1, -1])
plane_normal = np.array([3, -1, 2])
plane_d = 10

point, t = line_plane_intersection(line_point, line_dir, plane_normal, plane_d)
if point is not None:
    print(f"Intersection point: {point}")
    print(f"Parameter t: {t}")
    # Verify
    print(f"Verification (should equal {plane_d}): {np.dot(plane_normal, point)}")
else:
    print("Line is parallel to plane")
```

### 3.3 Area and Volume

```python
import numpy as np

def triangle_area(A, B, C):
    """Area of triangle with vertices A, B, C"""
    AB = B - A
    AC = C - A
    cross = np.cross(AB, AC)
    return 0.5 * np.linalg.norm(cross)

def parallelogram_area(a, b):
    """Area of parallelogram with adjacent sides a and b"""
    return np.linalg.norm(np.cross(a, b))

def parallelepiped_volume(a, b, c):
    """Volume of parallelepiped with edges a, b, c"""
    return abs(np.dot(a, np.cross(b, c)))

def tetrahedron_volume(A, B, C, D):
    """Volume of tetrahedron with vertices A, B, C, D"""
    AB = B - A
    AC = C - A
    AD = D - A
    return abs(np.dot(AB, np.cross(AC, AD))) / 6

# Examples
print("=== Areas and Volumes ===\n")

# Triangle
A = np.array([1, 0, 0])
B = np.array([0, 2, 0])
C = np.array([0, 0, 3])
print(f"Triangle area: {triangle_area(A, B, C):.4f}")

# Parallelogram
a = np.array([2, 1, 0])
b = np.array([1, 3, 1])
print(f"Parallelogram area: {parallelogram_area(a, b):.4f}")

# Parallelepiped
a = np.array([1, 0, 0])
b = np.array([1, 1, 0])
c = np.array([1, 1, 1])
print(f"Parallelepiped volume: {parallelepiped_volume(a, b, c):.4f}")

# Tetrahedron
A = np.array([0, 0, 0])
B = np.array([1, 0, 0])
C = np.array([0, 1, 0])
D = np.array([0, 0, 1])
print(f"Tetrahedron volume: {tetrahedron_volume(A, B, C, D):.4f}")
```

---

## 📖 Lab 4: Interactive Visualization

### 4.1 Animated Cross Product

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def animate_cross_product():
    """Animate a vector rotating and its cross product with a fixed vector"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fixed vector
    a = np.array([1, 0, 0])
    
    def update(frame):
        ax.clear()
        
        # Rotating vector
        theta = np.radians(frame * 5)
        b = np.array([np.cos(theta), np.sin(theta), 0])
        
        # Cross product
        c = np.cross(a, b)
        
        # Plot vectors
        origin = np.zeros(3)
        ax.quiver(*origin, *a, color='r', arrow_length_ratio=0.1, label='a')
        ax.quiver(*origin, *b, color='g', arrow_length_ratio=0.1, label='b')
        ax.quiver(*origin, *c, color='b', arrow_length_ratio=0.1, label='a × b')
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Cross Product Animation (θ = {frame * 5}°)')
        ax.legend()
    
    ani = FuncAnimation(fig, update, frames=72, interval=50, repeat=True)
    plt.show()
    return ani

# Uncomment to run animation:
# ani = animate_cross_product()
```

---

## ✅ Lab Completion Checklist

- [ ] Completed Lab 1: Vector operations
- [ ] Completed Lab 2: 3D visualization
- [ ] Completed Lab 3: Geometric calculations
- [ ] Completed Lab 4: Interactive visualization
- [ ] Saved all figures
- [ ] Created custom visualization

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. Implements all distance formulas (point-to-point, point-to-line, point-to-plane)
2. Visualizes two intersecting planes and their line of intersection
3. Computes and visualizes the trajectory of a particle under constant force
4. Creates an interactive visualization of your choice

Save as `Week5_Vectors_Lab.ipynb`

---

## 🔜 Tomorrow: Rest and Review

Day 35 consolidates Week 5 learning and previews partial derivatives.

---

*"Computation transforms abstract mathematics into visible, tangible understanding."*
