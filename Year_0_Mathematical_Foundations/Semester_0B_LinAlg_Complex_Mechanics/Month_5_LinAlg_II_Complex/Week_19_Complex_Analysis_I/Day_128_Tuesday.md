# Day 128: The Complex Plane — Topology and Geometry

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Topology of ℂ |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Work with sets in the complex plane
2. Understand open, closed, and compact sets
3. Define limits and continuity for complex functions
4. Work with sequences and series in ℂ
5. Understand the extended complex plane (Riemann sphere)
6. Describe paths and curves in the complex plane

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 2, Sections 11-17**: Regions in the Complex Plane

### Alternative: Ahlfors, "Complex Analysis"
- **Chapter 1.3-1.5**: Topology of the complex plane

---

## 📖 Core Content: Theory and Concepts

### 1. Sets in the Complex Plane

**Neighborhoods:**
The **ε-neighborhood** of z₀ is:
$$N_\varepsilon(z_0) = \{z \in \mathbb{C} : |z - z_0| < \varepsilon\}$$

This is an open disk of radius ε centered at z₀.

**Deleted neighborhood:**
$$N_\varepsilon^*(z_0) = \{z \in \mathbb{C} : 0 < |z - z_0| < \varepsilon\}$$

(Excludes the center point)

### 2. Interior, Boundary, and Exterior Points

For a set S ⊂ ℂ:

**Interior point:** z₀ is interior to S if some neighborhood of z₀ lies entirely in S.
**Exterior point:** z₀ is exterior to S if some neighborhood of z₀ lies entirely outside S.
**Boundary point:** z₀ is a boundary point of S if every neighborhood contains points both in S and outside S.

**Notation:**
- **int(S)** or **S°** = interior of S
- **∂S** = boundary of S
- **S̄** = closure of S = S ∪ ∂S

### 3. Open and Closed Sets

**Open set:** A set is open if it contains none of its boundary points (equivalently, every point is an interior point).

**Closed set:** A set is closed if it contains all its boundary points.

**Examples:**
| Set | Open? | Closed? |
|-----|-------|---------|
| {z : \|z\| < 1} (open disk) | Yes | No |
| {z : \|z\| ≤ 1} (closed disk) | No | Yes |
| {z : \|z\| = 1} (circle) | No | Yes |
| {z : 0 < \|z\| < 1} (punctured disk) | Yes | No |
| ℂ (whole plane) | Yes | Yes |
| ∅ (empty set) | Yes | Yes |

### 4. Connected and Simply Connected Sets

**Connected:** A set D is connected if any two points can be joined by a path lying entirely in D.

**Simply connected:** A connected set D is simply connected if every simple closed curve in D encloses only points of D (no "holes").

**Examples:**
- Open disk: simply connected
- Annulus {z : 1 < |z| < 2}: connected but NOT simply connected
- Punctured plane ℂ\{0}: connected but NOT simply connected

### 5. Bounded and Compact Sets

**Bounded:** S is bounded if S ⊂ {z : |z| < R} for some R.

**Compact:** S is compact if it is closed AND bounded.

**Heine-Borel Theorem:** In ℂ, compact ⟺ closed and bounded.

### 6. Limits in ℂ

**Definition:** 
$$\lim_{z \to z_0} f(z) = w_0$$
means: for every ε > 0, there exists δ > 0 such that
$$0 < |z - z_0| < \delta \implies |f(z) - w_0| < \varepsilon$$

**Key difference from ℝ:** In ℂ, z can approach z₀ from ANY direction!

**Connection to real limits:**
If f(z) = u(x,y) + iv(x,y) and z₀ = x₀ + iy₀, then:
$$\lim_{z \to z_0} f(z) = L \iff \lim_{(x,y) \to (x_0,y_0)} u(x,y) + i\lim_{(x,y) \to (x_0,y_0)} v(x,y) = L$$

### 7. Continuity

**Definition:** f is continuous at z₀ if:
1. f(z₀) is defined
2. lim_{z→z₀} f(z) exists
3. lim_{z→z₀} f(z) = f(z₀)

**Properties:** If f and g are continuous at z₀:
- f ± g continuous at z₀
- fg continuous at z₀
- f/g continuous at z₀ (if g(z₀) ≠ 0)

### 8. The Extended Complex Plane

**Point at infinity:** Add a point ∞ to ℂ.

**Extended complex plane:** ℂ∞ = ℂ ∪ {∞}

**Riemann Sphere:** Visualize ℂ∞ as a sphere via stereographic projection:
- North pole = ∞
- South pole = 0
- Equator = unit circle

**Neighborhoods of ∞:**
$$N_R(\infty) = \{z \in \mathbb{C} : |z| > R\} \cup \{\infty\}$$

### 9. Paths and Curves

**Path (parametric curve):**
$$\gamma: [a,b] \to \mathbb{C}, \quad \gamma(t) = x(t) + iy(t)$$

**Smooth path:** x(t) and y(t) are continuously differentiable.

**Simple path:** γ is one-to-one (doesn't cross itself).

**Closed path:** γ(a) = γ(b).

**Simple closed curve (Jordan curve):** Simple and closed.

---

## 🔬 Quantum Mechanics Connection

### State Space Topology

**Hilbert space:** Quantum states live in a complex Hilbert space.

**Continuity matters:** Small changes in parameters should give small changes in states (for adiabatic processes).

**Projective Hilbert space:** States differing by a phase are equivalent:
|ψ⟩ ~ e^(iθ)|ψ⟩

This is like the Riemann sphere for qubits! (Bloch sphere)

### Path Integrals (Preview)

**Feynman path integral:**
$$\langle x_f | e^{-iHt/\hbar} | x_i \rangle = \int \mathcal{D}[x(t)] e^{iS[x(t)]/\hbar}$$

- Sum over ALL paths from xᵢ to x_f
- Each path contributes with complex amplitude e^(iS/ℏ)
- Paths in complex plane essential for analytic continuation

### Berry Phase

**Geometric phase:** When parameters trace a closed path, state acquires a phase:
$$|\psi(T)\rangle = e^{i\gamma_g}e^{i\gamma_d}|\psi(0)\rangle$$

- γ_d = dynamical phase
- γ_g = geometric (Berry) phase

Berry phase depends on the path in parameter space — topology matters!

---

## ✏️ Worked Examples

### Example 1: Classifying Points

For S = {z : |z| < 2, z ≠ 1}, classify:
(a) z = 0
(b) z = 1
(c) z = 2
(d) z = 3

**Solution:**
(a) z = 0: Interior point (disk of radius 1 around 0 lies in S)
(b) z = 1: Boundary point (every neighborhood contains points in S and points not in S)
(c) z = 2: Boundary point (on |z| = 2 circle)
(d) z = 3: Exterior point (disk of radius 0.5 around 3 lies outside S)

### Example 2: Open/Closed Classification

Is S = {z : 1 ≤ |z| < 2} open, closed, or neither?

**Solution:**
- Points on |z| = 1 are in S but are boundary points → S is closed w.r.t. inner circle
- Points on |z| = 2 are boundary points not in S → S is not closed w.r.t. outer circle

S is **neither open nor closed**.

### Example 3: Simply Connected?

Is the set D = {z : |z| < 2, |z - 1| > 0.5} simply connected?

**Solution:**
D is the disk of radius 2 with a small disk around z = 1 removed.
A circle around z = 1 (inside D) encloses the hole, which is NOT in D.
Therefore D is **not simply connected**.

### Example 4: Computing a Limit

Find lim_{z→0} z̄/z.

**Solution:**
Let z = re^(iθ), then z̄ = re^(-iθ).
$$\frac{\bar{z}}{z} = \frac{re^{-i\theta}}{re^{i\theta}} = e^{-2i\theta}$$

This depends on θ (the direction of approach), so the limit does NOT exist!

- Along positive real axis (θ = 0): limit = 1
- Along positive imaginary axis (θ = π/2): limit = e^(-iπ) = -1
- Different directions give different limits!

### Example 5: Continuity Check

Is f(z) = |z|² continuous at z = 1 + i?

**Solution:**
f(z) = |z|² = x² + y² (where z = x + iy)

This is a polynomial in x and y, hence continuous everywhere in ℂ.

At z = 1 + i: f(1 + i) = 1² + 1² = 2

lim_{z→1+i} |z|² = |1 + i|² = 2 = f(1 + i) ✓

Yes, f is continuous at 1 + i.

---

## 📝 Practice Problems

### Level 1: Sets
1. Describe the set {z : Re(z) > 0}. Is it open? Bounded?

2. Find the boundary of S = {z : |z - i| ≤ 1}.

3. Is the set {z : |z| = 1} connected?

### Level 2: Open/Closed/Connected
4. Prove that the union of two open sets is open.

5. Is S = {z : Im(z) ≠ 0} connected?

6. Find a set that is both open and closed (besides ∅ and ℂ).

### Level 3: Limits
7. Prove lim_{z→0} z² = 0 using the ε-δ definition.

8. Show that lim_{z→0} Re(z)/|z| does not exist.

9. If lim_{z→z₀} f(z) = L, prove lim_{z→z₀} |f(z)| = |L|.

### Level 4: Paths and Topology
10. Parametrize the circle |z - 1| = 2 traversed counterclockwise.

11. Show that ℂ\{0} is not simply connected.

12. Prove: A set S is closed iff its complement is open.

---

## 💻 Evening Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection

# ============================================
# Visualizing Sets in the Complex Plane
# ============================================

def plot_complex_set(ax, condition_func, title, xlim=(-3, 3), ylim=(-3, 3), 
                     resolution=500, show_boundary=True):
    """Plot a set defined by a condition function"""
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    # Evaluate condition
    mask = condition_func(Z)
    
    # Plot
    ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], 
                colors=['lightblue'], alpha=0.7)
    if show_boundary:
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors=['blue'], linewidths=2)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

# Examples of different sets
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Open disk
plot_complex_set(axes[0, 0], lambda z: np.abs(z) < 1, 
                 'Open Disk: |z| < 1')

# Closed disk
plot_complex_set(axes[0, 1], lambda z: np.abs(z) <= 1, 
                 'Closed Disk: |z| ≤ 1')

# Annulus (not simply connected)
plot_complex_set(axes[0, 2], lambda z: (np.abs(z) > 0.5) & (np.abs(z) < 1.5), 
                 'Annulus: 0.5 < |z| < 1.5\n(Not Simply Connected)')

# Half-plane
plot_complex_set(axes[1, 0], lambda z: np.real(z) > 0, 
                 'Right Half-Plane: Re(z) > 0')

# Punctured disk
plot_complex_set(axes[1, 1], lambda z: (np.abs(z) < 1) & (np.abs(z) > 0.01), 
                 'Punctured Disk: 0 < |z| < 1')

# Disk with hole removed
plot_complex_set(axes[1, 2], lambda z: (np.abs(z) < 2) & (np.abs(z - 1) > 0.5), 
                 'Disk with Hole\n(Not Simply Connected)')

plt.tight_layout()
plt.savefig('complex_sets.png', dpi=150)
plt.show()

# ============================================
# Limits: Direction Dependence
# ============================================

def visualize_limit_paths():
    """Show how limits depend on direction of approach"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # f(z) = z̄/z - limit doesn't exist at 0
    ax = axes[0]
    
    # Draw paths approaching 0
    n_paths = 12
    for k in range(n_paths):
        theta = 2 * np.pi * k / n_paths
        t = np.linspace(0.1, 1, 50)
        path = t * np.exp(1j * theta)
        
        # Color by the "limit" value along this path
        limit_val = np.exp(-2j * theta)
        color = plt.cm.hsv(np.angle(limit_val) / (2*np.pi) + 0.5)
        
        ax.plot(path.real, path.imag, color=color, linewidth=2)
        ax.annotate(f'{np.real(limit_val):.1f}+{np.imag(limit_val):.1f}i', 
                   (path[-1].real, path[-1].imag), fontsize=8)
    
    ax.scatter([0], [0], c='red', s=100, zorder=5, label='z₀ = 0')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('$f(z) = \\bar{z}/z$: Different limits for different paths!')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # f(z) = z² - limit exists at 0
    ax = axes[1]
    
    for k in range(n_paths):
        theta = 2 * np.pi * k / n_paths
        t = np.linspace(0.1, 1, 50)
        path = t * np.exp(1j * theta)
        values = path**2
        
        ax.plot(path.real, path.imag, 'b-', linewidth=1, alpha=0.5)
        ax.plot(values.real, values.imag, 'r-', linewidth=2)
    
    ax.scatter([0], [0], c='green', s=100, zorder=5, label='Limit = 0')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('$f(z) = z²$: All paths give same limit')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('complex_limits.png', dpi=150)
    plt.show()

visualize_limit_paths()

# ============================================
# Riemann Sphere Visualization
# ============================================

def plot_riemann_sphere():
    """Visualize stereographic projection"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D view of Riemann sphere
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Draw sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax1.plot_surface(x, y, z, alpha=0.3, color='cyan')
    
    # Mark special points
    ax1.scatter([0], [0], [1], c='red', s=100, label='North Pole (∞)')
    ax1.scatter([0], [0], [-1], c='blue', s=100, label='South Pole (0)')
    ax1.scatter([1], [0], [0], c='green', s=100, label='z = 1')
    ax1.scatter([0], [1], [0], c='orange', s=100, label='z = i')
    
    # Draw equator
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 0, 'k-', linewidth=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Riemann Sphere')
    ax1.legend()
    
    # 2D complex plane with corresponding points
    ax2 = fig.add_subplot(122)
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit Circle (Equator)')
    
    # Mark corresponding points
    ax2.scatter([0], [0], c='blue', s=100, label='0 (South Pole)')
    ax2.scatter([1], [0], c='green', s=100, label='1')
    ax2.scatter([0], [1], c='orange', s=100, label='i')
    ax2.annotate('∞ (North Pole)', xy=(2, 2), fontsize=12, color='red')
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')
    ax2.set_title('Complex Plane')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('riemann_sphere.png', dpi=150)
    plt.show()

plot_riemann_sphere()

# ============================================
# Parametric Curves in ℂ
# ============================================

def plot_curves():
    """Visualize various parametric curves"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    t = np.linspace(0, 2*np.pi, 200)
    
    curves = [
        (lambda t: np.exp(1j*t), 'Unit Circle: $e^{it}$'),
        (lambda t: 2*np.exp(1j*t) + 1, 'Circle: $2e^{it} + 1$'),
        (lambda t: np.exp(2j*t), 'Double Speed: $e^{2it}$'),
        (lambda t: t*np.exp(1j*t), 'Spiral: $t e^{it}$'),
        (lambda t: np.cos(t) + 1j*np.sin(3*t), 'Lissajous'),
        (lambda t: (1 + 0.5*np.cos(5*t))*np.exp(1j*t), 'Flower Pattern'),
    ]
    
    for ax, (curve_func, title) in zip(axes.flatten(), curves):
        z = curve_func(t)
        
        # Color by parameter t
        points = np.array([z.real, z.imag]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        ax.scatter(z.real, z.imag, c=t, cmap='viridis', s=5)
        ax.scatter([z[0].real], [z[0].imag], c='green', s=100, zorder=5, label='Start')
        ax.scatter([z[-1].real], [z[-1].imag], c='red', s=100, zorder=5, label='End')
        
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('complex_curves.png', dpi=150)
    plt.show()

plot_curves()

# ============================================
# Testing Connectivity
# ============================================

print("=== Set Properties ===\n")

def analyze_set(name, condition_str, test_points):
    """Analyze properties of a set"""
    print(f"Set: {name}")
    print(f"Condition: {condition_str}")
    print("Test points:")
    for z, expected in test_points:
        print(f"  z = {z}: {expected}")
    print()

analyze_set(
    "Open unit disk",
    "|z| < 1",
    [(0, "interior"), (0.5, "interior"), (1, "boundary"), (2, "exterior")]
)

analyze_set(
    "Punctured plane",
    "z ≠ 0",
    [(0, "excluded"), (1, "in set"), (-1, "in set"), (1j, "in set")]
)
```

---

## ✅ Daily Checklist

- [ ] Understand neighborhoods in ℂ
- [ ] Classify interior, boundary, exterior points
- [ ] Know definitions of open, closed, bounded, compact
- [ ] Understand connected and simply connected
- [ ] Compute limits in the complex plane
- [ ] Recognize direction-dependent limit failures
- [ ] Parametrize simple curves
- [ ] Complete computational lab
- [ ] Solve at least 6 practice problems

---

## 🔜 Preview: Tomorrow

**Day 129: Complex Functions — Mappings and Transformations**
- Functions of a complex variable
- Real and imaginary parts
- Linear and Möbius transformations
- Conformal mappings preview
- QM Connection: Coordinate transformations in quantum systems

---

*"The complex plane is where algebra and geometry become one."*
— Visual Complex Analysis
