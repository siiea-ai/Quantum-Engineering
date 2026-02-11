# Day 141: Generalized Coordinates & Constraints

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Configuration Space |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define and use generalized coordinates
2. Distinguish holonomic from non-holonomic constraints
3. Count degrees of freedom
4. Transform between coordinate systems
5. Express kinetic energy in generalized coordinates
6. Understand configuration space geometry

---

## 📚 Required Reading

### Primary Text: Goldstein
- **Chapter 1, Sections 1.1-1.4**: Survey of Elementary Principles

### Alternative: Landau & Lifshitz
- **Chapter 1, Sections 1-3**: The Principle of Least Action

### Complementary
- **Taylor, Classical Mechanics**, Chapter 7

---

## 🎬 Video Resources

### MIT OpenCourseWare
- 8.01 Classical Mechanics: Generalized Coordinates

### Lectures
- Search "generalized coordinates Lagrangian" on YouTube

---

## 📖 Core Content: Theory and Concepts

### 1. Why Classical Mechanics Matters for Quantum

Before diving in, let's understand why:

| Classical Concept | Quantum Analog |
|-------------------|----------------|
| Lagrangian L | Path integral integrand |
| Action S = ∫L dt | Phase in propagator e^{iS/ℏ} |
| Hamilton's equations | Heisenberg equations |
| Poisson brackets | Commutators (×iℏ) |
| Canonical transformations | Unitary transformations |
| Hamilton-Jacobi equation | Schrödinger equation (classical limit) |

Understanding classical mechanics deeply makes quantum mechanics more intuitive!

---

### 2. Limitations of Newtonian Mechanics

**Newton's approach:**
$$\mathbf{F} = m\mathbf{a} = m\ddot{\mathbf{r}}$$

**Problems:**
1. Requires Cartesian coordinates (vector equation)
2. Constraint forces must be known
3. Difficult for complex geometries
4. Doesn't reveal underlying structure

**Example:** Simple pendulum
- Newton: Resolve forces along string and perpendicular
- Need to eliminate constraint force (tension)
- Messy coupled equations in (x, y)

**Better approach:** Use angle θ as single coordinate!

---

### 3. Generalized Coordinates

**Definition:** A set of **generalized coordinates** {q₁, q₂, ..., qₙ} is any set of n independent quantities that completely specifies the configuration of a system.

**Key Properties:**
- Can be any quantities (angles, distances, dimensionless ratios)
- Must be independent
- Number n = degrees of freedom
- Position of every particle expressible as rᵢ = rᵢ(q₁, ..., qₙ, t)

**Examples:**

| System | Cartesian | Generalized |
|--------|-----------|-------------|
| Particle in plane | (x, y) | (r, θ) or (x, y) |
| Simple pendulum | (x, y) with constraint | θ only |
| Double pendulum | (x₁,y₁,x₂,y₂) constrained | (θ₁, θ₂) |
| Rigid body | 3N particle coords | 6 (3 position + 3 orientation) |

---

### 4. Degrees of Freedom

**Definition:** The number of **degrees of freedom** (DOF) is the minimum number of independent coordinates needed to specify the system's configuration.

**Formula:**
$$\boxed{n = 3N - k}$$

where:
- N = number of particles
- k = number of independent constraints
- n = degrees of freedom

**Examples:**

| System | N | Constraints k | DOF n |
|--------|---|---------------|-------|
| Free particle in 3D | 1 | 0 | 3 |
| Particle on sphere | 1 | 1 (r = R) | 2 |
| Simple pendulum | 1 | 2 (plane + length) | 1 |
| Rigid body | ∞ (continuous) | ∞ - 6 | 6 |
| N particles, rigid | N | N(N-1)/2 distances | 6 (if N ≥ 3) |

---

### 5. Constraints

**Definition:** A **constraint** is a restriction on the possible configurations or motions of a system.

#### Types of Constraints:

**1. Holonomic Constraints**
Can be written as:
$$f(q_1, q_2, ..., q_n, t) = 0$$

Examples:
- Particle on sphere: x² + y² + z² - R² = 0
- Pendulum: x² + y² - L² = 0
- Rigid body distances: |rᵢ - rⱼ|² - dᵢⱼ² = 0

**2. Non-holonomic Constraints**
Cannot be integrated to the above form. Usually involve velocities:
$$g(q_1, ..., q_n, \dot{q}_1, ..., \dot{q}_n, t) = 0$$

Examples:
- Rolling without slipping: v = Rω (differential, not integrable)
- Disk rolling on plane

**Why it matters:**
- Holonomic: Can eliminate coordinates, reduce DOF
- Non-holonomic: Must use Lagrange multipliers or other methods

---

### 6. Configuration Space

**Definition:** The **configuration space** is the n-dimensional space of all generalized coordinates Q = (q₁, ..., qₙ).

**Properties:**
- Each point in Q represents one configuration
- Motion = curve in configuration space
- Constraints define submanifolds

**Examples:**

| System | Configuration Space |
|--------|---------------------|
| Free particle in 3D | ℝ³ |
| Particle on sphere | S² (2-sphere) |
| Simple pendulum | S¹ (circle) |
| Double pendulum | T² (2-torus) |
| Rigid body orientation | SO(3) |

**Geometry matters for quantum mechanics!**
- Topology of configuration space → quantization conditions
- Curvature → geometric phases

---

### 7. Kinetic Energy in Generalized Coordinates

The position of particle α in terms of generalized coordinates:
$$\mathbf{r}_\alpha = \mathbf{r}_\alpha(q_1, ..., q_n, t)$$

Velocity:
$$\mathbf{v}_\alpha = \dot{\mathbf{r}}_\alpha = \sum_i \frac{\partial \mathbf{r}_\alpha}{\partial q_i}\dot{q}_i + \frac{\partial \mathbf{r}_\alpha}{\partial t}$$

Kinetic energy:
$$T = \sum_\alpha \frac{1}{2}m_\alpha |\mathbf{v}_\alpha|^2$$

This can always be written as:
$$\boxed{T = \frac{1}{2}\sum_{i,j} M_{ij}(q,t)\dot{q}_i\dot{q}_j + \sum_i a_i(q,t)\dot{q}_i + T_0(q,t)}$$

where Mᵢⱼ is the **mass matrix** (symmetric, positive definite).

**For scleronomic systems** (time-independent constraints):
$$T = \frac{1}{2}\sum_{i,j} M_{ij}(q)\dot{q}_i\dot{q}_j$$

This is a quadratic form in velocities!

---

### 8. The Mass Matrix (Metric Tensor)

The mass matrix Mᵢⱼ defines a **metric** on configuration space:
$$ds^2 = \sum_{i,j} M_{ij}\,dq_i\,dq_j$$

This is the **kinetic energy metric** — it measures "distance" in configuration space weighted by inertia.

**Example: Particle in plane (polar coordinates)**

r = (r cos θ, r sin θ)

v = (ṙ cos θ - r θ̇ sin θ, ṙ sin θ + r θ̇ cos θ)

T = ½m(ṙ² + r²θ̇²)

Mass matrix:
$$M = m\begin{pmatrix} 1 & 0 \\ 0 & r^2 \end{pmatrix}$$

This is the metric of flat space in polar coordinates!

---

### 9. 🔬 Quantum Mechanics Connection

**Configuration Space in QM:**
The wavefunction ψ(q₁, ..., qₙ, t) lives on configuration space!

**Mass Matrix → Kinetic Energy Operator:**
$$\hat{T} = -\frac{\hbar^2}{2}\sum_{i,j} \frac{1}{\sqrt{g}}\frac{\partial}{\partial q_i}\left(\sqrt{g}\,g^{ij}\frac{\partial}{\partial q_j}\right)$$

where gᵢⱼ = Mᵢⱼ/m and g = det(gᵢⱼ).

**Topology Matters:**
- Particle on ring (S¹): ψ must be single-valued → quantized angular momentum
- Particle on sphere (S²): Spherical harmonics
- Identical particles: Configuration space has singularities → bosons/fermions

---

## ✏️ Worked Examples

### Example 1: Simple Pendulum

**Setup:** Mass m, length L, angle θ from vertical.

**Cartesian:**
- x = L sin θ
- y = -L cos θ (measuring down from pivot)

**Constraint:** x² + y² = L² (1 constraint)

**DOF:** 3 - 2 (plane motion) - 1 (length) = 1 ✓

**Generalized coordinate:** θ

**Kinetic energy:**
ẋ = L cos θ · θ̇, ẏ = L sin θ · θ̇

T = ½m(ẋ² + ẏ²) = ½m L²θ̇²(cos²θ + sin²θ) = **½mL²θ̇²**

**Mass matrix:** M = mL² (scalar for 1 DOF)

---

### Example 2: Double Pendulum

**Setup:** Two pendulums connected end-to-end. Masses m₁, m₂, lengths L₁, L₂.

**Generalized coordinates:** θ₁, θ₂ (angles from vertical)

**Positions:**
- x₁ = L₁ sin θ₁
- y₁ = -L₁ cos θ₁
- x₂ = L₁ sin θ₁ + L₂ sin θ₂
- y₂ = -L₁ cos θ₁ - L₂ cos θ₂

**Velocities:**
- ẋ₁ = L₁ cos θ₁ · θ̇₁
- ẏ₁ = L₁ sin θ₁ · θ̇₁
- ẋ₂ = L₁ cos θ₁ · θ̇₁ + L₂ cos θ₂ · θ̇₂
- ẏ₂ = L₁ sin θ₁ · θ̇₁ + L₂ sin θ₂ · θ̇₂

**Kinetic energy:**
$$T = \frac{1}{2}m_1(\dot{x}_1^2 + \dot{y}_1^2) + \frac{1}{2}m_2(\dot{x}_2^2 + \dot{y}_2^2)$$

After algebra:
$$T = \frac{1}{2}(m_1 + m_2)L_1^2\dot{\theta}_1^2 + \frac{1}{2}m_2 L_2^2\dot{\theta}_2^2 + m_2 L_1 L_2 \cos(\theta_1 - \theta_2)\dot{\theta}_1\dot{\theta}_2$$

**Mass matrix:**
$$M = \begin{pmatrix} (m_1+m_2)L_1^2 & m_2 L_1 L_2 \cos(\theta_1-\theta_2) \\ m_2 L_1 L_2 \cos(\theta_1-\theta_2) & m_2 L_2^2 \end{pmatrix}$$

Note: M depends on configuration (θ₁ - θ₂)!

---

### Example 3: Bead on Rotating Hoop

**Setup:** Bead of mass m slides on a vertical circular hoop of radius R that rotates about its vertical diameter with angular velocity ω.

**Constraint:** Bead stays on hoop (holonomic)

**DOF:** 1 (position on hoop, parameterized by angle θ from bottom)

**Position in rotating frame:**
- x' = R sin θ
- y' = 0
- z' = R(1 - cos θ)

**In lab frame:** The hoop rotates, so:
- x = R sin θ cos(ωt)
- y = R sin θ sin(ωt)
- z = R(1 - cos θ)

**Velocity:**
- ẋ = R cos θ cos(ωt) θ̇ - ωR sin θ sin(ωt)
- ẏ = R cos θ sin(ωt) θ̇ + ωR sin θ cos(ωt)
- ż = R sin θ θ̇

**Kinetic energy:**
$$T = \frac{1}{2}m(R^2\dot{\theta}^2 + \omega^2 R^2 \sin^2\theta)$$

This is a **rheonomic** (time-dependent) system — the second term comes from the rotation.

---

## 🔧 Practice Problems

### Level 1: Basic Coordinates
1. A particle moves on the surface of a cone z = √(x² + y²). How many DOF? What are good generalized coordinates?

2. Two particles connected by a rigid rod of length L move in a plane. Find the DOF and suitable generalized coordinates.

3. Express the kinetic energy of a particle in spherical coordinates.

### Level 2: Constraints
4. Classify each constraint as holonomic or non-holonomic:
   a) Particle on cylinder: x² + y² = R²
   b) Rolling sphere: v_contact = 0
   c) Particle in box: |x| < a, |y| < b, |z| < c

5. A disk of radius R rolls without slipping on a plane. Find the DOF and describe the configuration space.

### Level 3: Mass Matrix
6. Compute the mass matrix for a particle in 3D using:
   a) Cylindrical coordinates (ρ, φ, z)
   b) Spherical coordinates (r, θ, φ)

7. For the double pendulum, verify that det(M) > 0 for all configurations.

### Level 4: Theory
8. Prove that if all constraints are scleronomic (time-independent), then T is a homogeneous quadratic function of the velocities.

9. Show that the mass matrix Mᵢⱼ transforms as a tensor under coordinate changes.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def configuration_space_examples():
    """Visualize configuration spaces."""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Simple pendulum: S¹
    ax1 = fig.add_subplot(231)
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', lw=2)
    ax1.scatter([1], [0], c='red', s=100, zorder=5)
    ax1.set_title('Simple Pendulum\nConfig Space: S¹ (circle)')
    ax1.set_aspect('equal')
    ax1.set_xlabel('cos θ')
    ax1.set_ylabel('sin θ')
    
    # 2. Double pendulum: T²
    ax2 = fig.add_subplot(232, projection='3d')
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(u, v)
    R, r = 2, 0.5
    X = (R + r*np.cos(V)) * np.cos(U)
    Y = (R + r*np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax2.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
    ax2.set_title('Double Pendulum\nConfig Space: T² (torus)')
    
    # 3. Particle on sphere: S²
    ax3 = fig.add_subplot(233, projection='3d')
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2*np.pi, 30)
    PHI, THETA = np.meshgrid(phi, theta)
    X = np.sin(PHI) * np.cos(THETA)
    Y = np.sin(PHI) * np.sin(THETA)
    Z = np.cos(PHI)
    ax3.plot_surface(X, Y, Z, alpha=0.7, cmap='coolwarm')
    ax3.set_title('Particle on Sphere\nConfig Space: S²')
    
    # 4. Two particles on line (center of mass removed): ℝ
    ax4 = fig.add_subplot(234)
    x = np.linspace(-3, 3, 100)
    ax4.axhline(y=0, color='b', lw=2)
    ax4.scatter([0], [0], c='red', s=100, zorder=5)
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-1, 1)
    ax4.set_title('Two Particles on Line\nRelative coord: ℝ')
    ax4.set_xlabel('x₁ - x₂')
    
    # 5. Free rigid body orientation: SO(3) ≈ RP³
    ax5 = fig.add_subplot(235, projection='3d')
    # Visualize as ball with antipodal identification
    u = np.linspace(0, np.pi, 20)
    v = np.linspace(0, 2*np.pi, 40)
    U, V = np.meshgrid(u, v)
    R = np.pi  # Ball of radius π
    X = R * np.sin(U) * np.cos(V)
    Y = R * np.sin(U) * np.sin(V)
    Z = R * np.cos(U)
    ax5.plot_surface(X, Y, Z, alpha=0.3, cmap='plasma')
    ax5.set_title('Rigid Body Orientation\nConfig Space: SO(3)')
    
    # 6. Mass matrix visualization (pendulum)
    ax6 = fig.add_subplot(236)
    theta_vals = np.linspace(0, 2*np.pi, 100)
    L = 1
    m = 1
    M = m * L**2 * np.ones_like(theta_vals)
    ax6.plot(theta_vals, M, 'b-', lw=2)
    ax6.set_xlabel('θ')
    ax6.set_ylabel('M(θ)')
    ax6.set_title('Simple Pendulum\nMass Matrix (constant)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('configuration_spaces.png', dpi=150)
    plt.show()

configuration_space_examples()


def double_pendulum_mass_matrix():
    """Visualize mass matrix for double pendulum."""
    
    print("=" * 60)
    print("DOUBLE PENDULUM MASS MATRIX")
    print("=" * 60)
    
    # Parameters
    m1, m2 = 1.0, 1.0
    L1, L2 = 1.0, 1.0
    
    def mass_matrix(theta1, theta2):
        """Compute mass matrix at given configuration."""
        M11 = (m1 + m2) * L1**2
        M12 = m2 * L1 * L2 * np.cos(theta1 - theta2)
        M22 = m2 * L2**2
        return np.array([[M11, M12], [M12, M22]])
    
    # Visualize det(M) and eigenvalues
    theta1 = np.linspace(0, 2*np.pi, 100)
    theta2 = np.linspace(0, 2*np.pi, 100)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    det_M = np.zeros_like(T1)
    min_eig = np.zeros_like(T1)
    
    for i in range(len(theta1)):
        for j in range(len(theta2)):
            M = mass_matrix(theta1[i], theta2[j])
            det_M[j, i] = np.linalg.det(M)
            min_eig[j, i] = np.min(np.linalg.eigvalsh(M))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im0 = axes[0].contourf(T1, T2, det_M, levels=30, cmap='viridis')
    axes[0].set_xlabel('θ₁')
    axes[0].set_ylabel('θ₂')
    axes[0].set_title('det(M) for Double Pendulum\n(Always positive)')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(T1, T2, min_eig, levels=30, cmap='plasma')
    axes[1].set_xlabel('θ₁')
    axes[1].set_ylabel('θ₂')
    axes[1].set_title('Minimum Eigenvalue of M\n(Always positive)')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('double_pendulum_mass_matrix.png', dpi=150)
    plt.show()
    
    print(f"\ndet(M) range: [{np.min(det_M):.4f}, {np.max(det_M):.4f}]")
    print(f"min eigenvalue range: [{np.min(min_eig):.4f}, {np.max(min_eig):.4f}]")
    print("Mass matrix is positive definite everywhere!")

double_pendulum_mass_matrix()


def kinetic_energy_visualization():
    """Visualize kinetic energy in different coordinate systems."""
    
    print("\n" + "=" * 60)
    print("KINETIC ENERGY IN DIFFERENT COORDINATES")
    print("=" * 60)
    
    # Particle in 2D
    m = 1.0
    
    # Cartesian: T = (1/2)m(ẋ² + ẏ²)
    # Polar: T = (1/2)m(ṙ² + r²θ̇²)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Visualize metric in polar coordinates
    r = np.linspace(0.1, 3, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    R, THETA = np.meshgrid(r, theta)
    
    # The metric component g_θθ = r²
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    # Plot coordinate lines with spacing proportional to metric
    ax = axes[0]
    for ri in np.linspace(0.5, 2.5, 5):
        circle = plt.Circle((0, 0), ri, fill=False, color='blue', lw=1)
        ax.add_patch(circle)
    for ti in np.linspace(0, 2*np.pi, 12, endpoint=False):
        ax.plot([0, 3*np.cos(ti)], [0, 3*np.sin(ti)], 'r-', lw=1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Polar Coordinates\nr-lines (blue), θ-lines (red)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Visualize the metric tensor (as ellipses)
    ax = axes[1]
    for ri in [0.5, 1, 1.5, 2, 2.5]:
        for ti in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x0 = ri * np.cos(ti)
            y0 = ri * np.sin(ti)
            # At this point, metric is diag(1, r²)
            # Draw ellipse representing unit circle in velocity space
            angles = np.linspace(0, 2*np.pi, 50)
            # In (ṙ, θ̇) space, unit KE circle: ṙ² + r²θ̇² = 1
            # ṙ = cos(a), θ̇ = sin(a)/r
            # Transform to (vx, vy): vx = ṙ cos θ - r θ̇ sin θ, etc.
            scale = 0.15
            vr = scale * np.cos(angles)
            vtheta = scale * np.sin(angles) / ri
            vx = vr * np.cos(ti) - ri * vtheta * np.sin(ti)
            vy = vr * np.sin(ti) + ri * vtheta * np.cos(ti)
            ax.plot(x0 + vx, y0 + vy, 'b-', lw=0.5)
            ax.scatter([x0], [y0], c='red', s=10)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Metric Tensor Visualization\n(Unit KE ellipses in velocity space)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('kinetic_energy_metric.png', dpi=150)
    plt.show()

kinetic_energy_visualization()
```

---

## 📝 Summary

### Key Concepts

| Concept | Definition |
|---------|------------|
| Generalized coordinates | Minimal set of independent variables describing configuration |
| Degrees of freedom | n = 3N - k (particles minus constraints) |
| Holonomic constraint | f(q, t) = 0 (integrable) |
| Non-holonomic constraint | g(q, q̇, t) = 0 (not integrable) |
| Configuration space | n-dimensional space of all q values |
| Mass matrix | Mᵢⱼ such that T = ½ Σᵢⱼ Mᵢⱼ q̇ᵢ q̇ⱼ |

### Key Formulas

$$n = 3N - k \quad \text{(degrees of freedom)}$$

$$T = \frac{1}{2}\sum_{i,j} M_{ij}(q)\dot{q}_i\dot{q}_j \quad \text{(kinetic energy)}$$

---

## ✅ Daily Checklist

- [ ] Understand generalized coordinates concept
- [ ] Count degrees of freedom correctly
- [ ] Distinguish holonomic from non-holonomic constraints
- [ ] Transform kinetic energy to generalized coordinates
- [ ] Compute mass matrices
- [ ] Visualize configuration spaces
- [ ] Complete practice problems

---

## 🔮 Preview: Day 142

Tomorrow we introduce the **Principle of Least Action** — the deepest principle in physics that unifies mechanics, optics, electromagnetism, and leads directly to quantum mechanics!
