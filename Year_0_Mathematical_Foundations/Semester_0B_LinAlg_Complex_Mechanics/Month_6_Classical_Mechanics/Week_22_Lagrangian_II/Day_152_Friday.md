# Day 152: Rigid Body Motion — Introduction

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Rigid Body Kinematics |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define a rigid body and count its degrees of freedom
2. Describe rotation using Euler angles
3. Understand angular velocity and the rotation matrix
4. Compute kinetic energy of a rigid body
5. Define the inertia tensor
6. Connect to quantum angular momentum

---

## 📖 Core Content

### 1. What is a Rigid Body?

**Definition:** A rigid body is a collection of particles with fixed relative distances.

**Constraints:** For N particles, N(N-1)/2 distance constraints.

**Degrees of freedom:**
- 3D: 3 (translation) + 3 (rotation) = **6 DOF**
- 2D: 2 (translation) + 1 (rotation) = **3 DOF**

---

### 2. Configuration of a Rigid Body

**Specify completely by:**
1. Position of center of mass **R** = (X, Y, Z)
2. Orientation (rotation from reference)

**Orientation options:**
- Rotation matrix R (9 numbers, 6 constraints)
- Euler angles (α, β, γ) — 3 numbers
- Quaternions — 4 numbers, 1 constraint
- Axis-angle — 3 numbers (direction × angle)

---

### 3. Rotation Matrix

A 3×3 orthogonal matrix with det(R) = +1:
$$R^T R = I, \quad \det(R) = 1$$

**Properties:**
- Preserves lengths and angles
- Forms a group: SO(3)
- 3 independent parameters (3 DOF for rotation)

---

### 4. Euler Angles

**Convention (z-x-z or z-y-z):** Three successive rotations:
1. Rotate by α about z-axis
2. Rotate by β about new x-axis (line of nodes)
3. Rotate by γ about new z-axis

**Rotation matrix:**
$$R(\alpha, \beta, \gamma) = R_z(\gamma)R_x(\beta)R_z(\alpha)$$

**Ranges:** α, γ ∈ [0, 2π), β ∈ [0, π]

**Singularity:** Gimbal lock when β = 0 or π

---

### 5. Angular Velocity

**Definition:** The instantaneous angular velocity **ω** describes how fast and about which axis the body rotates.

**Relation to rotation matrix:**
$$\dot{R} = \tilde{\omega} R$$

where ω̃ is the antisymmetric matrix:
$$\tilde{\omega} = \begin{pmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{pmatrix}$$

**Velocity of a point:**
$$\mathbf{v} = \boldsymbol{\omega} \times \mathbf{r}$$

---

### 6. Body vs Space Frame

**Space frame:** Fixed in laboratory
**Body frame:** Fixed to rotating body

**Angular velocity can be expressed in either:**
- **ω** (space frame components)
- **ω'** (body frame components)

**Relation:** ω' = R^T ω

---

### 7. Kinetic Energy of Rigid Body

**Total kinetic energy:**
$$T = \frac{1}{2}M|\dot{\mathbf{R}}|^2 + \frac{1}{2}\sum_\alpha m_\alpha |\boldsymbol{\omega} \times \mathbf{r}'_\alpha|^2$$

where **r'**_α is position in body frame.

**Rotation part:**
$$T_{rot} = \frac{1}{2}\boldsymbol{\omega}^T \mathbf{I} \boldsymbol{\omega} = \frac{1}{2}\sum_{i,j} I_{ij}\omega_i\omega_j$$

where **I** is the **inertia tensor**.

---

### 8. Inertia Tensor

**Definition:**
$$I_{ij} = \sum_\alpha m_\alpha (|\mathbf{r}'_\alpha|^2\delta_{ij} - r'_{\alpha i}r'_{\alpha j})$$

For continuous body:
$$I_{ij} = \int \rho(\mathbf{r})(r^2\delta_{ij} - r_i r_j)\,d^3r$$

**Matrix form:**
$$\mathbf{I} = \begin{pmatrix} I_{xx} & I_{xy} & I_{xz} \\ I_{yx} & I_{yy} & I_{yz} \\ I_{zx} & I_{zy} & I_{zz} \end{pmatrix}$$

**Diagonal elements (moments of inertia):**
$$I_{xx} = \sum m(y^2 + z^2), \quad I_{yy} = \sum m(x^2 + z^2), \quad I_{zz} = \sum m(x^2 + y^2)$$

**Off-diagonal elements (products of inertia):**
$$I_{xy} = -\sum mxy$$

---

### 9. Principal Axes

**Principal axes:** Directions where I is diagonal.

**Find by eigenvalue problem:**
$$\mathbf{I}\hat{\mathbf{n}} = I_n\hat{\mathbf{n}}$$

**Principal moments:** I₁, I₂, I₃ (eigenvalues)

**In principal frame:**
$$T_{rot} = \frac{1}{2}(I_1\omega_1'^2 + I_2\omega_2'^2 + I_3\omega_3'^2)$$

---

### 10. 🔬 Quantum Connection

**Classical → Quantum angular momentum:**

| Classical | Quantum |
|-----------|---------|
| **L** = **I**·**ω** | L̂ operators |
| L² = I₁ω₁² + I₂ω₂² + I₃ω₃² | L̂² eigenvalue ℓ(ℓ+1)ℏ² |
| Principal axes | Quantization axes |
| Rigid rotor | Rotational spectroscopy |

**Rotational levels:**
$$E_J = \frac{\hbar^2}{2I}J(J+1)$$

---

## ✏️ Worked Examples

### Example 1: Moment of Inertia of a Rod

**Setup:** Uniform rod, mass M, length L, rotating about center.

$$I = \int_{-L/2}^{L/2} \rho x^2 dx = \rho \frac{x^3}{3}\bigg|_{-L/2}^{L/2} = \frac{M}{L} \cdot \frac{L^3}{12} = \frac{ML^2}{12}$$

About end: I = ML²/3 (parallel axis theorem)

---

### Example 2: Inertia Tensor of a Cube

**Setup:** Uniform cube, side a, mass M, at origin.

By symmetry, I is diagonal in axes through center parallel to edges:
$$I_{xx} = I_{yy} = I_{zz} = \frac{Ma^2}{6}$$

---

## 🔧 Practice Problems

### Level 1
1. Calculate the moment of inertia of a uniform disk about its axis.
2. Find the inertia tensor of two point masses m at (±a, 0, 0).

### Level 2
3. For a rectangular plate (a × b, mass M), find the inertia tensor.
4. Verify the parallel axis theorem: I = I_cm + Md².

### Level 3
5. Find the principal moments of inertia for a cone.
6. Express angular velocity in terms of Euler angle derivatives.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rigid_body_demo():
    """Demonstrate rigid body concepts."""
    
    print("=" * 70)
    print("RIGID BODY MECHANICS")
    print("=" * 70)
    
    # Example: Rectangular plate
    a, b, c = 3, 2, 0.1  # dimensions
    M = 1.0  # mass
    
    # Inertia tensor (principal axes aligned with plate)
    Ixx = M * (b**2 + c**2) / 12
    Iyy = M * (a**2 + c**2) / 12
    Izz = M * (a**2 + b**2) / 12
    
    I = np.diag([Ixx, Iyy, Izz])
    
    print(f"\nRectangular plate: {a} × {b} × {c}, mass = {M}")
    print(f"Principal moments of inertia:")
    print(f"  I_xx = {Ixx:.4f}")
    print(f"  I_yy = {Iyy:.4f}")
    print(f"  I_zz = {Izz:.4f}")
    
    fig = plt.figure(figsize=(15, 5))
    
    # Visualize the inertia ellipsoid
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Inertia ellipsoid: x²/I₁ + y²/I₂ + z²/I₃ = 1
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    
    x = np.sqrt(1/Ixx) * np.outer(np.cos(u), np.sin(v))
    y = np.sqrt(1/Iyy) * np.outer(np.sin(u), np.sin(v))
    z = np.sqrt(1/Izz) * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax1.plot_surface(x, y, z, alpha=0.7, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Inertia Ellipsoid')
    
    # Euler angle visualization
    ax2 = fig.add_subplot(132, projection='3d')
    
    def rotation_matrix(alpha, beta, gamma):
        """Euler rotation matrix (z-x-z convention)."""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        
        Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
        Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        
        return Rz2 @ Rx @ Rz1
    
    # Draw coordinate frames
    origin = np.array([0, 0, 0])
    
    # Space frame
    for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['X', 'Y', 'Z'])):
        vec = np.zeros(3)
        vec[i] = 1
        ax2.quiver(*origin, *vec, color=color, arrow_length_ratio=0.1, lw=2)
        ax2.text(*(vec*1.1), label, color=color, fontsize=12)
    
    # Rotated frame
    alpha, beta, gamma = np.pi/4, np.pi/3, np.pi/6
    R = rotation_matrix(alpha, beta, gamma)
    
    for i, color in enumerate(['darkred', 'darkgreen', 'darkblue']):
        vec = R[:, i]
        ax2.quiver(*origin, *vec, color=color, arrow_length_ratio=0.1, lw=2, linestyle='--')
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_title(f'Euler Angles\nα={np.degrees(alpha):.0f}°, β={np.degrees(beta):.0f}°, γ={np.degrees(gamma):.0f}°')
    
    # Kinetic energy vs angular velocity direction
    ax3 = fig.add_subplot(133)
    
    theta = np.linspace(0, 2*np.pi, 100)
    omega_mag = 1.0
    
    # Angular velocity in xy plane
    omega_x = omega_mag * np.cos(theta)
    omega_y = omega_mag * np.sin(theta)
    omega_z = np.zeros_like(theta)
    
    T_rot = 0.5 * (Ixx * omega_x**2 + Iyy * omega_y**2 + Izz * omega_z**2)
    
    ax3.polar(theta, T_rot, 'b-', lw=2)
    ax3.set_title('Rotational KE vs Direction\n(ω in xy-plane)')
    
    plt.tight_layout()
    plt.savefig('rigid_body.png', dpi=150)
    plt.show()

rigid_body_demo()
```

---

## 📝 Summary

### Rigid Body Configuration

| Quantity | DOF | Description |
|----------|-----|-------------|
| CM position | 3 | Translation |
| Orientation | 3 | Rotation (Euler angles) |
| **Total** | **6** | |

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Kinetic energy | T = ½MṘ² + ½**ω**ᵀ**I****ω** |
| Inertia tensor | Iᵢⱼ = Σm(r²δᵢⱼ - rᵢrⱼ) |
| Angular momentum | **L** = **I**·**ω** |

---

## ✅ Daily Checklist

- [ ] Understand rigid body DOF
- [ ] Describe orientation with Euler angles
- [ ] Compute inertia tensor
- [ ] Find principal axes
- [ ] Calculate rotational kinetic energy
- [ ] Connect to quantum mechanics

---

## 🔮 Preview: Day 153

Tomorrow is our **Computational Lab** where we simulate rigid body dynamics and explore Euler's equations!
