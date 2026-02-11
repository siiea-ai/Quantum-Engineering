# Day 414: Rotation Operators

## Week 60, Day 1 | Month 15: Angular Momentum

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning** | 2.5 hrs | Classical rotations and SO(3) group structure |
| **Afternoon** | 2.5 hrs | Quantum rotation operators and worked problems |
| **Evening** | 2 hrs | Computational lab: rotation operator construction |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Describe classical rotation matrices** as elements of the special orthogonal group SO(3)
2. **Derive the quantum rotation operator** $$\hat{R}(\hat{n},\theta) = e^{-i\theta\hat{n}\cdot\hat{\mathbf{J}}/\hbar}$$
3. **Apply infinitesimal rotations** to identify angular momentum as the generator
4. **Demonstrate non-commutativity** of finite rotations (non-Abelian structure)
5. **Connect rotation operators to SU(2)** and understand the double cover relationship
6. **Implement rotation operators** computationally for various angular momentum values

---

## Morning Session: Classical Rotations and SO(3)

### Classical Rotation Matrices

A rotation in three-dimensional space preserves lengths and orientations. Mathematically, a rotation is represented by a $$3 \times 3$$ orthogonal matrix $$R$$ with determinant $$+1$$:

$$R^T R = R R^T = I, \quad \det(R) = +1$$

The set of all such matrices forms the **special orthogonal group** SO(3).

### Rotations About Coordinate Axes

Rotation about the $$z$$-axis by angle $$\theta$$:

$$R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

Rotation about the $$y$$-axis by angle $$\theta$$:

$$R_y(\theta) = \begin{pmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{pmatrix}$$

Rotation about the $$x$$-axis by angle $$\theta$$:

$$R_x(\theta) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{pmatrix}$$

### Axis-Angle Representation

Any rotation can be parameterized by an axis $$\hat{n}$$ and angle $$\theta$$. The **Rodrigues formula** gives:

$$\boxed{R(\hat{n},\theta) = I + \sin\theta\, N + (1-\cos\theta)N^2}$$

where $$N$$ is the skew-symmetric matrix associated with $$\hat{n} = (n_x, n_y, n_z)$$:

$$N = \begin{pmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{pmatrix}$$

This can also be written as the matrix exponential:

$$R(\hat{n},\theta) = e^{\theta N}$$

### Generators of SO(3)

The **generators** of rotations are obtained by differentiating at $$\theta = 0$$:

$$J_x^{(\text{cl})} = \left.\frac{dR_x}{d\theta}\right|_{\theta=0} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}$$

$$J_y^{(\text{cl})} = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix}, \quad J_z^{(\text{cl})} = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

These satisfy the **SO(3) Lie algebra**:

$$[J_i^{(\text{cl})}, J_j^{(\text{cl})}] = \epsilon_{ijk}J_k^{(\text{cl})}$$

---

## Quantum Rotation Operators

### From Classical to Quantum

In quantum mechanics, a rotation transforms a state $$|\psi\rangle$$ into $$|\psi'\rangle = \hat{R}|\psi\rangle$$. The rotation operator must be **unitary** to preserve probability:

$$\hat{R}^\dagger \hat{R} = \hat{I}$$

### Angular Momentum as Generator

For an **infinitesimal rotation** by angle $$d\theta$$ about axis $$\hat{n}$$:

$$\boxed{\hat{R}(\hat{n}, d\theta) = \hat{I} - \frac{i}{\hbar}d\theta\,\hat{n}\cdot\hat{\mathbf{J}} + O(d\theta^2)}$$

where $$\hat{\mathbf{J}} = (\hat{J}_x, \hat{J}_y, \hat{J}_z)$$ is the **angular momentum operator**.

The angular momentum operators are the **generators of rotations** in quantum mechanics.

### Finite Rotation Operator

Building up a finite rotation from infinitesimal rotations:

$$\hat{R}(\hat{n},\theta) = \lim_{N\to\infty}\left(\hat{I} - \frac{i\theta}{N\hbar}\hat{n}\cdot\hat{\mathbf{J}}\right)^N$$

This gives the **exponential form**:

$$\boxed{\hat{R}(\hat{n},\theta) = e^{-i\theta\hat{n}\cdot\hat{\mathbf{J}}/\hbar} = \exp\left(-\frac{i\theta}{\hbar}(n_x\hat{J}_x + n_y\hat{J}_y + n_z\hat{J}_z)\right)}$$

### Rotations About Coordinate Axes

$$\hat{R}_z(\theta) = e^{-i\theta\hat{J}_z/\hbar}$$

$$\hat{R}_y(\theta) = e^{-i\theta\hat{J}_y/\hbar}$$

$$\hat{R}_x(\theta) = e^{-i\theta\hat{J}_x/\hbar}$$

### Action on Angular Momentum Eigenstates

For a state $$|j,m\rangle$$ with definite $$j$$ and $$m$$:

$$\hat{R}_z(\theta)|j,m\rangle = e^{-im\theta}|j,m\rangle$$

The action of $$\hat{R}_x$$ and $$\hat{R}_y$$ mixes different $$m$$ values while preserving $$j$$.

---

## Non-Commutativity of Rotations

### Non-Abelian Group Structure

Finite rotations do **not commute** in general:

$$\hat{R}_x(\theta_1)\hat{R}_y(\theta_2) \neq \hat{R}_y(\theta_2)\hat{R}_x(\theta_1)$$

This is because the angular momentum operators satisfy:

$$[\hat{J}_i, \hat{J}_j] = i\hbar\epsilon_{ijk}\hat{J}_k$$

### Baker-Campbell-Hausdorff Formula

For non-commuting operators $$\hat{A}$$ and $$\hat{B}$$:

$$e^{\hat{A}}e^{\hat{B}} = e^{\hat{A}+\hat{B}+\frac{1}{2}[\hat{A},\hat{B}]+\frac{1}{12}[\hat{A},[\hat{A},\hat{B}]]+\cdots}$$

This shows that $$\hat{R}(\hat{n}_1,\theta_1)\hat{R}(\hat{n}_2,\theta_2) \neq \hat{R}(\hat{n}_1+\hat{n}_2, \theta_1+\theta_2)$$ in general.

### Physical Example: Sequential Rotations

Consider rotating a book:
1. Rotate $$90°$$ about $$x$$, then $$90°$$ about $$y$$
2. Rotate $$90°$$ about $$y$$, then $$90°$$ about $$x$$

The final orientations are different! This demonstrates the non-Abelian nature of SO(3).

---

## SO(3) vs SU(2): The Double Cover

### Spin-1/2 Rotation

For $$j = 1/2$$, the rotation operator takes a special form. Using $$\hat{\mathbf{J}} = \frac{\hbar}{2}\boldsymbol{\sigma}$$:

$$\hat{R}(\hat{n},\theta) = e^{-i\theta\hat{n}\cdot\boldsymbol{\sigma}/2}$$

Using the identity $$(\hat{n}\cdot\boldsymbol{\sigma})^2 = \hat{I}$$:

$$\boxed{\hat{R}(\hat{n},\theta) = \cos\frac{\theta}{2}\hat{I} - i\sin\frac{\theta}{2}(\hat{n}\cdot\boldsymbol{\sigma})}$$

### The 4π Periodicity

For spin-1/2:

$$\hat{R}(\hat{n}, 2\pi) = \cos\pi\,\hat{I} - i\sin\pi(\hat{n}\cdot\boldsymbol{\sigma}) = -\hat{I}$$

A $$2\pi$$ rotation gives a **sign flip**, not the identity! Only a $$4\pi$$ rotation returns to the identity.

This is the mathematical statement that **SU(2) is a double cover of SO(3)**:
- Every SO(3) rotation corresponds to two SU(2) elements: $$+U$$ and $$-U$$
- SU(2) is simply connected; SO(3) is not

### Physical Consequence

This sign flip is physically observable! In neutron interferometry experiments, rotating one path by $$2\pi$$ causes destructive interference, confirming the spinor nature of fermions.

---

## Quantum Computing Connection

### Single-Qubit Gates as Rotations

Every single-qubit unitary operation is an element of SU(2), hence a rotation of the Bloch sphere.

**Pauli Gates:**
$$X = -i\hat{R}_x(\pi) = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

$$Y = -i\hat{R}_y(\pi) = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

$$Z = -i\hat{R}_z(\pi) = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Rotation Gates:**
$$R_x(\theta) = e^{-i\theta X/2} = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_y(\theta) = e^{-i\theta Y/2} = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

### Universal Gate Set

Any single-qubit gate can be written as:

$$U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$$

This is the **ZYZ decomposition** used in quantum compilers.

---

## Worked Examples

### Example 1: Rotation of Spin-1/2 State

**Problem:** A spin-1/2 particle is in state $$|+z\rangle$$. Find the state after rotation by $$\pi/2$$ about the $$y$$-axis.

**Solution:**

The rotation operator is:
$$\hat{R}_y(\pi/2) = \cos\frac{\pi}{4}\hat{I} - i\sin\frac{\pi}{4}\sigma_y$$

$$= \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} - \frac{i}{\sqrt{2}}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

$$= \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$

Acting on $$|+z\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$:

$$|\psi'\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}(|+z\rangle + |-z\rangle) = |+x\rangle$$

$$\boxed{\hat{R}_y(\pi/2)|+z\rangle = |+x\rangle}$$

This makes geometric sense: rotating the $$+z$$ direction by $$90°$$ about $$y$$ gives the $$+x$$ direction.

---

### Example 2: Non-Commutativity Demonstration

**Problem:** Show that $$\hat{R}_x(\pi/2)\hat{R}_z(\pi/2) \neq \hat{R}_z(\pi/2)\hat{R}_x(\pi/2)$$ for spin-1/2.

**Solution:**

Compute each matrix:

$$\hat{R}_x(\pi/2) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -i \\ -i & 1 \end{pmatrix}$$

$$\hat{R}_z(\pi/2) = \begin{pmatrix} e^{-i\pi/4} & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1-i & 0 \\ 0 & 1+i \end{pmatrix}$$

Product $$\hat{R}_x(\pi/2)\hat{R}_z(\pi/2)$$:

$$= \frac{1}{2}\begin{pmatrix} 1 & -i \\ -i & 1 \end{pmatrix}\begin{pmatrix} 1-i & 0 \\ 0 & 1+i \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1-i & -i(1+i) \\ -i(1-i) & 1+i \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 1-i & 1-i \\ -1-i & 1+i \end{pmatrix}$$

Product $$\hat{R}_z(\pi/2)\hat{R}_x(\pi/2)$$:

$$= \frac{1}{2}\begin{pmatrix} 1-i & 0 \\ 0 & 1+i \end{pmatrix}\begin{pmatrix} 1 & -i \\ -i & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1-i & -i(1-i) \\ -i(1+i) & 1+i \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 1-i & -1-i \\ 1-i & 1+i \end{pmatrix}$$

Since the off-diagonal elements differ:

$$\boxed{\hat{R}_x(\pi/2)\hat{R}_z(\pi/2) \neq \hat{R}_z(\pi/2)\hat{R}_x(\pi/2)}$$

---

### Example 3: Rotation Operator for j = 1

**Problem:** Find $$\hat{R}_z(\theta)$$ in the $$|j=1,m\rangle$$ basis.

**Solution:**

The eigenstates are $$|1,1\rangle, |1,0\rangle, |1,-1\rangle$$ with $$\hat{J}_z|1,m\rangle = m\hbar|1,m\rangle$$.

$$\hat{R}_z(\theta) = e^{-i\theta\hat{J}_z/\hbar}$$

Acting on basis states:
$$\hat{R}_z(\theta)|1,m\rangle = e^{-im\theta}|1,m\rangle$$

In matrix form:

$$\boxed{\hat{R}_z(\theta) = \begin{pmatrix} e^{-i\theta} & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & e^{i\theta} \end{pmatrix}}$$

Note: A $$2\pi$$ rotation gives the identity for integer $$j$$, unlike half-integer $$j$$.

---

## Practice Problems

### Level 1: Direct Application

1. **Rotation Matrix Elements:** Calculate $$\hat{R}_y(\pi)$$ for spin-1/2 and show it equals $$-i\sigma_y$$.

2. **State Transformation:** Starting from $$|-z\rangle$$, find the state after $$\hat{R}_x(\pi/2)$$.

3. **Generator Identification:** Verify that $$-i\hbar\frac{d}{d\theta}\hat{R}_z(\theta)\big|_{\theta=0} = \hat{J}_z$$ for $$j = 1$$.

### Level 2: Intermediate

4. **Composition of Rotations:** For spin-1/2, compute $$\hat{R}_z(\alpha)\hat{R}_y(\beta)\hat{R}_z(\gamma)$$ and express the result as a single $$2\times 2$$ matrix.

5. **Hadamard Gate:** Show that the Hadamard gate $$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$ can be written as a rotation about the axis $$\frac{\hat{x}+\hat{z}}{\sqrt{2}}$$ by angle $$\pi$$, up to a global phase.

6. **Rotation of Operator:** If $$\hat{A}$$ is an operator, the rotated operator is $$\hat{A}' = \hat{R}\hat{A}\hat{R}^\dagger$$. Show that $$\hat{R}_z(\theta)\hat{J}_x\hat{R}_z^\dagger(\theta) = \cos\theta\,\hat{J}_x + \sin\theta\,\hat{J}_y$$.

### Level 3: Challenging

7. **Baker-Campbell-Hausdorff:** Use BCH to show that for small angles $$\epsilon$$:
   $$\hat{R}_x(\epsilon)\hat{R}_y(\epsilon)\hat{R}_x(-\epsilon)\hat{R}_y(-\epsilon) \approx \hat{R}_z(-\epsilon^2)$$
   This is the "commutator" of rotations.

8. **Arbitrary Axis Rotation:** Derive the explicit $$2\times 2$$ matrix for $$\hat{R}(\hat{n},\theta)$$ where $$\hat{n} = (\sin\phi, 0, \cos\phi)$$ lies in the $$xz$$-plane.

9. **Double Cover Topology:** Explain why two paths on the rotation group—one being a $$2\pi$$ rotation and the other the identity—cannot be continuously deformed into each other in SO(3), but can in SU(2).

---

## Computational Lab: Rotation Operator Construction

```python
"""
Day 414 Computational Lab: Rotation Operators in Quantum Mechanics
Week 60: Rotations and Wigner D-Matrices
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Part 1: Angular Momentum Matrices
# =============================================================================

def angular_momentum_matrices(j):
    """
    Construct Jx, Jy, Jz matrices for angular momentum j.
    Uses basis |j,m> with m = j, j-1, ..., -j
    """
    dim = int(2*j + 1)
    m_vals = np.arange(j, -j-1, -1)  # j, j-1, ..., -j

    # Jz is diagonal
    Jz = np.diag(m_vals).astype(complex)

    # J+ and J- matrices
    Jplus = np.zeros((dim, dim), dtype=complex)
    Jminus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim-1):
        m = m_vals[i+1]  # m value for lower state
        # J+|j,m> = sqrt(j(j+1)-m(m+1)) |j,m+1>
        Jplus[i, i+1] = np.sqrt(j*(j+1) - m*(m+1))
        # J-|j,m> = sqrt(j(j+1)-m(m-1)) |j,m-1>
        m = m_vals[i]  # m value for upper state
        Jminus[i+1, i] = np.sqrt(j*(j+1) - m*(m-1))

    # Jx = (J+ + J-)/2, Jy = (J+ - J-)/(2i)
    Jx = (Jplus + Jminus) / 2
    Jy = (Jplus - Jminus) / (2j)
    Jy = (Jplus - Jminus) / (2*1j)

    return Jx, Jy, Jz

# Verify for spin-1/2
print("="*60)
print("Angular Momentum Matrices for j = 1/2")
print("="*60)
Jx, Jy, Jz = angular_momentum_matrices(0.5)
print(f"Jx = \n{Jx}")
print(f"\nJy = \n{Jy}")
print(f"\nJz = \n{Jz}")

# Verify commutation relation [Jx, Jy] = i*Jz
commutator = Jx @ Jy - Jy @ Jx
print(f"\n[Jx, Jy] = \n{commutator}")
print(f"i*Jz = \n{1j * Jz}")
print(f"Commutation relation verified: {np.allclose(commutator, 1j*Jz)}")

# =============================================================================
# Part 2: Rotation Operators
# =============================================================================

def rotation_operator(j, axis, theta):
    """
    Construct rotation operator R(axis, theta) = exp(-i*theta*J_axis)
    axis: 'x', 'y', 'z' or unit vector [nx, ny, nz]
    """
    Jx, Jy, Jz = angular_momentum_matrices(j)

    if isinstance(axis, str):
        if axis == 'x':
            J = Jx
        elif axis == 'y':
            J = Jy
        elif axis == 'z':
            J = Jz
    else:
        # axis is a unit vector
        n = np.array(axis)
        n = n / np.linalg.norm(n)
        J = n[0]*Jx + n[1]*Jy + n[2]*Jz

    return expm(-1j * theta * J)

# Rotation of spin-1/2 about y-axis by pi/2
print("\n" + "="*60)
print("Rotation Operators for Spin-1/2")
print("="*60)

Ry_pi2 = rotation_operator(0.5, 'y', np.pi/2)
print(f"R_y(pi/2) for j=1/2:\n{Ry_pi2}")

# Apply to |+z> state
plus_z = np.array([1, 0], dtype=complex)
rotated = Ry_pi2 @ plus_z
print(f"\nR_y(pi/2)|+z> = {rotated}")
print(f"This is |+x> = (|+z> + |-z>)/sqrt(2)")

# Verify: rotation by 2*pi gives -I for spin-1/2
R_2pi = rotation_operator(0.5, 'z', 2*np.pi)
print(f"\nR_z(2*pi) for j=1/2:\n{R_2pi}")
print(f"This equals -I: {np.allclose(R_2pi, -np.eye(2))}")

# Rotation by 4*pi gives +I
R_4pi = rotation_operator(0.5, 'z', 4*np.pi)
print(f"\nR_z(4*pi) for j=1/2:\n{R_4pi}")
print(f"This equals +I: {np.allclose(R_4pi, np.eye(2))}")

# =============================================================================
# Part 3: Non-Commutativity Demonstration
# =============================================================================

print("\n" + "="*60)
print("Non-Commutativity of Rotations")
print("="*60)

# For spin-1/2
Rx = rotation_operator(0.5, 'x', np.pi/2)
Rz = rotation_operator(0.5, 'z', np.pi/2)

product1 = Rx @ Rz
product2 = Rz @ Rx

print(f"R_x(pi/2) @ R_z(pi/2) =\n{product1}")
print(f"\nR_z(pi/2) @ R_x(pi/2) =\n{product2}")
print(f"\nAre they equal? {np.allclose(product1, product2)}")

# Difference
print(f"\nDifference:\n{product1 - product2}")

# =============================================================================
# Part 4: Visualization - Bloch Sphere Rotations
# =============================================================================

def bloch_vector(state):
    """Convert a 2D quantum state to Bloch sphere coordinates."""
    rho = np.outer(state, np.conj(state))

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])

    x = np.real(np.trace(rho @ sx))
    y = np.real(np.trace(rho @ sy))
    z = np.real(np.trace(rho @ sz))

    return np.array([x, y, z])

# Create figure for Bloch sphere visualization
fig = plt.figure(figsize=(14, 5))

# Subplot 1: Rotation about z-axis
ax1 = fig.add_subplot(131, projection='3d')

# Draw Bloch sphere
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 30)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

# Initial state: |+x>
state = np.array([1, 1], dtype=complex) / np.sqrt(2)
trajectory = []

# Rotate about z-axis
for theta in np.linspace(0, 2*np.pi, 100):
    R = rotation_operator(0.5, 'z', theta)
    rotated_state = R @ state
    trajectory.append(bloch_vector(rotated_state))

trajectory = np.array(trajectory)
ax1.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'b-', linewidth=2)
ax1.scatter([trajectory[0,0]], [trajectory[0,1]], [trajectory[0,2]],
            color='green', s=100, label='Start')
ax1.scatter([trajectory[-1,0]], [trajectory[-1,1]], [trajectory[-1,2]],
            color='red', s=100, label='End')
ax1.set_title('Rotation about z-axis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Subplot 2: Rotation about y-axis
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

state = np.array([1, 0], dtype=complex)  # |+z>
trajectory = []

for theta in np.linspace(0, 2*np.pi, 100):
    R = rotation_operator(0.5, 'y', theta)
    rotated_state = R @ state
    trajectory.append(bloch_vector(rotated_state))

trajectory = np.array(trajectory)
ax2.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'r-', linewidth=2)
ax2.scatter([trajectory[0,0]], [trajectory[0,1]], [trajectory[0,2]],
            color='green', s=100, label='Start')
ax2.set_title('Rotation about y-axis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

# Subplot 3: Arbitrary axis rotation
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

# Rotation axis: (1,1,1)/sqrt(3)
axis = np.array([1, 1, 1]) / np.sqrt(3)
state = np.array([1, 0], dtype=complex)  # |+z>
trajectory = []

for theta in np.linspace(0, 2*np.pi, 100):
    R = rotation_operator(0.5, axis, theta)
    rotated_state = R @ state
    trajectory.append(bloch_vector(rotated_state))

trajectory = np.array(trajectory)
ax3.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'g-', linewidth=2)

# Draw rotation axis
ax3.quiver(0, 0, 0, axis[0], axis[1], axis[2], color='purple',
           arrow_length_ratio=0.1, linewidth=2, label='Rotation axis')
ax3.scatter([trajectory[0,0]], [trajectory[0,1]], [trajectory[0,2]],
            color='green', s=100, label='Start')
ax3.set_title('Rotation about (1,1,1)/sqrt(3)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.legend()

plt.tight_layout()
plt.savefig('bloch_rotations.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 5: Quantum Gates as Rotations
# =============================================================================

print("\n" + "="*60)
print("Quantum Gates as SU(2) Rotations")
print("="*60)

# Pauli X gate
X = np.array([[0, 1], [1, 0]], dtype=complex)
Rx_pi = rotation_operator(0.5, 'x', np.pi)
print(f"X gate:\n{X}")
print(f"\nR_x(pi) (up to phase):\n{Rx_pi}")
print(f"R_x(pi) = -i*X: {np.allclose(Rx_pi, -1j*X)}")

# Hadamard gate
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
# H = exp(-i*pi/2) * R_(x+z)(pi)
axis_H = np.array([1, 0, 1]) / np.sqrt(2)
R_H = rotation_operator(0.5, axis_H, np.pi)
phase = np.exp(-1j*np.pi/2)
print(f"\nHadamard gate:\n{H}")
print(f"\nR_((x+z)/sqrt(2))(pi):\n{R_H}")
print(f"H = e^(-i*pi/2)*R: {np.allclose(H, phase*R_H)}")

# =============================================================================
# Part 6: Higher Spin Rotations
# =============================================================================

print("\n" + "="*60)
print("Rotation Matrices for j = 1")
print("="*60)

Rz_j1 = rotation_operator(1, 'z', np.pi/4)
print(f"R_z(pi/4) for j=1:\n{Rz_j1}")

Ry_j1 = rotation_operator(1, 'y', np.pi/2)
print(f"\nR_y(pi/2) for j=1:\n{np.round(Ry_j1, 4)}")

# Verify R_z(2*pi) = I for j=1
R_2pi_j1 = rotation_operator(1, 'z', 2*np.pi)
print(f"\nR_z(2*pi) for j=1:\n{np.round(R_2pi_j1, 4)}")
print(f"This equals +I: {np.allclose(R_2pi_j1, np.eye(3))}")

print("\n" + "="*60)
print("Computational Lab Complete")
print("="*60)
```

---

## Summary

### Key Concepts

| Concept | Mathematical Form | Physical Meaning |
|---------|-------------------|------------------|
| Rotation operator | $$\hat{R}(\hat{n},\theta) = e^{-i\theta\hat{n}\cdot\hat{\mathbf{J}}/\hbar}$$ | Unitary transformation rotating quantum states |
| Infinitesimal rotation | $$\hat{R}(\hat{n},d\theta) \approx \hat{I} - \frac{i}{\hbar}d\theta\,\hat{n}\cdot\hat{\mathbf{J}}$$ | Angular momentum generates rotations |
| Non-commutativity | $$[\hat{R}_i, \hat{R}_j] \neq 0$$ | Order of rotations matters |
| Spin-1/2 rotation | $$\cos\frac{\theta}{2}\hat{I} - i\sin\frac{\theta}{2}(\hat{n}\cdot\boldsymbol{\sigma})$$ | $$4\pi$$ periodicity (SU(2)) |
| Integer spin rotation | $$e^{-im\theta}|j,m\rangle$$ | $$2\pi$$ periodicity (SO(3)) |

### Fundamental Relationships

$$\boxed{\text{Angular Momentum} \leftrightarrow \text{Rotation Generator}}$$

$$\boxed{\text{SU(2) is double cover of SO(3)}}$$

$$\boxed{\text{Single-qubit gates} = \text{SU(2) elements} = \text{Bloch sphere rotations}}$$

---

## Daily Checklist

- [ ] I can write the rotation operator as a matrix exponential
- [ ] I understand why angular momentum is the generator of rotations
- [ ] I can demonstrate the non-commutativity of rotations
- [ ] I understand the $$4\pi$$ periodicity for half-integer spin
- [ ] I can connect rotation operators to single-qubit gates
- [ ] I have completed the computational lab

---

## Preview: Day 415

Tomorrow we introduce **Euler angles**, the standard parameterization for rotations:

$$R(\alpha,\beta,\gamma) = R_z(\alpha)R_y(\beta)R_z(\gamma)$$

We will explore:
- Physical interpretation of each angle
- Gimbal lock and its implications
- Converting between axis-angle and Euler representations
- The measure on rotation space

---

*Day 414 of 2184 | Week 60 of 312 | Month 15 of 72*
