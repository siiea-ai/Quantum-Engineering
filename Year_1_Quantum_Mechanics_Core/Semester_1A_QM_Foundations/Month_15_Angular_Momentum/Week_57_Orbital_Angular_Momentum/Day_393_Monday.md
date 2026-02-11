# Day 393: Classical to Quantum Angular Momentum

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Classical angular momentum, quantization procedure |
| **Afternoon** | 2.5 hours | Component operators, physical interpretation |
| **Evening** | 1.5 hours | Computational lab: angular momentum visualization |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Express** classical angular momentum as $\mathbf{L} = \mathbf{r} \times \mathbf{p}$ and interpret its physical meaning
2. **Apply** canonical quantization to derive the quantum angular momentum operator $\hat{\mathbf{L}} = -i\hbar(\mathbf{r} \times \nabla)$
3. **Write** explicit forms of $\hat{L}_x$, $\hat{L}_y$, $\hat{L}_z$ in Cartesian coordinates
4. **Transform** angular momentum operators to spherical coordinates
5. **Explain** the physical interpretation of angular momentum in quantum systems
6. **Visualize** angular momentum vectors and their quantum mechanical behavior

---

## 1. Classical Angular Momentum Review

### 1.1 Definition and Properties

In classical mechanics, the **angular momentum** of a particle relative to an origin is defined as:

$$\boxed{\mathbf{L} = \mathbf{r} \times \mathbf{p}}$$

where $\mathbf{r}$ is the position vector and $\mathbf{p} = m\mathbf{v}$ is the linear momentum.

**In component form:**

$$L_x = yp_z - zp_y$$
$$L_y = zp_x - xp_z$$
$$L_z = xp_y - yp_x$$

Or using the Levi-Civita symbol:

$$L_i = \epsilon_{ijk}x_jp_k$$

where $\epsilon_{ijk}$ is the fully antisymmetric tensor:
- $\epsilon_{123} = \epsilon_{231} = \epsilon_{312} = +1$
- $\epsilon_{132} = \epsilon_{213} = \epsilon_{321} = -1$
- All other components = 0

### 1.2 Conservation Law

Angular momentum is conserved when the net torque is zero:

$$\frac{d\mathbf{L}}{dt} = \boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}$$

For a **central force** $\mathbf{F} = F(r)\hat{r}$, the torque vanishes, and angular momentum is conserved.

### 1.3 Magnitude and Direction

The magnitude of angular momentum:

$$|\mathbf{L}| = |\mathbf{r}||\mathbf{p}|\sin\theta = mvr\sin\theta$$

The direction is perpendicular to the plane of motion (right-hand rule).

---

## 2. Canonical Quantization

### 2.1 The Quantization Recipe

To quantize classical mechanics, we replace classical observables with operators following the **canonical quantization prescription**:

$$\boxed{x \to \hat{x}, \quad p \to \hat{p} = -i\hbar\nabla}$$

with the fundamental commutation relation:

$$[\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}$$

### 2.2 Angular Momentum Operators

Applying canonical quantization to $\mathbf{L} = \mathbf{r} \times \mathbf{p}$:

$$\hat{\mathbf{L}} = \hat{\mathbf{r}} \times \hat{\mathbf{p}} = -i\hbar(\mathbf{r} \times \nabla)$$

**The component operators in Cartesian coordinates:**

$$\boxed{\hat{L}_x = \hat{y}\hat{p}_z - \hat{z}\hat{p}_y = -i\hbar\left(y\frac{\partial}{\partial z} - z\frac{\partial}{\partial y}\right)}$$

$$\boxed{\hat{L}_y = \hat{z}\hat{p}_x - \hat{x}\hat{p}_z = -i\hbar\left(z\frac{\partial}{\partial x} - x\frac{\partial}{\partial z}\right)}$$

$$\boxed{\hat{L}_z = \hat{x}\hat{p}_y - \hat{y}\hat{p}_x = -i\hbar\left(x\frac{\partial}{\partial y} - y\frac{\partial}{\partial x}\right)}$$

### 2.3 Hermiticity of Angular Momentum

Each component is Hermitian: $\hat{L}_i^\dagger = \hat{L}_i$.

**Proof for $\hat{L}_z$:**

$$\hat{L}_z^\dagger = (\hat{x}\hat{p}_y - \hat{y}\hat{p}_x)^\dagger = \hat{p}_y^\dagger\hat{x}^\dagger - \hat{p}_x^\dagger\hat{y}^\dagger = \hat{p}_y\hat{x} - \hat{p}_x\hat{y}$$

Since $[\hat{x}, \hat{p}_y] = 0$ (different components commute):

$$\hat{L}_z^\dagger = \hat{x}\hat{p}_y - \hat{y}\hat{p}_x = \hat{L}_z \quad \checkmark$$

---

## 3. Spherical Coordinate Representation

### 3.1 Coordinate Transformation

The transformation from Cartesian to spherical coordinates:

$$x = r\sin\theta\cos\phi$$
$$y = r\sin\theta\sin\phi$$
$$z = r\cos\theta$$

Inverse relations:
$$r = \sqrt{x^2 + y^2 + z^2}$$
$$\theta = \arccos(z/r)$$
$$\phi = \arctan(y/x)$$

### 3.2 Gradient in Spherical Coordinates

$$\nabla = \hat{r}\frac{\partial}{\partial r} + \hat{\theta}\frac{1}{r}\frac{\partial}{\partial\theta} + \hat{\phi}\frac{1}{r\sin\theta}\frac{\partial}{\partial\phi}$$

### 3.3 Angular Momentum in Spherical Coordinates

After a detailed transformation (see derivation below), we obtain:

$$\boxed{\hat{L}_z = -i\hbar\frac{\partial}{\partial\phi}}$$

$$\boxed{\hat{L}_x = i\hbar\left(\sin\phi\frac{\partial}{\partial\theta} + \cot\theta\cos\phi\frac{\partial}{\partial\phi}\right)}$$

$$\boxed{\hat{L}_y = i\hbar\left(-\cos\phi\frac{\partial}{\partial\theta} + \cot\theta\sin\phi\frac{\partial}{\partial\phi}\right)}$$

**The $\hat{L}_z$ Derivation:**

Starting from $\hat{L}_z = -i\hbar(x\partial_y - y\partial_x)$, we use the chain rule:

$$\frac{\partial}{\partial x} = \frac{\partial r}{\partial x}\frac{\partial}{\partial r} + \frac{\partial\theta}{\partial x}\frac{\partial}{\partial\theta} + \frac{\partial\phi}{\partial x}\frac{\partial}{\partial\phi}$$

For $\phi = \arctan(y/x)$:

$$\frac{\partial\phi}{\partial x} = -\frac{y}{x^2 + y^2} = -\frac{\sin\phi}{r\sin\theta}$$

$$\frac{\partial\phi}{\partial y} = \frac{x}{x^2 + y^2} = \frac{\cos\phi}{r\sin\theta}$$

Therefore:

$$x\frac{\partial}{\partial y} - y\frac{\partial}{\partial x} = \frac{\partial}{\partial\phi}$$

Thus: $\hat{L}_z = -i\hbar\frac{\partial}{\partial\phi}$

### 3.4 The Total Angular Momentum Squared

$$\boxed{\hat{L}^2 = \hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2 = -\hbar^2\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]}$$

This is precisely the **angular part of the Laplacian** in spherical coordinates:

$$\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial}{\partial r}\right) + \frac{1}{r^2}\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]$$

So:

$$\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial}{\partial r}\right) - \frac{\hat{L}^2}{\hbar^2 r^2}$$

---

## 4. Physical Interpretation

### 4.1 Angular Momentum as Generator of Rotations

Angular momentum generates infinitesimal rotations. For a rotation by angle $d\phi$ about the z-axis:

$$\psi(\mathbf{r}) \to \psi(R_z^{-1}\mathbf{r}) = \psi(x\cos(d\phi) + y\sin(d\phi), -x\sin(d\phi) + y\cos(d\phi), z)$$

Expanding to first order:

$$\psi \to \psi + d\phi\left(y\frac{\partial\psi}{\partial x} - x\frac{\partial\psi}{\partial y}\right) = \psi + \frac{i}{\hbar}d\phi\,\hat{L}_z\psi$$

Thus: $\hat{U}(d\phi) = 1 + \frac{i}{\hbar}d\phi\,\hat{L}_z$

For finite rotation:

$$\boxed{\hat{U}(\phi) = e^{i\phi\hat{L}_z/\hbar}}$$

### 4.2 Orbital vs. Spin Angular Momentum

**Orbital angular momentum** $\hat{\mathbf{L}}$:
- Arises from spatial motion
- Depends on position and momentum operators
- Has integer eigenvalues only: $\ell = 0, 1, 2, \ldots$

**Spin angular momentum** $\hat{\mathbf{S}}$ (Week 58):
- Intrinsic property with no classical analog
- Independent of spatial coordinates
- Can have half-integer values: $s = 1/2, 3/2, \ldots$

### 4.3 Uncertainty Relations

Since $[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$ (proven tomorrow), we have:

$$\Delta L_x \Delta L_y \geq \frac{\hbar}{2}|\langle\hat{L}_z\rangle|$$

**Key insight:** We cannot simultaneously know all three components of angular momentum precisely (unless $\langle\hat{L}\rangle = 0$).

---

## 5. Quantum Mechanics Connections

### 5.1 Central Potential Problems

For any central potential $V(r)$, angular momentum is conserved:

$$[\hat{H}, \hat{L}_i] = 0 \quad \text{for all } i$$

This means we can find simultaneous eigenstates of $\hat{H}$, $\hat{L}^2$, and $\hat{L}_z$.

### 5.2 Connection to Atomic Structure

The quantization of angular momentum explains atomic orbital structure:

| $\ell$ | Orbital Name | Degeneracy $2\ell+1$ |
|--------|--------------|----------------------|
| 0 | s | 1 |
| 1 | p | 3 |
| 2 | d | 5 |
| 3 | f | 7 |

### 5.3 Quantum Computing Connection

In quantum computing, the **Bloch sphere** representation uses angular momentum language:

$$|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$$

Rotation gates are generated by Pauli matrices, which satisfy angular momentum-like commutation relations:

$$[\sigma_x, \sigma_y] = 2i\sigma_z$$

This is exactly the spin-1/2 angular momentum algebra (up to a factor of 2).

---

## 6. Worked Examples

### Example 1: Verify $\hat{L}_z$ Action on a Function

**Problem:** Apply $\hat{L}_z$ to $\psi = e^{im\phi}$.

**Solution:**

$$\hat{L}_z\psi = -i\hbar\frac{\partial}{\partial\phi}e^{im\phi} = -i\hbar \cdot im \cdot e^{im\phi} = m\hbar e^{im\phi}$$

$$\boxed{\hat{L}_z e^{im\phi} = m\hbar e^{im\phi}}$$

Therefore, $e^{im\phi}$ is an eigenfunction of $\hat{L}_z$ with eigenvalue $m\hbar$.

For single-valuedness: $\psi(\phi + 2\pi) = \psi(\phi)$, so $e^{im(\phi + 2\pi)} = e^{im\phi}$, requiring $m \in \mathbb{Z}$.

---

### Example 2: Angular Momentum of the Hydrogen Ground State

**Problem:** Calculate $\langle\hat{L}^2\rangle$ and $\langle\hat{L}_z\rangle$ for the hydrogen ground state $\psi_{100} = \frac{1}{\sqrt{\pi}a_0^{3/2}}e^{-r/a_0}$.

**Solution:**

The ground state has $\ell = 0$, $m = 0$, so it is spherically symmetric (no angular dependence).

$$\hat{L}^2\psi_{100} = -\hbar^2\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]\psi_{100}$$

Since $\psi_{100}$ has no $\theta$ or $\phi$ dependence:

$$\hat{L}^2\psi_{100} = 0$$

$$\boxed{\langle\hat{L}^2\rangle = 0, \quad \langle\hat{L}_z\rangle = 0}$$

The s-orbital has zero angular momentum.

---

### Example 3: Computing $[\hat{L}_z, \hat{x}]$

**Problem:** Calculate the commutator $[\hat{L}_z, \hat{x}]$.

**Solution:**

$$[\hat{L}_z, \hat{x}] = [\hat{x}\hat{p}_y - \hat{y}\hat{p}_x, \hat{x}]$$

Using $[\hat{A}\hat{B}, \hat{C}] = \hat{A}[\hat{B}, \hat{C}] + [\hat{A}, \hat{C}]\hat{B}$:

$$= \hat{x}[\hat{p}_y, \hat{x}] + [\hat{x}, \hat{x}]\hat{p}_y - \hat{y}[\hat{p}_x, \hat{x}] - [\hat{y}, \hat{x}]\hat{p}_x$$

Using $[\hat{p}_y, \hat{x}] = 0$, $[\hat{x}, \hat{x}] = 0$, $[\hat{p}_x, \hat{x}] = -i\hbar$, $[\hat{y}, \hat{x}] = 0$:

$$= 0 + 0 - \hat{y}(-i\hbar) - 0 = i\hbar\hat{y}$$

$$\boxed{[\hat{L}_z, \hat{x}] = i\hbar\hat{y}}$$

By cyclic permutation: $[\hat{L}_x, \hat{y}] = i\hbar\hat{z}$, $[\hat{L}_y, \hat{z}] = i\hbar\hat{x}$, etc.

---

## 7. Practice Problems

### Level 1: Direct Application

1. **Problem 1.1:** Write out $\hat{L}_y$ explicitly in Cartesian coordinates and verify it is Hermitian.

2. **Problem 1.2:** Apply $\hat{L}_z$ to $\psi = \sin\theta\cos\phi$. Express your answer in terms of known functions.

3. **Problem 1.3:** Calculate $[\hat{L}_z, \hat{y}]$ and $[\hat{L}_z, \hat{z}]$.

### Level 2: Intermediate

4. **Problem 2.1:** Show that $[\hat{L}_z, \hat{r}] = 0$ where $\hat{r} = \sqrt{\hat{x}^2 + \hat{y}^2 + \hat{z}^2}$.

5. **Problem 2.2:** Prove that $[\hat{L}_z, \hat{p}_x] = i\hbar\hat{p}_y$ and $[\hat{L}_z, \hat{p}_y] = -i\hbar\hat{p}_x$.

6. **Problem 2.3:** For a 2D rotor with $\psi(\phi) = A(\cos\phi + i\sin\phi)$, find the normalization constant $A$ and calculate $\langle\hat{L}_z\rangle$.

### Level 3: Challenging

7. **Problem 3.1:** Derive the expression for $\hat{L}_x$ in spherical coordinates starting from the Cartesian form.

8. **Problem 3.2:** Show that $[\hat{L}_i, \hat{r}_j] = i\hbar\epsilon_{ijk}\hat{r}_k$ (angular momentum components rotate position operators).

9. **Problem 3.3:** Consider a state $\psi(\phi) = A(e^{i\phi} + 2e^{-i\phi})$. Find $A$, $\langle\hat{L}_z\rangle$, $\langle\hat{L}_z^2\rangle$, and $\Delta L_z$.

---

## 8. Computational Lab: Visualizing Angular Momentum

```python
"""
Day 393 Computational Lab: Classical to Quantum Angular Momentum
================================================================
Visualization of angular momentum concepts in classical and quantum mechanics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Set up publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 10),
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11
})

# =============================================================================
# Part 1: Classical Angular Momentum Visualization
# =============================================================================

def classical_angular_momentum():
    """
    Visualize classical angular momentum L = r x p for a circular orbit.
    """
    fig = plt.figure(figsize=(14, 5))

    # Circular orbit in xy-plane
    ax1 = fig.add_subplot(131, projection='3d')

    t = np.linspace(0, 2*np.pi, 100)
    r = 2  # radius
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.zeros_like(t)

    ax1.plot(x, y, z, 'b-', linewidth=2, label='Orbit')

    # Position and momentum at one point
    t0 = np.pi/4
    r_vec = np.array([r*np.cos(t0), r*np.sin(t0), 0])
    p_vec = np.array([-np.sin(t0), np.cos(t0), 0])  # tangent direction
    L_vec = np.cross(r_vec, p_vec)

    # Draw vectors
    ax1.quiver(0, 0, 0, *r_vec, color='r', arrow_length_ratio=0.1,
               linewidth=2, label='r')
    ax1.quiver(*r_vec, *p_vec, color='g', arrow_length_ratio=0.1,
               linewidth=2, label='p')
    ax1.quiver(0, 0, 0, *L_vec*2, color='purple', arrow_length_ratio=0.1,
               linewidth=3, label='L = r x p')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Classical Angular Momentum\nL = r × p')
    ax1.legend()
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.set_zlim([-3, 3])

    # Tilted orbit
    ax2 = fig.add_subplot(132, projection='3d')

    # Rotation matrix for tilted orbit
    theta_tilt = np.pi/4
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_tilt), -np.sin(theta_tilt)],
        [0, np.sin(theta_tilt), np.cos(theta_tilt)]
    ])

    orbit_tilted = np.array([Rx @ np.array([r*np.cos(ti), r*np.sin(ti), 0])
                             for ti in t])

    ax2.plot(orbit_tilted[:, 0], orbit_tilted[:, 1], orbit_tilted[:, 2],
             'b-', linewidth=2)

    # Angular momentum for tilted orbit
    L_tilted = Rx @ np.array([0, 0, 1])
    ax2.quiver(0, 0, 0, *L_tilted*2, color='purple', arrow_length_ratio=0.1,
               linewidth=3, label='L')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('Tilted Orbit\nL perpendicular to orbital plane')
    ax2.legend()
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.set_zlim([-3, 3])

    # Multiple orbits with different L
    ax3 = fig.add_subplot(133, projection='3d')

    colors = ['b', 'r', 'g']
    angles = [0, np.pi/6, np.pi/3]

    for color, angle in zip(colors, angles):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        orbit = np.array([Rx @ np.array([r*np.cos(ti), r*np.sin(ti), 0])
                          for ti in t])
        L = Rx @ np.array([0, 0, 1])

        ax3.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2],
                 color=color, linewidth=1, alpha=0.5)
        ax3.quiver(0, 0, 0, *L*2, color=color, arrow_length_ratio=0.1,
                   linewidth=2)

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('Different Orientations\nof Angular Momentum')
    ax3.set_xlim([-3, 3])
    ax3.set_ylim([-3, 3])
    ax3.set_zlim([-3, 3])

    plt.tight_layout()
    plt.savefig('classical_angular_momentum.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 2: Lz Eigenfunctions Visualization
# =============================================================================

def Lz_eigenfunctions():
    """
    Visualize eigenfunctions of Lz operator: e^(im*phi).
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    phi = np.linspace(0, 2*np.pi, 200)
    m_values = [-2, -1, 0, 1, 2, 3]

    for idx, m in enumerate(m_values):
        ax = axes[idx // 3, idx % 3]

        psi = np.exp(1j * m * phi)

        # Plot real and imaginary parts
        ax.plot(phi, np.real(psi), 'b-', linewidth=2, label='Re$(e^{im\\phi})$')
        ax.plot(phi, np.imag(psi), 'r--', linewidth=2, label='Im$(e^{im\\phi})$')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        ax.set_xlabel('$\\phi$')
        ax.set_ylabel('$\\psi$')
        ax.set_title(f'$m = {m}$, $L_z = {m}\\hbar$')
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([-1.5, 1.5])
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Eigenfunctions of $\\hat{L}_z$: $\\psi_m(\\phi) = e^{im\\phi}$',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('Lz_eigenfunctions.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 3: Polar Representation of Lz Eigenfunctions
# =============================================================================

def Lz_polar_representation():
    """
    Polar plot of |psi|^2 for different angular momentum states.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4),
                             subplot_kw={'projection': 'polar'})

    phi = np.linspace(0, 2*np.pi, 200)

    # Different superposition states
    states = [
        ('$|m=0\\rangle$', lambda p: np.ones_like(p)),
        ('$|m=1\\rangle$', lambda p: np.exp(1j*p)),
        ('$\\frac{1}{\\sqrt{2}}(|1\\rangle + |-1\\rangle)$',
         lambda p: (np.exp(1j*p) + np.exp(-1j*p))/np.sqrt(2)),
        ('$\\frac{1}{\\sqrt{2}}(|2\\rangle + |-2\\rangle)$',
         lambda p: (np.exp(2j*p) + np.exp(-2j*p))/np.sqrt(2))
    ]

    for ax, (label, state_func) in zip(axes, states):
        psi = state_func(phi)
        prob = np.abs(psi)**2

        ax.plot(phi, prob, 'b-', linewidth=2)
        ax.fill(phi, prob, alpha=0.3)
        ax.set_title(label, fontsize=12)
        ax.set_rticks([0.5, 1.0, 1.5, 2.0])

    plt.suptitle('Probability Density $|\\psi(\\phi)|^2$ in Polar Coordinates',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('Lz_polar.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 4: Rotation Generated by Lz
# =============================================================================

def rotation_visualization():
    """
    Demonstrate that Lz generates rotations about the z-axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Initial wavefunction: Gaussian centered at phi=0
    phi = np.linspace(-np.pi, np.pi, 200)
    sigma = 0.3
    psi0 = np.exp(-phi**2 / (2*sigma**2))
    psi0 = psi0 / np.sqrt(np.trapezoid(np.abs(psi0)**2, phi))  # Normalize

    # Rotation angles
    rotation_angles = [0, np.pi/4, np.pi/2]

    for ax, alpha in zip(axes, rotation_angles):
        # After rotation by alpha, the wavefunction shifts
        psi_rotated = np.exp(-((phi - alpha)**2) / (2*sigma**2))
        psi_rotated = psi_rotated / np.sqrt(np.trapezoid(np.abs(psi_rotated)**2, phi))

        ax.fill_between(phi, 0, np.abs(psi0)**2, alpha=0.3, color='blue',
                        label='Initial $|\\psi|^2$')
        ax.plot(phi, np.abs(psi_rotated)**2, 'r-', linewidth=2,
                label=f'After rotation by $\\alpha={alpha:.2f}$')

        ax.set_xlabel('$\\phi$')
        ax.set_ylabel('$|\\psi|^2$')
        ax.set_title(f'$\\hat{{U}} = e^{{i\\alpha\\hat{{L}}_z/\\hbar}}$\n$\\alpha = {alpha/np.pi:.2f}\\pi$')
        ax.legend()
        ax.set_xlim([-np.pi, np.pi])
        ax.grid(True, alpha=0.3)

    plt.suptitle('$\\hat{L}_z$ as Generator of Rotations about z-axis', fontsize=14)
    plt.tight_layout()
    plt.savefig('Lz_rotation.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 5: Matrix Representation of Angular Momentum
# =============================================================================

def Lz_matrix_representation():
    """
    Matrix representation of Lz in the |l,m> basis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, l in zip(axes, [1, 2, 3]):
        dim = 2*l + 1
        Lz_matrix = np.diag(np.arange(-l, l+1)[::-1])  # m values from l to -l

        im = ax.imshow(Lz_matrix, cmap='RdBu_r', aspect='equal',
                       vmin=-l, vmax=l)

        # Add value annotations
        for i in range(dim):
            for j in range(dim):
                val = Lz_matrix[i, j]
                if val != 0:
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                            fontsize=12, fontweight='bold')

        ax.set_title(f'$\\ell = {l}$\n$\\hat{{L}}_z/\\hbar$ matrix ({dim}×{dim})')
        ax.set_xticks(range(dim))
        ax.set_yticks(range(dim))
        ax.set_xticklabels([f'$|{l},{m}\\rangle$' for m in range(l, -l-1, -1)],
                           fontsize=8, rotation=45)
        ax.set_yticklabels([f'$\\langle{l},{m}|$' for m in range(l, -l-1, -1)],
                           fontsize=8)

        plt.colorbar(im, ax=ax, label='$m$ eigenvalue')

    plt.suptitle('Matrix Representation of $\\hat{L}_z$ in $|\\ell,m\\rangle$ Basis',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('Lz_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 6: Quantum vs Classical Angular Momentum Comparison
# =============================================================================

def quantum_vs_classical():
    """
    Compare classical and quantum angular momentum properties.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Classical: continuous values
    ax1 = axes[0]
    L_classical = np.linspace(0, 5, 100)
    ax1.fill_between(L_classical, 0, np.ones_like(L_classical),
                     alpha=0.3, color='blue')
    ax1.axhline(y=1, color='blue', linestyle='-', linewidth=2)
    ax1.set_xlabel('Angular Momentum $L$ (units of $\\hbar$)')
    ax1.set_ylabel('Allowed?')
    ax1.set_title('Classical: Continuous Values')
    ax1.set_ylim([0, 1.5])
    ax1.text(2.5, 0.5, 'Any value allowed', fontsize=12, ha='center')

    # Quantum: discrete values
    ax2 = axes[1]
    l_values = [0, 1, 2, 3, 4]
    L_squared_values = [l*(l+1) for l in l_values]
    L_values = [np.sqrt(l*(l+1)) for l in l_values]

    for l, L in zip(l_values, L_values):
        ax2.vlines(L, 0, 1, colors='red', linewidth=3)
        ax2.plot(L, 1, 'ro', markersize=10)
        ax2.annotate(f'$\\ell={l}$\n$L=\\sqrt{{{l}({l}+1)}}\\hbar$',
                     (L, 1.05), ha='center', fontsize=9)

    ax2.set_xlabel('Angular Momentum $|\\mathbf{L}|$ (units of $\\hbar$)')
    ax2.set_ylabel('Allowed?')
    ax2.set_title('Quantum: Discrete Values')
    ax2.set_ylim([0, 1.5])
    ax2.set_xlim([-0.5, 5])

    plt.suptitle('Classical vs Quantum Angular Momentum', fontsize=14)
    plt.tight_layout()
    plt.savefig('classical_vs_quantum_L.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Day 393 Lab: Classical to Quantum Angular Momentum")
    print("=" * 55)

    print("\n1. Classical Angular Momentum Visualization...")
    classical_angular_momentum()

    print("\n2. Lz Eigenfunctions...")
    Lz_eigenfunctions()

    print("\n3. Polar Representation...")
    Lz_polar_representation()

    print("\n4. Rotation Generated by Lz...")
    rotation_visualization()

    print("\n5. Matrix Representation of Lz...")
    Lz_matrix_representation()

    print("\n6. Quantum vs Classical Comparison...")
    quantum_vs_classical()

    print("\nLab complete! Figures saved to current directory.")
```

---

## 9. Summary

### Key Formulas Table

| Concept | Formula |
|---------|---------|
| Classical angular momentum | $\mathbf{L} = \mathbf{r} \times \mathbf{p}$ |
| Quantum $\hat{L}_x$ | $-i\hbar\left(y\frac{\partial}{\partial z} - z\frac{\partial}{\partial y}\right)$ |
| Quantum $\hat{L}_y$ | $-i\hbar\left(z\frac{\partial}{\partial x} - x\frac{\partial}{\partial z}\right)$ |
| Quantum $\hat{L}_z$ | $-i\hbar\frac{\partial}{\partial\phi}$ |
| Rotation generator | $\hat{U}(\phi) = e^{i\phi\hat{L}_z/\hbar}$ |
| $L_z$ eigenvalue equation | $\hat{L}_z e^{im\phi} = m\hbar e^{im\phi}$ |

### Main Takeaways

1. **Canonical quantization** transforms $\mathbf{L} = \mathbf{r} \times \mathbf{p}$ into $\hat{\mathbf{L}} = -i\hbar(\mathbf{r} \times \nabla)$

2. **Spherical coordinates** simplify angular momentum: $\hat{L}_z = -i\hbar\frac{\partial}{\partial\phi}$

3. **$\hat{L}_z$ eigenfunctions** are $e^{im\phi}$ with $m \in \mathbb{Z}$ (single-valuedness)

4. **Angular momentum generates rotations**: $\hat{U}(\phi) = e^{i\phi\hat{L}_z/\hbar}$

5. **All components are Hermitian** and represent physical observables

---

## 10. Daily Checklist

- [ ] Derived angular momentum operators from canonical quantization
- [ ] Wrote $\hat{L}_x$, $\hat{L}_y$, $\hat{L}_z$ in Cartesian coordinates
- [ ] Transformed $\hat{L}_z$ to spherical coordinates
- [ ] Verified eigenfunctions of $\hat{L}_z$ are $e^{im\phi}$
- [ ] Understood angular momentum as generator of rotations
- [ ] Completed computational lab visualizations
- [ ] Solved at least 3 practice problems

---

## 11. Preview: Day 394

Tomorrow we derive the **fundamental commutation relations** of angular momentum:

$$[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$$

These relations define the **Lie algebra** of the rotation group SO(3) and lead directly to:
- The impossibility of simultaneous eigenstates for all three components
- The emergence of ladder operators
- The quantization of angular momentum eigenvalues

---

*Day 393 of Year 1: Quantum Mechanics Core*
*Week 57: Orbital Angular Momentum*
*QSE Self-Study Curriculum*
