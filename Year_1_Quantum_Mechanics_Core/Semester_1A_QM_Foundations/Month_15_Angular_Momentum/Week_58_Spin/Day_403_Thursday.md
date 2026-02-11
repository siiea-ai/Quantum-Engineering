# Day 403: Spin States and the Bloch Sphere

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Bloch sphere representation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 403, you will be able to:

1. Parameterize arbitrary spin-1/2 states using θ and φ
2. Map spin states to points on the Bloch sphere
3. Calculate the Bloch vector from a state vector
4. Relate density matrices to Bloch vectors
5. Visualize qubit operations geometrically

---

## Core Content

### 1. General Spin-1/2 State Parameterization

Any normalized spin-1/2 state can be written (up to global phase):

$$\boxed{|\psi\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + e^{i\phi}\sin\frac{\theta}{2}|\downarrow\rangle}$$

where 0 ≤ θ ≤ π and 0 ≤ φ < 2π.

**Why θ/2?** So that θ represents the polar angle on the Bloch sphere.

### 2. The Bloch Sphere

The state |ψ⟩ corresponds to the point on the unit sphere with spherical coordinates (θ, φ):

$$\hat{n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$$

**Key points:**
- |↑⟩ (north pole): θ = 0
- |↓⟩ (south pole): θ = π
- |+⟩ (positive x): θ = π/2, φ = 0
- |-⟩ (negative x): θ = π/2, φ = π
- |+i⟩ (positive y): θ = π/2, φ = π/2
- |-i⟩ (negative y): θ = π/2, φ = 3π/2

### 3. The Bloch Vector

For state |ψ⟩, the **Bloch vector** is:

$$\boxed{\mathbf{r} = \langle\psi|\boldsymbol{\sigma}|\psi\rangle = (\langle\sigma_x\rangle, \langle\sigma_y\rangle, \langle\sigma_z\rangle)}$$

For the standard parameterization:
$$\mathbf{r} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) = \hat{n}$$

**The Bloch vector IS the unit vector pointing to the state on the sphere!**

### 4. Eigenstates of n̂·σ

The state |ψ(θ,φ)⟩ is the +1 eigenstate of:

$$\hat{n}\cdot\boldsymbol{\sigma} = n_x\sigma_x + n_y\sigma_y + n_z\sigma_z = \begin{pmatrix} \cos\theta & \sin\theta e^{-i\phi} \\ \sin\theta e^{i\phi} & -\cos\theta \end{pmatrix}$$

This represents spin measurement along direction n̂.

### 5. Density Matrix and Bloch Vector

For a pure state:
$$\boxed{\rho = |\psi\rangle\langle\psi| = \frac{1}{2}(I + \mathbf{r}\cdot\boldsymbol{\sigma})}$$

For mixed states, |**r**| < 1 (inside the sphere):
$$\rho = \frac{1}{2}(I + \mathbf{r}\cdot\boldsymbol{\sigma}), \quad |\mathbf{r}| \leq 1$$

The maximally mixed state (ρ = I/2) is at the origin.

### 6. Orthogonal States

Two states are orthogonal iff their Bloch vectors are **antipodal** (opposite sides of sphere).

$$\langle\psi_1|\psi_2\rangle = 0 \iff \mathbf{r}_1 = -\mathbf{r}_2$$

---

## Quantum Computing Connection

| Bloch Sphere | Quantum Computing |
|--------------|-------------------|
| North pole \|↑⟩ | \|0⟩ |
| South pole \|↓⟩ | \|1⟩ |
| Equator | Superposition states |
| Rotation about axis | Single-qubit gates |
| n̂ measurement | Basis measurement |

**Every single-qubit gate is a rotation on the Bloch sphere!**

---

## Worked Examples

### Example 1: State to Bloch Vector

**Problem:** Find the Bloch vector for |+⟩ = (|↑⟩ + |↓⟩)/√2.

**Solution:**
$$\langle\sigma_x\rangle = \langle +|\sigma_x|+\rangle = 1$$
$$\langle\sigma_y\rangle = \langle +|\sigma_y|+\rangle = 0$$
$$\langle\sigma_z\rangle = \langle +|\sigma_z|+\rangle = 0$$

$$\mathbf{r} = (1, 0, 0)$$

This is on the positive x-axis, as expected!

### Example 2: Bloch Vector to State

**Problem:** Write the state corresponding to n̂ = (0, 1, 0) (positive y-axis).

**Solution:**
θ = π/2, φ = π/2

$$|\psi\rangle = \cos\frac{\pi/4}{}|\uparrow\rangle + e^{i\pi/2}\sin\frac{\pi/4}{}|\downarrow\rangle = \frac{1}{\sqrt{2}}(|\uparrow\rangle + i|\downarrow\rangle) = |+i\rangle$$

### Example 3: Density Matrix

**Problem:** Find the density matrix for |+⟩.

**Solution:**
$$\rho = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 \\ 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

Verify: ρ = (I + σₓ)/2 = (1/2)[[1,0],[0,1]] + (1/2)[[0,1],[1,0]] = (1/2)[[1,1],[1,1]] ✓

---

## Practice Problems

### Direct Application

1. Find the Bloch vector for |ψ⟩ = (|↑⟩ - i|↓⟩)/√2.

2. What state corresponds to n̂ = (1, 1, 1)/√3?

3. Calculate the density matrix for |↑⟩.

### Intermediate

4. Show that antipodal points on the Bloch sphere correspond to orthogonal states.

5. Find all states with ⟨σᵤ⟩ = 1/2.

6. Express ρ = (1/2)[[1, (1-i)/√2], [(1+i)/√2, 1]] as (I + **r**·**σ**)/2.

### Challenging

7. Prove that any rotation R(n̂, θ) on the Bloch sphere corresponds to the unitary e^{-iθn̂·σ/2}.

8. Show that the purity Tr(ρ²) = (1 + |**r**|²)/2.

---

## Computational Lab

```python
"""
Day 403 Computational Lab: Bloch Sphere
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def state_to_bloch(state):
    """Convert state vector to Bloch vector."""
    state = np.array(state, dtype=complex).flatten()
    state = state / np.linalg.norm(state)  # Normalize

    rx = np.real(np.vdot(state, sigma_x @ state))
    ry = np.real(np.vdot(state, sigma_y @ state))
    rz = np.real(np.vdot(state, sigma_z @ state))

    return np.array([rx, ry, rz])

def bloch_to_state(theta, phi):
    """Convert Bloch sphere angles to state vector."""
    return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])

def plot_bloch_sphere(states=None, labels=None):
    """Plot Bloch sphere with optional state vectors."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere wireframe
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.1, color='lightblue')
    ax.plot_wireframe(x, y, z, alpha=0.2, color='gray', linewidth=0.5)

    # Draw axes
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='blue', arrow_length_ratio=0.1, linewidth=2)

    ax.text(1.5, 0, 0, 'X', fontsize=12, color='red')
    ax.text(0, 1.5, 0, 'Y', fontsize=12, color='green')
    ax.text(0, 0, 1.5, 'Z (|0⟩)', fontsize=12, color='blue')
    ax.text(0, 0, -1.3, '|1⟩', fontsize=12, color='blue')

    # Plot special states
    special_states = {
        '|0⟩': [1, 0],
        '|1⟩': [0, 1],
        '|+⟩': [1, 1],
        '|-⟩': [1, -1],
        '|+i⟩': [1, 1j],
        '|-i⟩': [1, -1j],
    }

    for name, state in special_states.items():
        r = state_to_bloch(state)
        ax.scatter(*r, s=100, c='black', marker='o')
        ax.text(r[0]*1.1, r[1]*1.1, r[2]*1.1, name, fontsize=10)

    # Plot additional states if provided
    if states is not None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
        for i, (state, label) in enumerate(zip(states, labels or ['']*len(states))):
            r = state_to_bloch(state)
            ax.quiver(0, 0, 0, r[0], r[1], r[2], color=colors[i],
                      arrow_length_ratio=0.1, linewidth=3)
            ax.text(r[0]*1.15, r[1]*1.15, r[2]*1.15, label, fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere')

    # Equal aspect ratio
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    return fig, ax

def demonstrate_bloch_vectors():
    """Demonstrate Bloch vector calculations."""
    print("Bloch Vector Examples")
    print("=" * 50)

    states = {
        '|↑⟩': [1, 0],
        '|↓⟩': [0, 1],
        '|+⟩': [1, 1],
        '|-⟩': [1, -1],
        '|+i⟩': [1, 1j],
        '(|↑⟩ + 2|↓⟩)/√5': [1, 2],
    }

    for name, state in states.items():
        r = state_to_bloch(state)
        print(f"\n{name}:")
        print(f"  Bloch vector: ({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})")
        print(f"  |r| = {np.linalg.norm(r):.4f}")

def visualize_state_trajectory():
    """Visualize a state moving on the Bloch sphere."""
    fig, ax = plot_bloch_sphere()

    # Trajectory: rotation about z-axis (varying φ)
    theta = np.pi/3
    phi_values = np.linspace(0, 2*np.pi, 100)

    x_traj = np.sin(theta) * np.cos(phi_values)
    y_traj = np.sin(theta) * np.sin(phi_values)
    z_traj = np.cos(theta) * np.ones_like(phi_values)

    ax.plot(x_traj, y_traj, z_traj, 'r-', linewidth=2, label='Precession path')
    ax.legend()

    plt.savefig('bloch_sphere.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Day 403: Bloch Sphere")
    print("=" * 50)

    demonstrate_bloch_vectors()

    print("\nVisualizing Bloch sphere...")
    visualize_state_trajectory()

    print("\nLab complete!")
```

---

## Summary

| Concept | Formula |
|---------|---------|
| State parameterization | \|ψ⟩ = cos(θ/2)\|↑⟩ + e^{iφ}sin(θ/2)\|↓⟩ |
| Bloch vector | **r** = (⟨σₓ⟩, ⟨σᵧ⟩, ⟨σᵤ⟩) |
| Unit vector | n̂ = (sinθ cosφ, sinθ sinφ, cosθ) |
| Density matrix | ρ = (I + **r**·**σ**)/2 |
| Pure state | \|**r**\| = 1 |
| Orthogonality | **r**₁ = -**r**₂ |

---

## Daily Checklist

- [ ] I can convert states to Bloch vectors
- [ ] I understand the geometry of the Bloch sphere
- [ ] I can relate density matrices to Bloch vectors
- [ ] I see how gates are rotations
- [ ] I completed the computational lab

---

## Preview: Day 404

Tomorrow we study spin dynamics—how spins precess in magnetic fields, leading to Larmor precession and Rabi oscillations.

---

**Next:** [Day_404_Friday.md](Day_404_Friday.md) — Spin Dynamics
