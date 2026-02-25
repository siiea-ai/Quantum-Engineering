# Day 509: Bloch Sphere for Mixed States

## Overview

**Day 509** | Week 73, Day 5 | Year 1, Month 19 | The Bloch Ball Representation

Today we extend the Bloch sphere representation to include mixed states, revealing that the interior of the Bloch ball represents all possible single-qubit states. This geometric picture provides powerful intuition for quantum operations.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Bloch ball theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Problem solving |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | 3D visualization lab |

---

## Learning Objectives

By the end of today, you will be able to:

1. Parameterize any qubit state using Bloch coordinates
2. Convert between density matrix and Bloch vector representations
3. Identify pure states (surface) vs mixed states (interior)
4. Visualize quantum operations as transformations of the Bloch ball
5. Understand the geometric meaning of decoherence
6. Describe measurement in the Bloch picture

---

## Core Content

### The Bloch Ball Representation

Every single-qubit density matrix can be written as:

$$\boxed{\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma}) = \frac{1}{2}(I + r_x X + r_y Y + r_z Z)}$$

where r⃗ = (rₓ, rᵧ, r_z) is the **Bloch vector** satisfying |r⃗| ≤ 1.

**Explicit form:**
$$\rho = \frac{1}{2}\begin{pmatrix} 1 + r_z & r_x - ir_y \\ r_x + ir_y & 1 - r_z \end{pmatrix}$$

### Recovering the Bloch Vector

Given a density matrix ρ, we can extract the Bloch vector:

$$\boxed{r_x = \text{Tr}(\rho X), \quad r_y = \text{Tr}(\rho Y), \quad r_z = \text{Tr}(\rho Z)}$$

These are the expectation values: r⃗ = (⟨X⟩, ⟨Y⟩, ⟨Z⟩).

### Pure States: The Bloch Sphere

For pure states ρ = |ψ⟩⟨ψ|:
- |r⃗| = 1 (exactly on the sphere surface)
- Any point on the unit sphere corresponds to a unique pure state

**Standard states:**
| State | Bloch Vector |
|-------|--------------|
| \|0⟩ | (0, 0, +1) |
| \|1⟩ | (0, 0, -1) |
| \|+⟩ | (+1, 0, 0) |
| \|−⟩ | (-1, 0, 0) |
| \|+i⟩ | (0, +1, 0) |
| \|−i⟩ | (0, -1, 0) |

### Mixed States: Inside the Ball

For mixed states:
- |r⃗| < 1 (strictly inside the ball)
- The center r⃗ = 0 is the maximally mixed state I/2

**Purity relationship:**
$$\gamma = \text{Tr}(\rho^2) = \frac{1 + |\vec{r}|^2}{2}$$

### Geometric Interpretation

| Region | |r⃗| | State Type | Purity |
|--------|-----|------------|--------|
| Surface | 1 | Pure | 1 |
| Interior | < 1 | Mixed | < 1 |
| Center | 0 | Max mixed | 1/2 |

### Eigenvalues from Bloch Vector

The eigenvalues of ρ are:

$$\boxed{\lambda_\pm = \frac{1 \pm |\vec{r}|}{2}}$$

This gives:
- Pure states (|r⃗| = 1): λ₊ = 1, λ₋ = 0
- Maximally mixed (|r⃗| = 0): λ₊ = λ₋ = 1/2

### Quantum Operations as Bloch Ball Transformations

**Unitary operations:** Rotate the Bloch vector (preserve |r⃗|)
$$U\rho U^\dagger \quad \Leftrightarrow \quad \vec{r} \rightarrow R\vec{r}$$

where R is a 3×3 rotation matrix (SO(3)).

**Examples:**
- X gate: 180° rotation about x-axis
- Y gate: 180° rotation about y-axis
- Z gate: 180° rotation about z-axis
- Hadamard: 180° rotation about (x+z)/√2 axis

**Decoherence:** Shrinks the Bloch vector toward the center
$$\vec{r} \rightarrow (1-p)\vec{r}$$

This is **depolarizing noise**.

### Dephasing in Bloch Picture

**T₂ dephasing** (z-basis):
$$\vec{r} = (r_x, r_y, r_z) \rightarrow (e^{-t/T_2}r_x, e^{-t/T_2}r_y, r_z)$$

The Bloch vector shrinks in the xy-plane while z remains unchanged.

### Measurement in Bloch Picture

Measuring in the z-basis projects onto the z-axis:

$$\vec{r} = (r_x, r_y, r_z) \xrightarrow{\text{measure}} \begin{cases} (0, 0, +1) & \text{outcome } |0\rangle \\ (0, 0, -1) & \text{outcome } |1\rangle \end{cases}$$

Non-selective measurement: r⃗ → (0, 0, r_z)
(Projects onto z-axis, destroys x and y components)

---

## Quantum Computing Connection

### Visualizing Gate Sequences

The Bloch sphere helps understand gate sequences:
1. Start at |0⟩: r⃗ = (0, 0, 1)
2. Apply H: r⃗ → (1, 0, 0) (now |+⟩)
3. Apply S: r⃗ → (0, 1, 0) (now |+i⟩)
4. Apply T: rotates by 45° about z

### Error Visualization

Errors shrink or rotate the Bloch vector:
- **Bit flip (X error):** Unwanted rotation about x
- **Phase flip (Z error):** Unwanted rotation about z
- **Depolarizing:** Uniform shrinkage

### State Tomography

To reconstruct ρ, measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩:
$$\rho = \frac{1}{2}(I + \langle X \rangle X + \langle Y \rangle Y + \langle Z \rangle Z)$$

Each expectation requires many shots.

---

## Worked Examples

### Example 1: Bloch Vector from Density Matrix

**Problem:** Find the Bloch vector for ρ = ⅔|0⟩⟨0| + ⅓|1⟩⟨1|.

**Solution:**

$$\rho = \begin{pmatrix} 2/3 & 0 \\ 0 & 1/3 \end{pmatrix}$$

$$r_x = \text{Tr}(\rho X) = \text{Tr}\begin{pmatrix} 0 & 2/3 \\ 1/3 & 0 \end{pmatrix} = 0$$

$$r_y = \text{Tr}(\rho Y) = \text{Tr}\begin{pmatrix} 0 & -2i/3 \\ i/3 & 0 \end{pmatrix} = 0$$

$$r_z = \text{Tr}(\rho Z) = \text{Tr}\begin{pmatrix} 2/3 & 0 \\ 0 & -1/3 \end{pmatrix} = \frac{2}{3} - \frac{1}{3} = \frac{1}{3}$$

Bloch vector: **r⃗ = (0, 0, 1/3)**

Length: |r⃗| = 1/3 < 1 (mixed state, inside the ball)

### Example 2: Density Matrix from Bloch Vector

**Problem:** Write the density matrix for r⃗ = (1/2, 0, √3/2).

**Solution:**

First verify: |r⃗|² = 1/4 + 0 + 3/4 = 1. This is a pure state!

$$\rho = \frac{1}{2}\begin{pmatrix} 1 + \sqrt{3}/2 & 1/2 \\ 1/2 & 1 - \sqrt{3}/2 \end{pmatrix}$$

$$\rho = \begin{pmatrix} (2+\sqrt{3})/4 & 1/4 \\ 1/4 & (2-\sqrt{3})/4 \end{pmatrix}$$

### Example 3: Effect of Dephasing

**Problem:** State |+⟩ undergoes 50% dephasing. Find the final Bloch vector and purity.

**Solution:**

Initial: r⃗ = (1, 0, 0)

After 50% dephasing (xy-components multiplied by 0.5):
$$\vec{r}' = (0.5, 0, 0)$$

Purity:
$$\gamma = \frac{1 + |0.5|^2}{2} = \frac{1 + 0.25}{2} = 0.625$$

The state is now mixed with purity 0.625.

---

## Practice Problems

### Direct Application

**Problem 1:** Find the Bloch vector for |−⟩ = (|0⟩ - |1⟩)/√2.

**Problem 2:** What density matrix corresponds to r⃗ = (0, 0.6, 0.8)?

**Problem 3:** A state has Bloch vector r⃗ = (0.3, 0.4, 0). Find its purity.

### Intermediate

**Problem 4:** Show that the Bloch vector for |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩ is (sin θ cos φ, sin θ sin φ, cos θ).

**Problem 5:** A qubit starts at r⃗ = (1, 0, 0). After a unitary, r⃗' = (0, 0, -1). What unitary was applied?

**Problem 6:** Derive the formula λ± = (1 ± |r⃗|)/2 for eigenvalues.

### Challenging

**Problem 7:** Show that any 2×2 positive semidefinite matrix with trace 1 can be written in Bloch form.

**Problem 8:** Prove that unitary operations preserve the length of the Bloch vector.

**Problem 9:** For amplitude damping with parameter γ, find how the Bloch vector transforms.

---

## Computational Lab

```python
"""
Day 509: Bloch Sphere for Mixed States
3D visualization of the Bloch ball
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def bloch_to_density(r):
    """Convert Bloch vector to density matrix"""
    return 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

def density_to_bloch(rho):
    """Convert density matrix to Bloch vector"""
    rx = np.trace(rho @ X).real
    ry = np.trace(rho @ Y).real
    rz = np.trace(rho @ Z).real
    return np.array([rx, ry, rz])

def purity_from_bloch(r):
    """Compute purity from Bloch vector"""
    return (1 + np.dot(r, r)) / 2

def draw_bloch_sphere(ax, alpha=0.1):
    """Draw the Bloch sphere wireframe"""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, alpha=alpha, color='blue')

    # Draw axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.3)

    # Label axes
    ax.text(1.3, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.3, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.3, 'Z', fontsize=12)

print("=" * 60)
print("BLOCH BALL VISUALIZATION")
print("=" * 60)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Pure states on surface
ax1 = fig.add_subplot(221, projection='3d')
draw_bloch_sphere(ax1)

# Standard pure states
pure_states = {
    '|0⟩': (0, 0, 1),
    '|1⟩': (0, 0, -1),
    '|+⟩': (1, 0, 0),
    '|−⟩': (-1, 0, 0),
    '|+i⟩': (0, 1, 0),
    '|−i⟩': (0, -1, 0),
}

for name, r in pure_states.items():
    ax1.scatter([r[0]], [r[1]], [r[2]], s=100, c='red', marker='o')
    ax1.text(r[0]*1.15, r[1]*1.15, r[2]*1.15, name, fontsize=10)

ax1.set_title('Pure States (Surface of Bloch Sphere)', fontsize=12)
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_zlim(-1.5, 1.5)

# 2. Mixed states inside
ax2 = fig.add_subplot(222, projection='3d')
draw_bloch_sphere(ax2)

# Random mixed states (inside the ball)
np.random.seed(42)
n_mixed = 50
for _ in range(n_mixed):
    # Random point inside unit ball
    r = np.random.randn(3)
    r = r / np.linalg.norm(r) * np.random.uniform(0, 0.9)
    gamma = purity_from_bloch(r)
    color = plt.cm.coolwarm(gamma)
    ax2.scatter([r[0]], [r[1]], [r[2]], c=[color], s=30, alpha=0.7)

# Maximally mixed state at center
ax2.scatter([0], [0], [0], s=200, c='green', marker='*', label='Maximally mixed')

ax2.set_title('Mixed States (Interior of Bloch Ball)', fontsize=12)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_zlim(-1.5, 1.5)

# 3. Dephasing trajectory
ax3 = fig.add_subplot(223, projection='3d')
draw_bloch_sphere(ax3)

# Start at |+⟩, dephase toward z-axis
r_start = np.array([1, 0, 0])
dephasing_params = np.linspace(0, 1, 20)

colors = plt.cm.viridis(np.linspace(0, 1, len(dephasing_params)))
for i, p in enumerate(dephasing_params):
    r = r_start * (1 - p)
    ax3.scatter([r[0]], [r[1]], [r[2]], c=[colors[i]], s=50)

ax3.plot([1, 0], [0, 0], [0, 0], 'r--', lw=2, label='Dephasing trajectory')
ax3.set_title('Dephasing: Shrinkage Toward Center', fontsize=12)
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_zlim(-1.5, 1.5)

# 4. Unitary rotation (gate)
ax4 = fig.add_subplot(224, projection='3d')
draw_bloch_sphere(ax4)

# Hadamard gate: rotation by π about (X+Z)/√2 axis
def hadamard_rotation(r):
    """Apply Hadamard rotation to Bloch vector"""
    # Hadamard swaps X and Z components, negates Y
    return np.array([r[2], -r[1], r[0]])

# Show several states before and after Hadamard
test_states = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0)]
colors_before = ['red', 'blue', 'green', 'orange']

for r, c in zip(test_states, colors_before):
    r = np.array(r)
    r_after = hadamard_rotation(r)

    ax4.scatter([r[0]], [r[1]], [r[2]], c=c, s=100, marker='o', alpha=0.5)
    ax4.scatter([r_after[0]], [r_after[1]], [r_after[2]], c=c, s=100, marker='^')
    ax4.plot([r[0], r_after[0]], [r[1], r_after[1]], [r[2], r_after[2]],
             c=c, ls='--', alpha=0.5)

ax4.set_title('Hadamard Gate: Rotation (○ before, △ after)', fontsize=12)
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax4.set_zlim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('bloch_ball_complete.png', dpi=150, bbox_inches='tight')
plt.show()

# Verification examples
print("\n" + "=" * 60)
print("BLOCH VECTOR CALCULATIONS")
print("=" * 60)

# Example 1: Mixed state
rho_mixed = np.array([[2/3, 0], [0, 1/3]], dtype=complex)
r_mixed = density_to_bloch(rho_mixed)
print(f"\nρ = diag(2/3, 1/3)")
print(f"Bloch vector: r = {r_mixed}")
print(f"|r| = {np.linalg.norm(r_mixed):.4f}")
print(f"Purity: γ = {purity_from_bloch(r_mixed):.4f}")

# Verify round-trip
rho_reconstructed = bloch_to_density(r_mixed)
print(f"Reconstructed ρ:\n{rho_reconstructed}")
print(f"Match: {np.allclose(rho_mixed, rho_reconstructed)}")

# Example 2: Pure state with phase
theta = np.pi/4
phi = np.pi/3
r_pure = np.array([np.sin(theta)*np.cos(phi),
                   np.sin(theta)*np.sin(phi),
                   np.cos(theta)])
print(f"\nPure state with θ=π/4, φ=π/3:")
print(f"Bloch vector: r = {r_pure}")
print(f"|r| = {np.linalg.norm(r_pure):.4f}")
print(f"Purity: γ = {purity_from_bloch(r_pure):.4f}")

print("\n" + "=" * 60)
print("Day 509 Complete: Bloch Sphere for Mixed States")
print("=" * 60)
```

---

## Summary

### Bloch Ball Representation

| Formula | Description |
|---------|-------------|
| ρ = ½(I + r⃗·σ⃗) | General qubit state |
| r⃗ = (⟨X⟩, ⟨Y⟩, ⟨Z⟩) | Extracting Bloch vector |
| γ = (1 + \|r⃗\|²)/2 | Purity from Bloch vector |
| λ± = (1 ± \|r⃗\|)/2 | Eigenvalues |

### Geometric Classification

| |r⃗| | Location | State Type |
|-----|----------|------------|
| 1 | Surface | Pure |
| < 1 | Interior | Mixed |
| 0 | Center | Max mixed |

### Quantum Operations

| Operation | Effect on r⃗ |
|-----------|--------------|
| Unitary | Rotation (preserves \|r⃗\|) |
| Dephasing | Shrinks toward axis |
| Depolarizing | Uniform shrinkage |
| Measurement | Projects to axis |

---

## Daily Checklist

- [ ] I can convert between density matrix and Bloch vector
- [ ] I understand the geometric meaning of purity
- [ ] I can visualize unitary gates as rotations
- [ ] I understand decoherence as shrinkage
- [ ] I can describe measurement in the Bloch picture

---

## Preview: Day 510

Tomorrow we'll learn about **distinguishing quantum states** using trace distance and fidelity—key measures for quantifying how "close" two quantum states are.

---

*Next: Day 510 — Distinguishing States*
