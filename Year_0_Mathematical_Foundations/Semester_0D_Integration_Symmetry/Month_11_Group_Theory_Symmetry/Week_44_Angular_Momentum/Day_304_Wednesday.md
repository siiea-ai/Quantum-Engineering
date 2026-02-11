# Day 304: Spin Angular Momentum

## Overview

**Month 11, Week 44, Day 3 — Wednesday**

Today we study spin — intrinsic angular momentum with no classical analog. Unlike orbital angular momentum, spin can take half-integer values, fundamentally distinguishing fermions from bosons. The spin-1/2 system (electron, proton, qubit) is the simplest non-trivial quantum system and the foundation of quantum information.

## Learning Objectives

1. Understand spin as intrinsic angular momentum
2. Master the spin-1/2 formalism and Pauli matrices
3. Analyze the Stern-Gerlach experiment
4. Describe spin states on the Bloch sphere
5. Connect to quantum computing qubits

---

## 1. The Discovery of Spin

### The Stern-Gerlach Experiment (1922)

**Setup:** Silver atoms passed through inhomogeneous magnetic field.

**Classical expectation:** Continuous distribution of deflections.

**Observation:** Only two discrete spots — quantized angular momentum!

**Puzzle:** Silver has $\ell = 0$ in ground state. Where does angular momentum come from?

### Goudsmit and Uhlenbeck (1925)

**Resolution:** Electrons have **intrinsic angular momentum** called **spin**.

- Spin quantum number: $s = 1/2$
- Magnetic quantum number: $m_s = \pm 1/2$
- Only two states: "spin up" $|\uparrow\rangle$ and "spin down" $|\downarrow\rangle$

---

## 2. Spin Algebra

### The Operators

Spin operators $\hat{S}_x, \hat{S}_y, \hat{S}_z$ satisfy the same algebra as orbital angular momentum:

$$\boxed{[\hat{S}_i, \hat{S}_j] = i\hbar \epsilon_{ijk} \hat{S}_k}$$

### Eigenvalue Equations

$$\hat{\mathbf{S}}^2 |s, m_s\rangle = \hbar^2 s(s+1) |s, m_s\rangle$$
$$\hat{S}_z |s, m_s\rangle = \hbar m_s |s, m_s\rangle$$

### Key Difference from Orbital

| Property | Orbital $\ell$ | Spin $s$ |
|----------|---------------|----------|
| Values | $0, 1, 2, \ldots$ | $0, \frac{1}{2}, 1, \frac{3}{2}, \ldots$ |
| Origin | $\mathbf{L} = \mathbf{r} \times \mathbf{p}$ | Intrinsic |
| Wavefunction | $Y_\ell^m(\theta, \phi)$ | Spinor |
| Single-valuedness | Required | Not required |

---

## 3. Spin-1/2: The Two-State System

### State Space

The spin-1/2 Hilbert space is $\mathbb{C}^2$. Basis states:

$$|\uparrow\rangle \equiv |{\textstyle\frac{1}{2}}, +{\textstyle\frac{1}{2}}\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

$$|\downarrow\rangle \equiv |{\textstyle\frac{1}{2}}, -{\textstyle\frac{1}{2}}\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

General state:
$$|\chi\rangle = \alpha|\uparrow\rangle + \beta|\downarrow\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}, \quad |\alpha|^2 + |\beta|^2 = 1$$

### The Pauli Matrices

$$\boxed{\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}}$$

Spin operators:
$$\hat{S}_i = \frac{\hbar}{2}\sigma_i$$

### Pauli Matrix Properties

**Algebraic:**
$$\sigma_i^2 = I, \quad \sigma_i \sigma_j = i\epsilon_{ijk}\sigma_k \quad (i \neq j)$$
$$\{\sigma_i, \sigma_j\} = 2\delta_{ij}I$$

**Trace:**
$$\text{Tr}(\sigma_i) = 0, \quad \text{Tr}(\sigma_i \sigma_j) = 2\delta_{ij}$$

**Completeness:**
$$\sigma_i \sigma_j = \delta_{ij}I + i\epsilon_{ijk}\sigma_k$$

---

## 4. Measuring Spin Components

### Eigenstates of $\hat{S}_z$

$$|\uparrow\rangle: \quad S_z = +\frac{\hbar}{2}$$
$$|\downarrow\rangle: \quad S_z = -\frac{\hbar}{2}$$

### Eigenstates of $\hat{S}_x$

$$|+x\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}(|\uparrow\rangle + |\downarrow\rangle)$$

$$|-x\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}(|\uparrow\rangle - |\downarrow\rangle)$$

### Eigenstates of $\hat{S}_y$

$$|+y\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix} = \frac{1}{\sqrt{2}}(|\uparrow\rangle + i|\downarrow\rangle)$$

$$|-y\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix} = \frac{1}{\sqrt{2}}(|\uparrow\rangle - i|\downarrow\rangle)$$

### Sequential Measurements

**Example:** Prepare $|\uparrow\rangle$, measure $S_x$, then measure $S_z$.

1. $|\uparrow\rangle = \frac{1}{\sqrt{2}}(|+x\rangle + |-x\rangle)$
2. After $S_x$ measurement with result $+\hbar/2$: state collapses to $|+x\rangle$
3. $|+x\rangle = \frac{1}{\sqrt{2}}(|\uparrow\rangle + |\downarrow\rangle)$
4. $S_z$ measurement: 50% chance each of $\pm\hbar/2$

**Measuring one component "erases" information about perpendicular components!**

---

## 5. The Bloch Sphere

### General Pure State

Any normalized spin-1/2 state can be written:
$$|\psi\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + e^{i\phi}\sin\frac{\theta}{2}|\downarrow\rangle$$

where $0 \leq \theta \leq \pi$ and $0 \leq \phi < 2\pi$.

### The Bloch Vector

$$\boxed{\vec{n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)}$$

This maps states to the unit sphere!

### Expectation Values

$$\langle \vec{S} \rangle = \frac{\hbar}{2}\vec{n}$$

The Bloch vector points in the direction of maximum spin component.

### Key States on Bloch Sphere

| State | $(\theta, \phi)$ | Position |
|-------|------------------|----------|
| $\|\uparrow\rangle$ | $(0, -)$ | North pole |
| $\|\downarrow\rangle$ | $(\pi, -)$ | South pole |
| $\|+x\rangle$ | $(\pi/2, 0)$ | Positive x-axis |
| $\|-x\rangle$ | $(\pi/2, \pi)$ | Negative x-axis |
| $\|+y\rangle$ | $(\pi/2, \pi/2)$ | Positive y-axis |
| $\|-y\rangle$ | $(\pi/2, 3\pi/2)$ | Negative y-axis |

### Orthogonal States

Orthogonal states correspond to **antipodal points** on the Bloch sphere.

---

## 6. Spin Rotations

### Rotation Operator

$$\hat{U}(\hat{n}, \theta) = e^{-i\theta \hat{n}\cdot\vec{S}/\hbar} = e^{-i\theta \hat{n}\cdot\vec{\sigma}/2}$$

Using the identity:
$$e^{-i\theta \hat{n}\cdot\vec{\sigma}/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}(\hat{n}\cdot\vec{\sigma})$$

### Examples

**Rotation about z by angle $\phi$:**
$$R_z(\phi) = \begin{pmatrix} e^{-i\phi/2} & 0 \\ 0 & e^{i\phi/2} \end{pmatrix}$$

**Rotation about x by angle $\theta$:**
$$R_x(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

**Rotation about y by angle $\theta$:**
$$R_y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

### The $2\pi$ Rotation Phase

$$R_z(2\pi)|\psi\rangle = e^{-i\pi}|\psi\rangle = -|\psi\rangle$$

A $2\pi$ rotation gives a **minus sign** for spin-1/2!

$$R_z(4\pi)|\psi\rangle = |\psi\rangle$$

A full rotation requires $4\pi$ to return to the original state.

---

## 7. Magnetic Moment and Stern-Gerlach

### Magnetic Moment

$$\boxed{\vec{\mu} = -g_s \frac{e}{2m_e}\vec{S} = -g_s \mu_B \frac{\vec{S}}{\hbar}}$$

where:
- $g_s \approx 2.002$ (electron g-factor)
- $\mu_B = e\hbar/(2m_e)$ (Bohr magneton)

### Energy in Magnetic Field

$$\hat{H} = -\vec{\mu}\cdot\vec{B} = \frac{g_s \mu_B}{\hbar}\vec{S}\cdot\vec{B}$$

For $\vec{B} = B_0\hat{z}$:
$$\hat{H} = \omega_0 \hat{S}_z$$

where $\omega_0 = g_s \mu_B B_0/\hbar$ (Larmor frequency).

### Energy Levels

$$E_\uparrow = +\frac{\hbar\omega_0}{2}, \quad E_\downarrow = -\frac{\hbar\omega_0}{2}$$

**Zeeman splitting:** $\Delta E = \hbar\omega_0 = g_s \mu_B B_0$

### Stern-Gerlach Force

In inhomogeneous field with gradient $\partial B_z/\partial z$:

$$F_z = -\frac{\partial E}{\partial z} = \mp \mu_B g_s \frac{\partial B_z}{\partial z}$$

This separates spin-up and spin-down beams.

---

## 8. Larmor Precession

### Time Evolution

For Hamiltonian $\hat{H} = \omega_0 \hat{S}_z$:
$$|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle = e^{-i\omega_0 t \sigma_z/2}|\psi(0)\rangle$$

### Precession of Expectation Values

Starting from $|\psi(0)\rangle = |+x\rangle$:

$$\langle S_x(t) \rangle = \frac{\hbar}{2}\cos(\omega_0 t)$$
$$\langle S_y(t) \rangle = \frac{\hbar}{2}\sin(\omega_0 t)$$
$$\langle S_z(t) \rangle = 0$$

**The spin precesses around the field direction at the Larmor frequency!**

---

## 9. Computational Lab

```python
"""
Day 304: Spin-1/2 Angular Momentum
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma = [sigma_x, sigma_y, sigma_z]
I2 = np.eye(2, dtype=complex)

# Basis states
spin_up = np.array([1, 0], dtype=complex)
spin_down = np.array([0, 1], dtype=complex)


class SpinHalf:
    """Spin-1/2 quantum system."""

    def __init__(self, state=None, theta=0, phi=0):
        """
        Initialize spin state.
        If state is None, use Bloch sphere angles (theta, phi).
        """
        if state is None:
            self.state = (np.cos(theta/2) * spin_up +
                         np.exp(1j*phi) * np.sin(theta/2) * spin_down)
        else:
            self.state = np.array(state, dtype=complex)
            self.state = self.state / np.linalg.norm(self.state)

    def expectation(self, operator):
        """Compute expectation value <ψ|O|ψ>."""
        return np.real(self.state.conj() @ operator @ self.state)

    def bloch_vector(self):
        """Return Bloch vector components."""
        return np.array([
            self.expectation(sigma_x),
            self.expectation(sigma_y),
            self.expectation(sigma_z)
        ])

    def measure_z(self):
        """Simulate measurement of S_z. Returns ±1/2 (in units of hbar)."""
        prob_up = np.abs(self.state[0])**2
        if np.random.random() < prob_up:
            self.state = spin_up.copy()
            return 0.5
        else:
            self.state = spin_down.copy()
            return -0.5

    def rotate(self, axis, angle):
        """Apply rotation about axis by angle."""
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        n_dot_sigma = sum(axis[i] * sigma[i] for i in range(3))
        U = np.cos(angle/2) * I2 - 1j * np.sin(angle/2) * n_dot_sigma
        self.state = U @ self.state

    def evolve(self, H, t):
        """Evolve under Hamiltonian H for time t."""
        U = expm(-1j * H * t)
        self.state = U @ self.state


def rotation_matrices():
    """Generate common rotation matrices."""

    def Rx(theta):
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)

    def Ry(theta):
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)

    def Rz(phi):
        return np.array([
            [np.exp(-1j*phi/2), 0],
            [0, np.exp(1j*phi/2)]
        ], dtype=complex)

    return Rx, Ry, Rz


def demonstrate_pauli_algebra():
    """Verify Pauli matrix relations."""
    print("=" * 50)
    print("PAULI MATRIX ALGEBRA")
    print("=" * 50)

    # Square to identity
    print("\nσ_i² = I:")
    for i, name in enumerate(['x', 'y', 'z']):
        result = sigma[i] @ sigma[i]
        is_identity = np.allclose(result, I2)
        print(f"  σ_{name}² = I: {is_identity}")

    # Anticommutation
    print("\n{σ_i, σ_j} = 2δ_ij I:")
    for i in range(3):
        for j in range(i, 3):
            anticomm = sigma[i] @ sigma[j] + sigma[j] @ sigma[i]
            expected = 2 * I2 if i == j else np.zeros((2, 2))
            print(f"  {{σ_{['x','y','z'][i]}, σ_{['x','y','z'][j]}}} correct: "
                  f"{np.allclose(anticomm, expected)}")

    # Commutation
    print("\n[σ_i, σ_j] = 2i ε_ijk σ_k:")
    print(f"  [σ_x, σ_y] = 2i σ_z: {np.allclose(sigma_x @ sigma_y - sigma_y @ sigma_x, 2j * sigma_z)}")
    print(f"  [σ_y, σ_z] = 2i σ_x: {np.allclose(sigma_y @ sigma_z - sigma_z @ sigma_y, 2j * sigma_x)}")
    print(f"  [σ_z, σ_x] = 2i σ_y: {np.allclose(sigma_z @ sigma_x - sigma_x @ sigma_z, 2j * sigma_y)}")


def demonstrate_spin_states():
    """Show eigenstates of different spin components."""
    print("\n" + "=" * 50)
    print("SPIN EIGENSTATES")
    print("=" * 50)

    # Eigenstates
    states = {
        '|↑⟩ (Sz = +1/2)': spin_up,
        '|↓⟩ (Sz = -1/2)': spin_down,
        '|+x⟩ (Sx = +1/2)': (spin_up + spin_down) / np.sqrt(2),
        '|-x⟩ (Sx = -1/2)': (spin_up - spin_down) / np.sqrt(2),
        '|+y⟩ (Sy = +1/2)': (spin_up + 1j*spin_down) / np.sqrt(2),
        '|-y⟩ (Sy = -1/2)': (spin_up - 1j*spin_down) / np.sqrt(2),
    }

    for name, state in states.items():
        spin = SpinHalf(state)
        vec = spin.bloch_vector()
        print(f"\n{name}:")
        print(f"  State: [{state[0]:.3f}, {state[1]:.3f}]")
        print(f"  Bloch vector: ({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})")


def stern_gerlach_simulation(n_particles=10000):
    """Simulate Stern-Gerlach experiment."""
    print("\n" + "=" * 50)
    print("STERN-GERLACH SIMULATION")
    print("=" * 50)

    # Particles prepared in |+x⟩ state
    results_z = []
    for _ in range(n_particles):
        spin = SpinHalf((spin_up + spin_down) / np.sqrt(2))
        results_z.append(spin.measure_z())

    up_count = sum(1 for r in results_z if r > 0)
    down_count = n_particles - up_count

    print(f"\nParticles in |+x⟩ measured along z:")
    print(f"  Spin up:   {up_count}/{n_particles} ({100*up_count/n_particles:.1f}%)")
    print(f"  Spin down: {down_count}/{n_particles} ({100*down_count/n_particles:.1f}%)")
    print(f"  Expected: 50% each")


def visualize_bloch_sphere():
    """Create Bloch sphere visualization."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='gray')

    # Draw axes
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='k', arrow_length_ratio=0.1)
    ax.text(1.4, 0, 0, 'x')
    ax.text(0, 1.4, 0, 'y')
    ax.text(0, 0, 1.4, 'z')

    # Plot key states
    states = {
        '|↑⟩': spin_up,
        '|↓⟩': spin_down,
        '|+x⟩': (spin_up + spin_down) / np.sqrt(2),
        '|+y⟩': (spin_up + 1j*spin_down) / np.sqrt(2),
    }

    colors = ['blue', 'red', 'green', 'purple']
    for (name, state), color in zip(states.items(), colors):
        spin = SpinHalf(state)
        vec = spin.bloch_vector()
        ax.scatter(*vec, s=100, color=color, label=name)
        ax.quiver(0, 0, 0, *vec, color=color, arrow_length_ratio=0.1)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere')
    ax.legend()

    plt.savefig('bloch_sphere.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bloch_sphere.png")


def larmor_precession():
    """Visualize Larmor precession."""
    print("\n" + "=" * 50)
    print("LARMOR PRECESSION")
    print("=" * 50)

    omega_0 = 1.0  # Larmor frequency
    H = omega_0 * sigma_z / 2  # Hamiltonian (in units of hbar)

    # Start in |+x⟩
    spin = SpinHalf((spin_up + spin_down) / np.sqrt(2))

    times = np.linspace(0, 4*np.pi/omega_0, 100)
    sx, sy, sz = [], [], []

    for t in times:
        test_spin = SpinHalf((spin_up + spin_down) / np.sqrt(2))
        test_spin.evolve(H, t)
        vec = test_spin.bloch_vector()
        sx.append(vec[0])
        sy.append(vec[1])
        sz.append(vec[2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time evolution plot
    ax1.plot(times * omega_0 / (2*np.pi), sx, label=r'$\langle S_x \rangle$')
    ax1.plot(times * omega_0 / (2*np.pi), sy, label=r'$\langle S_y \rangle$')
    ax1.plot(times * omega_0 / (2*np.pi), sz, label=r'$\langle S_z \rangle$')
    ax1.set_xlabel(r'Time ($\omega_0 t / 2\pi$)')
    ax1.set_ylabel('Expectation value (units of ℏ/2)')
    ax1.set_title('Larmor Precession')
    ax1.legend()
    ax1.grid(True)

    # 3D trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(sx, sy, sz, 'b-', linewidth=2)
    ax2.scatter(sx[0], sy[0], sz[0], s=100, c='green', label='Start')
    ax2.scatter(sx[-1], sy[-1], sz[-1], s=100, c='red', label='End')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x, y, z, alpha=0.1, color='gray')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Bloch Sphere Trajectory')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('larmor_precession.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: larmor_precession.png")


def two_pi_rotation():
    """Demonstrate 2π rotation phase."""
    print("\n" + "=" * 50)
    print("2π ROTATION PHASE")
    print("=" * 50)

    state = spin_up.copy()
    spin = SpinHalf(state)

    print(f"\nInitial state: {state}")

    # Rotate by 2π about z
    spin.rotate([0, 0, 1], 2*np.pi)
    print(f"After 2π rotation about z: {spin.state.round(4)}")
    print(f"Phase factor: {(spin.state[0] / state[0]).round(4)}")

    # Rotate by another 2π (total 4π)
    spin.rotate([0, 0, 1], 2*np.pi)
    print(f"After 4π rotation about z: {spin.state.round(4)}")
    print(f"Phase factor: {(spin.state[0] / state[0]).round(4)}")


# Main execution
if __name__ == "__main__":
    demonstrate_pauli_algebra()
    demonstrate_spin_states()
    stern_gerlach_simulation()
    two_pi_rotation()
    visualize_bloch_sphere()
    larmor_precession()
```

---

## 10. Practice Problems

### Problem 1: Pauli Matrix Identity

Prove that $(\vec{a}\cdot\vec{\sigma})(\vec{b}\cdot\vec{\sigma}) = (\vec{a}\cdot\vec{b})I + i(\vec{a}\times\vec{b})\cdot\vec{\sigma}$.

### Problem 2: General Spin State

A spin-1/2 particle is in the state:
$$|\chi\rangle = \frac{1}{\sqrt{3}}|\uparrow\rangle + \sqrt{\frac{2}{3}}|\downarrow\rangle$$

Find:
- (a) The Bloch sphere angles $(\theta, \phi)$
- (b) $\langle S_x \rangle$, $\langle S_y \rangle$, $\langle S_z \rangle$
- (c) Probability of measuring $S_z = +\hbar/2$

### Problem 3: Rotation Operator

Show that $R_y(\pi)|\uparrow\rangle = i|\downarrow\rangle$ (up to global phase).

### Problem 4: Sequential Measurements

A particle starts in $|\uparrow\rangle$. Calculate the probability of measuring $S_z = +\hbar/2$ if we:
1. First measure $S_x$ and get $+\hbar/2$
2. Then measure $S_z$

### Problem 5: Larmor Period

For an electron in a 1 Tesla magnetic field, calculate the Larmor frequency and period.

---

## Summary

### Spin-1/2 Formalism

$$\boxed{\hat{S}_i = \frac{\hbar}{2}\sigma_i, \quad [\hat{S}_i, \hat{S}_j] = i\hbar\epsilon_{ijk}\hat{S}_k}$$

### The Bloch Sphere

$$|\psi\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + e^{i\phi}\sin\frac{\theta}{2}|\downarrow\rangle \leftrightarrow \vec{n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$$

### Key Properties

| Property | Value |
|----------|-------|
| Spin magnitude | $\|\mathbf{S}\| = \frac{\sqrt{3}}{2}\hbar$ |
| $S_z$ eigenvalues | $\pm\frac{\hbar}{2}$ |
| $2\pi$ rotation | Phase factor $-1$ |
| $4\pi$ rotation | Returns to original |

---

## Preview: Day 305

Tomorrow we study the **addition of angular momenta** — how to combine the angular momenta of two particles. This is essential for understanding multi-electron atoms, nuclear physics, and entangled quantum states.
