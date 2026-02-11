# Day 404: Spin Dynamics — Precession in Magnetic Fields

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Magnetic fields and spin |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 404, you will be able to:

1. Calculate the magnetic moment of a spin-1/2 particle
2. Write the Hamiltonian for spin in a magnetic field
3. Derive and visualize Larmor precession
4. Solve the time evolution of spin states
5. Connect spin dynamics to qubit control

---

## Core Content

### 1. Magnetic Moment of Spin

A particle with spin has an intrinsic magnetic moment:

$$\boxed{\boldsymbol{\mu} = \gamma\hat{\mathbf{S}} = \frac{g_s e}{2m}\hat{\mathbf{S}}}$$

where:
- γ = gyromagnetic ratio
- gₛ ≈ 2 for electron (from Dirac equation)
- e = electron charge, m = electron mass

For spin-1/2: **μ** = γ(ℏ/2)**σ** = μ_B g_s **σ**/2

where μ_B = eℏ/2m is the Bohr magneton.

### 2. Spin Hamiltonian

In magnetic field **B**, the Hamiltonian is:

$$\boxed{\hat{H} = -\boldsymbol{\mu}\cdot\mathbf{B} = -\gamma\hat{\mathbf{S}}\cdot\mathbf{B}}$$

For **B** = B₀ ẑ (field along z):

$$\hat{H} = -\gamma B_0 \hat{S}_z = -\frac{\gamma B_0\hbar}{2}\sigma_z = -\frac{\hbar\omega_L}{2}\sigma_z$$

where ω_L = γB₀ is the **Larmor frequency**.

### 3. Energy Levels

The eigenstates of Ĥ are |↑⟩ and |↓⟩ with energies:

$$E_\uparrow = -\frac{\hbar\omega_L}{2}, \quad E_\downarrow = +\frac{\hbar\omega_L}{2}$$

Energy splitting: ΔE = ℏω_L

### 4. Time Evolution

The time evolution operator:

$$\hat{U}(t) = e^{-i\hat{H}t/\hbar} = e^{i\omega_L t\sigma_z/2}$$

For initial state |ψ(0)⟩ = α|↑⟩ + β|↓⟩:

$$|\psi(t)\rangle = \alpha e^{i\omega_L t/2}|\uparrow\rangle + \beta e^{-i\omega_L t/2}|\downarrow\rangle$$

### 5. Larmor Precession

The Bloch vector precesses about the z-axis:

$$\mathbf{r}(t) = (r_x\cos\omega_L t - r_y\sin\omega_L t,\, r_x\sin\omega_L t + r_y\cos\omega_L t,\, r_z)$$

**This is rotation about z at frequency ω_L!**

In operator form:
$$\frac{d\langle\hat{\mathbf{S}}\rangle}{dt} = \gamma\langle\hat{\mathbf{S}}\rangle \times \mathbf{B}$$

This is the same as classical precession!

### 6. Rabi Oscillations (Preview)

Adding an oscillating field **B**₁(t) = B₁cos(ωt)x̂ perpendicular to **B**₀ causes transitions between |↑⟩ and |↓⟩ at the **Rabi frequency**:

$$\Omega_R = \gamma B_1$$

This is how we control qubits!

---

## Quantum Computing Connection

| Physics | Quantum Computing |
|---------|-------------------|
| Static B field | Z rotation gate |
| Larmor precession | Rz(θ) = e^{-iθZ/2} |
| Resonant drive | X, Y rotations |
| Rabi oscillations | Single-qubit gates |

**Qubit control IS spin control in a magnetic field!**

---

## Worked Examples

### Example 1: Larmor Frequency

**Problem:** An electron is in a 1 T magnetic field. Find the Larmor frequency.

**Solution:**
$$\omega_L = \gamma B_0 = \frac{g_s e B_0}{2m_e} = \frac{2 \times 1.6\times10^{-19} \times 1}{2 \times 9.1\times10^{-31}}$$
$$\omega_L \approx 1.76 \times 10^{11} \text{ rad/s} \approx 28 \text{ GHz}$$

### Example 2: Time Evolution

**Problem:** |ψ(0)⟩ = |+⟩ = (|↑⟩ + |↓⟩)/√2. Find |ψ(t)⟩ and the Bloch vector.

**Solution:**
$$|\psi(t)\rangle = \frac{1}{\sqrt{2}}(e^{i\omega_L t/2}|\uparrow\rangle + e^{-i\omega_L t/2}|\downarrow\rangle)$$

Bloch vector:
$$r_x(t) = \cos(\omega_L t), \quad r_y(t) = \sin(\omega_L t), \quad r_z(t) = 0$$

The state precesses on the equator of the Bloch sphere!

### Example 3: Expectation Values

**Problem:** For |ψ(t)⟩ above, find ⟨Ŝₓ(t)⟩.

**Solution:**
$$\langle\hat{S}_x\rangle = \frac{\hbar}{2}\langle\sigma_x\rangle = \frac{\hbar}{2}r_x(t) = \frac{\hbar}{2}\cos(\omega_L t)$$

The x-component of spin oscillates!

---

## Practice Problems

### Direct Application

1. Calculate the Larmor frequency for a proton in a 10 T field.

2. Find the energy splitting ΔE for an electron in Earth's magnetic field (~50 μT).

3. If |ψ(0)⟩ = |↑⟩, what is |ψ(t)⟩?

### Intermediate

4. For |ψ(0)⟩ = cos(π/4)|↑⟩ + sin(π/4)|↓⟩, find ⟨Ŝᵤ(t)⟩.

5. At what time does a spin starting at |+⟩ return to |+⟩?

6. Calculate the transition probability P(↑→↓) after time t in a static z-field.

### Challenging

7. Derive the equation of motion d⟨**Ŝ**⟩/dt = γ⟨**Ŝ**⟩ × **B** from the Heisenberg equation.

8. For **B** = B₀ẑ + B₁x̂ (tilted field), find the new energy levels.

---

## Computational Lab

```python
"""
Day 404 Computational Lab: Spin Dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants (in natural units for simplicity)
hbar = 1.0

def time_evolve_spin(initial_state, omega_L, t_array):
    """
    Time evolution of spin-1/2 in magnetic field along z.
    """
    alpha = initial_state[0]
    beta = initial_state[1]

    states = []
    bloch_vectors = []

    for t in t_array:
        # Time-evolved state
        psi_t = np.array([
            alpha * np.exp(1j * omega_L * t / 2),
            beta * np.exp(-1j * omega_L * t / 2)
        ])

        # Normalize
        psi_t = psi_t / np.linalg.norm(psi_t)
        states.append(psi_t)

        # Bloch vector
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        rx = np.real(np.vdot(psi_t, sigma_x @ psi_t))
        ry = np.real(np.vdot(psi_t, sigma_y @ psi_t))
        rz = np.real(np.vdot(psi_t, sigma_z @ psi_t))

        bloch_vectors.append([rx, ry, rz])

    return np.array(states), np.array(bloch_vectors)

def plot_precession():
    """Visualize Larmor precession on Bloch sphere."""
    # Initial state: |+⟩
    initial = np.array([1, 1], dtype=complex) / np.sqrt(2)
    omega_L = 2 * np.pi  # One full rotation in t=1

    t = np.linspace(0, 1, 200)
    states, bloch_vecs = time_evolve_spin(initial, omega_L, t)

    # 3D plot
    fig = plt.figure(figsize=(12, 5))

    # Bloch sphere trajectory
    ax1 = fig.add_subplot(121, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_surface(x, y, z, alpha=0.1, color='lightblue')

    # Trajectory
    ax1.plot(bloch_vecs[:, 0], bloch_vecs[:, 1], bloch_vecs[:, 2],
             'r-', linewidth=2, label='Precession')
    ax1.scatter([bloch_vecs[0, 0]], [bloch_vecs[0, 1]], [bloch_vecs[0, 2]],
                color='green', s=100, label='Start')

    # Axes
    ax1.quiver(0, 0, 0, 0, 0, 1.3, color='blue', linewidth=2)
    ax1.text(0, 0, 1.5, 'B (z)', fontsize=10)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Larmor Precession on Bloch Sphere')
    ax1.legend()

    # Expectation values vs time
    ax2 = fig.add_subplot(122)
    ax2.plot(t, bloch_vecs[:, 0], 'r-', label='⟨σₓ⟩', linewidth=2)
    ax2.plot(t, bloch_vecs[:, 1], 'g-', label='⟨σᵧ⟩', linewidth=2)
    ax2.plot(t, bloch_vecs[:, 2], 'b-', label='⟨σᵤ⟩', linewidth=2)

    ax2.set_xlabel('Time (in units of 2π/ω_L)')
    ax2.set_ylabel('Expectation Value')
    ax2.set_title('Spin Components vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('larmor_precession.png', dpi=150)
    plt.show()

def compare_initial_states():
    """Compare precession for different initial states."""
    omega_L = 2 * np.pi
    t = np.linspace(0, 1, 200)

    initial_states = {
        '|↑⟩': np.array([1, 0], dtype=complex),
        '|↓⟩': np.array([0, 1], dtype=complex),
        '|+⟩': np.array([1, 1], dtype=complex) / np.sqrt(2),
        '|+i⟩': np.array([1, 1j], dtype=complex) / np.sqrt(2),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, initial) in zip(axes.flat, initial_states.items()):
        _, bloch_vecs = time_evolve_spin(initial, omega_L, t)

        ax.plot(t, bloch_vecs[:, 0], 'r-', label='⟨σₓ⟩')
        ax.plot(t, bloch_vecs[:, 1], 'g-', label='⟨σᵧ⟩')
        ax.plot(t, bloch_vecs[:, 2], 'b-', label='⟨σᵤ⟩')

        ax.set_title(f'Initial: {name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Expectation Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig('spin_dynamics_comparison.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Day 404: Spin Dynamics")
    print("=" * 50)

    print("\nVisualizing Larmor precession...")
    plot_precession()

    print("\nComparing different initial states...")
    compare_initial_states()

    print("\nLab complete!")
```

---

## Summary

| Concept | Formula |
|---------|---------|
| Magnetic moment | **μ** = γ**Ŝ** |
| Hamiltonian | Ĥ = -**μ**·**B** = -γ**Ŝ**·**B** |
| Larmor frequency | ω_L = γB₀ |
| Time evolution | \|ψ(t)⟩ = e^{iω_L tσ_z/2}\|ψ(0)⟩ |
| Precession | d⟨**Ŝ**⟩/dt = γ⟨**Ŝ**⟩ × **B** |

---

## Daily Checklist

- [ ] I can calculate Larmor frequencies
- [ ] I understand spin Hamiltonian in B field
- [ ] I can solve time evolution problems
- [ ] I visualize precession on Bloch sphere
- [ ] I completed the computational lab

---

## Preview: Day 405

Tomorrow we generalize to higher spin values (s = 1, 3/2, ...) and see how the 2×2 spin-1/2 formalism extends to larger matrices.

---

**Next:** [Day_405_Saturday.md](Day_405_Saturday.md) — Higher Spin
