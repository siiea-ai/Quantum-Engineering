# Day 406: Week 58 Review — Spin as Qubits

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Comprehensive review |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Qiskit lab |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Synthesis |

---

## Week 58 Summary

This week we discovered that spin-1/2 IS the qubit:

| Day | Topic | Key Result |
|-----|-------|------------|
| 400 | Stern-Gerlach | Spin discovered experimentally |
| 401 | Spin-1/2 Formalism | 2D Hilbert space |
| 402 | Pauli Matrices | σₓ, σᵧ, σᵤ algebra |
| 403 | Bloch Sphere | Geometric visualization |
| 404 | Spin Dynamics | Larmor precession |
| 405 | Higher Spin | Qutrits and beyond |

---

## Master Formula Sheet

### Pauli Matrices

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### Spin Operators

$$\hat{S}_i = \frac{\hbar}{2}\sigma_i$$

### Key Relations

$$\sigma_i^2 = I, \quad \sigma_i\sigma_j = \delta_{ij}I + i\varepsilon_{ijk}\sigma_k$$

$$[\sigma_i, \sigma_j] = 2i\varepsilon_{ijk}\sigma_k, \quad \{\sigma_i, \sigma_j\} = 2\delta_{ij}I$$

### Bloch Sphere

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

$$\rho = \frac{1}{2}(I + \mathbf{r}\cdot\boldsymbol{\sigma})$$

### Spin Dynamics

$$\hat{H} = -\gamma\hat{\mathbf{S}}\cdot\mathbf{B} = -\frac{\omega_L\hbar}{2}\sigma_z$$

$$|\psi(t)\rangle = e^{i\omega_L t\sigma_z/2}|\psi(0)\rangle$$

---

## The Spin-Qubit Dictionary

| Spin Physics | Quantum Computing |
|--------------|-------------------|
| \|↑⟩, \|↓⟩ | \|0⟩, \|1⟩ |
| σₓ | X gate (NOT) |
| σᵧ | Y gate |
| σᵤ | Z gate |
| e^{-iθσₓ/2} | Rx(θ) |
| e^{-iθσᵧ/2} | Ry(θ) |
| e^{-iθσᵤ/2} | Rz(θ) |
| (I + σₓ)/√2 | H (Hadamard) |
| Sᵤ measurement | Z-basis measurement |
| Sₓ measurement | X-basis measurement |

---

## Comprehensive Qiskit Lab

```python
"""
Day 406 Computational Lab: Spin-1/2 as Qubits with Qiskit
"""

import numpy as np
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector, DensityMatrix

def demonstrate_pauli_gates():
    """Show how Pauli matrices are quantum gates."""
    print("Pauli Matrices as Quantum Gates")
    print("=" * 50)

    # X gate (σₓ)
    qc_x = QuantumCircuit(1)
    qc_x.x(0)

    sv_x = Statevector.from_instruction(qc_x)
    print(f"\nX|0⟩ = {sv_x.data}")  # Should be |1⟩

    # Y gate (σᵧ)
    qc_y = QuantumCircuit(1)
    qc_y.y(0)

    sv_y = Statevector.from_instruction(qc_y)
    print(f"Y|0⟩ = {sv_y.data}")  # Should be i|1⟩

    # Z gate (σᵤ)
    qc_z = QuantumCircuit(1)
    qc_z.z(0)

    sv_z = Statevector.from_instruction(qc_z)
    print(f"Z|0⟩ = {sv_z.data}")  # Should be |0⟩

def demonstrate_rotations():
    """Show rotation gates as spin rotations."""
    print("\nRotation Gates = Spin Rotations")
    print("=" * 50)

    # Rz(π) should give global phase on |0⟩
    # But on |+⟩ = (|0⟩ + |1⟩)/√2, it gives |-⟩

    qc = QuantumCircuit(1)
    qc.h(0)      # Create |+⟩
    qc.rz(np.pi, 0)  # Rz(π)

    sv = Statevector.from_instruction(qc)
    print(f"\nRz(π)|+⟩ = {sv.data}")  # Should be |-⟩ up to phase

    # Rx(π) is X gate (up to global phase)
    qc2 = QuantumCircuit(1)
    qc2.rx(np.pi, 0)

    sv2 = Statevector.from_instruction(qc2)
    print(f"Rx(π)|0⟩ = {sv2.data}")  # Should be -i|1⟩

    # Ry(π/2) creates |+⟩ from |0⟩
    qc3 = QuantumCircuit(1)
    qc3.ry(np.pi/2, 0)

    sv3 = Statevector.from_instruction(qc3)
    print(f"Ry(π/2)|0⟩ = {sv3.data}")  # Should be |+⟩

def bloch_sphere_visualization():
    """Visualize states on Bloch sphere."""
    print("\nBloch Sphere Visualization")
    print("=" * 50)

    states_to_plot = []
    labels = []

    # |0⟩
    sv0 = Statevector([1, 0])
    states_to_plot.append(sv0)
    labels.append('|0⟩')

    # |1⟩
    sv1 = Statevector([0, 1])
    states_to_plot.append(sv1)
    labels.append('|1⟩')

    # |+⟩
    sv_plus = Statevector([1, 1]) / np.sqrt(2)
    states_to_plot.append(sv_plus)
    labels.append('|+⟩')

    # |+i⟩
    sv_plus_i = Statevector([1, 1j]) / np.sqrt(2)
    states_to_plot.append(sv_plus_i)
    labels.append('|+i⟩')

    # Plot each
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, sv, label in zip(axes, states_to_plot, labels):
        # Manual Bloch vector calculation
        rho = DensityMatrix(sv)
        r = np.array([
            np.real(np.trace(rho.data @ np.array([[0,1],[1,0]]))),
            np.real(np.trace(rho.data @ np.array([[0,-1j],[1j,0]]))),
            np.real(np.trace(rho.data @ np.array([[1,0],[0,-1]])))
        ])

        ax.set_title(f'{label}\nr = ({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f})')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('qiskit_bloch_states.png', dpi=150)
    plt.show()

    print("States plotted! See qiskit_bloch_states.png")

def larmor_precession_simulation():
    """Simulate Larmor precession using Rz gates."""
    print("\nSimulating Larmor Precession")
    print("=" * 50)

    # Start in |+⟩ state
    # Apply Rz(θ) for various θ to simulate precession

    angles = np.linspace(0, 2*np.pi, 100)
    sx_expect = []
    sy_expect = []
    sz_expect = []

    for theta in angles:
        qc = QuantumCircuit(1)
        qc.h(0)  # Prepare |+⟩
        qc.rz(theta, 0)  # Rz rotation (Larmor precession)

        sv = Statevector.from_instruction(qc)
        rho = DensityMatrix(sv)

        # Expectation values
        sx = np.real(np.trace(rho.data @ np.array([[0,1],[1,0]])))
        sy = np.real(np.trace(rho.data @ np.array([[0,-1j],[1j,0]])))
        sz = np.real(np.trace(rho.data @ np.array([[1,0],[0,-1]])))

        sx_expect.append(sx)
        sy_expect.append(sy)
        sz_expect.append(sz)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(angles/np.pi, sx_expect, 'r-', label='⟨X⟩', linewidth=2)
    ax.plot(angles/np.pi, sy_expect, 'g-', label='⟨Y⟩', linewidth=2)
    ax.plot(angles/np.pi, sz_expect, 'b-', label='⟨Z⟩', linewidth=2)

    ax.set_xlabel('Rotation angle θ/π (= ω_L t)', fontsize=12)
    ax.set_ylabel('Expectation value', fontsize=12)
    ax.set_title('Larmor Precession: Rz(θ)|+⟩', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('larmor_qiskit.png', dpi=150)
    plt.show()

def measurement_in_different_bases():
    """Demonstrate measurement in X, Y, Z bases."""
    print("\nMeasurement in Different Bases")
    print("=" * 50)

    simulator = AerSimulator()
    shots = 10000

    # State: |+⟩
    initial_state = QuantumCircuit(1)
    initial_state.h(0)

    # Z-basis measurement (direct)
    qc_z = initial_state.copy()
    qc_z.measure_all()

    job_z = simulator.run(transpile(qc_z, simulator), shots=shots)
    counts_z = job_z.result().get_counts()
    print(f"\n|+⟩ measured in Z-basis: {counts_z}")
    print(f"  P(0) ≈ {counts_z.get('0', 0)/shots:.3f}")
    print(f"  P(1) ≈ {counts_z.get('1', 0)/shots:.3f}")

    # X-basis measurement (apply H then measure)
    qc_x = initial_state.copy()
    qc_x.h(0)  # Transform to Z-basis
    qc_x.measure_all()

    job_x = simulator.run(transpile(qc_x, simulator), shots=shots)
    counts_x = job_x.result().get_counts()
    print(f"\n|+⟩ measured in X-basis: {counts_x}")
    print(f"  P(+) ≈ {counts_x.get('0', 0)/shots:.3f}")
    print(f"  P(-) ≈ {counts_x.get('1', 0)/shots:.3f}")

    # Y-basis measurement (apply S†H then measure)
    qc_y = initial_state.copy()
    qc_y.sdg(0)  # S†
    qc_y.h(0)
    qc_y.measure_all()

    job_y = simulator.run(transpile(qc_y, simulator), shots=shots)
    counts_y = job_y.result().get_counts()
    print(f"\n|+⟩ measured in Y-basis: {counts_y}")
    print(f"  P(+i) ≈ {counts_y.get('0', 0)/shots:.3f}")
    print(f"  P(-i) ≈ {counts_y.get('1', 0)/shots:.3f}")

def spin_gate_identity():
    """Demonstrate exp(-iθn·σ/2) = cos(θ/2)I - i sin(θ/2)(n·σ)."""
    print("\nSpin Rotation Formula")
    print("=" * 50)

    theta = np.pi / 3

    # Rx(θ) = exp(-iθσₓ/2) = cos(θ/2)I - i sin(θ/2)σₓ
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    # Get unitary matrix
    from qiskit.quantum_info import Operator
    U = Operator(qc).data

    print(f"\nRx({theta:.4f}) matrix:")
    print(U)

    # Verify formula
    I = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    U_formula = np.cos(theta/2) * I - 1j * np.sin(theta/2) * sigma_x

    print(f"\ncos(θ/2)I - i sin(θ/2)σₓ:")
    print(U_formula)

    print(f"\nMatrices match: {np.allclose(U, U_formula)}")

if __name__ == "__main__":
    print("Day 406: Spin-1/2 as Qubits - Qiskit Lab")
    print("=" * 60)

    demonstrate_pauli_gates()
    demonstrate_rotations()
    bloch_sphere_visualization()
    larmor_precession_simulation()
    measurement_in_different_bases()
    spin_gate_identity()

    print("\nWeek 58 complete!")
    print("Spin-1/2 physics IS qubit physics!")
```

---

## Week 58 Synthesis

### The Fundamental Connection

**Quantum mechanics was discovered by studying spin. Quantum computing uses spin.**

Every single-qubit gate is a rotation in the Bloch sphere:
$$U = e^{-i\theta\hat{n}\cdot\boldsymbol{\sigma}/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}(\hat{n}\cdot\boldsymbol{\sigma})$$

### Universal Single-Qubit Gates

Any rotation can be decomposed as:
$$U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$$

This is the **ZYZ decomposition** used by quantum compilers.

---

## Preview: Week 59

Next week we learn to add angular momenta—essential for understanding:
- Multi-qubit systems
- Entanglement from the spin perspective
- The singlet state (Bell state!)
- Coupling atoms and spins

---

## Week 58 Checklist

- [ ] I can write all Pauli matrices from memory
- [ ] I understand the Bloch sphere representation
- [ ] I can calculate spin dynamics in B fields
- [ ] I know higher spin generalizations
- [ ] I see the spin ↔ qubit correspondence
- [ ] I completed the Qiskit lab

---

**Next Week:** [Week_59_Addition_Angular_Momentum/README.md](../Week_59_Addition_Angular_Momentum/README.md) — Addition of Angular Momenta
