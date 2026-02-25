# Day 563: Phase Gates (S, T, and Z-axis Rotations)

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: S gate, T gate, general phase rotations |
| Afternoon | 2.5 hours | Problem solving: Phase kickback and applications |
| Evening | 1.5 hours | Computational lab: Phase gate circuits |

## Learning Objectives

By the end of today, you will be able to:

1. **Define the S and T gates** and understand their relationship to Z
2. **Compute the matrix forms** of S, T, S†, and T† gates
3. **Implement Z-axis rotations** using phase gates
4. **Explain phase kickback** and its role in quantum algorithms
5. **Understand the Clifford hierarchy** (Z, S, T structure)
6. **Apply phase gates** in practical quantum circuits

---

## Core Content

### 1. The Phase Gate Family

Phase gates rotate qubits around the Z-axis of the Bloch sphere. They leave $|0\rangle$ unchanged while adding a phase to $|1\rangle$:

$$P(\phi) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$$

**Special cases:**
| Gate | φ | Matrix | Relation |
|------|---|--------|----------|
| I | 0 | diag(1, 1) | Identity |
| Z | π | diag(1, -1) | Pauli Z |
| S | π/2 | diag(1, i) | √Z |
| T | π/4 | diag(1, e^(iπ/4)) | √S |

### 2. The S Gate (Phase Gate)

$$\boxed{S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = P(\pi/2)}$$

**Key properties:**
- $S^2 = Z$ (S is the square root of Z)
- $S^4 = I$
- $S|0\rangle = |0\rangle$
- $S|1\rangle = i|1\rangle$

**Action on superposition:**
$$S|+\rangle = S \cdot \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle) = |+_y\rangle$$

The S gate rotates $|+\rangle$ to $|+_y\rangle$, the +1 eigenstate of Y!

**Relation to Z:**
$$\boxed{S = \sqrt{Z} = e^{i\pi/4}R_z(\pi/2)}$$

More precisely: $S = e^{i\pi/4}\begin{pmatrix} e^{-i\pi/4} & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$

### 3. The S† (S-dagger) Gate

$$\boxed{S^\dagger = S^{-1} = S^3 = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix} = P(-\pi/2)}$$

**Properties:**
- $SS^\dagger = S^\dagger S = I$
- $(S^\dagger)^2 = Z$
- $S^\dagger|1\rangle = -i|1\rangle$

### 4. The T Gate (π/8 Gate)

$$\boxed{T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = P(\pi/4)}$$

**Key properties:**
- $T^2 = S$ (T is the square root of S)
- $T^4 = Z$
- $T^8 = I$
- $T|1\rangle = e^{i\pi/4}|1\rangle = \frac{1+i}{\sqrt{2}}|1\rangle$

**Why "π/8 gate"?**
$$T = e^{i\pi/8}R_z(\pi/4) = e^{i\pi/8}\begin{pmatrix} e^{-i\pi/8} & 0 \\ 0 & e^{i\pi/8} \end{pmatrix}$$

The total accumulated phase is π/4, but the half-angle in the rotation is π/8.

### 5. The T† Gate

$$\boxed{T^\dagger = T^{-1} = T^7 = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix} = P(-\pi/4)}$$

### 6. The Complete Hierarchy

The phase gates form a nested hierarchy:

$$\boxed{I \xleftarrow{T^8} T \xleftarrow{T^2} S \xleftarrow{S^2} Z \xleftarrow{Z^2} I}$$

Or in terms of powers:
- $T^1$: T gate
- $T^2$: S gate
- $T^4$: Z gate
- $T^8$: I (identity)

### 7. Bloch Sphere Interpretation

All phase gates are **Z-axis rotations**:

$$R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

Comparison with P(φ):
$$P(\phi) = e^{i\phi/2}R_z(\phi)$$

The gates differ by a global phase, which is physically unobservable for single gates but matters in controlled operations.

**Rotation angles:**
| Gate | P(φ) angle | Rz(θ) angle |
|------|------------|-------------|
| Z | π | π |
| S | π/2 | π/2 |
| T | π/4 | π/4 |

### 8. Phase Kickback

**Phase kickback** is a crucial quantum phenomenon where a phase intended for a target qubit "kicks back" to the control qubit.

**Setup:** Consider a controlled-U operation where U has eigenvector $|u\rangle$ with eigenvalue $e^{i\phi}$:
$$U|u\rangle = e^{i\phi}|u\rangle$$

**Controlled-U on** $|+\rangle|u\rangle$:
$$\text{C-}U(|+\rangle|u\rangle) = \text{C-}U\left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}|u\rangle\right)$$

$$= \frac{1}{\sqrt{2}}(|0\rangle|u\rangle + |1\rangle U|u\rangle)$$

$$= \frac{1}{\sqrt{2}}(|0\rangle|u\rangle + e^{i\phi}|1\rangle|u\rangle)$$

$$= \frac{1}{\sqrt{2}}(|0\rangle + e^{i\phi}|1\rangle)|u\rangle$$

**Result:** The phase $e^{i\phi}$ has "kicked back" to the control qubit!

**Key insight:** The target qubit remains unchanged (it's an eigenstate), but the control qubit acquires a relative phase.

### 9. Phase Kickback with Phase Gates

**Example: Controlled-S on** $|+\rangle|1\rangle$:

Since $S|1\rangle = i|1\rangle$, we have eigenvalue $e^{i\pi/2} = i$.

$$\text{C-}S(|+\rangle|1\rangle) = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)|1\rangle = |+_y\rangle|1\rangle$$

The control qubit rotated from $|+\rangle$ to $|+_y\rangle$!

### 10. Applications of Phase Gates

| Application | Role of Phase Gates |
|-------------|---------------------|
| Quantum Fourier Transform | T gates create phase relationships |
| Phase Estimation | Phase kickback extracts eigenvalue information |
| Grover's Algorithm | Phase oracle marks solutions |
| Quantum Error Correction | S and T in Clifford hierarchy |
| Magic State Distillation | T gate enables universal computation |

### 11. Clifford vs Non-Clifford

The **Clifford group** consists of gates that map Pauli operators to Pauli operators under conjugation:
$$C \in \text{Clifford} \iff CPC^\dagger \in \{\pm I, \pm X, \pm Y, \pm Z\} \text{ for all Pauli } P$$

**Clifford gates:** H, S, CNOT (and compositions)

**T is NOT Clifford!**
$$TXT^\dagger = e^{i\pi/4}\begin{pmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{pmatrix} \neq \text{Pauli}$$

**Importance:** Clifford circuits can be efficiently simulated classically (Gottesman-Knill theorem). Adding T gates enables universal quantum computation!

---

## Quantum Computing Connection

Phase gates are essential for quantum advantage:

1. **QFT structure:** The quantum Fourier transform uses controlled phase rotations
2. **Universal gate sets:** {H, T, CNOT} forms a universal set
3. **Magic states:** T gates can be implemented via magic state injection
4. **Error correction threshold:** T gates are the "hardest" to implement fault-tolerantly

---

## Worked Examples

### Example 1: Verifying S² = Z

**Problem:** Show that $S^2 = Z$ by matrix multiplication.

**Solution:**
$$S^2 = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & i^2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = Z \checkmark$$

### Example 2: Phase Kickback Calculation

**Problem:** Compute the effect of controlled-T on $|+\rangle|1\rangle$.

**Solution:**

The eigenvalue of T for $|1\rangle$ is $e^{i\pi/4}$.

Initial state:
$$|+\rangle|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)|1\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |11\rangle)$$

After controlled-T:
$$\text{C-}T|+\rangle|1\rangle = \frac{1}{\sqrt{2}}(|01\rangle + e^{i\pi/4}|11\rangle)$$

$$= \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)|1\rangle$$

The control qubit has acquired the phase $e^{i\pi/4}$!

### Example 3: S Gate on Y-eigenstates

**Problem:** Find $S|+_y\rangle$ where $|+_y\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$.

**Solution:**
$$S|+_y\rangle = S \cdot \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$$

$$= \frac{1}{\sqrt{2}}(S|0\rangle + i \cdot S|1\rangle)$$

$$= \frac{1}{\sqrt{2}}(|0\rangle + i \cdot i|1\rangle)$$

$$= \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = |-\rangle$$

The S gate rotates Y-eigenstates to X-eigenstates!

---

## Practice Problems

### Direct Application

1. Calculate $T^2$ and verify it equals S.

2. Compute $S|−\rangle$ where $|−\rangle = \frac{1}{\sqrt{2}}(|0\rangle − |1\rangle)$.

3. What is $T^4$? Verify by computing $T^4 = (T^2)^2 = S^2$.

### Intermediate

4. **Phase gate cycle:** Show that the sequence of states $|+\rangle \xrightarrow{S} |+_y\rangle \xrightarrow{S} |-\rangle \xrightarrow{S} |-_y\rangle \xrightarrow{S} |+\rangle$ forms a cycle, where $|-_y\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)$.

5. **Conjugation:** Compute $SXS^\dagger$. Is S a Clifford gate?

6. Prove that $P(\phi_1)P(\phi_2) = P(\phi_1 + \phi_2)$. Why does this make sense geometrically?

### Challenging

7. **T-gate conjugation:** Compute $TXT^\dagger$ explicitly. Explain why this shows T is not in the Clifford group.

8. **Eigenvalue extraction:** In the phase estimation algorithm, we use controlled-$U^{2^k}$ operations. If $U = T$, compute controlled-$T^{2^k}$ for k = 0, 1, 2, 3 and identify the pattern.

9. **Phase polynomial:** Any single-qubit state can be written as $|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$. After applying T, what are the new parameters θ' and φ'?

---

## Computational Lab: Phase Gates and Kickback

```python
"""
Day 563: Phase Gates Simulation
Exploring S, T gates, phase kickback, and Clifford structure
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define basic gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase gates
S = np.array([[1, 0], [0, 1j]], dtype=complex)
S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
T_dag = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

def P(phi):
    """General phase gate P(φ)."""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)
ket_plus_y = (ket_0 + 1j * ket_1) / np.sqrt(2)
ket_minus_y = (ket_0 - 1j * ket_1) / np.sqrt(2)

print("=" * 60)
print("PHASE GATES: S AND T")
print("=" * 60)

# 1. Verify gate hierarchies
print("\n1. Gate Hierarchy Verification:")
print(f"   S² = Z: {np.allclose(S @ S, Z)}")
print(f"   S⁴ = I: {np.allclose(np.linalg.matrix_power(S, 4), I)}")
print(f"   T² = S: {np.allclose(T @ T, S)}")
print(f"   T⁴ = Z: {np.allclose(np.linalg.matrix_power(T, 4), Z)}")
print(f"   T⁸ = I: {np.allclose(np.linalg.matrix_power(T, 8), I)}")

# 2. Verify inverse relations
print("\n2. Inverse Relations:")
print(f"   S·S† = I: {np.allclose(S @ S_dag, I)}")
print(f"   T·T† = I: {np.allclose(T @ T_dag, I)}")
print(f"   S† = S³: {np.allclose(S_dag, np.linalg.matrix_power(S, 3))}")
print(f"   T† = T⁷: {np.allclose(T_dag, np.linalg.matrix_power(T, 7))}")

# 3. Action on basis states
print("\n3. Action on Computational Basis:")
print(f"   S|0⟩ = |0⟩: {np.allclose(S @ ket_0, ket_0)}")
print(f"   S|1⟩ = i|1⟩: {np.allclose(S @ ket_1, 1j * ket_1)}")
print(f"   T|0⟩ = |0⟩: {np.allclose(T @ ket_0, ket_0)}")
print(f"   T|1⟩ = exp(iπ/4)|1⟩: {np.allclose(T @ ket_1, np.exp(1j*np.pi/4) * ket_1)}")

# 4. Action on superposition states
print("\n4. Action on Superposition States:")
print(f"   S|+⟩ = |+y⟩: {np.allclose(S @ ket_plus, ket_plus_y)}")
print(f"   S|-⟩ = |-y⟩: {np.allclose(S @ ket_minus, ket_minus_y)}")
print(f"   S|+y⟩ = |-⟩: {np.allclose(S @ ket_plus_y, ket_minus)}")
print(f"   S|-y⟩ = |+⟩: {np.allclose(S @ ket_minus_y, ket_plus)}")

# 5. S gate cycle
print("\n5. S Gate Cycle (|+⟩ → |+y⟩ → |-⟩ → |-y⟩ → |+⟩):")
state = ket_plus.copy()
states_names = ['|+⟩', '|+y⟩', '|-⟩', '|-y⟩', '|+⟩']
expected = [ket_plus, ket_plus_y, ket_minus, ket_minus_y, ket_plus]
for i in range(5):
    print(f"   After S^{i}: {states_names[i]}, matches: {np.allclose(state, expected[i])}")
    state = S @ state

# 6. Clifford group test
print("\n" + "=" * 60)
print("CLIFFORD GROUP ANALYSIS")
print("=" * 60)

def is_pauli_up_to_phase(M):
    """Check if M is a Pauli matrix up to global phase."""
    paulis = [I, X, Y, Z, -I, -X, -Y, -Z, 1j*I, 1j*X, 1j*Y, 1j*Z,
              -1j*I, -1j*X, -1j*Y, -1j*Z]
    for P in paulis:
        if np.allclose(M, P):
            return True
    return False

print("\n6. Testing if gates are Clifford (map Paulis to Paulis):")
for gate_name, gate in [('H', H), ('S', S), ('T', T)]:
    is_clifford = True
    for pauli_name, pauli in [('X', X), ('Y', Y), ('Z', Z)]:
        conjugated = gate @ pauli @ gate.conj().T
        pauli_result = is_pauli_up_to_phase(conjugated)
        if not pauli_result:
            is_clifford = False
    print(f"   {gate_name} is Clifford: {is_clifford}")

print("\n7. Explicit conjugation by T:")
TXT_dag = T @ X @ T_dag
TYT_dag = T @ Y @ T_dag
TZT_dag = T @ Z @ T_dag
print(f"   TXT† = \n{TXT_dag}")
print(f"   TZT† = Z: {np.allclose(TZT_dag, Z)}")
print("   Note: TXT† is NOT a Pauli matrix!")

# 8. Phase kickback demonstration
print("\n" + "=" * 60)
print("PHASE KICKBACK DEMONSTRATION")
print("=" * 60)

def controlled_gate(U):
    """Create controlled-U gate (control first, target second)."""
    return np.block([[I, np.zeros((2,2))],
                     [np.zeros((2,2)), U]])

def state_to_bloch(psi):
    """Convert pure state to Bloch coordinates."""
    psi = psi.flatten()
    rho = np.outer(psi, psi.conj())
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(Y @ rho))
    z = np.real(np.trace(Z @ rho))
    return np.array([x, y, z])

# Phase kickback with various controlled phase gates
print("\n8. Phase Kickback: Control = |+⟩, Target = |1⟩")
print("   Control qubit Bloch vector after controlled-P(φ):")

control_target = np.kron(ket_plus, ket_1)

for gate_name, phi in [('Z (φ=π)', np.pi), ('S (φ=π/2)', np.pi/2),
                       ('T (φ=π/4)', np.pi/4), ('T† (φ=-π/4)', -np.pi/4)]:
    CU = controlled_gate(P(phi))
    result = CU @ control_target

    # Extract control qubit state (trace out target)
    # Since target is |1⟩, control state is just the last two components normalized
    control_state = result[2:4] / np.linalg.norm(result[2:4])
    bloch = state_to_bloch(control_state)

    expected_phase = np.exp(1j * phi)
    expected_state = (ket_0 + expected_phase * ket_1) / np.sqrt(2)

    print(f"   {gate_name}: Bloch = ({bloch[0]:.3f}, {bloch[1]:.3f}, {bloch[2]:.3f})")
    print(f"      Expected |0⟩ + e^(i·{phi/np.pi:.2f}π)|1⟩")

# 9. Visualize phase kickback on Bloch sphere
print("\n" + "=" * 60)
print("BLOCH SPHERE VISUALIZATION")
print("=" * 60)

fig = plt.figure(figsize=(15, 5))

# Plot 1: S gate cycle
ax1 = fig.add_subplot(131, projection='3d')

# Draw Bloch sphere
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# S gate cycle: |+⟩ → |+y⟩ → |-⟩ → |-y⟩ → |+⟩
cycle_states = [ket_plus, ket_plus_y, ket_minus, ket_minus_y]
cycle_names = ['|+⟩', '|+y⟩', '|-⟩', '|-y⟩']
colors = ['red', 'green', 'blue', 'purple']

for i, (state, name, color) in enumerate(zip(cycle_states, cycle_names, colors)):
    bloch = state_to_bloch(state)
    ax1.scatter(*bloch, c=color, s=100, marker='o')
    ax1.text(bloch[0]*1.3, bloch[1]*1.3, bloch[2]*1.3, name, fontsize=10)

    # Draw arrow to next state
    next_state = cycle_states[(i+1) % 4]
    next_bloch = state_to_bloch(next_state)
    ax1.quiver(bloch[0], bloch[1], bloch[2],
               (next_bloch[0]-bloch[0])*0.9,
               (next_bloch[1]-bloch[1])*0.9,
               (next_bloch[2]-bloch[2])*0.9,
               color='gray', arrow_length_ratio=0.2, alpha=0.7)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('S Gate Cycle\n(π/2 rotations about Z)')

# Plot 2: T gate creates smaller rotations
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# Show T gate sequence from |+⟩
state = ket_plus.copy()
t_states = [state]
for i in range(8):
    state = T @ state
    t_states.append(state)

for i, state in enumerate(t_states):
    bloch = state_to_bloch(state)
    alpha = 0.3 + 0.7 * (i / 8)
    ax2.scatter(*bloch, c='blue', s=50, alpha=alpha, marker='o')
    if i < len(t_states) - 1:
        next_bloch = state_to_bloch(t_states[i+1])
        ax2.plot([bloch[0], next_bloch[0]], [bloch[1], next_bloch[1]],
                [bloch[2], next_bloch[2]], 'b-', alpha=0.3)

ax2.scatter(*state_to_bloch(ket_plus), c='red', s=100, marker='*', label='Start/End')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('T Gate Sequence from |+⟩\n(π/4 steps, 8 applications → |+⟩)')
ax2.legend()

# Plot 3: Phase kickback visualization
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# Show control qubit state after C-P(φ) for various φ
phi_values = np.linspace(0, 2*np.pi, 17)
kickback_states = []

for phi in phi_values:
    # After phase kickback: (|0⟩ + e^(iφ)|1⟩)/√2
    kicked_state = (ket_0 + np.exp(1j * phi) * ket_1) / np.sqrt(2)
    kickback_states.append(kicked_state)
    bloch = state_to_bloch(kicked_state)
    ax3.scatter(*bloch, c=plt.cm.hsv(phi / (2*np.pi)), s=50, alpha=0.8)

# Draw equator (where kickback states lie)
theta = np.linspace(0, 2*np.pi, 100)
eq_x = np.cos(theta)
eq_y = np.sin(theta)
eq_z = np.zeros_like(theta)
ax3.plot(eq_x, eq_y, eq_z, 'k--', alpha=0.5, label='Equator')

ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Phase Kickback States\nColor = kicked phase φ')
ax3.legend()

plt.tight_layout()
plt.savefig('phase_gates_bloch.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: phase_gates_bloch.png")

# 10. Phase estimation preview
print("\n" + "=" * 60)
print("PHASE ESTIMATION PREVIEW")
print("=" * 60)

print("\n10. Controlled-T^(2^k) for phase estimation:")
print("    If T|1⟩ = e^(iπ/4)|1⟩, then T^(2^k)|1⟩ = e^(i·2^k·π/4)|1⟩")

for k in range(4):
    power = 2**k
    phase = power * np.pi / 4
    phase_normalized = (phase % (2*np.pi)) / np.pi
    T_power = np.linalg.matrix_power(T, power)
    eigenvalue = T_power[1, 1]

    print(f"    k={k}: T^{power}|1⟩ = e^(i·{power}π/4)|1⟩ = e^(i·{phase_normalized:.2f}π)|1⟩")
    print(f"           Actual eigenvalue: {eigenvalue:.4f}")

print("\n    In phase estimation, these phases create interference patterns")
print("    that reveal the eigenvalue with high precision!")

# 11. Interference with T gates
print("\n" + "=" * 60)
print("INTERFERENCE PATTERNS")
print("=" * 60)

fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (gate_name, gate) in enumerate([('Z', Z), ('S', S), ('T', T)]):
    n_applications = 16 if gate_name == 'T' else (4 if gate_name == 'S' else 2)

    probabilities = []

    for n in range(n_applications + 1):
        # Circuit: |0⟩ → H → Gate^n → H → measure
        state = ket_0.copy()
        state = H @ state
        for _ in range(n):
            state = gate @ state
        state = H @ state
        prob_0 = np.abs(state[0, 0])**2
        probabilities.append(prob_0)

    axes[idx].bar(range(n_applications + 1), probabilities, color='steelblue', alpha=0.7)
    axes[idx].set_xlabel(f'Number of {gate_name} applications')
    axes[idx].set_ylabel('P(|0⟩)')
    axes[idx].set_title(f'Interference: H → {gate_name}^n → H')
    axes[idx].set_ylim([0, 1.1])
    axes[idx].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('phase_gate_interference.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: phase_gate_interference.png")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| S gate | $S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = P(\pi/2)$ |
| T gate | $T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = P(\pi/4)$ |
| General phase | $P(\phi) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$ |
| Hierarchy | $T^2 = S$, $S^2 = Z$, $T^8 = I$ |
| Z-axis rotation | $R_z(\theta) = e^{-i\theta/2}P(\theta)$ |
| Phase kickback | $\text{C-}U\|+\rangle\|u\rangle = \frac{1}{\sqrt{2}}(\|0\rangle + e^{i\phi}\|1\rangle)\|u\rangle$ |
| S on $\|+\rangle$ | $S\|+\rangle = \|+_y\rangle$ |

### Main Takeaways

1. **Phase gates rotate about Z:** All phase gates P(φ) are Z-axis rotations by angle φ
2. **Hierarchical structure:** T → S → Z through squaring (T² = S, S² = Z)
3. **Phase kickback:** Eigenvalue phases transfer to control qubits - key for quantum algorithms
4. **Clifford boundary:** S is Clifford, T is not - T enables universal computation
5. **Phase estimation:** Controlled phase gates extract eigenvalue information

---

## Daily Checklist

- [ ] I can write S and T matrices from memory
- [ ] I understand the hierarchy T² = S, S² = Z
- [ ] I can explain phase kickback conceptually and mathematically
- [ ] I understand why T is non-Clifford while S is Clifford
- [ ] I completed the phase gate simulation lab
- [ ] I solved at least 3 practice problems

---

## Preview of Day 564

Tomorrow we study **rotation gates** Rx(θ), Ry(θ), Rz(θ) in their full generality. We'll derive the exponential form $R_j(\theta) = e^{-i\theta\sigma_j/2}$ and understand how any angle rotation can be achieved. These continuous rotations connect to the discrete gates (Pauli, S, T) as special cases.
