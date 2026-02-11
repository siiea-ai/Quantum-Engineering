# Day 575: Circuit Diagrams

## Overview
**Day 575** | Week 83, Day 1 | Year 1, Month 21 | Circuit Model Fundamentals

Today we introduce the graphical language of quantum circuits—the standard notation for representing quantum computations as sequences of gates acting on qubits.

---

## Learning Objectives

1. Read and interpret quantum circuit diagrams
2. Understand wire conventions (qubits as horizontal lines)
3. Apply the convention that time flows left to right
4. Represent multi-qubit operations in circuit notation
5. Convert between Dirac notation and circuit diagrams
6. Draw circuits for common quantum operations

---

## Core Content

### The Circuit Model

The **circuit model** represents quantum computation as:
- **Wires**: Horizontal lines representing qubits evolving in time
- **Gates**: Boxes or symbols representing unitary operations
- **Time**: Flows from left to right
- **Measurement**: Special symbols at circuit end (or mid-circuit)

### Basic Wire Conventions

```
Single qubit wire:
|ψ⟩ ─────────────────────── |ψ'⟩

Multiple qubit wires (top = qubit 0, or first/most significant):
q₀: |ψ₀⟩ ───────────────────
q₁: |ψ₁⟩ ───────────────────
q₂: |ψ₂⟩ ───────────────────
```

**Convention:** In tensor products $|q_0 q_1 q_2\rangle$, the top wire is $q_0$.

### Single-Qubit Gate Notation

Gates are drawn as boxes on wires:

```
         ┌───┐
|ψ⟩ ─────┤ U ├───── U|ψ⟩
         └───┘

Common gates:
         ┌───┐           ┌───┐           ┌───┐
    ─────┤ X ├─────  ────┤ H ├─────  ────┤ T ├─────
         └───┘           └───┘           └───┘
```

**Special notation for Pauli gates:**
```
    ──[X]──    ──[Y]──    ──[Z]──    ──[H]──
```

### Multi-Qubit Gate Notation

**Controlled gates** use a control dot (●) connected to the target:

```
CNOT (Controlled-X):
q₀: ───●───          Control qubit
       │
q₁: ───⊕───          Target qubit (XOR symbol for X)

Controlled-Z:
q₀: ───●───
       │
q₁: ───●───          (Symmetric - both qubits get phase)

Controlled-U:
q₀: ───●───
       │
       ┌┴┐
q₁: ───┤U├───
       └─┘
```

### The Time-Ordering Convention

**Critical rule:** Time flows left to right.

```
Circuit:
         ┌───┐   ┌───┐
|ψ⟩ ─────┤ A ├───┤ B ├───── |ψ'⟩
         └───┘   └───┘

Means: First apply A, then apply B

Matrix form: |ψ'⟩ = B · A · |ψ⟩
```

$$\boxed{|ψ'\rangle = U_n \cdots U_2 \cdot U_1 |ψ\rangle}$$

**Note:** Gates are read left-to-right in circuit, but matrices multiply right-to-left!

### Multi-Qubit Systems

For an $n$-qubit system, the state space is $(\mathbb{C}^2)^{\otimes n}$.

**Computational basis** (n=3):
$$|000\rangle, |001\rangle, |010\rangle, |011\rangle, |100\rangle, |101\rangle, |110\rangle, |111\rangle$$

**Wire labeling convention:**
```
q₀: ─── (most significant bit, coefficient 2²)
q₁: ─── (coefficient 2¹)
q₂: ─── (least significant bit, coefficient 2⁰)

|q₀q₁q₂⟩ represents number q₀·4 + q₁·2 + q₂·1
```

### Standard Circuit Elements

**Identity (doing nothing):**
```
    ─────────
```

**Initialization:**
```
|0⟩ ────     or    |ψ⟩ ────
```

**Measurement:**
```
    ────[M]────     Projective measurement in Z-basis

    ────╔═╗════     Classical wire output (double line)
        ║M║
        ╚═╝
```

### Barrier and Labels

```
Barrier (synchronization point):
q₀: ───|───
q₁: ───|───
q₂: ───|───

Labels:
         ┌─────────┐
q₀: ─────┤         ├─────
         │  Label  │
q₁: ─────┤         ├─────
         └─────────┘
```

### SWAP Gate

```
Standard SWAP:
q₀: ───✕───
       │
q₁: ───✕───

Alternative notation:
q₀: ───×───
       │
q₁: ───×───
```

### Toffoli (CCNOT) Gate

```
q₀: ───●───     Control 1
       │
q₁: ───●───     Control 2
       │
q₂: ───⊕───     Target
```

---

## Worked Examples

### Example 1: Bell State Preparation Circuit

Draw and analyze the circuit that prepares $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.

**Solution:**

```
         ┌───┐
|0⟩ ─────┤ H ├────●────
         └───┘    │
|0⟩ ──────────────⊕────
```

**Step-by-step analysis:**

1. Initial state: $|00\rangle$

2. After Hadamard on qubit 0:
$$H|0\rangle \otimes |0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

3. After CNOT:
$$\text{CNOT}\left(\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)\right) = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$$

**Matrix form:**
$$|\Phi^+\rangle = \text{CNOT} \cdot (H \otimes I) \cdot |00\rangle$$

### Example 2: Circuit to Matrix Translation

Convert this circuit to a matrix expression:

```
         ┌───┐   ┌───┐
q₀: ─────┤ X ├───┤ H ├───●────
         └───┘   └───┘   │
                   ┌───┐ │
q₁: ───────────────┤ Y ├─⊕────
                   └───┘
```

**Solution:**

Reading left to right, the operations are:
1. X on q₀, I on q₁: $X \otimes I$
2. H on q₀, Y on q₁: $H \otimes Y$
3. CNOT (q₀ control, q₁ target): CNOT

**Total unitary:**
$$U = \text{CNOT} \cdot (H \otimes Y) \cdot (X \otimes I)$$

**Explicit calculation:**
$$X \otimes I = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$$

$$H \otimes Y = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & -i & 0 & -i \\ i & 0 & i & 0 \\ 0 & -i & 0 & i \\ i & 0 & -i & 0 \end{pmatrix}$$

### Example 3: Three-Qubit GHZ State

Draw the circuit to prepare $|GHZ\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$.

**Solution:**

```
         ┌───┐
|0⟩ ─────┤ H ├────●─────●────
         └───┘    │     │
|0⟩ ──────────────⊕─────│────
                        │
|0⟩ ────────────────────⊕────
```

**Verification:**
1. Start: $|000\rangle$
2. After H on q₀: $\frac{1}{\sqrt{2}}(|000\rangle + |100\rangle)$
3. After CNOT(0,1): $\frac{1}{\sqrt{2}}(|000\rangle + |110\rangle)$
4. After CNOT(0,2): $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle) = |GHZ\rangle$

---

## Practice Problems

### Problem 1: Circuit Reading
What is the output state of this circuit?
```
         ┌───┐   ┌───┐
|0⟩ ─────┤ H ├───┤ Z ├────
         └───┘   └───┘
```

### Problem 2: Draw the Circuit
Draw a circuit that transforms $|0\rangle$ to $|1\rangle$ and $|1\rangle$ to $|0\rangle$ (i.e., the X gate).

### Problem 3: Multi-Qubit Circuit
Write out the matrix for this circuit:
```
q₀: ───●───
       │
q₁: ───⊕───●───
           │
q₂: ───────⊕───
```

### Problem 4: State Preparation
Design a circuit to prepare $|\psi\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$ from $|00\rangle$.

---

## Computational Lab

```python
"""Day 575: Circuit Diagrams - Visualization and Simulation"""
import numpy as np
import matplotlib.pyplot as plt

# Define standard gates
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]])
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

# Two-qubit gates
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

def tensor(*gates):
    """Compute tensor product of multiple gates"""
    result = gates[0]
    for gate in gates[1:]:
        result = np.kron(result, gate)
    return result

def apply_circuit(initial_state, gates_list):
    """
    Apply a sequence of gates to initial state.
    gates_list: list of numpy arrays (gates already in matrix form)
    """
    state = initial_state.copy()
    for gate in gates_list:
        state = gate @ state
    return state

def state_to_string(state, n_qubits):
    """Convert state vector to readable string"""
    result = []
    for i, amp in enumerate(state):
        if np.abs(amp) > 1e-10:
            bits = format(i, f'0{n_qubits}b')
            result.append(f"({amp:.4f})|{bits}>")
    return " + ".join(result)

# Example 1: Bell state preparation
print("=" * 60)
print("Example 1: Bell State Preparation")
print("=" * 60)
print("\nCircuit:")
print("         +---+")
print("|0> -----| H |----*----")
print("         +---+    |")
print("|0> --------------X----")
print()

# Initial state |00>
state_00 = np.array([1, 0, 0, 0], dtype=complex)

# Gates: H on q0, then CNOT
gate1 = tensor(H, I)  # H on qubit 0
gate2 = CNOT

# Apply circuit
state_after_H = gate1 @ state_00
state_final = gate2 @ state_after_H

print("Initial state: |00>")
print(f"After H on q0: {state_to_string(state_after_H, 2)}")
print(f"After CNOT:    {state_to_string(state_final, 2)}")
print(f"This is |Phi+> = (|00> + |11>)/sqrt(2)")

# Example 2: GHZ state preparation
print("\n" + "=" * 60)
print("Example 2: GHZ State Preparation")
print("=" * 60)
print("\nCircuit:")
print("         +---+")
print("|0> -----| H |----*-----*----")
print("         +---+    |     |")
print("|0> --------------X-----|----")
print("                        |")
print("|0> --------------------X----")
print()

# Initial state |000>
state_000 = np.zeros(8, dtype=complex)
state_000[0] = 1

# Gates
gate1 = tensor(H, I, I)          # H on qubit 0
gate2 = tensor(CNOT, I)          # CNOT on qubits 0,1
# CNOT on qubits 0,2 (need to construct)
CNOT_02 = np.zeros((8, 8), dtype=complex)
for i in range(8):
    q0, q1, q2 = (i >> 2) & 1, (i >> 1) & 1, i & 1
    if q0 == 1:
        q2_new = 1 - q2
    else:
        q2_new = q2
    j = (q0 << 2) | (q1 << 1) | q2_new
    CNOT_02[j, i] = 1

state1 = gate1 @ state_000
state2 = gate2 @ state1
state_ghz = CNOT_02 @ state2

print("Initial state: |000>")
print(f"After H:           {state_to_string(state1, 3)}")
print(f"After CNOT(0,1):   {state_to_string(state2, 3)}")
print(f"After CNOT(0,2):   {state_to_string(state_ghz, 3)}")
print(f"This is |GHZ> = (|000> + |111>)/sqrt(2)")

# Example 3: Circuit composition
print("\n" + "=" * 60)
print("Example 3: General Circuit Composition")
print("=" * 60)
print("\nCircuit:")
print("         +---+   +---+")
print("q0: -----| X |---| H |----*----")
print("         +---+   +---+    |")
print("                 +---+    |")
print("q1: ------------| Y |----X----")
print("                 +---+")
print()

# Build the circuit
gate1 = tensor(X, I)      # X on q0
gate2 = tensor(H, Y)      # H on q0, Y on q1
gate3 = CNOT              # CNOT

# Total unitary
U_total = gate3 @ gate2 @ gate1

print("Circuit unitary U = CNOT * (H tensor Y) * (X tensor I)")
print("\nU_total =")
print(np.round(U_total, 3))

# Test on |00>
state = np.array([1, 0, 0, 0], dtype=complex)
result = U_total @ state
print(f"\nU|00> = {state_to_string(result, 2)}")

# Visualize gate action on Bloch sphere (single qubit)
def bloch_coords(state):
    """Get Bloch sphere coordinates from single-qubit state"""
    a, b = state[0], state[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a)**2 - np.abs(b)**2
    return x, y, z

print("\n" + "=" * 60)
print("Example 4: Single-Qubit Gate Visualization")
print("=" * 60)

# Track state through circuit: |0> -> H -> T -> H
state0 = np.array([1, 0], dtype=complex)
state1 = H @ state0
state2 = T @ state1
state3 = H @ state2

states = [state0, state1, state2, state3]
labels = ['|0>', 'H|0>', 'TH|0>', 'HTH|0>']

print("\nCircuit: |0> --> H --> T --> H")
print("\nBloch sphere coordinates:")
for state, label in zip(states, labels):
    x, y, z = bloch_coords(state)
    print(f"{label:10s}: ({x:.3f}, {y:.3f}, {z:.3f})")

# Create Bloch sphere visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# Plot axes
ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'k-', alpha=0.3)
ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'k-', alpha=0.3)
ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'k-', alpha=0.3)

# Plot state trajectory
coords = [bloch_coords(s) for s in states]
colors = ['red', 'blue', 'green', 'purple']
for i, (coord, label, color) in enumerate(zip(coords, labels, colors)):
    ax.scatter(*coord, s=100, c=color, label=label)
    if i > 0:
        prev = coords[i-1]
        ax.plot([prev[0], coord[0]], [prev[1], coord[1]],
                [prev[2], coord[2]], c=colors[i], linestyle='--', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Bloch Sphere: Circuit |0> -> H -> T -> H')
ax.legend()
plt.tight_layout()
plt.savefig('day_575_bloch_trajectory.png', dpi=150)
plt.show()

print("\n[Plot saved as day_575_bloch_trajectory.png]")
```

**Expected Output:**
```
============================================================
Example 1: Bell State Preparation
============================================================

Circuit:
         +---+
|0> -----| H |----*----
         +---+    |
|0> --------------X----

Initial state: |00>
After H on q0: (0.7071+0.0000j)|00> + (0.7071+0.0000j)|10>
After CNOT:    (0.7071+0.0000j)|00> + (0.7071+0.0000j)|11>
This is |Phi+> = (|00> + |11>)/sqrt(2)

============================================================
Example 2: GHZ State Preparation
============================================================

Initial state: |000>
After H:           (0.7071+0.0000j)|000> + (0.7071+0.0000j)|100>
After CNOT(0,1):   (0.7071+0.0000j)|000> + (0.7071+0.0000j)|110>
After CNOT(0,2):   (0.7071+0.0000j)|000> + (0.7071+0.0000j)|111>
This is |GHZ> = (|000> + |111>)/sqrt(2)
```

---

## Summary

### Key Conventions

| Element | Notation | Meaning |
|---------|----------|---------|
| Wire | `─────` | Qubit evolving in time |
| Single gate | `[U]` | Apply unitary U |
| Control | `●` | Control qubit |
| Target (X) | `⊕` | Target of CNOT |
| Measurement | `[M]` | Projective measurement |
| Time | Left → Right | Order of operations |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Circuit to matrix | $U = U_n \cdots U_2 \cdot U_1$ |
| Parallel gates | $U_{parallel} = U_A \otimes U_B$ |
| n-qubit state space | $(\mathbb{C}^2)^{\otimes n}$, dimension $2^n$ |
| Computational basis | $\|b_{n-1}\cdots b_1 b_0\rangle$ |

### Key Takeaways

1. **Time flows left to right** in circuit diagrams
2. **Gates multiply right to left** in matrix representation
3. **Top wire** typically represents the most significant qubit
4. **Controlled gates** use vertical lines connecting control (●) to target
5. **CNOT** is the fundamental two-qubit entangling gate
6. Circuits provide a **visual programming language** for quantum algorithms

---

## Daily Checklist

- [ ] I can read and interpret quantum circuit diagrams
- [ ] I understand the time-ordering convention (left to right)
- [ ] I can convert between circuit diagrams and matrix expressions
- [ ] I can draw circuits for Bell and GHZ state preparation
- [ ] I understand multi-qubit wire labeling conventions
- [ ] I ran the computational lab and visualized Bloch sphere trajectories

---

*Next: Day 576 — Circuit Composition*
