# Day 578: Classical Control

## Overview
**Day 578** | Week 83, Day 4 | Year 1, Month 21 | Feedforward and Adaptive Circuits

Today we explore classical control in quantum circuits—how measurement outcomes can control subsequent operations through feedforward, enabling adaptive quantum algorithms and error correction protocols.

---

## Learning Objectives

1. Implement if-then gates controlled by classical bits
2. Understand feedforward operations in quantum circuits
3. Apply the measure-and-correct paradigm
4. Design adaptive quantum circuits
5. Connect classical control to teleportation and error correction
6. Distinguish classical control from quantum control

---

## Core Content

### Classical vs Quantum Control

**Quantum Control (CNOT):**
- Control qubit remains in superposition
- Creates entanglement
- Coherent operation

```
q₀: ───●───    (quantum superposition on control)
       │
q₁: ───⊕───
```

**Classical Control:**
- Control is a classical bit (measurement outcome)
- No entanglement possible
- Operates on collapsed states

```
q₀: ───[M]═══●═══    (classical bit controls)
             ║
q₁: ─────────X───
```

### Feedforward Notation

Classical control uses double lines (`═`) and classical-to-quantum control:

```
Standard notation:
         ┌───┐
q₀: ─────┤ H ├───[M]═══●═══════
         └───┘         ║
                     ┌─╨─┐
q₁: ─────────────────┤ X ├─────
                     └───┘

Alternative with explicit c-register:
q₀: ───[H]───[M]
              │
c₀: ══════════●══════
              ║
q₁: ──────────X──────
```

### The If-Then Gate

Apply gate $U$ only if classical bit $c = 1$:

$$U^c = \begin{cases} I & \text{if } c = 0 \\ U & \text{if } c = 1 \end{cases}$$

```
c: ════●════
       ║
q: ────U────
```

This is **not** a unitary operation on the full Hilbert space—it's a classically-parameterized operation.

### General Classical Control

Multiple classical bits can control a gate:

```
c₀: ════●════
        ║
c₁: ════●════
        ║
q:  ────U────
```

Apply $U$ only if $c_0 = 1$ AND $c_1 = 1$.

### Measure-and-Correct Paradigm

A fundamental pattern in quantum computing:

```
         ┌───┐   ┌───┐          ┌─────────┐
|ψ⟩ ─────┤ E ├───┤ M ├════●═════┤Correct  ├────
         └───┘   └───┘    ║     │(based   │
                          ║     │on m)    │
                          ╚═════┤         ├════ outcome m
                                └─────────┘
```

1. **Error** E corrupts the state
2. **Measure** syndrome (not the data!)
3. **Correct** based on syndrome

This is the basis of **quantum error correction**.

### Teleportation Circuit

The canonical example of classical control:

```
                   ┌───┐
|ψ⟩ ────────────●──┤ H ├──[M]═══════════●═════════
                │  └───┘                ║
         ┌───┐  │                       ║
|0⟩ ─────┤ H ├──⊕─────────[M]═══════●══╬═════════
         └───┘                      ║  ║
                                  ┌─╨──╨─┐
|0⟩ ────────────────────────────┤ X  Z  ├──── |ψ⟩
                                  └──────┘
```

**Classical control operations:**
- If $m_1 = 1$: Apply $X$ to output
- If $m_0 = 1$: Apply $Z$ to output

$$|ψ'\rangle = Z^{m_0} X^{m_1} |ψ\rangle$$

### Pauli Frame Tracking

Instead of applying corrections in real-time, track them classically:

```
Actual state:    Z^a X^b |ψ⟩
Pauli frame:     (a, b) stored classically
Logical state:   |ψ⟩
```

Apply corrections only when needed (e.g., at final measurement).

### Delayed Choice and Timing

Classical control introduces timing constraints:

```
q₀: ──[M]══════════●═══    Measurement must complete
                   ║        before gate can be applied
q₁: ───────────────X───    (latency requirement)
```

**Real hardware consideration:** Measurement is slow (~μs), gates are fast (~ns).

### Reset Operations

Classical control enables qubit reset:

```
q: ───[M]═══●═══────
            ║
            X    (if m=1, flip to get |0⟩)

Or simply:
q: ───[Reset]───   (measure and conditionally flip)
```

This prepares $|0\rangle$ regardless of input state.

### Repeat-Until-Success

Classical control enables probabilistic protocols:

```
      ┌──────────────────────────────┐
      │  ┌───┐                       │
|0⟩ ──┼──┤ U ├───[M]════●════════════┼──── success if m=0
      │  └───┘          ║            │
      │                 ║            │
      │            if m=1: retry     │
      └──────────────────────────────┘
```

### Classical Register Operations

Classical bits can also be processed:

```
c₀: ════════════●═══════════════
                ║
c₁: ════════════⊕═══════════════  (classical XOR)

c₀, c₁: ═══════[AND]═══════════   (classical AND)
                 │
c₂: ════════════●═══════════════
```

---

## Worked Examples

### Example 1: Teleportation with Classical Control

Walk through teleportation of $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$.

**Solution:**

**Initial state:** $|ψ\rangle_{ABC} = |+\rangle_A \otimes |\Phi^+\rangle_{BC}$

where $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$

**Step 1:** Expand:
$$|ψ\rangle = \frac{1}{2}(|0\rangle + |1\rangle)_A (|00\rangle + |11\rangle)_{BC}$$
$$= \frac{1}{2}(|000\rangle + |011\rangle + |100\rangle + |111\rangle)_{ABC}$$

**Step 2:** CNOT on A,B (A control):
$$= \frac{1}{2}(|000\rangle + |011\rangle + |110\rangle + |101\rangle)$$

**Step 3:** H on A:
$$= \frac{1}{2\sqrt{2}}[(|0\rangle+|1\rangle)|00\rangle + (|0\rangle+|1\rangle)|11\rangle + (|0\rangle-|1\rangle)|10\rangle + (|0\rangle-|1\rangle)|01\rangle]$$

**Step 4:** Regroup by AB measurement outcomes:
$$= \frac{1}{2}[|00\rangle(|0\rangle+|1\rangle) + |01\rangle(|1\rangle+|0\rangle) + |10\rangle(|0\rangle-|1\rangle) + |11\rangle(|1\rangle-|0\rangle)]_C$$

$$= \frac{1}{2}[|00\rangle|+\rangle + |01\rangle X|+\rangle + |10\rangle Z|+\rangle + |11\rangle XZ|+\rangle]$$

**Step 5:** Measure A,B with outcomes $m_A, m_B$:

| $m_A$ | $m_B$ | State of C | Correction |
|-------|-------|------------|------------|
| 0 | 0 | $\|+\rangle$ | None |
| 0 | 1 | $X\|+\rangle = \|+\rangle$ | X (gives $\|+\rangle$) |
| 1 | 0 | $Z\|+\rangle = \|-\rangle$ | Z (gives $\|+\rangle$) |
| 1 | 1 | $XZ\|+\rangle = -\|-\rangle$ | ZX (gives $\|+\rangle$) |

**Step 6:** Apply $Z^{m_A} X^{m_B}$: C becomes $|+\rangle$ in all cases!

### Example 2: Quantum Random Number Generator

Design a circuit that generates a random bit using quantum mechanics.

**Solution:**

```
         ┌───┐
|0⟩ ─────┤ H ├───[M]═══ random bit c
         └───┘
```

**Analysis:**
$$H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

Measurement gives:
- 0 with probability 1/2
- 1 with probability 1/2

This is truly random (quantum indeterminacy), not pseudorandom!

### Example 3: Conditional Phase Kickback

```
         ┌───┐         ┌───┐
|0⟩ ─────┤ H ├───●─────┤ H ├───[M]═══════════●═══
         └───┘   │     └───┘                 ║
                 │                         ┌─╨─┐
|−⟩ ─────────────⊕─────────────────────────┤ Z ├───
                                           └───┘
```

**Goal:** Demonstrate phase kickback and classical correction.

**Analysis:**

After H on q₀: $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |-\rangle$

After CNOT: Phase kickback gives $\frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) \otimes |-\rangle$

After H on q₀: $|1\rangle \otimes |-\rangle$

Measurement gives $m = 1$ with certainty.

Classical control: $Z^1 |-\rangle = -|-\rangle = |-\rangle$ (global phase).

---

## Practice Problems

### Problem 1: Reset Circuit
Design a circuit that resets any single-qubit state $|\psi\rangle$ to $|0\rangle$ using measurement and classical control.

### Problem 2: Conditional Bell State
Create a circuit that prepares $|\Phi^+\rangle$ if $c=0$ and $|\Psi^+\rangle$ if $c=1$, using one classical control bit.

### Problem 3: State Preparation
Using classical control, design a circuit that prepares $|1\rangle$ with certainty, starting from an unknown state.

### Problem 4: Multiple Controls
What does this circuit do?
```
         ┌───┐
q₀: ─────┤ H ├───[M]═══●═══════════
         └───┘         ║
         ┌───┐         ║
q₁: ─────┤ H ├───[M]═══╬═══●═══════
         └───┘         ║   ║
                     ┌─╨───╨─┐
q₂: ─────────────────┤ X   Z ├─────
                     └───────┘
```

---

## Computational Lab

```python
"""Day 578: Classical Control in Quantum Circuits"""
import numpy as np
from numpy.random import choice

# Standard gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

def tensor(*args):
    """Tensor product"""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result

def measure(state, qubit, n_qubits):
    """Measure qubit, return (outcome, collapsed_state)"""
    dim = 2**n_qubits

    # Build projectors
    P0 = np.zeros((2, 2), dtype=complex)
    P1 = np.zeros((2, 2), dtype=complex)
    P0[0, 0] = 1
    P1[1, 1] = 1

    ops0 = [I] * n_qubits
    ops0[qubit] = P0
    Proj0 = tensor(*ops0)

    ops1 = [I] * n_qubits
    ops1[qubit] = P1
    Proj1 = tensor(*ops1)

    # Probabilities
    p0 = np.real(np.vdot(state, Proj0 @ state))
    p1 = np.real(np.vdot(state, Proj1 @ state))

    # Sample
    outcome = choice([0, 1], p=[p0, p1])

    if outcome == 0:
        new_state = Proj0 @ state / np.sqrt(p0)
    else:
        new_state = Proj1 @ state / np.sqrt(p1)

    return outcome, new_state

def conditional_gate(state, classical_bit, gate, qubit, n_qubits):
    """Apply gate to qubit if classical_bit == 1"""
    if classical_bit == 1:
        ops = [I] * n_qubits
        ops[qubit] = gate
        U = tensor(*ops)
        return U @ state
    else:
        return state

def state_str(state, n_qubits):
    """Pretty print"""
    terms = []
    for i, amp in enumerate(state):
        if np.abs(amp) > 1e-10:
            bits = format(i, f'0{n_qubits}b')
            terms.append(f"({amp:.3f})|{bits}>")
    return " + ".join(terms) if terms else "0"

# ===== Example 1: Quantum Teleportation =====
print("=" * 60)
print("Example 1: Quantum Teleportation")
print("=" * 60)

def teleport(input_state):
    """Teleport a single-qubit state using classical control"""

    # Initial state: |psi>_A |00>_BC
    # We'll prepare Bell pair on BC first
    state = tensor(input_state, np.array([1, 0]), np.array([1, 0]))

    # Create Bell pair on BC
    state = tensor(np.eye(2), H, I) @ state  # H on B
    # CNOT on BC
    CNOT_BC = tensor(I, CNOT)
    state = CNOT_BC @ state

    print(f"After Bell pair creation: {state_str(state, 3)}")

    # CNOT on AB (A control)
    CNOT_AB = np.zeros((8, 8), dtype=complex)
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        b_new = b ^ a
        j = (a << 2) | (b_new << 1) | c
        CNOT_AB[j, i] = 1

    state = CNOT_AB @ state

    # H on A
    state = tensor(H, I, I) @ state

    print(f"Before measurement: {state_str(state, 3)}")

    # Measure A and B
    m_A, state = measure(state, 0, 3)
    m_B, state = measure(state, 1, 3)

    print(f"Measured: m_A={m_A}, m_B={m_B}")
    print(f"After measurement: {state_str(state, 3)}")

    # Apply corrections to C
    state = conditional_gate(state, m_B, X, 2, 3)
    state = conditional_gate(state, m_A, Z, 2, 3)

    # Extract qubit C
    output = np.zeros(2, dtype=complex)
    for i in range(8):
        c = i & 1
        output[c] += state[i]

    # Normalize
    output = output / np.linalg.norm(output)

    return output, m_A, m_B

# Test with |+>
print("\nTeleporting |+> state:")
input_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
print(f"Input: {state_str(input_plus, 1)}")

# Run multiple times
for trial in range(3):
    output, mA, mB = teleport(input_plus.copy())
    print(f"Trial {trial+1}: Output = {state_str(output, 1)}, m_A={mA}, m_B={mB}")

# ===== Example 2: Reset Circuit =====
print("\n" + "=" * 60)
print("Example 2: Reset Circuit")
print("=" * 60)

def reset_qubit(state):
    """Reset qubit to |0> using measure + conditional X"""
    outcome, collapsed = measure(state, 0, 1)
    if outcome == 1:
        collapsed = X @ collapsed
    return collapsed

print("Resetting various states to |0>:")
test_states = [
    (np.array([1, 0], dtype=complex), "|0>"),
    (np.array([0, 1], dtype=complex), "|1>"),
    (np.array([1, 1], dtype=complex)/np.sqrt(2), "|+>"),
    (np.array([1, -1], dtype=complex)/np.sqrt(2), "|->"),
]

for state, name in test_states:
    result = reset_qubit(state.copy())
    print(f"  {name} -> {state_str(result, 1)}")

# ===== Example 3: Conditional Bell State Preparation =====
print("\n" + "=" * 60)
print("Example 3: Conditional Bell State Preparation")
print("=" * 60)

def conditional_bell_prep(classical_bit):
    """
    Prepare |Phi+> if c=0, |Psi+> if c=1
    |Phi+> = (|00> + |11>)/sqrt(2)
    |Psi+> = (|01> + |10>)/sqrt(2)
    """
    # Start with |00>
    state = np.array([1, 0, 0, 0], dtype=complex)

    # H on q0
    state = tensor(H, I) @ state

    # CNOT
    state = CNOT @ state

    # Now we have |Phi+>

    # If c=1, apply X to q1 to get |Psi+>
    if classical_bit == 1:
        state = tensor(I, X) @ state

    return state

print("c=0 (should be |Phi+>):")
bell0 = conditional_bell_prep(0)
print(f"  {state_str(bell0, 2)}")

print("c=1 (should be |Psi+>):")
bell1 = conditional_bell_prep(1)
print(f"  {state_str(bell1, 2)}")

# ===== Example 4: Repeat-Until-Success =====
print("\n" + "=" * 60)
print("Example 4: Repeat-Until-Success Protocol")
print("=" * 60)

def rus_circuit():
    """
    Repeat-until-success: try to prepare a special state.
    Apply H, measure. If |0>, success. If |1>, retry.
    """
    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        attempts += 1
        state = np.array([1, 0], dtype=complex)
        state = H @ state  # Now |+>

        outcome, _ = measure(state, 0, 1)

        if outcome == 0:
            return attempts, "success"

    return attempts, "max attempts reached"

print("Running repeat-until-success 10 times:")
total_attempts = 0
for i in range(10):
    attempts, status = rus_circuit()
    total_attempts += attempts
    print(f"  Run {i+1}: {status} after {attempts} attempt(s)")

print(f"Average attempts: {total_attempts/10:.2f} (expected: 2)")

# ===== Example 5: Pauli Frame Tracking =====
print("\n" + "=" * 60)
print("Example 5: Pauli Frame Tracking")
print("=" * 60)

class PauliFrame:
    """Track Pauli corrections classically"""
    def __init__(self):
        self.x_correction = False  # Track if X needed
        self.z_correction = False  # Track if Z needed

    def update_from_measurement(self, m_x, m_z):
        """Update frame based on measurements"""
        self.x_correction ^= m_x  # XOR
        self.z_correction ^= m_z

    def apply_gate(self, gate_name):
        """Propagate Pauli frame through a gate"""
        if gate_name == 'H':
            # H X H = Z, H Z H = X
            self.x_correction, self.z_correction = self.z_correction, self.x_correction
        elif gate_name == 'S':
            # S X S^dag = Y = iXZ, S Z S^dag = Z
            # For Pauli frame: S X -> XZ, S Z -> Z
            if self.x_correction:
                self.z_correction ^= True
        # Add more gates as needed

    def get_correction(self):
        """Get the correction to apply at the end"""
        return (self.z_correction, self.x_correction)  # Z^z X^x

# Demo
frame = PauliFrame()
print("Initial frame: X={}, Z={}".format(frame.x_correction, frame.z_correction))

# Simulate teleportation measurements
m_A, m_B = 1, 1
frame.update_from_measurement(m_B, m_A)  # X from m_B, Z from m_A
print(f"After measurements m_A={m_A}, m_B={m_B}: X={frame.x_correction}, Z={frame.z_correction}")

# Propagate through H gate
frame.apply_gate('H')
print(f"After propagating through H: X={frame.x_correction}, Z={frame.z_correction}")

z_corr, x_corr = frame.get_correction()
print(f"Final correction needed: Z^{int(z_corr)} X^{int(x_corr)}")

print("\n" + "=" * 60)
print("Summary: Classical Control Patterns")
print("=" * 60)
print("""
1. If-Then Gate: Apply U only if classical bit c=1
2. Feedforward: Use measurement result to control future gates
3. Reset: Measure and conditionally flip to prepare |0>
4. Teleportation: Classic example of measure-and-correct
5. Pauli Frame: Track corrections classically, apply only when needed
6. Repeat-Until-Success: Retry until desired outcome
""")
```

---

## Summary

### Classical Control Notation

| Symbol | Meaning |
|--------|---------|
| `═══` | Classical wire |
| `═══●═══` | Classical control |
| `[M]` | Measurement producing classical bit |
| `X^c` | Conditional X gate |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Conditional gate | $U^c = (1-c)I + cU$ |
| Teleportation correction | $Z^{m_A} X^{m_B}$ |
| Reset operation | Measure, then $X^m$ |
| Pauli frame | Track $(z, x)$ for $Z^z X^x$ |

### Key Takeaways

1. **Classical control** uses measurement outcomes to decide gate application
2. **Feedforward** allows adaptive quantum algorithms
3. **Measure-and-correct** is fundamental to error correction
4. **Teleportation** demonstrates the power of classical control
5. **Pauli frame tracking** defers corrections for efficiency
6. **Classical control ≠ quantum control**: no entanglement created

---

## Daily Checklist

- [ ] I can distinguish classical control from quantum control
- [ ] I can implement feedforward operations in circuits
- [ ] I understand the measure-and-correct paradigm
- [ ] I can trace through teleportation with classical corrections
- [ ] I understand Pauli frame tracking
- [ ] I ran the computational lab and implemented classical control

---

*Next: Day 579 — Circuit Complexity*
