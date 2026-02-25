# Day 842: T-Gate Fundamentals

## Week 121, Day 2 | Month 31: Fault-Tolerant QC I | Semester 2B: Fault Tolerance & Hardware

### Overview

Today we introduce the T-gate (also called the $\pi/8$ gate), the simplest non-Clifford gate that, when combined with Clifford operations, enables universal quantum computation. We explore its mathematical properties, prove it lies outside the Clifford group, and understand why this single gate is both essential for universality and challenging for fault-tolerant implementation.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | T-gate definition and properties |
| **Afternoon** | 2.5 hours | Non-Clifford proof and implications |
| **Evening** | 1.5 hours | Computational lab: T-gate analysis |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define the T-gate** and write its matrix representation
2. **Derive the relationship** between T, S, and Z gates ($T^2 = S$, $T^4 = Z$)
3. **Prove that T is non-Clifford** using conjugation arguments
4. **Explain why T enables universality** when combined with Cliffords
5. **Understand the fault-tolerance challenge** posed by non-Clifford gates
6. **Calculate T-gate action** on various quantum states

---

## Part 1: T-Gate Definition

### Matrix Form

The T-gate (also called $T$, $\pi/8$ gate, or $\sqrt{S}$ gate) is defined as:

$$\boxed{T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}}$$

### Alternative Representations

**As a Z-rotation:**
$$T = e^{-i\pi/8} R_z(\pi/4) = e^{-i\pi/8} \begin{pmatrix} e^{-i\pi/8} & 0 \\ 0 & e^{i\pi/8} \end{pmatrix}$$

**Explicit form:**
$$T = \begin{pmatrix} 1 & 0 \\ 0 & \frac{1+i}{\sqrt{2}} \end{pmatrix}$$

since $e^{i\pi/4} = \cos(\pi/4) + i\sin(\pi/4) = \frac{1}{\sqrt{2}} + \frac{i}{\sqrt{2}} = \frac{1+i}{\sqrt{2}}$.

### Why "Pi/8 Gate"?

The name comes from viewing T as a rotation by $\pi/4$ about the Z-axis:

$$R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

For $\theta = \pi/4$:
$$R_z(\pi/4) = \begin{pmatrix} e^{-i\pi/8} & 0 \\ 0 & e^{i\pi/8} \end{pmatrix}$$

The $T$ gate is $R_z(\pi/4)$ up to a global phase, and $\pi/8$ appears in the matrix elements.

---

## Part 2: Key Properties of the T-Gate

### Relationship to Other Gates

The T-gate generates a sequence of important gates through powers:

$$\boxed{T^2 = S, \quad T^4 = Z, \quad T^8 = I}$$

**Proof of $T^2 = S$:**
$$T^2 = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}^2 = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/2} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = S \checkmark$$

**Proof of $T^4 = Z$:**
$$T^4 = S^2 = \begin{pmatrix} 1 & 0 \\ 0 & i^2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = Z \checkmark$$

**Proof of $T^8 = I$:**
$$T^8 = Z^2 = I \checkmark$$

### Gate Hierarchy

```
T-Gate Powers:
    T^1 = T         (non-Clifford)
    T^2 = S         (Clifford)
    T^3 = TS        (non-Clifford)
    T^4 = Z         (Clifford, Pauli)
    T^5 = TZ        (non-Clifford)
    T^6 = S^3 = S†  (Clifford)
    T^7 = T†        (non-Clifford)
    T^8 = I         (identity)
```

### Hermitian Conjugate

$$T^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix} = T^7$$

Also written as $T^\dagger = T^{-1}$ since $TT^\dagger = I$.

### Unitarity

$$T^\dagger T = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = I \checkmark$$

---

## Part 3: Proof That T Is Non-Clifford

### The Clifford Criterion

A gate $U$ is Clifford if and only if $UPU^\dagger \in \mathcal{P}_1$ for all Paulis $P$.

### Testing T on Paulis

**Case 1: $P = I$**
$$TIT^\dagger = I \in \mathcal{P}_1 \checkmark$$

**Case 2: $P = Z$**
$$TZT^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$$
$$= \begin{pmatrix} 1 & 0 \\ 0 & -e^{i\pi/4} \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = Z \in \mathcal{P}_1 \checkmark$$

**Case 3: $P = X$ (THE CRITICAL CASE)**
$$TXT^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$$

$$= \begin{pmatrix} 0 & 1 \\ e^{i\pi/4} & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix} = \begin{pmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{pmatrix}$$

This is NOT a Pauli matrix!

**Verification:** The Pauli matrices are:
- $I = \begin{psmallmatrix} 1 & 0 \\ 0 & 1 \end{psmallmatrix}$, $X = \begin{psmallmatrix} 0 & 1 \\ 1 & 0 \end{psmallmatrix}$, $Y = \begin{psmallmatrix} 0 & -i \\ i & 0 \end{psmallmatrix}$, $Z = \begin{psmallmatrix} 1 & 0 \\ 0 & -1 \end{psmallmatrix}$

The matrix $\begin{psmallmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{psmallmatrix}$ does not equal any Pauli times a phase.

$$\boxed{TXT^\dagger = e^{-i\pi/4}\frac{X + Y}{\sqrt{2}} \notin \mathcal{P}_1}$$

### Explicit Calculation

Let's verify this formula:
$$e^{-i\pi/4}\frac{X + Y}{\sqrt{2}} = e^{-i\pi/4} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & 1-i \\ 1+i & 0 \end{pmatrix}$$

$$= \frac{e^{-i\pi/4}}{\sqrt{2}}\begin{pmatrix} 0 & 1-i \\ 1+i & 0 \end{pmatrix}$$

Since $e^{-i\pi/4} = \frac{1-i}{\sqrt{2}}$:

$$= \frac{1}{2}\begin{pmatrix} 0 & (1-i)^2 \\ (1-i)(1+i) & 0 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 0 & -2i \\ 2 & 0 \end{pmatrix}$$

Wait, let me recalculate more carefully:

$$TXT^\dagger = \begin{pmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{pmatrix}$$

This can be written as:
$$= \cos(\pi/4) X + \sin(\pi/4) Y$$

since:
$$\cos(\pi/4) X + \sin(\pi/4) Y = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} + \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$
$$= \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & 1-i \\ 1+i & 0 \end{pmatrix}$$

And $e^{\pm i\pi/4} = \frac{1 \pm i}{\sqrt{2}}$, so:
$$\begin{pmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{pmatrix} = \begin{pmatrix} 0 & \frac{1-i}{\sqrt{2}} \\ \frac{1+i}{\sqrt{2}} & 0 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & 1-i \\ 1+i & 0 \end{pmatrix} \checkmark$$

### Conclusion

$$\boxed{T \notin \mathcal{C}_1 \text{ because } TXT^\dagger \notin \mathcal{P}_1}$$

The T-gate is **non-Clifford**.

---

## Part 4: T-Gate in the Clifford Hierarchy

### Position in the Hierarchy

Recall:
- $\mathcal{C}_1 = \mathcal{P}_n$ (Pauli group)
- $\mathcal{C}_2 = $ Clifford group (normalizer of $\mathcal{P}_n$)
- $\mathcal{C}_k = \{U : U\mathcal{P}_n U^\dagger \subseteq \mathcal{C}_{k-1}\}$

**Claim:** $T \in \mathcal{C}_3$

**Proof:** We need $TXT^\dagger \in \mathcal{C}_2$.

We showed $TXT^\dagger = \frac{X+Y}{\sqrt{2}}$ (up to phase). Is this Clifford?

Check: $\frac{X+Y}{\sqrt{2}}$ is a $\pi/2$ rotation about the axis $(1,1,0)$ on the Bloch sphere, which is a Clifford operation (rotations by multiples of $\pi/2$ about Pauli axes are Clifford).

More directly: $\frac{X+Y}{\sqrt{2}} = HSH^\dagger$ (can verify), which is clearly Clifford.

Therefore, $T \in \mathcal{C}_3$.

### Level 3 Enables Universality

**Key Theorem:** $\mathcal{C}_2 \cup \mathcal{C}_3$ generates a dense subgroup of $U(2^n)$.

In other words, Clifford + T is universal!

---

## Part 5: T-Gate Action on States

### Action on Computational Basis

$$T|0\rangle = |0\rangle$$
$$T|1\rangle = e^{i\pi/4}|1\rangle$$

### Action on Superposition States

$$T|+\rangle = T\frac{|0\rangle + |1\rangle}{\sqrt{2}} = \frac{|0\rangle + e^{i\pi/4}|1\rangle}{\sqrt{2}}$$

This is the **magic state** $|T\rangle$ (tomorrow's topic)!

$$T|-\rangle = \frac{|0\rangle - e^{i\pi/4}|1\rangle}{\sqrt{2}}$$

$$T|+i\rangle = T\frac{|0\rangle + i|1\rangle}{\sqrt{2}} = \frac{|0\rangle + ie^{i\pi/4}|1\rangle}{\sqrt{2}} = \frac{|0\rangle + e^{i3\pi/4}|1\rangle}{\sqrt{2}}$$

### Bloch Sphere Interpretation

On the Bloch sphere:
- $|0\rangle$ is at the north pole (0,0,1)
- $|1\rangle$ is at the south pole (0,0,-1)
- $|+\rangle$ is at (1,0,0)
- T rotates by $\pi/4$ about the Z-axis

Starting from $|+\rangle$ at $(1,0,0)$:
$$T|+\rangle \text{ moves to } (\cos(\pi/4), \sin(\pi/4), 0) = (1/\sqrt{2}, 1/\sqrt{2}, 0)$$

---

## Part 6: Why T Is Challenging for Fault Tolerance

### The Transversality Problem

**Transversal gates** apply the same single-qubit operation to each physical qubit:
$$\bar{U} = U^{\otimes n}$$

Transversal gates are naturally fault-tolerant because errors don't spread between qubits.

**The Problem:** For CSS codes (like the surface code), transversal T-gates don't preserve the code space!

**Eastin-Knill Theorem Preview:** No quantum error-correcting code can have a universal transversal gate set.

### Consequences

1. **Clifford gates CAN be transversal** on many codes
   - Easy fault-tolerant implementation
   - Low overhead

2. **T-gate CANNOT be transversal** on CSS codes
   - Requires alternative approaches
   - Magic state injection (this week's main topic)
   - Significant resource overhead

### The Magic State Solution

Instead of implementing T directly on encoded qubits:

1. **Prepare** a "magic state" $|T\rangle = T|+\rangle$ (non-fault-tolerant)
2. **Distill** to high fidelity using Clifford operations
3. **Inject** via gate teleportation (uses only Cliffords + measurement)
4. **Apply corrections** based on measurement outcomes

This converts the non-transversal T-gate into fault-tolerant operations!

---

## Part 7: Quantum Computing Connection

### T-Count as Resource Metric

In fault-tolerant quantum computing, the **T-count** (number of T-gates) is the dominant cost metric.

**Why T-gates are expensive:**
- Each T-gate requires a distilled magic state
- Distillation has significant overhead (15-to-1 protocol, etc.)
- T-gates dominate runtime and qubit count

### T-Counts for Common Operations

| Operation | Approximate T-Count |
|-----------|---------------------|
| Toffoli (CCX) | 7 |
| Controlled-S | 2 |
| Arbitrary single-qubit | O(log(1/ε)) |
| n-bit addition | O(n) |
| n-bit multiplication | O(n²) |
| Quantum Fourier Transform (n-bit) | O(n²) or O(n log n) |
| Shor's algorithm (n-bit factoring) | O(n³) |

### Optimization Goal

Modern quantum algorithm research focuses heavily on **T-count reduction**:
- Clever circuit decompositions
- Approximate synthesis (Solovay-Kitaev, gridsynth)
- Algorithm-level optimizations

---

## Worked Examples

### Example 1: Verify $T^4 = Z$

**Problem:** Prove that $T^4 = Z$ by direct matrix multiplication.

**Solution:**

$$T^2 = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}^2 = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/2} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = S$$

$$T^4 = (T^2)^2 = S^2 = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}^2 = \begin{pmatrix} 1 & 0 \\ 0 & i^2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = Z$$

**Verified:** $\boxed{T^4 = Z}$

---

### Example 2: Calculate $HTH$

**Problem:** Compute the gate $HTH$ and express it in a simple form.

**Solution:**

$$HTH = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

First, compute $TH$:
$$TH = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ e^{i\pi/4} & -e^{i\pi/4} \end{pmatrix}$$

Then $HTH$:
$$HTH = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ e^{i\pi/4} & -e^{i\pi/4} \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 1 + e^{i\pi/4} & 1 - e^{i\pi/4} \\ 1 - e^{i\pi/4} & 1 + e^{i\pi/4} \end{pmatrix}$$

This is a rotation about the X-axis by $\pi/4$:

$$\boxed{HTH = R_x(\pi/4) = e^{-i\pi X/8}}$$

up to global phase.

---

### Example 3: Express Toffoli in Terms of T-Gates

**Problem:** The Toffoli gate (CCX) can be decomposed into Clifford + T gates. How many T-gates are needed?

**Solution:**

The standard decomposition of Toffoli uses:
- 6 CNOT gates
- 2 Hadamard gates
- 7 T-gates (or T^dag gates)

The minimal T-count for Toffoli is:

$$\boxed{\text{T-count(Toffoli)} = 7}$$

The circuit is:
```
a: ─────────────────●───────●───T───●───T†──●───────────
                    │       │       │       │
b: ───●───T†───●────┼───T───┼───────┼───T†──┼───●───T───
      │        │    │       │       │       │   │
c: ─H─X───T────X────X───T†──X───────X───T───X───X───T†──H─
```

(Note: Exact gate ordering may vary by decomposition)

---

## Practice Problems

### Problem Set A: Direct Application

**A1.** Compute $T^3$ and show it equals $TS$.

**A2.** Calculate $T|+i\rangle$ where $|+i\rangle = (|0\rangle + i|1\rangle)/\sqrt{2}$.

**A3.** Show that $T^\dagger XT = e^{i\pi/4}(X - Y)/\sqrt{2}$ (up to phase corrections).

### Problem Set B: Intermediate

**B1.** Prove that $TYT^\dagger$ is not a Pauli (verify T is non-Clifford using Y).

**B2.** Calculate the gate sequence $THTH$ and identify it.

**B3.** Show that $T$ and $T^\dagger$ together with Cliffords generate the same group as $T$ and Cliffords.

### Problem Set C: Challenging

**C1.** Prove that any single-qubit gate can be approximated to accuracy $\epsilon$ using $O(\log(1/\epsilon))$ gates from $\{H, T\}$.

**C2.** Show that the Controlled-T gate is in $\mathcal{C}_4$ (fourth level of Clifford hierarchy).

**C3.** **(Research-level)** The T-count of Toffoli is 7, but the T-depth (number of T-layers) can be reduced to 4. Construct such a circuit.

---

## Computational Lab

```python
"""
Day 842 Computational Lab: T-Gate Analysis and Properties
Explores the T-gate, its non-Clifford nature, and universality implications
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Part 1: Gate Definitions
# =============================================================================

# Basic gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
T_dag = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

def print_matrix(M: np.ndarray, name: str = "M", precision: int = 4):
    """Pretty print a matrix."""
    print(f"\n{name} =")
    for row in M:
        row_str = "  ["
        for val in row:
            if np.isclose(val.imag, 0):
                row_str += f" {val.real:>{precision+3}.{precision}f} "
            else:
                row_str += f" {val.real:+.2f}{val.imag:+.2f}j "
        row_str += "]"
        print(row_str)

# =============================================================================
# Part 2: Verify T-Gate Powers
# =============================================================================

print("=" * 60)
print("T-GATE POWERS VERIFICATION")
print("=" * 60)

# Compute powers of T
T_powers = {
    'T^1': T,
    'T^2': T @ T,
    'T^3': T @ T @ T,
    'T^4': T @ T @ T @ T,
    'T^5': np.linalg.matrix_power(T, 5),
    'T^6': np.linalg.matrix_power(T, 6),
    'T^7': np.linalg.matrix_power(T, 7),
    'T^8': np.linalg.matrix_power(T, 8),
}

# Compare with known gates
comparisons = [
    ('T^2', S, 'S'),
    ('T^4', Z, 'Z'),
    ('T^6', np.linalg.matrix_power(S, 3), 'S^3 = S†'),
    ('T^7', T_dag, 'T†'),
    ('T^8', I, 'I'),
]

print("\nPower relationships:")
for power_name, expected, expected_name in comparisons:
    actual = T_powers[power_name]
    is_equal = np.allclose(actual, expected) or np.allclose(actual, -expected)
    status = "✓" if is_equal else "✗"
    print(f"  {power_name} = {expected_name}: {status}")

# =============================================================================
# Part 3: Prove T is Non-Clifford
# =============================================================================

print("\n" + "=" * 60)
print("PROVING T IS NON-CLIFFORD")
print("=" * 60)

def is_pauli_matrix(M: np.ndarray) -> Tuple[bool, str]:
    """Check if M is a Pauli matrix (up to phase)."""
    paulis = [('I', I), ('X', X), ('Y', Y), ('Z', Z)]

    for name, P in paulis:
        # Check if M = phase * P
        for phase in [1, -1, 1j, -1j]:
            if np.allclose(M, phase * P):
                return True, f"{phase}*{name}"

    return False, "NOT PAULI"

# Test T conjugation on all Paulis
print("\nT-gate conjugation on Paulis:")
for name, P in [('I', I), ('X', X), ('Y', Y), ('Z', Z)]:
    conjugated = T @ P @ T.conj().T
    is_pauli, result = is_pauli_matrix(conjugated)
    print(f"  T {name} T† = {result}")

    if not is_pauli:
        print(f"    → THIS PROVES T IS NOT CLIFFORD!")
        print_matrix(conjugated, f"    T{name}T†")

# Analyze TXT† more carefully
print("\nDetailed analysis of TXT†:")
TXT = T @ X @ T.conj().T
print(f"  TXT† = ")
print(f"    [0, e^{{-iπ/4}}]   [0, {np.exp(-1j*np.pi/4):.4f}]")
print(f"    [e^{{iπ/4}}, 0 ] = [{np.exp(1j*np.pi/4):.4f}, 0]")

print(f"\n  This equals (X + Y)/√2 up to a phase:")
XY_combo = (X + Y) / np.sqrt(2)
print_matrix(XY_combo, "  (X+Y)/√2")

# =============================================================================
# Part 4: T-Gate Action on States
# =============================================================================

print("\n" + "=" * 60)
print("T-GATE ACTION ON STATES")
print("=" * 60)

# Define standard states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)
ket_plus_i = (ket_0 + 1j * ket_1) / np.sqrt(2)
ket_minus_i = (ket_0 - 1j * ket_1) / np.sqrt(2)

states = [
    ('|0⟩', ket_0),
    ('|1⟩', ket_1),
    ('|+⟩', ket_plus),
    ('|-⟩', ket_minus),
    ('|+i⟩', ket_plus_i),
    ('|-i⟩', ket_minus_i),
]

print("\nT-gate action:")
for name, state in states:
    T_state = T @ state
    print(f"  T{name} = [{T_state[0]:.4f}, {T_state[1]:.4f}]")

# Magic state
magic_T = T @ ket_plus
print(f"\n  Magic state |T⟩ = T|+⟩ = {magic_T}")
print(f"  |T⟩ = (|0⟩ + e^{{iπ/4}}|1⟩)/√2")

# =============================================================================
# Part 5: Bloch Sphere Visualization
# =============================================================================

def state_to_bloch(state: np.ndarray) -> Tuple[float, float, float]:
    """Convert quantum state to Bloch sphere coordinates."""
    # Ensure normalization
    state = state / np.linalg.norm(state)

    # Bloch coordinates
    x = 2 * np.real(state[0].conj() * state[1])
    y = 2 * np.imag(state[0].conj() * state[1])
    z = np.abs(state[0])**2 - np.abs(state[1])**2

    return x, y, z

print("\n" + "=" * 60)
print("BLOCH SPHERE COORDINATES")
print("=" * 60)

print("\nState transformations under T:")
for name, state in states:
    x0, y0, z0 = state_to_bloch(state)
    T_state = T @ state
    x1, y1, z1 = state_to_bloch(T_state)
    print(f"  {name}: ({x0:.3f}, {y0:.3f}, {z0:.3f}) → ({x1:.3f}, {y1:.3f}, {z1:.3f})")

# =============================================================================
# Part 6: T-Count Analysis for Common Gates
# =============================================================================

print("\n" + "=" * 60)
print("T-COUNT FOR COMMON OPERATIONS")
print("=" * 60)

t_counts = {
    'Single T-gate': 1,
    'S gate (= T²)': 0,
    'Z gate (= T⁴)': 0,
    'Controlled-Z': 0,
    'Controlled-S': 2,
    'Controlled-T': 4,
    'Toffoli (CCX)': 7,
    'Fredkin (CSWAP)': 7,
    'n-bit adder': 'O(n)',
    'n-bit multiplier': 'O(n²)',
    'QFT (n-bit)': 'O(n log n)',
    'Shor factoring': 'O(n³)',
}

print("\nT-counts for common operations:")
for op, count in t_counts.items():
    print(f"  {op}: {count}")

# =============================================================================
# Part 7: Visualization
# =============================================================================

fig = plt.figure(figsize=(16, 10))

# Plot 1: T-gate powers on unit circle (complex plane)
ax1 = fig.add_subplot(2, 3, 1)
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

# Plot T^k phases
for k in range(8):
    phase = np.exp(1j * k * np.pi / 4)
    ax1.scatter(phase.real, phase.imag, s=150, zorder=5,
                label=f'T^{k} (k={k})')

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.set_title('T-Gate Powers: Phases on Unit Circle', fontsize=11)
ax1.set_xlabel('Real')
ax1.set_ylabel('Imaginary')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Bloch sphere with T action
ax2 = fig.add_subplot(2, 3, 2, projection='3d')

# Draw sphere wireframe
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))
ax2.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# Plot stabilizer states and their T-transforms
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
for (name, state), color in zip(states, colors):
    x0, y0, z0 = state_to_bloch(state)
    T_state = T @ state
    x1, y1, z1 = state_to_bloch(T_state)

    ax2.scatter([x0], [y0], [z0], s=100, c=color, marker='o', alpha=0.5)
    ax2.scatter([x1], [y1], [z1], s=100, c=color, marker='^')
    ax2.plot([x0, x1], [y0, y1], [z0, z1], color=color, linestyle='--', alpha=0.5)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('T-Gate Action on Bloch Sphere\n(circles→triangles)', fontsize=11)

# Plot 3: Conjugation results
ax3 = fig.add_subplot(2, 3, 3)

# Show TXT† is NOT Pauli
paulis_names = ['I', 'X', 'Y', 'Z']
paulis_list = [I, X, Y, Z]

results = []
for P in paulis_list:
    conj = T @ P @ T.conj().T
    is_pauli, _ = is_pauli_matrix(conj)
    results.append(1 if is_pauli else 0)

ax3.bar(paulis_names, results, color=['green' if r else 'red' for r in results])
ax3.set_ylim(0, 1.5)
ax3.set_ylabel('Is Pauli? (1=Yes, 0=No)')
ax3.set_title('T-Gate Conjugation Results\n(TPT† ∈ Paulis?)', fontsize=11)

for i, (name, r) in enumerate(zip(paulis_names, results)):
    ax3.text(i, r + 0.1, '✓' if r else '✗', ha='center', fontsize=16)

# Plot 4: Gate hierarchy
ax4 = fig.add_subplot(2, 3, 4)

gates_hierarchy = ['I', 'Z', 'S', 'T', 'T³', 'HTH']
in_clifford = [1, 1, 1, 0, 0, 0]  # 1 = Clifford, 0 = Non-Clifford

colors = ['blue' if c else 'red' for c in in_clifford]
ax4.barh(gates_hierarchy, [1]*len(gates_hierarchy), color=colors)
ax4.set_xlim(0, 1.5)
ax4.set_xlabel('')
ax4.set_title('Gate Classification\n(Blue=Clifford, Red=Non-Clifford)', fontsize=11)

# Add annotations
for i, (gate, cliff) in enumerate(zip(gates_hierarchy, in_clifford)):
    label = 'Clifford' if cliff else 'Non-Clifford'
    ax4.text(1.1, i, label, va='center', fontsize=9)

# Plot 5: T-count comparison
ax5 = fig.add_subplot(2, 3, 5)

operations = ['Toffoli', 'C-T', 'C-S', 'CNOT', 'H', 'T']
counts = [7, 4, 2, 0, 0, 1]

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(operations)))
bars = ax5.bar(operations, counts, color=colors, edgecolor='black')
ax5.set_ylabel('T-Count')
ax5.set_title('T-Count for Common Gates', fontsize=11)

for bar, count in zip(bars, counts):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(count), ha='center', fontsize=10)

# Plot 6: Magic state on equator
ax6 = fig.add_subplot(2, 3, 6)

# Draw equator circle
theta = np.linspace(0, 2*np.pi, 100)
ax6.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)

# Stabilizer states on equator
stab_equator = [
    ('|+⟩', 0),
    ('|-⟩', np.pi),
    ('|+i⟩', np.pi/2),
    ('|-i⟩', 3*np.pi/2),
]

for name, angle in stab_equator:
    ax6.scatter(np.cos(angle), np.sin(angle), s=150, c='blue',
                edgecolors='black', zorder=5)
    ax6.annotate(name, (np.cos(angle), np.sin(angle)),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

# Magic state |T⟩ at π/4
magic_angle = np.pi/4
ax6.scatter(np.cos(magic_angle), np.sin(magic_angle), s=200, c='red',
            edgecolors='black', marker='*', zorder=5)
ax6.annotate('|T⟩', (np.cos(magic_angle), np.sin(magic_angle)),
            xytext=(10, 5), textcoords='offset points', fontsize=11, color='red')

# Show T rotation from |+⟩
ax6.annotate('', xy=(np.cos(magic_angle), np.sin(magic_angle)),
            xytext=(np.cos(0), np.sin(0)),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax6.text(0.3, 0.5, 'T', fontsize=14, color='green')

ax6.set_xlim(-1.5, 1.5)
ax6.set_ylim(-1.5, 1.5)
ax6.set_aspect('equal')
ax6.set_title('Equator of Bloch Sphere\nT rotates |+⟩ to |T⟩ (magic state)', fontsize=11)
ax6.set_xlabel('X')
ax6.set_ylabel('Y')

plt.tight_layout()
plt.savefig('day_842_t_gate_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Visualization saved to: day_842_t_gate_analysis.png")
print("=" * 60)

# =============================================================================
# Part 8: Summary
# =============================================================================

print("\n" + "=" * 60)
print("KEY RESULTS SUMMARY")
print("=" * 60)

summary = """
T-GATE DEFINITION:
  T = diag(1, e^{iπ/4})
  T² = S, T⁴ = Z, T⁸ = I

NON-CLIFFORD PROOF:
  TXT† = (X + Y)/√2 ∉ Pauli group
  Therefore T ∉ Clifford group

UNIVERSALITY:
  {H, S, CNOT, T} = Universal gate set
  Clifford + T generates dense subset of U(2^n)

FAULT-TOLERANCE CHALLENGE:
  T-gate cannot be transversal on CSS codes
  Solution: Magic state injection
  |T⟩ = T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2

T-COUNT IMPORTANCE:
  Dominant cost metric in fault-tolerant QC
  Toffoli: 7 T-gates
  Algorithms measured by T-count
"""
print(summary)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| T-gate definition | $T = \text{diag}(1, e^{i\pi/4})$ |
| T-S-Z relationship | $T^2 = S, \quad T^4 = Z, \quad T^8 = I$ |
| Non-Clifford proof | $TXT^\dagger = \frac{X+Y}{\sqrt{2}} \notin \mathcal{P}_1$ |
| Magic state | $\|T\rangle = T\|+\rangle = \frac{\|0\rangle + e^{i\pi/4}\|1\rangle}{\sqrt{2}}$ |
| Universality | Clifford + T = Universal |
| Hierarchy position | $T \in \mathcal{C}_3 \setminus \mathcal{C}_2$ |

### Main Takeaways

1. **The T-gate is the simplest non-Clifford gate** - It enables universal computation when combined with Cliffords

2. **T is non-Clifford because $TXT^\dagger$ is not a Pauli** - This breaks the classical simulability of Clifford circuits

3. **T-gate powers generate S and Z** - The relationship $T^2 = S$, $T^4 = Z$, $T^8 = I$ shows T is the "square root of S"

4. **T-gates dominate fault-tolerant costs** - Magic state distillation makes T-gates expensive; T-count is the key resource metric

5. **Magic states solve the transversality problem** - Since T cannot be transversal, we use gate teleportation with $|T\rangle$ states

---

## Daily Checklist

- [ ] Can write the T-gate matrix and its properties
- [ ] Can prove $T^2 = S$, $T^4 = Z$, $T^8 = I$
- [ ] Can prove T is non-Clifford (via $TXT^\dagger$)
- [ ] Understand why Clifford + T is universal
- [ ] Know the magic state $|T\rangle = T|+\rangle$
- [ ] Understand why T-gates are expensive in fault-tolerant QC
- [ ] Completed computational lab exercises

---

## Preview: Day 843

Tomorrow we dive deep into **magic states**: the resource states that enable fault-tolerant T-gate implementation. We will:

- Define the magic state $|T\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$
- Introduce the alternative magic state $|H\rangle$
- Explore the geometric meaning on the Bloch sphere
- Understand the "stabilizer polytope" that separates stabilizer states from magic states

Magic states are the key resource for universal fault-tolerant quantum computation!

---

*"The T-gate is quantum computing's most expensive gate, but also its most essential. Without it, we have only classical computation in quantum disguise."*

---

**Day 842 Complete** | **Next: Day 843 - Magic State Definition**
