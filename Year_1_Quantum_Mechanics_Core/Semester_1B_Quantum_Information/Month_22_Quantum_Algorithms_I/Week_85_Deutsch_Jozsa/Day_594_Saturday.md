# Day 594: Quantum Oracle Construction

## Overview

**Day 594** | Week 85, Day 6 | Month 22 | Quantum Algorithms I

Today we explore how to actually build quantum oracles from classical function descriptions. While theoretical analyses treat oracles as black boxes, practical quantum computing requires explicit circuit implementations. We'll learn systematic techniques for constructing oracles using standard quantum gates.

---

## Learning Objectives

1. Understand the oracle model's role in algorithm analysis vs. implementation
2. Construct oracles for Boolean functions using reversible logic
3. Implement oracles using Toffoli and controlled gates
4. Analyze oracle complexity in terms of gate count
5. Handle multi-output functions and workspace qubits
6. Recognize the overhead of oracle construction

---

## Core Content

### From Theory to Practice

In theoretical analysis, we say "given oracle access to $f$." In practice, we must:

1. **Know the function:** Have a classical description of $f$
2. **Compile to reversible form:** Make $f$ invertible
3. **Decompose into gates:** Express using available quantum gates
4. **Optimize:** Minimize gate count, depth, and ancilla qubits

### Standard Oracle Form

The standard oracle implements:
$$U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$$

This is automatically reversible (applying twice returns to original state).

### Building Blocks: Reversible Gates

**NOT gate (X):** $|x\rangle \mapsto |x \oplus 1\rangle$

**CNOT:** $|c\rangle|t\rangle \mapsto |c\rangle|t \oplus c\rangle$
```
c ───●─── c
     │
t ───⊕─── t ⊕ c
```

**Toffoli (CCNOT):** $|c_1\rangle|c_2\rangle|t\rangle \mapsto |c_1\rangle|c_2\rangle|t \oplus (c_1 \land c_2)\rangle$
```
c₁ ───●─── c₁
      │
c₂ ───●─── c₂
      │
t  ───⊕─── t ⊕ (c₁ ∧ c₂)
```

### Constructing Oracles from Boolean Functions

**Method 1: Sum of Products (SOP)**

Any Boolean function can be written as:
$$f(x) = \bigvee_{\text{minterms}} (x_{i_1} \land x_{i_2} \land \cdots)$$

**Example:** $f(x_1, x_2) = x_1 \oplus x_2$ (XOR)

Minterms: $(\bar{x}_1 \land x_2) \lor (x_1 \land \bar{x}_2)$

**Circuit construction:**
1. Compute each minterm using Toffoli gates
2. OR results using CNOT gates
3. Uncompute minterms to restore ancillas

### Oracle for XOR Function

$f(x_1, x_2) = x_1 \oplus x_2$

**Direct approach:** XOR is simply CNOT!
```
x₁ ───●─────●─── x₁
      │     │
x₂ ───┼─────●─── x₂
      │     │
y  ───⊕─────⊕─── y ⊕ x₁ ⊕ x₂ = y ⊕ f(x)
```

This uses 2 CNOT gates.

### Oracle for AND Function

$f(x_1, x_2) = x_1 \land x_2$

**Direct approach:** Toffoli gate!
```
x₁ ───●─── x₁
      │
x₂ ───●─── x₂
      │
y  ───⊕─── y ⊕ (x₁ ∧ x₂)
```

This uses 1 Toffoli gate (decomposes to ~6 CNOT + single-qubit gates).

### Oracle for OR Function

$f(x_1, x_2) = x_1 \lor x_2 = \lnot(\lnot x_1 \land \lnot x_2)$

Using De Morgan's law:
```
x₁ ──[X]───●───[X]─── x₁
           │
x₂ ──[X]───●───[X]─── x₂
           │
y  ────────⊕───[X]─── y ⊕ (x₁ ∨ x₂)
```

Alternative using ancilla:
```
x₁ ───●─────────────●─── x₁
      │             │
x₂ ───┼──●───────●──┼─── x₂
      │  │       │  │
a  ───⊕──⊕───●───⊕──⊕─── a (restored)
             │
y  ──────────⊕─────────── y ⊕ (x₁ ∨ x₂)
```

### Multi-Bit Output Functions

For $f: \{0,1\}^n \to \{0,1\}^m$, we need $m$ output qubits:

$$U_f|x\rangle|y_1\rangle|y_2\rangle\cdots|y_m\rangle = |x\rangle|y_1 \oplus f_1(x)\rangle|y_2 \oplus f_2(x)\rangle\cdots|y_m \oplus f_m(x)\rangle$$

Build separate circuits for each output bit $f_i(x)$.

### Workspace (Ancilla) Qubits

Complex functions often require **ancilla qubits** to store intermediate results.

**Rules for ancillas:**
1. Initialize to $|0\rangle$
2. Use for intermediate computation
3. **Must uncompute** to restore to $|0\rangle$ before measurement
4. Failure to uncompute causes **garbage bits** that create unwanted entanglement

### Bennett's Trick for Uncomputation

To compute $f(x)$ reversibly without leaving garbage:

```
|x⟩|0⟩|0⟩  →  Compute   →  |x⟩|f(x)⟩|garbage⟩
           →  Copy f(x)  →  |x⟩|f(x)⟩|garbage⟩|f(x)⟩
           →  Uncompute  →  |x⟩|0⟩|0⟩|f(x)⟩
```

This doubles the computation but cleans up ancillas.

### Oracle Complexity

The **oracle complexity** or **query complexity** counts oracle calls, not gates.

But for actual implementation, we care about:
- **Gate count:** Total number of elementary gates
- **Depth:** Longest path through circuit (parallel time)
- **Width:** Number of qubits including ancillas

**Typical overheads:**
- $n$-bit Toffoli: $O(n)$ depth, $O(n)$ ancillas with linear depth
- Generic $n$-bit function: $O(2^n)$ gates worst case (exponential!)

### Phase Oracle from Standard Oracle

To convert standard oracle to phase oracle, use the $|-\rangle$ ancilla trick:

```
|x⟩ ──────●────── |x⟩
          │
|−⟩ ──────⊕────── (-1)^f(x)|−⟩
```

This is why algorithms prepare $|1\rangle$ then apply $H$: $H|1\rangle = |-\rangle$.

### Marking Oracle

Some algorithms need a **marking oracle** that marks specific states:

$$O_f|x\rangle = (-1)^{f(x)}|x\rangle$$

This is the phase oracle form, crucial for Grover's algorithm.

---

## Worked Examples

### Example 1: Majority Function Oracle

Build oracle for $f(x_1, x_2, x_3) = \text{MAJ}(x_1, x_2, x_3) = (x_1 \land x_2) \lor (x_1 \land x_3) \lor (x_2 \land x_3)$

**Solution:**

**Truth table:**

| $x_1x_2x_3$ | MAJ |
|-------------|-----|
| 000 | 0 |
| 001 | 0 |
| 010 | 0 |
| 011 | 1 |
| 100 | 0 |
| 101 | 1 |
| 110 | 1 |
| 111 | 1 |

**Circuit using ancillas:**
```
x₁ ──●─────●───────●─────●───────── x₁
     │     │       │     │
x₂ ──●─────┼──●────┼──●──┼───────── x₂
     │     │  │    │  │  │
x₃ ──┼─────●──●────●──●──┼───────── x₃
     │     │  │    │  │  │
a₁ ──⊕─────┼──┼────┼──┼──⊕──●────── a₁ (x₁∧x₂, uncomputed)
           │  │    │  │     │
a₂ ────────⊕──┼────⊕──┼─────●────── a₂ (x₁∧x₃, uncomputed)
              │       │     │
a₃ ───────────⊕───────⊕─────●────── a₃ (x₂∧x₃, uncomputed)
                            │
y  ─────────────────────────⊕────── y ⊕ MAJ
```

This requires 3 ancilla qubits and multiple Toffoli gates.

### Example 2: Bernstein-Vazirani Oracle

For $f(x) = s \cdot x$ with $s = 101$ (3 bits).

**Solution:**

$f(x) = s_1 x_1 \oplus s_2 x_2 \oplus s_3 x_3 = x_1 \oplus x_3$

We only XOR into $y$ for positions where $s_i = 1$:

```
x₁ ───●─── x₁
      │
x₂ ───┼─── x₂
      │
x₃ ───┼●── x₃
      ││
y  ───⊕⊕── y ⊕ x₁ ⊕ x₃
```

Just 2 CNOT gates! Very efficient.

**General pattern:** For Bernstein-Vazirani with secret string $s$, the oracle is just CNOTs from each $x_i$ where $s_i = 1$.

### Example 3: Simon's Oracle

For Simon's function with $s = 11$ (2 bits), $f$ is two-to-one.

Example: $f(00) = f(11) = 0$, $f(01) = f(10) = 1$

**Solution:**

This is equivalent to $f(x) = x_1 \oplus x_2$ (parity).

```
x₁ ───●─────── x₁
      │
x₂ ───┼──●──── x₂
      │  │
y  ───⊕──⊕──── y ⊕ (x₁ ⊕ x₂)
```

For more complex Simon functions, the oracle construction depends on the specific $f$.

---

## Practice Problems

### Problem 1: NAND Oracle

Construct the quantum oracle for $f(x_1, x_2) = \lnot(x_1 \land x_2)$ (NAND).

### Problem 2: 4-Bit Parity

Build an efficient oracle for 4-bit parity: $f(x) = x_1 \oplus x_2 \oplus x_3 \oplus x_4$.
What is the minimum number of CNOT gates needed?

### Problem 3: Comparison Oracle

Design an oracle that marks states where $x > 5$ for 3-bit inputs $x \in \{0, ..., 7\}$.

### Problem 4: Uncomputation

Given a circuit that computes $f(x)$ but leaves garbage $g(x)$:
$$|x\rangle|0\rangle|0\rangle \mapsto |x\rangle|f(x)\rangle|g(x)\rangle$$

Show how to use Bennett's trick to create a clean oracle.

---

## Computational Lab

```python
"""Day 594: Quantum Oracle Construction"""
import numpy as np
from typing import Callable, List, Tuple

def cnot_matrix():
    """CNOT gate matrix (control on qubit 0, target on qubit 1)"""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

def toffoli_matrix():
    """Toffoli (CCNOT) gate matrix"""
    T = np.eye(8)
    T[6, 6] = 0
    T[7, 7] = 0
    T[6, 7] = 1
    T[7, 6] = 1
    return T

def x_gate():
    """Pauli X (NOT) gate"""
    return np.array([[0, 1], [1, 0]])

def identity(n):
    """n-qubit identity"""
    return np.eye(2**n)

def tensor(*gates):
    """Tensor product of gates"""
    result = gates[0]
    for g in gates[1:]:
        result = np.kron(result, g)
    return result

def controlled_gate(gate, num_controls=1, control_state=1):
    """Create controlled version of a gate"""
    gate_dim = gate.shape[0]
    total_dim = 2**num_controls * gate_dim

    result = np.eye(total_dim, dtype=complex)

    # Apply gate when control qubits are in control_state
    start_idx = control_state * gate_dim
    result[start_idx:start_idx+gate_dim, start_idx:start_idx+gate_dim] = gate

    return result

class OracleBuilder:
    """Class for constructing quantum oracles"""

    def __init__(self, n_input: int, n_output: int = 1, n_ancilla: int = 0):
        self.n_input = n_input
        self.n_output = n_output
        self.n_ancilla = n_ancilla
        self.n_total = n_input + n_output + n_ancilla
        self.gates = []

    def add_x(self, qubit: int):
        """Add X gate on specified qubit"""
        self.gates.append(('X', qubit))

    def add_cnot(self, control: int, target: int):
        """Add CNOT gate"""
        self.gates.append(('CNOT', control, target))

    def add_toffoli(self, control1: int, control2: int, target: int):
        """Add Toffoli gate"""
        self.gates.append(('TOFFOLI', control1, control2, target))

    def build_matrix(self) -> np.ndarray:
        """Build the unitary matrix for the oracle"""
        n = self.n_total
        result = np.eye(2**n, dtype=complex)

        for gate in self.gates:
            if gate[0] == 'X':
                qubit = gate[1]
                # X on specified qubit
                gate_matrix = np.eye(2**n, dtype=complex)
                for i in range(2**n):
                    # Flip the bit at position qubit
                    j = i ^ (1 << (n - 1 - qubit))
                    gate_matrix[i, i] = 0
                    gate_matrix[j, i] = 1
                result = gate_matrix @ result

            elif gate[0] == 'CNOT':
                control, target = gate[1], gate[2]
                gate_matrix = np.eye(2**n, dtype=complex)
                for i in range(2**n):
                    # Check if control bit is 1
                    control_bit = (i >> (n - 1 - control)) & 1
                    if control_bit == 1:
                        # Flip target
                        j = i ^ (1 << (n - 1 - target))
                        gate_matrix[i, i] = 0
                        gate_matrix[j, i] = 1
                result = gate_matrix @ result

            elif gate[0] == 'TOFFOLI':
                c1, c2, target = gate[1], gate[2], gate[3]
                gate_matrix = np.eye(2**n, dtype=complex)
                for i in range(2**n):
                    c1_bit = (i >> (n - 1 - c1)) & 1
                    c2_bit = (i >> (n - 1 - c2)) & 1
                    if c1_bit == 1 and c2_bit == 1:
                        j = i ^ (1 << (n - 1 - target))
                        gate_matrix[i, i] = 0
                        gate_matrix[j, i] = 1
                result = gate_matrix @ result

        return result

    def verify(self, f: Callable[[int], int]) -> bool:
        """Verify oracle implements the correct function"""
        U = self.build_matrix()
        n = self.n_total

        for x in range(2**self.n_input):
            for y in range(2**self.n_output):
                # Input state: |x⟩|y⟩|0...0⟩ (ancillas in |0⟩)
                input_idx = (x << (self.n_output + self.n_ancilla)) | (y << self.n_ancilla)

                # Expected output: |x⟩|y ⊕ f(x)⟩|0...0⟩
                fx = f(x) & ((1 << self.n_output) - 1)
                expected_y = y ^ fx
                expected_idx = (x << (self.n_output + self.n_ancilla)) | (expected_y << self.n_ancilla)

                # Find actual output
                input_state = np.zeros(2**n)
                input_state[input_idx] = 1
                output_state = U @ input_state

                # Check
                if abs(output_state[expected_idx] - 1) > 1e-10:
                    print(f"Failed: x={x:0{self.n_input}b}, y={y}, "
                          f"expected idx {expected_idx}, got something else")
                    return False

        return True

def build_xor_oracle(n: int) -> OracleBuilder:
    """Build oracle for n-bit XOR (parity)"""
    # f(x) = x_0 ⊕ x_1 ⊕ ... ⊕ x_{n-1}
    builder = OracleBuilder(n, 1)

    # CNOT from each input bit to output
    output_qubit = n  # Output is after all inputs

    for i in range(n):
        builder.add_cnot(i, output_qubit)

    return builder

def build_and_oracle() -> OracleBuilder:
    """Build oracle for 2-bit AND"""
    builder = OracleBuilder(2, 1)
    # Toffoli: target = target ⊕ (c1 ∧ c2)
    builder.add_toffoli(0, 1, 2)
    return builder

def build_or_oracle() -> OracleBuilder:
    """Build oracle for 2-bit OR using De Morgan"""
    # x1 ∨ x2 = ¬(¬x1 ∧ ¬x2)
    builder = OracleBuilder(2, 1)

    # NOT both inputs
    builder.add_x(0)
    builder.add_x(1)

    # AND (gives ¬x1 ∧ ¬x2)
    builder.add_toffoli(0, 1, 2)

    # NOT output (gives ¬(¬x1 ∧ ¬x2) = x1 ∨ x2)
    builder.add_x(2)

    # Restore inputs
    builder.add_x(0)
    builder.add_x(1)

    return builder

def build_bv_oracle(s: int, n: int) -> OracleBuilder:
    """Build Bernstein-Vazirani oracle for secret string s"""
    # f(x) = s · x mod 2
    builder = OracleBuilder(n, 1)
    output_qubit = n

    for i in range(n):
        if (s >> (n - 1 - i)) & 1:
            builder.add_cnot(i, output_qubit)

    return builder

def build_majority_oracle() -> OracleBuilder:
    """Build oracle for 3-bit majority function"""
    # MAJ(x1,x2,x3) = (x1∧x2) ∨ (x1∧x3) ∨ (x2∧x3)
    # Using 3 ancillas for the three AND terms

    builder = OracleBuilder(3, 1, 3)
    # Qubits: 0,1,2 = inputs; 3 = output; 4,5,6 = ancillas

    # Compute x1 ∧ x2 into ancilla 4
    builder.add_toffoli(0, 1, 4)

    # Compute x1 ∧ x3 into ancilla 5
    builder.add_toffoli(0, 2, 5)

    # Compute x2 ∧ x3 into ancilla 6
    builder.add_toffoli(1, 2, 6)

    # OR all three ancillas into output
    # OR(a,b,c) = a ⊕ b ⊕ c ⊕ (a∧b) ⊕ (a∧c) ⊕ (b∧c) ⊕ (a∧b∧c)
    # This is getting complex... let's use a simpler approach

    # Actually, for majority: at least 2 of 3 are 1
    # A cleaner circuit exists but requires more gates

    # Simple approach: XOR all terms into output (works due to majority structure)
    builder.add_cnot(4, 3)
    builder.add_cnot(5, 3)
    builder.add_cnot(6, 3)

    # Need to account for overlaps...
    # Actually let's compute directly

    # Uncompute ancillas
    builder.add_toffoli(1, 2, 6)
    builder.add_toffoli(0, 2, 5)
    builder.add_toffoli(0, 1, 4)

    return builder

# Test all oracles
print("=" * 60)
print("ORACLE CONSTRUCTION TESTS")
print("=" * 60)

# XOR oracle
print("\n--- XOR (Parity) Oracle ---")
for n in [2, 3, 4]:
    builder = build_xor_oracle(n)
    f = lambda x, n=n: bin(x).count('1') % 2
    passed = builder.verify(f)
    print(f"n={n}: {'PASSED' if passed else 'FAILED'}, {len(builder.gates)} gates")

# AND oracle
print("\n--- AND Oracle ---")
builder = build_and_oracle()
f_and = lambda x: (x >> 1) & (x & 1)
passed = builder.verify(f_and)
print(f"Verification: {'PASSED' if passed else 'FAILED'}")
print(f"Gates: {builder.gates}")

# OR oracle
print("\n--- OR Oracle ---")
builder = build_or_oracle()
f_or = lambda x: ((x >> 1) | (x & 1)) & 1
passed = builder.verify(f_or)
print(f"Verification: {'PASSED' if passed else 'FAILED'}")
print(f"Gates: {builder.gates}")

# BV oracle
print("\n--- Bernstein-Vazirani Oracle ---")
for s, n in [(0b101, 3), (0b1011, 4), (0b11, 2)]:
    builder = build_bv_oracle(s, n)
    f_bv = lambda x, s=s: bin(x & s).count('1') % 2
    passed = builder.verify(f_bv)
    gates_used = len([g for g in builder.gates if g[0] == 'CNOT'])
    print(f"s={s:0{n}b}: {'PASSED' if passed else 'FAILED'}, {gates_used} CNOTs")

# Gate count analysis
print("\n" + "=" * 60)
print("GATE COUNT ANALYSIS")
print("=" * 60)

print("\n| Function | Input bits | CNOTs | Toffolis | X gates |")
print("|----------|------------|-------|----------|---------|")

for name, builder in [
    ("XOR (3-bit)", build_xor_oracle(3)),
    ("AND (2-bit)", build_and_oracle()),
    ("OR (2-bit)", build_or_oracle()),
    ("BV s=101", build_bv_oracle(0b101, 3)),
]:
    cnots = sum(1 for g in builder.gates if g[0] == 'CNOT')
    toffolis = sum(1 for g in builder.gates if g[0] == 'TOFFOLI')
    x_gates = sum(1 for g in builder.gates if g[0] == 'X')
    print(f"| {name:16s} | {builder.n_input:10d} | {cnots:5d} | {toffolis:8d} | {x_gates:7d} |")

# Demonstrate phase oracle construction
print("\n" + "=" * 60)
print("PHASE ORACLE DEMONSTRATION")
print("=" * 60)

def standard_to_phase_oracle(U_standard: np.ndarray, n_input: int) -> np.ndarray:
    """
    Convert standard oracle U_f|x⟩|y⟩ = |x⟩|y⊕f(x)⟩
    to phase oracle O_f|x⟩ = (-1)^{f(x)}|x⟩

    Uses ancilla prepared in |−⟩
    """
    # The standard oracle with |y⟩ = |−⟩ gives phase kickback
    # We simulate this by computing f(x) and applying Z-like operation

    dim = 2 ** n_input

    # Extract f(x) from standard oracle
    fx = []
    for x in range(dim):
        # Apply U to |x⟩|0⟩
        input_vec = np.zeros(dim * 2)
        input_vec[2 * x] = 1  # |x⟩|0⟩
        output_vec = U_standard @ input_vec

        # Output is |x⟩|f(x)⟩
        # f(x) = 0 if amplitude at |x⟩|0⟩ is 1
        # f(x) = 1 if amplitude at |x⟩|1⟩ is 1
        if abs(output_vec[2*x + 1]) > 0.5:
            fx.append(1)
        else:
            fx.append(0)

    # Build phase oracle
    O_phase = np.diag([(-1)**fx[x] for x in range(dim)])
    return O_phase

# Example: AND function phase oracle
builder = build_and_oracle()
U_and = builder.build_matrix()
O_and_phase = standard_to_phase_oracle(U_and, 2)

print("\nAND function phase oracle O_f|x⟩ = (-1)^{f(x)}|x⟩:")
for x in range(4):
    phase = O_and_phase[x, x]
    print(f"  |{x:02b}⟩ → {'+1' if phase > 0 else '-1'} |{x:02b}⟩  (f({x:02b}) = {1 if phase < 0 else 0})")

# Visualization
print("\n" + "=" * 60)
print("GENERATING CIRCUIT VISUALIZATION")
print("=" * 60)

import matplotlib.pyplot as plt

def draw_circuit(builder: OracleBuilder, title: str, filename: str):
    """Draw a simple circuit diagram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    n = builder.n_total
    n_gates = len(builder.gates)

    # Set up axes
    ax.set_xlim(-1, n_gates + 2)
    ax.set_ylim(-0.5, n + 0.5)
    ax.axis('off')

    # Draw qubit lines
    for i in range(n):
        ax.plot([-0.5, n_gates + 1.5], [n - 1 - i, n - 1 - i], 'k-', linewidth=1)
        if i < builder.n_input:
            label = f'x_{i+1}'
        elif i < builder.n_input + builder.n_output:
            label = f'y_{i - builder.n_input + 1}'
        else:
            label = f'a_{i - builder.n_input - builder.n_output + 1}'
        ax.text(-0.8, n - 1 - i, label, fontsize=10, va='center', ha='right')

    # Draw gates
    for g_idx, gate in enumerate(builder.gates):
        x_pos = g_idx + 0.5

        if gate[0] == 'X':
            qubit = gate[1]
            y_pos = n - 1 - qubit
            circle = plt.Circle((x_pos, y_pos), 0.15, fill=False, color='blue')
            ax.add_patch(circle)
            ax.plot([x_pos - 0.1, x_pos + 0.1], [y_pos, y_pos], 'b-', linewidth=2)
            ax.plot([x_pos, x_pos], [y_pos - 0.1, y_pos + 0.1], 'b-', linewidth=2)

        elif gate[0] == 'CNOT':
            control, target = gate[1], gate[2]
            y_ctrl = n - 1 - control
            y_tgt = n - 1 - target
            # Control dot
            ax.plot(x_pos, y_ctrl, 'ko', markersize=8)
            # Target circle
            circle = plt.Circle((x_pos, y_tgt), 0.15, fill=False, color='black')
            ax.add_patch(circle)
            ax.plot([x_pos, x_pos], [y_tgt - 0.15, y_tgt + 0.15], 'k-', linewidth=1)
            ax.plot([x_pos - 0.15, x_pos + 0.15], [y_tgt, y_tgt], 'k-', linewidth=1)
            # Vertical line
            ax.plot([x_pos, x_pos], [min(y_ctrl, y_tgt), max(y_ctrl, y_tgt)], 'k-', linewidth=1)

        elif gate[0] == 'TOFFOLI':
            c1, c2, target = gate[1], gate[2], gate[3]
            y_c1 = n - 1 - c1
            y_c2 = n - 1 - c2
            y_tgt = n - 1 - target
            # Control dots
            ax.plot(x_pos, y_c1, 'ko', markersize=8)
            ax.plot(x_pos, y_c2, 'ko', markersize=8)
            # Target circle
            circle = plt.Circle((x_pos, y_tgt), 0.15, fill=False, color='black')
            ax.add_patch(circle)
            ax.plot([x_pos, x_pos], [y_tgt - 0.15, y_tgt + 0.15], 'k-', linewidth=1)
            ax.plot([x_pos - 0.15, x_pos + 0.15], [y_tgt, y_tgt], 'k-', linewidth=1)
            # Vertical line
            y_min = min(y_c1, y_c2, y_tgt)
            y_max = max(y_c1, y_c2, y_tgt)
            ax.plot([x_pos, x_pos], [y_min, y_max], 'k-', linewidth=1)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Circuit saved to '{filename}'")

# Generate circuit diagrams
draw_circuit(build_and_oracle(), "AND Oracle Circuit", "oracle_and.png")
draw_circuit(build_or_oracle(), "OR Oracle Circuit (De Morgan)", "oracle_or.png")
draw_circuit(build_bv_oracle(0b101, 3), "Bernstein-Vazirani Oracle (s=101)", "oracle_bv.png")
```

**Expected Output:**
```
============================================================
ORACLE CONSTRUCTION TESTS
============================================================

--- XOR (Parity) Oracle ---
n=2: PASSED, 2 gates
n=3: PASSED, 3 gates
n=4: PASSED, 4 gates

--- AND Oracle ---
Verification: PASSED
Gates: [('TOFFOLI', 0, 1, 2)]

--- OR Oracle ---
Verification: PASSED
Gates: [('X', 0), ('X', 1), ('TOFFOLI', 0, 1, 2), ('X', 2), ('X', 0), ('X', 1)]
...
```

---

## Summary

### Key Formulas

| Oracle Type | Form | Implementation |
|-------------|------|----------------|
| Standard | $U_f\|x\rangle\|y\rangle = \|x\rangle\|y \oplus f(x)\rangle$ | Reversible gates |
| Phase | $O_f\|x\rangle = (-1)^{f(x)}\|x\rangle$ | Standard + $\|-\rangle$ ancilla |
| Marking | Same as phase | Used in Grover's search |

### Key Takeaways

1. **Oracle construction** bridges theory and practice
2. **Reversible computing** requires all intermediate values be uncomputable
3. **Bennett's trick** allows clean computation with ancilla qubits
4. **Gate complexity** can be exponential in the worst case
5. **Efficient oracles** exist for structured functions (XOR, arithmetic)
6. **Phase kickback** converts standard oracles to phase oracles naturally

---

## Daily Checklist

- [ ] I understand the difference between oracle model and explicit circuits
- [ ] I can construct simple oracles using Toffoli and CNOT gates
- [ ] I know how to convert standard oracles to phase oracles
- [ ] I understand the role of ancilla qubits and uncomputation
- [ ] I can estimate gate count for basic Boolean functions
- [ ] I ran the lab and verified oracle correctness

---

*Next: Day 595 - Week Review*
