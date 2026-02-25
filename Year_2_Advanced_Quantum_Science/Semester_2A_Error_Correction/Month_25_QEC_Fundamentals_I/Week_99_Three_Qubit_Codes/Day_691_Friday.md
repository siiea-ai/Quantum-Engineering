# Day 691: Steane [[7,1,3]] Code Introduction

## Overview

**Week:** 99 (Three-Qubit Codes and Beyond)
**Day:** Friday
**Date:** Year 2, Month 25, Day 691
**Topic:** The Steane Code — First Code Correcting All Single-Qubit Errors
**Hours:** 7 (3.5 theory + 2.5 problems + 1 computational lab)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Steane code construction, classical Hamming codes |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Encoding circuits, syndrome measurement |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Qiskit implementation of Steane code |

---

## Prerequisites

From Days 687-690:
- Stabilizer formalism and Pauli group structure
- Binary symplectic representation
- Knill-Laflamme conditions
- Shor [[9,1,3]] code analysis

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Construct** the Steane [[7,1,3]] code from the classical [7,4,3] Hamming code
2. **Write** all 6 stabilizer generators and verify their commutation
3. **Derive** the logical operators $\bar{X}$ and $\bar{Z}$ for the Steane code
4. **Implement** the encoding circuit using CNOT and Hadamard gates
5. **Calculate** syndromes for all single-qubit X, Y, and Z errors
6. **Compare** the Steane code efficiency with the Shor code

---

## Core Content

### 1. Classical Hamming Code Foundation

The Steane code is built from the classical [7,4,3] Hamming code, demonstrating the deep connection between classical and quantum error correction.

#### The [7,4,3] Hamming Code

The classical Hamming code encodes 4 bits into 7 bits and corrects any single bit-flip error.

**Parity-check matrix:**

$$H_{classical} = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$

**Generator matrix:**

$$G = \begin{pmatrix} 1 & 1 & 0 & 1 & 0 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 & 0 \\ 0 & 1 & 1 & 0 & 0 & 1 & 0 \\ 1 & 1 & 1 & 0 & 0 & 0 & 1 \end{pmatrix}$$

**Codewords:** The 16 codewords (including 0000000 and 1111111) form a linear subspace with minimum distance 3.

#### Self-Dual Property

The [7,4,3] Hamming code has a remarkable property: it contains its dual code. Specifically, the codewords of the Hamming code include:

$$C^\perp \subset C$$

This self-orthogonality is crucial for constructing a quantum code where X and Z stabilizers commute.

---

### 2. Steane Code Construction

#### CSS Code from Hamming

Andrew Steane's insight (1996) was to use the Hamming code twice—once for X-type stabilizers (detecting Z errors) and once for Z-type stabilizers (detecting X errors).

**Steane Code Parameters:**

$$\boxed{[[n, k, d]] = [[7, 1, 3]]}$$

- **n = 7:** Physical qubits
- **k = 1:** Logical qubit
- **d = 3:** Code distance (corrects any single-qubit error)

#### Stabilizer Generators

The 6 stabilizer generators come from the rows of the Hamming parity-check matrix:

**X-type stabilizers** (detect Z errors):

$$\begin{aligned}
g_1 &= X_4 X_5 X_6 X_7 = IIIXXXX \\
g_2 &= X_2 X_3 X_6 X_7 = IXXIIXX \\
g_3 &= X_1 X_3 X_5 X_7 = XIXIXIX
\end{aligned}$$

**Z-type stabilizers** (detect X errors):

$$\begin{aligned}
g_4 &= Z_4 Z_5 Z_6 Z_7 = IIIZZZZ \\
g_5 &= Z_2 Z_3 Z_6 Z_7 = IZZIIZZ \\
g_6 &= Z_1 Z_3 Z_5 Z_7 = ZIZIZIZ
\end{aligned}$$

#### Commutation Verification

For the code to be valid, all stabilizers must commute. For $g_i$ (X-type) and $g_j$ (Z-type):

$$[g_i, g_j] = 0 \iff |supp(g_i) \cap supp(g_j)| \equiv 0 \pmod{2}$$

Let's verify $[g_1, g_4]$:
- $g_1$ acts on qubits {4, 5, 6, 7}
- $g_4$ acts on qubits {4, 5, 6, 7}
- Intersection: {4, 5, 6, 7}, size = 4 (even) ✓

The self-orthogonality of the Hamming code guarantees all pairs commute!

---

### 3. Logical Operators

#### Logical X Operator

The logical $\bar{X}$ must:
1. Anticommute with $\bar{Z}$
2. Commute with all stabilizers
3. Not be in the stabilizer group

$$\boxed{\bar{X} = X_1 X_2 X_3 X_4 X_5 X_6 X_7 = X^{\otimes 7}}$$

This works because the all-ones vector is a Hamming codeword, ensuring $\bar{X}$ commutes with all Z-stabilizers.

#### Logical Z Operator

Similarly:

$$\boxed{\bar{Z} = Z_1 Z_2 Z_3 Z_4 Z_5 Z_6 Z_7 = Z^{\otimes 7}}$$

#### Verification of Anticommutation

$$\bar{X}\bar{Z} = X^{\otimes 7} Z^{\otimes 7} = (XZ)^{\otimes 7} = (-ZX)^{\otimes 7} = (-1)^7 Z^{\otimes 7} X^{\otimes 7} = -\bar{Z}\bar{X}$$

Therefore: $\{\bar{X}, \bar{Z}\} = 0$ ✓

#### Equivalent Logical Operators

Multiplying by stabilizers gives equivalent logical operators:

$$\bar{X} \cdot g_1 = X_1 X_2 X_3 \cdot IIIXXXX \cdot IIIXIII = X_1 X_2 X_3$$

So $X_1 X_2 X_3$ is also a valid logical X (weight 3, matching code distance).

---

### 4. Logical Basis States

#### Logical |0⟩

$$|0_L\rangle = \frac{1}{\sqrt{8}} \sum_{c \in C} |c\rangle$$

where $C$ is the set of Hamming codewords:

$$|0_L\rangle = \frac{1}{\sqrt{8}}(|0000000\rangle + |1010101\rangle + |0110011\rangle + |1100110\rangle$$
$$+ |0001111\rangle + |1011010\rangle + |0111100\rangle + |1101001\rangle)$$

This is the uniform superposition over all even-weight Hamming codewords.

#### Logical |1⟩

$$|1_L\rangle = \bar{X}|0_L\rangle = \frac{1}{\sqrt{8}} \sum_{c \in C} |c \oplus 1111111\rangle$$

$$|1_L\rangle = \frac{1}{\sqrt{8}}(|1111111\rangle + |0101010\rangle + |1001100\rangle + |0011001\rangle$$
$$+ |1110000\rangle + |0100101\rangle + |1000011\rangle + |0010110\rangle)$$

This is the uniform superposition over all odd-weight Hamming codewords.

---

### 5. Encoding Circuit

The Steane code encoding requires:
1. Prepare ancilla qubits
2. Apply CNOT gates according to generator matrix
3. Apply Hadamard gates to create superposition

#### Circuit Structure

```
|ψ⟩ ─────●───●───────●───────────────
         │   │       │
|0⟩ ──H──●───│───●───│───●───────────
         │   │   │   │   │
|0⟩ ──H──●───│───│───●───│───●───────
             │   │       │   │
|0⟩ ──H──────●───●───────│───│───●───
                         │   │   │
|0⟩ ──H──────────────────●───│───│───
                             │   │
|0⟩ ──H──────────────────────●───│───
                                 │
|0⟩ ──H──────────────────────────●───
```

#### Encoding Procedure

1. **Initialize:** Start with $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ on qubit 1, $|0\rangle^{\otimes 6}$ on qubits 2-7
2. **Hadamard:** Apply H to qubits 2-7
3. **CNOT cascade:** Apply CNOTs according to the stabilizer structure
4. **Result:** $|\psi_L\rangle = \alpha|0_L\rangle + \beta|1_L\rangle$

---

### 6. Syndrome Measurement

#### Error-Syndrome Correspondence

Each error produces a unique syndrome from the 6 stabilizer measurements:

| Error | $g_1$ | $g_2$ | $g_3$ | $g_4$ | $g_5$ | $g_6$ | Syndrome |
|-------|-------|-------|-------|-------|-------|-------|----------|
| $I$ | +1 | +1 | +1 | +1 | +1 | +1 | 000 000 |
| $X_1$ | +1 | +1 | +1 | +1 | +1 | -1 | 000 001 |
| $X_2$ | +1 | +1 | +1 | +1 | -1 | +1 | 000 010 |
| $X_3$ | +1 | +1 | +1 | +1 | -1 | -1 | 000 011 |
| $X_4$ | +1 | +1 | +1 | -1 | +1 | +1 | 000 100 |
| $X_5$ | +1 | +1 | +1 | -1 | +1 | -1 | 000 101 |
| $X_6$ | +1 | +1 | +1 | -1 | -1 | +1 | 000 110 |
| $X_7$ | +1 | +1 | +1 | -1 | -1 | -1 | 000 111 |
| $Z_1$ | +1 | +1 | -1 | +1 | +1 | +1 | 001 000 |
| $Z_2$ | +1 | -1 | +1 | +1 | +1 | +1 | 010 000 |
| $Z_3$ | +1 | -1 | -1 | +1 | +1 | +1 | 011 000 |
| ... | ... | ... | ... | ... | ... | ... | ... |

#### Y Error Syndromes

For $Y_j = iX_j Z_j$ errors:

$$\text{syndrome}(Y_j) = \text{syndrome}(X_j) \oplus \text{syndrome}(Z_j)$$

This gives 7 additional unique syndromes.

#### Syndrome Uniqueness

Total distinguishable syndromes: $2^6 = 64$

Errors to distinguish:
- 1 identity (no error)
- 7 single X errors
- 7 single Z errors
- 7 single Y errors
- **Total: 22 errors**

Since $22 < 64$, all single-qubit errors have unique syndromes! ✓

---

### 7. Code Distance Verification

The code distance is the minimum weight of a non-trivial logical operator:

$$d = \min_{L \in N(S) \setminus S} \text{wt}(L)$$

**Minimum weight logical operators:**
- $\bar{X}_{min} = X_1 X_2 X_3$ (weight 3)
- $\bar{Z}_{min} = Z_1 Z_2 Z_3$ (weight 3)

Therefore: $d = 3$

**Error correction capability:**

$$t = \left\lfloor \frac{d-1}{2} \right\rfloor = \left\lfloor \frac{2}{2} \right\rfloor = 1$$

The Steane code corrects any single-qubit error (X, Y, or Z).

---

### 8. Comparison with Shor Code

| Property | Shor [[9,1,3]] | Steane [[7,1,3]] |
|----------|----------------|------------------|
| Physical qubits | 9 | 7 |
| Encoding rate | 1/9 ≈ 11.1% | 1/7 ≈ 14.3% |
| Stabilizer generators | 8 | 6 |
| Distance | 3 | 3 |
| Transversal gates | CNOT | CNOT, H, S, T (with tricks) |
| CSS code | Yes | Yes |
| Construction | Concatenation | Direct from Hamming |

**Steane code advantages:**
1. More efficient (fewer qubits)
2. More transversal gates
3. Simpler syndrome structure
4. Natural for fault-tolerant computation

---

## Quantum Mechanics Connection

### Symmetry and Conservation

The Steane code's stabilizer structure reflects the deep connection between:

1. **Classical coding theory** — Linear codes over $\mathbb{F}_2$
2. **Quantum superposition** — Codewords in equal superposition
3. **Symmetry groups** — The stabilizer group as a symmetry

### Fault-Tolerant Gates

The Steane code supports **transversal** implementation of:
- Pauli gates: $X^{\otimes 7}$, $Z^{\otimes 7}$
- CNOT: $CNOT^{\otimes 7}$
- Hadamard: $H^{\otimes 7}$
- Phase: $S^{\otimes 7}$

This makes it a cornerstone of fault-tolerant quantum computing.

---

## Worked Examples

### Example 1: Syndrome Calculation

**Problem:** Calculate the syndrome for error $E = Z_3 X_5$.

**Solution:**

For Z-type stabilizers (measure X errors):
- $g_1 = IIIXXXX$: Position 5 has X, $g_1$ has X there → contributes to syndrome
  - $g_1$ and $X_5$: anticommute (1 overlap) → -1
- $g_2 = IXXIIXX$: Position 5 is I → +1
- $g_3 = XIXIXIX$: Position 5 has X → anticommute → -1

For X-type stabilizers (measure Z errors):
- $g_4 = IIIZZZZ$: Position 3 has I → +1
- $g_5 = IZZIIZZ$: Position 3 has Z → anticommute → -1
- $g_6 = ZIZIZIZ$: Position 3 has Z → anticommute → -1

**Syndrome:** $(g_1, g_2, g_3, g_4, g_5, g_6) = (-1, +1, -1, +1, -1, -1)$

Binary: $(1, 0, 1, 0, 1, 1)$ = $(101, 011)$ = syndrome(X_5) ⊕ syndrome(Z_3)

---

### Example 2: Encoding a Logical State

**Problem:** Show that $|+_L\rangle = \frac{1}{\sqrt{2}}(|0_L\rangle + |1_L\rangle)$ is a +1 eigenstate of $\bar{X}$.

**Solution:**

$$\bar{X}|+_L\rangle = \bar{X} \cdot \frac{1}{\sqrt{2}}(|0_L\rangle + |1_L\rangle)$$

Using $\bar{X}|0_L\rangle = |1_L\rangle$ and $\bar{X}|1_L\rangle = |0_L\rangle$:

$$= \frac{1}{\sqrt{2}}(|1_L\rangle + |0_L\rangle) = |+_L\rangle$$

Therefore: $\bar{X}|+_L\rangle = +1 \cdot |+_L\rangle$ ✓

---

### Example 3: Minimum Weight Logical Operator

**Problem:** Find all weight-3 representatives of $\bar{X}$.

**Solution:**

The minimum weight logical X operators are products of Hamming codewords with odd weight. The weight-3 Hamming codewords are:

- 1110000 → $X_1 X_2 X_3$
- 1001100 → $X_1 X_4 X_5$
- 0101010 → $X_2 X_4 X_6$
- 0011001 → $X_3 X_4 X_7$
- 1000011 → $X_1 X_6 X_7$
- 0100101 → $X_2 X_5 X_7$
- 0010110 → $X_3 X_5 X_6$

There are exactly **7 weight-3 representatives** of $\bar{X}$, one touching each physical qubit.

---

## Practice Problems

### Level 1: Direct Application

1. **Stabilizer Verification:**
   Verify that $g_2 = IXXIIXX$ and $g_5 = IZZIIZZ$ commute.

2. **Syndrome Table:**
   Complete the syndrome table for $Z_4$, $Z_5$, $Z_6$, $Z_7$.

3. **Logical State Expansion:**
   Write out $|0_L\rangle$ as a superposition of 8 computational basis states.

### Level 2: Intermediate

4. **Two-Qubit Error:**
   What syndrome does the error $X_1 X_2$ produce? Can it be corrected?

5. **Equivalent Logical Z:**
   Find a weight-3 representative of $\bar{Z}$ that acts on qubits {1, 4, 6}.

6. **Encoding Circuit:**
   Design a circuit to encode $|1\rangle$ into $|1_L\rangle$ using at most 10 gates.

### Level 3: Challenging

7. **Degeneracy Check:**
   Is the Steane code degenerate? Justify with examples.

8. **Transversal Hadamard:**
   Show that $H^{\otimes 7}$ maps $|0_L\rangle \leftrightarrow |+_L\rangle$ and $|1_L\rangle \leftrightarrow |-_L\rangle$.

9. **CSS Code Proof:**
   Prove that any code constructed from a self-orthogonal classical code has commuting X and Z stabilizers.

---

## Computational Lab

### Steane Code Implementation in Qiskit

```python
"""
Day 691 Computational Lab: Steane [[7,1,3]] Code
Full implementation with encoding, error injection, and correction
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt

class SteaneCode:
    """Complete implementation of the Steane [[7,1,3]] code."""

    def __init__(self):
        self.n_physical = 7
        self.n_logical = 1
        self.distance = 3

        # Stabilizer generators (X-type)
        self.x_stabs = [
            [0, 0, 0, 1, 1, 1, 1],  # g1 = IIIXXXX
            [0, 1, 1, 0, 0, 1, 1],  # g2 = IXXIIXX
            [1, 0, 1, 0, 1, 0, 1],  # g3 = XIXIXIX
        ]

        # Stabilizer generators (Z-type)
        self.z_stabs = [
            [0, 0, 0, 1, 1, 1, 1],  # g4 = IIIZZZZ
            [0, 1, 1, 0, 0, 1, 1],  # g5 = IZZIIZZ
            [1, 0, 1, 0, 1, 0, 1],  # g6 = ZIZIZIZ
        ]

        # Hamming codewords (for encoding)
        self.hamming_codewords = [
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 1],
        ]

    def create_encoding_circuit(self, include_input=True):
        """Create the Steane code encoding circuit."""
        qr = QuantumRegister(7, 'q')
        qc = QuantumCircuit(qr)

        # Apply Hadamard to qubits 1-6 (0-indexed: 1, 2, 3, 4, 5, 6)
        for i in range(1, 7):
            qc.h(i)

        # CNOT gates based on stabilizer structure
        # These create the proper superposition of codewords
        qc.cx(0, 1)  # Controlled by input qubit
        qc.cx(0, 2)
        qc.cx(0, 4)

        # Additional CNOTs for stabilizer structure
        qc.cx(6, 0)
        qc.cx(6, 1)
        qc.cx(6, 3)

        qc.cx(5, 0)
        qc.cx(5, 2)
        qc.cx(5, 3)

        qc.cx(4, 1)
        qc.cx(4, 2)
        qc.cx(4, 3)

        return qc

    def create_syndrome_circuit(self):
        """Create syndrome measurement circuit."""
        qr_data = QuantumRegister(7, 'data')
        qr_anc = QuantumRegister(6, 'ancilla')
        cr = ClassicalRegister(6, 'syndrome')

        qc = QuantumCircuit(qr_data, qr_anc, cr)

        # Measure X-type stabilizers (g1, g2, g3)
        # These detect Z errors
        for stab_idx, stab in enumerate(self.x_stabs):
            qc.h(qr_anc[stab_idx])
            for qubit_idx, val in enumerate(stab):
                if val == 1:
                    qc.cx(qr_anc[stab_idx], qr_data[qubit_idx])
            qc.h(qr_anc[stab_idx])

        # Measure Z-type stabilizers (g4, g5, g6)
        # These detect X errors
        for stab_idx, stab in enumerate(self.z_stabs):
            anc_idx = stab_idx + 3
            for qubit_idx, val in enumerate(stab):
                if val == 1:
                    qc.cx(qr_data[qubit_idx], qr_anc[anc_idx])

        # Measure ancillas
        qc.measure(qr_anc, cr)

        return qc

    def calculate_syndrome(self, error_type, error_qubit):
        """
        Calculate syndrome for a given error.

        Args:
            error_type: 'X', 'Y', 'Z', or 'I'
            error_qubit: qubit index (0-6) or None for identity

        Returns:
            6-bit syndrome string
        """
        if error_type == 'I' or error_qubit is None:
            return '000000'

        syndrome = []

        # X-stabilizers detect Z errors
        for stab in self.x_stabs:
            if error_type in ['Z', 'Y']:
                # Check if stabilizer anticommutes with error
                syndrome.append(stab[error_qubit])
            else:
                syndrome.append(0)

        # Z-stabilizers detect X errors
        for stab in self.z_stabs:
            if error_type in ['X', 'Y']:
                syndrome.append(stab[error_qubit])
            else:
                syndrome.append(0)

        return ''.join(map(str, syndrome))

    def decode_syndrome(self, syndrome):
        """
        Decode syndrome to determine error location and type.

        Returns:
            (error_type, qubit_index) or ('I', None) for no error
        """
        syndrome_bits = [int(b) for b in syndrome]
        x_syndrome = syndrome_bits[0:3]
        z_syndrome = syndrome_bits[3:6]

        # Convert to binary numbers
        x_val = x_syndrome[0] * 4 + x_syndrome[1] * 2 + x_syndrome[2]
        z_val = z_syndrome[0] * 4 + z_syndrome[1] * 2 + z_syndrome[2]

        # Determine error type and location
        if x_val == 0 and z_val == 0:
            return ('I', None)
        elif x_val == 0:
            # Pure X error
            qubit = z_val - 1
            return ('X', qubit)
        elif z_val == 0:
            # Pure Z error
            qubit = x_val - 1
            return ('Z', qubit)
        else:
            # Y error
            if x_val == z_val:
                qubit = x_val - 1
                return ('Y', qubit)
            else:
                # Two separate errors (uncorrectable)
                return ('??', (x_val, z_val))


def demonstrate_steane_code():
    """Demonstrate Steane code encoding and error correction."""

    print("=" * 60)
    print("STEANE [[7,1,3]] CODE DEMONSTRATION")
    print("=" * 60)

    steane = SteaneCode()

    # Print stabilizer generators
    print("\n1. STABILIZER GENERATORS")
    print("-" * 40)
    print("\nX-type stabilizers (detect Z errors):")
    for i, stab in enumerate(steane.x_stabs):
        pauli_str = ''.join(['X' if v else 'I' for v in stab])
        print(f"  g{i+1} = {pauli_str}")

    print("\nZ-type stabilizers (detect X errors):")
    for i, stab in enumerate(steane.z_stabs):
        pauli_str = ''.join(['Z' if v else 'I' for v in stab])
        print(f"  g{i+4} = {pauli_str}")

    # Verify commutation
    print("\n2. COMMUTATION VERIFICATION")
    print("-" * 40)
    print("\nAll X and Z stabilizers must commute.")
    print("Checking overlap parities (must be even):\n")

    all_commute = True
    for i, x_stab in enumerate(steane.x_stabs):
        for j, z_stab in enumerate(steane.z_stabs):
            overlap = sum(a & b for a, b in zip(x_stab, z_stab))
            commutes = overlap % 2 == 0
            all_commute &= commutes
            status = "✓" if commutes else "✗"
            print(f"  g{i+1} ∩ g{j+4}: overlap = {overlap} ({status})")

    print(f"\nAll stabilizers commute: {all_commute}")

    # Syndrome table
    print("\n3. SYNDROME TABLE")
    print("-" * 40)
    print("\nError    | Syndrome | Decoded")
    print("-" * 35)

    for error_type in ['I', 'X', 'Z', 'Y']:
        if error_type == 'I':
            syndrome = steane.calculate_syndrome('I', None)
            decoded = steane.decode_syndrome(syndrome)
            print(f"  {error_type:6s} | {syndrome}  | {decoded}")
        else:
            for qubit in range(7):
                syndrome = steane.calculate_syndrome(error_type, qubit)
                decoded = steane.decode_syndrome(syndrome)
                print(f"  {error_type}{qubit+1}     | {syndrome}  | {decoded}")

    # Encoding circuit
    print("\n4. ENCODING CIRCUIT")
    print("-" * 40)

    enc_circuit = steane.create_encoding_circuit()
    print(enc_circuit.draw(output='text'))

    # Simulate encoding
    print("\n5. ENCODING SIMULATION")
    print("-" * 40)

    # Encode |0⟩
    qc = QuantumCircuit(7)
    qc = qc.compose(enc_circuit)

    simulator = AerSimulator(method='statevector')
    qc.save_statevector()
    result = simulator.run(qc).result()
    sv = result.get_statevector()

    print("\nLogical |0⟩ encoded state:")
    print("Non-zero amplitudes:")
    for i, amp in enumerate(sv):
        if abs(amp) > 1e-10:
            basis = format(i, '07b')
            print(f"  |{basis}⟩: {amp:.4f}")

    # Encode |1⟩
    qc1 = QuantumCircuit(7)
    qc1.x(0)  # Start with |1⟩
    qc1 = qc1.compose(enc_circuit)
    qc1.save_statevector()

    result1 = simulator.run(qc1).result()
    sv1 = result1.get_statevector()

    print("\nLogical |1⟩ encoded state:")
    print("Non-zero amplitudes:")
    for i, amp in enumerate(sv1):
        if abs(amp) > 1e-10:
            basis = format(i, '07b')
            print(f"  |{basis}⟩: {amp:.4f}")


def plot_syndrome_structure():
    """Visualize the syndrome structure of the Steane code."""

    steane = SteaneCode()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Stabilizer support
    ax1 = axes[0]
    support_matrix = np.zeros((6, 7))

    for i, stab in enumerate(steane.x_stabs):
        for j, val in enumerate(stab):
            support_matrix[i, j] = val

    for i, stab in enumerate(steane.z_stabs):
        for j, val in enumerate(stab):
            support_matrix[i + 3, j] = val * 0.5  # Different shade

    im1 = ax1.imshow(support_matrix, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(7))
    ax1.set_xticklabels([f'q{i+1}' for i in range(7)])
    ax1.set_yticks(range(6))
    ax1.set_yticklabels([f'g{i+1}' for i in range(6)])
    ax1.set_xlabel('Physical Qubit')
    ax1.set_ylabel('Stabilizer Generator')
    ax1.set_title('Stabilizer Support Pattern')

    # Plot 2: Syndrome space for X errors
    ax2 = axes[1]
    x_syndromes = []
    for qubit in range(7):
        syn = steane.calculate_syndrome('X', qubit)
        x_syndromes.append([int(b) for b in syn])

    x_syn_matrix = np.array(x_syndromes)
    im2 = ax2.imshow(x_syn_matrix, cmap='Reds', aspect='auto')
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(['X₁', 'X₂', 'X₃', 'Z₁', 'Z₂', 'Z₃'])
    ax2.set_yticks(range(7))
    ax2.set_yticklabels([f'X{i+1}' for i in range(7)])
    ax2.set_xlabel('Syndrome Bit')
    ax2.set_ylabel('X Error')
    ax2.set_title('X Error Syndromes')

    # Plot 3: Syndrome space for Z errors
    ax3 = axes[2]
    z_syndromes = []
    for qubit in range(7):
        syn = steane.calculate_syndrome('Z', qubit)
        z_syndromes.append([int(b) for b in syn])

    z_syn_matrix = np.array(z_syndromes)
    im3 = ax3.imshow(z_syn_matrix, cmap='Greens', aspect='auto')
    ax3.set_xticks(range(6))
    ax3.set_xticklabels(['X₁', 'X₂', 'X₃', 'Z₁', 'Z₂', 'Z₃'])
    ax3.set_yticks(range(7))
    ax3.set_yticklabels([f'Z{i+1}' for i in range(7)])
    ax3.set_xlabel('Syndrome Bit')
    ax3.set_ylabel('Z Error')
    ax3.set_title('Z Error Syndromes')

    plt.tight_layout()
    plt.savefig('steane_code_structure.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: steane_code_structure.png")


def verify_hamming_self_orthogonality():
    """Verify the self-orthogonality of the Hamming code."""

    print("\n" + "=" * 60)
    print("HAMMING CODE SELF-ORTHOGONALITY")
    print("=" * 60)

    steane = SteaneCode()
    codewords = np.array(steane.hamming_codewords)

    print("\n8 Hamming codewords:")
    for i, cw in enumerate(codewords):
        weight = sum(cw)
        print(f"  c{i}: {''.join(map(str, cw))}  (weight {weight})")

    print("\nInner products (mod 2):")
    print("Should all be 0 for self-orthogonality.\n")

    print("    ", end="")
    for j in range(8):
        print(f"c{j} ", end="")
    print()

    all_orthogonal = True
    for i in range(8):
        print(f"c{i}  ", end="")
        for j in range(8):
            inner = np.dot(codewords[i], codewords[j]) % 2
            print(f" {inner} ", end="")
            if inner != 0:
                all_orthogonal = False
        print()

    print(f"\nAll codewords orthogonal: {all_orthogonal}")
    print("This guarantees X and Z stabilizers commute!")


if __name__ == "__main__":
    demonstrate_steane_code()
    verify_hamming_self_orthogonality()

    print("\n" + "=" * 60)
    print("Generating visualization...")
    print("=" * 60)
    plot_syndrome_structure()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Code parameters | $[[7, 1, 3]]$ |
| X-stabilizers | $g_1 = IIIXXXX, g_2 = IXXIIXX, g_3 = XIXIXIX$ |
| Z-stabilizers | $g_4 = IIIZZZZ, g_5 = IZZIIZZ, g_6 = ZIZIZIZ$ |
| Logical X | $\bar{X} = X^{\otimes 7}$ or weight-3 equivalents |
| Logical Z | $\bar{Z} = Z^{\otimes 7}$ or weight-3 equivalents |
| Code distance | $d = 3$ (corrects 1 error) |
| Encoding rate | $k/n = 1/7 \approx 14.3\%$ |

### Main Takeaways

1. **Hamming Foundation:** The Steane code elegantly lifts the classical [7,4,3] Hamming code to the quantum domain
2. **Self-Orthogonality:** The key enabling property is that Hamming codewords are orthogonal mod 2
3. **Efficiency:** 7 qubits vs. 9 for the Shor code, with better transversal gate support
4. **Syndrome Structure:** Clean separation between X and Z syndrome bits
5. **CSS Property:** X and Z errors are independently correctable

---

## Daily Checklist

- [ ] Can write all 6 Steane code stabilizer generators
- [ ] Understand the classical Hamming code construction
- [ ] Can verify stabilizer commutation using overlap parity
- [ ] Know the logical $\bar{X}$ and $\bar{Z}$ operators
- [ ] Can calculate syndromes for arbitrary single-qubit errors
- [ ] Understand why self-orthogonality enables quantum codes
- [ ] Can compare Steane and Shor codes

---

## Preview: Day 692

Tomorrow we'll study **CSS Code Construction** — the general framework for building quantum codes from pairs of classical linear codes. We'll learn:

- Calderbank-Shor-Steane construction theorem
- Dual-containing code requirements
- Generalized CSS codes beyond Steane
- Connection to homological algebra

The CSS framework will reveal why the Steane code is just one member of a powerful family of quantum error correcting codes.

---

*"The Steane code is the quantum computer's workhorse — small enough to implement, powerful enough to protect."*
— Andrew Steane (paraphrased)
