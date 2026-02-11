# Day 704: Clifford Circuits and Classical Simulation

## Overview

**Date:** Day 704 of 1008
**Week:** 101 (Advanced Stabilizer Theory)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Efficient Classical Simulation of Stabilizer Circuits

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Stabilizer states and Gottesman-Knill theorem |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Update rules and measurement simulation |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define stabilizer states** and their representation via stabilizer generators
2. **State the Gottesman-Knill theorem** and its implications for classical simulation
3. **Derive update rules** for Clifford gates acting on stabilizer tableaux
4. **Simulate Pauli measurements** classically using stabilizer formalism
5. **Analyze computational complexity** of stabilizer simulation: $O(n^2)$ per gate, $O(n^3)$ per measurement
6. **Identify limitations** - when does classical simulation fail?

---

## Core Content

### 1. Stabilizer States

#### Definition

An $n$-qubit **stabilizer state** $|\psi\rangle$ is uniquely determined by its stabilizer group:

$$\mathcal{S} = \{P \in \mathcal{P}_n : P|\psi\rangle = |\psi\rangle\}$$

where $\mathcal{S}$ is an abelian subgroup of $\mathcal{P}_n$ with $-I \notin \mathcal{S}$.

#### Key Properties

1. **Size:** $|\mathcal{S}| = 2^n$ (including phases)
2. **Generators:** Need only $n$ independent generators $g_1, \ldots, g_n$
3. **Uniqueness:** The generators uniquely specify $|\psi\rangle$

**Examples:**

| State | Stabilizer Generators |
|-------|----------------------|
| $\|0\rangle$ | $Z$ |
| $\|+\rangle$ | $X$ |
| $\|00\rangle$ | $Z_1, Z_2$ |
| Bell state $\|\Phi^+\rangle$ | $X_1 X_2, Z_1 Z_2$ |
| GHZ state | $X_1 X_2 X_3, Z_1 Z_2, Z_2 Z_3$ |

---

### 2. The Gottesman-Knill Theorem

#### Theorem Statement

**Gottesman-Knill Theorem:** A quantum circuit consisting of:
1. **Preparation** in computational basis states $|0\rangle^{\otimes n}$
2. **Clifford gates** (H, S, CNOT)
3. **Pauli measurements**
4. **Classical control** based on measurement outcomes

can be efficiently simulated on a classical computer in time $O(n^2)$ per gate and $O(n^3)$ per measurement.

#### Key Insight

Instead of tracking $2^n$ amplitudes, track:
- $n$ stabilizer generators (each an $n$-qubit Pauli)
- Total classical information: $O(n^2)$ bits

#### What This Does NOT Mean

The theorem does **not** imply:
- Quantum computers are useless (non-Clifford gates break classical simulation)
- Entanglement is classically simulable in general (stabilizer states are special)
- Measurement outcomes are deterministic (randomness still present)

---

### 3. Stabilizer Tableau Representation

#### The Tableau

Store stabilizer state as a $(2n+1) \times (2n)$ binary matrix plus phases:

$$\text{Tableau} = \begin{pmatrix}
\mathbf{x}_1 & | & \mathbf{z}_1 & | & r_1 \\
\vdots & | & \vdots & | & \vdots \\
\mathbf{x}_n & | & \mathbf{z}_n & | & r_n \\
\hline
\mathbf{x}_{\bar{1}} & | & \mathbf{z}_{\bar{1}} & | & r_{\bar{1}} \\
\vdots & | & \vdots & | & \vdots \\
\mathbf{x}_{\bar{n}} & | & \mathbf{z}_{\bar{n}} & | & r_{\bar{n}}
\end{pmatrix}$$

**Components:**
- **Rows 1 to $n$:** Stabilizer generators (determines state)
- **Rows $\bar{1}$ to $\bar{n}$:** Destabilizers (auxiliary, for measurement)
- **$\mathbf{x}_i, \mathbf{z}_i$:** Binary vectors encoding Pauli ($X^{\mathbf{x}} Z^{\mathbf{z}}$)
- **$r_i$:** Phase bit (0 for $+1$, 1 for $-1$, ignoring $i$ phase for now)

#### Initial State $|0\rangle^{\otimes n}$

Stabilizers: $Z_1, Z_2, \ldots, Z_n$
Destabilizers: $X_1, X_2, \ldots, X_n$

$$\text{Initial Tableau} = \begin{pmatrix}
I_n & | & 0 & | & \mathbf{0} \\
\hline
0 & | & I_n & | & \mathbf{0}
\end{pmatrix}$$

Wait, let me correct - for $|0\rangle^{\otimes n}$:
- Stabilizers are $Z_i$: $(\mathbf{x}_i = 0, \mathbf{z}_i = e_i)$
- Destabilizers are $X_i$: $(\mathbf{x}_i = e_i, \mathbf{z}_i = 0)$

---

### 4. Update Rules for Clifford Gates

#### Hadamard on Qubit $j$

Swaps $X_j \leftrightarrow Z_j$:

$$H_j: \quad x_{ij} \leftrightarrow z_{ij} \quad \text{for all rows } i$$

Phase update: $r_i \gets r_i \oplus (x_{ij} \cdot z_{ij})$

**Complexity:** $O(n)$ operations.

#### Phase Gate $S$ on Qubit $j$

Maps $X_j \to Y_j = iXZ$:

$$S_j: \quad z_{ij} \gets z_{ij} \oplus x_{ij}$$

Phase update: $r_i \gets r_i \oplus (x_{ij} \cdot z_{ij})$

**Complexity:** $O(n)$ operations.

#### CNOT with Control $c$, Target $t$

$$\text{CNOT}_{ct}: \begin{cases}
x_{it} \gets x_{it} \oplus x_{ic} \\
z_{ic} \gets z_{ic} \oplus z_{it}
\end{cases}$$

Phase update: $r_i \gets r_i \oplus x_{ic} z_{it} (x_{it} \oplus z_{ic} \oplus 1)$

**Complexity:** $O(n)$ operations.

#### Summary of Update Rules

| Gate | X-part Update | Z-part Update | Phase Rule |
|------|---------------|---------------|------------|
| $H_j$ | $x_{ij} \leftrightarrow z_{ij}$ | (swap) | $r_i \oplus= x \cdot z$ |
| $S_j$ | unchanged | $z_{ij} \oplus= x_{ij}$ | $r_i \oplus= x \cdot z$ |
| $\text{CNOT}_{ct}$ | $x_{it} \oplus= x_{ic}$ | $z_{ic} \oplus= z_{it}$ | (complex) |

---

### 5. Simulating Pauli Measurements

#### Measurement of Single Pauli $P$

**Goal:** Measure Pauli operator $P$ on stabilizer state $|\psi\rangle$.

**Algorithm:**

1. **Check if $P$ commutes with all stabilizers:**
   - If $P$ anticommutes with no generator: $P$ or $-P$ is in stabilizer group
   - Outcome is deterministic ($+1$ or $-1$)

2. **If $P$ anticommutes with some generator $g_k$:**
   - Outcome is random (50/50)
   - State collapses: replace $g_k$ with $\pm P$ (depending on outcome)
   - Update other generators to maintain commutativity

**Detailed Procedure for Random Case:**

Let $P$ anticommute with generator $g_p$ (the first such one).

1. For all other generators $g_i$ that anticommute with $P$:
   - Replace $g_i \gets g_i \cdot g_p$ (now commutes with $P$)

2. Replace $g_p \gets \pm P$ (sign from measurement outcome)

3. Update destabilizers similarly

**Complexity:** $O(n^2)$ to check commutation, $O(n^2)$ to update generators = $O(n^2)$ total, or $O(n^3)$ if Gaussian elimination needed.

---

### 6. Full Simulation Algorithm

#### Aaronson-Gottesman Algorithm (CHP Simulator)

```
Algorithm: Simulate Stabilizer Circuit
Input: Circuit with Clifford gates and Pauli measurements
Output: Measurement outcomes

1. Initialize tableau for |0⟩^⊗n

2. For each operation in circuit:
   a. If Clifford gate:
      - Apply appropriate update rule to tableau
      - Complexity: O(n) per gate

   b. If Pauli measurement:
      - Find generators anticommuting with measured Pauli
      - If none: outcome deterministic (read from tableau)
      - If some: outcome random, update tableau
      - Complexity: O(n²) per measurement

3. Return measurement outcomes
```

**Total Complexity:**
- $m$ Clifford gates: $O(mn)$
- $k$ measurements: $O(kn^2)$
- Overall: $O(mn + kn^2)$

---

### 7. Why Clifford Simulation Works

#### The Heisenberg Picture

Clifford gates transform Pauli operators to Pauli operators:

$$C P C^\dagger = P'$$

Tracking evolution of $n$ generators is equivalent to tracking the full quantum state!

#### Stabilizer State Dimension

Stabilizer states form a **finite set** (not a continuous manifold):

$$\# \text{stabilizer states} = 2^n \prod_{k=1}^{n}(2^k + 1)$$

This is exponential in $n$, but still efficiently enumerable.

For $n=1$: 6 states (eigenstates of $X, Y, Z$)
For $n=2$: 60 states
For $n=3$: 1080 states

---

### 8. Breaking Efficient Simulation

#### T Gate (Non-Clifford)

The T gate $T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$:

$$T X T^\dagger = \frac{1}{\sqrt{2}}(X + Y)$$

This is **not a Pauli operator** - cannot be represented in stabilizer tableau!

#### Magic States

Non-stabilizer states (created by T gates):

$$|T\rangle = T|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

#### Universal Computation

Clifford + T = Universal. The T gate breaks classical simulation, enabling quantum advantage.

---

## Worked Examples

### Example 1: Simulate Creating a Bell State

**Problem:** Track the stabilizer tableau through $H_1 \cdot \text{CNOT}_{12}$ on $|00\rangle$.

**Solution:**

**Initial state $|00\rangle$:**

| | $x_1$ | $x_2$ | $z_1$ | $z_2$ | $r$ |
|---|---|---|---|---|---|
| Stab 1 ($Z_1$) | 0 | 0 | 1 | 0 | 0 |
| Stab 2 ($Z_2$) | 0 | 0 | 0 | 1 | 0 |

**After $H_1$ (swap $x_1 \leftrightarrow z_1$):**

| | $x_1$ | $x_2$ | $z_1$ | $z_2$ | $r$ |
|---|---|---|---|---|---|
| Stab 1 ($X_1$) | 1 | 0 | 0 | 0 | 0 |
| Stab 2 ($Z_2$) | 0 | 0 | 0 | 1 | 0 |

**After $\text{CNOT}_{12}$:**
- $x_{i2} \gets x_{i2} \oplus x_{i1}$
- $z_{i1} \gets z_{i1} \oplus z_{i2}$

| | $x_1$ | $x_2$ | $z_1$ | $z_2$ | $r$ |
|---|---|---|---|---|---|
| Stab 1 | 1 | 1 | 0 | 0 | 0 |
| Stab 2 | 0 | 0 | 1 | 1 | 0 |

Stabilizers: $X_1 X_2$, $Z_1 Z_2$ ✓

This is the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.

---

### Example 2: Simulate Z Measurement on Bell State

**Problem:** Measure $Z_1$ on the Bell state from Example 1.

**Solution:**

**Check commutation with $Z_1 = (0,0|1,0)$:**

- Stab 1 ($X_1 X_2$): $\langle (1,1|0,0), (0,0|1,0) \rangle_s = 1 \cdot 1 + 1 \cdot 0 = 1$ → anticommutes
- Stab 2 ($Z_1 Z_2$): $\langle (0,0|1,1), (0,0|1,0) \rangle_s = 0$ → commutes

**Since $Z_1$ anticommutes with Stab 1:**
- Measurement outcome is **random** (50/50)
- Suppose we measure $+1$

**Update tableau:**
- Replace Stab 1 with $+Z_1$

| | $x_1$ | $x_2$ | $z_1$ | $z_2$ | $r$ |
|---|---|---|---|---|---|
| Stab 1 ($+Z_1$) | 0 | 0 | 1 | 0 | 0 |
| Stab 2 ($Z_1 Z_2$) | 0 | 0 | 1 | 1 | 0 |

**Interpret:** Now stabilized by $Z_1$ (eigenvalue $+1$) and $Z_1 Z_2$.

From $Z_1 = +1$ and $Z_1 Z_2 = +1$, we get $Z_2 = +1$.

Post-measurement state: $|00\rangle$

If we had measured $-1$: state would be $|11\rangle$.

---

### Example 3: Deterministic Measurement

**Problem:** Measure $Z_1 Z_2$ on the Bell state.

**Solution:**

**Check commutation with $Z_1 Z_2 = (0,0|1,1)$:**

- Stab 1 ($X_1 X_2$): $\langle (1,1|0,0), (0,0|1,1) \rangle_s = 1 \cdot 1 + 1 \cdot 1 = 0$ → commutes
- Stab 2 ($Z_1 Z_2$): $\langle (0,0|1,1), (0,0|1,1) \rangle_s = 0$ → commutes

**All generators commute with $Z_1 Z_2$.**

Check if $Z_1 Z_2 \in \mathcal{S}$: Yes, it's exactly Stab 2!

**Outcome:** Deterministic $+1$ (since phase $r_2 = 0$)

**State unchanged** (no projection needed).

---

## Practice Problems

### Direct Application

1. **Problem 1:** Write the initial tableau for $|+\rangle^{\otimes 3}$ (3 qubits, all in $|+\rangle$).

2. **Problem 2:** Apply $S_2$ to the tableau for $|++\rangle$ and read off the new stabilizers.

3. **Problem 3:** Given stabilizers $X_1 X_2, Z_1 Z_2$, determine if measuring $X_1$ gives a deterministic or random outcome.

### Intermediate

4. **Problem 4:** Simulate the circuit $H_1 \cdot H_2 \cdot CZ_{12}$ starting from $|00\rangle$ using tableau updates.

5. **Problem 5:** Show that the number of gates to create an arbitrary $n$-qubit stabilizer state is $O(n^2/\log n)$.

6. **Problem 6:** Prove that if $P$ anticommutes with exactly $k$ stabilizer generators, we can always reduce to $k=1$ by generator multiplication.

### Challenging

7. **Problem 7:** Implement a full CHP-style simulator and verify it reproduces known results for GHZ state creation and measurement.

8. **Problem 8:** Extend the tableau to track phases correctly for $Y$ measurements (requires tracking $i$ phases).

9. **Problem 9:** Prove that simulating $\text{poly}(n)$ T gates with stabilizer circuits requires exponential classical resources.

---

## Computational Lab

```python
"""
Day 704: Clifford Circuits and Classical Simulation
Week 101: Advanced Stabilizer Theory

Implements the Aaronson-Gottesman (CHP) stabilizer simulator.
"""

import numpy as np
from typing import Tuple, List, Optional
import random

class StabilizerTableau:
    """
    Stabilizer tableau for efficient classical simulation.

    Based on Aaronson & Gottesman "Improved Simulation of Stabilizer Circuits"
    Implements the CHP (CNOT-Hadamard-Phase) simulator.
    """

    def __init__(self, n_qubits: int):
        """Initialize tableau for |0⟩^⊗n."""
        self.n = n_qubits

        # Tableau: 2n rows (n stabilizers + n destabilizers)
        # Each row: [x1...xn | z1...zn | r]
        # x, z are n-bit vectors, r is phase (0 or 1)

        # Initialize for |0⟩^⊗n
        # Destabilizers (rows 0 to n-1): X_i
        # Stabilizers (rows n to 2n-1): Z_i

        self.x = np.zeros((2*n_qubits, n_qubits), dtype=int)
        self.z = np.zeros((2*n_qubits, n_qubits), dtype=int)
        self.r = np.zeros(2*n_qubits, dtype=int)

        for i in range(n_qubits):
            # Destabilizer i = X_i
            self.x[i, i] = 1

            # Stabilizer i = Z_i
            self.z[n_qubits + i, i] = 1

    def _rowmult(self, h: int, i: int):
        """
        Multiply row h by row i (row h = row h * row i).
        Updates phase according to Pauli multiplication rules.
        """
        # Phase calculation for Pauli multiplication
        # Uses the formula from Aaronson-Gottesman
        phase = 0
        for j in range(self.n):
            # g function: phase exponent from multiplying P_h and P_i at qubit j
            xi, zi = self.x[i, j], self.z[i, j]
            xh, zh = self.x[h, j], self.z[h, j]

            if xi == 1 and zi == 1:  # Y
                phase += zh - xh
            elif xi == 1 and zi == 0:  # X
                phase += zh * (2 * xh - 1)
            elif xi == 0 and zi == 1:  # Z
                phase += xh * (1 - 2 * zh)
            # else: I, no phase contribution

        # Update row h
        self.x[h] = (self.x[h] + self.x[i]) % 2
        self.z[h] = (self.z[h] + self.z[i]) % 2
        self.r[h] = (self.r[h] + self.r[i] + (phase % 4) // 2) % 2

    def hadamard(self, qubit: int):
        """Apply Hadamard gate to specified qubit."""
        for i in range(2 * self.n):
            # Phase update: r ^= x AND z
            self.r[i] = (self.r[i] + self.x[i, qubit] * self.z[i, qubit]) % 2
            # Swap x and z
            self.x[i, qubit], self.z[i, qubit] = self.z[i, qubit], self.x[i, qubit]

    def phase_gate(self, qubit: int):
        """Apply S (phase) gate to specified qubit."""
        for i in range(2 * self.n):
            # Phase update: r ^= x AND z
            self.r[i] = (self.r[i] + self.x[i, qubit] * self.z[i, qubit]) % 2
            # z ^= x
            self.z[i, qubit] = (self.z[i, qubit] + self.x[i, qubit]) % 2

    def cnot(self, control: int, target: int):
        """Apply CNOT with specified control and target."""
        for i in range(2 * self.n):
            # Phase update (complex formula)
            self.r[i] = (self.r[i] +
                         self.x[i, control] * self.z[i, target] *
                         (self.x[i, target] ^ self.z[i, control] ^ 1)) % 2
            # x_target ^= x_control
            self.x[i, target] = (self.x[i, target] + self.x[i, control]) % 2
            # z_control ^= z_target
            self.z[i, control] = (self.z[i, control] + self.z[i, target]) % 2

    def measure(self, qubit: int, basis: str = 'Z') -> int:
        """
        Measure specified qubit in given basis.

        Args:
            qubit: Which qubit to measure
            basis: 'X', 'Y', or 'Z'

        Returns:
            Measurement outcome: 0 or 1
        """
        # Transform to Z-basis measurement
        if basis == 'X':
            self.hadamard(qubit)
        elif basis == 'Y':
            self.phase_gate(qubit)
            self.phase_gate(qubit)
            self.phase_gate(qubit)  # S^† = S^3
            self.hadamard(qubit)

        # Now measure in Z basis
        outcome = self._measure_z(qubit)

        # Transform back
        if basis == 'X':
            self.hadamard(qubit)
        elif basis == 'Y':
            self.hadamard(qubit)
            self.phase_gate(qubit)

        return outcome

    def _measure_z(self, qubit: int) -> int:
        """Measure qubit in computational (Z) basis."""
        n = self.n

        # Find a stabilizer generator that anticommutes with Z_qubit
        # Z_qubit anticommutes with Pauli iff x[qubit] = 1
        p = None
        for i in range(n, 2*n):  # Search stabilizers only
            if self.x[i, qubit] == 1:
                p = i
                break

        if p is None:
            # Deterministic measurement
            # The eigenvalue is in the destabilizers
            # Need to compute from stabilizer group

            # Find which destabilizers have x[qubit] = 1
            # and XOR them into a temporary row
            self.x = np.vstack([self.x, np.zeros(n, dtype=int)])
            self.z = np.vstack([self.z, np.zeros(n, dtype=int)])
            self.r = np.append(self.r, 0)

            for i in range(n):
                if self.x[i, qubit] == 1:
                    self._rowmult(2*n, i + n)

            # Outcome determined by phase of accumulated row
            outcome = self.r[2*n]

            # Remove temporary row
            self.x = self.x[:-1]
            self.z = self.z[:-1]
            self.r = self.r[:-1]

            return outcome

        else:
            # Random measurement
            # Row p anticommutes with Z_qubit

            # For all other rows that anticommute with Z_qubit,
            # multiply them by row p to make them commute
            for i in range(2*n):
                if i != p and self.x[i, qubit] == 1:
                    self._rowmult(i, p)

            # Move destabilizer p-n to row p (swap roles)
            # and set row p-n to be Z_qubit
            self.x[p - n] = self.x[p].copy()
            self.z[p - n] = self.z[p].copy()
            self.r[p - n] = self.r[p]

            # Row p becomes Z_qubit
            self.x[p] = np.zeros(n, dtype=int)
            self.z[p] = np.zeros(n, dtype=int)
            self.z[p, qubit] = 1

            # Random outcome
            outcome = random.randint(0, 1)
            self.r[p] = outcome

            return outcome

    def get_stabilizers(self) -> List[str]:
        """Return stabilizer generators as strings."""
        stabilizers = []
        for i in range(self.n, 2*self.n):
            s = self._row_to_string(i)
            stabilizers.append(s)
        return stabilizers

    def _row_to_string(self, row: int) -> str:
        """Convert tableau row to Pauli string."""
        sign = '-' if self.r[row] == 1 else '+'
        paulis = []
        for j in range(self.n):
            x, z = self.x[row, j], self.z[row, j]
            if x == 0 and z == 0:
                paulis.append('I')
            elif x == 1 and z == 0:
                paulis.append('X')
            elif x == 0 and z == 1:
                paulis.append('Z')
            else:
                paulis.append('Y')
        return sign + ''.join(paulis)


def demonstrate_stabilizer_simulation():
    """Demonstrate the stabilizer simulator."""

    print("=" * 70)
    print("CLIFFORD CIRCUIT CLASSICAL SIMULATION")
    print("=" * 70)

    # Example 1: Create Bell state
    print("\n1. CREATING BELL STATE")
    print("-" * 50)

    tab = StabilizerTableau(2)
    print("Initial state |00⟩:")
    print(f"  Stabilizers: {tab.get_stabilizers()}")

    tab.hadamard(0)
    print("After H on qubit 0:")
    print(f"  Stabilizers: {tab.get_stabilizers()}")

    tab.cnot(0, 1)
    print("After CNOT(0,1):")
    print(f"  Stabilizers: {tab.get_stabilizers()}")
    print("  (This is the Bell state |Φ⁺⟩)")

    # Example 2: Measure Bell state
    print("\n2. MEASURING BELL STATE")
    print("-" * 50)

    print("Measuring Z on qubit 0:")
    # Create fresh Bell state
    tab = StabilizerTableau(2)
    tab.hadamard(0)
    tab.cnot(0, 1)

    outcome = tab.measure(0, 'Z')
    print(f"  Outcome: {outcome}")
    print(f"  Post-measurement stabilizers: {tab.get_stabilizers()}")

    # Example 3: GHZ state
    print("\n3. CREATING 3-QUBIT GHZ STATE")
    print("-" * 50)

    tab = StabilizerTableau(3)
    tab.hadamard(0)
    tab.cnot(0, 1)
    tab.cnot(1, 2)
    print("State |GHZ⟩ = (|000⟩ + |111⟩)/√2:")
    print(f"  Stabilizers: {tab.get_stabilizers()}")

    # Example 4: Multiple runs to show randomness
    print("\n4. RANDOMNESS IN MEASUREMENT (100 trials)")
    print("-" * 50)

    counts = {'0': 0, '1': 0}
    for _ in range(100):
        tab = StabilizerTableau(2)
        tab.hadamard(0)
        tab.cnot(0, 1)
        outcome = tab.measure(0, 'Z')
        counts[str(outcome)] += 1

    print(f"  Z measurement on Bell state qubit 0:")
    print(f"    |0⟩: {counts['0']}%, |1⟩: {counts['1']}%")

    # Example 5: Deterministic measurement
    print("\n5. DETERMINISTIC MEASUREMENT")
    print("-" * 50)

    tab = StabilizerTableau(2)
    tab.hadamard(0)
    tab.cnot(0, 1)

    # Measure Z1Z2 indirectly by checking if |00⟩ + |11⟩
    # We use a different approach: measure in a basis that commutes

    print("For Bell state, measuring X on both qubits:")
    print("(Product X₁X₂ is a stabilizer, so X₁⊗X₂ = +1 always)")

    # Fresh state for proper demonstration
    tab = StabilizerTableau(1)
    tab.hadamard(0)
    print("\n  State |+⟩, stabilizer:", tab.get_stabilizers())

    outcomes_x = []
    for _ in range(10):
        tab = StabilizerTableau(1)
        tab.hadamard(0)
        out = tab.measure(0, 'X')
        outcomes_x.append(out)
    print(f"  X measurements: {outcomes_x}")
    print("  (All should be 0, since X|+⟩ = |+⟩)")

    # Example 6: Circuit with S gates
    print("\n6. CIRCUIT WITH PHASE GATES")
    print("-" * 50)

    tab = StabilizerTableau(1)
    print("Starting from |0⟩")
    print(f"  Stabilizers: {tab.get_stabilizers()}")

    tab.hadamard(0)
    print("After H: |+⟩")
    print(f"  Stabilizers: {tab.get_stabilizers()}")

    tab.phase_gate(0)
    print("After S: |+i⟩ = (|0⟩ + i|1⟩)/√2")
    print(f"  Stabilizers: {tab.get_stabilizers()}")

    # Y eigenstate
    tab.phase_gate(0)
    print("After another S: |−⟩")
    print(f"  Stabilizers: {tab.get_stabilizers()}")

    # Example 7: Complexity demonstration
    print("\n7. SCALING DEMONSTRATION")
    print("-" * 50)

    import time

    for n in [10, 50, 100, 200]:
        tab = StabilizerTableau(n)

        start = time.time()
        # Apply random Clifford circuit
        for _ in range(n * 10):
            gate = random.choice(['H', 'S', 'CNOT'])
            if gate == 'H':
                tab.hadamard(random.randint(0, n-1))
            elif gate == 'S':
                tab.phase_gate(random.randint(0, n-1))
            else:
                c, t = random.sample(range(n), 2)
                tab.cnot(c, t)

        # Do some measurements
        for _ in range(n):
            tab.measure(random.randint(0, n-1), 'Z')

        elapsed = time.time() - start
        print(f"  n={n:3d} qubits, {n*10} gates, {n} measurements: {elapsed:.3f}s")

    print("\n  (Note: Classical simulation scales polynomially!)")


if __name__ == "__main__":
    demonstrate_stabilizer_simulation()
```

**Expected Output:**
```
======================================================================
CLIFFORD CIRCUIT CLASSICAL SIMULATION
======================================================================

1. CREATING BELL STATE
--------------------------------------------------
Initial state |00⟩:
  Stabilizers: ['+ZI', '+IZ']
After H on qubit 0:
  Stabilizers: ['+XI', '+IZ']
After CNOT(0,1):
  Stabilizers: ['+XX', '+ZZ']
  (This is the Bell state |Φ⁺⟩)

2. MEASURING BELL STATE
--------------------------------------------------
Measuring Z on qubit 0:
  Outcome: 0
  Post-measurement stabilizers: ['+ZI', '+ZZ']

3. CREATING 3-QUBIT GHZ STATE
--------------------------------------------------
State |GHZ⟩ = (|000⟩ + |111⟩)/√2:
  Stabilizers: ['+XXX', '+ZZI', '+IZZ']

4. RANDOMNESS IN MEASUREMENT (100 trials)
--------------------------------------------------
  Z measurement on Bell state qubit 0:
    |0⟩: ~50%, |1⟩: ~50%

5. DETERMINISTIC MEASUREMENT
--------------------------------------------------
...

6. CIRCUIT WITH PHASE GATES
--------------------------------------------------
Starting from |0⟩
  Stabilizers: ['+Z']
After H: |+⟩
  Stabilizers: ['+X']
After S: |+i⟩ = (|0⟩ + i|1⟩)/√2
  Stabilizers: ['+Y']
After another S: |−⟩
  Stabilizers: ['-X']

7. SCALING DEMONSTRATION
--------------------------------------------------
  n= 10 qubits, 100 gates, 10 measurements: ~0.01s
  n= 50 qubits, 500 gates, 50 measurements: ~0.1s
  n=100 qubits, 1000 gates, 100 measurements: ~0.5s
  n=200 qubits, 2000 gates, 200 measurements: ~2s
```

---

## Summary

### Key Formulas

| Concept | Formula/Rule |
|---------|--------------|
| **Stabilizer state** | $\|\psi\rangle$ uniquely determined by $n$ generators |
| **Gottesman-Knill** | Clifford circuits classically simulable in $O(\text{poly}(n))$ |
| **Hadamard update** | $x_j \leftrightarrow z_j$, $r \oplus= x_j \cdot z_j$ |
| **Phase update** | $z_j \oplus= x_j$, $r \oplus= x_j \cdot z_j$ |
| **CNOT update** | $x_t \oplus= x_c$, $z_c \oplus= z_t$ |
| **Measurement** | Check commutation, update if random |
| **Complexity** | $O(n)$ per gate, $O(n^2)$ per measurement |

### Main Takeaways

1. **Stabilizer tableaux** compress $2^n$ amplitudes to $O(n^2)$ bits
2. **Clifford gates** update tableau in $O(n)$ time via simple rules
3. **Measurements** may be deterministic or random; $O(n^2)$ to simulate
4. **Gottesman-Knill theorem** enables efficient classical simulation
5. **Non-Clifford gates (T)** break efficient simulation—source of quantum advantage

---

## Daily Checklist

- [ ] Define stabilizer states via stabilizer generators
- [ ] State and understand Gottesman-Knill theorem
- [ ] Apply update rules for H, S, CNOT gates
- [ ] Simulate Pauli measurements on stabilizer states
- [ ] Implement a working stabilizer simulator
- [ ] Understand why T gates break classical simulation

---

## Preview: Day 705

Tomorrow we study **Stabilizer Tableaux and the Aaronson-Gottesman Algorithm** in more depth, covering:
- Optimized tableau representations (CHP vs. graph states)
- Canonical forms for stabilizer states
- The `stim` library for high-performance simulation
- Applications to quantum error correction decoding
