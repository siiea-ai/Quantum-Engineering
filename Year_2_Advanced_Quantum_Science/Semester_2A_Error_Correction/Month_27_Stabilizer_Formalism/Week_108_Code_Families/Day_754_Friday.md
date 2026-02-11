# Day 754: Gottesman-Knill Theorem

## Overview

**Day:** 754 of 1008
**Week:** 108 (Code Families & Construction Techniques)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Efficient Classical Simulation of Stabilizer Circuits

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Theorem statement and proof |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Tableau simulation |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **State** the Gottesman-Knill theorem precisely
2. **Explain** why Clifford circuits are classically simulable
3. **Implement** stabilizer tableau representation
4. **Perform** update rules for Clifford gates
5. **Understand** implications for quantum advantage
6. **Identify** the boundary between classical and quantum

---

## The Gottesman-Knill Theorem

### Statement

**Theorem (Gottesman 1998, Knill 2001):**

The following can be efficiently simulated on a classical computer:
1. Preparation of computational basis states |0⟩^⊗n
2. Clifford gates (H, S, CNOT)
3. Measurements in the computational basis
4. Classical feed-forward based on measurement outcomes

**Complexity:** O(n²) per gate, O(n³) for measurement.

### Significance

**Not universal:** Clifford circuits alone cannot provide quantum speedup.

**Boundary:** Need non-Clifford resources (T gates, magic states) for quantum advantage.

### What is NOT Simulable

- Arbitrary single-qubit rotations
- T gates (or any non-Clifford gate)
- General multi-qubit entangling gates

---

## Stabilizer Formalism Foundation

### Stabilizer States

A state |ψ⟩ is a **stabilizer state** if:
$$S|ψ⟩ = |ψ⟩ \text{ for all } S \in \mathcal{S}$$

where S is an abelian group of n-qubit Pauli operators with |S| = 2^n.

### Classical Description

An n-qubit stabilizer state is specified by n independent generators:
$$\mathcal{S} = \langle g_1, g_2, \ldots, g_n \rangle$$

Each generator is a Pauli string: O(n) bits per generator.

**Total:** O(n²) bits to describe n-qubit state.

### Why Efficient

| Quantum State | Classical Description |
|---------------|----------------------|
| General | 2^n complex amplitudes |
| Stabilizer | n² bits (generators) |

Exponential compression for stabilizer states!

---

## The Stabilizer Tableau

### Representation

The **tableau** stores:
- n generators of stabilizer group (rows 1 to n)
- n destabilizers (rows n+1 to 2n)
- Phase information

**Matrix form:**
$$\text{Tableau} = \begin{pmatrix}
\mathbf{x}_1 & \mathbf{z}_1 & r_1 \\
\vdots & \vdots & \vdots \\
\mathbf{x}_n & \mathbf{z}_n & r_n \\
\mathbf{x}_{n+1} & \mathbf{z}_{n+1} & r_{n+1} \\
\vdots & \vdots & \vdots \\
\mathbf{x}_{2n} & \mathbf{z}_{2n} & r_{2n}
\end{pmatrix}$$

where:
- $\mathbf{x}_i, \mathbf{z}_i \in \{0,1\}^n$
- $r_i \in \{0,1\}$ (phase: 0 for +1, 1 for -1)

### Initial State |0⟩^⊗n

Stabilizers: $Z_1, Z_2, \ldots, Z_n$
Destabilizers: $X_1, X_2, \ldots, X_n$

**Initial tableau:**
$$\begin{pmatrix}
I_n & 0 & 0 \\
0 & I_n & 0
\end{pmatrix}$$

(X part | Z part | phase)

---

## Gate Update Rules

### Hadamard Gate

H on qubit j swaps X ↔ Z:
- $X_j \to Z_j$
- $Z_j \to X_j$
- Phase update: $r_i \to r_i \oplus x_{ij} z_{ij}$

**Tableau update:** Swap columns j and n+j, update phases.

### Phase Gate (S)

S on qubit j: $X_j \to Y_j = iX_jZ_j$
- $X_j \to X_j Z_j$
- $Z_j \to Z_j$
- Phase: $r_i \to r_i \oplus x_{ij} z_{ij}$

**Tableau update:** $z_{ij} \to z_{ij} \oplus x_{ij}$, update phase.

### CNOT Gate

CNOT from control c to target t:
- $X_c \to X_c X_t$
- $Z_t \to Z_c Z_t$
- $X_t \to X_t$
- $Z_c \to Z_c$

**Tableau update:**
- $x_{it} \to x_{it} \oplus x_{ic}$ (X propagates forward)
- $z_{ic} \to z_{ic} \oplus z_{it}$ (Z propagates backward)
- Phase: $r_i \to r_i \oplus x_{ic} z_{it} (x_{it} \oplus z_{ic} \oplus 1)$

### Complexity

Each gate update: O(n) operations.

Total for m gates: O(mn).

---

## Measurement Simulation

### Z-basis Measurement

To measure qubit j in Z basis:

**Case 1:** No stabilizer has X_j = 1.
- Outcome is deterministic
- Find destabilizer with X_j = 1
- Outcome = 0 if commutes, 1 if anticommutes

**Case 2:** Some stabilizer g_i has X_j = 1.
- Outcome is random (50/50)
- Update tableau to reflect measurement outcome

### Algorithm

1. Check if any stabilizer anticommutes with Z_j
2. If no: deterministic outcome, compute from destabilizers
3. If yes: random outcome, update stabilizer group

**Complexity:** O(n²) per measurement.

---

## Proof Sketch

### Why It Works

1. **Closure:** Clifford gates map Pauli operators to Pauli operators
2. **Polynomial representation:** n stabilizer generators, each O(n) bits
3. **Efficient update:** Each gate transforms generators in O(n) time
4. **Measurement:** Finding anticommuting stabilizer is O(n²)

### Key Insight

Stabilizer states form a discrete set (not continuous).

Clifford gates permute this set.

Classical computer can track the permutation efficiently.

---

## Implications

### Not Universal

**Theorem:** Clifford + computational basis measurements cannot implement arbitrary quantum computations.

**Proof:** Classically simulable ≠ universal quantum.

### Quantum Advantage Requires

To achieve quantum speedup:
- T gates (or other non-Clifford)
- Magic states
- Non-Pauli measurements

### The Clifford Hierarchy

**Level 1:** Pauli group
**Level 2:** Clifford group (normalizer of Paulis)
**Level 3:** Gates that map Cliffords to Cliffords
- T gate is level 3

Higher levels are progressively harder to simulate.

---

## Worked Examples

### Example 1: Bell State Preparation

**Circuit:** H on qubit 1, CNOT from 1 to 2.

**Initial:** $\mathcal{S} = \langle Z_1, Z_2 \rangle$

**After H on qubit 1:**
- $Z_1 \to X_1$
- $\mathcal{S} = \langle X_1, Z_2 \rangle$

**After CNOT(1→2):**
- $X_1 \to X_1 X_2$
- $Z_2 \to Z_1 Z_2$
- $\mathcal{S} = \langle X_1 X_2, Z_1 Z_2 \rangle$

**Final state:** Bell state $\frac{1}{\sqrt{2}}(|00⟩ + |11⟩)$ ✓

### Example 2: GHZ State

**Circuit:** H on qubit 1, CNOT(1→2), CNOT(1→3).

**After H:** $\mathcal{S} = \langle X_1, Z_2, Z_3 \rangle$

**After CNOT(1→2):** $\mathcal{S} = \langle X_1 X_2, Z_1 Z_2, Z_3 \rangle$

**After CNOT(1→3):** $\mathcal{S} = \langle X_1 X_2 X_3, Z_1 Z_2, Z_1 Z_3 \rangle$

**GHZ stabilizers:** $\{X_1X_2X_3, Z_1Z_2, Z_1Z_3\}$ ✓

### Example 3: Measurement Outcome

**State:** Bell state with $\mathcal{S} = \langle X_1 X_2, Z_1 Z_2 \rangle$

**Measure qubit 1 in Z basis:**
- Check: Does any stabilizer anticommute with Z_1?
- $X_1 X_2$ anticommutes with $Z_1$ (X_1 part)
- Outcome is random!

**If outcome = 0:**
- New stabilizer: $Z_1$
- Other stabilizer: $Z_1 Z_2 \cdot X_1 X_2 = ...$ (eliminate X_1)

---

## Practice Problems

### Level 1: Direct Application

**P1.1** Apply H to qubit 2 of the state with stabilizers $\langle Z_1, Z_2, Z_3 \rangle$.

**P1.2** What are the stabilizers after CNOT(2→3) on $\langle X_1, X_2, Z_3 \rangle$?

**P1.3** Is the measurement of Z_1 deterministic for $\langle Z_1 Z_2, X_1 X_2 \rangle$?

### Level 2: Intermediate

**P2.1** Prove that the tableau update for CNOT is correct by verifying the transformation of all Pauli operators.

**P2.2** Show that the circuit H-CNOT-H implements a CZ gate using tableau updates.

**P2.3** For 3-qubit state $\langle X_1X_2X_3, Z_1Z_2, Z_2Z_3 \rangle$:
a) Determine if Z_1 measurement is deterministic
b) If random, compute both possible post-measurement states

### Level 3: Challenging

**P3.1** Prove that the number of stabilizer states on n qubits is $2^n \prod_{k=1}^n (2^k + 1)$.

**P3.2** Show that adding a single T gate to a Clifford circuit can make it hard to simulate classically.

**P3.3** Derive the phase update rule for CNOT in the tableau representation.

---

## Computational Lab

```python
"""
Day 754: Gottesman-Knill Theorem
================================

Implementing stabilizer tableau simulation.
"""

import numpy as np
from typing import Tuple, List, Optional


class StabilizerTableau:
    """
    Stabilizer tableau for n-qubit simulation.

    Represents stabilizer state using 2n generators.
    """

    def __init__(self, n: int):
        """Initialize to |0...0⟩ state."""
        self.n = n

        # Tableau: 2n rows, each row has (x | z | r)
        # x, z are n-bit vectors, r is 1-bit phase
        # Rows 0 to n-1: stabilizers
        # Rows n to 2n-1: destabilizers

        self.x = np.zeros((2*n, n), dtype=int)
        self.z = np.zeros((2*n, n), dtype=int)
        self.r = np.zeros(2*n, dtype=int)

        # Initialize: stabilizers = Z_i, destabilizers = X_i
        for i in range(n):
            self.z[i, i] = 1  # Stabilizer Z_i
            self.x[n + i, i] = 1  # Destabilizer X_i

    def _rowsum(self, h: int, i: int):
        """
        Add row i to row h (used for Gaussian elimination).

        Implements: g_h = g_h * g_i with proper phase tracking.
        """
        # Phase calculation
        def g(x1, z1, x2, z2):
            """Phase function for Pauli multiplication."""
            if x1 == 0 and z1 == 0:
                return 0
            elif x1 == 1 and z1 == 1:
                return z2 - x2
            elif x1 == 1 and z1 == 0:
                return z2 * (2*x2 - 1)
            else:  # x1 == 0 and z1 == 1
                return x2 * (1 - 2*z2)

        phase = 0
        for j in range(self.n):
            phase += g(self.x[i,j], self.z[i,j], self.x[h,j], self.z[h,j])

        phase = (phase + 2*self.r[h] + 2*self.r[i]) % 4

        self.r[h] = 0 if phase == 0 or phase == 2 else 1
        self.x[h] = (self.x[h] + self.x[i]) % 2
        self.z[h] = (self.z[h] + self.z[i]) % 2

    def hadamard(self, j: int):
        """Apply Hadamard to qubit j."""
        for i in range(2 * self.n):
            self.r[i] = (self.r[i] + self.x[i,j] * self.z[i,j]) % 2
            self.x[i,j], self.z[i,j] = self.z[i,j], self.x[i,j]

    def phase(self, j: int):
        """Apply S (phase) gate to qubit j."""
        for i in range(2 * self.n):
            self.r[i] = (self.r[i] + self.x[i,j] * self.z[i,j]) % 2
            self.z[i,j] = (self.z[i,j] + self.x[i,j]) % 2

    def cnot(self, control: int, target: int):
        """Apply CNOT from control to target."""
        for i in range(2 * self.n):
            self.r[i] = (self.r[i] +
                        self.x[i, control] * self.z[i, target] *
                        (self.x[i, target] ^ self.z[i, control] ^ 1)) % 2
            self.x[i, target] = self.x[i, target] ^ self.x[i, control]
            self.z[i, control] = self.z[i, control] ^ self.z[i, target]

    def measure(self, j: int) -> int:
        """
        Measure qubit j in Z basis.

        Returns measurement outcome (0 or 1).
        """
        # Find if any stabilizer anticommutes with Z_j
        p = None
        for i in range(self.n):
            if self.x[i, j] == 1:
                p = i
                break

        if p is None:
            # Deterministic outcome
            # Compute from destabilizers
            outcome = 0
            for i in range(self.n, 2*self.n):
                if self.x[i, j] == 1:
                    outcome = self.r[i]
                    break
            return outcome
        else:
            # Random outcome
            # Update tableau
            for i in range(2 * self.n):
                if i != p and self.x[i, j] == 1:
                    self._rowsum(i, p)

            # Move stabilizer p to destabilizer
            self.x[self.n + p] = self.x[p].copy()
            self.z[self.n + p] = self.z[p].copy()
            self.r[self.n + p] = self.r[p]

            # New stabilizer is Z_j
            self.x[p] = np.zeros(self.n, dtype=int)
            self.z[p] = np.zeros(self.n, dtype=int)
            self.z[p, j] = 1

            # Random outcome
            outcome = np.random.randint(2)
            self.r[p] = outcome

            return outcome

    def stabilizers(self) -> List[str]:
        """Return stabilizer generators as Pauli strings."""
        stabs = []
        for i in range(self.n):
            sign = '+' if self.r[i] == 0 else '-'
            pauli = ''
            for j in range(self.n):
                if self.x[i,j] == 0 and self.z[i,j] == 0:
                    pauli += 'I'
                elif self.x[i,j] == 1 and self.z[i,j] == 0:
                    pauli += 'X'
                elif self.x[i,j] == 0 and self.z[i,j] == 1:
                    pauli += 'Z'
                else:
                    pauli += 'Y'
            stabs.append(sign + pauli)
        return stabs


def bell_state_demo():
    """Demonstrate Bell state preparation."""
    tab = StabilizerTableau(2)
    print("Initial: |00⟩")
    print(f"Stabilizers: {tab.stabilizers()}")

    tab.hadamard(0)
    print("\nAfter H on qubit 0:")
    print(f"Stabilizers: {tab.stabilizers()}")

    tab.cnot(0, 1)
    print("\nAfter CNOT(0→1):")
    print(f"Stabilizers: {tab.stabilizers()}")

    return tab


def ghz_state_demo(n: int = 3):
    """Demonstrate GHZ state preparation."""
    tab = StabilizerTableau(n)
    print(f"Preparing {n}-qubit GHZ state")

    tab.hadamard(0)
    for i in range(1, n):
        tab.cnot(0, i)

    print(f"Stabilizers: {tab.stabilizers()}")
    return tab


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 754: Gottesman-Knill Theorem")
    print("=" * 60)

    # Example 1: Bell state
    print("\n1. Bell State Preparation")
    print("-" * 40)
    bell_state_demo()

    # Example 2: GHZ state
    print("\n2. GHZ State Preparation")
    print("-" * 40)
    ghz_state_demo(4)

    # Example 3: Measurement
    print("\n3. Measurement Simulation")
    print("-" * 40)

    tab = StabilizerTableau(2)
    tab.hadamard(0)
    tab.cnot(0, 1)
    print("Bell state stabilizers:", tab.stabilizers())

    outcome = tab.measure(0)
    print(f"Measured qubit 0: {outcome}")
    print(f"Post-measurement stabilizers: {tab.stabilizers()}")

    # Example 4: Circuit simulation
    print("\n4. Random Clifford Circuit")
    print("-" * 40)

    np.random.seed(42)
    n = 4
    tab = StabilizerTableau(n)

    gates = []
    for _ in range(10):
        gate = np.random.choice(['H', 'S', 'CNOT'])
        if gate == 'H':
            j = np.random.randint(n)
            tab.hadamard(j)
            gates.append(f"H({j})")
        elif gate == 'S':
            j = np.random.randint(n)
            tab.phase(j)
            gates.append(f"S({j})")
        else:
            c, t = np.random.choice(n, 2, replace=False)
            tab.cnot(c, t)
            gates.append(f"CNOT({c},{t})")

    print("Gates:", ' → '.join(gates))
    print("Final stabilizers:", tab.stabilizers())

    # Example 5: Complexity demonstration
    print("\n5. Scalability (Gottesman-Knill in action)")
    print("-" * 40)

    import time

    for n in [10, 50, 100, 500]:
        tab = StabilizerTableau(n)

        start = time.time()
        # Apply random Clifford circuit
        for _ in range(n):
            tab.hadamard(np.random.randint(n))
            if n > 1:
                c, t = np.random.choice(n, 2, replace=False)
                tab.cnot(c, t)

        elapsed = time.time() - start
        print(f"n = {n}: {elapsed:.4f} seconds for {n} gates")

    print("\n" + "=" * 60)
    print("Clifford circuits are classically simulable!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula/Rule |
|---------|--------------|
| Tableau size | O(n²) bits |
| Gate complexity | O(n) per gate |
| Measurement | O(n²) |
| Total simulation | O(n³) for n gates + measurements |

### Main Takeaways

1. **Gottesman-Knill:** Clifford circuits are classically simulable in poly time
2. **Tableau representation:** n² bits describe any stabilizer state
3. **Update rules:** Each Clifford gate has efficient tableau update
4. **Implication:** Clifford alone cannot provide quantum speedup
5. **Boundary:** Non-Clifford gates (T) are required for quantum advantage

---

## Daily Checklist

- [ ] I can state the Gottesman-Knill theorem
- [ ] I understand the tableau representation
- [ ] I can apply gate update rules
- [ ] I know why Clifford is not universal
- [ ] I understand the role of T gates
- [ ] I can implement basic stabilizer simulation

---

## Preview: Day 755

Tomorrow we explore **advanced code constructions**:

- Subsystem color codes
- Floquet codes
- Homological codes
- Beyond stabilizer formalism

These cutting-edge constructions push the boundaries of quantum error correction!
