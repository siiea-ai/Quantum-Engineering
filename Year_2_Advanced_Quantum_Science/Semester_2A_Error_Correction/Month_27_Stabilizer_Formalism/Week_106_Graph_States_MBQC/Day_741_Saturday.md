# Day 741: Cluster States and Universality

## Overview

**Day:** 741 of 1008
**Week:** 106 (Graph States & MBQC)
**Month:** 27 (Stabilizer Formalism)
**Topic:** 2D Cluster States and Universal Quantum Computation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Cluster state structure |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Universality proofs |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational examples |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define** cluster states on 1D and 2D lattices
2. **Prove** computational universality of 2D clusters
3. **Implement** universal gate sets via measurements
4. **Understand** computational depth in MBQC
5. **Analyze** error propagation in cluster states
6. **Connect** MBQC to fault-tolerant quantum computation

---

## Core Content

### The 2D Cluster State

**Definition:**
The 2D cluster state on an m × n rectangular lattice is the graph state where:
- Vertices are arranged in a grid
- Edges connect nearest neighbors (horizontally and vertically)

$$|C_{m \times n}\rangle = \prod_{\langle i,j \rangle} CZ_{ij} |+\rangle^{\otimes mn}$$

**Adjacency Structure:**
$$\Gamma_{(r,c),(r',c')} = 1 \iff |r-r'| + |c-c'| = 1$$

### Cluster State Stabilizers

**For vertex at position (r, c):**
$$K_{r,c} = X_{r,c} \cdot Z_{r-1,c} \cdot Z_{r+1,c} \cdot Z_{r,c-1} \cdot Z_{r,c+1}$$

(boundary vertices have fewer Z neighbors)

**Interior vertex:** 5-body operator (X plus 4 Z's)
**Edge vertex:** 4-body operator
**Corner vertex:** 3-body operator

### Why 2D is Necessary for Universality

**1D Cluster (Chain):**
- Only implements single-qubit gates
- No entangling gates possible
- Equivalent to sequential measurements

**2D Cluster:**
- Enables CNOT and two-qubit gates
- Horizontal "wires" carry qubits
- Vertical connections implement entangling operations

**Theorem (Raussendorf-Briegel 2001):**
The 2D cluster state is a **universal resource** for quantum computation.

### Universality Proof Outline

**Universal Gate Set:** {H, T, CNOT}

**Single-qubit gates (1D subcluster):**
- H: Measure in X-basis, then Z-basis
- T = R_Z(π/4): Measure at angle π/4
- Arbitrary rotation: Chain of appropriate measurements

**Two-qubit gates (2D required):**
- CNOT: Use the "cross" pattern

### The CNOT Pattern

**Resource:**
```
1—2—3
  |
  4—5
```

**Pattern:**
- Qubit 1: Input (control)
- Qubit 4: Input (target)
- Qubits 2, 4: Measured
- Qubits 3, 5: Output

**Measurements:**
- Measure 2 in X-basis
- Measure 4 in X-basis
- Byproducts on outputs

**Result:** CNOT with Pauli corrections

### Computational Depth

**Definition:**
Computational **depth** in MBQC is the number of measurement rounds, where measurements within a round can be performed in parallel.

**Key Result:**
MBQC depth ≤ Circuit depth + O(1)

**Parallelism:**
Measurements on non-interacting qubits can be simultaneous.

### Measurement Patterns

**Notation:**
- $E_{ij}$: Prepare edge entanglement
- $M_a^\alpha$: Measure qubit a at angle α
- $X_a^s$: Correction on qubit a

**Example Pattern for CNOT:**
```
E_{12} E_{23} E_{24} E_{45}
M_2^{0,0}(0) M_4^{0,0}(0)
X_3^{s_2} Z_5^{s_2} X_5^{s_4} Z_3^{s_4}
```

### Error Propagation

**Pauli Errors:**
- X error on measured qubit: Flips outcome, propagates as Z
- Z error on measured qubit: Propagates unchanged

**Error Model:**
In cluster state MBQC:
$$X_a \xrightarrow{M_a(θ)} Z_{\text{neighbors}}$$

**Threshold:**
MBQC inherits error thresholds from the underlying cluster state error model.

### Topological Protection

**Connection to Surface Codes:**
- Cluster states on certain lattices are related to surface codes
- Can achieve topological protection
- Leads to fault-tolerant MBQC

**Raussendorf Lattice:**
A 3D cluster state enables topologically protected MBQC.

### Resource Efficiency

**Qubit Overhead:**
MBQC requires more physical qubits than circuit model:
- Each gate consumes ancilla qubits
- Trade-off: simple operations (only measurements)

**Comparison:**
| Model | Qubits | Operations |
|-------|--------|------------|
| Circuit | ~n | Gates + Measurements |
| MBQC | ~n × depth | Only Measurements |

### Practical Considerations

**Advantages of MBQC:**
1. All entanglement prepared offline
2. Only single-qubit measurements needed online
3. Natural for photonic systems

**Challenges:**
1. Large resource state preparation
2. Measurement feed-forward timing
3. Error accumulation in large clusters

---

## Worked Examples

### Example 1: Encoding a Qubit in 1D Cluster

**5-qubit chain:** 1—2—3—4—5

**Encoding |ψ⟩:**
- Prepare |ψ⟩ on qubit 1
- Chain carries it to qubit 5 via measurements

**Measurement sequence:**
- M_1(θ_1): Implements R_Z(θ_1)
- M_2(θ_2): Implements R_X(θ_2)
- M_3(θ_3): Implements R_Z(θ_3)
- M_4(θ_4): Implements R_X(θ_4)

**Output on qubit 5:**
$$U|\psi\rangle = R_X(\theta_4)R_Z(\theta_3)R_X(\theta_2)R_Z(\theta_1)|\psi\rangle$$

with byproduct corrections.

### Example 2: CNOT on 2D Cluster

**Minimal cluster for CNOT:**
```
C—a—C'
    |
T—b—T'
```

C = control input, T = target input
C' = control output, T' = target output
a, b = ancillas to be measured

**State after CZ gates:**
$$|C_{CNOT}\rangle = CZ_{Ca} CZ_{aC'} CZ_{ab} CZ_{Tb} CZ_{bT'} |+++++\rangle$$

**Measurements:**
- M_C(0): Teleport control
- M_T(0): Teleport target
- M_a(0): Entangle
- M_b(0): Complete CNOT

**Result:** CNOT_{C'T'} with Pauli byproduct

### Example 3: Depth Analysis

**Circuit:** H—CNOT—T—T—CNOT—H (6 gates)

**MBQC depth:**
- H: 2 measurements (sequential)
- CNOT: 2 measurements
- T: 1 measurement
- Total: 2 + 2 + 1 + 1 + 2 + 2 = 10 measurements

**But many can be parallel!**
Actual depth ~ 4-5 (with proper arrangement)

### Example 4: Error Propagation

**X error before measurement in X-basis:**

$$X_a |G\rangle \xrightarrow{M_a^X} Z_{\text{neighbors}}$$

The X error flips the measurement outcome, which propagates Z errors to neighbors.

**Z error before measurement:**
$$Z_a |G\rangle \xrightarrow{M_a^X} (\text{no effect on neighbors})$$

Z errors don't flip X-basis outcomes.

---

## Practice Problems

### Level 1: Direct Application

1. **Cluster Stabilizers:** Write the stabilizer generator for the center qubit of a 3×3 cluster.

2. **Gate Count:** How many ancilla qubits are consumed to implement:
   a) A single Hadamard
   b) A CNOT
   c) A Toffoli (CCZ)

3. **1D vs 2D:** Explain why a 1D cluster cannot implement CNOT.

### Level 2: Intermediate

4. **Depth Calculation:** For the circuit H⊗H—CNOT—T⊗I—CNOT—H⊗H, compute the MBQC depth.

5. **Error Analysis:** If an X error occurs on the central qubit of the CNOT cluster pattern, which output qubits are affected?

6. **Resource State:** Design a cluster state pattern for the Toffoli gate.

### Level 3: Challenging

7. **Universality Proof:** Prove that {H, T, CNOT} implemented via MBQC is universal for quantum computation.

8. **Threshold:** Research and explain the error threshold for MBQC on 2D cluster states.

9. **Topological MBQC:** Describe how the 3D Raussendorf lattice achieves fault tolerance.

---

## Computational Lab

```python
"""
Day 741: Cluster States and Universality
========================================
Implementing universal computation via 2D clusters.
"""

import numpy as np
from typing import List, Tuple, Dict

# Basic quantum operations
def tensor(*args) -> np.ndarray:
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def R_Z(theta):
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])

def CZ():
    return np.diag([1, 1, 1, -1]).astype(complex)

def CNOT():
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)

def plus_state():
    return np.array([1, 1], dtype=complex) / np.sqrt(2)

class ClusterState:
    """2D Cluster state representation and operations."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.n_qubits = rows * cols
        self._build_adjacency()
        self._build_state()

    def _idx(self, r: int, c: int) -> int:
        """Convert 2D index to linear index."""
        return r * self.cols + c

    def _build_adjacency(self):
        """Build adjacency matrix for 2D grid."""
        n = self.n_qubits
        self.adjacency = np.zeros((n, n), dtype=int)

        for r in range(self.rows):
            for c in range(self.cols):
                i = self._idx(r, c)
                # Right neighbor
                if c < self.cols - 1:
                    j = self._idx(r, c+1)
                    self.adjacency[i, j] = 1
                    self.adjacency[j, i] = 1
                # Down neighbor
                if r < self.rows - 1:
                    j = self._idx(r+1, c)
                    self.adjacency[i, j] = 1
                    self.adjacency[j, i] = 1

    def _build_state(self):
        """Build the cluster state vector."""
        # Start with |+⟩^⊗n
        self.state = np.ones(2**self.n_qubits, dtype=complex)
        self.state /= np.sqrt(2**self.n_qubits)

        # Apply CZ for each edge
        for i in range(self.n_qubits):
            for j in range(i+1, self.n_qubits):
                if self.adjacency[i, j] == 1:
                    self._apply_cz(i, j)

    def _apply_cz(self, i: int, j: int):
        """Apply CZ to qubits i and j."""
        for x in range(2**self.n_qubits):
            bit_i = (x >> (self.n_qubits - 1 - i)) & 1
            bit_j = (x >> (self.n_qubits - 1 - j)) & 1
            if bit_i == 1 and bit_j == 1:
                self.state[x] *= -1

    def get_stabilizer(self, r: int, c: int) -> str:
        """Return stabilizer string for qubit at (r, c)."""
        n = self.n_qubits
        pauli = ['I'] * n
        i = self._idx(r, c)
        pauli[i] = 'X'

        # Add Z on neighbors
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                j = self._idx(nr, nc)
                pauli[j] = 'Z'

        return ''.join(pauli)

    def display_grid(self, show_connections: bool = True):
        """Display the cluster grid."""
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                row_str += "●"
                if c < self.cols - 1 and show_connections:
                    row_str += "—"
            print(row_str)
            if r < self.rows - 1 and show_connections:
                print("|   " * self.cols)

def simulate_1d_cluster_computation(input_state: np.ndarray,
                                     angles: List[float]) -> Tuple[np.ndarray, List[int]]:
    """
    Simulate MBQC on a 1D cluster.

    Parameters:
    -----------
    input_state : np.ndarray
        Initial qubit state
    angles : List[float]
        Measurement angles for each qubit

    Returns:
    --------
    output_state : np.ndarray
        Final qubit state
    outcomes : List[int]
        Measurement outcomes
    """
    n = len(angles) + 1  # Input + ancillas

    # Build linear cluster with input
    # |ψ⟩|+⟩|+⟩...
    state = input_state.copy()
    for _ in range(n - 1):
        state = np.kron(state, plus_state())

    # Apply CZ between consecutive qubits
    for i in range(n - 1):
        for x in range(2**n):
            bit_i = (x >> (n - 1 - i)) & 1
            bit_j = (x >> (n - 1 - i - 1)) & 1
            if bit_i == 1 and bit_j == 1:
                state[x] *= -1

    # Measure each qubit except the last
    outcomes = []
    for i, theta in enumerate(angles):
        # Measurement basis
        plus_theta = np.array([1, np.exp(1j * theta)]) / np.sqrt(2)
        minus_theta = np.array([1, -np.exp(1j * theta)]) / np.sqrt(2)

        # Build projectors for qubit i
        proj_plus = np.eye(1, dtype=complex)
        proj_minus = np.eye(1, dtype=complex)

        for q in range(n):
            if q == i:
                proj_plus = np.kron(proj_plus, np.outer(plus_theta, plus_theta.conj()))
                proj_minus = np.kron(proj_minus, np.outer(minus_theta, minus_theta.conj()))
            else:
                proj_plus = np.kron(proj_plus, I)
                proj_minus = np.kron(proj_minus, I)

        # Measure
        p_plus = np.real(state.conj() @ proj_plus @ state)
        p_minus = 1 - p_plus

        if np.random.random() < p_plus:
            outcome = 0
            state = proj_plus @ state / np.sqrt(p_plus)
        else:
            outcome = 1
            state = proj_minus @ state / np.sqrt(p_minus)

        outcomes.append(outcome)

    # Extract output qubit (last qubit)
    # After all measurements, state is factorized
    output = np.zeros(2, dtype=complex)
    for x in range(2**n):
        last_bit = x & 1
        output[last_bit] += state[x]

    output = output / np.linalg.norm(output)

    return output, outcomes

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 741: Cluster States and Universality")
    print("=" * 60)

    np.random.seed(42)

    # Example 1: Build 2D cluster
    print("\n1. 2D Cluster State (3×3)")
    print("-" * 40)

    cluster = ClusterState(3, 3)
    print("Grid structure:")
    cluster.display_grid()

    print("\nAdjacency matrix:")
    print(cluster.adjacency)

    # Example 2: Stabilizers
    print("\n2. Cluster Stabilizers")
    print("-" * 40)

    print("Center stabilizer (1,1):", cluster.get_stabilizer(1, 1))
    print("Corner stabilizer (0,0):", cluster.get_stabilizer(0, 0))
    print("Edge stabilizer (0,1):", cluster.get_stabilizer(0, 1))

    # Example 3: 1D cluster computation
    print("\n3. 1D Cluster Computation (Rotations)")
    print("-" * 40)

    psi = np.array([1, 0], dtype=complex)  # |0⟩
    angles = [np.pi/4, np.pi/2, np.pi/4]  # Three rotations

    print(f"Input: |0⟩")
    print(f"Measurement angles: {[f'{a/np.pi:.2f}π' for a in angles]}")

    for trial in range(3):
        output, outcomes = simulate_1d_cluster_computation(psi.copy(), angles)
        print(f"  Trial {trial+1}: outcomes={outcomes}, |output|² = {np.abs(output)**2}")

    # Example 4: Verify single gate
    print("\n4. Verify R_Z(π/4) via 1D Cluster")
    print("-" * 40)

    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+⟩
    expected = R_Z(np.pi/4) @ psi

    print(f"Input: |+⟩")
    print(f"Expected R_Z(π/4)|+⟩: {expected}")

    successes = 0
    n_trials = 100
    for _ in range(n_trials):
        output, [s] = simulate_1d_cluster_computation(psi.copy(), [np.pi/4])
        # Apply X^s correction
        if s == 1:
            output = X @ output
        # Check
        if np.abs(np.vdot(output, expected)) > 0.99:
            successes += 1

    print(f"Success rate: {successes/n_trials:.1%}")

    # Example 5: Gate depth comparison
    print("\n5. Gate Depth vs MBQC Depth")
    print("-" * 40)

    circuits = [
        ("H", 1, 2),  # H uses 2 measurements
        ("T", 1, 1),  # T uses 1 measurement
        ("H-T-H", 3, 4),  # Sequential
        ("CNOT", 1, 2),  # Needs 2D, 2 measurements
    ]

    print("Gate(s)    | Circuit Depth | MBQC Depth")
    print("-----------+---------------+-----------")
    for name, circ_depth, mbqc_depth in circuits:
        print(f"{name:10s} | {circ_depth:13d} | {mbqc_depth}")

    # Example 6: Error propagation
    print("\n6. Error Propagation Example")
    print("-" * 40)

    print("X error on qubit 0 before X-measurement:")
    print("  → Outcome flips: s → s⊕1")
    print("  → Propagates Z to neighbors")
    print("")
    print("Z error on qubit 0 before X-measurement:")
    print("  → No effect on outcome")
    print("  → Absorbed by measurement")

    print("\n" + "=" * 60)
    print("End of Day 741 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| 2D Cluster | $\|C\rangle = \prod_{\langle ij \rangle} CZ_{ij} \|+\rangle^{\otimes mn}$ |
| Interior stabilizer | $K = X \cdot Z^{\otimes 4}$ (4 neighbors) |
| Depth relation | MBQC depth ≤ Circuit depth + O(1) |
| X error propagation | $X_a \to Z_{\text{neighbors}}$ (via measurement) |

### Main Takeaways

1. **2D cluster states** are universal resources for QC
2. **1D chains** implement single-qubit gates only
3. **CNOT** requires 2D connectivity
4. **Depth** is comparable to circuit model
5. **Errors propagate** predictably through measurements
6. **Topological MBQC** achieves fault tolerance

---

## Daily Checklist

- [ ] I understand 2D cluster state structure
- [ ] I know why 2D is needed for universality
- [ ] I can implement gates via cluster measurements
- [ ] I understand computational depth in MBQC
- [ ] I know how errors propagate
- [ ] I understand the connection to fault tolerance

---

## Preview: Day 742

Tomorrow is **Week 106 Synthesis**:
- Comprehensive review of graph states
- Integration with MBQC concepts
- Practice problems
- Preparation for CSS codes (Week 107)
