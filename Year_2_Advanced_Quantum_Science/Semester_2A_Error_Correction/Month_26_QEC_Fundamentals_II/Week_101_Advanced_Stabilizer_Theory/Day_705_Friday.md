# Day 705: Stabilizer Tableaux and Optimized Simulation

## Overview

**Date:** Day 705 of 1008
**Week:** 101 (Advanced Stabilizer Theory)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Advanced Stabilizer Representations and the Stim Library

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Canonical forms and graph states |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Optimized representations and Stim |
| Evening | 7:00 PM - 8:00 PM | 1 hr | High-performance simulation lab |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Convert stabilizer states** to canonical tableau form
2. **Represent stabilizer states** as graph states
3. **Use the Stim library** for high-performance stabilizer simulation
4. **Implement Pauli frame tracking** for efficient error simulation
5. **Optimize syndrome extraction** for error correction applications
6. **Compare simulation approaches** and their trade-offs

---

## Core Content

### 1. Canonical Forms for Stabilizer Tableaux

#### Row Echelon Form

A stabilizer tableau is in **row echelon form** if:
1. Each stabilizer generator has a "pivot" position
2. Pivots are ordered left to right
3. No generator has X or Z at positions to the left of its pivot

**Reduction Algorithm:**

```
For each qubit j from 1 to n:
    1. Find generator with X_j or Z_j = 1 (pivot candidate)
    2. Use row operations to eliminate X_j, Z_j from other rows
    3. Move pivot row to correct position
```

**Benefits:**
- Unique representation (up to phase conventions)
- Efficient extraction of logical operators
- Easy identification of state properties

#### Reduced Row Echelon Form

Further reduce so each pivot column has exactly one 1:

$$\text{Stabilizers} = \begin{pmatrix}
I & A \\
0 & B
\end{pmatrix}$$

where $A$ encodes correlations, $B$ is lower triangular.

---

### 2. Graph State Representation

#### Definition

A **graph state** is a stabilizer state defined by a graph $G = (V, E)$:

1. Each vertex $v \in V$ corresponds to a qubit initialized to $|+\rangle$
2. Each edge $(u, v) \in E$ applies a CZ gate between qubits $u$ and $v$

$$|G\rangle = \prod_{(u,v) \in E} CZ_{uv} |+\rangle^{\otimes n}$$

#### Stabilizer Generators for Graph States

For graph state $|G\rangle$, stabilizer generators are:

$$K_v = X_v \prod_{u \in N(v)} Z_u$$

where $N(v)$ is the neighborhood of vertex $v$.

**Example - 3-qubit line graph:**

Graph: 1 — 2 — 3

$$K_1 = X_1 Z_2, \quad K_2 = Z_1 X_2 Z_3, \quad K_3 = Z_2 X_3$$

#### Universal Property

**Theorem:** Every stabilizer state is locally Clifford equivalent to a graph state.

That is, for any stabilizer state $|\psi\rangle$:

$$|\psi\rangle = (C_1 \otimes \cdots \otimes C_n)|G\rangle$$

where each $C_i$ is a single-qubit Clifford and $|G\rangle$ is a graph state.

---

### 3. Adjacency Matrix Representation

For graph state $|G\rangle$, the adjacency matrix $\Gamma$ encodes the graph:

$$\Gamma_{uv} = \begin{cases} 1 & \text{if } (u,v) \in E \\ 0 & \text{otherwise} \end{cases}$$

**Stabilizer generators in matrix form:**

$$X_v \prod_u Z_u^{\Gamma_{vu}} \quad \text{for each } v$$

**Advantage:** Graph state fully specified by $n \times n$ binary symmetric matrix.

---

### 4. The Stim Library

#### Introduction

**Stim** is a high-performance stabilizer circuit simulator written in C++ with Python bindings, developed by Craig Gidney at Google.

Key features:
- **Speed:** 100-1000× faster than naive implementations
- **Detector error models:** Direct support for error correction
- **Pauli frame simulation:** Efficient error tracking
- **Memory efficient:** Handles millions of qubits for simple circuits

#### Basic Usage

```python
import stim

# Create circuit
circuit = stim.Circuit("""
    H 0
    CNOT 0 1
    M 0 1
""")

# Sample measurements
sampler = circuit.compile_sampler()
samples = sampler.sample(shots=1000)
```

#### Detector Error Model

Stim can extract a **detector error model** showing how physical errors propagate to detection events:

```python
# Define noisy circuit
circuit = stim.Circuit("""
    H 0
    DEPOLARIZE1(0.01) 0
    CNOT 0 1
    DEPOLARIZE2(0.01) 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
""")

# Get error model
dem = circuit.detector_error_model()
```

---

### 5. Pauli Frame Tracking

#### Concept

Instead of tracking the full stabilizer state, track only the **Pauli frame** — the Pauli errors that have accumulated.

**Key insight:** For stabilizer circuits, errors always result in Pauli operators that either:
- Commute with the stabilizers (undetectable)
- Anticommute with some stabilizers (detectable)

#### Frame Simulation

1. **Initialize:** Frame = Identity
2. **Apply Clifford gate $C$:** Frame $\to C \cdot \text{Frame} \cdot C^\dagger$
3. **Apply error $E$:** Frame $\to E \cdot \text{Frame}$
4. **Measurement:** Check frame anticommutation with measurement basis

**Advantage:** $O(1)$ per gate instead of $O(n)$ for many circuits.

#### Application to QEC

For quantum error correction:
1. Track Pauli errors through encoding circuit
2. Compute syndromes from frame
3. Apply decoder to find recovery operation
4. No need to track full quantum state!

---

### 6. Syndrome Extraction Optimization

#### Stabilizer Measurement Circuits

To measure stabilizer generator $g = P_1 P_2 \cdots P_k$:

```
     ┌───┐
|0⟩──┤ H ├──●──●──●──┤ H ├──M
     └───┘  │  │  │  └───┘
            │  │  │
 q1  ───────X──┼──┼─────────
               │  │
 q2  ──────────Z──┼─────────
                  │
 q3  ─────────────X─────────
```

For $g = X_1 Z_2 X_3$, controlled gates extract parity.

#### Flag Qubits

For fault tolerance, use **flag qubits** to detect high-weight errors from syndrome extraction:

```
     ┌───┐
|0⟩──┤ H ├──●──────●──●──────●──┤ H ├──M  (syndrome)
     └───┘  │      │  │      │  └───┘
|0⟩─────────┼──●───┼──┼───●──┼─────────M  (flag)
            │  │   │  │   │  │
 q1  ───────X──X───┼──┼───┼──┼─────────
                   │  │   │  │
 q2  ──────────────Z──┼───Z──┼─────────
                      │      │
 q3  ─────────────────X──────X─────────
```

---

### 7. Benchmark: Simulation Approaches

| Approach | Memory | Time/Gate | Time/Measurement | Best For |
|----------|--------|-----------|------------------|----------|
| State vector | $O(2^n)$ | $O(2^n)$ | $O(2^n)$ | Small systems |
| Stabilizer tableau | $O(n^2)$ | $O(n)$ | $O(n^2)$ | General Clifford |
| Graph state | $O(n^2)$ | $O(n)$ | $O(n^2)$ | Specific states |
| Pauli frame | $O(n)$ | $O(1)$* | $O(1)$* | Error tracking |
| Stim | $O(n^2)$ | $O(n)$** | $O(n)$** | Production QEC |

*For simple circuits; **Highly optimized

---

### 8. Advanced Stim Features

#### Tableau Operations

```python
import stim

# Create tableau for specific state
tab = stim.Tableau.from_named_gate("H")
print(tab)

# Compose tableaux
cnot = stim.Tableau.from_named_gate("CNOT")
```

#### Flow Analysis

Stim can analyze how Pauli operators flow through circuits:

```python
circuit = stim.Circuit("""
    H 0
    CNOT 0 1
    S 1
""")

# Track how Z0 transforms
# (This is the essence of stabilizer simulation)
```

#### Matching with PyMatching

Stim integrates with PyMatching for MWPM decoding:

```python
import stim
import pymatching

# Create surface code circuit
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01
)

# Get matching graph
dem = circuit.detector_error_model()
matcher = pymatching.Matching.from_detector_error_model(dem)
```

---

## Worked Examples

### Example 1: Convert to Graph State

**Problem:** Convert the GHZ state to graph state form with local Cliffords.

**Solution:**

GHZ state stabilizers: $X_1 X_2 X_3$, $Z_1 Z_2$, $Z_2 Z_3$

**Step 1:** Find graph state with these correlations.

Apply $H_2$ and $H_3$:
- $X_1 X_2 X_3 \to X_1 Z_2 Z_3$
- $Z_1 Z_2 \to Z_1 X_2$
- $Z_2 Z_3 \to X_2 X_3$

These aren't quite graph state form. Try different approach:

**Alternative:** GHZ state is local-Clifford equivalent to:

$$|GHZ\rangle = (I \otimes H \otimes H)|G_{\text{star}}\rangle$$

where $G_{\text{star}}$ is the star graph (center at qubit 1):
```
    2
    |
1 - ○
    |
    3
```

Star graph stabilizers:
- $K_1 = X_1 Z_2 Z_3$
- $K_2 = Z_1 X_2$
- $K_3 = Z_1 X_3$

After $(I \otimes H \otimes H)$:
- $X_1 Z_2 Z_3 \to X_1 X_2 X_3$ ✓
- $Z_1 X_2 \to Z_1 Z_2$ ✓
- $Z_1 X_3 \to Z_1 Z_3$ (need $Z_2 Z_3$...)

Actually: $Z_1 Z_2 \cdot Z_1 Z_3 = Z_2 Z_3$ ✓

---

### Example 2: Stim Circuit for Bell State Tomography

**Problem:** Use Stim to simulate Bell state creation and measure correlations.

**Solution:**

```python
import stim
import numpy as np

def bell_state_correlations():
    """Measure XX, YY, ZZ correlations of Bell state."""

    correlations = {}

    for basis_pair, prep in [
        ('ZZ', ''),
        ('XX', 'H 0\nH 1'),
        ('YY', 'S 0\nS 1\nH 0\nH 1')  # S†H for Y basis
    ]:
        circuit = stim.Circuit(f"""
            H 0
            CNOT 0 1
            {prep}
            M 0 1
        """)

        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=10000)

        # Correlation = P(same) - P(different)
        same = np.sum(samples[:, 0] == samples[:, 1])
        diff = 10000 - same
        correlations[basis_pair] = (same - diff) / 10000

    return correlations

# Expected: ZZ = +1, XX = +1, YY = -1
print(bell_state_correlations())
```

---

### Example 3: Efficient Error Simulation

**Problem:** Simulate 1000 rounds of surface code with Stim.

**Solution:**

```python
import stim

# Generate rotated surface code circuit
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3,
    rounds=1000,
    after_clifford_depolarization=0.001,
    after_reset_flip_probability=0.001,
    before_measure_flip_probability=0.001
)

# This circuit has thousands of operations
print(f"Circuit has {circuit.num_operations} operations")
print(f"Circuit has {circuit.num_qubits} qubits")

# Sample is still fast!
sampler = circuit.compile_detector_sampler()
detection_events, observables = sampler.sample(
    shots=1000,
    separate_observables=True
)

# Analyze logical error rate
logical_errors = np.sum(observables)
print(f"Logical error rate: {logical_errors/1000}")
```

---

## Practice Problems

### Direct Application

1. **Problem 1:** Find the graph state representation for the 2-qubit Bell state $|\Phi^+\rangle$.

2. **Problem 2:** Write the adjacency matrix for a 4-qubit ring graph and derive its stabilizer generators.

3. **Problem 3:** Use Stim to create and sample from a 5-qubit GHZ state.

### Intermediate

4. **Problem 4:** Prove that $CZ$ gates between qubits correspond to edges in the graph state representation.

5. **Problem 5:** Implement Pauli frame tracking for a circuit with only X errors.

6. **Problem 6:** Convert a random 4-qubit stabilizer tableau to reduced row echelon form.

### Challenging

7. **Problem 7:** Use Stim to simulate the [[7,1,3]] Steane code with depolarizing noise and extract the detector error model.

8. **Problem 8:** Prove that graph state stabilizers $K_v = X_v \prod_{u \in N(v)} Z_u$ satisfy the commutation requirements.

9. **Problem 9:** Implement a function to convert any stabilizer state to its locally-equivalent graph state.

---

## Computational Lab

```python
"""
Day 705: Stabilizer Tableaux and Optimized Simulation
Week 101: Advanced Stabilizer Theory

Demonstrates advanced stabilizer representations and Stim usage.
"""

import numpy as np
from typing import List, Tuple, Optional
import time

# Try importing stim; provide fallback if not available
try:
    import stim
    STIM_AVAILABLE = True
except ImportError:
    STIM_AVAILABLE = False
    print("Note: stim not installed. Run: pip install stim")


class GraphState:
    """Graph state representation of stabilizer states."""

    def __init__(self, n_qubits: int):
        """Initialize empty graph (product of |+⟩ states)."""
        self.n = n_qubits
        self.adj = np.zeros((n_qubits, n_qubits), dtype=int)

    @classmethod
    def from_edges(cls, n_qubits: int, edges: List[Tuple[int, int]]):
        """Create graph state from edge list."""
        gs = cls(n_qubits)
        for u, v in edges:
            gs.add_edge(u, v)
        return gs

    def add_edge(self, u: int, v: int):
        """Add edge (apply CZ) between qubits u and v."""
        self.adj[u, v] = (self.adj[u, v] + 1) % 2
        self.adj[v, u] = (self.adj[v, u] + 1) % 2

    def get_stabilizers(self) -> List[str]:
        """Return stabilizer generators as strings."""
        stabs = []
        for v in range(self.n):
            paulis = ['I'] * self.n
            paulis[v] = 'X'
            for u in range(self.n):
                if self.adj[v, u] == 1:
                    paulis[u] = 'Z' if paulis[u] == 'I' else 'Y'
            stabs.append('+' + ''.join(paulis))
        return stabs

    def local_complementation(self, v: int):
        """
        Apply local complementation at vertex v.

        This is equivalent to applying (√iX)_v (up to global phase)
        and corresponds to complementing the induced subgraph on N(v).
        """
        neighbors = [u for u in range(self.n) if self.adj[v, u] == 1]
        for i, u in enumerate(neighbors):
            for w in neighbors[i+1:]:
                self.adj[u, w] = (self.adj[u, w] + 1) % 2
                self.adj[w, u] = (self.adj[w, u] + 1) % 2


class CanonicalTableau:
    """Stabilizer tableau with canonical form support."""

    def __init__(self, n: int):
        self.n = n
        # Combined tableau: [X | Z | phase]
        # Rows 0..n-1: destabilizers
        # Rows n..2n-1: stabilizers
        self.tab = np.zeros((2*n, 2*n + 1), dtype=int)

        # Initialize for |0⟩^⊗n
        for i in range(n):
            self.tab[i, i] = 1  # Destab i: X_i
            self.tab[n + i, n + i] = 1  # Stab i: Z_i

    def to_row_echelon(self):
        """Convert stabilizer part to row echelon form."""
        n = self.n
        current_row = n  # Start of stabilizers

        for col in range(2 * n):
            # Find pivot row
            pivot = None
            for row in range(current_row, 2 * n):
                if self.tab[row, col] == 1:
                    pivot = row
                    break

            if pivot is None:
                continue

            # Swap to current position
            if pivot != current_row:
                self.tab[[current_row, pivot]] = self.tab[[pivot, current_row]]

            # Eliminate other rows
            for row in range(n, 2 * n):
                if row != current_row and self.tab[row, col] == 1:
                    self.tab[row] = (self.tab[row] + self.tab[current_row]) % 2

            current_row += 1
            if current_row >= 2 * n:
                break


def demonstrate_graph_states():
    """Demonstrate graph state representations."""

    print("=" * 70)
    print("GRAPH STATE REPRESENTATIONS")
    print("=" * 70)

    # Bell state as graph
    print("\n1. BELL STATE AS GRAPH STATE")
    print("-" * 50)

    # Bell state |Φ⁺⟩ is the graph state for single edge 0-1
    bell_graph = GraphState.from_edges(2, [(0, 1)])
    print(f"  Graph: 0 — 1")
    print(f"  Adjacency matrix:\n{bell_graph.adj}")
    print(f"  Stabilizers: {bell_graph.get_stabilizers()}")
    print("  (Note: X₀Z₁ and Z₀X₁ stabilize Bell state)")

    # GHZ state as star graph
    print("\n2. GHZ STATE (via local Clifford)")
    print("-" * 50)

    star = GraphState.from_edges(3, [(0, 1), (0, 2)])
    print(f"  Star graph: 1—0—2")
    print(f"  Adjacency matrix:\n{star.adj}")
    print(f"  Star stabilizers: {star.get_stabilizers()}")
    print("  Apply H on qubits 1,2 to get GHZ stabilizers")

    # Linear cluster state
    print("\n3. LINEAR CLUSTER STATE")
    print("-" * 50)

    cluster = GraphState.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    print(f"  Graph: 0—1—2—3")
    print(f"  Stabilizers: {cluster.get_stabilizers()}")

    # Local complementation
    print("\n4. LOCAL COMPLEMENTATION")
    print("-" * 50)

    triangle = GraphState.from_edges(3, [(0, 1), (1, 2), (0, 2)])
    print(f"  Triangle graph stabilizers: {triangle.get_stabilizers()}")

    triangle.local_complementation(0)
    print(f"  After LC at vertex 0: {triangle.get_stabilizers()}")
    print(f"  New adjacency:\n{triangle.adj}")


def demonstrate_stim():
    """Demonstrate Stim library capabilities."""

    if not STIM_AVAILABLE:
        print("\n[Skipping Stim demonstration - not installed]")
        return

    print("\n" + "=" * 70)
    print("STIM HIGH-PERFORMANCE SIMULATION")
    print("=" * 70)

    # Basic circuit
    print("\n1. BASIC STIM CIRCUIT")
    print("-" * 50)

    circuit = stim.Circuit("""
        H 0
        CNOT 0 1
        M 0 1
    """)

    print(f"  Circuit:\n{circuit}")

    sampler = circuit.compile_sampler()
    samples = sampler.sample(shots=10)
    print(f"  10 samples:\n{samples.astype(int)}")

    # Noisy circuit
    print("\n2. NOISY CIRCUIT WITH DETECTORS")
    print("-" * 50)

    noisy_circuit = stim.Circuit("""
        R 0 1
        H 0
        DEPOLARIZE1(0.01) 0
        CNOT 0 1
        DEPOLARIZE2(0.01) 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """)

    print(f"  Circuit with 1% depolarizing noise")

    det_sampler = noisy_circuit.compile_detector_sampler()
    detections = det_sampler.sample(shots=1000)
    detection_rate = np.mean(detections)
    print(f"  Detection rate: {detection_rate:.3f}")

    # Performance benchmark
    print("\n3. PERFORMANCE BENCHMARK")
    print("-" * 50)

    for n in [10, 100, 1000]:
        # Create circuit: random Clifford then measure
        ops = []
        ops.append(f"H {' '.join(map(str, range(n)))}")
        for _ in range(n):
            i, j = np.random.choice(n, 2, replace=False)
            ops.append(f"CNOT {i} {j}")
        ops.append(f"M {' '.join(map(str, range(n)))}")

        circuit = stim.Circuit('\n'.join(ops))

        start = time.time()
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=1000)
        elapsed = time.time() - start

        print(f"  n={n:4d} qubits, 1000 shots: {elapsed:.4f}s")

    # Tableau operations
    print("\n4. STIM TABLEAU OPERATIONS")
    print("-" * 50)

    h_tab = stim.Tableau.from_named_gate("H")
    print(f"  Hadamard tableau:\n{h_tab}")

    s_tab = stim.Tableau.from_named_gate("S")
    print(f"\n  S gate tableau:\n{s_tab}")

    # Compose gates
    hs_tab = h_tab.then(s_tab)
    print(f"\n  H then S:\n{hs_tab}")

    # Surface code generation
    print("\n5. SURFACE CODE GENERATION")
    print("-" * 50)

    for d in [3, 5, 7]:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=d,
            after_clifford_depolarization=0.001
        )
        print(f"  Distance {d}: {circuit.num_qubits} qubits, "
              f"{circuit.num_operations} ops")


def demonstrate_pauli_frame():
    """Demonstrate Pauli frame tracking concept."""

    print("\n" + "=" * 70)
    print("PAULI FRAME TRACKING")
    print("=" * 70)

    print("\n1. CONCEPT: TRACKING ERRORS THROUGH CLIFFORDS")
    print("-" * 50)

    print("""
    Instead of tracking full quantum state:
    - Initialize: Pauli frame = I (identity)
    - Error E occurs: frame → E · frame
    - Clifford C applied: frame → C · frame · C†

    At measurement: check if frame anticommutes with measurement.
    """)

    # Simple demonstration
    print("\n2. EXAMPLE: X ERROR THROUGH H GATE")
    print("-" * 50)

    print("  Initial state: |0⟩")
    print("  X error occurs → X|0⟩ = |1⟩")
    print("  Apply H → H|1⟩ = |−⟩")
    print()
    print("  Frame tracking:")
    print("    Frame starts: I")
    print("    X error: Frame = X")
    print("    H gate: Frame = HXH† = Z")
    print("    Final frame: Z")
    print()
    print("  Measurement in Z basis:")
    print("    Frame Z commutes with Z → deterministic")
    print("    But |−⟩ has Z eigenvalue -1 → outcome flipped")

    # With Stim
    if STIM_AVAILABLE:
        print("\n3. STIM PAULI FRAME SIMULATION")
        print("-" * 50)

        circuit = stim.Circuit("""
            R 0
            X_ERROR(0.1) 0
            H 0
            M 0
        """)

        # Frame simulation
        frame_sim = circuit.compile_sampler()
        samples = frame_sim.sample(shots=10000)

        ones = np.sum(samples)
        print(f"  Circuit: |0⟩ → 10% X error → H → Measure")
        print(f"  P(1) = {ones/10000:.3f} (expect ~0.1 from X errors)")


def compare_simulation_speeds():
    """Compare different simulation approaches."""

    print("\n" + "=" * 70)
    print("SIMULATION APPROACH COMPARISON")
    print("=" * 70)

    # Our basic tableau
    from Day_704_Thursday import StabilizerTableau  # Would need actual import

    print("\n  Testing at various qubit counts...")
    print("\n  (Detailed comparison requires implementing multiple backends)")
    print("  Key takeaways:")
    print("  - State vector: O(2^n) - exponential, unusable beyond ~30 qubits")
    print("  - Basic tableau: O(n²) memory, O(n) per gate")
    print("  - Stim: Same complexity but 100-1000× faster constants")
    print("  - Pauli frame: O(n) but only for error tracking")


if __name__ == "__main__":
    demonstrate_graph_states()
    demonstrate_stim()
    demonstrate_pauli_frame()
    compare_simulation_speeds()
```

**Expected Output:**
```
======================================================================
GRAPH STATE REPRESENTATIONS
======================================================================

1. BELL STATE AS GRAPH STATE
--------------------------------------------------
  Graph: 0 — 1
  Adjacency matrix:
[[0 1]
 [1 0]]
  Stabilizers: ['+XZ', '+ZX']
  (Note: X₀Z₁ and Z₀X₁ stabilize Bell state)

2. GHZ STATE (via local Clifford)
--------------------------------------------------
  Star graph: 1—0—2
  Adjacency matrix:
[[0 1 1]
 [1 0 0]
 [1 0 0]]
  Star stabilizers: ['+XZZ', '+ZXI', '+ZIX']
  Apply H on qubits 1,2 to get GHZ stabilizers

3. LINEAR CLUSTER STATE
--------------------------------------------------
  Graph: 0—1—2—3
  Stabilizers: ['+XZII', '+ZXZI', '+IZXZ', '+IIZX']

======================================================================
STIM HIGH-PERFORMANCE SIMULATION
======================================================================

1. BASIC STIM CIRCUIT
--------------------------------------------------
  Circuit:
H 0
CNOT 0 1
M 0 1
  10 samples:
[[0 0]
 [1 1]
 [0 0]
 ...
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Canonical form** | Unique row echelon representation of tableau |
| **Graph states** | Stabilizer states from CZ on $\|+\rangle^{\otimes n}$ |
| **Stim** | High-performance stabilizer simulator |
| **Pauli frame** | Efficient error-only tracking |
| **Detector error model** | How errors map to syndromes |

### Main Takeaways

1. **Graph states** provide intuitive geometric representation of stabilizer states
2. **Every stabilizer state** is LC-equivalent to a graph state
3. **Stim** enables production-scale QEC simulation
4. **Pauli frame tracking** dramatically speeds up error simulation
5. **Canonical forms** enable unique state identification

---

## Daily Checklist

- [ ] Understand canonical tableau forms
- [ ] Construct graph states from edge lists
- [ ] Use Stim for circuit simulation
- [ ] Explain Pauli frame tracking concept
- [ ] Compare simulation approach trade-offs
- [ ] Connect to QEC applications

---

## Preview: Day 706

Tomorrow we study the **Normalizer Structure of the Clifford Group**, examining:
- Automorphisms of the Pauli group
- Clifford group as semidirect product
- Generating sets and normal forms
- Applications to circuit synthesis
