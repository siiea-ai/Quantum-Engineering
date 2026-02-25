# Day 883: Logical Circuit Model

## Overview

**Day:** 883 of 1008
**Week:** 127 (Logical Gate Compilation)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** From Algorithm to Logical Circuit - Abstraction Layers in Quantum Compilation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Compilation stack and abstraction layers |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Universal gate sets and logical representation |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Describe** the complete quantum compilation stack from algorithm to physical operations
2. **Identify** the abstraction layers separating logical and physical computation
3. **Explain** why universal gate sets are essential for quantum compilation
4. **Compare** different universal gate sets and their trade-offs
5. **Construct** logical circuit representations from high-level algorithm descriptions
6. **Analyze** the role of logical qubit routing in compilation

---

## The Quantum Compilation Problem

### Why Compilation Matters

Quantum algorithms are written at a high level of abstraction:

```python
# High-level algorithm (e.g., VQE)
circuit.rx(theta, qubit_0)
circuit.cnot(qubit_0, qubit_1)
circuit.measure_all()
```

But fault-tolerant hardware executes at a very different level:

```
# Physical operations
Measure X stabilizer on logical qubit patch A
Perform lattice surgery merge between patches A and B
Inject magic state from factory 3
...
```

**The gap** between these levels spans several orders of magnitude in complexity. Compilation bridges this gap.

### The Compilation Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Level 5: Quantum Algorithm                                 │
│  (Shor's, Grover's, VQE, QAOA, quantum simulation)         │
├─────────────────────────────────────────────────────────────┤
│  Level 4: High-Level Circuit                                │
│  (Arbitrary unitaries, controlled operations, oracles)      │
├─────────────────────────────────────────────────────────────┤
│  Level 3: Logical Circuit (Universal Gate Set)              │
│  (Clifford+T gates on logical qubits)                       │
├─────────────────────────────────────────────────────────────┤
│  Level 2: Fault-Tolerant Instructions                       │
│  (Lattice surgery, transversal gates, magic injection)      │
├─────────────────────────────────────────────────────────────┤
│  Level 1: Physical Operations                               │
│  (Syndrome measurements, physical qubit control)            │
└─────────────────────────────────────────────────────────────┘
```

### Compilation Transformations

Each level transition requires specific transformations:

| Transition | Transformation | Key Challenge |
|------------|----------------|---------------|
| Level 5 → 4 | Algorithm synthesis | Optimal oracle construction |
| Level 4 → 3 | Gate decomposition | T-count minimization |
| Level 3 → 2 | FT mapping | Surgery scheduling |
| Level 2 → 1 | Physical mapping | Error management |

---

## Abstraction Layers

### Level 5: Quantum Algorithm

At this level, we describe computational tasks:

**Example: Phase Estimation**

$$|\psi\rangle \xrightarrow{\text{QPE}} |\tilde{\phi}\rangle$$

where $\tilde{\phi}$ approximates the eigenvalue phase.

The algorithm specifies:
- Number of precision qubits
- Target eigenvalue precision
- Ancilla requirements

**No gate-level detail** - just computational intent.

### Level 4: High-Level Circuit

Algorithms decompose into standard circuit components:

**Phase Estimation Circuit:**

$$\text{QPE} = \text{QFT}^{-1} \cdot \prod_{j=0}^{n-1} \text{controlled-}U^{2^j} \cdot H^{\otimes n}$$

**Components:**
- Quantum Fourier Transform (QFT)
- Controlled unitary powers
- Hadamard layer

Each component may use arbitrary rotations and multi-qubit gates.

### Level 3: Logical Circuit

The **logical circuit** expresses the computation in a **universal gate set** that can be implemented fault-tolerantly.

**Standard Clifford+T Gate Set:**

$$\boxed{\mathcal{G} = \{H, S, \text{CNOT}, T\}}$$

where:

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad
S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \quad
T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

**Key Properties:**
- Clifford gates ({H, S, CNOT}) are "cheap" (transversal or Pauli frame)
- T gates are "expensive" (require magic states)
- Together, universal for quantum computation

### Level 2: Fault-Tolerant Instructions

Logical gates become sequences of FT operations:

**Logical CNOT (Lattice Surgery):**
```
1. Prepare auxiliary patch in |+⟩_L
2. Merge with control patch (XX measurement)
3. Merge with target patch (ZZ measurement)
4. Split operations
5. Byproduct Pauli corrections
```

**Logical T Gate:**
```
1. Consume magic state |T⟩ = T|+⟩
2. CNOT with data qubit
3. Measure data in X basis
4. Classically controlled S correction
```

### Level 1: Physical Operations

Each FT instruction becomes physical operations:

**Single Lattice Surgery Merge:**
- $d$ rounds of syndrome measurement
- $d \times d$ physical qubits per logical patch
- Classical decoding between rounds
- Tens of thousands of physical operations

---

## Universal Gate Sets

### Definition of Universality

**Definition:** A gate set $\mathcal{G}$ is **universal** if any unitary $U \in SU(2^n)$ can be approximated to arbitrary precision $\epsilon$ using gates from $\mathcal{G}$.

**Formally:**
$$\forall U \in SU(2^n), \forall \epsilon > 0, \exists g_1, \ldots, g_m \in \mathcal{G}: \|U - g_m \cdots g_1\| < \epsilon$$

### The Solovay-Kitaev Theorem

**Theorem (Solovay-Kitaev):** Let $\mathcal{G}$ be a universal gate set that generates a dense subgroup of $SU(2)$. Then any $U \in SU(2)$ can be $\epsilon$-approximated using:

$$\boxed{O\left(\log^c\left(\frac{1}{\epsilon}\right)\right) \text{ gates from } \mathcal{G}}$$

where $c \approx 3.97$ for the original algorithm.

**Significance:** Gate count grows only polylogarithmically with precision.

### Common Universal Gate Sets

| Gate Set | Generators | FT Implementation | Notes |
|----------|------------|-------------------|-------|
| Clifford+T | H, S, CNOT, T | Standard | Most common |
| Clifford+Toffoli | H, S, CNOT, CCZ | Via decomposition | Algorithm-natural |
| Fibonacci | $F = e^{4\pi i/5} R_z(2\pi/5)$ | Topological | Intrinsically FT |
| Rotation | H, CNOT, $R_z(\theta)$ | Via T synthesis | Variational |

### Clifford+T Universality Proof

**Theorem:** $\{H, S, \text{CNOT}, T\}$ is universal.

**Proof Sketch:**

1. **Single-qubit universality:** Show $\{H, T\}$ is dense in $SU(2)$

   The group generated by H and T contains:
   $$HTH = e^{-i\pi/8} R_x(\pi/4)$$

   Combined with $T = R_z(\pi/4)$, we get rotations around two non-parallel axes.

   **Key lemma:** Rotations around two non-parallel axes generate all of $SU(2)$.

2. **Multi-qubit universality:** CNOT enables entangling operations

   Any $U \in SU(2^n)$ can be decomposed into single-qubit gates and CNOTs (Barenco et al., 1995).

**Conclusion:** $\{H, S, T, \text{CNOT}\}$ is universal. (S is technically redundant: $S = T^2$, but included for convenience.) $\square$

---

## Logical Circuit Representation

### Circuit Graph Representation

A logical circuit can be represented as a directed acyclic graph (DAG):

**Nodes:** Gates
**Edges:** Qubit dependencies

```
Example: Bell State Preparation + Measurement

     ┌───┐     ┌───┐
q0: ─┤ H ├──●──┤ M ├
     └───┘  │  └───┘
            │
     ┌───┐ ┌─┴─┐┌───┐
q1: ─┤ I ├─┤ X ├┤ M ├
     └───┘ └───┘└───┘

DAG:
  H(q0) ──→ CNOT(q0,q1) ──→ M(q0)
               │
               └──────────→ M(q1)
```

### Gate Representation

**Data Structure for Logical Gates:**

```python
@dataclass
class LogicalGate:
    """
    Logical gate in the compilation pipeline.

    Attributes:
        gate_type: Type of gate (H, S, T, CNOT, etc.)
        qubits: Tuple of qubit indices involved
        parameters: Gate parameters (for rotations)
        dependencies: Gates that must complete before this
    """
    gate_type: str
    qubits: Tuple[int, ...]
    parameters: Tuple[float, ...] = ()
    dependencies: List[int] = field(default_factory=list)
```

### Circuit Intermediate Representation (IR)

Modern quantum compilers use IRs similar to classical compilers:

**QASM3 (OpenQASM 3.0):**
```qasm
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
c = measure q;
```

**LLVM-Style Quantum IR:**
```
%q0 = qubit.alloc
%q1 = qubit.alloc
%q0 = h %q0
%q0, %q1 = cnot %q0, %q1
%c0 = measure %q0
%c1 = measure %q1
```

---

## Logical Qubit Routing

### The Routing Problem

Physical constraints often limit which qubits can directly interact.

**Constraint Graph:** Defines allowed two-qubit operations

```
Linear topology:     q0 ─ q1 ─ q2 ─ q3 ─ q4

Square grid:         q0 ─ q1 ─ q2
                      │    │    │
                     q3 ─ q4 ─ q5
                      │    │    │
                     q6 ─ q7 ─ q8
```

**Challenge:** Execute CNOT(q0, q8) on square grid where only adjacent qubits interact.

### SWAP Networks

**SWAP gate:** Exchanges qubit states

$$\text{SWAP} = \begin{pmatrix} 1&0&0&0 \\ 0&0&1&0 \\ 0&1&0&0 \\ 0&0&0&1 \end{pmatrix}$$

**SWAP decomposition:**
$$\text{SWAP} = \text{CNOT}_{12} \cdot \text{CNOT}_{21} \cdot \text{CNOT}_{12}$$

**Routing CNOT(q0, q8) on grid:**
```
1. SWAP(q0, q1), SWAP(q1, q4), SWAP(q4, q7)
2. CNOT(q7, q8)
3. SWAP(q7, q4), SWAP(q4, q1), SWAP(q1, q0)
```

**Cost:** 6 SWAPs = 18 CNOTs (naive) → Can optimize to ~9 CNOTs

### Routing Algorithms

| Algorithm | Complexity | Quality | Notes |
|-----------|------------|---------|-------|
| Greedy | O(n) | Poor | Fast but suboptimal |
| Sabre | O(n²) | Good | IBM default |
| TOQM | O(exp) | Optimal | Limited scalability |
| RL-based | Varies | Good | Learning-based |

### Routing in Lattice Surgery

For surface code lattice surgery, routing works differently:

**Routing Operations:**
- **Move:** Teleport logical qubit to new position
- **Turn:** Rotate qubit orientation
- **Merge routes:** Create entanglement paths

**Advantage:** Routing can often be performed in parallel with computation via "long-range entanglement" through auxiliary patches.

---

## Algorithm Design Implications

### T-Count Aware Algorithm Design

Since T gates dominate cost, algorithms should minimize T-count:

**Example: Toffoli Gate**

Standard decomposition: **7 T gates**

```
     ┌───┐                                        ┌───┐
q0: ─┤ T ├────────●────────●────────●─────────────┤T†├───
     └───┘        │        │        │             └───┘
     ┌───┐      ┌─┴─┐    ┌───┐    ┌─┴─┐          ┌───┐
q1: ─┤ T ├──────┤ X ├────┤T†├────┤ X ├──────────┤ T ├───
     └───┘      └───┘    └───┘    └───┘          └───┘
     ┌───┐┌───┐       ┌───┐    ┌───┐    ┌───┐┌───┐
q2: ─┤ H ├┤ T ├───────┤ X ├────┤ T ├────┤ X ├┤ H ├───
     └───┘└───┘       └───┘    └───┘    └───┘└───┘
```

**With measurement:** **4 T gates** (Gidney, 2018)

### Arithmetic Circuit Design

**Ripple-carry adder:** $O(n)$ Toffolis → $O(7n)$ T gates
**Cuccaro adder:** $O(n)$ Toffolis → $O(7n)$ T gates
**Draper QFT adder:** $O(n^2)$ rotations → $O(n^2 \log(n/\epsilon))$ T gates

**Trade-off:** Space (ancillas) vs. T-count vs. depth

### Oracle Construction

For algorithms like Grover's search, the oracle is often the dominant cost:

**Example: Database search oracle for N items**

$$O_f |x\rangle = (-1)^{f(x)} |x\rangle$$

**T-count:** Depends on function complexity
- Simple comparisons: $O(\log N)$ T gates per call
- Cryptographic functions: $O(N)$ T gates per call

**Grover iterations:** $O(\sqrt{N})$

**Total T-count:** $O(\sqrt{N} \cdot T_{\text{oracle}})$

---

## Worked Examples

### Example 1: Circuit Decomposition

**Problem:** Decompose the circuit $U = R_y(\pi/4) \cdot \text{CNOT}_{01} \cdot H_0$ into Clifford+T.

**Solution:**

**Step 1:** Identify components
- $H_0$: Already Clifford
- $\text{CNOT}_{01}$: Already Clifford
- $R_y(\pi/4)$: Requires decomposition

**Step 2:** Decompose $R_y(\theta)$

Using the identity:
$$R_y(\theta) = R_z(-\pi/2) \cdot R_x(\theta) \cdot R_z(\pi/2)$$

And:
$$R_x(\theta) = H \cdot R_z(\theta) \cdot H$$

So:
$$R_y(\pi/4) = S^{\dagger} \cdot H \cdot R_z(\pi/4) \cdot H \cdot S$$

**Step 3:** Approximate $R_z(\pi/4)$

$R_z(\pi/4) = T$ (exactly!)

**Result:**
$$R_y(\pi/4) = S^{\dagger} H T H S$$

**Complete decomposition:**
$$U = S^{\dagger} H T H S \cdot \text{CNOT}_{01} \cdot H_0$$

**T-count:** 1 (plus Cliffords)

### Example 2: Dependency Graph Construction

**Problem:** Construct the dependency DAG for the following circuit:

```
q0: ──H──●──T──●──
         │     │
q1: ──●──┼──●──X──
      │  │  │
q2: ──X──X──X─────
```

**Solution:**

**Gates in order:**
1. H(q0)
2. CNOT(q1→q2) [control q1, target q2]
3. CNOT(q0→q2)
4. T(q0)
5. CNOT(q1→q2)
6. CNOT(q0→q1)

**Dependencies:**
- Gate 1 (H on q0): No dependencies
- Gate 2 (CNOT q1→q2): No dependencies (different qubits from Gate 1)
- Gate 3 (CNOT q0→q2): Depends on Gate 1 (q0), Gate 2 (q2)
- Gate 4 (T on q0): Depends on Gate 3 (q0)
- Gate 5 (CNOT q1→q2): Depends on Gate 3 (q2), Gate 2 (q1)
- Gate 6 (CNOT q0→q1): Depends on Gate 4 (q0), Gate 5 (q1)

**DAG:**
```
     (1:H)─────→(3:CNOT)──→(4:T)──→(6:CNOT)
                   ↑                    ↑
     (2:CNOT)──────┴──────→(5:CNOT)────┘
```

**Critical Path:** 1 → 3 → 4 → 6 (length 4)

### Example 3: Gate Set Conversion

**Problem:** Convert the circuit using {$R_x$, $R_z$, CZ} to Clifford+T.

Original: $CZ \cdot R_x(\pi/2)_0 \cdot R_z(\theta)_1$

**Solution:**

**Step 1:** Convert CZ to CNOT
$$CZ = (I \otimes H) \cdot \text{CNOT} \cdot (I \otimes H)$$

**Step 2:** Convert $R_x(\pi/2)$
$$R_x(\pi/2) = H \cdot S \cdot H = HSH$$

**Step 3:** Convert $R_z(\theta)$ for arbitrary $\theta$
This requires Solovay-Kitaev or gridsynth synthesis.

For $\theta = \pi/4$: $R_z(\pi/4) = T$
For $\theta = \pi/2$: $R_z(\pi/2) = S$
For other $\theta$: Approximate with T-gates

**Result for $\theta = \pi/4$:**
$$U = H_1 \cdot \text{CNOT}_{01} \cdot H_1 \cdot HSH_0 \cdot T_1$$

Simplified:
$$U = H_1 \cdot \text{CNOT}_{01} \cdot H_1 \cdot H_0 \cdot S_0 \cdot H_0 \cdot T_1$$

---

## Practice Problems

### Level 1: Direct Application

**P1.1** How many abstraction levels separate a quantum algorithm (Level 5) from physical operations (Level 1)? Name each level.

**P1.2** Given the gate set {H, CNOT, $R_z(\pi/4)$}, express:
a) The S gate
b) The T gate
c) The X gate

**P1.3** Draw the dependency DAG for:
```
q0: ──H──●──
         │
q1: ──H──X──H──
```

### Level 2: Intermediate

**P2.1** Prove that the gate set {H, T} generates a dense subgroup of SU(2).

*Hint:* Show that $HTH$ and $T$ are rotations around non-parallel axes.

**P2.2** A circuit has the following dependency structure with gates labeled A through F:
- A and B have no dependencies
- C depends on A
- D depends on A and B
- E depends on C and D
- F depends on D

a) Draw the DAG
b) Find the critical path length
c) What is the maximum parallelization speedup?

**P2.3** Design a SWAP network to execute CNOT(q0, q3) on a linear chain q0-q1-q2-q3. Minimize the total CNOT count.

### Level 3: Challenging

**P3.1** The Solovay-Kitaev algorithm achieves gate count $O(\log^c(1/\epsilon))$ with $c \approx 4$. Modern synthesis achieves $c \approx 1$.

a) For $\epsilon = 10^{-10}$, estimate the gate count ratio between these approaches.
b) Explain why this improvement is critical for practical fault-tolerant computing.

**P3.2** Design a compilation strategy for a variational quantum eigensolver (VQE) circuit with 50 qubits and 200 parameterized rotation gates. Assume each rotation requires synthesis to $\epsilon = 10^{-8}$.

a) Estimate the total T-count
b) Propose an optimization strategy

**P3.3** Consider a quantum algorithm with the following structure:
- 100 logical qubits
- 10,000 CNOT gates
- 5,000 T gates
- 2,000 arbitrary rotations (each requiring ~50 T gates after synthesis)

a) Calculate the total T-count
b) Estimate the compilation depth assuming perfect parallelism
c) What is the bottleneck in this compilation?

---

## Computational Lab

```python
"""
Day 883: Logical Circuit Model
==============================

Implementing the logical circuit representation and basic compilation utilities.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


# =============================================================================
# Gate Definitions
# =============================================================================

@dataclass
class LogicalGate:
    """
    Represents a logical gate in the quantum circuit.

    Attributes:
        gate_id: Unique identifier for this gate
        gate_type: Type of gate (H, S, T, CNOT, etc.)
        qubits: Tuple of qubit indices this gate acts on
        parameters: Optional parameters (for rotation gates)
        t_cost: T-gate cost for this gate
    """
    gate_id: int
    gate_type: str
    qubits: Tuple[int, ...]
    parameters: Tuple[float, ...] = ()
    t_cost: int = 0

    def __post_init__(self):
        """Set T-cost based on gate type."""
        t_costs = {
            'T': 1, 'Tdg': 1, 'S': 0, 'Sdg': 0,
            'H': 0, 'X': 0, 'Y': 0, 'Z': 0,
            'CNOT': 0, 'CZ': 0, 'SWAP': 0,
            'Toffoli': 7, 'CCZ': 7,
        }
        if self.gate_type in t_costs:
            self.t_cost = t_costs[self.gate_type]
        elif self.gate_type.startswith('Rz'):
            # Rotation gate - cost depends on precision
            self.t_cost = 50  # Placeholder for synthesis cost

    def __repr__(self):
        if self.parameters:
            return f"{self.gate_type}({self.parameters})@{self.qubits}"
        return f"{self.gate_type}@{self.qubits}"


# =============================================================================
# Logical Circuit Class
# =============================================================================

class LogicalCircuit:
    """
    Represents a quantum circuit at the logical level.

    Supports:
    - Gate addition
    - Dependency analysis
    - T-count computation
    - Critical path analysis
    """

    def __init__(self, num_qubits: int):
        """Initialize circuit with given number of qubits."""
        self.num_qubits = num_qubits
        self.gates: List[LogicalGate] = []
        self.next_gate_id = 0

        # Track last gate on each qubit for dependency computation
        self._last_gate_on_qubit: Dict[int, int] = {}

        # Dependency graph: gate_id -> list of gate_ids it depends on
        self.dependencies: Dict[int, List[int]] = defaultdict(list)

    def add_gate(self, gate_type: str, qubits: Tuple[int, ...],
                 parameters: Tuple[float, ...] = ()) -> int:
        """
        Add a gate to the circuit.

        Returns the gate ID.
        """
        gate = LogicalGate(
            gate_id=self.next_gate_id,
            gate_type=gate_type,
            qubits=qubits,
            parameters=parameters
        )

        # Compute dependencies based on qubits touched
        deps = set()
        for q in qubits:
            if q in self._last_gate_on_qubit:
                deps.add(self._last_gate_on_qubit[q])

        self.dependencies[gate.gate_id] = list(deps)

        # Update last gate on each qubit
        for q in qubits:
            self._last_gate_on_qubit[q] = gate.gate_id

        self.gates.append(gate)
        self.next_gate_id += 1

        return gate.gate_id

    # Convenience methods for common gates
    def h(self, qubit: int): return self.add_gate('H', (qubit,))
    def s(self, qubit: int): return self.add_gate('S', (qubit,))
    def t(self, qubit: int): return self.add_gate('T', (qubit,))
    def x(self, qubit: int): return self.add_gate('X', (qubit,))
    def cnot(self, control: int, target: int):
        return self.add_gate('CNOT', (control, target))
    def toffoli(self, c1: int, c2: int, target: int):
        return self.add_gate('Toffoli', (c1, c2, target))
    def rz(self, qubit: int, theta: float):
        return self.add_gate('Rz', (qubit,), (theta,))

    def t_count(self) -> int:
        """Return total T-count of the circuit."""
        return sum(g.t_cost for g in self.gates)

    def gate_count(self) -> Dict[str, int]:
        """Return count of each gate type."""
        counts = defaultdict(int)
        for g in self.gates:
            counts[g.gate_type] += 1
        return dict(counts)

    def depth(self) -> int:
        """
        Compute circuit depth (critical path length).

        Uses topological sorting with level computation.
        """
        if not self.gates:
            return 0

        # Compute level for each gate
        levels = {}

        def get_level(gate_id: int) -> int:
            if gate_id in levels:
                return levels[gate_id]

            deps = self.dependencies[gate_id]
            if not deps:
                levels[gate_id] = 0
            else:
                levels[gate_id] = 1 + max(get_level(d) for d in deps)

            return levels[gate_id]

        for gate in self.gates:
            get_level(gate.gate_id)

        return max(levels.values()) + 1

    def critical_path(self) -> List[int]:
        """
        Find the critical path (longest dependency chain).

        Returns list of gate IDs in the critical path.
        """
        if not self.gates:
            return []

        # Compute level and predecessor for each gate
        levels = {}
        predecessors = {}

        def compute_level(gate_id: int) -> int:
            if gate_id in levels:
                return levels[gate_id]

            deps = self.dependencies[gate_id]
            if not deps:
                levels[gate_id] = 0
                predecessors[gate_id] = None
            else:
                max_dep = max(deps, key=lambda d: compute_level(d))
                levels[gate_id] = 1 + levels[max_dep]
                predecessors[gate_id] = max_dep

            return levels[gate_id]

        for gate in self.gates:
            compute_level(gate.gate_id)

        # Find gate with maximum level
        end_gate = max(levels.keys(), key=lambda g: levels[g])

        # Trace back
        path = []
        current = end_gate
        while current is not None:
            path.append(current)
            current = predecessors[current]

        return list(reversed(path))

    def parallelism_factor(self) -> float:
        """
        Compute parallelism factor: total_gates / depth.

        Higher values indicate more parallelism opportunity.
        """
        d = self.depth()
        if d == 0:
            return 0.0
        return len(self.gates) / d

    def dependency_graph_dot(self) -> str:
        """Generate DOT format for visualization."""
        lines = ["digraph Circuit {"]
        lines.append("  rankdir=LR;")

        for gate in self.gates:
            label = str(gate).replace('"', '\\"')
            lines.append(f'  {gate.gate_id} [label="{label}"];')

        for gate_id, deps in self.dependencies.items():
            for dep in deps:
                lines.append(f"  {dep} -> {gate_id};")

        lines.append("}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"LogicalCircuit(qubits={self.num_qubits}, "
                f"gates={len(self.gates)}, t_count={self.t_count()}, "
                f"depth={self.depth()})")


# =============================================================================
# Universal Gate Set Conversion
# =============================================================================

class GateSetConverter:
    """
    Convert between different universal gate sets.
    """

    @staticmethod
    def cz_to_cnot(control: int, target: int) -> List[Tuple[str, Tuple[int, ...]]]:
        """Convert CZ to CNOT + H."""
        return [
            ('H', (target,)),
            ('CNOT', (control, target)),
            ('H', (target,)),
        ]

    @staticmethod
    def swap_to_cnot(q1: int, q2: int) -> List[Tuple[str, Tuple[int, ...]]]:
        """Convert SWAP to 3 CNOTs."""
        return [
            ('CNOT', (q1, q2)),
            ('CNOT', (q2, q1)),
            ('CNOT', (q1, q2)),
        ]

    @staticmethod
    def toffoli_to_clifford_t(c1: int, c2: int, target: int) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        Decompose Toffoli into Clifford+T gates.

        Standard decomposition with 7 T gates.
        """
        return [
            ('H', (target,)),
            ('CNOT', (c2, target)),
            ('Tdg', (target,)),
            ('CNOT', (c1, target)),
            ('T', (target,)),
            ('CNOT', (c2, target)),
            ('Tdg', (target,)),
            ('CNOT', (c1, target)),
            ('T', (c2,)),
            ('T', (target,)),
            ('H', (target,)),
            ('CNOT', (c1, c2)),
            ('T', (c1,)),
            ('Tdg', (c2,)),
            ('CNOT', (c1, c2)),
        ]


# =============================================================================
# Routing Utilities
# =============================================================================

class LinearRouter:
    """
    Route circuits on a linear qubit topology.

    Qubits are arranged: 0 - 1 - 2 - ... - (n-1)
    Only adjacent qubits can interact directly.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def route_cnot(self, control: int, target: int) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        Route a CNOT on linear topology.

        Returns sequence of native operations (CNOTs on adjacent qubits).
        """
        if abs(control - target) == 1:
            # Already adjacent
            return [('CNOT', (control, target))]

        if abs(control - target) == 0:
            raise ValueError("Control and target must be different")

        # Need to route through intermediate qubits
        operations = []

        # Move control towards target using SWAPs
        current_control = control
        direction = 1 if target > control else -1

        # SWAP control to be adjacent to target
        while abs(current_control - target) > 1:
            next_pos = current_control + direction
            # SWAP = 3 CNOTs
            operations.extend([
                ('CNOT', (current_control, next_pos)),
                ('CNOT', (next_pos, current_control)),
                ('CNOT', (current_control, next_pos)),
            ])
            current_control = next_pos

        # Execute the CNOT
        operations.append(('CNOT', (current_control, target)))

        # SWAP back (optional - depends on whether we need to preserve mapping)
        # Here we leave the mapping changed for simplicity

        return operations

    def routing_cost(self, control: int, target: int) -> int:
        """Return CNOT cost for routing."""
        distance = abs(control - target)
        if distance <= 1:
            return 1
        # Each intermediate step requires 3 CNOTs for SWAP + 1 final CNOT
        return 3 * (distance - 1) + 1


# =============================================================================
# Demo / Main
# =============================================================================

def demo_circuit_analysis():
    """Demonstrate circuit analysis capabilities."""

    print("=" * 70)
    print("Day 883: Logical Circuit Model - Demonstration")
    print("=" * 70)

    # Example 1: Simple Bell state circuit
    print("\n1. Bell State Circuit")
    print("-" * 40)

    bell = LogicalCircuit(2)
    bell.h(0)
    bell.cnot(0, 1)

    print(f"Circuit: {bell}")
    print(f"Gate counts: {bell.gate_count()}")
    print(f"T-count: {bell.t_count()}")
    print(f"Depth: {bell.depth()}")
    print(f"Critical path: {bell.critical_path()}")

    # Example 2: More complex circuit
    print("\n2. Complex Circuit with T gates")
    print("-" * 40)

    circuit = LogicalCircuit(3)
    circuit.h(0)
    circuit.h(1)
    circuit.t(0)
    circuit.cnot(0, 1)
    circuit.t(1)
    circuit.cnot(1, 2)
    circuit.toffoli(0, 1, 2)

    print(f"Circuit: {circuit}")
    print(f"Gate counts: {circuit.gate_count()}")
    print(f"T-count: {circuit.t_count()}")
    print(f"Depth: {circuit.depth()}")
    print(f"Critical path: {circuit.critical_path()}")
    print(f"Parallelism factor: {circuit.parallelism_factor():.2f}")

    # Example 3: Gate decomposition
    print("\n3. Toffoli Decomposition")
    print("-" * 40)

    toffoli_decomp = GateSetConverter.toffoli_to_clifford_t(0, 1, 2)
    print("Toffoli decomposes into:")
    for i, (gate, qubits) in enumerate(toffoli_decomp):
        print(f"  {i+1}. {gate}{qubits}")

    t_count = sum(1 for g, _ in toffoli_decomp if g in ['T', 'Tdg'])
    print(f"\nTotal T-count: {t_count}")

    # Example 4: Routing
    print("\n4. Linear Routing")
    print("-" * 40)

    router = LinearRouter(5)

    print("Routing CNOT(0, 3) on linear topology 0-1-2-3-4:")
    operations = router.route_cnot(0, 3)
    for i, (gate, qubits) in enumerate(operations):
        print(f"  {i+1}. {gate}{qubits}")
    print(f"Total CNOTs: {len(operations)}")

    # Example 5: Dependency graph
    print("\n5. Dependency Graph (DOT format)")
    print("-" * 40)

    small_circuit = LogicalCircuit(2)
    small_circuit.h(0)
    small_circuit.t(1)
    small_circuit.cnot(0, 1)
    small_circuit.h(1)

    print(small_circuit.dependency_graph_dot())

    # Example 6: T-count estimation for larger circuit
    print("\n6. T-count Scaling Analysis")
    print("-" * 40)

    for n in [10, 50, 100, 200]:
        circuit = LogicalCircuit(n)
        # Simulate a circuit with Toffoli-heavy computation
        for i in range(n - 2):
            circuit.toffoli(i, i+1, i+2)

        print(f"  n={n:3d}: T-count = {circuit.t_count():6d}, "
              f"depth = {circuit.depth():3d}, "
              f"parallelism = {circuit.parallelism_factor():.2f}")


if __name__ == "__main__":
    demo_circuit_analysis()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Solovay-Kitaev | $n_{\text{gates}} = O(\log^c(1/\epsilon))$, $c \approx 4$ |
| SWAP decomposition | $\text{SWAP} = \text{CNOT}^3$ |
| Circuit depth | $D = $ length of longest dependency path |
| Parallelism factor | $P = \text{total gates} / D$ |
| T-count dominance | $\text{Cost} \approx c_T \cdot n_T$ |

### Main Takeaways

1. **Quantum compilation** bridges five abstraction levels from algorithm to physical operations
2. **Universal gate sets** (especially Clifford+T) enable any quantum computation
3. **T-count** is the primary cost metric for fault-tolerant circuits
4. **Dependency graphs** reveal parallelism and critical paths
5. **Routing** may add significant overhead on constrained topologies
6. **Algorithm design** should be T-count aware from the start

---

## Daily Checklist

- [ ] I can describe the five levels of the quantum compilation stack
- [ ] I understand why Clifford+T is universal
- [ ] I can compute the T-count of a simple circuit
- [ ] I can construct a dependency DAG for a circuit
- [ ] I can identify the critical path and parallelism factor
- [ ] I understand the routing problem and SWAP networks

---

## Preview: Day 884

Tomorrow we dive deep into **Clifford+T decomposition**:

- The Clifford group and its efficient simulation
- The Solovay-Kitaev algorithm in detail
- Modern optimal synthesis: Ross-Selinger gridsynth
- T-count minimization techniques: TODD, T-par
- Practical synthesis for rotation angles

The T-gate decomposition is the bridge between abstract algorithms and fault-tolerant implementation.
