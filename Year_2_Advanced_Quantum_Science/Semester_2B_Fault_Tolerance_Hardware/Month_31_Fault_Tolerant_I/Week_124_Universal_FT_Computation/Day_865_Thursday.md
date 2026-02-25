# Day 865: Fault-Tolerant Circuit Compilation

## Week 124: Universal Fault-Tolerant Computation | Month 31: Fault-Tolerant QC I

---

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Logical to physical mapping |
| Afternoon | 2.5 hrs | FT gate implementations |
| Evening | 2.0 hrs | Optimization and scheduling |

---

### Learning Objectives

By the end of today, you will be able to:

1. **Describe the complete FT compilation pipeline** from algorithm to physical qubits
2. **Map logical circuits to encoded operations** in surface codes
3. **Implement fault-tolerant gates** using code surgery and magic state injection
4. **Apply circuit optimization passes** for depth and resource reduction
5. **Schedule parallel operations** respecting hardware constraints
6. **Analyze the overhead** of fault-tolerant compilation

---

### Core Content

#### Part 1: The Compilation Pipeline

**Complete FT Compilation Stack:**

```
┌─────────────────────────────────────────┐
│    Quantum Algorithm (high-level)       │
│    - Unitary operators, oracles         │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│    Gate Decomposition                   │
│    - Break into 1- and 2-qubit gates    │
│    - Standard gate set (Rx, Ry, Rz, CX) │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│    Clifford+T Synthesis                 │
│    - Replace rotations with Clifford+T  │
│    - Solovay-Kitaev or Gridsynth        │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│    Logical Circuit                      │
│    - All gates in {H, S, T, CNOT}       │
│    - Abstract logical qubits            │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│    Error Correction Encoding            │
│    - Choose code (surface, color, etc.) │
│    - Map logical qubits to patches      │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│    FT Gate Implementation               │
│    - Transversal gates (Cliffords)      │
│    - Magic state injection (T-gates)    │
│    - Lattice surgery (multi-qubit)      │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│    Physical Circuit                     │
│    - Syndrome measurement schedule      │
│    - Error correction rounds            │
│    - Hardware-native operations         │
└─────────────────────────────────────────┘
```

**Overhead at Each Level:**

| Stage | Overhead Type |
|-------|--------------|
| Gate decomposition | 3-4 gates per arbitrary 1-qubit gate |
| Clifford+T synthesis | $\sim 3\log_2(1/\epsilon)$ T-gates per rotation |
| Error encoding | $d^2$ physical qubits per logical qubit |
| FT gates | $O(d)$ rounds per logical gate |
| Magic states | Distillation factory overhead |

---

#### Part 2: Encoding into Surface Codes

**Surface Code Basics:**

A distance-$d$ surface code:
- Uses $d^2$ data qubits + $(d^2-1)$ syndrome qubits
- Encodes 1 logical qubit
- Corrects up to $\lfloor(d-1)/2\rfloor$ errors

**Logical Qubit Layout:**

```
    ●─Z─●─Z─●─Z─●
    │   │   │   │
    X   X   X   X
    │   │   │   │
    ●─Z─●─Z─●─Z─●
    │   │   │   │
    X   X   X   X
    │   │   │   │
    ●─Z─●─Z─●─Z─●
```

where ● = data qubit, X = X-stabilizer ancilla, Z = Z-stabilizer ancilla.

**Logical Operators:**

$$\bar{X} = \prod_{i \in \text{left edge}} X_i, \quad \bar{Z} = \prod_{i \in \text{top edge}} Z_i$$

**Code Distance Choice:**

Target logical error rate $p_L$ with physical error rate $p$:

$$p_L \approx 0.1 \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

For $p = 10^{-3}$, $p_{\text{th}} = 10^{-2}$:
- $d = 3$: $p_L \approx 10^{-2}$
- $d = 7$: $p_L \approx 10^{-5}$
- $d = 13$: $p_L \approx 10^{-9}$

---

#### Part 3: Fault-Tolerant Gate Implementation

**Transversal Gates (Clifford Group):**

In the surface code, Clifford gates are implemented transversally or via code deformations:

| Gate | Implementation | Time (code cycles) |
|------|----------------|-------------------|
| $\bar{X}$ | Transversal X on left edge | 1 |
| $\bar{Z}$ | Transversal Z on top edge | 1 |
| $\bar{H}$ | Code deformation (rotate lattice) | $O(d)$ |
| $\bar{S}$ | Inject $\|Y\rangle$ state | $O(d)$ |
| $\bar{\text{CNOT}}$ | Lattice surgery | $O(d)$ |

**Lattice Surgery for CNOT:**

Two logical patches perform CNOT via:

1. **Merge:** Combine Z-boundaries of control and target
2. **Measure:** Joint $\bar{Z} \otimes \bar{Z}$ stabilizer
3. **Split:** Separate patches with classical correction

```
Before:          During merge:           After:
┌───┐  ┌───┐    ┌───────────┐           ┌───┐  ┌───┐
│ C │  │ T │ →  │  C ═══ T  │      →    │ C │  │ T │
└───┘  └───┘    └───────────┘           └───┘  └───┘
                (d code cycles)
```

**Time Cost:** Each lattice surgery operation takes $\Theta(d)$ code cycles.

---

**T-Gate via Magic State Injection:**

The T-gate cannot be transversal. We use magic state injection:

**Protocol:**

1. **Prepare** magic state $|T\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$ via distillation
2. **Inject** using teleportation:

```
|ψ⟩_L ────●────────────────────── |T ψ⟩_L (after correction)
          │
|T⟩_L ────⊕────M_X────→ S^{m·†}
```

3. **Correct** based on measurement outcome $m$

**Resource Cost:**

- 1 magic state consumed per T-gate
- Lattice surgery operations: 2-3 per injection
- Total time: $O(d)$ code cycles

**Magic State Factory:**

Distillation factory produces magic states in parallel:

$$\boxed{R_{\text{factory}} = \frac{\text{T-gates per algorithm}}{\text{Runtime} / \text{Distillation time}}}$$

Factory size determines how many T-gates can run in parallel.

---

#### Part 4: Circuit-Level Compilation

**Step 1: Gate Counting and Scheduling**

For a logical circuit with:
- $n_q$ logical qubits
- $n_C$ Clifford gates
- $n_T$ T-gates
- $n_{\text{CNOT}}$ CNOT gates

**Step 2: T-Gate Scheduling**

T-gates are the bottleneck. Identify:
- **Independent T-gates:** Can execute in parallel
- **T-depth:** Minimum sequential layers of T-gates

**Example Circuit:**

```
q0: ─T─●───T───●───
      │       │
q1: ─T─⊕─T─T──⊕───
```

T-count = 5, but T-depth = 2 (first layer: 2 parallel T's, second layer: 3 T's... but q1 has sequential T's).

Actually: Layer 1: T on q0, T on q1 (parallel). Layer 2: T on q1. Layer 3: T on q0, T on q1.

Better analysis: T-depth = 3 for this circuit.

**Step 3: Resource Allocation**

Allocate:
- Logical qubit patches: $n_q \times d^2$ physical qubits
- Magic state factories: Sized to match T-throughput
- Routing space: For lattice surgery connections

---

#### Part 5: Optimization Passes

**Pass 1: Gate Cancellation**

Adjacent inverse gates cancel:
- $TT^\dagger = I$
- $SS^\dagger = I$
- $HH = I$
- $\text{CNOT} \cdot \text{CNOT} = I$

**Pass 2: Commutation Rules**

Reorder gates to expose cancellations:

| Commutation | Rule |
|-------------|------|
| T through CNOT (control) | $T_c \cdot \text{CNOT} = \text{CNOT} \cdot T_c$ |
| T through CNOT (target) | Does not commute |
| S through CNOT (control) | Commutes |
| Z through any | Commutes with Z-diagonal gates |

**Pass 3: T-Gate Reduction**

**Example: Toffoli Gate**

Naive decomposition: 7 T-gates
```
          ●───●───────●───────●───────●───────●───●
          │   │       │       │       │       │   │
          ●───┼───●───┼───●───┼───●───┼───●───┼───●
          │   │   │   │   │   │   │   │   │   │   │
      H───⊕───T───⊕───T†──⊕───T───⊕───T†──⊕───T───⊕───H
```

Optimized with ancillas: 4 T-gates + measurement

**Pass 4: Clifford Optimization**

Clifford gates are "free" (transversal), but minimize for depth:
- Merge adjacent Cliffords into single operation
- Use shortest Clifford decomposition

**Pass 5: Layout Optimization**

For surface codes, physical layout matters:
- Minimize routing distance for CNOT
- Place frequently interacting qubits adjacent
- Balance factory placement

---

#### Part 6: Scheduling and Timing

**Code Cycle Timing:**

One code cycle = one round of syndrome extraction:
- Measure all X and Z stabilizers
- Decode syndromes and track errors
- Duration: $\tau_{\text{cycle}} \approx 1 \mu s$ (typical superconducting)

**Gate Timing in Code Cycles:**

| Operation | Duration (code cycles) |
|-----------|----------------------|
| Pauli gate | 0 (software only) |
| H gate (code deformation) | $d$ |
| S gate (injection) | $d$ |
| T gate (injection) | $d + \tau_{\text{distill}}$ |
| CNOT (lattice surgery) | $2d$ |

**Critical Path Analysis:**

The runtime is determined by the longest path through the circuit, weighted by gate times.

$$\boxed{T_{\text{total}} = \max_{\text{paths}} \sum_{\text{gates } g \text{ on path}} \tau_g}$$

**Parallelization:**

- **Qubit-level:** Operations on different qubits can be parallel
- **Factory-level:** Multiple T-gates if factory has capacity
- **Spatial:** Non-overlapping lattice surgeries

---

### Algorithm Design Implications

**Compilation-Aware Algorithm Design:**

1. **Minimize T-count:** Use T-optimal subroutines
2. **Maximize T-parallelism:** Increase T-depth efficiency
3. **Reduce two-qubit gates:** CNOT requires lattice surgery
4. **Consider layout:** Design for efficient routing

**Example: QFT Compilation Strategy**

Quantum Fourier Transform on $n$ qubits:
- Naive: $O(n^2)$ controlled rotations
- Each rotation: $O(\log(1/\epsilon))$ T-gates
- Total: $O(n^2 \log(1/\epsilon))$ T-gates

Optimized QFT:
- Approximate QFT: Only keep $O(\log n)$ rotations per qubit
- Reduces to $O(n \log n \log(1/\epsilon))$ T-gates

---

### Worked Examples

#### Example 1: Complete Compilation of a Small Circuit

**Problem:** Compile the circuit $H \cdot \text{CNOT} \cdot (T \otimes I) \cdot \text{CNOT} \cdot H$ to fault-tolerant operations on surface code with $d=5$.

**Solution:**

**Step 1: Initial circuit (2 qubits)**
```
q0: ─H─●─T─●─H─
       │   │
q1: ───⊕───⊕───
```

**Step 2: Gate inventory**
- 2 H gates
- 2 CNOT gates
- 1 T gate

**Step 3: FT implementation**

| Gate | FT Operation | Time (cycles) |
|------|--------------|---------------|
| H on q0 | Code deformation | 5 |
| CNOT(q0, q1) | Lattice surgery | 10 |
| T on q0 | Magic state injection | 5 + factory |
| CNOT(q0, q1) | Lattice surgery | 10 |
| H on q0 | Code deformation | 5 |

**Step 4: Schedule**

Sequential execution (no parallelism):
- Total: $5 + 10 + 5 + 10 + 5 = 35$ code cycles
- Plus magic state availability time

**Step 5: Resource count**
- Qubits: $2 \times 5^2 = 50$ data qubits per logical qubit = 100 total
- Plus ancillas: $\approx 50$ per patch
- Plus factory: $\approx 1000$ physical qubits for $d=5$ factory
- **Total: $\approx 1300$ physical qubits**

---

#### Example 2: T-Gate Parallelism Analysis

**Problem:** Analyze T-parallelism in the following circuit:

```
q0: ─T─────────●─T─
               │
q1: ─T─●───────⊕───
       │
q2: ───⊕─T─T───────
```

**Solution:**

**Step 1: Identify dependencies**
- q0: T at time 0, then T after CNOT with q1
- q1: T at time 0, CNOT with q2, CNOT with q0
- q2: CNOT from q1, then T, T

**Step 2: T-depth layers**

Layer 1 (time 0): T on q0, T on q1 (parallel)
- After: CNOT(q1, q2)

Layer 2: T on q2
- After: T on q2, CNOT(q0, q1)

Layer 3: T on q0

**Step 3: Summary**
- T-count: 5
- T-depth: 3
- Maximum parallel T-gates: 2

**Factory requirement:** Need at least 2 T-states available at layer 1.

---

#### Example 3: Layout Optimization

**Problem:** Given 4 logical qubits with CNOT pattern as shown, optimize the 2D layout:

```
CNOT(q0, q1), CNOT(q1, q2), CNOT(q2, q3), CNOT(q0, q3)
```

**Solution:**

**Option A: Linear layout**
```
q0 ─ q1 ─ q2 ─ q3
```

Distances: (q0,q1)=1, (q1,q2)=1, (q2,q3)=1, (q0,q3)=3
Total routing cost: $1 + 1 + 1 + 3 = 6$

**Option B: Square layout**
```
q0 ─ q1
│     │
q3 ─ q2
```

Distances: (q0,q1)=1, (q1,q2)=1, (q2,q3)=1, (q0,q3)=1
Total routing cost: $1 + 1 + 1 + 1 = 4$

**Optimal:** Square layout reduces routing by 33%.

For surface codes, non-adjacent CNOTs require intermediate routing space, multiplying the overhead.

---

### Practice Problems

#### Level 1: Direct Application

**Problem 1.1:** Calculate the physical qubit count for 10 logical qubits in a distance-7 surface code, including ancillas.

**Problem 1.2:** A circuit has 20 T-gates with T-depth 5. If the magic state factory produces 2 T-states per cycle, what is the minimum T-gate-limited runtime in code cycles?

**Problem 1.3:** List the gates that can be applied transversally in the surface code.

---

#### Level 2: Intermediate

**Problem 2.1:** Design a lattice surgery schedule for the circuit:
```
q0: ─●───●─
     │   │
q1: ─⊕─●─│─
       │ │
q2: ───⊕─⊕─
```
Assume each surgery takes $d$ code cycles.

**Problem 2.2:** A Toffoli gate requires 7 T-gates in its standard decomposition. How many T-states must the factory produce during the algorithm's runtime if there are 100 Toffoli gates?

**Problem 2.3:** Prove that the commutation $[T_{\text{control}}, \text{CNOT}] = 0$ (T on control qubit commutes with CNOT).

---

#### Level 3: Challenging

**Problem 3.1:** **(Factory Sizing)**
Design a magic state factory for an algorithm with:
- 10,000 T-gates total
- T-depth 100
- Target runtime: 1 second
- Code cycle time: 1 microsecond
- Distillation time: 100 code cycles per batch of 15 raw states

How many parallel distillation units are needed?

**Problem 3.2:** **(Routing Optimization)**
For $n$ logical qubits arranged on a 2D grid with all-to-all CNOT connectivity required, analyze the routing overhead as a function of $n$ and $d$.

**Problem 3.3:** **(End-to-End Compilation)**
Compile Grover's algorithm for 3-qubit search to a fault-tolerant surface code implementation:
1. Start from the oracle and diffusion operators
2. Decompose to Clifford+T
3. Schedule on surface code patches
4. Estimate total physical resources

---

### Computational Lab

```python
"""
Day 865 Computational Lab: Fault-Tolerant Circuit Compilation
Complete pipeline from logical circuit to FT implementation
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

@dataclass
class LogicalGate:
    """Represents a logical gate"""
    name: str
    qubits: Tuple[int, ...]
    t_count: int = 0
    is_clifford: bool = True

    def __str__(self):
        return f"{self.name}({','.join(map(str, self.qubits))})"

@dataclass
class FTOperation:
    """Represents a fault-tolerant operation"""
    name: str
    qubits: List[int]
    duration_cycles: int
    resources: Dict[str, int]

class LogicalCircuit:
    """Logical quantum circuit representation"""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[LogicalGate] = []

    def add_gate(self, gate: LogicalGate):
        self.gates.append(gate)

    def h(self, q: int):
        self.add_gate(LogicalGate("H", (q,), t_count=0, is_clifford=True))

    def s(self, q: int):
        self.add_gate(LogicalGate("S", (q,), t_count=0, is_clifford=True))

    def t(self, q: int):
        self.add_gate(LogicalGate("T", (q,), t_count=1, is_clifford=False))

    def cnot(self, control: int, target: int):
        self.add_gate(LogicalGate("CNOT", (control, target), t_count=0, is_clifford=True))

    def toffoli(self, c1: int, c2: int, target: int):
        # Toffoli = 7 T-gates in standard decomposition
        self.add_gate(LogicalGate("Toffoli", (c1, c2, target), t_count=7, is_clifford=False))

    def rz(self, q: int, t_count: int):
        """Rz rotation approximated with given T-count"""
        self.add_gate(LogicalGate(f"Rz", (q,), t_count=t_count, is_clifford=False))

    def get_stats(self) -> Dict:
        """Get circuit statistics"""
        t_count = sum(g.t_count for g in self.gates)
        clifford_count = sum(1 for g in self.gates if g.is_clifford)
        non_clifford_count = len(self.gates) - clifford_count
        cnot_count = sum(1 for g in self.gates if g.name == "CNOT")

        return {
            'n_qubits': self.n_qubits,
            'total_gates': len(self.gates),
            't_count': t_count,
            'clifford_gates': clifford_count,
            'non_clifford_gates': non_clifford_count,
            'cnot_count': cnot_count,
        }

    def __str__(self):
        return " -> ".join(str(g) for g in self.gates)


class TDepthAnalyzer:
    """Analyze T-depth of a circuit"""

    def __init__(self, circuit: LogicalCircuit):
        self.circuit = circuit

    def compute_t_depth(self) -> Tuple[int, List[Set[int]]]:
        """
        Compute T-depth and T-layers.

        Returns:
            (t_depth, list of T-gate sets per layer)
        """
        # Track when each qubit is free
        qubit_time = {q: 0 for q in range(self.circuit.n_qubits)}
        t_layers = defaultdict(set)

        for i, gate in enumerate(self.circuit.gates):
            # Find earliest time this gate can execute
            start_time = max(qubit_time[q] for q in gate.qubits)

            # If T-gate, record layer
            if gate.t_count > 0:
                t_layers[start_time].add(i)

            # Update qubit availability
            end_time = start_time + (1 if gate.t_count > 0 else 0)
            for q in gate.qubits:
                qubit_time[q] = end_time

        t_depth = len(t_layers)
        layers = [t_layers[t] for t in sorted(t_layers.keys())]

        return t_depth, layers


class FTCompiler:
    """Fault-tolerant circuit compiler for surface codes"""

    def __init__(self, code_distance: int, cycle_time_us: float = 1.0):
        self.d = code_distance
        self.cycle_time = cycle_time_us  # microseconds

        # Physical qubit costs
        self.qubits_per_logical = 2 * code_distance ** 2  # data + ancilla

        # Gate durations in code cycles
        self.gate_durations = {
            'H': code_distance,
            'S': code_distance,
            'T': code_distance,  # injection only, + distillation
            'CNOT': 2 * code_distance,
            'Toffoli': 2 * code_distance,  # approximation
            'Rz': code_distance,
        }

    def compile(self, circuit: LogicalCircuit) -> Dict:
        """
        Compile logical circuit to FT resources.

        Returns:
            Dictionary of resource estimates
        """
        stats = circuit.get_stats()

        # Physical qubits for data
        data_qubits = stats['n_qubits'] * self.qubits_per_logical

        # Magic state factory sizing
        # Assume 15-to-1 distillation, need ~100*d^2 qubits per factory
        factory_qubits = 100 * self.d ** 2

        # Routing overhead (rough estimate: 50% for lattice surgery)
        routing_qubits = int(0.5 * data_qubits)

        total_physical_qubits = data_qubits + factory_qubits + routing_qubits

        # Timing analysis
        t_analyzer = TDepthAnalyzer(circuit)
        t_depth, t_layers = t_analyzer.compute_t_depth()

        # Gate-by-gate duration
        total_cycles = 0
        for gate in circuit.gates:
            base_name = gate.name.split('(')[0]
            duration = self.gate_durations.get(base_name, self.d)
            total_cycles += duration

        # Add distillation time for T-gates (pipelined, so only first batch)
        distillation_cycles = 10 * self.d  # rough estimate
        total_cycles += distillation_cycles

        runtime_us = total_cycles * self.cycle_time

        return {
            'logical_qubits': stats['n_qubits'],
            'code_distance': self.d,
            'physical_qubits': {
                'data': data_qubits,
                'factory': factory_qubits,
                'routing': routing_qubits,
                'total': total_physical_qubits,
            },
            't_count': stats['t_count'],
            't_depth': t_depth,
            'total_code_cycles': total_cycles,
            'runtime_us': runtime_us,
            'runtime_ms': runtime_us / 1000,
        }


class CircuitOptimizer:
    """Basic circuit optimization passes"""

    @staticmethod
    def cancel_adjacent_inverses(circuit: LogicalCircuit) -> LogicalCircuit:
        """Cancel adjacent inverse gates"""
        # Pairs that cancel: TT†, SS†, HH, CNOT·CNOT
        optimized = LogicalCircuit(circuit.n_qubits)
        skip_next = False

        for i, gate in enumerate(circuit.gates):
            if skip_next:
                skip_next = False
                continue

            if i + 1 < len(circuit.gates):
                next_gate = circuit.gates[i + 1]

                # Check for cancellation
                if gate.qubits == next_gate.qubits:
                    if (gate.name == "H" and next_gate.name == "H"):
                        skip_next = True
                        continue
                    if (gate.name == "CNOT" and next_gate.name == "CNOT"):
                        skip_next = True
                        continue

            optimized.add_gate(gate)

        return optimized

    @staticmethod
    def merge_rotations(circuit: LogicalCircuit) -> LogicalCircuit:
        """Merge adjacent single-qubit rotations"""
        # Simplified: just count potential merges
        merged = LogicalCircuit(circuit.n_qubits)
        pending_rz = {}  # qubit -> total t_count

        for gate in circuit.gates:
            if gate.name == "Rz" and len(gate.qubits) == 1:
                q = gate.qubits[0]
                pending_rz[q] = pending_rz.get(q, 0) + gate.t_count
            else:
                # Flush pending rotations
                for q, tc in pending_rz.items():
                    merged.add_gate(LogicalGate("Rz", (q,), t_count=tc, is_clifford=False))
                pending_rz = {}
                merged.add_gate(gate)

        # Final flush
        for q, tc in pending_rz.items():
            merged.add_gate(LogicalGate("Rz", (q,), t_count=tc, is_clifford=False))

        return merged


def visualize_compilation(circuit: LogicalCircuit, resources: Dict):
    """Visualize circuit and resource usage"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Resource breakdown
    ax1 = axes[0]
    phys = resources['physical_qubits']
    categories = ['Data\nQubits', 'Magic State\nFactory', 'Routing\nOverhead']
    values = [phys['data'], phys['factory'], phys['routing']]
    colors = ['steelblue', 'coral', 'mediumseagreen']

    bars = ax1.bar(categories, values, color=colors)
    ax1.set_ylabel('Physical Qubits')
    ax1.set_title(f"Physical Qubit Breakdown\nTotal: {phys['total']:,} qubits")

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom')

    # Plot 2: Circuit statistics
    ax2 = axes[1]
    stats = circuit.get_stats()
    metrics = ['T-count', 'T-depth', 'CNOTs', 'Code Cycles']
    values2 = [stats['t_count'], resources['t_depth'],
               stats['cnot_count'], resources['total_code_cycles']]

    bars2 = ax2.bar(metrics, values2, color=['crimson', 'darkorange', 'royalblue', 'purple'])
    ax2.set_ylabel('Count')
    ax2.set_title(f"Circuit Metrics\nRuntime: {resources['runtime_ms']:.2f} ms")
    ax2.set_yscale('log')

    for bar, val in zip(bars2, values2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('ft_compilation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


# Demonstrations
print("="*70)
print("Fault-Tolerant Circuit Compilation")
print("="*70)

# Example 1: Simple circuit
print("\n--- Example 1: Basic Circuit ---")
circuit1 = LogicalCircuit(2)
circuit1.h(0)
circuit1.cnot(0, 1)
circuit1.t(0)
circuit1.t(1)
circuit1.cnot(0, 1)
circuit1.h(0)

print(f"Circuit: {circuit1}")
print(f"Stats: {circuit1.get_stats()}")

compiler = FTCompiler(code_distance=7)
resources1 = compiler.compile(circuit1)

print(f"\nFT Compilation (d={resources1['code_distance']}):")
print(f"  Physical qubits: {resources1['physical_qubits']['total']:,}")
print(f"  T-count: {resources1['t_count']}")
print(f"  T-depth: {resources1['t_depth']}")
print(f"  Runtime: {resources1['runtime_ms']:.2f} ms")

# Example 2: Grover-like circuit
print("\n--- Example 2: Grover-style Circuit (3 qubits) ---")
circuit2 = LogicalCircuit(3)

# Initialize
for q in range(3):
    circuit2.h(q)

# Oracle (Toffoli-based)
circuit2.toffoli(0, 1, 2)

# Diffusion
for q in range(3):
    circuit2.h(q)
for q in range(3):
    circuit2.s(q)
circuit2.toffoli(0, 1, 2)
for q in range(3):
    circuit2.s(q)
for q in range(3):
    circuit2.h(q)

print(f"Stats: {circuit2.get_stats()}")

resources2 = compiler.compile(circuit2)
print(f"\nFT Compilation:")
print(f"  Physical qubits: {resources2['physical_qubits']['total']:,}")
print(f"  T-count: {resources2['t_count']}")
print(f"  Runtime: {resources2['runtime_ms']:.2f} ms")

# Example 3: QFT-like circuit with rotations
print("\n--- Example 3: QFT-style Circuit (4 qubits) ---")
circuit3 = LogicalCircuit(4)

# Approximate QFT rotations
for q in range(4):
    circuit3.h(q)
    for k in range(q+1, 4):
        # Controlled rotation with T-count based on precision
        precision_bits = 4 - (k - q)
        t_count = max(1, precision_bits * 3)  # ~3 T per bit of precision
        circuit3.rz(k, t_count)
        circuit3.cnot(q, k)

print(f"Stats: {circuit3.get_stats()}")

# Compile at different code distances
print("\nScaling with code distance:")
print(f"{'d':<5} {'Phys Qubits':<15} {'Runtime (ms)':<15}")
print("-" * 35)

for d in [3, 5, 7, 9, 11, 13]:
    comp = FTCompiler(code_distance=d)
    res = comp.compile(circuit3)
    print(f"{d:<5} {res['physical_qubits']['total']:<15,} {res['runtime_ms']:<15.2f}")

# Optimization demonstration
print("\n--- Optimization Pass Demo ---")
circuit4 = LogicalCircuit(2)
circuit4.h(0)
circuit4.h(0)  # Should cancel
circuit4.cnot(0, 1)
circuit4.cnot(0, 1)  # Should cancel
circuit4.t(0)
circuit4.s(0)

print(f"Before optimization: {len(circuit4.gates)} gates")
print(f"  {circuit4}")

optimized = CircuitOptimizer.cancel_adjacent_inverses(circuit4)
print(f"After optimization: {len(optimized.gates)} gates")
print(f"  {optimized}")

# T-depth analysis
print("\n--- T-Depth Analysis ---")
circuit5 = LogicalCircuit(3)
circuit5.t(0)
circuit5.t(1)
circuit5.cnot(0, 2)
circuit5.t(2)
circuit5.t(0)
circuit5.cnot(1, 2)
circuit5.t(1)

analyzer = TDepthAnalyzer(circuit5)
t_depth, layers = analyzer.compute_t_depth()

print(f"Circuit: {circuit5}")
print(f"T-count: {circuit5.get_stats()['t_count']}")
print(f"T-depth: {t_depth}")
print(f"T-layers: {[len(l) for l in layers]} gates per layer")
print(f"Maximum parallel T-gates: {max(len(l) for l in layers) if layers else 0}")

# Visualize example
visualize_compilation(circuit3, compiler.compile(circuit3))

# Summary table
print("\n" + "="*70)
print("Compilation Summary")
print("="*70)

circuits = [
    ("Basic (2q)", circuit1),
    ("Grover (3q)", circuit2),
    ("QFT-style (4q)", circuit3),
]

print(f"{'Circuit':<18} {'Logical Q':<10} {'T-count':<10} {'Phys Q':<12} {'Runtime':<10}")
print("-" * 60)

compiler = FTCompiler(code_distance=7)
for name, circ in circuits:
    stats = circ.get_stats()
    res = compiler.compile(circ)
    print(f"{name:<18} {stats['n_qubits']:<10} {res['t_count']:<10} "
          f"{res['physical_qubits']['total']:<12,} {res['runtime_ms']:<10.2f} ms")

print("\n" + "="*70)
print("FT Compilation Lab Complete")
print("="*70)
```

---

### Summary

#### Key Formulas

| Concept | Formula |
|---------|---------|
| Physical qubits per logical | $\approx 2d^2$ for surface code |
| Lattice surgery time | $O(d)$ code cycles |
| Magic state factory | $\approx 100d^2$ qubits |
| Total physical qubits | $n_q \cdot 2d^2 + \text{factory} + \text{routing}$ |
| Runtime | $\sum_g \tau_g$ (gate-limited) |

#### Main Takeaways

1. **FT compilation has massive overhead:** 1000s of physical qubits per logical qubit
2. **T-gates dominate resources:** Magic state distillation is the bottleneck
3. **Lattice surgery enables multi-qubit gates:** But at $O(d)$ cycle cost
4. **Optimization is essential:** Gate cancellation, rotation merging, layout optimization
5. **Code distance trades space for reliability:** Higher $d$ = more qubits, lower errors

---

### Daily Checklist

- [ ] Understand the complete FT compilation pipeline
- [ ] Can calculate physical qubit requirements for given logical circuit
- [ ] Understand lattice surgery for CNOT implementation
- [ ] Know how T-gates are implemented via magic state injection
- [ ] Can apply basic optimization passes
- [ ] Completed computational lab with FT compiler

---

### Preview: Day 866

Tomorrow we develop a complete **Resource Estimation Framework**---quantifying the total cost of fault-tolerant quantum algorithms. We'll analyze T-count, qubit count, and time overhead systematically, with detailed case studies of Shor's algorithm and quantum simulation. This provides the tools to answer: "How expensive is this quantum algorithm, really?"

---

*Day 865 shows how to compile circuits---tomorrow we learn to estimate what they cost.*
