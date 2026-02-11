# Day 826: Week 118 Synthesis - Lattice Surgery & Logical Gates

## Week 118: Lattice Surgery & Logical Gates | Month 30: Surface Codes

### Semester 2A: Error Correction | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Complete protocol review, compilation pipeline |
| **Afternoon** | 2.5 hours | End-to-end algorithm execution, resource estimation |
| **Evening** | 1.5 hours | Research frontiers, weekly assessment |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 826, you will be able to:

1. **Synthesize all lattice surgery primitives** into complete protocols
2. **Compile quantum circuits** to lattice surgery instruction sequences
3. **Execute end-to-end algorithms** on simulated surface code architectures
4. **Compare lattice surgery** with other fault-tolerant approaches
5. **Analyze research frontiers** in surface code computation
6. **Assess your mastery** of Week 118 material through comprehensive problems

---

## 1. Week 118 Review: The Complete Picture

### Lattice Surgery Toolkit

Over this week, we developed a complete toolkit for fault-tolerant quantum computation:

| Day | Topic | Key Operation |
|-----|-------|---------------|
| 820 | Fundamentals | Patch geometry, boundary types |
| 821 | Merge Operations | Joint ZZ/XX measurements |
| 822 | Split Operations | Separating patches, state prep |
| 823 | CNOT | Two-qubit entangling gate |
| 824 | Multi-Patch | Layouts, routing, scheduling |
| 825 | T-Gates | Magic states, distillation |

### The Universal Gate Set

With lattice surgery, we can implement:

**Clifford Gates:**
- $X$, $Y$, $Z$: Pauli frame (classical tracking)
- $H$: Patch rotation (exchange boundaries) or transversal
- $S$: Transversal or via surgery
- CNOT: ZZ merge + XX merge protocol

**Non-Clifford Gates:**
- $T$: Magic state injection + teleportation
- Arbitrary $R_z(\theta)$: Via magic state synthesis

$$\boxed{\{H, S, T, \text{CNOT}\} = \text{Universal Gate Set}}$$

---

## 2. The Lattice Surgery Compilation Pipeline

### From Algorithm to Physical Operations

```
┌─────────────────────────────────────────────────────────────┐
│                  COMPILATION PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Quantum    │────▶│   Clifford   │────▶│    Lattice       │
│   Circuit    │     │   + T        │     │    Surgery       │
│   (abstract) │     │   Decomp     │     │    Instructions  │
└──────────────┘     └──────────────┘     └──────────────────┘
                           │                      │
                           ▼                      ▼
                     ┌──────────────┐     ┌──────────────────┐
                     │    T-count   │     │     Physical     │
                     │   Optimizer  │     │     Layout       │
                     └──────────────┘     └──────────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────────┐
                                          │    Scheduled     │
                                          │    Operations    │
                                          └──────────────────┘
```

### Step 1: Circuit Decomposition

Convert arbitrary gates to Clifford + T:

**Single-qubit:** Any $U$ can be approximated:
$$U \approx H^{a_n} T^{b_n} H^{a_{n-1}} T^{b_{n-1}} \cdots H^{a_1} T^{b_1}$$

**Solovay-Kitaev:** $\epsilon$-approximation with $O(\log^c(1/\epsilon))$ gates

**Modern synthesis:** Optimal or near-optimal decompositions

### Step 2: T-Count Optimization

Minimize number of T gates (most expensive resource):

- **T-count:** Number of T gates in circuit
- **T-depth:** Maximum T gates on any path

**Optimization techniques:**
- Phase gadget synthesis
- Ancilla-based T-count reduction
- ZX-calculus simplification

### Step 3: Lattice Surgery Translation

Convert to surgery primitives:

| Gate | Surgery Implementation |
|------|----------------------|
| $X$, $Y$, $Z$ | Pauli frame update |
| $H$ | Boundary rotation |
| $S$ | $S = T^2$ or transversal |
| CNOT | ZZ merge → XX merge |
| $T$ | Magic state + teleportation |

### Step 4: Layout Assignment

Map logical qubits to physical patches:
- Assign patch positions
- Allocate ancilla regions
- Place magic state factories

### Step 5: Scheduling

Order operations to maximize parallelism while respecting:
- Data dependencies
- Patch availability
- Routing constraints

---

## 3. Complete Compilation Example: Toffoli Gate

### Target: Toffoli (CCNOT)

The Toffoli gate on qubits $a, b, c$:
$$\text{Toffoli}|a, b, c\rangle = |a, b, c \oplus (a \wedge b)\rangle$$

### Standard Decomposition

Toffoli = 6 CNOTs + 7 T gates (or T†):

```
a: ───────●───────────●───T───●───────●───T†──
          │           │       │       │
b: ───●───┼───●───T───┼───T†──┼───●───┼───●───T───
      │   │   │       │       │   │   │   │
c: ─H─X─T†┼─X─┼───T───X───────X─T†┼─X─┼─T─┼─X─H─
          │   │                   │   │   │
```

### T-Optimized Version

With ancilla, reduce to 4 T gates:

```
Using |CCZ⟩ resource state
```

### Lattice Surgery Translation

**Patches needed:**
- 3 data patches (a, b, c)
- 1 ancilla patch for CNOTs
- Magic state patches (for T gates)

**Operations sequence:**

1. CNOT(b, c): ZZ merge(b-anc) → XX merge(anc-c)
2. T†(c): Inject T† magic state
3. CNOT(a, c): ZZ merge(a-anc) → XX merge(anc-c)
4. T(c): Inject T magic state
5. ... (continue for all gates)

**Total surgery operations:**
- 6 CNOT × 2 merges = 12 merges
- 7 T gates = 7 magic state injections
- ~19 merge/inject operations

### Resource Summary

For distance $d$:

| Resource | Count |
|----------|-------|
| Data patches | 3 |
| Ancilla patches | 1-2 |
| Magic states | 7 |
| Merge operations | 12 |
| Total time | ~50d cycles |
| Physical qubits | ~10d² |

$$\boxed{\text{Toffoli: } \sim 50d \text{ cycles}, \sim 10d^2 \text{ qubits}}$$

---

## 4. End-to-End Algorithm: Quantum Phase Estimation

### The Algorithm

Quantum Phase Estimation (QPE) estimates eigenvalues of a unitary $U$:
- Input: Eigenstate $|u\rangle$ with $U|u\rangle = e^{2\pi i\phi}|u\rangle$
- Output: $n$-bit approximation of $\phi$

### Circuit Structure

```
|0⟩ ─H─────●─────────────────────────●───QFT†───M
|0⟩ ─H─────┼────●────────────────────┼───QFT†───M
|0⟩ ─H─────┼────┼────●───────────────┼───QFT†───M
...        │    │    │               │
|u⟩ ───────U────U²───U⁴───...───U^(2^n)────────────
```

### Compilation to Lattice Surgery

**Step 1: Decompose controlled-U**

Each controlled-$U^{2^k}$ decomposes to Clifford + T:
- If $U$ is Clifford: Easy (transversal or merge-based)
- If $U$ has T gates: Controlled-T implementation needed

**Step 2: Compile QFT†**

Inverse QFT = sequence of H gates and controlled-$R_z$ rotations:
$$R_z(\theta) \approx T^{a_1} H T^{a_2} H \cdots$$

**Step 3: Resource Estimation**

For $n$-bit precision QPE:

| Component | Operations |
|-----------|------------|
| Hadamards | $2n$ |
| Controlled-U | $2^n - 1$ applications |
| QFT† | $O(n^2)$ gates |

If $U$ has $t$ T-gates:
- Total T count: $O(2^n \cdot t + n^2)$
- Total CNOT count: $O(2^n + n^2)$

---

## 5. Resource Estimation Framework

### Space-Time Volume

The fundamental resource metric:

$$V = N_{\text{qubits}} \times T_{\text{cycles}}$$

### Complete Estimation Formula

For an algorithm with:
- $n$ logical qubits
- $g_C$ Clifford gates
- $g_T$ T gates
- Code distance $d$

**Physical qubits:**
$$N_{\text{phys}} = n \times 2d^2 \times f_{\text{routing}} + N_{\text{factory}}$$

where $f_{\text{routing}} \approx 2$ and $N_{\text{factory}} \approx 100 \times 30d^2$.

**Time:**
$$T = \max(g_T \times t_T / p, g_C \times t_C)$$

where $t_T \approx 10d$, $t_C \approx 3d$, and $p$ is T-gate parallelism.

**Space-time volume:**
$$V = N_{\text{phys}} \times T$$

$$\boxed{V \approx (2nd^2 + 3000d^2) \times \max(10d \cdot g_T, 3d \cdot g_C)}$$

### Example: 100-Qubit, 10⁶ T-Gate Algorithm

| Parameter | Value |
|-----------|-------|
| Logical qubits $n$ | 100 |
| T gates $g_T$ | 10⁶ |
| Code distance $d$ | 11 |
| T parallelism | 100 |

**Physical qubits:**
$$N = 100 \times 2 \times 121 \times 2 + 100 \times 30 \times 121 = 48,400 + 363,000 = 411,400$$

**Time:**
$$T = 10^6 \times 10 \times 11 / 100 = 1.1 \times 10^6 \text{ cycles}$$

At 1 MHz: 1.1 seconds

**Volume:**
$$V = 4 \times 10^5 \times 1.1 \times 10^6 = 4.4 \times 10^{11} \text{ qubit-cycles}$$

$$\boxed{\text{Example: } 400k \text{ qubits}, 1 \text{ second}, 10}^{11} \text{ qubit-cycles}}$$

---

## 6. Comparison with Other Approaches

### Lattice Surgery vs. Transversal Gates

| Aspect | Lattice Surgery | Transversal |
|--------|-----------------|-------------|
| Connectivity | Local (2D) | All-to-all |
| CNOT time | $O(d)$ | $O(1)$ |
| Code distance | Preserved | Preserved |
| Hardware fit | Excellent | Difficult |

**Verdict:** Lattice surgery wins for realistic 2D architectures.

### Lattice Surgery vs. Braiding (Defects)

| Aspect | Lattice Surgery | Defect Braiding |
|--------|-----------------|-----------------|
| Space overhead | Moderate | Lower |
| Scheduling | Simpler | Complex |
| Time overhead | Similar | Similar |
| Flexibility | High | High |

**Verdict:** Lattice surgery is easier to analyze; braiding can be more compact.

### Lattice Surgery vs. Code Concatenation

| Aspect | Lattice Surgery | Concatenated |
|--------|-----------------|--------------|
| Physical qubits | $O(d^2)$ | $O(d^{\log_2 k})$ |
| Threshold | ~1% | ~0.01-0.1% |
| Gate overhead | Moderate | High |
| Connectivity | 2D | Hierarchical |

**Verdict:** Surface codes dominate for near-term hardware with 2D connectivity.

---

## 7. Research Frontiers (2024-2026)

### Recent Advances

**1. Twist Defects**
- Reduce overhead for certain operations
- Enable "twist-based" CNOT without ancilla
- Active research area

**2. Floquet Codes**
- Dynamic stabilizer codes with time-varying measurements
- Potentially lower overhead
- Honeycomb code experiments

**3. LDPC Codes**
- Low-Density Parity Check codes
- Better encoding rate: $k/n > 0$
- Challenge: Non-local checks

**4. Hardware Demonstrations**
- Google: Distance-3 and distance-5 surface codes (2023)
- IBM: Error suppression in heavy-hex (2024)
- Quantinuum: Real-time decoding (2024)

### Open Problems

1. **Optimal T-factory design:** What is the minimum overhead for magic states?
2. **Decoder speed:** Can we decode fast enough for real-time correction?
3. **Code switching:** Efficiently convert between different code types?
4. **Algorithmic compilation:** Optimal mapping of algorithms to surgery?

### Industry Roadmaps

| Company | Target | Timeline |
|---------|--------|----------|
| Google | 1M physical qubits | 2029 |
| IBM | 100k qubits | 2033 |
| Microsoft | Topological qubits | TBD |
| IonQ | Networked modules | 2028 |

---

## 8. Weekly Assessment

### Comprehensive Problem Set

**Problem 1: Protocol Design (20 points)**

Design a lattice surgery protocol to implement the controlled-S gate:
$$CS = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes S$$

(a) Express CS in terms of CNOT and T gates.
(b) Convert to lattice surgery operations.
(c) Calculate the space-time volume for distance $d=7$.

---

**Problem 2: Resource Analysis (20 points)**

A quantum algorithm has:
- 50 logical qubits
- 10⁵ T gates
- 10⁶ Clifford gates
- Target logical error rate: 10⁻¹⁰

(a) Determine the required code distance.
(b) Calculate physical qubit count.
(c) Estimate runtime at 1 MHz cycle rate.
(d) Compare with classical simulation (assume 10⁹ operations/second).

---

**Problem 3: Compilation (25 points)**

Compile the following circuit to lattice surgery:

```
|ψ₁⟩ ─H──●──T──●──H──
         │     │
|ψ₂⟩ ────X──●──X─────
            │
|ψ₃⟩ ───────X────────
```

(a) List all operations in order.
(b) Identify which operations can be parallelized.
(c) Draw a space-time diagram.
(d) Calculate total time in syndrome cycles.

---

**Problem 4: Factory Design (20 points)**

Design a magic state factory that can sustain an algorithm requiring:
- 10⁴ T gates per second
- Physical error rate: 10⁻³
- Target T-state error: 10⁻¹²
- Cycle time: 1 μs

(a) How many distillation levels are needed?
(b) How many distillation units are required?
(c) What is the total factory footprint in physical qubits?
(d) Is the factory or the computation the bottleneck?

---

**Problem 5: Conceptual Understanding (15 points)**

Answer the following:

(a) Why can't T gates be implemented transversally on surface codes?
(b) What is the key difference between merge and split operations?
(c) Why does teleportation beat SWAP for routing in lattice surgery?
(d) How does the threshold theorem enable arbitrarily long computations?
(e) What is the main advantage of 2D surface codes over 3D color codes?

---

## 9. Computational Lab: Complete Lattice Surgery Compiler

```python
"""
Day 826 Computational Lab: Lattice Surgery Compiler
Complete compilation from quantum circuits to surgery operations

This lab implements a full compilation pipeline from abstract
quantum circuits to scheduled lattice surgery instructions.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class GateType(Enum):
    """Gate types for compilation."""
    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"
    H = "H"
    S = "S"
    T = "T"
    CNOT = "CNOT"
    CZ = "CZ"


@dataclass
class Gate:
    """Represents a quantum gate."""
    gate_type: GateType
    qubits: List[int]
    parameters: Optional[Dict] = None


@dataclass
class SurgeryOp:
    """Represents a lattice surgery operation."""
    op_type: str  # 'ZZ_MERGE', 'XX_MERGE', 'SPLIT', 'INJECT_T', 'PAULI_FRAME'
    patches: List[int]
    time_start: int
    duration: int
    outcome: Optional[int] = None


class QuantumCircuit:
    """Simple quantum circuit representation."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[Gate] = []

    def add_gate(self, gate_type: GateType, qubits: List[int], **params):
        """Add a gate to the circuit."""
        self.gates.append(Gate(gate_type, qubits, params if params else None))

    def h(self, qubit: int):
        self.add_gate(GateType.H, [qubit])

    def s(self, qubit: int):
        self.add_gate(GateType.S, [qubit])

    def t(self, qubit: int):
        self.add_gate(GateType.T, [qubit])

    def x(self, qubit: int):
        self.add_gate(GateType.X, [qubit])

    def z(self, qubit: int):
        self.add_gate(GateType.Z, [qubit])

    def cnot(self, control: int, target: int):
        self.add_gate(GateType.CNOT, [control, target])

    def cz(self, q1: int, q2: int):
        self.add_gate(GateType.CZ, [q1, q2])

    def t_count(self) -> int:
        """Count T gates in circuit."""
        return sum(1 for g in self.gates if g.gate_type == GateType.T)

    def cnot_count(self) -> int:
        """Count CNOT gates."""
        return sum(1 for g in self.gates if g.gate_type == GateType.CNOT)


class LatticeSurgeryCompiler:
    """
    Compiles quantum circuits to lattice surgery operations.
    """

    def __init__(self, n_qubits: int, distance: int = 5):
        """
        Initialize compiler.

        Parameters:
        -----------
        n_qubits : int
            Number of logical qubits
        distance : int
            Surface code distance
        """
        self.n_qubits = n_qubits
        self.d = distance

        # Patch allocation: data patches + ancilla patches
        self.n_data_patches = n_qubits
        self.n_ancilla_patches = max(2, n_qubits // 2)
        self.total_patches = self.n_data_patches + self.n_ancilla_patches

        # Timing
        self.current_time = 0
        self.operations: List[SurgeryOp] = []

        # Pauli frame tracking
        self.pauli_frame = {i: {'X': 0, 'Z': 0} for i in range(n_qubits)}

    def compile(self, circuit: QuantumCircuit) -> List[SurgeryOp]:
        """
        Compile a quantum circuit to surgery operations.

        Parameters:
        -----------
        circuit : QuantumCircuit
            Input circuit

        Returns:
        --------
        operations : List[SurgeryOp]
            Scheduled surgery operations
        """
        self.operations = []
        self.current_time = 0

        for gate in circuit.gates:
            self._compile_gate(gate)

        return self.operations

    def _compile_gate(self, gate: Gate):
        """Compile a single gate to surgery operations."""
        if gate.gate_type == GateType.X:
            self._compile_pauli(gate.qubits[0], 'X')

        elif gate.gate_type == GateType.Z:
            self._compile_pauli(gate.qubits[0], 'Z')

        elif gate.gate_type == GateType.Y:
            self._compile_pauli(gate.qubits[0], 'X')
            self._compile_pauli(gate.qubits[0], 'Z')

        elif gate.gate_type == GateType.H:
            self._compile_hadamard(gate.qubits[0])

        elif gate.gate_type == GateType.S:
            self._compile_s_gate(gate.qubits[0])

        elif gate.gate_type == GateType.T:
            self._compile_t_gate(gate.qubits[0])

        elif gate.gate_type == GateType.CNOT:
            self._compile_cnot(gate.qubits[0], gate.qubits[1])

        elif gate.gate_type == GateType.CZ:
            self._compile_cz(gate.qubits[0], gate.qubits[1])

    def _compile_pauli(self, qubit: int, pauli: str):
        """Compile Pauli gate (just frame update)."""
        op = SurgeryOp(
            op_type=f'PAULI_FRAME_{pauli}',
            patches=[qubit],
            time_start=self.current_time,
            duration=0
        )
        self.operations.append(op)
        self.pauli_frame[qubit][pauli] ^= 1

    def _compile_hadamard(self, qubit: int):
        """
        Compile Hadamard gate.

        In lattice surgery, H can be implemented by:
        1. Rotating the patch (exchange boundaries)
        2. Or via a sequence of merge operations

        For simplicity, we model it as a single operation.
        """
        op = SurgeryOp(
            op_type='HADAMARD',
            patches=[qubit],
            time_start=self.current_time,
            duration=self.d  # One syndrome round for boundary rotation
        )
        self.operations.append(op)
        self.current_time += self.d

    def _compile_s_gate(self, qubit: int):
        """
        Compile S gate.

        S = T^2, but can also be done transversally.
        Model as single operation.
        """
        op = SurgeryOp(
            op_type='S_GATE',
            patches=[qubit],
            time_start=self.current_time,
            duration=self.d
        )
        self.operations.append(op)
        self.current_time += self.d

    def _compile_t_gate(self, qubit: int):
        """
        Compile T gate via magic state injection.

        Protocol:
        1. ZZ merge with magic state patch
        2. XX measurement
        3. S correction (frame update)
        """
        # Allocate magic state patch
        magic_patch = self.n_data_patches  # First ancilla patch

        # ZZ merge
        op1 = SurgeryOp(
            op_type='ZZ_MERGE',
            patches=[qubit, magic_patch],
            time_start=self.current_time,
            duration=self.d
        )
        self.operations.append(op1)
        self.current_time += self.d

        # XX measurement (split)
        op2 = SurgeryOp(
            op_type='XX_SPLIT',
            patches=[qubit, magic_patch],
            time_start=self.current_time,
            duration=self.d
        )
        self.operations.append(op2)
        self.current_time += self.d

        # S correction is tracked in Pauli frame

    def _compile_cnot(self, control: int, target: int):
        """
        Compile CNOT via lattice surgery.

        Protocol:
        1. Prepare ancilla in |+⟩
        2. ZZ merge (control, ancilla)
        3. XX merge (ancilla, target)
        4. Pauli corrections based on outcomes
        """
        ancilla_patch = self.n_data_patches + 1  # Second ancilla patch

        # ZZ merge
        op1 = SurgeryOp(
            op_type='ZZ_MERGE',
            patches=[control, ancilla_patch],
            time_start=self.current_time,
            duration=self.d
        )
        self.operations.append(op1)
        self.current_time += self.d

        # XX merge
        op2 = SurgeryOp(
            op_type='XX_MERGE',
            patches=[ancilla_patch, target],
            time_start=self.current_time,
            duration=self.d
        )
        self.operations.append(op2)
        self.current_time += self.d

        # Split (implicit in XX merge)
        op3 = SurgeryOp(
            op_type='SPLIT',
            patches=[control, ancilla_patch, target],
            time_start=self.current_time,
            duration=self.d
        )
        self.operations.append(op3)
        self.current_time += self.d

    def _compile_cz(self, q1: int, q2: int):
        """Compile CZ = H₂ CNOT H₂."""
        self._compile_hadamard(q2)
        self._compile_cnot(q1, q2)
        self._compile_hadamard(q2)

    def get_statistics(self) -> Dict:
        """Return compilation statistics."""
        stats = {
            'total_operations': len(self.operations),
            'total_time': self.current_time,
            'zz_merges': sum(1 for op in self.operations if 'ZZ' in op.op_type),
            'xx_merges': sum(1 for op in self.operations if 'XX' in op.op_type),
            't_injections': sum(1 for op in self.operations
                              if op.op_type == 'ZZ_MERGE'
                              and any(p >= self.n_data_patches for p in op.patches)),
            'physical_qubits': self.total_patches * 2 * self.d**2
        }
        return stats


def visualize_compilation(compiler: LatticeSurgeryCompiler, title: str = "Surgery Schedule"):
    """
    Visualize the compiled surgery operations.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color map for operation types
    colors = {
        'ZZ_MERGE': 'blue',
        'XX_MERGE': 'red',
        'XX_SPLIT': 'orange',
        'SPLIT': 'green',
        'HADAMARD': 'purple',
        'S_GATE': 'cyan',
        'PAULI_FRAME_X': 'lightgray',
        'PAULI_FRAME_Z': 'lightgray'
    }

    # Plot operations
    for op in compiler.operations:
        if op.duration == 0:
            continue  # Skip frame updates

        color = colors.get(op.op_type, 'gray')

        for i, patch in enumerate(op.patches[:2]):  # Show up to 2 patches
            y = patch
            ax.barh(y, op.duration, left=op.time_start, height=0.8,
                   color=color, alpha=0.7, edgecolor='black')

            if i == 0:
                ax.text(op.time_start + op.duration/2, y,
                       op.op_type.replace('_', '\n'),
                       ha='center', va='center', fontsize=7)

    # Draw patch labels
    ax.set_yticks(range(compiler.total_patches))
    labels = [f'Q{i}' for i in range(compiler.n_data_patches)]
    labels += [f'Anc{i}' for i in range(compiler.n_ancilla_patches)]
    ax.set_yticklabels(labels)

    ax.set_xlabel('Time (syndrome cycles)')
    ax.set_ylabel('Patch')
    ax.set_title(title)
    ax.set_xlim(0, compiler.current_time + 1)

    # Legend
    handles = [plt.Rectangle((0,0), 1, 1, color=c, alpha=0.7)
              for c in ['blue', 'red', 'green', 'purple']]
    ax.legend(handles, ['ZZ Merge', 'XX Merge/Split', 'Split', 'Hadamard'],
             loc='upper right')

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    return fig


def compile_toffoli():
    """Compile and analyze Toffoli gate."""
    print("\n" + "="*60)
    print("TOFFOLI GATE COMPILATION")
    print("="*60)

    # Toffoli decomposition (simplified)
    circuit = QuantumCircuit(3)

    # Standard decomposition: CNOT + T sequence
    circuit.h(2)
    circuit.cnot(1, 2)
    circuit.t(2)  # T†
    circuit.cnot(0, 2)
    circuit.t(2)
    circuit.cnot(1, 2)
    circuit.t(2)  # T†
    circuit.cnot(0, 2)
    circuit.t(2)
    circuit.cnot(0, 1)
    circuit.t(1)  # T†
    circuit.cnot(0, 1)
    circuit.t(0)
    circuit.t(1)
    circuit.h(2)

    print(f"\nCircuit statistics:")
    print(f"  Qubits: {circuit.n_qubits}")
    print(f"  T gates: {circuit.t_count()}")
    print(f"  CNOT gates: {circuit.cnot_count()}")

    # Compile
    compiler = LatticeSurgeryCompiler(n_qubits=3, distance=5)
    operations = compiler.compile(circuit)

    stats = compiler.get_statistics()
    print(f"\nCompilation results:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Visualize
    fig = visualize_compilation(compiler, "Toffoli Gate - Lattice Surgery")
    plt.savefig('toffoli_compilation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nCompilation saved to 'toffoli_compilation.png'")


def compile_qft(n_qubits: int = 4):
    """Compile and analyze QFT circuit."""
    print("\n" + "="*60)
    print(f"QFT({n_qubits}) COMPILATION")
    print("="*60)

    circuit = QuantumCircuit(n_qubits)

    # Build QFT circuit
    for i in range(n_qubits):
        circuit.h(i)
        for j in range(i + 1, n_qubits):
            # Controlled R_z(π/2^(j-i))
            # Approximate with T gates
            circuit.cnot(j, i)
            for _ in range(2**(j - i - 1)):
                circuit.t(i)
            circuit.cnot(j, i)

    print(f"\nCircuit statistics:")
    print(f"  Qubits: {circuit.n_qubits}")
    print(f"  T gates: {circuit.t_count()}")
    print(f"  CNOT gates: {circuit.cnot_count()}")
    print(f"  Total gates: {len(circuit.gates)}")

    # Compile
    compiler = LatticeSurgeryCompiler(n_qubits=n_qubits, distance=5)
    operations = compiler.compile(circuit)

    stats = compiler.get_statistics()
    print(f"\nCompilation results:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return stats


def resource_scaling_analysis():
    """Analyze resource scaling for different circuits."""
    print("\n" + "="*60)
    print("RESOURCE SCALING ANALYSIS")
    print("="*60)

    # Different circuit sizes
    qft_sizes = [2, 3, 4, 5, 6]
    distances = [3, 5, 7, 9, 11]

    results = []

    for n in qft_sizes:
        circuit = QuantumCircuit(n)
        # Simple circuit: layer of H, then CNOTs, then Ts
        for i in range(n):
            circuit.h(i)
        for i in range(n - 1):
            circuit.cnot(i, i + 1)
        for i in range(n):
            circuit.t(i)

        for d in distances:
            compiler = LatticeSurgeryCompiler(n_qubits=n, distance=d)
            compiler.compile(circuit)
            stats = compiler.get_statistics()
            results.append({
                'n_qubits': n,
                'distance': d,
                **stats
            })

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Time vs qubits (for different distances)
    ax = axes[0, 0]
    for d in distances:
        subset = [r for r in results if r['distance'] == d]
        ax.plot([r['n_qubits'] for r in subset],
               [r['total_time'] for r in subset],
               'o-', label=f'd={d}')
    ax.set_xlabel('Logical Qubits')
    ax.set_ylabel('Total Time (cycles)')
    ax.set_title('Execution Time Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Physical qubits vs distance
    ax = axes[0, 1]
    for n in qft_sizes:
        subset = [r for r in results if r['n_qubits'] == n]
        ax.plot([r['distance'] for r in subset],
               [r['physical_qubits'] for r in subset],
               'o-', label=f'n={n}')
    ax.set_xlabel('Code Distance')
    ax.set_ylabel('Physical Qubits')
    ax.set_title('Physical Qubit Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Operations breakdown
    ax = axes[1, 0]
    n_fixed = 4
    d_fixed = 5
    r = [r for r in results if r['n_qubits'] == n_fixed and r['distance'] == d_fixed][0]

    ops = ['zz_merges', 'xx_merges', 't_injections']
    values = [r[op] for op in ops]
    ax.bar(ops, values, color=['blue', 'red', 'green'], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'Operations Breakdown (n={n_fixed}, d={d_fixed})')
    ax.grid(True, alpha=0.3)

    # Space-time volume
    ax = axes[1, 1]
    for d in [5, 7, 9]:
        subset = [r for r in results if r['distance'] == d]
        volumes = [r['physical_qubits'] * r['total_time'] for r in subset]
        ax.semilogy([r['n_qubits'] for r in subset], volumes, 'o-', label=f'd={d}')
    ax.set_xlabel('Logical Qubits')
    ax.set_ylabel('Space-Time Volume')
    ax.set_title('Space-Time Volume Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Lattice Surgery Resource Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('resource_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResource analysis saved to 'resource_scaling.png'")


def main():
    """Run all Day 826 demonstrations."""
    print("Day 826: Week 118 Synthesis")
    print("="*60)

    # Compile Toffoli
    compile_toffoli()

    # Compile QFT
    compile_qft(4)

    # Resource scaling
    resource_scaling_analysis()

    # Summary statistics
    print("\n" + "="*60)
    print("WEEK 118 SUMMARY")
    print("="*60)

    print("\nKey Operations Learned:")
    print("  1. Merge (ZZ, XX) - Joint Pauli measurements")
    print("  2. Split - Separating patches")
    print("  3. CNOT - ZZ merge + XX merge + corrections")
    print("  4. T-gate - Magic state injection + teleportation")

    print("\nKey Formulas:")
    print("  - CNOT time: 3d cycles")
    print("  - T-gate time: 2d cycles (+ factory wait)")
    print("  - Physical qubits: 2d² per logical qubit")
    print("  - Distillation: 15-to-1, p_out = 35p³")

    print("\nReady for:")
    print("  - Compiling arbitrary circuits to surgery")
    print("  - Designing patch layouts")
    print("  - Estimating algorithm resources")
    print("  - Understanding fault-tolerant QC architecture")

    print("\n" + "="*60)
    print("Day 826 / Week 118 Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 10. Summary

### Week 118 Key Concepts

| Day | Topic | Core Idea |
|-----|-------|-----------|
| 820 | Fundamentals | Patches encode qubits via stabilizers |
| 821 | Merge | Joint measurements create entanglement |
| 822 | Split | Separate patches while preserving info |
| 823 | CNOT | ZZ + XX protocol implements entangling gate |
| 824 | Architecture | Layout and scheduling for scalability |
| 825 | T-gates | Magic states enable universal computation |
| 826 | Synthesis | Complete compilation pipeline |

### Key Formulas Summary

| Quantity | Formula |
|----------|---------|
| Data qubits per patch | $2d^2$ |
| CNOT time | $3d$ cycles |
| T-gate time | $2d$ cycles |
| Distillation error | $p_{\text{out}} = 35p_{\text{in}}^3$ |
| Logical error rate | $P_L \sim (p/p_{\text{th}})^{(d+1)/2}$ |
| Space-time volume | $V = N_{\text{qubits}} \times T_{\text{cycles}}$ |

### Looking Ahead

Week 119 will cover **Decoders and Real-Time Error Correction**:
- Minimum-weight perfect matching
- Union-find decoders
- Neural network approaches
- Decoder speed requirements

---

## 11. Daily Checklist

- [ ] I can compile a quantum circuit to lattice surgery operations
- [ ] I understand the complete T-gate injection protocol
- [ ] I can estimate resources for a given algorithm
- [ ] I know the key differences between surgery and other approaches
- [ ] I am aware of current research frontiers
- [ ] I completed the weekly assessment problems
- [ ] I ran the compiler lab and analyzed results

---

## 12. Week 119 Preview

Next week: **Decoders and Real-Time Error Correction**

- Minimum-weight perfect matching (MWPM)
- Union-find decoder algorithms
- Machine learning approaches
- Hardware implementation challenges
- Latency requirements for real-time correction

Decoding is where theoretical error correction meets practical implementation - without fast, accurate decoders, surface codes remain theoretical constructs.

---

*"Lattice surgery transforms the surface code from a static error-correcting code into a dynamic computational substrate - the foundation of fault-tolerant quantum computing."*

---

## Week 118 Completion Certificate

**Topics Mastered:**
- [x] Surface code patch geometry and boundaries
- [x] Merge operations (ZZ and XX)
- [x] Split operations and state preparation
- [x] Fault-tolerant CNOT via lattice surgery
- [x] Multi-patch architectures and scheduling
- [x] T-gate injection and magic state distillation
- [x] Circuit compilation to surgery primitives

**Skills Developed:**
- Designing fault-tolerant gate protocols
- Resource estimation for quantum algorithms
- Understanding space-time trade-offs
- Connecting theory to hardware constraints

**Ready for:** Advanced topics in surface code implementation, decoding algorithms, and near-term quantum error correction experiments.
