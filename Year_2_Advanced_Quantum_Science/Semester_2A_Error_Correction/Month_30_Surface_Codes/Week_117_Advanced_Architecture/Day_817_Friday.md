# Day 817: Ancilla Design and Connectivity

## Week 117, Day 5 | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Overview

Ancilla qubits are the workhorses of quantum error correction—they extract syndrome information without disturbing the encoded logical state. Today we explore the design principles behind ancilla placement, connectivity requirements, and syndrome extraction circuits. We'll analyze how the choice between 4-way and 3-way connectivity affects circuit depth, gate count, and ultimately the error threshold. This is where abstract stabilizer theory meets concrete hardware implementation.

---

## Daily Schedule

| Session | Duration | Content |
|---------|----------|---------|
| Morning | 3 hours | Ancilla roles, connectivity requirements, syndrome circuits |
| Afternoon | 2 hours | Flag qubit protocols, circuit optimization |
| Evening | 2 hours | Python simulation of syndrome extraction circuits |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the role of ancilla qubits in syndrome extraction
2. **Design** syndrome extraction circuits for weight-4 stabilizers
3. **Compare** 4-way versus 3-way qubit connectivity requirements
4. **Implement** flag qubit protocols for fault tolerance
5. **Analyze** circuit depth and gate count for different ancilla layouts
6. **Optimize** syndrome extraction schedules to minimize error propagation

---

## Core Content

### 1. Ancilla Qubit Fundamentals

**Purpose of Ancilla Qubits:**
- Extract stabilizer eigenvalues (syndromes) without measuring data qubits
- Transfer quantum information to classical measurement outcomes
- Enable repeated error detection without destroying the logical state

**Basic Syndrome Extraction Protocol:**
1. Prepare ancilla in $|+\rangle$ (for X-stabilizer) or $|0\rangle$ (for Z-stabilizer)
2. Apply controlled operations between ancilla and data qubits
3. Measure ancilla in appropriate basis
4. Classical outcome reveals stabilizer eigenvalue

### 2. Syndrome Extraction Circuits

**For a Z-Stabilizer $S_Z = Z_1 Z_2 Z_3 Z_4$:**

```
|+⟩ ──●──●──●──●── H ── M
      |  |  |  |
D1 ───Z──┼──┼──┼───────
         |  |  |
D2 ──────Z──┼──┼───────
            |  |
D3 ─────────Z──┼───────
               |
D4 ────────────Z───────
```

Actually, for Z-stabilizers, we use CNOT with data as control:

```
|0⟩ ──────────────────── M
      ⊕  ⊕  ⊕  ⊕
D1 ───●──┼──┼──┼───────
         |  |  |
D2 ──────●──┼──┼───────
            |  |
D3 ─────────●──┼───────
               |
D4 ────────────●───────
```

**For an X-Stabilizer $S_X = X_1 X_2 X_3 X_4$:**

```
|+⟩ ──●──●──●──●── H ── M
      |  |  |  |
D1 ───X──┼──┼──┼───────
         |  |  |
D2 ──────X──┼──┼───────
            |  |
D3 ─────────X──┼───────
               |
D4 ────────────X───────
```

### 3. Gate Count and Circuit Depth

**Minimum Requirements for Weight-w Stabilizer:**
- Gates: $w$ two-qubit gates (CNOTs)
- Depth: At least $w$ if sequential, can be reduced with parallelization

**For Weight-4 Stabilizers:**
$$\boxed{\text{Minimum: 4 CNOTs, depth 4 (sequential)}}$$

### 4. Connectivity Requirements

**4-Way Connectivity (Square Lattice):**
```
     D1
      |
D4 ──A── D2
      |
     D3
```
Each ancilla (A) connects to 4 data qubits.

**Advantages:**
- Simple syndrome circuits
- Parallel syndrome extraction possible
- Standard surface code layout

**3-Way Connectivity (Heavy-Hex):**
```
D1 ── A ── D2
      |
     D3
```
Ancilla connects to only 3 neighbors directly.

**Challenge:** How to measure weight-4 stabilizers with 3-way connectivity?

### 5. Heavy-Hex Syndrome Extraction

IBM's heavy-hex architecture requires modified circuits:

**Option 1: Use Bridge Qubits**
Route syndrome information through intermediate qubits.

**Option 2: Measure in Stages**
1. Measure partial stabilizer (3 qubits)
2. Use classical information to reconstruct full syndrome

**Option 3: Flag Qubits**
Add extra ancilla qubits to detect hook errors.

### 6. Flag Qubit Protocols

**Problem:** A single fault during syndrome extraction can cause a weight-2 error on data qubits, which might be indistinguishable from two independent errors.

**Solution:** Flag qubits detect when dangerous faults occur.

**Flag Circuit for Weight-4 Stabilizer:**

```
|0⟩ ────────X──────X──────── M (Flag)
            |      |
|+⟩ ──●──●──●──●──●──●── H ── M (Syndrome)
      |  |     |  |
D1 ───Z──┼─────┼──┼─────────
         |     |  |
D2 ──────Z─────┼──┼─────────
               |  |
D3 ─────────────Z──┼─────────
                  |
D4 ────────────────Z─────────
```

Wait, let me draw this more carefully:

**Flag Protocol (Chao-Reichardt):**

For a weight-4 X-stabilizer:
```
Flag   |0⟩ ─────────────●───●─────────────── M
                        |   |
Ancilla|+⟩ ──●──●──────X───X──●──●── H ── M
             |  |              |  |
D1     ─────X──┼──────────────┼──┼─────────
                |              |  |
D2     ────────X──────────────┼──┼─────────
                              |  |
D3     ───────────────────────X──┼─────────
                                 |
D4     ──────────────────────────X─────────
```

**How Flags Work:**
1. If no error: Flag measures $|0\rangle$
2. If hook error (fault creates weight-2 data error): Flag measures $|1\rangle$
3. When flag is raised, decoder adjusts interpretation of syndrome

### 7. Parallel Syndrome Extraction

With 4-way connectivity, we can extract multiple syndromes simultaneously:

**Schedule for Non-overlapping Stabilizers:**
- Time 1: Measure all "even" X-stabilizers
- Time 2: Measure all "odd" X-stabilizers
- Time 3: Measure all "even" Z-stabilizers
- Time 4: Measure all "odd" Z-stabilizers

**Total depth per syndrome round:** 4 time steps

**With Overlapping Stabilizers:**
Adjacent stabilizers share data qubits, requiring careful scheduling to avoid conflicts.

### 8. CNOT Ordering and Hook Errors

The order of CNOTs in a syndrome circuit affects error propagation:

**Hook Error Example:**
If a fault occurs on the ancilla after the first two CNOTs:
```
|+⟩ ──●──●──[FAULT]──●──●── M
      |  |           |  |
D1 ───X──┼───────────┼──┼──  (error propagates here)
         |           |  |
D2 ──────X───────────┼──┼──  (error propagates here)
                     |  |
D3 ─────────────────X──┼──
                        |
D4 ──────────────────────X──
```

The fault causes errors on D1 and D2, creating a weight-2 error!

**Solution:** Order CNOTs to minimize correlated errors:
- Use "windmill" or "bow-tie" patterns
- Insert flag qubits at critical points

### 9. Syndrome Extraction Efficiency Metrics

**Key Metrics:**

1. **Gate count:** Total number of 2-qubit gates per syndrome bit
   - Minimum: weight of stabilizer (4 for surface code)

2. **Circuit depth:** Number of time steps
   - Critical for error accumulation during measurement

3. **Ancilla count:** Number of ancilla qubits needed
   - Standard: 1 per stabilizer
   - With flags: 1-2 additional per stabilizer

4. **Parallelism:** Fraction of operations done simultaneously
   - Higher parallelism → shorter total time

### 10. Optimized Circuit Designs

**Google's Approach:**
- Sycamore uses carefully scheduled CNOT ordering
- Minimize moments where ancilla carries partial syndrome
- Custom calibration for each qubit pair

**IBM's Approach:**
- Heavy-hex uses flag qubits extensively
- Echoed cross-resonance (ECR) gates
- Dynamical decoupling during idle periods

**General Optimization Principles:**
$$\boxed{\text{Minimize: } T_{\text{syndrome}} = \sum_i (t_{\text{gate},i} + t_{\text{idle},i})}$$

---

## Quantum Computing Connection

### Google Willow (2024) Syndrome Extraction

Google's Willow processor implements:
- 4-way connected qubits in a square grid
- Parallel syndrome extraction completing in ~1 μs
- Error rates below threshold for distance-3, 5, 7

**Key Innovation:** Careful CNOT ordering and leakage reduction.

### IBM's Flag Qubit Implementation

IBM Eagle/Heron processors use:
- 3-way connectivity requiring flag protocols
- Dynamical decoupling during syndrome extraction
- Real-time classical processing for flag interpretation

**Challenge:** Extra gates for flags increase total error rate.

### Trade-off Analysis

| Approach | Pros | Cons |
|----------|------|------|
| 4-way + parallel | Fast, simple circuits | Hard to fabricate |
| 3-way + flags | Easier fabrication | More gates, lower threshold |
| 3-way + staged | No extra qubits | More complex classical processing |

---

## Worked Examples

### Example 1: CNOT Count for Surface Code

**Problem:** For a distance-5 rotated surface code, calculate the total number of CNOTs per syndrome extraction round.

**Solution:**

**Stabilizer counts:**
- X-stabilizers: 12 (approximately $d^2/2$ in bulk + boundary)
- Z-stabilizers: 12 (similarly)
- Total stabilizers: 24

**CNOTs per stabilizer:**
- Bulk stabilizers (weight-4): 4 CNOTs each
- Boundary stabilizers (weight-2): 2 CNOTs each

Assume 8 bulk + 4 boundary for each type:
- X-stabilizers: $8 \times 4 + 4 \times 2 = 40$ CNOTs
- Z-stabilizers: $8 \times 4 + 4 \times 2 = 40$ CNOTs

**Total CNOTs per round:** 80

**With parallelization:** Circuit depth is ~4 (not 80) due to parallel execution.

---

### Example 2: Flag Qubit Overhead

**Problem:** A surface code uses flag qubits for all weight-4 stabilizers. Calculate the additional qubit overhead for a distance-7 code.

**Solution:**

**Stabilizer count:**
- Total stabilizers: $d^2 - 1 = 48$
- Bulk (weight-4): Approximately 32
- Boundary (weight-2): Approximately 16

**Flag qubits needed:**
- Weight-4 stabilizers need 1 flag each: 32 flags
- Weight-2 stabilizers don't need flags: 0 flags

**Qubit counts:**
- Data qubits: $d^2 = 49$
- Syndrome ancillas: 48
- Flag qubits: 32

**Total with flags:** 49 + 48 + 32 = 129

**Without flags:** 49 + 48 = 97

**Overhead from flags:**
$$\frac{129 - 97}{97} \times 100\% = 33\%$$

---

### Example 3: Syndrome Extraction Timing

**Problem:** Given a 2-qubit gate time of 30 ns and single-qubit gate time of 20 ns, calculate the minimum syndrome extraction time for a weight-4 stabilizer.

**Solution:**

**Circuit components:**
1. Ancilla initialization (1 single-qubit gate): 20 ns
2. Four CNOTs (sequential for simple analysis): $4 \times 30 = 120$ ns
3. Hadamard before measurement (if X-stabilizer): 20 ns
4. Measurement: ~300 ns (typical)

**Total sequential time:**
$$T = 20 + 120 + 20 + 300 = 460 \text{ ns}$$

**With parallel CNOTs (2 at a time):**
$$T = 20 + 60 + 20 + 300 = 400 \text{ ns}$$

**Minimum theoretical (fully parallel CNOTs):**
$$T = 20 + 30 + 20 + 300 = 370 \text{ ns}$$

---

## Practice Problems

### Direct Application

**Problem 1:** Draw the complete syndrome extraction circuit for a Z-type stabilizer $S_Z = Z_1 Z_2 Z_3 Z_4$ including ancilla preparation and measurement.

**Problem 2:** For a distance-9 surface code, how many ancilla qubits are needed? How many total physical qubits (data + ancilla)?

**Problem 3:** If each CNOT has a 0.5% error rate and we have 4 CNOTs per weight-4 stabilizer, what is the approximate probability that syndrome extraction introduces at least one error?

### Intermediate

**Problem 4:** Design a syndrome extraction schedule for a 2x2 patch of the rotated surface code that minimizes circuit depth by parallelizing non-conflicting measurements.

**Problem 5:** Explain why the order of CNOTs matters for error propagation. Give a specific example of a "hook error" and how a flag qubit detects it.

**Problem 6:** Compare the syndrome extraction circuit depth for a weight-6 hexagonal stabilizer versus a weight-4 square stabilizer. Which is more vulnerable to decoherence?

### Challenging

**Problem 7:** Prove that for a weight-$w$ stabilizer, the minimum circuit depth for syndrome extraction is $\lceil w/k \rceil$ where $k$ is the connectivity of the ancilla qubit.

**Problem 8:** Design a "bare ancilla" protocol (no flag qubits) that still achieves fault tolerance for a weight-4 stabilizer by using repeated measurements. How many rounds are needed?

**Problem 9:** In the presence of leakage errors (qubit leaves the computational subspace), how should syndrome extraction circuits be modified? Propose a leakage reduction unit (LRU) integration.

---

## Computational Lab

### Lab 817: Syndrome Extraction Circuit Simulation

```python
"""
Day 817 Computational Lab: Ancilla Design and Syndrome Extraction
==================================================================

This lab simulates syndrome extraction circuits, analyzes connectivity
requirements, and compares different ancilla layouts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Gate:
    """Represents a quantum gate in a circuit."""
    name: str
    qubits: Tuple[int, ...]
    time: int  # Time step when gate is applied

@dataclass
class SyndromeCircuit:
    """Represents a syndrome extraction circuit."""
    n_qubits: int
    gates: List[Gate]
    ancilla_idx: int
    data_indices: List[int]
    circuit_type: str  # "X" or "Z"

def create_z_stabilizer_circuit(data_qubits: List[int], ancilla: int) -> SyndromeCircuit:
    """
    Create a Z-stabilizer syndrome extraction circuit.

    For Z-stabilizer, ancilla starts in |0⟩, data qubits are controls.
    """
    gates = []
    n_qubits = max(data_qubits + [ancilla]) + 1

    # CNOTs with data as control, ancilla as target
    for t, d in enumerate(data_qubits):
        gates.append(Gate("CNOT", (d, ancilla), t))

    # Measurement at the end
    gates.append(Gate("MZ", (ancilla,), len(data_qubits)))

    return SyndromeCircuit(
        n_qubits=n_qubits,
        gates=gates,
        ancilla_idx=ancilla,
        data_indices=data_qubits,
        circuit_type="Z"
    )

def create_x_stabilizer_circuit(data_qubits: List[int], ancilla: int) -> SyndromeCircuit:
    """
    Create an X-stabilizer syndrome extraction circuit.

    For X-stabilizer, ancilla starts in |+⟩, ancilla is control.
    """
    gates = []
    n_qubits = max(data_qubits + [ancilla]) + 1

    # Hadamard to prepare |+⟩
    gates.append(Gate("H", (ancilla,), 0))

    # CNOTs with ancilla as control, data as target
    for t, d in enumerate(data_qubits):
        gates.append(Gate("CNOT", (ancilla, d), t + 1))

    # Hadamard before measurement
    gates.append(Gate("H", (ancilla,), len(data_qubits) + 1))

    # Measurement
    gates.append(Gate("MZ", (ancilla,), len(data_qubits) + 2))

    return SyndromeCircuit(
        n_qubits=n_qubits,
        gates=gates,
        ancilla_idx=ancilla,
        data_indices=data_qubits,
        circuit_type="X"
    )

def create_flagged_circuit(data_qubits: List[int], ancilla: int, flag: int) -> SyndromeCircuit:
    """
    Create a flagged syndrome extraction circuit.

    Flag qubit detects hook errors.
    """
    gates = []
    n_qubits = max(data_qubits + [ancilla, flag]) + 1

    # Prepare ancilla in |+⟩
    gates.append(Gate("H", (ancilla,), 0))

    # First two CNOTs
    gates.append(Gate("CNOT", (ancilla, data_qubits[0]), 1))
    gates.append(Gate("CNOT", (ancilla, data_qubits[1]), 2))

    # Flag CNOTs
    gates.append(Gate("CNOT", (ancilla, flag), 3))

    # Last two CNOTs
    gates.append(Gate("CNOT", (ancilla, data_qubits[2]), 4))
    gates.append(Gate("CNOT", (ancilla, data_qubits[3]), 5))

    # Second flag CNOT
    gates.append(Gate("CNOT", (ancilla, flag), 6))

    # Measurements
    gates.append(Gate("H", (ancilla,), 7))
    gates.append(Gate("MZ", (ancilla,), 8))
    gates.append(Gate("MZ", (flag,), 8))

    return SyndromeCircuit(
        n_qubits=n_qubits,
        gates=gates,
        ancilla_idx=ancilla,
        data_indices=data_qubits,
        circuit_type="X_flagged"
    )


def visualize_circuit(circuit: SyndromeCircuit, figsize=(14, 6)):
    """Visualize a syndrome extraction circuit."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    n_qubits = circuit.n_qubits
    max_time = max(g.time for g in circuit.gates) + 1

    # Draw qubit lines
    for i in range(n_qubits):
        y = n_qubits - i - 1
        ax.hlines(y, -0.5, max_time + 0.5, colors='gray', linewidth=1)

        # Label qubits
        if i == circuit.ancilla_idx:
            label = f"Ancilla ({i})"
            color = 'red'
        elif i in circuit.data_indices:
            label = f"Data {circuit.data_indices.index(i)+1} ({i})"
            color = 'blue'
        else:
            label = f"Flag ({i})"
            color = 'green'
        ax.text(-0.7, y, label, ha='right', va='center', fontsize=10, color=color)

    # Draw gates
    for gate in circuit.gates:
        t = gate.time
        if gate.name == "H":
            q = gate.qubits[0]
            y = n_qubits - q - 1
            rect = Rectangle((t - 0.2, y - 0.2), 0.4, 0.4,
                            facecolor='yellow', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(t, y, 'H', ha='center', va='center', fontsize=12, fontweight='bold')

        elif gate.name == "CNOT":
            control, target = gate.qubits
            y_c = n_qubits - control - 1
            y_t = n_qubits - target - 1

            # Control dot
            ax.plot(t, y_c, 'ko', markersize=10)

            # Target circle (XOR symbol)
            circle = Circle((t, y_t), 0.2, facecolor='white',
                           edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.plot([t - 0.15, t + 0.15], [y_t, y_t], 'k-', linewidth=2)
            ax.plot([t, t], [y_t - 0.15, y_t + 0.15], 'k-', linewidth=2)

            # Connecting line
            ax.plot([t, t], [min(y_c, y_t) + 0.2, max(y_c, y_t) - 0.2],
                   'k-', linewidth=2)

        elif gate.name == "MZ":
            q = gate.qubits[0]
            y = n_qubits - q - 1
            # Measurement symbol
            rect = Rectangle((t - 0.25, y - 0.25), 0.5, 0.5,
                            facecolor='lightgray', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(t, y, 'M', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(-1.5, max_time + 1)
    ax.set_ylim(-0.5, n_qubits + 0.5)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_title(f'Syndrome Extraction Circuit ({circuit.circuit_type})', fontsize=14)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig, ax


def analyze_hook_errors(circuit: SyndromeCircuit):
    """
    Analyze potential hook errors in a circuit.

    A hook error occurs when a single fault causes a weight-2 data error.
    """
    print(f"\nHook Error Analysis for {circuit.circuit_type} circuit")
    print("=" * 50)

    # Find CNOT gates
    cnots = [g for g in circuit.gates if g.name == "CNOT"]

    # Check for hook errors: fault on ancilla after first k CNOTs
    # causes errors on first k data qubits
    print("\nPotential hook errors (fault on ancilla after CNOT):")
    for i, cnot in enumerate(cnots):
        # Data qubits affected if fault occurs after this CNOT
        affected = []
        for prev_cnot in cnots[:i+1]:
            if circuit.circuit_type in ["X", "X_flagged"]:
                # Ancilla is control, so fault propagates to targets
                affected.append(prev_cnot.qubits[1])
            else:
                # Ancilla is target, fault stays on ancilla
                pass

        if len(affected) >= 2:
            print(f"  After CNOT {i+1}: Fault creates weight-{len(affected)} error on {affected}")

    return


def compare_connectivity_requirements():
    """Compare syndrome extraction for different connectivity levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 4-way connectivity
    ax1 = axes[0]
    ax1.set_title('4-Way Connectivity', fontsize=12)

    # Draw ancilla in center
    ax1.add_patch(Circle((2, 2), 0.3, facecolor='red', edgecolor='black', linewidth=2))
    ax1.annotate('A', (2, 2), ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    # Draw 4 data qubits
    data_pos = [(2, 3.5), (3.5, 2), (2, 0.5), (0.5, 2)]
    for i, pos in enumerate(data_pos):
        ax1.add_patch(Circle(pos, 0.3, facecolor='steelblue', edgecolor='black', linewidth=2))
        ax1.annotate(f'D{i+1}', pos, ha='center', va='center', fontsize=10, color='white')
        ax1.plot([2, pos[0]], [2, pos[1]], 'k-', linewidth=2)

    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_aspect('equal')
    ax1.text(2, -0.3, 'Direct access to all 4 data qubits', ha='center', fontsize=9)
    ax1.axis('off')

    # 3-way connectivity
    ax2 = axes[1]
    ax2.set_title('3-Way Connectivity', fontsize=12)

    ax2.add_patch(Circle((2, 2), 0.3, facecolor='red', edgecolor='black', linewidth=2))
    ax2.annotate('A', (2, 2), ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    # Only 3 direct connections
    data_pos_3 = [(2, 3.5), (3.5, 2), (0.5, 2)]
    for i, pos in enumerate(data_pos_3):
        ax2.add_patch(Circle(pos, 0.3, facecolor='steelblue', edgecolor='black', linewidth=2))
        ax2.annotate(f'D{i+1}', pos, ha='center', va='center', fontsize=10, color='white')
        ax2.plot([2, pos[0]], [2, pos[1]], 'k-', linewidth=2)

    # Fourth qubit needs routing
    ax2.add_patch(Circle((2, 0.5), 0.3, facecolor='steelblue', edgecolor='black', linewidth=2, alpha=0.5))
    ax2.annotate('D4', (2, 0.5), ha='center', va='center', fontsize=10, color='white')
    ax2.plot([2, 2], [2, 0.5], 'r--', linewidth=2)
    ax2.text(2.3, 1.25, '?', fontsize=16, color='red')

    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 4.5)
    ax2.set_aspect('equal')
    ax2.text(2, -0.3, 'Need bridge qubit or flag protocol', ha='center', fontsize=9)
    ax2.axis('off')

    # With flag qubit
    ax3 = axes[2]
    ax3.set_title('3-Way + Flag Qubit', fontsize=12)

    ax3.add_patch(Circle((2, 2), 0.3, facecolor='red', edgecolor='black', linewidth=2))
    ax3.annotate('A', (2, 2), ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    ax3.add_patch(Circle((2, 3), 0.2, facecolor='green', edgecolor='black', linewidth=2))
    ax3.annotate('F', (2, 3), ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax3.plot([2, 2], [2.3, 2.8], 'k-', linewidth=2)

    data_pos = [(0.5, 2), (3.5, 2), (1, 0.5), (3, 0.5)]
    for i, pos in enumerate(data_pos):
        ax3.add_patch(Circle(pos, 0.25, facecolor='steelblue', edgecolor='black', linewidth=2))
        ax3.annotate(f'D{i+1}', pos, ha='center', va='center', fontsize=9, color='white')

    ax3.set_xlim(-0.5, 4.5)
    ax3.set_ylim(-0.5, 4.5)
    ax3.set_aspect('equal')
    ax3.text(2, -0.3, 'Flag detects hook errors', ha='center', fontsize=9)
    ax3.axis('off')

    plt.tight_layout()
    return fig


def calculate_circuit_metrics():
    """Calculate and compare metrics for different circuit types."""
    print("\nCircuit Metrics Comparison")
    print("=" * 60)

    circuits = [
        ("Z-stabilizer (no flag)", create_z_stabilizer_circuit([0, 1, 2, 3], 4)),
        ("X-stabilizer (no flag)", create_x_stabilizer_circuit([0, 1, 2, 3], 4)),
        ("X-stabilizer (flagged)", create_flagged_circuit([0, 1, 2, 3], 4, 5)),
    ]

    print(f"{'Circuit Type':<25} {'CNOTs':<8} {'Depth':<8} {'Qubits':<8}")
    print("-" * 50)

    for name, circuit in circuits:
        cnots = sum(1 for g in circuit.gates if g.name == "CNOT")
        depth = max(g.time for g in circuit.gates) + 1
        qubits = circuit.n_qubits

        print(f"{name:<25} {cnots:<8} {depth:<8} {qubits:<8}")

    return circuits


def simulate_error_propagation(circuit: SyndromeCircuit, fault_location: int):
    """
    Simulate error propagation given a fault at a specific location.

    fault_location: index of gate after which fault occurs
    """
    print(f"\nSimulating fault after gate {fault_location}")

    cnots = [g for g in circuit.gates if g.name == "CNOT"]
    if fault_location >= len(cnots):
        print("Fault location beyond circuit")
        return

    # Assume X fault on ancilla
    ancilla = circuit.ancilla_idx
    affected_qubits = {ancilla: 'X'}

    # Propagate through remaining CNOTs
    for i, cnot in enumerate(cnots[fault_location:]):
        control, target = cnot.qubits

        if circuit.circuit_type in ["X", "X_flagged"]:
            # CNOT: control X -> target X (if control has X error)
            if control in affected_qubits and affected_qubits[control] == 'X':
                if target not in affected_qubits:
                    affected_qubits[target] = 'X'
        else:
            # Z-stabilizer: data is control
            # CNOT: target X -> control X (back-action)
            pass

    print(f"  Affected qubits: {affected_qubits}")
    data_errors = {k: v for k, v in affected_qubits.items() if k in circuit.data_indices}
    print(f"  Data qubit errors: {data_errors} (weight-{len(data_errors)})")

    return affected_qubits


# Main execution
if __name__ == "__main__":
    print("Creating syndrome extraction circuits...")

    # Create basic circuits
    z_circuit = create_z_stabilizer_circuit([0, 1, 2, 3], 4)
    x_circuit = create_x_stabilizer_circuit([0, 1, 2, 3], 4)
    flagged_circuit = create_flagged_circuit([0, 1, 2, 3], 4, 5)

    # Visualize circuits
    fig1, ax1 = visualize_circuit(z_circuit)
    plt.savefig('z_syndrome_circuit.png', dpi=150, bbox_inches='tight')
    print("Saved z_syndrome_circuit.png")

    fig2, ax2 = visualize_circuit(x_circuit)
    plt.savefig('x_syndrome_circuit.png', dpi=150, bbox_inches='tight')
    print("Saved x_syndrome_circuit.png")

    fig3, ax3 = visualize_circuit(flagged_circuit)
    plt.savefig('flagged_circuit.png', dpi=150, bbox_inches='tight')
    print("Saved flagged_circuit.png")

    # Analyze hook errors
    analyze_hook_errors(x_circuit)
    analyze_hook_errors(flagged_circuit)

    # Compare connectivity
    fig4 = compare_connectivity_requirements()
    plt.savefig('connectivity_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved connectivity_comparison.png")

    # Calculate metrics
    circuits = calculate_circuit_metrics()

    # Simulate error propagation
    print("\n" + "=" * 60)
    print("Error Propagation Analysis (X-stabilizer without flag)")
    print("=" * 60)
    for fault_loc in [0, 1, 2, 3]:
        simulate_error_propagation(x_circuit, fault_loc)

    plt.show()
```

### Lab Exercises

1. **Modify the flagged circuit** to use a different CNOT ordering and analyze how it affects hook error detection.

2. **Implement a parallel syndrome extraction** scheduler that minimizes circuit depth for a 3x3 surface code patch.

3. **Calculate the effective threshold** reduction due to syndrome extraction errors for different circuit designs.

4. **Design a leakage reduction unit (LRU)** that can be inserted between syndrome extraction rounds.

---

## Summary

### Key Formulas

| Metric | Value/Formula |
|--------|---------------|
| CNOTs per weight-$w$ stabilizer | $w$ minimum |
| Circuit depth (sequential) | $w + 2$ (with prep/measure) |
| Flag qubit overhead | ~1 per weight-4+ stabilizer |
| Hook error weight | Up to $\lfloor w/2 \rfloor$ |
| 4-way threshold | ~1% |
| 3-way + flags threshold | ~0.5-0.8% |

### Main Takeaways

1. **Ancillas extract syndromes:** They couple to data qubits via CNOTs to measure stabilizers.

2. **CNOT order matters:** Different orderings affect hook error vulnerability.

3. **Flag qubits detect hooks:** Extra ancillas identify when dangerous faults occur.

4. **4-way > 3-way:** Higher connectivity enables simpler, more robust circuits.

5. **Parallelization is key:** Minimizing circuit depth reduces decoherence effects.

---

## Daily Checklist

- [ ] I can design syndrome extraction circuits for X and Z stabilizers
- [ ] I understand the difference between 4-way and 3-way connectivity requirements
- [ ] I can explain what a hook error is and how flags detect it
- [ ] I can calculate circuit depth and gate count for syndrome extraction
- [ ] I have run the computational lab and analyzed error propagation

---

## Preview: Day 818

Tomorrow we tackle **Error Budgets and Code Distance Selection**. We'll discover:
- How to partition error budgets across gate types
- The $d = 2t + 1$ criterion for distance selection
- Threshold calculations and scaling analysis
- Practical distance selection for target logical error rates

Error budgeting connects physical error rates to logical performance.

---

*"The syndrome extraction circuit is the heartbeat of a surface code—its design determines whether the code lives or dies."*

— Day 817 Reflection
