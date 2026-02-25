# Day 801: Syndrome Extraction Circuits

## Month 29: Topological Codes | Week 115: Surface Code Implementation
### Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Ancilla-based syndrome measurement, CNOT orderings |
| **Afternoon** | 2.5 hours | Hook errors, mitigation strategies, problems |
| **Evening** | 1.5 hours | Computational lab: Circuit simulation |

**Total Study Time**: 7 hours

---

## Learning Objectives

By the end of Day 801, you will be able to:

1. **Design** syndrome extraction circuits for X and Z stabilizers
2. **Explain** the role of CNOT ordering in error propagation
3. **Identify** hook errors and their impact on code distance
4. **Implement** boundary stabilizer circuits (weight-2 and weight-3)
5. **Compare** different CNOT scheduling strategies
6. **Analyze** the depth-vs-error tradeoff in syndrome extraction

---

## Morning Session: Syndrome Extraction Theory (3 hours)

### 1. The Syndrome Measurement Problem

In a surface code, we must repeatedly measure all stabilizer generators without collapsing the encoded quantum state. This requires **indirect measurement** using ancilla qubits.

#### The Basic Principle

For a stabilizer $S = \prod_i P_i$ (where $P_i \in \{X, Z\}$), we:

1. Prepare an ancilla in $|+\rangle$ (for X-stabilizers) or $|0\rangle$ (for Z-stabilizers)
2. Apply controlled operations between ancilla and data qubits
3. Measure the ancilla in the appropriate basis
4. The measurement outcome reveals the syndrome bit

#### Why Indirect Measurement?

Direct measurement of multi-qubit Pauli operators would require:
- Non-local operations
- Precise joint measurements

Instead, we use local two-qubit gates (CNOTs) to transfer parity information to a single ancilla.

### 2. X-Stabilizer Measurement Circuit

For an X-stabilizer $S_X = X_1 X_2 X_3 X_4$ (weight-4):

```
Data q1:  ─────●─────────────────────
               │
Data q2:  ─────┼────●────────────────
               │    │
Data q3:  ─────┼────┼────●───────────
               │    │    │
Data q4:  ─────┼────┼────┼────●──────
               │    │    │    │
Ancilla:  ─|+⟩─X────X────X────X──|M⟩─
```

Wait—this is wrong. For X-stabilizers, we need CNOTs that propagate X from ancilla to data. Let me correct:

**Correct X-stabilizer circuit**:

```
Data q1:  ─────X─────────────────────
               │
Data q2:  ─────┼────X────────────────
               │    │
Data q3:  ─────┼────┼────X───────────
               │    │    │
Data q4:  ─────┼────┼────┼────X──────
               │    │    │    │
Ancilla:  ─|+⟩─●────●────●────●──|Mx⟩─
```

No, still not right. Let me be more careful about the circuit conventions.

#### Correct Protocol for X-Stabilizer

To measure $S_X = X_1 X_2 X_3 X_4$:

1. **Prepare** ancilla in $|0\rangle$
2. **Apply** Hadamard to ancilla: $|0\rangle \to |+\rangle$
3. **Apply** CNOT from ancilla (control) to each data qubit (target)
4. **Apply** Hadamard to ancilla
5. **Measure** ancilla in Z-basis

The circuit:

```
Data q1:  ───────────X─────────────────────────
                     │
Data q2:  ───────────┼────X────────────────────
                     │    │
Data q3:  ───────────┼────┼────X───────────────
                     │    │    │
Data q4:  ───────────┼────┼────┼────X──────────
                     │    │    │    │
Ancilla:  ─|0⟩──H────●────●────●────●────H──|Mz⟩─
```

**Analysis**:
- After H, ancilla is in $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$
- Each CNOT flips the target if ancilla is $|1\rangle$
- Net effect: $|+\rangle \to \frac{1}{\sqrt{2}}(|0\rangle + X_1 X_2 X_3 X_4 |1\rangle)$
- Final H and measurement gives eigenvalue of $X_1 X_2 X_3 X_4$

### 3. Z-Stabilizer Measurement Circuit

For a Z-stabilizer $S_Z = Z_1 Z_2 Z_3 Z_4$:

```
Data q1:  ─────●─────────────────────────────
               │
Data q2:  ─────┼────●────────────────────────
               │    │
Data q3:  ─────┼────┼────●───────────────────
               │    │    │
Data q4:  ─────┼────┼────┼────●──────────────
               │    │    │    │
Ancilla:  ─|0⟩─X────X────X────X──────────|Mz⟩─
```

**Analysis**:
- Ancilla starts in $|0\rangle$
- Each CNOT (data = control, ancilla = target) flips ancilla if data is $|1\rangle$
- Final ancilla state encodes parity $Z_1 Z_2 Z_3 Z_4$
- Measurement gives the syndrome bit

### 4. CNOT Ordering: The Critical Detail

The **order** in which CNOTs are applied matters enormously for fault tolerance.

#### The Problem: Error Propagation

Consider a faulty ancilla that experiences an X-error after the second CNOT in an X-stabilizer circuit:

```
Data q1:  ───────────X─────────────────────────────────
                     │
Data q2:  ───────────┼────X────────────────────────────
                     │    │  ↓ Error here
Data q3:  ───────────┼────┼────────X───────────────────
                     │    │        │
Data q4:  ───────────┼────┼────────┼────X──────────────
                     │    │        │    │
Ancilla:  ─|0⟩──H────●────●──[X]───●────●────H──|Mz⟩───
```

After the ancilla X-error:
- The subsequent CNOTs propagate this error to q3 and q4
- Result: Correlated X-errors on q3 and q4

This is a **hook error**: a single fault creates a weight-2 data error!

### 5. Hook Errors in Detail

#### Definition

A **hook error** occurs when:
1. A single fault (ancilla error or gate error) in syndrome extraction
2. Propagates through subsequent operations
3. Creates a multi-qubit correlated error on data qubits

#### Impact on Code Distance

If hook errors can create weight-2 errors, and the minimum-weight logical operator has weight $d$, then:
- Effective distance for hook errors: $d_{\text{eff}} = \lceil d/2 \rceil$

For $d=5$: A hook error (weight-2) + 2 more errors could complete a logical path!

$$\boxed{\text{Hook errors can reduce effective distance by } \sim 2\times}$$

### 6. Mitigating Hook Errors

#### Strategy 1: CNOT Ordering

Order CNOTs so that hook errors create errors that are **more detectable**.

**Optimal ordering for weight-4 X-stabilizer**:

```
      2───1
      │   │
      3───4

CNOT order: 1, 2, 3, 4 (diagonal pairing)
```

If an error occurs after CNOT 2:
- Propagates to qubits 3 and 4
- These form a "diagonal" pair
- Error is detected by two other stabilizers

#### Strategy 2: Flag Qubits

Add an extra "flag" ancilla that detects hook errors:

```
Data q1:  ────────X───────────────────────────
                  │
Data q2:  ────────┼──X────────────────────────
                  │  │
Flag:     ─|0⟩────┼──●──●─────────────────|Mz⟩
                  │     │
Data q3:  ────────┼─────X──X──────────────────
                  │        │
Data q4:  ────────┼────────┼──X───────────────
                  │        │  │
Ancilla:  ─|+⟩────●────────●──●────H──────|Mz⟩
```

The flag qubit signals if a mid-circuit error occurred.

#### Strategy 3: Repeated Syndrome Extraction

Measure each stabilizer multiple times:
- Single measurement: Can have faulty outcomes
- $d$ measurements: Majority voting gives reliable syndrome
- Time-distance matching: $d$ rounds for distance-$d$ protection

### 7. Boundary Stabilizer Circuits

At boundaries, stabilizers have reduced weight.

#### Weight-3 Stabilizer (Edge)

```
Data q1:  ─────────X─────────────────────
                   │
Data q2:  ─────────┼────X────────────────
                   │    │
Data q3:  ─────────┼────┼────X───────────
                   │    │    │
Ancilla:  ─|0⟩──H──●────●────●────H──|Mz⟩
```

Circuit depth: 3 CNOT layers (vs. 4 for weight-4)

#### Weight-2 Stabilizer (Corner)

```
Data q1:  ─────────X─────────────────
                   │
Data q2:  ─────────┼────X────────────
                   │    │
Ancilla:  ─|0⟩──H──●────●────H──|Mz⟩
```

Circuit depth: 2 CNOT layers

---

## Quantum Mechanics Connection

### Measurement-Based Quantum Error Correction

Syndrome extraction exemplifies **measurement-based** quantum computing principles:

1. **Weak measurement**: Ancilla couples weakly to data, extracting only parity information
2. **No wavefunction collapse**: The data remains in a code superposition
3. **Classical processing**: Syndrome bits fed to decoder

#### The Quantum-Classical Interface

Syndrome extraction is where quantum meets classical:

$$|\psi_{\text{encoded}}\rangle \xrightarrow{\text{syndrome extraction}} |\psi'\rangle \otimes |s_1 s_2 \cdots s_m\rangle_{\text{classical}}$$

The classical syndrome $s$ is used to infer errors without knowing $|\psi\rangle$.

### Fault Tolerance Requirements

For fault-tolerant operation:

1. **Single faults → single syndromes**: One error should trigger at most a few syndrome bits
2. **Error distinguishability**: Different errors should give different syndrome patterns
3. **Hook error control**: Multi-qubit propagation must not exceed half the distance

The surface code achieves all three with proper circuit design.

---

## Afternoon Session: Worked Examples (2.5 hours)

### Example 1: Complete Z-Stabilizer Circuit

**Problem**: Design the full circuit for a weight-4 Z-stabilizer with preparation, CNOTs, and measurement. Analyze error propagation.

**Solution**:

```
Circuit for Z-stabilizer S_Z = Z_1 Z_2 Z_3 Z_4:

     ┌───┐                         ┌───┐
q1: ─┤   ├────●────────────────────┤   ├─
     │   │    │                    │   │
q2: ─┤ I ├────┼────●───────────────┤ I ├─
     │   │    │    │               │   │
q3: ─┤   ├────┼────┼────●──────────┤   ├─
     │   │    │    │    │          │   │
q4: ─┤   ├────┼────┼────┼────●─────┤   ├─
     └───┘    │    │    │    │     └───┘
              │    │    │    │
anc: ─|0⟩─────X────X────X────X─────M_z──
```

**Error propagation analysis**:

1. **X-error on ancilla before any CNOT**:
   - Propagates nowhere (X on ancilla commutes with CNOT targets)
   - Only affects measurement outcome → syndrome error

2. **Z-error on ancilla**:
   - Propagates as Z through CNOT to all subsequent controls
   - After CNOT 2: Z on q1, q2, and ancilla
   - This is problematic!

Wait, let me reconsider. For CNOT with data as control:
- X on target (ancilla) stays on target
- Z on target propagates to control

So a Z-error on ancilla before CNOT 3:
- After CNOT 3: Z on ancilla propagates to q3
- After CNOT 4: Additional Z to q4
- Result: Z on q3, q4, and ancilla

This creates a weight-2 Z-error on data—a hook error!

**Mitigation**: Order CNOTs diagonally:

```
CNOT order: 1, 3, 2, 4 (or 1, 4, 2, 3)
```

Hook error now affects non-adjacent qubits, more detectable.

$$\boxed{\text{Diagonal CNOT ordering reduces hook error impact}}$$

### Example 2: Syndrome Round Timing

**Problem**: Calculate the circuit depth for one complete syndrome extraction round on a distance-5 surface code.

**Solution**:

In parallel, we can perform multiple stabilizer measurements if they don't share data qubits.

For the surface code:
- X-stabilizers and Z-stabilizers can be measured in parallel (they commute)
- But stabilizers of the same type that share qubits must be serialized

**Optimal scheduling**:

Layer 1: Prepare all ancillas
Layer 2: First CNOT for all stabilizers (NW corners)
Layer 3: Second CNOT (NE corners)
Layer 4: Third CNOT (SW corners)
Layer 5: Fourth CNOT (SE corners)
Layer 6: Measure all ancillas

Total: **6 layers** for weight-4 stabilizers

For the rotated code with careful scheduling:
- 4 CNOT layers (with parallel X and Z)
- Total: **6 layers** (prep, 4 CNOTs, measure)

For $d=5$ with 12 stabilizers: All measured in 6 parallel layers!

$$\boxed{\text{Syndrome extraction depth} = O(1) \text{ per round}}$$

### Example 3: Flag Qubit Overhead

**Problem**: Calculate the additional overhead for flag-qubit syndrome extraction on a distance-5 surface code.

**Solution**:

Standard approach:
- Data qubits: 13 (rotated $d=5$)
- Ancilla qubits: 12 (one per stabilizer)
- Total: 25 qubits

With flag qubits:
- Each weight-4 stabilizer needs 1 flag qubit
- Weight-4 stabilizers: ~8 (bulk)
- Flag qubits needed: 8
- New total: 33 qubits

Overhead: $\frac{33 - 25}{25} = 32\%$ increase

Alternative: Use $d$ syndrome rounds instead of flags
- No qubit overhead
- Time overhead: factor of $d$

$$\boxed{\text{Flag qubits: } 32\% \text{ qubit overhead; Alternative: } d\times \text{ time overhead}}$$

---

## Practice Problems

### Problem Set 801

#### Direct Application

1. **Circuit drawing**: Draw the complete syndrome extraction circuit for a weight-3 X-stabilizer at a smooth boundary. Include ancilla preparation and measurement.

2. **Error propagation**: An X-error occurs on the ancilla after the 2nd CNOT of a 4-CNOT Z-stabilizer circuit. Trace the error through the circuit and identify which data qubits are affected.

3. **Parallel scheduling**: For a distance-3 surface code with 4 X-stabilizers and 4 Z-stabilizers, design a parallel CNOT schedule that completes all measurements in minimum depth.

#### Intermediate

4. **Hook error analysis**: For a weight-4 Z-stabilizer with linear CNOT ordering (1,2,3,4), calculate the probability that a hook error is created given single-qubit error rate $p$.

5. **Flag circuit design**: Design a flag-qubit circuit for a weight-4 X-stabilizer. Verify that the flag is triggered if and only if a hook error occurs.

6. **Measurement errors**: If ancilla measurement has error rate $p_m$, how many syndrome rounds are needed to achieve effective measurement error rate below $p_m^2$?

#### Challenging

7. **Optimal CNOT ordering**: For a $d=7$ surface code, design a CNOT ordering scheme that minimizes the maximum hook error weight across all stabilizers.

8. **Syndrome extraction depth**: Prove that the surface code syndrome extraction can be performed in $O(1)$ depth regardless of code distance (assuming parallel operations on non-overlapping qubits).

9. **Correlated errors**: Analyze syndrome extraction when two-qubit gate errors are correlated (e.g., both qubits experience errors with some probability). How does this affect the hook error model?

---

## Evening Session: Computational Lab (1.5 hours)

### Lab 801: Syndrome Extraction Circuit Simulation

```python
"""
Day 801 Computational Lab: Syndrome Extraction Circuits
Simulation of stabilizer measurements and hook errors
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class Pauli(Enum):
    """Pauli operators."""
    I = 0
    X = 1
    Y = 2
    Z = 3


@dataclass
class StabilizerCircuit:
    """
    Represents a syndrome extraction circuit for one stabilizer.

    Attributes:
        stab_type: 'X' or 'Z' stabilizer
        data_qubits: Indices of data qubits in the stabilizer
        ancilla: Index of the ancilla qubit
        cnot_order: Order of CNOT operations (indices into data_qubits)
    """
    stab_type: str
    data_qubits: List[int]
    ancilla: int
    cnot_order: Optional[List[int]] = None

    def __post_init__(self):
        if self.cnot_order is None:
            # Default: sequential order
            self.cnot_order = list(range(len(self.data_qubits)))


class SyndromeExtractor:
    """
    Simulates syndrome extraction with error injection.
    """

    def __init__(self, n_data: int, circuits: List[StabilizerCircuit]):
        """
        Initialize the syndrome extractor.

        Parameters:
            n_data: Number of data qubits
            circuits: List of stabilizer circuits
        """
        self.n_data = n_data
        self.circuits = circuits
        self.n_ancilla = len(circuits)

        # Track errors (Pauli frame)
        self.data_errors = [Pauli.I] * n_data
        self.ancilla_errors = [Pauli.I] * self.n_ancilla

    def reset(self):
        """Reset all errors."""
        self.data_errors = [Pauli.I] * self.n_data
        self.ancilla_errors = [Pauli.I] * self.n_ancilla

    def inject_data_error(self, qubit: int, pauli: Pauli):
        """Inject an error on a data qubit."""
        self.data_errors[qubit] = self._compose_pauli(
            self.data_errors[qubit], pauli
        )

    def inject_ancilla_error(self, ancilla: int, pauli: Pauli, after_cnot: int):
        """
        Inject an error on an ancilla after a specific CNOT.

        This simulates mid-circuit errors that can cause hook errors.
        """
        # Store for later processing during circuit simulation
        if not hasattr(self, 'pending_ancilla_errors'):
            self.pending_ancilla_errors = []
        self.pending_ancilla_errors.append((ancilla, pauli, after_cnot))

    def _compose_pauli(self, p1: Pauli, p2: Pauli) -> Pauli:
        """Compose two Pauli operators (ignoring phase)."""
        if p1 == Pauli.I:
            return p2
        if p2 == Pauli.I:
            return p1
        if p1 == p2:
            return Pauli.I
        # XY=Z, XZ=Y, YZ=X (up to phase)
        paulis = {Pauli.X, Pauli.Y, Pauli.Z}
        return (paulis - {p1, p2}).pop()

    def simulate_cnot_x_stab(self, circuit: StabilizerCircuit):
        """
        Simulate CNOTs for an X-stabilizer.

        For X-stabilizer: ancilla is control, data is target.
        X on control → X on target
        Z on target → Z on control
        """
        anc_idx = self.circuits.index(circuit)

        for i, data_idx in enumerate([circuit.data_qubits[j] for j in circuit.cnot_order]):
            # Check for pending errors
            if hasattr(self, 'pending_ancilla_errors'):
                for (a, p, after) in self.pending_ancilla_errors:
                    if a == anc_idx and after == i:
                        self.ancilla_errors[anc_idx] = self._compose_pauli(
                            self.ancilla_errors[anc_idx], p
                        )

            # Error propagation through CNOT (ancilla=control, data=target)
            anc_err = self.ancilla_errors[anc_idx]
            data_err = self.data_errors[data_idx]

            # X on control propagates to target
            if anc_err in [Pauli.X, Pauli.Y]:
                self.data_errors[data_idx] = self._compose_pauli(data_err, Pauli.X)

            # Z on target propagates to control
            if data_err in [Pauli.Z, Pauli.Y]:
                self.ancilla_errors[anc_idx] = self._compose_pauli(anc_err, Pauli.Z)

    def simulate_cnot_z_stab(self, circuit: StabilizerCircuit):
        """
        Simulate CNOTs for a Z-stabilizer.

        For Z-stabilizer: data is control, ancilla is target.
        X on control → X on target
        Z on target → Z on control
        """
        anc_idx = self.circuits.index(circuit)

        for i, data_idx in enumerate([circuit.data_qubits[j] for j in circuit.cnot_order]):
            # Check for pending errors
            if hasattr(self, 'pending_ancilla_errors'):
                for (a, p, after) in self.pending_ancilla_errors:
                    if a == anc_idx and after == i:
                        self.ancilla_errors[anc_idx] = self._compose_pauli(
                            self.ancilla_errors[anc_idx], p
                        )

            # Error propagation through CNOT (data=control, ancilla=target)
            anc_err = self.ancilla_errors[anc_idx]
            data_err = self.data_errors[data_idx]

            # X on control propagates to target
            if data_err in [Pauli.X, Pauli.Y]:
                self.ancilla_errors[anc_idx] = self._compose_pauli(anc_err, Pauli.X)

            # Z on target propagates to control
            if anc_err in [Pauli.Z, Pauli.Y]:
                self.data_errors[data_idx] = self._compose_pauli(data_err, Pauli.Z)

    def run_syndrome_extraction(self) -> Dict:
        """
        Run one round of syndrome extraction.

        Returns dictionary with:
        - 'syndromes': List of syndrome bits
        - 'data_errors': Final data errors
        - 'hook_errors': List of (ancilla, affected_qubits) for hook errors
        """
        hook_errors = []

        for i, circuit in enumerate(self.circuits):
            initial_data = self.data_errors.copy()

            if circuit.stab_type == 'X':
                self.simulate_cnot_x_stab(circuit)
            else:
                self.simulate_cnot_z_stab(circuit)

            # Check for hook errors (multiple qubits changed)
            changed = [j for j in range(self.n_data)
                       if self.data_errors[j] != initial_data[j]]
            if len(changed) > 1:
                hook_errors.append((i, changed))

        # Calculate syndromes (simplified - just parity of relevant errors)
        syndromes = []
        for circuit in self.circuits:
            if circuit.stab_type == 'X':
                # X-stabilizer detects Z-errors
                parity = sum(1 for q in circuit.data_qubits
                             if self.data_errors[q] in [Pauli.Z, Pauli.Y]) % 2
            else:
                # Z-stabilizer detects X-errors
                parity = sum(1 for q in circuit.data_qubits
                             if self.data_errors[q] in [Pauli.X, Pauli.Y]) % 2
            syndromes.append(parity)

        # Clear pending errors
        if hasattr(self, 'pending_ancilla_errors'):
            self.pending_ancilla_errors = []

        return {
            'syndromes': syndromes,
            'data_errors': self.data_errors.copy(),
            'hook_errors': hook_errors
        }


def demo_hook_error():
    """Demonstrate hook error occurrence."""
    print("\n" + "="*60)
    print("Hook Error Demonstration")
    print("="*60)

    # Simple 4-qubit example with one X-stabilizer
    circuits = [
        StabilizerCircuit(
            stab_type='X',
            data_qubits=[0, 1, 2, 3],
            ancilla=0,
            cnot_order=[0, 1, 2, 3]  # Linear ordering
        )
    ]

    extractor = SyndromeExtractor(n_data=4, circuits=circuits)

    # Inject X-error on ancilla after 2nd CNOT
    extractor.inject_ancilla_error(ancilla=0, pauli=Pauli.X, after_cnot=2)

    result = extractor.run_syndrome_extraction()

    print("\nScenario: X-error on ancilla after 2nd CNOT (linear order)")
    print(f"CNOT order: [0, 1, 2, 3]")
    print(f"\nData errors after extraction:")
    for i, err in enumerate(result['data_errors']):
        print(f"  Qubit {i}: {err.name}")

    print(f"\nHook errors detected: {result['hook_errors']}")
    print(f"Syndrome: {result['syndromes']}")

    # Now try with diagonal ordering
    print("\n" + "-"*40)
    print("With diagonal CNOT ordering:")

    circuits_diag = [
        StabilizerCircuit(
            stab_type='X',
            data_qubits=[0, 1, 2, 3],
            ancilla=0,
            cnot_order=[0, 3, 1, 2]  # Diagonal ordering
        )
    ]

    extractor2 = SyndromeExtractor(n_data=4, circuits=circuits_diag)
    extractor2.inject_ancilla_error(ancilla=0, pauli=Pauli.X, after_cnot=2)

    result2 = extractor2.run_syndrome_extraction()

    print(f"CNOT order: [0, 3, 1, 2]")
    print(f"\nData errors after extraction:")
    for i, err in enumerate(result2['data_errors']):
        print(f"  Qubit {i}: {err.name}")

    print(f"\nHook errors: {result2['hook_errors']}")
    print("(Diagonal ordering spreads hook error to non-adjacent qubits)")


def visualize_cnot_schedule():
    """Visualize CNOT scheduling for surface code syndrome extraction."""
    print("\n" + "="*60)
    print("CNOT Schedule Visualization")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Weight-4 stabilizer CNOT orderings
    ax1 = axes[0]

    # Draw the stabilizer (plaquette)
    positions = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}

    for i, (x, y) in positions.items():
        ax1.scatter(x, y, s=400, c='black', zorder=5)
        ax1.annotate(f'q{i}', (x, y), ha='center', va='center',
                     color='white', fontsize=12, fontweight='bold', zorder=6)

    # Ancilla at center
    ax1.scatter(0.5, 0.5, s=600, c='blue', marker='s', zorder=5)
    ax1.annotate('A', (0.5, 0.5), ha='center', va='center',
                 color='white', fontsize=14, fontweight='bold', zorder=6)

    # Draw CNOT order (linear)
    order_linear = [0, 1, 2, 3]
    for i, q in enumerate(order_linear):
        x, y = positions[q]
        ax1.annotate(f'{i+1}', (x + 0.15, y + 0.15),
                     fontsize=10, color='red', fontweight='bold')

    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Linear CNOT Order: 1-2-3-4\n(Vulnerable to hook errors)', fontsize=12)
    ax1.axis('off')

    # Right: Diagonal ordering
    ax2 = axes[1]

    for i, (x, y) in positions.items():
        ax2.scatter(x, y, s=400, c='black', zorder=5)
        ax2.annotate(f'q{i}', (x, y), ha='center', va='center',
                     color='white', fontsize=12, fontweight='bold', zorder=6)

    ax2.scatter(0.5, 0.5, s=600, c='blue', marker='s', zorder=5)
    ax2.annotate('A', (0.5, 0.5), ha='center', va='center',
                 color='white', fontsize=14, fontweight='bold', zorder=6)

    # Diagonal order
    order_diag = [0, 3, 1, 2]
    for i, q in enumerate(order_diag):
        x, y = positions[q]
        ax2.annotate(f'{i+1}', (x + 0.15, y + 0.15),
                     fontsize=10, color='green', fontweight='bold')

    # Draw diagonal arrows
    ax2.annotate('', xy=(1, 0), xytext=(0, 1),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.annotate('', xy=(0, 0), xytext=(1, 1),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('Diagonal CNOT Order: 1-4-2-3\n(Hook errors spread diagonally)', fontsize=12)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('cnot_orderings.png', dpi=150, bbox_inches='tight')
    plt.show()


def circuit_depth_analysis():
    """Analyze circuit depth for various code distances."""
    print("\n" + "="*60)
    print("Circuit Depth Analysis")
    print("="*60)

    distances = [3, 5, 7, 9, 11]

    print(f"\n{'Distance':<10} {'Stabilizers':<15} {'Parallel Depth':<15} {'Total Layers':<15}")
    print("-" * 55)

    for d in distances:
        n_stab = d * d - 1  # Approximate for rotated code
        # With parallel execution: prep(1) + CNOTs(4) + measure(1) = 6
        parallel_depth = 6
        total_layers = parallel_depth

        print(f"{d:<10} {n_stab:<15} {parallel_depth:<15} {total_layers:<15}")

    print("\nKey insight: Circuit depth is O(1) for any distance!")
    print("All stabilizers can be measured in parallel layers.")


def syndrome_round_simulation():
    """Simulate multiple syndrome rounds with errors."""
    print("\n" + "="*60)
    print("Syndrome Round Simulation")
    print("="*60)

    # Distance-3 rotated code (simplified)
    # 5 data qubits, 2 X-stabilizers, 2 Z-stabilizers
    circuits = [
        StabilizerCircuit('X', [0, 1, 2], 0, [0, 1, 2]),
        StabilizerCircuit('X', [2, 3, 4], 1, [0, 1, 2]),
        StabilizerCircuit('Z', [0, 2, 3], 2, [0, 1, 2]),
        StabilizerCircuit('Z', [1, 2, 4], 3, [0, 1, 2]),
    ]

    extractor = SyndromeExtractor(n_data=5, circuits=circuits)

    # Inject a single Z-error on qubit 2
    extractor.inject_data_error(2, Pauli.Z)

    print("Injected error: Z on qubit 2")
    print(f"Initial data errors: {[e.name for e in extractor.data_errors]}")

    result = extractor.run_syndrome_extraction()

    print(f"\nAfter syndrome extraction:")
    print(f"Syndromes: {result['syndromes']}")
    print(f"  X-stab 0 (q0,q1,q2): {result['syndromes'][0]}")
    print(f"  X-stab 1 (q2,q3,q4): {result['syndromes'][1]}")
    print(f"  Z-stab 0 (q0,q2,q3): {result['syndromes'][2]}")
    print(f"  Z-stab 1 (q1,q2,q4): {result['syndromes'][3]}")

    print("\nExpected: X-stabilizers detect Z-errors")
    print("Both X-stabilizers should trigger (qubit 2 is in both)")


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Day 801: Syndrome Extraction Circuits")
    print("=" * 60)

    # Demonstrate hook errors
    demo_hook_error()

    # Visualize CNOT schedules
    visualize_cnot_schedule()

    # Analyze circuit depth
    circuit_depth_analysis()

    # Simulate syndrome rounds
    syndrome_round_simulation()

    print("\n" + "=" * 60)
    print("Lab complete.")
    print("=" * 60)
```

### Expected Output

```
============================================================
Day 801: Syndrome Extraction Circuits
============================================================

============================================================
Hook Error Demonstration
============================================================

Scenario: X-error on ancilla after 2nd CNOT (linear order)
CNOT order: [0, 1, 2, 3]

Data errors after extraction:
  Qubit 0: I
  Qubit 1: I
  Qubit 2: X
  Qubit 3: X

Hook errors detected: [(0, [2, 3])]
Syndrome: [0]

----------------------------------------
With diagonal CNOT ordering:
CNOT order: [0, 3, 1, 2]

Data errors after extraction:
  Qubit 0: I
  Qubit 1: X
  Qubit 2: X
  Qubit 3: I

Hook errors: [(0, [1, 2])]
(Diagonal ordering spreads hook error to non-adjacent qubits)

============================================================
Circuit Depth Analysis
============================================================

Distance   Stabilizers     Parallel Depth  Total Layers
-------------------------------------------------------
3          8               6               6
5          24              6               6
7          48              6               6
9          80              6               6
11         120             6               6

Key insight: Circuit depth is O(1) for any distance!
All stabilizers can be measured in parallel layers.

============================================================
Lab complete.
============================================================
```

---

## Summary

### Key Formulas

| Circuit Element | Formula/Specification |
|-----------------|----------------------|
| X-stabilizer prep | $\|0\rangle \xrightarrow{H} \|+\rangle$ |
| Z-stabilizer prep | $\|0\rangle$ (no Hadamard) |
| X-stab CNOT | Ancilla = control, Data = target |
| Z-stab CNOT | Data = control, Ancilla = target |
| Syndrome depth | $O(1)$ (6 layers typical) |
| Hook error probability | $\sim p \cdot (w-1)$ for weight-$w$ stabilizer |

### Key Takeaways

1. **Indirect measurement via ancillas** allows syndrome extraction without collapsing the code state

2. **CNOT ordering critically affects hook errors**: Linear ordering creates adjacent correlated errors; diagonal ordering spreads them

3. **Hook errors can halve effective distance**: A single fault creating weight-2 errors is dangerous

4. **Mitigation strategies**: Flag qubits, repeated rounds, or optimized CNOT orderings

5. **Syndrome extraction is O(1) depth**: All stabilizers measured in parallel regardless of code size

6. **Boundary stabilizers use shorter circuits**: Weight-3 and weight-2 stabilizers need fewer CNOTs

---

## Daily Checklist

Before moving to Day 802, verify you can:

- [ ] Draw complete X and Z stabilizer measurement circuits
- [ ] Trace error propagation through CNOT gates
- [ ] Identify when hook errors occur
- [ ] Design parallel CNOT schedules
- [ ] Explain flag qubit operation
- [ ] Calculate syndrome extraction circuit depth

---

## Preview: Day 802

Tomorrow we study **logical operators on the planar code**:
- Logical X as smooth-to-smooth string
- Logical Z as rough-to-rough string
- String-net interpretation
- Complementary operator pairing

---

## References

1. Fowler, A. G., et al. "Surface codes: Towards practical large-scale quantum computation." *Physical Review A* 86, 032324 (2012)
2. Tomita, Y., & Svore, K. M. "Low-distance surface codes under realistic quantum noise." *Physical Review A* 90, 062320 (2014)
3. Chamberland, C., & Beverland, M. E. "Flag fault-tolerant error correction with arbitrary distance codes." *Quantum* 2, 53 (2018)

---

*Day 801 reveals the careful engineering required to extract error information without corrupting the quantum state—syndrome extraction is the heartbeat of fault-tolerant operation.*
