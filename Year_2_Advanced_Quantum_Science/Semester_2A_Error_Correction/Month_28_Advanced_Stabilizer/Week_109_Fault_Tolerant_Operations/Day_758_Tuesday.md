# Day 758: Error Propagation Analysis

## Overview

**Day:** 758 of 1008
**Week:** 109 (Fault-Tolerant Quantum Operations)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Systematic Analysis of Error Propagation in Quantum Circuits

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Error propagation rules |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Fault path enumeration |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational analysis |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Derive** error propagation rules for all Clifford gates
2. **Track** multi-qubit error evolution through circuits
3. **Count** fault paths in syndrome extraction circuits
4. **Identify** "bad" locations that cause correlated errors
5. **Analyze** the fault structure of arbitrary stabilizer circuits
6. **Apply** propagation rules to verify fault tolerance

---

## Core Content

### 1. Clifford Gate Error Propagation

Every Clifford gate transforms Pauli errors in a specific way. Understanding these transformations is essential for fault tolerance analysis.

#### Single-Qubit Gates

**Hadamard (H):**
$$H X H^{\dagger} = Z, \quad H Z H^{\dagger} = X, \quad H Y H^{\dagger} = -Y$$

Propagation rule: $X \leftrightarrow Z$

**Phase (S):**
$$S X S^{\dagger} = Y, \quad S Z S^{\dagger} = Z, \quad S Y S^{\dagger} = -X$$

Propagation rule: $X \rightarrow Y$

**T Gate:**
$$T X T^{\dagger} = \frac{1}{\sqrt{2}}(X + Y), \quad T Z T^{\dagger} = Z$$

Note: T is not Clifford; output is not a simple Pauli!

#### Two-Qubit Gates

**CNOT (Control-Target):**
$$\boxed{CNOT: X_c \rightarrow X_c X_t, \quad X_t \rightarrow X_t}$$
$$\boxed{CNOT: Z_c \rightarrow Z_c, \quad Z_t \rightarrow Z_c Z_t}$$

**CZ (Controlled-Z):**
$$CZ: X_a \rightarrow X_a Z_b, \quad X_b \rightarrow Z_a X_b$$
$$CZ: Z_a \rightarrow Z_a, \quad Z_b \rightarrow Z_b$$

**SWAP:**
$$SWAP: X_a \rightarrow X_b, \quad X_b \rightarrow X_a$$
$$SWAP: Z_a \rightarrow Z_b, \quad Z_b \rightarrow Z_a$$

### 2. Propagation Through Gate Sequences

For a sequence of gates $G_1, G_2, \ldots, G_n$, an initial error $E$ evolves as:

$$E \xrightarrow{G_1} G_1 E G_1^{\dagger} \xrightarrow{G_2} G_2 G_1 E G_1^{\dagger} G_2^{\dagger} \rightarrow \cdots$$

**Key principle:** Track the error through each gate, applying transformation rules.

#### Example: CNOT Cascade

Consider X error on q₀ through:
```
q₀: ──●──●──
      │  │
q₁: ──⊕──│──
         │
q₂: ─────⊕──
```

Evolution:
1. Initial: $X_0$
2. After CNOT(0,1): $X_0 X_1$
3. After CNOT(0,2): $X_0 X_1 X_2$

**Result:** Weight-1 error → Weight-3 error!

### 3. Fault Path Enumeration

A **fault** is any single component failure. A **fault path** is the sequence of error propagations from that fault to the output.

#### Fault Locations

In a circuit, faults can occur at:
1. **Preparation:** Initial state prepared incorrectly
2. **Gate:** Gate implements wrong operation
3. **Wait:** Idle qubit experiences decoherence
4. **Measurement:** Readout gives wrong result

#### Fault Types

At each location, multiple fault types are possible:

| Location Type | Possible Faults |
|--------------|-----------------|
| Single qubit | X, Y, Z (3 types) |
| Two-qubit gate | IX, IY, IZ, XI, YI, ZI, XX, ... (15 types) |
| Measurement | Bit flip (1 type) |

### 4. Bad Locations Analysis

A **bad location** is a fault location where a single fault can lead to:
- Weight > t error on data qubits, OR
- Incorrect syndrome + weight > 0 error

#### Identifying Bad Locations

For syndrome extraction with n data qubits and 1 ancilla:

**Standard circuit (BAD):**
```
d₁: ──●──────────
      │
d₂: ──│──●───────
      │  │
d₃: ──│──│──●────
      │  │  │
a:  ──⊕──⊕──⊕── M
```

**Bad locations:** Faults on ancilla after first CNOT
- X fault propagates to remaining data qubits
- Creates weight > 1 error

**Number of bad locations:** n - 1 (all but last CNOT)

### 5. Extended Rectangles

An **extended rectangle (exRec)** is a fault-tolerant gadget plus trailing error correction.

$$\boxed{\text{exRec} = \text{Gadget} + \text{EC round}}$$

**Properties of exRecs:**
- If no faults in exRec: Output state is ideal
- Single fault in exRec: Output has correctable error
- Two faults in same exRec: May cause logical error

### 6. The Malignant Set

A set of fault locations is **malignant** if faults at those locations can cause logical error.

**For t-error-correcting code:**
- Need at least t + 1 faults for malignant set
- Fault tolerance requires: Few malignant pairs/triples

**Counting malignant pairs:**
If a gadget has N locations, naively there are $\binom{N}{2}$ pairs.
For FT, we need: (malignant pairs)/(total pairs) to be small.

---

## Worked Examples

### Example 1: Complete CNOT Propagation Table

**Problem:** Derive the full error propagation table for CNOT.

**Solution:**

Starting with arbitrary two-qubit Pauli $P_c \otimes P_t$:

| Input Error | After CNOT |
|-------------|------------|
| $I \otimes I$ | $I \otimes I$ |
| $X \otimes I$ | $X \otimes X$ |
| $Y \otimes I$ | $Y \otimes X$ |
| $Z \otimes I$ | $Z \otimes I$ |
| $I \otimes X$ | $I \otimes X$ |
| $I \otimes Y$ | $Z \otimes Y$ |
| $I \otimes Z$ | $Z \otimes Z$ |
| $X \otimes X$ | $X \otimes I$ |
| $X \otimes Y$ | $Y \otimes Z$ |
| $X \otimes Z$ | $Y \otimes Y$ |
| $Y \otimes X$ | $Y \otimes I$ |
| $Y \otimes Y$ | $X \otimes Z$ |
| $Y \otimes Z$ | $X \otimes Y$ |
| $Z \otimes X$ | $Z \otimes X$ |
| $Z \otimes Y$ | $I \otimes Y$ |
| $Z \otimes Z$ | $I \otimes Z$ |

**Key observations:**
- X component on control spreads to target
- Z component on target spreads to control
- Weight is not always preserved (can decrease!)

### Example 2: Fault Path Counting

**Problem:** Count all fault paths for the 3-qubit bit-flip code syndrome measurement.

**Solution:**

Circuit structure:
```
d₁: ──●──●─────────── (2 CNOTs)
      │  │
d₂: ──⊕──│──●──●──── (2 CNOTs)
         │  │  │
d₃: ─────⊕──⊕──│──── (2 CNOTs)
               │
a₁: ───────────⊕── M (syndrome Z₁Z₂Z₃)

(Similar for second stabilizer)
```

**Fault locations per syndrome:**
- 3 CNOTs, each with ~15 fault types: 45 faults
- Ancilla preparation fault: 3 types
- Measurement fault: 1 type
- **Total:** ~50 fault scenarios per syndrome

**For complete syndrome extraction (2 stabilizers):** ~100 scenarios

**Analysis:** Must check each scenario produces correctable error.

### Example 3: Verifying Steane Syndrome Extraction

**Problem:** Show that Steane-style syndrome extraction is fault-tolerant.

**Solution:**

**Steane method:** Use encoded ancilla instead of single ancilla.

```
Data block:     |ψ⟩_L ──┬──┬──┬──┬──┬──┬── |ψ'⟩_L
                        │  │  │  │  │  │
Ancilla block:  |0⟩_L ──⊕──⊕──⊕──⊕──⊕──⊕── Measure each
                        ↑  ↑  ↑  ↑  ↑  ↑
                    Transversal CNOT
```

**Why fault-tolerant:**
1. Transversal CNOT is FT (single fault → weight-1 error per block)
2. Single fault on ancilla: Weight-1 error, possibly wrong syndrome
3. Single fault on data: Weight-1 error, correct syndrome
4. Repeat measurement → distinguish data error from ancilla error

**Conclusion:** Steane extraction is FT because:
- Uses transversal operations only
- Each fault affects at most one qubit per block

---

## Practice Problems

### Problem Set A: Propagation Rules

**A1.** Derive the error propagation rules for the controlled-Y (CY) gate.

**A2.** For the circuit:
```
q₁: ──H──●──H──
         │
q₂: ─────⊕─────
```
How does an X error on q₁ at the input propagate to the output?

**A3.** Prove that CZ is symmetric: a fault on either qubit has the same effect.

### Problem Set B: Fault Counting

**B1.** A syndrome extraction circuit uses 5 CNOTs and 2 ancilla measurements. Give an upper bound on the number of single-fault scenarios to analyze.

**B2.** For the [[5,1,3]] code with 4 stabilizers, each requiring 4 CNOTs, how many total fault locations exist in one syndrome extraction round?

**B3.** If a circuit has N fault locations and each pair of faults is equally likely to be malignant, what fraction of pairs must be malignant for the logical error rate to scale as p² (instead of p³)?

### Problem Set C: Advanced Analysis

**C1.** Consider flag qubit syndrome extraction:
```
d₁: ──●────────────
      │
flag: ⊕──●─────────
         │
d₂: ─────⊕──●──────
            │
a:   |0⟩────⊕── M
```
Identify which faults are caught by the flag measurement.

**C2.** For the [[7,1,3]] Steane code, count the number of extended rectangles in a full EC round (syndrome extraction + correction).

**C3.** Design a syndrome extraction circuit for a single Z stabilizer (e.g., ZZZZ) that has no bad locations.

---

## Computational Lab

```python
"""
Day 758 Computational Lab: Error Propagation Analysis
=====================================================

Systematic tracking of error propagation through Clifford circuits.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import IntEnum
from itertools import product

class Pauli(IntEnum):
    """Single-qubit Paulis encoded as 2-bit integers."""
    I = 0b00
    X = 0b01
    Z = 0b10
    Y = 0b11

    def __mul__(self, other: 'Pauli') -> 'Pauli':
        """Multiply Paulis (ignoring phase)."""
        return Pauli(self.value ^ other.value)


@dataclass
class PauliString:
    """n-qubit Pauli operator as list of single-qubit Paulis."""
    paulis: List[Pauli]

    @classmethod
    def identity(cls, n: int) -> 'PauliString':
        return cls([Pauli.I] * n)

    @classmethod
    def single(cls, n: int, qubit: int, pauli: Pauli) -> 'PauliString':
        """Single non-identity Pauli at specified position."""
        ps = [Pauli.I] * n
        ps[qubit] = pauli
        return cls(ps)

    def weight(self) -> int:
        """Number of non-identity Paulis."""
        return sum(1 for p in self.paulis if p != Pauli.I)

    def x_weight(self) -> int:
        """Number of X or Y components."""
        return sum(1 for p in self.paulis if p in [Pauli.X, Pauli.Y])

    def z_weight(self) -> int:
        """Number of Z or Y components."""
        return sum(1 for p in self.paulis if p in [Pauli.Z, Pauli.Y])

    def copy(self) -> 'PauliString':
        return PauliString(self.paulis.copy())

    def __str__(self):
        symbols = {Pauli.I: 'I', Pauli.X: 'X', Pauli.Y: 'Y', Pauli.Z: 'Z'}
        return ''.join(symbols[p] for p in self.paulis)


class CliffordPropagator:
    """
    Propagate Pauli errors through Clifford circuits.

    Uses symplectic representation for efficiency.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits

    def apply_H(self, error: PauliString, qubit: int) -> PauliString:
        """Apply Hadamard: X ↔ Z."""
        result = error.copy()
        p = result.paulis[qubit]
        if p == Pauli.X:
            result.paulis[qubit] = Pauli.Z
        elif p == Pauli.Z:
            result.paulis[qubit] = Pauli.X
        # Y → -Y (phase only, same Pauli)
        return result

    def apply_S(self, error: PauliString, qubit: int) -> PauliString:
        """Apply S: X → Y, Z → Z."""
        result = error.copy()
        p = result.paulis[qubit]
        if p == Pauli.X:
            result.paulis[qubit] = Pauli.Y
        elif p == Pauli.Y:
            result.paulis[qubit] = Pauli.X
        return result

    def apply_CNOT(self, error: PauliString,
                   control: int, target: int) -> PauliString:
        """
        Apply CNOT from control to target.

        X_c → X_c X_t
        Z_t → Z_c Z_t
        """
        result = error.copy()
        pc, pt = result.paulis[control], result.paulis[target]

        # Extract X and Z components
        xc = pc.value & 0b01
        zc = (pc.value & 0b10) >> 1
        xt = pt.value & 0b01
        zt = (pt.value & 0b10) >> 1

        # Apply CNOT transformation
        new_xt = xt ^ xc  # X spreads forward
        new_zc = zc ^ zt  # Z spreads backward

        # Reconstruct Paulis
        result.paulis[control] = Pauli((new_zc << 1) | xc)
        result.paulis[target] = Pauli((zt << 1) | new_xt)

        return result

    def apply_CZ(self, error: PauliString,
                 qubit_a: int, qubit_b: int) -> PauliString:
        """
        Apply CZ.

        X_a → X_a Z_b
        X_b → Z_a X_b
        """
        result = error.copy()
        pa, pb = result.paulis[qubit_a], result.paulis[qubit_b]

        xa = pa.value & 0b01
        za = (pa.value & 0b10) >> 1
        xb = pb.value & 0b01
        zb = (pb.value & 0b10) >> 1

        # CZ transformation
        new_za = za ^ xb
        new_zb = zb ^ xa

        result.paulis[qubit_a] = Pauli((new_za << 1) | xa)
        result.paulis[qubit_b] = Pauli((new_zb << 1) | xb)

        return result


def analyze_syndrome_circuit(n_data: int,
                            stabilizer_weight: int) -> Dict:
    """
    Analyze fault tolerance of standard syndrome extraction.

    Returns statistics on fault propagation.
    """
    n_total = n_data + 1  # +1 for ancilla
    ancilla = n_data

    prop = CliffordPropagator(n_total)
    results = {
        'n_data': n_data,
        'n_cnots': stabilizer_weight,
        'fault_analysis': [],
        'max_data_weight': 0,
        'bad_locations': []
    }

    # Analyze fault at each CNOT location
    for fault_after_cnot in range(stabilizer_weight):
        for fault_type in [Pauli.X, Pauli.Z, Pauli.Y]:
            # Start with error on ancilla after fault_after_cnot CNOTs
            error = PauliString.identity(n_total)
            error.paulis[ancilla] = fault_type

            # Apply remaining CNOTs
            for i in range(fault_after_cnot + 1, stabilizer_weight):
                error = prop.apply_CNOT(error, i, ancilla)

            # Count error weight on data qubits only
            data_weight = sum(1 for i in range(n_data)
                            if error.paulis[i] != Pauli.I)

            analysis = {
                'fault_location': fault_after_cnot,
                'fault_type': str(fault_type),
                'final_error': str(error),
                'data_weight': data_weight
            }
            results['fault_analysis'].append(analysis)

            if data_weight > results['max_data_weight']:
                results['max_data_weight'] = data_weight

            if data_weight > 1:
                results['bad_locations'].append(
                    (fault_after_cnot, str(fault_type))
                )

    return results


def generate_cnot_propagation_table() -> Dict[Tuple[Pauli, Pauli],
                                              Tuple[Pauli, Pauli]]:
    """Generate complete CNOT error propagation table."""
    prop = CliffordPropagator(2)
    table = {}

    for pc in Pauli:
        for pt in Pauli:
            error = PauliString([pc, pt])
            result = prop.apply_CNOT(error, 0, 1)
            table[(pc, pt)] = (result.paulis[0], result.paulis[1])

    return table


def count_fault_scenarios(circuit_spec: Dict) -> Dict:
    """
    Count fault scenarios in a circuit.

    circuit_spec: {
        'single_qubit_gates': int,
        'two_qubit_gates': int,
        'preparations': int,
        'measurements': int
    }
    """
    counts = {
        'single_faults': 0,
        'pair_faults': 0,
        'by_type': {}
    }

    # Single-qubit: 3 fault types (X, Y, Z)
    sq_faults = circuit_spec.get('single_qubit_gates', 0) * 3
    counts['by_type']['single_qubit'] = sq_faults

    # Two-qubit: 15 fault types
    tq_faults = circuit_spec.get('two_qubit_gates', 0) * 15
    counts['by_type']['two_qubit'] = tq_faults

    # Preparation: 3 fault types
    prep_faults = circuit_spec.get('preparations', 0) * 3
    counts['by_type']['preparation'] = prep_faults

    # Measurement: 1 fault type (bit flip)
    meas_faults = circuit_spec.get('measurements', 0) * 1
    counts['by_type']['measurement'] = meas_faults

    counts['single_faults'] = sq_faults + tq_faults + prep_faults + meas_faults
    counts['pair_faults'] = counts['single_faults'] * (counts['single_faults'] - 1) // 2

    return counts


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 758: ERROR PROPAGATION ANALYSIS")
    print("=" * 70)

    # Demo 1: CNOT propagation table
    print("\n" + "=" * 70)
    print("Demo 1: Complete CNOT Error Propagation Table")
    print("=" * 70)

    table = generate_cnot_propagation_table()
    print("\n  Input (c⊗t)  →  Output (c⊗t)")
    print("  " + "-" * 30)
    for (pc, pt), (rc, rt) in table.items():
        print(f"  {pc.name}{pt.name}  →  {rc.name}{rt.name}")

    # Demo 2: Track error through circuit
    print("\n" + "=" * 70)
    print("Demo 2: Error Propagation Through Circuit")
    print("=" * 70)

    prop = CliffordPropagator(4)

    # Circuit: H - CNOT(0,1) - CNOT(1,2) - H
    print("\nCircuit: H₀ - CNOT(0,1) - CNOT(1,2) - H₀")
    print("Initial X error on qubit 0:")

    error = PauliString.single(4, 0, Pauli.X)
    print(f"  Initial:        {error} (weight {error.weight()})")

    error = prop.apply_H(error, 0)
    print(f"  After H₀:       {error} (weight {error.weight()})")

    error = prop.apply_CNOT(error, 0, 1)
    print(f"  After CNOT(0,1): {error} (weight {error.weight()})")

    error = prop.apply_CNOT(error, 1, 2)
    print(f"  After CNOT(1,2): {error} (weight {error.weight()})")

    error = prop.apply_H(error, 0)
    print(f"  After H₀:       {error} (weight {error.weight()})")

    # Demo 3: Syndrome extraction analysis
    print("\n" + "=" * 70)
    print("Demo 3: Syndrome Extraction Fault Analysis")
    print("=" * 70)

    results = analyze_syndrome_circuit(4, 4)
    print(f"\nCircuit: 4-qubit stabilizer (ZZZZ) syndrome extraction")
    print(f"Number of CNOTs: {results['n_cnots']}")
    print(f"Maximum data error weight: {results['max_data_weight']}")
    print(f"Number of bad locations: {len(results['bad_locations'])}")

    print("\nBad locations (cause weight > 1 on data):")
    for loc, ftype in results['bad_locations'][:5]:
        print(f"  Fault after CNOT {loc}, type {ftype}")

    # Demo 4: Fault counting
    print("\n" + "=" * 70)
    print("Demo 4: Fault Scenario Counting")
    print("=" * 70)

    spec = {
        'single_qubit_gates': 7,
        'two_qubit_gates': 6,
        'preparations': 1,
        'measurements': 1
    }

    counts = count_fault_scenarios(spec)
    print(f"\nCircuit specification:")
    for k, v in spec.items():
        print(f"  {k}: {v}")

    print(f"\nFault counts:")
    print(f"  Total single-fault scenarios: {counts['single_faults']}")
    print(f"  Total fault pairs to check: {counts['pair_faults']}")
    print(f"  Breakdown by type: {counts['by_type']}")

    # Summary
    print("\n" + "=" * 70)
    print("ERROR PROPAGATION RULES SUMMARY")
    print("=" * 70)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  CLIFFORD GATE ERROR PROPAGATION                            │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  H:  X ↔ Z                                                 │
    │  S:  X → Y, Z → Z                                          │
    │                                                             │
    │  CNOT(c,t):  X_c → X_c X_t  (X forward)                    │
    │              Z_t → Z_c Z_t  (Z backward)                   │
    │                                                             │
    │  CZ(a,b):   X_a → X_a Z_b  (symmetric)                     │
    │             X_b → Z_a X_b                                   │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │  KEY INSIGHT: Error weight can increase through CNOTs      │
    │               Bad locations: where weight increases > 1     │
    └─────────────────────────────────────────────────────────────┘
    """)

    print("=" * 70)
    print("Day 758 Complete: Error Propagation Analysis Mastered")
    print("=" * 70)
```

---

## Summary

### Key Propagation Rules

| Gate | X Propagation | Z Propagation |
|------|---------------|---------------|
| H | X → Z | Z → X |
| S | X → Y | Z → Z |
| CNOT(c,t) | X_c → X_c X_t | Z_t → Z_c Z_t |
| CZ | X_a → X_a Z_b | Z unchanged |

### Critical Equations

$$\boxed{CNOT: X_c \rightarrow X_c X_t \text{ (spreads forward)}}$$
$$\boxed{CNOT: Z_t \rightarrow Z_c Z_t \text{ (spreads backward)}}$$

### Fault Analysis Framework

1. **Enumerate** all fault locations
2. **Classify** faults by type (X, Y, Z, IX, XI, ...)
3. **Propagate** each fault through remaining circuit
4. **Check** if output weight ≤ t (correctable)
5. **Identify** bad locations and malignant sets

---

## Daily Checklist

- [ ] Derived Clifford gate propagation rules
- [ ] Tracked errors through multi-gate circuits
- [ ] Counted fault paths in example circuits
- [ ] Identified bad locations
- [ ] Ran computational propagation analysis
- [ ] Completed practice problems

---

## Preview: Day 759

Tomorrow we study **Fault-Tolerant State Preparation**, learning how to:
- Prepare encoded |0⟩_L and |+⟩_L without weight-2 errors
- Use verification circuits to catch preparation errors
- Construct cat states for ancilla encoding
- Apply flag qubit techniques
