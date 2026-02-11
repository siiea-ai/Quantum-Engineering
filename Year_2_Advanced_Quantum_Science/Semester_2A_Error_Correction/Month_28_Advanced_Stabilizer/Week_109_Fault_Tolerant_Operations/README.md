# Week 109: Fault-Tolerant Quantum Operations

## Overview

**Days:** 757-763 (7 days)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Fault-Tolerant Gadgets and Universal Quantum Computation

---

## Status: ✅ COMPLETE

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 757 | Monday | Fault Tolerance Foundations | ✅ Complete |
| 758 | Tuesday | Error Propagation Analysis | ✅ Complete |
| 759 | Wednesday | Fault-Tolerant State Preparation | ✅ Complete |
| 760 | Thursday | Fault-Tolerant Measurement | ✅ Complete |
| 761 | Friday | Transversal Gates & Universality | ✅ Complete |
| 762 | Saturday | Magic State Injection | ✅ Complete |
| 763 | Sunday | Week 109 Synthesis | ✅ Complete |

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Define** fault tolerance and identify when circuits are fault-tolerant
2. **Analyze** error propagation through quantum gates
3. **Construct** fault-tolerant state preparation circuits
4. **Design** fault-tolerant syndrome measurement protocols
5. **Explain** transversal gate limitations (Eastin-Knill theorem)
6. **Implement** magic state injection for non-Clifford gates
7. **Compare** different approaches to universal fault-tolerant computation

---

## Core Concepts

### What is Fault Tolerance?

**Definition:** A quantum operation is *fault-tolerant* if a single physical fault causes at most one error per code block.

$$\boxed{\text{Single fault} \rightarrow \text{Weight-1 error (correctable)}}$$

### Error Propagation

CNOT gates can spread errors:
- X error on control → X error on target (spreads)
- Z error on target → Z error on control (spreads)

$$CNOT: X_c \rightarrow X_c X_t, \quad Z_t \rightarrow Z_c Z_t$$

### Transversal Gates

**Definition:** A gate is *transversal* if it acts qubit-by-qubit between code blocks.

$$\bar{U} = U^{\otimes n}$$

**Property:** Transversal gates are automatically fault-tolerant.

### Eastin-Knill Theorem

**Statement:** No code can have a universal set of transversal gates.

**Implication:** Need magic states or code switching for universality.

---

## Key Equations

**Fault-Tolerant Condition:**
$$\boxed{\text{FT gadget: } \forall \text{ single fault}, \text{ output error weight} \leq 1}$$

**Error Propagation Through CNOT:**
$$\boxed{CNOT |E_c\rangle|E_t\rangle = |E_c \oplus E_t\rangle|E_t\rangle \text{ (for X errors)}}$$

**Magic State for T Gate:**
$$\boxed{|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)}$$

**Gate Teleportation:**
$$\boxed{T|\psi\rangle = S^m (X^m T |\psi\rangle \otimes \langle +|) \text{ (up to correction)}}$$

---

## Daily Breakdown

### Day 757: Fault Tolerance Foundations
- Motivation for fault tolerance
- Definition and formal criteria
- Non-fault-tolerant examples
- Historical development

### Day 758: Error Propagation Analysis
- CNOT error propagation
- Multi-qubit gate analysis
- Counting fault paths
- Bad locations in circuits

### Day 759: Fault-Tolerant State Preparation
- Preparing encoded |0⟩ and |+⟩
- Verification circuits
- Cat state preparation
- Flag qubit methods

### Day 760: Fault-Tolerant Measurement
- Syndrome extraction circuits
- Shor-style ancilla
- Steane-style extraction
- Flag qubit syndrome measurement

### Day 761: Transversal Gates & Universality
- Transversal gate sets by code
- Eastin-Knill theorem proof
- Clifford hierarchy
- Approaches to universality

### Day 762: Magic State Injection
- Magic states |T⟩, |H⟩
- Gate teleportation protocol
- Injection circuits
- Magic state distillation preview

### Day 763: Week 109 Synthesis
- Comprehensive review
- Fault-tolerant protocol design
- Integration problems
- Preparation for Week 110

---

## Computational Skills

```python
import numpy as np
from typing import List, Tuple, Set

def count_fault_paths(circuit_depth: int,
                      gates_per_layer: int,
                      max_faults: int = 1) -> int:
    """
    Count number of fault paths in a circuit.

    A fault path is a set of locations where faults occur.
    For fault tolerance, we need single faults → correctable errors.
    """
    total_locations = circuit_depth * gates_per_layer

    if max_faults == 1:
        return total_locations
    elif max_faults == 2:
        return total_locations * (total_locations - 1) // 2
    else:
        from math import comb
        return comb(total_locations, max_faults)


def is_transversal(gate_matrix: np.ndarray, n: int) -> bool:
    """
    Check if a 2^n × 2^n gate matrix is transversal.

    A transversal gate has the form U ⊗ U ⊗ ... ⊗ U.
    """
    dim = 2 ** n
    if gate_matrix.shape != (dim, dim):
        return False

    # Extract single-qubit gate (if transversal)
    single_qubit = gate_matrix[:2, :2]

    # Construct tensor product
    expected = single_qubit
    for _ in range(n - 1):
        expected = np.kron(expected, single_qubit)

    return np.allclose(gate_matrix, expected)


class FaultPathAnalyzer:
    """Analyze fault propagation in quantum circuits."""

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.x_errors = [False] * n_qubits
        self.z_errors = [False] * n_qubits

    def apply_cnot(self, control: int, target: int):
        """Apply CNOT and track error propagation."""
        # X errors: spread from control to target
        if self.x_errors[control]:
            self.x_errors[target] = not self.x_errors[target]

        # Z errors: spread from target to control
        if self.z_errors[target]:
            self.z_errors[control] = not self.z_errors[control]

    def inject_x_error(self, qubit: int):
        """Inject X error on qubit."""
        self.x_errors[qubit] = True

    def inject_z_error(self, qubit: int):
        """Inject Z error on qubit."""
        self.z_errors[qubit] = True

    def error_weight(self) -> Tuple[int, int]:
        """Return (X_weight, Z_weight)."""
        return sum(self.x_errors), sum(self.z_errors)
```

---

## References

### Primary Sources
- Shor, "Fault-tolerant quantum computation" (1996)
- Preskill, "Fault-tolerant quantum computation" (1998)
- Gottesman, "Theory of fault-tolerant quantum computation" (1998)

### Key Papers
- Eastin & Knill, "Restrictions on transversal encoded quantum gate sets" (2009)
- Chao & Reichardt, "Quantum error correction with only two extra qubits" (2018)
- Chamberland & Beverland, "Flag fault-tolerant error correction" (2018)

---

## Connections

### Prerequisites (Month 27)
- Stabilizer formalism
- CSS code construction
- Transversal gate analysis
- Magic state basics

### Leads to (Week 110)
- Threshold theorem proofs
- Concatenated code analysis
- Noise model comparison
