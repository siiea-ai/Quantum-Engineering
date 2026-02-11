# Week 113: Toric Code Fundamentals

## Overview

**Days:** 785-791 (7 days)
**Month:** 29 (Topological Codes)
**Topic:** Kitaev's Toric Code - Foundation of Topological Quantum Error Correction

---

## Status: IN PROGRESS

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 785 | Monday | Kitaev's Toric Code Introduction | Not Started |
| 786 | Tuesday | Star and Plaquette Operators | Not Started |
| 787 | Wednesday | Ground State and Code Space | Not Started |
| 788 | Thursday | Toric Code as CSS Code | Not Started |
| 789 | Friday | Logical Operators on Torus | Not Started |
| 790 | Saturday | Error Model and Distance | Not Started |
| 791 | Sunday | Week 113 Synthesis | Not Started |

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Describe** the toric code lattice structure with qubits on edges
2. **Construct** star and plaquette operators and verify their commutation
3. **Prove** the ground state degeneracy equals 4 on the torus
4. **Identify** the toric code as a CSS code with specific parameters
5. **Define** logical operators as non-contractible loops on the torus
6. **Calculate** the code distance and error correction capability
7. **Explain** the connection between topology and quantum error protection
8. **Implement** toric code simulations in Python

---

## Core Concepts

### Kitaev's Toric Code (1997, 2003)

The toric code is defined on a square lattice embedded on a torus (periodic boundary conditions). Qubits reside on **edges** of the lattice.

**Key insight:** Topological protection arises from the global properties of the torus, not from any local structure.

### Stabilizer Structure

**Star operators** (at vertices):
$$A_v = \prod_{e \ni v} X_e$$

**Plaquette operators** (at faces):
$$B_p = \prod_{e \in \partial p} Z_e$$

**Critical property:** All operators commute:
$$[A_v, A_{v'}] = [B_p, B_{p'}] = [A_v, B_p] = 0$$

### Code Parameters

For an $L \times L$ torus:

$$\boxed{[[n, k, d]] = [[2L^2, 2, L]]}$$

- **n = 2L²**: Number of physical qubits (edges)
- **k = 2**: Number of logical qubits (from torus topology)
- **d = L**: Code distance (minimum non-trivial loop)

### Hamiltonian

The toric code Hamiltonian:

$$H = -\sum_v A_v - \sum_p B_p$$

Ground states satisfy: $A_v|\psi\rangle = +1|\psi\rangle$ and $B_p|\psi\rangle = +1|\psi\rangle$ for all v, p.

### CSS Structure

The toric code is a CSS code:
- X-stabilizers: Star operators (vertices)
- Z-stabilizers: Plaquette operators (faces)

This follows from the chain complex structure on the lattice.

---

## Weekly Breakdown

### Day 785: Kitaev's Toric Code Introduction
- Historical context and Kitaev's contributions
- Torus topology and periodic boundaries
- Lattice structure: vertices, edges, faces
- Qubit placement on edges
- Code parameter derivation: $[[2L^2, 2, L]]$

### Day 786: Star and Plaquette Operators
- Star operator construction: $A_v = \prod_{e \ni v} X_e$
- Plaquette operator construction: $B_p = \prod_{e \in \partial p} Z_e$
- Proof of commutation relations
- Stabilizer group structure and constraints

### Day 787: Ground State and Code Space
- Ground state condition: $A_v|\psi\rangle = B_p|\psi\rangle = +1|\psi\rangle$
- Ground state degeneracy calculation
- Logical qubit interpretation
- Toric code Hamiltonian and energy gap

### Day 788: Toric Code as CSS Code
- X-stabilizers from star operators
- Z-stabilizers from plaquette operators
- Chain complex and boundary operators
- Connection to homology groups
- Dual lattice picture

### Day 789: Logical Operators on Torus
- Non-contractible loops on the torus
- Logical X operators along horizontal/vertical cycles
- Logical Z operators along dual cycles
- Commutation and anti-commutation relations
- Topological protection

### Day 790: Error Model and Distance
- X, Z, and Y errors as anyon creation
- Error chains and their endpoints
- Code distance = minimum non-trivial loop length
- Error threshold ~10.9% for independent noise
- Comparison with random codes

### Day 791: Week 113 Synthesis
- Comprehensive concept map
- Master formula reference
- Integration problems
- Preview of anyons (Week 114)

---

## Key Equations

**Star Operator:**
$$\boxed{A_v = \prod_{e \ni v} X_e}$$

**Plaquette Operator:**
$$\boxed{B_p = \prod_{e \in \partial p} Z_e}$$

**Commutation:**
$$\boxed{[A_v, B_p] = 0 \text{ for all } v, p}$$

**Code Parameters:**
$$\boxed{[[2L^2, 2, L]] \text{ on } L \times L \text{ torus}}$$

**Ground State:**
$$\boxed{A_v|\psi_{GS}\rangle = B_p|\psi_{GS}\rangle = +1|\psi_{GS}\rangle}$$

**Hamiltonian:**
$$\boxed{H = -\sum_v A_v - \sum_p B_p}$$

---

## Computational Skills

```python
import numpy as np
from typing import Tuple, List, Set

class ToricCode:
    """
    Toric code on L x L lattice.

    Qubits on edges, indexed by (row, col, direction).
    direction: 0 = horizontal, 1 = vertical
    """

    def __init__(self, L: int):
        """Initialize L x L toric code."""
        self.L = L
        self.n_qubits = 2 * L * L  # edges
        self.n_vertices = L * L
        self.n_faces = L * L

    def edge_index(self, row: int, col: int, direction: int) -> int:
        """Convert (row, col, direction) to linear index."""
        row, col = row % self.L, col % self.L
        return direction * self.L * self.L + row * self.L + col

    def star_operator(self, v_row: int, v_col: int) -> List[int]:
        """Return edge indices for star operator at vertex (v_row, v_col)."""
        edges = [
            self.edge_index(v_row, v_col, 0),      # right horizontal
            self.edge_index(v_row, v_col - 1, 0),  # left horizontal
            self.edge_index(v_row, v_col, 1),      # down vertical
            self.edge_index(v_row - 1, v_col, 1),  # up vertical
        ]
        return edges

    def plaquette_operator(self, p_row: int, p_col: int) -> List[int]:
        """Return edge indices for plaquette at face (p_row, p_col)."""
        edges = [
            self.edge_index(p_row, p_col, 0),      # top horizontal
            self.edge_index(p_row + 1, p_col, 0),  # bottom horizontal
            self.edge_index(p_row, p_col, 1),      # left vertical
            self.edge_index(p_row, p_col + 1, 1),  # right vertical
        ]
        return edges

    def code_parameters(self) -> Tuple[int, int, int]:
        """Return [[n, k, d]] parameters."""
        return (self.n_qubits, 2, self.L)
```

---

## References

### Primary Sources

- Kitaev, A. "Fault-tolerant quantum computation by anyons" (2003) - arXiv:quant-ph/9707021
- Kitaev, A. "Quantum computations: algorithms and error correction" (1997)
- Dennis, E. et al. "Topological quantum memory" (2002) - J. Math. Phys. 43, 4452

### Key Papers

- Bravyi, S. & Kitaev, A. "Quantum codes on a lattice with boundary" (1998)
- Fowler, A. et al. "Surface codes: Towards practical large-scale quantum computation" (2012)
- Bombin, H. "An Introduction to Topological Quantum Codes" (2013)

### Online Resources

- [Toric Code - Error Correction Zoo](https://errorcorrectionzoo.org/c/toric)
- [Surface Code Tutorial - IBM Quantum](https://learning.quantum.ibm.com/)
- [Kitaev's Toric Code Explained](https://www.youtube.com/watch?v=mfcFJtMDPKQ)

---

## Connections

### Prerequisites (Week 112)

- CSS code construction and properties
- Stabilizer formalism and binary representation
- Fault-tolerant syndrome measurement

### Leads to (Week 114)

- Anyonic excitations: e (charge) and m (flux)
- Topological order and long-range entanglement
- Braiding statistics and topological phases

---

## Summary

Week 113 introduces the toric code, Kitaev's foundational topological quantum error-correcting code. By placing qubits on edges of a lattice embedded on a torus, the code achieves topological protection: logical information is encoded in global, non-local degrees of freedom that cannot be disturbed by local errors. The star and plaquette operators form a stabilizer group with CSS structure, and the ground state degeneracy of 4 encodes 2 logical qubits. Understanding the toric code is essential for all modern approaches to fault-tolerant quantum computing.
