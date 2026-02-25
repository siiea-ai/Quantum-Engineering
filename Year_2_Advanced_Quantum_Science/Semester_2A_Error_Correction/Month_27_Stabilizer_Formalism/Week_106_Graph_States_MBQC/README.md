# Week 106: Graph States & Measurement-Based Quantum Computation

## Overview

**Days:** 736-742 (7 days)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Graph States, Local Complementation, and MBQC

---

## Status: ✅ COMPLETE

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 736 | Monday | Introduction to Graph States | ✅ Complete |
| 737 | Tuesday | Stabilizer Structure of Graph States | ✅ Complete |
| 738 | Wednesday | Local Complementation | ✅ Complete |
| 739 | Thursday | Local Clifford Equivalence | ✅ Complete |
| 740 | Friday | Measurement-Based QC Foundations | ✅ Complete |
| 741 | Saturday | Cluster States and Universality | ✅ Complete |
| 742 | Sunday | Week Synthesis | ✅ Complete |

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Construct** graph states from adjacency matrices
2. **Derive** the stabilizer generators from graph structure
3. **Perform** local complementation on graphs
4. **Classify** graph states by LC equivalence classes
5. **Explain** the foundations of measurement-based quantum computation
6. **Understand** why cluster states enable universal QC
7. **Connect** graph states to quantum error correction
8. **Implement** graph state operations computationally

---

## Core Concepts

### Graph States

**Definition:**
Given a graph G = (V, E) with n vertices, the graph state |G⟩ is:

$$|G\rangle = \prod_{(i,j) \in E} CZ_{ij} |+\rangle^{\otimes n}$$

Each vertex is a qubit initialized in |+⟩, and each edge applies a CZ gate.

### Stabilizer Generators

For a graph state |G⟩, the stabilizer generators are:

$$K_a = X_a \prod_{b \in N(a)} Z_b$$

where N(a) is the neighborhood of vertex a (all vertices connected by edges).

### Local Complementation

**Definition:**
Local complementation at vertex a transforms G to G*a where:
- Edges within N(a) are complemented (added if absent, removed if present)
- All other edges unchanged

**Key Property:** |G⟩ and |G*a⟩ are related by local Clifford operations.

### Local Clifford Equivalence

Two graph states are **LC-equivalent** if one can be transformed to the other by:
- Single-qubit Clifford operations
- Graph isomorphism

LC-equivalence classes characterize the entanglement structure.

### Measurement-Based Quantum Computation

**Key Idea:** Computation by adaptive single-qubit measurements on an entangled resource state.

**Process:**
1. Prepare resource state (e.g., cluster state)
2. Perform single-qubit measurements in chosen bases
3. Apply classical corrections based on outcomes
4. Read out result

### Cluster States

**2D Cluster State:** Graph state on a 2D square lattice.

$$|C_{m \times n}\rangle = \prod_{\text{edges}} CZ |+\rangle^{\otimes mn}$$

**Universality:** 2D cluster states enable universal quantum computation through MBQC.

---

## Weekly Breakdown

### Day 736: Introduction to Graph States

- Graph theory basics for quantum states
- Constructing |G⟩ from adjacency matrix
- Simple examples: linear, star, ring graphs
- Physical implementation considerations

### Day 737: Stabilizer Structure of Graph States

- Deriving stabilizers from graph structure
- Adjacency matrix formalism
- Connection to CSS codes
- Graph state as codeword

### Day 738: Local Complementation

- Definition and algorithm
- Effect on stabilizers
- Generating LC-equivalent graphs
- Computational complexity

### Day 739: Local Clifford Equivalence

- LC equivalence classes
- Interlacement and edge-local complementation
- Classification of small graph states
- Entanglement under LC

### Day 740: Measurement-Based QC Foundations

- Single-qubit measurements
- Byproduct operators and corrections
- Teleportation as MBQC primitive
- Gate implementation

### Day 741: Cluster States and Universality

- 2D cluster state structure
- Universal gate set via measurements
- Computational depth
- Error propagation

### Day 742: Week Synthesis

- Comprehensive review
- Integration problems
- Preparation for Week 107

---

## Key Equations

**Graph State:**
$$\boxed{|G\rangle = \prod_{(i,j) \in E} CZ_{ij} |+\rangle^{\otimes n}}$$

**Stabilizer Generator:**
$$\boxed{K_a = X_a \bigotimes_{b \in N(a)} Z_b}$$

**Adjacency Matrix Form:**
$$K_a = X_a Z^{\Gamma_a}$$

where $\Gamma_a$ is row a of adjacency matrix Γ.

**Local Complementation:**
$$|G\rangle \xrightarrow{U_a^\dagger} |G*a\rangle$$

where $U_a = \exp(-i\frac{\pi}{4}X_a)\prod_{b \in N(a)}\sqrt{Z_b}$

---

## Computational Skills

```python
import numpy as np
import networkx as nx

def create_graph_state_stabilizers(adjacency_matrix):
    """
    Generate stabilizer generators for graph state.

    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        n×n symmetric binary matrix

    Returns:
    --------
    stabilizers : list
        List of Pauli strings for stabilizers
    """
    n = len(adjacency_matrix)
    stabilizers = []

    for a in range(n):
        pauli = ['I'] * n
        pauli[a] = 'X'
        for b in range(n):
            if adjacency_matrix[a, b] == 1:
                pauli[b] = 'Z' if pauli[b] == 'I' else 'Y'
        stabilizers.append(''.join(pauli))

    return stabilizers

def local_complement(adjacency_matrix, vertex):
    """
    Perform local complementation at vertex.

    Complements edges within neighborhood of vertex.
    """
    adj = adjacency_matrix.copy()
    n = len(adj)
    neighbors = [i for i in range(n) if adj[vertex, i] == 1]

    for i in neighbors:
        for j in neighbors:
            if i < j:
                adj[i, j] = 1 - adj[i, j]
                adj[j, i] = 1 - adj[j, i]

    return adj
```

---

## References

### Primary Sources

- Hein et al., "Entanglement in Graph States and its Applications" (2006)
- Raussendorf & Briegel, "A One-Way Quantum Computer" (2001)
- Van den Nest et al., "Graphical description of LC-equivalence" (2004)

### Key Papers

- Briegel et al., "Measurement-based quantum computation" (2009)
- Raussendorf, Browne, Briegel, "Computational model for MBQC" (2003)

### Online Resources

- [Graph State Zoo](https://graphstatezoo.github.io/)
- [Error Correction Zoo - Graph States](https://errorcorrectionzoo.org/)

---

## Connections

### Prerequisites (Week 105)

- Binary symplectic representation
- Stabilizer formalism
- Clifford operations

### Leads to (Week 107)

- CSS code constructions
- Surface codes as graph states
- Topological protection

---

## Summary

Graph states provide a powerful framework connecting graph theory, entanglement, and quantum computation. Their stabilizer structure enables efficient classical description, while measurement-based quantum computation demonstrates their computational universality. Understanding graph states is essential for modern quantum error correction and quantum computing architectures.
