# Week 116: Error Chains & Logical Operations

## Overview

**Days:** 806-812 (7 days)
**Month:** 29 (Topological Codes)
**Topic:** Error Chains, Decoding, and Logical Operations in Surface Codes

---

## Status: IN PROGRESS

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 806 | Monday | Error Chains and Homology | Not Started |
| 807 | Tuesday | MWPM for Surface Codes | Not Started |
| 808 | Wednesday | Logical Error Rate Scaling | Not Started |
| 809 | Thursday | Lattice Surgery Operations | Not Started |
| 810 | Friday | Transversal and Non-Transversal Gates | Not Started |
| 811 | Saturday | Advanced Topological Operations | Not Started |
| 812 | Sunday | Month 29 Synthesis | Not Started |

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Represent** errors as 1-chains on the lattice and syndromes as boundaries
2. **Apply** minimum weight perfect matching (MWPM) for decoding
3. **Calculate** logical error rate scaling below threshold
4. **Implement** lattice surgery for logical CNOT gates
5. **Distinguish** transversal from non-transversal gate implementations
6. **Analyze** magic state distillation for universal computation
7. **Explore** advanced operations: twist defects, code deformation
8. **Synthesize** the complete Month 29 framework

---

## Core Concepts

### Error Chains as Homology

Errors in surface codes form **1-chains** on the lattice:

$$\boxed{\text{Error chain } E = \sum_e c_e \cdot e, \quad c_e \in \mathbb{Z}_2}$$

The **syndrome** is the boundary of the error chain:

$$\boxed{\partial E = \text{syndrome locations}}$$

### Decoding with MWPM

Minimum weight perfect matching pairs syndromes to minimize total error probability:

$$\boxed{\text{Decoder: } \min_{E : \partial E = \text{syndrome}} |E|}$$

### Logical Error Rate Scaling

Below threshold, the logical error rate scales as:

$$\boxed{p_L \sim \left(\frac{p}{p_{th}}\right)^{d/2}}$$

where $d$ is the code distance and $p_{th} \approx 0.7\%$ (circuit-level) to $1\%$ (phenomenological).

### Lattice Surgery

Logical operations via merging and splitting surface code patches:

| Operation | Effect | Physical Implementation |
|-----------|--------|------------------------|
| Merge | Measure $\bar{Z}_1 \bar{Z}_2$ or $\bar{X}_1 \bar{X}_2$ | Turn off boundary stabilizers |
| Split | Prepare entangled state | Re-establish boundaries |
| CNOT | $\bar{X}_1 \to \bar{X}_1 \bar{X}_2$, $\bar{Z}_2 \to \bar{Z}_1 \bar{Z}_2$ | Merge-split sequence |

### Gate Implementation

| Gate Type | Surface Code Method | Resources |
|-----------|-------------------|-----------|
| Pauli | Transversal or software | Minimal |
| Clifford | Lattice surgery | O(d) time |
| T-gate | Magic state injection | T-factory |
| Arbitrary | Compiled from T+Clifford | Many T-states |

---

## Weekly Breakdown

### Day 806: Error Chains and Homology
- Errors as 1-chains on the lattice graph
- Syndrome as the boundary operator $\partial$
- Homology class determines logical effect
- Equivalent error chains differ by stabilizers
- Minimum weight representative

### Day 807: MWPM for Surface Codes
- Syndrome graph construction with virtual boundaries
- Edge weights from error probabilities
- Blossom algorithm for perfect matching
- Handling Y-errors (correlated X and Z)
- Decoder performance metrics

### Day 808: Logical Error Rate Scaling
- Below-threshold exponential suppression
- Phenomenological vs circuit-level thresholds
- Resource overhead: qubits scale as O(d^2)
- Space-time tradeoffs
- Comparison with concatenated codes

### Day 809: Lattice Surgery Operations
- Merge operation: measure multi-qubit product
- Split operation: prepare Bell pair from patch
- CNOT implementation via ZZ-merge, XX-split
- Temporal scheduling for parallel operations
- Defect-based alternatives

### Day 810: Transversal and Non-Transversal Gates
- Transversal Hadamard (requires square geometry)
- Non-transversal T via state injection
- Magic state distillation protocols
- T-factory architecture design
- Resource costs for universal computation

### Day 811: Advanced Topological Operations
- Twist defects: Majorana-like endpoints
- Topological charge and braiding
- Code deformation techniques
- Connection to color codes
- Future: non-Abelian anyons for braiding-based QC

### Day 812: Month 29 Synthesis
- Complete topological QEC framework
- Key formulas from all 4 weeks
- Performance comparison of topological approaches
- Open problems and research frontiers
- Preparation for Month 30: Beyond Topological Codes

---

## Key Equations

**Error Chain:**
$$\boxed{E \in C_1(\mathcal{L}; \mathbb{Z}_2)}$$

**Syndrome:**
$$\boxed{\sigma = \partial E}$$

**Homology Condition for Logical Error:**
$$\boxed{[E] \neq 0 \in H_1(\mathcal{L}; \mathbb{Z}_2) \Rightarrow \text{logical error}}$$

**Logical Error Rate:**
$$\boxed{p_L \approx C \cdot \binom{d}{(d+1)/2} p^{(d+1)/2}}$$

**Lattice Surgery Merge:**
$$\boxed{M_{ZZ}: |00\rangle \pm |11\rangle \leftrightarrow \pm1 \text{ outcome}}$$

**Magic State:**
$$\boxed{|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)}$$

---

## Computational Skills

```python
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import List, Tuple, Dict

class SurfaceCodeDecoder:
    """
    MWPM decoder for surface codes.

    Uses syndrome graph with virtual boundary vertices.
    """

    def __init__(self, d: int, p: float):
        """
        Initialize decoder for distance-d surface code.

        Args:
            d: Code distance
            p: Physical error probability
        """
        self.d = d
        self.p = p
        self.log_odds = np.log((1-p)/p)  # For edge weights

    def construct_syndrome_graph(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Build weighted graph for MWPM.

        Nodes: syndrome locations + virtual boundary
        Edges: weighted by Manhattan distance × log(p/(1-p))
        """
        syndrome_locs = np.argwhere(syndrome)
        n_syndromes = len(syndrome_locs)

        # Add virtual boundary vertex
        n_nodes = n_syndromes + 1
        graph = np.zeros((n_nodes, n_nodes))

        # Syndrome-syndrome edges
        for i in range(n_syndromes):
            for j in range(i+1, n_syndromes):
                dist = np.abs(syndrome_locs[i] - syndrome_locs[j]).sum()
                weight = dist * self.log_odds
                graph[i, j] = graph[j, i] = weight

        # Syndrome-boundary edges (distance to nearest boundary)
        for i, loc in enumerate(syndrome_locs):
            boundary_dist = min(loc[0], loc[1],
                               self.d - 1 - loc[0], self.d - 1 - loc[1])
            graph[i, -1] = graph[-1, i] = boundary_dist * self.log_odds

        return graph

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode syndrome using MWPM.

        Returns:
            Correction operator as qubit array
        """
        # Build graph and run MWPM
        graph = self.construct_syndrome_graph(syndrome)
        matching = self.mwpm(graph)

        # Convert matching to correction
        return self.matching_to_correction(matching, syndrome)
```

---

## Prerequisites

### From Week 115
- Surface code with rough/smooth boundaries
- Logical operators from boundary topology
- Code distance and parameters

### Mathematical Background
- Chain complexes and boundary operators
- Homology groups over Z₂
- Graph algorithms (matching)

---

## References

### Primary Sources
- Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (2012)
- Horsman et al., "Surface code quantum computing by lattice surgery" (2012)
- Litinski, "A Game of Surface Codes" (2019)

### Key Papers
- Dennis et al., "Topological quantum memory" (2002)
- Bombin, "Clifford gates by code deformation" (2010)
- Bravyi & Kitaev, "Universal quantum computation with ideal Clifford gates" (2005)

### Online Resources
- [Surface Code - Error Correction Zoo](https://errorcorrectionzoo.org/c/surface)
- [Lattice Surgery Tutorial](https://arxiv.org/abs/1704.08670)
- [MWPM Decoder - PyMatching](https://pymatching.readthedocs.io/)

---

## Connections

### From Previous Weeks
- Week 113: Toric code fundamentals → Lattice structure
- Week 114: Anyons → Physical interpretation of errors
- Week 115: Surface code boundaries → Practical architecture

### To Future Topics
- Month 30: Quantum LDPC codes and beyond-surface-code approaches
- Color codes and transversal non-Clifford gates
- Measurement-based topological QC

---

## Summary

Week 116 completes our Month 29 study of topological codes by examining how errors form chains, how decoders correct them, and how logical operations are performed. The MWPM decoder achieves near-optimal error correction by treating syndromes as graph vertices and finding minimum weight matchings. Lattice surgery enables fault-tolerant logical gates by merging and splitting code patches. Magic state distillation extends the gate set to universality. This week synthesizes the entire month's content, establishing the foundation for practical fault-tolerant quantum computing with surface codes.
