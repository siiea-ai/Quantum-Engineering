# Day 755: Advanced Code Constructions

## Overview

**Day:** 755 of 1008
**Week:** 108 (Code Families & Construction Techniques)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Cutting-Edge Quantum Code Designs

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Subsystem and Floquet codes |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Homological and beyond |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Research frontiers |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Explain** subsystem codes and gauge operators
2. **Describe** Floquet codes and measurement-based error correction
3. **Understand** homological code constructions
4. **Identify** codes beyond the stabilizer formalism
5. **Recognize** current research frontiers
6. **Appreciate** the diversity of quantum code designs

---

## Subsystem Codes

### Motivation

**Problem with stabilizer codes:** Measuring high-weight stabilizers is difficult.

**Solution:** Subsystem codes allow measuring lower-weight operators.

### Definition

A **subsystem code** encodes logical information in a subsystem of the code space:
$$\mathcal{H}_{code} = \mathcal{H}_{logical} \otimes \mathcal{H}_{gauge}$$

The gauge subsystem can be in any state—we don't care about it!

### Gauge Group

**Gauge group G:** Includes stabilizer group S as a subgroup.
$$S \subseteq G, \quad [G, G] \subseteq S$$

- Stabilizers: Must have +1 eigenvalue
- Gauge operators: Eigenvalue doesn't matter

### Bacon-Shor Codes

The **Bacon-Shor code** is a subsystem code on a 2D grid.

**Gauge operators:**
- $X_i X_j$ for horizontal pairs
- $Z_i Z_j$ for vertical pairs

**Stabilizers:** Products of gauge operators
- $\prod_{\text{row}} X_i X_j$
- $\prod_{\text{col}} Z_i Z_j$

**Advantage:** Measure only weight-2 gauge operators!

### Parameters

Bacon-Shor [[n², 1, n]]:
- n² physical qubits
- 1 logical qubit
- Distance n

---

## Floquet Codes

### Dynamic Error Correction

**Floquet codes:** Error correction through periodic measurement sequences.

**Key idea:** The code itself changes with each measurement round!

### Honeycomb Code

**Structure:** Qubits on vertices of honeycomb lattice.

**Measurement schedule:**
- Round 1: Measure XX on red edges
- Round 2: Measure YY on green edges
- Round 3: Measure ZZ on blue edges
- Repeat...

**Remarkable:** The instantaneous code is NOT a stabilizer code, but errors are still corrected!

### Automorphism Codes

**Automorphism codes:** Generalize Floquet codes using automorphisms of the Pauli group.

**Sequence:** Apply automorphism, measure, repeat.

### Advantages

- Lower-weight measurements
- Potential for higher thresholds
- Natural fit for certain hardware

---

## Homological Codes

### Chain Complex Perspective

**Homological codes** arise from chain complexes:
$$C_3 \xrightarrow{\partial_3} C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0$$

- Qubits on cells of dimension 1
- X stabilizers from boundaries
- Z stabilizers from coboundaries

### Surface Codes Revisited

Surface codes are homological codes on 2D:
$$C_2 \xrightarrow{\partial} C_1 \xrightarrow{\partial} C_0$$

**k = dim(H₁)** = first Betti number = number of "holes"

### Higher-Dimensional Codes

**3D codes:** Can have more interesting topology.

**4D toric code:** Self-correcting (thermal stability)!

### Fiber Bundle Codes

Combine base and fiber structures:
- Good classical codes as fibers
- Expander graphs as base
- Achieve almost-linear distance

---

## Beyond Stabilizer Formalism

### Non-Stabilizer Codes

**Permutation-invariant codes:**
- Use symmetric subspace
- Not describable by stabilizers

**Approximate codes:**
- Allow small encoding errors
- Can achieve better parameters

### Bosonic Codes

**Cat codes:**
$$|0_L\rangle = |+\alpha\rangle + |-\alpha\rangle$$
$$|1_L\rangle = |+\alpha\rangle - |-\alpha\rangle$$

Encode in superpositions of coherent states.

**GKP codes:** Encode in grid states of harmonic oscillator.

### Topological Codes

**Anyonic codes:** Based on topological phases of matter.
- Fibonacci anyons: Universal computation
- Non-Abelian statistics provide protection

---

## Research Frontiers

### Current Challenges

1. **Decoding:** Efficient decoders for new code families
2. **Threshold:** Understanding noise thresholds
3. **Implementation:** Hardware-efficient designs
4. **Fault tolerance:** Complete protocols

### Active Areas

| Area | Key Question |
|------|--------------|
| qLDPC | Can we decode efficiently? |
| Floquet | What are optimal schedules? |
| Bosonic | Hardware implementation? |
| Topological | Practical anyon systems? |

### Recent Breakthroughs

- 2021: Good qLDPC codes exist
- 2022: Quantum Tanner codes
- 2023: Improved thresholds for Floquet codes
- Ongoing: Practical implementations

---

## Code Comparison

### Summary Table

| Code Family | Rate | Distance | Weight | Special Feature |
|-------------|------|----------|--------|-----------------|
| Surface | 1/n | √n | 4 | High threshold |
| Color | 1/n | √n | Varies | Transversal Clifford |
| Bacon-Shor | 1/n | √n | 2 | Subsystem |
| Floquet | Varies | Varies | 2 | Dynamic |
| Good qLDPC | Θ(1) | Θ(n) | O(1) | Optimal scaling |

### Trade-offs

- **Threshold vs Overhead:** Surface codes have high threshold but poor scaling
- **Weight vs Complexity:** Lower weight often means more complex structure
- **Static vs Dynamic:** Floquet codes trade simplicity for flexibility

---

## Worked Examples

### Example 1: Bacon-Shor Gauge Operators

**Problem:** For 3×3 Bacon-Shor code, list gauge operators on the first row.

**Solution:**

Qubits labeled 1-9:
```
1 - 2 - 3
|   |   |
4 - 5 - 6
|   |   |
7 - 8 - 9
```

First row gauge operators (X type):
- $X_1 X_2$ (horizontal)
- $X_2 X_3$ (horizontal)

First column gauge operators (Z type):
- $Z_1 Z_4$ (vertical)
- $Z_4 Z_7$ (vertical)

Row stabilizer: $X_1 X_2 \cdot X_2 X_3 = X_1 X_3$

### Example 2: Floquet Measurement Sequence

**Problem:** Describe one period of honeycomb Floquet code measurements.

**Solution:**

Honeycomb lattice edges colored red, green, blue.

**Period:**
1. Measure $X_i X_j$ on all red edges
2. Measure $Y_i Y_j$ on all green edges
3. Measure $Z_i Z_j$ on all blue edges

After each round, the "code" changes, but the logical information is preserved through the sequence.

### Example 3: Homological Parameters

**Problem:** A 2D surface has 100 vertices, 200 edges, 99 faces. What code parameters?

**Solution:**

Euler characteristic: $\chi = V - E + F = 100 - 200 + 99 = -1$

For orientable surface: $\chi = 2 - 2g$ where g is genus.
$-1 = 2 - 2g \Rightarrow g = 1.5$ (not integer, so non-orientable)

Actually, for surface code:
- n = E = 200 (qubits on edges)
- Number of X checks ≈ F = 99
- Number of Z checks ≈ V = 100
- k = n - (V-1) - (F-1) = 200 - 99 - 98 = 3

(Depends on boundary conditions)

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For a 4×4 Bacon-Shor code, how many gauge operators are there?

**P1.2** In Floquet honeycomb code, what operator is measured on a blue edge?

**P1.3** If a surface has genus 2, what is k for the associated toric code?

### Level 2: Intermediate

**P2.1** Show that the product of all X-type gauge operators in Bacon-Shor gives a stabilizer.

**P2.2** Explain why Floquet codes can have lower-weight measurements than static codes.

**P2.3** For a 3D toric code on a 3-torus, compute k using homology.

### Level 3: Challenging

**P3.1** Prove that subsystem codes can have single-shot error correction.

**P3.2** Analyze the threshold of honeycomb Floquet code under circuit-level noise.

**P3.3** Design a hybrid code combining surface code and subsystem features.

---

## Computational Lab

```python
"""
Day 755: Advanced Code Constructions
====================================

Exploring subsystem codes and beyond.
"""

import numpy as np
from typing import List, Tuple, Set


class BaconShorCode:
    """
    Bacon-Shor subsystem code on L × L grid.

    Gauge operators: X_iX_j (horizontal), Z_iZ_j (vertical)
    """

    def __init__(self, L: int):
        self.L = L
        self.n = L * L

        self._build_gauge_operators()
        self._build_stabilizers()

    def _qubit_index(self, row: int, col: int) -> int:
        """Convert (row, col) to qubit index."""
        return row * self.L + col

    def _build_gauge_operators(self):
        """Build X and Z gauge operators."""
        self.x_gauge = []  # Horizontal XX
        self.z_gauge = []  # Vertical ZZ

        for row in range(self.L):
            for col in range(self.L - 1):
                i = self._qubit_index(row, col)
                j = self._qubit_index(row, col + 1)
                self.x_gauge.append((i, j))

        for row in range(self.L - 1):
            for col in range(self.L):
                i = self._qubit_index(row, col)
                j = self._qubit_index(row + 1, col)
                self.z_gauge.append((i, j))

    def _build_stabilizers(self):
        """Build stabilizers from gauge products."""
        self.x_stabs = []  # Row stabilizers
        self.z_stabs = []  # Column stabilizers

        # Row stabilizers: product of XX along row
        for row in range(self.L):
            stab = set()
            for col in range(self.L):
                stab.add(self._qubit_index(row, col))
            self.x_stabs.append(stab)

        # Column stabilizers: product of ZZ along column
        for col in range(self.L):
            stab = set()
            for row in range(self.L):
                stab.add(self._qubit_index(row, col))
            self.z_stabs.append(stab)

    def logical_operators(self) -> Tuple[Set[int], Set[int]]:
        """Return logical X and Z operators."""
        # Logical X: entire row
        X_L = set(range(self.L))

        # Logical Z: entire column
        Z_L = {self._qubit_index(row, 0) for row in range(self.L)}

        return X_L, Z_L

    def gauge_to_string(self, gauge: Tuple[int, int], pauli: str) -> str:
        """Convert gauge operator to string."""
        i, j = gauge
        s = ['I'] * self.n
        s[i] = pauli
        s[j] = pauli
        return ''.join(s)

    def __repr__(self) -> str:
        return f"Bacon-Shor [[{self.n}, 1, {self.L}]]"


class FloquetSchedule:
    """
    Simple Floquet code schedule.
    """

    def __init__(self, n_qubits: int, schedule: List[List[Tuple[int, int, str]]]):
        """
        Initialize Floquet schedule.

        schedule: List of rounds, each round is list of (i, j, pauli)
        """
        self.n = n_qubits
        self.schedule = schedule
        self.period = len(schedule)

    def round_measurements(self, round_idx: int) -> List[str]:
        """Get measurement operators for a round."""
        round_idx = round_idx % self.period
        measurements = []
        for i, j, pauli in self.schedule[round_idx]:
            s = ['I'] * self.n
            s[i] = pauli
            s[j] = pauli
            measurements.append(''.join(s))
        return measurements


def honeycomb_floquet(L: int) -> FloquetSchedule:
    """
    Create honeycomb Floquet code schedule.

    Simplified version for demonstration.
    """
    # This would build actual honeycomb lattice
    # For now, return simple 3-round schedule
    n = 2 * L * L  # Approximate

    schedule = [
        [(0, 1, 'X'), (2, 3, 'X')],  # Red edges
        [(0, 2, 'Y'), (1, 3, 'Y')],  # Green edges
        [(0, 3, 'Z'), (1, 2, 'Z')],  # Blue edges
    ]

    return FloquetSchedule(4, schedule)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 755: Advanced Code Constructions")
    print("=" * 60)

    # Example 1: Bacon-Shor code
    print("\n1. Bacon-Shor Code")
    print("-" * 40)

    for L in [3, 4, 5]:
        bs = BaconShorCode(L)
        print(f"{bs}")
        print(f"  X gauge operators: {len(bs.x_gauge)}")
        print(f"  Z gauge operators: {len(bs.z_gauge)}")
        print(f"  X stabilizers: {len(bs.x_stabs)}")
        print(f"  Z stabilizers: {len(bs.z_stabs)}")

    # Example 2: Gauge operators
    print("\n2. Gauge Operator Details (3×3)")
    print("-" * 40)

    bs = BaconShorCode(3)
    print("X gauge operators (horizontal XX):")
    for i, g in enumerate(bs.x_gauge[:3]):
        print(f"  {g}: {bs.gauge_to_string(g, 'X')}")

    print("\nZ gauge operators (vertical ZZ):")
    for i, g in enumerate(bs.z_gauge[:3]):
        print(f"  {g}: {bs.gauge_to_string(g, 'Z')}")

    # Example 3: Floquet schedule
    print("\n3. Floquet Code Schedule")
    print("-" * 40)

    floquet = honeycomb_floquet(2)
    print(f"Period: {floquet.period} rounds")

    for r in range(floquet.period):
        print(f"\nRound {r+1}:")
        for m in floquet.round_measurements(r):
            print(f"  {m}")

    # Example 4: Comparison
    print("\n4. Code Family Comparison")
    print("-" * 40)

    print(f"{'Code':<20} {'Rate':<12} {'Distance':<12} {'Meas. Weight':<12}")
    print("-" * 56)

    codes = [
        ("Surface (d=5)", "1/25", "5", "4"),
        ("Color (d=5)", "1/19", "5", "6"),
        ("Bacon-Shor (5×5)", "1/25", "5", "2"),
        ("Good qLDPC", "~0.1", "~n/10", "O(1)"),
    ]

    for name, rate, dist, weight in codes:
        print(f"{name:<20} {rate:<12} {dist:<12} {weight:<12}")

    # Example 5: Research directions
    print("\n5. Active Research Areas")
    print("-" * 40)

    areas = [
        "Good qLDPC decoding",
        "Floquet thresholds",
        "Bosonic code implementation",
        "Topological quantum memory",
        "Hardware-efficient codes"
    ]

    for area in areas:
        print(f"  • {area}")

    print("\n" + "=" * 60)
    print("Quantum codes: a rich and active research frontier!")
    print("=" * 60)
```

---

## Summary

### Key Concepts

| Code Type | Key Feature |
|-----------|-------------|
| Subsystem | Gauge operators, low-weight measurements |
| Floquet | Dynamic, periodic measurement sequences |
| Homological | From chain complexes, topological |
| Bosonic | Continuous-variable encoding |

### Main Takeaways

1. **Subsystem codes** trade rate for simpler measurements
2. **Floquet codes** use time-varying measurement patterns
3. **Homological codes** connect to algebraic topology
4. **Beyond stabilizer:** Bosonic and topological codes
5. Active research continues to expand the landscape

---

## Daily Checklist

- [ ] I understand gauge operators in subsystem codes
- [ ] I can describe Floquet code measurement schedules
- [ ] I know the connection between codes and homology
- [ ] I'm aware of non-stabilizer code families
- [ ] I can identify current research directions
- [ ] I appreciate the diversity of code designs

---

## Preview: Day 756

Tomorrow we complete **Month 27** with a comprehensive synthesis:

- Review of all four weeks
- Integration across topics
- Master formula sheet
- Preparation for Month 28

This month has established the complete foundation for quantum error correction!
