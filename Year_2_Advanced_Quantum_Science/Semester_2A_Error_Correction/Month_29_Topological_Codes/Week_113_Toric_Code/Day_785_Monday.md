# Day 785: Kitaev's Toric Code Introduction

## Overview

**Day:** 785 of 1008
**Week:** 113 (Toric Code Fundamentals)
**Month:** 29 (Topological Codes)
**Topic:** Introduction to Kitaev's Toric Code

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Historical context and torus topology |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Lattice structure and code parameters |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Explain** the historical development of topological quantum codes
2. **Describe** torus topology and periodic boundary conditions
3. **Construct** the square lattice structure with qubits on edges
4. **Derive** the code parameters $[[2L^2, 2, L]]$
5. **Identify** vertices, edges, and faces in the toric lattice
6. **Connect** topological protection to fault tolerance

---

## Historical Background

### Alexei Kitaev's Revolutionary Contribution

In 1997, Alexei Kitaev introduced a radically new approach to quantum error correction that would reshape the field. Rather than using algebraic structures like classical codes, Kitaev leveraged **topology**---the mathematical study of properties preserved under continuous deformations.

**Key insight:** Information encoded in topological properties is inherently protected from local perturbations.

### Timeline of Development

| Year | Milestone |
|------|-----------|
| 1997 | Kitaev proposes toric code (unpublished notes) |
| 2003 | Full paper "Fault-tolerant quantum computation by anyons" |
| 2002 | Dennis et al. prove threshold theorem for toric code |
| 2007 | Raussendorf-Harrington prove high threshold (~1%) |
| 2012 | Fowler et al. establish surface code as leading candidate |
| 2023 | Google demonstrates surface code below threshold |

### Why Topology?

Classical error correction relies on **redundancy**: copying information and using majority voting. Quantum mechanics forbids copying (no-cloning theorem).

**Kitaev's solution:** Encode information in the **global** topology of a system, where:
- Local errors cannot access the encoded information
- Only errors spanning the entire system can cause logical faults
- The larger the system, the more protected the information

---

## Core Theory

### 1. Torus Topology

A **torus** is a surface with the topology of a donut. It can be constructed by:
1. Taking a square $[0, L] \times [0, L]$
2. Identifying opposite edges (periodic boundary conditions)

$$\text{Left edge} \equiv \text{Right edge}$$
$$\text{Top edge} \equiv \text{Bottom edge}$$

**Visualization:**

```
    ┌─────────────────┐
    │                 │
    │    ↑      →     │
    │                 │
    │   Torus T²      │
    │                 │
    └─────────────────┘
    ↑                 ↑
   Same points identified
```

### 2. The Square Lattice

On the torus, we define a square lattice with:
- **Vertices (v):** $L \times L = L^2$ vertices
- **Edges (e):** $2L^2$ edges (L² horizontal + L² vertical)
- **Faces/Plaquettes (p):** $L^2$ square faces

**Euler characteristic for torus:**
$$\chi = V - E + F = L^2 - 2L^2 + L^2 = 0$$

This confirms we're on a torus ($\chi = 0$ for torus, $\chi = 2$ for sphere).

### 3. Qubit Placement

**Key design choice:** Place one qubit on each **edge** of the lattice.

$$\boxed{\text{Number of physical qubits: } n = 2L^2}$$

For an $L \times L$ torus:
- $L^2$ horizontal edges
- $L^2$ vertical edges
- Total: $2L^2$ qubits

**Why edges?** This allows both vertex (star) and face (plaquette) operators to act on 4 qubits each, creating the symmetric structure essential for the code.

### 4. Lattice Indexing

We index the lattice as follows:

**Vertices:** $(i, j)$ where $i, j \in \{0, 1, ..., L-1\}$ with periodic boundaries

**Edges:** $(i, j, d)$ where:
- $d = 0$: horizontal edge from $(i, j)$ to $(i, j+1)$
- $d = 1$: vertical edge from $(i, j)$ to $(i+1, j)$

**Faces:** $(i, j)$ labels the plaquette with top-left corner at vertex $(i, j)$

```
        j      j+1
    ────●──────●────
        │      │
     i  │  Pᵢⱼ │
        │      │
    ────●──────●────
       i+1
```

### 5. Code Parameters: $[[2L^2, 2, L]]$

The toric code is a $[[n, k, d]]$ quantum code where:

**n = 2L² (Physical qubits):**
As derived above, one qubit per edge.

**k = 2 (Logical qubits):**
The torus has two independent non-contractible loops (one around each "hole"). Each contributes one logical qubit.

$$k = \dim H_1(\mathbb{T}^2; \mathbb{Z}_2) = 2$$

where $H_1$ is the first homology group of the torus.

**d = L (Code distance):**
The minimum weight logical operator is a string of length L wrapping around the torus.

$$\boxed{[[n, k, d]] = [[2L^2, 2, L]]}$$

### 6. Encoding Rate

The **rate** of the code is:

$$R = \frac{k}{n} = \frac{2}{2L^2} = \frac{1}{L^2}$$

This decreases as the code grows---a trade-off for increased protection.

**Comparison:**

| Code | Rate | Distance |
|------|------|----------|
| [[7,1,3]] Steane | 1/7 ≈ 0.14 | 3 |
| [[9,1,3]] Shor | 1/9 ≈ 0.11 | 3 |
| [[18,2,3]] Toric (L=3) | 1/9 ≈ 0.11 | 3 |
| [[50,2,5]] Toric (L=5) | 1/25 = 0.04 | 5 |

---

## Quantum Mechanics Connection

### From Local to Topological

Traditional QEC uses local redundancy:
$$|0_L\rangle = |000\rangle, \quad |1_L\rangle = |111\rangle$$

The toric code encodes information in **topological degrees of freedom**:
- The state space is the ground space of a local Hamiltonian
- Logical operators are non-local (wrap around the torus)
- Local errors cannot distinguish logical states

### Connection to Condensed Matter Physics

The toric code Hamiltonian:
$$H = -\sum_v A_v - \sum_p B_p$$

describes a **topologically ordered** phase of matter with:
- Degenerate ground state protected by a gap
- Anyonic excitations (to be studied in Week 114)
- Long-range entanglement

This is the simplest example of a **topological quantum field theory** in 2+1 dimensions.

### The Threshold Theorem for Toric Codes

Dennis, Kitaev, Landahl, and Preskill (2002) proved:

**Theorem:** For independent, identically distributed bit-flip and phase-flip errors with probability $p < p_{th} \approx 10.9\%$, the toric code can protect quantum information indefinitely as $L \to \infty$.

This threshold is remarkably high compared to concatenated codes (~1%).

---

## Worked Examples

### Example 1: Counting Lattice Elements

**Problem:** For a $4 \times 4$ toric code, count the vertices, edges, faces, and verify Euler characteristic.

**Solution:**

Given $L = 4$:

**Vertices:** $V = L^2 = 16$

**Edges:** $E = 2L^2 = 32$

**Faces:** $F = L^2 = 16$

**Euler characteristic:**
$$\chi = V - E + F = 16 - 32 + 16 = 0 \checkmark$$

(Confirms torus topology)

**Physical qubits:** $n = E = 32$

**Code parameters:** $[[32, 2, 4]]$

### Example 2: Converting to Linear Index

**Problem:** In a $3 \times 3$ toric code, find the linear index of the vertical edge from vertex $(1, 2)$ to vertex $(2, 2)$.

**Solution:**

Using the indexing: edge $(i, j, d)$ where $d=1$ for vertical.

Linear index formula:
$$\text{index} = d \cdot L^2 + i \cdot L + j$$

For $(i, j, d) = (1, 2, 1)$ with $L = 3$:
$$\text{index} = 1 \cdot 9 + 1 \cdot 3 + 2 = 9 + 3 + 2 = 14$$

**Verification:**
- Horizontal edges: indices 0-8
- Vertical edges: indices 9-17
- Edge $(1, 2, 1)$: index 14 $\checkmark$

### Example 3: Code Distance Scaling

**Problem:** How many physical qubits are needed for a toric code with distance $d = 10$?

**Solution:**

For toric code: $d = L$, so $L = 10$.

Number of qubits:
$$n = 2L^2 = 2 \times 100 = 200$$

The code is $[[200, 2, 10]]$.

**Interpretation:** 200 physical qubits protect 2 logical qubits with distance 10.

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For a $5 \times 5$ toric code:
a) How many physical qubits are there?
b) How many stabilizer generators exist?
c) What are the code parameters $[[n, k, d]]$?

**P1.2** Draw the $2 \times 2$ toric lattice, labeling all vertices, edges, and faces. How many of each element exists?

**P1.3** Convert the following edge coordinates to linear indices for $L = 4$:
a) Horizontal edge at $(0, 0)$
b) Vertical edge at $(3, 3)$
c) Horizontal edge at $(2, 1)$

### Level 2: Intermediate

**P2.1** Prove that for an $L \times L$ torus, $V - E + F = 0$ algebraically (not just by counting).

**P2.2** The rate of a code is $R = k/n$. For what value of $L$ does the toric code have rate $R = 1/50$?

**P2.3** If we want a toric code with at least 1000 physical qubits and distance at least 20, what is the smallest valid $L$?

### Level 3: Challenging

**P3.1** Consider a rectangular torus with dimensions $L_1 \times L_2$ (not necessarily square). Derive the code parameters $[[n, k, d]]$.

**P3.2** The surface code is a planar version of the toric code with boundaries. Without detailed analysis, explain qualitatively why it encodes only 1 logical qubit instead of 2.

**P3.3** Prove that for any stabilizer code on an $L \times L$ lattice with local (geometrically bounded) stabilizers, the distance satisfies $d \leq O(L)$. Why is $d = L$ optimal for the toric code?

---

## Computational Lab

```python
"""
Day 785: Kitaev's Toric Code Introduction
==========================================

Building the foundational data structures for toric code simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ToricLattice:
    """
    Toric code lattice on L x L torus.

    Vertices: (i, j) with i, j in {0, ..., L-1}
    Edges: (i, j, d) with d=0 (horizontal) or d=1 (vertical)
    Faces: (i, j) labels plaquette with top-left at vertex (i, j)

    All coordinates are periodic (mod L).
    """

    L: int

    def __post_init__(self):
        """Compute derived quantities."""
        self.n_vertices = self.L ** 2
        self.n_edges = 2 * self.L ** 2
        self.n_faces = self.L ** 2

    def vertex_index(self, i: int, j: int) -> int:
        """Convert vertex (i, j) to linear index."""
        i, j = i % self.L, j % self.L
        return i * self.L + j

    def edge_index(self, i: int, j: int, d: int) -> int:
        """
        Convert edge (i, j, d) to linear index.

        d = 0: horizontal edge from (i,j) to (i, j+1)
        d = 1: vertical edge from (i,j) to (i+1, j)
        """
        i, j = i % self.L, j % self.L
        return d * self.L**2 + i * self.L + j

    def face_index(self, i: int, j: int) -> int:
        """Convert face (i, j) to linear index."""
        i, j = i % self.L, j % self.L
        return i * self.L + j

    def edge_from_index(self, idx: int) -> Tuple[int, int, int]:
        """Convert linear index back to (i, j, d)."""
        d = idx // (self.L ** 2)
        rem = idx % (self.L ** 2)
        i = rem // self.L
        j = rem % self.L
        return (i, j, d)

    def vertex_edges(self, i: int, j: int) -> List[int]:
        """
        Return indices of all edges incident to vertex (i, j).

        These are the edges involved in the star operator A_v.
        """
        i, j = i % self.L, j % self.L
        return [
            self.edge_index(i, j, 0),       # right horizontal
            self.edge_index(i, j-1, 0),     # left horizontal
            self.edge_index(i, j, 1),       # down vertical
            self.edge_index(i-1, j, 1),     # up vertical
        ]

    def face_edges(self, i: int, j: int) -> List[int]:
        """
        Return indices of all edges bounding face (i, j).

        These are the edges involved in the plaquette operator B_p.
        """
        i, j = i % self.L, j % self.L
        return [
            self.edge_index(i, j, 0),       # top horizontal
            self.edge_index(i+1, j, 0),     # bottom horizontal
            self.edge_index(i, j, 1),       # left vertical
            self.edge_index(i, j+1, 1),     # right vertical
        ]

    def euler_characteristic(self) -> int:
        """Compute Euler characteristic V - E + F."""
        return self.n_vertices - self.n_edges + self.n_faces

    def code_parameters(self) -> Tuple[int, int, int]:
        """Return [[n, k, d]] code parameters."""
        n = self.n_edges
        k = 2  # Two non-contractible cycles on torus
        d = self.L
        return (n, k, d)


def visualize_toric_lattice(L: int, highlight_vertex: Optional[Tuple[int, int]] = None,
                            highlight_face: Optional[Tuple[int, int]] = None) -> None:
    """
    Visualize the toric code lattice.

    Parameters:
    -----------
    L : int
        Lattice size
    highlight_vertex : tuple, optional
        Vertex (i, j) to highlight (shows star operator edges)
    highlight_face : tuple, optional
        Face (i, j) to highlight (shows plaquette operator edges)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    lattice = ToricLattice(L)

    # Draw edges
    for i in range(L):
        for j in range(L):
            # Horizontal edge
            ax.plot([j, j+1], [L-1-i, L-1-i], 'b-', linewidth=2, alpha=0.5)
            # Vertical edge
            ax.plot([j, j], [L-1-i, L-i], 'b-', linewidth=2, alpha=0.5)

    # Show periodicity
    for i in range(L):
        # Horizontal wrap
        ax.annotate('', xy=(L+0.1, L-1-i), xytext=(L+0.3, L-1-i),
                   arrowprops=dict(arrowstyle='->', color='gray'))
        ax.annotate('', xy=(-0.3, L-1-i), xytext=(-0.1, L-1-i),
                   arrowprops=dict(arrowstyle='->', color='gray'))
        # Vertical wrap
        ax.annotate('', xy=(i, L+0.1), xytext=(i, L+0.3),
                   arrowprops=dict(arrowstyle='->', color='gray'))
        ax.annotate('', xy=(i, -0.3), xytext=(i, -0.1),
                   arrowprops=dict(arrowstyle='->', color='gray'))

    # Draw vertices
    for i in range(L):
        for j in range(L):
            ax.plot(j, L-1-i, 'ko', markersize=10)

    # Label vertices
    for i in range(L):
        for j in range(L):
            ax.annotate(f'({i},{j})', (j, L-1-i), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)

    # Highlight star operator
    if highlight_vertex is not None:
        vi, vj = highlight_vertex
        edges = lattice.vertex_edges(vi, vj)
        for edge_idx in edges:
            ei, ej, ed = lattice.edge_from_index(edge_idx)
            if ed == 0:  # horizontal
                ax.plot([ej % L, (ej+1) % L if (ej+1) % L != 0 else L],
                       [L-1-ei, L-1-ei], 'r-', linewidth=4, label='Star' if edge_idx == edges[0] else '')
            else:  # vertical
                ax.plot([ej, ej], [L-1-ei, L-ei if ei+1 < L else L],
                       'r-', linewidth=4)
        ax.plot(vj, L-1-vi, 'r*', markersize=20)

    # Highlight plaquette operator
    if highlight_face is not None:
        fi, fj = highlight_face
        edges = lattice.face_edges(fi, fj)
        # Draw plaquette outline
        rect = plt.Rectangle((fj + 0.1, L-1-fi - 0.9), 0.8, 0.8,
                             fill=True, facecolor='green', alpha=0.3)
        ax.add_patch(rect)
        for edge_idx in edges:
            ei, ej, ed = lattice.edge_from_index(edge_idx)
            if ed == 0:  # horizontal
                ax.plot([ej, ej+1], [L-1-ei, L-1-ei], 'g-', linewidth=4)
            else:  # vertical
                ax.plot([ej, ej], [L-1-ei, L-ei], 'g-', linewidth=4)

    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, L + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Toric Code Lattice (L = {L})\nPeriodic boundaries: edges wrap around')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('toric_lattice.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_code_parameters(L_values: List[int]) -> None:
    """Analyze toric code parameters for various L."""
    print("=" * 60)
    print("Toric Code Parameters Analysis")
    print("=" * 60)
    print(f"{'L':>5} {'n (qubits)':>12} {'k (logical)':>12} {'d (distance)':>12} {'Rate':>10}")
    print("-" * 60)

    for L in L_values:
        lattice = ToricLattice(L)
        n, k, d = lattice.code_parameters()
        rate = k / n
        print(f"{L:>5} {n:>12} {k:>12} {d:>12} {rate:>10.4f}")


def verify_lattice_structure(L: int) -> None:
    """Verify lattice counting and indexing."""
    print(f"\nLattice Structure Verification (L = {L})")
    print("-" * 40)

    lattice = ToricLattice(L)

    print(f"Vertices: {lattice.n_vertices}")
    print(f"Edges: {lattice.n_edges}")
    print(f"Faces: {lattice.n_faces}")
    print(f"Euler characteristic: {lattice.euler_characteristic()}")

    # Verify edge indexing round-trip
    print(f"\nEdge indexing verification:")
    for idx in [0, L**2 - 1, L**2, 2*L**2 - 1]:
        i, j, d = lattice.edge_from_index(idx)
        recovered = lattice.edge_index(i, j, d)
        direction = "horizontal" if d == 0 else "vertical"
        print(f"  Index {idx} -> ({i}, {j}, {direction}) -> {recovered}")
        assert idx == recovered, "Indexing error!"

    # Verify star operator has 4 edges
    print(f"\nStar operator edges at (0, 0):")
    star_edges = lattice.vertex_edges(0, 0)
    print(f"  {len(star_edges)} edges: {star_edges}")

    # Verify plaquette operator has 4 edges
    print(f"\nPlaquette operator edges at (0, 0):")
    plaq_edges = lattice.face_edges(0, 0)
    print(f"  {len(plaq_edges)} edges: {plaq_edges}")


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 785: KITAEV'S TORIC CODE INTRODUCTION")
    print("=" * 70)

    # Demo 1: Analyze code parameters
    print("\n" + "=" * 70)
    print("Demo 1: Code Parameters for Various Lattice Sizes")
    print("=" * 70)

    analyze_code_parameters([2, 3, 4, 5, 7, 10, 15, 20])

    # Demo 2: Verify lattice structure
    print("\n" + "=" * 70)
    print("Demo 2: Lattice Structure Verification")
    print("=" * 70)

    verify_lattice_structure(3)
    verify_lattice_structure(5)

    # Demo 3: Visualize lattice (uncomment to generate plot)
    print("\n" + "=" * 70)
    print("Demo 3: Lattice Visualization")
    print("=" * 70)

    # Visualize a 4x4 lattice with highlighted operators
    # visualize_toric_lattice(4, highlight_vertex=(1, 1), highlight_face=(1, 2))
    print("(Visualization code ready - uncomment to generate plot)")

    # Demo 4: Historical context
    print("\n" + "=" * 70)
    print("Demo 4: Toric Code in Context")
    print("=" * 70)

    print("""
    KITAEV'S TORIC CODE (1997/2003)
    ===============================

    Key Innovation: Use TOPOLOGY to protect quantum information

    Traditional QEC:
    ----------------
    - Encode in algebraic redundancy
    - |0_L> = |000>, |1_L> = |111>
    - Local errors can flip bits

    Topological QEC:
    ----------------
    - Encode in global topological properties
    - Logical states differ by non-local operations
    - Local errors CANNOT access encoded information

    The Toric Code [[2L^2, 2, L]]:
    ------------------------------
    - n = 2L^2 qubits on edges of L x L torus
    - k = 2 logical qubits (from torus topology)
    - d = L (errors must wrap around torus)

    Why it matters:
    - First example of topological order in QEC
    - High threshold (~10.9% vs ~1% for concatenated codes)
    - Foundation for surface codes (practical implementation)
    - Connects QEC to condensed matter physics (anyons)
    """)

    # Demo 5: Scaling analysis
    print("\n" + "=" * 70)
    print("Demo 5: Resource Scaling")
    print("=" * 70)

    target_distances = [5, 10, 20, 50, 100]
    print(f"\nQubits required for target distance:")
    print(f"{'Distance':>10} {'L':>6} {'Qubits':>10}")
    print("-" * 30)
    for d in target_distances:
        L = d  # For toric code, d = L
        n = 2 * L**2
        print(f"{d:>10} {L:>6} {n:>10}")

    print("\n" + "=" * 70)
    print("Day 785 Complete: Toric Code Foundations Established")
    print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Physical qubits | $n = 2L^2$ |
| Logical qubits | $k = 2$ |
| Code distance | $d = L$ |
| Code parameters | $[[2L^2, 2, L]]$ |
| Euler characteristic | $\chi = V - E + F = 0$ |
| Encoding rate | $R = 1/L^2$ |

### Main Takeaways

1. **Kitaev's toric code** (1997/2003) introduced topological quantum error correction
2. **Qubits on edges** of a square lattice on a torus gives $n = 2L^2$ qubits
3. **Torus topology** with two non-contractible cycles encodes $k = 2$ logical qubits
4. **Code distance** $d = L$ means errors must span the entire torus to cause logical faults
5. **Topological protection** arises from non-local encoding, not algebraic redundancy
6. **High threshold** (~10.9%) makes topological codes practically attractive

---

## Daily Checklist

- [ ] I understand why Kitaev chose topological encoding
- [ ] I can construct an $L \times L$ toric lattice with proper indexing
- [ ] I can count vertices, edges, and faces and verify Euler characteristic
- [ ] I can derive code parameters $[[2L^2, 2, L]]$
- [ ] I understand why k = 2 (from torus homology)
- [ ] I can run the computational lab and interpret outputs

---

## Preview: Day 786

Tomorrow we dive into the **star and plaquette operators**:

- Explicit construction of $A_v = \prod_{e \ni v} X_e$
- Explicit construction of $B_p = \prod_{e \in \partial p} Z_e$
- Proof that all operators commute: $[A_v, B_p] = 0$
- Stabilizer group structure and independence constraints

These operators are the heart of the toric code---they define what it means to have no errors.
