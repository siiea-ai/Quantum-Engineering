# Day 746: Surface Codes as CSS

## Overview

**Day:** 746 of 1008
**Week:** 107 (CSS Codes & Related Constructions)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Surface Codes and Their CSS Structure

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Toric code structure |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Planar codes and boundaries |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Construct** the toric code on a periodic lattice
2. **Identify** stabilizer generators from vertices and faces
3. **Derive** surface code parameters [[n, k, d]]
4. **Understand** logical operators as topological cycles
5. **Distinguish** toric codes from planar codes
6. **Analyze** boundary conditions and their effects

---

## The Toric Code

### Lattice Structure

The **toric code** is defined on a 2D square lattice with **periodic boundary conditions** (torus topology).

**Components:**
- **Qubits:** On edges of the lattice
- **Vertices:** Define Z stabilizers (star operators)
- **Faces (plaquettes):** Define X stabilizers (plaquette operators)

For an L × L lattice:
- n = 2L² qubits (edges)
- L² vertices → L² - 1 independent Z stabilizers
- L² faces → L² - 1 independent X stabilizers
- k = 2 logical qubits

### Stabilizer Operators

**Star (Vertex) Operator:**
$$A_v = \prod_{e \ni v} Z_e$$

For each vertex v, apply Z to all 4 adjacent edges.

**Plaquette (Face) Operator:**
$$B_p = \prod_{e \in \partial p} X_e$$

For each plaquette p, apply X to all 4 boundary edges.

### CSS Structure

The toric code is CSS:
- **X stabilizers:** Plaquette operators (all X)
- **Z stabilizers:** Star operators (all Z)

**Commutation:** A star and plaquette share 0 or 2 edges.
- 0 edges: trivially commute
- 2 edges: $X^2 Z^2 = I$, so commute ✓

### Toric Code Parameters

$$\boxed{[[2L^2, 2, L]]}$$

- n = 2L² (number of edges)
- k = 2 (two logical qubits)
- d = L (minimum weight of logical operator)

**Why d = L?**
Logical operators must wrap around the torus. Minimum length = L edges.

### Logical Operators

**Logical $\bar{Z}_1$:** Z on edges forming a horizontal cycle
$$\bar{Z}_1 = \prod_{e \in C_h} Z_e$$

**Logical $\bar{X}_1$:** X on edges forming a vertical cycle
$$\bar{X}_1 = \prod_{e \in C_v} X_e$$

**Second logical qubit:** Cycles in perpendicular direction.

$$\bar{Z}_2 = \prod_{e \in C_v'} Z_e, \quad \bar{X}_2 = \prod_{e \in C_h'} X_e$$

---

## Planar (Surface) Code

### Boundary Conditions

The **planar code** replaces periodic boundaries with open boundaries.

**Rough boundary:** Missing vertex stabilizers (Z)
**Smooth boundary:** Missing plaquette stabilizers (X)

### Standard Surface Code

Typical configuration:
- Top/bottom: Rough boundaries
- Left/right: Smooth boundaries

This gives k = 1 logical qubit.

### Parameters

For d × d planar code:
$$\boxed{[[d^2 + (d-1)^2, 1, d]] \approx [[2d^2 - 2d + 1, 1, d]]}$$

**Simplified:** $[[n, 1, d]]$ with $n \approx 2d^2$

### Logical Operators in Planar Code

**Logical Z:** Chain of Z operators from rough boundary to rough boundary

**Logical X:** Chain of X operators from smooth boundary to smooth boundary

---

## CSS Interpretation

### Classical Codes for Surface Code

The surface code CSS structure uses:

**For toric code:**
- C₁: Classical code associated with vertex stabilizers
- C₂: Classical code associated with face stabilizers

The CSS condition $C_2^\perp \subseteq C_1$ is automatically satisfied by the lattice geometry.

### Chain Complex Perspective

Surface codes arise from chain complexes:

$$C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0$$

- C₂ = faces
- C₁ = edges (qubits)
- C₀ = vertices

**X stabilizers:** Image of ∂₂ (face boundaries)
**Z stabilizers:** Coboundary of ∂₁ (vertex coboundaries)

### Homological Quantum Codes

$$k = \dim H_1(M) = \text{first Betti number}$$

For torus: $H_1(T^2) = \mathbb{Z}^2$, so k = 2
For disk: $H_1(D^2) = 0$, so k = 0
For planar with boundaries: k = 1

---

## Toric Code Detailed Analysis

### Small Example: 2×2 Toric Code

**Setup:**
- 2×2 lattice on torus
- 8 edges (qubits)
- 4 vertices
- 4 faces

**Qubits:** Number edges 1-8

```
    ←1→   ←2→
   ↑   ↑   ↑
   3   4   3
   ↓   ↓   ↓
    ←5→   ←6→
   ↑   ↑   ↑
   7   8   7
   ↓   ↓   ↓
    ←1→   ←2→
```

(Periodic: edges wrap around)

**Star operators:**
- $A_1 = Z_1 Z_3 Z_5 Z_7$
- $A_2 = Z_2 Z_4 Z_6 Z_8$
- $A_3 = Z_1 Z_3 Z_5 Z_7$ (= A₁, due to periodicity)
- $A_4 = Z_2 Z_4 Z_6 Z_8$ (= A₂)

Actually only 2 independent! Plus global constraint: $\prod A_v = I$.

For L=2: 4 vertices, 4 faces, but only 4+4-2 = 6 independent stabilizers.
n = 8, so k = 8 - 6 = 2. ✓

### Error Detection and Correction

**X error on edge e:**
- Creates pair of defects at endpoints of e
- Defects = violated star operators

**Z error on edge e:**
- Creates pair of defects on adjacent faces
- Defects = violated plaquette operators

**Decoding:** Pair up defects, find minimum weight matching.

---

## Surface Code Distance

### Minimum Weight Logical Operators

**Logical operator requirement:**
- Commutes with all stabilizers
- Not in stabilizer group

For toric code:
- Logical Z: Cycle in Z operators around torus
- Minimum weight = L (horizontal or vertical cycle)

**Distance:** $d = L$

### Error Threshold

Surface codes have high threshold (~1%) for:
- Depolarizing noise
- Biased noise (even higher)

**Why?** Local errors create local defects; decoding is tractable.

---

## Rotated Surface Code

### Improved Layout

The **rotated surface code** achieves same [[d², 1, d]] with fewer qubits than standard planar code.

**Key insight:** Rotate lattice 45° and use data qubits at vertices.

**Parameters:** $[[d^2, 1, d]]$ exactly (not $2d^2$)

### Example: d=3 Rotated Code

```
    ●───●───●
    │ X │ Z │
    ●───●───●
    │ Z │ X │
    ●───●───●
```

9 data qubits, distance 3.

---

## Worked Examples

### Example 1: Count Toric Code Parameters

**Problem:** For a 3×3 toric code, find n, k, d.

**Solution:**

L = 3

n = 2L² = 2(9) = 18 qubits

Vertices: L² = 9
Faces: L² = 9

Independent stabilizers:
- Star operators: 9 - 1 = 8 (global constraint)
- Plaquette operators: 9 - 1 = 8 (global constraint)
- Total: 16

k = n - (independent stabilizers) = 18 - 16 = 2 ✓

d = L = 3

**Result:** [[18, 2, 3]]

### Example 2: Verify CSS Commutation

**Problem:** Show that $A_v$ and $B_p$ commute for any vertex v and plaquette p.

**Solution:**

Case 1: v not adjacent to p
- $A_v$ and $B_p$ act on disjoint edges
- Trivially commute ✓

Case 2: v is a corner of p
- Share exactly 2 edges: e₁ and e₂
- $A_v$ has $Z_{e_1} Z_{e_2}$
- $B_p$ has $X_{e_1} X_{e_2}$
- $Z_{e_1} X_{e_1} = -X_{e_1} Z_{e_1}$ (anticommute)
- $Z_{e_2} X_{e_2} = -X_{e_2} Z_{e_2}$ (anticommute)
- Two minus signs → overall commute ✓

### Example 3: Find Logical Operators

**Problem:** For 2×2 toric code, write explicit logical operators.

**Solution:**

Label edges:
- Horizontal: 1, 2 (top), 5, 6 (bottom)
- Vertical: 3, 4 (left column), 7, 8 (right column)

**Logical $\bar{Z}_1$:** Horizontal cycle at row 1
$$\bar{Z}_1 = Z_1 Z_2$$

**Logical $\bar{X}_1$:** Vertical cycle at column 1
$$\bar{X}_1 = X_3 X_7$$

**Verify anticommutation:**
- $\bar{Z}_1$ on edges {1, 2}
- $\bar{X}_1$ on edges {3, 7}
- Overlap: edges 1,2 and 3,7 share...

Actually, need to check based on lattice connectivity. In standard toric code, the Z cycle and X cycle intersect exactly once.

**Second logical qubit:** Perpendicular cycles.

$$\bar{Z}_2 = Z_3 Z_7, \quad \bar{X}_2 = X_1 X_5$$

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For a 4×4 toric code, compute:
a) Number of physical qubits
b) Number of logical qubits
c) Code distance

**P1.2** Draw the star operator $A_v$ for a vertex with 4 adjacent edges labeled a, b, c, d.

**P1.3** How many X stabilizer generators does an L×L planar code have?

### Level 2: Intermediate

**P2.1** Prove that $\prod_v A_v = I$ (product of all star operators is identity).

**P2.2** For the planar code with rough top/bottom and smooth left/right boundaries:
a) Draw the stabilizer structure for a 3×3 lattice
b) Count independent stabilizers
c) Verify k = 1

**P2.3** Show that the minimum weight logical Z operator in the toric code has weight exactly L.

### Level 3: Challenging

**P3.1** Prove that the toric code is self-dual under X ↔ Z exchange (up to lattice rotation).

**P3.2** Design a surface code on a different topology (e.g., Klein bottle) and determine k.

**P3.3** Analyze the CSS structure of surface codes: what classical codes C₁ and C₂ do they correspond to?

---

## Computational Lab

```python
"""
Day 746: Surface Codes as CSS
=============================

Implementing toric and planar surface codes.
"""

import numpy as np
from typing import List, Tuple, Set, Dict
import matplotlib.pyplot as plt


class ToricCode:
    """
    Toric code on L × L lattice with periodic boundaries.

    Qubits on edges, stabilizers on vertices (Z) and faces (X).
    """

    def __init__(self, L: int):
        """
        Initialize L × L toric code.

        Parameters:
        -----------
        L : int
            Linear size of lattice
        """
        self.L = L
        self.n = 2 * L * L  # Number of qubits (edges)
        self.k = 2  # Logical qubits (genus 1 surface)
        self.d = L  # Distance

        # Label edges: horizontal (0 to L²-1), vertical (L² to 2L²-1)
        self._build_stabilizers()

    def _edge_index(self, x: int, y: int, direction: str) -> int:
        """Get edge index. direction = 'h' (horizontal) or 'v' (vertical)."""
        x, y = x % self.L, y % self.L
        if direction == 'h':
            return y * self.L + x
        else:  # vertical
            return self.L * self.L + y * self.L + x

    def _build_stabilizers(self):
        """Build star (Z) and plaquette (X) stabilizers."""
        L = self.L
        self.star_ops = []  # Z stabilizers
        self.plaq_ops = []  # X stabilizers

        # Star operators (vertices)
        for y in range(L):
            for x in range(L):
                # Edges adjacent to vertex (x, y)
                edges = [
                    self._edge_index(x, y, 'h'),      # right
                    self._edge_index(x-1, y, 'h'),    # left
                    self._edge_index(x, y, 'v'),      # up
                    self._edge_index(x, y-1, 'v')     # down
                ]
                self.star_ops.append(sorted(set(edges)))

        # Plaquette operators (faces)
        for y in range(L):
            for x in range(L):
                # Edges around plaquette at (x, y)
                edges = [
                    self._edge_index(x, y, 'h'),      # bottom
                    self._edge_index(x, y+1, 'h'),    # top
                    self._edge_index(x, y, 'v'),      # left
                    self._edge_index(x+1, y, 'v')     # right
                ]
                self.plaq_ops.append(sorted(set(edges)))

    def H_Z(self) -> np.ndarray:
        """Z stabilizer parity check matrix (star operators)."""
        H = np.zeros((len(self.star_ops), self.n), dtype=int)
        for i, edges in enumerate(self.star_ops):
            for e in edges:
                H[i, e] = 1
        return H

    def H_X(self) -> np.ndarray:
        """X stabilizer parity check matrix (plaquette operators)."""
        H = np.zeros((len(self.plaq_ops), self.n), dtype=int)
        for i, edges in enumerate(self.plaq_ops):
            for e in edges:
                H[i, e] = 1
        return H

    def verify_css(self) -> bool:
        """Verify CSS commutation: H_Z · H_X^T = 0 mod 2."""
        H_Z = self.H_Z()
        H_X = self.H_X()
        product = (H_Z @ H_X.T) % 2
        return np.all(product == 0)

    def logical_Z1(self) -> np.ndarray:
        """Logical Z_1: horizontal cycle."""
        op = np.zeros(self.n, dtype=int)
        for x in range(self.L):
            op[self._edge_index(x, 0, 'h')] = 1
        return op

    def logical_X1(self) -> np.ndarray:
        """Logical X_1: vertical cycle."""
        op = np.zeros(self.n, dtype=int)
        for y in range(self.L):
            op[self._edge_index(0, y, 'v')] = 1
        return op

    def logical_Z2(self) -> np.ndarray:
        """Logical Z_2: vertical cycle."""
        op = np.zeros(self.n, dtype=int)
        for y in range(self.L):
            op[self._edge_index(0, y, 'v')] = 1
        return op

    def logical_X2(self) -> np.ndarray:
        """Logical X_2: horizontal cycle."""
        op = np.zeros(self.n, dtype=int)
        for x in range(self.L):
            op[self._edge_index(x, 0, 'h')] = 1
        return op

    def syndrome(self, x_error: np.ndarray, z_error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute syndrome for X and Z errors.

        Returns (star_syndrome, plaq_syndrome).
        """
        H_Z = self.H_Z()
        H_X = self.H_X()

        star_syn = (H_Z @ x_error) % 2  # X errors detected by Z stabs
        plaq_syn = (H_X @ z_error) % 2  # Z errors detected by X stabs

        return star_syn, plaq_syn

    def __repr__(self) -> str:
        return f"Toric Code [[{self.n}, {self.k}, {self.d}]]"


class PlanarCode:
    """
    Planar surface code with boundaries.

    Rough boundaries (top/bottom), smooth boundaries (left/right).
    """

    def __init__(self, d: int):
        """
        Initialize distance-d planar code.

        Parameters:
        -----------
        d : int
            Code distance
        """
        self.d = d
        # Rotated surface code: d² data qubits
        self.n = d * d
        self.k = 1

        self._build_stabilizers()

    def _build_stabilizers(self):
        """Build stabilizers for rotated surface code."""
        d = self.d
        self.x_stabs = []  # X stabilizers
        self.z_stabs = []  # Z stabilizers

        # In rotated code, stabilizers are on alternating squares
        # Pattern depends on parity

        # Simple model: checkerboard pattern
        for row in range(d - 1):
            for col in range(d - 1):
                # 2×2 plaquette at (row, col)
                qubits = [
                    row * d + col,
                    row * d + col + 1,
                    (row + 1) * d + col,
                    (row + 1) * d + col + 1
                ]

                if (row + col) % 2 == 0:
                    self.x_stabs.append(qubits)
                else:
                    self.z_stabs.append(qubits)

        # Boundary stabilizers (weight 2)
        # Top boundary
        for col in range(0, d - 1, 2):
            self.z_stabs.append([col, col + 1])

        # Bottom boundary
        for col in range(1, d - 1, 2):
            self.x_stabs.append([(d-1)*d + col, (d-1)*d + col + 1])

    def H_X(self) -> np.ndarray:
        """X stabilizer parity check."""
        H = np.zeros((len(self.x_stabs), self.n), dtype=int)
        for i, qubits in enumerate(self.x_stabs):
            for q in qubits:
                if 0 <= q < self.n:
                    H[i, q] = 1
        return H

    def H_Z(self) -> np.ndarray:
        """Z stabilizer parity check."""
        H = np.zeros((len(self.z_stabs), self.n), dtype=int)
        for i, qubits in enumerate(self.z_stabs):
            for q in qubits:
                if 0 <= q < self.n:
                    H[i, q] = 1
        return H

    def logical_Z(self) -> np.ndarray:
        """Logical Z: vertical chain."""
        op = np.zeros(self.n, dtype=int)
        for row in range(self.d):
            op[row * self.d] = 1  # Left column
        return op

    def logical_X(self) -> np.ndarray:
        """Logical X: horizontal chain."""
        op = np.zeros(self.n, dtype=int)
        for col in range(self.d):
            op[col] = 1  # Top row
        return op

    def __repr__(self) -> str:
        return f"Planar Code [[{self.n}, {self.k}, {self.d}]]"


def visualize_toric_syndrome(code: ToricCode, x_errors: List[int], z_errors: List[int]):
    """Visualize errors and syndromes on toric code."""
    L = code.L

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # X errors and star syndrome
    ax1 = axes[0]
    ax1.set_title("X Errors → Star (Z) Syndrome")

    # Draw lattice
    for i in range(L + 1):
        ax1.axhline(i, color='gray', linewidth=0.5)
        ax1.axvline(i, color='gray', linewidth=0.5)

    # Mark X errors (on horizontal edges)
    x_error_vec = np.zeros(code.n, dtype=int)
    for e in x_errors:
        x_error_vec[e] = 1
        if e < L * L:  # Horizontal edge
            row, col = e // L, e % L
            ax1.plot(col + 0.5, row, 'rx', markersize=15, markeredgewidth=3)
        else:  # Vertical edge
            idx = e - L * L
            row, col = idx // L, idx % L
            ax1.plot(col, row + 0.5, 'rx', markersize=15, markeredgewidth=3)

    # Mark syndrome (violated vertices)
    star_syn, _ = code.syndrome(x_error_vec, np.zeros(code.n, dtype=int))
    for v, s in enumerate(star_syn):
        if s == 1:
            row, col = v // L, v % L
            ax1.plot(col, row, 'bo', markersize=20, fillstyle='none', markeredgewidth=2)

    ax1.set_xlim(-0.5, L + 0.5)
    ax1.set_ylim(-0.5, L + 0.5)
    ax1.set_aspect('equal')

    # Z errors and plaquette syndrome
    ax2 = axes[1]
    ax2.set_title("Z Errors → Plaquette (X) Syndrome")

    for i in range(L + 1):
        ax2.axhline(i, color='gray', linewidth=0.5)
        ax2.axvline(i, color='gray', linewidth=0.5)

    z_error_vec = np.zeros(code.n, dtype=int)
    for e in z_errors:
        z_error_vec[e] = 1
        if e < L * L:
            row, col = e // L, e % L
            ax2.plot(col + 0.5, row, 'gx', markersize=15, markeredgewidth=3)
        else:
            idx = e - L * L
            row, col = idx // L, idx % L
            ax2.plot(col, row + 0.5, 'gx', markersize=15, markeredgewidth=3)

    _, plaq_syn = code.syndrome(np.zeros(code.n, dtype=int), z_error_vec)
    for p, s in enumerate(plaq_syn):
        if s == 1:
            row, col = p // L, p % L
            ax2.add_patch(plt.Rectangle((col, row), 1, 1, fill=False,
                                         edgecolor='red', linewidth=2))

    ax2.set_xlim(-0.5, L + 0.5)
    ax2.set_ylim(-0.5, L + 0.5)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('toric_syndrome.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: toric_syndrome.png")


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 746: Surface Codes as CSS")
    print("=" * 60)

    # Example 1: Toric Code
    print("\n1. Toric Code Analysis")
    print("-" * 40)

    for L in [2, 3, 4]:
        toric = ToricCode(L)
        print(f"L = {L}: {toric}")
        print(f"  CSS valid: {toric.verify_css()}")

        H_Z = toric.H_Z()
        H_X = toric.H_X()
        print(f"  Z stabilizers: {H_Z.shape[0]}, X stabilizers: {H_X.shape[0]}")

        # Check rank
        from numpy.linalg import matrix_rank
        rank_Z = matrix_rank(H_Z)
        rank_X = matrix_rank(H_X)
        print(f"  Rank H_Z: {rank_Z}, Rank H_X: {rank_X}")
        print(f"  n - rank_Z - rank_X = {toric.n - rank_Z - rank_X} (should be k=2)")

    # Example 2: Syndrome calculation
    print("\n2. Syndrome Example (L=3)")
    print("-" * 40)

    toric = ToricCode(3)

    # Single X error on edge 0
    x_error = np.zeros(toric.n, dtype=int)
    x_error[0] = 1
    star_syn, plaq_syn = toric.syndrome(x_error, np.zeros(toric.n, dtype=int))
    print(f"X error on edge 0:")
    print(f"  Star syndrome: {star_syn}")
    print(f"  Violated vertices: {np.where(star_syn == 1)[0]}")

    # Single Z error on edge 0
    z_error = np.zeros(toric.n, dtype=int)
    z_error[0] = 1
    star_syn, plaq_syn = toric.syndrome(np.zeros(toric.n, dtype=int), z_error)
    print(f"\nZ error on edge 0:")
    print(f"  Plaquette syndrome: {plaq_syn}")
    print(f"  Violated plaquettes: {np.where(plaq_syn == 1)[0]}")

    # Example 3: Logical operators
    print("\n3. Logical Operators (L=3)")
    print("-" * 40)

    Z1 = toric.logical_Z1()
    X1 = toric.logical_X1()

    print(f"Logical Z_1 (weight {np.sum(Z1)}): edges {np.where(Z1 == 1)[0]}")
    print(f"Logical X_1 (weight {np.sum(X1)}): edges {np.where(X1 == 1)[0]}")

    # Verify commutation with stabilizers
    H_Z = toric.H_Z()
    H_X = toric.H_X()

    syn_Z1 = (H_X @ Z1) % 2  # Z operator checked against X stabilizers
    syn_X1 = (H_Z @ X1) % 2  # X operator checked against Z stabilizers

    print(f"\nZ_1 syndrome (should be 0): {np.all(syn_Z1 == 0)}")
    print(f"X_1 syndrome (should be 0): {np.all(syn_X1 == 0)}")

    # Anticommutation
    overlap = np.sum(Z1 * X1) % 2
    print(f"Z_1 · X_1 overlap (should be 1): {overlap}")

    # Example 4: Planar Code
    print("\n4. Planar Code")
    print("-" * 40)

    for d in [3, 5, 7]:
        planar = PlanarCode(d)
        print(f"d = {d}: {planar}")

    # Example 5: Visualize (for L=3)
    print("\n5. Visualization")
    print("-" * 40)

    try:
        toric = ToricCode(3)
        visualize_toric_syndrome(toric, [0, 1], [9, 10])
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("\n" + "=" * 60)
    print("Surface codes: topological protection meets CSS structure!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Toric code parameters | $[[2L^2, 2, L]]$ |
| Star operator | $A_v = \prod_{e \ni v} Z_e$ |
| Plaquette operator | $B_p = \prod_{e \in \partial p} X_e$ |
| Planar code (approx) | $[[2d^2, 1, d]]$ |
| Rotated surface code | $[[d^2, 1, d]]$ |

### Main Takeaways

1. **Surface codes are CSS** with X stabilizers on faces and Z stabilizers on vertices
2. **Toric code** on L×L torus encodes 2 logical qubits with distance L
3. **Planar code** with boundaries encodes 1 logical qubit
4. **Logical operators** are topological cycles wrapping around the surface
5. **Errors create defect pairs** that can be matched for correction
6. **Homology** determines the number of logical qubits

---

## Daily Checklist

- [ ] I can construct stabilizers from a lattice
- [ ] I understand why surface codes are CSS
- [ ] I can compute toric code parameters
- [ ] I can identify logical operators as cycles
- [ ] I understand the role of boundaries in planar codes
- [ ] I can compute syndromes for localized errors

---

## Preview: Day 747

Tomorrow we explore **hypergraph product codes**:

- Product construction from classical codes
- Parameter analysis: [[n₁n₂ + m₁m₂, k₁k₂, min(d₁, d₂)]]
- qLDPC codes with constant rate
- Going beyond surface codes

Hypergraph products systematically generate quantum codes from classical codes!
