# Day 699: Surface Codes Introduction

## Overview

**Week:** 100 (QEC Conditions)
**Day:** Saturday
**Date:** Year 2, Month 25, Day 699
**Topic:** Toric and Surface Codes — Topological Quantum Error Correction
**Hours:** 7 (3.5 theory + 2.5 problems + 1 computational lab)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Toric code, surface code geometry |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Anyons, syndrome decoding |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Surface code simulation |

---

## Prerequisites

From Days 694-698:
- Stabilizer formalism
- CSS code structure
- Threshold theorem

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Describe** the toric code and its topological properties
2. **Construct** the planar surface code from the toric code
3. **Identify** X and Z stabilizers on a surface code lattice
4. **Explain** error correction via anyon pair annihilation
5. **Apply** minimum-weight perfect matching for decoding
6. **Understand** why surface codes are practically dominant

---

## Core Content

### 1. The Toric Code: Origins

#### Kitaev's Vision (1997-2003)

Alexei Kitaev introduced the **toric code** — a topological quantum code with remarkable properties:

- Qubits on edges of a square lattice
- Stabilizers defined by local geometry
- Logical information encoded in topology
- Errors appear as anyonic excitations

#### Toric Code Geometry

Consider a square lattice on a **torus** (periodic boundaries):

```
    ○───○───○───○───
    │   │   │   │
    ○───○───○───○───
    │   │   │   │
    ○───○───○───○───
    │   │   │   │
    ○───○───○───○───
```

- **Vertices (○):** Define Z-stabilizers
- **Plaquettes (□):** Define X-stabilizers
- **Edges (─/│):** Host qubits

#### Toric Code Parameters

For an $L \times L$ lattice on a torus:

$$\boxed{[[2L^2, 2, L]]}$$

- **n = 2L²:** Qubits (two per unit cell, on edges)
- **k = 2:** Logical qubits (from torus topology)
- **d = L:** Distance scales with lattice size

---

### 2. Stabilizer Structure

#### Vertex (Z-type) Stabilizers

At each vertex $v$, define:

$$\boxed{A_v = \prod_{e \ni v} Z_e}$$

Product of Z on all edges meeting at vertex.

```
        Z
        │
    Z───○───Z
        │
        Z
```

#### Plaquette (X-type) Stabilizers

For each plaquette $p$, define:

$$\boxed{B_p = \prod_{e \in \partial p} X_e}$$

Product of X on all edges bordering plaquette.

```
    ┌─X─┐
    X   X
    └─X─┘
```

#### Commutation

All stabilizers commute because:
- Any edge borders exactly 2 plaquettes and 2 vertices
- Each $A_v$ and $B_p$ share 0 or 2 edges
- Even overlap → commutation

$$[A_v, B_p] = 0 \quad \forall v, p$$

---

### 3. Logical Operators

#### Non-Contractible Loops

Logical operators correspond to **non-contractible loops** on the torus:

**Logical Z:** Z operators along a vertical loop (wrapping the torus)
$$\bar{Z}_1 = \prod_{e \in \gamma_v} Z_e$$

**Logical X:** X operators along a horizontal loop
$$\bar{X}_1 = \prod_{e \in \gamma_h} X_e$$

The second logical qubit comes from loops in the orthogonal direction.

#### Topological Protection

- Logical operators must span the entire system
- Weight of logical operator = L (lattice size)
- Local errors cannot affect logical information

**This is topological protection!**

---

### 4. The Surface Code: Planar Version

#### From Torus to Plane

The **surface code** is the planar version:
- Open boundaries instead of periodic
- Encodes 1 logical qubit (not 2)
- More practical for 2D qubit arrays

#### Boundary Conditions

```
    ─X─X─X─X─     (Rough boundary: X-type)
    │ ○ │ ○ │
    ─○─○─○─○─
    │ ○ │ ○ │
    ─X─X─X─X─     (Rough boundary: X-type)
    Z   Z   Z     (Smooth boundary: Z-type)
```

- **Rough boundaries:** Terminate Z-stabilizers, X strings can end
- **Smooth boundaries:** Terminate X-stabilizers, Z strings can end

#### Surface Code Parameters

For $d \times d$ lattice:

$$\boxed{[[d^2 + (d-1)^2, 1, d]] \approx [[2d^2, 1, d]]}$$

- Roughly 2d² qubits for distance d
- Single logical qubit
- Distance = lattice linear dimension

---

### 5. Errors as Anyons

#### Anyon Creation

**X error** on an edge:
- Anticommutes with 2 adjacent Z-stabilizers
- Creates pair of Z-syndrome defects ("e-anyons")

**Z error** on an edge:
- Anticommutes with 2 adjacent X-stabilizers
- Creates pair of X-syndrome defects ("m-anyons")

```
    X error on edge:
    ○───○       ●───○
    │   │   →   │ X │
    ○───○       ●───○
                ↑   ↑
             e-anyon pair
```

#### Error Chains

Multiple errors form **chains**:
- Anyons appear at chain endpoints
- Interior of chain: no syndrome (errors "cancel")
- Closed loops: no syndrome at all (stabilizer or logical)

---

### 6. Decoding: Minimum-Weight Perfect Matching

#### The Decoding Problem

Given syndrome (anyon locations):
1. Find likely error chains connecting anyon pairs
2. Apply correction to annihilate anyons
3. Hope the correction + error is trivial (stabilizer)

#### MWPM Algorithm

**Minimum-Weight Perfect Matching:**

1. Build graph with anyon locations as vertices
2. Edge weights = distance between anyons
3. Find minimum-weight matching (pairs all anyons)
4. Correction = edges of the matching

#### Why It Works

- Most likely error has minimum total weight
- MWPM finds minimum-weight pairing
- Polynomial time: $O(n^3)$ or better

#### Degeneracy Advantage

Many error chains give same syndrome → degeneracy!
- Any matching that pairs anyons correctly works
- Don't need to find exact error, just equivalence class

---

### 7. Why Surface Codes Dominate

#### Practical Advantages

1. **High threshold (~1%):** Much better than concatenated codes
2. **Local stabilizers:** Only 4-body measurements
3. **2D geometry:** Natural for superconducting, trapped-ion, neutral atom platforms
4. **Efficient decoding:** MWPM is polynomial time
5. **Degenerate:** Tolerates correlated errors

#### Comparison with Other Codes

| Property | Surface | Steane | Shor | [[5,1,3]] |
|----------|---------|--------|------|-----------|
| Threshold | ~1% | ~0.01% | ~0.01% | ~0.01% |
| Locality | 4-body | 4-body | 2-body | 4-body |
| Scalable $d$ | Yes | Fixed | Fixed | Fixed |
| Overhead | ~1000:1 | 7:1 | 9:1 | 5:1 |

Surface codes win on threshold; fixed codes win on overhead for low $d$.

---

## Quantum Mechanics Connection

### Topological Order

The toric/surface code exhibits **topological order**:

- Ground state degeneracy depends on topology
- Excitations are anyons (neither bosons nor fermions)
- Logical operations = braiding anyons

### Anyonic Statistics

In 2D, particle exchange can give phases other than ±1:

$$|\psi_1 \psi_2\rangle \xrightarrow{\text{exchange}} e^{i\theta} |\psi_2 \psi_1\rangle$$

For toric code anyons, $\theta = \pi$ (fermion-like), but mutual statistics are non-trivial.

---

## Worked Examples

### Example 1: Stabilizer Counting

**Problem:** For a 3×3 surface code, count qubits and stabilizers.

**Solution:**

Lattice structure:
- Data qubits on edges: $(3 \times 2) + (2 \times 3) = 6 + 6 = 12$ qubits?

Wait, let's be more careful. For a distance-3 surface code:

Standard convention: $d \times d$ array of data qubits with stabilizers between them.

For $d = 3$:
- Data qubits: approximately $2 \times 3^2 - 1 = 17$ (varies by convention)
- X-stabilizers: $(d-1) \times d / 2$ roughly
- Z-stabilizers: similar

Exact count depends on boundary conditions. For rotated surface code:
- $d^2 = 9$ data qubits
- $(d^2 - 1)/2 = 4$ X-stabilizers
- $(d^2 - 1)/2 = 4$ Z-stabilizers
- $k = 9 - 8 = 1$ logical qubit ✓

---

### Example 2: Syndrome Pattern

**Problem:** An X error occurs on edge $e$. Describe the syndrome.

**Solution:**

An X error anticommutes with Z-stabilizers at the two vertices sharing edge $e$.

If $e$ connects vertices $v_1$ and $v_2$:
- $A_{v_1}$ measurement: -1 (error detected)
- $A_{v_2}$ measurement: -1 (error detected)
- All other $A_v$: +1

**Syndrome:** Two adjacent vertex defects (anyon pair).

To correct: Apply X on any path connecting $v_1$ to $v_2$ (or to boundaries in surface code).

---

### Example 3: Logical Error

**Problem:** When does an error chain cause a logical error?

**Solution:**

An error chain causes a logical error if:
1. The chain + correction forms a **non-trivial cycle**
2. The cycle wraps around the torus (toric code) or connects opposite boundary types (surface code)

For surface code:
- Z-chain connecting rough boundaries = logical X error
- X-chain connecting smooth boundaries = logical Z error

Minimum such chain has weight $d$ = code distance.

---

## Practice Problems

### Level 1: Direct Application

1. **Stabilizer Structure:**
   Draw a 2×2 portion of the surface code lattice and label X and Z stabilizers.

2. **Anyon Pairs:**
   An Z error occurs. What type of anyons are created and where?

3. **Distance:**
   For an $L = 5$ surface code, what is the code distance? How many qubits approximately?

### Level 2: Intermediate

4. **Logical Operators:**
   On a distance-5 surface code, draw a weight-5 logical X operator and a weight-5 logical Z operator.

5. **MWPM Example:**
   Given 4 anyon defects at positions (0,0), (2,0), (1,2), (3,2), find the MWPM and resulting correction.

6. **Threshold Calculation:**
   If physical error rate is 0.5% and threshold is 1%, estimate the logical error rate for a distance-7 code.

### Level 3: Challenging

7. **Boundary Effects:**
   Explain how rough and smooth boundaries differ in their effect on X and Z stabilizers.

8. **Homology:**
   Relate surface code logical operators to homology classes of the underlying surface.

9. **Lattice Surgery:**
   How can two surface codes be merged to perform a logical CNOT? Sketch the procedure.

---

## Computational Lab

### Surface Code Simulation

```python
"""
Day 699 Computational Lab: Surface Code Simulation
Basic surface code structure and syndrome analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Dict
from collections import defaultdict

class SurfaceCode:
    """Simple surface code implementation."""

    def __init__(self, distance: int):
        """
        Initialize surface code.

        Args:
            distance: Code distance (odd number recommended)
        """
        self.d = distance
        self.n_data = distance * distance
        self.n_x_stab = (distance - 1) * distance // 2 + (distance - 1) * (distance - 1) // 2
        self.n_z_stab = self.n_x_stab

        # Create lattice
        self._create_lattice()

    def _create_lattice(self):
        """Create the surface code lattice structure."""
        d = self.d

        # Data qubit positions (on a grid)
        self.data_qubits = []
        for i in range(d):
            for j in range(d):
                self.data_qubits.append((i, j))

        # X-stabilizers (plaquettes)
        self.x_stabilizers = []
        for i in range(d - 1):
            for j in range(d - 1):
                if (i + j) % 2 == 0:  # Checkerboard pattern
                    # Plaquette at (i+0.5, j+0.5)
                    qubits = [
                        (i, j), (i+1, j), (i, j+1), (i+1, j+1)
                    ]
                    qubits = [q for q in qubits if 0 <= q[0] < d and 0 <= q[1] < d]
                    if qubits:
                        self.x_stabilizers.append(qubits)

        # Z-stabilizers (vertices)
        self.z_stabilizers = []
        for i in range(d - 1):
            for j in range(d - 1):
                if (i + j) % 2 == 1:  # Opposite checkerboard
                    qubits = [
                        (i, j), (i+1, j), (i, j+1), (i+1, j+1)
                    ]
                    qubits = [q for q in qubits if 0 <= q[0] < d and 0 <= q[1] < d]
                    if qubits:
                        self.z_stabilizers.append(qubits)

        # Map qubit position to index
        self.qubit_to_idx = {q: i for i, q in enumerate(self.data_qubits)}

    def apply_error(self, error_type: str, qubit_pos: Tuple[int, int]) -> np.ndarray:
        """
        Apply an error and return syndrome.

        Args:
            error_type: 'X', 'Y', or 'Z'
            qubit_pos: (i, j) position of error

        Returns:
            Syndrome array
        """
        x_syndrome = []
        z_syndrome = []

        if error_type in ['X', 'Y']:
            # X error detected by Z-stabilizers
            for stab in self.z_stabilizers:
                if qubit_pos in stab:
                    z_syndrome.append(1)
                else:
                    z_syndrome.append(0)
        else:
            z_syndrome = [0] * len(self.z_stabilizers)

        if error_type in ['Z', 'Y']:
            # Z error detected by X-stabilizers
            for stab in self.x_stabilizers:
                if qubit_pos in stab:
                    x_syndrome.append(1)
                else:
                    x_syndrome.append(0)
        else:
            x_syndrome = [0] * len(self.x_stabilizers)

        return np.array(x_syndrome + z_syndrome)

    def get_syndrome_defects(self, syndrome: np.ndarray) -> Tuple[List, List]:
        """Get positions of syndrome defects."""
        n_x = len(self.x_stabilizers)

        x_defects = []
        z_defects = []

        for i, syn in enumerate(syndrome[:n_x]):
            if syn == 1:
                # X-stabilizer defect (from Z error)
                stab = self.x_stabilizers[i]
                center = (np.mean([q[0] for q in stab]),
                          np.mean([q[1] for q in stab]))
                x_defects.append(center)

        for i, syn in enumerate(syndrome[n_x:]):
            if syn == 1:
                # Z-stabilizer defect (from X error)
                stab = self.z_stabilizers[i]
                center = (np.mean([q[0] for q in stab]),
                          np.mean([q[1] for q in stab]))
                z_defects.append(center)

        return x_defects, z_defects


def visualize_surface_code(code: SurfaceCode, error_pos: Tuple[int, int] = None,
                          error_type: str = None):
    """Visualize surface code with optional error."""

    fig, ax = plt.subplots(figsize=(10, 10))

    d = code.d

    # Draw data qubits
    for q in code.data_qubits:
        ax.plot(q[1], q[0], 'ko', markersize=15)

    # Draw X-stabilizers (plaquettes)
    for stab in code.x_stabilizers:
        xs = [q[1] for q in stab]
        ys = [q[0] for q in stab]
        center_x = np.mean(xs)
        center_y = np.mean(ys)
        rect = plt.Rectangle((center_x - 0.4, center_y - 0.4), 0.8, 0.8,
                              fill=True, facecolor='lightblue', edgecolor='blue',
                              alpha=0.5)
        ax.add_patch(rect)
        ax.text(center_x, center_y, 'X', ha='center', va='center', fontsize=8)

    # Draw Z-stabilizers
    for stab in code.z_stabilizers:
        xs = [q[1] for q in stab]
        ys = [q[0] for q in stab]
        center_x = np.mean(xs)
        center_y = np.mean(ys)
        rect = plt.Rectangle((center_x - 0.4, center_y - 0.4), 0.8, 0.8,
                              fill=True, facecolor='lightgreen', edgecolor='green',
                              alpha=0.5)
        ax.add_patch(rect)
        ax.text(center_x, center_y, 'Z', ha='center', va='center', fontsize=8)

    # Highlight error
    if error_pos is not None:
        ax.plot(error_pos[1], error_pos[0], 'rx', markersize=20, markeredgewidth=3)
        ax.annotate(f'{error_type} error', (error_pos[1] + 0.3, error_pos[0] + 0.3),
                    fontsize=12, color='red')

        # Show syndrome
        syndrome = code.apply_error(error_type, error_pos)
        x_defects, z_defects = code.get_syndrome_defects(syndrome)

        for defect in x_defects:
            ax.plot(defect[1], defect[0], 'b*', markersize=25)
        for defect in z_defects:
            ax.plot(defect[1], defect[0], 'g*', markersize=25)

    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(-0.5, d - 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Surface Code (d = {d})', fontsize=14)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('surface_code_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fig


def demonstrate_surface_code():
    """Demonstrate surface code properties."""

    print("=" * 60)
    print("SURFACE CODE DEMONSTRATION")
    print("=" * 60)

    # Create distance-5 code
    code = SurfaceCode(5)

    print(f"\nDistance-{code.d} Surface Code:")
    print(f"  Data qubits: {len(code.data_qubits)}")
    print(f"  X-stabilizers: {len(code.x_stabilizers)}")
    print(f"  Z-stabilizers: {len(code.z_stabilizers)}")
    print(f"  Logical qubits: 1")

    # Test single errors
    print("\n" + "-" * 40)
    print("SINGLE ERROR SYNDROMES:")
    print("-" * 40)

    test_pos = (2, 2)  # Center qubit

    for error_type in ['X', 'Y', 'Z']:
        syndrome = code.apply_error(error_type, test_pos)
        n_defects = np.sum(syndrome)
        print(f"\n{error_type} error at {test_pos}:")
        print(f"  Total defects: {n_defects}")
        print(f"  Syndrome: {syndrome}")


def analyze_threshold_scaling():
    """Analyze how error rate scales with distance."""

    print("\n" + "=" * 60)
    print("SURFACE CODE SCALING ANALYSIS")
    print("=" * 60)

    p_physical = 0.003  # 0.3% physical error rate
    p_threshold = 0.01  # 1% threshold

    distances = [3, 5, 7, 9, 11, 13, 15]

    print(f"\nPhysical error rate: {p_physical*100:.2f}%")
    print(f"Threshold: {p_threshold*100:.1f}%")
    print("\n" + "-" * 50)
    print(f"{'Distance':>10} {'Qubits':>10} {'p_logical':>15}")
    print("-" * 50)

    for d in distances:
        n_qubits = 2 * d * d  # Approximate
        # Approximate formula: p_L ≈ A * (p/p_th)^((d+1)/2)
        p_logical = 0.1 * (p_physical / p_threshold) ** ((d + 1) / 2)
        print(f"{d:>10} {n_qubits:>10} {p_logical:>15.2e}")


if __name__ == "__main__":
    demonstrate_surface_code()
    analyze_threshold_scaling()

    print("\n" + "=" * 60)
    print("Generating visualization...")
    print("=" * 60)

    code = SurfaceCode(5)
    visualize_surface_code(code, error_pos=(2, 2), error_type='X')
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Toric code | $[[2L^2, 2, L]]$ |
| Surface code | $[[2d^2, 1, d]]$ (approximate) |
| Vertex stabilizer | $A_v = \prod_{e \ni v} Z_e$ |
| Plaquette stabilizer | $B_p = \prod_{e \in \partial p} X_e$ |
| Threshold | $p_{th} \approx 1\%$ |

### Main Takeaways

1. **Topological origin:** Toric code encodes information in topology
2. **Local stabilizers:** Only 4-body measurements needed
3. **Anyonic errors:** Errors create anyon pairs at chain endpoints
4. **MWPM decoding:** Polynomial-time optimal decoding
5. **High threshold:** ~1% enables practical quantum computing
6. **Scalable distance:** Code distance grows with lattice size

---

## Daily Checklist

- [ ] Understand toric code on a torus
- [ ] Know surface code lattice structure
- [ ] Can identify X and Z stabilizers
- [ ] Understand anyon pair creation
- [ ] Know MWPM decoding concept
- [ ] Understand why surface codes dominate

---

## Preview: Day 700

Tomorrow is the **Month 25 Synthesis** — our capstone day bringing together all QEC fundamentals:

- Complete QEC framework review
- Code comparison and selection
- Research frontiers and open problems
- Preparation for advanced topics (Month 26+)

---

*"The surface code is to quantum computing what the transistor is to classical computing — the enabling technology."*
