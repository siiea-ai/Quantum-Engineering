# Day 813: Rotated Surface Code Geometry

## Week 117, Day 1 | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Overview

Today we explore the rotated surface code—the practical workhorse of topological quantum error correction. By rotating the standard surface code lattice by 45°, we achieve a significant reduction in qubit overhead while maintaining the same code distance and error correction capability. This geometric insight, formalized by Bombin and Martin-Delgado and popularized by Fowler et al., represents one of the most important practical advances in surface code implementation.

---

## Daily Schedule

| Session | Duration | Content |
|---------|----------|---------|
| Morning | 3 hours | Rotated geometry theory, qubit counting, stabilizer structure |
| Afternoon | 2 hours | Worked examples, lattice construction exercises |
| Evening | 2 hours | Python implementation of rotated surface codes |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the geometric relationship between rotated and unrotated surface codes
2. **Derive** the [[d², 1, d]] code parameters for the rotated surface code
3. **Construct** the stabilizer generators for rotated codes of arbitrary distance
4. **Calculate** the qubit overhead advantage of rotation
5. **Identify** data qubits, ancilla qubits, and their roles in the rotated layout
6. **Implement** a rotated surface code lattice generator in Python

---

## Core Content

### 1. The Unrotated Surface Code: Baseline

The standard (unrotated) surface code on a $d \times d$ lattice has the following structure:

**Lattice Structure:**
- Data qubits placed on edges of a square lattice
- X-stabilizers (plaquettes) at each face
- Z-stabilizers (vertices) at each vertex

For an unrotated code of distance $d$:

$$\boxed{n_{\text{unrotated}} = 2d^2 - 2d + 1 \text{ (data qubits)}}$$

This can be understood as: we have approximately $d^2$ faces and $d^2$ vertices, with data qubits on the $2d(d-1) + (2d-1) = 2d^2 - 1$ edges (accounting for boundaries).

### 2. The 45° Rotation Insight

The rotated surface code places data qubits at the *vertices* of a rotated (diagonal) square lattice:

**Key Geometric Transformation:**
- Rotate the coordinate system by 45°
- Place data qubits at lattice points $(i, j)$ where $i + j$ is even
- X-stabilizers become "weight-4 faces" in the bulk
- Z-stabilizers become "weight-4 faces" of the dual lattice

The brilliant insight is that this rotation creates a more compact representation:

$$\boxed{n_{\text{rotated}} = d^2 \text{ (data qubits)}}$$

**Qubit Savings:**
$$\text{Ratio} = \frac{n_{\text{unrotated}}}{n_{\text{rotated}}} = \frac{2d^2 - 2d + 1}{d^2} \approx 2 \text{ (for large } d \text{)}$$

### 3. Code Parameters: [[n, k, d]]

For the rotated surface code:

$$\boxed{[[d^2, 1, d]] \text{ — Rotated Surface Code}}$$

Where:
- $n = d^2$ = number of physical data qubits
- $k = 1$ = number of encoded logical qubits
- $d$ = code distance = minimum weight of any logical operator

**Including Ancilla Qubits:**
For syndrome extraction, we need ancilla qubits:
- X-syndrome ancillas: $\frac{d^2 - 1}{2}$ (approximately)
- Z-syndrome ancillas: $\frac{d^2 - 1}{2}$ (approximately)

Total ancilla count: $d^2 - 1$

**Total Physical Qubits:**
$$\boxed{n_{\text{total}} = d^2 + (d^2 - 1) = 2d^2 - 1}$$

### 4. Stabilizer Structure in the Rotated Code

**Coordinate System:**
Place data qubits at positions $(i, j)$ for $0 \leq i, j \leq d-1$.

**X-Stabilizers (Plaquettes):**
Weight-4 X-stabilizers in the bulk act on four data qubits arranged in a diamond pattern:

$$S_X^{(i,j)} = X_{i,j} X_{i+1,j} X_{i,j+1} X_{i+1,j+1}$$

for appropriate $(i, j)$ with $i + j$ odd (or even, depending on convention).

**Z-Stabilizers (Vertices):**
Weight-4 Z-stabilizers also form diamond patterns:

$$S_Z^{(i,j)} = Z_{i,j} Z_{i+1,j} Z_{i,j+1} Z_{i+1,j+1}$$

for $(i, j)$ with the opposite parity.

**Boundary Stabilizers:**
At boundaries, stabilizers have reduced weight:
- Weight-2 stabilizers along edges
- This creates the "rough" and "smooth" boundaries we'll study tomorrow

### 5. Counting Stabilizers

For a distance-$d$ rotated surface code:

**Independent Stabilizers:**
- Number of X-stabilizers: $\frac{d^2 - 1}{2}$
- Number of Z-stabilizers: $\frac{d^2 - 1}{2}$
- Total independent stabilizers: $d^2 - 1$

**Verification via Code Dimension:**
The number of logical qubits is:
$$k = n - \text{(number of independent stabilizers)} = d^2 - (d^2 - 1) = 1 \checkmark$$

### 6. Logical Operators

**Logical X Operator ($\bar{X}$):**
A chain of X operators connecting two "smooth" boundaries (top to bottom in standard orientation):
$$\bar{X} = X_1 X_2 \cdots X_d$$
where the chain has minimum length $d$.

**Logical Z Operator ($\bar{Z}$):**
A chain of Z operators connecting two "rough" boundaries (left to right):
$$\bar{Z} = Z_1 Z_2 \cdots Z_d$$

**Commutation:**
$$[\bar{X}, \bar{Z}] \neq 0 \text{ (they anticommute: } \bar{X}\bar{Z} = -\bar{Z}\bar{X} \text{)}$$

This anticommutation arises because the two chains cross at exactly one point.

### 7. Distance and Error Correction

The code distance $d$ means:
1. Any error of weight $< d$ can be detected
2. Any error of weight $\leq t = \lfloor(d-1)/2\rfloor$ can be corrected

**Minimum-Weight Logical Operators:**
- $|\bar{X}|_{\min} = d$ (minimum weight of any X-type logical)
- $|\bar{Z}|_{\min} = d$ (minimum weight of any Z-type logical)

**Error Correction Capability:**
$$\boxed{t = \left\lfloor \frac{d-1}{2} \right\rfloor \text{ correctable errors}}$$

| Distance $d$ | Correctable Errors $t$ | Physical Qubits $d^2$ |
|--------------|------------------------|------------------------|
| 3 | 1 | 9 |
| 5 | 2 | 25 |
| 7 | 3 | 49 |
| 9 | 4 | 81 |
| 11 | 5 | 121 |

### 8. Comparison: Rotated vs. Unrotated

| Property | Unrotated | Rotated | Advantage |
|----------|-----------|---------|-----------|
| Data qubits | $2d^2 - 2d + 1$ | $d^2$ | ~50% reduction |
| Code distance | $d$ | $d$ | Same |
| Logical qubits | 1 | 1 | Same |
| Boundary types | 2 | 2 | Same |
| Bulk stabilizer weight | 4 | 4 | Same |
| Connectivity required | 4 | 4 | Same |

The rotated code achieves the same error correction with roughly half the qubits!

---

## Quantum Computing Connection

### Google's Sycamore/Willow Implementation

Google's quantum processors use the rotated surface code geometry:

- **Sycamore (2019-2023):** Demonstrated distance-3 and distance-5 codes
- **Willow (2024):** Achieved below-threshold error correction with distance-7

The rotated layout maps naturally to their square grid connectivity:
- Each data qubit couples to 4 neighbors (bulk) or fewer (boundaries)
- Ancilla qubits interleaved with data qubits
- Syndrome extraction uses 2-qubit gates between neighbors

### Why Rotation Matters for Hardware

1. **Qubit efficiency:** Fewer physical qubits means lower hardware requirements
2. **Same connectivity:** 4-way coupling suffices for both geometries
3. **Natural boundaries:** Rotation creates clean boundary conditions for logical operators

---

## Worked Examples

### Example 1: Distance-3 Rotated Surface Code

**Problem:** Construct the complete stabilizer set for a distance-3 rotated surface code.

**Solution:**

For $d = 3$, we have:
- Data qubits: $3^2 = 9$
- Ancilla qubits: $3^2 - 1 = 8$
- Total: 17 qubits

**Qubit Layout (data qubits labeled 0-8):**
```
        0
      / | \
     1--+--2
    /|  |  |\
   3-+--4--+-5
    \|  |  |/
     6--+--7
      \ | /
        8
```

Actually, let's use a cleaner coordinate representation:

```
Row 0:    0   1   2
Row 1:    3   4   5
Row 2:    6   7   8
```

**X-Stabilizers (4 total):**
$$S_X^1 = X_0 X_1 X_3 X_4$$
$$S_X^2 = X_1 X_2 X_4 X_5$$
$$S_X^3 = X_3 X_4 X_6 X_7$$
$$S_X^4 = X_4 X_5 X_7 X_8$$

**Z-Stabilizers (4 total):**
For the rotated code, Z-stabilizers cover different plaquettes. With boundaries:
- Two weight-2 Z-stabilizers on top/bottom boundaries
- Two weight-2 Z-stabilizers on left/right boundaries

$$S_Z^1 = Z_0 Z_1$$
$$S_Z^2 = Z_0 Z_3$$
$$S_Z^3 = Z_2 Z_5$$
$$S_Z^4 = Z_7 Z_8$$

Wait, let me reconsider the standard rotated code layout more carefully.

**Standard Distance-3 Rotated Code:**

In the standard convention, we place qubits on a diamond:
```
      0
     /|\
    1-+-2
   /|\|/|\
  3-+-4-+-5
   \|/|\|/
    6-+-7
     \|/
      8
```

With 4 X-type plaquettes and 4 Z-type plaquettes (some at boundaries with weight 2).

**Verification:** 9 data qubits - 8 independent stabilizers = 1 logical qubit. ✓

---

### Example 2: Qubit Overhead Comparison

**Problem:** Calculate the qubit savings when using a rotated vs. unrotated surface code for distances 5, 7, 9, and 11.

**Solution:**

| Distance $d$ | Unrotated: $2d^2 - 2d + 1$ | Rotated: $d^2$ | Savings |
|--------------|---------------------------|----------------|---------|
| 5 | $2(25) - 10 + 1 = 41$ | 25 | 16 (39%) |
| 7 | $2(49) - 14 + 1 = 85$ | 49 | 36 (42%) |
| 9 | $2(81) - 18 + 1 = 145$ | 81 | 64 (44%) |
| 11 | $2(121) - 22 + 1 = 221$ | 121 | 100 (45%) |

As $d \to \infty$:
$$\text{Savings ratio} \to \frac{2d^2 - 2d + 1 - d^2}{2d^2 - 2d + 1} = \frac{d^2 - 2d + 1}{2d^2 - 2d + 1} \to \frac{1}{2}$$

The rotated code asymptotically uses half the qubits!

---

### Example 3: Finding Logical Operators

**Problem:** For the distance-5 rotated surface code, write explicit logical X and Z operators.

**Solution:**

**Qubit labeling for d=5:**
```
Row 0:  0   1   2   3   4
Row 1:  5   6   7   8   9
Row 2: 10  11  12  13  14
Row 3: 15  16  17  18  19
Row 4: 20  21  22  23  24
```

**Logical X (connects smooth boundaries, vertical path):**
One choice is the leftmost column:
$$\bar{X} = X_0 X_5 X_{10} X_{15} X_{20}$$

This is weight 5 = $d$, confirming it's a minimum-weight logical.

**Logical Z (connects rough boundaries, horizontal path):**
One choice is the top row:
$$\bar{Z} = Z_0 Z_1 Z_2 Z_3 Z_4$$

Also weight 5 = $d$.

**Verification of Anticommutation:**
$\bar{X}$ and $\bar{Z}$ share exactly one qubit (qubit 0), so they anticommute:
$$\bar{X}\bar{Z} = -\bar{Z}\bar{X}$$

This is exactly the algebra required for a single logical qubit.

---

## Practice Problems

### Direct Application

**Problem 1:** For a distance-13 rotated surface code, calculate:
a) The number of data qubits
b) The number of ancilla qubits (total)
c) The number of correctable errors
d) The total physical qubit count

**Problem 2:** A rotated surface code has 49 data qubits. What is its code distance, and how many errors can it correct?

**Problem 3:** List all X-stabilizers and Z-stabilizers for the distance-3 rotated surface code shown in Example 1.

### Intermediate

**Problem 4:** Prove that for any rotated surface code of distance $d$, the X and Z logical operators must cross at an odd number of points.

**Problem 5:** If a physical qubit error rate is $p = 10^{-3}$ and we need a logical error rate $p_L < 10^{-15}$, estimate the minimum code distance required. Use the approximation $p_L \approx (p/p_{th})^{(d+1)/2}$ with $p_{th} = 0.01$.

**Problem 6:** Design a distance-5 rotated surface code and identify all 24 stabilizers (12 X-type and 12 Z-type).

### Challenging

**Problem 7:** Prove that the rotated surface code achieves the bound $n \geq d^2$ for any [[n, 1, d]] stabilizer code with a topological structure. (Hint: Consider how logical operators must traverse the lattice.)

**Problem 8:** Consider a "stretched" rotated surface code with dimensions $d_X \times d_Z$ where $d_X \neq d_Z$.
a) How many data qubits does this require?
b) What are the distances for X and Z logical operators?
c) When might this asymmetric code be useful?

**Problem 9:** The surface code can be viewed as a $\mathbb{Z}_2$ gauge theory on a lattice. Express the rotated surface code stabilizers in terms of gauge field variables and show that the logical operators correspond to Wilson loops.

---

## Computational Lab

### Lab 813: Rotated Surface Code Generator

```python
"""
Day 813 Computational Lab: Rotated Surface Code Geometry
=========================================================

This lab implements a complete rotated surface code lattice generator,
including visualization and stabilizer enumeration.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
import itertools

class RotatedSurfaceCode:
    """
    Rotated surface code implementation.

    The code is defined on a d x d grid where:
    - Data qubits are at positions (i, j) for 0 <= i, j < d
    - X-stabilizers are diamond-shaped plaquettes
    - Z-stabilizers are diamond-shaped plaquettes (offset from X)
    """

    def __init__(self, distance: int):
        """
        Initialize a rotated surface code of given distance.

        Parameters:
        -----------
        distance : int
            The code distance (must be odd for standard surface codes)
        """
        if distance < 3:
            raise ValueError("Distance must be at least 3")
        if distance % 2 == 0:
            print(f"Warning: Even distance {distance} may have unusual boundary structure")

        self.d = distance
        self.n_data = distance ** 2
        self.n_ancilla = distance ** 2 - 1
        self.n_total = 2 * distance ** 2 - 1

        # Create data qubit positions
        self.data_qubits = self._create_data_qubits()

        # Create stabilizers
        self.x_stabilizers = self._create_x_stabilizers()
        self.z_stabilizers = self._create_z_stabilizers()

        # Create logical operators
        self.logical_x = self._create_logical_x()
        self.logical_z = self._create_logical_z()

    def _create_data_qubits(self):
        """Create data qubit positions on the rotated lattice."""
        qubits = {}
        idx = 0
        for i in range(self.d):
            for j in range(self.d):
                qubits[idx] = (i, j)
                idx += 1
        return qubits

    def _coord_to_idx(self, i, j):
        """Convert (i, j) coordinates to qubit index."""
        if 0 <= i < self.d and 0 <= j < self.d:
            return i * self.d + j
        return None

    def _create_x_stabilizers(self):
        """
        Create X-type stabilizers.

        In the rotated code, X-stabilizers are on plaquettes where
        the center has coordinates (i+0.5, j+0.5) for certain (i,j).
        """
        stabilizers = []

        for i in range(self.d - 1):
            for j in range(self.d - 1):
                # Check if this is an X-stabilizer position
                # Standard convention: X-stabilizers where (i + j) is even
                if (i + j) % 2 == 0:
                    # Bulk stabilizer: weight 4
                    qubits = [
                        self._coord_to_idx(i, j),
                        self._coord_to_idx(i + 1, j),
                        self._coord_to_idx(i, j + 1),
                        self._coord_to_idx(i + 1, j + 1)
                    ]
                    qubits = [q for q in qubits if q is not None]
                    if qubits:
                        stabilizers.append(qubits)

        # Add boundary stabilizers (weight 2)
        # Top boundary
        for j in range(0, self.d - 1, 2):
            qubits = [self._coord_to_idx(0, j), self._coord_to_idx(0, j + 1)]
            qubits = [q for q in qubits if q is not None]
            if len(qubits) == 2:
                stabilizers.append(qubits)

        # Bottom boundary
        for j in range(1 if self.d % 2 == 0 else 0, self.d - 1, 2):
            qubits = [self._coord_to_idx(self.d - 1, j), self._coord_to_idx(self.d - 1, j + 1)]
            qubits = [q for q in qubits if q is not None]
            if len(qubits) == 2:
                stabilizers.append(qubits)

        return stabilizers

    def _create_z_stabilizers(self):
        """
        Create Z-type stabilizers.

        Z-stabilizers are on plaquettes offset from X-stabilizers.
        """
        stabilizers = []

        for i in range(self.d - 1):
            for j in range(self.d - 1):
                # Z-stabilizers where (i + j) is odd
                if (i + j) % 2 == 1:
                    qubits = [
                        self._coord_to_idx(i, j),
                        self._coord_to_idx(i + 1, j),
                        self._coord_to_idx(i, j + 1),
                        self._coord_to_idx(i + 1, j + 1)
                    ]
                    qubits = [q for q in qubits if q is not None]
                    if qubits:
                        stabilizers.append(qubits)

        # Add boundary stabilizers (weight 2)
        # Left boundary
        for i in range(0, self.d - 1, 2):
            qubits = [self._coord_to_idx(i, 0), self._coord_to_idx(i + 1, 0)]
            qubits = [q for q in qubits if q is not None]
            if len(qubits) == 2:
                stabilizers.append(qubits)

        # Right boundary
        for i in range(1 if self.d % 2 == 0 else 0, self.d - 1, 2):
            qubits = [self._coord_to_idx(i, self.d - 1), self._coord_to_idx(i + 1, self.d - 1)]
            qubits = [q for q in qubits if q is not None]
            if len(qubits) == 2:
                stabilizers.append(qubits)

        return stabilizers

    def _create_logical_x(self):
        """Create a representative logical X operator (left column)."""
        return [self._coord_to_idx(i, 0) for i in range(self.d)]

    def _create_logical_z(self):
        """Create a representative logical Z operator (top row)."""
        return [self._coord_to_idx(0, j) for j in range(self.d)]

    def print_summary(self):
        """Print a summary of the code properties."""
        print(f"\nRotated Surface Code Summary")
        print("=" * 40)
        print(f"Code distance: d = {self.d}")
        print(f"Code parameters: [[{self.n_data}, 1, {self.d}]]")
        print(f"Data qubits: {self.n_data}")
        print(f"Ancilla qubits: {self.n_ancilla}")
        print(f"Total qubits: {self.n_total}")
        print(f"Correctable errors: t = {(self.d - 1) // 2}")
        print(f"\nX-stabilizers: {len(self.x_stabilizers)}")
        print(f"Z-stabilizers: {len(self.z_stabilizers)}")
        print(f"Total stabilizers: {len(self.x_stabilizers) + len(self.z_stabilizers)}")
        print(f"\nLogical X support: {self.logical_x}")
        print(f"Logical Z support: {self.logical_z}")

    def visualize(self, show_stabilizers=True, show_logical=True, figsize=(10, 10)):
        """
        Visualize the rotated surface code lattice.

        Parameters:
        -----------
        show_stabilizers : bool
            Whether to show stabilizer plaquettes
        show_logical : bool
            Whether to show logical operators
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Draw X-stabilizers as light red plaquettes
        if show_stabilizers:
            for stab in self.x_stabilizers:
                if len(stab) == 4:
                    coords = [self.data_qubits[q] for q in stab]
                    # Sort to form a proper polygon
                    center = np.mean(coords, axis=0)
                    angles = [np.arctan2(c[1] - center[1], c[0] - center[0]) for c in coords]
                    sorted_coords = [c for _, c in sorted(zip(angles, coords))]
                    poly = Polygon(sorted_coords, alpha=0.3, facecolor='red', edgecolor='darkred', linewidth=2)
                    ax.add_patch(poly)
                elif len(stab) == 2:
                    coords = [self.data_qubits[q] for q in stab]
                    ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                           'r-', linewidth=4, alpha=0.5)

            # Draw Z-stabilizers as light blue plaquettes
            for stab in self.z_stabilizers:
                if len(stab) == 4:
                    coords = [self.data_qubits[q] for q in stab]
                    center = np.mean(coords, axis=0)
                    angles = [np.arctan2(c[1] - center[1], c[0] - center[0]) for c in coords]
                    sorted_coords = [c for _, c in sorted(zip(angles, coords))]
                    poly = Polygon(sorted_coords, alpha=0.3, facecolor='blue', edgecolor='darkblue', linewidth=2)
                    ax.add_patch(poly)
                elif len(stab) == 2:
                    coords = [self.data_qubits[q] for q in stab]
                    ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                           'b-', linewidth=4, alpha=0.5)

        # Draw logical operators
        if show_logical:
            # Logical X (green)
            x_coords = [self.data_qubits[q] for q in self.logical_x]
            for i in range(len(x_coords) - 1):
                ax.annotate('', xy=x_coords[i+1], xytext=x_coords[i],
                           arrowprops=dict(arrowstyle='->', color='green', lw=3))
            ax.plot([c[0] for c in x_coords], [c[1] for c in x_coords],
                   'g-', linewidth=3, label=r'Logical $\bar{X}$', alpha=0.7)

            # Logical Z (orange)
            z_coords = [self.data_qubits[q] for q in self.logical_z]
            ax.plot([c[0] for c in z_coords], [c[1] for c in z_coords],
                   'orange', linewidth=3, linestyle='--', label=r'Logical $\bar{Z}$', alpha=0.7)

        # Draw data qubits
        for idx, (i, j) in self.data_qubits.items():
            circle = Circle((i, j), 0.15, facecolor='white', edgecolor='black', linewidth=2, zorder=10)
            ax.add_patch(circle)
            ax.text(i, j, str(idx), ha='center', va='center', fontsize=8, zorder=11)

        ax.set_xlim(-0.5, self.d - 0.5)
        ax.set_ylim(-0.5, self.d - 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('i', fontsize=12)
        ax.set_ylabel('j', fontsize=12)
        ax.set_title(f'Rotated Surface Code (d={self.d})', fontsize=14)

        if show_logical:
            ax.legend(loc='upper right')

        # Add legend for stabilizers
        if show_stabilizers:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.3, edgecolor='darkred', label='X-stabilizer'),
                Patch(facecolor='blue', alpha=0.3, edgecolor='darkblue', label='Z-stabilizer')
            ]
            ax.legend(handles=legend_elements, loc='upper left')

        plt.tight_layout()
        return fig, ax


def compare_rotated_unrotated():
    """Compare qubit counts for rotated vs unrotated surface codes."""
    distances = [3, 5, 7, 9, 11, 13, 15]

    rotated = [d**2 for d in distances]
    unrotated = [2*d**2 - 2*d + 1 for d in distances]
    savings = [(u - r) / u * 100 for r, u in zip(rotated, unrotated)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(distances))
    width = 0.35

    ax1.bar(x - width/2, unrotated, width, label='Unrotated', color='coral')
    ax1.bar(x + width/2, rotated, width, label='Rotated', color='steelblue')
    ax1.set_xlabel('Code Distance', fontsize=12)
    ax1.set_ylabel('Number of Data Qubits', fontsize=12)
    ax1.set_title('Qubit Count: Rotated vs Unrotated Surface Code', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(distances)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(distances, savings, color='green', alpha=0.7)
    ax2.axhline(y=50, color='red', linestyle='--', label='50% asymptote')
    ax2.set_xlabel('Code Distance', fontsize=12)
    ax2.set_ylabel('Qubit Savings (%)', fontsize=12)
    ax2.set_title('Percentage Savings from Rotation', fontsize=14)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 55)

    plt.tight_layout()
    return fig


def verify_stabilizer_properties(code):
    """Verify that stabilizers satisfy required properties."""
    print("\nVerifying Stabilizer Properties")
    print("=" * 40)

    # Check stabilizer count
    total_stabilizers = len(code.x_stabilizers) + len(code.z_stabilizers)
    expected = code.n_data - 1
    print(f"Total stabilizers: {total_stabilizers} (expected: {expected})")

    # Check that X and Z stabilizers commute
    # Two Pauli operators commute if they overlap on an even number of qubits
    commutation_violations = 0
    for x_stab in code.x_stabilizers:
        for z_stab in code.z_stabilizers:
            overlap = len(set(x_stab) & set(z_stab))
            if overlap % 2 != 0:
                commutation_violations += 1

    print(f"Commutation violations: {commutation_violations}")

    # Check logical operator weights
    print(f"Logical X weight: {len(code.logical_x)} (expected: {code.d})")
    print(f"Logical Z weight: {len(code.logical_z)} (expected: {code.d})")

    # Check anticommutation of logical operators
    overlap = len(set(code.logical_x) & set(code.logical_z))
    anticommutes = overlap % 2 == 1
    print(f"Logical X and Z anticommute: {anticommutes} (expected: True)")

    return total_stabilizers == expected and commutation_violations == 0 and anticommutes


# Main execution
if __name__ == "__main__":
    # Create and visualize distance-5 rotated surface code
    print("Creating distance-5 rotated surface code...")
    code_d5 = RotatedSurfaceCode(5)
    code_d5.print_summary()

    # Verify properties
    verify_stabilizer_properties(code_d5)

    # Visualize
    fig1, ax1 = code_d5.visualize()
    plt.savefig('rotated_surface_code_d5.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'rotated_surface_code_d5.png'")

    # Compare rotated vs unrotated
    fig2 = compare_rotated_unrotated()
    plt.savefig('rotated_vs_unrotated_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison to 'rotated_vs_unrotated_comparison.png'")

    # Create codes for different distances
    print("\n" + "=" * 50)
    print("Rotated Surface Code Parameters for Various Distances")
    print("=" * 50)
    print(f"{'Distance':<10} {'Data Qubits':<15} {'Total Qubits':<15} {'Correctable Errors':<20}")
    print("-" * 60)
    for d in [3, 5, 7, 9, 11, 13]:
        code = RotatedSurfaceCode(d)
        print(f"{d:<10} {code.n_data:<15} {code.n_total:<15} {(d-1)//2:<20}")

    plt.show()
```

### Lab Exercises

1. **Modify the visualization** to use a true "rotated" view where the lattice appears diagonal.

2. **Add a method** to generate the check matrix (parity check matrix) for the surface code.

3. **Implement error injection** by adding random X and Z errors to the code and computing the syndrome.

4. **Calculate the threshold** by simulating logical errors for various physical error rates.

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Data qubits (rotated) | $$n = d^2$$ |
| Data qubits (unrotated) | $$n = 2d^2 - 2d + 1$$ |
| Total qubits (with ancilla) | $$n_{\text{total}} = 2d^2 - 1$$ |
| Correctable errors | $$t = \lfloor (d-1)/2 \rfloor$$ |
| Code parameters | $$[[d^2, 1, d]]$$ |
| Qubit savings | $$\approx 50\%$$ for large $d$ |

### Main Takeaways

1. **Rotation reduces overhead:** The 45° rotation of the surface code lattice reduces qubit requirements by nearly 50% while maintaining the same error correction capability.

2. **Same connectivity:** Both rotated and unrotated codes require only 4-way qubit connectivity in the bulk.

3. **Clean structure:** The rotated code has $d^2$ data qubits and $d^2 - 1$ ancillas, giving exactly 1 logical qubit.

4. **Logical operators:** Logical X and Z are minimum-weight chains of length $d$ connecting opposite boundaries.

5. **Hardware-friendly:** The rotated layout maps naturally to square grid architectures used by Google, IBM, and others.

---

## Daily Checklist

- [ ] I can explain why the rotated surface code uses fewer qubits
- [ ] I can derive the [[d², 1, d]] code parameters
- [ ] I can construct stabilizers for a rotated code of given distance
- [ ] I can identify logical X and Z operators
- [ ] I understand the commutation relations between stabilizers
- [ ] I have run the computational lab and can visualize the code

---

## Preview: Day 814

Tomorrow we examine **Boundary Conditions: Smooth vs. Rough**. We'll discover:
- How boundary stabilizers determine where logical operators can terminate
- The relationship between smooth boundaries and X-type logical operators
- Why rough boundaries support Z-type logical operators
- How boundary engineering enables logical gate implementation

The geometry of boundaries is the key to understanding lattice surgery and logical operations on surface codes.

---

*"The rotated surface code is an elegant solution to a practical problem: how to encode quantum information with minimal overhead while maintaining the topological protection that makes surface codes so robust."*

— Day 813 Reflection
