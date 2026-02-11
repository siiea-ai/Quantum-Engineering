# Day 814: Boundary Conditions — Smooth vs. Rough

## Week 117, Day 2 | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Overview

The boundaries of a surface code are not mere edges—they are the defining features that determine which logical operators exist and how they can be manipulated. Today we explore the fundamental dichotomy between **smooth boundaries** (supporting Z-stabilizers, terminating X logical operators) and **rough boundaries** (supporting X-stabilizers, terminating Z logical operators). This understanding is essential for lattice surgery, logical gate implementation, and surface code architecture design.

---

## Daily Schedule

| Session | Duration | Content |
|---------|----------|---------|
| Morning | 3 hours | Boundary theory, stabilizer truncation, logical operator termination |
| Afternoon | 2 hours | Boundary engineering exercises, worked examples |
| Evening | 2 hours | Python visualization of boundary types and logical operators |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Distinguish** smooth and rough boundaries by their stabilizer structure
2. **Explain** why logical X terminates on smooth boundaries and logical Z on rough boundaries
3. **Derive** boundary stabilizer forms from bulk stabilizer truncation
4. **Design** surface code patches with specified boundary configurations
5. **Calculate** how boundary choice affects logical operator weight
6. **Implement** boundary-aware surface code visualizations

---

## Core Content

### 1. Bulk Stabilizers and Their Boundaries

In the bulk of a rotated surface code, stabilizers are weight-4 operators:

**Bulk X-stabilizer:**
$$S_X = X_1 X_2 X_3 X_4$$

**Bulk Z-stabilizer:**
$$S_Z = Z_1 Z_2 Z_3 Z_4$$

At boundaries, some of these qubits "disappear," and stabilizers become **truncated**.

### 2. Smooth Boundaries: Z-Type

A **smooth boundary** is characterized by:
- **Truncated Z-stabilizers** of weight 2 at the edge
- **X-stabilizers terminate** at the boundary
- **Logical X operators** can end on smooth boundaries

**Mathematical Definition:**
At a smooth boundary, Z-stabilizers lose qubits and become:
$$S_Z^{\text{smooth}} = Z_1 Z_2 \quad \text{(weight 2)}$$

The boundary can be thought of as where we "smooth out" the lattice by removing vertices.

**Physical Picture:**
```
Bulk:           Boundary:
  Z               Z
 /|\             /|
Z-+-Z    -->    Z-+  (smooth edge)
 \|/             \|
  Z               Z
```

**Key Property:** A string of X operators can terminate on a smooth boundary without creating a syndrome, because there is no X-stabilizer to detect the endpoint.

### 3. Rough Boundaries: X-Type

A **rough boundary** is characterized by:
- **Truncated X-stabilizers** of weight 2 at the edge
- **Z-stabilizers terminate** at the boundary
- **Logical Z operators** can end on rough boundaries

**Mathematical Definition:**
At a rough boundary, X-stabilizers lose qubits and become:
$$S_X^{\text{rough}} = X_1 X_2 \quad \text{(weight 2)}$$

**Physical Picture:**
```
Bulk:           Boundary:
  X               X
 /|\             /|
X-+-X    -->    X-+  (rough edge)
 \|/             \|
  X               X
```

**Key Property:** A string of Z operators can terminate on a rough boundary without creating a syndrome.

### 4. Standard Surface Code Boundary Configuration

The standard rotated surface code has:
- **Top and bottom:** Smooth boundaries
- **Left and right:** Rough boundaries

$$\boxed{\text{Standard configuration: } \begin{matrix} \text{Smooth} \\ \text{Rough} \leftarrow \square \rightarrow \text{Rough} \\ \text{Smooth} \end{matrix}}$$

This configuration ensures:
- Logical $\bar{X}$: Vertical chain connecting smooth (top) to smooth (bottom)
- Logical $\bar{Z}$: Horizontal chain connecting rough (left) to rough (right)

### 5. Boundary-Logical Operator Correspondence

The relationship between boundaries and logical operators follows a fundamental rule:

$$\boxed{\bar{X}: \text{smooth} \leftrightarrow \text{smooth}}$$
$$\boxed{\bar{Z}: \text{rough} \leftrightarrow \text{rough}}$$

**Why This Works:**

Consider a string of X operators (potential logical $\bar{X}$):
- In the bulk, the string commutes with all Z-stabilizers (passes through)
- At a **smooth boundary**, the endpoint has no X-stabilizer to detect it
- At a **rough boundary**, the endpoint would anticommute with the weight-2 X-stabilizer, creating a syndrome

Therefore, X-strings can only terminate on smooth boundaries without being detected.

The dual argument applies to Z-strings and rough boundaries.

### 6. Boundary Stabilizer Counting

For a distance-$d$ rotated surface code with alternating boundaries:

**Bulk stabilizers:**
- Bulk X-stabilizers: $\frac{(d-1)^2}{2}$ (approximately)
- Bulk Z-stabilizers: $\frac{(d-1)^2}{2}$ (approximately)

**Boundary stabilizers:**
- Weight-2 stabilizers on each edge: $\frac{d-1}{2}$ per edge

**Total check:**
$$n_X + n_Z = d^2 - 1$$

This confirms $k = n - (n_X + n_Z) = d^2 - (d^2 - 1) = 1$ logical qubit.

### 7. Alternative Boundary Configurations

We can create surface codes with different boundary arrangements:

**All-Smooth Boundaries (Toric Code on a Plane):**
- 4 smooth boundaries
- Logical Z operator wraps around (doesn't exist for finite planar code)
- Used in certain theoretical constructions

**All-Rough Boundaries:**
- 4 rough boundaries
- Logical X operator cannot terminate
- Also used in theoretical constructions

**Mixed Configurations:**
- Different arrangements enable different logical gates
- Basis for defect-based computation

### 8. Logical Operator Weight and Boundaries

The minimum weight of a logical operator depends on the boundary geometry:

For a $d \times d$ surface code with standard boundaries:
$$|\bar{X}|_{\min} = d \quad \text{(vertical distance)}$$
$$|\bar{Z}|_{\min} = d \quad \text{(horizontal distance)}$$

For a $d_X \times d_Z$ rectangular code:
$$|\bar{X}|_{\min} = d_X \quad \text{(height)}$$
$$|\bar{Z}|_{\min} = d_Z \quad \text{(width)}$$

This allows **asymmetric codes** optimized for different error types:

$$\boxed{[[d_X \cdot d_Z, 1, \min(d_X, d_Z)]]}$$

### 9. Boundary Engineering for Lattice Surgery

Lattice surgery—the primary method for logical gates in surface codes—relies on boundary manipulation:

**Merge Operation:**
- Join two patches by turning off stabilizers at the shared boundary
- Creates larger logical qubit(s)
- Used for CNOT and other entangling gates

**Split Operation:**
- Divide one patch into two by activating boundary stabilizers
- Measures logical operators
- Completes gate implementation

The boundary types must match for proper operation:
$$\boxed{\text{Smooth} \leftrightarrow \text{Smooth merge: measures } Z \otimes Z}$$
$$\boxed{\text{Rough} \leftrightarrow \text{Rough merge: measures } X \otimes X}$$

---

## Quantum Computing Connection

### Google Willow (2024) Implementation

Google's Willow processor demonstrates boundary control:

1. **Flexible patch sizes:** Can configure patches of different dimensions
2. **Boundary stabilizers:** Weight-2 stabilizers measured at edges
3. **Logical readout:** Measures logical operators by connecting boundaries

The boundary configuration determines which logical Pauli basis is measured during readout.

### IBM's Approach

IBM's heavy-hex lattice uses modified boundary structures:
- 3-way connectivity requires different stabilizer arrangements
- Boundaries adapted for heavy-hex geometry
- Flag qubits used for syndrome extraction at boundaries

---

## Worked Examples

### Example 1: Identifying Boundary Types

**Problem:** Given a distance-5 surface code with the following stabilizer structure at the top edge (qubits 0-4), determine the boundary type.

Top edge stabilizers:
- $S_1 = Z_0 Z_1$
- $S_2 = Z_2 Z_3$

**Solution:**

The edge has weight-2 Z-stabilizers. This is the definition of a **smooth boundary**.

At this boundary:
- Z-stabilizers are truncated (weight 2 instead of 4)
- X-stabilizers are absent
- X-type logical operators can terminate here

Therefore, if an X-string reaches this edge, it won't create a syndrome. The top edge is smooth.

---

### Example 2: Designing a Rectangular Code

**Problem:** Design a surface code optimized for Z-errors (which are 3x more common than X-errors). Target: correct up to 2 Z-errors and 1 X-error.

**Solution:**

For correcting $t$ errors, we need distance $d \geq 2t + 1$:
- Z-errors: $d_Z \geq 2(2) + 1 = 5$
- X-errors: $d_X \geq 2(1) + 1 = 3$

Design a $3 \times 5$ rectangular code:
- Width (rough boundaries): 5 qubits → $d_Z = 5$
- Height (smooth boundaries): 3 qubits → $d_X = 3$

**Physical qubits:** $3 \times 5 = 15$ data qubits

**Comparison to symmetric code:**
- Symmetric $d=5$ code: 25 data qubits
- Asymmetric $3 \times 5$ code: 15 data qubits
- Savings: 40%

The rectangular code is more efficient when error types are asymmetric.

---

### Example 3: Boundary Merge Analysis

**Problem:** Two distance-3 surface code patches are positioned side-by-side, sharing a rough boundary. When we merge them, what operator is measured?

**Solution:**

**Initial Configuration:**
```
Patch A          Patch B
+-------+        +-------+
|       |        |       |
| □  A  | rough  | □  B  |
|       |        |       |
+-------+        +-------+
```

At the shared rough boundary:
- Both patches have weight-2 X-stabilizers at their edges
- When we merge, we activate bulk stabilizers that span both patches

**Merge Operation:**
The new stabilizers across the boundary are X-type (bulk stabilizers).

**Measured Operator:**
When rough boundaries merge, the measurement outcome tells us:
$$\bar{X}_A \otimes \bar{X}_B$$

This is because the X-type stabilizers that were at the boundary now span both patches, effectively measuring the product of logical X operators.

$$\boxed{\text{Rough-rough merge measures } \bar{X}_A \bar{X}_B}$$

For smooth-smooth merge, the measured operator would be $\bar{Z}_A \bar{Z}_B$.

---

## Practice Problems

### Direct Application

**Problem 1:** A distance-7 surface code has smooth boundaries on the top and bottom, rough boundaries on the left and right. How many weight-2 stabilizers are on each type of boundary?

**Problem 2:** Identify the boundary types in a surface code where:
- Top edge has stabilizers: $X_0 X_1$, $X_2 X_3$, $X_4 X_5$
- Right edge has stabilizers: $Z_5 Z_{12}$, $Z_{19} Z_{26}$

**Problem 3:** For a $5 \times 7$ rectangular surface code, what are:
a) The minimum weight of the logical X operator?
b) The minimum weight of the logical Z operator?
c) The number of correctable X-errors?
d) The number of correctable Z-errors?

### Intermediate

**Problem 4:** Prove that a string of X operators must anticommute with a weight-2 X-stabilizer if it terminates at a rough boundary (and therefore cannot be a valid endpoint for logical X).

**Problem 5:** Design a boundary configuration for a surface code that encodes 2 logical qubits instead of 1. (Hint: Consider a code with holes or multiple connected regions.)

**Problem 6:** Two surface code patches will undergo lattice surgery. Patch A has a smooth right boundary; Patch B has a rough left boundary. Can they be directly merged? If not, what modification is needed?

### Challenging

**Problem 7:** Consider a "twisted" boundary condition where the boundary type alternates along an edge (smooth-rough-smooth-rough...). What are the implications for logical operators?

**Problem 8:** Derive the stabilizer generators for a hexagonal surface code patch. How do boundary conditions differ from the square lattice case?

**Problem 9:** In the Raussendorf-Harrington-Goyal 3D cluster state model, boundaries exist in both space and time. How does the smooth/rough distinction extend to this 3D setting?

---

## Computational Lab

### Lab 814: Boundary Visualization and Analysis

```python
"""
Day 814 Computational Lab: Boundary Conditions in Surface Codes
================================================================

This lab visualizes smooth and rough boundaries, demonstrates logical
operator termination, and explores rectangular code configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

class BoundaryAwareSurfaceCode:
    """
    Surface code with explicit boundary type tracking.
    """

    def __init__(self, d_x: int, d_z: int):
        """
        Initialize a rectangular surface code.

        Parameters:
        -----------
        d_x : int
            Distance for X-type errors (height, smooth boundary separation)
        d_z : int
            Distance for Z-type errors (width, rough boundary separation)
        """
        self.d_x = d_x
        self.d_z = d_z
        self.n_data = d_x * d_z

        # Boundary types
        self.boundaries = {
            'top': 'smooth',
            'bottom': 'smooth',
            'left': 'rough',
            'right': 'rough'
        }

        # Create data qubit grid
        self.data_qubits = self._create_data_qubits()

        # Create stabilizers with boundary awareness
        self.x_stabilizers, self.z_stabilizers = self._create_stabilizers()

        # Create logical operators
        self.logical_x = self._create_logical_x()
        self.logical_z = self._create_logical_z()

    def _create_data_qubits(self):
        """Create data qubit positions."""
        qubits = {}
        idx = 0
        for i in range(self.d_x):
            for j in range(self.d_z):
                qubits[idx] = (i, j)
                idx += 1
        return qubits

    def _coord_to_idx(self, i, j):
        """Convert coordinates to qubit index."""
        if 0 <= i < self.d_x and 0 <= j < self.d_z:
            return i * self.d_z + j
        return None

    def _create_stabilizers(self):
        """Create X and Z stabilizers with proper boundary handling."""
        x_stabs = []
        z_stabs = []

        # Bulk and boundary X-stabilizers
        for i in range(self.d_x - 1):
            for j in range(self.d_z - 1):
                if (i + j) % 2 == 0:  # X-stabilizer positions
                    qubits = [
                        self._coord_to_idx(i, j),
                        self._coord_to_idx(i + 1, j),
                        self._coord_to_idx(i, j + 1),
                        self._coord_to_idx(i + 1, j + 1)
                    ]
                    qubits = [q for q in qubits if q is not None]
                    x_stabs.append({'qubits': qubits, 'type': 'bulk'})

        # Left boundary (rough): weight-2 X-stabilizers
        for i in range(0, self.d_x - 1, 2):
            qubits = [self._coord_to_idx(i, 0), self._coord_to_idx(i + 1, 0)]
            qubits = [q for q in qubits if q is not None]
            if len(qubits) == 2:
                x_stabs.append({'qubits': qubits, 'type': 'boundary', 'edge': 'left'})

        # Right boundary (rough): weight-2 X-stabilizers
        for i in range(0, self.d_x - 1, 2):
            qubits = [self._coord_to_idx(i, self.d_z - 1), self._coord_to_idx(i + 1, self.d_z - 1)]
            qubits = [q for q in qubits if q is not None]
            if len(qubits) == 2:
                x_stabs.append({'qubits': qubits, 'type': 'boundary', 'edge': 'right'})

        # Bulk and boundary Z-stabilizers
        for i in range(self.d_x - 1):
            for j in range(self.d_z - 1):
                if (i + j) % 2 == 1:  # Z-stabilizer positions
                    qubits = [
                        self._coord_to_idx(i, j),
                        self._coord_to_idx(i + 1, j),
                        self._coord_to_idx(i, j + 1),
                        self._coord_to_idx(i + 1, j + 1)
                    ]
                    qubits = [q for q in qubits if q is not None]
                    z_stabs.append({'qubits': qubits, 'type': 'bulk'})

        # Top boundary (smooth): weight-2 Z-stabilizers
        for j in range(0, self.d_z - 1, 2):
            qubits = [self._coord_to_idx(0, j), self._coord_to_idx(0, j + 1)]
            qubits = [q for q in qubits if q is not None]
            if len(qubits) == 2:
                z_stabs.append({'qubits': qubits, 'type': 'boundary', 'edge': 'top'})

        # Bottom boundary (smooth): weight-2 Z-stabilizers
        for j in range(0, self.d_z - 1, 2):
            qubits = [self._coord_to_idx(self.d_x - 1, j), self._coord_to_idx(self.d_x - 1, j + 1)]
            qubits = [q for q in qubits if q is not None]
            if len(qubits) == 2:
                z_stabs.append({'qubits': qubits, 'type': 'boundary', 'edge': 'bottom'})

        return x_stabs, z_stabs

    def _create_logical_x(self):
        """Logical X: vertical chain (smooth to smooth)."""
        return [self._coord_to_idx(i, 0) for i in range(self.d_x)]

    def _create_logical_z(self):
        """Logical Z: horizontal chain (rough to rough)."""
        return [self._coord_to_idx(0, j) for j in range(self.d_z)]

    def get_boundary_stabilizers(self, edge: str):
        """Get stabilizers on a specific edge."""
        if edge in ['left', 'right']:
            return [s for s in self.x_stabilizers if s.get('edge') == edge]
        else:
            return [s for s in self.z_stabilizers if s.get('edge') == edge]

    def print_summary(self):
        """Print code summary."""
        print(f"\nRectangular Surface Code: {self.d_x} x {self.d_z}")
        print("=" * 45)
        print(f"Data qubits: {self.n_data}")
        print(f"X-distance (height): {self.d_x}")
        print(f"Z-distance (width): {self.d_z}")
        print(f"Code distance: {min(self.d_x, self.d_z)}")
        print(f"\nBoundary configuration:")
        for edge, btype in self.boundaries.items():
            print(f"  {edge}: {btype}")
        print(f"\nLogical X weight: {len(self.logical_x)}")
        print(f"Logical Z weight: {len(self.logical_z)}")


def visualize_boundaries(code, figsize=(12, 10)):
    """
    Create detailed boundary visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw boundary regions
    # Smooth boundaries (top/bottom) - green
    ax.axhspan(-0.6, -0.4, color='green', alpha=0.3, label='Smooth boundary')
    ax.axhspan(code.d_x - 0.6, code.d_x - 0.4, color='green', alpha=0.3)

    # Rough boundaries (left/right) - orange
    ax.axvspan(-0.6, -0.4, color='orange', alpha=0.3, label='Rough boundary')
    ax.axvspan(code.d_z - 0.6, code.d_z - 0.4, color='orange', alpha=0.3)

    # Draw bulk X-stabilizers (red)
    for stab in code.x_stabilizers:
        if stab['type'] == 'bulk':
            coords = [code.data_qubits[q] for q in stab['qubits']]
            center = np.mean(coords, axis=0)
            angles = [np.arctan2(c[1] - center[1], c[0] - center[0]) for c in coords]
            sorted_coords = [c for _, c in sorted(zip(angles, coords))]
            poly = Polygon(sorted_coords, alpha=0.2, facecolor='red', edgecolor='darkred', linewidth=1.5)
            ax.add_patch(poly)
        else:
            # Boundary X-stabilizer (weight 2)
            coords = [code.data_qubits[q] for q in stab['qubits']]
            ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                   'r-', linewidth=6, alpha=0.4)

    # Draw bulk Z-stabilizers (blue)
    for stab in code.z_stabilizers:
        if stab['type'] == 'bulk':
            coords = [code.data_qubits[q] for q in stab['qubits']]
            center = np.mean(coords, axis=0)
            angles = [np.arctan2(c[1] - center[1], c[0] - center[0]) for c in coords]
            sorted_coords = [c for _, c in sorted(zip(angles, coords))]
            poly = Polygon(sorted_coords, alpha=0.2, facecolor='blue', edgecolor='darkblue', linewidth=1.5)
            ax.add_patch(poly)
        else:
            # Boundary Z-stabilizer (weight 2)
            coords = [code.data_qubits[q] for q in stab['qubits']]
            ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                   'b-', linewidth=6, alpha=0.4)

    # Draw logical X (green arrow, vertical)
    x_coords = [code.data_qubits[q] for q in code.logical_x]
    for i in range(len(x_coords) - 1):
        dx = x_coords[i+1][0] - x_coords[i][0]
        dy = x_coords[i+1][1] - x_coords[i][1]
        ax.arrow(x_coords[i][0], x_coords[i][1], dx*0.8, dy*0.8,
                head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2, zorder=5)

    # Draw logical Z (purple arrow, horizontal)
    z_coords = [code.data_qubits[q] for q in code.logical_z]
    for i in range(len(z_coords) - 1):
        dx = z_coords[i+1][0] - z_coords[i][0]
        dy = z_coords[i+1][1] - z_coords[i][1]
        ax.arrow(z_coords[i][0], z_coords[i][1], dx*0.8, dy*0.8,
                head_width=0.15, head_length=0.1, fc='purple', ec='purple', linewidth=2, zorder=5)

    # Draw data qubits
    for idx, (i, j) in code.data_qubits.items():
        circle = Circle((i, j), 0.12, facecolor='white', edgecolor='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(i, j, str(idx), ha='center', va='center', fontsize=7, zorder=11)

    # Add boundary labels
    ax.text(code.d_x/2, -0.8, 'Smooth (Z-type)', ha='center', fontsize=10, color='darkgreen', weight='bold')
    ax.text(code.d_x/2, code.d_z + 0.3, 'Smooth (Z-type)', ha='center', fontsize=10, color='darkgreen', weight='bold')
    ax.text(-0.8, code.d_z/2, 'Rough\n(X-type)', ha='center', va='center', fontsize=10, color='darkorange', weight='bold', rotation=90)
    ax.text(code.d_x + 0.3, code.d_z/2, 'Rough\n(X-type)', ha='center', va='center', fontsize=10, color='darkorange', weight='bold', rotation=90)

    ax.set_xlim(-1, code.d_x + 0.5)
    ax.set_ylim(-1, code.d_z + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Row (i)', fontsize=12)
    ax.set_ylabel('Column (j)', fontsize=12)
    ax.set_title(f'Boundary Structure: {code.d_x}x{code.d_z} Surface Code', fontsize=14)

    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.3, label='X-stabilizer'),
        mpatches.Patch(facecolor='blue', alpha=0.3, label='Z-stabilizer'),
        mpatches.Patch(facecolor='green', alpha=0.3, label='Smooth boundary'),
        mpatches.Patch(facecolor='orange', alpha=0.3, label='Rough boundary'),
        Line2D([0], [0], color='green', linewidth=2, label=r'Logical $\bar{X}$'),
        Line2D([0], [0], color='purple', linewidth=2, label=r'Logical $\bar{Z}$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig, ax


def demonstrate_logical_termination():
    """
    Show why logical operators terminate on specific boundaries.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Example with X-string termination
    ax1 = axes[0]
    ax1.set_title("X-string termination: Smooth boundary OK", fontsize=12)

    # Simple 3x3 grid
    for i in range(3):
        for j in range(3):
            circle = Circle((i, j), 0.15, facecolor='white', edgecolor='black', linewidth=2)
            ax1.add_patch(circle)

    # X-string from top to bottom (smooth to smooth)
    for i in range(3):
        ax1.annotate('X', (0, i), fontsize=14, color='green', ha='center', va='bottom')
        if i < 2:
            ax1.arrow(0, i + 0.2, 0, 0.55, head_width=0.1, head_length=0.05, fc='green', ec='green')

    # Show smooth boundaries (Z-stabilizers)
    ax1.plot([-0.3, -0.3], [-0.3, 2.3], 'g-', linewidth=5, alpha=0.5)
    ax1.text(-0.5, 1, 'Smooth\n(no X-stab)', ha='right', fontsize=9, color='green')

    ax1.plot([2.3, 2.3], [-0.3, 2.3], 'g-', linewidth=5, alpha=0.5)

    # Show where X-string would fail (rough boundary)
    ax1.plot([-0.3, 2.3], [-0.3, -0.3], 'orange', linewidth=5, alpha=0.5)
    ax1.text(1, -0.6, 'Rough (X-stab would detect endpoint)', ha='center', fontsize=9, color='darkorange')

    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-1, 3)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Example with Z-string termination
    ax2 = axes[1]
    ax2.set_title("Z-string termination: Rough boundary OK", fontsize=12)

    for i in range(3):
        for j in range(3):
            circle = Circle((i, j), 0.15, facecolor='white', edgecolor='black', linewidth=2)
            ax2.add_patch(circle)

    # Z-string from left to right (rough to rough)
    for j in range(3):
        ax2.annotate('Z', (j, 2), fontsize=14, color='purple', ha='center', va='bottom')
        if j < 2:
            ax2.arrow(j + 0.2, 2, 0.55, 0, head_width=0.1, head_length=0.05, fc='purple', ec='purple')

    # Show rough boundaries (X-stabilizers)
    ax2.plot([-0.3, 2.3], [2.3, 2.3], 'orange', linewidth=5, alpha=0.5)
    ax2.text(1, 2.5, 'Rough (no Z-stab)', ha='center', fontsize=9, color='darkorange')

    ax2.plot([-0.3, 2.3], [-0.3, -0.3], 'orange', linewidth=5, alpha=0.5)

    # Show where Z-string would fail (smooth boundary)
    ax2.plot([-0.3, -0.3], [-0.3, 2.3], 'g-', linewidth=5, alpha=0.5)
    ax2.text(-0.5, 1, 'Smooth\n(Z-stab\nwould\ndetect)', ha='right', fontsize=8, color='green')

    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-1, 3)
    ax2.set_aspect('equal')
    ax2.axis('off')

    plt.tight_layout()
    return fig


def compare_rectangular_codes():
    """
    Compare different rectangular code configurations.
    """
    configs = [
        (3, 3, "Symmetric d=3"),
        (3, 5, "Asymmetric 3x5"),
        (3, 7, "Asymmetric 3x7"),
        (5, 5, "Symmetric d=5"),
        (5, 7, "Asymmetric 5x7"),
        (7, 7, "Symmetric d=7"),
    ]

    data = []
    for d_x, d_z, name in configs:
        code = BoundaryAwareSurfaceCode(d_x, d_z)
        data.append({
            'name': name,
            'd_x': d_x,
            'd_z': d_z,
            'n_data': code.n_data,
            'code_distance': min(d_x, d_z),
            'x_errors': (d_x - 1) // 2,
            'z_errors': (d_z - 1) // 2
        })

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [d['name'] for d in data]
    x = np.arange(len(names))

    # Qubit count
    ax1 = axes[0]
    ax1.bar(x, [d['n_data'] for d in data], color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Data Qubits')
    ax1.set_title('Qubit Count')
    ax1.grid(axis='y', alpha=0.3)

    # Code distances
    ax2 = axes[1]
    width = 0.35
    ax2.bar(x - width/2, [d['d_x'] for d in data], width, label='X-distance', color='coral')
    ax2.bar(x + width/2, [d['d_z'] for d in data], width, label='Z-distance', color='lightgreen')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Distance')
    ax2.set_title('X vs Z Distance')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Correctable errors
    ax3 = axes[2]
    ax3.bar(x - width/2, [d['x_errors'] for d in data], width, label='X-errors', color='coral')
    ax3.bar(x + width/2, [d['z_errors'] for d in data], width, label='Z-errors', color='lightgreen')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('Correctable Errors')
    ax3.set_title('Error Correction Capacity')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    # Create a rectangular surface code
    print("Creating 3x5 rectangular surface code...")
    code = BoundaryAwareSurfaceCode(3, 5)
    code.print_summary()

    # Visualize boundaries
    fig1, ax1 = visualize_boundaries(code)
    plt.savefig('boundary_visualization.png', dpi=150, bbox_inches='tight')
    print("\nSaved boundary visualization to 'boundary_visualization.png'")

    # Demonstrate logical operator termination
    fig2 = demonstrate_logical_termination()
    plt.savefig('logical_termination.png', dpi=150, bbox_inches='tight')
    print("Saved logical termination demo to 'logical_termination.png'")

    # Compare rectangular codes
    fig3 = compare_rectangular_codes()
    plt.savefig('rectangular_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved rectangular comparison to 'rectangular_comparison.png'")

    # Summary table
    print("\n" + "=" * 60)
    print("Boundary Types and Their Properties")
    print("=" * 60)
    print(f"{'Boundary':<12} {'Stabilizer Type':<18} {'Logical Operator':<20}")
    print("-" * 50)
    print(f"{'Smooth':<12} {'Z (weight 2)':<18} {'X terminates here':<20}")
    print(f"{'Rough':<12} {'X (weight 2)':<18} {'Z terminates here':<20}")

    plt.show()
```

### Lab Exercises

1. **Modify the code** to swap boundary types (smooth ↔ rough) and observe how logical operators change.

2. **Implement a merge simulation** that shows what happens when two patches with matching boundaries are joined.

3. **Create a visualization** of an error chain that crosses from one boundary type to another, showing the resulting syndrome.

4. **Design an asymmetric code** optimized for a specific error model where X-errors are twice as likely as Z-errors.

---

## Summary

### Key Formulas

| Concept | Description |
|---------|-------------|
| Smooth boundary | Z-stabilizers truncated to weight 2; X logical terminates |
| Rough boundary | X-stabilizers truncated to weight 2; Z logical terminates |
| $\bar{X}$ path | Smooth $\leftrightarrow$ Smooth |
| $\bar{Z}$ path | Rough $\leftrightarrow$ Rough |
| Rectangular code | $[[d_X \cdot d_Z, 1, \min(d_X, d_Z)]]$ |
| Smooth-smooth merge | Measures $\bar{Z} \otimes \bar{Z}$ |
| Rough-rough merge | Measures $\bar{X} \otimes \bar{X}$ |

### Main Takeaways

1. **Boundaries define logical operators:** The boundary types determine where logical X and Z operators can terminate.

2. **Smooth = Z-type stabilizers:** Smooth boundaries have weight-2 Z-stabilizers and allow X-strings to end.

3. **Rough = X-type stabilizers:** Rough boundaries have weight-2 X-stabilizers and allow Z-strings to end.

4. **Rectangular codes are efficient:** When error types are asymmetric, rectangular codes save qubits.

5. **Lattice surgery needs matching boundaries:** Merge operations require compatible boundary types.

---

## Daily Checklist

- [ ] I can distinguish smooth from rough boundaries by their stabilizers
- [ ] I understand why logical X terminates on smooth boundaries
- [ ] I can design rectangular codes for asymmetric error models
- [ ] I understand the boundary requirements for lattice surgery
- [ ] I have run the computational lab and visualized boundary structures

---

## Preview: Day 815

Tomorrow we explore **Twist Defects and Topological Properties**. We'll discover:
- Twist defects: points where boundary type changes
- How twists carry topological charge
- The role of twists in enabling non-Clifford gates
- Corner structures and their computational uses

Twist defects represent a bridge between topological error correction and topological quantum computation.

---

*"The boundaries of a surface code are not limitations—they are the handles by which we manipulate quantum information."*

— Day 814 Reflection
