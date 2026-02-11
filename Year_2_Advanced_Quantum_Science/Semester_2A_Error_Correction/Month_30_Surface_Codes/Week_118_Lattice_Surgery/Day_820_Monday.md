# Day 820: Lattice Surgery Fundamentals

## Week 118: Lattice Surgery & Logical Gates | Month 30: Surface Codes

### Semester 2A: Error Correction | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Code deformation theory, boundary types, logical operator topology |
| **Afternoon** | 2.5 hours | Boundary manipulation problems, stabilizer updates |
| **Evening** | 1.5 hours | Computational lab: Visualizing surface code boundaries |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 820, you will be able to:

1. **Explain the lattice surgery paradigm** and contrast it with transversal gate approaches
2. **Distinguish rough and smooth boundaries** on surface code patches
3. **Trace logical operator paths** across different boundary configurations
4. **Describe code deformation** as stabilizer modification in real-time
5. **Analyze the fault-tolerance properties** of boundary manipulation operations
6. **Implement boundary visualization** for planar surface code patches

---

## 1. Introduction: Beyond Transversal Gates

### The Challenge of Universal Fault-Tolerant Computation

Recall from Week 117 that the surface code provides excellent error correction properties:
- Threshold around 1% for depolarizing noise
- Efficient 2D planar layout
- Local stabilizer measurements only

However, implementing **logical gates** presents challenges:

**Transversal gates** (applying physical gates to each qubit) only work for some operations:
- Logical X: Apply X to all data qubits on rough boundary
- Logical Z: Apply Z to all data qubits on smooth boundary
- Logical H: Requires rotating the entire code (exchange rough ↔ smooth)

**The problem:** Transversal CNOT between surface code patches requires:
- All qubits of patch A to interact with corresponding qubits of patch B
- Long-range connections in hardware
- Impractical for 2D nearest-neighbor architectures

### The Lattice Surgery Solution

**Key insight:** Instead of moving quantum information, we **deform the code itself**.

$$\boxed{\text{Lattice Surgery} = \text{Code Deformation} + \text{Joint Measurements}}$$

Core operations:
1. **Merge:** Grow boundaries to connect two patches, creating joint logical operators
2. **Split:** Shrink merged region to separate patches, projecting into product state

This approach requires only **local operations** and fits naturally with planar qubit layouts.

---

## 2. Surface Code Patch Geometry

### Boundary Types and Logical Operators

A surface code patch has a **rectangular** (or more generally, planar) layout with two types of boundaries:

**Rough Boundaries (Z-type):**
- Z stabilizers terminate at the boundary
- X stabilizers extend to the edge with fewer qubits
- Logical Z operator runs along the rough boundary
- Colored "blue" in standard conventions

**Smooth Boundaries (X-type):**
- X stabilizers terminate at the boundary
- Z stabilizers extend to the edge
- Logical X operator runs along the smooth boundary
- Colored "red" in standard conventions

### Standard Patch Configuration

For a distance-d surface code patch:

```
     Smooth (X-type) boundary
    ┌────────────────────────┐
    │  ●──●──●──●──●──●──●   │
R   │  │  │  │  │  │  │  │   │ R
o   │  ●──●──●──●──●──●──●   │ o
u   │  │  │  │  │  │  │  │   │ u
g   │  ●──●──●──●──●──●──●   │ g
h   │  │  │  │  │  │  │  │   │ h
    │  ●──●──●──●──●──●──●   │
    └────────────────────────┘
     Smooth (X-type) boundary

● = data qubit
Logical Z: vertical path (rough to rough)
Logical X: horizontal path (smooth to smooth)
```

### Logical Operator Topology

**Logical Z operator:**
$$\bar{Z} = \prod_{i \in \text{vertical path}} Z_i$$

The path connects the two rough (Z-type) boundaries.

**Logical X operator:**
$$\bar{X} = \prod_{i \in \text{horizontal path}} X_i$$

The path connects the two smooth (X-type) boundaries.

$$\boxed{\bar{Z}\bar{X} = (-1)^{|\text{intersection}|} \bar{X}\bar{Z} = -\bar{X}\bar{Z}}$$

The operators anti-commute because any valid Z path and X path intersect an odd number of times.

---

## 3. Code Deformation Theory

### Stabilizer Modification During Deformation

Code deformation involves **adding or removing stabilizers** while preserving:
1. The stabilizer group structure (all stabilizers commute)
2. The logical qubit information (at least one degree of freedom)
3. Fault-tolerance (local errors remain correctable)

**Adding qubits and stabilizers:**

When we grow a boundary:
1. Initialize new qubits in appropriate eigenstates
2. Measure new stabilizers involving the fresh qubits
3. The code distance may temporarily decrease locally

**Removing qubits and stabilizers:**

When we shrink a boundary:
1. Stop measuring certain stabilizers
2. Measure out qubits in single-qubit basis
3. Update remaining stabilizers accordingly

### Mathematical Framework

Let $\mathcal{S} = \langle S_1, S_2, \ldots, S_n \rangle$ be the stabilizer group.

**Deformation as group modification:**

$$\mathcal{S} \rightarrow \mathcal{S}' = \langle S_1', S_2', \ldots, S_m' \rangle$$

Requirements:
1. All $S_i' \in \mathcal{P}_n$ (Pauli group on $n$ qubits)
2. $[S_i', S_j'] = 0$ for all $i, j$
3. $\bar{X}', \bar{Z}'$ anti-commute and commute with all $S_i'$

$$\boxed{\text{Valid deformation} \Leftrightarrow \text{Continuous path through stabilizer groups}}$$

---

## 4. Boundary Manipulation Operations

### Growing a Boundary

To extend a rough boundary by one row:

**Step 1:** Add new data qubits initialized in $|0\rangle$

$$|\psi\rangle_{\text{old}} \otimes |0\rangle^{\otimes k} \rightarrow |\psi\rangle_{\text{extended}}$$

**Step 2:** Measure new Z stabilizers (now have support on new qubits)

**Step 3:** Update X stabilizers at the boundary to include new qubits

**Step 4:** Correct based on measurement outcomes

### Shrinking a Boundary

To reduce a rough boundary by one row:

**Step 1:** Measure boundary X stabilizers that will be removed

**Step 2:** Remove those stabilizers from the group

**Step 3:** Measure out data qubits in Z basis

**Step 4:** Apply corrections based on measurement outcomes

### Fault-Tolerance During Deformation

**Key principle:** At every step, the code maintains error correction capability.

- Transient reduction in distance at boundaries is acceptable
- Errors during deformation are caught by subsequent stabilizer measurements
- Measurement errors are handled by repetition

$$\boxed{P_{\text{logical error}} \sim \left(\frac{p}{p_{\text{th}}}\right)^{d/2}}$$

where $d$ is the minimum distance during deformation.

---

## 5. Patch Configurations for Lattice Surgery

### Standard Patch Layout

For lattice surgery between two patches A and B:

```
    Patch A              Merge Region           Patch B
┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│              │    │                │    │              │
│    A (d×d)   │ ←→ │   d qubits     │ ←→ │    B (d×d)   │
│              │    │                │    │              │
└──────────────┘    └────────────────┘    └──────────────┘
    Rough                Smooth               Rough
   boundary              region              boundary
```

### Qubit Count and Distance

For a distance-$d$ surface code patch:
- Data qubits: approximately $d^2$
- Ancilla qubits: approximately $d^2$
- Total qubits per patch: $\sim 2d^2$

**Merge region:** Adds $O(d)$ qubits along the boundary.

$$\boxed{N_{\text{logical operation}} = O(d^2) \text{ physical qubits}}$$

### Time Overhead

Each operation (stabilizer measurement round) takes time $\tau_{\text{cycle}}$.

- **Merge:** $d$ cycles (measure joint stabilizers for $d$ rounds)
- **Split:** $d$ cycles (measure individual stabilizers for $d$ rounds)

$$\boxed{t_{\text{logical gate}} = O(d) \times \tau_{\text{cycle}}}$$

---

## 6. Worked Examples

### Example 1: Boundary Stabilizer Analysis

**Problem:** For a distance-3 surface code patch, list all stabilizers at a rough (Z-type) boundary and show how they differ from bulk stabilizers.

**Solution:**

Consider a 3×3 data qubit patch:

```
Qubit layout:
  1 - 2 - 3
  |   |   |
  4 - 5 - 6
  |   |   |
  7 - 8 - 9
```

**Bulk X stabilizer (e.g., centered at position 5):**
$$X_2 X_4 X_5 X_6 X_8$$
Wait, this isn't right for surface code. Let me reconsider.

For a proper surface code:
```
    1 - 2 - 3 - 4 - 5
    |   |   |   |   |
    6 - 7 - 8 - 9 - 10
    |   |   |   |   |
   11 -12 -13 -14 -15
```

**Bulk Z stabilizer** (weight 4):
$$S_Z = Z_2 Z_3 Z_7 Z_8$$

**Boundary Z stabilizer at rough edge** (weight 2):
$$S_Z^{\text{boundary}} = Z_1 Z_6$$

**The difference:**
- Bulk stabilizers have weight 4
- Boundary stabilizers have weight 2
- Lower weight at boundaries doesn't reduce code distance because logical operators must span the full patch

$$\boxed{\text{Boundary stabilizers: weight 2; Bulk stabilizers: weight 4}}$$

---

### Example 2: Logical Operator Path Finding

**Problem:** Given a distance-5 surface code patch with rough boundaries on top and bottom, and smooth boundaries on left and right, trace valid logical Z and X operators.

**Solution:**

**Patch layout:**
```
Smooth │ ● ● ● ● ● │ Smooth
       │ ● ● ● ● ● │
Rough ━━━━━━━━━━━━━━━ Rough (top)
       │ ● ● ● ● ● │
       │ ● ● ● ● ● │
       │ ● ● ● ● ● │
Rough ━━━━━━━━━━━━━━━ Rough (bottom)
       │ ● ● ● ● ● │
Smooth │ ● ● ● ● ● │ Smooth
```

Wait, I need to reconfigure. Standard convention:

```
        Smooth boundary (top)
        ═══════════════════
      R │ ● ● ● ● ● │ R
      o │ ● ● ● ● ● │ o
      u │ ● ● ● ● ● │ u
      g │ ● ● ● ● ● │ g
      h │ ● ● ● ● ● │ h
        ═══════════════════
        Smooth boundary (bottom)
```

**Logical Z:** Vertical path from smooth (top) to smooth (bottom)
$$\bar{Z} = Z_{\text{col}_j} = Z_{1,j} Z_{2,j} Z_{3,j} Z_{4,j} Z_{5,j}$$

Any column works; all are equivalent modulo stabilizers.

**Logical X:** Horizontal path from rough (left) to rough (right)
$$\bar{X} = X_{\text{row}_i} = X_{i,1} X_{i,2} X_{i,3} X_{i,4} X_{i,5}$$

Any row works.

**Verification:** These operators anti-commute:
$$\bar{Z}\bar{X} = Z_{1,j} \cdots Z_{5,j} \cdot X_{i,1} \cdots X_{i,5}$$

At intersection $(i,j)$: $Z_{i,j} X_{i,j} = -X_{i,j} Z_{i,j}$

$$\boxed{\bar{Z}\bar{X} = -\bar{X}\bar{Z} \text{ (verified)}}$$

---

### Example 3: Stabilizer Group During Boundary Extension

**Problem:** Show explicitly how the stabilizer group changes when extending a rough boundary by one qubit.

**Solution:**

**Initial configuration (3 qubits in a row along rough boundary):**
```
Boundary: ● - ● - ●
          1   2   3
```

Boundary Z stabilizers: $Z_1 Z_2$, $Z_2 Z_3$ (weight-2)

**After adding qubit 4:**
```
New boundary: ● - ● - ● - ●
              1   2   3   4
```

**Step 1:** Initialize qubit 4 in $|0\rangle$ (eigenstate of $Z_4$)

**Step 2:** Add new stabilizer $Z_3 Z_4$

**Step 3:** The stabilizer group grows:
$$\mathcal{S} = \langle Z_1 Z_2, Z_2 Z_3 \rangle \rightarrow \mathcal{S}' = \langle Z_1 Z_2, Z_2 Z_3, Z_3 Z_4 \rangle$$

**Logical operator update:**
- Old: $\bar{X} = X_1 X_2 X_3$
- New: $\bar{X}' = X_1 X_2 X_3 X_4$

The logical operator extends to include the new qubit.

$$\boxed{\mathcal{S}' = \mathcal{S} \cup \{Z_3 Z_4\}, \quad \bar{X}' = \bar{X} \cdot X_4}$$

---

## 7. Practice Problems

### Problem Set A: Direct Application

**A1.** For a distance-7 surface code patch, calculate:
- (a) The approximate number of data qubits
- (b) The number of X-type stabilizers
- (c) The number of Z-type stabilizers
- (d) The weight of boundary stabilizers vs. bulk stabilizers

**A2.** Draw the logical X and Z operators for a surface code patch with:
- Rough boundaries on North and South
- Smooth boundaries on East and West

**A3.** If a code deformation operation temporarily reduces the minimum distance from $d=7$ to $d=5$, how does the logical error rate change? Express in terms of physical error rate $p$.

---

### Problem Set B: Intermediate

**B1.** Consider two adjacent surface code patches that will be merged:
- Patch A: distance 5, rough boundary facing right
- Patch B: distance 5, rough boundary facing left

Draw the configuration and identify:
- (a) Which qubits are added during merge
- (b) What new stabilizers are introduced
- (c) How the logical operators connect after merge

**B2.** Prove that the code distance is preserved when growing a smooth boundary by one column, assuming proper stabilizer initialization.

**B3.** Calculate the space-time volume (qubit-cycles) for a merge operation on two distance-5 patches, assuming each stabilizer measurement cycle takes 1 unit of time.

---

### Problem Set C: Challenging

**C1.** Design a code deformation sequence that rotates a surface code patch by 90°, effectively exchanging rough and smooth boundaries. How does this implement logical Hadamard?

**C2.** Analyze the fault-tolerance of boundary extension when the initialization of new qubits has error rate $p_{\text{init}}$. What is the probability that a logical error is introduced?

**C3.** For a distance-$d$ surface code, derive the optimal merge schedule that minimizes total time while maintaining distance at least $d/2$ throughout the operation.

---

## 8. Computational Lab: Surface Code Boundary Visualization

```python
"""
Day 820 Computational Lab: Surface Code Boundary Visualization
Lattice Surgery Fundamentals

This lab provides tools for visualizing surface code patches and their boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

class SurfaceCodePatch:
    """
    Represents a surface code patch with boundaries and logical operators.

    Convention:
    - Rough boundaries (Z-type): top and bottom
    - Smooth boundaries (X-type): left and right

    This gives:
    - Logical Z: horizontal path (left to right)
    - Logical X: vertical path (top to bottom)
    """

    def __init__(self, distance, position=(0, 0), name="Patch"):
        """
        Initialize a surface code patch.

        Parameters:
        -----------
        distance : int
            Code distance (determines size)
        position : tuple
            (x, y) position of bottom-left corner
        name : str
            Identifier for the patch
        """
        self.distance = distance
        self.position = position
        self.name = name

        # Calculate dimensions
        # For distance d: approximately d x d data qubits
        self.width = distance
        self.height = distance

        # Generate qubit positions (data qubits on a grid)
        self.data_qubits = self._generate_data_qubits()
        self.x_stabilizers = self._generate_x_stabilizers()
        self.z_stabilizers = self._generate_z_stabilizers()

    def _generate_data_qubits(self):
        """Generate data qubit positions."""
        qubits = []
        x0, y0 = self.position
        for i in range(self.height):
            for j in range(self.width):
                qubits.append((x0 + j, y0 + i))
        return qubits

    def _generate_x_stabilizers(self):
        """Generate X stabilizer positions (faces)."""
        stabilizers = []
        x0, y0 = self.position
        # X stabilizers are on alternating faces
        for i in range(self.height - 1):
            for j in range(self.width - 1):
                if (i + j) % 2 == 0:  # Checkerboard pattern
                    stabilizers.append((x0 + j + 0.5, y0 + i + 0.5))
        return stabilizers

    def _generate_z_stabilizers(self):
        """Generate Z stabilizer positions (faces)."""
        stabilizers = []
        x0, y0 = self.position
        # Z stabilizers are on alternating faces
        for i in range(self.height - 1):
            for j in range(self.width - 1):
                if (i + j) % 2 == 1:  # Opposite checkerboard
                    stabilizers.append((x0 + j + 0.5, y0 + i + 0.5))
        return stabilizers

    def get_logical_z_path(self):
        """Return positions of qubits in logical Z operator (horizontal)."""
        x0, y0 = self.position
        row = self.height // 2  # Middle row
        return [(x0 + j, y0 + row) for j in range(self.width)]

    def get_logical_x_path(self):
        """Return positions of qubits in logical X operator (vertical)."""
        x0, y0 = self.position
        col = self.width // 2  # Middle column
        return [(x0 + col, y0 + i) for i in range(self.height)]

    def get_boundary(self, side):
        """
        Get boundary qubit positions.

        Parameters:
        -----------
        side : str
            'top', 'bottom' (rough), 'left', 'right' (smooth)
        """
        x0, y0 = self.position
        if side == 'top':
            return [(x0 + j, y0 + self.height - 1) for j in range(self.width)]
        elif side == 'bottom':
            return [(x0 + j, y0) for j in range(self.width)]
        elif side == 'left':
            return [(x0, y0 + i) for i in range(self.height)]
        elif side == 'right':
            return [(x0 + self.width - 1, y0 + i) for i in range(self.height)]


def visualize_surface_code_patch(patch, ax=None, show_logical=True,
                                  show_stabilizers=True, title=None):
    """
    Visualize a surface code patch with boundaries and logical operators.

    Parameters:
    -----------
    patch : SurfaceCodePatch
        The patch to visualize
    ax : matplotlib axis
        Axis to plot on (creates new if None)
    show_logical : bool
        Whether to show logical operator paths
    show_stabilizers : bool
        Whether to show stabilizer positions
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw patch background
    x0, y0 = patch.position
    rect = Rectangle((x0 - 0.3, y0 - 0.3),
                     patch.width - 0.4, patch.height - 0.4,
                     facecolor='lightgray', edgecolor='none', alpha=0.3)
    ax.add_patch(rect)

    # Draw boundaries with colors
    # Rough boundaries (top, bottom) - blue
    for side in ['top', 'bottom']:
        boundary = patch.get_boundary(side)
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        ax.plot(xs, ys, 'b-', linewidth=4, alpha=0.7, label='Rough (Z)' if side == 'top' else '')

    # Smooth boundaries (left, right) - red
    for side in ['left', 'right']:
        boundary = patch.get_boundary(side)
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        ax.plot(xs, ys, 'r-', linewidth=4, alpha=0.7, label='Smooth (X)' if side == 'left' else '')

    # Draw data qubits
    for qx, qy in patch.data_qubits:
        circle = Circle((qx, qy), 0.15, facecolor='black', edgecolor='white')
        ax.add_patch(circle)

    # Draw stabilizers
    if show_stabilizers:
        # X stabilizers (red squares)
        for sx, sy in patch.x_stabilizers:
            rect = Rectangle((sx - 0.2, sy - 0.2), 0.4, 0.4,
                            facecolor='red', edgecolor='darkred', alpha=0.5)
            ax.add_patch(rect)

        # Z stabilizers (blue squares)
        for sz_x, sz_y in patch.z_stabilizers:
            rect = Rectangle((sz_x - 0.2, sz_y - 0.2), 0.4, 0.4,
                            facecolor='blue', edgecolor='darkblue', alpha=0.5)
            ax.add_patch(rect)

    # Draw logical operators
    if show_logical:
        # Logical Z (horizontal) - green
        z_path = patch.get_logical_z_path()
        zx = [p[0] for p in z_path]
        zy = [p[1] for p in z_path]
        ax.plot(zx, zy, 'g-', linewidth=3, marker='o', markersize=8,
               label='Logical Z', alpha=0.8)

        # Logical X (vertical) - purple
        x_path = patch.get_logical_x_path()
        xx = [p[0] for p in x_path]
        xy = [p[1] for p in x_path]
        ax.plot(xx, xy, 'm-', linewidth=3, marker='s', markersize=8,
               label='Logical X', alpha=0.8)

    # Formatting
    ax.set_xlim(x0 - 1, x0 + patch.width)
    ax.set_ylim(y0 - 1, y0 + patch.height)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(title or f"{patch.name} (distance {patch.distance})")
    ax.grid(True, alpha=0.3)

    return ax


def visualize_boundary_types():
    """Create an educational diagram of boundary types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Standard orientation
    patch1 = SurfaceCodePatch(distance=5, position=(0, 0), name="Standard Patch")
    visualize_surface_code_patch(patch1, ax=axes[0],
                                  title="Surface Code Boundaries\n(Rough=Blue, Smooth=Red)")
    axes[0].annotate('Rough boundary\n(Z stabilizers terminate)',
                    xy=(2, 4.3), fontsize=10, ha='center')
    axes[0].annotate('Smooth boundary\n(X stabilizers terminate)',
                    xy=(-0.7, 2), fontsize=10, ha='center', rotation=90)

    # Show logical operators explicitly
    ax = axes[1]
    patch2 = SurfaceCodePatch(distance=5, position=(0, 0), name="Logical Operators")
    visualize_surface_code_patch(patch2, ax=ax, show_stabilizers=False,
                                  title="Logical Operator Paths")

    # Add annotations for operators
    ax.annotate('$\\bar{Z} = Z_1 Z_2 Z_3 Z_4 Z_5$\n(connects smooth boundaries)',
                xy=(2, 2.5), xytext=(4.5, 1), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green')
    ax.annotate('$\\bar{X} = X_1 X_2 X_3 X_4 X_5$\n(connects rough boundaries)',
                xy=(2, 2.5), xytext=(4.5, 3.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='purple'),
                color='purple')

    plt.tight_layout()
    plt.savefig('surface_code_boundaries.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Boundary visualization saved to 'surface_code_boundaries.png'")


def visualize_boundary_extension():
    """
    Visualize the process of extending a surface code boundary.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Initial patch
    patch1 = SurfaceCodePatch(distance=5, position=(0, 0), name="Initial")
    visualize_surface_code_patch(patch1, ax=axes[0], show_stabilizers=False,
                                  title="Step 1: Initial Patch (d=5)")

    # Adding new qubits (shown as hollow)
    ax = axes[1]
    patch2 = SurfaceCodePatch(distance=5, position=(0, 0), name="Growing")
    visualize_surface_code_patch(patch2, ax=ax, show_stabilizers=False, show_logical=False,
                                  title="Step 2: Initialize New Qubits")

    # Add new qubits (column on the right)
    for i in range(5):
        circle = Circle((5, i), 0.15, facecolor='white', edgecolor='green', linewidth=2)
        ax.add_patch(circle)
    ax.annotate('New qubits\ninitialized in $|0\\rangle$',
                xy=(5.5, 2), fontsize=10, color='green')
    ax.set_xlim(-1, 7)

    # Extended patch
    ax = axes[2]
    patch3 = SurfaceCodePatch(distance=6, position=(0, 0), name="Extended")
    visualize_surface_code_patch(patch3, ax=ax, show_stabilizers=False,
                                  title="Step 3: Extended Patch (d=6)")
    ax.annotate('New stabilizers\nmeasured', xy=(4.5, 2.5), fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig('boundary_extension.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Boundary extension visualization saved to 'boundary_extension.png'")


def analyze_patch_resources(distance):
    """
    Calculate and display resource requirements for a surface code patch.

    Parameters:
    -----------
    distance : int
        Code distance
    """
    # Approximate formulas for rotated surface code
    data_qubits = distance ** 2
    ancilla_qubits = (distance - 1) ** 2 + (distance - 1) ** 2  # X and Z ancillas
    total_qubits = data_qubits + ancilla_qubits

    x_stabilizers = (distance - 1) * distance // 2 + (distance - 1) * (distance - 1) // 2
    z_stabilizers = x_stabilizers  # Same count by symmetry

    # Boundary stabilizers (weight 2)
    boundary_stabilizers = 4 * (distance - 1)
    bulk_stabilizers = x_stabilizers + z_stabilizers - boundary_stabilizers

    print(f"\n{'='*50}")
    print(f"Surface Code Patch Analysis (distance d = {distance})")
    print(f"{'='*50}")
    print(f"\nQubit Count:")
    print(f"  Data qubits:    {data_qubits}")
    print(f"  Ancilla qubits: ~{ancilla_qubits}")
    print(f"  Total qubits:   ~{total_qubits}")
    print(f"\nStabilizers:")
    print(f"  X-type: ~{x_stabilizers}")
    print(f"  Z-type: ~{z_stabilizers}")
    print(f"\nBoundary Analysis:")
    print(f"  Boundary stabilizers (weight 2): ~{boundary_stabilizers}")
    print(f"  Bulk stabilizers (weight 4):     ~{bulk_stabilizers}")
    print(f"\nLogical Operators:")
    print(f"  Logical Z weight: {distance}")
    print(f"  Logical X weight: {distance}")
    print(f"\nError Correction Capability:")
    print(f"  Correctable errors: {(distance - 1) // 2}")
    print(f"  Detectable errors:  {distance - 1}")
    print(f"{'='*50}\n")

    return {
        'data_qubits': data_qubits,
        'total_qubits': total_qubits,
        'correctable_errors': (distance - 1) // 2
    }


def main():
    """Run all Day 820 visualizations and analyses."""
    print("Day 820: Lattice Surgery Fundamentals")
    print("="*50)

    # Basic visualization
    print("\n1. Visualizing surface code boundaries...")
    visualize_boundary_types()

    # Boundary extension
    print("\n2. Visualizing boundary extension...")
    visualize_boundary_extension()

    # Resource analysis for various distances
    print("\n3. Resource analysis for different code distances:")
    for d in [3, 5, 7, 11, 15]:
        analyze_patch_resources(d)

    # Create comparison plot
    distances = range(3, 21, 2)
    data_qubits = [d**2 for d in distances]
    total_qubits = [d**2 + 2*(d-1)**2 for d in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, data_qubits, 'bo-', label='Data qubits', markersize=8)
    ax.plot(distances, total_qubits, 'rs-', label='Total qubits', markersize=8)
    ax.set_xlabel('Code Distance (d)', fontsize=12)
    ax.set_ylabel('Number of Qubits', fontsize=12)
    ax.set_title('Surface Code Resource Scaling', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(distances)

    plt.tight_layout()
    plt.savefig('resource_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nResource scaling plot saved to 'resource_scaling.png'")
    print("\nDay 820 Computational Lab Complete!")


if __name__ == "__main__":
    main()
```

---

## 9. Summary

### Key Formulas Table

| Concept | Formula/Expression |
|---------|-------------------|
| Logical Z operator | $\bar{Z} = \prod_{i \in \text{Z-path}} Z_i$ |
| Logical X operator | $\bar{X} = \prod_{i \in \text{X-path}} X_i$ |
| Anti-commutation | $\bar{Z}\bar{X} = -\bar{X}\bar{Z}$ |
| Data qubits (distance $d$) | $N_{\text{data}} \approx d^2$ |
| Total qubits per patch | $N_{\text{total}} \approx 2d^2$ |
| Correctable errors | $t = \lfloor(d-1)/2\rfloor$ |
| Boundary stabilizer weight | 2 |
| Bulk stabilizer weight | 4 |
| Logical error probability | $P_L \sim (p/p_{th})^{d/2}$ |

### Key Takeaways

1. **Lattice surgery** implements logical gates through code deformation, not transversal operations
2. **Boundary types** (rough/smooth) determine where logical operators can terminate
3. **Code deformation** involves adding/removing qubits and stabilizers while preserving logical information
4. **Local operations only** are needed - no long-range qubit connectivity required
5. **Time scaling** for operations is $O(d)$ syndrome measurement cycles

---

## 10. Daily Checklist

- [ ] I can distinguish rough (Z-type) from smooth (X-type) boundaries
- [ ] I can trace logical X and Z operators on a surface code patch
- [ ] I understand how stabilizers differ at boundaries vs. bulk
- [ ] I can describe the code deformation process for boundary extension
- [ ] I can calculate resource requirements for different code distances
- [ ] I ran the computational lab and visualized boundary configurations

---

## 11. Preview: Day 821

Tomorrow we dive into **Merge Operations**, the core primitive of lattice surgery:

- XX merge: Joint X-type measurement connecting two patches
- ZZ merge: Joint Z-type measurement for entanglement creation
- Measurement outcomes and Pauli corrections
- Fault-tolerance during merge operations

We will see how merge operations enable non-local entanglement using only local operations - the key insight that makes lattice surgery so powerful for near-term quantum hardware.

---

*"The surface code patch is not just a passive error-correcting container - it is a dynamic structure whose boundaries can be sculpted to perform quantum logic."*
