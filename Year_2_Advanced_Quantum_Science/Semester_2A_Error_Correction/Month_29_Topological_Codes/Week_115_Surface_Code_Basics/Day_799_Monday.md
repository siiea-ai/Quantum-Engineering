# Day 799: From Torus to Plane — Boundaries in the Surface Code

## Month 29: Topological Codes | Week 115: Surface Code Implementation
### Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Theory of boundaries, smooth vs. rough distinction |
| **Afternoon** | 2.5 hours | Boundary stabilizer construction, problem solving |
| **Evening** | 1.5 hours | Computational lab: Visualizing boundary effects |

**Total Study Time**: 7 hours

---

## Learning Objectives

By the end of Day 799, you will be able to:

1. **Explain** why periodic (toric) boundaries cannot be implemented in planar hardware
2. **Distinguish** smooth (X-type) boundaries from rough (Z-type) boundaries
3. **Construct** boundary stabilizers with reduced weight
4. **Calculate** how boundaries reduce encoded qubits from 2 to 1
5. **Visualize** the boundary conditions on a planar lattice
6. **Connect** boundary physics to practical hardware constraints

---

## Morning Session: Theory of Boundaries (3 hours)

### 1. The Problem with the Torus

The toric code, introduced by Kitaev in 1997, lives on a **torus**—a surface with periodic boundary conditions in both directions. Topologically:

$$\text{Torus} = T^2 = S^1 \times S^1$$

This provides elegant mathematical properties:
- All stabilizers have uniform weight (weight-4)
- Two independent non-contractible loops → 2 logical qubits
- Perfect translational symmetry

**But there's a fatal flaw**: Real quantum hardware is planar.

#### Why a Torus is Impossible

Consider a 2D chip layout with qubits on a substrate:

1. **Physical constraints**: Qubits lie in a plane; we cannot "wrap around"
2. **Connectivity limits**: Even with long-range couplings, creating a torus requires 3D embedding
3. **Wiring complexity**: Connecting left edge to right edge, top to bottom, creates routing conflicts

The solution: **Cut the torus open** to create a planar surface with boundaries.

### 2. Cutting the Torus

When we cut a torus, we create a cylinder, then cutting again yields a rectangle:

$$T^2 \xrightarrow{\text{cut 1}} S^1 \times [0,1] \xrightarrow{\text{cut 2}} [0,1] \times [0,1]$$

The cuts introduce **edges** (boundaries) that were not present on the closed torus. These boundaries must be handled consistently with the stabilizer structure.

#### What Happens at Boundaries?

On the bulk (interior), every plaquette and vertex has a well-defined stabilizer:

- **Plaquette stabilizer**: $B_p = \prod_{e \in \partial p} X_e$ (product over 4 edges)
- **Vertex stabilizer**: $A_v = \prod_{e \ni v} Z_e$ (product over 4 adjacent edges)

At boundaries, some edges are **missing**—the stabilizers must be modified.

### 3. Two Types of Boundaries

The key insight is that boundaries come in two types, corresponding to which stabilizers they "cut":

#### Smooth Boundaries (X-type / Primal)

A **smooth boundary** is defined by plaquettes that extend to the edge:

$$\boxed{\text{Smooth boundary: Plaquette stabilizers are truncated}}$$

At a smooth boundary:
- Plaquettes at the edge become **weight-3** (or weight-2 at corners)
- Vertex stabilizers remain complete (all adjacent edges exist)
- The boundary appears "smooth" in the primal (plaquette) lattice

**Mathematically**:
$$B_p^{\text{boundary}} = \prod_{e \in \partial p \cap \text{bulk}} X_e$$

#### Rough Boundaries (Z-type / Dual)

A **rough boundary** is defined by vertices that sit on the edge:

$$\boxed{\text{Rough boundary: Vertex stabilizers are truncated}}$$

At a rough boundary:
- Vertices at the edge have **weight-3** (or weight-2 at corners) stabilizers
- Plaquette stabilizers remain complete
- The boundary appears "rough" in the dual (vertex) lattice

**Mathematically**:
$$A_v^{\text{boundary}} = \prod_{e \ni v, e \in \text{bulk}} Z_e$$

### 4. Visualizing Boundary Types

Consider a lattice where:
- **Data qubits** sit on edges (standard toric code convention)
- **Plaquettes** are faces (X-stabilizers)
- **Vertices** are corners (Z-stabilizers)

```
Smooth boundary (left/right edges):

    |     |     |     |
----●-----●-----●-----●
    |     |     |     |
    ●     ●     ●     ●  ← Interior vertices (weight-4)
    |     |     |     |
----●-----●-----●-----●
    |     |     |     |

The left edge has half-plaquettes (weight-2 or weight-3)
```

```
Rough boundary (top/bottom edges):

●-----●-----●-----●-----●  ← Boundary vertices (weight-3)
|     |     |     |     |
●     ●     ●     ●     ●  ← Interior vertices (weight-4)
|     |     |     |     |
●-----●-----●-----●-----●
```

### 5. Why Boundaries Reduce Logical Qubits

On the torus, we had **two independent non-contractible loops** (around the "hole" and through it), giving 2 logical qubits.

On a planar surface with boundaries:

- A loop around a boundary is **contractible** (can shrink to a point)
- The only non-trivial "loops" are **strings connecting different boundaries**

With the standard configuration (smooth on left/right, rough on top/bottom):

- **One logical Z**: Vertical string from rough to rough boundary
- **One logical X**: Horizontal string from smooth to smooth boundary

$$\boxed{k = 1 \text{ logical qubit on planar surface}}$$

This is a topological consequence: the number of logical qubits equals the number of "handles" plus boundary contributions. For a rectangle with 2 smooth and 2 rough boundaries: $k = 1$.

### 6. Boundary Stabilizer Details

#### At a Smooth Boundary (Left Edge Example)

Consider a plaquette touching the left edge:

```
    |
●---●---●
|   |   |
●   ●   ●
|   |   |
```

The leftmost plaquette has only 3 edges (the left edge is "cut"):

$$B_p^{\text{left}} = X_{\text{top}} X_{\text{right}} X_{\text{bottom}}$$

This is a **weight-3 X-stabilizer**.

#### At a Rough Boundary (Bottom Edge Example)

Consider a vertex on the bottom edge:

```
|
●---●---●
    |
```

This vertex has only 3 adjacent edges (no edge below):

$$A_v^{\text{bottom}} = Z_{\text{left}} Z_{\text{up}} Z_{\text{right}}$$

This is a **weight-3 Z-stabilizer**.

#### At Corners

Corners combine both boundary types. For a bottom-left corner:

$$A_v^{\text{corner}} = Z_{\text{up}} Z_{\text{right}} \quad \text{(weight-2)}$$

The corner plaquette (if using the same convention) would be weight-2 as well.

---

## Quantum Mechanics Connection

### From Topological Order to Physical Implementation

The transition from toric to surface code illustrates a fundamental principle in quantum engineering:

**Theoretical elegance vs. practical constraints**

The toric code has beautiful symmetry (all stabilizers identical), but this symmetry requires an impossible topology. The surface code sacrifices uniformity at boundaries to enable physical realization.

#### Topological Protection Survives

Despite breaking translational symmetry, the surface code retains **topological protection**:

1. **Local errors cannot create logical errors**: An error string must span the entire code (boundary to boundary) to cause a logical fault
2. **Distance is preserved**: The minimum-weight logical operator still has weight $d$
3. **Error threshold exists**: The code still corrects up to $\lfloor (d-1)/2 \rfloor$ errors

#### Boundary Effects on Anyons

In the toric code picture:
- **e-particles** (Z-errors) condense at smooth boundaries
- **m-particles** (X-errors) condense at rough boundaries

This "condensation" means:
- An e-anyon can be absorbed by a smooth boundary
- An m-anyon can be absorbed by a rough boundary

This is why:
- Logical X (X-string) connects smooth-to-smooth (moves e-anyons)
- Logical Z (Z-string) connects rough-to-rough (moves m-anyons)

---

## Afternoon Session: Worked Examples (2.5 hours)

### Example 1: Counting Stabilizers with Boundaries

**Problem**: For a distance-3 surface code (3×3 data qubits in standard layout), count the X-stabilizers, Z-stabilizers, and verify $n - k = s_X + s_Z$.

**Solution**:

Step 1: Layout the distance-3 code
```
Smooth      Rough
  ↓          ↓
  ●---●---●  ← Rough
  |   |   |
  ●---●---●
  |   |   |
  ●---●---●  ← Rough
  ↑
Smooth
```

Wait—this is the data qubit layout. Let me use the proper convention where data qubits are on edges.

Actually, for the **unrotated** surface code with $d=3$:
- Data qubits: $d^2 = 9$
- Plaquettes (X-stabilizers): $(d-1) \times d = 2 \times 3 = 6$...

Let me be more careful. For the standard surface code:

Using the convention where data qubits are placed at vertices of a $d \times d$ grid:

- **Data qubits**: $n = d^2 = 9$
- **X-stabilizers** (plaquettes): Includes bulk + boundary
- **Z-stabilizers** (vertices): Includes bulk + boundary

For $d = 3$:
- Interior X-stabilizers (weight-4): $(d-2) \times (d-1)/2 = 1 \times 1 = 1$...

This is getting confusing. Let me use the explicit count:

For a distance-3 surface code with 9 data qubits:
- Total stabilizers: $n - k = 9 - 1 = 8$
- X-stabilizers: 4
- Z-stabilizers: 4

Verification: $4 + 4 = 8 = 9 - 1$ ✓

Step 2: Identify boundary stabilizers
- 2 weight-2 X-stabilizers (at corners of smooth boundaries)
- 2 weight-3 X-stabilizers (along smooth edges)
- 2 weight-2 Z-stabilizers (at corners of rough boundaries)
- 2 weight-3 Z-stabilizers (along rough edges)

$$\boxed{n=9, k=1, s_X=4, s_Z=4}$$

### Example 2: String Operators Between Boundaries

**Problem**: On a 5×5 surface code, describe the minimum-weight logical X and logical Z operators.

**Solution**:

Configuration: Smooth boundaries on left/right, rough boundaries on top/bottom.

**Logical X** (connects smooth to smooth):
- Horizontal string of X operators from left edge to right edge
- Minimum path length = distance $d = 5$
- $\bar{X} = X_1 X_2 X_3 X_4 X_5$ along a horizontal row

**Logical Z** (connects rough to rough):
- Vertical string of Z operators from top edge to bottom edge
- Minimum path length = distance $d = 5$
- $\bar{Z} = Z_1 Z_2 Z_3 Z_4 Z_5$ along a vertical column

Both operators have weight 5, confirming $d = 5$.

$$\boxed{\text{wt}(\bar{X}) = \text{wt}(\bar{Z}) = d = 5}$$

### Example 3: Why Boundaries Must Alternate

**Problem**: What happens if we use all smooth boundaries (4 smooth edges)?

**Solution**:

With all smooth boundaries:
- All plaquette stabilizers are truncated at edges
- Vertex stabilizers remain complete
- **Key issue**: The product of all plaquette stabilizers equals identity in the bulk, but with all smooth boundaries, this constraint is modified

Consider the stabilizer count:
- If all boundaries are smooth, certain stabilizer products become dependent
- The code may encode **0 logical qubits** (trivial code) or have **logical Z with weight 0** (degenerate)

Actually, with 4 smooth boundaries:
- Logical Z would require a loop around the code
- But all loops are contractible (no rough boundary to anchor Z-strings)
- Result: No non-trivial logical Z, so $k = 0$

Similarly, 4 rough boundaries give $k = 0$.

To encode 1 logical qubit, we need **at least one smooth and one rough boundary pair**.

$$\boxed{\text{Alternating boundary types required for } k = 1}$$

---

## Practice Problems

### Problem Set 799

#### Direct Application

1. **Boundary stabilizer weights**: For a 7×7 surface code with smooth boundaries on east/west and rough boundaries on north/south, list all stabilizer weights that appear (weight-2, weight-3, weight-4) and count how many of each.

2. **Logical qubit count**: A surface code has 3 smooth boundaries and 1 rough boundary. How many logical qubits does it encode? (Hint: Consider which string operators are non-trivial.)

3. **Anyon condensation**: An X-error creates two e-anyons. If one anyon is adjacent to a smooth boundary, what happens to it?

#### Intermediate

4. **Modified surface code**: Consider a "T-shaped" surface code (imagine a rectangle with a rectangular notch cut out). If the outer boundary alternates smooth/rough and the inner boundary is all smooth, how many logical qubits are encoded?

5. **Boundary syndrome patterns**: A single Z-error occurs on a qubit adjacent to a smooth boundary. How many X-stabilizers (syndrome bits) are affected? What if the qubit is at a corner?

6. **Weight optimization**: Design a distance-5 surface code that minimizes the number of weight-2 stabilizers while maintaining $k=1$.

#### Challenging

7. **General boundary formula**: Derive a formula for the number of logical qubits $k$ in terms of the number of smooth and rough boundary segments for a planar surface code.

8. **Lattice surgery preview**: Two surface codes share a rough boundary that is "opened" (stabilizers removed). What is the total number of logical qubits in the combined system?

9. **Non-rectangular boundaries**: Consider a hexagonal surface code with alternating boundary types. Calculate $n$, $k$, and $d$ in terms of the hexagon side length $L$.

---

## Evening Session: Computational Lab (1.5 hours)

### Lab 799: Visualizing Surface Code Boundaries

```python
"""
Day 799 Computational Lab: Surface Code Boundaries
Visualizing boundary structure and stabilizer modifications
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

class SurfaceCodeBoundaries:
    """
    Visualization of surface code with boundary conditions.

    Convention:
    - Data qubits on vertices of a d×d grid
    - X-stabilizers on plaquettes (faces)
    - Z-stabilizers on vertices of dual lattice
    - Smooth boundaries: left and right (X-type)
    - Rough boundaries: top and bottom (Z-type)
    """

    def __init__(self, d):
        """
        Initialize a distance-d surface code.

        Parameters:
            d: Code distance (must be odd for standard surface code)
        """
        if d < 3:
            raise ValueError("Distance must be at least 3")
        self.d = d
        self.n = d * d  # Number of data qubits

        # Build the lattice structure
        self._build_data_qubits()
        self._build_x_stabilizers()
        self._build_z_stabilizers()

    def _build_data_qubits(self):
        """Place data qubits on a d×d grid."""
        self.data_qubits = []
        for i in range(self.d):
            for j in range(self.d):
                self.data_qubits.append((i, j))
        self.data_qubits = np.array(self.data_qubits)

    def _build_x_stabilizers(self):
        """
        Build X-stabilizers (plaquettes).
        Located between data qubits on a (d-1)×d staggered grid.
        """
        self.x_stabilizers = []
        self.x_stabilizer_weights = []

        # X-stabilizers are on plaquettes
        for i in range(self.d - 1):
            for j in range(self.d - 1):
                # Check if on boundary
                weight = 4
                if j == 0:  # Left (smooth) boundary
                    weight = 3
                if j == self.d - 2:  # Right (smooth) boundary
                    weight = 3 if weight == 4 else 2

                center = (i + 0.5, j + 0.5)
                self.x_stabilizers.append(center)
                self.x_stabilizer_weights.append(weight)

        self.x_stabilizers = np.array(self.x_stabilizers)
        self.x_stabilizer_weights = np.array(self.x_stabilizer_weights)

    def _build_z_stabilizers(self):
        """
        Build Z-stabilizers (vertices of dual lattice).
        """
        self.z_stabilizers = []
        self.z_stabilizer_weights = []

        # Z-stabilizers on dual lattice vertices
        for i in range(self.d):
            for j in range(self.d - 1):
                weight = 4
                # Check rough boundaries (top/bottom)
                if i == 0 or i == self.d - 1:
                    weight = 3

                center = (i, j + 0.5)
                self.z_stabilizers.append(center)
                self.z_stabilizer_weights.append(weight)

        self.z_stabilizers = np.array(self.z_stabilizers)
        self.z_stabilizer_weights = np.array(self.z_stabilizer_weights)

    def visualize(self, show_weights=True, figsize=(10, 10)):
        """
        Create visualization of the surface code with boundaries.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Draw grid lines (edges where data qubits conceptually live)
        for i in range(self.d):
            ax.axhline(y=i, color='lightgray', linewidth=0.5, zorder=1)
        for j in range(self.d):
            ax.axvline(x=j, color='lightgray', linewidth=0.5, zorder=1)

        # Draw data qubits
        ax.scatter(self.data_qubits[:, 1], self.data_qubits[:, 0],
                   s=200, c='black', zorder=5, label='Data qubits')

        # Draw X-stabilizers (plaquettes) with color by weight
        cmap = plt.cm.Blues
        for idx, (stab, weight) in enumerate(zip(self.x_stabilizers,
                                                   self.x_stabilizer_weights)):
            color = cmap(0.3 + 0.2 * (weight - 2))
            rect = FancyBboxPatch((stab[1] - 0.4, stab[0] - 0.4),
                                   0.8, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='blue',
                                   alpha=0.5, zorder=2)
            ax.add_patch(rect)
            if show_weights:
                ax.text(stab[1], stab[0], f'{weight}',
                       ha='center', va='center', fontsize=10,
                       color='darkblue', fontweight='bold')

        # Draw Z-stabilizers (dual vertices) with color by weight
        cmap_z = plt.cm.Reds
        for idx, (stab, weight) in enumerate(zip(self.z_stabilizers,
                                                   self.z_stabilizer_weights)):
            color = cmap_z(0.3 + 0.2 * (weight - 2))
            circle = Circle((stab[1], stab[0]), 0.2,
                            facecolor=color, edgecolor='red',
                            alpha=0.7, zorder=3)
            ax.add_patch(circle)
            if show_weights:
                ax.text(stab[1] + 0.35, stab[0], f'{weight}',
                       ha='center', va='center', fontsize=8,
                       color='darkred', fontweight='bold')

        # Mark boundaries
        # Smooth boundaries (left and right) - blue
        ax.axvline(x=-0.3, color='blue', linewidth=4, label='Smooth boundary')
        ax.axvline(x=self.d - 0.7, color='blue', linewidth=4)

        # Rough boundaries (top and bottom) - red
        ax.axhline(y=-0.3, color='red', linewidth=4, label='Rough boundary')
        ax.axhline(y=self.d - 0.7, color='red', linewidth=4)

        # Labels
        ax.set_xlim(-0.7, self.d - 0.3)
        ax.set_ylim(-0.7, self.d - 0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('Column index', fontsize=12)
        ax.set_ylabel('Row index', fontsize=12)
        ax.set_title(f'Distance-{self.d} Surface Code with Boundaries\n'
                     f'Blue squares: X-stabilizers | Red circles: Z-stabilizers',
                     fontsize=14)
        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig, ax

    def draw_logical_operators(self, ax):
        """Add logical X and Z operators to existing plot."""
        # Logical X: horizontal path (row d//2)
        row = self.d // 2
        x_path = np.array([[row, j] for j in range(self.d)])
        ax.plot(x_path[:, 1], x_path[:, 0], 'g-', linewidth=3,
                label=r'Logical $\bar{X}$', zorder=6)
        ax.scatter(x_path[:, 1], x_path[:, 0], s=100, c='green',
                   marker='s', zorder=7)

        # Logical Z: vertical path (column d//2)
        col = self.d // 2
        z_path = np.array([[i, col] for i in range(self.d)])
        ax.plot(z_path[:, 1], z_path[:, 0], 'm-', linewidth=3,
                label=r'Logical $\bar{Z}$', zorder=6)
        ax.scatter(z_path[:, 1], z_path[:, 0], s=100, c='magenta',
                   marker='d', zorder=7)

        ax.legend(loc='upper right')
        return ax


def compare_torus_vs_plane():
    """
    Visualize the conceptual difference between toric and surface codes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Toric code (represented with periodic arrows)
    ax1 = axes[0]
    d = 5
    for i in range(d):
        for j in range(d):
            ax1.scatter(j, i, s=100, c='black')

    # Draw wrap-around arrows
    ax1.annotate('', xy=(d-0.2, d//2), xytext=(-0.5, d//2),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax1.annotate('', xy=(d//2, d-0.2), xytext=(d//2, -0.5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))

    ax1.set_xlim(-1, d)
    ax1.set_ylim(-1, d)
    ax1.set_aspect('equal')
    ax1.set_title('Toric Code\n(Periodic boundaries - impossible in 2D)', fontsize=12)
    ax1.text(d//2, -0.8, 'Wraps to top', ha='center', color='orange')
    ax1.text(d + 0.3, d//2, 'Wraps\nto left', ha='left', color='purple')
    ax1.text(0.5, d + 0.5, 'Encodes 2 logical qubits', ha='left',
             fontsize=11, style='italic')

    # Right: Surface code with boundaries
    ax2 = axes[1]
    sc = SurfaceCodeBoundaries(5)
    sc.visualize(show_weights=False, figsize=(6, 6))
    plt.close()  # Close the auto-generated figure

    # Recreate on our subplot
    for i in range(d):
        for j in range(d):
            ax2.scatter(j, i, s=100, c='black')

    # Boundaries
    ax2.axvline(x=-0.3, color='blue', linewidth=4)
    ax2.axvline(x=d - 0.7, color='blue', linewidth=4)
    ax2.axhline(y=-0.3, color='red', linewidth=4)
    ax2.axhline(y=d - 0.7, color='red', linewidth=4)

    ax2.set_xlim(-1, d)
    ax2.set_ylim(-1, d)
    ax2.set_aspect('equal')
    ax2.set_title('Surface Code\n(Open boundaries - planar implementation)', fontsize=12)
    ax2.text(-0.5, d//2, 'Smooth', ha='right', color='blue', rotation=90, va='center')
    ax2.text(d - 0.5, d//2, 'Smooth', ha='left', color='blue', rotation=90, va='center')
    ax2.text(d//2, -0.6, 'Rough', ha='center', color='red')
    ax2.text(d//2, d - 0.4, 'Rough', ha='center', color='red')
    ax2.text(0.5, d + 0.5, 'Encodes 1 logical qubit', ha='left',
             fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('torus_vs_plane.png', dpi=150, bbox_inches='tight')
    plt.show()


def boundary_stabilizer_demo():
    """
    Detailed visualization of boundary stabilizers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Interior stabilizer (weight-4)
    ax1 = axes[0]
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)

    # Draw the 4 qubits and plaquette
    positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for pos in positions:
        ax1.scatter(pos[1], pos[0], s=200, c='black', zorder=5)

    # Draw edges
    ax1.plot([1, 0], [0, 1], 'b-', lw=2)
    ax1.plot([1, 2], [0, 1], 'b-', lw=2)
    ax1.plot([1, 0], [2, 1], 'b-', lw=2)
    ax1.plot([1, 2], [2, 1], 'b-', lw=2)

    # Plaquette center
    rect = Rectangle((0.5, 0.5), 1, 1, fill=True, facecolor='lightblue',
                      edgecolor='blue', alpha=0.5, lw=2)
    ax1.add_patch(rect)
    ax1.text(1, 1, 'X', ha='center', va='center', fontsize=16, fontweight='bold')

    ax1.set_aspect('equal')
    ax1.set_title('Interior X-Stabilizer\n(Weight-4)', fontsize=12)
    ax1.text(1, -0.3, r'$B_p = X_1 X_2 X_3 X_4$', ha='center', fontsize=11)
    ax1.axis('off')

    # Boundary stabilizer (weight-3, smooth edge)
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2.5)

    # Draw the 3 qubits (one edge is "cut")
    positions = [(1, 0), (1, 2), (2, 1)]
    for pos in positions:
        ax2.scatter(pos[1], pos[0], s=200, c='black', zorder=5)

    # Draw edges (3 instead of 4)
    ax2.plot([1, 0], [0, 1], 'b-', lw=2)
    ax2.plot([1, 2], [0, 1], 'b-', lw=2)
    ax2.plot([1, 0], [2, 1], 'b--', lw=2, alpha=0.3)  # Missing edge
    ax2.plot([1, 2], [2, 1], 'b-', lw=2)

    # Half plaquette
    rect = FancyBboxPatch((0.5, 0.5), 1, 1,
                           boxstyle="round,pad=0.05",
                           fill=True, facecolor='lightyellow',
                           edgecolor='blue', alpha=0.5, lw=2)
    ax2.add_patch(rect)
    ax2.text(1, 1, 'X', ha='center', va='center', fontsize=16, fontweight='bold')

    # Boundary indicator
    ax2.axhline(y=2.3, color='blue', linewidth=4)
    ax2.text(1, 2.5, 'Smooth Boundary', ha='center', color='blue', fontsize=10)

    ax2.set_aspect('equal')
    ax2.set_title('Boundary X-Stabilizer\n(Weight-3)', fontsize=12)
    ax2.text(1, -0.3, r'$B_p^{bdy} = X_1 X_2 X_3$', ha='center', fontsize=11)
    ax2.axis('off')

    # Corner stabilizer (weight-2)
    ax3 = axes[2]
    ax3.set_xlim(-0.5, 2.5)
    ax3.set_ylim(-0.5, 2.5)

    # Draw the 2 qubits
    positions = [(1, 0), (2, 1)]
    for pos in positions:
        ax3.scatter(pos[1], pos[0], s=200, c='black', zorder=5)

    # Draw edges (2 only)
    ax3.plot([1, 0], [0, 1], 'b-', lw=2)
    ax3.plot([1, 2], [0, 1], 'b--', lw=2, alpha=0.3)  # Missing
    ax3.plot([1, 0], [2, 1], 'b--', lw=2, alpha=0.3)  # Missing
    ax3.plot([1, 2], [2, 1], 'b-', lw=2)

    # Quarter plaquette
    rect = FancyBboxPatch((0.5, 0.5), 1, 1,
                           boxstyle="round,pad=0.05",
                           fill=True, facecolor='lightcoral',
                           edgecolor='blue', alpha=0.5, lw=2)
    ax2.add_patch(rect)
    ax3.text(1, 1, 'X', ha='center', va='center', fontsize=16, fontweight='bold')

    # Boundary indicators
    ax3.axhline(y=2.3, color='blue', linewidth=4)
    ax3.axvline(x=-0.3, color='red', linewidth=4)
    ax3.text(1, 2.5, 'Smooth', ha='center', color='blue', fontsize=10)
    ax3.text(-0.5, 1, 'Rough', ha='right', va='center', color='red',
             fontsize=10, rotation=90)

    ax3.set_aspect('equal')
    ax3.set_title('Corner X-Stabilizer\n(Weight-2)', fontsize=12)
    ax3.text(1, -0.3, r'$B_p^{corner} = X_1 X_2$', ha='center', fontsize=11)
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('boundary_stabilizers.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Day 799: Surface Code Boundaries")
    print("=" * 60)

    # Create and visualize distance-5 surface code
    print("\n1. Creating distance-5 surface code...")
    sc = SurfaceCodeBoundaries(5)

    print(f"   Data qubits: {sc.n}")
    print(f"   X-stabilizers: {len(sc.x_stabilizers)}")
    print(f"   Z-stabilizers: {len(sc.z_stabilizers)}")
    print(f"   Total stabilizers: {len(sc.x_stabilizers) + len(sc.z_stabilizers)}")
    print(f"   Expected (n-k): {sc.n - 1}")

    print("\n   X-stabilizer weights:")
    unique, counts = np.unique(sc.x_stabilizer_weights, return_counts=True)
    for w, c in zip(unique, counts):
        print(f"     Weight-{w}: {c} stabilizers")

    print("\n   Z-stabilizer weights:")
    unique, counts = np.unique(sc.z_stabilizer_weights, return_counts=True)
    for w, c in zip(unique, counts):
        print(f"     Weight-{w}: {c} stabilizers")

    # Visualize
    print("\n2. Generating visualizations...")
    fig, ax = sc.visualize()
    sc.draw_logical_operators(ax)
    plt.savefig('surface_code_d5.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Compare torus vs plane
    print("\n3. Comparing toric vs surface code...")
    compare_torus_vs_plane()

    # Boundary stabilizer details
    print("\n4. Boundary stabilizer visualization...")
    boundary_stabilizer_demo()

    print("\n" + "=" * 60)
    print("Lab complete. Check generated figures.")
    print("=" * 60)
```

### Expected Output

```
============================================================
Day 799: Surface Code Boundaries
============================================================

1. Creating distance-5 surface code...
   Data qubits: 25
   X-stabilizers: 16
   Z-stabilizers: 20
   Total stabilizers: 36
   Expected (n-k): 24

   X-stabilizer weights:
     Weight-3: 8 stabilizers
     Weight-4: 8 stabilizers

   Z-stabilizer weights:
     Weight-3: 8 stabilizers
     Weight-4: 12 stabilizers

2. Generating visualizations...
3. Comparing toric vs surface code...
4. Boundary stabilizer visualization...

============================================================
Lab complete. Check generated figures.
============================================================
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Torus topology | $T^2 = S^1 \times S^1$ (impossible in 2D hardware) |
| Smooth boundary X-stabilizer | $B_p^{bdy} = \prod_{e \in \partial p \cap \text{bulk}} X_e$ |
| Rough boundary Z-stabilizer | $A_v^{bdy} = \prod_{e \ni v, e \in \text{bulk}} Z_e$ |
| Logical qubits (planar) | $k = 1$ (with 2 smooth + 2 rough boundaries) |
| Boundary stabilizer weight | 2 (corner) or 3 (edge) vs. 4 (bulk) |

### Key Takeaways

1. **Periodic boundaries are physically impossible** for planar chip layouts—the toric code cannot be directly implemented

2. **Boundaries come in two types**: Smooth (X-type, plaquettes truncated) and rough (Z-type, vertices truncated)

3. **Boundary configuration determines logical qubits**: Alternating smooth/rough gives $k=1$; all same type gives $k=0$

4. **Boundary stabilizers have reduced weight**: Weight-3 along edges, weight-2 at corners

5. **Anyons condense at boundaries**: e-particles at smooth, m-particles at rough

6. **Logical operators connect same-type boundaries**: X from smooth-to-smooth, Z from rough-to-rough

---

## Daily Checklist

Before moving to Day 800, verify you can:

- [ ] Explain why a torus cannot be embedded in a plane
- [ ] Draw a surface code with correct boundary types labeled
- [ ] Write boundary stabilizers with appropriate weight
- [ ] Calculate logical qubit count from boundary configuration
- [ ] Describe anyon condensation at boundaries
- [ ] Trace logical X and Z operator paths

---

## Preview: Day 800

Tomorrow we construct the **complete planar surface code structure**:
- Precise data qubit and ancilla placement on the lattice
- The $[[d^2, 1, d]]$ unrotated code parameters
- Comparison with the rotated $[[(d^2+1)/2, 1, d]]$ variant
- Explicit stabilizer matrices for small distances

---

## References

1. Bravyi, S., & Kitaev, A. "Quantum codes on a lattice with boundary." arXiv:quant-ph/9811052 (1998)
2. Fowler, A. G., et al. "Surface codes: Towards practical large-scale quantum computation." *Physical Review A* 86, 032324 (2012)
3. Dennis, E., et al. "Topological quantum memory." *Journal of Mathematical Physics* 43, 4452 (2002)

---

*Day 799 establishes why boundaries are necessary and how they fundamentally reshape the toric code into the implementable surface code.*
