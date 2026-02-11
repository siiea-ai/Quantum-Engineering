# Day 803: Defects and Holes in the Surface Code

## Month 29: Topological Codes | Week 115: Surface Code Implementation
### Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Twist defects, genons, and their properties |
| **Afternoon** | 2.5 hours | Holes as logical qubits, defect braiding |
| **Evening** | 1.5 hours | Computational lab: Defect manipulation |

**Total Study Time**: 7 hours

---

## Learning Objectives

By the end of Day 803, you will be able to:

1. **Define** twist defects (genons) and explain their topological origin
2. **Construct** holes in the surface code lattice as logical qubit storage
3. **Describe** how defect braiding implements logical gates
4. **Compare** defect-based operations with lattice surgery
5. **Calculate** the resource overhead of defect encodings
6. **Analyze** the gate sets achievable through defect manipulation

---

## Morning Session: Twist Defects and Genons (3 hours)

### 1. Beyond Standard Boundaries

So far, we've studied surface codes with smooth and rough boundaries—these are "trivial" boundary conditions that simply terminate the lattice. But richer physics emerges when we introduce **defects** within the bulk of the code.

#### What is a Defect?

A **defect** is a local modification of the stabilizer structure that:
- Breaks translational symmetry at specific points
- Can carry topological charge
- Enables new logical operations through braiding

The two main types we'll study:
1. **Twist defects** (genons): Points where boundary types "twist"
2. **Holes**: Removed regions that create new boundaries

### 2. Twist Defects (Genons)

A **twist defect** occurs at a point where the lattice structure changes in a non-trivial way—specifically, where a smooth boundary transitions to a rough boundary (or vice versa) at a single point.

#### Construction

Consider a surface code with a "seam" running through it:

```
        Smooth               Rough
           |                   |
    ───────●───────────●───────────
           |           ↑
         Twist      Normal
        defect      boundary
```

At the twist point:
- X-stabilizers on one side meet Z-stabilizers on the other
- The defect carries **half** a unit of anyonic charge

#### The "Genon" Property

Twist defects are called **genons** because:
- They are neither e-type nor m-type anyons
- They have **non-Abelian** exchange statistics
- Braiding two genons can perform non-trivial operations

Mathematically:
$$\sigma_{\text{genon}}^2 = e \cdot m = \epsilon \quad \text{(the fermion)}$$

#### Anyonic Condensation at Twist Defects

At a twist defect:
- An e-anyon approaching from the smooth side becomes an m-anyon on the other side
- This "transmutation" is the defining property

$$e \xrightarrow{\text{twist}} m, \quad m \xrightarrow{\text{twist}} e$$

### 3. Creating Twist Defects in Hardware

To implement a twist defect:

1. **Modify stabilizers** at the defect location:
   - Some plaquette stabilizers become vertex-type
   - Some vertex stabilizers become plaquette-type

2. **Boundary condition mixing**:
   ```
   Before defect:
   ● Z ● Z ● Z ● Z ●
   X   X   X   X
   ● Z ● Z ● Z ● Z ●

   After introducing twist at center:
   ● Z ● Z ● | Z ● Z ●
   X   X     |     X   X
   ● Z ● Z ● | X ● X ●
             |
           twist
   ```

3. **Stabilizer weight changes**: The defect typically creates a weight-3 stabilizer mixing X and Z.

### 4. Holes as Logical Qubits

An alternative to boundary-encoded qubits is **hole-encoded** qubits.

#### Creating a Hole

A hole is created by:
1. Removing a connected set of data qubits from the lattice
2. Removing all stabilizers that involve these qubits
3. The hole boundary has a definite type (smooth or rough)

#### Hole Boundaries

**Smooth hole** (X-type boundary around hole):
- Plaquette stabilizers are truncated at the hole edge
- The hole "eats" m-anyons

**Rough hole** (Z-type boundary around hole):
- Vertex stabilizers are truncated at the hole edge
- The hole "eats" e-anyons

#### Logical Operators from Holes

A pair of smooth holes encodes one logical qubit:
- **Logical Z**: String connecting the two holes
- **Logical X**: Loop around one hole

```
    Hole 1          Hole 2
      ○──────Z──────○
      │
      X (loop around hole 1)
```

For a single hole with outer boundary:
- Logical Z: String from hole to outer boundary (same type)
- Logical X: Loop around hole (or string to opposite-type boundary)

### 5. Multiple Holes and Encoding Rate

With $n$ holes (pairs):
- Logical qubits: Approximately $n/2$ to $n$
- Depends on hole boundary types and outer boundary

#### Encoding Rate Comparison

| Encoding Method | Physical Qubits | Logical Qubits | Rate |
|-----------------|-----------------|----------------|------|
| Standard boundary | $d^2$ | 1 | $1/d^2$ |
| Two smooth holes | $\sim d^2$ | 1 | $\sim 1/d^2$ |
| $2k$ holes | $\sim k \cdot d^2$ | $k$ | $\sim 1/d^2$ |

The encoding rate is asymptotically the same, but holes provide more flexibility for operations.

---

## Quantum Mechanics Connection

### Non-Abelian Anyons from Defects

While the basic surface code supports only Abelian anyons (e and m), twist defects create **non-Abelian** behavior:

#### Braiding Statistics

When two genons are exchanged:
$$|\psi\rangle \to R |\psi\rangle$$

where $R$ is a **matrix** (not just a phase), acting on a degenerate ground state manifold.

This is the signature of non-Abelian anyons:
$$R^{(12)} R^{(23)} R^{(12)} \neq R^{(23)} R^{(12)} R^{(23)} \quad \text{(braid group)}$$

#### Implications for Quantum Computing

Non-Abelian braiding enables:
- **Topological quantum gates**: Errors during braiding are suppressed
- **Protected operations**: Only topological invariants matter
- **Potential for universality**: With the right anyon model

However, surface code genons alone give only Clifford gates—magic states are still needed for universality.

### Topological Quantum Field Theory Perspective

In TQFT language:
- Holes are **punctures** in the manifold
- Twist defects are **branch points**
- Logical operators are elements of mapping class groups
- Braiding corresponds to Dehn twists

The surface code realizes a $\mathbb{Z}_2$ gauge theory, and defects modify its structure locally.

---

## Afternoon Session: Defect Braiding and Lattice Surgery (2.5 hours)

### 1. Braiding Operations

#### Physical Implementation of Braiding

To braid two defects:
1. Deform one defect's position through the lattice
2. Move it around the other defect
3. Return to (permuted) original position

During motion:
- Stabilizers are continuously modified
- The code maintains error correction capability
- The logical state transforms

#### Braiding Smooth Holes

Two smooth holes encode one logical qubit. Braiding them:

$$\text{Exchange: } |0\rangle_L \to |0\rangle_L, \quad |1\rangle_L \to e^{i\pi/4}|1\rangle_L$$

This implements a **T gate** (up to Clifford corrections)? Actually, for simple smooth holes, braiding gives:

$$\text{Exchange} = \exp\left(i\frac{\pi}{4}\bar{Z}\right) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1+i & 0 \\ 0 & 1-i \end{pmatrix}$$

This is related to the S gate (phase gate).

#### Braiding Twist Defects

For genon braiding:
$$\sigma_1 \sigma_2 = \text{CNOT-like operation}$$

More precisely, with 4 genons encoding 2 logical qubits:
$$\sigma_i \sigma_j \sim \text{logical Clifford gate}$$

### 2. Logical Gates from Defects

#### Gates Achievable by Braiding

| Operation | Gate | Defect Manipulation |
|-----------|------|---------------------|
| Single exchange | $S$ or $\sqrt{Z}$ | Braid two smooth holes |
| Double exchange | $Z$ | Braid twice |
| Genon pair exchange | CNOT-like | Braid genons |
| Move defect around loop | Logical Pauli | Anyon transport |

#### Example: Hadamard from Defect Fusion

To implement Hadamard-like operations:
1. Create a pair of defects (smooth and rough)
2. Braid with the logical qubit's encoding
3. Fuse defects back together

The topological process transforms $\bar{X} \leftrightarrow \bar{Z}$.

### 3. Lattice Surgery: An Alternative

**Lattice surgery** provides another method for logical operations without defect braiding.

#### Basic Idea

1. **Merge** two surface code patches by measuring joint stabilizers
2. The merged patch has modified logical operators
3. **Split** patches by turning off joint stabilizers

#### Lattice Surgery vs. Defect Braiding

| Aspect | Defect Braiding | Lattice Surgery |
|--------|-----------------|-----------------|
| Qubit movement | Defects move through lattice | Patches are static |
| Gate implementation | Continuous deformation | Discrete merge/split |
| Time overhead | Path length dependent | Constant depth |
| Space overhead | Extra space for paths | Extra patch edges |
| Hardware requirements | Flexible stabilizer modification | Boundary control |

#### Current Experimental Preference

Most near-term proposals favor **lattice surgery** because:
- Simpler stabilizer modifications
- Better understood error models
- Compatible with fixed qubit layouts

However, defect braiding may offer advantages for certain architectures.

### 4. Worked Example: Hole Pair Encoding

**Problem**: Design an encoding using two smooth holes and calculate the logical operators.

**Solution**:

Consider a surface code with two smooth holes (call them $H_1$ and $H_2$):

```
┌───────────────────────────────────┐
│                                   │
│      ┌───┐           ┌───┐       │
│      │H_1│           │H_2│       │
│      │   │           │   │       │
│      └───┘           └───┘       │
│                                   │
└───────────────────────────────────┘
```

**Logical Z** ($\bar{Z}$):
- Z-string connecting $H_1$ to $H_2$
- Weight = distance between holes
- Any path works (deformable)

**Logical X** ($\bar{X}$):
- X-loop around $H_1$ (or equivalently, around $H_2$)
- Weight = perimeter of hole
- Loops around both holes are equivalent to stabilizers

**Verification**:
- $\bar{Z}$ and $\bar{X}$ anticommute (string crosses loop once)
- Both commute with all stabilizers
- Neither is a stabilizer product

$$\boxed{\text{Two smooth holes encode 1 logical qubit}}$$

### 5. Resource Analysis

#### Space Overhead for Defect Encoding

For a distance-$d$ code with two holes:
- Minimum hole separation: $d$ (for code distance)
- Hole size: $O(d)$ perimeter for distance
- Total area: $O(d^2)$ per logical qubit

#### Time Overhead for Braiding

To braid two defects:
- Path length: $O(d)$ lattice steps
- Each step: $O(1)$ stabilizer modifications
- Total time: $O(d)$ syndrome rounds

#### Error Accumulation

During braiding:
- Errors can accumulate along the path
- Effective error rate: $p \cdot (\text{path length})$
- Threshold considerations modified

---

## Practice Problems

### Problem Set 803

#### Direct Application

1. **Twist defect construction**: Draw a 5×5 surface code lattice with a twist defect at the center. Label which stabilizers are X-type and which are Z-type on each side of the defect.

2. **Hole encoding**: For a surface code with one rough hole and standard smooth/rough outer boundaries, identify the logical X and Z operators.

3. **Braiding path**: Sketch the path for braiding two smooth holes in a 7×7 surface code. Calculate the minimum number of stabilizer modifications needed.

#### Intermediate

4. **Gate from braiding**: Show that a full exchange (360° rotation) of two identical smooth holes implements the identity (or a Pauli, depending on convention).

5. **Hole fusion**: When two smooth holes are brought together and "fused" (their boundaries merged), what happens to the encoded logical qubit?

6. **Distance preservation**: Prove that for a surface code with two smooth holes separated by distance $d$, the code distance is at least $d$.

#### Challenging

7. **Genon fusion rules**: Derive the fusion rules for genons in the surface code. What happens when two genons are brought together?

8. **Universal gate set**: Explain why defect braiding in the surface code cannot achieve a universal gate set. What additional resource is needed?

9. **Optimal braiding path**: For braiding $n$ defects to implement a specific Clifford circuit, determine an algorithm to find the minimum-length braiding path.

---

## Evening Session: Computational Lab (1.5 hours)

### Lab 803: Defect Manipulation Simulation

```python
"""
Day 803 Computational Lab: Defects and Holes in Surface Codes
Simulation of defect creation, manipulation, and braiding
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum

class DefectType(Enum):
    """Types of defects in the surface code."""
    SMOOTH_HOLE = "smooth_hole"  # X-type boundary
    ROUGH_HOLE = "rough_hole"    # Z-type boundary
    TWIST = "twist"              # Genon (boundary type transition)


@dataclass
class Defect:
    """Represents a defect in the surface code."""
    defect_type: DefectType
    position: Tuple[float, float]  # (row, col)
    size: float = 1.0              # Radius for holes
    label: str = ""


class DefectSurfaceCode:
    """
    Surface code with defects (holes and twists).

    Supports visualization and basic defect manipulation.
    """

    def __init__(self, d: int):
        """
        Initialize a distance-d surface code.

        Parameters:
            d: Code distance (lattice size)
        """
        self.d = d
        self.defects: List[Defect] = []
        self.braiding_history: List[List[Tuple[float, float]]] = []

    def add_hole(self, row: float, col: float, hole_type: str, size: float = 1.0, label: str = ""):
        """Add a hole to the surface code."""
        if hole_type == "smooth":
            dtype = DefectType.SMOOTH_HOLE
        elif hole_type == "rough":
            dtype = DefectType.ROUGH_HOLE
        else:
            raise ValueError("hole_type must be 'smooth' or 'rough'")

        defect = Defect(dtype, (row, col), size, label)
        self.defects.append(defect)
        return defect

    def add_twist(self, row: float, col: float, label: str = ""):
        """Add a twist defect (genon) to the surface code."""
        defect = Defect(DefectType.TWIST, (row, col), 0.3, label)
        self.defects.append(defect)
        return defect

    def braid_defects(self, idx1: int, idx2: int, clockwise: bool = True):
        """
        Record a braiding operation between two defects.

        This simulates moving defect idx1 around defect idx2.
        """
        d1 = self.defects[idx1]
        d2 = self.defects[idx2]

        # Generate braiding path (semicircle around d2)
        center = d2.position
        start = d1.position
        radius = np.sqrt((start[0] - center[0])**2 + (start[1] - center[1])**2)

        # Create path points
        n_points = 20
        start_angle = np.arctan2(start[0] - center[0], start[1] - center[1])
        if clockwise:
            angles = np.linspace(start_angle, start_angle + np.pi, n_points)
        else:
            angles = np.linspace(start_angle, start_angle - np.pi, n_points)

        path = [(center[0] + radius * np.sin(a),
                 center[1] + radius * np.cos(a)) for a in angles]

        self.braiding_history.append(path)

        # Update defect position
        self.defects[idx1] = Defect(d1.defect_type, path[-1], d1.size, d1.label)

    def get_logical_operators(self) -> dict:
        """
        Determine logical operators based on defect configuration.

        Returns dict with 'X' and 'Z' operator descriptions.
        """
        smooth_holes = [d for d in self.defects if d.defect_type == DefectType.SMOOTH_HOLE]
        rough_holes = [d for d in self.defects if d.defect_type == DefectType.ROUGH_HOLE]

        operators = {}

        if len(smooth_holes) >= 2:
            # Two smooth holes: Z connects them, X loops around one
            operators['Z'] = f"String from {smooth_holes[0].label} to {smooth_holes[1].label}"
            operators['X'] = f"Loop around {smooth_holes[0].label}"
        elif len(rough_holes) >= 2:
            # Two rough holes: X connects them, Z loops around one
            operators['X'] = f"String from {rough_holes[0].label} to {rough_holes[1].label}"
            operators['Z'] = f"Loop around {rough_holes[0].label}"
        else:
            # Standard encoding (or mixed)
            operators['Z'] = "Vertical string (rough to rough boundary)"
            operators['X'] = "Horizontal string (smooth to smooth boundary)"

        return operators

    def visualize(self, show_logical_ops: bool = True, figsize: Tuple[int, int] = (10, 10)):
        """
        Visualize the surface code with defects.
        """
        fig, ax = plt.subplots(figsize=figsize)
        d = self.d

        # Draw lattice (simplified)
        for i in range(d + 1):
            ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5, zorder=1)
            ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5, zorder=1)

        # Draw standard boundaries
        # Smooth (left/right)
        ax.plot([-0.5, -0.5], [-0.5, d - 0.5], 'b-', linewidth=4)
        ax.plot([d - 0.5, d - 0.5], [-0.5, d - 0.5], 'b-', linewidth=4)
        # Rough (top/bottom)
        ax.plot([-0.5, d - 0.5], [-0.5, -0.5], 'r-', linewidth=4)
        ax.plot([-0.5, d - 0.5], [d - 0.5, d - 0.5], 'r-', linewidth=4)

        # Draw defects
        for defect in self.defects:
            row, col = defect.position

            if defect.defect_type == DefectType.SMOOTH_HOLE:
                # Blue circle for smooth hole
                circle = Circle((col, d - 1 - row), defect.size,
                                facecolor='lightblue', edgecolor='blue',
                                linewidth=2, zorder=3)
                ax.add_patch(circle)
                ax.text(col, d - 1 - row, defect.label or 'S',
                       ha='center', va='center', fontsize=12, fontweight='bold')

            elif defect.defect_type == DefectType.ROUGH_HOLE:
                # Red circle for rough hole
                circle = Circle((col, d - 1 - row), defect.size,
                                facecolor='lightsalmon', edgecolor='red',
                                linewidth=2, zorder=3)
                ax.add_patch(circle)
                ax.text(col, d - 1 - row, defect.label or 'R',
                       ha='center', va='center', fontsize=12, fontweight='bold')

            elif defect.defect_type == DefectType.TWIST:
                # Star marker for twist defect
                ax.scatter(col, d - 1 - row, s=400, c='purple',
                          marker='*', zorder=4)
                ax.text(col + 0.3, d - 1 - row, defect.label or 'T',
                       fontsize=10, color='purple', fontweight='bold')

        # Draw braiding paths
        for path in self.braiding_history:
            path_plot = [(p[1], d - 1 - p[0]) for p in path]
            xs, ys = zip(*path_plot)
            ax.plot(xs, ys, 'g--', linewidth=2, alpha=0.7, zorder=2)
            ax.annotate('', xy=path_plot[-1], xytext=path_plot[-2],
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))

        # Show logical operators if requested
        if show_logical_ops:
            ops = self.get_logical_operators()
            text = f"Logical Z: {ops.get('Z', 'N/A')}\nLogical X: {ops.get('X', 'N/A')}"
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlim(-1, d)
        ax.set_ylim(-1, d)
        ax.set_aspect('equal')
        ax.set_title(f'Surface Code with Defects (d={d})\n'
                     f'Blue: Smooth boundary/hole | Red: Rough boundary/hole',
                     fontsize=12)
        ax.axis('off')

        plt.tight_layout()
        return fig, ax


def demo_hole_encoding():
    """Demonstrate encoding a logical qubit using two smooth holes."""
    print("\n" + "="*60)
    print("Hole-Based Logical Qubit Encoding")
    print("="*60)

    # Create a d=7 surface code with two smooth holes
    code = DefectSurfaceCode(d=7)

    # Add two smooth holes
    code.add_hole(row=3, col=1.5, hole_type="smooth", size=0.8, label="H₁")
    code.add_hole(row=3, col=5.5, hole_type="smooth", size=0.8, label="H₂")

    print("\nConfiguration:")
    print("  Two smooth holes (H₁ and H₂) in a distance-7 surface code")
    print("  Separation: 4 lattice units (gives distance-4 for hole-encoded qubit)")

    ops = code.get_logical_operators()
    print("\nLogical Operators:")
    print(f"  Z̄: {ops['Z']}")
    print(f"  X̄: {ops['X']}")

    # Visualize
    fig, ax = code.visualize()
    plt.savefig('hole_encoding.png', dpi=150, bbox_inches='tight')
    plt.show()

    return code


def demo_twist_defects():
    """Demonstrate twist defects (genons)."""
    print("\n" + "="*60)
    print("Twist Defect (Genon) Demonstration")
    print("="*60)

    code = DefectSurfaceCode(d=7)

    # Add four twist defects (encodes 2 logical qubits)
    code.add_twist(row=2, col=2, label="τ₁")
    code.add_twist(row=2, col=5, label="τ₂")
    code.add_twist(row=5, col=2, label="τ₃")
    code.add_twist(row=5, col=5, label="τ₄")

    print("\nConfiguration:")
    print("  Four twist defects (τ₁, τ₂, τ₃, τ₄) in a 2×2 arrangement")
    print("  This configuration encodes 2 logical qubits")

    print("\nGenon properties:")
    print("  - At each twist, e-anyons transmute to m-anyons (and vice versa)")
    print("  - Braiding genons implements non-trivial logical gates")
    print("  - The braid group representation gives Clifford gates")

    # Visualize
    fig, ax = code.visualize(show_logical_ops=False)
    ax.set_title('Surface Code with Twist Defects (Genons)', fontsize=12)
    plt.savefig('twist_defects.png', dpi=150, bbox_inches='tight')
    plt.show()

    return code


def demo_braiding():
    """Demonstrate braiding of smooth holes."""
    print("\n" + "="*60)
    print("Defect Braiding Demonstration")
    print("="*60)

    code = DefectSurfaceCode(d=9)

    # Add two smooth holes
    h1 = code.add_hole(row=4, col=2, hole_type="smooth", size=0.7, label="H₁")
    h2 = code.add_hole(row=4, col=6, hole_type="smooth", size=0.7, label="H₂")

    print("\nInitial configuration:")
    print(f"  H₁ at position (4, 2)")
    print(f"  H₂ at position (4, 6)")

    # Perform a braid (H₁ around H₂)
    code.braid_defects(0, 1, clockwise=True)

    print("\nAfter braiding H₁ clockwise around H₂:")
    print(f"  H₁ moves to opposite side of H₂")
    print(f"  This implements: |ψ⟩ → e^(iπ/4 Z̄)|ψ⟩ (S gate, up to conventions)")

    # Visualize
    fig, ax = code.visualize()
    ax.set_title('Braiding Two Smooth Holes\n(Green path shows H₁ trajectory)', fontsize=12)
    plt.savefig('defect_braiding.png', dpi=150, bbox_inches='tight')
    plt.show()

    return code


def compare_defect_vs_surgery():
    """Compare defect braiding with lattice surgery."""
    print("\n" + "="*60)
    print("Defect Braiding vs. Lattice Surgery Comparison")
    print("="*60)

    print("\n" + "-"*40)
    print("DEFECT BRAIDING")
    print("-"*40)
    print("""
Advantages:
  + Continuous operation (topologically protected)
  + Natural representation of logical gates
  + Elegant mathematical structure

Disadvantages:
  - Requires moving defects through lattice
  - Time overhead proportional to path length
  - More complex stabilizer modifications

Gate examples:
  - S gate: Exchange two smooth holes
  - CNOT: Braid genon pairs
  - Hadamard: Fuse/create defect pairs
    """)

    print("\n" + "-"*40)
    print("LATTICE SURGERY")
    print("-"*40)
    print("""
Advantages:
  + Fixed qubit layout (no movement)
  + Constant-depth operations
  + Simpler error analysis
  + Better for current hardware

Disadvantages:
  - Requires additional ancilla patches
  - Edge measurement overhead
  - Less intuitive for some operations

Gate examples:
  - CNOT: Merge and split patches
  - Hadamard: Transversal + magic state
  - T gate: Magic state injection
    """)

    print("\n" + "-"*40)
    print("CURRENT EXPERIMENTAL STATUS")
    print("-"*40)
    print("""
Google (Sycamore): Lattice surgery demonstrated (2023)
IBM (Eagle): Surface code patches, surgery in development
Quantinuum (H-series): Flexible, exploring both approaches
    """)


def visualize_logical_operator_on_holes():
    """Visualize logical operators for hole-encoded qubits."""
    print("\n" + "="*60)
    print("Logical Operators on Hole-Encoded Qubits")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    d = 7

    # Left: Logical Z (string between holes)
    ax1 = axes[0]
    code1 = DefectSurfaceCode(d=d)
    code1.add_hole(row=3, col=1.5, hole_type="smooth", size=0.8, label="H₁")
    code1.add_hole(row=3, col=5.5, hole_type="smooth", size=0.8, label="H₂")

    # Draw basic structure
    for i in range(d + 1):
        ax1.axhline(y=i - 0.5, color='lightgray', linewidth=0.5, zorder=1)
        ax1.axvline(x=i - 0.5, color='lightgray', linewidth=0.5, zorder=1)

    # Draw holes
    for defect in code1.defects:
        row, col = defect.position
        circle = Circle((col, d - 1 - row), defect.size,
                        facecolor='lightblue', edgecolor='blue',
                        linewidth=2, zorder=3)
        ax1.add_patch(circle)
        ax1.text(col, d - 1 - row, defect.label,
                ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw logical Z (string)
    ax1.plot([1.5 + 0.8, 5.5 - 0.8], [d - 1 - 3, d - 1 - 3],
            'purple', linewidth=4, zorder=4, label='Logical Z̄')
    ax1.scatter([2.3, 3.5, 4.7], [d - 1 - 3]*3, s=150, c='purple', zorder=5)

    ax1.set_xlim(-1, d)
    ax1.set_ylim(-1, d)
    ax1.set_aspect('equal')
    ax1.set_title('Logical Z̄: String Connecting Holes', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.axis('off')

    # Right: Logical X (loop around hole)
    ax2 = axes[1]

    for i in range(d + 1):
        ax2.axhline(y=i - 0.5, color='lightgray', linewidth=0.5, zorder=1)
        ax2.axvline(x=i - 0.5, color='lightgray', linewidth=0.5, zorder=1)

    for defect in code1.defects:
        row, col = defect.position
        circle = Circle((col, d - 1 - row), defect.size,
                        facecolor='lightblue', edgecolor='blue',
                        linewidth=2, zorder=3)
        ax2.add_patch(circle)
        ax2.text(col, d - 1 - row, defect.label,
                ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw logical X (loop around H₁)
    theta = np.linspace(0, 2 * np.pi, 50)
    loop_r = 1.5
    loop_x = 1.5 + loop_r * np.cos(theta)
    loop_y = (d - 1 - 3) + loop_r * np.sin(theta)
    ax2.plot(loop_x, loop_y, 'green', linewidth=4, zorder=4, label='Logical X̄')

    ax2.set_xlim(-1, d)
    ax2.set_ylim(-1, d)
    ax2.set_aspect('equal')
    ax2.set_title('Logical X̄: Loop Around Hole', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('hole_logical_operators.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Day 803: Defects and Holes in the Surface Code")
    print("=" * 60)

    # Demonstrate hole encoding
    demo_hole_encoding()

    # Demonstrate twist defects
    demo_twist_defects()

    # Demonstrate braiding
    demo_braiding()

    # Compare approaches
    compare_defect_vs_surgery()

    # Visualize logical operators on holes
    visualize_logical_operator_on_holes()

    print("\n" + "=" * 60)
    print("Lab complete.")
    print("=" * 60)
```

### Expected Output

```
============================================================
Day 803: Defects and Holes in the Surface Code
============================================================

============================================================
Hole-Based Logical Qubit Encoding
============================================================

Configuration:
  Two smooth holes (H₁ and H₂) in a distance-7 surface code
  Separation: 4 lattice units (gives distance-4 for hole-encoded qubit)

Logical Operators:
  Z̄: String from H₁ to H₂
  X̄: Loop around H₁

============================================================
Twist Defect (Genon) Demonstration
============================================================

Configuration:
  Four twist defects (τ₁, τ₂, τ₃, τ₄) in a 2×2 arrangement
  This configuration encodes 2 logical qubits

Genon properties:
  - At each twist, e-anyons transmute to m-anyons (and vice versa)
  - Braiding genons implements non-trivial logical gates
  - The braid group representation gives Clifford gates

============================================================
Defect Braiding Demonstration
============================================================

Initial configuration:
  H₁ at position (4, 2)
  H₂ at position (4, 6)

After braiding H₁ clockwise around H₂:
  H₁ moves to opposite side of H₂
  This implements: |ψ⟩ → e^(iπ/4 Z̄)|ψ⟩ (S gate, up to conventions)

============================================================
Lab complete.
============================================================
```

---

## Summary

### Key Formulas

| Concept | Formula/Description |
|---------|---------------------|
| Twist defect (genon) | $e \xrightarrow{\text{twist}} m$, $m \xrightarrow{\text{twist}} e$ |
| Genon fusion | $\sigma^2 = \epsilon$ (fermion) |
| Hole encoding | 2 same-type holes → 1 logical qubit |
| Hole logical Z | String connecting holes |
| Hole logical X | Loop around one hole |
| Braiding gate | Exchange → $\sim S$ gate (Clifford) |

### Key Takeaways

1. **Twist defects (genons)** are points where boundary types transition, creating non-Abelian behavior

2. **Holes in the lattice** create internal boundaries that can encode logical qubits

3. **Defect braiding implements logical gates** through topological processes

4. **Clifford gates are achievable** via defect manipulation; universality requires magic states

5. **Lattice surgery is often preferred** in near-term experiments due to simpler implementation

6. **Space-time tradeoffs exist** between defect braiding (time overhead) and lattice surgery (space overhead)

---

## Daily Checklist

Before moving to Day 804, verify you can:

- [ ] Explain what a twist defect (genon) is and how it differs from holes
- [ ] Construct hole-encoded logical qubits and their operators
- [ ] Describe how braiding defects implements gates
- [ ] Compare defect braiding with lattice surgery
- [ ] Calculate resource overhead for defect-based encodings
- [ ] Identify which gates are achievable via defect manipulation

---

## Preview: Day 804

Tomorrow we study the **rotated surface code**:
- 45-degree rotation for efficient qubit layout
- Reduced qubit count: $(d^2 + 1)/2$ data qubits
- Heavy-hex and other hardware-native layouts
- Practical implementation considerations

---

## References

1. Bombin, H. "Topological order with a twist: Ising anyons from an Abelian model." *Physical Review Letters* 105, 030403 (2010)
2. Hastings, M. B., & Geller, A. "Reduced space-time and time costs using dislocation codes." *Quantum Information & Computation* 15, 962 (2015)
3. Brown, B. J., et al. "Poking holes and cutting corners to achieve Clifford gates with the surface code." *Physical Review X* 7, 021029 (2017)

---

*Day 803 reveals the rich structure hiding within the surface code—defects and holes transform a simple error-correcting code into a platform for topological quantum computation.*
