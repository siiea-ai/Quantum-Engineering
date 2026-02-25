# Day 816: Alternative Lattice Geometries

## Week 117, Day 4 | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Overview

While the square lattice surface code is the most studied, real quantum hardware often demands alternative geometries. Today we explore **hexagonal, triangular, and heavy-hex lattice** variants of the surface code. Each geometry offers different trade-offs between qubit connectivity, syndrome extraction efficiency, and error thresholds. Understanding these alternatives is essential for designing surface codes that match the constraints of specific hardware platforms.

---

## Daily Schedule

| Session | Duration | Content |
|---------|----------|---------|
| Morning | 3 hours | Hexagonal, triangular, and heavy-hex geometries |
| Afternoon | 2 hours | Connectivity analysis, threshold comparison |
| Evening | 2 hours | Python implementation of alternative lattices |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Describe** hexagonal and triangular surface code variants
2. **Explain** the heavy-hex architecture used by IBM
3. **Analyze** connectivity requirements for different geometries
4. **Compare** error thresholds across lattice types
5. **Calculate** qubit overhead for alternative geometries
6. **Implement** lattice construction algorithms for multiple geometries

---

## Core Content

### 1. Why Alternative Geometries?

The standard square lattice surface code requires **4-way connectivity**:

```
      o
      |
  o---●---o
      |
      o
```

Each data qubit must couple to 4 neighbors in the bulk. However:
- Some hardware naturally provides different connectivity
- Lower connectivity may be easier to fabricate
- Different geometries may have better error thresholds for specific noise models

### 2. The Hexagonal (Honeycomb) Lattice

**Structure:**
```
    o       o       o
     \     / \     /
      o---o   o---o
     /     \ /     \
    o       o       o
```

**Properties:**
- **Connectivity:** 3-way (each qubit connects to 3 neighbors)
- **Coordination number:** 3
- **Plaquettes:** Hexagonal faces (6 qubits each)

**Stabilizer Structure:**
- Weight-6 stabilizers in the bulk
- Weight-3 or weight-4 at boundaries

**Advantages:**
- Matches natural 3-way connectivity of some platforms
- Lower fabrication complexity

**Disadvantages:**
- Higher weight stabilizers require more gates per syndrome measurement
- Lower error threshold compared to square lattice

### 3. The Triangular Lattice

**Structure:**
```
    o---o---o---o
     \ / \ / \ /
      o---o---o
     / \ / \ / \
    o---o---o---o
```

**Properties:**
- **Connectivity:** 6-way (each qubit connects to 6 neighbors)
- **Coordination number:** 6
- **Plaquettes:** Triangular faces (3 qubits each)

**Stabilizer Structure:**
- Weight-3 stabilizers
- Simpler syndrome extraction

**Trade-offs:**
- Low stabilizer weight is good for syndrome extraction
- High connectivity (6-way) is hard to achieve in hardware
- Useful for theoretical analysis and certain 2D architectures

### 4. The Heavy-Hex Lattice (IBM)

IBM's superconducting quantum processors use a **heavy-hex** architecture:

**Structure:**
```
    ●=====●=====●=====●
    |     |     |     |
    ○     ○     ○     ○
    |     |     |     |
    ●=====●=====●=====●
    |     |     |     |
    ○     ○     ○     ○
    |     |     |     |
    ●=====●=====●=====●
```

Where:
- ● = "Heavy" vertices (connected to 3 neighbors)
- ○ = Auxiliary qubits on edges
- ===== = Couplings

**Key Properties:**
- **Connectivity:** 3-way maximum
- **No 4-way junctions:** Reduces crosstalk and fabrication complexity
- **Flag qubits:** Used for fault-tolerant syndrome extraction

**IBM's Rationale:**
$$\boxed{\text{Heavy-hex: Reduce crosstalk at the cost of more qubits}}$$

### 5. Subsystem Surface Codes

**Concept:**
Introduce gauge qubits that can be measured in different bases:

$$H = H_{\text{stabilizer}} + H_{\text{gauge}}$$

**Bacon-Shor Code:**
A subsystem code that works on a square lattice with only 2-way connectivity!

**Advantages:**
- Works with nearest-neighbor connectivity on a line
- Simpler syndrome extraction circuits

**Trade-offs:**
- Lower threshold than standard surface code
- More complex classical decoding

### 6. Floquet Codes

**Dynamic Approach:**
Instead of measuring the same stabilizers each cycle, alternate between different measurement patterns:

**Measurement Schedule:**
1. Measure X-stabilizers in one pattern
2. Measure Z-stabilizers in another pattern
3. Repeat with shifted patterns

**Honeycomb Floquet Code (Hastings-Haah):**
- Uses 3-way connectivity
- Weight-2 measurements only
- Emerges topological protection dynamically

$$\boxed{\text{Floquet: Topological order from dynamics, not statics}}$$

### 7. Connectivity Requirements Comparison

| Geometry | Connectivity | Bulk Stabilizer Weight | Notes |
|----------|--------------|----------------------|-------|
| Square (rotated) | 4-way | 4 | Standard choice |
| Hexagonal | 3-way | 6 | Larger stabilizers |
| Triangular | 6-way | 3 | High connectivity |
| Heavy-hex | 3-way | 4 (with flags) | IBM's approach |
| Bacon-Shor | 2-way | 2 (gauge) | Subsystem code |
| Floquet honeycomb | 3-way | 2 (dynamic) | Dynamic code |

### 8. Error Threshold Comparison

The **error threshold** depends on geometry and noise model:

| Code | Threshold (phenomenological) | Threshold (circuit-level) |
|------|----------------------------|--------------------------|
| Square surface | ~11% | ~0.5-1% |
| Hexagonal surface | ~7% | ~0.3-0.5% |
| Triangular surface | ~10% | ~0.4-0.6% |
| Heavy-hex (with flags) | ~8-10% | ~0.3-0.5% |
| Floquet honeycomb | ~3-5% | ~0.1-0.3% |

**Key Insight:**
Higher connectivity generally allows higher thresholds because errors can spread through more paths (paradoxically making them easier to track).

### 9. Qubit Overhead Analysis

For encoding one logical qubit with distance $d$:

**Square Lattice:**
$$n_{\text{square}} = 2d^2 - 1 \approx 2d^2$$

**Hexagonal Lattice:**
$$n_{\text{hex}} \approx 3d^2$$

**Heavy-hex Lattice:**
$$n_{\text{heavy-hex}} \approx 2.5d^2 - 3d$$

The additional overhead in heavy-hex and hexagonal comes from:
- Larger stabilizers requiring more ancilla
- Flag qubits for fault tolerance

### 10. Hardware Implementations

| Platform | Preferred Geometry | Reason |
|----------|-------------------|--------|
| Google Sycamore | Square | 4-way coupler design |
| IBM Eagle/Condor | Heavy-hex | Reduced crosstalk |
| Rigetti Aspen | Octagonal | 3-way connectivity natural |
| IonQ/Quantinuum | All-to-all | Flexible geometry |
| Neutral atoms | Programmable | Can implement any geometry |

---

## Quantum Computing Connection

### IBM's Heavy-Hex Journey (2020-2025)

IBM adopted heavy-hex for several practical reasons:

1. **Crosstalk reduction:** 4-way junctions cause frequency collisions
2. **Scalability:** Heavy-hex tiles uniformly across large chips
3. **Error correction compatibility:** Can still implement surface-like codes

**IBM Heron (2024):** 156 qubits in heavy-hex, demonstrating improved coherence.

### Google's Square Lattice Approach

Google maintains the square lattice because:

1. **Higher threshold:** Better theoretical error correction
2. **Simpler syndrome extraction:** Standard 4-qubit measurements
3. **Willow (2024):** Demonstrated below-threshold operation

### Neutral Atom Flexibility

Neutral atom platforms (QuEra, Pasqal) can:
- Rearrange atoms into any geometry
- Switch between square, triangular, or custom layouts
- Implement different codes on the same hardware

---

## Worked Examples

### Example 1: Heavy-Hex Stabilizer Construction

**Problem:** Construct the X and Z stabilizers for a minimal heavy-hex patch.

**Solution:**

Consider a 2x2 cell of the heavy-hex lattice:
```
    1=====2=====3
    |     |     |
    a     b     c
    |     |     |
    4=====5=====6
```

Where 1-6 are data qubits and a, b, c are flag/auxiliary qubits.

**X-Stabilizers:**
Without flags (weight-4):
$$S_X^1 = X_1 X_a X_4 X_2$$

Actually, in heavy-hex, we use flag-based extraction:
1. Prepare flag qubit in $|+\rangle$
2. Apply CNOTs to data qubits
3. Measure flag and data ancilla

**Z-Stabilizers:**
$$S_Z^1 = Z_1 Z_2 Z_5 Z_4$$

The exact circuit depends on the flag protocol.

---

### Example 2: Comparing Qubit Overhead

**Problem:** Calculate the number of physical qubits needed to encode one logical qubit at distance $d = 5$ for square, hexagonal, and heavy-hex lattices.

**Solution:**

**Square lattice:**
$$n_{\text{square}} = 2d^2 - 1 = 2(25) - 1 = 49$$

**Hexagonal lattice:**
$$n_{\text{hex}} \approx 3d^2 = 3(25) = 75$$

**Heavy-hex lattice:**
$$n_{\text{heavy-hex}} \approx 2.5d^2 - 3d = 2.5(25) - 15 = 47.5 \approx 48$$

**Comparison:**
| Geometry | Qubits for d=5 | Overhead vs. Square |
|----------|----------------|---------------------|
| Square | 49 | 1.00x |
| Heavy-hex | 48 | 0.98x |
| Hexagonal | 75 | 1.53x |

Heavy-hex is competitive with square in qubit count, while hexagonal has significant overhead.

---

### Example 3: Threshold Scaling

**Problem:** A quantum computer has a physical error rate of $p = 0.1\%$. Compare the logical error rates for square and hexagonal lattice codes at distance $d = 7$.

**Solution:**

Using the scaling formula:
$$p_L \approx A \left(\frac{p}{p_{th}}\right)^{\lceil d/2 \rceil}$$

With $A \approx 0.1$ and $\lceil 7/2 \rceil = 4$:

**Square lattice** ($p_{th} \approx 1\%$):
$$p_L^{\text{square}} \approx 0.1 \times \left(\frac{0.001}{0.01}\right)^4 = 0.1 \times (0.1)^4 = 10^{-5}$$

**Hexagonal lattice** ($p_{th} \approx 0.5\%$):
$$p_L^{\text{hex}} \approx 0.1 \times \left(\frac{0.001}{0.005}\right)^4 = 0.1 \times (0.2)^4 = 1.6 \times 10^{-4}$$

**Comparison:**
$$\frac{p_L^{\text{hex}}}{p_L^{\text{square}}} = \frac{1.6 \times 10^{-4}}{10^{-5}} = 16$$

The hexagonal code has 16x higher logical error rate at the same distance due to its lower threshold.

---

## Practice Problems

### Direct Application

**Problem 1:** Draw a hexagonal lattice surface code patch of "distance 3" (smallest non-trivial). Count the data qubits and identify the stabilizers.

**Problem 2:** For a triangular lattice code, what is the weight of the X and Z stabilizers? Why does the high connectivity help syndrome extraction?

**Problem 3:** IBM's Eagle processor has 127 qubits in a heavy-hex layout. Estimate the maximum code distance achievable for a single logical qubit.

### Intermediate

**Problem 4:** Prove that any planar graph can support a surface-code-like construction, where stabilizers are defined on faces and vertices.

**Problem 5:** The Floquet honeycomb code uses only weight-2 measurements. Explain how topological order emerges despite never measuring weight-4 stabilizers directly.

**Problem 6:** Compare the syndrome extraction circuit depth for square (weight-4) and hexagonal (weight-6) lattice codes. Which is more vulnerable to mid-circuit errors?

### Challenging

**Problem 7:** Design a "morphing" surface code that transitions from square to hexagonal geometry across the patch. What happens at the interface?

**Problem 8:** Derive the error threshold for a triangular lattice surface code under depolarizing noise using a statistical mechanics mapping.

**Problem 9:** Heavy-hex codes use flag qubits for fault tolerance. Prove that without flags, a single fault during syndrome extraction could create a weight-2 error, reducing the effective code distance.

---

## Computational Lab

### Lab 816: Alternative Lattice Geometries

```python
"""
Day 816 Computational Lab: Alternative Lattice Geometries
==========================================================

This lab implements and visualizes different lattice geometries
for surface codes, including hexagonal, triangular, and heavy-hex.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

class SquareLattice:
    """Standard square lattice surface code."""

    def __init__(self, d: int):
        self.d = d
        self.name = "Square"
        self.connectivity = 4
        self.n_data = d ** 2
        self.n_total = 2 * d ** 2 - 1
        self.threshold = 0.01  # Approximate

    def get_positions(self):
        """Return qubit positions."""
        positions = []
        for i in range(self.d):
            for j in range(self.d):
                positions.append((i, j))
        return positions

    def get_edges(self):
        """Return edges (couplings) between qubits."""
        edges = []
        for i in range(self.d):
            for j in range(self.d):
                if i < self.d - 1:
                    edges.append(((i, j), (i+1, j)))
                if j < self.d - 1:
                    edges.append(((i, j), (i, j+1)))
        return edges


class HexagonalLattice:
    """Hexagonal (honeycomb) lattice surface code."""

    def __init__(self, d: int):
        self.d = d
        self.name = "Hexagonal"
        self.connectivity = 3
        self.n_data = int(1.5 * d ** 2)  # Approximate
        self.n_total = int(3 * d ** 2)  # With ancillas
        self.threshold = 0.007  # Approximate

    def get_positions(self):
        """Return qubit positions in honeycomb arrangement."""
        positions = []
        for i in range(self.d):
            for j in range(self.d):
                # Offset every other row
                x_offset = 0.5 if i % 2 == 1 else 0
                # Two atoms per unit cell
                positions.append((i * np.sqrt(3), j * 1.5 + x_offset))
                if j < self.d - 1:
                    positions.append((i * np.sqrt(3), j * 1.5 + 0.5 + x_offset))
        return positions

    def get_edges(self):
        """Return edges for honeycomb lattice."""
        edges = []
        positions = self.get_positions()

        # Connect nearby qubits (distance < 1.2)
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i < j:
                    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    if dist < 1.2:
                        edges.append((p1, p2))
        return edges


class TriangularLattice:
    """Triangular lattice surface code."""

    def __init__(self, d: int):
        self.d = d
        self.name = "Triangular"
        self.connectivity = 6
        self.n_data = d ** 2
        self.n_total = int(1.5 * d ** 2)
        self.threshold = 0.01  # Similar to square

    def get_positions(self):
        """Return qubit positions in triangular arrangement."""
        positions = []
        for i in range(self.d):
            for j in range(self.d):
                x_offset = 0.5 * (i % 2)
                positions.append((j + x_offset, i * np.sqrt(3) / 2))
        return positions

    def get_edges(self):
        """Return edges for triangular lattice (6-way connectivity)."""
        edges = []
        positions = self.get_positions()

        # Connect neighbors (distance ~ 1)
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i < j:
                    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    if dist < 1.1:
                        edges.append((p1, p2))
        return edges


class HeavyHexLattice:
    """IBM's heavy-hex lattice."""

    def __init__(self, d: int):
        self.d = d
        self.name = "Heavy-Hex"
        self.connectivity = 3
        self.n_data = int(d ** 2)  # Data qubits
        self.n_aux = int(0.5 * d ** 2)  # Auxiliary qubits
        self.n_total = self.n_data + self.n_aux
        self.threshold = 0.008  # Approximate

    def get_positions(self):
        """Return heavy and light qubit positions."""
        heavy_positions = []  # Vertices (degree 3)
        light_positions = []  # Edge qubits (degree 2)

        for i in range(self.d):
            for j in range(self.d):
                # Heavy qubits at grid points
                heavy_positions.append((2*j, 2*i))

                # Light qubits on horizontal edges
                if j < self.d - 1:
                    light_positions.append((2*j + 1, 2*i))

        return heavy_positions, light_positions

    def get_edges(self):
        """Return edges connecting qubits."""
        edges = []
        heavy, light = self.get_positions()

        # Edges from heavy through light to heavy (horizontal)
        for i in range(self.d):
            for j in range(self.d - 1):
                h1 = (2*j, 2*i)
                l = (2*j + 1, 2*i)
                h2 = (2*j + 2, 2*i)
                edges.append((h1, l))
                edges.append((l, h2))

        # Vertical edges (no intermediate qubit)
        for i in range(self.d - 1):
            for j in range(self.d):
                h1 = (2*j, 2*i)
                h2 = (2*j, 2*i + 2)
                edges.append((h1, h2))

        return edges


def visualize_lattice(lattice, figsize=(10, 10)):
    """Visualize a lattice geometry."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(lattice, HeavyHexLattice):
        heavy, light = lattice.get_positions()
        positions = heavy + light

        # Draw edges
        for e in lattice.get_edges():
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                   'k-', linewidth=2, zorder=1)

        # Draw heavy qubits
        for p in heavy:
            circle = Circle(p, 0.25, facecolor='steelblue',
                          edgecolor='black', linewidth=2, zorder=5)
            ax.add_patch(circle)

        # Draw light qubits
        for p in light:
            circle = Circle(p, 0.15, facecolor='coral',
                          edgecolor='black', linewidth=1.5, zorder=5)
            ax.add_patch(circle)

        legend_elements = [
            mpatches.Patch(facecolor='steelblue', label='Heavy (degree 3)'),
            mpatches.Patch(facecolor='coral', label='Light (degree 2)'),
        ]
    else:
        positions = lattice.get_positions()

        # Draw edges
        for e in lattice.get_edges():
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                   'k-', linewidth=1.5, zorder=1)

        # Draw qubits
        for p in positions:
            circle = Circle(p, 0.15, facecolor='steelblue',
                          edgecolor='black', linewidth=2, zorder=5)
            ax.add_patch(circle)

        legend_elements = []

    # Formatting
    all_x = [p[0] for p in positions]
    all_y = [p[1] for p in positions]

    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_aspect('equal')
    ax.set_title(f'{lattice.name} Lattice (d={lattice.d}, connectivity={lattice.connectivity})',
                fontsize=14)

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')

    # Add stats
    stats_text = f"Data qubits: {lattice.n_data}\nTotal qubits: ~{lattice.n_total}\nThreshold: ~{lattice.threshold*100:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig, ax


def compare_geometries():
    """Compare qubit overhead and thresholds across geometries."""
    distances = [3, 5, 7, 9, 11]

    geometries = {
        'Square': {'factor': 2.0, 'threshold': 1.0},
        'Hexagonal': {'factor': 3.0, 'threshold': 0.7},
        'Triangular': {'factor': 1.5, 'threshold': 1.0},
        'Heavy-Hex': {'factor': 2.5, 'threshold': 0.8},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Qubit count comparison
    ax1 = axes[0]
    for name, params in geometries.items():
        qubits = [params['factor'] * d**2 for d in distances]
        ax1.plot(distances, qubits, 'o-', linewidth=2, markersize=8, label=name)

    ax1.set_xlabel('Code Distance', fontsize=12)
    ax1.set_ylabel('Number of Qubits', fontsize=12)
    ax1.set_title('Qubit Overhead', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Threshold comparison
    ax2 = axes[1]
    names = list(geometries.keys())
    thresholds = [geometries[n]['threshold'] for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bars = ax2.bar(names, thresholds, color=colors)
    ax2.set_ylabel('Relative Threshold', fontsize=12)
    ax2.set_title('Error Threshold Comparison', fontsize=14)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Square baseline')
    ax2.legend()

    # Efficiency metric: threshold / overhead
    ax3 = axes[2]
    efficiency = [geometries[n]['threshold'] / geometries[n]['factor'] for n in names]
    bars = ax3.bar(names, efficiency, color=colors)
    ax3.set_ylabel('Efficiency (threshold/overhead)', fontsize=12)
    ax3.set_title('Overall Efficiency Metric', fontsize=14)

    plt.tight_layout()
    return fig


def visualize_stabilizers_comparison():
    """Show stabilizer structure for different geometries."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Square lattice stabilizer
    ax1 = axes[0]
    ax1.set_title('Square: Weight-4 Stabilizer', fontsize=12)

    # Draw 4 qubits in a square
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for p in positions:
        circle = Circle(p, 0.15, facecolor='steelblue', edgecolor='black', linewidth=2)
        ax1.add_patch(circle)

    # Draw stabilizer region
    stab = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)],
                  alpha=0.3, facecolor='red', edgecolor='darkred', linewidth=2)
    ax1.add_patch(stab)

    ax1.annotate('X X\nX X', (0.5, 0.5), ha='center', va='center', fontsize=14, color='darkred')
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Hexagonal lattice stabilizer
    ax2 = axes[1]
    ax2.set_title('Hexagonal: Weight-6 Stabilizer', fontsize=12)

    # Draw 6 qubits in a hexagon
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    hex_pos = [(np.cos(a), np.sin(a)) for a in angles]
    for p in hex_pos:
        circle = Circle(p, 0.15, facecolor='steelblue', edgecolor='black', linewidth=2)
        ax2.add_patch(circle)

    # Draw stabilizer region
    stab = RegularPolygon((0, 0), 6, radius=1.1,
                          alpha=0.3, facecolor='red', edgecolor='darkred', linewidth=2)
    ax2.add_patch(stab)

    ax2.annotate('XXXXXX', (0, 0), ha='center', va='center', fontsize=12, color='darkred')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Triangular lattice stabilizer
    ax3 = axes[2]
    ax3.set_title('Triangular: Weight-3 Stabilizer', fontsize=12)

    # Draw 3 qubits in a triangle
    tri_pos = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
    for p in tri_pos:
        circle = Circle(p, 0.15, facecolor='steelblue', edgecolor='black', linewidth=2)
        ax3.add_patch(circle)

    # Draw stabilizer region
    stab = Polygon(tri_pos, alpha=0.3, facecolor='red', edgecolor='darkred', linewidth=2)
    ax3.add_patch(stab)

    ax3.annotate('X X X', (0.5, 0.3), ha='center', va='center', fontsize=14, color='darkred')
    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_aspect('equal')
    ax3.axis('off')

    plt.tight_layout()
    return fig


def analyze_connectivity():
    """Analyze and visualize connectivity requirements."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    lattices = [
        ('Square', 4, 4, 'Standard surface code'),
        ('Heavy-Hex', 3, 4, 'IBM approach'),
        ('Hexagonal', 3, 6, 'Honeycomb'),
        ('Triangular', 6, 3, 'High connectivity'),
        ('Floquet', 3, 2, 'Dynamic code'),
        ('Bacon-Shor', 2, 2, 'Subsystem code'),
    ]

    names = [l[0] for l in lattices]
    connectivity = [l[1] for l in lattices]
    stabilizer_weight = [l[2] for l in lattices]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, connectivity, width, label='Qubit Connectivity', color='steelblue')
    bars2 = ax.bar(x + width/2, stabilizer_weight, width, label='Stabilizer Weight', color='coral')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Connectivity vs. Stabilizer Weight Trade-off', fontsize=14)
    ax.legend()

    # Add annotations
    for i, lattice in enumerate(lattices):
        ax.annotate(lattice[3], (i, max(lattice[1], lattice[2]) + 0.5),
                   ha='center', fontsize=8, style='italic')

    ax.set_ylim(0, 8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    print("Generating lattice geometry visualizations...")

    # Create and visualize different lattices
    lattices = [
        SquareLattice(5),
        HexagonalLattice(5),
        TriangularLattice(5),
        HeavyHexLattice(4),
    ]

    for lattice in lattices:
        fig, ax = visualize_lattice(lattice)
        filename = f'{lattice.name.lower()}_lattice.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()

    # Comparison plots
    fig1 = compare_geometries()
    plt.savefig('geometry_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved geometry_comparison.png")

    fig2 = visualize_stabilizers_comparison()
    plt.savefig('stabilizer_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved stabilizer_comparison.png")

    fig3 = analyze_connectivity()
    plt.savefig('connectivity_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved connectivity_analysis.png")

    # Summary table
    print("\n" + "=" * 70)
    print("Lattice Geometry Comparison Summary")
    print("=" * 70)
    print(f"{'Geometry':<15} {'Connectivity':<12} {'Stabilizer Wt':<14} {'Threshold':<12}")
    print("-" * 55)
    print(f"{'Square':<15} {'4':<12} {'4':<14} {'~1.0%':<12}")
    print(f"{'Hexagonal':<15} {'3':<12} {'6':<14} {'~0.7%':<12}")
    print(f"{'Triangular':<15} {'6':<12} {'3':<14} {'~1.0%':<12}")
    print(f"{'Heavy-Hex':<15} {'3':<12} {'4 (flags)':<14} {'~0.8%':<12}")
    print(f"{'Floquet':<15} {'3':<12} {'2 (dynamic)':<14} {'~0.3%':<12}")

    plt.show()
```

### Lab Exercises

1. **Implement a Bacon-Shor code** on a 2D grid with only nearest-neighbor connectivity.

2. **Calculate and plot** the qubit overhead as a function of code distance for all geometries.

3. **Design a hybrid lattice** that transitions from square to hexagonal geometry.

4. **Simulate syndrome extraction** for a weight-6 hexagonal stabilizer and compare circuit depth to weight-4.

---

## Summary

### Key Formulas

| Geometry | Connectivity | Stabilizer Weight | Approximate Threshold |
|----------|--------------|-------------------|----------------------|
| Square | 4 | 4 | ~1.0% |
| Hexagonal | 3 | 6 | ~0.7% |
| Triangular | 6 | 3 | ~1.0% |
| Heavy-hex | 3 | 4 (with flags) | ~0.8% |
| Floquet | 3 | 2 (dynamic) | ~0.3% |

### Main Takeaways

1. **Geometry is a design choice:** Different lattices trade off connectivity, threshold, and overhead.

2. **Heavy-hex reduces crosstalk:** IBM sacrifices some threshold for better fabrication.

3. **Higher connectivity helps:** Generally allows higher thresholds (square, triangular).

4. **Dynamic codes are emerging:** Floquet codes achieve topological protection with simpler measurements.

5. **Hardware dictates choice:** The optimal geometry depends on the underlying physical platform.

---

## Daily Checklist

- [ ] I can describe the differences between square, hexagonal, triangular, and heavy-hex lattices
- [ ] I understand the connectivity-threshold trade-off
- [ ] I can calculate qubit overhead for different geometries
- [ ] I understand why IBM chose the heavy-hex architecture
- [ ] I have run the computational lab and compared lattice geometries

---

## Preview: Day 817

Tomorrow we examine **Ancilla Design and Connectivity**. We'll discover:
- How ancilla qubits are used for syndrome extraction
- 4-way vs. 3-way connectivity requirements
- Flag qubit protocols for fault tolerance
- Syndrome extraction circuit optimization

Ancilla design is the bridge between abstract error correction and physical gate sequences.

---

*"The geometry of a quantum error correcting code is not just mathematics—it is a contract between theory and hardware."*

— Day 816 Reflection
