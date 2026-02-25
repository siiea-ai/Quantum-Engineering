# Day 824: Multi-Patch Architectures

## Week 118: Lattice Surgery & Logical Gates | Month 30: Surface Codes

### Semester 2A: Error Correction | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Patch layouts, routing algorithms, parallelization strategies |
| **Afternoon** | 2.5 hours | Architecture optimization, practice problems |
| **Evening** | 1.5 hours | Computational lab: Layout simulation and scheduling |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 824, you will be able to:

1. **Design multi-patch layouts** for surface code processors
2. **Route operations** between non-adjacent patches using ancilla channels
3. **Schedule parallel operations** to maximize throughput
4. **Analyze defect-based architectures** as alternatives to patches
5. **Calculate space-time trade-offs** for different layout strategies
6. **Optimize layouts** for specific quantum algorithms

---

## 1. Introduction: Scaling Up Lattice Surgery

### From Single Gates to Algorithms

Previous days covered single operations:
- Merge/split on two patches
- CNOT using three patches
- State preparation

**Real algorithms require:**
- Many logical qubits ($n = 100$ to $10^6$)
- Many gates ($g = 10^3$ to $10^{12}$)
- Parallel execution for reasonable runtime

$$\boxed{\text{Architecture Design} = \text{Patch Layout} + \text{Routing} + \text{Scheduling}}$$

### Key Constraints

**1. Locality:** Lattice surgery only works between adjacent patches
**2. Connectivity:** Must route operations between any pair of logical qubits
**3. Parallelism:** Execute independent gates simultaneously
**4. Overhead:** Minimize ancilla and routing qubits

---

## 2. Patch Layout Strategies

### Linear Array

The simplest layout: patches arranged in a line.

```
┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐
│ 0 │──│ 1 │──│ 2 │──│ 3 │──│ 4 │
└───┘  └───┘  └───┘  └───┘  └───┘
```

**Advantages:**
- Simple routing
- Minimal overhead

**Disadvantages:**
- Only nearest-neighbor connectivity
- Long routes for distant qubits: $O(n)$ hops

**Connectivity:** Average distance = $n/3$ for $n$ qubits

### 2D Grid Layout

Patches arranged on a square lattice.

```
┌───┐  ┌───┐  ┌───┐  ┌───┐
│ 0 │──│ 1 │──│ 2 │──│ 3 │
└─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘
  │      │      │      │
┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
│ 4 │──│ 5 │──│ 6 │──│ 7 │
└─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘
  │      │      │      │
┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
│ 8 │──│ 9 │──│10 │──│11 │
└───┘  └───┘  └───┘  └───┘
```

**Advantages:**
- Better connectivity: $O(\sqrt{n})$ average distance
- Natural fit for 2D qubit arrays

**Disadvantages:**
- Routing conflicts possible
- Requires careful scheduling

### Interleaved Data-Ancilla Layout

Alternate data and ancilla patches for routing:

```
┌───┐  ┌───┐  ┌───┐  ┌───┐
│ D │──│ A │──│ D │──│ A │
└─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘
  │      │      │      │
┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
│ A │──│ R │──│ A │──│ R │
└─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘
  │      │      │      │
┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
│ D │──│ A │──│ D │──│ A │
└───┘  └───┘  └───┘  └───┘

D = Data patch, A = Ancilla patch, R = Routing channel
```

**Overhead:** 50% ancilla patches for full connectivity

$$\boxed{\text{2D Grid: } N_{\text{physical}} = 4d^2 \times n_{\text{logical}} \text{ (with routing)}}$$

---

## 3. Routing Algorithms

### The Routing Problem

**Input:** Source patch $S$, target patch $T$, operation type (ZZ or XX merge)

**Output:** Sequence of patches to traverse, boundary alignments

**Constraints:**
- Only adjacent patches can merge
- Boundary types must match (rough-rough for ZZ, smooth-smooth for XX)

### SWAP-Based Routing

Move logical qubits through the array using SWAP gates:

$$\text{SWAP} = \text{CNOT}_{A\to B} \cdot \text{CNOT}_{B\to A} \cdot \text{CNOT}_{A\to B}$$

**Cost per SWAP:** 3 CNOT = 9d cycles

**Total routing cost:** $O(d_{\text{Manhattan}}) \times 9d$ cycles

### Teleportation-Based Routing

Use pre-shared Bell pairs to teleport state:

1. Create Bell pair between source and destination
2. Measure source qubit jointly with data qubit
3. Apply corrections at destination

**Cost:** 2d cycles (Bell pair creation) + 2d cycles (measurement/correction)

**Advantage:** Faster than SWAP for distances > 2 hops

$$\boxed{t_{\text{teleport}} = 4d, \quad t_{\text{SWAP}} = 9d \times d_{\text{hop}}}$$

### Comparison

| Distance (hops) | SWAP Time | Teleport Time | Better |
|----------------|-----------|---------------|--------|
| 1 | 9d | 4d | Teleport |
| 2 | 18d | 4d | Teleport |
| 3+ | 27d+ | 4d | Teleport |

Teleportation wins for any distance > 0!

---

## 4. Parallel Gate Scheduling

### Gate Dependencies

Not all gates can execute in parallel:
- Gates on same qubit: Must be sequential
- Gates on different qubits: Can be parallel

**Dependency graph:** DAG where edges represent ordering constraints

### Layer-Based Scheduling

Group gates into **layers** of non-conflicting operations:

```
Layer 1: CNOT(0,1), CNOT(2,3), CNOT(4,5)
Layer 2: CNOT(1,2), CNOT(3,4)
Layer 3: CNOT(0,3), H(5)
...
```

**Circuit depth:** Number of layers

**Quantum volume limited by:** $\text{depth} \times d$ cycles

### Scheduling with Routing

**Challenge:** Routing operations also consume patches and time

**Greedy algorithm:**
1. Identify all executable gates (no dependencies)
2. Attempt to route each gate
3. If routing conflicts: delay or find alternative path
4. Execute non-conflicting gates in parallel
5. Repeat

### Time-Optimal vs. Space-Optimal Scheduling

**Time-optimal:** Maximize parallelism, use more ancilla patches
**Space-optimal:** Minimize qubits, accept longer runtime

$$\boxed{\text{Trade-off: } T \times S \approx \text{constant} \times \text{circuit volume}}$$

---

## 5. Defect-Based Architectures

### Beyond Patch-Based Layouts

**Alternative:** Use a single large surface code with **defects** (holes).

**Defect = hole:** Region where stabilizers are not measured

Logical qubits are encoded in defect pairs:
- **Primal defect pair:** Encodes using rough boundaries
- **Dual defect pair:** Encodes using smooth boundaries

### Defect Operations

**Logical X:** String connecting defects (rough)
**Logical Z:** String connecting defects (smooth)

**Gates via braiding:**
- Move defects around each other
- Topological operation implements logical gates

```
    ○ Defect 1
    │
    │   Move around →
    │
    ○ Defect 2
```

### Advantages of Defect-Based

1. **More compact:** No wasted space between patches
2. **Flexible routing:** Defects can move anywhere
3. **Natural parallelism:** Multiple braiding operations

### Disadvantages

1. **Complex scheduling:** Defect paths can conflict
2. **Variable distance:** Code distance changes during movement
3. **Implementation complexity:** More sophisticated control

$$\boxed{\text{Defect density: } \rho = \frac{n_{\text{logical}}}{L^2} \leq \frac{1}{4d^2}}$$

---

## 6. Worked Examples

### Example 1: Layout for 4-Qubit Circuit

**Problem:** Design a patch layout for 4 logical qubits that can execute CNOT between any pair with at most 2 hops.

**Solution:**

**Layout:** 2×2 grid with dedicated ancilla

```
┌───┐     ┌───┐
│ 0 │─────│ 1 │
└─┬─┘     └─┬─┘
  │    A    │
  │  ┌───┐  │
  └──│ * │──┘
     └─┬─┘
  ┌────┴────┐
┌─┴─┐     ┌─┴─┐
│ 2 │─────│ 3 │
└───┘     └───┘
```

**Connectivity check:**
- 0↔1: 1 hop (direct)
- 0↔2: 1 hop (direct)
- 0↔3: 2 hops (via 1 or 2)
- 1↔2: 2 hops (via 0 or 3)
- 1↔3: 1 hop (direct)
- 2↔3: 1 hop (direct)

**Ancilla needed:** Central patch (*) for routing 0↔3 and 1↔2

$$\boxed{\text{4-qubit layout: 5 patches (4 data + 1 routing)}}$$

---

### Example 2: Parallel CNOT Scheduling

**Problem:** Schedule the gates: CNOT(0,1), CNOT(2,3), CNOT(1,2), CNOT(0,3) on a linear array.

**Solution:**

**Linear layout:** 0 - 1 - 2 - 3

**Dependency analysis:**
- CNOT(0,1) and CNOT(2,3): No conflict → Layer 1
- CNOT(1,2): Depends on CNOT(0,1) completing (qubit 1 busy) → Layer 2
- CNOT(0,3): Needs routing (not adjacent) → After routing setup

**Schedule:**

| Layer | Operations | Time |
|-------|-----------|------|
| 1 | CNOT(0,1), CNOT(2,3) | 3d |
| 2 | CNOT(1,2) | 3d |
| 3 | SWAP(0,1) | 9d |
| 4 | CNOT(1,3) [was 0,3] | 3d |
| 5 | SWAP(0,1) [restore] | 9d |

**Total time:** 27d cycles

**Alternative (teleportation):**
| Layer | Operations | Time |
|-------|-----------|------|
| 1 | CNOT(0,1), CNOT(2,3), Bell(0-3) | 3d |
| 2 | CNOT(1,2), Teleport finish(0→3') | 3d |
| 3 | CNOT(0',3) [teleported] | 3d |

**Total time:** 9d cycles (much faster!)

$$\boxed{\text{Teleportation reduces 27d → 9d for this circuit}}$$

---

### Example 3: Resource Estimation for Shor's Algorithm

**Problem:** Estimate physical qubits needed for Shor's algorithm factoring a 2048-bit number using surface codes.

**Solution:**

**Logical qubits needed:** ~4000 (for 2048-bit factoring)
**T gates:** ~$10^{10}$
**Code distance required:** $d \approx 27$ (for $10^{-15}$ logical error rate)

**Physical qubits per logical qubit:**
- Data patch: $2d^2 = 2(729) = 1458$ qubits
- With routing: $\approx 3000$ qubits per logical qubit

**Total physical qubits:**
$$N_{\text{physical}} = 4000 \times 3000 = 12 \text{ million qubits}$$

**Runtime:**
- T gate time: ~1000d cycles (with magic state distillation)
- Total T gates: $10^{10}$
- Parallelism: ~100 T gates in parallel
- Total cycles: $10^{10} / 100 \times 1000 \times 27 \approx 3 \times 10^{12}$ cycles
- At 1 MHz cycle rate: ~35 days

$$\boxed{\text{Shor's 2048-bit: ~12M qubits, ~35 days runtime}}$$

---

## 7. Practice Problems

### Problem Set A: Direct Application

**A1.** Calculate the number of patches needed for a 10-qubit fully connected layout using the interleaved grid strategy.

**A2.** For qubits arranged in a 5×5 grid, what is the maximum Manhattan distance between any pair?

**A3.** How many SWAP gates are needed to route from position (0,0) to (4,4) in a 5×5 grid?

---

### Problem Set B: Intermediate

**B1.** Design a layout for 8 logical qubits that minimizes the number of ancilla patches while ensuring any CNOT requires at most 2 hops.

**B2.** Given the circuit: CNOT(0,2), CNOT(1,3), CNOT(0,1), CNOT(2,3), schedule it on a linear 4-qubit array. What is the minimum depth?

**B3.** Compare space-time volume for executing 100 random CNOTs on:
(a) A 10-qubit linear array
(b) A 4×3 grid with routing patches

---

### Problem Set C: Challenging

**C1.** Prove that any n-qubit circuit can be executed on a linear array with at most O(n) SWAP overhead per gate.

**C2.** Design an algorithm to dynamically route operations in a defect-based architecture, avoiding collisions between moving defects.

**C3.** For a given quantum algorithm, derive the optimal aspect ratio (width vs. height) of a 2D grid that minimizes total space-time volume.

---

## 8. Computational Lab: Multi-Patch Layout Simulation

```python
"""
Day 824 Computational Lab: Multi-Patch Architecture Simulation
Layout design, routing, and scheduling for surface code processors

This lab implements tools for designing and analyzing multi-patch
layouts for lattice surgery quantum computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, deque
import heapq

class SurfaceCodeLayout:
    """
    Represents a 2D layout of surface code patches.

    Handles:
    - Patch placement
    - Routing between patches
    - Operation scheduling
    """

    def __init__(self, width, height, patch_distance=5):
        """
        Initialize layout grid.

        Parameters:
        -----------
        width, height : int
            Grid dimensions
        patch_distance : int
            Code distance for patches
        """
        self.width = width
        self.height = height
        self.d = patch_distance

        # Grid: 0 = empty, 'D' = data, 'A' = ancilla, 'R' = routing
        self.grid = [[None for _ in range(width)] for _ in range(height)]

        # Track data qubit locations
        self.data_qubits = {}  # qubit_id -> (row, col)
        self.ancilla_patches = []

    def place_data_patch(self, qubit_id, row, col):
        """Place a data qubit patch at given position."""
        if self.grid[row][col] is not None:
            raise ValueError(f"Position ({row}, {col}) already occupied")

        self.grid[row][col] = ('D', qubit_id)
        self.data_qubits[qubit_id] = (row, col)

    def place_ancilla_patch(self, row, col):
        """Place an ancilla/routing patch."""
        if self.grid[row][col] is not None:
            raise ValueError(f"Position ({row}, {col}) already occupied")

        self.grid[row][col] = ('A', len(self.ancilla_patches))
        self.ancilla_patches.append((row, col))

    def get_neighbors(self, row, col):
        """Get adjacent grid positions."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbors.append((nr, nc))
        return neighbors

    def manhattan_distance(self, q1, q2):
        """Calculate Manhattan distance between two qubits."""
        r1, c1 = self.data_qubits[q1]
        r2, c2 = self.data_qubits[q2]
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, q1, q2):
        """
        Find routing path between two qubits using BFS.

        Returns list of (row, col) positions.
        """
        start = self.data_qubits[q1]
        end = self.data_qubits[q2]

        # BFS
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (row, col), path = queue.popleft()

            if (row, col) == end:
                return path

            for nr, nc in self.get_neighbors(row, col):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))

        return None  # No path found

    def calculate_routing_cost(self, q1, q2, method='teleport'):
        """
        Calculate cost of operation between two qubits.

        Parameters:
        -----------
        q1, q2 : int
            Qubit IDs
        method : str
            'swap' or 'teleport'

        Returns:
        --------
        cycles : int
            Number of syndrome cycles
        """
        distance = self.manhattan_distance(q1, q2)

        if distance == 1:
            # Adjacent: direct merge
            return 3 * self.d  # One CNOT

        if method == 'swap':
            # SWAP-based routing
            swaps_needed = distance - 1
            return swaps_needed * 9 * self.d + 3 * self.d

        else:  # teleport
            # Teleportation: constant time regardless of distance
            return 4 * self.d  # Bell pair + measurement + correction

    def physical_qubits(self):
        """Calculate total physical qubits in layout."""
        occupied = sum(1 for row in self.grid for cell in row if cell is not None)
        qubits_per_patch = 2 * self.d ** 2
        return occupied * qubits_per_patch


def create_linear_layout(n_qubits, distance=5):
    """Create a linear array layout."""
    layout = SurfaceCodeLayout(n_qubits, 1, distance)
    for i in range(n_qubits):
        layout.place_data_patch(i, 0, i)
    return layout


def create_grid_layout(rows, cols, distance=5, interleaved=False):
    """
    Create a 2D grid layout.

    Parameters:
    -----------
    rows, cols : int
        Grid dimensions
    distance : int
        Code distance
    interleaved : bool
        If True, alternate data and ancilla patches (checkerboard)
    """
    if interleaved:
        layout = SurfaceCodeLayout(cols * 2 - 1, rows * 2 - 1, distance)
        qubit_id = 0
        for i in range(rows):
            for j in range(cols):
                layout.place_data_patch(qubit_id, i * 2, j * 2)
                qubit_id += 1
        # Fill ancilla patches
        for i in range(rows * 2 - 1):
            for j in range(cols * 2 - 1):
                if layout.grid[i][j] is None:
                    layout.place_ancilla_patch(i, j)
    else:
        layout = SurfaceCodeLayout(cols, rows, distance)
        qubit_id = 0
        for i in range(rows):
            for j in range(cols):
                layout.place_data_patch(qubit_id, i, j)
                qubit_id += 1

    return layout


def visualize_layout(layout, title="Surface Code Layout", highlight_path=None):
    """
    Visualize a surface code layout.

    Parameters:
    -----------
    layout : SurfaceCodeLayout
        Layout to visualize
    title : str
        Plot title
    highlight_path : list
        Path to highlight (list of (row, col) tuples)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw patches
    for row in range(layout.height):
        for col in range(layout.width):
            cell = layout.grid[row][col]
            if cell is not None:
                cell_type, cell_id = cell

                if cell_type == 'D':
                    color = 'lightblue'
                    label = f'Q{cell_id}'
                else:  # Ancilla
                    color = 'lightgray'
                    label = 'A'

                rect = Rectangle((col - 0.4, row - 0.4), 0.8, 0.8,
                                facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(col, row, label, ha='center', va='center',
                       fontsize=10, fontweight='bold')

    # Draw connections
    for row in range(layout.height):
        for col in range(layout.width):
            if layout.grid[row][col] is not None:
                for nr, nc in layout.get_neighbors(row, col):
                    if layout.grid[nr][nc] is not None:
                        # Draw connection line
                        ax.plot([col, nc], [row, nr], 'k-', alpha=0.3, linewidth=1)

    # Highlight path if provided
    if highlight_path:
        for i in range(len(highlight_path) - 1):
            r1, c1 = highlight_path[i]
            r2, c2 = highlight_path[i + 1]
            ax.annotate('', xy=(c2, r2), xytext=(c1, r1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlim(-0.5, layout.width - 0.5)
    ax.set_ylim(-0.5, layout.height - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


class GateScheduler:
    """
    Schedule gates on a surface code layout.
    """

    def __init__(self, layout):
        """
        Initialize scheduler with a layout.

        Parameters:
        -----------
        layout : SurfaceCodeLayout
            The physical layout
        """
        self.layout = layout
        self.schedule = []  # List of layers, each layer is a list of gates

    def schedule_circuit(self, gates):
        """
        Schedule a list of gates into layers.

        Parameters:
        -----------
        gates : list
            List of (gate_type, qubit1, qubit2) tuples

        Returns:
        --------
        schedule : list
            List of layers, each containing non-conflicting gates
        """
        # Build dependency graph
        n_gates = len(gates)
        dependencies = [set() for _ in range(n_gates)]

        # Gate i depends on gate j if they share a qubit and j < i
        qubit_last_gate = {}
        for i, (gtype, q1, q2) in enumerate(gates):
            for q in [q1, q2]:
                if q is not None and q in qubit_last_gate:
                    dependencies[i].add(qubit_last_gate[q])
                if q is not None:
                    qubit_last_gate[q] = i

        # Schedule using list scheduling
        scheduled = [False] * n_gates
        layers = []

        while not all(scheduled):
            # Find all ready gates (no unscheduled dependencies)
            ready = []
            for i in range(n_gates):
                if not scheduled[i]:
                    if all(scheduled[j] for j in dependencies[i]):
                        ready.append(i)

            if not ready:
                raise ValueError("Circular dependency detected")

            # Greedily assign to layer (avoiding qubit conflicts)
            layer = []
            used_qubits = set()

            for i in ready:
                gtype, q1, q2 = gates[i]
                qubits = {q for q in [q1, q2] if q is not None}

                if not qubits & used_qubits:
                    layer.append((i, gates[i]))
                    used_qubits |= qubits
                    scheduled[i] = True

            layers.append(layer)

        self.schedule = layers
        return layers

    def calculate_total_time(self, routing_method='teleport'):
        """
        Calculate total execution time.

        Parameters:
        -----------
        routing_method : str
            'swap' or 'teleport'

        Returns:
        --------
        total_cycles : int
            Total syndrome measurement cycles
        """
        total = 0

        for layer in self.schedule:
            # Time for layer = max time among all gates in layer
            layer_time = 0
            for _, (gtype, q1, q2) in layer:
                if gtype == 'CNOT' and q1 is not None and q2 is not None:
                    gate_time = self.layout.calculate_routing_cost(
                        q1, q2, routing_method)
                else:
                    gate_time = self.layout.d  # Single-qubit gate

                layer_time = max(layer_time, gate_time)

            total += layer_time

        return total


def compare_layouts():
    """Compare different layout strategies."""
    print("\n" + "="*60)
    print("LAYOUT COMPARISON")
    print("="*60)

    n_qubits = 16
    distance = 5

    # Linear layout
    linear = create_linear_layout(n_qubits, distance)

    # 4x4 grid
    grid = create_grid_layout(4, 4, distance, interleaved=False)

    # Interleaved grid
    interleaved = create_grid_layout(4, 4, distance, interleaved=True)

    layouts = [
        ("Linear (16x1)", linear),
        ("Grid (4x4)", grid),
        ("Interleaved (7x7)", interleaved)
    ]

    print("\nPhysical Qubit Count:")
    print("-" * 40)
    for name, layout in layouts:
        n_phys = layout.physical_qubits()
        print(f"  {name}: {n_phys:,} physical qubits")

    # Calculate average routing cost
    print("\nAverage CNOT Cost (cycles):")
    print("-" * 40)
    for name, layout in layouts:
        total_cost_swap = 0
        total_cost_teleport = 0
        n_pairs = 0

        for q1 in layout.data_qubits:
            for q2 in layout.data_qubits:
                if q1 < q2:
                    total_cost_swap += layout.calculate_routing_cost(q1, q2, 'swap')
                    total_cost_teleport += layout.calculate_routing_cost(q1, q2, 'teleport')
                    n_pairs += 1

        avg_swap = total_cost_swap / n_pairs
        avg_teleport = total_cost_teleport / n_pairs
        print(f"  {name}:")
        print(f"    SWAP routing: {avg_swap:.1f} cycles")
        print(f"    Teleport routing: {avg_teleport:.1f} cycles")

    # Visualize layouts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, layout) in zip(axes, layouts):
        visualize_layout(layout, title=name)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig('layout_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nLayout comparison saved to 'layout_comparison.png'")


def scheduling_demo():
    """Demonstrate gate scheduling."""
    print("\n" + "="*60)
    print("GATE SCHEDULING DEMO")
    print("="*60)

    # Create 4x4 grid
    layout = create_grid_layout(4, 4, distance=5)

    # Sample circuit: some CNOTs
    gates = [
        ('CNOT', 0, 1),
        ('CNOT', 2, 3),
        ('CNOT', 4, 5),
        ('CNOT', 6, 7),
        ('CNOT', 1, 2),  # Depends on first CNOT
        ('CNOT', 5, 6),  # Depends on third and fourth
        ('CNOT', 0, 15), # Long-range
        ('CNOT', 3, 12), # Long-range
    ]

    print(f"\nCircuit: {len(gates)} gates")
    for i, (gtype, q1, q2) in enumerate(gates):
        print(f"  Gate {i}: {gtype}({q1}, {q2})")

    scheduler = GateScheduler(layout)
    layers = scheduler.schedule_circuit(gates)

    print(f"\nScheduled into {len(layers)} layers:")
    for i, layer in enumerate(layers):
        gate_str = ", ".join(f"{g[0]}({g[1]},{g[2]})" for _, g in layer)
        print(f"  Layer {i+1}: {gate_str}")

    # Calculate time
    time_swap = scheduler.calculate_total_time('swap')
    time_teleport = scheduler.calculate_total_time('teleport')

    print(f"\nTotal execution time:")
    print(f"  With SWAP routing: {time_swap} cycles")
    print(f"  With teleportation: {time_teleport} cycles")

    # Visualize schedule
    fig, ax = plt.subplots(figsize=(12, 6))

    for layer_idx, layer in enumerate(layers):
        for gate_idx, (orig_idx, (gtype, q1, q2)) in enumerate(layer):
            y = gate_idx
            ax.barh(y, 1, left=layer_idx, height=0.8, alpha=0.7,
                   label=f'{gtype}({q1},{q2})' if layer_idx == 0 else '')
            ax.text(layer_idx + 0.5, y, f'{gtype}({q1},{q2})',
                   ha='center', va='center', fontsize=8)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Gate slot within layer')
    ax.set_title('Gate Schedule Visualization')
    ax.set_xticks(range(len(layers) + 1))
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('gate_schedule.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSchedule visualization saved to 'gate_schedule.png'")


def resource_scaling_analysis():
    """Analyze resource scaling for different architectures."""
    print("\n" + "="*60)
    print("RESOURCE SCALING ANALYSIS")
    print("="*60)

    n_qubits_list = [4, 9, 16, 25, 36, 49, 64, 81, 100]
    distances = [5, 7, 9, 11]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Physical qubits vs logical qubits
    ax = axes[0, 0]
    for d in distances:
        phys = [2 * d**2 * n * 2 for n in n_qubits_list]  # *2 for routing
        ax.semilogy(n_qubits_list, phys, 'o-', label=f'd={d}')
    ax.set_xlabel('Logical Qubits')
    ax.set_ylabel('Physical Qubits')
    ax.set_title('Physical Qubit Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Average CNOT depth for random circuits
    ax = axes[0, 1]
    depths = []
    for n in n_qubits_list:
        side = int(np.sqrt(n))
        layout = create_grid_layout(side, side, distance=5)

        # Simulate random circuit
        n_gates = n * 10
        random_gates = [('CNOT', np.random.randint(n), np.random.randint(n))
                       for _ in range(n_gates)]
        # Ensure distinct qubits
        random_gates = [('CNOT', q1, q2) for q1, q2 in
                       [(np.random.randint(n), np.random.randint(n))
                        for _ in range(n_gates)] if q1 != q2][:n_gates]

        try:
            scheduler = GateScheduler(layout)
            layers = scheduler.schedule_circuit(random_gates)
            depths.append(len(layers))
        except:
            depths.append(n_gates)  # Fallback

    ax.plot(n_qubits_list, depths, 'bo-', markersize=8)
    ax.set_xlabel('Logical Qubits')
    ax.set_ylabel('Circuit Depth (layers)')
    ax.set_title('Circuit Depth for Random 10n-gate Circuits')
    ax.grid(True, alpha=0.3)

    # Space-time volume
    ax = axes[1, 0]
    for d in distances:
        volumes = []
        for n in n_qubits_list:
            phys = 2 * d**2 * n * 2
            # Assume depth ~ 10n for random circuit
            time = 10 * n * 3 * d  # 3d cycles per CNOT
            volumes.append(phys * time)
        ax.semilogy(n_qubits_list, volumes, 'o-', label=f'd={d}')
    ax.set_xlabel('Logical Qubits')
    ax.set_ylabel('Space-Time Volume (qubit-cycles)')
    ax.set_title('Space-Time Volume Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Overhead ratio
    ax = axes[1, 1]
    for d in distances:
        overhead = [2 * d**2 * 2 for n in n_qubits_list]  # Per logical qubit
        ax.plot(n_qubits_list, overhead, 'o-', label=f'd={d}')
    ax.set_xlabel('Logical Qubits')
    ax.set_ylabel('Physical/Logical Qubit Ratio')
    ax.set_title('Physical Qubit Overhead')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Multi-Patch Architecture Resource Scaling', fontsize=14)
    plt.tight_layout()
    plt.savefig('resource_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResource scaling saved to 'resource_scaling.png'")


def main():
    """Run all Day 824 demonstrations."""
    print("Day 824: Multi-Patch Architectures")
    print("="*60)

    # Compare layouts
    compare_layouts()

    # Scheduling demo
    scheduling_demo()

    # Resource scaling
    resource_scaling_analysis()

    # Example: visualize a specific layout with routing
    print("\n" + "="*60)
    print("ROUTING EXAMPLE")
    print("="*60)

    layout = create_grid_layout(4, 4, distance=5)
    path = layout.find_path(0, 15)
    print(f"\nPath from Q0 to Q15: {path}")
    print(f"SWAP cost: {layout.calculate_routing_cost(0, 15, 'swap')} cycles")
    print(f"Teleport cost: {layout.calculate_routing_cost(0, 15, 'teleport')} cycles")

    fig, ax = visualize_layout(layout, "Routing Q0 → Q15", highlight_path=path)
    plt.savefig('routing_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Routing example saved to 'routing_example.png'")

    print("\n" + "="*60)
    print("Day 824 Computational Lab Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 9. Summary

### Key Formulas Table

| Concept | Formula/Expression |
|---------|-------------------|
| Linear array distance | $d_{\text{avg}} = n/3$ |
| 2D grid distance | $d_{\text{avg}} = O(\sqrt{n})$ |
| SWAP routing time | $t = 9d \times d_{\text{hop}} + 3d$ |
| Teleport routing time | $t = 4d$ (constant) |
| Physical qubits (grid) | $N = 4d^2 \times n_{\text{logical}}$ |
| Interleaved overhead | 50-100% additional patches |
| Defect density limit | $\rho \leq 1/(4d^2)$ |
| Circuit depth (random) | $\sim O(n)$ layers for $O(n)$ gates |

### Key Takeaways

1. **Layout choice** significantly affects performance and overhead
2. **Teleportation** beats SWAP routing for any distance > 0
3. **2D grids** offer better connectivity than linear arrays
4. **Interleaved layouts** provide routing channels at 2x qubit cost
5. **Defect-based architectures** can be more compact but harder to schedule
6. **Scheduling** parallelizes independent operations to reduce runtime

---

## 10. Daily Checklist

- [ ] I can design patch layouts for different qubit counts
- [ ] I understand routing strategies (SWAP vs teleportation)
- [ ] I can schedule gates into parallel layers
- [ ] I can calculate space-time volume for a circuit
- [ ] I understand defect-based alternatives to patch layouts
- [ ] I completed the computational lab and analyzed layouts

---

## 11. Preview: Day 825

Tomorrow we tackle **T-Gate Injection and Magic State Integration**:

- The T gate and its role in universal computation
- Magic state distillation protocols
- Integrating T factories into patch architectures
- Resource estimation for T-heavy algorithms

The T gate is the most expensive operation in fault-tolerant quantum computing, and efficient magic state factories are crucial for practical algorithms.

---

*"Architecture is the bridge between theoretical fault tolerance and practical quantum advantage - every routing decision shapes the quantum computer's capability."*
