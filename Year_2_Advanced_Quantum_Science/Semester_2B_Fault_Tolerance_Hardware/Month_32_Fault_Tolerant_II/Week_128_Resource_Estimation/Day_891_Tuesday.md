# Day 891: Space-Time Volume Analysis

## Week 128, Day 2 | Month 32: Fault-Tolerant Quantum Computing II

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Litinski's space-time framework |
| Afternoon | 2.5 hours | Problem solving: Volume optimization |
| Evening | 2 hours | Computational lab: Trade-off analysis tools |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Define and calculate space-time volume** for fault-tolerant quantum computations
2. **Apply Litinski's framework** for analyzing surface code computations
3. **Identify space-time trade-offs** and optimize for different constraints
4. **Understand the "Game of Surface Codes"** methodology for algorithm compilation
5. **Analyze parallelization strategies** and their volume implications
6. **Compare different compilation strategies** using volume metrics

---

## Core Content

### 1. The Space-Time Volume Concept

#### Fundamental Definition

The **space-time volume** of a quantum computation is:

$$\boxed{V = A \times T}$$

where:
- $A$ = spatial footprint (number of physical qubits)
- $T$ = temporal extent (number of code cycles)

This product represents the total "computational resources" consumed, analogous to the area under a curve in classical complexity analysis.

#### Why Volume Matters

Space-time volume captures a fundamental insight: **qubits and time are fungible resources**. We can often trade:

- More qubits for less time (parallelization)
- Less qubits for more time (serialization)

The minimum achievable volume is a property of the algorithm itself, independent of the execution strategy.

#### Units and Scaling

Standard units in surface code analysis:

| Quantity | Unit | Description |
|----------|------|-------------|
| Space | $d \times d$ patches | One logical qubit footprint |
| Time | Code cycles | One round of syndrome measurement |
| Volume | $(d \times d) \times \text{cycles}$ | Patch-cycles |

### 2. Litinski's Framework

Daniel Litinski's 2019 paper "A Game of Surface Codes" revolutionized how we think about surface code computations.

#### Key Principles

**Principle 1: Compact Data Blocks**
Logical qubits are represented as $d \times d$ surface code patches. Multiple patches tile a 2D plane.

**Principle 2: Lattice Surgery for Operations**
All logical operations performed via lattice surgery between adjacent patches.

**Principle 3: Time-Optimal Layout**
Arrange patches to minimize the critical path length through the computation.

#### The Routing Problem

Lattice surgery requires patches to be adjacent. The **routing challenge**:
- Patches must "move" to become adjacent
- Movement is achieved through chains of lattice surgery operations
- Routing consumes both space (intermediate patches) and time

```
Before surgery:          After routing:
┌───┐     ┌───┐         ┌───┐───┌───┐
│ A │     │ B │   →     │ A │ R │ B │
└───┘     └───┘         └───┘───└───┘
                        (R = routing patch)
```

### 3. Volume Decomposition

#### Components of Space-Time Volume

Total volume decomposes into:

$$V_{total} = V_{data} + V_{magic} + V_{routing} + V_{idle}$$

| Component | Description | Typical Fraction |
|-----------|-------------|------------------|
| $V_{data}$ | Logical qubit storage | 20-40% |
| $V_{magic}$ | Magic state factories | 30-50% |
| $V_{routing}$ | Lattice surgery channels | 10-20% |
| $V_{idle}$ | Waiting/synchronization | 5-15% |

#### Data Volume

For $n$ logical qubits over $T$ cycles:

$$V_{data} = n \times d^2 \times T$$

#### Magic State Volume

Factory contribution over algorithm runtime:

$$V_{magic} = n_{factories} \times A_{factory} \times T$$

where $A_{factory}$ is the factory footprint in physical qubits.

### 4. Space-Time Trade-offs

#### The Fundamental Trade-off

For a fixed computation (fixed total work), we can choose:

$$\boxed{A \times T = V_{min} \quad \text{(constant for given algorithm)}}$$

This defines a **hyperbola** in space-time space:

```
Time (T)
    │
    │╲
    │ ╲  V = constant
    │  ╲
    │   ╲___
    │       ╲___
    └──────────────── Space (A)
```

#### Parallelization Factor

Define the parallelization factor $\pi$:

$$\pi = \frac{T_{serial}}{T_{parallel}}$$

With perfect parallelization:
- Space increases by factor $\pi$: $A_{parallel} = \pi \times A_{serial}$
- Time decreases by factor $\pi$: $T_{parallel} = T_{serial} / \pi$
- Volume remains constant: $V = A_{parallel} \times T_{parallel} = A_{serial} \times T_{serial}$

#### Practical Constraints

Real systems face constraints that limit trade-offs:

1. **Minimum qubits**: Cannot reduce below algorithm's logical qubit count
2. **Maximum qubits**: Hardware limits total physical qubits
3. **Minimum time**: Communication latency, factory throughput limits
4. **Maximum time**: Decoherence, practical time limits

$$A_{min} \leq A \leq A_{max}$$
$$T_{min} \leq T \leq T_{max}$$

### 5. Volume Optimization Strategies

#### Strategy 1: Factory Placement Optimization

Place T-factories to minimize routing distance to consumers:

$$V_{routing} \propto \sum_i d_i \times n_i$$

where $d_i$ = distance to factory $i$, $n_i$ = T-gates near factory $i$.

**Optimal placement**: Distribute factories uniformly across the logical qubit array.

#### Strategy 2: Operation Scheduling

Reorder operations to maximize parallelism while respecting dependencies:

```
Before optimization:     After optimization:
T1 → T2 → T3 → T4       T1 ─┬─ T2
                            ├─ T3
Time: 4 units               └─ T4
                        Time: 2 units (if independent)
```

**Scheduling algorithm**: Critical path analysis on the operation DAG.

#### Strategy 3: Compact Layouts

Use space-filling arrangements to minimize routing:

**Linear layout**:
```
┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐
│1│─│2│─│3│─│4│─│5│
└─┘ └─┘ └─┘ └─┘ └─┘
Max routing distance: O(n)
```

**Square layout**:
```
┌─┐ ┌─┐ ┌─┐
│1│─│2│─│3│
└─┘ └─┘ └─┘
 │   │   │
┌─┐ ┌─┐ ┌─┐
│4│─│5│─│6│
└─┘ └─┘ └─┘
Max routing distance: O(√n)
```

#### Strategy 4: Pipeline Factories

Instead of waiting for T-states:

$$T_{wait} = T_{distill} \times \lceil T_{count} / n_{factories} \rceil$$

Pipeline production to hide latency:

$$T_{pipeline} = T_{distill} + (T_{count} - 1) / n_{factories}$$

### 6. The Game of Surface Codes

Litinski's "game" analogy provides intuition for volume optimization:

#### Game Rules

1. **Board**: 2D grid of surface code patches
2. **Pieces**: Logical qubits (data) and factories (generators)
3. **Moves**: Lattice surgery operations
4. **Objective**: Complete computation in minimum time

#### Winning Strategies

**Strategy: Tetris Packing**
Pack operations as tightly as possible, eliminating gaps:

```
Poor packing:          Optimal packing:
█░░█░░█░               ████████
░█░░█░░█               ████████
█░░█░░█░               ████████
```

**Strategy: Critical Path Focus**
Identify and optimize the longest dependency chain:

$$T_{min} = \max_{\text{paths}} \sum_{\text{ops in path}} t_{op}$$

**Strategy: Factory Bandwidth Matching**
Match factory production rate to consumption rate:

$$r_{production} \geq \frac{T_{count}}{T_{available}}$$

---

## Practical Benchmarks

### RSA-2048: Space-Time Analysis

From Gidney-Ekerå (2021):

| Configuration | Physical Qubits | Time | Volume |
|---------------|-----------------|------|--------|
| Minimal space | ~14M | ~16 hours | $2.2 \times 10^{14}$ |
| Balanced | ~20M | ~8 hours | $1.6 \times 10^{14}$ |
| Minimal time | ~50M | ~3 hours | $1.5 \times 10^{14}$ |

**Observation**: ~30% volume reduction possible through optimization, but diminishing returns.

### Volume Comparison Across Algorithms

| Algorithm | Volume (patch-cycles) | Dominant Component |
|-----------|----------------------|-------------------|
| RSA-2048 | $10^{14}$ | Magic states |
| FeMoco | $10^{16}$ | Magic states |
| QAOA-100 | $10^{10}$ | Data storage |
| Grover-1000 | $10^{12}$ | Balanced |

### Volume Scaling Laws

For common quantum algorithms:

| Algorithm | Volume Scaling |
|-----------|---------------|
| Shor's algorithm | $O(n^3 \log n)$ for $n$-bit factoring |
| Grover's search | $O(\sqrt{N} \cdot \log N)$ for $N$-item search |
| Quantum simulation | $O(t^2 / \epsilon)$ for time $t$, error $\epsilon$ |
| QAOA | $O(p \cdot n^2)$ for $p$ layers, $n$ qubits |

---

## Worked Examples

### Example 1: Volume Calculation

**Problem:** Calculate the space-time volume for an algorithm with:
- 200 logical qubits
- Code distance 17
- 10 T-factories, each $150d^2$ area
- Runtime: $10^6$ code cycles

**Solution:**

Step 1: Data qubit space
$$A_{data} = n \times 2d^2 = 200 \times 2 \times 289 = 115,600$$

Step 2: Factory space
$$A_{factory} = 10 \times 150 \times 289 = 433,500$$

Step 3: Routing (40% overhead)
$$A_{routing} = 0.4 \times (115,600 + 433,500) = 219,640$$

Step 4: Total space
$$A_{total} = 115,600 + 433,500 + 219,640 = 768,740$$

Step 5: Volume
$$V = A \times T = 768,740 \times 10^6 = 7.69 \times 10^{11} \text{ qubit-cycles}$$

$$\boxed{V \approx 7.7 \times 10^{11} \text{ qubit-cycles}}$$

---

### Example 2: Space-Time Trade-off

**Problem:** An algorithm has minimum volume $V = 10^{12}$ qubit-cycles. Hardware constraint: maximum 500,000 qubits. What is the minimum runtime?

**Solution:**

From $V = A \times T$:
$$T_{min} = \frac{V}{A_{max}} = \frac{10^{12}}{500,000} = 2 \times 10^6 \text{ cycles}$$

With 1 μs cycle time:
$$T_{min} = 2 \times 10^6 \times 10^{-6} = 2 \text{ seconds}$$

$$\boxed{T_{min} = 2 \times 10^6 \text{ cycles} = 2 \text{ seconds}}$$

Conversely, if we want 1 second runtime:
$$A_{required} = \frac{V}{T} = \frac{10^{12}}{10^6} = 10^6 \text{ qubits}$$

This exceeds hardware limits, so 2 seconds is the achievable minimum.

---

### Example 3: Factory Optimization

**Problem:** An algorithm needs $10^9$ T-gates. Each factory produces 1 T-state per 100 cycles and occupies $40,000$ physical qubits. Minimize total volume subject to completing in $10^7$ cycles.

**Solution:**

Step 1: Minimum factories for timing
$$n_{min} = \frac{T_{count} \times t_{produce}}{T_{available}} = \frac{10^9 \times 100}{10^7} = 10,000$$

This is the minimum to meet the deadline.

Step 2: Volume with $n$ factories
$$V(n) = n \times A_{factory} \times T(n)$$
where $T(n) = \frac{10^9 \times 100}{n}$ (assumes T-gate limited)

$$V(n) = n \times 40,000 \times \frac{10^{11}}{n} = 4 \times 10^{15}$$

**Key insight**: Volume is constant with respect to factory count (for T-limited algorithms)!

$$\boxed{V = 4 \times 10^{15} \text{ qubit-cycles, } n_{factories} \geq 10,000}$$

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the space-time volume for:
- 100 logical qubits, d=13
- 5 factories at $150d^2$ each
- Runtime: $5 \times 10^5$ cycles
- 30% routing overhead

**Problem 1.2:** An algorithm has volume $5 \times 10^{10}$ qubit-cycles. If we have 200,000 qubits available, what is the minimum runtime?

**Problem 1.3:** Two implementations have the same volume but different configurations:
- Config A: 1M qubits, 1000 seconds
- Config B: 100K qubits, 10000 seconds
Which is preferable if hardware limit is 500K qubits?

### Level 2: Intermediate Analysis

**Problem 2.1:** An algorithm can be parallelized with efficiency $\eta(\pi) = 1 - 0.1\log(\pi)$ where $\pi$ is the parallelization factor. Find the optimal $\pi$ that minimizes runtime given a qubit budget of 1M and serial volume $10^{15}$.

**Problem 2.2:** Compare two factory strategies:
- Strategy A: 20 factories, $30,000$ qubits each, 50 cycles per T-state
- Strategy B: 5 factories, $100,000$ qubits each, 20 cycles per T-state
Which has better volume efficiency for producing $10^8$ T-states?

**Problem 2.3:** Derive the relationship between factory count $n_f$, total runtime $T$, and algorithm T-count $N_T$ when factories are the bottleneck.

### Level 3: Challenging Problems

**Problem 3.1:** **Optimal Layout Design**
For 64 logical qubits and 4 factories, design a 2D layout that minimizes maximum routing distance. Prove your layout is optimal or near-optimal.

**Problem 3.2:** **Volume Lower Bound**
Prove that for an algorithm with $N_T$ T-gates, each requiring volume $V_T$, and with $k$ independent parallel sections of $N_T/k$ T-gates each, the minimum achievable volume is:
$$V_{min} = N_T \cdot V_T \cdot f(k)$$
and find $f(k)$.

**Problem 3.3:** **Multi-Objective Optimization**
Formulate the resource optimization as a linear program:
- Minimize: $\alpha \cdot A + \beta \cdot T$ (weighted objective)
- Subject to: $A \times T \geq V_{min}$, $A \leq A_{max}$, $T \leq T_{max}$
Solve for the Pareto frontier.

---

## Computational Lab

### Space-Time Volume Analyzer

```python
"""
Day 891: Space-Time Volume Analysis Tool
Litinski's framework implementation for surface code optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.optimize import minimize_scalar, minimize

@dataclass
class VolumeComponents:
    """Breakdown of space-time volume components."""
    data_volume: float
    factory_volume: float
    routing_volume: float
    idle_volume: float

    @property
    def total(self) -> float:
        return self.data_volume + self.factory_volume + self.routing_volume + self.idle_volume


@dataclass
class AlgorithmSpec:
    """Specification of a quantum algorithm for resource analysis."""
    n_logical: int
    t_count: int
    circuit_depth: int
    parallelization_factor: float = 1.0
    name: str = "Algorithm"


class SpaceTimeAnalyzer:
    """
    Analyze space-time trade-offs for fault-tolerant quantum computation.
    Implements Litinski's framework.
    """

    def __init__(
        self,
        code_distance: int = 17,
        factory_area: float = 150,  # in d² units
        cycles_per_t_state: int = 100,
        routing_overhead: float = 0.4
    ):
        self.d = code_distance
        self.d_squared = code_distance ** 2
        self.factory_area = factory_area * self.d_squared
        self.cycles_per_t = cycles_per_t_state
        self.routing_overhead = routing_overhead

    def calculate_volume(
        self,
        algo: AlgorithmSpec,
        n_factories: int,
        include_breakdown: bool = False
    ) -> VolumeComponents:
        """
        Calculate space-time volume for an algorithm configuration.

        Returns volume breakdown by component.
        """
        # Calculate runtime (T-limited)
        t_production_rate = n_factories / self.cycles_per_t  # T-states per cycle
        runtime_cycles = algo.t_count / t_production_rate

        # Apply parallelization to non-T operations
        effective_depth = algo.circuit_depth / algo.parallelization_factor

        # Total runtime is max of T-limited and depth-limited
        total_cycles = max(runtime_cycles, effective_depth)

        # Space components
        data_space = algo.n_logical * 2 * self.d_squared * algo.parallelization_factor
        factory_space = n_factories * self.factory_area

        # Routing overhead
        routing_space = self.routing_overhead * (data_space + factory_space)

        # Volume components
        data_vol = data_space * total_cycles
        factory_vol = factory_space * total_cycles
        routing_vol = routing_space * total_cycles

        # Idle volume (synchronization overhead, ~5%)
        idle_vol = 0.05 * (data_vol + factory_vol)

        return VolumeComponents(
            data_volume=data_vol,
            factory_volume=factory_vol,
            routing_volume=routing_vol,
            idle_volume=idle_vol
        )

    def optimize_factories(
        self,
        algo: AlgorithmSpec,
        max_qubits: int,
        min_factories: int = 1,
        max_factories: int = 1000
    ) -> Tuple[int, float, float]:
        """
        Find optimal number of factories to minimize volume.

        Returns: (optimal_factories, min_volume, runtime)
        """
        best_volume = float('inf')
        best_n = min_factories
        best_runtime = 0

        for n in range(min_factories, max_factories + 1):
            vol = self.calculate_volume(algo, n)

            # Check qubit constraint
            total_qubits = (
                algo.n_logical * 2 * self.d_squared +
                n * self.factory_area +
                self.routing_overhead * (algo.n_logical * 2 * self.d_squared + n * self.factory_area)
            )

            if total_qubits > max_qubits:
                continue

            if vol.total < best_volume:
                best_volume = vol.total
                best_n = n
                best_runtime = algo.t_count * self.cycles_per_t / n

        return best_n, best_volume, best_runtime

    def space_time_tradeoff_curve(
        self,
        algo: AlgorithmSpec,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the space-time trade-off curve.

        Returns: (qubit_counts, runtimes, volumes)
        """
        factories_range = np.linspace(1, 500, n_points).astype(int)
        factories_range = np.unique(factories_range)

        qubits = []
        runtimes = []
        volumes = []

        for n_f in factories_range:
            vol = self.calculate_volume(algo, n_f)

            # Calculate space
            data_space = algo.n_logical * 2 * self.d_squared
            factory_space = n_f * self.factory_area
            routing_space = self.routing_overhead * (data_space + factory_space)
            total_space = data_space + factory_space + routing_space

            # Calculate time
            runtime = algo.t_count * self.cycles_per_t / n_f

            qubits.append(total_space)
            runtimes.append(runtime)
            volumes.append(vol.total)

        return np.array(qubits), np.array(runtimes), np.array(volumes)


def plot_space_time_tradeoff(analyzer: SpaceTimeAnalyzer, algo: AlgorithmSpec):
    """Visualize the space-time trade-off curve."""
    qubits, runtimes, volumes = analyzer.space_time_tradeoff_curve(algo)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Space vs Time (hyperbola)
    axes[0].loglog(qubits, runtimes, 'b-', linewidth=2)
    axes[0].set_xlabel('Physical Qubits', fontsize=12)
    axes[0].set_ylabel('Runtime (cycles)', fontsize=12)
    axes[0].set_title(f'Space-Time Trade-off: {algo.name}', fontsize=14)
    axes[0].grid(True, alpha=0.3, which='both')

    # Mark key points
    min_vol_idx = np.argmin(volumes)
    axes[0].scatter([qubits[min_vol_idx]], [runtimes[min_vol_idx]],
                    color='red', s=100, zorder=5, label='Min Volume')
    axes[0].legend()

    # Volume vs Factories
    factories = np.linspace(1, 500, len(volumes)).astype(int)
    axes[1].semilogy(factories[:len(volumes)], volumes, 'g-', linewidth=2)
    axes[1].set_xlabel('Number of Factories', fontsize=12)
    axes[1].set_ylabel('Volume (qubit-cycles)', fontsize=12)
    axes[1].set_title('Volume vs Factory Count', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=min(volumes), color='r', linestyle='--', label='Minimum')
    axes[1].legend()

    # Volume breakdown at optimal point
    vol = analyzer.calculate_volume(algo, 100)
    labels = ['Data', 'Factory', 'Routing', 'Idle']
    sizes = [vol.data_volume, vol.factory_volume, vol.routing_volume, vol.idle_volume]
    colors = ['steelblue', 'darkorange', 'forestgreen', 'gray']

    axes[2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[2].set_title('Volume Breakdown (100 factories)', fontsize=14)

    plt.tight_layout()
    plt.savefig('space_time_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_algorithms():
    """Compare space-time requirements across benchmark algorithms."""
    analyzer = SpaceTimeAnalyzer(code_distance=17)

    algorithms = [
        AlgorithmSpec(n_logical=6189, t_count=int(2e10), circuit_depth=int(2e10), name="RSA-2048"),
        AlgorithmSpec(n_logical=4000, t_count=int(1e12), circuit_depth=int(1e12), name="FeMoco"),
        AlgorithmSpec(n_logical=100, t_count=int(1e6), circuit_depth=int(1e4), name="QAOA-100"),
        AlgorithmSpec(n_logical=1000, t_count=int(1e8), circuit_depth=int(1e6), name="Grover-1000"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, algo in enumerate(algorithms):
        ax = axes[idx // 2, idx % 2]
        qubits, runtimes, volumes = analyzer.space_time_tradeoff_curve(algo)

        ax.loglog(qubits, runtimes, 'b-', linewidth=2)
        ax.set_xlabel('Physical Qubits', fontsize=11)
        ax.set_ylabel('Runtime (cycles)', fontsize=11)
        ax.set_title(f'{algo.name}', fontsize=13)
        ax.grid(True, alpha=0.3, which='both')

        # Annotate min volume point
        min_idx = np.argmin(volumes)
        ax.scatter([qubits[min_idx]], [runtimes[min_idx]], color='red', s=80, zorder=5)
        ax.annotate(f'V={volumes[min_idx]:.1e}',
                   xy=(qubits[min_idx], runtimes[min_idx]),
                   xytext=(10, 10), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def parallelization_analysis():
    """Analyze the effect of parallelization on volume."""
    analyzer = SpaceTimeAnalyzer(code_distance=15)
    base_algo = AlgorithmSpec(n_logical=100, t_count=int(1e8), circuit_depth=int(1e6))

    parallel_factors = [1, 2, 4, 8, 16, 32]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    runtimes = []
    qubits = []
    volumes = []

    for p in parallel_factors:
        algo = AlgorithmSpec(
            n_logical=base_algo.n_logical,
            t_count=base_algo.t_count,
            circuit_depth=base_algo.circuit_depth,
            parallelization_factor=p
        )
        vol = analyzer.calculate_volume(algo, n_factories=50)

        # Approximate space scaling with parallelization
        space = (base_algo.n_logical * p * 2 * analyzer.d_squared +
                50 * analyzer.factory_area) * (1 + analyzer.routing_overhead)

        # Time scales inversely (ideal)
        time = base_algo.circuit_depth / p

        qubits.append(space)
        runtimes.append(time)
        volumes.append(vol.total)

    # Runtime and qubits vs parallelization
    ax1 = axes[0]
    ax1.plot(parallel_factors, runtimes, 'b-o', label='Runtime', linewidth=2)
    ax1.set_xlabel('Parallelization Factor', fontsize=12)
    ax1.set_ylabel('Runtime (cycles)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(parallel_factors, qubits, 'r-s', label='Qubits', linewidth=2)
    ax1_twin.set_ylabel('Physical Qubits', fontsize=12, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    ax1.set_title('Space-Time Trade-off with Parallelization', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Volume vs parallelization
    axes[1].plot(parallel_factors, volumes, 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Parallelization Factor', fontsize=12)
    axes[1].set_ylabel('Volume (qubit-cycles)', fontsize=12)
    axes[1].set_title('Volume vs Parallelization', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parallelization_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


class LitinskiGameSimulator:
    """
    Simulate the 'Game of Surface Codes' for visualization.
    """

    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.operations = []

    def place_logical_qubit(self, x: int, y: int, label: int):
        """Place a logical qubit at position (x, y)."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = label

    def lattice_surgery(self, x1: int, y1: int, x2: int, y2: int, op_type: str = "ZZ"):
        """Record a lattice surgery operation."""
        self.operations.append({
            'type': op_type,
            'pos1': (x1, y1),
            'pos2': (x2, y2)
        })

    def visualize(self, title: str = "Surface Code Layout"):
        """Visualize the current game state."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(y=i, color='gray', linewidth=0.5)
            ax.axvline(x=i, color='gray', linewidth=0.5)

        # Draw patches
        cmap = plt.cm.Set3
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] > 0:
                    color = cmap(int(self.grid[y, x]) % 12)
                    rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x + 0.5, y + 0.5, f'Q{int(self.grid[y, x])}',
                           ha='center', va='center', fontsize=12, fontweight='bold')

        # Draw operations
        for op in self.operations:
            x1, y1 = op['pos1']
            x2, y2 = op['pos2']
            ax.annotate('', xy=(x2 + 0.5, y2 + 0.5), xytext=(x1 + 0.5, y1 + 0.5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X position (d units)', fontsize=12)
        ax.set_ylabel('Y position (d units)', fontsize=12)

        plt.tight_layout()
        plt.savefig('game_of_surface_codes.png', dpi=150, bbox_inches='tight')
        plt.show()


# Demonstration
if __name__ == "__main__":
    print("Space-Time Volume Analysis - Day 891")
    print("="*50)

    # Create analyzer
    analyzer = SpaceTimeAnalyzer(code_distance=17)

    # Example: RSA-2048
    rsa = AlgorithmSpec(
        n_logical=6189,
        t_count=int(2e10),
        circuit_depth=int(2e10),
        name="RSA-2048"
    )

    print("\n--- RSA-2048 Analysis ---")
    vol = analyzer.calculate_volume(rsa, n_factories=28)
    print(f"Volume breakdown:")
    print(f"  Data:    {vol.data_volume:.2e} ({100*vol.data_volume/vol.total:.1f}%)")
    print(f"  Factory: {vol.factory_volume:.2e} ({100*vol.factory_volume/vol.total:.1f}%)")
    print(f"  Routing: {vol.routing_volume:.2e} ({100*vol.routing_volume/vol.total:.1f}%)")
    print(f"  Idle:    {vol.idle_volume:.2e} ({100*vol.idle_volume/vol.total:.1f}%)")
    print(f"  TOTAL:   {vol.total:.2e}")

    # Optimization
    print("\n--- Factory Optimization ---")
    opt_n, opt_vol, opt_runtime = analyzer.optimize_factories(rsa, max_qubits=25_000_000)
    print(f"Optimal factories: {opt_n}")
    print(f"Minimum volume: {opt_vol:.2e}")
    print(f"Runtime: {opt_runtime:.2e} cycles")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_space_time_tradeoff(analyzer, rsa)
    compare_algorithms()
    parallelization_analysis()

    # Game visualization
    game = LitinskiGameSimulator(grid_size=8)
    for i in range(4):
        for j in range(4):
            game.place_logical_qubit(i*2, j*2, i*4 + j + 1)
    game.lattice_surgery(0, 0, 2, 0)
    game.lattice_surgery(0, 2, 0, 4)
    game.visualize("Example Surface Code Layout")

    print("\nAnalysis complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula | Description |
|---------|---------|-------------|
| Space-Time Volume | $V = A \times T$ | Fundamental resource measure |
| Volume Decomposition | $V = V_{data} + V_{magic} + V_{routing} + V_{idle}$ | Component breakdown |
| Trade-off Constraint | $A \times T \geq V_{min}$ | Minimum achievable volume |
| Parallelization | $A_p = \pi A_s$, $T_p = T_s/\pi$ | Space-time trade-off |

### Litinski's Framework Key Points

1. **Volume is conserved**: Trade-offs between space and time preserve total volume
2. **Factories dominate**: For T-heavy algorithms, factory volume exceeds data volume
3. **Routing matters**: Poor layouts can add 50%+ overhead
4. **Optimization pays**: Careful scheduling reduces volume 20-30%

### Practical Guidelines

| Constraint | Strategy |
|------------|----------|
| Limited qubits | Minimize factories, accept longer runtime |
| Limited time | Maximize factories within qubit budget |
| Balanced | Optimize for minimum total volume |

---

## Daily Checklist

- [ ] I understand space-time volume as $V = A \times T$
- [ ] I can decompose volume into data, factory, routing components
- [ ] I understand the space-time trade-off hyperbola
- [ ] I can apply Litinski's framework to analyze algorithms
- [ ] I can optimize factory count for given constraints
- [ ] I understand parallelization effects on volume
- [ ] I can use the space-time analyzer tool

---

## Preview: Day 892

Tomorrow we dive deep into **T-Factory Footprint** analysis:

- 15-to-1 and 20-to-1 distillation protocols
- Multi-level factory architectures
- Production rate optimization
- Area-time product minimization
- Factory design calculator

Factories are the "engines" of fault-tolerant computation—understanding their design is essential for practical resource estimation.

---

*Day 891 of 2184 | Week 128 of 312 | Month 32 of 72*

*"In the game of surface codes, you either optimize your volume or run out of qubits."*
