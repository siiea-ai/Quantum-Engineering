# Day 892: T-Factory Footprint

## Week 128, Day 3 | Month 32: Fault-Tolerant Quantum Computing II

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Factory architectures and distillation protocols |
| Afternoon | 2.5 hours | Problem solving: Factory design optimization |
| Evening | 2 hours | Computational lab: Factory calculator implementation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Analyze the 15-to-1 distillation protocol** including qubit requirements and error suppression
2. **Design multi-level factory architectures** for achieving target output fidelities
3. **Calculate factory production rates** and throughput optimization
4. **Minimize area-time products** for factory configurations
5. **Compare different factory designs** (15-to-1, 20-to-4, catalyzed protocols)
6. **Implement factory footprint calculators** for resource estimation

---

## Core Content

### 1. Magic State Distillation Fundamentals

#### Why T-Factories?

In the surface code, Clifford gates are "free" (transversal or via lattice surgery), but T-gates require **magic states**:

$$|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

Raw magic states have errors. Distillation purifies them:

$$p_{out} \ll p_{in}$$

#### The Distillation Trade-off

Every distillation protocol trades:
- **Input states** (noisy) → **Output states** (cleaner)
- Space (qubits for distillation circuit)
- Time (cycles to complete distillation)

### 2. The 15-to-1 Protocol

#### Protocol Overview

The canonical 15-to-1 protocol:
- **Input**: 15 noisy T-states with error $p$
- **Output**: 1 clean T-state with error $\sim 35p^3$
- **Overhead**: $15 \times$ per level

#### Error Suppression

The output error scales as:

$$\boxed{p_{out} = 35p_{in}^3}$$

For $p_{in} = 10^{-3}$:
$$p_{out} = 35 \times (10^{-3})^3 = 3.5 \times 10^{-8}$$

A single level achieves 5 orders of magnitude improvement!

#### Footprint Analysis

The 15-to-1 protocol requires a specific surface code layout:

```
Level 1 Factory Layout (schematic):
┌──────────────────────────────────┐
│  T1  T2  T3  T4  T5              │
│  T6  T7  T8  T9  T10   [Output]  │
│  T11 T12 T13 T14 T15             │
│         [Verification]            │
└──────────────────────────────────┘
```

**Dimensions** (in code distance units):
- Width: $12d$
- Height: $6d$
- Area: $72d^2$ physical qubits

More precisely, accounting for routing:

$$\boxed{A_{15-to-1} = 12d \times 6d = 72d^2}$$

#### Time Cost

Distillation time per level:

$$t_{distill} = 8 \times d \text{ code cycles}$$

For $d = 17$: $t_{distill} = 136$ cycles

With 1 μs cycle time: 136 μs per T-state (single factory).

### 3. Multi-Level Factories

#### Achieving Ultra-Low Error Rates

For algorithms requiring $p_{out} < 10^{-15}$, cascade multiple levels:

**Two-Level Factory:**
$$p_{out}^{(2)} = 35(35p_{in}^3)^3 = 35^4 p_{in}^9$$

For $p_{in} = 10^{-3}$:
$$p_{out}^{(2)} = 35^4 \times 10^{-27} \approx 1.5 \times 10^{-21}$$

**Three-Level Factory:**
$$p_{out}^{(3)} = 35^{13} p_{in}^{27}$$

#### Level Stacking

Each level feeds the next:

```
Level 1: 15 raw → 1 clean (p ≈ 10⁻⁸)
    ↓ (need 15 outputs from level 1)
Level 2: 15 × L1 → 1 ultra-clean (p ≈ 10⁻²¹)
```

#### Area Accumulation

For $k$ levels:

$$A_{total} = \sum_{i=1}^{k} 15^{i-1} \times A_{level_i}$$

**Two-level factory:**
$$A^{(2)} = A_1 + 15 \times A_1 \approx 16 \times 72d^2 \approx 1152d^2$$

Wait—this is wrong! The levels don't need to run simultaneously at full capacity.

#### Pipelined Factories

With pipelining, we only need enough level-1 factories to keep level-2 fed:

$$A_{pipelined}^{(2)} = A_2 + \frac{t_2}{t_1} \times 15 \times A_1$$

If $t_2 \approx t_1$ (same distillation time):

$$A_{pipelined}^{(2)} \approx A_2 + 15A_1 = 16 \times 72d^2 \approx 150d^2$$

**Key insight**: Pipelining dramatically reduces footprint compared to parallel duplication.

### 4. Factory Design Variations

#### 20-to-4 Protocol

An alternative protocol:
- **Input**: 20 noisy T-states
- **Output**: 4 clean T-states
- **Error**: $p_{out} \approx 4p_{in}^2$ (weaker suppression but better rate)

**Comparison:**

| Protocol | Input | Output | Suppression | Area | Time |
|----------|-------|--------|-------------|------|------|
| 15-to-1 | 15 | 1 | $35p^3$ | $72d^2$ | $8d$ |
| 20-to-4 | 20 | 4 | $4p^2$ | $90d^2$ | $6d$ |
| 116-to-12 | 116 | 12 | $\sim p^4$ | Large | $\sim 10d$ |

#### Catalyzed Protocols

Advanced protocols use **catalysis**—a clean state helps distill others:

**CCZ Factory** (produces CCZ states directly):
- Area: $\sim 100d^2$
- Time: $\sim 10d$
- Produces CCZ (equivalent to 4 T-gates)

**T-state with CCZ catalysis:**
- Can convert CCZ + 4 T-states → 7 T-states efficiently

### 5. Production Rate Optimization

#### Single Factory Throughput

For a 15-to-1 factory:

$$r_{factory} = \frac{1}{t_{distill}} = \frac{1}{8d} \text{ T-states per cycle}$$

For $d = 17$: $r = 1/136 \approx 0.0074$ T-states/cycle

#### Multiple Factories

With $n$ factories operating in parallel:

$$r_{total} = n \times r_{factory} = \frac{n}{8d}$$

#### Pipeline Optimization

In a pipelined multi-level factory, the bottleneck is the slowest level:

$$r_{pipelined} = \min_i \left( \frac{\text{capacity}_i}{t_i} \right)$$

For matched pipelines: $r \approx 1/(8d)$ per complete factory.

### 6. Area-Time Product Optimization

#### The AT Product

The area-time product measures factory efficiency:

$$AT = A_{factory} \times t_{distill}$$

Lower AT means more efficient resource usage.

**15-to-1 factory:**
$$AT = 72d^2 \times 8d = 576d^3$$

#### Comparing Factory Designs

| Design | Area ($d^2$) | Time ($d$) | AT ($d^3$) | Output Error |
|--------|-------------|-----------|-----------|--------------|
| 15-to-1 (1 level) | 72 | 8 | 576 | $35p^3$ |
| 15-to-1 (2 level) | 150 | 16 | 2400 | $35^4p^9$ |
| 20-to-4 | 90 | 6 | 540 | $4p^2$ |
| Fast 15-to-1 | 200 | 4 | 800 | $35p^3$ |

**Trade-off**: Faster factories require more area, but AT product varies.

### 7. Factory Placement Strategy

#### Uniform Distribution

Distribute factories evenly across the logical qubit array to minimize routing:

```
┌────┬────┬────┬────┐
│ Q  │ Q  │ Q  │ F  │
├────┼────┼────┼────┤
│ Q  │ Q  │ Q  │ Q  │
├────┼────┼────┼────┤
│ F  │ Q  │ Q  │ Q  │
├────┼────┼────┼────┤
│ Q  │ Q  │ F  │ Q  │
└────┴────┴────┴────┘
Q = Logical qubit, F = Factory
```

**Optimization criterion:**
$$\min \sum_i d_i \times n_i$$
where $d_i$ = distance to nearest factory, $n_i$ = T-gates at location $i$.

#### Centralized vs. Distributed

| Strategy | Advantages | Disadvantages |
|----------|------------|---------------|
| Centralized | Simpler layout, less complexity | Long routing distances |
| Distributed | Short routing, balanced load | Complex scheduling |
| Hybrid | Best of both worlds | Design complexity |

---

## Practical Benchmarks

### Factory Requirements for Key Algorithms

| Algorithm | T-count | Target Error | Factory Levels | Factories Needed |
|-----------|---------|--------------|----------------|------------------|
| RSA-2048 | $2 \times 10^{10}$ | $10^{-15}$ | 2 | 28 |
| FeMoco | $10^{12}$ | $10^{-18}$ | 2-3 | 100+ |
| QAOA-100 | $10^6$ | $10^{-10}$ | 1 | 2-4 |
| Shor-4096 | $10^{11}$ | $10^{-15}$ | 2 | 50+ |

### Physical Qubit Cost Breakdown

For RSA-2048 with 28 two-level factories ($d = 27$):

$$A_{factories} = 28 \times 150 \times 27^2 = 3,061,800 \text{ qubits}$$

This represents ~15% of the total 20M qubit count.

### Production Rate Requirements

For RSA-2048 in 8 hours:

$$r_{required} = \frac{2 \times 10^{10}}{8 \times 3600 \times 10^6 / \mu s} = 0.69 \text{ T-states}/\mu s$$

With 28 factories, each at rate $1/(8 \times 27) = 0.0046$/cycle = 4600/s:

$$r_{actual} = 28 \times 4600 = 128,800 \text{ T-states/s} = 0.13 \text{ T-states}/\mu s$$

This suggests the 8-hour estimate assumes faster factories or higher parallelism.

---

## Worked Examples

### Example 1: Single-Level Factory Design

**Problem:** Design a factory to produce T-states with error < $10^{-8}$ from raw states with error $10^{-3}$. Calculate area and production rate for $d = 15$.

**Solution:**

Step 1: Check if single level suffices
$$p_{out} = 35p_{in}^3 = 35 \times (10^{-3})^3 = 3.5 \times 10^{-8}$$

This meets the $10^{-8}$ requirement. Single level is sufficient.

Step 2: Calculate area
$$A = 72d^2 = 72 \times 225 = 16,200 \text{ physical qubits}$$

Step 3: Calculate production rate
$$t_{distill} = 8d = 120 \text{ cycles}$$

With 1 μs/cycle:
$$r = \frac{1}{120 \mu s} = 8,333 \text{ T-states/second}$$

$$\boxed{A = 16,200 \text{ qubits}, \quad r = 8,333 \text{ T/s}}$$

---

### Example 2: Two-Level Factory Sizing

**Problem:** An algorithm needs T-states with error < $10^{-20}$. Raw error is $10^{-3}$. Design a two-level factory and calculate total footprint.

**Solution:**

Step 1: Verify two levels achieve target
$$p_{out}^{(2)} = 35^4 \times p_{in}^9 = 1.5 \times 10^6 \times 10^{-27} = 1.5 \times 10^{-21}$$

Yes, this exceeds the $10^{-20}$ requirement.

Step 2: Level 1 area (15 factories to feed level 2)
$$A_1 = 15 \times 72d^2 = 1080d^2$$

Step 3: Level 2 area (1 factory)
$$A_2 = 72d^2$$

Step 4: Total with pipelining consideration

If levels have equal time, we can pipeline:
- Level 2 needs 15 states per output
- Level 1 produces 1 state per $8d$ cycles
- Need 15 level-1 factories running in parallel

Actually, with smart pipelining:
$$A_{total} = 15 \times A_1^{single} + A_2 = 15 \times 72d^2 + 72d^2 = 1152d^2$$

For $d = 17$:
$$A_{total} = 1152 \times 289 = 332,928 \text{ physical qubits}$$

$$\boxed{A_{total} = 1152d^2 = 332,928 \text{ qubits (for } d=17\text{)}}$$

---

### Example 3: Factory Count for Runtime Target

**Problem:** An algorithm has T-count $5 \times 10^9$. Target runtime is 2 hours. Each factory (2-level) has area $150d^2$ and produces one T-state per 200 cycles. Code distance is 17, cycle time 1 μs. How many factories are needed?

**Solution:**

Step 1: Convert runtime to cycles
$$T_{available} = 2 \times 3600 \times 10^6 = 7.2 \times 10^9 \text{ cycles}$$

Step 2: Required production rate
$$r_{needed} = \frac{5 \times 10^9}{7.2 \times 10^9} = 0.694 \text{ T-states/cycle}$$

Step 3: Per-factory production rate
$$r_{factory} = \frac{1}{200} = 0.005 \text{ T-states/cycle}$$

Step 4: Number of factories
$$n = \frac{r_{needed}}{r_{factory}} = \frac{0.694}{0.005} = 139$$

Step 5: Factory qubit cost
$$Q_{factory} = 139 \times 150 \times 17^2 = 6,025,350 \text{ physical qubits}$$

$$\boxed{n = 139 \text{ factories}, \quad Q = 6.0 \times 10^6 \text{ qubits}}$$

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the area of a 15-to-1 factory for code distances $d = 11, 15, 21, 27$.

**Problem 1.2:** A factory produces one T-state per 100 cycles. How many T-states can it produce in 1 second with 1 μs cycle time?

**Problem 1.3:** What output error does a two-level 15-to-1 factory achieve with input error $5 \times 10^{-4}$?

### Level 2: Intermediate Analysis

**Problem 2.1:** Compare the area-time products of:
- Three 15-to-1 factories running in parallel
- One fast factory with $A = 200d^2$, $t = 4d$
Both should produce ~3 T-states per production cycle.

**Problem 2.2:** An algorithm uses $10^8$ T-gates and $10^7$ CCZ gates. Compare:
- Strategy A: Produce all as T-states (CCZ = 4 T's)
- Strategy B: Use dedicated CCZ factory producing 1 CCZ per 150 cycles
Which requires fewer factory qubits?

**Problem 2.3:** Derive the optimal number of level-1 factories to feed a single level-2 factory, assuming:
- Level-1 time: $t_1 = 8d$
- Level-2 time: $t_2 = 10d$ (slightly slower due to cleaner input)
- Level-2 needs 15 inputs per output

### Level 3: Challenging Problems

**Problem 3.1:** **Factory Placement Optimization**
Given an 8×8 grid of logical qubits with T-gate density $\rho(x, y) = \rho_0 e^{-((x-4)^2 + (y-4)^2)/8}$ (concentrated in center), find optimal placement for 4 factories.

**Problem 3.2:** **Multi-Level Optimization**
For target output error $\epsilon$, input error $p$, derive the optimal number of distillation levels $k$ that minimizes total factory area.

**Problem 3.3:** **Dynamic Factory Allocation**
An algorithm has two phases:
- Phase 1: High T-rate ($10^8$/hour)
- Phase 2: Low T-rate ($10^6$/hour)
Design a reconfigurable factory system that adapts to the phase, minimizing average qubit usage.

---

## Computational Lab

### T-Factory Design Calculator

```python
"""
Day 892: T-Factory Footprint Calculator
Design and analyze magic state distillation factories.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class DistillationProtocol(Enum):
    """Available distillation protocols."""
    FIFTEEN_TO_ONE = "15-to-1"
    TWENTY_TO_FOUR = "20-to-4"
    CCZ_FACTORY = "CCZ"


@dataclass
class FactoryLevel:
    """Single level of a distillation factory."""
    protocol: DistillationProtocol
    input_count: int
    output_count: int
    error_suppression_exp: float  # p_out ∝ p_in^exp
    error_coefficient: float
    area_coefficient: float  # in d² units
    time_coefficient: float  # in d units

    def output_error(self, input_error: float) -> float:
        """Calculate output error from input error."""
        return self.error_coefficient * (input_error ** self.error_suppression_exp)

    def area(self, code_distance: int) -> int:
        """Calculate area in physical qubits."""
        return int(self.area_coefficient * code_distance ** 2)

    def time(self, code_distance: int) -> int:
        """Calculate distillation time in code cycles."""
        return int(self.time_coefficient * code_distance)


# Standard protocols
PROTOCOLS = {
    DistillationProtocol.FIFTEEN_TO_ONE: FactoryLevel(
        protocol=DistillationProtocol.FIFTEEN_TO_ONE,
        input_count=15,
        output_count=1,
        error_suppression_exp=3,
        error_coefficient=35,
        area_coefficient=72,
        time_coefficient=8
    ),
    DistillationProtocol.TWENTY_TO_FOUR: FactoryLevel(
        protocol=DistillationProtocol.TWENTY_TO_FOUR,
        input_count=20,
        output_count=4,
        error_suppression_exp=2,
        error_coefficient=4,
        area_coefficient=90,
        time_coefficient=6
    ),
    DistillationProtocol.CCZ_FACTORY: FactoryLevel(
        protocol=DistillationProtocol.CCZ_FACTORY,
        input_count=8,
        output_count=1,  # 1 CCZ = 4 T-equivalents
        error_suppression_exp=2,
        error_coefficient=10,
        area_coefficient=100,
        time_coefficient=10
    )
}


@dataclass
class MultiLevelFactory:
    """Multi-level distillation factory."""
    levels: List[FactoryLevel]
    code_distance: int
    input_error: float

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def output_error(self) -> float:
        """Calculate final output error through all levels."""
        error = self.input_error
        for level in self.levels:
            error = level.output_error(error)
        return error

    def total_area(self, pipelined: bool = True) -> int:
        """
        Calculate total factory area.

        If pipelined, accounts for overlapping execution.
        """
        if not pipelined or self.num_levels == 1:
            # Simple case: all level-1 factories running
            total = 0
            multiplier = 1
            for level in self.levels:
                total += multiplier * level.area(self.code_distance)
                multiplier *= level.input_count
            return total

        # Pipelined: only need enough to keep pipeline fed
        total = 0
        for i, level in enumerate(self.levels):
            if i == 0:
                # Level 1: need input_count of level 2
                if self.num_levels > 1:
                    n_factories = self.levels[1].input_count
                else:
                    n_factories = 1
            else:
                n_factories = 1

            total += n_factories * level.area(self.code_distance)
        return total

    def production_time(self) -> int:
        """Calculate time to produce one output state."""
        # Pipeline latency dominated by sum of all levels
        return sum(level.time(self.code_distance) for level in self.levels)

    def production_rate(self) -> float:
        """Calculate output rate in states per cycle."""
        # In steady state, rate limited by slowest level
        max_time = max(level.time(self.code_distance) for level in self.levels)
        return 1.0 / max_time

    def area_time_product(self) -> float:
        """Calculate AT product (efficiency metric)."""
        return self.total_area() * self.production_time()

    def summary(self) -> dict:
        """Generate complete factory summary."""
        return {
            'num_levels': self.num_levels,
            'code_distance': self.code_distance,
            'input_error': self.input_error,
            'output_error': self.output_error(),
            'total_area_qubits': self.total_area(),
            'production_time_cycles': self.production_time(),
            'production_rate': self.production_rate(),
            'at_product': self.area_time_product()
        }


def design_factory(
    target_error: float,
    input_error: float,
    code_distance: int,
    protocol: DistillationProtocol = DistillationProtocol.FIFTEEN_TO_ONE
) -> MultiLevelFactory:
    """
    Design a multi-level factory to achieve target error.

    Returns factory with minimum levels needed.
    """
    base_level = PROTOCOLS[protocol]
    levels = []
    current_error = input_error

    while current_error > target_error:
        levels.append(base_level)
        current_error = base_level.output_error(current_error)

        if len(levels) > 10:  # Safety limit
            raise ValueError("Cannot achieve target error in reasonable levels")

    return MultiLevelFactory(levels=levels, code_distance=code_distance, input_error=input_error)


def calculate_factories_needed(
    t_count: int,
    runtime_cycles: int,
    factory: MultiLevelFactory
) -> int:
    """Calculate number of factories needed to meet runtime target."""
    rate_per_factory = factory.production_rate()
    required_rate = t_count / runtime_cycles
    return int(np.ceil(required_rate / rate_per_factory))


class FactoryFloorPlanner:
    """Plan factory placement on a 2D grid."""

    def __init__(self, grid_width: int, grid_height: int):
        self.width = grid_width
        self.height = grid_height
        self.grid = np.zeros((grid_height, grid_width))
        self.factories = []
        self.logical_qubits = []

    def add_factory(self, x: int, y: int, width: int, height: int, factory_id: int):
        """Add a factory to the floor plan."""
        self.factories.append({
            'id': factory_id,
            'x': x, 'y': y,
            'width': width, 'height': height
        })
        self.grid[y:y+height, x:x+width] = factory_id + 100

    def add_logical_qubit(self, x: int, y: int, qubit_id: int):
        """Add a logical qubit to the floor plan."""
        self.logical_qubits.append({'id': qubit_id, 'x': x, 'y': y})
        self.grid[y, x] = qubit_id

    def calculate_routing_distance(self, qubit_id: int) -> float:
        """Calculate minimum Manhattan distance from qubit to nearest factory."""
        qubit = next(q for q in self.logical_qubits if q['id'] == qubit_id)
        qx, qy = qubit['x'], qubit['y']

        min_dist = float('inf')
        for factory in self.factories:
            # Distance to factory center
            fx = factory['x'] + factory['width'] // 2
            fy = factory['y'] + factory['height'] // 2
            dist = abs(qx - fx) + abs(qy - fy)
            min_dist = min(min_dist, dist)

        return min_dist

    def average_routing_distance(self) -> float:
        """Calculate average routing distance across all qubits."""
        if not self.logical_qubits:
            return 0.0
        distances = [self.calculate_routing_distance(q['id']) for q in self.logical_qubits]
        return np.mean(distances)

    def visualize(self, title: str = "Factory Floor Plan"):
        """Visualize the floor plan."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw grid
        cmap = plt.cm.Set3
        im = ax.imshow(self.grid, cmap=cmap, origin='lower', alpha=0.7)

        # Draw factory outlines
        for factory in self.factories:
            rect = plt.Rectangle(
                (factory['x'] - 0.5, factory['y'] - 0.5),
                factory['width'], factory['height'],
                fill=False, edgecolor='red', linewidth=3
            )
            ax.add_patch(rect)
            ax.text(
                factory['x'] + factory['width']/2,
                factory['y'] + factory['height']/2,
                f"F{factory['id']}",
                ha='center', va='center', fontsize=12, fontweight='bold', color='red'
            )

        # Mark logical qubits
        for qubit in self.logical_qubits:
            ax.plot(qubit['x'], qubit['y'], 'ko', markersize=8)
            ax.text(qubit['x'] + 0.3, qubit['y'] + 0.3, f"Q{qubit['id']}", fontsize=8)

        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xlabel('X position', fontsize=12)
        ax.set_ylabel('Y position', fontsize=12)
        ax.set_title(f'{title}\nAvg routing distance: {self.average_routing_distance():.2f}', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('factory_floor_plan.png', dpi=150, bbox_inches='tight')
        plt.show()


def compare_protocols():
    """Compare different distillation protocols."""
    input_error = 1e-3
    code_distance = 17

    protocols = [
        DistillationProtocol.FIFTEEN_TO_ONE,
        DistillationProtocol.TWENTY_TO_FOUR,
    ]

    print("\n" + "="*70)
    print("DISTILLATION PROTOCOL COMPARISON")
    print("="*70)
    print(f"Input error: {input_error:.0e}, Code distance: {code_distance}")
    print("-"*70)

    for protocol in protocols:
        level = PROTOCOLS[protocol]
        print(f"\n{protocol.value}:")
        print(f"  Input/Output: {level.input_count} → {level.output_count}")
        print(f"  Output error: {level.output_error(input_error):.2e}")
        print(f"  Area: {level.area(code_distance):,} qubits")
        print(f"  Time: {level.time(code_distance)} cycles")
        print(f"  AT product: {level.area(code_distance) * level.time(code_distance):,}")


def plot_factory_scaling():
    """Visualize how factory requirements scale."""
    distances = [11, 13, 15, 17, 19, 21, 25, 27]
    factory = PROTOCOLS[DistillationProtocol.FIFTEEN_TO_ONE]

    areas = [factory.area(d) for d in distances]
    times = [factory.time(d) for d in distances]
    at_products = [a * t for a, t in zip(areas, times)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Area scaling
    axes[0].plot(distances, areas, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Code Distance', fontsize=12)
    axes[0].set_ylabel('Area (physical qubits)', fontsize=12)
    axes[0].set_title('Factory Area Scaling', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Fit quadratic
    d_fit = np.linspace(10, 28, 100)
    a_fit = 72 * d_fit**2
    axes[0].plot(d_fit, a_fit, 'r--', label='$72d^2$', linewidth=1.5)
    axes[0].legend()

    # Time scaling
    axes[1].plot(distances, times, 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Code Distance', fontsize=12)
    axes[1].set_ylabel('Time (cycles)', fontsize=12)
    axes[1].set_title('Distillation Time Scaling', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # AT product
    axes[2].semilogy(distances, at_products, 'm-o', linewidth=2, markersize=8)
    axes[2].set_xlabel('Code Distance', fontsize=12)
    axes[2].set_ylabel('AT Product', fontsize=12)
    axes[2].set_title('Area-Time Product Scaling', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('factory_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_multilevel_comparison():
    """Compare 1, 2, and 3 level factories."""
    input_error = 1e-3
    code_distance = 17

    levels_range = [1, 2, 3]
    target_errors = []
    areas = []
    times = []

    for num_levels in levels_range:
        factory = MultiLevelFactory(
            levels=[PROTOCOLS[DistillationProtocol.FIFTEEN_TO_ONE]] * num_levels,
            code_distance=code_distance,
            input_error=input_error
        )
        target_errors.append(factory.output_error())
        areas.append(factory.total_area())
        times.append(factory.production_time())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Error reduction
    axes[0].semilogy(levels_range, target_errors, 'r-o', linewidth=2, markersize=10)
    axes[0].set_xlabel('Number of Levels', fontsize=12)
    axes[0].set_ylabel('Output Error', fontsize=12)
    axes[0].set_title('Error Suppression vs. Levels', fontsize=14)
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].set_xticks(levels_range)

    # Area growth
    axes[1].bar(levels_range, areas, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Number of Levels', fontsize=12)
    axes[1].set_ylabel('Total Area (qubits)', fontsize=12)
    axes[1].set_title('Factory Area vs. Levels', fontsize=14)
    axes[1].set_xticks(levels_range)

    # Time growth
    axes[2].bar(levels_range, times, color='darkorange', edgecolor='black')
    axes[2].set_xlabel('Number of Levels', fontsize=12)
    axes[2].set_ylabel('Production Time (cycles)', fontsize=12)
    axes[2].set_title('Production Time vs. Levels', fontsize=14)
    axes[2].set_xticks(levels_range)

    plt.tight_layout()
    plt.savefig('multilevel_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main demonstration
if __name__ == "__main__":
    print("T-Factory Footprint Calculator - Day 892")
    print("="*50)

    # Example 1: Design factory for RSA-2048
    print("\n--- Example: RSA-2048 Factory Design ---")
    target_error = 1e-15
    input_error = 1e-3
    code_distance = 27

    factory = design_factory(target_error, input_error, code_distance)
    summary = factory.summary()

    print(f"Target error: {target_error:.0e}")
    print(f"Designed factory: {summary['num_levels']} levels")
    print(f"Achieved error: {summary['output_error']:.2e}")
    print(f"Total area: {summary['total_area_qubits']:,} qubits")
    print(f"Production time: {summary['production_time_cycles']} cycles")
    print(f"AT product: {summary['at_product']:,.0f}")

    # Calculate factories needed for RSA-2048
    t_count = int(2e10)
    runtime_hours = 8
    runtime_cycles = runtime_hours * 3600 * int(1e6)  # 1 μs cycles

    n_factories = calculate_factories_needed(t_count, runtime_cycles, factory)
    print(f"\nTo execute in {runtime_hours} hours:")
    print(f"  Factories needed: {n_factories}")
    print(f"  Total factory qubits: {n_factories * summary['total_area_qubits']:,}")

    # Protocol comparison
    compare_protocols()

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_factory_scaling()
    plot_multilevel_comparison()

    # Floor plan example
    print("\n--- Factory Floor Plan Example ---")
    planner = FactoryFloorPlanner(20, 20)

    # Add factories
    planner.add_factory(0, 0, 4, 3, 1)
    planner.add_factory(16, 0, 4, 3, 2)
    planner.add_factory(0, 17, 4, 3, 3)
    planner.add_factory(16, 17, 4, 3, 4)

    # Add logical qubits
    for i in range(8):
        for j in range(8):
            planner.add_logical_qubit(5 + i, 5 + j, i * 8 + j + 1)

    planner.visualize("RSA-2048 Factory Layout (64 qubits, 4 factories)")

    print("\nFactory analysis complete!")
```

---

## Summary

### Key Formulas

| Formula | Expression | Description |
|---------|------------|-------------|
| 15-to-1 error | $p_{out} = 35p_{in}^3$ | Single level suppression |
| Factory area | $A = 72d^2$ | 15-to-1 footprint |
| Distillation time | $t = 8d$ | Cycles per T-state |
| Two-level error | $p_{out}^{(2)} = 35^4 p_{in}^9$ | Cascaded suppression |
| AT product | $AT = 576d^3$ | Efficiency metric |

### Factory Design Guidelines

| Target Error | Levels | Area | Notes |
|--------------|--------|------|-------|
| $10^{-8}$ | 1 | $72d^2$ | Basic applications |
| $10^{-15}$ | 2 | $~150d^2$ | Cryptographic |
| $10^{-25}$ | 3 | $~300d^2$ | Extreme precision |

### Key Insights

1. **Error suppression is rapid**: Each level cubes the error (for 15-to-1)
2. **Area scales as $d^2$**: Code distance has outsized impact on footprint
3. **Pipelining is essential**: Reduces area dramatically vs. parallel duplication
4. **Factory placement matters**: Uniform distribution minimizes routing overhead

---

## Daily Checklist

- [ ] I understand the 15-to-1 distillation protocol
- [ ] I can calculate factory area as $72d^2$
- [ ] I can design multi-level factories for target error rates
- [ ] I understand pipelining and its benefits
- [ ] I can calculate production rates and throughput
- [ ] I can compare different factory protocols
- [ ] I can use the factory calculator tool

---

## Preview: Day 893

Tomorrow we tackle **Runtime Analysis**:

- T-count dominated execution time
- Factory throughput as bottleneck
- Benchmark: RSA-2048 runtime breakdown
- Benchmark: Quantum chemistry simulations
- Optimization strategies for practical runtimes

We'll connect factory design to overall algorithm execution time, answering: "How long will this actually take?"

---

*Day 892 of 2184 | Week 128 of 312 | Month 32 of 72*

*"The factory is the engine of fault-tolerant computation—its throughput determines everything."*
