# Day 850: Distillation Factory Architecture

## Week 122: State Distillation Protocols | Month 31: Fault-Tolerant Quantum Computing I

### Semester 2B: Fault Tolerance & Hardware | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Factory architecture fundamentals, space-time volume analysis |
| **Afternoon** | 2.5 hours | Multi-level factories, pipelining strategies, surface code integration |
| **Evening** | 1.5 hours | Computational lab: Factory design and optimization |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 850, you will be able to:

1. **Design multi-level distillation factories** with optimal space-time trade-offs
2. **Implement pipelining strategies** to maximize throughput
3. **Integrate factories with surface code computation zones** using lattice surgery
4. **Calculate space-time volume** for various factory configurations
5. **Apply the Litinski factory model** for resource estimation
6. **Optimize factory layout** for specific algorithm requirements

---

## 1. Introduction: From Protocol to Factory

### The Challenge

Yesterday we learned the 15-to-1 distillation protocol. But how do we:
- **Run it continuously** to supply magic states on demand?
- **Minimize space** (qubits) while maintaining throughput?
- **Integrate with computation** without bottlenecks?

$$\boxed{\text{Factory} = \text{Protocol} + \text{Scheduling} + \text{Layout}}$$

### Factory Metrics

**Throughput**: Magic states produced per unit time
$$R = \frac{n_{\text{out}}}{T_{\text{cycle}}}$$

**Space**: Physical qubits required
$$Q = n_{\text{patches}} \times q_{\text{per patch}}$$

**Space-time volume**: Resource usage per output
$$V = Q \times T_{\text{cycle}} / n_{\text{out}}$$

**Efficiency**: Ratio to theoretical minimum
$$\eta = \frac{V_{\text{min}}}{V_{\text{actual}}}$$

---

## 2. Single-Level Factory Architecture

### Basic Factory Unit

A single 15-to-1 distillation unit requires:

**Input zone**: 15 surface code patches for noisy magic states
**Distillation zone**: Clifford operations and syndrome measurement
**Output zone**: 1 surface code patch for clean magic state

### Surface Code Patch Size

For code distance $d$:
- Data qubits per patch: $\approx d^2$
- Ancilla qubits per patch: $\approx d^2$
- **Total per patch**: $\approx 2d^2$

**Factory footprint** (15 input + workspace + 1 output):
$$Q_{\text{1-level}} \approx 15 \times 2d^2 + \text{routing} \approx 40d^2$$

### Time Analysis

**Distillation cycle time**:
1. State preparation: $\sim d$ cycles
2. Encoding (CNOT network): $\sim 10d$ cycles
3. Syndrome measurement: $\sim d$ cycles
4. Decoding and output: $\sim d$ cycles

$$T_{\text{1-level}} \approx 13d \text{ cycles}$$

### Space-Time Volume

$$V_{\text{1-level}} = Q_{\text{1-level}} \times T_{\text{1-level}} = 40d^2 \times 13d = 520d^3 \text{ qubit-cycles}$$

$$\boxed{V_{\text{15-to-1}} \approx 500d^3 \text{ qubit-cycles per distilled state}}$$

---

## 3. Multi-Level Factory Design

### Level-2 Factory Structure

For 2-level distillation (target error $\sim 10^{-21}$ from $10^{-3}$ raw):

**Level 1**: 15 units producing 15 intermediate states
**Level 2**: 1 unit consuming level-1 outputs

**Naive approach** (sequential):
$$Q_{\text{2-level, naive}} = 15 \times Q_{\text{level-1}} + Q_{\text{level-2}}$$
$$= 15 \times 40d^2 + 40d^2 = 640d^2$$

$$T_{\text{2-level, naive}} = 15 \times T_{\text{level-1}} + T_{\text{level-2}}$$
$$= 15 \times 13d + 13d = 208d$$

### Pipelined Level-2 Factory

**Key insight**: Run level-1 units in parallel, feeding level-2 as states become ready.

**Parallel level-1**:
- 15 level-1 units run simultaneously
- All produce outputs at approximately the same time
- Level-2 begins immediately when inputs ready

**Space** (parallel):
$$Q_{\text{2-level, parallel}} = 15 \times 40d^2 + 40d^2 = 640d^2$$

**Time** (pipelined):
$$T_{\text{2-level, pipelined}} = T_{\text{level-1}} + T_{\text{level-2}} = 26d$$

**Space-time volume**:
$$V_{\text{2-level}} = 640d^2 \times 26d = 16,640d^3$$

Per output magic state:
$$\boxed{V_{\text{2-level}} \approx 16,600d^3 \text{ qubit-cycles}}$$

### Comparison: Sequential vs. Pipelined

| Configuration | Space ($d^2$) | Time (cycles) | Volume ($d^3$) |
|--------------|---------------|---------------|----------------|
| 1-level | 40 | 13d | 520 |
| 2-level sequential | 640 | 208d | 133,120 |
| 2-level pipelined | 640 | 26d | 16,640 |

$$\boxed{\text{Pipelining gives } 8\times \text{ throughput improvement!}}$$

---

## 4. Factory Throughput Optimization

### Steady-State Production

For continuous magic state production:

**Production rate** (states per cycle):
$$R = \frac{1}{T_{\text{cycle}}}$$

For pipelined 2-level factory:
$$R_{\text{2-level}} = \frac{1}{26d} \text{ states/cycle}$$

### Parallelization for Higher Throughput

If algorithm requires $R_{\text{required}}$ magic states per cycle:

**Number of parallel factories needed:**
$$N_{\text{factories}} = \left\lceil R_{\text{required}} \times T_{\text{cycle}} \right\rceil$$

**Total space:**
$$Q_{\text{total}} = N_{\text{factories}} \times Q_{\text{factory}}$$

### Example: Shor's Algorithm

**Requirements:**
- T-gates: $\sim 10^{10}$
- Execution time: Must complete before coherence loss
- Target: $10^6$ cycles at $d = 13$

**Required rate:**
$$R_{\text{required}} = \frac{10^{10}}{10^6} = 10^4 \text{ T-gates/cycle}$$

**Factories needed:**
$$N = 10^4 \times 26 \times 13 = 3.38 \times 10^6$$

This is impractical! We need better factory designs.

---

## 5. The Litinski Factory Model

### Key Innovations (2019)

Daniel Litinski revolutionized factory design with:

1. **Block-based distillation**: Process magic states in blocks, not individually
2. **Reduced footprint**: Clever qubit reuse during distillation
3. **Optimized routing**: Minimize movement overhead
4. **Layer optimization**: Choose best level structure

### Litinski 15-to-1 Factory

**Space**: $8 \times 4 = 32$ "tiles" where each tile is $d \times d$
$$Q_{\text{Litinski}} \approx 32 d^2$$

**Time**: $\sim 9d$ cycles
$$T_{\text{Litinski}} \approx 9d$$

**Volume**:
$$V_{\text{Litinski}} = 32d^2 \times 9d = 288d^3$$

$$\boxed{V_{\text{Litinski, 15-to-1}} \approx 288d^3 \text{ (45\% less than naive!)}}$$

### Litinski Multi-Level Factory

For 2-level distillation:
$$V_{\text{Litinski, 2-level}} \approx 4,500 d^3$$

Compared to naive 16,640 $d^3$: **73% reduction!**

### Magic State Distillation is "Not as Costly as You Think"

Litinski's famous 2019 paper showed:
- Traditional estimates were $10\times$ to $100\times$ too pessimistic
- Careful engineering dramatically reduces overhead
- T-gate overhead is $< 1000$ physical qubits per logical T-gate

$$\boxed{\text{Litinski: } \sim 1000 \text{ qubits per T-gate at } d = 13}$$

---

## 6. Surface Code Integration

### Factory-Computation Interface

The factory must connect to the main computation zone where algorithms run.

**Layout options:**

**Option 1: Peripheral Factories**
```
┌─────────────────────────────────────────┐
│                                         │
│     COMPUTATION ZONE                    │
│     (Logical qubits for algorithm)      │
│                                         │
├────────────┬────────────┬───────────────┤
│  Factory 1 │  Factory 2 │  Factory 3... │
└────────────┴────────────┴───────────────┘
```

**Option 2: Distributed Factories**
```
┌───────┬──────────┬───────┬──────────┬───────┐
│ Fact  │  Comp    │ Fact  │  Comp    │ Fact  │
│   1   │  Zone A  │   2   │  Zone B  │   3   │
└───────┴──────────┴───────┴──────────┴───────┘
```

**Option 3: Interleaved**
```
Logical qubits and factories share 2D array,
magic states route through computation zone
```

### Magic State Routing

After distillation, magic states must reach target qubits:

**Routing methods:**
1. **Lattice surgery teleportation**: Merge-split to move states
2. **Braiding**: Topological transport (for topological codes)
3. **Direct SWAP chain**: Physical qubit swaps

**Routing overhead:**
$$T_{\text{route}} \approx L \times d \text{ cycles}$$

where $L$ is Manhattan distance to target.

### Routing Space-Time Volume

$$V_{\text{route}} = (\text{qubits in path}) \times T_{\text{route}}$$
$$\approx L \times 2d^2 \times L \times d = 2L^2 d^3$$

**Total per T-gate:**
$$V_{\text{total}} = V_{\text{distill}} + V_{\text{route}}$$

---

## 7. Pipelining Strategies

### Basic Pipelining

**Goal**: Keep all factory components busy.

**Strategy**: Stagger distillation rounds so level-2 never waits for level-1.

```
Time →
Level-1 Unit 1:  [──Round 1──][──Round 2──][──Round 3──]
Level-1 Unit 2:    [──Round 1──][──Round 2──][──Round 3──]
...
Level-1 Unit 15:                    [──Round 1──][──Round 2──]
Level-2:                [──Round 1──][──Round 2──][──Round 3──]
```

### Double-Buffering

Maintain two sets of intermediate states:
- While level-2 uses set A, level-1 prepares set B
- Swap roles when each completes

**Advantage**: No idle time
**Cost**: 2× intermediate storage

### Adaptive Distillation

**Idea**: Adjust distillation level based on current algorithm phase.

**Early algorithm phase**: Use level-2 for highest fidelity
**Late algorithm phase**: Use level-1 if remaining T-count is low

$$\epsilon_{\text{adaptive}} = \min(k) \text{ such that } \epsilon_k \times N_{\text{remaining}} < P_{\text{target}}$$

---

## 8. Worked Examples

### Example 1: Single-Level Factory Sizing

**Problem**: Design a single-level 15-to-1 factory for code distance $d = 11$ and calculate its space-time volume.

**Solution:**

**Step 1: Calculate patch size**
$$q_{\text{patch}} = 2d^2 = 2 \times 121 = 242 \text{ qubits}$$

**Step 2: Calculate factory space**
Using Litinski model:
$$Q = 32d^2 = 32 \times 121 = 3,872 \text{ qubits}$$

**Step 3: Calculate cycle time**
$$T = 9d = 99 \text{ cycles}$$

**Step 4: Calculate space-time volume**
$$V = Q \times T = 3,872 \times 99 = 383,328 \text{ qubit-cycles}$$

In $d^3$ units:
$$V = 288d^3 = 288 \times 1331 = 383,328 \text{ (consistent!)}$$

$$\boxed{Q = 3,872 \text{ qubits}, T = 99 \text{ cycles}, V = 383,328 \text{ qubit-cycles}}$$

---

### Example 2: Factory Throughput for Algorithm

**Problem**: An algorithm requires $10^7$ T-gates to complete within $10^5$ cycles at $d = 13$. Design the factory system.

**Solution:**

**Step 1: Calculate required throughput**
$$R_{\text{required}} = \frac{10^7}{10^5} = 100 \text{ T-gates/cycle}$$

**Step 2: Calculate single factory throughput**
Litinski 2-level factory: $T_{\text{cycle}} \approx 18d = 234$ cycles
$$R_{\text{single}} = \frac{1}{234} \approx 0.0043 \text{ states/cycle}$$

**Step 3: Calculate factories needed**
$$N = \left\lceil \frac{R_{\text{required}}}{R_{\text{single}}} \right\rceil = \left\lceil \frac{100}{0.0043} \right\rceil = 23,256$$

**Step 4: Calculate total factory space**
Litinski 2-level: $Q_{\text{factory}} \approx 150 d^2 = 25,350$ qubits
$$Q_{\text{total}} = 23,256 \times 25,350 = 5.9 \times 10^8 \text{ qubits}$$

This is extremely large! Need to reconsider algorithm or accept longer runtime.

**Step 5: Trade-off analysis**
If we extend to $10^6$ cycles:
$$R_{\text{required}} = 10 \text{ T-gates/cycle}$$
$$N = 2,326 \text{ factories}$$
$$Q_{\text{total}} \approx 5.9 \times 10^7 \text{ qubits}$$

$$\boxed{\text{Feasible design: 2,326 factories, 59M qubits, } 10^6 \text{ cycles}}$$

---

### Example 3: Routing Overhead Analysis

**Problem**: A square computation zone is $100d \times 100d$. Factories are on the boundary. Calculate average routing overhead.

**Solution:**

**Step 1: Average routing distance**
For target uniformly distributed in square with side $L = 100d$:
$$\langle L_{\text{route}} \rangle \approx \frac{L}{3} = \frac{100d}{3} \approx 33d$$

**Step 2: Routing time**
$$T_{\text{route}} = L \times d = 33d \times d = 33d^2 \text{ cycles per qubit}$$

Wait, this should be:
$$T_{\text{route}} = L_{\text{route}} = 33d \text{ cycles}$$

(One cycle per unit distance in lattice surgery)

**Step 3: Compare to distillation time**
$$\frac{T_{\text{route}}}{T_{\text{distill}}} = \frac{33d}{18d} \approx 1.8$$

Routing takes almost twice as long as distillation!

**Step 4: Optimize layout**
Use distributed factories to reduce average routing distance to $\sim 15d$.

$$\boxed{\text{Routing overhead: } 1.8\times \text{ distillation time; use distributed layout}}$$

---

## 9. Practice Problems

### Problem Set A: Direct Application

**A1.** Calculate the space-time volume for a single-level Litinski factory at code distance $d = 15$.

**A2.** How many parallel 2-level factories are needed to produce 50 magic states per cycle at $d = 11$?

**A3.** A factory produces states at rate $R = 0.01$/cycle. How long to produce $10^6$ magic states?

---

### Problem Set B: Intermediate

**B1.** Design a 3-level distillation factory. Calculate:
- Space (in $d^2$ units)
- Time (in cycles)
- Space-time volume (in $d^3$ units)
Assume Litinski model for each level.

**B2.** Compare peripheral vs. distributed factory layout for a $50d \times 50d$ computation zone with 4 factories. Calculate average routing overhead for each.

**B3.** An algorithm has T-gate "bursts" - 1000 T-gates in 10 cycles, then 0 T-gates for 1000 cycles. Design a factory with output buffering to handle this workload.

---

### Problem Set C: Challenging

**C1.** Derive the optimal number of distillation levels as a function of target error $\epsilon_{\text{target}}$ and raw error $\epsilon_0$ to minimize total space-time volume.

**C2.** Design an "elastic" factory that can dynamically adjust its production rate based on demand. How much extra space is needed for 2x peak capacity?

**C3.** Consider a factory design where failed distillation attempts can be "recycled" - partially used to inform the next attempt. Estimate the efficiency improvement.

---

## 10. Computational Lab: Factory Design and Optimization

```python
"""
Day 850 Computational Lab: Distillation Factory Architecture
Design, Optimize, and Visualize Magic State Factories

This lab provides tools for designing and analyzing
magic state distillation factories for fault-tolerant QC.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class FactoryConfig:
    """Configuration for a distillation factory."""
    name: str
    n_levels: int
    space_per_level: List[float]  # in d^2 units
    time_per_level: List[float]   # in d cycles
    error_reduction: List[float]  # output/input per level


class DistillationFactory:
    """
    Model of a magic state distillation factory.
    """

    def __init__(self, config: FactoryConfig, code_distance: int):
        """
        Initialize factory with given configuration.

        Parameters:
        -----------
        config : FactoryConfig
            Factory configuration
        code_distance : int
            Surface code distance
        """
        self.config = config
        self.d = code_distance
        self.n_levels = config.n_levels

    def space(self) -> float:
        """Total factory space in physical qubits."""
        total_d2 = sum(self.config.space_per_level)
        return total_d2 * self.d**2

    def time_sequential(self) -> float:
        """Total time for sequential operation."""
        return sum([t * 15**(self.n_levels - i - 1)
                   for i, t in enumerate(self.config.time_per_level)]) * self.d

    def time_pipelined(self) -> float:
        """Total time with full pipelining."""
        return sum(self.config.time_per_level) * self.d

    def space_time_volume(self, pipelined: bool = True) -> float:
        """
        Space-time volume per output magic state.

        Parameters:
        -----------
        pipelined : bool
            Whether to use pipelined timing
        """
        t = self.time_pipelined() if pipelined else self.time_sequential()
        return self.space() * t

    def throughput(self, pipelined: bool = True) -> float:
        """
        Production rate in magic states per cycle.
        """
        t = self.time_pipelined() if pipelined else self.time_sequential()
        return 1.0 / t

    def output_error(self, raw_error: float) -> float:
        """
        Calculate output error rate.

        Parameters:
        -----------
        raw_error : float
            Input magic state error rate
        """
        eps = raw_error
        for reduction in self.config.error_reduction:
            eps = reduction * eps**3
        return eps


# Standard factory configurations
NAIVE_15_TO_1 = FactoryConfig(
    name="Naive 15-to-1",
    n_levels=1,
    space_per_level=[40],
    time_per_level=[13],
    error_reduction=[35]
)

LITINSKI_15_TO_1 = FactoryConfig(
    name="Litinski 15-to-1",
    n_levels=1,
    space_per_level=[32],
    time_per_level=[9],
    error_reduction=[35]
)

NAIVE_2_LEVEL = FactoryConfig(
    name="Naive 2-Level",
    n_levels=2,
    space_per_level=[40*15, 40],
    time_per_level=[13, 13],
    error_reduction=[35, 35]
)

LITINSKI_2_LEVEL = FactoryConfig(
    name="Litinski 2-Level",
    n_levels=2,
    space_per_level=[150],  # Combined optimized
    time_per_level=[18],
    error_reduction=[35, 35]
)


def compare_factories(code_distance: int = 11):
    """Compare different factory configurations."""
    print("\n" + "="*70)
    print(f"FACTORY COMPARISON (d = {code_distance})")
    print("="*70)

    configs = [NAIVE_15_TO_1, LITINSKI_15_TO_1, NAIVE_2_LEVEL, LITINSKI_2_LEVEL]

    print(f"\n{'Factory':<20} | {'Space (qubits)':<15} | {'Time (cycles)':<15} | {'Volume':<15}")
    print("-" * 70)

    for config in configs:
        factory = DistillationFactory(config, code_distance)
        print(f"{config.name:<20} | {factory.space():<15,.0f} | "
              f"{factory.time_pipelined():<15,.0f} | {factory.space_time_volume():<15,.0f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Space-Time Volume comparison
    ax = axes[0]
    names = [c.name for c in configs]
    volumes = [DistillationFactory(c, code_distance).space_time_volume() for c in configs]

    bars = ax.bar(range(len(configs)), volumes, color=['red', 'green', 'red', 'green'])
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Space-Time Volume (qubit-cycles)', fontsize=11)
    ax.set_title(f'Factory Space-Time Volume (d = {code_distance})', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, vol in zip(bars, volumes):
        ax.annotate(f'{vol/1e6:.2f}M', xy=(bar.get_x() + bar.get_width()/2, vol),
                   ha='center', va='bottom', fontsize=10)

    # Throughput comparison
    ax = axes[1]
    throughputs = [DistillationFactory(c, code_distance).throughput() * 1000 for c in configs]

    bars = ax.bar(range(len(configs)), throughputs, color=['red', 'green', 'red', 'green'])
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Throughput (states per 1000 cycles)', fontsize=11)
    ax.set_title(f'Factory Throughput (d = {code_distance})', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('factory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFactory comparison saved to 'factory_comparison.png'")


def visualize_factory_layout(code_distance: int = 11):
    """Visualize physical factory layout."""
    print("\n" + "="*60)
    print("FACTORY LAYOUT VISUALIZATION")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Single-level Litinski factory
    ax = axes[0]
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 6)

    # Draw tiles (8 x 4 = 32 tiles)
    for i in range(8):
        for j in range(4):
            color = 'lightblue' if (i + j) % 2 == 0 else 'lightyellow'
            if i < 4 and j < 2:
                color = 'lightgreen'  # Input zone
            if i >= 6 and j >= 2:
                color = 'lightcoral'  # Output zone

            rect = FancyBboxPatch((i, j), 0.9, 0.9,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor='gray')
            ax.add_patch(rect)

    ax.annotate('Input\nZone', xy=(1.5, 0.5), fontsize=10, ha='center', va='center')
    ax.annotate('Output', xy=(7, 3), fontsize=10, ha='center', va='center')
    ax.set_aspect('equal')
    ax.set_title(f'Litinski 15-to-1 Factory\n(32 tiles, {32 * code_distance**2:,} qubits)', fontsize=12)
    ax.set_xlabel('Tile position (x)', fontsize=11)
    ax.set_ylabel('Tile position (y)', fontsize=11)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', label='Input'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Processing'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Two-level factory with computation zone
    ax = axes[1]
    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 12)

    # Computation zone
    comp_rect = FancyBboxPatch((0, 4), 18, 6,
                               boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(comp_rect)
    ax.annotate('COMPUTATION ZONE\n(Logical Qubits)', xy=(9, 7),
               fontsize=12, ha='center', va='center')

    # Factories
    for i, x in enumerate([0, 6, 12]):
        factory_rect = FancyBboxPatch((x, 0), 4, 3,
                                      boxstyle="round,pad=0.05",
                                      facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(factory_rect)
        ax.annotate(f'Factory {i+1}', xy=(x+2, 1.5), fontsize=10, ha='center')

        # Routing channel
        ax.annotate('', xy=(x+2, 4), xytext=(x+2, 3),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_aspect('equal')
    ax.set_title('Factory-Computation Integration', fontsize=12)
    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel('Position', fontsize=11)

    plt.tight_layout()
    plt.savefig('factory_layout.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFactory layout saved to 'factory_layout.png'")


def analyze_scaling(target_t_gates: int = 10**6, max_time: int = 10**5):
    """Analyze factory requirements for different scenarios."""
    print("\n" + "="*70)
    print(f"SCALING ANALYSIS: {target_t_gates:.0e} T-gates in {max_time:.0e} cycles")
    print("="*70)

    code_distances = [7, 9, 11, 13, 15, 17]
    factory_config = LITINSKI_2_LEVEL

    results = []

    for d in code_distances:
        factory = DistillationFactory(factory_config, d)
        throughput = factory.throughput()
        required_rate = target_t_gates / max_time

        n_factories = int(np.ceil(required_rate / throughput))
        total_qubits = n_factories * factory.space()

        results.append({
            'd': d,
            'throughput': throughput,
            'n_factories': n_factories,
            'total_qubits': total_qubits
        })

        print(f"\nd = {d}:")
        print(f"  Single factory throughput: {throughput:.6f} states/cycle")
        print(f"  Factories needed: {n_factories:,}")
        print(f"  Total factory qubits: {total_qubits:,.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Factories needed vs distance
    ax = axes[0]
    ds = [r['d'] for r in results]
    n_factories = [r['n_factories'] for r in results]

    ax.semilogy(ds, n_factories, 'bo-', markersize=10, linewidth=2)
    ax.set_xlabel('Code Distance (d)', fontsize=12)
    ax.set_ylabel('Number of Factories', fontsize=12)
    ax.set_title(f'Factories Needed\n({target_t_gates:.0e} T-gates in {max_time:.0e} cycles)', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Total qubits vs distance
    ax = axes[1]
    total_qubits = [r['total_qubits'] for r in results]

    ax.semilogy(ds, total_qubits, 'ro-', markersize=10, linewidth=2)
    ax.set_xlabel('Code Distance (d)', fontsize=12)
    ax.set_ylabel('Total Factory Qubits', fontsize=12)
    ax.set_title('Total Factory Resources', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('factory_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nScaling analysis saved to 'factory_scaling.png'")

    return results


def optimize_factory_placement(comp_zone_size: int = 50, n_factories: int = 4):
    """Optimize factory placement to minimize routing overhead."""
    print("\n" + "="*60)
    print(f"FACTORY PLACEMENT OPTIMIZATION")
    print(f"Computation zone: {comp_zone_size} x {comp_zone_size}, Factories: {n_factories}")
    print("="*60)

    def average_routing_distance(factory_positions, zone_size):
        """Calculate average routing distance from factories to zone."""
        # Sample random target positions
        n_samples = 10000
        targets = np.random.rand(n_samples, 2) * zone_size

        min_distances = []
        for target in targets:
            dists = [np.abs(target[0] - f[0]) + np.abs(target[1] - f[1])
                    for f in factory_positions]
            min_distances.append(min(dists))

        return np.mean(min_distances)

    # Placement strategies
    L = comp_zone_size

    # Strategy 1: Peripheral (on edges)
    peripheral = [(0, L/4), (0, 3*L/4), (L, L/4), (L, 3*L/4)]
    dist_peripheral = average_routing_distance(peripheral, L)

    # Strategy 2: Corners
    corners = [(0, 0), (L, 0), (0, L), (L, L)]
    dist_corners = average_routing_distance(corners, L)

    # Strategy 3: Distributed (inside zone)
    distributed = [(L/4, L/4), (3*L/4, L/4), (L/4, 3*L/4), (3*L/4, 3*L/4)]
    dist_distributed = average_routing_distance(distributed, L)

    # Strategy 4: Center
    center = [(L/2, 0), (L, L/2), (L/2, L), (0, L/2)]
    dist_center = average_routing_distance(center, L)

    print(f"\nAverage routing distance (in d units):")
    print(f"  Peripheral:  {dist_peripheral:.1f}")
    print(f"  Corners:     {dist_corners:.1f}")
    print(f"  Distributed: {dist_distributed:.1f}")
    print(f"  Center:      {dist_center:.1f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    strategies = [
        ('Peripheral', peripheral, dist_peripheral),
        ('Corners', corners, dist_corners),
        ('Distributed', distributed, dist_distributed),
        ('Center', center, dist_center)
    ]

    for ax, (name, positions, avg_dist) in zip(axes.flat, strategies):
        # Draw computation zone
        rect = plt.Rectangle((0, 0), L, L, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Draw factories
        for i, pos in enumerate(positions):
            circle = plt.Circle(pos, L/15, color='blue', alpha=0.7)
            ax.add_patch(circle)
            ax.annotate(f'F{i+1}', xy=pos, ha='center', va='center',
                       color='white', fontweight='bold')

        ax.set_xlim(-5, L+5)
        ax.set_ylim(-5, L+5)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nAvg. routing: {avg_dist:.1f}d', fontsize=12)
        ax.set_xlabel('Position (d units)')
        ax.set_ylabel('Position (d units)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('factory_placement.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFactory placement optimization saved to 'factory_placement.png'")

    return {
        'peripheral': dist_peripheral,
        'corners': dist_corners,
        'distributed': dist_distributed,
        'center': dist_center
    }


def main():
    """Run all Day 850 demonstrations."""
    print("Day 850: Distillation Factory Architecture")
    print("=" * 70)

    # Factory comparison
    compare_factories(code_distance=11)

    # Layout visualization
    visualize_factory_layout(code_distance=11)

    # Scaling analysis
    analyze_scaling(target_t_gates=10**6, max_time=10**5)

    # Placement optimization
    optimize_factory_placement(comp_zone_size=50, n_factories=4)

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. Factory design critically impacts fault-tolerant QC feasibility
2. Litinski optimization: ~45% space-time volume reduction
3. Pipelining gives ~8x throughput vs. sequential operation
4. Routing overhead can exceed distillation time - layout matters!
5. Factory scaling: O(d^3) volume per magic state
6. Distributed factory placement minimizes routing overhead
""")

    print("\nDay 850 Computational Lab Complete!")


if __name__ == "__main__":
    main()
```

---

## 11. Summary

### Key Formulas Table

| Concept | Formula/Expression |
|---------|-------------------|
| Patch size | $q_{\text{patch}} \approx 2d^2$ qubits |
| Naive 1-level space | $Q \approx 40d^2$ |
| Naive 1-level time | $T \approx 13d$ cycles |
| Litinski 1-level space | $Q \approx 32d^2$ |
| Litinski 1-level time | $T \approx 9d$ cycles |
| Space-time volume (Litinski) | $V \approx 288d^3$ per state |
| Pipelining speedup | $\sim 8\times$ (2-level) |
| Routing time | $T_{\text{route}} \approx L$ cycles |
| Factories needed | $N = \lceil R_{\text{required}} / R_{\text{factory}} \rceil$ |

### Key Takeaways

1. **Factory = Protocol + Scheduling + Layout**: Engineering matters as much as theory
2. **Pipelining is essential**: Keeps all factory components busy
3. **Litinski designs**: 45% volume reduction through clever optimization
4. **Routing overhead**: Can dominate total cost; use distributed layouts
5. **Space-time trade-offs**: More factories = faster but more qubits
6. **Practical numbers**: ~1000 qubits per T-gate at $d = 13$ (Litinski)

---

## 12. Daily Checklist

- [ ] I understand the space-time volume metric for factories
- [ ] I can design a multi-level pipelined distillation factory
- [ ] I know how to integrate factories with computation zones
- [ ] I can calculate factory requirements for a given algorithm
- [ ] I understand routing overhead and placement optimization
- [ ] I completed the computational lab on factory design

---

## 13. Preview: Day 851

Tomorrow we explore **Bravyi-Haah Protocols**:

- Triorthogonal codes beyond Reed-Muller
- 10-to-2 distillation with $O(\epsilon^2)$ scaling
- Improved asymptotic overhead
- When to prefer Bravyi-Haah over 15-to-1
- State-of-the-art distillation protocols

We will see how newer protocols achieve better error suppression with lower overhead.

---

*"The gap between theoretical protocols and practical factories is where quantum engineering lives. Careful design makes the difference between feasibility and impossibility."*
— Craig Gidney

