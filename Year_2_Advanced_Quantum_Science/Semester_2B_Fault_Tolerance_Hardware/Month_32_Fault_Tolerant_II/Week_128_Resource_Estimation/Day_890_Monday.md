# Day 890: Physical Qubit Counting

## Week 128, Day 1 | Month 32: Fault-Tolerant Quantum Computing II

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Logical-to-physical qubit mapping |
| Afternoon | 2.5 hours | Problem solving: Overhead calculations |
| Evening | 2 hours | Computational lab: Qubit counting tools |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Calculate the physical qubit overhead** from logical qubit requirements using code distance formulas
2. **Determine optimal code distances** based on target error rates and algorithm depth
3. **Quantify factory overhead** including distillation qubit requirements
4. **Account for routing and ancilla qubits** in practical architectures
5. **Apply the complete qubit counting formula** to benchmark quantum algorithms
6. **Understand the scaling implications** of physical qubit requirements

---

## Core Content

### 1. The Fundamental Overhead Problem

Fault-tolerant quantum computing requires encoding logical qubits into many physical qubits. Understanding this overhead is essential for hardware roadmaps and algorithm feasibility analysis.

#### The Basic Relationship

For a topological code with code distance $d$:

$$\boxed{Q_{data} = n_{logical} \times d^2}$$

This represents only the **data qubits**. The complete physical qubit count includes:

$$\boxed{Q_{total} = Q_{data} + Q_{syndrome} + Q_{factory} + Q_{routing} + Q_{ancilla}}$$

#### Component Breakdown

| Component | Description | Typical Contribution |
|-----------|-------------|---------------------|
| $Q_{data}$ | Logical qubit storage | 40-60% |
| $Q_{syndrome}$ | Error detection | Included in $d^2$ for surface codes |
| $Q_{factory}$ | Magic state distillation | 20-40% |
| $Q_{routing}$ | Lattice surgery pathways | 10-20% |
| $Q_{ancilla}$ | Intermediate computations | 5-10% |

### 2. Surface Code Physical Layout

#### Data and Measurement Qubits

In the rotated surface code, each logical qubit requires:

$$Q_{per\_logical} = d^2 + (d-1)^2 = 2d^2 - 2d + 1$$

For practical purposes, we approximate:

$$Q_{per\_logical} \approx 2d^2$$

where the factor of 2 accounts for both data qubits (forming the code) and measurement qubits (for syndrome extraction).

#### Detailed Layout Analysis

```
Rotated Surface Code (d=3):
    ○─●─○
    │ │ │
    ●─○─●      ○ = Data qubit
    │ │ │      ● = Measurement qubit
    ○─●─○

Physical qubits = d² (data) + (d-1)² (Z-type) + (d-1)² (X-type)
                = 9 + 4 + 4 = 17 (exact for d=3)
                ≈ 2d² = 18 (approximation)
```

### 3. Code Distance Selection

#### Error Rate Requirement

The logical error rate per round scales as:

$$p_L \approx c \cdot \left(\frac{p_{phys}}{p_{th}}\right)^{(d+1)/2}$$

where:
- $p_{phys}$ = physical error rate
- $p_{th}$ = threshold error rate (~1% for surface codes)
- $c$ = constant (~0.1)

#### Determining Required Distance

For an algorithm with:
- $n$ logical qubits
- $D$ circuit depth (number of logical operations)
- $\epsilon$ target total error probability

We require:

$$p_L \times n \times D \leq \epsilon$$

Solving for distance:

$$\boxed{d \geq \frac{\log(c \cdot n \cdot D / \epsilon)}{\log(p_{th}/p_{phys})} \times 2 - 1}$$

#### Practical Distance Values

| $p_{phys}$ | Target $\epsilon$ | $n \times D$ | Required $d$ |
|------------|-------------------|--------------|--------------|
| $10^{-3}$ | $10^{-2}$ | $10^6$ | 11 |
| $10^{-3}$ | $10^{-2}$ | $10^9$ | 17 |
| $10^{-3}$ | $10^{-2}$ | $10^{12}$ | 23 |
| $10^{-4}$ | $10^{-2}$ | $10^{12}$ | 15 |

### 4. Factory Overhead

#### Magic State Requirements

Non-Clifford gates (T gates, Toffoli gates) require magic states produced by distillation factories. The factory footprint dominates for many algorithms.

#### Factory Area Formula

For a multi-level distillation factory:

$$A_{factory} = A_{level1} + A_{level2} + ... + A_{levelk}$$

A typical 15-to-1 factory at each level:

$$A_{15-to-1} \approx 12d \times 6d = 72d^2$$

For two-level distillation achieving output error $p_{out} \sim 10^{-15}$:

$$A_{factory}^{(2-level)} \approx 150d^2 \text{ to } 200d^2$$

#### Number of Factories

The number of factories needed depends on T-count and desired runtime:

$$n_{factories} = \frac{T_{count} \times t_{cycle}}{T_{runtime} \times r_{production}}$$

where:
- $T_{count}$ = total T gates in algorithm
- $t_{cycle}$ = surface code cycle time
- $T_{runtime}$ = target runtime
- $r_{production}$ = T-states produced per factory per cycle

### 5. Routing Overhead

#### Lattice Surgery Requirements

Lattice surgery operations require routing space between logical qubits. The routing overhead depends on the algorithm structure.

#### Routing Models

**Dense Model (pessimistic):**
$$Q_{routing} = n_{logical} \times d^2$$

This doubles the qubit count, assuming each logical qubit needs an adjacent routing region.

**Sparse Model (optimistic):**
$$Q_{routing} = \sqrt{n_{logical}} \times d^2$$

Assumes efficient routing through shared channels.

**Practical Estimate:**
$$Q_{routing} \approx 0.3 \times n_{logical} \times d^2 \text{ to } 0.5 \times n_{logical} \times d^2$$

### 6. Complete Physical Qubit Formula

Combining all components:

$$\boxed{Q_{total} = n_{logical} \times 2d^2 \times (1 + f_{routing}) + n_{factories} \times A_{factory}}$$

where:
- $f_{routing} \approx 0.3 - 0.5$ = routing overhead factor
- $A_{factory} \approx 150d^2 - 200d^2$ = factory area

#### Simplified Form

For order-of-magnitude estimates:

$$Q_{total} \approx 3 \times n_{logical} \times d^2 + 200 \times n_{factories} \times d^2$$

---

## Practical Benchmarks

### RSA-2048 Factoring (Gidney-Ekerå)

The landmark 2021 paper established:

| Parameter | Value |
|-----------|-------|
| Logical qubits | 6,189 |
| T-count | $2.04 \times 10^{10}$ |
| Code distance | 27 |
| T-factories | 28 |
| Physical qubits | **20,165,344** (~20M) |

#### Breakdown:
- Logical qubit area: $6189 \times 2 \times 27^2 = 9.02 \times 10^6$
- Factory area: $28 \times 148 \times 27^2 = 3.02 \times 10^6$
- Routing: $\approx 8 \times 10^6$
- Total: $\approx 20 \times 10^6$

### Quantum Chemistry: FeMoco

Simulating the FeMo-cofactor for nitrogen fixation:

| Parameter | Value |
|-----------|-------|
| Logical qubits | ~4,000 |
| T-count | $10^{12} - 10^{14}$ |
| Code distance | 31-35 |
| Physical qubits | ~4M (with optimizations) |

### NISQ-to-FTQC Transition

| Era | Physical Qubits | Logical Qubits | Code Distance |
|-----|-----------------|----------------|---------------|
| NISQ (2024) | 1,000 | 0 | N/A |
| Early FTQC | 10,000 | 10-50 | 5-7 |
| Cryptographic | 1M | 1,000 | 15-20 |
| Scientific | 20M | 6,000 | 25-30 |
| Full-scale | 100M+ | 10,000+ | 30+ |

---

## Worked Examples

### Example 1: Basic Qubit Count

**Problem:** Calculate physical qubits for a 100-logical-qubit algorithm with code distance 15, 4 T-factories, and 40% routing overhead.

**Solution:**

Step 1: Logical qubit area
$$Q_{logical} = n \times 2d^2 = 100 \times 2 \times 15^2 = 100 \times 450 = 45,000$$

Step 2: Routing overhead
$$Q_{routing} = f_{route} \times Q_{logical} = 0.4 \times 45,000 = 18,000$$

Step 3: Factory area (using $A_{factory} = 150d^2$)
$$Q_{factory} = n_{fac} \times A_{fac} = 4 \times 150 \times 225 = 135,000$$

Step 4: Total
$$\boxed{Q_{total} = 45,000 + 18,000 + 135,000 = 198,000 \text{ physical qubits}}$$

**Note:** Factories dominate! This is typical for T-gate-heavy algorithms.

---

### Example 2: Determining Code Distance

**Problem:** An algorithm has 500 logical qubits and $10^8$ operations. Physical error rate is $10^{-3}$, threshold is $10^{-2}$, and target total error is 1%. Find the minimum code distance.

**Solution:**

Step 1: Required logical error rate
$$p_L \leq \frac{\epsilon}{n \times D} = \frac{0.01}{500 \times 10^8} = 2 \times 10^{-13}$$

Step 2: Using the scaling formula $p_L = c(p/p_{th})^{(d+1)/2}$ with $c = 0.1$:
$$2 \times 10^{-13} = 0.1 \times (0.1)^{(d+1)/2}$$
$$2 \times 10^{-12} = (0.1)^{(d+1)/2}$$

Step 3: Taking logarithms:
$$\log(2 \times 10^{-12}) = \frac{d+1}{2} \times \log(0.1)$$
$$-11.7 = \frac{d+1}{2} \times (-1)$$
$$d+1 = 23.4$$
$$\boxed{d = 23 \text{ (round up to odd number)}}$$

---

### Example 3: Factory Allocation

**Problem:** An algorithm has T-count $10^{10}$. Each factory produces one T-state per 100 microseconds. Target runtime is 1 hour. How many factories are needed?

**Solution:**

Step 1: Available time in microseconds
$$T_{runtime} = 1 \text{ hour} = 3600 \times 10^6 \mu s = 3.6 \times 10^9 \mu s$$

Step 2: T-states needed per unit time
$$\text{Rate needed} = \frac{10^{10}}{3.6 \times 10^9 \mu s} = 2.78 \text{ T-states}/\mu s$$

Step 3: Production rate per factory
$$r_{factory} = \frac{1}{100 \mu s} = 0.01 \text{ T-states}/\mu s$$

Step 4: Number of factories
$$n_{factories} = \frac{2.78}{0.01} = 278$$

$$\boxed{n_{factories} = 278 \text{ factories}}$$

This would require $278 \times 150d^2$ additional physical qubits for factories alone.

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the physical qubit count for:
- 50 logical qubits
- Code distance d = 11
- 2 T-factories with $A_{factory} = 150d^2$
- 35% routing overhead

**Problem 1.2:** What code distance is needed for:
- 1000 logical qubits
- $10^6$ circuit depth
- Physical error rate $5 \times 10^{-4}$
- Target error probability 5%

**Problem 1.3:** How many physical qubits per logical qubit for d = 7, 11, 15, 21?

### Level 2: Intermediate Analysis

**Problem 2.1:** Compare the resource requirements for two implementations:
- Implementation A: d = 15, 10 factories
- Implementation B: d = 21, 4 factories
- Both have 200 logical qubits with 40% routing overhead
Which requires fewer physical qubits?

**Problem 2.2:** An algorithm can be compiled with either:
- Version 1: 500 logical qubits, T-count $5 \times 10^9$
- Version 2: 800 logical qubits, T-count $10^9$
Assuming factories dominate, which version requires fewer resources?

**Problem 2.3:** Derive the relationship between physical error rate improvement and code distance reduction while maintaining the same logical error rate.

### Level 3: Challenging Problems

**Problem 3.1:** **Optimal Factory Allocation**
Given a fixed physical qubit budget of 500,000, maximize T-state production rate. You have:
- 100 logical qubits required (non-negotiable)
- d = 13 minimum
- $A_{factory} = 150d^2$
- Each factory produces 1 T-state per 150 cycles
- 40% routing overhead

**Problem 3.2:** **Cross-Over Analysis**
At what T-count does the factory overhead equal the logical qubit overhead? Express in terms of $n_{logical}$, $d$, and factory parameters.

**Problem 3.3:** **Scaling Analysis**
For Shor's algorithm factoring an n-bit number:
- Logical qubits scale as $O(n)$
- T-count scales as $O(n^2 \log n)$
- Code distance scales as $O(\log n)$

Derive the physical qubit scaling. Is it dominated by logical qubits or factories for large n?

---

## Computational Lab

### Qubit Counting Calculator

```python
"""
Day 890: Physical Qubit Counting Tool
Comprehensive resource estimation for fault-tolerant quantum computing.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SurfaceCodeParams:
    """Parameters for surface code resource estimation."""
    physical_error_rate: float = 1e-3
    threshold_error_rate: float = 1e-2
    scaling_constant: float = 0.1
    measurement_qubits_factor: float = 2.0  # data + syndrome qubits
    routing_overhead: float = 0.4
    factory_area_coefficient: float = 150  # in units of d²

def calculate_required_distance(
    n_logical: int,
    circuit_depth: int,
    target_error: float,
    params: SurfaceCodeParams
) -> int:
    """
    Calculate minimum code distance for target error rate.

    Based on: p_L = c * (p_phys/p_th)^((d+1)/2)
    Requirement: p_L * n * D <= epsilon
    """
    # Required logical error rate per qubit per operation
    p_logical_required = target_error / (n_logical * circuit_depth)

    # Ratio of physical to threshold error
    ratio = params.physical_error_rate / params.threshold_error_rate

    # Solve for d
    # p_logical_required = c * ratio^((d+1)/2)
    # log(p_logical_required/c) = ((d+1)/2) * log(ratio)

    if ratio >= 1:
        print("Warning: Physical error rate above threshold!")
        return 999  # Indicates failure

    log_term = np.log(p_logical_required / params.scaling_constant)
    log_ratio = np.log(ratio)

    d_float = 2 * log_term / log_ratio - 1

    # Round up to nearest odd integer
    d = int(np.ceil(d_float))
    if d % 2 == 0:
        d += 1

    return max(3, d)  # Minimum distance of 3


def calculate_physical_qubits(
    n_logical: int,
    code_distance: int,
    n_factories: int,
    params: SurfaceCodeParams
) -> dict:
    """
    Calculate total physical qubit requirements.

    Returns detailed breakdown of qubit allocation.
    """
    d_squared = code_distance ** 2

    # Logical qubit area (data + measurement qubits)
    logical_area = n_logical * params.measurement_qubits_factor * d_squared

    # Routing overhead
    routing_area = params.routing_overhead * logical_area

    # Factory area
    factory_area = n_factories * params.factory_area_coefficient * d_squared

    # Total
    total = logical_area + routing_area + factory_area

    return {
        'logical_qubits': n_logical,
        'code_distance': code_distance,
        'n_factories': n_factories,
        'logical_area': int(logical_area),
        'routing_area': int(routing_area),
        'factory_area': int(factory_area),
        'total_physical_qubits': int(total),
        'qubits_per_logical': total / n_logical
    }


def calculate_factory_requirements(
    t_count: int,
    target_runtime_seconds: float,
    cycle_time_us: float = 1.0,
    distillation_cycles: int = 100
) -> dict:
    """
    Calculate number of factories needed for target runtime.

    Parameters:
    - t_count: Total T gates in algorithm
    - target_runtime_seconds: Desired execution time
    - cycle_time_us: Surface code cycle time in microseconds
    - distillation_cycles: Cycles per T-state production
    """
    # Convert runtime to microseconds
    runtime_us = target_runtime_seconds * 1e6

    # Time per T-state production
    t_state_time = distillation_cycles * cycle_time_us

    # Required production rate (T-states per microsecond)
    required_rate = t_count / runtime_us

    # Production rate per factory
    factory_rate = 1 / t_state_time

    # Number of factories
    n_factories = int(np.ceil(required_rate / factory_rate))

    return {
        't_count': t_count,
        'target_runtime_s': target_runtime_seconds,
        'required_rate': required_rate,
        'factory_rate': factory_rate,
        'n_factories': n_factories,
        'actual_runtime_s': t_count * t_state_time / (n_factories * 1e6)
    }


class QuantumResourceEstimator:
    """Complete resource estimation for fault-tolerant quantum algorithms."""

    def __init__(self, params: Optional[SurfaceCodeParams] = None):
        self.params = params or SurfaceCodeParams()

    def estimate_resources(
        self,
        n_logical: int,
        t_count: int,
        circuit_depth: int,
        target_error: float = 0.01,
        target_runtime_hours: float = 1.0
    ) -> dict:
        """
        Complete resource estimation for a quantum algorithm.
        """
        # Step 1: Determine code distance
        code_distance = calculate_required_distance(
            n_logical, circuit_depth, target_error, self.params
        )

        # Step 2: Determine factory requirements
        factory_info = calculate_factory_requirements(
            t_count, target_runtime_hours * 3600
        )
        n_factories = factory_info['n_factories']

        # Step 3: Calculate physical qubits
        qubit_info = calculate_physical_qubits(
            n_logical, code_distance, n_factories, self.params
        )

        # Combine results
        return {
            **qubit_info,
            **factory_info,
            'target_error': target_error,
            'target_runtime_hours': target_runtime_hours
        }

    def print_report(self, results: dict):
        """Print formatted resource estimation report."""
        print("\n" + "="*60)
        print("QUANTUM RESOURCE ESTIMATION REPORT")
        print("="*60)

        print(f"\n{'Algorithm Parameters':-^60}")
        print(f"  Logical qubits:        {results['logical_qubits']:,}")
        print(f"  T-count:               {results['t_count']:.2e}")
        print(f"  Target error:          {results['target_error']:.1%}")
        print(f"  Target runtime:        {results['target_runtime_hours']:.1f} hours")

        print(f"\n{'Code Parameters':-^60}")
        print(f"  Code distance:         {results['code_distance']}")
        print(f"  Number of factories:   {results['n_factories']}")

        print(f"\n{'Physical Qubit Breakdown':-^60}")
        print(f"  Logical qubit area:    {results['logical_area']:,}")
        print(f"  Routing area:          {results['routing_area']:,}")
        print(f"  Factory area:          {results['factory_area']:,}")
        print(f"  TOTAL PHYSICAL QUBITS: {results['total_physical_qubits']:,}")

        print(f"\n{'Efficiency Metrics':-^60}")
        print(f"  Physical/Logical ratio: {results['qubits_per_logical']:.1f}")
        print(f"  Actual runtime:        {results['actual_runtime_s']/3600:.2f} hours")

        print("="*60 + "\n")


def plot_distance_scaling(params: SurfaceCodeParams):
    """Visualize how code distance affects resources."""
    distances = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 29]
    n_logical = 100
    n_factories = 10

    qubits = []
    for d in distances:
        result = calculate_physical_qubits(n_logical, d, n_factories, params)
        qubits.append(result['total_physical_qubits'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].plot(distances, qubits, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Code Distance', fontsize=12)
    axes[0].set_ylabel('Physical Qubits', fontsize=12)
    axes[0].set_title(f'Physical Qubit Scaling (n={n_logical} logical)', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Log scale
    axes[1].semilogy(distances, qubits, 'b-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Code Distance', fontsize=12)
    axes[1].set_ylabel('Physical Qubits (log scale)', fontsize=12)
    axes[1].set_title('Physical Qubit Scaling (Log Scale)', fontsize=14)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('distance_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_benchmark_comparison():
    """Compare resource requirements for benchmark algorithms."""
    benchmarks = {
        'RSA-2048': {'n_logical': 6189, 't_count': 2e10, 'depth': 2e10, 'd': 27},
        'FeMoco': {'n_logical': 4000, 't_count': 1e12, 'depth': 1e12, 'd': 31},
        'QAOA-100': {'n_logical': 100, 't_count': 1e6, 'depth': 1e4, 'd': 11},
        'VQE-50': {'n_logical': 50, 't_count': 1e5, 'depth': 1e3, 'd': 7},
        'Grover-1000': {'n_logical': 1000, 't_count': 1e8, 'depth': 1e6, 'd': 17}
    }

    params = SurfaceCodeParams()

    names = list(benchmarks.keys())
    physical_qubits = []
    logical_portion = []
    factory_portion = []

    for name, bm in benchmarks.items():
        result = calculate_physical_qubits(
            bm['n_logical'], bm['d'],
            max(1, int(bm['t_count'] / 1e9)),  # Rough factory estimate
            params
        )
        physical_qubits.append(result['total_physical_qubits'])
        logical_portion.append(result['logical_area'] + result['routing_area'])
        factory_portion.append(result['factory_area'])

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.6

    bars1 = ax.bar(x, logical_portion, width, label='Logical + Routing', color='steelblue')
    bars2 = ax.bar(x, factory_portion, width, bottom=logical_portion,
                   label='Factories', color='darkorange')

    ax.set_ylabel('Physical Qubits', fontsize=12)
    ax.set_title('Physical Qubit Requirements by Algorithm', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (total, log_p) in enumerate(zip(physical_qubits, logical_portion)):
        ax.annotate(f'{total/1e6:.1f}M', xy=(i, total), ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# Demonstration
if __name__ == "__main__":
    print("Physical Qubit Counting Tool - Day 890")
    print("="*50)

    # Create estimator
    estimator = QuantumResourceEstimator()

    # Example 1: RSA-2048
    print("\n--- Example: RSA-2048 Factoring ---")
    results = estimator.estimate_resources(
        n_logical=6189,
        t_count=int(2e10),
        circuit_depth=int(2e10),
        target_error=0.01,
        target_runtime_hours=8
    )
    estimator.print_report(results)

    # Example 2: Small algorithm
    print("\n--- Example: Small VQE Algorithm ---")
    results_vqe = estimator.estimate_resources(
        n_logical=50,
        t_count=int(1e5),
        circuit_depth=1000,
        target_error=0.05,
        target_runtime_hours=0.1
    )
    estimator.print_report(results_vqe)

    # Generate visualizations
    params = SurfaceCodeParams()
    plot_distance_scaling(params)
    plot_benchmark_comparison()

    # Parameter sensitivity analysis
    print("\n--- Code Distance Sensitivity ---")
    for d in [7, 11, 15, 21, 27]:
        result = calculate_physical_qubits(100, d, 10, params)
        print(f"d={d:2d}: {result['total_physical_qubits']:>10,} physical qubits "
              f"({result['qubits_per_logical']:.0f} per logical)")
```

---

## Summary

### Key Formulas

| Formula | Expression | Application |
|---------|------------|-------------|
| Data qubits | $Q_{data} = n \times d^2$ | Basic logical storage |
| Total qubits | $Q_{total} = 2nd^2(1+f_r) + n_f A_f$ | Complete count |
| Distance selection | $d \geq 2\frac{\log(cnD/\epsilon)}{\log(p_{th}/p_{phys})} - 1$ | Error requirements |
| Factory count | $n_f = \frac{T \times t_c}{T_{run} \times r}$ | Runtime target |

### Physical Qubit Breakdown Components

1. **Logical area**: $2d^2$ per logical qubit (data + measurement)
2. **Routing overhead**: 30-50% of logical area
3. **Factory overhead**: Often dominates for T-heavy algorithms
4. **Ancilla**: Additional workspace for intermediate computations

### Key Insights

1. **Quadratic scaling**: Physical qubits scale as $d^2$, making code distance reduction extremely valuable
2. **Factory dominance**: For many algorithms, factories consume more qubits than logical storage
3. **Trade-off space**: Runtime vs. qubit count trade-off through factory allocation
4. **Error rate leverage**: Each order of magnitude improvement in $p_{phys}$ allows ~4-6 reduction in distance

---

## Daily Checklist

- [ ] I can calculate physical qubit overhead from logical requirements
- [ ] I understand the components: data, syndrome, factory, routing, ancilla
- [ ] I can determine required code distance from error specifications
- [ ] I can estimate factory qubit requirements
- [ ] I understand the RSA-2048 benchmark (~20M qubits)
- [ ] I can use the qubit counting calculator
- [ ] I recognize when factories vs. logical qubits dominate

---

## Preview: Day 891

Tomorrow we explore **Space-Time Volume Analysis**, introducing Litinski's framework for optimizing quantum computations. We'll learn:

- The space-time volume concept: $V = A \times T$
- Trade-offs between qubit count and execution time
- Volume optimization strategies
- The "Game of Surface Codes" methodology

The key insight is that physical qubits and time are fungible resources—we can often trade one for the other while keeping the total "volume" constant.

---

*Day 890 of 2184 | Week 128 of 312 | Month 32 of 72*

*"The first step in building a quantum computer is counting how many qubits you actually need."*
