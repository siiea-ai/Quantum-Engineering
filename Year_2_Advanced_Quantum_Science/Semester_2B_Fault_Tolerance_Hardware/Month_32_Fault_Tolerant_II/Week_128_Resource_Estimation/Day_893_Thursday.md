# Day 893: Runtime Analysis

## Week 128, Day 4 | Month 32: Fault-Tolerant Quantum Computing II

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Runtime models and bottleneck analysis |
| Afternoon | 2.5 hours | Problem solving: Benchmark algorithm estimation |
| Evening | 2 hours | Computational lab: Runtime calculator implementation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Derive runtime formulas** from T-count, factory throughput, and parallelism
2. **Identify computational bottlenecks** in fault-tolerant algorithms
3. **Analyze RSA-2048 runtime** using the Gidney-Ekerå framework
4. **Estimate quantum chemistry runtimes** for molecular simulation
5. **Apply optimization strategies** to reduce algorithm execution time
6. **Build practical runtime estimators** for arbitrary quantum algorithms

---

## Core Content

### 1. The Runtime Formula

#### Fundamental Runtime Expression

For a fault-tolerant quantum algorithm:

$$\boxed{T_{runtime} = \max\left( T_{critical}, T_{magic}, T_{Clifford} \right)}$$

where:
- $T_{critical}$ = critical path through the circuit (parallelism limit)
- $T_{magic}$ = time to produce all magic states
- $T_{Clifford}$ = time for Clifford operations (usually negligible)

#### Magic State Bottleneck

For most algorithms, magic state production dominates:

$$\boxed{T_{runtime} \approx \frac{N_T \times t_{distill}}{n_{factories}}}$$

where:
- $N_T$ = total T-count of the algorithm
- $t_{distill}$ = time per T-state production (cycles)
- $n_{factories}$ = number of distillation factories

#### Converting to Real Time

With code cycle time $\tau_{cycle}$:

$$T_{runtime}^{(real)} = T_{runtime}^{(cycles)} \times \tau_{cycle}$$

Typical values: $\tau_{cycle} \approx 0.1 - 10 \mu s$ depending on hardware.

### 2. Bottleneck Analysis

#### The Three Regimes

**Regime 1: Factory-Limited**
$$T \approx \frac{N_T}{n_f \times r_f}$$
*When factory throughput cannot keep up with consumption.*

**Regime 2: Parallelism-Limited**
$$T \approx D_{critical} \times t_{gate}$$
*When algorithm has long sequential dependency chains.*

**Regime 3: Communication-Limited**
$$T \approx n_{ops} \times t_{routing}$$
*When lattice surgery routing dominates.*

#### Identifying the Bottleneck

Calculate each component and take the maximum:

```
Algorithm: N_T = 10¹⁰, D_critical = 10⁸, n_factories = 100

Factory time:     10¹⁰ / (100 × 0.01) = 10¹⁰ cycles
Critical path:    10⁸ × 10 = 10⁹ cycles
Routing overhead: 10⁹ × 1 = 10⁹ cycles

Bottleneck: Factory-limited (10¹⁰ >> 10⁹)
```

### 3. T-Count Analysis

#### Sources of T-Gates

| Operation | T-count | Notes |
|-----------|---------|-------|
| T gate | 1 | Fundamental non-Clifford |
| T† gate | 1 | Conjugate |
| Toffoli | 4-7 | Depending on decomposition |
| Arbitrary rotation | $3 \log(1/\epsilon)$ | Solovay-Kitaev / gridsynth |
| Phase gate | 2 | Via T |
| CCZ | 4 | Three-qubit phase |

#### T-Count Reduction Techniques

1. **T-gate optimization**: Combine adjacent T's
2. **Clifford rewriting**: Reduce circuit depth
3. **Ancilla trading**: Use more qubits for fewer T's
4. **Measurement-based**: Convert some T's to measurements

**Example**: Toffoli optimization
- Naive: 7 T-gates
- Optimized (with ancilla): 4 T-gates
- Relative synthesis: ~2.7 T-gates (amortized)

### 4. Benchmark: RSA-2048 Factoring

#### The Gidney-Ekerå Algorithm

The 2021 landmark paper optimized Shor's algorithm for RSA-2048:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Bits to factor | 2048 | Standard RSA key size |
| Logical qubits | 6,189 | For modular arithmetic |
| T-count | $2.04 \times 10^{10}$ | After optimization |
| Toffoli count | $5.1 \times 10^9$ | Core operations |
| Code distance | 27 | For $p_{phys} = 10^{-3}$ |
| Physical qubits | 20,165,344 | Including factories |
| Factories | 28 | Two-level 15-to-1 |
| Runtime | ~8 hours | At 1 MHz cycle rate |

#### Runtime Breakdown

Step 1: T-state production time
$$t_{T-state} = 8d \times 2 = 16 \times 27 = 432 \text{ cycles (two-level)}$$

Step 2: T-states per factory per cycle
$$r_{factory} = 1/432 \text{ T-states/cycle}$$

Step 3: Total production rate
$$r_{total} = 28 / 432 = 0.0648 \text{ T-states/cycle}$$

Step 4: Runtime in cycles
$$T_{cycles} = \frac{2.04 \times 10^{10}}{0.0648} = 3.15 \times 10^{11} \text{ cycles}$$

Step 5: Convert to real time (1 μs cycles)
$$T_{real} = 3.15 \times 10^{11} \times 10^{-6} s = 3.15 \times 10^5 s = 87.5 \text{ hours}$$

**Wait—this doesn't match!** The 8-hour claim requires:
- Faster cycle time: ~0.1 μs (100 ns)
- More parallelism in factory design
- Optimized factory pipelining

With 100 ns cycles: $T = 3.15 \times 10^4 s = 8.75$ hours ✓

### 5. Benchmark: Quantum Chemistry

#### FeMoco Simulation

The FeMo-cofactor is central to nitrogen fixation catalysis:

| Parameter | Value | Notes |
|-----------|-------|-------|
| System | Fe₇MoS₉C | Nitrogenase active site |
| Qubits needed | ~4,000 | For chemical accuracy |
| T-count | $10^{12} - 10^{14}$ | Depending on precision |
| Target precision | 1 mHartree | Chemical accuracy |
| Runtime estimate | Days to weeks | Current estimates |

#### Chemistry Runtime Scaling

For molecular simulation:

$$T_{count} \propto \frac{N^4}{\epsilon^{2-3}}$$

where $N$ = number of orbitals, $\epsilon$ = target precision.

**Comparison:**

| Molecule | Orbitals | T-count | Est. Runtime |
|----------|----------|---------|--------------|
| H₂ | 4 | $10^4$ | Seconds |
| H₂O | 13 | $10^7$ | Minutes |
| Benzene | 30 | $10^9$ | Hours |
| Fe₂S₂ cluster | 100 | $10^{12}$ | Days |
| FeMoco | 200 | $10^{14}$ | Weeks |

### 6. Runtime Optimization Strategies

#### Strategy 1: Increase Factory Count

Doubling factories halves runtime (if factory-limited):

$$T' = \frac{T}{2} \text{ with } n'_f = 2n_f$$

**Trade-off**: Doubles factory qubit overhead.

#### Strategy 2: Reduce T-Count

Algorithmic optimization to reduce $N_T$:

$$T' = T \times \frac{N'_T}{N_T}$$

**Techniques**:
- Gate synthesis optimization (Ross-Selinger)
- Circuit recompilation
- Ancilla-based T reduction

#### Strategy 3: Faster Cycle Time

Improve physical hardware speed:

$$T'_{real} = T_{real} \times \frac{\tau'_{cycle}}{\tau_{cycle}}$$

**Current targets**:
- Superconducting: 100 ns - 1 μs
- Trapped ion: 10 μs - 100 μs
- Neutral atom: 1 μs - 10 μs

#### Strategy 4: Reduce Code Distance

If physical error rate improves:

$$d' < d \Rightarrow t_{distill} \propto d' < d$$

**Cascading benefits**:
- Smaller factories
- Faster distillation
- Fewer physical qubits

### 7. Practical Runtime Estimation Framework

#### The Complete Formula

$$\boxed{T_{runtime} = \frac{N_T \times t_{cycle} \times d \times k}{n_f} + T_{overhead}}$$

where:
- $N_T$ = T-count
- $t_{cycle}$ = code cycle time
- $d$ = code distance
- $k$ = distillation factor (8-16 for 1-2 levels)
- $n_f$ = number of factories
- $T_{overhead}$ = initialization, measurement, classical processing

#### Quick Estimation Table

| T-count | Factories | d=17 | d=27 |
|---------|-----------|------|------|
| $10^8$ | 10 | 2.3 min | 5.8 min |
| $10^9$ | 10 | 23 min | 58 min |
| $10^{10}$ | 10 | 3.8 hr | 9.6 hr |
| $10^{10}$ | 100 | 23 min | 58 min |
| $10^{12}$ | 100 | 38 hr | 96 hr |

*Assuming 1 μs cycle time, 8d cycles per T-state.*

---

## Practical Benchmarks

### Cross-Algorithm Comparison

| Algorithm | T-count | Factories | Qubits | Runtime |
|-----------|---------|-----------|--------|---------|
| RSA-2048 | $2 \times 10^{10}$ | 28 | 20M | 8 hr |
| RSA-4096 | $2 \times 10^{11}$ | 56 | 50M | 40 hr |
| ECDLP-256 | $5 \times 10^9$ | 20 | 5M | 2 hr |
| FeMoco | $10^{13}$ | 200 | 10M | 2 weeks |
| Hubbard (10×10) | $10^{11}$ | 50 | 2M | 20 hr |
| QAOA p=100 | $10^8$ | 5 | 100K | 30 min |

### Hardware Requirements for Different Runtimes

**Target: 1-hour runtime**

| Algorithm | Required Factories | Total Qubits |
|-----------|-------------------|--------------|
| RSA-2048 | 224 | ~100M |
| FeMoco | Impractical | >1B |
| ECDLP-256 | 10 | ~3M |
| VQE-100 | 2 | ~50K |

### Breaking Points

When does quantum advantage become practical?

| Milestone | Requirement | Timeline (est.) |
|-----------|-------------|-----------------|
| Quantum supremacy | 50-100 physical qubits | Achieved (2019) |
| Error-corrected qubit | 1000s physical → 1 logical | ~2025-2027 |
| Useful FTQC | 100 logical, d=15 | ~2028-2030 |
| RSA-2048 in 8 hours | 20M physical | ~2035+ |
| Full chemistry | 100M+ physical | ~2040+ |

---

## Worked Examples

### Example 1: RSA Runtime Calculation

**Problem:** Verify the 8-hour runtime for RSA-2048 with:
- T-count: $2 \times 10^{10}$
- Factories: 28
- Code distance: 27
- Cycle time: 100 ns

**Solution:**

Step 1: T-state production time per factory
$$t_{T} = 8d \times 2 \text{ levels} = 16 \times 27 = 432 \text{ cycles}$$

Step 2: Total T-state production rate
$$r = \frac{n_f}{t_T} = \frac{28}{432} = 0.0648 \text{ T/cycle}$$

Step 3: Time in cycles
$$T_{cycles} = \frac{N_T}{r} = \frac{2 \times 10^{10}}{0.0648} = 3.09 \times 10^{11} \text{ cycles}$$

Step 4: Convert to seconds
$$T_{real} = 3.09 \times 10^{11} \times 100 \times 10^{-9} = 3.09 \times 10^4 s$$

Step 5: Convert to hours
$$T = \frac{3.09 \times 10^4}{3600} = 8.6 \text{ hours}$$

$$\boxed{T \approx 8.6 \text{ hours} \checkmark}$$

---

### Example 2: Factory Count Optimization

**Problem:** An algorithm has T-count $10^{11}$. You need to complete it in under 24 hours. Each factory produces 1 T-state per 200 cycles. Cycle time is 1 μs. How many factories are required?

**Solution:**

Step 1: Available time in cycles
$$T_{available} = 24 \times 3600 \times 10^6 = 8.64 \times 10^{10} \text{ cycles}$$

Step 2: Required T-production rate
$$r_{required} = \frac{10^{11}}{8.64 \times 10^{10}} = 1.157 \text{ T/cycle}$$

Step 3: Rate per factory
$$r_{factory} = \frac{1}{200} = 0.005 \text{ T/cycle}$$

Step 4: Number of factories
$$n_f = \frac{1.157}{0.005} = 231.5$$

Round up: $n_f = 232$ factories

Step 5: Verify
$$T = \frac{10^{11}}{232 \times 0.005} = 8.62 \times 10^{10} \text{ cycles} = 23.9 \text{ hours } \checkmark$$

$$\boxed{n_f = 232 \text{ factories}}$$

---

### Example 3: Algorithm Comparison

**Problem:** Two algorithms solve the same problem:
- Algorithm A: 1000 logical qubits, T-count $10^{11}$
- Algorithm B: 2000 logical qubits, T-count $10^{10}$

Which is faster assuming 20 factories each, d=17, 1 μs cycles?

**Solution:**

For Algorithm A:
$$T_A = \frac{10^{11}}{20 / (8 \times 17)} = \frac{10^{11}}{0.147} = 6.8 \times 10^{11} \text{ cycles}$$

For Algorithm B:
$$T_B = \frac{10^{10}}{0.147} = 6.8 \times 10^{10} \text{ cycles}$$

Convert to hours:
$$T_A = \frac{6.8 \times 10^{11} \times 10^{-6}}{3600} = 189 \text{ hours}$$
$$T_B = \frac{6.8 \times 10^{10} \times 10^{-6}}{3600} = 18.9 \text{ hours}$$

$$\boxed{\text{Algorithm B is 10× faster despite using 2× more qubits}}$$

**Insight**: T-count reduction dominates over qubit overhead for runtime.

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate runtime for:
- T-count: $5 \times 10^9$
- Factories: 50
- Distillation time: 150 cycles/T-state
- Cycle time: 500 ns

**Problem 1.2:** How many factories are needed to run an algorithm with T-count $10^{12}$ in 100 hours, assuming 200 cycles per T-state and 1 μs cycles?

**Problem 1.3:** An algorithm takes 48 hours with 20 factories. How long with 80 factories?

### Level 2: Intermediate Analysis

**Problem 2.1:** Compare two hardware platforms:
- Platform A: 1 μs cycles, d=27
- Platform B: 0.1 μs cycles, d=17

For the same algorithm (T-count $10^{10}$, 30 factories), which is faster?

**Problem 2.2:** A VQE algorithm has:
- 100 variational parameters
- 1000 iterations expected
- $10^5$ T-gates per iteration
- 5 factories available

Estimate total runtime. What is the bottleneck?

**Problem 2.3:** Derive the break-even point where Algorithm A ($N_T^A$, $n^A$ qubits) and Algorithm B ($N_T^B$, $n^B$ qubits) have equal runtime, given a fixed total qubit budget $Q$.

### Level 3: Challenging Problems

**Problem 3.1:** **Optimal Resource Allocation**
You have a budget of 1 million physical qubits and need to minimize runtime. The algorithm needs 100 logical qubits and has T-count $10^9$. Find the optimal code distance and factory count.

**Problem 3.2:** **Pipeline Optimization**
Design an optimal factory pipeline for an algorithm that alternates between:
- Phase 1: 10 T-gates/cycle for 10^6 cycles
- Phase 2: 0.1 T-gates/cycle for 10^7 cycles

Minimize the total factory footprint while meeting demand.

**Problem 3.3:** **Scheduling Problem**
An algorithm has the following T-gate pattern over time:

| Phase | Duration | T-rate |
|-------|----------|--------|
| Init | $10^4$ cycles | 1 T/cycle |
| Compute | $10^8$ cycles | 0.001 T/cycle |
| Measure | $10^5$ cycles | 0.1 T/cycle |

Design a dynamic factory allocation strategy and calculate total runtime.

---

## Computational Lab

### Runtime Analysis Calculator

```python
"""
Day 893: Runtime Analysis Calculator
Comprehensive runtime estimation for fault-tolerant quantum algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

@dataclass
class HardwareParams:
    """Hardware-specific parameters."""
    cycle_time_us: float = 1.0  # Code cycle time in microseconds
    physical_error_rate: float = 1e-3
    threshold_error_rate: float = 1e-2


@dataclass
class FactoryParams:
    """Factory configuration parameters."""
    n_factories: int
    distillation_cycles: int = 136  # 8d for d=17
    factory_area: int = 43350  # 150d² for d=17
    output_error: float = 1e-15


@dataclass
class AlgorithmProfile:
    """Complete algorithm specification."""
    name: str
    n_logical_qubits: int
    t_count: int
    circuit_depth: int
    parallelism_factor: float = 1.0
    description: str = ""


class RuntimeAnalyzer:
    """Analyze and optimize quantum algorithm runtimes."""

    def __init__(
        self,
        hardware: HardwareParams,
        code_distance: int = 17
    ):
        self.hardware = hardware
        self.d = code_distance

    def calculate_runtime(
        self,
        algorithm: AlgorithmProfile,
        factory: FactoryParams
    ) -> Dict:
        """
        Calculate complete runtime analysis.

        Returns detailed breakdown of runtime components.
        """
        # Factory production rate
        t_production = factory.distillation_cycles
        factory_rate = factory.n_factories / t_production  # T-states per cycle

        # Time components
        t_factory_limited = algorithm.t_count / factory_rate
        t_depth_limited = algorithm.circuit_depth / algorithm.parallelism_factor
        t_clifford = algorithm.circuit_depth * 0.01  # Clifford overhead ~1%

        # Bottleneck identification
        bottleneck_time = max(t_factory_limited, t_depth_limited, t_clifford)

        if t_factory_limited >= t_depth_limited:
            bottleneck = "Factory-limited"
        else:
            bottleneck = "Depth-limited"

        # Convert to real time
        runtime_cycles = bottleneck_time
        runtime_seconds = runtime_cycles * self.hardware.cycle_time_us * 1e-6
        runtime_hours = runtime_seconds / 3600

        return {
            'algorithm': algorithm.name,
            't_count': algorithm.t_count,
            'n_factories': factory.n_factories,
            'factory_rate': factory_rate,
            't_factory_limited_cycles': t_factory_limited,
            't_depth_limited_cycles': t_depth_limited,
            'bottleneck': bottleneck,
            'runtime_cycles': runtime_cycles,
            'runtime_seconds': runtime_seconds,
            'runtime_hours': runtime_hours,
            'runtime_days': runtime_hours / 24
        }

    def optimize_factories(
        self,
        algorithm: AlgorithmProfile,
        target_runtime_hours: float,
        max_factories: int = 1000
    ) -> Tuple[int, Dict]:
        """Find minimum factories needed for target runtime."""
        for n in range(1, max_factories + 1):
            factory = FactoryParams(
                n_factories=n,
                distillation_cycles=8 * self.d * 2  # Two-level
            )
            result = self.calculate_runtime(algorithm, factory)

            if result['runtime_hours'] <= target_runtime_hours:
                return n, result

        return max_factories, result

    def print_report(self, result: Dict):
        """Print formatted runtime report."""
        print("\n" + "="*60)
        print(f"RUNTIME ANALYSIS: {result['algorithm']}")
        print("="*60)
        print(f"T-count:           {result['t_count']:.2e}")
        print(f"Factories:         {result['n_factories']}")
        print(f"Factory rate:      {result['factory_rate']:.4f} T/cycle")
        print("-"*60)
        print(f"Factory-limited:   {result['t_factory_limited_cycles']:.2e} cycles")
        print(f"Depth-limited:     {result['t_depth_limited_cycles']:.2e} cycles")
        print(f"BOTTLENECK:        {result['bottleneck']}")
        print("-"*60)
        print(f"Runtime (cycles):  {result['runtime_cycles']:.2e}")
        print(f"Runtime (seconds): {result['runtime_seconds']:.2e}")
        print(f"Runtime (hours):   {result['runtime_hours']:.2f}")
        print(f"Runtime (days):    {result['runtime_days']:.2f}")
        print("="*60)


def benchmark_algorithms():
    """Define and analyze standard benchmark algorithms."""
    algorithms = [
        AlgorithmProfile(
            name="RSA-2048",
            n_logical_qubits=6189,
            t_count=int(2e10),
            circuit_depth=int(2e10),
            description="Shor's algorithm for 2048-bit RSA"
        ),
        AlgorithmProfile(
            name="RSA-4096",
            n_logical_qubits=12000,
            t_count=int(2e11),
            circuit_depth=int(2e11),
            description="Shor's algorithm for 4096-bit RSA"
        ),
        AlgorithmProfile(
            name="ECDLP-256",
            n_logical_qubits=2330,
            t_count=int(5e9),
            circuit_depth=int(5e9),
            description="Discrete log on 256-bit elliptic curve"
        ),
        AlgorithmProfile(
            name="FeMoco",
            n_logical_qubits=4000,
            t_count=int(1e13),
            circuit_depth=int(1e13),
            description="Nitrogen fixation catalyst simulation"
        ),
        AlgorithmProfile(
            name="Hubbard-10x10",
            n_logical_qubits=200,
            t_count=int(1e11),
            circuit_depth=int(1e11),
            description="10x10 Hubbard model ground state"
        ),
        AlgorithmProfile(
            name="QAOA-100",
            n_logical_qubits=100,
            t_count=int(1e8),
            circuit_depth=int(1e6),
            parallelism_factor=10,
            description="QAOA with 100 variables, p=100"
        ),
        AlgorithmProfile(
            name="VQE-50",
            n_logical_qubits=50,
            t_count=int(1e6),
            circuit_depth=int(1e4),
            parallelism_factor=5,
            description="VQE for small molecule"
        )
    ]
    return algorithms


def plot_runtime_scaling():
    """Visualize runtime scaling with factories."""
    hardware = HardwareParams(cycle_time_us=1.0)
    analyzer = RuntimeAnalyzer(hardware, code_distance=17)

    # RSA-2048 as example
    rsa = AlgorithmProfile(
        name="RSA-2048",
        n_logical_qubits=6189,
        t_count=int(2e10),
        circuit_depth=int(2e10)
    )

    factory_counts = [5, 10, 20, 50, 100, 200, 500]
    runtimes = []

    for n in factory_counts:
        factory = FactoryParams(n_factories=n, distillation_cycles=136 * 2)
        result = analyzer.calculate_runtime(rsa, factory)
        runtimes.append(result['runtime_hours'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear plot
    axes[0].plot(factory_counts, runtimes, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Factories', fontsize=12)
    axes[0].set_ylabel('Runtime (hours)', fontsize=12)
    axes[0].set_title('RSA-2048 Runtime vs. Factory Count', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=8, color='r', linestyle='--', label='8-hour target')
    axes[0].legend()

    # Log-log plot
    axes[1].loglog(factory_counts, runtimes, 'b-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Factories', fontsize=12)
    axes[1].set_ylabel('Runtime (hours)', fontsize=12)
    axes[1].set_title('Log-Log Scaling', fontsize=14)
    axes[1].grid(True, alpha=0.3, which='both')

    # Fit 1/n curve
    n_fit = np.linspace(5, 500, 100)
    t_fit = (2e10 * 272) / (n_fit * 1e6 * 3600)  # Approximate
    axes[1].plot(n_fit, t_fit, 'g--', linewidth=1.5, label='1/n scaling')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('runtime_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_algorithm_comparison():
    """Compare runtimes across benchmark algorithms."""
    hardware = HardwareParams(cycle_time_us=0.1)  # 100 ns cycles
    analyzer = RuntimeAnalyzer(hardware, code_distance=27)
    algorithms = benchmark_algorithms()

    names = []
    runtimes = []
    t_counts = []

    # Use different factory counts based on algorithm scale
    factory_configs = {
        'RSA-2048': 28,
        'RSA-4096': 56,
        'ECDLP-256': 20,
        'FeMoco': 200,
        'Hubbard-10x10': 50,
        'QAOA-100': 5,
        'VQE-50': 2
    }

    for algo in algorithms:
        n_factories = factory_configs.get(algo.name, 20)
        factory = FactoryParams(n_factories=n_factories, distillation_cycles=8*27*2)
        result = analyzer.calculate_runtime(algo, factory)

        names.append(algo.name)
        runtimes.append(result['runtime_hours'])
        t_counts.append(algo.t_count)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(names, runtimes, color='steelblue', edgecolor='black')
    ax.set_ylabel('Runtime (hours)', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_title('Estimated Runtime for Benchmark Algorithms\n(100 ns cycle time, d=27)', fontsize=14)
    ax.set_yscale('log')
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, runtime in zip(bars, runtimes):
        if runtime < 24:
            label = f'{runtime:.1f}h'
        else:
            label = f'{runtime/24:.1f}d'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Reference lines
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='1 hour')
    ax.axhline(y=8, color='orange', linestyle='--', alpha=0.7, label='8 hours')
    ax.axhline(y=24, color='red', linestyle='--', alpha=0.7, label='1 day')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_hardware_comparison():
    """Compare runtimes across different hardware parameters."""
    rsa = AlgorithmProfile(
        name="RSA-2048",
        n_logical_qubits=6189,
        t_count=int(2e10),
        circuit_depth=int(2e10)
    )

    configs = [
        ('Current (1μs, d=27)', 1.0, 27),
        ('Near-term (0.5μs, d=21)', 0.5, 21),
        ('Advanced (0.1μs, d=17)', 0.1, 17),
        ('Future (0.01μs, d=13)', 0.01, 13),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, cycle_us, d in configs:
        hardware = HardwareParams(cycle_time_us=cycle_us)
        analyzer = RuntimeAnalyzer(hardware, code_distance=d)

        factory_counts = np.arange(10, 201, 10)
        runtimes = []

        for n in factory_counts:
            factory = FactoryParams(n_factories=n, distillation_cycles=8*d*2)
            result = analyzer.calculate_runtime(rsa, factory)
            runtimes.append(result['runtime_hours'])

        ax.semilogy(factory_counts, runtimes, '-o', label=name, markersize=4)

    ax.set_xlabel('Number of Factories', fontsize=12)
    ax.set_ylabel('Runtime (hours)', fontsize=12)
    ax.set_title('RSA-2048 Runtime: Hardware Evolution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(y=8, color='black', linestyle='--', alpha=0.5, label='8-hour target')

    plt.tight_layout()
    plt.savefig('hardware_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def sensitivity_analysis():
    """Analyze sensitivity of runtime to various parameters."""
    base_hardware = HardwareParams(cycle_time_us=1.0)
    base_d = 17

    rsa = AlgorithmProfile(
        name="RSA-2048",
        n_logical_qubits=6189,
        t_count=int(2e10),
        circuit_depth=int(2e10)
    )

    # Base case
    analyzer = RuntimeAnalyzer(base_hardware, base_d)
    factory = FactoryParams(n_factories=28, distillation_cycles=8*base_d*2)
    base_result = analyzer.calculate_runtime(rsa, factory)
    base_runtime = base_result['runtime_hours']

    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"Base case: {base_runtime:.2f} hours")
    print("-"*60)

    # Sensitivity to each parameter
    sensitivities = []

    # T-count sensitivity
    for factor in [0.5, 0.75, 1.25, 1.5]:
        algo_mod = AlgorithmProfile(
            name="Modified",
            n_logical_qubits=rsa.n_logical_qubits,
            t_count=int(rsa.t_count * factor),
            circuit_depth=int(rsa.circuit_depth * factor)
        )
        result = analyzer.calculate_runtime(algo_mod, factory)
        print(f"T-count × {factor}: {result['runtime_hours']:.2f} hours ({result['runtime_hours']/base_runtime:.2f}×)")

    print("-"*60)

    # Factory count sensitivity
    for n in [14, 28, 56, 112]:
        factory_mod = FactoryParams(n_factories=n, distillation_cycles=8*base_d*2)
        result = analyzer.calculate_runtime(rsa, factory_mod)
        print(f"Factories = {n}: {result['runtime_hours']:.2f} hours ({result['runtime_hours']/base_runtime:.2f}×)")

    print("-"*60)

    # Cycle time sensitivity
    for cycle_us in [2.0, 1.0, 0.5, 0.1]:
        hw_mod = HardwareParams(cycle_time_us=cycle_us)
        analyzer_mod = RuntimeAnalyzer(hw_mod, base_d)
        result = analyzer_mod.calculate_runtime(rsa, factory)
        print(f"Cycle = {cycle_us}μs: {result['runtime_hours']:.2f} hours ({result['runtime_hours']/base_runtime:.2f}×)")

    print("="*60)


# Main demonstration
if __name__ == "__main__":
    print("Runtime Analysis Calculator - Day 893")
    print("="*50)

    # Setup
    hardware = HardwareParams(cycle_time_us=0.1)  # 100 ns
    analyzer = RuntimeAnalyzer(hardware, code_distance=27)

    # Analyze RSA-2048
    print("\n--- RSA-2048 Analysis ---")
    rsa = AlgorithmProfile(
        name="RSA-2048",
        n_logical_qubits=6189,
        t_count=int(2e10),
        circuit_depth=int(2e10)
    )
    factory = FactoryParams(n_factories=28, distillation_cycles=8*27*2)
    result = analyzer.calculate_runtime(rsa, factory)
    analyzer.print_report(result)

    # Find optimal factories for 8-hour target
    print("\n--- Factory Optimization ---")
    opt_n, opt_result = analyzer.optimize_factories(rsa, target_runtime_hours=8.0)
    print(f"Minimum factories for 8-hour runtime: {opt_n}")
    analyzer.print_report(opt_result)

    # Analyze all benchmarks
    print("\n--- Benchmark Suite ---")
    algorithms = benchmark_algorithms()
    for algo in algorithms:
        n_factories = max(5, int(algo.t_count / 1e9))  # Rough estimate
        n_factories = min(n_factories, 500)
        factory = FactoryParams(n_factories=n_factories, distillation_cycles=8*27*2)
        result = analyzer.calculate_runtime(algo, factory)
        print(f"{algo.name:15s}: {result['runtime_hours']:10.2f} hours ({n_factories} factories)")

    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    plot_runtime_scaling()
    plot_algorithm_comparison()
    plot_hardware_comparison()

    # Sensitivity analysis
    sensitivity_analysis()

    print("\nRuntime analysis complete!")
```

---

## Summary

### Key Formulas

| Formula | Expression | Description |
|---------|------------|-------------|
| Runtime (factory-limited) | $T = \frac{N_T \cdot t_{distill}}{n_f}$ | Primary formula |
| Factory production rate | $r = n_f / t_{distill}$ | T-states per cycle |
| Real time conversion | $T_{real} = T_{cycles} \times \tau_{cycle}$ | Cycles to seconds |
| Bottleneck condition | $T = \max(T_{factory}, T_{depth})$ | Limiting factor |

### Benchmark Reference Values

| Algorithm | T-count | Factories | Runtime |
|-----------|---------|-----------|---------|
| RSA-2048 | $2 \times 10^{10}$ | 28 | ~8 hours |
| FeMoco | $10^{13}$ | 200 | ~weeks |
| ECDLP-256 | $5 \times 10^9$ | 20 | ~2 hours |
| QAOA-100 | $10^8$ | 5 | ~30 min |

### Key Insights

1. **T-count dominates**: Most algorithms are factory-limited, not depth-limited
2. **Linear scaling**: Runtime scales as $1/n_f$ with factory count
3. **Hardware matters**: 10× faster cycles = 10× shorter runtime
4. **Trade-off space**: More factories = faster time but more qubits

---

## Daily Checklist

- [ ] I can derive runtime from T-count and factory parameters
- [ ] I can identify factory-limited vs. depth-limited regimes
- [ ] I understand the RSA-2048 benchmark (~8 hours, ~20M qubits)
- [ ] I can estimate quantum chemistry runtimes
- [ ] I can apply optimization strategies for runtime reduction
- [ ] I can use the runtime calculator tool
- [ ] I understand the sensitivity of runtime to parameters

---

## Preview: Day 894

Tomorrow we explore **Code Choice Comparison**:

- Surface code overhead analysis
- Color code advantages and disadvantages
- Concatenated code trade-offs
- LDPC codes: future potential
- Optimization for specific algorithm types

Different error-correcting codes offer different trade-offs—choosing the right code can dramatically impact resource requirements.

---

*Day 893 of 2184 | Week 128 of 312 | Month 32 of 72*

*"The runtime of a quantum algorithm is written in T-gates."*
