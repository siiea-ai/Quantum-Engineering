# Day 769: Resource Scaling Analysis

## Overview

**Day:** 769 of 1008
**Week:** 110 (Threshold Theorems & Analysis)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Quantifying the Cost of Fault-Tolerant Quantum Computation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Qubit and gate overhead |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Space-time tradeoffs |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Architecture comparison |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Derive** qubit overhead formulas for different code families
2. **Compute** gate overhead for fault-tolerant operations
3. **Analyze** time overhead including logical clock cycles
4. **Optimize** space-time tradeoffs
5. **Compare** resource costs across architectures
6. **Plan** realistic resource requirements for quantum algorithms

---

## Core Content

### 1. Qubit Overhead

The number of physical qubits needed per logical qubit.

#### Concatenated Codes

For an [[n,k,d]] code with $\ell$ levels of concatenation:

$$\boxed{N_{phys} = \left(\frac{n}{k}\right)^\ell \cdot N_{logical}}$$

**Example:** [[7,1,3]] code with 4 levels:
$$N_{phys} = 7^4 = 2401 \text{ physical qubits per logical qubit}$$

#### Surface Codes

For surface code achieving logical error rate $\epsilon_L$:

**Code distance required:**
$$d \approx \frac{\log(1/\epsilon_L)}{\log(p_{th}/p)} + O(1)$$

**Physical qubits per logical:**
$$\boxed{N_{phys} = 2d^2 - 1 \approx 2d^2}$$

**Total scaling:**
$$N_{phys} \sim O\left(\log^2(1/\epsilon_L)\right)$$

#### Comparison

| Code Family | Qubit Scaling | Typical Value |
|-------------|---------------|---------------|
| Concatenated | $n^\ell$ | 2,401 (4 levels) |
| Surface | $2d^2$ | 578 (d=17) |
| Color | $\frac{3d^2+1}{4}$ | 217 (d=17) |

### 2. Gate Overhead

Cost of implementing logical gates fault-tolerantly.

#### Clifford Gates

For transversal or code-switching Clifford gates:

$$\boxed{G_{Clifford} = O(d) \text{ physical gates per logical gate}}$$

**Surface code:**
- Transversal CNOT: $d^2$ physical CNOTs
- S gate via code deformation: $O(d)$ time steps

#### T-Gate and Magic State Distillation

T-gate requires magic state distillation:

$$\boxed{G_T = G_{distill} \cdot \frac{1}{\gamma_{success}}}$$

**15-to-1 distillation:**
- Input: 15 noisy T-states with error $p_T$
- Output: 1 T-state with error $\sim 35 p_T^3$
- Success probability: $\sim 1 - 15 p_T$

**Multi-level distillation:**
$$\epsilon_T^{(k)} \sim (c \cdot \epsilon_T^{(0)})^{3^k}$$

Total overhead:
$$\boxed{G_T \sim O\left(\log^\gamma(1/\epsilon_T)\right) \quad \text{where } \gamma \approx 2.5}$$

#### Total Gate Count

For algorithm with $L$ logical gates, $L_T$ T-gates:

$$\boxed{G_{total} = L \cdot G_{Clifford} + L_T \cdot G_T}$$

### 3. Time Overhead

Duration of fault-tolerant computation.

#### Logical Clock Cycle

One logical clock cycle includes:
1. Apply logical gate
2. Syndrome extraction
3. Classical decoding
4. Error correction

$$\boxed{\tau_{logical} = \tau_{gate} + \tau_{syndrome} + \tau_{decode} + \tau_{correct}}$$

**Surface code:**
- $\tau_{syndrome} = d$ rounds of measurement
- $\tau_{decode} = O(d^2)$ classical (can be parallelized)

#### Total Computation Time

$$\boxed{T_{total} = D_{circuit} \cdot \tau_{logical}}$$

where $D_{circuit}$ is the circuit depth in logical gates.

#### Reaction Time Constraint

Decoder must complete before errors accumulate:
$$\boxed{\tau_{decode} < \tau_{coherence} / d}$$

This limits practical code distances!

### 4. Space-Time Tradeoffs

Balancing qubit count vs computation time.

#### Magic State Factories

**Parallel factories:** More qubits, faster T-gates
$$T_{T-gate} \propto \frac{1}{N_{factories}}$$

**Factory footprint:**
$$N_{factory} \sim 10d^2 \text{ to } 100d^2 \text{ qubits}$$

#### Code Distance Selection

Higher $d$:
- Pro: Lower logical error rate
- Con: More qubits, slower gates

**Optimal distance:**
$$\boxed{d^* = \arg\min_d \left[ N_{phys}(d) \cdot T_{total}(d) \right]}$$

subject to $\epsilon_L(d) \leq \epsilon_{target}$.

#### Lattice Surgery vs Braiding

| Method | Space | Time | Flexibility |
|--------|-------|------|-------------|
| Braiding | Lower | Higher | Limited |
| Lattice surgery | Higher | Lower | High |

### 5. Architecture-Specific Costs

Different hardware has different constraints.

#### Superconducting Qubits

**Parameters:**
- Gate time: ~20-50 ns
- Measurement: ~300-500 ns
- T1, T2: ~50-100 μs
- Connectivity: Typically 2D grid

**Implications:**
- Surface code natural fit
- Measurement-limited cycle time
- ~1 μs logical clock cycle

#### Trapped Ions

**Parameters:**
- Gate time: ~10-100 μs (two-qubit)
- Measurement: ~100 μs
- Coherence: ~seconds to minutes
- Connectivity: All-to-all (small chains)

**Implications:**
- Can use higher-distance codes
- Shuttling overhead for scaling
- ~1 ms logical clock cycle

#### Photonics

**Parameters:**
- Gate time: ~ps-ns
- Measurement: ~ns
- Loss: Major error source
- Connectivity: Reconfigurable

**Implications:**
- Fusion-based computing
- Loss-tolerant codes essential
- Very fast but probabilistic

### 6. Practical Resource Estimates

Real numbers for algorithms of interest.

#### Shor's Algorithm for RSA-2048

**Classical estimate:** $L \sim 10^{10}$ gates, $L_T \sim 10^9$ T-gates

**Surface code resources:**
$$N_{logical} \approx 2n + O(\log n) \approx 4100 \text{ logical qubits}$$
$$d \approx 27 \text{ (for } 10^{-15} \text{ logical error)}$$
$$N_{physical} \approx 4100 \times 2 \times 27^2 \approx 6 \text{ million qubits}$$

**Time:**
$$T_{total} \approx 8 \text{ hours (with 10 ns clock)}$$

#### Quantum Chemistry (FeMoCo)

**Estimate:** $N_{logical} \sim 100$, $L_T \sim 10^{11}$ T-gates

**Resources:**
$$N_{physical} \sim 1 \text{ million qubits}$$
$$T_{total} \sim 1 \text{ week}$$

---

## Worked Examples

### Example 1: Surface Code Qubit Overhead

**Problem:** How many physical qubits are needed for 100 logical qubits at logical error rate $10^{-10}$ with physical error rate $p = 0.1\%$?

**Solution:**

Step 1: Find required code distance.
$$p_L \approx c \cdot \left(\frac{p}{p_{th}}\right)^{(d+1)/2}$$

With $p = 0.001$, $p_{th} = 0.01$, $c = 0.1$:
$$10^{-10} = 0.1 \cdot (0.1)^{(d+1)/2}$$
$$10^{-9} = 10^{-(d+1)/2}$$
$$(d+1)/2 = 9$$
$$d = 17$$

Step 2: Physical qubits per logical.
$$n_{phys} = 2d^2 = 2 \times 17^2 = 578$$

Step 3: Total physical qubits.
$$N_{phys} = 100 \times 578 = 57,800$$

Step 4: Add magic state factories (~20% overhead).
$$N_{total} \approx 57,800 \times 1.2 \approx \boxed{69,000 \text{ physical qubits}}$$

### Example 2: T-Gate Distillation Overhead

**Problem:** Starting with T-states at error $p_T = 1\%$, how many raw T-states are needed to produce one T-state at error $10^{-12}$?

**Solution:**

Using 15-to-1 distillation:
$$\epsilon^{(k)} \approx 35 \cdot (\epsilon^{(k-1)})^3$$

Level 0: $\epsilon^{(0)} = 0.01$
Level 1: $\epsilon^{(1)} \approx 35 \times 10^{-6} = 3.5 \times 10^{-5}$
Level 2: $\epsilon^{(2)} \approx 35 \times (3.5 \times 10^{-5})^3 \approx 1.5 \times 10^{-12}$

Two levels needed!

Raw T-states required:
$$N_{raw} = 15^2 = 225$$

Accounting for failure probability (~15% each level):
$$N_{raw}^{effective} \approx \frac{225}{0.85^2} \approx \boxed{311 \text{ raw T-states}}$$

### Example 3: Logical Clock Cycle

**Problem:** For a surface code with $d=15$ on superconducting hardware (20 ns gates, 300 ns measurement), what is the logical clock cycle time?

**Solution:**

Syndrome extraction requires $d$ rounds:
$$\tau_{syndrome} = d \times (\tau_{2Q} + \tau_{meas}) = 15 \times (20 + 300) = 4,800 \text{ ns}$$

Classical decoding (assuming fast MWPM):
$$\tau_{decode} \approx 100 \text{ ns (parallelized)}$$

Logical gate (if transversal):
$$\tau_{gate} \approx 20 \text{ ns}$$

Total logical clock cycle:
$$\tau_{logical} = 4800 + 100 + 20 \approx \boxed{5 \text{ μs}}$$

Logical gates per second:
$$f_{logical} \approx 200,000 \text{ Hz} = 200 \text{ kHz}$$

---

## Practice Problems

### Problem Set A: Qubit Overhead

**A1.** Compare qubit overhead for [[7,1,3]] concatenated code vs surface code to achieve $\epsilon_L = 10^{-15}$ with $p = 0.1\%$.

**A2.** For a color code with $n = (3d^2+1)/4$, derive the qubit scaling as a function of target logical error rate.

**A3.** Design a hybrid architecture using concatenated codes inside surface code patches. What are the tradeoffs?

### Problem Set B: Gate and Time Overhead

**B1.** For an algorithm with 1000 logical T-gates, estimate total T-gate overhead with and without parallel magic state factories.

**B2.** Derive the maximum achievable logical clock rate as a function of code distance and physical gate speeds.

**B3.** If decoder latency scales as $O(d^2)$, at what distance does decoding become the bottleneck for a 1 GHz physical clock?

### Problem Set C: Architecture Optimization

**C1.** Optimize code distance for minimum space-time volume given $p = 0.5\%$, $\epsilon_L = 10^{-10}$, and 1000 logical qubits.

**C2.** Compare total resources for factoring RSA-2048 on superconducting vs trapped ion architectures.

**C3.** Design a magic state factory layout that minimizes latency for a surface code on a 2D grid.

---

## Computational Lab

```python
"""
Day 769 Computational Lab: Resource Scaling Analysis
====================================================

Analyze and optimize fault-tolerant quantum computing resources.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class HardwareParams:
    """Hardware-specific parameters."""
    name: str
    gate_time_1q: float  # seconds
    gate_time_2q: float  # seconds
    measurement_time: float  # seconds
    t1: float  # seconds
    t2: float  # seconds
    physical_error_rate: float
    connectivity: str


@dataclass
class ResourceEstimate:
    """Complete resource estimate for a computation."""
    logical_qubits: int
    code_distance: int
    physical_qubits: int
    total_gates: int
    t_gates: int
    computation_time: float
    logical_error_rate: float


# Hardware profiles
SUPERCONDUCTING = HardwareParams(
    name="Superconducting",
    gate_time_1q=20e-9,
    gate_time_2q=50e-9,
    measurement_time=300e-9,
    t1=100e-6,
    t2=50e-6,
    physical_error_rate=0.001,
    connectivity="2D grid"
)

TRAPPED_ION = HardwareParams(
    name="Trapped Ion",
    gate_time_1q=1e-6,
    gate_time_2q=100e-6,
    measurement_time=100e-6,
    t1=10.0,  # seconds!
    t2=1.0,
    physical_error_rate=0.001,
    connectivity="All-to-all"
)


def surface_code_distance(p_phys: float, p_logical: float,
                         p_th: float = 0.01, c: float = 0.1) -> int:
    """
    Compute required surface code distance.

    p_L = c * (p/p_th)^((d+1)/2)
    """
    if p_phys >= p_th:
        return float('inf')

    ratio = p_phys / p_th

    # Solve for d: p_L = c * ratio^((d+1)/2)
    # log(p_L/c) = ((d+1)/2) * log(ratio)
    # d = 2 * log(p_L/c) / log(ratio) - 1

    d = 2 * np.log(p_logical / c) / np.log(ratio) - 1
    return int(np.ceil(d)) | 1  # Round up to odd number


def concatenation_levels(p_phys: float, p_logical: float,
                        p_th: float = 0.01) -> int:
    """
    Compute required concatenation levels.

    p_L^(k) = p_th * (p/p_th)^(2^k)
    """
    if p_phys >= p_th:
        return float('inf')

    ratio = p_phys / p_th

    # Solve: p_L = p_th * ratio^(2^k)
    # log(p_L/p_th) = 2^k * log(ratio)
    # k = log2(log(p_L/p_th) / log(ratio))

    k = np.log2(np.log(p_logical / p_th) / np.log(ratio))
    return int(np.ceil(k))


def surface_code_qubits(d: int) -> int:
    """Physical qubits per logical qubit for surface code."""
    return 2 * d * d - 1


def concatenated_code_qubits(n: int, k: int, levels: int) -> int:
    """Physical qubits per logical qubit for concatenated code."""
    return (n // k) ** levels


def magic_state_factory_size(d: int, distillation_levels: int = 2) -> int:
    """
    Qubits needed for magic state factory.

    Rough estimate: 15^levels * surface_code_patch
    """
    factory_patches = 15 ** distillation_levels
    qubits_per_patch = surface_code_qubits(d)
    return factory_patches * qubits_per_patch


def logical_clock_cycle(d: int, hw: HardwareParams) -> float:
    """
    Compute logical clock cycle time.

    Includes syndrome extraction and decoding.
    """
    # Syndrome extraction: d rounds
    syndrome_time = d * (hw.gate_time_2q + hw.measurement_time)

    # Classical decoding (assume fast)
    decode_time = 100e-9 * d  # Scales with d for MWPM

    # Logical gate
    gate_time = hw.gate_time_2q

    return syndrome_time + decode_time + gate_time


def t_gate_distillation_overhead(p_in: float, p_target: float,
                                 protocol: str = "15-to-1") -> int:
    """
    Raw T-states needed per distilled T-state.
    """
    if protocol == "15-to-1":
        # p_out = 35 * p_in^3
        levels = 0
        p_current = p_in
        while p_current > p_target and levels < 10:
            p_current = 35 * p_current ** 3
            levels += 1

        # Each level uses 15 input states
        return 15 ** levels

    return 1


def estimate_resources(n_logical: int, circuit_depth: int,
                      n_t_gates: int, target_error: float,
                      hw: HardwareParams) -> ResourceEstimate:
    """
    Complete resource estimation for a quantum computation.
    """
    # 1. Code distance
    d = surface_code_distance(hw.physical_error_rate, target_error / circuit_depth)

    # 2. Physical qubits (data + ancilla)
    data_qubits = n_logical * surface_code_qubits(d)

    # 3. Magic state factories
    factory_qubits = magic_state_factory_size(d, distillation_levels=2)
    n_factories = max(1, n_t_gates // 1000)  # Rough heuristic
    total_factory_qubits = n_factories * factory_qubits

    # 4. Total physical qubits
    total_qubits = data_qubits + total_factory_qubits

    # 5. Time
    tau = logical_clock_cycle(d, hw)
    total_time = circuit_depth * tau

    # 6. T-gate overhead
    t_overhead = t_gate_distillation_overhead(hw.physical_error_rate * 10,
                                              target_error / n_t_gates)
    total_t_gates = n_t_gates * t_overhead

    # 7. Total gates
    total_gates = circuit_depth * n_logical * d + total_t_gates

    return ResourceEstimate(
        logical_qubits=n_logical,
        code_distance=d,
        physical_qubits=total_qubits,
        total_gates=total_gates,
        t_gates=total_t_gates,
        computation_time=total_time,
        logical_error_rate=target_error
    )


def optimize_code_distance(n_logical: int, p_phys: float, p_target: float,
                          hw: HardwareParams) -> Tuple[int, float]:
    """
    Find optimal code distance minimizing space-time volume.
    """
    best_d = 3
    best_volume = float('inf')

    for d in range(3, 51, 2):
        # Check if this d achieves target error
        p_L = 0.1 * (p_phys / 0.01) ** ((d + 1) / 2)
        if p_L > p_target:
            continue

        # Compute space-time volume
        qubits = n_logical * surface_code_qubits(d)
        cycle_time = logical_clock_cycle(d, hw)
        volume = qubits * cycle_time

        if volume < best_volume:
            best_volume = volume
            best_d = d

    return best_d, best_volume


def compare_architectures(n_logical: int, depth: int, n_t: int,
                         target_error: float) -> Dict[str, ResourceEstimate]:
    """Compare resources across hardware platforms."""
    results = {}

    for hw in [SUPERCONDUCTING, TRAPPED_ION]:
        results[hw.name] = estimate_resources(
            n_logical, depth, n_t, target_error, hw
        )

    return results


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 769: RESOURCE SCALING ANALYSIS")
    print("=" * 70)

    # Demo 1: Code distance requirements
    print("\n" + "=" * 70)
    print("Demo 1: Code Distance vs Target Error")
    print("=" * 70)

    print("\nSurface code distance for p_phys = 0.1%:")
    print(f"{'Target Error':<15} {'Distance':<10} {'Qubits/Logical':<15}")
    print("-" * 45)

    for log_error in [6, 9, 12, 15, 18]:
        p_L = 10 ** (-log_error)
        d = surface_code_distance(0.001, p_L)
        n_qubits = surface_code_qubits(d)
        print(f"10^-{log_error:<13} {d:<10} {n_qubits:<15}")

    # Demo 2: Qubit overhead comparison
    print("\n" + "=" * 70)
    print("Demo 2: Qubit Overhead Comparison")
    print("=" * 70)

    print("\nPhysical qubits per logical qubit (target error 10^-12):")
    print(f"{'Code':<25} {'Distance/Levels':<15} {'Qubits':<10}")
    print("-" * 55)

    # Surface code
    d_surf = surface_code_distance(0.001, 1e-12)
    n_surf = surface_code_qubits(d_surf)
    print(f"{'Surface code':<25} {'d = ' + str(d_surf):<15} {n_surf:<10}")

    # Concatenated [[7,1,3]]
    levels_concat = concatenation_levels(0.001, 1e-12)
    n_concat = concatenated_code_qubits(7, 1, levels_concat)
    print(f"{'Concatenated [[7,1,3]]':<25} {'k = ' + str(levels_concat):<15} {n_concat:<10}")

    # Concatenated [[5,1,3]]
    n_concat_5 = concatenated_code_qubits(5, 1, levels_concat)
    print(f"{'Concatenated [[5,1,3]]':<25} {'k = ' + str(levels_concat):<15} {n_concat_5:<10}")

    # Demo 3: Logical clock cycle
    print("\n" + "=" * 70)
    print("Demo 3: Logical Clock Cycle")
    print("=" * 70)

    print("\nClock cycle time for different distances:")
    print(f"{'Distance':<10} {'Superconducting':<20} {'Trapped Ion':<20}")
    print("-" * 55)

    for d in [5, 11, 17, 23, 29]:
        tau_sc = logical_clock_cycle(d, SUPERCONDUCTING)
        tau_ti = logical_clock_cycle(d, TRAPPED_ION)
        print(f"{d:<10} {tau_sc*1e6:<20.2f} μs {tau_ti*1e3:<20.2f} ms")

    # Demo 4: T-gate overhead
    print("\n" + "=" * 70)
    print("Demo 4: Magic State Distillation Overhead")
    print("=" * 70)

    print("\nRaw T-states per distilled T-state:")
    print(f"{'Input Error':<15} {'Target Error':<15} {'Raw T-states':<15}")
    print("-" * 50)

    for p_in in [0.01, 0.005, 0.001]:
        for p_target in [1e-6, 1e-9, 1e-12]:
            n_raw = t_gate_distillation_overhead(p_in, p_target)
            print(f"{p_in:<15.3f} {p_target:<15.0e} {n_raw:<15}")

    # Demo 5: Full resource estimation
    print("\n" + "=" * 70)
    print("Demo 5: Full Resource Estimation")
    print("=" * 70)

    # Example: Modest quantum chemistry simulation
    n_logical = 50
    depth = 10_000
    n_t = 1_000_000
    target = 1e-6

    print(f"\nAlgorithm parameters:")
    print(f"  Logical qubits: {n_logical}")
    print(f"  Circuit depth: {depth:,}")
    print(f"  T-gates: {n_t:,}")
    print(f"  Target error: {target}")

    results = compare_architectures(n_logical, depth, n_t, target)

    print(f"\n{'Architecture':<20} {'Qubits':<15} {'Time':<15} {'Distance':<10}")
    print("-" * 65)

    for name, res in results.items():
        time_str = f"{res.computation_time:.2f} s" if res.computation_time < 60 else \
                   f"{res.computation_time/60:.2f} min"
        print(f"{name:<20} {res.physical_qubits:<15,} {time_str:<15} {res.code_distance:<10}")

    # Demo 6: Space-time optimization
    print("\n" + "=" * 70)
    print("Demo 6: Space-Time Volume Optimization")
    print("=" * 70)

    print("\nOptimal code distance (minimizing space-time volume):")
    print(f"{'Target Error':<15} {'Optimal d':<12} {'Qubits':<12} {'Cycle (μs)':<12}")
    print("-" * 55)

    for log_err in [6, 9, 12]:
        d_opt, volume = optimize_code_distance(100, 0.001, 10**(-log_err),
                                               SUPERCONDUCTING)
        n_qubits = 100 * surface_code_qubits(d_opt)
        tau = logical_clock_cycle(d_opt, SUPERCONDUCTING)
        print(f"10^-{log_err:<13} {d_opt:<12} {n_qubits:<12,} {tau*1e6:<12.2f}")

    # Demo 7: Scaling projections
    print("\n" + "=" * 70)
    print("Demo 7: Algorithm Resource Projections")
    print("=" * 70)

    algorithms = [
        ("VQE (small molecule)", 20, 1000, 10000, 1e-3),
        ("QAOA (100 nodes)", 100, 100000, 100000, 1e-6),
        ("Quantum chemistry", 200, 1000000, 10000000, 1e-9),
        ("Shor (RSA-2048)", 4100, 1e10, 1e9, 1e-15),
    ]

    print(f"{'Algorithm':<25} {'Logical Q':<12} {'Physical Q':<15} {'Time':<15}")
    print("-" * 70)

    for name, n_log, depth, n_t, err in algorithms:
        res = estimate_resources(n_log, int(depth), int(n_t), err, SUPERCONDUCTING)

        if res.computation_time < 60:
            time_str = f"{res.computation_time:.1f} s"
        elif res.computation_time < 3600:
            time_str = f"{res.computation_time/60:.1f} min"
        elif res.computation_time < 86400:
            time_str = f"{res.computation_time/3600:.1f} hr"
        else:
            time_str = f"{res.computation_time/86400:.1f} days"

        print(f"{name:<25} {res.logical_qubits:<12,} "
              f"{res.physical_qubits:<15,} {time_str:<15}")

    # Summary
    print("\n" + "=" * 70)
    print("RESOURCE SCALING SUMMARY")
    print("=" * 70)

    print("""
    +---------------------------------------------------------------+
    |  FAULT-TOLERANT RESOURCE SCALING                              |
    +---------------------------------------------------------------+
    |                                                               |
    |  QUBIT OVERHEAD:                                              |
    |    Surface code: N_phys = 2d^2 per logical qubit             |
    |    Concatenated: N_phys = n^k per logical qubit              |
    |    Surface typically wins for d > 7                          |
    |                                                               |
    |  CODE DISTANCE:                                               |
    |    d ~ log(1/epsilon) / log(p_th/p)                          |
    |    Rule of thumb: d = 2*log10(1/epsilon) for p = 0.1%        |
    |                                                               |
    |  T-GATE OVERHEAD:                                             |
    |    15-to-1 distillation: 15^k raw states for k levels        |
    |    Typically 100-1000x overhead per T-gate                   |
    |    Dominates for T-gate heavy algorithms                     |
    |                                                               |
    |  CLOCK CYCLE:                                                 |
    |    tau ~ d * (t_gate + t_measure)                            |
    |    Superconducting: ~1-10 microseconds                       |
    |    Trapped ions: ~1-10 milliseconds                          |
    |                                                               |
    |  SPACE-TIME TRADEOFF:                                         |
    |    More factories -> faster T-gates                          |
    |    Higher d -> lower error but slower                        |
    |    Optimize for minimum space-time volume                    |
    |                                                               |
    +---------------------------------------------------------------+
    """)

    # Key formulas
    print("\nKEY SCALING FORMULAS:")
    print("-" * 50)
    print("  Physical qubits: N = O(n_logical * d^2)")
    print("  Code distance: d = O(log(1/epsilon))")
    print("  T-gate overhead: O(log^2.5(1/epsilon_T))")
    print("  Total gates: O(L * d + L_T * T_overhead)")
    print("  Computation time: O(depth * d * t_phys)")

    print("=" * 70)
    print("Day 769 Complete: Resource Scaling Analysis Mastered")
    print("=" * 70)
```

---

## Summary

### Resource Scaling Laws

| Resource | Surface Code | Concatenated |
|----------|--------------|--------------|
| Qubits/logical | $2d^2$ | $n^k$ |
| Distance for $\epsilon_L$ | $O(\log 1/\epsilon_L)$ | $O(\log\log 1/\epsilon_L)$ |
| Gate overhead | $O(d)$ | $O(n^k)$ |
| T-gate overhead | $O(\log^{2.5} 1/\epsilon_T)$ | Same |

### Critical Equations

$$\boxed{N_{phys} = 2d^2 \cdot N_{logical} \quad \text{(surface code)}}$$

$$\boxed{d = O\left(\frac{\log(1/\epsilon_L)}{\log(p_{th}/p)}\right)}$$

$$\boxed{\tau_{logical} = d \cdot (\tau_{gate} + \tau_{meas}) + \tau_{decode}}$$

$$\boxed{G_T \sim 15^k \text{ raw T-states per distilled (k levels)}}$$

---

## Daily Checklist

- [ ] Computed qubit overhead for different codes
- [ ] Analyzed gate and T-gate overhead
- [ ] Calculated logical clock cycles
- [ ] Optimized space-time tradeoffs
- [ ] Compared hardware architectures
- [ ] Estimated resources for real algorithms

---

## Preview: Day 770

Tomorrow is **Week 110 Synthesis**:
- Comprehensive threshold review
- Integrated concept map
- Master formula compilation
- Synthesis problems
- Preparation for Week 111: Decoding Algorithms
