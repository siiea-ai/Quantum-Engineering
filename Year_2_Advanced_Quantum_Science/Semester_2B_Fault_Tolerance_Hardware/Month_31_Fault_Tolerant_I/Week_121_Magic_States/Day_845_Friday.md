# Day 845: Magic State Injection

## Week 121, Day 5 | Month 31: Fault-Tolerant QC I | Semester 2B: Fault Tolerance & Hardware

### Overview

Today we study **magic state injection**: the process of transferring a magic state from a separate preparation region into the encoded logical space where computation occurs. This is the critical interface between magic state factories (which produce distilled $|T\rangle$ states) and the fault-tolerant quantum computer. We focus on lattice surgery approaches for surface codes, error propagation analysis, and practical implementation considerations.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Injection protocols and lattice surgery |
| **Afternoon** | 2.5 hours | Error analysis and fault tolerance |
| **Evening** | 1.5 hours | Computational lab: Injection simulation |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain the injection process** from physical magic states to logical qubits
2. **Construct lattice surgery protocols** for magic state injection
3. **Analyze error propagation** during injection
4. **Calculate resource requirements** for injection operations
5. **Compare different injection methods** (direct vs. lattice surgery)
6. **Design injection schedules** for fault-tolerant algorithms

---

## Part 1: The Injection Problem

### Context: Magic State Factory to Computation

```
┌─────────────────────────────────────────────────────────────┐
│                    QUANTUM COMPUTER                          │
│                                                              │
│  ┌──────────────┐      INJECTION      ┌──────────────┐      │
│  │  MAGIC STATE │  ──────────────────→│   LOGICAL    │      │
│  │   FACTORY    │                     │  COMPUTATION │      │
│  │              │                     │              │      │
│  │ • Prepare    │                     │ • Surface    │      │
│  │ • Distill    │                     │   code       │      │
│  │ • Store      │                     │ • Clifford   │      │
│  │              │                     │   gates +    │      │
│  │   |T⟩_phys   │        →            │   T-gates    │      │
│  └──────────────┘                     └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### The Challenge

**We have:** A distilled magic state $|T\rangle$ at the physical level (or in a small code)

**We need:** The magic state encoded in the logical space of the main computation

**Constraints:**
1. Must preserve the quantum information in $|T\rangle$
2. Must be compatible with surface code / lattice surgery
3. Must not introduce more errors than distillation removed
4. Must be time-efficient (T-gates often on critical path)

### Two Approaches

**1. Direct Encoding:**
- Encode $|T\rangle$ directly into surface code
- Simple but introduces errors from encoding circuit

**2. Lattice Surgery Injection:**
- Use measurement-based operations
- More complex but naturally fault-tolerant
- Standard approach for surface codes

---

## Part 2: Direct Injection

### Basic Protocol

**Step 1:** Prepare $|T\rangle$ on a physical qubit (or small code)

**Step 2:** Initialize a logical $|0\rangle_L$ in the surface code

**Step 3:** Perform logical CNOT from magic state to surface code

**Step 4:** Measure magic state, apply corrections

### Circuit Representation

```
Physical |T⟩ ─────●───── M_X ──→ m
                  │
Logical |0⟩_L ────X───────────→ |T⟩_L (after correction)
```

### Error Analysis

**Problem:** The CNOT between physical and logical qubits is NOT transversal!

If we use a non-FT CNOT:
- Errors on the physical qubit can spread to multiple data qubits
- A single fault can cause a logical error

**Solution:** Use lattice surgery (no direct transversal gate needed)

---

## Part 3: Lattice Surgery Injection

### Overview

Lattice surgery performs logical operations through:
1. **Merging** code patches (measuring joint stabilizers)
2. **Splitting** code patches (measuring individual stabilizers)

For magic state injection, we use a **merge + split** sequence.

### The Protocol

**Setup:**
- Logical data qubit $|\psi\rangle_L$ in surface code patch A
- Magic state $|T\rangle$ in small surface code patch B (or physical qubit)

**Step 1: Rough Merge (ZZ measurement)**

Merge patches A and B along a rough boundary, measuring $\bar{Z}_A \bar{Z}_B$:

```
Before merge:
┌─────────┐    ┌───┐
│    A    │    │ B │
│  |ψ⟩_L  │    │|T⟩│
└─────────┘    └───┘

During merge:
┌─────────────────┐
│    A      B     │
│  |ψ⟩_L ⊗ |T⟩   │
│                 │
│  Measure Z̄_A Z̄_B│
└─────────────────┘
```

This implements a controlled-Z type operation between A and B.

**Step 2: Measure B in X-basis**

Measure $\bar{X}_B$ (logical X on patch B):

```
Measure X̄_B → outcome m ∈ {0, 1}
```

**Step 3: Apply Correction**

Based on measurement outcomes, apply Pauli corrections to A.

**Result:** Logical T-gate applied to $|\psi\rangle_L$

### Why This Works

The merge-measure sequence implements gate teleportation at the logical level:

1. **ZZ measurement** creates entanglement between A and B
2. **X measurement** on B teleports the T-gate to A
3. **Corrections** (from ZZ and X outcomes) are Pauli operations

This is exactly gate teleportation, but using lattice surgery instead of CNOT!

---

## Part 4: Detailed Lattice Surgery Analysis

### Merge Operation

When we merge two surface code patches, we:
1. Add ancilla qubits along the merge boundary
2. Measure new stabilizers connecting the patches
3. The product of boundary stabilizers gives $\bar{Z}_A \bar{Z}_B$ (or $\bar{X}_A \bar{X}_B$ for smooth merge)

### Timing

| Operation | Duration | Notes |
|-----------|----------|-------|
| Rough merge | $d$ code cycles | Measure d-times for reliability |
| X measurement on B | $d$ code cycles | Destructive measurement |
| Correction | $O(1)$ cycles | Tracked in Pauli frame |

**Total injection time:** $\sim 2d$ code cycles

### Stabilizer Evolution

**Before merge:**
- Patch A: Stabilizers $\{A_v, B_p\}$ (vertices and plaquettes)
- Patch B: Stabilizers $\{A'_v, B'_p\}$

**After merge:**
- Combined stabilizers plus boundary measurements
- Logical $\bar{Z}_A \bar{Z}_B$ becomes a stabilizer (up to measurement outcome)

### Error Propagation

**Key Question:** How do errors on the magic state affect the output?

**Analysis:**
- If $|T\rangle$ has error $\epsilon$: $\rho_T = (1-\epsilon)|T\rangle\langle T| + \epsilon \cdot \text{noise}$
- After injection, logical qubit has error $\sim \epsilon$
- No error amplification from injection itself (if done correctly)

**Fault-Tolerance Requirement:**
- Distillation must reduce $\epsilon$ below logical error rate of surface code
- Typically need $\epsilon < p_L(d) \sim (p/p_{th})^{(d+1)/2}$

---

## Part 5: Error Analysis

### Sources of Error in Injection

| Error Source | Magnitude | Mitigation |
|--------------|-----------|------------|
| Magic state infidelity | $\epsilon_{magic}$ | Distillation |
| Merge operation errors | $O(p^{d/2})$ | Large code distance |
| Measurement errors | $O(p)$ | Repeated measurement |
| Timing/synchronization | Small | Careful scheduling |

### Error Budget

For a fault-tolerant T-gate, total error should be:

$$\epsilon_{T-gate} = \epsilon_{magic} + \epsilon_{injection} + \epsilon_{correction}$$

**Design constraint:**
$$\epsilon_{T-gate} < \epsilon_{target}$$

where $\epsilon_{target}$ is set by algorithm requirements.

### Example Calculation

**Given:**
- Physical error rate: $p = 0.1\%$
- Surface code distance: $d = 17$
- Threshold: $p_{th} = 1\%$

**Logical error rate:**
$$p_L \approx 0.1 \cdot (0.001/0.01)^{(17+1)/2} = 0.1 \cdot (0.1)^9 = 10^{-10}$$

**Magic state requirement:**
For injection to not dominate, need:
$$\epsilon_{magic} < 10^{-10}$$

This requires multiple rounds of distillation!

---

## Part 6: Practical Considerations

### Magic State Factories

In practice, magic states are produced in dedicated "factories":

```
                    MAGIC STATE FACTORY
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  ┌─────┐   ┌──────────┐   ┌─────┐         │
    │  │ Raw │ → │Distill L1│ → │Store│         │
    │  │Prep │   └──────────┘   └─────┘         │
    │  └─────┘                     │             │
    │     ↓                        ↓             │
    │  ┌─────┐   ┌──────────┐   ┌─────┐         │
    │  │ Raw │ → │Distill L1│ → │Store│──→ To   │
    │  │Prep │   └──────────┘   └─────┘   Main  │
    │  └─────┘                     │      Code  │
    │     ⋮          ⋮             ↓             │
    │              ┌──────────┐                  │
    │              │Distill L2│──→ High-fidelity │
    │              └──────────┘    |T⟩          │
    │                                             │
    └─────────────────────────────────────────────┘
```

### Factory Design Parameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Level-1 distillation | 15:1 | 15 input, 1 output |
| Level-2 distillation | 15:1 | Further purification |
| Factory footprint | $\sim 10d^2$ qubits | Per distillation block |
| Production rate | 1 per $\sim 10d$ cycles | After pipelining |

### Scheduling

**T-gate hungry algorithms:**
- Many algorithms are limited by T-gate rate
- Factory must produce magic states faster than consumption
- Scheduling is a critical optimization problem

**Example: 1000 T-gates with $d=17$ code:**
- Each T-gate: $\sim 2d = 34$ cycles for injection
- Plus distillation overhead
- Total time dominated by magic state supply

### Space-Time Tradeoff

$$\text{Space} \times \text{Time} = \text{Space-Time Volume}$$

**Option 1:** Large factory, fast production, more qubits
**Option 2:** Small factory, slow production, fewer qubits

Optimal choice depends on:
- Available physical qubits
- Algorithm T-count
- Acceptable runtime

---

## Part 7: Advanced Injection Techniques

### Parallel Injection

For algorithms with many T-gates, can inject multiple magic states simultaneously:

```
Factory 1 → Inject to Qubit 1
Factory 2 → Inject to Qubit 2
Factory 3 → Inject to Qubit 3
     ⋮            ⋮
```

### Pipelined Distillation

Distillation can be pipelined to hide latency:

```
Time:    1    2    3    4    5    6    7
Factory: Prep Dist Prep Dist Prep Dist ...
Output:            |T⟩      |T⟩      |T⟩
```

### Adaptive Injection

Some protocols adapt injection based on syndrome measurements:
- If errors detected during injection, discard and retry
- Trades time for reliability

---

## Worked Examples

### Example 1: Calculate Injection Time

**Problem:** How long does magic state injection take for a distance-17 surface code with 1 μs code cycle?

**Solution:**

**Injection steps:**
1. Rough merge: $d = 17$ cycles
2. X measurement: $d = 17$ cycles (for reliable measurement)
3. Correction: Tracked in Pauli frame (0 cycles)

**Total:**
$$t_{inject} = 2d \times t_{cycle} = 2 \times 17 \times 1\mu s = 34 \mu s$$

$$\boxed{t_{inject} = 34 \mu s}$$

---

### Example 2: Factory Size Estimation

**Problem:** How many physical qubits are needed for a magic state factory that produces one $|T\rangle$ per 100 code cycles?

**Solution:**

**Single 15-to-1 distillation block:**
- 15 input magic states (small codes)
- 1 output magic state (larger code)
- Processing time: $\sim 10d$ cycles

**For $d = 17$:**
- Block processes in $10 \times 17 = 170$ cycles
- To produce 1 output per 100 cycles, need $\sim 2$ blocks in parallel

**Qubits per block:**
- 15 small codes ($d_{small} \sim 5$): $15 \times 2(5)^2 \approx 750$ qubits
- 1 output code ($d = 17$): $2(17)^2 \approx 580$ qubits
- Overhead: $\sim 50\%$

**Total:** $\sim 2 \times (750 + 580) \times 1.5 \approx 4000$ qubits

$$\boxed{N_{factory} \approx 4000 \text{ physical qubits}}$$

---

### Example 3: Error Budget Allocation

**Problem:** An algorithm requires $10^6$ T-gates with target total error $< 1\%$. What magic state fidelity is needed?

**Solution:**

**Per-T-gate error budget:**
$$\epsilon_{T} < \frac{0.01}{10^6} = 10^{-8}$$

**Error components:**
- $\epsilon_{magic}$: Magic state infidelity
- $\epsilon_{inject}$: Injection error
- $\epsilon_{correction}$: Correction error (negligible)

**If injection has error $10^{-10}$ (from surface code):**
$$\epsilon_{magic} < 10^{-8} - 10^{-10} \approx 10^{-8}$$

**Distillation levels needed:**
- Level 1: $\epsilon_{out} \approx 35\epsilon_{in}^3$
- Starting from $\epsilon_{in} = 1\%$: $\epsilon_1 \approx 35 \times 10^{-6} \approx 3.5 \times 10^{-5}$
- Level 2: $\epsilon_2 \approx 35 \times (3.5 \times 10^{-5})^3 \approx 1.5 \times 10^{-12}$

$$\boxed{\text{2 levels of distillation suffice}}$$

---

## Practice Problems

### Problem Set A: Direct Application

**A1.** Calculate the injection time for $d = 9$ surface code with 500 ns code cycle.

**A2.** If a factory produces magic states at rate 1 per 50 cycles, and an algorithm needs 1000 T-gates, what is the minimum runtime (in cycles)?

**A3.** What is the error rate of a magic state after one round of 15-to-1 distillation starting from $\epsilon_{in} = 0.5\%$?

### Problem Set B: Intermediate

**B1.** Design a lattice surgery sequence for injecting a magic state when the data qubit is in the middle of a 2D qubit array.

**B2.** Compare the space-time volume of (a) one large factory vs (b) two small factories for producing magic states at the same rate.

**B3.** If injection error is $p_{inject} = cp^{d/2}$ for constant $c$, at what code distance does injection error match magic state error of $10^{-8}$?

### Problem Set C: Challenging

**C1.** Design an adaptive injection protocol that discards faulty injections. What is the overhead?

**C2.** Prove that lattice surgery injection is fault-tolerant: a single fault during injection causes at most one error in the output.

**C3.** **(Research-level)** Optimize the factory layout for a specific algorithm with known T-gate pattern. Consider qubit routing constraints.

---

## Computational Lab

```python
"""
Day 845 Computational Lab: Magic State Injection Analysis
Analyzes injection protocols, error propagation, and resource requirements
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# =============================================================================
# Part 1: Injection Time Calculator
# =============================================================================

def injection_time(d: int, cycle_time_us: float = 1.0,
                   merge_cycles: int = None, measure_cycles: int = None) -> float:
    """
    Calculate magic state injection time.

    Args:
        d: Surface code distance
        cycle_time_us: Time per code cycle in microseconds
        merge_cycles: Override for merge duration (default: d)
        measure_cycles: Override for measurement duration (default: d)

    Returns:
        Total injection time in microseconds
    """
    if merge_cycles is None:
        merge_cycles = d
    if measure_cycles is None:
        measure_cycles = d

    total_cycles = merge_cycles + measure_cycles
    return total_cycles * cycle_time_us

print("=" * 60)
print("MAGIC STATE INJECTION ANALYSIS")
print("=" * 60)

# Calculate for various distances
print("\nInjection Times (1 μs cycle time):")
print("-" * 40)
for d in [5, 9, 13, 17, 21, 25]:
    t = injection_time(d)
    print(f"  d = {d:2d}: {t:6.1f} μs ({int(2*d)} cycles)")

# =============================================================================
# Part 2: Error Budget Analysis
# =============================================================================

def distillation_error(epsilon_in: float, protocol: str = '15-to-1') -> float:
    """Calculate output error of distillation protocol."""
    if protocol == '15-to-1':
        # Standard 15-to-1 protocol: epsilon_out ≈ 35 * epsilon_in^3
        return 35 * epsilon_in**3
    elif protocol == '7-to-1':
        # Alternative: epsilon_out ≈ 7 * epsilon_in^2
        return 7 * epsilon_in**2
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

def logical_error_rate(p_phys: float, d: int, p_th: float = 0.01) -> float:
    """Calculate logical error rate for surface code."""
    if p_phys >= p_th:
        return 1.0  # Above threshold
    lambda_val = p_th / p_phys
    return 0.1 * (1/lambda_val)**((d+1)/2)

print("\n" + "=" * 60)
print("ERROR BUDGET ANALYSIS")
print("=" * 60)

# Distillation cascade
print("\nDistillation Cascade (starting from 1% error):")
epsilon = 0.01
for level in range(1, 5):
    epsilon = distillation_error(epsilon)
    print(f"  Level {level}: ε = {epsilon:.2e}")

# Compare magic state error to surface code error
print("\nMagic State vs Surface Code Errors:")
print("-" * 50)
print(f"{'Distance':<10} {'p_L (p=0.1%)':<20} {'Distill needed':<15}")

for d in [9, 13, 17, 21, 25]:
    p_L = logical_error_rate(0.001, d)

    # How many distillation levels to match p_L?
    eps = 0.01
    levels = 0
    while eps > p_L and levels < 10:
        eps = distillation_error(eps)
        levels += 1

    print(f"  d={d:<5} {p_L:<20.2e} {levels} levels")

# =============================================================================
# Part 3: Factory Resource Estimation
# =============================================================================

def factory_qubits(d_target: int, d_input: int = 5,
                   n_parallel: int = 1) -> Dict[str, int]:
    """
    Estimate qubits needed for magic state factory.

    Args:
        d_target: Target code distance for output
        d_input: Distance of input magic state codes
        n_parallel: Number of parallel distillation units

    Returns:
        Dictionary with qubit counts
    """
    # Qubits per small (input) surface code
    qubits_small = 2 * d_input**2 - 1

    # Qubits per large (output) surface code
    qubits_large = 2 * d_target**2 - 1

    # 15-to-1 distillation block
    qubits_per_block = 15 * qubits_small + qubits_large

    # Overhead for routing and ancillas (~50%)
    overhead_factor = 1.5

    return {
        'per_block': int(qubits_per_block * overhead_factor),
        'total': int(n_parallel * qubits_per_block * overhead_factor),
        'input_codes': 15 * n_parallel,
        'output_codes': n_parallel,
    }

print("\n" + "=" * 60)
print("FACTORY RESOURCE ESTIMATION")
print("=" * 60)

for d in [9, 13, 17, 21]:
    resources = factory_qubits(d)
    print(f"\nTarget distance d = {d}:")
    print(f"  Qubits per block: {resources['per_block']}")
    print(f"  Input codes: {resources['input_codes']}")
    print(f"  Output codes: {resources['output_codes']}")

# =============================================================================
# Part 4: Algorithm Resource Planning
# =============================================================================

def algorithm_resources(t_count: int, d: int, cycle_time_us: float = 1.0,
                        factory_rate: float = 0.01) -> Dict[str, float]:
    """
    Calculate resources for running an algorithm.

    Args:
        t_count: Number of T-gates in algorithm
        d: Surface code distance
        cycle_time_us: Time per code cycle
        factory_rate: Magic states produced per cycle

    Returns:
        Resource estimates
    """
    # Injection time per T-gate
    t_inject = injection_time(d, cycle_time_us)

    # Time if T-gates are sequential (worst case)
    time_sequential = t_count * t_inject / 1e6  # Convert to seconds

    # Time if limited by factory rate
    cycles_per_magic = 1 / factory_rate
    time_factory_limited = t_count * cycles_per_magic * cycle_time_us / 1e6

    # Practical estimate (whichever is larger)
    time_practical = max(time_sequential, time_factory_limited)

    return {
        't_count': t_count,
        'injection_time_us': t_inject,
        'time_sequential_s': time_sequential,
        'time_factory_s': time_factory_limited,
        'time_practical_s': time_practical,
        'factory_rate': factory_rate,
    }

print("\n" + "=" * 60)
print("ALGORITHM RESOURCE PLANNING")
print("=" * 60)

algorithms = [
    ('Small VQE', 1000),
    ('RSA-2048', 10**9),
    ('Quantum chemistry', 10**12),
]

for name, t_count in algorithms:
    print(f"\n{name} (T-count = {t_count:.0e}):")
    for d in [17, 21, 25]:
        res = algorithm_resources(t_count, d)
        print(f"  d={d}: {res['time_practical_s']:.2e} seconds "
              f"({res['time_practical_s']/3600:.2f} hours)")

# =============================================================================
# Part 5: Visualization
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Injection time vs distance
ax1 = axes[0, 0]
distances = np.arange(5, 31, 2)
times = [injection_time(d) for d in distances]

ax1.plot(distances, times, 'bo-', markersize=6, linewidth=2)
ax1.set_xlabel('Code Distance d', fontsize=10)
ax1.set_ylabel('Injection Time (μs)', fontsize=10)
ax1.set_title('Magic State Injection Time', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Distillation cascade
ax2 = axes[0, 1]
levels = range(0, 6)
epsilon_15to1 = [0.01]  # Starting error
epsilon_7to1 = [0.01]

for _ in range(5):
    epsilon_15to1.append(distillation_error(epsilon_15to1[-1], '15-to-1'))
    epsilon_7to1.append(distillation_error(epsilon_7to1[-1], '7-to-1'))

ax2.semilogy(levels, epsilon_15to1, 'b-o', markersize=8, linewidth=2, label='15-to-1')
ax2.semilogy(levels, epsilon_7to1, 'r-s', markersize=8, linewidth=2, label='7-to-1')
ax2.set_xlabel('Distillation Level', fontsize=10)
ax2.set_ylabel('Output Error ε', fontsize=10)
ax2.set_title('Distillation Cascade', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Factory qubits vs target fidelity
ax3 = axes[0, 2]

target_errors = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
factory_sizes = []

for target in target_errors:
    # Determine distillation levels needed
    eps = 0.01
    levels = 0
    while eps > target and levels < 5:
        eps = distillation_error(eps)
        levels += 1

    # Factory grows with levels
    size = factory_qubits(17)['per_block'] * (1 + 0.5 * levels)  # Rough scaling
    factory_sizes.append(size)

ax3.semilogx(target_errors, factory_sizes, 'go-', markersize=8, linewidth=2)
ax3.set_xlabel('Target Error Rate', fontsize=10)
ax3.set_ylabel('Factory Size (qubits)', fontsize=10)
ax3.set_title('Factory Size vs Target Fidelity', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()

# Plot 4: Error comparison
ax4 = axes[1, 0]

distances_plot = np.arange(5, 31, 2)
p_phys = 0.001  # 0.1%

logical_errors = [logical_error_rate(p_phys, d) for d in distances_plot]

# Magic state errors after 1 and 2 distillation levels
magic_L1 = distillation_error(0.01)
magic_L2 = distillation_error(magic_L1)

ax4.semilogy(distances_plot, logical_errors, 'b-', linewidth=2, label='Surface code p_L')
ax4.axhline(y=magic_L1, color='orange', linestyle='--', linewidth=2, label='Magic (1 dist.)')
ax4.axhline(y=magic_L2, color='green', linestyle='--', linewidth=2, label='Magic (2 dist.)')
ax4.set_xlabel('Code Distance d', fontsize=10)
ax4.set_ylabel('Error Rate', fontsize=10)
ax4.set_title('Surface Code vs Magic State Errors\n(p = 0.1%)', fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Factory-limited vs injection-limited runtime
ax5 = axes[1, 1]

t_counts = np.logspace(3, 12, 20)
d = 17

times_sequential = []
times_factory = []

for t_count in t_counts:
    res = algorithm_resources(int(t_count), d, factory_rate=0.01)
    times_sequential.append(res['time_sequential_s'])
    times_factory.append(res['time_factory_s'])

ax5.loglog(t_counts, times_sequential, 'b-', linewidth=2, label='Injection-limited')
ax5.loglog(t_counts, times_factory, 'r-', linewidth=2, label='Factory-limited')
ax5.set_xlabel('T-count', fontsize=10)
ax5.set_ylabel('Runtime (seconds)', fontsize=10)
ax5.set_title(f'Algorithm Runtime (d={d})', fontsize=11)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Annotate crossover
crossover_idx = np.argmin(np.abs(np.array(times_sequential) - np.array(times_factory)))
ax5.axvline(x=t_counts[crossover_idx], color='gray', linestyle=':', alpha=0.7)
ax5.text(t_counts[crossover_idx]*1.5, times_sequential[crossover_idx],
         'Crossover', fontsize=9)

# Plot 6: Injection protocol diagram
ax6 = axes[1, 2]
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)

# Draw patches
# Data patch
rect_data = plt.Rectangle((1, 5.5), 3, 3, facecolor='lightblue',
                            edgecolor='blue', linewidth=2)
ax6.add_patch(rect_data)
ax6.text(2.5, 7, 'Data\n|ψ⟩_L', ha='center', va='center', fontsize=10)

# Magic state patch
rect_magic = plt.Rectangle((6, 5.5), 1.5, 3, facecolor='lightcoral',
                             edgecolor='red', linewidth=2)
ax6.add_patch(rect_magic)
ax6.text(6.75, 7, '|T⟩', ha='center', va='center', fontsize=10)

# Merge region
rect_merge = plt.Rectangle((4, 5.5), 2, 3, facecolor='lightyellow',
                             edgecolor='orange', linewidth=2, linestyle='--')
ax6.add_patch(rect_merge)
ax6.text(5, 7, 'Merge\nRegion', ha='center', va='center', fontsize=9)

# Arrow showing process
ax6.annotate('', xy=(5, 4.5), xytext=(5, 5.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Result
rect_result = plt.Rectangle((2, 1), 4, 2.5, facecolor='lightgreen',
                              edgecolor='green', linewidth=2)
ax6.add_patch(rect_result)
ax6.text(4, 2.25, 'T|ψ⟩_L\n(after correction)', ha='center', va='center', fontsize=10)

# Labels
ax6.text(5, 9.5, 'Lattice Surgery Injection', ha='center', fontsize=12, fontweight='bold')
ax6.text(2.5, 4.8, '1. Merge', fontsize=9)
ax6.text(6.75, 4.8, '2. Measure', fontsize=9)
ax6.text(4, 0.5, '3. Correct', fontsize=9)

ax6.set_aspect('equal')
ax6.axis('off')

plt.tight_layout()
plt.savefig('day_845_injection_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Visualization saved to: day_845_injection_analysis.png")
print("=" * 60)

# =============================================================================
# Part 6: Summary
# =============================================================================

print("\n" + "=" * 60)
print("KEY RESULTS SUMMARY")
print("=" * 60)

summary = """
LATTICE SURGERY INJECTION:
  1. Merge data patch with magic state patch
  2. Measure magic state in X-basis
  3. Apply Pauli corrections based on outcomes
  Result: T-gate on data qubit

TIMING:
  Injection time ≈ 2d code cycles
  For d=17, 1μs cycle: 34 μs per T-gate

ERROR ANALYSIS:
  Total error = ε_magic + ε_injection + ε_correction
  Injection error ≈ O(p^{d/2}) for distance d
  Need distillation to match surface code error rate

FACTORY DESIGN:
  15-to-1 protocol: 15 input → 1 output magic state
  Factory size: ~4000 qubits for d=17 target
  Production rate: ~1 magic state per 10d cycles

ALGORITHM IMPACT:
  T-gates often dominate runtime
  Factory rate can be bottleneck for large T-counts
  Optimization: parallel factories, pipelining
"""
print(summary)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Injection time | $t_{inject} \approx 2d \times t_{cycle}$ |
| Distillation error (15-to-1) | $\epsilon_{out} \approx 35\epsilon_{in}^3$ |
| Logical error rate | $p_L \approx 0.1(p/p_{th})^{(d+1)/2}$ |
| Factory qubits (15-to-1) | $N \approx 15 \times 2d_{small}^2 + 2d_{large}^2$ |
| Production rate | $\sim 1/(10d)$ magic states per cycle |

### Main Takeaways

1. **Lattice surgery injection is fault-tolerant** - Errors during injection don't amplify; the process is naturally protected

2. **Injection time scales linearly with distance** - $\sim 2d$ code cycles for merge + measurement

3. **Magic state error must match surface code error** - Distillation levels determined by target logical error rate

4. **Factories are a significant resource** - Thousands of qubits dedicated to magic state production

5. **T-gate rate often limits algorithms** - Factory design is critical for practical fault-tolerant computation

---

## Daily Checklist

- [ ] Understand the injection problem and why it matters
- [ ] Can describe lattice surgery injection protocol
- [ ] Can calculate injection time for given parameters
- [ ] Understand error propagation during injection
- [ ] Can estimate factory resource requirements
- [ ] Know how to allocate error budget for T-gates
- [ ] Completed computational lab exercises

---

## Preview: Day 846

Tomorrow is our **Computational Lab** day! We will:

- Implement complete magic state preparation in Qiskit
- Simulate T-gate via gate teleportation
- Verify correctness with various input states
- Analyze error propagation experimentally
- Build intuition through hands-on coding

Get ready for practical implementation of everything we've learned this week!

---

*"Magic state injection is where the rubber meets the road - all the theory of distillation and teleportation must work together flawlessly."*

---

**Day 845 Complete** | **Next: Day 846 - Computational Lab**
