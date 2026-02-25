# Day 922: Error Correction Requirements

## Schedule Overview

| Time Block | Duration | Topic |
|------------|----------|-------|
| Morning | 3 hours | Error correction thresholds and codes |
| Afternoon | 2.5 hours | Platform-specific overhead analysis |
| Evening | 1.5 hours | Computational lab: Resource estimation |

## Learning Objectives

By the end of today, you will be able to:

1. Calculate fault-tolerance thresholds for different error models
2. Estimate physical qubit overhead for surface code implementation
3. Compare error correction requirements across platforms
4. Analyze magic state distillation overhead
5. Evaluate timeline to fault-tolerant operation
6. Design error correction strategies matched to hardware capabilities

## Core Content

### 1. Fault-Tolerance Threshold Fundamentals

#### Threshold Theorem

The threshold theorem guarantees that arbitrarily long quantum computation is possible if:

$$\boxed{p_{physical} < p_{threshold}}$$

For the surface code with standard depolarizing noise:

$$p_{threshold} \approx 1\%$$

The logical error rate scales as:

$$p_{logical} \approx A \left(\frac{p_{physical}}{p_{threshold}}\right)^{(d+1)/2}$$

where $d$ is the code distance and $A$ is a constant of order 1.

#### Code Distance Requirements

For target logical error rate $p_L$ with physical error rate $p$:

$$d \geq \frac{2\log(p_L/A)}{\log(p/p_{th})} - 1$$

**Example:** For $p_{phys} = 10^{-3}$, $p_{th} = 10^{-2}$, $p_L = 10^{-10}$:
$$d \geq \frac{2\log(10^{-10})}{\log(0.1)} - 1 = \frac{-20}{-1} - 1 = 19$$

### 2. Platform-Specific Error Models

#### Superconducting Qubits

**Dominant Error Types:**
1. Depolarizing noise (T1, T2 decay)
2. Measurement errors (~1-5%)
3. Leakage to non-computational states (~0.1-1%)
4. Crosstalk errors (~0.1-0.5%)

**Effective Error Model:**
$$\mathcal{E}(\rho) = (1-p_d)\rho + \frac{p_d}{3}(X\rho X + Y\rho Y + Z\rho Z) + p_L|\text{leak}\rangle\langle\text{leak}|$$

**Threshold Modifications:**
- Leakage reduces threshold: $p_{th}^{eff} \approx p_{th}(1 - 3p_L/p_d)$
- Correlated errors from crosstalk: additional threshold reduction

**Current Status:**
| Metric | Value | Threshold |
|--------|-------|-----------|
| 2Q gate error | 0.3-1% | 1% |
| Measurement error | 1-3% | ~1% |
| Leakage | 0.1-0.5% | <0.5% |

#### Trapped Ions

**Dominant Error Types:**
1. Coherent over-rotation (laser noise)
2. Motional heating
3. Measurement errors (~0.1-1%)
4. Dephasing (magnetic field fluctuations)

**Error Model:**
$$\mathcal{E}(\rho) = (1-p_z)U_\theta \rho U_\theta^\dagger + p_z Z\rho Z$$

where $U_\theta$ is a small coherent rotation.

**Threshold Advantages:**
- Lower measurement error improves syndrome extraction
- Coherent errors can be actively corrected
- All-to-all connectivity reduces SWAP overhead

**Current Status:**
| Metric | Value | Threshold |
|--------|-------|-----------|
| 2Q gate error | 0.1-0.5% | 1% |
| Measurement error | 0.1-0.5% | ~1% |
| T2 limitation | 1-10s | Minimal |

#### Neutral Atoms

**Dominant Error Types:**
1. Rydberg decay (~0.5-2% per gate)
2. Atom loss (~0.1% per operation)
3. Position-dependent errors
4. Measurement errors (atom loss during imaging)

**Error Model:**
$$\mathcal{E}(\rho) = (1-p_{loss})\mathcal{E}_{gate}(\rho) + p_{loss}|loss\rangle\langle loss|$$

**Threshold Considerations:**
- Atom loss is a "leakage" that requires replacement
- Reconfigurable connectivity can optimize code layout
- Mid-circuit measurement challenging

**Current Status:**
| Metric | Value | Threshold |
|--------|-------|-----------|
| 2Q gate error | 0.5-2% | 1% |
| Atom loss | 0.1-0.5% | <0.5% |
| Measurement | 1-3% | ~1% |

### 3. Surface Code Implementation

#### Physical Qubit Layout

For a distance-$d$ surface code:
- Data qubits: $d^2$
- Measure qubits: $d^2 - 1$ (alternating X and Z)
- Total: $2d^2 - 1 \approx 2d^2$

**Per Logical Qubit:**
$$\boxed{n_{phys/log} \approx 2d^2}$$

#### Syndrome Extraction Cycle

Each error correction round requires:

1. Initialize ancilla qubits
2. Apply CNOT gates (4 per ancilla)
3. Measure ancillas
4. Decode syndrome

**Cycle Time:**
$$t_{cycle} = t_{init} + 4t_{CNOT} + t_{measure} + t_{decode}$$

| Platform | t_cycle | Cycles per T2 |
|----------|---------|---------------|
| SC | ~1 μs | ~100 |
| TI | ~1 ms | ~1000 |
| NA | ~100 μs | ~5000 |

#### Connectivity Requirements

Surface code requires nearest-neighbor connectivity on a 2D lattice.

**Superconducting:** Native match to heavy-hex/square lattice
**Trapped Ion:** Requires SWAP network or zone-based implementation
**Neutral Atom:** Can be arranged to match (reconfigurable)

### 4. Overhead Estimation

#### Physical Qubit Overhead

For $n_L$ logical qubits with code distance $d$:

$$n_{physical} = n_L \cdot 2d^2 \cdot (1 + f_{routing} + f_{magic})$$

where:
- $f_{routing} \approx 0.1-0.5$ for logical qubit routing
- $f_{magic} \approx 1-10$ for magic state factories

#### Code Distance by Platform

Given current error rates:

| Platform | p_phys | d (for $p_L=10^{-10}$) | Qubits/logical |
|----------|--------|------------------------|----------------|
| SC | 0.3% | 17 | 578 |
| TI | 0.1% | 11 | 242 |
| NA | 0.5% | 21 | 882 |

#### Magic State Distillation

T gates require magic states, which must be distilled:

**15-to-1 Protocol:**
- Input: 15 noisy $|T\rangle$ states at error $p$
- Output: 1 state at error $35p^3$
- Overhead: 15× per level

**Rounds Needed:**
$$k = \lceil \log_{35}(p_{target}/p_{input}) \rceil$$

For $p_{input} = 0.1\%$, $p_{target} = 10^{-10}$:
$$k = \lceil \log_{35}(10^{-7}) \rceil = \lceil 4.5 \rceil = 5$$

**Total Overhead:**
$$n_{magic} = 15^k \approx 759,000$$ noisy states per output state

More efficient protocols (e.g., 20-to-4) reduce this significantly.

### 5. Timeline to Fault-Tolerance

#### Error Rate Improvement Trajectory

Historical improvement: ~2× per 2-3 years

**Superconducting:**
- 2020: ~1% two-qubit error
- 2024: ~0.3% two-qubit error
- 2027 (projected): ~0.1%
- 2030 (projected): ~0.03%

**Trapped Ion:**
- 2020: ~0.5% two-qubit error
- 2024: ~0.1% two-qubit error
- 2027 (projected): ~0.03%
- 2030 (projected): ~0.01%

#### Qubit Count Requirements

| Application | Logical Qubits | Physical (SC) | Physical (TI) | Physical (NA) |
|-------------|---------------|---------------|---------------|---------------|
| Demo (d=3) | 1 | 18 | 18 | 18 |
| Small algo | 10 | 5,000 | 2,000 | 8,000 |
| Chemistry | 100 | 50,000 | 20,000 | 80,000 |
| Cryptography | 4,000 | 2M | 1M | 3.5M |

#### Projected Milestones

| Milestone | SC | TI | NA |
|-----------|-----|-----|-----|
| Below threshold | 2023 ✓ | 2021 ✓ | 2024 |
| Logical qubit demo | 2023 ✓ | 2024 | 2025 |
| QEC breakeven | 2025 | 2026 | 2027 |
| Useful fault-tolerant | 2030 | 2032 | 2030 |

### 6. Platform-Specific QEC Strategies

#### Superconducting: Heavy-Hex Surface Code

Advantages:
- Native 2D connectivity
- Fast syndrome cycles (~1 μs)
- Established fabrication

Challenges:
- Leakage reduction required
- Measurement fidelity improvement
- Frequency collision management

**Optimal Strategy:**
- Surface code with distance 15-21
- Leakage reduction units every cycle
- Fast reset for ancilla reuse

#### Trapped Ion: Distributed Surface Code

Advantages:
- Low physical error rates
- High-fidelity measurement
- All-to-all logical connectivity

Challenges:
- Slow gates limit syndrome rate
- Zone management overhead
- Shuttling errors

**Optimal Strategy:**
- QCCD with surface code patches per zone
- Lattice surgery for logical operations
- Real-time adaptive decoding

#### Neutral Atom: Reconfigurable Codes

Advantages:
- Flexible connectivity
- Large qubit counts
- Parallel operations

Challenges:
- Atom loss (erasure errors)
- Mid-circuit measurement
- Gate fidelity improvement needed

**Optimal Strategy:**
- Exploit erasure conversion (atom loss → erasure)
- Reconfigure for optimal code layout
- Parallel syndrome extraction

### 7. Advanced Code Considerations

#### LDPC Codes

Low-Density Parity-Check codes offer better scaling:

$$n_{phys/log} \sim O(d)$$ vs. $O(d^2)$ for surface code

Challenges:
- Non-local connectivity required
- Higher threshold sensitivity
- Complex decoding

**Platform Fit:**
- TI: Natural for non-local codes (all-to-all)
- NA: Reconfigurable to match code graph
- SC: Challenging without long-range coupling

#### Erasure Conversion

Neutral atoms can detect atom loss as "erasure":
$$p_{erasure} + p_{error} = p_{total}$$

Erasure threshold is higher (~50%) than standard threshold:
$$p_{th}^{erasure} \gg p_{th}^{depolarizing}$$

**Advantage:** If 80% of errors are erasures, effective threshold increases ~3×.

## Quantum Computing Applications

### Error Budget Allocation

For a fault-tolerant algorithm:

$$\epsilon_{total} = \epsilon_{logical} \cdot N_{logical\_ops} + \epsilon_{magic} \cdot N_T$$

Optimize code distance and magic state fidelity to minimize physical qubits while meeting error budget.

### Runtime Estimation

Logical operation time:
$$t_{logical} = t_{cycle} \cdot d$$

For d=17, SC: $t_{logical} \approx 17$ μs
For d=11, TI: $t_{logical} \approx 11$ ms

**Algorithm Runtime:**
$$T_{total} = N_{logical\_ops} \cdot t_{logical} + N_T \cdot t_{magic}$$

## Worked Examples

### Example 1: Code Distance Calculation

**Problem:** A superconducting processor has 0.2% two-qubit error rate. Calculate the code distance needed for 1000 logical operations with total failure probability <1%.

**Solution:**

1. Per-operation logical error budget:
$$p_L = \frac{0.01}{1000} = 10^{-5}$$

2. Using threshold $p_{th} = 1\%$:
$$\frac{p_{phys}}{p_{th}} = \frac{0.002}{0.01} = 0.2$$

3. Code distance from:
$$p_L = A\left(\frac{p_{phys}}{p_{th}}\right)^{(d+1)/2}$$

Taking $A = 0.1$:
$$10^{-5} = 0.1 \times 0.2^{(d+1)/2}$$

$$(d+1)/2 = \frac{\log(10^{-4})}{\log(0.2)} = \frac{-4}{-0.699} = 5.72$$

$$d = 2 \times 5.72 - 1 = 10.4 \rightarrow d = 11$$

4. Physical qubits per logical:
$$n = 2 \times 11^2 = 242$$

**Answer:** Code distance d=11, requiring 242 physical qubits per logical qubit.

### Example 2: Magic State Factory Sizing

**Problem:** An algorithm requires $10^9$ T gates with logical error rate $10^{-12}$ per T gate. Design the magic state factory assuming input state fidelity 99.9%.

**Solution:**

1. Input error: $p_{in} = 0.001$
2. Target error: $p_{target} = 10^{-12}$

3. Using 15-to-1 distillation ($p_{out} = 35p_{in}^3$):

Level 1: $p_1 = 35 \times (0.001)^3 = 3.5 \times 10^{-8}$
Level 2: $p_2 = 35 \times (3.5 \times 10^{-8})^3 = 1.5 \times 10^{-21}$

Only 2 levels needed!

4. Factory size:
- Level 1: 15 noisy states → 1 intermediate
- Level 2: 15 intermediate → 1 output

Total input states per output: $15^2 = 225$

5. Production rate: $10^9$ T gates / algorithm time
For 1-second algorithm: $10^9$ states/s

6. Factory qubits (assuming distance-11 code):
Each magic state requires ~15 logical qubits during distillation
Factory: ~15 × 2 levels × 242 phys/log = ~7,000 physical qubits

**Answer:** 2-level 15-to-1 distillation, ~7,000 physical qubits per factory.

### Example 3: Platform Comparison for Small Algorithm

**Problem:** Compare the physical qubit requirements for a 10-logical-qubit algorithm on SC (0.3% error), TI (0.1% error), and NA (0.5% error) platforms, targeting $10^{-10}$ logical error rate.

**Solution:**

**Superconducting:**
$$d_{SC} = \left\lceil\frac{2\log(10^{-10}/0.1)}{\log(0.003/0.01)}\right\rceil = \left\lceil\frac{-18}{-0.52}\right\rceil = 35$$

$n_{SC} = 10 \times 2 \times 35^2 = 24,500$

**Trapped Ion:**
$$d_{TI} = \left\lceil\frac{-18}{\log(0.001/0.01)}\right\rceil = \left\lceil\frac{-18}{-1}\right\rceil = 18$$

$n_{TI} = 10 \times 2 \times 18^2 = 6,480$

**Neutral Atom:**
$$d_{NA} = \left\lceil\frac{-18}{\log(0.005/0.01)}\right\rceil = \left\lceil\frac{-18}{-0.30}\right\rceil = 60$$

$n_{NA} = 10 \times 2 \times 60^2 = 72,000$

**Answer:** TI: 6,480 qubits, SC: 24,500 qubits, NA: 72,000 qubits

## Practice Problems

### Level 1: Direct Application

1. Calculate the code distance needed for a logical error rate of $10^{-8}$ given physical error rate 0.5% and threshold 1%.

2. How many physical qubits are in a distance-15 surface code? What is the theoretical logical error rate if physical error is 0.2%?

3. A magic state factory produces states with error $10^{-10}$ from input states with error $10^{-3}$. How many levels of 15-to-1 distillation are needed?

### Level 2: Intermediate Analysis

4. Compare the syndrome cycle time for SC (50ns gates, 500ns measurement) and TI (200μs gates, 100μs measurement). How does this affect the logical clock speed?

5. An algorithm requires $10^6$ T gates. Calculate the total magic state factory overhead (qubits and time) for SC and TI platforms.

6. For neutral atoms with 1% total error but 80% erasure fraction, calculate the effective threshold advantage using erasure codes.

### Level 3: Advanced Research-Level

7. Derive the optimal code distance as a function of physical error rate and algorithm length, accounting for the trade-off between qubit overhead and logical error rate.

8. Design a fault-tolerant protocol for a 50-logical-qubit quantum chemistry simulation on a trapped ion QCCD architecture with 10 zones of 50 ions each.

9. Analyze the break-even point where error correction provides net benefit vs. unencoded qubits for each platform, considering both qubit overhead and time overhead.

## Computational Lab: Resource Estimation

```python
"""
Day 922 Computational Lab: Error Correction Resource Estimation
Comprehensive analysis of QEC requirements across platforms
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from dataclasses import dataclass

# Set plotting style
plt.rcParams['figure.figsize'] = (14, 12)
plt.rcParams['font.size'] = 11

# =============================================================================
# Part 1: Error Model and Threshold Analysis
# =============================================================================

@dataclass
class PlatformSpec:
    """Specification for a quantum computing platform"""
    name: str
    p_2q: float          # Two-qubit gate error
    p_1q: float          # Single-qubit gate error
    p_measure: float     # Measurement error
    p_idle: float        # Idling error per cycle
    t_2q: float          # Two-qubit gate time (s)
    t_measure: float     # Measurement time (s)
    connectivity: str    # 'nn' or 'all'

# Platform specifications (2024 state-of-art)
platforms = {
    'Superconducting': PlatformSpec(
        name='Superconducting',
        p_2q=0.003, p_1q=0.0003, p_measure=0.01, p_idle=1e-5,
        t_2q=50e-9, t_measure=500e-9, connectivity='nn'
    ),
    'Trapped Ion': PlatformSpec(
        name='Trapped Ion',
        p_2q=0.001, p_1q=0.0001, p_measure=0.003, p_idle=1e-6,
        t_2q=200e-6, t_measure=100e-6, connectivity='all'
    ),
    'Neutral Atom': PlatformSpec(
        name='Neutral Atom',
        p_2q=0.005, p_1q=0.002, p_measure=0.02, p_idle=1e-5,
        t_2q=1e-6, t_measure=10e-6, connectivity='nn'
    )
}

def effective_error_rate(spec: PlatformSpec) -> float:
    """Calculate effective error rate for surface code cycle"""
    # Surface code cycle: 4 CNOTs + measurement + reset
    # Simplified model: dominant contribution from 2Q gates
    p_eff = 4 * spec.p_2q + spec.p_measure + 4 * spec.p_1q
    return p_eff

def code_distance_required(p_phys: float, p_log_target: float,
                          p_threshold: float = 0.01, A: float = 0.1) -> int:
    """Calculate minimum code distance for target logical error"""
    if p_phys >= p_threshold:
        return float('inf')

    ratio = p_phys / p_threshold
    exponent = np.log(p_log_target / A) / np.log(ratio)
    d = 2 * exponent - 1

    return max(3, int(np.ceil(d)) | 1)  # Round up to odd number

def logical_error_rate(d: int, p_phys: float,
                      p_threshold: float = 0.01, A: float = 0.1) -> float:
    """Calculate logical error rate for given distance"""
    if p_phys >= p_threshold:
        return 1.0
    return A * (p_phys / p_threshold) ** ((d + 1) / 2)

# Calculate effective error rates
print("="*70)
print("EFFECTIVE ERROR RATES")
print("="*70)

for name, spec in platforms.items():
    p_eff = effective_error_rate(spec)
    print(f"{name}: p_eff = {p_eff*100:.3f}%")

# =============================================================================
# Part 2: Code Distance and Qubit Overhead
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Code distance vs target error rate
ax = axes[0, 0]
p_log_targets = np.logspace(-15, -3, 50)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for idx, (name, spec) in enumerate(platforms.items()):
    p_eff = effective_error_rate(spec)
    distances = [code_distance_required(p_eff, p_t) for p_t in p_log_targets]
    ax.semilogx(p_log_targets, distances, '-', label=name,
                color=colors[idx], linewidth=2)

ax.set_xlabel('Target Logical Error Rate')
ax.set_ylabel('Required Code Distance')
ax.set_title('Code Distance vs Target Error')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 50)

# Plot 2: Physical qubits per logical
ax = axes[0, 1]

n_log = 1  # Per logical qubit
for idx, (name, spec) in enumerate(platforms.items()):
    p_eff = effective_error_rate(spec)
    distances = [code_distance_required(p_eff, p_t) for p_t in p_log_targets]
    qubits = [2 * d**2 if d < 100 else np.nan for d in distances]
    ax.semilogx(p_log_targets, qubits, '-', label=name,
                color=colors[idx], linewidth=2)

ax.set_xlabel('Target Logical Error Rate')
ax.set_ylabel('Physical Qubits per Logical')
ax.set_title('Qubit Overhead vs Target Error')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 5000)

# Plot 3: Logical error rate vs physical error
ax = axes[0, 2]
p_phys_range = np.linspace(0.0001, 0.015, 100)

for d in [7, 11, 17, 21]:
    p_log = [logical_error_rate(d, p) for p in p_phys_range]
    ax.semilogy(p_phys_range * 100, p_log, '-', label=f'd={d}', linewidth=2)

# Mark platform positions
for idx, (name, spec) in enumerate(platforms.items()):
    p_eff = effective_error_rate(spec)
    ax.axvline(x=p_eff*100, color=colors[idx], linestyle='--', alpha=0.5,
              label=f'{name[:2]} ({p_eff*100:.2f}%)')

ax.set_xlabel('Physical Error Rate (%)')
ax.set_ylabel('Logical Error Rate')
ax.set_title('Logical vs Physical Error')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-15, 1)

# =============================================================================
# Part 3: Magic State Factory Analysis
# =============================================================================

def magic_state_distillation(p_in: float, protocol: str = '15-to-1') -> Tuple[float, int]:
    """
    Calculate output error and resource usage for distillation

    Returns: (p_out, num_input_states)
    """
    if protocol == '15-to-1':
        p_out = 35 * p_in ** 3
        n_in = 15
    elif protocol == '20-to-4':
        p_out = 35 * p_in ** 3  # Similar leading order
        n_in = 5  # Per output state
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return p_out, n_in

def magic_factory_overhead(p_in: float, p_target: float) -> Dict:
    """Calculate magic state factory requirements"""
    p_current = p_in
    levels = 0
    total_input_states = 1

    while p_current > p_target:
        p_current, n_in = magic_state_distillation(p_current)
        levels += 1
        total_input_states *= n_in

        if levels > 10:  # Safety limit
            break

    return {
        'levels': levels,
        'input_states_per_output': total_input_states,
        'output_error': p_current
    }

# Plot 4: Magic state distillation levels
ax = axes[1, 0]

p_in_range = np.logspace(-4, -2, 50)
p_targets = [1e-8, 1e-10, 1e-12]

for p_target in p_targets:
    levels = [magic_factory_overhead(p_in, p_target)['levels']
              for p_in in p_in_range]
    ax.semilogx(p_in_range, levels, '-', label=f'target={p_target:.0e}', linewidth=2)

ax.set_xlabel('Input Magic State Error')
ax.set_ylabel('Distillation Levels')
ax.set_title('Magic State Distillation Depth')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 6)

# =============================================================================
# Part 4: Total Resource Estimation
# =============================================================================

def total_resources(n_logical: int, n_T_gates: int,
                   spec: PlatformSpec, p_log_target: float = 1e-10) -> Dict:
    """
    Estimate total physical resources for an algorithm

    Parameters:
    -----------
    n_logical: Number of logical qubits
    n_T_gates: Number of T gates in algorithm
    spec: Platform specification
    p_log_target: Target logical error per operation

    Returns:
    --------
    Dictionary with resource estimates
    """
    p_eff = effective_error_rate(spec)

    # Code distance for data qubits
    d_data = code_distance_required(p_eff, p_log_target / n_logical)

    # Physical qubits for data
    n_data = n_logical * 2 * d_data**2

    # Magic state factory
    p_magic_in = spec.p_2q  # Approximate
    p_magic_target = p_log_target / n_T_gates
    factory = magic_factory_overhead(p_magic_in, p_magic_target)

    # Factory qubits (rough estimate: 15 logical qubits per level)
    d_factory = code_distance_required(p_eff, p_magic_target)
    n_factory = 15 * factory['levels'] * 2 * d_factory**2

    # Routing overhead (20% for lattice surgery)
    n_routing = int(0.2 * n_data)

    # Total
    n_total = n_data + n_factory + n_routing

    # Time estimates
    t_cycle = 4 * spec.t_2q + spec.t_measure
    t_logical_op = d_data * t_cycle

    return {
        'n_logical': n_logical,
        'code_distance': d_data,
        'n_data_qubits': n_data,
        'n_factory_qubits': n_factory,
        'n_routing_qubits': n_routing,
        'n_total': n_total,
        'distillation_levels': factory['levels'],
        't_cycle': t_cycle,
        't_logical_op': t_logical_op
    }

# Plot 5: Total qubits for various applications
ax = axes[1, 1]

applications = [
    ('Small demo', 10, 1e4),
    ('VQE (chemistry)', 50, 1e6),
    ('Full chemistry', 200, 1e9),
    ('Cryptography', 4000, 1e12)
]

x = np.arange(len(applications))
width = 0.25

for idx, (name, spec) in enumerate(platforms.items()):
    totals = []
    for app_name, n_log, n_T in applications:
        try:
            res = total_resources(n_log, int(n_T), spec)
            totals.append(res['n_total'])
        except:
            totals.append(np.nan)

    bars = ax.bar(x + idx*width, totals, width, label=name, color=colors[idx])

ax.set_xticks(x + width)
ax.set_xticklabels([app[0] for app in applications], rotation=15)
ax.set_ylabel('Total Physical Qubits')
ax.set_title('Resource Requirements by Application')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Time comparison
ax = axes[1, 2]

# Logical operation times
logical_ops = np.logspace(3, 9, 50)

for idx, (name, spec) in enumerate(platforms.items()):
    p_eff = effective_error_rate(spec)
    d = code_distance_required(p_eff, 1e-10)
    t_cycle = 4 * spec.t_2q + spec.t_measure
    t_op = d * t_cycle

    times = logical_ops * t_op

    ax.loglog(logical_ops, times, '-', label=f'{name} (d={d})',
              color=colors[idx], linewidth=2)

ax.axhline(y=1, color='gray', linestyle=':', label='1 second')
ax.axhline(y=3600, color='gray', linestyle='--', label='1 hour')

ax.set_xlabel('Number of Logical Operations')
ax.set_ylabel('Total Time (s)')
ax.set_title('Algorithm Runtime')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qec_resource_estimation.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 5: Detailed Resource Table
# =============================================================================

print("\n" + "="*80)
print("DETAILED RESOURCE ESTIMATION")
print("="*80)

target_apps = [
    ('QEC Demo', 1, 100, 1e-6),
    ('Small Algorithm', 10, 10000, 1e-8),
    ('Chemistry Simulation', 100, int(1e8), 1e-10),
    ('Shor (2048-bit)', 4000, int(1e12), 1e-12)
]

for app_name, n_log, n_T, p_target in target_apps:
    print(f"\n{app_name} ({n_log} logical qubits, {n_T:.0e} T gates):")
    print(f"{'Platform':<18} {'Code d':<8} {'Data Q':<12} {'Factory Q':<12} {'Total Q':<12} {'t_op':<10}")
    print("-" * 72)

    for name, spec in platforms.items():
        try:
            res = total_resources(n_log, n_T, spec, p_target)
            t_op_str = f"{res['t_logical_op']*1e6:.1f} μs" if res['t_logical_op'] < 1e-3 else f"{res['t_logical_op']*1e3:.1f} ms"
            print(f"{name:<18} {res['code_distance']:<8} {res['n_data_qubits']:<12,} "
                  f"{res['n_factory_qubits']:<12,} {res['n_total']:<12,} {t_op_str:<10}")
        except Exception as e:
            print(f"{name:<18} Error: {e}")

# =============================================================================
# Part 6: Break-even Analysis
# =============================================================================

print("\n" + "="*80)
print("BREAK-EVEN ANALYSIS")
print("="*80)

def breakeven_depth(spec: PlatformSpec) -> Tuple[int, int]:
    """
    Find circuit depth where QEC becomes beneficial

    Returns:
    --------
    (circuit_depth, code_distance) where F_encoded > F_unencoded
    """
    p_eff = effective_error_rate(spec)

    for depth in range(10, 10000, 10):
        # Unencoded fidelity
        F_unencoded = (1 - spec.p_2q) ** depth

        # Find minimum distance that beats unencoded
        for d in range(3, 51, 2):
            # Encoded circuit with overhead
            p_log = logical_error_rate(d, p_eff)
            encoded_depth = depth * d  # Overhead from code cycles
            F_encoded = (1 - p_log) ** depth

            # Account for overhead (encoded circuit is deeper)
            if F_encoded > F_unencoded:
                return depth, d

    return np.nan, np.nan

print("\nBreak-even point (where QEC becomes beneficial):")
print(f"{'Platform':<20} {'Depth':<10} {'Code Distance':<15}")
print("-" * 45)

for name, spec in platforms.items():
    depth, d = breakeven_depth(spec)
    print(f"{name:<20} {depth:<10} {d:<15}")

# =============================================================================
# Part 7: Summary Statistics
# =============================================================================

print("\n" + "="*80)
print("QEC REQUIREMENTS SUMMARY")
print("="*80)

print("""
Key Findings:

1. THRESHOLD STATUS:
   - Superconducting: Below threshold (0.3% << 1%), but marginal for some operations
   - Trapped Ion: Well below threshold (0.1% << 1%), best positioned
   - Neutral Atom: Near threshold (0.5% ~ 1%), needs improvement

2. QUBIT OVERHEAD (for chemistry-scale applications):
   - SC: ~100k-1M physical qubits
   - TI: ~50k-500k physical qubits
   - NA: ~200k-2M physical qubits

3. TIME OVERHEAD:
   - SC: Fastest logical operations (~10 μs)
   - TI: Slowest (~10 ms), but highest fidelity
   - NA: Intermediate (~100 μs)

4. MAGIC STATE OVERHEAD:
   - All platforms require 2-3 distillation levels for practical algorithms
   - Factory typically adds 50-100% to total qubit count

5. BREAK-EVEN POINT:
   - Circuit depth ~100-1000 where QEC provides net benefit
   - Earlier for TI due to lower error rates
   - Later for NA due to higher error rates

6. TIMELINE TO USEFUL FAULT-TOLERANCE:
   - QEC demonstrations: 2023-2025 (all platforms)
   - Error correction advantage: 2025-2027
   - Practical fault-tolerant computing: 2028-2032
""")
```

## Summary

### Error Correction Thresholds by Platform

| Platform | Effective Error | Below Threshold? | Margin |
|----------|-----------------|------------------|--------|
| Superconducting | 1.2% | Marginal | ~1× |
| Trapped Ion | 0.4% | Yes | 2.5× |
| Neutral Atom | 2.2% | No | - |

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Code distance | $$d \approx 2\log(p_L/A)/\log(p/p_{th}) - 1$$ |
| Logical error | $$p_L = A(p/p_{th})^{(d+1)/2}$$ |
| Physical qubits | $$n_{phys} = 2d^2$$ per logical |
| 15-to-1 distillation | $$p_{out} = 35p_{in}^3$$ |

### Resource Requirements Summary

| Application | Logical Qubits | SC Phys | TI Phys | NA Phys |
|-------------|---------------|---------|---------|---------|
| Demo | 1 | 100 | 50 | 200 |
| Chemistry | 100 | 100k | 50k | 200k |
| Cryptography | 4000 | 5M | 2M | 10M |

### Main Takeaways

1. **Trapped ions** have the best error rates and lowest QEC overhead
2. **Superconducting** systems are near threshold but have fastest logical clock
3. **Neutral atoms** need ~3× fidelity improvement to be competitive for QEC
4. **Magic state distillation** dominates resource overhead for T-gate heavy algorithms
5. **Timeline**: First useful fault-tolerant computation likely 2028-2032

## Daily Checklist

- [ ] I can calculate code distance requirements for target logical error rates
- [ ] I understand how physical error rates map to logical error rates
- [ ] I can estimate physical qubit overhead for surface code
- [ ] I understand magic state distillation overhead
- [ ] I can compare QEC requirements across platforms
- [ ] I know the timeline milestones for fault-tolerant computing

## Preview of Day 923

Tomorrow we analyze **NISQ vs Fault-Tolerant Roadmaps**, comparing near-term applications achievable without full error correction to the longer-term path toward fault-tolerant quantum computing.
