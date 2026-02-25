# Day 838: Scaling Roadmaps (1000+ Logical Qubits)

## Week 120, Day 5 | Month 30: Surface Codes | Semester 2A: Quantum Error Correction

### Overview

Today we examine the engineering and scientific roadmaps toward building quantum computers with 1000+ logical qubits—the scale needed for transformative applications in chemistry, optimization, and cryptography. We analyze resource estimates from leading research groups, company roadmaps, and the formidable engineering challenges that must be overcome to reach million-qubit systems.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Resource estimation for algorithms |
| **Afternoon** | 2.5 hours | Company roadmaps and engineering challenges |
| **Evening** | 1.5 hours | Computational lab: Scaling projections |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Estimate logical qubit requirements** - Calculate resources for target algorithms
2. **Analyze company roadmaps** - Evaluate timelines from Google, IBM, Microsoft, IonQ
3. **Identify scaling bottlenecks** - Understand cryogenic, control, and wiring limitations
4. **Project timeline scenarios** - Model optimistic and conservative scaling trajectories
5. **Assess economic viability** - Estimate costs for fault-tolerant systems
6. **Evaluate modular approaches** - Compare monolithic vs. distributed architectures

---

## Core Content

### 1. Resource Estimation Framework

#### 1.1 From Algorithm to Physical Qubits

The path from algorithm requirements to physical resources:

```
Algorithm Specification
        ↓
Logical Circuit (gates, depth)
        ↓
Fault-Tolerant Compilation
        ↓
Logical Qubit Count & Code Distance
        ↓
Physical Qubit Count
        ↓
Hardware Requirements
```

#### 1.2 Key Scaling Relations

For a surface code with distance $d$ and error suppression factor $\lambda$:

**Physical qubits per logical qubit:**
$$\boxed{n_{\text{phys/log}} = 2d^2 - 1 \approx 2d^2}$$

**Logical error rate:**
$$p_L(d) = p_0 \cdot \lambda^{-(d-d_0)/2}$$

**Distance for target error rate:**
$$\boxed{d = d_0 + 2\left\lceil \frac{\ln(p_0/p_{\text{target}})}{\ln(\lambda)} \right\rceil}$$

For Willow ($\lambda = 2.14$, $p_0 = 0.143\%$ at $d_0 = 7$):

| Target $p_L$ | Required $d$ | Physical/Logical |
|--------------|--------------|------------------|
| $10^{-4}$ | 13 | 337 |
| $10^{-6}$ | 19 | 721 |
| $10^{-8}$ | 25 | 1249 |
| $10^{-10}$ | 31 | 1921 |
| $10^{-12}$ | 37 | 2737 |

### 2. Algorithm Resource Estimates

#### 2.1 Shor's Algorithm for RSA-2048

Breaking 2048-bit RSA requires factoring a 2048-bit number.

**Logical resources (optimized implementations):**
- Logical qubits: ~4,000-6,000
- T gates: ~$10^{10}$ (dominated by modular exponentiation)
- Circuit depth: ~$10^9$ T gates (sequential)

**Target error rate:**
With $10^{10}$ operations, need $p_L < 10^{-12}$ per gate:
$$d_{\text{RSA}} \approx 37$$

**Physical qubit estimate:**
$$N_{\text{phys}} = 6,000 \times 2,737 \approx 16 \text{ million}$$

**With magic state distillation overhead (~10×):**
$$\boxed{N_{\text{RSA-2048}} \approx 100-200 \text{ million physical qubits}}$$

#### 2.2 Quantum Chemistry: FeMoCo

The FeMo cofactor (nitrogen fixation catalyst) is a benchmark for quantum advantage in chemistry.

**Problem size:** ~100 spin orbitals

**Logical resources:**
- Logical qubits: ~2,000-4,000
- Toffoli gates: ~$10^{12}$
- Runtime: ~days to weeks

**Physical estimate (with optimizations):**
$$\boxed{N_{\text{FeMoCo}} \approx 1-10 \text{ million physical qubits}}$$

#### 2.3 Quantum Simulation: Hubbard Model

Simulating a 100×100 Hubbard model at finite temperature:

**Logical resources:**
- Logical qubits: ~20,000
- Trotter steps: ~$10^6$
- T gates per step: ~$10^5$

**Physical estimate:**
$$\boxed{N_{\text{Hubbard}} \approx 20-50 \text{ million physical qubits}}$$

### 3. Company Roadmaps

#### 3.1 Google Quantum AI

**Current (2024):** 105 qubits (Willow), below-threshold demonstrated

**Roadmap:**
| Year | Target | Milestone |
|------|--------|-----------|
| 2025 | 300-500 qubits | Multiple logical qubits |
| 2026-2027 | 1,000+ qubits | Logical operations |
| 2028-2029 | 10,000 qubits | Early fault-tolerant |
| 2030+ | 100,000+ qubits | Useful QEC |
| 2033+ | 1,000,000 qubits | Cryptographic applications |

**Key strategy:** Improve λ through engineering, scale monolithically

#### 3.2 IBM Quantum

**Current (2024):** 1,121 qubits (Condor), 133 qubits (Heron with error mitigation)

**Roadmap:**
| Year | Processor | Qubits | Focus |
|------|-----------|--------|-------|
| 2024 | Heron | 133 | Quality over quantity |
| 2025 | Crossbill | 408 | Multi-chip integration |
| 2026 | Flamingo | 1,000+ | Error correction |
| 2029 | Starling | 10,000+ | Modular scaling |
| 2033 | Kookaburra | 100,000+ | Fault-tolerant |

**Key strategy:** Modular multi-chip architecture with classical interconnects

#### 3.3 Microsoft Azure Quantum

**Approach:** Topological qubits (Majorana-based) + partnerships

**Roadmap:**
| Phase | Target | Timeline |
|-------|--------|----------|
| Milestone 1 | Demonstrate topological qubit | 2024-2025 |
| Milestone 2 | Multiple topological qubits | 2026-2027 |
| Milestone 3 | Topological error correction | 2028-2030 |
| Milestone 4 | Useful computation | 2030+ |

**Advantage if successful:** Near-zero physical error rate → fewer qubits needed

#### 3.4 IonQ and Trapped Ions

**Current (2024):** 36 algorithmic qubits (#AQ), 99.8% 2Q fidelity

**Roadmap:**
| Year | Target | Strategy |
|------|--------|----------|
| 2025 | 64 AQ | Improved gates |
| 2026-2027 | 256 AQ | Photonic networking |
| 2028-2029 | 1,024 AQ | Modular architecture |
| 2030+ | 4,096+ AQ | Distributed QEC |

**Key strategy:** Modular ion traps connected via photonic links

#### 3.5 Neutral Atom Companies

**QuEra:**
- 2024: 256 atoms, Rydberg gates
- 2026: 1,000+ atoms, error correction demos
- 2028: 10,000+ atoms, logical operations
- 2030+: 100,000+ atoms

**Atom Computing:**
- 2024: 1,000+ atom array demonstrated
- Focus: Nuclear spin qubits for long coherence

### 4. Engineering Challenges

#### 4.1 Cryogenic Scaling

**Current dilution refrigerators:**
- Cooling power at 20 mK: ~10-100 μW
- Space for ~100-1,000 qubits

**Heat load from control signals:**
- Each control line: ~10 nW at 20 mK
- 1 million qubits × 2 lines/qubit = 2 million lines
- Total heat: ~20 mW (exceeds cooling capacity by 100×!)

**Solutions being developed:**
1. Cryogenic CMOS control electronics
2. Multiplexed readout (1 line per 10-100 qubits)
3. Optical interconnects
4. Higher-temperature operation (4K for control)

#### 4.2 Control Electronics

**Current approach:** Room temperature electronics per qubit

**Scaling challenge:**
$$\boxed{\text{Wires} \propto N_{\text{qubits}} \rightarrow \text{unsustainable}}$$

**Solutions:**
1. **Cryo-CMOS:** Control chips at 4K (Intel, Google developing)
2. **Multiplexing:** Frequency-division control
3. **Photonic control:** Optical signals into cryostat

**Target:** <1 wire per qubit average

#### 4.3 Wiring and Packaging

**Wire count estimate:**
- DC bias: 1 per qubit
- Microwave drive: 1 per qubit
- Readout: 1 per ~10 qubits (multiplexed)
- Flux: 1 per coupler

For 1 million qubits: ~2-3 million coaxial cables (impossible!)

**3D integration approaches:**
- Multi-layer PCB at each temperature stage
- Flip-chip bonding of qubit and interposer
- Superconducting through-silicon vias (TSVs)

#### 4.4 Fabrication Yield

**Current yields:** ~90-95% working qubits

**For 1 million qubits at 95% yield:**
$$P(\text{all working}) = 0.95^{1,000,000} \approx 0$$

**Defect tolerance strategies:**
1. Redundant qubit placement
2. Reconfigurable routing
3. Defect-aware compilation
4. Born-bad qubit deactivation

### 5. Timeline Projections

#### 5.1 Physical Qubit Scaling

Historical exponential growth (Moore's Law analog):

$$\boxed{N(t) = N_0 \cdot 2^{(t-t_0)/\tau}}$$

| Optimistic | Conservative |
|------------|--------------|
| $\tau$ = 1 year | $\tau$ = 2 years |
| 1M qubits by 2030 | 1M qubits by 2035 |
| 100M by 2035 | 100M by 2045 |

#### 5.2 Logical Qubit Projections

More relevant metric: useful logical qubits

| Year | Logical Qubits (optimistic) | Applications Enabled |
|------|----------------------------|---------------------|
| 2025 | 1-10 | Demonstrations |
| 2027 | 10-100 | Small chemistry |
| 2030 | 100-1,000 | Materials science |
| 2033 | 1,000-10,000 | Drug discovery |
| 2035 | 10,000+ | Cryptography |

### 6. Economic Considerations

#### 6.1 Cost Scaling

**Current system costs:**
- 100-qubit system: ~$10-50 million
- Dilution refrigerator: ~$1-5 million
- Control electronics: ~$100K per qubit
- Facilities: ~$10-50 million

**Projected costs at scale:**
$$C(N) \approx C_0 + c \cdot N^\alpha$$

where $\alpha < 1$ (economies of scale) optimistically, $\alpha \approx 1$ pessimistically.

For 1 million qubits:
- Optimistic: ~$1-10 billion
- Conservative: ~$10-100 billion

#### 6.2 Operating Costs

**Power consumption:**
- Dilution refrigerator: ~50-100 kW
- Control electronics: ~100 W per qubit
- Classical processing: ~1-10 MW

**For million-qubit system:**
$$P_{\text{total}} \approx 100 \text{ MW}$$

**Annual operating cost:** ~$50-100 million (power alone)

---

## Worked Examples

### Example 1: Resource Estimation for Drug Discovery

**Problem:** A quantum simulation of a drug binding site requires 500 logical qubits with $10^8$ T gates. The target success probability is 99%. Estimate physical qubit requirements assuming λ = 2.5.

**Solution:**

**Target logical error rate:**
$$p_L^{\text{total}} < 1 - 0.99 = 0.01$$

Per-gate error rate:
$$p_L^{\text{gate}} < \frac{0.01}{10^8} = 10^{-10}$$

**Required code distance:**
Using $p_L(d) = 0.00143 \cdot (1/2.5)^{(d-7)/2}$:

$$10^{-10} = 0.00143 \cdot (0.4)^{(d-7)/2}$$

$$(0.4)^{(d-7)/2} = 7 \times 10^{-8}$$

$$\frac{d-7}{2} \ln(0.4) = \ln(7 \times 10^{-8})$$

$$\frac{d-7}{2} = \frac{-16.5}{-0.92} = 17.9$$

$$d = 7 + 36 = 43$$

**Physical qubits per logical:**
$$n_{\text{phys}} = 2 \times 43^2 - 1 = 3697$$

**Total physical qubits (without magic state overhead):**
$$N = 500 \times 3697 = 1.85 \text{ million}$$

**With magic state factories (~10× overhead):**
$$\boxed{N_{\text{total}} \approx 15-20 \text{ million physical qubits}}$$

### Example 2: Timeline to 1000 Logical Qubits

**Problem:** Starting from 10 logical qubits in 2026, project when 1000 logical qubits will be achieved assuming doubling every 18 months.

**Solution:**

$$N(t) = 10 \cdot 2^{(t-2026)/1.5}$$

For $N = 1000$:
$$1000 = 10 \cdot 2^{(t-2026)/1.5}$$
$$100 = 2^{(t-2026)/1.5}$$
$$\log_2(100) = \frac{t-2026}{1.5}$$
$$6.64 = \frac{t-2026}{1.5}$$
$$t = 2026 + 10.0 = 2036$$

**With slower doubling (24 months):**
$$t = 2026 + 6.64 \times 2 = 2039$$

$$\boxed{\text{1000 logical qubits: 2036 (optimistic) to 2039 (conservative)}}$$

### Example 3: Cryogenic Power Budget

**Problem:** A quantum computer has 10,000 qubits. Each qubit requires 2 control lines with 10 nW heat load each. The dilution refrigerator provides 100 μW cooling power at 20 mK. Can it support the qubits?

**Solution:**

**Total heat load from wiring:**
$$P_{\text{wire}} = 10,000 \times 2 \times 10 \text{ nW} = 200 \text{ μW}$$

**Cooling capacity:** 100 μW

$$\frac{P_{\text{wire}}}{P_{\text{cool}}} = \frac{200}{100} = 2$$

**The heat load exceeds cooling by 2×. This is not sustainable.**

**Solutions needed:**
1. Reduce wire count by 4× (multiplexing)
2. Increase cooling power by 2× (multiple fridges)
3. Cryo-CMOS to reduce heat per line

$$\boxed{\text{Heat load } 2\times \text{ cooling capacity - need multiplexing or more cooling}}$$

---

## Practice Problems

### Direct Application

**Problem 1:** Calculate the code distance needed for $p_L = 10^{-15}$ (useful for very long computations) assuming λ = 2.14 and baseline $p_L(d=7) = 0.143\%$.

**Problem 2:** A company claims they will reach 10,000 physical qubits by 2027 starting from 100 in 2024. What annual growth rate does this imply? Is this realistic compared to historical trends?

**Problem 3:** Estimate the physical qubit count for 100 logical qubits at distance d=17 (roughly what's needed for $p_L \sim 10^{-6}$).

### Intermediate

**Problem 4:** The magic state distillation factory for a T gate typically requires 15-20 physical qubits per logical T gate with high throughput. For a circuit requiring $10^9$ T gates with cycle time 1 μs, estimate:
a) Required factory size for real-time distillation
b) Total physical qubit overhead

**Problem 5:** Compare modular vs. monolithic scaling. If photonic links have 90% fidelity and a distributed surface code needs 10 links per logical qubit, estimate the effective error rate increase. What link fidelity is needed to be negligible?

**Problem 6:** IBM plans multi-chip modules with classical interconnects (no quantum links between chips). What are the implications for error correction across chip boundaries?

### Challenging

**Problem 7:** Derive the optimal code distance as a function of:
- Physical error rate $p$
- Target logical error rate $p_L$
- Total computation time $T$
- Available physical qubits $N$

Express the trade-off between more logical qubits (larger computation) vs. higher distance (lower error rate).

**Problem 8:** Model the economics of quantum computing at scale. If a million-qubit system costs $10B to build and $100M/year to operate, and can solve one "useful" problem per day that generates $1M value, calculate the break-even timeline. How does this change with scale?

---

## Computational Lab: Scaling Projection Tool

```python
"""
Day 838 Computational Lab: Scaling Roadmaps and Projections
Analyzes paths to 1000+ logical qubits
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import List, Tuple

# =============================================================================
# Part 1: Resource Estimation Functions
# =============================================================================

def code_distance_for_error(p_target: float, lambda_val: float,
                            p_ref: float = 0.00143, d_ref: int = 7) -> int:
    """
    Calculate required code distance for target error rate.

    Parameters:
    -----------
    p_target : float
        Target logical error rate
    lambda_val : float
        Error suppression factor
    p_ref : float
        Reference error rate at d_ref
    d_ref : int
        Reference code distance

    Returns:
    --------
    d : int
        Required code distance (odd)
    """
    if p_target >= p_ref:
        return d_ref

    ratio = np.log(p_ref / p_target) / np.log(lambda_val)
    d = d_ref + 2 * np.ceil(ratio)

    # Ensure odd
    d = int(d)
    if d % 2 == 0:
        d += 1

    return d

def physical_qubits_per_logical(d: int) -> int:
    """Physical qubits for distance-d surface code."""
    return 2 * d**2 - 1

def total_physical_qubits(n_logical: int, d: int,
                          magic_state_overhead: float = 10.0) -> int:
    """
    Total physical qubits including magic state distillation.

    Parameters:
    -----------
    n_logical : int
        Number of logical qubits
    d : int
        Code distance
    magic_state_overhead : float
        Multiplier for magic state distillation factories

    Returns:
    --------
    n_physical : int
        Total physical qubits
    """
    n_data = n_logical * physical_qubits_per_logical(d)
    n_magic = int(n_data * (magic_state_overhead - 1))
    return n_data + n_magic

print("=" * 70)
print("RESOURCE ESTIMATION ANALYSIS")
print("=" * 70)

# Test cases
applications = [
    ("Small chemistry", 100, 1e-6, 2.14),
    ("Drug discovery", 500, 1e-8, 2.14),
    ("RSA-2048", 6000, 1e-12, 2.14),
    ("FeMoCo", 3000, 1e-10, 2.14),
    ("Large simulation", 20000, 1e-8, 2.14),
]

print(f"\n{'Application':<20} {'Logical Q':<12} {'Target pL':<12} {'Distance':<10} {'Physical Q':<15}")
print("-" * 75)

for name, n_log, p_target, lam in applications:
    d = code_distance_for_error(p_target, lam)
    n_phys = total_physical_qubits(n_log, d)
    print(f"{name:<20} {n_log:<12} {p_target:<12.0e} {d:<10} {n_phys/1e6:<15.2f}M")

# =============================================================================
# Part 2: Scaling Trajectory Models
# =============================================================================

@dataclass
class ScalingScenario:
    name: str
    start_year: float
    start_qubits: int
    doubling_time: float  # years
    saturation: int = None  # optional saturation point

def project_qubits(scenario: ScalingScenario, year: float) -> int:
    """Project qubit count for a given year."""
    dt = year - scenario.start_year
    if dt < 0:
        return 0

    n = scenario.start_qubits * (2 ** (dt / scenario.doubling_time))

    if scenario.saturation:
        n = min(n, scenario.saturation)

    return int(n)

# Define scenarios
scenarios = [
    ScalingScenario("Google (optimistic)", 2024, 105, 1.5),
    ScalingScenario("Google (conservative)", 2024, 105, 2.5),
    ScalingScenario("IBM (optimistic)", 2024, 1121, 1.5),
    ScalingScenario("IBM (conservative)", 2024, 1121, 2.5),
    ScalingScenario("Neutral Atoms", 2024, 1000, 1.0),
]

print("\n" + "=" * 70)
print("PHYSICAL QUBIT PROJECTIONS")
print("=" * 70)

years = [2024, 2026, 2028, 2030, 2033, 2035, 2040]
print(f"\n{'Scenario':<25}", end="")
for y in years:
    print(f"{y:<12}", end="")
print()
print("-" * 100)

for scenario in scenarios:
    print(f"{scenario.name:<25}", end="")
    for y in years:
        n = project_qubits(scenario, y)
        if n > 1e6:
            print(f"{n/1e6:.1f}M{'':<7}", end="")
        elif n > 1e3:
            print(f"{n/1e3:.0f}K{'':<8}", end="")
        else:
            print(f"{n:<12}", end="")
    print()

# =============================================================================
# Part 3: Logical Qubit Projections
# =============================================================================

def physical_to_logical(n_physical: int, lambda_val: float,
                        target_error: float = 1e-6) -> Tuple[int, int]:
    """
    Convert physical qubit count to logical qubits.

    Returns:
    --------
    n_logical : int
        Number of logical qubits
    d : int
        Code distance used
    """
    d = code_distance_for_error(target_error, lambda_val)
    overhead = physical_qubits_per_logical(d) * 10  # With magic states
    n_logical = n_physical // overhead
    return max(1, n_logical), d

print("\n" + "=" * 70)
print("LOGICAL QUBIT PROJECTIONS (target pL = 1e-6, λ = 2.14)")
print("=" * 70)

print(f"\n{'Year':<8}", end="")
for scenario in scenarios[:4]:
    print(f"{scenario.name.split()[0]:<20}", end="")
print()
print("-" * 90)

for y in years:
    print(f"{y:<8}", end="")
    for scenario in scenarios[:4]:
        n_phys = project_qubits(scenario, y)
        n_log, d = physical_to_logical(n_phys, 2.14)
        print(f"{n_log:<20}", end="")
    print()

# =============================================================================
# Part 4: Engineering Constraints Analysis
# =============================================================================

print("\n" + "=" * 70)
print("ENGINEERING CONSTRAINTS ANALYSIS")
print("=" * 70)

def cryogenic_analysis(n_qubits: int,
                       heat_per_wire: float = 10e-9,  # 10 nW
                       wires_per_qubit: float = 2.0,
                       cooling_power: float = 100e-6) -> dict:  # 100 μW
    """
    Analyze cryogenic heat budget.
    """
    total_heat = n_qubits * wires_per_qubit * heat_per_wire
    fridges_needed = np.ceil(total_heat / cooling_power)

    return {
        'total_heat_uW': total_heat * 1e6,
        'cooling_power_uW': cooling_power * 1e6,
        'fridges_needed': fridges_needed,
        'multiplexing_needed': total_heat / cooling_power
    }

qubit_counts = [1000, 10000, 100000, 1000000]

print(f"\n{'Qubits':<12} {'Heat (μW)':<15} {'Fridges':<12} {'Mux Needed':<12}")
print("-" * 55)

for n in qubit_counts:
    analysis = cryogenic_analysis(n)
    print(f"{n:<12} {analysis['total_heat_uW']:<15.0f} {analysis['fridges_needed']:<12.0f} {analysis['multiplexing_needed']:<12.0f}×")

# =============================================================================
# Part 5: Cost Modeling
# =============================================================================

def system_cost(n_qubits: int,
                base_cost: float = 50e6,  # $50M for base system
                per_qubit: float = 10e3,  # $10K per qubit at scale
                scaling_exp: float = 0.8) -> float:  # Economies of scale
    """
    Estimate system cost.
    """
    return base_cost + per_qubit * (n_qubits ** scaling_exp)

def operating_cost(n_qubits: int,
                   power_per_qubit: float = 100,  # 100W per qubit
                   electricity_rate: float = 0.10,  # $0.10/kWh
                   staff_cost: float = 20e6) -> float:  # $20M/year staff
    """
    Estimate annual operating cost.
    """
    power_kw = n_qubits * power_per_qubit / 1000
    annual_kwh = power_kw * 8760  # hours per year
    electricity_cost = annual_kwh * electricity_rate
    return electricity_cost + staff_cost

print("\n" + "=" * 70)
print("COST ANALYSIS")
print("=" * 70)

print(f"\n{'Qubits':<12} {'Build Cost':<15} {'Annual Op':<15} {'Cost/Qubit':<15}")
print("-" * 60)

for n in qubit_counts:
    build = system_cost(n)
    operate = operating_cost(n)
    per_q = build / n

    print(f"{n:<12} ${build/1e9:.2f}B{'':<8} ${operate/1e6:.0f}M{'':<8} ${per_q/1e3:.0f}K")

# =============================================================================
# Part 6: Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Physical qubit scaling trajectories
ax1 = axes[0, 0]
year_range = np.linspace(2024, 2040, 100)
colors = ['#4285f4', '#34a853', '#ea4335', '#fbbc04', '#9333ea']

for i, scenario in enumerate(scenarios):
    qubits = [project_qubits(scenario, y) for y in year_range]
    ax1.semilogy(year_range, qubits, '-', linewidth=2, color=colors[i],
                label=scenario.name)

# Add milestone lines
ax1.axhline(y=1e6, color='gray', linestyle='--', alpha=0.5)
ax1.text(2024.5, 1.2e6, '1M qubits', fontsize=9, color='gray')
ax1.axhline(y=1e8, color='gray', linestyle='--', alpha=0.5)
ax1.text(2024.5, 1.2e8, '100M qubits', fontsize=9, color='gray')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Physical Qubits', fontsize=12)
ax1.set_title('Physical Qubit Scaling Projections', fontsize=14)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(2024, 2040)
ax1.set_ylim(100, 1e10)

# Plot 2: Logical qubits over time
ax2 = axes[0, 1]

for i, scenario in enumerate(scenarios[:4]):
    logical_qubits = []
    for y in year_range:
        n_phys = project_qubits(scenario, y)
        n_log, _ = physical_to_logical(n_phys, 2.14, 1e-6)
        logical_qubits.append(n_log)

    ax2.semilogy(year_range, logical_qubits, '-', linewidth=2, color=colors[i],
                label=scenario.name)

# Mark application requirements
ax2.axhline(y=100, color='green', linestyle=':', alpha=0.7)
ax2.text(2040.5, 100, 'Small chemistry', fontsize=8, color='green', va='center')
ax2.axhline(y=1000, color='orange', linestyle=':', alpha=0.7)
ax2.text(2040.5, 1000, 'Drug discovery', fontsize=8, color='orange', va='center')
ax2.axhline(y=6000, color='red', linestyle=':', alpha=0.7)
ax2.text(2040.5, 6000, 'RSA-2048', fontsize=8, color='red', va='center')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Logical Qubits (at pL = 10^-6)', fontsize=12)
ax2.set_title('Logical Qubit Projections', fontsize=14)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(2024, 2040)
ax2.set_ylim(1, 1e6)

# Plot 3: Physical qubits vs application requirements
ax3 = axes[1, 0]

app_names = [name for name, _, _, _ in applications]
app_qubits = [total_physical_qubits(n_log, code_distance_for_error(p, lam), 10)
              for _, n_log, p, lam in applications]

y_pos = np.arange(len(app_names))
bars = ax3.barh(y_pos, [q/1e6 for q in app_qubits], color='steelblue',
                edgecolor='black', linewidth=2)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(app_names)
ax3.set_xlabel('Physical Qubits (Millions)', fontsize=12)
ax3.set_title('Physical Qubit Requirements by Application', fontsize=14)
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, app_qubits):
    ax3.text(val/1e6 * 1.2, bar.get_y() + bar.get_height()/2,
             f'{val/1e6:.1f}M', va='center', fontsize=10)

# Plot 4: Cost projections
ax4 = axes[1, 1]

n_range = np.logspace(3, 8, 50)
build_costs = [system_cost(n) for n in n_range]
operate_costs = [operating_cost(n) for n in n_range]

ax4.loglog(n_range, [c/1e9 for c in build_costs], 'b-', linewidth=2, label='Build cost')
ax4.loglog(n_range, [c/1e6 for c in operate_costs], 'r--', linewidth=2, label='Annual operating (M$)')

ax4.set_xlabel('Number of Qubits', fontsize=12)
ax4.set_ylabel('Cost ($ Billions for build, $ Millions for ops)', fontsize=12)
ax4.set_title('System Cost Scaling', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, which='both')

# Mark key points
for n in [1e4, 1e6, 1e8]:
    ax4.axvline(x=n, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('day_838_scaling_roadmaps.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("Visualization saved to: day_838_scaling_roadmaps.png")
print("=" * 70)

# =============================================================================
# Part 7: Timeline to Milestones
# =============================================================================

def years_to_milestone(target_qubits: int, scenario: ScalingScenario) -> float:
    """Calculate years until a qubit milestone is reached."""
    if target_qubits <= scenario.start_qubits:
        return 0

    ratio = target_qubits / scenario.start_qubits
    years = scenario.doubling_time * np.log2(ratio)
    return scenario.start_year + years

print("\n" + "=" * 70)
print("TIMELINE TO MILESTONES")
print("=" * 70)

milestones = [10000, 100000, 1000000, 10000000, 100000000]

print(f"\n{'Scenario':<25}", end="")
for m in milestones:
    print(f"{m/1e6:.1f}M{'':<8}", end="")
print()
print("-" * 90)

for scenario in scenarios:
    print(f"{scenario.name:<25}", end="")
    for m in milestones:
        year = years_to_milestone(m, scenario)
        if year < 2050:
            print(f"{year:.0f}{'':<10}", end="")
        else:
            print(f">{2050:<10}", end="")
    print()

# =============================================================================
# Part 8: Summary Dashboard
# =============================================================================

print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY: PATH TO USEFUL QUANTUM COMPUTING")
print("=" * 70)

print("""
KEY FINDINGS:

1. NEAR-TERM (2025-2027):
   - 1,000-10,000 physical qubits
   - 1-10 logical qubits at useful error rates
   - Limited to small demonstrations

2. MEDIUM-TERM (2028-2032):
   - 100,000+ physical qubits
   - 100+ logical qubits
   - Small-scale quantum advantage possible
   - Chemistry/materials applications viable

3. LONG-TERM (2033+):
   - Million+ physical qubits needed
   - 1,000+ logical qubits
   - Drug discovery, cryptography applications
   - Estimated cost: $1-100 billion systems

4. KEY BOTTLENECKS:
   - Cryogenic cooling: Need 100-1000× multiplexing
   - Control electronics: Need cryo-CMOS
   - Fabrication yield: Need defect tolerance
   - Cost: Need orders of magnitude reduction

5. CRITICAL SUCCESS FACTORS:
   - Improve λ from 2.14 to 3+ (reduces qubit count 10×)
   - Develop efficient magic state factories
   - Scale modularly vs. monolithically
   - Reduce per-qubit cost by 100×
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Physical qubits per logical | $n = 2d^2 - 1$ |
| Distance for target error | $d = d_0 + 2\lceil\ln(p_0/p_{\text{target}})/\ln(\lambda)\rceil$ |
| Total with magic states | $N = n_{\text{log}} \times n_{\text{phys/log}} \times 10$ |
| Exponential scaling | $N(t) = N_0 \cdot 2^{(t-t_0)/\tau}$ |
| Cryogenic heat load | $P = N \times \text{wires} \times 10$ nW |

### Main Takeaways

1. **RSA-2048 requires ~100 million physical qubits** at current λ = 2.14
2. **Improving λ to 3+ would reduce requirements by 10×** - Critical engineering target
3. **Current doubling time is ~1.5-2 years** - Million qubits possible by 2030-2035
4. **Cryogenic scaling is a major bottleneck** - Need multiplexing and cryo-CMOS
5. **Cost for useful systems: $1-100 billion** - Requires major investment
6. **Drug discovery may be achievable before cryptography** - Lower qubit requirements

### Daily Checklist

- [ ] I can estimate physical qubit requirements for target applications
- [ ] I understand the major company roadmaps (Google, IBM, IonQ)
- [ ] I can identify the key engineering bottlenecks
- [ ] I can project scaling trajectories under different scenarios
- [ ] I understand the economics of fault-tolerant quantum computing
- [ ] I completed the scaling projection lab

---

## Preview: Day 839

Tomorrow is the Semester 2A Capstone Project. You will design a complete surface code quantum computer for a specific algorithm, integrating all the knowledge from the past six months. This includes selecting an algorithm, estimating resources, choosing a platform, and developing a realistic implementation plan.

**Capstone requirements:**
- Select an algorithm with practical value
- Complete resource estimation
- Platform selection and justification
- Error budget analysis
- Implementation timeline
