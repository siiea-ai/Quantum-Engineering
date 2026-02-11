# Day 839: Semester 2A Capstone Project

## Week 120, Day 6 | Month 30: Surface Codes | Semester 2A: Quantum Error Correction

### Overview

Today is the Semester 2A Capstone Project. You will design a complete fault-tolerant quantum computer for a specific application, integrating all knowledge from the past six months: error correction codes, decoders, fault-tolerant operations, and real hardware implementations. This project demonstrates mastery of quantum error correction from theory to practice.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Application selection and resource estimation |
| **Afternoon** | 2.5 hours | System design and error budget |
| **Evening** | 1.5 hours | Implementation simulation and documentation |

---

## Capstone Project Specification

### Project Goal

Design a fault-tolerant quantum computing system capable of executing a specific quantum algorithm with practical value. Your design must include:

1. **Algorithm Selection** - Choose an algorithm with real-world impact
2. **Resource Estimation** - Calculate logical and physical qubit requirements
3. **Platform Selection** - Choose and justify a hardware platform
4. **Error Budget Analysis** - Allocate error tolerances across components
5. **Decoder Selection** - Design the classical decoding subsystem
6. **Implementation Timeline** - Realistic roadmap to realization
7. **Working Simulation** - Python code demonstrating key concepts

---

## Part 1: Algorithm Selection (Choose One)

### Option A: Quantum Chemistry - Catalyst Design

**Application:** Design a catalyst for ammonia synthesis (Haber-Bosch alternative)

**Problem specification:**
- Active site: 20-30 atoms (transition metal cluster)
- Required accuracy: 1 kcal/mol (chemical accuracy)
- Target molecule: N₂ activation on iron site

**Algorithm:** Quantum Phase Estimation (QPE) for ground state energy
- Logical qubits needed: 200-500
- T gates: ~$10^9$
- Target success probability: 95%

### Option B: Optimization - Supply Chain Logistics

**Application:** Global shipping route optimization

**Problem specification:**
- Variables: 1000 shipping routes
- Constraints: 5000 delivery deadlines, capacity limits
- Objective: Minimize cost + time

**Algorithm:** Quantum Approximate Optimization Algorithm (QAOA) with error correction
- Logical qubits needed: 500-1000
- Circuit depth: ~1000 layers
- Target improvement over classical: 10%

### Option C: Cryptography - Post-Quantum Transition

**Application:** Factor 1024-bit RSA (demonstrating vulnerability)

**Problem specification:**
- Factor N where $\log_2(N) = 1024$
- Must complete in reasonable time (<1 month)
- Demonstrate cryptographic relevance

**Algorithm:** Shor's algorithm with optimized circuits
- Logical qubits needed: 2000-3000
- T gates: ~$10^9$
- Target success probability: 99%

### Option D: Machine Learning - Drug-Protein Interaction

**Application:** Predict binding affinity for drug candidates

**Problem specification:**
- Protein size: ~200 amino acids
- Drug molecules: ~50 atoms
- Accuracy: Better than molecular dynamics

**Algorithm:** Variational Quantum Eigensolver (VQE) + Quantum Kernel Methods
- Logical qubits needed: 100-200
- Circuit depth: ~500
- Required throughput: 1000 molecules/day

---

## Part 2: Design Template

### 2.1 Algorithm Analysis

**Selected Algorithm:** [Your choice]

**Circuit Characteristics:**
- Number of logical qubits: $n_{\text{log}}$
- T gate count: $N_T$
- Clifford gate count: $N_C$
- Circuit depth: $D$
- Number of measurements: $M$

**Target Error Rate:**
For total success probability $P_{\text{success}}$:

$$p_L^{\text{total}} < 1 - P_{\text{success}}$$

Per-operation error rate:

$$\boxed{p_L^{\text{gate}} < \frac{1 - P_{\text{success}}}{N_T + N_C}}$$

### 2.2 Surface Code Parameters

**Required code distance:**

Using $p_L(d) = p_0 \cdot \lambda^{-(d-d_0)/2}$:

$$d = d_0 + 2\left\lceil \frac{\ln(p_0/p_L^{\text{gate}})}{\ln(\lambda)} \right\rceil$$

**Physical qubits per logical qubit:**

$$n_{\text{phys/log}} = 2d^2 - 1$$

**Magic state factory requirements:**

T gate throughput needed:
$$R_T = \frac{N_T}{T_{\text{compute}}}$$

Factory size (15:1 distillation):
$$n_{\text{factory}} = 15 \times n_{\text{phys/log}} \times \lceil R_T \times T_{\text{cycle}} \rceil$$

### 2.3 Error Budget Allocation

| Component | Allocation | Budget |
|-----------|------------|--------|
| Two-qubit gates | 50% | $p_{2Q}$ |
| Single-qubit gates | 10% | $p_{1Q}$ |
| Measurement | 20% | $p_M$ |
| Idle/memory | 15% | $p_{\text{idle}}$ |
| Decoder | 5% | $p_{\text{decode}}$ |
| **Total** | 100% | $p_{\text{cycle}}$ |

### 2.4 Platform Selection Criteria

| Criterion | Weight | Platform A | Platform B | Platform C |
|-----------|--------|------------|------------|------------|
| Gate fidelity | 30% | | | |
| Qubit count | 25% | | | |
| Connectivity | 15% | | | |
| Cycle time | 15% | | | |
| Scalability | 15% | | | |
| **Score** | 100% | | | |

### 2.5 Decoder Design

**Selected decoder:** [MWPM / Union-Find / Neural / Hybrid]

**Specifications:**
- Latency requirement: $T_{\text{decode}} < T_{\text{cycle}}$
- Throughput: $R_{\text{decode}} > n_{\text{stabilizers}} / T_{\text{cycle}}$
- Hardware: [FPGA / GPU / ASIC]

### 2.6 Implementation Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Phase 1 | Year 1-2 | Component demonstration |
| Phase 2 | Year 2-3 | Single logical qubit |
| Phase 3 | Year 3-4 | Multiple logical qubits |
| Phase 4 | Year 4-5 | Full system integration |
| Phase 5 | Year 5+ | Production operation |

---

## Part 3: Example Capstone Solution

### Example: Option A - Quantum Chemistry Catalyst Design

#### 3.1 Algorithm Analysis

**Application:** Design iron-sulfur catalyst for nitrogen fixation

**Algorithm:** Quantum Phase Estimation
- Logical qubits: 300 (50 spatial orbitals × 2 spin × 3 for ancilla)
- T gates: $5 \times 10^9$
- Clifford gates: $10^{10}$
- Circuit depth: $10^7$ (with parallelization)
- Success probability target: 95%

**Per-gate error budget:**
$$p_L^{\text{gate}} < \frac{0.05}{5 \times 10^9 + 10^{10}} = \frac{0.05}{1.5 \times 10^{10}} = 3.3 \times 10^{-12}$$

#### 3.2 Surface Code Design

**Error suppression factor:** λ = 2.5 (target with improved hardware)

**Reference:** $p_0 = 10^{-3}$ at $d_0 = 7$

**Required distance:**
$$d = 7 + 2\left\lceil \frac{\ln(10^{-3}/3.3 \times 10^{-12})}{\ln(2.5)} \right\rceil$$
$$d = 7 + 2\left\lceil \frac{21.8}{0.92} \right\rceil = 7 + 2 \times 24 = 55$$

**Physical qubits per logical:**
$$n_{\text{phys/log}} = 2 \times 55^2 - 1 = 6049$$

**Data qubit count:**
$$N_{\text{data}} = 300 \times 6049 = 1.81 \text{ million}$$

#### 3.3 Magic State Factory

**T gate rate needed:**
Assuming 1 year runtime: $T = 3 \times 10^7$ seconds
$$R_T = \frac{5 \times 10^9}{3 \times 10^7} = 167 \text{ T gates/second}$$

**Factory design:**
- 15:1 distillation: 15 physical T states → 1 logical T
- Level-2 distillation for higher fidelity
- Factory size: ~100,000 physical qubits
- 10 factories running in parallel

**Total magic state overhead:** ~1 million qubits

#### 3.4 Total System Requirements

| Component | Qubits |
|-----------|--------|
| Data (300 logical) | 1.81 M |
| Magic state factories | 1.0 M |
| Routing/workspace | 0.5 M |
| **Total** | 3.31 M |

$$\boxed{N_{\text{total}} \approx 3.5 \text{ million physical qubits}}$$

#### 3.5 Platform Selection

| Criterion | Google SC | IBM SC | Neutral Atoms |
|-----------|-----------|--------|---------------|
| Gate fidelity (30%) | 8/10 | 7/10 | 7/10 |
| Scalability (25%) | 7/10 | 8/10 | 9/10 |
| Connectivity (15%) | 8/10 | 6/10 | 9/10 |
| Cycle time (15%) | 9/10 | 7/10 | 6/10 |
| Maturity (15%) | 9/10 | 9/10 | 6/10 |
| **Score** | 8.0 | 7.4 | 7.5 |

**Selection:** Superconducting (Google-style architecture)

**Justification:**
- Demonstrated below-threshold operation
- Fastest syndrome cycles
- Clear scaling roadmap

#### 3.6 Error Budget

| Source | Budget | Required Spec |
|--------|--------|---------------|
| CZ gates (4/cycle) | 0.5% | 0.125%/gate |
| Measurement | 0.2% | 0.2% |
| Memory | 0.2% | T1 > 500 μs |
| Single-qubit | 0.05% | 0.025%/gate |
| Decoder | 0.05% | 99.95% accuracy |
| **Total** | 1.0% | Below d=55 threshold |

#### 3.7 Decoder Design

**Primary decoder:** Hierarchical Union-Find + Neural refinement

**Architecture:**
1. **First stage:** Fast Union-Find on FPGA
   - Latency: 100 ns per stabilizer
   - Handles 95% of syndromes

2. **Second stage:** Neural network correction
   - Handles ambiguous cases
   - GPU-accelerated
   - Latency: 1 μs

**Throughput:** 10 million stabilizers/second

#### 3.8 Implementation Timeline

| Year | Milestone | Qubits | Capability |
|------|-----------|--------|------------|
| 2025 | Prototype | 10K | d=9 single logical |
| 2027 | Demo | 100K | 5 logical qubits |
| 2029 | Alpha | 500K | 30 logical qubits |
| 2031 | Beta | 1.5M | 150 logical qubits |
| 2033 | Production | 3.5M | Full chemistry system |

**Total development time:** 9 years
**Estimated cost:** $5-10 billion

---

## Part 4: Capstone Simulation Code

```python
"""
Day 839 Capstone Project: Fault-Tolerant Quantum Computer Design
Complete simulation for chemistry application
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List

# =============================================================================
# Part 1: System Specification
# =============================================================================

@dataclass
class AlgorithmSpec:
    """Specification for target algorithm."""
    name: str
    logical_qubits: int
    t_gates: int
    clifford_gates: int
    circuit_depth: int
    success_probability: float

@dataclass
class SurfaceCodeSpec:
    """Surface code parameters."""
    distance: int
    lambda_factor: float
    cycle_time_us: float
    physical_per_logical: int

@dataclass
class SystemDesign:
    """Complete system design."""
    algorithm: AlgorithmSpec
    code: SurfaceCodeSpec
    total_physical_qubits: int
    magic_state_qubits: int
    runtime_hours: float
    power_mw: float
    cost_billions: float

# =============================================================================
# Part 2: Design Functions
# =============================================================================

def calculate_per_gate_error(algorithm: AlgorithmSpec) -> float:
    """Calculate required per-gate logical error rate."""
    total_ops = algorithm.t_gates + algorithm.clifford_gates
    failure_budget = 1 - algorithm.success_probability
    return failure_budget / total_ops

def calculate_code_distance(p_target: float, lambda_val: float,
                           p_ref: float = 1e-3, d_ref: int = 7) -> int:
    """Calculate required surface code distance."""
    if p_target >= p_ref:
        return d_ref

    ratio = np.log(p_ref / p_target) / np.log(lambda_val)
    d = d_ref + 2 * int(np.ceil(ratio))

    # Ensure odd
    if d % 2 == 0:
        d += 1

    return d

def calculate_magic_state_factory(t_gates: int, runtime_seconds: float,
                                  qubits_per_factory: int = 10000,
                                  distillation_rate: float = 100) -> int:
    """Calculate magic state factory qubit requirements."""
    t_rate = t_gates / runtime_seconds  # T gates per second
    factories_needed = int(np.ceil(t_rate / distillation_rate))
    return factories_needed * qubits_per_factory

def design_system(algorithm: AlgorithmSpec, lambda_factor: float = 2.5,
                  cycle_time_us: float = 1.0) -> SystemDesign:
    """
    Design complete fault-tolerant quantum computing system.

    Parameters:
    -----------
    algorithm : AlgorithmSpec
        Target algorithm specification
    lambda_factor : float
        Error suppression factor
    cycle_time_us : float
        Syndrome extraction cycle time

    Returns:
    --------
    design : SystemDesign
        Complete system design
    """
    # Calculate error requirements
    p_gate = calculate_per_gate_error(algorithm)

    # Calculate code distance
    d = calculate_code_distance(p_gate, lambda_factor)

    # Physical qubits per logical
    phys_per_log = 2 * d**2 - 1

    # Data qubits
    data_qubits = algorithm.logical_qubits * phys_per_log

    # Runtime estimate (assuming 1 logical gate per 10 cycles)
    cycles_per_gate = 10
    total_cycles = algorithm.circuit_depth * cycles_per_gate
    runtime_us = total_cycles * cycle_time_us
    runtime_hours = runtime_us / (3600 * 1e6)

    # Magic state factories
    runtime_seconds = runtime_hours * 3600
    magic_qubits = calculate_magic_state_factory(
        algorithm.t_gates, runtime_seconds
    )

    # Routing overhead (20%)
    routing_qubits = int(0.2 * (data_qubits + magic_qubits))

    # Total
    total_qubits = data_qubits + magic_qubits + routing_qubits

    # Power estimate (100W per 1000 qubits)
    power_mw = total_qubits / 1000 * 0.1

    # Cost estimate ($10K per qubit at scale + $1B base)
    cost_billions = 1 + total_qubits * 1e4 / 1e9

    code_spec = SurfaceCodeSpec(
        distance=d,
        lambda_factor=lambda_factor,
        cycle_time_us=cycle_time_us,
        physical_per_logical=phys_per_log
    )

    return SystemDesign(
        algorithm=algorithm,
        code=code_spec,
        total_physical_qubits=total_qubits,
        magic_state_qubits=magic_qubits,
        runtime_hours=runtime_hours,
        power_mw=power_mw,
        cost_billions=cost_billions
    )

# =============================================================================
# Part 3: Define Algorithms
# =============================================================================

algorithms = {
    'chemistry': AlgorithmSpec(
        name="Catalyst Design (QPE)",
        logical_qubits=300,
        t_gates=int(5e9),
        clifford_gates=int(1e10),
        circuit_depth=int(1e7),
        success_probability=0.95
    ),
    'optimization': AlgorithmSpec(
        name="Supply Chain (QAOA)",
        logical_qubits=500,
        t_gates=int(1e8),
        clifford_gates=int(5e8),
        circuit_depth=int(1e5),
        success_probability=0.90
    ),
    'cryptography': AlgorithmSpec(
        name="RSA-1024 (Shor)",
        logical_qubits=2000,
        t_gates=int(1e9),
        clifford_gates=int(5e9),
        circuit_depth=int(1e8),
        success_probability=0.99
    ),
    'ml': AlgorithmSpec(
        name="Drug Binding (VQE)",
        logical_qubits=150,
        t_gates=int(1e7),
        clifford_gates=int(1e8),
        circuit_depth=int(1e4),
        success_probability=0.95
    )
}

# =============================================================================
# Part 4: Design All Systems
# =============================================================================

print("=" * 80)
print("FAULT-TOLERANT QUANTUM COMPUTER DESIGN")
print("=" * 80)

designs = {}
for key, algo in algorithms.items():
    designs[key] = design_system(algo)

print("\n" + "-" * 80)
print(f"{'Application':<25} {'Logical Q':<12} {'Distance':<10} {'Physical Q':<15} {'Cost':<10}")
print("-" * 80)

for key, design in designs.items():
    print(f"{design.algorithm.name:<25} "
          f"{design.algorithm.logical_qubits:<12} "
          f"{design.code.distance:<10} "
          f"{design.total_physical_qubits/1e6:.2f}M{'':<8} "
          f"${design.cost_billions:.1f}B")

# =============================================================================
# Part 5: Detailed Analysis for Selected Algorithm
# =============================================================================

selected = designs['chemistry']

print("\n" + "=" * 80)
print(f"DETAILED DESIGN: {selected.algorithm.name}")
print("=" * 80)

print(f"""
ALGORITHM SPECIFICATION:
  Logical qubits: {selected.algorithm.logical_qubits}
  T gates: {selected.algorithm.t_gates:.2e}
  Clifford gates: {selected.algorithm.clifford_gates:.2e}
  Circuit depth: {selected.algorithm.circuit_depth:.2e}
  Success probability: {selected.algorithm.success_probability*100}%

SURFACE CODE PARAMETERS:
  Code distance: d = {selected.code.distance}
  Error suppression factor: λ = {selected.code.lambda_factor}
  Physical qubits per logical: {selected.code.physical_per_logical}
  Cycle time: {selected.code.cycle_time_us} μs

SYSTEM REQUIREMENTS:
  Total physical qubits: {selected.total_physical_qubits/1e6:.2f} million
  Magic state factory qubits: {selected.magic_state_qubits/1e6:.2f} million
  Estimated runtime: {selected.runtime_hours:.1f} hours
  Power consumption: {selected.power_mw:.1f} MW
  Estimated cost: ${selected.cost_billions:.1f} billion
""")

# =============================================================================
# Part 6: Error Budget Simulation
# =============================================================================

def simulate_error_budget(design: SystemDesign,
                          n_cycles: int = 1000) -> Dict[str, np.ndarray]:
    """
    Simulate error accumulation over syndrome cycles.
    """
    # Error rates (per cycle)
    p_2q = 0.005  # 0.5% two-qubit gate error
    p_1q = 0.0005  # 0.05% single-qubit gate error
    p_meas = 0.002  # 0.2% measurement error
    p_idle = 0.002  # 0.2% idle error

    # Per-cycle contributions
    n_2q = 4  # 4 CZ gates per data qubit per cycle
    n_1q = 2  # 2 single-qubit gates per cycle

    cycle_error = n_2q * p_2q + n_1q * p_1q + p_meas + p_idle

    # Logical error rate (simplified model)
    d = design.code.distance
    lambda_val = design.code.lambda_factor

    # Threshold assumption
    p_th = 0.01
    p_L_per_cycle = (cycle_error / p_th) ** ((d + 1) / 2)

    # Simulate error accumulation
    cycles = np.arange(n_cycles)
    cumulative_error = 1 - (1 - p_L_per_cycle) ** cycles

    # Component breakdown
    errors = {
        'cycles': cycles,
        'cumulative': cumulative_error,
        'two_qubit': np.ones(n_cycles) * n_2q * p_2q / cycle_error,
        'one_qubit': np.ones(n_cycles) * n_1q * p_1q / cycle_error,
        'measurement': np.ones(n_cycles) * p_meas / cycle_error,
        'idle': np.ones(n_cycles) * p_idle / cycle_error,
        'p_L_per_cycle': p_L_per_cycle
    }

    return errors

errors = simulate_error_budget(selected)

print("\nERROR BUDGET BREAKDOWN:")
print(f"  Two-qubit gates: {errors['two_qubit'][0]*100:.1f}%")
print(f"  Single-qubit gates: {errors['one_qubit'][0]*100:.1f}%")
print(f"  Measurement: {errors['measurement'][0]*100:.1f}%")
print(f"  Idle: {errors['idle'][0]*100:.1f}%")
print(f"  Logical error per cycle: {errors['p_L_per_cycle']:.2e}")

# =============================================================================
# Part 7: Scaling Analysis
# =============================================================================

def analyze_scaling(algorithm: AlgorithmSpec,
                    lambda_range: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze how system size scales with error suppression factor.
    """
    distances = []
    qubits = []

    p_gate = calculate_per_gate_error(algorithm)

    for lam in lambda_range:
        d = calculate_code_distance(p_gate, lam)
        n = algorithm.logical_qubits * (2 * d**2 - 1)
        distances.append(d)
        qubits.append(n)

    return {
        'lambda': lambda_range,
        'distance': np.array(distances),
        'qubits': np.array(qubits)
    }

lambda_range = np.linspace(1.5, 4.0, 50)
scaling = analyze_scaling(selected.algorithm, lambda_range)

# =============================================================================
# Part 8: Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: System comparison
ax1 = axes[0, 0]
names = [d.algorithm.name.split('(')[0].strip() for d in designs.values()]
qubits = [d.total_physical_qubits / 1e6 for d in designs.values()]
costs = [d.cost_billions for d in designs.values()]

x = np.arange(len(names))
width = 0.35

bars1 = ax1.bar(x - width/2, qubits, width, label='Qubits (M)', color='steelblue')
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, costs, width, label='Cost ($B)', color='coral')

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15, ha='right')
ax1.set_ylabel('Physical Qubits (Millions)', fontsize=12, color='steelblue')
ax1_twin.set_ylabel('Cost ($ Billions)', fontsize=12, color='coral')
ax1.set_title('System Requirements by Application', fontsize=14)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Plot 2: Error budget pie chart
ax2 = axes[0, 1]
labels = ['2Q Gates', '1Q Gates', 'Measurement', 'Idle']
sizes = [errors['two_qubit'][0], errors['one_qubit'][0],
         errors['measurement'][0], errors['idle'][0]]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0, 0)

wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels,
                                    colors=colors, autopct='%1.1f%%',
                                    shadow=True, startangle=90)
ax2.set_title('Error Budget Breakdown', fontsize=14)

# Plot 3: Scaling with lambda
ax3 = axes[1, 0]
ax3.semilogy(scaling['lambda'], scaling['qubits'] / 1e6, 'b-', linewidth=2)
ax3.axvline(x=2.14, color='r', linestyle='--', label='Willow (λ=2.14)')
ax3.axvline(x=selected.code.lambda_factor, color='g', linestyle='--',
            label=f'Design target (λ={selected.code.lambda_factor})')

ax3.set_xlabel('Error Suppression Factor λ', fontsize=12)
ax3.set_ylabel('Physical Qubits (Millions)', fontsize=12)
ax3.set_title(f'Scaling for {selected.algorithm.name}', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add annotation
current_q = selected.total_physical_qubits / 1e6
ax3.annotate(f'{current_q:.1f}M qubits\nat λ={selected.code.lambda_factor}',
             xy=(selected.code.lambda_factor, current_q),
             xytext=(3.0, current_q * 2),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10)

# Plot 4: Implementation timeline
ax4 = axes[1, 1]
years = [2025, 2027, 2029, 2031, 2033]
milestones = ['Prototype\n10K qubits', 'Demo\n100K qubits',
              'Alpha\n500K qubits', 'Beta\n1.5M qubits',
              'Production\n3.5M qubits']
qubit_targets = [10000, 100000, 500000, 1500000, 3500000]

ax4.semilogy(years, qubit_targets, 'go-', markersize=12, linewidth=2)
for i, (y, m, q) in enumerate(zip(years, milestones, qubit_targets)):
    ax4.annotate(m, (y, q), textcoords="offset points",
                xytext=(10, 10 if i % 2 == 0 else -30),
                fontsize=9, ha='left')

ax4.axhline(y=selected.total_physical_qubits, color='r', linestyle='--',
            label=f'Target: {selected.total_physical_qubits/1e6:.1f}M')

ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Physical Qubits', fontsize=12)
ax4.set_title('Implementation Timeline', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(2024, 2035)

plt.tight_layout()
plt.savefig('day_839_capstone_design.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("Visualization saved to: day_839_capstone_design.png")
print("=" * 80)

# =============================================================================
# Part 9: Design Summary Report
# =============================================================================

def generate_report(design: SystemDesign) -> str:
    """Generate formatted design report."""
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FAULT-TOLERANT QUANTUM COMPUTER DESIGN                     ║
║                            EXECUTIVE SUMMARY                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Application: {design.algorithm.name:<62} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              ALGORITHM REQUIREMENTS                           ║
╟──────────────────────────────────────────────────────────────────────────────╢
║   Logical Qubits:        {design.algorithm.logical_qubits:<51} ║
║   T Gate Count:          {design.algorithm.t_gates:<51.2e} ║
║   Circuit Depth:         {design.algorithm.circuit_depth:<51.2e} ║
║   Success Probability:   {design.algorithm.success_probability*100:.0f}%{'':<49} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              SURFACE CODE DESIGN                              ║
╟──────────────────────────────────────────────────────────────────────────────╢
║   Code Distance:         d = {design.code.distance:<47} ║
║   Suppression Factor:    λ = {design.code.lambda_factor:<47} ║
║   Physical/Logical:      {design.code.physical_per_logical:<51} ║
║   Cycle Time:            {design.code.cycle_time_us} μs{'':<46} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              SYSTEM REQUIREMENTS                              ║
╟──────────────────────────────────────────────────────────────────────────────╢
║   Total Physical Qubits: {design.total_physical_qubits/1e6:.2f} million{'':<40} ║
║   Magic State Qubits:    {design.magic_state_qubits/1e6:.2f} million{'':<40} ║
║   Runtime:               {design.runtime_hours:.1f} hours{'':<42} ║
║   Power:                 {design.power_mw:.1f} MW{'':<46} ║
║   Estimated Cost:        ${design.cost_billions:.1f} billion{'':<42} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              RECOMMENDED PLATFORM                             ║
╟──────────────────────────────────────────────────────────────────────────────╢
║   Architecture:          Superconducting (Google Willow-derivative)          ║
║   Connectivity:          2D grid with tunable couplers                       ║
║   Decoder:               Hierarchical Union-Find + Neural                    ║
║   Cryogenics:            Multiple dilution refrigerators + cryo-CMOS         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              IMPLEMENTATION TIMELINE                          ║
╟──────────────────────────────────────────────────────────────────────────────╢
║   Phase 1 (2025):        Prototype - 10K qubits, single logical qubit       ║
║   Phase 2 (2027):        Demo - 100K qubits, 5 logical qubits               ║
║   Phase 3 (2029):        Alpha - 500K qubits, 30 logical qubits             ║
║   Phase 4 (2031):        Beta - 1.5M qubits, 150 logical qubits             ║
║   Phase 5 (2033):        Production - 3.5M qubits, full capability          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return report

report = generate_report(selected)
print(report)

# Save report to file
with open('capstone_design_report.txt', 'w') as f:
    f.write(report)

print("Report saved to: capstone_design_report.txt")
```

---

## Capstone Deliverables Checklist

### Required Deliverables

- [ ] **Algorithm Selection Document** (1 page)
  - Justification for chosen application
  - Expected scientific/commercial impact
  - Comparison with classical alternatives

- [ ] **Resource Estimation Spreadsheet**
  - Logical qubit count with justification
  - Gate count breakdown
  - Error budget allocation

- [ ] **System Design Document** (3-5 pages)
  - Surface code parameters
  - Magic state factory design
  - Decoder architecture
  - Hardware platform selection

- [ ] **Implementation Timeline** (1 page)
  - Phase milestones
  - Risk assessment
  - Cost estimates per phase

- [ ] **Working Python Simulation**
  - Resource calculator
  - Error budget analyzer
  - Scaling projections

- [ ] **Presentation Slides** (10-15 slides)
  - Executive summary
  - Technical deep-dive
  - Comparison with alternatives

---

## Grading Rubric

| Component | Weight | Criteria |
|-----------|--------|----------|
| Algorithm Selection | 15% | Relevance, impact, feasibility |
| Resource Estimation | 25% | Accuracy, methodology, completeness |
| System Design | 25% | Technical depth, optimization, realism |
| Error Budget | 15% | Proper allocation, margin analysis |
| Implementation Plan | 10% | Timeline, milestones, risk assessment |
| Code Quality | 10% | Documentation, correctness, visualization |

---

## Summary

### Capstone Learning Outcomes

1. **Integrated system design** - Combined all Semester 2A concepts
2. **Real-world application** - Targeted practical quantum advantage
3. **Quantitative analysis** - Rigorous resource estimation
4. **Engineering trade-offs** - Platform and decoder selection
5. **Professional communication** - Technical documentation

### Next Steps

This capstone prepares you for:
- Research in fault-tolerant quantum computing
- Industry positions in quantum hardware/software
- Further study in quantum algorithms and applications

---

## Preview: Day 840

Tomorrow concludes Semester 2A with a comprehensive synthesis of all six months of quantum error correction study. We will review key concepts from Months 25-30, connect theoretical foundations to experimental implementations, and look ahead to Semester 2B: Quantum Algorithms and Applications.
