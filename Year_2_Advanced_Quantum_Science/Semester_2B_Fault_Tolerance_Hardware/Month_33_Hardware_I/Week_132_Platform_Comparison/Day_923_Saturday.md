# Day 923: NISQ vs Fault-Tolerant Roadmaps

## Schedule Overview

| Time Block | Duration | Topic |
|------------|----------|-------|
| Morning | 3 hours | NISQ era analysis and limitations |
| Afternoon | 2.5 hours | Fault-tolerant computing roadmaps |
| Evening | 1.5 hours | Computational lab: Roadmap projections |

## Learning Objectives

By the end of today, you will be able to:

1. Characterize NISQ-era capabilities and fundamental limitations
2. Identify viable NISQ applications for each platform
3. Analyze fault-tolerant computing timelines and milestones
4. Evaluate hybrid classical-quantum approaches
5. Compare platform advantages for different application eras
6. Project the evolution from NISQ to fault-tolerant computing

## Core Content

### 1. NISQ Era Definition and Scope

#### What is NISQ?

NISQ = Noisy Intermediate-Scale Quantum

Defined by John Preskill (2018):
- **Noisy**: Errors not corrected, fidelity degrades with circuit depth
- **Intermediate-Scale**: 50-1000 qubits (beyond classical simulation, below FT)
- **Quantum**: Genuine quantum effects exploited

**NISQ Limitations:**

$$F_{circuit} = F_{1Q}^{n_{1Q}} \cdot F_{2Q}^{n_{2Q}} \cdot e^{-T/T_2}$$

For useful quantum advantage:
$$\boxed{n_{gates} < \frac{\ln(F_{min})}{\ln(F_{gate})} \approx \frac{0.7}{\epsilon_{gate}}}$$

With 0.3% two-qubit error: $n_{gates} \lesssim 230$

#### NISQ Eras by Platform

| Platform | Current Era | Qubit Count | Max Useful Depth |
|----------|-------------|-------------|------------------|
| SC | Mid-NISQ | 100-1000 | 50-100 |
| TI | Late-NISQ | 50-100 | 100-500 |
| NA | Early-NISQ | 100-1000 | 20-50 |

### 2. NISQ Applications Analysis

#### Variational Quantum Algorithms

**Variational Quantum Eigensolver (VQE):**

$$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$$

Optimized classically to find ground state.

**Platform Suitability:**

| Criteria | SC | TI | NA |
|----------|-----|-----|-----|
| Molecular orbital mapping | Good | Excellent | Moderate |
| Connectivity for chemistry | Moderate | Excellent | Good (reconfig) |
| Speed for optimization | Excellent | Poor | Good |
| Noise resilience | Moderate | Good | Moderate |

**Depth Requirements:**
- Small molecules (H2, LiH): ~10-50 layers
- Medium molecules (H2O, N2): ~50-200 layers
- Large molecules (FeMoco): ~500+ layers (FT needed)

#### Quantum Approximate Optimization Algorithm (QAOA)

$$|\gamma, \beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l H_M} e^{-i\gamma_l H_C} |+\rangle^{\otimes n}$$

**Platform Comparison for QAOA:**

| Problem Type | Best Platform | Reason |
|--------------|---------------|--------|
| Dense graphs | TI | All-to-all connectivity |
| Sparse/local | SC, NA | Native nearest-neighbor |
| Large instances | SC, NA | Qubit count |
| High-p QAOA | TI | Circuit depth tolerance |

**Current QAOA Performance:**
- Classical simulation tractable up to n~30 qubits
- NISQ advantage unclear for optimization
- Sampling-based advantages more promising

#### Quantum Machine Learning (QML)

**Variational Classifiers:**
$$p(y|x) = |\langle y|U(\theta)|x\rangle|^2$$

**Kernel Methods:**
$$K(x_i, x_j) = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$$

**Platform Considerations:**
- SC: Fast data loading, rapid iteration
- TI: High fidelity for kernel estimation
- NA: Native for certain data structures (images, graphs)

**NISQ QML Status:**
- No definitive quantum advantage demonstrated
- Barren plateau problem limits scalability
- Classical surrogates often competitive

#### Quantum Simulation

**Digital Simulation (Trotterization):**
$$U(t) = \left(\prod_k e^{-iH_k dt/r}\right)^r + O(dt^2/r)$$

**Analog Simulation:**
Direct implementation of $H_{sim}$ using hardware Hamiltonian.

**Platform Strengths:**

| Simulation Type | SC | TI | NA |
|-----------------|-----|-----|-----|
| 2D spin models | Excellent | Moderate | Excellent |
| Long-range interactions | Moderate | Excellent | Good |
| Dynamics | Good | Moderate | Good |
| Analog simulation | Excellent | Good | Excellent |

### 3. Fault-Tolerant Computing Requirements

#### Milestones to Fault-Tolerance

1. **Below Threshold**: Physical error < threshold (~1%)
2. **Logical Qubit**: Demonstrate logical qubit with lower error than physical
3. **QEC Advantage**: Logical error decreases with code distance
4. **Magic States**: Fault-tolerant non-Clifford gates
5. **Logical Algorithm**: Complete algorithm on logical qubits
6. **Utility-Scale FT**: Useful computation impossible classically

#### Platform Progress Toward FT

**Superconducting:**
| Milestone | Status | Year |
|-----------|--------|------|
| Below threshold | ✓ | 2022 |
| Logical qubit demo | ✓ | 2023 |
| QEC advantage | In progress | 2025 |
| Utility-scale FT | Projected | 2028-2030 |

**Trapped Ion:**
| Milestone | Status | Year |
|-----------|--------|------|
| Below threshold | ✓ | 2020 |
| Logical qubit demo | ✓ | 2024 |
| QEC advantage | Projected | 2025-2026 |
| Utility-scale FT | Projected | 2030-2032 |

**Neutral Atom:**
| Milestone | Status | Year |
|-----------|--------|------|
| Below threshold | In progress | 2024-2025 |
| Logical qubit demo | Projected | 2025-2026 |
| QEC advantage | Projected | 2027-2028 |
| Utility-scale FT | Projected | 2030-2032 |

### 4. Hybrid Approaches

#### Error Mitigation Techniques

Bridge between NISQ and FT:

**Zero-Noise Extrapolation (ZNE):**
$$\langle O\rangle_0 = \lim_{\lambda \to 0} f(\langle O\rangle_\lambda)$$

where $\lambda$ is the noise scaling factor.

**Probabilistic Error Cancellation (PEC):**
$$\mathcal{U} = \sum_i c_i \mathcal{P}_i \circ \mathcal{E}$$

where $\mathcal{P}_i$ are Pauli operations and $c_i$ are quasi-probabilities.

**Overhead:**
| Technique | Sampling Overhead | Effective Depth Increase |
|-----------|-------------------|--------------------------|
| ZNE | 3-10× | 2-3× |
| PEC | Exponential in errors | 1× |
| DD | 1× | 2-5× |

#### Partial Error Correction

**Small Codes (d=3, 5):**
- Correct single errors
- Reduce effective error by ~10×
- Modest qubit overhead (18-50 per logical)

**Platform Application:**
- SC: d=3 surface code demonstrated
- TI: Steane code (7 qubits) demonstrated
- NA: Small codes in development

**Hybrid Error Correction:**
$$\epsilon_{effective} = \epsilon_{partial\_QEC} + \epsilon_{mitigation}$$

Combine small codes with error mitigation for intermediate advantage.

### 5. Application Roadmaps

#### Near-Term (2024-2027): NISQ Era

| Application | Platform Choice | Qubit Need | Status |
|-------------|-----------------|------------|--------|
| Quantum advantage demo | SC | 50-100 | Claimed |
| Small molecule VQE | TI | 20-50 | Active |
| QAOA sampling | SC, NA | 100-500 | Active |
| Quantum simulation | NA, SC | 100-1000 | Active |
| QML exploration | All | 50-200 | Research |

#### Medium-Term (2027-2032): Early FT Era

| Application | Platform Choice | Logical Qubits | Status |
|-------------|-----------------|----------------|--------|
| Error-corrected chemistry | TI | 10-50 | Development |
| Certified random sampling | SC | 20-100 | Development |
| Cryptographic protocols | TI, SC | 50-200 | Research |
| Materials simulation | NA | 100-500 | Research |

#### Long-Term (2032+): Mature FT Era

| Application | Platform Choice | Logical Qubits | Status |
|-------------|-----------------|----------------|--------|
| Shor's algorithm | TI, SC | 4000+ | Future |
| Drug discovery | All | 200-1000 | Future |
| Optimization at scale | Unknown | 1000+ | Future |
| Machine learning | Unknown | 1000+ | Future |

### 6. Platform-Specific Roadmaps

#### Superconducting Roadmap

**IBM Quantum Roadmap:**
- 2023: 1000+ qubits (Condor)
- 2024: Error mitigation at scale
- 2025: First utility demonstrations
- 2027: 100,000+ qubit systems
- 2030+: Fault-tolerant computing

**Google Roadmap:**
- 2024: Below breakeven demonstrations
- 2025: Error-corrected logical qubit advantage
- 2029: Useful fault-tolerant computation

**Key Challenges:**
- Scaling dilution refrigerators
- Control line bottleneck
- Coherence at scale

#### Trapped Ion Roadmap

**IonQ Roadmap:**
- 2024: 64 algorithmic qubits (#AQ)
- 2025: 1024 #AQ
- 2028: Broad quantum advantage

**Quantinuum Roadmap:**
- 2024: H2 system (56 qubits, 99.8% 2Q fidelity)
- 2026: Error-corrected computation
- 2030: Commercial fault-tolerant systems

**Key Challenges:**
- Scaling ion number per trap
- Shuttling overhead in QCCD
- Photonic interconnect efficiency

#### Neutral Atom Roadmap

**QuEra Roadmap:**
- 2024: 256 logical qubits (d=3)
- 2026: Error-corrected operations
- 2028: Practical quantum advantage

**Pasqual Roadmap:**
- 2024: 1000 atoms
- 2025: Mid-circuit measurement
- 2027: Fault-tolerant protocols

**Key Challenges:**
- Two-qubit gate fidelity improvement
- Mid-circuit measurement implementation
- Atom loss mitigation

### 7. Comparative Analysis

#### Best Platform by Application Timeline

| Era | Application | Best Platform | Rationale |
|-----|-------------|---------------|-----------|
| Now | Quantum simulation (analog) | NA | Native programmability |
| Now | Sampling/advantage demos | SC | Speed, scale |
| Now | High-fidelity algorithms | TI | Lowest errors |
| 2027 | Early error correction | TI | Best below threshold |
| 2027 | Large-scale NISQ | SC, NA | Qubit count |
| 2030 | Fault-tolerant chemistry | TI | FT maturity |
| 2030 | Cryptography | SC | Speed at scale |

#### Risk Assessment

| Platform | Technical Risk | Commercial Risk | Timeline Risk |
|----------|---------------|-----------------|---------------|
| SC | Medium (scaling) | Low (mature) | Low |
| TI | Low (physics known) | Medium (scaling) | Medium |
| NA | High (fidelity) | Medium (newer) | High |

## Quantum Computing Applications

### Decision Framework: NISQ vs Wait for FT

**Use NISQ if:**
- Problem maps naturally to shallow circuits
- Approximate solutions acceptable
- Variational approaches sufficient
- Simulation/sampling task

**Wait for FT if:**
- Deep circuits required
- Exact results needed
- Cryptographic applications
- Fault-tolerant gates essential

### Economic Analysis

**NISQ Economics:**
$$\text{Value} = f(\text{speedup}) \times \text{problem\_value} - \text{cost}$$

For NISQ: speedup often <10× for practical problems → limited economic value

**FT Economics:**
$$\text{Value} = \text{exponential\_speedup} \times \text{problem\_value} - \text{cost}$$

For FT: exponential speedup for some problems → transformative value

## Worked Examples

### Example 1: VQE Feasibility Analysis

**Problem:** Determine if a 20-qubit VQE for a chemistry problem is feasible on each platform. The ansatz requires 100 two-qubit gates.

**Solution:**

1. Calculate circuit fidelity for each platform:

**Superconducting (F_2Q = 99.5%):**
$$F_{circuit} = 0.995^{100} = 0.606$$
Marginal - errors significant

**Trapped Ion (F_2Q = 99.9%):**
$$F_{circuit} = 0.999^{100} = 0.905$$
Good - errors manageable

**Neutral Atom (F_2Q = 99.5%):**
$$F_{circuit} = 0.995^{100} = 0.606$$
Marginal - similar to SC

2. Consider error mitigation:
With ZNE (3× overhead), effective fidelity improvement ~1.5×:
- SC: $0.606 \times 1.5 \approx 0.76$ (improved)
- TI: $0.905 \times 1.1 \approx 0.95$ (excellent)

3. Consider sampling overhead:
Chemical accuracy requires many measurements: $N \propto 1/\epsilon^2$
- SC: Fast measurement cycle → practical
- TI: Slow but fewer samples needed

**Answer:** TI best for fidelity, SC competitive with error mitigation for speed.

### Example 2: Timeline to Useful Quantum Advantage

**Problem:** Estimate when each platform achieves quantum advantage for a 50-logical-qubit quantum chemistry simulation.

**Solution:**

1. Requirements:
- 50 logical qubits
- $10^8$ T gates
- Target logical error: $10^{-8}$

2. Physical qubit requirements (from Day 922):
- SC (d=15): $50 \times 450 \times 3 = 67,500$ physical qubits
- TI (d=11): $50 \times 242 \times 3 = 36,300$ physical qubits

3. Current trajectory:
- SC: 1000 qubits (2023) → 100,000 by 2027 (doubling ~1.5 years)
- TI: 56 qubits (2023) → 36,000 by 2030 (doubling ~2 years)

4. Error rate requirements:
- Need p_eff < 0.3% for reasonable code distance
- SC: Currently 0.3-0.5% → achieved ~2025
- TI: Currently 0.1% → already achieved

5. System integration time: +2-3 years after milestones met

**Answer:**
- SC: Physical qubits by 2027, FT system by 2029-2030
- TI: Physical qubits by 2030, FT system by 2032

### Example 3: NISQ Algorithm Selection

**Problem:** A research team has access to a 50-qubit SC processor and wants to demonstrate quantum utility. Recommend an application.

**Solution:**

1. Analyze constraints:
- 50 qubits: Beyond classical simulation for certain problems
- SC depth limit: ~50-100 2Q gates before noise dominates
- Connectivity: Square lattice (nearest-neighbor)

2. Evaluate candidates:

**Sampling (Random Circuit Sampling):**
- Depth 20 achievable with high fidelity
- XEB fidelity measurable
- Quantum advantage claimed
- ✓ Good fit

**QAOA (Max-Cut):**
- Low-p QAOA (p=1-3) feasible
- Limited depth matches platform
- Unclear advantage over classical
- ✓ Moderate fit

**VQE (Small molecule):**
- H6-H10 chain feasible
- Chemistry connectivity harder
- Error mitigation needed
- ✓ Moderate fit

**Quantum Simulation (2D Ising):**
- Native to lattice connectivity
- Analog or digital Trotter
- Scientifically interesting dynamics
- ✓ Excellent fit

**Answer:** Recommend quantum simulation of 2D spin dynamics - natural connectivity match, scientifically valuable, achievable depth.

## Practice Problems

### Level 1: Direct Application

1. Calculate the maximum circuit depth (in 2Q gates) achievable with 90% fidelity on a platform with 99.3% two-qubit gate fidelity.

2. A QAOA circuit for a 30-node Max-Cut problem requires 90 two-qubit gates. Is this feasible on a trapped ion system with 99.8% two-qubit fidelity?

3. How many additional qubits does error mitigation via PEC require if the circuit has 50 errors to cancel probabilistically?

### Level 2: Intermediate Analysis

4. Compare the utility of a 100-qubit SC processor vs. a 30-qubit TI processor for VQE on a 20-qubit chemistry problem requiring 200 two-qubit gates.

5. Estimate when quantum advantage for optimization (QAOA on 1000 variables) might be achieved, given current scaling trajectories.

6. Design a hybrid NISQ-FT approach that uses partial error correction (d=3) with error mitigation. Estimate the effective logical error rate.

### Level 3: Advanced Research-Level

7. Analyze the economic break-even point for quantum computing: what problem value and quantum speedup combination justifies current cloud quantum computing costs (~$1/shot)?

8. Develop a decision tree for platform selection given an application's requirements for qubit count, circuit depth, connectivity, and timeline.

9. Project the transition region from NISQ to fault-tolerant computing, identifying the "useful intermediate" applications that emerge during this period.

## Computational Lab: Roadmap Analysis

```python
"""
Day 923 Computational Lab: NISQ vs Fault-Tolerant Roadmap Analysis
Projects platform evolution and application feasibility
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple

# Set plotting style
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

# =============================================================================
# Part 1: Platform Evolution Projections
# =============================================================================

# Historical and projected data
years = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])

# Qubit counts
sc_qubits = np.array([53, 65, 127, 433, 1121, 1500, 5000, 15000, 50000, 100000, 200000, 500000])
ti_qubits = np.array([20, 32, 32, 32, 56, 72, 200, 500, 1000, 5000, 15000, 30000])
na_qubits = np.array([51, 196, 256, 289, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000])

# Two-qubit gate fidelity (%)
sc_fidelity = np.array([99.1, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.85, 99.9, 99.92, 99.95, 99.97])
ti_fidelity = np.array([99.5, 99.7, 99.8, 99.85, 99.9, 99.92, 99.95, 99.97, 99.98, 99.99, 99.99, 99.995])
na_fidelity = np.array([98.0, 98.5, 99.0, 99.2, 99.5, 99.6, 99.7, 99.8, 99.85, 99.9, 99.93, 99.95])

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot 1: Qubit scaling
ax = axes[0, 0]
ax.semilogy(years, sc_qubits, 'o-', label='Superconducting', color=colors[0], markersize=8)
ax.semilogy(years, ti_qubits, 's-', label='Trapped Ion', color=colors[1], markersize=8)
ax.semilogy(years, na_qubits, '^-', label='Neutral Atom', color=colors[2], markersize=8)

ax.axhline(y=1e6, color='red', linestyle=':', alpha=0.5, label='FT threshold (~10^6)')
ax.axhspan(50, 1000, alpha=0.1, color='blue', label='NISQ range')
ax.set_xlabel('Year')
ax.set_ylabel('Qubit Count')
ax.set_title('Qubit Scaling Roadmap')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(10, 2e6)

# Plot 2: Fidelity evolution
ax = axes[0, 1]
ax.plot(years, sc_fidelity, 'o-', label='Superconducting', color=colors[0], markersize=8)
ax.plot(years, ti_fidelity, 's-', label='Trapped Ion', color=colors[1], markersize=8)
ax.plot(years, na_fidelity, '^-', label='Neutral Atom', color=colors[2], markersize=8)

ax.axhline(y=99.0, color='red', linestyle=':', label='Threshold (~99%)')
ax.axhline(y=99.9, color='green', linestyle=':', label='FT comfortable (99.9%)')
ax.set_xlabel('Year')
ax.set_ylabel('Two-Qubit Gate Fidelity (%)')
ax.set_title('Fidelity Improvement Trajectory')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(97.5, 100.05)

# =============================================================================
# Part 2: Circuit Depth Feasibility
# =============================================================================

def max_circuit_depth(fidelity: float, target_circuit_fidelity: float = 0.5) -> int:
    """Calculate maximum circuit depth for given gate fidelity"""
    error = 1 - fidelity / 100
    if error <= 0:
        return 10000
    return int(np.log(target_circuit_fidelity) / np.log(1 - error))

# Plot 3: Achievable circuit depth over time
ax = axes[0, 2]

sc_depth = [max_circuit_depth(f) for f in sc_fidelity]
ti_depth = [max_circuit_depth(f) for f in ti_fidelity]
na_depth = [max_circuit_depth(f) for f in na_fidelity]

ax.semilogy(years, sc_depth, 'o-', label='Superconducting', color=colors[0], markersize=8)
ax.semilogy(years, ti_depth, 's-', label='Trapped Ion', color=colors[1], markersize=8)
ax.semilogy(years, na_depth, '^-', label='Neutral Atom', color=colors[2], markersize=8)

# Application depth requirements
applications_depth = {
    'QAOA (p=3)': 50,
    'VQE (small)': 100,
    'VQE (medium)': 500,
    'Chemistry (FT)': 5000,
}

for app, depth in applications_depth.items():
    ax.axhline(y=depth, color='gray', linestyle='--', alpha=0.5)
    ax.text(2030.5, depth, app, fontsize=8, va='center')

ax.set_xlabel('Year')
ax.set_ylabel('Max Circuit Depth (50% fidelity)')
ax.set_title('Achievable Circuit Depth')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(10, 20000)
ax.set_xlim(2019, 2031)

# =============================================================================
# Part 3: NISQ Application Feasibility Map
# =============================================================================

ax = axes[1, 0]

# Application requirements
applications = {
    'Sampling/RCS': {'qubits': 50, 'depth': 30, 'fidelity': 99.0},
    'QAOA (low-p)': {'qubits': 100, 'depth': 50, 'fidelity': 99.3},
    'VQE (H6)': {'qubits': 20, 'depth': 100, 'fidelity': 99.5},
    'VQE (FeMo)': {'qubits': 200, 'depth': 1000, 'fidelity': 99.9},
    'QEC demo': {'qubits': 100, 'depth': 500, 'fidelity': 99.5},
    'Quantum sim': {'qubits': 100, 'depth': 50, 'fidelity': 99.3},
}

# Create feasibility heatmap
def check_feasibility(platform_qubits, platform_fidelity, app):
    """Check if application is feasible"""
    if platform_qubits < app['qubits']:
        return 0  # Insufficient qubits
    if platform_fidelity < app['fidelity']:
        return 0.5  # Marginal fidelity
    if max_circuit_depth(platform_fidelity) < app['depth']:
        return 0.5  # Marginal depth
    return 1  # Feasible

# 2024 status
year_idx = 5  # 2024

feasibility_data = []
app_names = list(applications.keys())

for platform_name, qubits, fidelity in [('SC', sc_qubits[year_idx], sc_fidelity[year_idx]),
                                         ('TI', ti_qubits[year_idx], ti_fidelity[year_idx]),
                                         ('NA', na_qubits[year_idx], na_fidelity[year_idx])]:
    row = []
    for app_name, app_req in applications.items():
        feas = check_feasibility(qubits, fidelity, app_req)
        row.append(feas)
    feasibility_data.append(row)

feasibility_array = np.array(feasibility_data)

im = ax.imshow(feasibility_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(app_names)))
ax.set_xticklabels(app_names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels(['Superconducting', 'Trapped Ion', 'Neutral Atom'])
ax.set_title('Application Feasibility (2024)')

# Add text annotations
for i in range(3):
    for j in range(len(app_names)):
        text = '✓' if feasibility_array[i, j] == 1 else ('~' if feasibility_array[i, j] == 0.5 else '✗')
        ax.text(j, i, text, ha='center', va='center', fontsize=12,
               color='white' if feasibility_array[i, j] < 0.7 else 'black')

plt.colorbar(im, ax=ax, label='Feasibility')

# =============================================================================
# Part 4: Timeline to Milestones
# =============================================================================

ax = axes[1, 1]

milestones = [
    'Below threshold',
    'QEC demo',
    'Logical advantage',
    'Early FT',
    'Useful FT',
]

# Estimated years
sc_milestones = [2022, 2023, 2025, 2028, 2030]
ti_milestones = [2020, 2024, 2026, 2030, 2032]
na_milestones = [2024, 2026, 2028, 2030, 2032]

x = np.arange(len(milestones))
width = 0.25

bars1 = ax.barh(x - width, sc_milestones, width, label='Superconducting', color=colors[0])
bars2 = ax.barh(x, ti_milestones, width, label='Trapped Ion', color=colors[1])
bars3 = ax.barh(x + width, na_milestones, width, label='Neutral Atom', color=colors[2])

ax.axvline(x=2024, color='red', linestyle='--', label='Current (2024)')

ax.set_yticks(x)
ax.set_yticklabels(milestones)
ax.set_xlabel('Year')
ax.set_title('Milestone Timeline')
ax.legend(loc='lower right')
ax.set_xlim(2018, 2035)
ax.grid(True, alpha=0.3, axis='x')

# =============================================================================
# Part 5: Application Era Diagram
# =============================================================================

ax = axes[1, 2]

# Define eras
eras = {
    'NISQ': (2019, 2027),
    'Early FT': (2027, 2032),
    'Mature FT': (2032, 2040)
}

# Applications by era
era_apps = {
    'NISQ': ['Sampling', 'Variational', 'Analog sim', 'QML explore'],
    'Early FT': ['Small QEC algo', 'Certified random', 'Crypto protocols'],
    'Mature FT': ['Cryptanalysis', 'Drug discovery', 'Materials', 'Optimization']
}

y_positions = {'NISQ': 3, 'Early FT': 2, 'Mature FT': 1}

for era, (start, end) in eras.items():
    y = y_positions[era]
    ax.barh(y, end - start, left=start, height=0.6, alpha=0.3,
           label=era if era == 'NISQ' else None)
    ax.text(start + (end - start) / 2, y, era,
           ha='center', va='center', fontweight='bold', fontsize=11)

    # Add applications
    apps = era_apps[era]
    for i, app in enumerate(apps):
        x_pos = start + (end - start) * (i + 0.5) / len(apps)
        ax.text(x_pos, y - 0.35, app, ha='center', va='top', fontsize=8, rotation=45)

ax.axvline(x=2024, color='red', linestyle='--', linewidth=2, label='Current')

ax.set_xlim(2018, 2040)
ax.set_ylim(0, 4)
ax.set_xlabel('Year')
ax.set_title('Quantum Computing Era Transitions')
ax.set_yticks([])
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('nisq_ft_roadmap.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 6: Hybrid Strategy Analysis
# =============================================================================

print("\n" + "="*80)
print("HYBRID NISQ-FT STRATEGY ANALYSIS")
print("="*80)

def hybrid_fidelity(p_phys: float, code_distance: int,
                    mitigation_factor: float = 1.5) -> float:
    """
    Calculate effective fidelity with small-code QEC + mitigation
    """
    # Small code provides error suppression
    p_logical = 0.1 * (p_phys * 10) ** ((code_distance + 1) / 2)

    # Error mitigation provides additional factor
    p_effective = p_logical / mitigation_factor

    return 1 - p_effective

print("\nEffective error rates with hybrid QEC + mitigation:")
print(f"{'Platform':<18} {'Raw p':<10} {'d=3 code':<12} {'d=5 code':<12} {'+ mitigation':<12}")
print("-" * 64)

platforms_p = {
    'Superconducting': 0.003,
    'Trapped Ion': 0.001,
    'Neutral Atom': 0.005
}

for name, p_raw in platforms_p.items():
    f_d3 = 1 - 0.1 * (p_raw * 10) ** 2
    f_d5 = 1 - 0.1 * (p_raw * 10) ** 3
    f_hybrid = hybrid_fidelity(p_raw, 5, 2.0)

    print(f"{name:<18} {p_raw*100:.2f}%     {f_d3*100:.3f}%     {f_d5*100:.4f}%    {f_hybrid*100:.5f}%")

# =============================================================================
# Part 7: Economic Analysis
# =============================================================================

print("\n" + "="*80)
print("ECONOMIC VIABILITY ANALYSIS")
print("="*80)

# Current cloud quantum pricing (approximate)
pricing = {
    'IBM': {'cost_per_shot': 0.001, 'qubits': 127},  # Free tier available
    'IonQ': {'cost_per_shot': 0.01, 'qubits': 32},
    'Rigetti': {'cost_per_shot': 0.003, 'qubits': 80},
}

# Break-even analysis
def break_even_speedup(classical_time: float, quantum_time: float,
                       quantum_cost: float, classical_cost: float) -> float:
    """Calculate required speedup for economic break-even"""
    # Total cost = time * rate
    # Break even when: classical_cost * classical_time = quantum_cost * quantum_time
    # Speedup = classical_time / quantum_time
    # Required speedup = quantum_cost / classical_cost
    return quantum_cost / classical_cost if classical_cost > 0 else float('inf')

print("\nBreak-even analysis for quantum computing:")
print(f"Cloud classical compute: ~$0.001/CPU-hour")
print(f"Cloud quantum compute: ~$0.01-1.00/shot")
print(f"Required speedup for break-even: 10-1000× minimum")
print(f"\nCurrent NISQ speedups: Often <10× for practical problems")
print(f"Expected FT speedups: Exponential for certain problems (crypto, simulation)")

# =============================================================================
# Part 8: Decision Framework
# =============================================================================

print("\n" + "="*80)
print("PLATFORM SELECTION DECISION FRAMEWORK")
print("="*80)

print("""
Use this decision tree for platform selection:

1. QUBIT COUNT REQUIREMENT
   - <50 qubits: All platforms viable
   - 50-200 qubits: SC, NA preferred
   - >200 qubits: NA (now), SC (scaling)

2. CIRCUIT DEPTH REQUIREMENT
   - <50 layers: All platforms viable
   - 50-200 layers: TI preferred (highest fidelity)
   - >200 layers: TI only OR wait for FT

3. CONNECTIVITY REQUIREMENT
   - Local (2D lattice): SC, NA optimal
   - Long-range/all-to-all: TI optimal
   - Reconfigurable: NA optimal

4. TIME-TO-SOLUTION
   - Fastest iteration: SC (μs gates)
   - Moderate: NA (μs gates)
   - Slowest but highest quality: TI (ms gates)

5. TIMELINE
   - Need results now: Use best available
   - Can wait 2-3 years: Consider early FT benefits
   - Long-term R&D: Position for FT era

6. APPLICATION TYPE
   - Sampling/advantage demos: SC
   - Variational algorithms: TI (fidelity) or SC (speed)
   - Quantum simulation: NA (analog) or SC (digital)
   - Error correction research: All platforms active
""")

print("\n" + "="*80)
print("KEY ROADMAP INSIGHTS")
print("="*80)
print("""
1. NISQ ERA STATUS (2024):
   - Quantum advantage demonstrated for sampling (disputed utility)
   - No clear advantage for optimization or ML
   - Variational algorithms show promise but limited scale
   - Quantum simulation provides scientific insights

2. TRANSITION PERIOD (2025-2030):
   - Error mitigation extends NISQ capabilities
   - Small-code QEC provides intermediate benefits
   - First useful fault-tolerant demonstrations expected
   - Platform differentiation increases

3. FAULT-TOLERANT ERA (2030+):
   - SC likely leads in scale and speed
   - TI leads in fidelity and may enable FT first
   - NA could provide best cost-performance ratio
   - Application-specific optimization becomes crucial

4. RECOMMENDATIONS:
   - For immediate results: Use hybrid error mitigation
   - For research positioning: Engage with FT development
   - For production planning: Plan for 2028-2030 FT availability
   - For platform selection: Match to application requirements
""")
```

## Summary

### NISQ vs FT Comparison

| Aspect | NISQ Era | Fault-Tolerant Era |
|--------|----------|-------------------|
| Error handling | Mitigation, accept noise | Full correction |
| Circuit depth | 50-500 gates | Unlimited (logical) |
| Qubit overhead | 1:1 | 100-1000:1 |
| Applications | Variational, sampling | All quantum algorithms |
| Timeline | Now-2027 | 2028+ |

### Platform Roadmap Summary

| Milestone | SC | TI | NA |
|-----------|-----|-----|-----|
| NISQ scale (100+) | 2021 ✓ | 2024 | 2023 ✓ |
| QEC demo | 2023 ✓ | 2024 | 2026 |
| Logical advantage | 2025 | 2026 | 2028 |
| Useful FT | 2030 | 2032 | 2030 |

### Key Formulas

| Quantity | Formula |
|----------|---------|
| NISQ depth limit | $$n_{gates} < 0.7/\epsilon_{gate}$$ |
| Circuit fidelity | $$F = F_{gate}^{n_{gates}}$$ |
| Mitigation overhead | $$N_{samples} \propto e^{O(n\epsilon)}$$ |

### Main Takeaways

1. **NISQ era** is mature but has limited proven utility beyond sampling
2. **Hybrid approaches** (mitigation + small codes) bridge to FT
3. **Platform choice** depends critically on application requirements
4. **Timeline to useful FT** is 2028-2032 across all platforms
5. **Economic break-even** requires significant speedup not yet demonstrated for most problems

## Daily Checklist

- [ ] I understand NISQ limitations and feasibility constraints
- [ ] I can identify viable NISQ applications for each platform
- [ ] I know the milestones on the path to fault-tolerant computing
- [ ] I can evaluate hybrid classical-quantum approaches
- [ ] I understand platform-specific roadmaps and timelines
- [ ] I can apply a decision framework for platform selection

## Preview of Day 924

Tomorrow we conclude Month 33 with **Month Synthesis**, reviewing platform selection criteria, surveying the industry landscape, and identifying key research directions for hardware development.
