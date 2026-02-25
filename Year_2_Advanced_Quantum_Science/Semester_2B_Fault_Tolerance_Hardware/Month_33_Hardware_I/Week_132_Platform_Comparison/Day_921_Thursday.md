# Day 921: Scalability Analysis

## Schedule Overview

| Time Block | Duration | Topic |
|------------|----------|-------|
| Morning | 3 hours | Scaling theory and engineering constraints |
| Afternoon | 2.5 hours | Platform-specific scalability analysis |
| Evening | 1.5 hours | Computational lab: Resource projection |

## Learning Objectives

By the end of today, you will be able to:

1. Identify key scalability bottlenecks for each quantum platform
2. Analyze control system complexity as a function of qubit count
3. Evaluate cryogenic and vacuum engineering requirements
4. Project resource requirements for fault-tolerant systems
5. Compare scalability roadmaps across platforms
6. Assess the role of modular architectures in scaling

## Core Content

### 1. Scalability Fundamentals

#### Defining Scalability

Scalability encompasses multiple dimensions:

1. **Qubit count** ($n$): Number of physical qubits
2. **Gate fidelity at scale** ($F(n)$): How fidelity degrades with system size
3. **Control complexity** ($C(n)$): Resources needed for qubit control
4. **Engineering overhead** ($E(n)$): Infrastructure requirements

True scalability requires:
$$\boxed{\lim_{n \to \infty} \frac{F(n)}{F_{threshold}} > 1 \text{ and } \frac{C(n)}{n^k} < \infty \text{ for small } k}$$

#### Scalability Metrics

**Qubit density:**
$$\rho_q = \frac{n_{qubits}}{A_{chip}} \quad [\text{qubits/mm}^2]$$

**Control line scaling:**
$$N_{lines}(n) = \alpha n + \beta \sqrt{n} + \gamma$$

**Power consumption:**
$$P_{total}(n) = P_{cryo}(n) + P_{control}(n) + P_{readout}(n)$$

### 2. Superconducting Qubit Scalability

#### Current State and Roadmap

| Year | Qubits | Lab/Company |
|------|--------|-------------|
| 2019 | 53 | Google (Sycamore) |
| 2021 | 127 | IBM (Eagle) |
| 2022 | 433 | IBM (Osprey) |
| 2023 | 1,121 | IBM (Condor) |
| 2025+ | 100,000+ | Projected |

#### Control Line Bottleneck

Each qubit requires:
- DC bias lines: 1-2 per tunable qubit
- Microwave drive: 1 XY line + 1 Z line (shared possible)
- Readout: 1 line per ~10 qubits (multiplexed)

Total lines:
$$N_{lines} \approx 2.5n + 10\sqrt{n}$$

**Thermal Load:**
Each coax line conducts heat to the mixing chamber:

$$\dot{Q}_{line} = \kappa \cdot \frac{A}{L} \cdot (T_{300K} - T_{mK})$$

Total heat load:
$$\dot{Q}_{total} = N_{lines} \cdot \dot{Q}_{line} + P_{dissipation}$$

For 1000 qubits: $\dot{Q} \sim 100$ mW at 20 mK (challenging!)

#### Frequency Crowding

With fixed bandwidth $\Delta f$ and qubit frequencies:
$$\delta f_{avg} = \frac{\Delta f}{n}$$

Collision probability:
$$P_{collision} \approx 1 - \prod_{i<j}\left(1 - \frac{|\chi_{ij}|}{\delta f_{avg}}\right)$$

For $n > 100$: requires careful frequency allocation or tunable qubits.

#### Multiplexed Readout

Multiple qubits read through single feedline:

$$H_{readout} = \sum_i \chi_i a^\dagger a \sigma_z^{(i)}$$

Frequency spacing constraint:
$$\delta f_{readout} > \kappa + 2|\chi|$$

Maximum multiplexing: ~10-20 qubits per feedline with current technology.

#### Cryogenic Scaling

Dilution refrigerator capacity:
$$\dot{Q}_{mixing} = \dot{n}_{3He} \cdot (T_{mixing}^2 - T_{still}^2) \cdot k$$

Current systems: 10-30 μW at 20 mK

Scaling requirements:
| Qubits | Heat Load | Refrigerator Size |
|--------|-----------|-------------------|
| 100 | ~10 μW | Standard DR |
| 1,000 | ~100 μW | Large DR |
| 10,000 | ~1 mW | Multiple DRs / Advanced cooling |
| 100,000 | ~10 mW | Novel architectures |

#### Modular Approaches

**Multi-chip Modules:**
- Inter-chip coupling via flip-chip bonding
- ~MHz coupling strength
- Limited connectivity across chips

**Cryogenic Interconnects:**
- Microwave photon links between modules
- Entanglement distribution
- Adds latency and error

### 3. Trapped Ion Scalability

#### Current State and Roadmap

| Year | Qubits | Lab/Company |
|------|--------|-------------|
| 2019 | 20 | IonQ |
| 2021 | 32 | Quantinuum |
| 2023 | 56 | Quantinuum (H2) |
| 2025+ | 1,000+ | Projected (QCCD) |

#### Single-Zone Limits

In a linear Paul trap, mode spacing decreases as:
$$\omega_m^{(k)} \approx \omega_z \sqrt{k} \quad \text{for } k \ll N$$

**Spectral Crowding:**
For N ions: 3N motional modes

Minimum gate detuning:
$$\delta_{gate} > \max(\delta\omega_m) + \Omega_{gate}$$

Practical limit: ~50-100 ions per zone

#### QCCD Architecture

Quantum Charge-Coupled Device architecture:
- Multiple trapping zones
- Ion shuttling between zones
- Junction operations for routing

**Shuttling Overhead:**
$$t_{shuttle} = t_{accel} + \frac{d}{v_{max}} + t_{decel}$$

Typical: 10-100 μs per zone crossing

**Junction Fidelity:**
X-junctions, T-junctions for routing:
$$F_{junction} > 99.99\%$$ required for deep circuits

#### Optical Addressing Challenges

Individual beam addressing requires:
$$\Delta_{beam} < d_{ion} / 2$$

where $d_{ion} \sim 5$ μm is ion spacing.

**Crosstalk:**
$$\epsilon_{crosstalk} = \left(\frac{\Omega_{neighbor}}{\Omega_{target}}\right)^2 \approx \left(\frac{w_0}{d_{ion}}\right)^2$$

For 1% crosstalk: $w_0 < 0.5$ μm (diffraction-limited)

#### Scaling Solutions

**Global Beams + Local Modulation:**
- Acousto-optic deflectors (AODs) for beam steering
- Integrated photonics for scalable delivery

**Photonic Interconnects:**
- Entanglement distribution via photons
- Rate limited by photon collection efficiency
- $\eta_{photon} \sim 10^{-4}$ to $10^{-2}$

**Multiplexed Trap Arrays:**
- 2D trap arrays with multiple zones
- Scalable to thousands of qubits

### 4. Neutral Atom Scalability

#### Current State and Roadmap

| Year | Qubits | Lab/Company |
|------|--------|-------------|
| 2021 | 256 | Harvard/QuEra |
| 2022 | 289 | Pasqual |
| 2023 | 1,000+ | Multiple groups |
| 2025+ | 10,000+ | Projected |

#### Optical Tweezer Scaling

Number of tweezers limited by:
- Laser power: $n \propto P_{laser}/P_{trap}$
- Optical access: geometric constraints
- Control bandwidth: AOD/SLM limitations

**SLM-based Arrays:**
$$n_{max} \sim \frac{A_{SLM}}{(\lambda/NA)^2}$$

Current: >10,000 trap sites demonstrated

**AOD-based Rearrangement:**
Switching time: $\tau_{AOD} \sim 1-10$ μs
Rearrangement: $t_{rearr} \sim N_{moves} \cdot \tau_{AOD}$

#### Atom Loading and Loss

Stochastic loading: initial filling ~50%

Rearrangement algorithm fills target sites:
$$n_{filled} = n_{target}$$ with high probability

**Atom Loss Rate:**
$$\frac{dn}{dt} = -\gamma_{bg} n - \gamma_{light} n - \gamma_{Rydberg} n^2/V$$

Typical vacuum lifetime: 10-100 s

#### Rydberg Interaction Scaling

Van der Waals interaction:
$$V_{dd} = \frac{C_6}{r^6}$$

**Blockade Scaling:**
$$r_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6}$$

For n=50: $C_6 \sim 2\pi \times 100$ GHz μm$^6$
With $\Omega = 2\pi \times 10$ MHz: $r_b \sim 5$ μm

**Parallel Gate Limit:**
Maximum parallel CZ gates = $n/(2r_b/d)$ where $d$ is atom spacing

#### 3D Arrays

Extending to 3D:
$$n_{3D} \sim (n_{1D})^3 \propto 1000 \times \text{increase}$$

Challenges:
- Optical access for imaging/control
- Rearrangement in 3D
- Rydberg interaction anisotropy

### 5. Control System Scaling

#### Classical Control Requirements

**Signal Generation:**
| Platform | Signals/Qubit | Bandwidth |
|----------|---------------|-----------|
| SC | 2-3 | DC-10 GHz |
| TI | 1-2 | 1-100 MHz |
| NA | 0.1-1 | 1-100 MHz |

**AWG/DAC Requirements:**
$$N_{channels} = k \cdot n_{qubits}$$

with k = 2-3 for SC, k ~ 0.1-1 for NA (global control)

#### Wiring and Integration

**Superconducting:**
$$N_{coax} \approx 3n \text{ (optimistic with multiplexing)}$$

Cryogenic wiring density limit: ~1000 lines per refrigerator

**Integrated Control:**
Cryo-CMOS at 4K:
- Reduces line count
- Adds heat load: P ~ 1 mW per control unit
- Active research area

#### FPGA/Compute Requirements

Real-time control latency: $t_{control} < t_{gate}$

| Platform | Gate Time | Control Latency Budget |
|----------|-----------|----------------------|
| SC | 20-100 ns | ~100 ns |
| TI | 10-100 μs | ~1 μs |
| NA | 0.1-1 μs | ~100 ns |

Feedback for error correction:
$$t_{syndrome} + t_{decode} + t_{correction} < t_{coherence}$$

### 6. Comparative Scalability Analysis

#### Scaling Law Summary

| Aspect | SC | TI | NA |
|--------|-----|-----|-----|
| Qubit count | $n \sim 10^3$ (current) | $n \sim 10^2$ | $n \sim 10^3$ |
| Control lines | $O(n)$ | $O(\sqrt{n})$ | $O(1)$ global |
| Cryogenic | $P \sim n$ | Room temp | $P \sim$ const |
| Interconnect | On-chip | Shuttling | Rearrangement |
| Fidelity scaling | Moderate degradation | Stable | Stable |

#### Cost Scaling

$$C_{total}(n) = C_{setup} + C_{qubit} \cdot n^\alpha + C_{control} \cdot n^\beta$$

| Platform | α (qubit) | β (control) | Current $/qubit |
|----------|-----------|-------------|-----------------|
| SC | 1.0 | 1.2 | ~$50k |
| TI | 0.5 | 1.0 | ~$100k |
| NA | 0.2 | 0.5 | ~$10k |

## Quantum Computing Applications

### Fault-Tolerant Scale Requirements

For useful quantum advantage:
- **Shor's algorithm** (2048-bit RSA): ~4,000 logical qubits → 4-20 million physical qubits
- **Quantum chemistry** (FeMoco): ~200 logical qubits → 200k-2M physical qubits
- **Optimization** (practical): ~1,000 logical qubits → 1-10 million physical qubits

### Resource Estimation Framework

Total physical qubits:
$$n_{physical} = n_{logical} \cdot \left(\frac{d}{2}\right)^2 \cdot k_{overhead}$$

where $d$ is code distance and $k_{overhead} \sim 2-5$ for routing/magic states.

For $\epsilon_{physical} = 10^{-3}$ and target $\epsilon_{logical} = 10^{-10}$:
$$d \approx \frac{\log(\epsilon_{logical})}{\log(\epsilon_{physical})} \approx 17$$

Physical qubits per logical: $(17/2)^2 \times 3 \approx 217$

## Worked Examples

### Example 1: Cryogenic Heat Load

**Problem:** A 1000-qubit SC processor uses 2500 coaxial cables with thermal conductance 1 μW/K per cable between 300K and 4K stages. Calculate the heat load at 4K.

**Solution:**

1. Heat per cable:
$$\dot{Q}_{cable} = G \cdot \Delta T = 1 \text{ μW/K} \times (300 - 4) \text{ K} = 296 \text{ μW}$$

2. Total heat load:
$$\dot{Q}_{total} = 2500 \times 296 \text{ μW} = 740 \text{ mW}$$

3. This must be absorbed by the 4K stage (typically ~1W capacity)

4. For the mixing chamber (20 mK), assuming 10^-4 thermal filtering:
$$\dot{Q}_{MC} \approx 740 \times 10^{-4} = 74 \text{ μW}$$

**Answer:** 740 mW at 4K, ~74 μW at 20 mK (requiring a large dilution refrigerator)

### Example 2: Trapped Ion Shuttling Overhead

**Problem:** A QCCD processor has 10 zones with 50 ions each (500 total). An algorithm requires 1000 two-qubit gates between random pairs. Estimate the total shuttling time.

**Solution:**

1. Average distance between random ion pairs:
   - Probability of same zone: 1/10
   - Average zones apart if different: ~3.3

2. Gates requiring shuttling: $1000 \times 0.9 = 900$

3. Average shuttles per cross-zone gate: 2 (one ion moves to partner's zone)

4. Total zone crossings: $900 \times 2 \times 3.3 = 5940$

5. Time per crossing: ~50 μs
$$t_{shuttle} = 5940 \times 50 \text{ μs} = 297 \text{ ms}$$

6. Gate time: $1000 \times 200$ μs = 200 ms

7. Total time: 297 + 200 = 497 ms

**Answer:** ~500 ms total, with shuttling taking ~60% of the time

### Example 3: Neutral Atom Array Scaling

**Problem:** Calculate the laser power needed for a 10,000 atom tweezer array with trap depth 1 mK, waist 0.8 μm, at λ = 852 nm.

**Solution:**

1. Single trap power:
$$P_{trap} = \frac{\pi k_B T \cdot w_0^2}{\alpha' / I_{sat}}$$

where $\alpha'$ is the polarizability ratio.

2. For Cs at magic wavelength:
$$P_{trap} \approx \frac{U_{trap} \cdot \pi w_0^2}{2\alpha}$$

3. Numerical estimate:
$$P_{single} \approx 1 \text{ mW per trap (typical)}$$

4. Total power:
$$P_{total} = 10,000 \times 1 \text{ mW} = 10 \text{ W}$$

5. Including losses (AOD efficiency ~50%, optics ~70%):
$$P_{laser} = \frac{10}{0.5 \times 0.7} = 28.6 \text{ W}$$

**Answer:** ~30W laser power required (achievable with current Ti:Sapph or fiber amplifier systems)

## Practice Problems

### Level 1: Direct Application

1. A dilution refrigerator has 20 μW cooling power at 20 mK. How many qubits can be supported if each qubit adds 10 nW heat load?

2. Calculate the number of motional modes in a linear ion trap with 40 ions. What is the minimum frequency spacing if the trap frequency is 1 MHz?

3. An optical tweezer array uses an SLM with 1920×1080 pixels. If each trap requires 5×5 pixels, what is the maximum number of traps?

### Level 2: Intermediate Analysis

4. Compare the control line count for 1000-qubit systems on SC (3 lines/qubit, 10:1 readout multiplexing) vs TI (QCCD with 20 zones, 4 global beams + 1 addressed beam per zone).

5. A trapped ion QCCD has junction error 10^-4. For a circuit requiring 10,000 junction operations, what is the circuit success probability?

6. Design a modular SC architecture for 10,000 qubits using 100-qubit chips. How many inter-chip connections are needed for a heavy-hex-like connectivity?

### Level 3: Advanced Research-Level

7. Derive the optimal zone size for a QCCD processor that minimizes total operation time for random circuits, considering gate time, shuttling time, and zone-crossing overhead.

8. For a 3D neutral atom array with 100×100×100 sites, calculate the total reconfiguration time to arrange atoms into a defect-free sub-array of 500,000 atoms.

9. Analyze the trade-off between integrated cryo-CMOS control (reduced wiring but added heat) vs room-temperature control (more wiring but no on-chip heat) for a 10,000-qubit SC processor.

## Computational Lab: Scalability Projections

```python
"""
Day 921 Computational Lab: Scalability Analysis
Projects resource requirements for scaling quantum computers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Set plotting style
plt.rcParams['figure.figsize'] = (14, 12)
plt.rcParams['font.size'] = 11

# =============================================================================
# Part 1: Historical Scaling Data
# =============================================================================

# Historical qubit counts
years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

sc_qubits = np.array([9, 17, 20, 53, 65, 127, 433, 1121, 1500])
ti_qubits = np.array([5, 7, 11, 20, 32, 32, 32, 56, 72])
na_qubits = np.array([10, 51, 51, 196, 256, 256, 289, 1000, 2000])

# Fit exponential growth
def exp_growth(x, a, b, c):
    return a * np.exp(b * (x - 2016)) + c

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Historical qubit scaling
ax = axes[0, 0]
ax.semilogy(years, sc_qubits, 'o-', label='Superconducting', color='#1f77b4', markersize=8)
ax.semilogy(years, ti_qubits, 's-', label='Trapped Ion', color='#ff7f0e', markersize=8)
ax.semilogy(years, na_qubits, '^-', label='Neutral Atom', color='#2ca02c', markersize=8)

# Projections
years_proj = np.arange(2024, 2031)
for qubits, color, name in [(sc_qubits, '#1f77b4', 'SC'),
                             (ti_qubits, '#ff7f0e', 'TI'),
                             (na_qubits, '#2ca02c', 'NA')]:
    try:
        popt, _ = curve_fit(exp_growth, years[-5:], qubits[-5:],
                           p0=[10, 0.5, 10], maxfev=10000)
        proj = exp_growth(years_proj, *popt)
        ax.semilogy(years_proj, proj, '--', color=color, alpha=0.5)
    except:
        pass

ax.axhline(y=1e6, color='red', linestyle=':', label='FT threshold (~10^6)')
ax.set_xlabel('Year')
ax.set_ylabel('Qubit Count')
ax.set_title('Qubit Count Scaling')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(2016, 2030)
ax.set_ylim(1, 1e8)

# =============================================================================
# Part 2: Control Complexity Scaling
# =============================================================================

n_qubits = np.logspace(1, 5, 100)

# Control line models
def sc_control_lines(n):
    """Superconducting: ~3 lines per qubit with multiplexing"""
    return 2.5 * n + 10 * np.sqrt(n)  # Optimistic multiplexing

def ti_control_lines(n):
    """Trapped Ion QCCD: scales with zones, not individual ions"""
    n_zones = np.sqrt(n) / 5 + 1  # 50 ions per zone
    return 10 * n_zones + 5  # Global + zone-specific beams

def na_control_lines(n):
    """Neutral Atom: mostly global control"""
    return 5 + 2 * np.log10(n)  # Nearly constant

ax = axes[0, 1]
ax.loglog(n_qubits, sc_control_lines(n_qubits), '-', label='Superconducting', linewidth=2)
ax.loglog(n_qubits, ti_control_lines(n_qubits), '-', label='Trapped Ion', linewidth=2)
ax.loglog(n_qubits, na_control_lines(n_qubits), '-', label='Neutral Atom', linewidth=2)

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('Control Lines/Channels')
ax.set_title('Control Complexity Scaling')
ax.legend()
ax.grid(True, alpha=0.3)

# =============================================================================
# Part 3: Cryogenic/Vacuum Requirements
# =============================================================================

def sc_power(n):
    """Power consumption scaling for SC (cryogenic dominated)"""
    P_cryo_base = 5000  # 5 kW base for DR
    P_per_qubit = 0.5  # W per qubit (control electronics)
    return P_cryo_base + P_per_qubit * n

def ti_power(n):
    """Power for TI (room temp trap, laser dominated)"""
    P_trap = 100  # Base trap power
    P_laser = 50 * np.sqrt(n)  # Laser power scales moderately
    P_control = 10 * np.log10(n)
    return P_trap + P_laser + P_control

def na_power(n):
    """Power for NA (vacuum + lasers)"""
    P_vacuum = 200  # Constant vacuum system
    P_laser = 100 * (n / 1000) ** 0.3  # Weak scaling with n
    P_control = 5 * np.log10(n)
    return P_vacuum + P_laser + P_control

ax = axes[0, 2]
ax.loglog(n_qubits, sc_power(n_qubits), '-', label='Superconducting', linewidth=2)
ax.loglog(n_qubits, ti_power(n_qubits), '-', label='Trapped Ion', linewidth=2)
ax.loglog(n_qubits, na_power(n_qubits), '-', label='Neutral Atom', linewidth=2)

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('System Power (W)')
ax.set_title('Power Consumption Scaling')
ax.legend()
ax.grid(True, alpha=0.3)

# =============================================================================
# Part 4: Fidelity Scaling
# =============================================================================

def sc_fidelity_scaling(n, F0=0.995, crosstalk=1e-5):
    """2Q fidelity degradation with system size"""
    return F0 * (1 - crosstalk * n)

def ti_fidelity_scaling(n, F0=0.999, spectral_crowding=1e-6):
    """Fidelity degradation from mode crowding"""
    return F0 * (1 - spectral_crowding * n**1.5)

def na_fidelity_scaling(n, F0=0.995, crosstalk=1e-6):
    """Minimal degradation due to isolated operation"""
    return F0 * (1 - crosstalk * np.sqrt(n))

ax = axes[1, 0]
n_plot = np.linspace(10, 10000, 100)

ax.plot(n_plot, sc_fidelity_scaling(n_plot), '-', label='Superconducting', linewidth=2)
ax.plot(n_plot, ti_fidelity_scaling(n_plot), '-', label='Trapped Ion', linewidth=2)
ax.plot(n_plot, na_fidelity_scaling(n_plot), '-', label='Neutral Atom', linewidth=2)

ax.axhline(y=0.99, color='red', linestyle=':', label='99% threshold')
ax.axhline(y=0.999, color='green', linestyle=':', label='99.9% threshold')

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('2-Qubit Gate Fidelity')
ax.set_title('Fidelity Scaling with System Size')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.98, 1.001)

# =============================================================================
# Part 5: Cost Scaling
# =============================================================================

def sc_cost(n):
    """Total system cost in $M"""
    C_base = 5  # $5M base (dilution fridge, cleanroom)
    C_per_qubit = 0.02  # $20k per qubit
    C_control = 0.005 * n ** 1.1  # Control electronics
    return C_base + C_per_qubit * n + C_control

def ti_cost(n):
    """Cost for TI system"""
    C_base = 3  # $3M base (trap, vacuum, lasers)
    C_per_zone = 0.5  # $500k per zone
    n_zones = max(1, n / 50)
    return C_base + C_per_zone * n_zones

def na_cost(n):
    """Cost for NA system"""
    C_base = 1  # $1M base (vacuum, basic lasers)
    C_laser = 0.1 * (n / 100) ** 0.5  # SLM/AOD scaling
    return C_base + C_laser

ax = axes[1, 1]
ax.loglog(n_qubits, sc_cost(n_qubits), '-', label='Superconducting', linewidth=2)
ax.loglog(n_qubits, ti_cost(n_qubits), '-', label='Trapped Ion', linewidth=2)
ax.loglog(n_qubits, na_cost(n_qubits), '-', label='Neutral Atom', linewidth=2)

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('System Cost ($M)')
ax.set_title('Estimated Cost Scaling')
ax.legend()
ax.grid(True, alpha=0.3)

# =============================================================================
# Part 6: Resource Requirements for FT Computing
# =============================================================================

# Parameters for resource estimation
physical_error_rates = {
    'Superconducting': 0.003,
    'Trapped Ion': 0.001,
    'Neutral Atom': 0.005
}

def code_distance_needed(p_physical, p_logical_target=1e-10):
    """Estimate surface code distance needed"""
    # Simplified: d ~ log(p_logical) / log(p_physical)
    return int(np.ceil(np.log(p_logical_target) / np.log(p_physical)))

def physical_qubits_per_logical(d):
    """Physical qubits for one surface code logical qubit"""
    return 2 * d ** 2  # Approximate

def total_physical_qubits(n_logical, p_physical, magic_state_overhead=3):
    """Total physical qubits needed"""
    d = code_distance_needed(p_physical)
    per_logical = physical_qubits_per_logical(d)
    return n_logical * per_logical * magic_state_overhead

# Calculate for different logical qubit targets
n_logical_targets = [100, 1000, 10000, 100000]

ax = axes[1, 2]
x = np.arange(len(n_logical_targets))
width = 0.25

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
platforms = list(physical_error_rates.keys())

for idx, (platform, p_err) in enumerate(physical_error_rates.items()):
    physical_counts = [total_physical_qubits(n, p_err) for n in n_logical_targets]
    bars = ax.bar(x + idx*width, physical_counts, width, label=platform, color=colors[idx])

ax.set_xticks(x + width)
ax.set_xticklabels([f'{n:,}' for n in n_logical_targets])
ax.set_xlabel('Logical Qubits Needed')
ax.set_ylabel('Physical Qubits Required')
ax.set_title('Fault-Tolerant Resource Requirements')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('scalability_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 7: Detailed Resource Estimation
# =============================================================================

print("\n" + "="*80)
print("FAULT-TOLERANT RESOURCE ESTIMATION")
print("="*80)

applications = {
    'Shor 2048-bit': {'logical_qubits': 4000, 'T_gates': 1e12},
    'FeMoCo simulation': {'logical_qubits': 200, 'T_gates': 1e10},
    'Quantum advantage optimization': {'logical_qubits': 1000, 'T_gates': 1e8}
}

print(f"\n{'Application':<30} {'Platform':<15} {'Physical Qubits':<18} {'Code Distance'}")
print("-" * 80)

for app_name, app_params in applications.items():
    n_logical = app_params['logical_qubits']

    for platform, p_err in physical_error_rates.items():
        d = code_distance_needed(p_err)
        n_physical = total_physical_qubits(n_logical, p_err)

        print(f"{app_name:<30} {platform:<15} {n_physical:>15,} {d:>10}")

# =============================================================================
# Part 8: Modular Architecture Analysis
# =============================================================================

print("\n" + "="*80)
print("MODULAR ARCHITECTURE REQUIREMENTS")
print("="*80)

def modular_analysis(n_total, n_per_module, interconnect_fidelity=0.99):
    """Analyze modular architecture requirements"""
    n_modules = int(np.ceil(n_total / n_per_module))
    n_interconnects = n_modules * (n_modules - 1) // 2  # Complete connectivity

    # Effective fidelity considering interconnects
    avg_interconnects_per_op = 0.1 * n_modules  # Rough estimate
    F_effective = interconnect_fidelity ** avg_interconnects_per_op

    return {
        'modules': n_modules,
        'interconnects': n_interconnects,
        'effective_fidelity': F_effective
    }

target_qubits = 1000000  # 1M physical qubits

module_sizes = {
    'Superconducting': 1000,  # Current multi-chip limit
    'Trapped Ion': 200,       # QCCD zone capacity
    'Neutral Atom': 10000     # Single chamber limit
}

print(f"\nFor {target_qubits:,} physical qubits:")
print(f"{'Platform':<20} {'Module Size':<15} {'Modules Needed':<18} {'Interconnects'}")
print("-" * 70)

for platform, mod_size in module_sizes.items():
    result = modular_analysis(target_qubits, mod_size)
    print(f"{platform:<20} {mod_size:<15,} {result['modules']:<18,} {result['interconnects']:,}")

# =============================================================================
# Part 9: Timeline Projection
# =============================================================================

print("\n" + "="*80)
print("SCALABILITY TIMELINE PROJECTION")
print("="*80)

milestones = [
    (1000, "NISQ demonstrations"),
    (10000, "Early error correction"),
    (100000, "Logical qubit operations"),
    (1000000, "Fault-tolerant computing"),
    (10000000, "Practical quantum advantage")
]

# Exponential fit parameters (rough estimates)
growth_rates = {
    'Superconducting': 0.6,  # ~2x per year
    'Trapped Ion': 0.35,     # ~1.4x per year
    'Neutral Atom': 0.7      # ~2x per year
}

current_qubits = {
    'Superconducting': 1500,
    'Trapped Ion': 72,
    'Neutral Atom': 2000
}

print(f"\nProjected year to reach qubit milestones:")
print(f"{'Milestone':<12} {'Application':<25} {'SC':<8} {'TI':<8} {'NA':<8}")
print("-" * 65)

for n_target, application in milestones:
    row = f"{n_target:<12,} {application:<25}"
    for platform in ['Superconducting', 'Trapped Ion', 'Neutral Atom']:
        n0 = current_qubits[platform]
        r = growth_rates[platform]
        if n_target > n0:
            years_needed = np.log(n_target / n0) / r
            year = 2024 + years_needed
            row += f"{int(year):<8}"
        else:
            row += f"{'Done':<8}"
    print(row)

print("\n" + "="*80)
print("KEY SCALABILITY INSIGHTS")
print("="*80)
print("""
1. Superconducting systems face control line and cryogenic bottlenecks
   at 10,000+ qubits, requiring modular architectures.

2. Trapped ions have excellent fidelity scaling but qubit count is
   limited by spectral crowding; QCCD enables ~1000 qubits.

3. Neutral atoms offer the most favorable qubit scaling (~10,000+
   demonstrated) with minimal control overhead, but gate fidelity
   needs improvement.

4. Fault-tolerant computing requires 10^5-10^7 physical qubits
   depending on platform error rates and application.

5. Modular architectures are essential for all platforms at the
   fault-tolerant scale; interconnect fidelity becomes critical.

6. Cost per qubit favors neutral atoms at scale, while trapped ions
   may be cost-effective for small, high-fidelity systems.
""")
```

## Summary

### Scalability Metrics by Platform

| Metric | Superconducting | Trapped Ion | Neutral Atom |
|--------|-----------------|-------------|--------------|
| Current max | ~1500 | ~72 | ~2000 |
| 2030 projection | ~100k | ~10k | ~1M |
| Control scaling | O(n) | O(√n) | O(1) |
| Main bottleneck | Cryogenic | Spectral crowding | Gate fidelity |

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Control lines (SC) | $$N \approx 2.5n + 10\sqrt{n}$$ |
| Heat load | $$\dot{Q} \propto N_{lines}$$ |
| Code distance | $$d \approx \log(\epsilon_L)/\log(\epsilon_P)$$ |
| Physical qubits | $$n_{phys} \approx 2d^2 \times n_{log}$$ |

### Main Takeaways

1. **Control complexity** differs dramatically: SC scales linearly, NA is nearly constant
2. **Cryogenic limitations** constrain SC scaling; requires modular approaches
3. **Spectral crowding** limits TI single-zone size; QCCD architecture essential
4. **Neutral atoms** have most favorable raw scaling but need fidelity improvements
5. **Fault-tolerant computing** requires 10^5-10^7 physical qubits across all platforms
6. **Modular interconnects** become the key challenge at large scale

## Daily Checklist

- [ ] I can identify scalability bottlenecks for each platform
- [ ] I understand control system complexity scaling
- [ ] I can estimate cryogenic/vacuum engineering requirements
- [ ] I can project fault-tolerant resource requirements
- [ ] I understand the role of modular architectures
- [ ] I can compare scalability roadmaps across platforms

## Preview of Day 922

Tomorrow we examine **Error Correction Requirements**, analyzing fault-tolerance thresholds, physical qubit overhead, and the timeline to achieve fault-tolerant quantum computing on each platform.
