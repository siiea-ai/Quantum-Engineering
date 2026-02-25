# Day 991: Hardware Scaling Progress

## Month 36, Week 142, Day 4 | Research Frontiers

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Theory: Multi-Platform Scaling Analysis |
| Afternoon | 2.5 hrs | Critical Analysis: Scaling Challenges |
| Evening | 2 hrs | Lab: Scaling Projections & Visualization |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Compare** scaling trajectories across quantum computing platforms
2. **Analyze** the technical challenges limiting each approach
3. **Evaluate** modular and monolithic scaling strategies
4. **Assess** connectivity and topology trade-offs
5. **Project** future capabilities based on current trends
6. **Identify** key bottlenecks for each technology

---

## Core Content

### 1. The Scaling Challenge

#### Defining Scalability

True quantum computing scalability requires simultaneous improvement in:

$$\boxed{\text{Scalability} = f(\text{Qubits}, \text{Fidelity}, \text{Connectivity}, \text{Speed})}$$

Increasing qubit count while degrading quality is not genuine scaling.

#### Key Scaling Metrics

| Metric | Definition | Target (2030) |
|--------|------------|---------------|
| Physical qubits | Number of controllable qubits | 10,000+ |
| Logical qubits | Error-corrected computational units | 100+ |
| Two-qubit gate fidelity | $1 - p_{\text{error}}$ | 99.99%+ |
| Connectivity | Average qubit degree | >3 |
| Gate speed | Operations per second | >10 MHz |
| Coherence ratio | $T_2 / t_{\text{gate}}$ | >10,000 |

### 2. Superconducting Qubit Scaling

#### Current State (2025)

**Major Players:**
- Google: 105 qubits (Willow)
- IBM: 1,121 qubits (Condor), 133 qubits (Heron)
- Rigetti: 84 qubits (Ankaa-2)
- IQM: 54 qubits
- Alibaba/Baidu: 100+ qubits (limited public data)

**Scaling Trajectory:**

| Year | Google | IBM | Rigetti |
|------|--------|-----|---------|
| 2019 | 53 | 53 | 32 |
| 2020 | 53 | 65 | 32 |
| 2021 | 72 | 127 | 80 |
| 2022 | 72 | 433 | 80 |
| 2023 | 72 | 1,121 | 84 |
| 2024 | 105 | 133 (Heron) | 84 |
| 2025 | 120 (proj) | 200+ (Flamingo) | 100 |

#### Technical Challenges

**1. Wiring Density**

Each qubit requires:
- DC bias lines: ~3 per qubit
- Microwave control: 1-2 per qubit
- Readout lines: shared among ~5-10 qubits

For n qubits: $\sim 4n$ wires through dilution refrigerator

$$\text{Cryostat capacity} \approx 1000-2000 \text{ wires currently}$$

**Bottleneck:** Beyond ~500 qubits, wiring becomes limiting factor.

**Solutions being developed:**
- Cryo-CMOS control electronics (Intel, Google)
- Multiplexed readout
- On-chip signal generation

**2. Frequency Crowding**

Transmon frequencies: 4-6 GHz range
With anharmonicity ~200 MHz, spacing needed: ~50 MHz

$$N_{\text{max}} = \frac{2 \text{ GHz}}{50 \text{ MHz}} = 40 \text{ unique frequencies}$$

Beyond this, frequency collisions cause crosstalk.

**Solutions:**
- Tunable qubits (frequency tuning during gates)
- Parametric gates (avoid frequency matching)
- Advanced microwave shaping

**3. Thermal Management**

Dilution refrigerators provide ~20 mW cooling at 20 mK.
Each active qubit dissipates ~1 μW.

$$N_{\text{max}} = \frac{20 \text{ mW}}{1 \text{ μW}} = 20,000 \text{ qubits}$$

Not immediately limiting, but approaches physical limits for large systems.

#### Modular Approaches

**IBM's Flamingo Architecture:**

Multiple chips connected via:
- Microwave resonator coupling (short-range)
- Photonic links (longer-range, under development)

$$\text{Inter-chip gate fidelity target} > 99\%$$

**Google's Approach:**

Monolithic scaling with improved fabrication:
- Single-chip designs up to 1000 qubits planned
- Focus on quality over quantity initially

### 3. Trapped-Ion Scaling

#### Current State (2025)

**Major Players:**
- Quantinuum (H2): 56 qubits, 99.9% 2Q gates
- IonQ (Forte): 36 algorithmic qubits
- Alpine Quantum (formerly AQT): 24 qubits
- Oxford Ionics: 20+ qubits

**Scaling Trajectory:**

| Year | Quantinuum | IonQ |
|------|------------|------|
| 2020 | 10 | 32 |
| 2021 | 12 | 32 |
| 2022 | 20 | 32 |
| 2023 | 32 | 36 |
| 2024 | 56 | 36 |
| 2025 | 72 (proj) | 64 (proj) |
| 2030 | 1000+ (modular) | 1024 (modular) |

#### Technical Challenges

**1. Ion Chain Length**

Single linear trap limitations:
- Radial confinement weakens with chain length
- Heating rate increases
- Practical limit: ~50-100 ions per chain

$$\text{Heating rate} \propto N_{\text{ions}}^{\alpha}, \quad \alpha \approx 2$$

**2. Motional Mode Crowding**

Gate speed limited by motional mode spacing:

$$\omega_{\text{spacing}} \propto \frac{1}{N_{\text{ions}}}$$

More ions → slower gates → more decoherence.

**3. All-to-All Connectivity Cost**

While ions have intrinsic all-to-all connectivity, realizing it requires:
- Ion shuttling (Quantinuum approach)
- Photonic interconnects (IonQ approach)

Shuttling time: ~100 μs per reconfiguration
Photonic link fidelity: currently ~90% (improving to 99%+)

#### Modular Architectures

**Quantum Charge-Coupled Device (QCCD):**

Quantinuum's approach:
- Multiple trap zones
- Ion shuttling between zones
- Junction operations for routing

$$\text{Shuttling overhead} \sim 10\% \text{ of circuit time}$$

**Photonic Interconnects:**

IonQ and others:
- Entanglement via photon interference
- Remote entanglement rate: ~100-1000 Hz
- Enables distributed quantum computing

### 4. Neutral Atom Scaling

#### Current State (2025)

**Major Players:**
- QuEra: 256 atoms, logical qubit demos
- Pasqal: 200+ atoms, analog/digital hybrid
- Atom Computing: 1200+ atoms (record)
- ColdQuanta/Infleqtion: 100+ atoms

**Key Advantages:**
- Natural parallelism (identical atoms)
- Reconfigurable geometry
- No wiring per qubit

**Scaling Trajectory:**

| Year | QuEra | Atom Computing | Pasqal |
|------|-------|----------------|--------|
| 2021 | 256 | 100 | 100 |
| 2022 | 256 | 225 | 196 |
| 2023 | 280 | 1200 | 200 |
| 2024 | 300 | 1200+ | 300 |
| 2025 | 1000 (proj) | 2000+ (proj) | 500+ |

#### Technical Challenges

**1. Atom Loss**

Atoms can be lost from traps:
- Collision with background gas
- Off-resonant scattering during gates
- Rydberg autoionization

$$\text{Loss rate} \sim 1\% \text{ per second typical}$$

For a 100 ms computation with 1000 atoms: ~10 atoms lost

**2. Two-Qubit Gate Fidelity**

Rydberg-mediated gates currently at 99.5%:
- Limited by laser noise
- Atomic motion
- Rydberg state lifetime

Target: 99.9%+ needed for error correction

**3. Atom Rearrangement**

Preparing defect-free arrays requires:
- Detection of filled sites
- Rearrangement to fill vacancies
- Time: 10-100 ms per attempt

$$\text{Filling probability} = (1 - p_{\text{loss}})^{N_{\text{attempts}}}$$

#### Scaling Strategies

**Tweezer Arrays:**
- Individual optical tweezers per atom
- Highly reconfigurable
- Limited by optical aberrations at large scale

**Optical Lattices:**
- Periodic potentials, many atoms
- Less control over individual atoms
- Natural for analog simulation

**Hybrid Approaches:**
- Zones for storage and computation
- Shuttling atoms between regions

### 5. Photonic Quantum Computing

#### Current State (2025)

**Major Players:**
- Xanadu: Borealis (216 modes), programmable
- PsiQuantum: Fault-tolerant architecture (development)
- Quix: Linear optical processors

**Unique Features:**
- Room temperature operation
- Natural for communication
- Probabilistic gates

#### Technical Challenges

**1. Probabilistic Gates**

Linear optical CNOT succeeds with probability:

$$p_{\text{success}} = \frac{1}{9} \text{ (basic KLM)}$$

With multiplexing and feed-forward: improved to ~1%

**2. Photon Loss**

In optical fiber: ~0.2 dB/km
In integrated circuits: ~0.1 dB/cm

For a 100-gate circuit: >99.5% transmission needed per component

**3. Photon Generation**

Single photon sources:
- Parametric down-conversion: probabilistic
- Quantum dots: deterministic but challenging to integrate

Rate-fidelity trade-off limits clock speed.

#### Scaling Strategy

**Fusion-Based Quantum Computing:**

PsiQuantum/Xanadu approach:
1. Create small entangled states (resource states)
2. Fuse states together
3. Error correct against failed fusions

$$\text{Threshold for fusion failures} \sim 10\%$$

This approach is fundamentally different from circuit model and designed for massive scale.

### 6. Comparative Scaling Analysis

#### Scaling Laws

| Platform | Qubit Scaling | Fidelity Scaling | Connectivity |
|----------|---------------|------------------|--------------|
| Superconducting | ~2× / 2 years | Improving slowly | Fixed (2D grid) |
| Trapped Ion | ~2× / 2 years | Stable high | All-to-all (with overhead) |
| Neutral Atom | ~3× / 2 years | Improving fast | Reconfigurable |
| Photonic | Architecture-dependent | Challenging | Graph states |

#### Quality vs Quantity Trade-off

Define a figure of merit:

$$\boxed{Q = N_{\text{qubits}} \times F_{\text{2Q}}^d \times C}$$

where:
- $N$ = qubit count
- $F$ = two-qubit gate fidelity
- $d$ = circuit depth (typical)
- $C$ = connectivity factor

**Platform Comparison (2025):**

| Platform | N | F | d=100 | C | Q |
|----------|---|---|-------|---|---|
| Google Willow | 105 | 99.7% | 0.74 | 0.5 | 39 |
| IBM Condor | 1121 | 99.5% | 0.61 | 0.3 | 205 |
| Quantinuum H2 | 56 | 99.9% | 0.90 | 1.0 | 50 |
| QuEra | 256 | 99.5% | 0.61 | 0.8 | 125 |

### 7. Projections to 2030

#### Optimistic Scenarios

| Platform | Qubits (2030) | Logical Qubits | Key Assumption |
|----------|---------------|----------------|----------------|
| Superconducting | 100,000 | 1,000 | Modular success |
| Trapped Ion | 10,000 | 500 | Photonic links work |
| Neutral Atom | 100,000 | 1,000 | Gate fidelity improves |
| Photonic | 1,000,000 | 100 | Fusion architecture works |

#### Conservative Scenarios

| Platform | Qubits (2030) | Logical Qubits | Limiting Factor |
|----------|---------------|----------------|-----------------|
| Superconducting | 10,000 | 100 | Wiring, crosstalk |
| Trapped Ion | 1,000 | 50 | Link fidelity |
| Neutral Atom | 10,000 | 100 | Gate fidelity |
| Photonic | 10,000 | 10 | Photon generation |

---

## Worked Examples

### Example 1: Wiring Density Analysis

**Problem:** Calculate the maximum qubit count for a superconducting processor given cryostat wiring constraints.

**Given:**
- Maximum wires through mixing chamber: 2000
- Lines per qubit: 4 (2 control, 1 readout, 1 flux)
- Shared readout: 8 qubits per readout line

**Solution:**

With shared readout:
$$\text{Lines per qubit} = 3 + \frac{1}{8} = 3.125$$

Maximum qubits:
$$N_{\text{max}} = \frac{2000}{3.125} = 640 \text{ qubits}$$

**With multiplexed control (future):**
If control lines can be multiplexed 4:1:
$$\text{Lines per qubit} = \frac{2}{4} + \frac{1}{8} + 1 = 1.625$$

$$N_{\text{max}} = \frac{2000}{1.625} = 1230 \text{ qubits}$$

**Conclusion:** Current wiring limits single-cryostat systems to ~1000 qubits without advanced multiplexing.

### Example 2: Ion Shuttling Overhead

**Problem:** Calculate the computational overhead of ion shuttling in a QCCD architecture.

**Given:**
- Trap zones: 10
- Qubits per zone: 10 (100 total)
- Gate time: 100 μs
- Shuttling time: 50 μs (between adjacent zones)
- Average gates requiring shuttling: 30%

**Solution:**

For a circuit with $G$ gates:
- Gates requiring shuttling: $0.3G$
- Average shuttling distance: 2.5 zones → 125 μs per shuffle

Total time:
$$T = G \times 100 \text{ μs} + 0.3G \times 125 \text{ μs} = G \times 137.5 \text{ μs}$$

Overhead:
$$\text{Overhead} = \frac{0.3 \times 125}{100} = 37.5\%$$

**For 1000 gates:**
- Without shuttling: 100 ms
- With shuttling: 137.5 ms

**Scaling consideration:** As system grows, average shuttling distance increases, potentially to $O(\sqrt{N})$ zones.

### Example 3: Neutral Atom Array Filling

**Problem:** Analyze the time required to prepare a defect-free 1000-atom array.

**Given:**
- Single-shot loading probability: 50%
- Detection time: 10 ms
- Rearrangement time: 20 ms per iteration
- Target: 99% filling

**Solution:**

Expected filled sites per load: $1000 \times 0.5 = 500$

Rearrangement can fill ~10% of empty sites per iteration:
$$N(k) = N_0 \times (1 - 0.5 \times 0.9^k)$$

Iterations to reach 990 atoms:
$$0.99 = 1 - 0.5 \times 0.9^k$$
$$0.9^k = 0.02$$
$$k = \frac{\ln(0.02)}{\ln(0.9)} = 37 \text{ iterations}$$

Total time:
$$T = 10 \text{ ms (detect)} + 37 \times 20 \text{ ms (rearrange)} = 750 \text{ ms}$$

**Improvement strategies:**
- Higher loading probability (60%): reduces to ~20 iterations
- Faster rearrangement: proportional time reduction
- Parallel rearrangement: significant speedup possible

---

## Practice Problems

### Problem 1: Cryostat Scaling (Direct Application)

A quantum computing company plans to scale from 100 to 1000 qubits using a modular approach with 4 cryostats.

a) How many qubits per cryostat?
b) If inter-cryostat links have 99% fidelity, what is the effective two-qubit gate fidelity for cross-cryostat operations?
c) What fraction of qubit pairs are in different cryostats?

### Problem 2: Platform Selection (Intermediate)

You are consulting for a company that needs to run quantum chemistry calculations requiring:
- 50 logical qubits
- Depth-1000 circuits
- Completion within 1 second

Analyze which platform (superconducting, trapped-ion, neutral atom) is best suited and why.

### Problem 3: Scaling Projection (Challenging)

Using the data provided in this lesson:
a) Fit exponential growth models to each platform's qubit scaling
b) Project when each platform will reach 10,000 physical qubits
c) Accounting for fidelity trends, project when each will achieve 100 logical qubits
d) Identify the key assumptions and uncertainties in your projections

---

## Computational Lab: Scaling Analysis Tools

```python
"""
Day 991 Lab: Hardware Scaling Analysis
Analyzing and projecting quantum hardware capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================
# 1. Historical Qubit Count Data
# ============================================================

# Years as fractional values from 2019
years = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025])

# Qubit counts by platform
google_qubits = np.array([53, 53, 72, 72, 72, 105, 120])
ibm_qubits = np.array([53, 65, 127, 433, 1121, 133, 200])  # Heron is smaller but better
ibm_qubits_peak = np.array([53, 65, 127, 433, 1121, 1121, 1200])
quantinuum_qubits = np.array([10, 10, 12, 20, 32, 56, 72])
neutral_atom_qubits = np.array([51, 100, 256, 256, 280, 300, 1000])
ionq_qubits = np.array([32, 32, 32, 32, 36, 36, 64])

# Gate fidelities (two-qubit)
google_fidelity = np.array([0.995, 0.996, 0.996, 0.996, 0.996, 0.997, 0.998])
ibm_fidelity = np.array([0.990, 0.992, 0.993, 0.994, 0.994, 0.995, 0.996])
quantinuum_fidelity = np.array([0.997, 0.998, 0.998, 0.999, 0.999, 0.999, 0.9995])
neutral_atom_fidelity = np.array([0.990, 0.992, 0.993, 0.994, 0.995, 0.995, 0.996])

# ============================================================
# Figure 1: Qubit Count Scaling
# ============================================================

fig1, ax1 = plt.subplots(figsize=(12, 7))

ax1.semilogy(years, google_qubits, 'o-', color='#4285F4',
             markersize=10, linewidth=2.5, label='Google')
ax1.semilogy(years, ibm_qubits_peak, 's-', color='#052D6E',
             markersize=10, linewidth=2.5, label='IBM (peak)')
ax1.semilogy(years, quantinuum_qubits, '^-', color='#00A4E4',
             markersize=10, linewidth=2.5, label='Quantinuum')
ax1.semilogy(years, neutral_atom_qubits, 'd-', color='#FF6B6B',
             markersize=10, linewidth=2.5, label='Neutral Atoms')

# Projections
future_years = np.array([2026, 2027, 2028, 2029, 2030])

# Fit exponential growth
def exp_growth(x, a, b):
    return a * np.exp(b * (x - 2019))

# Google projection
popt_google, _ = curve_fit(exp_growth, years[-4:], google_qubits[-4:], p0=[50, 0.2])
google_proj = exp_growth(future_years, *popt_google)

# IBM projection
popt_ibm, _ = curve_fit(exp_growth, years[-4:], ibm_qubits_peak[-4:], p0=[50, 0.5])
ibm_proj = exp_growth(future_years, *popt_ibm)

# Quantinuum projection
popt_quant, _ = curve_fit(exp_growth, years[-4:], quantinuum_qubits[-4:], p0=[10, 0.4])
quant_proj = exp_growth(future_years, *popt_quant)

# Neutral atom projection
popt_atom, _ = curve_fit(exp_growth, years[-4:], neutral_atom_qubits[-4:], p0=[50, 0.4])
atom_proj = exp_growth(future_years, *popt_atom)

# Plot projections
ax1.semilogy(future_years, google_proj, 'o--', color='#4285F4', alpha=0.5, markersize=8)
ax1.semilogy(future_years, ibm_proj, 's--', color='#052D6E', alpha=0.5, markersize=8)
ax1.semilogy(future_years, quant_proj, '^--', color='#00A4E4', alpha=0.5, markersize=8)
ax1.semilogy(future_years, atom_proj, 'd--', color='#FF6B6B', alpha=0.5, markersize=8)

ax1.axvline(x=2025, color='gray', linestyle=':', alpha=0.7)
ax1.text(2025.1, 20, 'Projections →', fontsize=10, color='gray')

# Reference lines
ax1.axhline(y=1000, color='green', linestyle='--', alpha=0.5, label='1,000 qubits')
ax1.axhline(y=10000, color='orange', linestyle='--', alpha=0.5, label='10,000 qubits')
ax1.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='100,000 qubits')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Physical Qubit Count (log scale)', fontsize=12)
ax1.set_title('Quantum Hardware Scaling: Physical Qubits', fontsize=14)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_ylim([5, 1e6])
ax1.set_xlim([2018.5, 2031])
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qubit_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 2: Gate Fidelity Trends
# ============================================================

fig2, ax2 = plt.subplots(figsize=(12, 6))

ax2.plot(years, (1-google_fidelity)*100, 'o-', color='#4285F4',
         markersize=10, linewidth=2.5, label='Google')
ax2.plot(years, (1-ibm_fidelity)*100, 's-', color='#052D6E',
         markersize=10, linewidth=2.5, label='IBM')
ax2.plot(years, (1-quantinuum_fidelity)*100, '^-', color='#00A4E4',
         markersize=10, linewidth=2.5, label='Quantinuum')
ax2.plot(years, (1-neutral_atom_fidelity)*100, 'd-', color='#FF6B6B',
         markersize=10, linewidth=2.5, label='Neutral Atoms')

# Threshold lines
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
            label='1% threshold')
ax2.axhline(y=0.1, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label='0.1% (high-fidelity)')
ax2.axhline(y=0.01, color='purple', linestyle='--', linewidth=2, alpha=0.7,
            label='0.01% (target)')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Two-Qubit Gate Error Rate (%)', fontsize=12)
ax2.set_title('Gate Fidelity Improvement by Platform', fontsize=14)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_yscale('log')
ax2.set_ylim([0.005, 2])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fidelity_trends.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 3: Quality-Quantity Trade-off
# ============================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))

# Calculate figure of merit: N * F^100 (100-depth circuit capability)
depth = 100

google_qm = google_qubits * google_fidelity**depth
ibm_qm = ibm_qubits * ibm_fidelity**depth
quantinuum_qm = quantinuum_qubits * quantinuum_fidelity**depth
neutral_qm = neutral_atom_qubits * neutral_atom_fidelity**depth

ax3.semilogy(years, google_qm, 'o-', color='#4285F4',
             markersize=10, linewidth=2.5, label='Google')
ax3.semilogy(years, ibm_qm, 's-', color='#052D6E',
             markersize=10, linewidth=2.5, label='IBM')
ax3.semilogy(years, quantinuum_qm, '^-', color='#00A4E4',
             markersize=10, linewidth=2.5, label='Quantinuum')
ax3.semilogy(years, neutral_qm, 'd-', color='#FF6B6B',
             markersize=10, linewidth=2.5, label='Neutral Atoms')

ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel(f'Quality Metric: N × F^{depth}', fontsize=12)
ax3.set_title(f'Quality-Adjusted Qubit Metric (depth={depth})', fontsize=14)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quality_metric.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 4: Scaling Bottleneck Analysis
# ============================================================

fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# Superconducting: Wiring constraint
ax1 = axes[0, 0]
qubits = np.arange(100, 5000, 100)
wires_basic = 4 * qubits
wires_multiplexed = 2 * qubits
cryostat_limit = 2000

ax1.plot(qubits, wires_basic, 'b-', linewidth=2, label='Basic wiring (4/qubit)')
ax1.plot(qubits, wires_multiplexed, 'g-', linewidth=2, label='Multiplexed (2/qubit)')
ax1.axhline(y=cryostat_limit, color='red', linestyle='--', linewidth=2,
            label='Cryostat limit (2000)')
ax1.fill_between(qubits, wires_basic, cryostat_limit,
                  where=wires_basic > cryostat_limit, alpha=0.3, color='red')
ax1.set_xlabel('Qubits', fontsize=11)
ax1.set_ylabel('Wire Count', fontsize=11)
ax1.set_title('Superconducting: Wiring Bottleneck', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Trapped Ion: Shuttling overhead
ax2 = axes[0, 1]
ion_count = np.arange(10, 500, 10)
zones = np.ceil(ion_count / 10)  # 10 ions per zone
avg_shuttle_dist = np.sqrt(zones) / 2
shuttle_time_per_gate = 50 * avg_shuttle_dist  # μs
gate_time = 100  # μs
overhead = (shuttle_time_per_gate * 0.3) / gate_time * 100

ax2.plot(ion_count, overhead, 'b-', linewidth=2)
ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% overhead limit')
ax2.fill_between(ion_count, overhead, 50, where=overhead > 50, alpha=0.3, color='red')
ax2.set_xlabel('Ion Count', fontsize=11)
ax2.set_ylabel('Shuttling Overhead (%)', fontsize=11)
ax2.set_title('Trapped Ion: Shuttling Bottleneck', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Neutral Atom: Filling time
ax3 = axes[1, 0]
atom_count = np.arange(100, 5000, 100)
load_prob = 0.5
iterations = np.log(0.01) / np.log(load_prob * 0.9)  # To reach 99%
fill_time = 10 + iterations * 20  # ms

ax3.plot(atom_count, fill_time, 'b-', linewidth=2)
ax3.axhline(y=1000, color='orange', linestyle='--', linewidth=2, label='1 second')
ax3.set_xlabel('Target Atom Count', fontsize=11)
ax3.set_ylabel('Array Filling Time (ms)', fontsize=11)
ax3.set_title('Neutral Atom: Array Filling Time', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Photonic: Photon generation rate
ax4 = axes[1, 1]
modes = np.arange(10, 1000, 10)
photon_rate = 1e6  # Hz per source
multiplex = 10
effective_rate = photon_rate * multiplex
circuit_time = modes * 10 / effective_rate * 1000  # ms

ax4.plot(modes, circuit_time, 'b-', linewidth=2)
ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='1 ms target')
ax4.set_xlabel('Optical Modes', fontsize=11)
ax4.set_ylabel('Circuit Time (ms)', fontsize=11)
ax4.set_title('Photonic: Photon Generation Rate', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_bottlenecks.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 5: Path to 100 Logical Qubits
# ============================================================

fig5, ax5 = plt.subplots(figsize=(12, 7))

# Requirements for logical qubits
# Assuming surface code with d=11 for 10^-8 logical error rate
# Physical qubits per logical: ~200 for d=11

physical_per_logical = 200

# Current logical capability (rough estimates)
current_logical = {
    'Google': 1,
    'IBM': 0,  # Error mitigation only
    'Quantinuum': 2,
    'Neutral Atoms': 6
}

# Projected timeline to 100 logical
# Need: 100 * 200 = 20,000 physical qubits with 99.9% gates

timeline = np.arange(2025, 2036)

# Projections (optimistic)
google_logical = 1 * 2**((timeline - 2025) / 2)  # Doubling every 2 years
quant_logical = 2 * 2**((timeline - 2025) / 1.5)  # Faster for trapped ions
atom_logical = 6 * 2**((timeline - 2025) / 1.5)

# Cap at realistic maximum
google_logical = np.minimum(google_logical, 500)
quant_logical = np.minimum(quant_logical, 500)
atom_logical = np.minimum(atom_logical, 500)

ax5.semilogy(timeline, google_logical, 'o-', color='#4285F4',
             markersize=8, linewidth=2, label='Google (projected)')
ax5.semilogy(timeline, quant_logical, '^-', color='#00A4E4',
             markersize=8, linewidth=2, label='Quantinuum (projected)')
ax5.semilogy(timeline, atom_logical, 'd-', color='#FF6B6B',
             markersize=8, linewidth=2, label='Neutral Atoms (projected)')

ax5.axhline(y=100, color='green', linestyle='--', linewidth=2,
            label='100 logical qubits (major milestone)')
ax5.axhline(y=1000, color='purple', linestyle='--', linewidth=2,
            label='1000 logical qubits (cryptographic)')

ax5.set_xlabel('Year', fontsize=12)
ax5.set_ylabel('Logical Qubit Count (projected)', fontsize=12)
ax5.set_title('Projected Path to Fault-Tolerant Quantum Computing', fontsize=14)
ax5.legend(loc='upper left', fontsize=10)
ax5.set_ylim([0.5, 2000])
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logical_qubit_projections.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 6: Platform Comparison Radar (2025)
# ============================================================

fig6, ax6 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

categories = ['Qubit Count', 'Gate Fidelity', 'Connectivity',
              'Gate Speed', 'Coherence', 'Scalability']
N = len(categories)

# Scores (0-10)
google_scores = [7, 8, 4, 10, 7, 8]
ibm_scores = [10, 6, 3, 9, 6, 7]
quantinuum_scores = [4, 10, 10, 2, 8, 6]
neutral_scores = [8, 6, 8, 6, 7, 9]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for scores, color, label in [
    (google_scores, '#4285F4', 'Google'),
    (ibm_scores, '#052D6E', 'IBM'),
    (quantinuum_scores, '#00A4E4', 'Quantinuum'),
    (neutral_scores, '#FF6B6B', 'Neutral Atoms')
]:
    scores = scores + scores[:1]
    ax6.plot(angles, scores, 'o-', linewidth=2, color=color, label=label)
    ax6.fill(angles, scores, alpha=0.1, color=color)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, fontsize=10)
ax6.set_ylim([0, 10])
ax6.set_title('Platform Capabilities Comparison (2025)', fontsize=14, pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

plt.tight_layout()
plt.savefig('platform_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary Statistics
# ============================================================

print("\n" + "="*60)
print("HARDWARE SCALING SUMMARY (2025)")
print("="*60)

print("\n--- Qubit Counts ---")
print(f"Google Willow: 120 qubits")
print(f"IBM: 1,200+ (peak), 200 (modular Flamingo)")
print(f"Quantinuum H2: 72 qubits")
print(f"Neutral Atoms: 1,000+ qubits")

print("\n--- Gate Fidelities ---")
print(f"Google: 99.7% two-qubit")
print(f"IBM: 99.5% two-qubit")
print(f"Quantinuum: 99.9% two-qubit")
print(f"Neutral Atoms: 99.5% two-qubit")

print("\n--- Projections to 100 Logical Qubits ---")
print(f"Google: ~2030 (optimistic)")
print(f"Quantinuum: ~2029 (optimistic)")
print(f"Neutral Atoms: ~2029 (optimistic)")
print(f"IBM: ~2033 (per roadmap)")

print("\n--- Key Bottlenecks ---")
print("Superconducting: Wiring density, crosstalk")
print("Trapped Ion: Shuttling overhead, photonic links")
print("Neutral Atom: Gate fidelity, atom loss")
print("Photonic: Photon generation, loss")

print("="*60)
```

---

## Summary

### Scaling Status (2025)

| Platform | Physical Qubits | 2Q Gate Fidelity | Key Strength |
|----------|-----------------|------------------|--------------|
| Google | 120 | 99.7% | Quality + error correction |
| IBM | 1,200 | 99.5% | Quantity + roadmap |
| Quantinuum | 72 | 99.9% | Fidelity + fault tolerance |
| Neutral Atoms | 1,000+ | 99.5% | Scalability + flexibility |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Quality metric | $$Q = N \times F^d$$ |
| Wiring limit | $$N_{\text{max}} = \frac{W_{\text{cryostat}}}{w_{\text{per qubit}}}$$ |
| Shuttling overhead | $$\tau_{\text{overhead}} = \tau_{\text{shuttle}} \times f_{\text{remote}} \times \sqrt{N_{\text{zones}}}$$ |

### Main Takeaways

1. **All platforms are scaling** - but with different trajectories
2. **Quality vs quantity trade-off** - More qubits often means lower fidelity
3. **Modular is the future** - All platforms moving toward modular architectures
4. **Bottlenecks are technical** - Each platform has specific engineering challenges
5. **100 logical qubits by ~2030** - Optimistic but plausible target

---

## Daily Checklist

- [ ] I can compare scaling trajectories across platforms
- [ ] I understand the key bottlenecks for each technology
- [ ] I can calculate wiring and overhead constraints
- [ ] I can evaluate modular vs monolithic approaches
- [ ] I can project future capabilities from current trends
- [ ] I ran the analysis code and can interpret results

---

## Preview: Day 992

Tomorrow we examine **Algorithmic & Software Advances** - new quantum algorithms, compiler optimizations, error mitigation techniques, and the software stack enabling practical quantum computing.

---

*"Scaling quantum computers is not just about adding more qubits - it requires simultaneous advances in control, connectivity, and error rates. The platform that solves these challenges together will lead the race to useful quantum computing."*
