# Day 990: Error Correction Demonstrations

## Month 36, Week 142, Day 3 | Research Frontiers

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Theory: Recent EC Experimental Results |
| Afternoon | 2.5 hrs | Critical Analysis: Comparing Implementations |
| Evening | 2 hrs | Lab: Error Correction Data Analysis |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Summarize** major error correction demonstrations from 2024-2026
2. **Compare** surface code implementations across platforms
3. **Analyze** LDPC and alternative code demonstrations
4. **Evaluate** real-time decoding implementations
5. **Distinguish** error suppression from error correction
6. **Assess** the path from demonstrations to practical fault tolerance

---

## Core Content

### 1. The State of Error Correction (2025)

#### From Theory to Reality

Quantum error correction has transitioned from theoretical proposals to experimental reality:

| Era | Period | Achievement |
|-----|--------|-------------|
| Theoretical | 1995-2005 | Code discovery, threshold theorems |
| Proof of concept | 2005-2015 | First encoded qubits, basic error detection |
| Small-scale | 2015-2022 | Logical operations, limited cycles |
| Below threshold | 2023-2025 | Exponential error suppression demonstrated |
| Early fault-tolerant | 2025+ | Repeated error correction, logical algorithms |

#### Key Metrics for Evaluation

**Physical Error Rate:**
$$p_{\text{phys}} = 1 - F_{\text{gate}}$$

**Logical Error Rate (per round):**
$$p_L = \text{Pr}[\text{logical error per syndrome cycle}]$$

**Error Suppression Factor:**
$$\boxed{\Lambda = \frac{p_L(d)}{p_L(d+2)}}$$

Below threshold requires $\Lambda > 2$.

### 2. Surface Code Demonstrations

#### Google's Willow Results (December 2024)

The most complete surface code demonstration to date:

**Experimental Configuration:**
- 105 transmon qubits
- Nearest-neighbor coupling on 2D grid
- Distance 3, 5, 7 surface codes

**Key Results:**

| Distance | Physical Qubits | Logical Error Rate | $\Lambda$ |
|----------|-----------------|-------------------|-----------|
| d = 3 | 17 | 3.0% per round | - |
| d = 5 | 49 | 1.4% per round | 2.14 |
| d = 7 | 97 | 0.7% per round | 2.0 |

**Syndrome Extraction Cycle:**
$$\tau_{\text{cycle}} = \tau_{\text{gates}} + \tau_{\text{measure}} + \tau_{\text{reset}} \approx 1 \text{ μs}$$

**Error Budget Analysis:**

| Error Source | Contribution |
|--------------|--------------|
| Two-qubit gates | 45% |
| Measurement | 25% |
| Leakage | 15% |
| Crosstalk | 10% |
| Other | 5% |

#### IBM's Surface Code Path

IBM has taken a different approach, prioritizing qubit count over code distance:

**Heron Processor (2024-2025):**
- 133 tunable-coupler transmons
- 99.5% two-qubit gate fidelity
- Heavy-hex lattice topology

**Surface Code Compatibility:**
The heavy-hex layout requires modifications:

$$\text{Effective distance} = d_{\text{hex}} < d_{\text{square}}$$

IBM's strategy: combine error mitigation with eventual error correction:

$$\langle O \rangle_{\text{corrected}} = \sum_k \alpha_k M_k \langle O \rangle_{\text{mitigated}}$$

#### Comparative Analysis: Google vs IBM

| Aspect | Google | IBM |
|--------|--------|-----|
| Gate fidelity | 99.7% | 99.5% |
| Qubit count | 105 | 1121 (Condor) |
| Code distance | d = 7 | Mitigation focus |
| Topology | Square grid | Heavy-hex |
| Logical error | $10^{-3}$ | N/A (mitigation) |
| Strategy | Depth-first (quality) | Breadth-first (scale) |

### 3. Trapped-Ion Error Correction

#### Quantinuum's Demonstrations

Trapped ions offer different trade-offs for error correction:

**H2 Processor Specifications (2025):**
- 56 qubits (ytterbium-171)
- 99.9% two-qubit gate fidelity
- All-to-all connectivity via ion shuttling
- Mid-circuit measurement and reset

**Color Code Implementation:**

The [[7,1,3]] Steane code on H2:

$$\boxed{|0_L\rangle = \frac{1}{\sqrt{8}}(|0000000\rangle + |1010101\rangle + |0110011\rangle + \cdots)}$$

**Key Results:**
- Logical error rate: 0.01% per round (10× below physical)
- Maintained through 50+ correction cycles
- Logical T-gate via magic state injection

**Repeated Error Correction:**

Quantinuum's landmark achievement: maintaining logical coherence through many rounds:

$$|\psi_L(t)\rangle = \prod_{k=1}^{N} \mathcal{R}_k |\psi_L(0)\rangle$$

where $\mathcal{R}_k$ is the $k$-th round of syndrome extraction and correction.

**Results:**
- No accumulated error over 50 rounds
- Logical fidelity > 99% maintained
- First demonstration of "indefinite" error correction

#### IonQ's Approach

IonQ focuses on error-mitigated rather than error-corrected computation:

**Current Status (2025):**
- 36 algorithmic qubits
- 99.5% two-qubit gates
- Focus on application-specific error mitigation
- Limited error correction demonstrations

### 4. Neutral Atom Error Correction

#### QuEra's Logical Qubits (2024-2025)

Neutral atoms have emerged as a competitive platform:

**System Specifications:**
- 256+ rubidium atoms
- Rydberg-mediated entanglement
- Reconfigurable geometry
- Parallel gate operations

**Error Correction Demonstrations:**

Using the [[48,6,8]] code (optimized for atom arrays):

| Metric | Demonstrated |
|--------|--------------|
| Logical qubits | 6 simultaneously |
| Code distance | d = 8 |
| Logical operations | Transversal CNOTs |
| Error rate | ~1% per round |

**Key Innovation: Transversal Gates**

Neutral atoms enable transversal operations not easily available in other platforms:

$$\text{CNOT}_L = \prod_{i=1}^{n} \text{CNOT}_{i,i+n}$$

Applied simultaneously to all physical qubits.

#### Atom Computing

Focus on coherence time as error correction enabler:

**Record Achievement (2024):**
- T2 coherence: 40+ seconds (nuclear spin qubits)
- Allows deep circuits before error correction needed
- Hybrid approach: long coherence + light error correction

### 5. LDPC and Alternative Codes

#### Beyond Surface Codes

Low-Density Parity-Check (LDPC) codes offer efficiency advantages:

**Surface Code Limitations:**
- Encoding rate: $k/n = O(1/d^2)$ (very inefficient)
- Large qubit overhead for high distance

**LDPC Codes Promise:**
- Encoding rate: $k/n = O(1)$ (constant)
- Fewer physical qubits per logical qubit
- More complex connectivity requirements

#### Recent LDPC Demonstrations

**IBM's Gross Code Experiment (2024):**

Using the [[144,12,12]] hypergraph product code:

| Property | Value |
|----------|-------|
| Physical qubits | 144 |
| Logical qubits | 12 |
| Code distance | 12 |
| Rate | 8.3% (12/144) |

Compare to surface code with d=12: would need 288 physical qubits for 1 logical qubit (rate 0.35%).

**Challenges:**
- Requires long-range connectivity
- Complex syndrome extraction
- Higher-weight stabilizers

#### Floquet Codes

Time-varying codes showing promise:

**Honeycomb Code (Microsoft/others):**

$$H(t) = \sum_{\text{edges } e \in E_t} J_e Z_i Z_j$$

Stabilizers vary periodically, enabling:
- Lower-weight measurements
- Natural implementation on specific hardware
- Potential for lower overhead

**Experimental Status (2025):**
- Proof-of-principle demonstrations
- Not yet competitive with surface codes
- Active theoretical development

### 6. Real-Time Decoding

#### The Decoding Bottleneck

Error correction requires fast classical processing:

$$\tau_{\text{decode}} < \tau_{\text{coherence}} / N_{\text{rounds}}$$

For a 1 μs cycle time and 1000 rounds before logical error:
$$\tau_{\text{decode}} < 1 \text{ μs}$$

#### Decoding Approaches

**1. Minimum Weight Perfect Matching (MWPM)**

Traditional decoder, but slow:
$$\text{Time} = O(n^3) \text{ for } n \text{ syndrome bits}$$

**Optimizations:**
- Sparse matching: $O(n \cdot \text{poly}(\log n))$
- Parallelization on FPGAs

**2. Union-Find Decoder**

Faster approximate decoder:
$$\text{Time} = O(n \cdot \alpha(n)) \approx O(n)$$

where $\alpha$ is the inverse Ackermann function.

**Trade-off:** Slightly higher logical error rate for much faster decoding.

**3. Neural Network Decoders**

Machine learning approach:
- Train on simulated error data
- Fast inference on specialized hardware
- Potential to outperform MWPM

**Current Results (2025):**
| Decoder | Speed | Accuracy | Hardware |
|---------|-------|----------|----------|
| MWPM | ~100 μs | Best | CPU |
| Union-Find | ~1 μs | Good | FPGA |
| Neural | ~0.1 μs | Variable | TPU/ASIC |

#### Real-Time Demonstrations

**Google's Real-Time Decoder (2024):**
- FPGA-based Union-Find
- 440 ns per round
- Integrated with Willow processor

**Quantinuum's Approach:**
- Longer cycle time (ms) allows CPU decoding
- Standard MWPM sufficient
- Focus on correction fidelity over speed

### 7. Error Suppression vs Error Correction

#### Important Distinction

Not all "error reduction" is error correction:

**Error Suppression:**
- Reduces errors through better control
- No syndrome measurement
- Examples: dynamical decoupling, optimized pulses
- Scales poorly with circuit depth

**Error Mitigation:**
- Classical post-processing
- Extrapolates to zero-noise limit
- Examples: ZNE, PEC, Clifford data regression
- Overhead scales exponentially

**Error Correction:**
- Active measurement and feedback
- Syndrome extraction and decoding
- Scales to arbitrary accuracy (below threshold)
- Overhead is polynomial in 1/ε

$$\boxed{\text{Only error correction enables fault-tolerant computation}}$$

#### Distinguishing in Experiments

| Feature | Suppression | Mitigation | Correction |
|---------|-------------|------------|------------|
| Syndrome measurement | No | No | Yes |
| Real-time feedback | No | No | Yes |
| Scaling with depth | Poor | Exponential cost | Polynomial cost |
| Overhead | Physical | Classical | Physical + classical |

---

## Worked Examples

### Example 1: Analyzing Surface Code Scaling Data

**Problem:** Given the following experimental data from a surface code experiment, determine if the system is operating below threshold and estimate the threshold error rate.

| Distance | Logical Error Rate |
|----------|-------------------|
| 3 | 5.2% |
| 5 | 2.1% |
| 7 | 1.0% |
| 9 | 0.52% |

**Solution:**

Calculate error suppression factors:
$$\Lambda_{3 \to 5} = \frac{5.2}{2.1} = 2.48$$
$$\Lambda_{5 \to 7} = \frac{2.1}{1.0} = 2.10$$
$$\Lambda_{7 \to 9} = \frac{1.0}{0.52} = 1.92$$

All $\Lambda > 1$, confirming below-threshold operation.

**Threshold Estimation:**

For surface codes below threshold:
$$p_L(d) \approx 0.1 \times (100 \times p/p_{\text{th}})^{(d+1)/2}$$

Taking the ratio:
$$\frac{p_L(d)}{p_L(d+2)} = (100 \times p/p_{\text{th}})^{-1} = \frac{p_{\text{th}}}{100p}$$

From $\Lambda \approx 2$:
$$\frac{p_{\text{th}}}{100p} \approx 2 \implies p \approx \frac{p_{\text{th}}}{200}$$

If $p_{\text{th}} \approx 1\%$ (depolarizing):
$$p \approx 0.005\% \text{ effective error rate}$$

This suggests physical operations are at approximately 0.3% error rate (accounting for syndrome extraction overhead).

### Example 2: Comparing Code Efficiency

**Problem:** Compare the resource efficiency of a [[49,1,7]] surface code versus a [[48,6,8]] LDPC code.

**Solution:**

**Surface Code [[49,1,7]]:**
- Physical qubits: n = 49
- Logical qubits: k = 1
- Distance: d = 7
- Rate: k/n = 1/49 = 2.04%

**LDPC Code [[48,6,8]]:**
- Physical qubits: n = 48
- Logical qubits: k = 6
- Distance: d = 8
- Rate: k/n = 6/48 = 12.5%

**Efficiency Comparison:**

Per-logical-qubit overhead:
- Surface: 49 physical per logical
- LDPC: 48/6 = 8 physical per logical

The LDPC code is **6× more efficient** in physical qubit usage while having higher distance.

**But consider trade-offs:**
1. LDPC requires long-range connectivity
2. LDPC has higher-weight stabilizers (harder to measure)
3. Surface code has simpler layout

**Effective comparison:**
$$\eta = \frac{d^2}{n/k}$$

- Surface: $\eta_S = 7^2 / 49 = 1.0$
- LDPC: $\eta_L = 8^2 / 8 = 8.0$

By this metric, the LDPC code is 8× better.

### Example 3: Real-Time Decoding Requirements

**Problem:** A surface code with d = 11 requires real-time decoding. Calculate the decoding time requirement and evaluate different decoder options.

**Solution:**

**System Parameters:**
- Distance: d = 11
- Syndrome bits per round: $n_s = 2(d^2 - 1)/2 = d^2 - 1 = 120$
- Cycle time: $\tau_{\text{cycle}} = 1$ μs
- Target logical error rate: $p_L = 10^{-6}$

**Decoding Time Budget:**

For fault tolerance, decoding must complete before the next syndrome is needed:
$$\tau_{\text{decode}} < \tau_{\text{cycle}} = 1 \text{ μs}$$

**Decoder Analysis:**

1. **MWPM (Blossom algorithm):**
   - Complexity: $O(n_s^3) = O(120^3) \approx 1.7 \times 10^6$
   - At 1 GHz: ~2 ms (too slow)

2. **Sparse MWPM:**
   - Complexity: $O(n_s \cdot d^2) = O(120 \times 121) \approx 15,000$
   - At 1 GHz: ~15 μs (marginal)

3. **Union-Find:**
   - Complexity: $O(n_s) = O(120)$
   - At 1 GHz with FPGA optimization: ~200 ns (acceptable)

4. **Neural Network:**
   - Inference: ~100 operations after training
   - On TPU: ~50 ns (excellent)

**Recommendation:** Union-Find on FPGA or neural decoder for this scale.

---

## Practice Problems

### Problem 1: Error Rate Analysis (Direct Application)

A color code experiment reports the following:
- Physical two-qubit gate error: 0.1%
- Measurement error: 0.2%
- Logical error rate for [[7,1,3]] code: 0.8% per round

a) Is this consistent with below-threshold operation?
b) What is the error suppression factor compared to physical error?
c) Estimate the logical error rate for a [[17,1,5]] code.

### Problem 2: Decoder Comparison (Intermediate)

You are designing a real-time decoding system for a d = 15 surface code.

a) Calculate the number of syndrome bits per round
b) Estimate the time required for MWPM, Union-Find, and neural decoders
c) If the cycle time is 500 ns, which decoders are viable?
d) What is the trade-off between accuracy and speed?

### Problem 3: Code Selection (Challenging)

A quantum computer manufacturer must choose between:
- Option A: Surface code with d = 9, native square grid connectivity
- Option B: LDPC code with [[144,12,12]], requiring degree-6 connectivity

For an application requiring 50 logical qubits with $p_L < 10^{-8}$:
a) How many physical qubits does each option require?
b) What are the connectivity requirements?
c) Analyze the trade-offs and recommend an approach

---

## Computational Lab: Error Correction Analysis

```python
"""
Day 990 Lab: Error Correction Data Analysis
Analyzing experimental error correction results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================
# 1. Surface Code Scaling Analysis
# ============================================================

def logical_error_model(d, p_eff, p_th=0.01, A=0.1):
    """
    Theoretical model for logical error rate.
    p_L = A * (p_eff / p_th)^((d+1)/2)
    """
    return A * (p_eff / p_th) ** ((d + 1) / 2)

def fit_threshold(distances, logical_errors):
    """
    Fit experimental data to extract effective error rate and threshold.
    """
    def model(d, p_eff, A):
        return A * (100 * p_eff) ** ((d + 1) / 2)

    popt, pcov = curve_fit(model, distances, logical_errors,
                           p0=[0.005, 0.1], bounds=([0, 0], [0.1, 1]))
    return popt[0], popt[1], pcov

# Experimental data (simulated based on real results)
google_data = {
    'distances': np.array([3, 5, 7]),
    'logical_errors': np.array([0.030, 0.014, 0.007]),
    'label': 'Google Willow (2024)'
}

quantinuum_data = {
    'distances': np.array([3, 5]),
    'logical_errors': np.array([0.008, 0.002]),
    'label': 'Quantinuum H2 (2025)'
}

neutral_atom_data = {
    'distances': np.array([3, 5, 7]),
    'logical_errors': np.array([0.05, 0.025, 0.015]),
    'label': 'Neutral Atoms (2025)'
}

# Figure 1: Error scaling comparison
fig1, ax1 = plt.subplots(figsize=(10, 7))

datasets = [google_data, quantinuum_data, neutral_atom_data]
colors = ['#4285F4', '#00A4E4', '#FF6B6B']
markers = ['o', '^', 's']

for data, color, marker in zip(datasets, colors, markers):
    d = data['distances']
    p_L = data['logical_errors']

    ax1.semilogy(d, p_L, f'{marker}-', color=color, markersize=12,
                 linewidth=2.5, label=data['label'])

    # Fit and extrapolate
    if len(d) >= 2:
        z = np.polyfit(d, np.log10(p_L), 1)
        d_ext = np.array([3, 5, 7, 9, 11, 13])
        p_ext = 10**(z[0] * d_ext + z[1])
        ax1.semilogy(d_ext, p_ext, '--', color=color, alpha=0.4, linewidth=1.5)

# Reference lines
ax1.axhline(y=1e-4, color='green', linestyle=':', linewidth=2,
            label='Near-term target (10^-4)')
ax1.axhline(y=1e-10, color='purple', linestyle=':', linewidth=2,
            label='Fault-tolerant target (10^-10)')

ax1.set_xlabel('Code Distance d', fontsize=12)
ax1.set_ylabel('Logical Error Rate per Round', fontsize=12)
ax1.set_title('Error Correction Scaling Across Platforms', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xticks([3, 5, 7, 9, 11, 13])
ax1.set_ylim([1e-12, 0.1])
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ec_scaling_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 2. Error Suppression Factor Analysis
# ============================================================

def calculate_lambda(distances, logical_errors):
    """Calculate error suppression factors between consecutive distances."""
    lambdas = []
    for i in range(len(distances) - 1):
        if distances[i+1] - distances[i] == 2:  # Consecutive odd distances
            lam = logical_errors[i] / logical_errors[i+1]
            lambdas.append(lam)
    return np.array(lambdas)

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Calculate Lambda for each dataset
for data, color in zip(datasets, colors):
    if len(data['distances']) >= 2:
        lambdas = calculate_lambda(data['distances'], data['logical_errors'])
        mid_distances = (data['distances'][:-1] + data['distances'][1:]) / 2
        ax2.bar(mid_distances + (colors.index(color) - 1) * 0.3, lambdas,
                width=0.25, color=color, label=data['label'], alpha=0.8)

ax2.axhline(y=2, color='green', linestyle='--', linewidth=2,
            label='Λ = 2 (below threshold)')
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2,
            label='Λ = 1 (at threshold)')

ax2.set_xlabel('Mid-Distance (between d and d+2)', fontsize=12)
ax2.set_ylabel('Error Suppression Factor Λ', fontsize=12)
ax2.set_title('Error Suppression Factor by Platform', fontsize=14)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim([0, 5])
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lambda_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 3. Code Efficiency Comparison
# ============================================================

# Define various codes
codes = {
    'Surface d=3': {'n': 17, 'k': 1, 'd': 3},
    'Surface d=5': {'n': 49, 'k': 1, 'd': 5},
    'Surface d=7': {'n': 97, 'k': 1, 'd': 7},
    'Surface d=9': {'n': 161, 'k': 1, 'd': 9},
    'Steane [[7,1,3]]': {'n': 7, 'k': 1, 'd': 3},
    'Color [[17,1,5]]': {'n': 17, 'k': 1, 'd': 5},
    'LDPC [[48,6,8]]': {'n': 48, 'k': 6, 'd': 8},
    'LDPC [[144,12,12]]': {'n': 144, 'k': 12, 'd': 12},
}

fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Physical qubits per logical qubit
ax1 = axes[0]
code_names = list(codes.keys())
overhead = [codes[c]['n'] / codes[c]['k'] for c in code_names]
distances = [codes[c]['d'] for c in code_names]

colors_codes = plt.cm.viridis(np.linspace(0, 1, len(codes)))
bars = ax1.barh(code_names, overhead, color=colors_codes)

ax1.set_xlabel('Physical Qubits per Logical Qubit', fontsize=12)
ax1.set_title('Encoding Overhead Comparison', fontsize=13)
ax1.grid(True, alpha=0.3, axis='x')

# Annotate with distance
for bar, d in zip(bars, distances):
    ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             f'd={d}', va='center', fontsize=10)

# Plot 2: Efficiency metric (d^2 / overhead)
ax2 = axes[1]
efficiency = [codes[c]['d']**2 / (codes[c]['n'] / codes[c]['k']) for c in code_names]

bars2 = ax2.barh(code_names, efficiency, color=colors_codes)
ax2.set_xlabel('Efficiency Metric (d² / overhead)', fontsize=12)
ax2.set_title('Code Efficiency Comparison', fontsize=13)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('code_efficiency.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 4. Real-Time Decoding Requirements
# ============================================================

def decoding_time(n_syndrome, decoder_type):
    """
    Estimate decoding time in nanoseconds.
    """
    if decoder_type == 'MWPM':
        # O(n^3) complexity, ~1 ns per operation on modern hardware
        return n_syndrome**3 * 1
    elif decoder_type == 'Sparse MWPM':
        # O(n * d^2) with d ~ sqrt(n)
        return n_syndrome * np.sqrt(n_syndrome) * 10
    elif decoder_type == 'Union-Find':
        # O(n * alpha(n)) ~ O(n)
        return n_syndrome * 5
    elif decoder_type == 'Neural':
        # Constant time inference after training
        return 50 + n_syndrome * 0.1
    else:
        raise ValueError(f"Unknown decoder: {decoder_type}")

fig4, ax4 = plt.subplots(figsize=(10, 6))

distances = np.arange(3, 21, 2)
n_syndrome = (distances**2 - 1)  # Approximate syndrome bits

decoders = ['MWPM', 'Sparse MWPM', 'Union-Find', 'Neural']
styles = ['-', '--', '-.', ':']
colors_dec = ['red', 'orange', 'blue', 'green']

for decoder, style, color in zip(decoders, styles, colors_dec):
    times = [decoding_time(n, decoder) for n in n_syndrome]
    ax4.semilogy(distances, times, style, color=color, linewidth=2.5,
                 label=decoder, markersize=8)

# Typical cycle times
ax4.axhline(y=1000, color='gray', linestyle=':', linewidth=2,
            label='1 μs cycle (superconducting)')
ax4.axhline(y=1000000, color='purple', linestyle=':', linewidth=2,
            label='1 ms cycle (trapped ion)')

ax4.fill_between(distances, 0, 1000, alpha=0.1, color='green',
                 label='Viable for superconducting')

ax4.set_xlabel('Code Distance d', fontsize=12)
ax4.set_ylabel('Decoding Time (ns)', fontsize=12)
ax4.set_title('Real-Time Decoding Requirements', fontsize=14)
ax4.legend(loc='upper left', fontsize=10)
ax4.set_ylim([10, 1e10])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decoding_requirements.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 5. Error Budget Visualization
# ============================================================

fig5, axes = plt.subplots(1, 3, figsize=(15, 5))

# Error budgets for different platforms
platforms = ['Google Willow', 'Quantinuum H2', 'Neutral Atoms']
error_sources = ['Two-qubit gates', 'Measurement', 'Leakage',
                 'Crosstalk', 'State prep', 'Other']

# Data (percentages of total error budget)
google_budget = [45, 25, 15, 10, 3, 2]
quantinuum_budget = [35, 30, 5, 5, 20, 5]
neutral_budget = [50, 20, 10, 5, 10, 5]
budgets = [google_budget, quantinuum_budget, neutral_budget]

colors_budget = plt.cm.Set3(np.linspace(0, 1, len(error_sources)))

for ax, platform, budget in zip(axes, platforms, budgets):
    wedges, texts, autotexts = ax.pie(budget, labels=error_sources,
                                       autopct='%1.0f%%', colors=colors_budget,
                                       startangle=90)
    ax.set_title(platform, fontsize=12)

plt.suptitle('Error Budget Breakdown by Platform', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('error_budgets.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. Logical vs Physical Lifetime
# ============================================================

fig6, ax6 = plt.subplots(figsize=(10, 6))

# Simulated data: coherence times
time_points = np.linspace(0, 1, 100)  # Time in ms

# Physical decay (exponential with T1 = 0.1 ms)
T1_physical = 0.1
physical_fidelity = np.exp(-time_points / T1_physical)

# Logical decay (with error correction, T1 = 0.4 ms)
T1_logical_d5 = 0.25
logical_fidelity_d5 = np.exp(-time_points / T1_logical_d5)

T1_logical_d7 = 0.4
logical_fidelity_d7 = np.exp(-time_points / T1_logical_d7)

ax6.plot(time_points * 1000, physical_fidelity, 'r-', linewidth=2.5,
         label=f'Physical (T₁ = {T1_physical*1000:.0f} μs)')
ax6.plot(time_points * 1000, logical_fidelity_d5, 'b-', linewidth=2.5,
         label=f'Logical d=5 (T₁ = {T1_logical_d5*1000:.0f} μs)')
ax6.plot(time_points * 1000, logical_fidelity_d7, 'g-', linewidth=2.5,
         label=f'Logical d=7 (T₁ = {T1_logical_d7*1000:.0f} μs)')

ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
ax6.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.7, label='1/e threshold')

ax6.set_xlabel('Time (μs)', fontsize=12)
ax6.set_ylabel('Fidelity', fontsize=12)
ax6.set_title('Logical Qubit Lifetime Extension', fontsize=14)
ax6.legend(loc='upper right', fontsize=10)
ax6.set_xlim([0, 1000])
ax6.set_ylim([0, 1])
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lifetime_extension.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary Statistics
# ============================================================

print("\n" + "="*60)
print("ERROR CORRECTION DEMONSTRATIONS SUMMARY (2024-2026)")
print("="*60)

print("\n--- Surface Code Results ---")
print("Google Willow: d=7, pL=0.7%, Λ=2.0")
print("Below threshold: YES")
print("Logical lifetime: 4× physical")

print("\n--- Trapped Ion Results ---")
print("Quantinuum H2: d=5 (color), pL=0.2%")
print("Below threshold: YES")
print("Repeated EC cycles: 50+")

print("\n--- Neutral Atom Results ---")
print("QuEra: d=8 (LDPC), 6 logical qubits")
print("Below threshold: Marginal")
print("Key advantage: Transversal gates")

print("\n--- Decoder Status ---")
print("MWPM: Accurate but slow (ms for d>10)")
print("Union-Find: Fast (μs), slightly lower accuracy")
print("Neural: Fastest (ns), requires training")

print("\n" + "="*60)
```

---

## Summary

### Key Demonstrations (2024-2026)

| Platform | Code | Distance | Logical Error | Status |
|----------|------|----------|---------------|--------|
| Google Willow | Surface | d=7 | 0.7% | Below threshold |
| Quantinuum H2 | Color | d=5 | 0.2% | Repeated EC |
| Neutral Atoms | LDPC | d=8 | 1% | Multi-qubit logical |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Error suppression | $$\Lambda = p_L(d)/p_L(d+2)$$ |
| Below threshold | $$\Lambda > 2$$ |
| Code efficiency | $$\eta = d^2 / (n/k)$$ |
| Decoding time | $$\tau_{\text{decode}} < \tau_{\text{cycle}}$$ |

### Main Takeaways

1. **Multiple platforms below threshold** - Surface codes, color codes, and LDPC all demonstrated
2. **Repeated EC is possible** - Quantinuum showed 50+ rounds without error accumulation
3. **Trade-offs persist** - Speed vs fidelity, simplicity vs efficiency
4. **Decoding is crucial** - Real-time decoding enables practical fault tolerance
5. **Error suppression ≠ correction** - Only correction scales to useful computation

---

## Daily Checklist

- [ ] I can summarize major EC demonstrations from 2024-2026
- [ ] I can calculate error suppression factors from data
- [ ] I can compare code efficiency metrics
- [ ] I understand real-time decoding requirements
- [ ] I can distinguish error suppression from correction
- [ ] I can analyze error budgets for different platforms
- [ ] I ran the analysis code and can interpret the results

---

## Preview: Day 991

Tomorrow we examine **Hardware Scaling Progress** - analyzing how different quantum computing platforms are scaling in qubit count, connectivity, and overall system performance. We'll compare superconducting, trapped-ion, neutral atom, and photonic approaches.

---

*"The transition from demonstrating error correction to deploying it requires not just lower errors, but faster decoders, better calibration, and more qubits. The next few years will reveal which architectures can scale while maintaining quality."*
