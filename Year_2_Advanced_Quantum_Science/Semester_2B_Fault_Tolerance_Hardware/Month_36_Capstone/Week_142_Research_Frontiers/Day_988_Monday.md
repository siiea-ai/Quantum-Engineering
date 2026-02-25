# Day 988: Logical Qubit Milestones (Google, IBM, Quantinuum)

## Month 36, Week 142, Day 1 | Research Frontiers

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Theory: Logical Qubit Achievements 2024-2026 |
| Afternoon | 2.5 hrs | Critical Analysis: Evaluating Claims |
| Evening | 2 hrs | Lab: Visualizing Progress Metrics |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Summarize** Google's Willow processor achievements and their significance
2. **Analyze** IBM's trajectory from utility to fault-tolerant computing
3. **Evaluate** Quantinuum's trapped-ion logical qubit demonstrations
4. **Compare** different platforms' approaches to logical qubit milestones
5. **Critically assess** claims about error suppression below threshold
6. **Identify** key metrics for evaluating logical qubit quality

---

## Core Content

### 1. The Logical Qubit Era Begins

The period 2024-2026 marks a fundamental transition in quantum computing: from the NISQ era (Noisy Intermediate-Scale Quantum) to the early fault-tolerant era. For the first time, multiple groups have demonstrated logical qubits that outperform their physical constituents.

#### Defining "Logical Qubit Success"

A logical qubit milestone requires demonstrating:

$$\boxed{\tau_{\text{logical}} > \tau_{\text{physical}}}$$

where $\tau$ represents coherence time or effective error rate. This means the encoded, error-corrected qubit performs better than any single physical qubit in the system.

More stringent criteria include:
- Error suppression with increasing code distance
- Logical error rate below physical error rate threshold
- Preservation of quantum information through error correction cycles

### 2. Google Quantum AI: Willow Processor

#### The December 2024 Announcement

Google's Willow processor represents a landmark in superconducting qubit development:

| Specification | Willow (2024) | Sycamore (2019) |
|---------------|---------------|-----------------|
| Physical qubits | 105 | 53 |
| Two-qubit gate fidelity | 99.7% | 99.5% |
| Readout fidelity | 99.7% | 97% |
| T1 coherence | 70 μs (avg) | 20 μs |
| Surface code distance | 3, 5, 7 | N/A |

#### Key Achievements

**1. Error Suppression Below Threshold**

Google demonstrated that increasing the surface code distance reduces logical errors:

$$\Lambda = \frac{p_L(d)}{p_L(d+2)} > 2$$

where $p_L(d)$ is the logical error rate at distance $d$. They achieved:
- $\Lambda_{3 \to 5} \approx 2.1$
- $\Lambda_{5 \to 7} \approx 2.0$

This exponential suppression indicates operation below the error threshold.

**2. Logical Qubit Lifetime Extension**

The logical qubit coherence time exceeded the best physical qubit:

$$T_1^{\text{logical}} \approx 280 \text{ μs} > T_1^{\text{physical, best}} \approx 70 \text{ μs}$$

This factor of 4× improvement demonstrates genuine error correction benefit.

**3. Random Circuit Sampling Speedup**

The headline claim: completing a random circuit sampling task in under 5 minutes that would take classical supercomputers an estimated $10^{25}$ years.

**Critical Analysis of Google's Claims**

| Claim | Evidence | Caveats |
|-------|----------|---------|
| Below threshold | Scaling data | Limited to d=7, statistical uncertainties |
| $10^{25}$ year speedup | Complexity estimates | Classical algorithms improve; unverifiable |
| Logical > Physical | Lifetime measurements | Specific encoding, not universal computation |

### 3. IBM Quantum: The Utility Era

#### Condor and Heron Architectures

IBM's 2024-2025 strategy diverges from pure error correction:

**Condor Processor (2023-2024)**
- 1,121 physical qubits
- Cross-resonance gates, fixed-frequency transmons
- Focus on "utility-scale" experiments rather than fault tolerance
- Demonstrated quantum advantage claims in condensed matter simulation

**Heron Processor (2024-2025)**
- 133 qubits with improved connectivity
- Tunable couplers for 99.5% two-qubit gates
- Lower crosstalk for error mitigation
- Foundation for modular Flamingo architecture

#### IBM's Roadmap to Fault Tolerance

IBM's explicit roadmap targets:

| Year | Milestone | Approach |
|------|-----------|----------|
| 2024 | Heron 133 qubits | Error mitigation at scale |
| 2025 | Flamingo modular | Multi-chip connectivity |
| 2026 | 10,000 qubit system | Distributed error correction |
| 2029 | 100,000+ qubits | Fault-tolerant universal gates |
| 2033 | 100×100 problem | 100 logical qubits, depth 100 gates |

#### Circuit Knitting and Error Mitigation

IBM's distinctive contribution: sophisticated classical-quantum hybrid techniques

$$\langle O \rangle_{\text{mitigated}} = \sum_i c_i \langle O \rangle_i$$

where different circuits $i$ are combined classically to reduce errors. This "circuit knitting" approach extends utility without full fault tolerance.

**Critical Analysis of IBM's Approach**

| Strength | Limitation |
|----------|------------|
| Clear multi-year roadmap | Error mitigation doesn't scale |
| Industrial partnerships | Lower gate fidelities than competitors |
| Software ecosystem (Qiskit) | Condor connectivity challenges |
| Modular architecture plans | Modular coupling adds overhead |

### 4. Quantinuum: Trapped-Ion Precision

#### H2 Processor Achievements

Quantinuum's trapped-ion approach delivers the highest gate fidelities:

| Metric | H2 (2025) | Industry Best Transmon |
|--------|-----------|------------------------|
| Two-qubit gate fidelity | 99.9% | 99.7% |
| Single-qubit fidelity | 99.997% | 99.95% |
| State prep/measure | 99.9% | 99.7% |
| Physical qubits | 56 | 105 (Google) |
| Connectivity | All-to-all | Nearest-neighbor |

#### Fault-Tolerant Demonstrations

Quantinuum's key logical qubit achievements:

**1. Repeated Error Correction Cycles**

First demonstration of maintaining logical qubit through many rounds of active error correction without accumulated errors:

$$p_L^{(n)} < n \cdot p_L^{(1)}$$

where $n$ is the number of syndrome extraction cycles.

**2. Hardware-Efficient Encodings**

Development of color codes and other encodings optimized for all-to-all connectivity:

$$d_{\text{color}} = 3 \text{ using only } 7 \text{ physical qubits}$$

compared to surface code requiring 17 physical qubits for $d=3$.

**3. Teleported CNOT Gates**

Demonstration of logical two-qubit gates via lattice surgery and teleportation:

$$\text{CNOT}_L = \text{MXX} \cdot \text{MZZ} \cdot \text{Corrections}$$

**Critical Analysis of Quantinuum's Approach**

| Strength | Limitation |
|----------|------------|
| Highest fidelity operations | Slower gate times (ms vs μs) |
| All-to-all connectivity | Limited qubit count |
| Demonstrated fault tolerance | Scaling requires ion shuttling |
| Strong theoretical foundation | Complex trap engineering needed |

### 5. Comparative Analysis Framework

#### Logical Qubit Quality Metrics

Define a logical qubit figure of merit:

$$\boxed{Q_L = \frac{d}{\sqrt{n_{\text{phys}}}} \cdot \frac{p_{\text{phys}}}{p_L} \cdot \frac{t_{\text{gate,phys}}}{t_{\text{gate,L}}}}$$

This captures:
- Code efficiency: distance per physical qubit
- Error suppression: physical to logical error reduction
- Speed penalty: logical gate overhead

#### Platform Comparison (2025 Data)

| Platform | Physical Qubits | Logical Qubits | Best $p_L$ | Speed |
|----------|-----------------|----------------|------------|-------|
| Google Willow | 105 | 1 (d=7) | $10^{-3}$ | μs gates |
| IBM Condor | 1121 | 0 (mitigation) | N/A | μs gates |
| Quantinuum H2 | 56 | 1 (d=3) | $10^{-4}$ | ms gates |
| Neutral atom | 256 | 4+ (d=3) | $10^{-2}$ | μs gates |

### 6. Trends and Projections

#### The Below-Threshold Transition

The critical milestone of 2024-2025: crossing the threshold where error correction provides net benefit.

$$\text{Physical error rate } p < p_{\text{th}} \implies \text{Error correction helps}$$

Observed threshold crossings:
- Google: $p \approx 0.3\%$ vs $p_{\text{th}} \approx 1\%$ for surface code
- Quantinuum: $p \approx 0.1\%$ vs $p_{\text{th}} \approx 0.6\%$ for color code

#### Projected Trajectories

Extrapolating current progress:

| Year | Google (projected) | IBM (projected) | Quantinuum (projected) |
|------|-------------------|-----------------|------------------------|
| 2025 | d=7, $p_L = 10^{-3}$ | Error mitigation | d=5, $p_L = 10^{-4}$ |
| 2027 | d=11, $p_L = 10^{-5}$ | First logical qubits | d=7, $p_L = 10^{-6}$ |
| 2030 | d=17, $p_L = 10^{-8}$ | 100 logical qubits | d=11, $p_L = 10^{-8}$ |

---

## Worked Examples

### Example 1: Analyzing Error Suppression Data

**Problem:** Google reports logical error rates of $p_L(d=3) = 3.0\%$, $p_L(d=5) = 1.4\%$, and $p_L(d=7) = 0.7\%$. Analyze whether this demonstrates operation below threshold.

**Solution:**

Calculate the error suppression factor:

$$\Lambda_{3 \to 5} = \frac{p_L(d=3)}{p_L(d=5)} = \frac{0.030}{0.014} = 2.14$$

$$\Lambda_{5 \to 7} = \frac{p_L(d=5)}{p_L(d=7)} = \frac{0.014}{0.007} = 2.0$$

For below-threshold operation, we expect:

$$p_L(d) \propto \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

The suppression factor should be:

$$\Lambda = \left(\frac{p_{\text{th}}}{p}\right) > 2$$

Since $\Lambda > 2$ for both transitions, this is consistent with $p/p_{\text{th}} < 0.5$, indicating solid below-threshold operation.

**Critical observation:** The suppression factor is not increasing with distance, suggesting we may be approaching a floor set by correlated errors or leakage.

### Example 2: Comparing Platform Efficiency

**Problem:** Compare the physical-to-logical overhead for Google's surface code vs. Quantinuum's color code approach.

**Solution:**

**Google Surface Code (d=7):**
- Physical qubits used: $n = d^2 + (d-1)^2 = 49 + 36 = 85$ (approximate)
- Actually uses ~100 qubits including ancillas
- Achieved $p_L \approx 10^{-3}$
- Gate time: ~50 ns physical, ~1 μs logical cycle

**Quantinuum Color Code (d=3):**
- Physical qubits: 7 data + 3 ancilla = 10 qubits
- Achieved $p_L \approx 10^{-4}$
- Gate time: ~1 ms physical, ~10 ms logical cycle

**Efficiency comparison:**

Per-qubit logical error rate (normalized):
$$\eta_{\text{Google}} = \frac{p_L}{n \cdot t_{\text{cycle}}} = \frac{10^{-3}}{100 \cdot 10^{-6}} = 10^{4} \text{ s}^{-1}$$

$$\eta_{\text{Quantinuum}} = \frac{p_L}{n \cdot t_{\text{cycle}}} = \frac{10^{-4}}{10 \cdot 10^{-2}} = 10^{-1} \text{ s}^{-1}$$

**Interpretation:** Quantinuum achieves lower logical error rate per qubit-second by 5 orders of magnitude, but the absolute speed favors superconducting for algorithms requiring many gates.

### Example 3: Projecting to Useful Fault Tolerance

**Problem:** Estimate when each platform might achieve $p_L = 10^{-10}$ (needed for useful quantum algorithms like Shor's algorithm on cryptographic keys).

**Solution:**

Assuming continued exponential improvement in physical error rates and code distance scaling:

**Model:** $p_L(t) = p_L(0) \cdot 10^{-\alpha t}$ where $t$ is years and $\alpha$ is improvement rate.

From historical data:
- Google: $\alpha \approx 1$ (order of magnitude per year recently)
- IBM: $\alpha \approx 0.5$ (slower improvement but larger systems)
- Quantinuum: $\alpha \approx 1.5$ (rapid improvement from higher starting fidelity)

Time to $p_L = 10^{-10}$:

**Google:** From $10^{-3}$ → $10^{-10}$ requires 7 orders of magnitude
$$t_{\text{Google}} = 7/\alpha = 7 \text{ years} \implies 2032$$

**Quantinuum:** From $10^{-4}$ → $10^{-10}$ requires 6 orders of magnitude
$$t_{\text{Quantinuum}} = 6/1.5 = 4 \text{ years} \implies 2029$$

**Caveat:** These projections assume sustained exponential improvement, which historically slows as systems mature.

---

## Practice Problems

### Problem 1: Literature Analysis (Direct Application)

Read Google's Willow paper (Nature, 2024) and answer:
a) What is the dominant error mechanism limiting $\Lambda$?
b) How does leakage affect the surface code performance?
c) What improvements do they project for next-generation chips?

### Problem 2: Threshold Calculation (Intermediate)

Given that the surface code threshold is approximately $p_{\text{th}} = 1.0\%$ for depolarizing noise:
a) If physical error rate is $p = 0.3\%$, what logical error rate is expected at $d = 7$?
b) How many physical qubits are needed for $p_L = 10^{-6}$?
c) What is the break-even code distance where logical and physical error rates match?

**Hint:** Use $p_L \approx 0.1 \cdot (100p/p_{\text{th}})^{(d+1)/2}$

### Problem 3: Architecture Comparison (Challenging)

Design a comparison framework for evaluating logical qubit quality that accounts for:
a) Different gate speeds across platforms
b) Different connectivity constraints
c) Different syndrome extraction overheads
d) Practical implementation constraints

Apply your framework to rank Google, IBM, and Quantinuum for:
i) Near-term quantum chemistry applications
ii) Long-term cryptographic applications

---

## Computational Lab: Visualizing Progress Metrics

```python
"""
Day 988 Lab: Logical Qubit Milestone Visualization
Analyzing progress from major quantum computing companies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================
# Data: Key milestones and metrics (2019-2026)
# ============================================================

# Google Quantum AI timeline
google_dates = [
    datetime(2019, 10, 1),  # Sycamore quantum supremacy
    datetime(2021, 7, 1),   # Error correction demonstration
    datetime(2023, 2, 1),   # Improved logical operations
    datetime(2024, 12, 1),  # Willow below-threshold
    datetime(2025, 6, 1),   # Projected improvements
]
google_logical_error = [None, 0.03, 0.01, 0.001, 0.0003]  # None for pre-logical era
google_physical_qubits = [53, 72, 72, 105, 120]

# IBM timeline
ibm_dates = [
    datetime(2020, 9, 1),   # Quantum volume 128
    datetime(2021, 11, 1),  # Eagle 127 qubits
    datetime(2022, 11, 1),  # Osprey 433 qubits
    datetime(2023, 12, 1),  # Condor 1121 qubits
    datetime(2024, 12, 1),  # Heron 133 with better fidelity
    datetime(2025, 6, 1),   # Flamingo modular
]
ibm_qubits = [27, 127, 433, 1121, 133, 200]  # Heron is smaller but better
ibm_two_qubit_error = [0.01, 0.008, 0.007, 0.008, 0.005, 0.003]

# Quantinuum timeline
quantinuum_dates = [
    datetime(2021, 6, 1),   # H1 launch
    datetime(2022, 4, 1),   # QV 4096
    datetime(2023, 4, 1),   # H1-1 with 20 qubits
    datetime(2024, 5, 1),   # H2 with 56 qubits
    datetime(2024, 12, 1),  # Fault-tolerant demos
    datetime(2025, 6, 1),   # Projected H2+
]
quantinuum_qubits = [10, 12, 20, 32, 56, 72]
quantinuum_two_qubit_error = [0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0005]
quantinuum_logical_error = [None, None, 0.01, 0.003, 0.0001, 0.00005]

# ============================================================
# Figure 1: Physical qubit scaling over time
# ============================================================

fig1, ax1 = plt.subplots(figsize=(12, 6))

ax1.semilogy(google_dates, google_physical_qubits, 'o-',
             color='#4285F4', markersize=10, linewidth=2, label='Google')
ax1.semilogy(ibm_dates, ibm_qubits, 's-',
             color='#052D6E', markersize=10, linewidth=2, label='IBM')
ax1.semilogy(quantinuum_dates, quantinuum_qubits, '^-',
             color='#00A4E4', markersize=10, linewidth=2, label='Quantinuum')

# Add annotations for key milestones
ax1.annotate('Willow\n105 qubits', xy=(google_dates[3], google_physical_qubits[3]),
             xytext=(10, 20), textcoords='offset points', fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax1.annotate('Condor\n1121 qubits', xy=(ibm_dates[3], ibm_qubits[3]),
             xytext=(10, -30), textcoords='offset points', fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax1.annotate('H2\n56 qubits', xy=(quantinuum_dates[4], quantinuum_qubits[4]),
             xytext=(-60, 20), textcoords='offset points', fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))

ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Physical Qubits (log scale)', fontsize=12)
ax1.set_title('Physical Qubit Scaling: Major Platforms (2019-2026)', fontsize=14)
ax1.legend(loc='upper left', fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.set_ylim([5, 5000])
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qubit_scaling_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 2: Two-qubit gate error improvement
# ============================================================

fig2, ax2 = plt.subplots(figsize=(12, 6))

ax2.semilogy(ibm_dates, ibm_two_qubit_error, 's-',
             color='#052D6E', markersize=10, linewidth=2, label='IBM')
ax2.semilogy(quantinuum_dates, quantinuum_two_qubit_error, '^-',
             color='#00A4E4', markersize=10, linewidth=2, label='Quantinuum')

# Add threshold line
ax2.axhline(y=0.01, color='red', linestyle='--', linewidth=2,
            label='~1% threshold (surface code)')
ax2.axhline(y=0.001, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label='0.1% (high-threshold codes)')

ax2.fill_between([datetime(2019, 1, 1), datetime(2026, 1, 1)],
                  0.01, 0.1, alpha=0.1, color='red', label='Above threshold')
ax2.fill_between([datetime(2019, 1, 1), datetime(2026, 1, 1)],
                  0.0001, 0.01, alpha=0.1, color='green', label='Below threshold')

ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Two-Qubit Gate Error Rate (log scale)', fontsize=12)
ax2.set_title('Two-Qubit Gate Fidelity Improvement', fontsize=14)
ax2.legend(loc='upper right', fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_ylim([0.0001, 0.1])
ax2.set_xlim([datetime(2019, 1, 1), datetime(2026, 1, 1)])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gate_error_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 3: Logical error rate comparison
# ============================================================

fig3, ax3 = plt.subplots(figsize=(12, 6))

# Filter out None values for plotting
google_logical_dates = [d for d, e in zip(google_dates, google_logical_error) if e is not None]
google_logical_vals = [e for e in google_logical_error if e is not None]

quantinuum_logical_dates = [d for d, e in zip(quantinuum_dates, quantinuum_logical_error) if e is not None]
quantinuum_logical_vals = [e for e in quantinuum_logical_error if e is not None]

ax3.semilogy(google_logical_dates, google_logical_vals, 'o-',
             color='#4285F4', markersize=12, linewidth=2.5, label='Google (surface code)')
ax3.semilogy(quantinuum_logical_dates, quantinuum_logical_vals, '^-',
             color='#00A4E4', markersize=12, linewidth=2.5, label='Quantinuum (color code)')

# Target regions
ax3.axhspan(1e-6, 1e-4, alpha=0.2, color='yellow', label='Near-term algorithms')
ax3.axhspan(1e-12, 1e-8, alpha=0.2, color='green', label='Cryptographic applications')

# Add projections
future_dates = [datetime(2026, 1, 1), datetime(2027, 1, 1), datetime(2028, 1, 1)]
google_projected = [1e-4, 3e-5, 1e-5]
quantinuum_projected = [1e-5, 3e-6, 1e-6]

ax3.semilogy(future_dates, google_projected, 'o--', color='#4285F4',
             alpha=0.5, markersize=8, linewidth=1.5)
ax3.semilogy(future_dates, quantinuum_projected, '^--', color='#00A4E4',
             alpha=0.5, markersize=8, linewidth=1.5)

ax3.axvline(x=datetime(2025, 6, 1), color='gray', linestyle=':', alpha=0.7)
ax3.text(datetime(2025, 8, 1), 0.05, 'Projections →', fontsize=10, color='gray')

ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Logical Error Rate per Round (log scale)', fontsize=12)
ax3.set_title('Logical Qubit Error Rate: Demonstrated vs Projected', fontsize=14)
ax3.legend(loc='upper right', fontsize=10)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.set_ylim([1e-12, 0.1])
ax3.set_xlim([datetime(2020, 1, 1), datetime(2029, 1, 1)])
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logical_error_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 4: Error suppression factor (Lambda)
# ============================================================

fig4, ax4 = plt.subplots(figsize=(10, 6))

# Google's reported Lambda values
distances = [3, 5, 7, 9, 11]  # Including projections
google_lambda = [2.14, 2.0, 1.9, 1.8, 1.7]  # Measured + projected decrease
google_lambda_projected = [None, None, 1.9, 1.8, 1.7]

# Theoretical Lambda for different physical error rates
p_over_threshold = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
lambda_theoretical = 1 / p_over_threshold

ax4.bar(np.array(distances) - 0.2, [2.14, 2.0, None, None, None],
        width=0.4, color='#4285F4', label='Google measured', alpha=0.8)
ax4.bar(np.array(distances)[2:] + 0.2, google_lambda[2:],
        width=0.4, color='#4285F4', label='Google projected', alpha=0.4)

ax4.axhline(y=2, color='green', linestyle='--', linewidth=2,
            label='Λ = 2 (below threshold)')
ax4.axhline(y=1, color='red', linestyle='--', linewidth=2,
            label='Λ = 1 (at threshold)')

ax4.set_xlabel('Code Distance d', fontsize=12)
ax4.set_ylabel('Error Suppression Factor Λ', fontsize=12)
ax4.set_title('Error Suppression Factor vs Code Distance', fontsize=14)
ax4.legend(loc='upper right', fontsize=10)
ax4.set_ylim([0, 3])
ax4.set_xticks(distances)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lambda_vs_distance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 5: Platform comparison radar chart
# ============================================================

fig5, ax5 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

categories = ['Qubit Count', 'Gate Fidelity', 'Connectivity',
              'Speed', 'Logical Qubit', 'Scalability']
N = len(categories)

# Normalize scores (0-10 scale)
google_scores = [7, 8, 5, 10, 8, 8]   # 105 qubits, 99.7%, nearest-neighbor, fast, d=7, good scaling
ibm_scores = [10, 6, 4, 9, 3, 7]      # 1121 qubits, 99.5%, limited, fast, mitigation only
quantinuum_scores = [5, 10, 10, 3, 9, 5]  # 56 qubits, 99.9%, all-to-all, slow, d=3 color

# Close the radar chart
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

google_scores += google_scores[:1]
ibm_scores += ibm_scores[:1]
quantinuum_scores += quantinuum_scores[:1]

ax5.plot(angles, google_scores, 'o-', linewidth=2, color='#4285F4', label='Google')
ax5.fill(angles, google_scores, alpha=0.15, color='#4285F4')
ax5.plot(angles, ibm_scores, 's-', linewidth=2, color='#052D6E', label='IBM')
ax5.fill(angles, ibm_scores, alpha=0.15, color='#052D6E')
ax5.plot(angles, quantinuum_scores, '^-', linewidth=2, color='#00A4E4', label='Quantinuum')
ax5.fill(angles, quantinuum_scores, alpha=0.15, color='#00A4E4')

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, fontsize=11)
ax5.set_ylim([0, 10])
ax5.set_title('Platform Comparison (2025)', fontsize=14, pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('platform_radar_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary Statistics
# ============================================================

print("\n" + "="*60)
print("LOGICAL QUBIT MILESTONE SUMMARY (2024-2026)")
print("="*60)

print("\n--- Google Quantum AI ---")
print(f"Physical qubits (Willow): 105")
print(f"Surface code distance achieved: d = 7")
print(f"Logical error rate: ~0.1% per round")
print(f"Error suppression factor Λ: 2.0 - 2.1")
print(f"Key achievement: Below threshold operation demonstrated")

print("\n--- IBM Quantum ---")
print(f"Physical qubits (Condor): 1,121")
print(f"Physical qubits (Heron): 133 with 99.5% 2Q gates")
print(f"Approach: Error mitigation, circuit knitting")
print(f"Roadmap: 100,000+ qubits by 2033")
print(f"Key achievement: Utility-scale experiments")

print("\n--- Quantinuum ---")
print(f"Physical qubits (H2): 56")
print(f"Two-qubit gate fidelity: 99.9%+")
print(f"Logical error rate: ~0.01% per round")
print(f"Key achievement: Repeated fault-tolerant error correction")

print("\n" + "="*60)
print("PROJECTIONS TO USEFUL FAULT TOLERANCE")
print("="*60)
print(f"Target logical error rate: 10^-10 (cryptographic)")
print(f"Google projected timeline: ~2032")
print(f"Quantinuum projected timeline: ~2029")
print(f"IBM projected timeline: ~2033 (100x100 target)")
print("="*60)
```

---

## Summary

### Key Achievements (2024-2026)

| Company | Milestone | Significance |
|---------|-----------|--------------|
| Google | Willow below threshold | First superconducting logical qubit beating physical |
| IBM | 1000+ qubit chip | Largest physical qubit count |
| Quantinuum | Repeated fault-tolerant EC | Highest fidelity logical operations |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Error suppression | $$\Lambda = p_L(d)/p_L(d+2) > 2$$ |
| Below threshold | $$p_L \propto (p/p_{\text{th}})^{(d+1)/2}$$ |
| Logical lifetime | $$T_1^{\text{logical}} > T_1^{\text{physical}}$$ |

### Main Takeaways

1. **The threshold has been crossed** - Multiple platforms now demonstrate below-threshold operation
2. **Different strategies emerge** - Surface codes (Google), error mitigation (IBM), color codes (Quantinuum)
3. **Trade-offs persist** - Speed vs fidelity, qubit count vs quality
4. **Utility gap remains** - Years of improvement needed for cryptographic applications
5. **Competition accelerates progress** - Multiple viable paths to fault tolerance

---

## Daily Checklist

- [ ] I can explain Google's Willow achievements and their significance
- [ ] I can analyze IBM's roadmap and error mitigation approach
- [ ] I can evaluate Quantinuum's fault-tolerant demonstrations
- [ ] I can compare platform strengths and weaknesses
- [ ] I can calculate error suppression factors from experimental data
- [ ] I understand the path from current demos to useful fault tolerance
- [ ] I ran the visualization code and can interpret the results

---

## Preview: Day 989

Tomorrow we examine **Quantum Advantage & Verification** - analyzing claims of quantum computational speedup, the ongoing classical competition, and methods for verifying quantum results. We'll critically evaluate the $10^{25}$ year classical runtime claims and discuss what "useful quantum advantage" really means.

---

*"The demonstration of error correction below threshold represents a phase transition in quantum computing - from fighting errors to systematically suppressing them. The question now is how quickly we can scale these demonstrations to useful applications."*
