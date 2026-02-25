# Day 834: Google Willow Architecture

## Week 120, Day 1 | Month 30: Surface Codes | Semester 2A: Quantum Error Correction

### Overview

Today we examine Google's Willow quantum processor, the first system to achieve below-threshold surface code operation. With 105 qubits arranged in a Sycamore-style layout, Willow represents a major milestone in quantum error correction, demonstrating that increasing code distance genuinely suppresses logical errors exponentially.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Willow architecture deep dive |
| **Afternoon** | 2.5 hours | Device physics and engineering |
| **Evening** | 1.5 hours | Computational lab: Architecture simulation |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Describe Willow's qubit layout** - Explain the 105-qubit arrangement for d=3,5,7 surface codes
2. **Analyze transmon design choices** - Understand frequency allocation and coherence optimization
3. **Explain tunable coupler operation** - Detail the flux-tunable coupler mechanism for two-qubit gates
4. **Evaluate connectivity patterns** - Assess how the grid connectivity enables surface code operation
5. **Quantify gate performance** - Interpret single and two-qubit gate fidelities achieved
6. **Compare with predecessor systems** - Contrast Willow with Sycamore (2019) and Weber (2021)

---

## Core Content

### 1. The Road to Willow

#### Historical Context

Google's quantum supremacy demonstration (2019) used 53 qubits of the 54-qubit Sycamore processor. Willow (2024) represents five years of iterative improvement:

| Generation | Year | Qubits | Best 2Q Gate Error | Surface Code Result |
|------------|------|--------|---------------------|---------------------|
| Sycamore | 2019 | 54 | 0.36% | N/A (supremacy demo) |
| Weber | 2021 | 53 | 0.50% | Repetition code only |
| Willow | 2024 | 105 | 0.25% | Below threshold at d=7 |

#### The Breakthrough Requirement

For surface code operation below threshold, we need:

$$\boxed{p_{\text{physical}} < p_{\text{threshold}} \approx 1\%}$$

But the effective threshold depends on the full error model including:
- Single-qubit gate errors
- Two-qubit gate errors
- Measurement errors
- Leakage to non-computational states
- Crosstalk between qubits

Google's achievement required simultaneous improvement across all error channels.

### 2. Willow Qubit Architecture

#### 2.1 Transmon Design

The Willow transmon qubits are fixed-frequency superconducting qubits with:

**Josephson Junction Parameters:**
$$E_J/E_C \approx 50-80$$

where $E_J$ is the Josephson energy and $E_C$ is the charging energy:

$$E_C = \frac{e^2}{2C_{\Sigma}}$$

$$E_J = \frac{\Phi_0 I_c}{2\pi}$$

The transmon frequency is approximately:

$$\boxed{\omega_{01} \approx \sqrt{8 E_J E_C} - E_C}$$

**Anharmonicity:**
$$\alpha = \omega_{12} - \omega_{01} \approx -E_C \approx -200 \text{ to } -300 \text{ MHz}$$

This anharmonicity allows selective addressing of the $|0\rangle \leftrightarrow |1\rangle$ transition.

#### 2.2 Frequency Allocation

With 105 qubits on a 2D grid, frequency collision management is critical:

**Frequency Bands:**
- Data qubits: 5.0-6.0 GHz (spread across ~200 MHz bands)
- Measure qubits: 5.5-6.5 GHz (interleaved with data)

**Collision Avoidance:**
For neighboring qubits $i$ and $j$:
$$|\omega_i - \omega_j| > 100 \text{ MHz}$$
$$|\omega_i - \omega_j| \neq |n\alpha|, \quad n = 1, 2$$

The second condition avoids resonances that could cause leakage.

#### 2.3 Coherence Properties

**Willow Coherence Times (Median Values):**

| Property | Symbol | Willow (2024) | Sycamore (2019) |
|----------|--------|---------------|-----------------|
| Relaxation time | $T_1$ | 68 μs | 15 μs |
| Dephasing time | $T_2^*$ | 30 μs | 10 μs |
| Echo time | $T_2^{\text{echo}}$ | 85 μs | 20 μs |

The improvement factor of ~4× in coherence was essential for below-threshold operation.

### 3. Tunable Coupler Engineering

#### 3.1 Coupler Architecture

Each pair of neighboring data qubits is connected through a tunable coupler—a separate transmon with adjustable frequency:

```
Data Qubit A ----[Tunable Coupler]---- Data Qubit B
    ↓                   ↓                   ↓
  Fixed ω_A      Flux-tunable ω_c       Fixed ω_B
```

The effective coupling between qubits A and B:

$$\boxed{g_{\text{eff}} = g_{Ac}g_{Bc}\left(\frac{1}{\Delta_{Ac}} + \frac{1}{\Delta_{Bc}}\right)}$$

where:
- $g_{Ac}, g_{Bc}$ = capacitive coupling to coupler
- $\Delta_{Ac} = \omega_A - \omega_c$, $\Delta_{Bc} = \omega_B - \omega_c$

#### 3.2 Coupler Operation Modes

**OFF State (Idling):**
$$\omega_c \approx \frac{\omega_A + \omega_B}{2}$$

This symmetric detuning cancels the effective coupling:
$$g_{\text{eff}} \approx 0$$

**ON State (Two-Qubit Gate):**
$$\omega_c \rightarrow \omega_c^{\text{gate}}$$

Asymmetric detuning activates coupling for iSWAP-type gates.

#### 3.3 Two-Qubit Gate Implementation

Willow uses CZ gates implemented via controlled-phase accumulation:

$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

**Gate Mechanism:**
When both qubits are in $|1\rangle$, the $|11\rangle$ state experiences an energy shift:

$$\Delta E_{11} = \frac{g_{\text{eff}}^2}{\Delta_{11}} - \frac{g_{\text{eff}}^2}{\Delta_{02}} - \frac{g_{\text{eff}}^2}{\Delta_{20}}$$

The gate time is set to accumulate a $\pi$ phase:

$$\boxed{t_{\text{gate}} = \frac{\pi \hbar}{\Delta E_{11}} \approx 20-30 \text{ ns}}$$

### 4. Surface Code Layout on Willow

#### 4.1 Qubit Arrangement

The 105 qubits are arranged on a 2D grid optimized for surface codes:

```
Distance-7 Surface Code Layout (105 qubits total):

    ●───○───●───○───●───○───●
    │   │   │   │   │   │   │
    ○───●───○───●───○───●───○
    │   │   │   │   │   │   │
    ●───○───●───○───●───○───●
    │   │   │   │   │   │   │
    ○───●───○───●───○───●───○
    │   │   │   │   │   │   │
    ●───○───●───○───●───○───●
    │   │   │   │   │   │   │
    ○───●───○───●───○───●───○
    │   │   │   │   │   │   │
    ●───○───●───○───●───○───●

    ● = Data qubit
    ○ = Measure qubit (syndrome)
```

#### 4.2 Qubit Counts for Each Distance

| Code Distance | Data Qubits | Measure Qubits | Total |
|--------------|-------------|----------------|-------|
| d = 3 | 9 | 8 | 17 |
| d = 5 | 25 | 24 | 49 |
| d = 7 | 49 | 48 | 97 |

The d=7 code uses 97 of the 105 available qubits.

#### 4.3 Stabilizer Measurement Circuit

Each error correction cycle involves:

1. **Initialize measure qubits**: Reset to $|0\rangle$
2. **CNOT/CZ layer 1**: Connect to first neighbors
3. **CNOT/CZ layer 2**: Connect to second neighbors
4. **CNOT/CZ layer 3**: Connect to third neighbors
5. **CNOT/CZ layer 4**: Connect to fourth neighbors
6. **Readout**: Measure syndrome qubits

**Circuit Depth:** 4 CZ layers + measurement = ~1 μs cycle

$$\boxed{T_{\text{cycle}} \approx 1.0 \text{ μs}}$$

### 5. Gate Performance Metrics

#### 5.1 Single-Qubit Gates

Single-qubit gates are implemented via microwave pulses:

$$R_\phi(\theta) = \exp\left(-i\frac{\theta}{2}(\cos\phi \, \sigma_x + \sin\phi \, \sigma_y)\right)$$

**Willow Single-Qubit Gate Metrics:**
| Metric | Value |
|--------|-------|
| Gate time | 25 ns |
| Pauli error | 0.035% |
| Leakage rate | 0.01% |

#### 5.2 Two-Qubit Gates

**Willow CZ Gate Metrics:**
| Metric | Value |
|--------|-------|
| Gate time | 26 ns |
| Pauli error | 0.25% |
| Leakage rate | 0.10% |

#### 5.3 Measurement

**Willow Readout Metrics:**
| Metric | Value |
|--------|-------|
| Readout time | 500 ns |
| Assignment error | 0.7% |
| Reset fidelity | 99.5% |

### 6. Error Budget Analysis

#### 6.1 Cycle Error Contributions

The total error per syndrome measurement cycle:

$$\boxed{p_{\text{cycle}} = p_{\text{1Q}} + p_{\text{2Q}} + p_{\text{meas}} + p_{\text{idle}} + p_{\text{leak}}}$$

For Willow:
| Error Source | Contribution | Notes |
|--------------|--------------|-------|
| 2Q gates (4 per cycle) | 4 × 0.25% = 1.0% | Dominant source |
| Measurement | 0.7% | Assignment errors |
| Idle ($T_1$ decay) | 0.5% | During 1 μs cycle |
| 1Q gates | 0.07% | Two gates per cycle |
| Leakage | 0.2% | Accumulates over cycles |

**Total cycle error budget:** ~2.5% per qubit per cycle

#### 6.2 Why It Still Works

Even with ~2.5% per-qubit cycle error, below-threshold operation is achieved because:

1. **Correlated errors are rare**: Most errors are independent
2. **Decoder efficiency**: MWPM achieves near-optimal correction
3. **Logical weight**: Errors must form chains of length $d/2$ to cause logical failure

---

## Worked Examples

### Example 1: Calculating Coupler Frequency for Zero Coupling

**Problem:** Two data qubits have frequencies $\omega_A = 5.2$ GHz and $\omega_B = 5.4$ GHz. The capacitive couplings are $g_{Ac} = g_{Bc} = 30$ MHz. Find the coupler frequency $\omega_c$ that makes $g_{\text{eff}} = 0$.

**Solution:**

For zero effective coupling:
$$g_{\text{eff}} = g_{Ac}g_{Bc}\left(\frac{1}{\omega_A - \omega_c} + \frac{1}{\omega_B - \omega_c}\right) = 0$$

This requires:
$$\frac{1}{\omega_A - \omega_c} = -\frac{1}{\omega_B - \omega_c}$$

$$\omega_B - \omega_c = -(\omega_A - \omega_c)$$

$$\omega_B - \omega_c = -\omega_A + \omega_c$$

$$\omega_A + \omega_B = 2\omega_c$$

$$\boxed{\omega_c = \frac{\omega_A + \omega_B}{2} = \frac{5.2 + 5.4}{2} = 5.3 \text{ GHz}}$$

The coupler should be placed symmetrically between the qubit frequencies.

### Example 2: Gate Time Estimation

**Problem:** A CZ gate is implemented with effective coupling $g_{\text{eff}} = 15$ MHz when activated. The energy shift of $|11\rangle$ relative to other computational states is $\Delta E_{11} = 2\pi \times 25$ MHz. Calculate the gate time.

**Solution:**

The CZ gate requires a $\pi$ phase accumulation on $|11\rangle$:

$$\phi = \frac{\Delta E_{11} \cdot t}{\hbar} = \pi$$

$$t = \frac{\pi \hbar}{\Delta E_{11}} = \frac{\pi}{2\pi \times 25 \text{ MHz}}$$

$$t = \frac{1}{2 \times 25 \text{ MHz}} = \frac{1}{50 \times 10^6 \text{ Hz}}$$

$$\boxed{t_{\text{gate}} = 20 \text{ ns}}$$

### Example 3: Coherence-Limited Error Rate

**Problem:** A qubit has $T_1 = 68$ μs and $T_2 = 85$ μs. The error correction cycle takes 1 μs. What is the idle error probability per cycle?

**Solution:**

The idle error has contributions from both relaxation and dephasing:

**Relaxation contribution:**
$$p_{T_1} = 1 - e^{-t/T_1} \approx \frac{t}{T_1} = \frac{1 \text{ μs}}{68 \text{ μs}} = 0.0147$$

**Pure dephasing contribution:**
$$\frac{1}{T_\phi} = \frac{1}{T_2} - \frac{1}{2T_1}$$

$$T_\phi = \left(\frac{1}{85} - \frac{1}{136}\right)^{-1} \text{ μs} = 227 \text{ μs}$$

$$p_{T_\phi} \approx \frac{t}{T_\phi} = \frac{1}{227} = 0.0044$$

**Total idle error:**
$$p_{\text{idle}} \approx p_{T_1} + p_{T_\phi} = 0.0147 + 0.0044 = 0.019$$

$$\boxed{p_{\text{idle}} \approx 1.9\%}$$

This is higher than the 0.5% quoted because the actual surface code idle is only on data qubits, which are idle for less than 1 μs during the syndrome extraction.

---

## Practice Problems

### Direct Application

**Problem 1:** A Willow transmon has $E_J/h = 20$ GHz and $E_C/h = 300$ MHz. Calculate:
a) The qubit frequency $\omega_{01}$
b) The anharmonicity $\alpha$
c) The $|1\rangle \rightarrow |2\rangle$ transition frequency

**Problem 2:** The d=5 surface code on Willow uses 49 qubits. If each CZ gate has error rate 0.25%, how many CZ gates are in one syndrome extraction cycle, and what is the total two-qubit gate error contribution?

**Problem 3:** A tunable coupler is biased to create effective coupling $g_{\text{eff}} = 10$ MHz. How long should the coupling be active to implement an iSWAP gate (which requires $\pi/2$ rotation)?

### Intermediate

**Problem 4:** The logical error rate for the d=7 code is 0.143% per cycle. If $T_{\text{cycle}} = 1$ μs, what is the logical error rate per microsecond? Compare this to the best physical qubit error rate of $1/T_1 = 1/(68 \text{ μs})$.

**Problem 5:** Design a frequency allocation scheme for a 9-qubit (d=3) surface code patch where all neighboring qubits differ by at least 100 MHz and no two qubits are separated by exactly $|\alpha| = 250$ MHz.

**Problem 6:** The Willow processor demonstrates a 4× improvement in coherence over Sycamore. If gate fidelities scale linearly with coherence, estimate the expected improvement in below-threshold performance.

### Challenging

**Problem 7:** Consider the trade-off between gate speed and leakage. Faster gates have higher leakage rates approximately as $p_{\text{leak}} \propto 1/t_{\text{gate}}^2$. If a 26 ns gate has 0.10% leakage, what gate time minimizes total error (leakage + idle) assuming idle error is 0.02% per nanosecond?

**Problem 8:** The coupler introduces a residual ZZ interaction even in the OFF state:
$$H_{ZZ} = \frac{\zeta}{4}Z_A Z_B$$
where $\zeta \approx 30$ kHz typically. Calculate the phase error accumulated on $|11\rangle$ during a 1 μs cycle and estimate its contribution to the error budget.

---

## Computational Lab: Willow Architecture Simulation

```python
"""
Day 834 Computational Lab: Google Willow Architecture Analysis
Simulates key architectural features of the Willow quantum processor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# =============================================================================
# Part 1: Transmon Physics
# =============================================================================

def transmon_frequencies(EJ, EC, n_levels=3):
    """
    Calculate transmon energy levels using perturbation theory.

    Parameters:
    -----------
    EJ : float
        Josephson energy in GHz
    EC : float
        Charging energy in GHz
    n_levels : int
        Number of levels to compute

    Returns:
    --------
    frequencies : array
        Transition frequencies in GHz
    anharmonicity : float
        Anharmonicity alpha in MHz
    """
    # Approximate formula valid for EJ/EC >> 1
    ratio = EJ / EC

    # Ground state energy
    E0 = -EJ + np.sqrt(8 * EJ * EC) * 0.5 - EC / 4

    # Energy levels (in units of EC)
    levels = []
    for n in range(n_levels):
        # Approximate energy from harmonic oscillator + anharmonic correction
        E_n = -EJ + np.sqrt(8 * EJ * EC) * (n + 0.5) - (EC / 12) * (6 * n**2 + 6 * n + 3)
        levels.append(E_n)

    levels = np.array(levels)
    transitions = np.diff(levels)

    # Anharmonicity
    if len(transitions) >= 2:
        anharmonicity = (transitions[1] - transitions[0]) * 1000  # Convert to MHz
    else:
        anharmonicity = -EC * 1000  # Approximate value

    return transitions, anharmonicity

# Example: Willow-like transmon
EJ = 20  # GHz
EC = 0.3  # GHz

freqs, alpha = transmon_frequencies(EJ, EC)
print("=" * 60)
print("TRANSMON PHYSICS ANALYSIS")
print("=" * 60)
print(f"EJ/EC ratio: {EJ/EC:.1f}")
print(f"Qubit frequency (ω01): {freqs[0]:.3f} GHz")
if len(freqs) > 1:
    print(f"12 transition (ω12): {freqs[1]:.3f} GHz")
print(f"Anharmonicity: {alpha:.1f} MHz")

# =============================================================================
# Part 2: Tunable Coupler Analysis
# =============================================================================

def effective_coupling(omega_A, omega_B, omega_c, g_Ac, g_Bc):
    """
    Calculate effective coupling between qubits A and B via coupler.

    Parameters:
    -----------
    omega_A, omega_B : float
        Qubit frequencies in GHz
    omega_c : float
        Coupler frequency in GHz
    g_Ac, g_Bc : float
        Capacitive couplings in MHz

    Returns:
    --------
    g_eff : float
        Effective coupling in MHz
    """
    delta_Ac = (omega_A - omega_c) * 1000  # Convert to MHz
    delta_Bc = (omega_B - omega_c) * 1000  # Convert to MHz

    # Avoid division by zero
    if abs(delta_Ac) < 1 or abs(delta_Bc) < 1:
        return float('inf')

    g_eff = g_Ac * g_Bc * (1/delta_Ac + 1/delta_Bc)
    return g_eff

# Sweep coupler frequency
omega_A = 5.2  # GHz
omega_B = 5.4  # GHz
g_Ac = g_Bc = 30  # MHz

omega_c_range = np.linspace(4.5, 6.5, 1000)
g_eff_values = [effective_coupling(omega_A, omega_B, wc, g_Ac, g_Bc)
                for wc in omega_c_range]

# Find zero crossing
zero_coupling_freq = (omega_A + omega_B) / 2

print("\n" + "=" * 60)
print("TUNABLE COUPLER ANALYSIS")
print("=" * 60)
print(f"Qubit A frequency: {omega_A} GHz")
print(f"Qubit B frequency: {omega_B} GHz")
print(f"Zero coupling at: {zero_coupling_freq} GHz")

# =============================================================================
# Part 3: Surface Code Layout
# =============================================================================

def create_surface_code_layout(distance):
    """
    Create a surface code qubit layout.

    Parameters:
    -----------
    distance : int
        Code distance (odd number)

    Returns:
    --------
    data_qubits : list of (x, y)
    measure_qubits : list of (x, y, type)  # type = 'X' or 'Z'
    """
    data_qubits = []
    measure_qubits = []

    for i in range(distance):
        for j in range(distance):
            # Data qubits at integer coordinates
            data_qubits.append((i, j))

    # Measure qubits between data qubits
    for i in range(distance + 1):
        for j in range(distance + 1):
            # Skip corners based on parity
            if i == 0 or i == distance:
                if j == 0 or j == distance:
                    continue
                if (i + j) % 2 == 1:
                    continue
            if j == 0 or j == distance:
                if (i + j) % 2 == 0:
                    continue

            # Determine X or Z stabilizer
            if (i + j) % 2 == 0:
                measure_qubits.append((i - 0.5, j - 0.5, 'Z'))
            else:
                measure_qubits.append((i - 0.5, j - 0.5, 'X'))

    return data_qubits, measure_qubits

def count_qubits(distance):
    """Count data and measure qubits for a given code distance."""
    n_data = distance ** 2
    n_measure = distance ** 2 - 1
    return n_data, n_measure, n_data + n_measure

print("\n" + "=" * 60)
print("SURFACE CODE LAYOUT")
print("=" * 60)
for d in [3, 5, 7]:
    n_data, n_meas, n_total = count_qubits(d)
    print(f"Distance {d}: {n_data} data + {n_meas} measure = {n_total} total qubits")

# =============================================================================
# Part 4: Gate Error Analysis
# =============================================================================

def cycle_error_budget(p_1q=0.00035, p_2q=0.0025, p_meas=0.007,
                       p_idle=0.005, p_leak=0.002,
                       n_1q=2, n_2q=4):
    """
    Calculate error budget for one syndrome extraction cycle.

    Parameters:
    -----------
    p_1q : float
        Single-qubit gate error
    p_2q : float
        Two-qubit gate error
    p_meas : float
        Measurement error
    p_idle : float
        Idle error during cycle
    p_leak : float
        Leakage error
    n_1q : int
        Number of 1Q gates per data qubit per cycle
    n_2q : int
        Number of 2Q gates per data qubit per cycle

    Returns:
    --------
    budget : dict
        Error contributions by source
    """
    budget = {
        'single_qubit': n_1q * p_1q,
        'two_qubit': n_2q * p_2q,
        'measurement': p_meas,
        'idle': p_idle,
        'leakage': p_leak
    }
    budget['total'] = sum(budget.values())
    return budget

# Willow error budget
budget = cycle_error_budget()

print("\n" + "=" * 60)
print("ERROR BUDGET PER CYCLE (Willow)")
print("=" * 60)
for source, value in budget.items():
    print(f"{source:15s}: {value*100:.3f}%")

# =============================================================================
# Part 5: Coherence-Limited Performance
# =============================================================================

def coherence_limited_error(T1, T2, t_cycle):
    """
    Calculate coherence-limited error per cycle.

    Parameters:
    -----------
    T1 : float
        Relaxation time in μs
    T2 : float
        Dephasing time in μs
    t_cycle : float
        Cycle time in μs

    Returns:
    --------
    p_error : float
        Error probability per cycle
    """
    # Relaxation contribution
    p_T1 = 1 - np.exp(-t_cycle / T1)

    # Pure dephasing contribution
    T_phi = 1 / (1/T2 - 1/(2*T1))
    p_T_phi = 1 - np.exp(-t_cycle / T_phi)

    return p_T1, p_T_phi, p_T1 + p_T_phi

# Willow coherence
T1 = 68  # μs
T2 = 85  # μs
t_cycle = 1  # μs

p_T1, p_T_phi, p_total = coherence_limited_error(T1, T2, t_cycle)

print("\n" + "=" * 60)
print("COHERENCE-LIMITED ERRORS")
print("=" * 60)
print(f"T1 = {T1} μs, T2 = {T2} μs")
print(f"Cycle time = {t_cycle} μs")
print(f"Relaxation error: {p_T1*100:.3f}%")
print(f"Dephasing error: {p_T_phi*100:.3f}%")
print(f"Total coherence error: {p_total*100:.3f}%")

# =============================================================================
# Part 6: Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Effective coupling vs coupler frequency
ax1 = axes[0, 0]
ax1.plot(omega_c_range, g_eff_values, 'b-', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=zero_coupling_freq, color='r', linestyle='--',
            label=f'Zero coupling: {zero_coupling_freq} GHz')
ax1.axvline(x=omega_A, color='g', linestyle=':', alpha=0.7, label=f'ωA = {omega_A} GHz')
ax1.axvline(x=omega_B, color='m', linestyle=':', alpha=0.7, label=f'ωB = {omega_B} GHz')
ax1.set_xlim(4.5, 6.5)
ax1.set_ylim(-50, 50)
ax1.set_xlabel('Coupler Frequency (GHz)', fontsize=12)
ax1.set_ylabel('Effective Coupling (MHz)', fontsize=12)
ax1.set_title('Tunable Coupler Response', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Surface code layout for d=5
ax2 = axes[0, 1]
data_qubits, measure_qubits = create_surface_code_layout(5)

for x, y in data_qubits:
    ax2.scatter(x, y, s=200, c='blue', marker='o', edgecolors='black', zorder=3)

for x, y, stype in measure_qubits:
    color = 'red' if stype == 'X' else 'green'
    ax2.scatter(x, y, s=150, c=color, marker='s', edgecolors='black', zorder=3)

# Draw connections
for x, y in data_qubits:
    for mx, my, _ in measure_qubits:
        if abs(x - mx - 0.5) < 0.6 and abs(y - my - 0.5) < 0.6:
            ax2.plot([x, mx], [y, my], 'gray', linewidth=1, alpha=0.5)

ax2.set_xlabel('X position', fontsize=12)
ax2.set_ylabel('Y position', fontsize=12)
ax2.set_title('Distance-5 Surface Code Layout', fontsize=14)
ax2.set_aspect('equal')
ax2.legend(['Data qubits (blue)', 'X stabilizer (red)', 'Z stabilizer (green)'])
ax2.grid(True, alpha=0.3)

# Plot 3: Error budget breakdown
ax3 = axes[1, 0]
labels = ['2Q Gates', 'Measurement', 'Idle', 'Leakage', '1Q Gates']
values = [budget['two_qubit'], budget['measurement'], budget['idle'],
          budget['leakage'], budget['single_qubit']]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

wedges, texts, autotexts = ax3.pie(values, labels=labels, autopct='%1.1f%%',
                                    colors=colors, explode=[0.1, 0, 0, 0, 0])
ax3.set_title('Error Budget per Syndrome Cycle', fontsize=14)

# Plot 4: Coherence improvement comparison
ax4 = axes[1, 1]
generations = ['Sycamore\n(2019)', 'Weber\n(2021)', 'Willow\n(2024)']
T1_values = [15, 25, 68]
T2_values = [10, 20, 30]

x = np.arange(len(generations))
width = 0.35

bars1 = ax4.bar(x - width/2, T1_values, width, label='T1', color='steelblue')
bars2 = ax4.bar(x + width/2, T2_values, width, label='T2*', color='coral')

ax4.set_xlabel('Processor Generation', fontsize=12)
ax4.set_ylabel('Coherence Time (μs)', fontsize=12)
ax4.set_title('Coherence Time Evolution', fontsize=14)
ax4.set_xticks(x)
ax4.set_xticklabels(generations)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax4.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax4.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('day_834_willow_architecture.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Visualization saved to: day_834_willow_architecture.png")
print("=" * 60)

# =============================================================================
# Part 7: CZ Gate Time Calculation
# =============================================================================

def cz_gate_time(g_eff_active, anharmonicity=-250):
    """
    Estimate CZ gate time from effective coupling.

    Parameters:
    -----------
    g_eff_active : float
        Active coupling strength in MHz
    anharmonicity : float
        Transmon anharmonicity in MHz

    Returns:
    --------
    t_gate : float
        Gate time in ns
    """
    # Simplified model: phase from avoided crossing
    delta_E = g_eff_active**2 / abs(anharmonicity)  # MHz
    t_gate = 0.5 / delta_E * 1000  # Convert to ns (π phase needs half period)
    return t_gate

# Example
g_active = 15  # MHz
t_gate = cz_gate_time(g_active)
print(f"\nWith g_eff = {g_active} MHz:")
print(f"Estimated CZ gate time: {t_gate:.1f} ns")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Transmon frequency | $\omega_{01} \approx \sqrt{8 E_J E_C} - E_C$ |
| Anharmonicity | $\alpha \approx -E_C$ |
| Effective coupling | $g_{\text{eff}} = g_{Ac}g_{Bc}\left(\frac{1}{\Delta_{Ac}} + \frac{1}{\Delta_{Bc}}\right)$ |
| Zero coupling condition | $\omega_c = \frac{\omega_A + \omega_B}{2}$ |
| CZ gate time | $t_{\text{gate}} = \frac{\pi \hbar}{\Delta E_{11}}$ |
| Cycle time | $T_{\text{cycle}} \approx 1$ μs |
| Coherence error | $p_{\text{idle}} \approx t/T_1 + t/T_\phi$ |

### Main Takeaways

1. **Willow represents a 4× coherence improvement** over Sycamore, enabling below-threshold operation
2. **Tunable couplers** enable fast, high-fidelity CZ gates by dynamically controlling qubit-qubit interaction
3. **The grid connectivity** perfectly matches surface code stabilizer requirements
4. **Error budget is dominated by two-qubit gates** (~1% of ~2.5% total per cycle)
5. **105 qubits support up to distance-7** surface codes with 97 qubits active

### Daily Checklist

- [ ] I understand the transmon qubit physics underlying Willow
- [ ] I can calculate effective coupling in a tunable coupler system
- [ ] I know the qubit counts for d=3, 5, 7 surface codes
- [ ] I can break down the error budget for syndrome extraction
- [ ] I understand why 2-qubit gate error is the dominant limitation
- [ ] I completed the computational lab and analyzed the results

---

## Preview: Day 835

Tomorrow we analyze the historic below-threshold results from Willow in detail. We will quantify the error suppression factor λ = 2.14, understand the statistical methods used to verify below-threshold operation, and explore the implications for quantum error correction scaling.

**Key topics:**
- Logical error rate measurements at d = 3, 5, 7
- Error suppression factor extraction
- Logical qubit lifetime exceeding physical qubit lifetime
- Statistical confidence and uncertainty analysis
