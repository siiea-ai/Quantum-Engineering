# Day 818: Error Budgets and Code Distance Selection

## Week 117, Day 6 | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Overview

Designing a fault-tolerant quantum computer requires answering a fundamental question: *How many physical qubits do we need?* The answer depends on the error budget—how physical error rates translate into logical error rates—and the code distance required to achieve target performance. Today we develop the quantitative framework for error budget analysis and distance selection, connecting theoretical thresholds to practical engineering decisions.

---

## Daily Schedule

| Session | Duration | Content |
|---------|----------|---------|
| Morning | 3 hours | Error budget decomposition, threshold analysis |
| Afternoon | 2 hours | Distance selection, worked calculations |
| Evening | 2 hours | Python modeling of logical error rates |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Decompose** total error budgets into gate, measurement, and idle components
2. **Apply** the $d = 2t + 1$ criterion for distance selection
3. **Calculate** logical error rates from physical error rates
4. **Determine** minimum code distance for target logical error rates
5. **Analyze** threshold behavior and sub-threshold scaling
6. **Optimize** resource allocation given error budget constraints

---

## Core Content

### 1. The Error Budget Framework

**Total Physical Error Rate:**
The effective error rate per syndrome cycle combines multiple sources:

$$\boxed{p_{\text{eff}} = p_{\text{gate}} + p_{\text{meas}} + p_{\text{idle}} + p_{\text{leak}} + \cdots}$$

**Component Breakdown:**

| Error Source | Typical Range | Description |
|--------------|---------------|-------------|
| $p_{\text{gate}}$ | 0.1% - 1% | Two-qubit gate errors |
| $p_{\text{meas}}$ | 0.5% - 3% | Measurement errors |
| $p_{\text{idle}}$ | 0.01% - 0.1% | Decoherence during idle |
| $p_{\text{leak}}$ | 0.01% - 0.1% | Leakage out of computational space |
| $p_{\text{crosstalk}}$ | 0.01% - 0.5% | Unwanted qubit interactions |

### 2. The Error Correction Threshold

**Threshold Definition:**
The threshold $p_{th}$ is the physical error rate below which increasing code distance reduces logical error rate:

$$\boxed{p < p_{th} \Rightarrow p_L \xrightarrow{d \to \infty} 0}$$

**Surface Code Thresholds:**

| Noise Model | Threshold $p_{th}$ |
|-------------|-------------------|
| Phenomenological (ideal measurement) | ~11% |
| Code capacity (perfect gates) | ~15% |
| Circuit-level (realistic) | ~0.5% - 1% |

**Key Insight:** The circuit-level threshold accounts for errors during syndrome extraction, making it the relevant metric for real hardware.

### 3. Logical Error Rate Scaling

Below threshold, the logical error rate scales as:

$$\boxed{p_L \approx A \left(\frac{p}{p_{th}}\right)^{\lceil (d+1)/2 \rceil}}$$

Where:
- $p_L$ = logical error rate per syndrome cycle
- $p$ = effective physical error rate
- $p_{th}$ = error threshold
- $d$ = code distance
- $A$ = constant prefactor (typically 0.03 - 0.3)

**Alternative Form:**
$$p_L \approx A \cdot \left(\frac{p}{p_{th}}\right)^{t+1}$$

where $t = \lfloor (d-1)/2 \rfloor$ is the number of correctable errors.

### 4. The Distance Selection Criterion

**Fundamental Relationship:**

$$\boxed{d = 2t + 1}$$

A code of distance $d$ can correct up to $t$ errors.

**Choosing Distance for Target $p_L$:**

Given target logical error rate $p_L^{\text{target}}$ and physical error rate $p$:

$$\frac{d+1}{2} \geq \frac{\log(p_L^{\text{target}}/A)}{\log(p/p_{th})}$$

Solving for minimum $d$:

$$\boxed{d_{\min} = 2 \left\lceil \frac{\log(A/p_L^{\text{target}})}{\log(p_{th}/p)} \right\rceil - 1}$$

### 5. Example Distance Calculations

**Google Willow Parameters (2024):**
- Physical error rate: $p \approx 0.3\%$
- Threshold: $p_{th} \approx 1\%$
- Ratio: $p/p_{th} = 0.3$

**Target:** $p_L = 10^{-6}$ (one error per million cycles)

$$\frac{d+1}{2} \geq \frac{\log(10^{-6}/0.1)}{\log(0.3)} = \frac{\log(10^{-5})}{\log(0.3)} = \frac{-5}{-0.52} \approx 9.6$$

$$d_{\min} = 2 \times 10 - 1 = 19$$

**Qubit Count:** $n = 2d^2 - 1 = 2(361) - 1 = 721$ physical qubits per logical qubit.

### 6. Error Budget Allocation

**Optimal Allocation:**
To maximize performance, allocate error budget proportionally to impact:

1. **Gate errors dominate:** Invest in higher-fidelity gates
2. **Measurement errors dominate:** Improve readout
3. **Idle errors dominate:** Shorten syndrome cycles or add dynamical decoupling

**Rule of Thumb:**
$$\boxed{p_{\text{gate}} : p_{\text{meas}} : p_{\text{idle}} \approx 3 : 2 : 1}$$

This allocation reflects typical hardware where gates are measured more precisely than readout.

### 7. Effective Error Rate Calculation

**Per-Cycle Error Rate:**

For a syndrome extraction cycle with:
- $n_{\text{CNOT}}$ = number of CNOTs per data qubit
- $n_{\text{idle}}$ = idle time steps
- 1 measurement per ancilla

$$p_{\text{cycle}} \approx n_{\text{CNOT}} \cdot p_{\text{CNOT}} + p_{\text{meas}} + n_{\text{idle}} \cdot p_{\text{idle}}$$

**For Surface Code:**
- $n_{\text{CNOT}} \approx 4$ (each data qubit participates in 4 CNOTs)
- $n_{\text{idle}} \approx 4$ (typical idle periods)

$$p_{\text{cycle}} \approx 4p_{\text{CNOT}} + p_{\text{meas}} + 4p_{\text{idle}}$$

### 8. Threshold Sensitivity Analysis

**How Threshold Depends on Parameters:**

The threshold is sensitive to:
1. **Decoder quality:** Better decoders → higher threshold
2. **Circuit structure:** More compact circuits → higher threshold
3. **Noise model:** Biased noise can increase threshold

**MWPM vs. Neural Decoders:**
| Decoder | Threshold (circuit-level) |
|---------|--------------------------|
| MWPM | ~0.5% - 0.7% |
| Union-Find | ~0.4% - 0.6% |
| Neural | ~0.8% - 1.1% |

### 9. Space-Time Trade-offs

**More Distance = More Qubits:**
$$n_{\text{physical}} = 2d^2 - 1$$

**More Rounds = More Time:**
For algorithm with $T$ logical cycles:
$$T_{\text{total}} = T \cdot t_{\text{syndrome}}$$

**Teraquop Estimate (Fowler 2012):**
To achieve $10^{12}$ logical operations at $p_L = 10^{-12}$:
- Distance: $d \approx 27$
- Physical qubits per logical: ~1,400
- For 1000 logical qubits: ~1.4 million physical qubits

### 10. Practical Distance Selection Guidelines

**Conservative Approach (Factor of Safety = 2):**
$$d_{\text{practical}} = 2 \cdot d_{\text{min}} - 1$$

**Google's Demonstrated Progression:**
| Year | Distance | Physical Error Rate | Logical Error Rate |
|------|----------|---------------------|-------------------|
| 2023 | 3 | 0.5% | 3% |
| 2023 | 5 | 0.5% | 1% |
| 2024 | 7 | 0.3% | 0.1% |

**IBM's Roadmap:**
- 2024: Demonstrations at $d = 3, 5$
- 2025: Target $d = 7$ with heavy-hex
- 2030: Target $d > 15$ for practical advantage

---

## Quantum Computing Connection

### Google Willow (2024) Error Budget

Google's breakthrough demonstrated below-threshold operation:

**Measured Parameters:**
- Two-qubit gate error: 0.2% - 0.4%
- Measurement error: 0.5% - 1%
- Idle error per μs: ~0.01%
- Syndrome cycle time: ~1 μs

**Effective Rate:** $p_{\text{eff}} \approx 0.3\%$

**Result:** Exponential suppression observed from $d=3$ to $d=7$.

### IBM Error Mitigation Strategy

IBM combines error correction with error mitigation:

1. **Zero-Noise Extrapolation (ZNE):** Amplify and extrapolate errors
2. **Probabilistic Error Cancellation (PEC):** Quasi-probability error inversion
3. **Surface codes:** For long computations requiring full fault tolerance

**Trade-off:** Mitigation extends useful range before full QEC is needed.

### The $\Lambda$ Parameter

Quantifies exponential suppression:

$$\boxed{\Lambda = \frac{p_{th}}{p}}$$

For Google Willow: $\Lambda \approx 3$

**Interpretation:** Each increase in distance by 2 reduces logical error by factor of $\Lambda \approx 3$.

---

## Worked Examples

### Example 1: Distance Selection for Quantum Advantage

**Problem:** A quantum algorithm requires $10^8$ logical operations with a target failure probability of 1%. What code distance is needed if $p = 0.5\%$ and $p_{th} = 1\%$?

**Solution:**

**Target logical error rate per operation:**
$$p_L = \frac{0.01}{10^8} = 10^{-10}$$

**Using scaling formula with $A = 0.1$:**
$$\frac{d+1}{2} = \frac{\log(A/p_L)}{\log(p_{th}/p)} = \frac{\log(0.1/10^{-10})}{\log(1/0.5)} = \frac{\log(10^9)}{\log(2)} = \frac{9}{0.301} \approx 30$$

$$d_{\min} = 2 \times 30 - 1 = 59$$

**Physical qubits per logical qubit:**
$$n = 2(59)^2 - 1 = 6,961$$

**For 100 logical qubits:** ~700,000 physical qubits

---

### Example 2: Error Budget Decomposition

**Problem:** A quantum processor has the following error rates:
- CNOT error: 0.3%
- Measurement error: 1%
- T1 time: 100 μs
- Syndrome cycle: 1 μs

Determine if it's below threshold and estimate the logical error rate at $d = 5$.

**Solution:**

**Idle error rate (per μs):**
$$p_{\text{idle}} = \frac{t_{\text{cycle}}}{T_1} = \frac{1}{100} = 1\%$$

But idle time is only a fraction of the cycle. Assume 4 idle steps of 0.1 μs each:
$$p_{\text{idle,total}} = 4 \times 0.1\% = 0.4\%$$

**Effective error rate:**
$$p_{\text{eff}} = 4 \times 0.3\% + 1\% + 0.4\% = 1.2\% + 1\% + 0.4\% = 2.6\%$$

**Threshold comparison:**
With $p_{th} \approx 1\%$, we have $p_{\text{eff}} = 2.6\% > p_{th}$.

**This is above threshold!** Increasing distance will not help.

**Recommendation:** Reduce measurement error (biggest contributor) to below 0.3% to achieve $p_{\text{eff}} < 1\%$.

---

### Example 3: Optimizing Error Allocation

**Problem:** Given a total error budget of 0.6% per cycle, how should it be allocated between gates, measurement, and idle errors to minimize logical error rate?

**Solution:**

The optimal allocation depends on how each error type affects the logical error rate.

**General Principle:** Allocate inversely proportional to "sensitivity"—error types that more easily cause logical failures should have lower allocations.

**For Surface Codes:**
- Gate errors can create correlated errors (hook errors) → higher sensitivity
- Measurement errors are detected in subsequent rounds → lower sensitivity
- Idle errors accumulate over time → moderate sensitivity

**Balanced Allocation:**
$$p_{\text{gate}} = 0.2\%, \quad p_{\text{meas}} = 0.3\%, \quad p_{\text{idle}} = 0.1\%$$

**Verification:** $0.2\% + 0.3\% + 0.1\% = 0.6\%$ ✓

**Effective Rate (with weighting):**
$$p_{\text{eff}} \approx 4 \times 0.2\% + 0.3\% + 4 \times 0.1\% = 0.8\% + 0.3\% + 0.4\% = 1.5\%$$

Hmm, this exceeds our budget because of the gate multiplication. Let's recalculate:

**Adjusted Allocation:**
$$p_{\text{CNOT}} = 0.1\%, \quad p_{\text{meas}} = 0.2\%, \quad p_{\text{idle}} = 0.05\%$$

$$p_{\text{eff}} \approx 4 \times 0.1\% + 0.2\% + 4 \times 0.05\% = 0.4\% + 0.2\% + 0.2\% = 0.8\%$$

This is below threshold, enabling error correction!

---

## Practice Problems

### Direct Application

**Problem 1:** A surface code has $p = 0.2\%$ and $p_{th} = 0.8\%$. Calculate the logical error rate for $d = 5, 7, 9$. Use $A = 0.1$.

**Problem 2:** If a logical qubit requires $p_L < 10^{-12}$ for a 1-hour computation at 1 MHz cycle rate, what code distance is needed? Assume $p = 0.3\%$, $p_{th} = 1\%$.

**Problem 3:** For a $d = 11$ surface code, calculate the number of physical qubits and the approximate physical-to-logical qubit ratio.

### Intermediate

**Problem 4:** Derive the minimum distance formula from the logical error rate scaling equation.

**Problem 5:** A hardware team can improve either gate fidelity by 2x or measurement fidelity by 3x. Given current errors of 0.3% (gate) and 1.5% (measurement), which improvement gives a larger reduction in effective error rate?

**Problem 6:** Compare the qubit overhead for achieving $p_L = 10^{-9}$ using:
a) A single $d = 15$ code
b) Concatenated $d = 5$ codes (3 levels)

### Challenging

**Problem 7:** Derive the relationship between the $\Lambda$ parameter and the number of syndrome cycles needed for reliable computation.

**Problem 8:** In the presence of biased noise (Z errors 100x more likely than X errors), how should the code be modified and how does this affect the error budget?

**Problem 9:** Design an error budget for a fault-tolerant quantum computer that achieves $10^{12}$ operations at $p_L = 10^{-12}$ using no more than 10,000 physical qubits per logical qubit.

---

## Computational Lab

### Lab 818: Error Budget Analysis and Distance Selection

```python
"""
Day 818 Computational Lab: Error Budgets and Code Distance Selection
=====================================================================

This lab implements error budget analysis, logical error rate modeling,
and distance selection optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

class ErrorBudget:
    """
    Manages error budget allocation and analysis.
    """

    def __init__(self, p_gate: float, p_meas: float, p_idle: float,
                 n_gates_per_qubit: int = 4, n_idle_per_cycle: int = 4):
        """
        Initialize error budget.

        Parameters:
        -----------
        p_gate : float
            Two-qubit gate error rate
        p_meas : float
            Measurement error rate
        p_idle : float
            Idle error rate per time step
        n_gates_per_qubit : int
            Number of 2Q gates per data qubit per syndrome cycle
        n_idle_per_cycle : int
            Number of idle time steps per cycle
        """
        self.p_gate = p_gate
        self.p_meas = p_meas
        self.p_idle = p_idle
        self.n_gates = n_gates_per_qubit
        self.n_idle = n_idle_per_cycle

    def effective_rate(self) -> float:
        """Calculate effective error rate per syndrome cycle."""
        return (self.n_gates * self.p_gate +
                self.p_meas +
                self.n_idle * self.p_idle)

    def component_breakdown(self) -> dict:
        """Return breakdown of error contributions."""
        total = self.effective_rate()
        return {
            'gate': self.n_gates * self.p_gate,
            'gate_fraction': self.n_gates * self.p_gate / total,
            'measurement': self.p_meas,
            'measurement_fraction': self.p_meas / total,
            'idle': self.n_idle * self.p_idle,
            'idle_fraction': self.n_idle * self.p_idle / total,
            'total': total
        }

    def print_summary(self):
        """Print error budget summary."""
        breakdown = self.component_breakdown()
        print("\nError Budget Summary")
        print("=" * 50)
        print(f"Gate error (per gate): {self.p_gate*100:.3f}%")
        print(f"Measurement error: {self.p_meas*100:.3f}%")
        print(f"Idle error (per step): {self.p_idle*100:.3f}%")
        print(f"\nContributions to effective rate:")
        print(f"  Gate ({self.n_gates} gates): {breakdown['gate']*100:.3f}% ({breakdown['gate_fraction']*100:.1f}%)")
        print(f"  Measurement: {breakdown['measurement']*100:.3f}% ({breakdown['measurement_fraction']*100:.1f}%)")
        print(f"  Idle ({self.n_idle} steps): {breakdown['idle']*100:.3f}% ({breakdown['idle_fraction']*100:.1f}%)")
        print(f"\nEffective rate: {breakdown['total']*100:.3f}%")


class LogicalErrorModel:
    """
    Models logical error rate as function of physical rate and distance.
    """

    def __init__(self, p_threshold: float = 0.01, prefactor: float = 0.1):
        """
        Initialize logical error model.

        Parameters:
        -----------
        p_threshold : float
            Error threshold
        prefactor : float
            Prefactor A in p_L = A * (p/p_th)^((d+1)/2)
        """
        self.p_th = p_threshold
        self.A = prefactor

    def logical_error_rate(self, p_physical: float, distance: int) -> float:
        """
        Calculate logical error rate.

        Parameters:
        -----------
        p_physical : float
            Physical (effective) error rate
        distance : int
            Code distance

        Returns:
        --------
        float
            Logical error rate per syndrome cycle
        """
        if p_physical >= self.p_th:
            # Above threshold - no improvement from distance
            return min(1.0, self.A * (p_physical / self.p_th) ** ((distance + 1) / 2))

        exponent = (distance + 1) / 2
        return self.A * (p_physical / self.p_th) ** exponent

    def lambda_parameter(self, p_physical: float) -> float:
        """Calculate Lambda = p_th / p."""
        return self.p_th / p_physical

    def minimum_distance(self, p_physical: float, target_p_L: float) -> int:
        """
        Calculate minimum distance for target logical error rate.

        Parameters:
        -----------
        p_physical : float
            Physical error rate
        target_p_L : float
            Target logical error rate

        Returns:
        --------
        int
            Minimum odd code distance
        """
        if p_physical >= self.p_th:
            return float('inf')  # Impossible

        # Solve: A * (p/p_th)^((d+1)/2) <= target_p_L
        # (d+1)/2 >= log(A/target_p_L) / log(p_th/p)
        ratio = np.log(self.A / target_p_L) / np.log(self.p_th / p_physical)
        d_min_continuous = 2 * ratio - 1

        # Round up to nearest odd integer
        d_min = int(np.ceil(d_min_continuous))
        if d_min % 2 == 0:
            d_min += 1

        return max(3, d_min)

    def physical_qubits(self, distance: int) -> int:
        """Calculate physical qubits for rotated surface code."""
        return 2 * distance ** 2 - 1


def plot_logical_error_scaling():
    """Plot logical error rate vs code distance for various physical rates."""
    model = LogicalErrorModel(p_threshold=0.01, prefactor=0.1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Logical error rate vs distance
    ax1 = axes[0]
    distances = np.arange(3, 25, 2)
    physical_rates = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011]
    colors = plt.cm.viridis(np.linspace(0, 1, len(physical_rates)))

    for p, color in zip(physical_rates, colors):
        logical_rates = [model.logical_error_rate(p, d) for d in distances]
        label = f'p = {p*100:.1f}%'
        style = '-' if p < model.p_th else '--'
        ax1.semilogy(distances, logical_rates, style, color=color, linewidth=2,
                    marker='o', markersize=5, label=label)

    ax1.axhline(y=1e-12, color='red', linestyle=':', label='Target ($10^{-12}$)')
    ax1.set_xlabel('Code Distance', fontsize=12)
    ax1.set_ylabel('Logical Error Rate', fontsize=12)
    ax1.set_title('Logical Error Rate vs Code Distance', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-15, 1)

    # Right: Physical qubits vs logical error rate
    ax2 = axes[1]
    target_rates = np.logspace(-15, -3, 50)
    physical_rate = 0.003

    distances_needed = [model.minimum_distance(physical_rate, p_L) for p_L in target_rates]
    qubits_needed = [model.physical_qubits(d) if d != float('inf') else np.nan
                    for d in distances_needed]

    ax2.loglog(target_rates, qubits_needed, 'b-', linewidth=2)
    ax2.set_xlabel('Target Logical Error Rate', fontsize=12)
    ax2.set_ylabel('Physical Qubits per Logical Qubit', fontsize=12)
    ax2.set_title(f'Qubit Overhead (p = {physical_rate*100:.1f}%)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Mark some key points
    key_targets = [1e-6, 1e-9, 1e-12]
    for target in key_targets:
        d = model.minimum_distance(physical_rate, target)
        q = model.physical_qubits(d)
        ax2.plot(target, q, 'ro', markersize=10)
        ax2.annotate(f'd={d}\n{q} qubits', (target, q),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9)

    plt.tight_layout()
    return fig


def plot_error_budget_sensitivity():
    """Analyze sensitivity of effective rate to different error components."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    base_budget = ErrorBudget(p_gate=0.003, p_meas=0.01, p_idle=0.001)
    base_effective = base_budget.effective_rate()

    # Sweep each parameter
    multipliers = np.linspace(0.1, 3, 50)

    # Gate error sensitivity
    ax1 = axes[0]
    gate_sweep = [ErrorBudget(base_budget.p_gate * m, base_budget.p_meas, base_budget.p_idle).effective_rate()
                  for m in multipliers]
    ax1.plot(multipliers, [r*100 for r in gate_sweep], 'b-', linewidth=2)
    ax1.axhline(y=base_effective*100, color='gray', linestyle='--', label='Baseline')
    ax1.axhline(y=1.0, color='red', linestyle=':', label='Threshold (1%)')
    ax1.set_xlabel('Gate Error Multiplier', fontsize=12)
    ax1.set_ylabel('Effective Error Rate (%)', fontsize=12)
    ax1.set_title('Sensitivity to Gate Error', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Measurement error sensitivity
    ax2 = axes[1]
    meas_sweep = [ErrorBudget(base_budget.p_gate, base_budget.p_meas * m, base_budget.p_idle).effective_rate()
                  for m in multipliers]
    ax2.plot(multipliers, [r*100 for r in meas_sweep], 'g-', linewidth=2)
    ax2.axhline(y=base_effective*100, color='gray', linestyle='--', label='Baseline')
    ax2.axhline(y=1.0, color='red', linestyle=':', label='Threshold (1%)')
    ax2.set_xlabel('Measurement Error Multiplier', fontsize=12)
    ax2.set_ylabel('Effective Error Rate (%)', fontsize=12)
    ax2.set_title('Sensitivity to Measurement Error', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Idle error sensitivity
    ax3 = axes[2]
    idle_sweep = [ErrorBudget(base_budget.p_gate, base_budget.p_meas, base_budget.p_idle * m).effective_rate()
                  for m in multipliers]
    ax3.plot(multipliers, [r*100 for r in idle_sweep], 'orange', linewidth=2)
    ax3.axhline(y=base_effective*100, color='gray', linestyle='--', label='Baseline')
    ax3.axhline(y=1.0, color='red', linestyle=':', label='Threshold (1%)')
    ax3.set_xlabel('Idle Error Multiplier', fontsize=12)
    ax3.set_ylabel('Effective Error Rate (%)', fontsize=12)
    ax3.set_title('Sensitivity to Idle Error', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_distance_selection_nomogram():
    """Create a nomogram for distance selection."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    model = LogicalErrorModel(p_threshold=0.01, prefactor=0.1)

    # Physical error rates
    p_values = [0.001, 0.002, 0.003, 0.005, 0.007]
    # Target logical error rates
    target_values = np.logspace(-15, -3, 100)

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(p_values)))

    for p, color in zip(p_values, colors):
        distances = [model.minimum_distance(p, t) for t in target_values]
        valid = [d != float('inf') for d in distances]
        ax.semilogy([d for d, v in zip(distances, valid) if v],
                   [t for t, v in zip(target_values, valid) if v],
                   '-', color=color, linewidth=2, label=f'p = {p*100:.1f}%')

    ax.set_xlabel('Code Distance', fontsize=14)
    ax.set_ylabel('Achievable Logical Error Rate', fontsize=14)
    ax.set_title('Distance Selection Nomogram', fontsize=16)
    ax.legend(title='Physical Error Rate')
    ax.grid(True, alpha=0.3)

    # Add useful reference lines
    for d in [5, 7, 9, 11, 13, 15, 17, 19, 21]:
        ax.axvline(x=d, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlim(3, 25)
    ax.set_ylim(1e-15, 1e-3)

    plt.tight_layout()
    return fig


def analyze_teraquop_requirements():
    """Analyze requirements for teraquop-scale computation."""
    print("\nTeraquop Analysis (10^12 operations at p_L = 10^{-12})")
    print("=" * 60)

    model = LogicalErrorModel(p_threshold=0.01, prefactor=0.1)

    physical_rates = [0.001, 0.002, 0.003, 0.005]

    print(f"\n{'p_phys':<10} {'Distance':<10} {'Qubits/logical':<18} {'1000 logicals':<15}")
    print("-" * 55)

    for p in physical_rates:
        # For 10^12 operations at 10^-12 per operation
        target_p_L = 1e-12
        d = model.minimum_distance(p, target_p_L)
        qubits = model.physical_qubits(d)

        print(f"{p*100:.2f}%{'':<5} {d:<10} {qubits:<18,} {qubits*1000:>15,}")

    # Cost analysis
    print("\n" + "=" * 60)
    print("Resource Estimates for Practical Quantum Advantage")
    print("=" * 60)

    algorithms = [
        ("Shor (2048-bit)", 1e9, 4000, 1e-10),
        ("Grover (database)", 1e6, 100, 1e-6),
        ("VQE (chemistry)", 1e8, 200, 1e-8),
        ("Simulation (materials)", 1e10, 1000, 1e-10),
    ]

    print(f"\n{'Algorithm':<25} {'Operations':<12} {'Logicals':<10} {'p_L':<10} {'Qubits (p=0.3%)':<15}")
    print("-" * 75)

    for name, ops, n_logical, p_L in algorithms:
        target = p_L / ops  # Per-operation requirement
        d = model.minimum_distance(0.003, target)
        qubits_per = model.physical_qubits(d)
        total = qubits_per * n_logical

        print(f"{name:<25} {ops:<12.0e} {n_logical:<10} {p_L:<10.0e} {total:>15,}")


# Main execution
if __name__ == "__main__":
    print("Error Budget and Distance Selection Analysis")
    print("=" * 60)

    # Create and analyze an error budget
    budget = ErrorBudget(p_gate=0.003, p_meas=0.01, p_idle=0.001)
    budget.print_summary()

    # Logical error model
    model = LogicalErrorModel(p_threshold=0.01, prefactor=0.1)
    print(f"\nλ = p_th/p = {model.lambda_parameter(budget.effective_rate()):.2f}")

    # Distance selection examples
    print("\nDistance Selection Examples:")
    targets = [1e-6, 1e-9, 1e-12, 1e-15]
    for target in targets:
        d = model.minimum_distance(budget.effective_rate(), target)
        q = model.physical_qubits(d)
        p_L = model.logical_error_rate(budget.effective_rate(), d)
        print(f"  Target: {target:.0e} → d = {d}, qubits = {q}, achieved = {p_L:.2e}")

    # Generate plots
    fig1 = plot_logical_error_scaling()
    plt.savefig('logical_error_scaling.png', dpi=150, bbox_inches='tight')
    print("\nSaved logical_error_scaling.png")

    fig2 = plot_error_budget_sensitivity()
    plt.savefig('error_budget_sensitivity.png', dpi=150, bbox_inches='tight')
    print("Saved error_budget_sensitivity.png")

    fig3 = plot_distance_selection_nomogram()
    plt.savefig('distance_nomogram.png', dpi=150, bbox_inches='tight')
    print("Saved distance_nomogram.png")

    # Teraquop analysis
    analyze_teraquop_requirements()

    plt.show()
```

### Lab Exercises

1. **Create an optimizer** that finds the best error budget allocation for a fixed total budget.

2. **Model the effect of decoder latency** on the effective error rate.

3. **Calculate the space-time cost** (qubits × time) for different algorithm requirements.

4. **Compare concatenated codes** versus single high-distance codes for the same target error rate.

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Correctable errors | $t = \lfloor (d-1)/2 \rfloor$ |
| Distance criterion | $d = 2t + 1$ |
| Logical error rate | $p_L \approx A(p/p_{th})^{(d+1)/2}$ |
| $\Lambda$ parameter | $\Lambda = p_{th}/p$ |
| Minimum distance | $d_{min} = 2\lceil \log(A/p_L)/\log(\Lambda) \rceil - 1$ |
| Physical qubits | $n = 2d^2 - 1$ |

### Main Takeaways

1. **Error budget decomposes:** Total error comes from gates, measurement, and idle contributions.

2. **Threshold is critical:** Below threshold, increasing $d$ exponentially reduces $p_L$.

3. **Distance selection is quantitative:** Given $p$ and target $p_L$, minimum $d$ is calculable.

4. **Qubit overhead grows:** Achieving $p_L = 10^{-12}$ requires hundreds to thousands of qubits per logical qubit.

5. **$\Lambda$ determines scaling:** Each distance increase by 2 improves $p_L$ by factor $\Lambda$.

---

## Daily Checklist

- [ ] I can decompose an error budget into gate, measurement, and idle components
- [ ] I understand the significance of the error threshold
- [ ] I can calculate the minimum distance for a target logical error rate
- [ ] I can estimate qubit overhead for practical quantum computation
- [ ] I have run the computational lab and generated error scaling plots

---

## Preview: Day 819

Tomorrow is **Week 117 Synthesis Day**. We'll:
- Integrate all architectural concepts from this week
- Build a comprehensive surface code architecture analyzer
- Compare different design choices quantitatively
- Prepare for lattice surgery in Week 118

This synthesis brings together geometry, boundaries, connectivity, and error budgets into a unified framework.

---

*"Error budgets are the currency of fault-tolerant quantum computing—spend them wisely."*

— Day 818 Reflection
