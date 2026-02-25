# Day 808: Logical Error Rate Scaling

## Year 2, Semester 2A: Quantum Error Correction
### Month 29: Topological Codes | Week 116: Error Chains & Logical Operations

---

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3 hours | Threshold theory and scaling laws |
| Afternoon | 2.5 hours | Resource overhead and tradeoffs |
| Evening | 1.5 hours | Computational lab: threshold simulation |

---

## Learning Objectives

By the end of today, you will be able to:

1. Derive the logical error rate scaling formula below threshold
2. Distinguish phenomenological from circuit-level noise models
3. Calculate threshold values for different noise models
4. Analyze qubit overhead for target logical error rates
5. Compare surface codes to concatenated codes
6. Optimize code distance for given physical error rates
7. Perform threshold simulations numerically

---

## Core Content: Error Rate Scaling

### The Fundamental Scaling Law

Below threshold, the logical error rate decreases **exponentially** with code distance:

$$\boxed{p_L \approx A \left(\frac{p}{p_{th}}\right)^{(d+1)/2}}$$

where:
- $p_L$ = logical error rate per round
- $p$ = physical error rate
- $p_{th}$ = threshold error rate
- $d$ = code distance
- $A$ = code-dependent constant

**Key insight:** Each additional pair of distance suppresses errors by factor $p/p_{th}$.

### Why $(d+1)/2$?

A logical error requires a non-trivial homology class, which needs:
- Minimum $d$ errors to span the code
- But decoder corrects up to $\lfloor(d-1)/2\rfloor$ errors
- Logical error needs $\lceil(d+1)/2\rceil$ errors that "fool" the decoder

**Combinatorial argument:**

$$\boxed{p_L \sim \binom{d}{\lceil(d+1)/2\rceil} p^{(d+1)/2} (1-p)^{d-(d+1)/2}}$$

For $p \ll 1$:

$$p_L \sim C \cdot p^{(d+1)/2}$$

### Threshold Definition

The **threshold** $p_{th}$ is where logical and physical error rates are equal:

$$\boxed{p_{th}: \quad p_L(p_{th}, d) = p_{th} \text{ for } d \to \infty}$$

Below threshold ($p < p_{th}$): increasing $d$ improves reliability
Above threshold ($p > p_{th}$): increasing $d$ makes things worse!

---

## Noise Models and Thresholds

### Code Capacity Model

**Assumptions:**
- Errors only on data qubits
- Perfect syndrome measurement
- No measurement errors

**Surface code threshold:**

$$\boxed{p_{th}^{\text{code capacity}} \approx 10.9\%}$$

This matches the percolation threshold for independent errors on 2D lattice.

### Phenomenological Model

**Assumptions:**
- Errors on data qubits (probability $p$)
- Syndrome measurement errors (probability $p$)
- Multiple syndrome rounds

**Surface code threshold:**

$$\boxed{p_{th}^{\text{phenomenological}} \approx 2.9\%}$$

Measurement errors create "time-like" error chains.

### Circuit-Level Model

**Assumptions:**
- All operations have errors
- CNOT gates: depolarizing with probability $p$
- Single-qubit gates: error probability $p$
- Measurements: error probability $p$
- State preparation: error probability $p$

**Surface code threshold:**

$$\boxed{p_{th}^{\text{circuit}} \approx 0.5\% - 1.0\%}$$

The exact value depends on:
- Gate set (CZ vs CNOT)
- Scheduling (parallel vs sequential)
- Decoder (MWPM vs Union-Find vs neural)

### Threshold Comparison Table

| Noise Model | Assumptions | Threshold | Realism |
|-------------|-------------|-----------|---------|
| Code capacity | Perfect measurements | ~10.9% | Unrealistic |
| Phenomenological | Noisy measurements | ~2.9% | Moderate |
| Circuit-level (std) | Full circuit noise | ~0.7% | Realistic |
| Circuit-level (opt) | Optimized circuits | ~1.0% | Best case |

---

## Resource Overhead Analysis

### Qubit Overhead

For distance-$d$ planar surface code:

$$\boxed{n = 2d^2 - 1 \approx 2d^2 \text{ physical qubits per logical qubit}}$$

To achieve logical error rate $p_L$ with physical error rate $p$:

$$d \approx 2\log_{p/p_{th}}(p_L/A) + 1$$

### Example Calculation

**Target:** $p_L = 10^{-15}$ (for useful quantum computation)
**Physical:** $p = 10^{-3}$, $p_{th} = 10^{-2}$

$$\frac{d+1}{2} \log\left(\frac{p}{p_{th}}\right) = \log(p_L)$$

$$\frac{d+1}{2} \cdot \log(0.1) = \log(10^{-15})$$

$$\frac{d+1}{2} = \frac{-15}{-1} = 15$$

$$d = 29$$

**Qubits needed:** $n \approx 2 \times 29^2 = 1682$ per logical qubit!

### Overhead Scaling Table

| Target $p_L$ | $d$ (for $p=10^{-3}$) | Qubits per logical |
|--------------|----------------------|-------------------|
| $10^{-6}$ | 11 | ~242 |
| $10^{-9}$ | 17 | ~578 |
| $10^{-12}$ | 23 | ~1058 |
| $10^{-15}$ | 29 | ~1682 |

### Space-Time Tradeoff

Surface code operations take **$O(d)$ syndrome rounds**.

Total space-time overhead for one logical operation:

$$\boxed{\text{ST overhead} = O(d^3) \text{ qubit-rounds per logical qubit-operation}}$$

---

## Comparison with Other Approaches

### Surface Code vs Concatenated Codes

| Property | Surface Code | Concatenated (7-qubit) |
|----------|--------------|------------------------|
| Threshold | ~0.7% | ~0.01% (without tricks) |
| Scaling | $p_L \sim (p/p_{th})^{d/2}$ | $p_L \sim p^{2^L}$ |
| Overhead | $O(d^2)$ qubits | $O(7^L) = O(\log^c(1/p_L))$ |
| Connectivity | 2D local | Non-local (typically) |
| Gates | Lattice surgery | Transversal |

**Surface code advantages:**
- Higher threshold (by ~70×)
- 2D local connectivity (hardware friendly)

**Concatenated code advantages:**
- Faster double-exponential scaling
- Simpler logical gates

### Break-Even Analysis

When is surface code better than concatenation?

For physical error rate $p$ and target $p_L$:

**Surface code:** $n_S \approx 2 \cdot \left(\frac{\log(1/p_L)}{\log(p_{th}/p)}\right)^2$

**Concatenated:** $n_C \approx 7^{\log_2 \log(1/p_L)}$

Surface code wins for moderate $p_L$ and $p$ close to threshold.

---

## Worked Examples

### Example 1: Distance Selection

**Problem:** Select code distance for $p = 0.1\%$, target $p_L = 10^{-10}$.

**Solution:**

Given $p_{th} \approx 1\%$:

$$p_L \approx \left(\frac{0.001}{0.01}\right)^{(d+1)/2} = (0.1)^{(d+1)/2}$$

Setting $p_L = 10^{-10}$:

$$(0.1)^{(d+1)/2} = 10^{-10}$$

$$\frac{d+1}{2} = 10$$

$$\boxed{d = 19}$$

**Qubits:** $n \approx 2 \times 19^2 = 722$ physical qubits.

---

### Example 2: Threshold Estimation from Data

**Data:** Logical error rates for different distances at $p = 0.5\%$:
- $d=3$: $p_L = 0.045$
- $d=5$: $p_L = 0.012$
- $d=7$: $p_L = 0.0032$

**Find threshold and scaling:**

Taking ratios:
$$\frac{p_L(5)}{p_L(3)} = \frac{0.012}{0.045} = 0.267 \approx \left(\frac{p}{p_{th}}\right)^{(5-3)/2} = \left(\frac{p}{p_{th}}\right)$$

$$\frac{p}{p_{th}} \approx 0.267$$

$$p_{th} \approx \frac{0.005}{0.267} \approx 1.87\%$$

**Verification with $d=7$:**
$$p_L(7) \approx (0.267)^{(7+1)/2 - (3+1)/2} \cdot p_L(3) = (0.267)^2 \cdot 0.045 = 0.0032 \quad \checkmark$$

---

### Example 3: Memory Lifetime

**Problem:** How long can a surface code store a qubit with 99% fidelity?

**Setup:** $d = 17$, $p = 0.1\%$, $p_{th} = 0.7\%$

**Logical error per round:**
$$p_L \approx \left(\frac{0.001}{0.007}\right)^{(17+1)/2} = (0.143)^9 \approx 2.6 \times 10^{-8}$$

**Rounds for 1% failure:**
$$N \cdot p_L \leq 0.01$$
$$N \leq \frac{0.01}{2.6 \times 10^{-8}} \approx 385,000 \text{ rounds}$$

**Memory time:** At 1 MHz cycle rate, lifetime $\approx 385$ seconds!

---

## Practice Problems

### Problem Set A: Scaling Calculations

**A1.** For $p = 0.2\%$ and $p_{th} = 1.0\%$:
(a) Calculate $p_L$ for $d = 5, 7, 9, 11$
(b) Plot $\log(p_L)$ vs $d$
(c) Verify the slope matches $(1/2)\log(p/p_{th})$

**A2.** A quantum algorithm requires $10^{12}$ logical operations with total failure probability $<1\%$.
(a) What logical error rate per operation is needed?
(b) What code distance suffices for $p = 0.1\%$?
(c) How many physical qubits per logical qubit?

**A3.** Derive the prefactor $A$ in $p_L \approx A(p/p_{th})^{(d+1)/2}$ from the binomial formula for $d=5$.

### Problem Set B: Threshold Analysis

**B1.** Simulation data shows:

| $p$ | $p_L(d=5)$ | $p_L(d=9)$ |
|-----|------------|------------|
| 0.5% | 0.010 | 0.0008 |
| 1.0% | 0.025 | 0.0050 |
| 1.5% | 0.045 | 0.015 |

Estimate the threshold by finding where $p_L(d=5) = p_L(d=9)$.

**B2.** Prove that at threshold, $p_L$ is independent of $d$ (to leading order).

**B3.** The circuit-level threshold is lower than phenomenological because:
- CNOT gates spread errors
- Measurement circuits have multiple failure points

Estimate the threshold reduction factor if each syndrome extraction circuit has 10 locations where errors can occur.

### Problem Set C: Resource Optimization

**C1.** For a fixed number of physical qubits $N$, should you use:
- Fewer large codes: $k$ logical qubits at distance $d = \sqrt{N/(2k)}$
- More small codes: $2k$ logical qubits at distance $d' = d/\sqrt{2}$

Analyze when each is better.

**C2.** Magic state distillation has overhead $O(\log^c(1/\epsilon))$ for target error $\epsilon$. If surface code has overhead $O(d^2)$ and $d \sim \log(1/\epsilon)$, what is the total overhead for a T-gate?

**C3.** Compare surface code memory lifetime with classical hard drive bit flip rates ($\sim 10^{-15}$ per bit per year). What parameters make quantum memory competitive?

---

## Computational Lab: Threshold Simulation

```python
"""
Day 808 Computational Lab: Logical Error Rate Scaling
Simulating threshold behavior and resource estimates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import comb
from typing import List, Tuple

class ThresholdSimulator:
    """
    Simulate logical error rates for surface codes.

    Uses simplified error model for pedagogical purposes.
    """

    def __init__(self, p_th: float = 0.01):
        """
        Initialize threshold simulator.

        Args:
            p_th: Threshold error rate
        """
        self.p_th = p_th

    def logical_error_rate_analytic(self, p: float, d: int) -> float:
        """
        Analytic formula for logical error rate.

        Uses binomial approximation for errors.

        Args:
            p: Physical error rate
            d: Code distance

        Returns:
            Estimated logical error rate
        """
        # Number of errors needed for logical failure
        n_errors = (d + 1) // 2

        # Approximate prefactor
        A = 0.1

        # Leading order scaling
        p_L = A * (p / self.p_th) ** n_errors

        return min(p_L, 0.5)  # Cap at 0.5

    def logical_error_rate_binomial(self, p: float, d: int) -> float:
        """
        More accurate binomial estimate.

        Considers exact number of error configurations.
        """
        # Minimum errors for logical failure
        t = (d - 1) // 2  # Correctable errors
        min_errors = t + 1

        # Sum over all error numbers >= min_errors
        p_L = 0
        for k in range(min_errors, d + 1):
            # Number of error configurations of weight k that cause logical error
            # Approximate as fraction of all weight-k errors
            fraction_logical = 0.1  # Rough estimate

            # Probability of exactly k errors
            p_k = comb(d, k, exact=True) * (p ** k) * ((1 - p) ** (d - k))

            p_L += fraction_logical * p_k

        return min(p_L, 0.5)

    def simulate_threshold_crossing(self, p_values: np.ndarray,
                                    d_values: List[int]) -> dict:
        """
        Simulate logical error rates across threshold.

        Args:
            p_values: Array of physical error rates
            d_values: List of code distances

        Returns:
            Dictionary with results
        """
        results = {'p': p_values, 'd': d_values, 'p_L': {}}

        for d in d_values:
            p_L = [self.logical_error_rate_analytic(p, d) for p in p_values]
            results['p_L'][d] = np.array(p_L)

        return results

    def estimate_threshold_from_data(self, p_values: np.ndarray,
                                     p_L_d1: np.ndarray,
                                     p_L_d2: np.ndarray,
                                     d1: int, d2: int) -> float:
        """
        Estimate threshold from crossing point of two distances.

        At threshold, p_L(d1) = p_L(d2).
        """
        # Find crossing point by interpolation
        diff = p_L_d1 - p_L_d2

        # Find sign change
        for i in range(len(diff) - 1):
            if diff[i] * diff[i+1] < 0:
                # Linear interpolation
                p_th = p_values[i] - diff[i] * (p_values[i+1] - p_values[i]) / (diff[i+1] - diff[i])
                return p_th

        return None


def plot_threshold_crossing():
    """Visualize threshold crossing behavior."""
    sim = ThresholdSimulator(p_th=0.01)

    p_values = np.linspace(0.001, 0.025, 100)
    d_values = [3, 5, 7, 9, 11, 13]

    results = sim.simulate_threshold_crossing(p_values, d_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Linear scale
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(d_values)))

    for i, d in enumerate(d_values):
        ax1.plot(p_values * 100, results['p_L'][d], '-', color=colors[i],
                linewidth=2, label=f'd={d}')

    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax1.plot(p_values * 100, p_values, 'k:', alpha=0.5, label='p_L = p')

    ax1.set_xlabel('Physical Error Rate p (%)', fontsize=12)
    ax1.set_ylabel('Logical Error Rate $p_L$', fontsize=12)
    ax1.set_title('Threshold Crossing (Linear Scale)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 0.5)

    # Panel 2: Log scale
    ax2 = axes[1]

    for i, d in enumerate(d_values):
        ax2.semilogy(p_values * 100, results['p_L'][d], '-', color=colors[i],
                    linewidth=2, label=f'd={d}')

    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Threshold')

    ax2.set_xlabel('Physical Error Rate p (%)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate $p_L$ (log)', fontsize=12)
    ax2.set_title('Threshold Crossing (Log Scale)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2.5)

    plt.tight_layout()
    plt.savefig('threshold_crossing.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_scaling_below_threshold():
    """Visualize exponential suppression below threshold."""
    sim = ThresholdSimulator(p_th=0.01)

    d_values = np.arange(3, 25, 2)
    p_values = [0.001, 0.002, 0.003, 0.005, 0.007]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(p_values)))

    for i, p in enumerate(p_values):
        p_L = [sim.logical_error_rate_analytic(p, d) for d in d_values]
        ax.semilogy(d_values, p_L, 'o-', color=colors[i],
                   linewidth=2, markersize=8, label=f'p={p*100:.1f}%')

    ax.set_xlabel('Code Distance d', fontsize=14)
    ax.set_ylabel('Logical Error Rate $p_L$', fontsize=14)
    ax.set_title('Exponential Suppression Below Threshold\n$p_L \\sim (p/p_{th})^{(d+1)/2}$', fontsize=14)
    ax.legend(title='Physical Error Rate')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(d_values)

    plt.tight_layout()
    plt.savefig('scaling_below_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()


def resource_overhead_analysis():
    """Analyze qubit overhead for different targets."""
    p_th = 0.01
    p_values = [0.001, 0.002, 0.005]
    target_p_L = np.logspace(-6, -15, 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Distance vs target error rate
    ax1 = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(p_values)))

    for i, p in enumerate(p_values):
        distances = []
        for pL in target_p_L:
            # Solve (p/p_th)^((d+1)/2) = pL
            # (d+1)/2 = log(pL) / log(p/p_th)
            ratio = p / p_th
            d = 2 * np.log(pL) / np.log(ratio) - 1
            d = max(3, int(np.ceil(d)))
            if d % 2 == 0:
                d += 1
            distances.append(d)

        ax1.semilogx(target_p_L, distances, '-', color=colors[i],
                    linewidth=2, label=f'p={p*100:.1f}%')

    ax1.set_xlabel('Target Logical Error Rate $p_L$', fontsize=12)
    ax1.set_ylabel('Required Code Distance d', fontsize=12)
    ax1.set_title('Code Distance Scaling', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Panel 2: Physical qubits vs target error rate
    ax2 = axes[1]

    for i, p in enumerate(p_values):
        qubits = []
        for pL in target_p_L:
            ratio = p / p_th
            d = 2 * np.log(pL) / np.log(ratio) - 1
            d = max(3, int(np.ceil(d)))
            if d % 2 == 0:
                d += 1
            n = 2 * d * d - 1
            qubits.append(n)

        ax2.loglog(target_p_L, qubits, '-', color=colors[i],
                  linewidth=2, label=f'p={p*100:.1f}%')

    ax2.set_xlabel('Target Logical Error Rate $p_L$', fontsize=12)
    ax2.set_ylabel('Physical Qubits per Logical Qubit', fontsize=12)
    ax2.set_title('Qubit Overhead Scaling', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig('resource_overhead.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_noise_models():
    """Compare different noise model thresholds."""
    noise_models = {
        'Code Capacity': 0.109,
        'Phenomenological': 0.029,
        'Circuit-Level (Standard)': 0.007,
        'Circuit-Level (Optimized)': 0.010
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # For each noise model, plot scaling curve
    p_below = np.linspace(0.0001, 0.02, 100)
    d = 11

    colors = plt.cm.Set2(np.linspace(0, 1, len(noise_models)))

    for i, (name, p_th) in enumerate(noise_models.items()):
        p_L = []
        for p in p_below:
            if p < p_th:
                ratio = p / p_th
                pL = 0.1 * ratio ** ((d + 1) / 2)
            else:
                pL = 0.5
            p_L.append(pL)

        ax.semilogy(p_below * 100, p_L, '-', color=colors[i],
                   linewidth=2, label=f'{name} ($p_{{th}}$={p_th*100:.1f}%)')
        ax.axvline(x=p_th * 100, color=colors[i], linestyle='--', alpha=0.5)

    ax.set_xlabel('Physical Error Rate p (%)', fontsize=12)
    ax.set_ylabel('Logical Error Rate $p_L$ (d=11)', fontsize=12)
    ax.set_title('Noise Model Comparison\nHigher threshold = larger operating range', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)

    plt.tight_layout()
    plt.savefig('noise_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run scaling analysis demonstrations."""
    print("=" * 70)
    print("DAY 808: LOGICAL ERROR RATE SCALING")
    print("=" * 70)

    # Key scaling formula
    print("\n1. The Fundamental Scaling Law")
    print("-" * 40)
    print("""
    Below threshold, logical error rate scales as:

    ┌─────────────────────────────────────────┐
    │                 (d+1)/2                 │
    │   p_L ≈ A ( p / p_th )                  │
    │                                         │
    │   p     = physical error rate           │
    │   p_th  = threshold (~0.7% circuit)     │
    │   d     = code distance                 │
    │   A     ≈ 0.1 (code-dependent)          │
    └─────────────────────────────────────────┘

    Key insight: Each +2 in distance reduces p_L by factor (p/p_th)
    """)

    # Numerical example
    print("\n2. Numerical Example")
    print("-" * 40)
    print("Parameters: p = 0.1%, p_th = 1%")
    print("\n Distance |  p_L estimate  | Qubits")
    print("-" * 40)

    p, p_th = 0.001, 0.01
    A = 0.1

    for d in [3, 5, 7, 9, 11, 13, 17, 21]:
        p_L = A * (p / p_th) ** ((d + 1) / 2)
        n = 2 * d * d - 1
        print(f"    {d:2d}    |  {p_L:.2e}       | {n:5d}")

    # Threshold values
    print("\n3. Threshold Summary")
    print("-" * 40)
    print("""
    ┌───────────────────────────────┬───────────┐
    │ Noise Model                   │ Threshold │
    ├───────────────────────────────┼───────────┤
    │ Code capacity (perfect meas.) │   ~10.9%  │
    │ Phenomenological              │   ~2.9%   │
    │ Circuit-level (standard)      │   ~0.7%   │
    │ Circuit-level (optimized)     │   ~1.0%   │
    └───────────────────────────────┴───────────┘
    """)

    # Generate plots
    print("\n4. Generating visualizations...")
    plot_threshold_crossing()
    plot_scaling_below_threshold()
    resource_overhead_analysis()
    compare_noise_models()

    print("\n" + "=" * 70)
    print("Key takeaway: Surface codes enable exponential error suppression")
    print("below threshold, with practical overhead of ~1000 physical qubits")
    print("per logical qubit for realistic error rates and targets.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Logical error rate | $p_L \approx A(p/p_{th})^{(d+1)/2}$ |
| Threshold condition | $\frac{\partial p_L}{\partial d}\big|_{p=p_{th}} = 0$ |
| Qubit overhead | $n = 2d^2 - 1$ |
| Distance for target | $d \approx 2\frac{\log(p_L/A)}{\log(p/p_{th})} - 1$ |
| Circuit threshold | $p_{th} \approx 0.7\%$ |

### Threshold Values

| Noise Model | Threshold |
|-------------|-----------|
| Code capacity | 10.9% |
| Phenomenological | 2.9% |
| Circuit-level | 0.7% |

### Main Takeaways

1. **Exponential suppression**: Below threshold, each +2 distance squares the suppression factor

2. **Threshold is critical**: Operating below threshold is essential for QEC to work

3. **Circuit-level is realistic**: True threshold ~0.7% accounts for all noise sources

4. **Overhead is polynomial**: ~1000 physical qubits per logical qubit for practical targets

5. **Tradeoffs exist**: Higher distance costs more qubits but gives better protection

---

## Daily Checklist

### Morning Session (3 hours)
- [ ] Derive scaling law from combinatorial argument
- [ ] Understand threshold definition and crossing
- [ ] Study different noise models

### Afternoon Session (2.5 hours)
- [ ] Complete Problem Sets A and B
- [ ] Calculate resource requirements for target applications
- [ ] Compare with concatenated codes

### Evening Session (1.5 hours)
- [ ] Run computational lab
- [ ] Generate threshold crossing plots
- [ ] Complete Problem Set C

### Self-Assessment
1. Can you calculate $p_L$ given $p$, $d$, $p_{th}$?
2. Can you select code distance for target error rate?
3. Do you understand why circuit-level threshold is lower?
4. Can you estimate qubit overhead for practical algorithms?

---

## Preview: Day 809

Tomorrow we study **Lattice Surgery Operations**:
- Merge operation for measuring $\bar{Z}_1\bar{Z}_2$
- Split operation for preparing entangled pairs
- CNOT implementation via merge-split sequence
- Scheduling and parallelism

---

*Day 808 of 2184 | Year 2, Month 29, Week 116 | Quantum Engineering PhD Curriculum*
