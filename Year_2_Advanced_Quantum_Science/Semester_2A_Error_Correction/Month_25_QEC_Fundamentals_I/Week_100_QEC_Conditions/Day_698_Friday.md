# Day 698: The Threshold Theorem

## Overview

**Week:** 100 (QEC Conditions)
**Day:** Friday
**Date:** Year 2, Month 25, Day 698
**Topic:** Fault-Tolerant Quantum Computing and the Threshold Theorem
**Hours:** 7 (3.5 theory + 2.5 problems + 1 computational lab)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Threshold theorem statement and proof outline |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Threshold values and experimental progress |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Threshold simulation |

---

## Prerequisites

From Days 694-697:
- Quantum bounds and code parameters
- Degeneracy and approximate QEC
- Stabilizer codes (Shor, Steane)

---

## Learning Objectives

By the end of this day, you will be able to:

1. **State** the threshold theorem precisely
2. **Explain** the physical significance of error thresholds
3. **Describe** the concatenation argument for threshold existence
4. **Compare** threshold values for different code families
5. **Analyze** experimental progress toward sub-threshold operation
6. **Calculate** resource requirements for target logical error rates

---

## Core Content

### 1. The Fundamental Question

#### The Problem

**Question:** Can we perform arbitrarily long quantum computations despite physical noise?

**Naive answer:** No — errors accumulate, coherence decays, quantum information is lost.

**Surprising truth:** Yes! — if the physical error rate is below a threshold.

#### The Key Insight

With quantum error correction:
- Errors can be detected and corrected
- Logical qubits are more reliable than physical qubits
- **Concatenation** amplifies this advantage exponentially

---

### 2. Statement of the Threshold Theorem

#### Theorem (Aharonov-Ben-Or, Kitaev, 1997)

**Threshold Theorem:**

There exists a threshold error rate $p_{th} > 0$ such that if the physical error rate per gate/qubit satisfies:

$$\boxed{p < p_{th}}$$

then any quantum computation can be performed with arbitrarily small logical error rate $\epsilon$ using:

$$\boxed{O\left(\text{poly}\log(1/\epsilon)\right)}$$

overhead in physical qubits and gates.

#### Quantitative Version

For concatenated codes with base physical error rate $p$:

$$\boxed{\epsilon(L) \approx \left(\frac{p}{p_{th}}\right)^{2^L}}$$

where $L$ is the concatenation level.

---

### 3. Physical Interpretation

#### Below Threshold ($p < p_{th}$)

- Error correction **wins** against noise
- Each concatenation level reduces error exponentially
- Arbitrary precision achievable with polynomial overhead

#### At Threshold ($p = p_{th}$)

- Error correction **balances** noise
- Logical error rate equals physical error rate
- No improvement from encoding

#### Above Threshold ($p > p_{th}$)

- Noise **wins** against error correction
- Encoding makes things worse!
- Quantum computation not scalable

---

### 4. The Concatenation Argument

#### Level-0: Physical Qubits

Error rate: $p_0 = p$

#### Level-1: Single Encoding

Using a distance-3 code (corrects 1 error):
- Logical error occurs if ≥2 physical errors
- Probability: $p_1 \approx c \cdot p^2$ for some constant $c$

#### Level-L: L-fold Concatenation

Recursively encoding:
$$p_L \approx c \cdot p_{L-1}^2 = c \cdot (c \cdot p_{L-2}^2)^2 = \ldots$$

**Result:**
$$p_L \approx \frac{1}{c}\left(c \cdot p\right)^{2^L}$$

#### Threshold Condition

For improvement: $p_L < p_{L-1}$

Requires: $c \cdot p < 1$, i.e., $p < p_{th} = 1/c$

**Key insight:** The threshold $p_{th} = 1/c$ depends on the code and fault-tolerant gadget design.

---

### 5. Threshold Values

#### Concatenated Codes

| Code | Threshold | Notes |
|------|-----------|-------|
| Steane [[7,1,3]] | $\sim 10^{-4}$ | Early estimates |
| Bacon-Shor | $\sim 10^{-4}$ | Subsystem code |
| Golay [[23,1,7]] | $\sim 10^{-3}$ | Higher distance helps |

#### Topological/Surface Codes

| Code | Threshold | Notes |
|------|-----------|-------|
| Toric code | $\sim 11\%$ | Ideal (phenomenological) |
| Surface code (circuit) | $\sim 1\%$ | Realistic circuits |
| Surface code (experimental) | $\sim 0.7\%$ | Google Willow 2024 |

#### Threshold Hierarchy

$$p_{th}^{concat} \ll p_{th}^{surface} \approx 1\%$$

Surface codes have **much higher thresholds** — a major reason for their dominance.

---

### 6. Fault-Tolerant Operations

#### What Must Be Fault-Tolerant?

1. **State preparation:** Create encoded $|0_L\rangle$, $|+_L\rangle$
2. **Gate application:** Logical gates on encoded qubits
3. **Syndrome extraction:** Measure stabilizers without spreading errors
4. **Measurement:** Read out logical qubits

#### Transversal Gates

A gate is **transversal** if it can be implemented qubit-by-qubit:

$$\bar{U} = U^{\otimes n}$$

**Advantage:** Errors don't spread between qubits!

**Example:** For Steane code: CNOT, H, S, T (with tricks) are transversal.

#### The Eastin-Knill Theorem

**Theorem:** No quantum code can have a universal set of transversal gates.

**Implication:** Achieving universal fault-tolerant computation requires additional techniques (magic state distillation, code switching).

---

### 7. Experimental Progress

#### Google Quantum AI (December 2024)

**Willow Processor:**
- 105 superconducting qubits
- Surface code with distance 3, 5, 7
- **Key result:** Logical error rate *decreases* with code distance
- Factor 2.14× improvement per distance increase
- First demonstration of "below threshold" scaling

#### IBM Quantum (2024)

**Heron Processor:**
- Distance-5 surface codes
- Logical error rate: 0.068% per round
- 14 syndrome extraction cycles sustained

#### Quantinuum (2025)

**H2 Trapped-Ion System:**
- 12 logical qubits
- Error rate: 0.0011 (22× better than physical)
- Highest-fidelity logical operations to date

---

### 8. Resource Scaling

#### Overhead Formula

For target logical error rate $\epsilon$:

**Concatenated codes:**
$$\text{Qubits} \approx n_0 \cdot \left(\log(1/\epsilon)\right)^{\log n_0}$$

where $n_0$ is base code size.

**Surface codes:**
$$\text{Qubits} \approx d^2 \approx O\left(\log(1/\epsilon)\right)^2$$

#### Practical Numbers

| Target $\epsilon$ | Surface code $d$ | Qubits per logical |
|-------------------|------------------|-------------------|
| $10^{-6}$ | ~17 | ~300 |
| $10^{-10}$ | ~27 | ~750 |
| $10^{-15}$ | ~37 | ~1400 |

For useful quantum computation ($\sim 10^6$ gates):
- Need $\epsilon \lesssim 10^{-10}$
- Requires $\sim 1000$ physical qubits per logical qubit

---

## Quantum Mechanics Connection

### Why Thresholds Exist

The threshold theorem reflects a balance between:

1. **Information-preserving dynamics:** Error correction maps errors back to identity
2. **Information-destroying noise:** Physical errors accumulate

Below threshold, correction dominates. This is a **phase transition** in the reliability of quantum information!

### Analogy: Classical Repetition

Classical systems achieve reliability through redundancy:
- Triple modular redundancy in spacecraft computers
- ECC memory in servers

Quantum error correction extends this principle to quantum information, despite no-cloning and measurement disturbance.

---

## Worked Examples

### Example 1: Concatenation Calculation

**Problem:** For a code with $c = 100$ (so $p_{th} = 0.01$), calculate the logical error rate after 3 levels of concatenation with $p = 0.005$.

**Solution:**

Level 0: $p_0 = 0.005$

Level 1: $p_1 = c \cdot p_0^2 = 100 \times 0.005^2 = 100 \times 0.000025 = 0.0025$

Level 2: $p_2 = c \cdot p_1^2 = 100 \times 0.0025^2 = 100 \times 6.25 \times 10^{-6} = 6.25 \times 10^{-4}$

Level 3: $p_3 = c \cdot p_2^2 = 100 \times (6.25 \times 10^{-4})^2 = 100 \times 3.9 \times 10^{-7} = 3.9 \times 10^{-5}$

**Result:** After 3 levels, logical error rate is $3.9 \times 10^{-5}$ — reduced by factor ~100× from physical rate.

---

### Example 2: Surface Code Distance Selection

**Problem:** For a surface code with threshold $p_{th} = 1\%$ and physical error rate $p = 0.3\%$, what distance is needed for logical error rate $\epsilon < 10^{-9}$?

**Solution:**

For surface codes, approximate formula:
$$\epsilon \approx A \left(\frac{p}{p_{th}}\right)^{(d+1)/2}$$

With $p/p_{th} = 0.3$ and $A \approx 0.1$:

$$10^{-9} = 0.1 \times 0.3^{(d+1)/2}$$

$$10^{-8} = 0.3^{(d+1)/2}$$

$$\log(10^{-8}) = \frac{d+1}{2} \log(0.3)$$

$$-8 = \frac{d+1}{2} \times (-0.52)$$

$$d + 1 = \frac{16}{0.52} \approx 31$$

$$d \approx 30$$

**Need:** Distance-31 surface code (approximately 1000 physical qubits).

---

### Example 3: Threshold from Data

**Problem:** Experiments show logical error rates:
- d=3: $p_L = 2.0\%$
- d=5: $p_L = 0.9\%$
- d=7: $p_L = 0.4\%$

Is the system below threshold?

**Solution:**

Below threshold, error rate should decrease exponentially with distance.

Ratio check:
- $p_L(d=5)/p_L(d=3) = 0.9/2.0 = 0.45$
- $p_L(d=7)/p_L(d=5) = 0.4/0.9 = 0.44$

Consistent ratio ~0.44 per +2 distance indicates **below threshold**!

If above threshold, ratios would be >1 (error increases with distance).

**Conclusion:** System operating below threshold with suppression factor ~2.3× per distance step.

---

## Practice Problems

### Level 1: Direct Application

1. **Threshold Definition:**
   What happens to logical error rate as concatenation level increases when $p < p_{th}$? When $p > p_{th}$?

2. **Simple Calculation:**
   With $p = 0.001$ and $p_{th} = 0.01$, calculate the logical error rate after 2 levels of concatenation assuming $c = 1/p_{th} = 100$.

3. **Distance Scaling:**
   For surface codes, if doubling the distance reduces error by 10×, what distance is needed to go from $\epsilon = 1\%$ to $\epsilon = 0.01\%$?

### Level 2: Intermediate

4. **Threshold Comparison:**
   Why do surface codes have much higher thresholds than concatenated codes?

5. **Resource Estimate:**
   Estimate the number of physical qubits needed for 100 logical qubits at $\epsilon = 10^{-10}$ using surface codes.

6. **Experimental Analysis:**
   Google Willow achieved 2.14× error suppression per +2 distance. What is the implied ratio $p/p_{th}$?

### Level 3: Challenging

7. **Threshold Proof Sketch:**
   Outline the main steps in proving the threshold theorem using concatenated codes.

8. **Magic States:**
   Explain why the Eastin-Knill theorem doesn't prevent universal fault-tolerant computation.

9. **Overhead Optimization:**
   Compare qubit overhead for concatenated vs surface codes to achieve $\epsilon = 10^{-12}$. Which is better?

---

## Computational Lab

### Threshold Analysis

```python
"""
Day 698 Computational Lab: Threshold Theorem Analysis
Concatenation and surface code threshold behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def concatenated_error_rate(p_physical: float, p_threshold: float,
                            levels: int) -> List[float]:
    """
    Calculate logical error rate through concatenation levels.

    Args:
        p_physical: Physical error rate
        p_threshold: Threshold error rate
        levels: Number of concatenation levels

    Returns:
        List of error rates [p_0, p_1, ..., p_L]
    """
    c = 1 / p_threshold  # Constant in p_L = c * p_{L-1}^2
    error_rates = [p_physical]

    for _ in range(levels):
        p_new = c * error_rates[-1]**2
        error_rates.append(min(p_new, 1.0))

    return error_rates


def surface_code_error_rate(p_physical: float, p_threshold: float,
                            distance: int) -> float:
    """
    Estimate logical error rate for surface code.

    Approximate formula: p_L ≈ A * (p/p_th)^((d+1)/2)
    """
    A = 0.1  # Typical prefactor
    ratio = p_physical / p_threshold
    exponent = (distance + 1) / 2
    return A * (ratio ** exponent)


def analyze_threshold_behavior():
    """Analyze behavior above and below threshold."""

    print("=" * 60)
    print("THRESHOLD THEOREM ANALYSIS")
    print("=" * 60)

    p_threshold = 0.01  # 1% threshold

    # Below threshold
    print("\n1. BELOW THRESHOLD (p = 0.5%)")
    print("-" * 40)
    rates_below = concatenated_error_rate(0.005, p_threshold, 5)
    for L, p in enumerate(rates_below):
        print(f"Level {L}: p_L = {p:.2e}")

    # At threshold
    print("\n2. AT THRESHOLD (p = 1%)")
    print("-" * 40)
    rates_at = concatenated_error_rate(0.01, p_threshold, 5)
    for L, p in enumerate(rates_at):
        print(f"Level {L}: p_L = {p:.2e}")

    # Above threshold
    print("\n3. ABOVE THRESHOLD (p = 2%)")
    print("-" * 40)
    rates_above = concatenated_error_rate(0.02, p_threshold, 5)
    for L, p in enumerate(rates_above):
        print(f"Level {L}: p_L = {p:.2e}")


def compare_code_families():
    """Compare concatenated and surface codes."""

    print("\n" + "=" * 60)
    print("CODE FAMILY COMPARISON")
    print("=" * 60)

    # Target logical error rates
    targets = [1e-3, 1e-6, 1e-9, 1e-12]

    print("\nResources needed for target error rates:")
    print("-" * 60)
    print(f"{'Target ε':>12} | {'Concat levels':>15} | {'Surface d':>12} | {'Qubits (surface)':>18}")
    print("-" * 60)

    p_phys_concat = 0.001
    p_th_concat = 0.01
    p_phys_surface = 0.003
    p_th_surface = 0.01

    for target in targets:
        # Concatenated: solve (cp)^(2^L) < target
        # 2^L * log(cp) < log(target)
        # L > log(log(target)/log(cp)) / log(2)
        cp = p_phys_concat / p_th_concat
        if cp < 1:
            L = int(np.ceil(np.log2(np.log(target) / np.log(cp))))
            L = max(1, L)
        else:
            L = float('inf')

        # Surface: solve A * (p/p_th)^((d+1)/2) < target
        # (d+1)/2 > log(target/A) / log(p/p_th)
        ratio = p_phys_surface / p_th_surface
        A = 0.1
        if ratio < 1:
            d = int(np.ceil(2 * np.log(target/A) / np.log(ratio) - 1))
            d = max(3, d)
            qubits = d * d * 2  # Approximate
        else:
            d = float('inf')
            qubits = float('inf')

        print(f"{target:>12.0e} | {L:>15} | {d:>12} | {qubits:>18}")


def plot_threshold_diagram():
    """Create comprehensive threshold visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Concatenation levels for different p
    ax1 = axes[0, 0]
    p_threshold = 0.01
    levels = np.arange(0, 8)

    for p, color in [(0.002, 'blue'), (0.005, 'green'),
                     (0.008, 'orange'), (0.012, 'red')]:
        rates = concatenated_error_rate(p, p_threshold, 7)
        label = f'p = {p*100:.1f}%'
        style = '-' if p < p_threshold else '--'
        ax1.semilogy(levels, rates, color + style + 'o', label=label, linewidth=2)

    ax1.axhline(y=p_threshold, color='black', linestyle=':', label='Threshold')
    ax1.set_xlabel('Concatenation level')
    ax1.set_ylabel('Logical error rate')
    ax1.set_title('Concatenation: Below vs Above Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-15, 1)

    # Plot 2: Surface code distance scaling
    ax2 = axes[0, 1]
    distances = np.arange(3, 31, 2)
    p_threshold_surface = 0.01

    for p, color in [(0.001, 'blue'), (0.003, 'green'),
                     (0.005, 'orange'), (0.008, 'red')]:
        rates = [surface_code_error_rate(p, p_threshold_surface, d) for d in distances]
        ax2.semilogy(distances, rates, color + '-o', label=f'p = {p*100:.1f}%', linewidth=2)

    ax2.set_xlabel('Code distance d')
    ax2.set_ylabel('Logical error rate')
    ax2.set_title('Surface Code: Error vs Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Threshold phase diagram
    ax3 = axes[1, 0]
    p_range = np.linspace(0.001, 0.02, 100)
    d_range = np.arange(3, 21, 2)

    P, D = np.meshgrid(p_range, d_range)
    Z = np.log10(surface_code_error_rate(P, 0.01, D))

    contour = ax3.contourf(P*100, D, Z, levels=20, cmap='RdYlGn_r')
    ax3.axvline(x=1.0, color='white', linestyle='--', linewidth=2, label='Threshold')
    ax3.set_xlabel('Physical error rate (%)')
    ax3.set_ylabel('Code distance')
    ax3.set_title('Logical Error Rate (log₁₀)')
    plt.colorbar(contour, ax=ax3, label='log₁₀(ε)')

    # Plot 4: Experimental progress timeline
    ax4 = axes[1, 1]

    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    google_data = [None, 0.6, 0.5, 0.4, 0.3, 0.14, None]  # % logical error
    ibm_data = [None, None, 0.8, 0.5, 0.3, 0.07, None]
    quantinuum_data = [None, None, None, 0.5, 0.2, 0.1, 0.011]

    ax4.semilogy([y for y, d in zip(years, google_data) if d],
                 [d for d in google_data if d], 'b-o', label='Google', linewidth=2, markersize=8)
    ax4.semilogy([y for y, d in zip(years, ibm_data) if d],
                 [d for d in ibm_data if d], 'r-s', label='IBM', linewidth=2, markersize=8)
    ax4.semilogy([y for y, d in zip(years, quantinuum_data) if d],
                 [d for d in quantinuum_data if d], 'g-^', label='Quantinuum', linewidth=2, markersize=8)

    ax4.axhline(y=0.1, color='gray', linestyle='--', label='0.1% (useful regime)')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Logical error rate (%)')
    ax4.set_title('Experimental Progress (Approximate)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('threshold_theorem_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: threshold_theorem_analysis.png")


def demonstrate_threshold_significance():
    """Show why threshold matters."""

    print("\n" + "=" * 60)
    print("WHY THE THRESHOLD THEOREM MATTERS")
    print("=" * 60)

    print("""
    THE BREAKTHROUGH:
    ─────────────────
    Before threshold theorem (pre-1996):
    • Quantum computers seemed impossible at scale
    • Any noise would accumulate and destroy computation
    • "Quantum computers are fundamentally unreliable"

    After threshold theorem (1996+):
    • Scalable quantum computing is POSSIBLE in principle
    • If p < p_th, arbitrary precision is achievable
    • "Just" an engineering challenge (though immense!)

    CURRENT STATUS (2025):
    ──────────────────────
    • Google Willow: First clear below-threshold operation
    • Surface code thresholds: ~1% (achievable!)
    • Physical error rates: 0.1-0.5% (below threshold!)

    THE PATH FORWARD:
    ─────────────────
    • Need ~1000 physical qubits per logical qubit
    • Current: ~100 logical qubits at high fidelity
    • Goal: ~1,000,000 physical qubits → 1000 logical qubits
    • Timeline: Late 2020s to 2030s for useful computation
    """)


if __name__ == "__main__":
    analyze_threshold_behavior()
    compare_code_families()
    demonstrate_threshold_significance()

    print("\n" + "=" * 60)
    print("Generating visualization...")
    print("=" * 60)
    plot_threshold_diagram()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Threshold condition | $p < p_{th}$ |
| Concatenation error | $p_L \approx (p/p_{th})^{2^L}$ |
| Surface code scaling | $\epsilon \approx (p/p_{th})^{(d+1)/2}$ |
| Overhead (concatenated) | $O(\text{poly}\log(1/\epsilon))$ |
| Overhead (surface) | $O(\log^2(1/\epsilon))$ qubits |

### Main Takeaways

1. **Threshold Existence:** A critical error rate $p_{th}$ separates scalable from unscalable QC
2. **Below Threshold:** Error decreases exponentially with concatenation/distance
3. **Surface Codes Win:** Much higher thresholds (~1%) than concatenated codes (~0.01%)
4. **Experimental Progress:** Google, IBM, Quantinuum operating below threshold (2024-2025)
5. **Resource Scaling:** ~1000 physical qubits per logical qubit for practical computation

---

## Daily Checklist

- [ ] Can state the threshold theorem
- [ ] Understand the concatenation argument
- [ ] Know threshold values for different codes
- [ ] Can analyze experimental threshold data
- [ ] Understand resource scaling implications
- [ ] Know current experimental progress

---

## Preview: Day 699

Tomorrow we introduce **Surface Codes** — the leading candidate for practical quantum error correction:

- Toric code foundations
- Planar surface code geometry
- Anyon-based error correction
- Why surface codes dominate

Surface codes are the bridge from theory to practical quantum computing!

---

*"The threshold theorem is to quantum computing what the Shannon coding theorem is to classical communication — it tells us what's possible."*
— John Preskill
