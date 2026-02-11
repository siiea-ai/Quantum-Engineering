# Day 727: Practical Capacity Applications

## Overview

**Date:** Day 727 of 1008
**Week:** 104 (Code Capacity)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Real-World Applications of Capacity Theory

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Realistic noise models |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Communication and computation |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Resource analysis |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Apply** capacity theory to realistic noise scenarios
2. **Analyze** quantum communication channel requirements
3. **Calculate** resource overhead from capacity constraints
4. **Compare** different noise models' capacity implications
5. **Evaluate** practical vs theoretical thresholds
6. **Design** systems that approach capacity limits

---

## Core Content

### 1. Realistic Noise Models

#### Beyond Depolarizing

Real quantum systems experience:

| Noise Type | Physical Origin | Capacity Impact |
|------------|-----------------|-----------------|
| **T1 (amplitude damping)** | Energy relaxation | Q varies with γ |
| **T2 (dephasing)** | Phase randomization | Q = 1 - H(p) |
| **Leakage** | Transitions to non-computational states | Reduces effective dimension |
| **Correlated errors** | Crosstalk, cosmic rays | May require regularization |
| **Non-Markovian** | Memory effects | Standard capacity may not apply |

#### Combined T1/T2 Noise

**Generalized amplitude damping + dephasing:**

$$\mathcal{N}_{T1,T2}(\rho) = \mathcal{N}_{\text{GAD}}(\gamma, N_{th}) \circ \mathcal{N}_{\text{deph}}(p_z)$$

**Approximate capacity:**

For low noise:
$$Q \approx 1 - H(\gamma) - H(p_z)$$

For moderate noise: numerical computation required.

#### Biased Noise

Many physical systems have $T_2 < T_1$, meaning:
$$p_Z \gg p_X, p_Y$$

**Capacity advantage:**

Biased noise has higher threshold than symmetric:
- Symmetric depolarizing: $p_{\text{th}} \approx 18.9\%$
- Pure Z-noise: $p_{\text{th}} = 50\%$
- Z-biased (10:1): $p_{\text{th}} \approx 25-35\%$

---

### 2. Quantum Communication Channels

#### Optical Fiber Channel

**Loss model:** Each photon lost with probability $\eta$

$$\mathcal{N}_{\text{loss}}(\rho) = (1-\eta)\rho + \eta|0\rangle\langle 0|$$

**Quantum capacity:**
$$Q = \max(0, -\log_2(1-\eta) - \eta\log_2\frac{\eta}{1-\eta})$$

For high loss ($\eta \to 1$): $Q \to 0$

**Practical limitation:** Loss of 0.2 dB/km → repeaters needed every ~50 km

#### Free-Space Optical Channel

**Turbulence + loss:**

More complex capacity formula depending on:
- Beam divergence
- Atmospheric conditions
- Adaptive optics

#### Satellite Quantum Links

**Uplink vs downlink:**
- Downlink: ~20 dB loss for LEO
- Uplink: Higher loss due to turbulence

**Capacity-based design:**

For 90% channel fidelity: need rate $R < Q \approx 0.5$

---

### 3. Resource Overhead Analysis

#### Physical Qubits per Logical Qubit

**From capacity:**
$$n_{\text{phys}} \geq \frac{k_{\text{log}}}{Q(\mathcal{N})}$$

**Example:** $p = 0.1$ depolarizing
- $Q \approx 0.43$
- Need at least 2.3 physical qubits per logical

**In practice:** Much higher due to:
- Finite code efficiency
- Fault-tolerance overhead
- Practical decoder limitations

#### Code Efficiency Factor

$$\eta_{\text{code}} = \frac{R_{\text{code}}}{Q(\mathcal{N})}$$

| Code | Rate | At p=0.05 | Efficiency |
|------|------|-----------|------------|
| [[7,1,3]] Steane | 0.143 | 0.635 | 22.5% |
| [[17,1,5]] | 0.059 | 0.635 | 9.3% |
| Surface d=5 | 0.04 | 0.635 | 6.3% |
| Good qLDPC | 0.5 | 0.635 | 78.7% |

#### Total Overhead

$$\text{Overhead} = \frac{n_{\text{phys}}}{k_{\text{log}}} = \frac{1}{R_{\text{code}}}$$

**Capacity sets the minimum:**
$$\text{Overhead}_{\min} = \frac{1}{Q(\mathcal{N})}$$

---

### 4. Threshold Comparison

#### Code Capacity vs Practical Threshold

| Metric | Definition | Depolarizing |
|--------|------------|--------------|
| **Hashing threshold** | $Q = 0$ | 18.93% |
| **Code capacity threshold** | $Q = 0$ | 18.93% |
| **Phenomenological** | Include measurement errors | ~3% |
| **Circuit-level** | Full fault-tolerant model | ~1% |

**Key insight:** Code capacity is an upper bound on achievable threshold.

#### Why the Gap?

1. **Measurement errors:** Syndrome extraction is noisy
2. **Gate overhead:** QEC circuits add error opportunities
3. **Correlated errors:** Space-time correlations
4. **Decoder limitations:** Practical decoders sub-optimal

#### Closing the Gap

**Strategies:**
- Better decoders (ML approaching capacity)
- Single-shot codes (reduce measurement rounds)
- Bias-exploiting codes (match noise to code)
- Error mitigation + QEC hybrid

---

### 5. Application: Quantum Repeater Design

#### The Repeater Problem

Direct transmission over fiber: $Q \propto 10^{-\alpha L/10}$

For $L = 1000$ km, $\alpha = 0.2$ dB/km: $Q \approx 10^{-20}$

**Solution:** Quantum repeaters

#### Capacity-Based Design

**Per-segment capacity:** $Q_{\text{seg}}$ for length $L/n$

**End-to-end rate:**
$$R_{\text{total}} \leq \min_i Q_i$$

**Optimal segment length:**
Balance loss per segment vs repeater overhead

#### Example Design

**Target:** 1000 km link, 1 kHz entanglement rate

**Approach:**
1. Choose segment: 50 km → $Q_{\text{seg}} \approx 0.1$
2. Number of segments: 20
3. Repeater protocol: entanglement swapping
4. End-to-end rate: $\approx 0.1^{20} \times f_{\text{swap}}$

**Need:** Error-corrected repeaters (third generation)

---

### 6. Application: Fault-Tolerant Computation

#### Resource Estimation

**For $N$ logical gates at error rate $\epsilon$:**

**Step 1:** Required code distance
$$d \geq \frac{\log(N/\epsilon)}{\log(1/p_{\text{eff}})}$$

**Step 2:** Physical qubits (surface code)
$$n \approx 2d^2 \times N_{\text{logical}}$$

**Step 3:** Compare to capacity minimum
$$n_{\min} = \frac{N_{\text{logical}}}{Q(\mathcal{N})}$$

#### Example: 1000 Logical Qubits

At $p = 0.001$ physical error rate:

| Approach | Physical Qubits |
|----------|-----------------|
| Capacity minimum | ~1,050 |
| Surface code d=11 | ~242,000 |
| Good qLDPC | ~10,000 |

**Gap:** Surface code uses 230× more than minimum!

---

## Worked Examples

### Example 1: Fiber Link Capacity

**Problem:** Calculate the quantum capacity for a 100 km fiber link with 0.2 dB/km loss.

**Solution:**

**Step 1:** Total loss
$$\text{Loss} = 0.2 \times 100 = 20 \text{ dB}$$
$$\eta = 1 - 10^{-20/10} = 1 - 0.01 = 0.99$$

**Step 2:** Capacity (pure loss channel)
$$Q = \max(0, \log_2(1-\eta))$$

For $\eta = 0.99$:
$$Q = \log_2(0.01) = -6.64$$

This is negative! For $\eta > 0.5$: $Q = 0$ for pure loss.

**Interpretation:** Direct transmission impossible. Need repeaters.

---

### Example 2: Biased Noise Advantage

**Problem:** Compare thresholds for symmetric vs 10:1 Z-biased noise.

**Solution:**

**Symmetric:** $(p/3, p/3, p/3)$ for X, Y, Z

Threshold: $p_{\text{th}} \approx 18.9\%$

**Z-biased (10:1):** $p_X = p_Y = p/12$, $p_Z = 10p/12$

$$H(\text{errors}) = H(1-p, p/12, p/12, 10p/12)$$

Solving $H = 1$ numerically: $p_{\text{th}} \approx 29\%$

**Advantage:** 50% higher threshold with biased noise!

---

### Example 3: Resource Scaling

**Problem:** Estimate physical qubits needed for a quantum computer running Shor's algorithm to factor 2048-bit integers, assuming $p = 0.001$ physical error rate.

**Solution:**

**Step 1:** Logical requirements
- Qubits: ~4000 (rough estimate)
- Gates: ~$10^{12}$ (rough)

**Step 2:** Capacity minimum
$$Q \approx 1 - H(0.001) - 0.001 \log_2 3 \approx 0.99$$
$$n_{\min} = 4000 / 0.99 \approx 4040$$

**Step 3:** Surface code estimate
- Distance needed: $d \approx 27$ for $10^{-15}$ logical error rate
- Per logical: $2 \times 27^2 \approx 1460$
- Total: $4000 \times 1460 \approx 5.8 \times 10^6$

**Gap:** $5.8 \times 10^6 / 4040 \approx 1400×$

Surface code uses 1400× the capacity minimum!

---

## Practice Problems

### Direct Application

1. **Problem 1:** What is the capacity of a 50 km fiber with 0.18 dB/km loss?

2. **Problem 2:** Calculate efficiency of the [[15,1,3]] code at $p = 0.03$.

3. **Problem 3:** For a channel with $Q = 0.2$, what's the minimum overhead for 100 logical qubits?

### Intermediate

4. **Problem 4:** Design a quantum repeater chain for 500 km with target rate 1 Hz.

5. **Problem 5:** Compare resource requirements for surface code vs good qLDPC at $p = 0.005$.

6. **Problem 6:** How does biased noise affect optimal code choice?

### Challenging

7. **Problem 7:** Derive the capacity of a combined T1/T2 channel analytically for small parameters.

8. **Problem 8:** Analyze the capacity-threshold gap for the surface code and explain each contributing factor.

9. **Problem 9:** Design a capacity-optimal protocol for a satellite quantum link with 30 dB downlink loss.

---

## Computational Lab

```python
"""
Day 727: Practical Capacity Applications
Real-world capacity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def hashing_bound(p):
    if p <= 0:
        return 1
    if p >= 0.75:
        return 0
    return max(0, 1 - binary_entropy(p) - p * np.log2(3))

def biased_hashing_bound(p_total, z_ratio=1/3):
    """Hashing bound for biased Pauli noise."""
    if p_total <= 0:
        return 1
    if p_total >= 1:
        return 0

    p_z = p_total * z_ratio
    p_xy = p_total * (1 - z_ratio) / 2
    p_i = 1 - p_total

    probs = [p_i, p_xy, p_xy, p_z]
    probs = [p for p in probs if p > 0]
    H = -sum(p * np.log2(p) for p in probs)

    return max(0, 1 - H)

def fiber_capacity(length_km, loss_db_per_km=0.2):
    """Quantum capacity of fiber channel."""
    total_loss_db = loss_db_per_km * length_km
    transmittance = 10**(-total_loss_db / 10)

    if transmittance < 0.5:
        return 0  # Below 50% loss threshold

    # Pure loss channel capacity (approximate)
    return max(0, np.log2(2 * transmittance - 1)) if transmittance > 0.5 else 0

def plot_fiber_capacity():
    """Plot fiber channel capacity vs distance."""
    distances = np.linspace(1, 200, 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    for loss in [0.15, 0.2, 0.25]:
        caps = [fiber_capacity(d, loss) for d in distances]
        ax.plot(distances, caps, linewidth=2, label=f'Loss = {loss} dB/km')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Quantum Capacity (qubits/use)')
    ax.set_title('Fiber Channel Capacity vs Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 200])

    plt.tight_layout()
    plt.savefig('fiber_capacity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fiber_capacity.png")

def plot_bias_advantage():
    """Plot threshold advantage from biased noise."""
    p_values = np.linspace(0.001, 0.5, 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Different bias ratios
    biases = [
        (1/3, 'Symmetric (1:1:1)'),
        (0.5, '2:1:1 Z-bias'),
        (10/12, '10:1:1 Z-bias'),
        (0.99, 'Near-pure Z'),
    ]

    for z_ratio, label in biases:
        caps = [biased_hashing_bound(p, z_ratio) for p in p_values]
        ax.plot(p_values * 100, caps, linewidth=2, label=label)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Total Error Rate (%)')
    ax.set_ylabel('Hashing Bound')
    ax.set_title('Capacity vs Noise Bias')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bias_advantage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bias_advantage.png")

def analyze_code_efficiency():
    """Analyze efficiency of various codes."""
    print("=" * 70)
    print("Code Efficiency Analysis")
    print("=" * 70)

    codes = [
        ('[[5,1,3]] Perfect', 5, 1),
        ('[[7,1,3]] Steane', 7, 1),
        ('[[9,1,3]] Shor', 9, 1),
        ('[[17,1,5]]', 17, 1),
        ('Surface d=3', 18, 1),
        ('Surface d=5', 50, 1),
        ('Surface d=7', 98, 1),
        ('Good qLDPC (est)', 100, 10),
    ]

    error_rates = [0.01, 0.03, 0.05, 0.1]

    print(f"\n{'Code':<20} {'Rate':<10}", end='')
    for p in error_rates:
        print(f"η @ p={p:.0%}   ", end='')
    print()
    print("-" * 80)

    for name, n, k in codes:
        rate = k / n
        print(f"{name:<20} {rate:<10.4f}", end='')
        for p in error_rates:
            Q = hashing_bound(p)
            if Q > 0:
                eff = rate / Q
                print(f"{eff:<12.1%}", end='')
            else:
                print(f"{'N/A':<12}", end='')
        print()

def analyze_threshold_gap():
    """Analyze gap between code capacity and practical thresholds."""
    print("\n" + "=" * 70)
    print("Threshold Gap Analysis")
    print("=" * 70)

    print("""
    Noise Model                  | Code Capacity | Practical    | Gap Factor
    -----------------------------|---------------|--------------|------------
    Depolarizing (code cap)      | 18.93%        | N/A          | 1.0x
    Depolarizing (phenomenolog.) | 18.93%        | ~3%          | 6.3x
    Depolarizing (circuit)       | 18.93%        | ~1%          | 19x
    Biased Z (10:1) code cap     | ~29%          | N/A          | 1.0x
    Biased Z (10:1) circuit      | ~29%          | ~5%          | 5.8x
    Pure Z (code cap)            | 50%           | N/A          | 1.0x
    Pure Z (circuit)             | 50%           | ~10%         | 5x

    The gap comes from:
    1. Syndrome measurement errors (~3x)
    2. Gate errors during QEC (~2x)
    3. Correlated error effects (~1.5x)
    4. Decoder sub-optimality (~1.3x)
    """)

def resource_estimation():
    """Estimate resources for quantum computation."""
    print("\n" + "=" * 70)
    print("Resource Estimation")
    print("=" * 70)

    print("\nScenario: Fault-tolerant quantum computation")
    print("Requirements: 1000 logical qubits, 10^9 logical gates, p_log < 10^-12")

    physical_error_rates = [0.01, 0.001, 0.0001]

    print(f"\n{'p_phys':<10} {'Q(N)':<10} {'n_min':<12} {'Surface':<15} {'qLDPC':<15} {'Gap'}")
    print("-" * 75)

    for p in physical_error_rates:
        Q = hashing_bound(p)
        n_min = 1000 / Q if Q > 0 else float('inf')

        # Surface code estimate: d needed for p_log < 10^-12
        # Roughly: p_log ~ (p/p_th)^d, need d ~ 12*log(10)/log(p_th/p)
        if p < 0.01:
            d = max(3, int(np.ceil(15 * np.log10(1/(10*p)))))
        else:
            d = 25
        n_surface = 1000 * 2 * d**2

        # Good qLDPC: overhead ~10x (rate 0.1)
        n_ldpc = 1000 * 10

        gap = n_surface / n_min if n_min > 0 else float('inf')

        print(f"{p:<10.4f} {Q:<10.3f} {n_min:<12.0f} {n_surface:<15.0f} {n_ldpc:<15.0f} {gap:.0f}x")

def main():
    print("=" * 70)
    print("Practical Capacity Applications")
    print("=" * 70)

    # Code efficiency
    analyze_code_efficiency()

    # Threshold gap
    analyze_threshold_gap()

    # Resource estimation
    resource_estimation()

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Plots...")
    plot_fiber_capacity()
    plot_bias_advantage()

    print("\n" + "=" * 70)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Realistic noise** | T1/T2, biased, correlated |
| **Resource overhead** | $1/Q$ minimum, much higher in practice |
| **Threshold gap** | Code capacity >> circuit-level threshold |
| **Bias advantage** | Biased noise has higher threshold |
| **Efficiency** | $\eta = R_{\text{code}}/Q$ measures capacity use |

### Key Numbers

| Metric | Depolarizing | Biased Z (10:1) | Pure Z |
|--------|--------------|-----------------|--------|
| Hashing threshold | 18.9% | ~29% | 50% |
| Circuit threshold | ~1% | ~5% | ~10% |
| Gap factor | 19× | 6× | 5× |

### Main Takeaways

1. **Capacity sets the ultimate limit** but practical systems far below
2. **Biased noise** significantly improves achievable thresholds
3. **Resource gap** is 10-1000× between capacity and practice
4. **Good qLDPC codes** can close much of the efficiency gap
5. **Threshold gap** comes from measurements, gates, correlations

---

## Daily Checklist

- [ ] I can apply capacity theory to realistic noise
- [ ] I understand the threshold gap and its causes
- [ ] I can calculate resource overhead
- [ ] I know how bias affects capacity
- [ ] I can compare code efficiency to capacity
- [ ] I completed the computational lab

---

## Preview: Day 728

Tomorrow we conclude Week 104 with **Week Synthesis**, including:
- Comprehensive review of code capacity
- Integration with QEC fundamentals
- Month 26 summary
- Preparation for Month 27 (Stabilizer Formalism)
