# Day 722: Introduction to Code Capacity

## Overview

**Date:** Day 722 of 1008
**Week:** 104 (Code Capacity)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Fundamental Limits on Quantum Error Correction

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Classical channel capacity review |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Quantum channel capacity introduction |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Capacity calculations |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Review** Shannon's classical channel capacity theorem
2. **Define** quantum channel capacity and its variants
3. **Explain** the role of capacity in quantum error correction
4. **State** the relationship between code rate and capacity
5. **Distinguish** between different quantum capacities (Q, C, E)
6. **Motivate** the study of capacity for QEC code design

---

## Core Content

### 1. Classical Channel Capacity Review

#### Shannon's Channel Coding Theorem

For a classical channel with transition probabilities $p(y|x)$:

**Channel Capacity:**
$$C = \max_{p(x)} I(X; Y)$$

where the mutual information is:
$$I(X; Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$$

**Shannon's Theorem:** For any rate $R < C$, there exist codes achieving arbitrarily low error probability. For $R > C$, reliable communication is impossible.

#### The Binary Symmetric Channel (BSC)

**Model:** Each bit flipped with probability $p$

**Capacity:**
$$C_{\text{BSC}} = 1 - H(p) = 1 - [-p \log_2 p - (1-p) \log_2(1-p)]$$

| $p$ | $C_{\text{BSC}}$ |
|-----|------------------|
| 0 | 1 |
| 0.1 | 0.531 |
| 0.11 | 0.5 |
| 0.5 | 0 |

**Key insight:** Error correction is possible if and only if $p < 0.5$.

---

### 2. Quantum Channels and Capacity

#### Quantum Channel Definition

A quantum channel $\mathcal{N}: \mathcal{B}(\mathcal{H}_A) \to \mathcal{B}(\mathcal{H}_B)$ is a completely positive trace-preserving (CPTP) map.

**Kraus representation:**
$$\mathcal{N}(\rho) = \sum_k E_k \rho E_k^\dagger, \quad \sum_k E_k^\dagger E_k = I$$

#### Multiple Quantum Capacities

Unlike classical channels, quantum channels have multiple capacities:

| Capacity | Symbol | Use Case |
|----------|--------|----------|
| **Quantum capacity** | $Q(\mathcal{N})$ | Transmit quantum states |
| **Classical capacity** | $C(\mathcal{N})$ | Transmit classical bits |
| **Entanglement-assisted** | $C_E(\mathcal{N})$ | Classical with shared entanglement |
| **Private capacity** | $P(\mathcal{N})$ | Secure classical communication |

**Hierarchy:**
$$Q(\mathcal{N}) \leq P(\mathcal{N}) \leq C(\mathcal{N}) \leq C_E(\mathcal{N})$$

---

### 3. The Quantum Capacity

#### Definition

The **quantum capacity** $Q(\mathcal{N})$ is the maximum rate at which quantum information can be reliably transmitted through channel $\mathcal{N}$.

**Formal definition:**
$$Q(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} Q^{(1)}(\mathcal{N}^{\otimes n})$$

where the single-shot quantum capacity is:
$$Q^{(1)}(\mathcal{N}) = \max_\rho I_c(\rho, \mathcal{N})$$

#### Coherent Information

**Definition:**
$$I_c(\rho, \mathcal{N}) = S(\mathcal{N}(\rho)) - S_e(\rho, \mathcal{N})$$

where:
- $S(\sigma) = -\text{Tr}(\sigma \log_2 \sigma)$ is von Neumann entropy
- $S_e(\rho, \mathcal{N})$ is the entropy exchange

**Key property:** Coherent information can be negative!

#### The LSD Theorem

**Lloyd-Shor-Devetak Theorem:**
$$Q(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \max_\rho I_c(\rho^{(n)}, \mathcal{N}^{\otimes n})$$

**Important:** The regularization (limit over $n$) is necessary—coherent information is not additive in general.

---

### 4. Connection to Quantum Error Correction

#### Code Rate and Capacity

For a quantum error-correcting code $[[n, k, d]]$:

**Code rate:**
$$R = \frac{k}{n}$$

**Capacity constraint:**
$$R \leq Q(\mathcal{N})$$

for reliable error correction over channel $\mathcal{N}$.

#### The Threshold Perspective

**Threshold:** Error rate $p_{\text{th}}$ such that for $p < p_{\text{th}}$, arbitrarily good error correction is possible.

**Capacity perspective:** $p_{\text{th}}$ is where $Q(\mathcal{N}_p) = 0$.

#### Example: Depolarizing Channel

**Definition:**
$$\mathcal{N}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

**Quantum capacity:** $Q(\mathcal{N}_p) > 0$ for $p < p_c$ where $p_c \approx 0.1893$.

This is the **hashing bound threshold** for the depolarizing channel.

---

### 5. Achievability and Converse

#### Achievability (Direct Theorem)

**Statement:** For any $R < Q(\mathcal{N})$, there exist codes achieving rate $R$ with error $\to 0$ as $n \to \infty$.

**Method:** Random coding arguments, similar to Shannon's proof but with quantum refinements.

#### Converse (Strong Converse)

**Statement:** For any $R > Q(\mathcal{N})$, the error probability $\to 1$ as $n \to \infty$.

**Implication:** Capacity is a sharp threshold—no intermediate behavior.

#### Finite Block Length Effects

For finite $n$, the achievable rate is approximately:
$$R \approx Q(\mathcal{N}) - \sqrt{\frac{V}{n}} Q^{-1}(\epsilon) + O\left(\frac{\log n}{n}\right)$$

where $V$ is the channel dispersion and $\epsilon$ is the target error.

---

### 6. Why Capacity Matters for QEC

#### Fundamental Limits

Capacity tells us:
1. **Maximum possible rate** for any code
2. **Threshold error rates** for specific channels
3. **Fundamental trade-offs** between rate and reliability

#### Code Design Guidance

**Good codes approach capacity:**
- Random codes achieve capacity (non-constructive)
- LDPC codes can approach capacity efficiently
- Surface codes have sub-optimal rate but good threshold

#### Comparing Code Families

| Code Family | Rate $k/n$ | Distance | Capacity Efficiency |
|-------------|------------|----------|---------------------|
| Repetition | $1/n$ | $n$ | Very poor |
| Steane [[7,1,3]] | 1/7 | 3 | Moderate |
| Surface code | $O(1/d^2)$ | $d$ | Poor rate |
| Good LDPC | $\Theta(1)$ | $\Theta(n)$ | Near-optimal |

---

### 7. Types of Capacity Problems

#### Code Capacity vs Fault-Tolerant Capacity

**Code capacity:** Assume perfect syndrome measurement
- Simpler to analyze
- Upper bound on fault-tolerant threshold

**Fault-tolerant capacity:** Include measurement errors
- More realistic
- Typically lower threshold

#### Independent vs Correlated Errors

**Independent errors:** Each qubit fails independently
- Standard capacity theory applies
- Memoryless channel assumption

**Correlated errors:** Errors may be spatially/temporally correlated
- More complex capacity theory
- May require different codes

---

## Worked Examples

### Example 1: BSC Capacity Calculation

**Problem:** Calculate the capacity of a binary symmetric channel with crossover probability $p = 0.1$.

**Solution:**

**Step 1:** Binary entropy function
$$H(p) = -p \log_2 p - (1-p) \log_2(1-p)$$
$$H(0.1) = -0.1 \log_2(0.1) - 0.9 \log_2(0.9)$$
$$= -0.1 \times (-3.322) - 0.9 \times (-0.152)$$
$$= 0.332 + 0.137 = 0.469$$

**Step 2:** Capacity
$$C = 1 - H(p) = 1 - 0.469 = 0.531$$

**Interpretation:** Can transmit at most 0.531 bits per channel use reliably.

---

### Example 2: Depolarizing Channel Coherent Information

**Problem:** Compute the coherent information for the maximally mixed input to a depolarizing channel with $p = 0.1$.

**Solution:**

**Step 1:** Input state
$$\rho = \frac{I}{2}$$

**Step 2:** Output state
$$\mathcal{N}_p(\rho) = (1-p)\frac{I}{2} + \frac{p}{3} \cdot 3 \cdot \frac{I}{2} = \frac{I}{2}$$

(Maximally mixed state is a fixed point!)

**Step 3:** Output entropy
$$S(\mathcal{N}_p(\rho)) = S\left(\frac{I}{2}\right) = 1$$

**Step 4:** Entropy exchange
For depolarizing channel and maximally mixed input:
$$S_e = H\left(1-p, \frac{p}{3}, \frac{p}{3}, \frac{p}{3}\right)$$
$$= -(1-p)\log_2(1-p) - 3 \cdot \frac{p}{3}\log_2\frac{p}{3}$$
$$= -(1-p)\log_2(1-p) - p\log_2\frac{p}{3}$$

For $p = 0.1$:
$$S_e = -0.9\log_2(0.9) - 0.1\log_2(0.0333)$$
$$= 0.137 + 0.1 \times 4.91 = 0.137 + 0.491 = 0.628$$

**Step 5:** Coherent information
$$I_c = S(\mathcal{N}_p(\rho)) - S_e = 1 - 0.628 = 0.372$$

**Note:** This is not the optimal input—the quantum capacity requires optimization.

---

### Example 3: Code Rate vs Capacity

**Problem:** A [[100, 10, 7]] quantum code is used over a channel with capacity $Q = 0.15$. Is this code operating below capacity?

**Solution:**

**Code rate:**
$$R = \frac{k}{n} = \frac{10}{100} = 0.1$$

**Comparison:**
$$R = 0.1 < 0.15 = Q$$

**Yes**, the code operates below capacity, so reliable error correction is possible in principle.

**Efficiency:**
$$\eta = \frac{R}{Q} = \frac{0.1}{0.15} = 66.7\%$$

The code uses 66.7% of the available capacity.

---

## Practice Problems

### Direct Application

1. **Problem 1:** Calculate the capacity of a BSC with $p = 0.05$.

2. **Problem 2:** For a depolarizing channel with $p = 0.05$, is the rate $R = 0.8$ achievable? (Use hashing bound $Q \geq 1 - H(p) - p\log_2 3$.)

3. **Problem 3:** A [[49, 1, 7]] code has what rate? How does this compare to typical capacity values?

### Intermediate

4. **Problem 4:** Prove that the quantum capacity of the completely depolarizing channel ($p = 3/4$) is zero.

5. **Problem 5:** For the erasure channel $\mathcal{N}(\rho) = (1-p)\rho + p|e\rangle\langle e|$, show that $Q = 1 - 2p$ for $p < 1/2$.

6. **Problem 6:** A family of codes has rate $R = c/\sqrt{n}$ for constant $c$. Argue that this family cannot approach capacity for any fixed noise rate.

### Challenging

7. **Problem 7:** Prove that coherent information is concave in the input state but not necessarily additive across channels.

8. **Problem 8:** For the amplitude damping channel, compute the coherent information for an arbitrary pure input state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$.

9. **Problem 9:** Explain why the quantum capacity of some channels (like certain Pauli channels) is additive, while others require regularization.

---

## Computational Lab

```python
"""
Day 722: Introduction to Code Capacity
Classical and quantum capacity calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from typing import Tuple, Callable

def binary_entropy(p: float) -> float:
    """
    Compute binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def bsc_capacity(p: float) -> float:
    """
    Capacity of binary symmetric channel.
    C = 1 - H(p)
    """
    return 1 - binary_entropy(p)

def quaternary_entropy(probs: np.ndarray) -> float:
    """
    Compute entropy H(p1, p2, p3, p4) for a 4-outcome distribution.
    """
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))

def depolarizing_hashing_bound(p: float) -> float:
    """
    Hashing bound (lower bound on quantum capacity) for depolarizing channel.
    Q >= 1 - H(p) - p*log2(3)
    """
    if p >= 0.25 * (1 + 3**(1/3) / 3**(1/3)):  # Approximate threshold
        return 0.0
    return max(0, 1 - binary_entropy(p) - p * np.log2(3))

def depolarizing_coherent_info_maximally_mixed(p: float) -> float:
    """
    Coherent information for maximally mixed input to depolarizing channel.
    """
    if p <= 0:
        return 1.0
    if p >= 0.75:
        return 0.0

    # Output entropy (always 1 for maximally mixed)
    S_out = 1.0

    # Entropy exchange
    probs = [1 - p, p/3, p/3, p/3]
    S_e = quaternary_entropy(probs)

    return max(0, S_out - S_e)

def erasure_channel_capacity(p: float) -> float:
    """
    Quantum capacity of erasure channel.
    Q = max(0, 1 - 2p)
    """
    return max(0, 1 - 2 * p)

def amplitude_damping_capacity(gamma: float, num_points: int = 100) -> float:
    """
    Numerically estimate quantum capacity of amplitude damping channel.
    Optimize over pure input states |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
    """
    def coherent_info(theta: float) -> float:
        """Compute coherent information for given input."""
        # Input state probabilities
        p0 = np.cos(theta)**2
        p1 = np.sin(theta)**2

        # Output state (2x2 density matrix)
        rho_out = np.array([
            [p0 + gamma * p1, np.sqrt(1 - gamma) * np.cos(theta) * np.sin(theta)],
            [np.sqrt(1 - gamma) * np.cos(theta) * np.sin(theta), (1 - gamma) * p1]
        ])

        # Output entropy
        eigenvalues = np.linalg.eigvalsh(rho_out)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        S_out = -np.sum(eigenvalues * np.log2(eigenvalues))

        # Entropy exchange (environment state)
        rho_env = np.array([
            [(1 - gamma) * p1, 0],
            [0, p0 + gamma * p1]
        ])
        # Actually need to compute entropy of joint reference-environment state
        # For amplitude damping with pure input, this simplifies

        # Complementary channel output
        lambda_0 = p0 + gamma * p1
        lambda_1 = (1 - gamma) * p1

        if lambda_0 > 1e-15 and lambda_1 > 1e-15:
            S_e = -lambda_0 * np.log2(lambda_0) - lambda_1 * np.log2(lambda_1)
        elif lambda_0 > 1e-15:
            S_e = 0
        else:
            S_e = 0

        return S_out - S_e

    # Optimize over theta
    best_ic = 0
    for theta in np.linspace(0, np.pi/2, num_points):
        ic = coherent_info(theta)
        best_ic = max(best_ic, ic)

    return max(0, best_ic)

def plot_capacities():
    """Plot various channel capacities vs error parameter."""
    p_values = np.linspace(0.001, 0.5, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Classical BSC capacity
    ax1 = axes[0]
    bsc_caps = [bsc_capacity(p) for p in p_values]
    ax1.plot(p_values, bsc_caps, 'b-', linewidth=2, label='BSC Capacity')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.11, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Crossover Probability p')
    ax1.set_ylabel('Capacity (bits per use)')
    ax1.set_title('Binary Symmetric Channel Capacity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.5])
    ax1.set_ylim([0, 1.1])

    # Quantum channel capacities
    ax2 = axes[1]

    # Depolarizing - hashing bound
    dep_hash = [depolarizing_hashing_bound(p) for p in p_values]
    ax2.plot(p_values, dep_hash, 'r-', linewidth=2, label='Depolarizing (hashing)')

    # Erasure channel
    erasure_caps = [erasure_channel_capacity(p) for p in p_values]
    ax2.plot(p_values, erasure_caps, 'g-', linewidth=2, label='Erasure')

    # Amplitude damping (approximate)
    gamma_values = p_values  # Use same x-axis
    ad_caps = [amplitude_damping_capacity(g) for g in gamma_values[::10]]  # Subsample for speed
    ax2.plot(gamma_values[::10], ad_caps, 'b-o', linewidth=2, markersize=4, label='Amp. Damping')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Error Parameter')
    ax2.set_ylabel('Quantum Capacity (qubits per use)')
    ax2.set_title('Quantum Channel Capacities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.savefig('channel_capacities.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: channel_capacities.png")

def plot_code_rates_vs_capacity():
    """Compare various code rates to depolarizing channel capacity."""
    p_values = np.linspace(0.001, 0.2, 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Capacity (hashing bound)
    capacity = [depolarizing_hashing_bound(p) for p in p_values]
    ax.fill_between(p_values, capacity, alpha=0.3, color='green', label='Achievable region')
    ax.plot(p_values, capacity, 'g-', linewidth=2, label='Hashing bound')

    # Code rates
    codes = [
        ('[[5,1,3]]', 1/5, 'b'),
        ('[[7,1,3]] Steane', 1/7, 'r'),
        ('[[9,1,3]] Shor', 1/9, 'm'),
        ('[[17,1,5]]', 1/17, 'c'),
        ('Surface d=5', 1/25, 'orange'),
    ]

    for name, rate, color in codes:
        ax.axhline(y=rate, color=color, linestyle='--', alpha=0.7, label=f'{name}: R={rate:.3f}')

    ax.set_xlabel('Depolarizing Parameter p')
    ax.set_ylabel('Rate (logical qubits per physical qubit)')
    ax.set_title('Code Rates vs Channel Capacity')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.2])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('code_rates_capacity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: code_rates_capacity.png")

# Main demonstration
print("=" * 60)
print("Introduction to Code Capacity")
print("=" * 60)

# 1. Classical BSC capacity
print("\n1. Binary Symmetric Channel Capacity")
print("-" * 40)

print(f"\n{'p':<10} {'H(p)':<12} {'C = 1-H(p)':<12}")
print("-" * 35)
for p in [0.01, 0.05, 0.1, 0.11, 0.15, 0.2, 0.5]:
    H_p = binary_entropy(p)
    C = bsc_capacity(p)
    print(f"{p:<10.2f} {H_p:<12.4f} {C:<12.4f}")

# 2. Quantum channel capacities
print("\n2. Quantum Channel Capacities (Depolarizing)")
print("-" * 40)

print(f"\n{'p':<10} {'Hashing Bound':<15} {'Coherent Info (max mix)':<20}")
print("-" * 50)
for p in [0.01, 0.05, 0.1, 0.15, 0.18, 0.19, 0.20]:
    hashing = depolarizing_hashing_bound(p)
    ci_mm = depolarizing_coherent_info_maximally_mixed(p)
    print(f"{p:<10.2f} {hashing:<15.4f} {ci_mm:<20.4f}")

# 3. Threshold identification
print("\n3. Threshold Identification")
print("-" * 40)

# Find where hashing bound goes to zero
for p in np.linspace(0.18, 0.20, 21):
    if depolarizing_hashing_bound(p) < 0.001:
        print(f"Depolarizing hashing bound threshold: p ≈ {p:.3f}")
        break

# Erasure channel threshold
print(f"Erasure channel threshold: p = 0.5 (exact)")

# 4. Code comparison
print("\n4. Code Rate Comparison")
print("-" * 40)

codes = [
    ('[[5,1,3]] Perfect', 5, 1, 3),
    ('[[7,1,3]] Steane', 7, 1, 3),
    ('[[9,1,3]] Shor', 9, 1, 3),
    ('[[9,1,4,3]] Bacon-Shor', 9, 1, 3),
    ('[[15,1,3]] Reed-Muller', 15, 1, 3),
    ('[[17,1,5]]', 17, 1, 5),
    ('[[25,1,5]] Surface', 25, 1, 5),
    ('[[49,1,7]] Surface', 49, 1, 7),
]

print(f"\n{'Code':<25} {'n':<5} {'k':<5} {'d':<5} {'Rate':<10} {'Efficiency*':<12}")
print("-" * 70)
print("* Efficiency = Rate / Hashing bound at p=0.05")

hashing_005 = depolarizing_hashing_bound(0.05)
for name, n, k, d in codes:
    rate = k / n
    eff = rate / hashing_005 * 100
    print(f"{name:<25} {n:<5} {k:<5} {d:<5} {rate:<10.4f} {eff:<12.1f}%")

# 5. Capacity vs threshold
print("\n5. Capacity Perspective on Thresholds")
print("-" * 40)

print("""
Channel Type        | Capacity Threshold | Practical Threshold
--------------------|--------------------|-----------------------
Depolarizing        | p ≈ 18.9%          | ~15-19% (MWPM decoder)
Bit-flip (X only)   | p ≈ 11%            | ~10.3% (surface code)
Erasure             | p = 50%            | ~50% (exactly achievable)
Amplitude Damping   | γ ≈ 50%            | ~40% (depends on code)

Note: Practical thresholds depend on decoder and code choice.
Capacity gives fundamental upper limit.
""")

# Generate plots
print("\n6. Generating Visualizations...")
plot_capacities()
plot_code_rates_vs_capacity()

print("\n" + "=" * 60)
print("Analysis complete!")
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Channel capacity** | Maximum reliable transmission rate |
| **Quantum capacity $Q$** | Rate for transmitting quantum states |
| **Coherent information** | Quantum analog of mutual information |
| **Hashing bound** | Lower bound on quantum capacity |
| **Code rate** | $R = k/n$ for $[[n,k,d]]$ code |
| **Threshold** | Error rate where $Q \to 0$ |

### Key Equations

$$\boxed{C_{\text{BSC}} = 1 - H(p)}$$

$$\boxed{Q(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \max_\rho I_c(\rho^{(n)}, \mathcal{N}^{\otimes n})}$$

$$\boxed{I_c(\rho, \mathcal{N}) = S(\mathcal{N}(\rho)) - S_e(\rho, \mathcal{N})}$$

$$\boxed{Q_{\text{dep}} \geq 1 - H(p) - p \log_2 3 \quad \text{(hashing bound)}}$$

### Main Takeaways

1. **Capacity sets fundamental limits** on error correction
2. **Quantum channels have multiple capacities** (Q, C, C_E, P)
3. **Coherent information** determines quantum capacity
4. **Hashing bound** gives achievable lower bound
5. **Code rate must satisfy** $R \leq Q(\mathcal{N})$ for reliable QEC

---

## Daily Checklist

- [ ] I can compute classical channel capacity
- [ ] I understand the different quantum capacities
- [ ] I can explain coherent information
- [ ] I know how capacity relates to QEC thresholds
- [ ] I can compare code rates to capacity
- [ ] I completed the computational lab

---

## Preview: Day 723

Tomorrow we study **Quantum Channel Capacity** in depth, including:
- Detailed coherent information calculations
- The Lloyd-Shor-Devetak theorem
- Additivity and regularization
- Degradable and anti-degradable channels
