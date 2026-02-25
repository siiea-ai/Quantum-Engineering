# Day 724: Hashing Bound and Threshold Theorem

## Overview

**Date:** Day 724 of 1008
**Week:** 104 (Code Capacity)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Achievable Rates and Error Correction Thresholds

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Hashing bound derivation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Threshold theorem |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Numerical analysis |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Derive** the hashing bound for quantum channels
2. **Explain** how random stabilizer codes achieve the hashing bound
3. **State** the threshold theorem for quantum error correction
4. **Relate** capacity to error correction thresholds
5. **Compute** threshold values for standard channels
6. **Compare** different code families' approaches to capacity

---

## Core Content

### 1. The Hashing Bound

#### Statement

**Theorem (Hashing Bound):**
For a Pauli channel $\mathcal{N}$ with error probabilities $\{p_I, p_X, p_Y, p_Z\}$:

$$Q(\mathcal{N}) \geq 1 - H(p_I, p_X, p_Y, p_Z)$$

where $H$ is the Shannon entropy.

**For the depolarizing channel:**
$$Q(\mathcal{N}_p) \geq 1 - H(p) - p \log_2 3$$

where $p$ is the total error probability.

#### Interpretation

The hashing bound says:
- **Rate achievable:** $R = 1 - H(\text{error distribution})$
- **Meaning:** "Hash out" the entropy introduced by errors
- **Method:** Random stabilizer codes

#### Why "Hashing"?

Classical hashing: compress data to remove redundancy

Quantum hashing: project onto stabilizer subspace to remove error entropy

$$\text{Code rate} = 1 - \frac{\text{Entropy from errors}}{\text{Qubits}}$$

---

### 2. Derivation via Random Coding

#### Random Stabilizer Codes

**Construction:**
1. Pick $n - k$ random commuting Pauli operators
2. These generate a stabilizer group $\mathcal{S}$
3. Code space is the $+1$ eigenspace

**Key property:** Random codes are "typical" — they capture the average behavior.

#### Achievability Argument

**Step 1: Error model**

Pauli channel applies error $E \in \{I, X, Y, Z\}^{\otimes n}$ with probability $p(E)$.

**Step 2: Syndrome distribution**

For a random stabilizer code, different errors give different syndromes with high probability.

**Step 3: Decoding**

Maximum likelihood decoder: find most likely error consistent with syndrome.

**Step 4: Error probability**

For $k/n < 1 - H(\text{errors})$, the probability of decoding failure $\to 0$ as $n \to \infty$.

#### Mathematical Sketch

**Typical error sequences:**
$$|\{E : p(E) \approx 2^{-nH}\}| \approx 2^{nH}$$

**Syndrome space:**
$$|\text{Syndromes}| = 2^{n-k}$$

**Decodability condition:**
$$2^{nH} \leq 2^{n-k}$$
$$H \leq 1 - k/n$$
$$k/n \leq 1 - H$$

This is the hashing bound!

---

### 3. Explicit Hashing Bound Formulas

#### Depolarizing Channel

**Error distribution:** $(1-p, p/3, p/3, p/3)$

**Entropy:**
$$H = -(1-p)\log_2(1-p) - 3 \cdot \frac{p}{3}\log_2\frac{p}{3}$$
$$= H(p) + p\log_2 3$$

**Hashing bound:**
$$Q \geq 1 - H(p) - p\log_2 3$$

| $p$ | $H(p) + p\log_2 3$ | Hashing bound |
|-----|---------------------|---------------|
| 0.01 | 0.106 | 0.894 |
| 0.05 | 0.365 | 0.635 |
| 0.10 | 0.569 | 0.431 |
| 0.15 | 0.734 | 0.266 |
| 0.18 | 0.832 | 0.168 |
| 0.189 | 0.867 | 0.133 |

**Threshold:** $p_{\text{hash}} \approx 0.1893$ where bound $\to 0$.

#### Bit-Flip Channel

**Error distribution:** $(1-p, p, 0, 0)$ — only X errors

**Hashing bound:**
$$Q \geq 1 - H(p)$$

**Threshold:** $p = 0.5$ (but CSS codes limited to $p \approx 0.11$)

#### Phase-Flip Channel

**Error distribution:** $(1-p, 0, 0, p)$ — only Z errors

**Hashing bound:**
$$Q \geq 1 - H(p)$$

Same as bit-flip by symmetry.

#### Asymmetric Channels

For general Pauli channel with $\{p_I, p_X, p_Y, p_Z\}$:
$$Q \geq 1 - H(p_I, p_X, p_Y, p_Z)$$

**Example:** Biased noise with $p_Z \gg p_X, p_Y$

This motivates **biased-noise codes** that protect better against Z errors.

---

### 4. The Threshold Theorem

#### Informal Statement

**Theorem (Threshold):**
There exists a threshold error rate $p_{\text{th}} > 0$ such that:
- For $p < p_{\text{th}}$: Arbitrarily reliable quantum computation is possible
- For $p > p_{\text{th}}$: Reliable computation becomes impossible

#### Capacity-Based Threshold

**Definition:** The **code capacity threshold** is:
$$p_{\text{th}}^{\text{cap}} = \max\{p : Q(\mathcal{N}_p) > 0\}$$

For the depolarizing channel: $p_{\text{th}}^{\text{cap}} \approx 0.1893$

#### Fault-Tolerant Threshold

The **fault-tolerant threshold** accounts for:
- Measurement errors
- Gate errors during syndrome extraction
- Correlated failures

Typically: $p_{\text{th}}^{\text{FT}} < p_{\text{th}}^{\text{cap}}$

**Example (Surface code):**
- Code capacity: ~10.3% (bit-flip)
- Phenomenological: ~3.3%
- Circuit-level: ~1%

---

### 5. Threshold Analysis

#### Threshold Equation

At threshold, the hashing bound equals zero:
$$1 - H(p) - p\log_2 3 = 0$$

**Solving numerically:**
$$H(p) + p\log_2 3 = 1$$

This gives $p \approx 0.1893$.

#### Different Noise Models

| Noise Model | Threshold (Hashing) | Notes |
|-------------|---------------------|-------|
| Depolarizing | 18.93% | Symmetric Pauli |
| Bit-flip only | 11%* | CSS X-code limit |
| Phase-flip only | 11%* | CSS Z-code limit |
| Erasure | 50% | Exact threshold |
| Amplitude damping | 50% | Degradable threshold |

*The 11% for bit-flip comes from typical CSS code construction, not hashing.

#### Phase Transition Picture

```
           Rate R
           ↑
           │     /
      1    │    /  Achievable region
           │   /   (R < 1-H)
           │  /
           │ /
           │/─────────────────→ p
           0     p_th      0.5
                 ↓
           Error correction
           possible here
```

---

### 6. Codes Approaching the Hashing Bound

#### Random Stabilizer Codes

**Rate:** $R = 1 - H - \epsilon$ for any $\epsilon > 0$

**Problem:**
- No efficient encoding/decoding
- Exponential complexity

#### CSS Codes

**Construction:** From classical codes $C_1 \supset C_2$

**Rate:** $R = (k_1 - k_2)/n$

**Limit:** CSS structure limits achievable rate

For symmetric errors: CSS can approach hashing bound.

#### LDPC Codes

**Quantum Low-Density Parity-Check codes:**
- Sparse stabilizer matrices
- Efficient decoding (belief propagation)
- Can approach capacity

**Recent breakthrough (2020s):** Good qLDPC codes with $k, d = \Theta(n)$!

#### Comparison Table

| Code Family | Rate | Distance | Decoding | Capacity Fraction |
|-------------|------|----------|----------|-------------------|
| Random stabilizer | $\to 1-H$ | $\Theta(n)$ | Exponential | 100% |
| Surface code | $O(1/d^2)$ | $d$ | Polynomial | $\sim 0\%$ |
| CSS/Steane | $O(1)$ | $O(1)$ | Polynomial | Moderate |
| Good qLDPC | $\Theta(1)$ | $\Theta(n)$ | Polynomial | $\to 100\%$ |

---

### 7. Beyond the Hashing Bound

#### Is Hashing Bound Tight?

For Pauli channels: **Yes** (approximately)

The hashing bound is essentially the quantum capacity for:
- Depolarizing channel
- Pauli channels in general

**Reason:** Coherent information achieved by maximally mixed input.

#### Tighter Bounds

For non-Pauli channels, other techniques needed:
- Degradable extensions
- Semi-definite programming
- PPT (positive partial transpose) bound

#### Upper Bounds

**No-cloning bound:** $Q \leq 1$

**Degradable extension:** If $\mathcal{N} \leq \mathcal{M}$ (degradable), then $Q(\mathcal{N}) \leq Q(\mathcal{M})$

**Rains bound:** From PPT entanglement theory

---

## Worked Examples

### Example 1: Compute Hashing Bound

**Problem:** Calculate the hashing bound for a depolarizing channel with $p = 0.12$.

**Solution:**

**Step 1:** Binary entropy
$$H(0.12) = -0.12\log_2(0.12) - 0.88\log_2(0.88)$$
$$= -0.12 \times (-3.06) - 0.88 \times (-0.184)$$
$$= 0.367 + 0.162 = 0.529$$

**Step 2:** Additional term
$$p\log_2 3 = 0.12 \times 1.585 = 0.190$$

**Step 3:** Hashing bound
$$Q \geq 1 - 0.529 - 0.190 = 0.281$$

**Interpretation:** Can achieve rate $R \leq 0.281$ logical qubits per physical qubit.

---

### Example 2: Find Threshold

**Problem:** Find the threshold for a biased Pauli channel with $p_X = p_Y = 0$ and $p_Z = p$.

**Solution:**

**Error distribution:** $(1-p, 0, 0, p)$

**Entropy:**
$$H = -(1-p)\log_2(1-p) - p\log_2 p = H(p)$$

**Hashing bound:**
$$Q \geq 1 - H(p)$$

**Threshold:** $H(p) = 1 \Rightarrow p = 0.5$

But this is for pure Z errors. In practice:
- Repetition code in X basis achieves this
- Surface code achieves ~10.3% for practical decoding

---

### Example 3: Code Rate Comparison

**Problem:** A [[100, 20, 5]] code is used at error rate $p = 0.05$. Is it operating efficiently relative to capacity?

**Solution:**

**Hashing bound at $p = 0.05$:**
$$Q \geq 1 - H(0.05) - 0.05\log_2 3$$
$$= 1 - 0.286 - 0.079 = 0.635$$

**Code rate:**
$$R = 20/100 = 0.2$$

**Efficiency:**
$$\eta = R/Q = 0.2/0.635 = 31.5\%$$

The code uses only 31.5% of the available capacity — significant room for improvement.

---

## Practice Problems

### Direct Application

1. **Problem 1:** Calculate the hashing bound for $p = 0.08$ depolarizing noise.

2. **Problem 2:** What is the maximum rate achievable at $p = 0.15$ depolarizing?

3. **Problem 3:** For a biased channel with $p_Z = 0.1$, $p_X = p_Y = 0.01$, compute the hashing bound.

### Intermediate

4. **Problem 4:** Prove that the hashing bound for the erasure channel is $1 - 2p$ (matching the quantum capacity).

5. **Problem 5:** Show that the hashing bound is maximized when errors are uniform (depolarizing) among Pauli types.

6. **Problem 6:** A code family has rate $R(n) = 1 - c/\sqrt{n}$. Does it approach capacity as $n \to \infty$?

### Challenging

7. **Problem 7:** Derive the threshold equation for a general Pauli channel and solve it numerically for the depolarizing case.

8. **Problem 8:** Prove that CSS codes cannot exceed rate $1 - 2H(p)$ for symmetric bit-flip and phase-flip errors.

9. **Problem 9:** Show that the threshold for concatenated codes is lower than the hashing bound threshold.

---

## Computational Lab

```python
"""
Day 724: Hashing Bound and Threshold Theorem
Threshold calculations and code capacity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from typing import Tuple, List

def binary_entropy(p: float) -> float:
    """Compute H(p) = -p log₂(p) - (1-p) log₂(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def quaternary_entropy(probs: List[float]) -> float:
    """Compute entropy of 4-outcome distribution."""
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def hashing_bound_depolarizing(p: float) -> float:
    """Hashing bound for depolarizing channel."""
    if p <= 0:
        return 1.0
    if p >= 0.75:
        return 0.0
    return max(0, 1 - binary_entropy(p) - p * np.log2(3))

def hashing_bound_pauli(p_I: float, p_X: float, p_Y: float, p_Z: float) -> float:
    """Hashing bound for general Pauli channel."""
    probs = [p_I, p_X, p_Y, p_Z]
    return max(0, 1 - quaternary_entropy(probs))

def hashing_bound_biased(p_Z: float, p_X: float = None, p_Y: float = None) -> float:
    """Hashing bound for biased noise (Z-dominated)."""
    if p_X is None:
        p_X = 0
    if p_Y is None:
        p_Y = 0
    p_I = 1 - p_X - p_Y - p_Z
    return hashing_bound_pauli(p_I, p_X, p_Y, p_Z)

def find_threshold_depolarizing() -> float:
    """Find threshold where hashing bound = 0."""
    def f(p):
        return hashing_bound_depolarizing(p)

    # Find where hashing bound crosses zero
    threshold = brentq(f, 0.01, 0.25)
    return threshold

def find_threshold_general(p_X_ratio: float = 1/3, p_Y_ratio: float = 1/3,
                           p_Z_ratio: float = 1/3) -> float:
    """
    Find threshold for general Pauli channel.
    Ratios determine how total error p is distributed.
    """
    def hashing_bound(p):
        p_X = p * p_X_ratio
        p_Y = p * p_Y_ratio
        p_Z = p * p_Z_ratio
        p_I = 1 - p
        return hashing_bound_pauli(p_I, p_X, p_Y, p_Z)

    try:
        threshold = brentq(hashing_bound, 0.01, 0.9)
    except ValueError:
        threshold = 0.5  # Default if no zero crossing

    return threshold

def plot_hashing_bounds():
    """Plot hashing bounds for various channels."""
    p_values = np.linspace(0.001, 0.5, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Different Pauli channels
    ax1 = axes[0]

    # Depolarizing
    dep = [hashing_bound_depolarizing(p) for p in p_values]
    ax1.plot(p_values, dep, 'b-', linewidth=2, label='Depolarizing')

    # Pure Z (dephasing)
    pure_z = [max(0, 1 - binary_entropy(p)) for p in p_values]
    ax1.plot(p_values, pure_z, 'g-', linewidth=2, label='Pure Z (dephasing)')

    # Biased Z (Z:X:Y = 10:1:1)
    biased = []
    for p in p_values:
        p_Z = p * 10/12
        p_X = p * 1/12
        p_Y = p * 1/12
        p_I = 1 - p
        biased.append(hashing_bound_pauli(p_I, p_X, p_Y, p_Z))
    ax1.plot(p_values, biased, 'r-', linewidth=2, label='Biased (Z:X:Y=10:1:1)')

    # Erasure (exact)
    erasure = [max(0, 1 - 2*p) for p in p_values]
    ax1.plot(p_values, erasure, 'm--', linewidth=2, label='Erasure (exact)')

    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Error Probability p')
    ax1.set_ylabel('Hashing Bound (Rate)')
    ax1.set_title('Hashing Bounds for Different Channels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.5])
    ax1.set_ylim([-0.1, 1.1])

    # Threshold analysis
    ax2 = axes[1]

    # Thresholds for different bias ratios
    bias_ratios = np.linspace(0.33, 0.99, 50)  # Z ratio from 1/3 to 99%
    thresholds = []

    for z_ratio in bias_ratios:
        xy_ratio = (1 - z_ratio) / 2
        th = find_threshold_general(xy_ratio, xy_ratio, z_ratio)
        thresholds.append(th)

    ax2.plot(bias_ratios, thresholds, 'b-', linewidth=2)
    ax2.axhline(y=0.1893, color='gray', linestyle='--', label='Depolarizing threshold')
    ax2.axhline(y=0.5, color='red', linestyle=':', label='Pure Z threshold')

    ax2.set_xlabel('Z-error Bias Ratio (p_Z / p_total)')
    ax2.set_ylabel('Hashing Bound Threshold')
    ax2.set_title('Threshold vs Noise Bias')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.33, 1.0])

    plt.tight_layout()
    plt.savefig('hashing_bounds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: hashing_bounds.png")

def plot_code_comparison():
    """Compare code rates to hashing bound."""
    p_values = np.linspace(0.001, 0.2, 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Hashing bound
    hashing = [hashing_bound_depolarizing(p) for p in p_values]
    ax.fill_between(p_values, hashing, alpha=0.3, color='green')
    ax.plot(p_values, hashing, 'g-', linewidth=2, label='Hashing bound')

    # Specific codes (constant rate)
    codes = [
        ('[[5,1,3]] Perfect', 1/5, 'blue'),
        ('[[7,1,3]] Steane', 1/7, 'red'),
        ('[[17,1,5]]', 1/17, 'purple'),
        ('Surface d=3', 1/9, 'orange'),
        ('Surface d=5', 1/25, 'brown'),
    ]

    for name, rate, color in codes:
        ax.axhline(y=rate, color=color, linestyle='--', alpha=0.7, label=name)

    # Good LDPC (asymptotically approaches bound)
    ldpc_rate = [0.9 * hashing_bound_depolarizing(p) for p in p_values]
    ax.plot(p_values, ldpc_rate, 'k-', linewidth=2, label='Good qLDPC (90% capacity)')

    ax.set_xlabel('Depolarizing Error Rate p')
    ax.set_ylabel('Code Rate k/n')
    ax.set_title('Code Rates vs Hashing Bound')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.2])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('code_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: code_rate_comparison.png")

def analyze_thresholds():
    """Analyze thresholds for different noise models."""
    print("=" * 60)
    print("Threshold Analysis")
    print("=" * 60)

    # Depolarizing
    th_dep = find_threshold_depolarizing()
    print(f"\nDepolarizing channel threshold: {th_dep:.4f} ({th_dep*100:.2f}%)")

    # Different biases
    print("\nThresholds for biased noise (Z:X:Y ratios):")
    print(f"{'Ratio':<20} {'Threshold':<15} {'vs Depolarizing'}")
    print("-" * 50)

    ratios = [
        ('1:1:1 (symmetric)', 1/3, 1/3, 1/3),
        ('2:1:1', 2/4, 1/4, 1/4),
        ('5:1:1', 5/7, 1/7, 1/7),
        ('10:1:1', 10/12, 1/12, 1/12),
        ('100:1:1', 100/102, 1/102, 1/102),
        ('Z only', 1, 0, 0),
    ]

    for name, z_r, x_r, y_r in ratios:
        if z_r == 1:
            th = 0.5  # Pure dephasing
        else:
            th = find_threshold_general(x_r, y_r, z_r)
        improvement = th / th_dep
        print(f"{name:<20} {th:.4f} ({th*100:5.2f}%)   {improvement:.2f}x")

def analyze_code_efficiency():
    """Analyze efficiency of various codes relative to capacity."""
    print("\n" + "=" * 60)
    print("Code Efficiency Analysis (at p = 0.05)")
    print("=" * 60)

    p = 0.05
    capacity = hashing_bound_depolarizing(p)
    print(f"\nHashing bound at p={p}: {capacity:.4f}")

    codes = [
        ('[[5,1,3]] Perfect', 5, 1, 3),
        ('[[7,1,3]] Steane', 7, 1, 3),
        ('[[9,1,3]] Shor', 9, 1, 3),
        ('[[15,1,3]] RM', 15, 1, 3),
        ('[[17,1,5]]', 17, 1, 5),
        ('[[25,1,5]] Surface', 25, 1, 5),
        ('[[49,1,7]] Surface', 49, 1, 7),
        ('[[100,10,?]] LDPC', 100, 10, None),
        ('[[1000,500,?]] Good LDPC', 1000, 500, None),
    ]

    print(f"\n{'Code':<25} {'Rate':<10} {'Efficiency':<12} {'Capacity Used'}")
    print("-" * 65)

    for name, n, k, d in codes:
        rate = k / n
        eff = rate / capacity
        used = f"{eff*100:.1f}%"
        print(f"{name:<25} {rate:<10.4f} {eff:<12.4f} {used}")

def random_code_simulation():
    """Simulate random stabilizer code performance."""
    print("\n" + "=" * 60)
    print("Random Stabilizer Code Performance (Conceptual)")
    print("=" * 60)

    print("""
Random Stabilizer Code Properties:

1. ACHIEVABLE RATE:
   R = 1 - H(error distribution) - ε
   for any ε > 0, as n → ∞

2. DISTANCE:
   Random codes have distance d = Θ(n)
   (Gilbert-Varshamov bound)

3. ENCODING:
   No efficient algorithm known
   Requires exponential resources

4. DECODING:
   Maximum likelihood: exponential
   Practical decoders: don't exist for random codes

5. COMPARISON TO STRUCTURED CODES:

   Code Type       | Rate    | Distance | Encoding | Decoding
   ----------------|---------|----------|----------|----------
   Random          | Optimal | Optimal  | Exp      | Exp
   Surface         | Poor    | Good     | Poly     | Poly
   LDPC (sparse)   | Good    | Varies   | Poly     | Poly
   Good qLDPC      | Good    | Optimal  | Poly     | Poly*

   * Recent breakthrough codes (Panteleev-Kalachev, etc.)

6. KEY INSIGHT:
   Random codes prove EXISTENCE of good codes
   Structured codes provide CONSTRUCTIVE solutions
""")

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Hashing Bound and Threshold Analysis")
    print("=" * 60)

    # Hashing bounds table
    print("\nHashing Bounds for Depolarizing Channel:")
    print(f"{'p':<10} {'H(p)':<10} {'p·log₂3':<10} {'Bound':<10}")
    print("-" * 42)

    for p in [0.01, 0.05, 0.10, 0.12, 0.15, 0.18, 0.189, 0.20]:
        H_p = binary_entropy(p)
        extra = p * np.log2(3)
        bound = hashing_bound_depolarizing(p)
        print(f"{p:<10.3f} {H_p:<10.4f} {extra:<10.4f} {bound:<10.4f}")

    # Threshold analysis
    analyze_thresholds()

    # Code efficiency
    analyze_code_efficiency()

    # Random code discussion
    random_code_simulation()

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    plot_hashing_bounds()
    plot_code_comparison()

    print("\n" + "=" * 60)
    print("Analysis complete!")
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Hashing bound** | $Q \geq 1 - H(\text{errors})$ |
| **Threshold** | Error rate where capacity → 0 |
| **Random codes** | Achieve capacity but not constructive |
| **LDPC codes** | Efficient and approach capacity |
| **Code efficiency** | $\eta = R/Q$ — fraction of capacity used |

### Key Equations

$$\boxed{Q_{\text{dep}} \geq 1 - H(p) - p\log_2 3}$$

$$\boxed{p_{\text{th}}^{\text{dep}} \approx 18.93\%}$$

$$\boxed{Q_{\text{erasure}} = 1 - 2p, \quad p_{\text{th}} = 50\%}$$

$$\boxed{\text{Code Efficiency: } \eta = \frac{k/n}{Q(\mathcal{N})}}$$

### Main Takeaways

1. **Hashing bound** gives achievable rate for random codes
2. **Threshold** is where positive-rate error correction becomes impossible
3. **Depolarizing threshold** is ~18.9% (code capacity)
4. **Practical codes** often operate far below capacity
5. **Good qLDPC codes** can approach capacity with polynomial complexity

---

## Daily Checklist

- [ ] I can derive the hashing bound for Pauli channels
- [ ] I understand how random codes achieve the bound
- [ ] I can compute thresholds for different noise models
- [ ] I know how biased noise affects thresholds
- [ ] I can evaluate code efficiency relative to capacity
- [ ] I completed the computational lab

---

## Preview: Day 725

Tomorrow we study **LDPC Code Capacity**, including:
- Classical LDPC codes and belief propagation
- Quantum LDPC constructions
- Good qLDPC codes (constant rate, linear distance)
- Approaches to capacity-achieving codes
