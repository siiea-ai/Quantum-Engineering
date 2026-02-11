# Day 983: Good qLDPC Codes: Constant Rate & Distance

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Good Code Theory & Historical Context |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Expansion & Existence Proofs |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Parameter Analysis |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 983, you will be able to:

1. Define "good" quantum codes: constant rate AND linear distance
2. Explain why hypergraph product codes are not "good"
3. Describe the quantum Gilbert-Varshamov bound
4. Identify the role of expansion in achieving good codes
5. Outline the historical barriers to good qLDPC construction
6. Appreciate the 2021-2022 breakthrough significance

---

## Core Content

### 1. What Makes a Code "Good"?

In coding theory, a family of codes $\{C_n\}$ is called **asymptotically good** if both the rate and relative distance remain constant as $n \to \infty$.

**Classical Definition:**

A family of $[n, k, d]$ codes is good if:
$$R = \frac{k}{n} = \Theta(1) \text{ (constant rate)}$$
$$\delta = \frac{d}{n} = \Theta(1) \text{ (constant relative distance)}$$

**Quantum Definition:**

A family of $[[n, k, d]]$ quantum codes is good if:

$$\boxed{k = \Theta(n) \text{ and } d = \Theta(n)}$$

Equivalently:
$$\text{Rate: } R = k/n \geq R_0 > 0 \text{ for all } n$$
$$\text{Relative distance: } \delta = d/n \geq \delta_0 > 0 \text{ for all } n$$

---

### 2. Why "Good" Matters: The Overhead Problem

**Surface Code Scaling:**

For the surface code with distance $d$:
- Physical qubits: $n = O(d^2)$
- Logical qubits: $k = 1$
- Rate: $R = O(1/d^2) \to 0$ as $d \to \infty$

**Overhead per logical qubit:**
$$\text{Overhead} = \frac{n}{k} = O(d^2)$$

For error rate $p$ requiring distance $d \approx 1/p$:
$$\text{Overhead} = O(1/p^2)$$

**Good Code Scaling:**

For a good code with $k = \Theta(n)$ and $d = \Theta(n)$:
$$\text{Overhead} = \frac{n}{k} = O(1)$$

**The Implication:**

| Code Type | 1M logical qubits at $d=100$ |
|-----------|------------------------------|
| Surface | $\sim 10^{10}$ physical qubits |
| Good qLDPC | $\sim 10^7$ physical qubits |

A **1000x improvement** in asymptotic scaling!

---

### 3. The Quantum Gilbert-Varshamov Bound

Classical coding theory has the Gilbert-Varshamov (GV) bound showing good codes exist.

**Classical GV Bound:**

There exist $[n, k, d]$ codes with:
$$\frac{k}{n} \geq 1 - H_2\left(\frac{d}{n}\right) + o(1)$$

where $H_2(x) = -x\log_2(x) - (1-x)\log_2(1-x)$ is binary entropy.

**Quantum GV Bound:**

There exist $[[n, k, d]]$ quantum codes with:
$$\frac{k}{n} \geq 1 - 2H_2\left(\frac{d}{n}\right) + o(1)$$

**Interpretation:**

The quantum GV bound guarantees the **existence** of good quantum codes! The factor of 2 comes from needing to handle both X and Z errors.

**The Catch:**

Random codes achieve GV bound but are:
- Dense (no sparsity structure)
- Exponentially hard to decode
- Not constructive

**The Challenge:** Can we achieve GV-like parameters with **sparse** (LDPC) structure?

---

### 4. Why Hypergraph Product Codes Are Not Good

Recall the hypergraph product construction:

$$[[n_1 n_2 + m_1 m_2, k_1 k_2, \min(d_1, d_2)]]$$

**Rate Analysis:**

If input codes have rate $R_i = k_i/n_i$:
$$R_{\text{HP}} = \frac{k_1 k_2}{n_1 n_2 + m_1 m_2} \approx R_1 R_2$$

For constant rate inputs, output has constant rate. **Rate is good!**

**Distance Analysis:**

The hypergraph product distance is:
$$d_{\text{HP}} = \min(d_1, d_2)$$

If inputs have $d_i = \Theta(\sqrt{n_i})$ (typical for random LDPC):
$$d_{\text{HP}} = \Theta(\sqrt{n_1}) = \Theta(n^{1/4})$$

**Relative distance goes to zero!** Not good.

**The Bottleneck:**

Even if classical codes have linear distance $d_i = \Theta(n_i)$, the hypergraph product gives:
$$d_{\text{HP}} = \Theta(n_1) = \Theta(\sqrt{n_{\text{HP}}})$$

Distance scales as **square root** of block length, not linearly.

$$\delta = \frac{d}{n} = \frac{\Theta(\sqrt{n})}{n} = \Theta(n^{-1/2}) \to 0$$

---

### 5. The Expansion Property

The key to achieving linear distance is **expansion** - a graph property where small sets have large boundaries.

**Definition:** A bipartite graph $G = (L, R, E)$ is a $(c, d)$-expander if for every set $S \subseteq L$ with $|S| \leq c|L|$:
$$|\mathcal{N}(S)| \geq d|S|$$

where $\mathcal{N}(S)$ is the set of neighbors of $S$.

**Spectral Gap:**

For a $d$-regular graph with adjacency matrix $A$:
$$\lambda_2 \leq d - \Omega(1)$$

implies expansion. Here $\lambda_2$ is the second-largest eigenvalue.

**Ramanujan Graphs:**

Optimal expanders satisfying:
$$\lambda_2 \leq 2\sqrt{d-1}$$

**Why Expansion Helps:**

In a code's Tanner graph:
- Low-weight errors have small support
- Expansion ensures small support $\Rightarrow$ non-zero syndrome
- Therefore: non-zero syndrome $\Rightarrow$ high-weight error
- Result: High minimum distance!

---

### 6. Historical Barriers

The quest for good qLDPC codes spanned 25+ years.

**Early Attempts (1996-2010):**

| Approach | Rate | Distance | Status |
|----------|------|----------|--------|
| CSS from random LDPC | $\Theta(1)$ | $O(\sqrt{n}\log n)$ | Not good |
| Hypergraph product | $\Theta(1)$ | $O(\sqrt{n})$ | Not good |
| Topological (surface) | $O(1/d^2)$ | $O(\sqrt{n})$ | Not good |

**The "No-Go" Intuition:**

Many believed that the CSS constraint $H_X H_Z^T = 0$ fundamentally limited qLDPC codes. Arguments:
1. Self-orthogonality is restrictive
2. Sparse + self-orthogonal $\Rightarrow$ algebraic constraints
3. Quantum errors are "harder" than classical

**Progress (2010-2020):**

| Construction | Year | Parameters | Notes |
|--------------|------|------------|-------|
| Homological codes | 2010s | Various | Limited distance |
| Fiber bundle (3D) | 2021 | $d = n^{1/2}$ | 3D local! |
| Expander product | 2020 | Approaching | Not quite good |

**The Breakthrough (2021-2022):**

Three independent groups achieved good qLDPC:
1. **Panteleev-Kalachev** (2021): Lifted product codes
2. **Leverrier-Zémor** (2022): Quantum Tanner codes
3. **Dinur et al.** (2022): Good with linear-time decoding

---

### 7. The Key Insight: Product of Expanders

The breakthrough insight: take the hypergraph product of **expanding** classical codes with specific algebraic structure.

**Classical Expander Codes:**

Sipser-Spielman (1996) showed that classical codes from expander graphs have:
- Constant rate
- Linear distance
- Efficient decoding

**The Quantum Challenge:**

Direct hypergraph product of expander codes still gives $d = O(\sqrt{n})$.

**The Solution:**

Use more sophisticated products:
1. **Lifted products:** Algebraically lift over group actions
2. **Balanced products:** Carefully balance expansion on both sides
3. **Left-right Cayley complexes:** Exploit double expansion

The magic happens when expansion properties "multiply" rather than taking minimum.

---

### 8. The Breakthrough Parameters

**Panteleev-Kalachev Codes (2021):**

$$\boxed{[[n, k, d]] = [[n, \Omega(n), \Omega(\sqrt{n}\log n)]]}$$

First construction with $d = \omega(\sqrt{n})$!

**Improved Version (2022):**

$$\boxed{[[n, k, d]] = [[n, \Theta(n), \Theta(n)]]}$$

Truly good codes exist!

**Leverrier-Zémor Quantum Tanner Codes (2022):**

$$[[n, k, d]] = [[n, cn, \delta n]]$$

for explicit constants $c, \delta > 0$.

**Dinur et al. (2022):**

Good qLDPC with **linear-time decoding**!

---

### 9. Comparison of Code Families

| Code Family | Rate $k/n$ | Distance $d$ | Locality | Decoding |
|-------------|------------|--------------|----------|----------|
| Surface | $O(1/d^2)$ | $O(\sqrt{n})$ | 2D local | $O(n)$ |
| Hypergraph Product | $\Theta(1)$ | $O(\sqrt{n})$ | Non-local | $O(n^2)$ |
| Fiber Bundle | $\Theta(1)$ | $O(\sqrt{n})$ | 3D local | $O(n)$ |
| **Good qLDPC** | $\Theta(1)$ | $\Theta(n)$ | Non-local | $O(n)$-$O(n^2)$ |

---

## Practical Applications

### Implications for Fault-Tolerant Quantum Computing

**Memory Overhead:**

For storing $N$ logical qubits with error rate $p$:

| Code | Physical Qubits |
|------|-----------------|
| Surface | $O(N/p^2)$ |
| Good qLDPC | $O(N)$ |

**Computation Overhead:**

Fault-tolerant gates on good codes:
- Transversal gates limited (Eastin-Knill theorem)
- Magic state distillation still needed
- But base overhead is constant!

**Timeline:**

- Near-term (2025-2030): Surface codes dominate (2D hardware)
- Medium-term (2030-2040): Bivariate bicycle codes (moderate non-locality)
- Long-term (2040+): Full qLDPC with advanced connectivity

---

## Worked Examples

### Example 1: Rate Comparison

**Problem:** Compare the rate of a surface code and a hypothetical good qLDPC code, both at distance $d = 50$.

**Solution:**

**Surface Code:**
$$n_{\text{surface}} = 2d^2 - 1 = 2(50)^2 - 1 = 4999$$
$$k = 1$$
$$R_{\text{surface}} = \frac{1}{4999} \approx 0.0002$$

**Good qLDPC (assume $k = 0.1n$, $d = 0.1n$):**

For $d = 50$: $n = 500$ qubits
$$k = 0.1 \times 500 = 50 \text{ logical qubits}$$
$$R_{\text{qLDPC}} = 0.1$$

**Comparison:**

To encode 50 logical qubits at distance 50:
- Surface: $50 \times 4999 = 249,950$ physical qubits
- Good qLDPC: $500$ physical qubits

**Factor of 500x improvement!**

---

### Example 2: GV Bound Calculation

**Problem:** Using the quantum Gilbert-Varshamov bound, what is the maximum achievable rate for a quantum code with relative distance $\delta = 0.1$?

**Solution:**

$$R \geq 1 - 2H_2(\delta)$$

where $H_2(x) = -x \log_2(x) - (1-x)\log_2(1-x)$.

$$H_2(0.1) = -0.1 \log_2(0.1) - 0.9\log_2(0.9)$$
$$= -0.1 \times (-3.322) - 0.9 \times (-0.152)$$
$$= 0.332 + 0.137 = 0.469$$

$$R \geq 1 - 2(0.469) = 1 - 0.938 = 0.062$$

**Result:** Quantum codes with 10% relative distance can have rate up to ~6%.

Good qLDPC codes approach this bound!

---

### Example 3: Distance Scaling

**Problem:** A code family has $n = 1000m^2$, $k = 100m^2$, $d = 10m$ for parameter $m$. Is this family good?

**Solution:**

Express everything in terms of $n$:
$$n = 1000m^2 \Rightarrow m = \sqrt{n/1000}$$

$$k = 100m^2 = 100 \cdot \frac{n}{1000} = \frac{n}{10} = 0.1n$$

$$d = 10m = 10\sqrt{n/1000} = \frac{10}{\sqrt{1000}}\sqrt{n} \approx 0.316\sqrt{n}$$

**Analysis:**
- Rate: $R = k/n = 0.1 = \Theta(1)$ - Constant! Good.
- Relative distance: $\delta = d/n = 0.316/\sqrt{n} \to 0$ - Not constant. Bad!

**Conclusion:** This family is NOT good. Distance grows as $\sqrt{n}$, not $n$.

---

## Practice Problems

### Level 1: Direct Application

1. **Good Code Check:** A code family has $[[10^6, 10^5, 10^3]]$. Is it asymptotically good? Compute rate and relative distance.

2. **Overhead Calculation:** Compare the overhead (physical/logical) for surface code at $d=7$ versus a good qLDPC code with $k/n = 0.05$ and $d/n = 0.05$ at equivalent distance.

3. **GV Bound:** Compute the quantum GV bound rate limit for $\delta = 0.05$.

### Level 2: Intermediate

4. **Expansion Requirement:** If a Tanner graph has expansion $|\mathcal{N}(S)| \geq 1.5|S|$ for $|S| \leq 0.1n$, estimate the minimum distance of the corresponding code.

5. **Hypergraph Product Limitation:** Prove that if input codes to hypergraph product have $d_i = c\sqrt{n_i}$, the output has relative distance $\delta \to 0$.

6. **Rate Trade-off:** Using the quantum GV bound, plot achievable rate vs relative distance for $\delta \in [0, 0.25]$.

### Level 3: Challenging

7. **Spectral Gap:** Research and explain how the spectral gap of a Cayley graph relates to the expansion property and code distance.

8. **CSS vs Non-CSS:** Explain why the self-orthogonality constraint $H H^T = 0$ was historically believed to prevent good qLDPC codes. What was missing in this intuition?

9. **Decoding Implications:** If a good qLDPC code has $d = cn$ for constant $c$, what does this imply about the number of correctable errors? How does this affect decoding complexity?

---

## Computational Lab

### Objective
Analyze scaling of various code families and visualize the "good code" regime.

```python
"""
Day 983 Computational Lab: Good qLDPC Codes Analysis
QLDPC Codes & Constant-Overhead QEC - Week 141
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import xlogy

# =============================================================================
# Part 1: Quantum Gilbert-Varshamov Bound
# =============================================================================

print("=" * 70)
print("Part 1: Quantum Gilbert-Varshamov Bound")
print("=" * 70)

def binary_entropy(x):
    """Compute binary entropy H_2(x) in bits."""
    if x <= 0 or x >= 1:
        return 0
    return -x * np.log2(x) - (1-x) * np.log2(1-x)

def quantum_gv_bound(delta):
    """Quantum Gilbert-Varshamov bound: max rate for relative distance delta."""
    if delta <= 0:
        return 1.0
    if delta >= 0.5:
        return 0.0
    return max(0, 1 - 2 * binary_entropy(delta))

# Plot GV bound
delta_range = np.linspace(0.001, 0.49, 200)
gv_rates = [quantum_gv_bound(d) for d in delta_range]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(delta_range, gv_rates, 'b-', linewidth=2, label='Quantum GV Bound')
ax1.fill_between(delta_range, 0, gv_rates, alpha=0.3)
ax1.set_xlabel('Relative Distance δ = d/n')
ax1.set_ylabel('Rate R = k/n')
ax1.set_title('Quantum Gilbert-Varshamov Bound')
ax1.set_xlim([0, 0.5])
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3)
ax1.legend()

# Mark key points
ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
ax1.axhline(y=quantum_gv_bound(0.1), color='red', linestyle='--', alpha=0.5)
ax1.annotate(f'δ=0.1: R≤{quantum_gv_bound(0.1):.3f}',
             xy=(0.1, quantum_gv_bound(0.1)), xytext=(0.15, 0.3),
             arrowprops=dict(arrowstyle='->', color='red'))

ax1.text(0.25, 0.5, 'Achievable Region\n(Good codes exist here)',
         fontsize=10, ha='center')

# Classical vs Quantum GV
classical_gv = [1 - binary_entropy(d) for d in delta_range]
ax2 = axes[1]
ax2.plot(delta_range, classical_gv, 'g-', linewidth=2, label='Classical GV')
ax2.plot(delta_range, gv_rates, 'b-', linewidth=2, label='Quantum GV')
ax2.set_xlabel('Relative Distance δ')
ax2.set_ylabel('Rate R')
ax2.set_title('Classical vs Quantum GV Bounds')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_983_gv_bounds.png', dpi=150, bbox_inches='tight')
plt.show()
print("GV bounds saved to 'day_983_gv_bounds.png'")

# =============================================================================
# Part 2: Code Family Scaling Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Code Family Scaling Analysis")
print("=" * 70)

def surface_code(d):
    """Surface code parameters for distance d."""
    n = 2 * d**2 - 1  # Rotated surface code
    k = 1
    return n, k, d

def hypergraph_product_family(base_n):
    """Hypergraph product parameters (simplified model)."""
    # Assume input is [n, n/2, sqrt(n)]
    n = 2 * base_n**2
    k = (base_n // 2)**2
    d = int(np.sqrt(base_n))
    return n, k, d

def good_qldpc(target_d, rate=0.1, rel_dist=0.1):
    """Good qLDPC code parameters."""
    n = int(target_d / rel_dist)
    k = int(rate * n)
    d = target_d
    return n, k, d

# Generate code families
distances = range(3, 51, 2)

surface_data = [surface_code(d) for d in distances]
hp_data = [hypergraph_product_family(d*10) for d in distances]
good_data = [good_qldpc(d) for d in distances]

# Extract for plotting
surface_n = [x[0] for x in surface_data]
surface_k = [x[1] for x in surface_data]
surface_d = [x[2] for x in surface_data]

hp_n = [x[0] for x in hp_data]
hp_k = [x[1] for x in hp_data]
hp_d = [x[2] for x in hp_data]

good_n = [x[0] for x in good_data]
good_k = [x[1] for x in good_data]
good_d = [x[2] for x in good_data]

# Rate vs n
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.loglog(surface_n, [k/n for k, n in zip(surface_k, surface_n)],
           'ro-', linewidth=2, label='Surface Code')
ax1.loglog(hp_n, [k/n for k, n in zip(hp_k, hp_n)],
           'gs-', linewidth=2, label='Hypergraph Product')
ax1.loglog(good_n, [k/n for k, n in zip(good_k, good_n)],
           'b^-', linewidth=2, label='Good qLDPC')
ax1.set_xlabel('Block Length n')
ax1.set_ylabel('Rate k/n')
ax1.set_title('Rate Scaling')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Relative distance vs n
ax2 = axes[0, 1]
ax2.loglog(surface_n, [d/n for d, n in zip(surface_d, surface_n)],
           'ro-', linewidth=2, label='Surface Code')
ax2.loglog(hp_n, [d/n for d, n in zip(hp_d, hp_n)],
           'gs-', linewidth=2, label='Hypergraph Product')
ax2.loglog(good_n, [d/n for d, n in zip(good_d, good_n)],
           'b^-', linewidth=2, label='Good qLDPC')
ax2.set_xlabel('Block Length n')
ax2.set_ylabel('Relative Distance d/n')
ax2.set_title('Relative Distance Scaling')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.1, color='blue', linestyle='--', alpha=0.5, label='Constant δ')

# Overhead per logical qubit
ax3 = axes[1, 0]
ax3.semilogy(distances, [n/k for n, k in zip(surface_n, surface_k)],
             'ro-', linewidth=2, label='Surface Code')
ax3.semilogy(distances, [n/k for n, k in zip(hp_n, hp_k)],
             'gs-', linewidth=2, label='Hypergraph Product')
ax3.semilogy(distances, [n/k for n, k in zip(good_n, good_k)],
             'b^-', linewidth=2, label='Good qLDPC')
ax3.set_xlabel('Target Distance d')
ax3.set_ylabel('Overhead (n/k)')
ax3.set_title('Physical/Logical Overhead')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Total qubits for 1000 logical qubits
ax4 = axes[1, 1]
target_k = 1000
ax4.semilogy(distances, [target_k * n / k for n, k in zip(surface_n, surface_k)],
             'ro-', linewidth=2, label='Surface Code')
ax4.semilogy(distances, [target_k * n / k for n, k in zip(hp_n, hp_k)],
             'gs-', linewidth=2, label='Hypergraph Product')
ax4.semilogy(distances, [target_k * n / k for n, k in zip(good_n, good_k)],
             'b^-', linewidth=2, label='Good qLDPC')
ax4.set_xlabel('Target Distance d')
ax4.set_ylabel('Physical Qubits for 1000 Logical')
ax4.set_title('Scaling for 1000 Logical Qubits')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_983_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
print("Scaling analysis saved to 'day_983_scaling.png'")

# =============================================================================
# Part 3: The "Goodness" Criterion Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Rate-Distance Trade-off")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# GV bound curve
gv_curve = [(d, quantum_gv_bound(d)) for d in np.linspace(0.01, 0.49, 100)]
ax.plot([x[0] for x in gv_curve], [x[1] for x in gv_curve],
        'k-', linewidth=3, label='Quantum GV Bound')

# Plot code families
# Surface codes at various distances (d = 3, 5, 7, ..., 21)
for d in [3, 5, 7, 9, 11, 15, 21]:
    n, k, _ = surface_code(d)
    rate = k / n
    rel_dist = d / n
    if d == 3:
        ax.scatter([rel_dist], [rate], c='red', s=100, marker='o',
                   label='Surface Codes', zorder=5)
    else:
        ax.scatter([rel_dist], [rate], c='red', s=100, marker='o', zorder=5)
    ax.annotate(f'd={d}', (rel_dist, rate), textcoords="offset points",
                xytext=(5, 5), fontsize=8, color='red')

# Hypergraph product codes
for base in [10, 20, 30, 40, 50]:
    n, k, d = hypergraph_product_family(base)
    rate = k / n
    rel_dist = d / n
    if base == 10:
        ax.scatter([rel_dist], [rate], c='green', s=100, marker='s',
                   label='Hypergraph Product', zorder=5)
    else:
        ax.scatter([rel_dist], [rate], c='green', s=100, marker='s', zorder=5)

# Good qLDPC codes (constant rate and relative distance)
for r in [0.05, 0.1, 0.15]:
    for d_rel in [0.05, 0.1, 0.15]:
        if d_rel == 0.1 and r == 0.1:
            ax.scatter([d_rel], [r], c='blue', s=150, marker='^',
                       label='Good qLDPC', zorder=5)
        else:
            ax.scatter([d_rel], [r], c='blue', s=100, marker='^', zorder=5)

# Mark the "good" region
ax.axhline(y=0.02, color='purple', linestyle='--', alpha=0.5)
ax.axvline(x=0.02, color='purple', linestyle='--', alpha=0.5)
ax.fill_between([0.02, 0.5], 0.02, 1, alpha=0.1, color='green')
ax.text(0.25, 0.5, 'GOOD CODE\nREGION', fontsize=14,
        ha='center', va='center', color='green', fontweight='bold')

# Mark surface code trajectory
ax.annotate('Surface codes:\nRate → 0', xy=(0.02, 0.01), xytext=(0.1, 0.03),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

ax.set_xlabel('Relative Distance δ = d/n', fontsize=12)
ax.set_ylabel('Rate R = k/n', fontsize=12)
ax.set_title('Code Families in Rate-Distance Space', fontsize=14)
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.6])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_983_rate_distance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Rate-distance plot saved to 'day_983_rate_distance.png'")

# =============================================================================
# Part 4: Overhead Comparison Table
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Overhead Comparison")
print("=" * 70)

print("\nFor 10,000 logical qubits at distance d:")
print("-" * 70)
print(f"{'d':>5} {'Surface n':>15} {'HP n':>15} {'Good qLDPC n':>15} {'Savings':>12}")
print("-" * 70)

for d in [5, 10, 20, 50, 100]:
    n_surf, k_surf, _ = surface_code(d)
    surf_total = 10000 * n_surf / k_surf

    # Good qLDPC with rate 0.1, rel_dist 0.1
    n_good, k_good, _ = good_qldpc(d, rate=0.1, rel_dist=0.1)
    good_total = 10000 * n_good / k_good

    # HP (approximate)
    n_hp, k_hp, d_hp = hypergraph_product_family(d*10)
    if d_hp >= d:
        hp_total = 10000 * n_hp / k_hp
    else:
        hp_total = float('inf')

    savings = surf_total / good_total if good_total > 0 else float('inf')

    hp_str = f"{int(hp_total):,}" if hp_total < float('inf') else "N/A"
    print(f"{d:>5} {int(surf_total):>15,} {hp_str:>15} {int(good_total):>15,} {savings:>11.0f}x")

# =============================================================================
# Part 5: Timeline and Key Breakthroughs
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Historical Timeline")
print("=" * 70)

timeline = """
TIMELINE OF QUANTUM LDPC CODE DEVELOPMENT:

1995: Shor code, Steane code - First quantum error correcting codes
1996: CSS construction - Systematic code building
1997: Quantum threshold theorem - FT possible!
      Surface codes introduced (Kitaev)

2000s: Topological codes dominate
       Surface code becomes "default" for FTQC

2009: Tillich-Zémor hypergraph product
      Rate Θ(1) but distance O(√n)

2015: Fiber bundle attempts
      Progress toward 3D locality

2020: Hastings et al. fiber bundle codes
      d = Ω(√n log n), 3D local

2021: BREAKTHROUGH YEAR!
      Panteleev-Kalachev: First d = ω(√n) qLDPC
      "Asymptotically Good QLDPC Codes"

2022: Complete resolution!
      Panteleev-Kalachev: d = Θ(n) achieved
      Leverrier-Zémor: Quantum Tanner codes
      Dinur et al.: Linear-time decoding

2023-2024: Implementation focus
      IBM: Bivariate bicycle codes
      Focus shifts to practical realizations

CURRENT STATUS (2026):
      Good qLDPC codes EXIST
      Challenge: Practical implementation
      Non-locality remains the bottleneck
"""
print(timeline)

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary: The Good qLDPC Revolution")
print("=" * 70)

summary = """
KEY TAKEAWAYS:

1. GOOD CODE DEFINITION:
   - Rate R = k/n = Θ(1)       [doesn't vanish]
   - Rel. distance δ = d/n = Θ(1)  [doesn't vanish]

2. QUANTUM GV BOUND:
   R ≥ 1 - 2H₂(δ)
   Shows good codes EXIST but doesn't construct them

3. WHY HYPERGRAPH PRODUCT ISN'T ENOUGH:
   - Rate is constant: ✓
   - Distance is O(√n): ✗
   - Relative distance → 0

4. THE BREAKTHROUGH (2021-2022):
   - Lifted product codes (Panteleev-Kalachev)
   - Quantum Tanner codes (Leverrier-Zémor)
   - Key: Expansion properties compound correctly

5. PRACTICAL IMPLICATIONS:
   - Surface code: O(d²) overhead per logical qubit
   - Good qLDPC: O(1) overhead per logical qubit
   - Potential 100-1000x reduction in physical qubits!

6. THE CATCH:
   - Non-local connectivity required
   - Complex syndrome extraction
   - Current hardware not ready

Tomorrow: Panteleev-Kalachev and Quantum Tanner code constructions!
"""
print(summary)

print("\n" + "=" * 70)
print("Day 983 Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Good code definition | $k = \Theta(n)$, $d = \Theta(n)$ |
| Quantum GV bound | $R \geq 1 - 2H_2(\delta)$ |
| Surface code rate | $R = O(1/d^2)$ |
| Surface code overhead | $O(d^2)$ per logical qubit |
| Good qLDPC overhead | $O(1)$ per logical qubit |

### Main Takeaways

1. **Good codes** have both constant rate AND constant relative distance
2. **Hypergraph product** achieves constant rate but distance scales as $\sqrt{n}$
3. **Quantum GV bound** proves good codes exist (non-constructively)
4. **Expansion** is the key property enabling linear distance
5. **2021-2022 breakthroughs** finally constructed explicit good qLDPC codes
6. **Practical implications**: Potentially 100-1000x reduction in qubit overhead

---

## Daily Checklist

- [ ] Define "good" codes precisely
- [ ] Explain why hypergraph product distance is $O(\sqrt{n})$
- [ ] State the quantum Gilbert-Varshamov bound
- [ ] Describe the role of expansion
- [ ] Complete Level 1 practice problems
- [ ] Attempt Level 2 problems
- [ ] Run computational lab and interpret plots
- [ ] Explain the significance of the 2021-2022 breakthroughs

---

## Preview: Day 984

Tomorrow we dive into the **Panteleev-Kalachev and Quantum Tanner code constructions** that achieved the impossible:
- Lifted product codes over group algebras
- Cayley graphs and double expansion
- Left-right Cayley complexes
- Explicit parameter calculations
- The algebraic magic that makes it work

---

*"The existence of good quantum LDPC codes settles one of the most important open problems in quantum information theory."*
--- Commentary on the 2021-2022 breakthroughs

---

**Next:** Day 984 - Panteleev-Kalachev & Quantum Tanner Codes
