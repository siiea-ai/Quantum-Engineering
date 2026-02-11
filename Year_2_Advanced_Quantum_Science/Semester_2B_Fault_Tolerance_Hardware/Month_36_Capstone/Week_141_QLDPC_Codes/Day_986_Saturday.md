# Day 986: QLDPC vs Surface Code Comparison

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Detailed Feature Comparison |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Trade-off Analysis & Decision Framework |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Comparative Analysis |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 986, you will be able to:

1. Compare surface codes and qLDPC codes across multiple dimensions
2. Analyze locality constraints and hardware implications
3. Evaluate decoding complexity and latency trade-offs
4. Assess syndrome extraction circuit requirements
5. Apply a decision framework for code selection
6. Predict which code family suits different applications

---

## Core Content

### 1. Master Comparison Table

| Property | Surface Code | Good qLDPC | Winner |
|----------|--------------|------------|--------|
| **Rate** $k/n$ | $O(1/d^2) \to 0$ | $\Theta(1)$ | qLDPC |
| **Distance** | $d = O(\sqrt{n})$ | $d = \Theta(n)$ | qLDPC |
| **Overhead** | $O(d^2)$ per logical | $O(1)$ per logical | qLDPC |
| **Locality** | 2D planar | Non-local | Surface |
| **Stabilizer weight** | 4 | $O(1)$ but ~10-20 | Surface |
| **Threshold** | ~1% | ~0.1% | Surface |
| **Decoding** | $O(n)$ or $O(n^{3/2})$ | $O(n)$ to $O(n^2)$ | Tie |
| **Hardware maturity** | Demonstrated | Experimental | Surface |
| **Gate implementation** | Lattice surgery | Complex | Surface |
| **Scalability** | Polynomial overhead | Constant overhead | qLDPC |

---

### 2. Locality Analysis

**Surface Code Locality:**

- Each qubit interacts with at most 4 neighbors
- Perfect for 2D planar chip architectures
- Syndrome extraction with local gates only

```
Weight-4 Stabilizer:
    |
  --X--
    |

All interactions are nearest-neighbor in 2D
```

**qLDPC Non-Locality:**

- Each qubit participates in $O(1)$ stabilizers, but...
- Stabilizers connect qubits across long distances
- Requires either:
  - All-to-all connectivity
  - Reconfigurable connections
  - 3D+ architectures
  - Long-range entanglement protocols

```
Non-local Stabilizer (schematic):
Qubit 1 -------- Qubit 47
   \              /
    \   Qubit 156
     \    /
      \  /
    Qubit 203

Connections span entire code block
```

**Locality Metrics:**

Define interaction distance:
$$D_{\text{int}} = \max_{S} \max_{i,j \in S} d(i, j)$$

where $S$ ranges over stabilizers and $d(i,j)$ is physical distance.

| Code | $D_{\text{int}}$ |
|------|------------------|
| Surface | $O(1)$ (constant) |
| qLDPC | $O(\sqrt{n})$ to $O(n)$ |

---

### 3. Decoding Complexity

**Surface Code Decoding:**

**Minimum-Weight Perfect Matching (MWPM):**
- Build graph of syndrome defects
- Find minimum-weight matching
- Complexity: $O(n^3)$ naive, $O(n^{3/2})$ optimized

**Union-Find Decoder:**
- Near-linear time: $O(n \cdot \alpha(n))$
- Slightly suboptimal but fast
- Practical for real-time decoding

**qLDPC Decoding:**

**Belief Propagation (BP):**
- Natural for LDPC structure
- Complexity: $O(n \cdot w \cdot \text{iterations})$
- Challenge: degeneracy causes BP to fail/oscillate

**BP + OSD (Ordered Statistics Decoding):**
- BP provides soft information
- OSD explores likely error patterns
- Complexity: $O(n^2)$ to $O(n^3)$

**Linear-Time Decoders (Dinur et al.):**
- Exist in theory
- Complexity: $O(n)$
- Practical implementation still developing

**Comparison:**

| Decoder | Complexity | Practical Speed | Accuracy |
|---------|------------|-----------------|----------|
| MWPM (surface) | $O(n^{3/2})$ | Fast | Optimal |
| Union-Find | $O(n \alpha(n))$ | Very fast | Near-optimal |
| BP (qLDPC) | $O(n)$ | Fast | Suboptimal |
| BP+OSD | $O(n^2)$ | Moderate | Good |
| Linear-time | $O(n)$ | Fast | Optimal |

---

### 4. Syndrome Extraction Circuits

**Surface Code Syndrome:**

```
Ancilla prep: |+⟩ or |0⟩
       |
     CNOT × 4 (to data qubits)
       |
    Measure
```

- Depth: $O(1)$ (constant, ~6 time steps)
- Gates: 4 CNOTs per stabilizer
- All operations local

**qLDPC Syndrome:**

```
Ancilla prep: |+⟩ or |0⟩
       |
     CNOT × w (w = stabilizer weight, ~10-20)
       |
    Measure
```

- Depth: $O(w)$ if sequential, $O(\log w)$ if parallel with auxiliary qubits
- Gates: $w$ CNOTs per stabilizer
- Non-local operations required

**Depth Comparison:**

| Operation | Surface | qLDPC |
|-----------|---------|-------|
| One round | 6 steps | 20-50 steps |
| Error per round | $6p$ | $50p$ |
| Effective threshold | $p_{\text{th}}$ | $p_{\text{th}}/8$ |

The higher syndrome depth partly offsets qLDPC's rate advantage!

---

### 5. Threshold Analysis

**Surface Code Threshold:**

From extensive simulations:
$$p_{\text{th}}^{\text{surface}} \approx 1.0\% \text{ to } 1.1\%$$

With phenomenological noise:
$$p_{\text{th}}^{\text{surface}} \approx 2.9\%$$

**qLDPC Threshold:**

Depends heavily on:
- Stabilizer weight $w$
- Relative distance $\delta$
- Decoder efficiency

Estimates:
$$p_{\text{th}}^{\text{qLDPC}} \approx 0.1\% \text{ to } 0.5\%$$

**Trade-off:**

Surface has 5-10x higher threshold, but...
- Below threshold: qLDPC overhead stays constant
- Surface overhead grows as $O(\log^2(1/\epsilon))$

**Crossover Analysis:**

At what error rate does qLDPC become preferable?

Define effective overhead:
$$\text{Overhead}_{\text{eff}} = \text{Overhead} \times (\text{Target}/\text{Achieved threshold})^a$$

For large computations requiring very low logical error rates, qLDPC wins despite lower threshold.

---

### 6. Gate Implementation

**Surface Code Gates:**

**Transversal:**
- CNOT: Between two patches
- Pauli: Single-qubit

**Lattice Surgery:**
- Logical CNOT via merge/split
- Requires $O(d)$ time per gate
- Space-time trade-offs possible

**Magic States:**
- Inject T-gates via teleportation
- Distillation factories are large

**qLDPC Gates:**

**Transversal:**
- Similar Pauli and CNOT structure
- Non-local connectivity makes this easier!

**Non-Clifford:**
- Magic state distillation still required
- But: fewer levels needed (lower base error)
- Or: code switching/surgery (complex)

**Comparison:**

| Gate Type | Surface | qLDPC |
|-----------|---------|-------|
| Pauli | Trivial | Trivial |
| CNOT | Lattice surgery | Transversal (easier!) |
| Hadamard | Transversal | Depends on code |
| T | Magic states | Magic states |
| CCZ | Multiple T | Multiple T |

---

### 7. Hardware Platform Suitability

**Superconducting Qubits:**

- Naturally 2D planar
- Surface code: **Excellent fit**
- qLDPC: Requires 3D packaging or long-range couplers
- Current status: Surface code demonstrated

**Trapped Ions:**

- Moderate connectivity (shuttle-based)
- Surface code: Adaptable
- qLDPC: Better fit than superconducting
- Current status: Small codes demonstrated

**Neutral Atoms:**

- Reconfigurable arrays
- Can create arbitrary connectivity
- Surface code: Works, but underutilizes capability
- qLDPC: **Natural fit**
- Current status: Rapidly advancing

**Photonics:**

- Long-range entanglement natural
- Surface code: Inefficient use
- qLDPC: **Excellent fit**
- Current status: Measurement-based approaches

**Summary:**

| Platform | Surface | qLDPC |
|----------|---------|-------|
| Superconducting | Excellent | Poor |
| Trapped ions | Good | Moderate |
| Neutral atoms | Good | Excellent |
| Photonics | Moderate | Excellent |

---

### 8. Decision Framework

**When to Choose Surface Codes:**

1. Hardware is fundamentally 2D (superconducting chips)
2. Physical error rates near or above 0.1%
3. Near-term demonstrations needed
4. Moderate logical qubit count (<1000)
5. Well-understood toolchain important

**When to Choose qLDPC Codes:**

1. Hardware supports reconfigurable connectivity
2. Physical error rates well below 0.1%
3. Very large logical qubit count needed (>10,000)
4. Long-term scalability is priority
5. Willing to invest in new toolchain development

**Hybrid Approach:**

For intermediate regimes:
- Use **bivariate bicycle codes** (moderate qLDPC)
- Or **concatenation**: qLDPC outer, surface inner
- Trade-off: partial benefits of both

---

## Practical Applications

### Case Study: 1 Million Logical Qubit Quantum Computer

**Target Specs:**
- $10^6$ logical qubits
- Logical error rate: $10^{-12}$ per cycle
- T-gate rate: $10^9$ per second

**Surface Code Design:**

Required distance: $d \approx 25$
Physical qubits per logical: $\sim 625$
Magic state factories: $\sim 10^6$

**Total physical qubits: $\sim 10^{12}$** (one trillion!)

**qLDPC Design:**

Rate: $R = 0.05$
Physical qubits per logical: $\sim 20$
Magic state factories: $\sim 10^5$

**Total physical qubits: $\sim 10^8$** (100 million)

**Reduction: 10,000x**

---

## Worked Examples

### Example 1: Locality Impact on Error Rate

**Problem:** A surface code has syndrome circuit depth 6. A qLDPC has depth 40. If physical error rate is $p = 10^{-3}$, compare the effective error accumulation per round.

**Solution:**

**Surface code:**
Errors per round $\approx 6 \times n \times p = 6 \times 10^{-3} \times n$

**qLDPC:**
Errors per round $\approx 40 \times n \times p = 40 \times 10^{-3} \times n$

**Ratio:** qLDPC accumulates $40/6 \approx 6.7$ times more errors per round.

**Impact:** This effectively reduces the qLDPC threshold by factor of ~7.

To compensate, qLDPC needs physical $p < 0.15 \times 10^{-3}$ for equivalent performance.

---

### Example 2: Code Selection

**Problem:** A trapped-ion system has 100 qubits with all-to-all connectivity and $p = 2 \times 10^{-3}$. Which code family is better for demonstrating fault tolerance?

**Solution:**

**Analysis:**

1. **Qubit count:** 100 is small
   - Surface: $d=5$ gives 25 qubits for 1 logical
   - qLDPC: Hard to find good small codes

2. **Error rate:** $p = 2 \times 10^{-3}$
   - Surface threshold: ~1%, so $p/p_{\text{th}} = 0.2$ (good!)
   - qLDPC threshold: ~0.3%, so $p/p_{\text{th}} = 0.67$ (marginal)

3. **Connectivity:** All-to-all favors qLDPC, but...

4. **Maturity:** Surface has proven toolchain

**Recommendation:** Use **surface code** for this system.
- 4 logical qubits at $d=5$ (100 physical)
- Or 1 logical at $d=9$ (81 physical)
- Save qLDPC for lower error rates

---

### Example 3: Future Scaling

**Problem:** Project when qLDPC becomes preferable to surface code for a superconducting system roadmap.

**Solution:**

**Current (2026):**
- Qubits: 1,000
- Error rate: $10^{-3}$
- Surface code dominates

**2030 projection:**
- Qubits: 10,000
- Error rate: $10^{-4}$
- Surface still preferred (error rate above qLDPC threshold)

**2035 projection:**
- Qubits: 100,000
- Error rate: $3 \times 10^{-5}$
- qLDPC becomes viable
- 3D packaging or modular architecture needed

**Crossover:** When $p < 0.1\%$ AND qubit count > 10,000 AND connectivity improves, qLDPC wins.

---

## Practice Problems

### Level 1: Direct Application

1. **Overhead Comparison:** Surface code at $d=15$ vs qLDPC with rate 0.08. Which has lower qubit overhead?

2. **Stabilizer Weight:** Surface has weight 4, qLDPC has weight 12. How many more CNOTs per syndrome round for qLDPC (per stabilizer)?

3. **Threshold Check:** Physical error rate is $5 \times 10^{-4}$. Is this below threshold for (a) surface code, (b) qLDPC with $p_{\text{th}} = 3 \times 10^{-4}$?

### Level 2: Intermediate

4. **Effective Threshold:** If syndrome circuit depth scales as $O(w)$ where $w$ is stabilizer weight, derive the effective threshold in terms of bare threshold and $w$.

5. **Platform Selection:** You have a neutral atom system with 1,000 qubits, reconfigurable connectivity, and $p = 5 \times 10^{-4}$. Recommend a code family and justify.

6. **Gate Overhead:** Compare the time to perform 1000 logical CNOTs on surface code (via lattice surgery, $O(d)$ each) vs qLDPC (transversal, $O(1)$ each) at $d = 20$.

### Level 3: Challenging

7. **Hybrid Design:** Propose a concatenated code with surface code as inner code and qLDPC as outer code. What are the trade-offs?

8. **Decoder Race:** The decoder must complete before the next syndrome round (backlog prevention). For a 1 MHz syndrome rate, what decoding complexity is acceptable?

9. **Economic Analysis:** Factor in hardware costs. If long-range connections cost 10x per qubit, when does qLDPC still save overall resources?

---

## Computational Lab

### Objective
Comprehensive comparison of surface code and qLDPC across multiple metrics.

```python
"""
Day 986 Computational Lab: QLDPC vs Surface Code Comparison
QLDPC Codes & Constant-Overhead QEC - Week 141
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# =============================================================================
# Part 1: Comprehensive Metric Comparison
# =============================================================================

print("=" * 70)
print("Part 1: Multi-Dimensional Comparison")
print("=" * 70)

# Define code families
def surface_code_metrics(d):
    """Compute metrics for surface code at distance d."""
    n = d ** 2
    k = 1
    rate = k / n
    threshold = 0.01
    stab_weight = 4
    syndrome_depth = 6
    decoder_complexity = n ** 1.5  # MWPM
    locality = 1  # 2D local

    return {
        'n': n, 'k': k, 'd': d,
        'rate': rate,
        'threshold': threshold,
        'stab_weight': stab_weight,
        'syndrome_depth': syndrome_depth,
        'decoder_complexity': decoder_complexity,
        'locality': locality,
        'overhead': n / k
    }

def qldpc_metrics(n, rate=0.1, rel_dist=0.1, stab_weight=12):
    """Compute metrics for qLDPC code."""
    k = int(n * rate)
    d = int(n * rel_dist)
    threshold = 0.003  # Lower threshold
    syndrome_depth = 4 * stab_weight  # Higher
    decoder_complexity = n ** 2  # BP + OSD
    locality = np.sqrt(n)  # Non-local

    return {
        'n': n, 'k': k, 'd': d,
        'rate': rate,
        'threshold': threshold,
        'stab_weight': stab_weight,
        'syndrome_depth': syndrome_depth,
        'decoder_complexity': decoder_complexity,
        'locality': locality,
        'overhead': n / k
    }

# Compare at similar distances
distances = [5, 7, 9, 11, 15, 21, 31]

print("\nComparison at various distances:")
print(f"{'d':>4} | {'Surface':>12} | {'qLDPC':>12} | {'OH Ratio':>10}")
print(f"{'':>4} | {'Overhead':>12} | {'Overhead':>12} | {'qLDPC/Surf':>10}")
print("-" * 50)

for d in distances:
    surf = surface_code_metrics(d)

    # qLDPC with same distance
    n_qldpc = int(d / 0.1)  # rel_dist = 0.1
    qldpc = qldpc_metrics(n_qldpc)

    ratio = qldpc['overhead'] / surf['overhead']
    print(f"{d:>4} | {surf['overhead']:>12.1f} | {qldpc['overhead']:>12.1f} | {ratio:>10.2f}")

# =============================================================================
# Part 2: Radar Chart Comparison
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Radar Chart Visualization")
print("=" * 70)

def normalize_metrics(metrics, reference):
    """Normalize metrics relative to reference (higher is better)."""
    normalized = {}

    # Rate: higher is better
    normalized['Rate'] = metrics['rate'] / reference['rate']

    # Threshold: higher is better
    normalized['Threshold'] = metrics['threshold'] / reference['threshold']

    # Overhead: lower is better, so invert
    normalized['Efficiency'] = reference['overhead'] / metrics['overhead']

    # Locality: lower is better (1 is ideal)
    normalized['Locality'] = 1 / metrics['locality']

    # Decoder: lower is better
    normalized['Decoding'] = np.sqrt(reference['decoder_complexity'] / metrics['decoder_complexity'])

    # Syndrome depth: lower is better
    normalized['Syndrome'] = reference['syndrome_depth'] / metrics['syndrome_depth']

    return normalized

# Use medium-scale codes
surf = surface_code_metrics(15)
qldpc = qldpc_metrics(1000)

# Normalize both to surface code
surf_norm = normalize_metrics(surf, surf)
qldpc_norm = normalize_metrics(qldpc, surf)

# Radar chart
categories = list(surf_norm.keys())
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# Surface code
values_surf = [surf_norm[cat] for cat in categories]
values_surf += values_surf[:1]
ax.plot(angles, values_surf, 'o-', linewidth=2, label='Surface Code', color='red')
ax.fill(angles, values_surf, alpha=0.25, color='red')

# qLDPC
values_qldpc = [qldpc_norm[cat] for cat in categories]
values_qldpc += values_qldpc[:1]
ax.plot(angles, values_qldpc, 's-', linewidth=2, label='Good qLDPC', color='blue')
ax.fill(angles, values_qldpc, alpha=0.25, color='blue')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 2.5)
ax.set_title('Code Family Comparison\n(normalized to surface code)', size=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('day_986_radar.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nRadar chart saved to 'day_986_radar.png'")

# =============================================================================
# Part 3: Scaling Crossover Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Scaling Crossover Analysis")
print("=" * 70)

def total_resources(code_type, logical_qubits, target_error, **kwargs):
    """
    Calculate total physical resources for a computation.
    """
    if code_type == 'surface':
        # Find required distance
        p_phys = kwargs.get('p_phys', 1e-3)
        p_th = 0.01
        d = 3
        while (p_phys / p_th) ** ((d+1)/2) > target_error:
            d += 2

        overhead = d ** 2
        total = logical_qubits * overhead

        # Magic state factories (rough estimate)
        factory_overhead = 10 * overhead
        total += factory_overhead

    else:  # qLDPC
        rate = kwargs.get('rate', 0.1)
        rel_dist = kwargs.get('rel_dist', 0.1)

        overhead = 1 / rate
        total = logical_qubits * overhead

        # Smaller factory overhead
        factory_overhead = overhead * 5
        total += factory_overhead

    return total

# Compare across scales
logical_counts = [10, 100, 1000, 10000, 100000, 1000000]
target_error = 1e-10

surface_resources = []
qldpc_resources = []

print("\nTotal Physical Qubits Required:")
print(f"{'Logical':>10} | {'Surface':>15} | {'qLDPC':>15} | {'Winner':>10}")
print("-" * 60)

for k in logical_counts:
    surf_res = total_resources('surface', k, target_error)
    qldpc_res = total_resources('qldpc', k, target_error)

    surface_resources.append(surf_res)
    qldpc_resources.append(qldpc_res)

    winner = 'Surface' if surf_res < qldpc_res else 'qLDPC'
    print(f"{k:>10,} | {surf_res:>15,.0f} | {qldpc_res:>15,.0f} | {winner:>10}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(logical_counts, surface_resources, 'ro-', linewidth=2,
          markersize=10, label='Surface Code')
ax.loglog(logical_counts, qldpc_resources, 'bs-', linewidth=2,
          markersize=10, label='Good qLDPC')

# Find crossover
for i in range(len(logical_counts)-1):
    if surface_resources[i] < qldpc_resources[i] and surface_resources[i+1] > qldpc_resources[i+1]:
        ax.axvline(x=logical_counts[i], color='green', linestyle='--',
                   label=f'Crossover ~{logical_counts[i]}')

ax.set_xlabel('Logical Qubits', fontsize=12)
ax.set_ylabel('Physical Qubits', fontsize=12)
ax.set_title('Resource Scaling: Surface Code vs qLDPC', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_986_crossover.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nCrossover analysis saved to 'day_986_crossover.png'")

# =============================================================================
# Part 4: Platform Suitability Matrix
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Platform Suitability Analysis")
print("=" * 70)

platforms = ['Superconducting', 'Trapped Ion', 'Neutral Atom', 'Photonic']

# Suitability scores (1-10)
suitability = {
    'Surface': {
        'Superconducting': 9,
        'Trapped Ion': 7,
        'Neutral Atom': 7,
        'Photonic': 5
    },
    'qLDPC': {
        'Superconducting': 3,
        'Trapped Ion': 6,
        'Neutral Atom': 9,
        'Photonic': 8
    }
}

# Reasons
reasons = {
    'Superconducting': {
        'Surface': '2D planar architecture matches perfectly',
        'qLDPC': 'Needs 3D packaging or long-range couplers'
    },
    'Trapped Ion': {
        'Surface': 'Works well with ion shuttling',
        'qLDPC': 'Moderate all-to-all helps, but limited scale'
    },
    'Neutral Atom': {
        'Surface': 'Good but underutilizes reconfigurability',
        'qLDPC': 'Reconfigurable connectivity is ideal'
    },
    'Photonic': {
        'Surface': 'Cluster state approaches work',
        'qLDPC': 'Long-range entanglement is natural'
    }
}

print("\nPlatform Suitability Scores:")
print(f"{'Platform':<16} | {'Surface':>8} | {'qLDPC':>8} | {'Preferred':<12}")
print("-" * 55)

for platform in platforms:
    s_score = suitability['Surface'][platform]
    q_score = suitability['qLDPC'][platform]
    preferred = 'Surface' if s_score > q_score else 'qLDPC' if q_score > s_score else 'Tie'
    print(f"{platform:<16} | {s_score:>8} | {q_score:>8} | {preferred:<12}")

# Heatmap visualization
fig, ax = plt.subplots(figsize=(10, 6))

data = np.array([[suitability['Surface'][p] for p in platforms],
                 [suitability['qLDPC'][p] for p in platforms]])

im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)

ax.set_xticks(range(len(platforms)))
ax.set_xticklabels(platforms, fontsize=11)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Surface Code', 'Good qLDPC'], fontsize=11)

# Add text annotations
for i in range(2):
    for j in range(len(platforms)):
        text = ax.text(j, i, data[i, j], ha='center', va='center',
                       fontsize=14, fontweight='bold')

ax.set_title('Platform Suitability (1-10 scale)', fontsize=14)
plt.colorbar(im, ax=ax, label='Suitability Score')

plt.tight_layout()
plt.savefig('day_986_platforms.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlatform suitability saved to 'day_986_platforms.png'")

# =============================================================================
# Part 5: Decision Flowchart (Text)
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Code Selection Decision Framework")
print("=" * 70)

decision_tree = """
CODE SELECTION DECISION TREE:

┌─────────────────────────────────────────────────────────────────────┐
│                    What is your hardware platform?                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│Superconducting│     │  Trapped Ion  │     │Neutral Atom/  │
│   (2D chip)   │     │   or Hybrid   │     │   Photonic    │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        ▼                     ▼                     ▼
   SURFACE CODE         Check error rate       Check error rate
   (strong rec.)              │                     │
                              ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │ p > 0.1%?       │   │ p > 0.1%?       │
                    └────┬───────┬────┘   └────┬───────┬────┘
                    Yes  │       │ No     Yes  │       │ No
                         ▼       ▼             ▼       ▼
                    SURFACE   Consider      SURFACE   qLDPC
                    CODE      qLDPC or      CODE      (strong
                              hybrid                   rec.)

┌─────────────────────────────────────────────────────────────────────┐
│                  Additional Considerations:                          │
│                                                                      │
│  • Scale < 1000 logical qubits → Surface code (simpler)             │
│  • Scale > 10000 logical qubits → Consider qLDPC (efficiency)       │
│  • Near-term demo → Surface code (proven toolchain)                 │
│  • Long-term scalability → Invest in qLDPC development              │
│  • Hybrid approach: bivariate bicycle codes as middle ground        │
└─────────────────────────────────────────────────────────────────────┘
"""
print(decision_tree)

# =============================================================================
# Part 6: Summary Table
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Final Summary")
print("=" * 70)

summary_table = """
COMPREHENSIVE COMPARISON SUMMARY:

┌────────────────────┬─────────────────────┬─────────────────────┐
│ Dimension          │ Surface Code        │ Good qLDPC          │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Rate (k/n)         │ O(1/d²) → 0         │ Θ(1) ~ 0.05-0.1     │
│ Distance           │ O(√n)               │ Θ(n)                │
│ Qubit Overhead     │ O(d²)               │ O(1)                │
│ Threshold          │ ~1% (high)          │ ~0.1% (lower)       │
│ Locality           │ 2D planar           │ Non-local           │
│ Stabilizer Weight  │ 4                   │ ~10-20              │
│ Syndrome Depth     │ O(1) ~ 6            │ O(w) ~ 40-80        │
│ Decoding           │ O(n^1.5) MWPM       │ O(n)-O(n²) BP+OSD   │
│ Gate (CNOT)        │ Lattice surgery     │ Transversal         │
│ Magic States       │ Many levels         │ Fewer levels        │
│ Hardware Maturity  │ Demonstrated        │ Experimental        │
│ Best Platform      │ Superconducting     │ Neutral atom/photon │
├────────────────────┼─────────────────────┼─────────────────────┤
│ OVERALL VERDICT:                                                │
│                                                                  │
│ • TODAY (2026): Surface code for most practical applications    │
│ • NEAR-TERM (2028-2032): Bivariate bicycle codes emerge        │
│ • LONG-TERM (2035+): Good qLDPC for large-scale QC             │
└────────────────────┴─────────────────────┴─────────────────────┘
"""
print(summary_table)

print("\n" + "=" * 70)
print("Day 986 Complete!")
print("=" * 70)
```

---

## Summary

### Key Comparison Points

| Dimension | Surface | qLDPC | Winner (Context) |
|-----------|---------|-------|------------------|
| Rate | $O(1/d^2)$ | $\Theta(1)$ | qLDPC (asymptotic) |
| Threshold | ~1% | ~0.1% | Surface (tolerance) |
| Locality | 2D | Non-local | Surface (hardware) |
| Scalability | Polynomial | Constant | qLDPC (large scale) |
| Maturity | Proven | Emerging | Surface (today) |

### Main Takeaways

1. **Surface codes** excel in current 2D hardware with high threshold
2. **qLDPC codes** provide asymptotically optimal overhead scaling
3. **Threshold vs overhead** is the fundamental trade-off
4. **Platform choice** strongly influences optimal code family
5. **Crossover** occurs at large scale (>10,000 logical qubits)
6. **Hybrid approaches** (bivariate bicycle) bridge the gap

---

## Daily Checklist

- [ ] Complete the full comparison table
- [ ] Analyze locality impact on different platforms
- [ ] Calculate crossover point for specific scenario
- [ ] Apply decision framework to example system
- [ ] Complete Level 1 practice problems
- [ ] Attempt Level 2 problems
- [ ] Run computational lab
- [ ] Justify code choice for given constraints

---

## Preview: Day 987

Tomorrow we conclude the week with **Implementation Challenges & Week Synthesis**:
- Practical barriers to qLDPC implementation
- Near-term experimental progress
- 3D architecture proposals
- Complete week synthesis
- Open problems in the field
- Future research directions

---

*"The choice between surface codes and qLDPC codes is not about which is 'better' - it's about matching code properties to hardware capabilities and application requirements."*
--- Practical quantum error correction perspective

---

**Next:** Day 987 - Implementation Challenges & Week Synthesis
