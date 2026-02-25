# Day 985: Constant-Overhead Fault Tolerance

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Overhead Analysis & Threshold Theorems |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Gate Implementation on qLDPC |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Overhead Calculations |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 985, you will be able to:

1. Define constant overhead and its implications for scalability
2. Compare overhead scaling of surface codes vs good qLDPC codes
3. Explain the threshold theorem for constant-overhead schemes
4. Describe gate implementation challenges on qLDPC codes
5. Analyze magic state distillation in the qLDPC context
6. Evaluate the practical timeline for constant-overhead quantum computing

---

## Core Content

### 1. What is Overhead?

**Definition:** The overhead of an error-correcting scheme is the ratio of physical resources to logical resources.

**Qubit Overhead:**
$$\text{Overhead}_{\text{qubit}} = \frac{n_{\text{physical}}}{k_{\text{logical}}}$$

**Space-Time Overhead:**
For a fault-tolerant computation:
$$\text{Overhead}_{\text{ST}} = \frac{\text{Physical qubits} \times \text{Time steps}}{\text{Logical qubits} \times \text{Logical gates}}$$

---

### 2. Overhead Scaling Comparison

**Surface Code:**

For logical error rate $\epsilon$ at physical error rate $p < p_{\text{th}}$:
$$\epsilon \sim \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

To achieve $\epsilon$:
$$d \sim \log(1/\epsilon) / \log(p_{\text{th}}/p)$$

Qubit overhead:
$$\boxed{\text{Overhead}_{\text{surface}} = O(d^2) = O(\log^2(1/\epsilon))}$$

**Good qLDPC Code:**

With constant rate $R$ and relative distance $\delta$:
$$n = k/R$$
$$d = \delta n = \delta k / R$$

Qubit overhead:
$$\boxed{\text{Overhead}_{\text{qLDPC}} = \frac{n}{k} = \frac{1}{R} = O(1)}$$

**The Revolution:**

| Property | Surface Code | Good qLDPC |
|----------|--------------|------------|
| Overhead | $O(\log^2(1/\epsilon))$ | $O(1)$ |
| For $\epsilon = 10^{-15}$ | $\sim 10^4$ | $\sim 10-100$ |
| Million logical qubits | $\sim 10^{10}$ physical | $\sim 10^7$ physical |

---

### 3. The Constant-Overhead Threshold Theorem

**Classical Threshold Theorem (Aharonov-Ben-Or 1997):**

If the physical error rate $p < p_{\text{th}}$, arbitrarily reliable computation is possible with overhead:
$$\text{Overhead} = O(\text{poly}\log(1/\epsilon))$$

**Gottesman's Constant-Overhead Theorem (2014):**

Using good qLDPC codes, if $p < p_{\text{th}}$:

$$\boxed{\text{Overhead} = O(1)}$$

independent of target error rate $\epsilon$!

**Key Requirements:**

1. **Good qLDPC code family:** $k = \Theta(n)$, $d = \Theta(n)$
2. **Efficient decoding:** Polynomial (preferably linear) time
3. **Fault-tolerant gadgets:** For syndrome extraction and gates
4. **Non-local connectivity:** Physical requirement

---

### 4. Why Constant Overhead is Possible

**Intuition:**

For a good code with $d = \delta n$:
- Can correct $\lfloor(\delta n - 1)/2\rfloor$ errors
- Error rate per round: $p \cdot n \cdot w$ where $w$ is stabilizer weight
- Errors accumulate over $T$ rounds

**Error Budget Analysis:**

After $T$ syndrome rounds:
$$\text{Expected errors} \approx p \cdot n \cdot w \cdot T$$

Need: Expected errors $< d/2$
$$p \cdot n \cdot w \cdot T < \delta n / 2$$
$$T < \frac{\delta}{2pw}$$

For constant $\delta$, $w$, and $p < p_{\text{th}}$:
- Can run $\Theta(1)$ rounds before correction
- Correction succeeds with high probability
- No overhead growth needed!

**The Threshold Condition:**
$$p_{\text{th}} = \frac{\delta}{2w \cdot c}$$

where $c$ is a constant depending on the decoder.

---

### 5. Gate Implementation on qLDPC Codes

Implementing logical gates on qLDPC codes is more challenging than on surface codes.

**Transversal Gates:**

A gate is transversal if it acts independently on each qubit:
$$\bar{U} = U^{\otimes n}$$

For CSS codes, transversal gates include:
- **CNOT:** Transversal between two code blocks
- **Pauli gates:** Always transversal
- **Hadamard:** Transversal for self-dual CSS codes

**The Eastin-Knill Theorem:**

No quantum code has a universal set of transversal gates.

$$\text{Transversal} \cap \text{Universal} = \emptyset$$

**Implication:** Need non-transversal methods for universal computation.

---

### 6. Magic State Distillation on qLDPC

**Magic States:**

$$\ket{T} = \frac{1}{\sqrt{2}}(\ket{0} + e^{i\pi/4}\ket{1})$$

Allow non-Clifford gates via gate teleportation.

**Distillation Protocol:**

1. Prepare noisy $\ket{T}$ states
2. Encode in distillation code
3. Measure and post-select
4. Output purified $\ket{T}$

**Overhead Analysis:**

For 15-to-1 distillation with input error $p$:
- Output error: $O(p^3)$
- Overhead per level: $\sim 15$

After $L$ levels:
- Output error: $O(p^{3^L})$
- Total overhead: $15^L$

**qLDPC Advantage:**

With good qLDPC codes:
- Base logical error rate is lower (per unit overhead)
- Fewer distillation levels needed
- Magic state factories more efficient

**Comparison:**

| Scheme | Base Overhead | Distillation Levels | Total |
|--------|---------------|---------------------|-------|
| Surface $d=20$ | 400 | 3 | $\sim 10^6$ |
| qLDPC $R=0.1$ | 10 | 2 | $\sim 10^3$ |

---

### 7. Syndrome Extraction Overhead

A hidden cost in qLDPC: syndrome extraction circuits.

**Surface Code:**
- Weight-4 stabilizers
- 4 CNOT gates per stabilizer
- Can parallelize all stabilizers
- Depth: $O(1)$ per round

**qLDPC Code:**
- Weight $w = O(1)$ but typically $w \sim 10-20$
- $w$ CNOT gates per stabilizer
- Non-local gates required
- Depth: $O(w)$ per round

**Effective Overhead:**

True overhead includes syndrome depth:
$$\text{Overhead}_{\text{eff}} = \text{Overhead}_{\text{qubit}} \times \frac{\text{Syndrome depth (qLDPC)}}{\text{Syndrome depth (surface)}}$$

If syndrome depth is $O(w)$:
$$\text{Overhead}_{\text{eff}} = O(1) \times O(w) = O(w) = O(1)$$

Still constant! The depth overhead is hidden in the constant.

---

### 8. Practical Considerations

**Hardware Requirements for qLDPC:**

1. **All-to-all connectivity:** Or reconfigurable connections
2. **Fast classical processing:** For real-time decoding
3. **Low-latency feedback:** Syndrome $\to$ correction $\to$ next round
4. **High gate fidelity:** To achieve threshold

**Current Status (2026):**

| Platform | Connectivity | Status |
|----------|--------------|--------|
| Superconducting | 2D local | Not ready for full qLDPC |
| Trapped ions | Moderate | Better, but scaling limited |
| Neutral atoms | Reconfigurable | Most promising near-term |
| Photonics | Long-range | Natural fit, but other challenges |

**Timeline Estimate:**

- 2025-2027: Bivariate bicycle codes demonstrated
- 2028-2032: Partial qLDPC with moderate non-locality
- 2033+: Full good qLDPC with constant overhead

---

## Practical Applications

### Impact on Quantum Advantage

**Cryptography (Shor's Algorithm):**

To factor 2048-bit RSA:
- Logical qubits: ~4000
- T-gates: ~$10^{11}$

**Surface Code ($d=20$):**
- Physical qubits: ~400 × 4000 × 15 (factory) = $2.4 \times 10^7$
- Time: ~months

**Good qLDPC:**
- Physical qubits: ~10 × 4000 × 3 (factory) = $1.2 \times 10^5$
- Time: ~hours to days

**Factor: 200x reduction in physical qubits!**

---

## Worked Examples

### Example 1: Overhead Calculation

**Problem:** Compare qubit overhead for surface code and qLDPC to achieve logical error rate $10^{-12}$.

**Solution:**

**Surface Code:**

Assume $p = 10^{-3}$, $p_{\text{th}} = 10^{-2}$.

$$\epsilon \sim (p/p_{\text{th}})^{(d+1)/2} = (0.1)^{(d+1)/2}$$

For $\epsilon = 10^{-12}$:
$$10^{-12} = 10^{-(d+1)/2}$$
$$d + 1 = 24$$
$$d = 23$$

Overhead: $n = d^2 = 529$ physical qubits per logical.

**Good qLDPC ($R = 0.05$, $\delta = 0.05$):**

Rate constraint: $n = k/R = 20k$

For $k$ logical qubits, need $n = 20k$ physical qubits.

Distance: $d = 0.05n = k$

For large enough $k$, the code can achieve $\epsilon = 10^{-12}$ with:

Overhead: $n/k = 20$ physical qubits per logical.

**Comparison:** 529 vs 20 = **26x improvement**

---

### Example 2: Threshold Estimation

**Problem:** A qLDPC code has stabilizer weight $w = 12$ and relative distance $\delta = 0.08$. Estimate the threshold.

**Solution:**

Using simplified model:
$$p_{\text{th}} \approx \frac{\delta}{2w \cdot c}$$

Assume $c \approx 5$ (typical decoder constant).

$$p_{\text{th}} \approx \frac{0.08}{2 \times 12 \times 5} = \frac{0.08}{120} \approx 6.7 \times 10^{-4}$$

This is lower than typical surface code threshold ($\sim 1\%$).

**Trade-off:** Lower threshold but constant overhead.

For $p = 10^{-4}$: Below threshold, constant overhead applies!

---

### Example 3: Magic State Factory Sizing

**Problem:** Design a magic state factory for a good qLDPC code to support $10^6$ T-gates per second.

**Solution:**

**Distillation Protocol:** 15-to-1

**Input magic states per output:** 15

**Cycles per distillation:** ~10 syndrome rounds

**Required output rate:** $10^6$ T-gates/sec

**Assuming 1 MHz cycle rate:**
- Distillations per second: $10^5$
- Raw magic states needed: $15 \times 10^6 = 1.5 \times 10^7$/sec

**With qLDPC ($R = 0.1$):**
- Logical qubits per magic state: ~50 (for distillation)
- Physical qubits per logical: 10
- Factory size: ~500 physical qubits per concurrent distillation

**For $10^5$ concurrent distillations:**
- Naive: $5 \times 10^7$ qubits (too many!)
- Pipelined: ~$5 \times 10^4$ qubits with 1000 concurrent distillations

**Comparison with surface code:**
- Surface would need ~10x more for same output rate

---

## Practice Problems

### Level 1: Direct Application

1. **Overhead Ratio:** A surface code has $n = 289$ physical qubits and $k = 1$ logical. A qLDPC has $n = 500$ and $k = 50$. What are the overheads?

2. **Distance Requirement:** For logical error rate $10^{-10}$ with surface code at $p/p_{\text{th}} = 0.1$, what distance is needed?

3. **Rate Calculation:** A good qLDPC has 10,000 physical qubits and 500 logical qubits. What is the rate and overhead?

### Level 2: Intermediate

4. **Threshold Analysis:** Derive the threshold condition for a qLDPC code in terms of physical error rate, stabilizer weight, and relative distance.

5. **Time Overhead:** Surface code syndrome extraction takes 1 microsecond. qLDPC takes 5 microseconds. If qubit overhead is 20x better for qLDPC, what is the break-even for total computation time?

6. **Factory Design:** A 15-to-1 magic state distillation on qLDPC requires 225 physical qubits. How many factories are needed for $10^4$ T-gates per second at 100 KHz cycle rate?

### Level 3: Challenging

7. **Concatenation vs qLDPC:** Compare the overhead of concatenated codes (Steane [[7,1,3]]) with good qLDPC for achieving $10^{-15}$ error rate.

8. **Non-Clifford Gates:** Research and describe how non-Clifford gates can be implemented on qLDPC codes without magic states (if possible).

9. **Single-Shot Error Correction:** Explain the concept and its relevance to qLDPC codes. How does it affect overhead?

---

## Computational Lab

### Objective
Compare overhead scaling and analyze fault-tolerant resource requirements.

```python
"""
Day 985 Computational Lab: Constant-Overhead Fault Tolerance
QLDPC Codes & Constant-Overhead QEC - Week 141
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# =============================================================================
# Part 1: Overhead Scaling Comparison
# =============================================================================

print("=" * 70)
print("Part 1: Overhead Scaling Analysis")
print("=" * 70)

def surface_code_overhead(target_error, p_phys=1e-3, p_th=1e-2):
    """
    Calculate surface code overhead for target logical error rate.
    """
    # d such that (p/p_th)^((d+1)/2) ≤ target_error
    ratio = p_phys / p_th
    d = 1
    while ratio ** ((d + 1) / 2) > target_error:
        d += 2  # Distance must be odd

    n = d ** 2  # Physical qubits (rotated surface code)
    k = 1  # Logical qubits

    return {
        'n': n,
        'k': k,
        'd': d,
        'overhead': n / k,
        'achieved_error': ratio ** ((d + 1) / 2)
    }

def qldpc_overhead(target_error, rate=0.1, rel_dist=0.1, p_phys=1e-3, p_th=5e-4):
    """
    Calculate qLDPC overhead for target logical error rate.

    Simplified model: error ~ exp(-d * c) for some constant c.
    """
    if p_phys >= p_th:
        return {'n': float('inf'), 'k': 1, 'overhead': float('inf')}

    # Estimate required distance
    c = np.log(p_th / p_phys) / 2  # Suppression constant
    d_required = int(-np.log(target_error) / c) + 1

    # From rel_dist = d/n and rate = k/n
    # n = d / rel_dist
    # k = rate * n
    n = int(d_required / rel_dist)
    k = int(rate * n)

    return {
        'n': n,
        'k': k,
        'd': d_required,
        'overhead': n / k if k > 0 else float('inf'),
        'achieved_error': np.exp(-d_required * c)
    }

# Compare over range of target error rates
target_errors = np.logspace(-6, -15, 20)

surface_overheads = []
qldpc_overheads = []

print("\nTarget Error | Surface d | Surface Overhead | qLDPC d | qLDPC Overhead | Improvement")
print("-" * 90)

for eps in [1e-6, 1e-9, 1e-12, 1e-15]:
    surf = surface_code_overhead(eps)
    qldpc = qldpc_overhead(eps)

    improvement = surf['overhead'] / qldpc['overhead'] if qldpc['overhead'] > 0 else float('inf')

    print(f"   10^{int(np.log10(eps)):3d}    |    {surf['d']:3d}    |      {surf['overhead']:6.0f}       |   {qldpc['d']:3d}   |      {qldpc['overhead']:6.1f}       |    {improvement:5.1f}x")

# Full data for plotting
for eps in target_errors:
    surface_overheads.append(surface_code_overhead(eps)['overhead'])
    qldpc_overheads.append(qldpc_overhead(eps)['overhead'])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.loglog(target_errors, surface_overheads, 'ro-', linewidth=2,
           markersize=6, label='Surface Code')
ax1.loglog(target_errors, qldpc_overheads, 'bs-', linewidth=2,
           markersize=6, label='Good qLDPC')
ax1.set_xlabel('Target Logical Error Rate')
ax1.set_ylabel('Qubit Overhead (n/k)')
ax1.set_title('Overhead Scaling Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

ax2 = axes[1]
improvements = [s/q for s, q in zip(surface_overheads, qldpc_overheads)]
ax2.semilogx(target_errors, improvements, 'g^-', linewidth=2, markersize=8)
ax2.set_xlabel('Target Logical Error Rate')
ax2.set_ylabel('Improvement Factor (Surface / qLDPC)')
ax2.set_title('qLDPC Advantage')
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig('day_985_overhead_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nOverhead comparison saved to 'day_985_overhead_comparison.png'")

# =============================================================================
# Part 2: Threshold Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Threshold Analysis")
print("=" * 70)

def estimate_threshold(rel_dist, stab_weight, decoder_constant=5):
    """Estimate threshold for qLDPC code."""
    return rel_dist / (2 * stab_weight * decoder_constant)

# Parameter sweep
rel_dists = np.linspace(0.01, 0.15, 10)
stab_weights = [6, 8, 10, 12, 15, 20]

print("\nEstimated Thresholds (decoder constant c=5):")
print(f"{'δ':>8}", end='')
for w in stab_weights:
    print(f" | w={w:2d}", end='')
print()
print("-" * 60)

for delta in [0.02, 0.05, 0.08, 0.1, 0.12]:
    print(f"{delta:>8.2f}", end='')
    for w in stab_weights:
        thresh = estimate_threshold(delta, w)
        print(f" | {thresh:.4f}", end='')
    print()

# Plot threshold landscape
fig, ax = plt.subplots(figsize=(10, 6))

for w in stab_weights:
    thresholds = [estimate_threshold(d, w) for d in rel_dists]
    ax.plot(rel_dists, thresholds, '-o', label=f'w={w}', linewidth=2)

ax.axhline(y=1e-3, color='red', linestyle='--', linewidth=2, label='Typical p_phys')
ax.set_xlabel('Relative Distance δ')
ax.set_ylabel('Threshold p_th')
ax.set_title('qLDPC Threshold vs Parameters')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 0.01])

plt.tight_layout()
plt.savefig('day_985_threshold.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nThreshold analysis saved to 'day_985_threshold.png'")

# =============================================================================
# Part 3: Magic State Distillation Overhead
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Magic State Distillation Analysis")
print("=" * 70)

def magic_state_overhead(base_overhead, input_error, target_error,
                         protocol='15-to-1'):
    """
    Calculate magic state distillation overhead.

    15-to-1 protocol: output_error ≈ 35 * input_error^3
    """
    if protocol == '15-to-1':
        error_reduction = lambda e: 35 * e**3
        overhead_per_level = 15
    elif protocol == '20-to-4':
        error_reduction = lambda e: e**2
        overhead_per_level = 5
    else:
        raise ValueError("Unknown protocol")

    current_error = input_error
    levels = 0
    total_overhead = 1

    while current_error > target_error and levels < 10:
        current_error = error_reduction(current_error)
        total_overhead *= overhead_per_level
        levels += 1

    return {
        'levels': levels,
        'output_error': current_error,
        'distillation_overhead': total_overhead,
        'total_overhead': base_overhead * total_overhead
    }

# Compare surface vs qLDPC for magic states
print("\nMagic State Distillation Comparison:")
print(f"{'Base':>12} | {'Levels':>6} | {'Distill OH':>10} | {'Total OH':>10}")
print("-" * 50)

for code_name, base_oh, input_err in [
    ('Surface d=15', 225, 1e-3),
    ('Surface d=21', 441, 1e-4),
    ('qLDPC R=0.1', 10, 1e-3),
    ('qLDPC R=0.05', 20, 5e-4)
]:
    result = magic_state_overhead(base_oh, input_err, 1e-10)
    print(f"{code_name:>12} | {result['levels']:>6} | {result['distillation_overhead']:>10} | {result['total_overhead']:>10.0f}")

# =============================================================================
# Part 4: Resource Estimation for Shor's Algorithm
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Shor's Algorithm Resource Estimation")
print("=" * 70)

def shor_resources(bits, code_type='surface', d=17, rate=0.1):
    """
    Estimate resources for factoring n-bit number.

    Based on: https://arxiv.org/abs/2103.13855
    """
    # Logical resources (simplified model)
    logical_qubits = 2 * bits + 3  # For addition circuits
    t_gates = 60 * bits ** 3  # Approximate T-count

    if code_type == 'surface':
        # Surface code overhead
        physical_per_logical = d ** 2
        magic_overhead = magic_state_overhead(physical_per_logical, 1e-3, 1e-10)['total_overhead']
        factory_qubits = magic_overhead * 10  # 10 factories

        total_physical = logical_qubits * physical_per_logical + factory_qubits

        # Time estimate (1 MHz cycle)
        cycle_time = 1e-6  # 1 microsecond
        t_gate_time = 100  # cycles per T-gate
        total_time = t_gates * t_gate_time * cycle_time

    else:  # qLDPC
        physical_per_logical = int(1 / rate)
        magic_overhead = magic_state_overhead(physical_per_logical, 1e-4, 1e-10)['total_overhead']
        factory_qubits = magic_overhead * 3  # 3 factories (more efficient)

        total_physical = logical_qubits * physical_per_logical + factory_qubits

        # Time (potentially faster with parallel operations)
        cycle_time = 5e-6  # 5 microseconds (deeper syndrome)
        t_gate_time = 50  # cycles per T-gate (better parallelism)
        total_time = t_gates * t_gate_time * cycle_time

    return {
        'bits': bits,
        'logical_qubits': logical_qubits,
        't_gates': t_gates,
        'physical_qubits': total_physical,
        'time_seconds': total_time,
        'time_hours': total_time / 3600
    }

print("\nFactoring Resource Estimates:")
print(f"{'Bits':>6} | {'Code':>12} | {'Logical':>8} | {'Physical':>12} | {'Time (h)':>10}")
print("-" * 65)

for bits in [256, 512, 1024, 2048]:
    for code in ['surface', 'qldpc']:
        if code == 'surface':
            res = shor_resources(bits, 'surface', d=21)
        else:
            res = shor_resources(bits, 'qldpc', rate=0.1)

        print(f"{bits:>6} | {code:>12} | {res['logical_qubits']:>8} | {res['physical_qubits']:>12,.0f} | {res['time_hours']:>10.1f}")

# =============================================================================
# Part 5: Space-Time Trade-off Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Space-Time Trade-off")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Physical qubits vs target error
ax1 = axes[0]
target_errs = np.logspace(-8, -15, 8)

for label, code_func in [
    ('Surface d-adaptive', lambda e: surface_code_overhead(e)['n'] * 1000),
    ('qLDPC R=0.1', lambda e: qldpc_overhead(e, rate=0.1)['n'] * 100),
    ('qLDPC R=0.05', lambda e: qldpc_overhead(e, rate=0.05)['n'] * 200)
]:
    qubits = [code_func(e) for e in target_errs]
    ax1.loglog(target_errs, qubits, '-o', linewidth=2, label=label)

ax1.set_xlabel('Target Logical Error Rate')
ax1.set_ylabel('Physical Qubits (for 1000 logical)')
ax1.set_title('Physical Qubit Requirements')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# Time vs physical qubits (Pareto frontier)
ax2 = axes[1]

bit_sizes = [256, 512, 1024, 2048]
surf_results = [shor_resources(b, 'surface', d=21) for b in bit_sizes]
qldpc_results = [shor_resources(b, 'qldpc', rate=0.1) for b in bit_sizes]

ax2.loglog([r['physical_qubits'] for r in surf_results],
           [r['time_hours'] for r in surf_results],
           'ro-', markersize=10, linewidth=2, label='Surface Code')
ax2.loglog([r['physical_qubits'] for r in qldpc_results],
           [r['time_hours'] for r in qldpc_results],
           'bs-', markersize=10, linewidth=2, label='Good qLDPC')

for i, bits in enumerate(bit_sizes):
    ax2.annotate(f'{bits}b', (surf_results[i]['physical_qubits'],
                              surf_results[i]['time_hours']),
                 textcoords="offset points", xytext=(10, 5), fontsize=8)
    ax2.annotate(f'{bits}b', (qldpc_results[i]['physical_qubits'],
                              qldpc_results[i]['time_hours']),
                 textcoords="offset points", xytext=(10, -10), fontsize=8)

ax2.set_xlabel('Physical Qubits')
ax2.set_ylabel('Time (hours)')
ax2.set_title("Shor's Algorithm: Space-Time Trade-off")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_985_spacetime.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSpace-time trade-off saved to 'day_985_spacetime.png'")

# =============================================================================
# Part 6: Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary: Constant-Overhead Fault Tolerance")
print("=" * 70)

summary = """
KEY RESULTS:

1. OVERHEAD SCALING:
   - Surface code: O(log²(1/ε)) per logical qubit
   - Good qLDPC: O(1) per logical qubit
   - Improvement: 10-100x for practical error rates

2. THRESHOLD:
   - Surface code: ~1% (high, but overhead grows)
   - Good qLDPC: ~0.1% (lower, but constant overhead)
   - Trade-off: threshold vs asymptotic scaling

3. MAGIC STATE DISTILLATION:
   - qLDPC needs fewer levels (better base error)
   - Total overhead reduction: 10-100x

4. PRACTICAL TIMELINE:
   - 2026: Bivariate bicycle demonstrations
   - 2030: Partial qLDPC with moderate connectivity
   - 2035+: Full constant-overhead FT possible

5. REQUIREMENTS:
   - Non-local qubit connectivity
   - Fast classical decoding
   - Low physical error rates

BOTTOM LINE:
Constant-overhead fault tolerance with qLDPC codes could reduce
physical qubit requirements by 100-1000x compared to surface codes,
fundamentally changing the scalability of quantum computing.
"""
print(summary)

print("\n" + "=" * 70)
print("Day 985 Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Surface overhead | $O(d^2) = O(\log^2(1/\epsilon))$ |
| qLDPC overhead | $O(1)$ (constant!) |
| Threshold condition | $p_{\text{th}} \approx \delta / (2w \cdot c)$ |
| Magic state output error | $\epsilon_{\text{out}} \approx 35\epsilon_{\text{in}}^3$ (15-to-1) |

### Main Takeaways

1. **Constant overhead** means physical qubits scale linearly with logical qubits
2. **Surface codes** have polylogarithmic overhead, qLDPC has constant
3. **Threshold** for qLDPC is lower but overhead doesn't grow
4. **Magic state distillation** is more efficient with qLDPC base codes
5. **Hardware requirements** (non-locality) are the main barrier
6. **Practical impact**: 100-1000x reduction in physical qubits for large computations

---

## Daily Checklist

- [ ] Calculate overhead for surface codes at different distances
- [ ] Compare overhead scaling between code families
- [ ] Estimate threshold from code parameters
- [ ] Analyze magic state distillation requirements
- [ ] Complete Level 1 practice problems
- [ ] Attempt Level 2 problems
- [ ] Run computational lab
- [ ] Understand practical timeline for constant-overhead QC

---

## Preview: Day 986

Tomorrow we provide a **detailed comparison of qLDPC vs Surface Codes**:
- Locality and hardware constraints
- Decoding complexity analysis
- Syndrome measurement circuits
- Near-term vs long-term trade-offs
- Decision framework for code selection

---

*"Constant-overhead fault tolerance transforms quantum error correction from a resource bottleneck into a manageable engineering challenge."*
--- Gottesman, on the asymptotic result

---

**Next:** Day 986 - QLDPC vs Surface Code Comparison
