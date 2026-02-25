# Day 621: Optimal Iteration Count O(sqrt(N))

## Overview
**Day 621** | Week 89, Day 5 | Year 1, Month 23 | Grover's Search Algorithm

Today we rigorously analyze the optimal number of Grover iterations and prove the $O(\sqrt{N})$ query complexity. We also examine what happens with suboptimal iteration counts.

---

## Learning Objectives

1. Derive the optimal iteration count formula
2. Prove the $O(\sqrt{N})$ query complexity
3. Analyze the error from non-optimal iterations
4. Understand the lower bound (BBBV theorem)
5. Compare exact vs approximate formulas
6. Implement strategies for unknown N

---

## Core Content

### Optimal Iteration Formula

From the geometric analysis, the success probability after $k$ iterations is:
$$P_{success}(k) = \sin^2((2k+1)\theta)$$

Maximum probability occurs when $(2k+1)\theta = \pi/2$:

$$k_{opt} = \frac{\pi/2 - \theta}{2\theta} = \frac{\pi}{4\theta} - \frac{1}{2}$$

Since we need an integer number of iterations:

$$\boxed{k_{opt} = \left\lfloor\frac{\pi}{4\theta}\right\rfloor \approx \left\lfloor\frac{\pi}{4}\sqrt{N}\right\rfloor}$$

### Asymptotic Analysis

For large $N$ with $M = 1$ marked item:
$$\theta = \arcsin\frac{1}{\sqrt{N}} \approx \frac{1}{\sqrt{N}}$$

Therefore:
$$k_{opt} \approx \frac{\pi}{4} \cdot \frac{1}{1/\sqrt{N}} = \frac{\pi\sqrt{N}}{4}$$

**Query Complexity:** $O(\sqrt{N})$ oracle calls.

### Success Probability at Optimal k

At the optimal iteration:
$$P_{success}(k_{opt}) = \sin^2\left(\frac{\pi}{2} - \epsilon\right) = \cos^2(\epsilon)$$

where $\epsilon$ is the "rounding error" from taking the floor.

**Error bound:** For large $N$:
$$\epsilon \leq \theta \approx \frac{1}{\sqrt{N}}$$

$$P_{success} \geq \cos^2\left(\frac{1}{\sqrt{N}}\right) \geq 1 - \frac{1}{N}$$

For practical purposes: $P_{success} \to 1$ as $N \to \infty$.

### Exact Formula Analysis

**Theorem:** For any $N \geq 1$ and $M = 1$:
$$P_{success}(k_{opt}) \geq 1 - \frac{1}{N}$$

**Proof:**
Let $k_{opt} = \lfloor\frac{\pi}{4\theta}\rfloor$. Then:
$$(2k_{opt}+1)\theta \in \left[\frac{\pi}{2} - \theta, \frac{\pi}{2}\right]$$

Since $\sin$ is monotonic on $[0, \pi/2]$:
$$P_{success} = \sin^2((2k_{opt}+1)\theta) \geq \sin^2\left(\frac{\pi}{2} - \theta\right) = \cos^2\theta = 1 - \sin^2\theta = 1 - \frac{1}{N}$$
∎

### The BBBV Lower Bound

**Theorem (Bennett, Bernstein, Brassard, Vazirani 1997):**
Any quantum algorithm for unstructured search requires $\Omega(\sqrt{N})$ oracle queries.

**Proof Sketch (Hybrid Argument):**
1. Consider a sequence of $N$ oracles $O_0, O_1, ..., O_{N-1}$ where $O_i$ marks item $i$
2. After $k$ queries, the quantum states for different oracles differ by at most $O(k/\sqrt{N})$ in trace distance
3. To distinguish which oracle was used requires trace distance $\geq$ constant
4. Therefore $k = \Omega(\sqrt{N})$

**Conclusion:** Grover's algorithm is **optimal** for unstructured search!

### Error Analysis for Wrong Iteration Count

If we use $k \neq k_{opt}$ iterations:

**Too few iterations ($k < k_{opt}$):**
$$P_{success} = \sin^2((2k+1)\theta) < 1$$
The state hasn't rotated far enough.

**Too many iterations ($k > k_{opt}$):**
$$P_{success} = \sin^2((2k+1)\theta)$$
The state has "overshot" and is rotating away from $|w\rangle$.

**Periodic behavior:** Success probability oscillates with period $\pi/(2\theta) \approx \frac{\pi\sqrt{N}}{2}$ iterations.

### Strategies When N is Unknown

In practice, we might not know $N$ exactly. Several strategies exist:

**1. Exponential Search:**
- Try $k = 1, 2, 4, 8, 16, ...$ iterations
- Check result, repeat if not found
- Expected queries: $O(\sqrt{N})$

**2. Random Iteration Count:**
- Choose $k$ uniformly from $[1, c\sqrt{N}]$ for estimated $N$
- Success probability $\geq 1/2$ for appropriate $c$

**3. Fixed-Point Amplification (Day 626):**
- Modified algorithm that doesn't overshoot
- Always amplifies toward the target

### Comparison Table

| $N$ | $\theta$ (rad) | $k_{opt}$ | $P_{success}$ | Classical Queries |
|-----|---------------|-----------|---------------|-------------------|
| 4 | 0.5236 | 1 | 1.0000 | 2 |
| 16 | 0.2527 | 3 | 0.9612 | 8 |
| 64 | 0.1253 | 6 | 0.9961 | 32 |
| 256 | 0.0626 | 12 | 0.9995 | 128 |
| 1024 | 0.0313 | 25 | 0.9999 | 512 |
| $N$ | $\sim 1/\sqrt{N}$ | $\sim \frac{\pi}{4}\sqrt{N}$ | $\to 1$ | $N/2$ |

---

## Worked Examples

### Example 1: Exact Calculation for N = 100
Calculate the optimal iterations and success probability for $N = 100$.

**Solution:**
$\theta = \arcsin(1/10) = 0.1002$ rad

$k_{opt} = \lfloor\frac{\pi}{4 \times 0.1002}\rfloor = \lfloor 7.84 \rfloor = 7$

Check: $(2 \times 7 + 1) \times 0.1002 = 1.503$ rad

$P_{success} = \sin^2(1.503) = 0.9975$ (99.75%)

Classical would need ~50 queries on average; Grover needs 7!

### Example 2: Suboptimal Iterations
For $N = 64$, compare success probabilities at $k = 5, 6, 7$.

**Solution:**
$\theta = \arcsin(1/8) = 0.1253$ rad

| $k$ | $(2k+1)\theta$ | $\sin^2$ | $P_{success}$ |
|-----|----------------|----------|---------------|
| 5 | 1.378 | 0.9511 | 95.1% |
| 6 | 1.628 | 0.9961 | 99.6% |
| 7 | 1.879 | 0.9470 | 94.7% |

Optimal is $k = 6$. Both $k = 5$ and $k = 7$ give ~95% success.

### Example 3: Unknown N Strategy
Describe the exponential search strategy when $N$ is unknown.

**Solution:**
1. Set $k = 1$, run Grover, measure
2. If success, done
3. If failure, set $k = 2k$, repeat

Analysis: If actual optimal is $k^*$, we'll try $k = 2^j$ where $2^{j-1} < k^* \leq 2^j$.

Total queries: $1 + 2 + 4 + ... + 2^j = 2^{j+1} - 1 < 4k^* = O(\sqrt{N})$

Expected number of repetitions is constant, so total complexity remains $O(\sqrt{N})$.

---

## Practice Problems

### Problem 1: Iteration Count
For a database with $N = 2^{20} \approx 10^6$ items:
a) Calculate the optimal number of Grover iterations
b) What is the success probability?
c) How many classical queries would be needed on average?

### Problem 2: Error Tolerance
If we can tolerate a 10% error (90% success probability), what range of $k$ values is acceptable for $N = 256$?

### Problem 3: Repeated Search
If Grover's algorithm fails (measures wrong item), we must restart. What is the expected total number of iterations to find the marked item?

---

## Computational Lab

```python
"""Day 621: Optimal Iteration Count Analysis"""
import numpy as np
import matplotlib.pyplot as plt

def theta_exact(N, M=1):
    """Exact rotation angle."""
    return np.arcsin(np.sqrt(M/N))

def theta_approx(N, M=1):
    """Approximate rotation angle for large N."""
    return np.sqrt(M/N)

def optimal_k_exact(N, M=1):
    """Exact optimal iteration count."""
    theta = theta_exact(N, M)
    return int(np.floor(np.pi / (4 * theta)))

def optimal_k_approx(N, M=1):
    """Approximate optimal iteration count."""
    return int(np.round(np.pi / 4 * np.sqrt(N/M)))

def success_probability(k, N, M=1):
    """Success probability after k iterations."""
    theta = theta_exact(N, M)
    return np.sin((2*k + 1) * theta)**2

def analyze_iteration_sensitivity(N):
    """Analyze how success probability varies with iteration count."""
    k_opt = optimal_k_exact(N)
    k_range = range(max(0, k_opt - 5), k_opt + 6)

    probs = [success_probability(k, N) for k in k_range]

    print(f"\nIteration Sensitivity Analysis (N={N}):")
    print(f"Optimal k = {k_opt}")
    print("-" * 40)
    print(f"{'k':>5} | {'|k - k_opt|':>12} | {'P_success':>12}")
    print("-" * 40)
    for k, p in zip(k_range, probs):
        diff = abs(k - k_opt)
        marker = " <-- optimal" if k == k_opt else ""
        print(f"{k:>5} | {diff:>12} | {p:>12.6f}{marker}")

    return k_range, probs

def verify_lower_bound(N_values):
    """Verify that optimal iterations scale as sqrt(N)."""
    k_opts = [optimal_k_exact(N) for N in N_values]
    sqrt_N = [np.sqrt(N) for N in N_values]

    # Linear regression in log-log space
    log_N = np.log(N_values)
    log_k = np.log(k_opts)
    slope, intercept = np.polyfit(log_N, log_k, 1)

    print("\nScaling Verification:")
    print("-" * 50)
    print(f"Fitted scaling: k_opt ~ N^{slope:.4f}")
    print(f"Expected: k_opt ~ N^0.5 = sqrt(N)")
    print(f"Ratio k_opt / sqrt(N):")
    for N, k in zip(N_values, k_opts):
        ratio = k / np.sqrt(N)
        print(f"  N = {N:>6}: k_opt = {k:>4}, ratio = {ratio:.4f}")
    print(f"Theoretical ratio: π/4 ≈ {np.pi/4:.4f}")

    return slope

def exponential_search_simulation(N_true, max_trials=1000):
    """Simulate exponential search strategy for unknown N."""
    total_queries_list = []

    for _ in range(max_trials):
        total_queries = 0
        k = 1
        found = False

        while not found and k < 10 * np.sqrt(N_true):
            # Simulate Grover with k iterations
            p_success = success_probability(k, N_true)
            total_queries += k

            # Random success based on probability
            if np.random.random() < p_success:
                found = True
            else:
                k = min(k * 2, int(np.pi/4 * np.sqrt(N_true) * 2))

        total_queries_list.append(total_queries)

    return np.mean(total_queries_list), np.std(total_queries_list)

def plot_complexity_comparison():
    """Plot classical vs quantum complexity."""
    N_values = np.logspace(1, 6, 50)

    classical = N_values / 2  # Expected classical queries
    quantum = np.pi / 4 * np.sqrt(N_values)  # Grover queries

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.loglog(N_values, classical, 'b-', label='Classical O(N)', linewidth=2)
    plt.loglog(N_values, quantum, 'r-', label='Quantum O(√N)', linewidth=2)
    plt.xlabel('Database Size N', fontsize=12)
    plt.ylabel('Number of Queries', fontsize=12)
    plt.title('Query Complexity: Classical vs Grover', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    speedup = classical / quantum
    plt.semilogx(N_values, speedup, 'g-', linewidth=2)
    plt.xlabel('Database Size N', fontsize=12)
    plt.ylabel('Speedup Factor (Classical/Quantum)', fontsize=12)
    plt.title('Grover Speedup Factor ≈ √N / (π/2)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grover_complexity_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_success_probability_vs_k(N):
    """Plot success probability as function of iterations."""
    k_opt = optimal_k_exact(N)
    k_max = 4 * k_opt

    k_values = np.arange(k_max + 1)
    probs = [success_probability(k, N) for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, probs, 'b-', linewidth=2)
    plt.axvline(x=k_opt, color='r', linestyle='--', linewidth=2,
                label=f'Optimal k={k_opt}')
    plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    # Mark period
    period = np.pi / (2 * theta_exact(N))
    for i in range(1, 4):
        plt.axvline(x=k_opt + i*period, color='orange', linestyle=':',
                    alpha=0.5)

    plt.xlabel('Number of Iterations k', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title(f'Grover Success Probability (N={N})\n'
              f'Period ≈ {period:.1f} iterations', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig('grover_success_vs_k.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
print("=" * 60)
print("Optimal Iteration Count Analysis")
print("=" * 60)

# Analyze for various N
print("\n1. OPTIMAL ITERATIONS FOR VARIOUS N")
print("-" * 60)
print(f"{'N':>8} | {'θ (exact)':>10} | {'θ (approx)':>10} | "
      f"{'k_opt':>6} | {'k_approx':>8} | {'P_success':>10}")
print("-" * 60)

for N in [4, 16, 64, 256, 1024, 4096, 16384]:
    te = theta_exact(N)
    ta = theta_approx(N)
    ke = optimal_k_exact(N)
    ka = optimal_k_approx(N)
    ps = success_probability(ke, N)
    print(f"{N:>8} | {te:>10.6f} | {ta:>10.6f} | {ke:>6} | {ka:>8} | {ps:>10.6f}")

# Verify sqrt(N) scaling
print("\n2. SCALING VERIFICATION")
N_values = [2**k for k in range(4, 18)]
slope = verify_lower_bound(N_values)

# Iteration sensitivity
print("\n3. ITERATION SENSITIVITY")
analyze_iteration_sensitivity(64)

# Exponential search
print("\n4. EXPONENTIAL SEARCH STRATEGY")
print("-" * 50)
for N in [100, 1000, 10000]:
    mean_queries, std_queries = exponential_search_simulation(N, 1000)
    k_opt = optimal_k_exact(N)
    print(f"N = {N:>5}: Mean queries = {mean_queries:.1f} ± {std_queries:.1f}")
    print(f"         Optimal single-shot = {k_opt}")
    print(f"         Ratio = {mean_queries/k_opt:.2f}")

# Visualizations
print("\n5. GENERATING PLOTS...")
plot_complexity_comparison()
plot_success_probability_vs_k(256)

# Final summary
print("\n" + "=" * 60)
print("SUMMARY: Grover's Algorithm Complexity")
print("=" * 60)
print(f"• Query complexity: O(√N)")
print(f"• Optimal iterations: k ≈ (π/4)√N")
print(f"• Success probability: 1 - O(1/N)")
print(f"• Classical comparison: O(N) queries needed")
print(f"• Speedup: √N factor (quadratic)")
print(f"• Lower bound (BBBV): Grover is optimal")
```

**Expected Output:**
```
============================================================
Optimal Iteration Count Analysis
============================================================

1. OPTIMAL ITERATIONS FOR VARIOUS N
------------------------------------------------------------
       N |   θ (exact) |  θ (approx) |  k_opt | k_approx | P_success
------------------------------------------------------------
       4 |   0.523599 |   0.500000 |      1 |        1 |   1.000000
      16 |   0.252680 |   0.250000 |      3 |        3 |   0.961258
      64 |   0.125329 |   0.125000 |      6 |        6 |   0.996094
     256 |   0.062582 |   0.062500 |     12 |       13 |   0.999512
    1024 |   0.031266 |   0.031250 |     25 |       25 |   0.999878
    4096 |   0.015627 |   0.015625 |     50 |       50 |   0.999970
   16384 |   0.007813 |   0.007812 |    100 |      100 |   0.999992

2. SCALING VERIFICATION
--------------------------------------------------
Fitted scaling: k_opt ~ N^0.5002
Expected: k_opt ~ N^0.5 = sqrt(N)
Theoretical ratio: π/4 ≈ 0.7854
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Optimal iterations | $k_{opt} = \lfloor\frac{\pi}{4\theta}\rfloor \approx \frac{\pi}{4}\sqrt{N}$ |
| Query complexity | $O(\sqrt{N})$ |
| Success probability | $P \geq 1 - 1/N$ at optimal $k$ |
| Period of oscillation | $\frac{\pi}{2\theta} \approx \frac{\pi}{2}\sqrt{N}$ |
| BBBV lower bound | $\Omega(\sqrt{N})$ queries required |

### Key Takeaways

1. **Optimal iterations** scale as $\sqrt{N}$
2. **Grover is provably optimal** (matches lower bound)
3. **Success probability** approaches 1 for large N
4. **Suboptimal iterations** reduce but don't eliminate success
5. **Periodic behavior** means overshooting eventually recovers
6. **Unknown N** can be handled with exponential search

---

## Daily Checklist

- [ ] I can derive the optimal iteration formula
- [ ] I understand the O(sqrt(N)) complexity
- [ ] I know the BBBV lower bound result
- [ ] I can analyze error from non-optimal iterations
- [ ] I understand strategies for unknown N
- [ ] I ran the computational lab and verified the scaling

---

*Next: Day 622 — Multiple Solutions Case*
