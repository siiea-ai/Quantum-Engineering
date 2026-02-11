# Day 627: Oblivious Amplitude Amplification

## Overview
**Day 627** | Week 90, Day 4 | Year 1, Month 23 | Amplitude Amplification

Today we study oblivious amplitude amplification, which works even when we don't know the success probability and can't easily verify success.

---

## Learning Objectives

1. Understand the "oblivious" setting and its challenges
2. Define oblivious amplitude amplification
3. Learn exponential search strategies
4. Analyze success guarantees without knowing amplitude
5. Apply to problems with unknown solution count
6. Compare oblivious vs standard approaches

---

## Core Content

### The Oblivious Setting

**Standard AA assumption:** We know (or can measure) whether a state is "good"

**Oblivious setting:** We cannot efficiently verify success!

**Examples:**
- Searching for a satisfying assignment when verification is expensive
- Finding a state with a property we can only check probabilistically
- Subroutines in larger algorithms where checking ruins superposition

### The Challenge

Without knowing $a = \sin^2\theta$:
1. We can't compute optimal $k = \lfloor\pi/(4\theta)\rfloor$
2. Choosing wrong $k$ might overshoot
3. Can't verify if measurement succeeded

**Question:** Can we still amplify effectively?

### Exponential Search Strategy

**Algorithm (BHT 2000):**
1. Set $m = 1$
2. Repeat:
   a. Choose $k$ uniformly from $\{0, 1, ..., m-1\}$
   b. Apply $k$ iterations of $Q$
   c. Measure
   d. Set $m = \min(2m, \sqrt{N})$
3. Until we've made $O(\sqrt{N/a})$ total queries

**Key insight:** At least one value of $k$ in the range will be "close enough" to optimal.

### Theoretical Guarantee

**Theorem:** Exponential search finds a solution with constant probability using:
$$O\left(\frac{1}{\sqrt{a}}\right) \text{ queries}$$

This matches optimal complexity without knowing $a$!

### Why Exponential Search Works

Consider the probability of success for random $k \in [0, m-1]$:

$$\bar{P}_m = \frac{1}{m}\sum_{k=0}^{m-1}\sin^2((2k+1)\theta)$$

**Claim:** For $m \geq \frac{\pi}{4\theta}$, we have $\bar{P}_m \geq \frac{1}{4}$.

**Proof sketch:**
- The function $\sin^2((2k+1)\theta)$ oscillates
- Over a full period, average is $1/2$
- Even partial periods give average $\geq 1/4$

### Variable-Time Amplitude Amplification

**Berry, Childs, Cleve et al.** developed algorithms where the preparation time varies:

$$A|0\rangle|0\rangle = \sum_t \alpha_t |t\rangle|\psi_t\rangle$$

where $|t\rangle$ encodes "time" and $|\psi_t\rangle$ is the state prepared after time $t$.

**Oblivious AA** can handle such variable-time preparations.

### Robust Oblivious Amplification

**Algorithm:**
1. Start with $m = 1$, total queries $T = 0$
2. While $T < T_{max}$:
   - Choose $k$ uniformly from $\{0, ..., m-1\}$
   - Run $k$ iterations of $Q$
   - Measure; if "good" pattern seen, return
   - Update: $m \leftarrow \min(\lambda m, M_{max})$, $T \leftarrow T + k$
3. Return failure

where $\lambda \approx 6/5$ and $M_{max} = O(\sqrt{N})$.

### Comparison of Approaches

| Setting | Knowledge | Strategy | Complexity |
|---------|-----------|----------|------------|
| Known $a$ | Full | Optimal $k$ | $O(1/\sqrt{a})$ |
| Unknown $a$, verifiable | Can check | Repeated runs | $O(1/\sqrt{a})$ |
| Oblivious | Nothing | Exponential search | $O(1/\sqrt{a})$ |

All achieve optimal complexity, but oblivious has larger constants.

---

## Worked Examples

### Example 1: Exponential Search Simulation
Simulate exponential search for unknown $a = 0.04$.

**Solution:**
True optimal: $k_{opt} = \lfloor\pi/(4 \times 0.2)\rfloor = 3$

Exponential search rounds:
- Round 1: $m = 1$, try $k \in \{0\}$. Expected P = 0.04 (low)
- Round 2: $m = 2$, try $k \in \{0, 1\}$. Expected P ≈ 0.25
- Round 3: $m = 4$, try $k \in \{0, 1, 2, 3\}$. Expected P ≈ 0.5

By round 3, we have good success probability.

Total queries: $0 + 1 + (0+1+2+3)/4 \times 4 \approx 6$ on average

### Example 2: Average Success Probability
Calculate $\bar{P}_m$ for $\theta = \pi/6$ and $m = 3$.

**Solution:**
$\bar{P}_3 = \frac{1}{3}[\sin^2(\theta) + \sin^2(3\theta) + \sin^2(5\theta)]$

$= \frac{1}{3}[\sin^2(30°) + \sin^2(90°) + \sin^2(150°)]$

$= \frac{1}{3}[0.25 + 1 + 0.25] = 0.5$

Good average success probability!

### Example 3: Query Counting
For exponential search with $a = 0.01$, count expected queries.

**Solution:**
$\theta = 0.1$, optimal $k \approx 7.8$

Need to reach $m \geq 8$.

Rounds: $m = 1, 2, 4, 8$

Average queries per round $r$ with $m_r$: $\frac{1}{m_r}\sum_{k=0}^{m_r-1} k = \frac{m_r - 1}{2}$

Total ≈ $0 + 0.5 + 1.5 + 3.5 = 5.5$ per iteration of outer loop.

Expected iterations until success: $O(1)$ once $m$ is large enough.

Total: $O(1/\sqrt{a})$ queries ✓

---

## Practice Problems

### Problem 1: Average Probability Bound
Prove that for any $\theta > 0$ and $m \geq \frac{\pi}{4\theta}$:
$$\bar{P}_m = \frac{1}{m}\sum_{k=0}^{m-1}\sin^2((2k+1)\theta) \geq \frac{1}{4}$$

### Problem 2: Exponential Search Design
Design an exponential search protocol for a problem with $N = 1024$ items and unknown $M$ solutions. Specify the sequence of $m$ values.

### Problem 3: Constant Factor
What is the expected number of Grover iterations in exponential search as a multiple of the optimal count?

---

## Computational Lab

```python
"""Day 627: Oblivious Amplitude Amplification"""
import numpy as np
import matplotlib.pyplot as plt

def success_probability(k, theta):
    """Success probability after k iterations."""
    return np.sin((2*k + 1) * theta)**2

def average_probability(m, theta):
    """Average success probability for random k in [0, m-1]."""
    return np.mean([success_probability(k, theta) for k in range(m)])

def exponential_search_simulation(N, M, max_rounds=20, num_trials=1000):
    """
    Simulate exponential search for oblivious AA.

    Returns:
        success_rate, avg_queries, query_distribution
    """
    a = M / N
    theta = np.arcsin(np.sqrt(a))

    successes = 0
    total_queries = []

    for _ in range(num_trials):
        m = 1
        queries = 0
        found = False

        for round_num in range(max_rounds):
            # Choose random k in [0, m-1]
            k = np.random.randint(0, m)
            queries += k

            # Probability of success with this k
            p_success = success_probability(k, theta)

            # Simulate measurement
            if np.random.random() < p_success:
                found = True
                successes += 1
                break

            # Double m (capped at sqrt(N))
            m = min(2 * m, int(np.sqrt(N)))

        total_queries.append(queries)

    return successes / num_trials, np.mean(total_queries), total_queries

def optimal_vs_oblivious_comparison(a_values):
    """Compare optimal (known a) vs oblivious search."""
    results = []

    for a in a_values:
        theta = np.arcsin(np.sqrt(a))
        k_optimal = int(np.round(np.pi / (4 * theta) - 0.5))

        # Optimal: k_opt queries, P ≈ 1
        optimal_queries = k_optimal

        # Oblivious: simulate
        N = 10000
        M = int(a * N)
        success_rate, avg_queries, _ = exponential_search_simulation(N, M, num_trials=500)

        results.append({
            'a': a,
            'optimal_k': k_optimal,
            'oblivious_queries': avg_queries,
            'oblivious_success': success_rate,
            'ratio': avg_queries / max(1, k_optimal)
        })

    return results

def visualize_average_probability():
    """Visualize average probability as function of m."""
    theta_values = [0.1, 0.2, 0.3, 0.5]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for theta in theta_values:
        k_opt = np.pi / (4 * theta)
        m_values = range(1, 20)
        avg_probs = [average_probability(m, theta) for m in m_values]

        plt.plot(m_values, avg_probs, 'o-', label=f'θ={theta:.2f} (k_opt≈{k_opt:.1f})')

    plt.axhline(y=0.25, color='red', linestyle='--', label='P=0.25 threshold')
    plt.xlabel('Range size m', fontsize=12)
    plt.ylabel('Average Success Probability', fontsize=12)
    plt.title('Average P for Random k ∈ [0, m-1]', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # For fixed theta, show convergence
    theta = 0.2
    m_values = range(1, 30)
    avg_probs = [average_probability(m, theta) for m in m_values]

    plt.plot(m_values, avg_probs, 'b-', linewidth=2)
    plt.axhline(y=0.25, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)

    # Mark k_opt
    k_opt = int(np.pi / (4 * theta))
    plt.axvline(x=k_opt, color='orange', linestyle=':', label=f'k_opt={k_opt}')

    plt.xlabel('Range size m', fontsize=12)
    plt.ylabel('Average Success Probability', fontsize=12)
    plt.title(f'Convergence of Average P (θ={theta})', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('oblivious_avg_probability.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_exponential_search():
    """Visualize exponential search behavior."""
    N = 1000
    M = 10  # a = 0.01
    theta = np.arcsin(np.sqrt(M/N))
    k_opt = int(np.pi / (4 * theta))

    # Track a single run in detail
    np.random.seed(42)

    m_history = [1]
    k_history = []
    p_history = []
    cumulative_queries = [0]

    m = 1
    queries = 0

    for round_num in range(15):
        k = np.random.randint(0, m)
        k_history.append(k)
        queries += k
        cumulative_queries.append(queries)

        p = success_probability(k, theta)
        p_history.append(p)

        m = min(2 * m, int(np.sqrt(N)))
        m_history.append(m)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # m evolution
    ax1 = axes[0, 0]
    ax1.step(range(len(m_history)), m_history, 'b-', where='post', linewidth=2)
    ax1.axhline(y=k_opt, color='red', linestyle='--', label=f'k_opt={k_opt}')
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Range size m', fontsize=11)
    ax1.set_title('m Evolution (doubles each round)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # k choices
    ax2 = axes[0, 1]
    ax2.bar(range(len(k_history)), k_history, color='green', alpha=0.7)
    ax2.axhline(y=k_opt, color='red', linestyle='--', label=f'k_opt={k_opt}')
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Chosen k', fontsize=11)
    ax2.set_title('Random k Selection', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Success probability at each k
    ax3 = axes[1, 0]
    ax3.bar(range(len(p_history)), p_history, color='purple', alpha=0.7)
    ax3.axhline(y=0.25, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('P(success) for chosen k', fontsize=11)
    ax3.set_title('Success Probability per Round', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Cumulative queries
    ax4 = axes[1, 1]
    ax4.plot(range(len(cumulative_queries)), cumulative_queries, 'b-o', linewidth=2)
    ax4.axhline(y=k_opt, color='red', linestyle='--', label=f'Optimal: {k_opt} queries')
    ax4.set_xlabel('Round', fontsize=11)
    ax4.set_ylabel('Cumulative Queries', fontsize=11)
    ax4.set_title('Query Accumulation', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exponential_search_trace.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
print("="*60)
print("Oblivious Amplitude Amplification")
print("="*60)

# Visualizations
print("\n1. AVERAGE PROBABILITY ANALYSIS")
visualize_average_probability()

print("\n2. EXPONENTIAL SEARCH VISUALIZATION")
visualize_exponential_search()

# Comparison
print("\n3. OPTIMAL VS OBLIVIOUS COMPARISON")
print("-"*60)

a_values = [0.01, 0.04, 0.1, 0.25]
results = optimal_vs_oblivious_comparison(a_values)

print(f"{'a':>8} | {'k_opt':>8} | {'Obliv. Queries':>15} | {'Success':>10} | {'Ratio':>8}")
print("-"*60)
for r in results:
    print(f"{r['a']:>8.4f} | {r['optimal_k']:>8} | {r['oblivious_queries']:>15.1f} | "
          f"{r['oblivious_success']:>10.3f} | {r['ratio']:>8.2f}")

# Large-scale simulation
print("\n4. LARGE-SCALE SIMULATION")
print("-"*50)

N = 10000
for M in [1, 10, 100, 1000]:
    success, avg_q, _ = exponential_search_simulation(N, M, num_trials=1000)
    a = M/N
    k_opt = int(np.round(np.pi / (4 * np.arcsin(np.sqrt(a))) - 0.5))
    print(f"M={M:>4} (a={a:.4f}): Success={success:.3f}, "
          f"Avg queries={avg_q:.1f}, k_opt={k_opt}")

# Theoretical analysis
print("\n5. THEORETICAL BOUNDS")
print("-"*50)
print("For exponential search:")
print("- Expected queries: O(1/√a) = O(√(N/M))")
print("- Success probability: ≥ 1/4 per round (once m large enough)")
print("- Constant factor overhead: ~2-5x compared to optimal")
print("- No knowledge of a required!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Average probability | $\bar{P}_m = \frac{1}{m}\sum_{k=0}^{m-1}\sin^2((2k+1)\theta)$ |
| Bound for large m | $\bar{P}_m \geq 1/4$ when $m \geq \pi/(4\theta)$ |
| Exponential search complexity | $O(1/\sqrt{a})$ |
| m sequence | $m_r = \min(2^r, \sqrt{N})$ |

### Key Takeaways

1. **Oblivious setting** = don't know success probability
2. **Exponential search** achieves optimal complexity
3. **Random k selection** averages out oscillations
4. **Constant factor overhead** but same asymptotic complexity
5. **No verification needed** during search
6. **Practical** when solution checking is expensive

---

## Daily Checklist

- [ ] I understand the oblivious setting
- [ ] I can explain why random k selection works
- [ ] I know the exponential search strategy
- [ ] I can calculate average success probability
- [ ] I understand the complexity analysis
- [ ] I ran the computational lab and compared approaches

---

*Next: Day 628 — Quantum Counting*
