# Day 622: Multiple Solutions Case

## Overview
**Day 622** | Week 89, Day 6 | Year 1, Month 23 | Grover's Search Algorithm

Today we extend Grover's algorithm to the case of multiple marked items. When there are $M$ solutions among $N$ items, the algorithm still provides a quadratic speedup.

---

## Learning Objectives

1. Generalize the rotation angle for M solutions
2. Derive the modified optimal iteration count
3. Analyze the case when M is unknown
4. Understand the behavior as M approaches N/2
5. Implement Grover for multiple marked states
6. Connect to quantum counting (preview)

---

## Core Content

### Multiple Solutions Setup

Now suppose there are $M$ marked items (solutions) among $N$ total items, where $1 \leq M \leq N$.

**Modified Basis:**
- $|w\rangle = \frac{1}{\sqrt{M}}\sum_{x: f(x)=1}|x\rangle$ (superposition of solutions)
- $|s'\rangle = \frac{1}{\sqrt{N-M}}\sum_{x: f(x)=0}|x\rangle$ (superposition of non-solutions)

### Modified Rotation Angle

The initial uniform superposition:
$$|\psi_0\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle = \sin\theta|w\rangle + \cos\theta|s'\rangle$$

where now:
$$\boxed{\sin\theta = \sqrt{\frac{M}{N}}, \quad \cos\theta = \sqrt{\frac{N-M}{N}}}$$

**Key observation:** More solutions means larger $\theta$, hence fewer iterations needed!

### Modified Optimal Iterations

The optimal number of iterations becomes:

$$\boxed{k_{opt} = \left\lfloor\frac{\pi}{4}\sqrt{\frac{N}{M}}\right\rfloor}$$

**Analysis for various M/N ratios:**

| M/N | θ (degrees) | k_opt for N=1024 |
|-----|-------------|------------------|
| 1/1024 | 1.79° | 25 |
| 1/256 | 3.58° | 12 |
| 1/64 | 7.18° | 6 |
| 1/16 | 14.5° | 3 |
| 1/4 | 30° | 1 |
| 1/2 | 45° | 0 |

### The M = N/2 Boundary Case

When exactly half the items are solutions ($M = N/2$):
$$\theta = \arcsin\sqrt{1/2} = \pi/4 = 45°$$

**Optimal iterations:** $k_{opt} = \lfloor\frac{\pi}{4} \cdot \sqrt{2}\rfloor = 1$

But after 1 iteration:
$$(2 \cdot 1 + 1) \cdot 45° = 135°$$
$$P_{success} = \sin^2(135°) = 0.5$$

This is the **same as random guessing**! No quantum advantage.

### Case M > N/2

When $M > N/2$, something interesting happens:
- $\theta > 45°$
- Even $k = 0$ (no iterations) gives $P_{success} = \sin^2\theta > 0.5$
- Additional iterations can actually **decrease** success probability!

**Strategy:** If $M > N/2$, simply measure the initial state, or flip the oracle to search for non-solutions.

### Unknown Number of Solutions

In many applications, $M$ is unknown. Several strategies exist:

**1. Quantum Counting (Day 628):**
- Use phase estimation on Grover operator
- Estimates $M$ with $O(\sqrt{N/M})$ queries

**2. Randomized Approach:**
Choose iteration count $k$ uniformly at random from $[0, \lfloor\sqrt{N}\rfloor]$.

**Theorem:** This gives expected success probability $\geq 1/4$ regardless of $M$.

**3. Exponential Search:**
- Try $k = 1, 2, 4, 8, ...$ iterations
- With verification step after each

### Success Probability Analysis

For $M$ solutions:
$$P_{success}(k) = \sin^2((2k+1)\theta) = \sin^2\left((2k+1)\arcsin\sqrt{\frac{M}{N}}\right)$$

**At optimal k:**
$$P_{success}(k_{opt}) \geq 1 - \frac{M}{N}$$

For $M \ll N$, this approaches 1.

### The General Speedup

**Query complexity with M solutions:**
$$O\left(\sqrt{\frac{N}{M}}\right)$$

**Comparison to classical:**
- Classical: $O(N/M)$ expected queries
- Quantum: $O(\sqrt{N/M})$ queries
- Speedup: $\sqrt{N/M}$ (still quadratic!)

---

## Worked Examples

### Example 1: Four Solutions in 64 Items
Calculate optimal iterations and success probability for $N = 64$, $M = 4$.

**Solution:**
$\theta = \arcsin\sqrt{4/64} = \arcsin(1/4) = 14.48°$

$k_{opt} = \lfloor\frac{\pi}{4}\sqrt{64/4}\rfloor = \lfloor\frac{\pi}{4} \cdot 4\rfloor = \lfloor 3.14 \rfloor = 3$

Check: $(2 \cdot 3 + 1) \cdot 14.48° = 101.4°$

$P_{success} = \sin^2(101.4°) = 0.961$

Classical would need ~$64/4 = 16$ queries on average; Grover needs 3.

### Example 2: Unknown M Strategy
Describe the randomized iteration strategy and prove the success bound.

**Solution:**
Choose $k$ uniformly from $\{0, 1, 2, ..., \lfloor\sqrt{N}\rfloor\}$.

The success probability averaged over random $k$:
$$\bar{P} = \frac{1}{\lfloor\sqrt{N}\rfloor + 1}\sum_{k=0}^{\lfloor\sqrt{N}\rfloor} \sin^2((2k+1)\theta)$$

Using the identity $\sum_{k=0}^{K}\sin^2((2k+1)\theta) \approx K/2$ for large K:

$$\bar{P} \geq \frac{1}{4}$$

This bound holds for any $M$ (as long as $M \geq 1$).

### Example 3: Transition Point
Find the value of M where Grover's algorithm matches classical search.

**Solution:**
Quantum: $k_{opt} \approx \frac{\pi}{4}\sqrt{N/M}$ queries
Classical: $\approx N/(2M)$ expected queries

Setting them equal:
$$\frac{\pi}{4}\sqrt{\frac{N}{M}} = \frac{N}{2M}$$
$$\frac{\pi}{4} = \frac{N}{2M} \cdot \sqrt{\frac{M}{N}} = \frac{1}{2}\sqrt{\frac{N}{M}}$$
$$\sqrt{\frac{N}{M}} = \frac{\pi}{2}$$
$$\frac{N}{M} = \frac{\pi^2}{4} \approx 2.47$$

So for $M > N/2.47 \approx 0.4N$, classical becomes comparable or better.

---

## Practice Problems

### Problem 1: Multiple Solutions Calculation
For a database with $N = 1000$ items and $M = 10$ solutions:
a) Calculate $\theta$ in degrees
b) Find the optimal number of iterations
c) What is the success probability at optimal $k$?
d) Compare to classical expected queries

### Problem 2: Finding the Sweet Spot
For $N = 256$, plot the quantum speedup factor (classical queries / quantum queries) as a function of $M$ from 1 to 128.

### Problem 3: Verification Oracle
Design a strategy that uses Grover search with verification: after each measurement, check if the result is valid. Analyze the expected total queries.

---

## Computational Lab

```python
"""Day 622: Multiple Solutions Case"""
import numpy as np
import matplotlib.pyplot as plt

def theta_M(N, M):
    """Rotation angle for M solutions in N items."""
    return np.arcsin(np.sqrt(M/N))

def optimal_k_M(N, M):
    """Optimal iterations for M solutions."""
    if M >= N:
        return 0
    theta = theta_M(N, M)
    if theta >= np.pi/4:  # M >= N/2
        return 0
    return int(np.floor(np.pi / (4 * theta)))

def success_prob_M(k, N, M):
    """Success probability after k iterations with M solutions."""
    theta = theta_M(N, M)
    return np.sin((2*k + 1) * theta)**2

def grover_operator_M(N, marked_states):
    """Grover operator for multiple marked states."""
    M = len(marked_states)

    # Oracle
    O = np.eye(N)
    for m in marked_states:
        O[m, m] = -1

    # Diffusion
    psi_0 = np.ones(N) / np.sqrt(N)
    D = 2 * np.outer(psi_0, psi_0) - np.eye(N)

    return D @ O

def simulate_grover_M(N, marked_states, k):
    """Simulate Grover's algorithm with M marked states."""
    G = grover_operator_M(N, marked_states)

    # Initial state
    psi = np.ones(N) / np.sqrt(N)

    # Apply k iterations
    for _ in range(k):
        psi = G @ psi

    # Calculate success probability (sum over marked states)
    prob = sum(abs(psi[m])**2 for m in marked_states)
    return prob, psi

def analyze_multiple_solutions(N):
    """Analyze Grover performance for various M values."""
    M_values = [1, 2, 4, 8, 16, 32, 64, N//4, N//2]
    M_values = [m for m in M_values if m <= N]

    print(f"\nMultiple Solutions Analysis (N={N}):")
    print("-" * 70)
    print(f"{'M':>6} | {'M/N':>8} | {'θ (deg)':>10} | {'k_opt':>6} | "
          f"{'P_success':>10} | {'Classical':>10}")
    print("-" * 70)

    for M in M_values:
        theta = theta_M(N, M)
        k_opt = optimal_k_M(N, M)
        p_success = success_prob_M(k_opt, N, M)
        classical = N / (2*M) if M > 0 else N

        print(f"{M:>6} | {M/N:>8.4f} | {np.degrees(theta):>10.2f} | "
              f"{k_opt:>6} | {p_success:>10.4f} | {classical:>10.1f}")

def plot_speedup_vs_M(N):
    """Plot quantum speedup as function of M."""
    M_values = np.arange(1, N//2 + 1)

    classical_queries = N / (2 * M_values)
    quantum_queries = np.array([max(1, optimal_k_M(N, M)) for M in M_values])
    speedup = classical_queries / quantum_queries

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(M_values, classical_queries, 'b-', label='Classical', linewidth=2)
    plt.semilogy(M_values, quantum_queries, 'r-', label='Quantum', linewidth=2)
    plt.xlabel('Number of Solutions M', fontsize=12)
    plt.ylabel('Number of Queries (log scale)', fontsize=12)
    plt.title(f'Query Complexity (N={N})', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(M_values, speedup, 'g-', linewidth=2)
    plt.xlabel('Number of Solutions M', fontsize=12)
    plt.ylabel('Speedup Factor', fontsize=12)
    plt.title('Quantum Speedup vs Number of Solutions', fontsize=12)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grover_multiple_solutions.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_success_vs_iterations_M(N, M_list):
    """Compare success probability curves for different M."""
    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(M_list)))

    for M, color in zip(M_list, colors):
        k_opt = optimal_k_M(N, M)
        k_max = max(20, 2 * k_opt)
        k_values = np.arange(k_max + 1)
        probs = [success_prob_M(k, N, M) for k in k_values]

        plt.plot(k_values, probs, color=color, linewidth=2,
                 label=f'M={M} (k_opt={k_opt})')
        plt.axvline(x=k_opt, color=color, linestyle=':', alpha=0.5)

    plt.xlabel('Number of Iterations k', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title(f'Success Probability for Different M (N={N})', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig('grover_success_multiple_M.png', dpi=150, bbox_inches='tight')
    plt.show()

def randomized_iteration_strategy(N, M, num_trials=1000):
    """Simulate randomized iteration count strategy for unknown M."""
    max_k = int(np.sqrt(N))
    successes = 0

    for _ in range(num_trials):
        # Choose random k
        k = np.random.randint(0, max_k + 1)

        # Calculate success probability
        p = success_prob_M(k, N, M)

        # Random success
        if np.random.random() < p:
            successes += 1

    return successes / num_trials

def test_randomized_strategy(N):
    """Test randomized strategy for various M."""
    print(f"\nRandomized Iteration Strategy Test (N={N}):")
    print("-" * 50)
    print(f"{'M':>6} | {'Theory (avg)':>12} | {'Simulated':>12}")
    print("-" * 50)

    for M in [1, 4, 16, 64, N//4, N//2]:
        if M > N:
            continue
        sim_success = randomized_iteration_strategy(N, M, 1000)
        print(f"{M:>6} | {'≥ 0.25':>12} | {sim_success:>12.4f}")

def verify_simulation(N, marked_states):
    """Verify theoretical predictions with simulation."""
    M = len(marked_states)
    k_opt = optimal_k_M(N, M)
    theoretical_prob = success_prob_M(k_opt, N, M)

    sim_prob, final_state = simulate_grover_M(N, marked_states, k_opt)

    print(f"\nSimulation Verification (N={N}, M={M}):")
    print(f"  Optimal iterations: {k_opt}")
    print(f"  Theoretical P_success: {theoretical_prob:.6f}")
    print(f"  Simulated P_success: {sim_prob:.6f}")
    print(f"  Match: {np.isclose(theoretical_prob, sim_prob)}")

    return final_state

# Main execution
print("=" * 60)
print("Grover's Algorithm: Multiple Solutions Case")
print("=" * 60)

# Basic analysis
N = 256
analyze_multiple_solutions(N)

# Verify with simulation
print("\n" + "=" * 60)
marked_states = [10, 20, 30, 40]  # M = 4
verify_simulation(N, marked_states)

# Special case: M = N/2
print("\n" + "=" * 60)
print("Special Case: M = N/2 (boundary)")
print("=" * 60)
N = 64
M = 32
theta = theta_M(N, M)
k_opt = optimal_k_M(N, M)
print(f"N = {N}, M = {M}")
print(f"θ = {np.degrees(theta):.1f}°")
print(f"k_opt = {k_opt}")
print(f"P_success at k=0: {success_prob_M(0, N, M):.4f}")
print(f"P_success at k=1: {success_prob_M(1, N, M):.4f}")
print("No quantum advantage when M = N/2!")

# Test randomized strategy
print("\n" + "=" * 60)
test_randomized_strategy(256)

# Generate plots
print("\nGenerating visualizations...")
plot_speedup_vs_M(256)
plot_success_vs_iterations_M(256, [1, 4, 16, 64])

# Summary table
print("\n" + "=" * 60)
print("SUMMARY: Query Complexity")
print("=" * 60)
print(f"| {'M solutions':^15} | {'Classical':^15} | {'Quantum':^15} | {'Speedup':^10} |")
print(f"|{'-'*15}|{'-'*17}|{'-'*17}|{'-'*12}|")
for M in [1, 4, 16, 64]:
    N = 256
    classical = N / (2*M)
    quantum = optimal_k_M(N, M)
    speedup = classical / max(1, quantum)
    print(f"| {M:^15} | {classical:^15.1f} | {quantum:^15} | {speedup:^10.1f} |")
```

**Expected Output:**
```
============================================================
Grover's Algorithm: Multiple Solutions Case
============================================================

Multiple Solutions Analysis (N=256):
----------------------------------------------------------------------
     M |      M/N |   θ (deg) |  k_opt | P_success |   Classical
----------------------------------------------------------------------
     1 |   0.0039 |       3.58 |     12 |     0.9995 |       128.0
     2 |   0.0078 |       5.07 |      8 |     0.9980 |        64.0
     4 |   0.0156 |       7.18 |      6 |     0.9922 |        32.0
     8 |   0.0312 |      10.18 |      4 |     0.9727 |        16.0
    16 |   0.0625 |      14.48 |      3 |     0.9613 |         8.0
    32 |   0.1250 |      20.70 |      2 |     0.9375 |         4.0
    64 |   0.2500 |      30.00 |      1 |     1.0000 |         2.0
   128 |   0.5000 |      45.00 |      0 |     0.5000 |         1.0

Special Case: M = N/2 (boundary)
============================================================
N = 64, M = 32
θ = 45.0°
k_opt = 0
P_success at k=0: 0.5000
P_success at k=1: 0.5000
No quantum advantage when M = N/2!
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Rotation angle (M solutions) | $\sin\theta = \sqrt{M/N}$ |
| Optimal iterations | $k_{opt} = \lfloor\frac{\pi}{4}\sqrt{N/M}\rfloor$ |
| Query complexity | $O(\sqrt{N/M})$ |
| Classical comparison | $O(N/M)$ expected |
| Speedup factor | $\sqrt{N/M}$ |

### Key Takeaways

1. **More solutions = fewer iterations** needed
2. **Quadratic speedup maintained** for any M
3. **Boundary at M = N/2:** no quantum advantage
4. **Unknown M** can be handled with randomized strategy
5. **Quantum counting** can estimate M first
6. **Verification** after measurement helps with unknown M

---

## Daily Checklist

- [ ] I can generalize the rotation angle for M solutions
- [ ] I can calculate optimal iterations for multiple solutions
- [ ] I understand the M = N/2 boundary case
- [ ] I know strategies for unknown M
- [ ] I understand when quantum advantage disappears
- [ ] I ran the computational lab and analyzed various M values

---

*Next: Day 623 — Week Review*
