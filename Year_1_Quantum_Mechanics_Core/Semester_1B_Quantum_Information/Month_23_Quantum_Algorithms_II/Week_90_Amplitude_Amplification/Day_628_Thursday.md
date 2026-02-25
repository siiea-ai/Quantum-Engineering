# Day 628: Quantum Counting

## Overview
**Day 628** | Week 90, Day 5 | Year 1, Month 23 | Amplitude Amplification

Today we learn quantum counting, which uses amplitude estimation to count the number of solutions to a search problem. This enables us to know M before running Grover's algorithm.

---

## Learning Objectives

1. Define the counting problem formally
2. Apply amplitude estimation to count solutions
3. Derive the precision of counting
4. Implement quantum counting circuits
5. Analyze query complexity
6. Connect counting to optimization problems

---

## Core Content

### The Counting Problem

**Problem:** Given an oracle $f: \{0,1\}^n \to \{0,1\}$, determine $M = |\{x : f(x) = 1\}|$.

**Classical approach:** Query all $N = 2^n$ items: $O(N)$ queries.

**Quantum approach:** Use amplitude estimation on Grover operator.

### Counting via Amplitude Estimation

**Key relation:** $a = M/N = \sin^2\theta$

If we can estimate $\theta$, we can compute:
$$\boxed{M = N \sin^2\theta}$$

**Algorithm:**
1. Construct Grover operator $G$
2. Run amplitude estimation with $m$ precision qubits
3. Obtain estimate $\tilde{\theta}$
4. Compute $\tilde{M} = N \sin^2\tilde{\theta}$

### Quantum Counting Circuit

```
|0⟩^m ──H^⊗m──[Controlled-G^{2^k}]──[QFT†]──[Measure]──→ θ̃

|0⟩^n ──H^⊗n──[          ↑        ]────────────────────→
```

The phase estimation extracts the eigenphase of $G$, which encodes $\theta$.

### Precision Analysis

With $m$ precision qubits:
- Phase estimate: $\tilde{\theta}$ accurate to $\pm \frac{\pi}{2^{m+1}}$
- Amplitude estimate: $\tilde{a} = \sin^2\tilde{\theta}$
- Count estimate: $\tilde{M} = N\tilde{a}$

**Error in M:**
$$|\tilde{M} - M| \leq N \cdot 2\sin\theta\cos\theta \cdot \frac{\pi}{2^{m+1}} = \frac{\pi\sqrt{M(N-M)}}{2^m}$$

### Additive vs Multiplicative Error

**Additive error $\epsilon$:** $|\tilde{M} - M| \leq \epsilon$

Requires: $m = O(\log(N/\epsilon))$ qubits

Query complexity: $O(N/\epsilon)$ controlled-G operations

Wait, that doesn't seem like an improvement. Let's be more careful:

Total queries: $\sum_{k=0}^{m-1} 2^k = 2^m - 1$

For additive error $\epsilon$:
- Need $\frac{\pi\sqrt{MN}}{2^m} \leq \epsilon$
- So $2^m \geq \frac{\pi\sqrt{MN}}{\epsilon}$
- Queries: $O\left(\frac{\sqrt{MN}}{\epsilon}\right)$

**For constant $\epsilon$ (like $\epsilon = 1$):**
- Queries: $O(\sqrt{MN})$

**Classical counting:** $O(N)$ queries

**Quantum speedup:** From $O(N)$ to $O(\sqrt{MN})$

### Special Cases

**When M is small ($M \ll N$):**
$$\text{Quantum: } O(\sqrt{MN}) \ll O(N) \text{ (classical)}$$

Strong speedup!

**When M ≈ N/2:**
$$\sqrt{MN} \approx N/\sqrt{2}$$

Still better than classical by factor $\sqrt{2}$.

### Multiplicative Error

For relative error $\delta$: $|\tilde{M} - M| \leq \delta M$

Analysis: Need $\frac{\sqrt{MN}}{2^m} \leq \delta M$

$2^m \geq \frac{\sqrt{N}}{δ\sqrt{M}}$

Queries: $O\left(\frac{\sqrt{N/M}}{\delta}\right) = O\left(\frac{1}{\delta\sqrt{a}}\right)$

This is the same as amplitude estimation!

### Applications of Quantum Counting

1. **Determine optimal Grover iterations:** Once we know $M$, we can compute $k_{opt}$

2. **Decision problems:** Is $M > 0$? Is $M > k$?

3. **Approximate counting:** Estimate size of solution sets

4. **SAT counting:** Count satisfying assignments

5. **Graph problems:** Count cliques, matchings, etc.

---

## Worked Examples

### Example 1: Basic Counting
Use quantum counting to estimate $M$ when $N = 1024$ and true $M = 64$.

**Solution:**
$a = 64/1024 = 1/16$
$\theta = \arcsin(1/4) = 0.2527$ rad

For additive error $\epsilon = 10$:
Need $2^m \geq \frac{\pi \sqrt{64 \times 1024}}{10} = \frac{\pi \times 256}{10} \approx 80$

$m = 7$ qubits (since $2^7 = 128 > 80$)

Total queries: $2^7 - 1 = 127$

Classical would need 1024 queries.

### Example 2: Deciding M > 0
Determine if a database has any solution ($M \geq 1$).

**Solution:**
Classical: Must check all N in worst case.

Quantum counting with error $\epsilon = 0.5$:
- If $M = 0$: estimate $\tilde{M} < 0.5$, declare "no solutions"
- If $M \geq 1$: estimate $\tilde{M} > 0.5$, declare "solutions exist"

For $M = 1$: Need $2^m \geq \frac{\pi\sqrt{N}}{0.5} = 2\pi\sqrt{N}$

Queries: $O(\sqrt{N})$ — quadratic speedup!

### Example 3: Precision Tradeoff
How does counting precision scale with queries?

**Solution:**
With $Q$ queries (Q = $2^m$):

Additive error: $\epsilon \sim \frac{\sqrt{MN}}{Q}$

To halve the error, double the queries.

This is like "standard quantum limit" scaling in metrology.

---

## Practice Problems

### Problem 1: Counting Circuit
For $n = 4$ qubits and $m = 3$ precision qubits, draw the quantum counting circuit and list all controlled operations.

### Problem 2: Error Analysis
If true $M = 100$ in $N = 10000$, and we use $m = 8$ precision qubits:
a) What is the expected error in $\tilde{M}$?
b) What is the relative error?

### Problem 3: Query Lower Bound
Prove that quantum counting cannot achieve better than $O(\sqrt{N})$ queries for distinguishing $M = 0$ from $M = 1$.

---

## Computational Lab

```python
"""Day 628: Quantum Counting"""
import numpy as np
import matplotlib.pyplot as plt

def quantum_counting_estimate(N, M, m_precision):
    """
    Simulate quantum counting.

    Args:
        N: Total number of items
        M: True number of solutions
        m_precision: Number of precision qubits

    Returns:
        Estimated M and error
    """
    # True parameters
    a_true = M / N
    theta_true = np.arcsin(np.sqrt(a_true))

    # Phase estimation gives 2θ/2π (or the negative)
    true_phase = theta_true / np.pi  # θ/π = 2θ/(2π)

    # Discretize to m bits
    k = int(np.round(true_phase * 2**m_precision))
    k = k % 2**m_precision

    estimated_phase = k / 2**m_precision
    theta_est = estimated_phase * np.pi

    # Could get positive or negative θ; both give same sin²
    a_est = np.sin(theta_est)**2
    M_est = N * a_est

    error = abs(M_est - M)

    return M_est, error

def quantum_counting_simulation(N, M, m_precision, num_trials=1000):
    """
    Simulate quantum counting with noise/randomness.
    """
    a_true = M / N
    theta_true = np.arcsin(np.sqrt(a_true))
    true_phase = theta_true / np.pi

    estimates = []

    for _ in range(num_trials):
        # Add some noise to simulate QPE imperfection
        # In reality, QPE has probability distribution over outcomes
        noise = np.random.uniform(-0.5, 0.5) / 2**m_precision
        measured_phase = true_phase + noise

        # Discretize
        k = int(np.round(measured_phase * 2**m_precision))
        k = k % 2**m_precision
        estimated_phase = k / 2**m_precision

        theta_est = estimated_phase * np.pi
        a_est = np.sin(theta_est)**2
        M_est = N * a_est

        estimates.append(M_est)

    return np.array(estimates)

def analyze_counting_precision(N, M_values, m_values):
    """Analyze counting error vs precision qubits."""
    results = []

    for M in M_values:
        for m in m_values:
            estimates = quantum_counting_simulation(N, M, m, 500)
            mean_est = np.mean(estimates)
            std_est = np.std(estimates)
            error = abs(mean_est - M)

            results.append({
                'M': M,
                'm': m,
                'mean': mean_est,
                'std': std_est,
                'error': error,
                'relative_error': error / M if M > 0 else 0
            })

    return results

def plot_counting_vs_classical(N_values):
    """Compare quantum vs classical counting complexity."""
    # For additive error ε = 1
    epsilon = 1

    classical_queries = N_values  # Must check everything

    quantum_queries = []
    for N in N_values:
        # Assume M = sqrt(N) for typical case
        M = int(np.sqrt(N))
        # Queries = O(sqrt(MN))
        q = np.sqrt(M * N)
        quantum_queries.append(q)

    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, classical_queries, 'b-', label='Classical O(N)', linewidth=2)
    plt.loglog(N_values, quantum_queries, 'r-', label='Quantum O(√MN)', linewidth=2)

    plt.xlabel('Database Size N', fontsize=12)
    plt.ylabel('Number of Queries', fontsize=12)
    plt.title('Quantum Counting vs Classical', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantum_counting_complexity.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_counting_accuracy():
    """Visualize counting accuracy for different M."""
    N = 1024

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    M_values = [4, 16, 64, 256]
    m_precision = 8

    for idx, M in enumerate(M_values):
        ax = axes[idx // 2, idx % 2]

        estimates = quantum_counting_simulation(N, M, m_precision, 1000)

        ax.hist(estimates, bins=30, density=True, alpha=0.7, color='blue')
        ax.axvline(x=M, color='red', linestyle='--', linewidth=2,
                   label=f'True M = {M}')

        mean_est = np.mean(estimates)
        ax.axvline(x=mean_est, color='green', linestyle=':',
                   label=f'Mean = {mean_est:.1f}')

        ax.set_xlabel('Estimated M', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'M = {M}, a = {M/N:.4f}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('counting_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()

def counting_error_vs_precision():
    """Show how counting error decreases with precision."""
    N = 1024
    M = 64

    m_values = range(3, 12)
    errors = []
    theoretical = []

    for m in m_values:
        estimates = quantum_counting_simulation(N, M, m, 500)
        error = np.std(estimates)
        errors.append(error)

        # Theoretical: error ~ sqrt(MN) / 2^m
        theo_error = np.sqrt(M * N) / 2**m * np.pi
        theoretical.append(theo_error)

    plt.figure(figsize=(10, 6))
    plt.semilogy(m_values, errors, 'bo-', label='Simulated Error', linewidth=2)
    plt.semilogy(m_values, theoretical, 'r--', label='Theoretical O(√MN/2^m)', linewidth=2)

    plt.xlabel('Precision Qubits m', fontsize=12)
    plt.ylabel('Error in M Estimate', fontsize=12)
    plt.title(f'Counting Error vs Precision (N={N}, M={M})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('counting_error_precision.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
print("="*60)
print("Quantum Counting")
print("="*60)

# Basic example
N = 1024
M = 64
m = 8

print(f"\n1. BASIC COUNTING EXAMPLE")
print(f"   N = {N}, True M = {M}")
print(f"   Using m = {m} precision qubits")

M_est, error = quantum_counting_estimate(N, M, m)
print(f"   Estimated M = {M_est:.2f}")
print(f"   Error = {error:.2f}")

# Theoretical error
theo_error = np.pi * np.sqrt(M * N) / 2**m
print(f"   Theoretical error bound: {theo_error:.2f}")

# Statistical analysis
print(f"\n2. STATISTICAL ANALYSIS")
print("-"*50)

estimates = quantum_counting_simulation(N, M, m, 1000)
print(f"   Mean estimate: {np.mean(estimates):.2f}")
print(f"   Std deviation: {np.std(estimates):.2f}")
print(f"   Min: {np.min(estimates):.2f}, Max: {np.max(estimates):.2f}")

# Complexity comparison
print(f"\n3. COMPLEXITY COMPARISON")
print("-"*50)

N_values = [2**k for k in range(6, 16)]
plot_counting_vs_classical(N_values)

# Accuracy visualization
print(f"\n4. ACCURACY VISUALIZATION")
plot_counting_accuracy()

# Error vs precision
print(f"\n5. ERROR VS PRECISION")
counting_error_vs_precision()

# Query complexity table
print(f"\n6. QUERY COMPLEXITY TABLE")
print("-"*60)
print(f"{'N':>10} | {'M':>8} | {'Classical':>12} | {'Quantum':>12} | {'Speedup':>10}")
print("-"*60)

for N in [256, 1024, 4096, 16384]:
    for M_ratio in [0.01, 0.1]:
        M = int(M_ratio * N)
        classical = N
        # For additive error 1
        quantum = int(np.sqrt(M * N) * np.pi)
        speedup = classical / quantum

        print(f"{N:>10} | {M:>8} | {classical:>12} | {quantum:>12} | {speedup:>10.1f}x")

print("-"*60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Count from amplitude | $M = N\sin^2\theta$ |
| Additive error | $\|\tilde{M} - M\| \leq \frac{\pi\sqrt{MN}}{2^m}$ |
| Quantum queries | $O(\sqrt{MN}/\epsilon)$ for error $\epsilon$ |
| Classical queries | $O(N)$ |
| Speedup | $\sqrt{N/M}$ for small M |

### Key Takeaways

1. **Quantum counting** uses amplitude estimation on Grover
2. **Quadratic speedup** over classical exact counting
3. **Useful for determining optimal k** before Grover search
4. **Error decreases** exponentially with precision qubits
5. **Applications** in SAT counting, graph problems
6. **Foundation for** quantum approximate counting

---

## Daily Checklist

- [ ] I understand the counting problem
- [ ] I know how amplitude estimation gives M
- [ ] I can analyze counting precision
- [ ] I understand the query complexity
- [ ] I know applications of quantum counting
- [ ] I ran the computational lab and verified the analysis

---

*Next: Day 629 — Applications of Amplitude Amplification*
