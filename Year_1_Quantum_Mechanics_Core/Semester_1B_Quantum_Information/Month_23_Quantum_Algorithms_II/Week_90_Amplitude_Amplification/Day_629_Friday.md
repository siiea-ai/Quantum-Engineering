# Day 629: Applications of Amplitude Amplification

## Overview
**Day 629** | Week 90, Day 6 | Year 1, Month 23 | Amplitude Amplification

Today we explore practical applications of amplitude amplification, including SAT solving, quantum speedups for Monte Carlo methods, and optimization problems.

---

## Learning Objectives

1. Apply amplitude amplification to SAT problems
2. Understand quantum speedup for Monte Carlo
3. Design amplitude amplification subroutines
4. Analyze practical speedups and limitations
5. Connect to quantum machine learning
6. Evaluate when quantum advantages apply

---

## Core Content

### Application 1: SAT Solving

**Problem:** Given a Boolean formula $\phi(x_1, ..., x_n)$, find a satisfying assignment.

**Classical:** Brute force: $O(2^n)$ evaluations

**Quantum (Grover):** $O(2^{n/2})$ oracle calls

**With structure:** If partial solutions can be identified, amplitude amplification on subproblems.

### SAT Oracle Construction

For a formula $\phi$ in CNF:
$$\phi = (l_{11} \lor l_{12} \lor l_{13}) \land (l_{21} \lor l_{22}) \land ...$$

**Oracle circuit:**
1. Compute each clause into ancilla
2. AND all clause results
3. Apply phase flip if all satisfied
4. Uncompute

**Complexity:** Oracle has $O(\text{poly}(n, m))$ gates for $n$ variables, $m$ clauses.

### Application 2: Quantum Monte Carlo

**Problem:** Estimate $\mathbb{E}[f(X)]$ where $X$ is a random variable.

**Classical Monte Carlo:**
- $N$ samples give error $O(1/\sqrt{N})$
- For error $\epsilon$: need $O(1/\epsilon^2)$ samples

**Quantum approach (Montanaro 2015):**
1. Prepare superposition encoding distribution
2. Use amplitude estimation
3. Error $\epsilon$ with $O(1/\epsilon)$ queries

**Quadratic speedup in precision!**

### Quantum Monte Carlo Circuit

```
|0⟩^n ──[Prepare X]──[Compute f]──[Amplitude Estimation]──→ Ẽ[f(X)]
```

where "Prepare X" creates:
$$\sum_x \sqrt{P(x)}|x\rangle$$

### Application 3: Minimum Finding

**Problem:** Find minimum of function $f: \{0,...,N-1\} \to \mathbb{R}$

**Durr-Hoyer Algorithm:**
1. Pick random $y$ as current minimum candidate
2. Mark all $x$ with $f(x) < f(y)$
3. Apply Grover to find such $x$
4. Update $y \leftarrow x$
5. Repeat until no improvement

**Complexity:** $O(\sqrt{N})$ expected queries

### Application 4: Quantum Speedup for NP

For NP problems with $M$ solutions among $N$ candidates:

| Problem | Classical | Quantum |
|---------|-----------|---------|
| Decision | $O(N)$ | $O(\sqrt{N})$ |
| Search | $O(N/M)$ | $O(\sqrt{N/M})$ |
| Counting | $O(N)$ | $O(\sqrt{MN})$ |

**Important:** This is polynomial speedup, not exponential.

NP-complete problems remain hard: $O(2^{n/2})$ vs $O(2^n)$.

### Application 5: Quantum Walk Speedups

Amplitude amplification combines with quantum walks:

**Element distinctness:** Given list of $N$ items, are any two equal?
- Classical: $O(N)$
- Quantum: $O(N^{2/3})$ using quantum walks + AA

**Triangle finding:** Does a graph have a triangle?
- Classical: $O(N^2)$ or $O(N^\omega)$ (matrix multiplication)
- Quantum: $O(N^{5/4})$

### Application 6: Machine Learning Speedups

**Quantum sampling:** Amplitude amplification for sampling from distributions

**Feature selection:** Search over feature subsets

**Hyperparameter optimization:** Search over parameter space

**Caveat:** Data loading often dominates, limiting practical speedup.

### Limitations and Caveats

1. **Oracle construction:** Often dominates query complexity
2. **Coherence time:** Deep circuits need error correction
3. **Classical heuristics:** May outperform quantum for structured problems
4. **Data loading:** QRAM assumptions not always practical
5. **Constant factors:** Quantum overhead in practice

---

## Worked Examples

### Example 1: 3-SAT with Grover
A 3-SAT formula with $n = 20$ variables has an unknown number of solutions.

**Analysis:**
Search space: $N = 2^{20} \approx 10^6$

Classical brute force: $2^{20}$ evaluations

Quantum (Grover): $\frac{\pi}{4}\sqrt{2^{20}} = \frac{\pi}{4} \times 1024 \approx 804$ iterations

If $M$ solutions exist: $\frac{\pi}{4}\sqrt{2^{20}/M}$ iterations

**Speedup:** $\sqrt{2^{20}/M} = 2^{10}/\sqrt{M} \approx 1024/\sqrt{M}$

### Example 2: Monte Carlo Integration
Estimate $\int_0^1 f(x) dx$ to precision $\epsilon = 0.001$.

**Classical:** Need $N = O(1/\epsilon^2) = 10^6$ samples

**Quantum:** Need $O(1/\epsilon) = 1000$ amplitude estimation queries

**Speedup:** $1000\times$

**Caveat:** Each quantum query requires coherent evaluation of $f$.

### Example 3: Minimum Finding
Find minimum in array of $N = 10^6$ elements.

**Classical:** $O(N) = 10^6$ comparisons

**Quantum (Durr-Hoyer):** Expected $O(\sqrt{N}) = 1000$ queries

**Speedup:** $1000\times$

---

## Practice Problems

### Problem 1: SAT Oracle
Design the oracle circuit for the formula:
$$\phi = (x_1 \lor \neg x_2) \land (x_2 \lor x_3) \land (\neg x_1 \lor \neg x_3)$$

### Problem 2: Monte Carlo Speedup
A financial model requires Monte Carlo with 1 billion samples classically. How many quantum queries would achieve equivalent precision?

### Problem 3: Hybrid Algorithm
Design an algorithm that uses amplitude amplification as a subroutine within a larger classical optimization loop.

---

## Computational Lab

```python
"""Day 629: Applications of Amplitude Amplification"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def sat_oracle(formula, n_vars):
    """
    Create oracle for SAT formula.

    formula: list of clauses, each clause is list of literals
             positive int = variable, negative = negation
    """
    N = 2**n_vars

    def evaluate(assignment):
        """Evaluate formula on assignment (tuple of 0/1)."""
        for clause in formula:
            clause_satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                var_val = assignment[var_idx]
                if lit > 0 and var_val == 1:
                    clause_satisfied = True
                    break
                if lit < 0 and var_val == 0:
                    clause_satisfied = True
                    break
            if not clause_satisfied:
                return False
        return True

    # Build oracle matrix
    oracle = np.eye(N)
    satisfying = []

    for i in range(N):
        assignment = tuple(int(b) for b in format(i, f'0{n_vars}b'))
        if evaluate(assignment):
            oracle[i, i] = -1
            satisfying.append((i, assignment))

    return oracle, satisfying

def grover_for_sat(formula, n_vars, max_iter=None):
    """
    Apply Grover's algorithm to find SAT solution.
    """
    N = 2**n_vars
    oracle, satisfying = sat_oracle(formula, n_vars)
    M = len(satisfying)

    if M == 0:
        return None, 0, "UNSAT"

    # Diffusion operator
    psi_0 = np.ones(N) / np.sqrt(N)
    diffusion = 2 * np.outer(psi_0, psi_0) - np.eye(N)

    # Grover operator
    G = diffusion @ oracle

    # Optimal iterations
    theta = np.arcsin(np.sqrt(M/N))
    if max_iter is None:
        k_opt = int(np.round(np.pi / (4*theta) - 0.5))
    else:
        k_opt = max_iter

    # Apply iterations
    state = psi_0.copy()
    for _ in range(k_opt):
        state = G @ state

    # Probability of satisfying assignments
    prob_sat = sum(abs(state[s[0]])**2 for s in satisfying)

    return satisfying, k_opt, prob_sat

def monte_carlo_comparison():
    """Compare classical vs quantum Monte Carlo scaling."""
    epsilon_values = np.logspace(-1, -4, 20)

    classical = 1 / epsilon_values**2
    quantum = 1 / epsilon_values

    plt.figure(figsize=(10, 6))
    plt.loglog(epsilon_values, classical, 'b-', label='Classical O(1/ε²)', linewidth=2)
    plt.loglog(epsilon_values, quantum, 'r-', label='Quantum O(1/ε)', linewidth=2)

    plt.xlabel('Required Precision ε', fontsize=12)
    plt.ylabel('Number of Samples/Queries', fontsize=12)
    plt.title('Monte Carlo: Classical vs Quantum', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig('monte_carlo_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def durr_hoyer_simulation(arr, num_trials=100):
    """
    Simulate Durr-Hoyer minimum finding algorithm.
    """
    N = len(arr)
    true_min_idx = np.argmin(arr)
    true_min = arr[true_min_idx]

    total_queries = []

    for _ in range(num_trials):
        # Start with random candidate
        y_idx = np.random.randint(N)
        y = arr[y_idx]
        queries = 1

        while True:
            # Count elements smaller than y
            smaller = [i for i in range(N) if arr[i] < y]
            M = len(smaller)

            if M == 0:
                # y is the minimum
                break

            # Grover iterations to find smaller element
            theta = np.arcsin(np.sqrt(M/N))
            k = int(np.round(np.pi / (4*theta) - 0.5))
            queries += k

            # Simulate finding a smaller element
            # (In real Grover, we'd measure and get one)
            new_idx = np.random.choice(smaller)
            y_idx = new_idx
            y = arr[y_idx]

        total_queries.append(queries)
        assert y == true_min, "Algorithm failed to find minimum"

    return np.mean(total_queries), np.std(total_queries)

def analyze_np_speedups():
    """Analyze quantum speedups for NP problems."""
    n_values = np.arange(10, 31)

    classical = 2.0**n_values
    quantum = 2.0**(n_values/2)

    plt.figure(figsize=(10, 6))
    plt.semilogy(n_values, classical, 'b-', label='Classical O(2ⁿ)', linewidth=2)
    plt.semilogy(n_values, quantum, 'r-', label='Quantum O(2^{n/2})', linewidth=2)

    # Mark practical limits
    plt.axhline(y=1e9, color='gray', linestyle='--', alpha=0.5,
                label='~1 billion ops (practical limit)')

    plt.xlabel('Problem Size n', fontsize=12)
    plt.ylabel('Operations Required', fontsize=12)
    plt.title('NP Problems: Classical vs Quantum Brute Force', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('np_speedup.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Find crossover with practical limit
    classical_limit = np.log2(1e9)
    quantum_limit = 2 * np.log2(1e9)
    print(f"\nPractical problem sizes (≤ 1B operations):")
    print(f"  Classical: n ≤ {classical_limit:.0f}")
    print(f"  Quantum: n ≤ {quantum_limit:.0f}")

# Main execution
print("="*60)
print("Applications of Amplitude Amplification")
print("="*60)

# SAT example
print("\n1. SAT SOLVING EXAMPLE")
print("-"*50)

# Formula: (x₁ ∨ ¬x₂) ∧ (x₂ ∨ x₃) ∧ (¬x₁ ∨ ¬x₃)
formula = [[1, -2], [2, 3], [-1, -3]]
n_vars = 3

oracle, satisfying = sat_oracle(formula, n_vars)
print(f"Formula with {n_vars} variables")
print(f"Satisfying assignments ({len(satisfying)} found):")
for idx, assignment in satisfying:
    print(f"  |{idx:03b}⟩ = {assignment}")

result, iterations, prob = grover_for_sat(formula, n_vars)
print(f"\nGrover search: {iterations} iterations")
print(f"Success probability: {prob:.4f}")

# Monte Carlo comparison
print("\n2. MONTE CARLO COMPARISON")
print("-"*50)
monte_carlo_comparison()

# Durr-Hoyer simulation
print("\n3. MINIMUM FINDING (DURR-HOYER)")
print("-"*50)

for N in [100, 1000, 10000]:
    arr = np.random.rand(N)
    mean_queries, std_queries = durr_hoyer_simulation(arr, 100)
    expected_sqrt = np.sqrt(N)
    print(f"N = {N:>5}: Avg queries = {mean_queries:.1f} ± {std_queries:.1f}, "
          f"√N = {expected_sqrt:.1f}")

# NP speedup analysis
print("\n4. NP PROBLEM SPEEDUPS")
print("-"*50)
analyze_np_speedups()

# Application summary
print("\n5. APPLICATION SUMMARY")
print("-"*60)
print(f"{'Application':^25} | {'Classical':^15} | {'Quantum':^15} | {'Speedup':^10}")
print("-"*60)

applications = [
    ("SAT (n vars)", "O(2^n)", "O(2^{n/2})", "√2^n"),
    ("Monte Carlo", "O(1/ε²)", "O(1/ε)", "1/ε"),
    ("Minimum finding", "O(N)", "O(√N)", "√N"),
    ("Database search", "O(N)", "O(√N)", "√N"),
    ("Counting", "O(N)", "O(√MN)", "√N/M"),
    ("Element distinct.", "O(N)", "O(N^{2/3})", "N^{1/3}"),
]

for app, classical, quantum, speedup in applications:
    print(f"{app:^25} | {classical:^15} | {quantum:^15} | {speedup:^10}")

print("-"*60)
print("\nNote: All speedups are polynomial (quadratic or less), not exponential.")
print("Quantum computers do NOT solve NP-complete problems efficiently!")
```

---

## Summary

### Key Applications

| Application | Classical | Quantum | Speedup |
|-------------|-----------|---------|---------|
| SAT solving | $O(2^n)$ | $O(2^{n/2})$ | $\sqrt{2^n}$ |
| Monte Carlo | $O(1/\epsilon^2)$ | $O(1/\epsilon)$ | $1/\epsilon$ |
| Min/Max finding | $O(N)$ | $O(\sqrt{N})$ | $\sqrt{N}$ |
| Counting | $O(N)$ | $O(\sqrt{MN})$ | varies |

### Key Takeaways

1. **SAT solving** gets quadratic speedup (still exponential)
2. **Monte Carlo** gets quadratic speedup in precision
3. **Optimization** benefits from minimum finding
4. **Not magic:** NP-hard remains hard
5. **Practical limits:** Oracle construction, coherence
6. **Best applications:** Structured problems with quantum-friendly oracles

---

## Daily Checklist

- [ ] I can apply AA to SAT problems
- [ ] I understand quantum Monte Carlo speedup
- [ ] I know the Durr-Hoyer minimum algorithm
- [ ] I can evaluate practical quantum advantages
- [ ] I understand the limitations
- [ ] I ran the computational lab and explored applications

---

*Next: Day 630 — Week Review*
