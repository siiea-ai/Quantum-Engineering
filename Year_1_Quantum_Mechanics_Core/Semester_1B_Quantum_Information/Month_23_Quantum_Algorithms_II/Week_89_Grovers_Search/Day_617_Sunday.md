# Day 617: The Unstructured Search Problem

## Overview
**Day 617** | Week 89, Day 1 | Year 1, Month 23 | Grover's Search Algorithm

Today we introduce the unstructured search problem—finding a marked item in an unsorted database. We establish the classical lower bound and motivate the quantum approach that will achieve a quadratic speedup.

---

## Learning Objectives

1. Define the unstructured search problem formally
2. Prove the classical lower bound of O(N) queries
3. Understand the oracle model of computation
4. State Grover's speedup to O(sqrt(N)) queries
5. Appreciate the significance of quadratic speedup
6. Set up the quantum search framework

---

## Core Content

### The Unstructured Search Problem

**Problem Statement:** Given a function $f: \{0,1,...,N-1\} \to \{0,1\}$ with exactly one marked element $w$ such that $f(w) = 1$ and $f(x) = 0$ for all $x \neq w$, find $w$.

**Classical Setting:**
- We have a "black box" (oracle) that evaluates $f$
- Each oracle query costs one unit of time
- The database is unsorted—no structure to exploit

### Classical Lower Bound

**Theorem:** Any classical algorithm requires $\Omega(N)$ oracle queries to find the marked element with high probability.

**Proof Sketch:**
Consider any deterministic algorithm. Before querying, all N elements are equally likely to be marked. After $k$ queries, we have eliminated at most $k$ possibilities.

- Probability of success after $k$ queries: $\leq k/N$
- For constant success probability: need $k = \Omega(N)$

Even randomized algorithms cannot do better on average:

$$\mathbb{E}[\text{queries}] \geq \frac{N}{2}$$

### The Oracle Model

In quantum computing, we represent the oracle as a unitary operator:

**Classical Oracle (Query):**
$$O_f: |x\rangle|q\rangle \mapsto |x\rangle|q \oplus f(x)\rangle$$

**Phase Oracle (More Common):**
$$O_f: |x\rangle \mapsto (-1)^{f(x)}|x\rangle$$

The phase oracle can be constructed from the classical oracle:

$$O_{phase} = (I \otimes H) O_{classical} (I \otimes H)$$

with ancilla in state $|{-}\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$.

### Quantum Search Setup

**Initial State:**
$$|\psi_0\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle$$

where $N = 2^n$.

**Key Insight:** The initial state has equal amplitude $1/\sqrt{N}$ on all basis states, including the marked state $|w\rangle$.

**Goal:** Amplify the amplitude of $|w\rangle$ so that measurement yields $w$ with high probability.

### Two-Dimensional Subspace

Define:
- **Target state:** $|w\rangle$ (the marked element)
- **Orthogonal complement:** $|s'\rangle = \frac{1}{\sqrt{N-1}}\sum_{x \neq w}|x\rangle$

The initial uniform superposition lies in this 2D subspace:

$$|\psi_0\rangle = \sin\theta|w\rangle + \cos\theta|s'\rangle$$

where:
$$\sin\theta = \frac{1}{\sqrt{N}}, \quad \cos\theta = \sqrt{\frac{N-1}{N}}$$

For large $N$: $\theta \approx 1/\sqrt{N}$ (small angle).

### Grover's Promise

**Grover's Algorithm (1996):** Finds the marked element with high probability using only:

$$\boxed{O\left(\sqrt{N}\right) \text{ oracle queries}}$$

This is a **quadratic speedup** over classical algorithms!

### Significance of Quadratic Speedup

| N | Classical Queries | Quantum Queries |
|---|-------------------|-----------------|
| 100 | 50 (avg) | ~8 |
| 10,000 | 5,000 (avg) | ~79 |
| 1,000,000 | 500,000 (avg) | ~785 |
| $10^{12}$ | $5 \times 10^{11}$ | ~$10^6$ |

For cryptographic applications (breaking symmetric keys):
- AES-128: Classical brute force needs $2^{128}$ operations
- Grover's algorithm: $2^{64}$ operations (effectively halves key length)

### Lower Bound for Quantum Search

**Theorem (BBBV 1997):** Any quantum algorithm for unstructured search requires $\Omega(\sqrt{N})$ queries.

Grover's algorithm is **optimal** for this problem!

---

## Worked Examples

### Example 1: Search Space Size
A database has $N = 2^{20} \approx 10^6$ items. Compare classical and quantum search.

**Solution:**
- Classical: Expected queries $= N/2 = 524,288$
- Quantum: Grover queries $\approx \frac{\pi}{4}\sqrt{N} = \frac{\pi}{4} \times 1024 \approx 804$

Speedup factor: $\frac{524,288}{804} \approx 652\times$ faster

### Example 2: Initial Amplitude
For $N = 4$ (2 qubits), compute the initial amplitudes and angle $\theta$.

**Solution:**
Initial state: $|\psi_0\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$

If $|w\rangle = |11\rangle$ is marked:
- Amplitude of $|w\rangle$: $1/2$
- Amplitude of each non-marked state: $1/2$

Angle: $\sin\theta = 1/\sqrt{4} = 1/2$, so $\theta = \pi/6 = 30°$

### Example 3: Probability Before Amplification
What is the probability of finding the marked item by measuring the initial state?

**Solution:**
$$P_{success} = |\langle w|\psi_0\rangle|^2 = \frac{1}{N}$$

For $N = 10^6$: $P_{success} = 10^{-6}$ (essentially zero)

This motivates the need for amplitude amplification!

---

## Practice Problems

### Problem 1: Query Complexity
A classical algorithm solves an unstructured search problem in $T$ queries on average. What is the expected number of Grover iterations needed?

### Problem 2: Angle Calculation
For a search space of $N = 64$ items with one marked element:
a) Calculate $\sin\theta$ and $\theta$
b) Express $|\psi_0\rangle$ in terms of $|w\rangle$ and $|s'\rangle$

### Problem 3: Phase Oracle Construction
Given a classical oracle $O_f|x\rangle|b\rangle = |x\rangle|b \oplus f(x)\rangle$, show how to construct the phase oracle using an ancilla qubit.

---

## Computational Lab

```python
"""Day 617: Unstructured Search Problem Analysis"""
import numpy as np
import matplotlib.pyplot as plt

def classical_search_simulation(N, num_trials=1000):
    """Simulate classical random search"""
    queries_list = []
    for _ in range(num_trials):
        marked = np.random.randint(N)
        queries = 0
        found = False
        indices = np.random.permutation(N)
        for idx in indices:
            queries += 1
            if idx == marked:
                found = True
                break
        queries_list.append(queries)
    return np.mean(queries_list), np.std(queries_list)

def compare_complexities(N_values):
    """Compare classical and quantum query complexities"""
    classical = []
    quantum = []

    for N in N_values:
        # Classical: expected N/2
        classical.append(N / 2)
        # Quantum: approximately pi/4 * sqrt(N)
        quantum.append(np.pi / 4 * np.sqrt(N))

    return classical, quantum

def initial_state_analysis(n):
    """Analyze the initial uniform superposition for n qubits"""
    N = 2**n

    # Create uniform superposition
    psi_0 = np.ones(N) / np.sqrt(N)

    # Assume last state is marked
    marked_idx = N - 1

    # Amplitude of marked state
    amp_marked = psi_0[marked_idx]

    # Calculate theta
    sin_theta = 1 / np.sqrt(N)
    theta = np.arcsin(sin_theta)

    print(f"n = {n} qubits, N = {N} states")
    print(f"Amplitude of marked state: {amp_marked:.6f}")
    print(f"Probability without amplification: {amp_marked**2:.6f}")
    print(f"sin(theta) = {sin_theta:.6f}")
    print(f"theta = {np.degrees(theta):.4f} degrees")
    print(f"Optimal Grover iterations: {int(np.pi/4 * np.sqrt(N))}")
    print()

    return theta

# Simulate classical search
print("Classical Search Simulation:")
print("-" * 40)
for N in [100, 1000, 10000]:
    mean_queries, std_queries = classical_search_simulation(N, 1000)
    print(f"N = {N}: Mean queries = {mean_queries:.1f} +/- {std_queries:.1f}")
    print(f"  Theoretical mean = {N/2}")
print()

# Compare complexities
print("Query Complexity Comparison:")
print("-" * 40)
N_values = [2**k for k in range(4, 21)]
classical, quantum = compare_complexities(N_values)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(N_values, classical, 'b-', label='Classical O(N)', linewidth=2)
plt.loglog(N_values, quantum, 'r-', label='Quantum O(√N)', linewidth=2)
plt.xlabel('Database Size N')
plt.ylabel('Number of Queries')
plt.title('Query Complexity: Classical vs Quantum')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
speedup = np.array(classical) / np.array(quantum)
plt.semilogx(N_values, speedup, 'g-', linewidth=2)
plt.xlabel('Database Size N')
plt.ylabel('Speedup Factor')
plt.title('Quantum Speedup = √N / (π/4)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grover_complexity.png', dpi=150, bbox_inches='tight')
plt.show()

# Initial state analysis
print("\nInitial State Analysis:")
print("-" * 40)
for n in [2, 4, 6, 8, 10]:
    initial_state_analysis(n)
```

**Expected Output:**
```
Classical Search Simulation:
----------------------------------------
N = 100: Mean queries = 50.2 +/- 28.9
  Theoretical mean = 50
N = 1000: Mean queries = 499.1 +/- 289.2
  Theoretical mean = 500
N = 10000: Mean queries = 5012.3 +/- 2887.1
  Theoretical mean = 5000

Initial State Analysis:
----------------------------------------
n = 2 qubits, N = 4 states
Amplitude of marked state: 0.500000
Probability without amplification: 0.250000
sin(theta) = 0.500000
theta = 30.0000 degrees
Optimal Grover iterations: 1

n = 4 qubits, N = 16 states
Amplitude of marked state: 0.250000
Probability without amplification: 0.062500
sin(theta) = 0.250000
theta = 14.4775 degrees
Optimal Grover iterations: 3
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Classical lower bound | $\Omega(N)$ queries |
| Quantum complexity | $O(\sqrt{N})$ queries |
| Initial probability | $P = 1/N$ |
| Initial angle | $\sin\theta = 1/\sqrt{N}$ |
| Phase oracle | $O_f\|x\rangle = (-1)^{f(x)}\|x\rangle$ |

### Key Takeaways

1. **Unstructured search** has no structure to exploit classically
2. **Classical lower bound** is $\Omega(N)$ queries
3. **Grover's algorithm** achieves $O(\sqrt{N})$—a quadratic speedup
4. **This is provably optimal** for quantum computers
5. **Two-dimensional subspace** spanned by $|w\rangle$ and $|s'\rangle$
6. **Initial state** has small overlap with marked state

---

## Daily Checklist

- [ ] I can state the unstructured search problem formally
- [ ] I understand why classical algorithms need O(N) queries
- [ ] I can explain the phase oracle model
- [ ] I understand the two-dimensional subspace formulation
- [ ] I can calculate the initial angle theta
- [ ] I ran the computational lab and understood the outputs

---

*Next: Day 618 — The Grover Oracle*
