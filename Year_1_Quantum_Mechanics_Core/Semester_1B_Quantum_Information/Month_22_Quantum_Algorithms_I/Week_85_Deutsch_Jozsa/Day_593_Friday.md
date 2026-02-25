# Day 593: Simon's Algorithm Introduction

## Overview

**Day 593** | Week 85, Day 5 | Month 22 | Quantum Algorithms I

Today we study Simon's algorithm - the most important precursor to Shor's factoring algorithm. Simon's algorithm demonstrates an **exponential** quantum speedup for the hidden subgroup problem, reducing query complexity from $O(2^{n/2})$ to $O(n)$. This algorithm directly inspired Shor to develop his revolutionary factoring algorithm.

---

## Learning Objectives

1. Define Simon's problem (hidden period/subgroup)
2. Prove the classical $\Omega(2^{n/2})$ lower bound
3. Derive the quantum algorithm's $O(n)$ query complexity
4. Understand the classical post-processing (linear algebra over $\mathbb{Z}_2$)
5. Recognize the connection to period-finding and Shor's algorithm
6. Implement Simon's algorithm computationally

---

## Core Content

### Simon's Problem

**Problem Statement:** Given oracle access to $f: \{0,1\}^n \to \{0,1\}^n$ with the promise that there exists a hidden string $s \in \{0,1\}^n$ such that:

$$f(x) = f(y) \Leftrightarrow y = x \text{ or } y = x \oplus s$$

Find $s$.

**Interpretation:**
- If $s = 0^n$: $f$ is one-to-one (injective)
- If $s \neq 0^n$: $f$ is two-to-one, with $f(x) = f(x \oplus s)$ for all $x$
- The string $s$ is the "period" under XOR

**Example (n = 3, s = 110):**

| x | f(x) |
|---|------|
| 000 | 011 |
| 001 | 010 |
| 010 | 100 |
| 011 | 101 |
| 100 | 101 |
| 101 | 100 |
| 110 | 011 |
| 111 | 010 |

Notice: $f(000) = f(110)$, $f(001) = f(111)$, etc. (pairs differ by $s = 110$)

### Classical Complexity

**Theorem:** Any classical algorithm requires $\Omega(2^{n/2})$ queries.

**Proof Sketch (Birthday Paradox Argument):**

To find $s$, we need to find a **collision**: two inputs $x, y$ with $f(x) = f(y)$.

By the birthday paradox, we need $\Theta(2^{n/2})$ random queries before expecting a collision.

Even a randomized algorithm cannot do better than $\Omega(2^{n/2})$ queries. $\square$

### Quantum Algorithm: Circuit

```
|0⟩^⊗n ─[H^⊗n]───●───[H^⊗n]─── Measure → y
                 │
|0⟩^⊗n ─────────⊕f──────────── (discarded)
                U_f
```

**Note:** The second register is n qubits (unlike Deutsch-Jozsa where it was 1).

### State Evolution

**Step 1: Initial State**
$$|\psi_0\rangle = |0\rangle^{\otimes n}|0\rangle^{\otimes n}$$

**Step 2: After Hadamard on First Register**
$$|\psi_1\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle|0\rangle^{\otimes n}$$

**Step 3: After Oracle**
$$|\psi_2\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle|f(x)\rangle$$

**Key Observation:** For $s \neq 0$, each value $f(x)$ appears exactly twice:
$$|\psi_2\rangle = \frac{1}{\sqrt{2^{n-1}}}\sum_{z \in \text{Image}(f)}\frac{1}{\sqrt{2}}(|x_z\rangle + |x_z \oplus s\rangle)|z\rangle$$

where $x_z$ is one preimage of $z$.

**Step 4: Measure Second Register (Conceptually)**

If we measured the second register and got $z$, the first register collapses to:
$$\frac{1}{\sqrt{2}}(|x_z\rangle + |x_z \oplus s\rangle)$$

But we don't actually measure - we just apply Hadamard!

**Step 5: After Hadamard on First Register**

For the state $\frac{1}{\sqrt{2}}(|x_z\rangle + |x_z \oplus s\rangle)$:

$$H^{\otimes n}\left[\frac{1}{\sqrt{2}}(|x_z\rangle + |x_z \oplus s\rangle)\right]$$

$$= \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2^n}}\sum_y\left[(-1)^{x_z \cdot y} + (-1)^{(x_z \oplus s) \cdot y}\right]|y\rangle$$

$$= \frac{1}{\sqrt{2^{n+1}}}\sum_y(-1)^{x_z \cdot y}\left[1 + (-1)^{s \cdot y}\right]|y\rangle$$

**Crucial Observation:**
$$1 + (-1)^{s \cdot y} = \begin{cases} 2 & \text{if } s \cdot y = 0 \\ 0 & \text{if } s \cdot y = 1 \end{cases}$$

**Result:** Only outcomes $y$ with $s \cdot y = 0$ have non-zero amplitude!

### Measurement Outcome

$$\boxed{\text{Measure } y \text{ such that } s \cdot y = 0 \pmod{2}}$$

Each measurement gives a linear constraint on $s$!

### Classical Post-Processing

**Goal:** Collect enough constraints to uniquely determine $s$.

**Approach:**
1. Run the quantum circuit $O(n)$ times
2. Collect outcomes $y_1, y_2, \ldots, y_m$
3. Solve the system: $s \cdot y_i = 0$ for all $i$
4. Find the non-trivial solution

**Linear Algebra over $\mathbb{Z}_2$:**

The constraints form a homogeneous system:
$$\begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{pmatrix} \cdot s = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}$$

We need $n-1$ linearly independent constraints to pin down $s$ (up to the trivial solution $s = 0$).

**Theorem:** With $O(n)$ queries, we get $n-1$ linearly independent constraints with high probability.

### Success Probability Analysis

Each query gives a uniformly random $y$ from $\{y : s \cdot y = 0\}$.

The probability that $m$ measurements give a rank-deficient system (rank < $n-1$) decreases exponentially with $m$.

After $n + O(1)$ queries:
$$\Pr[\text{success}] \geq 1 - 2^{-\Omega(1)} \geq 1 - \epsilon$$

### Comparison to Classical

| Algorithm | Query Complexity | Total Complexity |
|-----------|------------------|------------------|
| Classical (best) | $\Theta(2^{n/2})$ | $\Theta(2^{n/2})$ |
| Simon's (quantum) | $O(n)$ | $O(n^3)$ |

The quantum part has exponential speedup; classical post-processing is polynomial.

### Connection to Shor's Algorithm

Simon's algorithm finds the **period under XOR** (hidden subgroup of $\mathbb{Z}_2^n$).

Shor's algorithm finds the **period under addition** (hidden subgroup of $\mathbb{Z}_N$).

The key insight Shor borrowed: use **Fourier transform** to map period information to measurable outcomes, then use **classical post-processing** to recover the period.

---

## Worked Examples

### Example 1: n = 2, s = 11

Let $f(00) = f(11) = 00$ and $f(01) = f(10) = 01$.

**Solution:**

**After oracle:** (simplified analysis)

The superposition pairs up:
- $f^{-1}(00) = \{00, 11\}$
- $f^{-1}(01) = \{01, 10\}$

For each outcome, the first register has:
- $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$, or
- $\frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$

**After Hadamard:**

For $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

Applying $H^{\otimes 2}$:
- $H^{\otimes 2}|00\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$
- $H^{\otimes 2}|11\rangle = \frac{1}{2}(|00\rangle - |01\rangle - |10\rangle + |11\rangle)$

Sum: $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ (after normalization)

Check: $s \cdot 00 = 0$ ✓, $s \cdot 11 = 1 \oplus 1 = 0$ ✓

**Measurements:** Get either $|00\rangle$ or $|11\rangle$.

**Solving:** If we get $y = 00$: constraint $s_1 \cdot 0 + s_2 \cdot 0 = 0$ (trivial)

If we get $y = 11$: constraint $s_1 + s_2 = 0$, so $s_1 = s_2$.

Combined with non-triviality: $s = 11$.

### Example 2: n = 3, s = 101

**Measurement outcomes must satisfy:** $s \cdot y = y_1 \oplus y_3 = 0$

Valid outcomes: $\{000, 010, 101, 111\}$

**Three measurements:** Suppose we get $y_1 = 010$, $y_2 = 111$, $y_3 = 101$.

**System:**
- $y_1$: $0 \cdot s_1 + 1 \cdot s_2 + 0 \cdot s_3 = 0 \Rightarrow s_2 = 0$
- $y_2$: $1 \cdot s_1 + 1 \cdot s_2 + 1 \cdot s_3 = 0 \Rightarrow s_1 + s_2 + s_3 = 0$
- $y_3$: $1 \cdot s_1 + 0 \cdot s_2 + 1 \cdot s_3 = 0 \Rightarrow s_1 + s_3 = 0$

From $s_2 = 0$ and $s_1 + s_3 = 0$: $s_1 = s_3$.

Solutions: $s = 000$ (trivial) or $s = 101$ (non-trivial).

**Answer:** $s = 101$ ✓

### Example 3: Insufficient Constraints

With $n = 4$, suppose first two measurements give $y_1 = 0011$ and $y_2 = 0110$.

**Constraints:**
- $s_3 + s_4 = 0$
- $s_2 + s_3 = 0$

This gives $s_2 = s_3 = s_4$, but $s_1$ is undetermined.

**Need more measurements!** Third measurement might give $y_3 = 1010$:
- $s_1 + s_3 = 0 \Rightarrow s_1 = s_3$

Now: $s_1 = s_2 = s_3 = s_4$. Non-trivial solution: $s = 1111$ or variations.

---

## Practice Problems

### Problem 1: Small Example

For $n = 2$, $s = 10$, write out:
(a) A valid function $f$ satisfying Simon's promise
(b) The set of valid measurement outcomes
(c) The expected number of measurements to determine $s$

### Problem 2: Constraint Independence

Prove that if $y_1, \ldots, y_m$ are uniformly random from $\{y : s \cdot y = 0\}$ (a subspace of dimension $n-1$), then the expected number of measurements to get $n-1$ linearly independent vectors is $O(n)$.

### Problem 3: Failure Mode

What happens in Simon's algorithm if $s = 0^n$ (i.e., $f$ is one-to-one)? How would you detect this case?

### Problem 4: Period Under Addition

Consider a function $f: \mathbb{Z}_8 \to \mathbb{Z}_8$ with $f(x) = f(x + 4 \mod 8)$. This has period 4 under addition mod 8. Why can't Simon's algorithm (designed for XOR) solve this directly?

---

## Computational Lab

```python
"""Day 593: Simon's Algorithm Implementation"""
import numpy as np
from typing import Tuple, List, Optional

def create_simon_function(s: int, n: int) -> dict:
    """
    Create a function f: {0,1}^n -> {0,1}^n satisfying Simon's promise
    f(x) = f(y) iff y = x or y = x ⊕ s
    """
    if s == 0:
        # One-to-one function
        return {x: x for x in range(2**n)}

    # Two-to-one function
    f = {}
    used_values = set()
    next_value = 0

    for x in range(2**n):
        if x in f:
            continue

        y = x ^ s
        # Assign same value to both x and x ⊕ s
        while next_value in used_values:
            next_value += 1
        f[x] = next_value
        f[y] = next_value
        used_values.add(next_value)
        next_value += 1

    return f

def hadamard_n(n: int) -> np.ndarray:
    """n-qubit Hadamard transform"""
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    result = H1
    for _ in range(n - 1):
        result = np.kron(result, H1)
    return result

def simon_oracle(f: dict, n: int) -> np.ndarray:
    """
    Build oracle U_f|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
    Total dimension: 2^(2n)
    """
    dim = 2 ** (2 * n)
    U = np.zeros((dim, dim))

    for x in range(2**n):
        fx = f[x]
        for y in range(2**n):
            input_idx = (x << n) | y
            output_y = y ^ fx
            output_idx = (x << n) | output_y
            U[output_idx, input_idx] = 1

    return U

def simon_single_query(f: dict, n: int) -> int:
    """
    Execute one iteration of Simon's algorithm
    Returns: measurement outcome y (integer)
    """
    dim_half = 2 ** n
    dim_full = 2 ** (2 * n)

    # Initial state |0⟩^⊗n |0⟩^⊗n
    state = np.zeros(dim_full, dtype=complex)
    state[0] = 1

    # Apply H^⊗n ⊗ I to first register
    H_n = hadamard_n(n)
    I_n = np.eye(dim_half)
    H_first = np.kron(H_n, I_n)
    state = H_first @ state

    # Apply oracle
    U_f = simon_oracle(f, n)
    state = U_f @ state

    # Apply H^⊗n ⊗ I to first register again
    state = H_first @ state

    # Compute measurement probabilities for first register
    # Probability of measuring y in first register = Σ_z |⟨yz|ψ⟩|²
    probs = np.zeros(dim_half)
    for y in range(dim_half):
        for z in range(dim_half):
            idx = (y << n) | z
            probs[y] += abs(state[idx])**2

    # Sample from distribution
    y = np.random.choice(dim_half, p=probs / probs.sum())
    return y

def inner_product_z2(x: int, y: int, n: int) -> int:
    """Compute x · y mod 2"""
    return bin(x & y).count('1') % 2

def gaussian_elimination_z2(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Gaussian elimination over Z_2
    Returns: (row echelon form, rank)
    """
    A = matrix.copy().astype(int) % 2
    rows, cols = A.shape
    pivot_row = 0

    for col in range(cols):
        # Find pivot
        found = False
        for row in range(pivot_row, rows):
            if A[row, col] == 1:
                # Swap rows
                A[[pivot_row, row]] = A[[row, pivot_row]]
                found = True
                break

        if not found:
            continue

        # Eliminate below
        for row in range(pivot_row + 1, rows):
            if A[row, col] == 1:
                A[row] = (A[row] + A[pivot_row]) % 2

        pivot_row += 1

    rank = pivot_row
    return A, rank

def solve_simon_system(measurements: List[int], n: int) -> Optional[int]:
    """
    Solve the system y_i · s = 0 for all measured y_i
    Returns: non-trivial solution s, or None if only trivial solution
    """
    if not measurements:
        return None

    # Build matrix where each row is a measurement
    m = len(measurements)
    matrix = np.zeros((m, n), dtype=int)
    for i, y in enumerate(measurements):
        for j in range(n):
            matrix[i, j] = (y >> (n - 1 - j)) & 1

    # Gaussian elimination
    ref, rank = gaussian_elimination_z2(matrix)

    if rank >= n:
        # Only trivial solution (shouldn't happen with valid Simon function)
        return 0

    # Find solution in null space
    # Start with s having 1 in the first free variable position
    # then back-substitute

    # Find free variables (columns without pivot)
    pivot_cols = []
    row = 0
    for col in range(n):
        if row < rank and ref[row, col] == 1:
            pivot_cols.append(col)
            row += 1

    free_cols = [c for c in range(n) if c not in pivot_cols]

    if not free_cols:
        return 0

    # Set first free variable to 1, others to 0, back-substitute
    s_bits = [0] * n
    s_bits[free_cols[0]] = 1

    # Back-substitution
    for row in range(rank - 1, -1, -1):
        # Find pivot column for this row
        pivot_col = pivot_cols[row]
        # s[pivot_col] = sum of s[j] for j > pivot_col where ref[row,j] = 1
        total = 0
        for col in range(pivot_col + 1, n):
            if ref[row, col] == 1:
                total ^= s_bits[col]
        s_bits[pivot_col] = total

    # Convert to integer
    s = 0
    for i, bit in enumerate(s_bits):
        s = (s << 1) | bit

    return s

def simon_algorithm(s_true: int, n: int, verbose: bool = True) -> Tuple[int, int]:
    """
    Full Simon's algorithm
    Returns: (recovered s, number of queries)
    """
    f = create_simon_function(s_true, n)

    if verbose:
        print(f"Hidden string s = {s_true:0{n}b}")
        print(f"Function f (two-to-one with period s):")
        for x in sorted(f.keys())[:min(8, 2**n)]:
            print(f"  f({x:0{n}b}) = {f[x]:0{n}b}")
        if 2**n > 8:
            print("  ...")

    measurements = []
    queries = 0
    max_queries = 10 * n  # Safety limit

    while queries < max_queries:
        y = simon_single_query(f, n)
        queries += 1
        measurements.append(y)

        if verbose and queries <= 5:
            print(f"Query {queries}: measured y = {y:0{n}b}, "
                  f"s·y = {inner_product_z2(s_true, y, n)}")

        # Try to solve
        s_recovered = solve_simon_system(measurements, n)

        if s_recovered is not None and s_recovered != 0:
            # Verify
            if f.get(0) == f.get(s_recovered):
                if verbose:
                    print(f"\nRecovered s = {s_recovered:0{n}b} after {queries} queries")
                return s_recovered, queries

    # Fallback: s = 0 (one-to-one function)
    if verbose:
        print(f"\nNo period found after {queries} queries (s = 0?)")
    return 0, queries

def test_simon():
    """Test Simon's algorithm on various cases"""
    print("=" * 60)
    print("SIMON'S ALGORITHM TESTS")
    print("=" * 60)

    test_cases = [
        (3, 0b101),
        (3, 0b110),
        (4, 0b1011),
        (4, 0b0110),
    ]

    for n, s in test_cases:
        print(f"\n{'='*40}")
        print(f"n = {n}, s = {s:0{n}b}")
        print(f"{'='*40}")

        s_found, queries = simon_algorithm(s, n, verbose=True)

        if s_found == s:
            print(f"SUCCESS! Found s = {s_found:0{n}b} in {queries} queries")
        else:
            print(f"FAILED! Expected {s:0{n}b}, got {s_found:0{n}b}")

def complexity_analysis():
    """Analyze query complexity empirically"""
    print("\n" + "=" * 60)
    print("QUERY COMPLEXITY ANALYSIS")
    print("=" * 60)

    trials = 50

    print("\n| n | Avg Queries | Classical Lower Bound |")
    print("|---|-------------|----------------------|")

    for n in range(3, 8):
        total_queries = 0
        for _ in range(trials):
            s = np.random.randint(1, 2**n)  # Non-zero s
            _, queries = simon_algorithm(s, n, verbose=False)
            total_queries += queries

        avg_queries = total_queries / trials
        classical_bound = 2 ** (n / 2)

        print(f"| {n} | {avg_queries:.1f}         | {classical_bound:.1f}                  |")

    print("\nQuantum: O(n) queries")
    print("Classical: O(2^(n/2)) queries")
    print("Exponential speedup confirmed!")

def demonstrate_constraint_buildup():
    """Show how constraints accumulate"""
    print("\n" + "=" * 60)
    print("CONSTRAINT ACCUMULATION DEMO")
    print("=" * 60)

    n = 4
    s = 0b1010

    print(f"\nHidden string s = {s:0{n}b}")
    print(f"Valid measurements y satisfy: s · y = 0")
    print(f"Valid y's: ", end="")

    valid_ys = [y for y in range(2**n) if inner_product_z2(s, y, n) == 0]
    print([f"{y:0{n}b}" for y in valid_ys[:8]])

    print("\nSimulating measurements:")
    measurements = []
    f = create_simon_function(s, n)

    for i in range(6):
        y = simon_single_query(f, n)
        measurements.append(y)

        # Build matrix
        matrix = np.array([[((m >> (n-1-j)) & 1) for j in range(n)]
                          for m in measurements])
        _, rank = gaussian_elimination_z2(matrix)

        print(f"  Query {i+1}: y = {y:0{n}b}, "
              f"matrix rank = {rank}/{n-1} needed")

        if rank >= n - 1:
            print(f"  → Sufficient constraints! Can solve for s.")
            break

# Run all tests
test_simon()
complexity_analysis()
demonstrate_constraint_buildup()

# Visualization
print("\n" + "=" * 60)
print("GENERATING COMPARISON VISUALIZATION")
print("=" * 60)

import matplotlib.pyplot as plt

def plot_complexity_comparison():
    """Plot classical vs quantum complexity"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_values = np.arange(2, 21)
    classical = 2 ** (n_values / 2)
    quantum = 2 * n_values  # O(n) with constant factor

    ax.semilogy(n_values, classical, 'r-', linewidth=2, label='Classical: O(2^{n/2})')
    ax.semilogy(n_values, quantum, 'b-', linewidth=2, label='Quantum: O(n)')

    ax.set_xlabel('Number of bits n', fontsize=12)
    ax.set_ylabel('Number of queries', fontsize=12)
    ax.set_title("Simon's Algorithm: Query Complexity", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate specific points
    ax.annotate(f'n=20: Classical ≈ {2**10:.0f}', xy=(20, 2**10),
                xytext=(15, 2**12), arrowprops=dict(arrowstyle='->'),
                fontsize=10)
    ax.annotate(f'n=20: Quantum ≈ 40', xy=(20, 40),
                xytext=(15, 200), arrowprops=dict(arrowstyle='->'),
                fontsize=10)

    plt.tight_layout()
    plt.savefig('simon_complexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Complexity comparison saved to 'simon_complexity.png'")

plot_complexity_comparison()
```

**Expected Output:**
```
============================================================
SIMON'S ALGORITHM TESTS
============================================================

========================================
n = 3, s = 101
========================================
Hidden string s = 101
Function f (two-to-one with period s):
  f(000) = 000
  f(001) = 001
  f(010) = 010
  f(011) = 011
  f(100) = 001
  f(101) = 000
  f(110) = 011
  f(111) = 010
Query 1: measured y = 010, s·y = 0
Query 2: measured y = 000, s·y = 0
Query 3: measured y = 111, s·y = 0

Recovered s = 101 after 3 queries
SUCCESS! Found s = 101 in 3 queries
...
```

---

## Summary

### Key Formulas

| Expression | Formula |
|------------|---------|
| Simon's promise | $f(x) = f(y) \Leftrightarrow y \in \{x, x \oplus s\}$ |
| Measurement outcome | $y$ such that $s \cdot y = 0$ |
| Classical complexity | $\Omega(2^{n/2})$ |
| Quantum complexity | $O(n)$ queries, $O(n^3)$ total |

### Key Takeaways

1. **Exponential speedup**: First algorithm with provable exponential quantum advantage
2. **Hidden subgroup structure**: Period under XOR is a subgroup of $\mathbb{Z}_2^n$
3. **Fourier sampling**: Hadamard projects onto orthogonal subspace to period
4. **Classical post-processing**: Linear algebra over $\mathbb{Z}_2$ to solve for $s$
5. **Shor's inspiration**: Same structure with QFT replacing Hadamard for $\mathbb{Z}_N$

---

## Daily Checklist

- [ ] I can state Simon's problem and the hidden period structure
- [ ] I understand the birthday paradox argument for classical lower bound
- [ ] I can explain why measurements give $s \cdot y = 0$
- [ ] I understand the linear algebra post-processing
- [ ] I see the connection to Shor's period-finding
- [ ] I ran the lab and verified exponential query reduction

---

*Next: Day 594 - Quantum Oracle Construction*
