# Day 752: Triorthogonal Codes

## Overview

**Day:** 752 of 1008
**Week:** 108 (Code Families & Construction Techniques)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Triorthogonal Codes for Magic State Distillation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Triorthogonality theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Distillation protocols |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Define** the triorthogonality condition for binary matrices
2. **Verify** whether a code is triorthogonal
3. **Connect** triorthogonality to T-gate distillation
4. **Design** triorthogonal codes for specific applications
5. **Analyze** distillation overhead and error reduction
6. **Compare** different distillation protocols

---

## Triorthogonality Definition

### The Condition

A binary matrix G is **triorthogonal** if:

$$\boxed{\sum_{j=1}^n G_{aj} G_{bj} G_{cj} = 0 \pmod 2 \quad \text{for all } a, b, c}$$

This means every triple product of rows sums to 0 (mod 2).

### Interpretation

- **Single row:** $\sum_j G_{aj}^3 = \sum_j G_{aj} = wt(row_a) \equiv 0 \pmod 2$
- **Pair of rows:** $\sum_j G_{aj}^2 G_{bj} = \sum_j G_{aj} G_{bj} = |supp(a) \cap supp(b)| \equiv 0 \pmod 2$
- **Triple:** $\sum_j G_{aj} G_{bj} G_{cj} = |supp(a) \cap supp(b) \cap supp(c)| \equiv 0 \pmod 2$

### Hierarchy

**Doubly-even:** All row weights ≡ 0 mod 4
**Self-orthogonal:** All pairwise overlaps ≡ 0 mod 2
**Triorthogonal:** All triple overlaps ≡ 0 mod 2

$$\text{Doubly-even} \Rightarrow \text{Self-orthogonal} \Rightarrow \text{Triorthogonal}$$

---

## Connection to Magic States

### T Gate and Phase

The T gate: $T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$

Under $T^{\otimes n}$, a stabilizer $X^{\mathbf{g}}$ transforms as:
$$T^{\otimes n} X^{\mathbf{g}} (T^\dagger)^{\otimes n} = e^{i\phi(\mathbf{g})} X^{\mathbf{g}} Z^{\mathbf{g}}$$

where the phase $\phi(\mathbf{g})$ depends on the weight pattern.

### Phase Consistency

For $T^{\otimes n}$ to preserve the stabilizer group (up to stabilizer elements):
- The phases must be consistent
- The $Z^{\mathbf{g}}$ factors must combine to stabilizer elements

**Triorthogonality** ensures phase consistency!

### Magic State Injection

With triorthogonal codes:
1. Encode noisy T states
2. Measure stabilizers (Clifford operations)
3. If all pass, decode to get improved T state
4. Phase errors are detected by syndrome

---

## Triorthogonal CSS Codes

### CSS with Triorthogonality

A CSS code with X generators G (rows) is **triorthogonal** if G satisfies the triorthogonality condition.

**Code structure:**
- X stabilizers: rows of G
- Z stabilizers: derived from dual structure
- Logical operators: outside stabilizer group

### Parameters

For triorthogonal [[n, k, d]] code:
- n physical qubits
- k logical qubits (often k = 1 for distillation)
- d ≥ 3 for error detection

### The 15-to-1 Protocol

Using [[15, 1, 3]] triorthogonal code:

**Input:** 15 noisy |T⟩ states with error ε
**Process:**
1. Encode into [[15, 1, 3]]
2. Measure all stabilizers
3. Accept if all measurements = +1
**Output:** 1 purified |T⟩ with error O(ε³)

**Yield:** 15 → 1 (significant overhead)

---

## Constructing Triorthogonal Codes

### Method 1: Punctured Reed-Muller

The [[15, 1, 3]] code from punctured RM(1, 4):

**Generator (X stabilizers):**
$$G = \begin{pmatrix}
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 \\
1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1
\end{pmatrix}$$

**Verify triorthogonality:** Check all $\binom{4}{3} = 4$ triples of rows.

### Method 2: Direct Construction

Design G with:
1. All row weights ≡ 0 mod 2
2. All pairwise overlaps ≡ 0 mod 2
3. All triple overlaps ≡ 0 mod 2

### Method 3: Algebraic Construction

Use algebraic structures (e.g., from BCH codes) that naturally satisfy triorthogonality.

---

## Distillation Protocols

### 15-to-1 Protocol

**Code:** [[15, 1, 3]]
**Error reduction:** ε → 35ε³
**Overhead:** 15× per round

**Multiple rounds:**
- Round 1: ε → 35ε³
- Round 2: 35ε³ → 35(35ε³)³ = 35⁴ε⁹
- Exponential improvement!

### 7-to-1 Protocol (Steane)

The Steane code can also distill, but with different error behavior.

**Code:** [[7, 1, 3]]
**Error reduction:** ε → 7ε²
**Overhead:** 7× per round

Less aggressive improvement but lower overhead.

### Comparison

| Protocol | Code | Reduction | Overhead |
|----------|------|-----------|----------|
| 15-to-1 | [[15,1,3]] RM | ε³ | 15 |
| 7-to-1 | [[7,1,3]] Steane | ε² | 7 |

For very noisy initial states, 15-to-1 is more efficient.

---

## Worked Examples

### Example 1: Verify Triorthogonality

**Problem:** Check if the following matrix is triorthogonal:
$$G = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0
\end{pmatrix}$$

**Solution:**

Check all overlaps:

**Row weights:**
- Row 1: 4 ≡ 0 mod 2 ✓
- Row 2: 4 ≡ 0 mod 2 ✓
- Row 3: 3 ≡ 1 mod 2 ✗

Row 3 has odd weight, so NOT triorthogonal.

### Example 2: Triple Overlap Calculation

**Problem:** For the [[15, 1, 3]] code generators, compute the triple overlap of rows 1, 2, 3.

**Solution:**

Row 1: {0,1,2,3,4,5,6,7}
Row 2: {0,1,2,3,8,9,10,11}
Row 3: {0,1,4,5,8,9,12,13}

Triple intersection: {0,1}

$|supp(1) \cap supp(2) \cap supp(3)| = 2 \equiv 0 \pmod 2$ ✓

### Example 3: Distillation Efficiency

**Problem:** Starting with ε = 0.1 error rate, how many rounds of 15-to-1 distillation are needed to reach ε < 10⁻¹⁵?

**Solution:**

Round 0: ε₀ = 0.1
Round 1: ε₁ = 35 × (0.1)³ = 0.035
Round 2: ε₂ = 35 × (0.035)³ ≈ 1.5 × 10⁻³
Round 3: ε₃ = 35 × (1.5 × 10⁻³)³ ≈ 1.2 × 10⁻⁷
Round 4: ε₄ = 35 × (1.2 × 10⁻⁷)³ ≈ 6 × 10⁻²⁰

**Answer:** 4 rounds (actually 3 is sufficient for < 10⁻¹⁵)

---

## Practice Problems

### Level 1: Direct Application

**P1.1** Verify that the 4×8 identity matrix is NOT triorthogonal.

**P1.2** For a matrix G where all rows have weight 4, what additional condition is needed for triorthogonality?

**P1.3** If a [[15, 1, 3]] code reduces error as ε → 35ε³, find output error when ε = 0.05.

### Level 2: Intermediate

**P2.1** Prove that a doubly-even code is always triorthogonal.

**P2.2** Design a 3×6 triorthogonal matrix.

**P2.3** Compare total qubit overhead for reaching 10⁻¹⁰ error using:
a) 15-to-1 protocol starting from ε = 0.1
b) 7-to-1 protocol starting from ε = 0.1

### Level 3: Challenging

**P3.1** Prove that the [[15, 1, 3]] code is triorthogonal using its Reed-Muller structure.

**P3.2** Design a triorthogonal code with parameters [[n, 1, 5]] for n minimal.

**P3.3** Analyze the threshold behavior: what initial error rate ε makes distillation impossible?

---

## Computational Lab

```python
"""
Day 752: Triorthogonal Codes
============================

Implementing triorthogonality verification and distillation analysis.
"""

import numpy as np
from typing import Tuple, List, Optional
from itertools import combinations


def is_triorthogonal(G: np.ndarray) -> Tuple[bool, Optional[Tuple]]:
    """
    Check if matrix G is triorthogonal.

    Returns (True, None) if triorthogonal.
    Returns (False, (a, b, c)) with first failing triple.
    """
    G = np.array(G) % 2
    k, n = G.shape

    # Check all triples (including repeated indices)
    for a in range(k):
        for b in range(a, k):
            for c in range(b, k):
                triple_sum = np.sum(G[a] * G[b] * G[c]) % 2
                if triple_sum != 0:
                    return False, (a, b, c)

    return True, None


def check_all_conditions(G: np.ndarray) -> dict:
    """
    Check all orthogonality conditions for matrix G.

    Returns dict with:
    - even_weights: all row weights even
    - self_orthogonal: all pairwise overlaps even
    - triorthogonal: all triple overlaps even
    """
    G = np.array(G) % 2
    k, n = G.shape

    results = {
        'even_weights': True,
        'self_orthogonal': True,
        'triorthogonal': True,
        'failing_conditions': []
    }

    # Check row weights
    for i in range(k):
        weight = np.sum(G[i])
        if weight % 2 != 0:
            results['even_weights'] = False
            results['failing_conditions'].append(f"Row {i} has odd weight {weight}")

    # Check pairwise overlaps
    for i in range(k):
        for j in range(i, k):
            overlap = np.sum(G[i] * G[j]) % 2
            if overlap != 0:
                results['self_orthogonal'] = False
                results['failing_conditions'].append(f"Rows {i},{j} have odd overlap")

    # Check triple overlaps
    is_tri, failing = is_triorthogonal(G)
    if not is_tri:
        results['triorthogonal'] = False
        results['failing_conditions'].append(f"Triple {failing} has odd overlap")

    return results


def distillation_error(initial_error: float, rounds: int,
                       protocol: str = '15-to-1') -> List[float]:
    """
    Compute error after each round of distillation.

    Parameters:
    -----------
    initial_error : float
        Starting error rate
    rounds : int
        Number of distillation rounds
    protocol : str
        '15-to-1' or '7-to-1'

    Returns:
    --------
    List of error rates after each round
    """
    errors = [initial_error]
    e = initial_error

    for _ in range(rounds):
        if protocol == '15-to-1':
            e = 35 * (e ** 3)
        elif protocol == '7-to-1':
            e = 7 * (e ** 2)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        errors.append(e)
        if e < 1e-50:
            break

    return errors


def distillation_overhead(initial_error: float, target_error: float,
                          protocol: str = '15-to-1') -> Tuple[int, int]:
    """
    Compute rounds and total qubits needed for distillation.

    Returns (rounds, total_input_qubits).
    """
    if protocol == '15-to-1':
        multiplier = 15
    elif protocol == '7-to-1':
        multiplier = 7
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    e = initial_error
    rounds = 0
    total_qubits = 1

    while e > target_error and rounds < 20:
        if protocol == '15-to-1':
            e = 35 * (e ** 3)
        else:
            e = 7 * (e ** 2)

        total_qubits *= multiplier
        rounds += 1

    return rounds, total_qubits


def rm_15_1_3_generator() -> np.ndarray:
    """
    Generator matrix for [[15, 1, 3]] triorthogonal code.

    Based on punctured RM(1, 4).
    """
    G = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    ])
    return G


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 752: Triorthogonal Codes")
    print("=" * 60)

    # Example 1: [[15, 1, 3]] triorthogonality
    print("\n1. [[15, 1, 3]] Triorthogonality Check")
    print("-" * 40)

    G_15 = rm_15_1_3_generator()
    print("Generator matrix (4 × 15):")
    print(G_15)

    results = check_all_conditions(G_15)
    print(f"\nEven weights: {results['even_weights']}")
    print(f"Self-orthogonal: {results['self_orthogonal']}")
    print(f"Triorthogonal: {results['triorthogonal']}")

    # Example 2: Non-triorthogonal example
    print("\n2. Non-Triorthogonal Example")
    print("-" * 40)

    G_bad = np.array([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0]
    ])
    print("Matrix:")
    print(G_bad)

    results = check_all_conditions(G_bad)
    print(f"\nEven weights: {results['even_weights']}")
    print(f"Self-orthogonal: {results['self_orthogonal']}")
    print(f"Triorthogonal: {results['triorthogonal']}")
    if results['failing_conditions']:
        print(f"Failures: {results['failing_conditions']}")

    # Example 3: Distillation error progression
    print("\n3. Distillation Error Progression")
    print("-" * 40)

    initial = 0.1
    print(f"Initial error: {initial}")

    print("\n15-to-1 protocol:")
    errors_15 = distillation_error(initial, 5, '15-to-1')
    for i, e in enumerate(errors_15):
        print(f"  Round {i}: {e:.2e}")

    print("\n7-to-1 protocol:")
    errors_7 = distillation_error(initial, 8, '7-to-1')
    for i, e in enumerate(errors_7):
        print(f"  Round {i}: {e:.2e}")

    # Example 4: Overhead comparison
    print("\n4. Overhead to Reach Target Error")
    print("-" * 40)

    target = 1e-10
    print(f"Target error: {target:.0e}")
    print(f"Initial error: 0.1")

    rounds_15, qubits_15 = distillation_overhead(0.1, target, '15-to-1')
    rounds_7, qubits_7 = distillation_overhead(0.1, target, '7-to-1')

    print(f"\n15-to-1: {rounds_15} rounds, {qubits_15} total input qubits")
    print(f"7-to-1: {rounds_7} rounds, {qubits_7} total input qubits")

    # Example 5: Verify specific triples
    print("\n5. Triple Overlap Verification")
    print("-" * 40)

    G = rm_15_1_3_generator()
    for a, b, c in combinations(range(4), 3):
        overlap = np.sum(G[a] * G[b] * G[c]) % 2
        print(f"Triple ({a},{b},{c}): overlap = {overlap}")

    print("\n" + "=" * 60)
    print("Triorthogonality: the key to magic state distillation!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Triorthogonality | $\sum_j G_{aj} G_{bj} G_{cj} \equiv 0 \pmod 2$ |
| 15-to-1 error | $\epsilon_{out} = 35\epsilon_{in}^3$ |
| 7-to-1 error | $\epsilon_{out} = 7\epsilon_{in}^2$ |
| Rounds needed | $\epsilon_r \approx c^{(3^r-1)/2} \epsilon_0^{3^r}$ |

### Main Takeaways

1. **Triorthogonality** ensures T-gate phase consistency
2. The [[15, 1, 3]] code is triorthogonal from RM structure
3. **15-to-1 distillation** gives cubic error reduction
4. Multiple rounds achieve exponential improvement
5. Trade-off between overhead and error reduction rate

---

## Daily Checklist

- [ ] I can verify triorthogonality of a matrix
- [ ] I understand connection to T-gate distillation
- [ ] I can compute distillation error progression
- [ ] I know the 15-to-1 protocol structure
- [ ] I can compare different distillation protocols
- [ ] I understand the overhead requirements

---

## Preview: Day 753

Tomorrow we explore **good qLDPC codes**:

- Asymptotically good parameters
- Constant rate and linear distance
- Expander-based constructions
- Recent breakthrough results

Good qLDPC codes promise efficient fault-tolerant quantum computing!
