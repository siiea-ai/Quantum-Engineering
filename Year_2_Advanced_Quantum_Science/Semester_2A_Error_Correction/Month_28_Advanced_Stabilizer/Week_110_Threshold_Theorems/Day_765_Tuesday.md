# Day 765: Concatenated Code Analysis

## Overview

**Day:** 765 of 1008
**Week:** 110 (Threshold Theorems & Analysis)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Multi-Level Error Correction Through Concatenation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Concatenation structure |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Error rate analysis |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Optimal depth |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Construct** multi-level concatenated codes
2. **Derive** the error rate recursion formula
3. **Compute** optimal concatenation depth
4. **Analyze** resource scaling with concatenation
5. **Compare** different base codes for concatenation
6. **Optimize** code selection for given error rates

---

## Core Content

### 1. Concatenation Structure

**Concatenation** applies error correction recursively, encoding encoded qubits into higher-level codes.

#### Level Hierarchy

**Level 0:** Physical qubits
$$|0\rangle, |1\rangle$$

**Level 1:** First encoding (n qubits per logical)
$$|0\rangle_L^{(1)} = |0_1 0_2 \cdots 0_n\rangle$$

**Level 2:** Encode each level-1 qubit
$$|0\rangle_L^{(2)} = |0\rangle_L^{(1)} \otimes |0\rangle_L^{(1)} \otimes \cdots$$

**Level k:**
$$\boxed{|0\rangle_L^{(k)}: n^k \text{ physical qubits}}$$

### 2. The Recursion Relation

For a distance-d code correcting t = ⌊(d-1)/2⌋ errors:

**Key observation:** Need t+1 faults at level k for one error at level k+1.

#### General Form

$$\boxed{p^{(k+1)} \leq A \binom{N}{t+1} (p^{(k)})^{t+1}}$$

where:
- A: Constant depending on gadget structure
- N: Number of fault locations in gadget
- t: Errors corrected by code

#### For t = 1 (e.g., [[7,1,3]])

$$p^{(k+1)} = c \cdot (p^{(k)})^2$$

where $c = A \binom{N}{2}$ is the "malignant pair count."

### 3. Deriving the Recursion Constant

**Step 1: Count fault locations**

For a FT-CNOT between two [[7,1,3]] blocks:
- 7 transversal CNOTs
- Error correction on each block (~20 locations each)
- Total: N ≈ 50-100 locations

**Step 2: Count malignant pairs**

Pairs that cause logical error:
- Two faults in same EC block → possible weight-2 error
- Faults that combine across blocks

**Step 3: Compute constant**

$$c = (\text{malignant pairs}) \cdot (\text{failure prob per pair})$$

Typical values: c ≈ 100-1000 depending on gadget.

### 4. Optimal Concatenation Depth

**Goal:** Achieve target error $\epsilon$ with minimum overhead.

**Error at level k:**
$$p^{(k)} = p_{th} \cdot \left(\frac{p}{p_{th}}\right)^{2^k}$$

**Required depth:**
$$k^* = \lceil \log_2(\log(p_{th}/\epsilon) / \log(p_{th}/p)) \rceil$$

#### Resource Counting

Physical qubits: $n^{k^*}$

For target error $\epsilon$:
$$\boxed{n^{k^*} = n^{O(\log\log(1/\epsilon))} = O(\text{polylog}(1/\epsilon))}$$

### 5. Comparing Base Codes

| Code | n | d | t | c (typical) | Threshold |
|------|---|---|---|-------------|-----------|
| [[5,1,3]] | 5 | 3 | 1 | ~50 | ~2% |
| [[7,1,3]] | 7 | 3 | 1 | ~100 | ~1% |
| [[23,1,7]] | 23 | 7 | 3 | ~500 | ~0.2% |

**Trade-off:**
- Larger code: More overhead per level
- Higher distance: Faster error suppression per level
- Optimal choice depends on physical error rate

### 6. Beyond Simple Concatenation

#### Heterogeneous Concatenation

Use different codes at different levels:
- Level 1: High-threshold code (e.g., [[7,1,3]])
- Level 2+: High-distance code (e.g., [[23,1,7]])

#### Concatenation with Topological Codes

- Inner level: Concatenated code
- Outer level: Surface code

Best of both: High threshold + fast gates

---

## Worked Examples

### Example 1: Full Recursion Derivation

**Problem:** For [[7,1,3]] code with c = 100, compute error rates for k = 0, 1, 2, 3, 4 given p = 0.001.

**Solution:**

Threshold: $p_{th} = 1/c = 0.01$

Recursion: $p^{(k+1)} = 100 \cdot (p^{(k)})^2$

| Level | Error Rate |
|-------|------------|
| 0 | 0.001 = 10⁻³ |
| 1 | 100 × (10⁻³)² = 10⁻⁴ |
| 2 | 100 × (10⁻⁴)² = 10⁻⁶ |
| 3 | 100 × (10⁻⁶)² = 10⁻¹⁰ |
| 4 | 100 × (10⁻¹⁰)² = 10⁻¹⁸ |

Each level: Error rate squares (in exponent)!

### Example 2: Optimal Depth Calculation

**Problem:** Find minimum k for logical error < 10⁻¹² with p = 0.005, p_th = 0.01.

**Solution:**

Using the formula:
$$p^{(k)} = p_{th} \cdot (p/p_{th})^{2^k} = 0.01 \cdot (0.5)^{2^k}$$

Want: $0.01 \cdot (0.5)^{2^k} < 10^{-12}$

$(0.5)^{2^k} < 10^{-10}$

$2^k \cdot \log_{10}(0.5) < -10$

$2^k \cdot (-0.301) < -10$

$2^k > 33.2$

$k > 5.05$

**Answer:** k = 6 levels

### Example 3: Resource Comparison

**Problem:** Compare physical qubits for [[5,1,3]] vs [[7,1,3]] codes to achieve error 10⁻¹⁵.

**Solution:**

Assume both have similar threshold ≈ 0.01 and p = 0.005.

From Example 2, need k ≈ 6 levels (assuming similar c).

| Code | n | Physical qubits |
|------|---|-----------------|
| [[5,1,3]] | 5 | 5⁶ = 15,625 |
| [[7,1,3]] | 7 | 7⁶ = 117,649 |

**Conclusion:** [[5,1,3]] has ~7.5× lower qubit overhead.

But! If [[5,1,3]] has lower threshold, may need more levels.

---

## Practice Problems

### Problem Set A: Recursion Analysis

**A1.** For recursion $p' = 500 p^3$, find the threshold and compute p^(3) for p = 0.01.

**A2.** Compare [[7,1,3]] (c=100, quadratic) vs [[23,1,7]] (c=500, p⁴) for p = 0.001.

**A3.** Derive the recursion constant c for a code with N = 80 fault locations and 150 malignant pairs.

### Problem Set B: Optimization

**B1.** For fixed total qubit budget of 50,000, what's the maximum concatenation depth for [[7,1,3]]?

**B2.** Optimize code choice: [[5,1,3]] (p_th = 2%) or [[7,1,3]] (p_th = 1%) for physical error 1.5%.

**B3.** Design a heterogeneous concatenation scheme using [[7,1,3]] inner and [[23,1,7]] outer.

### Problem Set C: Advanced

**C1.** How does threshold change if 5% of fault locations are "bad" (cause weight-3 errors)?

**C2.** Compute the break-even point: at what p does 2-level [[7,1,3]] equal 1-level [[23,1,7]]?

**C3.** For quantum memory (no computation), how does concatenation analysis change?

---

## Computational Lab

```python
"""
Day 765 Computational Lab: Concatenated Code Analysis
=====================================================

Analyze concatenated code performance and optimization.
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class ConcatenatedCode:
    """Represents a concatenated quantum error correcting code."""
    name: str
    n: int          # Physical qubits per logical
    k: int          # Logical qubits
    d: int          # Distance
    c: float        # Recursion constant
    exponent: int   # t+1 in recursion

    @property
    def threshold(self) -> float:
        """Compute threshold from recursion constant."""
        return (1.0 / self.c) ** (1.0 / (self.exponent - 1))

    def error_at_level(self, p: float, level: int) -> float:
        """Compute logical error rate at given level."""
        if level == 0:
            return p

        p_th = self.threshold
        if p >= p_th:
            return 1.0  # Above threshold

        ratio = p / p_th
        return p_th * (ratio ** (self.exponent ** level))

    def levels_for_target(self, p: float, target: float) -> int:
        """Minimum levels to achieve target error."""
        if p >= self.threshold:
            return -1

        level = 0
        while self.error_at_level(p, level) > target and level < 20:
            level += 1

        return level

    def physical_qubits(self, levels: int) -> int:
        """Physical qubits per logical qubit."""
        return self.n ** levels


def compare_codes(codes: List[ConcatenatedCode],
                 physical_error: float,
                 target_error: float) -> Dict:
    """Compare codes for given error requirements."""
    results = []

    for code in codes:
        levels = code.levels_for_target(physical_error, target_error)
        if levels < 0:
            continue

        qubits = code.physical_qubits(levels)
        final_error = code.error_at_level(physical_error, levels)

        results.append({
            'code': code.name,
            'threshold': code.threshold,
            'levels': levels,
            'qubits': qubits,
            'final_error': final_error
        })

    return sorted(results, key=lambda x: x['qubits'])


def optimize_concatenation(code: ConcatenatedCode,
                          physical_error: float,
                          qubit_budget: int) -> Dict:
    """Find best achievable error within qubit budget."""
    max_levels = 1
    while code.physical_qubits(max_levels) <= qubit_budget:
        max_levels += 1
    max_levels -= 1

    if max_levels < 1:
        return {'error': 'Qubit budget too small'}

    error = code.error_at_level(physical_error, max_levels)

    return {
        'levels': max_levels,
        'qubits_used': code.physical_qubits(max_levels),
        'achievable_error': error
    }


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 765: CONCATENATED CODE ANALYSIS")
    print("=" * 70)

    # Define codes
    steane = ConcatenatedCode("[[7,1,3]]", 7, 1, 3, 100, 2)
    perfect = ConcatenatedCode("[[5,1,3]]", 5, 1, 3, 50, 2)
    golay = ConcatenatedCode("[[23,1,7]]", 23, 1, 7, 500, 4)

    codes = [steane, perfect, golay]

    # Demo 1: Error progression
    print("\n" + "=" * 70)
    print("Demo 1: Error Progression Through Levels")
    print("=" * 70)

    p = 0.005
    print(f"\nPhysical error: p = {p}")

    print(f"\n{'Level':<8} ", end="")
    for code in codes:
        print(f"{code.name:<15} ", end="")
    print()
    print("-" * 55)

    for level in range(6):
        print(f"{level:<8} ", end="")
        for code in codes:
            error = code.error_at_level(p, level)
            print(f"{error:<15.2e} ", end="")
        print()

    # Demo 2: Threshold comparison
    print("\n" + "=" * 70)
    print("Demo 2: Code Thresholds")
    print("=" * 70)

    print(f"\n{'Code':<15} {'n':<5} {'d':<5} {'c':<8} {'p_th':<10}")
    print("-" * 50)

    for code in codes:
        print(f"{code.name:<15} {code.n:<5} {code.d:<5} "
              f"{code.c:<8.0f} {code.threshold:<10.4f}")

    # Demo 3: Resource comparison
    print("\n" + "=" * 70)
    print("Demo 3: Resource Comparison for Target Error")
    print("=" * 70)

    target = 1e-15
    p = 0.005

    print(f"\nTarget error: {target}")
    print(f"Physical error: {p}")

    comparison = compare_codes(codes, p, target)
    print(f"\n{'Code':<15} {'Levels':<8} {'Qubits':<12} {'Final Error':<12}")
    print("-" * 55)

    for result in comparison:
        print(f"{result['code']:<15} {result['levels']:<8} "
              f"{result['qubits']:<12} {result['final_error']:<12.2e}")

    # Demo 4: Budget optimization
    print("\n" + "=" * 70)
    print("Demo 4: Optimization with Qubit Budget")
    print("=" * 70)

    budget = 50000
    print(f"\nQubit budget: {budget}")

    for code in codes:
        result = optimize_concatenation(code, 0.005, budget)
        if 'error' not in result:
            print(f"\n{code.name}:")
            print(f"  Levels: {result['levels']}")
            print(f"  Qubits used: {result['qubits_used']}")
            print(f"  Achievable error: {result['achievable_error']:.2e}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCATENATION ANALYSIS SUMMARY")
    print("=" * 70)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  CONCATENATED CODE ANALYSIS                                 │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  RECURSION RELATION:                                        │
    │    p^(k+1) = c × (p^(k))^(t+1)                             │
    │                                                             │
    │  THRESHOLD:                                                 │
    │    p_th = (1/c)^(1/t)  for t-error-correcting code        │
    │                                                             │
    │  ERROR SCALING:                                             │
    │    p^(k) = p_th × (p/p_th)^((t+1)^k)                       │
    │    Doubly exponential improvement!                         │
    │                                                             │
    │  RESOURCE SCALING:                                          │
    │    Physical qubits = n^k                                   │
    │    Polylogarithmic in 1/ε                                  │
    │                                                             │
    │  TRADE-OFFS:                                                │
    │    • Larger n: More overhead per level                     │
    │    • Higher d: Faster convergence per level                │
    │    • Optimal choice depends on p/p_th ratio               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)

    print("=" * 70)
    print("Day 765 Complete: Concatenated Code Analysis Mastered")
    print("=" * 70)
```

---

## Summary

### Concatenation Hierarchy

| Level | Qubits | Error Rate |
|-------|--------|------------|
| 0 | 1 | p |
| 1 | n | c⋅p² |
| 2 | n² | c⋅(c⋅p²)² |
| k | n^k | p_th⋅(p/p_th)^(2^k) |

### Critical Equations

$$\boxed{p^{(k+1)} = c \cdot (p^{(k)})^{t+1}}$$

$$\boxed{p_{th} = (1/c)^{1/t}}$$

$$\boxed{\text{Qubits} = n^{O(\log\log(1/\epsilon))}}$$

---

## Daily Checklist

- [ ] Constructed multi-level concatenated codes
- [ ] Derived error recursion formulas
- [ ] Computed optimal concatenation depths
- [ ] Compared different base codes
- [ ] Ran optimization simulations
- [ ] Completed practice problems

---

## Preview: Day 766

Tomorrow we study **Noise Models & Assumptions**:
- Depolarizing channel
- Erasure errors
- Biased noise models
- Impact on thresholds
