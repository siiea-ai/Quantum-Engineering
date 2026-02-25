# Day 695: Quantum Hamming Bound & Perfect Codes

## Overview

**Week:** 100 (QEC Conditions)
**Day:** Tuesday
**Date:** Year 2, Month 25, Day 695
**Topic:** The Quantum Hamming Bound and Perfect Quantum Codes
**Hours:** 7 (3.5 theory + 2.5 problems + 1 computational lab)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Hamming bound derivation, sphere-packing |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Perfect codes analysis |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Bound comparison implementation |

---

## Prerequisites

From Day 694:
- Quantum Singleton bound
- MDS codes
- Code parameter constraints

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Derive** the quantum Hamming bound using sphere-packing arguments
2. **Calculate** the bound for specific code parameters
3. **Define** perfect quantum codes and identify examples
4. **Explain** why the [[5,1,3]] code is both MDS and perfect
5. **Compare** Singleton and Hamming bounds
6. **Understand** the rarity of perfect quantum codes

---

## Core Content

### 1. The Sphere-Packing Argument

#### Classical Intuition

In classical coding theory, the Hamming bound arises from "packing spheres":
- Each codeword has a "sphere" of strings within Hamming distance $t$
- These spheres must be disjoint for error correction
- The total volume of spheres cannot exceed the space volume

#### Quantum Extension

For quantum codes, we generalize:
- Each logical state has a "sphere" of error-affected states
- Errors are Pauli operators (I, X, Y, Z on each qubit)
- Spheres must allow unique syndrome identification

---

### 2. The Quantum Hamming Bound

#### Statement

**Theorem (Quantum Hamming Bound):**

For a **non-degenerate** quantum code with parameters $[[n, k, d]]$ where $d = 2t + 1$:

$$\boxed{\sum_{j=0}^{t} 3^j \binom{n}{j} \leq 2^{n-k}}$$

For single-error correction ($t = 1$, $d = 3$):

$$\boxed{1 + 3n \leq 2^{n-k}}$$

#### Derivation

**Step 1:** Count distinguishable errors.

For a code correcting $t$ errors, distinct errors include:
- Identity (no error): 1 way
- Single Pauli on 1 qubit: $3 \binom{n}{1}$ ways (X, Y, or Z on one of $n$ qubits)
- Two Paulis on 2 qubits: $3^2 \binom{n}{2}$ ways
- ...
- $j$ Paulis on $j$ qubits: $3^j \binom{n}{j}$ ways

**Step 2:** Total distinguishable error patterns.

$$N_{errors} = \sum_{j=0}^{t} 3^j \binom{n}{j}$$

**Step 3:** Each error must map to a unique syndrome.

The syndrome space has dimension $n - k$ (number of stabilizer generators).
Total syndromes: $2^{n-k}$

**Step 4:** Counting constraint.

For unique syndrome mapping: $N_{errors} \leq 2^{n-k}$

$$\sum_{j=0}^{t} 3^j \binom{n}{j} \leq 2^{n-k}$$

---

### 3. Interpretation and Examples

#### Single-Error Correction (t = 1)

$$1 + 3n \leq 2^{n-k}$$

Solving for $k$:
$$k \leq n - \log_2(1 + 3n)$$

| n | $1 + 3n$ | $2^{n-k}$ needed | Max $k$ |
|---|----------|------------------|---------|
| 5 | 16 | 16 | 1 |
| 6 | 19 | 32 | 1 |
| 7 | 22 | 32 | 1 |
| 8 | 25 | 32 | 2 |
| 9 | 28 | 32 | 3 |
| 15 | 46 | 64 | 8 |

#### Double-Error Correction (t = 2, d = 5)

$$1 + 3n + 9\binom{n}{2} \leq 2^{n-k}$$

For $n = 11$:
$$1 + 33 + 9(55) = 1 + 33 + 495 = 529$$
$$\log_2(529) \approx 9.05$$
$$k \leq 11 - 10 = 1$$

So [[11, 1, 5]] is the maximum for $d = 5$ with 11 qubits.

---

### 4. Perfect Quantum Codes

#### Definition

A quantum code is **perfect** if the Hamming bound is satisfied with equality:

$$\boxed{\sum_{j=0}^{t} 3^j \binom{n}{j} = 2^{n-k}}$$

**Interpretation:** Every syndrome corresponds to exactly one correctable error — no "wasted" syndromes.

#### Characterization

Perfect codes achieve:
1. Maximum information rate for given $n$ and $d$
2. Optimal use of syndrome space
3. Every error pattern within distance $t$ has unique correction

#### Known Perfect Quantum Codes

| Code | Parameters | Sphere Size | Syndrome Count | Perfect? |
|------|------------|-------------|----------------|----------|
| [[5,1,3]] | $t=1$ | $1 + 15 = 16$ | $2^4 = 16$ | ✓ |
| Trivial [[1,1,1]] | $t=0$ | $1$ | $1$ | ✓ |

**Remarkable fact:** The [[5,1,3]] code is the **only** non-trivial perfect quantum code that corrects single-qubit errors!

#### Why Perfect Codes Are Rare

The equation $1 + 3n = 2^{n-k}$ has very few integer solutions:
- $n = 5, k = 1$: $1 + 15 = 16 = 2^4$ ✓
- $n = 2^r - 1$ for some patterns, but quantum constraints further restrict

Classical Hamming codes form an infinite family of perfect codes. Quantum perfect codes are extremely rare.

---

### 5. The [[5,1,3]] Code: Doubly Optimal

The five-qubit code is special because it simultaneously:

1. **Saturates Singleton:** $k = n - 2(d-1) = 5 - 4 = 1$ ✓ (MDS)
2. **Saturates Hamming:** $1 + 3(5) = 16 = 2^4$ ✓ (Perfect)

#### Stabilizer Structure

$$\begin{aligned}
g_1 &= XZZXI \\
g_2 &= IXZZX \\
g_3 &= XIXZZ \\
g_4 &= ZXIXZ
\end{aligned}$$

**Properties:**
- 4 generators → $2^4 = 16$ syndromes
- 16 single-qubit errors (including identity)
- Each error has unique syndrome!

#### Syndrome Table

| Error | Syndrome $(g_1, g_2, g_3, g_4)$ |
|-------|--------------------------------|
| I | 0000 |
| $X_1$ | 1001 |
| $X_2$ | 0101 |
| $X_3$ | 1010 |
| $X_4$ | 0110 |
| $X_5$ | 0011 |
| $Z_1$ | 0110 |
| ... | ... |
| $Y_5$ | 1111 |

(Note: Actual syndromes depend on stabilizer ordering)

---

### 6. Comparing Singleton and Hamming Bounds

#### Which Is Tighter?

| Scenario | Singleton | Hamming | Winner |
|----------|-----------|---------|--------|
| Small $n$, small $d$ | Often tighter | Looser | Singleton |
| Large $n$, moderate $d$ | Looser | Tighter | Hamming |
| Large $d$ | Tighter | Looser | Singleton |

#### Example: $n = 15, d = 3$

**Singleton:** $k \leq 15 - 4 = 11$

**Hamming:** $1 + 45 = 46 \leq 2^{15-k}$
$$2^{15-k} \geq 46 \Rightarrow 15 - k \geq 6 \Rightarrow k \leq 9$$

**Winner:** Hamming bound is tighter (allows $k \leq 9$, not $k \leq 11$)

The [[15,7,3]] CSS code exists, well within both bounds.

#### Example: $n = 5, d = 3$

**Singleton:** $k \leq 5 - 4 = 1$

**Hamming:** $1 + 15 = 16 \leq 2^{5-k}$
$$2^{5-k} \geq 16 \Rightarrow 5 - k \geq 4 \Rightarrow k \leq 1$$

**Winner:** Tie! Both give $k \leq 1$.

The [[5,1,3]] code saturates both — it's MDS AND perfect!

---

### 7. Beyond Non-Degenerate Codes

#### Limitation of Hamming Bound

The quantum Hamming bound assumes **non-degeneracy** — each error produces a unique syndrome.

**Degenerate codes** can violate this assumption:
- Multiple errors produce the same syndrome
- But they act identically on the code space
- Effectively "more" errors become correctable

#### Example: Degenerate Violation

A degenerate code might have:
- Sphere size counting suggests 64 distinguishable errors
- But only 32 syndromes
- Yet the code still corrects all single-qubit errors!

The Steane [[7,1,3]] code is degenerate and has:
- $1 + 21 = 22$ single-qubit errors
- $2^6 = 64$ syndromes
- Many syndromes are "unused" but degeneracy helps

---

## Quantum Mechanics Connection

### Measurement and Distinguishability

The Hamming bound fundamentally relies on quantum measurement theory:

1. **Syndrome measurement** extracts classical information
2. **Distinguishable errors** must produce orthogonal syndrome states
3. **Sphere-packing** limits how many orthogonal states fit in syndrome space

### Holevo Bound Connection

The number of distinguishable syndromes relates to the Holevo bound on accessible information. With $n-k$ ancilla qubits for syndrome extraction:

$$\text{Distinguishable syndromes} \leq 2^{n-k}$$

This is the fundamental information-theoretic limit.

---

## Worked Examples

### Example 1: Verify [[5,1,3]] is Perfect

**Problem:** Show the [[5,1,3]] code saturates the Hamming bound.

**Solution:**

For $n = 5$, $k = 1$, $d = 3$ ($t = 1$):

LHS of Hamming bound:
$$\sum_{j=0}^{1} 3^j \binom{5}{j} = 1 \cdot 1 + 3 \cdot 5 = 1 + 15 = 16$$

RHS of Hamming bound:
$$2^{n-k} = 2^{5-1} = 2^4 = 16$$

Since $16 = 16$, the bound is saturated. The [[5,1,3]] code is perfect. ∎

---

### Example 2: Check [[7,1,3]] Against Hamming

**Problem:** Does the Steane [[7,1,3]] code satisfy the Hamming bound? Is it perfect?

**Solution:**

For $n = 7$, $k = 1$, $d = 3$ ($t = 1$):

LHS: $1 + 3(7) = 22$

RHS: $2^{7-1} = 64$

Check: $22 \leq 64$ ✓

The bound is satisfied, but not with equality ($22 \neq 64$).

**Conclusion:** Steane code is NOT perfect. It has 42 "unused" syndromes.

---

### Example 3: Maximum k for n = 23, d = 7

**Problem:** Find the maximum logical qubits for a [[23, k, 7]] code.

**Solution:**

For $d = 7$, $t = 3$ (corrects 3 errors).

**Singleton bound:** $k \leq 23 - 2(6) = 11$

**Hamming bound:**
$$\sum_{j=0}^{3} 3^j \binom{23}{j} = 1 + 3(23) + 9(253) + 27(1771)$$
$$= 1 + 69 + 2277 + 47817 = 50164$$

Need: $2^{23-k} \geq 50164$
$$23 - k \geq \log_2(50164) \approx 15.6$$
$$k \leq 23 - 16 = 7$$

**Conclusion:** Hamming bound gives $k \leq 7$, tighter than Singleton's $k \leq 11$.

Maximum: **k = 7** for [[23, k, 7]].

---

## Practice Problems

### Level 1: Direct Application

1. **Bound Calculation:**
   Calculate the Hamming bound for [[9, k, 3]] codes. What is the maximum $k$?

2. **Perfect Check:**
   Is a [[6, 0, 4]] code perfect? Calculate both sides of the Hamming bound ($t = 1$).

3. **Sphere Size:**
   How many distinguishable single-qubit errors exist for $n = 10$ qubits?

### Level 2: Intermediate

4. **Bound Comparison:**
   For $n = 11$, $d = 5$, which bound is tighter — Singleton or Hamming?

5. **Non-Existence Proof:**
   Use the Hamming bound to prove no [[8, 4, 3]] code exists (we showed this with Hamming in Day 694).

6. **Degenerate vs Non-Degenerate:**
   The Shor [[9,1,3]] code has $2^8 = 256$ syndromes. How many single-qubit errors exist? What does the ratio tell you?

### Level 3: Challenging

7. **Perfect Code Search:**
   Find all $(n, k)$ pairs with $n \leq 20$ where a perfect single-error correcting code ($d = 3$) could exist.

8. **Generalization:**
   Derive the Hamming bound for codes correcting $t$ errors. Express it in closed form.

9. **Tightness Analysis:**
   For what code parameters is the Singleton bound tighter than Hamming? Characterize the regime.

---

## Computational Lab

### Hamming Bound Analysis

```python
"""
Day 695 Computational Lab: Quantum Hamming Bound Analysis
Sphere-packing and perfect code exploration
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb, log2, ceil, floor
from typing import List, Tuple

def sphere_size(n: int, t: int) -> int:
    """
    Calculate the size of error sphere for t-error correction.

    Args:
        n: Number of qubits
        t: Number of correctable errors

    Returns:
        Total number of distinguishable errors
    """
    total = 0
    for j in range(t + 1):
        total += (3 ** j) * comb(n, j)
    return total


def hamming_bound_max_k(n: int, d: int) -> int:
    """
    Calculate maximum k allowed by Hamming bound.

    Args:
        n: Number of physical qubits
        d: Code distance

    Returns:
        Maximum number of logical qubits
    """
    t = (d - 1) // 2
    sphere = sphere_size(n, t)

    if sphere <= 0:
        return n

    # 2^(n-k) >= sphere
    # n - k >= log2(sphere)
    # k <= n - log2(sphere)
    k_max = n - ceil(log2(sphere))
    return max(0, k_max)


def singleton_bound_max_k(n: int, d: int) -> int:
    """Calculate maximum k from Singleton bound."""
    return max(0, n - 2 * (d - 1))


def is_perfect(n: int, k: int, d: int) -> bool:
    """Check if [[n,k,d]] parameters would make a perfect code."""
    t = (d - 1) // 2
    sphere = sphere_size(n, t)
    syndrome_space = 2 ** (n - k)
    return sphere == syndrome_space


def analyze_hamming_bound():
    """Comprehensive Hamming bound analysis."""

    print("=" * 70)
    print("QUANTUM HAMMING BOUND ANALYSIS")
    print("=" * 70)

    # 1. Sphere sizes for single-error correction
    print("\n1. ERROR SPHERE SIZES (t = 1, d = 3)")
    print("-" * 50)
    print(f"{'n':>4} {'Sphere 1+3n':>15} {'log₂(sphere)':>15}")
    print("-" * 50)

    for n in range(3, 21):
        sphere = sphere_size(n, 1)
        log_sphere = log2(sphere)
        print(f"{n:>4} {sphere:>15} {log_sphere:>15.2f}")

    # 2. Compare Singleton and Hamming bounds
    print("\n2. SINGLETON VS HAMMING BOUND COMPARISON")
    print("-" * 70)
    print(f"{'n':>4} {'d':>4} {'Singleton k≤':>14} {'Hamming k≤':>14} {'Tighter':>12}")
    print("-" * 70)

    for n in range(5, 21):
        for d in [3, 5, 7]:
            if 2*(d-1) > n:
                continue
            s_max = singleton_bound_max_k(n, d)
            h_max = hamming_bound_max_k(n, d)

            if h_max < s_max:
                tighter = "Hamming"
            elif s_max < h_max:
                tighter = "Singleton"
            else:
                tighter = "Equal"

            print(f"{n:>4} {d:>4} {s_max:>14} {h_max:>14} {tighter:>12}")

    # 3. Search for perfect codes
    print("\n3. PERFECT CODE SEARCH (d = 3)")
    print("-" * 50)
    print("Codes where sphere_size = 2^(n-k):")
    print(f"{'n':>4} {'k':>4} {'Sphere':>10} {'Syndromes':>12} {'Perfect?':>10}")
    print("-" * 50)

    for n in range(3, 25):
        sphere = sphere_size(n, 1)
        # Check if sphere is a power of 2
        if sphere > 0 and (sphere & (sphere - 1)) == 0:
            n_minus_k = int(log2(sphere))
            k = n - n_minus_k
            if k >= 0:
                syndromes = 2 ** n_minus_k
                is_perf = "YES" if sphere == syndromes else "NO"
                print(f"{n:>4} {k:>4} {sphere:>10} {syndromes:>12} {is_perf:>10}")


def plot_bound_comparison():
    """Visualize Singleton vs Hamming bounds."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Maximum k for d=3
    ax1 = axes[0]
    n_values = range(5, 26)

    singleton_k = [singleton_bound_max_k(n, 3) for n in n_values]
    hamming_k = [hamming_bound_max_k(n, 3) for n in n_values]

    ax1.plot(n_values, singleton_k, 'b-o', label='Singleton bound', markersize=6)
    ax1.plot(n_values, hamming_k, 'r-s', label='Hamming bound', markersize=6)
    ax1.fill_between(n_values, 0, [min(s, h) for s, h in zip(singleton_k, hamming_k)],
                     alpha=0.3, color='green', label='Achievable region')

    # Mark known codes
    known_codes_d3 = [(5, 1), (7, 1), (9, 1), (15, 7)]
    for n, k in known_codes_d3:
        if 5 <= n <= 25:
            ax1.scatter([n], [k], c='gold', s=200, marker='*',
                       edgecolors='black', linewidth=1, zorder=5)

    ax1.set_xlabel('Number of physical qubits n', fontsize=12)
    ax1.set_ylabel('Maximum logical qubits k', fontsize=12)
    ax1.set_title('Code Parameter Bounds (d = 3)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error sphere growth
    ax2 = axes[1]
    n_range = range(3, 21)

    sphere_t1 = [sphere_size(n, 1) for n in n_range]
    sphere_t2 = [sphere_size(n, 2) for n in n_range]
    syndrome_space = [2**n for n in n_range]

    ax2.semilogy(n_range, sphere_t1, 'b-o', label='Sphere size (t=1)', markersize=6)
    ax2.semilogy(n_range, sphere_t2, 'r-s', label='Sphere size (t=2)', markersize=6)
    ax2.semilogy(n_range, syndrome_space, 'k--', label='Total space 2ⁿ', alpha=0.5)

    ax2.set_xlabel('Number of qubits n', fontsize=12)
    ax2.set_ylabel('Number of states (log scale)', fontsize=12)
    ax2.set_title('Error Sphere Growth vs Total Space', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hamming_bound_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: hamming_bound_analysis.png")


def demonstrate_five_qubit_perfection():
    """Show why [[5,1,3]] is perfect."""

    print("\n" + "=" * 70)
    print("THE [[5,1,3]] CODE: DOUBLY OPTIMAL")
    print("=" * 70)

    n, k, d = 5, 1, 3
    t = (d - 1) // 2

    print(f"\nCode parameters: [[{n}, {k}, {d}]]")
    print(f"Correctable errors: t = {t}")

    # Singleton check
    singleton_limit = n - 2 * (d - 1)
    print(f"\n1. SINGLETON BOUND:")
    print(f"   k ≤ n - 2(d-1) = {n} - {2*(d-1)} = {singleton_limit}")
    print(f"   Actual k = {k}")
    print(f"   Saturated: {'YES (MDS)' if k == singleton_limit else 'NO'}")

    # Hamming check
    sphere = sphere_size(n, t)
    syndromes = 2 ** (n - k)
    print(f"\n2. HAMMING BOUND:")
    print(f"   Error sphere = 1 + 3n = 1 + {3*n} = {sphere}")
    print(f"   Syndrome space = 2^(n-k) = 2^{n-k} = {syndromes}")
    print(f"   Saturated: {'YES (Perfect)' if sphere == syndromes else 'NO'}")

    # Error enumeration
    print(f"\n3. ERROR ENUMERATION:")
    print(f"   Identity: 1")
    print(f"   X errors: {n}")
    print(f"   Y errors: {n}")
    print(f"   Z errors: {n}")
    print(f"   Total: 1 + 3×{n} = {1 + 3*n}")

    print(f"\n4. SYNDROME USAGE:")
    print(f"   Available syndromes: {syndromes}")
    print(f"   Required syndromes: {sphere}")
    print(f"   Efficiency: {sphere}/{syndromes} = {100*sphere/syndromes:.1f}%")

    print("\n   The [[5,1,3]] code uses EVERY syndrome exactly once!")
    print("   This is why it's called 'perfect' — no wasted redundancy.")


def code_possibility_table():
    """Generate comprehensive table of possible codes."""

    print("\n" + "=" * 70)
    print("COMPREHENSIVE CODE POSSIBILITY TABLE")
    print("=" * 70)

    print("\nFor each (n, d), showing maximum k from each bound and known codes:")
    print("-" * 70)
    print(f"{'n':>3} {'d':>3} {'Singleton':>10} {'Hamming':>10} {'Best':>8} {'Known':>15}")
    print("-" * 70)

    # Known codes for reference
    known = {
        (5, 3): "[[5,1,3]] ✓",
        (7, 3): "[[7,1,3]] ✓",
        (9, 3): "[[9,1,3]] ✓",
        (15, 3): "[[15,7,3]] ✓",
        (4, 2): "[[4,2,2]] ✓",
        (6, 4): "[[6,0,4]] ✓",
    }

    for n in range(4, 16):
        for d in [2, 3, 4, 5]:
            if d > n or 2*(d-1) > n:
                continue

            s_max = singleton_bound_max_k(n, d)
            h_max = hamming_bound_max_k(n, d)
            best = min(s_max, h_max)

            known_str = known.get((n, d), "")

            print(f"{n:>3} {d:>3} {s_max:>10} {h_max:>10} {best:>8} {known_str:>15}")


if __name__ == "__main__":
    analyze_hamming_bound()
    demonstrate_five_qubit_perfection()
    code_possibility_table()

    print("\n" + "=" * 70)
    print("Generating visualization...")
    print("=" * 70)
    plot_bound_comparison()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Error sphere size | $\sum_{j=0}^{t} 3^j \binom{n}{j}$ |
| Hamming bound | $\sum_{j=0}^{t} 3^j \binom{n}{j} \leq 2^{n-k}$ |
| Single-error ($t=1$) | $1 + 3n \leq 2^{n-k}$ |
| Perfect code condition | Equality in Hamming bound |
| [[5,1,3]] sphere | $1 + 15 = 16 = 2^4$ |

### Main Takeaways

1. **Sphere-Packing:** The Hamming bound counts distinguishable errors that must fit in syndrome space
2. **Perfect Codes:** Rare codes that use every syndrome exactly once
3. **[[5,1,3]] Uniqueness:** The only non-trivial perfect single-error correcting quantum code
4. **Bound Comparison:** Hamming often tighter than Singleton for larger codes
5. **Degeneracy Exception:** Degenerate codes can "beat" the Hamming bound interpretation

---

## Daily Checklist

- [ ] Can derive the quantum Hamming bound
- [ ] Understand sphere-packing interpretation
- [ ] Can identify perfect codes
- [ ] Know why [[5,1,3]] is doubly optimal
- [ ] Can compare Singleton and Hamming bounds
- [ ] Understand the non-degenerate assumption

---

## Preview: Day 696

Tomorrow we explore **Degeneracy in Quantum Codes** — how some codes can "cheat" the Hamming bound:

- Formal definition of degenerate codes
- Why degeneracy helps error correction
- Steane and Shor codes as degenerate examples
- Implications for practical quantum computing

Degeneracy is key to understanding why surface codes work so well!

---

*"The [[5,1,3]] code is the Platonic ideal of quantum error correction — perfect in every sense."*
