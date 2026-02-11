# Day 694: Quantum Singleton Bound

## Overview

**Week:** 100 (QEC Conditions)
**Day:** Monday
**Date:** Year 2, Month 25, Day 694
**Topic:** The Quantum Singleton Bound — Fundamental Limits of Quantum Codes
**Hours:** 7 (3.5 theory + 2.5 problems + 1 computational lab)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Singleton bound derivation, MDS codes |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem solving, bound verification |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational exploration of code bounds |

---

## Prerequisites

From Week 99:
- Stabilizer formalism and code parameters $[[n, k, d]]$
- Knill-Laflamme conditions
- CSS code construction
- Shor and Steane code analysis

---

## Learning Objectives

By the end of this day, you will be able to:

1. **State** the quantum Singleton bound and explain its significance
2. **Derive** the bound using quantum information-theoretic arguments
3. **Compare** quantum and classical Singleton bounds
4. **Identify** codes that saturate the bound (MDS codes)
5. **Apply** the bound to verify code parameters are achievable
6. **Understand** why the quantum bound is tighter by a factor of 2

---

## Core Content

### 1. The Quantum Singleton Bound

#### Statement of the Bound

**Theorem (Quantum Singleton Bound / Knill-Laflamme Bound):**

For any quantum error correcting code with parameters $[[n, k, d]]$:

$$\boxed{k \leq n - 2(d - 1)}$$

Equivalently:

$$\boxed{n - k \geq 2(d - 1)}$$

or

$$\boxed{d \leq \frac{n - k}{2} + 1}$$

#### Physical Interpretation

- **$n - k$:** Number of physical qubits dedicated to redundancy (stabilizer generators)
- **$2(d-1)$:** Minimum redundancy required to correct $t = \lfloor(d-1)/2\rfloor$ errors
- **Factor of 2:** Quantum codes must protect against both X and Z errors

#### Why "Singleton"?

Named after Richard Singleton who proved the classical version in 1964. The quantum version was established by Knill and Laflamme (1997).

---

### 2. Classical vs Quantum Singleton Bounds

#### Classical Singleton Bound

For a classical $[n, k, d]$ code:

$$k \leq n - d + 1$$

or equivalently:

$$d \leq n - k + 1$$

#### Quantum Singleton Bound

For a quantum $[[n, k, d]]$ code:

$$k \leq n - 2(d - 1) = n - 2d + 2$$

#### The Factor of 2 Difference

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| Bound | $k \leq n - d + 1$ | $k \leq n - 2d + 2$ |
| Redundancy | $n - k \geq d - 1$ | $n - k \geq 2(d-1)$ |
| Error types | Bit-flip only | Bit-flip AND phase-flip |
| Information | One syndrome | Two syndromes (X and Z) |

**Physical reason:** Quantum information requires protection against two independent error types. Classical information only needs protection against one.

#### Example Comparison

For $n = 7$, $d = 3$:

- **Classical:** $k \leq 7 - 3 + 1 = 5$ — The [7,4,3] Hamming code achieves $k = 4$
- **Quantum:** $k \leq 7 - 2(2) = 3$ — The [[7,1,3]] Steane code achieves $k = 1$

---

### 3. Derivation of the Quantum Singleton Bound

#### Method 1: Cleaning Lemma Approach

**Setup:** Consider an $[[n, k, d]]$ code encoding $k$ logical qubits.

**Step 1:** The code can correct any error on up to $t = \lfloor(d-1)/2\rfloor$ qubits.

**Step 2:** Divide the $n$ qubits into two sets:
- Set A: First $d-1$ qubits
- Set B: Remaining $n - (d-1)$ qubits

**Step 3 (Cleaning Lemma):** Any logical operator can be "cleaned" from set A, meaning there exists an equivalent logical operator supported entirely on set B.

**Step 4:** The code space on set B must contain all $2^k$ logical states distinguishably.

**Step 5:** The Hilbert space of B has dimension $2^{n-(d-1)}$.

**Conclusion:**
$$2^k \leq 2^{n-(d-1)} \cdot 2^{-(d-1)} = 2^{n-2(d-1)}$$

Therefore: $k \leq n - 2(d-1)$

#### Method 2: Entanglement Argument

**Setup:** Encode $k$ halves of EPR pairs into the code.

**Step 1:** The other halves (reference system R) are maximally entangled with the code.

**Step 2:** After erasure of $d-1$ qubits, the remaining qubits plus R must still contain all information.

**Step 3:** The entanglement entropy between R and the remaining system bounds the recoverable information.

**Step 4:** Information-theoretic analysis yields $k \leq n - 2(d-1)$.

---

### 4. Maximum Distance Separable (MDS) Codes

#### Definition

A quantum code is **MDS (Maximum Distance Separable)** if it achieves the Singleton bound with equality:

$$k = n - 2(d - 1)$$

#### Properties of MDS Codes

1. **Optimal encoding rate** for given distance
2. **Tight information storage** — no wasted redundancy
3. **Rare existence** — most codes don't achieve MDS

#### Known Quantum MDS Codes

| Code | $n$ | $k$ | $d$ | Check: $k = n - 2(d-1)$? |
|------|-----|-----|-----|--------------------------|
| [[5,1,3]] | 5 | 1 | 3 | $1 = 5 - 4$ ✓ |
| [[4,2,2]] | 4 | 2 | 2 | $2 = 4 - 2$ ✓ |
| [[6,0,4]] | 6 | 0 | 4 | $0 = 6 - 6$ ✓ |
| [[n,n-2,2]] | n | n-2 | 2 | $(n-2) = n - 2$ ✓ |

#### Non-MDS Codes

| Code | $n$ | $k$ | $d$ | Singleton allows | Gap |
|------|-----|-----|-----|------------------|-----|
| Steane [[7,1,3]] | 7 | 1 | 3 | $k \leq 3$ | 2 |
| Shor [[9,1,3]] | 9 | 1 | 3 | $k \leq 5$ | 4 |
| [[15,7,3]] | 15 | 7 | 3 | $k \leq 11$ | 4 |

---

### 5. The Perfect Five-Qubit Code

The [[5,1,3]] code is the smallest quantum code that:
1. Corrects arbitrary single-qubit errors
2. Achieves the Singleton bound (MDS)
3. Achieves the Hamming bound (perfect)

#### Stabilizer Generators

$$\begin{aligned}
g_1 &= XZZXI \\
g_2 &= IXZZX \\
g_3 &= XIXZZ \\
g_4 &= ZXIXZ
\end{aligned}$$

**Note:** Cyclic structure — each generator is a cyclic shift.

#### Logical Operators

$$\bar{X} = XXXXX, \quad \bar{Z} = ZZZZZ$$

Or minimum weight versions:
$$\bar{X} = ZYZYI \text{ (weight 4)}, \quad \bar{Z} = XYZYX \text{ (weight 5)}$$

Actually, minimum weight logical operators have weight 3:
$$\bar{X}_{min} = ZXZII \text{ (equivalent)}, \quad \bar{Z}_{min} = ZIZXI$$

#### Why It's Special

- **Smallest correcting code:** No $[[4,1,3]]$ code exists (violates Singleton)
- **Perfect:** Saturates Hamming bound
- **MDS:** Saturates Singleton bound
- **Non-CSS:** Mixed X/Z stabilizers (cannot be written as CSS)

---

### 6. Proving Code Impossibility

The Singleton bound is a powerful tool for proving certain codes cannot exist.

#### Example 1: No [[4,1,3]] Code

**Claim:** A [[4,1,3]] quantum code does not exist.

**Proof:**
Apply Singleton bound: $k \leq n - 2(d-1)$
$$1 \leq 4 - 2(3-1) = 4 - 4 = 0$$

This gives $1 \leq 0$, which is false. ∎

#### Example 2: No [[6,2,3]] Code

**Claim:** A [[6,2,3]] code cannot exist.

**Proof:**
$$2 \leq 6 - 2(2) = 2$$

The bound allows $k = 2$! So Singleton doesn't rule it out.

But we can show non-existence via other arguments (Hamming bound or explicit construction attempts fail).

#### Example 3: What's Possible for n = 9, d = 3?

Singleton bound: $k \leq 9 - 4 = 5$

- [[9,1,3]] Shor code exists ✓
- [[9,3,3]] might exist (need to check other bounds)
- [[9,5,3]] would be MDS (unknown if exists)

---

### 7. Beyond the Singleton Bound

#### Tighter Bounds

For specific code families, tighter bounds exist:

1. **Hamming Bound:** $\sum_{j=0}^{t} 3^j \binom{n}{j} \leq 2^{n-k}$
2. **Linear Programming Bounds:** Numerical optimization over weight distributions
3. **Shadow Bounds:** Constrain code existence via dual weight enumerators

#### Codes That "Beat" Singleton

Some extensions allow codes exceeding the standard Singleton bound:

1. **Entanglement-assisted codes:** Pre-shared entanglement between encoder/decoder
2. **Subsystem codes:** Encoding into subsystem rather than subspace
3. **Approximate codes:** Relaxed Knill-Laflamme conditions

---

## Quantum Mechanics Connection

### Information-Theoretic Perspective

The Singleton bound reflects fundamental quantum information constraints:

1. **No-cloning theorem:** Cannot create redundant copies freely
2. **Holevo bound:** Information accessible from quantum states is limited
3. **Entanglement monogamy:** Limits how information spreads across subsystems

### The "Factor of 2" as Quantum Signature

The doubling of redundancy requirements in quantum codes vs classical codes is a direct consequence of:

- **Complementarity:** X and Z observables are incompatible
- **Heisenberg uncertainty:** Cannot simultaneously measure position/momentum analogues
- **Superposition protection:** Must preserve both amplitude and phase

---

## Worked Examples

### Example 1: Verify Steane Code Satisfies Singleton

**Problem:** Show the [[7,1,3]] Steane code satisfies the quantum Singleton bound.

**Solution:**

The Singleton bound states: $k \leq n - 2(d-1)$

For the Steane code:
- $n = 7$, $k = 1$, $d = 3$

Check: $1 \leq 7 - 2(3-1) = 7 - 4 = 3$ ✓

The bound is satisfied. Moreover, the gap is $3 - 1 = 2$, showing Steane is not MDS.

---

### Example 2: Maximum k for Given n and d

**Problem:** What is the maximum number of logical qubits for a code with $n = 11$ physical qubits and distance $d = 5$?

**Solution:**

Singleton bound: $k \leq n - 2(d-1) = 11 - 2(4) = 11 - 8 = 3$

Maximum logical qubits: $k_{max} = 3$

An [[11,3,5]] code would be MDS if it exists.

---

### Example 3: Can a [[8,4,3]] Code Exist?

**Problem:** Determine if the Singleton bound allows a [[8,4,3]] quantum code.

**Solution:**

Singleton bound: $k \leq n - 2(d-1)$
$$4 \leq 8 - 2(2) = 8 - 4 = 4$$

The bound gives $4 \leq 4$ ✓

The Singleton bound *allows* this code. However, existence requires checking other bounds (Hamming) and attempting explicit construction.

Hamming bound for $d = 3$ ($t = 1$):
$$1 + 3n \leq 2^{n-k}$$
$$1 + 24 = 25 \leq 2^{8-4} = 16$$

This gives $25 \leq 16$, which is FALSE!

**Conclusion:** The Hamming bound rules out [[8,4,3]].

---

## Practice Problems

### Level 1: Direct Application

1. **Bound Verification:**
   Verify that the Shor [[9,1,3]] code satisfies the Singleton bound. Is it MDS?

2. **Maximum Distance:**
   For a [[10, 2, d]] code, what is the maximum possible distance $d$?

3. **Parameter Check:**
   Can a [[6,1,4]] code exist according to Singleton? What about [[6,0,4]]?

### Level 2: Intermediate

4. **Code Comparison:**
   List all possible $[[n, k, 3]]$ codes for $n \leq 9$ allowed by Singleton. Which actually exist?

5. **Classical Comparison:**
   A classical [15,11,3] Hamming code exists. What does the quantum Singleton bound say about [[15, k, 3]] codes?

6. **MDS Family:**
   Prove that $[[n, n-2, 2]]$ codes are MDS for all $n \geq 2$. What do these codes detect?

### Level 3: Challenging

7. **Cleaning Lemma:**
   For the [[5,1,3]] code, explicitly demonstrate the cleaning lemma by showing a logical operator can be moved from any single qubit to the remaining four.

8. **Entanglement Argument:**
   Using the entanglement-based derivation, show why $d-1$ erasures can be corrected if and only if the remaining qubits contain full information.

9. **Beyond Singleton:**
   Research subsystem codes. Explain how the [[9,1,3,3]] subsystem code relates to the Singleton bound (where the extra parameter is subsystem dimension).

---

## Computational Lab

### Exploring Code Bounds

```python
"""
Day 694 Computational Lab: Quantum Singleton Bound Analysis
Systematic exploration of quantum code parameter space
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Tuple, Optional

def singleton_bound(n: int, d: int) -> int:
    """
    Calculate maximum k allowed by quantum Singleton bound.

    Args:
        n: Number of physical qubits
        d: Code distance

    Returns:
        Maximum number of logical qubits k
    """
    k_max = n - 2 * (d - 1)
    return max(0, k_max)


def hamming_bound(n: int, d: int) -> int:
    """
    Calculate maximum k allowed by quantum Hamming bound.

    Args:
        n: Number of physical qubits
        d: Code distance

    Returns:
        Maximum k satisfying Hamming bound
    """
    t = (d - 1) // 2  # Correctable errors

    # Sum of sphere sizes
    sphere_size = sum(
        3**j * int(np.math.comb(n, j))
        for j in range(t + 1)
    )

    # 2^k * sphere_size <= 2^n
    # k <= n - log2(sphere_size)
    if sphere_size <= 0:
        return n

    k_max = n - np.ceil(np.log2(sphere_size))
    return int(max(0, k_max))


def classical_singleton(n: int, d: int) -> int:
    """Classical Singleton bound: k <= n - d + 1"""
    return max(0, n - d + 1)


def is_mds(n: int, k: int, d: int) -> bool:
    """Check if [[n,k,d]] code is MDS."""
    return k == n - 2 * (d - 1)


def code_allowed_by_singleton(n: int, k: int, d: int) -> bool:
    """Check if code parameters satisfy Singleton bound."""
    return k <= singleton_bound(n, d)


def code_allowed_by_hamming(n: int, k: int, d: int) -> bool:
    """Check if code parameters satisfy Hamming bound."""
    return k <= hamming_bound(n, d)


def analyze_code_parameters():
    """Analyze code parameters and bounds."""

    print("=" * 70)
    print("QUANTUM SINGLETON BOUND ANALYSIS")
    print("=" * 70)

    # Known quantum codes
    known_codes = [
        ("3-qubit bit-flip", 3, 1, 1),
        ("[[5,1,3]] Perfect", 5, 1, 3),
        ("[[4,2,2]]", 4, 2, 2),
        ("Steane [[7,1,3]]", 7, 1, 3),
        ("Shor [[9,1,3]]", 9, 1, 3),
        ("[[15,7,3]]", 15, 7, 3),
        ("[[6,0,4]]", 6, 0, 4),
    ]

    print("\n1. ANALYSIS OF KNOWN CODES")
    print("-" * 70)
    print(f"{'Code':<20} {'n':>3} {'k':>3} {'d':>3} {'Singleton':>10} {'Hamming':>10} {'MDS':>5}")
    print("-" * 70)

    for name, n, k, d in known_codes:
        s_max = singleton_bound(n, d)
        h_max = hamming_bound(n, d)
        mds = "Yes" if is_mds(n, k, d) else "No"

        s_status = "✓" if k <= s_max else "✗"
        h_status = "✓" if k <= h_max else "✗"

        print(f"{name:<20} {n:>3} {k:>3} {d:>3} {s_status} k≤{s_max:<5} {h_status} k≤{h_max:<5} {mds:>5}")

    # Compare quantum vs classical Singleton
    print("\n2. QUANTUM VS CLASSICAL SINGLETON BOUND")
    print("-" * 70)
    print(f"{'n':>3} {'d':>3} {'Classical k≤':>14} {'Quantum k≤':>14} {'Difference':>12}")
    print("-" * 70)

    for n in range(5, 16):
        for d in [3, 5]:
            if 2*(d-1) > n:
                continue
            classical = classical_singleton(n, d)
            quantum = singleton_bound(n, d)
            diff = classical - quantum
            print(f"{n:>3} {d:>3} {classical:>14} {quantum:>14} {diff:>12}")

    # Search for MDS codes
    print("\n3. POSSIBLE MDS CODES (Singleton-saturating)")
    print("-" * 70)
    print("Codes that would be MDS if they exist:")
    print(f"{'[[n,k,d]]':<15} {'Status':>20}")
    print("-" * 70)

    mds_candidates = []
    for n in range(4, 16):
        for d in range(2, (n+2)//2 + 1):
            k = n - 2*(d-1)
            if k >= 0:
                mds_candidates.append((n, k, d))

    # Known MDS codes
    known_mds = {(5,1,3), (4,2,2), (6,0,4), (2,0,2), (3,1,2)}

    for n, k, d in mds_candidates[:15]:
        status = "EXISTS" if (n,k,d) in known_mds else "Unknown"
        print(f"[[{n},{k},{d}]]{'':>7} {status:>20}")


def plot_singleton_landscape():
    """Visualize the quantum Singleton bound landscape."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: k_max vs n for different d
    ax1 = axes[0]
    n_values = np.arange(3, 21)

    for d in [2, 3, 5, 7]:
        k_max = [singleton_bound(n, d) for n in n_values]
        ax1.plot(n_values, k_max, 'o-', label=f'd = {d}', markersize=6)

    ax1.set_xlabel('Number of physical qubits n', fontsize=12)
    ax1.set_ylabel('Maximum logical qubits k', fontsize=12)
    ax1.set_title('Quantum Singleton Bound: k ≤ n - 2(d-1)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_values[::2])

    # Plot 2: Quantum vs Classical comparison
    ax2 = axes[1]

    d_values = np.arange(2, 10)
    n_fixed = 15

    classical_k = [classical_singleton(n_fixed, d) for d in d_values]
    quantum_k = [singleton_bound(n_fixed, d) for d in d_values]

    width = 0.35
    x = np.arange(len(d_values))

    bars1 = ax2.bar(x - width/2, classical_k, width, label='Classical [n,k,d]', color='steelblue')
    bars2 = ax2.bar(x + width/2, quantum_k, width, label='Quantum [[n,k,d]]', color='coral')

    ax2.set_xlabel('Code distance d', fontsize=12)
    ax2.set_ylabel('Maximum k', fontsize=12)
    ax2.set_title(f'Classical vs Quantum Singleton Bound (n = {n_fixed})', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(d_values)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('singleton_bound_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: singleton_bound_analysis.png")


def find_code_parameters():
    """Find all valid code parameters within bounds."""

    print("\n" + "=" * 70)
    print("VALID CODE PARAMETERS SEARCH")
    print("=" * 70)

    print("\nAll [[n,k,d]] with n ≤ 10, d = 3, satisfying both bounds:")
    print("-" * 50)

    valid_codes = []

    for n in range(5, 11):
        d = 3
        s_max = singleton_bound(n, d)
        h_max = hamming_bound(n, d)
        k_max = min(s_max, h_max)

        for k in range(1, k_max + 1):
            if code_allowed_by_singleton(n, k, d) and code_allowed_by_hamming(n, k, d):
                mds = "MDS" if is_mds(n, k, d) else ""
                valid_codes.append((n, k, d, mds))
                print(f"  [[{n},{k},{d}]] {mds}")

    print(f"\nTotal valid parameter sets: {len(valid_codes)}")


def demonstrate_factor_of_two():
    """Demonstrate why quantum codes need 2x redundancy."""

    print("\n" + "=" * 70)
    print("THE FACTOR OF 2: WHY QUANTUM NEEDS MORE REDUNDANCY")
    print("=" * 70)

    print("""
    Classical Code [n, k, d]:
    ─────────────────────────
    • Protects against: Bit-flip errors
    • Redundancy needed: n - k ≥ d - 1
    • Information structure: Single syndrome

    Example: [7, 4, 3] Hamming
    • 4 information bits, 3 parity bits
    • Corrects 1 bit-flip error

    Quantum Code [[n, k, d]]:
    ─────────────────────────
    • Protects against: Bit-flip AND phase-flip errors
    • Redundancy needed: n - k ≥ 2(d - 1)
    • Information structure: Two independent syndromes

    Example: [[7, 1, 3]] Steane
    • 1 logical qubit, 6 stabilizer qubits
    • Corrects 1 arbitrary error (X, Y, or Z)

    Why 2x?
    ───────
    1. X errors produce Z-syndrome (detected by X-stabilizers)
    2. Z errors produce X-syndrome (detected by Z-stabilizers)
    3. Y = iXZ produces both syndromes

    Each logical qubit must be protected in BOTH X and Z bases,
    effectively requiring double the classical redundancy.
    """)


if __name__ == "__main__":
    analyze_code_parameters()
    find_code_parameters()
    demonstrate_factor_of_two()

    print("\n" + "=" * 70)
    print("Generating visualization...")
    print("=" * 70)
    plot_singleton_landscape()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Quantum Singleton Bound | $k \leq n - 2(d-1)$ |
| Classical Singleton Bound | $k \leq n - d + 1$ |
| MDS Condition | $k = n - 2(d-1)$ (equality) |
| Minimum redundancy | $n - k \geq 2(d-1)$ |
| Factor of 2 origin | Protecting X and Z independently |

### Main Takeaways

1. **Fundamental Limit:** The Singleton bound establishes the maximum information rate for a given error correction capability
2. **Factor of 2:** Quantum codes require double the redundancy of classical codes
3. **MDS Codes:** Codes achieving the bound with equality are optimal (but rare)
4. **[[5,1,3]]:** The smallest code correcting arbitrary single-qubit errors, and it's MDS
5. **Design Tool:** Use the bound to quickly eliminate impossible code parameters

---

## Daily Checklist

- [ ] Can state the quantum Singleton bound
- [ ] Understand derivation via cleaning lemma
- [ ] Know why quantum bound is 2x classical
- [ ] Can identify MDS codes
- [ ] Can use bound to prove code impossibility
- [ ] Know the [[5,1,3]] code is MDS

---

## Preview: Day 695

Tomorrow we explore the **Quantum Hamming Bound** — a sphere-packing argument that provides a different constraint on code parameters. We'll learn:

- Sphere-packing interpretation for quantum codes
- Perfect quantum codes
- Why the [[5,1,3]] code is perfect
- Relationship between Singleton and Hamming bounds

The Hamming bound often provides tighter constraints than Singleton, especially for large codes.

---

*"The Singleton bound is the first question to ask about any proposed code — if it violates the bound, the code cannot exist."*
