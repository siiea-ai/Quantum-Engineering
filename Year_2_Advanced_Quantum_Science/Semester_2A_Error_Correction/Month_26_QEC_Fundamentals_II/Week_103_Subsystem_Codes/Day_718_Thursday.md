# Day 718: Subsystem Code Properties

## Overview

**Date:** Day 718 of 1008
**Week:** 103 (Subsystem Codes)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Distance, Error Correction Conditions, and Fundamental Bounds

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Distance definitions and error correction |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Subsystem code bounds |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Implementation and comparisons |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define and compute** the distance of a subsystem code
2. **State the subsystem error correction conditions** (generalized Knill-Laflamme)
3. **Prove** the relationship between bare/dressed operators and distance
4. **Apply the singleton bound** to subsystem codes
5. **Compare** subsystem vs stabilizer code parameters
6. **Explain** the gauge-distance trade-off

---

## Core Content

### 1. Distance in Subsystem Codes

#### Revisiting the Definition

For a subsystem code $\mathcal{C} = A \otimes B$:

**Distance $d$:** The minimum weight of an operator that acts non-trivially on $A$ while commuting with the gauge group $\mathcal{G}$.

$$d = \min\{|E| : E \in N(\mathcal{G}) \setminus C(\bar{X}_j, \bar{Z}_j \text{ for all } j) \cdot \mathcal{G}\}$$

More precisely:
$$d = \min\{|E| : E \text{ implements non-identity logical on } A\}$$

#### Bare vs Dressed Distance

**Bare distance $d_{\text{bare}}$:** Minimum weight of bare logical operators
$$d_{\text{bare}} = \min_j \min(|\bar{X}_j|, |\bar{Z}_j|)$$

**Dressed distance $d_{\text{dressed}}$:** Minimum weight after dressing with gauge
$$d_{\text{dressed}} = \min_j \min_{g \in \mathcal{G}}(|\bar{X}_j \cdot g|, |\bar{Z}_j \cdot g|)$$

**Key inequality:**
$$d_{\text{dressed}} \leq d_{\text{bare}}$$

The code distance is $d = d_{\text{dressed}}$.

---

### 2. Subsystem Error Correction Conditions

#### Generalized Knill-Laflamme Conditions

For a subsystem code to correct error set $\mathcal{E}$:

**Theorem (Subsystem Knill-Laflamme):**
A subsystem code $\mathcal{C} = A \otimes B$ can correct errors $\mathcal{E}$ if and only if for all $E_a, E_b \in \mathcal{E}$:

$$P_{\mathcal{C}} E_a^\dagger E_b P_{\mathcal{C}} = I_A \otimes B_{ab}$$

where $P_{\mathcal{C}}$ is the projector onto the code space and $B_{ab}$ acts only on $B$.

#### Comparison to Stabilizer Codes

| Property | Stabilizer Code | Subsystem Code |
|----------|-----------------|----------------|
| Error condition | $P E_a^\dagger E_b P = c_{ab} P$ | $P E_a^\dagger E_b P = I_A \otimes B_{ab}$ |
| Scalar | Must be scalar | Can be operator on $B$ |
| Gauge freedom | None | Errors absorbed by gauge |

#### Syndrome-Based Condition

**Equivalent condition:** For all $E_a, E_b \in \mathcal{E}$, either:
1. $E_a^\dagger E_b \in \mathcal{G}$ (errors equivalent mod gauge), or
2. $E_a^\dagger E_b$ anticommutes with some stabilizer $s \in \mathcal{S}$

This means: errors are either gauge-equivalent or distinguishable.

---

### 3. Proof of Error Correction Capability

#### Theorem: Distance Implies Correction

**Claim:** A subsystem code with distance $d$ can correct any error of weight $\lfloor(d-1)/2\rfloor$.

**Proof:**

Let $E_a, E_b$ be errors with $|E_a|, |E_b| \leq t = \lfloor(d-1)/2\rfloor$.

Then $|E_a^\dagger E_b| \leq |E_a| + |E_b| \leq 2t < d$.

**Case 1:** $E_a^\dagger E_b \in \mathcal{G}$

Then errors are gauge-equivalent. The logical information in $A$ is unaffected.
$$P_{\mathcal{C}} E_a^\dagger E_b P_{\mathcal{C}} = I_A \otimes \tilde{B}$$

**Case 2:** $E_a^\dagger E_b \notin \mathcal{G}$

Since $|E_a^\dagger E_b| < d$ and $E_a^\dagger E_b$ is not in $\mathcal{G}$, it must anticommute with some stabilizer (by definition of distance). The syndromes are distinct.

In either case, the subsystem error correction condition is satisfied. $\square$

---

### 4. The Singleton Bound for Subsystem Codes

#### Statement

**Theorem (Subsystem Singleton Bound):**
For an $[[n, k, r, d]]$ subsystem code:
$$k + r \leq n - 2(d-1)$$

Or equivalently:
$$d \leq \frac{n - k - r}{2} + 1$$

#### Proof Sketch

The code space has dimension $2^{k+r}$ (logical + gauge).

To correct $(d-1)/2$ errors, we need to distinguish error syndromes on overlapping supports.

The stabilizer group has $n - k - r$ independent generators.

By counting arguments similar to the stabilizer Singleton bound:
$$2^{k+r} \cdot 2^{n-k-r-(d-1)} \leq 2^n$$

Simplifying gives the bound.

#### Implications

- More gauge qubits ($r$) → lower distance possible
- Trade-off between gauge freedom and error correction
- Can't have unlimited gauge without sacrificing protection

---

### 5. The Gauge-Distance Trade-off

#### The Fundamental Trade-off

For fixed $n$ and $k$:

$$\uparrow r \text{ (more gauge)} \implies \downarrow d \text{ (lower distance)}$$

**Physical interpretation:**
- More gauge freedom = more operators that don't affect logical space
- But these operators might help errors "sneak through"

#### Example: Bacon-Shor Family

For $m \times n$ Bacon-Shor codes with $m \leq n$:

| $m$ | $n$ | Physical | Logical | Gauge | Distance |
|-----|-----|----------|---------|-------|----------|
| 2 | 2 | 4 | 1 | 1 | 2 |
| 3 | 3 | 9 | 1 | 4 | 3 |
| 4 | 4 | 16 | 1 | 9 | 4 |
| 5 | 5 | 25 | 1 | 16 | 5 |

Pattern: $r = (m-1)(n-1)$, $d = \min(m,n)$

As $m,n$ grow, gauge qubits grow as $(m-1)(n-1)$ but distance only as $\min(m,n)$.

---

### 6. Comparison: Subsystem vs Stabilizer Codes

#### Parameter Comparison

**Stabilizer code:** $[[n, k, d]]$
- $n - k$ stabilizer generators
- Code space: $2^k$ dimensional
- All encoded qubits are logical

**Subsystem code:** $[[n, k, r, d]]$
- $n - k - r$ stabilizer generators
- Code space: $2^{k+r}$ dimensional
- Only $k$ qubits carry protected information

#### When to Use Each

**Prefer stabilizer codes when:**
- Need maximum logical qubits per physical qubit
- High-distance codes exist (e.g., surface codes)
- Measurement weight is acceptable

**Prefer subsystem codes when:**
- Low-weight measurements are critical
- Fault tolerance constraints are severe
- Gauge flexibility simplifies implementation

#### The "No Free Lunch" Principle

From $[[n, k, r, d]]$ subsystem to $[[n, k+r, d']]$ stabilizer (gauge fixing):
- Gain: more logical qubits
- Cost: potentially harder measurements, loss of gauge flexibility

From $[[n, k, d]]$ stabilizer to $[[n, k', r', d]]$ subsystem (gauge unfixing):
- Gain: simpler measurements
- Cost: fewer logical qubits ($k' < k$)

---

### 7. Error Detection vs Correction

#### Detection Distance

**Detection distance:** Minimum weight operator causing logical error
$$d_{\text{detect}} = \min\{|E| : E \in N(\mathcal{G}), E \notin \mathcal{G}, [E, \bar{X}_j] = 0 \text{ or } [E, \bar{Z}_j] = 0 \text{ for some } j\}$$

For symmetric codes: $d_{\text{detect}} = d_{\text{correct}} = d$

#### Asymmetric Error Handling

Some subsystem codes have different X and Z distances:

**X-distance:** Minimum weight Z-type logical
**Z-distance:** Minimum weight X-type logical

Asymmetric codes useful when error rates differ by type.

---

## Worked Examples

### Example 1: Verify Bacon-Shor Distance

**Problem:** Prove that the $3 \times 3$ Bacon-Shor code has distance 3.

**Solution:**

**Step 1: Find minimum weight logical**

Bare $\bar{X}$: X on any row, weight 3
Bare $\bar{Z}$: Z on any column, weight 3

**Step 2: Check if dressing reduces weight**

Can we dress $\bar{X} = X_{1,1}X_{1,2}X_{1,3}$ with gauge operators to reduce weight?

X-gauge operators: $X_{i,j}X_{i,j+1}$ (horizontal)
Z-gauge operators: $Z_{i,j}Z_{i+1,j}$ (vertical)

Multiplying $\bar{X}$ by X-gauge doesn't reduce weight (adds more X's).
Multiplying by Z-gauge: $\bar{X} \cdot Z_{1,1}Z_{2,1} = X_{1,1}X_{1,2}X_{1,3}Z_{1,1}Z_{2,1}$
This has weight 5 (increases!).

**Step 3: Verify no weight-2 logical exists**

Any weight-2 Pauli must be:
- Two X's: Either horizontal (= gauge) or not on same row (doesn't span column)
- Two Z's: Either vertical (= gauge) or not on same column (doesn't span row)
- X and Z: Cannot implement logical (different types)

No weight-2 operator can be logical.

**Step 4: Conclusion**

$d = \min(3, 3) = 3$ ✓

---

### Example 2: Singleton Bound Check

**Problem:** Does the $[[9, 1, 4, 3]]$ Bacon-Shor code saturate the Singleton bound?

**Solution:**

**Singleton bound:**
$$k + r \leq n - 2(d-1)$$
$$1 + 4 \leq 9 - 2(3-1)$$
$$5 \leq 9 - 4 = 5$$

Yes! The Bacon-Shor code saturates the bound with equality.

This means it's an **optimal** subsystem code for these parameters.

---

### Example 3: Subsystem Error Correction Condition

**Problem:** Show that the $[[4, 1, 1, 2]]$ subsystem code satisfies the error correction conditions for single-qubit errors.

**Solution:**

**Setup:**
- Stabilizers: $S = \langle X_1X_2X_3X_4, Z_1Z_2Z_3Z_4 \rangle$
- Gauge: $\mathcal{G} = \langle X_1X_2, X_3X_4, Z_1Z_3, Z_2Z_4 \rangle$
- Errors: $\mathcal{E} = \{I, X_i, Z_i, Y_i : i = 1,2,3,4\}$

**Check condition:** For all $E_a, E_b \in \mathcal{E}$:

**Case: $E_a = E_b = X_1$**
$E_a^\dagger E_b = X_1 X_1 = I \in \mathcal{G}$ ✓

**Case: $E_a = X_1, E_b = X_2$**
$E_a^\dagger E_b = X_1 X_2 \in \mathcal{G}$ ✓ (it's a gauge generator!)

**Case: $E_a = X_1, E_b = X_3$**
$E_a^\dagger E_b = X_1 X_3$
Check: Does this anticommute with a stabilizer?
$[X_1X_3, Z_1Z_2Z_3Z_4] = $ anticommutes (odd overlap) ✓

**Case: $E_a = X_1, E_b = Z_1$**
$E_a^\dagger E_b = X_1 Z_1 = -iY_1$
Check: $[Y_1, X_1X_2X_3X_4] = $ anticommutes ✓

All pairs checked: errors are either gauge-equivalent or have distinct syndromes.

The code corrects up to $\lfloor(2-1)/2\rfloor = 0$ errors — it **detects** single errors.

---

## Practice Problems

### Direct Application

1. **Problem 1:** For a $4 \times 3$ Bacon-Shor code, compute $k$, $r$, and $d$.

2. **Problem 2:** Verify that the $[[4, 1, 1, 2]]$ code satisfies the Singleton bound.

3. **Problem 3:** If a subsystem code has $n = 15$, $k = 1$, $d = 5$, what is the maximum number of gauge qubits?

### Intermediate

4. **Problem 4:** Prove that for any subsystem code, $d_{\text{dressed}} \leq d_{\text{bare}}$.

5. **Problem 5:** A code has gauge group generated by weight-2 operators and distance 5. What is the minimum dressed logical weight?

6. **Problem 6:** Design a subsystem code with parameters $[[6, 1, r, 2]]$ and find the maximum $r$.

### Challenging

7. **Problem 7:** Prove that if a stabilizer code has all weight-$w$ stabilizers, converting it to a subsystem code with weight-2 gauge operators must reduce the distance.

8. **Problem 8:** Show that no $[[n, k, r, d]]$ subsystem code can have $d > n/2 + 1$ for $k \geq 1$.

9. **Problem 9:** Derive the quantum Hamming bound for subsystem codes.

---

## Computational Lab

```python
"""
Day 718: Subsystem Code Properties
Analysis of distance, bounds, and error correction conditions
"""

import numpy as np
from itertools import combinations, product
from typing import List, Tuple, Set

def pauli_weight(pauli: np.ndarray) -> int:
    """Compute weight of a Pauli operator (number of non-identity sites)."""
    # pauli is (2n,) binary vector: [x | z]
    n = len(pauli) // 2
    x_part = pauli[:n]
    z_part = pauli[n:]
    # Non-identity where x OR z is 1
    return np.sum((x_part | z_part) > 0)

def symplectic_inner_product(a: np.ndarray, b: np.ndarray) -> int:
    """Compute symplectic inner product [a, b] = a_x · b_z + a_z · b_x mod 2."""
    n = len(a) // 2
    return (np.dot(a[:n], b[n:]) + np.dot(a[n:], b[:n])) % 2

def commutes(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if two Paulis commute."""
    return symplectic_inner_product(a, b) == 0

class SubsystemCode:
    """Class to analyze subsystem code properties."""

    def __init__(self, n: int, stabilizers: List[np.ndarray],
                 gauge_generators: List[np.ndarray]):
        """
        Initialize subsystem code.

        Parameters:
        -----------
        n : int
            Number of physical qubits
        stabilizers : List[np.ndarray]
            Stabilizer generators as binary vectors
        gauge_generators : List[np.ndarray]
            Gauge generators (including stabilizers)
        """
        self.n = n
        self.stabilizers = [np.array(s) for s in stabilizers]
        self.gauge_generators = [np.array(g) for g in gauge_generators]

        # Compute parameters
        self.num_stabilizers = len(stabilizers)
        self.num_gauge = len(gauge_generators)
        self.k = n - self.num_gauge  # Rough estimate for logical qubits

    def is_in_gauge_group(self, pauli: np.ndarray, max_products: int = 100) -> bool:
        """
        Check if a Pauli is in the gauge group (approximately).
        For exact check, would need to enumerate all products.
        """
        # Check if it's one of the generators
        for g in self.gauge_generators:
            if np.array_equal(pauli, g):
                return True

        # Check products of pairs
        for g1, g2 in combinations(self.gauge_generators, 2):
            prod = (g1 + g2) % 2  # XOR for group operation
            if np.array_equal(pauli, prod):
                return True

        return False

    def has_distinct_syndrome(self, error: np.ndarray) -> bool:
        """Check if error anticommutes with some stabilizer."""
        for s in self.stabilizers:
            if not commutes(error, s):
                return True
        return False

    def check_error_pair(self, e1: np.ndarray, e2: np.ndarray) -> Tuple[bool, str]:
        """
        Check if error pair satisfies correction condition.

        Returns:
        --------
        (satisfies, reason)
        """
        # Compute E1† E2 = E1 * E2 (in binary representation)
        combined = (e1 + e2) % 2

        if self.is_in_gauge_group(combined):
            return True, "Gauge equivalent"

        if self.has_distinct_syndrome(combined):
            return True, "Distinct syndromes"

        return False, "FAILS: Neither gauge equiv nor distinguishable"

    def estimate_distance(self, max_weight: int = None) -> int:
        """
        Estimate code distance by searching for logical operators.
        """
        if max_weight is None:
            max_weight = self.n

        # Generate all Paulis up to given weight
        for w in range(1, max_weight + 1):
            for positions in combinations(range(self.n), w):
                # Try all Pauli types on these positions
                for pauli_types in product([1, 2, 3], repeat=w):  # 1=X, 2=Z, 3=Y
                    pauli = np.zeros(2 * self.n, dtype=int)
                    for pos, ptype in zip(positions, pauli_types):
                        if ptype == 1:  # X
                            pauli[pos] = 1
                        elif ptype == 2:  # Z
                            pauli[self.n + pos] = 1
                        else:  # Y = XZ
                            pauli[pos] = 1
                            pauli[self.n + pos] = 1

                    # Check if this is a non-trivial logical
                    # Must commute with all stabilizers
                    commutes_all = all(commutes(pauli, s) for s in self.stabilizers)

                    if commutes_all and not self.is_in_gauge_group(pauli):
                        # Found a logical operator!
                        return w

        return max_weight + 1  # Distance exceeds search range

def create_bacon_shor_code(m: int, n: int) -> SubsystemCode:
    """Create an m x n Bacon-Shor code."""
    num_qubits = m * n

    def qubit_index(i, j):
        """Convert (row, col) to linear index."""
        return i * n + j

    # Create gauge generators
    gauge_gens = []

    # X-type gauge: XX on horizontal pairs
    for i in range(m):
        for j in range(n - 1):
            g = np.zeros(2 * num_qubits, dtype=int)
            g[qubit_index(i, j)] = 1  # X on (i,j)
            g[qubit_index(i, j+1)] = 1  # X on (i,j+1)
            gauge_gens.append(g)

    # Z-type gauge: ZZ on vertical pairs
    for i in range(m - 1):
        for j in range(n):
            g = np.zeros(2 * num_qubits, dtype=int)
            g[num_qubits + qubit_index(i, j)] = 1  # Z on (i,j)
            g[num_qubits + qubit_index(i+1, j)] = 1  # Z on (i+1,j)
            gauge_gens.append(g)

    # Create stabilizer generators (products of gauge)
    stabilizers = []

    # X-type stabilizers: product of X-gauges in column pairs
    for j in range(n - 1):
        s = np.zeros(2 * num_qubits, dtype=int)
        for i in range(m):
            s[qubit_index(i, j)] = 1
            s[qubit_index(i, j+1)] = 1
        # Note: X*X = I, so pairs cancel; need to be careful
        # Actually, it's product of all X_{i,j}X_{i,j+1} for fixed j
        # This gives X on columns j and j+1 for all rows
        stabilizers.append(s)

    # Z-type stabilizers: product of Z-gauges in row pairs
    for i in range(m - 1):
        s = np.zeros(2 * num_qubits, dtype=int)
        for j in range(n):
            s[num_qubits + qubit_index(i, j)] = 1
            s[num_qubits + qubit_index(i+1, j)] = 1
        stabilizers.append(s)

    return SubsystemCode(num_qubits, stabilizers, gauge_gens)

def singleton_bound(n: int, k: int, r: int) -> int:
    """Compute maximum distance from Singleton bound."""
    return (n - k - r) // 2 + 1

def verify_singleton(n: int, k: int, r: int, d: int) -> bool:
    """Check if parameters satisfy Singleton bound."""
    return k + r <= n - 2 * (d - 1)

# Main demonstration
print("=" * 60)
print("Subsystem Code Properties Analysis")
print("=" * 60)

# Example 1: Bacon-Shor codes
print("\n1. Bacon-Shor Code Parameters:")
print("-" * 40)

for m, n_col in [(2, 2), (3, 3), (4, 4), (3, 5)]:
    n_qubits = m * n_col
    k = 1  # Always 1 logical qubit
    r = (m - 1) * (n_col - 1)  # Gauge qubits
    d = min(m, n_col)  # Distance

    # Check Singleton
    satisfies = verify_singleton(n_qubits, k, r, d)
    d_max = singleton_bound(n_qubits, k, r)

    print(f"\n{m}×{n_col} Bacon-Shor: [[{n_qubits}, {k}, {r}, {d}]]")
    print(f"  Singleton bound: d ≤ {d_max}")
    print(f"  Satisfies bound: {satisfies}")
    print(f"  Saturates bound: {d == d_max}")

# Example 2: Error correction analysis for 2×2 code
print("\n" + "=" * 60)
print("2. Error Correction Analysis: [[4,1,1,2]] Code")
print("-" * 40)

code_2x2 = create_bacon_shor_code(2, 2)

print(f"Number of physical qubits: {code_2x2.n}")
print(f"Number of stabilizers: {code_2x2.num_stabilizers}")
print(f"Number of gauge generators: {code_2x2.num_gauge}")

# Check some error pairs
print("\nError pair analysis:")
single_x_errors = []
for i in range(4):
    e = np.zeros(8, dtype=int)
    e[i] = 1  # X on qubit i
    single_x_errors.append((f"X_{i+1}", e))

for name1, e1 in single_x_errors[:2]:
    for name2, e2 in single_x_errors[:2]:
        satisfies, reason = code_2x2.check_error_pair(e1, e2)
        print(f"  {name1}, {name2}: {reason}")

# Example 3: Distance estimation
print("\n" + "=" * 60)
print("3. Distance Estimation")
print("-" * 40)

print("\nSearching for minimum weight logical operators...")
for m, n_col in [(2, 2), (3, 3)]:
    code = create_bacon_shor_code(m, n_col)
    d_found = code.estimate_distance(max_weight=min(m, n_col) + 1)
    d_expected = min(m, n_col)
    print(f"\n{m}×{n_col} Bacon-Shor:")
    print(f"  Found distance: {d_found}")
    print(f"  Expected: {d_expected}")
    print(f"  Match: {'✓' if d_found == d_expected else '✗'}")

# Example 4: Singleton bound exploration
print("\n" + "=" * 60)
print("4. Singleton Bound Exploration")
print("-" * 40)

print("\nFor n=9, k=1, varying r:")
n = 9
k = 1
for r in range(9):
    d_max = singleton_bound(n, k, r)
    if d_max > 0:
        print(f"  r={r}: d_max = {d_max}, code = [[{n},{k},{r},{d_max}]]")

print("\nBacon-Shor achieves [[9,1,4,3]]")
print(f"Singleton gives: d ≤ {singleton_bound(9, 1, 4)}")
print("This saturates the bound!")

# Example 5: Gauge-distance trade-off visualization data
print("\n" + "=" * 60)
print("5. Gauge-Distance Trade-off")
print("-" * 40)

print("\nFor fixed n=16 (4×4 lattice), varying shapes:")
print("| Shape | r (gauge) | d (distance) | r × d |")
print("|-------|-----------|--------------|-------|")
for m in [2, 4, 8, 16]:
    n_col = 16 // m
    r = (m - 1) * (n_col - 1)
    d = min(m, n_col)
    print(f"| {m}×{n_col:2d}  |    {r:3d}    |      {d:2d}      |  {r*d:3d}  |")

print("\nNote: Square lattices (4×4) optimize the distance!")
print("Trade-off: more gauge qubits ≠ more protection")
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Code distance** | Minimum weight logical operator (dressed) |
| **Bare vs dressed** | $d_{\text{dressed}} \leq d_{\text{bare}}$ |
| **Subsystem K-L** | $P E_a^\dagger E_b P = I_A \otimes B_{ab}$ |
| **Singleton bound** | $k + r \leq n - 2(d-1)$ |
| **Gauge trade-off** | More gauge → potentially lower distance |

### Key Equations

$$\boxed{d = \min_{L \text{ logical}} |L \cdot g|, \quad g \in \mathcal{G}}$$

$$\boxed{k + r \leq n - 2(d-1) \quad \text{(Singleton)}}$$

$$\boxed{P_{\mathcal{C}} E_a^\dagger E_b P_{\mathcal{C}} = I_A \otimes B_{ab} \quad \text{(Subsystem K-L)}}$$

### Main Takeaways

1. **Distance** is determined by minimum-weight dressed logical operators
2. **Subsystem Knill-Laflamme** allows operators on gauge subsystem (not just scalars)
3. **Singleton bound** constrains $k + r + d$ trade-offs
4. **Gauge freedom** doesn't come free — affects achievable distance
5. **Bacon-Shor codes** saturate the Singleton bound (optimal)

---

## Daily Checklist

- [ ] I can define and compute subsystem code distance
- [ ] I understand the subsystem Knill-Laflamme conditions
- [ ] I can apply the Singleton bound to subsystem codes
- [ ] I understand the gauge-distance trade-off
- [ ] I can compare subsystem and stabilizer code parameters
- [ ] I completed the computational lab

---

## Preview: Day 719

Tomorrow we study **Advantages of Subsystem Codes**, including:
- Reduced measurement weight
- Fault-tolerance benefits
- Single-shot error correction
- Practical implementation advantages
