# Day 734: Logical Operators & Distance

## Overview

**Day:** 734 of 1008
**Week:** 105 (Binary Representation & F₂ Linear Algebra)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Computing Logical Operators and Code Distance

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Logical operator theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Distance computation |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define** the centralizer and normalizer of a stabilizer group
2. **Compute** logical operators from the symplectic complement
3. **Understand** the relationship between logicals and code space
4. **Calculate** code distance from minimum weight
5. **Apply** bounds: quantum Singleton, quantum Hamming
6. **Implement** algorithms for distance computation

---

## Core Content

### Centralizer and Normalizer

**Definitions:**
For stabilizer group S ⊆ Pₙ:

**Centralizer:**
$$C(S) = \{P \in \mathcal{P}_n : PS = SP \text{ for all } S \in \mathcal{S}\}$$

All Paulis that commute with every stabilizer.

**Normalizer:**
$$N(S) = \{P \in \mathcal{P}_n : PSP^{-1} \in \mathcal{S} \text{ for all } S \in \mathcal{S}\}$$

All Paulis that preserve S under conjugation.

**Key Theorem:**
For stabilizer codes, $C(S) = N(S)$.

### Binary Representation of Centralizer

In binary form, the centralizer is the **symplectic complement**:
$$C(S)_{\text{bin}} = \{v \in \mathbb{F}_2^{2n} : \langle v, s \rangle_s = 0 \text{ for all } s \in S_{\text{bin}}\}$$

Using the parity check matrix H:
$$\boxed{C(S)_{\text{bin}} = \ker(H \Omega)^T}$$

### Structure of the Centralizer

**Dimension:**
$$\dim(C(S)_{\text{bin}}) = 2n - \text{rank}(H) = 2n - (n-k) = n + k$$

**Decomposition:**
$$C(S)_{\text{bin}} = S_{\text{bin}} \oplus L$$

where L is the **logical operator space** of dimension 2k.

### Logical Operators

**Definition:**
Logical operators are elements of $C(S) \setminus S$ that act non-trivially on the code space.

For an [[n, k, d]] code:
- k logical X operators: $\bar{X}_1, \ldots, \bar{X}_k$
- k logical Z operators: $\bar{Z}_1, \ldots, \bar{Z}_k$

**Commutation Relations:**
$$[\bar{X}_i, \bar{X}_j] = 0, \quad [\bar{Z}_i, \bar{Z}_j] = 0$$
$$\{\bar{X}_i, \bar{Z}_j\} = 2\delta_{ij}\bar{X}_i\bar{Z}_j$$

(Anticommute if i = j, commute otherwise)

### Finding Logical Operators

**Algorithm:**
1. Compute $C = \ker(H\Omega)^T$ (centralizer basis)
2. Reduce to find vectors not in row space of H
3. Pair into X-type and Z-type logicals

**Alternative Method (Generator Matrix):**
Construct the generator matrix G ∈ F₂^{2k × 2n} where:
$$G \Omega H^T = 0$$
$$G \Omega G^T = \begin{pmatrix} 0 & I_k \\ I_k & 0 \end{pmatrix}$$

### Code Distance

**Definition:**
The distance d of an [[n, k, d]] code is the minimum weight of any non-trivial logical operator:

$$\boxed{d = \min_{P \in C(S) \setminus S} \text{wt}(P)}$$

Equivalently, d is the minimum weight of any Pauli that:
1. Commutes with all stabilizers
2. Is not itself a stabilizer

**Interpretation:**
- d errors are required to cause a logical error
- Any d-1 errors can be detected
- Any ⌊(d-1)/2⌋ errors can be corrected

### Computing Distance

**Brute Force:**
Enumerate all vectors in C(S) \ S and find minimum weight.

Complexity: O(2^{n+k}) — exponential!

**Improvement for Small d:**
Check all weight-1 Paulis, then weight-2, etc.

For each weight w:
1. Generate all weight-w Paulis
2. Check if each commutes with all stabilizers
3. Check if it's in S
4. If found in C(S) \ S, return d = w

### Quantum Singleton Bound

**Theorem:**
For any [[n, k, d]] stabilizer code:
$$\boxed{k \leq n - 2(d-1)}$$

Or equivalently: $2k + 2d \leq n + 2$.

**MDS (Maximum Distance Separable) Codes:**
Codes achieving equality are called MDS codes.

**Examples:**
- [[5,1,3]]: k = 1, n - 2(d-1) = 5 - 4 = 1 ✓ MDS
- [[7,1,3]]: k = 1, n - 2(d-1) = 7 - 4 = 3, so not MDS

### Quantum Hamming Bound

**Theorem:**
For a non-degenerate [[n, k, d]] code with d = 2t + 1:
$$\boxed{2^{n-k} \geq \sum_{j=0}^{t} \binom{n}{j} 3^j}$$

**Interpretation:** The number of syndromes must accommodate all correctable errors.

**Perfect Codes:**
Codes achieving equality are perfect.

**Example:**
[[5,1,3]]: t = 1, need $2^4 = 16 \geq 1 + 3(5) = 16$ ✓ Perfect!

### Degenerate Codes

**Definition:**
A code is **degenerate** if different errors have the same effect on the code space.

For degenerate codes:
- Multiple errors may share a syndrome
- Distance can exceed the error correction capability

**Example:**
Surface codes are highly degenerate — many error configurations lead to the same logical effect.

### Distance for CSS Codes

For CSS codes with X-stabilizers from $C_1^\perp$ and Z-stabilizers from $C_2^\perp$:

$$d = \min(d_X, d_Z)$$

where:
- $d_X$ = minimum weight of $C_1 \setminus C_2^\perp$
- $d_Z$ = minimum weight of $C_2 \setminus C_1^\perp$

---

## Worked Examples

### Example 1: Logicals for [[4,2,2]] Code

**Stabilizers:**
- $S_1 = XXXX \leftrightarrow (1,1,1,1|0,0,0,0)$
- $S_2 = ZZZZ \leftrightarrow (0,0,0,0|1,1,1,1)$

**Parity check matrix:**
$$H = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 1 & 1
\end{pmatrix}$$

**Finding centralizer:**
Need vectors v with $\langle h_i, v \rangle_s = 0$ for all rows $h_i$.

For $h_1 = (1,1,1,1|0,0,0,0)$:
$\langle h_1, (a|b) \rangle_s = (1,1,1,1) \cdot b + (0,0,0,0) \cdot a = b_1 + b_2 + b_3 + b_4 = 0$

For $h_2 = (0,0,0,0|1,1,1,1)$:
$\langle h_2, (a|b) \rangle_s = (0,0,0,0) \cdot b + (1,1,1,1) \cdot a = a_1 + a_2 + a_3 + a_4 = 0$

**Centralizer conditions:**
- $a_1 + a_2 + a_3 + a_4 = 0$ (even number of X's)
- $b_1 + b_2 + b_3 + b_4 = 0$ (even number of Z's)

**Basis for centralizer** (dim = 6):
$$\{(1,1,0,0|0,0,0,0), (1,0,1,0|0,0,0,0), (1,0,0,1|0,0,0,0),$$
$$(0,0,0,0|1,1,0,0), (0,0,0,0|1,0,1,0), (0,0,0,0|1,0,0,1)\}$$

**Logical operators** (not in stabilizer span):
- $\bar{X}_1 = X_1X_2$ (or any even-weight X)
- $\bar{X}_2 = X_1X_3$
- $\bar{Z}_1 = Z_1Z_2$
- $\bar{Z}_2 = Z_1Z_3$

**Verify anticommutation:**
$\langle \bar{X}_1, \bar{Z}_1 \rangle_s = (1,1,0,0)\cdot(1,1,0,0) = 0$ ... Wait, should anticommute!

Let me recalculate: $\bar{X}_1 = (1,1,0,0|0,0,0,0)$, $\bar{Z}_1 = (0,0,0,0|1,1,0,0)$
$\langle \bar{X}_1, \bar{Z}_1 \rangle_s = (1,1,0,0)\cdot(1,1,0,0) + (0,0,0,0)\cdot(0,0,0,0) = 0$

This commutes! Need different pairing.

Actually for [[4,2,2]], logical pairs:
- $\bar{X}_1 = X_1X_2$, $\bar{Z}_1 = Z_1Z_3$ (check: positions 1,2 and 1,3 overlap at 1 → anticommute)
- $\bar{X}_2 = X_1X_3$, $\bar{Z}_2 = Z_1Z_2$

Let me verify: $\bar{X}_1 = (1,1,0,0|0,0,0,0)$, $\bar{Z}_1 = (0,0,0,0|1,0,1,0)$
$\langle \bar{X}_1, \bar{Z}_1 \rangle_s = (1,1,0,0)\cdot(1,0,1,0) + 0 = 1 + 0 = 1$ ✓ Anticommute!

### Example 2: Distance of [[5,1,3]] Code

**Stabilizers span** (dimension 4):
All vectors in row space of H.

**Need:** Minimum weight in C(S) \ S.

Check weight-1: $X_i, Z_i, Y_i$ for i = 1,...,5
- Do they commute with all stabilizers? Check syndromes.

For $X_1$: syndrome = H·Ω·$(1,0,0,0,0|0,0,0,0)^T$
Computing shows non-zero syndrome → not in C(S).

Similarly for all weight-1 → no weight-1 logicals.

Check weight-2:
$X_1X_2$: Check if syndrome = 0 and not in S.
...

After checking, minimum weight in C(S) \ S is 3.

**Known:** $\bar{X} = X_1X_2X_3$ (weight 3)
**Known:** $\bar{Z} = Z_1Z_2Z_3$ (weight 3)

Therefore $d = 3$.

### Example 3: Quantum Singleton Bound Check

**[[7,1,3]] Steane code:**
$k \leq n - 2(d-1) \Rightarrow 1 \leq 7 - 2(2) = 3$ ✓

The code doesn't saturate the bound (could have k up to 3 for d=3, but only encodes 1).

**[[5,1,3]] code:**
$1 \leq 5 - 4 = 1$ ✓ Saturates! (MDS)

---

## Practice Problems

### Level 1: Direct Application

1. **Centralizer Dimension:** For an [[n, k, d]] code, what is dim(C(S)_bin)?

2. **Logical Count:** How many independent logical operators does an [[8, 2, 3]] code have?

3. **Singleton Check:** Which codes are possible?
   a) [[6, 2, 3]]
   b) [[9, 1, 5]]
   c) [[7, 3, 3]]

### Level 2: Intermediate

4. **Finding Logicals:** For the 3-qubit bit flip code with $H = \begin{pmatrix} 0&0&0&1&1&0 \\ 0&0&0&0&1&1 \end{pmatrix}$:
   a) Find the centralizer basis
   b) Identify logical $\bar{X}$ and $\bar{Z}$
   c) Verify the code distance is 1

5. **Hamming Bound:** Verify the [[5,1,3]] code saturates the quantum Hamming bound.

6. **CSS Distance:** For a CSS code from [7,4,3] Hamming:
   - C₁ = [7,4,3], C₂ = C₁
   - Show the resulting [[7,1,3]] code has distance 3

### Level 3: Challenging

7. **General Distance Algorithm:** Write pseudocode for computing the distance of an [[n, k, d]] code given H.

8. **Degeneracy:** For the [[9,1,3]] Shor code:
   a) Find two distinct weight-2 errors with the same syndrome
   b) Show these errors have the same effect on the code space

9. **Optimal Codes:** Prove that no [[6, 2, 3]] code exists by showing violation of a bound.

---

## Solutions

### Level 1 Solutions

1. dim(C(S)_bin) = 2n - rank(H) = 2n - (n-k) = n + k

2. [[8,2,3]] has 2k = 4 logical operators (2 $\bar{X}$ and 2 $\bar{Z}$)

3. Singleton bound k ≤ n - 2(d-1):
   a) [[6,2,3]]: 2 ≤ 6 - 4 = 2 ✓ Possible
   b) [[9,1,5]]: 1 ≤ 9 - 8 = 1 ✓ Possible
   c) [[7,3,3]]: 3 ≤ 7 - 4 = 3 ✓ Possible

---

## Computational Lab

```python
"""
Day 734: Logical Operators and Code Distance
=============================================
Implementation of logical operator and distance computation.
"""

import numpy as np
from typing import List, Tuple, Set
from itertools import combinations, product

def mod2(M: np.ndarray) -> np.ndarray:
    """Reduce mod 2."""
    return np.array(M) % 2

def symplectic_matrix(n: int) -> np.ndarray:
    """Create 2n × 2n symplectic form matrix."""
    return np.block([
        [np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
        [np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]
    ])

def symplectic_inner_product(v1: np.ndarray, v2: np.ndarray) -> int:
    """Compute symplectic inner product."""
    n = len(v1) // 2
    a1, b1 = v1[:n], v1[n:]
    a2, b2 = v2[:n], v2[n:]
    return (np.dot(a1, b2) + np.dot(b1, a2)) % 2

def null_space_f2(M: np.ndarray) -> np.ndarray:
    """Compute null space basis over F₂."""
    M = mod2(M.copy())
    nrows, ncols = M.shape

    # RREF
    pivots = []
    pivot_row = 0
    for col in range(ncols):
        if pivot_row >= nrows:
            break
        # Find pivot
        found = False
        for row in range(pivot_row, nrows):
            if M[row, col] == 1:
                M[[pivot_row, row]] = M[[row, pivot_row]]
                found = True
                break
        if not found:
            continue

        pivots.append(col)
        for row in range(nrows):
            if row != pivot_row and M[row, col] == 1:
                M[row] = mod2(M[row] + M[pivot_row])
        pivot_row += 1

    # Free columns
    free_cols = [j for j in range(ncols) if j not in pivots]

    if not free_cols:
        return np.zeros((0, ncols), dtype=int)

    null_basis = np.zeros((len(free_cols), ncols), dtype=int)
    for i, fc in enumerate(free_cols):
        null_basis[i, fc] = 1
        for j, pc in enumerate(pivots):
            if j < len(pivots) and M[j, fc] == 1:
                null_basis[i, pc] = 1

    return mod2(null_basis)

def find_centralizer(H: np.ndarray) -> np.ndarray:
    """
    Find basis for centralizer C(S) = ker((H·Ω)^T).
    """
    n = H.shape[1] // 2
    Omega = symplectic_matrix(n)
    HO = mod2(H @ Omega)
    return null_space_f2(HO.T)

def row_space_f2(M: np.ndarray) -> np.ndarray:
    """Compute row space basis over F₂."""
    M = mod2(M.copy())
    nrows, ncols = M.shape

    # RREF
    pivot_row = 0
    for col in range(ncols):
        if pivot_row >= nrows:
            break
        found = False
        for row in range(pivot_row, nrows):
            if M[row, col] == 1:
                M[[pivot_row, row]] = M[[row, pivot_row]]
                found = True
                break
        if not found:
            continue
        for row in range(nrows):
            if row != pivot_row and M[row, col] == 1:
                M[row] = mod2(M[row] + M[pivot_row])
        pivot_row += 1

    # Non-zero rows
    return M[:pivot_row]

def is_in_row_space(v: np.ndarray, basis: np.ndarray) -> bool:
    """Check if vector v is in the row space of basis."""
    if len(basis) == 0:
        return np.all(v == 0)

    # Augment basis with v and check rank
    aug = np.vstack([basis, v])
    orig_rank = np.linalg.matrix_rank(mod2(basis))
    new_rank = np.linalg.matrix_rank(mod2(aug))
    return orig_rank == new_rank

def pauli_weight(v: np.ndarray) -> int:
    """Compute Pauli weight (number of non-I positions)."""
    n = len(v) // 2
    a, b = v[:n], v[n:]
    return np.sum((a | b) > 0)

def find_logical_operators(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find logical X and Z operators.

    Returns basis vectors for logical operators
    (not paired into X-Z pairs, just the full logical space).
    """
    n = H.shape[1] // 2
    n_k = H.shape[0]
    k = n - n_k

    # Find centralizer
    centralizer = find_centralizer(H)

    # Find stabilizer row space
    stab_basis = row_space_f2(H)

    # Find logicals: centralizer elements not in stabilizer space
    logicals = []
    for v in centralizer:
        if not is_in_row_space(v, stab_basis):
            logicals.append(v)

    if len(logicals) == 0:
        return np.zeros((0, 2*n), dtype=int), np.zeros((0, 2*n), dtype=int)

    logicals = np.array(logicals)

    # Separate into X-type and Z-type (simplified)
    # Actually return full logical space basis
    return logicals, logicals

def compute_distance(H: np.ndarray, max_weight: int = None) -> int:
    """
    Compute code distance.

    Finds minimum weight non-trivial logical operator.
    """
    n = H.shape[1] // 2
    n_k = H.shape[0]

    if max_weight is None:
        max_weight = n

    Omega = symplectic_matrix(n)
    stab_basis = row_space_f2(H)

    def commutes_with_all(v):
        """Check if v commutes with all stabilizers."""
        syndromes = mod2(H @ Omega @ v)
        return np.all(syndromes == 0)

    def is_stabilizer(v):
        """Check if v is in stabilizer group."""
        return is_in_row_space(v, stab_basis)

    # Check increasing weights
    for w in range(1, max_weight + 1):
        # Generate all weight-w Paulis
        for positions in combinations(range(n), w):
            # Try all Pauli types at these positions
            for pauli_types in product(range(1, 4), repeat=w):
                # pauli_types: 1=X, 2=Z, 3=Y
                v = np.zeros(2*n, dtype=int)
                for pos, ptype in zip(positions, pauli_types):
                    if ptype == 1:  # X
                        v[pos] = 1
                    elif ptype == 2:  # Z
                        v[pos + n] = 1
                    else:  # Y
                        v[pos] = 1
                        v[pos + n] = 1

                if commutes_with_all(v) and not is_stabilizer(v):
                    return w

    return max_weight + 1  # Not found

def verify_singleton_bound(n: int, k: int, d: int) -> bool:
    """Check quantum Singleton bound: k <= n - 2(d-1)."""
    return k <= n - 2*(d - 1)

def verify_hamming_bound(n: int, k: int, t: int) -> bool:
    """Check quantum Hamming bound for t-error correction."""
    from math import comb
    lhs = 2**(n - k)
    rhs = sum(comb(n, j) * (3**j) for j in range(t + 1))
    return lhs >= rhs

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 734: Logical Operators and Code Distance")
    print("=" * 60)

    # Example 1: [[4,2,2]] code
    print("\n1. [[4,2,2]] Code Logical Operators")
    print("-" * 40)

    H_422 = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ])

    centralizer = find_centralizer(H_422)
    print(f"Centralizer dimension: {len(centralizer)}")
    print(f"Expected: n + k = 4 + 2 = 6")

    stab_basis = row_space_f2(H_422)
    print(f"\nStabilizer dimension: {len(stab_basis)}")

    logicals, _ = find_logical_operators(H_422)
    print(f"\nLogical space dimension: {len(logicals)}")
    print("Logical operator representatives:")
    for i, l in enumerate(logicals[:4]):
        # Convert to Pauli string
        n = len(l) // 2
        chars = []
        for j in range(n):
            if l[j] == 0 and l[j+n] == 0:
                chars.append('I')
            elif l[j] == 1 and l[j+n] == 0:
                chars.append('X')
            elif l[j] == 0 and l[j+n] == 1:
                chars.append('Z')
            else:
                chars.append('Y')
        print(f"  L{i+1}: {''.join(chars)} (weight {pauli_weight(l)})")

    # Example 2: [[5,1,3]] code distance
    print("\n2. [[5,1,3]] Code Distance")
    print("-" * 40)

    H_513 = np.array([
        [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    ])

    d = compute_distance(H_513, max_weight=5)
    print(f"Computed distance: {d}")
    print(f"Expected: 3")

    # Example 3: Bounds verification
    print("\n3. Code Bounds Verification")
    print("-" * 40)

    codes = [
        (5, 1, 3, "[[5,1,3]] perfect"),
        (7, 1, 3, "[[7,1,3]] Steane"),
        (4, 2, 2, "[[4,2,2]]"),
        (9, 1, 3, "[[9,1,3]] Shor")
    ]

    for n, k, d, name in codes:
        singleton_ok = verify_singleton_bound(n, k, d)
        t = (d - 1) // 2
        hamming_ok = verify_hamming_bound(n, k, t)
        print(f"{name}:")
        print(f"  Singleton: k={k} ≤ n-2(d-1)={n-2*(d-1)} → {singleton_ok}")
        print(f"  Hamming (t={t}): {2**(n-k)} ≥ Σ C(n,j)3^j → {hamming_ok}")

    # Example 4: [[7,1,3]] Steane code analysis
    print("\n4. [[7,1,3]] Steane Code Analysis")
    print("-" * 40)

    H_713 = np.array([
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    ])

    centralizer_713 = find_centralizer(H_713)
    print(f"Centralizer dimension: {len(centralizer_713)}")
    print(f"Expected: n + k = 7 + 1 = 8")

    d_713 = compute_distance(H_713, max_weight=4)
    print(f"Computed distance: {d_713}")

    # Example 5: Degenerate errors
    print("\n5. Error Analysis for [[4,2,2]]")
    print("-" * 40)

    Omega = symplectic_matrix(4)

    errors = [
        ('X1', np.array([1,0,0,0,0,0,0,0])),
        ('X2', np.array([0,1,0,0,0,0,0,0])),
        ('Z1', np.array([0,0,0,0,1,0,0,0])),
        ('Z2', np.array([0,0,0,0,0,1,0,0])),
        ('Y1', np.array([1,0,0,0,1,0,0,0])),
    ]

    print("Syndromes:")
    for name, e in errors:
        s = mod2(H_422 @ Omega @ e)
        print(f"  {name}: syndrome = {s}")

    print("\n" + "=" * 60)
    print("End of Day 734 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Centralizer | $C(S)_{\text{bin}} = \ker(H\Omega)^T$ |
| dim(C(S)) | $n + k$ |
| Distance | $d = \min_{P \in C(S) \setminus S} \text{wt}(P)$ |
| Singleton bound | $k \leq n - 2(d-1)$ |
| Hamming bound | $2^{n-k} \geq \sum_{j=0}^t \binom{n}{j} 3^j$ |

### Main Takeaways

1. **Logical operators** are in the centralizer but not the stabilizer group
2. **Code distance** is the minimum weight logical operator
3. **Quantum Singleton bound** limits rate for given distance
4. **Perfect codes** saturate the Hamming bound ([[5,1,3]])
5. **MDS codes** saturate the Singleton bound

---

## Daily Checklist

- [ ] I understand the centralizer C(S)
- [ ] I can compute logical operators from H
- [ ] I know how to determine code distance
- [ ] I can verify Singleton and Hamming bounds
- [ ] I understand degeneracy in quantum codes
- [ ] I implemented distance computation

---

## Preview: Day 735

Tomorrow is **Week 105 Synthesis**:
- Comprehensive review of binary formalism
- Integration of F₂, symplectic, and GF(4) perspectives
- Practice problems spanning the week
- Preparation for graph states (Week 106)
