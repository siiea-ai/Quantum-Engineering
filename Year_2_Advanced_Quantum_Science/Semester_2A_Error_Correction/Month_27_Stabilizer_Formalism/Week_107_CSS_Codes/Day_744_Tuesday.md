# Day 744: Dual Containment Condition

## Overview

**Day:** 744 of 1008
**Week:** 107 (CSS Codes & Related Constructions)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Deep Dive into C₂⊥ ⊆ C₁

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Dual code theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Self-orthogonal codes |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Compute** the dual code C⊥ from a generator or parity check matrix
2. **Verify** dual containment for specific code pairs
3. **Identify** self-orthogonal and self-dual codes
4. **Construct** CSS codes from weakly self-dual codes
5. **Apply** MacWilliams identities to relate code distances
6. **Design** CSS codes with guaranteed parameters

---

## Classical Dual Codes

### Definition of Dual Code

For a linear code C ⊆ $\mathbb{F}_2^n$:

$$\boxed{C^\perp = \{\mathbf{v} \in \mathbb{F}_2^n : \mathbf{v} \cdot \mathbf{c} = 0 \text{ for all } \mathbf{c} \in C\}}$$

**Properties:**
- $\dim(C) + \dim(C^\perp) = n$
- $(C^\perp)^\perp = C$
- $C_1 \subseteq C_2 \Rightarrow C_2^\perp \subseteq C_1^\perp$

### Parity Check/Generator Duality

If C has:
- Generator matrix G (k × n)
- Parity check matrix H ((n-k) × n)

Then C⊥ has:
- Generator matrix H
- Parity check matrix G

**Relation:** $G \cdot H^T = 0$

### Computing the Dual

**Algorithm:** Given parity check H for C:
1. The rows of H generate C⊥
2. Row reduce H to find basis for C⊥
3. This basis is the generator matrix for C⊥

---

## Self-Orthogonal and Self-Dual Codes

### Definitions

**Self-orthogonal:** $C^\perp \subseteq C$ (equivalently: $\mathbf{c} \cdot \mathbf{c}' = 0$ for all $\mathbf{c}, \mathbf{c}' \in C$)

**Self-dual:** $C = C^\perp$ (equivalently: self-orthogonal AND $\dim(C) = n/2$)

**Weakly self-dual:** $C \subseteq C^\perp$ (same as $C^\perp$ being self-orthogonal)

### Hierarchy

$$\text{Self-dual} \subset \text{Self-orthogonal} \subset \text{General}$$

For n-bit codes:
- Self-dual codes have $k = n/2$ (requires n even)
- Self-orthogonal codes have $k \leq n/2$

### Key Theorem for CSS

**Theorem:** If C is self-orthogonal ($C^\perp \subseteq C$), then CSS(C, C) is a valid quantum code.

**Proof:**
- Need to verify $C_2^\perp \subseteq C_1$ where $C_1 = C_2 = C$
- This is exactly $C^\perp \subseteq C$, which holds by assumption ✓

### CSS from Self-Orthogonal Code

If C is an [n, k, d] self-orthogonal code:
$$CSS(C, C) = [[n, 2k - n, d']]$$

where $d' \geq d(C^\perp)$ (distance of dual code).

---

## Constructing Self-Orthogonal Codes

### Method 1: Direct Construction

**Condition:** A code is self-orthogonal iff its generator matrix G satisfies:
$$G \cdot G^T = 0 \pmod{2}$$

This means all rows are orthogonal to each other AND to themselves.

**Self-orthogonality of rows:** $\mathbf{g}_i \cdot \mathbf{g}_i = wt(\mathbf{g}_i) \mod 2$

So all codewords must have **even weight** (doubly-even is even stronger).

### Method 2: Construction X

Start with code C and add vectors to make it self-orthogonal:
1. Find vectors orthogonal to C
2. Adjoin them to C
3. Result is self-orthogonal extension

### Method 3: Puncturing and Shortening

If C is self-orthogonal/self-dual:
- **Shortening** preserves self-orthogonality
- **Puncturing** may destroy it (need careful analysis)

---

## Doubly-Even Codes

### Definition

A code is **doubly-even** if all codewords have weight divisible by 4.

### Connection to Self-Orthogonality

**Theorem:** A doubly-even code is self-orthogonal.

**Proof:** For $\mathbf{c}, \mathbf{c}' \in C$:
$$\mathbf{c} \cdot \mathbf{c}' = \frac{1}{2}(wt(\mathbf{c} + \mathbf{c}') - wt(\mathbf{c}) - wt(\mathbf{c}')) \pmod{2}$$

If all weights are divisible by 4, this is 0 mod 2. ✓

### Famous Doubly-Even Codes

| Code | Parameters | Notes |
|------|------------|-------|
| Extended Hamming | [8, 4, 4] | Self-dual |
| Extended Golay | [24, 12, 8] | Self-dual, optimal |
| Reed-Muller RM(1,m) | [2^m, m+1, 2^{m-1}] | m ≥ 3 |

---

## Verifying Dual Containment

### General Case: C₂⊥ ⊆ C₁

**Method 1:** Direct inclusion check
1. Find generators for C₂⊥ (rows of H₂)
2. Check each generator is a codeword of C₁
3. I.e., verify $H_1 \cdot H_2^T = 0$

**Method 2:** Via dimensions
1. $C_2^\perp \subseteq C_1$ iff $C_1^\perp \subseteq C_2$
2. Check $G_1 \subseteq$ kernel of $H_2$
3. I.e., verify $H_2 \cdot G_1^T = 0$

### Practical Test

$$\boxed{C_2^\perp \subseteq C_1 \Leftrightarrow H_1 \cdot H_2^T = 0 \pmod{2}}$$

When H₁ = H₂ = H (self-orthogonal case): $H \cdot H^T = 0$

---

## Distance Bounds for CSS

### MacWilliams Identity

The weight enumerator of C⊥ is determined by that of C:
$$W_{C^\perp}(x, y) = \frac{1}{|C|} W_C(x + y, x - y)$$

This constrains the minimum distance of C⊥.

### CSS Distance Formula

For CSS(C₁, C₂):

**X-error distance:** minimum weight in $C_1 \setminus C_2^\perp$
$$d_X = \min\{wt(\mathbf{c}) : \mathbf{c} \in C_1, \mathbf{c} \notin C_2^\perp\}$$

**Z-error distance:** minimum weight in $C_2 \setminus C_1^\perp$
$$d_Z = \min\{wt(\mathbf{c}) : \mathbf{c} \in C_2, \mathbf{c} \notin C_1^\perp\}$$

**Code distance:** $d = \min(d_X, d_Z)$

### Symmetric Case

For CSS(C, C) with self-orthogonal C:
$$d_X = d_Z = \min\{wt(\mathbf{c}) : \mathbf{c} \in C, \mathbf{c} \notin C^\perp\}$$

Since C⊥ ⊊ C, this is the minimum weight of codewords NOT in C⊥.

---

## Examples of Code Families

### Extended Hamming [8, 4, 4]

**Generator matrix:**
$$G = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 1 & 1 & 1 \\
0 & 1 & 0 & 0 & 1 & 0 & 1 & 1 \\
0 & 0 & 1 & 0 & 1 & 1 & 0 & 1 \\
0 & 0 & 0 & 1 & 1 & 1 & 1 & 0
\end{pmatrix}$$

**Verify:** $G \cdot G^T$ = ?

All rows have weight 4 (doubly-even).
Pairwise products: need to check.

**Result:** Self-dual, so CSS(C,C) = [[8, 0, 4]] (no logical qubits!)

### Reed-Muller Codes

**RM(r, m):** Code with parameters $[2^m, \sum_{i=0}^r \binom{m}{i}, 2^{m-r}]$

**Key property:** $RM(r, m)^\perp = RM(m-r-1, m)$

**CSS construction:** Need $RM(m-r-1, m) \subseteq RM(r, m)$

This holds when $m - r - 1 \leq r$, i.e., $r \geq (m-1)/2$.

**Example:** RM(1, 4) = [16, 5, 8]
- $RM(1, 4)^\perp = RM(2, 4)$ [16, 11, 4]
- $RM(2, 4)^\perp = RM(1, 4)$ ⊆ RM(1, 4) ✓

CSS(RM(1,4), RM(1,4)) = [[16, 2·5-16, ?]] = [[16, -6, ?]]

Hmm, negative k means RM(1,4) is NOT self-orthogonal. We need:
- CSS(C₁, C₂) with different codes.

**Correct:** CSS(RM(2,4), RM(1,4)):
- $C_2^\perp = RM(2, 4)$
- Check $RM(2,4) \subseteq RM(2,4)$ ✓

k = 11 + 5 - 16 = 0 (no logical qubits)

---

## Worked Examples

### Example 1: Verify Self-Orthogonality

**Problem:** Is the [6, 3, 3] code with parity check
$$H = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0
\end{pmatrix}$$
self-orthogonal?

**Solution:**

Check $H \cdot H^T$:
$$H H^T = \begin{pmatrix}
4 & 2 & 2 \\
2 & 4 & 2 \\
2 & 2 & 3
\end{pmatrix} \mod 2 = \begin{pmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 1
\end{pmatrix}$$

Not zero! So this code is NOT self-orthogonal.

Row 3 has odd weight (3), so $\mathbf{h}_3 \cdot \mathbf{h}_3 = 1 \neq 0$.

### Example 2: Construct Self-Orthogonal Extension

**Problem:** Extend the [4, 2, 2] code to a self-orthogonal code.

**Solution:**

Original generator:
$$G = \begin{pmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & 1 & 0
\end{pmatrix}$$

Check: $G G^T = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} \mod 2 = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix} \neq 0$

Not self-orthogonal. Extend by adding parity bit:

$$G' = \begin{pmatrix}
1 & 0 & 1 & 1 & 1 \\
0 & 1 & 1 & 0 & 0
\end{pmatrix}$$

Check: $G' (G')^T = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix} \mod 2 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \neq 0$

Still not self-orthogonal. Need different extension:

$$G'' = \begin{pmatrix}
1 & 0 & 1 & 1 & 0 & 1 \\
0 & 1 & 1 & 0 & 1 & 1
\end{pmatrix}$$

Check: Row weights = 4, 4 (both even)
$(G'')(G'')^T = \begin{pmatrix} 4 & 2 \\ 2 & 4 \end{pmatrix} \mod 2 = 0$ ✓

This gives [6, 2] self-orthogonal code.

### Example 3: CSS Parameters

**Problem:** Compute parameters for CSS(C₁, C₂) where:
- C₁ = [7, 4, 3] Hamming code
- C₂ = [7, 4, 3] Hamming code (same)

with verified $C_2^\perp \subseteq C_1$.

**Solution:**

$$k = k_1 + k_2 - n = 4 + 4 - 7 = 1$$

$C^\perp$ for [7, 4, 3] Hamming is [7, 3, 4].

Distance analysis:
- $C_2^\perp$ = [7, 3, 4], so $d(C_2^\perp) = 4$
- C₁ = [7, 4, 3], so $d(C_1) = 3$

But we need minimum weight in $C_1 \setminus C_2^\perp$:
- $|C_1| = 2^4 = 16$ codewords
- $|C_2^\perp| = 2^3 = 8$ codewords
- $|C_1 \setminus C_2^\perp| = 8$ codewords

The minimum weight among these 8 is 3.

**Result:** [[7, 1, 3]] Steane code.

---

## Practice Problems

### Level 1: Direct Application

**P1.1** Compute $H \cdot H^T$ for the [7,4,3] Hamming code parity check matrix and verify whether it's self-orthogonal.

**P1.2** Given a [5, 2, 3] code with generator
$$G = \begin{pmatrix} 1 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 1 & 1 \end{pmatrix}$$
find its dual code.

**P1.3** If C₁ is [n, k₁] and C₂ is [n, k₂] with $C_2^\perp \subseteq C_1$, what constraint does this place on k₁ and k₂?

### Level 2: Intermediate

**P2.1** Prove that if C is doubly-even (all weights ≡ 0 mod 4), then C is self-orthogonal.

**P2.2** For the extended Hamming [8, 4, 4] code:
a) Verify it is self-dual
b) Construct CSS(C, C)
c) Why does this give k = 0?

**P2.3** Find all [n, k] dimensions where self-dual codes can exist.

### Level 3: Challenging

**P3.1** Prove that $C_2^\perp \subseteq C_1$ iff $C_1^\perp \subseteq C_2$.

**P3.2** For Reed-Muller codes:
a) Show $RM(r, m)^\perp = RM(m-r-1, m)$
b) Find all (r, m) pairs where RM(r, m) is self-orthogonal
c) Construct a CSS code from RM codes

**P3.3** Design a CSS code with parameters [[15, 7, 3]] using BCH codes.

---

## Computational Lab

```python
"""
Day 744: Dual Containment and Self-Orthogonality
================================================

Exploring classical code duality and CSS conditions.
"""

import numpy as np
from typing import Tuple, List, Optional


def gf2_row_reduce(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
    """Row reduce matrix over GF(2), return (reduced, rank)."""
    M = matrix.copy() % 2
    rows, cols = M.shape
    rank = 0

    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row
                break

        if pivot is None:
            continue

        M[[rank, pivot]] = M[[pivot, rank]]

        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2

        rank += 1

    return M, rank


def gf2_null_space(H: np.ndarray) -> np.ndarray:
    """Compute null space basis of H over GF(2)."""
    m, n = H.shape

    # Augment: [H^T | I_n]
    augmented = np.hstack([H.T, np.eye(n, dtype=int)]) % 2

    # Row reduce
    reduced, rank = gf2_row_reduce(augmented)

    # Find zero rows in H^T part - these give null space
    null_vectors = []
    for i in range(n):
        if np.all(reduced[i, :m] == 0):
            null_vectors.append(reduced[i, m:])

    if not null_vectors:
        return np.array([]).reshape(0, n)

    return np.array(null_vectors) % 2


class DualityAnalyzer:
    """Analyze dual codes and self-orthogonality."""

    def __init__(self, parity_check: np.ndarray):
        """Initialize with parity check matrix H."""
        self.H = np.array(parity_check) % 2
        self.m, self.n = self.H.shape
        _, rank = gf2_row_reduce(self.H)
        self.k = self.n - rank

    def generator_matrix(self) -> np.ndarray:
        """Get generator matrix (null space of H)."""
        return gf2_null_space(self.H)

    def is_self_orthogonal(self) -> bool:
        """Check if C^⊥ ⊆ C (equivalently H·H^T = 0)."""
        product = (self.H @ self.H.T) % 2
        return np.all(product == 0)

    def is_self_dual(self) -> bool:
        """Check if C = C^⊥ (self-orthogonal and dim = n/2)."""
        if not self.is_self_orthogonal():
            return False
        return self.k == self.n // 2

    def orthogonality_defect(self) -> np.ndarray:
        """Return H·H^T to see which rows fail orthogonality."""
        return (self.H @ self.H.T) % 2

    def dual_parity_check(self) -> np.ndarray:
        """Get parity check of dual code (= generator of original)."""
        return self.generator_matrix()

    def verify_dual_containment(self, other: 'DualityAnalyzer') -> bool:
        """
        Check if other^⊥ ⊆ self.

        This is the CSS condition: C2^⊥ ⊆ C1.
        """
        # other^⊥ has generator = other.H (rows of parity check)
        # self has parity check = self.H
        # other^⊥ ⊆ self iff self.H · other.H^T = 0
        product = (self.H @ other.H.T) % 2
        return np.all(product == 0)

    def codeword_weights(self) -> List[int]:
        """Enumerate codeword weights (for small codes)."""
        G = self.generator_matrix()
        if G.size == 0:
            return [0]  # Only zero codeword

        k = G.shape[0]
        if k > 15:
            return None  # Too many codewords

        weights = []
        for i in range(2**k):
            coeffs = np.array([(i >> j) & 1 for j in range(k)])
            codeword = (coeffs @ G) % 2
            weights.append(int(np.sum(codeword)))

        return sorted(set(weights))

    def minimum_distance(self) -> int:
        """Compute minimum distance (for small codes)."""
        weights = self.codeword_weights()
        if weights is None:
            return -1  # Unknown
        nonzero = [w for w in weights if w > 0]
        return min(nonzero) if nonzero else 0


def check_doubly_even(H: np.ndarray) -> bool:
    """Check if code is doubly-even (all weights ≡ 0 mod 4)."""
    analyzer = DualityAnalyzer(H)
    G = analyzer.generator_matrix()

    if G.size == 0:
        return True

    k = G.shape[0]
    if k > 15:
        return None

    for i in range(2**k):
        coeffs = np.array([(i >> j) & 1 for j in range(k)])
        codeword = (coeffs @ G) % 2
        weight = int(np.sum(codeword))
        if weight % 4 != 0:
            return False

    return True


def css_distance(H1: np.ndarray, H2: np.ndarray) -> Tuple[int, int, int]:
    """
    Compute CSS code distances.

    Returns (d_X, d_Z, d) where:
    - d_X = min weight in C1 \ C2^⊥
    - d_Z = min weight in C2 \ C1^⊥
    - d = min(d_X, d_Z)
    """
    C1 = DualityAnalyzer(H1)
    C2 = DualityAnalyzer(H2)

    G1 = C1.generator_matrix()
    G2 = C2.generator_matrix()

    if G1.size == 0 or G2.size == 0:
        return (0, 0, 0)

    k1, k2 = G1.shape[0], G2.shape[0]

    if k1 > 12 or k2 > 12:
        return (-1, -1, -1)  # Too large

    # C2^⊥ has generator = H2
    # Enumerate C1 and check which are NOT in C2^⊥
    d_X = float('inf')
    for i in range(1, 2**k1):
        coeffs = np.array([(i >> j) & 1 for j in range(k1)])
        c1 = (coeffs @ G1) % 2

        # Check if c1 is in C2^⊥ (syndrome w.r.t. C2 generator)
        # c1 in C2^⊥ iff G2 · c1 = 0
        if G2.size > 0:
            syndrome = (G2 @ c1) % 2
            in_C2_dual = np.all(syndrome == 0)
        else:
            in_C2_dual = False

        if not in_C2_dual:
            weight = int(np.sum(c1))
            d_X = min(d_X, weight)

    # Similarly for d_Z: min weight in C2 \ C1^⊥
    d_Z = float('inf')
    for i in range(1, 2**k2):
        coeffs = np.array([(i >> j) & 1 for j in range(k2)])
        c2 = (coeffs @ G2) % 2

        if G1.size > 0:
            syndrome = (G1 @ c2) % 2
            in_C1_dual = np.all(syndrome == 0)
        else:
            in_C1_dual = False

        if not in_C1_dual:
            weight = int(np.sum(c2))
            d_Z = min(d_Z, weight)

    d_X = int(d_X) if d_X != float('inf') else 0
    d_Z = int(d_Z) if d_Z != float('inf') else 0

    return (d_X, d_Z, min(d_X, d_Z))


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 744: Dual Containment Analysis")
    print("=" * 60)

    # Example 1: Hamming code
    print("\n1. [7,4,3] Hamming Code")
    print("-" * 40)

    H_hamming = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ])

    hamming = DualityAnalyzer(H_hamming)
    print(f"Code: [{hamming.n}, {hamming.k}]")
    print(f"Self-orthogonal: {hamming.is_self_orthogonal()}")
    print(f"Orthogonality defect H·H^T:\n{hamming.orthogonality_defect()}")

    G = hamming.generator_matrix()
    print(f"\nGenerator matrix:\n{G}")

    weights = hamming.codeword_weights()
    print(f"Codeword weights: {weights}")
    print(f"Minimum distance: {hamming.minimum_distance()}")

    # Example 2: Extended Hamming
    print("\n2. [8,4,4] Extended Hamming Code")
    print("-" * 40)

    H_ext = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ])

    ext_hamming = DualityAnalyzer(H_ext)
    print(f"Code: [{ext_hamming.n}, {ext_hamming.k}]")
    print(f"Self-orthogonal: {ext_hamming.is_self_orthogonal()}")
    print(f"Self-dual: {ext_hamming.is_self_dual()}")
    print(f"Doubly-even: {check_doubly_even(H_ext)}")

    # Example 3: CSS condition verification
    print("\n3. CSS Dual Containment")
    print("-" * 40)

    # Use two different codes
    H1 = H_hamming  # [7,4,3]
    H2 = H_hamming  # [7,4,3]

    C1 = DualityAnalyzer(H1)
    C2 = DualityAnalyzer(H2)

    # For Steane code, we need C2^⊥ ⊆ C1
    # C2^⊥ has dim = 7 - 4 = 3, generated by H2
    # Check if rows of H2 are codewords of C1

    print(f"C1: [{C1.n}, {C1.k}]")
    print(f"C2: [{C2.n}, {C2.k}]")
    print(f"C2^⊥ ⊆ C1: {C1.verify_dual_containment(C2)}")

    # CSS parameters
    k_css = C1.k + C2.k - C1.n
    print(f"\nCSS code dimension k = {C1.k} + {C2.k} - {C1.n} = {k_css}")

    d_X, d_Z, d = css_distance(H1, H2)
    print(f"CSS distances: d_X = {d_X}, d_Z = {d_Z}, d = {d}")

    # Example 4: Non-self-orthogonal code
    print("\n4. Non-Self-Orthogonal Example")
    print("-" * 40)

    H_bad = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [1, 0, 0, 1, 0]
    ])

    bad_code = DualityAnalyzer(H_bad)
    print(f"Code: [{bad_code.n}, {bad_code.k}]")
    print(f"Self-orthogonal: {bad_code.is_self_orthogonal()}")
    print(f"Defect:\n{bad_code.orthogonality_defect()}")

    # Example 5: Creating self-orthogonal code
    print("\n5. Constructing Self-Orthogonal Code")
    print("-" * 40)

    # Start with even-weight code
    H_even = np.array([
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0]  # Need weight even
    ])

    # Adjust to make self-orthogonal
    # Row 3 has odd weight, so H·H^T will have (3,3) = 1
    # Replace with even-weight row
    H_fixed = np.array([
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0]
    ])

    fixed_code = DualityAnalyzer(H_fixed)
    print(f"Fixed code: [{fixed_code.n}, {fixed_code.k}]")
    print(f"Self-orthogonal: {fixed_code.is_self_orthogonal()}")

    print("\n" + "=" * 60)
    print("Dual containment is the foundation of CSS code design!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Dual code | $C^\perp = \{\mathbf{v} : \mathbf{v} \cdot \mathbf{c} = 0, \forall \mathbf{c} \in C\}$ |
| Self-orthogonal | $C^\perp \subseteq C \Leftrightarrow G \cdot G^T = 0$ |
| Self-dual | $C = C^\perp$ (requires $k = n/2$) |
| Dual containment test | $C_2^\perp \subseteq C_1 \Leftrightarrow H_1 \cdot H_2^T = 0$ |
| Doubly-even | $wt(\mathbf{c}) \equiv 0 \pmod{4}$ for all $\mathbf{c}$ |

### Main Takeaways

1. **Dual containment** $C_2^\perp \subseteq C_1$ is the CSS validity condition
2. **Self-orthogonal codes** satisfy $G \cdot G^T = 0$ (mod 2)
3. **Doubly-even codes** are automatically self-orthogonal
4. CSS **distance** requires analyzing $C_1 \setminus C_2^\perp$ and $C_2 \setminus C_1^\perp$
5. The test $H_1 \cdot H_2^T = 0$ verifies dual containment

---

## Daily Checklist

- [ ] I can compute the dual code from a parity check matrix
- [ ] I can verify self-orthogonality using $H \cdot H^T$
- [ ] I understand why doubly-even implies self-orthogonal
- [ ] I can verify dual containment for CSS construction
- [ ] I can compute CSS code parameters
- [ ] I understand the distance formula for CSS codes

---

## Preview: Day 745

Tomorrow we explore **concrete CSS code examples**:

- The Steane [[7, 1, 3]] code in detail
- Repetition-based CSS codes
- Reed-Muller quantum codes
- Systematic construction methods

Each example will illustrate different aspects of CSS design!
