# Day 674: Generator and Parity-Check Matrices Deep Dive

## Week 97: Classical Error Correction Review | Month 25: QEC Fundamentals I

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Matrix Representations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 674, you will be able to:

1. Convert generator matrices to systematic form using row operations
2. Prove that minimum distance equals the minimum number of linearly dependent columns of H
3. Construct parity-check matrices for codes with desired properties
4. Implement systematic encoding algorithms
5. Analyze code structure through matrix rank and null space
6. Understand the relationship between code duality and matrix transposition

---

## Core Content

### 1. Systematic Form and Row Reduction

**Goal:** Convert any generator matrix to systematic form $G = [I_k \mid P]$.

**Algorithm (Gaussian Elimination over $\mathbb{F}_2$):**
1. For each row $i$ from 1 to $k$:
   - Find a pivot in column $i$ (a row with 1 in position $i$)
   - Swap rows if necessary
   - Eliminate all other 1s in column $i$ using row additions

**Example:** Convert to systematic form.

$$G = \begin{pmatrix} 1 & 1 & 0 & 1 \\ 0 & 1 & 1 & 1 \end{pmatrix}$$

This is a [4, 2] code. We want form $[I_2 \mid P]$.

**Step 1:** Column 1 already has pivot in row 1. ✓
**Step 2:** Column 2 has 1s in both rows. Add row 2 to row 1:

$$G' = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 1 \end{pmatrix} = [I_2 \mid P]$$

where $P = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$.

**Key Property:** Row operations don't change the code—they just change the basis.

---

### 2. The Fundamental Distance Theorem

**Theorem:** For a linear code with parity-check matrix $H$, the minimum distance $d$ equals the minimum number of linearly dependent columns of $H$.

$$\boxed{d = \min\{w : \text{some } w \text{ columns of } H \text{ are linearly dependent}\}}$$

**Proof:**

Let $\mathbf{c} = (c_1, c_2, \ldots, c_n)$ be a codeword. Then $H\mathbf{c}^T = 0$, which means:

$$c_1 \mathbf{h}_1 + c_2 \mathbf{h}_2 + \cdots + c_n \mathbf{h}_n = \mathbf{0}$$

where $\mathbf{h}_i$ is the $i$-th column of $H$.

The weight of $\mathbf{c}$ is the number of nonzero $c_i$. A nonzero codeword of weight $w$ corresponds to $w$ columns of $H$ that sum to zero (i.e., are linearly dependent over $\mathbb{F}_2$).

The minimum distance is the minimum weight of a nonzero codeword, which equals the minimum number of linearly dependent columns. ∎

**Corollary:** $d \geq d_{\min}$ if and only if every $d_{\min} - 1$ columns of $H$ are linearly independent.

---

### 3. Constructing Codes from H

**Design Principle:** To construct a code with minimum distance $d$, ensure that any $d-1$ columns of $H$ are linearly independent.

**Example: Constructing a [7, 4, 3] Code**

For $d = 3$, we need any 2 columns of $H$ to be linearly independent.

$H$ has dimensions $(n-k) \times n = 3 \times 7$.

**Strategy:** Make all columns of $H$ distinct and nonzero.

There are exactly $2^3 - 1 = 7$ nonzero binary vectors of length 3. Use them as columns:

$$H = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$

Columns are binary representations of 1 through 7.

**Verification:** Any 2 distinct nonzero columns are linearly independent (their sum is another nonzero vector). So $d \geq 3$.

But columns 1, 2, 3 sum to zero: $(0,0,1)^T + (0,1,0)^T + (0,1,1)^T = (0,0,0)^T$. So $d \leq 3$.

Therefore $d = 3$. This is the **Hamming [7, 4, 3] code**!

---

### 4. The Null Space Perspective

**Definition:** The code $C$ is the **null space** (kernel) of $H$:

$$C = \ker(H) = \{\mathbf{x} \in \mathbb{F}_2^n : H\mathbf{x}^T = \mathbf{0}\}$$

**Equivalently:** The code $C$ is the **row space** of $G$:

$$C = \text{rowspace}(G) = \{\mathbf{m} \cdot G : \mathbf{m} \in \mathbb{F}_2^k\}$$

**Dimension Formula:**
$$\dim(\ker(H)) = n - \text{rank}(H) = n - (n-k) = k$$

This confirms that $|C| = 2^k$.

**Dual Perspective:**
$$C^\perp = \ker(G^T) = \text{rowspace}(H)$$

---

### 5. Encoding Algorithms

**Systematic Encoding** (when $G = [I_k \mid P]$):

```
Input: k-bit message m = (m_1, ..., m_k)
Output: n-bit codeword c = (m_1, ..., m_k, p_1, ..., p_{n-k})

Algorithm:
1. Copy message bits: c_i = m_i for i = 1, ..., k
2. Compute parity bits: p_j = sum_{i=1}^{k} m_i * P_{ij} (mod 2)
3. Return c = (m, p)
```

**Complexity:** $O(k \cdot (n-k))$ — much faster than general matrix multiplication.

**Non-Systematic Encoding:**
$$\mathbf{c} = \mathbf{m} \cdot G$$

Standard matrix-vector multiplication over $\mathbb{F}_2$.

---

### 6. Equivalent Codes

**Definition:** Two codes are **equivalent** if one can be obtained from the other by:
- Permuting coordinate positions
- Permuting code symbols within a position (for non-binary codes)

**Important:** Equivalent codes have the same parameters $[n, k, d]$.

**Generator Matrix Equivalence:**
Two generator matrices $G$ and $G'$ generate equivalent codes if $G'$ can be obtained from $G$ by:
- Row operations (doesn't change the code at all)
- Column permutations (changes coordinate ordering)

---

### 7. Extended and Shortened Codes

**Extended Code:** Add an overall parity bit.

From $[n, k, d]$ code $C$, create $[n+1, k, d']$ extended code $\bar{C}$:

$$\bar{C} = \{(c_1, \ldots, c_n, p) : (c_1, \ldots, c_n) \in C, \; p = \sum_{i=1}^n c_i\}$$

If $d$ is odd, then $d' = d + 1$ (all codewords now have even weight).

**Shortened Code:** Fix some positions to zero and remove them.

From $[n, k, d]$ code $C$, fixing position $i$ to zero gives $[n-1, k-1, d']$ code with $d' \geq d$.

**Punctured Code:** Remove a coordinate position.

From $[n, k, d]$ code $C$, deleting position $i$ gives $[n-1, k, d']$ code with $d' \geq d - 1$.

---

## Physical Interpretation

### Why Matrix Structure Matters

The generator and parity-check matrices encode the *logical structure* of the code:

- **G tells you how to encode:** Each row of G contributes parity bits
- **H tells you what to check:** Each row of H defines a parity constraint
- **Columns of H are "error signatures":** Each position has a unique syndrome contribution

### Error Detection via Null Space

Received word $\mathbf{r}$ is a valid codeword ⟺ $H\mathbf{r}^T = \mathbf{0}$ ⟺ $\mathbf{r}$ is in the null space of $H$.

If $\mathbf{r}$ is NOT in the null space, an error has occurred!

---

## Worked Examples

### Example 1: Converting to Systematic Form

**Problem:** Convert the following generator matrix to systematic form.

$$G = \begin{pmatrix} 1 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 & 1 \end{pmatrix}$$

**Solution:**

This is a [5, 3] code. We want $G' = [I_3 \mid P]$.

**Step 1:** Pivot on position (1,1). Row 1 has 1 in position 1. ✓

Eliminate column 1 in other rows:
- R2 ← R2 + R1: $(1,0,1,1,0) + (1,1,1,0,0) = (0,1,0,1,0)$

$$G_1 = \begin{pmatrix} 1 & 1 & 1 & 0 & 0 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 & 1 \end{pmatrix}$$

**Step 2:** Pivot on position (2,2). Row 2 has 1 in position 2. ✓

Eliminate column 2 in other rows:
- R1 ← R1 + R2: $(1,1,1,0,0) + (0,1,0,1,0) = (1,0,1,1,0)$
- R3 ← R3 + R2: $(0,1,1,0,1) + (0,1,0,1,0) = (0,0,1,1,1)$

$$G_2 = \begin{pmatrix} 1 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 1 & 1 \end{pmatrix}$$

**Step 3:** Pivot on position (3,3). Row 3 has 1 in position 3. ✓

Eliminate column 3 in other rows:
- R1 ← R1 + R3: $(1,0,1,1,0) + (0,0,1,1,1) = (1,0,0,0,1)$

$$G' = \begin{pmatrix} 1 & 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 1 & 1 \end{pmatrix} = [I_3 \mid P]$$

where $P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{pmatrix}$.

**Parity-check matrix:**
$$H = [P^T \mid I_2] = \begin{pmatrix} 0 & 1 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$ ∎

---

### Example 2: Finding Minimum Distance from H

**Problem:** Find the minimum distance of the code with parity-check matrix:

$$H = \begin{pmatrix} 1 & 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 & 1 \end{pmatrix}$$

**Solution:**

The code has parameters $[5, 3, d]$ where $d$ is to be determined.

**Check for $d \geq 2$:** Are all columns nonzero? Yes ✓

**Check for $d \geq 3$:** Are any 2 columns linearly dependent?
- Columns 1 and 2: $(1,0)^T + (0,1)^T = (1,1)^T \neq 0$ ✓
- Columns 1 and 3: $(1,0)^T + (1,1)^T = (0,1)^T \neq 0$ ✓
- Columns 1 and 4: $(1,0)^T + (1,0)^T = (0,0)^T$ ✗

Columns 1 and 4 are identical! They sum to zero.

**Conclusion:** $d = 2$ (the code can detect 1 error but not correct any). ∎

---

### Example 3: Designing a Code with d = 4

**Problem:** Construct a binary code with $n = 7$, $d = 4$.

**Solution:**

For $d = 4$, any 3 columns of $H$ must be linearly independent.

$H$ will have dimensions $(n-k) \times 7$. For any 3 columns to be independent, we need at least 3 rows (so columns can span a 3D space).

Try $H$ with 4 rows (giving a [7, 3, ≥4] code):

**Strategy:** Ensure no 3 columns sum to zero.

$$H = \begin{pmatrix} 1 & 1 & 1 & 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \\ 0 & 1 & 0 & 1 & 0 & 1 & 1 \end{pmatrix}$$

**Verification:** Check that no three columns sum to zero (tedious but doable).

Actually, let's use the extended Hamming code approach: add an overall parity row to the [7, 4, 3] Hamming code.

The extended [8, 4, 4] Hamming code has:
$$H = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{pmatrix}$$

This is a $[8, 4, 4]$ code. Puncturing (removing a column) gives a $[7, 4, 3]$ or $[7, 3, 4]$ depending on which column is removed. ∎

---

## Practice Problems

### Level 1: Direct Application

1. Convert to systematic form:
$$G = \begin{pmatrix} 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 & 1 \end{pmatrix}$$

2. Given the parity-check matrix, find the generator matrix:
$$H = \begin{pmatrix} 1 & 1 & 1 & 1 & 0 & 0 \\ 1 & 1 & 0 & 0 & 1 & 0 \\ 1 & 0 & 1 & 0 & 0 & 1 \end{pmatrix}$$

3. Determine the minimum distance of the code with:
$$H = \begin{pmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{pmatrix}$$

### Level 2: Intermediate

4. Prove that extending a code with odd minimum distance $d$ by adding an overall parity bit increases the distance to $d + 1$.

5. Show that the dual of the [7, 4, 3] Hamming code is a [7, 3, 4] code.

6. Construct a [6, 2, 4] binary code by finding an appropriate parity-check matrix.

### Level 3: Challenging

7. Prove that for a binary self-dual code, the minimum distance $d$ satisfies $d \equiv 0 \pmod{2}$.

8. Show that the number of codewords of weight $w$ in a code $C$ equals the number of codewords of weight $n - w$ in an appropriately chosen code (MacWilliams identity preview).

9. **Research:** How does the systematic encoding property relate to the structure of CSS quantum codes?

---

## Computational Lab

### Objective
Implement systematic encoding and verify the distance theorem computationally.

```python
"""
Day 674 Computational Lab: Generator and Parity-Check Matrices
Year 2: Advanced Quantum Science
"""

import numpy as np
from itertools import combinations

# =============================================================================
# Part 1: Row Reduction over GF(2)
# =============================================================================

print("=" * 60)
print("Part 1: Gaussian Elimination over GF(2)")
print("=" * 60)

def gf2_row_reduce(M):
    """
    Perform Gaussian elimination on matrix M over GF(2).
    Returns the row-reduced form and the list of pivot columns.
    """
    A = M.copy().astype(int)
    rows, cols = A.shape
    pivot_row = 0
    pivot_cols = []

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Find pivot in this column
        pivot_found = False
        for row in range(pivot_row, rows):
            if A[row, col] == 1:
                # Swap rows
                A[[pivot_row, row]] = A[[row, pivot_row]]
                pivot_found = True
                break

        if not pivot_found:
            continue

        pivot_cols.append(col)

        # Eliminate other 1s in this column
        for row in range(rows):
            if row != pivot_row and A[row, col] == 1:
                A[row] = (A[row] + A[pivot_row]) % 2

        pivot_row += 1

    return A, pivot_cols

# Test with our example
G_original = np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1]
], dtype=int)

print("\nOriginal generator matrix:")
print(G_original)

G_reduced, pivots = gf2_row_reduce(G_original)
print("\nRow-reduced form:")
print(G_reduced)
print(f"Pivot columns: {pivots}")

# =============================================================================
# Part 2: Convert to Systematic Form
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Systematic Form Conversion")
print("=" * 60)

def to_systematic_form(G):
    """
    Convert generator matrix to systematic form [I_k | P].
    Returns the systematic G and the column permutation applied.
    """
    k, n = G.shape
    A = G.copy().astype(int)

    # Record column permutation
    col_order = list(range(n))

    for i in range(k):
        # Find a pivot for row i
        pivot_col = None
        for j in range(i, n):
            if A[i, col_order[j]] == 1:
                # Check if this column can be a pivot
                col_idx = col_order[j]
                if all(A[row, col_idx] == 0 for row in range(i)) or i == 0:
                    pivot_col = j
                    break
            # Also check if we can swap to get a pivot
            for row in range(i, k):
                if A[row, col_order[j]] == 1:
                    # Swap rows
                    A[[i, row]] = A[[row, i]]
                    pivot_col = j
                    break
            if pivot_col is not None:
                break

        if pivot_col is None:
            # Try harder: look for any 1 in this row
            for j in range(n):
                if A[i, col_order[j]] == 1:
                    pivot_col = j
                    break

        if pivot_col is not None and pivot_col != i:
            # Swap columns in our ordering
            col_order[i], col_order[pivot_col] = col_order[pivot_col], col_order[i]

        # Now eliminate in the pivot column
        pivot_idx = col_order[i]
        for row in range(k):
            if row != i and A[row, pivot_idx] == 1:
                A[row] = (A[row] + A[i]) % 2

    # Reorder columns to match systematic form
    G_sys = A[:, col_order]
    return G_sys, col_order

G_sys, col_perm = to_systematic_form(G_original)
print("\nSystematic form:")
print(G_sys)
print(f"Column permutation: {col_perm}")

# Extract P matrix
k = G_sys.shape[0]
P = G_sys[:, k:]
print(f"\nParity matrix P:")
print(P)

# Construct H
n_minus_k = G_sys.shape[1] - k
H = np.hstack([P.T, np.eye(n_minus_k, dtype=int)])
print(f"\nParity-check matrix H:")
print(H)

# Verify G * H^T = 0
print(f"\nVerification G * H^T (mod 2):")
print(np.mod(G_sys @ H.T, 2))

# =============================================================================
# Part 3: Minimum Distance from H
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Computing Minimum Distance from H")
print("=" * 60)

def find_minimum_distance(H):
    """
    Find minimum distance by checking linear dependence of columns.
    Returns (d, dependent_cols) where dependent_cols is the first set found.
    """
    n = H.shape[1]

    # Check for d = 1 (zero column)
    for i in range(n):
        if np.sum(H[:, i]) == 0:
            return 1, [i]

    # Check for d = 2 (two identical columns)
    for i, j in combinations(range(n), 2):
        if np.array_equal(H[:, i], H[:, j]):
            return 2, [i, j]
        if np.all((H[:, i] + H[:, j]) % 2 == 0):
            return 2, [i, j]

    # Check for larger d
    for d in range(3, n + 1):
        for cols in combinations(range(n), d):
            col_sum = np.zeros(H.shape[0], dtype=int)
            for c in cols:
                col_sum = (col_sum + H[:, c]) % 2
            if np.all(col_sum == 0):
                return d, list(cols)

    return n + 1, []  # No dependence found

# Test with Hamming code
H_hamming = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1]
], dtype=int)

print("\nHamming [7, 4, 3] parity-check matrix:")
print(H_hamming)

d, dep_cols = find_minimum_distance(H_hamming)
print(f"\nMinimum distance: d = {d}")
print(f"First linearly dependent columns: {dep_cols}")

# Verify by showing these columns sum to zero
if dep_cols:
    print("\nVerification - column sum:")
    col_sum = np.zeros(H_hamming.shape[0], dtype=int)
    for c in dep_cols:
        print(f"  Column {c}: {H_hamming[:, c]}")
        col_sum = (col_sum + H_hamming[:, c]) % 2
    print(f"  Sum (mod 2): {col_sum}")

# =============================================================================
# Part 4: Systematic Encoding
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Systematic Encoding Algorithm")
print("=" * 60)

def systematic_encode(message, P):
    """
    Encode message using systematic form.
    codeword = [message | parity]
    where parity = message * P (mod 2)
    """
    message = np.array(message, dtype=int)
    parity = np.mod(message @ P, 2)
    return np.concatenate([message, parity])

# Hamming code P matrix
P_hamming = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
], dtype=int)

print("\nHamming code parity matrix P:")
print(P_hamming)

print("\nEncoding all 16 messages:")
print("-" * 50)
print(f"{'Message':<15} {'Codeword':<25} {'Weight'}")
print("-" * 50)

for m in range(16):
    message = [(m >> i) & 1 for i in range(4)]
    codeword = systematic_encode(message, P_hamming)
    weight = np.sum(codeword)
    print(f"{str(message):<15} {str(codeword):<25} {weight}")

# =============================================================================
# Part 5: Extended Hamming Code
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Extended Hamming [8, 4, 4] Code")
print("=" * 60)

def extend_code(G):
    """Add overall parity bit to create extended code."""
    k, n = G.shape
    parity_col = np.sum(G, axis=1, keepdims=True) % 2
    G_ext = np.hstack([G, parity_col])
    return G_ext

G_hamming = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=int)

G_ext = extend_code(G_hamming)
print("\nExtended Hamming generator matrix:")
print(G_ext)

# Generate all codewords and check weights
print("\nCodeword weights in extended code:")
weights = []
for m in range(16):
    message = [(m >> i) & 1 for i in range(4)]
    message = np.array(message, dtype=int)
    codeword = np.mod(message @ G_ext, 2)
    weights.append(np.sum(codeword))

print(f"Weight distribution: {sorted(set(weights))}")
print(f"All weights even: {all(w % 2 == 0 for w in weights)}")

# Find minimum distance
min_nonzero_weight = min(w for w in weights if w > 0)
print(f"Minimum distance (from weights): {min_nonzero_weight}")

# =============================================================================
# Part 6: Dual Code Analysis
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Dual Code of Hamming Code")
print("=" * 60)

# The dual of [7, 4, 3] is [7, 3, d⊥]
# Generator of dual = H of original
G_dual = H_hamming.copy()

print("\nDual code generator (= original H):")
print(G_dual)

print("\nDual codewords:")
dual_weights = []
for m in range(8):  # 2^3 messages
    message = [(m >> i) & 1 for i in range(3)]
    message = np.array(message, dtype=int)
    codeword = np.mod(message @ G_dual, 2)
    weight = np.sum(codeword)
    dual_weights.append(weight)
    print(f"  {message} -> {codeword}  (weight {weight})")

min_dual_weight = min(w for w in dual_weights if w > 0)
print(f"\nDual code minimum distance: {min_dual_weight}")
print(f"Dual code parameters: [7, 3, {min_dual_weight}]")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Systematic form | $G = [I_k \mid P]$, $H = [P^T \mid I_{n-k}]$ |
| Distance theorem | $d = \min\{w : w \text{ cols of } H \text{ lin. dep.}\}$ |
| Code as null space | $C = \ker(H)$ |
| Dual code | $C^\perp = \text{rowspace}(H) = \ker(G^T)$ |
| Extended code | $[n+1, k, d+1]$ from $[n, k, d]$ (odd $d$) |
| Punctured code | $[n-1, k, d-1]$ from $[n, k, d]$ |

### Main Takeaways

1. **Row reduction preserves the code** — only changes the basis
2. **Minimum distance = minimum linear dependence** of $H$ columns
3. **Systematic form** enables efficient encoding
4. **Code modifications** (extension, puncturing) adjust parameters predictably
5. **Dual codes** swap the roles of $G$ and $H$

---

## Daily Checklist

- [ ] Practice converting matrices to systematic form
- [ ] Use the distance theorem on at least 3 examples
- [ ] Verify G · H^T = 0 for a code you construct
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Understand why dual code structure matters for quantum codes

---

## Preview: Day 675

Tomorrow we study **syndrome decoding** — the algorithmic heart of error correction. You'll learn how to detect errors, identify their locations, and correct them using only the syndrome (never directly observing the codeword).

---

*"The key insight is that we can identify errors without identifying the data."*
— This is exactly what quantum error correction needs!

---

**Next:** [Day_675_Wednesday.md](Day_675_Wednesday.md) — Syndrome Decoding
