# Day 730: F₂ Vector Spaces

## Overview

**Day:** 730 of 1008
**Week:** 105 (Binary Representation & F₂ Linear Algebra)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Linear Algebra over the Binary Field

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | F₂ theory and vector spaces |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Gaussian elimination and algorithms |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Describe** the finite field F₂ and its algebraic properties
2. **Define** vector spaces over F₂ and their bases
3. **Perform** Gaussian elimination modulo 2
4. **Compute** rank, null space, and row space over F₂
5. **Apply** F₂ linear algebra to stabilizer codes
6. **Implement** efficient algorithms for binary matrix operations

---

## Core Content

### The Finite Field F₂

**Definition:**
The binary field F₂ = {0, 1} is the smallest finite field, with arithmetic defined modulo 2.

**Addition Table:**
| + | 0 | 1 |
|---|---|---|
| 0 | 0 | 1 |
| 1 | 1 | 0 |

Note: Addition is XOR, and 1 + 1 = 0.

**Multiplication Table:**
| × | 0 | 1 |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |

Multiplication is AND.

**Field Properties:**
- Additive identity: 0
- Multiplicative identity: 1
- Every non-zero element has multiplicative inverse: 1⁻¹ = 1
- Characteristic 2: 1 + 1 = 0, so -1 = 1
- All elements are self-inverse: a + a = 0

### Vector Spaces over F₂

**Definition:**
An F₂-vector space V is a set with:
- Vector addition: v + w ∈ V for v, w ∈ V
- Scalar multiplication: 0·v = 0, 1·v = v
- Satisfies usual vector space axioms

**Standard Space:**
$$\mathbb{F}_2^n = \{(x_1, x_2, \ldots, x_n) : x_i \in \{0, 1\}\}$$

This has $2^n$ elements.

**Key Difference from ℝⁿ:**
- No "negative" of a vector (a = -a)
- Limited scalars (only 0 and 1)
- Discrete structure (finite number of vectors)

### Linear Independence over F₂

**Definition:**
Vectors v₁, ..., vₖ ∈ F₂ⁿ are **linearly independent** if:
$$c_1 v_1 + c_2 v_2 + \cdots + c_k v_k = 0 \implies c_1 = c_2 = \cdots = c_k = 0$$

Since cᵢ ∈ {0, 1}, this means no non-trivial sum of a subset equals zero.

**Example:**
In F₂³:
- v₁ = (1, 0, 0), v₂ = (0, 1, 0), v₃ = (1, 1, 0)
- v₁ + v₂ + v₃ = (0, 0, 0) → linearly dependent!

### Basis and Dimension

**Basis:** A maximal linearly independent set, or equivalently, a minimal spanning set.

**Dimension:** For V ⊆ F₂ⁿ, dim(V) = number of vectors in any basis.

**Key Property:** Every subspace of F₂ⁿ has dimension at most n.

### Gaussian Elimination mod 2

The same algorithm as over ℝ, but all arithmetic is mod 2.

**Algorithm:**
1. Find leftmost column with a 1
2. Swap rows to put 1 in pivot position
3. Add pivot row to all other rows with 1 in that column
4. Repeat for remaining submatrix

**Key Simplification:** No division needed (since 1/1 = 1).

**Example:**
$$A = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}$$

Step 1: First column has pivots. Add R1 to R2:
$$\begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}$$

Step 2: Second column. Add R2 to R3:
$$\begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

Row echelon form achieved. Rank = 2.

### Row Echelon Form (REF)

**Definition:** A matrix is in REF if:
1. All zero rows are at the bottom
2. Leading 1 (pivot) of each row is to the right of pivot above
3. All entries below a pivot are 0

**Reduced REF (RREF):** Additionally:
4. All entries above each pivot are 0
5. Each pivot is 1 (automatic in F₂)

### Rank and Null Space

**Rank:**
$$\text{rank}(A) = \text{number of pivots in REF} = \dim(\text{row space})$$

**Null Space:**
$$\ker(A) = \{x \in \mathbb{F}_2^n : Ax = 0\}$$

**Rank-Nullity Theorem:**
$$\boxed{\text{rank}(A) + \dim(\ker(A)) = n}$$

where n = number of columns.

### Finding the Null Space

**Algorithm:**
1. Compute RREF of A
2. Express pivot variables in terms of free variables
3. Each free variable generates one basis vector for ker(A)

**Example:**
$$A = \begin{pmatrix} 1 & 1 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 1 \end{pmatrix}$$

RREF:
$$\begin{pmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Rank = 2, so dim(ker) = 4 - 2 = 2.

Free variables: x₃, x₄

From RREF:
- x₁ = x₃ + x₄
- x₂ = x₃

Setting (x₃, x₄) = (1, 0) and (0, 1):
$$\ker(A) = \text{span}\{(1, 1, 1, 0), (1, 0, 0, 1)\}$$

### Application to Stabilizer Codes

**Connection:** For an [[n, k, d]] stabilizer code:
- Parity check matrix H ∈ F₂^{(n-k) × 2n}
- Stabilizers span a subspace of dimension n-k
- Logical operators come from ker(H·Ω) modulo row space of H

**Key Operations:**
1. **Check independence:** Are stabilizers independent?
2. **Find logicals:** Compute vectors commuting with all stabilizers
3. **Compute distance:** Minimum weight in logical space

### Row Space and Column Space

**Row Space:**
$$\text{row}(A) = \text{span of rows of } A$$

**Column Space:**
$$\text{col}(A) = \text{span of columns of } A = \{Ax : x \in \mathbb{F}_2^n\}$$

**Fact:** rank(A) = dim(row(A)) = dim(col(A))

### Orthogonal Complement

**Definition:** For subspace V ⊆ F₂ⁿ:
$$V^\perp = \{w \in \mathbb{F}_2^n : \langle v, w \rangle = 0 \text{ for all } v \in V\}$$

where $\langle v, w \rangle = \sum_i v_i w_i \pmod{2}$.

**Dimension:**
$$\dim(V) + \dim(V^\perp) = n$$

**Self-orthogonal:** V is self-orthogonal if V ⊆ V⊥.

**Self-dual:** V is self-dual if V = V⊥.

### CSS Code Connection

CSS codes use classical codes C₁ and C₂ with C₂⊥ ⊆ C₁.

In F₂ terms:
- C₁ = ker(H₁) for some parity check H₁
- C₂ = ker(H₂)
- Condition: C₂⊥ ⊆ C₁ means row(H₂) ⊆ ker(H₁)

---

## Worked Examples

### Example 1: Complete Gaussian Elimination

Reduce the following matrix to RREF over F₂:
$$M = \begin{pmatrix}
1 & 0 & 1 & 1 & 0 \\
0 & 1 & 1 & 0 & 1 \\
1 & 1 & 0 & 1 & 1 \\
1 & 1 & 0 & 0 & 0
\end{pmatrix}$$

**Step 1:** Column 1 pivot at R1. Eliminate:
- R3 ← R3 + R1: (0, 1, 1, 0, 1)
- R4 ← R4 + R1: (0, 1, 1, 1, 0)

$$\begin{pmatrix}
1 & 0 & 1 & 1 & 0 \\
0 & 1 & 1 & 0 & 1 \\
0 & 1 & 1 & 0 & 1 \\
0 & 1 & 1 & 1 & 0
\end{pmatrix}$$

**Step 2:** Column 2 pivot at R2. Eliminate:
- R3 ← R3 + R2: (0, 0, 0, 0, 0)
- R4 ← R4 + R2: (0, 0, 0, 1, 1)

$$\begin{pmatrix}
1 & 0 & 1 & 1 & 0 \\
0 & 1 & 1 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1
\end{pmatrix}$$

**Step 3:** Swap R3 and R4:
$$\begin{pmatrix}
1 & 0 & 1 & 1 & 0 \\
0 & 1 & 1 & 0 & 1 \\
0 & 0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0 & 0
\end{pmatrix}$$

**Step 4:** Back-substitute for RREF:
- R1 ← R1 + R3: (1, 0, 1, 0, 1)

$$\text{RREF} = \begin{pmatrix}
1 & 0 & 1 & 0 & 1 \\
0 & 1 & 1 & 0 & 1 \\
0 & 0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0 & 0
\end{pmatrix}$$

**Rank = 3**, free variables: x₃, x₅

### Example 2: Computing Null Space

From the RREF above, find ker(M).

Pivot columns: 1, 2, 4
Free columns: 3, 5

From RREF:
- x₁ + x₃ + x₅ = 0 → x₁ = x₃ + x₅
- x₂ + x₃ + x₅ = 0 → x₂ = x₃ + x₅
- x₄ + x₅ = 0 → x₄ = x₅

For x₃ = 1, x₅ = 0:
$$n_1 = (1, 1, 1, 0, 0)$$

For x₃ = 0, x₅ = 1:
$$n_2 = (1, 1, 0, 1, 1)$$

$$\ker(M) = \text{span}\{(1,1,1,0,0), (1,1,0,1,1)\}$$

**Verification:**
$$M \cdot (1,1,1,0,0)^T = (1+1, 1+1, 0+1+1, 0+1+1) = (0,0,0,0)^T$$ ✓

### Example 3: Stabilizer Code Analysis

The [[4,2,2]] code has stabilizers:
$$S_1 = X_1 X_2 X_3 X_4, \quad S_2 = Z_1 Z_2 Z_3 Z_4$$

**Parity check matrix:**
$$H = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 1 & 1
\end{pmatrix}$$

Check self-orthogonality under symplectic form.

**Symplectic matrix:**
$$\Omega = \begin{pmatrix} 0 & I_4 \\ I_4 & 0 \end{pmatrix}$$

$$H \Omega H^T = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 1 & 1
\end{pmatrix}
\begin{pmatrix} 0 & I_4 \\ I_4 & 0 \end{pmatrix}
\begin{pmatrix}
1 & 0 \\ 1 & 0 \\ 1 & 0 \\ 1 & 0 \\
0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1
\end{pmatrix}$$

$$= \begin{pmatrix}
0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0
\end{pmatrix}
\begin{pmatrix}
1 & 0 \\ 1 & 0 \\ 1 & 0 \\ 1 & 0 \\
0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1
\end{pmatrix}
= \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$$

✓ Self-orthogonality verified.

---

## Practice Problems

### Level 1: Direct Application

1. **F₂ Arithmetic:** Compute over F₂:
   a) (1 + 1 + 1) × (1 + 0)
   b) 1 + 1 + 1 + 1 + 1
   c) (1, 0, 1) + (1, 1, 0)

2. **Linear Independence:** Determine if these vectors are linearly independent over F₂:
   a) (1, 0), (0, 1), (1, 1)
   b) (1, 1, 0), (0, 1, 1), (1, 0, 1)
   c) (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)

3. **RREF:** Find the RREF of:
   $$A = \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{pmatrix}$$

### Level 2: Intermediate

4. **Null Space:** Find a basis for ker(A) where:
   $$A = \begin{pmatrix}
   1 & 0 & 1 & 0 & 1 \\
   0 & 1 & 1 & 1 & 0 \\
   1 & 1 & 0 & 1 & 1
   \end{pmatrix}$$

5. **Rank Computation:** Find the rank of the parity check matrix for the [[7,1,3]] Steane code (6 × 14 matrix).

6. **Orthogonal Complement:** Let V = span{(1,0,1,0), (0,1,0,1)} in F₂⁴. Find V⊥.

### Level 3: Challenging

7. **Self-Orthogonality:** Prove that for a valid stabilizer code, the rows of H are mutually orthogonal under the symplectic inner product.

8. **Dimension Counting:** An [[n, k, d]] stabilizer code has parity check matrix H ∈ F₂^{(n-k) × 2n}.
   a) What is rank(H)?
   b) What is dim(ker(H)) in F₂^{2n}?
   c) How does this relate to the number of logical qubits?

9. **CSS Construction:** Given H₁ for a [7,4,3] Hamming code, construct the quantum parity check matrix for the resulting [[7,1,3]] CSS code. Verify self-orthogonality.

---

## Solutions

### Level 1 Solutions

1. **F₂ Arithmetic:**
   a) (1 + 1 + 1) × (1 + 0) = 1 × 1 = 1
   b) 1 + 1 + 1 + 1 + 1 = 0 + 1 + 1 + 1 = 0 + 0 + 1 = 1
   c) (1, 0, 1) + (1, 1, 0) = (0, 1, 1)

2. **Linear Independence:**
   a) (1, 0) + (0, 1) = (1, 1), so dependent
   b) Check: (1,1,0) + (0,1,1) + (1,0,1) = (0,0,0), dependent
   c) (1,0,0) + (0,1,0) + (0,0,1) + (1,1,1) = (0,0,0), dependent

3. **RREF:**
   R3 ← R3 + R1: (0, 1, 1)
   R3 ← R3 + R2: (0, 0, 0)
   R1 ← R1 + R2: (1, 0, 1)
   $$\text{RREF} = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

### Level 2 Solutions

4. **Null Space:**
   RREF:
   $$\begin{pmatrix}
   1 & 0 & 1 & 0 & 1 \\
   0 & 1 & 1 & 1 & 0 \\
   0 & 0 & 0 & 0 & 0
   \end{pmatrix}$$

   Free variables: x₃, x₄, x₅
   - x₁ = x₃ + x₅
   - x₂ = x₃ + x₄

   Basis: {(1,1,1,0,0), (0,1,0,1,0), (1,0,0,0,1)}

---

## Computational Lab

```python
"""
Day 730: F₂ Vector Spaces and Linear Algebra
=============================================
Implementation of linear algebra operations over F₂.
"""

import numpy as np
from typing import Tuple, List, Optional

def mod2(M: np.ndarray) -> np.ndarray:
    """Reduce matrix entries mod 2."""
    return M % 2

def gauss_eliminate_f2(M: np.ndarray) -> Tuple[np.ndarray, int, List[int]]:
    """
    Gaussian elimination over F₂ to row echelon form.

    Parameters:
    -----------
    M : np.ndarray
        Input matrix over F₂

    Returns:
    --------
    ref : np.ndarray
        Row echelon form
    rank : int
        Rank of the matrix
    pivots : List[int]
        Column indices of pivots
    """
    M = mod2(M.copy()).astype(int)
    nrows, ncols = M.shape
    pivots = []
    pivot_row = 0

    for col in range(ncols):
        if pivot_row >= nrows:
            break

        # Find pivot in current column
        pivot_idx = None
        for row in range(pivot_row, nrows):
            if M[row, col] == 1:
                pivot_idx = row
                break

        if pivot_idx is None:
            continue  # No pivot in this column

        # Swap rows
        if pivot_idx != pivot_row:
            M[[pivot_row, pivot_idx]] = M[[pivot_idx, pivot_row]]

        pivots.append(col)

        # Eliminate below
        for row in range(pivot_row + 1, nrows):
            if M[row, col] == 1:
                M[row] = mod2(M[row] + M[pivot_row])

        pivot_row += 1

    return M, len(pivots), pivots


def rref_f2(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute reduced row echelon form over F₂.

    Returns:
    --------
    rref : np.ndarray
        Reduced row echelon form
    pivots : List[int]
        Pivot column indices
    """
    ref, rank, pivots = gauss_eliminate_f2(M)

    # Back substitution
    for i in range(len(pivots) - 1, -1, -1):
        pivot_col = pivots[i]
        for j in range(i):
            if ref[j, pivot_col] == 1:
                ref[j] = mod2(ref[j] + ref[i])

    return ref, pivots


def null_space_f2(M: np.ndarray) -> np.ndarray:
    """
    Compute basis for null space of M over F₂.

    Parameters:
    -----------
    M : np.ndarray
        Matrix over F₂

    Returns:
    --------
    null_basis : np.ndarray
        Matrix whose rows form a basis for ker(M)
    """
    M = mod2(M).astype(int)
    nrows, ncols = M.shape

    rref, pivots = rref_f2(M)
    rank = len(pivots)

    # Free variables
    free_cols = [j for j in range(ncols) if j not in pivots]
    null_dim = ncols - rank

    if null_dim == 0:
        return np.zeros((0, ncols), dtype=int)

    # Build null space basis
    null_basis = np.zeros((null_dim, ncols), dtype=int)

    for i, free_col in enumerate(free_cols):
        null_basis[i, free_col] = 1

        # Express pivot variables in terms of this free variable
        for j, pivot_col in enumerate(pivots):
            null_basis[i, pivot_col] = rref[j, free_col]

    return mod2(null_basis)


def rank_f2(M: np.ndarray) -> int:
    """Compute rank of matrix over F₂."""
    _, rank, _ = gauss_eliminate_f2(M)
    return rank


def is_linearly_independent_f2(vectors: np.ndarray) -> bool:
    """Check if rows of matrix are linearly independent over F₂."""
    M = mod2(vectors).astype(int)
    return rank_f2(M) == M.shape[0]


def span_f2(vectors: np.ndarray) -> np.ndarray:
    """
    Find all vectors in the span of given vectors over F₂.

    Warning: exponential in dimension of span!
    """
    M = mod2(vectors).astype(int)
    rref, pivots = rref_f2(M)
    rank = len(pivots)

    # Extract basis
    basis = rref[:rank]

    # Generate all 2^rank combinations
    span_vectors = []
    for i in range(2**rank):
        coeffs = np.array([(i >> j) & 1 for j in range(rank)])
        v = mod2(coeffs @ basis)
        span_vectors.append(v)

    return np.unique(span_vectors, axis=0)


def orthogonal_complement_f2(V: np.ndarray) -> np.ndarray:
    """
    Compute orthogonal complement V^⊥ over F₂.

    V^⊥ = {w : <v,w> = 0 for all v in V}
    """
    return null_space_f2(V.T)


def symplectic_form_matrix(n: int) -> np.ndarray:
    """Create 2n × 2n symplectic form matrix Ω."""
    return np.block([
        [np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
        [np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]
    ])


def check_symplectic_orthogonality(H: np.ndarray) -> bool:
    """
    Check if H satisfies symplectic self-orthogonality: H Ω H^T = 0

    H should be an (n-k) × 2n matrix for stabilizer code.
    """
    n = H.shape[1] // 2
    Omega = symplectic_form_matrix(n)
    result = mod2(H @ Omega @ H.T)
    return np.all(result == 0)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 730: F₂ Vector Spaces and Linear Algebra")
    print("=" * 60)

    # Example 1: Basic Gaussian elimination
    print("\n1. Gaussian Elimination over F₂")
    print("-" * 40)

    M = np.array([
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 0, 0]
    ])

    print("Original matrix:")
    print(M)

    rref, pivots = rref_f2(M)
    print(f"\nRREF:")
    print(rref)
    print(f"Pivot columns: {pivots}")
    print(f"Rank: {len(pivots)}")

    # Example 2: Null space computation
    print("\n2. Null Space Computation")
    print("-" * 40)

    null_basis = null_space_f2(M)
    print(f"Null space basis (dim = {len(null_basis)}):")
    print(null_basis)

    print("\nVerification (M × null_vectors should be 0):")
    for i, v in enumerate(null_basis):
        result = mod2(M @ v)
        print(f"  M × n{i+1} = {result}")

    # Example 3: Linear independence check
    print("\n3. Linear Independence Check")
    print("-" * 40)

    v1 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    print("Vectors:")
    print(v1)
    print(f"Linearly independent: {is_linearly_independent_f2(v1)}")

    v2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    print("\nVectors:")
    print(v2)
    print(f"Linearly independent: {is_linearly_independent_f2(v2)}")

    # Example 4: Orthogonal complement
    print("\n4. Orthogonal Complement")
    print("-" * 40)

    V = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    print("Subspace V (rows):")
    print(V)

    V_perp = orthogonal_complement_f2(V)
    print(f"\nOrthogonal complement V^⊥:")
    print(V_perp)

    print("\nVerification (v · w should be 0):")
    for v in V:
        for w in V_perp:
            print(f"  {v} · {w} = {np.dot(v, w) % 2}")

    # Example 5: Stabilizer code analysis
    print("\n5. Stabilizer Code Symplectic Orthogonality")
    print("-" * 40)

    # [[5,1,3]] code parity check matrix
    H_513 = np.array([
        [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],  # XZZXI
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # IXZZX
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],  # XIXZZ
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]   # ZXIXZ
    ])

    print("[[5,1,3]] code parity check matrix H:")
    print(H_513)

    is_valid = check_symplectic_orthogonality(H_513)
    print(f"\nSymplectic self-orthogonality (H Ω H^T = 0): {is_valid}")

    # Show H Ω H^T explicitly
    n = 5
    Omega = symplectic_form_matrix(n)
    HOHT = mod2(H_513 @ Omega @ H_513.T)
    print(f"\nH Ω H^T mod 2:")
    print(HOHT)

    # Example 6: Finding logical operators
    print("\n6. Finding Logical Operators")
    print("-" * 40)

    # Logical operators must commute with all stabilizers
    # They are in ker(H Ω) but not in row(H)

    # Find vectors commuting with H (in symplectic sense)
    # v commutes with all rows of H iff v^T Ω h_i = 0 for all i
    # iff (H Ω)^T v = 0
    # iff v in ker((H Ω)^T) = ker(Ω^T H^T) = ker(Ω H^T)

    HOmega = mod2(H_513 @ Omega)
    centralizer_basis = null_space_f2(HOmega.T)

    print("Centralizer basis (vectors commuting with H):")
    print(f"Dimension: {len(centralizer_basis)}")

    # The logical operators are centralizer minus stabilizer span
    print("\nStabilizer rank:", rank_f2(H_513))
    print("Centralizer dimension:", len(centralizer_basis))
    print("Logical qubits k = (centralizer_dim - 2*rank)/2 =",
          (len(centralizer_basis) - 2*rank_f2(H_513))//2 + rank_f2(H_513))

    print("\n" + "=" * 60)
    print("End of Day 730 Lab")
    print("=" * 60)
```

**Expected Output:**
```
============================================================
Day 730: F₂ Vector Spaces and Linear Algebra
============================================================

1. Gaussian Elimination over F₂
----------------------------------------
Original matrix:
[[1 0 1 1 0]
 [0 1 1 0 1]
 [1 1 0 1 1]
 [1 1 0 0 0]]

RREF:
[[1 0 1 0 1]
 [0 1 1 0 1]
 [0 0 0 1 1]
 [0 0 0 0 0]]
Pivot columns: [0, 1, 3]
Rank: 3

2. Null Space Computation
----------------------------------------
Null space basis (dim = 2):
[[1 1 1 0 0]
 [1 1 0 1 1]]

Verification (M × null_vectors should be 0):
  M × n1 = [0 0 0 0]
  M × n2 = [0 0 0 0]
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| F₂ arithmetic | $1 + 1 = 0$, $a \cdot a = a$ |
| Rank-Nullity | $\text{rank}(A) + \dim(\ker A) = n$ |
| Orthogonal complement | $\dim(V) + \dim(V^\perp) = n$ |
| Self-orthogonality | $H \Omega H^T = 0$ for stabilizer codes |

### Main Takeaways

1. **F₂ arithmetic** is simple: addition is XOR, multiplication is AND
2. **Gaussian elimination** works over F₂ with no division needed
3. **Null space** and **rank** are computed the same way as over ℝ
4. **Stabilizer codes** require symplectic self-orthogonality
5. **Logical operators** come from the symplectic centralizer modulo stabilizers

---

## Daily Checklist

- [ ] I can perform arithmetic in F₂
- [ ] I can determine linear independence over F₂
- [ ] I can compute RREF using Gaussian elimination mod 2
- [ ] I can find the null space of a matrix over F₂
- [ ] I understand the rank-nullity theorem over F₂
- [ ] I can check symplectic self-orthogonality

---

## Preview: Day 731

Tomorrow we study the **Symplectic Inner Product** in depth:
- Formal definition and properties
- Symplectic vector spaces
- Lagrangian subspaces and stabilizer codes
- Symplectic group and Clifford connection
- Computing with the symplectic form
