# Day 733: Parity Check Matrices

## Overview

**Day:** 733 of 1008
**Week:** 105 (Binary Representation & F₂ Linear Algebra)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Parity Check Matrix Formalism for Stabilizer Codes

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Parity check construction |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Standard form and applications |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Construct** the parity check matrix H from stabilizer generators
2. **Transform** H to standard form using row operations
3. **Derive** encoding circuits from the standard form
4. **Compute** syndromes using H
5. **Identify** CSS code structure in H
6. **Apply** these techniques to common codes

---

## Core Content

### Parity Check Matrix Definition

**Definition:**
For an [[n, k, d]] stabilizer code with generators $S_1, \ldots, S_{n-k}$, the **parity check matrix** H ∈ F₂^{(n-k) × 2n} has rows given by the binary representation of each generator:

$$H = \begin{pmatrix}
— \text{bin}(S_1) — \\
— \text{bin}(S_2) — \\
\vdots \\
— \text{bin}(S_{n-k}) —
\end{pmatrix}$$

**Block Form:**
$$\boxed{H = (H_X | H_Z)}$$

where $H_X$ contains the X-parts and $H_Z$ contains the Z-parts.

### Self-Orthogonality Condition

**Theorem:**
H defines a valid stabilizer code if and only if:
$$\boxed{H \Omega H^T = 0}$$

where $\Omega = \begin{pmatrix} 0 & I_n \\ I_n & 0 \end{pmatrix}$.

**Expanded form:**
$$H_X H_Z^T + H_Z H_X^T = 0 \pmod{2}$$

This is equivalent to all stabilizer generators commuting pairwise.

### Row Operations on H

**Allowed operations** (preserve the code):
1. **Row swap:** Reorders generators
2. **Row addition:** $S_i \to S_i S_j$ (still a stabilizer)
3. **Column operations within X or Z block:** Qubit relabeling

**Not allowed:**
- Mixing X and Z columns arbitrarily
- Non-symplectic column operations

### Standard Form

**Theorem (Standard Form):**
Any stabilizer parity check matrix can be transformed to:

$$H_{\text{std}} = \left(\begin{array}{ccc|ccc}
I_r & A_1 & A_2 & B & C_1 & C_2 \\
0 & 0 & 0 & D & I_s & E
\end{array}\right)$$

where:
- $r + s = n - k$ (total number of generators)
- First block: r rows with X-pivots
- Second block: s rows with Z-pivots only

**CSS Standard Form:**
For CSS codes (X and Z stabilizers separate):
$$H_{\text{CSS}} = \left(\begin{array}{c|c}
H_1 & 0 \\
0 & H_2
\end{array}\right)$$

### Encoding from Standard Form

**Key Insight:** The standard form directly gives the encoding circuit.

**Logical operators** can be read off:
- $\bar{X}_j$: Acts on "logical" qubits with corrections from A, B matrices
- $\bar{Z}_j$: Similarly determined by the structure

**Encoding procedure:**
1. Initialize k data qubits in logical state
2. Initialize n-k ancilla qubits in $|0\rangle$
3. Apply CNOT gates according to matrix structure
4. Apply Hadamards for X-type stabilizers

### Syndrome Extraction

For error E with binary representation $e = (e_X | e_Z)$:

**Syndrome:**
$$\boxed{\mathbf{s} = H \Omega e^T = H_X e_Z^T + H_Z e_X^T}$$

The syndrome is an (n-k)-bit string identifying which stabilizers anticommute with E.

**Interpretation:**
- $s_i = 0$: Error commutes with $S_i$
- $s_i = 1$: Error anticommutes with $S_i$

### Example: [[7,1,3]] Steane Code

**Stabilizers:**
$$S_1 = IIIXXXX, \quad S_2 = IXXIIXX, \quad S_3 = XIXIXIX$$
$$S_4 = IIIZZZZ, \quad S_5 = IZZIIZZ, \quad S_6 = ZIZIZIZ$$

**Parity check matrix:**
$$H = \left(\begin{array}{ccccccc|ccccccc}
0 & 0 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1
\end{array}\right)$$

**CSS Structure:**
$$H_X = \begin{pmatrix}
0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0 & 1
\end{pmatrix} = H_1$$

$$H_Z = \begin{pmatrix}
0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0 & 1
\end{pmatrix} = H_1$$

Note: $H_X = H_Z$ for the Steane code (self-dual structure).

### Example: [[5,1,3]] Perfect Code

**Stabilizers:**
$$S_1 = XZZXI, \quad S_2 = IXZZX, \quad S_3 = XIXZZ, \quad S_4 = ZXIXZ$$

**Parity check matrix:**
$$H = \left(\begin{array}{ccccc|ccccc}
1 & 0 & 0 & 1 & 0 & 0 & 1 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 1 & 0 & 0 & 1 & 1 & 0 \\
1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 1 \\
0 & 1 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 1
\end{array}\right)$$

**Note:** This is NOT CSS (X and Z are mixed in each stabilizer).

### Degeneracy and the Parity Check Matrix

**Degenerate errors:** Multiple errors with the same syndrome.

If $E_1$ and $E_2$ have the same syndrome:
$$H \Omega e_1^T = H \Omega e_2^T \implies H \Omega (e_1 + e_2)^T = 0$$

This means $E_1 E_2$ commutes with all stabilizers, so $E_1 E_2 \in S$ or is a logical operator.

**Degeneracy is good:** Different physical errors can have equivalent effects.

### Generator Matrix

The **generator matrix** G describes the code space:

For an [[n, k]] code:
- G ∈ F₂^{2k × 2n}
- Rows are binary representations of logical operators
- $\bar{X}_1, \ldots, \bar{X}_k, \bar{Z}_1, \ldots, \bar{Z}_k$

**Relationship:**
$$G \Omega H^T = 0$$

Logical operators commute with all stabilizers.

---

## Worked Examples

### Example 1: Building H for [[4,2,2]] Code

The [[4,2,2]] code has stabilizers:
$$S_1 = X_1 X_2 X_3 X_4, \quad S_2 = Z_1 Z_2 Z_3 Z_4$$

**Binary representations:**
- $S_1 = XXXX \leftrightarrow (1,1,1,1 | 0,0,0,0)$
- $S_2 = ZZZZ \leftrightarrow (0,0,0,0 | 1,1,1,1)$

**Parity check matrix:**
$$H = \begin{pmatrix}
1 & 1 & 1 & 1 & | & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & | & 1 & 1 & 1 & 1
\end{pmatrix}$$

**Verify self-orthogonality:**
$$H_X H_Z^T = \begin{pmatrix} 1&1&1&1 \\ 0&0&0&0 \end{pmatrix} \begin{pmatrix} 0&1 \\ 0&1 \\ 0&1 \\ 0&1 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$$

$$H_Z H_X^T = \begin{pmatrix} 0&0&0&0 \\ 1&1&1&1 \end{pmatrix} \begin{pmatrix} 1&0 \\ 1&0 \\ 1&0 \\ 1&0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$$

✓ Self-orthogonal.

### Example 2: Computing Syndromes

For the Steane code, compute the syndrome of error $E = X_3$ (X error on qubit 3).

**Error vector:**
$e = (0,0,1,0,0,0,0 | 0,0,0,0,0,0,0)$

**Syndrome:**
$$\mathbf{s} = H_X e_Z^T + H_Z e_X^T$$

Since $e_Z = (0,0,0,0,0,0,0)$:
$$H_X e_Z^T = 0$$

For $H_Z e_X^T$:
$$H_Z e_X^T = \begin{pmatrix}
0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0 & 1
\end{pmatrix} \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}$$

The syndrome for the Z-stabilizers is $(0, 1, 1)$.

The full syndrome is $\mathbf{s} = (0, 0, 0, 0, 1, 1)$.

**Interpretation:** $X_3$ anticommutes with $S_5$ and $S_6$ (the Z-stabilizers).

### Example 3: Transforming to Standard Form

Start with the [[4,2,2]] code H and transform to standard form.

$$H = \begin{pmatrix}
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 1 & 1
\end{pmatrix}$$

This is already in a form close to standard:
- Row 1 has X-pivot in column 1
- Row 2 has Z-pivot in column 5

**Standard form:**
$$H_{\text{std}} = \begin{pmatrix}
1 & 1 & 1 & 1 & | & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & | & 1 & 1 & 1 & 1
\end{pmatrix}$$

**Reading off structure:**
- r = 1 (one X-pivot)
- s = 1 (one Z-pivot)
- n - k = 2 ✓

---

## Practice Problems

### Level 1: Direct Application

1. **H Construction:** Write the parity check matrix for:
   a) The 3-qubit bit flip code: $S_1 = Z_1Z_2$, $S_2 = Z_2Z_3$
   b) The 3-qubit phase flip code: $S_1 = X_1X_2$, $S_2 = X_2X_3$

2. **Self-Orthogonality:** Verify that $H\Omega H^T = 0$ for the [[5,1,3]] code.

3. **Syndrome Calculation:** For the [[4,2,2]] code, compute syndromes for:
   a) $E = X_1$
   b) $E = Z_2$
   c) $E = Y_3 = iX_3Z_3$

### Level 2: Intermediate

4. **Standard Form:** Transform the following H to standard form:
   $$H = \begin{pmatrix}
   1 & 1 & 0 & 0 & | & 0 & 0 & 1 & 1 \\
   0 & 0 & 1 & 1 & | & 1 & 1 & 0 & 0
   \end{pmatrix}$$

5. **Logical Operators:** For the [[4,2,2]] code with the H above:
   a) Find two logical X operators
   b) Find two logical Z operators
   c) Verify they have the correct commutation relations

6. **CSS Identification:** Determine if the following H defines a CSS code:
   $$H = \begin{pmatrix}
   1 & 0 & 1 & 0 & | & 0 & 1 & 0 & 1 \\
   0 & 1 & 0 & 1 & | & 1 & 0 & 1 & 0
   \end{pmatrix}$$

### Level 3: Challenging

7. **Encoding Circuit:** From the standard form of the Steane code, derive the encoding circuit that maps $|\psi\rangle \to |\bar{\psi}\rangle$.

8. **Dual Code:** For a CSS code with $H = (H_1 | 0; 0 | H_2)$:
   a) Show that the dual code has $H' = (H_2 | 0; 0 | H_1)$
   b) What is the relationship between logical X and Z operators?

9. **Generator Matrix:** For the [[5,1,3]] code:
   a) Find the 2×10 generator matrix G
   b) Verify $G \Omega H^T = 0$
   c) Verify logical X and Z anticommute

---

## Solutions

### Level 1 Solutions

1. **H Construction:**
   a) Bit flip code:
      $S_1 = Z_1Z_2 \leftrightarrow (0,0,0 | 1,1,0)$
      $S_2 = Z_2Z_3 \leftrightarrow (0,0,0 | 0,1,1)$
      $$H = \begin{pmatrix} 0&0&0 & | & 1&1&0 \\ 0&0&0 & | & 0&1&1 \end{pmatrix}$$

   b) Phase flip code:
      $S_1 = X_1X_2 \leftrightarrow (1,1,0 | 0,0,0)$
      $S_2 = X_2X_3 \leftrightarrow (0,1,1 | 0,0,0)$
      $$H = \begin{pmatrix} 1&1&0 & | & 0&0&0 \\ 0&1&1 & | & 0&0&0 \end{pmatrix}$$

3. **Syndromes for [[4,2,2]]:**
   a) $E = X_1$: $e = (1,0,0,0 | 0,0,0,0)$
      $\mathbf{s} = H_Z e_X^T = (0,0,0,0 | 1,1,1,1)(1,0,0,0)^T = (0, 1)$

   b) $E = Z_2$: $e = (0,0,0,0 | 0,1,0,0)$
      $\mathbf{s} = H_X e_Z^T = (1,1,1,1)(0,1,0,0)^T = (1, 0)$

   c) $E = Y_3$: $e = (0,0,1,0 | 0,0,1,0)$
      $\mathbf{s} = (1,1,1,1)(0,0,1,0)^T + (1,1,1,1)(0,0,1,0)^T = (1,1)$

---

## Computational Lab

```python
"""
Day 733: Parity Check Matrices for Stabilizer Codes
====================================================
Implementation of H matrix construction and manipulation.
"""

import numpy as np
from typing import List, Tuple, Optional

def mod2(M: np.ndarray) -> np.ndarray:
    """Reduce mod 2."""
    return np.array(M) % 2

def symplectic_matrix(n: int) -> np.ndarray:
    """Create 2n × 2n symplectic form matrix Ω."""
    return np.block([
        [np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
        [np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]
    ])

def pauli_to_binary(pauli_str: str) -> np.ndarray:
    """Convert Pauli string to binary (a|b) vector."""
    n = len(pauli_str)
    a = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)
    for i, p in enumerate(pauli_str.upper()):
        if p == 'X':
            a[i] = 1
        elif p == 'Z':
            b[i] = 1
        elif p == 'Y':
            a[i] = 1
            b[i] = 1
    return np.concatenate([a, b])

def binary_to_pauli(v: np.ndarray) -> str:
    """Convert binary vector to Pauli string."""
    n = len(v) // 2
    a, b = v[:n], v[n:]
    chars = []
    for i in range(n):
        if a[i] == 0 and b[i] == 0:
            chars.append('I')
        elif a[i] == 1 and b[i] == 0:
            chars.append('X')
        elif a[i] == 0 and b[i] == 1:
            chars.append('Z')
        else:
            chars.append('Y')
    return ''.join(chars)

def build_parity_check(stabilizers: List[str]) -> np.ndarray:
    """
    Build parity check matrix H from stabilizer strings.

    Parameters:
    -----------
    stabilizers : List[str]
        List of Pauli strings for stabilizer generators

    Returns:
    --------
    H : np.ndarray
        Parity check matrix (n-k) × 2n
    """
    return np.array([pauli_to_binary(s) for s in stabilizers])

def check_self_orthogonality(H: np.ndarray) -> bool:
    """
    Check if H satisfies H Ω H^T = 0 (mod 2).
    """
    n = H.shape[1] // 2
    Omega = symplectic_matrix(n)
    result = mod2(H @ Omega @ H.T)
    return np.all(result == 0)

def compute_syndrome(H: np.ndarray, error: np.ndarray) -> np.ndarray:
    """
    Compute syndrome s = H Ω e^T.

    Parameters:
    -----------
    H : np.ndarray
        Parity check matrix
    error : np.ndarray
        Error vector (a|b) in F_2^{2n}

    Returns:
    --------
    syndrome : np.ndarray
        Syndrome vector in F_2^{n-k}
    """
    n = H.shape[1] // 2
    Omega = symplectic_matrix(n)
    return mod2(H @ Omega @ error)

def is_css_code(H: np.ndarray) -> bool:
    """
    Check if H has CSS structure (X and Z generators separate).

    CSS codes have H = [[H_X, 0], [0, H_Z]] structure.
    """
    n = H.shape[1] // 2
    n_k = H.shape[0]

    H_X = H[:, :n]
    H_Z = H[:, n:]

    # Check if each row is purely X or purely Z
    for i in range(n_k):
        x_part = np.any(H_X[i] != 0)
        z_part = np.any(H_Z[i] != 0)
        if x_part and z_part:
            return False
    return True

def get_css_structure(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract H_X and H_Z from a CSS code's parity check matrix.

    Returns matrices with only the non-trivial rows.
    """
    n = H.shape[1] // 2

    H_X = H[:, :n]
    H_Z = H[:, n:]

    # Find X-only and Z-only rows
    x_rows = []
    z_rows = []

    for i in range(H.shape[0]):
        x_part = np.any(H_X[i] != 0)
        z_part = np.any(H_Z[i] != 0)

        if x_part and not z_part:
            x_rows.append(H_X[i])
        elif z_part and not x_part:
            z_rows.append(H_Z[i])

    return np.array(x_rows) if x_rows else np.zeros((0, n), dtype=int), \
           np.array(z_rows) if z_rows else np.zeros((0, n), dtype=int)

def rref_f2(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Compute RREF over F₂."""
    M = mod2(M.copy())
    nrows, ncols = M.shape
    pivots = []
    pivot_row = 0

    for col in range(ncols):
        if pivot_row >= nrows:
            break

        # Find pivot
        pivot_idx = None
        for row in range(pivot_row, nrows):
            if M[row, col] == 1:
                pivot_idx = row
                break

        if pivot_idx is None:
            continue

        # Swap rows
        M[[pivot_row, pivot_idx]] = M[[pivot_idx, pivot_row]]
        pivots.append(col)

        # Eliminate all other rows
        for row in range(nrows):
            if row != pivot_row and M[row, col] == 1:
                M[row] = mod2(M[row] + M[pivot_row])

        pivot_row += 1

    return M, pivots

def to_standard_form(H: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Transform H to standard form.

    Standard form:
    [[I_r, A1, A2 | B, C1, C2],
     [0,   0,  0  | D, I_s, E]]

    Returns:
    --------
    H_std : np.ndarray
        H in standard form
    x_pivots : List[int]
        Pivot columns in X block
    z_pivots : List[int]
        Pivot columns in Z block
    """
    H = mod2(H.copy())
    n = H.shape[1] // 2
    n_k = H.shape[0]

    # First pass: RREF on X block
    H_x_rref, x_pivots = rref_f2(H[:, :n])

    # Apply same row operations to full H
    # (simplified version - just do RREF on full H)
    H_std, full_pivots = rref_f2(H)

    # Identify X-pivots and Z-pivots
    x_pivots = [p for p in full_pivots if p < n]
    z_pivots = [p - n for p in full_pivots if p >= n]

    return H_std, x_pivots, z_pivots

def find_logical_operators(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find logical X and Z operators from H.

    Logical operators are in ker(H Ω) but not in row(H).

    Returns:
    --------
    logical_X : np.ndarray
        k×2n matrix with logical X operators
    logical_Z : np.ndarray
        k×2n matrix with logical Z operators
    """
    n = H.shape[1] // 2
    n_k = H.shape[0]
    k = n - n_k  # Number of logical qubits

    Omega = symplectic_matrix(n)

    # Find centralizer: ker((H Ω)^T)
    HO = mod2(H @ Omega)
    # Need null space of HO.T

    # Use RREF to find null space
    rref, pivots = rref_f2(HO.T)
    rank = len(pivots)

    # Free columns form null space basis
    free_cols = [j for j in range(2*n) if j not in pivots]

    null_basis = np.zeros((len(free_cols), 2*n), dtype=int)
    for i, fc in enumerate(free_cols):
        null_basis[i, fc] = 1
        for j, pc in enumerate(pivots):
            if rref[j, fc] == 1:
                null_basis[i, pc] = 1

    # Centralizer has dimension 2n - rank(H) = 2n - (n-k) = n + k
    # Stabilizers are dimension n-k
    # Logical space has dimension 2k

    # Separate into X-type and Z-type logicals
    # (Simplified - return first k pairs)
    logical_X = null_basis[:k] if len(null_basis) >= k else null_basis
    logical_Z = null_basis[k:2*k] if len(null_basis) >= 2*k else null_basis[-k:]

    return logical_X, logical_Z

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 733: Parity Check Matrices")
    print("=" * 60)

    # Example 1: Build H for common codes
    print("\n1. Building Parity Check Matrices")
    print("-" * 40)

    # [[4,2,2]] code
    stabs_422 = ['XXXX', 'ZZZZ']
    H_422 = build_parity_check(stabs_422)
    print("[[4,2,2]] code:")
    print(f"Stabilizers: {stabs_422}")
    print(f"H =\n{H_422}")
    print(f"Self-orthogonal: {check_self_orthogonality(H_422)}")

    # [[5,1,3]] code
    stabs_513 = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
    H_513 = build_parity_check(stabs_513)
    print("\n[[5,1,3]] code:")
    print(f"Stabilizers: {stabs_513}")
    print(f"H =\n{H_513}")
    print(f"Self-orthogonal: {check_self_orthogonality(H_513)}")

    # [[7,1,3]] Steane code
    stabs_713 = [
        'IIIXXXX', 'IXXIIXX', 'XIXIXIX',
        'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ'
    ]
    H_713 = build_parity_check(stabs_713)
    print("\n[[7,1,3]] Steane code:")
    print(f"Self-orthogonal: {check_self_orthogonality(H_713)}")
    print(f"Is CSS: {is_css_code(H_713)}")

    # Example 2: Syndrome computation
    print("\n2. Syndrome Computation")
    print("-" * 40)

    errors_422 = ['XIII', 'IXII', 'ZIIII'[:4], 'YIII']
    print("[[4,2,2]] code syndromes:")
    for e_str in errors_422:
        if len(e_str) < 4:
            e_str = e_str + 'I' * (4 - len(e_str))
        e_str = e_str[:4]
        e = pauli_to_binary(e_str)
        s = compute_syndrome(H_422, e)
        print(f"  E = {e_str}: syndrome = {s}")

    # Example 3: Steane code syndromes
    print("\n3. Steane Code Syndrome Table")
    print("-" * 40)

    single_errors = []
    for i in range(7):
        for p in ['X', 'Z', 'Y']:
            e_str = 'I' * i + p + 'I' * (6 - i)
            single_errors.append(e_str)

    print("Single-qubit errors:")
    for e_str in single_errors[:12]:  # First 12
        e = pauli_to_binary(e_str)
        s = compute_syndrome(H_713, e)
        print(f"  {e_str}: s = {s}")

    # Example 4: CSS structure
    print("\n4. CSS Structure Analysis")
    print("-" * 40)

    print("[[7,1,3]] Steane code CSS structure:")
    H_X_css, H_Z_css = get_css_structure(H_713)
    print(f"H_X (X-stabilizers):\n{H_X_css}")
    print(f"H_Z (Z-stabilizers):\n{H_Z_css}")
    print(f"H_X == H_Z: {np.array_equal(H_X_css, H_Z_css)}")

    # Example 5: Standard form
    print("\n5. Standard Form Transformation")
    print("-" * 40)

    print("[[4,2,2]] code:")
    H_std, x_piv, z_piv = to_standard_form(H_422)
    print(f"Standard form:\n{H_std}")
    print(f"X-pivots: {x_piv}, Z-pivots: {z_piv}")

    print("\n[[5,1,3]] code:")
    H_std_513, x_piv_513, z_piv_513 = to_standard_form(H_513)
    print(f"Standard form:\n{H_std_513}")
    print(f"X-pivots: {x_piv_513}, Z-pivots: {z_piv_513}")

    # Example 6: Degenerate errors
    print("\n6. Degenerate Errors")
    print("-" * 40)

    # Find errors with same syndrome in [[4,2,2]]
    print("[[4,2,2]] code - errors with same syndrome:")
    syndrome_dict = {}
    for i in range(4):
        for p in ['I', 'X', 'Y', 'Z']:
            e_str = 'I' * i + p + 'I' * (3 - i)
            e = pauli_to_binary(e_str)
            s = tuple(compute_syndrome(H_422, e))
            if s not in syndrome_dict:
                syndrome_dict[s] = []
            syndrome_dict[s].append(e_str)

    for s, errors in syndrome_dict.items():
        if len(errors) > 1:
            print(f"  Syndrome {s}: {errors}")

    print("\n" + "=" * 60)
    print("End of Day 733 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Parity check | $H = (H_X \| H_Z) \in \mathbb{F}_2^{(n-k) \times 2n}$ |
| Self-orthogonality | $H \Omega H^T = 0$ |
| Syndrome | $\mathbf{s} = H \Omega e^T$ |
| CSS structure | $H = (H_1 \| 0; 0 \| H_2)$ |
| Centralizer | $\ker(H \Omega)$ |

### Main Takeaways

1. **The parity check matrix H** encodes all stabilizer structure
2. **Self-orthogonality** $H\Omega H^T = 0$ ensures valid stabilizer code
3. **Syndromes** identify which stabilizers anticommute with an error
4. **CSS codes** have block-diagonal H structure
5. **Standard form** reveals encoding circuit structure

---

## Daily Checklist

- [ ] I can construct H from stabilizer generators
- [ ] I can verify self-orthogonality
- [ ] I can compute syndromes for errors
- [ ] I can identify CSS structure in H
- [ ] I understand the standard form transformation
- [ ] I can find logical operators from H

---

## Preview: Day 734

Tomorrow we study **Logical Operators & Distance**:
- Computing logical operators from null space
- The normalizer vs centralizer distinction
- Code distance from minimum weight
- Algorithms for distance computation
- Bounds: Singleton, Hamming, quantum Singleton
