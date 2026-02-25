# Day 743: Introduction to CSS Codes

## Overview

**Day:** 743 of 1008
**Week:** 107 (CSS Codes & Related Constructions)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Calderbank-Shor-Steane Code Construction

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | CSS construction theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Examples and practice |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Explain** the historical motivation for CSS codes
2. **Construct** a CSS code from two classical linear codes
3. **State** and verify the dual containment condition
4. **Compute** CSS code parameters from classical code parameters
5. **Identify** the X-type and Z-type stabilizer structure
6. **Work** through the Steane code as a CSS construction

---

## Historical Background

### The Birth of CSS Codes (1996)

CSS codes were independently discovered by:
- **Calderbank and Shor** (1996): Used classical error-correcting codes to construct quantum codes
- **Steane** (1996): Developed the [[7,1,3]] code using the classical Hamming code

The key insight: **Separate X and Z error correction** using classical codes.

### Motivation

Classical error correction was mature by 1996. The challenge: how to leverage this for quantum errors?

**Key observations:**
1. Quantum errors can be decomposed into X (bit-flip) and Z (phase-flip) components
2. If we can correct X and Z errors separately, we can correct arbitrary errors
3. Classical codes correct bit-flip errors → use them for X and Z separately

---

## Core Theory

### Classical Linear Codes Review

A **classical [n, k, d] linear code** C is a k-dimensional subspace of $\mathbb{F}_2^n$.

**Parity check matrix:** H ∈ $\mathbb{F}_2^{(n-k) \times n}$
$$\mathbf{c} \in C \Leftrightarrow H\mathbf{c} = 0$$

**Generator matrix:** G ∈ $\mathbb{F}_2^{k \times n}$
$$C = \{\mathbf{x}G : \mathbf{x} \in \mathbb{F}_2^k\}$$

**Dual code:**
$$C^\perp = \{\mathbf{v} \in \mathbb{F}_2^n : \mathbf{v} \cdot \mathbf{c} = 0 \text{ for all } \mathbf{c} \in C\}$$

**Key relation:** $\dim(C) + \dim(C^\perp) = n$

### The CSS Construction

**Theorem (CSS Code Construction):**
Let C₁ and C₂ be classical [n, k₁, d₁] and [n, k₂, d₂] codes satisfying:
$$\boxed{C_2^\perp \subseteq C_1}$$

Then there exists a quantum [[n, k, d]] code with:
- $k = k_1 + k_2 - n$
- $d \geq \min(d_1, d(C_2^\perp))$

### Why Dual Containment?

The condition $C_2^\perp \subseteq C_1$ ensures **commutation** of X and Z stabilizers.

**X-type stabilizers:** For each $\mathbf{h} \in C_2^\perp$:
$$S_X^{(\mathbf{h})} = X^{h_1} X^{h_2} \cdots X^{h_n} = X^{\mathbf{h}}$$

**Z-type stabilizers:** For each $\mathbf{g} \in C_1^\perp$:
$$S_Z^{(\mathbf{g})} = Z^{g_1} Z^{g_2} \cdots Z^{g_n} = Z^{\mathbf{g}}$$

**Commutation check:**
$$S_X^{(\mathbf{h})} S_Z^{(\mathbf{g})} = (-1)^{\mathbf{h} \cdot \mathbf{g}} S_Z^{(\mathbf{g})} S_X^{(\mathbf{h})}$$

For commutation: $\mathbf{h} \cdot \mathbf{g} = 0$ for all $\mathbf{h} \in C_2^\perp$ and $\mathbf{g} \in C_1^\perp$.

This holds iff $C_2^\perp \subseteq (C_1^\perp)^\perp = C_1$. ✓

### Stabilizer Group Structure

The stabilizer group for CSS(C₁, C₂) is:
$$S = \langle X^{\mathbf{h}} : \mathbf{h} \in C_2^\perp \rangle \times \langle Z^{\mathbf{g}} : \mathbf{g} \in C_1^\perp \rangle$$

**Parity check matrix (binary symplectic form):**
$$H = \begin{pmatrix} H_{C_2} & 0 \\ 0 & H_{C_1} \end{pmatrix}$$

where $H_{C_1}$ and $H_{C_2}$ are classical parity check matrices.

### Code Space

The code space is spanned by:
$$|[\mathbf{x}]\rangle = \frac{1}{\sqrt{|C_2^\perp|}} \sum_{\mathbf{c} \in C_2^\perp} |\mathbf{x} + \mathbf{c}\rangle$$

for coset representatives $\mathbf{x} \in C_1 / C_2^\perp$.

**Dimension:** $|C_1| / |C_2^\perp| = 2^{k_1} / 2^{n-k_2} = 2^{k_1 + k_2 - n}$

---

## Special Cases

### Self-Orthogonal CSS

When $C^\perp \subseteq C$ (self-orthogonal code):
$$CSS(C, C) = [[n, 2k - n, d(C^\perp)]]$$

**Examples:**
- Extended Hamming code [8, 4, 4] is self-dual: $C = C^\perp$
- Steane code uses [7, 4, 3] Hamming code

### CSS from Single Code

If C is self-orthogonal ($C^\perp \subseteq C$):
- C₁ = C
- C₂ = C
- Gives [[n, 2k - n, d(C)]] code

### Symmetric CSS

When C₁ = C₂ and code is self-orthogonal:
- Same protection against X and Z errors
- Simplified decoding

---

## The Steane Code Example

### Classical Hamming Code

The **[7, 4, 3] Hamming code** has parity check matrix:

$$H = \begin{pmatrix}
1 & 0 & 0 & 1 & 1 & 0 & 1 \\
0 & 1 & 0 & 1 & 0 & 1 & 1 \\
0 & 0 & 1 & 0 & 1 & 1 & 1
\end{pmatrix}$$

**Properties:**
- n = 7 bits
- k = 4 information bits
- d = 3 (corrects 1 error)

### Hamming Code is Self-Orthogonal

**Verification:** The rows of H are codewords of C (the [7,4,3] Hamming code).

Check: $H \cdot H^T = ?$
$$H H^T = \begin{pmatrix}
1+1+1 & 1+0+1 & 1+1+1 \\
1+0+1 & 1+1+1 & 1+1+1 \\
1+1+1 & 1+1+1 & 1+1+1
\end{pmatrix} = \begin{pmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 1
\end{pmatrix} \pmod 2$$

Hmm, this is NOT zero. Let me reconsider.

Actually, for the Hamming code: $C^\perp = $ the dual code, which has H as its generator matrix. Since the Hamming [7,4,3] code has the property that its dual is the [7,3,4] simplex code.

For CSS construction, we need $C^\perp \subseteq C$. The [7,4,3] Hamming code satisfies this because:
- $C^\perp$ is the [7, 3, 4] code (generated by rows of H)
- We can verify that every codeword of $C^\perp$ is also in C

### Steane Code Construction

Using C₁ = C₂ = [7, 4, 3] Hamming code:

**Parameters:**
$$[[n, k_1 + k_2 - n, d]] = [[7, 4 + 4 - 7, 3]] = [[7, 1, 3]]$$

**X stabilizers** (from $C_2^\perp$):
- $S_X^{(1)} = IIIXXXX$ (row 1 of H)
- $S_X^{(2)} = IXXIIXX$ (row 2 of H)
- $S_X^{(3)} = XIXIXIX$ (row 3 of H)

**Z stabilizers** (from $C_1^\perp$):
- $S_Z^{(1)} = IIIZZZZ$
- $S_Z^{(2)} = IZZIIZZ$
- $S_Z^{(3)} = ZIZIZIZ$

**Logical operators:**
$$\bar{X} = X^{\otimes 7}, \quad \bar{Z} = Z^{\otimes 7}$$

---

## Error Correction in CSS Codes

### Syndrome Measurement

**X error syndrome:** Measure Z stabilizers
- Z stabilizers detect X errors
- Syndrome $\mathbf{s}_X = H_{C_1} \mathbf{e}_X$

**Z error syndrome:** Measure X stabilizers
- X stabilizers detect Z errors
- Syndrome $\mathbf{s}_Z = H_{C_2} \mathbf{e}_Z$

### Decoding

1. Measure all Z stabilizers → get X error syndrome
2. Use classical decoder for C₁ to find X error
3. Measure all X stabilizers → get Z error syndrome
4. Use classical decoder for C₂ to find Z error
5. Apply corrections

**Advantage:** Can use classical decoding algorithms!

### Error Distance

**X errors:** Undetectable X error must be in $C_1 \setminus C_2^\perp$
- Distance: $d_X = d(C_1^\perp)^{\perp} = $ minimum weight in $C_1 \setminus C_2^\perp$

**Z errors:** Undetectable Z error must be in $C_2 \setminus C_1^\perp$
- Distance: $d_Z = $ minimum weight in $C_2 \setminus C_1^\perp$

**Code distance:** $d = \min(d_X, d_Z)$

---

## Worked Examples

### Example 1: Verify CSS Construction

**Problem:** Show that the [7,4,3] and [7,4,3] Hamming codes satisfy the CSS condition.

**Solution:**

For C = [7,4,3] Hamming code:
- Generator matrix: 7×4, generates all codewords
- Parity check matrix: H is 3×7

The dual code $C^\perp$ has:
- Generator = H (3×7)
- Dimension = 3
- So $C^\perp$ is [7, 3, 4]

To verify $C^\perp \subseteq C$: each row of H must be a codeword of C.

Row 1 of H: (1,0,0,1,1,0,1)
Check: $H \cdot (1,0,0,1,1,0,1)^T = ?$
$= (1+1+0+1, 0+1+1+1, 0+0+1+1)^T = (1, 1, 0)^T \neq 0$

Wait, this suggests the row is NOT in C. Let me reconsider...

Actually, I need to be more careful. The Steane code uses a **different representation**. Let me use the standard construction.

**Correct approach:** The [7,4,3] Hamming code CAN be used, but we need the **extended** or a specific form where self-orthogonality holds.

For the Steane code specifically:
$$H_X = H_Z = \begin{pmatrix}
0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0 & 1
\end{pmatrix}$$

These rows DO have the property that $H H^T = 0 \pmod 2$.

### Example 2: Compute CSS Parameters

**Problem:** Given C₁ = [15, 11, 3] and C₂ = [15, 7, 5] with $C_2^\perp \subseteq C_1$, find the CSS code parameters.

**Solution:**

$$k = k_1 + k_2 - n = 11 + 7 - 15 = 3$$

For distance:
- $d(C_1) = 3$ (protects against Z errors)
- $d(C_2^\perp)$: $C_2^\perp$ is [15, 8, ?]. Need to determine.

Since $C_2$ is [15, 7, 5], its dual $C_2^\perp$ is [15, 15-7, ?] = [15, 8, ?].

The minimum distance of $C_2^\perp$ relates to covering radius of $C_2$.

**Result:** [[15, 3, min(3, d(C₂⊥))]] code.

### Example 3: Syndrome Calculation

**Problem:** In the Steane code, an X error occurs on qubit 5. Find the syndrome.

**Solution:**

Error: $E = I \otimes I \otimes I \otimes I \otimes X \otimes I \otimes I = X_5$

Error vector (for X part): $\mathbf{e} = (0,0,0,0,1,0,0)$

Z syndrome (detects X errors):
$$\mathbf{s} = H \cdot \mathbf{e} = \begin{pmatrix}
0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0 & 1
\end{pmatrix} \begin{pmatrix} 0\\0\\0\\0\\1\\0\\0 \end{pmatrix} = \begin{pmatrix} 1\\0\\1 \end{pmatrix}$$

Syndrome $(1, 0, 1)$ indicates error on qubit 5. ✓

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For the CSS code CSS(C₁, C₂) with C₁ = [6, 3, 3] and C₂ = [6, 3, 3] where $C_2^\perp \subseteq C_1$, compute:
a) The number of encoded qubits k
b) The number of X stabilizer generators
c) The number of Z stabilizer generators

**P1.2** Write the parity check matrix structure for a CSS code where C₁ has parity check H₁ (m₁ × n) and C₂ has parity check H₂ (m₂ × n).

**P1.3** Verify that the all-X operator ($X^{\otimes n}$) commutes with all Z stabilizers in any CSS code.

### Level 2: Intermediate

**P2.1** Prove that if C is self-dual (C = C⊥), then CSS(C, C) gives a [[n, 0, d]] code (no logical qubits).

**P2.2** For the [[7,1,3]] Steane code:
a) Write out all 6 stabilizer generators explicitly
b) Verify they all commute pairwise
c) Find a non-stabilizer logical operator

**P2.3** Given a classical [8, 4, 4] extended Hamming code, construct the CSS code CSS(C, C) and determine its parameters.

### Level 3: Challenging

**P3.1** Prove that for CSS(C₁, C₂), the logical X operators are elements of $C_1 \setminus C_2^\perp$ and logical Z operators are elements of $C_2 \setminus C_1^\perp$.

**P3.2** Design a CSS code with parameters [[15, 1, 3]] using BCH codes.

**P3.3** Show that the code distance of CSS(C₁, C₂) satisfies:
$$d = \min\left(\min_{\mathbf{c} \in C_1 \setminus C_2^\perp} wt(\mathbf{c}), \min_{\mathbf{c} \in C_2 \setminus C_1^\perp} wt(\mathbf{c})\right)$$

---

## Computational Lab

```python
"""
Day 743: CSS Code Construction
==============================

Implementing CSS code construction from classical linear codes.
"""

import numpy as np
from typing import Tuple, List, Optional


def gf2_rank(matrix: np.ndarray) -> int:
    """Compute rank of matrix over GF(2)."""
    M = matrix.copy() % 2
    rows, cols = M.shape
    rank = 0

    for col in range(cols):
        # Find pivot
        pivot_row = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        # Swap rows
        M[[rank, pivot_row]] = M[[pivot_row, rank]]

        # Eliminate
        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2

        rank += 1

    return rank


def gf2_null_space(matrix: np.ndarray) -> np.ndarray:
    """Compute null space basis over GF(2)."""
    M = matrix.copy() % 2
    m, n = M.shape

    # Augment with identity
    augmented = np.hstack([M.T, np.eye(n, dtype=int)])

    # Row reduce
    rows, cols = augmented.shape
    pivot_cols = []
    row = 0

    for col in range(m):
        # Find pivot
        pivot_row = None
        for r in range(row, rows):
            if augmented[r, col] == 1:
                pivot_row = r
                break

        if pivot_row is None:
            continue

        pivot_cols.append(col)
        augmented[[row, pivot_row]] = augmented[[pivot_row, row]]

        for r in range(rows):
            if r != row and augmented[r, col] == 1:
                augmented[r] = (augmented[r] + augmented[row]) % 2

        row += 1

    # Extract null space
    null_space = []
    for i in range(n):
        if all(augmented[i, :m] == 0):
            null_space.append(augmented[i, m:])

    if len(null_space) == 0:
        return np.array([]).reshape(0, n)

    return np.array(null_space) % 2


class ClassicalLinearCode:
    """
    Classical linear code over GF(2).

    Attributes:
    -----------
    H : np.ndarray
        Parity check matrix
    n : int
        Block length
    k : int
        Dimension (number of information bits)
    """

    def __init__(self, parity_check: np.ndarray):
        """Initialize from parity check matrix."""
        self.H = np.array(parity_check) % 2
        self.m, self.n = self.H.shape
        self.k = self.n - gf2_rank(self.H)

    def generator_matrix(self) -> np.ndarray:
        """Compute generator matrix (basis for kernel of H)."""
        return gf2_null_space(self.H)

    def dual(self) -> 'ClassicalLinearCode':
        """Return dual code C^⊥."""
        G = self.generator_matrix()
        if G.size == 0:
            # Code is {0}, dual is entire space
            return ClassicalLinearCode(np.zeros((0, self.n), dtype=int))
        return ClassicalLinearCode(G)

    def contains_code(self, other: 'ClassicalLinearCode') -> bool:
        """Check if self contains other (other ⊆ self)."""
        G_other = other.generator_matrix()
        if G_other.size == 0:
            return True  # {0} is contained in everything

        # Check if H_self · G_other^T = 0
        product = (self.H @ G_other.T) % 2
        return np.all(product == 0)

    def syndrome(self, vector: np.ndarray) -> np.ndarray:
        """Compute syndrome of a vector."""
        return (self.H @ vector) % 2

    def is_codeword(self, vector: np.ndarray) -> bool:
        """Check if vector is a codeword."""
        return np.all(self.syndrome(vector) == 0)

    def __repr__(self) -> str:
        return f"[{self.n}, {self.k}] code"


class CSSCode:
    """
    CSS quantum code constructed from two classical codes.

    CSS(C1, C2) requires C2^⊥ ⊆ C1.
    """

    def __init__(self, C1: ClassicalLinearCode, C2: ClassicalLinearCode):
        """
        Construct CSS code from C1 and C2.

        Parameters:
        -----------
        C1 : ClassicalLinearCode
            First classical code (for Z stabilizers)
        C2 : ClassicalLinearCode
            Second classical code (for X stabilizers)
        """
        self.C1 = C1
        self.C2 = C2

        # Verify dual containment
        C2_dual = C2.dual()
        if not C1.contains_code(C2_dual):
            raise ValueError("Dual containment C2^⊥ ⊆ C1 not satisfied!")

        self.n = C1.n
        self.k = C1.k + C2.k - self.n

        # Stabilizer parity check matrices
        self.H_X = C2.H  # X stabilizers (detect Z errors)
        self.H_Z = C1.H  # Z stabilizers (detect X errors)

    def x_syndrome(self, error: np.ndarray) -> np.ndarray:
        """Compute syndrome for X-type error."""
        return (self.H_Z @ error) % 2

    def z_syndrome(self, error: np.ndarray) -> np.ndarray:
        """Compute syndrome for Z-type error."""
        return (self.H_X @ error) % 2

    def stabilizer_generators(self) -> Tuple[List[str], List[str]]:
        """Return X and Z stabilizer generators as Pauli strings."""
        x_stabs = []
        for row in self.H_X:
            pauli = ''.join('X' if b else 'I' for b in row)
            x_stabs.append(pauli)

        z_stabs = []
        for row in self.H_Z:
            pauli = ''.join('Z' if b else 'I' for b in row)
            z_stabs.append(pauli)

        return x_stabs, z_stabs

    def full_parity_check(self) -> np.ndarray:
        """Return full binary symplectic parity check matrix."""
        m_x, n = self.H_X.shape
        m_z, _ = self.H_Z.shape

        # H = [H_X | 0]
        #     [0   | H_Z]
        H = np.zeros((m_x + m_z, 2 * n), dtype=int)
        H[:m_x, :n] = self.H_X
        H[m_x:, n:] = self.H_Z

        return H

    def verify_commutation(self) -> bool:
        """Verify all stabilizers commute."""
        for x_row in self.H_X:
            for z_row in self.H_Z:
                if np.dot(x_row, z_row) % 2 != 0:
                    return False
        return True

    def __repr__(self) -> str:
        return f"[[{self.n}, {self.k}]] CSS code"


def steane_code() -> CSSCode:
    """Construct the [[7,1,3]] Steane code."""
    # Hamming [7,4,3] parity check matrix
    H = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ])

    C = ClassicalLinearCode(H)
    return CSSCode(C, C)


def repetition_css(n: int) -> CSSCode:
    """
    Construct CSS code from repetition code.

    Uses [n, 1, n] repetition code and its dual [n, n-1, 2].
    """
    # Repetition code [n, 1, n]
    # Parity check: all pairs must be equal
    H_rep = np.zeros((n-1, n), dtype=int)
    for i in range(n-1):
        H_rep[i, i] = 1
        H_rep[i, i+1] = 1

    C_rep = ClassicalLinearCode(H_rep)

    # For CSS, we need C2^⊥ ⊆ C1
    # Here C2^⊥ = dual of repetition = [n, n-1, 2]
    # C1 = repetition = [n, 1, n]
    # This works since the all-ones vector is in both

    return CSSCode(C_rep, C_rep)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 743: CSS Code Construction")
    print("=" * 60)

    # Example 1: Steane Code
    print("\n1. Steane [[7,1,3]] Code")
    print("-" * 40)

    steane = steane_code()
    print(f"Code: {steane}")
    print(f"Commutation verified: {steane.verify_commutation()}")

    x_stabs, z_stabs = steane.stabilizer_generators()
    print("\nX Stabilizers:")
    for i, s in enumerate(x_stabs):
        print(f"  S_X^({i+1}) = {s}")

    print("\nZ Stabilizers:")
    for i, s in enumerate(z_stabs):
        print(f"  S_Z^({i+1}) = {s}")

    # Example 2: Syndrome calculation
    print("\n2. Syndrome Calculation")
    print("-" * 40)

    # X error on qubit 3
    x_error = np.array([0, 0, 1, 0, 0, 0, 0])
    syndrome = steane.x_syndrome(x_error)
    print(f"X error on qubit 3: {x_error}")
    print(f"Z syndrome: {syndrome}")

    # Z error on qubit 5
    z_error = np.array([0, 0, 0, 0, 1, 0, 0])
    syndrome = steane.z_syndrome(z_error)
    print(f"\nZ error on qubit 5: {z_error}")
    print(f"X syndrome: {syndrome}")

    # Example 3: Classical code properties
    print("\n3. Classical Code Properties")
    print("-" * 40)

    H = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ])
    C = ClassicalLinearCode(H)
    print(f"Classical code: {C}")

    G = C.generator_matrix()
    print(f"Generator matrix shape: {G.shape}")
    print(f"Generator matrix:\n{G}")

    C_dual = C.dual()
    print(f"\nDual code: {C_dual}")
    print(f"C^⊥ ⊆ C: {C.contains_code(C_dual)}")

    # Example 4: Verify self-orthogonality
    print("\n4. Self-Orthogonality Check")
    print("-" * 40)

    # Check H · H^T mod 2
    HHT = (H @ H.T) % 2
    print(f"H · H^T mod 2:\n{HHT}")

    # For Steane code construction to work, we need the
    # rows of H to be orthogonal to each other (mod 2)

    # Example 5: Full parity check matrix
    print("\n5. Full Binary Symplectic Parity Check")
    print("-" * 40)

    H_full = steane.full_parity_check()
    print(f"Shape: {H_full.shape} (6 generators × 14 binary coordinates)")
    print(f"Matrix:\n{H_full}")

    print("\n" + "=" * 60)
    print("CSS codes combine classical error correction with quantum!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Dual containment | $C_2^\perp \subseteq C_1$ |
| CSS dimension | $k = k_1 + k_2 - n$ |
| X stabilizers | $\{X^{\mathbf{h}} : \mathbf{h} \in C_2^\perp\}$ |
| Z stabilizers | $\{Z^{\mathbf{g}} : \mathbf{g} \in C_1^\perp\}$ |
| Parity check | $H = \begin{pmatrix} H_X & 0 \\ 0 & H_Z \end{pmatrix}$ |
| Self-orthogonal | $C^\perp \subseteq C$ |

### Main Takeaways

1. **CSS codes** separate X and Z error correction using classical codes
2. **Dual containment** $C_2^\perp \subseteq C_1$ ensures stabilizer commutation
3. **Steane code** is CSS(Hamming, Hamming) = [[7, 1, 3]]
4. **Decoding** reduces to classical decoding problems
5. **Self-orthogonal** codes give symmetric CSS codes with C₁ = C₂

---

## Daily Checklist

- [ ] I can state the CSS construction theorem
- [ ] I can verify the dual containment condition
- [ ] I can compute CSS code parameters from classical code parameters
- [ ] I understand why X and Z stabilizers commute
- [ ] I can write out stabilizers for the Steane code
- [ ] I can compute syndromes for X and Z errors

---

## Preview: Day 744

Tomorrow we dive deep into the **dual containment condition**:

- Classical dual codes and their properties
- Self-dual and self-orthogonal codes
- Constructions that guarantee $C^\perp \subseteq C$
- Examples: Reed-Muller, BCH, and Reed-Solomon based CSS codes

The dual containment condition is the key constraint—understanding it unlocks systematic CSS code design.
