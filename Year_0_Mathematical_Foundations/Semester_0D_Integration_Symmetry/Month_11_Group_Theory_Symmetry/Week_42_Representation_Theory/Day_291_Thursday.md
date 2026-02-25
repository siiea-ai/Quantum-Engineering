# Day 291: Characters — The Fingerprints of Representations

## Overview

**Month 11, Week 42, Day 4 — Thursday**

Today we study **characters**, the single most useful tool for working with representations. A character encodes essential information about a representation in a single function, and remarkably, this function completely determines the representation up to equivalence. Character theory provides elegant formulas for decomposing representations, counting multiplicities, and understanding group structure.

## Prerequisites

From Days 288-290:
- Group representations and matrix representations
- Irreducible representations
- Schur's lemma

## Learning Objectives

By the end of today, you will be able to:

1. Define characters and compute them for given representations
2. Prove that characters are class functions
3. Build and interpret character tables
4. Use characters to determine if representations are equivalent
5. Compute characters of direct sums and tensor products
6. Apply character theory to quantum mechanics

---

## 1. Definition of Characters

### The Character Function

**Definition:** The **character** of a representation $\rho: G \to GL(V)$ is the function $\chi: G \to \mathbb{C}$ defined by:

$$\boxed{\chi(g) = \text{Tr}(\rho(g)) = \sum_{i=1}^{n} D(g)_{ii}}$$

### Basic Properties

**Property 1:** $\chi(e) = \dim(V) = n$

*Proof:* $\chi(e) = \text{Tr}(I_n) = n$. ∎

**Property 2:** Characters are **class functions** (constant on conjugacy classes):
$$\chi(hgh^{-1}) = \chi(g) \quad \forall g, h \in G$$

*Proof:*
$$\chi(hgh^{-1}) = \text{Tr}(\rho(hgh^{-1})) = \text{Tr}(\rho(h)\rho(g)\rho(h)^{-1}) = \text{Tr}(\rho(g)) = \chi(g)$$
using the cyclic property of trace. ∎

**Property 3:** $\chi(g^{-1}) = \overline{\chi(g)}$ for unitary representations.

*Proof:* For unitary $D$: $D(g^{-1}) = D(g)^\dagger$, so $\chi(g^{-1}) = \text{Tr}(D(g)^\dagger) = \overline{\text{Tr}(D(g))}$. ∎

### Characters Determine Representations

**Theorem:** Two representations of a finite group over $\mathbb{C}$ are equivalent if and only if they have the same character.

This is powerful: instead of comparing infinite families of matrices, we just compare a single function!

---

## 2. Characters of Standard Operations

### Direct Sum

For $\rho = \rho_1 \oplus \rho_2$:
$$\boxed{\chi_{\rho_1 \oplus \rho_2}(g) = \chi_1(g) + \chi_2(g)}$$

*Proof:* $D(g) = \begin{pmatrix} D_1(g) & 0 \\ 0 & D_2(g) \end{pmatrix}$, so $\text{Tr}(D(g)) = \text{Tr}(D_1(g)) + \text{Tr}(D_2(g))$. ∎

### Tensor Product

For $\rho = \rho_1 \otimes \rho_2$:
$$\boxed{\chi_{\rho_1 \otimes \rho_2}(g) = \chi_1(g) \cdot \chi_2(g)}$$

*Proof:* $\text{Tr}(A \otimes B) = \text{Tr}(A) \cdot \text{Tr}(B)$. ∎

### Dual Representation

For the dual $\rho^*$:
$$\chi_{\rho^*}(g) = \overline{\chi(g)} = \chi(g^{-1})$$

### Conjugate Representation

For $\bar{\rho}$:
$$\chi_{\bar{\rho}}(g) = \overline{\chi(g)}$$

---

## 3. Character Tables

### Structure of Character Tables

A **character table** for a finite group $G$ has:
- Rows: one for each irreducible representation
- Columns: one for each conjugacy class
- Entry $(i, j)$: value of $\chi_i$ on class $C_j$

### Properties of Character Tables

1. Number of rows = number of columns = number of conjugacy classes
2. First row (trivial rep): all 1s
3. First column (identity): dimensions of irreps
4. The table is a square matrix (over $\mathbb{C}$)

### Example: Character Table of $S_3$

Conjugacy classes: $\{e\}$, $\{(12), (13), (23)\}$, $\{(123), (132)\}$

| Rep | $e$ | $(12)$ | $(123)$ |
|-----|-----|--------|---------|
| Trivial | 1 | 1 | 1 |
| Sign | 1 | -1 | 1 |
| Standard | 2 | 0 | -1 |

### Example: Character Table of $\mathbb{Z}_4$

Four 1-dim irreps: $\rho_k(r) = i^k$ for $k = 0, 1, 2, 3$.

| Rep | $e$ | $r$ | $r^2$ | $r^3$ |
|-----|-----|-----|-------|-------|
| $\chi_0$ | 1 | 1 | 1 | 1 |
| $\chi_1$ | 1 | $i$ | -1 | $-i$ |
| $\chi_2$ | 1 | -1 | 1 | -1 |
| $\chi_3$ | 1 | $-i$ | -1 | $i$ |

---

## 4. The Orthogonality Relations

### Row Orthogonality

**Theorem (First Orthogonality):** For irreducible characters $\chi_\alpha, \chi_\beta$:

$$\boxed{\frac{1}{|G|} \sum_{g \in G} \overline{\chi_\alpha(g)} \chi_\beta(g) = \delta_{\alpha\beta}}$$

### Column Orthogonality

**Theorem (Second Orthogonality):** For conjugacy classes $C_i, C_j$:

$$\boxed{\sum_{\alpha} \overline{\chi_\alpha(g_i)} \chi_\alpha(g_j) = \frac{|G|}{|C_i|} \delta_{ij}}$$

where $g_i \in C_i$, $g_j \in C_j$.

### The Inner Product

Define the **inner product** on class functions:
$$\langle \chi, \psi \rangle = \frac{1}{|G|} \sum_{g \in G} \overline{\chi(g)} \psi(g)$$

Then irreducible characters form an **orthonormal basis** for class functions!

---

## 5. Decomposing Representations

### The Multiplicity Formula

For a representation with character $\chi$, the multiplicity of irrep $\rho_\alpha$ is:

$$\boxed{n_\alpha = \langle \chi_\alpha, \chi \rangle = \frac{1}{|G|} \sum_{g \in G} \overline{\chi_\alpha(g)} \chi(g)}$$

### Example: Decompose Permutation Rep of $S_3$

The permutation rep on $\mathbb{C}^3$ has character:
- $\chi(e) = 3$ (identity fixes all)
- $\chi((12)) = 1$ (transposition fixes one)
- $\chi((123)) = 0$ (3-cycle fixes none)

Using the character table:

$$n_{\text{triv}} = \frac{1}{6}(1 \cdot 3 + 3 \cdot 1 \cdot 1 + 2 \cdot 1 \cdot 0) = \frac{6}{6} = 1$$

$$n_{\text{sign}} = \frac{1}{6}(1 \cdot 3 + 3 \cdot (-1) \cdot 1 + 2 \cdot 1 \cdot 0) = \frac{0}{6} = 0$$

$$n_{\text{std}} = \frac{1}{6}(1 \cdot 3 \cdot 2 + 3 \cdot 0 \cdot 1 + 2 \cdot (-1) \cdot 0) = \frac{6}{6} = 1$$

So: $\text{perm} \cong \text{triv} \oplus \text{standard}$ ✓

### Criterion for Irreducibility

$$\boxed{\chi \text{ is irreducible} \Leftrightarrow \langle \chi, \chi \rangle = 1}$$

---

## 6. Quantum Mechanics Connection

### Characters in Quantum Mechanics

In QM, characters appear naturally:

1. **Partition function:** $Z = \sum_n e^{-\beta E_n} = \text{Tr}(e^{-\beta H})$ is a "thermal character"

2. **Degeneracy:** $\chi(e) = $ dimension = degree of degeneracy

3. **Selection rules:** Matrix elements $\langle \alpha | O | \beta \rangle$ vanish unless the tensor product $\chi_O \times \chi_\beta$ contains $\chi_\alpha$

### Projection Operators

The projection onto the $\alpha$-th irreducible component:

$$P_\alpha = \frac{d_\alpha}{|G|} \sum_{g \in G} \overline{\chi_\alpha(g)} \rho(g)$$

This extracts the part of the representation transforming as $\rho_\alpha$.

### Character and Trace in QM

For a system with symmetry group $G$:
- States form representations
- Character $\chi(g)$ = trace of symmetry operator $U(g)$
- Decomposition formula tells us which irreps appear and with what multiplicity

---

## 7. Worked Examples

### Example 1: Build Character Table for $D_3$

$D_3$ has 6 elements: $\{e, r, r^2, s, sr, sr^2\}$

Conjugacy classes:
- $\{e\}$ (1 element)
- $\{r, r^2\}$ (2 elements, rotations)
- $\{s, sr, sr^2\}$ (3 elements, reflections)

Three irreps (since 3 classes). Dimensions: $1^2 + 1^2 + 2^2 = 6$ ✓

| Rep | $\{e\}$ | $\{r, r^2\}$ | $\{s, sr, sr^2\}$ |
|-----|---------|--------------|-------------------|
| $\chi_1$ (trivial) | 1 | 1 | 1 |
| $\chi_2$ (sign) | 1 | 1 | -1 |
| $\chi_3$ (standard) | 2 | -1 | 0 |

**Verification:** Row orthogonality for $\chi_1$ and $\chi_3$:
$$\frac{1}{6}(1 \cdot 1 \cdot 2 + 2 \cdot 1 \cdot (-1) + 3 \cdot 1 \cdot 0) = \frac{2 - 2 + 0}{6} = 0$$ ✓

### Example 2: Decompose Regular Rep of $\mathbb{Z}_3$

Regular rep has $\chi_{\text{reg}}(e) = 3$, $\chi_{\text{reg}}(r) = 0$, $\chi_{\text{reg}}(r^2) = 0$.

For each 1-dim irrep $\chi_k$:
$$n_k = \frac{1}{3}(1 \cdot 3 + 1 \cdot 0 \cdot e^{2\pi ik/3} + 1 \cdot 0 \cdot e^{4\pi ik/3}) = \frac{3}{3} = 1$$

So: $\text{reg} = \chi_0 \oplus \chi_1 \oplus \chi_2$

Each irrep appears exactly once in the regular representation!

### Example 3: Character of Tensor Product

For $S_3$: $\chi_{\text{std}} \otimes \chi_{\text{std}}$

$$\chi_{\text{std} \otimes \text{std}}(g) = \chi_{\text{std}}(g)^2$$

| $g$ | $\chi_{\text{std}}$ | $\chi_{\text{std}}^2$ |
|-----|---------------------|----------------------|
| $e$ | 2 | 4 |
| $(12)$ | 0 | 0 |
| $(123)$ | -1 | 1 |

Decompose:
$$n_{\text{triv}} = \frac{1}{6}(4 + 0 + 2) = 1$$
$$n_{\text{sign}} = \frac{1}{6}(4 + 0 + 2) = 1$$
$$n_{\text{std}} = \frac{1}{6}(8 + 0 - 2) = 1$$

So: $\text{std} \otimes \text{std} \cong \text{triv} \oplus \text{sign} \oplus \text{std}$

---

## 8. Computational Lab

```python
"""
Day 291: Characters
Computing character tables and decomposing representations
"""

import numpy as np
from typing import List, Dict, Tuple

class CharacterTable:
    """
    Character table for a finite group.
    """

    def __init__(self, irrep_names: List[str], class_names: List[str],
                 class_sizes: List[int], characters: np.ndarray):
        """
        Initialize character table.

        Parameters:
            irrep_names: Names of irreducible representations
            class_names: Names of conjugacy classes
            class_sizes: Size of each conjugacy class
            characters: 2D array, characters[i,j] = χ_i(C_j)
        """
        self.irrep_names = irrep_names
        self.class_names = class_names
        self.class_sizes = np.array(class_sizes)
        self.characters = np.array(characters, dtype=complex)
        self.num_irreps = len(irrep_names)
        self.group_order = sum(class_sizes)

    def display(self):
        """Print the character table."""
        # Header
        header = f"{'':>10} | " + " | ".join(f"{c:>8}" for c in self.class_names)
        print(header)
        print("-" * len(header))

        # Rows
        for i, name in enumerate(self.irrep_names):
            row = f"{name:>10} | " + " | ".join(
                f"{self.characters[i,j].real:>8.2f}" if self.characters[i,j].imag == 0
                else f"{self.characters[i,j]:>8}" for j in range(len(self.class_names))
            )
            print(row)

        # Class sizes
        print("-" * len(header))
        sizes = f"{'|C|':>10} | " + " | ".join(f"{s:>8}" for s in self.class_sizes)
        print(sizes)

    def inner_product(self, chi1: np.ndarray, chi2: np.ndarray) -> complex:
        """
        Compute <χ1, χ2> = (1/|G|) Σ |C| χ1(C)* χ2(C)
        """
        return np.sum(self.class_sizes * np.conj(chi1) * chi2) / self.group_order

    def verify_orthogonality(self) -> bool:
        """Verify row orthogonality of character table."""
        for i in range(self.num_irreps):
            for j in range(self.num_irreps):
                ip = self.inner_product(self.characters[i], self.characters[j])
                expected = 1.0 if i == j else 0.0
                if not np.isclose(ip, expected):
                    return False
        return True

    def decompose(self, chi: np.ndarray) -> Dict[str, int]:
        """
        Decompose a character into irreducible components.

        Parameters:
            chi: Character values on conjugacy classes

        Returns:
            Dict mapping irrep names to multiplicities
        """
        multiplicities = {}
        for i, name in enumerate(self.irrep_names):
            n = self.inner_product(self.characters[i], chi)
            multiplicities[name] = int(round(n.real))
        return multiplicities

    def is_irreducible(self, chi: np.ndarray) -> bool:
        """Check if character is irreducible."""
        return np.isclose(self.inner_product(chi, chi), 1.0)

    def tensor_product_character(self, i: int, j: int) -> np.ndarray:
        """Compute character of tensor product of irreps i and j."""
        return self.characters[i] * self.characters[j]


def S3_character_table() -> CharacterTable:
    """Character table for S_3."""
    irreps = ['trivial', 'sign', 'standard']
    classes = ['{e}', '{(12)}', '{(123)}']
    sizes = [1, 3, 2]

    chars = np.array([
        [1, 1, 1],      # trivial
        [1, -1, 1],     # sign
        [2, 0, -1]      # standard
    ])

    return CharacterTable(irreps, classes, sizes, chars)


def Z4_character_table() -> CharacterTable:
    """Character table for Z_4."""
    irreps = ['χ_0', 'χ_1', 'χ_2', 'χ_3']
    classes = ['e', 'r', 'r²', 'r³']
    sizes = [1, 1, 1, 1]

    i = 1j
    chars = np.array([
        [1, 1, 1, 1],
        [1, i, -1, -i],
        [1, -1, 1, -1],
        [1, -i, -1, i]
    ])

    return CharacterTable(irreps, classes, sizes, chars)


def D4_character_table() -> CharacterTable:
    """Character table for D_4 (square symmetries)."""
    irreps = ['A1', 'A2', 'B1', 'B2', 'E']
    classes = ['{e}', '{r²}', '{r,r³}', '{s,sr²}', '{sr,sr³}']
    sizes = [1, 1, 2, 2, 2]

    chars = np.array([
        [1, 1, 1, 1, 1],      # A1 (trivial)
        [1, 1, 1, -1, -1],    # A2
        [1, 1, -1, 1, -1],    # B1
        [1, 1, -1, -1, 1],    # B2
        [2, -2, 0, 0, 0]      # E (2-dim)
    ])

    return CharacterTable(irreps, classes, sizes, chars)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("CHARACTER TABLES")
    print("=" * 60)

    # S_3 character table
    print("\n1. CHARACTER TABLE OF S_3")
    print("-" * 40)

    S3 = S3_character_table()
    S3.display()

    print(f"\nRow orthogonality: {S3.verify_orthogonality()}")

    # Decompose permutation representation
    print("\n2. DECOMPOSE PERMUTATION REP OF S_3")
    print("-" * 40)

    chi_perm = np.array([3, 1, 0])  # χ(e)=3, χ((12))=1, χ((123))=0
    print(f"Character of perm rep: {chi_perm}")

    decomp = S3.decompose(chi_perm)
    print(f"Decomposition: {decomp}")

    # Verify
    print(f"\nIs irreducible? {S3.is_irreducible(chi_perm)}")
    print(f"<χ,χ> = {S3.inner_product(chi_perm, chi_perm):.2f}")

    # Tensor product
    print("\n3. TENSOR PRODUCT: standard ⊗ standard")
    print("-" * 40)

    chi_tensor = S3.tensor_product_character(2, 2)  # standard ⊗ standard
    print(f"Character: {chi_tensor}")

    decomp_tensor = S3.decompose(chi_tensor)
    print(f"Decomposition: {decomp_tensor}")

    # Z_4 character table
    print("\n4. CHARACTER TABLE OF Z_4")
    print("-" * 40)

    Z4 = Z4_character_table()
    Z4.display()

    # D_4 character table
    print("\n5. CHARACTER TABLE OF D_4")
    print("-" * 40)

    D4 = D4_character_table()
    D4.display()

    # Decompose reducible rep
    print("\n6. DECOMPOSE A REDUCIBLE REP OF D_4")
    print("-" * 40)

    # 4-dim rep: permutation of vertices of square
    chi_4 = np.array([4, 0, 0, 2, 0])  # Guess based on fixed points
    print(f"Character: {chi_4}")

    decomp_4 = D4.decompose(chi_4)
    print(f"Decomposition: {decomp_4}")

    # Verify: dimensions should match
    total_dim = sum(int(S3.characters[i, 0].real) * v
                   for i, (k, v) in enumerate(decomp.items()))
    print(f"Total dimension check: {total_dim}")

    # Example: regular representation
    print("\n7. REGULAR REPRESENTATION OF S_3")
    print("-" * 40)

    chi_reg = np.array([6, 0, 0])  # χ(e)=|G|, χ(g)=0 for g≠e
    print(f"Character of regular rep: {chi_reg}")

    decomp_reg = S3.decompose(chi_reg)
    print(f"Decomposition: {decomp_reg}")

    # Each irrep appears with multiplicity = dimension
    print("\nVerification: n_α = d_α for regular rep")
    for i, name in enumerate(S3.irrep_names):
        d = int(S3.characters[i, 0].real)
        n = decomp_reg[name]
        print(f"  {name}: dim = {d}, multiplicity = {n}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Character χ(g) = Tr(D(g)) is a class function
    2. Characters determine representations up to equivalence
    3. Irreducible characters are orthonormal
    4. Decomposition: n_α = <χ_α, χ>
    5. χ is irreducible iff <χ, χ> = 1
    6. Regular rep: each irrep appears dim(irrep) times
    """)
```

---

## 9. Practice Problems

### Problem Set A: Character Computations

**A1.** Compute the character table for $\mathbb{Z}_6$.

**A2.** For the standard representation of $S_3$, verify that $\langle \chi_{\text{std}}, \chi_{\text{std}} \rangle = 1$.

**A3.** Compute $\chi_{\text{triv}} \otimes \chi_{\text{std}}$ for $S_3$ and show it equals $\chi_{\text{std}}$.

### Problem Set B: Decomposition

**B1.** Decompose the 4-dim permutation rep of $S_4$ using character theory.

**B2.** For $D_4$, decompose the regular representation and verify each irrep appears with multiplicity equal to its dimension.

**B3.** Find the character of $\text{std}^{\otimes 3}$ for $S_3$ and decompose it.

### Problem Set C: Applications

**C1.** **(Molecular Symmetry)** Water has $C_{2v}$ symmetry. Using the character table, determine which molecular vibrations are IR-active.

**C2.** Prove: The number of 1-dimensional irreps equals $|G/[G,G]|$.

**C3.** **(Physics)** In a system with $D_3$ symmetry, an electric field breaks the symmetry to $C_3$. How do the irreps of $D_3$ decompose into irreps of $C_3$?

---

## 10. Summary

### Key Formulas

$$\boxed{\chi(g) = \text{Tr}(D(g))}$$

$$\boxed{\langle \chi_\alpha, \chi_\beta \rangle = \frac{1}{|G|} \sum_g \overline{\chi_\alpha(g)} \chi_\beta(g) = \delta_{\alpha\beta}}$$

$$\boxed{n_\alpha = \langle \chi_\alpha, \chi \rangle}$$

$$\boxed{\chi_{\rho_1 \oplus \rho_2} = \chi_1 + \chi_2, \quad \chi_{\rho_1 \otimes \rho_2} = \chi_1 \cdot \chi_2}$$

---

## 11. Preview: Day 292

Tomorrow we explore the **Great Orthogonality Theorem** in full detail:
- Orthogonality of matrix elements
- Completeness relations
- Applications to physics

---

*"The character is the soul of the representation—everything essential is encoded in this one function." — J.-P. Serre*
