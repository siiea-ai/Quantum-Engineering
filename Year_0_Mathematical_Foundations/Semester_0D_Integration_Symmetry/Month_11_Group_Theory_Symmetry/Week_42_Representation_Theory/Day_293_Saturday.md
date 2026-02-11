# Day 293: Representations of the Symmetric Group

## Overview

**Month 11, Week 42, Day 6 — Saturday**

Today we study the representations of the symmetric group $S_n$, which are essential for understanding systems of identical particles in quantum mechanics. The irreducible representations of $S_n$ are classified by **partitions** of $n$ and can be constructed using **Young tableaux**. This elegant combinatorial approach connects group theory to the physics of bosons, fermions, and more exotic particle statistics.

## Prerequisites

From Week 42:
- General representation theory
- Characters and character tables
- Schur's lemma and orthogonality

## Learning Objectives

By the end of today, you will be able to:

1. Classify irreps of $S_n$ using partitions and Young diagrams
2. Construct Young tableaux and understand their role
3. Apply the hook length formula for dimensions
4. Connect $S_n$ representations to identical particle physics
5. Understand symmetrizers and antisymmetrizers
6. Use Young tableaux in quantum mechanics applications

---

## 1. Partitions and Young Diagrams

### Partitions

**Definition:** A **partition** of $n$ is a way of writing $n$ as a sum of positive integers:
$$\lambda = (\lambda_1, \lambda_2, \ldots, \lambda_k) \quad \text{where } \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_k > 0$$
and $\sum_i \lambda_i = n$.

We write $\lambda \vdash n$ to mean "$\lambda$ partitions $n$."

**Examples for $n = 4$:**
- $(4)$ — "four"
- $(3, 1)$ — "three and one"
- $(2, 2)$ — "two twos"
- $(2, 1, 1)$ — "two and two ones"
- $(1, 1, 1, 1)$ — "four ones"

### Young Diagrams

A **Young diagram** is a graphical representation of a partition:
- Draw $\lambda_1$ boxes in the first row
- Draw $\lambda_2$ boxes in the second row (left-aligned)
- Continue for all parts

**Example:** The partition $(3, 2, 1)$ gives:
```
□ □ □
□ □
□
```

### Conjugate Partition

The **conjugate** $\lambda'$ is obtained by transposing the diagram (swapping rows and columns).

For $(3, 2, 1)$: $\lambda' = (3, 2, 1)$ (self-conjugate!)

For $(4, 1)$: $\lambda' = (2, 1, 1, 1)$

---

## 2. Correspondence with Irreps

### The Fundamental Theorem

**Theorem:** The irreducible representations of $S_n$ are in one-to-one correspondence with partitions of $n$.

For each partition $\lambda \vdash n$, there is a unique (up to equivalence) irrep $S^\lambda$.

### Special Cases

| Partition | Young Diagram | Irrep Name | Dimension |
|-----------|---------------|------------|-----------|
| $(n)$ | Single row | Trivial | 1 |
| $(1^n)$ | Single column | Sign | 1 |
| $(n-1, 1)$ | Row + corner | Standard | $n-1$ |

### Dimension from Hook Lengths

**Definition:** The **hook** of a box is the set of boxes directly below and to the right, plus the box itself.

The **hook length** $h(i,j)$ is the number of boxes in the hook.

**Theorem (Hook Length Formula):**
$$\boxed{\dim(S^\lambda) = \frac{n!}{\prod_{(i,j) \in \lambda} h(i,j)}}$$

**Example:** For $\lambda = (3, 2)$ ($n = 5$):
```
5 3 1
3 1
```
Hook lengths shown in boxes.

$$\dim = \frac{5!}{5 \cdot 3 \cdot 1 \cdot 3 \cdot 1} = \frac{120}{45} = \frac{120}{45} = \frac{8}{3}$$

Wait, let me recalculate:
$5 \cdot 3 \cdot 1 \cdot 3 \cdot 1 = 45$, but $120/45 = 8/3$ which is not an integer. Let me recompute the hooks.

For $(3, 2)$:
```
Position (1,1): hook includes (1,1), (1,2), (1,3), (2,1) → h = 4
Position (1,2): hook includes (1,2), (1,3), (2,2) → h = 3
Position (1,3): hook includes (1,3) → h = 1
Position (2,1): hook includes (2,1), (2,2) → h = 2
Position (2,2): hook includes (2,2) → h = 1
```

$$\dim = \frac{5!}{4 \cdot 3 \cdot 1 \cdot 2 \cdot 1} = \frac{120}{24} = 5$$

---

## 3. Young Tableaux

### Standard Young Tableaux

**Definition:** A **standard Young tableau** of shape $\lambda$ is a filling of the Young diagram with numbers $1, 2, \ldots, n$ such that:
- Each number appears exactly once
- Rows increase left to right
- Columns increase top to bottom

**Example:** Standard tableaux of shape $(2, 1)$:
```
1 2       1 3
3         2
```

**Theorem:** The number of standard Young tableaux of shape $\lambda$ equals $\dim(S^\lambda)$.

### Semistandard Young Tableaux

Allow repeated numbers, require:
- Rows weakly increase
- Columns strictly increase

These are used in the representation theory of $GL_n$.

---

## 4. Constructing Irreps

### Young Symmetrizers

For a tableau $T$, define:
- **Row group** $R_T$: permutations that permute elements within rows
- **Column group** $C_T$: permutations that permute elements within columns

**Definition:** The **Young symmetrizer** is:
$$c_T = a_T b_T$$
where $a_T = \sum_{\sigma \in R_T} \sigma$ and $b_T = \sum_{\tau \in C_T} \text{sgn}(\tau) \tau$.

**Theorem:** $c_T$ is (proportional to) an idempotent in $\mathbb{C}[S_n]$, and $\mathbb{C}[S_n] c_T$ is isomorphic to $S^\lambda$.

### The Standard Module

The irrep $S^\lambda$ can be constructed as:
$$S^\lambda = \mathbb{C}[S_n] c_T$$

where $T$ is any standard tableau of shape $\lambda$.

---

## 5. Quantum Mechanics Connection

### Identical Particles

For $n$ identical particles:
- **Bosons:** Wave function symmetric under exchange → transforms trivially
- **Fermions:** Wave function antisymmetric → transforms under sign representation

More generally, particles can transform under any irrep of $S_n$!

### Parastatistics (Theoretical)

**Parabosons:** Transform under $(n)$ (symmetric)
**Parafermions:** Transform under $(1^n)$ (antisymmetric)
**Para-order $p$:** Transform under partition with $p$ rows

### The Spin-Statistics Theorem

In relativistic QFT:
- Integer spin → bosons → symmetric statistics
- Half-integer spin → fermions → antisymmetric statistics

This is enforced by the spin-statistics theorem!

### Multi-Electron Wave Functions

For $n$ electrons, the spatial wave function $\psi(x_1, \ldots, x_n)$ and spin wave function $\chi(s_1, \ldots, s_n)$ must combine to give an antisymmetric total:

$$\Psi = \psi \otimes \chi$$

If $\psi$ transforms under partition $\lambda$, then $\chi$ must transform under the conjugate $\lambda'$.

### Young Tableaux and Coupling

The reduction $S^\lambda \otimes S^\mu = \bigoplus_\nu c_{\lambda\mu}^\nu S^\nu$ uses **Littlewood-Richardson coefficients** $c_{\lambda\mu}^\nu$.

This generalizes Clebsch-Gordan coefficients to $S_n$!

---

## 6. Worked Examples

### Example 1: Irreps of $S_3$

Partitions of 3: $(3)$, $(2,1)$, $(1,1,1)$

| Partition | Dimension | Irrep |
|-----------|-----------|-------|
| $(3)$ | $\frac{3!}{3 \cdot 2 \cdot 1} = 1$ | Trivial |
| $(2,1)$ | $\frac{3!}{3 \cdot 1 \cdot 1} = 2$ | Standard |
| $(1,1,1)$ | $\frac{3!}{3 \cdot 2 \cdot 1} = 1$ | Sign |

Check: $1^2 + 2^2 + 1^2 = 6 = 3!$ ✓

### Example 2: Irreps of $S_4$

Partitions of 4: $(4)$, $(3,1)$, $(2,2)$, $(2,1,1)$, $(1,1,1,1)$

Dimensions using hook formula:
- $(4)$: $\frac{24}{4 \cdot 3 \cdot 2 \cdot 1} = 1$
- $(3,1)$: $\frac{24}{4 \cdot 2 \cdot 1 \cdot 1} = 3$
- $(2,2)$: $\frac{24}{3 \cdot 1 \cdot 2 \cdot 1} = 2$ (Wait, need to recompute)

For $(2,2)$:
```
□ □
□ □
```
Hooks: $h(1,1) = 3$, $h(1,2) = 1$, $h(2,1) = 2$, $h(2,2) = 1$

$\dim = \frac{24}{3 \cdot 1 \cdot 2 \cdot 1} = \frac{24}{6} = 2$ ✓

- $(2,1,1)$: Conjugate of $(3,1)$, so $\dim = 3$
- $(1,1,1,1)$: $\dim = 1$ (sign rep)

Check: $1 + 9 + 4 + 9 + 1 = 24 = 4!$ ✓

### Example 3: Two-Electron System

For 2 electrons in spatial states $\phi_a, \phi_b$:

**Symmetric spatial ($\lambda = (2)$):**
$$\psi_S = \frac{1}{\sqrt{2}}[\phi_a(1)\phi_b(2) + \phi_a(2)\phi_b(1)]$$

Spin must be antisymmetric: singlet $|0, 0\rangle$ ($\lambda' = (1,1)$).

**Antisymmetric spatial ($\lambda = (1,1)$):**
$$\psi_A = \frac{1}{\sqrt{2}}[\phi_a(1)\phi_b(2) - \phi_a(2)\phi_b(1)]$$

Spin must be symmetric: triplet $|1, m\rangle$ ($\lambda' = (2)$).

---

## 7. Computational Lab

```python
"""
Day 293: Representations of the Symmetric Group
Young tableaux and irreps of S_n
"""

import numpy as np
from typing import List, Tuple, Dict
from itertools import permutations
from functools import reduce
from math import factorial

def partitions(n: int) -> List[Tuple[int, ...]]:
    """Generate all partitions of n."""
    if n == 0:
        return [()]
    if n < 0:
        return []

    result = []

    def helper(remaining, max_val, current):
        if remaining == 0:
            result.append(tuple(current))
            return
        for i in range(min(remaining, max_val), 0, -1):
            helper(remaining - i, i, current + [i])

    helper(n, n, [])
    return result


def hook_lengths(partition: Tuple[int, ...]) -> List[List[int]]:
    """Compute hook lengths for a partition."""
    if not partition:
        return []

    n_rows = len(partition)
    hooks = []

    for i in range(n_rows):
        row_hooks = []
        for j in range(partition[i]):
            # Hook = boxes below + boxes to the right + 1
            below = sum(1 for k in range(i + 1, n_rows) if partition[k] > j)
            right = partition[i] - j - 1
            h = below + right + 1
            row_hooks.append(h)
        hooks.append(row_hooks)

    return hooks


def dimension_from_hooks(partition: Tuple[int, ...]) -> int:
    """Compute dimension using hook length formula."""
    n = sum(partition)
    hooks = hook_lengths(partition)

    product = 1
    for row in hooks:
        for h in row:
            product *= h

    return factorial(n) // product


def conjugate_partition(partition: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute conjugate partition."""
    if not partition:
        return ()

    # Conjugate: λ'_j = #{i : λ_i >= j}
    max_col = partition[0]
    conj = []
    for j in range(1, max_col + 1):
        col_len = sum(1 for row in partition if row >= j)
        conj.append(col_len)

    return tuple(conj)


def count_standard_tableaux(partition: Tuple[int, ...]) -> int:
    """Count standard Young tableaux of given shape."""
    # This equals the dimension by hook length formula
    return dimension_from_hooks(partition)


def young_symmetrizer_matrix(partition: Tuple[int, ...], tableau: List[List[int]]) -> np.ndarray:
    """
    Compute the Young symmetrizer matrix in the regular representation.

    This is a simplified version for small cases.
    """
    n = sum(partition)
    elements = list(permutations(range(n)))
    G = len(elements)
    elem_to_idx = {p: i for i, p in enumerate(elements)}

    # Row group: permutations within rows
    row_perms = [tuple(range(n))]  # Start with identity
    for row in tableau:
        # Generate all permutations of elements in this row
        from itertools import permutations as perm
        # This is complex; simplified version

    # For demonstration, return a placeholder
    return np.zeros((G, G))


def draw_young_diagram(partition: Tuple[int, ...]) -> str:
    """ASCII representation of Young diagram."""
    lines = []
    for row_len in partition:
        lines.append('□ ' * row_len)
    return '\n'.join(lines)


def draw_young_diagram_with_hooks(partition: Tuple[int, ...]) -> str:
    """Young diagram with hook lengths."""
    hooks = hook_lengths(partition)
    lines = []
    for row_hooks in hooks:
        lines.append(' '.join(str(h) for h in row_hooks))
    return '\n'.join(lines)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("REPRESENTATIONS OF THE SYMMETRIC GROUP")
    print("=" * 60)

    # Example 1: Partitions of small n
    print("\n1. PARTITIONS AND IRREPS")
    print("-" * 40)

    for n in range(1, 6):
        parts = partitions(n)
        print(f"\nPartitions of {n}: ({len(parts)} total)")
        for p in parts:
            d = dimension_from_hooks(p)
            conj = conjugate_partition(p)
            self_conj = "✓" if p == conj else ""
            print(f"  {p}: dim = {d}, conjugate = {conj} {self_conj}")

        # Verify dimension formula
        total = sum(dimension_from_hooks(p)**2 for p in parts)
        print(f"  Σ d² = {total}, {n}! = {factorial(n)}, Match: {total == factorial(n)}")

    # Example 2: Young diagrams
    print("\n2. YOUNG DIAGRAMS")
    print("-" * 40)

    for p in [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]:
        print(f"\nPartition {p}:")
        print(draw_young_diagram(p))
        print(f"Hook lengths:")
        print(draw_young_diagram_with_hooks(p))
        print(f"Dimension: {dimension_from_hooks(p)}")

    # Example 3: Hook length formula verification
    print("\n3. HOOK LENGTH FORMULA")
    print("-" * 40)

    # For S_5
    n = 5
    print(f"Irreps of S_{n}:")
    print(f"{'Partition':<20} {'Hooks':<25} {'Dimension':<10}")
    print("-" * 55)

    for p in partitions(n):
        hooks = hook_lengths(p)
        hooks_flat = [h for row in hooks for h in row]
        hook_prod = reduce(lambda x, y: x * y, hooks_flat)
        d = factorial(n) // hook_prod
        print(f"{str(p):<20} {str(hooks_flat):<25} {d:<10}")

    # Example 4: Two-electron example
    print("\n4. TWO-ELECTRON WAVE FUNCTIONS")
    print("-" * 40)

    print("Partitions of 2: (2), (1,1)")
    print()
    print("Spatial (2) + Spin (1,1) → Singlet:")
    print("  ψ_sym(r₁,r₂) ⊗ |↑↓-↓↑⟩/√2")
    print()
    print("Spatial (1,1) + Spin (2) → Triplet:")
    print("  ψ_antisym(r₁,r₂) ⊗ {|↑↑⟩, |↑↓+↓↑⟩/√2, |↓↓⟩}")

    # Example 5: Branching rules S_n → S_{n-1}
    print("\n5. BRANCHING S_n → S_{n-1}")
    print("-" * 40)

    print("When removing one box from Young diagram:")
    print("S^λ ↓ S_{n-1} = ⊕ S^μ (sum over valid removals)")
    print()

    # Example: (3,2,1) → (3,2), (3,1,1), (2,2,1)
    lambda_p = (3, 2, 1)
    print(f"Restriction of {lambda_p}:")

    # Can remove from end of any row where it doesn't break the pattern
    removals = []
    for i in range(len(lambda_p)):
        new_p = list(lambda_p)
        new_p[i] -= 1
        if new_p[i] == 0:
            new_p.pop(i)
        else:
            # Check it's still a valid partition
            if i + 1 < len(new_p) and new_p[i] < new_p[i + 1]:
                continue
        removals.append(tuple(new_p))

    for r in removals:
        print(f"  → {r}")

    # Example 6: Character values
    print("\n6. SOME CHARACTER VALUES FOR S_4")
    print("-" * 40)

    # Character table of S_4
    # Conjugacy classes: (1⁴), (2,1²), (2²), (3,1), (4)
    # Sizes: 1, 6, 3, 8, 6

    char_table = {
        (4,): [1, 1, 1, 1, 1],
        (3, 1): [3, 1, -1, 0, -1],
        (2, 2): [2, 0, 2, -1, 0],
        (2, 1, 1): [3, -1, -1, 0, 1],
        (1, 1, 1, 1): [1, -1, 1, 1, -1]
    }

    classes = ['(1⁴)', '(2,1²)', '(2²)', '(3,1)', '(4)']
    sizes = [1, 6, 3, 8, 6]

    print(f"{'Partition':<15}", end='')
    for c in classes:
        print(f"{c:>8}", end='')
    print()
    print("-" * 55)

    for p, chi in char_table.items():
        print(f"{str(p):<15}", end='')
        for val in chi:
            print(f"{val:>8}", end='')
        print()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Irreps of S_n ↔ Partitions of n
    2. Hook length formula: dim = n! / ∏ hooks
    3. # standard tableaux = dimension
    4. Conjugate partition → dual representation
    5. Bosons: (n), Fermions: (1^n)
    6. Spin-statistics: total wave function antisymmetric
    7. Young symmetrizers construct irreps explicitly
    """)
```

---

## 8. Practice Problems

### Problem Set A: Partitions and Dimensions

**A1.** List all partitions of 5 and compute the dimension of each corresponding irrep.

**A2.** Verify: For $S_5$, $\sum_\lambda d_\lambda^2 = 120$.

**A3.** Find all self-conjugate partitions of 6.

### Problem Set B: Young Tableaux

**B1.** Count the number of standard Young tableaux of shape $(3, 2)$.

**B2.** Draw all standard Young tableaux of shape $(2, 2)$.

**B3.** Use the hook length formula to find $\dim(S^{(4,2,1)})$ for $S_7$.

### Problem Set C: Quantum Mechanics

**C1.** For 3 identical spin-1/2 fermions in an $s$-orbital, what are the allowed total spin states?

**C2.** **(Helium)** The ground state of helium has both electrons in the 1s orbital. Explain why the spatial wave function must be symmetric and the spin state must be a singlet.

**C3.** Using the representation theory of $S_3$, classify all possible symmetry types for wave functions of 3 identical bosons.

---

## 9. Summary

### Key Results

| Concept | Description |
|---------|-------------|
| Partitions | Classify irreps of $S_n$ |
| Hook length | $\dim = n!/\prod h$ |
| Conjugate | Sign representation twists |
| Standard tableaux | Basis for irrep |
| Young symmetrizer | Projection onto irrep |

### Dimension Table (Small $n$)

| $n$ | Partitions | Irrep Dimensions |
|-----|------------|------------------|
| 1 | (1) | 1 |
| 2 | (2), (1,1) | 1, 1 |
| 3 | (3), (2,1), (1,1,1) | 1, 2, 1 |
| 4 | (4), (3,1), (2,2), (2,1,1), (1,1,1,1) | 1, 3, 2, 3, 1 |

---

## 10. Preview: Day 294

Tomorrow we conclude Week 42 with a comprehensive review of representation theory:
- Integration of all concepts
- Applications to physics
- Preparation for Lie groups

---

*"The theory of group representations, especially that of symmetric groups, is surely the most beautiful subject in all of mathematics." — G.-C. Rota*
