# Day 282: Subgroups and Cosets — Structure Within Groups

## Overview

**Month 11, Week 41, Day 2 — Tuesday**

Today we explore the internal structure of groups through **subgroups** and **cosets**. Subgroups are "groups within groups"—subsets that themselves satisfy the group axioms. Understanding subgroups is essential because they often correspond to restricted symmetries, and the way a group decomposes into cosets reveals deep structural information. The crown jewel of today is **Lagrange's Theorem**, which severely constrains what subgroups can exist.

## Prerequisites

From Day 281:
- Group axioms and basic group theory
- Examples of groups (integers, matrices, permutations)
- Cayley tables and group order

## Learning Objectives

By the end of today, you will be able to:

1. Define subgroups and verify the subgroup criteria
2. Find all subgroups of a given finite group
3. Compute left and right cosets
4. State and prove Lagrange's Theorem
5. Apply Lagrange's Theorem to determine possible subgroup orders
6. Identify normal subgroups and understand their significance

---

## 1. Subgroups

### Definition and Basic Properties

**Definition:** A subset $H \subseteq G$ is a **subgroup** of $G$ (written $H \leq G$) if $H$ is itself a group under the same operation as $G$.

**Trivial Subgroups:** Every group $G$ has at least two subgroups:
- The **trivial subgroup** $\{e\}$
- The **improper subgroup** $G$ itself

A subgroup $H$ with $\{e\} \subsetneq H \subsetneq G$ is called a **proper subgroup**.

### Subgroup Criterion

Checking all four group axioms is tedious. The following criterion simplifies this:

**Theorem (One-Step Subgroup Test):** A non-empty subset $H$ of group $G$ is a subgroup if and only if:
$$\boxed{\forall a, b \in H: \quad ab^{-1} \in H}$$

*Proof:*
($\Rightarrow$) If $H$ is a subgroup, it contains inverses and is closed under the operation.

($\Leftarrow$) Assume $ab^{-1} \in H$ for all $a, b \in H$.
- **Non-empty:** By assumption, $\exists a \in H$.
- **Identity:** $e = aa^{-1} \in H$.
- **Inverses:** For $a \in H$: $a^{-1} = ea^{-1} \in H$.
- **Closure:** For $a, b \in H$: $ab = a(b^{-1})^{-1} \in H$ (since $b^{-1} \in H$).
- **Associativity:** Inherited from $G$. ∎

**Theorem (Two-Step Subgroup Test):** A non-empty subset $H \leq G$ if and only if:
1. $H$ is closed under the operation
2. $H$ is closed under inverses

### Examples of Subgroups

**Example 1:** In $(\mathbb{Z}, +)$:
- $n\mathbb{Z} = \{nk : k \in \mathbb{Z}\}$ is a subgroup for any $n \in \mathbb{Z}$
- $2\mathbb{Z}$ = even integers, $3\mathbb{Z}$ = multiples of 3, etc.

**Example 2:** In $GL_n(\mathbb{R})$:
- $SL_n(\mathbb{R}) = \{A : \det(A) = 1\}$ (special linear group)
- $O_n(\mathbb{R}) = \{A : A^TA = I\}$ (orthogonal group)
- $SO_n(\mathbb{R}) = O_n \cap SL_n$ (special orthogonal group)

**Example 3:** In $D_n$ (dihedral group):
- Rotation subgroup $\{e, r, r^2, \ldots, r^{n-1}\}$ is a subgroup of order $n$
- $\{e, s\}$ is a subgroup of order 2 for each reflection $s$

### Generated Subgroups

**Definition:** The subgroup **generated** by element $a$ is:
$$\langle a \rangle = \{a^n : n \in \mathbb{Z}\} = \{\ldots, a^{-2}, a^{-1}, e, a, a^2, \ldots\}$$

This is the smallest subgroup containing $a$.

**Definition:** The **order** of element $a$, denoted $|a|$ or $\text{ord}(a)$, is the smallest positive integer $n$ such that $a^n = e$. If no such $n$ exists, $|a| = \infty$.

**Theorem:** For finite groups, $|a| = |\langle a \rangle|$.

---

## 2. Cosets

### Left and Right Cosets

**Definition:** For $H \leq G$ and $g \in G$:
- The **left coset** of $H$ by $g$ is: $gH = \{gh : h \in H\}$
- The **right coset** of $H$ by $g$ is: $Hg = \{hg : h \in H\}$

**Key Properties of Cosets:**

1. $|gH| = |H| = |Hg|$ (all cosets have the same size as $H$)

2. $gH = H \Leftrightarrow g \in H$

3. $aH = bH \Leftrightarrow a^{-1}b \in H \Leftrightarrow a \equiv b \pmod{H}$

4. Two left cosets are either identical or disjoint

5. The left cosets partition $G$: $G = \bigsqcup_{g \in G} gH$

### Example: Cosets in $\mathbb{Z}_6$

Let $G = \mathbb{Z}_6$ and $H = \{0, 3\}$ (subgroup of order 2).

Left cosets (which equal right cosets since $G$ is abelian):
- $0 + H = \{0, 3\}$
- $1 + H = \{1, 4\}$
- $2 + H = \{2, 5\}$
- $3 + H = \{3, 0\} = \{0, 3\} = 0 + H$
- etc.

There are 3 distinct cosets, and $6 = 2 \times 3 = |H| \times (\text{number of cosets})$.

### Index

**Definition:** The **index** of $H$ in $G$, denoted $[G:H]$, is the number of distinct left cosets of $H$ in $G$.

$$[G:H] = \frac{|G|}{|H|} \quad \text{(for finite groups)}$$

---

## 3. Lagrange's Theorem

### Statement and Proof

**Theorem (Lagrange):** If $G$ is a finite group and $H \leq G$, then $|H|$ divides $|G|$.

$$\boxed{|G| = |H| \cdot [G:H]}$$

*Proof:*
1. The distinct left cosets of $H$ partition $G$ (they are disjoint and cover $G$)
2. Each coset has exactly $|H|$ elements
3. If there are $[G:H]$ distinct cosets, then:
$$|G| = (\text{number of cosets}) \times (\text{elements per coset}) = [G:H] \times |H|$$
∎

### Consequences of Lagrange's Theorem

**Corollary 1:** The order of any element divides the order of the group.

*Proof:* $|a| = |\langle a \rangle|$ divides $|G|$ since $\langle a \rangle \leq G$. ∎

**Corollary 2:** For any $a \in G$: $a^{|G|} = e$

*Proof:* If $|a| = n$, then $n | |G|$, so $|G| = nk$ for some $k$. Thus $a^{|G|} = a^{nk} = (a^n)^k = e^k = e$. ∎

**Corollary 3 (Fermat's Little Theorem):** For prime $p$ and $a$ not divisible by $p$:
$$a^{p-1} \equiv 1 \pmod{p}$$

*Proof:* $a \in (\mathbb{Z}/p\mathbb{Z})^*$, which has order $p-1$. By Corollary 2, $a^{p-1} = 1$. ∎

**Corollary 4:** A group of prime order is cyclic.

*Proof:* Let $|G| = p$ prime. Take $a \neq e$. Then $|\langle a \rangle|$ divides $p$, so $|\langle a \rangle| = 1$ or $p$. Since $a \neq e$, $|\langle a \rangle| = p = |G|$, so $G = \langle a \rangle$. ∎

### What Lagrange's Theorem Does NOT Say

Lagrange's theorem says: If $H \leq G$, then $|H|$ divides $|G|$.

It does **NOT** say: If $d$ divides $|G|$, then there exists a subgroup of order $d$.

**Counterexample:** $A_4$ (alternating group on 4 elements) has order 12. While 6 divides 12, $A_4$ has no subgroup of order 6.

---

## 4. Normal Subgroups

### Definition and Motivation

**Definition:** A subgroup $N \leq G$ is **normal** (written $N \trianglelefteq G$) if:
$$\forall g \in G: \quad gN = Ng$$

Equivalently: $gNg^{-1} = N$ for all $g \in G$.

**Why Normal Subgroups Matter:**
- They are exactly the subgroups for which $G/N$ (quotient) is a group
- They are kernels of homomorphisms
- In physics: normal subgroups correspond to symmetries that "commute" with all transformations

### Examples

1. **Every subgroup of an abelian group is normal** (since $gN = Ng$ always)

2. In $S_3$: $A_3 = \{e, (123), (132)\}$ is normal (index 2 subgroups are always normal)

3. In $D_n$: The rotation subgroup $\{e, r, r^2, \ldots, r^{n-1}\}$ is normal

4. **$SL_n \trianglelefteq GL_n$** (kernel of determinant homomorphism)

### Normality Criterion

**Theorem:** The following are equivalent for $N \leq G$:
1. $N \trianglelefteq G$
2. $gN = Ng$ for all $g \in G$
3. $gNg^{-1} = N$ for all $g \in G$
4. $gNg^{-1} \subseteq N$ for all $g \in G$
5. $N$ is the kernel of some homomorphism from $G$

### Index 2 Subgroups

**Theorem:** Every subgroup of index 2 is normal.

*Proof:* Let $[G:H] = 2$. Then there are exactly 2 left cosets: $H$ and $gH$ for $g \notin H$.
Similarly, 2 right cosets: $H$ and $Hg$.
Since $gH \neq H$ and $Hg \neq H$, we must have $gH = Hg$ (both equal $G \setminus H$). ∎

---

## 5. Quantum Mechanics Connection

### Subgroups and Restricted Symmetries

In quantum mechanics, subgroups often represent restricted symmetries:

| Full Group | Subgroup | Physical Meaning |
|------------|----------|------------------|
| SO(3) | SO(2) | Axial symmetry (rotations about one axis) |
| Lorentz | Rotations | No boosts, only spatial rotations |
| $U(n)$ | $SU(n)$ | Unit determinant transformations |
| Poincaré | Translations | No rotations/boosts |

### Cosets and Equivalence Classes

In path integral formulation, gauge equivalent configurations are cosets:
$$[A] = A + \{\text{pure gauge transformations}\}$$

This is why we "mod out" by gauge transformations.

### Conservation Laws

**Noether's Theorem** connects continuous symmetries to conservation laws:
- Subgroup of time translations → Energy conservation
- Subgroup of spatial translations → Momentum conservation
- Subgroup of rotations → Angular momentum conservation

### Normal Subgroups and Gauge Symmetry

Normal subgroups play a crucial role in gauge theory:
- Gauge group $G$ acts on fields
- Center $Z(G)$ (always normal) represents global phase transformations
- Quotient $G/Z(G)$ represents "true" gauge degrees of freedom

---

## 6. Worked Examples

### Example 1: Find All Subgroups of $\mathbb{Z}_{12}$

By Lagrange's theorem, subgroup orders must divide 12: 1, 2, 3, 4, 6, 12.

- Order 1: $\{0\}$
- Order 2: $\{0, 6\}$ (generated by 6)
- Order 3: $\{0, 4, 8\}$ (generated by 4)
- Order 4: $\{0, 3, 6, 9\}$ (generated by 3)
- Order 6: $\{0, 2, 4, 6, 8, 10\}$ (generated by 2)
- Order 12: $\mathbb{Z}_{12}$ (generated by 1)

Subgroup lattice:
```
           Z_12
          /    \
       <2>      <3>
        |    ×    |
       <4>      <6>
          \    /
           <0>
```

### Example 2: Cosets in $S_3$

Let $H = \{e, (12)\} \leq S_3$.

Left cosets:
- $eH = H = \{e, (12)\}$
- $(13)H = \{(13), (13)(12)\} = \{(13), (132)\}$
- $(23)H = \{(23), (23)(12)\} = \{(23), (123)\}$

Right cosets:
- $He = H = \{e, (12)\}$
- $H(13) = \{(13), (12)(13)\} = \{(13), (123)\}$
- $H(23) = \{(23), (12)(23)\} = \{(23), (132)\}$

Notice: $(13)H \neq H(13)$, so $H$ is NOT normal in $S_3$.

### Example 3: Verify $A_n \trianglelefteq S_n$

The alternating group $A_n$ = even permutations.

**Method 1:** $[S_n : A_n] = 2$, so $A_n$ is normal (index 2 subgroups are normal).

**Method 2:** $A_n = \ker(\text{sign})$ where sign: $S_n \to \{1, -1\}$ is the sign homomorphism.

---

## 7. Computational Lab

```python
"""
Day 282: Subgroups and Cosets
Finding subgroups, computing cosets, verifying Lagrange's theorem
"""

import numpy as np
from itertools import combinations
from typing import List, Set, Tuple, Optional
from collections import defaultdict

class GroupAnalyzer:
    """
    Tools for analyzing subgroups and cosets of finite groups.
    """

    def __init__(self, elements: List, operation):
        """
        Initialize with group elements and operation.

        Parameters:
            elements: List of group elements
            operation: Binary operation function
        """
        self.elements = list(elements)
        self.n = len(elements)
        self.op = operation
        self.elem_to_idx = {e: i for i, e in enumerate(elements)}

        # Build multiplication table
        self.table = {}
        for a in elements:
            for b in elements:
                self.table[(a, b)] = operation(a, b)

        # Find identity
        self.identity = self._find_identity()

        # Build inverse map
        self.inverses = self._find_inverses()

    def _find_identity(self):
        """Find the identity element."""
        for e in self.elements:
            if all(self.op(e, a) == a and self.op(a, e) == a for a in self.elements):
                return e
        return None

    def _find_inverses(self):
        """Find inverse of each element."""
        inv = {}
        for a in self.elements:
            for b in self.elements:
                if self.op(a, b) == self.identity:
                    inv[a] = b
                    break
        return inv

    def is_subgroup(self, H: Set) -> bool:
        """
        Check if H is a subgroup using one-step test.

        Parameters:
            H: Set of elements to test

        Returns:
            True if H is a subgroup
        """
        if not H:  # Empty set
            return False

        H_list = list(H)

        # Check: for all a, b in H, a * b^(-1) in H
        for a in H_list:
            for b in H_list:
                if a not in self.inverses or b not in self.inverses:
                    return False
                b_inv = self.inverses[b]
                product = self.op(a, b_inv)
                if product not in H:
                    return False

        return True

    def find_all_subgroups(self) -> List[Set]:
        """
        Find all subgroups of the group.

        Returns:
            List of subgroups (each as a set)
        """
        subgroups = []

        # Check all subsets
        for r in range(1, self.n + 1):
            for subset in combinations(self.elements, r):
                H = set(subset)
                if self.is_subgroup(H):
                    subgroups.append(H)

        return subgroups

    def generated_subgroup(self, generators: List) -> Set:
        """
        Find the subgroup generated by given elements.

        Parameters:
            generators: List of generating elements

        Returns:
            The generated subgroup
        """
        H = set(generators)
        H.add(self.identity)

        # Keep adding products and inverses until closed
        changed = True
        while changed:
            changed = False
            new_elements = set()
            for a in H:
                # Add inverse
                if self.inverses[a] not in H:
                    new_elements.add(self.inverses[a])
                    changed = True
                # Add products
                for b in H:
                    prod = self.op(a, b)
                    if prod not in H:
                        new_elements.add(prod)
                        changed = True
            H.update(new_elements)

        return H

    def left_cosets(self, H: Set) -> List[Set]:
        """
        Compute all left cosets gH.

        Parameters:
            H: A subgroup

        Returns:
            List of left cosets
        """
        cosets = []
        covered = set()

        for g in self.elements:
            if g in covered:
                continue

            # Compute gH
            coset = set()
            for h in H:
                coset.add(self.op(g, h))

            cosets.append(coset)
            covered.update(coset)

        return cosets

    def right_cosets(self, H: Set) -> List[Set]:
        """
        Compute all right cosets Hg.

        Parameters:
            H: A subgroup

        Returns:
            List of right cosets
        """
        cosets = []
        covered = set()

        for g in self.elements:
            if g in covered:
                continue

            # Compute Hg
            coset = set()
            for h in H:
                coset.add(self.op(h, g))

            cosets.append(coset)
            covered.update(coset)

        return cosets

    def is_normal(self, H: Set) -> bool:
        """
        Check if H is a normal subgroup.

        Parameters:
            H: A subgroup to test

        Returns:
            True if H is normal
        """
        if not self.is_subgroup(H):
            return False

        # Check gHg^(-1) = H for all g
        for g in self.elements:
            g_inv = self.inverses[g]
            conjugate = set()
            for h in H:
                conjugate.add(self.op(self.op(g, h), g_inv))
            if conjugate != H:
                return False

        return True

    def index(self, H: Set) -> int:
        """Compute the index [G:H]."""
        return len(self.left_cosets(H))

    def verify_lagrange(self) -> dict:
        """
        Verify Lagrange's theorem for all subgroups.

        Returns:
            Dictionary with verification results
        """
        subgroups = self.find_all_subgroups()
        results = {
            'group_order': self.n,
            'subgroups': []
        }

        for H in subgroups:
            H_order = len(H)
            index = self.index(H)
            divides = (self.n % H_order == 0)
            product_check = (H_order * index == self.n)

            results['subgroups'].append({
                'subgroup': H,
                'order': H_order,
                'index': index,
                'divides_G': divides,
                '|H|*[G:H]=|G|': product_check,
                'is_normal': self.is_normal(H)
            })

        return results


def create_Zn(n: int):
    """Create cyclic group Z_n."""
    elements = list(range(n))
    operation = lambda a, b: (a + b) % n
    return GroupAnalyzer(elements, operation)


def create_Sn(n: int):
    """Create symmetric group S_n."""
    from itertools import permutations

    elements = list(permutations(range(n)))

    def compose(p1, p2):
        return tuple(p1[p2[i]] for i in range(n))

    return GroupAnalyzer(elements, compose)


def create_Dn(n: int):
    """Create dihedral group D_n."""
    elements = [(k, 0) for k in range(n)] + [(k, 1) for k in range(n)]

    def multiply(g1, g2):
        k1, s1 = g1
        k2, s2 = g2
        if s1 == 0:
            if s2 == 0:
                return ((k1 + k2) % n, 0)
            else:
                return ((-k1 + k2) % n, 1)
        else:
            if s2 == 0:
                return ((k1 - k2) % n, 1)
            else:
                return ((k1 - k2) % n, 0)

    return GroupAnalyzer(elements, multiply)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("SUBGROUPS AND COSETS")
    print("=" * 60)

    # Example 1: Z_12
    print("\n1. SUBGROUPS OF Z_12")
    print("-" * 40)

    Z12 = create_Zn(12)
    subgroups = Z12.find_all_subgroups()

    print(f"Group order: |Z_12| = 12")
    print(f"Divisors of 12: 1, 2, 3, 4, 6, 12")
    print(f"\nSubgroups found ({len(subgroups)} total):")

    for H in sorted(subgroups, key=len):
        print(f"  |H| = {len(H):2d}: {sorted(H)}")
        if len(H) > 1 and len(H) < 12:
            # Find generator
            for a in H:
                if a != 0 and Z12.generated_subgroup([a]) == H:
                    print(f"           Generated by: {a}")
                    break

    # Example 2: Cosets in S_3
    print("\n2. COSETS IN S_3")
    print("-" * 40)

    S3 = create_Sn(3)
    print(f"Group order: |S_3| = {S3.n}")

    # Find subgroup H = {e, (12)}
    e = (0, 1, 2)
    swap = (1, 0, 2)
    H = {e, swap}

    print(f"\nSubgroup H = {{{e}, {swap}}}, |H| = 2")

    left = S3.left_cosets(H)
    right = S3.right_cosets(H)

    print(f"\nLeft cosets (gH):")
    for i, coset in enumerate(left):
        print(f"  Coset {i+1}: {coset}")

    print(f"\nRight cosets (Hg):")
    for i, coset in enumerate(right):
        print(f"  Coset {i+1}: {coset}")

    print(f"\nIs H normal? {S3.is_normal(H)}")
    print(f"(Left cosets ≠ Right cosets, so H is not normal)")

    # Find normal subgroup A_3
    # A_3 = {e, (012), (021)} = {(0,1,2), (1,2,0), (2,0,1)}
    A3 = {(0, 1, 2), (1, 2, 0), (2, 0, 1)}
    print(f"\nAlternating group A_3 = {A3}")
    print(f"Is A_3 normal? {S3.is_normal(A3)}")
    print(f"Index [S_3 : A_3] = {S3.index(A3)}")

    # Example 3: Lagrange verification
    print("\n3. LAGRANGE'S THEOREM VERIFICATION (Z_8)")
    print("-" * 40)

    Z8 = create_Zn(8)
    results = Z8.verify_lagrange()

    print(f"|G| = {results['group_order']}")
    print(f"\nAll subgroups:")
    for info in results['subgroups']:
        H = sorted(info['subgroup'])
        print(f"  H = {H}")
        print(f"    |H| = {info['order']}, [G:H] = {info['index']}")
        print(f"    |H| divides |G|: {info['divides_G']}")
        print(f"    |H| × [G:H] = |G|: {info['|H|*[G:H]=|G|']}")
        print(f"    Normal: {info['is_normal']}")

    # Example 4: D_4 analysis
    print("\n4. DIHEDRAL GROUP D_4 (Square symmetries)")
    print("-" * 40)

    D4 = create_Dn(4)
    print(f"|D_4| = {D4.n}")

    subgroups = D4.find_all_subgroups()
    print(f"\nNumber of subgroups: {len(subgroups)}")

    # Group by order
    by_order = defaultdict(list)
    for H in subgroups:
        by_order[len(H)].append(H)

    for order in sorted(by_order.keys()):
        print(f"\n  Order {order}:")
        for H in by_order[order]:
            normal = "✓ normal" if D4.is_normal(H) else ""
            print(f"    {H} {normal}")

    # Example 5: Fermat's Little Theorem
    print("\n5. FERMAT'S LITTLE THEOREM")
    print("-" * 40)

    p = 7
    print(f"For prime p = {p}:")
    print(f"Fermat: a^(p-1) ≡ 1 (mod p) for a ≢ 0 (mod p)")

    for a in range(1, p):
        result = pow(a, p - 1, p)
        print(f"  {a}^{p-1} = {a}^{6} ≡ {result} (mod {p})")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Lagrange's Theorem: |H| always divides |G|
    2. Converse is FALSE: divisor doesn't guarantee subgroup
    3. Index 2 subgroups are always normal
    4. Cosets partition the group into equal-sized pieces
    5. Normal subgroups have equal left and right cosets
    6. In QM: normal subgroups → gauge-invariant physics
    """)
```

---

## 8. Practice Problems

### Problem Set A: Subgroup Verification

**A1.** Show that $H = \{1, -1\}$ is a subgroup of $(\mathbb{R}^*, \times)$ but not of $(\mathbb{R}, +)$.

**A2.** Verify that $\{0, 2, 4\}$ is a subgroup of $(\mathbb{Z}_6, +)$.

**A3.** Is the set of $2 \times 2$ diagonal matrices a subgroup of $GL_2(\mathbb{R})$? Justify.

### Problem Set B: Cosets and Lagrange

**B1.** Find all left cosets of $3\mathbb{Z}$ in $\mathbb{Z}$.

**B2.** In $\mathbb{Z}_{20}$, find all subgroups and verify Lagrange's theorem for each.

**B3.** Prove that a group of order 35 must have elements of order 5 and order 7.

### Problem Set C: Advanced

**C1.** Prove: If $H, K \leq G$ with $|H|$ and $|K|$ coprime, then $H \cap K = \{e\}$.

**C2.** Let $H \leq G$ with $[G:H] = 2$. Prove $H \trianglelefteq G$.

**C3.** **(Physics)** In the Standard Model, $SU(3) \times SU(2) \times U(1)$ acts on particle fields. Explain why the center $Z = \mathbb{Z}_6$ plays a special role.

---

## 9. Solutions

### Solution B3

$|G| = 35 = 5 \times 7$.

By Lagrange's theorem, element orders must divide 35. Possible orders: 1, 5, 7, 35.

If all non-identity elements had order 35, then $G$ would be cyclic and contain elements of all orders dividing 35, including 5 and 7. ✓

If not all elements have order 35, then some element $a \neq e$ has order 5 or 7.

**Claim:** $G$ contains an element of order 5.

Suppose not. Then all non-identity elements have order 7 or 35. Elements of order 7 generate subgroups of order 7. Subgroups of order 7 partition into 6 non-identity elements plus identity. So elements of order 7 come in groups of 6.

But $34 = |G| - 1$ is not divisible by 6 (34 = 5 × 6 + 4), contradiction.

Therefore $G$ has an element of order 5. Similarly for order 7. ∎

---

## 10. Summary

### Key Concepts

| Concept | Definition | Key Property |
|---------|------------|--------------|
| Subgroup | $H \leq G$ if $H$ is a group under same operation | $ab^{-1} \in H$ for all $a,b \in H$ |
| Coset | $gH = \{gh : h \in H\}$ | All cosets have same size |
| Index | $[G:H] = $ number of cosets | $|G| = |H| \cdot [G:H]$ |
| Normal | $gN = Ng$ for all $g$ | Kernel of homomorphism |

### Key Theorems

$$\boxed{\text{Lagrange: } |H| \text{ divides } |G|}$$

$$\boxed{a^{|G|} = e \text{ for all } a \in G}$$

$$\boxed{[G:H] = 2 \Rightarrow H \trianglelefteq G}$$

---

## 11. Preview: Day 283

Tomorrow we study **group homomorphisms**:
- Definition of homomorphisms and isomorphisms
- Kernel and image
- The First Isomorphism Theorem: $G/\ker(\phi) \cong \text{Im}(\phi)$
- Examples from physics: symmetry breaking and effective theories

---

*"The theory of groups is a branch of mathematics in which one does something to something and then compares the result with the result of doing the same thing to something else." — James Newman*
