# Day 284: Quotient Groups — Collapsing Structure

## Overview

**Month 11, Week 41, Day 4 — Thursday**

Today we construct **quotient groups** (also called factor groups), one of the most powerful constructions in abstract algebra. Given a normal subgroup $N \trianglelefteq G$, we can "collapse" $N$ to the identity, creating a smaller group $G/N$ that captures the "essential" structure of $G$ modulo $N$. This construction is fundamental in physics: gauge equivalence, symmetry breaking, and equivalence classes all involve quotient groups.

## Prerequisites

From Days 281-283:
- Group axioms and subgroups
- Cosets and normal subgroups
- Homomorphisms, kernels, and the First Isomorphism Theorem

## Learning Objectives

By the end of today, you will be able to:

1. Construct the quotient group $G/N$ from a normal subgroup
2. Verify that coset multiplication is well-defined
3. Prove that $G/N$ is a group
4. Compute quotient groups for standard examples
5. Apply the quotient construction to physical systems
6. Use quotient groups to simplify group-theoretic problems

---

## 1. The Quotient Construction

### Why Normal Subgroups?

Recall: $N \trianglelefteq G$ means $gN = Ng$ for all $g \in G$.

The key question: Can we define multiplication on cosets?

**Attempt:** Define $(aN)(bN) = (ab)N$

For this to make sense, we need **well-definedness**: if $aN = a'N$ and $bN = b'N$, then $(ab)N = (a'b')N$.

**Theorem:** Coset multiplication is well-defined if and only if $N$ is normal.

*Proof of necessity:*
Suppose coset multiplication is well-defined. Take $n \in N$ and $g \in G$. We have:
- $gN = gN$ and $nN = N$ (since $n \in N$)
- By well-definedness: $(gn)N = (g \cdot n)N = gN \cdot nN = gN \cdot N = gN$

So $gn \in gN$, meaning $gn = gn'$ for some $n' \in N$, thus $n = n'$.

Also, $(ng)N = nN \cdot gN = N \cdot gN = gN$, so $ng \in gN$.

This shows $gN = Ng$, hence $N$ is normal. ∎

*Proof of sufficiency:*
Suppose $N \trianglelefteq G$. Let $aN = a'N$ and $bN = b'N$.
Then $a' = an_1$ and $b' = bn_2$ for some $n_1, n_2 \in N$.

$$a'b' = an_1 \cdot bn_2 = a(n_1b)n_2$$

Since $N$ is normal, $n_1 b = b n_3$ for some $n_3 \in N$ (because $bN = Nb$).

$$a'b' = ab n_3 n_2 = ab(n_3 n_2)$$

Since $n_3 n_2 \in N$, we have $a'b' \in (ab)N$, so $(a'b')N = (ab)N$. ∎

### The Quotient Group

**Definition:** For $N \trianglelefteq G$, the **quotient group** (or factor group) is:

$$\boxed{G/N = \{gN : g \in G\}}$$

with operation:

$$\boxed{(aN)(bN) = (ab)N}$$

**Theorem:** $G/N$ is a group.

*Proof:*
- **Closure:** $(aN)(bN) = (ab)N \in G/N$ ✓
- **Associativity:** $((aN)(bN))(cN) = (ab)N \cdot cN = ((ab)c)N = (a(bc))N = aN \cdot (bc)N = (aN)((bN)(cN))$ ✓
- **Identity:** $N = eN$ is the identity since $(aN)(N) = (ae)N = aN$ ✓
- **Inverse:** $(aN)^{-1} = a^{-1}N$ since $(aN)(a^{-1}N) = (aa^{-1})N = N$ ✓ ∎

### The Quotient Homomorphism

**Definition:** The **quotient map** (or canonical projection) is:

$$\pi: G \to G/N, \quad \pi(g) = gN$$

**Theorem:** $\pi$ is a surjective homomorphism with $\ker(\pi) = N$.

*Proof:*
- Homomorphism: $\pi(ab) = (ab)N = (aN)(bN) = \pi(a)\pi(b)$ ✓
- Surjective: Every coset $gN$ is $\pi(g)$ ✓
- Kernel: $\pi(g) = N \Leftrightarrow gN = N \Leftrightarrow g \in N$ ✓ ∎

---

## 2. Computing Quotient Groups

### Example 1: $\mathbb{Z}/n\mathbb{Z}$

Let $G = \mathbb{Z}$ (integers under addition) and $N = n\mathbb{Z}$ (multiples of $n$).

Cosets: $k + n\mathbb{Z} = \{k + nm : m \in \mathbb{Z}\}$

There are exactly $n$ distinct cosets: $\{0 + n\mathbb{Z}, 1 + n\mathbb{Z}, \ldots, (n-1) + n\mathbb{Z}\}$

$$\boxed{\mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}_n}$$

This is why we call $\mathbb{Z}_n$ "integers modulo $n$."

### Example 2: $S_n/A_n$

$A_n \trianglelefteq S_n$ (alternating group is normal in symmetric group).

Cosets:
- $A_n$ (even permutations)
- $\tau A_n$ for any transposition $\tau$ (odd permutations)

Only 2 cosets, so $|S_n/A_n| = 2$.

$$\boxed{S_n/A_n \cong \mathbb{Z}_2}$$

### Example 3: $GL_n/SL_n$

$SL_n \trianglelefteq GL_n$ (kernel of determinant).

Cosets: $A \cdot SL_n = \{B \in GL_n : \det(B) = \det(A)\}$

The cosets are parameterized by determinants!

$$\boxed{GL_n(\mathbb{R})/SL_n(\mathbb{R}) \cong \mathbb{R}^*}$$

### Example 4: $D_n / \langle r \rangle$

In the dihedral group $D_n$, the rotation subgroup $\langle r \rangle = \{e, r, r^2, \ldots, r^{n-1}\}$ is normal.

Cosets:
- $\langle r \rangle$ (rotations)
- $s\langle r \rangle$ (reflections)

$$\boxed{D_n / \langle r \rangle \cong \mathbb{Z}_2}$$

### Example 5: Center Quotient

For any group $G$, the center $Z(G) = \{z \in G : zg = gz \text{ for all } g\}$ is normal.

The quotient $G/Z(G)$ is called the **inner automorphism group**.

For non-abelian groups, $G/Z(G)$ captures "how non-abelian" $G$ is.

---

## 3. The Isomorphism Theorems Revisited

### First Isomorphism Theorem (Restated)

If $\phi: G \to H$ is a homomorphism, the induced map:
$$\bar{\phi}: G/\ker(\phi) \to \text{Im}(\phi), \quad \bar{\phi}(g\ker(\phi)) = \phi(g)$$
is an isomorphism.

### Correspondence Theorem

**Theorem:** There is a bijection between:
- Subgroups of $G/N$
- Subgroups of $G$ containing $N$

given by $H \mapsto H/N$ and $K \mapsto \pi^{-1}(K)$.

Moreover, $H \trianglelefteq G$ if and only if $H/N \trianglelefteq G/N$.

### Third Isomorphism Theorem (Restated)

If $N \trianglelefteq K \trianglelefteq G$, then:
$$(G/N)/(K/N) \cong G/K$$

"Quotient of quotient is quotient by product."

---

## 4. Properties of Quotient Groups

### Order

$$|G/N| = [G:N] = \frac{|G|}{|N|}$$

### Abelianization

**Definition:** For any group $G$, the **commutator subgroup** is:
$$[G, G] = \langle aba^{-1}b^{-1} : a, b \in G \rangle$$

**Theorem:** $[G, G] \trianglelefteq G$ and $G/[G,G]$ is abelian.

The quotient $G/[G,G]$ is called the **abelianization** of $G$—it's the "most abelian" quotient of $G$.

### Simple Groups

**Definition:** A group $G$ is **simple** if its only normal subgroups are $\{e\}$ and $G$.

Simple groups are "atoms" of group theory—they cannot be broken down via quotients.

Examples:
- $\mathbb{Z}_p$ for prime $p$
- $A_n$ for $n \geq 5$
- Many finite simple groups (classified completely!)

---

## 5. Quantum Mechanics Connection

### Gauge Equivalence

In gauge theory, field configurations $A$ and $A'$ are **gauge equivalent** if they differ by a gauge transformation:
$$A' = A + d\lambda$$

The space of gauge-inequivalent configurations is a quotient:
$$\mathcal{A}/\mathcal{G}$$

where $\mathcal{A}$ is the space of all configurations and $\mathcal{G}$ is the gauge group.

### Projective Hilbert Space

In quantum mechanics, states $|\psi\rangle$ and $e^{i\theta}|\psi\rangle$ represent the same physical state.

The true state space is:
$$\mathbb{P}\mathcal{H} = \mathcal{H}/U(1)$$

This is projective Hilbert space—a quotient!

### Symmetry Breaking

When symmetry $G$ is spontaneously broken to subgroup $H$:
- Order parameter lives in $G/H$
- Goldstone bosons correspond to "directions" in $G/H$
- $\dim(G/H) = \dim(G) - \dim(H)$ massless modes

**Example:** Ferromagnet breaks SO(3) → SO(2)
$$SO(3)/SO(2) \cong S^2$$
Magnetization direction is on a 2-sphere.

### Quotient Groups in Particle Physics

The Standard Model gauge group is often written:
$$\frac{SU(3) \times SU(2) \times U(1)}{\mathbb{Z}_6}$$

The $\mathbb{Z}_6$ quotient identifies certain redundant configurations.

---

## 6. Worked Examples

### Example 1: Compute $D_6/\langle r^2 \rangle$

$D_6$ has elements: $\{e, r, r^2, r^3, r^4, r^5, s, sr, sr^2, sr^3, sr^4, sr^5\}$ with $|D_6| = 12$.

$\langle r^2 \rangle = \{e, r^2, r^4\}$ has order 3.

$|D_6/\langle r^2 \rangle| = 12/3 = 4$.

Cosets:
- $\{e, r^2, r^4\}$
- $\{r, r^3, r^5\}$
- $\{s, sr^2, sr^4\}$
- $\{sr, sr^3, sr^5\}$

Call these $E, R, S, SR$. Compute products:
- $R^2 = \{r^2, r^4, r^6\} = \{r^2, r^4, e\} = E$
- $S^2 = \{s^2, ...\} = E$
- $SR \cdot R = S \cdot R^2 = S \cdot E = S$... (similar pattern to $D_2$)

$$D_6/\langle r^2 \rangle \cong D_2 \cong V_4 \cong \mathbb{Z}_2 \times \mathbb{Z}_2$$

### Example 2: Verify $(\mathbb{Z} \times \mathbb{Z})/\langle(1,1)\rangle$

$N = \langle(1,1)\rangle = \{(n, n) : n \in \mathbb{Z}\}$

Cosets $(a, b) + N = \{(a+n, b+n) : n \in \mathbb{Z}\}$

Two elements $(a, b)$ and $(a', b')$ are in the same coset iff:
$(a, b) - (a', b') = (a-a', b-b') \in N$
$\Leftrightarrow a - a' = b - b'$
$\Leftrightarrow a - b = a' - b'$

So cosets are determined by $a - b$. The map:
$$\phi: (\mathbb{Z} \times \mathbb{Z})/N \to \mathbb{Z}, \quad (a,b) + N \mapsto a - b$$

is an isomorphism!

$$(\mathbb{Z} \times \mathbb{Z})/\langle(1,1)\rangle \cong \mathbb{Z}$$

### Example 3: Quaternion Group Quotient

The quaternion group $Q_8 = \{\pm 1, \pm i, \pm j, \pm k\}$ has center $Z(Q_8) = \{\pm 1\}$.

$|Q_8/Z(Q_8)| = 8/2 = 4$.

Cosets:
- $\{1, -1\}$
- $\{i, -i\}$
- $\{j, -j\}$
- $\{k, -k\}$

Multiplication table shows this is the Klein four-group:
$$Q_8/Z(Q_8) \cong V_4$$

---

## 7. Computational Lab

```python
"""
Day 284: Quotient Groups
Computing quotient groups and verifying their structure
"""

import numpy as np
from typing import List, Set, Dict, Callable, Optional, Tuple
from itertools import product

class QuotientGroup:
    """
    Construct and analyze quotient groups G/N.
    """

    def __init__(self, G_elements: List, G_operation: Callable,
                 N_elements: Set):
        """
        Initialize quotient group G/N.

        Parameters:
            G_elements: Elements of group G
            G_operation: Operation on G
            N_elements: Elements of normal subgroup N
        """
        self.G = list(G_elements)
        self.op = G_operation
        self.N = set(N_elements)

        # Find identity in G
        self.identity = self._find_identity()

        # Find inverses in G
        self.inverses = self._find_inverses()

        # Verify N is a subgroup
        if not self._is_subgroup(self.N):
            raise ValueError("N is not a subgroup of G")

        # Verify N is normal
        if not self._is_normal(self.N):
            raise ValueError("N is not normal in G")

        # Compute cosets
        self.cosets = self._compute_cosets()

        # Build quotient multiplication table
        self._build_quotient_table()

    def _find_identity(self):
        for e in self.G:
            if all(self.op(e, g) == g and self.op(g, e) == g for g in self.G):
                return e
        return None

    def _find_inverses(self):
        inv = {}
        for a in self.G:
            for b in self.G:
                if self.op(a, b) == self.identity:
                    inv[a] = b
                    break
        return inv

    def _is_subgroup(self, H: Set) -> bool:
        if self.identity not in H:
            return False
        for a in H:
            if self.inverses.get(a) not in H:
                return False
            for b in H:
                if self.op(a, b) not in H:
                    return False
        return True

    def _is_normal(self, N: Set) -> bool:
        for g in self.G:
            g_inv = self.inverses[g]
            for n in N:
                conjugate = self.op(self.op(g, n), g_inv)
                if conjugate not in N:
                    return False
        return True

    def _compute_cosets(self) -> List[frozenset]:
        """Compute all left cosets of N."""
        cosets = []
        covered = set()

        for g in self.G:
            if g in covered:
                continue
            coset = frozenset(self.op(g, n) for n in self.N)
            cosets.append(coset)
            covered.update(coset)

        return cosets

    def _get_coset(self, g) -> frozenset:
        """Find the coset containing g."""
        for coset in self.cosets:
            if g in coset:
                return coset
        return None

    def _build_quotient_table(self):
        """Build multiplication table for G/N."""
        self.quotient_table = {}

        for coset_a in self.cosets:
            for coset_b in self.cosets:
                # Pick representatives
                a = next(iter(coset_a))
                b = next(iter(coset_b))

                # Product coset
                product_elem = self.op(a, b)
                product_coset = self._get_coset(product_elem)

                self.quotient_table[(coset_a, coset_b)] = product_coset

    def quotient_multiply(self, coset_a: frozenset, coset_b: frozenset) -> frozenset:
        """Multiply two cosets in G/N."""
        return self.quotient_table.get((coset_a, coset_b))

    def verify_well_defined(self) -> bool:
        """Verify coset multiplication is well-defined."""
        for coset_a in self.cosets:
            for coset_b in self.cosets:
                # Try all pairs of representatives
                products = set()
                for a in coset_a:
                    for b in coset_b:
                        product = self.op(a, b)
                        products.add(self._get_coset(product))

                if len(products) != 1:
                    return False
        return True

    def verify_quotient_is_group(self) -> Dict:
        """Verify G/N satisfies group axioms."""
        results = {}

        # Closure
        results['closure'] = all(
            self.quotient_multiply(a, b) in self.cosets
            for a in self.cosets for b in self.cosets
        )

        # Identity (coset containing identity)
        identity_coset = self._get_coset(self.identity)
        results['identity_coset'] = identity_coset
        results['identity'] = all(
            self.quotient_multiply(identity_coset, c) == c and
            self.quotient_multiply(c, identity_coset) == c
            for c in self.cosets
        )

        # Inverses
        results['inverses'] = True
        for coset in self.cosets:
            found_inverse = False
            for other in self.cosets:
                if self.quotient_multiply(coset, other) == identity_coset:
                    found_inverse = True
                    break
            if not found_inverse:
                results['inverses'] = False
                break

        return results

    def order(self) -> int:
        """Return |G/N|."""
        return len(self.cosets)

    def is_abelian(self) -> bool:
        """Check if G/N is abelian."""
        for a in self.cosets:
            for b in self.cosets:
                if self.quotient_multiply(a, b) != self.quotient_multiply(b, a):
                    return False
        return True

    def print_cosets(self):
        """Print all cosets."""
        print(f"G/N has {len(self.cosets)} cosets:")
        for i, coset in enumerate(self.cosets):
            print(f"  Coset {i}: {set(coset)}")

    def print_quotient_table(self):
        """Print multiplication table for G/N."""
        n = len(self.cosets)
        coset_labels = {coset: f"C{i}" for i, coset in enumerate(self.cosets)}

        # Header
        header = "    | " + " | ".join(coset_labels[c] for c in self.cosets)
        print(header)
        print("-" * len(header))

        # Rows
        for a in self.cosets:
            row = [coset_labels[self.quotient_multiply(a, b)] for b in self.cosets]
            print(f"{coset_labels[a]} | " + " | ".join(row))


def create_Zn(n: int) -> Tuple[List, Callable]:
    """Create cyclic group Z_n."""
    elements = list(range(n))
    operation = lambda a, b: (a + b) % n
    return elements, operation


def create_Dn(n: int) -> Tuple[List, Callable]:
    """Create dihedral group D_n."""
    elements = [(k, 0) for k in range(n)] + [(k, 1) for k in range(n)]

    def operation(g1, g2):
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

    return elements, operation


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("QUOTIENT GROUPS")
    print("=" * 60)

    # Example 1: Z_12 / <4>
    print("\n1. Z_12 / <4> (multiples of 4)")
    print("-" * 40)

    G, op = create_Zn(12)
    N = {0, 4, 8}  # <4> in Z_12

    Q = QuotientGroup(G, op, N)

    print(f"|G| = {len(G)}, |N| = {len(N)}, |G/N| = {Q.order()}")
    Q.print_cosets()

    verify = Q.verify_quotient_is_group()
    print(f"\nVerification:")
    print(f"  Closure: {verify['closure']}")
    print(f"  Identity: {verify['identity']}")
    print(f"  Inverses: {verify['inverses']}")
    print(f"  Abelian: {Q.is_abelian()}")
    print(f"\n  Z_12/<4> ≅ Z_3")

    # Example 2: D_4 / <r^2>
    print("\n2. D_4 / <r²> (rotation by 180°)")
    print("-" * 40)

    G, op = create_Dn(4)
    # <r^2> = {(0,0), (2,0)} in D_4
    N = {(0, 0), (2, 0)}

    Q = QuotientGroup(G, op, N)

    print(f"|D_4| = {len(G)}, |<r²>| = {len(N)}, |D_4/<r²>| = {Q.order()}")
    Q.print_cosets()
    print(f"\nAbelian: {Q.is_abelian()}")
    print(f"\nD_4/<r²> ≅ V_4 (Klein four-group)")

    Q.print_quotient_table()

    # Example 3: D_6 / <r>
    print("\n3. D_6 / <r> (all rotations)")
    print("-" * 40)

    G, op = create_Dn(6)
    # <r> = {(0,0), (1,0), (2,0), (3,0), (4,0), (5,0)}
    N = {(k, 0) for k in range(6)}

    Q = QuotientGroup(G, op, N)

    print(f"|D_6| = {len(G)}, |<r>| = {len(N)}, |D_6/<r>| = {Q.order()}")
    Q.print_cosets()
    print(f"\nD_6/<r> ≅ Z_2")

    # Example 4: Verify well-definedness
    print("\n4. WELL-DEFINEDNESS VERIFICATION")
    print("-" * 40)

    G, op = create_Zn(12)
    N = {0, 3, 6, 9}  # <3> in Z_12

    Q = QuotientGroup(G, op, N)
    print(f"Coset multiplication well-defined: {Q.verify_well_defined()}")
    print(f"|Z_12/<3>| = {Q.order()} ≅ Z_3")

    # Example 5: Computing with quotient
    print("\n5. QUOTIENT ARITHMETIC")
    print("-" * 40)

    G, op = create_Zn(15)
    N = {0, 5, 10}  # <5> in Z_15

    Q = QuotientGroup(G, op, N)
    print(f"Z_15/<5> has {Q.order()} elements")

    # Compute (2 + <5>) * (3 + <5>)
    coset_2 = Q._get_coset(2)
    coset_3 = Q._get_coset(3)
    product = Q.quotient_multiply(coset_2, coset_3)

    print(f"\nCoset containing 2: {set(coset_2)}")
    print(f"Coset containing 3: {set(coset_3)}")
    print(f"(2+<5>) * (3+<5>) = {set(product)}")
    print(f"(This is the coset containing 5 ≡ 0 mod 5, which is <5> itself!)")

    # Example 6: Non-abelian quotient
    print("\n6. QUOTIENT OF NON-ABELIAN GROUP")
    print("-" * 40)

    G, op = create_Dn(4)
    # Center of D_4 is {(0,0), (2,0)}
    Z = {(0, 0), (2, 0)}

    Q = QuotientGroup(G, op, Z)
    print(f"D_4/Z(D_4):")
    Q.print_cosets()
    print(f"\nAbelian: {Q.is_abelian()}")
    print("D_4/Z(D_4) ≅ Z_2 × Z_2 (abelian even though D_4 is not!)")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. G/N is only a group when N is NORMAL
    2. |G/N| = |G|/|N| (Lagrange's theorem)
    3. First Isomorphism Theorem: G/ker(φ) ≅ Im(φ)
    4. Quotients "collapse" N to identity
    5. In physics: gauge equivalence = quotient by gauge group
    6. Projective space = quotient by phase transformations
    """)
```

---

## 8. Practice Problems

### Problem Set A: Basic Computations

**A1.** Compute $\mathbb{Z}_{12}/\langle 4 \rangle$ and identify the quotient group.

**A2.** Find all elements of $(\mathbb{Z} \times \mathbb{Z})/\langle(2, 0)\rangle$.

**A3.** Show that $D_4/\{e, r^2\} \cong \mathbb{Z}_2 \times \mathbb{Z}_2$.

### Problem Set B: Isomorphism Theorems

**B1.** Use the First Isomorphism Theorem to show $\mathbb{R}/\mathbb{Z} \cong S^1$ (the circle group).

**B2.** Prove: If $G$ is abelian and $n = |G|$, then $G/H \cong G/K$ implies $H \cong K$.

**B3.** Apply the Third Isomorphism Theorem to $\mathbb{Z}/6\mathbb{Z}$, $\mathbb{Z}/2\mathbb{Z}$, $\mathbb{Z}/6\mathbb{Z}/\mathbb{Z}/2\mathbb{Z}$.

### Problem Set C: Advanced

**C1.** Prove: $G/Z(G)$ is cyclic implies $G$ is abelian.

**C2.** For finite group $G$, show: $G$ is simple iff every non-trivial homomorphism from $G$ is injective.

**C3.** **(Physics)** In electromagnetism, the gauge group is $U(1)$. Explain why gauge-inequivalent configurations form the quotient $\mathcal{A}/U(1)$ where $\mathcal{A}$ is the space of vector potentials.

---

## 9. Summary

### Key Construction

$$\boxed{G/N = \{gN : g \in G\} \text{ with } (aN)(bN) = (ab)N}$$

### Key Properties

| Property | Formula |
|----------|---------|
| Order | $\|G/N\| = \|G\|/\|N\|$ |
| Identity | $N$ |
| Inverse | $(gN)^{-1} = g^{-1}N$ |
| Well-defined | Only when $N \trianglelefteq G$ |

### Key Theorems

$$\boxed{G/\ker(\phi) \cong \text{Im}(\phi)}$$

$$\boxed{(G/N)/(K/N) \cong G/K}$$

---

## 10. Preview: Day 285

Tomorrow we study **cyclic and abelian groups**:
- Structure theorem for finitely generated abelian groups
- Cyclic groups and generators
- Direct products and sums
- The Chinese Remainder Theorem for groups

---

*"Taking quotients is like putting on blinders—you see less, but what you see is clearer." — I.N. Herstein*
