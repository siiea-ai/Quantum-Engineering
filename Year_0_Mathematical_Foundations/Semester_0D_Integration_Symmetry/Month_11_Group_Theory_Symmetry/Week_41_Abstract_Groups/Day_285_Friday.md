# Day 285: Cyclic and Abelian Groups — Complete Classification

## Overview

**Month 11, Week 41, Day 5 — Friday**

Today we achieve one of the great classification results in algebra: the **structure theorem for finitely generated abelian groups**. Every such group decomposes uniquely (up to isomorphism) into cyclic components. We'll also explore cyclic groups in depth—they are the building blocks of all abelian groups and play a central role in quantum mechanics through phase rotations and the U(1) gauge group.

## Prerequisites

From Days 281-284:
- Group axioms and subgroups
- Homomorphisms and isomorphisms
- Quotient groups

## Learning Objectives

By the end of today, you will be able to:

1. Classify cyclic groups and find all their subgroups
2. Determine generators of cyclic groups
3. State and apply the Structure Theorem for finitely generated abelian groups
4. Decompose abelian groups into primary components
5. Apply the Chinese Remainder Theorem to group decomposition
6. Connect abelian groups to quantum phase symmetry

---

## 1. Cyclic Groups

### Definition and Basic Properties

**Definition:** A group $G$ is **cyclic** if there exists $g \in G$ such that:
$$G = \langle g \rangle = \{g^n : n \in \mathbb{Z}\}$$

The element $g$ is called a **generator** of $G$.

### Finite vs Infinite Cyclic Groups

**Theorem:** Every cyclic group is isomorphic to either:
- $\mathbb{Z}$ (infinite cyclic group), or
- $\mathbb{Z}_n$ for some $n \geq 1$ (finite cyclic group of order $n$)

*Proof:*
Let $G = \langle g \rangle$.

**Case 1:** All powers $g^n$ are distinct. Define $\phi: \mathbb{Z} \to G$ by $\phi(n) = g^n$. This is a bijective homomorphism, so $G \cong \mathbb{Z}$.

**Case 2:** Some powers coincide: $g^a = g^b$ for $a < b$. Then $g^{b-a} = e$. Let $n$ be the smallest positive integer with $g^n = e$. Then:
$$G = \{e, g, g^2, \ldots, g^{n-1}\}$$
and $\phi: \mathbb{Z}_n \to G$ by $\phi(k) = g^k$ is an isomorphism. ∎

### Subgroups of Cyclic Groups

**Theorem:** Every subgroup of a cyclic group is cyclic.

*Proof:*
Let $G = \langle g \rangle$ and $H \leq G$. If $H = \{e\}$, it's cyclic.

Otherwise, let $m$ be the smallest positive integer with $g^m \in H$. We claim $H = \langle g^m \rangle$.

Clearly $\langle g^m \rangle \subseteq H$. For the reverse, take $g^k \in H$. Divide: $k = qm + r$ with $0 \leq r < m$.
Then $g^r = g^k \cdot (g^m)^{-q} \in H$.

By minimality of $m$, we have $r = 0$, so $k = qm$ and $g^k = (g^m)^q \in \langle g^m \rangle$. ∎

**Theorem:** The subgroups of $\mathbb{Z}_n$ are exactly $\langle d \rangle$ for each divisor $d$ of $n$, and:
$$|\langle d \rangle| = n/d$$

**Corollary:** $\mathbb{Z}_n$ has exactly one subgroup of order $d$ for each divisor $d$ of $n$.

### Generators of $\mathbb{Z}_n$

**Theorem:** $k$ is a generator of $\mathbb{Z}_n$ if and only if $\gcd(k, n) = 1$.

*Proof:*
$\langle k \rangle = \mathbb{Z}_n \Leftrightarrow |\langle k \rangle| = n \Leftrightarrow |k| = n$.

The order of $k$ in $\mathbb{Z}_n$ is $n/\gcd(k, n)$.

So $|k| = n \Leftrightarrow \gcd(k, n) = 1$. ∎

**Corollary:** The number of generators of $\mathbb{Z}_n$ is $\phi(n)$ (Euler's totient function).

---

## 2. Direct Products

### Definition

**Definition:** The **direct product** of groups $G$ and $H$ is:
$$G \times H = \{(g, h) : g \in G, h \in H\}$$

with operation $(g_1, h_1) \cdot (g_2, h_2) = (g_1 g_2, h_1 h_2)$.

### Properties

1. $|G \times H| = |G| \cdot |H|$
2. $G \times H$ is abelian iff both $G$ and $H$ are abelian
3. $G \times H \cong H \times G$
4. $(G \times H) \times K \cong G \times (H \times K)$

### When Is a Direct Product Cyclic?

**Theorem:** $\mathbb{Z}_m \times \mathbb{Z}_n$ is cyclic if and only if $\gcd(m, n) = 1$.

When this holds: $\mathbb{Z}_m \times \mathbb{Z}_n \cong \mathbb{Z}_{mn}$

*Example:* $\mathbb{Z}_2 \times \mathbb{Z}_3 \cong \mathbb{Z}_6$ but $\mathbb{Z}_2 \times \mathbb{Z}_2 \not\cong \mathbb{Z}_4$.

---

## 3. The Chinese Remainder Theorem

### For Integers

**Theorem (CRT):** If $\gcd(m, n) = 1$, then:
$$\mathbb{Z}_{mn} \cong \mathbb{Z}_m \times \mathbb{Z}_n$$

The isomorphism is $\phi(k) = (k \mod m, k \mod n)$.

### General Form

**Theorem:** If $n = p_1^{a_1} p_2^{a_2} \cdots p_k^{a_k}$ (prime factorization), then:
$$\boxed{\mathbb{Z}_n \cong \mathbb{Z}_{p_1^{a_1}} \times \mathbb{Z}_{p_2^{a_2}} \times \cdots \times \mathbb{Z}_{p_k^{a_k}}}$$

*Example:* $\mathbb{Z}_{12} \cong \mathbb{Z}_4 \times \mathbb{Z}_3$ (since $12 = 4 \cdot 3$ and $\gcd(4,3) = 1$)

### Application: Counting Elements of Each Order

To count elements of order $d$ in $\mathbb{Z}_n$:
1. Factor $n$ and apply CRT
2. Use: order in product = lcm of orders in factors
3. Count systematically

---

## 4. Structure Theorem for Finitely Generated Abelian Groups

### The Fundamental Theorem

**Theorem (Structure Theorem):** Every finitely generated abelian group is isomorphic to:

$$\boxed{G \cong \mathbb{Z}^r \times \mathbb{Z}_{n_1} \times \mathbb{Z}_{n_2} \times \cdots \times \mathbb{Z}_{n_k}}$$

where:
- $r \geq 0$ is the **rank** (number of infinite cyclic factors)
- $n_1 | n_2 | \cdots | n_k$ (each divides the next)
- This decomposition is unique

The $n_i$ are called **invariant factors**.

### Primary Decomposition

**Alternative Form:** Every finite abelian group is isomorphic to:

$$G \cong \mathbb{Z}_{p_1^{a_1}} \times \mathbb{Z}_{p_2^{a_2}} \times \cdots \times \mathbb{Z}_{p_m^{a_m}}$$

where the $p_i^{a_i}$ are prime powers (not necessarily distinct primes).

These are called **elementary divisors**.

### Converting Between Forms

**From invariant factors to elementary divisors:**
Factor each $n_i$ into prime powers.

**From elementary divisors to invariant factors:**
Group by prime, take largest power first.

*Example:* $G \cong \mathbb{Z}_{2} \times \mathbb{Z}_{4} \times \mathbb{Z}_{3} \times \mathbb{Z}_{9}$

Elementary divisors: $2, 4, 3, 9$

Invariant factors:
- Largest from each prime: $4, 9$ → $\text{lcm} = 36$
- Remaining: $2, 3$ → $\text{lcm} = 6$

So $G \cong \mathbb{Z}_6 \times \mathbb{Z}_{36}$ (and $6 | 36$ ✓)

---

## 5. Classifying Finite Abelian Groups

### By Order

**Question:** How many abelian groups of order $n$ are there (up to isomorphism)?

**Answer:** Factor $n = p_1^{a_1} \cdots p_k^{a_k}$. The number of abelian groups is:
$$\prod_{i=1}^{k} p(a_i)$$

where $p(a)$ = number of partitions of $a$.

### Examples

**Order 4:** $4 = 2^2$, partitions of 2: $\{2\}, \{1,1\}$
- $\mathbb{Z}_4$ (from partition $\{2\}$)
- $\mathbb{Z}_2 \times \mathbb{Z}_2$ (from partition $\{1,1\}$)

**Order 8:** $8 = 2^3$, partitions of 3: $\{3\}, \{2,1\}, \{1,1,1\}$
- $\mathbb{Z}_8$
- $\mathbb{Z}_4 \times \mathbb{Z}_2$
- $\mathbb{Z}_2 \times \mathbb{Z}_2 \times \mathbb{Z}_2$

**Order 12:** $12 = 2^2 \cdot 3$, partitions: $p(2) \cdot p(1) = 2 \cdot 1 = 2$
- $\mathbb{Z}_{12} \cong \mathbb{Z}_4 \times \mathbb{Z}_3$
- $\mathbb{Z}_2 \times \mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_2 \times \mathbb{Z}_3$

---

## 6. Quantum Mechanics Connection

### U(1) Phase Symmetry

The group $U(1) = \{e^{i\theta} : \theta \in [0, 2\pi)\}$ is the circle group—an infinite abelian group.

In quantum mechanics:
- Global phase transformation: $|\psi\rangle \mapsto e^{i\theta}|\psi\rangle$
- Physically equivalent states differ by phase
- Projective Hilbert space: $\mathbb{P}\mathcal{H} = \mathcal{H}/U(1)$

### Finite Cyclic Symmetries

Discrete rotational symmetry $C_n$:
- Molecule with $n$-fold rotational symmetry
- Quantum states transform under $\mathbb{Z}_n$ representations
- Characters: $\chi_k(r) = e^{2\pi i k/n}$ for $k = 0, 1, \ldots, n-1$

### Abelian Gauge Theory

In electromagnetism:
- Gauge group: $U(1)$
- Charge quantization related to $\mathbb{Z}$ subgroup
- Magnetic monopoles ↔ topology of $U(1)$ bundles

### Fourier Transform and Characters

For abelian group $G$, the **character group** $\hat{G}$ consists of all homomorphisms $G \to U(1)$.

**Theorem:** For finite abelian $G$: $\hat{G} \cong G$.

The Fourier transform on $G$ decomposes functions into character components—this is exactly quantum state decomposition into energy eigenstates for abelian symmetry groups!

---

## 7. Worked Examples

### Example 1: Classify All Abelian Groups of Order 36

$36 = 2^2 \cdot 3^2$

Partitions of 2: $\{2\}, \{1,1\}$

For each combination:
1. $\mathbb{Z}_4 \times \mathbb{Z}_9 \cong \mathbb{Z}_{36}$ (invariant factors: 36)
2. $\mathbb{Z}_4 \times \mathbb{Z}_3 \times \mathbb{Z}_3$ (invariant factors: 3, 12)
3. $\mathbb{Z}_2 \times \mathbb{Z}_2 \times \mathbb{Z}_9$ (invariant factors: 2, 18)
4. $\mathbb{Z}_2 \times \mathbb{Z}_2 \times \mathbb{Z}_3 \times \mathbb{Z}_3$ (invariant factors: 6, 6)

Total: $2 \times 2 = 4$ abelian groups of order 36.

### Example 2: Find All Subgroups of $\mathbb{Z}_{24}$

Divisors of 24: 1, 2, 3, 4, 6, 8, 12, 24

Subgroups:
- $\langle 24 \rangle = \{0\}$ (order 1)
- $\langle 12 \rangle = \{0, 12\}$ (order 2)
- $\langle 8 \rangle = \{0, 8, 16\}$ (order 3)
- $\langle 6 \rangle = \{0, 6, 12, 18\}$ (order 4)
- $\langle 4 \rangle = \{0, 4, 8, 12, 16, 20\}$ (order 6)
- $\langle 3 \rangle = \{0, 3, 6, 9, 12, 15, 18, 21\}$ (order 8)
- $\langle 2 \rangle = \{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22\}$ (order 12)
- $\langle 1 \rangle = \mathbb{Z}_{24}$ (order 24)

Total: 8 subgroups (one for each divisor).

### Example 3: Decompose $(\mathbb{Z}/180\mathbb{Z})^*$

$180 = 4 \cdot 9 \cdot 5 = 2^2 \cdot 3^2 \cdot 5$

By CRT: $(\mathbb{Z}/180\mathbb{Z})^* \cong (\mathbb{Z}/4\mathbb{Z})^* \times (\mathbb{Z}/9\mathbb{Z})^* \times (\mathbb{Z}/5\mathbb{Z})^*$

We have:
- $(\mathbb{Z}/4\mathbb{Z})^* = \{1, 3\} \cong \mathbb{Z}_2$
- $(\mathbb{Z}/9\mathbb{Z})^* \cong \mathbb{Z}_6$ (cyclic, order $\phi(9) = 6$)
- $(\mathbb{Z}/5\mathbb{Z})^* \cong \mathbb{Z}_4$ (cyclic, order $\phi(5) = 4$)

So $(\mathbb{Z}/180\mathbb{Z})^* \cong \mathbb{Z}_2 \times \mathbb{Z}_4 \times \mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_4 \times \mathbb{Z}_2 \times \mathbb{Z}_3$

In invariant factor form: $\mathbb{Z}_2 \times \mathbb{Z}_{12} \times \mathbb{Z}_2$ ... needs more careful analysis.

Actually: $\mathbb{Z}_2 \times \mathbb{Z}_4 \times \mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_2 \times \mathbb{Z}_{12}$ (invariant factors: 2, 2, 12 doesn't satisfy divisibility)

Correct form: Elementary divisors $2, 2, 2, 3, 4$.
Invariant factors: pair up largest → $\text{lcm}(4,3) = 12$, $\text{lcm}(2,2) = 2$, remaining $2$.
So: $\mathbb{Z}_2 \times \mathbb{Z}_2 \times \mathbb{Z}_{12}$ but $2 \nmid 2$ for divisibility...

Let me recalculate: $\mathbb{Z}_2 \times \mathbb{Z}_4 \times \mathbb{Z}_6$
- Elementary: $2, 4, 2, 3$
- Group by prime: twos are $2, 4, 2$; threes are $3$
- Largest combo: $4, 3 \to 12$
- Next: $2$
- Next: $2$
- Invariant factors: $2 | 2 | 12$? No, need $2|2|12$...

$(\mathbb{Z}/180\mathbb{Z})^* \cong \mathbb{Z}_2 \times \mathbb{Z}_2 \times \mathbb{Z}_{12}$? Check order: $2 \times 2 \times 12 = 48 = \phi(180)$ ✓

---

## 8. Computational Lab

```python
"""
Day 285: Cyclic and Abelian Groups
Classification and decomposition algorithms
"""

import numpy as np
from math import gcd, lcm
from functools import reduce
from typing import List, Tuple, Dict
from collections import Counter

def euler_phi(n: int) -> int:
    """Compute Euler's totient function φ(n)."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result

def prime_factorization(n: int) -> Dict[int, int]:
    """Return prime factorization as {prime: power}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def divisors(n: int) -> List[int]:
    """Return all divisors of n in sorted order."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

def partitions(n: int) -> List[List[int]]:
    """Generate all partitions of n."""
    if n == 0:
        return [[]]
    if n < 0:
        return []

    result = []

    def helper(remaining, max_val, current):
        if remaining == 0:
            result.append(current[:])
            return
        for i in range(min(remaining, max_val), 0, -1):
            current.append(i)
            helper(remaining - i, i, current)
            current.pop()

    helper(n, n, [])
    return result

def count_abelian_groups(n: int) -> int:
    """Count number of abelian groups of order n (up to isomorphism)."""
    factors = prime_factorization(n)
    count = 1
    for p, a in factors.items():
        count *= len(partitions(a))
    return count

def list_abelian_groups(n: int) -> List[List[int]]:
    """
    List all abelian groups of order n as lists of cyclic group orders.

    Returns lists of elementary divisors.
    """
    factors = prime_factorization(n)

    # Get all partitions for each prime power
    prime_parts = {}
    for p, a in factors.items():
        prime_parts[p] = partitions(a)

    # Generate all combinations
    groups = []

    def generate(primes, current):
        if not primes:
            groups.append(sorted(current))
            return

        p = primes[0]
        for partition in prime_parts[p]:
            new_current = current + [p**k for k in partition]
            generate(primes[1:], new_current)

    generate(list(factors.keys()), [])
    return groups

def elementary_to_invariant(elementary: List[int]) -> List[int]:
    """Convert elementary divisors to invariant factors."""
    if not elementary:
        return []

    # Group by prime
    by_prime = {}
    for d in elementary:
        factors = prime_factorization(d)
        for p, a in factors.items():
            if p not in by_prime:
                by_prime[p] = []
            by_prime[p].append(p**a)

    # Sort each list in decreasing order
    for p in by_prime:
        by_prime[p].sort(reverse=True)

    # Find max length
    max_len = max(len(v) for v in by_prime.values())

    # Pad with 1s
    for p in by_prime:
        while len(by_prime[p]) < max_len:
            by_prime[p].append(1)

    # Compute invariant factors
    invariants = []
    for i in range(max_len):
        inv = 1
        for p in by_prime:
            inv *= by_prime[p][i]
        invariants.append(inv)

    # Remove trailing 1s and reverse
    while invariants and invariants[-1] == 1:
        invariants.pop()

    return invariants[::-1]  # Increasing order

def generators_of_Zn(n: int) -> List[int]:
    """Find all generators of Z_n."""
    return [k for k in range(1, n) if gcd(k, n) == 1]

def subgroups_of_Zn(n: int) -> Dict[int, List[int]]:
    """
    Find all subgroups of Z_n.

    Returns dict mapping divisor d to subgroup generated by n/d.
    """
    subgroups = {}
    for d in divisors(n):
        gen = n // d  # generator of subgroup of order d
        subgroup = [(gen * k) % n for k in range(d)]
        subgroups[d] = sorted(subgroup)
    return subgroups

def order_in_Zn(k: int, n: int) -> int:
    """Compute order of k in Z_n."""
    return n // gcd(k, n)

def element_orders_in_product(factors: List[int]) -> Dict[int, int]:
    """
    Count elements of each order in Z_{n1} × Z_{n2} × ...

    Parameters:
        factors: list of cyclic group orders

    Returns:
        dict mapping order to count
    """
    # For direct product, order = lcm of component orders
    # Element (a1, a2, ...) has order lcm(|a1|, |a2|, ...)

    orders = Counter()

    # Generate all elements
    ranges = [range(n) for n in factors]

    def order_of_element(elem):
        component_orders = [order_in_Zn(elem[i], factors[i]) for i in range(len(factors))]
        return reduce(lcm, component_orders)

    from itertools import product
    for elem in product(*ranges):
        ord = order_of_element(elem)
        orders[ord] += 1

    return dict(orders)

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("CYCLIC AND ABELIAN GROUPS")
    print("=" * 60)

    # Example 1: Generators of Z_n
    print("\n1. GENERATORS OF CYCLIC GROUPS")
    print("-" * 40)

    for n in [6, 8, 12, 15]:
        gens = generators_of_Zn(n)
        print(f"Generators of Z_{n}: {gens}")
        print(f"  Count: φ({n}) = {euler_phi(n)}")

    # Example 2: Subgroups of Z_24
    print("\n2. SUBGROUPS OF Z_24")
    print("-" * 40)

    subs = subgroups_of_Zn(24)
    for order in sorted(subs.keys()):
        print(f"Order {order:2d}: {subs[order]}")

    print(f"\nTotal subgroups: {len(subs)} (one per divisor of 24)")

    # Example 3: Classification of abelian groups
    print("\n3. ABELIAN GROUPS BY ORDER")
    print("-" * 40)

    for n in [4, 8, 12, 16, 36]:
        count = count_abelian_groups(n)
        groups = list_abelian_groups(n)
        print(f"\nOrder {n}: {count} abelian groups")
        for g in groups:
            inv = elementary_to_invariant(g)
            prod_str = " × ".join(f"Z_{d}" for d in g)
            inv_str = " × ".join(f"Z_{d}" for d in inv) if inv else "trivial"
            print(f"  Elementary: {prod_str}")
            print(f"  Invariant:  {inv_str}")

    # Example 4: Element orders in Z_12
    print("\n4. ELEMENT ORDERS IN Z_12")
    print("-" * 40)

    for k in range(12):
        ord = order_in_Zn(k, 12)
        print(f"  |{k:2d}| = {ord}")

    # Example 5: Element orders in product group
    print("\n5. ELEMENT ORDERS IN Z_2 × Z_4")
    print("-" * 40)

    orders = element_orders_in_product([2, 4])
    print(f"Z_2 × Z_4 has {2*4} = 8 elements")
    for ord in sorted(orders.keys()):
        print(f"  {orders[ord]} elements of order {ord}")

    print("\nCompare to Z_8:")
    orders_Z8 = element_orders_in_product([8])
    for ord in sorted(orders_Z8.keys()):
        print(f"  {orders_Z8[ord]} elements of order {ord}")

    print("\nZ_2 × Z_4 has no element of order 8, so Z_2 × Z_4 ≇ Z_8")

    # Example 6: Chinese Remainder Theorem
    print("\n6. CHINESE REMAINDER THEOREM")
    print("-" * 40)

    print("Z_12 ≅ Z_3 × Z_4 (since gcd(3,4) = 1)")
    print("\nIsomorphism: k ↦ (k mod 3, k mod 4)")
    for k in range(12):
        print(f"  {k:2d} ↦ ({k % 3}, {k % 4})")

    # Example 7: Structure theorem application
    print("\n7. STRUCTURE THEOREM APPLICATION")
    print("-" * 40)

    # Classify groups of order 72 = 8 × 9 = 2³ × 3²
    n = 72
    print(f"Abelian groups of order {n} = 2³ × 3²:")
    print(f"Partitions of 3: {partitions(3)}")
    print(f"Partitions of 2: {partitions(2)}")

    groups = list_abelian_groups(n)
    for i, g in enumerate(groups, 1):
        inv = elementary_to_invariant(g)
        elem_str = " × ".join(f"Z_{d}" for d in sorted(g, reverse=True))
        inv_str = " × ".join(f"Z_{d}" for d in inv)
        print(f"\n  Group {i}:")
        print(f"    Elementary divisors: {elem_str}")
        print(f"    Invariant factors: {inv_str}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Every cyclic group ≅ Z or Z_n
    2. Subgroups of cyclic groups are cyclic
    3. Z_n has exactly one subgroup of each order dividing n
    4. Structure Theorem: Every f.g. abelian group = Z^r × Z_{n₁} × ... × Z_{nₖ}
    5. Chinese Remainder: Z_{mn} ≅ Z_m × Z_n when gcd(m,n) = 1
    6. In QM: U(1) phase symmetry is central to gauge theory
    """)
```

---

## 9. Practice Problems

### Problem Set A: Cyclic Groups

**A1.** Find all generators of $\mathbb{Z}_{18}$. How many are there?

**A2.** Find all subgroups of $\mathbb{Z}_{30}$ and draw the subgroup lattice.

**A3.** Prove: $\mathbb{Z}_m \times \mathbb{Z}_n$ is cyclic iff $\gcd(m,n) = 1$.

### Problem Set B: Structure Theorem

**B1.** List all abelian groups of order 100 (up to isomorphism).

**B2.** Find the invariant factors of $\mathbb{Z}_2 \times \mathbb{Z}_4 \times \mathbb{Z}_8$.

**B3.** How many elements of order 6 are in $\mathbb{Z}_2 \times \mathbb{Z}_3 \times \mathbb{Z}_6$?

### Problem Set C: Advanced

**C1.** Prove: If $G$ is a finite abelian group and $d | |G|$, then $G$ has a subgroup of order $d$.

**C2.** Characterize all abelian groups in which every element has order dividing 12.

**C3.** **(Physics)** The gauge group of QED is $U(1) \cong S^1$. Explain why electric charge is quantized in terms of the group structure.

---

## 10. Summary

### Key Theorems

$$\boxed{\text{Cyclic groups: } G = \langle g \rangle \Rightarrow G \cong \mathbb{Z} \text{ or } \mathbb{Z}_n}$$

$$\boxed{\text{CRT: } \gcd(m,n) = 1 \Rightarrow \mathbb{Z}_{mn} \cong \mathbb{Z}_m \times \mathbb{Z}_n}$$

$$\boxed{\text{Structure: } G \cong \mathbb{Z}^r \times \mathbb{Z}_{n_1} \times \cdots \times \mathbb{Z}_{n_k}, \quad n_1 | n_2 | \cdots | n_k}$$

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $\phi(n)$ | Number of generators of $\mathbb{Z}_n$ |
| $n/\gcd(k,n)$ | Order of $k$ in $\mathbb{Z}_n$ |
| $\prod p(a_i)$ | Number of abelian groups of order $\prod p_i^{a_i}$ |

---

## 11. Preview: Day 286

Tomorrow we study **permutation groups**:
- The symmetric group $S_n$
- Cycle notation and cycle decomposition
- The alternating group $A_n$
- Permutation groups and identical particles in QM

---

*"The integers are the hydrogen atom of mathematics—simple, fundamental, and revealing deep structure upon close examination." — P.G. Lejeune Dirichlet*
