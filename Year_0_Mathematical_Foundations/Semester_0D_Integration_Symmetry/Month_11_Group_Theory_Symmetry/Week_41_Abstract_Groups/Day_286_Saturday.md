# Day 286: Permutation Groups — The Symmetric Group

## Overview

**Month 11, Week 41, Day 6 — Saturday**

Today we study **permutation groups**, especially the symmetric group $S_n$. Permutations are central to mathematics and physics—they describe arrangements, symmetries, and importantly in quantum mechanics, the exchange of identical particles. The symmetric group is the "universal" finite group in the sense that every finite group is isomorphic to a subgroup of some $S_n$ (Cayley's theorem).

## Prerequisites

From Days 281-285:
- Group axioms and subgroups
- Cyclic groups and their structure
- Normal subgroups and quotients

## Learning Objectives

By the end of today, you will be able to:

1. Write permutations in cycle notation
2. Compute products of permutations and cycles
3. Determine the order of a permutation from its cycle structure
4. Classify permutations as even or odd
5. Work with the alternating group $A_n$
6. Connect permutation groups to quantum particle exchange

---

## 1. The Symmetric Group

### Definition

**Definition:** The **symmetric group** $S_n$ is the group of all bijections $\sigma: \{1, 2, \ldots, n\} \to \{1, 2, \ldots, n\}$ under composition.

$$|S_n| = n!$$

### Two-Line Notation

A permutation $\sigma$ can be written as:
$$\sigma = \begin{pmatrix} 1 & 2 & 3 & \cdots & n \\ \sigma(1) & \sigma(2) & \sigma(3) & \cdots & \sigma(n) \end{pmatrix}$$

**Example:** In $S_4$:
$$\sigma = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 1 & 3 \end{pmatrix}$$

means $\sigma(1) = 2$, $\sigma(2) = 4$, $\sigma(3) = 1$, $\sigma(4) = 3$.

### Composition

Permutations compose right-to-left: $(\sigma \tau)(x) = \sigma(\tau(x))$.

**Example:** If $\sigma = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 3 & 1 \end{pmatrix}$ and $\tau = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 1 & 3 \end{pmatrix}$, then:

$$\sigma \tau(1) = \sigma(2) = 3, \quad \sigma \tau(2) = \sigma(1) = 2, \quad \sigma \tau(3) = \sigma(3) = 1$$

So $\sigma \tau = \begin{pmatrix} 1 & 2 & 3 \\ 3 & 2 & 1 \end{pmatrix}$.

---

## 2. Cycle Notation

### Cycles

**Definition:** A **cycle** $(a_1 \, a_2 \, \cdots \, a_k)$ is the permutation:
$$a_1 \mapsto a_2 \mapsto a_3 \mapsto \cdots \mapsto a_k \mapsto a_1$$

with all other elements fixed.

A cycle of length $k$ is called a **$k$-cycle**.
- 2-cycle = **transposition**
- 1-cycle = identity (often omitted)

**Example:** $(1 \, 3 \, 4)$ in $S_5$:
- $1 \mapsto 3$
- $3 \mapsto 4$
- $4 \mapsto 1$
- $2 \mapsto 2$, $5 \mapsto 5$ (fixed)

### Disjoint Cycle Decomposition

**Theorem:** Every permutation can be written uniquely (up to order) as a product of disjoint cycles.

**Algorithm:**
1. Start with smallest unmoved element
2. Follow its orbit until returning
3. Write down the cycle
4. Repeat with next unmoved element

**Example:** $\sigma = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 & 6 \\ 3 & 5 & 1 & 6 & 2 & 4 \end{pmatrix}$

- Start at 1: $1 \to 3 \to 1$, gives $(1 \, 3)$
- Next unmoved: 2: $2 \to 5 \to 2$, gives $(2 \, 5)$
- Next: 4: $4 \to 6 \to 4$, gives $(4 \, 6)$

So $\sigma = (1 \, 3)(2 \, 5)(4 \, 6)$.

### Cycle Type

The **cycle type** of a permutation is the tuple of cycle lengths in non-increasing order.

**Example:** $(1 \, 2 \, 3)(4 \, 5)$ has cycle type $(3, 2)$ or $3 + 2$.

Permutations in $S_n$ are conjugate iff they have the same cycle type.

---

## 3. Properties of Permutations

### Order of a Permutation

**Theorem:** The order of a permutation is the lcm of its cycle lengths.

$$\boxed{|\sigma| = \text{lcm}(\text{cycle lengths})}$$

*Proof:* Disjoint cycles commute, so $\sigma^k = e$ iff each cycle raised to $k$ is identity, which happens when $k$ is divisible by all cycle lengths. ∎

**Example:** $\sigma = (1 \, 2 \, 3)(4 \, 5)$ has order $\text{lcm}(3, 2) = 6$.

### Transposition Decomposition

**Theorem:** Every permutation can be written as a product of transpositions.

*Method:* $(a_1 \, a_2 \, \cdots \, a_k) = (a_1 \, a_k)(a_1 \, a_{k-1}) \cdots (a_1 \, a_2)$

**Example:** $(1 \, 2 \, 3 \, 4) = (1 \, 4)(1 \, 3)(1 \, 2)$

Check: $(1 \, 2)(1 \, 3)(1 \, 4)$:
- Apply $(1 \, 4)$: $1 \to 4$
- Apply $(1 \, 3)$: $1 \to 3$ (but 1 is now at position 4)... Let me recalculate.

Actually, right-to-left: $(1 \, 4)(1 \, 3)(1 \, 2)$:
- First $(1 \, 2)$: $1 \to 2$, $2 \to 1$
- Then $(1 \, 3)$: $1 \to 3$, $3 \to 1$
- Then $(1 \, 4)$: $1 \to 4$, $4 \to 1$

Track 1: $1 \xrightarrow{(1\,2)} 2 \xrightarrow{(1\,3)} 2 \xrightarrow{(1\,4)} 2$
Track 2: $2 \xrightarrow{(1\,2)} 1 \xrightarrow{(1\,3)} 3 \xrightarrow{(1\,4)} 3$
Track 3: $3 \xrightarrow{(1\,2)} 3 \xrightarrow{(1\,3)} 1 \xrightarrow{(1\,4)} 4$
Track 4: $4 \xrightarrow{(1\,2)} 4 \xrightarrow{(1\,3)} 4 \xrightarrow{(1\,4)} 1$

Result: $1 \to 2, 2 \to 3, 3 \to 4, 4 \to 1$. This is $(1 \, 2 \, 3 \, 4)$ ✓

### Sign of a Permutation

**Definition:** The **sign** (or parity) of a permutation $\sigma$ is:
$$\text{sgn}(\sigma) = (-1)^{\text{number of transpositions}}$$

**Theorem:** The sign is well-defined (independent of transposition decomposition).

**Theorem:** $\text{sgn}: S_n \to \{1, -1\}$ is a homomorphism.

**Formulas for sign:**
- $k$-cycle has sign $(-1)^{k-1}$
- Product: $\text{sgn}(\sigma \tau) = \text{sgn}(\sigma) \cdot \text{sgn}(\tau)$

**Example:** $(1 \, 2 \, 3) = (1 \, 3)(1 \, 2)$, two transpositions, so sign = $+1$ (even).

---

## 4. The Alternating Group

### Definition

**Definition:** The **alternating group** $A_n$ is the kernel of the sign homomorphism:
$$A_n = \ker(\text{sgn}) = \{\sigma \in S_n : \text{sgn}(\sigma) = 1\}$$

$A_n$ consists of all **even permutations**.

### Properties

$$|A_n| = \frac{n!}{2}$$

**Theorem:** $A_n \trianglelefteq S_n$ (it's the kernel of a homomorphism).

**Theorem:** $S_n / A_n \cong \mathbb{Z}_2$.

### Generators of $A_n$

**Theorem:** $A_n$ is generated by 3-cycles.

*Proof:* Every even permutation is a product of an even number of transpositions. But $(a \, b)(c \, d) = (a \, b \, c)(b \, c \, d)$ if all distinct, and $(a \, b)(a \, c) = (a \, c \, b)$. So products of pairs of transpositions are products of 3-cycles. ∎

### Simplicity of $A_n$

**Theorem:** $A_n$ is **simple** (has no proper normal subgroups) for $n \geq 5$.

This is one of the most important results in finite group theory!

---

## 5. Cayley's Theorem

**Theorem (Cayley):** Every group $G$ is isomorphic to a subgroup of $S_G$ (the symmetric group on $G$).

For finite $G$ with $|G| = n$: $G$ embeds in $S_n$.

*Proof:*
For each $g \in G$, define $\lambda_g: G \to G$ by $\lambda_g(x) = gx$ (left multiplication).

This is a bijection (with inverse $\lambda_{g^{-1}}$), so $\lambda_g \in S_G$.

The map $\phi: G \to S_G$, $\phi(g) = \lambda_g$ is an injective homomorphism:
- Homomorphism: $\lambda_{gh}(x) = (gh)x = g(hx) = \lambda_g(\lambda_h(x))$
- Injective: $\lambda_g = \lambda_h \Rightarrow ge = he \Rightarrow g = h$ ∎

---

## 6. Quantum Mechanics Connection

### Identical Particles

In quantum mechanics, identical particles are truly **indistinguishable**. When we exchange two particles, the wave function transforms:
$$\psi(x_1, x_2) \xrightarrow{\text{exchange}} \psi(x_2, x_1) = \pm \psi(x_1, x_2)$$

### Bosons and Fermions

**Bosons:** Symmetric wave functions ($+$ sign)
$$\psi(x_{\sigma(1)}, \ldots, x_{\sigma(n)}) = \psi(x_1, \ldots, x_n)$$

**Fermions:** Antisymmetric wave functions ($-$ sign for transpositions)
$$\psi(x_{\sigma(1)}, \ldots, x_{\sigma(n)}) = \text{sgn}(\sigma) \cdot \psi(x_1, \ldots, x_n)$$

### Pauli Exclusion Principle

For fermions, the antisymmetry implies:
$$\psi(x_1, x_2, \ldots) = -\psi(x_2, x_1, \ldots)$$

If $x_1 = x_2$: $\psi = -\psi \Rightarrow \psi = 0$.

**Two fermions cannot occupy the same state!**

### Slater Determinants

For $n$ fermions in single-particle states $\phi_1, \ldots, \phi_n$:

$$\Psi(x_1, \ldots, x_n) = \frac{1}{\sqrt{n!}} \begin{vmatrix}
\phi_1(x_1) & \phi_1(x_2) & \cdots & \phi_1(x_n) \\
\phi_2(x_1) & \phi_2(x_2) & \cdots & \phi_2(x_n) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_n(x_1) & \phi_n(x_2) & \cdots & \phi_n(x_n)
\end{vmatrix}$$

The determinant automatically antisymmetrizes!

### Symmetrization and Antisymmetrization

**Symmetrizer:** $S = \frac{1}{n!} \sum_{\sigma \in S_n} P_\sigma$

**Antisymmetrizer:** $A = \frac{1}{n!} \sum_{\sigma \in S_n} \text{sgn}(\sigma) P_\sigma$

where $P_\sigma$ permutes particle labels.

---

## 7. Worked Examples

### Example 1: Cycle Decomposition

Find the cycle decomposition of $\sigma = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\ 4 & 6 & 2 & 7 & 8 & 1 & 3 & 5 \end{pmatrix}$.

Follow orbits:
- $1 \to 4 \to 7 \to 3 \to 2 \to 6 \to 1$: cycle $(1 \, 4 \, 7 \, 3 \, 2 \, 6)$
- $5 \to 8 \to 5$: cycle $(5 \, 8)$

So $\sigma = (1 \, 4 \, 7 \, 3 \, 2 \, 6)(5 \, 8)$.

Cycle type: $(6, 2)$

Order: $\text{lcm}(6, 2) = 6$

Sign: $(-1)^{6-1} \cdot (-1)^{2-1} = (-1)^5 \cdot (-1)^1 = (-1)^6 = +1$ (even)

### Example 2: Computing Products

Compute $(1 \, 2 \, 3)(2 \, 3 \, 4)$ in $S_4$.

Apply right-to-left:
- $1 \xrightarrow{(2\,3\,4)} 1 \xrightarrow{(1\,2\,3)} 2$
- $2 \xrightarrow{(2\,3\,4)} 3 \xrightarrow{(1\,2\,3)} 1$
- $3 \xrightarrow{(2\,3\,4)} 4 \xrightarrow{(1\,2\,3)} 4$
- $4 \xrightarrow{(2\,3\,4)} 2 \xrightarrow{(1\,2\,3)} 3$

Result: $1 \to 2, 2 \to 1, 3 \to 4, 4 \to 3$, which is $(1 \, 2)(3 \, 4)$.

### Example 3: Conjugacy Classes in $S_4$

Conjugacy classes correspond to cycle types. Cycle types for $n = 4$:

| Cycle Type | Example | Count |
|------------|---------|-------|
| $(1,1,1,1)$ | $e$ | 1 |
| $(2,1,1)$ | $(1 \, 2)$ | $\binom{4}{2} = 6$ |
| $(2,2)$ | $(1 \, 2)(3 \, 4)$ | $\frac{1}{2}\binom{4}{2} = 3$ |
| $(3,1)$ | $(1 \, 2 \, 3)$ | $\frac{4!}{3 \cdot 1} = 8$ |
| $(4)$ | $(1 \, 2 \, 3 \, 4)$ | $\frac{4!}{4} = 6$ |

Total: $1 + 6 + 3 + 8 + 6 = 24 = 4!$ ✓

---

## 8. Computational Lab

```python
"""
Day 286: Permutation Groups
Working with symmetric groups and cycle notation
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from math import factorial, gcd
from functools import reduce
from itertools import permutations

class Permutation:
    """
    Represents a permutation with various notations and operations.
    """

    def __init__(self, mapping: List[int] = None, cycles: List[List[int]] = None, n: int = None):
        """
        Initialize from mapping (one-line notation) or cycle notation.

        Parameters:
            mapping: [σ(1), σ(2), ..., σ(n)] (1-indexed in math, 0-indexed in code)
            cycles: List of cycles, e.g., [[1,2,3], [4,5]] for (1 2 3)(4 5)
            n: Degree of permutation (if not inferable)
        """
        if mapping is not None:
            self.n = len(mapping)
            self.mapping = list(mapping)
        elif cycles is not None:
            # Find n from cycles
            all_elements = set()
            for cycle in cycles:
                all_elements.update(cycle)
            self.n = n if n else max(all_elements)
            # Build mapping from cycles
            self.mapping = list(range(self.n))
            for cycle in cycles:
                if len(cycle) > 1:
                    for i in range(len(cycle)):
                        # cycle[i] -> cycle[(i+1) % len]
                        self.mapping[cycle[i] - 1] = cycle[(i + 1) % len(cycle)] - 1
            # Convert back to 1-indexed output
            self.mapping = [m + 1 for m in self.mapping]
        else:
            raise ValueError("Must provide mapping or cycles")

    @classmethod
    def identity(cls, n: int) -> 'Permutation':
        """Create identity permutation on n elements."""
        return cls(mapping=list(range(1, n + 1)))

    def __call__(self, i: int) -> int:
        """Apply permutation to element i (1-indexed)."""
        return self.mapping[i - 1]

    def __mul__(self, other: 'Permutation') -> 'Permutation':
        """Compose permutations: (self * other)(x) = self(other(x))."""
        assert self.n == other.n
        new_mapping = [self(other(i)) for i in range(1, self.n + 1)]
        return Permutation(mapping=new_mapping)

    def inverse(self) -> 'Permutation':
        """Compute inverse permutation."""
        inv_mapping = [0] * self.n
        for i, j in enumerate(self.mapping):
            inv_mapping[j - 1] = i + 1
        return Permutation(mapping=inv_mapping)

    def __pow__(self, k: int) -> 'Permutation':
        """Compute σ^k."""
        if k == 0:
            return Permutation.identity(self.n)
        if k < 0:
            return self.inverse() ** (-k)

        result = Permutation.identity(self.n)
        base = self
        while k > 0:
            if k % 2 == 1:
                result = result * base
            base = base * base
            k //= 2
        return result

    def __eq__(self, other: 'Permutation') -> bool:
        return self.mapping == other.mapping

    def cycles(self) -> List[List[int]]:
        """Decompose into disjoint cycles."""
        seen = [False] * self.n
        result = []

        for start in range(1, self.n + 1):
            if seen[start - 1]:
                continue

            cycle = []
            current = start
            while not seen[current - 1]:
                seen[current - 1] = True
                cycle.append(current)
                current = self(current)

            if len(cycle) > 1:
                result.append(cycle)

        return result

    def cycle_type(self) -> Tuple[int, ...]:
        """Return cycle type as sorted tuple of lengths."""
        cycs = self.cycles()
        # Include fixed points as 1-cycles
        fixed = self.n - sum(len(c) for c in cycs)
        lengths = sorted([len(c) for c in cycs] + [1] * fixed, reverse=True)
        return tuple(lengths)

    def order(self) -> int:
        """Compute order of permutation."""
        cycs = self.cycles()
        if not cycs:
            return 1
        lengths = [len(c) for c in cycs]
        return reduce(lambda a, b: a * b // gcd(a, b), lengths)

    def sign(self) -> int:
        """Compute sign (+1 for even, -1 for odd)."""
        # sign = (-1)^(n - number of cycles including fixed points)
        cycs = self.cycles()
        num_cycles = len(cycs) + (self.n - sum(len(c) for c in cycs))
        return (-1) ** (self.n - num_cycles)

    def is_even(self) -> bool:
        return self.sign() == 1

    def to_transpositions(self) -> List[Tuple[int, int]]:
        """Decompose into transpositions."""
        transpositions = []
        for cycle in self.cycles():
            # (a1 a2 ... ak) = (a1 ak)(a1 a_{k-1})...(a1 a2)
            for i in range(len(cycle) - 1, 0, -1):
                transpositions.append((cycle[0], cycle[i]))
        return transpositions

    def __str__(self) -> str:
        """String representation in cycle notation."""
        cycs = self.cycles()
        if not cycs:
            return "()"  # identity
        return "".join(f"({' '.join(map(str, c))})" for c in cycs)

    def __repr__(self) -> str:
        return f"Permutation({self.mapping})"


class SymmetricGroup:
    """
    The symmetric group S_n.
    """

    def __init__(self, n: int):
        self.n = n
        self._elements = None

    @property
    def order(self) -> int:
        return factorial(self.n)

    def elements(self) -> List[Permutation]:
        """Generate all elements of S_n."""
        if self._elements is None:
            self._elements = [
                Permutation(mapping=list(p))
                for p in permutations(range(1, self.n + 1))
            ]
        return self._elements

    def identity(self) -> Permutation:
        return Permutation.identity(self.n)

    def conjugacy_classes(self) -> Dict[Tuple, List[Permutation]]:
        """Group elements by conjugacy class (cycle type)."""
        classes = {}
        for perm in self.elements():
            ct = perm.cycle_type()
            if ct not in classes:
                classes[ct] = []
            classes[ct].append(perm)
        return classes

    def alternating_group(self) -> List[Permutation]:
        """Return elements of A_n (even permutations)."""
        return [p for p in self.elements() if p.is_even()]


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("PERMUTATION GROUPS")
    print("=" * 60)

    # Example 1: Creating permutations
    print("\n1. CREATING PERMUTATIONS")
    print("-" * 40)

    # From mapping
    sigma = Permutation(mapping=[3, 1, 4, 2])
    print(f"σ = [3, 1, 4, 2] = {sigma}")
    print(f"σ(1) = {sigma(1)}, σ(2) = {sigma(2)}, σ(3) = {sigma(3)}, σ(4) = {sigma(4)}")

    # From cycles
    tau = Permutation(cycles=[[1, 2, 3], [4, 5]], n=5)
    print(f"\nτ = (1 2 3)(4 5) in S_5 = {tau}")
    print(f"Mapping: {tau.mapping}")

    # Example 2: Operations
    print("\n2. PERMUTATION OPERATIONS")
    print("-" * 40)

    sigma = Permutation(cycles=[[1, 2, 3]], n=4)
    tau = Permutation(cycles=[[2, 3, 4]], n=4)

    print(f"σ = {sigma}")
    print(f"τ = {tau}")
    print(f"στ = {sigma * tau}")
    print(f"τσ = {tau * sigma}")
    print(f"σ and τ commute: {sigma * tau == tau * sigma}")

    print(f"\nσ⁻¹ = {sigma.inverse()}")
    print(f"σ³ = {sigma ** 3}")

    # Example 3: Cycle structure
    print("\n3. CYCLE STRUCTURE")
    print("-" * 40)

    perm = Permutation(mapping=[4, 6, 2, 7, 8, 1, 3, 5])
    print(f"Permutation: {perm.mapping}")
    print(f"Cycle notation: {perm}")
    print(f"Cycle type: {perm.cycle_type()}")
    print(f"Order: {perm.order()}")
    print(f"Sign: {perm.sign()} ({'even' if perm.is_even() else 'odd'})")
    print(f"Transposition decomposition: {perm.to_transpositions()}")

    # Example 4: S_4 structure
    print("\n4. STRUCTURE OF S_4")
    print("-" * 40)

    S4 = SymmetricGroup(4)
    print(f"|S_4| = {S4.order}")

    classes = S4.conjugacy_classes()
    print("\nConjugacy classes:")
    for cycle_type, elements in sorted(classes.items(), key=lambda x: -len(x[0])):
        print(f"  Cycle type {cycle_type}: {len(elements)} elements")
        if len(elements) <= 6:
            print(f"    {[str(e) for e in elements]}")

    # Example 5: Alternating group
    print("\n5. ALTERNATING GROUP A_4")
    print("-" * 40)

    A4 = S4.alternating_group()
    print(f"|A_4| = {len(A4)} = 4!/2")

    print("\nElements of A_4:")
    for perm in A4:
        print(f"  {perm}")

    # Example 6: Order statistics
    print("\n6. ORDER STATISTICS IN S_5")
    print("-" * 40)

    S5 = SymmetricGroup(5)
    orders = {}
    for perm in S5.elements():
        ord = perm.order()
        orders[ord] = orders.get(ord, 0) + 1

    print(f"Element orders in S_5:")
    for ord in sorted(orders.keys()):
        print(f"  Order {ord}: {orders[ord]} elements")

    # Example 7: Quantum connection - Fermion exchange
    print("\n7. FERMION WAVE FUNCTION (2 particles)")
    print("-" * 40)

    def symmetric_wf(psi, x1, x2):
        """Symmetric wave function (bosons)."""
        return (psi(x1, x2) + psi(x2, x1)) / np.sqrt(2)

    def antisymmetric_wf(psi, x1, x2):
        """Antisymmetric wave function (fermions)."""
        return (psi(x1, x2) - psi(x2, x1)) / np.sqrt(2)

    # Example: product states
    def product_state(x1, x2):
        return np.exp(-x1**2) * np.exp(-2*x2**2)

    x1, x2 = 1.0, 2.0

    psi_sym = symmetric_wf(product_state, x1, x2)
    psi_asym = antisymmetric_wf(product_state, x1, x2)

    print(f"Product state ψ(x₁,x₂) at (1,2): {product_state(x1, x2):.4f}")
    print(f"Symmetric (boson): {psi_sym:.4f}")
    print(f"Antisymmetric (fermion): {psi_asym:.4f}")

    print(f"\nExchange x₁ ↔ x₂:")
    print(f"Symmetric at (2,1): {symmetric_wf(product_state, x2, x1):.4f} (same)")
    print(f"Antisymmetric at (2,1): {antisymmetric_wf(product_state, x2, x1):.4f} (opposite)")

    # When x1 = x2
    print(f"\nAt x₁ = x₂ = 1:")
    print(f"Antisymmetric: {antisymmetric_wf(product_state, 1.0, 1.0):.4f} (zero!)")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Every permutation decomposes into disjoint cycles
    2. Order of permutation = lcm of cycle lengths
    3. Sign is well-defined: even/odd parity
    4. A_n is simple for n ≥ 5 (no proper normal subgroups)
    5. Cayley: Every group embeds in some S_n
    6. In QM: Fermions ↔ odd sign, Bosons ↔ even sign
    7. Pauli exclusion: antisymmetry → no two fermions in same state
    """)
```

---

## 9. Practice Problems

### Problem Set A: Basic Computations

**A1.** Write $\sigma = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\ 3 & 5 & 4 & 1 & 2 \end{pmatrix}$ in cycle notation. Find its order and sign.

**A2.** Compute $(1 \, 3 \, 5)(2 \, 4)(1 \, 2 \, 3)$ in $S_5$.

**A3.** Find all elements of $S_3$ and their cycle types.

### Problem Set B: Structure

**B1.** How many permutations in $S_6$ have cycle type $(3, 2, 1)$?

**B2.** Prove that $A_4$ has no subgroup of order 6.

**B3.** Find the center of $S_n$ for $n \geq 3$.

### Problem Set C: Advanced

**C1.** Prove: Two permutations are conjugate in $S_n$ iff they have the same cycle type.

**C2.** Show that $S_n$ is generated by $(1 \, 2)$ and $(1 \, 2 \, \cdots \, n)$.

**C3.** **(Physics)** Three identical fermions are in single-particle states $\phi_1, \phi_2, \phi_3$. Write the properly antisymmetrized 3-particle wave function as a Slater determinant.

---

## 10. Summary

### Key Formulas

$$\boxed{|S_n| = n!, \quad |A_n| = \frac{n!}{2}}$$

$$\boxed{\text{Order of } \sigma = \text{lcm}(\text{cycle lengths})}$$

$$\boxed{\text{sgn}(\sigma) = (-1)^{n - \text{(number of cycles)}}}$$

### Key Theorems

1. **Cycle Decomposition:** Every permutation is a unique product of disjoint cycles
2. **Cayley's Theorem:** Every group embeds in some $S_n$
3. **Simplicity of $A_n$:** $A_n$ is simple for $n \geq 5$

---

## 11. Preview: Day 287

Tomorrow is the Week 41 Review where we:
- Synthesize all abstract group theory concepts
- Work through comprehensive problem sets
- Connect theory to physical applications
- Prepare for representation theory

---

*"The symmetric group is the group theorist's playground—every finite group lives inside some $S_n$." — Arthur Cayley*
