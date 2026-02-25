# Day 281: Introduction to Groups — The Mathematics of Symmetry

## Overview

**Month 11, Week 41, Day 1 — Monday**

Today we begin our study of **group theory**, the mathematical framework that captures the essence of symmetry. Groups are fundamental to all of physics—every symmetry of nature, from rotational invariance to gauge symmetry, is described by a group. Understanding groups is essential for quantum mechanics, where particles are classified by group representations and conservation laws arise from symmetries via Noether's theorem.

## Prerequisites

From previous months:
- Set theory basics (sets, functions, bijections)
- Matrix operations and determinants
- Complex numbers and their multiplication
- Modular arithmetic concepts

## Learning Objectives

By the end of today, you will be able to:

1. State and explain the four group axioms
2. Verify whether a given set with operation forms a group
3. Identify common groups: integers, rationals, matrices, roots of unity
4. Construct and interpret Cayley tables
5. Understand the connection between groups and physical symmetries
6. Implement group operations computationally

---

## 1. The Group Axioms

### What is a Group?

A **group** is one of the most fundamental structures in mathematics—a set equipped with a binary operation that satisfies four key properties. The concept was developed in the 19th century by Galois, Cayley, and others, originally to study polynomial equations.

**Definition:** A **group** $(G, \cdot)$ consists of:
- A set $G$
- A binary operation $\cdot : G \times G \to G$

satisfying the following axioms:

$$\boxed{\text{G1 (Closure): } \forall a, b \in G, \quad a \cdot b \in G}$$

$$\boxed{\text{G2 (Associativity): } \forall a, b, c \in G, \quad (a \cdot b) \cdot c = a \cdot (b \cdot c)}$$

$$\boxed{\text{G3 (Identity): } \exists e \in G \text{ such that } \forall a \in G, \quad e \cdot a = a \cdot e = a}$$

$$\boxed{\text{G4 (Inverse): } \forall a \in G, \exists a^{-1} \in G \text{ such that } a \cdot a^{-1} = a^{-1} \cdot a = e}$$

### Understanding Each Axiom

**Closure** ensures that combining two group elements always gives another group element—we never "leave" the group.

**Associativity** means we can compute products without worrying about parentheses: $a \cdot b \cdot c$ is unambiguous.

**Identity** provides a "do nothing" element that leaves everything unchanged.

**Inverse** guarantees every operation can be undone—for every transformation, there's a reverse transformation.

### Important Properties

**Theorem (Uniqueness of Identity):** The identity element is unique.

*Proof:* Suppose $e$ and $e'$ are both identities. Then $e = e \cdot e' = e'$. ∎

**Theorem (Uniqueness of Inverse):** Each element has a unique inverse.

*Proof:* Suppose $b$ and $c$ are both inverses of $a$. Then:
$$b = b \cdot e = b \cdot (a \cdot c) = (b \cdot a) \cdot c = e \cdot c = c$$
∎

**Theorem (Inverse of Product):** $(a \cdot b)^{-1} = b^{-1} \cdot a^{-1}$

*Proof:* $(a \cdot b) \cdot (b^{-1} \cdot a^{-1}) = a \cdot (b \cdot b^{-1}) \cdot a^{-1} = a \cdot e \cdot a^{-1} = a \cdot a^{-1} = e$. ∎

### Abelian Groups

**Definition:** A group is **abelian** (or commutative) if:
$$\forall a, b \in G: \quad a \cdot b = b \cdot a$$

Named after Norwegian mathematician Niels Henrik Abel.

---

## 2. Examples of Groups

### Example 1: Integers Under Addition $(\mathbb{Z}, +)$

- **Set:** $\mathbb{Z} = \{\ldots, -2, -1, 0, 1, 2, \ldots\}$
- **Operation:** Addition
- **Closure:** Sum of integers is an integer ✓
- **Associativity:** $(a + b) + c = a + (b + c)$ ✓
- **Identity:** $0$ (since $a + 0 = 0 + a = a$) ✓
- **Inverse:** $-a$ (since $a + (-a) = 0$) ✓

This is an **abelian** group (addition commutes).

### Example 2: Non-Zero Rationals Under Multiplication $(\mathbb{Q}^*, \times)$

- **Set:** $\mathbb{Q}^* = \mathbb{Q} \setminus \{0\}$
- **Operation:** Multiplication
- **Closure:** Product of non-zero rationals is non-zero rational ✓
- **Associativity:** Multiplication is associative ✓
- **Identity:** $1$ ✓
- **Inverse:** $1/a$ for each $a \neq 0$ ✓

Note: We must exclude 0 because it has no multiplicative inverse.

### Example 3: The General Linear Group $GL_n(\mathbb{R})$

- **Set:** Invertible $n \times n$ real matrices
- **Operation:** Matrix multiplication
- **Closure:** Product of invertible matrices is invertible (det(AB) = det(A)det(B) ≠ 0) ✓
- **Associativity:** Matrix multiplication is associative ✓
- **Identity:** Identity matrix $I_n$ ✓
- **Inverse:** Matrix inverse $A^{-1}$ ✓

This is **non-abelian** for $n \geq 2$ (matrix multiplication doesn't commute).

### Example 4: Roots of Unity $\mu_n$

The $n$-th roots of unity are:
$$\mu_n = \{e^{2\pi i k/n} : k = 0, 1, \ldots, n-1\} = \{1, \omega, \omega^2, \ldots, \omega^{n-1}\}$$

where $\omega = e^{2\pi i/n}$ is a primitive $n$-th root.

This forms a group under multiplication (cyclic group of order $n$).

### Example 5: Symmetries of an Equilateral Triangle — The Dihedral Group $D_3$

The symmetries of an equilateral triangle form a group $D_3$ with 6 elements:

- $e$: identity (do nothing)
- $r$: rotation by 120°
- $r^2$: rotation by 240°
- $s$: reflection about vertical axis
- $sr$: reflection about one axis
- $sr^2$: reflection about another axis

Key relations: $r^3 = e$, $s^2 = e$, $srs = r^{-1}$

This is **non-abelian**: $rs \neq sr$.

### Example 6: Integers Modulo $n$ — $\mathbb{Z}_n$

$$\mathbb{Z}_n = \{0, 1, 2, \ldots, n-1\}$$

with addition modulo $n$.

- Identity: 0
- Inverse of $a$: $n - a$

This is a cyclic group of order $n$.

---

## 3. Non-Examples (Not Groups)

### Non-Example 1: Natural Numbers $(\mathbb{N}, +)$

- Closure: ✓
- Associativity: ✓
- Identity: 0 ✓
- **Inverse: ✗** — The element 5 has no inverse in $\mathbb{N}$ (no natural number $n$ satisfies $5 + n = 0$)

### Non-Example 2: Integers Under Multiplication $(\mathbb{Z}, \times)$

- Closure: ✓
- Associativity: ✓
- Identity: 1 ✓
- **Inverse: ✗** — Most integers have no multiplicative inverse in $\mathbb{Z}$ (e.g., $1/2 \notin \mathbb{Z}$)

### Non-Example 3: Real Numbers Under Subtraction

- **Associativity: ✗** — $(a - b) - c \neq a - (b - c)$ in general

---

## 4. Cayley Tables

A **Cayley table** (or multiplication table) displays all products in a finite group.

### Cayley Table for $\mathbb{Z}_4$

| + | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 0 | 1 | 2 | 3 |
| 1 | 1 | 2 | 3 | 0 |
| 2 | 2 | 3 | 0 | 1 |
| 3 | 3 | 0 | 1 | 2 |

### Properties of Cayley Tables

1. **Every element appears exactly once in each row and column** (this is called the "Latin square" property)
2. The identity row/column reproduces the header
3. If the table is symmetric about the diagonal, the group is abelian

### Cayley Table for $D_3$ (Non-Abelian)

| · | e | r | r² | s | sr | sr² |
|---|---|---|---|---|---|---|
| e | e | r | r² | s | sr | sr² |
| r | r | r² | e | sr² | s | sr |
| r² | r² | e | r | sr | sr² | s |
| s | s | sr | sr² | e | r | r² |
| sr | sr | sr² | s | r² | e | r |
| sr² | sr² | s | sr | r | r² | e |

Note: This table is NOT symmetric about the diagonal (the group is non-abelian).

---

## 5. Quantum Mechanics Connection

### Symmetries as Groups

In quantum mechanics, symmetry transformations form groups:

**Rotation Group SO(3):** 3D rotations preserve lengths and angles
- Closure: Composition of rotations is a rotation
- Identity: No rotation
- Inverse: Rotate in opposite direction
- Associativity: Composition of functions is associative

**Unitary Group U(n):** Transformations preserving inner products in Hilbert space
$$U^\dagger U = UU^\dagger = I$$

### Operators and Observables

For every continuous symmetry, there's a corresponding observable:

| Symmetry | Group | Observable | Conservation Law |
|----------|-------|------------|-----------------|
| Time translation | $\mathbb{R}$ | Hamiltonian $\hat{H}$ | Energy |
| Space translation | $\mathbb{R}^3$ | Momentum $\hat{\mathbf{p}}$ | Linear momentum |
| Rotation | SO(3) | Angular momentum $\hat{\mathbf{L}}$ | Angular momentum |
| Phase | U(1) | Number operator $\hat{N}$ | Particle number |

### Commutators and Non-Commutativity

Non-abelian groups in QM lead to non-commuting observables:

$$[L_x, L_y] = i\hbar L_z \neq 0$$

This non-commutativity is directly related to SO(3) being non-abelian!

### Discrete Symmetries

- **Parity P:** Spatial inversion $(x, y, z) \to (-x, -y, -z)$
- **Time reversal T:** $t \to -t$
- **Charge conjugation C:** Particle ↔ antiparticle

These form the discrete group $\{I, P, T, C, PT, PC, TC, PTC\}$.

---

## 6. Worked Examples

### Example 1: Verify $(\mathbb{Z}_5^*, \times)$ is a Group

The set $\mathbb{Z}_5^* = \{1, 2, 3, 4\}$ consists of non-zero elements of $\mathbb{Z}_5$.

**Cayley Table:**

| × | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 1 | 2 | 3 | 4 |
| 2 | 2 | 4 | 1 | 3 |
| 3 | 3 | 1 | 4 | 2 |
| 4 | 4 | 3 | 2 | 1 |

**Verification:**
- Closure: All products are in $\{1, 2, 3, 4\}$ ✓
- Associativity: Inherited from $\mathbb{Z}$ ✓
- Identity: 1 ✓
- Inverses: $1^{-1}=1$, $2^{-1}=3$ (since $2 \times 3 = 6 \equiv 1$), $3^{-1}=2$, $4^{-1}=4$ ✓

### Example 2: Show the Klein Four-Group

The Klein four-group $V_4 = \{e, a, b, c\}$ has Cayley table:

| · | e | a | b | c |
|---|---|---|---|---|
| e | e | a | b | c |
| a | a | e | c | b |
| b | b | c | e | a |
| c | c | b | a | e |

Properties:
- Every non-identity element is its own inverse
- Abelian (symmetric table)
- Can be realized as $\mathbb{Z}_2 \times \mathbb{Z}_2$

**Physical example:** Symmetries of a rectangle (not square):
- $e$: identity
- $a$: rotation by 180°
- $b$: horizontal reflection
- $c$: vertical reflection

### Example 3: Matrix Group Verification

Show that $SL_2(\mathbb{R}) = \{A \in M_2(\mathbb{R}) : \det(A) = 1\}$ is a group.

**Closure:** If $\det(A) = 1$ and $\det(B) = 1$, then $\det(AB) = \det(A)\det(B) = 1$ ✓

**Identity:** $I_2$ has $\det(I_2) = 1$ ✓

**Inverse:** If $\det(A) = 1$, then $A$ is invertible with $\det(A^{-1}) = 1/\det(A) = 1$ ✓

**Associativity:** Inherited from matrix multiplication ✓

---

## 7. Computational Lab

```python
"""
Day 281: Introduction to Groups
Computational exploration of group structures
"""

import numpy as np
from itertools import product
from typing import Callable, List, Set, Tuple, Optional

class FiniteGroup:
    """
    A class representing a finite group with explicit multiplication table.

    Attributes:
        elements: List of group elements
        operation: Function computing the group operation
        table: Cayley table as numpy array of indices
    """

    def __init__(self, elements: List, operation: Callable):
        """
        Initialize a finite group.

        Parameters:
            elements: List of group elements
            operation: Binary operation function(a, b) -> c
        """
        self.elements = list(elements)
        self.n = len(elements)
        self.operation = operation
        self.element_to_index = {e: i for i, e in enumerate(elements)}

        # Build Cayley table
        self.table = np.zeros((self.n, self.n), dtype=int)
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                result = operation(a, b)
                self.table[i, j] = self.element_to_index[result]

    def multiply(self, a, b):
        """Compute a * b using the group operation."""
        return self.operation(a, b)

    def verify_closure(self) -> bool:
        """Check if operation is closed."""
        for a in self.elements:
            for b in self.elements:
                result = self.operation(a, b)
                if result not in self.element_to_index:
                    return False
        return True

    def find_identity(self) -> Optional[any]:
        """Find the identity element."""
        for e in self.elements:
            is_identity = True
            for a in self.elements:
                if self.operation(e, a) != a or self.operation(a, e) != a:
                    is_identity = False
                    break
            if is_identity:
                return e
        return None

    def find_inverse(self, a) -> Optional[any]:
        """Find the inverse of element a."""
        e = self.find_identity()
        if e is None:
            return None
        for b in self.elements:
            if self.operation(a, b) == e and self.operation(b, a) == e:
                return b
        return None

    def verify_group_axioms(self) -> dict:
        """
        Verify all four group axioms.

        Returns:
            Dictionary with verification results
        """
        results = {}

        # Closure
        results['closure'] = self.verify_closure()

        # Associativity (check all triples)
        results['associativity'] = True
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    ab_c = self.operation(self.operation(a, b), c)
                    a_bc = self.operation(a, self.operation(b, c))
                    if ab_c != a_bc:
                        results['associativity'] = False
                        break

        # Identity
        e = self.find_identity()
        results['identity'] = e is not None
        results['identity_element'] = e

        # Inverses
        results['inverses'] = True
        for a in self.elements:
            if self.find_inverse(a) is None:
                results['inverses'] = False
                break

        # Check if abelian
        results['abelian'] = all(
            self.operation(a, b) == self.operation(b, a)
            for a in self.elements for b in self.elements
        )

        return results

    def print_cayley_table(self):
        """Print the Cayley table."""
        # Header
        header = "  | " + " | ".join(str(e) for e in self.elements)
        print(header)
        print("-" * len(header))

        # Rows
        for i, a in enumerate(self.elements):
            row = [self.elements[self.table[i, j]] for j in range(self.n)]
            print(f"{a} | " + " | ".join(str(e) for e in row))

    def element_order(self, a) -> int:
        """Find the order of element a (smallest n > 0 such that a^n = e)."""
        e = self.find_identity()
        current = a
        order = 1
        while current != e:
            current = self.operation(current, a)
            order += 1
            if order > self.n:  # Safety check
                return -1
        return order


# Example 1: Integers modulo n
def create_Zn(n: int) -> FiniteGroup:
    """Create the cyclic group Z_n."""
    elements = list(range(n))
    operation = lambda a, b: (a + b) % n
    return FiniteGroup(elements, operation)


# Example 2: Multiplicative group of units modulo n
def create_Zn_star(n: int) -> FiniteGroup:
    """Create the multiplicative group (Z/nZ)*."""
    from math import gcd
    elements = [k for k in range(1, n) if gcd(k, n) == 1]
    operation = lambda a, b: (a * b) % n
    return FiniteGroup(elements, operation)


# Example 3: Roots of unity
def create_roots_of_unity(n: int) -> FiniteGroup:
    """Create the group of n-th roots of unity."""
    elements = [np.exp(2j * np.pi * k / n) for k in range(n)]
    # Round to avoid floating point issues
    def multiply(a, b):
        result = a * b
        # Find closest root of unity
        phases = [np.exp(2j * np.pi * k / n) for k in range(n)]
        dists = [abs(result - p) for p in phases]
        return phases[np.argmin(dists)]
    return FiniteGroup(elements, multiply)


# Example 4: Symmetric group S_n (permutations)
def create_symmetric_group(n: int) -> FiniteGroup:
    """Create the symmetric group S_n."""
    from itertools import permutations

    # Represent permutations as tuples
    elements = list(permutations(range(n)))

    def compose(p1, p2):
        """Compose permutations: (p1 ∘ p2)(i) = p1(p2(i))"""
        return tuple(p1[p2[i]] for i in range(n))

    return FiniteGroup(elements, compose)


# Example 5: Dihedral group D_n
def create_dihedral_group(n: int) -> FiniteGroup:
    """
    Create the dihedral group D_n (symmetries of regular n-gon).
    Elements are represented as (rotation, reflection) pairs.
    """
    # Elements: r^k for rotations, s*r^k for reflections
    # Encode as (k, 0) for r^k and (k, 1) for s*r^k
    elements = [(k, 0) for k in range(n)] + [(k, 1) for k in range(n)]

    def multiply(g1, g2):
        k1, s1 = g1
        k2, s2 = g2

        if s1 == 0:  # g1 = r^k1
            # r^k1 * r^k2 = r^(k1+k2)
            # r^k1 * s*r^k2 = s*r^(-k1+k2)
            if s2 == 0:
                return ((k1 + k2) % n, 0)
            else:
                return ((-k1 + k2) % n, 1)
        else:  # g1 = s*r^k1
            # s*r^k1 * r^k2 = s*r^(k1-k2)
            # s*r^k1 * s*r^k2 = r^(k1-k2)
            if s2 == 0:
                return ((k1 - k2) % n, 1)
            else:
                return ((k1 - k2) % n, 0)

    return FiniteGroup(elements, multiply)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("GROUP THEORY COMPUTATIONAL LAB")
    print("=" * 60)

    # Test Z_4
    print("\n1. CYCLIC GROUP Z_4")
    print("-" * 40)
    Z4 = create_Zn(4)
    results = Z4.verify_group_axioms()
    print(f"Group axioms verification:")
    for axiom, satisfied in results.items():
        print(f"  {axiom}: {satisfied}")
    print("\nCayley table:")
    Z4.print_cayley_table()
    print("\nElement orders:")
    for a in Z4.elements:
        print(f"  |{a}| = {Z4.element_order(a)}")

    # Test Z_5^*
    print("\n2. MULTIPLICATIVE GROUP (Z/5Z)*")
    print("-" * 40)
    Z5_star = create_Zn_star(5)
    results = Z5_star.verify_group_axioms()
    print(f"Elements: {Z5_star.elements}")
    print(f"Order: {Z5_star.n}")
    print(f"Abelian: {results['abelian']}")
    print("\nCayley table:")
    Z5_star.print_cayley_table()

    # Test S_3
    print("\n3. SYMMETRIC GROUP S_3")
    print("-" * 40)
    S3 = create_symmetric_group(3)
    results = S3.verify_group_axioms()
    print(f"Order: {S3.n}")
    print(f"Abelian: {results['abelian']}")
    print(f"Identity: {results['identity_element']}")

    # Test D_3
    print("\n4. DIHEDRAL GROUP D_3")
    print("-" * 40)
    D3 = create_dihedral_group(3)
    results = D3.verify_group_axioms()
    print(f"Order: {D3.n}")
    print(f"Abelian: {results['abelian']}")

    # Name the elements
    element_names = {
        (0, 0): 'e', (1, 0): 'r', (2, 0): 'r²',
        (0, 1): 's', (1, 1): 'sr', (2, 1): 'sr²'
    }
    print("\nElements:")
    for elem, name in element_names.items():
        order = D3.element_order(elem)
        print(f"  {name} = {elem}, order = {order}")

    # Matrix groups
    print("\n5. MATRIX GROUP VERIFICATION")
    print("-" * 40)

    # Pauli matrices form a group (with ±I, ±iI, ±σ_x, etc.)
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    print("Pauli matrix relations:")
    print(f"σ_x² = I: {np.allclose(sigma_x @ sigma_x, I)}")
    print(f"σ_y² = I: {np.allclose(sigma_y @ sigma_y, I)}")
    print(f"σ_z² = I: {np.allclose(sigma_z @ sigma_z, I)}")
    print(f"σ_x σ_y = iσ_z: {np.allclose(sigma_x @ sigma_y, 1j * sigma_z)}")
    print(f"σ_y σ_x = -iσ_z: {np.allclose(sigma_y @ sigma_x, -1j * sigma_z)}")
    print(f"[σ_x, σ_y] = 2iσ_z: {np.allclose(sigma_x @ sigma_y - sigma_y @ sigma_x, 2j * sigma_z)}")

    # Rotation matrices
    print("\n6. ROTATION GROUP SO(2)")
    print("-" * 40)

    def rotation_matrix(theta):
        """2D rotation matrix."""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    theta1, theta2 = np.pi/4, np.pi/3
    R1 = rotation_matrix(theta1)
    R2 = rotation_matrix(theta2)
    R12 = R1 @ R2
    R_sum = rotation_matrix(theta1 + theta2)

    print(f"R(π/4) @ R(π/3) = R(π/4 + π/3): {np.allclose(R12, R_sum)}")
    print(f"det(R(θ)) = 1: {np.allclose(np.linalg.det(R1), 1)}")
    print(f"R(θ)^T = R(-θ): {np.allclose(R1.T, rotation_matrix(-theta1))}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Groups capture the essence of symmetry mathematically
    2. Every symmetry transformation has an inverse (undo)
    3. Composition of symmetries is another symmetry (closure)
    4. The identity is the "do nothing" transformation
    5. Non-abelian groups lead to non-commuting operations
    6. In QM: non-commuting observables ↔ non-abelian symmetry groups
    """)
```

---

## 8. Practice Problems

### Problem Set A: Basic Verification

**A1.** Verify that $(\{1, -1, i, -i\}, \times)$ forms a group. Find its Cayley table.

**A2.** Does $(\mathbb{R}, -)$ (real numbers under subtraction) form a group? Explain which axiom(s) fail.

**A3.** Show that the set of $2 \times 2$ upper triangular matrices with 1s on the diagonal:
$$\left\{ \begin{pmatrix} 1 & a \\ 0 & 1 \end{pmatrix} : a \in \mathbb{R} \right\}$$
forms a group under matrix multiplication.

### Problem Set B: Cayley Tables

**B1.** Construct the Cayley table for $\mathbb{Z}_6$. Identify the identity and find the inverse of each element.

**B2.** The group $U(8) = \{1, 3, 5, 7\}$ is the multiplicative group of units modulo 8. Build its Cayley table and verify it's a group.

**B3.** By examining Cayley tables, show that $\mathbb{Z}_4$ and the Klein four-group $V_4$ are NOT isomorphic.

### Problem Set C: Challenging

**C1.** Prove that if every element of a group satisfies $a^2 = e$, then the group is abelian.

*Hint:* Consider $(ab)^2 = e$ and expand.

**C2.** Let $G$ be a finite group where $|G| = p$ for some prime $p$. Prove that $G$ must be cyclic.

*Hint:* Use Lagrange's theorem (which you'll learn tomorrow) and the fact that the order of any element must divide $|G|$.

**C3.** **(Physics Connection)** The set of Lorentz transformations in special relativity forms a group (the Lorentz group). Explain how each group axiom corresponds to a physical principle:
- What does closure mean physically?
- What is the identity element?
- What does the inverse of a Lorentz boost represent?

---

## 9. Solutions to Selected Problems

### Solution A1

The set $\{1, -1, i, -i\}$ under multiplication:

**Cayley Table:**
| × | 1 | -1 | i | -i |
|---|---|---|---|---|
| 1 | 1 | -1 | i | -i |
| -1 | -1 | 1 | -i | i |
| i | i | -i | -1 | 1 |
| -i | -i | i | 1 | -1 |

- **Closure:** All products are in the set ✓
- **Associativity:** Inherited from complex multiplication ✓
- **Identity:** 1 ✓
- **Inverses:** $1^{-1}=1$, $(-1)^{-1}=-1$, $i^{-1}=-i$, $(-i)^{-1}=i$ ✓

This is the cyclic group of order 4, generated by $i$.

### Solution C1

Suppose every element satisfies $a^2 = e$. We want to show $ab = ba$.

Consider $(ab)^2 = e$:
$$(ab)(ab) = e$$
$$abab = e$$

Multiply on left by $a$ and on right by $b$:
$$a(abab)b = aeb = ab$$
$$(a^2)(ba)(b^2) = ab$$
$$e \cdot ba \cdot e = ab$$
$$ba = ab$$

Therefore $G$ is abelian. ∎

---

## 10. Summary

### Key Concepts

| Concept | Definition | Importance |
|---------|------------|------------|
| Group | Set with associative operation, identity, inverses | Foundation of symmetry |
| Abelian group | Commutative group | Simpler structure |
| Cayley table | Complete multiplication table | Defines finite groups |
| Order | Number of elements in group | Basic invariant |
| Element order | Smallest $n$ with $a^n = e$ | Subgroup generation |

### Key Formulas

$$\boxed{(ab)^{-1} = b^{-1}a^{-1}}$$

$$\boxed{a^n = \underbrace{a \cdot a \cdots a}_{n \text{ times}}}$$

### Connection to Quantum Mechanics

- **Symmetry groups** describe invariances of physical systems
- **Non-abelian groups** lead to non-commuting observables
- **Discrete groups** describe discrete symmetries (parity, etc.)
- **Continuous groups** (Lie groups) describe continuous symmetries

---

## 11. Daily Checklist

### Understanding
- [ ] I can state all four group axioms
- [ ] I can verify if a given structure is a group
- [ ] I understand why inverses and identity must be unique
- [ ] I can construct Cayley tables for small groups

### Skills
- [ ] I can work with modular arithmetic groups
- [ ] I can compute products in matrix groups
- [ ] I can identify abelian vs non-abelian groups
- [ ] I can implement group operations in Python

### Connections
- [ ] I understand how symmetries form groups
- [ ] I can relate group commutativity to observable commutativity
- [ ] I see how group structure constrains physical systems

---

## 12. Preview: Day 282

Tomorrow we explore **subgroups and cosets**, including:
- Definition of subgroups and criteria for recognizing them
- Cosets and their properties
- **Lagrange's Theorem:** The order of a subgroup divides the order of the group
- Normal subgroups and their special role
- Connection to conservation laws in physics

Lagrange's theorem is one of the most important results in group theory—it severely constrains what subgroups can exist and has profound implications for physics.

---

*"The universe is built on symmetry. Group theory is the mathematics of symmetry." — Hermann Weyl*
