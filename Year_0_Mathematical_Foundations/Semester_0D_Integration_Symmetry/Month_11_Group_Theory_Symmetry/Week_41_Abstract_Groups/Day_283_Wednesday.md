# Day 283: Group Homomorphisms — Structure-Preserving Maps

## Overview

**Month 11, Week 41, Day 3 — Wednesday**

Today we study **homomorphisms**—functions between groups that preserve the group structure. Homomorphisms are the "right" maps to consider in group theory, just as continuous functions are natural in topology and linear maps are natural in linear algebra. The kernel and image of a homomorphism reveal deep structural information, culminating in the powerful **Isomorphism Theorems**.

## Prerequisites

From Days 281-282:
- Group axioms and basic properties
- Subgroups and normal subgroups
- Cosets and Lagrange's theorem

## Learning Objectives

By the end of today, you will be able to:

1. Define and identify group homomorphisms
2. Compute kernels and images of homomorphisms
3. Prove that the kernel is a normal subgroup
4. State and apply the First Isomorphism Theorem
5. Recognize isomorphisms and classify groups up to isomorphism
6. Connect homomorphisms to physical symmetry operations

---

## 1. Homomorphisms

### Definition

**Definition:** A **homomorphism** from group $(G, *)$ to group $(H, \cdot)$ is a function $\phi: G \to H$ such that:

$$\boxed{\phi(a * b) = \phi(a) \cdot \phi(b) \quad \forall a, b \in G}$$

The homomorphism property says: "the image of a product is the product of the images."

### Basic Properties

**Theorem:** Let $\phi: G \to H$ be a homomorphism. Then:

1. $\phi(e_G) = e_H$ (identity maps to identity)
2. $\phi(a^{-1}) = \phi(a)^{-1}$ (inverses map to inverses)
3. $\phi(a^n) = \phi(a)^n$ for all $n \in \mathbb{Z}$

*Proof of (1):*
$$\phi(e_G) = \phi(e_G \cdot e_G) = \phi(e_G) \cdot \phi(e_G)$$
Multiplying both sides by $\phi(e_G)^{-1}$: $e_H = \phi(e_G)$. ∎

*Proof of (2):*
$$\phi(a) \cdot \phi(a^{-1}) = \phi(a \cdot a^{-1}) = \phi(e_G) = e_H$$
So $\phi(a^{-1}) = \phi(a)^{-1}$. ∎

### Types of Homomorphisms

| Name | Definition | Significance |
|------|------------|--------------|
| **Monomorphism** | Injective homomorphism | $G$ embeds in $H$ |
| **Epimorphism** | Surjective homomorphism | $H$ is a "quotient" of $G$ |
| **Isomorphism** | Bijective homomorphism | $G \cong H$ (same structure) |
| **Endomorphism** | Homomorphism $G \to G$ | Self-map |
| **Automorphism** | Isomorphism $G \to G$ | Symmetry of $G$ |

---

## 2. Kernel and Image

### Definitions

**Definition:** For homomorphism $\phi: G \to H$:

$$\boxed{\ker(\phi) = \{g \in G : \phi(g) = e_H\} = \phi^{-1}(\{e_H\})}$$

$$\boxed{\text{Im}(\phi) = \{\phi(g) : g \in G\} = \phi(G)}$$

### Fundamental Properties

**Theorem:** Let $\phi: G \to H$ be a homomorphism. Then:
1. $\ker(\phi) \trianglelefteq G$ (kernel is a normal subgroup of $G$)
2. $\text{Im}(\phi) \leq H$ (image is a subgroup of $H$)

*Proof that $\ker(\phi) \trianglelefteq G$:*

First, $\ker(\phi)$ is a subgroup: For $a, b \in \ker(\phi)$:
$$\phi(ab^{-1}) = \phi(a)\phi(b)^{-1} = e_H \cdot e_H^{-1} = e_H$$
So $ab^{-1} \in \ker(\phi)$.

Now, normality: For $g \in G$ and $n \in \ker(\phi)$:
$$\phi(gng^{-1}) = \phi(g)\phi(n)\phi(g)^{-1} = \phi(g) \cdot e_H \cdot \phi(g)^{-1} = e_H$$
So $gng^{-1} \in \ker(\phi)$. ∎

**Theorem:** $\phi$ is injective $\Leftrightarrow$ $\ker(\phi) = \{e_G\}$

*Proof:*
($\Rightarrow$) If $\phi$ is injective and $\phi(g) = e_H = \phi(e_G)$, then $g = e_G$.
($\Leftarrow$) If $\phi(a) = \phi(b)$, then $\phi(ab^{-1}) = e_H$, so $ab^{-1} \in \ker(\phi) = \{e_G\}$, hence $a = b$. ∎

---

## 3. Examples of Homomorphisms

### Example 1: Determinant

The determinant $\det: GL_n(\mathbb{R}) \to \mathbb{R}^*$ is a homomorphism:
$$\det(AB) = \det(A) \det(B)$$

- **Kernel:** $\ker(\det) = SL_n(\mathbb{R})$ (matrices with determinant 1)
- **Image:** $\text{Im}(\det) = \mathbb{R}^*$ (all non-zero reals) — surjective!

### Example 2: Exponential Map

The exponential $\exp: (\mathbb{R}, +) \to (\mathbb{R}^+, \times)$ is a homomorphism:
$$e^{a+b} = e^a \cdot e^b$$

- **Kernel:** $\ker(\exp) = \{0\}$ — injective!
- **Image:** $\text{Im}(\exp) = \mathbb{R}^+$ — surjective!

This is an **isomorphism**: $(\mathbb{R}, +) \cong (\mathbb{R}^+, \times)$.

### Example 3: Reduction Modulo n

The map $\pi: \mathbb{Z} \to \mathbb{Z}_n$ defined by $\pi(k) = k \mod n$ is a homomorphism:
$$\pi(a + b) = (a + b) \mod n = \pi(a) + \pi(b)$$

- **Kernel:** $\ker(\pi) = n\mathbb{Z}$ (multiples of $n$)
- **Image:** $\text{Im}(\pi) = \mathbb{Z}_n$ — surjective!

### Example 4: Sign Homomorphism

The sign $\text{sgn}: S_n \to \{1, -1\}$ maps permutations to their parity:
$$\text{sgn}(\sigma \tau) = \text{sgn}(\sigma) \cdot \text{sgn}(\tau)$$

- **Kernel:** $\ker(\text{sgn}) = A_n$ (alternating group — even permutations)
- **Image:** $\{1, -1\}$ for $n \geq 2$

### Example 5: Complex Exponential

The map $\phi: (\mathbb{R}, +) \to (S^1, \times)$ defined by $\phi(t) = e^{2\pi i t}$ is a homomorphism:
$$\phi(s + t) = e^{2\pi i(s+t)} = e^{2\pi is} \cdot e^{2\pi it} = \phi(s) \cdot \phi(t)$$

- **Kernel:** $\ker(\phi) = \mathbb{Z}$ (integers)
- **Image:** $S^1 = \{z \in \mathbb{C} : |z| = 1\}$ — surjective!

---

## 4. The Isomorphism Theorems

### First Isomorphism Theorem

**Theorem (First Isomorphism Theorem):** If $\phi: G \to H$ is a homomorphism, then:

$$\boxed{G/\ker(\phi) \cong \text{Im}(\phi)}$$

More precisely: The map $\bar{\phi}: G/\ker(\phi) \to \text{Im}(\phi)$ defined by $\bar{\phi}(g \ker(\phi)) = \phi(g)$ is a well-defined isomorphism.

*Proof Sketch:*
1. **Well-defined:** If $g \ker(\phi) = h \ker(\phi)$, then $g^{-1}h \in \ker(\phi)$, so $\phi(g^{-1}h) = e_H$, hence $\phi(g) = \phi(h)$.

2. **Homomorphism:** $\bar{\phi}((g\ker)(h\ker)) = \bar{\phi}(gh\ker) = \phi(gh) = \phi(g)\phi(h) = \bar{\phi}(g\ker)\bar{\phi}(h\ker)$.

3. **Injective:** If $\bar{\phi}(g\ker) = e_H$, then $\phi(g) = e_H$, so $g \in \ker$, hence $g\ker = \ker$ is the identity coset.

4. **Surjective:** Every $\phi(g) \in \text{Im}(\phi)$ equals $\bar{\phi}(g\ker)$. ∎

### Applications of First Isomorphism Theorem

**Example 1:** $\det: GL_n \to \mathbb{R}^*$ with $\ker = SL_n$
$$GL_n / SL_n \cong \mathbb{R}^*$$

**Example 2:** $\pi: \mathbb{Z} \to \mathbb{Z}_n$ with $\ker = n\mathbb{Z}$
$$\mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}_n$$

**Example 3:** $\text{sgn}: S_n \to \{1, -1\}$ with $\ker = A_n$
$$S_n/A_n \cong \{1, -1\} \cong \mathbb{Z}_2$$

### Second and Third Isomorphism Theorems

**Second Isomorphism Theorem:** If $H \leq G$ and $N \trianglelefteq G$, then:
- $HN \leq G$ and $H \cap N \trianglelefteq H$
- $HN/N \cong H/(H \cap N)$

**Third Isomorphism Theorem:** If $N \trianglelefteq K \trianglelefteq G$, then:
- $K/N \trianglelefteq G/N$
- $(G/N)/(K/N) \cong G/K$

---

## 5. Isomorphisms and Group Classification

### What Isomorphism Means

Two groups $G$ and $H$ are **isomorphic** ($G \cong H$) if there exists a bijective homomorphism between them. Isomorphic groups are "the same" in terms of group structure—they differ only in the labeling of elements.

### Properties Preserved by Isomorphism

If $\phi: G \to H$ is an isomorphism:
- $|G| = |H|$ (same order)
- $G$ is abelian $\Leftrightarrow$ $H$ is abelian
- $G$ is cyclic $\Leftrightarrow$ $H$ is cyclic
- Element orders are preserved
- Subgroup lattices are isomorphic

### Classification Results

**Theorem:** Every cyclic group of order $n$ is isomorphic to $\mathbb{Z}_n$.

**Theorem:** Up to isomorphism, there are exactly:
- 1 group of order 1 (trivial)
- 1 group of order 2 ($\mathbb{Z}_2$)
- 1 group of order 3 ($\mathbb{Z}_3$)
- 2 groups of order 4 ($\mathbb{Z}_4$ and $\mathbb{Z}_2 \times \mathbb{Z}_2$)
- 1 group of order 5 ($\mathbb{Z}_5$)
- 2 groups of order 6 ($\mathbb{Z}_6$ and $S_3$)

### Proving Non-Isomorphism

To show $G \not\cong H$, find a group property they don't share:
- Different orders
- One abelian, one not
- Different numbers of elements of each order
- Different numbers of subgroups

**Example:** $\mathbb{Z}_4 \not\cong \mathbb{Z}_2 \times \mathbb{Z}_2$

In $\mathbb{Z}_4$: element 1 has order 4.
In $\mathbb{Z}_2 \times \mathbb{Z}_2$: all non-identity elements have order 2.

No element of order 4, so not isomorphic!

---

## 6. Quantum Mechanics Connection

### Symmetry Transformations as Homomorphisms

In quantum mechanics, symmetry groups act on Hilbert space via **unitary representations**:

$$U: G \to U(\mathcal{H})$$

This is a homomorphism from a physical symmetry group to the group of unitary operators.

The homomorphism property says:
$$U(g_1 g_2) = U(g_1) U(g_2)$$

"Composing symmetry operations" = "multiplying their representations"

### Projective Representations

Sometimes we only get a **projective representation**:
$$U(g_1 g_2) = e^{i\phi(g_1, g_2)} U(g_1) U(g_2)$$

The phase $\phi$ is a **cocycle**, and projective representations of $G$ correspond to true representations of a **central extension** of $G$.

**Example:** SO(3) has projective representations that are true representations of SU(2)—this is why spin-1/2 exists!

### Kernel = Symmetries Acting Trivially

If $U: G \to U(\mathcal{H})$ is a representation:
$$\ker(U) = \{g \in G : U(g) = I\}$$

These are symmetries that act trivially on the quantum state—they're "gauge redundancies."

### First Isomorphism Theorem in Physics

The "effective symmetry group" is:
$$G/\ker(U) \cong \text{Im}(U)$$

This is the group of symmetries that actually do something to the states.

### Spontaneous Symmetry Breaking

When a ground state $|0\rangle$ is not invariant under full symmetry $G$:
- The **stabilizer** $H = \{g : U(g)|0\rangle = |0\rangle\}$ is a subgroup
- The broken symmetries correspond to cosets $G/H$
- Goldstone bosons parameterize $G/H$

---

## 7. Worked Examples

### Example 1: Verify Homomorphism and Find Kernel

Let $\phi: \mathbb{Z} \to \mathbb{Z}_6$ be $\phi(n) = 2n \mod 6$.

**Is this a homomorphism?**
$$\phi(a + b) = 2(a+b) \mod 6 = (2a + 2b) \mod 6 = (2a \mod 6) + (2b \mod 6) \mod 6 = \phi(a) + \phi(b)$$
Yes! ✓

**Kernel:**
$\ker(\phi) = \{n \in \mathbb{Z} : 2n \equiv 0 \pmod{6}\} = \{n : n \equiv 0 \pmod{3}\} = 3\mathbb{Z}$

**Image:**
$\text{Im}(\phi) = \{0, 2, 4\}$ (even elements of $\mathbb{Z}_6$)

**First Isomorphism Theorem:**
$\mathbb{Z}/3\mathbb{Z} \cong \{0, 2, 4\} \cong \mathbb{Z}_3$

### Example 2: Construct an Isomorphism

Show $(\mathbb{Z}_6, +) \cong (\mathbb{Z}_2 \times \mathbb{Z}_3, +)$.

**Define:** $\phi: \mathbb{Z}_6 \to \mathbb{Z}_2 \times \mathbb{Z}_3$ by $\phi(n) = (n \mod 2, n \mod 3)$.

**Homomorphism check:**
$\phi(a + b) = ((a+b) \mod 2, (a+b) \mod 3) = (a \mod 2 + b \mod 2, a \mod 3 + b \mod 3) = \phi(a) + \phi(b)$ ✓

**Bijective?**
- $\phi(0) = (0, 0)$, $\phi(1) = (1, 1)$, $\phi(2) = (0, 2)$
- $\phi(3) = (1, 0)$, $\phi(4) = (0, 1)$, $\phi(5) = (1, 2)$

All different, so injective. Same cardinality, so bijective. ✓

This is the **Chinese Remainder Theorem** for groups!

### Example 3: Kernel of Matrix Homomorphism

Let $\phi: GL_2(\mathbb{C}) \to \mathbb{C}^*$ be $\phi(A) = \det(A)$.

**Kernel:**
$$\ker(\phi) = \{A \in GL_2(\mathbb{C}) : \det(A) = 1\} = SL_2(\mathbb{C})$$

**First Isomorphism Theorem:**
$$GL_2(\mathbb{C}) / SL_2(\mathbb{C}) \cong \mathbb{C}^*$$

Every coset $A \cdot SL_2$ consists of all matrices with the same determinant as $A$.

---

## 8. Computational Lab

```python
"""
Day 283: Group Homomorphisms
Computing kernels, images, and verifying isomorphism theorems
"""

import numpy as np
from typing import List, Set, Dict, Callable, Optional, Tuple
from itertools import product

class Homomorphism:
    """
    Represents a group homomorphism and computes its properties.
    """

    def __init__(self, domain_elements: List, codomain_elements: List,
                 domain_op: Callable, codomain_op: Callable,
                 mapping: Callable):
        """
        Initialize a homomorphism.

        Parameters:
            domain_elements: Elements of source group G
            codomain_elements: Elements of target group H
            domain_op: Operation on G
            codomain_op: Operation on H
            mapping: The homomorphism function φ: G → H
        """
        self.G = list(domain_elements)
        self.H = list(codomain_elements)
        self.op_G = domain_op
        self.op_H = codomain_op
        self.phi = mapping

        # Find identities
        self.e_G = self._find_identity(self.G, self.op_G)
        self.e_H = self._find_identity(self.H, self.op_H)

        # Find inverses
        self.inv_G = self._find_inverses(self.G, self.op_G, self.e_G)
        self.inv_H = self._find_inverses(self.H, self.op_H, self.e_H)

    def _find_identity(self, elements, op):
        """Find identity element."""
        for e in elements:
            if all(op(e, a) == a and op(a, e) == a for a in elements):
                return e
        return None

    def _find_inverses(self, elements, op, identity):
        """Find inverse of each element."""
        inv = {}
        for a in elements:
            for b in elements:
                if op(a, b) == identity:
                    inv[a] = b
                    break
        return inv

    def verify_homomorphism(self) -> Tuple[bool, List]:
        """
        Check if mapping is a valid homomorphism.

        Returns:
            (is_valid, list of failures if any)
        """
        failures = []
        for a in self.G:
            for b in self.G:
                lhs = self.phi(self.op_G(a, b))
                rhs = self.op_H(self.phi(a), self.phi(b))
                if lhs != rhs:
                    failures.append((a, b, lhs, rhs))

        return (len(failures) == 0, failures)

    def kernel(self) -> Set:
        """Compute the kernel of the homomorphism."""
        return {g for g in self.G if self.phi(g) == self.e_H}

    def image(self) -> Set:
        """Compute the image of the homomorphism."""
        return {self.phi(g) for g in self.G}

    def is_injective(self) -> bool:
        """Check if homomorphism is injective (monomorphism)."""
        return self.kernel() == {self.e_G}

    def is_surjective(self) -> bool:
        """Check if homomorphism is surjective (epimorphism)."""
        return self.image() == set(self.H)

    def is_isomorphism(self) -> bool:
        """Check if homomorphism is an isomorphism."""
        return self.is_injective() and self.is_surjective()

    def verify_kernel_normal(self) -> bool:
        """Verify that ker(φ) is a normal subgroup of G."""
        K = self.kernel()

        # Check gKg^(-1) ⊆ K for all g
        for g in self.G:
            g_inv = self.inv_G[g]
            for k in K:
                conjugate = self.op_G(self.op_G(g, k), g_inv)
                if conjugate not in K:
                    return False
        return True

    def cosets_of_kernel(self) -> List[Set]:
        """Compute cosets of kernel in G."""
        K = self.kernel()
        cosets = []
        covered = set()

        for g in self.G:
            if g in covered:
                continue
            coset = {self.op_G(g, k) for k in K}
            cosets.append(coset)
            covered.update(coset)

        return cosets

    def verify_first_isomorphism_theorem(self) -> Dict:
        """
        Verify the First Isomorphism Theorem: G/ker(φ) ≅ Im(φ)
        """
        K = self.kernel()
        cosets = self.cosets_of_kernel()
        im = self.image()

        # Map each coset to its image (should be single element)
        coset_to_image = {}
        for coset in cosets:
            images = {self.phi(g) for g in coset}
            if len(images) == 1:
                coset_to_image[frozenset(coset)] = images.pop()
            else:
                return {
                    'valid': False,
                    'error': f'Coset {coset} maps to multiple images: {images}'
                }

        # Check bijection with image
        image_values = set(coset_to_image.values())
        bijection = (image_values == im and len(coset_to_image) == len(im))

        return {
            'valid': bijection,
            'kernel': K,
            'num_cosets': len(cosets),
            'image': im,
            'image_size': len(im),
            '|G|/|ker|': len(self.G) // len(K) if K else None,
            'isomorphism_verified': bijection
        }


def create_homomorphism_Zn_to_Zm(n: int, m: int, multiplier: int) -> Homomorphism:
    """
    Create homomorphism Z_n → Z_m: k ↦ (multiplier * k) mod m
    """
    G = list(range(n))
    H = list(range(m))
    op_G = lambda a, b: (a + b) % n
    op_H = lambda a, b: (a + b) % m
    phi = lambda k: (multiplier * k) % m

    return Homomorphism(G, H, op_G, op_H, phi)


def create_det_homomorphism(matrices: List[np.ndarray]) -> Homomorphism:
    """
    Create determinant homomorphism for a list of matrices.
    """
    # Compute determinants
    dets = sorted(list(set(round(np.linalg.det(A).real, 10) for A in matrices)))

    op_G = lambda A, B: A @ B
    op_H = lambda a, b: round(a * b, 10)
    phi = lambda A: round(np.linalg.det(A).real, 10)

    return Homomorphism(matrices, dets, op_G, op_H, phi)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("GROUP HOMOMORPHISMS")
    print("=" * 60)

    # Example 1: Z_12 → Z_4
    print("\n1. HOMOMORPHISM Z_12 → Z_4: k ↦ k mod 4")
    print("-" * 40)

    hom1 = create_homomorphism_Zn_to_Zm(12, 4, 1)
    valid, failures = hom1.verify_homomorphism()

    print(f"Is homomorphism: {valid}")
    print(f"Kernel: {hom1.kernel()}")
    print(f"Image: {hom1.image()}")
    print(f"Injective: {hom1.is_injective()}")
    print(f"Surjective: {hom1.is_surjective()}")

    fit = hom1.verify_first_isomorphism_theorem()
    print(f"\nFirst Isomorphism Theorem:")
    print(f"  |ker| = {len(fit['kernel'])}")
    print(f"  |G/ker| = {fit['num_cosets']}")
    print(f"  |Im| = {fit['image_size']}")
    print(f"  Z_12 / ker ≅ Im: {fit['isomorphism_verified']}")

    # Example 2: Z_6 → Z_6 with multiplier 2
    print("\n2. HOMOMORPHISM Z_6 → Z_6: k ↦ 2k mod 6")
    print("-" * 40)

    hom2 = create_homomorphism_Zn_to_Zm(6, 6, 2)
    valid, _ = hom2.verify_homomorphism()

    print(f"Is homomorphism: {valid}")
    print(f"Kernel: {hom2.kernel()}")
    print(f"Image: {hom2.image()}")
    print(f"Kernel is normal: {hom2.verify_kernel_normal()}")

    # Example 3: Trivial homomorphism
    print("\n3. TRIVIAL HOMOMORPHISM Z_5 → Z_3: k ↦ 0")
    print("-" * 40)

    G = list(range(5))
    H = list(range(3))
    trivial = Homomorphism(G, H,
                           lambda a, b: (a + b) % 5,
                           lambda a, b: (a + b) % 3,
                           lambda k: 0)

    print(f"Kernel: {trivial.kernel()} (= entire domain)")
    print(f"Image: {trivial.image()} (= identity only)")

    # Example 4: Isomorphism Z_6 ≅ Z_2 × Z_3
    print("\n4. ISOMORPHISM Z_6 ≅ Z_2 × Z_3")
    print("-" * 40)

    G = list(range(6))
    H = [(i, j) for i in range(2) for j in range(3)]

    def op_G(a, b):
        return (a + b) % 6

    def op_H(a, b):
        return ((a[0] + b[0]) % 2, (a[1] + b[1]) % 3)

    def phi(k):
        return (k % 2, k % 3)

    iso = Homomorphism(G, H, op_G, op_H, phi)
    valid, _ = iso.verify_homomorphism()

    print(f"Is homomorphism: {valid}")
    print(f"Is isomorphism: {iso.is_isomorphism()}")
    print(f"\nMapping:")
    for k in range(6):
        print(f"  {k} ↦ {phi(k)}")

    # Example 5: Sign homomorphism S_3 → {1, -1}
    print("\n5. SIGN HOMOMORPHISM S_3 → {1, -1}")
    print("-" * 40)

    from itertools import permutations

    def perm_sign(p):
        """Compute sign of permutation."""
        n = len(p)
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if p[i] > p[j]:
                    inversions += 1
        return 1 if inversions % 2 == 0 else -1

    perms = list(permutations(range(3)))
    signs = [1, -1]

    def compose(p1, p2):
        return tuple(p1[p2[i]] for i in range(3))

    sign_hom = Homomorphism(perms, signs,
                            compose,
                            lambda a, b: a * b,
                            perm_sign)

    valid, _ = sign_hom.verify_homomorphism()
    kernel = sign_hom.kernel()

    print(f"Is homomorphism: {valid}")
    print(f"Kernel (alternating group A_3):")
    for p in kernel:
        print(f"  {p}")
    print(f"|ker| = {len(kernel)}, |S_3|/|ker| = {6 // len(kernel)} = |{{1,-1}}|")

    # Example 6: Matrix determinant
    print("\n6. DETERMINANT HOMOMORPHISM")
    print("-" * 40)

    # Create some 2x2 matrices with det = ±1
    matrices = [
        np.array([[1, 0], [0, 1]]),   # I, det=1
        np.array([[-1, 0], [0, -1]]), # -I, det=1
        np.array([[1, 0], [0, -1]]),  # det=-1
        np.array([[-1, 0], [0, 1]]),  # det=-1
    ]

    det_hom = create_det_homomorphism(matrices)
    valid, _ = det_hom.verify_homomorphism()

    print(f"Is homomorphism: {valid}")
    print(f"Image (determinant values): {det_hom.image()}")
    print(f"Kernel (matrices with det=1):")
    for A in det_hom.kernel():
        print(f"  {A.tolist()}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Homomorphisms preserve group structure: φ(ab) = φ(a)φ(b)
    2. Kernel is ALWAYS a normal subgroup
    3. First Isomorphism Theorem: G/ker(φ) ≅ Im(φ)
    4. Isomorphic groups have identical algebraic structure
    5. In QM: representations are homomorphisms G → U(H)
    6. Kernel = symmetries that act trivially on states
    """)
```

---

## 9. Practice Problems

### Problem Set A: Basic Computations

**A1.** Verify that $\phi: \mathbb{Z} \to \mathbb{Z}$ defined by $\phi(n) = 3n$ is a homomorphism. Find its kernel and image.

**A2.** For $\phi: GL_2(\mathbb{R}) \to \mathbb{R}^*$ with $\phi(A) = \det(A)$, verify the homomorphism property and identify $\ker(\phi)$.

**A3.** Show that $\phi: (\mathbb{R}, +) \to (\mathbb{R}^+, \times)$ given by $\phi(x) = 2^x$ is an isomorphism.

### Problem Set B: Isomorphism Theorems

**B1.** Use the First Isomorphism Theorem to prove $\mathbb{Z}/6\mathbb{Z} \cong \mathbb{Z}_6$.

**B2.** For the sign homomorphism $\text{sgn}: S_4 \to \{1, -1\}$, apply the First Isomorphism Theorem to conclude $S_4/A_4 \cong \mathbb{Z}_2$.

**B3.** Prove that $(\mathbb{R}^*, \times) / \{1, -1\} \cong (\mathbb{R}^+, \times)$.

### Problem Set C: Advanced

**C1.** Show that if $\phi: G \to H$ is a homomorphism and $G$ is abelian, then $\text{Im}(\phi)$ is abelian.

**C2.** Prove: The only homomorphism from $\mathbb{Z}_n$ to $\mathbb{Z}_m$ is trivial if $\gcd(n, m) = 1$.

**C3.** **(Physics)** In gauge theory, gauge transformations form a group $\mathcal{G}$. Physical states live in a Hilbert space $\mathcal{H}$. Explain why "gauge-invariant states" form the kernel of a certain action.

---

## 10. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| Homomorphism | $\phi(ab) = \phi(a)\phi(b)$ |
| Kernel | $\ker(\phi) = \{g : \phi(g) = e\}$ |
| Image | $\text{Im}(\phi) = \{\phi(g) : g \in G\}$ |
| Isomorphism | Bijective homomorphism |

### Key Theorems

$$\boxed{\ker(\phi) \trianglelefteq G}$$

$$\boxed{G/\ker(\phi) \cong \text{Im}(\phi)}$$

$$\boxed{\phi \text{ injective} \Leftrightarrow \ker(\phi) = \{e\}}$$

---

## 11. Preview: Day 284

Tomorrow we study **quotient groups**:
- Construction of $G/N$ for normal $N$
- Well-definedness of coset multiplication
- The quotient homomorphism $\pi: G \to G/N$
- Physical interpretation: modding out by symmetries

---

*"The study of homomorphisms is the study of how groups relate to each other—the grand architecture of algebra." — Emmy Noether*
