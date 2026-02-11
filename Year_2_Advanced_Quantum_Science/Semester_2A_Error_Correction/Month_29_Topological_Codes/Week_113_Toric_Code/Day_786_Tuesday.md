# Day 786: Star and Plaquette Operators

## Overview

**Day:** 786 of 1008
**Week:** 113 (Toric Code Fundamentals)
**Month:** 29 (Topological Codes)
**Topic:** Star and Plaquette Operators in the Toric Code

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Star operator construction and properties |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Plaquette operators and commutation proofs |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Construct** star operators $A_v = \prod_{e \ni v} X_e$ at each vertex
2. **Construct** plaquette operators $B_p = \prod_{e \in \partial p} Z_e$ at each face
3. **Prove** that all star operators commute with each other
4. **Prove** that all plaquette operators commute with each other
5. **Prove** that star and plaquette operators commute: $[A_v, B_p] = 0$
6. **Identify** the stabilizer group structure and constraints

---

## Core Theory

### 1. Star Operators (Vertex Operators)

At each vertex $v$ of the toric lattice, we define the **star operator**:

$$\boxed{A_v = \prod_{e \ni v} X_e}$$

where the product is over all edges $e$ incident to vertex $v$.

**Geometric interpretation:** Apply Pauli-X to all 4 edges meeting at vertex $v$.

```
           X
           │
     X ────●──── X
           │
           X

A_v = X_left · X_right · X_up · X_down
```

**Properties of $A_v$:**
- **Hermitian:** $A_v^\dagger = A_v$
- **Unitary:** $A_v A_v^\dagger = I$
- **Involutory:** $A_v^2 = I$ (since $X^2 = I$)
- **Eigenvalues:** $\pm 1$ only

### 2. Plaquette Operators (Face Operators)

At each face/plaquette $p$ of the lattice, we define the **plaquette operator**:

$$\boxed{B_p = \prod_{e \in \partial p} Z_e}$$

where $\partial p$ denotes the boundary of plaquette $p$ (the 4 edges forming its perimeter).

**Geometric interpretation:** Apply Pauli-Z to all 4 edges forming the boundary of face $p$.

```
     ┌───Z───┐
     │       │
     Z   p   Z
     │       │
     └───Z───┘

B_p = Z_top · Z_bottom · Z_left · Z_right
```

**Properties of $B_p$:**
- **Hermitian:** $B_p^\dagger = B_p$
- **Unitary:** $B_p B_p^\dagger = I$
- **Involutory:** $B_p^2 = I$ (since $Z^2 = I$)
- **Eigenvalues:** $\pm 1$ only

### 3. Explicit Operator Expressions

For vertex $(i, j)$ with edge indexing $(i, j, d)$:

$$A_{(i,j)} = X_{(i,j,0)} \cdot X_{(i,j-1,0)} \cdot X_{(i,j,1)} \cdot X_{(i-1,j,1)}$$

where indices are taken mod $L$.

For plaquette $(i, j)$ (top-left corner at vertex $(i, j)$):

$$B_{(i,j)} = Z_{(i,j,0)} \cdot Z_{(i+1,j,0)} \cdot Z_{(i,j,1)} \cdot Z_{(i,j+1,1)}$$

### 4. Commutation: Star Operators

**Theorem:** All star operators commute: $[A_v, A_{v'}] = 0$ for all vertices $v, v'$.

**Proof:**

Each $A_v$ is a product of X operators. X operators on different qubits commute:
$$[X_e, X_{e'}] = 0 \text{ for } e \neq e'$$

X operators on the same qubit satisfy $X_e \cdot X_e = I$.

Consider $A_v$ and $A_{v'}$:
- If $v$ and $v'$ share no edges: All X operators act on different qubits, so they commute.
- If $v$ and $v'$ are adjacent (share one edge $e$): Both $A_v$ and $A_{v'}$ contain $X_e$, which appears twice in the product $A_v A_{v'}$, giving $X_e^2 = I$.

In either case:
$$A_v A_{v'} = A_{v'} A_v \quad \Rightarrow \quad [A_v, A_{v'}] = 0 \quad \blacksquare$$

### 5. Commutation: Plaquette Operators

**Theorem:** All plaquette operators commute: $[B_p, B_{p'}] = 0$ for all faces $p, p'$.

**Proof:**

Identical argument to star operators, using $Z^2 = I$ and $[Z_e, Z_{e'}] = 0$.

Adjacent plaquettes share exactly one edge, where $Z_e$ appears in both operators:
$$B_p B_{p'} \text{ contains } Z_e^2 = I$$

Therefore $[B_p, B_{p'}] = 0$. $\blacksquare$

### 6. Commutation: Stars and Plaquettes

**Theorem:** Star and plaquette operators commute: $[A_v, B_p] = 0$ for all $v, p$.

**Proof:**

This is the crucial result. We use the commutation relation:
$$XZ = -ZX \quad \text{(anti-commute on same qubit)}$$

Count the number of shared qubits between $A_v$ and $B_p$:

**Case 1:** Vertex $v$ is not on the boundary of plaquette $p$.

Then $A_v$ and $B_p$ share no qubits. X and Z on different qubits commute, so $[A_v, B_p] = 0$.

**Case 2:** Vertex $v$ is on the boundary of plaquette $p$.

Each vertex of a plaquette has exactly **2 edges** that are both incident to the vertex and on the plaquette boundary.

```
    v ──────
    │
    │   p
    │
```

At vertex $v$ on corner of $p$: edges going right and down (from $v$) are shared.

**Shared edges = 2**

When computing $A_v B_p$:
$$A_v B_p = (X_{e_1} X_{e_2} \cdots)(Z_{e_1} Z_{e_2} \cdots)$$

For each shared edge $e$: $X_e Z_e = -Z_e X_e$

With 2 shared edges:
$$A_v B_p = (-1)^2 B_p A_v = B_p A_v$$

Therefore $[A_v, B_p] = 0$. $\blacksquare$

**Key insight:** The commutation relies on there being an **even number** of shared qubits, which is guaranteed by the lattice geometry.

### 7. Stabilizer Group Structure

The star and plaquette operators generate the **stabilizer group** $S$:

$$S = \langle A_v, B_p : v \in V, p \in F \rangle$$

**Number of generators:**
- Star operators: $L^2$ (one per vertex)
- Plaquette operators: $L^2$ (one per face)
- Total: $2L^2$ generators

**But these are not independent!**

**Constraint 1:** Product of all star operators
$$\prod_{v} A_v = I$$

Each edge appears in exactly 2 star operators (its two endpoints), so:
$$\prod_v A_v = \prod_e X_e^2 = I$$

**Constraint 2:** Product of all plaquette operators
$$\prod_{p} B_p = I$$

Each edge appears in exactly 2 plaquette operators (the two faces it borders), so:
$$\prod_p B_p = \prod_e Z_e^2 = I$$

**Independent generators:** $2L^2 - 2$

### 8. Number of Logical Qubits

Using the stabilizer formalism:

$$k = n - \text{(number of independent stabilizers)}$$

$$k = 2L^2 - (2L^2 - 2) = 2$$

This confirms $k = 2$ logical qubits, matching the topological argument from Day 785.

---

## Quantum Mechanics Connection

### Stabilizer Formalism Review

A stabilizer code is defined by an abelian subgroup $S$ of the Pauli group such that $-I \notin S$.

The code space is:
$$\mathcal{C} = \{|\psi\rangle : s|\psi\rangle = |\psi\rangle \text{ for all } s \in S\}$$

For the toric code:
- $S$ is generated by $\{A_v, B_p\}$
- Ground states satisfy $A_v|\psi\rangle = +|\psi\rangle$ and $B_p|\psi\rangle = +|\psi\rangle$

### Local Hamiltonian Perspective

The toric code Hamiltonian:
$$H = -\sum_v A_v - \sum_p B_p$$

**Ground state energy:** $E_0 = -2L^2$ (all operators have eigenvalue +1)

**Excited states:** Violating one stabilizer costs energy +2 (flipping eigenvalue from +1 to -1).

The energy gap protects the ground space from thermal excitations.

### Connection to Error Detection

- **X error on edge $e$:** Anti-commutes with the two plaquette operators containing $e$
- **Z error on edge $e$:** Anti-commutes with the two star operators containing $e$

Errors create pairs of "defects" (stabilizer violations) at their endpoints.

---

## Worked Examples

### Example 1: Construct Operators for L = 2

**Problem:** Write explicit star and plaquette operators for a $2 \times 2$ toric code.

**Solution:**

Qubits (8 total):
- Horizontal: $e_0 = (0,0,0)$, $e_1 = (0,1,0)$, $e_2 = (1,0,0)$, $e_3 = (1,1,0)$
- Vertical: $e_4 = (0,0,1)$, $e_5 = (0,1,1)$, $e_6 = (1,0,1)$, $e_7 = (1,1,1)$

**Star operators:**

$A_{(0,0)} = X_{e_0} X_{e_1} X_{e_4} X_{e_6}$ (edges: right, left, down, up with periodicity)

Wait, let's be more careful. For vertex $(0,0)$:
- Right horizontal: $(0,0,0) = e_0$
- Left horizontal: $(0,-1,0) = (0,1,0) = e_1$ (periodic)
- Down vertical: $(0,0,1) = e_4$
- Up vertical: $(-1,0,1) = (1,0,1) = e_6$ (periodic)

$$A_{(0,0)} = X_0 X_1 X_4 X_6$$

Similarly for other vertices.

**Plaquette operators:**

For plaquette $(0,0)$ (top-left at vertex $(0,0)$):
- Top: $(0,0,0) = e_0$
- Bottom: $(1,0,0) = e_2$
- Left: $(0,0,1) = e_4$
- Right: $(0,1,1) = e_5$

$$B_{(0,0)} = Z_0 Z_2 Z_4 Z_5$$

### Example 2: Verify Commutation

**Problem:** Verify $[A_{(0,0)}, B_{(0,0)}] = 0$ for the $2 \times 2$ toric code.

**Solution:**

From Example 1:
- $A_{(0,0)} = X_0 X_1 X_4 X_6$
- $B_{(0,0)} = Z_0 Z_2 Z_4 Z_5$

Find shared qubits:
- Qubit 0: in both
- Qubit 4: in both
- Total shared: 2

Compute commutation:
$$A_{(0,0)} B_{(0,0)} = (X_0 X_1 X_4 X_6)(Z_0 Z_2 Z_4 Z_5)$$

The X's and Z's on qubits 0 and 4 anti-commute:
$$= (-1)^2 (Z_0 Z_2 Z_4 Z_5)(X_0 X_1 X_4 X_6) = B_{(0,0)} A_{(0,0)}$$

Therefore $[A_{(0,0)}, B_{(0,0)}] = 0$. $\checkmark$

### Example 3: Count Independent Stabilizers

**Problem:** For $L = 3$, count independent stabilizer generators.

**Solution:**

Total generators: $L^2 + L^2 = 9 + 9 = 18$

Constraints:
- $\prod_v A_v = I$ (1 constraint)
- $\prod_p B_p = I$ (1 constraint)

Independent generators: $18 - 2 = 16$

Physical qubits: $n = 2L^2 = 18$

Logical qubits: $k = 18 - 16 = 2$ $\checkmark$

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For a $3 \times 3$ toric code, write the star operator $A_{(1,1)}$ in terms of edge indices.

**P1.2** For the same code, write the plaquette operator $B_{(0,2)}$.

**P1.3** How many edges does each star operator act on? How many edges does each plaquette operator act on?

### Level 2: Intermediate

**P2.1** Prove that if vertex $v$ is NOT adjacent to plaquette $p$ (not on its boundary), then $A_v$ and $B_p$ share no qubits.

**P2.2** Show that for any two adjacent vertices $v$ and $v'$, they share exactly one edge. Conclude that $A_v A_{v'} = A_{v'} A_v$.

**P2.3** Verify the constraint $\prod_v A_v = I$ by showing each edge is counted exactly twice in the product.

### Level 3: Challenging

**P3.1** For a rectangular torus $L_1 \times L_2$, determine:
a) Number of star and plaquette operators
b) Number of constraints
c) Number of logical qubits

**P3.2** Define a modified operator $\tilde{A}_v = i \prod_{e \ni v} X_e$. Does this still commute with all $B_p$? Does it change the code?

**P3.3** Consider a triangular lattice on a torus (vertices have 6 neighbors). How would you define star and plaquette operators? Do they still commute?

---

## Computational Lab

```python
"""
Day 786: Star and Plaquette Operators
======================================

Implementing and verifying toric code stabilizers.
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from functools import reduce
from operator import xor


class ToricCodeStabilizers:
    """
    Toric code stabilizer operators on L x L lattice.

    Implements star and plaquette operators with verification
    of commutation relations.
    """

    def __init__(self, L: int):
        """Initialize L x L toric code."""
        self.L = L
        self.n_qubits = 2 * L * L
        self.n_vertices = L * L
        self.n_faces = L * L

    def edge_index(self, i: int, j: int, d: int) -> int:
        """Convert (i, j, d) to linear edge index."""
        i, j = i % self.L, j % self.L
        return d * self.L**2 + i * self.L + j

    def star_support(self, vi: int, vj: int) -> Set[int]:
        """
        Return set of edge indices for star operator at vertex (vi, vj).

        Star operator: A_v = prod_{e ni v} X_e
        """
        vi, vj = vi % self.L, vj % self.L
        return {
            self.edge_index(vi, vj, 0),       # right horizontal
            self.edge_index(vi, vj - 1, 0),   # left horizontal
            self.edge_index(vi, vj, 1),       # down vertical
            self.edge_index(vi - 1, vj, 1),   # up vertical
        }

    def plaquette_support(self, pi: int, pj: int) -> Set[int]:
        """
        Return set of edge indices for plaquette operator at face (pi, pj).

        Plaquette operator: B_p = prod_{e in boundary(p)} Z_e
        """
        pi, pj = pi % self.L, pj % self.L
        return {
            self.edge_index(pi, pj, 0),       # top horizontal
            self.edge_index(pi + 1, pj, 0),   # bottom horizontal
            self.edge_index(pi, pj, 1),       # left vertical
            self.edge_index(pi, pj + 1, 1),   # right vertical
        }

    def all_stars(self) -> List[Set[int]]:
        """Return list of all star operator supports."""
        return [self.star_support(i, j)
                for i in range(self.L) for j in range(self.L)]

    def all_plaquettes(self) -> List[Set[int]]:
        """Return list of all plaquette operator supports."""
        return [self.plaquette_support(i, j)
                for i in range(self.L) for j in range(self.L)]

    def commutation_check(self, support_x: Set[int], support_z: Set[int]) -> int:
        """
        Check commutation of X-type and Z-type operators.

        Returns 0 if they commute, 1 if they anti-commute.

        X and Z anti-commute on same qubit, so:
        [prod_i X_i, prod_j Z_j] = 0 iff |support_x ∩ support_z| is even
        """
        overlap = len(support_x & support_z)
        return overlap % 2

    def verify_all_commutations(self) -> Dict[str, bool]:
        """Verify all stabilizer commutation relations."""
        stars = self.all_stars()
        plaquettes = self.all_plaquettes()

        results = {
            'stars_commute': True,
            'plaquettes_commute': True,
            'stars_plaquettes_commute': True,
        }

        # Stars with stars (X-X always commute)
        # No check needed: X operators always commute

        # Plaquettes with plaquettes (Z-Z always commute)
        # No check needed: Z operators always commute

        # Stars with plaquettes (X-Z need even overlap)
        for i, star in enumerate(stars):
            for j, plaq in enumerate(plaquettes):
                if self.commutation_check(star, plaq) != 0:
                    results['stars_plaquettes_commute'] = False
                    print(f"FAIL: Star {i} anti-commutes with Plaquette {j}")
                    print(f"  Star support: {star}")
                    print(f"  Plaquette support: {plaq}")
                    print(f"  Overlap: {star & plaq}")

        return results

    def verify_constraints(self) -> Dict[str, bool]:
        """Verify that product of all stars = I and product of all plaquettes = I."""
        results = {}

        # Product of all stars
        all_star_edges: Set[int] = set()
        for star in self.all_stars():
            all_star_edges ^= star  # XOR for mod-2 addition

        results['star_product_is_identity'] = len(all_star_edges) == 0

        # Product of all plaquettes
        all_plaq_edges: Set[int] = set()
        for plaq in self.all_plaquettes():
            all_plaq_edges ^= plaq

        results['plaquette_product_is_identity'] = len(all_plaq_edges) == 0

        return results

    def stabilizer_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return stabilizer check matrices H_X and H_Z.

        H_X[i, j] = 1 if X_j appears in i-th X-stabilizer (star)
        H_Z[i, j] = 1 if Z_j appears in i-th Z-stabilizer (plaquette)
        """
        H_X = np.zeros((self.n_vertices, self.n_qubits), dtype=int)
        H_Z = np.zeros((self.n_faces, self.n_qubits), dtype=int)

        for idx, star in enumerate(self.all_stars()):
            for edge in star:
                H_X[idx, edge] = 1

        for idx, plaq in enumerate(self.all_plaquettes()):
            for edge in plaq:
                H_Z[idx, edge] = 1

        return H_X, H_Z

    def print_operator(self, support: Set[int], op_type: str = 'X') -> str:
        """Create string representation of operator."""
        chars = ['I'] * self.n_qubits
        for idx in support:
            chars[idx] = op_type
        return ''.join(chars)


def visualize_operators(L: int, vertex: Tuple[int, int] = None,
                       face: Tuple[int, int] = None) -> None:
    """Visualize star and plaquette operators."""
    code = ToricCodeStabilizers(L)

    print(f"\nToric Code L = {L}")
    print(f"Qubits: {code.n_qubits}")
    print(f"Edge indexing: [horizontal 0..{L**2-1}][vertical {L**2}..{2*L**2-1}]")

    if vertex:
        vi, vj = vertex
        star = code.star_support(vi, vj)
        print(f"\nStar operator A_({vi},{vj}):")
        print(f"  Support (edge indices): {sorted(star)}")
        print(f"  Operator: {code.print_operator(star, 'X')}")

    if face:
        pi, pj = face
        plaq = code.plaquette_support(pi, pj)
        print(f"\nPlaquette operator B_({pi},{pj}):")
        print(f"  Support (edge indices): {sorted(plaq)}")
        print(f"  Operator: {code.print_operator(plaq, 'Z')}")

    if vertex and face:
        star = code.star_support(*vertex)
        plaq = code.plaquette_support(*face)
        overlap = star & plaq
        print(f"\nOverlap analysis:")
        print(f"  Shared qubits: {sorted(overlap)}")
        print(f"  Number of shared qubits: {len(overlap)}")
        print(f"  Commutation: {'Commute' if len(overlap) % 2 == 0 else 'Anti-commute'}")


def demonstrate_commutation(L: int) -> None:
    """Demonstrate commutation verification."""
    print(f"\n{'='*60}")
    print(f"Commutation Verification for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeStabilizers(L)

    # Verify commutation
    results = code.verify_all_commutations()
    print("\nCommutation results:")
    for key, value in results.items():
        status = "PASS" if value else "FAIL"
        print(f"  {key}: {status}")

    # Verify constraints
    constraints = code.verify_constraints()
    print("\nConstraint verification:")
    for key, value in constraints.items():
        status = "PASS" if value else "FAIL"
        print(f"  {key}: {status}")

    # Count independent generators
    n_generators = 2 * L**2
    n_constraints = 2
    n_independent = n_generators - n_constraints
    k = code.n_qubits - n_independent

    print(f"\nStabilizer counting:")
    print(f"  Total generators: {n_generators}")
    print(f"  Constraints: {n_constraints}")
    print(f"  Independent generators: {n_independent}")
    print(f"  Logical qubits k = n - r = {code.n_qubits} - {n_independent} = {k}")


def analyze_overlap_patterns(L: int) -> None:
    """Analyze overlap patterns between stars and plaquettes."""
    print(f"\n{'='*60}")
    print(f"Overlap Pattern Analysis for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeStabilizers(L)
    stars = code.all_stars()
    plaquettes = code.all_plaquettes()

    overlap_counts = {}
    for i, star in enumerate(stars):
        for j, plaq in enumerate(plaquettes):
            overlap = len(star & plaq)
            if overlap not in overlap_counts:
                overlap_counts[overlap] = 0
            overlap_counts[overlap] += 1

    print("\nOverlap distribution (star-plaquette pairs):")
    for overlap, count in sorted(overlap_counts.items()):
        parity = "even (commute)" if overlap % 2 == 0 else "odd (anti-commute)"
        print(f"  {overlap} shared qubits: {count} pairs - {parity}")

    # Verify all overlaps are even
    all_even = all(overlap % 2 == 0 for overlap in overlap_counts.keys())
    print(f"\nAll overlaps even: {all_even}")


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 786: STAR AND PLAQUETTE OPERATORS")
    print("=" * 70)

    # Demo 1: Visualize specific operators
    print("\n" + "=" * 70)
    print("Demo 1: Operator Visualization")
    print("=" * 70)

    visualize_operators(3, vertex=(1, 1), face=(1, 1))
    visualize_operators(3, vertex=(0, 0), face=(0, 0))

    # Demo 2: Verify commutation
    print("\n" + "=" * 70)
    print("Demo 2: Commutation Verification")
    print("=" * 70)

    for L in [2, 3, 4, 5]:
        demonstrate_commutation(L)

    # Demo 3: Overlap analysis
    print("\n" + "=" * 70)
    print("Demo 3: Overlap Pattern Analysis")
    print("=" * 70)

    analyze_overlap_patterns(3)
    analyze_overlap_patterns(5)

    # Demo 4: Stabilizer matrices
    print("\n" + "=" * 70)
    print("Demo 4: Stabilizer Check Matrices")
    print("=" * 70)

    code = ToricCodeStabilizers(2)
    H_X, H_Z = code.stabilizer_matrix()

    print(f"\nFor L = 2 ({code.n_qubits} qubits):")
    print(f"\nH_X (star operators, {H_X.shape[0]} rows):")
    print(H_X)
    print(f"\nH_Z (plaquette operators, {H_Z.shape[0]} rows):")
    print(H_Z)

    # Verify symplectic orthogonality: H_X @ H_Z.T = 0 mod 2
    product = (H_X @ H_Z.T) % 2
    print(f"\nH_X @ H_Z^T mod 2 (should be all zeros):")
    print(product)
    print(f"Symplectic orthogonality verified: {np.all(product == 0)}")

    # Demo 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Star and Plaquette Operators")
    print("=" * 70)

    print("""
    STAR OPERATORS (vertices):
    --------------------------
    A_v = prod_{e ni v} X_e

    - Act on 4 edges meeting at vertex v
    - X-type stabilizers
    - L^2 operators with 1 constraint (product = I)

    PLAQUETTE OPERATORS (faces):
    ----------------------------
    B_p = prod_{e in boundary(p)} Z_e

    - Act on 4 edges bounding face p
    - Z-type stabilizers
    - L^2 operators with 1 constraint (product = I)

    COMMUTATION:
    ------------
    [A_v, B_p] = 0 for all v, p

    Key: Each (star, plaquette) pair shares 0 or 2 qubits
         2 is even, so X's and Z's commute overall

    STABILIZER COUNT:
    -----------------
    Generators: 2L^2
    Constraints: 2
    Independent: 2L^2 - 2
    Logical qubits: n - (2L^2 - 2) = 2L^2 - 2L^2 + 2 = 2
    """)

    print("=" * 70)
    print("Day 786 Complete: Star and Plaquette Operators Mastered")
    print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Star operator | $A_v = \prod_{e \ni v} X_e$ |
| Plaquette operator | $B_p = \prod_{e \in \partial p} Z_e$ |
| Star-Star commutation | $[A_v, A_{v'}] = 0$ |
| Plaquette-Plaquette commutation | $[B_p, B_{p'}] = 0$ |
| Star-Plaquette commutation | $[A_v, B_p] = 0$ |
| Star constraint | $\prod_v A_v = I$ |
| Plaquette constraint | $\prod_p B_p = I$ |
| Independent generators | $2L^2 - 2$ |

### Main Takeaways

1. **Star operators** $A_v = \prod_{e \ni v} X_e$ are X-type stabilizers at vertices
2. **Plaquette operators** $B_p = \prod_{e \in \partial p} Z_e$ are Z-type stabilizers at faces
3. **All operators commute** because star-plaquette pairs share an even number (0 or 2) of qubits
4. **Two constraints** ($\prod_v A_v = I$ and $\prod_p B_p = I$) reduce independent generators
5. **Logical qubits** $k = n - (2L^2 - 2) = 2$, confirming topological counting

---

## Daily Checklist

- [ ] I can construct star operators for any vertex
- [ ] I can construct plaquette operators for any face
- [ ] I can prove star-plaquette commutation using qubit overlap
- [ ] I understand why the product of all stars equals identity
- [ ] I can count independent stabilizer generators
- [ ] I ran the computational lab and verified commutation

---

## Preview: Day 787

Tomorrow we explore the **ground state and code space**:

- Ground state condition: $A_v|\psi\rangle = B_p|\psi\rangle = +1|\psi\rangle$
- Ground state degeneracy = 4 on the torus
- Constructing the 4 ground states explicitly
- Toric code Hamiltonian and energy gap
- Logical qubits from topology

The ground states form the code space where logical information is protected.
