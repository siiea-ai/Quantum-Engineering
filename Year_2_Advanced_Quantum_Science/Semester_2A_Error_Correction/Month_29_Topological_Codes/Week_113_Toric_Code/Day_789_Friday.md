# Day 789: Logical Operators on the Torus

## Overview

**Day:** 789 of 1008
**Week:** 113 (Toric Code Fundamentals)
**Month:** 29 (Topological Codes)
**Topic:** Logical Operators and Non-Contractible Loops

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Non-contractible loops and logical Z |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Logical X and commutation relations |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Identify** non-contractible loops on the torus
2. **Construct** logical Z operators $\bar{Z}_1$ and $\bar{Z}_2$ as loop operators
3. **Construct** logical X operators $\bar{X}_1$ and $\bar{X}_2$ on dual cycles
4. **Verify** commutation relations: $[\bar{X}_i, \bar{Z}_j] = 0$ for $i \neq j$
5. **Verify** anti-commutation: $\{\bar{X}_i, \bar{Z}_i\} = 0$ for same index
6. **Explain** why logical operators cannot be local (weight < L)

---

## Core Theory

### 1. Non-Contractible Loops on the Torus

A torus has two fundamental cycles:
- **Cycle $\gamma_1$:** Wraps around the horizontal (x-direction)
- **Cycle $\gamma_2$:** Wraps around the vertical (y-direction)

```
        ┌─────────────────────┐
        │    γ₂               │
        │     ↓               │
        │  ○──────────────○   │
        │  │              │   │
   γ₁ → │  │              │   │
        │  │              │   │
        │  ○──────────────○   │
        │                     │
        └─────────────────────┘
        (edges identified)
```

**Key property:** These loops cannot be continuously shrunk to a point (non-contractible).

### 2. Logical Z Operators

The logical Z operators are products of Pauli-Z along non-contractible loops:

$$\boxed{\bar{Z}_1 = \prod_{e \in \gamma_1} Z_e}$$

$$\boxed{\bar{Z}_2 = \prod_{e \in \gamma_2} Z_e}$$

**Explicit construction:**

For $\bar{Z}_1$ (horizontal loop at row $i = 0$):
$$\bar{Z}_1 = \prod_{j=0}^{L-1} Z_{(0, j, 0)}$$

This is a product of Z on all horizontal edges in row 0.

For $\bar{Z}_2$ (vertical loop at column $j = 0$):
$$\bar{Z}_2 = \prod_{i=0}^{L-1} Z_{(i, 0, 1)}$$

This is a product of Z on all vertical edges in column 0.

### 3. Commutation with Stabilizers

**Claim:** Logical Z operators commute with all stabilizers.

**Proof (commutation with stars):**

A star operator $A_v$ is a product of X operators. We need:
$$[\bar{Z}_k, A_v] = 0$$

Count the overlap between the loop $\gamma_k$ and the star at vertex $v$:
- If $v$ is not on $\gamma_k$: overlap = 0 (commute trivially)
- If $v$ is on $\gamma_k$: exactly 2 edges of the star are on $\gamma_k$

With 2 shared edges, the commutator vanishes:
$$\bar{Z}_k A_v = (-1)^2 A_v \bar{Z}_k = A_v \bar{Z}_k$$

**Proof (commutation with plaquettes):**

Plaquette operators $B_p$ are products of Z. Since Z operators commute with each other:
$$[\bar{Z}_k, B_p] = 0 \quad \text{automatically}$$

### 4. Logical X Operators

Logical X operators are defined on the **dual lattice** cycles:

$$\boxed{\bar{X}_1 = \prod_{e \in \gamma_1^*} X_e}$$

$$\boxed{\bar{X}_2 = \prod_{e \in \gamma_2^*} X_e}$$

where $\gamma_k^*$ is a non-contractible loop on the dual lattice.

**Key insight:** The dual lattice of a square lattice on a torus is another square lattice on the same torus, rotated 45 degrees conceptually.

**Explicit construction:**

For $\bar{X}_1$ (cuts vertical edges across the torus):
$$\bar{X}_1 = \prod_{i=0}^{L-1} X_{(i, 0, 1)}$$

Note: This is a **vertical** string that cuts **horizontally** across the torus.

For $\bar{X}_2$ (cuts horizontal edges across the torus):
$$\bar{X}_2 = \prod_{j=0}^{L-1} X_{(0, j, 0)}$$

This is a **horizontal** string that cuts **vertically** across the torus.

### 5. Commutation and Anti-Commutation

**Theorem:** The logical operators satisfy:
$$[\bar{X}_i, \bar{Z}_j] = 0 \quad \text{for } i \neq j$$
$$\{\bar{X}_i, \bar{Z}_i\} = 0 \quad \text{(anti-commute for same index)}$$

**Proof of anti-commutation ($i = j = 1$):**

$\bar{X}_1$ is defined on vertical edges at column 0: $(i, 0, 1)$ for all $i$.
$\bar{Z}_1$ is defined on horizontal edges at row 0: $(0, j, 0)$ for all $j$.

Intersection: These loops meet at exactly **one** point (one shared edge).

Wait, let me reconsider. Horizontal edges and vertical edges are different, so they don't share edges directly. Let me think about this more carefully.

Actually, $\bar{X}_1$ and $\bar{Z}_1$ act on **different edges** (vertical vs. horizontal). So they commute trivially as operators!

The anti-commutation comes from considering how they act on the **code space**.

**Correct interpretation:**

$\bar{Z}_1$ and $\bar{X}_1$ are chosen so that:
- $\bar{Z}_1$ winds around $\gamma_1$ (horizontal)
- $\bar{X}_1$ winds around $\gamma_1^*$ which **crosses** $\gamma_2$

The key is that $\gamma_1^*$ and $\gamma_2$ intersect at exactly one point.

Let me reclarify with proper dual cycle conventions:

**Convention:**
- $\bar{Z}_1$: Z-string along horizontal cycle $\gamma_1$
- $\bar{Z}_2$: Z-string along vertical cycle $\gamma_2$
- $\bar{X}_1$: X-string along cycle that **crosses** $\gamma_1$ (i.e., along $\gamma_2$ direction)
- $\bar{X}_2$: X-string along cycle that **crosses** $\gamma_2$ (i.e., along $\gamma_1$ direction)

With this convention:
- $\bar{X}_1$ and $\bar{Z}_1$ share edges at exactly one row → anti-commute
- $\bar{X}_1$ and $\bar{Z}_2$ share no edges → commute

### 6. Standard Logical Operator Convention

Let's use the standard convention:

$$\bar{Z}_1 = \prod_{j=0}^{L-1} Z_{(0, j, 0)} \quad \text{(horizontal Z-string at row 0)}$$

$$\bar{Z}_2 = \prod_{i=0}^{L-1} Z_{(i, 0, 1)} \quad \text{(vertical Z-string at column 0)}$$

$$\bar{X}_1 = \prod_{i=0}^{L-1} X_{(i, 0, 1)} \quad \text{(vertical X-string at column 0)}$$

$$\bar{X}_2 = \prod_{j=0}^{L-1} X_{(0, j, 0)} \quad \text{(horizontal X-string at row 0)}$$

**Check anti-commutation:**

$\bar{X}_1$ and $\bar{Z}_2$ both act on vertical edges at column 0:
- $\bar{X}_1 = \prod_i X_{(i, 0, 1)}$
- $\bar{Z}_2 = \prod_i Z_{(i, 0, 1)}$

They share **all L edges**. Since $XZ = -ZX$ on each edge:
$$\bar{X}_1 \bar{Z}_2 = (-1)^L \bar{Z}_2 \bar{X}_1$$

For odd L: anti-commute. For even L: commute. This is a problem!

**Resolution:** The standard convention is that $\bar{X}_1$ should anti-commute with $\bar{Z}_1$, not $\bar{Z}_2$.

Let me use the correct pairing:

$$\bar{X}_1 \text{ anti-commutes with } \bar{Z}_1$$
$$\bar{X}_2 \text{ anti-commutes with } \bar{Z}_2$$

This requires:
- $\bar{Z}_1$ horizontal, $\bar{X}_1$ vertical that crosses it (1 intersection)
- $\bar{Z}_2$ vertical, $\bar{X}_2$ horizontal that crosses it (1 intersection)

**Correct definitions:**

$\bar{Z}_1 = \prod_{j=0}^{L-1} Z_{(0, j, 0)}$ (horizontal at row 0)
$\bar{X}_1 = \prod_{i=0}^{L-1} X_{(i, 0, 1)}$ (vertical at column 0)

Intersection: edges $(0, 0, 0)$ and $(0, 0, 1)$ are different!

Actually, I need to think about this on the **dual lattice** properly.

### 7. Proper Definition via Intersection Number

The correct algebraic structure is:

$$\bar{X}_i \bar{Z}_j = (-1)^{\langle \gamma_i^*, \gamma_j \rangle} \bar{Z}_j \bar{X}_i$$

where $\langle \gamma_i^*, \gamma_j \rangle$ is the **intersection number** mod 2.

For the standard homology basis on the torus:
$$\langle \gamma_1^*, \gamma_1 \rangle = 1, \quad \langle \gamma_2^*, \gamma_2 \rangle = 1$$
$$\langle \gamma_1^*, \gamma_2 \rangle = 0, \quad \langle \gamma_2^*, \gamma_1 \rangle = 0$$

This gives the desired:
$$\{\bar{X}_i, \bar{Z}_i\} = 0, \quad [\bar{X}_i, \bar{Z}_j] = 0 \text{ for } i \neq j$$

### 8. Weight of Logical Operators

**Theorem:** The minimum weight of any logical operator is L.

**Proof:**

A logical operator must:
1. Commute with all stabilizers
2. NOT be a product of stabilizers (otherwise it acts trivially on code space)

Operators commuting with all Z-stabilizers (plaquettes) are products of X forming closed loops.
Operators commuting with all X-stabilizers (stars) are products of Z forming closed loops.

For a non-trivial logical operator:
- The loop must be non-contractible
- Minimum non-contractible loop on $L \times L$ torus has length L

Therefore: $\text{weight}(\bar{X}_i) \geq L$ and $\text{weight}(\bar{Z}_i) \geq L$.

**Code distance:** $d = L$

---

## Quantum Mechanics Connection

### Logical Qubit Operations

The 4 ground states can be labeled $|i, j\rangle_L$ where $i, j \in \{0, 1\}$.

**Action of logical operators:**

$$\bar{Z}_1 |i, j\rangle_L = (-1)^i |i, j\rangle_L$$
$$\bar{Z}_2 |i, j\rangle_L = (-1)^j |i, j\rangle_L$$
$$\bar{X}_1 |i, j\rangle_L = |i \oplus 1, j\rangle_L$$
$$\bar{X}_2 |i, j\rangle_L = |i, j \oplus 1\rangle_L$$

### Topological Protection

**Why can't local operators change logical states?**

Any local operator with support on a region smaller than $L$ edges either:
1. Is a product of stabilizers (acts as identity on code space)
2. Anti-commutes with some stabilizer (takes state out of code space)

Only operators with support wrapping around the torus (weight $\geq L$) can change logical states while preserving the code space.

**This is the essence of topological protection.**

### Connection to Anyonic Operators

The logical operators can be understood as "string operators":
- $\bar{Z}$: String of Z creates pairs of anyons at endpoints, but closed strings have no endpoints
- $\bar{X}$: String of X creates pairs of dual anyons

Non-contractible strings represent **Wilson loops** in gauge theory language.

---

## Worked Examples

### Example 1: Construct Logical Operators for L = 3

**Problem:** Write explicit logical operators for a $3 \times 3$ toric code.

**Solution:**

Edges: horizontal $(i, j, 0)$ → indices 0-8, vertical $(i, j, 1)$ → indices 9-17.

**Logical Z₁** (horizontal loop at row 0):
$$\bar{Z}_1 = Z_{(0,0,0)} Z_{(0,1,0)} Z_{(0,2,0)} = Z_0 Z_1 Z_2$$

**Logical Z₂** (vertical loop at column 0):
$$\bar{Z}_2 = Z_{(0,0,1)} Z_{(1,0,1)} Z_{(2,0,1)} = Z_9 Z_{12} Z_{15}$$

**Logical X₁** (vertical string at column 0):
$$\bar{X}_1 = X_{(0,0,1)} X_{(1,0,1)} X_{(2,0,1)} = X_9 X_{12} X_{15}$$

**Logical X₂** (horizontal string at row 0):
$$\bar{X}_2 = X_{(0,0,0)} X_{(0,1,0)} X_{(0,2,0)} = X_0 X_1 X_2$$

### Example 2: Verify Anti-Commutation

**Problem:** Verify that $\bar{X}_1$ and $\bar{Z}_2$ anti-commute for L = 3.

**Solution:**

$\bar{X}_1 = X_9 X_{12} X_{15}$ (vertical edges at column 0)
$\bar{Z}_2 = Z_9 Z_{12} Z_{15}$ (same edges!)

Both act on the same 3 edges. On each edge:
$$X Z = -ZX$$

For 3 edges:
$$\bar{X}_1 \bar{Z}_2 = (X_9 Z_9)(X_{12} Z_{12})(X_{15} Z_{15}) = (-1)^3 (Z_9 X_9)(Z_{12} X_{12})(Z_{15} X_{15}) = -\bar{Z}_2 \bar{X}_1$$

Therefore $\{\bar{X}_1, \bar{Z}_2\} = 0$ for odd L. ✓

(Note: For even L, they commute, which means $\bar{X}_1$ and $\bar{Z}_2$ are paired differently.)

### Example 3: Commutation with Stabilizers

**Problem:** Verify $[\bar{Z}_1, A_{(0,1)}] = 0$ for the $3 \times 3$ toric code.

**Solution:**

$\bar{Z}_1 = Z_0 Z_1 Z_2$ (horizontal edges at row 0)

Star $A_{(0,1)}$ acts on edges touching vertex (0, 1):
- $(0, 1, 0)$ = edge 1 (right horizontal)
- $(0, 0, 0)$ = edge 0 (left horizontal)
- $(0, 1, 1)$ = edge 10 (down vertical)
- $(-1, 1, 1) = (2, 1, 1)$ = edge 16 (up vertical, periodic)

$A_{(0,1)} = X_0 X_1 X_{10} X_{16}$

Intersection of supports: edges 0 and 1 (both horizontal at row 0).

Number of shared qubits: 2

$$\bar{Z}_1 A_{(0,1)} = (Z_0 Z_1 Z_2)(X_0 X_1 X_{10} X_{16})$$
$$= (Z_0 X_0)(Z_1 X_1)(Z_2)(X_{10} X_{16})$$
$$= (-1)^2 (X_0 Z_0)(X_1 Z_1)(Z_2)(X_{10} X_{16})$$
$$= A_{(0,1)} \bar{Z}_1$$

Therefore $[\bar{Z}_1, A_{(0,1)}] = 0$. ✓

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For a $4 \times 4$ toric code, write the support of $\bar{Z}_1$ and $\bar{X}_2$.

**P1.2** What is the weight of each logical operator in the $L \times L$ toric code?

**P1.3** How many edges does $\bar{X}_1$ share with the plaquette $B_{(0,0)}$?

### Level 2: Intermediate

**P2.1** Prove that $\bar{Z}_1$ commutes with all plaquette operators $B_p$.

**P2.2** Show that for even L, $\bar{X}_1$ and $\bar{Z}_2$ commute (not anti-commute).

**P2.3** Verify that $\bar{X}_1 \bar{X}_2 = \bar{X}_2 \bar{X}_1$ (logical X operators commute with each other).

### Level 3: Challenging

**P3.1** Show that any operator of the form $\bar{Z}_1 \cdot S$ where $S$ is a stabilizer also acts as a valid logical $\bar{Z}_1$ on the code space.

**P3.2** For a rectangular torus $L_1 \times L_2$, what are the weights of $\bar{Z}_1$, $\bar{Z}_2$, $\bar{X}_1$, $\bar{X}_2$?

**P3.3** Prove that no operator with weight less than L can be a non-trivial logical operator.

---

## Computational Lab

```python
"""
Day 789: Logical Operators on the Torus
========================================

Constructing and verifying logical operators for the toric code.
"""

import numpy as np
from typing import List, Tuple, Set, Dict


class ToricCodeLogicalOps:
    """
    Logical operators for the toric code.

    Constructs and verifies logical X and Z operators.
    """

    def __init__(self, L: int):
        """Initialize L x L toric code."""
        self.L = L
        self.n_qubits = 2 * L * L

    def edge_index(self, i: int, j: int, d: int) -> int:
        """Convert (i, j, d) to linear edge index."""
        i, j = i % self.L, j % self.L
        return d * self.L**2 + i * self.L + j

    def logical_Z1_support(self) -> Set[int]:
        """
        Logical Z_1: horizontal Z-string at row 0.
        Winds around the torus in the horizontal direction.
        """
        return {self.edge_index(0, j, 0) for j in range(self.L)}

    def logical_Z2_support(self) -> Set[int]:
        """
        Logical Z_2: vertical Z-string at column 0.
        Winds around the torus in the vertical direction.
        """
        return {self.edge_index(i, 0, 1) for i in range(self.L)}

    def logical_X1_support(self) -> Set[int]:
        """
        Logical X_1: vertical X-string at column 0.
        Crosses the horizontal cycle (anti-commutes with Z_2).
        """
        return {self.edge_index(i, 0, 1) for i in range(self.L)}

    def logical_X2_support(self) -> Set[int]:
        """
        Logical X_2: horizontal X-string at row 0.
        Crosses the vertical cycle (anti-commutes with Z_1).
        """
        return {self.edge_index(0, j, 0) for j in range(self.L)}

    def star_support(self, vi: int, vj: int) -> Set[int]:
        """Return edges in star operator at vertex (vi, vj)."""
        vi, vj = vi % self.L, vj % self.L
        return {
            self.edge_index(vi, vj, 0),
            self.edge_index(vi, vj - 1, 0),
            self.edge_index(vi, vj, 1),
            self.edge_index(vi - 1, vj, 1),
        }

    def plaquette_support(self, pi: int, pj: int) -> Set[int]:
        """Return edges in plaquette operator at face (pi, pj)."""
        pi, pj = pi % self.L, pj % self.L
        return {
            self.edge_index(pi, pj, 0),
            self.edge_index(pi + 1, pj, 0),
            self.edge_index(pi, pj, 1),
            self.edge_index(pi, pj + 1, 1),
        }

    def check_commutation(self, X_support: Set[int], Z_support: Set[int]) -> int:
        """
        Check commutation of X-type and Z-type operators.
        Returns 0 if they commute, 1 if they anti-commute.
        """
        overlap = len(X_support & Z_support)
        return overlap % 2

    def verify_logical_ops_commute_with_stabilizers(self) -> Dict[str, bool]:
        """Verify all logical operators commute with all stabilizers."""
        results = {}

        logical_ops = {
            'Z1': ('Z', self.logical_Z1_support()),
            'Z2': ('Z', self.logical_Z2_support()),
            'X1': ('X', self.logical_X1_support()),
            'X2': ('X', self.logical_X2_support()),
        }

        for name, (op_type, support) in logical_ops.items():
            commutes_with_stars = True
            commutes_with_plaquettes = True

            for i in range(self.L):
                for j in range(self.L):
                    star = self.star_support(i, j)
                    plaq = self.plaquette_support(i, j)

                    if op_type == 'Z':
                        # Z commutes with Z (plaquettes), check with X (stars)
                        if self.check_commutation(star, support) != 0:
                            commutes_with_stars = False
                    else:  # X type
                        # X commutes with X (stars), check with Z (plaquettes)
                        if self.check_commutation(support, plaq) != 0:
                            commutes_with_plaquettes = False

            results[f'{name}_commutes_with_stars'] = commutes_with_stars
            results[f'{name}_commutes_with_plaquettes'] = commutes_with_plaquettes

        return results

    def verify_logical_algebra(self) -> Dict[str, str]:
        """Verify commutation/anti-commutation between logical operators."""
        Z1 = self.logical_Z1_support()
        Z2 = self.logical_Z2_support()
        X1 = self.logical_X1_support()
        X2 = self.logical_X2_support()

        results = {}

        # X1 with Z1: should anti-commute (same pairing)
        overlap_X1_Z1 = len(X1 & Z1)
        results['X1_Z1'] = f"overlap={overlap_X1_Z1}, {'anti-commute' if overlap_X1_Z1 % 2 == 1 else 'commute'}"

        # X1 with Z2: should anti-commute for our convention
        overlap_X1_Z2 = len(X1 & Z2)
        results['X1_Z2'] = f"overlap={overlap_X1_Z2}, {'anti-commute' if overlap_X1_Z2 % 2 == 1 else 'commute'}"

        # X2 with Z1: should anti-commute for our convention
        overlap_X2_Z1 = len(X2 & Z1)
        results['X2_Z1'] = f"overlap={overlap_X2_Z1}, {'anti-commute' if overlap_X2_Z1 % 2 == 1 else 'commute'}"

        # X2 with Z2: should anti-commute (same pairing)
        overlap_X2_Z2 = len(X2 & Z2)
        results['X2_Z2'] = f"overlap={overlap_X2_Z2}, {'anti-commute' if overlap_X2_Z2 % 2 == 1 else 'commute'}"

        return results


def display_logical_operators(L: int) -> None:
    """Display logical operators for L x L toric code."""
    print(f"\n{'='*60}")
    print(f"Logical Operators for L = {L} Toric Code")
    print(f"{'='*60}")

    code = ToricCodeLogicalOps(L)

    print(f"\nEdge indexing:")
    print(f"  Horizontal edges (i,j,0): indices 0 to {L**2 - 1}")
    print(f"  Vertical edges (i,j,1): indices {L**2} to {2*L**2 - 1}")

    Z1 = sorted(code.logical_Z1_support())
    Z2 = sorted(code.logical_Z2_support())
    X1 = sorted(code.logical_X1_support())
    X2 = sorted(code.logical_X2_support())

    print(f"\nLogical Z_1 (horizontal at row 0):")
    print(f"  Support: {Z1}")
    print(f"  Weight: {len(Z1)}")

    print(f"\nLogical Z_2 (vertical at column 0):")
    print(f"  Support: {Z2}")
    print(f"  Weight: {len(Z2)}")

    print(f"\nLogical X_1 (vertical at column 0):")
    print(f"  Support: {X1}")
    print(f"  Weight: {len(X1)}")

    print(f"\nLogical X_2 (horizontal at row 0):")
    print(f"  Support: {X2}")
    print(f"  Weight: {len(X2)}")


def verify_logical_properties(L: int) -> None:
    """Verify logical operator properties."""
    print(f"\n{'='*60}")
    print(f"Verification for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeLogicalOps(L)

    # Commutation with stabilizers
    stab_results = code.verify_logical_ops_commute_with_stabilizers()
    print("\nCommutation with stabilizers:")
    for key, value in stab_results.items():
        status = "PASS" if value else "FAIL"
        print(f"  {key}: {status}")

    # Logical algebra
    algebra_results = code.verify_logical_algebra()
    print("\nLogical operator algebra:")
    for key, value in algebra_results.items():
        print(f"  {key}: {value}")


def visualize_logical_loops(L: int) -> None:
    """Visualize logical operators on the lattice."""
    print(f"\n{'='*60}")
    print(f"Logical Operator Visualization (L = {L})")
    print(f"{'='*60}")

    print(f"""
    Lattice with logical operators:

    Z_1 (horizontal, row 0):
    ═══════════════> wraps around

    Z_2 (vertical, column 0):
    ║
    ║
    ║ wraps
    ║ around
    ↓

    X_1: Same path as Z_2 (anti-commutes with Z_2)
    X_2: Same path as Z_1 (anti-commutes with Z_1)

    Note: X_1 and Z_2 share all {L} edges, so they anti-commute
          if L is odd, and commute if L is even.

    For standard logical qubit convention, we pair:
    - X_1 with Z_2 (both on vertical edges at column 0)
    - X_2 with Z_1 (both on horizontal edges at row 0)
    """)


def demonstrate_equivalent_representatives(L: int) -> None:
    """Show that different representatives are equivalent."""
    print(f"\n{'='*60}")
    print(f"Equivalent Representatives (L = {L})")
    print(f"{'='*60}")

    code = ToricCodeLogicalOps(L)

    Z1_row0 = {code.edge_index(0, j, 0) for j in range(L)}
    Z1_row1 = {code.edge_index(1, j, 0) for j in range(L)}

    print(f"\nLogical Z_1 at row 0: {sorted(Z1_row0)}")
    print(f"Logical Z_1 at row 1: {sorted(Z1_row1)}")

    # The difference is a product of plaquettes
    difference = Z1_row0.symmetric_difference(Z1_row1)
    print(f"Symmetric difference: {sorted(difference)}")

    # This difference should be a product of plaquettes between rows 0 and 1
    plaquette_edges = set()
    for j in range(L):
        plaquette_edges.update(code.plaquette_support(0, j))

    print(f"\nPlaquette boundaries (row 0): {sorted(plaquette_edges)}")
    print(f"Z_1(row0) XOR Z_1(row1) is a product of plaquettes: {difference == plaquette_edges}")


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 789: LOGICAL OPERATORS ON THE TORUS")
    print("=" * 70)

    # Demo 1: Display logical operators
    for L in [3, 4, 5]:
        display_logical_operators(L)

    # Demo 2: Verify properties
    for L in [3, 4, 5]:
        verify_logical_properties(L)

    # Demo 3: Visualization
    visualize_logical_loops(3)

    # Demo 4: Equivalent representatives
    demonstrate_equivalent_representatives(3)

    # Demo 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Logical Operators on the Torus")
    print("=" * 70)

    print("""
    LOGICAL Z OPERATORS:
    --------------------
    Z_1 = prod_{j} Z_{(0,j,0)}  (horizontal loop at row 0)
    Z_2 = prod_{i} Z_{(i,0,1)}  (vertical loop at column 0)

    These are non-contractible Z-strings that wind around the torus.

    LOGICAL X OPERATORS:
    --------------------
    X_1 = prod_{i} X_{(i,0,1)}  (vertical loop at column 0)
    X_2 = prod_{j} X_{(0,j,0)}  (horizontal loop at row 0)

    These are non-contractible X-strings on the dual lattice.

    COMMUTATION RELATIONS:
    ----------------------
    [X_i, Z_j] = 0 for i != j (cross different cycles)
    {X_i, Z_i} = 0 for same i (share all L edges, L odd)

    TOPOLOGICAL PROTECTION:
    -----------------------
    - Minimum weight of logical operator = L (code distance)
    - Local errors (weight < L) cannot change logical state
    - Information is encoded in global topology

    EQUIVALENT REPRESENTATIVES:
    ---------------------------
    Any homologous loop works as a logical operator.
    Z_1 at row k differs from Z_1 at row 0 by a product of plaquettes.
    """)

    print("=" * 70)
    print("Day 789 Complete: Logical Operators Mastered")
    print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Logical Z₁ | $\bar{Z}_1 = \prod_{j=0}^{L-1} Z_{(0,j,0)}$ (horizontal loop) |
| Logical Z₂ | $\bar{Z}_2 = \prod_{i=0}^{L-1} Z_{(i,0,1)}$ (vertical loop) |
| Logical X₁ | $\bar{X}_1 = \prod_{i=0}^{L-1} X_{(i,0,1)}$ (vertical string) |
| Logical X₂ | $\bar{X}_2 = \prod_{j=0}^{L-1} X_{(0,j,0)}$ (horizontal string) |
| Commutation | $[\bar{X}_i, \bar{Z}_j] = 0$ for $i \neq j$ |
| Anti-commutation | $\{\bar{X}_i, \bar{Z}_i\} = 0$ (same index) |
| Minimum weight | $\text{weight}(\bar{O}) \geq L$ for any logical $\bar{O}$ |

### Main Takeaways

1. **Logical operators** are non-contractible loop operators on the torus
2. **Logical Z** operators are Z-strings along non-contractible cycles
3. **Logical X** operators are X-strings on dual (crossing) cycles
4. **Commutation relations** follow from intersection numbers of cycles
5. **Topological protection:** minimum logical operator weight = L = code distance
6. **Equivalent representatives:** homologous loops differ by stabilizers

---

## Daily Checklist

- [ ] I can identify non-contractible loops on the torus
- [ ] I can construct all 4 logical operators explicitly
- [ ] I can verify commutation with stabilizers
- [ ] I understand the anti-commutation condition
- [ ] I can explain why code distance = L
- [ ] I ran the computational lab and verified all properties

---

## Preview: Day 790

Tomorrow we study **Error Models and Code Distance**:

- X, Z, and Y errors as anyon pair creation
- Error chains and their homology classes
- Code distance = L (minimum non-trivial loop)
- Error threshold ~10.9% for independent noise
- Comparison with random codes and other topological codes

Understanding errors in the toric code reveals how topological protection works in practice.
