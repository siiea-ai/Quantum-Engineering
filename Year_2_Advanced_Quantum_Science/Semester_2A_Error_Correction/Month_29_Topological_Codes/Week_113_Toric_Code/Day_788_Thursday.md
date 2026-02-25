# Day 788: Toric Code as CSS Code

## Overview

**Day:** 788 of 1008
**Week:** 113 (Toric Code Fundamentals)
**Month:** 29 (Topological Codes)
**Topic:** CSS Structure of the Toric Code

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | CSS code structure |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Chain complexes and homology |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Identify** the toric code as a CSS code with X and Z stabilizers
2. **Construct** the parity check matrices $H_X$ and $H_Z$
3. **Explain** the chain complex structure underlying the toric code
4. **Define** boundary operators and verify $\partial^2 = 0$
5. **Connect** logical operators to homology classes
6. **Draw** the dual lattice and interpret its role

---

## Core Theory

### 1. CSS Code Review

A **Calderbank-Shor-Steane (CSS) code** has stabilizers of two types:
- **X-type stabilizers:** Products of Pauli-X operators
- **Z-type stabilizers:** Products of Pauli-Z operators

The key property: Every X-stabilizer commutes with every Z-stabilizer.

For classical codes $C_X$ and $C_Z$, the CSS condition is:
$$C_Z^\perp \subseteq C_X$$

### 2. Toric Code as CSS

The toric code is a CSS code where:

**X-stabilizers:** Star operators
$$S_X = \{A_v = \prod_{e \ni v} X_e : v \in \text{vertices}\}$$

**Z-stabilizers:** Plaquette operators
$$S_Z = \{B_p = \prod_{e \in \partial p} Z_e : p \in \text{faces}\}$$

**CSS property verified:** Star operators (X-type) commute with plaquette operators (Z-type) because they share an even number of qubits (0 or 2).

### 3. Parity Check Matrices

We can express the toric code stabilizers as binary matrices:

**X-stabilizer matrix $H_X$:** Rows are vertices, columns are edges
$$[H_X]_{v,e} = \begin{cases} 1 & \text{if edge } e \text{ touches vertex } v \\ 0 & \text{otherwise} \end{cases}$$

This is the **incidence matrix** of the lattice graph.

**Z-stabilizer matrix $H_Z$:** Rows are faces, columns are edges
$$[H_Z]_{p,e} = \begin{cases} 1 & \text{if edge } e \text{ is on boundary of face } p \\ 0 & \text{otherwise} \end{cases}$$

This is the **boundary matrix** of the lattice.

### 4. Chain Complex Structure

The toric code arises from a **chain complex** on the lattice:

$$C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0$$

where:
- $C_0 = \mathbb{Z}_2^V$ (vertices/0-chains)
- $C_1 = \mathbb{Z}_2^E$ (edges/1-chains)
- $C_2 = \mathbb{Z}_2^F$ (faces/2-chains)

**Boundary operators:**
- $\partial_1$: Maps edges to their boundary vertices
- $\partial_2$: Maps faces to their boundary edges

**Key relation:**
$$\boxed{\partial_1 \circ \partial_2 = 0}$$

In matrix form:
$$H_X \cdot H_Z^T = 0 \pmod{2}$$

This is precisely the CSS commutation condition!

### 5. Boundary Operator $\partial_1$

The boundary of an edge is its two endpoints:
$$\partial_1(e) = v_1 + v_2$$

where $v_1, v_2$ are the vertices connected by edge $e$.

**As a matrix:** $\partial_1 = H_X^T$

Example for edge $(i, j, 0)$ (horizontal from $(i,j)$ to $(i, j+1)$):
$$\partial_1((i, j, 0)) = (i, j) + (i, j+1)$$

### 6. Boundary Operator $\partial_2$

The boundary of a face is its four edges:
$$\partial_2(p) = e_1 + e_2 + e_3 + e_4$$

**As a matrix:** $\partial_2 = H_Z^T$

Example for plaquette $(i, j)$:
$$\partial_2((i, j)) = (i, j, 0) + (i+1, j, 0) + (i, j, 1) + (i, j+1, 1)$$

### 7. Verification: $\partial_1 \partial_2 = 0$

**Proof:**

The boundary of a face consists of 4 edges forming a closed loop. Each vertex on this loop is touched by exactly 2 of these edges.

When we compute $\partial_1(\partial_2(p))$:
- Each vertex on the boundary appears twice (once from each adjacent edge)
- In $\mathbb{Z}_2$: $1 + 1 = 0$

Therefore $\partial_1(\partial_2(p)) = 0$ for all faces $p$.

In matrix form: $H_X \cdot H_Z^T = 0 \pmod{2}$

This is exactly the CSS commutation condition: X and Z stabilizers commute.

### 8. Homology and Logical Operators

**Cycles:** $\ker(\partial_1) = $ edges that form closed loops (have no boundary)

**Boundaries:** $\text{im}(\partial_2) = $ edges that bound some collection of faces

**First homology group:**
$$H_1 = \ker(\partial_1) / \text{im}(\partial_2)$$

For the torus: $H_1(\mathbb{T}^2; \mathbb{Z}_2) \cong \mathbb{Z}_2 \times \mathbb{Z}_2$

**Physical interpretation:**
- **Cycles (closed loops):** Z-operators that commute with X-stabilizers
- **Boundaries (contractible loops):** Z-operators that are products of Z-stabilizers
- **Homology classes:** Equivalence classes of non-contractible loops

**Logical Z operators** correspond to non-trivial homology classes!

### 9. The Dual Lattice

The **dual lattice** is obtained by:
- Vertices ↔ Faces
- Edges remain edges (but orthogonal)
- Faces ↔ Vertices

```
Original Lattice:       Dual Lattice:
    ●───●───●             ○   ○   ○
    │   │   │           ───│───│───
    ●───●───●             ○   ○   ○
    │   │   │           ───│───│───
    ●───●───●             ○   ○   ○
```

**Duality exchange:**
- Star operators (vertices) ↔ Plaquette operators (dual faces)
- Plaquette operators (faces) ↔ Star operators (dual vertices)
- X errors ↔ Z errors (under duality)

### 10. CSS Code Parameters

From the chain complex viewpoint:

**Physical qubits:** $n = |E| = 2L^2$ (number of edges)

**X-stabilizer generators:** $|V| - 1 = L^2 - 1$ (vertices minus one constraint)

**Z-stabilizer generators:** $|F| - 1 = L^2 - 1$ (faces minus one constraint)

**Logical qubits:**
$$k = n - (|V| - 1) - (|F| - 1) = 2L^2 - 2L^2 + 2 = 2$$

This equals $\dim(H_1) = 2$, the number of independent non-contractible cycles.

---

## Quantum Mechanics Connection

### CSS Decoding Advantage

Because the toric code is CSS:
1. **X errors** create pairs of Z-stabilizer violations (plaquette defects)
2. **Z errors** create pairs of X-stabilizer violations (star defects)

We can decode X and Z errors **independently** using minimum-weight perfect matching (MWPM) on the dual and primal lattices respectively.

### Connection to Homological Codes

The toric code is the simplest **homological code** or **quantum LDPC code** from algebraic topology:
- Code space = kernel of boundary operators
- Logical operators = non-trivial cycles
- Error correction = finding minimum-weight homology representatives

**Generalization:** Replace the torus with any surface $\Sigma$:
$$k = 2g$$

where $g$ is the genus (number of "holes").

### Topological Field Theory Perspective

The toric code is the lattice realization of $\mathbb{Z}_2$ **Dijkgraaf-Witten theory**:
- Ground states = flat $\mathbb{Z}_2$ connections on the torus
- Excitations = gauge charges and fluxes
- Braiding = Aharonov-Bohm phase

---

## Worked Examples

### Example 1: Construct $H_X$ for L = 2

**Problem:** Write the X-stabilizer matrix $H_X$ for a $2 \times 2$ toric code.

**Solution:**

Vertices: $(0,0), (0,1), (1,0), (1,1)$ → indices 0, 1, 2, 3
Edges: horizontal $(i,j,0)$ → 0-3, vertical $(i,j,1)$ → 4-7

For vertex $(0,0)$, star touches edges:
- $(0,0,0) = 0$, $(0,1,0) = 1$ (periodic), $(0,0,1) = 4$, $(1,0,1) = 6$ (periodic)

Row 0 of $H_X$: $[1, 1, 0, 0, 1, 0, 1, 0]$

Continuing for all vertices:

$$H_X = \begin{pmatrix}
1 & 1 & 0 & 0 & 1 & 0 & 1 & 0 \\
1 & 1 & 0 & 0 & 0 & 1 & 0 & 1 \\
0 & 0 & 1 & 1 & 1 & 0 & 1 & 0 \\
0 & 0 & 1 & 1 & 0 & 1 & 0 & 1
\end{pmatrix}$$

Wait, let me recalculate more carefully with proper periodic boundaries.

For $L = 2$, vertex $(0,0)$ touches:
- Right: $(0,0,0)$ - edge 0
- Left: $(0,-1,0) = (0,1,0)$ - edge 1
- Down: $(0,0,1)$ - edge 4
- Up: $(-1,0,1) = (1,0,1)$ - edge 6

Vertex $(0,1)$ touches:
- Right: $(0,1,0)$ - edge 1
- Left: $(0,0,0)$ - edge 0
- Down: $(0,1,1)$ - edge 5
- Up: $(1,1,1)$ - edge 7

And so on.

### Example 2: Verify CSS Condition

**Problem:** Verify $H_X H_Z^T = 0$ for the $2 \times 2$ toric code.

**Solution:**

Each row of $H_X$ (star) and each row of $H_Z$ (plaquette) share exactly 0 or 2 edges.

Consider star at $(0,0)$ and plaquette at $(0,0)$:
- Star edges: 0, 1, 4, 6
- Plaquette edges: 0, 2, 4, 5

Intersection: $\{0, 4\}$ - size 2

Inner product mod 2: $2 \mod 2 = 0$ ✓

Since all pairs have even intersection, $H_X H_Z^T = 0$.

### Example 3: Identify Homology Classes

**Problem:** For a $3 \times 3$ torus, describe the 4 homology classes of 1-cycles.

**Solution:**

**Class [0,0]:** Trivial cycles (boundaries of collections of faces)
- Example: Single plaquette boundary
- These are exactly the Z-stabilizers

**Class [1,0]:** Horizontal non-contractible loops
- Example: Horizontal line of 3 edges at row $i$
- Wraps around the torus in the x-direction

**Class [0,1]:** Vertical non-contractible loops
- Example: Vertical line of 3 edges at column $j$
- Wraps around the torus in the y-direction

**Class [1,1]:** Diagonal (both directions)
- Example: Sum of horizontal and vertical loops
- Wraps around both directions

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For a $3 \times 3$ toric code, what are the dimensions of $H_X$ and $H_Z$?

**P1.2** Verify that row 0 of $H_X$ has exactly 4 nonzero entries (one star touches 4 edges).

**P1.3** What is the rank of $H_X$ for the toric code? (Hint: one constraint)

### Level 2: Intermediate

**P2.1** Prove that the product of all rows of $H_X$ (over $\mathbb{Z}_2$) equals the zero row.

**P2.2** Show that any edge belongs to exactly 2 stars and exactly 2 plaquettes.

**P2.3** For the dual lattice, describe what the X-stabilizers and Z-stabilizers become.

### Level 3: Challenging

**P3.1** Prove that for the toric code, $\ker(H_X^T)$ corresponds to closed loops, and $\text{im}(H_Z^T)$ corresponds to contractible loops.

**P3.2** Generalize: for a surface of genus $g$, how many logical qubits does the toric code encode?

**P3.3** Construct the chain complex for a triangular lattice on a torus and verify $\partial_1 \partial_2 = 0$.

---

## Computational Lab

```python
"""
Day 788: Toric Code as CSS Code
================================

Analyzing the CSS structure and chain complex of the toric code.
"""

import numpy as np
from typing import Tuple, List, Dict
from scipy.linalg import null_space


class ToricCodeCSS:
    """
    CSS structure analysis of the toric code.

    Constructs parity check matrices and verifies CSS conditions.
    """

    def __init__(self, L: int):
        """Initialize L x L toric code."""
        self.L = L
        self.n_qubits = 2 * L * L  # edges
        self.n_vertices = L * L
        self.n_faces = L * L

    def edge_index(self, i: int, j: int, d: int) -> int:
        """Convert (i, j, d) to linear edge index."""
        i, j = i % self.L, j % self.L
        return d * self.L**2 + i * self.L + j

    def vertex_index(self, i: int, j: int) -> int:
        """Convert (i, j) to linear vertex index."""
        i, j = i % self.L, j % self.L
        return i * self.L + j

    def face_index(self, i: int, j: int) -> int:
        """Convert (i, j) to linear face index."""
        i, j = i % self.L, j % self.L
        return i * self.L + j

    def build_H_X(self) -> np.ndarray:
        """
        Build X-stabilizer parity check matrix.

        H_X[v, e] = 1 if edge e touches vertex v.
        This is the incidence matrix = boundary_1^T.
        """
        H_X = np.zeros((self.n_vertices, self.n_qubits), dtype=int)

        for i in range(self.L):
            for j in range(self.L):
                v_idx = self.vertex_index(i, j)

                # Four edges touching vertex (i, j)
                edges = [
                    self.edge_index(i, j, 0),       # right horizontal
                    self.edge_index(i, j - 1, 0),   # left horizontal
                    self.edge_index(i, j, 1),       # down vertical
                    self.edge_index(i - 1, j, 1),   # up vertical
                ]

                for e_idx in edges:
                    H_X[v_idx, e_idx] = 1

        return H_X

    def build_H_Z(self) -> np.ndarray:
        """
        Build Z-stabilizer parity check matrix.

        H_Z[p, e] = 1 if edge e is on boundary of face p.
        This is boundary_2^T.
        """
        H_Z = np.zeros((self.n_faces, self.n_qubits), dtype=int)

        for i in range(self.L):
            for j in range(self.L):
                p_idx = self.face_index(i, j)

                # Four edges bounding face (i, j)
                edges = [
                    self.edge_index(i, j, 0),       # top horizontal
                    self.edge_index(i + 1, j, 0),   # bottom horizontal
                    self.edge_index(i, j, 1),       # left vertical
                    self.edge_index(i, j + 1, 1),   # right vertical
                ]

                for e_idx in edges:
                    H_Z[p_idx, e_idx] = 1

        return H_Z

    def verify_css_condition(self) -> Tuple[bool, np.ndarray]:
        """
        Verify CSS condition: H_X @ H_Z^T = 0 mod 2.

        Returns (is_valid, product_matrix).
        """
        H_X = self.build_H_X()
        H_Z = self.build_H_Z()

        product = (H_X @ H_Z.T) % 2
        is_valid = np.all(product == 0)

        return is_valid, product

    def compute_ranks(self) -> Dict[str, int]:
        """Compute ranks of parity check matrices."""
        H_X = self.build_H_X()
        H_Z = self.build_H_Z()

        # Compute rank over GF(2) using row reduction
        def gf2_rank(matrix):
            M = matrix.copy()
            rows, cols = M.shape
            rank = 0
            for col in range(cols):
                pivot = None
                for row in range(rank, rows):
                    if M[row, col] == 1:
                        pivot = row
                        break
                if pivot is None:
                    continue
                M[[rank, pivot]] = M[[pivot, rank]]
                for row in range(rows):
                    if row != rank and M[row, col] == 1:
                        M[row] = (M[row] + M[rank]) % 2
                rank += 1
            return rank

        return {
            'rank_H_X': gf2_rank(H_X),
            'rank_H_Z': gf2_rank(H_Z),
            'n_qubits': self.n_qubits,
            'logical_qubits': self.n_qubits - gf2_rank(H_X) - gf2_rank(H_Z)
        }

    def chain_complex_verification(self) -> Dict[str, bool]:
        """
        Verify chain complex property: boundary_1 @ boundary_2 = 0.

        boundary_1 = H_X^T (maps edges to vertices)
        boundary_2 = H_Z^T (maps faces to edges)
        """
        H_X = self.build_H_X()
        H_Z = self.build_H_Z()

        boundary_1 = H_X.T  # n_edges x n_vertices
        boundary_2 = H_Z.T  # n_qubits x n_faces

        # boundary_1 @ boundary_2 should be zero
        composition = (boundary_1.T @ boundary_2) % 2  # n_vertices x n_faces

        return {
            'boundary_squared_zero': np.all(composition == 0),
            'composition_matrix': composition
        }


def analyze_css_structure(L: int) -> None:
    """Analyze CSS structure for L x L toric code."""
    print(f"\n{'='*60}")
    print(f"CSS Structure Analysis for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeCSS(L)

    # Build matrices
    H_X = code.build_H_X()
    H_Z = code.build_H_Z()

    print(f"\nMatrix dimensions:")
    print(f"  H_X: {H_X.shape} (vertices x edges)")
    print(f"  H_Z: {H_Z.shape} (faces x edges)")

    # Verify CSS condition
    is_css, product = code.verify_css_condition()
    print(f"\nCSS condition H_X @ H_Z^T = 0 mod 2: {'VERIFIED' if is_css else 'FAILED'}")

    # Compute ranks
    ranks = code.compute_ranks()
    print(f"\nRank analysis:")
    print(f"  rank(H_X) = {ranks['rank_H_X']} (expected: {L**2 - 1})")
    print(f"  rank(H_Z) = {ranks['rank_H_Z']} (expected: {L**2 - 1})")
    print(f"  n - rank(H_X) - rank(H_Z) = {ranks['logical_qubits']} (expected: 2)")

    # Verify chain complex
    chain_result = code.chain_complex_verification()
    print(f"\nChain complex partial_1 @ partial_2 = 0: {'VERIFIED' if chain_result['boundary_squared_zero'] else 'FAILED'}")


def visualize_parity_matrices(L: int) -> None:
    """Visualize parity check matrices for small L."""
    if L > 3:
        print(f"L = {L} too large for visualization")
        return

    print(f"\n{'='*60}")
    print(f"Parity Check Matrices for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeCSS(L)
    H_X = code.build_H_X()
    H_Z = code.build_H_Z()

    print(f"\nH_X (X-stabilizers / stars):")
    print(H_X)

    print(f"\nH_Z (Z-stabilizers / plaquettes):")
    print(H_Z)

    print(f"\nH_X @ H_Z^T mod 2:")
    print((H_X @ H_Z.T) % 2)


def demonstrate_homology(L: int) -> None:
    """Demonstrate homology calculation."""
    print(f"\n{'='*60}")
    print(f"Homology and Logical Operators for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeCSS(L)
    H_X = code.build_H_X()
    H_Z = code.build_H_Z()

    print(f"\nChain complex: C_2 --d2--> C_1 --d1--> C_0")
    print(f"  C_0 (vertices): dimension {code.n_vertices}")
    print(f"  C_1 (edges):    dimension {code.n_qubits}")
    print(f"  C_2 (faces):    dimension {code.n_faces}")

    # Compute kernel and image dimensions
    def gf2_kernel_dim(matrix):
        """Compute dimension of kernel over GF(2)."""
        rows, cols = matrix.shape
        rank = 0
        M = matrix.copy()
        for col in range(cols):
            pivot = None
            for row in range(rank, rows):
                if M[row, col] == 1:
                    pivot = row
                    break
            if pivot is None:
                continue
            M[[rank, pivot]] = M[[pivot, rank]]
            for row in range(rows):
                if row != rank and M[row, col] == 1:
                    M[row] = (M[row] + M[rank]) % 2
            rank += 1
        return cols - rank

    boundary_1 = H_X.T
    boundary_2 = H_Z.T

    ker_d1 = gf2_kernel_dim(boundary_1.T)  # ker(d1) = {e : d1(e) = 0}
    im_d2 = code.n_qubits - gf2_kernel_dim(boundary_2)  # im(d2) = range of d2

    print(f"\nHomology calculation:")
    print(f"  ker(d1) = closed loops, dim = {ker_d1}")
    print(f"  im(d2) = boundaries, dim = {im_d2}")
    print(f"  H_1 = ker(d1) / im(d2), dim = {ker_d1 - im_d2}")
    print(f"  Expected: dim(H_1) = 2 for torus")


def construct_logical_z_operators(L: int) -> None:
    """Construct explicit logical Z operators."""
    print(f"\n{'='*60}")
    print(f"Logical Z Operators for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeCSS(L)

    # Logical Z_1: horizontal loop at row 0
    Z1_support = [code.edge_index(0, j, 0) for j in range(L)]

    # Logical Z_2: vertical loop at column 0
    Z2_support = [code.edge_index(i, 0, 1) for i in range(L)]

    print(f"\nLogical Z_1 (horizontal non-contractible loop):")
    print(f"  Support: {Z1_support}")
    Z1_vec = np.zeros(code.n_qubits, dtype=int)
    for idx in Z1_support:
        Z1_vec[idx] = 1
    print(f"  Vector: {Z1_vec}")

    print(f"\nLogical Z_2 (vertical non-contractible loop):")
    print(f"  Support: {Z2_support}")
    Z2_vec = np.zeros(code.n_qubits, dtype=int)
    for idx in Z2_support:
        Z2_vec[idx] = 1
    print(f"  Vector: {Z2_vec}")

    # Verify they commute with all X-stabilizers (plaquettes for Z operators? No, stars!)
    H_X = code.build_H_X()

    Z1_syndrome = (H_X @ Z1_vec) % 2
    Z2_syndrome = (H_X @ Z2_vec) % 2

    print(f"\nVerification (should be zero syndrome):")
    print(f"  H_X @ Z1 mod 2: {Z1_syndrome} (sum = {sum(Z1_syndrome)})")
    print(f"  H_X @ Z2 mod 2: {Z2_syndrome} (sum = {sum(Z2_syndrome)})")


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 788: TORIC CODE AS CSS CODE")
    print("=" * 70)

    # Demo 1: CSS structure analysis
    for L in [2, 3, 4, 5]:
        analyze_css_structure(L)

    # Demo 2: Visualize matrices for small L
    visualize_parity_matrices(2)

    # Demo 3: Homology demonstration
    demonstrate_homology(3)

    # Demo 4: Logical operators
    construct_logical_z_operators(3)

    # Demo 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Toric Code as CSS Code")
    print("=" * 70)

    print("""
    CSS STRUCTURE:
    --------------
    X-stabilizers: Star operators A_v (products of X on edges at vertex v)
    Z-stabilizers: Plaquette operators B_p (products of Z on edges around face p)

    PARITY CHECK MATRICES:
    ----------------------
    H_X: Incidence matrix (vertices x edges)
         H_X[v,e] = 1 if edge e touches vertex v

    H_Z: Boundary matrix (faces x edges)
         H_Z[p,e] = 1 if edge e bounds face p

    CSS CONDITION:
    --------------
    H_X @ H_Z^T = 0 mod 2

    This follows from: every face boundary is a closed loop (touches each
    vertex 0 or 2 times).

    CHAIN COMPLEX:
    --------------
    C_2 (faces) --d2--> C_1 (edges) --d1--> C_0 (vertices)

    d1 = H_X^T (boundary of edge = its two endpoints)
    d2 = H_Z^T (boundary of face = its four edges)

    d1 @ d2 = 0 because boundary of a face has no boundary.

    HOMOLOGY:
    ---------
    H_1 = ker(d1) / im(d2) = closed loops / boundaries

    dim(H_1) = 2 for torus (two non-contractible cycles)

    Logical Z operators = non-trivial elements of H_1
    Logical X operators = non-trivial elements of dual H_1
    """)

    print("=" * 70)
    print("Day 788 Complete: CSS Structure of Toric Code Understood")
    print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| X-stabilizer matrix | $[H_X]_{v,e} = 1$ if edge $e$ touches vertex $v$ |
| Z-stabilizer matrix | $[H_Z]_{p,e} = 1$ if edge $e$ bounds face $p$ |
| CSS condition | $H_X H_Z^T = 0 \pmod{2}$ |
| Boundary operators | $\partial_1 = H_X^T$, $\partial_2 = H_Z^T$ |
| Chain complex | $\partial_1 \circ \partial_2 = 0$ |
| First homology | $H_1 = \ker(\partial_1) / \text{im}(\partial_2)$ |
| Logical qubits | $k = \dim(H_1) = 2$ |

### Main Takeaways

1. **The toric code is a CSS code** with X-stabilizers (stars) and Z-stabilizers (plaquettes)
2. **Parity check matrices** $H_X$ and $H_Z$ are incidence and boundary matrices of the lattice
3. **CSS condition** $H_X H_Z^T = 0$ follows from $\partial_1 \partial_2 = 0$ (boundary of boundary is empty)
4. **Logical operators** correspond to non-trivial elements of the first homology group
5. **Dual lattice** exchanges X and Z roles, reflecting self-duality of the toric code
6. **Homological perspective** connects quantum error correction to algebraic topology

---

## Daily Checklist

- [ ] I can identify the toric code as a CSS code
- [ ] I can construct $H_X$ and $H_Z$ matrices
- [ ] I understand the chain complex $C_2 \to C_1 \to C_0$
- [ ] I can verify the CSS condition using boundary operators
- [ ] I understand how homology classes give logical operators
- [ ] I ran the computational lab and verified all CSS properties

---

## Preview: Day 789

Tomorrow we explore **Logical Operators on the Torus**:

- Non-contractible loops and their role as logical operators
- Explicit construction of $\bar{X}_1, \bar{X}_2, \bar{Z}_1, \bar{Z}_2$
- Commutation and anti-commutation relations
- Why logical operators cannot be local
- Topological protection from logical operator weight

The logical operators reveal how the toric code encodes quantum information topologically.
