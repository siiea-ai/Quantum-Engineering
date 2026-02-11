# Day 806: Error Chains and Homology

## Year 2, Semester 2A: Quantum Error Correction
### Month 29: Topological Codes | Week 116: Error Chains & Logical Operations

---

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3 hours | Chain complexes and boundary operators |
| Afternoon | 2.5 hours | Homology classes and logical errors |
| Evening | 1.5 hours | Computational lab: chain visualization |

---

## Learning Objectives

By the end of today, you will be able to:

1. Represent Pauli errors as 1-chains on the surface code lattice
2. Compute syndromes as boundary of error chains
3. Identify homology classes and their correspondence to logical errors
4. Determine when two error chains are equivalent (differ by stabilizers)
5. Apply homological algebra to error correction analysis
6. Visualize error chains and their boundaries computationally

---

## Core Content: Errors as Chains

### The Chain Complex Framework

The surface code lattice naturally forms a **chain complex** - a sequence of vector spaces connected by boundary operators:

$$C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0$$

| Chain Space | Elements | Physical Meaning |
|-------------|----------|------------------|
| $C_2$ | Faces (plaquettes) | Z-stabilizer supports |
| $C_1$ | Edges | Qubit locations |
| $C_0$ | Vertices | X-stabilizer centers |

Over $\mathbb{Z}_2$ (binary), we work mod 2 arithmetic.

### Errors as 1-Chains

A Pauli X error pattern defines a **1-chain**:

$$\boxed{E_X = \sum_{e \in \text{error locations}} e \in C_1(\mathcal{L}; \mathbb{Z}_2)}$$

Similarly for Z errors on the dual lattice:

$$E_Z = \sum_{e^* \in \text{error locations}} e^* \in C_1(\mathcal{L}^*; \mathbb{Z}_2)$$

**Example:** X errors on edges $e_1, e_3, e_7$ form the chain $E_X = e_1 + e_3 + e_7$.

### The Boundary Operator

The boundary operator $\partial_1: C_1 \to C_0$ maps each edge to its endpoints:

$$\partial_1(e) = v_+ + v_-$$

where $v_+, v_-$ are the vertices at the ends of edge $e$.

For an error chain:

$$\boxed{\partial_1 E = \sum_{v} (\text{number of error edges incident to } v \mod 2) \cdot v}$$

This is exactly the **syndrome**: vertices where an odd number of errors meet.

### Syndrome from Boundary

The fundamental relationship:

$$\boxed{\text{Syndrome} = \partial(\text{Error chain})}$$

| If error chain is... | Then syndrome is... |
|---------------------|---------------------|
| Empty | No syndrome (all +1) |
| Single edge | Two endpoints |
| Closed loop | Empty (no syndrome!) |
| Path from boundary to boundary | Empty (boundary absorbs) |

**Critical insight:** Errors that form closed loops or span boundaries are **undetectable** by syndrome measurements.

---

## Homology and Logical Errors

### Homology Groups

The **first homology group** captures "holes" in the syndrome detection:

$$\boxed{H_1(\mathcal{L}; \mathbb{Z}_2) = \ker(\partial_1) / \text{im}(\partial_2)}$$

- **Kernel of $\partial_1$**: Chains with zero boundary (closed loops)
- **Image of $\partial_2$**: Boundaries of 2-chains (contractible loops = stabilizers)
- **Homology**: Non-contractible loops (logical operators)

### Four Types of Error Chains

| Type | Boundary | Homology Class | Effect |
|------|----------|----------------|--------|
| Correctable | Non-empty | - | Detected, corrected |
| Stabilizer | Empty | Trivial [0] | No effect |
| Logical X | Empty | Non-trivial [γ₁] | Flips logical qubit |
| Logical Z | Empty | Non-trivial [γ₂] | Flips logical qubit |

### Equivalence Classes of Errors

Two error chains $E$ and $E'$ are **equivalent** if:

$$E - E' \in \text{im}(\partial_2) \quad \Leftrightarrow \quad [E] = [E'] \in H_1$$

**Physical meaning:** $E$ and $E'$ differ by a product of stabilizers.

$$\boxed{E \sim E' \quad \Leftrightarrow \quad E = E' \cdot S \text{ for some stabilizer } S}$$

### The Homology Determines Logical Effect

**Theorem:** The logical effect of an error depends only on its homology class.

For a planar surface code (with boundaries):
- $H_1 \cong \mathbb{Z}_2$ (one generator)
- Non-trivial class = logical error
- Trivial class = correctable or no-op

For toric code:
- $H_1 \cong \mathbb{Z}_2 \times \mathbb{Z}_2$ (two generators)
- Four classes: identity, $\bar{X}_1$, $\bar{X}_2$, $\bar{X}_1\bar{X}_2$

---

## Visual Understanding

### Error Chain Diagrams

```
Surface code with X-error chain:

    ●─────●─────●─────●─────●
    │     │     │     │     │
    ●─────●─────●═════●═════●  ← Syndrome at these vertices
    │     │     ║  X  ║  X  │     (odd # of error edges)
    ●─────●─────●═════●─────●
    │     │     ║  X  │     │
    ●─────●─────●═════●─────●  ← Syndrome here
    │     │     │     │     │
    ●─────●─────●─────●─────●

    ═══ = X error on edge
    Syndrome appears at chain endpoints
```

### Closed Loop (Undetectable)

```
    ●─────●─────●─────●─────●
    │     │     │     │     │
    ●─────●═════●═════●─────●
    │     ║     │     ║     │   No syndrome!
    ●─────●═════●═════●─────●   Loop is contractible
    │     │     │     │     │   = product of plaquettes
    ●─────●─────●─────●─────●   = stabilizer (no error)
```

### Logical Error (Non-contractible)

```
    Smooth ════════════════════ Smooth
           ●─────●─────●─────●
           │     │     │     │
    Rough  ●═════●═════●═════●  Rough    ← Spans rough boundaries
           │     │     │     │            = logical X error
           ●─────●─────●─────●            No syndrome!
    Smooth ════════════════════ Smooth
```

---

## Mathematical Formalism

### Chain Complex Notation

$$C_\bullet: \quad \mathbb{Z}_2^{|F|} \xrightarrow{\partial_2} \mathbb{Z}_2^{|E|} \xrightarrow{\partial_1} \mathbb{Z}_2^{|V|}$$

**Boundary matrices:**

$$(\partial_2)_{e,f} = \begin{cases} 1 & \text{if edge } e \in \partial f \\ 0 & \text{otherwise} \end{cases}$$

$$(\partial_1)_{v,e} = \begin{cases} 1 & \text{if vertex } v \in \partial e \\ 0 & \text{otherwise} \end{cases}$$

**Fundamental property:**

$$\boxed{\partial_1 \circ \partial_2 = 0}$$

This ensures that boundaries have no boundary (stabilizers have trivial syndrome).

### Computing Homology

For a $d \times d$ planar surface code:

$$\dim H_1 = \dim \ker(\partial_1) - \dim \text{im}(\partial_2)$$

Using rank-nullity:
- $\dim \ker(\partial_1) = |E| - \text{rank}(\partial_1)$
- $\dim \text{im}(\partial_2) = \text{rank}(\partial_2)$

**Result:** $\dim H_1 = 1$ for planar surface code (one logical qubit).

### Relative Homology for Boundaries

With boundaries, we use **relative homology**:

$$H_1(\mathcal{L}, \partial\mathcal{L}; \mathbb{Z}_2)$$

This counts chains that end on the boundary as having zero boundary (boundary absorbs endpoints).

---

## Worked Examples

### Example 1: Single X Error

**Setup:** X error on one edge $e_{23}$ connecting vertices $v_2$ and $v_3$.

**Error chain:** $E = e_{23}$

**Syndrome calculation:**
$$\partial_1 E = \partial_1(e_{23}) = v_2 + v_3$$

**Result:** Syndrome at vertices $v_2$ and $v_3$ (two stars violated).

**Homology class:** Not applicable (boundary is non-zero).

**Correction:** Any path connecting $v_2$ to $v_3$.

---

### Example 2: Two Errors Forming a Loop

**Setup:** X errors on edges forming a closed square: $e_{12}, e_{23}, e_{34}, e_{41}$.

**Error chain:** $E = e_{12} + e_{23} + e_{34} + e_{41}$

**Syndrome calculation:**
$$\partial_1 E = (v_1 + v_2) + (v_2 + v_3) + (v_3 + v_4) + (v_4 + v_1) = 0$$

**Result:** No syndrome (all vertices appear twice, canceling mod 2).

**Homology class:** $[E] = 0$ (contractible loop = stabilizer).

**Effect:** No logical error - this is a plaquette stabilizer!

---

### Example 3: Error Chain Spanning Code

**Setup:** d=5 surface code. X errors on horizontal path from left rough boundary to right rough boundary.

**Error chain:** $E = e_1 + e_2 + e_3 + e_4 + e_5$ (5 edges crossing).

**Syndrome:** At rough boundaries, no syndromes appear (boundary absorbs).

**Homology class:** $[E] \neq 0$ (non-trivial in relative homology).

**Effect:** This is $\bar{X}$! Logical X error occurred.

---

## Practice Problems

### Problem Set A: Fundamental Concepts

**A1.** For a 3×3 surface code:
(a) How many physical qubits (edges)?
(b) How many X-stabilizers (vertices, excluding boundary)?
(c) How many Z-stabilizers (faces)?
(d) Verify: n - (# X-stab) - (# Z-stab) = k = 1.

**A2.** Write down the boundary matrix $\partial_1$ for a 2×2 square lattice with 4 vertices and 4 edges (square boundary). Verify $\partial_1$ has rank 3.

**A3.** An error chain has syndrome at vertices $v_1, v_5, v_7, v_{12}$. What is the minimum weight correction that pairs $(v_1, v_5)$ and $(v_7, v_{12})$ vs $(v_1, v_7)$ and $(v_5, v_{12})$?

### Problem Set B: Homology Calculations

**B1.** For a torus with $L \times L$ lattice:
(a) Compute $\dim C_0$, $\dim C_1$, $\dim C_2$.
(b) Use Euler characteristic to find $\dim H_1$.
(c) Verify this matches the number of logical qubits.

**B2.** Two error chains $E_1$ and $E_2$ both have syndrome at vertices $v_3$ and $v_8$. When is $E_1 \sim E_2$ (equivalent)? Give a condition in terms of $E_1 - E_2$.

**B3.** Prove: If $E$ is a product of stabilizers (i.e., $E \in \text{im}(\partial_2)$), then $\partial_1 E = 0$.

### Problem Set C: Advanced Analysis

**C1.** A surface code has distance $d$. Prove that a logical error requires at least $d$ physical errors by showing any non-trivial homology representative has weight at least $d$.

**C2.** Consider depolarizing noise with probability $p$. If the decoder finds a minimum weight correction, show that logical error probability scales as $\sim p^{d/2}$ for large $d$.

**C3.** For a surface code with defects (holes), how does the homology group change? What is $H_1$ for a code with one interior hole?

---

## Computational Lab: Chain Visualization

```python
"""
Day 806 Computational Lab: Error Chains and Homology
Visualizing errors as chains and computing syndromes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

class ChainComplex:
    """
    Chain complex for surface code lattice.

    Implements boundary operators and homology calculations.
    """

    def __init__(self, d: int, boundary: str = 'planar'):
        """
        Initialize chain complex for d×d surface code.

        Args:
            d: Code distance (lattice size)
            boundary: 'planar' or 'toric'
        """
        self.d = d
        self.boundary = boundary

        # Count elements
        if boundary == 'planar':
            self.n_vertices = d * d
            self.n_edges_h = d * (d - 1)  # Horizontal edges
            self.n_edges_v = (d - 1) * d  # Vertical edges
            self.n_edges = self.n_edges_h + self.n_edges_v
            self.n_faces = (d - 1) * (d - 1)
        else:  # Toric
            self.n_vertices = d * d
            self.n_edges = 2 * d * d
            self.n_faces = d * d

        # Build boundary matrices
        self._build_boundary_matrices()

    def _build_boundary_matrices(self):
        """Construct boundary operators ∂₁ and ∂₂."""
        d = self.d

        if self.boundary == 'planar':
            # ∂₁: edges → vertices
            self.d1 = lil_matrix((self.n_vertices, self.n_edges), dtype=np.int8)

            # Horizontal edges
            for i in range(d):
                for j in range(d - 1):
                    edge_idx = i * (d - 1) + j
                    v_left = i * d + j
                    v_right = i * d + j + 1
                    self.d1[v_left, edge_idx] = 1
                    self.d1[v_right, edge_idx] = 1

            # Vertical edges
            offset = self.n_edges_h
            for i in range(d - 1):
                for j in range(d):
                    edge_idx = offset + i * d + j
                    v_top = i * d + j
                    v_bottom = (i + 1) * d + j
                    self.d1[v_top, edge_idx] = 1
                    self.d1[v_bottom, edge_idx] = 1

            # ∂₂: faces → edges
            self.d2 = lil_matrix((self.n_edges, self.n_faces), dtype=np.int8)

            for i in range(d - 1):
                for j in range(d - 1):
                    face_idx = i * (d - 1) + j
                    # Top edge (horizontal)
                    self.d2[i * (d - 1) + j, face_idx] = 1
                    # Bottom edge (horizontal)
                    self.d2[(i + 1) * (d - 1) + j, face_idx] = 1
                    # Left edge (vertical)
                    self.d2[offset + i * d + j, face_idx] = 1
                    # Right edge (vertical)
                    self.d2[offset + i * d + j + 1, face_idx] = 1

            self.d1 = csr_matrix(self.d1)
            self.d2 = csr_matrix(self.d2)

        else:  # Toric
            self.d1 = lil_matrix((self.n_vertices, self.n_edges), dtype=np.int8)
            self.d2 = lil_matrix((self.n_edges, self.n_faces), dtype=np.int8)

            for i in range(d):
                for j in range(d):
                    v = i * d + j
                    # Horizontal edge to the right
                    e_h = i * d + j
                    v_right = i * d + ((j + 1) % d)
                    self.d1[v, e_h] = 1
                    self.d1[v_right, e_h] = 1

                    # Vertical edge downward
                    e_v = d * d + i * d + j
                    v_down = ((i + 1) % d) * d + j
                    self.d1[v, e_v] = 1
                    self.d1[v_down, e_v] = 1

                    # Face boundary
                    f = i * d + j
                    # Top (horizontal)
                    self.d2[i * d + j, f] = 1
                    # Bottom (horizontal)
                    self.d2[((i + 1) % d) * d + j, f] = 1
                    # Left (vertical)
                    self.d2[d * d + i * d + j, f] = 1
                    # Right (vertical)
                    self.d2[d * d + i * d + ((j + 1) % d), f] = 1

            self.d1 = csr_matrix(self.d1)
            self.d2 = csr_matrix(self.d2)

    def boundary(self, chain: np.ndarray) -> np.ndarray:
        """Compute boundary ∂₁ of a 1-chain (mod 2)."""
        return (self.d1 @ chain) % 2

    def is_cycle(self, chain: np.ndarray) -> bool:
        """Check if chain is a cycle (∂chain = 0)."""
        return np.all(self.boundary(chain) == 0)

    def is_boundary(self, chain: np.ndarray) -> bool:
        """Check if chain is a boundary (chain = ∂₂ something)."""
        # Try to solve d2 @ x = chain over Z₂
        # This is a simplified check
        for x in range(2**self.n_faces):
            face_chain = np.array([(x >> i) & 1 for i in range(self.n_faces)])
            if np.array_equal((self.d2 @ face_chain) % 2, chain):
                return True
        return False

    def homology_class(self, chain: np.ndarray) -> str:
        """Determine homology class of a cycle."""
        if not self.is_cycle(chain):
            return "Not a cycle (has boundary)"

        if self.is_boundary(chain):
            return "Trivial (stabilizer)"
        else:
            return "Non-trivial (logical operator)"


class ErrorChainVisualizer:
    """Visualize error chains on surface code lattice."""

    def __init__(self, d: int):
        """Initialize visualizer for d×d code."""
        self.d = d
        self.chain_complex = ChainComplex(d, 'planar')

    def random_error_chain(self, p: float) -> np.ndarray:
        """Generate random error chain with probability p per edge."""
        return (np.random.random(self.chain_complex.n_edges) < p).astype(int)

    def chain_from_path(self, vertices: list) -> np.ndarray:
        """Create chain from a path of vertex indices."""
        d = self.d
        chain = np.zeros(self.chain_complex.n_edges, dtype=int)

        for i in range(len(vertices) - 1):
            v1, v2 = vertices[i], vertices[i + 1]
            r1, c1 = v1 // d, v1 % d
            r2, c2 = v2 // d, v2 % d

            if r1 == r2:  # Horizontal edge
                j = min(c1, c2)
                edge_idx = r1 * (d - 1) + j
                chain[edge_idx] = 1
            else:  # Vertical edge
                i_row = min(r1, r2)
                edge_idx = self.chain_complex.n_edges_h + i_row * d + c1
                chain[edge_idx] = 1

        return chain

    def visualize(self, chain: np.ndarray, title: str = "Error Chain"):
        """
        Visualize error chain and syndrome.

        Shows:
        - Lattice edges (gray)
        - Error chain (red, thick)
        - Syndrome vertices (blue circles)
        """
        d = self.d
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Draw all edges (light gray)
        edge_coords = []

        # Horizontal edges
        for i in range(d):
            for j in range(d - 1):
                edge_coords.append([(j, i), (j + 1, i)])

        # Vertical edges
        for i in range(d - 1):
            for j in range(d):
                edge_coords.append([(j, i), (j, i + 1)])

        # Draw background edges
        lc_bg = LineCollection(edge_coords, colors='lightgray', linewidths=1)
        ax.add_collection(lc_bg)

        # Draw error chain (red, thick)
        error_edges = []
        for idx in range(len(chain)):
            if chain[idx]:
                error_edges.append(edge_coords[idx])

        if error_edges:
            lc_error = LineCollection(error_edges, colors='red', linewidths=4, alpha=0.8)
            ax.add_collection(lc_error)

        # Draw vertices
        for i in range(d):
            for j in range(d):
                ax.plot(j, i, 'ko', markersize=6)

        # Compute and draw syndrome
        syndrome = self.chain_complex.boundary(chain)
        for v_idx in range(len(syndrome)):
            if syndrome[v_idx]:
                i, j = v_idx // d, v_idx % d
                circle = Circle((j, i), 0.2, color='blue', alpha=0.7)
                ax.add_patch(circle)

        # Determine homology class
        homology = self.chain_complex.homology_class(chain)

        # Annotations
        ax.set_xlim(-0.5, d - 0.5)
        ax.set_ylim(-0.5, d - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\nHomology: {homology}", fontsize=14)
        ax.axis('off')

        # Legend
        ax.plot([], [], 'r-', linewidth=4, label='Error chain')
        ax.plot([], [], 'bo', markersize=10, label='Syndrome')
        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig, ax


def demonstrate_chain_types():
    """Demonstrate different types of error chains."""
    d = 5
    viz = ErrorChainVisualizer(d)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 1. Single error (detectable)
    chain1 = np.zeros(viz.chain_complex.n_edges, dtype=int)
    chain1[6] = 1  # One horizontal edge

    ax = axes[0, 0]
    syndrome1 = viz.chain_complex.boundary(chain1)

    # Draw lattice
    for i in range(d):
        for j in range(d - 1):
            ax.plot([j, j+1], [i, i], 'lightgray', linewidth=1)
    for i in range(d - 1):
        for j in range(d):
            ax.plot([j, j], [i, i+1], 'lightgray', linewidth=1)

    # Draw error
    ax.plot([2, 3], [1, 1], 'r-', linewidth=4)

    # Draw syndrome
    for v in range(d*d):
        i, j = v // d, v % d
        if syndrome1[v]:
            ax.add_patch(Circle((j, i), 0.15, color='blue'))
        else:
            ax.plot(j, i, 'ko', markersize=4)

    ax.set_aspect('equal')
    ax.set_title('Single Error (Detectable)\nSyndrome at endpoints', fontsize=12)
    ax.axis('off')

    # 2. Error path
    chain2 = viz.chain_from_path([2, 7, 12, 17])

    ax = axes[0, 1]
    syndrome2 = viz.chain_complex.boundary(chain2)

    for i in range(d):
        for j in range(d - 1):
            ax.plot([j, j+1], [i, i], 'lightgray', linewidth=1)
    for i in range(d - 1):
        for j in range(d):
            ax.plot([j, j], [i, i+1], 'lightgray', linewidth=1)

    # Draw error path
    ax.plot([2, 2, 2, 2], [0, 1, 2, 3], 'r-', linewidth=4)

    for v in range(d*d):
        i, j = v // d, v % d
        if syndrome2[v]:
            ax.add_patch(Circle((j, i), 0.15, color='blue'))
        else:
            ax.plot(j, i, 'ko', markersize=4)

    ax.set_aspect('equal')
    ax.set_title('Error Path\nSyndrome at path endpoints', fontsize=12)
    ax.axis('off')

    # 3. Closed loop (stabilizer - no syndrome)
    ax = axes[1, 0]

    for i in range(d):
        for j in range(d - 1):
            ax.plot([j, j+1], [i, i], 'lightgray', linewidth=1)
    for i in range(d - 1):
        for j in range(d):
            ax.plot([j, j], [i, i+1], 'lightgray', linewidth=1)

    # Draw closed loop
    ax.plot([1, 2], [1, 1], 'r-', linewidth=4)
    ax.plot([2, 2], [1, 2], 'r-', linewidth=4)
    ax.plot([2, 1], [2, 2], 'r-', linewidth=4)
    ax.plot([1, 1], [2, 1], 'r-', linewidth=4)

    for i in range(d):
        for j in range(d):
            ax.plot(j, i, 'ko', markersize=4)

    ax.set_aspect('equal')
    ax.set_title('Closed Loop (Stabilizer)\nNo syndrome - trivial homology', fontsize=12)
    ax.axis('off')

    # 4. Spanning chain (logical error)
    ax = axes[1, 1]

    for i in range(d):
        for j in range(d - 1):
            ax.plot([j, j+1], [i, i], 'lightgray', linewidth=1)
    for i in range(d - 1):
        for j in range(d):
            ax.plot([j, j], [i, i+1], 'lightgray', linewidth=1)

    # Draw spanning chain
    ax.plot([0, 1, 2, 3, 4], [2, 2, 2, 2, 2], 'r-', linewidth=4)

    for i in range(d):
        for j in range(d):
            ax.plot(j, i, 'ko', markersize=4)

    # Mark boundaries
    ax.axvline(x=-0.3, color='green', linewidth=3, label='Rough boundary')
    ax.axvline(x=4.3, color='green', linewidth=3)

    ax.set_aspect('equal')
    ax.set_title('Boundary-Spanning Chain\nNo syndrome - NON-trivial homology\n(Logical X error!)', fontsize=12)
    ax.axis('off')

    plt.suptitle('Four Types of Error Chains', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('error_chain_types.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run error chain demonstrations."""
    print("=" * 70)
    print("DAY 806: ERROR CHAINS AND HOMOLOGY")
    print("=" * 70)

    # Basic chain complex info
    print("\n1. Chain Complex Structure (d=5 planar surface code):")
    cc = ChainComplex(5, 'planar')
    print(f"   Vertices (C₀): {cc.n_vertices}")
    print(f"   Edges (C₁):    {cc.n_edges}")
    print(f"   Faces (C₂):    {cc.n_faces}")
    print(f"   ∂₁ shape: {cc.d1.shape}")
    print(f"   ∂₂ shape: {cc.d2.shape}")

    # Verify ∂₁ ∘ ∂₂ = 0
    print("\n2. Verifying ∂₁ ∘ ∂₂ = 0:")
    d1d2 = (cc.d1 @ cc.d2) % 2
    print(f"   ||∂₁∂₂|| = {np.sum(np.abs(d1d2.toarray()))}")
    print("   ✓ Boundaries have no boundary!" if np.sum(np.abs(d1d2.toarray())) == 0 else "   ✗ Error!")

    # Demonstrate visualizations
    print("\n3. Creating error chain visualizations...")
    demonstrate_chain_types()

    # Homology summary
    print("\n4. Homology Classification:")
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    ERROR CHAIN CLASSIFICATION                │
    ├─────────────────┬──────────────┬──────────────┬─────────────┤
    │ Chain Type      │ Boundary     │ Homology     │ Effect      │
    ├─────────────────┼──────────────┼──────────────┼─────────────┤
    │ Single edge     │ Two vertices │ N/A          │ Detected    │
    │ Open path       │ Endpoints    │ N/A          │ Detected    │
    │ Closed loop     │ Empty        │ Trivial [0]  │ No effect   │
    │ Spans code      │ Empty*       │ Non-trivial  │ LOGICAL ERR │
    └─────────────────┴──────────────┴──────────────┴─────────────┘

    * Boundaries absorb chain endpoints (relative homology)
    """)

    print("\n" + "=" * 70)
    print("Key Insight: The DECODER's job is to find a chain with the")
    print("correct boundary (syndrome) that is in the TRIVIAL homology class.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Error chain | $E = \sum_e c_e \cdot e \in C_1$ |
| Syndrome | $\sigma = \partial_1 E$ |
| Cycle condition | $\partial_1 E = 0$ |
| Homology group | $H_1 = \ker(\partial_1)/\text{im}(\partial_2)$ |
| Equivalent errors | $E \sim E' \Leftrightarrow E - E' \in \text{im}(\partial_2)$ |

### Main Takeaways

1. **Errors are chains**: Pauli errors naturally form 1-chains on the lattice

2. **Syndromes are boundaries**: The syndrome is exactly $\partial E$, the boundary of the error chain

3. **Homology determines fate**: Whether an error causes a logical error depends only on its homology class

4. **Stabilizers are boundaries**: Products of stabilizers are exactly the boundary of face chains (trivial homology)

5. **Decoder goal**: Find any chain with correct boundary in trivial homology class

---

## Daily Checklist

### Morning Session (3 hours)
- [ ] Understand chain complexes and boundary operators
- [ ] Work through syndrome calculation examples
- [ ] Derive $\partial_1 \circ \partial_2 = 0$ property

### Afternoon Session (2.5 hours)
- [ ] Study homology classification of error chains
- [ ] Complete Problem Sets A and B
- [ ] Understand equivalence classes of errors

### Evening Session (1.5 hours)
- [ ] Run computational lab
- [ ] Visualize all four chain types
- [ ] Complete Problem Set C

### Self-Assessment
1. Can you compute syndrome from error chain?
2. Can you determine if a chain is a cycle?
3. Do you understand why closed loops may or may not be logical errors?
4. Can you explain the decoder's objective in homological terms?

---

## Preview: Day 807

Tomorrow we study **Minimum Weight Perfect Matching (MWPM)** for decoding:
- Syndrome graph construction
- Virtual boundary vertices
- Blossom algorithm
- Handling correlated X-Z errors

The decoder's goal: among all chains with correct boundary, find one in the trivial homology class with high probability.

---

*Day 806 of 2184 | Year 2, Month 29, Week 116 | Quantum Engineering PhD Curriculum*
