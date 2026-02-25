# Day 738: Local Complementation

## Overview

**Day:** 738 of 1008
**Week:** 106 (Graph States & MBQC)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Local Complementation and Graph Equivalence

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Local complementation theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Examples and applications |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define** local complementation on graphs
2. **Perform** local complementation operations
3. **Understand** the effect on graph states
4. **Identify** the corresponding unitary operation
5. **Generate** LC-equivalent graphs
6. **Analyze** LC orbits of graph states

---

## Core Content

### Definition of Local Complementation

**Definition (Local Complementation):**
Given graph G = (V, E) and vertex a ∈ V, the **local complement** of G at a, denoted G*a, is obtained by:

$$\boxed{G*a: \text{Complement edges within } N(a)}$$

Explicitly:
- For all pairs {b, c} where b, c ∈ N(a):
  - If {b, c} ∈ E, remove it
  - If {b, c} ∉ E, add it
- All other edges unchanged

**Notation:** G*a denotes local complementation at vertex a.

### Visualizing Local Complementation

**Before LC at vertex a:**
```
     b——c
      \ /
       a
      / \
     d   e
```

**After LC at a (complement within N(a) = {b, c, d, e}):**
- Toggle edges: b-c, b-d, b-e, c-d, c-e, d-e
- Edges to a unchanged

### Effect on Adjacency Matrix

If Γ is the adjacency matrix, then after LC at vertex a:

$$(\Gamma*a)_{bc} = \begin{cases}
\Gamma_{bc} \oplus 1 & \text{if } b, c \in N(a), b \neq c \\
\Gamma_{bc} & \text{otherwise}
\end{cases}$$

**Matrix form:**
Let $n_a = \Gamma_a$ be the characteristic vector of N(a).

$$\Gamma*a = \Gamma \oplus n_a n_a^T \oplus D_a$$

where $D_a$ is the diagonal correction (to ensure zero diagonal).

### The Unitary Correspondence

**Theorem:**
Local complementation on G corresponds to a local Clifford unitary on |G⟩:

$$\boxed{|G*a\rangle = U_a^\dagger |G\rangle}$$

where:
$$U_a = \exp\left(-i\frac{\pi}{4}X_a\right) \prod_{b \in N(a)} \sqrt{Z_b}$$

**Alternative form:**
$$U_a = \frac{1}{\sqrt{2}}(I - iX_a) \prod_{b \in N(a)} \frac{1}{\sqrt{2}}(I + iZ_b)$$

### Properties of Local Complementation

**1. Involution:**
$$(G*a)*a = G$$

Local complementation is its own inverse.

**2. Composition:**
Multiple local complementations compose, but generally don't commute:
$$G*a*b \neq G*b*a \text{ (in general)}$$

**3. Vertex Deletion:**
LC relates to vertex deletion by:
$$G*a - a = G - a \text{ (delete vertex } a \text{)}$$
after certain LC operations.

### LC-Equivalence

**Definition:**
Two graphs G and H are **LC-equivalent** (G ~LC H) if H can be obtained from G by a sequence of local complementations.

$$G \sim_{LC} H \iff H = G*a_1*a_2*\cdots*a_k$$

for some vertices $a_1, \ldots, a_k$.

**Physical meaning:** |G⟩ and |H⟩ are related by local Clifford unitaries.

### LC Orbits

**Definition:**
The **LC orbit** of G is the set of all graphs LC-equivalent to G:
$$[G]_{LC} = \{H : G \sim_{LC} H\}$$

**Size varies:** Some graphs have small LC orbits (e.g., complete graphs), others large.

### Key Theorem: LC Characterization

**Theorem (Van den Nest et al.):**
Two graph states |G⟩ and |H⟩ are related by local Clifford operations iff G and H are LC-equivalent.

This gives a complete combinatorial characterization of local Clifford equivalence for graph states!

### Edge-Local Complementation

**Definition:**
For edge {a, b} ∈ E, **edge-local complementation** is:
$$G * (a, b) = G * a * b * a$$

**Property:** Preserves the edge {a, b}.

**Pivot operation:**
Edge-LC is related to the "pivot" operation in graph theory.

### Computing LC Orbits

**Algorithm (Naive):**
1. Start with graph G
2. Apply LC at each vertex, record new graphs
3. Repeat until no new graphs found

**Complexity:** Can be exponential in |V|.

**Efficient representations:** Use canonical forms or invariants.

---

## Worked Examples

### Example 1: LC on a Triangle

**Graph:** Triangle K₃ with vertices 0, 1, 2

**Adjacency matrix:**
$$\Gamma = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}$$

**LC at vertex 0:**
N(0) = {1, 2}

Complement edges within {1, 2}:
- Edge {1, 2} exists → remove it

**Result G*0:**
$$\Gamma*0 = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}$$

This is a star graph! The triangle is LC-equivalent to a star.

### Example 2: LC on a Path

**Graph:** Path 0—1—2—3

**LC at vertex 1:**
N(1) = {0, 2}

Complement edges within {0, 2}:
- Edge {0, 2} doesn't exist → add it

**Result:**
```
Before: 0—1—2—3
After:  0—1—2—3
         \_/
```
A "kite" shape.

### Example 3: LC Involution

Verify (K₃*0)*0 = K₃.

From Example 1: K₃*0 is the star graph.

**LC on star at vertex 0:**
N(0) = {1, 2}

Complement within {1, 2}:
- Edge {1, 2} doesn't exist → add it

**Result:** Triangle K₃ again! ✓

### Example 4: Complete LC Orbit of K₃

Start with K₃:
- K₃*0 = Star (center 0)
- K₃*1 = Star (center 1)
- K₃*2 = Star (center 2)

LC on star at leaf:
- Star*1 = K₃ (back to triangle)

**LC orbit of K₃:**
{K₃, Star₀, Star₁, Star₂}

All triangles and 3-vertex stars are LC-equivalent!

### Example 5: The Unitary

For the triangle, compute $U_0$:
$$U_0 = e^{-i\frac{\pi}{4}X_0} \cdot \sqrt{Z_1} \cdot \sqrt{Z_2}$$

$$\sqrt{Z} = \frac{1}{\sqrt{2}}(I + iZ) = e^{i\frac{\pi}{4}Z}$$

$$e^{-i\frac{\pi}{4}X} = \frac{1}{\sqrt{2}}(I - iX)$$

This unitary transforms |K₃⟩ to |Star⟩ (up to global phase).

---

## Practice Problems

### Level 1: Direct Application

1. **Basic LC:** For the 4-vertex path 0—1—2—3:
   a) Compute the adjacency matrix after LC at vertex 1
   b) Compute after LC at vertex 2

2. **LC Involution:** Verify (G*a)*a = G for the square graph at vertex 0.

3. **Neighborhood:** For the complete bipartite graph K₂,₃:
   a) List N(a) for each vertex
   b) What happens under LC at a vertex from the size-2 part?

### Level 2: Intermediate

4. **LC Orbit:** Find the complete LC orbit of the 4-vertex path graph.

5. **Edge-LC:** For the square graph:
   a) Compute G*0*1*0 (edge-LC on edge {0,1})
   b) Verify the edge {0,1} is preserved

6. **Unitary Verification:** Show that $U_a^\dagger K_a U_a$ transforms appropriately under LC.

### Level 3: Challenging

7. **LC Invariants:** Prove that the number of edges mod 2 is NOT an LC invariant. Find a graph property that IS an LC invariant.

8. **LC Orbit Size:** Prove that the LC orbit of the complete graph $K_n$ has size at most $n$.

9. **Interlacement:** Research and explain how the interlacement graph relates to LC equivalence.

---

## Solutions

### Level 1 Solutions

1. **Path LC:**
   Path: $\Gamma = \begin{pmatrix} 0&1&0&0 \\ 1&0&1&0 \\ 0&1&0&1 \\ 0&0&1&0 \end{pmatrix}$

   a) LC at 1: N(1) = {0, 2}, toggle {0,2}:
   $$\Gamma*1 = \begin{pmatrix} 0&1&1&0 \\ 1&0&1&0 \\ 1&1&0&1 \\ 0&0&1&0 \end{pmatrix}$$

   b) LC at 2: N(2) = {1, 3}, toggle {1,3}:
   $$\Gamma*2 = \begin{pmatrix} 0&1&0&0 \\ 1&0&1&1 \\ 0&1&0&1 \\ 0&1&1&0 \end{pmatrix}$$

---

## Computational Lab

```python
"""
Day 738: Local Complementation
==============================
Implementation of LC operations on graphs.
"""

import numpy as np
from typing import List, Tuple, Set, FrozenSet
from itertools import combinations

def adjacency_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Create adjacency matrix from edges."""
    Gamma = np.zeros((n, n), dtype=int)
    for i, j in edges:
        Gamma[i, j] = 1
        Gamma[j, i] = 1
    return Gamma

def edges_from_matrix(Gamma: np.ndarray) -> List[Tuple[int, int]]:
    """Extract edge list from adjacency matrix."""
    n = len(Gamma)
    return [(i, j) for i in range(n) for j in range(i+1, n) if Gamma[i,j] == 1]

def neighborhood(Gamma: np.ndarray, a: int) -> List[int]:
    """Return neighbors of vertex a."""
    return [b for b in range(len(Gamma)) if Gamma[a, b] == 1]

def local_complement(Gamma: np.ndarray, a: int) -> np.ndarray:
    """
    Perform local complementation at vertex a.

    Complements all edges within N(a).
    """
    Gamma_new = Gamma.copy()
    N_a = neighborhood(Gamma, a)

    # Toggle edges within N(a)
    for b in N_a:
        for c in N_a:
            if b < c:
                Gamma_new[b, c] = 1 - Gamma_new[b, c]
                Gamma_new[c, b] = 1 - Gamma_new[c, b]

    return Gamma_new

def edge_local_complement(Gamma: np.ndarray, a: int, b: int) -> np.ndarray:
    """
    Edge-local complementation: G*a*b*a
    """
    G1 = local_complement(Gamma, a)
    G2 = local_complement(G1, b)
    G3 = local_complement(G2, a)
    return G3

def graph_hash(Gamma: np.ndarray) -> Tuple:
    """Create hashable representation of graph (for orbit computation)."""
    return tuple(map(tuple, Gamma))

def compute_lc_orbit(Gamma: np.ndarray) -> List[np.ndarray]:
    """
    Compute the complete LC orbit of a graph.

    Returns list of all LC-equivalent graphs (as adjacency matrices).
    """
    n = len(Gamma)
    orbit = {graph_hash(Gamma): Gamma.copy()}
    queue = [Gamma.copy()]

    while queue:
        G = queue.pop(0)
        for a in range(n):
            G_new = local_complement(G, a)
            h = graph_hash(G_new)
            if h not in orbit:
                orbit[h] = G_new.copy()
                queue.append(G_new)

    return list(orbit.values())

def describe_graph(Gamma: np.ndarray) -> str:
    """Simple description of graph structure."""
    n = len(Gamma)
    num_edges = sum(Gamma[i,j] for i in range(n) for j in range(i+1, n))
    degrees = [sum(Gamma[i]) for i in range(n)]
    return f"n={n}, edges={num_edges}, degrees={sorted(degrees)}"

def is_isomorphic_simple(Gamma1: np.ndarray, Gamma2: np.ndarray) -> bool:
    """
    Simple isomorphism check by comparing degree sequences.
    (Not complete - just for demonstration)
    """
    deg1 = sorted([sum(Gamma1[i]) for i in range(len(Gamma1))])
    deg2 = sorted([sum(Gamma2[i]) for i in range(len(Gamma2))])
    return deg1 == deg2

def lc_unitary_matrix(Gamma: np.ndarray, a: int) -> np.ndarray:
    """
    Compute the unitary U_a for LC at vertex a.

    U_a = exp(-i π/4 X_a) ∏_{b ∈ N(a)} sqrt(Z_b)
    """
    n = len(Gamma)
    N_a = neighborhood(Gamma, a)

    # Single-qubit gates
    sqrt_Z = np.array([[1, 0], [0, 1j]]) / np.sqrt(1)  # Actually e^{iπ/4 Z}
    sqrt_Z = np.array([[np.exp(1j*np.pi/4), 0], [0, np.exp(-1j*np.pi/4)]])

    exp_X = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)  # e^{-iπ/4 X}

    # Build full unitary (simplified for small n)
    I = np.eye(2, dtype=complex)

    # Start with identity
    U = np.eye(2**n, dtype=complex)

    # Apply sqrt(Z) to neighbors
    for b in N_a:
        # Build tensor product
        gate = np.array([1], dtype=complex)
        for i in range(n):
            if i == b:
                gate = np.kron(gate, sqrt_Z)
            else:
                gate = np.kron(gate, I)
        # gate is now 2^n × 2^n diagonal
        U = np.diag(np.diag(gate)) @ U

    # Apply exp(-iπ/4 X) to vertex a
    gate_X = np.array([1], dtype=complex)
    for i in range(n):
        if i == a:
            gate_X = np.kron(gate_X, exp_X)
        else:
            gate_X = np.kron(gate_X, I)
    U = gate_X.reshape(2**n, 2**n) @ U

    return U

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 738: Local Complementation")
    print("=" * 60)

    # Example 1: Triangle
    print("\n1. Local Complementation on Triangle (K₃)")
    print("-" * 40)

    Gamma_K3 = adjacency_matrix(3, [(0,1), (1,2), (0,2)])
    print(f"Original K₃:\n{Gamma_K3}")

    Gamma_K3_lc0 = local_complement(Gamma_K3, 0)
    print(f"\nAfter LC at vertex 0:\n{Gamma_K3_lc0}")
    print(f"Edges: {edges_from_matrix(Gamma_K3_lc0)}")
    print("→ Star graph!")

    # Verify involution
    Gamma_back = local_complement(Gamma_K3_lc0, 0)
    print(f"\nAfter LC at 0 again:\n{Gamma_back}")
    print(f"Back to K₃: {np.array_equal(Gamma_back, Gamma_K3)}")

    # Example 2: Path graph
    print("\n2. Local Complementation on Path (4 vertices)")
    print("-" * 40)

    Gamma_path = adjacency_matrix(4, [(0,1), (1,2), (2,3)])
    print(f"Original path 0-1-2-3:")
    print(f"Edges: {edges_from_matrix(Gamma_path)}")

    Gamma_path_lc1 = local_complement(Gamma_path, 1)
    print(f"\nAfter LC at vertex 1:")
    print(f"Edges: {edges_from_matrix(Gamma_path_lc1)}")

    # Example 3: LC orbit of triangle
    print("\n3. LC Orbit of Triangle")
    print("-" * 40)

    orbit_K3 = compute_lc_orbit(Gamma_K3)
    print(f"Orbit size: {len(orbit_K3)}")
    for i, G in enumerate(orbit_K3):
        print(f"  Graph {i}: {describe_graph(G)}")

    # Example 4: LC orbit of path
    print("\n4. LC Orbit of 4-Vertex Path")
    print("-" * 40)

    orbit_path = compute_lc_orbit(Gamma_path)
    print(f"Orbit size: {len(orbit_path)}")
    for i, G in enumerate(orbit_path[:6]):  # Show first 6
        print(f"  Graph {i}: edges={edges_from_matrix(G)}")
    if len(orbit_path) > 6:
        print(f"  ... and {len(orbit_path) - 6} more")

    # Example 5: Edge-local complementation
    print("\n5. Edge-Local Complementation")
    print("-" * 40)

    Gamma_square = adjacency_matrix(4, [(0,1), (1,2), (2,3), (3,0)])
    print(f"Square graph edges: {edges_from_matrix(Gamma_square)}")

    Gamma_elc = edge_local_complement(Gamma_square, 0, 1)
    print(f"After edge-LC on {0,1}:")
    print(f"Edges: {edges_from_matrix(Gamma_elc)}")
    print(f"Edge {{0,1}} preserved: {Gamma_elc[0,1] == 1}")

    # Example 6: LC orbit sizes for various graphs
    print("\n6. LC Orbit Sizes")
    print("-" * 40)

    graphs = [
        ("K₃", adjacency_matrix(3, [(0,1), (1,2), (0,2)])),
        ("Path₄", adjacency_matrix(4, [(0,1), (1,2), (2,3)])),
        ("Square", adjacency_matrix(4, [(0,1), (1,2), (2,3), (3,0)])),
        ("K₄", adjacency_matrix(4, [(i,j) for i in range(4) for j in range(i+1,4)])),
        ("Star₄", adjacency_matrix(4, [(0,1), (0,2), (0,3)])),
    ]

    for name, Gamma in graphs:
        orbit = compute_lc_orbit(Gamma)
        print(f"  {name}: orbit size = {len(orbit)}")

    print("\n" + "=" * 60)
    print("End of Day 738 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Local complement | Toggle edges within N(a) |
| Unitary | $U_a = e^{-i\frac{\pi}{4}X_a} \prod_{b \in N(a)} \sqrt{Z_b}$ |
| State relation | $\|G*a\rangle = U_a^\dagger \|G\rangle$ |
| Involution | $(G*a)*a = G$ |
| Edge-LC | $G*(a,b) = G*a*b*a$ |

### Main Takeaways

1. **Local complementation** toggles edges within a neighborhood
2. **LC corresponds** to local Clifford unitaries on graph states
3. **LC-equivalent graphs** give LCequivalent quantum states
4. **The LC orbit** characterizes local Clifford equivalence classes
5. **Edge-LC** preserves the pivot edge

---

## Daily Checklist

- [ ] I can perform local complementation on a graph
- [ ] I understand the effect on the adjacency matrix
- [ ] I know the corresponding unitary transformation
- [ ] I can compute LC orbits
- [ ] I understand LC as an involution
- [ ] I can identify LC-equivalent graphs

---

## Preview: Day 739

Tomorrow we explore **Local Clifford Equivalence**:
- Classification of graph states by LC classes
- LC invariants and orbit representatives
- Entanglement under LC operations
- Computational complexity of LC equivalence
