# Day 739: Local Clifford Equivalence

## Overview

**Day:** 739 of 1008
**Week:** 106 (Graph States & MBQC)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Classification of Graph States by LC Equivalence

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | LC equivalence theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Classification and invariants |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Classify** graph states by LC equivalence classes
2. **Identify** LC invariants for graphs
3. **Understand** the relationship to entanglement
4. **Compute** canonical representatives
5. **Analyze** complexity of LC equivalence testing
6. **Apply** classification to small graphs

---

## Core Content

### The LC Equivalence Problem

**Question:** Given graphs G and H, are they LC-equivalent?

**Equivalently:** Are |G⟩ and |H⟩ related by single-qubit Clifford operations?

**Importance:**
- Classifies graph states by entanglement type
- Determines resource equivalence in MBQC
- Connects graph theory to quantum information

### LC Equivalence Theorem

**Theorem (Van den Nest, Dehaene, De Moor 2004):**
Two graph states |G⟩ and |H⟩ are **local Clifford equivalent** if and only if G and H are **LC-equivalent** as graphs.

$$|G\rangle \sim_{LC} |H\rangle \iff G \sim_{LC} H$$

This reduces quantum equivalence to graph-theoretic equivalence!

### LC-Invariants

**Definition:**
An **LC-invariant** is a graph property preserved under local complementation.

**Examples of LC-invariants:**
1. Number of vertices n
2. Rank of adjacency matrix over F₂
3. Certain graph polynomials

**Non-invariants:**
- Number of edges (changes under LC)
- Maximum degree
- Chromatic number

### The Interlacement Graph

**Definition:**
For graph G with edge {a,b}, the **interlacement** of a and b is:
$$I(a,b) = |N(a) \triangle N(b) \setminus \{a,b\}| \pmod{2}$$

where △ is symmetric difference.

**Interlacement Graph:**
$\mathcal{I}(G)$ has:
- Same vertices as G
- Edge {a,b} if I(a,b) = 1 (odd interlacement)

**Key Theorem:**
$G \sim_{LC} H$ implies $\mathcal{I}(G) = \mathcal{I}(H)$ (up to isomorphism).

### Rank as LC-Invariant

**Theorem:**
The F₂-rank of the adjacency matrix is an LC-invariant:
$$\text{rank}_{\mathbb{F}_2}(\Gamma_G) = \text{rank}_{\mathbb{F}_2}(\Gamma_H) \text{ if } G \sim_{LC} H$$

**Proof Sketch:**
LC at vertex a modifies Γ by adding rank-1 matrices (outer products), which can change rank by at most 1. But the specific structure ensures rank is preserved.

### Entanglement Under LC

**Key Insight:** LC operations are local unitaries, so they cannot change:
- Entanglement measures that are LOCC-monotone
- Schmidt rank for any bipartition

**However:** The entanglement *structure* can look different in different representations.

**Example:** The triangle K₃ and star S₃ are LC-equivalent, but:
- K₃ looks "more symmetric"
- S₃ has a distinguished center

Yet they have identical entanglement properties!

### LC Orbit Representatives

**Canonical Form:**
For classification, we need canonical representatives of LC orbits.

**Approaches:**
1. **Lexicographically smallest:** Order graphs and pick smallest in orbit
2. **Minimum edges:** Pick graph with fewest edges
3. **Special structure:** Stars, complete graphs, etc.

### Classification of Small Graphs

**n = 2:**
- 1 orbit: Either edge or no edge (no edge is trivial)

**n = 3:**
- 2 non-trivial orbits:
  - {K₃, Star₃} (3 edges ↔ 2 edges)
  - {Path₃} (2 edges, different structure)

**n = 4:**
- Multiple orbits including paths, cycles, stars, complete, etc.

### Computational Complexity

**LC-Equivalence Testing:**
- Known to be in **P** (polynomial time)
- Algorithm uses interlacement and rank checks

**Finding LC Orbit:**
- Can be exponential in worst case
- Practical algorithms exist for moderate n

### The Vertex-Minor Relationship

**Definition:**
H is a **vertex-minor** of G if H can be obtained from G by:
- Local complementations
- Vertex deletions

**Significance:** Vertex-minors give a partial order on graphs relevant to entanglement.

### Connection to SLOCC

**SLOCC:** Stochastic Local Operations and Classical Communication

**Theorem:**
Two graph states are SLOCC-equivalent iff their graphs have the same vertex-minor closure (roughly).

This connects LC-equivalence to broader entanglement classification.

---

## Worked Examples

### Example 1: Classifying 3-Vertex Graphs

All simple graphs on 3 vertices:
1. Empty: ○ ○ ○ (trivial, no entanglement)
2. Single edge: ○—○ ○
3. Path: ○—○—○
4. Triangle: △

**LC orbit analysis:**

**Single edge:** LC at any vertex doesn't change structure (N = {other endpoint})
- Orbit: {single edge}

**Path 1—2—3:**
- LC at 1: N(1) = {2}, toggle nothing → same graph
- LC at 2: N(2) = {1,3}, toggle {1,3} → triangle!
- Path ~LC Triangle

**But wait:** Earlier we showed K₃ ~LC Star, and Star is Path!

So all connected 3-vertex graphs are LC-equivalent!

**Orbits on 3 vertices:**
1. {Empty}
2. {Single edge + isolated}
3. {Path, Triangle, Star} — all connected!

### Example 2: 4-Vertex Classification

**Path P₄: 0—1—2—3**
- LC at 1: adds edge {0,2}
- Can reach many configurations

**Square C₄:**
- LC operations transform within a family

**Star S₄:**
- Central vertex has degree 3

**Complete K₄:**
- Very symmetric, small orbit

Different orbits represent genuinely different entanglement types.

### Example 3: Computing Rank Invariant

**Triangle K₃:**
$$\Gamma_{K_3} = \begin{pmatrix} 0&1&1 \\ 1&0&1 \\ 1&1&0 \end{pmatrix}$$

Over F₂: rank = 2 (rows sum to zero)

**Star S₃:**
$$\Gamma_{S_3} = \begin{pmatrix} 0&1&1 \\ 1&0&0 \\ 1&0&0 \end{pmatrix}$$

Over F₂: rank = 2 (two independent rows)

Same rank confirms possible LC-equivalence ✓

### Example 4: Interlacement Calculation

**For path 0—1—2:**
- I(0,1): N(0)△N(1)\{0,1} = {1}△{0,2}\{0,1} = {2} → |{2}| = 1 (odd)
- I(0,2): N(0)△N(2)\{0,2} = {1}△{1}\{0,2} = {} → 0 (even)
- I(1,2): N(1)△N(2)\{1,2} = {0,2}△{1}\{1,2} = {0} → 1 (odd)

Interlacement graph: edges {0,1} and {1,2} → path again!

---

## Practice Problems

### Level 1: Direct Application

1. **Orbit Identification:** Determine if these pairs are LC-equivalent:
   a) Path P₃ and triangle K₃
   b) Star S₄ and path P₄
   c) Cycle C₄ and K₄

2. **Rank Computation:** Compute rank over F₂ for:
   a) Path P₄
   b) Cycle C₄
   c) Complete bipartite K₂,₂

3. **LC Invariant Check:** Is the number of triangles (3-cliques) an LC-invariant?

### Level 2: Intermediate

4. **Interlacement Graph:** Compute the interlacement graph for:
   a) Cycle C₄
   b) Star S₄

5. **Orbit Enumeration:** List all graphs in the LC orbit of the 4-vertex path.

6. **Canonical Representative:** For the orbit containing K₃, which graph has minimum edges?

### Level 3: Challenging

7. **Orbit Counting:** How many distinct LC orbits exist for connected graphs on 5 vertices?

8. **Invariant Proof:** Prove that the F₂-rank of the adjacency matrix is LC-invariant.

9. **Entanglement:** Show that two LC-equivalent graph states have the same entanglement entropy for any bipartition.

---

## Computational Lab

```python
"""
Day 739: Local Clifford Equivalence
===================================
Classification and invariants for LC equivalence.
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from itertools import combinations, permutations

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

def neighborhood(Gamma: np.ndarray, a: int) -> Set[int]:
    """Return neighbors of vertex a as set."""
    return {b for b in range(len(Gamma)) if Gamma[a, b] == 1}

def local_complement(Gamma: np.ndarray, a: int) -> np.ndarray:
    """Perform local complementation at vertex a."""
    Gamma_new = Gamma.copy()
    N_a = list(neighborhood(Gamma, a))
    for i, b in enumerate(N_a):
        for c in N_a[i+1:]:
            Gamma_new[b, c] = 1 - Gamma_new[b, c]
            Gamma_new[c, b] = 1 - Gamma_new[c, b]
    return Gamma_new

def rank_f2(M: np.ndarray) -> int:
    """Compute rank of matrix over F₂."""
    M = M.copy() % 2
    n_rows, n_cols = M.shape
    rank = 0
    for col in range(n_cols):
        # Find pivot
        pivot = None
        for row in range(rank, n_rows):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap rows
        M[[rank, pivot]] = M[[pivot, rank]]
        # Eliminate
        for row in range(n_rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank

def interlacement(Gamma: np.ndarray, a: int, b: int) -> int:
    """
    Compute interlacement of vertices a and b.

    I(a,b) = |N(a) △ N(b) \ {a,b}| mod 2
    """
    N_a = neighborhood(Gamma, a)
    N_b = neighborhood(Gamma, b)
    sym_diff = N_a.symmetric_difference(N_b) - {a, b}
    return len(sym_diff) % 2

def interlacement_graph(Gamma: np.ndarray) -> np.ndarray:
    """Compute the interlacement graph."""
    n = len(Gamma)
    I_graph = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(a+1, n):
            if interlacement(Gamma, a, b) == 1:
                I_graph[a, b] = 1
                I_graph[b, a] = 1
    return I_graph

def graph_hash(Gamma: np.ndarray) -> Tuple:
    """Create hashable representation."""
    return tuple(map(tuple, Gamma))

def compute_lc_orbit(Gamma: np.ndarray) -> List[np.ndarray]:
    """Compute complete LC orbit."""
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

def canonical_representative(orbit: List[np.ndarray]) -> np.ndarray:
    """Find canonical (lexicographically smallest) representative."""
    def graph_key(Gamma):
        return tuple(tuple(row) for row in Gamma)
    return min(orbit, key=graph_key)

def min_edges_representative(orbit: List[np.ndarray]) -> np.ndarray:
    """Find representative with minimum edges."""
    def num_edges(Gamma):
        return sum(sum(Gamma)) // 2
    return min(orbit, key=num_edges)

def are_lc_equivalent(G1: np.ndarray, G2: np.ndarray) -> bool:
    """Check if two graphs are LC-equivalent."""
    # Quick checks first
    if len(G1) != len(G2):
        return False
    if rank_f2(G1) != rank_f2(G2):
        return False

    # Check if G2 is in orbit of G1
    orbit = compute_lc_orbit(G1)
    h2 = graph_hash(G2)
    return any(graph_hash(G) == h2 for G in orbit)

def describe_graph(Gamma: np.ndarray) -> str:
    """Describe graph structure."""
    n = len(Gamma)
    edges = sum(sum(Gamma)) // 2
    degrees = sorted([sum(Gamma[i]) for i in range(n)], reverse=True)
    return f"n={n}, e={edges}, deg={degrees}"

# Classification functions
def all_connected_graphs(n: int) -> List[np.ndarray]:
    """Generate all connected simple graphs on n vertices."""
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    graphs = []

    for num_edges in range(n-1, len(all_possible_edges)+1):
        for edge_subset in combinations(all_possible_edges, num_edges):
            Gamma = adjacency_matrix(n, list(edge_subset))
            # Check connectivity (simplified)
            if is_connected(Gamma):
                graphs.append(Gamma)

    return graphs

def is_connected(Gamma: np.ndarray) -> bool:
    """Check if graph is connected using BFS."""
    n = len(Gamma)
    if n == 0:
        return True
    visited = {0}
    queue = [0]
    while queue:
        v = queue.pop(0)
        for u in range(n):
            if Gamma[v, u] == 1 and u not in visited:
                visited.add(u)
                queue.append(u)
    return len(visited) == n

def classify_by_lc(graphs: List[np.ndarray]) -> Dict[Tuple, List[np.ndarray]]:
    """Classify graphs by LC equivalence."""
    classes = {}
    assigned = set()

    for i, G in enumerate(graphs):
        if i in assigned:
            continue

        orbit = compute_lc_orbit(G)
        rep = graph_hash(canonical_representative(orbit))

        # Find all graphs in this orbit
        class_members = []
        for j, H in enumerate(graphs):
            if j not in assigned:
                if any(graph_hash(G_orbit) == graph_hash(H) for G_orbit in orbit):
                    class_members.append(H)
                    assigned.add(j)

        classes[rep] = class_members

    return classes

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 739: Local Clifford Equivalence")
    print("=" * 60)

    # Example 1: LC equivalence check
    print("\n1. LC Equivalence Testing")
    print("-" * 40)

    K3 = adjacency_matrix(3, [(0,1), (1,2), (0,2)])
    P3 = adjacency_matrix(3, [(0,1), (1,2)])
    S3 = adjacency_matrix(3, [(0,1), (0,2)])

    print(f"K₃ ~ P₃: {are_lc_equivalent(K3, P3)}")
    print(f"K₃ ~ S₃: {are_lc_equivalent(K3, S3)}")
    print(f"P₃ ~ S₃: {are_lc_equivalent(P3, S3)}")

    # Example 2: LC invariants
    print("\n2. LC Invariant: F₂ Rank")
    print("-" * 40)

    graphs_3 = [
        ("K₃", K3),
        ("P₃", P3),
        ("S₃", S3),
    ]

    for name, G in graphs_3:
        print(f"rank(Γ_{name}) = {rank_f2(G)}")

    # Example 3: Interlacement graph
    print("\n3. Interlacement Graphs")
    print("-" * 40)

    print("Path P₃ interlacement:")
    I_P3 = interlacement_graph(P3)
    print(f"  Edges: {edges_from_matrix(I_P3)}")

    print("\nTriangle K₃ interlacement:")
    I_K3 = interlacement_graph(K3)
    print(f"  Edges: {edges_from_matrix(I_K3)}")

    # Example 4: Orbit representatives
    print("\n4. Orbit Representatives")
    print("-" * 40)

    orbit_K3 = compute_lc_orbit(K3)
    print(f"K₃ orbit size: {len(orbit_K3)}")

    min_rep = min_edges_representative(orbit_K3)
    print(f"Minimum edges representative: {edges_from_matrix(min_rep)}")

    # Example 5: 4-vertex classification
    print("\n5. 4-Vertex Graph Analysis")
    print("-" * 40)

    graphs_4 = [
        ("P₄", adjacency_matrix(4, [(0,1), (1,2), (2,3)])),
        ("C₄", adjacency_matrix(4, [(0,1), (1,2), (2,3), (3,0)])),
        ("S₄", adjacency_matrix(4, [(0,1), (0,2), (0,3)])),
        ("K₄", adjacency_matrix(4, [(i,j) for i in range(4) for j in range(i+1,4)])),
    ]

    print("F₂ ranks:")
    for name, G in graphs_4:
        print(f"  rank(Γ_{name}) = {rank_f2(G)}")

    print("\nLC orbit sizes:")
    for name, G in graphs_4:
        orbit = compute_lc_orbit(G)
        print(f"  |[{name}]_LC| = {len(orbit)}")

    print("\nEquivalence check:")
    for i, (name1, G1) in enumerate(graphs_4):
        for name2, G2 in graphs_4[i+1:]:
            equiv = are_lc_equivalent(G1, G2)
            if equiv:
                print(f"  {name1} ~ {name2}")

    # Example 6: Entanglement properties
    print("\n6. Entanglement (Rank = Schmidt Rank for Bipartition)")
    print("-" * 40)

    # For graph states, rank of Γ_AB gives Schmidt rank
    C4 = adjacency_matrix(4, [(0,1), (1,2), (2,3), (3,0)])

    # Bipartition {0,1} | {2,3}
    Gamma_01_23 = C4[:2, 2:]
    print(f"C₄ bipartition {{0,1}}|{{2,3}}: rank = {rank_f2(Gamma_01_23)}")

    # Bipartition {0,2} | {1,3}
    perm = [0, 2, 1, 3]
    C4_perm = C4[np.ix_(perm, perm)]
    Gamma_02_13 = C4_perm[:2, 2:]
    print(f"C₄ bipartition {{0,2}}|{{1,3}}: rank = {rank_f2(Gamma_02_13)}")

    print("\n" + "=" * 60)
    print("End of Day 739 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| LC equivalence | $G \sim_{LC} H \Leftrightarrow \|G\rangle \sim_{LC} \|H\rangle$ |
| Interlacement | $I(a,b) = \|N(a) \triangle N(b) \setminus \{a,b\}\| \mod 2$ |
| Rank invariant | $\text{rank}(\Gamma_G) = \text{rank}(\Gamma_H)$ if $G \sim_{LC} H$ |

### Main Takeaways

1. **LC equivalence** is decidable in polynomial time
2. **F₂ rank** and **interlacement graph** are LC-invariants
3. **LC-equivalent states** have identical entanglement properties
4. **Classification** reduces infinite families to finite orbits
5. **Small graphs** can be fully classified

---

## Daily Checklist

- [ ] I understand LC equivalence for graph states
- [ ] I can compute LC invariants (rank, interlacement)
- [ ] I can test LC equivalence of graphs
- [ ] I know about canonical representatives
- [ ] I understand the connection to entanglement
- [ ] I can classify small graphs by LC orbits

---

## Preview: Day 740

Tomorrow we begin **Measurement-Based Quantum Computation**:
- Single-qubit measurements on graph states
- Byproduct operators and corrections
- Teleportation as MBQC primitive
- Gate implementation via measurement
