# Day 736: Introduction to Graph States

## Overview

**Day:** 736 of 1008
**Week:** 106 (Graph States & MBQC)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Graph States — Definition and Construction

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Graph state theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Examples and properties |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define** graph states from graphs and CZ gates
2. **Construct** graph states using the adjacency matrix
3. **Compute** simple graph state examples
4. **Identify** key properties of graph states
5. **Connect** graphs to quantum entanglement
6. **Implement** graph state construction algorithms

---

## Core Content

### From Graphs to Quantum States

**Motivation:**
Graph states provide a beautiful correspondence between combinatorial graph theory and quantum entanglement. Every simple undirected graph defines a unique quantum state (up to local unitaries).

**Definition (Graph):**
A graph G = (V, E) consists of:
- V = {1, 2, ..., n}: set of vertices
- E ⊆ {{i, j} : i, j ∈ V, i ≠ j}: set of edges

**Adjacency Matrix:**
The adjacency matrix Γ ∈ {0,1}^{n×n} has:
$$\Gamma_{ij} = \begin{cases} 1 & \text{if } \{i,j\} \in E \\ 0 & \text{otherwise} \end{cases}$$

Properties:
- Symmetric: Γ = Γᵀ
- Zero diagonal: Γᵢᵢ = 0 (no self-loops)

### The CZ Gate

**Definition:**
The controlled-Z (CZ) gate on qubits i and j:
$$CZ_{ij} = |00\rangle\langle00| + |01\rangle\langle01| + |10\rangle\langle10| - |11\rangle\langle11|$$

**Matrix form:**
$$CZ = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

**Key Properties:**
- Symmetric: $CZ_{ij} = CZ_{ji}$
- Self-inverse: $CZ^2 = I$
- Diagonal in computational basis
- Creates entanglement: $CZ|++\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle - |11\rangle)$

### Graph State Definition

**Definition (Graph State):**
Given graph G = (V, E) with n vertices, the graph state |G⟩ is:

$$\boxed{|G\rangle = \prod_{(i,j) \in E} CZ_{ij} |+\rangle^{\otimes n}}$$

**Construction Process:**
1. Initialize all n qubits in |+⟩ = (|0⟩ + |1⟩)/√2
2. For each edge {i, j} ∈ E, apply CZ to qubits i and j
3. Order of CZ gates doesn't matter (they all commute)

**Alternative Form:**
Using the adjacency matrix Γ:
$$|G\rangle = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} (-1)^{q(x)} |x\rangle$$

where $q(x) = \frac{1}{2} x^T \Gamma x = \sum_{i<j} \Gamma_{ij} x_i x_j$

### Simple Graph State Examples

#### Example 1: Two-Qubit Graph (Single Edge)

Graph: ○—○ (vertices 1 and 2 connected)

$$|G\rangle = CZ_{12}|++\rangle = CZ \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$
$$= \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle - |11\rangle)$$

This is the Bell state $|\Phi^+\rangle$ up to local Hadamard!

Actually: $|G\rangle = \frac{1}{\sqrt{2}}(|0+\rangle + |1-\rangle) = \frac{1}{\sqrt{2}}(|+0\rangle + |-1\rangle)$

#### Example 2: Three-Qubit Linear Graph

Graph: ○—○—○ (linear chain)
Edges: {1,2}, {2,3}

$$|G\rangle = CZ_{23} CZ_{12} |+++\rangle$$

Step 1: $CZ_{12}|+++\rangle$
$$= \frac{1}{2\sqrt{2}}(|00\rangle + |01\rangle + |10\rangle - |11\rangle) \otimes |+\rangle$$

Step 2: Apply $CZ_{23}$...

**Final state:**
$$|G_{\text{linear}}\rangle = \frac{1}{2\sqrt{2}}(|000\rangle + |001\rangle + |010\rangle - |011\rangle + |100\rangle - |101\rangle + |110\rangle + |111\rangle)$$

#### Example 3: Three-Qubit Complete Graph (Triangle)

Graph: △ (all pairs connected)
Edges: {1,2}, {2,3}, {1,3}

$$|K_3\rangle = CZ_{13} CZ_{23} CZ_{12} |+++\rangle$$

This is the GHZ-class state!

#### Example 4: Star Graph

Graph: Central vertex connected to all others
$$\text{Star}_n: \quad \circ \leftarrow \circ \rightarrow \circ$$

For n = 4 (center + 3 leaves):
$$|S_4\rangle = CZ_{12} CZ_{13} CZ_{14} |++++\rangle$$

### Entanglement in Graph States

**Key Insight:** Edges encode entanglement.
- More edges → more entanglement
- Graph connectivity → entanglement structure

**Bipartite Entanglement:**
For a bipartition (A, B) of vertices:
$$S(\rho_A) = \text{rank of } \Gamma_{AB}$$

where $\Gamma_{AB}$ is the submatrix of edges crossing the cut.

### Graph State Properties

**1. Stabilizer States:**
Every graph state is a stabilizer state — it can be described by n commuting Pauli operators (stabilizers).

**2. Local Clifford Equivalence:**
Graph states differing by single-qubit Clifford operations form equivalence classes.

**3. Computational Basis Amplitudes:**
$$\langle x | G \rangle = \frac{1}{\sqrt{2^n}} (-1)^{q(x)}$$

where $q(x) = \sum_{i<j} \Gamma_{ij} x_i x_j$.

**4. Pauli Measurements:**
Measuring qubits of |G⟩ in Pauli bases yields other graph states (up to local operations).

### Neighborhood and Degree

**Neighborhood:**
$$N(a) = \{b \in V : \{a, b\} \in E\} = \{b : \Gamma_{ab} = 1\}$$

**Degree:**
$$\deg(a) = |N(a)| = \sum_b \Gamma_{ab}$$

These will be crucial for stabilizer generators!

---

## Worked Examples

### Example 1: Four-Qubit Square Graph

**Graph:** Square with vertices 1,2,3,4 and edges {1,2}, {2,3}, {3,4}, {4,1}

**Adjacency matrix:**
$$\Gamma = \begin{pmatrix} 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \end{pmatrix}$$

**Construction:**
$$|G_{\square}\rangle = CZ_{14} CZ_{34} CZ_{23} CZ_{12} |++++\rangle$$

**Amplitude calculation:**
For $|x\rangle = |1010\rangle$:
$$q(x) = \Gamma_{12}x_1x_2 + \Gamma_{14}x_1x_4 + \Gamma_{23}x_2x_3 + \Gamma_{34}x_3x_4$$
$$= 1 \cdot 1 \cdot 0 + 1 \cdot 1 \cdot 1 + 1 \cdot 0 \cdot 1 + 1 \cdot 1 \cdot 0 = 1$$

$$\langle 1010 | G_{\square} \rangle = \frac{1}{4}(-1)^1 = -\frac{1}{4}$$

### Example 2: Complete Graph K₄

**All pairs connected:** 6 edges

$$\Gamma = \begin{pmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{pmatrix}$$

For $|1111\rangle$:
$$q(1111) = 1+1+1+1+1+1 = 6 \equiv 0 \pmod{2}$$
$$\langle 1111 | K_4 \rangle = \frac{1}{4}(-1)^0 = +\frac{1}{4}$$

For $|1100\rangle$:
$$q(1100) = \Gamma_{12} = 1$$
$$\langle 1100 | K_4 \rangle = \frac{1}{4}(-1)^1 = -\frac{1}{4}$$

### Example 3: Verifying CZ Effect

Show that $CZ|++\rangle$ creates the expected entangled state.

$$|++\rangle = \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle + |1\rangle) = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$

Apply CZ (only affects |11⟩):
$$CZ|++\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle - |11\rangle)$$

Verify entanglement by computing $\rho_1 = \text{Tr}_2(|\psi\rangle\langle\psi|)$:
$$\rho_1 = \frac{1}{2}(|0\rangle\langle0| + |1\rangle\langle1|) = \frac{I}{2}$$

This is maximally mixed → maximal entanglement! ✓

---

## Practice Problems

### Level 1: Direct Application

1. **Graph Construction:** Draw the graph corresponding to:
   $$\Gamma = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}$$

2. **Adjacency Matrix:** Write the adjacency matrix for:
   a) A path graph on 4 vertices: 1—2—3—4
   b) A star graph with center 1 and leaves 2,3,4,5

3. **Amplitude Calculation:** For the triangle graph (K₃), compute $\langle 101 | K_3 \rangle$.

### Level 2: Intermediate

4. **State Construction:** Write out the full state vector for:
   a) The 3-vertex path graph 1—2—3
   b) The 3-vertex star with center 1

5. **Entanglement Check:** For the 4-vertex square graph:
   a) Compute the reduced density matrix $\rho_{12}$
   b) Is the (12)|(34) bipartition maximally entangled?

6. **CZ Decomposition:** Express CZ in terms of CNOT and single-qubit gates.

### Level 3: Challenging

7. **General Amplitude Formula:** Prove that $\langle x | G \rangle = \frac{1}{\sqrt{2^n}}(-1)^{x^T \Gamma x / 2}$.

8. **Entanglement Entropy:** For a graph state |G⟩ and bipartition (A, B), prove that:
   $$S(\rho_A) = \text{rank}_{\mathbb{F}_2}(\Gamma_{AB})$$

9. **Graph Isomorphism:** Show that if graphs G and G' are isomorphic, then |G⟩ and |G'⟩ are related by qubit permutation.

---

## Solutions

### Level 1 Solutions

1. **Graph:**
   ```
   1 — 2
   |
   3
   ```
   (Star with center 1, leaves 2 and 3)

2. **Adjacency Matrices:**
   a) Path: $\Gamma = \begin{pmatrix} 0&1&0&0 \\ 1&0&1&0 \\ 0&1&0&1 \\ 0&0&1&0 \end{pmatrix}$

   b) Star: $\Gamma = \begin{pmatrix} 0&1&1&1&1 \\ 1&0&0&0&0 \\ 1&0&0&0&0 \\ 1&0&0&0&0 \\ 1&0&0&0&0 \end{pmatrix}$

3. **Amplitude for K₃:**
   $q(101) = \Gamma_{12} \cdot 1 \cdot 0 + \Gamma_{13} \cdot 1 \cdot 1 + \Gamma_{23} \cdot 0 \cdot 1 = 0 + 1 + 0 = 1$
   $\langle 101 | K_3 \rangle = \frac{1}{2\sqrt{2}}(-1)^1 = -\frac{1}{2\sqrt{2}}$

---

## Computational Lab

```python
"""
Day 736: Introduction to Graph States
======================================
Implementation of graph state construction and analysis.
"""

import numpy as np
from typing import List, Tuple, Set
from itertools import combinations

def adjacency_matrix_from_edges(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Create adjacency matrix from edge list.

    Parameters:
    -----------
    n : int
        Number of vertices (0-indexed)
    edges : List[Tuple[int, int]]
        List of edges as (i, j) tuples

    Returns:
    --------
    Gamma : np.ndarray
        n×n adjacency matrix
    """
    Gamma = np.zeros((n, n), dtype=int)
    for i, j in edges:
        Gamma[i, j] = 1
        Gamma[j, i] = 1
    return Gamma

def edges_from_adjacency(Gamma: np.ndarray) -> List[Tuple[int, int]]:
    """Extract edge list from adjacency matrix."""
    n = len(Gamma)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if Gamma[i, j] == 1:
                edges.append((i, j))
    return edges

def cz_gate() -> np.ndarray:
    """Return 4×4 CZ gate matrix."""
    return np.diag([1, 1, 1, -1])

def apply_cz(state: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    """
    Apply CZ gate to qubits i and j in an n-qubit state.

    Parameters:
    -----------
    state : np.ndarray
        State vector of length 2^n
    i, j : int
        Qubit indices (0-indexed)
    n : int
        Total number of qubits

    Returns:
    --------
    new_state : np.ndarray
        State after CZ application
    """
    new_state = state.copy()
    for x in range(2**n):
        # Extract bits i and j
        bit_i = (x >> (n-1-i)) & 1
        bit_j = (x >> (n-1-j)) & 1
        # CZ adds phase -1 when both bits are 1
        if bit_i == 1 and bit_j == 1:
            new_state[x] *= -1
    return new_state

def plus_state(n: int) -> np.ndarray:
    """Create |+⟩^⊗n state."""
    return np.ones(2**n) / np.sqrt(2**n)

def create_graph_state(Gamma: np.ndarray) -> np.ndarray:
    """
    Create graph state |G⟩ from adjacency matrix.

    Parameters:
    -----------
    Gamma : np.ndarray
        n×n adjacency matrix

    Returns:
    --------
    state : np.ndarray
        State vector of length 2^n
    """
    n = len(Gamma)
    state = plus_state(n)

    # Apply CZ for each edge
    for i in range(n):
        for j in range(i+1, n):
            if Gamma[i, j] == 1:
                state = apply_cz(state, i, j, n)

    return state

def compute_amplitude(Gamma: np.ndarray, x: int) -> complex:
    """
    Compute amplitude ⟨x|G⟩ using formula.

    ⟨x|G⟩ = (1/√2^n) × (-1)^q(x)
    where q(x) = (1/2) x^T Γ x
    """
    n = len(Gamma)

    # Convert x to binary vector
    x_bits = np.array([(x >> (n-1-i)) & 1 for i in range(n)])

    # Compute q(x) = sum over edges
    q = 0
    for i in range(n):
        for j in range(i+1, n):
            if Gamma[i, j] == 1:
                q += x_bits[i] * x_bits[j]

    return ((-1)**q) / np.sqrt(2**n)

def neighborhood(Gamma: np.ndarray, a: int) -> List[int]:
    """Return neighbors of vertex a."""
    return [b for b in range(len(Gamma)) if Gamma[a, b] == 1]

def degree(Gamma: np.ndarray, a: int) -> int:
    """Return degree of vertex a."""
    return sum(Gamma[a, :])

def partial_trace(state: np.ndarray, keep: List[int], n: int) -> np.ndarray:
    """
    Compute partial trace, keeping specified qubits.

    Parameters:
    -----------
    state : np.ndarray
        State vector of length 2^n
    keep : List[int]
        Indices of qubits to keep
    n : int
        Total number of qubits

    Returns:
    --------
    rho : np.ndarray
        Reduced density matrix
    """
    # Reshape state into tensor
    psi = state.reshape([2]*n)

    # Create density matrix
    rho_full = np.outer(state, state.conj())

    # Trace out complement
    trace_out = [i for i in range(n) if i not in keep]

    k = len(keep)
    rho = np.zeros((2**k, 2**k), dtype=complex)

    for x_keep in range(2**k):
        for y_keep in range(2**k):
            # Convert to bit strings
            x_bits = [(x_keep >> (k-1-i)) & 1 for i in range(k)]
            y_bits = [(y_keep >> (k-1-i)) & 1 for i in range(k)]

            # Sum over traced out indices
            for z in range(2**(n-k)):
                z_bits = [(z >> (n-k-1-i)) & 1 for i in range(n-k)]

                # Build full indices
                x_full = 0
                y_full = 0
                keep_idx = 0
                trace_idx = 0

                for i in range(n):
                    if i in keep:
                        x_full += x_bits[keep_idx] << (n-1-i)
                        y_full += y_bits[keep_idx] << (n-1-i)
                        keep_idx += 1
                    else:
                        x_full += z_bits[trace_idx] << (n-1-i)
                        y_full += z_bits[trace_idx] << (n-1-i)
                        trace_idx += 1

                rho[x_keep, y_keep] += rho_full[x_full, y_full]

    return rho

def entanglement_entropy(state: np.ndarray, partition_A: List[int], n: int) -> float:
    """Compute entanglement entropy for bipartition."""
    rho_A = partial_trace(state, partition_A, n)
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
    return -np.sum(eigenvalues * np.log2(eigenvalues))

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 736: Introduction to Graph States")
    print("=" * 60)

    # Example 1: Two-qubit graph (Bell pair)
    print("\n1. Two-Qubit Graph (Edge)")
    print("-" * 40)

    Gamma_2 = adjacency_matrix_from_edges(2, [(0, 1)])
    state_2 = create_graph_state(Gamma_2)

    print(f"Adjacency matrix:\n{Gamma_2}")
    print(f"\nGraph state |G⟩:")
    for x in range(4):
        amp = state_2[x]
        if abs(amp) > 1e-10:
            bits = format(x, '02b')
            print(f"  |{bits}⟩: {amp:.4f}")

    # Example 2: Three-qubit linear graph
    print("\n2. Three-Qubit Linear Graph")
    print("-" * 40)

    Gamma_linear = adjacency_matrix_from_edges(3, [(0, 1), (1, 2)])
    state_linear = create_graph_state(Gamma_linear)

    print(f"Graph: 0—1—2")
    print(f"Adjacency matrix:\n{Gamma_linear}")
    print(f"\nState amplitudes:")
    for x in range(8):
        amp = state_linear[x]
        bits = format(x, '03b')
        print(f"  |{bits}⟩: {amp:+.4f}")

    # Verify with formula
    print("\nVerification with amplitude formula:")
    for x in range(8):
        amp_formula = compute_amplitude(Gamma_linear, x)
        amp_direct = state_linear[x]
        match = "✓" if abs(amp_formula - amp_direct) < 1e-10 else "✗"
        print(f"  |{format(x, '03b')}⟩: formula={amp_formula:+.4f}, direct={amp_direct:+.4f} {match}")

    # Example 3: Triangle (K3)
    print("\n3. Triangle Graph (K₃)")
    print("-" * 40)

    Gamma_K3 = adjacency_matrix_from_edges(3, [(0, 1), (1, 2), (0, 2)])
    state_K3 = create_graph_state(Gamma_K3)

    print(f"Graph: Complete K₃")
    print(f"State amplitudes:")
    for x in range(8):
        amp = state_K3[x]
        bits = format(x, '03b')
        print(f"  |{bits}⟩: {amp:+.4f}")

    # Example 4: Four-qubit square
    print("\n4. Four-Qubit Square Graph")
    print("-" * 40)

    Gamma_square = adjacency_matrix_from_edges(4, [(0,1), (1,2), (2,3), (3,0)])
    state_square = create_graph_state(Gamma_square)

    print(f"Graph: Square 0-1-2-3-0")
    print(f"\nNeighborhoods:")
    for v in range(4):
        print(f"  N({v}) = {neighborhood(Gamma_square, v)}")

    # Entanglement analysis
    print("\n5. Entanglement Analysis")
    print("-" * 40)

    # Two-qubit graph entanglement
    rho_0 = partial_trace(state_2, [0], 2)
    print(f"Two-qubit graph, reduced ρ₀:")
    print(f"{rho_0}")
    S = entanglement_entropy(state_2, [0], 2)
    print(f"Entanglement entropy S = {S:.4f} (expect 1.0 for max)")

    # Linear graph entanglement
    S_linear = entanglement_entropy(state_linear, [0], 3)
    print(f"\nLinear graph, S(ρ₀) = {S_linear:.4f}")

    S_linear_01 = entanglement_entropy(state_linear, [0, 1], 3)
    print(f"Linear graph, S(ρ₀₁) = {S_linear_01:.4f}")

    print("\n" + "=" * 60)
    print("End of Day 736 Lab")
    print("=" * 60)
```

**Expected Output:**
```
============================================================
Day 736: Introduction to Graph States
============================================================

1. Two-Qubit Graph (Edge)
----------------------------------------
Adjacency matrix:
[[0 1]
 [1 0]]

Graph state |G⟩:
  |00⟩: 0.5000
  |01⟩: 0.5000
  |10⟩: 0.5000
  |11⟩: -0.5000

2. Three-Qubit Linear Graph
----------------------------------------
Graph: 0—1—2
State amplitudes:
  |000⟩: +0.3536
  |001⟩: +0.3536
  |010⟩: +0.3536
  |011⟩: -0.3536
  |100⟩: +0.3536
  |101⟩: -0.3536
  |110⟩: +0.3536
  |111⟩: +0.3536
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Graph state | $\|G\rangle = \prod_{(i,j) \in E} CZ_{ij} \|+\rangle^{\otimes n}$ |
| Amplitude | $\langle x \| G \rangle = \frac{1}{\sqrt{2^n}}(-1)^{q(x)}$ |
| Quadratic form | $q(x) = \sum_{i<j} \Gamma_{ij} x_i x_j$ |
| Neighborhood | $N(a) = \{b : \Gamma_{ab} = 1\}$ |

### Main Takeaways

1. **Graph states** provide a direct mapping from graphs to quantum states
2. **CZ gates** create entanglement corresponding to edges
3. **Amplitudes** are determined by the quadratic form $q(x)$
4. **All graph states are stabilizer states** — efficiently describable
5. **Entanglement structure** mirrors graph connectivity

---

## Daily Checklist

- [ ] I can construct a graph state from its adjacency matrix
- [ ] I understand the role of CZ gates
- [ ] I can compute amplitudes using the quadratic form
- [ ] I know the relationship between edges and entanglement
- [ ] I implemented graph state construction
- [ ] I understand neighborhoods and degrees

---

## Preview: Day 737

Tomorrow we explore the **Stabilizer Structure of Graph States**:
- Deriving stabilizer generators from graph structure
- The formula $K_a = X_a \prod_{b \in N(a)} Z_b$
- Connection to the adjacency matrix
- Verifying stabilizer properties
