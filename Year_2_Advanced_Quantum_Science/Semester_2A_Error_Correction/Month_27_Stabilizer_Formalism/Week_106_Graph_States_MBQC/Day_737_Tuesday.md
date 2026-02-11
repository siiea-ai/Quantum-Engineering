# Day 737: Stabilizer Structure of Graph States

## Overview

**Day:** 737 of 1008
**Week:** 106 (Graph States & MBQC)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Stabilizer Generators from Graph Structure

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Stabilizer derivation |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Examples and verification |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Derive** stabilizer generators for any graph state
2. **Express** stabilizers using the adjacency matrix
3. **Verify** that graph state stabilizers commute
4. **Connect** graph structure to error correction properties
5. **Compute** the parity check matrix from the graph
6. **Understand** graph states as codewords

---

## Core Content

### The Graph State Stabilizer Formula

**Theorem (Graph State Stabilizers):**
The graph state |G⟩ associated with graph G = (V, E) is uniquely stabilized by the generators:

$$\boxed{K_a = X_a \prod_{b \in N(a)} Z_b \quad \text{for each vertex } a \in V}$$

where N(a) is the neighborhood of a.

**Compact Notation:**
$$K_a = X_a Z^{N(a)}$$

or using the adjacency matrix row $\Gamma_a$:
$$K_a = X_a Z^{\Gamma_a}$$

### Proof of the Stabilizer Property

**Need to show:** $K_a |G\rangle = |G\rangle$ for all $a$.

**Proof:**
Starting from $|G\rangle = \prod_{(i,j) \in E} CZ_{ij} |+\rangle^{\otimes n}$:

The state $|+\rangle^{\otimes n}$ is stabilized by $X_a$ for all $a$:
$$X_a |+\rangle^{\otimes n} = |+\rangle^{\otimes n}$$

Now consider how CZ gates transform $X_a$:
$$CZ_{ab} \cdot X_a \cdot CZ_{ab} = X_a Z_b$$

This is because $CZ X_a CZ = X_a Z_b$ (CZ conjugates X on one qubit to XZ).

After applying all CZ gates for edges incident to $a$:
$$\left(\prod_{(a,b) \in E} CZ_{ab}\right) X_a \left(\prod_{(a,b) \in E} CZ_{ab}\right)^{-1} = X_a \prod_{b \in N(a)} Z_b = K_a$$

Since $X_a$ stabilizes $|+\rangle^{\otimes n}$, we have $K_a$ stabilizes $|G\rangle$. □

### Binary Representation

**Stabilizer in binary form:**
$$K_a \leftrightarrow (e_a | \Gamma_a)$$

where:
- $e_a$ is the standard basis vector (1 in position a, 0 elsewhere)
- $\Gamma_a$ is row a of the adjacency matrix

**Parity check matrix:**
$$H = (I_n | \Gamma)$$

This is an n × 2n matrix with the identity on the left and adjacency matrix on the right.

### Verifying Commutation

**Theorem:** All graph state stabilizers commute.

**Proof using symplectic form:**
For stabilizers $K_a = (e_a | \Gamma_a)$ and $K_b = (e_b | \Gamma_b)$:

$$\langle K_a, K_b \rangle_s = e_a \cdot \Gamma_b + \Gamma_a \cdot e_b = \Gamma_{ab} + \Gamma_{ba} = 2\Gamma_{ab} = 0 \pmod{2}$$

The symmetry of $\Gamma$ ensures commutation! □

### Graph States as CSS-like Codes

**Observation:** Graph state stabilizers have a special structure:
- Each stabilizer has exactly one X
- The Z pattern is determined by the graph

This is related to (but not exactly) CSS structure.

**Graph State Code:**
|G⟩ can be viewed as a codeword of an [[n, 0, d]] code where:
- n = number of vertices
- k = 0 (single state, not a proper code)
- d depends on the graph structure

### Distance from Graph Properties

For a graph state viewed as a "code":

**Minimum vertex cover:** The distance relates to the minimum vertex cover.

**Specifically:** The minimum weight operator in $C(S) \setminus S$ corresponds to graph-theoretic properties.

### Special Graph Structures

#### Linear Graph (Path)
Vertices: 1—2—3—...—n
$$K_1 = X_1 Z_2$$
$$K_i = Z_{i-1} X_i Z_{i+1} \quad (1 < i < n)$$
$$K_n = Z_{n-1} X_n$$

#### Star Graph
Center c connected to all others:
$$K_c = X_c Z_1 Z_2 \cdots Z_{n-1}$$
$$K_i = Z_c X_i \quad (i \neq c)$$

#### Complete Graph $K_n$
Every pair connected:
$$K_a = X_a \prod_{b \neq a} Z_b$$

#### Cycle Graph $C_n$
$$K_a = Z_{a-1} X_a Z_{a+1} \quad (\text{mod } n)$$

### Relation to Error Correction

**Graph state codes:** Using |G⟩ as a code:
- X error on qubit a: Detected by $K_b$ for $b \in N(a)$
- Z error on qubit a: Detected by $K_a$

**Syndrome:**
For error $E$ with binary vector $(e_X | e_Z)$:
$$s_a = \langle K_a, E \rangle_s = (e_X)_a + \sum_{b \in N(a)} (e_Z)_b$$

---

## Worked Examples

### Example 1: Three-Qubit Linear Graph

**Graph:** 1—2—3

**Adjacency matrix:**
$$\Gamma = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

**Stabilizers:**
- $K_1 = X_1 Z_2$ (N(1) = {2})
- $K_2 = Z_1 X_2 Z_3$ (N(2) = {1, 3})
- $K_3 = Z_2 X_3$ (N(3) = {2})

**Binary form:**
- $K_1 = (1,0,0 | 0,1,0)$
- $K_2 = (0,1,0 | 1,0,1)$
- $K_3 = (0,0,1 | 0,1,0)$

**Parity check matrix:**
$$H = \begin{pmatrix}
1 & 0 & 0 & | & 0 & 1 & 0 \\
0 & 1 & 0 & | & 1 & 0 & 1 \\
0 & 0 & 1 & | & 0 & 1 & 0
\end{pmatrix} = (I_3 | \Gamma)$$

**Verify commutation:**
$\langle K_1, K_2 \rangle_s = (1,0,0)\cdot(1,0,1) + (0,1,0)\cdot(0,1,0) = 1 + 1 = 0$ ✓

### Example 2: Square Graph Stabilizers

**Graph:** Square with vertices 1,2,3,4

**Adjacency matrix:**
$$\Gamma = \begin{pmatrix} 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \end{pmatrix}$$

**Stabilizers:**
- $K_1 = X_1 Z_2 Z_4$
- $K_2 = Z_1 X_2 Z_3$
- $K_3 = Z_2 X_3 Z_4$
- $K_4 = Z_1 Z_3 X_4$

**Verify $K_1 |G_\square\rangle = |G_\square\rangle$:**
Apply $K_1 = X_1 Z_2 Z_4$ to the state. The $X_1$ flips bit 1, while $Z_2 Z_4$ applies phases.

For computational basis state $|x_1 x_2 x_3 x_4\rangle$:
$$K_1 |x_1 x_2 x_3 x_4\rangle = (-1)^{x_2 + x_4} |\bar{x}_1 x_2 x_3 x_4\rangle$$

The graph state is symmetric under this transformation! ✓

### Example 3: Star Graph Stabilizers

**Graph:** Star with center 0 and leaves 1, 2, 3

**Stabilizers:**
- $K_0 = X_0 Z_1 Z_2 Z_3$ (center)
- $K_1 = Z_0 X_1$
- $K_2 = Z_0 X_2$
- $K_3 = Z_0 X_3$

**Observations:**
- Center stabilizer has high weight (n)
- Leaf stabilizers have weight 2
- Similar to a repetition code structure!

**The star graph state is related to the GHZ state:**
$$|S_4\rangle \propto |0+++\rangle + |1---\rangle$$

### Example 4: Syndrome Calculation

For the linear graph 1—2—3, compute syndromes.

**Error $E = X_2$:**
$e = (0,1,0 | 0,0,0)$

Syndromes:
- $s_1 = \langle K_1, E \rangle_s = (1,0,0)\cdot(0,0,0) + (0,1,0)\cdot(0,1,0) = 1$
- $s_2 = \langle K_2, E \rangle_s = (0,1,0)\cdot(0,0,0) + (1,0,1)\cdot(0,1,0) = 0$
- $s_3 = \langle K_3, E \rangle_s = (0,0,1)\cdot(0,0,0) + (0,1,0)\cdot(0,1,0) = 1$

Syndrome: $(1, 0, 1)$

**Error $E = Z_2$:**
$e = (0,0,0 | 0,1,0)$

- $s_1 = (1,0,0)\cdot(0,1,0) + 0 = 1$
- $s_2 = (0,1,0)\cdot(0,1,0) + 0 = 1$
- $s_3 = (0,0,1)\cdot(0,1,0) + 0 = 0$

Syndrome: $(1, 1, 0)$

---

## Practice Problems

### Level 1: Direct Application

1. **Stabilizer Derivation:** Write stabilizers for:
   a) The 4-vertex path: 1—2—3—4
   b) The complete graph K₄

2. **Binary Form:** For the triangle graph (K₃), write:
   a) All stabilizer generators
   b) The parity check matrix H

3. **Neighborhood Check:** For the cycle graph C₅ (pentagon):
   a) List N(a) for each vertex
   b) Write stabilizer K₃

### Level 2: Intermediate

4. **Commutation Verification:** For the square graph:
   a) Verify $[K_1, K_3] = 0$ using the symplectic product
   b) Verify $[K_2, K_4] = 0$

5. **Syndrome Table:** For the triangle graph, compute syndromes for all single-qubit X and Z errors.

6. **Weight Analysis:** For the star graph $S_n$ (1 center, n-1 leaves):
   a) What is the weight of each stabilizer?
   b) What does this imply about error detection?

### Level 3: Challenging

7. **Distance Calculation:** For the linear graph on n vertices:
   a) What is the minimum weight non-stabilizer Pauli commuting with all K_a?
   b) Relate this to the "distance" of the graph state.

8. **CSS Connection:** Show that the stabilizers of a bipartite graph state have a CSS-like structure (X's on one part, Z's on the other).

9. **Graph Product:** For graphs G₁ and G₂, describe the stabilizers of the tensor product graph state |G₁⟩ ⊗ |G₂⟩ versus the graph union G₁ ∪ G₂.

---

## Solutions

### Level 1 Solutions

1. **Stabilizers:**
   a) Path 1—2—3—4:
      - $K_1 = X_1 Z_2$
      - $K_2 = Z_1 X_2 Z_3$
      - $K_3 = Z_2 X_3 Z_4$
      - $K_4 = Z_3 X_4$

   b) Complete K₄:
      - $K_1 = X_1 Z_2 Z_3 Z_4$
      - $K_2 = Z_1 X_2 Z_3 Z_4$
      - $K_3 = Z_1 Z_2 X_3 Z_4$
      - $K_4 = Z_1 Z_2 Z_3 X_4$

2. **Triangle K₃:**
   a) Stabilizers:
      - $K_1 = X_1 Z_2 Z_3$
      - $K_2 = Z_1 X_2 Z_3$
      - $K_3 = Z_1 Z_2 X_3$

   b) Parity check:
      $$H = \begin{pmatrix}
      1 & 0 & 0 & | & 0 & 1 & 1 \\
      0 & 1 & 0 & | & 1 & 0 & 1 \\
      0 & 0 & 1 & | & 1 & 1 & 0
      \end{pmatrix}$$

---

## Computational Lab

```python
"""
Day 737: Stabilizer Structure of Graph States
==============================================
Computing stabilizers and syndromes from graphs.
"""

import numpy as np
from typing import List, Tuple

def adjacency_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Create adjacency matrix from edges."""
    Gamma = np.zeros((n, n), dtype=int)
    for i, j in edges:
        Gamma[i, j] = 1
        Gamma[j, i] = 1
    return Gamma

def neighborhood(Gamma: np.ndarray, a: int) -> List[int]:
    """Return neighbors of vertex a."""
    return [b for b in range(len(Gamma)) if Gamma[a, b] == 1]

def graph_state_stabilizer(Gamma: np.ndarray, a: int) -> str:
    """
    Return stabilizer K_a as Pauli string.

    K_a = X_a × Z^{N(a)}
    """
    n = len(Gamma)
    pauli = ['I'] * n
    pauli[a] = 'X'
    for b in neighborhood(Gamma, a):
        pauli[b] = 'Z' if pauli[b] == 'I' else 'Y'  # Y if already X
    return ''.join(pauli)

def graph_state_stabilizers(Gamma: np.ndarray) -> List[str]:
    """Return all stabilizer generators for graph state."""
    n = len(Gamma)
    return [graph_state_stabilizer(Gamma, a) for a in range(n)]

def stabilizer_to_binary(pauli_str: str) -> np.ndarray:
    """Convert Pauli string to binary vector."""
    n = len(pauli_str)
    a = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)
    for i, p in enumerate(pauli_str):
        if p == 'X':
            a[i] = 1
        elif p == 'Z':
            b[i] = 1
        elif p == 'Y':
            a[i] = 1
            b[i] = 1
    return np.concatenate([a, b])

def graph_parity_check(Gamma: np.ndarray) -> np.ndarray:
    """
    Construct parity check matrix H = (I | Γ).
    """
    n = len(Gamma)
    return np.hstack([np.eye(n, dtype=int), Gamma])

def symplectic_inner_product(v1: np.ndarray, v2: np.ndarray) -> int:
    """Compute symplectic inner product."""
    n = len(v1) // 2
    return (np.dot(v1[:n], v2[n:]) + np.dot(v1[n:], v2[:n])) % 2

def compute_syndrome(H: np.ndarray, error: np.ndarray) -> np.ndarray:
    """
    Compute syndrome for error.

    s_a = <K_a, E>_s
    """
    n_k = H.shape[0]
    n = H.shape[1] // 2
    syndrome = np.zeros(n_k, dtype=int)

    for a in range(n_k):
        syndrome[a] = symplectic_inner_product(H[a], error)

    return syndrome

def verify_commutation(Gamma: np.ndarray) -> bool:
    """Verify all stabilizers commute."""
    stabilizers = graph_state_stabilizers(Gamma)
    n = len(stabilizers)

    for i in range(n):
        for j in range(i+1, n):
            v_i = stabilizer_to_binary(stabilizers[i])
            v_j = stabilizer_to_binary(stabilizers[j])
            if symplectic_inner_product(v_i, v_j) != 0:
                return False
    return True

def single_qubit_errors(n: int) -> List[Tuple[str, np.ndarray]]:
    """Generate all single-qubit X and Z errors."""
    errors = []
    for i in range(n):
        # X error
        e_X = np.zeros(2*n, dtype=int)
        e_X[i] = 1
        label_X = 'I'*i + 'X' + 'I'*(n-i-1)
        errors.append((label_X, e_X))

        # Z error
        e_Z = np.zeros(2*n, dtype=int)
        e_Z[n+i] = 1
        label_Z = 'I'*i + 'Z' + 'I'*(n-i-1)
        errors.append((label_Z, e_Z))

    return errors

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 737: Stabilizer Structure of Graph States")
    print("=" * 60)

    # Example 1: Linear graph
    print("\n1. Three-Qubit Linear Graph (Path)")
    print("-" * 40)

    Gamma_linear = adjacency_matrix(3, [(0,1), (1,2)])
    print(f"Graph: 0—1—2")
    print(f"Adjacency matrix:\n{Gamma_linear}")

    stabilizers = graph_state_stabilizers(Gamma_linear)
    print(f"\nStabilizers:")
    for a, K in enumerate(stabilizers):
        print(f"  K_{a} = {K}")

    H = graph_parity_check(Gamma_linear)
    print(f"\nParity check H = (I | Γ):\n{H}")

    print(f"\nAll stabilizers commute: {verify_commutation(Gamma_linear)}")

    # Example 2: Square graph
    print("\n2. Four-Qubit Square Graph")
    print("-" * 40)

    Gamma_square = adjacency_matrix(4, [(0,1), (1,2), (2,3), (3,0)])
    print(f"Graph: 0-1-2-3-0 (square)")

    stabilizers_sq = graph_state_stabilizers(Gamma_square)
    print(f"\nStabilizers:")
    for a, K in enumerate(stabilizers_sq):
        N_a = neighborhood(Gamma_square, a)
        print(f"  K_{a} = {K}  (N({a}) = {N_a})")

    print(f"\nAll commute: {verify_commutation(Gamma_square)}")

    # Example 3: Complete graph K4
    print("\n3. Complete Graph K₄")
    print("-" * 40)

    edges_K4 = [(i,j) for i in range(4) for j in range(i+1, 4)]
    Gamma_K4 = adjacency_matrix(4, edges_K4)

    stabilizers_K4 = graph_state_stabilizers(Gamma_K4)
    print("Stabilizers:")
    for a, K in enumerate(stabilizers_K4):
        print(f"  K_{a} = {K}")

    # Example 4: Syndrome table
    print("\n4. Syndrome Table for Linear Graph")
    print("-" * 40)

    H_linear = graph_parity_check(Gamma_linear)
    errors = single_qubit_errors(3)

    print("Error | Syndrome")
    print("------|---------")
    for label, e in errors:
        s = compute_syndrome(H_linear, e)
        print(f"  {label}  |   {s}")

    # Example 5: Star graph
    print("\n5. Star Graph S₄ (center + 3 leaves)")
    print("-" * 40)

    Gamma_star = adjacency_matrix(4, [(0,1), (0,2), (0,3)])

    stabilizers_star = graph_state_stabilizers(Gamma_star)
    print("Stabilizers:")
    for a, K in enumerate(stabilizers_star):
        weight = sum(1 for c in K if c != 'I')
        print(f"  K_{a} = {K}  (weight {weight})")

    # Verify with explicit symplectic check
    print("\n6. Explicit Commutation Check (Square)")
    print("-" * 40)

    for i in range(4):
        for j in range(i+1, 4):
            v_i = stabilizer_to_binary(stabilizers_sq[i])
            v_j = stabilizer_to_binary(stabilizers_sq[j])
            sip = symplectic_inner_product(v_i, v_j)
            print(f"  <K_{i}, K_{j}>_s = {sip}")

    print("\n" + "=" * 60)
    print("End of Day 737 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Stabilizer generator | $K_a = X_a \prod_{b \in N(a)} Z_b$ |
| Binary form | $K_a \leftrightarrow (e_a \| \Gamma_a)$ |
| Parity check | $H = (I_n \| \Gamma)$ |
| Commutation | $\langle K_a, K_b \rangle_s = 2\Gamma_{ab} = 0$ |
| Syndrome | $s_a = (e_X)_a + \sum_{b \in N(a)} (e_Z)_b$ |

### Main Takeaways

1. **Graph state stabilizers** have the elegant form $K_a = X_a Z^{N(a)}$
2. **The adjacency matrix** directly gives the Z-pattern
3. **Symmetry of Γ** ensures all stabilizers commute
4. **The parity check matrix** is simply $(I | \Gamma)$
5. **Syndromes** detect errors based on graph structure

---

## Daily Checklist

- [ ] I can derive stabilizers from any graph
- [ ] I understand the formula $K_a = X_a Z^{N(a)}$
- [ ] I can construct the parity check matrix H
- [ ] I verified commutation using symplectic products
- [ ] I can compute syndromes for errors
- [ ] I understand the relationship to error detection

---

## Preview: Day 738

Tomorrow we study **Local Complementation**:
- Definition of local complementation on graphs
- How it transforms the graph state
- Generating LC-equivalent graphs
- Connection to single-qubit Clifford operations
