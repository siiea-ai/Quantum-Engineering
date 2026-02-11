# Day 984: Panteleev-Kalachev & Quantum Tanner Codes

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Lifted Product Construction |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Quantum Tanner Codes |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Code Analysis |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 984, you will be able to:

1. Explain the lifted product construction using group algebras
2. Describe how Cayley graphs provide the expansion needed
3. Construct quantum Tanner codes from left-right Cayley complexes
4. Calculate code parameters from the underlying graph structure
5. Compare the Panteleev-Kalachev and Leverrier-Zémor approaches
6. Appreciate the algebraic machinery behind good qLDPC codes

---

## Core Content

### 1. The Key Insight: Lifting

The breakthrough insight of Panteleev and Kalachev (2021-2022) is to **lift** the hypergraph product using group actions.

**Standard Hypergraph Product:**
$$H_X = [H_1 \otimes I, \quad I \otimes H_2^T]$$

**Lifted Product:**
$$\tilde{H}_X = [H_1 \otimes_\pi A, \quad B \otimes_\pi H_2^T]$$

where $\otimes_\pi$ is a **twisted tensor product** over a group $G$, and $A, B$ are matrices derived from the group algebra $\mathbb{F}_2[G]$.

**Why This Works:**

The group action introduces correlations that:
1. Maintain sparsity (low weight)
2. Preserve the CSS structure
3. **Amplify expansion properties**

---

### 2. Group Algebras

**Definition:** For a finite group $G$ and field $\mathbb{F}$, the group algebra is:
$$\mathbb{F}[G] = \left\{ \sum_{g \in G} a_g \cdot g \mid a_g \in \mathbb{F} \right\}$$

**Operations:**
- Addition: $\sum a_g g + \sum b_g g = \sum (a_g + b_g) g$
- Multiplication: $(g)(h) = gh$ (group operation)

**Example:** $\mathbb{F}_2[\mathbb{Z}_3] = \{0, 1, g, g^2, 1+g, 1+g^2, g+g^2, 1+g+g^2\}$

**Matrices over Group Algebras:**

An $m \times n$ matrix over $\mathbb{F}_2[G]$ represents an $m|G| \times n|G|$ matrix over $\mathbb{F}_2$.

$$A = \begin{pmatrix} g & 1+g^2 \\ g^2 & g \end{pmatrix} \in \mathbb{F}_2[\mathbb{Z}_3]^{2 \times 2}$$

Expands to a $6 \times 6$ binary matrix!

---

### 3. Cayley Graphs and Expansion

**Definition:** For a group $G$ and generating set $S \subseteq G$ (with $S = S^{-1}$), the Cayley graph $\text{Cay}(G, S)$ has:
- Vertices: Elements of $G$
- Edges: $(g, gs)$ for each $g \in G$ and $s \in S$

**Expansion Property:**

A family of Cayley graphs $\{\text{Cay}(G_n, S_n)\}$ is an **expander family** if the spectral gap remains bounded away from zero:
$$\lambda_1(G_n) - \lambda_2(G_n) \geq \epsilon > 0$$

**Examples of Expander Cayley Graphs:**

1. **Ramanujan graphs:** $\text{Cay}(\text{SL}_2(\mathbb{F}_p), S)$ with optimal expansion
2. **Random regular graphs:** Expanders with high probability
3. **Zig-zag products:** Combinatorial constructions

**Why Cayley Graphs?**

The group structure ensures:
- Regular degree
- Symmetry (vertex transitive)
- Algebraic analysis of spectrum

---

### 4. The Lifted Product Construction

**Panteleev-Kalachev Construction (Simplified):**

**Input:**
- Base codes: $C_1, C_2$ with parity-check matrices $H_1, H_2$ over $\mathbb{F}_2$
- Group $G$ with generating set $S$
- Lift: Replace each 1 in $H_1, H_2$ with a group element

**Lifted Parity-Check Matrices:**

Let $\tilde{H}_1 \in \mathbb{F}_2[G]^{m_1 \times n_1}$ and $\tilde{H}_2 \in \mathbb{F}_2[G]^{m_2 \times n_2}$ be lifted versions where entries are group elements or 0.

**Balanced Product:**

$$\boxed{\tilde{H}_X = [\tilde{H}_1 \otimes I_{n_2}, \quad I_{m_1} \otimes \tilde{H}_2^T]}$$
$$\boxed{\tilde{H}_Z = [I_{n_1} \otimes \tilde{H}_2, \quad \tilde{H}_1^T \otimes I_{m_2}]}$$

where tensor products are over $\mathbb{F}_2[G]$.

**Key Property:**

When viewed as matrices over $\mathbb{F}_2$ (expanding the group elements), the resulting code has:
- $n = |G| \cdot (n_1 n_2 + m_1 m_2)$ qubits
- $k = |G| \cdot k_1 k_2$ logical qubits
- $d = \Omega(d_1 d_2)$ distance (with expansion!)

---

### 5. Distance Improvement via Expansion

**The Magic:**

In standard hypergraph product: $d = \min(d_1, d_2)$

In lifted product with expanding group: $d = \Omega(d_1 \cdot d_2)$ (product, not minimum!)

**Why?**

Consider a logical operator (non-trivial cycle in the code's chain complex).

In the lifted product:
1. A cycle in the base complex lifts to $|G|$ cycles
2. Expansion ensures these cycles are "spread out"
3. Low-weight errors cannot simultaneously satisfy all lifted constraints
4. Result: minimum weight increases multiplicatively

**Formal Statement (Panteleev-Kalachev 2022):**

If $G$ is an $(\epsilon, \delta)$-expander and the base codes have distance $d_1, d_2$:
$$d \geq \frac{\epsilon \delta}{4} \cdot d_1 \cdot d_2$$

---

### 6. Quantum Tanner Codes (Leverrier-Zémor 2022)

An alternative construction achieving good codes via **left-right Cayley complexes**.

**Cayley Complex:**

For a group $G$ with left generators $S_L$ and right generators $S_R$:

$$C(G, S_L, S_R)$$

has:
- 0-cells (vertices): Elements of $G$
- 1-cells (edges): Two types - left edges $(g, s_L g)$ and right edges $(g, g s_R)$
- 2-cells (faces): Squares from $g \to s_L g \to s_L g s_R \to g s_R \to g$

**Tanner Code Construction:**

Assign a small classical code $C_0$ to each vertex, and constrain:
- Left edges by one set of parity checks
- Right edges by another set

**Parameters:**

$$[[n = |G| \cdot |S_L| \cdot |S_R|, \quad k = \Theta(n), \quad d = \Theta(n)]]$$

with explicit constants depending on $C_0$ and expansion of the Cayley graph.

**Advantage:**

The Leverrier-Zémor construction gives:
- Simpler analysis
- Explicit constants
- Clear geometric interpretation

---

### 7. Explicit Example: Small Lifted Product

**Setup:**
- Base code: $[4, 2, 2]$ code with $H = \begin{pmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix}$
- Group: $G = \mathbb{Z}_3 = \{0, 1, 2\}$ (additive)
- Lift: Replace non-zero entries with group elements

**Lifted Matrix:**
$$\tilde{H} = \begin{pmatrix} 1 & g & 0 & 0 \\ 0 & 0 & g^2 & 1 \end{pmatrix}$$

where $1 = e$ (identity), $g$ is the generator.

**Expansion to Binary:**

Each entry becomes a $3 \times 3$ circulant matrix:
- $1 \to I_3$
- $g \to $ cyclic permutation matrix
- $g^2 \to $ double cyclic permutation

**Resulting Code:**
- Size: $3 \times 4 = 12$ bits (or qubits after product)
- Increased redundancy from group structure

---

### 8. Code Parameters Summary

**Panteleev-Kalachev (2022):**

$$\boxed{[[n, k, d]] = [[n, \Theta(n), \Theta(\sqrt{n} \log n)]]} \text{ (first paper)}$$
$$\boxed{[[n, k, d]] = [[n, \Theta(n), \Theta(n)]]} \text{ (final result)}$$

**Leverrier-Zémor (2022):**

$$\boxed{[[n, k, d]] = [[n, cn, \delta n]]]}$$

with explicit $c \approx 0.01$ and $\delta \approx 0.01$ for some constructions.

**Dinur et al. (2022):**

Added **linear-time decoding**:
$$[[n, \Theta(n), \Theta(n)]] \text{ with } O(n) \text{ decoding}$$

**Comparison:**

| Construction | Rate | Rel. Distance | Decoding | Locality |
|--------------|------|---------------|----------|----------|
| Panteleev-Kalachev | ~0.01 | ~0.01 | $O(n^2)$ | Non-local |
| Leverrier-Zémor | ~0.01 | ~0.01 | $O(n^2)$ | Non-local |
| Dinur et al. | ~0.01 | ~0.01 | $O(n)$ | Non-local |

---

## Practical Applications

### Path to Implementation

**Near-Term Candidates:**

Rather than general good qLDPC, specific families are being explored:

**1. Bivariate Bicycle Codes (IBM, 2023):**
- Special case of lifted product
- Group: $\mathbb{Z}_l \times \mathbb{Z}_m$ (torus)
- Moderate non-locality
- $[[144, 12, 12]]$ demonstrated

**2. Hyperbolic Surface Codes:**
- Codes on hyperbolic surfaces
- Better rate than Euclidean surface codes
- Still $d = O(\sqrt{n})$ but improved constant

**3. Fiber Bundle Codes:**
- 3D local structure
- $d = O(\sqrt{n} \log n)$
- May be implementable with stacked chips

**Hardware Requirements:**

For full good qLDPC implementation:
- Long-range qubit connectivity
- Low-latency classical control
- Parallelizable syndrome extraction

---

## Worked Examples

### Example 1: Group Algebra Calculation

**Problem:** In $\mathbb{F}_2[\mathbb{Z}_4]$, compute $(1 + g)(g + g^3)$.

**Solution:**

Let $G = \mathbb{Z}_4 = \{e, g, g^2, g^3\}$ where $g^4 = e$.

$$(1 + g)(g + g^3) = 1 \cdot g + 1 \cdot g^3 + g \cdot g + g \cdot g^3$$
$$= g + g^3 + g^2 + g^4$$
$$= g + g^3 + g^2 + e$$
$$= 1 + g + g^2 + g^3$$

In $\mathbb{F}_2$: This is the "all ones" element.

---

### Example 2: Cayley Graph Analysis

**Problem:** Analyze the Cayley graph $\text{Cay}(\mathbb{Z}_8, \{1, 7\})$.

**Solution:**

**Vertices:** $\{0, 1, 2, 3, 4, 5, 6, 7\}$

**Edges:**
- From $i$: connect to $i+1 \mod 8$ and $i+7 \mod 8 = i-1 \mod 8$

This is the **cycle graph** $C_8$ (each vertex connected to neighbors).

**Degree:** 2 (regular)

**Spectrum:** Eigenvalues of adjacency matrix are $2\cos(2\pi k/8)$ for $k = 0, 1, \ldots, 7$.
- $\lambda_0 = 2$
- $\lambda_1 = \sqrt{2} \approx 1.41$
- $\lambda_4 = -2$

**Spectral gap:** $\lambda_0 - |\lambda_1| = 2 - 1.41 = 0.59$

This is a weak expander. For good codes, we need larger groups with better expansion.

---

### Example 3: Parameter Calculation

**Problem:** A lifted product uses $\mathbb{Z}_{101}$ (prime order group) with base codes $[100, 50, 10]$. Estimate the quantum code parameters.

**Solution:**

**Base code:** $[n_1, k_1, d_1] = [100, 50, 10]$, so $m_1 = 50$.

**Group size:** $|G| = 101$

**Quantum code size:**
$$n = |G| \cdot (n_1 \cdot n_1 + m_1 \cdot m_1) = 101 \cdot (10000 + 2500) = 101 \cdot 12500 = 1,262,500$$

**Logical qubits:**
$$k = |G| \cdot k_1 \cdot k_1 = 101 \cdot 2500 = 252,500$$

**Rate:** $R = k/n = 252500/1262500 = 0.2$

**Distance (with expansion):**

If $\mathbb{Z}_{101}$ with appropriate generators gives expansion $\epsilon$:
$$d \geq \epsilon \cdot d_1 \cdot d_1 / c = \epsilon \cdot 100 / c$$

For Ramanujan-type expansion, $d = \Theta(\sqrt{n}) \approx 1100$ or better.

With optimal construction: $d = \Theta(n^{1/2})$ to $\Theta(n)$.

---

## Practice Problems

### Level 1: Direct Application

1. **Group Algebra:** In $\mathbb{F}_2[\mathbb{Z}_5]$, simplify $(1 + g^2)(1 + g^3)$.

2. **Cayley Graph:** Draw the Cayley graph $\text{Cay}(\mathbb{Z}_6, \{1, 5\})$. What is its structure?

3. **Size Calculation:** A lifted product uses $|G| = 50$ with base codes having $n_1 = n_2 = 20$, $m_1 = m_2 = 10$. How many physical qubits?

### Level 2: Intermediate

4. **Expansion and Distance:** Explain qualitatively why expansion helps distance in lifted products. What happens to low-weight errors?

5. **CSS Constraint:** Show that the lifted product matrices $\tilde{H}_X$ and $\tilde{H}_Z$ satisfy $\tilde{H}_X \tilde{H}_Z^T = 0$ if the base matrices do.

6. **Ramanujan Bound:** For a $d$-regular Ramanujan graph, the second eigenvalue satisfies $|\lambda_2| \leq 2\sqrt{d-1}$. What is the spectral gap for $d = 10$?

### Level 3: Challenging

7. **Left-Right Cayley Complex:** For $G = S_3$ (symmetric group on 3 elements) with $S_L = \{(12)\}$ and $S_R = \{(23)\}$, describe the resulting Cayley complex. How many 2-cells are there?

8. **Rate-Distance Trade-off:** The quantum GV bound gives $R \leq 1 - 2H_2(\delta)$. For the Panteleev-Kalachev construction achieving $R = 0.01$ and $\delta = 0.01$, how far is this from the GV bound?

9. **Decoding Complexity:** Why does standard belief propagation fail for qLDPC codes, and how does the Dinur et al. linear-time decoder overcome this?

---

## Computational Lab

### Objective
Explore group algebras, Cayley graphs, and lifted product structure.

```python
"""
Day 984 Computational Lab: Panteleev-Kalachev & Quantum Tanner Codes
QLDPC Codes & Constant-Overhead QEC - Week 141
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product

# =============================================================================
# Part 1: Group Algebra Operations
# =============================================================================

print("=" * 70)
print("Part 1: Group Algebra over F_2[Z_n]")
print("=" * 70)

class CyclicGroupAlgebra:
    """F_2[Z_n] - Group algebra of cyclic group over F_2."""

    def __init__(self, n):
        self.n = n

    def element(self, coeffs):
        """Create element from coefficient list [a_0, a_1, ..., a_{n-1}]."""
        return np.array(coeffs, dtype=int) % 2

    def add(self, a, b):
        """Add two elements."""
        return (a + b) % 2

    def multiply(self, a, b):
        """Multiply two elements (convolution with wrap-around)."""
        result = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                if a[i] and b[j]:
                    result[(i + j) % self.n] += 1
        return result % 2

    def to_matrix(self, a):
        """Convert element to n x n circulant matrix."""
        M = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                M[i, j] = a[(j - i) % self.n]
        return M

    def from_generators(self, positions):
        """Create element that is sum of g^i for i in positions."""
        a = np.zeros(self.n, dtype=int)
        for p in positions:
            a[p % self.n] = 1
        return a

# Example: F_2[Z_5]
G = CyclicGroupAlgebra(5)

print(f"\nGroup algebra: F_2[Z_5]")
print(f"Group elements: 1, g, g^2, g^3, g^4 where g^5 = 1")

# Create some elements
a = G.element([1, 1, 0, 0, 0])  # 1 + g
b = G.element([0, 0, 1, 1, 0])  # g^2 + g^3

print(f"\na = 1 + g = {a}")
print(f"b = g^2 + g^3 = {b}")

# Multiply
ab = G.multiply(a, b)
print(f"\na * b = (1+g)(g^2+g^3) = {ab}")
print(f"      = g^2 + g^3 + g^3 + g^4 = g^2 + g^4 (in F_2)")

# Matrix representation
print(f"\nMatrix representation of a = 1 + g:")
print(G.to_matrix(a))

# =============================================================================
# Part 2: Cayley Graph Construction and Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Cayley Graphs")
print("=" * 70)

def cayley_graph(n, generators):
    """Construct Cayley graph of Z_n with given generators."""
    G = nx.DiGraph()

    # Add nodes
    for i in range(n):
        G.add_node(i)

    # Add edges for each generator
    for i in range(n):
        for s in generators:
            j = (i + s) % n
            G.add_edge(i, j)

    return G

def analyze_cayley(G, name=""):
    """Analyze Cayley graph properties."""
    n = G.number_of_nodes()

    # Convert to undirected for spectral analysis
    G_undir = G.to_undirected()

    # Adjacency matrix
    A = nx.adjacency_matrix(G_undir).todense()

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(A)
    eigenvalues = np.sort(eigenvalues)[::-1]

    spectral_gap = eigenvalues[0] - abs(eigenvalues[1])

    print(f"\n{name}:")
    print(f"  Nodes: {n}")
    print(f"  Degree: {eigenvalues[0]:.0f}")
    print(f"  Top eigenvalues: {eigenvalues[:4]}")
    print(f"  Spectral gap: {spectral_gap:.4f}")

    return eigenvalues, spectral_gap

# Example 1: Z_12 with generators {1, 11} (cycle)
G1 = cayley_graph(12, [1, 11])
eigs1, gap1 = analyze_cayley(G1, "Cay(Z_12, {1, 11}) - Cycle")

# Example 2: Z_12 with generators {1, 5, 7, 11} (better expander)
G2 = cayley_graph(12, [1, 5, 7, 11])
eigs2, gap2 = analyze_cayley(G2, "Cay(Z_12, {1, 5, 7, 11})")

# Example 3: Z_31 with quadratic residues (pseudo-Paley)
# Quadratic residues mod 31
qr_31 = [i**2 % 31 for i in range(1, 31)]
qr_31 = list(set(qr_31))  # Remove duplicates
G3 = cayley_graph(31, qr_31)
eigs3, gap3 = analyze_cayley(G3, "Cay(Z_31, QR) - Paley-like")

# Visualize Cayley graphs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, G, title in [(axes[0], G1, 'Cycle Z_12'),
                      (axes[1], G2, 'Better Expansion Z_12'),
                      (axes[2], G3, 'Paley-like Z_31')]:
    pos = nx.circular_layout(G)
    nx.draw(G, pos, ax=ax, node_size=200, node_color='lightblue',
            with_labels=True, font_size=8, arrows=True, arrowsize=10)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('day_984_cayley_graphs.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nCayley graphs saved to 'day_984_cayley_graphs.png'")

# =============================================================================
# Part 3: Lifted Product Structure
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Lifted Product Code Construction")
print("=" * 70)

def lift_matrix(H_base, group_size, lift_positions):
    """
    Lift a binary matrix over a cyclic group.

    Args:
        H_base: m x n binary matrix
        group_size: size of cyclic group
        lift_positions: dict mapping (i,j) to list of group positions

    Returns:
        Lifted matrix as m*group_size x n*group_size binary matrix
    """
    m, n = H_base.shape
    g = group_size

    H_lifted = np.zeros((m * g, n * g), dtype=int)
    G = CyclicGroupAlgebra(g)

    for i in range(m):
        for j in range(n):
            if H_base[i, j] == 1:
                # Get lift element
                if (i, j) in lift_positions:
                    positions = lift_positions[(i, j)]
                else:
                    positions = [0]  # Default: identity

                elem = G.from_generators(positions)
                block = G.to_matrix(elem)

                # Place block
                H_lifted[i*g:(i+1)*g, j*g:(j+1)*g] = block

    return H_lifted

# Base code: [6, 3, 2] code
H_base = np.array([
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
], dtype=int)

print("Base parity-check matrix (3 x 6):")
print(H_base)

# Lift over Z_4
group_size = 4
lift_positions = {
    (0, 0): [0],      # 1
    (0, 1): [1],      # g
    (0, 3): [2],      # g^2
    (1, 1): [0],      # 1
    (1, 2): [1],      # g
    (1, 4): [3],      # g^3
    (2, 0): [2],      # g^2
    (2, 2): [0],      # 1
    (2, 5): [1],      # g
}

H_lifted = lift_matrix(H_base, group_size, lift_positions)

print(f"\nLifted matrix over Z_4 ({H_lifted.shape[0]} x {H_lifted.shape[1]}):")
print(f"Shape: {H_lifted.shape}")
print(f"Sparsity: {np.sum(H_lifted) / H_lifted.size:.4f}")

# Visualize sparsity pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.imshow(H_base, cmap='Blues', aspect='auto')
ax1.set_title(f'Base Matrix H ({H_base.shape[0]}x{H_base.shape[1]})')
ax1.set_xlabel('Columns')
ax1.set_ylabel('Rows')

ax2 = axes[1]
ax2.imshow(H_lifted, cmap='Blues', aspect='auto')
ax2.set_title(f'Lifted Matrix ({H_lifted.shape[0]}x{H_lifted.shape[1]})')
ax2.set_xlabel('Columns')
ax2.set_ylabel('Rows')

plt.tight_layout()
plt.savefig('day_984_lifted_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nLifted matrix saved to 'day_984_lifted_matrix.png'")

# =============================================================================
# Part 4: Hypergraph Product with Lifting
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Lifted Hypergraph Product")
print("=" * 70)

def lifted_hypergraph_product(H1, H2, group_size):
    """
    Simplified lifted hypergraph product.

    For demonstration, we use random lifts.
    """
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    g = group_size

    # Create random lifts (simplified)
    np.random.seed(42)

    # Lift H1
    lift1 = {(i, j): [np.random.randint(g)]
             for i in range(m1) for j in range(n1) if H1[i, j] == 1}
    H1_lift = lift_matrix(H1, g, lift1)

    # Lift H2
    lift2 = {(i, j): [np.random.randint(g)]
             for i in range(m2) for j in range(n2) if H2[i, j] == 1}
    H2_lift = lift_matrix(H2, g, lift2)

    # Hypergraph product of lifted matrices
    # Hx = [H1_lift ⊗ I, I ⊗ H2_lift^T]
    n1_lift, m1_lift = H1_lift.shape[1], H1_lift.shape[0]
    n2_lift, m2_lift = H2_lift.shape[1], H2_lift.shape[0]

    I_n2 = np.eye(n2_lift, dtype=int)
    I_m1 = np.eye(m1_lift, dtype=int)

    Hx_left = np.kron(H1_lift, I_n2)
    Hx_right = np.kron(I_m1, H2_lift.T)
    Hx = np.hstack([Hx_left, Hx_right]) % 2

    I_n1 = np.eye(n1_lift, dtype=int)
    I_m2 = np.eye(m2_lift, dtype=int)

    Hz_left = np.kron(I_n1, H2_lift)
    Hz_right = np.kron(H1_lift.T, I_m2)
    Hz = np.hstack([Hz_left, Hz_right]) % 2

    # Code parameters
    n = Hx.shape[1]
    rank_x = np.linalg.matrix_rank(Hx)
    rank_z = np.linalg.matrix_rank(Hz)
    k = n - rank_x - rank_z

    return Hx, Hz, {
        'n': n,
        'k': k,
        'group_size': g,
        'Hx_shape': Hx.shape,
        'Hz_shape': Hz.shape
    }

# Small example
H1_small = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=int)
H2_small = H1_small.copy()

for g in [2, 3, 5]:
    Hx, Hz, params = lifted_hypergraph_product(H1_small, H2_small, g)

    # Verify CSS
    css_check = (Hx @ Hz.T) % 2
    is_css = np.sum(css_check) == 0

    print(f"\nGroup Z_{g}:")
    print(f"  Physical qubits: {params['n']}")
    print(f"  Logical qubits: {params['k']}")
    print(f"  Rate: {params['k']/params['n']:.4f}")
    print(f"  CSS valid: {is_css}")

# =============================================================================
# Part 5: Spectral Gap and Code Quality
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Spectral Gap vs Code Properties")
print("=" * 70)

# Study relationship between spectral gap and code quality
def construct_code_family(base_size, group_sizes):
    """Construct family of lifted codes and analyze."""
    results = []

    # Fixed base code structure
    H_base = np.zeros((base_size//2, base_size), dtype=int)
    for i in range(base_size//2):
        H_base[i, 2*i] = 1
        H_base[i, 2*i + 1] = 1
        H_base[i, (2*i + 2) % base_size] = 1

    for g in group_sizes:
        # Create Cayley graph with standard generators
        generators = [1, g-1] if g > 2 else [1]
        G_cayley = cayley_graph(g, generators)
        G_undir = G_cayley.to_undirected()

        # Spectral gap
        try:
            A = nx.adjacency_matrix(G_undir).todense()
            eigs = np.sort(np.linalg.eigvalsh(A))[::-1]
            gap = eigs[0] - abs(eigs[1])
        except:
            gap = 0

        # Build lifted code (simplified)
        n_lifted = base_size * g
        k_est = (base_size // 2) * g  # Rough estimate

        results.append({
            'group_size': g,
            'spectral_gap': gap,
            'n': n_lifted,
            'k_est': k_est,
            'rate_est': k_est / n_lifted
        })

    return results

base_size = 8
group_sizes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

results = construct_code_family(base_size, group_sizes)

print("\nSpectral Gap Analysis:")
print(f"{'|G|':>5} {'Gap':>8} {'n':>8} {'k_est':>8} {'Rate':>8}")
print("-" * 40)
for r in results:
    print(f"{r['group_size']:>5} {r['spectral_gap']:>8.3f} {r['n']:>8} {r['k_est']:>8} {r['rate_est']:>8.3f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

gaps = [r['spectral_gap'] for r in results]
sizes = [r['group_size'] for r in results]

ax.scatter(sizes, gaps, s=100, c='blue', edgecolors='black')
ax.plot(sizes, gaps, 'b--', alpha=0.5)

ax.set_xlabel('Group Size |G|')
ax.set_ylabel('Spectral Gap')
ax.set_title('Spectral Gap of Cayley Graphs Cay(Z_n, {1, n-1})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_984_spectral_gap.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSpectral gap analysis saved to 'day_984_spectral_gap.png'")

# =============================================================================
# Part 6: Summary of Constructions
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Summary of Good qLDPC Constructions")
print("=" * 70)

summary_table = """
COMPARISON OF GOOD qLDPC CONSTRUCTIONS:

┌─────────────────────┬──────────────┬──────────────┬──────────────┬─────────────┐
│ Construction        │ Rate         │ Rel. Dist.   │ Decoding     │ Locality    │
├─────────────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│ Hypergraph Product  │ Θ(1)         │ O(1/√n) → 0  │ O(n²)        │ Non-local   │
│ Panteleev-Kalachev  │ Θ(1) ~0.01   │ Θ(1) ~0.01   │ O(n²)        │ Non-local   │
│ Leverrier-Zémor     │ Θ(1) ~0.01   │ Θ(1) ~0.01   │ O(n²)        │ Non-local   │
│ Dinur et al.        │ Θ(1) ~0.01   │ Θ(1) ~0.01   │ O(n)         │ Non-local   │
│ Bivariate Bicycle   │ ~0.08        │ ~0.08        │ O(n)         │ Moderate    │
│ Fiber Bundle        │ Θ(1)         │ O(√n log n)  │ O(n)         │ 3D local    │
└─────────────────────┴──────────────┴──────────────┴──────────────┴─────────────┘

KEY INSIGHTS:

1. LIFTING is the key technique:
   - Standard product: d = min(d₁, d₂)
   - Lifted product: d = Ω(d₁ × d₂) with expansion!

2. GROUP ALGEBRA provides structure:
   - Cyclic groups (Z_n): Simple, regular
   - More complex groups: Better expansion possible
   - Cayley graphs: Natural expander families

3. EXPANSION is crucial:
   - Spectral gap determines distance amplification
   - Ramanujan graphs achieve optimal expansion
   - Trade-off with explicit construction complexity

4. PRACTICAL CODES sacrifice some optimality:
   - Bivariate bicycle: Better locality, lower rate/distance
   - Fiber bundle: 3D local, sublinear distance
   - Balance theory vs. hardware constraints
"""

print(summary_table)

print("\n" + "=" * 70)
print("Day 984 Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Group algebra | $\mathbb{F}_2[G] = \{\sum a_g g : a_g \in \mathbb{F}_2\}$ |
| Lifted product $H_X$ | $[\tilde{H}_1 \otimes I, I \otimes \tilde{H}_2^T]$ |
| Spectral gap | $\lambda_1 - |\lambda_2| > 0$ |
| Distance improvement | $d = \Omega(d_1 \cdot d_2)$ (with expansion) |
| Ramanujan bound | $|\lambda_2| \leq 2\sqrt{d-1}$ |

### Main Takeaways

1. **Lifting** over group algebras transforms hypergraph product into good codes
2. **Cayley graphs** provide the expansion properties needed for distance
3. **Spectral gap** quantifies expansion and determines code quality
4. **Panteleev-Kalachev** and **Leverrier-Zémor** achieved the breakthrough
5. **Dinur et al.** added linear-time decoding capability
6. **Practical codes** (bivariate bicycle) trade some optimality for implementability

---

## Daily Checklist

- [ ] Understand group algebra operations
- [ ] Draw and analyze simple Cayley graphs
- [ ] Explain the lifting mechanism
- [ ] Calculate spectral gap for small examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt Level 2 problems
- [ ] Run computational lab
- [ ] Compare different good qLDPC constructions

---

## Preview: Day 985

Tomorrow we explore **Constant-Overhead Fault Tolerance**:
- What constant overhead means for scalability
- Gate implementation on qLDPC codes
- Magic state distillation with qLDPC
- The threshold theorem revisited
- Practical implications for large-scale quantum computing

---

*"The lifted product construction shows that the correct algebraic framework can overcome seemingly fundamental barriers."*
--- Perspective on the Panteleev-Kalachev breakthrough

---

**Next:** Day 985 - Constant-Overhead Fault Tolerance
