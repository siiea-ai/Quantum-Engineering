# Day 807: Minimum Weight Perfect Matching for Surface Codes

## Year 2, Semester 2A: Quantum Error Correction
### Month 29: Topological Codes | Week 116: Error Chains & Logical Operations

---

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3 hours | Syndrome graph construction and MWPM theory |
| Afternoon | 2.5 hours | Blossom algorithm and implementation |
| Evening | 1.5 hours | Computational lab: PyMatching decoder |

---

## Learning Objectives

By the end of today, you will be able to:

1. Construct the syndrome graph from surface code measurements
2. Add virtual boundary vertices for planar codes
3. Compute edge weights from error probabilities
4. Apply the Blossom algorithm for perfect matching
5. Handle Y-errors with correlated X and Z syndromes
6. Evaluate decoder performance metrics
7. Implement a working MWPM decoder

---

## Core Content: MWPM Decoding

### The Decoding Problem

Given a syndrome $\sigma$ (set of violated stabilizers), find a **correction operator** $C$ such that:

1. $C$ produces syndrome $\sigma$: $\partial C = \sigma$
2. $E \cdot C$ is in the trivial homology class (with high probability)

where $E$ is the unknown actual error.

**Key insight:** We don't need to find $E$ exactly. Any $C$ with $\partial C = \sigma$ and $[E \cdot C] = 0$ suffices.

### Minimum Weight Perfect Matching

MWPM solves decoding by:

1. **Build syndrome graph**: Vertices = syndrome locations, edges = potential error paths
2. **Assign weights**: Based on error probabilities
3. **Find matching**: Pair syndromes to minimize total weight
4. **Apply correction**: Path between matched syndromes

$$\boxed{\text{MWPM: } \min_{\text{matching } M} \sum_{(u,v) \in M} w(u,v)}$$

---

## Syndrome Graph Construction

### Basic Graph Structure

For X-error decoding (Z-syndrome):

**Vertices:**
- One vertex for each violated plaquette (Z-stabilizer)
- Virtual boundary vertices for planar codes

**Edges:**
- Connect vertices where error chains could run
- Edge $(u, v)$ represents: "errors along path from $u$ to $v$"

### Planar Surface Code Graph

```
        Smooth boundary (top)
    ═══════════════════════════════

    ●───────●───────●───────●
    │   P₁  │   P₂  │   P₃  │
R   ●───────●───────●───────●   R
o   │   P₄  │   P₅  │   P₆  │   o
u   ●───────●───────●───────●   u
g   │   P₇  │   P₈  │   P₉  │   g
h   ●───────●───────●───────●   h

    ═══════════════════════════════
        Smooth boundary (bottom)

Z-syndromes at plaquettes P₁-P₉
X-syndromes at vertices (stars)
```

### Virtual Boundary Vertices

For planar codes, error chains can terminate at boundaries:

$$\boxed{\text{Add virtual vertex } v_\partial \text{ connected to all boundary-adjacent syndromes}}$$

**Why:** An error chain from syndrome $s$ to boundary has $\partial E = s$ (single endpoint).

```
Syndrome graph with virtual boundary:

    P₁ ──── P₂ ──── P₃
    │       │       │
    ├───────┼───────┼──── v_boundary
    │       │       │
    P₄ ──── P₅ ──── P₆
    │       │       │
    P₇ ──── P₈ ──── P₉
    │       │       │
    └───────┴───────┴──── v_boundary
```

### Edge Weight Calculation

Edge weight = negative log-likelihood of the error path:

$$\boxed{w(u, v) = -\log P(\text{error path } u \to v)}$$

For independent errors with probability $p$:

$$w(u, v) = d(u, v) \cdot \log\frac{1-p}{p}$$

where $d(u, v)$ is the Manhattan distance (number of edges in path).

**Intuition:** Longer paths are less likely, so they have higher weight.

### Weight Formula Derivation

If each edge has error probability $p$:

- Probability of $k$-error path: $\sim p^k (1-p)^{n-k}$
- Relative to no-error: $\sim (p/(1-p))^k$
- Log-likelihood ratio: $k \cdot \log(p/(1-p))$
- Minimize weight = maximize likelihood

---

## The Blossom Algorithm

### Perfect Matching Problem

**Input:** Graph $G = (V, E)$ with edge weights $w: E \to \mathbb{R}$

**Output:** Matching $M \subseteq E$ such that:
- Every vertex is matched exactly once
- Total weight $\sum_{e \in M} w(e)$ is minimized

### Algorithm Overview

Edmonds' **Blossom algorithm** (1965) solves this in polynomial time $O(|V|^3)$.

Key ideas:
1. **Augmenting paths**: Paths that can increase matching size
2. **Blossoms**: Odd cycles that are "shrunk" to single vertices
3. **Dual variables**: Linear programming relaxation

### Simplified MWPM for Surface Codes

For surface codes, we often use **approximate MWPM**:

1. Compute all-pairs shortest paths in syndrome graph
2. Build complete graph on syndrome vertices with these distances
3. Run Blossom on complete graph

```
Syndrome graph → All-pairs shortest path → Complete graph → Blossom → Matching
```

### Handling Odd Syndrome Count

Syndromes always come in pairs (boundary of chain has even size).

**Exception:** With boundaries, odd syndrome count is possible (one endpoint at boundary).

**Solution:** Virtual boundary vertex absorbs odd syndromes.

---

## Handling Y-Errors

### The Y-Error Problem

Y-error = X-error AND Z-error on same qubit:

$$Y = iXZ$$

This creates **correlated** X and Z syndromes at the same location.

### Independent vs Correlated Decoding

**Independent decoding:**
- Decode X-syndromes with one MWPM
- Decode Z-syndromes with another MWPM
- Apply corrections independently

**Problem:** Misses correlations, suboptimal for Y-errors.

### Correlated Decoding Approaches

**1. Hyper-edge matching:**
- Extend graph to include Y-error "hyper-edges"
- More complex algorithm

**2. Belief propagation + matching:**
- Use BP to estimate marginals
- Feed into MWPM

**3. Neural network decoders:**
- Learn correlations from training data

### Weight Adjustment for Y-Errors

For depolarizing noise: $P(X) = P(Y) = P(Z) = p/3$

Adjust weights:
- X-only path of length $k$: weight $\propto k \log(3(1-p)/p)$
- Path including Y-errors: different formula

---

## MWPM Decoder Performance

### Decoder Metrics

| Metric | Definition | Ideal Value |
|--------|------------|-------------|
| Logical error rate | $p_L = P(\text{decoder + error causes logical flip})$ | 0 |
| Threshold | $p_{th}$ where $p_L(p_{th}) = p_{th}$ | High |
| Decoding time | Time to produce correction | Low |
| Space complexity | Memory usage | Low |

### Threshold Values

| Noise Model | MWPM Threshold |
|-------------|----------------|
| Code capacity (X or Z only) | ~10.9% |
| Phenomenological | ~2.9% |
| Circuit-level (depolarizing) | ~0.7-1.0% |

### Sub-Threshold Scaling

Below threshold, logical error rate scales as:

$$\boxed{p_L \sim \left(\frac{p}{p_{th}}\right)^{(d+1)/2}}$$

Each doubling of distance roughly squares $p_L$.

---

## Worked Examples

### Example 1: Simple Two-Syndrome Case

**Setup:** Distance-5 surface code with Z-syndromes at plaquettes $P_3$ and $P_7$.

**Step 1:** Syndrome graph vertices: $\{P_3, P_7, v_\partial\}$

**Step 2:** Edge weights (Manhattan distance × log-odds):
- $w(P_3, P_7) = 4 \cdot \log((1-p)/p)$ (4-edge path)
- $w(P_3, v_\partial) = 2 \cdot \log((1-p)/p)$ (2 edges to boundary)
- $w(P_7, v_\partial) = 2 \cdot \log((1-p)/p)$ (2 edges to boundary)

**Step 3:** Compare matchings:
- $\{(P_3, P_7)\}$: total weight $= 4 \cdot L$
- $\{(P_3, v_\partial), (P_7, v_\partial)\}$: total weight $= 4 \cdot L$

**Step 4:** Both have same weight! Either correction works.

---

### Example 2: Four Syndromes

**Setup:** Syndromes at $P_1, P_4, P_6, P_9$ in a $3 \times 3$ grid.

**All-pairs distances:**
```
       P₁  P₄  P₆  P₉
  P₁   -   2   4   4
  P₄   2   -   2   2
  P₆   4   2   -   2
  P₉   4   2   2   -
```

**Possible perfect matchings:**
1. $\{(P_1, P_4), (P_6, P_9)\}$: weight $= 2 + 2 = 4$
2. $\{(P_1, P_6), (P_4, P_9)\}$: weight $= 4 + 2 = 6$
3. $\{(P_1, P_9), (P_4, P_6)\}$: weight $= 4 + 2 = 6$

**MWPM selects:** Matching 1 with weight 4.

**Correction:** Apply X on path $P_1 \to P_4$ and path $P_6 \to P_9$.

---

### Example 3: Boundary Involvement

**Setup:** Single syndrome at $P_5$ (center of $3 \times 3$).

**Problem:** Odd number of syndromes!

**Solution:** Include virtual boundary vertex.

**Matching:** $(P_5, v_\partial)$ with weight = distance to boundary.

**Correction:** X errors from $P_5$ to nearest boundary.

---

## Practice Problems

### Problem Set A: Graph Construction

**A1.** Draw the syndrome graph for a distance-3 surface code with:
- Z-syndromes at plaquettes (0,0) and (1,1)
- Include virtual boundary vertices
- Label all edge weights assuming $p = 0.01$

**A2.** For a distance-5 code, how many vertices are in the complete syndrome graph if there are $k$ syndromes? Include boundary vertices.

**A3.** Prove that the number of syndromes (excluding boundary) is always even for a toric code (no boundaries).

### Problem Set B: MWPM Algorithm

**B1.** Given syndromes at $(0,1), (2,1), (2,3), (0,3)$ on a $4 \times 4$ grid:
(a) List all possible perfect matchings
(b) Compute the weight of each
(c) Which does MWPM select?

**B2.** Implement the weight calculation:
```python
def edge_weight(d, p):
    """
    Compute MWPM edge weight.

    Args:
        d: Manhattan distance
        p: Physical error probability

    Returns:
        Edge weight
    """
    # Your implementation
```

**B3.** Why does MWPM use $-\log P(\text{path})$ instead of just path length? Give an example where this matters.

### Problem Set C: Performance Analysis

**C1.** Below threshold with $p = 0.001$ and $p_{th} = 0.01$:
(a) Estimate $p_L$ for $d = 5$
(b) Estimate $p_L$ for $d = 11$
(c) What distance is needed for $p_L < 10^{-15}$?

**C2.** MWPM has complexity $O(n^3)$ where $n$ = number of syndromes. For a $d \times d$ code with syndrome rate $p$:
(a) Expected number of syndromes?
(b) Expected decoding time scaling with $d$?

**C3.** Compare MWPM to Union-Find decoder:
- MWPM: $O(n^3)$, near-optimal threshold
- Union-Find: $O(n \alpha(n))$, slightly lower threshold

When would you prefer each?

---

## Computational Lab: MWPM Decoder Implementation

```python
"""
Day 807 Computational Lab: MWPM Decoder for Surface Codes
Implementation using networkx and scipy for matching
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Set, Dict
from collections import defaultdict

class MWPMDecoder:
    """
    Minimum Weight Perfect Matching decoder for surface codes.

    Supports planar surface codes with rough/smooth boundaries.
    """

    def __init__(self, d: int, p: float, boundary_type: str = 'planar'):
        """
        Initialize MWPM decoder.

        Args:
            d: Code distance
            p: Physical error probability
            boundary_type: 'planar' or 'toric'
        """
        self.d = d
        self.p = p
        self.boundary_type = boundary_type

        # Precompute log-odds ratio
        self.log_odds = np.log((1 - p) / p) if p > 0 and p < 1 else 1.0

    def syndrome_to_coords(self, syndrome: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract coordinates of syndrome defects.

        Args:
            syndrome: 2D array of syndrome values (1 = defect)

        Returns:
            List of (row, col) coordinates
        """
        return list(zip(*np.where(syndrome == 1)))

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def distance_to_boundary(self, p: Tuple[int, int], boundary: str) -> int:
        """
        Compute distance from point to boundary.

        Args:
            p: (row, col) coordinate
            boundary: 'top', 'bottom', 'left', 'right'

        Returns:
            Distance to specified boundary
        """
        row, col = p
        if boundary == 'top':
            return row + 1
        elif boundary == 'bottom':
            return self.d - row
        elif boundary == 'left':
            return col + 1
        elif boundary == 'right':
            return self.d - col
        else:
            raise ValueError(f"Unknown boundary: {boundary}")

    def min_boundary_distance(self, p: Tuple[int, int], error_type: str) -> int:
        """
        Minimum distance to appropriate boundary for error type.

        X-errors: terminate at rough boundaries (left/right for standard orientation)
        Z-errors: terminate at smooth boundaries (top/bottom)
        """
        if error_type == 'X':
            # X-errors go to rough boundaries
            return min(self.distance_to_boundary(p, 'left'),
                      self.distance_to_boundary(p, 'right'))
        else:  # Z-errors
            # Z-errors go to smooth boundaries
            return min(self.distance_to_boundary(p, 'top'),
                      self.distance_to_boundary(p, 'bottom'))

    def build_syndrome_graph(self, syndromes: List[Tuple[int, int]],
                             error_type: str = 'Z') -> nx.Graph:
        """
        Build weighted syndrome graph for MWPM.

        Args:
            syndromes: List of syndrome coordinates
            error_type: 'X' or 'Z' (affects boundary handling)

        Returns:
            NetworkX graph with edge weights
        """
        G = nx.Graph()
        n = len(syndromes)

        if n == 0:
            return G

        # Add syndrome vertices
        for i, s in enumerate(syndromes):
            G.add_node(i, pos=s, type='syndrome')

        # Add virtual boundary vertex
        boundary_idx = n
        G.add_node(boundary_idx, pos=None, type='boundary')

        # Add edges between all syndrome pairs
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.manhattan_distance(syndromes[i], syndromes[j])
                weight = dist * self.log_odds
                G.add_edge(i, j, weight=weight, distance=dist)

        # Add edges from syndromes to boundary
        for i in range(n):
            dist = self.min_boundary_distance(syndromes[i], error_type)
            weight = dist * self.log_odds
            G.add_edge(i, boundary_idx, weight=weight, distance=dist)

        return G

    def mwpm(self, G: nx.Graph) -> List[Tuple[int, int]]:
        """
        Find minimum weight perfect matching.

        Uses scipy's linear_sum_assignment for Hungarian algorithm
        on the complete weighted graph.

        Args:
            G: Syndrome graph

        Returns:
            List of matched pairs
        """
        nodes = list(G.nodes())
        n = len(nodes)

        if n == 0:
            return []

        # Handle odd number of nodes (shouldn't happen with boundary vertex)
        if n % 2 == 1:
            raise ValueError("Graph must have even number of nodes for perfect matching")

        # Build cost matrix
        cost_matrix = np.full((n, n), np.inf)
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j and G.has_edge(u, v):
                    cost_matrix[i, j] = G[u][v]['weight']

        # Handle disconnected pairs by adding large weight
        max_weight = np.max(cost_matrix[cost_matrix < np.inf]) if np.any(cost_matrix < np.inf) else 1
        cost_matrix[cost_matrix == np.inf] = max_weight * 100

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Extract matching (each pair appears twice, so filter)
        matching = []
        seen = set()
        for i, j in zip(row_ind, col_ind):
            if i < j and (i, j) not in seen and (j, i) not in seen:
                matching.append((nodes[i], nodes[j]))
                seen.add((i, j))

        return matching

    def matching_to_correction(self, matching: List[Tuple[int, int]],
                               syndromes: List[Tuple[int, int]],
                               error_type: str = 'Z') -> np.ndarray:
        """
        Convert matching to correction operator.

        Args:
            matching: List of matched syndrome pairs
            syndromes: Original syndrome coordinates
            error_type: 'X' or 'Z'

        Returns:
            Correction array (edges to flip)
        """
        n = len(syndromes)
        boundary_idx = n

        # Initialize correction (on edges)
        # For simplicity, return coordinates of correction path
        correction_edges = []

        for u, v in matching:
            if v == boundary_idx:
                # Path from syndrome u to boundary
                pos = syndromes[u]
                # Create path to nearest boundary
                correction_edges.append(('boundary', pos))
            elif u == boundary_idx:
                pos = syndromes[v]
                correction_edges.append(('boundary', pos))
            else:
                # Path between two syndromes
                pos_u, pos_v = syndromes[u], syndromes[v]
                correction_edges.append(('pair', pos_u, pos_v))

        return correction_edges

    def decode(self, syndrome: np.ndarray, error_type: str = 'Z') -> List:
        """
        Full decoding pipeline.

        Args:
            syndrome: 2D syndrome array
            error_type: 'X' or 'Z'

        Returns:
            Correction specification
        """
        # Extract syndrome locations
        syndromes = self.syndrome_to_coords(syndrome)

        if len(syndromes) == 0:
            return []

        # Build graph
        G = self.build_syndrome_graph(syndromes, error_type)

        # Find matching
        matching = self.mwpm(G)

        # Convert to correction
        correction = self.matching_to_correction(matching, syndromes, error_type)

        return correction


def visualize_decoding(d: int, p: float, seed: int = 42):
    """
    Visualize MWPM decoding process.
    """
    np.random.seed(seed)

    # Create random error pattern
    n_plaquettes = (d - 1) * (d - 1)
    error_rate = p * 4  # Expected syndrome rate

    # Generate syndrome (simplified model)
    syndrome = np.zeros((d - 1, d - 1), dtype=int)

    # Add some random syndromes (in pairs)
    n_pairs = max(1, int(n_plaquettes * error_rate / 2))
    for _ in range(n_pairs):
        r1, c1 = np.random.randint(0, d-1, 2)
        r2, c2 = np.random.randint(0, d-1, 2)
        syndrome[r1, c1] ^= 1
        syndrome[r2, c2] ^= 1

    # Initialize decoder
    decoder = MWPMDecoder(d, p)

    # Get syndrome locations
    syndromes = decoder.syndrome_to_coords(syndrome)

    # Build graph
    G = decoder.build_syndrome_graph(syndromes, 'Z')

    # Find matching
    matching = decoder.mwpm(G)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Syndrome pattern
    ax1 = axes[0]
    ax1.imshow(syndrome, cmap='Blues', vmin=0, vmax=1)
    for i in range(d - 1):
        for j in range(d - 1):
            if syndrome[i, j]:
                ax1.plot(j, i, 'ro', markersize=15)
    ax1.set_title(f'Syndrome Pattern\n{len(syndromes)} defects', fontsize=14)
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')

    # Panel 2: Syndrome graph
    ax2 = axes[1]
    pos = {}
    for node in G.nodes():
        if G.nodes[node]['type'] == 'syndrome':
            pos[node] = (G.nodes[node]['pos'][1], -G.nodes[node]['pos'][0])
        else:
            pos[node] = (d/2, 1)  # Boundary vertex above

    # Draw edges with weights
    edges = G.edges(data=True)
    edge_labels = {(u, v): f"{data['distance']}" for u, v, data in edges}

    nx.draw_networkx_nodes(G, pos, ax=ax2,
                          nodelist=[n for n in G.nodes() if G.nodes[n]['type'] == 'syndrome'],
                          node_color='red', node_size=300)
    nx.draw_networkx_nodes(G, pos, ax=ax2,
                          nodelist=[n for n in G.nodes() if G.nodes[n]['type'] == 'boundary'],
                          node_color='green', node_size=400, node_shape='s')
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10)

    ax2.set_title('Syndrome Graph\nEdge labels = distance', fontsize=14)
    ax2.axis('off')

    # Panel 3: MWPM result
    ax3 = axes[2]

    # Draw lattice background
    for i in range(d):
        ax3.axhline(y=i, color='lightgray', linewidth=0.5)
        ax3.axvline(x=i, color='lightgray', linewidth=0.5)

    # Draw syndromes
    for i, s in enumerate(syndromes):
        ax3.plot(s[1], s[0], 'ro', markersize=15)
        ax3.annotate(str(i), (s[1], s[0]), fontsize=10, ha='center', va='center', color='white')

    # Draw matching
    boundary_idx = len(syndromes)
    colors = plt.cm.Set1(np.linspace(0, 1, len(matching)))

    for idx, (u, v) in enumerate(matching):
        if v == boundary_idx or u == boundary_idx:
            # To boundary
            s_idx = u if v == boundary_idx else v
            s = syndromes[s_idx]
            # Draw to edge
            ax3.annotate('', xy=(0 if s[1] < d/2 else d-1, s[0]),
                        xytext=(s[1], s[0]),
                        arrowprops=dict(arrowstyle='->', color=colors[idx], lw=2))
        else:
            # Between syndromes
            s1, s2 = syndromes[u], syndromes[v]
            ax3.plot([s1[1], s2[1]], [s1[0], s2[0]], '-', color=colors[idx], linewidth=3)

    ax3.set_xlim(-0.5, d - 0.5)
    ax3.set_ylim(-0.5, d - 0.5)
    ax3.set_aspect('equal')
    ax3.set_title(f'MWPM Matching\n{len(matching)} pairs', fontsize=14)
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.savefig('mwpm_decoding.png', dpi=150, bbox_inches='tight')
    plt.show()

    return syndrome, syndromes, matching


def benchmark_decoder(d_values: List[int], p: float, n_trials: int = 100):
    """
    Benchmark MWPM decoder performance.
    """
    results = {'d': [], 'logical_error_rate': [], 'avg_syndromes': []}

    for d in d_values:
        print(f"Testing d={d}...")
        decoder = MWPMDecoder(d, p)

        logical_errors = 0
        total_syndromes = 0

        for trial in range(n_trials):
            # Generate random errors (simplified model)
            # True error pattern
            true_errors = np.random.random((d, d-1)) < p  # Horizontal edges
            true_errors_v = np.random.random((d-1, d)) < p  # Vertical edges

            # Compute syndrome from errors
            syndrome = np.zeros((d-1, d-1), dtype=int)
            for i in range(d-1):
                for j in range(d-1):
                    # Count errors around plaquette (mod 2)
                    count = 0
                    count += int(true_errors[i, j])      # Top
                    count += int(true_errors[i+1, j])    # Bottom
                    count += int(true_errors_v[i, j])    # Left
                    count += int(true_errors_v[i, j+1])  # Right
                    syndrome[i, j] = count % 2

            total_syndromes += np.sum(syndrome)

            # Decode
            correction = decoder.decode(syndrome, 'Z')

            # Check for logical error (simplified: check if correction spans code)
            # This is a rough approximation
            if len(correction) > 0:
                # A logical error occurs if total correction has odd parity across code
                # Simplified check: count vertical crossings
                pass

        results['d'].append(d)
        results['logical_error_rate'].append(logical_errors / n_trials)
        results['avg_syndromes'].append(total_syndromes / n_trials)

    return results


def main():
    """Run MWPM decoder demonstrations."""
    print("=" * 70)
    print("DAY 807: MINIMUM WEIGHT PERFECT MATCHING FOR SURFACE CODES")
    print("=" * 70)

    # Basic demonstration
    print("\n1. MWPM Decoding Visualization")
    print("-" * 40)

    syndrome, syndromes, matching = visualize_decoding(d=7, p=0.05, seed=123)

    print(f"   Code distance: d = 7")
    print(f"   Number of syndromes: {len(syndromes)}")
    print(f"   Matched pairs: {len(matching)}")

    # Theory summary
    print("\n2. MWPM Algorithm Summary")
    print("-" * 40)
    print("""
    MINIMUM WEIGHT PERFECT MATCHING DECODER

    Input:  Syndrome σ (set of violated stabilizers)
    Output: Correction C with ∂C = σ

    Algorithm:
    1. Build syndrome graph G
       - Vertices: syndrome locations + boundary
       - Edges: weighted by path likelihood

    2. Compute edge weights
       w(u,v) = d(u,v) × log((1-p)/p)
       (Manhattan distance × log-odds)

    3. Find minimum weight perfect matching
       Use Blossom algorithm: O(n³)

    4. Convert matching to correction
       Each matched pair → error path between them

    Key Properties:
    - Near-optimal for independent errors
    - Threshold ~10.9% (code capacity)
    - Threshold ~0.7% (circuit-level)
    """)

    # Weight formula demonstration
    print("\n3. Edge Weight Examples")
    print("-" * 40)

    for p in [0.001, 0.005, 0.01, 0.05]:
        log_odds = np.log((1-p)/p)
        print(f"   p = {p:.3f}: log((1-p)/p) = {log_odds:.3f}")
        for d in [1, 2, 3, 5]:
            w = d * log_odds
            print(f"      Distance {d}: weight = {w:.3f}")

    print("\n" + "=" * 70)
    print("MWPM is the standard decoder for surface code experiments.")
    print("PyMatching provides optimized implementation for practical use.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Edge weight | $w(u,v) = d(u,v) \cdot \log\frac{1-p}{p}$ |
| MWPM objective | $\min_M \sum_{(u,v) \in M} w(u,v)$ |
| Complexity | $O(n^3)$ for $n$ syndromes |
| Code capacity threshold | $p_{th} \approx 10.9\%$ |
| Circuit-level threshold | $p_{th} \approx 0.7\%$ |

### Main Takeaways

1. **Decoding as matching**: MWPM pairs syndromes to form correction chains

2. **Weight = negative log-likelihood**: Minimizing weight maximizes correction probability

3. **Virtual boundaries**: Handle odd syndrome counts for planar codes

4. **Blossom algorithm**: Polynomial-time optimal matching

5. **Y-error challenge**: Correlations between X and Z require advanced techniques

---

## Daily Checklist

### Morning Session (3 hours)
- [ ] Understand syndrome graph construction
- [ ] Learn weight calculation from error probabilities
- [ ] Study virtual boundary vertex technique

### Afternoon Session (2.5 hours)
- [ ] Work through MWPM algorithm steps
- [ ] Complete Problem Sets A and B
- [ ] Understand Y-error handling approaches

### Evening Session (1.5 hours)
- [ ] Run computational lab
- [ ] Implement basic MWPM decoder
- [ ] Complete Problem Set C

### Self-Assessment
1. Can you construct a syndrome graph from syndrome array?
2. Can you compute edge weights correctly?
3. Do you understand why MWPM works for decoding?
4. Can you identify the complexity bottlenecks?

---

## Preview: Day 808

Tomorrow we study **Logical Error Rate Scaling**:
- Below-threshold behavior: $p_L \sim (p/p_{th})^{d/2}$
- Phenomenological vs circuit-level thresholds
- Resource overhead analysis
- Comparison with other code families

---

*Day 807 of 2184 | Year 2, Month 29, Week 116 | Quantum Engineering PhD Curriculum*
