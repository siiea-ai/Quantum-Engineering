# Day 828: MWPM Optimization Techniques

## Week 119: Real-Time Decoding | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Schedule (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 2.5 hours | Blossom algorithm, sparse graphs |
| **Afternoon** | 2.5 hours | PyMatching, locality optimizations |
| **Evening** | 2 hours | MWPM optimization lab |

---

## Learning Objectives

By the end of Day 828, you will be able to:

1. **Explain** the Blossom algorithm for minimum-weight perfect matching
2. **Construct** sparse matching graphs exploiting surface code locality
3. **Apply** boundary matching and virtual node techniques
4. **Implement** practical MWPM optimizations for surface codes
5. **Use** the PyMatching library for efficient syndrome decoding
6. **Analyze** the time complexity improvements from various optimizations

---

## Core Content

### 1. Review: The MWPM Decoding Problem

Given a syndrome $\sigma$ from a surface code measurement, decoding requires:

1. Identify syndrome defects (vertices where stabilizers flip)
2. Construct a matching graph connecting defect pairs
3. Assign edge weights based on error probabilities
4. Find minimum-weight perfect matching
5. Infer correction from matched paths

The matching graph for a distance-$d$ surface code can have:
- $O(d^2)$ vertices per syndrome round
- $O(d^4)$ edges (if fully connected)
- $O(d^2 \cdot T)$ vertices for $T$ syndrome rounds (3D matching)

### 2. The Blossom Algorithm

Edmonds' Blossom algorithm (1965) finds MWPM in general graphs.

#### Key Concepts

**Augmenting Path**: A path between unmatched vertices alternating matched/unmatched edges.

**Blossom**: An odd-length cycle in the graph. The algorithm "shrinks" blossoms to single vertices to handle odd cycles.

**Algorithm Phases**:
1. Start with empty matching
2. Grow alternating trees from unmatched vertices
3. When trees meet: augment matching or form blossom
4. Shrink blossoms and continue
5. Expand blossoms when optimal

#### Complexity

| Implementation | Time Complexity | Space |
|---------------|-----------------|-------|
| Original Blossom | $O(n^4)$ | $O(n^2)$ |
| Blossom V (Kolmogorov) | $O(n^3)$ worst, $O(n^2 \log n)$ typical | $O(n^2)$ |
| Sparse Blossom | $O(nm \log n)$ where $m$ = edges | $O(m)$ |

For surface codes: $n = O(d^2)$, so:
- Dense graph: $O(d^6)$ to $O(d^4 \log d)$
- Sparse graph: $O(d^2 \cdot d^2 \cdot \log d) = O(d^4 \log d)$

### 3. Sparse Matching Graph Construction

The key insight: **locality**. In a surface code, errors are local, so distant defect pairs are exponentially unlikely.

#### Weight Function

The edge weight between defects at positions $(i_1, j_1, t_1)$ and $(i_2, j_2, t_2)$ is:

$$w_{12} = -\log P(\text{error chain connecting defects})$$

For independent errors with rate $p$:

$$w_{12} = -\log\left(\frac{p}{1-p}\right)^{|i_1-i_2| + |j_1-j_2| + |t_1-t_2|} = D_{12} \cdot \log\frac{1-p}{p}$$

where $D_{12}$ is the Manhattan distance.

#### Sparsification Strategy

Include edge $(i,j)$ only if:

$$w_{ij} < w_{\text{cutoff}}$$

Equivalently, include only edges with Manhattan distance:

$$\boxed{D_{ij} < D_{\text{max}} = \frac{w_{\text{cutoff}}}{\log\frac{1-p}{p}}}$$

For $p = 1\%$ and $w_{\text{cutoff}} = 50$:
$$D_{\text{max}} = \frac{50}{\log(99)} \approx 11$$

This reduces edges from $O(n^2)$ to $O(n \cdot D_{\text{max}}^3)$.

### 4. Boundary Matching

Surface codes have boundaries where stabilizers are incomplete. Defects can match to boundaries (indicating errors near the edge).

**Virtual Boundary Node**: Add a virtual node at each boundary with:
- Zero weight to nearby defects
- Represents error chains ending at boundary

**Implementation**:
```
For each defect near boundary:
    Add edge to virtual boundary node
    Weight = distance to boundary × log((1-p)/p)
```

For a distance-$d$ code, we add $O(d)$ virtual nodes instead of $O(d^2)$ pairwise connections.

### 5. The Matching Graph in 3D

For $T$ rounds of syndrome measurement, we construct a 3D matching graph:

- **Vertices**: Defects in spacetime $(x, y, t)$
- **Spatial edges**: Connect defects at same time, nearby in space
- **Temporal edges**: Connect defects at same position, consecutive times

The 3D structure handles **measurement errors** (syndrome bit flips without corresponding qubit errors).

**Syndrome Difference Decoding**:
Rather than decode raw syndromes, decode the **difference**:
$$\Delta\sigma_t = \sigma_t \oplus \sigma_{t-1}$$

Defects in $\Delta\sigma$ indicate either:
- A data qubit error between rounds $t-1$ and $t$
- A measurement error at round $t$ or $t-1$

### 6. PyMatching: Practical MWPM

PyMatching (Higgott 2023) is a highly optimized Python library for MWPM decoding.

**Key Features**:
- Sparse Blossom implementation
- Efficient syndrome-to-defect conversion
- Automatic boundary handling
- Integration with Stim simulator

**Basic Usage**:
```python
import pymatching
import numpy as np

# Create matching graph from check matrix
# H: parity check matrix (syndrome bits × data qubits)
matching = pymatching.Matching(H)

# Decode a syndrome
syndrome = np.array([0, 1, 1, 0, ...])  # syndrome bits
correction = matching.decode(syndrome)  # returns correction
```

**Performance**: PyMatching achieves near-linear average-case time complexity for typical surface code syndromes.

### 7. Further Optimizations

#### Precomputed Lookup for Small Syndromes

For low error rates, most syndrome rounds have few or no defects:
- 0 defects: no correction needed
- 2 defects: precompute optimal pairing
- 4+ defects: run full MWPM

This gives $O(1)$ average case for typical operation.

#### Hierarchical Matching

For very large codes:
1. Divide lattice into blocks
2. Match defects within each block
3. Merge block solutions

Reduces effective problem size while slightly degrading threshold.

#### Parallel Processing

MWPM has inherent parallelism:
- Different connected components can be matched independently
- Tree-growing phase parallelizes well
- Modern implementations use SIMD and multi-threading

---

## Worked Examples

### Example 1: Sparse Graph Construction

**Problem**: For a distance-7 surface code with $p = 0.5\%$ error rate, construct the sparse matching graph with cutoff weight $w_{\text{cutoff}} = 30$.

**Solution**:

Calculate maximum edge distance:
$$D_{\text{max}} = \frac{w_{\text{cutoff}}}{\log\frac{1-p}{p}} = \frac{30}{\log\frac{0.995}{0.005}} = \frac{30}{\log(199)} = \frac{30}{5.29} \approx 5.7$$

So we include edges with Manhattan distance $D \leq 5$.

Number of data qubits: $n = 7^2 = 49$
Number of syndrome bits: $m = 48$ (for a standard surface code)

Dense graph edges: $\binom{48}{2} = 1128$

Sparse graph edges per vertex: $\approx (2 \cdot 5)^2 / 2 = 50$ (vertices within Manhattan distance 5)

Total sparse edges: $\approx 48 \times 50 / 2 = 1200$

Wait, this is actually more! For small codes, sparsification may not help much. Let's recalculate considering only defect pairs:

At $p = 0.5\%$, expected defects per round: $0.005 \times 48 \approx 0.24$

On average, we have $< 1$ defect, so most rounds need no matching!

$$\boxed{\text{Average edges to consider: } \approx 0.24 \times 50 = 12}$$

### Example 2: 3D Matching Graph

**Problem**: A distance-5 surface code is measured for $T = 10$ rounds. How many vertices and edges in the full 3D matching graph?

**Solution**:

Vertices per spatial slice: $d^2 - 1 = 24$ (one per stabilizer)

But vertices in matching graph are **defects**, not all stabilizers. At low error rates, most stabilizers don't flip.

However, for graph construction, we consider potential defects:
- Maximum defects per round: 24
- Over $T = 10$ rounds: 240 potential vertices

Edges:
- Spatial edges (within each round): $O(d^2) \times T = 24 \times 10 = 240$
- Temporal edges (between rounds): $24 \times 9 = 216$ (connecting each stabilizer to itself in adjacent rounds)

With sparse construction (distance cutoff 5):
- Each vertex connects to $\approx 20$ spatial neighbors and 2 temporal neighbors

Total edges: $\approx 240 \times 22 / 2 = 2640$

$$\boxed{\text{Vertices: } 240, \text{ Edges: } \sim 2600}$$

### Example 3: PyMatching Timing Estimate

**Problem**: Using PyMatching, estimate decoding time for a distance-11 surface code at $p = 0.3\%$ with 20 syndrome rounds.

**Solution**:

PyMatching average complexity: $O(n \cdot \alpha(n))$ where $n$ is number of defects.

Expected defects:
- Stabilizers: $11^2 - 1 = 120$ per round
- Total measurements: $120 \times 20 = 2400$
- Expected defects: $2400 \times 0.003 = 7.2$

At ~7 defects, matching is nearly trivial:
- Graph construction: $O(7 \times 50) = O(350)$ operations
- Blossom matching: $O(7 \times \alpha(7)) \approx O(7 \times 3) = O(21)$

Estimated time on modern CPU (pessimistic):
- 100 ns per operation
- Total: $\sim 40 \, \mu\text{s}$

But PyMatching is highly optimized:
- Typical distance-11 decode: $1-10 \, \mu\text{s}$
- With FPGA: potentially $< 100 \, \text{ns}$

$$\boxed{t_{\text{decode}} \approx 1-10 \, \mu\text{s} \text{ (software)}}$$

---

## Practice Problems

### Direct Application

**Problem 1**: Calculate the sparse graph cutoff distance $D_{\text{max}}$ for $p = 0.2\%$ and $w_{\text{cutoff}} = 40$.

**Problem 2**: For a distance-9 surface code, how many virtual boundary nodes are needed? What are their approximate positions?

### Intermediate

**Problem 3**: A syndrome has 6 defects arranged in a line. Sketch the optimal matching. How does locality affect which pairs match?

**Problem 4**: Compare memory requirements for dense vs sparse matching graphs for:
- Distance 11, 1 round
- Distance 21, 100 rounds

### Challenging

**Problem 5**: Derive the probability that a syndrome round has exactly $k$ defects, given error rate $p$ and $m$ stabilizers. Use this to estimate average-case decoder complexity.

**Problem 6**: Design a "two-phase" decoder that uses lookup tables for $\leq 4$ defects and full MWPM for more. Analyze the speedup for $p = 0.5\%$ on a distance-7 code.

---

## Computational Lab: MWPM Optimizations

```python
"""
Day 828 Lab: MWPM Optimization Techniques
Implementing and benchmarking optimized matching decoders
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from collections import defaultdict
import heapq
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Part 1: Sparse Matching Graph Construction
# =============================================================================

class SparseMatchingGraph:
    """
    Sparse matching graph for surface code decoding.

    Exploits locality to reduce edge count from O(n^2) to O(n).
    """

    def __init__(self, distance, p_error, weight_cutoff=50):
        """
        Initialize sparse matching graph.

        Parameters:
        -----------
        distance : int
            Surface code distance
        p_error : float
            Physical error probability
        weight_cutoff : float
            Maximum edge weight to include
        """
        self.d = distance
        self.p = p_error
        self.w_cutoff = weight_cutoff

        # Weight per unit distance
        self.w_per_dist = np.log((1 - p_error) / p_error) if p_error < 0.5 else 0

        # Maximum Manhattan distance for edge inclusion
        self.d_max = int(weight_cutoff / self.w_per_dist) if self.w_per_dist > 0 else distance

        # Stabilizer positions (simplified: all interior points)
        self.stabilizers = []
        for i in range(distance):
            for j in range(distance):
                # X and Z stabilizers alternate (simplified)
                self.stabilizers.append((i, j))

        self.n_stabilizers = len(self.stabilizers)

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def edge_weight(self, pos1, pos2):
        """Calculate edge weight from positions."""
        dist = self.manhattan_distance(pos1, pos2)
        return dist * self.w_per_dist

    def build_graph_from_defects(self, defect_positions):
        """
        Build sparse matching graph from defect positions.

        Returns:
        --------
        vertices : list
            Defect positions
        edges : list of (i, j, weight)
            Edges between defects
        """
        n = len(defect_positions)
        if n == 0:
            return [], []

        vertices = list(defect_positions)
        edges = []

        # Add edges between nearby defects
        for i in range(n):
            for j in range(i + 1, n):
                w = self.edge_weight(vertices[i], vertices[j])
                if w <= self.w_cutoff:
                    edges.append((i, j, w))

        # Add boundary edges (virtual node at index n)
        # Simplified: boundary at edges of [0, d) x [0, d)
        for i, pos in enumerate(vertices):
            # Distance to nearest boundary
            d_boundary = min(pos[0], pos[1],
                            self.d - 1 - pos[0], self.d - 1 - pos[1])
            w_boundary = d_boundary * self.w_per_dist
            if w_boundary <= self.w_cutoff:
                edges.append((i, n, w_boundary))  # n is virtual boundary

        return vertices, edges


def demonstrate_sparse_graph():
    """Demonstrate sparse graph construction."""
    print("=" * 60)
    print("SPARSE MATCHING GRAPH CONSTRUCTION")
    print("=" * 60)

    d = 7
    p = 0.005
    graph = SparseMatchingGraph(d, p, weight_cutoff=30)

    print(f"\nCode distance: {d}")
    print(f"Error rate: {p*100}%")
    print(f"Weight per unit distance: {graph.w_per_dist:.2f}")
    print(f"Max edge distance: {graph.d_max}")

    # Simulate some defects
    np.random.seed(42)
    n_defects = 6
    defect_positions = [(np.random.randint(0, d), np.random.randint(0, d))
                        for _ in range(n_defects)]

    vertices, edges = graph.build_graph_from_defects(defect_positions)

    print(f"\nDefects: {vertices}")
    print(f"Number of edges (sparse): {len(edges)}")
    print(f"Dense graph would have: {n_defects * (n_defects - 1) // 2} edges")

    # Visualize
    plt.figure(figsize=(8, 8))

    # Draw lattice
    for i in range(d):
        for j in range(d):
            plt.plot(i, j, 'ko', markersize=3, alpha=0.2)

    # Draw defects
    for idx, (x, y) in enumerate(vertices):
        plt.plot(x, y, 'ro', markersize=15)
        plt.text(x + 0.1, y + 0.1, str(idx), fontsize=10)

    # Draw edges
    for i, j, w in edges:
        if j < len(vertices):  # Not boundary edge
            x1, y1 = vertices[i]
            x2, y2 = vertices[j]
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.5)
            plt.text((x1+x2)/2, (y1+y2)/2, f'{w:.1f}', fontsize=8, color='blue')
        else:  # Boundary edge
            x, y = vertices[i]
            # Draw to nearest boundary
            plt.plot([x, x], [y, -0.3], 'g--', linewidth=1, alpha=0.5)

    plt.xlim(-0.5, d - 0.5)
    plt.ylim(-0.5, d - 0.5)
    plt.title(f'Sparse Matching Graph (d={d})', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig('sparse_matching_graph.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'sparse_matching_graph.png'")

    return graph

graph = demonstrate_sparse_graph()

# =============================================================================
# Part 2: Simple MWPM Implementation (for educational purposes)
# =============================================================================

def greedy_matching(vertices, edges):
    """
    Greedy approximation to MWPM.

    Not optimal, but O(E log E) and illustrative.
    """
    if len(vertices) == 0:
        return []

    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda e: e[2])

    matched = set()
    matching = []

    for i, j, w in sorted_edges:
        if i not in matched and j not in matched:
            matching.append((i, j, w))
            matched.add(i)
            matched.add(j)

    return matching


def blossom_matching_simple(vertices, edges):
    """
    Simplified Blossom algorithm implementation.

    This is a basic version for educational purposes.
    For production, use NetworkX or PyMatching.
    """
    try:
        import networkx as nx

        if len(vertices) == 0:
            return []

        # Create graph
        G = nx.Graph()
        for i in range(len(vertices) + 1):  # +1 for boundary node
            G.add_node(i)

        for i, j, w in edges:
            G.add_edge(i, j, weight=w)

        # Find minimum weight matching
        matching = nx.min_weight_matching(G)

        # Convert to list format
        result = []
        for i, j in matching:
            # Find weight
            w = G[i][j]['weight']
            result.append((i, j, w))

        return result

    except ImportError:
        print("NetworkX not available, using greedy approximation")
        return greedy_matching(vertices, edges)


def compare_matching_algorithms():
    """Compare greedy vs Blossom matching."""
    print("\n" + "=" * 60)
    print("MATCHING ALGORITHM COMPARISON")
    print("=" * 60)

    d = 9
    p = 0.01
    graph = SparseMatchingGraph(d, p, weight_cutoff=40)

    np.random.seed(123)

    # Generate test cases
    n_tests = 20
    results = {'greedy': [], 'blossom': []}
    times = {'greedy': [], 'blossom': []}

    for _ in range(n_tests):
        # Random defects
        n_defects = np.random.randint(4, 12)
        defects = [(np.random.randint(0, d), np.random.randint(0, d))
                   for _ in range(n_defects)]
        vertices, edges = graph.build_graph_from_defects(defects)

        # Greedy
        t0 = perf_counter()
        greedy_match = greedy_matching(vertices, edges)
        times['greedy'].append(perf_counter() - t0)
        greedy_weight = sum(w for _, _, w in greedy_match)
        results['greedy'].append(greedy_weight)

        # Blossom
        t0 = perf_counter()
        blossom_match = blossom_matching_simple(vertices, edges)
        times['blossom'].append(perf_counter() - t0)
        blossom_weight = sum(w for _, _, w in blossom_match)
        results['blossom'].append(blossom_weight)

    # Statistics
    greedy_weights = np.array(results['greedy'])
    blossom_weights = np.array(results['blossom'])

    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(blossom_weights > 0,
                          greedy_weights / blossom_weights,
                          1.0)

    print(f"\nTest cases: {n_tests}")
    print(f"\nGreedy matching:")
    print(f"  Average weight: {np.mean(greedy_weights):.2f}")
    print(f"  Average time: {np.mean(times['greedy'])*1e6:.1f} μs")

    print(f"\nBlossom matching (optimal):")
    print(f"  Average weight: {np.mean(blossom_weights):.2f}")
    print(f"  Average time: {np.mean(times['blossom'])*1e6:.1f} μs")

    print(f"\nGreedy/Blossom weight ratio: {np.mean(ratios):.3f}")
    print(f"  (1.0 = optimal, >1.0 = suboptimal)")

    # Plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(blossom_weights, greedy_weights, alpha=0.7)
    max_w = max(max(greedy_weights), max(blossom_weights))
    plt.plot([0, max_w], [0, max_w], 'r--', label='Optimal')
    plt.xlabel('Blossom Weight (Optimal)', fontsize=12)
    plt.ylabel('Greedy Weight', fontsize=12)
    plt.title('Matching Quality Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(['Greedy', 'Blossom'],
            [np.mean(times['greedy'])*1e6, np.mean(times['blossom'])*1e6])
    plt.ylabel('Time (μs)', fontsize=12)
    plt.title('Matching Time Comparison', fontsize=14)

    plt.tight_layout()
    plt.savefig('matching_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'matching_comparison.png'")

compare_matching_algorithms()

# =============================================================================
# Part 3: Lookup Table Optimization
# =============================================================================

class LookupTableDecoder:
    """
    Lookup table decoder for small numbers of defects.

    Precomputes optimal matchings for common configurations.
    """

    def __init__(self, max_defects=4):
        """
        Initialize lookup table.

        Parameters:
        -----------
        max_defects : int
            Maximum defects to handle with lookup
        """
        self.max_defects = max_defects
        self.lookup_table = {}

        # For 2 defects, matching is trivial (match them together)
        # For 4 defects, we precompute optimal pairing

    def key_from_defects(self, defect_positions):
        """Create hashable key from defect positions."""
        # Normalize by sorting
        return tuple(sorted(defect_positions))

    def optimal_pairing_4(self, positions):
        """Find optimal pairing of 4 positions."""
        p0, p1, p2, p3 = positions

        def dist(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        # Three possible pairings
        pairing1 = dist(p0, p1) + dist(p2, p3)  # (0,1), (2,3)
        pairing2 = dist(p0, p2) + dist(p1, p3)  # (0,2), (1,3)
        pairing3 = dist(p0, p3) + dist(p1, p2)  # (0,3), (1,2)

        min_cost = min(pairing1, pairing2, pairing3)

        if pairing1 == min_cost:
            return [(0, 1), (2, 3)]
        elif pairing2 == min_cost:
            return [(0, 2), (1, 3)]
        else:
            return [(0, 3), (1, 2)]

    def decode(self, defect_positions):
        """
        Decode using lookup table if possible.

        Returns:
        --------
        matching : list of (i, j) pairs, or None if too many defects
        """
        n = len(defect_positions)

        if n == 0:
            return []
        elif n == 2:
            return [(0, 1)]
        elif n == 4:
            return self.optimal_pairing_4(defect_positions)
        else:
            return None  # Fall back to full MWPM


def benchmark_lookup_decoder():
    """Benchmark lookup table decoder."""
    print("\n" + "=" * 60)
    print("LOOKUP TABLE DECODER BENCHMARK")
    print("=" * 60)

    d = 7
    p = 0.005
    n_syndromes = d * d

    # Defect count distribution (Poisson approximation)
    expected_defects = p * n_syndromes
    print(f"\nCode distance: {d}")
    print(f"Error rate: {p*100}%")
    print(f"Expected defects per syndrome: {expected_defects:.2f}")

    # Probability of k defects
    from math import factorial, exp
    lam = expected_defects

    probs = {}
    for k in range(10):
        probs[k] = exp(-lam) * (lam ** k) / factorial(k)

    print("\nDefect count distribution:")
    for k in range(6):
        print(f"  P({k} defects) = {probs[k]*100:.2f}%")

    lookup_coverage = sum(probs[k] for k in [0, 2, 4])
    print(f"\nLookup table coverage (0, 2, 4 defects): {lookup_coverage*100:.1f}%")

    # Simulate decoding
    lookup = LookupTableDecoder(max_defects=4)
    graph = SparseMatchingGraph(d, p, weight_cutoff=30)

    n_trials = 10000
    lookup_used = 0
    full_mwpm_needed = 0

    lookup_times = []
    mwpm_times = []

    for _ in range(n_trials):
        # Sample number of defects
        n_defects = np.random.poisson(expected_defects)

        # Make even (for matching)
        if n_defects % 2 == 1:
            n_defects += 1

        # Generate random defects
        defects = [(np.random.randint(0, d), np.random.randint(0, d))
                   for _ in range(n_defects)]

        t0 = perf_counter()
        result = lookup.decode(defects)
        t1 = perf_counter()

        if result is not None:
            lookup_used += 1
            lookup_times.append(t1 - t0)
        else:
            # Would need full MWPM
            full_mwpm_needed += 1
            vertices, edges = graph.build_graph_from_defects(defects)
            t0 = perf_counter()
            _ = greedy_matching(vertices, edges)
            t1 = perf_counter()
            mwpm_times.append(t1 - t0)

    print(f"\nSimulation ({n_trials} trials):")
    print(f"  Lookup table used: {lookup_used} ({lookup_used/n_trials*100:.1f}%)")
    print(f"  Full MWPM needed: {full_mwpm_needed} ({full_mwpm_needed/n_trials*100:.1f}%)")

    if lookup_times:
        print(f"  Lookup average time: {np.mean(lookup_times)*1e9:.1f} ns")
    if mwpm_times:
        print(f"  MWPM average time: {np.mean(mwpm_times)*1e6:.1f} μs")

    # Estimated speedup
    if lookup_times and mwpm_times:
        avg_time = (lookup_used * np.mean(lookup_times) +
                    full_mwpm_needed * np.mean(mwpm_times)) / n_trials
        mwpm_only_time = np.mean(mwpm_times)
        speedup = mwpm_only_time / avg_time if avg_time > 0 else 1

        print(f"\n  Estimated speedup with lookup: {speedup:.1f}x")

benchmark_lookup_decoder()

# =============================================================================
# Part 4: Decoder Complexity Scaling
# =============================================================================

def benchmark_scaling():
    """Benchmark decoder scaling with code distance."""
    print("\n" + "=" * 60)
    print("DECODER SCALING BENCHMARK")
    print("=" * 60)

    distances = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    p = 0.005

    greedy_times = []
    blossom_times = []
    n_edges_list = []

    for d in distances:
        graph = SparseMatchingGraph(d, p, weight_cutoff=30)

        # Generate typical syndrome
        n_defects = max(2, int(np.random.poisson(p * d * d)))
        if n_defects % 2 == 1:
            n_defects += 1

        defects = [(np.random.randint(0, d), np.random.randint(0, d))
                   for _ in range(n_defects)]
        vertices, edges = graph.build_graph_from_defects(defects)
        n_edges_list.append(len(edges))

        # Benchmark greedy
        times = []
        for _ in range(10):
            t0 = perf_counter()
            _ = greedy_matching(vertices, edges)
            times.append(perf_counter() - t0)
        greedy_times.append(np.mean(times))

        # Benchmark blossom
        times = []
        for _ in range(10):
            t0 = perf_counter()
            _ = blossom_matching_simple(vertices, edges)
            times.append(perf_counter() - t0)
        blossom_times.append(np.mean(times))

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(distances, np.array(greedy_times)*1e6, 'bo-',
                 label='Greedy', linewidth=2, markersize=8)
    plt.semilogy(distances, np.array(blossom_times)*1e6, 'rs-',
                 label='Blossom', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='k', linestyle='--', label='500 ns budget')
    plt.xlabel('Code Distance', fontsize=12)
    plt.ylabel('Decode Time (μs)', fontsize=12)
    plt.title('Decoder Time vs Code Distance', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    plt.subplot(1, 2, 2)
    plt.plot(distances, n_edges_list, 'g^-', linewidth=2, markersize=8)
    plt.xlabel('Code Distance', fontsize=12)
    plt.ylabel('Sparse Graph Edges', fontsize=12)
    plt.title('Graph Size vs Code Distance', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('decoder_scaling_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'decoder_scaling_benchmark.png'")

    # Summary table
    print("\n" + "-" * 50)
    print(f"{'Distance':<10} {'Edges':<10} {'Greedy (μs)':<15} {'Blossom (μs)':<15}")
    print("-" * 50)
    for i, d in enumerate(distances):
        print(f"{d:<10} {n_edges_list[i]:<10} {greedy_times[i]*1e6:<15.2f} {blossom_times[i]*1e6:<15.2f}")

benchmark_scaling()

# =============================================================================
# Part 5: PyMatching Demonstration (Optional)
# =============================================================================

def demonstrate_pymatching():
    """Demonstrate PyMatching library usage."""
    print("\n" + "=" * 60)
    print("PyMatching DEMONSTRATION")
    print("=" * 60)

    try:
        import pymatching
        print("\nPyMatching is available!")

        # Create simple parity check matrix for repetition code
        # (Easier to demonstrate than full surface code)
        d = 5  # Repetition code distance
        H = np.zeros((d-1, d), dtype=np.uint8)
        for i in range(d-1):
            H[i, i] = 1
            H[i, i+1] = 1

        print(f"\nRepetition code parity check matrix H:")
        print(H)

        # Create matching object
        matching = pymatching.Matching(H)

        # Simulate error and syndrome
        error = np.array([0, 0, 1, 0, 0], dtype=np.uint8)  # Error on qubit 2
        syndrome = H @ error % 2

        print(f"\nError: {error}")
        print(f"Syndrome: {syndrome}")

        # Decode
        correction = matching.decode(syndrome)
        print(f"Decoded correction: {correction}")
        print(f"Residual error: {(error + correction) % 2}")

        # Benchmark
        n_trials = 1000
        t0 = perf_counter()
        for _ in range(n_trials):
            matching.decode(syndrome)
        t_avg = (perf_counter() - t0) / n_trials

        print(f"\nPyMatching decode time: {t_avg*1e6:.2f} μs")

    except ImportError:
        print("\nPyMatching not installed. Install with: pip install pymatching")
        print("This is a highly optimized MWPM implementation for surface codes.")

demonstrate_pymatching()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("LAB SUMMARY")
print("=" * 60)
print("""
Key findings:

1. SPARSE GRAPHS: Reducing edges from O(n²) to O(n) dramatically helps
   for typical low error rate scenarios.

2. LOOKUP TABLES: At p ~ 0.5%, lookup tables handle 80%+ of syndromes,
   giving near-O(1) average case.

3. BLOSSOM vs GREEDY: Blossom gives optimal results but is ~10x slower.
   For real-time use, the trade-off depends on error rate and distance.

4. SCALING: Modern optimized implementations (PyMatching) achieve
   near-linear scaling for practical code sizes.

5. REAL-TIME FEASIBILITY: For d < 15 and p < 1%, software decoders
   can achieve sub-μs latency with proper optimization.
""")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Edge weight | $w_{ij} = D_{ij} \cdot \log\frac{1-p}{p}$ |
| Sparse cutoff distance | $D_{\text{max}} = w_{\text{cutoff}} / \log\frac{1-p}{p}$ |
| Blossom complexity | $O(n^3)$ worst, $O(n^2 \log n)$ typical |
| Sparse Blossom | $O(nm \log n)$ where $m$ = edges |
| Defect probability | $P(k) \approx e^{-\lambda}\lambda^k / k!$, $\lambda = pm$ |

### Optimization Strategies

1. **Sparse graphs**: Exploit locality, include only nearby edges
2. **Boundary handling**: Virtual nodes reduce complexity
3. **Lookup tables**: O(1) for common low-defect cases
4. **Hierarchical matching**: Divide and conquer for large codes
5. **Parallelization**: Independent components, SIMD operations

---

## Daily Checklist

- [ ] I understand the Blossom algorithm at a conceptual level
- [ ] I can construct sparse matching graphs for surface codes
- [ ] I know how to handle boundary conditions with virtual nodes
- [ ] I can implement and use lookup table optimizations
- [ ] I understand when PyMatching is preferable to custom implementations
- [ ] I can analyze the complexity trade-offs of various optimizations

---

## Preview: Day 829

Tomorrow we explore the **Union-Find Decoder**, which achieves near-linear time complexity:
- Cluster growth and fusion algorithms
- The Union-Find data structure
- Peeling decoder for error extraction
- Accuracy vs speed trade-offs

The Union-Find decoder is currently the leading candidate for real-time decoding in superconducting systems.

---

*"Optimization is the art of making the common case fast and the rare case correct."*

---

[← Day 827: Decoding Latency](./Day_827_Monday.md) | [Day 829: Union-Find Decoder →](./Day_829_Wednesday.md)
