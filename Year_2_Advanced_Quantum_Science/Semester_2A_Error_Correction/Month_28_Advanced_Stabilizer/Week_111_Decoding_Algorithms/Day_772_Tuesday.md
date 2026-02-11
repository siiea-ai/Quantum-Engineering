# Day 772: Minimum Weight Perfect Matching (MWPM)

## Week 111: Decoding Algorithms | Month 28: Advanced Stabilizer Codes

---

## Daily Schedule

| Session | Time | Duration | Focus |
|---------|------|----------|-------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Syndrome Graphs & MWPM Theory |
| Afternoon | 1:00 PM - 4:00 PM | 3 hours | Blossom Algorithm & Implementation |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: PyMatching Implementation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Construct** the syndrome graph from surface code defect positions
2. **Explain** the mapping from decoding to minimum weight perfect matching
3. **Describe** the Blossom algorithm for polynomial-time matching
4. **Implement** MWPM decoding for rotated surface codes
5. **Analyze** the $O(n^3)$ complexity and threshold performance
6. **Handle** weighted edges for correlated and spatially-varying noise

---

## Core Content

### 1. From Syndromes to Graphs

The MWPM decoder exploits a beautiful connection between surface code decoding and graph theory. After syndrome measurement, we observe **defects**---locations where stabilizers report $-1$ outcomes.

**Key Insight:** In topological codes, errors create pairs of defects connected by error chains. The decoding problem becomes: find the minimum-weight pairing of defects.

**Syndrome Graph Construction:**

For a distance-$d$ surface code:
1. **Vertices**: Place a vertex at each defect location (syndrome bit = 1)
2. **Edges**: Connect all pairs of vertices with weighted edges
3. **Boundary vertices**: Add virtual vertices at boundaries (for chains ending at edges)
4. **Edge weights**: $w(v_i, v_j) = $ minimum weight of error chain connecting $v_i$ to $v_j$

$$\boxed{w(v_i, v_j) = d_{\text{Manhattan}}(v_i, v_j) \cdot |\log(p/(1-p))|}$$

where $d_{\text{Manhattan}}$ is the graph distance and $p$ is the physical error rate.

### 2. The Perfect Matching Problem

**Definition (Perfect Matching):** A perfect matching $M$ of a graph $G = (V, E)$ is a subset of edges such that every vertex is incident to exactly one edge in $M$.

**Minimum Weight Perfect Matching (MWPM):** Find a perfect matching minimizing total edge weight:

$$\boxed{M^* = \underset{M \text{ perfect}}{\text{argmin}} \sum_{e \in M} w(e)}$$

**Why MWPM Works for Surface Codes:**

1. Errors create defects in pairs (Pauli errors anticommute with exactly 2 stabilizers per type)
2. Each matching edge represents a hypothesized error chain
3. Minimum weight matching corresponds to minimum weight error hypothesis
4. This approximates MLD under i.i.d. noise

**Theorem:** For the surface code with i.i.d. depolarizing noise, MWPM achieves threshold:
$$\boxed{p_{\text{th}}^{\text{MWPM}} \approx 10.3\%}$$

compared to the optimal $p_{\text{th}}^{\text{MLD}} \approx 10.9\%$.

### 3. The Blossom Algorithm

Jack Edmonds' Blossom algorithm (1965) solves MWPM in polynomial time. The key challenge is handling **odd cycles** (blossoms).

**Algorithm Overview:**

1. Start with empty matching
2. Grow **alternating trees** from unmatched vertices
3. When trees meet: **augment** the matching
4. When odd cycle found: **shrink** it into a pseudo-vertex (blossom)
5. Recursively solve the contracted graph
6. **Expand** blossoms to recover original matching

**Complexity Analysis:**

$$\boxed{T_{\text{Blossom}}(n) = O(n^3)}$$

where $n$ is the number of vertices. For surface codes, $n \sim d^2$ defects in the worst case, giving $O(d^6)$ per decoding round.

Modern implementations (Blossom V) achieve practical efficiency through:
- Lazy edge evaluation
- Priority queue-based tree growth
- Sparse graph representations

### 4. Weighted Matching for Realistic Noise

Real quantum devices have non-uniform noise. MWPM naturally accommodates this through edge weights.

**Log-Likelihood Weights:**

For an error chain $C$ from $v_i$ to $v_j$ with probability $P(C)$:

$$w(v_i, v_j) = -\log P(C)$$

Under i.i.d. noise with error probability $p$ per qubit:

$$w(v_i, v_j) = |C| \cdot \left(-\log \frac{p/3}{1-p}\right) = |C| \cdot \log\frac{3(1-p)}{p}$$

**Handling Correlated Errors:**

For correlated noise (e.g., crosstalk), modify edge weights:

$$w(v_i, v_j) = -\log P(\text{optimal chain } v_i \to v_j | \text{noise model})$$

This requires precomputing shortest paths on a weighted lattice graph.

### 5. Boundary Handling

Surface codes have boundaries where error chains can terminate. This requires special treatment:

**Virtual Boundary Vertices:**
1. Add virtual vertices along each boundary
2. Connect defects to nearest boundary vertices
3. Virtual-to-virtual edges have weight 0

**Implementation:**
```
For each defect v:
    d_boundary = distance to nearest boundary
    Add edge (v, boundary_vertex) with weight d_boundary
```

This allows chains to terminate at boundaries with appropriate weight penalty.

### 6. Measurement Errors and Space-Time Decoding

With faulty syndrome measurements, we extend MWPM to 3D space-time:

**Space-Time Syndrome Graph:**
1. Stack 2D syndrome graphs for each measurement round
2. Connect defects between adjacent time slices
3. Measurement errors appear as defects at a single time that disappear

**3D Matching Complexity:**

For $d$ rounds of syndrome measurement on a distance-$d$ code:
$$T = O((d^2 \cdot d)^3) = O(d^9)$$

This can be reduced through windowed decoding or parallel approaches.

---

## Worked Examples

### Example 1: Syndrome Graph for Distance-3 Surface Code

**Problem:** Construct the syndrome graph for a distance-3 rotated surface code with defects at positions (0,1) and (2,1).

**Solution:**

The distance-3 rotated surface code has a 3x3 data qubit grid with:
- 4 X-stabilizers (measuring Z errors)
- 4 Z-stabilizers (measuring X errors)

For two defects at $(0,1)$ and $(2,1)$:

**Vertices:**
- $v_1$: defect at (0,1)
- $v_2$: defect at (2,1)
- $b_L$: left boundary vertex
- $b_R$: right boundary vertex

**Edges and Weights** (assuming $p = 0.1$, so $\log(3(1-p)/p) \approx 3.0$):

| Edge | Manhattan Distance | Weight |
|------|-------------------|--------|
| $(v_1, v_2)$ | 2 | 6.0 |
| $(v_1, b_L)$ | 1 | 3.0 |
| $(v_2, b_R)$ | 1 | 3.0 |
| $(b_L, b_R)$ | 0 | 0 |

**MWPM Solution:**

Compare matchings:
1. $\{(v_1, v_2), (b_L, b_R)\}$: total weight = 6.0 + 0 = 6.0
2. $\{(v_1, b_L), (v_2, b_R)\}$: total weight = 3.0 + 3.0 = 6.0

Both matchings have equal weight! This reflects ambiguity when defects are equidistant from each other and boundaries.

Correction:
- Matching 1 implies error chain between (0,1) and (2,1): errors on data qubits at (1,1)
- Matching 2 implies chains to boundaries: errors on edge qubits

### Example 2: Blossom Contraction

**Problem:** Apply the Blossom algorithm to a graph with vertices {A, B, C, D} where A, B, C form an odd cycle.

**Solution:**

Initial graph:
- Edges: (A,B), (B,C), (C,A), (C,D)
- Weights: all equal to 1

**Step 1:** Start alternating tree from unmatched vertex A
- A is the root

**Step 2:** Explore neighbors: B, C
- Tree: A - B, A - C

**Step 3:** Detect odd cycle A-B-C-A
- This is a blossom!

**Step 4:** Contract blossom {A, B, C} into pseudo-vertex $\beta$
- Contracted graph: vertices {$\beta$, D}
- Edges: ($\beta$, D) with weight 1

**Step 5:** Match in contracted graph
- Matching: ($\beta$, D)

**Step 6:** Expand blossom
- D connects to C in original graph
- Match C-D
- Remaining A, B must match: A-B

**Final Matching:** {(A, B), (C, D)}

### Example 3: Threshold Estimation via Monte Carlo

**Problem:** Estimate the MWPM threshold by simulating decoding at various error rates.

**Solution:**

Algorithm:
```
For each error rate p in [0.08, 0.09, 0.10, 0.11, 0.12]:
    For each code distance d in [3, 5, 7, 9]:
        failures = 0
        For trial = 1 to 10000:
            Generate random errors with rate p
            Compute syndrome
            Run MWPM decoder
            Check if logical error occurred
            If yes: failures += 1
        error_rate[p, d] = failures / 10000

Plot error_rate vs p for each d
Threshold = crossing point where curves intersect
```

Results show curves crossing near $p \approx 0.103$, confirming:

$$p_{\text{th}}^{\text{MWPM}} \approx 10.3\%$$

---

## Practice Problems

### Level A: Direct Application

**A1.** For a surface code with defects at positions (0,0), (2,0), (1,2), (3,2), draw the complete syndrome graph with all edge weights (assume unit weight per lattice step).

**A2.** Given edge weights $w_{12} = 4$, $w_{13} = 5$, $w_{23} = 3$, $w_{1b} = 2$, $w_{2b} = 3$, $w_{3b} = 4$, find the MWPM by enumeration.

**A3.** Calculate the edge weight for a chain of length 5 with physical error rate $p = 0.05$.

### Level B: Intermediate Analysis

**B1.** Prove that the number of defects in a surface code is always even (assuming no boundary effects).

**B2.** For a distance-5 surface code, what is the maximum number of defects from a weight-2 error? Construct an example.

**B3.** Analyze how the MWPM threshold changes when measurement error probability differs from data error probability. What is the threshold when $p_m = 2p_d$?

### Level C: Advanced Problems

**C1.** Derive the space-time syndrome graph structure for the surface code with $T$ rounds of syndrome measurement. How does the graph topology differ from the 2D case?

**C2.** Prove that MWPM on a complete graph with $n$ vertices has at most $(n-1)!! = (n-1)(n-3)(n-5)\cdots$ distinct perfect matchings.

**C3.** Design a modification to MWPM that accounts for Y-errors (which create defects in both X and Z syndrome graphs simultaneously). How should edge weights be modified?

---

## Computational Lab: PyMatching Implementation

```python
"""
Day 772 Computational Lab: MWPM Decoding with PyMatching
Implementing and analyzing Minimum Weight Perfect Matching for surface codes

This lab uses PyMatching for efficient MWPM and demonstrates
threshold estimation through Monte Carlo simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

# Note: PyMatching would be imported as: import pymatching
# For this educational lab, we implement a simplified version

class SyndromeGraph:
    """
    Syndrome graph for MWPM decoding.
    Represents defects as vertices and error chains as weighted edges.
    """

    def __init__(self, distance: int, error_rate: float = 0.1):
        """
        Initialize syndrome graph for distance-d surface code.

        Args:
            distance: Code distance
            error_rate: Physical error probability
        """
        self.d = distance
        self.p = error_rate

        # Weight per unit distance (log-likelihood ratio)
        if error_rate > 0 and error_rate < 1:
            self.weight_per_edge = np.log((1 - error_rate) / (error_rate / 3))
        else:
            self.weight_per_edge = 1.0

    def compute_edge_weight(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> float:
        """Compute edge weight between two defect positions."""
        manhattan_dist = abs(v1[0] - v2[0]) + abs(v1[1] - v2[1])
        return manhattan_dist * self.weight_per_edge

    def distance_to_boundary(self, v: Tuple[int, int], boundary: str) -> int:
        """Compute distance from vertex to specified boundary."""
        if boundary == 'left':
            return v[0]
        elif boundary == 'right':
            return self.d - 1 - v[0]
        elif boundary == 'top':
            return v[1]
        elif boundary == 'bottom':
            return self.d - 1 - v[1]
        return self.d  # Default large distance


class SimpleMWPMDecoder:
    """
    Simplified MWPM decoder using brute-force matching for small instances.
    For production, use PyMatching or Blossom V.
    """

    def __init__(self, distance: int, error_rate: float = 0.1):
        self.d = distance
        self.p = error_rate
        self.graph = SyndromeGraph(distance, error_rate)

    def decode(self, defects: List[Tuple[int, int]],
               syndrome_type: str = 'Z') -> List[Tuple[int, int]]:
        """
        Decode defects to find correction.

        Args:
            defects: List of defect positions (x, y)
            syndrome_type: 'Z' for X-error detection, 'X' for Z-error detection

        Returns:
            List of data qubit positions to correct
        """
        if len(defects) == 0:
            return []

        if len(defects) % 2 == 1:
            # Odd number of defects: one must pair with boundary
            defects = defects.copy()
            # Add virtual boundary vertex
            defects.append((-1, -1))  # Marker for boundary

        # Find minimum weight perfect matching
        matching = self._find_mwpm(defects)

        # Convert matching to corrections
        corrections = []
        for v1, v2 in matching:
            if v1 == (-1, -1) or v2 == (-1, -1):
                # Chain to boundary
                real_defect = v2 if v1 == (-1, -1) else v1
                chain = self._chain_to_boundary(real_defect, syndrome_type)
            else:
                # Chain between defects
                chain = self._chain_between(v1, v2)
            corrections.extend(chain)

        return corrections

    def _find_mwpm(self, vertices: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Find minimum weight perfect matching using brute force.
        Only suitable for small numbers of defects.
        """
        n = len(vertices)
        if n == 0:
            return []
        if n == 2:
            return [(vertices[0], vertices[1])]

        # For larger n, enumerate all perfect matchings
        # This is exponential but works for small examples
        min_weight = float('inf')
        best_matching = []

        for matching in self._enumerate_matchings(list(range(n))):
            weight = 0
            edges = []
            for i, j in matching:
                v1, v2 = vertices[i], vertices[j]
                w = self._edge_weight(v1, v2)
                weight += w
                edges.append((v1, v2))

            if weight < min_weight:
                min_weight = weight
                best_matching = edges

        return best_matching

    def _edge_weight(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> float:
        """Compute edge weight, handling boundary vertices."""
        if v1 == (-1, -1) and v2 == (-1, -1):
            return 0  # Boundary to boundary
        if v1 == (-1, -1):
            return min(v2[0], self.d - 1 - v2[0]) * self.graph.weight_per_edge
        if v2 == (-1, -1):
            return min(v1[0], self.d - 1 - v1[0]) * self.graph.weight_per_edge
        return self.graph.compute_edge_weight(v1, v2)

    def _enumerate_matchings(self, indices: List[int]):
        """Generate all perfect matchings of indices."""
        if len(indices) == 0:
            yield []
            return
        if len(indices) == 2:
            yield [(indices[0], indices[1])]
            return

        first = indices[0]
        rest = indices[1:]

        for i, partner in enumerate(rest):
            remaining = rest[:i] + rest[i+1:]
            for sub_matching in self._enumerate_matchings(remaining):
                yield [(first, partner)] + sub_matching

    def _chain_between(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find error chain between two defects (Manhattan path)."""
        chain = []
        x1, y1 = v1
        x2, y2 = v2

        # Horizontal moves
        while x1 != x2:
            if x1 < x2:
                chain.append((x1, y1))
                x1 += 1
            else:
                x1 -= 1
                chain.append((x1, y1))

        # Vertical moves
        while y1 != y2:
            if y1 < y2:
                chain.append((x1, y1))
                y1 += 1
            else:
                y1 -= 1
                chain.append((x1, y1))

        return chain

    def _chain_to_boundary(self, v: Tuple[int, int], syndrome_type: str) -> List[Tuple[int, int]]:
        """Find error chain from defect to nearest boundary."""
        x, y = v
        chain = []

        # Choose nearest horizontal boundary
        if x <= self.d - 1 - x:
            # Go left
            while x > 0:
                x -= 1
                chain.append((x, y))
        else:
            # Go right
            while x < self.d - 1:
                chain.append((x, y))
                x += 1

        return chain


class SurfaceCodeSimulator:
    """
    Surface code simulator for MWPM threshold estimation.
    """

    def __init__(self, distance: int):
        self.d = distance
        self.n_data = distance ** 2
        self.n_ancilla = (distance ** 2 - 1)  # Approximate

    def generate_errors(self, p: float) -> np.ndarray:
        """
        Generate random X errors with probability p on each data qubit.

        Returns:
            Boolean array of length d^2 indicating errors
        """
        return np.random.random(self.n_data) < p

    def compute_syndrome(self, errors: np.ndarray) -> List[Tuple[int, int]]:
        """
        Compute syndrome (defect positions) from error pattern.

        For simplified model, defects appear at corners of error regions.
        """
        # Reshape errors into 2D grid
        error_grid = errors.reshape((self.d, self.d))

        defects = []

        # Z-stabilizer checks (detecting X errors)
        # Defect at (i, j) if odd number of adjacent errors
        for i in range(self.d - 1):
            for j in range(self.d - 1):
                # Check 2x2 plaquette
                parity = (error_grid[i, j] ^ error_grid[i+1, j] ^
                         error_grid[i, j+1] ^ error_grid[i+1, j+1])
                if parity:
                    defects.append((i, j))

        return defects

    def check_logical_error(self, errors: np.ndarray,
                           corrections: List[Tuple[int, int]]) -> bool:
        """
        Check if error + correction results in logical error.

        A logical X error occurs if the net effect crosses the code
        horizontally (for Z-type logical operator).
        """
        # Create correction array
        correction_array = np.zeros(self.n_data, dtype=bool)
        for (x, y) in corrections:
            if 0 <= x < self.d and 0 <= y < self.d:
                idx = x * self.d + y
                if idx < self.n_data:
                    correction_array[idx] = True

        # Net error after correction
        net_error = errors ^ correction_array

        # Check for logical error: odd number of errors along any row
        net_grid = net_error.reshape((self.d, self.d))

        # Logical X error if odd parity across any column
        for j in range(self.d):
            if np.sum(net_grid[:, j]) % 2 == 1:
                return True

        return False


def threshold_estimation():
    """
    Estimate MWPM threshold through Monte Carlo simulation.
    """
    print("=" * 60)
    print("MWPM Threshold Estimation via Monte Carlo")
    print("=" * 60)

    distances = [3, 5, 7]
    error_rates = np.linspace(0.05, 0.15, 11)
    n_trials = 1000

    results = {d: [] for d in distances}

    for d in distances:
        print(f"\nSimulating distance-{d} code...")
        simulator = SurfaceCodeSimulator(d)
        decoder = SimpleMWPMDecoder(d)

        for p in error_rates:
            failures = 0

            for _ in range(n_trials):
                # Generate errors
                errors = simulator.generate_errors(p)

                # Get syndrome
                defects = simulator.compute_syndrome(errors)

                # Decode
                corrections = decoder.decode(defects)

                # Check for logical error
                if simulator.check_logical_error(errors, corrections):
                    failures += 1

            logical_error_rate = failures / n_trials
            results[d].append(logical_error_rate)

            print(f"  p={p:.3f}: logical error rate = {logical_error_rate:.4f}")

    # Plot results
    plt.figure(figsize=(10, 7))

    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    for i, d in enumerate(distances):
        plt.semilogy(error_rates * 100, results[d],
                    f'{colors[i]}{markers[i]}-',
                    label=f'd = {d}', linewidth=2, markersize=8)

    plt.axvline(x=10.3, color='gray', linestyle='--',
                label=f'Theoretical threshold (~10.3%)')

    plt.xlabel('Physical Error Rate (%)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('MWPM Decoder Threshold Estimation', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([5, 15])

    plt.tight_layout()
    plt.savefig('mwpm_threshold.png', dpi=150)
    plt.show()

    print("\nPlot saved to mwpm_threshold.png")

    # Estimate threshold from crossing point
    print("\n" + "=" * 60)
    print("Threshold Analysis")
    print("=" * 60)
    print("Threshold is where curves for different distances cross.")
    print("Theoretical MWPM threshold: ~10.3%")


def complexity_demo():
    """
    Demonstrate MWPM complexity scaling.
    """
    print("\n" + "=" * 60)
    print("MWPM Complexity Analysis")
    print("=" * 60)

    print("\nBlossom Algorithm Complexity: O(n^3)")
    print("where n = number of syndrome defects")
    print("\nFor distance-d surface code:")
    print("  Worst case defects: O(d^2)")
    print("  Decoding complexity: O(d^6)")
    print()

    # Timing analysis with our simple decoder
    distances = [3, 5, 7, 9]
    times = []

    for d in distances:
        simulator = SurfaceCodeSimulator(d)
        decoder = SimpleMWPMDecoder(d)

        # Generate worst-case scenario (many defects)
        total_time = 0
        n_trials = 100

        for _ in range(n_trials):
            errors = simulator.generate_errors(0.15)  # High error rate
            defects = simulator.compute_syndrome(errors)

            start = time.time()
            corrections = decoder.decode(defects)
            total_time += time.time() - start

        avg_time = total_time / n_trials
        times.append(avg_time)
        print(f"d = {d}: avg decode time = {avg_time*1000:.3f} ms")

    # Note: Our brute-force decoder scales worse than O(n^3)
    # Production decoders (PyMatching, Blossom V) achieve true O(n^3)


def matching_visualization():
    """
    Visualize syndrome graph and matching.
    """
    print("\n" + "=" * 60)
    print("Syndrome Graph Visualization")
    print("=" * 60)

    d = 5
    defects = [(0, 1), (2, 2), (3, 1), (4, 3)]

    print(f"\nDefect positions: {defects}")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Syndrome graph
    ax1.set_title('Syndrome Graph', fontsize=14)

    # Draw lattice
    for i in range(d):
        for j in range(d):
            ax1.plot(i, j, 'ko', markersize=5, alpha=0.3)

    # Draw defects
    for (x, y) in defects:
        ax1.plot(x, y, 'ro', markersize=15)
        ax1.annotate(f'({x},{y})', (x+0.1, y+0.1), fontsize=10)

    # Draw all possible edges
    for i, v1 in enumerate(defects):
        for v2 in defects[i+1:]:
            ax1.plot([v1[0], v2[0]], [v1[1], v2[1]],
                    'b-', alpha=0.2, linewidth=1)

    # Draw boundary connections
    for (x, y) in defects:
        # Left boundary
        ax1.plot([x, -0.5], [y, y], 'g--', alpha=0.3)
        # Right boundary
        ax1.plot([x, d-0.5], [y, y], 'g--', alpha=0.3)

    ax1.set_xlim(-1, d)
    ax1.set_ylim(-1, d)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')

    # Right plot: MWPM solution
    ax2.set_title('MWPM Solution', fontsize=14)

    # Draw lattice
    for i in range(d):
        for j in range(d):
            ax2.plot(i, j, 'ko', markersize=5, alpha=0.3)

    # Draw defects
    for (x, y) in defects:
        ax2.plot(x, y, 'ro', markersize=15)

    # Draw matching edges (computed manually for this example)
    # Optimal matching: (0,1)-(3,1) and (2,2)-(4,3) or similar
    matching_edges = [
        ((0, 1), (3, 1)),  # Weight = 3
        ((2, 2), (4, 3))   # Weight = 3
    ]

    for (v1, v2) in matching_edges:
        ax2.plot([v1[0], v2[0]], [v1[1], v2[1]],
                'b-', linewidth=3, label='Matched edge')

        # Draw error chain
        x1, y1 = v1
        x2, y2 = v2

        # Horizontal chain
        for x in range(min(x1, x2), max(x1, x2)):
            ax2.plot([x+0.5], [y1], 'gX', markersize=10)

    ax2.set_xlim(-1, d)
    ax2.set_ylim(-1, d)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')

    plt.tight_layout()
    plt.savefig('mwpm_visualization.png', dpi=150)
    plt.show()

    print("\nVisualization saved to mwpm_visualization.png")


def compare_with_naive():
    """
    Compare MWPM with naive minimum weight decoder.
    """
    print("\n" + "=" * 60)
    print("MWPM vs Naive Decoder Comparison")
    print("=" * 60)

    d = 5
    p = 0.10
    n_trials = 500

    simulator = SurfaceCodeSimulator(d)
    mwpm_decoder = SimpleMWPMDecoder(d, p)

    mwpm_failures = 0
    naive_failures = 0

    for _ in range(n_trials):
        errors = simulator.generate_errors(p)
        defects = simulator.compute_syndrome(errors)

        # MWPM decode
        mwpm_corrections = mwpm_decoder.decode(defects)
        if simulator.check_logical_error(errors, mwpm_corrections):
            mwpm_failures += 1

        # Naive decode: just pair defects arbitrarily
        naive_corrections = []
        for i in range(0, len(defects) - 1, 2):
            v1, v2 = defects[i], defects[i+1]
            # Connect with Manhattan path
            x1, y1 = v1
            x2, y2 = v2
            while x1 != x2:
                naive_corrections.append((min(x1, x2), y1))
                x1 = x1 + 1 if x1 < x2 else x1 - 1

        if simulator.check_logical_error(errors, naive_corrections):
            naive_failures += 1

    print(f"\nDistance-{d} code, p = {p}, {n_trials} trials:")
    print(f"  MWPM logical error rate: {mwpm_failures/n_trials:.4f}")
    print(f"  Naive logical error rate: {naive_failures/n_trials:.4f}")
    print(f"  Improvement factor: {naive_failures/max(mwpm_failures, 1):.2f}x")


if __name__ == "__main__":
    # Run demonstrations
    matching_visualization()
    complexity_demo()
    compare_with_naive()
    threshold_estimation()

    print("\n" + "=" * 60)
    print("Lab Complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Edge Weight (i.i.d.) | $$w(v_i, v_j) = d(v_i, v_j) \cdot \log\frac{3(1-p)}{p}$$ |
| MWPM Objective | $$M^* = \underset{M}{\text{argmin}} \sum_{e \in M} w(e)$$ |
| Blossom Complexity | $$O(n^3)$$ where $n$ = defects |
| Surface Code Decoding | $$O(d^6)$$ per round |
| MWPM Threshold | $$p_{\text{th}}^{\text{MWPM}} \approx 10.3\%$$ |
| Space-Time Decoding | $$O(d^9)$$ for $d$ rounds |

### Key Takeaways

1. **Syndromes map to graphs**: Defects become vertices, error chains become edges
2. **MWPM approximates MLD**: Minimum weight error assumption under i.i.d. noise
3. **Blossom algorithm is key**: Polynomial time through odd cycle contraction
4. **Boundaries require virtual vertices**: Enable chains to terminate at edges
5. **Correlated noise needs weighted graphs**: Log-likelihood weights handle non-uniform noise
6. **Threshold is 10.3%**: Near-optimal, practical for current experiments

---

## Daily Checklist

- [ ] Understood syndrome-to-graph mapping for surface codes
- [ ] Learned the Blossom algorithm at conceptual level
- [ ] Analyzed $O(n^3)$ complexity and its implications
- [ ] Implemented simplified MWPM decoder
- [ ] Ran threshold estimation simulations
- [ ] Compared MWPM against naive decoding
- [ ] Completed practice problems (at least Level A and B)

---

## Preview: Day 773

Tomorrow we explore **Union-Find Decoders**, which achieve almost-linear time complexity $O(n \cdot \alpha(n))$. While sacrificing some threshold accuracy compared to MWPM, Union-Find decoders are crucial for real-time decoding in large-scale quantum computers.

Key questions for tomorrow:
- How does cluster growth replace perfect matching?
- What is the Union-Find data structure and inverse Ackermann function?
- Can we achieve both speed and accuracy?

---

*Day 772 of 2184 | Week 111 | Month 28 | Year 2: Advanced Quantum Science*
