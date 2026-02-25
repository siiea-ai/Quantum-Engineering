# Day 829: Union-Find Decoder

## Week 119: Real-Time Decoding | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Schedule (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 2.5 hours | Union-Find data structure, cluster growth |
| **Afternoon** | 2.5 hours | Peeling decoder, accuracy analysis |
| **Evening** | 2 hours | Union-Find implementation lab |

---

## Learning Objectives

By the end of Day 829, you will be able to:

1. **Explain** the Union-Find data structure and its near-constant-time operations
2. **Describe** the cluster growth algorithm for syndrome decoding
3. **Implement** a complete Union-Find decoder for surface codes
4. **Analyze** the time complexity $O(n \cdot \alpha(n))$ and its practical implications
5. **Compare** Union-Find threshold (~9.9%) to MWPM threshold (~10.3%)
6. **Evaluate** the accuracy-speed trade-off in real-time decoding scenarios

---

## Core Content

### 1. The Union-Find Data Structure

The Union-Find (or Disjoint Set Union) data structure maintains a collection of disjoint sets, supporting:

- **Find(x)**: Return the representative of the set containing x
- **Union(x, y)**: Merge the sets containing x and y

With **path compression** and **union by rank**, both operations achieve:

$$\boxed{O(\alpha(n)) \text{ amortized time per operation}}$$

where $\alpha(n)$ is the **inverse Ackermann function**.

#### The Inverse Ackermann Function

The Ackermann function $A(m, n)$ grows extraordinarily fast:
- $A(1, n) = 2n$
- $A(2, n) = 2^n$
- $A(3, n) = 2^{2^{\cdot^{\cdot^{\cdot}}}}$ (tower of $n$ 2s)
- $A(4, n)$ is beyond astronomical

Its inverse $\alpha(n)$ grows correspondingly slowly:
- $\alpha(n) \leq 4$ for all $n < 10^{80}$ (number of atoms in universe)
- For practical purposes: $\alpha(n) \approx \text{constant}$

This makes Union-Find effectively $O(n)$ for $n$ operations.

### 2. Cluster Growth Algorithm

The Union-Find decoder (Delfosse & Nickerson 2021) works by growing clusters from syndrome defects.

#### Algorithm Overview

1. **Initialization**: Each defect is its own cluster
2. **Growth**: Clusters grow simultaneously by one lattice step
3. **Fusion**: When cluster boundaries touch, merge clusters
4. **Termination**: Continue until all clusters are even (contain even number of defects)
5. **Peeling**: Extract correction from cluster spanning trees

#### Key Insight

Unlike MWPM which finds global optimal matching, Union-Find grows clusters locally. The algorithm is **greedy** but achieves near-optimal performance for typical error patterns.

#### Cluster Parity

A cluster is **even** if it contains an even number of defects. Even clusters can be corrected internally. **Odd** clusters must grow to merge with other odd clusters.

### 3. Detailed Algorithm Steps

#### Step 1: Initialize Clusters

For each defect $v$:
- Create singleton cluster $C_v = \{v\}$
- $\text{parity}(C_v) = 1$ (odd)
- $\text{boundary}(C_v) = \{\text{edges adjacent to } v\}$

#### Step 2: Grow Clusters

```
while any cluster has odd parity:
    for each odd cluster C:
        for each edge e on boundary(C):
            if e connects to another cluster C':
                Union(C, C')
                parity(C ∪ C') = parity(C) XOR parity(C')
            else:
                grow C to include vertex at other end of e
```

#### Step 3: Fusion

When clusters $C_1$ and $C_2$ merge:
- $\text{parity}(C_1 \cup C_2) = \text{parity}(C_1) \oplus \text{parity}(C_2)$
- Combined cluster is even if both were odd

#### Step 4: Peeling

After all clusters are even:
1. Build spanning tree of each cluster
2. For each defect, trace path to matched partner
3. Correction = XOR of edges along paths

### 4. Time Complexity Analysis

Let $n = d^2$ be the number of syndrome bits.

**Growth phase**:
- At most $O(d)$ growth rounds (cluster radius bounded by code distance)
- Each round processes $O(n)$ boundary edges
- Each edge operation involves $O(\alpha(n))$ Union-Find operations

**Total complexity**:
$$\boxed{T(n) = O(d \cdot n \cdot \alpha(n)) = O(d^3 \cdot \alpha(d^2))}$$

For fixed $d$: $O(n \cdot \alpha(n))$ per syndrome.

In practice, clusters are sparse (few defects), so actual runtime is much better:
$$T_{\text{typical}} = O(k \cdot d \cdot \alpha(n))$$
where $k$ is the number of defects (typically $O(1)$ at low error rates).

### 5. Threshold Analysis

The Union-Find decoder achieves a threshold of approximately:

$$\boxed{p_{\text{th}}^{\text{UF}} \approx 9.9\% \pm 0.1\%}$$

compared to MWPM:
$$p_{\text{th}}^{\text{MWPM}} \approx 10.3\% \pm 0.1\%$$

The threshold gap is small (~4%) because:
- Most errors create simple, local syndrome patterns
- Greedy matching works well for low-density defects
- Suboptimality arises mainly for complex, overlapping error chains

### 6. Weighted Union-Find

To improve accuracy, weights can be incorporated:

**Growth Speed**: Clusters grow at speeds proportional to edge weights:
$$v_{\text{grow}}(e) \propto \frac{1}{w(e)}$$

High-probability edges (low weight) are traversed first, approximating MWPM behavior.

**Complexity**: Still $O(n \cdot \alpha(n))$, but with larger constants.

### 7. 3D Union-Find (Spacetime)

For repeated syndrome measurements, extend to 3D:

- Vertices: $(x, y, t)$ positions in spacetime
- Spatial edges: connect $(x, y, t)$ to neighbors at same $t$
- Temporal edges: connect $(x, y, t)$ to $(x, y, t \pm 1)$

The algorithm remains $O(n \cdot \alpha(n))$ per syndrome round with careful implementation.

---

## Worked Examples

### Example 1: Simple Cluster Growth

**Problem**: Four defects are arranged in a square pattern on a distance-5 surface code. Trace the Union-Find algorithm execution.

**Solution**:

Initial configuration:
```
. . . . .
. D . D .     D = defect
. . . . .
. D . D .
. . . . .
```

**Round 1**: Each defect grows by 1 step:
```
Cluster 1: {(1,1)} → boundary includes (0,1), (2,1), (1,0), (1,2)
Cluster 2: {(1,3)} → boundary includes (0,3), (2,3), (1,2), (1,4)
Cluster 3: {(3,1)} → boundary includes (2,1), (4,1), (3,0), (3,2)
Cluster 4: {(3,3)} → boundary includes (2,3), (4,3), (3,2), (3,4)
```

**Fusion**: Clusters 1 and 2 share edge (1,2)-(1,3), so they merge.
Similarly for 3 and 4.

New clusters:
- $C_{12}$: parity = $1 \oplus 1 = 0$ (even!)
- $C_{34}$: parity = $1 \oplus 1 = 0$ (even!)

**Termination**: All clusters even, stop growing.

**Peeling**: In $C_{12}$, match defect at (1,1) to defect at (1,3). Correction: flip edge between them.

$$\boxed{\text{Correction: horizontal edge at row 1}}$$

### Example 2: Complexity Calculation

**Problem**: For a distance-21 surface code with 10 defects, estimate Union-Find decode time assuming 10 ns per Union-Find operation.

**Solution**:

Number of syndrome bits: $n = 21^2 = 441$

Expected growth rounds: At most $d/2 = 10$ (if defects are at opposite corners)

Operations per defect per round: ~4 (boundary edges) × $\alpha(441) \approx 4$

Total operations estimate:
$$N_{\text{ops}} = 10 \text{ defects} \times 10 \text{ rounds} \times 4 \times 4 = 1600$$

But with path compression, many operations are cached. Realistic estimate:
$$N_{\text{ops}} \approx 10 \times 4 \times \log(d) \approx 10 \times 4 \times 5 = 200$$

Decode time:
$$t_{\text{decode}} = 200 \times 10 \text{ ns} = 2000 \text{ ns} = 2 \, \mu\text{s}$$

$$\boxed{t_{\text{decode}} \approx 1-5 \, \mu\text{s}}$$

### Example 3: Threshold Comparison

**Problem**: At physical error rate $p = 9.0\%$, compare logical error rates for Union-Find vs MWPM on a distance-7 code.

**Solution**:

For MWPM with $p_{\text{th}} = 10.3\%$:
$$p_L^{\text{MWPM}} \approx \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2} = \left(\frac{0.09}{0.103}\right)^4 = (0.874)^4 = 0.58$$

For Union-Find with $p_{\text{th}} = 9.9\%$:
$$p_L^{\text{UF}} \approx \left(\frac{0.09}{0.099}\right)^4 = (0.909)^4 = 0.68$$

The difference:
$$\frac{p_L^{\text{UF}}}{p_L^{\text{MWPM}}} = \frac{0.68}{0.58} = 1.17$$

Union-Find has ~17% higher logical error rate at this operating point.

At $p = 5\%$ (more realistic operating point):
$$p_L^{\text{MWPM}} \approx \left(\frac{0.05}{0.103}\right)^4 = 0.055$$
$$p_L^{\text{UF}} \approx \left(\frac{0.05}{0.099}\right)^4 = 0.065$$

$$\boxed{\text{Ratio: } 1.18 \text{ (Union-Find ~18\% worse)}}$$

---

## Practice Problems

### Direct Application

**Problem 1**: Implement the Find operation with path compression. What is the worst-case tree height after $n$ Find operations?

**Problem 2**: Three defects at positions (0,0), (2,2), and (4,0) on a distance-5 code. How many growth rounds until all clusters merge?

### Intermediate

**Problem 3**: A cluster contains defects at (1,1), (1,5), (5,1). What is the cluster parity? After growing twice, what is the cluster boundary?

**Problem 4**: Derive the condition under which Union-Find produces a different correction than MWPM for a given syndrome.

### Challenging

**Problem 5**: For weighted Union-Find, design a growth schedule where edges with weight $w$ are crossed after time $t \propto w$. Show this approximates MWPM for sparse errors.

**Problem 6**: Analyze the "almost-linear" claim: prove that Union-Find achieves $O(n \cdot \alpha(n))$ time complexity for surface code decoding.

---

## Computational Lab: Union-Find Implementation

```python
"""
Day 829 Lab: Union-Find Decoder Implementation
Complete implementation with visualization and benchmarking
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from time import perf_counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Part 1: Union-Find Data Structure
# =============================================================================

class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.

    Achieves O(α(n)) amortized time per operation.
    """

    def __init__(self, n):
        """
        Initialize n singleton sets.

        Parameters:
        -----------
        n : int
            Number of elements (0 to n-1)
        """
        self.parent = list(range(n))  # Each element is its own parent
        self.rank = [0] * n           # Rank for union by rank
        self.size = [1] * n           # Size of each set
        self.n_sets = n               # Number of disjoint sets

    def find(self, x):
        """
        Find representative of set containing x.

        Uses path compression for efficiency.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """
        Merge sets containing x and y.

        Uses union by rank for efficiency.
        Returns True if sets were different, False if already same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        self.n_sets -= 1
        return True

    def get_size(self, x):
        """Return size of set containing x."""
        return self.size[self.find(x)]

    def same_set(self, x, y):
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)


def test_union_find():
    """Test and demonstrate Union-Find."""
    print("=" * 60)
    print("UNION-FIND DATA STRUCTURE")
    print("=" * 60)

    uf = UnionFind(10)

    print("\nInitial state: 10 singleton sets")
    print(f"Parent array: {uf.parent}")

    # Perform some unions
    operations = [(0, 1), (2, 3), (4, 5), (1, 3), (5, 7)]

    for x, y in operations:
        uf.union(x, y)
        print(f"Union({x}, {y}): {uf.n_sets} sets remaining")

    print(f"\nFinal parent array: {uf.parent}")
    print(f"Find(0) = {uf.find(0)}, Find(3) = {uf.find(3)}")
    print(f"Same set? {uf.same_set(0, 3)}")

    # Benchmark
    n_large = 100000
    uf_large = UnionFind(n_large)

    t0 = perf_counter()
    for i in range(n_large - 1):
        uf_large.union(i, i + 1)
    t_union = perf_counter() - t0

    t0 = perf_counter()
    for i in range(n_large):
        uf_large.find(i)
    t_find = perf_counter() - t0

    print(f"\nBenchmark ({n_large} elements):")
    print(f"  {n_large-1} union operations: {t_union*1e6:.1f} μs")
    print(f"  {n_large} find operations: {t_find*1e6:.1f} μs")
    print(f"  Average per operation: {(t_union + t_find) / (2*n_large) * 1e9:.1f} ns")

test_union_find()

# =============================================================================
# Part 2: Surface Code Lattice Representation
# =============================================================================

class SurfaceCodeLattice:
    """
    Simplified surface code lattice for Union-Find decoding.
    """

    def __init__(self, distance):
        """
        Initialize lattice.

        Parameters:
        -----------
        distance : int
            Code distance (odd number)
        """
        self.d = distance

        # Stabilizer positions (simplified model)
        # X stabilizers at even positions, Z at odd (checkerboard)
        self.stabilizers = []
        self.stab_to_idx = {}
        idx = 0
        for i in range(distance):
            for j in range(distance):
                self.stabilizers.append((i, j))
                self.stab_to_idx[(i, j)] = idx
                idx += 1

        self.n_stab = len(self.stabilizers)

        # Build adjacency (4-connected grid)
        self.neighbors = defaultdict(list)
        for i, j in self.stabilizers:
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < distance and 0 <= nj < distance:
                    self.neighbors[(i, j)].append((ni, nj))

    def idx_to_pos(self, idx):
        """Convert index to (i, j) position."""
        return self.stabilizers[idx]

    def pos_to_idx(self, pos):
        """Convert (i, j) position to index."""
        return self.stab_to_idx.get(pos, -1)

    def get_neighbors(self, pos):
        """Get neighboring positions."""
        return self.neighbors[pos]

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# =============================================================================
# Part 3: Union-Find Decoder
# =============================================================================

class UnionFindDecoder:
    """
    Union-Find decoder for surface codes.

    Based on Delfosse & Nickerson (2021).
    """

    def __init__(self, lattice):
        """
        Initialize decoder.

        Parameters:
        -----------
        lattice : SurfaceCodeLattice
            The surface code lattice
        """
        self.lattice = lattice
        self.d = lattice.d

    def decode(self, syndrome):
        """
        Decode syndrome using Union-Find.

        Parameters:
        -----------
        syndrome : array-like
            Binary syndrome vector (1 = defect)

        Returns:
        --------
        correction : set of (pos1, pos2) tuples
            Edges to flip for correction
        clusters : list
            Final cluster assignments
        """
        syndrome = np.asarray(syndrome)
        n = len(syndrome)

        # Find defect positions
        defects = [self.lattice.idx_to_pos(i) for i in range(n) if syndrome[i]]

        if len(defects) == 0:
            return set(), []

        if len(defects) % 2 == 1:
            # Odd number of defects - one matches to boundary
            # Add virtual boundary defect (simplified)
            defects.append((-1, -1))  # Virtual boundary

        # Initialize Union-Find
        n_defects = len(defects)
        uf = UnionFind(n_defects)

        # Track cluster parity (odd = need to grow)
        parity = [1] * n_defects  # All start as odd (single defect)

        # Track cluster boundaries
        # boundary[i] = set of (defect_idx, neighbor_pos) for cluster i
        boundaries = [set() for _ in range(n_defects)]

        # Initialize boundaries
        for i, pos in enumerate(defects):
            if pos == (-1, -1):  # Virtual boundary
                continue
            for neighbor in self.lattice.get_neighbors(pos):
                boundaries[i].add((i, neighbor))

        # Cluster growth loop
        max_rounds = 2 * self.d  # Safety limit
        round_num = 0

        while round_num < max_rounds:
            round_num += 1

            # Check if all clusters are even
            all_even = True
            for i in range(n_defects):
                root = uf.find(i)
                if parity[root] == 1:  # Odd cluster
                    all_even = False
                    break

            if all_even:
                break

            # Grow odd clusters
            new_merges = []

            for i in range(n_defects):
                root_i = uf.find(i)
                if parity[root_i] == 0:  # Even cluster, don't grow
                    continue

                # Check boundary for merges
                for (defect_idx, neighbor) in list(boundaries[root_i]):
                    # See if neighbor is in another defect's cluster
                    for j in range(n_defects):
                        if defects[j] == neighbor:
                            root_j = uf.find(j)
                            if root_i != root_j:
                                new_merges.append((root_i, root_j))

            # Apply merges
            for root_i, root_j in new_merges:
                # Update actual roots (may have changed)
                root_i = uf.find(root_i)
                root_j = uf.find(root_j)

                if root_i != root_j:
                    # Merge parity
                    new_parity = parity[root_i] ^ parity[root_j]

                    # Merge sets
                    uf.union(root_i, root_j)

                    # Update parity of new root
                    new_root = uf.find(root_i)
                    parity[new_root] = new_parity

                    # Merge boundaries
                    boundaries[new_root] = boundaries[root_i] | boundaries[root_j]

            # Expand boundaries (grow by one step)
            for i in range(n_defects):
                root = uf.find(i)
                if parity[root] == 0:
                    continue

                new_boundary = set()
                for (defect_idx, pos) in boundaries[root]:
                    for neighbor in self.lattice.get_neighbors(pos):
                        if neighbor not in [defects[j] for j in range(n_defects)]:
                            new_boundary.add((defect_idx, neighbor))
                boundaries[root] |= new_boundary

        # Extract correction (simplified: just return cluster info)
        clusters = [uf.find(i) for i in range(n_defects)]

        # Generate correction by finding paths within clusters
        correction = self._extract_correction(defects, clusters, uf)

        return correction, clusters

    def _extract_correction(self, defects, clusters, uf):
        """
        Extract correction from cluster assignments.

        Uses simple path finding between matched defects.
        """
        correction = set()

        # Group defects by cluster
        cluster_defects = defaultdict(list)
        for i, (pos, cluster) in enumerate(zip(defects, clusters)):
            cluster_defects[cluster].append(pos)

        # For each cluster, pair up defects and find paths
        for cluster_id, positions in cluster_defects.items():
            # Simple greedy pairing
            remaining = list(positions)

            while len(remaining) >= 2:
                pos1 = remaining.pop(0)
                if pos1 == (-1, -1):  # Skip virtual boundary
                    continue

                # Find closest remaining defect
                best_dist = float('inf')
                best_idx = 0
                for idx, pos2 in enumerate(remaining):
                    if pos2 == (-1, -1):
                        dist = min(pos1[0], pos1[1],
                                  self.d - 1 - pos1[0], self.d - 1 - pos1[1])
                    else:
                        dist = self.lattice.manhattan_distance(pos1, pos2)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                pos2 = remaining.pop(best_idx)

                # Add path to correction (simplified: straight line)
                if pos2 != (-1, -1):
                    # Horizontal then vertical path
                    x, y = pos1
                    while x != pos2[0]:
                        next_x = x + (1 if pos2[0] > x else -1)
                        correction.add(((x, y), (next_x, y)))
                        x = next_x
                    while y != pos2[1]:
                        next_y = y + (1 if pos2[1] > y else -1)
                        correction.add(((x, y), (x, next_y)))
                        y = next_y

        return correction


def demonstrate_union_find_decoder():
    """Demonstrate Union-Find decoder."""
    print("\n" + "=" * 60)
    print("UNION-FIND DECODER DEMONSTRATION")
    print("=" * 60)

    d = 7
    lattice = SurfaceCodeLattice(d)
    decoder = UnionFindDecoder(lattice)

    # Create syndrome with 4 defects
    syndrome = np.zeros(lattice.n_stab, dtype=int)
    defect_positions = [(1, 1), (1, 5), (5, 1), (5, 5)]

    for pos in defect_positions:
        idx = lattice.pos_to_idx(pos)
        if idx >= 0:
            syndrome[idx] = 1

    print(f"\nCode distance: {d}")
    print(f"Defects at: {defect_positions}")

    # Decode
    t0 = perf_counter()
    correction, clusters = decoder.decode(syndrome)
    decode_time = perf_counter() - t0

    print(f"\nDecoding time: {decode_time*1e6:.1f} μs")
    print(f"Correction edges: {len(correction)}")
    print(f"Cluster assignments: {clusters}")

    # Visualize
    plt.figure(figsize=(8, 8))

    # Draw lattice
    for i in range(d):
        for j in range(d):
            plt.plot(i, j, 'ko', markersize=5, alpha=0.3)

    # Draw defects with cluster colors
    colors = ['red', 'blue', 'green', 'orange']
    unique_clusters = list(set(clusters))

    for idx, (pos, cluster) in enumerate(zip(defect_positions + [(-1,-1)]*(len(clusters)-len(defect_positions)), clusters)):
        if pos != (-1, -1) and pos in defect_positions:
            color = colors[unique_clusters.index(cluster) % len(colors)]
            plt.plot(pos[0], pos[1], 'o', markersize=20, color=color)
            plt.text(pos[0], pos[1], 'D', ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')

    # Draw correction
    for (p1, p2) in correction:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=3, alpha=0.7)

    plt.xlim(-0.5, d - 0.5)
    plt.ylim(-0.5, d - 0.5)
    plt.title('Union-Find Decoder Result', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig('union_find_decoder.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'union_find_decoder.png'")

demonstrate_union_find_decoder()

# =============================================================================
# Part 4: Performance Benchmarking
# =============================================================================

def benchmark_decoder_scaling():
    """Benchmark Union-Find decoder scaling."""
    print("\n" + "=" * 60)
    print("UNION-FIND DECODER SCALING BENCHMARK")
    print("=" * 60)

    distances = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    p_error = 0.01  # 1% error rate

    decode_times = []
    n_defects_avg = []

    for d in distances:
        lattice = SurfaceCodeLattice(d)
        decoder = UnionFindDecoder(lattice)
        n_stab = lattice.n_stab

        times = []
        defect_counts = []

        for trial in range(50):
            # Generate random syndrome
            n_defects = np.random.poisson(p_error * n_stab)
            if n_defects % 2 == 1:
                n_defects += 1
            n_defects = min(n_defects, n_stab)

            syndrome = np.zeros(n_stab, dtype=int)
            defect_indices = np.random.choice(n_stab, size=n_defects, replace=False)
            syndrome[defect_indices] = 1

            defect_counts.append(n_defects)

            t0 = perf_counter()
            _, _ = decoder.decode(syndrome)
            times.append(perf_counter() - t0)

        decode_times.append(np.mean(times))
        n_defects_avg.append(np.mean(defect_counts))

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(distances, np.array(decode_times) * 1e6, 'bo-',
                 linewidth=2, markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', label='1 μs target')
    plt.xlabel('Code Distance', fontsize=12)
    plt.ylabel('Decode Time (μs)', fontsize=12)
    plt.title('Union-Find Decode Time Scaling', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    plt.subplot(1, 2, 2)
    plt.plot(distances, n_defects_avg, 'gs-', linewidth=2, markersize=8)
    plt.xlabel('Code Distance', fontsize=12)
    plt.ylabel('Average Defects', fontsize=12)
    plt.title(f'Defects per Syndrome (p = {p_error*100}%)', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('union_find_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Summary table
    print("\n" + "-" * 50)
    print(f"{'Distance':<10} {'Defects':<12} {'Time (μs)':<15}")
    print("-" * 50)
    for i, d in enumerate(distances):
        print(f"{d:<10} {n_defects_avg[i]:<12.1f} {decode_times[i]*1e6:<15.2f}")

    print("\nFigure saved to 'union_find_scaling.png'")

benchmark_decoder_scaling()

# =============================================================================
# Part 5: Threshold Comparison
# =============================================================================

def compare_thresholds():
    """Compare Union-Find vs MWPM thresholds (simulated)."""
    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON (SIMULATED)")
    print("=" * 60)

    # Threshold values from literature
    p_th_mwpm = 0.103
    p_th_uf = 0.099

    distances = [5, 7, 9, 11, 15, 21]
    error_rates = np.linspace(0.01, 0.12, 50)

    plt.figure(figsize=(10, 6))

    for d in distances:
        # Approximate logical error rate
        p_L_mwpm = (error_rates / p_th_mwpm) ** ((d + 1) / 2)
        p_L_uf = (error_rates / p_th_uf) ** ((d + 1) / 2)

        # Clip to [0, 1]
        p_L_mwpm = np.clip(p_L_mwpm, 0, 1)
        p_L_uf = np.clip(p_L_uf, 0, 1)

        plt.semilogy(error_rates * 100, p_L_mwpm, '-',
                     label=f'd={d} MWPM', linewidth=2, alpha=0.8)
        plt.semilogy(error_rates * 100, p_L_uf, '--',
                     label=f'd={d} UF', linewidth=2, alpha=0.8)

    plt.axvline(x=p_th_mwpm * 100, color='blue', linestyle=':',
                label=f'MWPM threshold ({p_th_mwpm*100}%)')
    plt.axvline(x=p_th_uf * 100, color='red', linestyle=':',
                label=f'UF threshold ({p_th_uf*100}%)')

    plt.xlabel('Physical Error Rate (%)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('MWPM vs Union-Find Threshold Comparison', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.grid(True, alpha=0.3, which='both')
    plt.ylim(1e-10, 1)

    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nMWPM threshold: {p_th_mwpm*100}%")
    print(f"Union-Find threshold: {p_th_uf*100}%")
    print(f"Threshold gap: {(p_th_mwpm - p_th_uf)*100:.1f}%")
    print(f"Relative gap: {(1 - p_th_uf/p_th_mwpm)*100:.1f}%")

    print("\nFigure saved to 'threshold_comparison.png'")

compare_thresholds()

# =============================================================================
# Part 6: Real-Time Feasibility Analysis
# =============================================================================

def analyze_realtime_feasibility():
    """Analyze real-time feasibility for different platforms."""
    print("\n" + "=" * 60)
    print("REAL-TIME FEASIBILITY ANALYSIS")
    print("=" * 60)

    # Platform timing budgets (ns)
    platforms = {
        'superconducting': 500,      # ~500 ns decode budget
        'trapped_ion': 1_000_000,    # ~1 ms decode budget
        'photonic': 200,             # ~200 ns decode budget
    }

    # Estimated decode times from benchmarks (ns)
    # Assuming optimized C++ implementation is 100x faster than Python
    python_scale = 100  # Python overhead factor

    distances = [5, 7, 9, 11, 15, 21]

    print("\nReal-time operation feasibility:\n")
    print("-" * 70)
    print(f"{'Platform':<18} {'Budget':<12} {'d=5':<8} {'d=7':<8} {'d=11':<8} {'d=21':<8}")
    print("-" * 70)

    for platform, budget in platforms.items():
        row = f"{platform:<18} {budget:>8} ns  "

        for d in [5, 7, 11, 21]:
            # Estimate decode time (based on scaling)
            # O(d^2 * α(d^2)) with small constant
            estimated_time = d * d * 4 * 10 / python_scale  # ns

            if estimated_time < budget:
                row += f"{'OK':^8}"
            elif estimated_time < budget * 10:
                row += f"{'CLOSE':^8}"
            else:
                row += f"{'NO':^8}"

        print(row)

    print("-" * 70)
    print("\nNotes:")
    print("  OK = Well within budget (< budget)")
    print("  CLOSE = Near limit (< 10x budget)")
    print("  NO = Exceeds budget significantly")

analyze_realtime_feasibility()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("LAB SUMMARY")
print("=" * 60)
print("""
Key findings:

1. UNION-FIND DATA STRUCTURE: Achieves near-constant time operations
   through path compression and union by rank. α(n) ≤ 4 for all
   practical n.

2. CLUSTER GROWTH: The algorithm grows clusters from defects until
   all have even parity. This greedy approach is fast but suboptimal.

3. COMPLEXITY: O(n · α(n)) ≈ O(n) for n syndrome bits.
   Much faster than MWPM's O(n²) to O(n³).

4. THRESHOLD: ~9.9% vs MWPM's ~10.3%. The 4% gap is small and
   often acceptable for the speed benefit.

5. REAL-TIME FEASIBILITY: Union-Find can achieve sub-μs decoding
   for moderate code distances (d ≤ 15) on optimized hardware.

6. TRADE-OFF: For superconducting qubits, the speed advantage of
   Union-Find often outweighs its slightly lower threshold.
""")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Union-Find complexity | $O(\alpha(n))$ per operation |
| Total decode complexity | $O(n \cdot \alpha(n))$ where $n = d^2$ |
| Inverse Ackermann | $\alpha(n) \leq 4$ for $n < 10^{80}$ |
| Union-Find threshold | $p_{\text{th}}^{\text{UF}} \approx 9.9\%$ |
| Threshold gap | $\Delta p_{\text{th}} \approx 0.4\%$ |

### Algorithm Steps

1. **Initialize**: Each defect forms singleton cluster with odd parity
2. **Grow**: Odd clusters expand boundaries by one lattice step
3. **Fuse**: Merging clusters XOR their parities
4. **Terminate**: All clusters have even parity
5. **Peel**: Extract correction from cluster spanning trees

---

## Daily Checklist

- [ ] I can explain path compression and union by rank optimizations
- [ ] I understand why $\alpha(n)$ is effectively constant
- [ ] I can trace the cluster growth algorithm on a simple example
- [ ] I can implement the core Union-Find decoder
- [ ] I understand the threshold trade-off vs MWPM
- [ ] I can analyze when Union-Find is preferable to MWPM

---

## Preview: Day 830

Tomorrow we explore **Neural Network Decoders**:
- Training on simulated syndrome data
- Network architectures for decoding
- Inference speed optimizations
- Approaching ML optimality with learned decoders

Neural decoders offer an alternative path to real-time decoding through machine learning.

---

*"The beauty of Union-Find is its simplicity: each operation is trivial, yet together they achieve near-optimal performance."*

---

[← Day 828: MWPM Optimization](./Day_828_Tuesday.md) | [Day 830: Neural Network Decoders →](./Day_830_Thursday.md)
