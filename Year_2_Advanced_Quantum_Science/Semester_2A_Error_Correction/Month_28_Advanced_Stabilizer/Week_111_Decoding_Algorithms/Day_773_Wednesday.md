# Day 773: Union-Find Decoders

## Week 111: Decoding Algorithms | Month 28: Advanced Stabilizer Codes

---

## Daily Schedule

| Session | Time | Duration | Focus |
|---------|------|----------|-------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Union-Find Data Structure & Cluster Growth |
| Afternoon | 1:00 PM - 4:00 PM | 3 hours | Almost-Linear Complexity & Peeling Decoder |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Union-Find Implementation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Implement** the Union-Find data structure with path compression and union by rank
2. **Explain** the cluster growth algorithm for topological code decoding
3. **Derive** the almost-linear $O(n \cdot \alpha(n))$ time complexity
4. **Analyze** the inverse Ackermann function and why it's effectively constant
5. **Compare** Union-Find threshold (~9.9%) against MWPM (~10.3%)
6. **Implement** the peeling decoder variant for erasure channels

---

## Core Content

### 1. The Need for Faster Decoding

While MWPM achieves near-optimal thresholds, its $O(n^3)$ complexity becomes problematic for large codes:

| Code Distance | Data Qubits | MWPM Time | Required for Real-Time |
|---------------|-------------|-----------|------------------------|
| 17 | 289 | ~1 ms | 1 $\mu$s |
| 25 | 625 | ~10 ms | 1 $\mu$s |
| 50 | 2500 | ~1 s | 1 $\mu$s |

The **decoder bottleneck** demands algorithms with better scaling. Union-Find decoders achieve almost-linear time while maintaining competitive thresholds.

### 2. The Union-Find Data Structure

Union-Find (Disjoint Set Union) efficiently maintains a partition of elements into disjoint sets.

**Operations:**
- `MakeSet(x)`: Create a new set containing only $x$
- `Find(x)`: Return the representative (root) of $x$'s set
- `Union(x, y)`: Merge the sets containing $x$ and $y$

**Basic Implementation:**
```
parent[x] = x  # Each element starts as its own root

Find(x):
    if parent[x] != x:
        return Find(parent[x])
    return x

Union(x, y):
    root_x = Find(x)
    root_y = Find(y)
    if root_x != root_y:
        parent[root_x] = root_y
```

### 3. Optimizations: Path Compression and Union by Rank

Two optimizations make Union-Find extremely efficient:

**Path Compression:** During `Find`, make all nodes point directly to root:
```
Find(x):
    if parent[x] != x:
        parent[x] = Find(parent[x])  # Path compression
    return parent[x]
```

**Union by Rank:** Always attach smaller tree under larger:
```
Union(x, y):
    root_x, root_y = Find(x), Find(y)
    if root_x == root_y: return

    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1
```

**Complexity with Both Optimizations:**

$$\boxed{T(m \text{ operations}, n \text{ elements}) = O(m \cdot \alpha(n))}$$

where $\alpha(n)$ is the **inverse Ackermann function**.

### 4. The Inverse Ackermann Function

The Ackermann function $A(m, n)$ grows extraordinarily fast:
- $A(1, n) = 2n$
- $A(2, n) = 2^n$
- $A(3, n) = 2^{2^{2^{\cdot^{\cdot^{\cdot}}}}}$ (tower of $n$ 2's)
- $A(4, n)$ = tower of $A(3, n)$ 2's

The **inverse Ackermann** function $\alpha(n)$ is defined as:

$$\alpha(n) = \min\{k : A(k, k) \geq n\}$$

**Key Property:** For any physically realizable $n$:

$$\boxed{\alpha(n) \leq 4}$$

Even for $n = 10^{80}$ (atoms in the universe), $\alpha(n) = 4$.

This means Union-Find is **effectively $O(n)$** in practice!

### 5. Cluster Growth Algorithm for Decoding

The Union-Find decoder uses **cluster growth** instead of matching:

**Algorithm:**

1. **Initialize:** Each syndrome defect is a singleton cluster
2. **Grow:** Alternate between:
   - Growing all odd-parity clusters by one step
   - Merging adjacent clusters (using Union-Find)
3. **Terminate:** When all clusters have even parity
4. **Peel:** Extract corrections from cluster structure

**Key Insight:** Clusters with even parity (including zero defects) are self-correcting. We only need to grow odd-parity clusters until they merge with others or reach boundaries.

**Pseudocode:**
```
ClusterDecode(defects):
    # Initialize clusters
    for each defect d:
        MakeSet(d)
        parity[Find(d)] = 1

    # Growth phase
    while any cluster has odd parity:
        for each odd-parity cluster C:
            for each boundary edge e of C:
                grow C to include e
                if e connects to another cluster C':
                    Union(C, C')
                    parity[Find(C)] ^= parity[Find(C')]
                elif e reaches code boundary:
                    parity[Find(C)] = 0  # Even parity now

    # Peel corrections from cluster structure
    return extract_corrections(clusters)
```

### 6. Complexity Analysis

**Single Decoding Round:**

Let $n = d^2$ be the number of data qubits in a distance-$d$ code.

- Worst-case defects: $O(n)$
- Cluster growth steps: $O(d) = O(\sqrt{n})$
- Union operations: $O(n)$
- Find operations: $O(n)$

Total: $O(n \cdot \alpha(n))$

**Comparison:**

| Decoder | Complexity | Distance-50 Time |
|---------|------------|------------------|
| MWPM | $O(n^3) = O(d^6)$ | ~1 second |
| Union-Find | $O(n \cdot \alpha(n)) \approx O(n)$ | ~10 $\mu$s |

The speedup is dramatic for large codes!

### 7. Threshold Analysis

Union-Find trades some accuracy for speed:

$$\boxed{p_{\text{th}}^{\text{UF}} \approx 9.9\% \quad \text{vs} \quad p_{\text{th}}^{\text{MWPM}} \approx 10.3\%}$$

The 0.4% threshold reduction comes from:
- Greedy cluster merging (vs optimal matching)
- Local decisions (vs global optimization)

For many applications, this trade-off is worthwhile given the dramatic speedup.

### 8. The Peeling Decoder

For **erasure channels** (where error locations are known), the peeling decoder is optimal:

**Erasure Model:**
- Each qubit is erased (location known) with probability $p$
- Erased qubits have random Pauli error

**Peeling Algorithm:**

1. Mark all erased qubits
2. Find a check (stabilizer) with exactly one erased qubit
3. "Peel" that qubit: determine its error from the syndrome
4. Repeat until no erasures remain or decoding fails

**Complexity:** $O(n)$ strictly linear!

**Threshold:** $p_{\text{th}}^{\text{erasure}} = 50\%$ for surface codes (much higher than depolarizing)

---

## Worked Examples

### Example 1: Union-Find Operations

**Problem:** Trace through Union-Find operations for sets {0,1,2,3,4,5} with operations: Union(0,1), Union(2,3), Union(0,2), Find(3).

**Solution:**

**Initial State:**
```
parent: [0, 1, 2, 3, 4, 5]
rank:   [0, 0, 0, 0, 0, 0]
```

**Union(0, 1):**
- Find(0) = 0, Find(1) = 1
- Both rank 0, so parent[1] = 0, rank[0] = 1
```
parent: [0, 0, 2, 3, 4, 5]
rank:   [1, 0, 0, 0, 0, 0]
```

**Union(2, 3):**
- Find(2) = 2, Find(3) = 3
- Both rank 0, so parent[3] = 2, rank[2] = 1
```
parent: [0, 0, 2, 2, 4, 5]
rank:   [1, 0, 1, 0, 0, 0]
```

**Union(0, 2):**
- Find(0) = 0, Find(2) = 2
- Both rank 1, so parent[2] = 0, rank[0] = 2
```
parent: [0, 0, 0, 2, 4, 5]
rank:   [2, 0, 1, 0, 0, 0]
```

**Find(3) with path compression:**
- parent[3] = 2, so recurse
- parent[2] = 0, so recurse
- parent[0] = 0, return 0
- Path compression: parent[2] = 0 (already), parent[3] = 0
```
parent: [0, 0, 0, 0, 4, 5]  # 3 now points directly to 0
```

**Result:** Find(3) = 0

### Example 2: Cluster Growth Decoding

**Problem:** Apply cluster growth to a distance-3 surface code with defects at (0,0) and (2,2).

**Solution:**

**Step 0: Initialize**
- Cluster A = {(0,0)}, parity = 1 (odd)
- Cluster B = {(2,2)}, parity = 1 (odd)

**Step 1: Grow odd clusters**

Cluster A grows to include neighboring edges:
- Boundary at (0,0): includes edges to (-1,0), (0,-1), (1,0), (0,1)
- Edge (1,0) is interior, others reach boundary

Cluster B grows similarly around (2,2).

**Step 2: Continue growing**

After growth, clusters expand:
- A now covers region around (0,0)-(1,1)
- B covers region around (1,1)-(2,2)

**Step 3: Clusters meet**

When growth frontiers meet (around center), Union(A, B):
- Combined cluster has parity 1 + 1 = 0 (even)
- Growth stops!

**Step 4: Extract correction**

The merged cluster contains an even-weight error chain connecting (0,0) to (2,2).
Correction: flip qubits along the growth path.

### Example 3: Inverse Ackermann Bounds

**Problem:** Show that $\alpha(n) \leq 4$ for $n \leq 10^{80}$.

**Solution:**

The Ackermann function values:
- $A(1, 1) = 2$
- $A(2, 2) = 4$
- $A(3, 3) = 2^{2^{2}} = 2^4 = 16$
- $A(4, 4) = $ tower of $A(3,4) = 65536$ twos

$A(4, 4)$ is a tower of 65536 twos:
$$A(4,4) = 2^{2^{2^{\cdot^{\cdot^{\cdot}}}}} \text{ (65536 twos)}$$

This exceeds $10^{10^{10^{\cdots}}}$, far larger than $10^{80}$.

Since $A(4, 4) > 10^{80}$, we have:
$$\alpha(10^{80}) = \min\{k : A(k,k) \geq 10^{80}\} = 4$$

For any practical $n$, $\alpha(n) \leq 4$. QED.

---

## Practice Problems

### Level A: Direct Application

**A1.** Trace Union-Find with path compression for: MakeSet(0..4), Union(0,1), Union(1,2), Union(3,4), Union(0,3), Find(4).

**A2.** For a cluster with 3 defects, what is its parity? Will it continue growing?

**A3.** Calculate the number of Union and Find operations for decoding a distance-5 code with 4 defects.

### Level B: Intermediate Analysis

**B1.** Prove that path compression ensures every node is at most 2 hops from the root after a Find operation on that node.

**B2.** Compare the memory requirements of MWPM (storing edge weights) versus Union-Find (storing parent pointers) for a distance-$d$ code.

**B3.** Design a modification to cluster growth that achieves the MWPM threshold while maintaining $O(n \log n)$ complexity.

### Level C: Advanced Problems

**C1.** Prove that the Union-Find decoder correctly identifies the homology class of the error for the toric code (errors wrapping around the torus).

**C2.** Analyze the threshold of Union-Find when cluster growth is weighted by edge reliabilities instead of uniform growth.

**C3.** Design a parallel Union-Find algorithm for GPU implementation. What is the span (parallel depth) of your algorithm?

---

## Computational Lab: Union-Find Implementation

```python
"""
Day 773 Computational Lab: Union-Find Decoder
Implementing almost-linear time decoding for topological codes

This lab builds a Union-Find decoder from scratch and compares
its performance against MWPM.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict

class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.

    Achieves O(α(n)) amortized time per operation where α is the
    inverse Ackermann function (effectively constant).
    """

    def __init__(self, n: int):
        """Initialize n singleton sets."""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        """
        Find representative of x's set with path compression.

        Returns:
            Root of the tree containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Merge sets containing x and y using union by rank.

        Returns:
            True if sets were merged, False if already same set
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank: attach smaller tree under larger
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def set_size(self, x: int) -> int:
        """Return size of set containing x."""
        return self.size[self.find(x)]


@dataclass
class Cluster:
    """Represents a cluster in the Union-Find decoder."""
    id: int
    defects: Set[Tuple[int, int]] = field(default_factory=set)
    boundary: Set[Tuple[int, int]] = field(default_factory=set)
    parity: int = 0  # 0 = even, 1 = odd

    def is_odd(self) -> bool:
        return self.parity == 1


class UnionFindDecoder:
    """
    Union-Find decoder for surface codes.

    Uses cluster growth algorithm with almost-linear complexity O(n α(n)).
    """

    def __init__(self, distance: int):
        """
        Initialize decoder for distance-d surface code.

        Args:
            distance: Code distance
        """
        self.d = distance
        self.reset()

    def reset(self):
        """Reset decoder state."""
        # Node indexing: position (x, y) -> index x * d + y
        self.n_nodes = self.d * self.d
        self.uf = UnionFind(self.n_nodes + 4)  # +4 for boundary nodes

        # Cluster tracking
        self.cluster_parity = {}  # root -> parity
        self.cluster_boundary = {}  # root -> set of boundary positions

        # Boundary node indices
        self.BOUNDARY_LEFT = self.n_nodes
        self.BOUNDARY_RIGHT = self.n_nodes + 1
        self.BOUNDARY_TOP = self.n_nodes + 2
        self.BOUNDARY_BOTTOM = self.n_nodes + 3

    def pos_to_idx(self, x: int, y: int) -> int:
        """Convert (x, y) position to node index."""
        return x * self.d + y

    def idx_to_pos(self, idx: int) -> Tuple[int, int]:
        """Convert node index to (x, y) position."""
        return idx // self.d, idx % self.d

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighbors of position (x, y)."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.d and 0 <= ny < self.d:
                neighbors.append((nx, ny))
        return neighbors

    def is_boundary(self, x: int, y: int) -> Optional[int]:
        """Check if position is at boundary, return boundary node if so."""
        if x < 0:
            return self.BOUNDARY_LEFT
        if x >= self.d:
            return self.BOUNDARY_RIGHT
        if y < 0:
            return self.BOUNDARY_BOTTOM
        if y >= self.d:
            return self.BOUNDARY_TOP
        return None

    def decode(self, defects: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Decode syndrome using cluster growth algorithm.

        Args:
            defects: List of defect positions (x, y)

        Returns:
            List of qubit positions to correct
        """
        self.reset()

        if len(defects) == 0:
            return []

        # Initialize clusters for each defect
        for (x, y) in defects:
            idx = self.pos_to_idx(x, y)
            root = self.uf.find(idx)
            self.cluster_parity[root] = 1  # Odd parity
            self.cluster_boundary[root] = {(x, y)}

        # Growth phase: grow until all clusters have even parity
        max_iterations = 2 * self.d  # Maximum growth steps
        corrections = []

        for iteration in range(max_iterations):
            # Find all odd-parity clusters
            odd_clusters = []
            for root, parity in list(self.cluster_parity.items()):
                actual_root = self.uf.find(root)
                if actual_root in self.cluster_parity:
                    if self.cluster_parity[actual_root] == 1:
                        odd_clusters.append(actual_root)

            if not odd_clusters:
                break  # All clusters even, done!

            # Grow each odd cluster by one step
            for root in odd_clusters:
                self._grow_cluster(root, corrections)

        return corrections

    def _grow_cluster(self, root: int, corrections: List[Tuple[int, int]]):
        """Grow a cluster by one step in all directions."""
        if root not in self.cluster_boundary:
            return

        current_boundary = list(self.cluster_boundary[root])
        new_boundary = set()

        for (x, y) in current_boundary:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                # Check if reaching code boundary
                boundary_node = self.is_boundary(nx, ny)
                if boundary_node is not None:
                    # Merge with boundary (makes parity even)
                    if self.uf.union(root, boundary_node):
                        new_root = self.uf.find(root)
                        # Transfer parity (boundary has even parity)
                        old_parity = self.cluster_parity.get(root, 0)
                        self.cluster_parity[new_root] = 0  # Even now
                        corrections.append((x, y))  # Mark correction at boundary
                    continue

                # Check if neighbor is in another cluster
                neighbor_idx = self.pos_to_idx(nx, ny)
                neighbor_root = self.uf.find(neighbor_idx)

                if neighbor_root != self.uf.find(root):
                    # Merge clusters
                    old_parity_1 = self.cluster_parity.get(self.uf.find(root), 0)
                    old_parity_2 = self.cluster_parity.get(neighbor_root, 0)

                    if self.uf.union(root, neighbor_idx):
                        new_root = self.uf.find(root)
                        # XOR parities
                        self.cluster_parity[new_root] = old_parity_1 ^ old_parity_2

                        # Merge boundaries
                        b1 = self.cluster_boundary.get(root, set())
                        b2 = self.cluster_boundary.get(neighbor_root, set())
                        self.cluster_boundary[new_root] = b1 | b2

                        # Record correction along growth edge
                        corrections.append(((x + nx) // 2, (y + ny) // 2))
                else:
                    # Same cluster, add to boundary
                    new_boundary.add((nx, ny))

        # Update cluster boundary
        new_root = self.uf.find(root)
        if new_root in self.cluster_boundary:
            self.cluster_boundary[new_root] = (
                self.cluster_boundary[new_root] | new_boundary
            )
        else:
            self.cluster_boundary[new_root] = new_boundary


class PeelingDecoder:
    """
    Peeling decoder for erasure channels.

    Achieves O(n) complexity for erasure errors where locations are known.
    """

    def __init__(self, distance: int):
        self.d = distance

    def decode(self, erasures: List[Tuple[int, int]],
               syndrome: np.ndarray) -> List[Tuple[int, int, str]]:
        """
        Decode erasure errors using peeling.

        Args:
            erasures: List of erased qubit positions
            syndrome: Binary syndrome array

        Returns:
            List of (x, y, pauli) corrections
        """
        corrections = []
        remaining_erasures = set(erasures)
        remaining_syndrome = syndrome.copy()

        while remaining_erasures:
            # Find a check with exactly one remaining erasure
            peeled = False

            for (x, y) in list(remaining_erasures):
                # Find checks involving this qubit
                checks = self._get_checks(x, y)

                for check_idx, check_qubits in checks:
                    # Count erasures in this check
                    erasures_in_check = sum(
                        1 for q in check_qubits if q in remaining_erasures
                    )

                    if erasures_in_check == 1:
                        # Can peel this qubit!
                        if remaining_syndrome[check_idx]:
                            corrections.append((x, y, 'X'))
                            # Update syndrome
                            for c_idx, _ in self._get_checks(x, y):
                                remaining_syndrome[c_idx] ^= 1

                        remaining_erasures.remove((x, y))
                        peeled = True
                        break

                if peeled:
                    break

            if not peeled:
                # No more peeling possible - decoding fails
                break

        return corrections

    def _get_checks(self, x: int, y: int) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """Get stabilizer checks involving qubit at (x, y)."""
        # Simplified: return adjacent plaquettes
        checks = []
        check_idx = 0
        for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
            px, py = x + dx, y + dy
            if 0 <= px < self.d - 1 and 0 <= py < self.d - 1:
                qubits = [(px, py), (px+1, py), (px, py+1), (px+1, py+1)]
                checks.append((check_idx, qubits))
            check_idx += 1
        return checks


def benchmark_complexity():
    """
    Benchmark Union-Find decoder complexity scaling.
    """
    print("=" * 60)
    print("Union-Find Complexity Benchmark")
    print("=" * 60)

    distances = [5, 9, 13, 17, 21, 25]
    times_uf = []
    times_per_defect = []
    n_trials = 100

    for d in distances:
        decoder = UnionFindDecoder(d)
        total_time = 0
        total_defects = 0

        for _ in range(n_trials):
            # Generate random defects (average d defects)
            n_defects = np.random.poisson(d)
            defects = []
            for _ in range(n_defects):
                x = np.random.randint(0, d)
                y = np.random.randint(0, d)
                defects.append((x, y))

            start = time.time()
            decoder.decode(defects)
            elapsed = time.time() - start

            total_time += elapsed
            total_defects += len(defects)

        avg_time = total_time / n_trials
        avg_defects = total_defects / n_trials
        time_per_defect = total_time / max(total_defects, 1)

        times_uf.append(avg_time)
        times_per_defect.append(time_per_defect)

        print(f"d={d:2d}: avg time = {avg_time*1000:.3f} ms, "
              f"avg defects = {avg_defects:.1f}, "
              f"time/defect = {time_per_defect*1e6:.2f} μs")

    # Plot complexity scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Total time vs distance
    n_qubits = [d**2 for d in distances]
    ax1.loglog(n_qubits, [t*1000 for t in times_uf], 'bo-', linewidth=2, markersize=8)

    # Fit power law
    log_n = np.log(n_qubits)
    log_t = np.log(times_uf)
    slope, intercept = np.polyfit(log_n, log_t, 1)

    fit_t = np.exp(intercept) * np.array(n_qubits) ** slope
    ax1.loglog(n_qubits, [t*1000 for t in fit_t], 'r--',
               label=f'Fit: O(n^{slope:.2f})', linewidth=2)

    ax1.set_xlabel('Number of Qubits (n = d²)', fontsize=12)
    ax1.set_ylabel('Decode Time (ms)', fontsize=12)
    ax1.set_title('Union-Find Decoder Scaling', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Time per operation
    ax2.plot(distances, [t*1e6 for t in times_per_defect], 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=np.mean([t*1e6 for t in times_per_defect]), color='r', linestyle='--',
                label=f'Mean: {np.mean([t*1e6 for t in times_per_defect]):.2f} μs')

    ax2.set_xlabel('Code Distance', fontsize=12)
    ax2.set_ylabel('Time per Defect (μs)', fontsize=12)
    ax2.set_title('Amortized Time per Operation (≈ O(α(n)) ≈ O(1))', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('union_find_complexity.png', dpi=150)
    plt.show()

    print(f"\nFitted complexity: O(n^{slope:.2f})")
    print("Expected: O(n) with small constant (α(n) ≈ 4)")


def threshold_comparison():
    """
    Compare Union-Find threshold with theoretical values.
    """
    print("\n" + "=" * 60)
    print("Union-Find Threshold Estimation")
    print("=" * 60)

    distances = [5, 7, 9]
    error_rates = np.linspace(0.06, 0.14, 9)
    n_trials = 500

    results = {d: [] for d in distances}

    for d in distances:
        print(f"\nSimulating distance-{d} code...")
        decoder = UnionFindDecoder(d)

        for p in error_rates:
            failures = 0

            for _ in range(n_trials):
                # Generate random X errors
                errors = np.random.random(d * d) < p
                error_positions = [
                    (i // d, i % d) for i in range(d * d) if errors[i]
                ]

                # Compute syndrome (simplified)
                defects = compute_simple_syndrome(error_positions, d)

                # Decode
                corrections = decoder.decode(defects)

                # Check logical error (simplified)
                if check_logical_error_simple(error_positions, corrections, d):
                    failures += 1

            logical_error_rate = failures / n_trials
            results[d].append(logical_error_rate)
            print(f"  p={p:.2f}: logical error rate = {logical_error_rate:.4f}")

    # Plot
    plt.figure(figsize=(10, 7))

    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    for i, d in enumerate(distances):
        plt.semilogy(error_rates * 100, results[d],
                    f'{colors[i]}{markers[i]}-',
                    label=f'd = {d}', linewidth=2, markersize=8)

    plt.axvline(x=9.9, color='orange', linestyle='--',
                label='UF threshold (~9.9%)')
    plt.axvline(x=10.3, color='purple', linestyle=':',
                label='MWPM threshold (~10.3%)')

    plt.xlabel('Physical Error Rate (%)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('Union-Find Decoder Threshold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([6, 14])

    plt.tight_layout()
    plt.savefig('union_find_threshold.png', dpi=150)
    plt.show()


def compute_simple_syndrome(errors: List[Tuple[int, int]], d: int) -> List[Tuple[int, int]]:
    """Simplified syndrome computation."""
    defect_counts = defaultdict(int)

    for (x, y) in errors:
        # Each error affects 4 adjacent stabilizers
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            sx, sy = x + dx, y + dy
            if 0 <= sx < d and 0 <= sy < d:
                defect_counts[(sx, sy)] += 1

    # Defects where count is odd
    return [(x, y) for (x, y), count in defect_counts.items() if count % 2 == 1]


def check_logical_error_simple(errors: List[Tuple[int, int]],
                               corrections: List[Tuple[int, int]],
                               d: int) -> bool:
    """Simplified logical error check."""
    # Count net errors in each column
    error_set = set(errors)
    correction_set = set(corrections)

    # Net errors = errors XOR corrections
    net_errors = error_set.symmetric_difference(correction_set)

    # Check if any row has odd parity (logical X error)
    for row in range(d):
        row_errors = sum(1 for (x, y) in net_errors if x == row)
        if row_errors % 2 == 1:
            return True

    return False


def visualize_cluster_growth():
    """
    Visualize the cluster growth algorithm.
    """
    print("\n" + "=" * 60)
    print("Cluster Growth Visualization")
    print("=" * 60)

    d = 7
    defects = [(1, 1), (5, 5), (2, 4)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Manual cluster growth visualization
    growth_stages = [
        # Stage 0: Initial defects
        {(1, 1): 'A', (5, 5): 'B', (2, 4): 'C'},
        # Stage 1: First growth
        {(0, 1): 'A', (1, 0): 'A', (1, 1): 'A', (1, 2): 'A', (2, 1): 'A',
         (4, 5): 'B', (5, 4): 'B', (5, 5): 'B', (5, 6): 'B', (6, 5): 'B',
         (1, 4): 'C', (2, 3): 'C', (2, 4): 'C', (2, 5): 'C', (3, 4): 'C'},
        # Stage 2: More growth
        {(0, 0): 'A', (0, 1): 'A', (0, 2): 'A', (1, 0): 'A', (1, 1): 'A',
         (1, 2): 'A', (1, 3): 'A', (2, 0): 'A', (2, 1): 'A', (2, 2): 'A',
         (3, 5): 'B', (4, 4): 'B', (4, 5): 'B', (4, 6): 'B', (5, 3): 'B',
         (5, 4): 'B', (5, 5): 'B', (5, 6): 'B', (6, 4): 'B', (6, 5): 'B', (6, 6): 'B',
         (1, 3): 'C', (1, 4): 'C', (1, 5): 'C', (2, 3): 'C', (2, 4): 'C',
         (2, 5): 'C', (3, 3): 'C', (3, 4): 'C', (3, 5): 'C'},
        # Stage 3: Clusters merge
        {(x, y): 'AB' for x in range(d) for y in range(d)},  # All merged
    ]

    colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'AB': 'purple'}

    for idx, (ax, stage) in enumerate(zip(axes.flat, growth_stages[:6])):
        ax.set_xlim(-0.5, d - 0.5)
        ax.set_ylim(-0.5, d - 0.5)

        # Draw grid
        for i in range(d):
            for j in range(d):
                ax.plot(i, j, 'ko', markersize=5, alpha=0.3)

        # Draw cluster regions
        for (x, y), cluster in stage.items():
            if 0 <= x < d and 0 <= y < d:
                color = colors.get(cluster, 'gray')
                ax.add_patch(plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8,
                                          facecolor=color, alpha=0.5))

        # Mark original defects
        for (x, y) in defects:
            ax.plot(x, y, 'k*', markersize=15)

        ax.set_title(f'Growth Stage {idx}', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for ax in axes.flat[len(growth_stages):]:
        ax.axis('off')

    plt.suptitle('Union-Find Cluster Growth Algorithm', fontsize=14)
    plt.tight_layout()
    plt.savefig('cluster_growth.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    benchmark_complexity()
    threshold_comparison()
    visualize_cluster_growth()

    print("\n" + "=" * 60)
    print("Lab Complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Union-Find Complexity | $$O(m \cdot \alpha(n))$$ for $m$ operations |
| Inverse Ackermann | $$\alpha(n) = \min\\{k : A(k,k) \geq n\\} \leq 4$$ |
| Practical Complexity | $$O(n)$$ for $n$ qubits |
| UF Threshold | $$p_{\text{th}}^{\text{UF}} \approx 9.9\%$$ |
| Threshold Gap | $$\Delta p = p_{\text{th}}^{\text{MWPM}} - p_{\text{th}}^{\text{UF}} \approx 0.4\%$$ |
| Erasure Threshold | $$p_{\text{th}}^{\text{erasure}} = 50\%$$ |

### Key Takeaways

1. **Union-Find achieves almost-linear time**: $O(n \cdot \alpha(n)) \approx O(n)$ in practice
2. **Path compression is crucial**: Flattens trees during Find operations
3. **Cluster growth replaces matching**: Grow odd-parity clusters until they merge
4. **Threshold trade-off exists**: ~0.4% lower threshold than MWPM
5. **Peeling decoder for erasures**: Strictly $O(n)$ when error locations known
6. **Practical for large codes**: Enables real-time decoding at scale

---

## Daily Checklist

- [ ] Implemented Union-Find with path compression and union by rank
- [ ] Understood the inverse Ackermann function and why $\alpha(n) \leq 4$
- [ ] Traced through cluster growth algorithm examples
- [ ] Compared Union-Find vs MWPM thresholds
- [ ] Implemented peeling decoder for erasures
- [ ] Benchmarked complexity scaling
- [ ] Completed practice problems (at least Level A and B)

---

## Preview: Day 774

Tomorrow we explore **Neural Network Decoders**, a machine learning approach to quantum error correction. We'll learn how CNNs can recognize syndrome patterns, how to generate training data, and the challenges of real-time inference in quantum systems.

Key questions for tomorrow:
- How do we frame decoding as a classification problem?
- What neural architectures work best for syndrome patterns?
- Can neural decoders match or exceed MWPM thresholds?

---

*Day 773 of 2184 | Week 111 | Month 28 | Year 2: Advanced Quantum Science*
