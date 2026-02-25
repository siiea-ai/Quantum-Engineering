# Day 997: Semester 2A Review - Surface Codes

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Core Review: Surface Code Architecture & Operations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Qualifying Exam Problem Practice |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Synthesis and Implementation Analysis |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 997, you will be able to:

1. **Construct** rotated surface codes with explicit stabilizer assignments
2. **Design** lattice surgery protocols for logical CNOT operations
3. **Analyze** decoding algorithms and their complexity trade-offs
4. **Evaluate** real-time decoding constraints for practical implementations
5. **Interpret** experimental results from Google Willow and IBM systems
6. **Calculate** logical error rates from physical error rates

---

## Core Review Content

### 1. Surface Code Geometry

#### From Toric to Planar

The surface code is the toric code with boundaries:

| Feature | Toric Code | Surface Code |
|---------|------------|--------------|
| Topology | Torus (periodic) | Plane (open) |
| Logical qubits | 2 | 1 |
| Boundaries | None | Rough and smooth |
| Parameters | [[2L², 2, L]] | [[d², 1, d]] |

#### Rotated Surface Code

The standard implementation rotates the lattice 45°:

```
    ●───○───●
    │ Z │ X │
    ○───●───○
    │ X │ Z │
    ●───○───●

● = data qubit
○ = ancilla qubit
X = X-type stabilizer (detect Z errors)
Z = Z-type stabilizer (detect X errors)
```

**Parameters for distance $d$:**
$$\boxed{[[d^2, 1, d]] \text{ or } [[(d^2+1)/2 \text{ data} + (d^2-1)/2 \text{ ancilla}, 1, d]]}$$

#### Boundary Types

**Rough boundary:** Z stabilizers at edge, X logical string terminates
**Smooth boundary:** X stabilizers at edge, Z logical string terminates

```
Rough-Rough (horizontal boundaries): Z_L runs horizontally
Smooth-Smooth (vertical boundaries): X_L runs vertically
```

---

### 2. Stabilizer Structure

#### Distance-3 Rotated Surface Code

**Data qubits:** 9 (positions 0-8)
**Ancilla qubits:** 8 (4 X-type, 4 Z-type)

**X-type stabilizers (weight-4 interior, weight-2 boundary):**
$$X_0X_1X_3X_4, \quad X_1X_2X_4X_5, \quad X_3X_4X_6X_7, \quad X_4X_5X_7X_8$$

**Z-type stabilizers:**
$$Z_0Z_1, \quad Z_0Z_3, \quad Z_2Z_5, \quad Z_6Z_7, \quad Z_5Z_8, \quad Z_7Z_8$$

Wait, let me provide the correct structure:

**Interior X stabilizers:** $X_iX_jX_kX_l$ (weight 4)
**Boundary X stabilizers:** $X_iX_j$ (weight 2)
**Interior Z stabilizers:** $Z_iZ_jZ_kZ_l$ (weight 4)
**Boundary Z stabilizers:** $Z_iZ_j$ (weight 2)

#### Logical Operators

$$\bar{X} = X^{\otimes d} \text{ (horizontal chain)}$$
$$\bar{Z} = Z^{\otimes d} \text{ (vertical chain)}$$

Minimum weight: $d$ (the code distance)

---

### 3. Error Correction Cycle

#### Syndrome Extraction Circuit

```
1. Initialize ancillas to |0⟩ (X-type) or |+⟩ (Z-type)
2. Apply entangling gates between ancilla and data qubits
3. Measure ancillas
4. Repeat for multiple rounds (typically d rounds)
5. Decode and apply corrections
```

**For X-stabilizer measurement:**
```
     |0⟩ ─── H ─── CNOT ─── CNOT ─── CNOT ─── CNOT ─── H ─── Measure
               │         │         │         │
          data₁    data₂    data₃    data₄
```

#### Syndrome Volume

For $d$ rounds of measurements:
- Syndrome forms a 3D volume: $d \times d \times d$
- Errors create defects in this volume
- Decoding finds minimum-weight matching

---

### 4. Decoding Algorithms

#### Minimum Weight Perfect Matching (MWPM)

**Algorithm:**
1. Construct graph of syndrome defects
2. Weight edges by distance (Manhattan or Euclidean)
3. Find minimum weight perfect matching
4. Matching determines error chains

**Complexity:** $O(n^3)$ for $n$ defects using Blossom algorithm

**Threshold:** ~1% for depolarizing noise

#### Union-Find Decoder

**Algorithm:**
1. Grow clusters from each defect
2. Merge when clusters meet
3. Connect paired defects with error chains

**Complexity:** $O(n \cdot \alpha(n))$ (nearly linear)

**Threshold:** ~0.8% (slightly lower than MWPM)

**Advantage:** Much faster, suitable for real-time decoding

#### Decoder Comparison

| Decoder | Threshold | Complexity | Real-time? |
|---------|-----------|------------|------------|
| MWPM | ~1.0% | $O(n^3)$ | No |
| Union-Find | ~0.8% | $O(n\alpha(n))$ | Yes |
| Neural | ~1.0% | $O(n)$ | Yes |
| Tensor Network | ~1.1% | Exponential | No |

---

### 5. Lattice Surgery

#### Concept

Instead of transversal gates, perform logical operations by:
1. Merging code patches (creating joint stabilizers)
2. Splitting code patches (creating separate logical qubits)

#### Logical CNOT via Lattice Surgery

**Step 1: Prepare ancilla patch in $|+_L\rangle$**

**Step 2: Merge control with ancilla (ZZ measurement)**
- Measure $\bar{Z}_C \bar{Z}_A$
- Projects to $|0_C 0_A\rangle + |1_C 1_A\rangle$ or orthogonal

**Step 3: Split, then merge ancilla with target (XX measurement)**
- Measure $\bar{X}_A \bar{X}_T$
- Implements controlled operation

**Step 4: Measure and correct ancilla**

**Result:** Logical CNOT with $O(d)$ time steps

```
Control ═══╦═══════════
           ║ ZZ merge
Ancilla ═══╩═══╦═══════
               ║ XX merge
Target ════════╩═══════
```

#### Resource Cost

| Operation | Time (code cycles) | Space (patches) |
|-----------|-------------------|-----------------|
| Idle | 1 | 1 |
| Logical X, Z | 1 | 1 |
| Logical H | $O(d)$ | 1 |
| Logical CNOT | $O(d)$ | 3 |
| Logical T | $O(d)$ | 1 + factory |

---

### 6. Threshold and Scaling

#### Threshold Theorem for Surface Codes

If physical error rate $p < p_{th}$:

$$\boxed{p_L \approx 0.03 \left(\frac{p}{p_{th}}\right)^{(d+1)/2}}$$

**Typical threshold values:**
- Circuit-level depolarizing: $p_{th} \approx 0.6\%$
- Phenomenological: $p_{th} \approx 3\%$
- Code capacity: $p_{th} \approx 11\%$

#### Error Suppression Factor

$$\Lambda = \frac{p_L(d)}{p_L(d+2)}$$

For $p < p_{th}$: $\Lambda > 1$ (errors decrease with larger codes)
For $p > p_{th}$: $\Lambda < 1$ (errors increase - bad!)

**Google Willow (2024):** $\Lambda = 2.14 \pm 0.02$

---

### 7. Experimental Implementations

#### Google Sycamore/Willow (2024)

| Metric | Value |
|--------|-------|
| Qubits | 105 (Willow) |
| T1 | ~20 μs |
| T2 | ~30 μs |
| 1Q gate error | 0.1% |
| 2Q gate error | 0.5% |
| Readout error | 1% |
| Surface code distance | 3, 5, 7 |
| Below threshold? | **Yes** |
| Error suppression Λ | 2.14 |

**Key achievement:** First demonstration of below-threshold operation

#### IBM Eagle/Condor

| Metric | Value |
|--------|-------|
| Qubits | 1121 (Condor) |
| Topology | Heavy-hex |
| T1 | ~300 μs |
| T2 | ~200 μs |
| 2Q gate error | 1-2% |
| Error correction | Demonstrated |

#### Comparison

| Aspect | Google | IBM |
|--------|--------|-----|
| Qubit count | Lower | Higher |
| Connectivity | Grid | Heavy-hex |
| Gate speed | Faster | Slower |
| Coherence | Lower | Higher |
| QEC demo | Surface code | Various |

---

### 8. Real-Time Decoding Requirements

#### Timing Budget

For superconducting qubits:
- Syndrome extraction: ~1 μs
- Classical processing must complete before next round
- Decoder latency must be < 1 μs

#### Hardware Decoders

**FPGA-based:**
- Union-Find on FPGA: <100 ns latency
- Parallelizable across syndrome chunks

**ASIC-based:**
- Custom decoder chips
- Even lower latency

**Challenges:**
- Correlating errors across rounds
- Handling measurement errors
- Maintaining throughput

---

## Concept Map: Surface Codes

```
Toric Code (periodic)
       │
       ▼
Surface Code (boundaries)
       │
       ├──► Rotated geometry [[d², 1, d]]
       │
       ├──► Stabilizers: X-type, Z-type
       │          │
       │          ▼
       │    Syndrome extraction
       │          │
       │          ▼
       │    Decoding (MWPM, Union-Find)
       │
       ├──► Logical operators: X̄, Z̄
       │
       ├──► Lattice surgery
       │          │
       │          ├──► Merge/Split
       │          │
       │          └──► Logical gates
       │
       └──► Experiments
                 │
                 ├──► Google Willow (Λ = 2.14)
                 │
                 └──► IBM Condor
```

---

## Qualifying Exam Practice Problems

### Problem 1: Surface Code Parameters (20 points)

**Question:** For a distance-5 rotated surface code:

(a) How many data qubits are needed?
(b) How many ancilla qubits?
(c) How many X-type and Z-type stabilizers?
(d) What is the weight of boundary vs. interior stabilizers?

**Solution:**

**(a) Data qubits:**
For rotated surface code distance $d$: $d^2 = 25$ data qubits

**(b) Ancilla qubits:**
- X-type ancillas: $(d^2-1)/2 = 12$
- Z-type ancillas: $(d^2-1)/2 = 12$
- **Total: 24 ancilla qubits**

Alternatively: $d^2 - 1 = 24$ total ancillas

**(c) Stabilizers:**
- X-type stabilizers: 12
- Z-type stabilizers: 12
- **Total: 24 independent stabilizers**

Check: $k = n - m = 25 - 24 = 1$ logical qubit ✓

**(d) Stabilizer weights:**
- **Interior stabilizers:** Weight 4 (4-body)
- **Boundary stabilizers:** Weight 2 (2-body)

For $d=5$:
- Interior X-stabilizers: 9 (weight 4)
- Boundary X-stabilizers: 3 (weight 2)
- Similarly for Z-type

---

### Problem 2: Logical Error Rate Calculation (25 points)

**Question:** A surface code has threshold $p_{th} = 0.6\%$ and physical error rate $p = 0.3\%$.

(a) Calculate the logical error rate for $d = 3, 5, 7, 9$
(b) What distance is needed for $p_L < 10^{-10}$?
(c) How many physical qubits does this require?

**Solution:**

Using $p_L \approx 0.03 (p/p_{th})^{(d+1)/2}$

**(a)** With $p/p_{th} = 0.3/0.6 = 0.5$:

| d | $(d+1)/2$ | $(0.5)^{(d+1)/2}$ | $p_L$ |
|---|-----------|-------------------|-------|
| 3 | 2 | 0.25 | 0.0075 |
| 5 | 3 | 0.125 | 0.00375 |
| 7 | 4 | 0.0625 | 0.00188 |
| 9 | 5 | 0.03125 | 0.00094 |

**(b)** Need: $0.03 \times 0.5^{(d+1)/2} < 10^{-10}$

$$0.5^{(d+1)/2} < 3.33 \times 10^{-9}$$
$$(d+1)/2 > \log_{0.5}(3.33 \times 10^{-9}) = \frac{\ln(3.33 \times 10^{-9})}{\ln(0.5)} \approx 28.2$$
$$d > 55.4$$

**Answer: d ≥ 57**

**(c)** Physical qubits:
- Data: $d^2 = 57^2 = 3249$
- Ancilla: $d^2 - 1 = 3248$
- **Total: ~6500 physical qubits** for one logical qubit

---

### Problem 3: Lattice Surgery Protocol (25 points)

**Question:** Design a lattice surgery protocol for logical CNOT using three surface code patches.

(a) Draw the patch arrangement
(b) Describe each step of the protocol
(c) How many code cycles does the operation take?
(d) What is the dominant error mechanism?

**Solution:**

**(a) Patch arrangement:**
```
┌─────────┐     ┌─────────┐
│ Control │     │ Target  │
│    C    │     │    T    │
└─────────┘     └─────────┘

     ┌─────────┐
     │ Ancilla │
     │    A    │
     └─────────┘
```

**(b) Protocol steps:**

**Step 1: Initialize ancilla** (d cycles)
- Prepare patch A in $|+_L\rangle$ state
- Verify initialization with stabilizer measurements

**Step 2: ZZ merge (C-A)** (d cycles)
- Extend stabilizers to connect C and A
- Measure joint $\bar{Z}_C \bar{Z}_A$ stabilizers
- This correlates C and A

**Step 3: Split C-A** (d cycles)
- Remove connecting stabilizers
- Restore independent patches
- Apply corrections based on measurement outcomes

**Step 4: XX merge (A-T)** (d cycles)
- Extend stabilizers to connect A and T
- Measure joint $\bar{X}_A \bar{X}_T$ stabilizers
- This completes the CNOT teleportation

**Step 5: Split and correct** (d cycles)
- Remove connecting stabilizers
- Measure ancilla in X basis
- Apply Pauli corrections to C and T

**(c) Total time: ~5d code cycles**

For $d = 7$ with 1 μs per cycle: ~35 μs

**(d) Dominant error mechanism:**
- **Merge/split boundaries:** Most errors occur at boundaries during merge/split
- **Timelike errors:** Measurement errors during extended merge operations
- **Decoder correlation:** Errors that span the merge must be decoded consistently

---

### Problem 4: Decoder Analysis (20 points)

**Question:** Compare MWPM and Union-Find decoders:

(a) What is the syndrome graph for MWPM?
(b) How does Union-Find achieve near-linear time?
(c) For 1000 syndrome defects, estimate runtime ratio
(d) When would you choose each decoder?

**Solution:**

**(a) MWPM Syndrome Graph:**
- **Nodes:** Syndrome defects (violated stabilizers)
- **Edges:** Between all pairs of defects
- **Weights:** Distance metric (e.g., Manhattan distance in 3D syndrome space)
- **Boundary nodes:** Virtual nodes at boundaries

For $n$ defects: $O(n^2)$ edges

**(b) Union-Find Near-Linear Time:**
- Uses **disjoint-set forest** data structure
- **Union by rank:** Attach smaller tree to larger
- **Path compression:** Flatten tree during Find operations
- Combined: $O(n \cdot \alpha(n))$ where $\alpha$ is inverse Ackermann (nearly constant)

Growth process:
1. Each defect starts as singleton cluster
2. Clusters grow in parallel
3. Merge when clusters touch
4. Stop when all defects paired

**(c) Runtime Estimation:**

For $n = 1000$ defects:
- MWPM: $O(n^3) = O(10^9)$ operations
- Union-Find: $O(n \cdot \alpha(n)) \approx O(4000)$ operations

**Ratio: ~250,000x faster for Union-Find**

**(d) Decoder Selection:**

| Scenario | Preferred Decoder |
|----------|------------------|
| Offline analysis | MWPM (best threshold) |
| Real-time (superconducting) | Union-Find (speed) |
| Low error rate | Either (few defects) |
| High error rate | MWPM (better accuracy) |
| Research | MWPM or neural |
| Production | Union-Find or ASIC |

---

### Problem 5: Experimental Analysis (10 points)

**Question:** Google's 2024 Willow experiment measured error suppression factor $\Lambda = 2.14$ when going from distance-3 to distance-5.

(a) What does $\Lambda > 1$ indicate?
(b) Estimate the logical error rate ratio $p_L(d=3)/p_L(d=5)$
(c) If $p_L(d=3) = 3\%$, what is $p_L(d=5)$?

**Solution:**

**(a) $\Lambda > 1$ indicates:**
- Operating **below threshold**
- Larger codes have **lower** logical error rates
- Error correction is **working as intended**
- Quantum advantage in error correction demonstrated

**(b) Error rate ratio:**
By definition: $\Lambda = p_L(d)/p_L(d+2)$

So: $p_L(d=3)/p_L(d=5) = \Lambda = 2.14$

**(c) Logical error rate for d=5:**
$$p_L(d=5) = \frac{p_L(d=3)}{\Lambda} = \frac{0.03}{2.14} \approx 1.4\%$$

**Note:** This is still above fault-tolerance requirements (~0.1%), but demonstrates the path forward.

---

## Computational Review

```python
"""
Day 997 Computational Review: Surface Codes
Semester 2A Review - Week 143
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

# =============================================================================
# Part 1: Rotated Surface Code Structure
# =============================================================================

print("=" * 70)
print("Part 1: Rotated Surface Code Structure")
print("=" * 70)

def create_rotated_surface_code(d):
    """
    Create the structure of a rotated surface code.

    Returns:
        data_qubits: list of (x, y) positions
        x_stabilizers: list of lists of data qubit indices
        z_stabilizers: list of lists of data qubit indices
    """
    data_qubits = []
    x_stabs = []
    z_stabs = []

    # Data qubits on a d x d grid
    for i in range(d):
        for j in range(d):
            data_qubits.append((i, j))

    # X stabilizers (detect Z errors)
    # At positions (i+0.5, j+0.5) where i+j is even
    for i in range(d-1):
        for j in range(d-1):
            if (i + j) % 2 == 0:
                stab = []
                for di, dj in [(0,0), (0,1), (1,0), (1,1)]:
                    if 0 <= i+di < d and 0 <= j+dj < d:
                        stab.append(i*d + j + di*d + dj)
                if len(stab) > 0:
                    x_stabs.append(stab)

    # Z stabilizers (detect X errors)
    # At positions (i+0.5, j+0.5) where i+j is odd
    for i in range(d-1):
        for j in range(d-1):
            if (i + j) % 2 == 1:
                stab = []
                for di, dj in [(0,0), (0,1), (1,0), (1,1)]:
                    if 0 <= i+di < d and 0 <= j+dj < d:
                        stab.append(i*d + j + di*d + dj)
                if len(stab) > 0:
                    z_stabs.append(stab)

    # Add boundary stabilizers
    # (simplified - full implementation needs careful boundary handling)

    return data_qubits, x_stabs, z_stabs

d = 3
data, x_stabs, z_stabs = create_rotated_surface_code(d)

print(f"\nDistance-{d} Rotated Surface Code:")
print(f"  Data qubits: {len(data)}")
print(f"  X stabilizers: {len(x_stabs)}")
print(f"  Z stabilizers: {len(z_stabs)}")
print(f"  Logical qubits: {len(data) - len(x_stabs) - len(z_stabs)}")

# =============================================================================
# Part 2: Syndrome Computation
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Syndrome Computation")
print("=" * 70)

def compute_syndrome(error_pattern, stabilizers, d):
    """
    Compute syndrome for a given error pattern.

    Args:
        error_pattern: list of qubit indices with errors
        stabilizers: list of stabilizer supports
        d: code distance

    Returns:
        syndrome: binary vector
    """
    syndrome = []
    for stab in stabilizers:
        # Count overlap with error pattern
        overlap = len(set(error_pattern) & set(stab))
        syndrome.append(overlap % 2)
    return syndrome

# Example: single error at position (1, 1)
error_pos = [1 * d + 1]  # Center qubit
z_syndrome = compute_syndrome(error_pos, x_stabs, d)  # X stabs detect Z errors
x_syndrome = compute_syndrome(error_pos, z_stabs, d)  # Z stabs detect X errors

print(f"\nSingle Z error at center qubit:")
print(f"  X-stabilizer syndrome: {z_syndrome}")
print(f"  Z-stabilizer syndrome: {x_syndrome}")

# Two-error case
error_pos_2 = [0, 2]  # Two qubits
z_syndrome_2 = compute_syndrome(error_pos_2, x_stabs, d)
print(f"\nTwo Z errors at qubits 0 and 2:")
print(f"  X-stabilizer syndrome: {z_syndrome_2}")

# =============================================================================
# Part 3: Union-Find Decoder (Simplified)
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Union-Find Decoder")
print("=" * 70)

class UnionFind:
    """Simple Union-Find data structure."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

def union_find_decode(syndrome_positions, grid_size):
    """
    Simplified Union-Find decoder for 2D syndrome.

    Args:
        syndrome_positions: list of (x, y) defect positions
        grid_size: size of syndrome grid

    Returns:
        pairs: list of paired defects
    """
    if len(syndrome_positions) == 0:
        return []

    n = len(syndrome_positions)
    uf = UnionFind(n + 4)  # Extra nodes for boundaries

    # Grow clusters until all paired
    pairs = []

    # Simple pairing: connect nearest defects
    remaining = set(range(n))

    while len(remaining) > 1:
        # Find closest pair
        min_dist = float('inf')
        best_pair = None

        remaining_list = list(remaining)
        for i in range(len(remaining_list)):
            for j in range(i + 1, len(remaining_list)):
                idx_i, idx_j = remaining_list[i], remaining_list[j]
                pos_i = syndrome_positions[idx_i]
                pos_j = syndrome_positions[idx_j]
                dist = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (idx_i, idx_j)

        if best_pair:
            pairs.append(best_pair)
            remaining.discard(best_pair[0])
            remaining.discard(best_pair[1])
            uf.union(best_pair[0], best_pair[1])

    return pairs

# Test decoder
defects = [(0, 0), (2, 2), (0, 2), (2, 0)]
pairs = union_find_decode(defects, 3)
print(f"\nDefects at: {defects}")
print(f"Decoder pairs: {pairs}")

# =============================================================================
# Part 4: Threshold Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Threshold and Scaling")
print("=" * 70)

def logical_error_rate(p, p_th, d, prefactor=0.03):
    """Calculate logical error rate for surface code."""
    return prefactor * (p / p_th) ** ((d + 1) / 2)

p_th = 0.006  # 0.6% threshold
distances = [3, 5, 7, 9, 11, 13]
p_values = np.logspace(-4, -1.5, 100)

plt.figure(figsize=(10, 6))

for d in distances:
    p_L = [logical_error_rate(p, p_th, d) for p in p_values]
    plt.loglog(p_values * 100, p_L, label=f'd = {d}', linewidth=2)

plt.axvline(x=p_th * 100, color='k', linestyle='--', label=f'Threshold = {p_th*100}%')
plt.xlabel('Physical Error Rate (%)', fontsize=12)
plt.ylabel('Logical Error Rate', fontsize=12)
plt.title('Surface Code Logical Error Rate vs Physical Error Rate', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim([0.01, 3])
plt.ylim([1e-15, 1])
plt.savefig('day_997_surface_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved threshold plot")

# =============================================================================
# Part 5: Lattice Surgery Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Lattice Surgery")
print("=" * 70)

def visualize_lattice_surgery():
    """Create visualization of lattice surgery CNOT."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    titles = [
        'Initial: Three patches',
        'ZZ Merge: Control-Ancilla',
        'Split C-A',
        'XX Merge: Ancilla-Target',
        'Split A-T',
        'Final: CNOT complete'
    ]

    for idx, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 7)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

        # Draw patches
        control_color = 'lightblue'
        target_color = 'lightgreen'
        ancilla_color = 'lightyellow'

        # Control patch
        if idx in [0, 3, 4, 5]:
            ax.add_patch(plt.Rectangle((0, 4), 3, 2, facecolor=control_color,
                                        edgecolor='blue', linewidth=2))
            ax.text(1.5, 5, 'C', ha='center', va='center', fontsize=14, fontweight='bold')
        elif idx in [1, 2]:
            ax.add_patch(plt.Rectangle((0, 4), 3, 2, facecolor=control_color,
                                        edgecolor='blue', linewidth=2))
            ax.text(1.5, 5, 'C', ha='center', va='center', fontsize=14, fontweight='bold')

        # Target patch
        ax.add_patch(plt.Rectangle((7, 4), 3, 2, facecolor=target_color,
                                    edgecolor='green', linewidth=2))
        ax.text(8.5, 5, 'T', ha='center', va='center', fontsize=14, fontweight='bold')

        # Ancilla patch
        if idx == 0:
            ax.add_patch(plt.Rectangle((3.5, 1), 3, 2, facecolor=ancilla_color,
                                        edgecolor='orange', linewidth=2))
            ax.text(5, 2, 'A', ha='center', va='center', fontsize=14, fontweight='bold')
        elif idx == 1:
            # Merged with control
            ax.add_patch(plt.Rectangle((0, 1), 6.5, 5, facecolor='lavender',
                                        edgecolor='purple', linewidth=2, linestyle='--'))
            ax.text(3.25, 3.5, 'C+A\n(ZZ)', ha='center', va='center', fontsize=12)
        elif idx == 2:
            ax.add_patch(plt.Rectangle((3.5, 1), 3, 2, facecolor=ancilla_color,
                                        edgecolor='orange', linewidth=2))
            ax.text(5, 2, 'A', ha='center', va='center', fontsize=14, fontweight='bold')
        elif idx == 3:
            # Merged with target
            ax.add_patch(plt.Rectangle((3.5, 1), 6.5, 5, facecolor='lavender',
                                        edgecolor='purple', linewidth=2, linestyle='--'))
            ax.text(6.75, 3.5, 'A+T\n(XX)', ha='center', va='center', fontsize=12)
        elif idx == 4:
            ax.add_patch(plt.Rectangle((3.5, 1), 3, 2, facecolor=ancilla_color,
                                        edgecolor='orange', linewidth=2))
            ax.text(5, 2, 'A', ha='center', va='center', fontsize=14, fontweight='bold')
        elif idx == 5:
            ax.text(5, 2, 'Measured\n& discarded', ha='center', va='center',
                   fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('day_997_lattice_surgery.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved lattice surgery visualization")

visualize_lattice_surgery()

# =============================================================================
# Part 6: Experimental Results Summary
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Experimental Results")
print("=" * 70)

# Google Willow data
google_data = {
    'd=3': {'logical_error': 0.028, 'physical_qubits': 17},
    'd=5': {'logical_error': 0.013, 'physical_qubits': 49},
    'd=7': {'logical_error': 0.006, 'physical_qubits': 97}
}

print("\nGoogle Willow (2024) Results:")
print("-" * 40)
for d, data in google_data.items():
    print(f"  {d}: pL = {data['logical_error']:.1%}, qubits = {data['physical_qubits']}")

# Calculate Lambda
p_L_3 = google_data['d=3']['logical_error']
p_L_5 = google_data['d=5']['logical_error']
Lambda = p_L_3 / p_L_5
print(f"\n  Error suppression Λ = {Lambda:.2f}")
print(f"  Below threshold: {'Yes' if Lambda > 1 else 'No'}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Surface Code Review Summary")
print("=" * 70)

print("""
Key Results:
1. Rotated surface code: [[d², 1, d]] parameters
2. Threshold: ~0.6% for circuit-level noise
3. Logical error: p_L ~ 0.03 × (p/p_th)^((d+1)/2)
4. Lattice surgery enables fault-tolerant logical gates
5. Union-Find decoder: O(n·α(n)) ≈ linear time
6. Google Willow: First below-threshold demonstration (Λ = 2.14)

Key Formulas:
- Error suppression: Λ = p_L(d) / p_L(d+2)
- Below threshold: Λ > 1 ⟹ larger codes are better
- Resource: ~6500 physical qubits for one d=57 logical qubit
""")

print("Review complete!")
```

---

## Summary Tables

### Surface Code Formulas

| Quantity | Formula |
|----------|---------|
| Data qubits | $d^2$ |
| Ancilla qubits | $d^2 - 1$ |
| Total qubits | $2d^2 - 1$ |
| Logical error rate | $p_L \approx 0.03(p/p_{th})^{(d+1)/2}$ |
| Threshold (circuit) | $\approx 0.6\%$ |
| CNOT via surgery | $O(d)$ cycles |

### Decoder Comparison

| Decoder | Threshold | Complexity | Best For |
|---------|-----------|------------|----------|
| MWPM | ~1.0% | $O(n^3)$ | Research |
| Union-Find | ~0.8% | $O(n\alpha(n))$ | Real-time |
| Neural | ~1.0% | $O(n)$ | Fast, high rate |

### Experimental Milestones

| Year | Group | Achievement |
|------|-------|-------------|
| 2014 | Yale | Logical qubit demo |
| 2022 | Google | 72-qubit surface code |
| 2024 | Google | Below-threshold (Λ=2.14) |
| 2024 | IBM | 1000+ qubit chip |

---

## Self-Assessment Checklist

### Surface Code Structure
- [ ] Can draw rotated surface code layout
- [ ] Can identify X and Z stabilizers
- [ ] Understand boundary types
- [ ] Can write logical operators

### Operations
- [ ] Understand syndrome extraction circuit
- [ ] Can describe lattice surgery protocol
- [ ] Know decoder algorithms and trade-offs

### Experiments
- [ ] Can interpret error suppression factor
- [ ] Know current state-of-the-art results
- [ ] Understand remaining challenges

---

## Preview: Day 998

Tomorrow we review **Fault-Tolerant Quantum Computation**, covering:
- Magic states and T-gate implementation
- State distillation protocols
- Eastin-Knill theorem and its implications
- Transversal gates and code switching
- Resource estimation for fault-tolerant algorithms

---

*"The surface code is the hydrogen atom of quantum error correction."*
--- Austin Fowler

---

**Next:** [Day_998_Thursday.md](Day_998_Thursday.md) - Fault Tolerance Review
