# Day 920: Connectivity and Topology

## Schedule Overview

| Time Block | Duration | Topic |
|------------|----------|-------|
| Morning | 3 hours | Connectivity theory and graph metrics |
| Afternoon | 2.5 hours | Routing and SWAP overhead analysis |
| Evening | 1.5 hours | Computational lab: Topology comparison |

## Learning Objectives

By the end of today, you will be able to:

1. Characterize native connectivity topologies for each quantum platform
2. Calculate graph connectivity metrics (degree, diameter, edge connectivity)
3. Analyze SWAP gate overhead for different coupling maps
4. Evaluate routing algorithms and their efficiency
5. Compare connectivity trade-offs for algorithm implementation
6. Design optimal qubit placement strategies for specific applications

## Core Content

### 1. Native Connectivity Fundamentals

#### Connectivity Graph Definition

A quantum processor's connectivity is described by a graph $G = (V, E)$ where:
- Vertices $V = \{q_1, q_2, \ldots, q_n\}$ represent qubits
- Edges $E = \{(q_i, q_j)\}$ represent possible two-qubit gates

The adjacency matrix $A$ encodes connectivity:

$$A_{ij} = \begin{cases} 1 & \text{if } (q_i, q_j) \in E \\ 0 & \text{otherwise} \end{cases}$$

#### Key Graph Metrics

**Vertex Degree:**
$$d_i = \sum_j A_{ij}$$

Average degree: $\bar{d} = \frac{2|E|}{n}$

**Graph Diameter:**
$$D = \max_{i,j} \text{dist}(q_i, q_j)$$

where dist is the shortest path length.

**Edge Connectivity:**
$$\lambda(G) = \min\{|S| : G - S \text{ is disconnected}\}$$

**Clustering Coefficient:**
$$C_i = \frac{2|\{(j,k) : A_{ij}=A_{ik}=A_{jk}=1\}|}{d_i(d_i-1)}$$

### 2. Platform-Specific Topologies

#### Superconducting Qubits: Planar Architectures

**Heavy-Hex (IBM):**
- Degree: 2-3 (alternating)
- Diameter: $O(\sqrt{n})$
- Error correction friendly (low crosstalk)

$$\boxed{d_{avg} = \frac{12}{5} = 2.4 \text{ for heavy-hex}}$$

**Square Lattice (Google Sycamore):**
- Degree: 4 (interior), 2-3 (boundary)
- Diameter: $O(\sqrt{n})$
- Natural for surface code

**Coupling Strength:**
$$J_{ij} = g_i g_j / \Delta_{ij}$$

where $g$ is the coupling to bus resonator and $\Delta$ is detuning.

Crosstalk constraint: $J_{ij}/\delta\omega < 10^{-3}$ for non-neighbors.

#### Trapped Ions: All-to-All Connectivity

**Single Trap (Penning/Linear Paul):**
- Complete graph $K_n$ for N ions
- Degree: $n-1$ (each qubit connects to all others)
- Diameter: 1

$$\boxed{|E| = \frac{n(n-1)}{2} \text{ for complete graph}}$$

**Multi-zone Architecture:**
- Zones connected via shuttling
- Effective connectivity depends on shuttling time

Shuttling overhead:
$$t_{shuttle} = t_0 + v_{ion} \cdot d_{zone}$$

Typical: $t_{shuttle} \sim 10-100$ μs per zone crossing.

**QCCD (Quantum Charge-Coupled Device):**

Effective diameter with K zones:
$$D_{eff} = 2(K-1) + 1$$

#### Neutral Atoms: Reconfigurable Connectivity

**Optical Tweezer Arrays:**
- Initial arrangement: arbitrary 2D/3D geometry
- Dynamic rearrangement via AOD/SLM

Rearrangement time:
$$t_{rearr} = t_{move} + t_{recool}$$

Typical: $t_{rearr} \sim 1-10$ ms

**Rydberg Blockade Connectivity:**

Two atoms interact if within blockade radius:
$$r_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6}$$

Effective connectivity graph:
$$A_{ij} = \begin{cases} 1 & \text{if } |r_i - r_j| < r_b \\ 0 & \text{otherwise} \end{cases}$$

**Graph Reconfiguration:**
- Can implement any graph topology (up to geometric constraints)
- Trade-off: reconfiguration time vs. connectivity flexibility

### 3. Routing and SWAP Overhead

#### SWAP Gate Basics

SWAP exchanges quantum states between non-adjacent qubits:

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Decomposition into native gates:
- CNOT basis: SWAP = 3 CNOTs
- CZ basis: SWAP = 3 CZ + 4 single-qubit gates
- iSWAP basis: SWAP = 2 iSWAP + single-qubit gates

SWAP error accumulation:
$$\epsilon_{SWAP} = 3\epsilon_{2Q} + O(\epsilon_{2Q}^2)$$

#### Routing Problem

Given:
- Circuit with two-qubit gates on arbitrary qubit pairs
- Hardware connectivity graph G

Find:
- Optimal SWAP insertion to make circuit compatible with G
- Minimize total SWAP count (NP-hard in general)

**Routing Cost:**
$$C_{route} = \sum_{gates} (\text{dist}(q_i, q_j) - 1) \cdot c_{SWAP}$$

where dist is shortest path and $c_{SWAP}$ is cost of one SWAP.

#### Routing Algorithms

**Greedy Algorithm:**
1. For each gate (q_i, q_j) not in E
2. Find shortest SWAP path
3. Insert SWAPs and update qubit mapping

Complexity: $O(n^2 \cdot g)$ for g gates.

**Look-ahead Heuristics (SABRE):**
- Consider future gates when choosing SWAP direction
- Iterative improvement via forward-backward passes

**Optimal Routing (SMT/ILP):**
- Exact solution via constraint satisfaction
- Exponential worst-case but practical for small circuits

### 4. Comparative Connectivity Analysis

#### SWAP Overhead by Platform

For a random n-qubit circuit with g two-qubit gates:

**Superconducting (Square Lattice):**
$$\boxed{N_{SWAP} \approx g \cdot \left(\frac{D}{3}\right) \approx g \cdot \frac{\sqrt{n}}{3}}$$

**Trapped Ion (Complete Graph):**
$$N_{SWAP} = 0$$

**Neutral Atom (Reconfigurable):**
$$N_{SWAP} = 0 \text{ (with reconfiguration)}$$
$$\text{But: } t_{overhead} = t_{rearrange}$$

#### Effective Circuit Depth

Total depth including routing:

$$d_{eff} = d_{logical} + d_{SWAP}$$

For superconducting:
$$d_{SWAP} \approx 3 \cdot N_{SWAP} / n_{parallel}$$

**Comparison for 100-qubit random circuit:**

| Platform | d_logical | d_SWAP | d_total |
|----------|-----------|--------|---------|
| SC (heavy-hex) | 100 | ~150 | ~250 |
| SC (square) | 100 | ~100 | ~200 |
| TI (30 qubits) | 100 | 0 | 100 |
| NA (reconfig) | 100 | 0 | 100 + t_rearr |

#### Parallelism Analysis

**Superconducting:**
Maximum parallel 2Q gates: $\lfloor n/2 \rfloor$ (with matching)

With routing constraints:
$$P_{eff} = \frac{\text{parallel gates}}{\text{total gates per layer}} \approx 0.3-0.5$$

**Trapped Ion:**
Typically 1 gate at a time (individual addressing)
Parallel via multi-zone: P = K zones

**Neutral Atom:**
Global Rydberg drive: P ~ n/2 (parallel CZ)
Individual addressing: P ~ n/2k (k control zones)

### 5. Algorithm-Specific Connectivity Requirements

#### QAOA

Requires edges matching problem graph $G_P$:

$$H_C = \sum_{(i,j) \in G_P} J_{ij} Z_i Z_j$$

Overhead if $G_P \not\subseteq G_{hardware}$:
$$N_{SWAP} \propto |E_P - E_H|$$

**Example: Max-Cut on 3-regular graph (100 nodes):**
- SC square: ~50 SWAPs/layer
- TI complete: 0 SWAPs
- NA reconfigurable: 0 SWAPs (rearrange to match)

#### Quantum Simulation (Local Hamiltonian)

Local interactions naturally map to nearest-neighbor:

$$H = \sum_{\langle i,j \rangle} J_{ij} \vec{S}_i \cdot \vec{S}_j$$

**Platform Fit:**
- SC: Excellent for 2D Heisenberg models
- TI: Less natural, but SWAPs free
- NA: Reconfigure to match interaction graph

#### Quantum Chemistry (All-to-All Interactions)

Molecular Hamiltonians have long-range terms:

$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$$

Under Jordan-Wigner:
$$N_{2Q} \propto n^4 \text{ with connectivity } O(n)$$

**Routing Overhead:**
- SC: Significant (many SWAPs for long-range terms)
- TI: Zero SWAPs, but gate count still high
- NA: Reconfigurable matches interaction graph

### 6. Advanced Connectivity Concepts

#### Spectral Graph Properties

Laplacian matrix:
$$L = D - A$$

where $D_{ii} = d_i$ (degree matrix).

Algebraic connectivity (Fiedler value):
$$\lambda_2(L) = \min_{x \perp \mathbf{1}} \frac{x^T L x}{x^T x}$$

Higher $\lambda_2$ → better connectivity, faster mixing.

**Platform Comparison:**
- Complete graph: $\lambda_2 = n$
- Square lattice: $\lambda_2 \approx 4\sin^2(\pi/2\sqrt{n})$
- Heavy-hex: $\lambda_2 \approx 2\sin^2(\pi/2n^{1/2})$

#### Expander Graphs for Error Correction

Good LDPC codes require:
- Low degree (for practical checks)
- High expansion (for error correction)

Expansion ratio:
$$h(G) = \min_{|S| \leq n/2} \frac{|\partial S|}{|S|}$$

Platforms with reconfigurable connectivity (NA) can implement optimal expander topologies.

## Quantum Computing Applications

### Circuit Mapping Strategy

1. **Analyze circuit connectivity graph** $G_C$
2. **Compare with hardware graph** $G_H$
3. **Initial placement**: Map high-degree circuit nodes to high-degree hardware qubits
4. **Routing**: Apply SABRE or optimal routing
5. **Evaluate overhead**: SWAPs, depth increase, error accumulation

### Connectivity-Aware Compilation

Objective function:
$$\mathcal{L} = \alpha \cdot d_{total} + \beta \cdot N_{SWAP} + \gamma \cdot \epsilon_{route}$$

where:
- $d_{total}$ = total circuit depth
- $N_{SWAP}$ = SWAP gate count
- $\epsilon_{route}$ = accumulated routing error

## Worked Examples

### Example 1: SWAP Count for Linear Chain

**Problem:** A 10-qubit linear chain processor needs to implement a fully-connected 5-qubit circuit. Calculate minimum SWAP count for one layer of all-pairs CZ gates.

**Solution:**

1. All-pairs on 5 qubits: $\binom{5}{2} = 10$ CZ gates

2. On a linear chain with qubits at positions 1-10, place circuit qubits at:
   - Optimal: positions 1, 3, 5, 7, 9 (odd positions)

3. Calculate distances:
   - Adjacent (1,3), (3,5), etc.: distance 2 → 1 SWAP each (4 pairs)
   - Next-nearest (1,5), (3,7), etc.: distance 4 → 3 SWAPs each (3 pairs)
   - (1,7), (3,9): distance 6 → 5 SWAPs each (2 pairs)
   - (1,9): distance 8 → 7 SWAPs (1 pair)

4. Total SWAPs (naive):
   $$N_{SWAP} = 4(1) + 3(3) + 2(5) + 1(7) = 4 + 9 + 10 + 7 = 30$$

5. With parallel execution and SWAP reuse, can reduce to ~15 SWAPs.

**Answer:** Approximately 15-30 SWAPs depending on routing strategy.

### Example 2: Graph Diameter Comparison

**Problem:** Compare the graph diameter for 100-qubit systems: (a) 10×10 square lattice, (b) heavy-hex with 100 qubits, (c) complete graph.

**Solution:**

(a) **Square lattice (10×10):**
$$D = 2(10-1) = 18$$
(corner to corner along edges)

(b) **Heavy-hex:**
Approximately: $D \approx 2.5\sqrt{n} = 25$ for 100 qubits

(c) **Complete graph:**
$$D = 1$$
(all pairs directly connected)

**Answer:** Complete graph (D=1) >> Square lattice (D=18) > Heavy-hex (D≈25)

### Example 3: Routing Overhead for QAOA

**Problem:** A Max-Cut QAOA on a 3-regular graph with 50 nodes runs on a heavy-hex processor. Estimate the routing overhead per QAOA layer.

**Solution:**

1. 3-regular graph has $|E| = 50 \times 3 / 2 = 75$ edges

2. Heavy-hex has average degree 2.4, so ~40% of edges won't be native

3. Non-native edges: $0.4 \times 75 = 30$ edges

4. Average distance for non-native edge: ~3 (estimated from heavy-hex structure)

5. SWAPs per edge: 2 (need to move both qubits closer)

6. Total SWAPs per layer:
   $$N_{SWAP} \approx 30 \times 2 = 60$$

7. SWAP error contribution:
   $$\epsilon_{route} = 60 \times 3 \times 0.005 = 0.9$$

**Answer:** ~60 SWAPs per QAOA layer, contributing ~1% error per layer.

## Practice Problems

### Level 1: Direct Application

1. Calculate the average degree and diameter of a 27-qubit heavy-hex processor (IBM Falcon).

2. A SWAP gate has error 0.5%. How many SWAPs can be performed before fidelity drops below 50%?

3. For a linear 20-qubit chain, what is the minimum number of SWAPs to bring qubits 1 and 20 adjacent?

### Level 2: Intermediate Analysis

4. Design an initial qubit placement for a 4-qubit GHZ state circuit on a heavy-hex processor that minimizes SWAP count.

5. Compare the SWAP overhead for implementing a 2D Heisenberg model (10×10 lattice) on:
   - 100-qubit square lattice processor
   - 100-qubit linear chain
   - 100-qubit trapped ion (all-to-all)

6. Calculate the effective circuit depth increase when routing a random 50-qubit circuit (100 2-qubit gates) on a square lattice vs. all-to-all connectivity.

### Level 3: Advanced Research-Level

7. Prove that the SWAP routing problem is NP-complete by reduction from graph isomorphism subproblems.

8. Design an adaptive connectivity strategy for neutral atoms that balances reconfiguration time against SWAP overhead for a quantum chemistry simulation.

9. Analyze the connectivity requirements for implementing the surface code on each platform, considering both logical operations and error correction overhead.

## Computational Lab: Topology Analysis

```python
"""
Day 920 Computational Lab: Connectivity and Topology Analysis
Compares connectivity graphs and routing overhead across platforms
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['figure.figsize'] = (14, 12)
plt.rcParams['font.size'] = 11

# =============================================================================
# Part 1: Generate Platform Connectivity Graphs
# =============================================================================

def create_square_lattice(rows: int, cols: int) -> nx.Graph:
    """Create square lattice connectivity graph"""
    G = nx.grid_2d_graph(rows, cols)
    # Relabel to integer nodes
    mapping = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
    G = nx.relabel_nodes(G, mapping)
    return G

def create_heavy_hex(n_qubits: int) -> nx.Graph:
    """Create IBM heavy-hex style connectivity"""
    # Approximate heavy-hex structure
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))

    # Create hexagonal pattern with reduced connectivity
    rows = int(np.sqrt(n_qubits))
    cols = n_qubits // rows

    for i in range(n_qubits):
        row, col = i // cols, i % cols

        # Horizontal connections (every other)
        if col < cols - 1 and row % 2 == col % 2:
            G.add_edge(i, i + 1)

        # Vertical connections (sparse)
        if row < rows - 1:
            if col % 2 == 0:
                G.add_edge(i, i + cols)

    return G

def create_complete_graph(n: int) -> nx.Graph:
    """Create complete graph (all-to-all connectivity)"""
    return nx.complete_graph(n)

def create_linear_chain(n: int) -> nx.Graph:
    """Create linear chain (path graph)"""
    return nx.path_graph(n)

# Create graphs for comparison
n_qubits = 50

graphs = {
    'Square Lattice': create_square_lattice(7, 7),  # 49 qubits
    'Heavy-Hex': create_heavy_hex(50),
    'Complete (TI)': create_complete_graph(50),
    'Linear Chain': create_linear_chain(50)
}

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, G) in enumerate(graphs.items()):
    ax = axes[idx]

    if name == 'Complete (TI)':
        # Use circular layout for complete graph
        pos = nx.circular_layout(G)
        # Only draw subset of edges for visibility
        G_vis = G.copy()
        edges_to_remove = list(G.edges())[::5]
        G_vis.remove_edges_from(edges_to_remove)
        nx.draw(G_vis, pos, ax=ax, node_size=50, node_color='orange',
               edge_color='gray', alpha=0.5, with_labels=False)
        ax.set_title(f'{name}\n(edges subsampled for visibility)')
    else:
        if name == 'Square Lattice':
            pos = {i: (i % 7, i // 7) for i in G.nodes()}
        elif name == 'Linear Chain':
            pos = {i: (i, 0) for i in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=42)

        nx.draw(G, pos, ax=ax, node_size=100, node_color='skyblue',
               edge_color='gray', with_labels=False)
        ax.set_title(name)

    # Add metrics
    metrics_text = f"n={G.number_of_nodes()}, |E|={G.number_of_edges()}"
    if G.number_of_nodes() < 100 and name != 'Complete (TI)':
        metrics_text += f"\nD={nx.diameter(G)}, avg_deg={2*G.number_of_edges()/G.number_of_nodes():.2f}"
    ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('connectivity_graphs.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 2: Graph Metrics Comparison
# =============================================================================

def compute_graph_metrics(G: nx.Graph) -> Dict:
    """Compute comprehensive graph metrics"""
    metrics = {}

    # Basic metrics
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['avg_degree'] = 2 * G.number_of_edges() / G.number_of_nodes()

    # Degree distribution
    degrees = [d for n, d in G.degree()]
    metrics['min_degree'] = min(degrees)
    metrics['max_degree'] = max(degrees)

    # Distance metrics (expensive for large graphs)
    if G.number_of_nodes() <= 100:
        if nx.is_connected(G):
            metrics['diameter'] = nx.diameter(G)
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            metrics['diameter'] = float('inf')
            metrics['avg_path_length'] = float('inf')

        # Connectivity
        metrics['edge_connectivity'] = nx.edge_connectivity(G)
        metrics['node_connectivity'] = nx.node_connectivity(G)
    else:
        # Sample for large graphs
        metrics['diameter'] = 'N/A (large)'
        metrics['avg_path_length'] = 'N/A (large)'

    # Spectral properties
    if G.number_of_nodes() <= 100:
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        metrics['algebraic_connectivity'] = eigenvalues[1] if len(eigenvalues) > 1 else 0

    return metrics

print("\n" + "="*80)
print("GRAPH METRICS COMPARISON")
print("="*80)

all_metrics = {}
for name, G in graphs.items():
    all_metrics[name] = compute_graph_metrics(G)

# Print comparison table
print(f"\n{'Metric':<25} {'Square':<12} {'Heavy-Hex':<12} {'Complete':<12} {'Linear':<12}")
print("-" * 73)

metric_names = ['nodes', 'edges', 'avg_degree', 'diameter', 'avg_path_length',
                'edge_connectivity', 'algebraic_connectivity']

for metric in metric_names:
    row = f"{metric:<25}"
    for name in ['Square Lattice', 'Heavy-Hex', 'Complete (TI)', 'Linear Chain']:
        val = all_metrics[name].get(metric, 'N/A')
        if isinstance(val, float):
            row += f"{val:<12.3f}"
        else:
            row += f"{str(val):<12}"
    print(row)

# =============================================================================
# Part 3: SWAP Routing Analysis
# =============================================================================

def count_swaps_greedy(G: nx.Graph, gate_pairs: List[Tuple[int, int]]) -> int:
    """
    Count minimum SWAPs needed using greedy routing

    Parameters:
    -----------
    G: Hardware connectivity graph
    gate_pairs: List of (qubit_i, qubit_j) pairs for 2Q gates

    Returns:
    --------
    total_swaps: Total SWAP count
    """
    total_swaps = 0

    for q1, q2 in gate_pairs:
        if q1 >= G.number_of_nodes() or q2 >= G.number_of_nodes():
            continue

        if G.has_edge(q1, q2):
            continue  # Native connection

        try:
            # Find shortest path
            path = nx.shortest_path(G, q1, q2)
            # SWAPs needed = path length - 1 - 1 (for the gate itself)
            swaps_needed = len(path) - 2
            total_swaps += max(0, swaps_needed)
        except nx.NetworkXNoPath:
            total_swaps += float('inf')

    return total_swaps

def generate_random_circuit(n_qubits: int, n_gates: int,
                           connectivity: str = 'random') -> List[Tuple[int, int]]:
    """Generate random two-qubit gate pairs"""
    if connectivity == 'random':
        gates = []
        for _ in range(n_gates):
            q1, q2 = np.random.choice(n_qubits, 2, replace=False)
            gates.append((q1, q2))
        return gates
    elif connectivity == 'nearest_neighbor':
        gates = []
        for _ in range(n_gates):
            q1 = np.random.randint(0, n_qubits - 1)
            q2 = q1 + 1
            gates.append((q1, q2))
        return gates
    else:
        raise ValueError(f"Unknown connectivity type: {connectivity}")

# Compare SWAP overhead across platforms
np.random.seed(42)
n_qubits_test = 30  # Smaller for faster computation
n_gates = 100

# Create test graphs
test_graphs = {
    'Square (6x5)': create_square_lattice(6, 5),
    'Heavy-Hex': create_heavy_hex(30),
    'Complete': create_complete_graph(30),
    'Linear': create_linear_chain(30)
}

# Generate random circuit
random_gates = generate_random_circuit(n_qubits_test, n_gates, 'random')

print("\n" + "="*80)
print("SWAP OVERHEAD ANALYSIS")
print("="*80)

swap_counts = {}
for name, G in test_graphs.items():
    swaps = count_swaps_greedy(G, random_gates)
    swap_counts[name] = swaps
    print(f"{name}: {swaps} SWAPs for {n_gates} random gates")

# Visualize SWAP overhead
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SWAP count bar chart
ax = axes[0, 0]
platforms = list(swap_counts.keys())
counts = list(swap_counts.values())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(platforms, counts, color=colors)
ax.set_ylabel('Total SWAPs')
ax.set_title(f'SWAP Overhead for {n_gates} Random 2Q Gates')
ax.set_ylim(0, max(counts) * 1.2)

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
           str(count), ha='center', va='bottom')

# Plot 2: Scaling with circuit size
ax = axes[0, 1]
gate_counts = [20, 50, 100, 200, 500]

for idx, (name, G) in enumerate(test_graphs.items()):
    swaps_scaling = []
    for n_g in gate_counts:
        gates = generate_random_circuit(n_qubits_test, n_g, 'random')
        swaps = count_swaps_greedy(G, gates)
        swaps_scaling.append(swaps)
    ax.plot(gate_counts, swaps_scaling, 'o-', label=name, color=colors[idx])

ax.set_xlabel('Number of 2Q Gates')
ax.set_ylabel('Total SWAPs Required')
ax.set_title('SWAP Scaling with Circuit Size')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Distance distribution
ax = axes[1, 0]

for idx, (name, G) in enumerate(test_graphs.items()):
    if name == 'Complete':
        continue  # All distances = 1

    # Sample pairwise distances
    distances = []
    nodes = list(G.nodes())
    for _ in range(1000):
        n1, n2 = np.random.choice(nodes, 2, replace=False)
        try:
            d = nx.shortest_path_length(G, n1, n2)
            distances.append(d)
        except:
            pass

    ax.hist(distances, bins=range(1, max(distances)+2), alpha=0.5,
           label=name, color=colors[idx], density=True)

ax.set_xlabel('Shortest Path Distance')
ax.set_ylabel('Frequency')
ax.set_title('Pairwise Distance Distribution')
ax.legend()

# Plot 4: Effective circuit depth increase
ax = axes[1, 1]

# Model: depth increase = 3 * SWAP_count / parallelism
parallelism = {'Square (6x5)': 10, 'Heavy-Hex': 8, 'Complete': 15, 'Linear': 1}

depth_increase = {}
for name in test_graphs:
    swaps = swap_counts[name]
    # Each SWAP = 3 CZ gates, limited parallelism
    depth_increase[name] = 3 * swaps / parallelism[name]

ax.bar(platforms, [depth_increase[p] for p in platforms], color=colors)
ax.set_ylabel('Circuit Depth Increase')
ax.set_title('Routing-Induced Depth Overhead')

plt.tight_layout()
plt.savefig('routing_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 4: Algorithm-Specific Analysis
# =============================================================================

print("\n" + "="*80)
print("ALGORITHM-SPECIFIC CONNECTIVITY ANALYSIS")
print("="*80)

# QAOA on Max-Cut (3-regular graph)
def create_maxcut_circuit(n_nodes: int, regularity: int = 3) -> List[Tuple[int, int]]:
    """Generate edges for a random regular graph (Max-Cut instance)"""
    G_problem = nx.random_regular_graph(regularity, n_nodes, seed=42)
    return list(G_problem.edges())

# Quantum simulation (2D Heisenberg)
def create_heisenberg_circuit(rows: int, cols: int) -> List[Tuple[int, int]]:
    """Generate edges for 2D Heisenberg model"""
    G_model = nx.grid_2d_graph(rows, cols)
    mapping = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
    edges = [(mapping[e[0]], mapping[e[1]]) for e in G_model.edges()]
    return edges

# Quantum chemistry (all-to-all)
def create_chemistry_circuit(n_qubits: int, n_terms: int) -> List[Tuple[int, int]]:
    """Generate typical quantum chemistry circuit interactions"""
    gates = []
    for _ in range(n_terms):
        q1, q2 = np.random.choice(n_qubits, 2, replace=False)
        gates.append((q1, q2))
    return gates

# Compare SWAP overhead for different algorithms
algorithms = {
    'Max-Cut (QAOA)': create_maxcut_circuit(30, 3),
    '2D Heisenberg': create_heisenberg_circuit(5, 6),
    'Chemistry (random)': create_chemistry_circuit(30, 100)
}

print("\nSWAP overhead by algorithm and platform:")
print(f"{'Algorithm':<20} {'Square':<12} {'Heavy-Hex':<12} {'Complete':<12} {'Linear':<12}")
print("-" * 68)

for alg_name, gates in algorithms.items():
    row = f"{alg_name:<20}"
    for platform_name, G in test_graphs.items():
        swaps = count_swaps_greedy(G, gates)
        row += f"{swaps:<12}"
    print(row)

# =============================================================================
# Part 5: Summary Visualization
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Create summary radar chart data
categories = ['Connectivity\n(higher=better)', 'SWAP Overhead\n(lower=better)',
              'Parallelism\n(higher=better)', 'Scalability\n(higher=better)']

# Normalize metrics (0-1 scale, higher = better)
platform_scores = {
    'Square Lattice': [0.3, 0.5, 0.7, 0.9],
    'Heavy-Hex': [0.25, 0.4, 0.6, 0.95],
    'Complete (TI)': [1.0, 1.0, 0.3, 0.3],
    'Linear Chain': [0.1, 0.2, 0.2, 0.5]
}

x = np.arange(len(categories))
width = 0.2

for idx, (platform, scores) in enumerate(platform_scores.items()):
    ax.bar(x + idx*width, scores, width, label=platform, color=colors[idx])

ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(categories)
ax.set_ylabel('Normalized Score (higher = better)')
ax.set_title('Connectivity Trade-off Summary')
ax.legend(loc='upper right')
ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('connectivity_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. Complete connectivity (trapped ions) eliminates SWAP overhead but has
   limited parallelism and scalability challenges.

2. Square/heavy-hex lattices require significant SWAP routing but scale
   well and support good parallelism.

3. Algorithm choice matters: local algorithms (Heisenberg) favor lattice
   connectivity; non-local algorithms (chemistry) favor all-to-all.

4. Neutral atom reconfigurability can match circuit topology, eliminating
   SWAP overhead at the cost of reconfiguration time.

5. Linear chains have severe routing overhead and should be avoided for
   algorithms with non-local connectivity requirements.
""")
```

## Summary

### Connectivity Metrics by Platform

| Platform | Avg Degree | Diameter | Edge Connectivity |
|----------|-----------|----------|-------------------|
| Square Lattice | 4 | O(√n) | 2 |
| Heavy-Hex | 2.4 | O(√n) | 2 |
| Complete (TI) | n-1 | 1 | n-1 |
| Linear Chain | 2 | n-1 | 1 |
| Neutral Atom | Reconfigurable | Variable | Variable |

### Key Formulas

| Quantity | Formula |
|----------|---------|
| SWAP decomposition | $$\text{SWAP} = 3 \times \text{CNOT}$$ |
| Routing overhead | $$N_{SWAP} \approx g \cdot D/3$$ |
| Depth increase | $$d_{SWAP} = 3N_{SWAP}/P$$ |
| Algebraic connectivity | $$\lambda_2 = \min_{x\perp 1} \frac{x^T L x}{x^T x}$$ |

### Main Takeaways

1. **Connectivity-algorithm matching** is crucial for minimizing overhead
2. **SWAP overhead scales** with graph diameter for fixed architectures
3. **Parallelism trades off** against connectivity (complete graphs have low parallelism)
4. **Reconfigurable topologies** (neutral atoms) offer unique flexibility
5. **Error correction requirements** favor specific topologies (surface code → lattice)

## Daily Checklist

- [ ] I can characterize connectivity graphs with standard metrics
- [ ] I understand SWAP routing overhead for different architectures
- [ ] I can compare connectivity requirements for different algorithms
- [ ] I can evaluate trade-offs between connectivity and parallelism
- [ ] I can design qubit placement strategies for specific applications
- [ ] I understand how topology affects error correction overhead

## Preview of Day 921

Tomorrow we analyze **Scalability**, examining qubit count scaling, control complexity challenges, and the engineering requirements for building large-scale quantum processors on each platform.
