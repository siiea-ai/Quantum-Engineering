# Day 950: Noise-Aware Compilation

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 2.5 hours | Noise-aware mapping and routing theory |
| Afternoon | 2.5 hours | Problem solving and optimization techniques |
| Evening | 2 hours | Computational lab: Noise-adaptive compilation |

## Learning Objectives

By the end of today, you will be able to:

1. Characterize hardware noise profiles including T1, T2, and gate error rates
2. Implement noise-aware qubit mapping algorithms that minimize error exposure
3. Design routing strategies that account for heterogeneous error rates
4. Apply dynamical decoupling sequences for coherence extension
5. Understand pulse-level optimization for improved gate fidelities
6. Evaluate compiled circuit quality using noise-aware metrics

## Core Content

### 1. The Compilation Challenge on NISQ Devices

Standard quantum compilation optimizes for gate count and depth. On NISQ devices, we must also consider:

$$\boxed{\text{Circuit Quality} = f(\text{Gate Errors}, \text{Coherence}, \text{Crosstalk}, \text{Measurement})}$$

**Noise-aware compilation goals:**
1. Map logical qubits to physical qubits with best fidelities
2. Route operations through lowest-error paths
3. Schedule gates to minimize idle time and crosstalk
4. Insert error suppression sequences where beneficial

### 2. Hardware Noise Characterization

#### 2.1 Error Metrics per Qubit/Gate

**Single-qubit error rate:**
$$\epsilon_{1Q}^{(i)} = 1 - F_{1Q}^{(i)}$$

**Two-qubit error rate:**
$$\epsilon_{2Q}^{(i,j)} = 1 - F_{2Q}^{(i,j)}$$

**Readout error:**
$$\epsilon_{\text{ro}}^{(i)} = \frac{1}{2}(P(1|0) + P(0|1))$$

**Coherence-limited error per gate:**
$$\epsilon_{\text{coh}}^{(i)} = 1 - \exp\left(-\frac{t_{\text{gate}}}{T_2^{(i)}}\right) \approx \frac{t_{\text{gate}}}{T_2^{(i)}}$$

#### 2.2 Crosstalk Characterization

Crosstalk occurs when gates on one qubit affect neighboring qubits:

**ZZ crosstalk Hamiltonian:**
$$H_{\text{crosstalk}} = \sum_{i<j} \zeta_{ij} Z_i Z_j$$

where $\zeta_{ij}$ is the coupling strength (typically 10-100 kHz).

**Crosstalk error:**
$$\epsilon_{\text{XT}}^{(i,j)} \approx \left(\zeta_{ij} \cdot t_{\text{gate}}\right)^2$$

### 3. Noise-Aware Qubit Mapping

#### 3.1 The Mapping Problem

Given:
- Logical circuit on $n$ qubits
- Physical device with $m \geq n$ qubits
- Error rates for each physical qubit/edge

Find mapping $\pi: \{0, \ldots, n-1\} \rightarrow \{0, \ldots, m-1\}$ minimizing expected error.

#### 3.2 Error-Weighted Graph

Construct weighted graph $G_{\text{error}} = (V, E, w)$:
- Vertices: physical qubits
- Edges: allowed two-qubit gates
- Weights: $w_{ij} = -\log(1 - \epsilon_{2Q}^{(i,j)}) \approx \epsilon_{2Q}^{(i,j)}$

**Objective function:**
$$\boxed{\min_\pi \sum_{(i,j) \in E_{\text{circuit}}} w_{\pi(i), \pi(j)} + \sum_i c_i \cdot \epsilon_{1Q}^{(\pi(i))}}$$

where $c_i$ is the number of single-qubit gates on logical qubit $i$.

#### 3.3 Mapping Algorithms

**Greedy approach:**
1. Rank physical qubits by quality metric
2. Assign logical qubits with most gates to best physical qubits
3. Respect connectivity constraints

**Subgraph isomorphism:**
Find subgraph of device graph matching circuit interaction graph.

**Simulated annealing:**
Explore mapping space with temperature-controlled random swaps.

### 4. Noise-Aware Routing

#### 4.1 SWAP Insertion with Error Costs

When logical qubits need interaction but aren't adjacent, insert SWAPs.

**Error-aware SWAP cost:**
$$C_{\text{SWAP}}(i,j) = 3 \cdot \epsilon_{2Q}^{(i,j)} + \epsilon_{\text{idle}}$$

Choose SWAP path minimizing total accumulated error.

#### 4.2 Routing Algorithms

**A* search with error heuristic:**
$$f(s) = g(s) + h(s)$$

where:
- $g(s)$ = accumulated error to state $s$
- $h(s)$ = heuristic estimate of remaining error

**Lookahead routing:**
Consider not just current gate but future gates when choosing SWAPs.

$$C_{\text{lookahead}} = \sum_{t=0}^{T} \gamma^t \cdot C_{\text{gate}}(t)$$

### 5. Gate Scheduling and Parallelization

#### 5.1 Crosstalk-Aware Scheduling

**Constraint:** Avoid simultaneous gates with high crosstalk.

Build conflict graph $G_{\text{conflict}}$:
- Vertices: gates in circuit
- Edges: gates that shouldn't execute simultaneously

**Graph coloring** gives parallelization schedule.

#### 5.2 Idle Time Minimization

Idle qubits accumulate T1/T2 errors:

$$\epsilon_{\text{idle}}(t) = 1 - \exp\left(-\frac{t}{T_1}\right) \cdot \exp\left(-\frac{t}{T_\phi}\right)$$

**Strategy:** Minimize total idle time by:
1. Packing gates tightly
2. Delaying state preparation
3. Moving measurements earlier

### 6. Dynamical Decoupling

#### 6.1 Principle

Insert identity-equivalent pulse sequences to average out noise:

$$\boxed{U_{\text{DD}} = X \cdot \tau \cdot X \cdot \tau = I}$$

but noise Hamiltonian is suppressed.

#### 6.2 Common DD Sequences

**CPMG (Carr-Purcell-Meiboom-Gill):**
$$\tau - Y - 2\tau - Y - \tau$$

**XY4:**
$$\tau - X - \tau - Y - \tau - X - \tau - Y$$

**Universally Robust (UR) sequences:**
Higher-order suppression of multiple noise sources.

#### 6.3 DD Effectiveness

For dephasing noise with correlation time $\tau_c$:

$$T_2^{\text{DD}} \approx T_2 \cdot \sqrt{N_{\text{pulses}}}$$

where $N_{\text{pulses}}$ is the number of DD pulses.

**Trade-off:** More pulses improve coherence but add gate errors.

### 7. Pulse-Level Optimization

#### 7.1 Optimal Control Theory

Find control pulse $\Omega(t)$ minimizing:

$$J = 1 - F(U_{\text{target}}, U[\Omega]) + \lambda \int |\Omega(t)|^2 dt$$

**GRAPE algorithm (Gradient Ascent Pulse Engineering):**
Iteratively improve pulse shape using gradient of fidelity.

#### 7.2 Derivative-Based Optimization

$$\frac{\delta F}{\delta \Omega(t)} = \text{Re}\left[\text{Tr}\left[U_{\text{target}}^\dagger \frac{\partial U}{\partial \Omega(t)}\right]\right]$$

#### 7.3 Cross-Resonance Gate Optimization

For superconducting qubits, CR gate pulse can be optimized:

$$H_{\text{CR}} = \Omega_{\text{CR}}(t) \cdot (ZX + \text{parasitic terms})$$

Pulse shaping suppresses leakage and parasitic interactions.

## Quantum Computing Applications

### Application: VQE with Noise-Aware Compilation

For VQE circuits, noise-aware compilation can:
- Reduce chemical accuracy error by 2-5x
- Enable deeper circuits within coherence limits
- Improve gradient estimation quality

### Application: Error-Mitigated QAOA

Combine noise-aware compilation with error mitigation:
1. Compile with noise awareness
2. Apply zero-noise extrapolation
3. Achieve near-ideal results on NISQ hardware

## Worked Examples

### Example 1: Optimal Qubit Mapping

**Problem:** Map a 3-qubit GHZ circuit to a 5-qubit device with heterogeneous error rates.

**Device errors:**
| Edge | Error Rate |
|------|------------|
| (0,1) | 0.02 |
| (1,2) | 0.01 |
| (2,3) | 0.03 |
| (3,4) | 0.015 |
| (1,3) | 0.025 |

**Circuit:** CNOT(0,1), CNOT(1,2)

**Solution:**

Step 1: Identify interaction graph
Logical qubits 0-1 and 1-2 must be adjacent.

Step 2: Find valid mappings (linear chains in device)
- Path 0-1-2: edges (0,1), (1,2) with total error 0.02 + 0.01 = 0.03
- Path 1-2-3: edges (1,2), (2,3) with total error 0.01 + 0.03 = 0.04
- Path 2-3-4: edges (2,3), (3,4) with total error 0.03 + 0.015 = 0.045
- Path 1-3-2: edges (1,3), error depends on routing

Step 3: Select optimal
Best mapping: logical (0,1,2) → physical (0,1,2)
Total two-qubit error: 0.03

**Answer:** Map logical qubits 0,1,2 to physical qubits 0,1,2.

### Example 2: SWAP Path Selection

**Problem:** Execute CNOT(0,3) on a linear chain 0-1-2-3 with error rates:
- (0,1): 0.01
- (1,2): 0.02
- (2,3): 0.015

Find the minimum-error SWAP path.

**Solution:**

Step 1: Options for bringing qubits 0 and 3 adjacent
Option A: SWAP(0,1), SWAP(1,2), then CNOT(2,3)
Option B: SWAP(2,3), SWAP(1,2), then CNOT(0,1)

Step 2: Calculate error for Option A
- SWAP(0,1): 3 × 0.01 = 0.03
- SWAP(1,2): 3 × 0.02 = 0.06
- CNOT(2,3): 0.015
- Total: 0.105

Step 3: Calculate error for Option B
- SWAP(2,3): 3 × 0.015 = 0.045
- SWAP(1,2): 3 × 0.02 = 0.06
- CNOT(0,1): 0.01
- Total: 0.115

**Answer:** Option A is better with total error ≈ 0.105.

### Example 3: Dynamical Decoupling Benefit

**Problem:** A qubit has T2 = 100 μs. An algorithm requires 200 μs of idle time. Calculate the coherence with and without XY4 dynamical decoupling (assuming 10 DD cycles).

**Solution:**

Step 1: Without DD
$$\text{Coherence} = \exp\left(-\frac{200}{100}\right) = e^{-2} \approx 0.135$$

Step 2: With DD (10 XY4 cycles = 40 pulses)
Assuming $T_2^{\text{DD}} \approx T_2 \cdot \sqrt{N/4}$ for XY4:
$$T_2^{\text{DD}} \approx 100 \times \sqrt{10} = 316 \text{ μs}$$

$$\text{Coherence}_{\text{DD}} = \exp\left(-\frac{200}{316}\right) = e^{-0.63} \approx 0.53$$

Step 3: Account for DD pulse errors
If each pulse has error 0.001:
$$\text{Coherence}_{\text{net}} = 0.53 \times (1-0.001)^{40} = 0.53 \times 0.96 = 0.51$$

**Answer:** DD improves coherence from 13.5% to ~51%.

## Practice Problems

### Level 1: Direct Application

1. **Error ranking:** Given qubit T1 values [80, 120, 95, 110, 75] μs and a gate time of 500 ns, rank qubits by coherence-limited error rate.

2. **SWAP cost:** Calculate the error cost of a SWAP on an edge with 2-qubit fidelity 98.5%.

3. **Mapping validation:** For a 3-qubit circuit with CNOT(0,1) and CNOT(0,2), which device topology works without SWAPs: linear, star, or triangle?

### Level 2: Intermediate Analysis

4. **Routing optimization:** On a 2×3 grid, find the minimum-error path for CNOT between opposite corners, given uniform 2% edge error rates.

5. **Crosstalk scheduling:** Two CNOT gates on edges (0,1) and (2,3) have crosstalk coefficient ζ = 50 kHz. If gate time is 300 ns, should they execute in parallel?

6. **DD design:** Design a DD sequence for a 50 μs idle period on a qubit with T2 = 80 μs, using pulses with 0.1% error each.

### Level 3: Challenging Problems

7. **Optimal mapping:** Formulate the qubit mapping problem as an integer linear program (ILP) with error-weighted objectives.

8. **Pulse optimization:** For a cross-resonance gate, derive how pulse amplitude affects both gate speed and leakage error.

9. **Holistic optimization:** Design an algorithm that jointly optimizes mapping, routing, and scheduling for a VQE ansatz on a specific hardware backend.

## Computational Lab: Noise-Adaptive Compilation

### Lab 1: Noise-Aware Mapping and Routing

```python
"""
Day 950 Lab: Noise-Aware Compilation
Implementing error-adaptive qubit mapping and routing
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import (
    SabreLayout, SabreSwap, Optimize1qGates,
    CXCancellation, CommutativeCancellation
)
from qiskit.quantum_info import state_fidelity, Statevector
import networkx as nx

# ============================================================
# Part 1: Create Heterogeneous Noise Model
# ============================================================

def create_heterogeneous_noise_model(n_qubits: int, coupling_map: list,
                                     seed: int = 42) -> tuple:
    """
    Create noise model with varying error rates per qubit/edge.
    Returns noise model and error dictionaries.
    """
    np.random.seed(seed)

    # Generate random but realistic error rates
    single_qubit_errors = {}
    two_qubit_errors = {}
    t1_times = {}
    t2_times = {}

    for q in range(n_qubits):
        # T1, T2 in microseconds (100-300 μs range)
        t1_times[q] = np.random.uniform(100, 300)
        t2_times[q] = np.random.uniform(50, min(200, 2*t1_times[q]))

        # Single-qubit error (0.01% - 0.1%)
        single_qubit_errors[q] = np.random.uniform(0.0001, 0.001)

    for edge in coupling_map:
        i, j = edge
        # Two-qubit error (0.5% - 3%)
        two_qubit_errors[(i,j)] = np.random.uniform(0.005, 0.03)
        two_qubit_errors[(j,i)] = two_qubit_errors[(i,j)]

    # Build Qiskit noise model
    noise_model = NoiseModel()

    for q in range(n_qubits):
        error_1q = depolarizing_error(single_qubit_errors[q], 1)
        noise_model.add_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'], [q])

    for edge in coupling_map:
        error_2q = depolarizing_error(two_qubit_errors[tuple(edge)], 2)
        noise_model.add_quantum_error(error_2q, ['cx'], [edge[0], edge[1]])
        noise_model.add_quantum_error(error_2q, ['cx'], [edge[1], edge[0]])

    return noise_model, {
        'single_qubit': single_qubit_errors,
        'two_qubit': two_qubit_errors,
        't1': t1_times,
        't2': t2_times
    }

# Create 7-qubit device with heavy-hex-like topology
n_qubits = 7
coupling_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                 (1, 4)]  # Cross connection

coupling_map = CouplingMap(coupling_list)

noise_model, error_dict = create_heterogeneous_noise_model(
    n_qubits, coupling_list, seed=42
)

print("Device Error Characterization:")
print("\nSingle-qubit errors:")
for q, err in error_dict['single_qubit'].items():
    print(f"  Q{q}: {err*100:.4f}%")

print("\nTwo-qubit errors:")
for edge, err in error_dict['two_qubit'].items():
    if edge[0] < edge[1]:  # Print each edge once
        print(f"  {edge}: {err*100:.2f}%")

# Visualize device
G = nx.Graph()
G.add_edges_from(coupling_list)

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)

# Color nodes by error rate
node_colors = [error_dict['single_qubit'][q] * 10000 for q in range(n_qubits)]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='RdYlGn_r',
                                node_size=700, vmin=0, vmax=10)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Color edges by error rate
edge_colors = [error_dict['two_qubit'][(e[0], e[1])] * 100 for e in G.edges()]
edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                                edge_cmap=plt.cm.Reds, width=3)

plt.colorbar(nodes, label='1Q Error (×10⁻⁴)')
plt.title('Device Error Map', fontsize=14)
plt.axis('off')
plt.savefig('device_error_map.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 2: Noise-Aware Qubit Mapping
# ============================================================

def compute_mapping_cost(circuit: QuantumCircuit, mapping: dict,
                         error_dict: dict, coupling_list: list) -> float:
    """
    Compute error cost for a given qubit mapping.
    """
    cost = 0.0

    # Count gates per logical qubit
    gate_counts = {q: 0 for q in range(circuit.num_qubits)}
    two_qubit_gates = []

    for instruction in circuit.data:
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if len(qubits) == 1:
            gate_counts[qubits[0]] += 1
        elif len(qubits) == 2:
            two_qubit_gates.append(tuple(qubits))

    # Single-qubit gate costs
    for logical_q, count in gate_counts.items():
        physical_q = mapping[logical_q]
        cost += count * error_dict['single_qubit'][physical_q]

    # Two-qubit gate costs (with SWAP penalty for non-adjacent)
    coupling_edges = set(tuple(e) for e in coupling_list)
    coupling_edges.update((e[1], e[0]) for e in coupling_list)

    for lq1, lq2 in two_qubit_gates:
        pq1, pq2 = mapping[lq1], mapping[lq2]

        if (pq1, pq2) in coupling_edges:
            cost += error_dict['two_qubit'][(pq1, pq2)]
        else:
            # Need SWAPs - estimate cost
            cost += 0.1  # Penalty for requiring routing

    return cost

def find_best_mapping(circuit: QuantumCircuit, n_physical: int,
                      error_dict: dict, coupling_list: list,
                      n_trials: int = 100) -> dict:
    """
    Find best qubit mapping using random search.
    """
    n_logical = circuit.num_qubits
    best_mapping = None
    best_cost = float('inf')

    for _ in range(n_trials):
        # Random mapping
        physical_qubits = np.random.choice(n_physical, n_logical, replace=False)
        mapping = {i: int(physical_qubits[i]) for i in range(n_logical)}

        cost = compute_mapping_cost(circuit, mapping, error_dict, coupling_list)

        if cost < best_cost:
            best_cost = cost
            best_mapping = mapping

    return best_mapping, best_cost

# Create test circuit
test_circuit = QuantumCircuit(4)
test_circuit.h(0)
test_circuit.cx(0, 1)
test_circuit.cx(1, 2)
test_circuit.cx(2, 3)
test_circuit.cx(0, 3)
test_circuit.measure_all()

print("\nTest Circuit:")
print(test_circuit.draw(output='text'))

# Find best mapping
best_mapping, best_cost = find_best_mapping(
    test_circuit, n_qubits, error_dict, coupling_list
)

print(f"\nBest mapping found: {best_mapping}")
print(f"Estimated error cost: {best_cost:.4f}")

# ============================================================
# Part 3: Compare Compilation Strategies
# ============================================================

def evaluate_compilation(circuit: QuantumCircuit, noise_model: NoiseModel,
                         coupling_map: CouplingMap,
                         initial_layout: dict = None,
                         optimization_level: int = 3) -> dict:
    """
    Evaluate compiled circuit quality.
    """
    # Transpile
    transpiled = transpile(
        circuit,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        optimization_level=optimization_level
    )

    # Get gate counts
    ops = transpiled.count_ops()
    cx_count = ops.get('cx', 0)
    depth = transpiled.depth()

    # Simulate with noise
    backend = AerSimulator(noise_model=noise_model)
    job = backend.run(transpiled, shots=10000)
    counts = job.result().get_counts()

    # Success probability (for GHZ-like circuit, expect specific outcomes)
    total = sum(counts.values())
    max_prob = max(counts.values()) / total

    return {
        'cx_count': cx_count,
        'depth': depth,
        'success_prob': max_prob,
        'transpiled_circuit': transpiled
    }

# Compare random vs noise-aware mapping
print("\n" + "="*60)
print("Comparing Compilation Strategies")
print("="*60)

# Random initial layout
random_layout = {i: i for i in range(4)}
result_random = evaluate_compilation(
    test_circuit, noise_model, coupling_map,
    initial_layout=random_layout
)

# Noise-aware initial layout
result_aware = evaluate_compilation(
    test_circuit, noise_model, coupling_map,
    initial_layout=best_mapping
)

print(f"\nRandom mapping:")
print(f"  CX count: {result_random['cx_count']}")
print(f"  Depth: {result_random['depth']}")
print(f"  Success prob: {result_random['success_prob']:.4f}")

print(f"\nNoise-aware mapping:")
print(f"  CX count: {result_aware['cx_count']}")
print(f"  Depth: {result_aware['depth']}")
print(f"  Success prob: {result_aware['success_prob']:.4f}")

# ============================================================
# Part 4: SWAP Route Optimization
# ============================================================

def find_shortest_error_path(source: int, target: int,
                             coupling_list: list,
                             error_dict: dict) -> tuple:
    """
    Find shortest path weighted by error rates.
    """
    G = nx.Graph()
    for edge in coupling_list:
        weight = error_dict['two_qubit'][(edge[0], edge[1])]
        G.add_edge(edge[0], edge[1], weight=weight)

    path = nx.shortest_path(G, source, target, weight='weight')
    total_error = sum(error_dict['two_qubit'][(path[i], path[i+1])]
                      for i in range(len(path)-1))

    return path, total_error

# Example: Route CNOT(0, 6)
path, error = find_shortest_error_path(0, 6, coupling_list, error_dict)
print(f"\nOptimal path from Q0 to Q6: {path}")
print(f"Path error: {error:.4f}")

# Compare with alternative paths
print("\nAlternative paths:")
for alt_path in nx.all_simple_paths(nx.Graph(coupling_list), 0, 6):
    alt_error = sum(error_dict['two_qubit'][(alt_path[i], alt_path[i+1])]
                    for i in range(len(alt_path)-1))
    print(f"  {alt_path}: error = {alt_error:.4f}")

# ============================================================
# Part 5: Crosstalk-Aware Scheduling
# ============================================================

def analyze_crosstalk(coupling_list: list, gate_time_ns: float = 300,
                      crosstalk_freq_khz: float = 50) -> dict:
    """
    Identify gate pairs with significant crosstalk.
    """
    # For simplicity, assume crosstalk between gates sharing a qubit
    # or on adjacent edges
    G = nx.Graph(coupling_list)

    crosstalk_pairs = []
    edges = list(G.edges())

    for i, e1 in enumerate(edges):
        for e2 in edges[i+1:]:
            # Check if edges share a qubit or are adjacent
            shared = set(e1) & set(e2)
            if shared:
                # Crosstalk error estimate
                xt_error = (crosstalk_freq_khz * 1e3 * gate_time_ns * 1e-9 * 2 * np.pi)**2
                crosstalk_pairs.append((e1, e2, xt_error))

    return crosstalk_pairs

crosstalk = analyze_crosstalk(coupling_list)
print("\nCrosstalk analysis:")
for e1, e2, xt_err in crosstalk[:5]:
    print(f"  {e1} ↔ {e2}: crosstalk error = {xt_err:.6f}")

# ============================================================
# Part 6: Dynamical Decoupling Integration
# ============================================================

def add_dynamical_decoupling(circuit: QuantumCircuit,
                             idle_threshold_ns: float = 100) -> QuantumCircuit:
    """
    Add XY4 dynamical decoupling to idle periods.
    (Simplified demonstration)
    """
    # In practice, this requires timing analysis
    # Here we demonstrate the concept

    dd_circuit = circuit.copy()

    # Add DD sequence (XY4) as example
    # X - delay - Y - delay - X - delay - Y
    # For demonstration, we just show the concept

    return dd_circuit

print("\nDynamical decoupling would be inserted during idle periods")
print("to extend coherence times.")

# ============================================================
# Part 7: Benchmark Noise-Aware vs Standard Compilation
# ============================================================

def benchmark_compilation(n_circuits: int = 10, n_qubits_range: list = [3, 4, 5]):
    """
    Benchmark noise-aware vs standard compilation.
    """
    results = {
        'n_qubits': [],
        'standard_fidelity': [],
        'aware_fidelity': [],
        'improvement': []
    }

    for n_log in n_qubits_range:
        print(f"\nBenchmarking {n_log}-qubit circuits...")

        std_fids = []
        aware_fids = []

        for _ in range(n_circuits):
            # Random circuit
            qc = QuantumCircuit(n_log)
            for _ in range(n_log):
                q1, q2 = np.random.choice(n_log, 2, replace=False)
                qc.cx(int(q1), int(q2))
                qc.ry(np.random.uniform(0, np.pi), int(q1))
            qc.measure_all()

            # Standard compilation
            std_result = evaluate_compilation(
                qc, noise_model, coupling_map,
                optimization_level=1
            )

            # Noise-aware compilation
            best_map, _ = find_best_mapping(qc, n_qubits, error_dict,
                                            coupling_list, n_trials=50)
            aware_result = evaluate_compilation(
                qc, noise_model, coupling_map,
                initial_layout=best_map,
                optimization_level=3
            )

            std_fids.append(std_result['success_prob'])
            aware_fids.append(aware_result['success_prob'])

        results['n_qubits'].append(n_log)
        results['standard_fidelity'].append(np.mean(std_fids))
        results['aware_fidelity'].append(np.mean(aware_fids))
        results['improvement'].append(np.mean(aware_fids) / np.mean(std_fids))

    return results

print("\n" + "="*60)
print("Benchmarking Compilation Strategies")
print("="*60)

benchmark_results = benchmark_compilation(n_circuits=5, n_qubits_range=[3, 4])

print("\nResults:")
for i, n in enumerate(benchmark_results['n_qubits']):
    print(f"{n} qubits:")
    print(f"  Standard: {benchmark_results['standard_fidelity'][i]:.4f}")
    print(f"  Noise-aware: {benchmark_results['aware_fidelity'][i]:.4f}")
    print(f"  Improvement: {benchmark_results['improvement'][i]:.2f}x")

print("\nLab complete!")
```

### Lab 2: Pulse-Level Optimization Concepts

```python
"""
Day 950 Lab Part 2: Pulse-Level Optimization
Understanding pulse engineering for better gates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ============================================================
# Part 1: Single-Qubit Gate Pulse Simulation
# ============================================================

def bloch_equations(t, state, omega_func, delta=0):
    """
    Bloch equations for driven two-level system.
    state = [u, v, w] Bloch vector components
    omega_func: time-dependent Rabi frequency
    delta: detuning
    """
    u, v, w = state
    omega = omega_func(t)

    du = delta * v
    dv = -delta * u + omega * w
    dw = -omega * v

    return [du, dv, dw]

def simulate_pulse(omega_func, t_span, t_eval, initial_state=[0, 0, -1]):
    """Simulate Bloch vector evolution under pulse."""
    sol = solve_ivp(
        bloch_equations,
        t_span,
        initial_state,
        t_eval=t_eval,
        args=(omega_func,),
        method='RK45'
    )
    return sol

# Square pulse (standard)
def square_pulse(t, t_gate=20e-9, omega_0=np.pi/(20e-9)):
    """Square pulse for π rotation."""
    if 0 <= t <= t_gate:
        return omega_0
    return 0

# Gaussian pulse (reduced spectral width)
def gaussian_pulse(t, t_gate=20e-9, omega_0=np.pi/(20e-9)):
    """Gaussian-shaped pulse."""
    t_center = t_gate / 2
    sigma = t_gate / 6
    if 0 <= t <= t_gate:
        return omega_0 * 1.2 * np.exp(-(t - t_center)**2 / (2 * sigma**2))
    return 0

# DRAG pulse (Derivative Removal by Adiabatic Gate)
def drag_pulse(t, t_gate=20e-9, omega_0=np.pi/(20e-9), drag_coeff=0.5):
    """DRAG pulse for leakage suppression."""
    t_center = t_gate / 2
    sigma = t_gate / 6
    if 0 <= t <= t_gate:
        gauss = omega_0 * 1.2 * np.exp(-(t - t_center)**2 / (2 * sigma**2))
        derivative = gauss * (-(t - t_center) / sigma**2)
        return gauss + 1j * drag_coeff * derivative
    return 0

# Simulate pulses
t_gate = 20e-9
t_span = [0, t_gate]
t_eval = np.linspace(0, t_gate, 200)

print("Simulating different pulse shapes...")

results = {}
for name, pulse_func in [('Square', square_pulse),
                          ('Gaussian', gaussian_pulse)]:
    sol = simulate_pulse(lambda t: np.real(pulse_func(t)), t_span, t_eval)
    results[name] = sol

# Plot Bloch vector evolution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot pulses
ax = axes[0, 0]
t_plot = np.linspace(0, t_gate, 200)
ax.plot(t_plot * 1e9, [square_pulse(t) / 1e9 for t in t_plot],
        'b-', linewidth=2, label='Square')
ax.plot(t_plot * 1e9, [np.real(gaussian_pulse(t)) / 1e9 for t in t_plot],
        'r-', linewidth=2, label='Gaussian')
ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('Rabi Frequency (GHz)', fontsize=12)
ax.set_title('Pulse Shapes', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot Bloch vector z-component
ax = axes[0, 1]
for name, sol in results.items():
    ax.plot(sol.t * 1e9, sol.y[2], linewidth=2, label=name)
ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('w (z-component)', fontsize=12)
ax.set_title('Bloch Vector Evolution', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot in Bloch sphere projection (u-w plane)
ax = axes[1, 0]
for name, sol in results.items():
    ax.plot(sol.y[0], sol.y[2], linewidth=2, label=name)
ax.set_xlabel('u', fontsize=12)
ax.set_ylabel('w', fontsize=12)
ax.set_title('Bloch Sphere Trajectory (u-w plane)', fontsize=14)
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Final state fidelity comparison
ax = axes[1, 1]
final_states = {name: sol.y[:, -1] for name, sol in results.items()}
target_state = np.array([0, 0, 1])  # |1⟩ state after X gate

fidelities = {}
for name, state in final_states.items():
    fid = (1 + np.dot(state, target_state)) / 2
    fidelities[name] = fid

ax.bar(fidelities.keys(), fidelities.values(), color=['blue', 'red'], alpha=0.7)
ax.axhline(y=1, color='k', linestyle='--', label='Perfect')
ax.set_ylabel('Fidelity', fontsize=12)
ax.set_title('Gate Fidelity Comparison', fontsize=14)
ax.set_ylim([0.95, 1.01])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pulse_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFinal state fidelities:")
for name, fid in fidelities.items():
    print(f"  {name}: {fid:.6f}")

# ============================================================
# Part 2: GRAPE-like Pulse Optimization
# ============================================================

def compute_gate_fidelity(pulse_amplitudes, n_segments, t_gate, target_state):
    """
    Compute fidelity for piecewise-constant pulse.
    """
    dt = t_gate / n_segments

    def piecewise_pulse(t):
        idx = int(t / dt)
        if idx >= n_segments:
            idx = n_segments - 1
        return pulse_amplitudes[idx] * 1e9  # Scale

    sol = simulate_pulse(
        piecewise_pulse,
        [0, t_gate],
        np.linspace(0, t_gate, 100),
        initial_state=[0, 0, -1]
    )

    final_state = sol.y[:, -1]
    fidelity = (1 + np.dot(final_state, target_state)) / 2

    return fidelity

def optimize_pulse(n_segments=10, t_gate=20e-9, target_state=[0, 0, 1]):
    """
    Optimize piecewise-constant pulse for target state.
    """
    def cost(amplitudes):
        return 1 - compute_gate_fidelity(amplitudes, n_segments, t_gate, target_state)

    # Initialize with constant pulse
    initial_amp = np.ones(n_segments) * np.pi / (t_gate * 1e9)

    # Optimize
    result = minimize(
        cost,
        initial_amp,
        method='L-BFGS-B',
        bounds=[(0, 0.5) for _ in range(n_segments)],
        options={'maxiter': 100}
    )

    return result.x, 1 - result.fun

print("\n" + "="*60)
print("Pulse Optimization (GRAPE-like)")
print("="*60)

optimal_pulse, optimal_fid = optimize_pulse(n_segments=10)
print(f"Optimized pulse fidelity: {optimal_fid:.6f}")

# Visualize optimized pulse
t_gate = 20e-9
t_plot = np.linspace(0, t_gate, 100)
dt = t_gate / len(optimal_pulse)

plt.figure(figsize=(10, 5))

# Optimized pulse
pulse_plot = []
for t in t_plot:
    idx = min(int(t / dt), len(optimal_pulse) - 1)
    pulse_plot.append(optimal_pulse[idx])

plt.step(t_plot * 1e9, pulse_plot, 'b-', linewidth=2, where='post',
         label=f'Optimized (F={optimal_fid:.4f})')

# Reference square pulse
ref_amp = np.pi / (t_gate * 1e9)
plt.axhline(y=ref_amp, color='r', linestyle='--', label='Square pulse')

plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Amplitude (arb.)', fontsize=12)
plt.title('GRAPE-Optimized Pulse Shape', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('grape_pulse.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 3: Spectral Analysis of Pulses
# ============================================================

def pulse_spectrum(pulse_func, t_gate, n_points=1000):
    """Compute frequency spectrum of pulse."""
    t = np.linspace(0, t_gate, n_points)
    pulse = np.array([np.real(pulse_func(ti)) for ti in t])

    # FFT
    spectrum = np.fft.fft(pulse)
    freqs = np.fft.fftfreq(n_points, t_gate / n_points)

    return freqs, np.abs(spectrum)

print("\nComputing pulse spectra...")

fig, ax = plt.subplots(figsize=(10, 6))

for name, pulse_func in [('Square', square_pulse), ('Gaussian', gaussian_pulse)]:
    freqs, spectrum = pulse_spectrum(pulse_func, t_gate)
    # Plot positive frequencies only
    mask = freqs >= 0
    ax.semilogy(freqs[mask] / 1e9, spectrum[mask] / spectrum[mask].max(),
                linewidth=2, label=name)

ax.set_xlabel('Frequency (GHz)', fontsize=12)
ax.set_ylabel('Normalized Amplitude', fontsize=12)
ax.set_title('Pulse Frequency Spectra', fontsize=14)
ax.set_xlim([0, 0.5])
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('pulse_spectra.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGaussian pulses have narrower spectra, reducing off-resonant excitation!")
print("Lab complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Coherence-limited error | $\epsilon_{\text{coh}} \approx t_{\text{gate}}/T_2$ |
| SWAP error cost | $C_{\text{SWAP}} = 3 \cdot \epsilon_{2Q}$ |
| Crosstalk error | $\epsilon_{\text{XT}} \approx (\zeta \cdot t_{\text{gate}})^2$ |
| DD coherence extension | $T_2^{\text{DD}} \approx T_2 \sqrt{N_{\text{pulses}}}$ |
| Idle error | $\epsilon_{\text{idle}}(t) = 1 - e^{-t/T_1}e^{-t/T_\phi}$ |

### Key Takeaways

1. **Heterogeneous errors require noise-aware strategies** - uniform compilation wastes hardware potential.

2. **Mapping optimization** can significantly reduce error by using best qubits/edges.

3. **Error-weighted routing** finds SWAP paths minimizing accumulated error.

4. **Crosstalk-aware scheduling** prevents simultaneous conflicting gates.

5. **Dynamical decoupling** extends coherence during idle periods at cost of pulse errors.

6. **Pulse-level optimization** (GRAPE, DRAG) improves native gate fidelities.

## Daily Checklist

- [ ] I understand how to characterize heterogeneous hardware noise
- [ ] I can implement noise-aware qubit mapping algorithms
- [ ] I can design error-weighted routing strategies
- [ ] I understand dynamical decoupling principles and trade-offs
- [ ] I am familiar with pulse-level optimization concepts
- [ ] I completed the computational labs on noise-adaptive compilation

## Preview of Day 951

Tomorrow we explore **Hybrid Classical-Quantum Workflows** - the complete stack for running variational algorithms:
- Classical optimizer selection (gradient-free, gradient-based, natural gradient)
- Shot budget allocation and variance reduction
- Error-aware parameter updates
- Callback strategies and early stopping
- Integration with cloud quantum services

Combining noise-aware compilation with smart classical optimization creates practical NISQ workflows.
