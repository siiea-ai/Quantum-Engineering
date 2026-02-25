# Day 946: NISQ Era Characteristics

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 2.5 hours | NISQ device fundamentals and hardware landscape |
| Afternoon | 2.5 hours | Problem solving and analysis exercises |
| Evening | 2 hours | Computational lab: Hardware characterization |

## Learning Objectives

By the end of today, you will be able to:

1. Define the NISQ era and its distinguishing characteristics from fault-tolerant quantum computing
2. Quantify key hardware metrics: qubit count, T1/T2 coherence times, and gate fidelities
3. Analyze connectivity graphs and their impact on circuit compilation
4. Calculate effective circuit depth limits based on error rates
5. Compare different NISQ hardware platforms (superconducting, trapped ion, photonic)
6. Evaluate quantum volume and other benchmarking metrics

## Core Content

### 1. What is NISQ?

The term **Noisy Intermediate-Scale Quantum** was coined by John Preskill in 2018 to characterize the current era of quantum computing. NISQ devices occupy a unique position in quantum computing history:

$$\boxed{\text{NISQ: } 50-1000 \text{ qubits}, \text{ no error correction}, \text{ limited coherence}}$$

**Key characteristics:**

| Property | NISQ | Fault-Tolerant |
|----------|------|----------------|
| Qubit count | 50-1000+ | Millions (logical) |
| Error correction | None/limited | Full QEC |
| Circuit depth | ~100-1000 gates | Unlimited |
| Gate fidelity | 99-99.9% | >99.99% (logical) |
| Coherence limited | Yes | No |

### 2. Hardware Metrics Deep Dive

#### 2.1 Coherence Times

Qubits lose their quantum information through two primary mechanisms:

**T1 (Energy Relaxation):**
The time for an excited state $|1\rangle$ to decay to $|0\rangle$:

$$P_{|1\rangle}(t) = P_{|1\rangle}(0) e^{-t/T_1}$$

**T2 (Dephasing):**
The time for phase coherence to decay in a superposition:

$$|\psi(t)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\phi(t)}|1\rangle)$$

where phase fluctuations grow as:

$$\langle e^{i\phi(t)}\rangle = e^{-t/T_2}$$

**Fundamental relationship:**

$$\boxed{\frac{1}{T_2} \leq \frac{1}{2T_1} + \frac{1}{T_\phi}}$$

where $T_\phi$ is the pure dephasing time.

#### 2.2 Gate Fidelities

Gate fidelity quantifies how close an implemented gate $\mathcal{E}$ is to the ideal unitary $U$:

**Average Gate Fidelity:**
$$F_{\text{avg}}(\mathcal{E}, U) = \int d\psi \langle\psi|U^\dagger \mathcal{E}(|\psi\rangle\langle\psi|) U|\psi\rangle$$

For a depolarizing channel with error rate $p$:

$$\boxed{F = 1 - p\frac{d^2-1}{d^2}}$$

where $d=2^n$ is the Hilbert space dimension.

**Typical NISQ fidelities (2024-2026):**

| Gate Type | Superconducting | Trapped Ion |
|-----------|-----------------|-------------|
| Single-qubit | 99.9% | 99.99% |
| Two-qubit | 99-99.5% | 99-99.9% |
| Measurement | 99% | 99.9% |

#### 2.3 Circuit Depth Limits

The maximum useful circuit depth is limited by accumulated errors:

$$\boxed{d_{\max} \approx \frac{1}{\epsilon_{\text{eff}}} \approx \frac{T_{\text{coh}}}{t_{\text{gate}}}}$$

where:
- $\epsilon_{\text{eff}}$ is the effective error per layer
- $T_{\text{coh}} \sim \min(T_1, T_2)$
- $t_{\text{gate}}$ is the gate duration

**Effective circuit fidelity:**

$$F_{\text{circuit}} = \prod_{i=1}^{N_{\text{gates}}} F_i \approx F_{\text{avg}}^{N_{\text{gates}}}$$

For $N$ gates with average fidelity $F = 1-\epsilon$:

$$F_{\text{circuit}} \approx e^{-N\epsilon}$$

### 3. Connectivity and Topology

#### 3.1 Coupling Maps

NISQ devices have limited qubit connectivity described by a graph $G = (V, E)$:
- Vertices $V$: physical qubits
- Edges $E$: allowed two-qubit gates

**Common topologies:**

```
Linear:       0—1—2—3—4

Grid:         0—1—2
              |   |
              3—4—5
              |   |
              6—7—8

Heavy-hex:    IBM Eagle/Heron topology
              (sparse hexagonal lattice)
```

#### 3.2 Connectivity Metrics

**Graph density:**
$$\rho = \frac{2|E|}{|V|(|V|-1)}$$

**Average path length:**
$$\bar{d} = \frac{1}{|V|(|V|-1)} \sum_{i \neq j} d(i,j)$$

**Impact on circuit compilation:**
A CNOT between non-adjacent qubits requires SWAP insertion:

$$\text{SWAP}_{i,j} = \text{CNOT}_{i,j} \cdot \text{CNOT}_{j,i} \cdot \text{CNOT}_{i,j}$$

Each SWAP adds 3 CNOTs, dramatically increasing circuit depth.

### 4. Quantum Volume

Quantum Volume (QV) is a holistic benchmark capturing qubit count, connectivity, and gate fidelities:

$$\boxed{QV = 2^n}$$

where $n$ is the largest square circuit (width = depth = $n$) that can be executed with >2/3 success probability.

**Procedure:**
1. Generate random SU(4) circuits of width $n$, depth $n$
2. Compile to native gates
3. Execute and measure
4. Compare output distribution to ideal
5. Find largest $n$ with heavy output probability $>2/3$

**Heavy output probability:**
$$h_U = \Pr[\text{output in heavy set}] = \sum_{x: p_U(x) > \text{median}} p_{\text{exp}}(x)$$

Ideal random circuit: $h_U \approx (1 + \ln 2)/2 \approx 0.85$

Successful QV test: $h_U > 2/3 + 2\sigma$

### 5. NISQ Hardware Platforms

#### 5.1 Superconducting Qubits

**Architecture:** Transmon qubits with coplanar waveguide resonators

| Parameter | Typical Value |
|-----------|---------------|
| T1 | 100-500 μs |
| T2 | 100-300 μs |
| Gate time (1Q) | 20-50 ns |
| Gate time (2Q) | 200-500 ns |
| Connectivity | Sparse (degree 2-3) |

**Advantages:** Fast gates, scalable fabrication
**Challenges:** Cryogenic operation (15 mK), frequency crowding

#### 5.2 Trapped Ion Qubits

**Architecture:** Individual ions in linear Paul traps

| Parameter | Typical Value |
|-----------|---------------|
| T1 | Seconds to hours |
| T2 | 1-10 seconds |
| Gate time (1Q) | 1-10 μs |
| Gate time (2Q) | 100-1000 μs |
| Connectivity | All-to-all |

**Advantages:** High fidelity, full connectivity
**Challenges:** Slow gates, ion crystal reordering

#### 5.3 Neutral Atom Arrays

**Architecture:** Optical tweezers holding neutral atoms

| Parameter | Typical Value |
|-----------|---------------|
| T1 | Seconds |
| T2 | 100 ms - 1 s |
| Gate time | 0.1-1 μs (Rydberg) |
| Connectivity | Reconfigurable |

**Advantages:** Large arrays (1000+ qubits), programmable connectivity
**Challenges:** Atom loss, state preparation fidelity

## Quantum Computing Applications

### Application to Algorithm Design

Understanding NISQ limitations directly informs algorithm design:

1. **Circuit depth constraints** → Variational ansatzes must be shallow
2. **Limited connectivity** → Hardware-efficient ansatzes exploit native topology
3. **Measurement noise** → Shot budget optimization becomes critical
4. **Gate errors** → Error mitigation techniques are essential

### NISQ Algorithm Taxonomy

| Algorithm Class | Example | Depth Requirement |
|----------------|---------|-------------------|
| Variational | VQE, QAOA | O(poly log n) |
| Sampling | Boson sampling | O(n) - O(n²) |
| Near-term simulation | Trotterized dynamics | O(t/ε) |

## Worked Examples

### Example 1: Circuit Depth Analysis

**Problem:** A superconducting device has T2 = 200 μs, single-qubit gate time 30 ns, and two-qubit gate time 300 ns. What is the maximum circuit depth for a 90% fidelity target?

**Solution:**

Step 1: Calculate coherence-limited operations
$$N_{\text{coh}} = \frac{T_2}{t_{\text{gate}}} = \frac{200 \text{ μs}}{300 \text{ ns}} \approx 667 \text{ two-qubit gates}$$

Step 2: Include gate infidelities
Assume 99.5% two-qubit fidelity ($\epsilon = 0.005$):

$$F_{\text{circuit}} = e^{-N\epsilon} \geq 0.9$$

$$N \leq -\frac{\ln(0.9)}{\epsilon} = \frac{0.105}{0.005} = 21 \text{ layers}$$

Step 3: Account for measurement (~1%)
$$N_{\text{effective}} \approx 20 \text{ two-qubit gate layers}$$

**Answer:** Maximum practical depth is approximately 20 two-qubit layers.

### Example 2: SWAP Overhead Calculation

**Problem:** Implement a CNOT between qubits 0 and 4 on a linear 5-qubit chain (0-1-2-3-4). Calculate the total gate count.

**Solution:**

Step 1: Path from qubit 0 to qubit 4
Path: 0 → 1 → 2 → 3 → 4 (length 4)

Step 2: SWAP gates needed
Need 3 SWAPs to bring qubit 0 adjacent to qubit 4:
- SWAP(0,1), SWAP(1,2), SWAP(2,3) brings logical qubit 0 to physical qubit 3

Step 3: Execute CNOT
CNOT(3,4) on physical qubits

Step 4: Reverse SWAPs (optional, depends on subsequent operations)
3 more SWAPs to restore mapping

Step 5: Total gate count
$$N_{\text{CNOT}} = 3 \times 3 + 1 + 3 \times 3 = 19 \text{ CNOTs}$$

(Or 10 CNOTs if we don't restore the mapping)

**Answer:** 10-19 CNOT gates depending on whether state restoration is needed.

### Example 3: Quantum Volume Estimation

**Problem:** A device reports QV = 128. Estimate the number of qubits and typical two-qubit gate fidelity.

**Solution:**

Step 1: Extract circuit size
$$QV = 2^n \Rightarrow n = \log_2(128) = 7$$

So the device can run 7×7 random circuits successfully.

Step 2: Estimate gate count
A random 7×7 circuit has approximately:
- Layers: 7
- Two-qubit gates per layer: ~7/2 = 3.5
- Total two-qubit gates: ~25

Step 3: Required fidelity for success
Need $F_{\text{circuit}} > e^{-1/3}$ (heuristic for heavy output)

$$F_{2Q}^{25} > 0.72$$
$$F_{2Q} > 0.72^{1/25} = 0.987$$

**Answer:** Approximately 7+ qubits with two-qubit fidelity >98.7%.

## Practice Problems

### Level 1: Direct Application

1. **Coherence calculation:** A trapped ion system has T1 = 10 s and T_φ = 2 s. Calculate T2.

2. **Fidelity decay:** How many 99.9% fidelity single-qubit gates can be applied before circuit fidelity drops below 90%?

3. **Connectivity density:** Calculate the graph density for a 27-qubit heavy-hex topology with 36 edges.

### Level 2: Intermediate Analysis

4. **SWAP optimization:** On a 4×4 grid, find the minimum number of SWAPs needed for a CNOT between corners (0,0) and (3,3).

5. **Error budget:** A VQE circuit has 50 two-qubit gates and 100 single-qubit gates. If single-qubit fidelity is 99.95% and two-qubit fidelity is 99.2%, what is the expected circuit fidelity?

6. **Platform comparison:** Compare the maximum circuit depth for:
   - Superconducting: T2 = 200 μs, gate time = 300 ns, fidelity = 99.5%
   - Trapped ion: T2 = 1 s, gate time = 200 μs, fidelity = 99.9%

### Level 3: Challenging Problems

7. **Quantum volume analysis:** Derive the relationship between quantum volume and average two-qubit gate fidelity, assuming a linear connectivity topology.

8. **Optimal routing:** Given a 5-qubit linear chain and a circuit requiring the following CNOTs: (0,4), (1,3), (2,4), (0,2), find an optimal qubit mapping that minimizes total SWAP count.

9. **Cross-platform benchmark:** Design a hardware-agnostic benchmark circuit that can fairly compare superconducting and trapped-ion systems, accounting for their different strengths.

## Computational Lab: Hardware Characterization

### Lab 1: Simulating NISQ Device Characteristics

```python
"""
Day 946 Lab: NISQ Hardware Characterization
Simulate and analyze key NISQ device metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import random_unitary, state_fidelity
import networkx as nx

# ============================================================
# Part 1: Coherence Time Simulation
# ============================================================

def simulate_t1_decay(t1_us: float, times_us: np.ndarray, n_shots: int = 1000):
    """Simulate T1 decay of excited state population."""
    backend = AerSimulator()
    excited_probs = []

    for t in times_us:
        # Create circuit: prepare |1>, wait, measure
        qc = QuantumCircuit(1, 1)
        qc.x(0)  # Prepare |1>

        # Add thermal relaxation (T1 decay)
        # Convert time to gate-equivalent delay
        error = thermal_relaxation_error(
            t1=t1_us * 1e-6,
            t2=2*t1_us * 1e-6,  # T2 <= 2*T1
            time=t * 1e-6
        )

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['id'])

        qc.id(0)  # Identity gate where we apply T1 error
        qc.measure(0, 0)

        # Execute
        job = backend.run(qc, noise_model=noise_model, shots=n_shots)
        counts = job.result().get_counts()

        # Probability of remaining in |1>
        p1 = counts.get('1', 0) / n_shots
        excited_probs.append(p1)

    return np.array(excited_probs)

def fit_t1(times: np.ndarray, probs: np.ndarray):
    """Fit exponential decay to extract T1."""
    from scipy.optimize import curve_fit

    def decay(t, t1, a):
        return a * np.exp(-t / t1)

    popt, _ = curve_fit(decay, times, probs, p0=[times.mean(), 1.0])
    return popt[0]  # T1

# Simulate T1 decay
print("Simulating T1 decay...")
t1_actual = 100  # μs
times = np.linspace(0, 300, 20)  # μs

# Note: Full simulation is slow, using analytical model for demo
probs_analytical = np.exp(-times / t1_actual)

plt.figure(figsize=(10, 6))
plt.plot(times, probs_analytical, 'b-', linewidth=2, label='T1 decay')
plt.axhline(y=1/np.e, color='r', linestyle='--', label=f'1/e (T1 = {t1_actual} μs)')
plt.xlabel('Time (μs)', fontsize=12)
plt.ylabel('Excited State Probability', fontsize=12)
plt.title('T1 Relaxation: Excited State Decay', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('t1_decay.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 2: Gate Fidelity Analysis
# ============================================================

def analyze_fidelity_scaling(gate_fidelity: float, max_gates: int):
    """Analyze how circuit fidelity scales with gate count."""
    gate_counts = np.arange(1, max_gates + 1)
    circuit_fidelities = gate_fidelity ** gate_counts

    return gate_counts, circuit_fidelities

# Compare different gate fidelities
fidelities = [0.99, 0.995, 0.999, 0.9999]
max_gates = 200

plt.figure(figsize=(10, 6))
for f in fidelities:
    gates, circuit_f = analyze_fidelity_scaling(f, max_gates)
    label = f'F = {f:.4f} (error = {(1-f)*100:.2f}%)'
    plt.plot(gates, circuit_f, linewidth=2, label=label)

plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50% fidelity threshold')
plt.xlabel('Number of Gates', fontsize=12)
plt.ylabel('Circuit Fidelity', fontsize=12)
plt.title('Circuit Fidelity Decay with Gate Count', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.savefig('fidelity_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate max useful depth for each fidelity
print("\nMaximum gates for 90% circuit fidelity:")
for f in fidelities:
    max_n = int(np.log(0.9) / np.log(f))
    print(f"  Gate fidelity {f:.4f}: {max_n} gates")

# ============================================================
# Part 3: Connectivity Graph Analysis
# ============================================================

def create_connectivity_graphs():
    """Create common NISQ connectivity topologies."""
    graphs = {}

    # Linear chain (5 qubits)
    G_linear = nx.path_graph(5)
    graphs['Linear (5Q)'] = G_linear

    # 2D grid (3x3)
    G_grid = nx.grid_2d_graph(3, 3)
    # Relabel to integer nodes
    G_grid = nx.convert_node_labels_to_integers(G_grid)
    graphs['Grid (3x3)'] = G_grid

    # Heavy-hex inspired (simplified 7Q)
    G_hex = nx.Graph()
    G_hex.add_edges_from([
        (0, 1), (1, 2), (2, 3),
        (1, 4), (4, 5), (5, 2),
        (4, 6)
    ])
    graphs['Heavy-hex (7Q)'] = G_hex

    # All-to-all (5 qubits, like trapped ions)
    G_all = nx.complete_graph(5)
    graphs['All-to-all (5Q)'] = G_all

    return graphs

def analyze_connectivity(G: nx.Graph):
    """Compute connectivity metrics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    metrics = {
        'nodes': n,
        'edges': m,
        'density': 2 * m / (n * (n - 1)) if n > 1 else 0,
        'avg_degree': 2 * m / n if n > 0 else 0,
        'avg_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
        'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf')
    }
    return metrics

# Analyze topologies
graphs = create_connectivity_graphs()

print("\nConnectivity Analysis:")
print("-" * 70)
print(f"{'Topology':<20} {'Nodes':>6} {'Edges':>6} {'Density':>8} {'Avg Deg':>8} {'Diameter':>8}")
print("-" * 70)

for name, G in graphs.items():
    m = analyze_connectivity(G)
    print(f"{name:<20} {m['nodes']:>6} {m['edges']:>6} {m['density']:>8.3f} {m['avg_degree']:>8.2f} {m['diameter']:>8.1f}")

# Visualize topologies
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, (name, G) in zip(axes, graphs.items()):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
            node_size=500, font_size=12, font_weight='bold',
            edge_color='gray', width=2)
    ax.set_title(name, fontsize=14)

plt.tight_layout()
plt.savefig('connectivity_topologies.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 4: SWAP Overhead Calculation
# ============================================================

def calculate_swap_overhead(G: nx.Graph, source: int, target: int):
    """Calculate SWAP overhead for non-adjacent CNOT."""
    if G.has_edge(source, target):
        return 0, [source, target]

    path = nx.shortest_path(G, source, target)
    n_swaps = len(path) - 2  # Don't count the final CNOT position

    return n_swaps, path

# Calculate SWAP overhead for various qubit pairs
print("\nSWAP Overhead Analysis (Linear 5Q chain):")
G_linear = graphs['Linear (5Q)']

for source, target in [(0, 4), (0, 2), (1, 4), (1, 3)]:
    swaps, path = calculate_swap_overhead(G_linear, source, target)
    print(f"  CNOT({source},{target}): {swaps} SWAPs, path = {path}")
    print(f"    Total CNOTs = {3*swaps + 1}")

# ============================================================
# Part 5: Effective Circuit Depth
# ============================================================

def calculate_effective_depth(n_qubits: int, t2_us: float,
                              gate_time_ns: float, gate_fidelity: float,
                              target_fidelity: float = 0.5):
    """Calculate effective circuit depth considering coherence and errors."""

    # Coherence limit
    depth_coherence = t2_us * 1000 / gate_time_ns  # Convert to same units

    # Error limit
    depth_error = np.log(target_fidelity) / np.log(gate_fidelity)

    # Effective depth is minimum of both limits
    depth_effective = min(depth_coherence, depth_error)

    return {
        'coherence_limit': depth_coherence,
        'error_limit': depth_error,
        'effective_depth': depth_effective,
        'limiting_factor': 'coherence' if depth_coherence < depth_error else 'error'
    }

# Compare platforms
platforms = {
    'Superconducting': {'t2_us': 200, 'gate_time_ns': 300, 'gate_fidelity': 0.995},
    'Trapped Ion': {'t2_us': 1000000, 'gate_time_ns': 200000, 'gate_fidelity': 0.999},
    'Neutral Atom': {'t2_us': 100000, 'gate_time_ns': 1000, 'gate_fidelity': 0.99}
}

print("\nEffective Circuit Depth by Platform:")
print("-" * 60)
for name, params in platforms.items():
    result = calculate_effective_depth(10, **params)
    print(f"{name}:")
    print(f"  Coherence limit: {result['coherence_limit']:.0f} gates")
    print(f"  Error limit: {result['error_limit']:.0f} gates")
    print(f"  Effective depth: {result['effective_depth']:.0f} gates ({result['limiting_factor']} limited)")

# ============================================================
# Part 6: Quantum Volume Simulation
# ============================================================

def estimate_quantum_volume(n_qubits: int, two_qubit_fidelity: float,
                           connectivity: str = 'linear'):
    """Estimate quantum volume based on hardware parameters."""

    # Estimate compilation overhead for connectivity
    if connectivity == 'all_to_all':
        swap_overhead = 1.0  # No overhead
    elif connectivity == 'grid':
        swap_overhead = 1.5  # Moderate overhead
    elif connectivity == 'linear':
        swap_overhead = 2.0  # High overhead
    else:
        swap_overhead = 1.0

    # Gates per layer in square circuit
    gates_per_layer = n_qubits * swap_overhead

    # Total two-qubit gates
    total_gates = n_qubits * gates_per_layer

    # Circuit fidelity
    circuit_fidelity = two_qubit_fidelity ** total_gates

    # Successful if fidelity > threshold (approximation)
    threshold = 0.37  # ~1/e, rough approximation

    return {
        'n': n_qubits,
        'circuit_fidelity': circuit_fidelity,
        'success': circuit_fidelity > threshold,
        'qv': 2**n_qubits if circuit_fidelity > threshold else 0
    }

# Scan for quantum volume
print("\nQuantum Volume Estimation (99.5% 2Q fidelity, linear connectivity):")
for n in range(2, 10):
    result = estimate_quantum_volume(n, 0.995, 'linear')
    status = "PASS" if result['success'] else "FAIL"
    print(f"  n={n}: F_circuit={result['circuit_fidelity']:.4f} -> {status}")
    if not result['success']:
        print(f"  Estimated QV = 2^{n-1} = {2**(n-1)}")
        break

print("\nLab complete! Generated plots saved.")
```

### Lab 2: Real Hardware Comparison (IBM Quantum)

```python
"""
Day 946 Lab Part 2: Real IBM Quantum Hardware Analysis
Query IBM Quantum backends and compare specifications
"""

# Note: Requires IBM Quantum account
# from qiskit_ibm_runtime import QiskitRuntimeService

# For demonstration, using typical hardware specs
ibm_backends = {
    'ibm_sherbrooke': {
        'qubits': 127,
        'quantum_volume': 32,
        't1_us': 264,
        't2_us': 102,
        'single_qubit_error': 0.00033,
        'two_qubit_error': 0.0091,
        'readout_error': 0.015,
        'topology': 'heavy_hex'
    },
    'ibm_brisbane': {
        'qubits': 127,
        'quantum_volume': 32,
        't1_us': 281,
        't2_us': 140,
        'single_qubit_error': 0.00026,
        'two_qubit_error': 0.0082,
        'readout_error': 0.012,
        'topology': 'heavy_hex'
    },
    'ibm_osaka': {
        'qubits': 127,
        'quantum_volume': 32,
        't1_us': 295,
        't2_us': 168,
        'single_qubit_error': 0.00024,
        'two_qubit_error': 0.0076,
        'readout_error': 0.011,
        'topology': 'heavy_hex'
    }
}

def compare_backends(backends: dict):
    """Compare IBM Quantum backends."""
    print("\nIBM Quantum Backend Comparison:")
    print("=" * 80)

    headers = ['Backend', 'Qubits', 'QV', 'T1 (μs)', 'T2 (μs)', '1Q Err', '2Q Err', 'Read Err']
    print(f"{headers[0]:<18} {headers[1]:>6} {headers[2]:>4} {headers[3]:>8} {headers[4]:>8} {headers[5]:>7} {headers[6]:>7} {headers[7]:>8}")
    print("-" * 80)

    for name, specs in backends.items():
        print(f"{name:<18} {specs['qubits']:>6} {specs['quantum_volume']:>4} "
              f"{specs['t1_us']:>8.0f} {specs['t2_us']:>8.0f} "
              f"{specs['single_qubit_error']:>7.4f} {specs['two_qubit_error']:>7.4f} "
              f"{specs['readout_error']:>8.3f}")

    # Calculate effective circuit depth for each
    print("\nEffective Circuit Depth (50% fidelity target):")
    for name, specs in backends.items():
        f_2q = 1 - specs['two_qubit_error']
        max_gates = int(np.log(0.5) / np.log(f_2q))
        coherence_gates = int(specs['t2_us'] * 1000 / 500)  # Assume 500ns 2Q gate
        effective = min(max_gates, coherence_gates)
        print(f"  {name}: ~{effective} two-qubit gates")

compare_backends(ibm_backends)

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

backends_names = list(ibm_backends.keys())
t1_values = [b['t1_us'] for b in ibm_backends.values()]
t2_values = [b['t2_us'] for b in ibm_backends.values()]
error_2q = [b['two_qubit_error'] for b in ibm_backends.values()]

axes[0].bar(backends_names, t1_values, color='steelblue', alpha=0.7)
axes[0].bar(backends_names, t2_values, color='coral', alpha=0.7)
axes[0].set_ylabel('Time (μs)')
axes[0].set_title('Coherence Times (T1 blue, T2 orange)')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(backends_names, [e*100 for e in error_2q], color='crimson', alpha=0.7)
axes[1].set_ylabel('Error Rate (%)')
axes[1].set_title('Two-Qubit Gate Error Rate')
axes[1].tick_params(axis='x', rotation=45)

# Effective depth comparison
effective_depths = []
for specs in ibm_backends.values():
    f_2q = 1 - specs['two_qubit_error']
    max_gates = int(np.log(0.5) / np.log(f_2q))
    effective_depths.append(max_gates)

axes[2].bar(backends_names, effective_depths, color='forestgreen', alpha=0.7)
axes[2].set_ylabel('Max Two-Qubit Gates')
axes[2].set_title('Effective Circuit Depth')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ibm_backend_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAnalysis complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| T2 Bound | $\frac{1}{T_2} \leq \frac{1}{2T_1} + \frac{1}{T_\phi}$ |
| Circuit Fidelity | $F_{\text{circuit}} \approx e^{-N\epsilon}$ |
| Max Depth | $d_{\max} \approx \min\left(\frac{T_2}{t_{\text{gate}}}, \frac{1}{\epsilon}\right)$ |
| Graph Density | $\rho = \frac{2|E|}{|V|(|V|-1)}$ |
| Quantum Volume | $QV = 2^n$ for largest successful $n \times n$ circuit |
| SWAP Cost | 3 CNOTs per SWAP |

### Key Takeaways

1. **NISQ devices are fundamentally limited** by coherence times, gate fidelities, and connectivity.

2. **Gate fidelity dominates** circuit depth limits on current hardware (error-limited, not coherence-limited).

3. **Connectivity matters enormously** - SWAP overhead can increase gate count by 3-10x for distant qubits.

4. **Quantum Volume** provides a holistic benchmark but doesn't capture all aspects of hardware capability.

5. **Different platforms have different trade-offs:**
   - Superconducting: Fast but noisy, limited connectivity
   - Trapped ions: High fidelity but slow, all-to-all connectivity
   - Neutral atoms: Large scale, reconfigurable, intermediate fidelity

## Daily Checklist

- [ ] I can define the NISQ era and its key characteristics
- [ ] I can calculate T1 and T2 decay for quantum systems
- [ ] I understand how gate fidelity impacts circuit depth
- [ ] I can analyze connectivity graphs and calculate SWAP overhead
- [ ] I can interpret quantum volume measurements
- [ ] I completed the computational lab and generated visualizations
- [ ] I can compare different NISQ hardware platforms

## Preview of Day 947

Tomorrow we dive into the **Variational Quantum Eigensolver (VQE)** - the flagship NISQ algorithm. We will:
- Design parameterized quantum circuits (ansatzes)
- Implement VQE for finding molecular ground states
- Explore the H₂ molecule as our first chemical system
- Learn the parameter shift rule for gradient computation
- Optimize variational parameters using classical methods

VQE demonstrates how NISQ constraints directly shape algorithm design, making it the perfect application of today's hardware characterization.
