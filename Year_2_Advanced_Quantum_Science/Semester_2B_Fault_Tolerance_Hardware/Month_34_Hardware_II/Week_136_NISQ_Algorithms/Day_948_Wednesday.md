# Day 948: QAOA for Optimization

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 2.5 hours | QAOA theory, Hamiltonians, and approximation guarantees |
| Afternoon | 2.5 hours | Problem solving and depth analysis |
| Evening | 2 hours | Computational lab: QAOA for MaxCut |

## Learning Objectives

By the end of today, you will be able to:

1. Formulate combinatorial optimization problems as Ising/QUBO Hamiltonians
2. Construct cost and mixer Hamiltonians for QAOA
3. Implement the QAOA circuit structure with parameterized layers
4. Analyze approximation ratios and their dependence on circuit depth
5. Solve MaxCut problems using QAOA on graphs
6. Compare QAOA performance against classical heuristics

## Core Content

### 1. Combinatorial Optimization on Quantum Computers

Many important problems are NP-hard combinatorial optimization:

$$\boxed{\text{Goal: Find } \mathbf{x}^* = \arg\max_{\mathbf{x} \in \{0,1\}^n} C(\mathbf{x})}$$

where $C: \{0,1\}^n \rightarrow \mathbb{R}$ is the objective function.

**Examples:**
- **MaxCut:** Partition graph vertices to maximize cut edges
- **MAX-SAT:** Maximize satisfied clauses in Boolean formula
- **Traveling Salesman:** Find shortest tour visiting all cities
- **Portfolio Optimization:** Maximize returns subject to risk constraints

### 2. Problem Encoding: Cost Hamiltonians

Binary optimization problems map to diagonal Hamiltonians:

$$\hat{C} = \sum_i c_i \hat{Z}_i + \sum_{i<j} c_{ij} \hat{Z}_i \hat{Z}_j + \ldots$$

The computational basis states encode solutions:

$$\hat{C}|x_1 x_2 \ldots x_n\rangle = C(x_1, x_2, \ldots, x_n)|x_1 x_2 \ldots x_n\rangle$$

**Mapping convention:**
$$x_i = 0 \leftrightarrow |0\rangle, \quad x_i = 1 \leftrightarrow |1\rangle$$

or using spin variables $s_i = 1 - 2x_i$:
$$s_i = +1 \leftrightarrow |0\rangle, \quad s_i = -1 \leftrightarrow |1\rangle$$

### 3. MaxCut Problem

**Definition:** Given graph $G = (V, E)$ with edge weights $w_{ij}$, find partition $(S, \bar{S})$ maximizing:

$$C(S) = \sum_{(i,j) \in E: i \in S, j \in \bar{S}} w_{ij}$$

**Cost Hamiltonian:**

$$\boxed{\hat{C} = \sum_{(i,j) \in E} \frac{w_{ij}}{2}(1 - \hat{Z}_i \hat{Z}_j)}$$

**Intuition:**
- $\hat{Z}_i\hat{Z}_j|x_i x_j\rangle = |x_i x_j\rangle$ if $x_i = x_j$ (same partition)
- $\hat{Z}_i\hat{Z}_j|x_i x_j\rangle = -|x_i x_j\rangle$ if $x_i \neq x_j$ (different partitions)

So $(1 - Z_iZ_j)/2$ gives 1 if edge is cut, 0 otherwise.

### 4. QAOA Algorithm Structure

QAOA prepares a variational state through alternating application of:

1. **Cost unitary:** $U_C(\gamma) = e^{-i\gamma\hat{C}}$
2. **Mixer unitary:** $U_B(\beta) = e^{-i\beta\hat{B}}$

**Standard mixer:**
$$\hat{B} = \sum_{i=1}^n \hat{X}_i$$

**QAOA state at depth $p$:**

$$\boxed{|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = \prod_{l=1}^{p} U_B(\beta_l) U_C(\gamma_l) |+\rangle^{\otimes n}}$$

where $|+\rangle = H|0\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$.

**Algorithm:**
1. Initialize $|+\rangle^{\otimes n}$ (uniform superposition over all solutions)
2. Apply $p$ layers of $U_C(\gamma_l)$ and $U_B(\beta_l)$
3. Measure in computational basis
4. Optimize $\boldsymbol{\gamma}, \boldsymbol{\beta}$ to maximize $\langle\hat{C}\rangle$

### 5. QAOA Circuit Implementation

**For MaxCut, the cost unitary decomposes as:**

$$U_C(\gamma) = \prod_{(i,j) \in E} e^{-i\gamma\frac{w_{ij}}{2}(1 - Z_iZ_j)} = \prod_{(i,j) \in E} e^{i\gamma\frac{w_{ij}}{2}} e^{-i\gamma\frac{w_{ij}}{2}Z_iZ_j}$$

The $ZZ$ interaction implements:

$$e^{-i\theta Z_i Z_j} = \text{CNOT}_{ij} \cdot R_Z(2\theta)_j \cdot \text{CNOT}_{ij}$$

**Circuit diagram for one QAOA layer (3-node triangle):**

```
        ┌───┐┌─────────┐     ┌───┐                    ┌──────────┐
q0: |+⟩─┤   ├┤ Rz(2γ) ├──■──┤   ├────────────────■──┤ Rx(2β)  ├
        │   │└─────────┘  │  │   │                │  └──────────┘
        │ZZ │           ┌─┴─┐│ZZ │┌─────────┐   ┌─┴─┐┌──────────┐
q1: |+⟩─┤   ├───────────┤ X ├┤   ├┤ Rz(2γ) ├───┤ X ├┤ Rx(2β)  ├
        │   │           └───┘│   │└─────────┘   └───┘└──────────┘
        │   │                │   │                    ┌──────────┐
q2: |+⟩─┤   ├────────────────┤   ├────────────────────┤ Rx(2β)  ├
        └───┘                └───┘                    └──────────┘
```

**Mixer unitary:**
$$U_B(\beta) = \prod_{i=1}^n R_X(2\beta)_i$$

### 6. Approximation Ratio Analysis

The **approximation ratio** measures solution quality:

$$\boxed{r = \frac{\langle\boldsymbol{\gamma}, \boldsymbol{\beta}|\hat{C}|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle}{C_{\max}}}$$

**Theoretical results for MaxCut:**

| Depth $p$ | Approximation Ratio | Notes |
|-----------|---------------------|-------|
| $p = 1$ | $\geq 0.6924$ | For 3-regular graphs |
| $p \rightarrow \infty$ | $\rightarrow 1$ | Adiabatic limit |
| Classical best | 0.878 | Goemans-Williamson SDP |

**Farhi et al. (2014) result for 3-regular graphs at $p=1$:**

$$r_1 = \frac{1}{2} + \frac{1}{4\sqrt{2}} \approx 0.6924$$

### 7. Depth Scaling and Performance

**Key trade-offs:**

| Factor | Shallow ($p$ small) | Deep ($p$ large) |
|--------|---------------------|------------------|
| Parameters | $2p$ | $2p$ |
| Circuit depth | $O(p \cdot |E|)$ | $O(p \cdot |E|)$ |
| Approx. ratio | Lower | Higher |
| Trainability | Easier | Barren plateaus |
| Noise resilience | Better | Worse |

**Empirical observation:** For many problems, $p = 3-10$ gives good performance on NISQ devices.

### 8. QAOA Variants

**QAOA with Warm Start:**
Initialize with classically-obtained solution instead of $|+\rangle^{\otimes n}$.

**Multi-angle QAOA (ma-QAOA):**
Use different angles for each gate: $O(p \cdot |E|)$ parameters instead of $2p$.

**QAOA with Constraints (QAOA+):**
Modified mixer to preserve feasibility for constrained problems.

**Recursive QAOA (RQAOA):**
Iteratively fix variables based on measurement outcomes.

## Quantum Computing Applications

### Application: Network Design

QAOA can solve network optimization problems:
- Minimum vertex cover for sensor placement
- Graph coloring for frequency assignment
- Maximum independent set for resource allocation

### Application: Finance

Portfolio optimization as QUBO:
$$\min_{\mathbf{x}} \mathbf{x}^T \Sigma \mathbf{x} - \lambda \boldsymbol{\mu}^T \mathbf{x}$$

where $\Sigma$ is the covariance matrix and $\boldsymbol{\mu}$ are expected returns.

### Application: Machine Learning

QAOA for clustering and classification:
- MaxCut formulation for graph bisection
- Community detection in networks
- Feature selection

## Worked Examples

### Example 1: MaxCut Hamiltonian Construction

**Problem:** Construct the cost Hamiltonian for MaxCut on a 4-node cycle graph with unit weights.

**Solution:**

Graph edges: $E = \{(0,1), (1,2), (2,3), (3,0)\}$

Step 1: Write cost Hamiltonian
$$\hat{C} = \sum_{(i,j) \in E} \frac{1}{2}(1 - Z_i Z_j)$$

Step 2: Expand terms
$$\hat{C} = \frac{1}{2}(1 - Z_0 Z_1) + \frac{1}{2}(1 - Z_1 Z_2) + \frac{1}{2}(1 - Z_2 Z_3) + \frac{1}{2}(1 - Z_3 Z_0)$$

Step 3: Simplify
$$\hat{C} = 2I - \frac{1}{2}(Z_0 Z_1 + Z_1 Z_2 + Z_2 Z_3 + Z_3 Z_0)$$

Step 4: Verify eigenvalues
- $|0000\rangle, |1111\rangle$: All $ZZ = +1$, so $C = 2 - 2 = 0$ (no cut edges)
- $|0101\rangle, |1010\rangle$: All $ZZ = -1$, so $C = 2 + 2 = 4$ (all edges cut)

**Answer:** Maximum cut value is 4, achieved by alternating partitions.

### Example 2: QAOA Parameter Optimization

**Problem:** For a single edge $(0,1)$ with $p=1$, find the optimal QAOA parameters.

**Solution:**

Step 1: Cost Hamiltonian
$$\hat{C} = \frac{1}{2}(I - Z_0 Z_1)$$

Step 2: QAOA state
$$|\gamma, \beta\rangle = e^{-i\beta(X_0 + X_1)} e^{-i\gamma\hat{C}} |++\rangle$$

Step 3: Compute expectation
After algebra (using $e^{-i\theta ZZ}$ decomposition):

$$\langle\hat{C}\rangle = \frac{1}{2}(1 - \cos(2\gamma)\cos^2(2\beta) - \sin(2\gamma)\sin(4\beta)/2)$$

Step 4: Optimize
Taking derivatives and solving:
$$\gamma^* = \pi/4, \quad \beta^* = \pi/8$$

Step 5: Verify
$$\langle\hat{C}\rangle^* = \frac{1}{2}(1 + \frac{1}{\sqrt{2}}) \approx 0.854$$

Since $C_{\max} = 1$, approximation ratio $r \approx 0.854$.

### Example 3: Depth Scaling Analysis

**Problem:** For a random 3-regular graph with $n=20$ vertices, estimate the circuit depth for QAOA at $p=3$.

**Solution:**

Step 1: Count edges
3-regular graph: each vertex has degree 3
Total edges: $|E| = \frac{3n}{2} = \frac{3 \times 20}{2} = 30$

Step 2: Gates per layer
- Cost unitary: 30 $ZZ$ gates (each = 2 CNOTs + 1 $R_Z$)
- Mixer unitary: 20 $R_X$ gates

Step 3: CNOT count per layer
$30 \times 2 = 60$ CNOTs + 30 single-qubit = 90 gates

Step 4: Total for $p=3$
$3 \times 90 + 3 \times 20 = 270 + 60 = 330$ gates

Step 5: Depth analysis (assuming linear connectivity)
With SWAP overhead factor ~2: effective depth $\approx 660$ gates

**Answer:** ~660 native gates, likely requiring transpilation optimization.

## Practice Problems

### Level 1: Direct Application

1. **Hamiltonian construction:** Write the MaxCut cost Hamiltonian for a triangle graph (3 vertices, 3 edges).

2. **QAOA circuit:** Draw the QAOA circuit for 2 qubits with a single edge at $p=1$.

3. **Approximation ratio:** If QAOA achieves $\langle C\rangle = 5.5$ on a graph with $C_{\max} = 7$, what is the approximation ratio?

### Level 2: Intermediate Analysis

4. **Problem encoding:** Encode the MAX-2-SAT clause $(x_1 \vee x_2)$ as an Ising Hamiltonian.

5. **Parameter counting:** For QAOA on a complete graph $K_n$ at depth $p$, how many parameters are there? How many gates?

6. **Mixer design:** For the graph coloring problem (no adjacent vertices same color), explain why the standard $\sum X_i$ mixer might violate constraints and propose an alternative.

### Level 3: Challenging Problems

7. **Approximation analysis:** Prove that for any graph, QAOA with $p=1$ achieves approximation ratio $\geq 1/2$ for MaxCut.

8. **Symmetry exploitation:** A graph with vertex-transitive symmetry has the same local structure at each vertex. How can this symmetry reduce the number of QAOA parameters?

9. **Noise impact:** If each CNOT has error rate $\epsilon = 0.01$, estimate the effective approximation ratio degradation for QAOA at $p=3$ on a 10-node complete graph.

## Computational Lab: QAOA for MaxCut

### Lab 1: Complete QAOA Implementation

```python
"""
Day 948 Lab: QAOA for MaxCut
Complete implementation with analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import SparsePauliOp

# ============================================================
# Part 1: Graph and Hamiltonian Construction
# ============================================================

def create_random_graph(n_nodes: int, edge_prob: float = 0.5,
                        seed: int = 42) -> nx.Graph:
    """Create random Erdos-Renyi graph."""
    return nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)

def create_regular_graph(n_nodes: int, degree: int,
                         seed: int = 42) -> nx.Graph:
    """Create random regular graph."""
    return nx.random_regular_graph(degree, n_nodes, seed=seed)

def maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """
    Construct MaxCut cost Hamiltonian.
    C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2
    """
    n = graph.number_of_nodes()
    pauli_list = []

    for i, j in graph.edges():
        # Identity term: contributes 1/2 per edge
        # ZZ term: -1/2 * Z_i Z_j
        zz_string = ['I'] * n
        zz_string[i] = 'Z'
        zz_string[j] = 'Z'
        pauli_list.append((''.join(zz_string), -0.5))

    # Add constant offset (sum of 1/2 over edges)
    identity_string = 'I' * n
    pauli_list.append((identity_string, 0.5 * graph.number_of_edges()))

    return SparsePauliOp.from_list(pauli_list)

# Create test graph
n_nodes = 6
graph = create_random_graph(n_nodes, edge_prob=0.6, seed=42)

print("Graph properties:")
print(f"  Nodes: {graph.number_of_nodes()}")
print(f"  Edges: {graph.number_of_edges()}")
print(f"  Edge list: {list(graph.edges())}")

# Visualize graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(graph, seed=42)
nx.draw(graph, pos, with_labels=True, node_color='lightblue',
        node_size=700, font_size=14, font_weight='bold',
        edge_color='gray', width=2)
plt.title(f'Random Graph ({n_nodes} nodes)', fontsize=14)
plt.savefig('qaoa_graph.png', dpi=150, bbox_inches='tight')
plt.show()

# Create Hamiltonian
H = maxcut_hamiltonian(graph)
print(f"\nHamiltonian terms: {len(H)}")

# ============================================================
# Part 2: QAOA Circuit Construction
# ============================================================

def qaoa_circuit(graph: nx.Graph, gamma: list, beta: list) -> QuantumCircuit:
    """
    Construct QAOA circuit for MaxCut.

    Parameters:
        graph: NetworkX graph
        gamma: Cost layer parameters
        beta: Mixer layer parameters
    """
    n = graph.number_of_nodes()
    p = len(gamma)  # QAOA depth

    qc = QuantumCircuit(n)

    # Initial superposition
    for i in range(n):
        qc.h(i)

    # QAOA layers
    for layer in range(p):
        # Cost unitary: exp(-i * gamma * C)
        for i, j in graph.edges():
            # ZZ interaction: exp(-i * gamma * (1-ZZ)/2)
            # = exp(-i * gamma/2) * exp(i * gamma/2 * ZZ)
            qc.cx(i, j)
            qc.rz(gamma[layer], j)
            qc.cx(i, j)

        # Mixer unitary: exp(-i * beta * B) where B = sum_i X_i
        for i in range(n):
            qc.rx(2 * beta[layer], i)

    return qc

# Create example QAOA circuit
p = 2  # QAOA depth
gamma_test = [0.5, 0.3]
beta_test = [0.4, 0.2]
qc = qaoa_circuit(graph, gamma_test, beta_test)

print(f"\nQAOA Circuit (p={p}):")
print(f"  Depth: {qc.depth()}")
print(f"  Gate count: {qc.count_ops()}")
print(qc.draw(output='text', fold=100))

# ============================================================
# Part 3: QAOA Optimization
# ============================================================

def qaoa_cost(params: np.ndarray, graph: nx.Graph, p: int,
              estimator: Estimator) -> float:
    """
    Compute QAOA expectation value of cost Hamiltonian.
    """
    gamma = params[:p]
    beta = params[p:]

    qc = qaoa_circuit(graph, gamma.tolist(), beta.tolist())
    H = maxcut_hamiltonian(graph)

    job = estimator.run([(qc, H)])
    result = job.result()

    return -float(result[0].data.evs)  # Negative for minimization

def run_qaoa(graph: nx.Graph, p: int, n_trials: int = 5,
             seed: int = 42) -> dict:
    """
    Run QAOA optimization with multiple random starts.
    """
    np.random.seed(seed)
    estimator = Estimator()

    best_result = None
    all_results = []

    for trial in range(n_trials):
        # Random initialization
        init_params = np.random.uniform(0, np.pi, 2 * p)

        # Optimize
        result = minimize(
            qaoa_cost,
            init_params,
            args=(graph, p, estimator),
            method='COBYLA',
            options={'maxiter': 100}
        )

        all_results.append({
            'params': result.x,
            'cost': -result.fun,  # Convert back to maximization
            'success': result.success
        })

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return {
        'best_params': best_result.x,
        'best_cost': -best_result.fun,
        'gamma': best_result.x[:p],
        'beta': best_result.x[p:],
        'all_results': all_results
    }

# Run QAOA for different depths
print("\n" + "="*60)
print("Running QAOA optimization")
print("="*60)

depths = [1, 2, 3]
qaoa_results = {}

for p in depths:
    print(f"\nQAOA depth p = {p}...")
    qaoa_results[p] = run_qaoa(graph, p, n_trials=3)
    print(f"  Best cost: {qaoa_results[p]['best_cost']:.4f}")
    print(f"  Optimal γ: {qaoa_results[p]['gamma']}")
    print(f"  Optimal β: {qaoa_results[p]['beta']}")

# ============================================================
# Part 4: Classical Comparison
# ============================================================

def brute_force_maxcut(graph: nx.Graph) -> tuple:
    """Find optimal MaxCut by brute force."""
    n = graph.number_of_nodes()
    best_cut = 0
    best_partition = None

    for i in range(2**n):
        partition = [int(b) for b in format(i, f'0{n}b')]
        cut = sum(1 for (u, v) in graph.edges()
                  if partition[u] != partition[v])

        if cut > best_cut:
            best_cut = cut
            best_partition = partition

    return best_cut, best_partition

optimal_cut, optimal_partition = brute_force_maxcut(graph)
print(f"\nOptimal MaxCut: {optimal_cut}")
print(f"Optimal partition: {optimal_partition}")

# Approximation ratios
print("\nApproximation ratios:")
for p in depths:
    ratio = qaoa_results[p]['best_cost'] / optimal_cut
    print(f"  p={p}: {ratio:.4f}")

# ============================================================
# Part 5: Sample Solutions from QAOA
# ============================================================

def sample_qaoa_solutions(graph: nx.Graph, gamma: list, beta: list,
                          n_shots: int = 1000) -> dict:
    """Sample solutions from QAOA circuit."""
    qc = qaoa_circuit(graph, gamma, beta)
    qc.measure_all()

    sampler = Sampler()
    job = sampler.run([qc], shots=n_shots)
    result = job.result()

    counts = result[0].data.meas.get_counts()

    # Evaluate cut values for each solution
    solutions = {}
    for bitstring, count in counts.items():
        partition = [int(b) for b in bitstring[::-1]]  # Reverse for qubit ordering
        cut = sum(1 for (u, v) in graph.edges()
                  if partition[u] != partition[v])
        solutions[bitstring] = {'count': count, 'cut': cut}

    return solutions

# Sample from best QAOA solution
best_p = max(depths, key=lambda p: qaoa_results[p]['best_cost'])
solutions = sample_qaoa_solutions(
    graph,
    qaoa_results[best_p]['gamma'].tolist(),
    qaoa_results[best_p]['beta'].tolist(),
    n_shots=1000
)

print(f"\nTop solutions from QAOA (p={best_p}):")
sorted_solutions = sorted(solutions.items(),
                          key=lambda x: x[1]['cut'], reverse=True)
for bitstring, info in sorted_solutions[:5]:
    print(f"  {bitstring}: cut={info['cut']}, count={info['count']}")

# ============================================================
# Part 6: Visualize Optimization Landscape
# ============================================================

def compute_landscape(graph: nx.Graph, gamma_range: np.ndarray,
                      beta_range: np.ndarray) -> np.ndarray:
    """Compute QAOA cost landscape for p=1."""
    estimator = Estimator()
    landscape = np.zeros((len(gamma_range), len(beta_range)))

    for i, gamma in enumerate(gamma_range):
        for j, beta in enumerate(beta_range):
            params = np.array([gamma, beta])
            landscape[i, j] = -qaoa_cost(params, graph, 1, estimator)

    return landscape

print("\nComputing p=1 landscape...")
gamma_range = np.linspace(0, np.pi, 25)
beta_range = np.linspace(0, np.pi/2, 25)
landscape = compute_landscape(graph, gamma_range, beta_range)

# Plot landscape
plt.figure(figsize=(10, 8))
plt.contourf(beta_range, gamma_range, landscape, levels=20, cmap='viridis')
plt.colorbar(label='Expected Cut')
plt.xlabel(r'$\beta$', fontsize=14)
plt.ylabel(r'$\gamma$', fontsize=14)
plt.title('QAOA Cost Landscape (p=1)', fontsize=14)

# Mark optimal point
opt_gamma, opt_beta = qaoa_results[1]['gamma'][0], qaoa_results[1]['beta'][0]
plt.plot(opt_beta, opt_gamma, 'r*', markersize=15, label='Optimal')
plt.legend()
plt.savefig('qaoa_landscape.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 7: Depth Scaling Analysis
# ============================================================

def analyze_depth_scaling(n_nodes: int, max_depth: int = 5) -> dict:
    """Analyze how QAOA performance scales with depth."""
    graph = create_random_graph(n_nodes, edge_prob=0.5, seed=42)
    optimal_cut, _ = brute_force_maxcut(graph)

    results = {'depths': [], 'approx_ratios': [], 'gate_counts': []}

    for p in range(1, max_depth + 1):
        qaoa_result = run_qaoa(graph, p, n_trials=2)
        ratio = qaoa_result['best_cost'] / optimal_cut

        qc = qaoa_circuit(graph, [0]*p, [0]*p)

        results['depths'].append(p)
        results['approx_ratios'].append(ratio)
        results['gate_counts'].append(sum(qc.count_ops().values()))

    return results

print("\nDepth scaling analysis...")
scaling = analyze_depth_scaling(n_nodes=6, max_depth=4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(scaling['depths'], scaling['approx_ratios'], 'bo-',
         markersize=10, linewidth=2)
ax1.axhline(y=0.878, color='r', linestyle='--', label='GW bound')
ax1.set_xlabel('QAOA Depth (p)', fontsize=12)
ax1.set_ylabel('Approximation Ratio', fontsize=12)
ax1.set_title('QAOA Performance vs Depth', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(scaling['depths'], scaling['gate_counts'], 'go-',
         markersize=10, linewidth=2)
ax2.set_xlabel('QAOA Depth (p)', fontsize=12)
ax2.set_ylabel('Gate Count', fontsize=12)
ax2.set_title('Circuit Complexity vs Depth', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qaoa_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab complete!")
```

### Lab 2: QAOA with PennyLane

```python
"""
Day 948 Lab Part 2: QAOA with PennyLane
Clean implementation with autodiff
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================
# Part 1: Graph Setup
# ============================================================

# Create graph
graph = nx.Graph()
graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
n_nodes = 4

print(f"Graph: {n_nodes} nodes, {graph.number_of_edges()} edges")

# ============================================================
# Part 2: QAOA in PennyLane
# ============================================================

dev = qml.device('default.qubit', wires=n_nodes)

def cost_layer(gamma):
    """Apply cost unitary for MaxCut."""
    for i, j in graph.edges():
        qml.CNOT(wires=[i, j])
        qml.RZ(gamma, wires=j)
        qml.CNOT(wires=[i, j])

def mixer_layer(beta):
    """Apply mixer unitary."""
    for i in range(n_nodes):
        qml.RX(2 * beta, wires=i)

@qml.qnode(dev)
def qaoa_circuit(params):
    """QAOA circuit returning cost expectation."""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    # Initial superposition
    for i in range(n_nodes):
        qml.Hadamard(wires=i)

    # QAOA layers
    for layer in range(p):
        cost_layer(gammas[layer])
        mixer_layer(betas[layer])

    # Cost Hamiltonian
    H = qml.Hamiltonian(
        [0.5 for _ in graph.edges()],
        [qml.Identity(0) - qml.PauliZ(i) @ qml.PauliZ(j)
         for i, j in graph.edges()]
    )

    return qml.expval(H)

# ============================================================
# Part 3: Optimization
# ============================================================

p = 2  # QAOA depth
n_params = 2 * p

# Initialize
np.random.seed(42)
params = np.random.uniform(0, np.pi, n_params, requires_grad=True)

# Optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# Training loop
n_steps = 80
costs = []

print("\nOptimizing QAOA...")
for step in range(n_steps):
    params, cost = opt.step_and_cost(lambda p: -qaoa_circuit(p), params)
    costs.append(-cost)

    if (step + 1) % 20 == 0:
        print(f"Step {step+1}: Cost = {-cost:.4f}")

print(f"\nFinal cost: {costs[-1]:.4f}")
print(f"Optimal γ: {params[:p]}")
print(f"Optimal β: {params[p:]}")

# Brute force check
max_cut = 0
for i in range(2**n_nodes):
    partition = [int(b) for b in format(i, f'0{n_nodes}b')]
    cut = sum(1 for (u, v) in graph.edges() if partition[u] != partition[v])
    max_cut = max(max_cut, cut)

print(f"Optimal cut: {max_cut}")
print(f"Approximation ratio: {costs[-1]/max_cut:.4f}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(costs, 'b-', linewidth=2)
plt.axhline(y=max_cut, color='r', linestyle='--', label=f'Optimal = {max_cut}')
plt.xlabel('Optimization Step', fontsize=12)
plt.ylabel('Expected Cut', fontsize=12)
plt.title('PennyLane QAOA Convergence', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pennylane_qaoa.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 4: Sample Bitstrings
# ============================================================

@qml.qnode(dev)
def qaoa_probs(params):
    """Return probability distribution over bitstrings."""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    for i in range(n_nodes):
        qml.Hadamard(wires=i)

    for layer in range(p):
        cost_layer(gammas[layer])
        mixer_layer(betas[layer])

    return qml.probs(wires=range(n_nodes))

# Get probabilities
probs = qaoa_probs(params)

# Find best solutions
print("\nTop solutions by probability:")
sorted_idx = np.argsort(probs)[::-1]
for idx in sorted_idx[:5]:
    bitstring = format(idx, f'0{n_nodes}b')
    partition = [int(b) for b in bitstring]
    cut = sum(1 for (u, v) in graph.edges() if partition[u] != partition[v])
    print(f"  {bitstring}: prob={probs[idx]:.4f}, cut={cut}")

# ============================================================
# Part 5: Gradient Analysis
# ============================================================

# Compute gradient at optimal point
grad_fn = qml.grad(lambda p: -qaoa_circuit(p))
gradient = grad_fn(params)

print("\nGradient at optimum:")
print(f"  |∇| = {np.linalg.norm(gradient):.6f}")

print("\nLab complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| MaxCut Hamiltonian | $\hat{C} = \sum_{(i,j) \in E} \frac{w_{ij}}{2}(1 - Z_i Z_j)$ |
| QAOA State | $\|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = \prod_{l=1}^{p} U_B(\beta_l) U_C(\gamma_l) \|+\rangle^{\otimes n}$ |
| Cost Unitary | $U_C(\gamma) = e^{-i\gamma\hat{C}}$ |
| Mixer Unitary | $U_B(\beta) = e^{-i\beta\sum_i X_i}$ |
| Approximation Ratio | $r = \langle\hat{C}\rangle / C_{\max}$ |
| ZZ Gate | $e^{-i\theta Z_i Z_j} = \text{CNOT} \cdot R_Z(2\theta) \cdot \text{CNOT}$ |

### Key Takeaways

1. **QAOA is variational optimization** for combinatorial problems encoded as diagonal Hamiltonians.

2. **The ansatz alternates** between cost evolution (encodes problem) and mixer evolution (explores solution space).

3. **Depth $p$ controls** approximation quality - deeper circuits give better solutions but are harder to train.

4. **MaxCut is the canonical** QAOA problem with theoretical guarantees at $p=1$ for regular graphs.

5. **Circuit depth scales** as $O(p \cdot |E|)$, making dense graphs challenging for NISQ devices.

6. **Classical comparison** is essential - QAOA must outperform Goemans-Williamson (0.878) to be useful.

## Daily Checklist

- [ ] I can formulate combinatorial problems as Ising Hamiltonians
- [ ] I understand the QAOA circuit structure with cost and mixer unitaries
- [ ] I can implement QAOA for MaxCut problems
- [ ] I understand approximation ratios and their depth dependence
- [ ] I completed the computational labs with working QAOA code
- [ ] I can analyze QAOA performance against classical baselines

## Preview of Day 949

Tomorrow we confront a critical challenge in variational quantum algorithms: **Barren Plateaus**. We will explore:
- How gradient magnitudes vanish exponentially with circuit depth/width
- The trade-off between expressibility and trainability
- Circuit designs that avoid barren plateaus
- Initialization strategies for variational optimization
- Connection to quantum chaos and t-design theory

Understanding barren plateaus is essential for designing practical NISQ algorithms.
