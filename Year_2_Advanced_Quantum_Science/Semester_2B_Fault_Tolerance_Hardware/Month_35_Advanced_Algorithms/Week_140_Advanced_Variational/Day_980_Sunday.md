# Day 980: Month 35 Synthesis & Capstone Preview

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Month 35 Review & Integration |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Capstone Project Planning |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Integration Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 980, you will be able to:

1. Synthesize all Month 35 algorithms into a coherent framework
2. Compare and select algorithms for specific applications
3. Evaluate trade-offs between fault-tolerant and NISQ approaches
4. Design research-level quantum algorithm implementations
5. Outline a capstone project integrating Year 2 material
6. Identify open problems and research frontiers

---

## Core Content

### 1. Month 35 Algorithm Synthesis

This month covered four major algorithmic themes:

**Week 137: HHL Algorithm & Quantum Linear Algebra**
- Solving $Ax = b$ with potential exponential speedup
- Condition number dependence: $O(\kappa^2 \log N)$
- Challenges: state preparation, readout, dequantization

**Week 138: Quantum Simulation**
- Trotterization and product formulas
- Quantum signal processing and block encoding
- Near-term: variational quantum simulation

**Week 139: Quantum Machine Learning**
- Quantum feature maps and kernels
- Variational classifiers
- Expressibility vs trainability

**Week 140: Advanced Variational Methods**
- ADAPT-VQE for compact ansatze
- Symmetry preservation
- Barren plateau mitigation
- Error-mitigated VQE

---

### 2. Algorithm Selection Framework

When faced with a quantum computing problem, use this decision tree:

```
Is the problem naturally quantum?
├── YES (Hamiltonian simulation, etc.)
│   ├── High precision needed? → Fault-tolerant (QSVT, QPE)
│   └── NISQ hardware? → Variational (VQE, VQS)
│
├── NO (Classical problem: optimization, ML)
│   ├── Provable speedup exists? → Implement carefully
│   └── Heuristic advantage? → Benchmark empirically
│
└── HYBRID (Best of both worlds)
    └── Variational with error mitigation
```

**Decision Criteria:**

| Factor | Fault-Tolerant | NISQ/Variational |
|--------|----------------|------------------|
| Hardware | Future | Available now |
| Circuit depth | Deep OK | Must be shallow |
| Accuracy | Guaranteed | Approximate |
| Scaling | Polynomial in precision | Limited by noise |
| Development | Theory-focused | Implementation-focused |

---

### 3. Comparative Algorithm Analysis

**For Molecular Ground States:**

| Method | Qubits | Depth | Accuracy | Status |
|--------|--------|-------|----------|--------|
| UCCSD | O(N^4) | Very high | Chemical | Impractical on NISQ |
| ADAPT-VQE | O(N^2) adaptive | Low-medium | Chemical | State of the art |
| QPE | O(N) + ancilla | O(1/ε) | Arbitrary | Future hardware |
| DMRG-inspired | O(N) | O(N) | Chemical | Research frontier |

**For Optimization Problems:**

| Method | Type | Speedup | Practical |
|--------|------|---------|-----------|
| QAOA | Variational | Debated | Yes (small) |
| Grover-based | Fault-tolerant | √N | Future |
| Adiabatic | Hardware-specific | Problem-dependent | Special devices |
| VQE-inspired | Variational | Heuristic | Yes |

**For Machine Learning:**

| Method | Quantum | Classical Competition | Verdict |
|--------|---------|----------------------|---------|
| Quantum kernels | Feature space | Can be dequantized | Limited advantage |
| VQC | Parameterized circuits | Neural networks | No clear advantage yet |
| QNN | Hybrid | Deep learning | Research ongoing |

---

### 4. Integration Patterns

**Pattern 1: Variational Hybrid**
```
Classical optimizer ←→ Quantum circuit
      ↓
  Measurement → Cost function → Parameter update
```

**Pattern 2: Fault-Tolerant Pipeline**
```
State prep → QPE → Measurement → Classical post-processing
```

**Pattern 3: Error-Mitigated NISQ**
```
Noisy circuit → Error mitigation → Corrected expectation value
     ↓
Multiple noise levels for ZNE
```

**Pattern 4: Adaptive Algorithm**
```
while not converged:
    Evaluate gradients for operator pool
    Select best operator
    Optimize parameters
    Check convergence
```

---

### 5. Research Frontiers

**Open Problems in Quantum Algorithms:**

1. **Barren plateau solutions:** Beyond local cost functions
2. **Error mitigation limits:** How far can we push NISQ?
3. **Quantum advantage:** Clear, practical demonstrations
4. **Algorithm-hardware co-design:** Tailored solutions
5. **Classical simulability:** Where is the quantum boundary?

**Emerging Directions:**

- **Quantum error correction on NISQ:** Partial fault tolerance
- **Tensor network + quantum:** Hybrid classical-quantum
- **Problem-inspired ansatze:** Beyond generic HEA
- **Measurement-based VQE:** Alternative paradigms
- **Quantum-inspired classical:** Learning from quantum algorithms

---

### 6. Capstone Project Framework

**Year 2 Capstone Objective:**
Design, implement, and analyze a complete quantum algorithm for a research-relevant problem.

**Project Components:**

1. **Problem Selection**
   - Molecular simulation (chemistry)
   - Optimization (combinatorics)
   - Machine learning (classification/regression)
   - Custom research problem

2. **Algorithm Design**
   - Choose appropriate framework (VQE, QAOA, etc.)
   - Design problem-specific ansatz
   - Select optimization strategy
   - Plan error mitigation

3. **Implementation**
   - Code in Qiskit/PennyLane
   - Classical simulation verification
   - Hardware execution (if available)

4. **Analysis**
   - Accuracy benchmarks
   - Resource estimates
   - Scalability analysis
   - Comparison with classical methods

5. **Documentation**
   - Technical report
   - Code repository
   - Presentation

---

### 7. Capstone Project Ideas

**Project 1: Molecular Ground State of H2O**
- Full VQE with symmetry-adapted ansatz
- ADAPT-VQE comparison
- Error mitigation on real hardware
- Benchmark against classical CCSD(T)

**Project 2: MaxCut on Regular Graphs**
- QAOA implementation
- Ansatz optimization (depth, mixer)
- Scaling analysis to 20+ qubits
- Comparison with classical heuristics

**Project 3: Quantum Kernel Classification**
- Design custom feature map
- Implement quantum kernel SVM
- Benchmark on standard datasets
- Analyze expressibility

**Project 4: Variational Quantum Simulation**
- Time evolution of spin chain
- Trotterization vs variational
- Error accumulation analysis
- Phase transition detection

**Project 5: Error Mitigation Study**
- Implement ZNE, PEC, CDR
- Compare on same problem
- Hardware noise characterization
- Cost-benefit analysis

---

### 8. Year 2 Summary

**Semester 2A: Open Quantum Systems & Computation**
- Density matrices, master equations
- Quantum channels, Kraus operators
- Decoherence and error models
- Intro to quantum computing

**Semester 2B: Fault Tolerance & Hardware**
- Quantum error correction codes
- Fault-tolerant constructions
- Hardware platforms
- **Advanced algorithms (this month)**

**Key Achievements:**
- Research-level understanding of QEC
- Hardware-aware algorithm design
- Error mitigation proficiency
- Ready for original research

---

## Practical Applications

### Algorithm Selection Exercise

**Scenario:** You need to compute the ground state energy of a 20-qubit representation of a small molecule.

**Analysis:**

1. **Resources:** 20 qubits, ~200 two-qubit gates budget

2. **Options:**
   - UCCSD: Too deep (>1000 gates)
   - HEA: Risk of barren plateaus
   - ADAPT-VQE: Best balance

3. **Error Mitigation:** ZNE with 2-3 scale factors

4. **Decision:** ADAPT-VQE + linear ZNE + particle number verification

---

## Worked Examples

### Example 1: Algorithm Comparison for H4

**Problem:** Compare expected performance of VQE, ADAPT-VQE, and QPE for the H4 molecule.

**Solution:**

**System:** 4 hydrogen atoms, STO-3G basis, 8 spin-orbitals

**VQE with UCCSD:**
- Parameters: ~50-100
- Circuit depth: ~500 CNOTs
- Accuracy: Chemical accuracy possible
- NISQ feasibility: Marginal

**ADAPT-VQE:**
- Parameters: ~10-20 (adaptive)
- Circuit depth: ~50-100 CNOTs
- Accuracy: Chemical accuracy demonstrated
- NISQ feasibility: Good

**QPE:**
- Ancilla qubits: 10-20 for chemical accuracy
- Circuit depth: >10,000 gates
- Accuracy: Arbitrary precision
- NISQ feasibility: None

**Recommendation:** ADAPT-VQE for current hardware.

---

### Example 2: Capstone Scoping

**Problem:** Scope a capstone project on quantum optimization for portfolio optimization.

**Solution:**

**Week 1-2: Problem Formulation**
- Map portfolio optimization to QUBO
- Identify size constraints (assets = qubits)
- Define success metrics (Sharpe ratio, etc.)

**Week 3-4: Algorithm Implementation**
- QAOA implementation
- VQE variant for comparison
- Classical benchmark (simulated annealing)

**Week 5-6: Hardware Execution**
- Run on IBM Quantum or simulator
- Apply error mitigation
- Collect statistics

**Week 7-8: Analysis & Writing**
- Performance analysis
- Scaling discussion
- Documentation and presentation

**Deliverables:**
- Code repository
- 10-page technical report
- 15-minute presentation

---

### Example 3: Research Gap Identification

**Problem:** Identify a research gap in advanced variational methods.

**Solution:**

**Current State:**
- ADAPT-VQE works well for ground states
- Barren plateau mitigation is understood for shallow circuits
- Error mitigation adds overhead

**Gap:** Excited state ADAPT-VQE with noise

**Research Question:**
> How do barren plateaus and noise affect orthogonality-constrained VQE for excited states, and can adapted error mitigation strategies recover accuracy?

**Approach:**
1. Implement excited-state ADAPT-VQE
2. Analyze gradient variance for constrained optimization
3. Develop excited-state-specific mitigation
4. Benchmark on small molecules

---

## Practice Problems

### Level 1: Direct Application

1. For a 10-qubit VQE problem with 50 parameters, estimate the number of circuit evaluations for one gradient-based optimization step using parameter shift.

2. List three advantages of ADAPT-VQE over hardware-efficient ansatz.

3. What error mitigation method would you choose for a 20-gate circuit with known depolarizing noise?

### Level 2: Intermediate

4. Design an evaluation rubric for comparing QAOA and VQE on the same MaxCut instance.

5. Outline a plan to verify that your VQE implementation is correct before running on hardware.

6. For a molecular system with $D_2$ point group symmetry, describe how to reduce the ADAPT-VQE operator pool.

### Level 3: Challenging

7. Propose a novel ansatz design that combines symmetry preservation with hardware efficiency for a specific hardware topology.

8. Analyze the trade-off between shot count and ZNE scale factor choice for a fixed total measurement budget.

9. **Research:** Design a benchmark suite for evaluating quantum algorithms on NISQ devices that includes both accuracy and resource metrics.

---

## Computational Lab

### Objective
Integrate Month 35 techniques in a comprehensive VQE workflow.

```python
"""
Day 980 Computational Lab: Month 35 Synthesis
Advanced Variational Methods - Week 140
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# =============================================================================
# Part 1: Comprehensive VQE Pipeline
# =============================================================================

print("=" * 70)
print("Part 1: Comprehensive VQE Pipeline")
print("=" * 70)

n_qubits = 4

# Create a molecular-inspired Hamiltonian (simplified H2 in larger basis)
H = qml.Hamiltonian(
    [0.5, 0.5, 0.3, 0.3, -0.2, -0.2, 0.1, 0.1, 0.05, 0.05],
    [
        qml.PauliZ(0), qml.PauliZ(1),
        qml.PauliZ(2), qml.PauliZ(3),
        qml.PauliX(0) @ qml.PauliX(1),
        qml.PauliY(0) @ qml.PauliY(1),
        qml.PauliX(2) @ qml.PauliX(3),
        qml.PauliY(2) @ qml.PauliY(3),
        qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliZ(1) @ qml.PauliZ(3)
    ]
)

# Compute exact ground state
H_matrix = qml.matrix(H)
eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
E_exact = eigenvalues[0]
print(f"Exact ground state energy: {E_exact:.6f}")

# =============================================================================
# Part 2: Ansatz Comparison
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Comparing Ansatz Types")
print("=" * 70)

dev = qml.device('default.qubit', wires=n_qubits)

# Hardware-Efficient Ansatz
def hea_ansatz(params, n_layers):
    param_idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(params[param_idx], wires=i)
        param_idx += 1
    return param_idx

# Symmetry-Preserving Ansatz (particle number conserving)
def sym_ansatz(params, n_layers):
    param_idx = 0
    # Start in |0011⟩ state (N=2)
    qml.PauliX(2)
    qml.PauliX(3)

    for layer in range(n_layers):
        # Number-preserving XY gates
        for i in range(n_qubits - 1):
            qml.IsingXY(params[param_idx], wires=[i, i+1])
            param_idx += 1
    return param_idx

# Test both ansatze
results = {}

for ansatz_name, ansatz_func, get_n_params in [
    ('HEA-2L', lambda p: hea_ansatz(p, 2), lambda: 2 * n_qubits + n_qubits),
    ('HEA-3L', lambda p: hea_ansatz(p, 3), lambda: 3 * n_qubits + n_qubits),
    ('Sym-2L', lambda p: sym_ansatz(p, 2), lambda: 2 * (n_qubits - 1)),
    ('Sym-3L', lambda p: sym_ansatz(p, 3), lambda: 3 * (n_qubits - 1)),
]:
    n_params = get_n_params()

    @qml.qnode(dev)
    def circuit(params):
        if 'Sym' in ansatz_name:
            # Symmetry ansatz already has X gates
            pass
        ansatz_func(params)
        return qml.expval(H)

    def cost(params):
        return float(circuit(pnp.array(params)))

    # Optimize with multiple random starts
    best_energy = float('inf')
    start_time = time.time()
    for trial in range(5):
        x0 = np.random.uniform(-0.1, 0.1, n_params)
        result = minimize(cost, x0, method='COBYLA', options={'maxiter': 150})
        if result.fun < best_energy:
            best_energy = result.fun
    elapsed = time.time() - start_time

    error = (best_energy - E_exact) * 1000  # mHa
    results[ansatz_name] = {
        'energy': best_energy,
        'error': error,
        'params': n_params,
        'time': elapsed
    }
    print(f"{ansatz_name}: E = {best_energy:.6f}, error = {error:.2f} mHa, "
          f"params = {n_params}, time = {elapsed:.1f}s")

# =============================================================================
# Part 3: Error Mitigation Integration
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Error Mitigation")
print("=" * 70)

# Simulate noisy device
dev_noisy = qml.device('default.mixed', wires=n_qubits)
noise_rate = 0.01

def noisy_hea(params, noise_scale=1.0):
    """HEA with depolarizing noise."""
    param_idx = 0
    for layer in range(2):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            qml.DepolarizingChannel(noise_rate * noise_scale, wires=i)
            param_idx += 1
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
            qml.DepolarizingChannel(noise_rate * noise_scale, wires=i)
            qml.DepolarizingChannel(noise_rate * noise_scale, wires=i+1)
    for i in range(n_qubits):
        qml.RY(params[param_idx], wires=i)
        param_idx += 1

n_params_hea = 2 * n_qubits + n_qubits

@qml.qnode(dev_noisy)
def noisy_circuit(params, noise_scale=1.0):
    noisy_hea(params, noise_scale)
    return qml.expval(H)

# Find optimal parameters using noisy device
def noisy_cost(params):
    return float(noisy_circuit(pnp.array(params)))

result_noisy = minimize(noisy_cost, np.random.uniform(-0.1, 0.1, n_params_hea),
                       method='COBYLA', options={'maxiter': 100})
optimal_params = result_noisy.x

# Evaluate at different noise scales for ZNE
E_noisy_1 = float(noisy_circuit(optimal_params, noise_scale=1.0))
E_noisy_2 = float(noisy_circuit(optimal_params, noise_scale=2.0))
E_noisy_3 = float(noisy_circuit(optimal_params, noise_scale=3.0))

# Linear ZNE
E_zne_linear = (3 * E_noisy_1 - E_noisy_3) / 2

# Richardson ZNE
scales = np.array([1.0, 2.0, 3.0])
energies = np.array([E_noisy_1, E_noisy_2, E_noisy_3])
V = np.vstack([np.ones_like(scales), scales, scales**2]).T
coeffs = np.linalg.lstsq(V, energies, rcond=None)[0]
E_zne_richardson = coeffs[0]

# Ideal energy at these parameters
E_ideal = float(circuit(optimal_params))

print(f"\nError Mitigation Results:")
print(f"  Ideal (at noisy-optimal params): {E_ideal:.6f}")
print(f"  Noisy (scale=1):                 {E_noisy_1:.6f}, error = {abs(E_noisy_1-E_ideal)*1000:.2f} mHa")
print(f"  ZNE Linear:                      {E_zne_linear:.6f}, error = {abs(E_zne_linear-E_ideal)*1000:.2f} mHa")
print(f"  ZNE Richardson:                  {E_zne_richardson:.6f}, error = {abs(E_zne_richardson-E_ideal)*1000:.2f} mHa")

# =============================================================================
# Part 4: Full Pipeline Demonstration
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Complete VQE Pipeline")
print("=" * 70)

def full_vqe_pipeline(ansatz_type='hea', n_layers=2, use_mitigation=True):
    """Complete VQE pipeline with optional error mitigation."""

    if ansatz_type == 'hea':
        n_params = (n_layers + 1) * n_qubits
        def ansatz(params):
            return hea_ansatz(params, n_layers)
    else:
        n_params = n_layers * (n_qubits - 1)
        def ansatz(params):
            return sym_ansatz(params, n_layers)

    # Step 1: Optimize on noisy device
    print(f"  Step 1: Optimizing {ansatz_type} with {n_layers} layers...")

    @qml.qnode(dev_noisy)
    def cost_circuit(params):
        if ansatz_type == 'hea':
            noisy_hea(params, noise_scale=1.0)
        else:
            qml.PauliX(2)
            qml.PauliX(3)
            for layer in range(n_layers):
                for i in range(n_qubits - 1):
                    qml.IsingXY(params[layer*(n_qubits-1) + i], wires=[i, i+1])
                    qml.DepolarizingChannel(noise_rate, wires=i)
                    qml.DepolarizingChannel(noise_rate, wires=i+1)
        return qml.expval(H)

    def cost_func(params):
        return float(cost_circuit(pnp.array(params)))

    x0 = np.random.uniform(-0.1, 0.1, n_params)
    result = minimize(cost_func, x0, method='COBYLA', options={'maxiter': 100})
    opt_params = result.x

    # Step 2: Apply error mitigation
    E_raw = result.fun

    if use_mitigation:
        print("  Step 2: Applying ZNE error mitigation...")
        E1 = float(cost_circuit(pnp.array(opt_params)))

        # Modify circuit for higher noise
        @qml.qnode(dev_noisy)
        def cost_circuit_scaled(params, scale):
            if ansatz_type == 'hea':
                noisy_hea(params, noise_scale=scale)
            else:
                qml.PauliX(2)
                qml.PauliX(3)
                for layer in range(n_layers):
                    for i in range(n_qubits - 1):
                        qml.IsingXY(params[layer*(n_qubits-1) + i], wires=[i, i+1])
                        qml.DepolarizingChannel(noise_rate * scale, wires=i)
                        qml.DepolarizingChannel(noise_rate * scale, wires=i+1)
            return qml.expval(H)

        E3 = float(cost_circuit_scaled(pnp.array(opt_params), 3.0))
        E_final = (3 * E1 - E3) / 2
    else:
        E_final = E_raw

    return E_final, opt_params

# Run full pipeline
print("\nRunning complete VQE pipelines...")
E_hea_mit, _ = full_vqe_pipeline('hea', 2, use_mitigation=True)
E_sym_mit, _ = full_vqe_pipeline('sym', 2, use_mitigation=True)

print(f"\nFinal Pipeline Results:")
print(f"  Exact:                {E_exact:.6f}")
print(f"  HEA + ZNE:            {E_hea_mit:.6f}, error = {abs(E_hea_mit-E_exact)*1000:.2f} mHa")
print(f"  Symmetry + ZNE:       {E_sym_mit:.6f}, error = {abs(E_sym_mit-E_exact)*1000:.2f} mHa")

# =============================================================================
# Part 5: Month 35 Summary Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Summary Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Ansatz comparison
ax1 = axes[0, 0]
ansatze = list(results.keys())
errors = [results[a]['error'] for a in ansatze]
colors = ['steelblue' if 'HEA' in a else 'coral' for a in ansatze]
bars = ax1.bar(ansatze, errors, color=colors, edgecolor='black')
ax1.set_ylabel('Error (mHa)')
ax1.set_title('Ansatz Comparison')
ax1.axhline(y=1.6, color='red', linestyle='--', label='Chemical accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Parameter efficiency
ax2 = axes[0, 1]
params = [results[a]['params'] for a in ansatze]
ax2.scatter(params, errors, c=colors, s=200, edgecolors='black')
for i, a in enumerate(ansatze):
    ax2.annotate(a, (params[i], errors[i]), textcoords="offset points",
                xytext=(5, 5), fontsize=9)
ax2.set_xlabel('Number of Parameters')
ax2.set_ylabel('Error (mHa)')
ax2.set_title('Parameter Efficiency')
ax2.grid(True, alpha=0.3)

# Error mitigation effect
ax3 = axes[1, 0]
methods = ['Noisy', 'ZNE Linear', 'ZNE Richardson']
em_errors = [
    abs(E_noisy_1 - E_ideal) * 1000,
    abs(E_zne_linear - E_ideal) * 1000,
    abs(E_zne_richardson - E_ideal) * 1000
]
ax3.bar(methods, em_errors, color=['red', 'blue', 'green'], alpha=0.7, edgecolor='black')
ax3.set_ylabel('Error (mHa)')
ax3.set_title('Error Mitigation Effectiveness')
ax3.grid(True, alpha=0.3, axis='y')

# Algorithm selection flowchart (as text/boxes)
ax4 = axes[1, 1]
ax4.axis('off')
ax4.set_title('Month 35 Algorithm Summary', fontsize=14)

summary_text = """
Week 137: HHL Algorithm
  - Quantum linear systems
  - Exponential speedup (conditions apply)

Week 138: Quantum Simulation
  - Trotterization, QSP
  - Most promising NISQ application

Week 139: Quantum ML
  - Feature maps, kernels
  - Expressibility vs trainability

Week 140: Advanced Variational
  - ADAPT-VQE: compact ansatze
  - Symmetry preservation
  - Barren plateau mitigation
  - Error-mitigated VQE

Key Insight: Match algorithm to hardware!
"""

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('day_980_synthesis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_980_synthesis.png'")

# =============================================================================
# Part 6: Capstone Project Outline
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Capstone Project Outline")
print("=" * 70)

capstone_outline = """
Suggested Capstone: Error-Mitigated ADAPT-VQE for Small Molecules

Objectives:
1. Implement full ADAPT-VQE algorithm
2. Add symmetry constraints (N, Sz)
3. Integrate ZNE error mitigation
4. Benchmark on H2, LiH, BeH2 molecules
5. Compare with classical methods (FCI, CCSD)

Timeline:
- Week 1-2: ADAPT-VQE implementation and testing
- Week 3: Symmetry-adapted operator pools
- Week 4: Error mitigation integration
- Week 5-6: Hardware execution and data collection
- Week 7-8: Analysis and report writing

Deliverables:
- Code repository with documentation
- Technical report (15-20 pages)
- Presentation (20 minutes)
- Reproducibility package

Success Criteria:
- Chemical accuracy (< 1.6 mHa) for H2, LiH
- Working hardware demonstration
- Clear comparison with classical methods
"""

print(capstone_outline)

print("\n" + "=" * 70)
print("Month 35 Complete!")
print("Year 2 Nearly Complete!")
print("=" * 70)
```

---

## Summary

### Month 35 Key Achievements

| Week | Topic | Key Takeaway |
|------|-------|--------------|
| 137 | HHL & Linear Algebra | Exponential speedup with caveats |
| 138 | Quantum Simulation | Most natural quantum application |
| 139 | Quantum ML | Expressibility-trainability trade-off |
| 140 | Advanced Variational | ADAPT-VQE + error mitigation = NISQ ready |

### Algorithm Selection Guidelines

1. **Natural quantum problems** (simulation): Trotterization or variational
2. **Optimization**: QAOA for combinatorial, VQE for continuous
3. **Machine learning**: Quantum kernels for structured data
4. **Chemistry**: ADAPT-VQE with symmetry and mitigation

### Main Takeaways

1. **No universal algorithm:** Problem structure determines approach
2. **NISQ limitations:** Shallow circuits, error mitigation required
3. **Hybrid is practical:** Classical optimization + quantum circuits
4. **Research frontiers:** Barren plateaus, advantage proofs, error tolerance
5. **Capstone integration:** Apply all Year 2 knowledge to real problem

---

## Daily Checklist

- [ ] Review all four weeks of Month 35
- [ ] Complete algorithm selection exercise
- [ ] Work through capstone planning
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the comprehensive lab
- [ ] Draft capstone project proposal

---

## Preview: Month 36 (Year 2 Capstone)

Month 36 is the **Year 2 Capstone Project**. You will:
1. Select a research-relevant quantum computing problem
2. Design and implement a complete algorithmic solution
3. Execute on simulators and/or real hardware
4. Analyze results and compare with classical methods
5. Write a technical report suitable for publication

**This is your opportunity to synthesize all Year 2 learning into original work!**

---

*"The goal is not to build a quantum computer. The goal is to solve problems that matter."*
--- Applied quantum computing perspective

---

**Congratulations on completing Month 35: Advanced Quantum Algorithms!**

---

**Next:** Month 36 - Year 2 Capstone Project
