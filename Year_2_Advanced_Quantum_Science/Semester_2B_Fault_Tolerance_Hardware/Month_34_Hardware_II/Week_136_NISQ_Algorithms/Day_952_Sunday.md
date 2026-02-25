# Day 952: Month 34 Synthesis - NISQ Algorithm Design

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 2.5 hours | Comprehensive review and integration |
| Afternoon | 2.5 hours | Advanced problems and case studies |
| Evening | 2 hours | Capstone project: End-to-end NISQ implementation |

## Learning Objectives

By the end of today, you will be able to:

1. Synthesize all concepts from Week 136 into a coherent understanding of NISQ algorithms
2. Evaluate quantum advantage prospects for near-term applications
3. Design complete NISQ algorithm implementations considering all constraints
4. Identify the transition path from NISQ to fault-tolerant quantum computing
5. Apply best practices for variational algorithm development
6. Assess current state-of-the-art and future directions

## Core Content

### 1. Month 34 Conceptual Map

```
                    ┌─────────────────────────────────────┐
                    │       NISQ ALGORITHM DESIGN         │
                    │         (Month 34 / Week 136)       │
                    └─────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  HARDWARE       │        │  ALGORITHMS     │        │  OPTIMIZATION   │
│  CONSTRAINTS    │        │                 │        │  & EXECUTION    │
├─────────────────┤        ├─────────────────┤        ├─────────────────┤
│ • Qubit count   │        │ • VQE           │        │ • Classical     │
│ • Coherence T1  │        │ • QAOA          │        │   optimizers    │
│ • Gate fidelity │        │ • Ansatz design │        │ • Shot budgets  │
│ • Connectivity  │        │ • Gradients     │        │ • Error aware   │
│ • Quantum Vol.  │        │                 │        │   compilation   │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CHALLENGES                                    │
├─────────────────────────────────────────────────────────────────────┤
│  • Barren plateaus limit trainability                               │
│  • Noise corrupts computation exponentially with depth              │
│  • Limited connectivity requires SWAP overhead                       │
│  • Shot noise limits precision                                       │
│  • Classical simulation competes for small problems                  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SOLUTIONS                                     │
├─────────────────────────────────────────────────────────────────────┤
│  • Hardware-efficient ansatzes                                       │
│  • Noise-aware compilation and routing                               │
│  • Error mitigation (ZNE, PEC, readout correction)                  │
│  • Local cost functions                                              │
│  • Layerwise training, identity initialization                       │
│  • Hybrid classical-quantum workflows                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. NISQ Era Summary

#### 2.1 Hardware Landscape (2024-2026)

| Platform | Qubits | 2Q Fidelity | T2 | Connectivity | Status |
|----------|--------|-------------|-----|--------------|--------|
| IBM Superconducting | 100-1000+ | 99-99.5% | 100-300 μs | Heavy-hex | Production |
| Google Superconducting | 50-100 | 99.5%+ | 20-50 μs | Grid | Research |
| IonQ Trapped Ion | 20-35 | 99.5%+ | Seconds | All-to-all | Production |
| Quantinuum Trapped Ion | 20-56 | 99.8%+ | Seconds | All-to-all | Production |
| QuEra Neutral Atom | 256+ | 99%+ | ~1 s | Reconfigurable | Production |

#### 2.2 What NISQ Can (Potentially) Do

| Application | Problem Size | Classical Alternative | Advantage Prospect |
|-------------|--------------|----------------------|-------------------|
| Chemistry (ground state) | 10-50 qubits | CCSD(T), DMRG | Unclear |
| Optimization (QAOA) | 50-100 qubits | Classical heuristics | Unlikely near-term |
| Machine Learning | 10-50 qubits | Neural networks | Research phase |
| Quantum Simulation | 50-100 qubits | Tensor networks | Most promising |

#### 2.3 Current Limitations

**The fundamental challenge:**
$$\boxed{\text{Useful circuits} > \text{Noise-limited depth} \times \text{Available qubits}}$$

For most applications, the circuits needed exceed what NISQ devices can reliably execute.

### 3. Algorithm Integration Review

#### 3.1 VQE: Complete Picture

**When to use VQE:**
- Ground state energy estimation
- Small molecules (< 30 qubits effective)
- When approximate solutions are valuable

**VQE Design Checklist:**
1. [ ] Hamiltonian has reasonable term count
2. [ ] Ansatz depth within coherence limits
3. [ ] Local cost function to avoid barren plateaus
4. [ ] Noise-aware qubit mapping
5. [ ] Appropriate optimizer selected
6. [ ] Error mitigation strategy chosen
7. [ ] Shot budget calculated for target precision

**Key equation:**
$$E(\boldsymbol{\theta}) = \sum_i c_i \langle\psi(\boldsymbol{\theta})|\hat{P}_i|\psi(\boldsymbol{\theta})\rangle$$

#### 3.2 QAOA: Complete Picture

**When to use QAOA:**
- Combinatorial optimization problems
- Problems with natural graph structure
- When approximate solutions are acceptable

**QAOA Design Checklist:**
1. [ ] Problem encoded as QUBO/Ising
2. [ ] QAOA depth selected based on graph structure
3. [ ] Mixer appropriate for constraints
4. [ ] Warm-starting from classical solution
5. [ ] Circuit depth within hardware limits
6. [ ] Classical solver baseline established

**Key equation:**
$$|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = \prod_{l=1}^{p} e^{-i\beta_l \hat{B}} e^{-i\gamma_l \hat{C}} |+\rangle^{\otimes n}$$

#### 3.3 Barren Plateaus: Mitigation Summary

| Strategy | Mechanism | Cost |
|----------|-----------|------|
| Shallow circuits | Limit 2-design formation | Reduced expressibility |
| Local costs | Polynomial variance scaling | Problem reformulation |
| Identity init | Start near identity | May find local minima |
| Layerwise training | Gradual depth increase | Training time |
| Correlated parameters | Reduce effective dimension | Limited flexibility |

#### 3.4 Noise-Aware Compilation: Summary

| Technique | Benefit | Overhead |
|-----------|---------|----------|
| Error-weighted mapping | Better qubit selection | Compilation time |
| Noise-aware routing | Lower accumulated error | Path computation |
| Dynamical decoupling | Extended coherence | Pulse overhead |
| Pulse optimization | Higher gate fidelity | Calibration |

### 4. Transition to Fault Tolerance

#### 4.1 What Changes with QEC

| Aspect | NISQ | Fault-Tolerant |
|--------|------|----------------|
| Qubits | Physical | Logical (encoded) |
| Errors | Accumulate | Corrected |
| Depth | Limited | Unlimited (in principle) |
| Gates | Noisy native | Clean logical |
| Overhead | None | 1000-10000× physical qubits |

#### 4.2 The Transition Path

```
NISQ Era                    Early FT Era                    Scalable FT
(Now - 2030?)              (2030? - 2040?)                 (2040?+)
    │                            │                              │
    ▼                            ▼                              ▼
• Variational algorithms    • Small logical qubits          • Large-scale QEC
• Error mitigation         • Limited logical operations     • Full algorithms
• Hybrid classical-quantum • Error-mitigated + QEC         • Quantum advantage
• Hardware optimization    • Transition algorithms          • Practical applications
```

#### 4.3 Algorithms That Bridge NISQ to FT

**Quantum error mitigation + partial QEC:**
- Use limited QEC codes on critical subcircuits
- Apply mitigation to uncorrected parts

**Modular approaches:**
- Break problems into NISQ-sized pieces
- Use classical communication between modules

### 5. Quantum Advantage Assessment

#### 5.1 Requirements for Practical Advantage

$$\boxed{\text{Advantage} = \frac{\text{Classical Time} \times \text{Classical Cost}}{\text{Quantum Time} \times \text{Quantum Cost}} > 1}$$

**Factors:**
- Problem size scaling
- Error rates and mitigation overhead
- Classical algorithm improvements
- Hardware access costs

#### 5.2 Current State (2026)

| Domain | Advantage Claim | Status |
|--------|-----------------|--------|
| Random circuit sampling | Google 2019 | Disputed (classical improvements) |
| Boson sampling | Photonic systems | Limited practical value |
| Quantum simulation | Various | Most promising near-term |
| Optimization | QAOA | No clear advantage yet |
| Machine learning | QML | Research phase |

#### 5.3 Near-Term Opportunities

**Most promising areas:**
1. **Quantum simulation** of quantum systems (chemistry, materials)
2. **Verification and benchmarking** of quantum claims
3. **Hybrid algorithms** with quantum subroutines
4. **Quantum-inspired classical algorithms** development

### 6. Best Practices for NISQ Development

#### 6.1 Algorithm Design

1. **Start classical:** Understand the problem classically first
2. **Minimize circuit depth:** Every gate adds error
3. **Use hardware-native operations:** Avoid unnecessary compilation
4. **Consider noise from the start:** Design for noise, not against it
5. **Benchmark against classical:** Know your baseline

#### 6.2 Implementation

1. **Simulate first:** Debug on classical simulators
2. **Profile hardware:** Characterize device before running
3. **Use noise-aware compilation:** Don't waste good qubits
4. **Monitor convergence:** Use callbacks and logging
5. **Validate results:** Cross-check with available exact solutions

#### 6.3 Reporting

1. **State hardware used:** Specific device and calibration date
2. **Report error rates:** Relevant hardware metrics
3. **Include classical baseline:** What's the comparison?
4. **Provide reproducibility info:** Seeds, parameters, code
5. **Discuss limitations:** What didn't work?

## Worked Examples

### Example 1: Complete NISQ Algorithm Design

**Problem:** Design a VQE implementation for the LiH molecule on IBM hardware.

**Solution:**

Step 1: Problem analysis
- LiH in STO-3G basis: 12 spin-orbitals → 12 qubits (naively)
- With symmetry reduction: ~6-8 qubits feasible
- Hamiltonian: ~100 Pauli terms

Step 2: Hardware selection
- IBM Eagle (127 qubits): sufficient
- Typical 2Q fidelity: 99%
- T2 ≈ 200 μs, gate time ≈ 300 ns

Step 3: Circuit depth budget
$$d_{\max} = \frac{-\ln(0.5)}{\epsilon_{2Q}} = \frac{0.69}{0.01} \approx 70 \text{ two-qubit gates}$$

Step 4: Ansatz design
- UCCSD: too deep (~300 CNOTs)
- Hardware-efficient: 2 layers × 8 qubits × 2 = 32 CNOTs ✓
- Add one more layer: 48 CNOTs ✓

Step 5: Measurement grouping
- 100 terms → ~15 QWC groups
- Total measurements: 15 circuits

Step 6: Shot budget
- Target: 1 mHa precision
- Shots per group: ~10,000
- Total: 150,000 shots per energy evaluation

Step 7: Optimization
- Use SPSA (noise robust)
- ~100-200 iterations
- Total shots: ~30 million

Step 8: Error mitigation
- Readout error correction: essential
- ZNE: 3× circuit overhead, reduces bias

**Estimated total runtime:** 2-4 hours on IBM Quantum

### Example 2: QAOA vs Classical Comparison

**Problem:** For MaxCut on a 20-node random graph, compare QAOA at depth p=3 with classical Goemans-Williamson algorithm.

**Solution:**

Step 1: Classical baseline
- GW algorithm: 0.878 approximation ratio guaranteed
- Runtime: polynomial in graph size
- For 20 nodes: ~milliseconds

Step 2: QAOA resource analysis
- 20 qubits, ~40 edges (if density 0.2)
- Depth p=3: 3 × 40 = 120 ZZ gates
- With SWAP overhead: ~200-300 CNOTs

Step 3: Expected QAOA performance
- Theoretical for p=3 on random graphs: ~0.8-0.9 ratio
- With noise (1% error, 300 CNOTs): circuit fidelity ≈ 5%
- Effective ratio with noise: significantly degraded

Step 4: Conclusion
For this problem size, classical GW is:
- Faster (milliseconds vs hours)
- More accurate (0.878 guaranteed vs noisy ~0.5-0.7)
- Cheaper (laptop vs quantum cloud)

**Verdict:** No quantum advantage for this problem size.

### Example 3: Barren Plateau Diagnosis

**Problem:** A VQE optimization on 10 qubits with 4-layer HEA shows gradient magnitudes of ~10⁻⁵ at initialization. Diagnose and fix.

**Solution:**

Step 1: Barren plateau check
- 10 qubits, 4 layers → likely forming approximate 2-design
- Expected variance: O(1/2¹⁰) = O(10⁻³)
- Standard deviation: O(10⁻¹·⁵) ≈ 0.03
- Observed: 10⁻⁵ → significantly worse, possibly global cost issue

Step 2: Diagnosis
- Check cost function: is it global or local?
- If measuring $\langle Z_0 Z_1 ... Z_9 \rangle$: global, BP expected
- Circuit depth: 4 layers × 10 qubits = 40 CNOTs, borderline

Step 3: Fixes
Option A: Reduce to local cost
$$\mathcal{L}_{\text{local}} = \sum_i \langle Z_i\rangle^2 \text{ instead of } \langle Z_0...Z_9\rangle$$

Option B: Reduce circuit depth
- Try 2 layers instead of 4
- Check gradient variance increases

Option C: Identity initialization
- Set all parameters to 0 initially
- Gradients should be larger near identity

Step 4: Verification
After implementing Option A:
- Expected variance: O(1/poly(n)) ≈ O(1/100)
- Expected gradient: O(0.1)
- Much more trainable!

## Practice Problems

### Level 1: Comprehensive Review

1. **Hardware metrics:** A device has T1 = 150 μs, T2 = 100 μs, 1Q fidelity 99.95%, 2Q fidelity 99.2%, readout fidelity 98%. Calculate the maximum useful circuit depth for 90% overall fidelity.

2. **VQE design:** For a 6-qubit VQE with 20 Hamiltonian terms, design a shot allocation strategy achieving 0.01 Ha precision.

3. **QAOA encoding:** Encode the vertex cover problem on a triangle graph as a QAOA cost Hamiltonian.

### Level 2: Integration Problems

4. **Complete workflow:** Design a complete QAOA workflow for MaxCut on a 4×4 grid graph, including:
   - Hardware requirements
   - Ansatz depth selection
   - Optimizer choice
   - Expected performance

5. **Barren plateau analysis:** For a VQE with hardware-efficient ansatz, derive the relationship between gradient variance and:
   - Number of qubits n
   - Circuit depth L
   - Cost function locality k

6. **Compilation optimization:** Given a VQE circuit requiring CNOTs (0,3), (1,4), (2,5) on a 6-qubit linear chain, find the optimal initial mapping and SWAP schedule.

### Level 3: Research-Level Problems

7. **Quantum advantage threshold:** For a specific chemistry problem (H₂O in cc-pVDZ basis), estimate the hardware requirements (qubits, fidelity, coherence) needed to outperform classical CCSD(T) calculations.

8. **Hybrid algorithm design:** Propose a hybrid algorithm that uses NISQ circuits for a subroutine within a larger classical optimization, such that:
   - The quantum part is feasible on current hardware
   - The hybrid provides value over pure classical approaches
   - Error mitigation is integrated

9. **Transition planning:** Design a "bridge" algorithm that:
   - Works on current NISQ devices with error mitigation
   - Can smoothly transition to using partial error correction
   - Scales to full fault tolerance as hardware improves

## Computational Lab: Capstone Project

### Complete NISQ Implementation

```python
"""
Day 952 Lab: Capstone Project
End-to-end NISQ algorithm implementation
Complete VQE for H2 with all best practices
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ============================================================
# Part 1: Configuration and Setup
# ============================================================

@dataclass
class NISQConfig:
    """Complete NISQ algorithm configuration."""
    # Problem
    molecule: str = "H2"
    bond_length: float = 0.735

    # Hardware
    n_qubits: int = 4
    t1_us: float = 200.0
    t2_us: float = 150.0
    gate_error_1q: float = 0.0003
    gate_error_2q: float = 0.01
    readout_error: float = 0.02

    # Ansatz
    n_layers: int = 2
    ansatz_type: str = "hardware_efficient"

    # Optimization
    optimizer: str = "SPSA"
    max_iterations: int = 100
    shots_per_term: int = 1000
    convergence_tol: float = 1e-4

    # Error mitigation
    use_zne: bool = False
    use_readout_mitigation: bool = True

    # Monitoring
    verbose: bool = True
    save_history: bool = True

# ============================================================
# Part 2: Noise Model
# ============================================================

def create_realistic_noise_model(config: NISQConfig) -> NoiseModel:
    """Create noise model based on configuration."""
    noise_model = NoiseModel()

    # Single-qubit errors
    error_1q = depolarizing_error(config.gate_error_1q, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'u1', 'u2', 'u3'])

    # Two-qubit errors
    error_2q = depolarizing_error(config.gate_error_2q, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    return noise_model

# ============================================================
# Part 3: Hamiltonian and Ansatz
# ============================================================

def get_h2_hamiltonian(bond_length: float = 0.735) -> Tuple[SparsePauliOp, float]:
    """Get H2 Hamiltonian and exact ground state energy."""
    # Coefficients for H2 at various bond lengths
    coeffs = {
        'IIII': -0.8105,
        'IIIZ': 0.1721,
        'IIZI': -0.2257,
        'IZII': 0.1721,
        'ZIII': -0.2257,
        'IIZZ': 0.1209,
        'IZIZ': 0.1689,
        'IZZI': 0.0454,
        'ZIIZ': 0.0454,
        'ZIZI': 0.1689,
        'ZZII': 0.1209,
        'XXXX': 0.0454,
        'XXYY': 0.0454,
        'YYXX': 0.0454,
        'YYYY': 0.0454
    }

    pauli_list = [(p, c) for p, c in coeffs.items()]
    H = SparsePauliOp.from_list(pauli_list)

    # Exact ground state energy (FCI)
    exact_energy = -1.1373  # Hartree at equilibrium

    return H, exact_energy

def create_ansatz(config: NISQConfig, params: np.ndarray) -> QuantumCircuit:
    """Create parameterized ansatz circuit."""
    qc = QuantumCircuit(config.n_qubits)
    param_idx = 0

    for layer in range(config.n_layers):
        # Rotation layer
        for q in range(config.n_qubits):
            qc.ry(params[param_idx], q)
            param_idx += 1
            qc.rz(params[param_idx], q)
            param_idx += 1

        # Entangling layer
        for q in range(config.n_qubits - 1):
            qc.cx(q, q + 1)

    # Final rotation
    for q in range(config.n_qubits):
        qc.ry(params[param_idx], q)
        param_idx += 1

    return qc

def count_params(config: NISQConfig) -> int:
    """Count parameters in ansatz."""
    return 2 * config.n_qubits * config.n_layers + config.n_qubits

# ============================================================
# Part 4: VQE Engine with Best Practices
# ============================================================

class NISQVQEEngine:
    """Production VQE engine with all optimizations."""

    def __init__(self, config: NISQConfig):
        self.config = config
        self.estimator = Estimator()
        self.noise_model = create_realistic_noise_model(config)
        self.noisy_backend = AerSimulator(noise_model=self.noise_model)

        # Tracking
        self.energy_history = []
        self.param_history = []
        self.gradient_history = []
        self.n_evals = 0
        self.start_time = None

    def compute_energy(self, params: np.ndarray, hamiltonian: SparsePauliOp,
                       use_noise: bool = True) -> float:
        """Compute energy with optional noise simulation."""
        circuit = create_ansatz(self.config, params)

        if use_noise:
            # Transpile for noise model
            circuit = transpile(circuit, self.noisy_backend)

        job = self.estimator.run([(circuit, hamiltonian)])
        result = job.result()
        self.n_evals += 1

        return float(result[0].data.evs)

    def spsa_optimize(self, hamiltonian: SparsePauliOp,
                      initial_params: np.ndarray) -> np.ndarray:
        """SPSA optimization with monitoring."""
        params = initial_params.copy()
        n_params = len(params)

        # SPSA hyperparameters
        a = 0.05
        c = 0.1
        A = 10
        alpha = 0.602
        gamma = 0.101

        best_energy = float('inf')
        best_params = params.copy()
        patience_counter = 0

        for k in range(self.config.max_iterations):
            # Compute coefficients
            a_k = a / (k + 1 + A)**alpha
            c_k = c / (k + 1)**gamma

            # Random perturbation
            delta = 2 * np.random.randint(0, 2, n_params) - 1

            # Evaluate at perturbed points
            e_plus = self.compute_energy(params + c_k * delta, hamiltonian)
            e_minus = self.compute_energy(params - c_k * delta, hamiltonian)

            # Gradient estimate
            g_k = (e_plus - e_minus) / (2 * c_k) * delta

            # Update parameters
            params = params - a_k * g_k

            # Current energy
            energy = self.compute_energy(params, hamiltonian)

            # Track history
            self.energy_history.append(energy)
            self.param_history.append(params.copy())
            self.gradient_history.append(np.linalg.norm(g_k))

            # Best tracking
            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Convergence check
            if len(self.energy_history) >= 2:
                delta_e = abs(self.energy_history[-1] - self.energy_history[-2])
                if delta_e < self.config.convergence_tol:
                    if self.config.verbose:
                        print(f"Converged at iteration {k+1}")
                    break

            # Verbose output
            if self.config.verbose and (k + 1) % 10 == 0:
                print(f"Iter {k+1}: E = {energy:.6f} Ha, |∇| = {np.linalg.norm(g_k):.4f}")

        return best_params

    def run(self, hamiltonian: SparsePauliOp,
            exact_energy: float = None) -> Dict:
        """Run complete VQE optimization."""
        self.start_time = time.time()

        if self.config.verbose:
            print("="*60)
            print("NISQ VQE Engine")
            print("="*60)
            print(f"Molecule: {self.config.molecule}")
            print(f"Qubits: {self.config.n_qubits}")
            print(f"Ansatz layers: {self.config.n_layers}")
            print(f"Parameters: {count_params(self.config)}")
            print(f"Optimizer: {self.config.optimizer}")
            print("="*60 + "\n")

        # Initialize parameters (identity-like)
        n_params = count_params(self.config)
        initial_params = np.random.uniform(-0.1, 0.1, n_params)

        # Initial energy
        initial_energy = self.compute_energy(initial_params, hamiltonian)
        if self.config.verbose:
            print(f"Initial energy: {initial_energy:.6f} Ha\n")

        # Optimize
        optimal_params = self.spsa_optimize(hamiltonian, initial_params)

        # Final energy
        final_energy = self.energy_history[-1]
        total_time = time.time() - self.start_time

        # Results
        results = {
            'optimal_energy': final_energy,
            'optimal_params': optimal_params,
            'energy_history': self.energy_history,
            'gradient_history': self.gradient_history,
            'n_iterations': len(self.energy_history),
            'n_function_evals': self.n_evals,
            'total_time': total_time
        }

        if exact_energy is not None:
            error_mha = abs(final_energy - exact_energy) * 1000
            results['error_mha'] = error_mha
            results['exact_energy'] = exact_energy

        # Final report
        if self.config.verbose:
            print("\n" + "="*60)
            print("Results")
            print("="*60)
            print(f"Final energy: {final_energy:.6f} Ha")
            if exact_energy is not None:
                print(f"Exact energy: {exact_energy:.6f} Ha")
                print(f"Error: {error_mha:.2f} mHa")
                print(f"Chemical accuracy: {'YES' if error_mha < 1.6 else 'NO'}")
            print(f"Iterations: {len(self.energy_history)}")
            print(f"Function evaluations: {self.n_evals}")
            print(f"Total time: {total_time:.2f} s")
            print("="*60)

        return results

# ============================================================
# Part 5: Run Capstone VQE
# ============================================================

print("CAPSTONE PROJECT: Complete NISQ VQE Implementation\n")

# Configuration
config = NISQConfig(
    molecule="H2",
    bond_length=0.735,
    n_qubits=4,
    n_layers=2,
    optimizer="SPSA",
    max_iterations=80,
    gate_error_2q=0.01,
    verbose=True
)

# Get Hamiltonian
H, exact_energy = get_h2_hamiltonian(config.bond_length)
print(f"Hamiltonian: {len(H)} terms")
print(f"Exact ground state: {exact_energy:.6f} Ha\n")

# Run VQE
engine = NISQVQEEngine(config)
results = engine.run(H, exact_energy)

# ============================================================
# Part 6: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Energy convergence
ax = axes[0, 0]
ax.plot(results['energy_history'], 'b-', linewidth=2)
ax.axhline(y=exact_energy, color='r', linestyle='--', label='Exact')
ax.axhline(y=exact_energy + 0.0016, color='g', linestyle=':', label='Chemical accuracy')
ax.axhline(y=exact_energy - 0.0016, color='g', linestyle=':')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Energy (Ha)', fontsize=12)
ax.set_title('VQE Convergence', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Error convergence
ax = axes[0, 1]
errors = np.abs(np.array(results['energy_history']) - exact_energy) * 1000
ax.semilogy(errors, 'b-', linewidth=2)
ax.axhline(y=1.6, color='g', linestyle='--', label='Chemical accuracy (1.6 mHa)')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Error (mHa)', fontsize=12)
ax.set_title('Error Convergence', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Gradient norm
ax = axes[1, 0]
ax.semilogy(results['gradient_history'], 'b-', linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Gradient Norm', fontsize=12)
ax.set_title('Gradient Magnitude', fontsize=14)
ax.grid(True, alpha=0.3)

# Summary statistics
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
CAPSTONE VQE RESULTS
{'='*40}

Problem: {config.molecule} molecule
Bond length: {config.bond_length} Å
Qubits: {config.n_qubits}
Ansatz: {config.n_layers}-layer hardware-efficient

Hamiltonian terms: {len(H)}
Parameters: {count_params(config)}

Optimizer: {config.optimizer}
Iterations: {results['n_iterations']}
Function evaluations: {results['n_function_evals']}

Final energy: {results['optimal_energy']:.6f} Ha
Exact energy: {exact_energy:.6f} Ha
Error: {results['error_mha']:.2f} mHa

Chemical accuracy achieved: {'YES' if results['error_mha'] < 1.6 else 'NO'}

Total time: {results['total_time']:.2f} seconds
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('capstone_vqe_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 7: Hardware Comparison
# ============================================================

print("\n" + "="*60)
print("Hardware Comparison Study")
print("="*60)

error_rates = [0.005, 0.01, 0.02, 0.03]
final_errors = []

for err in error_rates:
    config.gate_error_2q = err
    config.verbose = False

    engine = NISQVQEEngine(config)
    np.random.seed(42)  # Reproducibility
    results = engine.run(H, exact_energy)

    final_errors.append(results['error_mha'])
    print(f"2Q error rate {err*100:.1f}%: Final error = {results['error_mha']:.2f} mHa")

plt.figure(figsize=(10, 6))
plt.plot([e*100 for e in error_rates], final_errors, 'bo-', markersize=10, linewidth=2)
plt.axhline(y=1.6, color='g', linestyle='--', label='Chemical accuracy')
plt.xlabel('Two-Qubit Gate Error Rate (%)', fontsize=12)
plt.ylabel('VQE Energy Error (mHa)', fontsize=12)
plt.title('VQE Performance vs Hardware Error Rate', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hardware_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCapstone project complete!")
print("All files saved.")
```

## Summary

### Week 136 Key Concepts

| Day | Topic | Key Equation |
|-----|-------|--------------|
| 946 | NISQ Characteristics | $d_{\max} \approx T_2/t_{\text{gate}}$ |
| 947 | VQE | $E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$ |
| 948 | QAOA | $|\gamma,\beta\rangle = \prod_p U_B U_C|+\rangle^n$ |
| 949 | Barren Plateaus | $\text{Var}[\partial_\theta L] \leq e^{-cn}$ |
| 950 | Noise-Aware Compilation | Error-weighted mapping & routing |
| 951 | Hybrid Workflows | Shot budget, optimizer selection |
| 952 | Synthesis | Integration of all concepts |

### Month 34 Takeaways

1. **NISQ is a specific regime** with unique constraints and opportunities.

2. **Variational algorithms** (VQE, QAOA) are the primary NISQ paradigm.

3. **Barren plateaus** fundamentally limit expressibility of trainable circuits.

4. **Noise-aware compilation** is essential for extracting maximum performance.

5. **Hybrid workflows** integrate classical and quantum resources effectively.

6. **Quantum advantage** remains elusive but quantum simulation shows promise.

7. **The transition to fault tolerance** will be gradual and hybrid approaches will bridge the gap.

## Daily Checklist

- [ ] I can synthesize all Week 136 concepts into a coherent understanding
- [ ] I can design complete NISQ algorithm implementations
- [ ] I understand the quantum advantage landscape and prospects
- [ ] I can evaluate the transition path to fault-tolerant computing
- [ ] I completed the capstone project with all best practices
- [ ] I can apply NISQ algorithm design principles to new problems

## Preview of Week 137

Next week we explore **Error Mitigation Techniques** - methods to improve NISQ results without full error correction:
- Zero-noise extrapolation (ZNE)
- Probabilistic error cancellation (PEC)
- Clifford data regression
- Readout error mitigation
- Symmetry verification
- Virtual distillation

Error mitigation bridges the gap between noisy NISQ execution and the ideal results we seek.

---

## End of Month 34: Hardware II

**Congratulations!** You have completed Month 34 of the Quantum Engineering curriculum, covering:

- Week 135: Advanced quantum hardware characterization
- Week 136: NISQ algorithm design (VQE, QAOA, barren plateaus, compilation, workflows)

**Next Month (35):** Fault-Tolerant Quantum Computing I - Quantum Error Correction codes and syndrome measurement, beginning the transition from NISQ to scalable quantum computation.
