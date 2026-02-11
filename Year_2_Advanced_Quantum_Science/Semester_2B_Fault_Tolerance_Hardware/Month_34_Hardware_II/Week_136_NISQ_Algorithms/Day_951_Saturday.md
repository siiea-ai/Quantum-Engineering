# Day 951: Hybrid Classical-Quantum Workflows

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 2.5 hours | Classical optimization theory for quantum circuits |
| Afternoon | 2.5 hours | Problem solving and workflow design |
| Evening | 2 hours | Computational lab: Complete VQE workflow |

## Learning Objectives

By the end of today, you will be able to:

1. Select appropriate classical optimizers for variational quantum algorithms
2. Design shot budget allocation strategies for efficient expectation estimation
3. Implement error-aware optimization that accounts for hardware noise
4. Construct complete hybrid workflows with proper callback and termination criteria
5. Interface with cloud quantum services for remote execution
6. Debug and monitor variational algorithm performance

## Core Content

### 1. The Hybrid Computing Paradigm

Variational quantum algorithms require tight integration between classical and quantum resources:

$$\boxed{\text{Hybrid Loop: } \theta_{t+1} = \theta_t - \eta \cdot \text{ClassicalUpdate}(\{E_i(\theta_t)\}_{\text{quantum}})}$$

**Key components:**
1. **Quantum execution:** State preparation, measurement
2. **Classical processing:** Gradient computation, parameter updates
3. **Communication:** Parameter transfer, result aggregation
4. **Control logic:** Convergence checking, resource management

### 2. Classical Optimizer Selection

#### 2.1 Gradient-Free Methods

**COBYLA (Constrained Optimization BY Linear Approximation):**
- Builds linear approximation to cost function
- No gradient computation required
- Works well with noisy evaluations

$$\theta_{k+1} = \arg\min_{\theta \in \text{simplex}} f_{\text{linear}}(\theta)$$

**Nelder-Mead (Simplex Method):**
- Maintains simplex of $n+1$ points
- Reflects, expands, contracts based on function values
- Robust to noise but slow convergence

**SPSA (Simultaneous Perturbation Stochastic Approximation):**
$$\boxed{\theta_{k+1} = \theta_k - a_k \frac{f(\theta_k + c_k\Delta) - f(\theta_k - c_k\Delta)}{2c_k} \Delta^{-1}}$$

where $\Delta$ is a random perturbation vector (often Bernoulli $\pm 1$).

**Advantages of SPSA:**
- Only 2 function evaluations per iteration (vs. $2p$ for finite difference)
- Naturally handles noise
- Good for high-dimensional problems

#### 2.2 Gradient-Based Methods

**Gradient Descent:**
$$\theta_{k+1} = \theta_k - \eta \nabla_\theta \mathcal{L}(\theta_k)$$

**Adam (Adaptive Moment Estimation):**
$$m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k$$
$$v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2$$
$$\theta_{k+1} = \theta_k - \eta \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon}$$

**Natural Gradient:**
$$\boxed{\theta_{k+1} = \theta_k - \eta F^{-1}(\theta_k) \nabla_\theta \mathcal{L}(\theta_k)}$$

where $F(\theta)$ is the quantum Fisher information matrix:

$$F_{ij} = \text{Re}\left[\langle\partial_i\psi|\partial_j\psi\rangle - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\right]$$

**Natural gradient advantages:**
- Reparametrization invariant
- Often faster convergence
- But expensive to compute ($O(p^2)$ measurements)

#### 2.3 Optimizer Comparison

| Optimizer | Gradient | Noise Tolerance | Convergence | Cost/Iteration |
|-----------|----------|-----------------|-------------|----------------|
| COBYLA | No | Medium | Slow | $O(p)$ |
| Nelder-Mead | No | High | Slow | $O(1)$ |
| SPSA | Approx | High | Medium | $O(1)$ |
| L-BFGS-B | Yes | Low | Fast | $O(p)$ |
| Adam | Yes | Medium | Fast | $O(p)$ |
| Natural GD | Yes | Medium | Fast | $O(p^2)$ |

### 3. Shot Budget Optimization

#### 3.1 Statistical Error in Expectation Values

For a Hamiltonian $H = \sum_i c_i P_i$:

$$\sigma_E^2 = \sum_i c_i^2 \frac{1 - \langle P_i\rangle^2}{N_i}$$

where $N_i$ is shots allocated to term $i$.

**Optimal allocation (minimize variance for fixed total shots):**

$$\boxed{N_i^* \propto |c_i| \sqrt{1 - \langle P_i\rangle^2}}$$

#### 3.2 Adaptive Shot Allocation

**Rosalin strategy:** Increase shots as optimization converges.

$$N_k = N_0 \cdot \left(\frac{k}{k_{\max}}\right)^\alpha + N_{\min}$$

**Intuition:** Early iterations need rough estimates; final iterations need precision.

#### 3.3 Grouping Commuting Observables

Pauli operators that commute can be measured simultaneously:

$$[P_i, P_j] = 0 \Rightarrow \text{measure together}$$

**Qubit-wise commuting (QWC) grouping:**
Operators are QWC if corresponding Paulis commute on each qubit.

Example: $Z_0Z_1$ and $Z_0I_1$ are QWC (share $Z$ on qubit 0).

### 4. Error-Aware Optimization

#### 4.1 Noise in Gradient Estimation

True gradient: $g = \nabla_\theta \mathcal{L}(\theta)$

Measured gradient: $\tilde{g} = g + \epsilon_{\text{shot}} + \epsilon_{\text{device}}$

**Shot noise:** $\text{Var}[\epsilon_{\text{shot}}] \propto 1/N$

**Device noise:** Bias from decoherence, gate errors

#### 4.2 Regularization and Smoothing

**Gradient averaging:**
$$\bar{g}_k = \frac{1}{M}\sum_{m=1}^{M} \tilde{g}_k^{(m)}$$

**Momentum:**
$$v_k = \beta v_{k-1} + (1-\beta)\tilde{g}_k$$

**Noise-aware learning rate:**
$$\eta_k = \eta_0 \cdot \min\left(1, \frac{\sigma_{\text{target}}}{\sigma_{\text{measured}}}\right)$$

#### 4.3 Error Mitigation in Optimization

Apply error mitigation to each energy evaluation:
- Zero-noise extrapolation (ZNE)
- Probabilistic error cancellation (PEC)
- Readout error mitigation

**Trade-off:** Mitigation reduces bias but increases variance.

### 5. Workflow Architecture

#### 5.1 Complete VQE Workflow

```
┌────────────────────────────────────────────────────────────────┐
│                    CLASSICAL CONTROLLER                         │
├────────────────────────────────────────────────────────────────┤
│  1. Initialize θ₀                                               │
│  2. While not converged:                                        │
│     a. Build circuits C(θₖ)                                     │
│     b. Submit to quantum backend                                │
│     c. Collect measurement results                              │
│     d. Compute E(θₖ), ∇E(θₖ)                                   │
│     e. Apply error mitigation                                   │
│     f. Update θₖ₊₁ = Optimizer.step(θₖ, ∇E)                    │
│     g. Check convergence criteria                               │
│  3. Return optimal θ*, E*                                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    QUANTUM BACKEND                              │
├────────────────────────────────────────────────────────────────┤
│  • Circuit compilation                                          │
│  • Queue management                                             │
│  • Execution on QPU                                             │
│  • Measurement and readout                                      │
└────────────────────────────────────────────────────────────────┘
```

#### 5.2 Callback Functions

Monitor optimization progress:

```python
def callback(iteration, params, energy, gradient_norm):
    print(f"Iter {iteration}: E = {energy:.6f}, |∇| = {gradient_norm:.4f}")

    # Early stopping
    if gradient_norm < 1e-5:
        return True  # Signal convergence

    # Logging
    log_to_file(iteration, params, energy)

    return False  # Continue
```

#### 5.3 Convergence Criteria

**Energy-based:**
$$|E_{k} - E_{k-1}| < \epsilon_E$$

**Gradient-based:**
$$\|\nabla E(\theta_k)\| < \epsilon_g$$

**Parameter-based:**
$$\|\theta_k - \theta_{k-1}\| < \epsilon_\theta$$

**Combined with patience:**
```
if criterion_met for patience consecutive iterations:
    converged = True
```

### 6. Cloud Quantum Integration

#### 6.1 IBM Quantum Runtime

```python
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session

service = QiskitRuntimeService()
backend = service.backend("ibm_sherbrooke")

with Session(service=service, backend=backend) as session:
    estimator = Estimator(session=session)
    # Run VQE within session for reduced latency
```

#### 6.2 Batch Execution

Group circuit evaluations to minimize queue overhead:

```python
# Instead of submitting one circuit at a time:
for theta in thetas:
    result = estimator.run(circuit(theta))  # Slow!

# Batch submission:
circuits = [circuit(theta) for theta in thetas]
results = estimator.run(circuits)  # Single queue entry
```

#### 6.3 Asynchronous Execution

```python
import asyncio

async def async_vqe_step(estimator, circuit, hamiltonian):
    job = estimator.run(circuit, hamiltonian)
    while not job.done():
        await asyncio.sleep(1)
    return job.result()
```

## Quantum Computing Applications

### Application: Production VQE

For chemistry applications requiring high precision:
1. Use natural gradient for faster convergence
2. Apply grouping to reduce measurement overhead
3. Implement ZNE for error mitigation
4. Monitor chemical accuracy (1.6 mHa)

### Application: QAOA Portfolio Optimization

For financial applications:
1. Encode portfolio as QUBO
2. Use SPSA for noise robustness
3. Warm-start from classical solution
4. Validate with classical solver

## Worked Examples

### Example 1: Shot Budget Calculation

**Problem:** A VQE Hamiltonian has 3 Pauli terms with coefficients $c_1 = 0.5$, $c_2 = 0.3$, $c_3 = 0.2$ (in Hartree). Target energy precision is 1 mHa. How many total shots are needed?

**Solution:**

Step 1: Variance formula
$$\sigma_E^2 = \sum_i \frac{c_i^2}{N_i}$$

Assuming $\langle P_i\rangle^2 \approx 0$ (worst case) and equal shots:
$$\sigma_E^2 = \frac{c_1^2 + c_2^2 + c_3^2}{N} = \frac{0.25 + 0.09 + 0.04}{N} = \frac{0.38}{N}$$

Step 2: Target precision
$$\sigma_E < 0.001 \text{ Ha}$$
$$\sigma_E^2 < 10^{-6}$$

Step 3: Required shots
$$N > \frac{0.38}{10^{-6}} = 380,000$$

Step 4: Optimal allocation
$$N_1 : N_2 : N_3 = |c_1| : |c_2| : |c_3| = 5 : 3 : 2$$

With 380,000 total: $N_1 = 190,000$, $N_2 = 114,000$, $N_3 = 76,000$

**Answer:** ~380,000 shots total with 5:3:2 allocation.

### Example 2: SPSA Parameter Selection

**Problem:** Design SPSA parameters for a 20-parameter VQE with expected final energy -1.5 Ha.

**Solution:**

Standard SPSA schedules:
$$a_k = \frac{a}{(k + A)^\alpha}, \quad c_k = \frac{c}{k^\gamma}$$

Step 1: Choose stability constant $A$
Rule of thumb: $A \approx 0.1 \times \text{max iterations}$
For 200 iterations: $A = 20$

Step 2: Choose $\alpha$ and $\gamma$
Standard: $\alpha = 0.602$, $\gamma = 0.101$

Step 3: Choose $a$ (learning rate)
Start with small steps: $a \approx 0.1$

Step 4: Choose $c$ (perturbation size)
Should be larger than noise level: $c \approx 0.1$

**SPSA parameters:**
```python
spsa_params = {
    'maxiter': 200,
    'a': 0.1,
    'c': 0.1,
    'A': 20,
    'alpha': 0.602,
    'gamma': 0.101
}
```

### Example 3: Convergence Analysis

**Problem:** Given VQE energy history [−1.0, −1.1, −1.12, −1.125, −1.127, −1.127, −1.128], determine if converged with tolerance $10^{-3}$ Ha and patience 3.

**Solution:**

Step 1: Compute energy differences
$$\Delta E_1 = |-1.1 - (-1.0)| = 0.1$$
$$\Delta E_2 = |-1.12 - (-1.1)| = 0.02$$
$$\Delta E_3 = |-1.125 - (-1.12)| = 0.005$$
$$\Delta E_4 = |-1.127 - (-1.125)| = 0.002$$
$$\Delta E_5 = |-1.127 - (-1.127)| = 0.000$$
$$\Delta E_6 = |-1.128 - (-1.127)| = 0.001$$

Step 2: Check against tolerance
$\Delta E_5 = 0.000 < 0.001$ ✓
$\Delta E_6 = 0.001 \leq 0.001$ ✓

Step 3: Check patience
Need 3 consecutive iterations below tolerance.
Only have 2 (iterations 5-6).

**Answer:** Not yet converged. Need one more iteration meeting criterion.

## Practice Problems

### Level 1: Direct Application

1. **Shot calculation:** For a 5-term Hamiltonian with all coefficients equal to 0.2, how many shots per term for 0.01 Ha precision?

2. **Optimizer selection:** Which optimizer would you choose for a 100-parameter VQE on noisy hardware with limited shot budget?

3. **Convergence check:** Energy values are [−2.0, −2.1, −2.11, −2.111]. Is this converged with tolerance 0.01 and patience 2?

### Level 2: Intermediate Analysis

4. **SPSA analysis:** Derive the variance of the SPSA gradient estimator as a function of perturbation size $c$ and function noise level $\sigma$.

5. **Grouping efficiency:** A 10-qubit Hamiltonian has 50 Pauli terms. If QWC grouping reduces this to 8 groups, what is the measurement speedup?

6. **Error mitigation trade-off:** ZNE with 3 noise levels increases circuit count by 3x. If this reduces bias by 50%, when is it beneficial?

### Level 3: Challenging Problems

7. **Natural gradient implementation:** Derive an efficient method to compute the quantum Fisher information matrix using $O(p)$ circuits instead of $O(p^2)$.

8. **Adaptive workflow:** Design an adaptive VQE workflow that automatically adjusts shot budget, learning rate, and error mitigation level based on observed convergence behavior.

9. **Multi-objective optimization:** Formulate VQE as a multi-objective problem minimizing both energy and circuit depth. How would you implement Pareto-optimal solutions?

## Computational Lab: Complete VQE Workflow

### Lab 1: Production-Ready VQE

```python
"""
Day 951 Lab: Complete Hybrid VQE Workflow
Production-ready implementation with all components
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict
from scipy.optimize import minimize
import time

from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

# ============================================================
# Part 1: VQE Workflow Configuration
# ============================================================

@dataclass
class VQEConfig:
    """Configuration for VQE workflow."""
    n_qubits: int
    n_layers: int
    optimizer: str = 'COBYLA'
    max_iterations: int = 100
    shots: int = 1000
    convergence_threshold: float = 1e-4
    patience: int = 5
    verbose: bool = True
    save_history: bool = True

@dataclass
class VQEResult:
    """Results from VQE optimization."""
    optimal_energy: float
    optimal_params: np.ndarray
    energy_history: List[float]
    param_history: List[np.ndarray]
    gradient_history: List[float]
    n_iterations: int
    n_function_evals: int
    converged: bool
    total_time: float

# ============================================================
# Part 2: Ansatz Builder
# ============================================================

class AnsatzBuilder:
    """Build parameterized quantum circuits."""

    @staticmethod
    def hardware_efficient(n_qubits: int, n_layers: int,
                          params: np.ndarray) -> QuantumCircuit:
        """Hardware-efficient ansatz with RY-RZ rotations."""
        qc = QuantumCircuit(n_qubits)
        param_idx = 0

        for layer in range(n_layers):
            for q in range(n_qubits):
                qc.ry(params[param_idx], q)
                param_idx += 1
                qc.rz(params[param_idx], q)
                param_idx += 1

            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)

        # Final rotation layer
        for q in range(n_qubits):
            qc.ry(params[param_idx], q)
            param_idx += 1

        return qc

    @staticmethod
    def count_params(n_qubits: int, n_layers: int) -> int:
        """Count parameters in hardware-efficient ansatz."""
        return 2 * n_qubits * n_layers + n_qubits

# ============================================================
# Part 3: Hamiltonian Builder
# ============================================================

class HamiltonianBuilder:
    """Build molecular Hamiltonians."""

    @staticmethod
    def h2_hamiltonian(bond_length: float = 0.735) -> SparsePauliOp:
        """H2 Hamiltonian in STO-3G basis."""
        # Coefficients for H2 at given bond length
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
        return SparsePauliOp.from_list(pauli_list)

# ============================================================
# Part 4: Callback and Monitoring
# ============================================================

class OptimizationCallback:
    """Callback for monitoring optimization."""

    def __init__(self, config: VQEConfig):
        self.config = config
        self.energy_history = []
        self.param_history = []
        self.gradient_norms = []
        self.iteration = 0
        self.converged = False
        self.patience_counter = 0

    def __call__(self, params: np.ndarray, energy: float,
                 gradient_norm: float = None) -> bool:
        """
        Called after each iteration.
        Returns True to stop optimization.
        """
        self.iteration += 1
        self.energy_history.append(energy)
        self.param_history.append(params.copy())

        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)

        # Check convergence
        if len(self.energy_history) >= 2:
            delta_E = abs(self.energy_history[-1] - self.energy_history[-2])

            if delta_E < self.config.convergence_threshold:
                self.patience_counter += 1
            else:
                self.patience_counter = 0

            if self.patience_counter >= self.config.patience:
                self.converged = True
                if self.config.verbose:
                    print(f"Converged at iteration {self.iteration}!")
                return True

        # Verbose output
        if self.config.verbose and self.iteration % 10 == 0:
            grad_str = f", |∇|={gradient_norm:.4f}" if gradient_norm else ""
            print(f"Iter {self.iteration}: E = {energy:.6f} Ha{grad_str}")

        return False

# ============================================================
# Part 5: VQE Engine
# ============================================================

class VQEEngine:
    """Main VQE optimization engine."""

    def __init__(self, config: VQEConfig):
        self.config = config
        self.estimator = Estimator()
        self.n_evals = 0

    def energy_function(self, params: np.ndarray,
                        hamiltonian: SparsePauliOp) -> float:
        """Compute energy expectation value."""
        circuit = AnsatzBuilder.hardware_efficient(
            self.config.n_qubits,
            self.config.n_layers,
            params
        )

        job = self.estimator.run([(circuit, hamiltonian)])
        result = job.result()
        self.n_evals += 1

        return float(result[0].data.evs)

    def parameter_shift_gradient(self, params: np.ndarray,
                                  hamiltonian: SparsePauliOp) -> np.ndarray:
        """Compute gradient using parameter shift rule."""
        gradient = np.zeros_like(params)
        shift = np.pi / 2

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += shift

            params_minus = params.copy()
            params_minus[i] -= shift

            e_plus = self.energy_function(params_plus, hamiltonian)
            e_minus = self.energy_function(params_minus, hamiltonian)

            gradient[i] = (e_plus - e_minus) / 2

        return gradient

    def run(self, hamiltonian: SparsePauliOp,
            initial_params: np.ndarray = None) -> VQEResult:
        """Run VQE optimization."""
        start_time = time.time()

        # Initialize parameters
        n_params = AnsatzBuilder.count_params(
            self.config.n_qubits, self.config.n_layers
        )

        if initial_params is None:
            initial_params = np.random.uniform(-0.1, 0.1, n_params)

        # Setup callback
        callback = OptimizationCallback(self.config)

        # Cost function wrapper
        def cost_fn(params):
            energy = self.energy_function(params, hamiltonian)
            callback(params, energy)
            return energy

        # Run optimizer
        if self.config.verbose:
            print(f"Starting VQE with {n_params} parameters...")
            print(f"Optimizer: {self.config.optimizer}")
            print(f"Initial energy: {cost_fn(initial_params):.6f} Ha\n")

        if self.config.optimizer == 'COBYLA':
            result = minimize(
                cost_fn,
                initial_params,
                method='COBYLA',
                options={'maxiter': self.config.max_iterations, 'rhobeg': 0.5}
            )
        elif self.config.optimizer == 'L-BFGS-B':
            result = minimize(
                cost_fn,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': self.config.max_iterations}
            )
        elif self.config.optimizer == 'SPSA':
            result = self._run_spsa(initial_params, hamiltonian, callback)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        total_time = time.time() - start_time

        return VQEResult(
            optimal_energy=callback.energy_history[-1],
            optimal_params=callback.param_history[-1],
            energy_history=callback.energy_history,
            param_history=callback.param_history,
            gradient_history=callback.gradient_norms,
            n_iterations=callback.iteration,
            n_function_evals=self.n_evals,
            converged=callback.converged,
            total_time=total_time
        )

    def _run_spsa(self, initial_params: np.ndarray,
                  hamiltonian: SparsePauliOp,
                  callback: OptimizationCallback):
        """SPSA optimization."""
        params = initial_params.copy()
        n_params = len(params)

        # SPSA parameters
        a = 0.1
        c = 0.1
        A = 10
        alpha = 0.602
        gamma = 0.101

        for k in range(self.config.max_iterations):
            # Compute coefficients
            a_k = a / (k + 1 + A)**alpha
            c_k = c / (k + 1)**gamma

            # Random perturbation
            delta = 2 * np.random.randint(0, 2, n_params) - 1

            # Evaluate perturbed points
            e_plus = self.energy_function(params + c_k * delta, hamiltonian)
            e_minus = self.energy_function(params - c_k * delta, hamiltonian)

            # Gradient estimate
            g_k = (e_plus - e_minus) / (2 * c_k * delta)

            # Update
            params = params - a_k * g_k

            # Current energy
            energy = self.energy_function(params, hamiltonian)

            if callback(params, energy, np.linalg.norm(g_k)):
                break

        class SPSAResult:
            x = params
            fun = energy

        return SPSAResult()

# ============================================================
# Part 6: Run Complete VQE
# ============================================================

print("="*60)
print("Complete VQE Workflow")
print("="*60)

# Configuration
config = VQEConfig(
    n_qubits=4,
    n_layers=2,
    optimizer='COBYLA',
    max_iterations=100,
    convergence_threshold=1e-4,
    patience=5,
    verbose=True
)

# Build Hamiltonian
H = HamiltonianBuilder.h2_hamiltonian(0.735)
print(f"\nH2 Hamiltonian: {len(H)} terms")

# Run VQE
engine = VQEEngine(config)
result = engine.run(H)

# Report results
print("\n" + "="*60)
print("VQE Results")
print("="*60)
print(f"Optimal energy: {result.optimal_energy:.6f} Ha")
print(f"Exact energy (FCI): -1.1373 Ha")
print(f"Error: {abs(result.optimal_energy - (-1.1373))*1000:.2f} mHa")
print(f"Iterations: {result.n_iterations}")
print(f"Function evaluations: {result.n_function_evals}")
print(f"Converged: {result.converged}")
print(f"Total time: {result.total_time:.2f} s")

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(result.energy_history, 'b-', linewidth=2)
axes[0].axhline(y=-1.1373, color='r', linestyle='--', label='Exact (FCI)')
axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('Energy (Ha)', fontsize=12)
axes[0].set_title('VQE Convergence', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Energy difference from optimal
energy_diff = np.abs(np.array(result.energy_history) - (-1.1373))
axes[1].semilogy(energy_diff * 1000, 'b-', linewidth=2)
axes[1].axhline(y=1.6, color='g', linestyle='--', label='Chemical accuracy')
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('Energy Error (mHa)', fontsize=12)
axes[1].set_title('Error Convergence', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vqe_workflow_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 7: Compare Optimizers
# ============================================================

print("\n" + "="*60)
print("Optimizer Comparison")
print("="*60)

optimizers = ['COBYLA', 'SPSA']
optimizer_results = {}

for opt in optimizers:
    print(f"\nTesting {opt}...")
    config.optimizer = opt
    config.verbose = False

    engine = VQEEngine(config)
    np.random.seed(42)  # Same initial point
    result = engine.run(H)

    optimizer_results[opt] = result
    print(f"  Final energy: {result.optimal_energy:.6f} Ha")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Time: {result.total_time:.2f} s")

# Plot comparison
plt.figure(figsize=(10, 6))
for opt, result in optimizer_results.items():
    plt.plot(result.energy_history, linewidth=2, label=opt)

plt.axhline(y=-1.1373, color='k', linestyle='--', alpha=0.5, label='Exact')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Energy (Ha)', fontsize=12)
plt.title('Optimizer Comparison for VQE', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab complete!")
```

### Lab 2: Shot Budget Optimization

```python
"""
Day 951 Lab Part 2: Shot Budget Optimization
Efficient measurement strategies for VQE
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

# ============================================================
# Part 1: Measurement Grouping
# ============================================================

def group_commuting_paulis(pauli_list: list) -> list:
    """
    Group Pauli strings that can be measured together.
    Uses qubit-wise commuting (QWC) criterion.
    """

    def qwc_commute(p1: str, p2: str) -> bool:
        """Check if two Pauli strings qubit-wise commute."""
        for c1, c2 in zip(p1, p2):
            if c1 != 'I' and c2 != 'I' and c1 != c2:
                return False
        return True

    groups = []
    used = set()

    for i, p1 in enumerate(pauli_list):
        if i in used:
            continue

        group = [p1]
        used.add(i)

        for j, p2 in enumerate(pauli_list):
            if j in used:
                continue

            # Check if p2 commutes with all in group
            if all(qwc_commute(p2, p) for p in group):
                group.append(p2)
                used.add(j)

        groups.append(group)

    return groups

# Example: H2 Hamiltonian terms
h2_terms = ['IIII', 'IIIZ', 'IIZI', 'IZII', 'ZIII',
            'IIZZ', 'IZIZ', 'IZZI', 'ZIIZ', 'ZIZI', 'ZZII',
            'XXXX', 'XXYY', 'YYXX', 'YYYY']

groups = group_commuting_paulis(h2_terms)

print("Measurement Grouping for H2:")
print(f"Original terms: {len(h2_terms)}")
print(f"Grouped into: {len(groups)} measurement circuits")
print("\nGroups:")
for i, group in enumerate(groups):
    print(f"  Group {i+1}: {group}")

# ============================================================
# Part 2: Optimal Shot Allocation
# ============================================================

def optimal_shot_allocation(coefficients: np.ndarray,
                            total_shots: int,
                            expectation_estimates: np.ndarray = None) -> np.ndarray:
    """
    Compute optimal shot allocation to minimize variance.

    N_i* ∝ |c_i| * sqrt(1 - <P_i>^2)
    """
    n_terms = len(coefficients)

    if expectation_estimates is None:
        # Worst case: assume <P_i> ≈ 0
        expectation_estimates = np.zeros(n_terms)

    # Optimal weights
    weights = np.abs(coefficients) * np.sqrt(1 - expectation_estimates**2)
    weights = weights / weights.sum()

    # Allocate shots
    shots = np.round(weights * total_shots).astype(int)

    # Ensure at least 1 shot per term
    shots = np.maximum(shots, 1)

    # Adjust to match total
    while shots.sum() > total_shots:
        idx = np.argmax(shots)
        shots[idx] -= 1
    while shots.sum() < total_shots:
        idx = np.argmin(shots)
        shots[idx] += 1

    return shots

# Example with H2 coefficients
coefficients = np.array([0.8105, 0.1721, 0.2257, 0.1721, 0.2257,
                         0.1209, 0.1689, 0.0454, 0.0454, 0.1689, 0.1209,
                         0.0454, 0.0454, 0.0454, 0.0454])

total_shots = 10000

uniform_shots = np.ones(len(coefficients)) * total_shots / len(coefficients)
optimal_shots = optimal_shot_allocation(coefficients, total_shots)

print("\nShot Allocation Comparison:")
print(f"Total shots: {total_shots}")
print(f"\n{'Term':<6} {'Coeff':>8} {'Uniform':>10} {'Optimal':>10}")
print("-" * 40)
for i in range(len(coefficients)):
    print(f"{i:<6} {coefficients[i]:>8.4f} {uniform_shots[i]:>10.0f} {optimal_shots[i]:>10}")

# Compare variances
def compute_variance(coeffs, shots):
    """Compute energy variance."""
    return np.sum(coeffs**2 / shots)

var_uniform = compute_variance(coefficients, uniform_shots)
var_optimal = compute_variance(coefficients, optimal_shots)

print(f"\nVariance comparison:")
print(f"  Uniform: {var_uniform:.6f}")
print(f"  Optimal: {var_optimal:.6f}")
print(f"  Reduction: {(1 - var_optimal/var_uniform)*100:.1f}%")

# ============================================================
# Part 3: Adaptive Shot Strategy
# ============================================================

def adaptive_vqe_shots(iteration: int, max_iterations: int,
                       min_shots: int = 100, max_shots: int = 10000,
                       strategy: str = 'linear') -> int:
    """
    Compute shots for given iteration.
    More shots as optimization converges.
    """
    if strategy == 'linear':
        return int(min_shots + (max_shots - min_shots) * iteration / max_iterations)
    elif strategy == 'quadratic':
        t = iteration / max_iterations
        return int(min_shots + (max_shots - min_shots) * t**2)
    elif strategy == 'exponential':
        t = iteration / max_iterations
        return int(min_shots * (max_shots / min_shots)**t)
    else:
        return max_shots

# Visualize strategies
iterations = np.arange(100)
strategies = ['linear', 'quadratic', 'exponential']

plt.figure(figsize=(10, 6))
for strategy in strategies:
    shots = [adaptive_vqe_shots(i, 100, strategy=strategy) for i in iterations]
    plt.plot(iterations, shots, linewidth=2, label=strategy)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Shots', fontsize=12)
plt.title('Adaptive Shot Strategies', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('adaptive_shots.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 4: Variance Reduction Analysis
# ============================================================

def analyze_shot_variance(n_shots_range: np.ndarray,
                          coefficients: np.ndarray) -> dict:
    """Analyze energy variance vs shot count."""
    variances_uniform = []
    variances_optimal = []

    for n_shots in n_shots_range:
        uniform = np.ones(len(coefficients)) * n_shots / len(coefficients)
        optimal = optimal_shot_allocation(coefficients, int(n_shots))

        variances_uniform.append(compute_variance(coefficients, uniform))
        variances_optimal.append(compute_variance(coefficients, optimal))

    return {
        'shots': n_shots_range,
        'uniform': np.array(variances_uniform),
        'optimal': np.array(variances_optimal)
    }

shot_range = np.logspace(2, 5, 20).astype(int)
variance_analysis = analyze_shot_variance(shot_range, coefficients)

plt.figure(figsize=(10, 6))
plt.loglog(variance_analysis['shots'], np.sqrt(variance_analysis['uniform']),
           'b-', linewidth=2, label='Uniform allocation')
plt.loglog(variance_analysis['shots'], np.sqrt(variance_analysis['optimal']),
           'r-', linewidth=2, label='Optimal allocation')
plt.axhline(y=0.001, color='g', linestyle='--', label='1 mHa target')
plt.xlabel('Total Shots', fontsize=12)
plt.ylabel('Energy Standard Deviation (Ha)', fontsize=12)
plt.title('Shot Requirement Analysis', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('shot_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# Find shots needed for 1 mHa precision
target_std = 0.001
idx = np.argmin(np.abs(np.sqrt(variance_analysis['optimal']) - target_std))
print(f"\nShots needed for 1 mHa precision:")
print(f"  Optimal allocation: ~{variance_analysis['shots'][idx]:.0f}")

idx_uniform = np.argmin(np.abs(np.sqrt(variance_analysis['uniform']) - target_std))
print(f"  Uniform allocation: ~{variance_analysis['shots'][idx_uniform]:.0f}")
print(f"  Savings: {(1 - variance_analysis['shots'][idx]/variance_analysis['shots'][idx_uniform])*100:.0f}%")

# ============================================================
# Part 5: Grouping + Optimal Allocation
# ============================================================

def compute_group_coefficients(groups: list, term_coeffs: dict) -> np.ndarray:
    """Compute effective coefficient for each measurement group."""
    group_coeffs = []
    for group in groups:
        # Sum of |c_i|^2 for terms in group (for variance)
        coeff_sq_sum = sum(term_coeffs.get(term, 0)**2 for term in group)
        group_coeffs.append(np.sqrt(coeff_sq_sum))
    return np.array(group_coeffs)

# Map term to coefficient
term_coeffs = dict(zip(h2_terms, coefficients))

group_coeffs = compute_group_coefficients(groups, term_coeffs)

print("\nGroup-level analysis:")
for i, (group, coeff) in enumerate(zip(groups, group_coeffs)):
    print(f"  Group {i+1}: {len(group)} terms, effective |c| = {coeff:.4f}")

# Optimal allocation across groups
group_shots = optimal_shot_allocation(group_coeffs, total_shots)

print(f"\nOptimal shots per group (total = {total_shots}):")
for i, (group, shots) in enumerate(zip(groups, group_shots)):
    print(f"  Group {i+1}: {shots} shots")

print("\nLab complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| SPSA Update | $\theta_{k+1} = \theta_k - a_k \frac{f^+ - f^-}{2c_k\Delta^{-1}}$ |
| Shot Noise | $\sigma_E^2 = \sum_i c_i^2(1-\langle P_i\rangle^2)/N_i$ |
| Optimal Allocation | $N_i^* \propto |c_i|\sqrt{1-\langle P_i\rangle^2}$ |
| Natural Gradient | $\theta_{k+1} = \theta_k - \eta F^{-1}\nabla\mathcal{L}$ |
| Adam Update | $\theta_{k+1} = \theta_k - \eta\hat{m}_k/(\sqrt{\hat{v}_k}+\epsilon)$ |

### Key Takeaways

1. **Optimizer choice matters:** SPSA and COBYLA work well for noisy quantum data; gradient-based methods need error mitigation.

2. **Shot budget optimization** can reduce measurement cost by 30-50% through optimal allocation.

3. **Grouping commuting observables** reduces circuit count, sometimes dramatically.

4. **Adaptive strategies** allocate more shots as optimization converges for efficiency.

5. **Complete workflows** integrate compilation, execution, mitigation, and optimization.

6. **Callbacks and monitoring** enable debugging and early stopping.

## Daily Checklist

- [ ] I can select appropriate classical optimizers for VQE/QAOA
- [ ] I understand shot budget optimization and allocation
- [ ] I can implement measurement grouping for efficiency
- [ ] I can design complete hybrid workflows with callbacks
- [ ] I understand error-aware optimization strategies
- [ ] I completed the computational labs on VQE workflows

## Preview of Day 952

Tomorrow we synthesize Month 34 with a comprehensive review:
- NISQ era summary: capabilities and limitations
- Algorithm landscape: VQE, QAOA, and beyond
- Key challenges: barren plateaus, noise, compilation
- Transition to fault tolerance: what changes?
- Future outlook: near-term quantum advantage
- Practice problems spanning all week's topics
