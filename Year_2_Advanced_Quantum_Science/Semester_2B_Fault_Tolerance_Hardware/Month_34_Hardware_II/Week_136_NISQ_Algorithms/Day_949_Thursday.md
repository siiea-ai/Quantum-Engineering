# Day 949: Barren Plateaus

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 2.5 hours | Barren plateau theory and mathematical foundations |
| Afternoon | 2.5 hours | Problem solving and mitigation strategies |
| Evening | 2 hours | Computational lab: Gradient analysis |

## Learning Objectives

By the end of today, you will be able to:

1. Define barren plateaus and understand their exponential gradient vanishing
2. Derive the variance scaling of gradients in deep random circuits
3. Analyze the trade-off between expressibility and trainability
4. Apply mitigation strategies: layerwise training, identity initialization, local cost functions
5. Distinguish cost-function-induced vs circuit-induced barren plateaus
6. Design circuits that avoid barren plateaus while maintaining expressibility

## Core Content

### 1. The Barren Plateau Problem

**Definition:** A barren plateau is a region of the parameter landscape where gradients vanish exponentially with system size, making optimization intractable.

For a parameterized quantum circuit $U(\boldsymbol{\theta})$ and cost function $\mathcal{L}$:

$$\boxed{\text{Var}\left[\frac{\partial\mathcal{L}}{\partial\theta_k}\right] \leq F(n) \in O(e^{-cn})}$$

where $n$ is the number of qubits and $c > 0$ is a constant.

**Consequence:** With exponentially small gradients:
- Random initialization lands in a barren plateau with high probability
- Finite-precision gradients become indistinguishable from zero
- Optimization requires exponentially many samples

### 2. Mathematical Foundation

#### 2.1 Gradient of Parameterized Circuits

For a unitary $U(\boldsymbol{\theta})$ and observable $\hat{O}$:

$$\mathcal{L}(\boldsymbol{\theta}) = \langle 0|U^\dagger(\boldsymbol{\theta})\hat{O}U(\boldsymbol{\theta})|0\rangle$$

The gradient with respect to parameter $\theta_k$ (using parameter shift):

$$\frac{\partial\mathcal{L}}{\partial\theta_k} = \frac{1}{2}\left(\mathcal{L}(\theta_k + \frac{\pi}{2}) - \mathcal{L}(\theta_k - \frac{\pi}{2})\right)$$

#### 2.2 Variance Analysis

Consider a circuit $U(\boldsymbol{\theta}) = V(\theta_k)W$ where $V(\theta_k) = e^{-i\theta_k G/2}$ and $W$ contains all other parameters.

The gradient can be written as:

$$\frac{\partial\mathcal{L}}{\partial\theta_k} = i\langle[\hat{O}', G]\rangle$$

where $\hat{O}' = W^\dagger U_L^\dagger \hat{O} U_L W$ with $U_L$ being gates after $\theta_k$.

**McClean et al. (2018) result:**

For random circuits forming approximate 2-designs:

$$\boxed{\text{Var}_{\boldsymbol{\theta}}\left[\frac{\partial\mathcal{L}}{\partial\theta_k}\right] \leq \frac{1}{2^n} \cdot \text{Tr}[\hat{O}^2]}$$

### 3. Sources of Barren Plateaus

#### 3.1 Circuit-Induced Barren Plateaus

**Cause:** Highly expressive circuits approach Haar-random behavior.

For a circuit that forms an approximate 2-design on the full Hilbert space:

$$\mathbb{E}_U[\langle\psi|U^\dagger \hat{O} U|\psi\rangle] = \frac{\text{Tr}[\hat{O}]}{2^n}$$

$$\text{Var}_U[\langle\psi|U^\dagger \hat{O} U|\psi\rangle] = \frac{\text{Tr}[\hat{O}^2] - \text{Tr}[\hat{O}]^2/2^n}{2^{2n} - 1}$$

**Scaling:** Variance decays as $O(1/2^n)$.

#### 3.2 Cost-Function-Induced Barren Plateaus

**Cause:** Global cost functions that act on all qubits.

**Global cost:**
$$\mathcal{L}_{\text{global}} = \text{Tr}[\hat{O}U\rho U^\dagger]$$

where $\hat{O}$ acts non-trivially on all qubits.

**Local cost:**
$$\mathcal{L}_{\text{local}} = \sum_i \text{Tr}[\hat{O}_i U\rho U^\dagger]$$

where each $\hat{O}_i$ acts on $O(1)$ qubits.

**Cerezo et al. (2021) result:**

| Cost Type | Gradient Variance |
|-----------|------------------|
| Global | $O(1/2^n)$ - severe BP |
| Local | $O(1/\text{poly}(n))$ - trainable |

#### 3.3 Noise-Induced Barren Plateaus

**Cause:** Quantum noise effectively scrambles information.

For depolarizing noise with rate $p$ per layer:

$$\text{Var}\left[\frac{\partial\mathcal{L}}{\partial\theta}\right] \leq (1-p)^{2L} \cdot \text{Var}_0$$

where $L$ is the circuit depth.

**Scaling:** Exponential suppression in depth, not just width.

### 4. Expressibility vs Trainability Trade-off

**Key insight:** More expressive circuits are harder to train.

**Expressibility:** Ability to reach arbitrary states in Hilbert space.

$$\mathcal{E} = D_{KL}\left(P_U(F) \| P_{\text{Haar}}(F)\right)$$

where $F = |\langle\psi|U|\phi\rangle|^2$ is the fidelity between states.

**Trainability:** Non-vanishing gradients that enable optimization.

$$\mathcal{T} \propto \text{Var}\left[\nabla_\theta \mathcal{L}\right]$$

**The dilemma:**

```
High Expressibility ──► Forms t-design ──► Barren Plateau ──► Untrainable
Low Expressibility ──► Limited states ──► Good gradients ──► Trainable but limited
```

### 5. Mitigation Strategies

#### 5.1 Structured Ansatzes

**Hardware-efficient with limited depth:**
Keep circuit depth $L = O(\log n)$ to avoid 2-design behavior.

**Locality-preserving ansatzes:**
Restrict entanglement growth to maintain trainability.

$$U = \prod_{l=1}^L \prod_{i} U_{i,i+1}^{(l)}$$

where each $U_{i,i+1}$ only entangles neighboring qubits.

#### 5.2 Initialization Strategies

**Identity initialization:**
Initialize parameters so $U(\boldsymbol{\theta}_0) \approx I$.

For $R_Y(\theta)R_Z(\phi)$ layers, set $\theta = \phi = 0$.

**Warm starting:**
Initialize with classically-obtained solution.

$$|\psi_0\rangle = \text{Classical approximation to solution}$$

#### 5.3 Layerwise Training

Train circuit incrementally:
1. Train first layer with few parameters
2. Fix layer 1, add and train layer 2
3. Continue until desired depth

$$\theta_l^* = \arg\min_{\theta_l} \mathcal{L}(U_1^* \cdot U_2^* \cdots U_l(\theta_l))$$

#### 5.4 Local Cost Functions

Replace global observables with local ones:

**Global (bad):**
$$\mathcal{L} = 1 - |\langle\psi_{\text{target}}|\psi(\theta)\rangle|^2$$

**Local (better):**
$$\mathcal{L} = \sum_i (1 - \text{Tr}[\rho_i^{\text{target}} \rho_i(\theta)])$$

where $\rho_i$ is the reduced density matrix on qubit $i$.

#### 5.5 Parameter Correlations

Reduce effective parameter count by sharing parameters:

$$U = \prod_l \prod_i R_Y(\theta_l)$$

Same $\theta_l$ for all qubits in layer $l$.

### 6. Detecting Barren Plateaus

**Gradient variance sampling:**
1. Sample random parameter sets
2. Compute gradients numerically
3. Calculate variance over samples
4. Plot variance vs system size

**Warning signs:**
- Variance decreasing exponentially with $n$
- Gradient magnitudes $\ll 10^{-6}$ for $n > 10$
- Optimization stalling regardless of learning rate

## Quantum Computing Applications

### Application: Ansatz Design Guidelines

From barren plateau theory, practical guidelines emerge:

| Design Choice | Barren Plateau Risk | Recommendation |
|---------------|---------------------|----------------|
| Depth $L$ | High if $L = O(n)$ | Keep $L = O(\log n)$ |
| Entanglement | High if all-to-all | Use local entanglement |
| Cost function | High if global | Use local terms |
| Initialization | High if random | Use structured init |

### Application: VQE Circuit Design

For VQE, barren plateau considerations suggest:
- Use problem-inspired ansatzes (UCCSD) over generic HEA
- Keep entanglement structure matching Hamiltonian locality
- Measure local Pauli terms individually rather than global operators

## Worked Examples

### Example 1: Variance Scaling Derivation

**Problem:** Show that for a Haar-random unitary and computational basis observable $\hat{O} = |0\rangle\langle 0|$, the variance of $\langle\hat{O}\rangle$ scales as $O(1/2^n)$.

**Solution:**

Step 1: Mean calculation
For Haar-random $U$:
$$\mathbb{E}_U[\langle 0|U^\dagger|0\rangle\langle 0|U|0\rangle] = \mathbb{E}_U[|\langle 0|U|0\rangle|^2]$$

$$= \int dU |\langle 0|U|0\rangle|^2 = \frac{1}{2^n}$$

Step 2: Second moment
$$\mathbb{E}_U[|\langle 0|U|0\rangle|^4] = \frac{2}{2^n(2^n + 1)}$$

Step 3: Variance
$$\text{Var} = \mathbb{E}[X^2] - \mathbb{E}[X]^2 = \frac{2}{2^n(2^n+1)} - \frac{1}{2^{2n}}$$

$$= \frac{2 \cdot 2^n - (2^n + 1)}{2^{2n}(2^n + 1)} = \frac{2^n - 1}{2^{2n}(2^n + 1)} \approx \frac{1}{2^{2n}}$$

**Answer:** Variance scales as $O(1/2^{2n})$ for this observable.

### Example 2: Layer Depth Threshold

**Problem:** Estimate the critical depth $L^*$ at which a hardware-efficient ansatz begins to form an approximate 2-design on 10 qubits.

**Solution:**

Step 1: 2-design formation criterion
Approximate 2-designs form when circuit explores most of Hilbert space.

Rule of thumb: $L^* \approx O(n)$ for linear connectivity.

Step 2: More precise estimate
For alternating layers of single-qubit gates and nearest-neighbor CNOTs:
$$L^* \approx n \text{ to } 2n$$

For 10 qubits: $L^* \approx 10$ to $20$ layers.

Step 3: Verification approach
Compute frame potential:
$$\mathcal{F}^{(2)} = \int dU dV |\langle U, V\rangle|^4$$

2-design achieved when $\mathcal{F}^{(2)} \rightarrow \mathcal{F}^{(2)}_{\text{Haar}}$.

**Answer:** Barren plateaus likely emerge around $L \approx 10-20$ layers for 10 qubits with linear connectivity.

### Example 3: Gradient Variance Estimation

**Problem:** For a VQE circuit on 8 qubits with 50 parameters, estimate the expected gradient magnitude if the circuit forms an approximate 2-design.

**Solution:**

Step 1: Variance bound
$$\text{Var}[\partial_\theta \mathcal{L}] \leq \frac{1}{2^n} = \frac{1}{2^8} = \frac{1}{256}$$

Step 2: Standard deviation
$$\sigma = \sqrt{\text{Var}} \leq \frac{1}{16} \approx 0.0625$$

Step 3: Expected gradient magnitude
For 50 parameters sampled from this distribution:
$$|\nabla \mathcal{L}| \approx \sqrt{50} \cdot \sigma \approx 7 \times 0.0625 = 0.44$$

Step 4: Per-component magnitude
$$|\partial_\theta \mathcal{L}| \sim 0.0625$$

Step 5: Practical consideration
With shot noise ($N = 10000$ shots):
$$\sigma_{\text{shot}} \approx \frac{1}{\sqrt{N}} = 0.01$$

Gradient signal-to-noise ratio: $\approx 6$ (still detectable).

**Answer:** Expected gradient component magnitude $\approx 0.06$, still above shot noise for 8 qubits.

## Practice Problems

### Level 1: Direct Application

1. **Variance calculation:** For a circuit on 12 qubits approaching a 2-design, estimate the gradient variance.

2. **Local vs global:** Classify these cost functions as local or global:
   - (a) $\mathcal{L} = 1 - |\langle +|^{\otimes n} U|0\rangle^{\otimes n}|^2$
   - (b) $\mathcal{L} = \sum_i (1 - \langle Z_i\rangle^2)$
   - (c) $\mathcal{L} = \langle\psi|H|\psi\rangle$ where $H$ has 2-local terms

3. **Critical depth:** If barren plateaus emerge at depth $L^* = 2n$ for $n$ qubits, what is the maximum trainable depth for 20 qubits?

### Level 2: Intermediate Analysis

4. **Noise analysis:** A circuit has 30 layers and each layer has depolarizing noise with $p = 0.001$. By what factor is the gradient variance suppressed compared to the noiseless case?

5. **Expressibility trade-off:** A restricted ansatz can only prepare $2^{n/2}$ distinct states. Estimate how this affects gradient variance compared to a fully expressive ansatz.

6. **Layerwise training:** For an 8-layer circuit, design a layerwise training schedule. How many optimization problems must be solved?

### Level 3: Challenging Problems

7. **Theoretical derivation:** Prove that for a cost function $\mathcal{L} = \text{Tr}[\hat{O}\rho(\theta)]$ where $\hat{O}$ acts non-trivially on $k$ qubits, the gradient variance scales as $O(1/2^{k})$, not $O(1/2^n)$.

8. **Initialization analysis:** Suppose parameters are initialized as $\theta_i \sim \mathcal{N}(0, \sigma^2)$ rather than uniformly. How does $\sigma$ affect barren plateau onset?

9. **Entanglement entropy:** Show that a circuit with bounded entanglement entropy $S_{\max}$ avoids barren plateaus if $S_{\max} = O(\log n)$.

## Computational Lab: Barren Plateau Analysis

### Lab 1: Gradient Variance Scaling

```python
"""
Day 949 Lab: Barren Plateau Analysis
Demonstrating exponential gradient vanishing
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp, random_statevector
import pennylane as qml
from pennylane import numpy as pnp

# ============================================================
# Part 1: Gradient Variance vs System Size (Qiskit)
# ============================================================

def create_hardware_efficient_ansatz(n_qubits: int, n_layers: int,
                                      params: np.ndarray) -> QuantumCircuit:
    """Create HEA circuit."""
    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for layer in range(n_layers):
        # Rotation layer
        for q in range(n_qubits):
            qc.ry(params[param_idx], q)
            param_idx += 1
            qc.rz(params[param_idx], q)
            param_idx += 1

        # Entangling layer
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc

def compute_gradient_variance(n_qubits: int, n_layers: int,
                              n_samples: int = 50) -> dict:
    """
    Estimate gradient variance by sampling random parameters.
    """
    n_params = 2 * n_qubits * n_layers
    estimator = Estimator()

    # Observable: Z on first qubit (local)
    obs_local = SparsePauliOp.from_list([('Z' + 'I'*(n_qubits-1), 1.0)])

    # Observable: Z on all qubits (global)
    obs_global = SparsePauliOp.from_list([('Z'*n_qubits, 1.0)])

    gradients_local = []
    gradients_global = []

    for _ in range(n_samples):
        # Random parameters
        params = np.random.uniform(0, 2*np.pi, n_params)

        # Compute gradient using parameter shift (for first parameter)
        params_plus = params.copy()
        params_plus[0] += np.pi/2
        params_minus = params.copy()
        params_minus[0] -= np.pi/2

        # Local observable gradient
        qc_plus = create_hardware_efficient_ansatz(n_qubits, n_layers, params_plus)
        qc_minus = create_hardware_efficient_ansatz(n_qubits, n_layers, params_minus)

        job = estimator.run([(qc_plus, obs_local), (qc_minus, obs_local),
                             (qc_plus, obs_global), (qc_minus, obs_global)])
        results = job.result()

        grad_local = (results[0].data.evs - results[1].data.evs) / 2
        grad_global = (results[2].data.evs - results[3].data.evs) / 2

        gradients_local.append(grad_local)
        gradients_global.append(grad_global)

    return {
        'local_mean': np.mean(gradients_local),
        'local_var': np.var(gradients_local),
        'global_mean': np.mean(gradients_global),
        'global_var': np.var(gradients_global)
    }

# Analyze scaling with system size
print("Analyzing gradient variance scaling...")
qubit_range = range(2, 10)
n_layers = 3  # Fixed depth

local_variances = []
global_variances = []

for n in qubit_range:
    print(f"  n = {n} qubits...")
    result = compute_gradient_variance(n, n_layers, n_samples=30)
    local_variances.append(result['local_var'])
    global_variances.append(result['global_var'])
    print(f"    Local var: {result['local_var']:.6f}")
    print(f"    Global var: {result['global_var']:.6f}")

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(list(qubit_range), local_variances, 'bo-', markersize=10,
             linewidth=2, label='Local cost')
plt.semilogy(list(qubit_range), global_variances, 'r^-', markersize=10,
             linewidth=2, label='Global cost')
plt.xlabel('Number of Qubits', fontsize=12)
plt.ylabel('Gradient Variance', fontsize=12)
plt.title('Gradient Variance Scaling', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Fit exponential
from scipy.optimize import curve_fit

def exp_decay(x, a, b):
    return a * np.exp(-b * x)

try:
    popt_global, _ = curve_fit(exp_decay, list(qubit_range), global_variances,
                                p0=[0.1, 0.5])
    x_fit = np.linspace(2, 10, 50)
    plt.subplot(1, 2, 1)
    plt.semilogy(x_fit, exp_decay(x_fit, *popt_global), 'r--', alpha=0.5,
                 label=f'Fit: exp(-{popt_global[1]:.2f}n)')
    plt.legend()
except:
    pass

plt.subplot(1, 2, 2)
# Ratio analysis
ratios = np.array(global_variances) / np.array(local_variances)
plt.plot(list(qubit_range), ratios, 'go-', markersize=10, linewidth=2)
plt.xlabel('Number of Qubits', fontsize=12)
plt.ylabel('Global/Local Variance Ratio', fontsize=12)
plt.title('Cost Function Comparison', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('barren_plateau_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 2: Depth Dependence
# ============================================================

def analyze_depth_dependence(n_qubits: int, max_layers: int) -> dict:
    """Analyze how gradient variance depends on circuit depth."""
    variances = []
    n_params_per_layer = 2 * n_qubits

    for n_layers in range(1, max_layers + 1):
        result = compute_gradient_variance(n_qubits, n_layers, n_samples=30)
        variances.append(result['global_var'])

    return {'depths': list(range(1, max_layers + 1)), 'variances': variances}

print("\nAnalyzing depth dependence (n=6 qubits)...")
depth_results = analyze_depth_dependence(6, 8)

plt.figure(figsize=(10, 6))
plt.semilogy(depth_results['depths'], depth_results['variances'],
             'bo-', markersize=10, linewidth=2)
plt.xlabel('Circuit Depth (layers)', fontsize=12)
plt.ylabel('Gradient Variance', fontsize=12)
plt.title('Gradient Variance vs Circuit Depth (n=6)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('depth_dependence.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 3: PennyLane Analysis with More Detail
# ============================================================

def analyze_with_pennylane(n_qubits: int, n_layers: int, n_samples: int = 100):
    """Detailed barren plateau analysis with PennyLane."""

    dev = qml.device('default.qubit', wires=n_qubits)
    n_params = 2 * n_qubits * n_layers

    @qml.qnode(dev, interface='autograd')
    def circuit_local(params):
        """Circuit with local cost (Z on qubit 0)."""
        param_idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q+1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev, interface='autograd')
    def circuit_global(params):
        """Circuit with global cost (product of Z)."""
        param_idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q+1])

        obs = qml.PauliZ(0)
        for q in range(1, n_qubits):
            obs = obs @ qml.PauliZ(q)
        return qml.expval(obs)

    # Compute gradients
    grad_fn_local = qml.grad(circuit_local)
    grad_fn_global = qml.grad(circuit_global)

    grads_local = []
    grads_global = []

    for _ in range(n_samples):
        params = pnp.random.uniform(0, 2*np.pi, n_params, requires_grad=True)
        g_local = grad_fn_local(params)
        g_global = grad_fn_global(params)
        grads_local.append(g_local[0])  # First parameter
        grads_global.append(g_global[0])

    return {
        'local_grads': np.array(grads_local),
        'global_grads': np.array(grads_global)
    }

print("\nDetailed PennyLane analysis (n=6, L=4)...")
pl_results = analyze_with_pennylane(6, 4, n_samples=200)

# Histogram of gradients
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(pl_results['local_grads'], bins=30, density=True,
             alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_xlabel('Gradient Value', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'Local Cost Gradients\nVar={np.var(pl_results["local_grads"]):.4f}', fontsize=14)

axes[1].hist(pl_results['global_grads'], bins=30, density=True,
             alpha=0.7, color='green', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--')
axes[1].set_xlabel('Gradient Value', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'Global Cost Gradients\nVar={np.var(pl_results["global_grads"]):.4f}', fontsize=14)

plt.tight_layout()
plt.savefig('gradient_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 4: Mitigation Strategy - Identity Initialization
# ============================================================

def compare_initializations(n_qubits: int, n_layers: int, n_trials: int = 20):
    """Compare random vs identity initialization."""

    dev = qml.device('default.qubit', wires=n_qubits)
    n_params = 2 * n_qubits * n_layers

    @qml.qnode(dev, interface='autograd')
    def circuit(params):
        param_idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q+1])

        # Simple cost: overlap with |0...0>
        return qml.probs(wires=range(n_qubits))[0]

    # Random initialization
    random_costs = []
    random_grads = []
    grad_fn = qml.grad(circuit)

    for _ in range(n_trials):
        params = pnp.random.uniform(0, 2*np.pi, n_params, requires_grad=True)
        random_costs.append(float(circuit(params)))
        g = grad_fn(params)
        random_grads.append(np.linalg.norm(g))

    # Identity initialization (small random around 0)
    identity_costs = []
    identity_grads = []

    for _ in range(n_trials):
        params = pnp.random.normal(0, 0.1, n_params, requires_grad=True)
        identity_costs.append(float(circuit(params)))
        g = grad_fn(params)
        identity_grads.append(np.linalg.norm(g))

    return {
        'random_costs': random_costs,
        'random_grads': random_grads,
        'identity_costs': identity_costs,
        'identity_grads': identity_grads
    }

print("\nComparing initialization strategies (n=8, L=4)...")
init_results = compare_initializations(8, 4)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cost comparison
axes[0].boxplot([init_results['random_costs'], init_results['identity_costs']],
                labels=['Random Init', 'Identity Init'])
axes[0].set_ylabel('Initial Cost', fontsize=12)
axes[0].set_title('Initial Cost Value', fontsize=14)
axes[0].grid(True, alpha=0.3, axis='y')

# Gradient magnitude comparison
axes[1].boxplot([init_results['random_grads'], init_results['identity_grads']],
                labels=['Random Init', 'Identity Init'])
axes[1].set_ylabel('Gradient Magnitude', fontsize=12)
axes[1].set_title('Initial Gradient Magnitude', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('initialization_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nRandom init - Mean gradient: {np.mean(init_results['random_grads']):.4f}")
print(f"Identity init - Mean gradient: {np.mean(init_results['identity_grads']):.4f}")

# ============================================================
# Part 5: Layerwise Training Demonstration
# ============================================================

def layerwise_training_demo(n_qubits: int = 6, total_layers: int = 4):
    """Demonstrate layerwise training strategy."""

    dev = qml.device('default.qubit', wires=n_qubits)

    def make_circuit(params_list, n_active_layers):
        """Create circuit with specified active layers."""
        @qml.qnode(dev, interface='autograd')
        def circuit(current_params):
            param_idx = 0

            # Apply all layers
            for layer in range(n_active_layers):
                if layer < n_active_layers - 1:
                    # Fixed layers
                    layer_params = params_list[layer]
                else:
                    # Active layer
                    layer_params = current_params

                for q in range(n_qubits):
                    qml.RY(layer_params[2*q], wires=q)
                    qml.RZ(layer_params[2*q + 1], wires=q)

                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q+1])

            return qml.expval(qml.PauliZ(0))

        return circuit

    # Target: minimize cost
    trained_params = []
    training_history = []

    for layer in range(1, total_layers + 1):
        print(f"\nTraining layer {layer}...")

        circuit = make_circuit(trained_params, layer)
        opt = qml.GradientDescentOptimizer(stepsize=0.1)

        # Initialize current layer
        current_params = pnp.zeros(2 * n_qubits, requires_grad=True)
        layer_history = []

        for step in range(50):
            current_params, cost = opt.step_and_cost(circuit, current_params)
            layer_history.append(float(cost))

            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}: Cost = {cost:.4f}")

        trained_params.append(current_params)
        training_history.append(layer_history)

    return training_history

print("\n" + "="*60)
print("Layerwise Training Demonstration")
print("="*60)

history = layerwise_training_demo(n_qubits=6, total_layers=4)

# Plot training curves
plt.figure(figsize=(12, 5))

colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
for i, layer_hist in enumerate(history):
    plt.plot(layer_hist, color=colors[i], linewidth=2,
             label=f'Layer {i+1}')

plt.xlabel('Optimization Step', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title('Layerwise Training Progress', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('layerwise_training.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab complete!")
```

### Lab 2: Expressibility vs Trainability

```python
"""
Day 949 Lab Part 2: Expressibility Analysis
Quantifying the expressibility-trainability trade-off
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pennylane as qml
from pennylane import numpy as pnp

# ============================================================
# Part 1: Expressibility Calculation
# ============================================================

def compute_expressibility(n_qubits: int, n_layers: int, n_samples: int = 1000):
    """
    Compute expressibility as KL divergence from Haar distribution.
    Uses fidelity distribution between random circuit states.
    """
    dev = qml.device('default.qubit', wires=n_qubits)
    n_params = 2 * n_qubits * n_layers

    @qml.qnode(dev)
    def circuit(params):
        param_idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q+1])
        return qml.state()

    # Sample fidelities
    fidelities = []
    for _ in range(n_samples):
        params1 = np.random.uniform(0, 2*np.pi, n_params)
        params2 = np.random.uniform(0, 2*np.pi, n_params)

        state1 = circuit(params1)
        state2 = circuit(params2)

        fid = np.abs(np.vdot(state1, state2))**2
        fidelities.append(fid)

    # Haar fidelity distribution: P(F) = (d-1)(1-F)^{d-2}
    d = 2**n_qubits

    # KL divergence approximation
    hist, bin_edges = np.histogram(fidelities, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Haar PDF
    haar_pdf = (d - 1) * (1 - bin_centers)**(d - 2)
    haar_pdf = haar_pdf / np.sum(haar_pdf) * len(bin_centers)

    # KL divergence (with smoothing)
    hist_smooth = hist + 1e-10
    haar_smooth = haar_pdf + 1e-10
    kl_div = np.sum(hist_smooth * np.log(hist_smooth / haar_smooth)) / len(hist)

    return {
        'fidelities': np.array(fidelities),
        'kl_divergence': kl_div,
        'hist': hist,
        'bin_centers': bin_centers,
        'haar_pdf': haar_pdf
    }

# Compare expressibility at different depths
print("Computing expressibility at different depths...")
n_qubits = 4
depths = [1, 2, 4, 8]

expr_results = {}
for n_layers in depths:
    print(f"  Depth = {n_layers}...")
    expr_results[n_layers] = compute_expressibility(n_qubits, n_layers, n_samples=500)
    print(f"    KL divergence: {expr_results[n_layers]['kl_divergence']:.4f}")

# Plot fidelity distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, n_layers in zip(axes, depths):
    result = expr_results[n_layers]
    ax.hist(result['fidelities'], bins=30, density=True, alpha=0.7,
            color='blue', edgecolor='black', label='Circuit')
    ax.plot(result['bin_centers'], result['haar_pdf'], 'r-',
            linewidth=2, label='Haar')
    ax.set_xlabel('Fidelity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Depth = {n_layers}, KL = {result["kl_divergence"]:.4f}', fontsize=14)
    ax.legend()

plt.tight_layout()
plt.savefig('expressibility_depth.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 2: Trainability at Different Depths
# ============================================================

def compute_trainability(n_qubits: int, n_layers: int, n_samples: int = 50):
    """Compute gradient variance as trainability measure."""
    dev = qml.device('default.qubit', wires=n_qubits)
    n_params = 2 * n_qubits * n_layers

    @qml.qnode(dev, interface='autograd')
    def circuit(params):
        param_idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q+1])
        return qml.expval(qml.PauliZ(0))

    grad_fn = qml.grad(circuit)
    gradients = []

    for _ in range(n_samples):
        params = pnp.random.uniform(0, 2*np.pi, n_params, requires_grad=True)
        g = grad_fn(params)
        gradients.append(g[0])  # First parameter gradient

    return np.var(gradients)

print("\nComputing trainability at different depths...")
trainabilities = []
for n_layers in depths:
    t = compute_trainability(n_qubits, n_layers, n_samples=100)
    trainabilities.append(t)
    print(f"  Depth = {n_layers}: Var = {t:.6f}")

# ============================================================
# Part 3: Expressibility vs Trainability Plot
# ============================================================

expressibilities = [expr_results[d]['kl_divergence'] for d in depths]

plt.figure(figsize=(10, 6))
plt.scatter(expressibilities, trainabilities, s=200, c=depths, cmap='viridis')

for i, d in enumerate(depths):
    plt.annotate(f'd={d}', (expressibilities[i], trainabilities[i]),
                 xytext=(10, 10), textcoords='offset points', fontsize=12)

plt.colorbar(label='Circuit Depth')
plt.xlabel('Expressibility (lower = more expressive)', fontsize=12)
plt.ylabel('Trainability (Gradient Variance)', fontsize=12)
plt.title('Expressibility vs Trainability Trade-off', fontsize=14)
plt.grid(True, alpha=0.3)

# Draw trade-off curve
z = np.polyfit(expressibilities, np.log(trainabilities), 1)
p = np.poly1d(z)
x_line = np.linspace(min(expressibilities), max(expressibilities), 100)
plt.plot(x_line, np.exp(p(x_line)), 'r--', alpha=0.5, label='Trend')
plt.legend()

plt.savefig('expr_train_tradeoff.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Part 4: Local vs Global Cost at Scale
# ============================================================

print("\nComparing local vs global costs at different scales...")

qubit_counts = [3, 4, 5, 6, 7]
local_vars = []
global_vars = []

for n_q in qubit_counts:
    print(f"  n = {n_q}...")

    dev = qml.device('default.qubit', wires=n_q)
    n_params = 2 * n_q * 3  # 3 layers

    @qml.qnode(dev, interface='autograd')
    def circuit_local(params):
        param_idx = 0
        for layer in range(3):
            for q in range(n_q):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            for q in range(n_q - 1):
                qml.CNOT(wires=[q, q+1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev, interface='autograd')
    def circuit_global(params):
        param_idx = 0
        for layer in range(3):
            for q in range(n_q):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            for q in range(n_q - 1):
                qml.CNOT(wires=[q, q+1])

        obs = qml.PauliZ(0)
        for q in range(1, n_q):
            obs = obs @ qml.PauliZ(q)
        return qml.expval(obs)

    grad_local = qml.grad(circuit_local)
    grad_global = qml.grad(circuit_global)

    grads_l = []
    grads_g = []

    for _ in range(50):
        params = pnp.random.uniform(0, 2*np.pi, n_params, requires_grad=True)
        grads_l.append(grad_local(params)[0])
        grads_g.append(grad_global(params)[0])

    local_vars.append(np.var(grads_l))
    global_vars.append(np.var(grads_g))

# Plot scaling
plt.figure(figsize=(10, 6))
plt.semilogy(qubit_counts, local_vars, 'bo-', markersize=10, linewidth=2,
             label='Local cost')
plt.semilogy(qubit_counts, global_vars, 'r^-', markersize=10, linewidth=2,
             label='Global cost')

# Theoretical lines
n_arr = np.array(qubit_counts)
plt.semilogy(n_arr, 0.1 * 2**(-n_arr), 'g--', alpha=0.5, label=r'$O(2^{-n})$')
plt.semilogy(n_arr, 0.1 * n_arr**(-2), 'm--', alpha=0.5, label=r'$O(n^{-2})$')

plt.xlabel('Number of Qubits', fontsize=12)
plt.ylabel('Gradient Variance', fontsize=12)
plt.title('Barren Plateau: Local vs Global Cost', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('local_global_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab complete!")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Barren Plateau | $\text{Var}[\partial_\theta \mathcal{L}] \leq e^{-cn}$ |
| 2-design variance | $\text{Var}[\langle O\rangle] = O(1/2^{2n})$ |
| Noise suppression | $\text{Var} \propto (1-p)^{2L}$ |
| Local cost scaling | $\text{Var}[\partial_\theta \mathcal{L}_{\text{local}}] = O(1/\text{poly}(n))$ |
| Critical depth | $L^* \sim O(n)$ for BP onset |

### Key Takeaways

1. **Barren plateaus are fundamental** - highly expressive circuits have exponentially vanishing gradients.

2. **Two sources of BPs:** Circuit-induced (depth/expressibility) and cost-function-induced (global observables).

3. **Local costs are trainable** - acting on O(1) qubits avoids exponential gradient suppression.

4. **Mitigation strategies exist:**
   - Identity initialization
   - Layerwise training
   - Structured/shallow ansatzes
   - Parameter correlations

5. **Expressibility-trainability trade-off** is fundamental - more expressive circuits are harder to train.

6. **Noise worsens BPs** - each layer of noise further suppresses gradients.

## Daily Checklist

- [ ] I understand why barren plateaus occur in variational circuits
- [ ] I can derive the variance scaling for random circuits
- [ ] I distinguish between circuit-induced and cost-function-induced BPs
- [ ] I can apply mitigation strategies: initialization, layerwise training
- [ ] I understand the expressibility vs trainability trade-off
- [ ] I completed the labs analyzing gradient variance scaling

## Preview of Day 950

Tomorrow we explore **Noise-Aware Compilation** - techniques to optimize quantum circuits specifically for noisy hardware:
- Error-adaptive qubit mapping considering T1, T2, and gate fidelities
- Noise-aware routing that minimizes error accumulation
- Pulse-level optimization for improved gate fidelity
- Dynamical decoupling for coherence extension
- Error-mitigated circuit design

Noise-aware compilation bridges the gap between abstract algorithms and real NISQ hardware.
