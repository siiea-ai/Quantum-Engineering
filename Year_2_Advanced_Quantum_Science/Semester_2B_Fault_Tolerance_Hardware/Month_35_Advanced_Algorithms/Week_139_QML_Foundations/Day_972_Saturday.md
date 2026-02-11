# Day 972: Expressibility & Trainability

## Year 2, Semester 2B: Fault Tolerance & Hardware
## Month 35: Advanced Algorithms - Week 139: QML Foundations

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Theory of expressibility and barren plateaus |
| **Afternoon** | 2 hours | Problem solving: analyzing circuit landscapes |
| **Evening** | 2 hours | Detecting and mitigating barren plateaus |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Quantify expressibility** of parameterized quantum circuits
2. **Explain the barren plateau phenomenon** and its mathematical origin
3. **Identify conditions** that lead to barren plateaus
4. **Apply mitigation strategies** for vanishing gradients
5. **Design circuits** balancing expressibility and trainability
6. **Analyze gradient landscapes** numerically and theoretically

---

## Morning Session: Theory (3 hours)

### 1. Expressibility: Covering Hilbert Space

#### Motivation

A parameterized quantum circuit $U(\boldsymbol{\theta})$ defines a mapping:

$$\boldsymbol{\theta} \in \mathbb{R}^p \mapsto U(\boldsymbol{\theta}) \in SU(2^n)$$

**Expressibility** measures how well the circuit can approximate arbitrary unitaries or states.

#### Formal Definition

**Definition (Sim et al., 2019):** The expressibility $\mathcal{E}$ of an ansatz is measured by comparing the distribution of output states to the Haar (uniform) distribution over $SU(2^n)$:

$$\boxed{\mathcal{E} = D_{KL}\left(\hat{P}_{U(\boldsymbol{\theta})} \| P_{\text{Haar}}\right)}$$

Where:
- $\hat{P}_{U(\boldsymbol{\theta})}$ = distribution of states from the parameterized circuit
- $P_{\text{Haar}}$ = uniform distribution over pure states (Haar measure)

**Lower $\mathcal{E}$ = More expressive** (closer to uniform coverage)

#### Computing Expressibility

Practical estimation uses the **fidelity distribution**:

$$F(\boldsymbol{\theta}, \boldsymbol{\theta}') = |\langle 0|U^\dagger(\boldsymbol{\theta})U(\boldsymbol{\theta}')|0\rangle|^2$$

Sample many pairs $(\boldsymbol{\theta}, \boldsymbol{\theta}')$ and compare:

$$\mathcal{E} \approx \int dF \left[\log\frac{\hat{p}(F)}{p_{\text{Haar}}(F)}\right] \hat{p}(F)$$

For the Haar distribution on $n$ qubits:
$$p_{\text{Haar}}(F) = (2^n - 1)(1 - F)^{2^n - 2}$$

### 2. Entangling Capability

#### Definition

The **entangling capability** measures how much entanglement a circuit can create:

$$\mathcal{Q} = \frac{1}{|\mathcal{S}|}\sum_{\boldsymbol{\theta} \in \mathcal{S}} E(|\psi(\boldsymbol{\theta})\rangle)$$

Where $E$ is an entanglement measure (e.g., Meyer-Wallach measure).

#### Meyer-Wallach Measure

For a pure state $|\psi\rangle$ on $n$ qubits:

$$Q(|\psi\rangle) = \frac{4}{n}\sum_{k=1}^n \left(1 - \text{Tr}[\rho_k^2]\right)$$

Where $\rho_k$ is the reduced density matrix of qubit $k$.

**Range:** $Q \in [0, 1]$, with 1 = maximally entangled.

### 3. The Barren Plateau Phenomenon

#### Discovery (McClean et al., 2018)

**Theorem (Barren Plateau):** For sufficiently deep random parameterized circuits, the gradient of the cost function vanishes exponentially:

$$\boxed{\text{Var}\left[\frac{\partial L}{\partial \theta_j}\right] \leq O\left(\frac{1}{2^n}\right)}$$

Where $n$ is the number of qubits.

#### Mathematical Origin

Consider a cost function:
$$L(\boldsymbol{\theta}) = \langle 0|U^\dagger(\boldsymbol{\theta}) O \, U(\boldsymbol{\theta})|0\rangle$$

The gradient is:
$$\frac{\partial L}{\partial \theta_j} = \frac{i}{2}\langle 0|U^\dagger(\boldsymbol{\theta}) [P_j, O] \, U(\boldsymbol{\theta})|0\rangle$$

For Haar-random unitaries:
$$\mathbb{E}[L] = \frac{\text{Tr}[O]}{2^n}, \quad \text{Var}[L] \propto \frac{1}{2^n}$$

**Consequence:** Gradients become exponentially small, making training impossible with polynomial resources.

#### Intuition: Why Barren Plateaus Occur

1. **High expressibility** → states spread uniformly over Hilbert space
2. **Uniform distribution** → cost function becomes flat (constant $\text{Tr}[O]/2^n$)
3. **Flat landscape** → gradients vanish

The more expressive the circuit, the more likely it exhibits barren plateaus!

### 4. Types of Barren Plateaus

#### 4.1 Depth-Induced Barren Plateaus

Occur when circuit depth exceeds a critical value:

$$L_{\text{crit}} \approx O(\log n)$$

Beyond this depth, the circuit approaches Haar-random behavior.

#### 4.2 Cost-Function-Induced Barren Plateaus

Global cost functions (measuring all qubits) lead to barren plateaus:

**Global cost:**
$$L_{\text{global}} = \text{Tr}[O \cdot \rho]$$

where $O$ acts on all qubits → **Barren plateau**

**Local cost:**
$$L_{\text{local}} = \frac{1}{n}\sum_{k=1}^n \text{Tr}[O_k \cdot \rho_k]$$

where $O_k$ acts on qubit $k$ → **May avoid barren plateau**

#### 4.3 Noise-Induced Barren Plateaus

Even shallow circuits can exhibit barren plateaus with noise:

$$\text{Var}[\partial_\theta L] \sim e^{-\gamma L}$$

Where $\gamma$ is the noise rate and $L$ is the depth.

#### 4.4 Entanglement-Induced Barren Plateaus

Highly entangling circuits are more prone to barren plateaus:

$$\text{Var}[\nabla L] \propto \frac{1}{2^{S}}$$

Where $S$ is the entanglement entropy across a cut.

### 5. Detecting Barren Plateaus

#### Variance of Gradients

Sample gradients at many random initializations:

$$\text{Var}[\partial_{\theta_j} L] = \mathbb{E}[(\partial_{\theta_j} L)^2] - \mathbb{E}[\partial_{\theta_j} L]^2$$

**Barren plateau criterion:** If $\text{Var} \sim O(2^{-n})$, barren plateau exists.

#### Practical Detection Algorithm

```python
def detect_barren_plateau(circuit, cost_fn, n_samples=100):
    grads = []
    for _ in range(n_samples):
        theta = np.random.uniform(-pi, pi, n_params)
        grad = compute_gradient(circuit, cost_fn, theta)
        grads.append(grad)

    variance = np.var(grads, axis=0)
    mean_variance = np.mean(variance)

    if mean_variance < 1e-6:  # Threshold depends on n
        return "Barren plateau detected"
    return "No barren plateau"
```

### 6. Mitigation Strategies

#### 6.1 Parameter Initialization

**Identity initialization:** Start near the identity:
$$U(\boldsymbol{\theta}_0) \approx I$$

Achieved by setting $\theta_j \approx 0$ for most rotations.

**Advantage:** Initial state $|0\rangle$ is preserved, gradients non-zero.

#### 6.2 Layer-wise Training

Train layer by layer:

1. Train layer 1, freeze
2. Add layer 2, train, freeze
3. Continue...

**Advantage:** Avoids deep-circuit barren plateaus during training.

#### 6.3 Local Cost Functions

Use cost functions that measure only a few qubits:

$$L_{\text{local}} = \sum_{k} c_k \langle O_k \rangle$$

Where each $O_k$ acts on $O(1)$ qubits.

**Theorem (Cerezo et al., 2021):** Local cost functions have gradient variance:
$$\text{Var}[\partial L_{\text{local}}] \geq \Omega\left(\frac{1}{\text{poly}(n)}\right)$$

Much better than exponential decay!

#### 6.4 Shallow Circuits

Keep depth below critical threshold:
$$L < O(\log n)$$

**Trade-off:** Reduces expressibility but maintains trainability.

#### 6.5 Problem-Inspired Ansätze

Design circuits based on problem structure:

**Example (QAOA):** Ansatz respects problem Hamiltonian structure.

**Example (VQE):** Use chemistry-inspired circuits (UCCSD).

These typically avoid barren plateaus for their target problems.

#### 6.6 Correlated Parameters

Reduce effective parameter count:
$$\theta_j = f_j(\boldsymbol{\phi})$$

Where $\boldsymbol{\phi}$ has fewer elements than $\boldsymbol{\theta}$.

### 7. The Expressibility-Trainability Trade-off

#### The Fundamental Dilemma

$$\text{High Expressibility} \leftrightarrow \text{Poor Trainability}$$

**Highly expressive circuits:**
- Can represent any function
- Suffer from barren plateaus
- Exponentially flat landscapes

**Less expressive circuits:**
- Limited function class
- Better gradient landscape
- Polynomial training cost

#### Optimal Design Principles

1. **Match ansatz to problem:** Don't use universal circuits for structured problems
2. **Start shallow, grow if needed:** Begin with few layers
3. **Use local costs when possible:** Avoid global observables
4. **Initialize carefully:** Near identity or classically meaningful points
5. **Monitor gradient variance:** Detect barren plateaus early

---

## Afternoon Session: Problem Solving (2 hours)

### Worked Example 1: Computing Expressibility

**Problem:** Estimate the expressibility of a single-qubit circuit $U(\theta_1, \theta_2) = R_Y(\theta_2)R_Z(\theta_1)$.

**Solution:**

**Step 1: Sample random parameters**

Sample $N = 1000$ pairs $(\boldsymbol{\theta}, \boldsymbol{\theta}')$ uniformly from $[0, 2\pi)^2$.

**Step 2: Compute fidelities**

$$F = |\langle 0|U^\dagger(\theta_1, \theta_2)U(\theta_1', \theta_2')|0\rangle|^2$$

For this circuit:
$$U(\boldsymbol{\theta}) = R_Y(\theta_2)R_Z(\theta_1)$$
$$U^\dagger(\boldsymbol{\theta})U(\boldsymbol{\theta}') = R_Z(-\theta_1)R_Y(-\theta_2)R_Y(\theta_2')R_Z(\theta_1')$$

**Step 3: Compare to Haar distribution**

For a single qubit, the Haar distribution of fidelities is uniform on $[0, 1]$.

If the empirical distribution closely matches uniform, the circuit is maximally expressive for a single qubit.

**Result:** This circuit (with 2 parameters) can reach any single-qubit state up to global phase → maximally expressive for 1 qubit.

---

### Worked Example 2: Gradient Variance Scaling

**Problem:** For a 4-qubit random circuit with depth $L$, estimate when barren plateaus onset.

**Solution:**

**Step 1: Set up the problem**

Random circuit: alternating layers of random rotations and CNOTs.

Cost function: $L = \langle Z_1 Z_2 Z_3 Z_4 \rangle$ (global)

**Step 2: Compute gradient variance numerically**

For each depth $L \in \{1, 2, 4, 8, 16\}$:

```
L=1:  Var[∂L/∂θ] ≈ 0.05
L=2:  Var[∂L/∂θ] ≈ 0.02
L=4:  Var[∂L/∂θ] ≈ 0.005
L=8:  Var[∂L/∂θ] ≈ 0.001
L=16: Var[∂L/∂θ] ≈ 0.0002
```

**Step 3: Analyze scaling**

The variance decreases roughly exponentially with depth.

$\text{Var} \approx c \cdot 2^{-\alpha L}$ where $\alpha \approx 0.5$.

**Step 4: Determine critical depth**

For 4 qubits, full Haar randomness would give $\text{Var} \sim 2^{-4} = 0.0625$.

We see this level by $L \approx 4-8$.

**Conclusion:** Barren plateau onset at $L \approx O(n) = O(4) = 4$ layers for this circuit.

---

### Worked Example 3: Local vs. Global Cost Functions

**Problem:** Compare gradient variance for global vs. local cost functions on a 6-qubit circuit.

**Solution:**

**Global cost:** $L_G = \langle Z^{\otimes 6} \rangle = \langle Z_1 Z_2 Z_3 Z_4 Z_5 Z_6 \rangle$

**Local cost:** $L_L = \frac{1}{6}\sum_{k=1}^6 \langle Z_k \rangle$

**Theoretical prediction:**

- Global: $\text{Var}[\nabla L_G] \sim O(2^{-n}) = O(2^{-6}) \approx 0.016$
- Local: $\text{Var}[\nabla L_L] \sim O(n^{-1}) = O(1/6) \approx 0.17$

**Numerical verification (depth 4 random circuit):**

```
Global cost gradient variance: 0.008
Local cost gradient variance:  0.12
```

**Conclusion:** Local cost function has ~15× larger gradient variance, making training much easier.

---

### Practice Problems

#### Problem 1: Expressibility Calculation (Direct Application)

Consider the circuit $U(\theta) = R_Y(\theta)$ on a single qubit.

a) What is the dimension of the manifold of states accessible?
b) Is this circuit maximally expressive for 1 qubit?
c) What additional gate would make it maximally expressive?

<details>
<summary>Solution</summary>

a) The accessible states are $\cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$, which form a great circle on the Bloch sphere. Dimension = 1.

b) No, a single $R_Y$ only covers a 1D subspace of the 2D Bloch sphere surface.

c) Adding $R_Z(\phi)$ before $R_Y(\theta)$: $U = R_Y(\theta)R_Z(\phi)$ can reach any point on the Bloch sphere. Or equivalently, $R_X(\phi)R_Y(\theta)$ or similar combinations.
</details>

#### Problem 2: Barren Plateau Onset (Intermediate)

A 10-qubit circuit has the form: $[R_Y^{\otimes 10} \cdot \text{CNOT cascade}]^L$

a) Estimate the critical depth for barren plateau onset
b) If you need to train this circuit, what maximum depth would you use?
c) Suggest a mitigation strategy if you need more expressibility

<details>
<summary>Solution</summary>

a) For random circuits on $n$ qubits, the critical depth is approximately $O(\log n) \approx \log_2(10) \approx 3-4$ layers.

b) To avoid barren plateaus, use $L \leq 3$ layers. This keeps gradient variance polynomially decaying rather than exponential.

c) Mitigation strategies:
- Use layer-wise training: add one layer at a time
- Use local cost functions instead of global
- Initialize near identity: $\theta \approx 0$
- Use a problem-specific ansatz instead of random structure
</details>

#### Problem 3: Cost Function Design (Challenging)

You want to train a VQE for a 4-qubit molecular Hamiltonian:
$$H = \sum_{ij} h_{ij} Z_i Z_j + \sum_i g_i Z_i + \text{const}$$

a) Is $\langle H \rangle$ a global or local cost function?
b) Will this naturally avoid barren plateaus? Why or why not?
c) Design an alternative training strategy using local costs

<details>
<summary>Solution</summary>

a) The Hamiltonian has both 2-local terms ($Z_i Z_j$) and 1-local terms ($Z_i$), so it's a sum of local terms. However, the full cost $\langle H \rangle$ involves correlations across many qubits.

b) Partial protection: The 1-local terms help avoid complete barren plateaus. The 2-local terms may still contribute to gradient decay if the circuit becomes deep enough. VQE with chemistry-inspired ansätze often works because:
- The ansatz is structured (not random)
- Initial states (Hartree-Fock) are good starting points
- The Hamiltonian has local structure

c) Alternative strategy:
1. Train on $\langle H_{\text{local}} \rangle = \sum_i g_i \langle Z_i \rangle$ first
2. Gradually add 2-local terms: $\langle H_{\text{local}} \rangle + \lambda \sum_{ij} h_{ij} \langle Z_i Z_j \rangle$
3. Increase $\lambda$ during training (curriculum learning)
4. Or use layer-wise training with the full Hamiltonian
</details>

---

## Evening Session: Computational Lab (2 hours)

### Lab: Analyzing and Mitigating Barren Plateaus

```python
"""
Day 972 Lab: Expressibility and Trainability
Analyzing barren plateaus and mitigation strategies
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 60)
print("Expressibility & Trainability Analysis")
print("=" * 60)

#######################################
# Part 1: Measuring Expressibility
#######################################

print("\n" + "-" * 40)
print("Part 1: Measuring Expressibility")
print("-" * 40)

def compute_expressibility(circuit_func, n_qubits, n_params, n_samples=500):
    """
    Compute expressibility by comparing fidelity distribution to Haar
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def fidelity_circuit(params1, params2):
        circuit_func(params1)
        qml.adjoint(circuit_func)(params2)
        return qml.probs(wires=range(n_qubits))

    # Sample random fidelities
    fidelities = []
    for _ in range(n_samples):
        params1 = np.random.uniform(-np.pi, np.pi, n_params)
        params2 = np.random.uniform(-np.pi, np.pi, n_params)
        probs = fidelity_circuit(params1, params2)
        fidelities.append(probs[0])  # P(|0...0⟩) = |⟨ψ₁|ψ₂⟩|²

    fidelities = np.array(fidelities)

    # Compare to Haar distribution
    # For Haar: P(F) = (2^n - 1)(1-F)^(2^n - 2)
    # We use KL divergence approximation
    dim = 2 ** n_qubits

    # Bin the fidelities
    bins = np.linspace(0, 1, 50)
    hist, _ = np.histogram(fidelities, bins=bins, density=True)

    # Haar theoretical
    bin_centers = (bins[:-1] + bins[1:]) / 2
    haar_pdf = (dim - 1) * (1 - bin_centers) ** (dim - 2)
    haar_pdf /= np.sum(haar_pdf) * (bins[1] - bins[0])

    # KL divergence (approximation)
    epsilon = 1e-10
    kl_div = np.sum(hist * np.log((hist + epsilon) / (haar_pdf + epsilon))) * (bins[1] - bins[0])

    return abs(kl_div), fidelities


# Define circuits with varying expressibility
n_qubits = 2

def low_expressibility_circuit(params):
    """Only single rotations, no entanglement"""
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)

def medium_expressibility_circuit(params):
    """Rotations with single CNOT"""
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)

def high_expressibility_circuit(params):
    """Deep circuit with multiple layers"""
    n_layers = 3
    idx = 0
    for l in range(n_layers):
        qml.RX(params[idx], wires=0)
        qml.RY(params[idx+1], wires=0)
        qml.RZ(params[idx+2], wires=0)
        qml.RX(params[idx+3], wires=1)
        qml.RY(params[idx+4], wires=1)
        qml.RZ(params[idx+5], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 0])
        idx += 6

circuits = {
    'Low': (low_expressibility_circuit, 2),
    'Medium': (medium_expressibility_circuit, 4),
    'High': (high_expressibility_circuit, 18)
}

print("\nExpressibility Comparison:")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, (circuit, n_params)) in enumerate(circuits.items()):
    expr, fids = compute_expressibility(circuit, n_qubits, n_params)
    print(f"{name:8s} | KL Divergence: {expr:.4f} | Mean Fidelity: {np.mean(fids):.3f}")

    # Plot histogram
    axes[idx].hist(fids, bins=30, density=True, alpha=0.7, label='Circuit')

    # Haar distribution
    x = np.linspace(0.001, 0.999, 100)
    dim = 2 ** n_qubits
    haar_pdf = (dim - 1) * (1 - x) ** (dim - 2)
    axes[idx].plot(x, haar_pdf, 'r-', linewidth=2, label='Haar')

    axes[idx].set_xlabel('Fidelity')
    axes[idx].set_ylabel('Density')
    axes[idx].set_title(f'{name} Expressibility')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('expressibility_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 2: Gradient Variance vs Depth
#######################################

print("\n" + "-" * 40)
print("Part 2: Gradient Variance vs Depth")
print("-" * 40)

def measure_gradient_variance(n_qubits, depth, n_samples=100, cost_type='global'):
    """
    Measure gradient variance for random circuits at different depths
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    n_params_per_layer = n_qubits * 3
    n_params = depth * n_params_per_layer

    @qml.qnode(dev)
    def circuit(params):
        # Random circuit structure
        for l in range(depth):
            for q in range(n_qubits):
                idx = l * n_params_per_layer + q * 3
                qml.RX(params[idx], wires=q)
                qml.RY(params[idx + 1], wires=q)
                qml.RZ(params[idx + 2], wires=q)
            # Entangling layer
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

        # Cost function
        if cost_type == 'global':
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))
        else:  # local
            return qml.expval(qml.PauliZ(0))

    # Sample gradients
    gradients = []
    for _ in range(n_samples):
        params = np.random.uniform(-np.pi, np.pi, n_params)
        grad = qml.grad(circuit)(params)
        gradients.append(grad[0])  # First parameter gradient

    return np.var(gradients)

# Measure for different depths
n_qubits = 4
depths = [1, 2, 3, 4, 5, 6, 8, 10]

variances_global = []
variances_local = []

print("\nComputing gradient variances (this may take a moment)...")
for d in depths:
    var_g = measure_gradient_variance(n_qubits, d, cost_type='global')
    var_l = measure_gradient_variance(n_qubits, d, cost_type='local')
    variances_global.append(var_g)
    variances_local.append(var_l)
    print(f"Depth {d:2d} | Global: {var_g:.2e} | Local: {var_l:.2e}")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(depths, variances_global, 'bo-', linewidth=2, markersize=8, label='Global Cost')
plt.semilogy(depths, variances_local, 'rs-', linewidth=2, markersize=8, label='Local Cost')

# Theoretical scaling
theoretical_global = [variances_global[0] * 2**(-0.5*(d-1)) for d in depths]
plt.semilogy(depths, theoretical_global, 'b--', alpha=0.5, label='Exponential fit (global)')

plt.xlabel('Circuit Depth', fontsize=12)
plt.ylabel('Gradient Variance', fontsize=12)
plt.title('Barren Plateau: Gradient Variance vs Depth', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('barren_plateau_depth.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 3: Qubit Scaling
#######################################

print("\n" + "-" * 40)
print("Part 3: Gradient Variance vs Qubits")
print("-" * 40)

def measure_variance_vs_qubits(n_qubits_list, depth=3, n_samples=100):
    """Measure gradient variance scaling with number of qubits"""
    variances = []

    for n in n_qubits_list:
        dev = qml.device('default.qubit', wires=n)
        n_params = depth * n * 3

        @qml.qnode(dev)
        def circuit(params):
            for l in range(depth):
                for q in range(n):
                    idx = l * n * 3 + q * 3
                    qml.RX(params[idx], wires=q)
                    qml.RY(params[idx + 1], wires=q)
                    qml.RZ(params[idx + 2], wires=q)
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])

            # Global cost
            obs = qml.PauliZ(0)
            for q in range(1, n):
                obs = obs @ qml.PauliZ(q)
            return qml.expval(obs)

        grads = []
        for _ in range(n_samples):
            params = np.random.uniform(-np.pi, np.pi, n_params)
            grad = qml.grad(circuit)(params)
            grads.append(grad[0])

        variances.append(np.var(grads))
        print(f"Qubits: {n} | Variance: {variances[-1]:.2e}")

    return variances

n_qubits_list = [2, 3, 4, 5, 6]
print("\nMeasuring gradient variance vs qubit count...")
variances_qubits = measure_variance_vs_qubits(n_qubits_list)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.semilogy(n_qubits_list, variances_qubits, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Qubits')
plt.ylabel('Gradient Variance')
plt.title('Gradient Variance vs Qubits (log scale)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
log_var = np.log2(variances_qubits)
plt.plot(n_qubits_list, log_var, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Qubits (n)')
plt.ylabel('log₂(Variance)')
plt.title('Exponential Decay Verification')
# Fit line
coeffs = np.polyfit(n_qubits_list, log_var, 1)
fit_line = np.poly1d(coeffs)
plt.plot(n_qubits_list, fit_line(n_qubits_list), 'r--',
         label=f'Fit: slope = {coeffs[0]:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('barren_plateau_qubits.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nExponential fit: Var ∝ 2^({coeffs[0]:.2f} × n)")
print(f"Theory predicts: Var ∝ 2^(-n) for deep circuits")


#######################################
# Part 4: Mitigation Strategies
#######################################

print("\n" + "-" * 40)
print("Part 4: Mitigation Strategies")
print("-" * 40)

n_qubits = 4
depth = 4
dev = qml.device('default.qubit', wires=n_qubits)

# Strategy 1: Identity initialization
print("\n1. Identity Initialization:")

@qml.qnode(dev)
def circuit_general(params):
    for l in range(depth):
        for q in range(n_qubits):
            idx = l * n_qubits * 3 + q * 3
            qml.RX(params[idx], wires=q)
            qml.RY(params[idx + 1], wires=q)
            qml.RZ(params[idx + 2], wires=q)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
    return qml.expval(qml.PauliZ(0))

n_params = depth * n_qubits * 3

# Random initialization
random_grads = []
for _ in range(100):
    params = np.random.uniform(-np.pi, np.pi, n_params)
    grad = qml.grad(circuit_general)(params)
    random_grads.append(np.mean(np.abs(grad)))

# Identity initialization (small parameters)
identity_grads = []
for _ in range(100):
    params = np.random.uniform(-0.1, 0.1, n_params)  # Near zero
    grad = qml.grad(circuit_general)(params)
    identity_grads.append(np.mean(np.abs(grad)))

print(f"Random init - Mean |grad|: {np.mean(random_grads):.4f}")
print(f"Identity init - Mean |grad|: {np.mean(identity_grads):.4f}")
print(f"Improvement: {np.mean(identity_grads) / np.mean(random_grads):.1f}x")


# Strategy 2: Local cost function
print("\n2. Local vs Global Cost Function:")

@qml.qnode(dev)
def circuit_local_cost(params):
    for l in range(depth):
        for q in range(n_qubits):
            idx = l * n_qubits * 3 + q * 3
            qml.RX(params[idx], wires=q)
            qml.RY(params[idx + 1], wires=q)
            qml.RZ(params[idx + 2], wires=q)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
    # Local cost: average of single-qubit expectations
    return (qml.expval(qml.PauliZ(0)) + qml.expval(qml.PauliZ(1)) +
            qml.expval(qml.PauliZ(2)) + qml.expval(qml.PauliZ(3))) / 4

local_grads = []
for _ in range(100):
    params = np.random.uniform(-np.pi, np.pi, n_params)
    grad = qml.grad(circuit_local_cost)(params)
    local_grads.append(np.mean(np.abs(grad)))

print(f"Global cost - Mean |grad|: {np.mean(random_grads):.4f}")
print(f"Local cost - Mean |grad|: {np.mean(local_grads):.4f}")
print(f"Improvement: {np.mean(local_grads) / np.mean(random_grads):.1f}x")


# Strategy 3: Layer-wise training simulation
print("\n3. Layer-wise Training Effect:")

def simulate_layerwise_training():
    """Simulate gradient at different training stages"""
    results = []

    for active_layers in range(1, depth + 1):
        @qml.qnode(dev)
        def circuit_layerwise(params):
            for l in range(depth):
                for q in range(n_qubits):
                    idx = l * n_qubits * 3 + q * 3
                    if l < active_layers:
                        # Active layer: use params
                        qml.RX(params[idx], wires=q)
                        qml.RY(params[idx + 1], wires=q)
                        qml.RZ(params[idx + 2], wires=q)
                    else:
                        # Frozen layer: identity (params ≈ 0)
                        qml.RX(0, wires=q)
                        qml.RY(0, wires=q)
                        qml.RZ(0, wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))

        grads = []
        for _ in range(50):
            params = np.random.uniform(-np.pi, np.pi, n_params)
            grad = qml.grad(circuit_layerwise)(params)
            grads.append(np.mean(np.abs(grad[:active_layers * n_qubits * 3])))

        results.append(np.mean(grads))
        print(f"Active layers: {active_layers} | Mean |grad|: {results[-1]:.4f}")

    return results

layerwise_results = simulate_layerwise_training()


#######################################
# Part 5: Visualization Summary
#######################################

print("\n" + "-" * 40)
print("Part 5: Summary Visualization")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Depth effect
axes[0, 0].semilogy(depths, variances_global, 'bo-', linewidth=2, label='Global')
axes[0, 0].semilogy(depths, variances_local, 'rs-', linewidth=2, label='Local')
axes[0, 0].set_xlabel('Circuit Depth')
axes[0, 0].set_ylabel('Gradient Variance')
axes[0, 0].set_title('Barren Plateau: Depth Effect')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Qubit effect
axes[0, 1].semilogy(n_qubits_list, variances_qubits, 'go-', linewidth=2)
axes[0, 1].set_xlabel('Number of Qubits')
axes[0, 1].set_ylabel('Gradient Variance')
axes[0, 1].set_title('Barren Plateau: Qubit Scaling')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Initialization comparison
init_types = ['Random', 'Identity']
init_values = [np.mean(random_grads), np.mean(identity_grads)]
axes[1, 0].bar(init_types, init_values, color=['coral', 'steelblue'], edgecolor='black')
axes[1, 0].set_ylabel('Mean |Gradient|')
axes[1, 0].set_title('Effect of Initialization')

# Plot 4: Layer-wise training
axes[1, 1].plot(range(1, depth + 1), layerwise_results, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Active Layers')
axes[1, 1].set_ylabel('Mean |Gradient|')
axes[1, 1].set_title('Layer-wise Training Effect')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('barren_plateau_summary.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 6: Recommendations
#######################################

print("\n" + "-" * 40)
print("Part 6: Practical Recommendations")
print("-" * 40)

recommendations = """
BARREN PLATEAU MITIGATION CHECKLIST
====================================

1. INITIALIZATION
   [✓] Use identity initialization (params ≈ 0)
   [✓] Or use problem-specific starting point
   [✓] Avoid uniform random in [-π, π]

2. CIRCUIT DEPTH
   [✓] Keep depth ≤ O(log n) for random circuits
   [✓] Use layer-wise training for deeper circuits
   [✓] Monitor gradient variance during training

3. COST FUNCTION
   [✓] Prefer local cost functions when possible
   [✓] Use sum of local terms instead of global observables
   [✓] Consider curriculum learning (local → global)

4. ARCHITECTURE
   [✓] Use problem-inspired ansätze (not random)
   [✓] Match entanglement structure to problem
   [✓] Limit entanglement for very deep circuits

5. MONITORING
   [✓] Track gradient variance during training
   [✓] If variance drops exponentially, reduce depth
   [✓] Compare train vs. test performance for overfitting
"""

print(recommendations)

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| **Expressibility** | $\mathcal{E} = D_{KL}(\hat{P}_U \| P_{\text{Haar}})$ |
| **Barren Plateau (depth)** | $\text{Var}[\nabla L] \sim O(2^{-n})$ for deep circuits |
| **Barren Plateau (qubits)** | $\text{Var}[\nabla L] \sim O(2^{-n})$ for $n$ qubits |
| **Local vs Global** | Local: $O(n^{-1})$, Global: $O(2^{-n})$ |
| **Critical Depth** | $L_{\text{crit}} \approx O(\log n)$ |

### The Expressibility-Trainability Trade-off

```
Expressibility:  Low ←―――――――――――――――→ High
                  │                     │
Trainability:   High ←―――――――――――――――→ Low
                  │                     │
Barren Plateau:  No  ←―――――――――――――――→ Yes
```

### Key Takeaways

1. **Expressibility** measures how well circuits cover state space
2. **Barren plateaus** cause exponentially vanishing gradients
3. **Deep + random circuits** are most susceptible
4. **Local cost functions** provide polynomial (not exponential) gradient decay
5. **Identity initialization** keeps gradients large initially
6. **Layer-wise training** avoids deep-circuit barren plateaus

### Mitigation Strategies Summary

| Strategy | Effect | Implementation |
|----------|--------|----------------|
| Identity init | 10-100x larger gradients | $\theta \approx 0$ |
| Local costs | Polynomial decay | $L = \sum_k \langle O_k \rangle$ |
| Shallow circuits | Avoid onset | $L < O(\log n)$ |
| Layer-wise training | Gradual depth increase | Train 1 layer at a time |
| Problem-inspired ansatz | Match structure | Use domain knowledge |

---

## Daily Checklist

- [ ] I can define and compute expressibility
- [ ] I understand why barren plateaus occur
- [ ] I can identify conditions leading to barren plateaus
- [ ] I know multiple mitigation strategies
- [ ] I can design circuits balancing expressibility and trainability
- [ ] I implemented barren plateau detection numerically

---

## Preview: Day 973

Tomorrow we conclude the week with **QML Advantages & Limitations**:

- When can quantum ML outperform classical?
- Dequantization and Tang's results
- Current limitations of QML
- Hype vs. reality assessment
- Future outlook for the field

---

*"The barren plateau is not a bug, it's a feature of expressibility - a reminder that quantum advantage must be earned, not assumed."*
— Patrick Coles
