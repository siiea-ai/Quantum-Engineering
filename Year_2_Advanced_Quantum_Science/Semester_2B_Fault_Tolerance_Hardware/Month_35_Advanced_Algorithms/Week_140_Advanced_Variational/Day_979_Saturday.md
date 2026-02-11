# Day 979: Error-Mitigated Variational Algorithms

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Error Mitigation in VQE |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 979, you will be able to:

1. Apply zero-noise extrapolation (ZNE) to VQE calculations
2. Implement probabilistic error cancellation (PEC) for variational circuits
3. Use Clifford data regression (CDR) for error mitigation
4. Combine symmetry verification with VQE
5. Understand cost-accuracy trade-offs in error-mitigated VQE
6. Design hybrid workflows integrating error mitigation with optimization

---

## Core Content

### 1. The Error Mitigation Imperative

NISQ devices introduce errors that systematically bias VQE results:

**Sources of Error:**
- Gate errors (~0.1-1% per two-qubit gate)
- Decoherence (T1, T2 decay)
- Measurement errors (~1-5%)
- Crosstalk between qubits

**Impact on VQE:**
$$E_{\text{measured}} = E_{\text{ideal}} + \delta E_{\text{noise}}$$

For chemistry accuracy (1 kcal/mol ~ 1.6 mHa), we need $\delta E_{\text{noise}} \lesssim 1$ mHa.

**Error Mitigation Goal:** Estimate $E_{\text{ideal}}$ from noisy measurements.

---

### 2. Zero-Noise Extrapolation (ZNE)

**Core Idea:** Amplify noise systematically, then extrapolate to zero noise.

**Procedure:**

1. Measure observable at baseline noise level: $E(\lambda = 1)$
2. Artificially increase noise by factor $\lambda$: $E(\lambda > 1)$
3. Fit curve $E(\lambda)$ and extrapolate to $\lambda \to 0$

**Noise Amplification Methods:**

*Pulse stretching:* Increase gate duration
$$\tau_{\text{gate}} \to \lambda \cdot \tau_{\text{gate}}$$

*Unitary folding:* Replace $U$ with $U(U^\dagger U)^k$
$$U \to U(U^\dagger U)^k, \quad \lambda = 2k + 1$$

**Extrapolation Models:**

*Linear:*
$$E(\lambda) = E_0 + a\lambda$$
$$E_{\text{ideal}} = E(\lambda=0) = E_0$$

*Richardson (polynomial):*
$$E(\lambda) = E_0 + a_1\lambda + a_2\lambda^2 + \cdots$$

*Exponential:*
$$E(\lambda) = E_\infty + (E_0 - E_\infty)e^{-\lambda/\tau}$$

**Key Equations:**

$$\boxed{E_{\text{mitigated}} = \sum_k c_k E(\lambda_k), \quad \sum_k c_k = 1, \quad \sum_k c_k \lambda_k = 0}$$

For linear extrapolation with $\lambda_1 = 1, \lambda_2 = 3$:
$$E_{\text{mitigated}} = \frac{3E_1 - E_3}{2}$$

---

### 3. Probabilistic Error Cancellation (PEC)

**Core Idea:** Represent ideal gates as quasi-probability sums of noisy implementable operations.

**Mathematical Framework:**

If noise channel is $\mathcal{N}$, ideal gate $\mathcal{G}$ can be written:
$$\mathcal{G} = \sum_i \gamma_i \mathcal{O}_i$$

where $\mathcal{O}_i$ are implementable operations and $\gamma_i$ can be negative.

**Sampling Protocol:**

1. Sample operation $\mathcal{O}_i$ with probability $|\gamma_i|/\gamma_{\text{sum}}$
2. Record sign $s_i = \text{sign}(\gamma_i)$
3. Run circuit with sampled operations
4. Multiply result by $s_i \cdot \gamma_{\text{sum}}$
5. Average over many samples

**Cost Scaling:**

The variance increases as:
$$\text{Var}[E_{\text{PEC}}] \sim \gamma_{\text{sum}}^{2L}$$

where $L$ is the number of gates. This limits applicability to shallow circuits.

---

### 4. Clifford Data Regression (CDR)

**Core Idea:** Learn the error map from near-Clifford circuits that can be classically simulated.

**Procedure:**

1. Generate training circuits: Replace non-Clifford gates with Clifford gates
2. Compute ideal expectation values classically
3. Measure noisy expectation values on hardware
4. Fit a linear regression model: $E_{\text{ideal}} = a \cdot E_{\text{noisy}} + b$
5. Apply correction to target VQE measurement

**Regression Model:**
$$E_{\text{mitigated}} = \frac{E_{\text{noisy}} - b}{a}$$

**Advantages:**
- Learns device-specific error model
- No noise amplification needed
- Works with any circuit structure

**Limitations:**
- Assumes error model is consistent
- Requires generating similar training circuits
- May not extrapolate well to very non-Clifford regimes

---

### 5. Symmetry Verification

For systems with known symmetries, post-selection on symmetry-preserving results:

**Particle Number Verification:**

1. Measure $\langle N \rangle$ alongside energy
2. Discard shots where $N \neq N_{\text{target}}$
3. Average only symmetry-valid results

**Symmetry-Expanded VQE:**

$$E_{\text{mitigated}} = \frac{\langle P_S H P_S \rangle}{\langle P_S \rangle}$$

where $P_S$ is the projector onto the correct symmetry sector.

**Virtual Distillation:**

Prepare two copies of the state:
$$\text{Tr}(O \rho) \to \frac{\text{Tr}(O \rho^2)}{\text{Tr}(\rho^2)}$$

This suppresses errors quadratically.

---

### 6. Combining Mitigation with Optimization

**Challenge:** Error mitigation increases variance, affecting optimization.

**Strategy 1: Mitigate Only Final Result**

1. Optimize with noisy evaluations (cheap)
2. Apply ZNE/PEC only to final energy (expensive)

**Strategy 2: Mitigated Gradients**

1. Compute gradients using parameter shift
2. Apply ZNE to each gradient component
3. Use mitigated gradients for optimization

**Cost Analysis:**

| Method | Overhead per evaluation |
|--------|------------------------|
| No mitigation | 1x |
| ZNE (3-point) | 3x |
| PEC | $\gamma^{2L}$x (exponential in depth) |
| CDR | ~10x (training + inference) |

**Practical Workflow:**

```
1. Initial optimization (no mitigation, fast)
2. Refinement (light mitigation, ZNE linear)
3. Final result (heavy mitigation, ZNE polynomial or PEC)
```

---

### 7. Trade-offs and Limitations

**Bias vs Variance Trade-off:**

| Method | Reduces Bias | Increases Variance |
|--------|--------------|-------------------|
| ZNE | Yes | Yes (moderately) |
| PEC | Yes (in principle exact) | Yes (exponentially) |
| CDR | Partially | Moderately |
| Symmetry | Partially | Reduces (post-selection) |

**When to Use What:**

| Circuit Depth | Recommended Method |
|--------------|-------------------|
| Very shallow (< 10 gates) | PEC if accurate noise model |
| Shallow (10-50 gates) | ZNE with Richardson |
| Medium (50-200 gates) | ZNE linear + symmetry |
| Deep (> 200 gates) | Error mitigation insufficient |

---

### 8. Error Mitigation for Gradients

**Challenge:** Gradients are differences, and mitigation must be consistent.

**Mitigated Parameter Shift:**

$$\frac{\partial E}{\partial \theta} = \frac{1}{2}\left[E_{\text{mitigated}}(\theta + \pi/2) - E_{\text{mitigated}}(\theta - \pi/2)\right]$$

Apply the same mitigation protocol to both shifted circuits.

**Variance Propagation:**

$$\text{Var}[\nabla E_{\text{mitigated}}] = \frac{1}{4}\left[\text{Var}[E_+] + \text{Var}[E_-]\right]$$

ZNE increases variance by factor ~$\sum_k |c_k|^2$.

---

## Practical Applications

### Error-Mitigated VQE for H2

**Workflow:**

1. **Setup:** H2 Hamiltonian, 2-4 qubit representation
2. **Ansatz:** Hardware-efficient, 2-3 layers
3. **Optimization:** COBYLA with unmitigated cost
4. **Final Result:** Apply ZNE at converged parameters

**Expected Improvement:**

| Method | Error (mHa) | Shots Required |
|--------|-------------|----------------|
| No mitigation | 10-50 | 1000 |
| ZNE (linear) | 2-10 | 3000 |
| ZNE (Richardson) | 0.5-5 | 5000+ |

---

## Worked Examples

### Example 1: Linear ZNE Extrapolation

**Problem:** Given measurements $E_1 = -1.10$ at $\lambda = 1$ and $E_3 = -1.02$ at $\lambda = 3$, compute the ZNE estimate.

**Solution:**

For linear extrapolation to $\lambda = 0$:
$$E(\lambda) = E_0 + m\lambda$$

The slope:
$$m = \frac{E_3 - E_1}{3 - 1} = \frac{-1.02 - (-1.10)}{2} = \frac{0.08}{2} = 0.04$$

The intercept:
$$E_0 = E_1 - m \cdot 1 = -1.10 - 0.04 = -1.14$$

**Or using the formula directly:**
$$E_{\text{mitigated}} = \frac{3 \cdot E_1 - E_3}{3 - 1} = \frac{3(-1.10) - (-1.02)}{2} = \frac{-3.30 + 1.02}{2} = \frac{-2.28}{2} = -1.14$$

**Result:** $E_{\text{mitigated}} = -1.14$

If the true value is $E_{\text{exact}} = -1.136$, the mitigated result is much closer than either noisy measurement!

---

### Example 2: Richardson Extrapolation

**Problem:** With three noise levels $\lambda = 1, 2, 3$ giving $E_1 = -1.10$, $E_2 = -1.06$, $E_3 = -1.02$, find the quadratic extrapolation.

**Solution:**

Fit $E(\lambda) = a_0 + a_1\lambda + a_2\lambda^2$:

System of equations:
$$E_1 = a_0 + a_1 + a_2 = -1.10$$
$$E_2 = a_0 + 2a_1 + 4a_2 = -1.06$$
$$E_3 = a_0 + 3a_1 + 9a_2 = -1.02$$

Solving:
- $E_2 - E_1 = a_1 + 3a_2 = 0.04$
- $E_3 - E_2 = a_1 + 5a_2 = 0.04$
- Subtracting: $2a_2 = 0 \Rightarrow a_2 = 0$

So $a_1 = 0.04$ and $a_0 = -1.10 - 0.04 = -1.14$.

**Result:** Same as linear (the quadratic term is zero in this case).

**Richardson weights for $\lambda = [1, 2, 3]$:**
$$c_1 = 3, \quad c_2 = -3, \quad c_3 = 1$$
$$E_{\text{Richardson}} = 3(-1.10) - 3(-1.06) + 1(-1.02) = -1.14$$

---

### Example 3: Variance Analysis

**Problem:** If ZNE uses $\lambda = [1, 3]$ with weights $c = [1.5, -0.5]$, and each measurement has variance $\sigma^2 = 0.01$, what is the variance of the mitigated estimate?

**Solution:**

The mitigated estimate:
$$E_{\text{ZNE}} = 1.5 \cdot E_1 - 0.5 \cdot E_3$$

Variance (assuming independence):
$$\text{Var}[E_{\text{ZNE}}] = 1.5^2 \cdot \sigma^2 + 0.5^2 \cdot \sigma^2 = (2.25 + 0.25) \cdot 0.01 = 0.025$$

**Variance inflation factor:** $2.5\times$

**Implication:** Need 2.5x more shots to achieve same precision as single unmitigated measurement.

---

## Practice Problems

### Level 1: Direct Application

1. For ZNE with $\lambda = [1, 2]$ and measurements $E_1 = -0.95$, $E_2 = -0.90$, compute the linear extrapolation.

2. Calculate the variance inflation factor for ZNE with weights $c = [2, -1]$.

3. If PEC has $\gamma_{\text{sum}} = 1.5$ per gate and the circuit has 20 gates, estimate the sampling overhead.

### Level 2: Intermediate

4. Derive the Richardson extrapolation weights for $\lambda = [1, 2, 3]$ assuming linear error model.

5. Design a CDR training set for a VQE circuit with one RY gate and two CNOT gates.

6. For symmetry verification with success probability 0.8, how many raw measurements are needed to get 1000 post-selected samples?

### Level 3: Challenging

7. Prove that the linear ZNE extrapolation is unbiased for error models of the form $E(\lambda) = E_0 + a\lambda$ but biased for $E(\lambda) = E_0 + a\lambda + b\lambda^2$.

8. Derive the quasi-probability representation for a depolarizing channel with error rate $p$. What is $\gamma_{\text{sum}}$?

9. **Research:** How can machine learning improve error mitigation? Investigate neural network-based denoising approaches.

---

## Computational Lab

### Objective
Implement error mitigation techniques for VQE.

```python
"""
Day 979 Computational Lab: Error-Mitigated Variational Algorithms
Advanced Variational Methods - Week 140
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# Part 1: Setup - VQE with Simulated Noise
# =============================================================================

print("=" * 70)
print("Part 1: VQE Setup with Simulated Noise")
print("=" * 70)

n_qubits = 2

# Create a simple 2-qubit Hamiltonian
H = qml.Hamiltonian(
    [1.0, 0.5, 0.5, -0.3],
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1),
     qml.PauliY(0) @ qml.PauliY(1)]
)

# Ideal device
dev_ideal = qml.device('default.qubit', wires=n_qubits)

# Noisy device (using default.mixed with noise)
dev_noisy = qml.device('default.mixed', wires=n_qubits)

def ansatz(params, wires):
    """Simple 2-qubit ansatz."""
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)

n_params = 4

@qml.qnode(dev_ideal)
def circuit_ideal(params):
    ansatz(params, range(n_qubits))
    return qml.expval(H)

# Noisy circuit with depolarizing noise
noise_strength = 0.02

@qml.qnode(dev_noisy)
def circuit_noisy(params, noise_scale=1.0):
    """Circuit with depolarizing noise after each gate."""
    qml.RY(params[0], wires=0)
    qml.DepolarizingChannel(noise_strength * noise_scale, wires=0)
    qml.RY(params[1], wires=1)
    qml.DepolarizingChannel(noise_strength * noise_scale, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.DepolarizingChannel(noise_strength * noise_scale, wires=0)
    qml.DepolarizingChannel(noise_strength * noise_scale, wires=1)
    qml.RY(params[2], wires=0)
    qml.DepolarizingChannel(noise_strength * noise_scale, wires=0)
    qml.RY(params[3], wires=1)
    qml.DepolarizingChannel(noise_strength * noise_scale, wires=1)
    return qml.expval(H)

# Find ground state (ideal)
def cost_ideal(params):
    return float(circuit_ideal(pnp.array(params)))

result_ideal = minimize(cost_ideal, np.random.uniform(0, np.pi, n_params),
                       method='COBYLA', options={'maxiter': 100})
optimal_params = result_ideal.x
E_exact = result_ideal.fun

print(f"Optimal parameters: {optimal_params}")
print(f"Exact ground state energy: {E_exact:.6f}")

# Noisy energy at optimal parameters
E_noisy = float(circuit_noisy(optimal_params, noise_scale=1.0))
print(f"Noisy energy: {E_noisy:.6f}")
print(f"Error: {abs(E_noisy - E_exact)*1000:.2f} mHa")

# =============================================================================
# Part 2: Zero-Noise Extrapolation
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Zero-Noise Extrapolation (ZNE)")
print("=" * 70)

def zne_linear(params, scale_factors=[1.0, 2.0]):
    """Linear ZNE extrapolation."""
    energies = []
    for scale in scale_factors:
        E = float(circuit_noisy(params, noise_scale=scale))
        energies.append(E)

    # Linear extrapolation to scale=0
    # E(scale) = E_0 + m * scale
    # Using scale_factors [1, 2], extrapolate to 0
    lambda1, lambda2 = scale_factors
    E1, E2 = energies

    E_mitigated = (lambda2 * E1 - lambda1 * E2) / (lambda2 - lambda1)
    return E_mitigated, energies

def zne_richardson(params, scale_factors=[1.0, 2.0, 3.0]):
    """Richardson (quadratic) ZNE extrapolation."""
    energies = []
    for scale in scale_factors:
        E = float(circuit_noisy(params, noise_scale=scale))
        energies.append(E)

    # Fit quadratic: E(s) = a + b*s + c*s^2
    # Use least squares or direct formula
    s = np.array(scale_factors)
    E = np.array(energies)

    # Vandermonde matrix
    V = np.vstack([np.ones_like(s), s, s**2]).T
    coeffs = np.linalg.lstsq(V, E, rcond=None)[0]

    E_mitigated = coeffs[0]  # Value at s=0
    return E_mitigated, energies

# Apply ZNE
E_zne_linear, energies_linear = zne_linear(optimal_params, [1.0, 3.0])
E_zne_richardson, energies_richardson = zne_richardson(optimal_params, [1.0, 2.0, 3.0])

print(f"\nZNE Results at optimal parameters:")
print(f"  Exact energy:        {E_exact:.6f}")
print(f"  Noisy energy (s=1):  {E_noisy:.6f}, error = {abs(E_noisy - E_exact)*1000:.2f} mHa")
print(f"  ZNE linear:          {E_zne_linear:.6f}, error = {abs(E_zne_linear - E_exact)*1000:.2f} mHa")
print(f"  ZNE Richardson:      {E_zne_richardson:.6f}, error = {abs(E_zne_richardson - E_exact)*1000:.2f} mHa")

# =============================================================================
# Part 3: Variance Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: ZNE Variance Analysis")
print("=" * 70)

def zne_with_variance(params, scale_factors, n_samples=50):
    """Compute ZNE with variance estimation."""
    # Simulate shot noise
    all_energies = {s: [] for s in scale_factors}

    for _ in range(n_samples):
        for scale in scale_factors:
            # Add simulated shot noise
            E = float(circuit_noisy(params, noise_scale=scale))
            E += np.random.normal(0, 0.02)  # Simulated shot noise
            all_energies[scale].append(E)

    # Compute mean energies
    mean_energies = [np.mean(all_energies[s]) for s in scale_factors]

    # Linear extrapolation
    s1, s2 = scale_factors[0], scale_factors[1]
    E1, E2 = mean_energies[0], mean_energies[1]
    E_zne = (s2 * E1 - s1 * E2) / (s2 - s1)

    # Variance of ZNE estimate
    var1 = np.var(all_energies[s1]) / n_samples
    var2 = np.var(all_energies[s2]) / n_samples
    c1 = s2 / (s2 - s1)
    c2 = -s1 / (s2 - s1)
    var_zne = c1**2 * var1 + c2**2 * var2

    return E_zne, np.sqrt(var_zne)

E_zne, std_zne = zne_with_variance(optimal_params, [1.0, 3.0])
print(f"ZNE estimate: {E_zne:.4f} +/- {std_zne:.4f}")

# Variance inflation
E_noisy_samples = [float(circuit_noisy(optimal_params)) + np.random.normal(0, 0.02)
                   for _ in range(50)]
std_noisy = np.std(E_noisy_samples) / np.sqrt(50)
print(f"Unmitigated std: {std_noisy:.4f}")
print(f"Variance inflation factor: {(std_zne/std_noisy)**2:.2f}x")

# =============================================================================
# Part 4: Clifford Data Regression (Simplified)
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Clifford Data Regression (CDR)")
print("=" * 70)

def generate_cdr_training_data(n_training=20):
    """Generate training data using near-Clifford circuits."""
    training_ideal = []
    training_noisy = []

    for _ in range(n_training):
        # Random Clifford-ish parameters (multiples of pi/2)
        clifford_params = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_params)
        # Add small perturbation
        params = clifford_params + np.random.uniform(-0.1, 0.1, n_params)

        E_ideal = float(circuit_ideal(params))
        E_noisy = float(circuit_noisy(params))

        training_ideal.append(E_ideal)
        training_noisy.append(E_noisy)

    return np.array(training_noisy), np.array(training_ideal)

X_train, y_train = generate_cdr_training_data(30)

# Linear regression
from numpy.polynomial import polynomial as P

# Fit E_ideal = a * E_noisy + b
coeffs = np.polyfit(X_train, y_train, 1)
a, b = coeffs

print(f"\nCDR linear fit: E_ideal = {a:.3f} * E_noisy + {b:.3f}")

# Apply to target
E_cdr = a * E_noisy + b
print(f"\nCDR Results:")
print(f"  Exact:  {E_exact:.6f}")
print(f"  Noisy:  {E_noisy:.6f}")
print(f"  CDR:    {E_cdr:.6f}")
print(f"  Error:  {abs(E_cdr - E_exact)*1000:.2f} mHa")

# =============================================================================
# Part 5: Error-Mitigated Optimization
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Error-Mitigated Optimization")
print("=" * 70)

def cost_noisy_wrapper(params):
    return float(circuit_noisy(pnp.array(params)))

def cost_zne(params):
    """Cost function with ZNE."""
    E_mitigated, _ = zne_linear(params, [1.0, 3.0])
    return E_mitigated

# Optimize without mitigation
print("\nOptimizing without mitigation...")
result_noisy = minimize(cost_noisy_wrapper, np.random.uniform(0, np.pi, n_params),
                       method='COBYLA', options={'maxiter': 100})
E_final_noisy = float(circuit_ideal(result_noisy.x))  # Evaluate ideal at noisy-optimal params

# Optimize with ZNE (more expensive)
print("Optimizing with ZNE (slower)...")
result_zne = minimize(cost_zne, np.random.uniform(0, np.pi, n_params),
                     method='COBYLA', options={'maxiter': 50})  # Fewer iters due to cost
E_final_zne = float(circuit_ideal(result_zne.x))

print(f"\nOptimization Results (evaluated on ideal device):")
print(f"  Noisy optimization:     {E_final_noisy:.6f}, error = {abs(E_final_noisy - E_exact)*1000:.2f} mHa")
print(f"  ZNE optimization:       {E_final_zne:.6f}, error = {abs(E_final_zne - E_exact)*1000:.2f} mHa")
print(f"  True optimum:           {E_exact:.6f}")

# =============================================================================
# Part 6: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ZNE extrapolation plot
ax1 = axes[0, 0]
scale_factors = [1.0, 2.0, 3.0, 4.0]
energies_scan = [float(circuit_noisy(optimal_params, noise_scale=s)) for s in scale_factors]
ax1.plot(scale_factors, energies_scan, 'bo-', markersize=10, label='Noisy measurements')
ax1.axhline(y=E_exact, color='g', linestyle='--', linewidth=2, label='Exact')
ax1.axhline(y=E_zne_linear, color='r', linestyle=':', linewidth=2, label='ZNE Linear')
ax1.axhline(y=E_zne_richardson, color='orange', linestyle='-.', linewidth=2, label='ZNE Richardson')
ax1.plot(0, E_exact, 'g*', markersize=15)
ax1.set_xlabel('Noise Scale Factor')
ax1.set_ylabel('Energy')
ax1.set_title('Zero-Noise Extrapolation')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, 4.5)

# CDR training data
ax2 = axes[0, 1]
ax2.scatter(X_train, y_train, alpha=0.6, label='Training data')
x_line = np.linspace(min(X_train), max(X_train), 100)
ax2.plot(x_line, a * x_line + b, 'r-', linewidth=2, label=f'Fit: y = {a:.2f}x + {b:.2f}')
ax2.scatter([E_noisy], [E_cdr], color='green', s=200, marker='*', label='CDR prediction')
ax2.set_xlabel('Noisy Energy')
ax2.set_ylabel('Ideal Energy')
ax2.set_title('Clifford Data Regression')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Method comparison
ax3 = axes[1, 0]
methods = ['Noisy', 'ZNE Linear', 'ZNE Richardson', 'CDR']
energies = [E_noisy, E_zne_linear, E_zne_richardson, E_cdr]
errors = [abs(e - E_exact) * 1000 for e in energies]
colors = ['red', 'blue', 'orange', 'green']
bars = ax3.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Error (mHa)')
ax3.set_title('Error Mitigation Comparison')
ax3.grid(True, alpha=0.3, axis='y')
for bar, err in zip(bars, errors):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{err:.1f}', ha='center', va='bottom')

# Variance comparison
ax4 = axes[1, 1]
# Simulate variance for different methods
n_samples = 30
noisy_samples = [float(circuit_noisy(optimal_params)) + np.random.normal(0, 0.02)
                 for _ in range(n_samples)]
zne_samples = []
for _ in range(n_samples):
    e1 = float(circuit_noisy(optimal_params, 1.0)) + np.random.normal(0, 0.02)
    e2 = float(circuit_noisy(optimal_params, 3.0)) + np.random.normal(0, 0.02)
    zne_samples.append((3*e1 - e2)/2)

ax4.violinplot([noisy_samples, zne_samples], positions=[1, 2])
ax4.axhline(y=E_exact, color='g', linestyle='--', label='Exact')
ax4.set_xticks([1, 2])
ax4.set_xticklabels(['Noisy', 'ZNE'])
ax4.set_ylabel('Energy')
ax4.set_title('Variance Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_979_error_mitigation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_979_error_mitigation.png'")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| ZNE extrapolation | $E_{\text{mit}} = \sum_k c_k E(\lambda_k)$, with $\sum c_k = 1$, $\sum c_k \lambda_k = 0$ |
| Linear ZNE | $E_0 = \frac{\lambda_2 E_1 - \lambda_1 E_2}{\lambda_2 - \lambda_1}$ |
| PEC overhead | $\gamma_{\text{total}} = \prod_g \gamma_g$ |
| CDR model | $E_{\text{ideal}} = a \cdot E_{\text{noisy}} + b$ |
| Variance inflation | $\text{Var}[E_{\text{ZNE}}] = \sum_k c_k^2 \text{Var}[E_k]$ |

### Main Takeaways

1. **Error mitigation is essential** for meaningful NISQ results
2. **ZNE is widely applicable** and relatively low overhead
3. **PEC gives unbiased results** but with exponential sampling cost
4. **CDR learns device-specific errors** from classically simulable circuits
5. **Symmetry verification** provides complementary error reduction
6. **Trade-offs exist** between bias reduction and variance increase
7. **Hybrid strategies** (coarse unmitigated + fine mitigated) are practical

---

## Daily Checklist

- [ ] Understand ZNE linear and Richardson extrapolation
- [ ] Calculate variance inflation factors
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and modify the computational lab
- [ ] Compare ZNE and CDR for a specific problem

---

## Preview: Day 980

Tomorrow concludes Month 35 with a **Synthesis and Capstone Preview**. We'll integrate all the algorithms covered—HHL, quantum simulation, QML, and advanced VQE—and prepare for the Year 2 capstone project.

---

*"Error mitigation is the bridge between imperfect hardware and useful computation."*
--- NISQ computing philosophy

---

**Next:** [Day_980_Sunday.md](Day_980_Sunday.md) - Month 35 Synthesis & Capstone Preview
