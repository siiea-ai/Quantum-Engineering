# Day 940: Zero-Noise Extrapolation - Noise Scaling and Richardson Extrapolation

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Noise scaling methods, Richardson extrapolation theory |
| Afternoon | 2 hours | Problem solving and error analysis |
| Evening | 2 hours | Computational lab - ZNE implementation |

## Learning Objectives

By the end of this day, you will be able to:

1. **Implement noise scaling** - Apply pulse stretching and gate folding to amplify noise
2. **Apply Richardson extrapolation** - Use linear and polynomial fits to extract zero-noise values
3. **Analyze extrapolation error** - Quantify uncertainty and optimal scale factor selection
4. **Choose scaling strategies** - Select appropriate methods for different noise types
5. **Optimize ZNE parameters** - Balance accuracy vs sampling cost
6. **Implement ZNE in Qiskit/Mitiq** - Build practical ZNE pipelines

## Core Content

### 1. Zero-Noise Extrapolation Fundamentals

Zero-Noise Extrapolation (ZNE) is based on a simple observation: if we can run a circuit at multiple noise levels, we can extrapolate to the zero-noise limit.

#### 1.1 The ZNE Principle

For an expectation value $\langle O \rangle$ measured at noise level $\lambda$:

$$\boxed{\langle O \rangle_\lambda = \langle O \rangle_0 + a_1 \lambda + a_2 \lambda^2 + \ldots}$$

By measuring at multiple noise levels $\lambda_1, \lambda_2, \ldots, \lambda_n$ and fitting this polynomial, we can extrapolate to $\lambda = 0$.

The key insight: **Physical noise has strength $\lambda = 1$. We artificially amplify it to $\lambda > 1$ and extrapolate backward to $\lambda = 0$.**

#### 1.2 Noise Model Assumptions

ZNE assumes the noise can be characterized by a single scale parameter:

$$\mathcal{E}_\lambda(\rho) = (1-\lambda p)\rho + \lambda p \mathcal{N}(\rho)$$

where $\mathcal{N}$ is the noise channel and $p$ is the base error probability.

For depolarizing noise:
$$\mathcal{E}_\lambda^{\text{dep}}(\rho) = (1-\lambda p)\rho + \frac{\lambda p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

### 2. Noise Scaling Methods

#### 2.1 Pulse Stretching

For coherent control, stretch pulse duration by factor $c$:

$$\boxed{T_{\text{gate}} \to c \cdot T_{\text{gate}}, \quad \Omega \to \Omega/c}$$

This increases decoherence during the gate while maintaining the same unitary.

**Advantages**: Smooth scaling, works for all gate types
**Disadvantages**: Requires pulse-level control

Effective noise scaling:
$$\lambda = c \quad \text{(for } T_1, T_2 \text{ limited errors)}$$

#### 2.2 Gate Folding (Unitary Folding)

Replace gate $G$ with $G \cdot G^\dagger \cdot G$ (identity operation, but 3x the noise):

$$\boxed{G \to G(G^\dagger G)^n, \quad \lambda = 2n + 1}$$

For $n$ fold operations, noise is amplified by factor $2n + 1$.

**Local Folding**: Fold individual gates
$$G_i \to G_i(G_i^\dagger G_i)^{n_i}$$

**Global Folding**: Fold the entire circuit
$$U \to U(U^\dagger U)^n$$

**Advantages**: Works with standard gate sets, no pulse control needed
**Disadvantages**: Circuit depth increases, discrete scaling factors

#### 2.3 Identity Insertion

Insert identity gates ($I = XX = YY = ZZ$) to add noise:

$$\boxed{G \to G \cdot I^n = G \cdot (XX)^n}$$

Each identity insertion adds gate error without changing the computation.

### 3. Richardson Extrapolation

#### 3.1 Linear Extrapolation

With two noise levels $\lambda_1 = 1$ and $\lambda_2 > 1$:

$$\langle O \rangle_0 = \frac{\lambda_2 \langle O \rangle_1 - \lambda_1 \langle O \rangle_{\lambda_2}}{\lambda_2 - \lambda_1}$$

For $\lambda_1 = 1, \lambda_2 = 3$ (single gate fold):

$$\boxed{\langle O \rangle_0 = \frac{3\langle O \rangle_1 - \langle O \rangle_3}{2}}$$

#### 3.2 Polynomial Extrapolation

For $n$ noise levels, fit polynomial of degree $n-1$:

$$\langle O \rangle_\lambda = \sum_{k=0}^{n-1} a_k \lambda^k$$

The zero-noise estimate:
$$\boxed{\langle O \rangle_0 = \sum_{i=1}^{n} \gamma_i \langle O \rangle_{\lambda_i}}$$

where coefficients $\gamma_i$ are determined by solving:

$$\sum_{i=1}^{n} \gamma_i \lambda_i^k = \delta_{k,0} \quad \text{for } k = 0, 1, \ldots, n-1$$

#### 3.3 Richardson Coefficients

For equally spaced scale factors $\lambda_i = 1 + (i-1)\Delta$:

$$\gamma_i = \prod_{j \neq i} \frac{-\lambda_j}{\lambda_i - \lambda_j}$$

**Example**: $\lambda = (1, 2, 3)$

$$\gamma_1 = \frac{(-2)(-3)}{(1-2)(1-3)} = \frac{6}{2} = 3$$
$$\gamma_2 = \frac{(-1)(-3)}{(2-1)(2-3)} = \frac{3}{-1} = -3$$
$$\gamma_3 = \frac{(-1)(-2)}{(3-1)(3-2)} = \frac{2}{2} = 1$$

Verify: $\gamma_1 + \gamma_2 + \gamma_3 = 3 - 3 + 1 = 1$ (correct normalization)

$$\boxed{\langle O \rangle_0 = 3\langle O \rangle_1 - 3\langle O \rangle_2 + \langle O \rangle_3}$$

### 4. Exponential Extrapolation

For some noise models, exponential decay is more appropriate:

$$\langle O \rangle_\lambda = A e^{-b\lambda} + c$$

At $\lambda = 0$:
$$\boxed{\langle O \rangle_0 = A + c}$$

This requires fitting three parameters $(A, b, c)$ and works well for depolarizing channels on expectation values.

### 5. Error Analysis

#### 5.1 Variance Amplification

Richardson extrapolation amplifies statistical noise:

$$\text{Var}[\langle O \rangle_0] = \sum_i \gamma_i^2 \cdot \text{Var}[\langle O \rangle_{\lambda_i}]$$

The variance amplification factor:

$$\boxed{\Gamma = \sum_i \gamma_i^2}$$

For $\lambda = (1, 3)$: $\Gamma = (3/2)^2 + (1/2)^2 = 2.5$
For $\lambda = (1, 2, 3)$: $\Gamma = 9 + 9 + 1 = 19$

**Trade-off**: Higher-order extrapolation reduces bias but increases variance.

#### 5.2 Optimal Noise Levels

Given total shot budget $N$ and $k$ noise levels, optimal allocation:

$$N_i \propto |\gamma_i| \sqrt{\text{Var}[\langle O \rangle_{\lambda_i}]}$$

For equal variance at all noise levels:

$$N_i \propto |\gamma_i|$$

#### 5.3 Extrapolation Bias

If true dependence has higher-order terms than the fit:

$$\text{Bias} = \sum_{k=n}^{\infty} a_k \sum_{i=1}^n \gamma_i \lambda_i^k$$

For well-behaved noise, bias decreases exponentially with polynomial order.

### 6. Practical Considerations

#### 6.1 Scale Factor Selection

**Guidelines**:
- Minimum: $\lambda_{\min} = 1$ (no amplification)
- Maximum: $\lambda_{\max} \lesssim 1/\epsilon$ where $\epsilon$ is base error rate
- Too large $\lambda$ gives near-random results

For 1% gate error: $\lambda_{\max} \lesssim 10-20$

#### 6.2 Noise Type Dependence

| Noise Type | Best Scaling Method | Extrapolation |
|------------|--------------------|--------------|
| Depolarizing | Gate folding | Polynomial or exponential |
| Dephasing | Pulse stretching | Polynomial |
| Amplitude damping | Pulse stretching | Exponential |
| Coherent errors | Identity insertion | Linear |

## Quantum Computing Applications

### VQE Energy Estimation

For variational quantum eigensolver, ZNE improves energy estimates:

$$E_{\text{mitigated}} = \sum_i \gamma_i E_{\lambda_i}$$

Typical improvement: 2-10x reduction in energy error.

### QAOA Optimization

Zero-noise extrapolation on QAOA cost function:
- Improves solution quality
- Helps avoid local minima caused by noise
- Particularly effective for shallow circuits

### Quantum Chemistry

For molecular simulations on NISQ devices:
- Error in energy directly affects chemical accuracy
- ZNE can bridge the gap to 1.6 mHa threshold

## Worked Examples

### Example 1: Linear Extrapolation

**Problem**: A circuit measured at $\lambda = 1$ gives $\langle Z \rangle = 0.7$, and at $\lambda = 3$ gives $\langle Z \rangle = 0.3$. Estimate the zero-noise value.

**Solution**:

Using linear extrapolation with $\lambda_1 = 1, \lambda_2 = 3$:

$$\langle Z \rangle_0 = \frac{\lambda_2 \langle Z \rangle_1 - \lambda_1 \langle Z \rangle_3}{\lambda_2 - \lambda_1}$$

$$= \frac{3 \times 0.7 - 1 \times 0.3}{3 - 1}$$

$$= \frac{2.1 - 0.3}{2} = \frac{1.8}{2}$$

$$\boxed{\langle Z \rangle_0 = 0.9}$$

### Example 2: Polynomial Extrapolation

**Problem**: Measurements at $\lambda = (1, 2, 3)$ give $\langle H \rangle = (-1.0, -0.8, -0.5)$ respectively. Find the zero-noise energy using quadratic extrapolation.

**Solution**:

Using Richardson coefficients $\gamma = (3, -3, 1)$:

$$\langle H \rangle_0 = 3 \times (-1.0) - 3 \times (-0.8) + 1 \times (-0.5)$$

$$= -3.0 + 2.4 - 0.5$$

$$\boxed{\langle H \rangle_0 = -1.1}$$

Alternatively, fit quadratic $a_0 + a_1\lambda + a_2\lambda^2$:
- At $\lambda = 1$: $a_0 + a_1 + a_2 = -1.0$
- At $\lambda = 2$: $a_0 + 2a_1 + 4a_2 = -0.8$
- At $\lambda = 3$: $a_0 + 3a_1 + 9a_2 = -0.5$

Solving: $a_0 = -1.1, a_1 = 0.05, a_2 = 0.05$

### Example 3: Variance Amplification

**Problem**: With scale factors $\lambda = (1, 2, 3)$ and $N = 30000$ total shots, how should shots be allocated to minimize variance?

**Solution**:

Richardson coefficients: $\gamma = (3, -3, 1)$

Optimal allocation proportional to $|\gamma_i|$:
$$N_i \propto |\gamma_i|$$

Sum of $|\gamma_i| = 3 + 3 + 1 = 7$

$$N_1 = \frac{3}{7} \times 30000 = 12857$$
$$N_2 = \frac{3}{7} \times 30000 = 12857$$
$$N_3 = \frac{1}{7} \times 30000 = 4286$$

$$\boxed{N_1 = 12857, \quad N_2 = 12857, \quad N_3 = 4286}$$

## Practice Problems

### Level 1: Direct Application

1. Calculate Richardson coefficients for scale factors $\lambda = (1, 5)$.

2. Given $\langle Z \rangle = (0.8, 0.6, 0.3)$ at $\lambda = (1, 2, 4)$, find the zero-noise expectation using linear extrapolation with the first two points.

3. How many gate folds are needed to achieve noise amplification $\lambda = 7$?

### Level 2: Intermediate

4. Derive the variance amplification factor $\Gamma$ for exponential extrapolation $\langle O \rangle_0 = A + c$ fitted from three noise levels.

5. Compare linear vs quadratic extrapolation for data $\langle O \rangle = (1.0, 0.7, 0.5, 0.35)$ at $\lambda = (1, 2, 3, 4)$. Which gives a more reasonable zero-noise estimate?

6. Design a ZNE protocol for a 20-gate circuit with 2% gate error. Specify scale factors and total shot budget to achieve 10% relative error.

### Level 3: Challenging

7. Derive the optimal polynomial order for ZNE given noise model $\langle O \rangle_\lambda = O_0(1-p)^{n\lambda}$ for an $n$-gate circuit with error $p$.

8. Prove that for depolarizing noise, exponential extrapolation is exact for Pauli expectation values.

9. Analyze the breakdown of ZNE when scale factors become too large: at what $\lambda_{\max}$ does shot noise dominate the extrapolated signal?

## Computational Lab

```python
"""
Day 940: Zero-Noise Extrapolation Implementation
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Tuple, Callable

# ============================================================
# Part 1: Noise Scaling Methods
# ============================================================

def fold_gate(qc: QuantumCircuit, gate_idx: int, num_folds: int) -> QuantumCircuit:
    """Fold a specific gate in the circuit."""
    # Get circuit data
    data = list(qc.data)
    gate = data[gate_idx]

    # Create folded sequence: G(G†G)^n
    folded_gates = [gate]
    for _ in range(num_folds):
        # Add G†
        inv_gate = gate.operation.inverse()
        folded_gates.append((inv_gate, gate.qubits, gate.clbits))
        # Add G
        folded_gates.append(gate)

    # Build new circuit
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for i, g in enumerate(data):
        if i == gate_idx:
            for fg in folded_gates:
                if isinstance(fg, tuple):
                    new_qc.append(fg[0], fg[1], fg[2])
                else:
                    new_qc.append(fg.operation, fg.qubits, fg.clbits)
        else:
            new_qc.append(g.operation, g.qubits, g.clbits)

    return new_qc

def fold_circuit_globally(qc: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
    """
    Fold entire circuit to achieve target scale factor.
    Scale factor = 1 + 2*num_folds for integer folds.
    For non-integer, we fold a fraction of gates locally.
    """
    if scale_factor < 1:
        raise ValueError("Scale factor must be >= 1")

    # Number of complete global folds
    num_global_folds = int((scale_factor - 1) / 2)
    remaining = scale_factor - (1 + 2 * num_global_folds)

    # Create base circuit without measurements
    base_qc = QuantumCircuit(qc.num_qubits)
    for gate in qc.data:
        if gate.operation.name != 'measure':
            base_qc.append(gate.operation, gate.qubits, gate.clbits)

    # Build folded circuit
    folded_qc = base_qc.copy()

    for _ in range(num_global_folds):
        # Add U†
        folded_qc = folded_qc.compose(base_qc.inverse())
        # Add U
        folded_qc = folded_qc.compose(base_qc)

    # Handle fractional folding by folding some gates locally
    if remaining > 0.01:  # Avoid floating point issues
        num_gates = len(base_qc.data)
        gates_to_fold = int(remaining * num_gates / 2)

        for i in range(gates_to_fold):
            gate = base_qc.data[i]
            inv_gate = gate.operation.inverse()
            folded_qc.append(inv_gate, gate.qubits)
            folded_qc.append(gate.operation, gate.qubits)

    return folded_qc

def create_test_circuit(n_qubits: int = 2, depth: int = 5) -> QuantumCircuit:
    """Create a test circuit for ZNE demonstration."""
    qc = QuantumCircuit(n_qubits)

    for d in range(depth):
        for q in range(n_qubits):
            qc.rx(np.pi/4, q)
            qc.rz(np.pi/3, q)

        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc

# ============================================================
# Part 2: Richardson Extrapolation
# ============================================================

def richardson_coefficients(scale_factors: List[float]) -> np.ndarray:
    """Compute Richardson extrapolation coefficients."""
    n = len(scale_factors)
    lambdas = np.array(scale_factors)

    # Build Vandermonde-like system
    # We want sum_i gamma_i * lambda_i^k = delta_{k,0}
    A = np.vander(lambdas, increasing=True).T[:n, :]
    b = np.zeros(n)
    b[0] = 1.0

    gamma = np.linalg.solve(A, b)
    return gamma

def linear_extrapolation(values: List[float], scale_factors: List[float]) -> float:
    """Perform linear extrapolation to zero noise."""
    if len(values) < 2:
        raise ValueError("Need at least 2 data points")

    # Use first two points
    lambda1, lambda2 = scale_factors[0], scale_factors[1]
    v1, v2 = values[0], values[1]

    return (lambda2 * v1 - lambda1 * v2) / (lambda2 - lambda1)

def polynomial_extrapolation(values: List[float], scale_factors: List[float],
                            order: int = None) -> Tuple[float, np.ndarray]:
    """Perform polynomial Richardson extrapolation."""
    n = len(values)
    if order is None:
        order = n - 1

    # Fit polynomial
    coeffs = np.polyfit(scale_factors[:order+1], values[:order+1], order)

    # Extrapolate to zero
    zero_noise_value = np.polyval(coeffs, 0)

    return zero_noise_value, coeffs

def exponential_extrapolation(values: List[float], scale_factors: List[float]) -> Tuple[float, dict]:
    """Fit exponential model and extrapolate to zero noise."""

    def exp_model(x, a, b, c):
        return a * np.exp(-b * x) + c

    try:
        popt, pcov = curve_fit(exp_model, scale_factors, values,
                              p0=[1, 0.5, 0], maxfev=5000)
        zero_noise = popt[0] + popt[2]  # A + c at lambda=0
        params = {'A': popt[0], 'b': popt[1], 'c': popt[2]}
        return zero_noise, params
    except:
        # Fallback to polynomial
        return polynomial_extrapolation(values, scale_factors)[0], {}

# ============================================================
# Part 3: ZNE Implementation
# ============================================================

class ZeroNoiseExtrapolator:
    """Complete ZNE implementation."""

    def __init__(self, scale_factors: List[float] = [1.0, 2.0, 3.0],
                 extrapolation_method: str = 'polynomial'):
        self.scale_factors = scale_factors
        self.extrapolation_method = extrapolation_method

    def scale_circuit(self, qc: QuantumCircuit, scale: float) -> QuantumCircuit:
        """Scale circuit noise by folding."""
        return fold_circuit_globally(qc, scale)

    def run_with_zne(self, qc: QuantumCircuit,
                     observable: Callable,
                     noise_model: NoiseModel,
                     shots: int = 8192) -> dict:
        """Run circuit with ZNE and return results."""

        simulator = AerSimulator(noise_model=noise_model)

        # Collect expectation values at each scale factor
        exp_values = []
        variances = []

        for scale in self.scale_factors:
            # Scale the circuit
            scaled_qc = self.scale_circuit(qc, scale)
            scaled_qc.measure_all()

            # Run and compute expectation value
            result = simulator.run(scaled_qc, shots=shots).result()
            counts = result.get_counts()

            # Compute observable expectation
            exp_val, var = observable(counts, shots)
            exp_values.append(exp_val)
            variances.append(var)

        # Extrapolate
        if self.extrapolation_method == 'linear':
            mitigated = linear_extrapolation(exp_values, self.scale_factors)
        elif self.extrapolation_method == 'polynomial':
            mitigated, _ = polynomial_extrapolation(exp_values, self.scale_factors)
        elif self.extrapolation_method == 'exponential':
            mitigated, _ = exponential_extrapolation(exp_values, self.scale_factors)
        else:
            raise ValueError(f"Unknown method: {self.extrapolation_method}")

        # Compute Richardson coefficients for error estimation
        gamma = richardson_coefficients(self.scale_factors)
        variance_amplification = np.sum(gamma**2 * np.array(variances))

        return {
            'mitigated': mitigated,
            'raw_values': exp_values,
            'scale_factors': self.scale_factors,
            'coefficients': gamma,
            'variance_amplification': variance_amplification,
            'mitigated_std': np.sqrt(variance_amplification)
        }

def z_expectation(counts: dict, shots: int) -> Tuple[float, float]:
    """Compute Z expectation value from counts."""
    exp_val = 0
    for bitstring, count in counts.items():
        # Count parity of first qubit
        parity = int(bitstring[-1])  # Last character is first qubit
        exp_val += (1 - 2*parity) * count

    exp_val /= shots
    variance = (1 - exp_val**2) / shots

    return exp_val, variance

# ============================================================
# Part 4: ZNE Demonstration
# ============================================================

def demonstrate_zne():
    """Full ZNE demonstration on a test circuit."""

    # Create test circuit
    qc = create_test_circuit(n_qubits=2, depth=5)
    print("Test Circuit:")
    print(qc.draw())

    # Create noise model
    p_error = 0.02  # 2% depolarizing error
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_error, 1), ['rx', 'rz', 'h']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_error * 5, 2), ['cx']
    )

    # Get ideal value
    ideal_sim = AerSimulator()
    ideal_qc = qc.copy()
    ideal_qc.measure_all()
    ideal_result = ideal_sim.run(ideal_qc, shots=100000).result()
    ideal_exp, _ = z_expectation(ideal_result.get_counts(), 100000)
    print(f"\nIdeal expectation value: {ideal_exp:.4f}")

    # Get noisy value (no mitigation)
    noisy_sim = AerSimulator(noise_model=noise_model)
    noisy_result = noisy_sim.run(ideal_qc, shots=100000).result()
    noisy_exp, _ = z_expectation(noisy_result.get_counts(), 100000)
    print(f"Noisy expectation value: {noisy_exp:.4f}")

    # Apply ZNE with different methods
    scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = {}
    for method in ['linear', 'polynomial', 'exponential']:
        zne = ZeroNoiseExtrapolator(scale_factors=scale_factors,
                                   extrapolation_method=method)
        results[method] = zne.run_with_zne(qc, z_expectation, noise_model)
        print(f"\n{method.capitalize()} ZNE result: {results[method]['mitigated']:.4f} "
              f"(std: {results[method]['mitigated_std']:.4f})")

    # Visualize extrapolation
    plt.figure(figsize=(12, 5))

    # Plot raw data points
    raw_values = results['polynomial']['raw_values']
    plt.subplot(1, 2, 1)
    plt.scatter(scale_factors, raw_values, s=100, c='blue', zorder=5,
               label='Measured values')

    # Plot extrapolation fits
    x_extrap = np.linspace(0, max(scale_factors), 100)

    # Linear fit
    coeffs_lin = np.polyfit(scale_factors[:2], raw_values[:2], 1)
    plt.plot(x_extrap, np.polyval(coeffs_lin, x_extrap), 'g--',
            label=f'Linear: {results["linear"]["mitigated"]:.3f}')

    # Polynomial fit
    coeffs_poly = np.polyfit(scale_factors, raw_values, len(scale_factors)-1)
    plt.plot(x_extrap, np.polyval(coeffs_poly, x_extrap), 'r-',
            label=f'Polynomial: {results["polynomial"]["mitigated"]:.3f}')

    # Exponential fit
    def exp_model(x, a, b, c):
        return a * np.exp(-b * x) + c
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(exp_model, scale_factors, raw_values, p0=[1, 0.5, 0])
        plt.plot(x_extrap, exp_model(x_extrap, *popt), 'm:',
                label=f'Exponential: {results["exponential"]["mitigated"]:.3f}')
    except:
        pass

    # Reference lines
    plt.axhline(y=ideal_exp, color='black', linestyle='-', linewidth=2,
               label=f'Ideal: {ideal_exp:.3f}')
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Scale Factor λ', fontsize=12)
    plt.ylabel('⟨Z⟩', fontsize=12)
    plt.title('Zero-Noise Extrapolation', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Error comparison
    plt.subplot(1, 2, 2)
    methods = ['Noisy', 'Linear', 'Polynomial', 'Exponential']
    errors = [
        abs(noisy_exp - ideal_exp),
        abs(results['linear']['mitigated'] - ideal_exp),
        abs(results['polynomial']['mitigated'] - ideal_exp),
        abs(results['exponential']['mitigated'] - ideal_exp)
    ]

    colors = ['red', 'green', 'blue', 'purple']
    bars = plt.bar(methods, errors, color=colors, edgecolor='black')

    for bar, err in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{err:.4f}', ha='center', va='bottom', fontsize=10)

    plt.ylabel('|Error|', fontsize=12)
    plt.title('Error Comparison', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('zne_demonstration.png', dpi=150)
    plt.show()

    # Print improvement factors
    print("\n" + "="*50)
    print("Error Improvement Factors:")
    base_error = abs(noisy_exp - ideal_exp)
    for method in ['linear', 'polynomial', 'exponential']:
        mit_error = abs(results[method]['mitigated'] - ideal_exp)
        improvement = base_error / mit_error if mit_error > 0 else float('inf')
        print(f"  {method.capitalize()}: {improvement:.2f}x improvement")

    return results

# ============================================================
# Part 5: Scale Factor Optimization
# ============================================================

def analyze_scale_factor_impact():
    """Analyze how scale factor choice affects ZNE performance."""

    qc = create_test_circuit(n_qubits=2, depth=5)

    # Noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.02, 1), ['rx', 'rz']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.10, 2), ['cx']
    )

    # Get ideal reference
    ideal_sim = AerSimulator()
    ideal_qc = qc.copy()
    ideal_qc.measure_all()
    ideal_result = ideal_sim.run(ideal_qc, shots=100000).result()
    ideal_exp, _ = z_expectation(ideal_result.get_counts(), 100000)

    # Test different scale factor ranges
    max_scales = [2, 3, 4, 5, 7, 10]
    num_points_options = [2, 3, 4, 5]

    results_matrix = np.zeros((len(max_scales), len(num_points_options)))

    for i, max_scale in enumerate(max_scales):
        for j, num_points in enumerate(num_points_options):
            if num_points > max_scale:
                results_matrix[i, j] = np.nan
                continue

            scale_factors = list(np.linspace(1, max_scale, num_points))

            try:
                zne = ZeroNoiseExtrapolator(scale_factors=scale_factors,
                                           extrapolation_method='polynomial')
                result = zne.run_with_zne(qc, z_expectation, noise_model)
                error = abs(result['mitigated'] - ideal_exp)
                results_matrix[i, j] = error
            except:
                results_matrix[i, j] = np.nan

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.imshow(results_matrix, aspect='auto', cmap='RdYlGn_r')
    plt.colorbar(label='|Error|')
    plt.xticks(range(len(num_points_options)), num_points_options)
    plt.yticks(range(len(max_scales)), max_scales)
    plt.xlabel('Number of Scale Factor Points')
    plt.ylabel('Maximum Scale Factor')
    plt.title('ZNE Error vs Scale Factor Configuration')

    # Add text annotations
    for i in range(len(max_scales)):
        for j in range(len(num_points_options)):
            val = results_matrix[i, j]
            if not np.isnan(val):
                plt.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color='white' if val > 0.1 else 'black')

    plt.tight_layout()
    plt.savefig('scale_factor_analysis.png', dpi=150)
    plt.show()

# ============================================================
# Part 6: Using Mitiq Library
# ============================================================

def demonstrate_mitiq_zne():
    """Demonstrate ZNE using the Mitiq library."""
    try:
        from mitiq import zne
        from mitiq.zne.scaling import fold_gates_at_random
        from mitiq.zne.inference import RichardsonFactory, LinearFactory

        print("\n" + "="*50)
        print("Mitiq ZNE Demonstration")
        print("="*50)

        # Create circuit
        qc = create_test_circuit(n_qubits=2, depth=3)

        # Define executor
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.02, 1), ['rx', 'rz']
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.08, 2), ['cx']
        )

        simulator = AerSimulator(noise_model=noise_model)

        def executor(circuit) -> float:
            """Execute circuit and return Z expectation."""
            meas_circuit = circuit.copy()
            meas_circuit.measure_all()
            result = simulator.run(meas_circuit, shots=8192).result()
            counts = result.get_counts()

            exp_val = 0
            for bitstring, count in counts.items():
                parity = int(bitstring[-1])
                exp_val += (1 - 2*parity) * count

            return exp_val / 8192

        # Run with different factories
        scale_factors = [1, 2, 3]

        # Linear extrapolation
        linear_factory = LinearFactory(scale_factors=scale_factors[:2])
        linear_result = zne.execute_with_zne(
            qc, executor,
            factory=linear_factory,
            scale_noise=fold_gates_at_random
        )
        print(f"Linear ZNE result: {linear_result:.4f}")

        # Richardson extrapolation
        richardson_factory = RichardsonFactory(scale_factors=scale_factors)
        richardson_result = zne.execute_with_zne(
            qc, executor,
            factory=richardson_factory,
            scale_noise=fold_gates_at_random
        )
        print(f"Richardson ZNE result: {richardson_result:.4f}")

        # Unmitigated
        unmitigated = executor(qc)
        print(f"Unmitigated result: {unmitigated:.4f}")

        # Ideal
        ideal_sim = AerSimulator()
        ideal_qc = qc.copy()
        ideal_qc.measure_all()
        ideal_result = ideal_sim.run(ideal_qc, shots=100000).result()
        ideal_exp, _ = z_expectation(ideal_result.get_counts(), 100000)
        print(f"Ideal result: {ideal_exp:.4f}")

    except ImportError:
        print("\nMitiq not installed. Install with: pip install mitiq")
        print("Skipping Mitiq demonstration.")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Day 940: Zero-Noise Extrapolation Lab")
    print("="*60)

    # Part 1: Core ZNE demonstration
    print("\n--- Part 1: ZNE Demonstration ---")
    results = demonstrate_zne()

    # Part 2: Scale factor analysis
    print("\n--- Part 2: Scale Factor Analysis ---")
    analyze_scale_factor_impact()

    # Part 3: Mitiq integration
    print("\n--- Part 3: Mitiq Library ---")
    demonstrate_mitiq_zne()

    print("\n" + "="*60)
    print("Lab Complete! Key Takeaways:")
    print("  1. Gate folding effectively amplifies noise for ZNE")
    print("  2. Polynomial order should match noise model complexity")
    print("  3. Scale factors shouldn't exceed 1/error_rate")
    print("  4. Variance amplification limits achievable accuracy")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Gate Folding | $G \to G(G^\dagger G)^n$, $\lambda = 2n + 1$ |
| Linear Extrapolation | $\langle O \rangle_0 = \frac{\lambda_2 \langle O \rangle_1 - \lambda_1 \langle O \rangle_2}{\lambda_2 - \lambda_1}$ |
| Richardson Coefficients | $\gamma_i = \prod_{j \neq i} \frac{-\lambda_j}{\lambda_i - \lambda_j}$ |
| Polynomial Extrapolation | $\langle O \rangle_0 = \sum_i \gamma_i \langle O \rangle_{\lambda_i}$ |
| Variance Amplification | $\Gamma = \sum_i \gamma_i^2$ |

### Main Takeaways

1. **ZNE trades classical computation for accuracy**: By running circuits at multiple noise levels and extrapolating, we estimate zero-noise results

2. **Noise scaling via folding**: Gate folding ($G \to GG^\dagger G$) provides discrete noise amplification without pulse-level control

3. **Richardson extrapolation**: Polynomial fitting with proper coefficients eliminates lower-order noise terms

4. **Variance-bias trade-off**: Higher-order extrapolation reduces bias but amplifies statistical noise

5. **Scale factor limits**: Maximum useful scale factor is approximately $1/\epsilon$ where $\epsilon$ is the gate error rate

## Daily Checklist

- [ ] I can implement gate folding to scale noise
- [ ] I can compute Richardson extrapolation coefficients
- [ ] I understand the variance-bias trade-off in ZNE
- [ ] I can choose appropriate scale factors for a given noise level
- [ ] I can implement ZNE using both custom code and Mitiq
- [ ] I understand when exponential vs polynomial extrapolation is appropriate

## Preview of Day 941

Tomorrow we explore **Probabilistic Error Cancellation (PEC)**, which provides unbiased error mitigation at the cost of sampling overhead:

- Quasi-probability decomposition of noisy channels
- Inverse noise channel representation
- Sampling overhead and cost analysis
- PEC implementation for Clifford+T circuits

PEC offers the strongest theoretical guarantees among error mitigation techniques but requires complete noise characterization.
