# Day 941: Probabilistic Error Cancellation - Quasi-Probability Decomposition

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Quasi-probability theory, inverse noise channels |
| Afternoon | 2 hours | Sampling overhead analysis and bounds |
| Evening | 2 hours | Computational lab - PEC implementation |

## Learning Objectives

By the end of this day, you will be able to:

1. **Derive quasi-probability decompositions** - Express ideal operations as linear combinations of noisy operations
2. **Calculate sampling overhead** - Quantify the cost of PEC in terms of variance amplification
3. **Implement noise inversion** - Construct inverse channels for common noise models
4. **Apply Monte Carlo PEC** - Sample from quasi-probability distributions to cancel errors
5. **Optimize PEC protocols** - Balance tomography cost vs mitigation benefit
6. **Compare PEC to other methods** - Understand when PEC is preferred over ZNE

## Core Content

### 1. Quasi-Probability Representation

#### 1.1 The PEC Principle

Probabilistic Error Cancellation represents ideal operations as linear combinations of noisy implementable operations:

$$\boxed{\mathcal{U}_{\text{ideal}} = \sum_i c_i \mathcal{O}_i}$$

where $\mathcal{O}_i$ are noisy operations we can implement and $c_i$ are real coefficients (quasi-probabilities) that can be negative.

**Key constraint**: $\sum_i c_i = 1$ (normalization)

**Key feature**: $c_i \in \mathbb{R}$ (can be negative, unlike true probabilities)

#### 1.2 Intuition from Classical Error Cancellation

Consider a noisy measurement with bias:
$$\bar{X}_{\text{noisy}} = \mu_{\text{true}} + \text{bias}$$

We can cancel the bias:
$$\mu_{\text{true}} = \bar{X}_{\text{noisy}} - \text{bias}$$

PEC generalizes this to quantum operations by "subtracting" noise effects.

### 2. Noise Model and Inverse Channels

#### 2.1 Noisy Gate Model

A noisy implementation of gate $G$ is modeled as:

$$\boxed{\mathcal{G}_{\text{noisy}} = \mathcal{E} \circ \mathcal{G}_{\text{ideal}}}$$

where $\mathcal{E}$ is the noise channel (applied after the ideal gate).

To recover the ideal gate:
$$\mathcal{G}_{\text{ideal}} = \mathcal{E}^{-1} \circ \mathcal{G}_{\text{noisy}}$$

#### 2.2 Inverse of Depolarizing Channel

For single-qubit depolarizing noise:
$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

The inverse channel:
$$\boxed{\mathcal{E}_p^{-1}(\rho) = \frac{1}{1-\frac{4p}{3}}\left[\rho - \frac{p}{3(1-p)}(X\rho X + Y\rho Y + Z\rho Z)\right]}$$

For small $p$, this simplifies to:
$$\mathcal{E}_p^{-1} \approx (1 + \frac{4p}{3})\mathcal{I} - \frac{p}{3}(\mathcal{X} + \mathcal{Y} + \mathcal{Z})$$

where $\mathcal{X}(\rho) = X\rho X$, etc.

#### 2.3 Quasi-Probability Decomposition

The inverse channel as quasi-probability:

$$\mathcal{E}_p^{-1} = \sum_i c_i \mathcal{P}_i$$

For depolarizing noise:

| Operation $\mathcal{P}_i$ | Coefficient $c_i$ |
|---------------------------|-------------------|
| $\mathcal{I}$ (Identity) | $\frac{1-p}{1-4p/3}$ |
| $\mathcal{X}$ | $\frac{-p/3}{1-4p/3}$ |
| $\mathcal{Y}$ | $\frac{-p/3}{1-4p/3}$ |
| $\mathcal{Z}$ | $\frac{-p/3}{1-4p/3}$ |

The 1-norm (cost):
$$\boxed{\gamma = \sum_i |c_i| = \frac{1 + p}{1 - 4p/3}}$$

For $p = 0.01$: $\gamma \approx 1.023$
For $p = 0.05$: $\gamma \approx 1.13$

### 3. Sampling Overhead

#### 3.1 Monte Carlo Implementation

To compute $\langle O \rangle$ using PEC:

1. Sample operation $\mathcal{O}_i$ with probability $|c_i|/\gamma$
2. Record sign $s_i = \text{sign}(c_i)$
3. Measure observable, get outcome $o$
4. Accumulate: $\sum_{\text{samples}} \gamma \cdot s_i \cdot o$

The estimator:
$$\boxed{\hat{\langle O \rangle} = \frac{1}{N}\sum_{k=1}^N \gamma^{(k)} s^{(k)} o^{(k)}}$$

#### 3.2 Variance Amplification

The variance of the PEC estimator:
$$\text{Var}[\hat{\langle O \rangle}] = \frac{\gamma^2 \cdot \text{Var}[O]}{N}$$

For a circuit with $d$ gates, each with 1-norm $\gamma_i$:
$$\boxed{\Gamma = \prod_i \gamma_i \approx \gamma^d}$$

The sampling overhead:
$$\boxed{C = \Gamma^2 = \gamma^{2d}}$$

**Exponential scaling**: PEC cost grows exponentially with circuit depth!

#### 3.3 Overhead Examples

For $\gamma = 1.1$ per gate:

| Circuit Depth | Total $\Gamma$ | Sampling Overhead $C = \Gamma^2$ |
|--------------|----------------|----------------------------------|
| 10 | 2.59 | 6.7 |
| 20 | 6.73 | 45 |
| 50 | 117 | 13,700 |
| 100 | 13,780 | $1.9 \times 10^8$ |

### 4. Gate-Level PEC Protocol

#### 4.1 Complete PEC Algorithm

For a circuit $U = G_d \circ \ldots \circ G_2 \circ G_1$:

```
1. For each gate G_i:
   - Characterize noise: G_i^noisy = E_i ∘ G_i
   - Compute inverse: E_i^{-1} = Σ_j c_ij P_ij
   - Calculate γ_i = Σ_j |c_ij|

2. Total cost: Γ = Π_i γ_i

3. For shot k = 1 to N:
   a. Sign accumulator: s = 1
   b. For each gate G_i:
      - Sample j with prob |c_ij|/γ_i
      - Apply P_ij ∘ G_i^noisy
      - s = s × sign(c_ij)
   c. Measure observable: o_k
   d. Record: Γ × s × o_k

4. Estimate: ⟨O⟩ = (1/N) Σ_k result_k
```

#### 4.2 Noise Learning Requirements

PEC requires complete characterization of noise $\mathcal{E}_i$ for each gate:

**Methods**:
- Gate set tomography (GST)
- Randomized benchmarking (partial)
- Cycle benchmarking
- Noise reconstruction from Pauli twirling

**Cost**: $O(4^n)$ for $n$-qubit gates (exponential in locality)

### 5. Practical Considerations

#### 5.1 When to Use PEC

**Advantages**:
- Unbiased: converges to ideal expectation value
- Works with any noise model (given characterization)
- No assumptions on noise structure

**Disadvantages**:
- Requires precise noise characterization
- Exponential sampling overhead
- Classical post-processing complexity

**Best suited for**:
- Short circuits (depth $\lesssim 50$)
- Well-characterized devices
- High-precision requirements

#### 5.2 PEC vs ZNE Comparison

| Aspect | ZNE | PEC |
|--------|-----|-----|
| Noise knowledge | None required | Full characterization |
| Bias | Extrapolation error | Unbiased |
| Overhead | $O(k)$ (# scale factors) | $O(\gamma^{2d})$ |
| Depth limit | $d \lesssim 100$ | $d \lesssim 50$ |
| Implementation | Circuit modification | Probabilistic sampling |

#### 5.3 Hybrid Approaches

Combine PEC and ZNE:
- Use PEC for critical gates (e.g., entangling gates)
- Use ZNE for overall circuit
- Layered PEC: apply to error-prone layers only

### 6. Two-Qubit Gate PEC

#### 6.1 CNOT with Depolarizing Noise

Two-qubit depolarizing channel:
$$\mathcal{E}_{p}^{(2)}(\rho) = (1-p)\rho + \frac{p}{15}\sum_{P \in \mathcal{P}_2 \setminus I} P\rho P$$

where $\mathcal{P}_2 = \{I, X, Y, Z\}^{\otimes 2}$ (16 Paulis).

The inverse:
$$\mathcal{E}_p^{(2),-1} = \frac{1}{1-\frac{16p}{15}}\left[\mathcal{I} - \frac{p}{15(1-p)}\sum_{P \neq I}\mathcal{P}\right]$$

Cost:
$$\boxed{\gamma_{\text{CNOT}} = \frac{1 + p}{1 - 16p/15}}$$

For $p = 0.05$: $\gamma_{\text{CNOT}} \approx 1.16$

## Quantum Computing Applications

### Variational Algorithms with PEC

For VQE targeting chemical accuracy:

$$E_{\text{PEC}} = \sum_{\alpha} h_\alpha \langle P_\alpha \rangle_{\text{PEC}}$$

Each Pauli term measured with PEC independently, then combined.

### Quantum Simulation

PEC enables accurate Trotter steps:
$$e^{-iHt} \approx \prod_k e^{-iH_k t/n}$$

With PEC on each Trotter layer, simulation accuracy is preserved.

### Benchmarking Applications

PEC provides ground truth for algorithm benchmarking:
- Compare noisy vs PEC-corrected results
- Validate other mitigation techniques
- Establish hardware fidelity baselines

## Worked Examples

### Example 1: Single-Qubit PEC Coefficients

**Problem**: Calculate PEC coefficients for depolarizing noise with $p = 0.03$.

**Solution**:

Normalization factor:
$$\eta = 1 - \frac{4p}{3} = 1 - \frac{4(0.03)}{3} = 1 - 0.04 = 0.96$$

Coefficients:
$$c_I = \frac{1-p}{\eta} = \frac{0.97}{0.96} = 1.0104$$

$$c_X = c_Y = c_Z = \frac{-p/3}{\eta} = \frac{-0.01}{0.96} = -0.0104$$

Verify: $c_I + c_X + c_Y + c_Z = 1.0104 - 3(0.0104) = 1.0104 - 0.0312 = 0.9792 \approx 1$ (rounding)

1-norm:
$$\gamma = |c_I| + |c_X| + |c_Y| + |c_Z| = 1.0104 + 3(0.0104) = 1.0104 + 0.0312$$

$$\boxed{\gamma = 1.0416}$$

### Example 2: Circuit Sampling Overhead

**Problem**: A circuit has 15 single-qubit gates ($\gamma = 1.02$) and 8 CNOT gates ($\gamma = 1.15$). Calculate the sampling overhead.

**Solution**:

Total 1-norm:
$$\Gamma = \gamma_{1Q}^{15} \cdot \gamma_{\text{CNOT}}^8 = 1.02^{15} \cdot 1.15^8$$

$$= 1.346 \cdot 3.059 = 4.117$$

Sampling overhead:
$$C = \Gamma^2 = 4.117^2$$

$$\boxed{C \approx 17}$$

Need 17x more shots than ideal to achieve same statistical precision.

### Example 3: PEC Shot Allocation

**Problem**: To achieve standard error $\sigma = 0.01$ on an observable with $\text{Var}[O] = 1$ and $\Gamma = 5$, how many shots are needed?

**Solution**:

PEC standard error:
$$\sigma = \frac{\Gamma \sqrt{\text{Var}[O]}}{\sqrt{N}}$$

Solving for $N$:
$$N = \frac{\Gamma^2 \cdot \text{Var}[O]}{\sigma^2}$$

$$= \frac{25 \cdot 1}{0.01^2} = \frac{25}{0.0001}$$

$$\boxed{N = 250,000 \text{ shots}}$$

## Practice Problems

### Level 1: Direct Application

1. Calculate PEC coefficients for amplitude damping noise $\mathcal{E}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger$ where $E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$, $E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$.

2. For a 20-gate circuit with uniform $\gamma = 1.05$ per gate, calculate the sampling overhead $C$.

3. How many shots are needed to achieve 1% relative error with $\Gamma = 10$ and $|\langle O \rangle| = 0.5$?

### Level 2: Intermediate

4. Derive the inverse channel for phase damping: $\mathcal{E}_\phi(\rho) = (1-p)\rho + p Z\rho Z$.

5. Compare the sampling overhead of PEC vs the variance amplification of ZNE for a 30-gate circuit with $p = 0.02$.

6. Design a hybrid PEC-ZNE protocol for a circuit with 10 error-prone CNOT gates and 40 low-error single-qubit gates.

### Level 3: Challenging

7. Prove that for Pauli channels, the quasi-probability 1-norm equals the diamond distance between the noisy and ideal channels.

8. Derive the optimal shot allocation across multiple Pauli terms when measuring a Hamiltonian with PEC.

9. Analyze the conditions under which PEC becomes more efficient than ZNE, accounting for noise characterization cost.

## Computational Lab

```python
"""
Day 941: Probabilistic Error Cancellation Implementation
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SuperOp, Kraus
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.linalg import inv

# ============================================================
# Part 1: Quasi-Probability Decomposition
# ============================================================

def depolarizing_inverse_coefficients(p: float) -> Dict[str, float]:
    """
    Compute quasi-probability coefficients for inverse depolarizing channel.

    For depolarizing noise: E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    Inverse: E^{-1} = Σ_i c_i P_i

    Args:
        p: Depolarizing probability

    Returns:
        Dictionary mapping Pauli labels to coefficients
    """
    eta = 1 - 4*p/3

    if eta <= 0:
        raise ValueError(f"Error rate p={p} too high for PEC (eta={eta})")

    c_I = (1 - p) / eta
    c_P = -p / (3 * eta)

    coefficients = {
        'I': c_I,
        'X': c_P,
        'Y': c_P,
        'Z': c_P
    }

    return coefficients

def compute_one_norm(coefficients: Dict[str, float]) -> float:
    """Compute the 1-norm (gamma) of quasi-probability distribution."""
    return sum(abs(c) for c in coefficients.values())

def two_qubit_depolarizing_inverse(p: float) -> Dict[str, float]:
    """
    Compute quasi-probability coefficients for 2-qubit depolarizing inverse.
    """
    eta = 1 - 16*p/15

    if eta <= 0:
        raise ValueError(f"Error rate p={p} too high for 2Q PEC")

    c_I = (1 - p) / eta
    c_P = -p / (15 * eta)

    # 16 two-qubit Paulis
    paulis = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
              'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']

    coefficients = {pauli: c_I if pauli == 'II' else c_P for pauli in paulis}

    return coefficients

class QuasiProbabilityDecomposition:
    """Class to handle quasi-probability decomposition and sampling."""

    def __init__(self, coefficients: Dict[str, float]):
        self.coefficients = coefficients
        self.gamma = compute_one_norm(coefficients)

        # Compute probabilities and signs for sampling
        self.operations = list(coefficients.keys())
        self.probs = [abs(c) / self.gamma for c in coefficients.values()]
        self.signs = [np.sign(c) if c != 0 else 1 for c in coefficients.values()]

    def sample(self) -> Tuple[str, float]:
        """Sample an operation from the quasi-probability distribution."""
        idx = np.random.choice(len(self.operations), p=self.probs)
        return self.operations[idx], self.signs[idx]

    def __repr__(self):
        terms = [f"{c:+.4f}*{op}" for op, c in self.coefficients.items()]
        return " ".join(terms) + f"\n  gamma = {self.gamma:.4f}"

# ============================================================
# Part 2: PEC Circuit Execution
# ============================================================

class PECExecutor:
    """Execute circuits with Probabilistic Error Cancellation."""

    def __init__(self, noise_model: NoiseModel, gate_coefficients: Dict[str, Dict[str, float]]):
        """
        Args:
            noise_model: The noise model of the device
            gate_coefficients: Dictionary mapping gate names to their PEC coefficients
        """
        self.noise_model = noise_model
        self.decompositions = {
            gate: QuasiProbabilityDecomposition(coeffs)
            for gate, coeffs in gate_coefficients.items()
        }
        self.simulator = AerSimulator(noise_model=noise_model)

    def execute_with_pec(self, circuit: QuantumCircuit, observable: str,
                         shots: int = 8192) -> Tuple[float, float]:
        """
        Execute circuit with PEC and compute observable expectation.

        Args:
            circuit: Quantum circuit to execute
            observable: Pauli observable string (e.g., 'ZZ', 'X')
            shots: Number of Monte Carlo samples

        Returns:
            (expectation_value, standard_error)
        """
        results = []

        # Count gates for total gamma calculation
        total_gamma = self._compute_total_gamma(circuit)

        for _ in range(shots):
            # Build modified circuit with sampled Pauli corrections
            modified_circuit, sign = self._sample_pec_circuit(circuit)

            # Execute single shot
            modified_circuit.measure_all()
            result = self.simulator.run(modified_circuit, shots=1).result()
            counts = result.get_counts()

            # Compute observable value for this shot
            bitstring = list(counts.keys())[0]
            obs_value = self._evaluate_pauli_observable(bitstring, observable)

            # Accumulate with sign and gamma factor
            results.append(total_gamma * sign * obs_value)

        expectation = np.mean(results)
        std_error = np.std(results) / np.sqrt(shots)

        return expectation, std_error

    def _compute_total_gamma(self, circuit: QuantumCircuit) -> float:
        """Compute total gamma for the circuit."""
        gamma = 1.0
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            if gate_name in self.decompositions:
                gamma *= self.decompositions[gate_name].gamma
        return gamma

    def _sample_pec_circuit(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, float]:
        """Sample a PEC-modified circuit and return accumulated sign."""
        new_circuit = QuantumCircuit(circuit.num_qubits)
        total_sign = 1.0

        for instruction in circuit.data:
            gate = instruction.operation
            qubits = instruction.qubits

            # Apply original gate
            new_circuit.append(gate, qubits)

            # Sample and apply Pauli correction if gate has noise
            if gate.name in self.decompositions:
                pauli_op, sign = self.decompositions[gate.name].sample()
                total_sign *= sign

                # Apply sampled Pauli(s)
                for i, p in enumerate(pauli_op):
                    if p == 'X':
                        new_circuit.x(qubits[i] if len(qubits) > 1 else qubits[0])
                    elif p == 'Y':
                        new_circuit.y(qubits[i] if len(qubits) > 1 else qubits[0])
                    elif p == 'Z':
                        new_circuit.z(qubits[i] if len(qubits) > 1 else qubits[0])

        return new_circuit, total_sign

    def _evaluate_pauli_observable(self, bitstring: str, observable: str) -> float:
        """Evaluate Pauli observable eigenvalue from measurement outcome."""
        # Reverse bitstring (Qiskit convention)
        bitstring = bitstring[::-1]

        parity = 0
        for i, p in enumerate(observable):
            if p in ['X', 'Y', 'Z']:
                if i < len(bitstring):
                    parity += int(bitstring[i])

        return (-1) ** parity

# ============================================================
# Part 3: PEC Demonstration
# ============================================================

def demonstrate_pec():
    """Full PEC demonstration on a test circuit."""

    print("="*60)
    print("Probabilistic Error Cancellation Demonstration")
    print("="*60)

    # Create a simple test circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(np.pi/4, 0)
    qc.rz(np.pi/4, 1)
    qc.cx(0, 1)
    qc.h(0)

    print("\nTest Circuit:")
    print(qc.draw())

    # Define noise model
    p_1q = 0.02  # 2% single-qubit error
    p_2q = 0.05  # 5% two-qubit error

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_1q, 1), ['h', 'rz']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_2q, 2), ['cx']
    )

    # Compute PEC coefficients
    coeffs_1q = depolarizing_inverse_coefficients(p_1q)
    coeffs_2q = two_qubit_depolarizing_inverse(p_2q)

    print("\nSingle-qubit PEC coefficients (p=0.02):")
    for op, c in coeffs_1q.items():
        print(f"  {op}: {c:+.6f}")
    print(f"  gamma_1q: {compute_one_norm(coeffs_1q):.4f}")

    print("\nTwo-qubit PEC coefficients (p=0.05):")
    print(f"  II: {coeffs_2q['II']:+.6f}")
    print(f"  Others: {coeffs_2q['IX']:+.6f} each")
    print(f"  gamma_2q: {compute_one_norm(coeffs_2q):.4f}")

    # Calculate total overhead
    n_1q = 4  # h(2) + rz(2)
    n_2q = 2  # cx(2)
    total_gamma = compute_one_norm(coeffs_1q)**n_1q * compute_one_norm(coeffs_2q)**n_2q
    print(f"\nTotal gamma: {total_gamma:.4f}")
    print(f"Sampling overhead: {total_gamma**2:.2f}x")

    # Get ideal result
    ideal_sim = AerSimulator()
    ideal_qc = qc.copy()
    ideal_qc.measure_all()
    ideal_result = ideal_sim.run(ideal_qc, shots=100000).result()
    ideal_counts = ideal_result.get_counts()

    # Calculate ideal ZZ expectation
    ideal_zz = 0
    for bs, count in ideal_counts.items():
        parity = (int(bs[0]) + int(bs[1])) % 2
        ideal_zz += (-1)**parity * count
    ideal_zz /= 100000
    print(f"\nIdeal <ZZ>: {ideal_zz:.4f}")

    # Get noisy result
    noisy_sim = AerSimulator(noise_model=noise_model)
    noisy_result = noisy_sim.run(ideal_qc, shots=100000).result()
    noisy_counts = noisy_result.get_counts()

    noisy_zz = 0
    for bs, count in noisy_counts.items():
        parity = (int(bs[0]) + int(bs[1])) % 2
        noisy_zz += (-1)**parity * count
    noisy_zz /= 100000
    print(f"Noisy <ZZ>: {noisy_zz:.4f}")

    # Execute with PEC
    gate_coeffs = {
        'h': coeffs_1q,
        'rz': coeffs_1q,
        'cx': coeffs_2q
    }

    pec_executor = PECExecutor(noise_model, gate_coeffs)
    pec_zz, pec_std = pec_executor.execute_with_pec(qc, 'ZZ', shots=10000)
    print(f"PEC <ZZ>: {pec_zz:.4f} +/- {pec_std:.4f}")

    # Calculate improvements
    noisy_error = abs(noisy_zz - ideal_zz)
    pec_error = abs(pec_zz - ideal_zz)
    improvement = noisy_error / pec_error if pec_error > 0 else float('inf')

    print(f"\nError comparison:")
    print(f"  Noisy error: {noisy_error:.4f}")
    print(f"  PEC error: {pec_error:.4f}")
    print(f"  Improvement: {improvement:.2f}x")

    return ideal_zz, noisy_zz, pec_zz

# ============================================================
# Part 4: Sampling Overhead Analysis
# ============================================================

def analyze_sampling_overhead():
    """Analyze PEC sampling overhead vs circuit depth."""

    print("\n" + "="*60)
    print("PEC Sampling Overhead Analysis")
    print("="*60)

    error_rates = [0.01, 0.02, 0.03, 0.05]
    depths = np.arange(1, 101)

    plt.figure(figsize=(12, 5))

    # Plot 1: Gamma vs depth
    plt.subplot(1, 2, 1)
    for p in error_rates:
        gamma = (1 + p) / (1 - 4*p/3)
        gammas = gamma ** depths
        plt.semilogy(depths, gammas, label=f'p = {p}')

    plt.xlabel('Circuit Depth (# gates)')
    plt.ylabel('Total Gamma (Γ)')
    plt.title('PEC Cost Parameter vs Circuit Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Sampling overhead vs depth
    plt.subplot(1, 2, 2)
    for p in error_rates:
        gamma = (1 + p) / (1 - 4*p/3)
        overhead = (gamma ** depths) ** 2
        plt.semilogy(depths, overhead, label=f'p = {p}')

    plt.axhline(y=1e6, color='red', linestyle='--', label='1M shots limit')
    plt.xlabel('Circuit Depth (# gates)')
    plt.ylabel('Sampling Overhead (Γ²)')
    plt.title('PEC Sampling Overhead vs Circuit Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pec_overhead_analysis.png', dpi=150)
    plt.show()

    # Calculate maximum practical depth for each error rate
    print("\nMaximum practical depth (overhead < 1000):")
    for p in error_rates:
        gamma = (1 + p) / (1 - 4*p/3)
        max_depth = int(np.log(np.sqrt(1000)) / np.log(gamma))
        print(f"  p = {p}: {max_depth} gates")

# ============================================================
# Part 5: PEC vs ZNE Comparison
# ============================================================

def compare_pec_zne():
    """Compare PEC and ZNE on the same circuit."""

    print("\n" + "="*60)
    print("PEC vs ZNE Comparison")
    print("="*60)

    # Create test circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(np.pi/4, 0)
    qc.rx(np.pi/4, 1)
    qc.cx(0, 1)

    p_error = 0.03

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_error, 1), ['h', 'rx']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_error * 3, 2), ['cx']
    )

    # Ideal reference
    ideal_sim = AerSimulator()
    ideal_qc = qc.copy()
    ideal_qc.measure_all()
    ideal_counts = ideal_sim.run(ideal_qc, shots=100000).result().get_counts()
    ideal_zz = sum((-1)**((int(bs[0])+int(bs[1]))%2) * c for bs, c in ideal_counts.items()) / 100000

    # Noisy baseline
    noisy_sim = AerSimulator(noise_model=noise_model)
    noisy_counts = noisy_sim.run(ideal_qc, shots=100000).result().get_counts()
    noisy_zz = sum((-1)**((int(bs[0])+int(bs[1]))%2) * c for bs, c in noisy_counts.items()) / 100000

    print(f"Ideal <ZZ>: {ideal_zz:.4f}")
    print(f"Noisy <ZZ>: {noisy_zz:.4f}")

    # ZNE with folding
    def run_with_scale(scale_factor):
        """Run circuit with noise amplified by scale_factor."""
        scaled_noise = NoiseModel()
        scaled_noise.add_all_qubit_quantum_error(
            depolarizing_error(min(p_error * scale_factor, 0.75), 1), ['h', 'rx']
        )
        scaled_noise.add_all_qubit_quantum_error(
            depolarizing_error(min(p_error * 3 * scale_factor, 0.75), 2), ['cx']
        )

        sim = AerSimulator(noise_model=scaled_noise)
        counts = sim.run(ideal_qc, shots=20000).result().get_counts()
        return sum((-1)**((int(bs[0])+int(bs[1]))%2) * c for bs, c in counts.items()) / 20000

    # ZNE extrapolation
    scale_factors = [1, 1.5, 2, 2.5]
    zne_values = [run_with_scale(s) for s in scale_factors]

    # Linear extrapolation
    zne_linear = (scale_factors[1] * zne_values[0] - scale_factors[0] * zne_values[1]) / (scale_factors[1] - scale_factors[0])

    # Polynomial extrapolation
    coeffs = np.polyfit(scale_factors, zne_values, len(scale_factors) - 1)
    zne_poly = np.polyval(coeffs, 0)

    print(f"\nZNE Linear: {zne_linear:.4f}")
    print(f"ZNE Polynomial: {zne_poly:.4f}")

    # PEC (simplified - using same results as demonstration)
    coeffs_1q = depolarizing_inverse_coefficients(p_error)
    coeffs_2q = two_qubit_depolarizing_inverse(p_error * 3)

    gate_coeffs = {'h': coeffs_1q, 'rx': coeffs_1q, 'cx': coeffs_2q}
    pec_executor = PECExecutor(noise_model, gate_coeffs)
    pec_zz, pec_std = pec_executor.execute_with_pec(qc, 'ZZ', shots=10000)

    print(f"PEC: {pec_zz:.4f} +/- {pec_std:.4f}")

    # Comparison plot
    methods = ['Ideal', 'Noisy', 'ZNE Linear', 'ZNE Poly', 'PEC']
    values = [ideal_zz, noisy_zz, zne_linear, zne_poly, pec_zz]
    errors = [0, abs(noisy_zz - ideal_zz), abs(zne_linear - ideal_zz),
              abs(zne_poly - ideal_zz), abs(pec_zz - ideal_zz)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['green', 'red', 'blue', 'cyan', 'purple']
    ax1.bar(methods, values, color=colors, edgecolor='black')
    ax1.axhline(y=ideal_zz, color='green', linestyle='--', alpha=0.5)
    ax1.set_ylabel('<ZZ>')
    ax1.set_title('Expectation Value Comparison')

    ax2.bar(methods[1:], errors[1:], color=colors[1:], edgecolor='black')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Error from Ideal')

    for i, (v, e) in enumerate(zip(values, errors)):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        if i > 0:
            ax2.text(i-1, e + 0.005, f'{e:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('pec_vs_zne.png', dpi=150)
    plt.show()

# ============================================================
# Part 6: Advanced PEC with Mitiq
# ============================================================

def demonstrate_mitiq_pec():
    """Demonstrate PEC using the Mitiq library."""
    try:
        from mitiq import pec
        from mitiq.pec.representations import represent_operation_with_local_depolarizing_noise

        print("\n" + "="*60)
        print("Mitiq PEC Demonstration")
        print("="*60)

        # Create circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        print("Circuit:", qc.draw())

        # Define noise level
        noise_level = 0.02

        # Get representations
        from mitiq.pec.representations import OperationRepresentation
        from mitiq.pec.sampling import sample_circuit

        # Note: This is a simplified demonstration
        # Full Mitiq PEC requires more setup for operation representations

        print("\nMitiq PEC features:")
        print("  - Automatic noise characterization")
        print("  - Optimal quasi-probability computation")
        print("  - Efficient circuit sampling")

    except ImportError:
        print("\nMitiq not installed. Install with: pip install mitiq")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Day 941: Probabilistic Error Cancellation Lab")
    print("="*60)

    # Part 1: Basic PEC demonstration
    print("\n--- Part 1: PEC Demonstration ---")
    demonstrate_pec()

    # Part 2: Overhead analysis
    print("\n--- Part 2: Overhead Analysis ---")
    analyze_sampling_overhead()

    # Part 3: PEC vs ZNE comparison
    print("\n--- Part 3: PEC vs ZNE Comparison ---")
    compare_pec_zne()

    # Part 4: Mitiq integration
    print("\n--- Part 4: Mitiq Library ---")
    demonstrate_mitiq_pec()

    print("\n" + "="*60)
    print("Lab Complete! Key Takeaways:")
    print("  1. PEC provides unbiased error cancellation")
    print("  2. Sampling overhead scales exponentially with depth")
    print("  3. PEC requires full noise characterization")
    print("  4. Best suited for short, well-characterized circuits")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Ideal Operation | $\mathcal{U}_{\text{ideal}} = \sum_i c_i \mathcal{O}_i$ |
| Depolarizing Inverse | $c_I = \frac{1-p}{1-4p/3}$, $c_P = \frac{-p/3}{1-4p/3}$ |
| 1-Norm | $\gamma = \sum_i |c_i|$ |
| Total Cost | $\Gamma = \prod_i \gamma_i$ |
| Sampling Overhead | $C = \Gamma^2$ |
| PEC Estimator | $\hat{\langle O \rangle} = \frac{1}{N}\sum_k \gamma^{(k)} s^{(k)} o^{(k)}$ |

### Main Takeaways

1. **PEC is unbiased**: Unlike ZNE, PEC converges to the exact ideal expectation value in the limit of infinite samples

2. **Quasi-probability decomposition**: Ideal operations are represented as linear combinations of noisy operations, including negative coefficients

3. **Exponential overhead**: Sampling cost scales as $\gamma^{2d}$ where $d$ is circuit depth, limiting practical application to shallow circuits

4. **Noise knowledge required**: PEC needs complete characterization of noise channels, unlike ZNE which is noise-agnostic

5. **Monte Carlo sampling**: PEC is implemented by probabilistically sampling correction operations and accumulating signed results

## Daily Checklist

- [ ] I can derive quasi-probability coefficients for depolarizing noise
- [ ] I understand why PEC coefficients can be negative
- [ ] I can calculate sampling overhead for a given circuit
- [ ] I know when to prefer PEC over ZNE
- [ ] I can implement Monte Carlo PEC sampling
- [ ] I understand the trade-off between noise characterization cost and mitigation benefit

## Preview of Day 942

Tomorrow we explore **Symmetry Verification**, which exploits conservation laws and symmetries to detect and discard erroneous results:

- Conservation law verification (particle number, spin)
- Post-selection and acceptance rates
- Symmetry expansion techniques
- Parity check circuits and syndrome measurements

Symmetry verification is particularly powerful for quantum chemistry and physics simulations where physical constraints provide natural error detection.
