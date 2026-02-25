# Day 945: Virtual Distillation - Exponential Error Suppression

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Virtual distillation theory, purity amplification |
| Afternoon | 2 hours | Sample complexity and practical limitations |
| Evening | 2 hours | Computational lab - virtual distillation implementation |

## Learning Objectives

By the end of this day, you will be able to:

1. **Explain virtual distillation** - Understand how multiple noisy copies simulate a purer state
2. **Derive the purification formula** - Compute $\text{Tr}(\rho^n O) / \text{Tr}(\rho^n)$ from SWAP tests
3. **Analyze error suppression** - Quantify exponential improvement with copy number
4. **Calculate sample complexity** - Determine overhead in terms of purity
5. **Implement practical protocols** - Design efficient circuits for virtual distillation
6. **Assess applicability** - Understand when virtual distillation is beneficial

## Core Content

### 1. Virtual Distillation Fundamentals

#### 1.1 The Purification Concept

A noisy quantum state $\rho$ can be written as:
$$\rho = (1-\epsilon)|\psi\rangle\langle\psi| + \epsilon \sigma$$

where $|\psi\rangle$ is the ideal pure state, $\sigma$ is the error state, and $\epsilon$ is the error probability.

**Virtual distillation** computes expectation values as if measured on a purer state, without physically purifying $\rho$.

#### 1.2 The Key Formula

For $n$ copies of $\rho$, the virtually distilled expectation value is:

$$\boxed{\langle O \rangle_{\text{distilled}} = \frac{\text{Tr}(\rho^n O)}{\text{Tr}(\rho^n)}}$$

This converges to the pure state expectation as $n \to \infty$:
$$\lim_{n \to \infty} \frac{\text{Tr}(\rho^n O)}{\text{Tr}(\rho^n)} = \langle \psi | O | \psi \rangle$$

#### 1.3 Intuition: Purity Amplification

For a mixed state with eigendecomposition:
$$\rho = \sum_k \lambda_k |k\rangle\langle k|, \quad \sum_k \lambda_k = 1$$

The $n$-th power is:
$$\rho^n = \sum_k \lambda_k^n |k\rangle\langle k|$$

**Key insight**: $\lambda_k^n$ amplifies the largest eigenvalue exponentially.

If $\lambda_{\max} = 1 - \epsilon$ (nearly pure):
$$\frac{\lambda_{\max}^n}{\sum_k \lambda_k^n} \to 1 \quad \text{as } n \to \infty$$

### 2. Error Suppression Analysis

#### 2.1 Two-Copy Case ($n = 2$)

For $\rho = (1-\epsilon)|\psi\rangle\langle\psi| + \epsilon \sigma$ with $\sigma \perp |\psi\rangle$:

$$\text{Tr}(\rho^2) = (1-\epsilon)^2 + \epsilon^2 \text{Tr}(\sigma^2)$$

For maximally mixed $\sigma$: $\text{Tr}(\sigma^2) = 1/d$ where $d$ is dimension.

The distilled expectation:
$$\langle O \rangle_{\text{distilled}} = \frac{(1-\epsilon)^2 \langle\psi|O|\psi\rangle + O(\epsilon^2)}{(1-\epsilon)^2 + O(\epsilon^2)}$$

$$\boxed{\langle O \rangle_{\text{distilled}} = \langle\psi|O|\psi\rangle + O(\epsilon^2)}$$

**Error reduced from $O(\epsilon)$ to $O(\epsilon^2)$!**

#### 2.2 General $n$-Copy Case

With $n$ copies:
$$\boxed{\text{Bias} = O(\epsilon^n)}$$

The error suppression is **exponential** in the number of copies.

#### 2.3 Purity Requirements

For useful distillation, need $\text{Tr}(\rho^n) > 0$:
- Purity: $\text{Tr}(\rho^2) = \gamma$
- For $n$ copies: $\text{Tr}(\rho^n) \approx \gamma^{(n-1)}$

When $\gamma \to 0$ (highly mixed), distillation fails.

### 3. Implementation via SWAP Test

#### 3.1 SWAP Test Circuit

The SWAP test measures overlap between two quantum states:

```
|0⟩ ──H──●──H──M
         │
|ψ⟩ ─────SWAP───
|φ⟩ ─────────────
```

Probability of measuring $|0\rangle$:
$$P(0) = \frac{1 + |\langle\psi|\phi\rangle|^2}{2}$$

For identical states $\rho \otimes \rho$:
$$P(0) = \frac{1 + \text{Tr}(\rho^2)}{2}$$

#### 3.2 Destructive SWAP Test

For computing $\text{Tr}(\rho^2 O)$:

1. Prepare two copies: $\rho \otimes \rho$
2. Apply controlled-SWAP with ancilla
3. Measure $O$ on the first copy
4. Post-select on ancilla outcome

**Result**: Samples from $\rho^2 O / \text{Tr}(\rho^2)$.

#### 3.3 General $n$-Copy Protocol

For $n$ copies:
1. Prepare $n$ identical copies: $\rho^{\otimes n}$
2. Apply cyclic permutation test
3. Measure observable on first copy
4. Combine results to estimate $\text{Tr}(\rho^n O) / \text{Tr}(\rho^n)$

### 4. Sample Complexity

#### 4.1 Variance Analysis

The variance of the distilled estimator:

$$\boxed{\text{Var}[\hat{O}_{\text{distilled}}] = \frac{\|O\|^2}{\text{Tr}(\rho^n)^2 \cdot N}}$$

where $N$ is the number of measurement shots.

#### 4.2 Required Samples

To achieve precision $\epsilon$:
$$N \geq \frac{\|O\|^2}{\epsilon^2 \cdot \text{Tr}(\rho^n)^2}$$

For purity $\gamma = \text{Tr}(\rho^2)$:
$$\boxed{N = O\left(\frac{1}{\gamma^{2(n-1)}}\right)}$$

**Trade-off**: More copies suppress bias but increase variance.

#### 4.3 Optimal Copy Number

Minimize total error (bias + variance):
- Bias: $O(\epsilon^n)$
- Variance: $O(1/(\gamma^{2(n-1)} N))$

Optimal $n$ depends on:
- State purity $\gamma$
- Available shots $N$
- Target precision

### 5. Practical Protocols

#### 5.1 Echo Verification Protocol

Alternative to SWAP test using time reversal:

1. Prepare $|\psi\rangle$ via circuit $U$
2. Apply $U^\dagger$ (inverse circuit)
3. Measure in computational basis
4. Post-select on $|0\rangle^{\otimes n}$

This effectively measures $|\langle 0|U^\dagger \rho U|0\rangle|^2 = |\langle\psi|\rho|\psi\rangle|^2$.

#### 5.2 Randomized Measurements

Use randomized Pauli measurements:

1. Apply random Clifford $C$
2. Measure in computational basis
3. Classically compute shadows
4. Estimate $\text{Tr}(\rho^n O)$ from shadows

**Advantage**: Avoids controlled-SWAP gates.

#### 5.3 Incremental Distillation

Build up from $n=2$:

$$\rho_2 = \frac{\rho^2}{\text{Tr}(\rho^2)}, \quad \rho_4 = \frac{\rho_2^2}{\text{Tr}(\rho_2^2)}, \ldots$$

Each step squares the effective purity.

### 6. Limitations and Considerations

#### 6.1 State Preparation

Need $n$ independent copies of $\rho$:
- Same circuit must be run $n$ times
- Noise must be statistically independent
- Correlated errors break the protocol

#### 6.2 Connectivity Requirements

SWAP tests require:
- All-to-all connectivity (or SWAP routing)
- High-fidelity SWAP gates
- Additional ancilla qubits

#### 6.3 When to Use Virtual Distillation

**Good candidates**:
- High purity states ($\gamma > 0.5$)
- Small number of qubits (manageable SWAP tests)
- High-precision requirements

**Poor candidates**:
- Low purity states
- Large systems
- Limited connectivity

### 7. Comparison with Other Techniques

| Technique | Error Reduction | Overhead | Noise Knowledge |
|-----------|-----------------|----------|-----------------|
| ZNE | Polynomial | Linear in scale factors | None |
| PEC | Complete (unbiased) | Exponential in depth | Full |
| Symmetry | Errors violating symmetry | Post-selection discard | Symmetry only |
| Virtual Distillation | Exponential in copies | Exponential in 1/purity | State purity |

**Complementary use**: Virtual distillation can be combined with ZNE or measurement mitigation.

## Quantum Computing Applications

### Variational Quantum Eigensolver (VQE)

Virtual distillation improves ground state energy estimates:
$$E_{\text{distilled}} = \frac{\text{Tr}(\rho^2 H)}{\text{Tr}(\rho^2)}$$

For near-ground states, this approaches the true ground state energy.

### Quantum Simulation

Time-evolved states under noisy dynamics:
- Error accumulates with evolution time
- Virtual distillation recovers ideal evolution (to some degree)

### Quantum Machine Learning

Kernel methods and inner products:
- $\text{Tr}(\rho_1 \rho_2)$ computed via SWAP test
- Virtual distillation improves kernel estimation

## Worked Examples

### Example 1: Two-Copy Error Suppression

**Problem**: A noisy state has fidelity $F = 0.9$ with the ideal state. What is the effective fidelity after two-copy virtual distillation?

**Solution**:

Model: $\rho = 0.9|\psi\rangle\langle\psi| + 0.1 |e\rangle\langle e|$ where $\langle\psi|e\rangle = 0$.

Purity:
$$\text{Tr}(\rho^2) = 0.9^2 + 0.1^2 = 0.81 + 0.01 = 0.82$$

Distilled state (effective):
$$\rho_{\text{eff}} = \frac{\rho^2}{\text{Tr}(\rho^2)} = \frac{0.81|\psi\rangle\langle\psi| + 0.01|e\rangle\langle e|}{0.82}$$

Effective fidelity:
$$F_{\text{distilled}} = \frac{0.81}{0.82}$$

$$\boxed{F_{\text{distilled}} \approx 0.988}$$

Error reduced from 10% to ~1.2%.

### Example 2: Sample Complexity

**Problem**: With purity $\gamma = 0.7$, how many samples are needed to achieve 1% precision using $n=2$ copies?

**Solution**:

For $n=2$ copies:
$$N \geq \frac{\|O\|^2}{\epsilon^2 \cdot \text{Tr}(\rho^2)^2}$$

Assuming $\|O\| = 1$ (Pauli observable) and $\text{Tr}(\rho^2) = \gamma = 0.7$:

$$N \geq \frac{1}{(0.01)^2 \cdot (0.7)^2} = \frac{1}{0.0001 \cdot 0.49}$$

$$N \geq \frac{1}{0.000049} \approx 20,400$$

$$\boxed{N \approx 20,000 \text{ samples}}$$

### Example 3: Optimal Copy Number

**Problem**: For a state with $\epsilon = 0.15$ error and purity $\gamma = 0.8$, what is the optimal number of copies given $N = 10^6$ shots?

**Solution**:

Bias with $n$ copies: $b_n = \epsilon^n = 0.15^n$
Standard deviation: $\sigma_n = 1/(\gamma^{n-1} \sqrt{N})$

Total error: $E_n = b_n + \sigma_n$

For $n = 2$:
$$E_2 = 0.15^2 + \frac{1}{0.8 \cdot 1000} = 0.0225 + 0.00125 = 0.024$$

For $n = 3$:
$$E_3 = 0.15^3 + \frac{1}{0.8^2 \cdot 1000} = 0.0034 + 0.00156 = 0.005$$

For $n = 4$:
$$E_4 = 0.15^4 + \frac{1}{0.8^3 \cdot 1000} = 0.0005 + 0.00195 = 0.0025$$

For $n = 5$:
$$E_5 = 0.15^5 + \frac{1}{0.8^4 \cdot 1000} = 0.00008 + 0.00244 = 0.0025$$

$$\boxed{n_{\text{opt}} = 4 \text{ or } 5}$$

## Practice Problems

### Level 1: Direct Application

1. For a state with $\rho = 0.95|0\rangle\langle 0| + 0.05|1\rangle\langle 1|$, calculate $\text{Tr}(\rho^2)$ and the distilled $\langle Z \rangle$.

2. If purity is $\gamma = 0.6$ and we need $\text{Tr}(\rho^n) > 0.01$ for reliable estimation, what is the maximum useful $n$?

3. Calculate the sample overhead for $n=3$ copy distillation with $\gamma = 0.75$.

### Level 2: Intermediate

4. Derive the formula $P(0) = (1 + \text{Tr}(\rho^2))/2$ for the SWAP test with state $\rho \otimes \rho$.

5. For a depolarizing channel with error rate $p$ on $k$ qubits, what is the purity $\text{Tr}(\rho^2)$?

6. Design a protocol to estimate $\text{Tr}(\rho^3)$ using only two copies at a time.

### Level 3: Challenging

7. Prove that virtual distillation with $n$ copies suppresses errors to $O(\epsilon^n)$ for the general case of non-orthogonal error states.

8. Derive the optimal $n$ that minimizes total error (bias + variance) as a function of $\epsilon$, $\gamma$, and $N$.

9. Analyze how correlated noise across copies affects virtual distillation performance.

## Computational Lab

```python
"""
Day 945: Virtual Distillation Implementation
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.linalg import sqrtm

# ============================================================
# Part 1: Theoretical Foundation
# ============================================================

def create_noisy_state(ideal_state: np.ndarray, epsilon: float) -> np.ndarray:
    """Create a noisy mixed state from ideal pure state."""
    d = len(ideal_state)

    # Ideal density matrix
    rho_ideal = np.outer(ideal_state, ideal_state.conj())

    # Maximally mixed state
    rho_mixed = np.eye(d) / d

    # Noisy state
    rho_noisy = (1 - epsilon) * rho_ideal + epsilon * rho_mixed

    return rho_noisy

def compute_purity(rho: np.ndarray) -> float:
    """Compute purity Tr(ρ²)."""
    return np.real(np.trace(rho @ rho))

def virtual_distillation_exact(rho: np.ndarray, O: np.ndarray, n: int) -> float:
    """Compute exactly distilled expectation Tr(ρⁿO)/Tr(ρⁿ)."""
    rho_n = np.linalg.matrix_power(rho, n)
    numerator = np.trace(rho_n @ O)
    denominator = np.trace(rho_n)

    return np.real(numerator / denominator)

def analyze_distillation_theory():
    """Analyze virtual distillation error suppression theoretically."""

    print("="*60)
    print("Virtual Distillation: Theoretical Analysis")
    print("="*60)

    # Create ideal state
    ideal_state = np.array([1, 0], dtype=complex)  # |0⟩

    # Observable
    Z = np.array([[1, 0], [0, -1]])
    ideal_expectation = np.real(ideal_state.conj() @ Z @ ideal_state)

    print(f"\nIdeal state: |0⟩")
    print(f"Observable: Z")
    print(f"Ideal <Z>: {ideal_expectation}")

    # Test different error rates
    error_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    copy_numbers = [1, 2, 3, 4, 5]

    results = np.zeros((len(error_rates), len(copy_numbers)))

    for i, epsilon in enumerate(error_rates):
        rho = create_noisy_state(ideal_state, epsilon)
        purity = compute_purity(rho)

        for j, n in enumerate(copy_numbers):
            distilled = virtual_distillation_exact(rho, Z, n)
            error = abs(distilled - ideal_expectation)
            results[i, j] = error

    # Plot error suppression
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i, epsilon in enumerate(error_rates):
        plt.semilogy(copy_numbers, results[i, :], 'o-', label=f'ε = {epsilon}')

    plt.xlabel('Number of Copies (n)')
    plt.ylabel('|Error|')
    plt.title('Error Suppression with Copy Number')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot error vs epsilon for fixed n
    plt.subplot(1, 2, 2)
    for j, n in enumerate(copy_numbers):
        plt.loglog(error_rates, results[:, j], 's-', label=f'n = {n}')

    # Add reference lines
    plt.loglog(error_rates, error_rates, 'k--', alpha=0.5, label='O(ε)')
    plt.loglog(error_rates, np.array(error_rates)**2, 'k:', alpha=0.5, label='O(ε²)')

    plt.xlabel('Error Rate (ε)')
    plt.ylabel('|Error|')
    plt.title('Error Scaling with ε')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('virtual_distillation_theory.png', dpi=150)
    plt.show()

    # Print numerical results
    print("\nError after distillation:")
    print("n\\ε  ", end="")
    for epsilon in error_rates:
        print(f"  {epsilon:.2f}  ", end="")
    print()

    for j, n in enumerate(copy_numbers):
        print(f"{n}   ", end="")
        for i in range(len(error_rates)):
            print(f"  {results[i,j]:.4f}", end="")
        print()

# ============================================================
# Part 2: SWAP Test Implementation
# ============================================================

def create_swap_test_circuit(n_qubits: int) -> QuantumCircuit:
    """Create SWAP test circuit for two n-qubit states."""
    # Ancilla + 2 copies of n qubits
    qc = QuantumCircuit(2 * n_qubits + 1, 1)

    ancilla = 2 * n_qubits

    # Hadamard on ancilla
    qc.h(ancilla)

    # Controlled-SWAP between the two copies
    for i in range(n_qubits):
        qc.cswap(ancilla, i, n_qubits + i)

    # Hadamard on ancilla
    qc.h(ancilla)

    # Measure ancilla
    qc.measure(ancilla, 0)

    return qc

def measure_purity_swap_test(state_circuit: QuantumCircuit,
                             noise_model: NoiseModel = None,
                             shots: int = 10000) -> float:
    """Measure purity using SWAP test."""
    n_qubits = state_circuit.num_qubits

    # Create SWAP test circuit
    swap_test = QuantumCircuit(2 * n_qubits + 1, 1)

    # Prepare two copies
    swap_test.compose(state_circuit, qubits=range(n_qubits), inplace=True)
    swap_test.compose(state_circuit, qubits=range(n_qubits, 2*n_qubits), inplace=True)

    # SWAP test
    ancilla = 2 * n_qubits
    swap_test.h(ancilla)

    for i in range(n_qubits):
        swap_test.cswap(ancilla, i, n_qubits + i)

    swap_test.h(ancilla)
    swap_test.measure(ancilla, 0)

    # Execute
    if noise_model:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()

    result = simulator.run(swap_test, shots=shots).result()
    counts = result.get_counts()

    # P(0) = (1 + Tr(ρ²))/2
    p0 = counts.get('0', 0) / shots
    purity = 2 * p0 - 1

    return purity

def demonstrate_swap_test():
    """Demonstrate SWAP test for purity measurement."""

    print("\n" + "="*60)
    print("SWAP Test for Purity Measurement")
    print("="*60)

    n_qubits = 1

    # Different states
    states = {
        'Pure |0⟩': lambda qc: None,  # Identity
        'Pure |+⟩': lambda qc: qc.h(0),
        'Mixed (rotation)': lambda qc: qc.ry(np.pi/3, 0)
    }

    # Add noise for mixed state simulation
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.1, 1), ['h', 'ry', 'id']
    )

    print("\nPurity measurements:")

    for name, prep_fn in states.items():
        # Create state preparation circuit
        qc = QuantumCircuit(n_qubits)
        prep_fn(qc)

        # Measure purity (ideal)
        purity_ideal = measure_purity_swap_test(qc, noise_model=None)

        # Measure purity (with noise on state prep)
        purity_noisy = measure_purity_swap_test(qc, noise_model=noise_model)

        print(f"  {name}:")
        print(f"    Ideal purity: {purity_ideal:.4f}")
        print(f"    Noisy purity: {purity_noisy:.4f}")

# ============================================================
# Part 3: Full Virtual Distillation Protocol
# ============================================================

class VirtualDistillation:
    """Virtual distillation for error mitigation."""

    def __init__(self, n_copies: int = 2):
        self.n_copies = n_copies

    def prepare_copies(self, state_circuit: QuantumCircuit) -> QuantumCircuit:
        """Prepare n copies of the state."""
        n_qubits = state_circuit.num_qubits
        total_qubits = self.n_copies * n_qubits

        qc = QuantumCircuit(total_qubits)

        for copy in range(self.n_copies):
            offset = copy * n_qubits
            qc.compose(state_circuit, qubits=range(offset, offset + n_qubits), inplace=True)

        return qc

    def estimate_trace_rho_n(self, state_circuit: QuantumCircuit,
                             noise_model: NoiseModel = None,
                             shots: int = 10000) -> float:
        """Estimate Tr(ρⁿ) using cyclic permutation test."""
        if self.n_copies == 2:
            return measure_purity_swap_test(state_circuit, noise_model, shots)

        # For n > 2, use generalized permutation test (simplified)
        # This is an approximation using pairwise SWAP tests
        purity = measure_purity_swap_test(state_circuit, noise_model, shots)
        return purity ** (self.n_copies - 1)

    def estimate_expectation(self, state_circuit: QuantumCircuit,
                            observable: str,
                            noise_model: NoiseModel = None,
                            shots: int = 10000) -> Tuple[float, float]:
        """
        Estimate distilled expectation value.

        Returns (distilled_expectation, raw_expectation)
        """
        n_qubits = state_circuit.num_qubits

        # Measure raw expectation
        qc_raw = state_circuit.copy()
        self._add_measurement_basis_change(qc_raw, observable)
        qc_raw.measure_all()

        if noise_model:
            simulator = AerSimulator(noise_model=noise_model)
        else:
            simulator = AerSimulator()

        result = simulator.run(qc_raw, shots=shots).result()
        counts = result.get_counts()
        raw_exp = self._compute_expectation_from_counts(counts, observable)

        # Estimate Tr(ρⁿ)
        trace_rho_n = self.estimate_trace_rho_n(state_circuit, noise_model, shots)

        # For distillation, we need Tr(ρⁿ O)
        # Simplified: use raw expectation scaled by purity correction
        # This is an approximation; full protocol requires more complex circuits

        # Approximate distilled expectation
        # For pure states: Tr(ρⁿ O) / Tr(ρⁿ) ≈ <O>
        # For mixed: correction factor based on purity

        distilled_exp = raw_exp  # Simplified; see full implementation below

        return distilled_exp, raw_exp

    def _add_measurement_basis_change(self, qc: QuantumCircuit, observable: str):
        """Add basis change for Pauli measurement."""
        for i, p in enumerate(observable[::-1]):
            if p == 'X':
                qc.h(i)
            elif p == 'Y':
                qc.sdg(i)
                qc.h(i)

    def _compute_expectation_from_counts(self, counts: dict, observable: str) -> float:
        """Compute Pauli expectation from measurement counts."""
        total = sum(counts.values())
        expectation = 0

        for bitstring, count in counts.items():
            # Compute eigenvalue
            eigenvalue = 1
            for i, p in enumerate(observable[::-1]):
                if p in ['X', 'Y', 'Z']:
                    bit = int(bitstring[-(i+1)])
                    eigenvalue *= (-1)**bit

            expectation += eigenvalue * count / total

        return expectation

def full_virtual_distillation_simulation():
    """Simulate full virtual distillation with density matrices."""

    print("\n" + "="*60)
    print("Full Virtual Distillation Simulation")
    print("="*60)

    # Create a simple 1-qubit state circuit
    qc = QuantumCircuit(1)
    qc.ry(np.pi/3, 0)  # Prepare non-trivial state

    # Get ideal state
    ideal_state = Statevector.from_instruction(qc)
    ideal_dm = DensityMatrix(ideal_state)

    # Define observable
    Z = np.array([[1, 0], [0, -1]])
    ideal_exp = np.real(np.trace(ideal_dm.data @ Z))

    print(f"Ideal state prepared with Ry(π/3)")
    print(f"Ideal <Z> = {ideal_exp:.4f}")

    # Simulate noisy state (depolarizing)
    error_rates = [0.05, 0.1, 0.15, 0.2]

    results = {'raw': [], 'n=2': [], 'n=3': [], 'n=4': []}

    for epsilon in error_rates:
        # Create noisy density matrix
        rho_noisy = (1 - epsilon) * ideal_dm.data + epsilon * np.eye(2) / 2

        # Raw expectation
        raw = np.real(np.trace(rho_noisy @ Z))
        results['raw'].append(abs(raw - ideal_exp))

        # Virtual distillation with different copy numbers
        for n in [2, 3, 4]:
            distilled = virtual_distillation_exact(rho_noisy, Z, n)
            results[f'n={n}'].append(abs(distilled - ideal_exp))

    # Plot results
    plt.figure(figsize=(10, 6))

    x = np.arange(len(error_rates))
    width = 0.2

    for i, (name, errors) in enumerate(results.items()):
        plt.bar(x + i*width, errors, width, label=name)

    plt.xlabel('Error Rate')
    plt.ylabel('|Error from Ideal|')
    plt.title('Virtual Distillation Error Reduction')
    plt.xticks(x + 1.5*width, [f'{e:.0%}' for e in error_rates])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('virtual_distillation_results.png', dpi=150)
    plt.show()

    # Print results
    print("\nError from ideal (smaller is better):")
    print(f"{'ε':<10}", end="")
    for name in results.keys():
        print(f"{name:<12}", end="")
    print()

    for i, eps in enumerate(error_rates):
        print(f"{eps:<10.2f}", end="")
        for name in results.keys():
            print(f"{results[name][i]:<12.6f}", end="")
        print()

# ============================================================
# Part 4: Sample Complexity Analysis
# ============================================================

def analyze_sample_complexity():
    """Analyze sample complexity of virtual distillation."""

    print("\n" + "="*60)
    print("Sample Complexity Analysis")
    print("="*60)

    purities = np.linspace(0.5, 0.99, 50)
    copy_numbers = [2, 3, 4, 5]

    plt.figure(figsize=(12, 5))

    # Plot 1: Effective Tr(ρⁿ) vs purity
    plt.subplot(1, 2, 1)
    for n in copy_numbers:
        trace_rho_n = purities ** (n - 1)
        plt.semilogy(purities, trace_rho_n, label=f'n = {n}')

    plt.xlabel('Purity γ = Tr(ρ²)')
    plt.ylabel('Tr(ρⁿ)')
    plt.title('Trace of ρⁿ vs Purity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Sample overhead vs purity
    plt.subplot(1, 2, 2)
    for n in copy_numbers:
        overhead = 1 / (purities ** (2 * (n - 1)))
        plt.semilogy(purities, overhead, label=f'n = {n}')

    plt.xlabel('Purity γ = Tr(ρ²)')
    plt.ylabel('Sample Overhead (1/Tr(ρⁿ)²)')
    plt.title('Sample Complexity vs Purity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sample_complexity.png', dpi=150)
    plt.show()

    # Numerical examples
    print("\nSample overhead examples:")
    print(f"{'Purity':<10}{'n=2':<15}{'n=3':<15}{'n=4':<15}")

    for gamma in [0.9, 0.8, 0.7, 0.6, 0.5]:
        overheads = [1 / (gamma ** (2 * (n - 1))) for n in [2, 3, 4]]
        print(f"{gamma:<10.1f}{overheads[0]:<15.1f}{overheads[1]:<15.1f}{overheads[2]:<15.1f}")

# ============================================================
# Part 5: Optimal Copy Number Selection
# ============================================================

def find_optimal_copies():
    """Find optimal number of copies for given parameters."""

    print("\n" + "="*60)
    print("Optimal Copy Number Selection")
    print("="*60)

    # Parameters
    epsilon = 0.1  # Error rate
    N_shots = 1e6  # Available shots

    purities = np.linspace(0.5, 0.95, 10)

    plt.figure(figsize=(12, 5))

    # For each purity, find optimal n
    optimal_ns = []

    for gamma in purities:
        best_n = 1
        best_error = float('inf')

        for n in range(1, 10):
            # Bias: O(ε^n)
            bias = epsilon ** n

            # Variance: 1 / (γ^(2(n-1)) * N)
            variance = 1 / (gamma ** (2 * (n - 1)) * N_shots)
            std = np.sqrt(variance)

            # Total error (simplified)
            total = bias + std

            if total < best_error:
                best_error = total
                best_n = n

        optimal_ns.append(best_n)

    # Plot 1: Optimal n vs purity
    plt.subplot(1, 2, 1)
    plt.plot(purities, optimal_ns, 'bo-', markersize=10)
    plt.xlabel('Purity γ')
    plt.ylabel('Optimal Number of Copies')
    plt.title(f'Optimal n (ε={epsilon}, N={N_shots:.0e})')
    plt.grid(True, alpha=0.3)

    # Plot 2: Error landscape for fixed purity
    plt.subplot(1, 2, 2)
    gamma_fixed = 0.8
    ns = range(1, 8)

    biases = [epsilon ** n for n in ns]
    stds = [1 / np.sqrt(gamma_fixed ** (2 * (n - 1)) * N_shots) for n in ns]
    totals = [b + s for b, s in zip(biases, stds)]

    plt.semilogy(list(ns), biases, 'b--', label='Bias')
    plt.semilogy(list(ns), stds, 'r--', label='Std Dev')
    plt.semilogy(list(ns), totals, 'g-', linewidth=2, label='Total Error')

    optimal = np.argmin(totals) + 1
    plt.axvline(x=optimal, color='gray', linestyle=':', label=f'Optimal n={optimal}')

    plt.xlabel('Number of Copies')
    plt.ylabel('Error')
    plt.title(f'Error Components (γ={gamma_fixed})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimal_copies.png', dpi=150)
    plt.show()

    print(f"\nFor ε={epsilon}, N={N_shots:.0e}:")
    for gamma, n_opt in zip(purities, optimal_ns):
        print(f"  γ = {gamma:.2f}: optimal n = {n_opt}")

# ============================================================
# Part 6: Combining with Other Techniques
# ============================================================

def combine_distillation_zne():
    """Demonstrate combining virtual distillation with ZNE."""

    print("\n" + "="*60)
    print("Combining Virtual Distillation with ZNE")
    print("="*60)

    # Create noisy state
    epsilon = 0.15
    ideal_state = np.array([np.cos(np.pi/6), np.sin(np.pi/6)], dtype=complex)
    rho = create_noisy_state(ideal_state, epsilon)

    # Observable
    Z = np.array([[1, 0], [0, -1]])
    ideal_exp = np.real(ideal_state.conj() @ Z @ ideal_state)

    print(f"Ideal <Z> = {ideal_exp:.4f}")
    print(f"Base error rate: {epsilon:.0%}")

    # Method 1: Raw noisy
    raw_exp = np.real(np.trace(rho @ Z))
    print(f"\n1. Raw noisy: {raw_exp:.4f} (error: {abs(raw_exp - ideal_exp):.4f})")

    # Method 2: Virtual distillation only (n=2)
    distilled_2 = virtual_distillation_exact(rho, Z, 2)
    print(f"2. VD (n=2): {distilled_2:.4f} (error: {abs(distilled_2 - ideal_exp):.4f})")

    # Method 3: ZNE only
    # Simulate ZNE by increasing error
    scale_factors = [1, 1.5, 2]
    zne_values = []

    for scale in scale_factors:
        eps_scaled = min(epsilon * scale, 0.5)
        rho_scaled = create_noisy_state(ideal_state, eps_scaled)
        zne_values.append(np.real(np.trace(rho_scaled @ Z)))

    # Linear extrapolation
    zne_exp = (scale_factors[1] * zne_values[0] - scale_factors[0] * zne_values[1]) / (scale_factors[1] - scale_factors[0])
    print(f"3. ZNE: {zne_exp:.4f} (error: {abs(zne_exp - ideal_exp):.4f})")

    # Method 4: VD + ZNE
    # Apply ZNE to distilled values
    vd_zne_values = []

    for scale in scale_factors:
        eps_scaled = min(epsilon * scale, 0.5)
        rho_scaled = create_noisy_state(ideal_state, eps_scaled)
        vd_val = virtual_distillation_exact(rho_scaled, Z, 2)
        vd_zne_values.append(vd_val)

    vd_zne_exp = (scale_factors[1] * vd_zne_values[0] - scale_factors[0] * vd_zne_values[1]) / (scale_factors[1] - scale_factors[0])
    print(f"4. VD + ZNE: {vd_zne_exp:.4f} (error: {abs(vd_zne_exp - ideal_exp):.4f})")

    # Summary plot
    methods = ['Raw', 'VD (n=2)', 'ZNE', 'VD + ZNE']
    values = [raw_exp, distilled_2, zne_exp, vd_zne_exp]
    errors = [abs(v - ideal_exp) for v in values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['red', 'blue', 'green', 'purple']

    ax1.bar(methods, values, color=colors)
    ax1.axhline(y=ideal_exp, color='black', linestyle='--', label='Ideal')
    ax1.set_ylabel('<Z>')
    ax1.set_title('Expectation Values')
    ax1.legend()

    ax2.bar(methods, errors, color=colors)
    ax2.set_ylabel('|Error|')
    ax2.set_title('Errors from Ideal')

    for i, (v, e) in enumerate(zip(values, errors)):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')
        ax2.text(i, e + 0.002, f'{e:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig('vd_zne_combination.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Day 945: Virtual Distillation Lab")
    print("="*60)

    # Part 1: Theoretical analysis
    print("\n--- Part 1: Theoretical Foundation ---")
    analyze_distillation_theory()

    # Part 2: SWAP test demonstration
    print("\n--- Part 2: SWAP Test ---")
    demonstrate_swap_test()

    # Part 3: Full simulation
    print("\n--- Part 3: Full Simulation ---")
    full_virtual_distillation_simulation()

    # Part 4: Sample complexity
    print("\n--- Part 4: Sample Complexity ---")
    analyze_sample_complexity()

    # Part 5: Optimal copies
    print("\n--- Part 5: Optimal Copy Selection ---")
    find_optimal_copies()

    # Part 6: Combination with ZNE
    print("\n--- Part 6: Combining Techniques ---")
    combine_distillation_zne()

    print("\n" + "="*60)
    print("Lab Complete! Key Takeaways:")
    print("  1. Virtual distillation provides exponential error suppression")
    print("  2. Sample complexity increases with copy number")
    print("  3. Optimal n balances bias reduction vs variance increase")
    print("  4. Can be combined with ZNE for additional improvement")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Distilled Expectation | $\langle O \rangle_{\text{distilled}} = \frac{\text{Tr}(\rho^n O)}{\text{Tr}(\rho^n)}$ |
| Error Suppression | Bias $= O(\epsilon^n)$ |
| SWAP Test | $P(0) = \frac{1 + \text{Tr}(\rho^2)}{2}$ |
| Sample Overhead | $N = O(1/\text{Tr}(\rho^n)^2) = O(1/\gamma^{2(n-1)})$ |
| Purity Limit | Need $\text{Tr}(\rho^n) > 0$ for reliable estimation |

### Main Takeaways

1. **Exponential error suppression**: With $n$ copies, errors reduce as $O(\epsilon^n)$

2. **Trade-off exists**: More copies suppress bias but increase variance due to $1/\text{Tr}(\rho^n)^2$ overhead

3. **Purity is critical**: Virtual distillation works best for high-purity states; fails for highly mixed states

4. **Implementation via SWAP test**: Purity and distilled expectations can be estimated through controlled-SWAP circuits

5. **Complementary technique**: Virtual distillation can be combined with ZNE, PEC, and measurement mitigation

## Daily Checklist

- [ ] I understand the purification formula $\text{Tr}(\rho^n O)/\text{Tr}(\rho^n)$
- [ ] I can analyze error suppression scaling with copy number
- [ ] I can implement SWAP tests for purity estimation
- [ ] I understand sample complexity trade-offs
- [ ] I can determine optimal copy number for given parameters
- [ ] I know when virtual distillation is appropriate

## Week 135 Summary

This week covered the major error mitigation techniques for NISQ devices:

| Day | Topic | Key Insight |
|-----|-------|-------------|
| 939 | Overview | Mitigation vs correction trade-offs |
| 940 | ZNE | Extrapolate to zero noise via scaling |
| 941 | PEC | Unbiased but exponential overhead |
| 942 | Symmetry | Exploit conservation laws for detection |
| 943 | Measurement | Invert confusion matrices |
| 944 | DD | Refocus noise during idle periods |
| 945 | Virtual Distillation | Exponential suppression via copies |

**Integration**: Real applications combine multiple techniques:
1. DD during circuit execution
2. ZNE or PEC for gate errors
3. Symmetry verification for error detection
4. Measurement mitigation for readout
5. Virtual distillation for high-precision estimates

## Preview of Week 136

Next week explores **Quantum Hardware Calibration and Benchmarking**:

- Gate calibration protocols
- Randomized benchmarking
- Cross-entropy benchmarking
- Quantum volume and other metrics
- Continuous calibration strategies
