# Day 954: Quantum Phase Estimation Review

## Week 137, Day 2 | Month 35: Advanced Quantum Algorithms

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: QPE algorithm and precision analysis |
| Afternoon | 2.5 hours | Problem solving: Eigenvalue extraction |
| Evening | 2 hours | Computational lab: QPE implementation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Derive the QPE algorithm** from quantum Fourier transform principles
2. **Analyze precision requirements** relating ancilla count to eigenvalue accuracy
3. **Calculate success probabilities** for eigenvalue estimation
4. **Design controlled unitary operations** for Hamiltonian simulation
5. **Connect QPE to HHL** understanding how eigenvalue extraction enables inversion
6. **Implement QPE circuits** in Qiskit with detailed analysis

---

## Core Content

### 1. The Phase Estimation Problem

Quantum Phase Estimation solves a fundamental problem: given a unitary operator $U$ and one of its eigenstates $|u\rangle$, find the eigenvalue $e^{2\pi i\phi}$.

#### Problem Statement

$$U|u\rangle = e^{2\pi i\phi}|u\rangle$$

**Input:**
- Unitary $U$ (via controlled operations)
- Eigenstate $|u\rangle$ (or superposition of eigenstates)

**Output:**
- $n$-bit approximation $\tilde{\phi}$ to the phase $\phi \in [0,1)$

**Precision:**
$$|\tilde{\phi} - \phi| \leq 2^{-n}$$

with high probability using $n$ ancilla qubits.

### 2. QPE Circuit Architecture

The QPE algorithm consists of three stages:

```
|0⟩──H──●────────────────────────●──────────QFT†──M
        │                        │
|0⟩──H──┼──●─────────────────────┼──●───────QFT†──M
        │  │                     │  │
|0⟩──H──┼──┼──●──────────────────┼──┼──●────QFT†──M
        │  │  │                  │  │  │
  ...   │  │  │    ...           │  │  │
        │  │  │                  │  │  │
|u⟩─────U──U²─U⁴─────────────────U^(2^(n-1))─────|u⟩
```

#### Stage 1: Hadamard Transform

Apply Hadamard to all $n$ ancilla qubits:
$$|0\rangle^{\otimes n} \xrightarrow{H^{\otimes n}} \frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} |k\rangle$$

#### Stage 2: Controlled Unitary Applications

Apply controlled-$U^{2^j}$ where ancilla qubit $j$ controls $U^{2^j}$:

$$\frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} |k\rangle |u\rangle \xrightarrow{C-U^{2^j}} \frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} e^{2\pi i \phi k} |k\rangle |u\rangle$$

This encodes the phase into the ancilla register!

#### Stage 3: Inverse QFT

Apply inverse Quantum Fourier Transform to extract $\phi$:
$$\frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} e^{2\pi i \phi k} |k\rangle \xrightarrow{QFT^{-1}} |2^n \phi\rangle$$

When $\phi = m/2^n$ exactly, we get $|m\rangle$ with certainty.

### 3. Mathematical Derivation

#### State After Controlled Unitaries

For eigenstate $|u\rangle$ with $U|u\rangle = e^{2\pi i\phi}|u\rangle$:

Each ancilla qubit $j$ in state $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ becomes:
$$\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 2^j \phi}|1\rangle)$$

The combined state of all $n$ ancillas:
$$\bigotimes_{j=0}^{n-1} \frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 2^j \phi}|1\rangle) = \frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} e^{2\pi i \phi k}|k\rangle$$

#### QFT Definition

The Quantum Fourier Transform on $n$ qubits:
$$QFT|j\rangle = \frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n}|k\rangle$$

#### Inverse QFT Application

The state before inverse QFT is exactly $QFT|2^n\phi\rangle$ when $2^n\phi$ is an integer.

Applying $QFT^{-1}$:
$$QFT^{-1} \left(\frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} e^{2\pi i \phi k}|k\rangle\right) = |m\rangle$$

where $m = 2^n\phi \mod 2^n$.

### 4. Precision and Error Analysis

#### Exact Phase Case

When $\phi = m/2^n$ for integer $m$:
- Measurement gives $m$ with probability 1
- Perfect precision achieved

#### Inexact Phase Case

When $\phi \neq m/2^n$, let $\phi = (m + \delta)/2^n$ where $|\delta| \leq 1/2$.

The probability of measuring $m$ (closest integer to $2^n\phi$):

$$\boxed{P(m) = \frac{\sin^2(\pi\delta)}{2^{2n}\sin^2(\pi\delta/2^n)} \geq \frac{4}{\pi^2} \approx 0.405}$$

More generally, the probability of being within $\pm 1$ of the correct value:
$$P(|m - 2^n\phi| \leq 1) \geq 1 - \frac{1}{2(2^n-1)} \approx 1 - 2^{-n}$$

#### Success Probability Bound

$$\boxed{P(\text{success}) \geq 1 - \epsilon \quad \text{using} \quad n = \log_2(1/\epsilon) + O(1) \text{ ancillas}}$$

### 5. Hamiltonian Simulation for HHL

In HHL, we need QPE for $U = e^{iAt}$ where $A$ is the system matrix.

#### Connection to Eigenvalues

If $A|u_j\rangle = \lambda_j|u_j\rangle$, then:
$$e^{iAt}|u_j\rangle = e^{i\lambda_j t}|u_j\rangle$$

The phase is $\phi_j = \lambda_j t / (2\pi)$.

From measured phase $\tilde{\phi}_j$:
$$\tilde{\lambda}_j = 2\pi\tilde{\phi}_j / t$$

#### Time Parameter Selection

To distinguish eigenvalues with gap $\Delta\lambda$:
$$t \geq \frac{2\pi}{\Delta\lambda}$$

To achieve precision $\epsilon$ in eigenvalue:
$$\epsilon_\lambda = \frac{2\pi}{t} \cdot 2^{-n}$$

#### Eigenvalue Range

If $\lambda \in [\lambda_{min}, \lambda_{max}]$, we need:
$$\phi = \frac{\lambda t}{2\pi} \in [0, 1)$$

This requires:
$$t < \frac{2\pi}{\lambda_{max}}$$

**Trade-off:** Large $t$ gives precision, small $t$ avoids phase wrapping.

### 6. Controlled Unitary Implementation

#### Efficient $U^{2^k}$ Computation

Rather than applying $U$ repeatedly $2^k$ times, we use:
$$U^{2^k} = (U^{2^{k-1}})^2$$

This requires only $k$ squaring operations, not $2^k$ applications.

#### Hamiltonian Simulation Methods

For $U = e^{iHt}$:

| Method | Complexity | Requirements |
|--------|------------|--------------|
| Trotter-Suzuki | $O(t^{1+1/p}/\epsilon^{1/p})$ | Sparse $H$ |
| Linear combination | $O(t \cdot \text{poly}(\log(t/\epsilon)))$ | Oracle access |
| Qubitization | $O(t + \log(1/\epsilon))$ | Block-encoded $H$ |

For HHL, the Hamiltonian simulation cost contributes significantly to total complexity.

### 7. QPE with Superposition Input

When the input is not an eigenstate but a superposition:
$$|b\rangle = \sum_j \beta_j |u_j\rangle$$

QPE produces:
$$\sum_j \beta_j |\tilde{\phi}_j\rangle |u_j\rangle$$

**Critical for HHL:** This allows simultaneous processing of all eigencomponents!

#### Entanglement Structure

After QPE:
- Ancilla register entangled with eigenstate register
- Each eigenvalue $\tilde{\phi}_j$ correlated with eigenstate $|u_j\rangle$
- Enables eigenvalue-dependent operations (the rotation in HHL)

### 8. QPE Resource Costs

#### Qubit Count

- $n$ ancilla qubits for $n$-bit precision
- $\log_2(N)$ qubits for $N$-dimensional eigenstate
- Total: $n + \log_2(N)$

#### Gate Count

- Hadamard gates: $n$
- Controlled unitaries: $n$ applications of controlled-$U^{2^k}$
- Inverse QFT: $O(n^2)$ gates (or $O(n\log n)$ with approximate QFT)

$$\boxed{T_{QPE} = O(n^2) + n \cdot T_{U}}$$

where $T_U$ is the cost of one controlled-$U$ application.

---

## Worked Examples

### Example 1: Two-Qubit QPE for Single Eigenvalue

**Problem:** Implement QPE with 2 ancillas for $U = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/2} \end{pmatrix}$ and eigenstate $|1\rangle$.

**Solution:**

The eigenvalue is $e^{i\pi/2} = e^{2\pi i \cdot (1/4)}$, so $\phi = 1/4 = 0.01_2$ in binary.

Step 1: Initial state
$$|00\rangle|1\rangle$$

Step 2: Hadamards on ancillas
$$\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)|1\rangle$$

Step 3: Controlled-$U^1$ (control on qubit 0)
$$\frac{1}{2}(|00\rangle + e^{i\pi/2}|01\rangle + |10\rangle + e^{i\pi/2}|11\rangle)|1\rangle$$

Step 4: Controlled-$U^2$ (control on qubit 1)
$$U^2 = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = Z$$

$$\frac{1}{2}(|00\rangle + e^{i\pi/2}|01\rangle - |10\rangle - e^{i\pi/2}|11\rangle)|1\rangle$$

Step 5: Inverse QFT on 2 qubits
The inverse QFT maps this to $|01\rangle|1\rangle$ (reading least significant bit first gives $\phi = 0.01_2 = 1/4$).

$$\boxed{\text{Measurement: } |01\rangle \text{ with high probability}}$$

---

### Example 2: Precision Requirement for HHL

**Problem:** For a matrix with eigenvalues in $[1, 10]$, how many ancilla qubits are needed to distinguish eigenvalues with gap $0.1$ and achieve 1% relative error?

**Solution:**

Step 1: Eigenvalue gap requirement
$$\Delta\lambda = 0.1$$

Need $2^n$ levels to span the range with this resolution:
$$2^n \geq \frac{\lambda_{max} - \lambda_{min}}{\Delta\lambda} = \frac{10-1}{0.1} = 90$$

So $n \geq 7$ bits (gives 128 levels).

Step 2: Relative error requirement
For 1% relative error on smallest eigenvalue $\lambda_{min} = 1$:
$$\epsilon_\lambda = 0.01 \cdot 1 = 0.01$$

Precision needed:
$$2^{-n} \cdot (\lambda_{max} - \lambda_{min}) \leq 0.01$$
$$2^{-n} \cdot 9 \leq 0.01$$
$$2^{-n} \leq 0.0011$$
$$n \geq 10$$

$$\boxed{n = 10 \text{ ancilla qubits required}}$$

---

### Example 3: Hamiltonian Simulation Time Selection

**Problem:** Matrix $A$ has eigenvalues in $[0.5, 8]$. Find the simulation time $t$ for QPE.

**Solution:**

Constraint 1: Avoid phase wrapping
$$\phi_{max} = \frac{\lambda_{max} \cdot t}{2\pi} < 1$$
$$t < \frac{2\pi}{\lambda_{max}} = \frac{2\pi}{8} = 0.785$$

Constraint 2: Resolve eigenvalue gap (assume gap $\Delta\lambda = 0.5$)
$$\Delta\phi = \frac{\Delta\lambda \cdot t}{2\pi} > 2^{-n}$$

For $n = 8$ ancillas:
$$t > \frac{2\pi \cdot 2^{-8}}{\Delta\lambda} = \frac{2\pi \cdot 0.0039}{0.5} = 0.049$$

**Optimal choice:**
$$\boxed{t \approx 0.5 \text{ (middle of feasible range)}}$$

This gives:
- $\phi_{max} = 8 \times 0.5 / (2\pi) \approx 0.64$ (no wrapping)
- $\Delta\phi = 0.5 \times 0.5 / (2\pi) \approx 0.04$ (well above $2^{-8} \approx 0.004$)

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** For $U = Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$, what phases correspond to the eigenstates $|0\rangle$ and $|1\rangle$?

**Problem 1.2:** How many ancilla qubits are needed to estimate a phase to 8 binary digits?

**Problem 1.3:** If QPE uses 10 ancilla qubits, what is the precision in radians of the estimated phase?

### Level 2: Intermediate Analysis

**Problem 2.1:** Prove that the state after controlled unitaries in QPE equals:
$$\frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} e^{2\pi i \phi k}|k\rangle |u\rangle$$

**Problem 2.2:** Calculate the probability distribution of outcomes when $\phi = 0.3$ exactly, using 4 ancilla qubits.

**Problem 2.3:** For QPE on $U = e^{iAt}$ with $A$ having condition number $\kappa = 100$, derive the relationship between $t$, $n$, and the precision of eigenvalue estimation.

### Level 3: Challenging Problems

**Problem 3.1:** **Iterative Phase Estimation**

An alternative to standard QPE uses a single ancilla with multiple rounds.
- Describe how to extract $n$ bits of $\phi$ using $n$ iterations
- Compare the circuit depth to standard QPE
- When is this approach preferable?

**Problem 3.2:** **Eigenvalue Resolution**

A matrix has eigenvalues $\lambda_1 = 2.0$ and $\lambda_2 = 2.1$.
- What is the minimum number of ancilla qubits to distinguish them with 95% success probability?
- How does the required $t$ depend on this eigenvalue gap?

**Problem 3.3:** **Error Propagation in HHL**

If QPE estimates eigenvalue $\lambda$ with error $\delta\lambda$, and HHL computes $1/\lambda$:
- Derive the error in $1/\lambda$ in terms of $\delta\lambda$
- Why does this make small eigenvalues problematic?
- How does condition number $\kappa$ affect total error?

---

## Computational Lab

### QPE Implementation in Qiskit

```python
"""
Day 954: Quantum Phase Estimation Implementation
Complete QPE with analysis and visualization.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt
from typing import Tuple, List
from fractions import Fraction


class QuantumPhaseEstimation:
    """Quantum Phase Estimation implementation and analysis."""

    def __init__(self, num_ancilla: int):
        """
        Initialize QPE with specified precision.

        Parameters:
        -----------
        num_ancilla : int
            Number of ancilla qubits (determines precision)
        """
        self.n = num_ancilla
        self.precision = 2 ** (-num_ancilla)

    def create_qpe_circuit(self, unitary: np.ndarray,
                           eigenstate: np.ndarray) -> QuantumCircuit:
        """
        Create QPE circuit for given unitary and eigenstate.

        Parameters:
        -----------
        unitary : ndarray
            2x2 unitary matrix
        eigenstate : ndarray
            Initial eigenstate to prepare

        Returns:
        --------
        QuantumCircuit
            Complete QPE circuit
        """
        # Create registers
        ancilla = QuantumRegister(self.n, 'ancilla')
        target = QuantumRegister(1, 'target')
        classical = ClassicalRegister(self.n, 'output')

        qc = QuantumCircuit(ancilla, target, classical)

        # Prepare eigenstate
        if not np.allclose(eigenstate, [1, 0]):
            # Prepare |eigenstate⟩ from |0⟩
            theta = 2 * np.arccos(np.abs(eigenstate[0]))
            phi = np.angle(eigenstate[1]) - np.angle(eigenstate[0])
            qc.ry(theta, target[0])
            qc.rz(phi, target[0])

        # Hadamard on all ancillas
        qc.h(ancilla)

        # Controlled unitary applications
        for j in range(self.n):
            power = 2 ** j
            self._add_controlled_unitary_power(qc, unitary, power,
                                               ancilla[j], target[0])

        qc.barrier()

        # Inverse QFT on ancilla register
        qft_inverse = QFT(self.n, inverse=True)
        qc.append(qft_inverse, ancilla)

        # Measurement
        qc.measure(ancilla, classical)

        return qc

    def _add_controlled_unitary_power(self, qc: QuantumCircuit,
                                       unitary: np.ndarray, power: int,
                                       control_qubit, target_qubit):
        """Add controlled U^power to circuit."""
        # Compute U^power
        U_power = np.linalg.matrix_power(unitary, power)

        # For 2x2 unitary, decompose into single-qubit gates
        # U = e^{i*alpha} * Rz(beta) * Ry(gamma) * Rz(delta)

        # Extract phase and angles (simplified for diagonal case)
        if np.allclose(U_power, np.diag(np.diag(U_power))):
            # Diagonal matrix: controlled phase
            phase = np.angle(U_power[1, 1]) - np.angle(U_power[0, 0])
            global_phase = np.angle(U_power[0, 0])
            qc.cp(phase, control_qubit, target_qubit)
        else:
            # General case: use controlled-U gate
            from qiskit.circuit.library import UnitaryGate
            controlled_U = UnitaryGate(U_power).control(1)
            qc.append(controlled_U, [control_qubit, target_qubit])

    def run_qpe(self, unitary: np.ndarray, eigenstate: np.ndarray,
                shots: int = 10000) -> dict:
        """
        Run QPE and return results.

        Parameters:
        -----------
        unitary : ndarray
            Unitary matrix
        eigenstate : ndarray
            Eigenstate to use
        shots : int
            Number of measurement shots

        Returns:
        --------
        dict : Results including estimated phase and distribution
        """
        # Create circuit
        qc = self.create_qpe_circuit(unitary, eigenstate)

        # Run simulation
        simulator = AerSimulator()
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled, shots=shots).result()
        counts = result.get_counts()

        # Analyze results
        phases = {}
        for bitstring, count in counts.items():
            # Convert bitstring to integer (reversed for Qiskit convention)
            value = int(bitstring[::-1], 2)
            phase = value / (2 ** self.n)
            phases[phase] = phases.get(phase, 0) + count

        # Find most likely phase
        max_phase = max(phases, key=phases.get)
        max_prob = phases[max_phase] / shots

        return {
            'circuit': qc,
            'counts': counts,
            'phase_distribution': phases,
            'estimated_phase': max_phase,
            'probability': max_prob,
            'shots': shots
        }

    def analyze_precision(self, true_phase: float) -> dict:
        """
        Analyze expected precision for a given true phase.

        Parameters:
        -----------
        true_phase : float
            True phase value in [0, 1)

        Returns:
        --------
        dict : Precision analysis
        """
        # Binary approximation
        approx_binary = round(true_phase * 2**self.n) % (2**self.n)
        estimated_phase = approx_binary / 2**self.n

        # Error
        error = abs(estimated_phase - true_phase)
        if error > 0.5:
            error = 1 - error  # Handle wrap-around

        # Success probability (simplified)
        delta = (true_phase * 2**self.n) % 1
        if delta > 0.5:
            delta = 1 - delta
        prob = np.sin(np.pi * delta)**2 / (2**(2*self.n) * np.sin(np.pi * delta / 2**self.n)**2) if delta != 0 else 1

        return {
            'true_phase': true_phase,
            'n_ancilla': self.n,
            'theoretical_precision': self.precision,
            'estimated_phase': estimated_phase,
            'error': error,
            'success_probability': min(prob, 1.0)
        }


def qpe_for_eigenvalue(eigenvalue: complex, n_ancilla: int) -> dict:
    """
    Create and run QPE for a specific eigenvalue.

    Parameters:
    -----------
    eigenvalue : complex
        Eigenvalue e^{i*theta}
    n_ancilla : int
        Number of ancilla qubits

    Returns:
    --------
    dict : QPE results
    """
    # Create unitary with specified eigenvalue for |1⟩
    U = np.array([[1, 0], [0, eigenvalue]])

    # True phase
    true_phase = np.angle(eigenvalue) / (2 * np.pi)
    if true_phase < 0:
        true_phase += 1

    # Run QPE
    qpe = QuantumPhaseEstimation(n_ancilla)
    result = qpe.run_qpe(U, np.array([0, 1]))

    # Add analysis
    result['true_phase'] = true_phase
    result['true_eigenvalue'] = eigenvalue
    result['phase_error'] = abs(result['estimated_phase'] - true_phase)

    return result


def precision_scaling_analysis():
    """Analyze how precision scales with ancilla count."""
    n_values = range(2, 12)
    true_phase = 0.3  # Not exactly representable

    errors = []
    probs = []

    for n in n_values:
        qpe = QuantumPhaseEstimation(n)
        analysis = qpe.analyze_precision(true_phase)
        errors.append(analysis['error'])
        probs.append(analysis['success_probability'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error scaling
    axes[0].semilogy(list(n_values), errors, 'bo-', linewidth=2, markersize=8)
    axes[0].semilogy(list(n_values), [2**(-n) for n in n_values], 'r--',
                     linewidth=2, label='$2^{-n}$ bound')
    axes[0].set_xlabel('Number of Ancilla Qubits', fontsize=12)
    axes[0].set_ylabel('Phase Error', fontsize=12)
    axes[0].set_title(f'QPE Precision Scaling (true phase = {true_phase})', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Success probability
    axes[1].plot(list(n_values), probs, 'go-', linewidth=2, markersize=8)
    axes[1].axhline(y=4/np.pi**2, color='r', linestyle='--', label='Lower bound 4/π²')
    axes[1].set_xlabel('Number of Ancilla Qubits', fontsize=12)
    axes[1].set_ylabel('Success Probability', fontsize=12)
    axes[1].set_title('QPE Success Probability', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('qpe_precision_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


def eigenvalue_extraction_demo():
    """Demonstrate eigenvalue extraction for multiple eigenvalues."""
    eigenvalues = [np.exp(2j * np.pi * 0.25),
                   np.exp(2j * np.pi * 0.333),
                   np.exp(2j * np.pi * 0.5),
                   np.exp(2j * np.pi * 0.75)]

    n_ancilla = 6

    print(f"QPE Eigenvalue Extraction (n = {n_ancilla} ancillas)")
    print("=" * 60)

    for ev in eigenvalues:
        result = qpe_for_eigenvalue(ev, n_ancilla)
        true_phase = result['true_phase']
        est_phase = result['estimated_phase']
        error = result['phase_error']

        print(f"\nEigenvalue: e^{{2πi × {true_phase:.4f}}}")
        print(f"  True phase:      {true_phase:.6f}")
        print(f"  Estimated phase: {est_phase:.6f}")
        print(f"  Error:           {error:.6f}")
        print(f"  Probability:     {result['probability']:.2%}")


def qpe_circuit_visualization():
    """Create and visualize a QPE circuit."""
    # Simple phase gate for demonstration
    theta = np.pi / 4  # T gate equivalent
    U = np.array([[1, 0], [0, np.exp(1j * theta)]])

    qpe = QuantumPhaseEstimation(4)
    qc = qpe.create_qpe_circuit(U, np.array([0, 1]))

    # Draw circuit
    print("QPE Circuit (4 ancilla qubits):")
    print(qc.draw(output='text', fold=100))

    return qc


def hhl_eigenvalue_simulation():
    """Simulate QPE for HHL-relevant eigenvalue structure."""
    # Simulate matrix with eigenvalues [1, 2, 4, 8]
    # For HHL, we need to invert these

    eigenvalues = [1, 2, 4, 8]
    condition_number = max(eigenvalues) / min(eigenvalues)

    print("HHL Eigenvalue Processing Simulation")
    print("=" * 60)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Condition number: κ = {condition_number}")

    # For e^{iλt}, choose t such that phases are in [0, 1)
    t = 2 * np.pi / (max(eigenvalues) * 1.5)

    print(f"\nSimulation time t = {t:.4f}")
    print(f"Phase range: [0, {max(eigenvalues) * t / (2*np.pi):.3f}]")

    n_ancilla = 6
    qpe = QuantumPhaseEstimation(n_ancilla)

    print(f"\nQPE with {n_ancilla} ancilla qubits:")
    print("-" * 60)

    for lam in eigenvalues:
        # Phase corresponding to eigenvalue
        true_phase = (lam * t) / (2 * np.pi)

        # Create unitary e^{iλt}
        U = np.array([[1, 0], [0, np.exp(1j * lam * t)]])
        result = qpe.run_qpe(U, np.array([0, 1]), shots=5000)

        # Reconstruct eigenvalue from phase
        est_lambda = result['estimated_phase'] * (2 * np.pi) / t

        print(f"\nλ = {lam}")
        print(f"  True phase:      φ = {true_phase:.4f}")
        print(f"  Estimated phase: φ̃ = {result['estimated_phase']:.4f}")
        print(f"  Reconstructed λ: {est_lambda:.4f}")
        print(f"  1/λ (for HHL):   {1/lam:.4f} → {1/est_lambda:.4f}")
        print(f"  Probability:     {result['probability']:.2%}")


# Demonstration
if __name__ == "__main__":
    print("Day 954: Quantum Phase Estimation")
    print("=" * 60)

    # Basic demonstration
    print("\n--- QPE Circuit Visualization ---")
    qc = qpe_circuit_visualization()

    # Precision scaling
    print("\n--- Precision Scaling Analysis ---")
    precision_scaling_analysis()

    # Eigenvalue extraction
    print("\n--- Eigenvalue Extraction Demo ---")
    eigenvalue_extraction_demo()

    # HHL simulation
    print("\n--- HHL Eigenvalue Simulation ---")
    hhl_eigenvalue_simulation()

    # Specific example: T gate phase
    print("\n--- Example: T Gate Phase Estimation ---")
    T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    true_phase = 1/8  # π/4 / 2π = 1/8

    for n in [3, 4, 5, 6]:
        result = qpe_for_eigenvalue(np.exp(1j * np.pi / 4), n)
        print(f"n={n}: estimated {result['estimated_phase']:.4f}, "
              f"true {true_phase:.4f}, error {result['phase_error']:.4f}")
```

---

## Summary

### Key Formulas

| Formula | Expression | Context |
|---------|------------|---------|
| Eigenvalue equation | $U\|u\rangle = e^{2\pi i\phi}\|u\rangle$ | Phase definition |
| Precision | $\|\tilde{\phi} - \phi\| \leq 2^{-n}$ | n ancilla qubits |
| Success probability | $P(\text{success}) \geq 4/\pi^2 \approx 0.405$ | Worst case |
| Gate complexity | $O(n^2) + n \cdot T_U$ | QPE circuit cost |
| Time selection | $t < 2\pi/\lambda_{max}$ | Avoid phase wrapping |

### QPE for HHL Connection

$$e^{iAt}|u_j\rangle = e^{i\lambda_j t}|u_j\rangle \implies \phi_j = \frac{\lambda_j t}{2\pi}$$

### Key Insights

1. **QPE extracts eigenvalues** encoded in unitary phases
2. **Precision scales exponentially** with ancilla count: $n$ qubits give $2^{-n}$ precision
3. **Superposition input enables parallelism** — all eigenvalues processed simultaneously
4. **Hamiltonian simulation is the bottleneck** — cost of implementing $e^{iAt}$
5. **Time $t$ must balance precision and phase wrapping**

---

## Daily Checklist

- [ ] I can derive the QPE algorithm step by step
- [ ] I understand the precision-ancilla relationship
- [ ] I can calculate success probabilities
- [ ] I know how to choose simulation time $t$
- [ ] I understand controlled unitary implementation
- [ ] I can connect QPE output to eigenvalue extraction
- [ ] I implemented and tested QPE in Qiskit

---

## Preview: Day 955

Tomorrow we derive the **complete HHL algorithm**, putting QPE to work for matrix inversion. We'll cover:

- The full HHL circuit: state preparation, QPE, controlled rotation, uncomputation
- The controlled rotation $R_y(\arcsin(C/\lambda))$ that encodes $1/\lambda$
- Amplitude amplification and success probability
- Complete complexity analysis: $O(\log(N) \cdot s^2 \cdot \kappa^2 / \epsilon)$
- Comparison with classical algorithms

HHL is where the pieces come together—QPE extracts eigenvalues, controlled rotation inverts them, and quantum parallelism processes all components simultaneously.

---

*Day 954 of 2184 | Week 137 of 312 | Month 35 of 72*

*"QPE is the quantum algorithm's way of asking: what eigenvalue does this eigenvector belong to?"*
