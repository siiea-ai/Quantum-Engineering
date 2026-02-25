# Day 957: State Preparation & Readout

## Week 137, Day 5 | Month 35: Advanced Quantum Algorithms

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Amplitude encoding and qRAM |
| Afternoon | 2.5 hours | Problem solving: Readout strategies |
| Evening | 2 hours | Computational lab: State preparation circuits |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Analyze the state preparation bottleneck** and its impact on HHL complexity
2. **Design amplitude encoding circuits** for small data sets
3. **Understand qRAM** requirements, proposals, and limitations
4. **Implement efficient readout strategies** for extracting useful information
5. **Distinguish expectation values from full state tomography**
6. **Evaluate end-to-end HHL complexity** including input/output costs

---

## Core Content

### 1. The State Preparation Challenge

The "fine print" of HHL: preparing $|b\rangle$ and reading out $|x\rangle$ often dominate the total cost.

#### The Hidden Cost

The HHL complexity claim:
$$T_{HHL} = O\left(\frac{\log(N) \cdot s^2 \cdot \kappa^2}{\epsilon}\right)$$

assumes $|b\rangle$ is **already prepared**. But converting classical data $\vec{b} = (b_1, \ldots, b_N)$ to:
$$|b\rangle = \frac{1}{\|b\|}\sum_{i=1}^{N} b_i |i\rangle$$

typically costs $O(N)$ operations!

#### The Encoding Problem

**Input:** Classical vector $\vec{b} \in \mathbb{R}^N$

**Output:** Quantum state $|b\rangle$ with amplitudes proportional to $b_i$

**Naive approach:** Prepare $|b\rangle$ using a sequence of controlled rotations:
$$T_{prep} = O(N)$$

This **destroys the exponential speedup** since $O(N) > O(\log N)$.

### 2. Amplitude Encoding Methods

Several approaches exist for loading classical data:

#### Method 1: Unary Encoding

Map each component to a separate qubit:
$$|b\rangle = \sum_i b_i |0\rangle^{\otimes i-1} |1\rangle |0\rangle^{\otimes N-i}$$

**Pros:** Simple preparation
**Cons:** Requires $N$ qubits, no exponential space advantage

#### Method 2: Binary Amplitude Encoding

Encode $N = 2^n$ components in $n$ qubits:
$$|b\rangle = \sum_{i=0}^{2^n-1} b_i |i\rangle$$

**Standard circuit construction:**

```
|0⟩^n → Apply rotations → |b⟩
```

Using Schmidt decomposition recursively:
$$T_{encode} = O(2^n) = O(N)$$

#### Method 3: Sparse State Preparation

If $|b\rangle$ has only $k$ non-zero amplitudes:
$$|b\rangle = \sum_{j=1}^{k} b_{i_j} |i_j\rangle$$

**Preparation cost:** $O(k \cdot n)$ where $n = \log_2(N)$

**When useful:** Sparse right-hand sides (localized sources, boundary conditions)

### 3. Quantum Random Access Memory (qRAM)

The theoretical solution to state preparation: hardware that provides quantum superposition access to classical data.

#### qRAM Definition

A qRAM implements:
$$|i\rangle|0\rangle \xrightarrow{qRAM} |i\rangle|b_i\rangle$$

for all $i$ simultaneously in superposition.

**Implication for amplitude encoding:**
$$|+\rangle^{\otimes n}|0\rangle \xrightarrow{qRAM} \frac{1}{\sqrt{N}}\sum_i |i\rangle|b_i\rangle \xrightarrow{} |b\rangle$$

**Query complexity:** $O(\log N)$ per query!

#### qRAM Architectures

**Bucket-Brigade qRAM (Giovannetti et al., 2008):**

Structure:
```
                    [Root]
                   /      \
               [L1]        [R1]
              /    \      /    \
           [L2]  [R2]  [L3]  [R3]
            |     |     |     |
           b_0   b_1   b_2   b_3
```

- Uses $O(N)$ physical qubits as routers
- Each query touches $O(\log N)$ routers
- Total resource: $O(N)$ qubits, $O(\log N)$ time per query

**Critical issue:** Errors accumulate over $O(N)$ qubits, making fault tolerance challenging.

#### qRAM Challenges

| Challenge | Description | Status |
|-----------|-------------|--------|
| Error sensitivity | $O(N)$ components, each can fail | Major concern |
| Decoherence | Long routing paths | Architecture-dependent |
| Physical realization | No working prototype | Research ongoing |
| Resource overhead | $O(N)$ qubits for $N$ entries | Comparable to classical |

**Current consensus:** qRAM remains theoretical; practical implementations face severe challenges.

### 4. When State Preparation is Efficient

Despite the general $O(N)$ barrier, some scenarios allow efficient preparation:

#### Scenario 1: Quantum-Generated Data

If $|b\rangle$ comes from another quantum computation:
$$|\psi\rangle \xrightarrow{U} |b\rangle$$

No classical data loading needed!

**Example:** Quantum simulation outputs become HHL inputs.

#### Scenario 2: Structured Data

Data with known structure can be prepared efficiently:

**Uniform superposition:**
$$|+\rangle^{\otimes n} = \frac{1}{\sqrt{N}}\sum_{i=0}^{N-1}|i\rangle$$
**Cost:** $O(n) = O(\log N)$

**Periodic functions:**
$$|f\rangle = \sum_i f(i)|i\rangle \text{ where } f \text{ is efficiently computable}$$
**Cost:** $O(\text{poly}(\log N))$

**Gaussian/smooth distributions:**
Using quantum arithmetic circuits.
**Cost:** $O(\text{poly}(\log N))$

#### Scenario 3: Oracle Access

If we have a quantum oracle computing $b_i$:
$$O_b: |i\rangle|0\rangle \to |i\rangle|b_i\rangle$$

Then amplitude encoding is efficient.

### 5. Readout Strategies

The output $|x\rangle$ is a quantum state. Extracting classical information is another bottleneck.

#### Full State Tomography

Reconstruct the complete classical vector $\vec{x}$:
- Requires $O(N)$ measurement settings
- Each setting needs $O(1/\epsilon^2)$ repetitions
- **Total cost:** $O(N/\epsilon^2)$

This **completely negates** quantum advantage.

#### Expectation Value Measurement

Often, we need only $\langle x|M|x\rangle$ for some observable $M$:

$$\langle M \rangle = \langle x|M|x\rangle$$

**Cost:** $O(1/\epsilon^2)$ repetitions (independent of $N$!)

**Useful when:**
- Computing inner products $\langle x|y\rangle$
- Evaluating quadratic forms
- Machine learning feature extraction

#### Sampling from $|x\rangle$

Measure $|x\rangle$ in computational basis:
- Probability of outcome $|i\rangle$: $|x_i|^2$
- Useful for Monte Carlo methods
- **Cost per sample:** $O(1)$ (times algorithm preparation)

### 6. The Complete End-to-End Picture

#### Honest HHL Complexity

Including all costs:

$$\boxed{T_{total} = T_{prep} + T_{HHL} + T_{read}}$$

where:
- $T_{prep}$: State preparation (often $O(N)$)
- $T_{HHL} = O(\log N \cdot s^2 \cdot \kappa^2 / \epsilon)$: Core algorithm
- $T_{read}$: Readout (depends on what's needed)

#### When Does Quantum Advantage Survive?

| Scenario | Prep Cost | Read Cost | Net Advantage? |
|----------|-----------|-----------|----------------|
| Quantum $|b\rangle$, expectation output | $O(1)$ | $O(1/\epsilon^2)$ | **Yes** |
| Classical $\vec{b}$, expectation output | $O(N)$ | $O(1/\epsilon^2)$ | No |
| Quantum $|b\rangle$, sampling output | $O(1)$ | $O(k)$ for $k$ samples | **Yes** |
| Classical $\vec{b}$, full $\vec{x}$ output | $O(N)$ | $O(N)$ | No |

**Key insight:** HHL provides advantage only when:
1. Input is already quantum (or efficiently preparable)
2. Output needs limited classical information

### 7. Amplitude Encoding Circuits

#### Small-Scale Construction

For $N = 4$ ($n = 2$ qubits), encode $|b\rangle = \sum_{i=0}^{3} b_i |i\rangle$:

**Step 1:** First rotation
$$|00\rangle \xrightarrow{R_y(\theta_0)} \cos(\theta_0/2)|00\rangle + \sin(\theta_0/2)|10\rangle$$

where $\theta_0 = 2\arccos(\sqrt{|b_0|^2 + |b_1|^2}/\|b\|)$

**Step 2:** Controlled rotations
$$\xrightarrow{C-R_y} \text{encode } b_0, b_1 \text{ in } |0x\rangle, \text{ and } b_2, b_3 \text{ in } |1x\rangle$$

**General pattern:**
```
|0⟩─────R_y(θ_0)──●────────────────
                   │
|0⟩───────────────R_y(θ_1)──●──────
                             │
...
```

**Gate count:** $O(2^n - 1)$ rotations

#### QRAM-Free Alternative: Data Loading Trees

For structured data, use quantum arithmetic:
$$|i\rangle \xrightarrow{compute} |i\rangle|f(i)\rangle$$

Then amplitude amplification to create superposition.

---

## Worked Examples

### Example 1: 4-Component State Preparation

**Problem:** Prepare $|b\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle) = |++\rangle$.

**Solution:**

This is the uniform superposition—a special case!

$$|00\rangle \xrightarrow{H \otimes H} \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$

**Circuit:**
```
|0⟩──H──
|0⟩──H──
```

**Cost:** $O(n) = O(2)$ gates, not $O(N) = O(4)$!

**Lesson:** Structured states can be prepared efficiently.

---

### Example 2: Arbitrary 4-Component State

**Problem:** Prepare $|b\rangle = \frac{1}{\sqrt{30}}(|00\rangle + 2|01\rangle + 3|10\rangle + 4|11\rangle)$.

**Solution:**

Amplitudes: $b_0 = 1, b_1 = 2, b_2 = 3, b_3 = 4$ (normalized by $\sqrt{30}$)

Step 1: First qubit rotation
$$\sqrt{b_0^2 + b_1^2} = \sqrt{5}, \quad \sqrt{b_2^2 + b_3^2} = 5$$
$$\theta_0 = 2\arctan\left(\frac{5}{\sqrt{5}}\right) = 2\arctan(\sqrt{5}) \approx 2.21 \text{ rad}$$

$$|00\rangle \xrightarrow{R_y(\theta_0)} \frac{\sqrt{5}}{\sqrt{30}}|0\rangle + \frac{5}{\sqrt{30}}|1\rangle$$

Step 2: Controlled rotations for second qubit
For $|0\rangle$ subspace: ratio $b_0 : b_1 = 1 : 2$
$$\theta_{01} = 2\arctan(2) \approx 2.21 \text{ rad}$$

For $|1\rangle$ subspace: ratio $b_2 : b_3 = 3 : 4$
$$\theta_{11} = 2\arctan(4/3) \approx 1.85 \text{ rad}$$

**Circuit:**
```
|0⟩──R_y(2.21)──●──────────●────────
                │          │
|0⟩─────────────X──R_y(θ)──X──●──R_y(θ')──●──
                              │           │
                              X           X
```

(Simplified; actual circuit needs controlled-$R_y$ gates)

$$\boxed{\text{Total gates: } 2^n - 1 = 3 \text{ rotations}}$$

---

### Example 3: Expectation Value Readout

**Problem:** After HHL produces $|x\rangle$, estimate $\langle x|Z_1|x\rangle$ to precision $\epsilon = 0.01$.

**Solution:**

Step 1: Measurement strategy
Measure qubit 1 in Z basis repeatedly.
$$\langle Z_1 \rangle = P(0) - P(1)$$

Step 2: Required samples
For precision $\epsilon$ with confidence $1-\delta$:
$$N_{samples} = O\left(\frac{1}{\epsilon^2} \log(1/\delta)\right)$$

For $\epsilon = 0.01$, $\delta = 0.05$:
$$N_{samples} \approx \frac{4}{0.01^2} \cdot \log(20) \approx 120,000$$

Step 3: Total cost
- HHL preparation: $T_{HHL}$
- Repetitions: $120,000 \times T_{HHL}$

$$\boxed{T_{total} \approx 10^5 \times T_{HHL}}$$

This is still **polynomial in $1/\epsilon$**, maintaining quantum advantage (if state prep is efficient).

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** How many controlled rotations are needed to prepare an arbitrary 8-component state?

**Problem 1.2:** What is the measurement count for estimating $\langle x|x\rangle = 1$ to precision $0.001$?

**Problem 1.3:** For sparse $|b\rangle$ with $k = 10$ non-zero amplitudes in dimension $N = 10^6$, what is the preparation cost?

### Level 2: Intermediate Analysis

**Problem 2.1:** Design a circuit to prepare:
$$|b\rangle = \frac{1}{2}(|000\rangle + |010\rangle + |100\rangle + |111\rangle)$$
Minimize the number of CNOT gates.

**Problem 2.2:** Compare the total cost of HHL (including state prep and readout) vs conjugate gradient for:
- $N = 10^6$, $\kappa = 10$
- Input: arbitrary classical vector
- Output: $\langle x|Z|x\rangle$ to precision $0.01$

**Problem 2.3:** Prove that amplitude encoding of an arbitrary $N$-dimensional vector requires $\Omega(N)$ gates.

### Level 3: Challenging Problems

**Problem 3.1:** **qRAM Error Analysis**

A bucket-brigade qRAM has $N = 2^{20}$ memory cells. Each router has error probability $p = 10^{-6}$.
- What is the probability of error-free query?
- How many queries succeed on average before an error?
- What error rate per router is needed for 99% query success?

**Problem 3.2:** **Optimal Readout**

Given $|x\rangle$ and observable $M = \sum_i m_i |i\rangle\langle i|$:
- Show that estimating $\langle M \rangle$ to precision $\epsilon$ requires $O(1/\epsilon^2)$ samples
- If we need $\langle M_1 \rangle, \ldots, \langle M_k \rangle$, what is the total sample cost?
- When is tomography more efficient than individual measurements?

**Problem 3.3:** **Structured State Preparation**

For a Gaussian distribution $b_i \propto e^{-(i-\mu)^2/(2\sigma^2)}$ over $N = 2^n$ points:
- Design an $O(\text{poly}(n))$ preparation circuit
- Analyze the approximation error
- Compare to general $O(N)$ preparation

---

## Computational Lab

### State Preparation Implementation

```python
"""
Day 957: State Preparation and Readout
Amplitude encoding circuits and readout strategies.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.stats import norm


class AmplitudeEncoder:
    """
    Amplitude encoding of classical data into quantum states.
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize with classical data vector.

        Parameters:
        -----------
        data : ndarray
            Classical vector to encode
        """
        # Pad to power of 2
        n = int(np.ceil(np.log2(len(data))))
        self.N = 2 ** n
        self.n_qubits = n

        # Normalize
        padded = np.zeros(self.N, dtype=complex)
        padded[:len(data)] = data
        self.data = padded / np.linalg.norm(padded)

    def build_circuit(self) -> QuantumCircuit:
        """
        Build amplitude encoding circuit using recursive decomposition.

        Returns:
        --------
        QuantumCircuit : Encoding circuit
        """
        qc = QuantumCircuit(self.n_qubits)

        # Recursive encoding
        self._encode_recursive(qc, self.data, list(range(self.n_qubits)))

        return qc

    def _encode_recursive(self, qc: QuantumCircuit,
                          amplitudes: np.ndarray,
                          qubits: List[int]):
        """Recursively build amplitude encoding."""
        if len(amplitudes) == 1:
            return

        n = len(amplitudes)
        half = n // 2

        # Split amplitudes
        left = amplitudes[:half]
        right = amplitudes[half:]

        # Compute rotation angle for first qubit
        norm_left = np.linalg.norm(left)
        norm_right = np.linalg.norm(right)

        if norm_left + norm_right < 1e-10:
            return

        theta = 2 * np.arccos(norm_left / np.sqrt(norm_left**2 + norm_right**2))

        # Apply rotation to first qubit
        qc.ry(theta, qubits[0])

        # Recurse on subspaces
        if len(qubits) > 1:
            # Left subspace (control qubit = 0)
            if norm_left > 1e-10:
                left_normalized = left / norm_left

                # Apply X to flip control for "0" case
                qc.x(qubits[0])

                # Controlled operations on remaining qubits
                self._encode_controlled(qc, left_normalized, qubits[0], qubits[1:])

                qc.x(qubits[0])

            # Right subspace (control qubit = 1)
            if norm_right > 1e-10:
                right_normalized = right / norm_right
                self._encode_controlled(qc, right_normalized, qubits[0], qubits[1:])

    def _encode_controlled(self, qc: QuantumCircuit,
                           amplitudes: np.ndarray,
                           control: int,
                           targets: List[int]):
        """Apply controlled encoding."""
        if len(amplitudes) <= 1 or len(targets) == 0:
            return

        n = len(amplitudes)
        half = n // 2

        left = amplitudes[:half]
        right = amplitudes[half:]

        norm_left = np.linalg.norm(left)
        norm_right = np.linalg.norm(right)

        if norm_left + norm_right < 1e-10:
            return

        theta = 2 * np.arccos(norm_left / np.sqrt(norm_left**2 + norm_right**2))

        # Controlled rotation
        qc.cry(theta, control, targets[0])

        # Continue recursion
        if len(targets) > 1:
            if norm_left > 1e-10:
                qc.x(targets[0])
                self._encode_multi_controlled(qc, left/norm_left,
                                              [control, targets[0]], targets[1:])
                qc.x(targets[0])

            if norm_right > 1e-10:
                self._encode_multi_controlled(qc, right/norm_right,
                                              [control, targets[0]], targets[1:])

    def _encode_multi_controlled(self, qc: QuantumCircuit,
                                  amplitudes: np.ndarray,
                                  controls: List[int],
                                  targets: List[int]):
        """Multi-controlled encoding (simplified)."""
        # For small cases, use direct multi-controlled gates
        # For production, use decomposition
        if len(targets) == 0 or len(amplitudes) <= 1:
            return

        # Simplified: treat as single control (first control)
        self._encode_controlled(qc, amplitudes, controls[0], targets)

    def verify(self) -> float:
        """
        Verify encoding by comparing statevector to target.

        Returns:
        --------
        float : Fidelity between prepared and target states
        """
        qc = self.build_circuit()
        statevector = Statevector(qc)
        prepared = statevector.data

        fidelity = np.abs(np.vdot(prepared, self.data)) ** 2
        return fidelity


class StatePreparation:
    """Various state preparation methods."""

    @staticmethod
    def uniform_superposition(n_qubits: int) -> QuantumCircuit:
        """
        Prepare uniform superposition |+⟩^n.

        Complexity: O(n)
        """
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        return qc

    @staticmethod
    def sparse_state(indices: List[int], amplitudes: List[complex],
                    n_qubits: int) -> QuantumCircuit:
        """
        Prepare sparse state with k non-zero amplitudes.

        Complexity: O(k * n)
        """
        # Normalize
        norm = np.sqrt(sum(np.abs(a)**2 for a in amplitudes))
        amplitudes = [a/norm for a in amplitudes]

        qc = QuantumCircuit(n_qubits)

        # Use superposition and controlled operations
        # Simplified implementation for small k

        if len(indices) == 1:
            # Single basis state
            binary = format(indices[0], f'0{n_qubits}b')
            for i, bit in enumerate(binary[::-1]):
                if bit == '1':
                    qc.x(i)
        else:
            # Multiple basis states - use recursive approach
            full_amplitudes = np.zeros(2**n_qubits, dtype=complex)
            for idx, amp in zip(indices, amplitudes):
                full_amplitudes[idx] = amp

            encoder = AmplitudeEncoder(full_amplitudes)
            return encoder.build_circuit()

        return qc

    @staticmethod
    def gaussian_state(n_qubits: int, mu: float = 0.5,
                       sigma: float = 0.1) -> QuantumCircuit:
        """
        Prepare discretized Gaussian distribution.

        Parameters:
        -----------
        n_qubits : int
            Number of qubits (2^n points)
        mu : float
            Mean (as fraction of range [0,1])
        sigma : float
            Standard deviation (as fraction of range)
        """
        N = 2 ** n_qubits

        # Discretize Gaussian
        x = np.linspace(0, 1, N)
        amplitudes = norm.pdf(x, loc=mu, scale=sigma)
        amplitudes = np.sqrt(amplitudes / np.sum(amplitudes))

        encoder = AmplitudeEncoder(amplitudes)
        return encoder.build_circuit()


class ReadoutStrategies:
    """Methods for extracting information from quantum states."""

    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize with prepared state circuit.
        """
        self.circuit = circuit
        self.n_qubits = circuit.num_qubits

    def measure_observable(self, observable: str, shots: int = 10000) -> Tuple[float, float]:
        """
        Measure expectation value of Pauli observable.

        Parameters:
        -----------
        observable : str
            Pauli string, e.g., 'ZZ', 'XI', 'YZ'
        shots : int
            Number of measurement shots

        Returns:
        --------
        Tuple[float, float] : (expectation value, standard error)
        """
        if len(observable) != self.n_qubits:
            raise ValueError("Observable length must match qubit count")

        qc = self.circuit.copy()

        # Add measurement basis rotations
        for i, pauli in enumerate(observable[::-1]):
            if pauli == 'X':
                qc.h(i)
            elif pauli == 'Y':
                qc.sdg(i)
                qc.h(i)
            # Z needs no rotation

        # Measure
        qc.measure_all()

        # Run
        simulator = AerSimulator()
        result = simulator.run(transpile(qc, simulator), shots=shots).result()
        counts = result.get_counts()

        # Compute expectation value
        expectation = 0
        for bitstring, count in counts.items():
            # Eigenvalue is (-1)^(parity of measured bits where observable is non-I)
            parity = 0
            for i, pauli in enumerate(observable[::-1]):
                if pauli != 'I':
                    parity ^= int(bitstring[self.n_qubits - 1 - i])
            eigenvalue = (-1) ** parity
            expectation += eigenvalue * count / shots

        # Standard error
        std_error = np.sqrt((1 - expectation**2) / shots)

        return expectation, std_error

    def sample_distribution(self, shots: int = 10000) -> dict:
        """
        Sample from the state in computational basis.

        Returns:
        --------
        dict : Measurement outcome distribution
        """
        qc = self.circuit.copy()
        qc.measure_all()

        simulator = AerSimulator()
        result = simulator.run(transpile(qc, simulator), shots=shots).result()
        counts = result.get_counts()

        # Convert to probabilities
        distribution = {k: v/shots for k, v in counts.items()}
        return distribution

    def estimate_norm(self, shots: int = 10000) -> float:
        """
        Estimate the norm of the encoded vector.

        For amplitude encoded |x⟩, this is always 1.
        But can estimate relative magnitudes via controlled operations.
        """
        # The norm of a normalized quantum state is always 1
        # This is more relevant for unnormalized computations
        return 1.0


def resource_analysis():
    """Analyze state preparation resources."""
    sizes = [2, 4, 8, 16, 32, 64]

    gate_counts = []
    depths = []

    for N in sizes:
        # Random state
        data = np.random.randn(N) + 1j * np.random.randn(N)
        encoder = AmplitudeEncoder(data)
        qc = encoder.build_circuit()

        # Count resources
        transpiled = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'])
        gate_counts.append(sum(transpiled.count_ops().values()))
        depths.append(transpiled.depth())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(sizes, gate_counts, 'bo-', linewidth=2, markersize=8)
    axes[0].plot(sizes, [s-1 for s in sizes], 'r--', label='O(N) bound')
    axes[0].set_xlabel('Vector Size N', fontsize=12)
    axes[0].set_ylabel('Gate Count', fontsize=12)
    axes[0].set_title('Amplitude Encoding Gate Count', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sizes, depths, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Vector Size N', fontsize=12)
    axes[1].set_ylabel('Circuit Depth', fontsize=12)
    axes[1].set_title('Amplitude Encoding Circuit Depth', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('state_prep_resources.png', dpi=150, bbox_inches='tight')
    plt.show()

    return sizes, gate_counts, depths


def readout_demonstration():
    """Demonstrate readout strategies."""
    # Prepare a known state
    data = np.array([1, 2, 3, 4]) / np.sqrt(30)
    encoder = AmplitudeEncoder(data)
    qc = encoder.build_circuit()

    print("Readout Demonstration")
    print("=" * 50)
    print(f"Target amplitudes: {data}")

    # Verify encoding
    fidelity = encoder.verify()
    print(f"Encoding fidelity: {fidelity:.6f}")

    # Create readout analyzer
    readout = ReadoutStrategies(qc)

    # Measure observables
    observables = ['ZZ', 'ZI', 'IZ', 'XI']

    print("\nObservable Measurements:")
    for obs in observables:
        exp, err = readout.measure_observable(obs, shots=10000)
        print(f"  <{obs}> = {exp:.4f} +/- {err:.4f}")

    # Compute expected values analytically
    sv = Statevector(qc)
    print("\nAnalytic Values:")
    for obs in observables:
        # Construct Pauli matrix
        paulis = {'I': np.eye(2), 'X': np.array([[0,1],[1,0]]),
                  'Y': np.array([[0,-1j],[1j,0]]), 'Z': np.array([[1,0],[0,-1]])}
        mat = paulis[obs[0]]
        for p in obs[1:]:
            mat = np.kron(mat, paulis[p])
        exp = np.real(sv.data.conj() @ mat @ sv.data)
        print(f"  <{obs}> = {exp:.4f}")

    # Sampling
    print("\nSampling Distribution:")
    dist = readout.sample_distribution(shots=10000)
    for state, prob in sorted(dist.items()):
        print(f"  |{state}⟩: {prob:.4f} (target: {np.abs(data[int(state,2)])**2:.4f})")


def special_states_demo():
    """Demonstrate efficient preparation of special states."""
    print("\nSpecial State Preparation")
    print("=" * 50)

    # Uniform superposition
    n = 3
    qc_uniform = StatePreparation.uniform_superposition(n)
    print(f"\n1. Uniform superposition (n={n}):")
    print(f"   Gate count: {sum(qc_uniform.count_ops().values())}")
    print(f"   Circuit:\n{qc_uniform.draw(output='text')}")

    # Gaussian state
    qc_gaussian = StatePreparation.gaussian_state(3, mu=0.5, sigma=0.15)
    print(f"\n2. Gaussian state (n=3):")
    print(f"   Gate count: {sum(qc_gaussian.count_ops().values())}")

    # Verify Gaussian
    sv = Statevector(qc_gaussian)
    probs = np.abs(sv.data)**2

    plt.figure(figsize=(8, 4))
    plt.bar(range(8), probs, color='steelblue')
    plt.xlabel('Basis State')
    plt.ylabel('Probability')
    plt.title('Gaussian State Distribution')
    plt.tight_layout()
    plt.savefig('gaussian_state.png', dpi=150, bbox_inches='tight')
    plt.show()


def qram_simulation():
    """Simulate qRAM-like behavior (conceptually)."""
    print("\nqRAM Simulation (Conceptual)")
    print("=" * 50)

    # Classical memory
    memory = np.array([0.1, 0.4, 0.3, 0.2])
    N = len(memory)

    print(f"Classical memory: {memory}")

    # qRAM query: superposition over addresses
    n_addr = 2
    qc = QuantumCircuit(n_addr + 1)  # Address + data qubit

    # Put address in superposition
    qc.h([0, 1])

    # "Query" memory (simulated with controlled rotations)
    for i, val in enumerate(memory):
        # Encode address i
        binary = format(i, f'0{n_addr}b')

        # Apply X gates for zero bits
        for j, bit in enumerate(binary[::-1]):
            if bit == '0':
                qc.x(j)

        # Controlled rotation encoding value
        theta = 2 * np.arccos(np.sqrt(1 - val))
        from qiskit.circuit.library import RYGate
        mcry = RYGate(theta).control(n_addr)
        qc.append(mcry, [0, 1, 2])

        # Uncompute X gates
        for j, bit in enumerate(binary[::-1]):
            if bit == '0':
                qc.x(j)

    print("\nqRAM Query Circuit:")
    print(qc.draw(output='text'))

    # Verify
    sv = Statevector(qc)
    print("\nResulting state amplitudes:")
    for i in range(8):
        if np.abs(sv.data[i]) > 0.01:
            binary = format(i, '03b')
            print(f"  |{binary}⟩: {sv.data[i]:.4f}")


# Main execution
if __name__ == "__main__":
    print("Day 957: State Preparation & Readout")
    print("=" * 60)

    # Test amplitude encoding
    print("\n--- Amplitude Encoding Test ---")
    test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    encoder = AmplitudeEncoder(test_data)
    qc = encoder.build_circuit()
    fidelity = encoder.verify()
    print(f"8-component state encoding fidelity: {fidelity:.6f}")
    print(f"Circuit:\n{qc.draw(output='text')}")

    # Resource analysis
    print("\n--- Resource Analysis ---")
    resource_analysis()

    # Readout demonstration
    print("\n--- Readout Strategies ---")
    readout_demonstration()

    # Special states
    special_states_demo()

    # qRAM simulation
    qram_simulation()
```

---

## Summary

### Key Formulas

| Formula | Expression | Context |
|---------|------------|---------|
| Amplitude encoding cost | $O(N)$ or $O(2^n)$ | General case |
| qRAM query | $O(\log N)$ | Theoretical |
| Expectation readout | $O(1/\epsilon^2)$ | Independent of $N$ |
| Full tomography | $O(N/\epsilon^2)$ | Destroys advantage |

### End-to-End Complexity

$$T_{total} = T_{prep} + T_{HHL} + T_{readout}$$

Quantum advantage requires:
- $T_{prep} = O(\text{poly}(\log N))$ — quantum input or structured data
- $T_{readout} = O(\text{poly}(\log N))$ — limited classical output

### Key Insights

1. **State preparation is often the bottleneck** — $O(N)$ loading kills advantage
2. **qRAM is theoretical** — no working implementations exist
3. **Structured data enables efficient encoding** — uniform, Gaussian, sparse
4. **Readout determines usefulness** — expectation values are efficient
5. **Quantum-to-quantum pipelines preserve advantage**

---

## Daily Checklist

- [ ] I understand the state preparation bottleneck
- [ ] I can design amplitude encoding circuits
- [ ] I know qRAM's promise and challenges
- [ ] I can implement efficient readout strategies
- [ ] I distinguish useful vs problematic I/O scenarios
- [ ] I can evaluate end-to-end HHL complexity

---

## Preview: Day 958

Tomorrow we explore **Dequantization and Classical Competition**:

- Tang's 2018 breakthrough algorithm
- Quantum-inspired classical algorithms
- When classical methods match HHL
- The refined landscape of quantum advantage
- Implications for quantum machine learning

The dequantization revolution reshapes our understanding of when HHL actually provides advantage.

---

*Day 957 of 2184 | Week 137 of 312 | Month 35 of 72*

*"The devil of quantum advantage is in the details of input and output."*
