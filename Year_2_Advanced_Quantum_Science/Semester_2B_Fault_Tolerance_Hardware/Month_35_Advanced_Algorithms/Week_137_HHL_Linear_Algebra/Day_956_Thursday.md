# Day 956: HHL Circuit Implementation

## Week 137, Day 4 | Month 35: Advanced Quantum Algorithms

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Circuit components and Hamiltonian simulation |
| Afternoon | 2.5 hours | Problem solving: Gate synthesis and optimization |
| Evening | 2 hours | Computational lab: Qiskit HHL implementation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Implement Hamiltonian simulation** circuits for $e^{iAt}$
2. **Construct controlled rotation** circuits with quantum arithmetic
3. **Design the complete HHL circuit** with proper ancilla management
4. **Analyze gate counts and circuit depth** for resource estimation
5. **Implement HHL in Qiskit** for small-scale demonstrations
6. **Identify optimization opportunities** in HHL circuits

---

## Core Content

### 1. HHL Circuit Architecture

The complete HHL circuit consists of several interconnected components:

```
                    ┌───────────┐     ┌─────────────────┐     ┌───────────┐
|0⟩^n ──────────────┤    QPE    ├─────┤ Eigenvalue Reg. ├─────┤   QPE⁻¹   ├───
                    │           │     │                 │     │           │
|b⟩   ──────────────┤  (Ctrl-U) ├─────┤   Eigenstates   ├─────┤  (Ctrl-U) ├───
                    └───────────┘     └────────┬────────┘     └───────────┘
                                               │
                                      ┌────────▼────────┐
|0⟩_anc ──────────────────────────────┤ Ctrl-R_y(λ)    ├──────────────────── M
                                      └─────────────────┘
```

#### Register Allocation

| Register | Qubits | Purpose |
|----------|--------|---------|
| Eigenvalue | $n$ | Store QPE phase estimates |
| State | $\log_2(N)$ | Encode $|b\rangle$ and $|x\rangle$ |
| Ancilla | 1 | Flag successful inversion |
| Work | Variable | Arithmetic operations |

### 2. Hamiltonian Simulation Methods

For QPE, we need controlled-$e^{iAt}$ operations. Several methods exist:

#### Trotter-Suzuki Product Formula

For $H = \sum_k H_k$ (sum of local terms):
$$e^{iHt} \approx \left(\prod_k e^{iH_k t/r}\right)^r + O(t^2/r)$$

**First-order Trotter:**
$$\boxed{e^{iHt} \approx \left(e^{iH_1 t/r} e^{iH_2 t/r} \cdots e^{iH_m t/r}\right)^r}$$

**Error:** $O(m^2 t^2 / r)$

**Second-order (symmetric) Trotter:**
$$e^{iHt} \approx \left(e^{iH_1 t/(2r)} e^{iH_2 t/r} \cdots e^{iH_1 t/(2r)}\right)^r$$

**Error:** $O(m^3 t^3 / r^2)$

#### Gate Complexity for Sparse Hamiltonians

For a sparse matrix $A$ with sparsity $s$ (non-zeros per row):

$$T_{Trotter} = O\left(\frac{s^2 t^2}{\epsilon}\right)$$

using first-order product formula with $r = O(s^2 t^2/\epsilon)$ repetitions.

### 3. Controlled Unitary Implementation

#### Structure of Controlled-$e^{iAt}$

```
     ┌─────────────────────┐
─────┤                     ├─────
     │  Controlled-e^{iAt} │
●────┤                     ├─────
     └─────────────────────┘
```

Implement using controlled versions of each gate in the Trotter decomposition.

#### Controlled Rotation Gates

For a term $e^{i\theta Z}$:
$$\text{Controlled-}R_z(\theta): |0\rangle|ψ\rangle \to |0\rangle|ψ\rangle, \quad |1\rangle|ψ\rangle \to |1\rangle R_z(\theta)|ψ\rangle$$

Implementation:
```
     ┌───┐     ┌───────────┐     ┌───┐
─────┤   ├──●──┤           ├──●──┤   ├─────
     │   │  │  │           │  │  │   │
─────┤   ├──┼──┤ R_z(θ/2)  ├──┼──┤   ├─────
     └───┘  │  └───────────┘  │  └───┘
            ▼                 ▼
         CNOT              CNOT
```

### 4. Eigenvalue-Controlled Rotation

The critical step: rotate ancilla by angle $\theta(\lambda) = 2\arcsin(C/\lambda)$.

#### Binary Representation of Eigenvalue

After QPE, eigenvalue stored as $n$-bit integer:
$$|\tilde{\lambda}\rangle = |k\rangle \quad \text{where} \quad \tilde{\lambda} = \frac{2\pi k}{\tau \cdot 2^n}$$

#### Rotation Circuit

**Approach 1: Lookup Table**

Precompute rotation angles for each possible $k$ value:
```
for k in range(2^n):
    θ_k = 2 * arcsin(C / λ(k))

|k⟩|0⟩ → |k⟩(cos(θ_k/2)|0⟩ + sin(θ_k/2)|1⟩)
```

Implementation via controlled rotations:
$$R_y^{(k)}(\theta_k) = \prod_{j: k_j=1} R_y^{(j)}(\theta_k^{(j)})$$

**Gate count:** $O(2^n)$ in worst case, but can be optimized.

#### Approach 2: Quantum Arithmetic

Compute $\arcsin(C/\lambda)$ using reversible arithmetic:
1. Compute reciprocal: $\lambda \to 1/\lambda$
2. Multiply: $1/\lambda \to C/\lambda$
3. Compute arcsin via polynomial approximation
4. Controlled rotation by computed angle

**Gate count:** $O(\text{poly}(n))$ but with large constants.

### 5. Complete Circuit for 2×2 System

For a 2×2 diagonal matrix $A = \begin{pmatrix} \lambda_0 & 0 \\ 0 & \lambda_1 \end{pmatrix}$:

```
     ┌───┐                                        ┌───────┐
q_0: ┤ H ├──●────────────────────●────────────────┤ QFT⁻¹ ├─── (eigenvalue bit 0)
     └───┘  │                    │                └───┬───┘
            │                    │                    │
     ┌───┐  │  ┌───┐             │  ┌───┐             │
q_1: ┤ H ├──┼──┤   ├──●──────────┼──┤   ├──●──────────┤───────── (eigenvalue bit 1)
     └───┘  │  └───┘  │          │  └───┘  │          │
            │         │          │         │          │
     ┌───┐  │         │  ┌───┐   │         │  ┌───┐   │
q_2: ┤ X ├──■─────────■──┤   ├───■─────────■──┤   ├───┼───────── (state qubit)
     └───┘  │         │  └───┘   │         │  └───┘   │
          U^1       U^2        U^1       U^2         │
                                                      │
            ┌─────────────────────────────────────────┴─────┐
q_3: ───────┤     Controlled-R_y(arcsin(C/λ))              ├── M (ancilla)
            └───────────────────────────────────────────────┘
```

### 6. Resource Analysis

#### Qubit Count

$$\boxed{Q_{total} = n + \log_2(N) + 1 + Q_{work}}$$

where:
- $n$ = QPE precision bits
- $\log_2(N)$ = state register
- $1$ = success ancilla
- $Q_{work}$ = arithmetic workspace

#### Gate Count

| Component | Gate Count |
|-----------|------------|
| QPE Hadamards | $n$ |
| Controlled-$U^{2^k}$ | $n \times T_{sim}$ |
| Inverse QFT | $O(n^2)$ |
| Controlled rotation | $O(2^n)$ or $O(\text{poly}(n))$ |
| Uncomputation | Same as forward |

**Total:**
$$T_{gates} = O(n \cdot T_{sim} + n^2 + T_{rot})$$

#### Circuit Depth

For parallel execution where possible:
$$D_{total} = O(T_{sim} + n + D_{rot})$$

### 7. Optimization Techniques

#### Approximate QFT

Replace exact QFT with approximate version ignoring small rotations:
- Keep only rotations $> 2^{-k}$ for cutoff $k$
- Reduces gate count from $O(n^2)$ to $O(n \log n)$
- Error $O(n/2^k)$

#### Iterative Phase Estimation

Replace parallel QPE with sequential single-ancilla approach:
- Uses 1 ancilla qubit instead of $n$
- Depth increases but qubit count decreases
- Useful for near-term devices

#### Eigenvalue Truncation

For controlled rotation, only implement for eigenvalues above threshold:
- Ignore $\lambda < \lambda_{cutoff}$
- Reduces circuit complexity
- Introduces approximation error

---

## Worked Examples

### Example 1: Trotter Steps for 2×2 Hamiltonian

**Problem:** Compute Trotter decomposition for $H = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 2 \end{pmatrix}$ with $t = 0.1$ and first-order Trotter.

**Solution:**

Step 1: Decompose $H$ into Pauli terms
$$H = \frac{3}{2}I + \frac{1}{2}X - \frac{1}{2}Z$$

So:
- $H_1 = \frac{1}{2}X$
- $H_2 = -\frac{1}{2}Z$
- $H_0 = \frac{3}{2}I$ (global phase, can ignore)

Step 2: First-order Trotter ($r=1$)
$$e^{iHt} \approx e^{iH_1 t} \cdot e^{iH_2 t} = e^{i(1/2)Xt} \cdot e^{i(-1/2)Zt}$$

$$= R_x(-t) \cdot R_z(t) = R_x(-0.1) \cdot R_z(0.1)$$

Step 3: Circuit
```
──R_x(-0.1)──R_z(0.1)──
```

Step 4: Error analysis
$$\|e^{iHt} - e^{iH_1 t}e^{iH_2 t}\| \leq \frac{t^2}{2}\|[H_1, H_2]\| = \frac{0.01}{2} \cdot \|0.5i Y\| = 0.0025$$

---

### Example 2: Controlled Rotation Circuit

**Problem:** Design circuit for controlled-$R_y(\theta)$ where $\theta$ is encoded in a 2-bit register.

**Solution:**

The 2-bit register can encode 4 values: $|00\rangle, |01\rangle, |10\rangle, |11\rangle$.

Corresponding angles: $\theta_0, \theta_1, \theta_2, \theta_3$.

Step 1: Decompose into single-bit controlled rotations

$$R_y(\theta) = R_y\left(\frac{\theta_0 + \theta_1 + \theta_2 + \theta_3}{4}\right) \cdot C^0 R_y(\Delta_0) \cdot C^1 R_y(\Delta_1) \cdot C^{01} R_y(\Delta_{01})$$

where the $\Delta$ terms encode differences.

Step 2: Gray code implementation

More efficient approach using Gray code ordering:
```
     ┌─────────────────┐
q_0: ┤                 ├──●─────────────●──
     │                 │  │             │
q_1: ┤                 ├──┼──●───────●──┼──
     │                 │  │  │       │  │
anc: ┤ R_y(θ_base)    ├──X──X──R_y──X──X──
     └─────────────────┘
```

Step 3: Full circuit (4 controlled rotations for 2-bit control):
```
     ●───────────●─────────────────────
     │           │
     ├───●───────┼────●────────────────
     │   │       │    │
─────X───X──R₁───X────X──R₂──(...)────
```

$$\boxed{\text{Gate count: } O(2^n) \text{ rotations for } n\text{-bit control}}$$

---

### Example 3: Complete 2-Qubit HHL

**Problem:** Write the complete HHL circuit for $A = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$, $|b\rangle = |+\rangle$.

**Solution:**

Step 1: Parameters
- Eigenvalues: $\lambda_0 = 1$, $\lambda_1 = 2$
- Condition number: $\kappa = 2$
- Choose $C = 0.9$ (less than $\lambda_{min} = 1$)

Step 2: Rotation angles
- For $\lambda_0 = 1$: $\theta_0 = 2\arcsin(0.9/1) = 2\arcsin(0.9) \approx 2.24$ rad
- For $\lambda_1 = 2$: $\theta_1 = 2\arcsin(0.9/2) = 2\arcsin(0.45) \approx 0.93$ rad

Step 3: Circuit (using 1 QPE ancilla for simplicity)

```
     ┌───┐               ┌───────┐
q_0: ┤ H ├──●────────────┤ H     ├──●─────────────●──
     └───┘  │            └───────┘  │             │
            │                       │             │
     ┌───┐  │                       │             │
q_1: ┤ H ├──U(t)────────────────────┼─────────────┼──
     └───┘                          │             │
                                    │             │
     ┌────────────────────────────┬─┴─┬─────────┬─┴─┐
q_2: ┤ 0                          │ X │ R_y(θ)  │ X ├── M
     └────────────────────────────┴───┴─────────┴───┘
```

where:
- $U(t) = e^{iAt} = \begin{pmatrix} e^{it} & 0 \\ 0 & e^{2it} \end{pmatrix}$ (controlled)
- The controlled-$R_y$ applies $\theta_0$ when control is $|0\rangle$ and $\theta_1$ when $|1\rangle$

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** How many Trotter steps are needed for $e^{iHt}$ with $\|H\| = 10$, $t = 0.5$, and error $\epsilon = 0.01$?

**Problem 1.2:** Calculate the rotation angle $\theta$ for $\lambda = 3$ with $C = 1$.

**Problem 1.3:** What is the qubit count for HHL on a 256×256 matrix with 8-bit QPE precision?

### Level 2: Intermediate Analysis

**Problem 2.1:** Design a circuit for controlled-$R_z(\theta)$ using only CNOT and single-qubit gates.

**Problem 2.2:** Compare gate counts for:
- Lookup table approach (8-bit eigenvalue)
- Newton-Raphson reciprocal (8-bit precision)

**Problem 2.3:** Analyze the depth-width tradeoff between standard QPE and iterative phase estimation for HHL.

### Level 3: Challenging Problems

**Problem 3.1:** **Trotter Error Optimization**

For a Hamiltonian $H = H_1 + H_2 + H_3$ with known commutators:
- Derive the optimal ordering of terms to minimize Trotter error
- When should second-order Trotter be preferred?

**Problem 3.2:** **Controlled Rotation Synthesis**

Prove that any eigenvalue-controlled rotation can be implemented with:
$$O(n \cdot 2^n)$$ T gates
using the decomposition into controlled-$R_y$ gates.

**Problem 3.3:** **Complete Resource Estimation**

For HHL on a 1024×1024 sparse matrix ($s=10$) with:
- $\kappa = 50$
- $\epsilon = 0.01$
- 12-bit QPE precision

Estimate:
- Total qubit count (including work qubits)
- Total gate count
- Circuit depth
- Success probability

---

## Computational Lab

### Qiskit HHL Implementation

```python
"""
Day 956: HHL Circuit Implementation in Qiskit
Complete implementation for small-scale demonstration.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class HHLCircuit:
    """
    Complete HHL circuit implementation for small systems.
    """

    def __init__(self, A: np.ndarray, num_qpe_qubits: int = 3):
        """
        Initialize HHL circuit builder.

        Parameters:
        -----------
        A : ndarray
            2x2 Hermitian matrix to invert
        num_qpe_qubits : int
            Number of qubits for QPE precision
        """
        if A.shape != (2, 2):
            raise ValueError("Currently only supports 2x2 matrices")
        if not np.allclose(A, A.conj().T):
            raise ValueError("Matrix must be Hermitian")

        self.A = A
        self.n_qpe = num_qpe_qubits

        # Eigendecomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(A)
        self.lambda_min = np.min(np.abs(self.eigenvalues))
        self.lambda_max = np.max(np.abs(self.eigenvalues))

        # Simulation time (normalized)
        self.t = 2 * np.pi / (1.2 * self.lambda_max)

        # Rotation constant
        self.C = 0.5 * self.lambda_min

    def _create_controlled_unitary(self, qc: QuantumCircuit,
                                    control: int, target: int,
                                    power: int):
        """Add controlled U^power to circuit."""
        # For diagonal matrix, this is simple
        # For general matrix, need Trotter decomposition

        # Compute U^power = exp(i * A * t * power)
        U_power = expm(1j * self.A * self.t * power)

        # Decompose into gates
        # For 2x2, we can use direct decomposition
        if np.allclose(self.A, np.diag(np.diag(self.A))):
            # Diagonal case: controlled phase
            phase_0 = self.eigenvalues[0] * self.t * power
            phase_1 = self.eigenvalues[1] * self.t * power

            # Apply controlled phases
            qc.cp(phase_1 - phase_0, control, target)
            qc.p(phase_0, control)  # Global phase on control
        else:
            # General case: use Qiskit's unitary gate
            from qiskit.circuit.library import UnitaryGate
            controlled_U = UnitaryGate(U_power).control(1)
            qc.append(controlled_U, [control, target])

    def _add_qpe(self, qc: QuantumCircuit,
                 qpe_register: List[int], state_qubit: int):
        """Add QPE circuit."""
        # Hadamard on all QPE qubits
        for i in qpe_register:
            qc.h(i)

        # Controlled unitary operations
        for j, qubit in enumerate(qpe_register):
            power = 2 ** j
            self._create_controlled_unitary(qc, qubit, state_qubit, power)

        # Inverse QFT
        qft_inv = QFT(len(qpe_register), inverse=True)
        qc.append(qft_inv, qpe_register)

    def _add_controlled_rotation(self, qc: QuantumCircuit,
                                  qpe_register: List[int],
                                  ancilla: int):
        """Add eigenvalue-controlled rotation."""
        # For each possible eigenvalue, compute and apply rotation
        n = len(qpe_register)

        # Compute rotation angles for each discretized eigenvalue
        for k in range(2**n):
            # Reconstruct eigenvalue from binary
            phase = k / (2**n)
            if phase > 0.5:
                phase -= 1  # Handle negative eigenvalues

            # Eigenvalue (avoid division by zero)
            lam = phase * 2 * np.pi / self.t if phase != 0 else 0.01

            if abs(lam) > self.C:
                # Compute rotation angle
                theta = 2 * np.arcsin(self.C / abs(lam))
            else:
                theta = np.pi  # Maximum rotation for small eigenvalues

            # Apply controlled rotation
            # Create binary control pattern
            binary = format(k, f'0{n}b')

            # Apply X gates to flip controls for 0s
            for j, bit in enumerate(binary[::-1]):
                if bit == '0':
                    qc.x(qpe_register[j])

            # Multi-controlled Ry
            if n == 1:
                qc.cry(theta, qpe_register[0], ancilla)
            else:
                # Use MCRYGate for multiple controls
                from qiskit.circuit.library import RYGate
                mcry = RYGate(theta).control(n)
                qc.append(mcry, qpe_register + [ancilla])

            # Uncompute X gates
            for j, bit in enumerate(binary[::-1]):
                if bit == '0':
                    qc.x(qpe_register[j])

    def _add_inverse_qpe(self, qc: QuantumCircuit,
                          qpe_register: List[int], state_qubit: int):
        """Add inverse QPE to uncompute eigenvalue register."""
        # QFT (not inverse, since we're reversing)
        qft = QFT(len(qpe_register), inverse=False)
        qc.append(qft, qpe_register)

        # Inverse controlled unitaries
        for j, qubit in enumerate(reversed(qpe_register)):
            power = 2 ** (len(qpe_register) - 1 - j)
            # Apply inverse = conjugate transpose
            self._create_controlled_unitary(qc, qubit, state_qubit, -power)

        # Hadamard (self-inverse)
        for i in qpe_register:
            qc.h(i)

    def build_circuit(self, b_state: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Build complete HHL circuit.

        Parameters:
        -----------
        b_state : ndarray, optional
            Initial state |b⟩. Default is |0⟩.

        Returns:
        --------
        QuantumCircuit : Complete HHL circuit
        """
        # Registers
        qpe_reg = QuantumRegister(self.n_qpe, 'qpe')
        state_reg = QuantumRegister(1, 'state')
        ancilla_reg = QuantumRegister(1, 'ancilla')
        classical_reg = ClassicalRegister(1, 'result')

        qc = QuantumCircuit(qpe_reg, state_reg, ancilla_reg, classical_reg)

        # Initialize |b⟩
        if b_state is not None:
            b_normalized = b_state / np.linalg.norm(b_state)
            qc.initialize(b_normalized, state_reg)
        else:
            # Default: |+⟩ state
            qc.h(state_reg[0])

        qc.barrier(label='State Prep')

        # QPE
        qpe_qubits = list(range(self.n_qpe))
        self._add_qpe(qc, qpe_qubits, self.n_qpe)

        qc.barrier(label='QPE')

        # Controlled rotation
        self._add_controlled_rotation(qc, qpe_qubits, self.n_qpe + 1)

        qc.barrier(label='Ctrl-Ry')

        # Inverse QPE
        self._add_inverse_qpe(qc, qpe_qubits, self.n_qpe)

        qc.barrier(label='QPE†')

        # Measure ancilla
        qc.measure(ancilla_reg[0], classical_reg[0])

        return qc

    def run(self, b_state: Optional[np.ndarray] = None,
            shots: int = 10000) -> dict:
        """
        Run HHL circuit and analyze results.
        """
        qc = self.build_circuit(b_state)

        # Simulate
        simulator = AerSimulator()
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled, shots=shots).result()
        counts = result.get_counts()

        # Analyze success
        success_count = counts.get('1', 0)
        success_rate = success_count / shots

        # Get statevector for successful outcomes
        # (In real implementation, we'd post-select)

        return {
            'circuit': qc,
            'counts': counts,
            'success_rate': success_rate,
            'shots': shots
        }


def expm(M):
    """Matrix exponential."""
    from scipy.linalg import expm as scipy_expm
    return scipy_expm(M)


class SimplifiedHHL:
    """
    Simplified HHL for 2x2 diagonal matrices.

    This version is more educational and easier to understand.
    """

    def __init__(self, lambda_0: float, lambda_1: float):
        """
        Initialize with diagonal eigenvalues.
        """
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.C = min(abs(lambda_0), abs(lambda_1)) * 0.5

    def build_circuit(self, n_qpe: int = 2) -> QuantumCircuit:
        """Build simplified HHL circuit."""
        qc = QuantumCircuit(n_qpe + 2, 1)  # QPE + state + ancilla

        # Initialize state qubit in |+⟩
        qc.h(n_qpe)

        # QPE section
        # Hadamard on QPE qubits
        for i in range(n_qpe):
            qc.h(i)

        # Controlled unitaries (simplified: just controlled phases)
        t = 2 * np.pi / (1.5 * max(abs(self.lambda_0), abs(self.lambda_1)))

        for j in range(n_qpe):
            # Controlled phase based on eigenvalue
            phase_diff = (self.lambda_1 - self.lambda_0) * t * (2**j)
            qc.cp(phase_diff, j, n_qpe)

        # Inverse QFT
        qft_inv = QFT(n_qpe, inverse=True)
        qc.append(qft_inv, range(n_qpe))

        qc.barrier()

        # Controlled rotation
        # For each QPE outcome, apply appropriate rotation
        theta_0 = 2 * np.arcsin(self.C / abs(self.lambda_0))
        theta_1 = 2 * np.arcsin(self.C / abs(self.lambda_1))

        # Simplified: average rotation plus correction
        avg_theta = (theta_0 + theta_1) / 2
        diff_theta = (theta_1 - theta_0) / 2

        qc.ry(avg_theta, n_qpe + 1)

        # Controlled correction based on QPE outcome
        for j in range(n_qpe):
            qc.cry(diff_theta / (2**j), j, n_qpe + 1)

        qc.barrier()

        # Inverse QPE (simplified)
        qft = QFT(n_qpe, inverse=False)
        qc.append(qft, range(n_qpe))

        for j in range(n_qpe):
            phase_diff = -(self.lambda_1 - self.lambda_0) * t * (2**j)
            qc.cp(phase_diff, j, n_qpe)

        for i in range(n_qpe):
            qc.h(i)

        # Measure ancilla
        qc.measure(n_qpe + 1, 0)

        return qc


def analyze_hhl_resources():
    """Analyze HHL circuit resources."""
    matrices = [
        np.array([[1, 0], [0, 2]]),
        np.array([[1, 0.3], [0.3, 2]]),
        np.array([[1, 0.5], [0.5, 3]])
    ]

    results = []

    for i, A in enumerate(matrices):
        for n_qpe in [2, 3, 4, 5]:
            hhl = HHLCircuit(A, n_qpe)
            qc = hhl.build_circuit()

            # Transpile to count gates
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import Unroller

            transpiled = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'])

            results.append({
                'matrix': i,
                'n_qpe': n_qpe,
                'total_qubits': qc.num_qubits,
                'depth': transpiled.depth(),
                'gate_count': sum(transpiled.count_ops().values()),
                'cx_count': transpiled.count_ops().get('cx', 0)
            })

    # Display results
    print("\nHHL Resource Analysis")
    print("=" * 60)
    print(f"{'Matrix':<8} {'QPE bits':<10} {'Qubits':<8} {'Depth':<8} {'Gates':<8} {'CNOTs':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['matrix']:<8} {r['n_qpe']:<10} {r['total_qubits']:<8} "
              f"{r['depth']:<8} {r['gate_count']:<8} {r['cx_count']:<8}")

    return results


def visualize_hhl_circuit():
    """Create and visualize an HHL circuit."""
    # Simple diagonal matrix
    A = np.array([[1, 0], [0, 2]])

    hhl = HHLCircuit(A, num_qpe_qubits=2)
    qc = hhl.build_circuit()

    print("HHL Circuit for A = diag(1, 2)")
    print("=" * 50)
    print(qc.draw(output='text', fold=80))

    # Save circuit diagram
    fig = qc.draw(output='mpl', fold=40)
    fig.savefig('hhl_circuit.png', dpi=150, bbox_inches='tight')
    plt.close()

    return qc


def run_hhl_demo():
    """Run complete HHL demonstration."""
    print("HHL Circuit Implementation Demo")
    print("=" * 60)

    # Test matrix
    A = np.array([[2, 0], [0, 4]])
    b = np.array([1, 1]) / np.sqrt(2)

    print(f"\nMatrix A:\n{A}")
    print(f"Vector b: {b}")

    # Classical solution
    x_classical = np.linalg.solve(A, b)
    x_normalized = x_classical / np.linalg.norm(x_classical)
    print(f"\nClassical solution (normalized): {x_normalized}")

    # HHL solution
    hhl = HHLCircuit(A, num_qpe_qubits=3)
    result = hhl.run(b, shots=10000)

    print(f"\nHHL Results:")
    print(f"  Success rate: {result['success_rate']:.2%}")
    print(f"  Measurement counts: {result['counts']}")

    # Draw circuit
    print("\nCircuit:")
    print(result['circuit'].draw(output='text', fold=60))


def trotter_decomposition_demo():
    """Demonstrate Trotter decomposition for Hamiltonian simulation."""
    print("\nTrotter Decomposition Demo")
    print("=" * 60)

    # Non-diagonal Hamiltonian
    H = np.array([[1, 0.5], [0.5, 2]])

    # Decompose into Pauli terms
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # H = a*I + b*X + c*Y + d*Z
    a = np.trace(H @ I) / 2
    b = np.trace(H @ X) / 2
    c = np.trace(H @ Y) / 2
    d = np.trace(H @ Z) / 2

    print(f"H = {a.real:.2f}*I + {b.real:.2f}*X + {c.real:.2f}*Y + {d.real:.2f}*Z")

    # Compare exact vs Trotter
    t = 0.5
    exact = expm(1j * H * t)

    # First-order Trotter
    U_I = expm(1j * a * t * I)
    U_X = expm(1j * b * t * X)
    U_Z = expm(1j * d * t * Z)
    trotter_1 = U_I @ U_X @ U_Z

    error = np.linalg.norm(exact - trotter_1)
    print(f"\nFirst-order Trotter error (r=1): {error:.6f}")

    # Higher-order
    r = 10
    U_X_r = expm(1j * b * t / r * X)
    U_Z_r = expm(1j * d * t / r * Z)

    trotter_r = np.eye(2)
    for _ in range(r):
        trotter_r = trotter_r @ U_X_r @ U_Z_r

    error_r = np.linalg.norm(exact - trotter_r)
    print(f"First-order Trotter error (r={r}): {error_r:.6f}")


# Main execution
if __name__ == "__main__":
    # Run demonstrations
    run_hhl_demo()

    print("\n" + "=" * 60)
    visualize_hhl_circuit()

    print("\n" + "=" * 60)
    analyze_hhl_resources()

    print("\n" + "=" * 60)
    trotter_decomposition_demo()
```

---

## Summary

### Key Formulas

| Formula | Expression | Context |
|---------|------------|---------|
| Trotter error | $O(m^2 t^2 / r)$ | First-order |
| QPE gates | $O(n \cdot T_{sim})$ | Controlled unitaries |
| Controlled rotation | $O(2^n)$ or $O(\text{poly}(n))$ | Lookup vs arithmetic |
| Total qubits | $n + \log_2(N) + O(1)$ | Register count |

### Circuit Components

1. **Hamiltonian simulation:** Trotter decomposition for $e^{iAt}$
2. **Controlled unitaries:** Apply $U^{2^k}$ controlled by QPE qubits
3. **Eigenvalue extraction:** Inverse QFT on QPE register
4. **Controlled rotation:** $R_y(2\arcsin(C/\lambda))$ based on eigenvalue
5. **Uncomputation:** Reverse QPE to disentangle registers

### Key Insights

1. **Trotter steps dominate** for non-diagonal matrices
2. **Controlled rotation is the bottleneck** for high-precision QPE
3. **Trade-offs exist** between qubit count and circuit depth
4. **Approximations help** (approximate QFT, eigenvalue truncation)
5. **Small systems are demonstrable** but don't show advantage

---

## Daily Checklist

- [ ] I understand Trotter decomposition for Hamiltonian simulation
- [ ] I can implement controlled unitaries for QPE
- [ ] I know how to construct eigenvalue-controlled rotations
- [ ] I can analyze gate counts and circuit depth
- [ ] I implemented HHL in Qiskit
- [ ] I understand the resource scaling

---

## Preview: Day 957

Tomorrow we tackle the **State Preparation and Readout** challenges:

- Amplitude encoding: loading classical $b$ into $|b\rangle$
- qRAM requirements and alternatives
- Readout: extracting useful information from $|x\rangle$
- Expectation values vs full state tomography
- The "fine print" of HHL complexity

These input/output challenges often determine whether HHL provides practical advantage.

---

*Day 956 of 2184 | Week 137 of 312 | Month 35 of 72*

*"A quantum algorithm is only as good as its implementation—and HHL has many moving parts."*
