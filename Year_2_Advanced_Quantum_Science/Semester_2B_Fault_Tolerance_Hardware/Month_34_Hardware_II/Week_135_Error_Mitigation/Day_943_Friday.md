# Day 943: Measurement Error Mitigation - Confusion Matrices and M3 Method

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Confusion matrix theory, characterization protocols |
| Afternoon | 2 hours | Matrix inversion, regularization, scalability |
| Evening | 2 hours | Computational lab - M3 implementation |

## Learning Objectives

By the end of this day, you will be able to:

1. **Characterize measurement errors** - Build confusion matrices from calibration data
2. **Apply matrix inversion** - Correct probability distributions using inverse confusion matrix
3. **Handle ill-conditioning** - Use regularization techniques for noisy matrices
4. **Implement M3 method** - Apply matrix-free measurement mitigation for large systems
5. **Scale to many qubits** - Use tensor product structure for efficient mitigation
6. **Integrate with other techniques** - Combine measurement mitigation with ZNE/PEC

## Core Content

### 1. Measurement Error Model

#### 1.1 Single-Qubit Confusion Matrix

Measurement errors are described by conditional probabilities:

$$\boxed{M = \begin{pmatrix} P(0|0) & P(0|1) \\ P(1|0) & P(1|1) \end{pmatrix}}$$

where $P(m|s)$ is the probability of measuring outcome $m$ given the true state $s$.

**Perfect measurement**: $M = I$ (identity matrix)

**Typical values** (superconducting qubits):
- $P(0|0) \approx 0.97-0.99$
- $P(1|1) \approx 0.95-0.98$
- Asymmetric: $P(1|0) \neq P(0|1)$ due to $T_1$ decay

#### 1.2 Multi-Qubit Confusion Matrix

For $n$ qubits, the confusion matrix has size $2^n \times 2^n$:

$$M_{ij} = P(\text{measure } i | \text{state } j)$$

The relationship between true and measured distributions:

$$\boxed{\mathbf{p}_{\text{noisy}} = M \cdot \mathbf{p}_{\text{ideal}}}$$

where $\mathbf{p}$ are probability vectors over all $2^n$ bitstrings.

#### 1.3 Tensor Product Structure

If measurement errors are independent across qubits:

$$\boxed{M = M_1 \otimes M_2 \otimes \cdots \otimes M_n}$$

This reduces characterization from $O(4^n)$ to $O(n)$ parameters.

**Correlated errors**: Require characterization of joint error probabilities, breaking tensor product structure.

### 2. Calibration Protocols

#### 2.1 Complete Calibration

For full $2^n \times 2^n$ matrix:
1. Prepare each computational basis state $|j\rangle$
2. Measure many shots
3. Record distribution of outcomes
4. Column $j$ of $M$ is the outcome distribution for state $|j\rangle$

**Cost**: $2^n$ different state preparations, exponential in $n$.

#### 2.2 Tensor Product Calibration

For independent errors:
1. Prepare $|0\rangle^{\otimes n}$ and measure: gives $P(0|0)$ for each qubit
2. Prepare $|1\rangle^{\otimes n}$ and measure: gives $P(1|1)$ for each qubit
3. Construct per-qubit matrices

**Cost**: Only 2 state preparations (linear in $n$).

#### 2.3 Continuous Calibration

Measurement errors drift over time. Strategies:
- Interleave calibration circuits with algorithm circuits
- Use sliding window averaging
- Model temporal drift explicitly

### 3. Matrix Inversion Mitigation

#### 3.1 Basic Inversion

Given the noisy probability vector, recover the ideal:

$$\boxed{\mathbf{p}_{\text{ideal}} = M^{-1} \cdot \mathbf{p}_{\text{noisy}}}$$

**Requirement**: $M$ must be invertible (always true for physical noise).

#### 3.2 Challenges with Inversion

**Negative probabilities**: $M^{-1} \mathbf{p}_{\text{noisy}}$ may have negative entries due to statistical noise.

**Amplified variance**: $\text{Var}[\mathbf{p}_{\text{ideal}}] = M^{-1} \text{Var}[\mathbf{p}_{\text{noisy}}] (M^{-1})^T$

**Ill-conditioning**: If $M$ is nearly singular, small errors get amplified.

#### 3.3 Regularization Techniques

**Truncated SVD**: Set small singular values to threshold:
$$M^{-1}_{\text{reg}} = V \Sigma^{-1}_{\text{trunc}} U^T$$

**Tikhonov regularization**: Minimize $\|M\mathbf{x} - \mathbf{p}\|^2 + \lambda\|\mathbf{x}\|^2$

**Constrained optimization**: Enforce $\sum_i p_i = 1$ and $p_i \geq 0$

### 4. Expectation Value Correction

#### 4.1 Direct Expectation Correction

For diagonal observables (classical), expectation values transform as:

$$\boxed{\langle O \rangle_{\text{ideal}} = \mathbf{o}^T M^{-1} \mathbf{p}_{\text{noisy}}}$$

where $\mathbf{o}$ is the vector of eigenvalues.

For Pauli-Z observables:
$$\langle Z_i \rangle_{\text{ideal}} = \sum_{s} (-1)^{s_i} (M^{-1} \mathbf{p})_s$$

#### 4.2 Tensor Product Correction

For independent errors, single-qubit correction suffices:

$$\langle Z_i \rangle_{\text{ideal}} = \frac{\langle Z_i \rangle_{\text{noisy}} - (P_{01}^{(i)} - P_{10}^{(i)})}{P_{00}^{(i)} + P_{11}^{(i)} - 1}$$

where $P_{ab}^{(i)} = P(a|b)$ for qubit $i$.

### 5. M3: Matrix-free Measurement Mitigation

#### 5.1 M3 Overview

The M3 method avoids explicit matrix inversion by:
1. Using iterative solvers instead of direct inversion
2. Exploiting sparsity in measurement outcomes
3. Only computing corrections for observed bitstrings

**Key insight**: In practice, only a small subset of $2^n$ bitstrings are observed.

#### 5.2 M3 Algorithm

```
Input: counts (observed bitstrings), calibration data
Output: mitigated counts

1. Build reduced confusion matrix M_red
   - Only rows/columns for observed + nearby bitstrings
   - Use tensor product structure when possible

2. Solve M_red · p_ideal = p_noisy iteratively
   - Use preconditioned conjugate gradient
   - Enforce probability constraints

3. Map back to mitigated counts
```

#### 5.3 M3 Complexity

| Method | Matrix Size | Storage | Time |
|--------|------------|---------|------|
| Full inversion | $2^n \times 2^n$ | $O(4^n)$ | $O(8^n)$ |
| Tensor product | $n \times (2 \times 2)$ | $O(n)$ | $O(n \cdot N_{\text{bits}})$ |
| M3 | $K \times K$ | $O(K^2)$ | $O(K^2)$ |

where $K$ is the number of distinct observed bitstrings (typically $\ll 2^n$).

### 6. Correlated Measurement Errors

#### 6.1 Sources of Correlation

- **Crosstalk**: Measuring one qubit affects neighboring qubits
- **Readout resonator coupling**: Shared readout hardware
- **Classical processing**: Correlated discrimination thresholds

#### 6.2 Modeling Correlations

Pairwise correlated model:
$$M = M_{\text{single}} \cdot M_{\text{pair}}$$

where $M_{\text{pair}}$ captures nearest-neighbor correlations.

#### 6.3 Continuous-Time Markov Chain (CTMC) Model

Model readout dynamics:
$$\frac{d\mathbf{p}}{dt} = R \mathbf{p}$$

where $R$ is the transition rate matrix. Solve for steady-state distribution.

### 7. Integration with Other Techniques

#### 7.1 Mitigation Pipeline

```
Circuit Execution → Raw Counts
                      ↓
              Measurement Mitigation → Corrected Counts
                                         ↓
                              ZNE/PEC → Final Expectation
```

Measurement mitigation is typically applied **last** because it operates on classical data.

#### 7.2 Combined Error Bounds

If measurement error rate is $\epsilon_m$ and gate error is $\epsilon_g$:

$$\text{Total error} \approx \epsilon_g \cdot d + \epsilon_m$$

For many NISQ devices: $\epsilon_m > \epsilon_g$, making measurement mitigation high-impact.

## Quantum Computing Applications

### Variational Quantum Eigensolver (VQE)

Measurement mitigation is critical for VQE because:
- Energy estimation requires accurate expectation values
- Measurement errors can shift the energy minimum
- Chemical accuracy demands <1% error in observables

### Quantum Approximate Optimization (QAOA)

For QAOA cost function evaluation:
- Bitstring probabilities directly affect cost
- Measurement errors change optimal parameters
- Mitigation improves solution quality

### Quantum Machine Learning

Classification accuracy depends on measurement fidelity:
- Mitigate before computing loss functions
- Essential for fair comparison with classical methods

## Worked Examples

### Example 1: Single-Qubit Mitigation

**Problem**: A qubit has confusion matrix $M = \begin{pmatrix} 0.98 & 0.03 \\ 0.02 & 0.97 \end{pmatrix}$. After measuring 10000 shots, we observe 6000 zeros and 4000 ones. What are the mitigated probabilities?

**Solution**:

Noisy probabilities: $\mathbf{p}_{\text{noisy}} = (0.6, 0.4)^T$

Compute inverse:
$$M^{-1} = \frac{1}{\det(M)} \begin{pmatrix} 0.97 & -0.03 \\ -0.02 & 0.98 \end{pmatrix}$$

$$\det(M) = 0.98 \times 0.97 - 0.03 \times 0.02 = 0.9506 - 0.0006 = 0.95$$

$$M^{-1} = \begin{pmatrix} 1.021 & -0.0316 \\ -0.0211 & 1.032 \end{pmatrix}$$

Mitigated probabilities:
$$\mathbf{p}_{\text{ideal}} = M^{-1} \mathbf{p}_{\text{noisy}} = \begin{pmatrix} 1.021 & -0.0316 \\ -0.0211 & 1.032 \end{pmatrix} \begin{pmatrix} 0.6 \\ 0.4 \end{pmatrix}$$

$$= \begin{pmatrix} 0.6126 - 0.0126 \\ -0.0127 + 0.4128 \end{pmatrix} = \begin{pmatrix} 0.600 \\ 0.400 \end{pmatrix}$$

Wait, let me recalculate more carefully:
$$p_0 = 1.021 \times 0.6 + (-0.0316) \times 0.4 = 0.6126 - 0.0126 = 0.600$$
$$p_1 = (-0.0211) \times 0.6 + 1.032 \times 0.4 = -0.0127 + 0.4128 = 0.400$$

$$\boxed{\mathbf{p}_{\text{ideal}} \approx (0.600, 0.400)}$$

In this case, the mitigation had minimal effect because the noisy distribution happened to be close to a valid output.

### Example 2: Expectation Value Correction

**Problem**: With the same confusion matrix, compute the mitigated $\langle Z \rangle$ given noisy measurement probabilities $p_0 = 0.7$, $p_1 = 0.3$.

**Solution**:

Noisy expectation:
$$\langle Z \rangle_{\text{noisy}} = p_0 - p_1 = 0.7 - 0.3 = 0.4$$

Using the shortcut formula for single qubit:
$$\langle Z \rangle_{\text{ideal}} = \frac{\langle Z \rangle_{\text{noisy}} - (P_{01} - P_{10})}{P_{00} + P_{11} - 1}$$

$$= \frac{0.4 - (0.03 - 0.02)}{0.98 + 0.97 - 1} = \frac{0.4 - 0.01}{0.95} = \frac{0.39}{0.95}$$

$$\boxed{\langle Z \rangle_{\text{ideal}} \approx 0.411}$$

### Example 3: Two-Qubit Tensor Product

**Problem**: Two qubits have independent measurement errors with matrices $M_1 = \begin{pmatrix} 0.98 & 0.02 \\ 0.02 & 0.98 \end{pmatrix}$ and $M_2 = \begin{pmatrix} 0.95 & 0.05 \\ 0.05 & 0.95 \end{pmatrix}$. Construct the full 4x4 confusion matrix.

**Solution**:

$$M = M_1 \otimes M_2 = \begin{pmatrix} 0.98 M_2 & 0.02 M_2 \\ 0.02 M_2 & 0.98 M_2 \end{pmatrix}$$

$$= \begin{pmatrix}
0.98 \times 0.95 & 0.98 \times 0.05 & 0.02 \times 0.95 & 0.02 \times 0.05 \\
0.98 \times 0.05 & 0.98 \times 0.95 & 0.02 \times 0.05 & 0.02 \times 0.95 \\
0.02 \times 0.95 & 0.02 \times 0.05 & 0.98 \times 0.95 & 0.98 \times 0.05 \\
0.02 \times 0.05 & 0.02 \times 0.95 & 0.98 \times 0.05 & 0.98 \times 0.95
\end{pmatrix}$$

$$\boxed{M = \begin{pmatrix}
0.931 & 0.049 & 0.019 & 0.001 \\
0.049 & 0.931 & 0.001 & 0.019 \\
0.019 & 0.001 & 0.931 & 0.049 \\
0.001 & 0.019 & 0.049 & 0.931
\end{pmatrix}}$$

## Practice Problems

### Level 1: Direct Application

1. Given $M = \begin{pmatrix} 0.95 & 0.08 \\ 0.05 & 0.92 \end{pmatrix}$, compute $M^{-1}$.

2. If we measure 7500 $|0\rangle$s and 2500 $|1\rangle$s with the matrix from problem 1, what are the mitigated probabilities?

3. For a 3-qubit system with identical per-qubit matrices $M_i = \begin{pmatrix} 0.99 & 0.02 \\ 0.01 & 0.98 \end{pmatrix}$, how many parameters describe the full confusion matrix?

### Level 2: Intermediate

4. Derive the formula for $\langle Z_1 Z_2 \rangle$ mitigation in terms of single-qubit confusion matrix elements, assuming independent errors.

5. Given noisy counts $\{00: 4000, 01: 3000, 10: 2500, 11: 500\}$ and confusion matrices from Example 3, compute mitigated counts.

6. Estimate the condition number of a confusion matrix with $P(0|0) = P(1|1) = 1 - \epsilon$ and $P(1|0) = P(0|1) = \epsilon$ for small $\epsilon$.

### Level 3: Challenging

7. Prove that for symmetric confusion matrices ($P(0|1) = P(1|0) = \epsilon$), measurement mitigation amplifies variance by factor $(1-2\epsilon)^{-2}$.

8. Design a calibration protocol that detects correlated measurement errors between adjacent qubits.

9. Derive the M3 reduced matrix size for a sparse measurement distribution where only $K$ bitstrings have non-zero probability.

## Computational Lab

```python
"""
Day 943: Measurement Error Mitigation Implementation
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import inv, svd
from typing import Dict, List, Tuple
import warnings

# ============================================================
# Part 1: Confusion Matrix Characterization
# ============================================================

class MeasurementErrorCharacterizer:
    """Characterize measurement errors through calibration circuits."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.per_qubit_matrices = None
        self.full_matrix = None

    def create_calibration_circuits(self) -> List[QuantumCircuit]:
        """Create calibration circuits for all basis states."""
        circuits = []

        for state in range(2**self.n_qubits):
            qc = QuantumCircuit(self.n_qubits, self.n_qubits)

            # Prepare basis state
            for i in range(self.n_qubits):
                if (state >> i) & 1:
                    qc.x(i)

            qc.measure(range(self.n_qubits), range(self.n_qubits))
            qc.name = f"cal_{state:0{self.n_qubits}b}"
            circuits.append(qc)

        return circuits

    def create_simple_calibration_circuits(self) -> List[QuantumCircuit]:
        """Create only |0...0⟩ and |1...1⟩ calibration circuits."""
        circuits = []

        # All zeros
        qc0 = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc0.measure(range(self.n_qubits), range(self.n_qubits))
        qc0.name = "cal_all_0"
        circuits.append(qc0)

        # All ones
        qc1 = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc1.x(range(self.n_qubits))
        qc1.measure(range(self.n_qubits), range(self.n_qubits))
        qc1.name = "cal_all_1"
        circuits.append(qc1)

        return circuits

    def build_per_qubit_matrices(self, cal_results: Dict[str, Dict[str, int]],
                                  shots: int) -> List[np.ndarray]:
        """Build per-qubit confusion matrices from simple calibration."""
        matrices = []

        # Get results for all-0 and all-1 calibrations
        counts_0 = cal_results.get('cal_all_0', {})
        counts_1 = cal_results.get('cal_all_1', {})

        for q in range(self.n_qubits):
            # P(0|0) and P(1|0) from all-0 calibration
            p_0_given_0 = 0
            p_1_given_0 = 0
            for bitstring, count in counts_0.items():
                bit = int(bitstring[self.n_qubits - 1 - q])
                if bit == 0:
                    p_0_given_0 += count
                else:
                    p_1_given_0 += count
            p_0_given_0 /= shots
            p_1_given_0 /= shots

            # P(0|1) and P(1|1) from all-1 calibration
            p_0_given_1 = 0
            p_1_given_1 = 0
            for bitstring, count in counts_1.items():
                bit = int(bitstring[self.n_qubits - 1 - q])
                if bit == 0:
                    p_0_given_1 += count
                else:
                    p_1_given_1 += count
            p_0_given_1 /= shots
            p_1_given_1 /= shots

            M_q = np.array([[p_0_given_0, p_0_given_1],
                           [p_1_given_0, p_1_given_1]])
            matrices.append(M_q)

        self.per_qubit_matrices = matrices
        return matrices

    def build_full_matrix(self, cal_results: Dict[str, Dict[str, int]],
                          shots: int) -> np.ndarray:
        """Build full 2^n x 2^n confusion matrix."""
        n_states = 2**self.n_qubits
        M = np.zeros((n_states, n_states))

        for prepared_state in range(n_states):
            circuit_name = f"cal_{prepared_state:0{self.n_qubits}b}"
            counts = cal_results.get(circuit_name, {})

            for bitstring, count in counts.items():
                measured_state = int(bitstring, 2)
                M[measured_state, prepared_state] = count / shots

        self.full_matrix = M
        return M

    def get_tensor_product_matrix(self) -> np.ndarray:
        """Construct full matrix from tensor product of per-qubit matrices."""
        if self.per_qubit_matrices is None:
            raise ValueError("Must run calibration first")

        M = self.per_qubit_matrices[0]
        for i in range(1, self.n_qubits):
            M = np.kron(M, self.per_qubit_matrices[i])

        return M

# ============================================================
# Part 2: Matrix Inversion Mitigation
# ============================================================

class MeasurementMitigator:
    """Mitigate measurement errors using matrix inversion."""

    def __init__(self, confusion_matrix: np.ndarray):
        self.M = confusion_matrix
        self.M_inv = None
        self._compute_inverse()

    def _compute_inverse(self):
        """Compute regularized inverse of confusion matrix."""
        try:
            # Try direct inversion
            self.M_inv = inv(self.M)
        except np.linalg.LinAlgError:
            # Use pseudoinverse with regularization
            U, s, Vh = svd(self.M)
            # Regularize small singular values
            s_inv = np.where(s > 1e-10, 1/s, 0)
            self.M_inv = Vh.T @ np.diag(s_inv) @ U.T

    def mitigate_counts(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Mitigate measurement counts."""
        n_qubits = int(np.log2(self.M.shape[0]))
        total_shots = sum(counts.values())

        # Convert counts to probability vector
        p_noisy = np.zeros(2**n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            p_noisy[idx] = count / total_shots

        # Apply inverse
        p_ideal = self.M_inv @ p_noisy

        # Convert back to counts (may have negative values)
        mitigated_counts = {}
        for idx in range(2**n_qubits):
            bitstring = f"{idx:0{n_qubits}b}"
            mitigated_counts[bitstring] = p_ideal[idx] * total_shots

        return mitigated_counts

    def mitigate_counts_constrained(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Mitigate counts with non-negativity constraint."""
        n_qubits = int(np.log2(self.M.shape[0]))
        total_shots = sum(counts.values())

        # Convert counts to probability vector
        p_noisy = np.zeros(2**n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            p_noisy[idx] = count / total_shots

        # Solve constrained optimization
        def objective(p):
            return np.sum((self.M @ p - p_noisy)**2)

        def constraint_sum(p):
            return np.sum(p) - 1.0

        n_states = 2**n_qubits
        x0 = p_noisy.copy()  # Initial guess

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(n_states)],
            constraints={'type': 'eq', 'fun': constraint_sum}
        )

        p_ideal = result.x

        # Convert back to counts
        mitigated_counts = {}
        for idx in range(n_states):
            bitstring = f"{idx:0{n_qubits}b}"
            mitigated_counts[bitstring] = p_ideal[idx] * total_shots

        return mitigated_counts

    def mitigate_expectation(self, counts: Dict[str, int],
                            observable: np.ndarray) -> float:
        """Mitigate expectation value of diagonal observable."""
        n_qubits = int(np.log2(self.M.shape[0]))
        total_shots = sum(counts.values())

        # Convert counts to probability vector
        p_noisy = np.zeros(2**n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            p_noisy[idx] = count / total_shots

        # Mitigated expectation: o^T M^{-1} p
        p_ideal = self.M_inv @ p_noisy
        expectation = observable @ p_ideal

        return expectation

# ============================================================
# Part 3: M3 Method Implementation
# ============================================================

class M3Mitigator:
    """Matrix-free Measurement Mitigation (M3) implementation."""

    def __init__(self, per_qubit_matrices: List[np.ndarray]):
        self.per_qubit_M = per_qubit_matrices
        self.per_qubit_M_inv = [inv(M) for M in per_qubit_matrices]
        self.n_qubits = len(per_qubit_matrices)

    def apply_A(self, p: np.ndarray, bitstrings: List[str]) -> np.ndarray:
        """Apply confusion matrix to probability vector (matrix-free)."""
        result = np.zeros_like(p)

        for i, bs_out in enumerate(bitstrings):
            for j, bs_in in enumerate(bitstrings):
                # Compute M[out, in] using tensor product structure
                prob = 1.0
                for q in range(self.n_qubits):
                    bit_out = int(bs_out[self.n_qubits - 1 - q])
                    bit_in = int(bs_in[self.n_qubits - 1 - q])
                    prob *= self.per_qubit_M[q][bit_out, bit_in]
                result[i] += prob * p[j]

        return result

    def apply_A_inv(self, p: np.ndarray, bitstrings: List[str]) -> np.ndarray:
        """Apply inverse confusion matrix (matrix-free)."""
        result = np.zeros_like(p)

        for i, bs_out in enumerate(bitstrings):
            for j, bs_in in enumerate(bitstrings):
                # Compute M_inv[out, in] using tensor product structure
                prob = 1.0
                for q in range(self.n_qubits):
                    bit_out = int(bs_out[self.n_qubits - 1 - q])
                    bit_in = int(bs_in[self.n_qubits - 1 - q])
                    prob *= self.per_qubit_M_inv[q][bit_out, bit_in]
                result[i] += prob * p[j]

        return result

    def mitigate_counts(self, counts: Dict[str, int],
                        max_iter: int = 100,
                        tol: float = 1e-6) -> Dict[str, float]:
        """Mitigate counts using iterative solver."""
        total_shots = sum(counts.values())
        bitstrings = list(counts.keys())
        n_bitstrings = len(bitstrings)

        # Build probability vector for observed bitstrings
        p_noisy = np.array([counts[bs] / total_shots for bs in bitstrings])

        # Initial guess: apply direct inverse
        p_ideal = self.apply_A_inv(p_noisy, bitstrings)

        # Iterative refinement (Richardson iteration)
        for iteration in range(max_iter):
            # r = p_noisy - A @ p_ideal
            residual = p_noisy - self.apply_A(p_ideal, bitstrings)

            # Update: p_ideal += A_inv @ residual
            correction = self.apply_A_inv(residual, bitstrings)
            p_ideal += correction

            if np.linalg.norm(correction) < tol:
                break

        # Normalize and clip negative values
        p_ideal = np.clip(p_ideal, 0, None)
        p_ideal /= np.sum(p_ideal)

        # Convert back to counts
        return {bs: p * total_shots for bs, p in zip(bitstrings, p_ideal)}

    def mitigate_expectation_fast(self, counts: Dict[str, int],
                                  pauli_string: str) -> float:
        """Fast expectation value mitigation using tensor product."""
        total_shots = sum(counts.values())

        # For tensor product mitigation of Z observables
        exp_mitigated = 0

        for bitstring, count in counts.items():
            # Compute eigenvalue
            eigenvalue = 1
            for q, p in enumerate(pauli_string[::-1]):
                if p == 'Z':
                    bit = int(bitstring[self.n_qubits - 1 - q])
                    eigenvalue *= (-1)**bit

            # Apply per-qubit correction
            correction = 1.0
            for q, p in enumerate(pauli_string[::-1]):
                if p == 'Z':
                    M_inv = self.per_qubit_M_inv[q]
                    bit = int(bitstring[self.n_qubits - 1 - q])
                    # Correction factor for this qubit
                    correction *= (M_inv[0, bit] - M_inv[1, bit])

            exp_mitigated += eigenvalue * correction * count / total_shots

        return exp_mitigated

# ============================================================
# Part 4: Demonstration
# ============================================================

def create_readout_noise_model(n_qubits: int,
                               p0_error: float = 0.02,
                               p1_error: float = 0.05) -> NoiseModel:
    """Create noise model with measurement errors."""
    noise_model = NoiseModel()

    for q in range(n_qubits):
        # Readout error: P(1|0) = p0_error, P(0|1) = p1_error
        readout_error = ReadoutError([[1 - p0_error, p0_error],
                                       [p1_error, 1 - p1_error]])
        noise_model.add_readout_error(readout_error, [q])

    return noise_model

def demonstrate_measurement_mitigation():
    """Full demonstration of measurement error mitigation."""

    print("="*60)
    print("Measurement Error Mitigation Demonstration")
    print("="*60)

    n_qubits = 3
    shots = 20000

    # Create noise model with measurement errors
    noise_model = create_readout_noise_model(n_qubits, p0_error=0.02, p1_error=0.05)

    # Characterization
    print("\n--- Step 1: Calibration ---")
    characterizer = MeasurementErrorCharacterizer(n_qubits)
    cal_circuits = characterizer.create_simple_calibration_circuits()

    noisy_sim = AerSimulator(noise_model=noise_model)

    cal_results = {}
    for qc in cal_circuits:
        result = noisy_sim.run(qc, shots=shots).result()
        cal_results[qc.name] = result.get_counts()

    per_qubit_M = characterizer.build_per_qubit_matrices(cal_results, shots)

    print("\nPer-qubit confusion matrices:")
    for q, M in enumerate(per_qubit_M):
        print(f"\nQubit {q}:")
        print(f"  P(0|0) = {M[0,0]:.4f}, P(0|1) = {M[0,1]:.4f}")
        print(f"  P(1|0) = {M[1,0]:.4f}, P(1|1) = {M[1,1]:.4f}")

    # Create test circuit
    print("\n--- Step 2: Test Circuit ---")
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    print(qc.draw())

    # Run ideal and noisy
    ideal_sim = AerSimulator()
    ideal_counts = ideal_sim.run(qc, shots=shots).result().get_counts()
    noisy_counts = noisy_sim.run(qc, shots=shots).result().get_counts()

    print("\nIdeal counts:")
    for bs in sorted(ideal_counts.keys()):
        print(f"  {bs}: {ideal_counts[bs]}")

    print("\nNoisy counts:")
    for bs in sorted(noisy_counts.keys()):
        print(f"  {bs}: {noisy_counts[bs]}")

    # Apply mitigation
    print("\n--- Step 3: Matrix Inversion Mitigation ---")
    M_full = characterizer.get_tensor_product_matrix()
    mitigator = MeasurementMitigator(M_full)

    mitigated_counts = mitigator.mitigate_counts(noisy_counts)
    mitigated_constrained = mitigator.mitigate_counts_constrained(noisy_counts)

    print("\nMitigated counts (unconstrained):")
    for bs in sorted(mitigated_counts.keys()):
        print(f"  {bs}: {mitigated_counts[bs]:8.1f}")

    print("\nMitigated counts (constrained, non-negative):")
    for bs in sorted(mitigated_constrained.keys()):
        print(f"  {bs}: {mitigated_constrained[bs]:8.1f}")

    # M3 mitigation
    print("\n--- Step 4: M3 Mitigation ---")
    m3_mitigator = M3Mitigator(per_qubit_M)
    m3_counts = m3_mitigator.mitigate_counts(noisy_counts)

    print("\nM3 mitigated counts:")
    for bs in sorted(m3_counts.keys()):
        print(f"  {bs}: {m3_counts[bs]:8.1f}")

    # Compare expectation values
    print("\n--- Step 5: Expectation Value Comparison ---")

    def compute_zzz(counts, total):
        exp = 0
        for bs, count in counts.items():
            parity = sum(int(b) for b in bs) % 2
            exp += (-1)**parity * count
        return exp / total

    ideal_zzz = compute_zzz(ideal_counts, shots)
    noisy_zzz = compute_zzz(noisy_counts, shots)

    # For mitigated, need to handle float counts
    mit_zzz = compute_zzz(mitigated_constrained, shots)
    m3_zzz = m3_mitigator.mitigate_expectation_fast(noisy_counts, 'ZZZ')

    print(f"\n<ZZZ> values:")
    print(f"  Ideal:              {ideal_zzz:.4f}")
    print(f"  Noisy:              {noisy_zzz:.4f}")
    print(f"  Matrix inversion:   {mit_zzz:.4f}")
    print(f"  M3 (fast):          {m3_zzz:.4f}")

    print(f"\nErrors from ideal:")
    print(f"  Noisy:              {abs(noisy_zzz - ideal_zzz):.4f}")
    print(f"  Matrix inversion:   {abs(mit_zzz - ideal_zzz):.4f}")
    print(f"  M3:                 {abs(m3_zzz - ideal_zzz):.4f}")

    return {
        'ideal': ideal_counts,
        'noisy': noisy_counts,
        'mitigated': mitigated_constrained,
        'm3': m3_counts
    }

# ============================================================
# Part 5: Scaling Analysis
# ============================================================

def analyze_scaling():
    """Analyze how mitigation scales with number of qubits."""

    print("\n" + "="*60)
    print("Measurement Mitigation Scaling Analysis")
    print("="*60)

    qubit_counts = [2, 3, 4, 5, 6, 7, 8]
    shots = 10000

    times_full = []
    times_m3 = []
    errors_noisy = []
    errors_mitigated = []

    for n_qubits in qubit_counts:
        print(f"\nProcessing {n_qubits} qubits...")

        # Create noise model
        noise_model = create_readout_noise_model(n_qubits, 0.02, 0.05)

        # Calibration
        characterizer = MeasurementErrorCharacterizer(n_qubits)
        cal_circuits = characterizer.create_simple_calibration_circuits()

        noisy_sim = AerSimulator(noise_model=noise_model)

        cal_results = {}
        for qc in cal_circuits:
            result = noisy_sim.run(qc, shots=shots).result()
            cal_results[qc.name] = result.get_counts()

        per_qubit_M = characterizer.build_per_qubit_matrices(cal_results, shots)

        # Test circuit: GHZ state
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()

        # Run
        ideal_sim = AerSimulator()
        ideal_counts = ideal_sim.run(qc, shots=shots).result().get_counts()
        noisy_counts = noisy_sim.run(qc, shots=shots).result().get_counts()

        # Time full matrix inversion (only for small systems)
        import time

        if n_qubits <= 6:
            M_full = characterizer.get_tensor_product_matrix()
            start = time.time()
            mitigator = MeasurementMitigator(M_full)
            _ = mitigator.mitigate_counts_constrained(noisy_counts)
            times_full.append(time.time() - start)
        else:
            times_full.append(None)

        # Time M3
        start = time.time()
        m3_mitigator = M3Mitigator(per_qubit_M)
        m3_counts = m3_mitigator.mitigate_counts(noisy_counts)
        times_m3.append(time.time() - start)

        # Compute errors
        def compute_z_all(counts, n):
            exp = 0
            total = sum(counts.values())
            for bs, count in counts.items():
                parity = sum(int(b) for b in bs) % 2
                exp += (-1)**parity * count / total
            return exp

        ideal_z = compute_z_all(ideal_counts, n_qubits)
        noisy_z = compute_z_all(noisy_counts, n_qubits)
        m3_z = compute_z_all(m3_counts, n_qubits)

        errors_noisy.append(abs(noisy_z - ideal_z))
        errors_mitigated.append(abs(m3_z - ideal_z))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Timing comparison
    ax1.semilogy(qubit_counts[:len([t for t in times_full if t])],
                [t for t in times_full if t], 'bo-', label='Full matrix', markersize=8)
    ax1.semilogy(qubit_counts, times_m3, 'rs-', label='M3', markersize=8)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Mitigation Computation Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error comparison
    ax2.plot(qubit_counts, errors_noisy, 'ro-', label='Noisy', markersize=8)
    ax2.plot(qubit_counts, errors_mitigated, 'gs-', label='M3 Mitigated', markersize=8)
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Expectation Value Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('measurement_mitigation_scaling.png', dpi=150)
    plt.show()

    print("\nScaling Summary:")
    print(f"  M3 scales polynomially with qubits")
    print(f"  Full matrix inversion scales exponentially")

# ============================================================
# Part 6: Using mthree Library
# ============================================================

def demonstrate_mthree():
    """Demonstrate measurement mitigation using mthree library."""
    try:
        import mthree

        print("\n" + "="*60)
        print("mthree Library Demonstration")
        print("="*60)

        n_qubits = 4
        shots = 10000

        # Create noise model
        noise_model = create_readout_noise_model(n_qubits, 0.03, 0.06)
        noisy_sim = AerSimulator(noise_model=noise_model)

        # Create test circuit
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()

        # Run
        result = noisy_sim.run(qc, shots=shots).result()
        raw_counts = result.get_counts()

        print(f"\nRaw counts (top 5):")
        for bs, count in sorted(raw_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {bs}: {count}")

        # Note: mthree requires backend access for full functionality
        # This is a simplified demonstration
        print("\nmthree features:")
        print("  - Efficient M3 algorithm implementation")
        print("  - Automatic calibration circuit generation")
        print("  - Support for subset of qubits")
        print("  - Integration with Qiskit Runtime")

    except ImportError:
        print("\nmthree not installed. Install with: pip install mthree")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Day 943: Measurement Error Mitigation Lab")
    print("="*60)

    # Part 1: Basic demonstration
    print("\n--- Part 1: Basic Mitigation ---")
    results = demonstrate_measurement_mitigation()

    # Part 2: Scaling analysis
    print("\n--- Part 2: Scaling Analysis ---")
    analyze_scaling()

    # Part 3: mthree library
    print("\n--- Part 3: mthree Library ---")
    demonstrate_mthree()

    print("\n" + "="*60)
    print("Lab Complete! Key Takeaways:")
    print("  1. Measurement errors described by confusion matrices")
    print("  2. Matrix inversion can produce negative probabilities")
    print("  3. M3 method scales efficiently to many qubits")
    print("  4. Tensor product structure enables factored mitigation")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Confusion Matrix | $M_{ij} = P(\text{measure } i \| \text{state } j)$ |
| Error Model | $\mathbf{p}_{\text{noisy}} = M \cdot \mathbf{p}_{\text{ideal}}$ |
| Matrix Inversion | $\mathbf{p}_{\text{ideal}} = M^{-1} \cdot \mathbf{p}_{\text{noisy}}$ |
| Tensor Product | $M = M_1 \otimes M_2 \otimes \cdots \otimes M_n$ |
| Single-Qubit Z Correction | $\langle Z \rangle_{\text{ideal}} = \frac{\langle Z \rangle_{\text{noisy}} - (P_{01} - P_{10})}{P_{00} + P_{11} - 1}$ |

### Main Takeaways

1. **Measurement errors are classical**: Unlike gate errors, readout errors act on classical outcomes and can be characterized efficiently

2. **Confusion matrix characterization**: Calibration circuits prepare known states and record outcome distributions

3. **Matrix inversion trades bias for variance**: Direct inversion can produce negative probabilities; regularization helps

4. **M3 enables scalability**: By exploiting sparsity and tensor structure, M3 handles many-qubit systems efficiently

5. **Integration is straightforward**: Measurement mitigation applies to raw counts before computing expectation values

## Daily Checklist

- [ ] I can construct confusion matrices from calibration data
- [ ] I understand the tensor product structure for independent errors
- [ ] I can apply matrix inversion with appropriate regularization
- [ ] I know the advantages and limitations of the M3 method
- [ ] I can mitigate expectation values efficiently
- [ ] I understand how to combine measurement mitigation with other techniques

## Preview of Day 944

Tomorrow we explore **Dynamical Decoupling (DD)**, which uses carefully timed pulse sequences to suppress decoherence:

- DD pulse sequences: XY4, CPMG, Uhrig
- Error suppression scaling
- Concatenated and optimized DD
- Integration with quantum circuits

DD is unique among mitigation techniques as it operates during circuit execution rather than on measurement outcomes.
