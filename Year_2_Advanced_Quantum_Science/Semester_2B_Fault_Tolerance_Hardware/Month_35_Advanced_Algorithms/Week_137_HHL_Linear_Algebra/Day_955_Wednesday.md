# Day 955: HHL Algorithm Derivation

## Week 137, Day 3 | Month 35: Advanced Quantum Algorithms

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Complete HHL algorithm derivation |
| Afternoon | 2.5 hours | Problem solving: Complexity analysis |
| Evening | 2 hours | Computational lab: Algorithm simulation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Derive the complete HHL algorithm** step by step from first principles
2. **Explain the eigenvalue inversion mechanism** via controlled rotation
3. **Analyze the success probability** and role of amplitude amplification
4. **Calculate the complexity** $O(\log(N) \cdot s^2 \cdot \kappa^2 / \epsilon)$
5. **Compare HHL to classical algorithms** quantifying the exponential speedup
6. **Identify the key assumptions** required for quantum advantage

---

## Core Content

### 1. The HHL Problem Statement

The HHL algorithm, introduced by Harrow, Hassidim, and Lloyd in 2009, solves:

**Given:**
- Sparse Hermitian matrix $A \in \mathbb{C}^{N \times N}$
- Quantum state $|b\rangle = \sum_i b_i|i\rangle$ encoding the right-hand side

**Output:**
- Quantum state $|x\rangle \propto A^{-1}|b\rangle$

**Key insight:** HHL doesn't output the classical solution vector, but rather a quantum state encoding it.

### 2. Mathematical Framework

#### Spectral Decomposition

Express $A$ in its eigenbasis:
$$A = \sum_{j=0}^{N-1} \lambda_j |u_j\rangle\langle u_j|$$

where $\lambda_j$ are eigenvalues and $|u_j\rangle$ are orthonormal eigenvectors.

#### Solution in Eigenbasis

Expand $|b\rangle$ in the eigenbasis:
$$|b\rangle = \sum_{j=0}^{N-1} \beta_j |u_j\rangle \quad \text{where } \beta_j = \langle u_j|b\rangle$$

The solution:
$$\boxed{|x\rangle = A^{-1}|b\rangle = \sum_{j=0}^{N-1} \frac{\beta_j}{\lambda_j} |u_j\rangle}$$

**Goal:** Transform $\beta_j \to \beta_j/\lambda_j$ for each eigencomponent.

### 3. HHL Algorithm Overview

The algorithm proceeds in four stages:

```
|0⟩_n ─────┬─── QPE ───┬─── Controlled R_y ───┬─── QPE† ───┬─── Measure ancilla
           │           │                      │            │
|b⟩        └───────────┴──────────────────────┴────────────┘

|0⟩_anc ──────────────────── R_y(θ(λ)) ─────────────────────── Measure → |1⟩ success
```

**Stage 1:** Quantum Phase Estimation
- Extract eigenvalues $\lambda_j$ into ancilla register

**Stage 2:** Controlled Rotation
- Rotate ancilla qubit by angle depending on $1/\lambda_j$

**Stage 3:** Uncomputation
- Reverse QPE to disentangle eigenvalue register

**Stage 4:** Measurement
- Measure ancilla; post-select on $|1\rangle$

### 4. Detailed Algorithm Derivation

#### Step 1: State Preparation

Begin with:
$$|0\rangle_n \otimes |b\rangle \otimes |0\rangle_{anc}$$

where:
- $n$ qubits for eigenvalue register (QPE precision)
- $\log_2(N)$ qubits for $|b\rangle$
- 1 ancilla qubit for rotation flag

#### Step 2: Quantum Phase Estimation

Apply QPE with unitary $U = e^{iA\tau}$ for suitable time $\tau$:

$$|0\rangle_n |b\rangle |0\rangle \xrightarrow{QPE} \sum_j \beta_j |\tilde{\lambda}_j\rangle |u_j\rangle |0\rangle$$

where $|\tilde{\lambda}_j\rangle$ is an $n$-bit approximation to $\lambda_j$ (encoded via phase $\phi_j = \lambda_j\tau/(2\pi)$).

**Critical:** The eigenvalue register is now **entangled** with the eigenstate.

#### Step 3: Controlled Rotation

Apply a rotation on the ancilla qubit, controlled by the eigenvalue register:

$$R_y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

with angle:
$$\boxed{\theta(\lambda) = 2\arcsin\left(\frac{C}{\lambda}\right)}$$

where $C \leq \lambda_{min}$ is a normalization constant.

This transforms:
$$|0\rangle_{anc} \to \sqrt{1 - \frac{C^2}{\lambda^2}}|0\rangle + \frac{C}{\lambda}|1\rangle$$

After controlled rotation:
$$\sum_j \beta_j |\tilde{\lambda}_j\rangle |u_j\rangle \left(\sqrt{1 - \frac{C^2}{\tilde{\lambda}_j^2}}|0\rangle + \frac{C}{\tilde{\lambda}_j}|1\rangle\right)$$

#### Step 4: Uncomputation (Inverse QPE)

Apply $QPE^\dagger$ to disentangle the eigenvalue register:

$$\sum_j \beta_j |0\rangle_n |u_j\rangle \left(\sqrt{1 - \frac{C^2}{\lambda_j^2}}|0\rangle + \frac{C}{\lambda_j}|1\rangle\right)$$

#### Step 5: Measurement and Post-Selection

Measure the ancilla qubit. Conditioned on outcome $|1\rangle$:

$$\boxed{|x\rangle = \frac{1}{\sqrt{p_{success}}} \sum_j \frac{C\beta_j}{\lambda_j} |u_j\rangle \propto A^{-1}|b\rangle}$$

**Success probability:**
$$p_{success} = \sum_j \left|\frac{C\beta_j}{\lambda_j}\right|^2 = C^2 \|A^{-1}|b\rangle\|^2$$

### 5. Success Probability Analysis

#### Probability Lower Bound

Since $|\lambda_j| \geq \lambda_{min}$ and $C \leq \lambda_{min}$:

$$p_{success} = C^2 \sum_j \frac{|\beta_j|^2}{\lambda_j^2} \geq \frac{C^2}{\lambda_{max}^2} \sum_j |\beta_j|^2 = \frac{C^2}{\lambda_{max}^2}$$

With $C = \lambda_{min}$:
$$\boxed{p_{success} \geq \frac{\lambda_{min}^2}{\lambda_{max}^2} = \frac{1}{\kappa^2}}$$

where $\kappa = \lambda_{max}/\lambda_{min}$ is the condition number.

#### Amplitude Amplification

Grover-style amplitude amplification can boost success probability:
- Initial success probability: $p \sim 1/\kappa^2$
- After $O(\kappa)$ amplification rounds: $p \to O(1)$

Total complexity includes $O(\kappa)$ repetitions.

### 6. Complete Complexity Analysis

The full complexity of HHL involves several components:

#### Component 1: State Preparation
$$T_{prep} = O(T_b)$$
Cost to prepare $|b\rangle$ from classical data (often the bottleneck!)

#### Component 2: Hamiltonian Simulation
For QPE, we need controlled-$e^{iA\tau \cdot 2^k}$ for $k = 0, 1, \ldots, n-1$.

Using product formulas:
$$T_{sim} = O(s^2 \tau^2 / \epsilon_{sim})$$

where $s$ is the sparsity (non-zeros per row).

#### Component 3: Phase Estimation
$$T_{QPE} = O(n \cdot T_{sim}) = O(\log(1/\epsilon) \cdot s^2 \tau^2 / \epsilon_{sim})$$

#### Component 4: Controlled Rotation
$$T_{rot} = O(\text{poly}(n))$$
Relatively cheap compared to other components.

#### Component 5: Amplitude Amplification
$$T_{amp} = O(\kappa)$$
repetitions of the entire circuit.

#### Total Complexity

Combining all factors:
$$\boxed{T_{HHL} = O\left(\frac{\log(N) \cdot s^2 \cdot \kappa^2}{\epsilon}\right)}$$

where:
- $N$ = matrix dimension (log enters through qubit count)
- $s$ = sparsity
- $\kappa$ = condition number
- $\epsilon$ = error tolerance

### 7. Classical Comparison

| Method | Time Complexity | Space | Output |
|--------|-----------------|-------|--------|
| Gaussian Elimination | $O(N^3)$ | $O(N^2)$ | Full $x$ |
| Conjugate Gradient | $O(N \cdot s \cdot \kappa)$ | $O(N \cdot s)$ | Full $x$ |
| **HHL** | $O(\log N \cdot s^2 \cdot \kappa^2/\epsilon)$ | $O(\log N)$ | $\|x\rangle$ |

**Exponential speedup in $N$** — but:
- HHL is **worse in $\kappa$**: $\kappa^2$ vs $\sqrt{\kappa}$ for CG
- HHL outputs **quantum state**, not classical vector
- HHL requires **quantum access** to matrix and vector

### 8. Requirements for Quantum Advantage

For HHL to provide practical advantage:

#### Requirement 1: Large Problem Size
$N$ must be large enough that $\log(N) \ll N$. Typically $N > 10^6$.

#### Requirement 2: Low Condition Number
$\kappa$ should be $\text{poly}(\log N)$ for exponential speedup. High $\kappa$ erases advantage.

#### Requirement 3: Efficient State Preparation
$|b\rangle$ must be preparable in $\text{poly}(\log N)$ time. Classical data loading can cost $O(N)$!

#### Requirement 4: Useful Quantum Output
The application must need only limited information from $|x\rangle$:
- Expectation values $\langle x|M|x\rangle$
- Sampling from $|x\rangle$
- **Not** the full classical vector $x$

### 9. The Eigenvalue Inversion Trick

The heart of HHL is encoding $1/\lambda$ into amplitudes:

#### Why Controlled Rotation Works

The rotation $R_y(\theta) |0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$

With $\theta = 2\arcsin(C/\lambda)$:
$$\sin(\theta/2) = \sin(\arcsin(C/\lambda)) = \frac{C}{\lambda}$$

The amplitude of $|1\rangle$ is exactly $C/\lambda$!

#### Implementing the Controlled Rotation

The rotation angle must depend on the eigenvalue register:
$$R_y\left(2\arcsin\left(\frac{C}{\tilde{\lambda}}\right)\right)$$

This requires:
1. Decoding the binary eigenvalue $|\tilde{\lambda}\rangle$
2. Computing $\arcsin(C/\tilde{\lambda})$
3. Applying the corresponding rotation

In practice, this is done via:
- Look-up tables encoded in quantum circuits
- Polynomial approximations to $\arcsin$
- Direct computation using reversible arithmetic

---

## Worked Examples

### Example 1: 2×2 System HHL

**Problem:** Apply HHL to solve:
$$\begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

**Solution:**

Step 1: Eigendecomposition
- Eigenvalues: $\lambda_1 = 2$, $\lambda_2 = 4$
- Eigenvectors: $|u_1\rangle = |0\rangle$, $|u_2\rangle = |1\rangle$

Step 2: Encode $|b\rangle$
$$|b\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = \frac{1}{\sqrt{2}}(|u_1\rangle + |u_2\rangle)$$

So $\beta_1 = \beta_2 = 1/\sqrt{2}$.

Step 3: Expected output
$$|x\rangle \propto \frac{\beta_1}{\lambda_1}|u_1\rangle + \frac{\beta_2}{\lambda_2}|u_2\rangle = \frac{1}{\sqrt{2}}\left(\frac{1}{2}|0\rangle + \frac{1}{4}|1\rangle\right)$$

Normalizing:
$$|x\rangle = \frac{1}{\sqrt{5/16}}\left(\frac{1}{2}|0\rangle + \frac{1}{4}|1\rangle\right) = \frac{2}{\sqrt{5}}|0\rangle + \frac{1}{\sqrt{5}}|1\rangle$$

Step 4: Classical verification
$$x = A^{-1}b = \begin{pmatrix} 1/2 \\ 1/4 \end{pmatrix}$$

Normalized: $\hat{x} = (2/\sqrt{5}, 1/\sqrt{5})^T$ ✓

**Success probability:**
$$p_{success} = C^2\left(\frac{1/2}{2^2} + \frac{1/2}{4^2}\right) = C^2 \cdot \frac{5}{32}$$

With $C = \lambda_{min} = 2$: $p_{success} = 4 \cdot 5/32 = 5/8$

---

### Example 2: Complexity Comparison

**Problem:** Compare HHL vs conjugate gradient for $N = 10^6$, $s = 10$, $\kappa = 100$, $\epsilon = 0.01$.

**Solution:**

**Conjugate Gradient:**
$$T_{CG} = O(N \cdot s \cdot \sqrt{\kappa}) = O(10^6 \cdot 10 \cdot 10) = O(10^8)$$

**HHL:**
$$T_{HHL} = O(\log N \cdot s^2 \cdot \kappa^2 / \epsilon) = O(20 \cdot 100 \cdot 10000 / 0.01) = O(2 \times 10^9)$$

Wait—HHL is **slower**!

**Analysis:** For this $\kappa$, the $\kappa^2$ term dominates. Let's find the crossover:

$$\log N \cdot s^2 \cdot \kappa^2 / \epsilon < N \cdot s \cdot \sqrt{\kappa}$$

For $N = 10^6$, $s = 10$, $\epsilon = 0.01$:
$$20 \cdot 100 \cdot \kappa^2 / 0.01 < 10^7 \cdot \sqrt{\kappa}$$
$$2 \times 10^5 \cdot \kappa^2 < 10^7 \cdot \sqrt{\kappa}$$
$$\kappa^{3/2} < 50$$
$$\kappa < 13.6$$

$$\boxed{\text{HHL wins only for } \kappa < 14 \text{ in this scenario}}$$

---

### Example 3: Precision Requirements

**Problem:** How many QPE ancilla qubits are needed for 1% relative error when $\lambda \in [1, 100]$?

**Solution:**

Step 1: Phase encoding
With simulation time $\tau$, phase $\phi = \lambda\tau/(2\pi)$.

Choose $\tau = 2\pi/110$ so $\phi_{max} = 100/110 < 1$.

Step 2: Eigenvalue resolution
For 1% relative error on $\lambda_{min} = 1$:
$$\delta\lambda = 0.01 \cdot 1 = 0.01$$

Phase precision needed:
$$\delta\phi = \delta\lambda \cdot \tau / (2\pi) = 0.01 / 110 \approx 9 \times 10^{-5}$$

Step 3: Ancilla count
$$2^{-n} < 9 \times 10^{-5}$$
$$n > \log_2(1.1 \times 10^4) \approx 13.4$$

$$\boxed{n = 14 \text{ ancilla qubits}}$$

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** For a matrix with eigenvalues $\{1, 3, 5\}$, what is the condition number?

**Problem 1.2:** Calculate the success probability for HHL with $C = 1$, $\lambda_j = \{2, 4\}$, and $|\beta_j|^2 = \{0.6, 0.4\}$.

**Problem 1.3:** What rotation angle $\theta$ is needed for eigenvalue $\lambda = 5$ with $C = 2$?

### Level 2: Intermediate Analysis

**Problem 2.1:** Derive the state after the controlled rotation step for a system with two eigenvalues $\lambda_1, \lambda_2$ and coefficients $\beta_1, \beta_2$.

**Problem 2.2:** If HHL uses 10 QPE ancillas and $\tau = 0.1$, what is the range of resolvable eigenvalues?

**Problem 2.3:** Prove that amplitude amplification can boost HHL success probability from $1/\kappa^2$ to $O(1)$ using $O(\kappa)$ iterations.

### Level 3: Challenging Problems

**Problem 3.1:** **Error Analysis**

The controlled rotation uses approximate eigenvalues $\tilde{\lambda}_j$ instead of true $\lambda_j$. If $|\tilde{\lambda}_j - \lambda_j| \leq \delta$:
- Derive the error in the output state
- How does this error depend on condition number?
- What precision is needed for total error $\epsilon$?

**Problem 3.2:** **Non-Hermitian Extension**

HHL requires Hermitian $A$. For non-Hermitian $A$, consider:
$$\tilde{A} = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}$$

- Show $\tilde{A}$ is Hermitian
- Express $(A)^{-1}$ in terms of $\tilde{A}^{-1}$
- What is the overhead in qubits and complexity?

**Problem 3.3:** **Optimal Parameter Selection**

For HHL with:
- Error tolerance $\epsilon$
- Matrix size $N$
- Condition number $\kappa$
- Sparsity $s$

Derive the optimal choice of:
- QPE precision $n$
- Simulation time $\tau$
- Constant $C$

that minimizes total complexity.

---

## Computational Lab

### HHL Algorithm Simulation

```python
"""
Day 955: HHL Algorithm Simulation
Complete implementation demonstrating the algorithm conceptually.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import Tuple, List


class HHLSimulator:
    """
    Classical simulation of the HHL algorithm.

    This demonstrates the mathematical steps without actual quantum hardware.
    """

    def __init__(self, A: np.ndarray, precision_bits: int = 8):
        """
        Initialize HHL simulator.

        Parameters:
        -----------
        A : ndarray
            Hermitian matrix to invert
        precision_bits : int
            Number of bits for eigenvalue precision
        """
        if not np.allclose(A, A.conj().T):
            raise ValueError("Matrix must be Hermitian")

        self.A = A
        self.N = A.shape[0]
        self.n = precision_bits

        # Eigendecomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(A)
        self.lambda_min = np.min(np.abs(self.eigenvalues))
        self.lambda_max = np.max(np.abs(self.eigenvalues))
        self.kappa = self.lambda_max / self.lambda_min

        # Simulation time (avoid phase wrapping)
        self.tau = 2 * np.pi / (1.5 * self.lambda_max)

        # Rotation constant
        self.C = self.lambda_min * 0.9  # Slightly less than lambda_min

    def analyze_matrix(self) -> dict:
        """Return matrix properties."""
        return {
            'dimension': self.N,
            'eigenvalues': self.eigenvalues,
            'condition_number': self.kappa,
            'lambda_min': self.lambda_min,
            'lambda_max': self.lambda_max,
            'simulation_time': self.tau,
            'rotation_constant': self.C
        }

    def encode_b(self, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode b in eigenbasis.

        Returns:
        --------
        betas : array
            Coefficients in eigenbasis
        b_normalized : array
            Normalized state |b⟩
        """
        b_normalized = b / np.linalg.norm(b)
        betas = self.eigenvectors.conj().T @ b_normalized
        return betas, b_normalized

    def simulate_qpe(self, betas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate QPE step.

        Returns:
        --------
        lambda_estimates : array
            Estimated eigenvalues
        qpe_state : array
            State after QPE (coefficients in computational basis)
        """
        # In ideal QPE, we get exact eigenvalues
        # In practice, we'd get n-bit approximations
        lambda_estimates = self.eigenvalues.copy()

        # Discretize eigenvalues to n bits
        phases = lambda_estimates * self.tau / (2 * np.pi)

        # Round to n-bit precision
        discrete_phases = np.round(phases * 2**self.n) / 2**self.n
        lambda_discretized = discrete_phases * 2 * np.pi / self.tau

        return lambda_discretized, betas

    def controlled_rotation(self, betas: np.ndarray,
                           lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate controlled rotation step.

        Returns:
        --------
        state_0 : array
            Amplitudes for ancilla |0⟩
        state_1 : array
            Amplitudes for ancilla |1⟩ (the desired output)
        """
        # Rotation angles
        # theta = 2 * arcsin(C / lambda)
        # sin(theta/2) = C / lambda
        # cos(theta/2) = sqrt(1 - C²/lambda²)

        sin_theta_half = self.C / lambdas
        cos_theta_half = np.sqrt(1 - sin_theta_half**2)

        # After rotation: |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        state_0 = betas * cos_theta_half
        state_1 = betas * sin_theta_half  # This encodes 1/λ!

        return state_0, state_1

    def run_hhl(self, b: np.ndarray) -> dict:
        """
        Run complete HHL simulation.

        Parameters:
        -----------
        b : ndarray
            Right-hand side vector

        Returns:
        --------
        dict : Complete simulation results
        """
        # Step 1: Encode b
        betas, b_normalized = self.encode_b(b)

        # Step 2: QPE
        lambda_est, qpe_coeffs = self.simulate_qpe(betas)

        # Step 3: Controlled rotation
        state_0, state_1 = self.controlled_rotation(qpe_coeffs, lambda_est)

        # Step 4: Post-selection on |1⟩
        success_prob = np.sum(np.abs(state_1)**2)
        x_unnormalized = state_1 / np.sqrt(success_prob) if success_prob > 0 else state_1

        # Convert back from eigenbasis
        x_state = self.eigenvectors @ x_unnormalized

        # Classical solution for comparison
        x_classical = np.linalg.solve(self.A, b)
        x_classical_normalized = x_classical / np.linalg.norm(x_classical)

        # Fidelity
        fidelity = np.abs(np.vdot(x_state, x_classical_normalized))**2

        return {
            'b_input': b,
            'b_normalized': b_normalized,
            'betas': betas,
            'lambda_estimates': lambda_est,
            'state_0_coeffs': state_0,
            'state_1_coeffs': state_1,
            'success_probability': success_prob,
            'x_state': x_state,
            'x_classical': x_classical_normalized,
            'fidelity': fidelity
        }

    def success_probability_analysis(self, b: np.ndarray) -> dict:
        """
        Analyze success probability in detail.
        """
        betas, _ = self.encode_b(b)

        # Theoretical minimum
        p_min = (self.C / self.lambda_max)**2

        # Theoretical maximum
        p_max = (self.C / self.lambda_min)**2

        # Actual for this b
        p_actual = np.sum(np.abs(betas)**2 * (self.C / self.eigenvalues)**2)

        return {
            'p_min': p_min,
            'p_max': p_max,
            'p_actual': p_actual,
            'amplification_rounds': int(np.ceil(np.pi / (4 * np.sqrt(p_actual))))
        }


def visualize_hhl_steps(simulator: HHLSimulator, b: np.ndarray):
    """Visualize the HHL algorithm steps."""
    result = simulator.run_hhl(b)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Input state |b⟩ in eigenbasis
    ax = axes[0, 0]
    ax.bar(range(len(result['betas'])), np.abs(result['betas'])**2, color='steelblue')
    ax.set_xlabel('Eigenstate index j')
    ax.set_ylabel('$|\\beta_j|^2$')
    ax.set_title('Step 1: Input $|b\\rangle$ in Eigenbasis')

    # 2. Eigenvalues
    ax = axes[0, 1]
    ax.bar(range(len(result['lambda_estimates'])), result['lambda_estimates'], color='green')
    ax.set_xlabel('Eigenstate index j')
    ax.set_ylabel('$\\lambda_j$')
    ax.set_title('Step 2: Eigenvalues (from QPE)')

    # 3. After controlled rotation - state |0⟩
    ax = axes[0, 2]
    ax.bar(range(len(result['state_0_coeffs'])), np.abs(result['state_0_coeffs'])**2,
           color='gray', alpha=0.7, label='|0⟩')
    ax.bar(range(len(result['state_1_coeffs'])), np.abs(result['state_1_coeffs'])**2,
           color='red', alpha=0.7, label='|1⟩ (success)')
    ax.set_xlabel('Eigenstate index j')
    ax.set_ylabel('Probability')
    ax.set_title('Step 3: After Controlled Rotation')
    ax.legend()

    # 4. Success probability pie chart
    ax = axes[1, 0]
    success = result['success_probability']
    ax.pie([success, 1-success], labels=['Success |1⟩', 'Failure |0⟩'],
           colors=['green', 'gray'], autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Step 4: Success Probability = {success:.2%}")

    # 5. Output state vs classical solution
    ax = axes[1, 1]
    x = np.arange(len(result['x_state']))
    width = 0.35
    ax.bar(x - width/2, np.abs(result['x_state'])**2, width, label='HHL |x⟩', color='blue')
    ax.bar(x + width/2, np.abs(result['x_classical'])**2, width, label='Classical', color='orange')
    ax.set_xlabel('Component')
    ax.set_ylabel('Probability')
    ax.set_title(f'Step 5: Output (Fidelity = {result["fidelity"]:.4f})')
    ax.legend()

    # 6. Inversion: |β_j|² vs |β_j/λ_j|²
    ax = axes[1, 2]
    original = np.abs(result['betas'])**2
    inverted = np.abs(result['betas'] / simulator.eigenvalues)**2
    inverted = inverted / np.sum(inverted)  # Normalize

    ax.bar(range(len(original)), original, alpha=0.5, label='Original $|\\beta_j|^2$')
    ax.bar(range(len(inverted)), inverted, alpha=0.5, label='Inverted $|\\beta_j/\\lambda_j|^2$')
    ax.set_xlabel('Eigenstate index j')
    ax.set_ylabel('Probability')
    ax.set_title('Eigenvalue Inversion Effect')
    ax.legend()

    plt.tight_layout()
    plt.savefig('hhl_algorithm_steps.png', dpi=150, bbox_inches='tight')
    plt.show()


def complexity_comparison():
    """Compare HHL vs classical complexity."""
    # Parameters
    N_values = np.logspace(3, 9, 20)
    s = 10  # sparsity
    epsilon = 0.01
    kappa_values = [10, 100, 1000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Fixed kappa, varying N
    ax = axes[0]
    for kappa in kappa_values:
        T_classical = N_values * s * np.sqrt(kappa)
        T_hhl = np.log2(N_values) * s**2 * kappa**2 / epsilon
        ax.loglog(N_values, T_classical, '--', label=f'CG (κ={kappa})')
        ax.loglog(N_values, T_hhl, '-', label=f'HHL (κ={kappa})')

    ax.set_xlabel('Problem Size N', fontsize=12)
    ax.set_ylabel('Time Complexity', fontsize=12)
    ax.set_title('HHL vs Conjugate Gradient', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Crossover analysis
    ax = axes[1]
    for kappa in kappa_values:
        # Find crossover point
        crossovers = []
        for N in N_values:
            T_classical = N * s * np.sqrt(kappa)
            T_hhl = np.log2(N) * s**2 * kappa**2 / epsilon
            crossovers.append(T_classical / T_hhl)

        ax.semilogx(N_values, crossovers, label=f'κ={kappa}')

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Problem Size N', fontsize=12)
    ax.set_ylabel('CG / HHL Complexity Ratio', fontsize=12)
    ax.set_title('Regime Analysis (>1 means HHL wins)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.01, 100])
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('hhl_complexity_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def condition_number_impact():
    """Analyze impact of condition number on HHL."""
    kappa_values = np.logspace(0, 4, 50)

    # Success probability
    p_success = 1 / kappa_values**2

    # Amplification rounds needed
    amp_rounds = np.pi / (4 * np.sqrt(p_success))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Success probability
    ax = axes[0]
    ax.loglog(kappa_values, p_success, 'b-', linewidth=2)
    ax.set_xlabel('Condition Number κ', fontsize=12)
    ax.set_ylabel('Success Probability', fontsize=12)
    ax.set_title('HHL Success Probability vs κ', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(y=0.01, color='red', linestyle='--', label='1% threshold')
    ax.legend()

    # Amplification rounds
    ax = axes[1]
    ax.loglog(kappa_values, amp_rounds, 'g-', linewidth=2)
    ax.set_xlabel('Condition Number κ', fontsize=12)
    ax.set_ylabel('Amplification Rounds', fontsize=12)
    ax.set_title('Required Amplitude Amplification', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('hhl_condition_number.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main demonstration
if __name__ == "__main__":
    print("Day 955: HHL Algorithm Simulation")
    print("=" * 60)

    # Example 1: Simple 4x4 system
    print("\n--- Example: 4×4 Hermitian System ---")

    # Create test matrix with known eigenvalues
    eigenvalues = np.array([1, 2, 3, 4])
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(4, 4))
    A = Q @ np.diag(eigenvalues) @ Q.T

    # Right-hand side
    b = np.array([1, 1, 1, 1])

    # Run HHL simulation
    simulator = HHLSimulator(A, precision_bits=8)
    props = simulator.analyze_matrix()

    print(f"Matrix dimension: {props['dimension']}")
    print(f"Eigenvalues: {props['eigenvalues']}")
    print(f"Condition number: {props['condition_number']:.2f}")

    result = simulator.run_hhl(b)
    print(f"\nSuccess probability: {result['success_probability']:.2%}")
    print(f"Output fidelity: {result['fidelity']:.4f}")

    prob_analysis = simulator.success_probability_analysis(b)
    print(f"Amplification rounds needed: {prob_analysis['amplification_rounds']}")

    # Visualize
    print("\n--- Generating Visualizations ---")
    visualize_hhl_steps(simulator, b)
    complexity_comparison()
    condition_number_impact()

    # Example 2: Ill-conditioned system
    print("\n--- Example: Ill-conditioned System (κ=100) ---")
    eigenvalues_ill = np.array([0.1, 1, 5, 10])
    A_ill = Q @ np.diag(eigenvalues_ill) @ Q.T

    simulator_ill = HHLSimulator(A_ill)
    result_ill = simulator_ill.run_hhl(b)

    print(f"Condition number: {simulator_ill.kappa:.2f}")
    print(f"Success probability: {result_ill['success_probability']:.4%}")
    print(f"Output fidelity: {result_ill['fidelity']:.4f}")
```

---

## Summary

### Key Formulas

| Formula | Expression | Context |
|---------|------------|---------|
| Solution | $\|x\rangle = \sum_j \frac{\beta_j}{\lambda_j}\|u_j\rangle$ | HHL output |
| Rotation angle | $\theta = 2\arcsin(C/\lambda)$ | Controlled rotation |
| Success probability | $p \geq 1/\kappa^2$ | Lower bound |
| Complexity | $O(\log N \cdot s^2 \cdot \kappa^2 / \epsilon)$ | Total time |

### HHL Algorithm Steps

1. **Prepare** $|0\rangle_n |b\rangle |0\rangle_{anc}$
2. **QPE** → $\sum_j \beta_j |\tilde{\lambda}_j\rangle |u_j\rangle |0\rangle$
3. **Rotate** → $\sum_j \beta_j |\tilde{\lambda}_j\rangle |u_j\rangle (\cos|0\rangle + \frac{C}{\lambda_j}|1\rangle)$
4. **Uncompute** → $|0\rangle_n (\sum_j ...) |u_j\rangle (...)$
5. **Measure** ancilla, post-select on $|1\rangle$

### Key Insights

1. **Exponential speedup in N** from quantum parallelism over eigenvalues
2. **Polynomial dependence on κ²** makes ill-conditioned systems harder
3. **Quantum output only** — extracting full classical solution costs $O(N)$
4. **State preparation is often the bottleneck**
5. **Amplitude amplification** boosts success from $1/\kappa^2$ to $O(1)$

---

## Daily Checklist

- [ ] I can derive HHL from state preparation through measurement
- [ ] I understand the controlled rotation mechanism
- [ ] I can calculate success probability
- [ ] I know the complete complexity $O(\log N \cdot s^2 \cdot \kappa^2 / \epsilon)$
- [ ] I can compare HHL to classical methods
- [ ] I understand when HHL provides advantage
- [ ] I implemented and visualized HHL simulation

---

## Preview: Day 956

Tomorrow we focus on **HHL Circuit Implementation**, constructing the actual quantum circuits:

- Hamiltonian simulation methods for $e^{iAt}$
- Implementing controlled rotations with quantum arithmetic
- Ancilla management and garbage collection
- Complete Qiskit implementation for small systems
- Resource counting: qubits, gates, depth

We'll move from mathematical understanding to practical circuit construction.

---

*Day 955 of 2184 | Week 137 of 312 | Month 35 of 72*

*"HHL teaches us that inverting eigenvalues is easy—the challenge is everything around it."*
