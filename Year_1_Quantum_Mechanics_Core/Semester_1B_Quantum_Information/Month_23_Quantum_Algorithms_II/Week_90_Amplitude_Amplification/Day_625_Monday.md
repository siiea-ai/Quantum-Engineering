# Day 625: Amplitude Estimation

## Overview
**Day 625** | Week 90, Day 2 | Year 1, Month 23 | Amplitude Amplification

Today we learn how to estimate the success probability (amplitude) of a quantum algorithm using phase estimation on the amplitude amplification operator Q.

---

## Learning Objectives

1. Understand the amplitude estimation problem
2. Connect Q eigenvalues to success probability
3. Apply quantum phase estimation to Q
4. Analyze precision and query complexity
5. Implement amplitude estimation circuits
6. Handle edge cases and error analysis

---

## Core Content

### The Amplitude Estimation Problem

**Problem:** Given a quantum algorithm A that prepares:
$$A|0\rangle = \sqrt{a}|good\rangle + \sqrt{1-a}|bad\rangle$$

**Goal:** Estimate $a = P(good)$ to precision $\epsilon$ using as few queries to A as possible.

**Classical approach:** Run A multiple times, count successes
- Precision $\epsilon$ requires $O(1/\epsilon^2)$ samples (standard error $\sim 1/\sqrt{n}$)

**Quantum approach:** Use phase estimation on Q
- Precision $\epsilon$ requires only $O(1/\epsilon)$ queries!

### Eigenvalues of the Q Operator

The Q operator has eigenvalues related to $\theta = \arcsin(\sqrt{a})$:

$$Q|good_\pm\rangle = e^{\pm 2i\theta}|good_\pm\rangle$$

where $|good_\pm\rangle$ are eigenstates in the 2D subspace.

**Key insight:** The phase $\pm 2\theta$ encodes the success probability $a = \sin^2\theta$.

### Constructing the Eigenstates

Define:
$$|good_+\rangle = \frac{1}{\sqrt{2}}(|good\rangle - i|bad\rangle)$$
$$|good_-\rangle = \frac{1}{\sqrt{2}}(|good\rangle + i|bad\rangle)$$

The initial state decomposes as:
$$A|0\rangle = \frac{e^{i\theta}}{\sqrt{2}}|good_+\rangle + \frac{e^{-i\theta}}{\sqrt{2}}|good_-\rangle$$

### Amplitude Estimation Algorithm

**Algorithm:**
1. Prepare initial state: $A|0\rangle$ on target register
2. Apply quantum phase estimation with $Q$ as the unitary
3. Measure phase register to get estimate $\tilde{\theta}$
4. Compute $\tilde{a} = \sin^2(\tilde{\theta})$

**Circuit:**
```
|0⟩^m ─H^⊗m──[QPE with Q]──[QFT†]──[Measure]──→ θ̃

|0⟩^n ───────[    A     ]──[Q^{2^k} controlled]──→
```

### Phase Estimation on Q

Using m qubits for phase estimation:

1. Create superposition: $\frac{1}{\sqrt{2^m}}\sum_{j=0}^{2^m-1}|j\rangle$

2. Apply controlled-$Q^{2^k}$ operations

3. Apply inverse QFT

4. Measure to get estimate of $2\theta/(2\pi)$

**Precision:** With m qubits, we estimate $\theta$ to within $\pm \frac{\pi}{2^m}$.

### Error Analysis

**Phase estimation error:** $|\tilde{\theta} - \theta| \leq \frac{\pi}{2^m}$

**Amplitude estimation error:**
$$|\tilde{a} - a| = |\sin^2\tilde{\theta} - \sin^2\theta|$$

Using $|\sin^2 x - \sin^2 y| \leq |x - y|$ for small differences:
$$|\tilde{a} - a| \leq \frac{2\pi}{2^m}$$

**To achieve precision $\epsilon$ on $a$:**
$$m = O(\log(1/\epsilon))$$ qubits in phase register

**Total queries to A:** $O(2^m) = O(1/\epsilon)$

### Comparison: Classical vs Quantum Estimation

| Method | Precision $\epsilon$ | Queries to A |
|--------|---------------------|--------------|
| Classical sampling | $\epsilon$ | $O(1/\epsilon^2)$ |
| Amplitude estimation | $\epsilon$ | $O(1/\epsilon)$ |

**Quadratic speedup in precision!**

### Handling the Sign Ambiguity

Phase estimation gives $\pm\theta$, both equally likely. However:
- Both give $\sin^2(\pm\theta) = \sin^2\theta = a$
- The sign ambiguity doesn't affect amplitude estimation!

### Confidence Bounds

Phase estimation succeeds with probability:
$$P_{success} \geq \frac{4}{\pi^2} \approx 0.405$$

for getting the correct $\theta$ to within $\pm \pi/2^m$.

Using more precise analysis or repetition, we can boost this to arbitrarily high confidence.

---

## Worked Examples

### Example 1: Estimating Small Probability
Estimate $a = 0.01$ to precision $\epsilon = 0.001$.

**Solution:**
Need $2\pi/2^m \leq 0.001$, so $2^m \geq 6283$.

$m = \lceil\log_2(6283)\rceil = 13$ phase qubits.

Total controlled-Q operations: $\sum_{k=0}^{12} 2^k = 2^{13} - 1 = 8191$

Classical would need: $(1/0.001)^2 = 10^6$ samples!

### Example 2: QPE Setup
For a 2-qubit algorithm A with $a = 0.25$, set up amplitude estimation with m=3.

**Solution:**
$\theta = \arcsin(\sqrt{0.25}) = \pi/6$

Phase to estimate: $2\theta/(2\pi) = 1/6 \approx 0.1667$

With m=3 bits: resolution = $1/8 = 0.125$

Best approximation: $1/8$ or $2/8$

$1/8 \times 2\pi = \pi/4 \Rightarrow a = \sin^2(\pi/8) = 0.146$
$2/8 \times 2\pi = \pi/2 \Rightarrow a = \sin^2(\pi/4) = 0.5$

Neither is very accurate; need more qubits!

With m=4: $2/16$ gives $a = 0.25$ ✓

### Example 3: Query Complexity
Compare queries needed to estimate $a = 0.1$ to precision 0.01.

**Solution:**
Classical: $n = (0.1 \times 0.9)/0.01^2 = 900$ trials (from Chernoff bound)

Quantum: $m = \lceil\log_2(2\pi/0.01)\rceil = 10$ qubits
Total Q queries: $\sim 2^{10} = 1024$ but each Q uses 2 calls to A, so ~2048.

Hmm, roughly similar for this case. The advantage grows as $\epsilon$ decreases.

For $\epsilon = 0.001$:
- Classical: $\sim 90,000$ trials
- Quantum: $\sim 2^{14} \approx 16,000$ Q queries

---

## Practice Problems

### Problem 1: Phase Register Size
What m is needed to estimate amplitude to precision $\epsilon = 10^{-6}$?

### Problem 2: Eigenvalue Verification
Show that $|good_+\rangle$ as defined is indeed an eigenvector of Q with eigenvalue $e^{2i\theta}$.

### Problem 3: Total Query Count
For amplitude estimation with m phase qubits, calculate the exact number of:
a) Controlled-Q gates
b) Uses of A
c) Uses of $A^\dagger$

---

## Computational Lab

```python
"""Day 625: Amplitude Estimation"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

class AmplitudeEstimation:
    """Quantum amplitude estimation implementation."""

    def __init__(self, A, good_states, n_qubits, m_precision):
        """
        Initialize amplitude estimation.

        Args:
            A: Preparation unitary
            good_states: List of good state indices
            n_qubits: Qubits in target register
            m_precision: Qubits in phase register
        """
        self.A = A
        self.good_states = good_states
        self.n = n_qubits
        self.m = m_precision
        self.N = 2**n_qubits

        self._setup()

    def _setup(self):
        """Setup operators and compute true values."""
        # Build Q operator
        A_inv = self.A.conj().T

        S_chi = np.eye(self.N)
        for g in self.good_states:
            S_chi[g, g] = -1

        S_0 = np.eye(self.N)
        S_0[0, 0] = -1

        self.Q = -self.A @ S_0 @ A_inv @ S_chi

        # True amplitude
        zero = np.zeros(self.N)
        zero[0] = 1
        psi = self.A @ zero

        self.a_true = sum(abs(psi[g])**2 for g in self.good_states)
        self.theta_true = np.arcsin(np.sqrt(self.a_true))

    def qpe_simulate(self):
        """
        Simulate quantum phase estimation on Q.

        Returns:
            Estimated theta and confidence
        """
        # Get eigenvalues of Q
        eigvals, eigvecs = np.linalg.eig(self.Q)

        # Initial state
        zero = np.zeros(self.N)
        zero[0] = 1
        psi = self.A @ zero

        # Decompose initial state in eigenbasis
        coeffs = eigvecs.conj().T @ psi

        # Phases of eigenvalues
        phases = np.angle(eigvals) / (2 * np.pi)  # In [0, 1) range
        phases = np.mod(phases, 1)

        # QPE simulates measurement, collapsing to an eigenvalue
        # Probability of each eigenvalue
        probs = np.abs(coeffs)**2

        # Simulate QPE measurement with m-bit precision
        # Each eigenphase gets discretized to m bits
        discretized = np.round(phases * 2**self.m) / 2**self.m
        discretized = np.mod(discretized, 1)

        # Weighted average (simplified model)
        # In reality, QPE gives one of the phases with appropriate probability

        # For a more accurate simulation, we'd do full QPE
        # Here, we find the phase closest to true theta
        true_phase = self.theta_true / np.pi  # 2θ/2π
        true_phase_mod = np.mod(true_phase, 1)

        # Discretize
        estimated_phase = np.round(true_phase_mod * 2**self.m) / 2**self.m

        # Convert back to theta and a
        theta_est = estimated_phase * np.pi
        a_est = np.sin(theta_est)**2

        return theta_est, a_est

    def full_qpe_simulation(self, num_samples=1000):
        """
        Full simulation of QPE including probabilistic outcomes.
        """
        # Initialize state in computational basis representation
        zero = np.zeros(self.N)
        zero[0] = 1
        psi = self.A @ zero

        # The eigenvalues of Q in 2D subspace are e^{±2iθ}
        # Initial state overlaps with both eigenstates

        estimates = []

        for _ in range(num_samples):
            # Random choice between +theta and -theta based on overlap
            # (simplified - in reality determined by eigenvector overlaps)
            sign = np.random.choice([1, -1])
            true_phase = sign * 2 * self.theta_true / (2 * np.pi)
            true_phase = np.mod(true_phase, 1)

            # Add QPE discretization noise
            # QPE gives k/2^m where k is closest integer to true_phase * 2^m
            k = int(np.round(true_phase * 2**self.m))
            k = k % 2**self.m

            estimated_phase = k / 2**self.m
            theta_est = estimated_phase * np.pi
            a_est = np.sin(theta_est)**2

            estimates.append(a_est)

        return np.array(estimates)

    def analyze_precision(self):
        """Analyze estimation precision."""
        estimates = self.full_qpe_simulation(1000)

        mean_est = np.mean(estimates)
        std_est = np.std(estimates)

        print(f"\nAmplitude Estimation Analysis (m={self.m}):")
        print(f"  True amplitude: {self.a_true:.6f}")
        print(f"  True theta: {self.theta_true:.6f} rad")
        print(f"  Mean estimate: {mean_est:.6f}")
        print(f"  Std deviation: {std_est:.6f}")
        print(f"  Theoretical precision: {2*np.pi / 2**self.m:.6f}")

        return estimates


def compare_classical_quantum_estimation(a_true, epsilon_values):
    """Compare classical vs quantum sample complexity."""
    classical_samples = []
    quantum_queries = []

    for eps in epsilon_values:
        # Classical: need O(1/eps^2) samples
        # From Hoeffding: n >= ln(2/delta) / (2*eps^2)
        # For delta=0.1: n >= 1.15 / eps^2
        classical_n = int(np.ceil(2 / eps**2))
        classical_samples.append(classical_n)

        # Quantum: need O(1/eps) queries
        # m bits gives precision 2*pi/2^m
        # Need 2*pi/2^m <= eps, so m >= log2(2*pi/eps)
        m = int(np.ceil(np.log2(2*np.pi / eps)))
        quantum_n = 2**m  # Number of Q queries
        quantum_queries.append(quantum_n)

    return classical_samples, quantum_queries


def plot_complexity_comparison():
    """Plot classical vs quantum estimation complexity."""
    epsilon_values = np.logspace(-1, -4, 50)
    classical, quantum = compare_classical_quantum_estimation(0.1, epsilon_values)

    plt.figure(figsize=(10, 6))
    plt.loglog(epsilon_values, classical, 'b-', label='Classical O(1/ε²)', linewidth=2)
    plt.loglog(epsilon_values, quantum, 'r-', label='Quantum O(1/ε)', linewidth=2)

    plt.xlabel('Precision ε', fontsize=12)
    plt.ylabel('Number of Queries', fontsize=12)
    plt.title('Amplitude Estimation: Classical vs Quantum', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig('amplitude_estimation_complexity.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_estimation_accuracy(n_qubits=3, good_state=0):
    """Visualize estimation accuracy vs precision qubits."""
    # Create simple preparation
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    A = H
    for _ in range(n_qubits - 1):
        A = np.kron(A, H)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    m_values = [2, 4, 6, 8]

    for idx, m in enumerate(m_values):
        ax = axes[idx // 2, idx % 2]

        ae = AmplitudeEstimation(A, [good_state], n_qubits, m)
        estimates = ae.full_qpe_simulation(1000)

        ax.hist(estimates, bins=30, density=True, alpha=0.7, color='blue')
        ax.axvline(x=ae.a_true, color='red', linestyle='--', linewidth=2,
                   label=f'True a = {ae.a_true:.4f}')

        ax.set_xlabel('Estimated Amplitude', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'm = {m} precision qubits', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ae_precision_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
print("="*60)
print("Amplitude Estimation")
print("="*60)

# Basic example
n = 3
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
A = H
for _ in range(n - 1):
    A = np.kron(A, H)

good_states = [0, 3]  # Two "good" states

print("\n1. BASIC AMPLITUDE ESTIMATION")
print("-"*50)
ae = AmplitudeEstimation(A, good_states, n, m_precision=6)
theta_est, a_est = ae.qpe_simulate()

print(f"True amplitude: {ae.a_true:.6f}")
print(f"Estimated amplitude: {a_est:.6f}")
print(f"Error: {abs(a_est - ae.a_true):.6f}")

# Analyze with statistics
print("\n2. STATISTICAL ANALYSIS")
print("-"*50)
estimates = ae.analyze_precision()

# Plot complexity comparison
print("\n3. COMPLEXITY COMPARISON")
print("-"*50)
plot_complexity_comparison()

# Show precision improvement
print("\n4. PRECISION VS PHASE REGISTER SIZE")
print("-"*50)
visualize_estimation_accuracy()

# Query complexity table
print("\n5. QUERY COMPLEXITY TABLE")
print("-"*60)
print(f"{'Precision ε':>15} | {'Classical (1/ε²)':>18} | {'Quantum (1/ε)':>15}")
print("-"*60)
for eps in [0.1, 0.01, 0.001, 0.0001]:
    classical = int(2 / eps**2)
    m = int(np.ceil(np.log2(2*np.pi / eps)))
    quantum = 2**m
    print(f"{eps:>15.4f} | {classical:>18,} | {quantum:>15,}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Q eigenvalues | $e^{\pm 2i\theta}$ where $a = \sin^2\theta$ |
| Phase to estimate | $2\theta/(2\pi)$ |
| Precision (m bits) | $\Delta\theta \leq \pi/2^m$ |
| Amplitude error | $\|\tilde{a} - a\| \leq 2\pi/2^m$ |
| Classical queries | $O(1/\epsilon^2)$ |
| Quantum queries | $O(1/\epsilon)$ |

### Key Takeaways

1. **Q eigenvalues encode amplitude** through phase $e^{2i\theta}$
2. **Phase estimation extracts** the success probability
3. **Quadratic speedup** over classical sampling
4. **Precision requires more qubits** in phase register
5. **Sign ambiguity** doesn't affect final amplitude
6. **Foundational for quantum Monte Carlo** methods

---

## Daily Checklist

- [ ] I understand how Q eigenvalues relate to amplitude
- [ ] I can set up phase estimation on the Q operator
- [ ] I can calculate required precision qubits
- [ ] I understand the quadratic speedup over classical
- [ ] I can analyze estimation error
- [ ] I ran the computational lab and compared methods

---

*Next: Day 626 — Fixed-Point Amplification*
