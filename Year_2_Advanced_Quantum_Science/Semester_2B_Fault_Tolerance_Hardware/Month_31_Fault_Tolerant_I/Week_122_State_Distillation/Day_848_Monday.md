# Day 848: Why Distillation - The Necessity of Magic State Purification

## Week 122: State Distillation Protocols | Month 31: Fault-Tolerant Quantum Computing I

### Semester 2B: Fault Tolerance & Hardware | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Noisy magic state preparation, error propagation theory |
| **Afternoon** | 2.5 hours | Error amplification analysis, threshold requirements |
| **Evening** | 1.5 hours | Computational lab: Error propagation simulation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 848, you will be able to:

1. **Explain why raw magic states are insufficient** for fault-tolerant computation
2. **Analyze error propagation** through T-gate injection circuits
3. **Calculate error amplification** without distillation
4. **Derive threshold requirements** for magic state quality
5. **Understand the distillation paradigm** as a solution to the magic state problem
6. **Quantify the gap** between achievable and required magic state fidelity

---

## 1. Introduction: The Magic State Quality Problem

### Recap: Why Magic States?

From Week 121, we established that fault-tolerant universal quantum computation requires:

1. **Clifford gates**: Implementable transversally on CSS codes (fault-tolerant by construction)
2. **Non-Clifford gates**: Required for universality (Gottesman-Knill theorem)
3. **T-gate solution**: Use magic states $|T\rangle$ with gate teleportation

The **T magic state**:
$$|T\rangle = T|+\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + e^{i\pi/4}|1\rangle\right)$$

### The Fundamental Problem

**Raw magic state preparation** through physical operations has significant errors:

$$\rho_{\text{raw}} = (1 - \epsilon)|T\rangle\langle T| + \epsilon \cdot E$$

where $\epsilon \sim 10^{-2}$ to $10^{-3}$ for current hardware and $E$ represents error states.

**Fault-tolerance requirement**: For reliable computation with $N$ T-gates:
$$\epsilon_{\text{magic}} \ll \frac{1}{N}$$

For algorithms like Shor's (with $N \sim 10^{10}$ T-gates), we need:
$$\epsilon_{\text{magic}} < 10^{-15}$$

$$\boxed{\text{Gap: } \epsilon_{\text{raw}} \sim 10^{-3} \longrightarrow \epsilon_{\text{required}} < 10^{-15}}$$

This 12 orders of magnitude gap cannot be bridged by better hardware alone.

---

## 2. Noisy Magic State Preparation

### Physical Preparation Methods

**Method 1: Direct State Preparation**
1. Prepare $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$
2. Apply physical T gate: $T|+\rangle = |T\rangle$
3. Encode into logical qubit

**Error sources:**
- State preparation errors: $\epsilon_{\text{prep}} \sim 10^{-3}$
- T-gate error: $\epsilon_T \sim 10^{-2}$ to $10^{-3}$
- Encoding errors: $\epsilon_{\text{encode}} \sim 10^{-3}$

**Total error:**
$$\epsilon_{\text{raw}} \approx \epsilon_{\text{prep}} + \epsilon_T + \epsilon_{\text{encode}} \sim 10^{-2}$$

### Error Model for Magic States

The general error model for a noisy magic state:

$$\rho = (1 - \epsilon)|T\rangle\langle T| + \frac{\epsilon}{3}(X|T\rangle\langle T|X + Y|T\rangle\langle T|Y + Z|T\rangle\langle T|Z)$$

**Depolarizing channel assumption**: Errors are equally likely X, Y, or Z.

For the T state specifically:
- $X|T\rangle$ is orthogonal to $|T\rangle$
- $Y|T\rangle = iXZ|T\rangle$ is also orthogonal
- $Z|T\rangle = e^{-i\pi/4}|T^*\rangle$ where $|T^*\rangle = T^{\dagger}|+\rangle$

$$\boxed{\text{Noisy magic state: } \rho = (1-\epsilon)|T\rangle\langle T| + \epsilon \cdot \rho_{\text{error}}}$$

---

## 3. Error Amplification Without Distillation

### T-Gate Injection Circuit

The gate teleportation circuit for applying T:

```
|ψ⟩ ─────●───── M_X ═══╗
         │             ║
|T⟩ ─────X────────────S^m──→ T|ψ⟩
```

**Error propagation analysis:**

If $|T\rangle$ has error $\epsilon$, the output error depends on the error type:

1. **Z error on $|T\rangle$**: Propagates as Z error on output (correctable)
2. **X error on $|T\rangle$**: Propagates as S error (uncorrectable without detection)
3. **Y error on $|T\rangle$**: Combination of X and Z effects

### Detailed Error Propagation

**Case 1: Perfect $|T\rangle$**

Output: $T|\psi\rangle$ (correct)

**Case 2: Z error on $|T\rangle$ → $Z|T\rangle = e^{-i\pi/4}|T^*\rangle$**

The protocol with $Z|T\rangle$:
- After CNOT: Error propagates through circuit
- X measurement: Outcome may differ
- Result: Phase error on output, equivalent to $ZT|\psi\rangle$

This is a **Pauli error** on the output, handled by error correction.

**Case 3: X error on $|T\rangle$ → $X|T\rangle$**

More problematic:
- $X|T\rangle = \frac{1}{\sqrt{2}}(|1\rangle + e^{i\pi/4}|0\rangle)$
- This introduces an S-type error: Output is $S^{\pm 1}T|\psi\rangle \neq T|\psi\rangle$

This is a **non-Pauli error** that corrupts the computation.

$$\boxed{\text{X/Y errors on magic state} \rightarrow \text{Non-Clifford errors on output}}$$

### Error Accumulation Through Multiple T-Gates

For a circuit with $N$ T-gates, each with magic state error $\epsilon$:

**Independent error model:**
$$P_{\text{failure}} = 1 - (1 - \epsilon)^N \approx N\epsilon \text{ for } N\epsilon \ll 1$$

**Required per-T error:**
$$\epsilon < \frac{P_{\text{target}}}{N}$$

**Example: Shor's algorithm for 2048-bit factoring**
- $N \approx 10^{10}$ T-gates
- $P_{\text{target}} = 0.01$ (1% failure probability)
- Required: $\epsilon < 10^{-12}$

$$\boxed{\epsilon_{\text{required}} < \frac{P_{\text{target}}}{N_T} \sim 10^{-12} \text{ to } 10^{-15}}$$

---

## 4. Threshold Requirements for Magic States

### Defining the Magic State Threshold

The **magic state threshold** $\epsilon_{\text{th}}$ is the maximum error rate at which distillation can improve state fidelity:

$$\epsilon_{\text{out}} < \epsilon_{\text{in}} \text{ when } \epsilon_{\text{in}} < \epsilon_{\text{th}}$$

For the 15-to-1 protocol:
$$\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3$$

**Threshold condition:** $\epsilon_{\text{out}} < \epsilon_{\text{in}}$
$$35\epsilon_{\text{in}}^3 < \epsilon_{\text{in}}$$
$$35\epsilon_{\text{in}}^2 < 1$$
$$\epsilon_{\text{in}} < \frac{1}{\sqrt{35}} \approx 0.169$$

$$\boxed{\epsilon_{\text{th}}^{(15-1)} \approx 16.9\%}$$

### Comparison of Protocol Thresholds

| Protocol | Error Scaling | Threshold |
|----------|--------------|-----------|
| 15-to-1 (Bravyi-Kitaev) | $35\epsilon^3$ | ~17% |
| 7-to-1 (Steane) | $7\epsilon^3$ | ~26% |
| Bravyi-Haah 10-to-2 | $O(\epsilon^2)$ | ~10% |
| MEK 4-to-1 | $O(\epsilon^2)$ | ~14% |

### Physical Error Rate vs. Threshold

Current state-of-the-art:
- Superconducting qubits: $\epsilon_{\text{phys}} \sim 10^{-3}$ (well below threshold)
- Trapped ions: $\epsilon_{\text{phys}} \sim 10^{-4}$ (excellent margin)
- Photonic: $\epsilon_{\text{phys}} \sim 10^{-2}$ (close to threshold)

$$\boxed{\text{Margin} = \frac{\epsilon_{\text{th}}}{\epsilon_{\text{phys}}} > 10^2 \text{ (sufficient for distillation)}}$$

---

## 5. The Distillation Paradigm

### Core Concept

**Distillation** is a process that:
1. Takes multiple noisy magic states as input
2. Performs Clifford operations (fault-tolerant) on them
3. Measures stabilizers to detect/correct errors
4. Outputs fewer, higher-fidelity magic states

$$\boxed{n_{\text{in}} \text{ noisy states} \xrightarrow{\text{Clifford + measure}} n_{\text{out}} \text{ clean states}}$$

### Why Distillation Works

**Key insight**: We can detect magic state errors using only Clifford operations.

The magic state $|T\rangle$ is a +1 eigenstate of:
$$M = e^{-i\pi/8}TXT^{\dagger} = \frac{1}{\sqrt{2}}(X + Y)$$

**Verification protocol:**
1. Prepare many copies of (noisy) $|T\rangle$
2. Measure collective stabilizers using Clifford operations
3. Post-select on correct outcomes
4. Remaining states have reduced error

### Error Detection vs. Correction

**Detection**: Identify that an error occurred (discard state)
**Correction**: Fix the error without discarding

Most distillation protocols use **detection**:
- Measure stabilizers of an error-detecting code
- Discard if any stabilizer outcome is wrong
- Accept if all outcomes are correct

**Probability of passing with $k$ errors:**
$$P_{\text{pass}}(k) = \begin{cases} 1 & k = 0 \\ 0 & 0 < k < t+1 \\ \text{depends on code} & k > t \end{cases}$$

where $t$ is the number of detectable errors.

---

## 6. Resource Overhead Preview

### Distillation Cost

For target error $\epsilon_{\text{target}}$ starting from $\epsilon_{\text{raw}}$:

**Number of levels**: $k$ where
$$35^{(3^k-1)/2}\epsilon_{\text{raw}}^{3^k} < \epsilon_{\text{target}}$$

**Raw states per output:**
$$n_{\text{raw}} = 15^k$$

**Example calculation:**
- $\epsilon_{\text{raw}} = 10^{-3}$
- $\epsilon_{\text{target}} = 10^{-15}$

Level 1: $\epsilon_1 = 35 \times (10^{-3})^3 = 3.5 \times 10^{-8}$
Level 2: $\epsilon_2 = 35 \times (3.5 \times 10^{-8})^3 \approx 1.5 \times 10^{-21}$

$\epsilon_2 < 10^{-15}$ ✓

**States required:** $15^2 = 225$ raw states per output

### Space-Time Volume

The **space-time volume** $V$ measures total resource usage:
$$V = (\text{qubits}) \times (\text{time})$$

For a level-$k$ factory with code distance $d$:
$$V \sim d^3 \times 15^k$$

$$\boxed{\text{Distillation overhead: } O\left(\log^{\gamma}(1/\epsilon)\right) \text{ where } \gamma \approx 1.6}$$

---

## 7. Worked Examples

### Example 1: Error Rate Requirement Calculation

**Problem:** An algorithm requires $10^8$ T-gates with 99% success probability. What magic state error rate is needed?

**Solution:**

**Step 1:** Define success requirement
- Total failure probability: $P_{\text{fail}} < 0.01$
- Per-T failure probability: $\epsilon_T < P_{\text{fail}}/N = 0.01/10^8 = 10^{-10}$

**Step 2:** Account for error propagation
Each T-gate injection can fail if the magic state has an X or Y error:
$$\epsilon_{\text{magic}} \approx \epsilon_T = 10^{-10}$$

**Step 3:** Verify distillation feasibility
Starting from $\epsilon_{\text{raw}} = 10^{-3}$:
- Level 1: $35 \times (10^{-3})^3 = 3.5 \times 10^{-8}$
- Level 2: $35 \times (3.5 \times 10^{-8})^3 = 1.5 \times 10^{-21}$

Two levels sufficient: $1.5 \times 10^{-21} < 10^{-10}$ ✓

$$\boxed{\epsilon_{\text{required}} = 10^{-10}, \text{ achieved with 2-level distillation}}$$

---

### Example 2: Error Amplification Analysis

**Problem:** Show that without distillation, using $10^6$ T-gates with $\epsilon = 10^{-3}$ magic states results in near-certain failure.

**Solution:**

**Step 1:** Calculate failure probability
$$P_{\text{fail}} = 1 - (1 - \epsilon)^N$$
$$= 1 - (1 - 10^{-3})^{10^6}$$

For large $N$ with small $\epsilon$:
$$(1 - \epsilon)^N \approx e^{-N\epsilon} = e^{-10^6 \times 10^{-3}} = e^{-1000}$$

$$P_{\text{fail}} = 1 - e^{-1000} \approx 1$$

**Step 2:** Interpret result
- $e^{-1000}$ is effectively zero
- Probability of all T-gates succeeding: $\approx 0$
- Algorithm failure is virtually certain

$$\boxed{P_{\text{fail}} \approx 100\% \text{ without distillation}}$$

---

### Example 3: Threshold Margin Calculation

**Problem:** A quantum computer has physical error rate $\epsilon_{\text{phys}} = 2 \times 10^{-3}$. Calculate the threshold margin for 15-to-1 distillation and the number of levels needed for $\epsilon_{\text{target}} = 10^{-12}$.

**Solution:**

**Step 1:** Calculate threshold margin
$$\text{Margin} = \frac{\epsilon_{\text{th}}}{\epsilon_{\text{phys}}} = \frac{0.169}{2 \times 10^{-3}} = 84.5$$

Margin > 10: Excellent ✓

**Step 2:** Calculate levels needed
$$\epsilon_0 = 2 \times 10^{-3}$$
$$\epsilon_1 = 35\epsilon_0^3 = 35 \times 8 \times 10^{-9} = 2.8 \times 10^{-7}$$
$$\epsilon_2 = 35\epsilon_1^3 = 35 \times 2.2 \times 10^{-20} = 7.7 \times 10^{-19}$$

$\epsilon_2 < 10^{-12}$ ✓

**Step 3:** Resource count
$$n_{\text{raw}} = 15^2 = 225 \text{ states per output}$$

$$\boxed{\text{Margin} = 84.5, \text{ Levels} = 2, \text{ Raw states} = 225}$$

---

## 8. Practice Problems

### Problem Set A: Direct Application

**A1.** Calculate the required magic state error rate for an algorithm with:
- $N = 10^{12}$ T-gates
- Target success probability: 99.9%

**A2.** For raw error $\epsilon = 5 \times 10^{-3}$, compute the output error after:
- (a) One level of 15-to-1 distillation
- (b) Two levels of 15-to-1 distillation
- (c) Three levels of 15-to-1 distillation

**A3.** Verify that an X error on the magic state in the T-gate teleportation circuit produces a non-Clifford error on the output.

---

### Problem Set B: Intermediate

**B1.** Derive the threshold condition for a hypothetical protocol with error scaling $\epsilon_{\text{out}} = c\epsilon_{\text{in}}^2$ where $c$ is a constant.

**B2.** Compare the number of raw magic states needed to achieve $\epsilon_{\text{target}} = 10^{-15}$ using:
- 15-to-1 protocol starting from $\epsilon = 10^{-2}$
- 15-to-1 protocol starting from $\epsilon = 10^{-3}$

Which starting error rate is more resource-efficient?

**B3.** An algorithm has T-gate depth $D = 1000$ (longest chain of T-gates). If each T-gate injection takes time $\tau$, and distillation takes time $T_{\text{distill}}$, what is the total execution time as a function of $\tau$, $T_{\text{distill}}$, and $D$?

---

### Problem Set C: Challenging

**C1.** Prove that the magic state $|T\rangle$ cannot be created from $|0\rangle$ using only Clifford operations and computational basis measurements.

**C2.** Analyze a "3-to-1" distillation protocol based on the $[[3,1,1]]$ repetition code. What is its error scaling? Why is it insufficient for practical use?

**C3.** Design an adaptive distillation strategy that uses fewer levels when the algorithm has lower T-count. What is the optimal number of distillation levels as a function of $N$ and $\epsilon_{\text{raw}}$?

---

## 9. Computational Lab: Error Propagation Simulation

```python
"""
Day 848 Computational Lab: Magic State Error Propagation
Understanding Why Distillation is Necessary

This lab simulates error propagation in magic state injection
and demonstrates why raw magic states are insufficient for
fault-tolerant computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from typing import Tuple, List

# Quantum gates and states
def pauli_matrices():
    """Return Pauli matrices."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z

def t_gate():
    """Return T gate matrix."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

def s_gate():
    """Return S gate matrix."""
    return np.array([[1, 0], [0, 1j]], dtype=complex)

def magic_state():
    """Return ideal |T> magic state."""
    ket_plus = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
    return t_gate() @ ket_plus

def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Calculate fidelity between two density matrices.
    For pure states: F = |<psi|phi>|^2
    """
    if rho1.shape == (2, 1):  # Pure state vector
        rho1 = rho1 @ rho1.conj().T
    if rho2.shape == (2, 1):
        rho2 = rho2 @ rho2.conj().T

    sqrt_rho1 = np.linalg.cholesky(rho1 + 1e-10*np.eye(2))
    inner = sqrt_rho1.conj().T @ rho2 @ sqrt_rho1
    return np.real(np.trace(np.sqrt(inner + 1e-10*np.eye(2))))**2


class NoisyMagicState:
    """
    Represents a noisy magic state with depolarizing error.
    """

    def __init__(self, error_rate: float):
        """
        Initialize noisy magic state.

        Parameters:
        -----------
        error_rate : float
            Probability of error (depolarizing)
        """
        self.error_rate = error_rate
        self.ideal = magic_state()
        I, X, Y, Z = pauli_matrices()
        self.paulis = [I, X, Y, Z]

    def sample(self) -> Tuple[np.ndarray, str]:
        """
        Sample a noisy magic state.

        Returns:
        --------
        state : ndarray
            The sampled state vector
        error_type : str
            Description of error applied
        """
        if np.random.random() > self.error_rate:
            return self.ideal.copy(), "none"
        else:
            # Depolarizing: equal probability of X, Y, Z
            pauli_idx = np.random.choice([1, 2, 3])
            error_names = ["X", "Y", "Z"]
            error_state = self.paulis[pauli_idx] @ self.ideal
            return error_state, error_names[pauli_idx - 1]

    def density_matrix(self) -> np.ndarray:
        """
        Return the density matrix of the noisy magic state.

        rho = (1-epsilon)|T><T| + (epsilon/3)(X|T><T|X + Y|T><T|Y + Z|T><T|Z)
        """
        ideal_dm = self.ideal @ self.ideal.conj().T
        error_dm = np.zeros((2, 2), dtype=complex)

        for P in self.paulis[1:]:  # X, Y, Z
            error_state = P @ self.ideal
            error_dm += error_state @ error_state.conj().T
        error_dm /= 3

        return (1 - self.error_rate) * ideal_dm + self.error_rate * error_dm


def t_gate_injection(psi: np.ndarray, magic: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Perform T-gate injection via gate teleportation.

    Parameters:
    -----------
    psi : ndarray
        Input state to apply T to
    magic : ndarray
        Magic state (possibly noisy)

    Returns:
    --------
    output : ndarray
        Output state
    measurement : int
        X measurement outcome (0 or 1)
    """
    # CNOT: |psi>|magic> -> |psi>(magic XOR psi)
    # Then measure first qubit in X basis

    # Joint state
    joint = np.kron(psi, magic)

    # CNOT
    cnot = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    after_cnot = cnot @ joint

    # Reshape to separate qubits
    after_cnot = after_cnot.reshape(2, 2)

    # X basis measurement on first qubit
    # |+> = (|0> + |1>)/sqrt(2)
    # |-> = (|0> - |1>)/sqrt(2)
    coeff_plus = (after_cnot[0, :] + after_cnot[1, :]) / np.sqrt(2)
    coeff_minus = (after_cnot[0, :] - after_cnot[1, :]) / np.sqrt(2)

    p_plus = np.sum(np.abs(coeff_plus)**2)

    if np.random.random() < p_plus:
        outcome = 0
        output = coeff_plus.reshape(2, 1)
    else:
        outcome = 1
        output = coeff_minus.reshape(2, 1)

    output = output / np.linalg.norm(output)

    # Apply S correction if outcome is 1
    if outcome == 1:
        output = s_gate() @ output

    return output, outcome


def analyze_error_propagation(epsilon: float, n_trials: int = 10000):
    """
    Analyze how magic state errors propagate through T-gate injection.

    Parameters:
    -----------
    epsilon : float
        Magic state error rate
    n_trials : int
        Number of Monte Carlo trials
    """
    print(f"\nError Propagation Analysis (epsilon = {epsilon})")
    print("=" * 60)

    T = t_gate()
    I, X, Y, Z = pauli_matrices()
    noisy_magic = NoisyMagicState(epsilon)

    # Test on |+> state
    ket_plus = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
    ideal_output = T @ ket_plus
    ideal_output = ideal_output / np.linalg.norm(ideal_output)

    error_counts = {"none": 0, "X": 0, "Y": 0, "Z": 0}
    fidelities = []

    for _ in range(n_trials):
        magic, error_type = noisy_magic.sample()
        output, _ = t_gate_injection(ket_plus, magic)

        # Calculate fidelity with ideal output
        fid = np.abs(ideal_output.conj().T @ output)[0, 0]**2
        fidelities.append(fid)

        error_counts[error_type] += 1

    print(f"\nError type distribution:")
    for error, count in error_counts.items():
        print(f"  {error}: {count/n_trials*100:.2f}%")

    print(f"\nOutput fidelity statistics:")
    print(f"  Mean fidelity: {np.mean(fidelities):.6f}")
    print(f"  Std fidelity: {np.std(fidelities):.6f}")
    print(f"  Min fidelity: {np.min(fidelities):.6f}")

    # Categorize by fidelity
    perfect = sum(1 for f in fidelities if f > 0.999)
    degraded = sum(1 for f in fidelities if 0.5 < f < 0.999)
    failed = sum(1 for f in fidelities if f < 0.5)

    print(f"\nOutcome categories:")
    print(f"  Perfect (F > 0.999): {perfect/n_trials*100:.2f}%")
    print(f"  Degraded (0.5 < F < 0.999): {degraded/n_trials*100:.2f}%")
    print(f"  Failed (F < 0.5): {failed/n_trials*100:.2f}%")

    return fidelities


def simulate_algorithm_failure(n_t_gates: int, epsilon: float, n_trials: int = 1000):
    """
    Simulate failure probability for an algorithm with N T-gates.

    Parameters:
    -----------
    n_t_gates : int
        Number of T gates in algorithm
    epsilon : float
        Magic state error rate
    n_trials : int
        Monte Carlo trials

    Returns:
    --------
    p_fail : float
        Estimated failure probability
    """
    failures = 0

    for _ in range(n_trials):
        # Each T-gate has epsilon probability of failure
        n_errors = np.random.binomial(n_t_gates, epsilon)
        if n_errors > 0:
            failures += 1

    return failures / n_trials


def plot_error_amplification():
    """Plot error amplification as function of T-gate count."""
    print("\nGenerating Error Amplification Plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Failure probability vs T-count for different epsilons
    ax = axes[0]
    n_range = np.logspace(2, 10, 50)

    for epsilon in [1e-2, 1e-3, 1e-4, 1e-6, 1e-10]:
        # P_fail = 1 - (1-epsilon)^N ≈ 1 - exp(-N*epsilon)
        p_fail = 1 - np.exp(-n_range * epsilon)
        ax.semilogx(n_range, p_fail, label=f'$\\epsilon = 10^{{{int(np.log10(epsilon))}}}$')

    ax.axhline(y=0.01, color='r', linestyle='--', label='1% threshold')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='50% threshold')
    ax.set_xlabel('Number of T-gates ($N$)', fontsize=12)
    ax.set_ylabel('Failure Probability', fontsize=12)
    ax.set_title('Algorithm Failure vs T-gate Count', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Plot 2: Required epsilon vs T-count for different target success rates
    ax = axes[1]
    n_range = np.logspace(3, 12, 50)

    for p_target in [0.99, 0.999, 0.9999]:
        # epsilon_required < -ln(1-P_fail)/N ≈ P_fail/N for small P_fail
        epsilon_required = -np.log(p_target) / n_range
        ax.loglog(n_range, epsilon_required, label=f'Success = {p_target*100:.1f}%')

    ax.axhline(y=1e-3, color='blue', linestyle=':', label='Raw $\\epsilon$ = $10^{-3}$')
    ax.fill_between(n_range, 1e-3, 1, alpha=0.1, color='red')
    ax.text(1e6, 1e-2, 'Distillation\nRequired', fontsize=11, color='red')

    ax.set_xlabel('Number of T-gates ($N$)', fontsize=12)
    ax.set_ylabel('Required $\\epsilon_{\\text{magic}}$', fontsize=12)
    ax.set_title('Required Magic State Fidelity', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('error_amplification.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Error amplification plot saved to 'error_amplification.png'")


def plot_distillation_need():
    """Visualize the gap between raw and required error rates."""
    print("\nGenerating Distillation Need Visualization...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Algorithm sizes and their required epsilon
    algorithms = {
        'Small QAOA\n(100 T)': 100,
        'VQE\n($10^3$ T)': 1e3,
        'Quantum\nSimulation\n($10^6$ T)': 1e6,
        "Grover's\n($10^8$ T)": 1e8,
        "Shor's 2048-bit\n($10^{10}$ T)": 1e10,
        'Fault-Tolerant\nChemistry\n($10^{12}$ T)': 1e12
    }

    x_pos = np.arange(len(algorithms))

    # Calculate required epsilon for 99% success
    required_eps = [0.01/n for n in algorithms.values()]

    # Raw error rate (constant)
    raw_eps = [1e-3] * len(algorithms)

    # Create bar chart
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, np.log10(raw_eps), width,
                   label='Raw Magic State Error', color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, np.log10(required_eps), width,
                   label='Required Error (99% success)', color='green', alpha=0.7)

    ax.set_ylabel('$\\log_{10}(\\epsilon)$', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_title('Magic State Error: Raw vs Required', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms.keys(), fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add "distillation needed" annotations
    for i, (raw, req) in enumerate(zip(np.log10(raw_eps), np.log10(required_eps))):
        gap = raw - req
        if gap > 1:
            ax.annotate(f'{gap:.0f} orders\nof magnitude',
                       xy=(i, (raw + req)/2), fontsize=8,
                       ha='center', color='purple')

    plt.tight_layout()
    plt.savefig('distillation_need.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Distillation need visualization saved to 'distillation_need.png'")


def demonstrate_threshold():
    """Demonstrate the distillation threshold concept."""
    print("\nDistillation Threshold Demonstration")
    print("=" * 60)

    # 15-to-1 protocol: epsilon_out = 35 * epsilon_in^3
    epsilon_range = np.logspace(-4, -0.5, 100)
    epsilon_out = 35 * epsilon_range**3

    # Find threshold (where epsilon_out = epsilon_in)
    threshold = 1 / np.sqrt(35)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(epsilon_range, epsilon_range, 'k--', label='$\\epsilon_{out} = \\epsilon_{in}$ (no improvement)', linewidth=2)
    ax.loglog(epsilon_range, epsilon_out, 'b-', label='15-to-1: $\\epsilon_{out} = 35\\epsilon_{in}^3$', linewidth=2)

    # Mark threshold
    ax.axvline(x=threshold, color='red', linestyle=':', linewidth=2)
    ax.annotate(f'Threshold\n$\\epsilon_{{th}} \\approx {threshold:.3f}$',
               xy=(threshold, 1e-2), xytext=(0.3, 1e-2),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=12, color='red')

    # Mark improvement region
    ax.fill_between(epsilon_range[epsilon_out < epsilon_range],
                    epsilon_out[epsilon_out < epsilon_range],
                    epsilon_range[epsilon_out < epsilon_range],
                    alpha=0.2, color='green', label='Improvement region')

    ax.set_xlabel('Input Error Rate $\\epsilon_{in}$', fontsize=12)
    ax.set_ylabel('Output Error Rate $\\epsilon_{out}$', fontsize=12)
    ax.set_title('Distillation Threshold for 15-to-1 Protocol', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1e-4, 0.5)
    ax.set_ylim(1e-12, 1)

    plt.tight_layout()
    plt.savefig('distillation_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n15-to-1 Threshold: epsilon_th = 1/sqrt(35) = {threshold:.4f}")
    print("Distillation threshold plot saved to 'distillation_threshold.png'")


def main():
    """Run all Day 848 demonstrations."""
    print("Day 848: Why Distillation - The Necessity of Magic State Purification")
    print("=" * 70)

    # Error propagation analysis
    for eps in [0.01, 0.001, 0.0001]:
        analyze_error_propagation(eps, n_trials=5000)

    # Visualizations
    plot_error_amplification()
    plot_distillation_need()
    demonstrate_threshold()

    # Summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Raw magic states have error ~10^-3, but algorithms need ~10^-15
2. Without distillation, algorithms with >10^3 T-gates will likely fail
3. Error accumulates as N*epsilon, making raw states unusable
4. Distillation threshold for 15-to-1 is ~17%, well above raw error
5. Distillation bridges the 12 orders of magnitude gap
""")

    print("\nDay 848 Computational Lab Complete!")


if __name__ == "__main__":
    main()
```

---

## 10. Summary

### Key Formulas Table

| Concept | Formula/Expression |
|---------|-------------------|
| Raw magic state error | $\epsilon_{\text{raw}} \sim 10^{-3}$ to $10^{-2}$ |
| Required magic state error | $\epsilon_{\text{required}} < P_{\text{target}}/N_T$ |
| Algorithm failure probability | $P_{\text{fail}} \approx 1 - e^{-N\epsilon}$ |
| 15-to-1 threshold | $\epsilon_{\text{th}} = 1/\sqrt{35} \approx 16.9\%$ |
| Distillation error scaling | $\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3$ |
| Raw states per output | $n_{\text{raw}} = 15^k$ for $k$ levels |
| Distillation overhead | $O(\log^{\gamma}(1/\epsilon))$, $\gamma \approx 1.6$ |

### Key Takeaways

1. **Raw magic states are insufficient**: Error rates of $10^{-3}$ cannot support algorithms with $>10^3$ T-gates
2. **Error amplification is severe**: Each T-gate contributes independently to failure probability
3. **12 orders of magnitude gap**: From raw ($10^{-3}$) to required ($10^{-15}$) error rates
4. **Distillation is the solution**: Converts many noisy states to few high-fidelity states
5. **Threshold margin is comfortable**: Physical errors well below 17% threshold
6. **Polynomial overhead**: Distillation adds $O(\log^{\gamma}(1/\epsilon))$ resource cost

---

## 11. Daily Checklist

- [ ] I understand why raw magic states cannot support fault-tolerant computation
- [ ] I can calculate the required magic state error rate for a given algorithm
- [ ] I know how X/Y/Z errors on magic states propagate through T-gate injection
- [ ] I can derive the threshold condition for 15-to-1 distillation
- [ ] I understand the gap between raw and required error rates
- [ ] I completed the computational lab simulating error propagation

---

## 12. Preview: Day 849

Tomorrow we dive into the **15-to-1 Distillation Protocol**:

- Reed-Muller code foundation: Why $[[15, 1, 3]]$?
- Circuit construction for distillation
- Detailed error analysis: Where does $35\epsilon^3$ come from?
- Multi-level distillation strategies
- Resource counting and optimization

We will see exactly how 15 noisy magic states become 1 clean magic state through the elegant structure of Reed-Muller codes.

---

*"The magic state problem is the critical bottleneck of fault-tolerant quantum computing. Distillation is not just a solution - it is the key that unlocks universality."*
— Sergey Bravyi

