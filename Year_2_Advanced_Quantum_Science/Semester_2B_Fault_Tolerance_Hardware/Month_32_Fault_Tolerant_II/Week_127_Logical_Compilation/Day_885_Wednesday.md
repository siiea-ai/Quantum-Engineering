# Day 885: Repeat-Until-Success (RUS) Circuits

## Overview

**Day:** 885 of 1008
**Week:** 127 (Logical Gate Compilation)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** Probabilistic Gate Synthesis with Repeat-Until-Success Protocols

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | RUS theory and construction |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Analysis and catalyst states |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Explain** the repeat-until-success paradigm for gate synthesis
2. **Construct** RUS circuits for specific non-Clifford rotations
3. **Analyze** expected T-count and runtime for RUS protocols
4. **Design** catalyst states for enhanced success probability
5. **Compare** deterministic vs. probabilistic synthesis strategies
6. **Implement** hybrid RUS-deterministic compilation

---

## The RUS Paradigm

### Motivation

Deterministic synthesis (gridsynth) gives:
$$n_T^{\text{det}} = 3\log_2(1/\epsilon) + O(1)$$

**Question:** Can we do better with probabilistic approaches?

**Answer:** For certain gates, RUS circuits achieve:
$$\mathbb{E}[n_T^{\text{RUS}}] = O(1) \text{ (constant!)}$$

with expected finite runtime.

### Basic Concept

A **Repeat-Until-Success** circuit:

1. Prepares an ancilla state
2. Applies a unitary involving target and ancilla
3. Measures the ancilla
4. **Success:** Target has desired gate applied
5. **Failure:** Target has known "wrong" gate applied → correct and retry

```
     ┌─────────────────────┐
     │  RUS Circuit        │
|ψ⟩──┤                     ├── success: V|ψ⟩
     │                     │
|a⟩──┤                     ├── measure → outcome
     └─────────────────────┘
           ↓
      if failure: apply correction, repeat
```

### Why RUS Works

Key insight: **Measurement can be a resource**, not just output.

The ancilla measurement "projects" onto a subspace where the target qubit has the correct transformation.

---

## Mathematical Framework

### RUS Circuit Structure

A general RUS circuit implements:

$$U_{\text{RUS}} = \begin{pmatrix} V & W \\ X & Y \end{pmatrix}$$

where the blocks act on (target, ancilla) space.

After measuring ancilla in $|0\rangle$:
- **Success (outcome 0):** Target state becomes $V|\psi\rangle$ (normalized)
- **Failure (outcome 1):** Target state becomes $X|\psi\rangle$ (normalized)

### Success Probability

For input $|\psi\rangle$:

$$p_{\text{success}} = \|V|\psi\rangle\|^2$$

For uniform input (Haar random $|\psi\rangle$):

$$\boxed{\bar{p}_{\text{success}} = \frac{1}{2}\text{Tr}(V^{\dagger}V)}$$

### Expected Iterations

With success probability $p$, the number of iterations follows a geometric distribution:

$$\boxed{\mathbb{E}[\text{iterations}] = \frac{1}{p}}$$

$$\text{Var}[\text{iterations}] = \frac{1-p}{p^2}$$

### T-Cost Analysis

If each RUS attempt costs $T_{\text{attempt}}$ T gates:

$$\boxed{\mathbb{E}[T_{\text{total}}] = \frac{T_{\text{attempt}}}{p}}$$

For this to beat deterministic synthesis:
$$\frac{T_{\text{attempt}}}{p} < 3\log_2(1/\epsilon)$$

---

## Canonical RUS Construction

### The $V_3$ Gate

One of the simplest non-trivial RUS examples implements the "axial rotation":

$$V_3 = R_z(\theta) \text{ where } \cos(\theta/2) = \cos(\pi/8)\cos(\pi/5)$$

This gate is NOT exactly synthesizable over Clifford+T.

### $V_3$ RUS Circuit

```
|ψ⟩ ──●──────H──●──────S†──
      │         │
|0⟩ ──X──T†──H──X──T──────M
```

**Success:** Measuring $|0\rangle$ gives $V_3|\psi\rangle$
**Failure:** Measuring $|1\rangle$ gives known Clifford → correct and repeat

**T-count per attempt:** 2
**Success probability:** $p = \cos^2(\pi/8) \approx 0.854$

**Expected T-count:**
$$\mathbb{E}[T] = \frac{2}{0.854} \approx 2.34$$

Compare to deterministic: ~50-100 T gates for high precision!

### General RUS for $R_z(\theta)$

For arbitrary $\theta$, we can construct RUS circuits, but the success probability depends on $\theta$:

$$p(\theta) = \frac{1 + \cos(\theta)}{2}$$

**Problem:** For $\theta$ close to $\pi$, success probability approaches 0.

**Solution:** Use catalyst states or nested RUS.

---

## Catalyst States

### Motivation

Some RUS circuits require special ancilla states called **catalysts**:

$$|C\rangle = \alpha|0\rangle + \beta|1\rangle$$

The catalyst:
1. Enhances success probability
2. Is recovered (approximately) after the RUS attempt
3. Can be reused for multiple RUS calls

### Example: $|Y\rangle$ Catalyst

$$|Y\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle) = S|+\rangle$$

This state can catalyze certain rotations:

```
|ψ⟩ ──●──────────●──────────
      │          │
|Y⟩ ──X──Rz(θ)──X──Rz(-θ)──M
```

### Catalyst Preparation

Catalysts themselves may require T gates to prepare:

| Catalyst | Preparation T-count | Benefit |
|----------|---------------------|---------|
| $\|Y\rangle = S\|+\rangle$ | 0 (Clifford) | Moderate |
| $\|T\rangle = T\|+\rangle$ | 1 | Magic state |
| Complex catalysts | 10-50 | High success prob |

**Trade-off:** Preparation cost vs. improved success rate

### Catalyst Recovery

After RUS, the catalyst may be in state $|C'\rangle \approx |C\rangle$.

**Fidelity:** $F = |\langle C | C' \rangle|^2$

If $F$ is high, catalyst can be reused; otherwise, re-prepare.

---

## Expected Runtime Analysis

### Single RUS Gate

For a single RUS gate with:
- T-count per attempt: $T_0$
- Success probability: $p$

$$\boxed{\mathbb{E}[T_{\text{gate}}] = \frac{T_0}{p}}$$

**Variance:**
$$\text{Var}[T_{\text{gate}}] = T_0^2 \cdot \frac{1-p}{p^2}$$

### Circuit with Multiple RUS Gates

For a circuit with $n$ independent RUS gates:

$$\mathbb{E}[T_{\text{total}}] = \sum_{i=1}^{n} \frac{T_0^{(i)}}{p_i}$$

**Depth consideration:** RUS gates may add to circuit depth due to retries.

### Worst-Case vs. Expected

| Metric | Deterministic | RUS |
|--------|---------------|-----|
| Worst-case T | $3\log_2(1/\epsilon)$ | $\infty$ (unbounded) |
| Expected T | $3\log_2(1/\epsilon)$ | $O(1)$ for special gates |
| Variance | 0 | Non-zero |
| Depth impact | Predictable | Variable |

### High-Probability Bounds

With probability $1 - \delta$:

$$T_{\text{RUS}} \leq \frac{T_0}{p} \cdot \ln(1/\delta)$$

**Example:** For 99% confidence with $p = 0.5$:
$$T \leq 2T_0 \cdot \ln(100) \approx 9.2 T_0$$

---

## RUS vs. Deterministic Synthesis

### When RUS Wins

RUS is advantageous when:

1. **Gate is "RUS-friendly":** Has efficient RUS construction with high success probability
2. **Expected cost matters:** Average-case performance acceptable
3. **Catalyst available:** Can amortize catalyst preparation
4. **Low precision needed:** Don't need $\epsilon < 10^{-10}$

### When Deterministic Wins

Deterministic synthesis is better when:

1. **Worst-case guarantees needed:** Real-time applications
2. **Very high precision:** $\epsilon < 10^{-15}$
3. **No good RUS construction:** Generic rotations
4. **Circuit depth critical:** Can't tolerate variable depth

### Hybrid Strategies

**Best practice:** Combine both approaches.

1. Use RUS for "cheap" gates (high success probability)
2. Fall back to deterministic for others
3. Batch RUS operations to amortize catalyst costs

### Comparison Table

| Gate | Deterministic T | RUS Expected T | Winner |
|------|-----------------|----------------|--------|
| $R_z(\pi/4)$ | 1 (exact) | N/A | Det |
| $R_z(\pi/8)$ | ~30 | ~2.5 | RUS |
| $R_z(0.123)$ | ~50 | ~10 | RUS |
| $R_z(\pi - 0.001)$ | ~50 | ~1000 | Det |

---

## Advanced RUS Techniques

### Nested RUS

For low-success-probability operations, use multiple RUS stages:

$$\text{RUS}_1 \to \text{RUS}_2 \to \cdots \to \text{RUS}_k$$

Each stage corrects failures from the previous stage.

### Programmable Ancilla Rotations (PAR)

Use magic states to implement rotations:

$$R_z(\theta) = \text{PAR}(|A(\theta)\rangle)$$

where $|A(\theta)\rangle$ is a prepared ancilla encoding angle $\theta$.

### Quantum Signal Processing

Modern approach: use **Quantum Signal Processing (QSP)** to implement arbitrary functions with optimal query complexity.

QSP achieves:
$$\text{Cost} = O(\text{degree of polynomial approximation})$$

---

## Worked Examples

### Example 1: $V_3$ Gate Analysis

**Problem:** Analyze the expected T-count for implementing 100 $V_3$ gates using RUS.

**Solution:**

For single $V_3$:
- $T_0 = 2$ (T gates per attempt)
- $p = \cos^2(\pi/8) \approx 0.854$
- $\mathbb{E}[T] = 2/0.854 \approx 2.34$

For 100 gates:
$$\mathbb{E}[T_{\text{total}}] = 100 \times 2.34 = 234 \text{ T gates}$$

**Variance:**
$$\text{Var}[T_{\text{single}}] = 4 \cdot \frac{0.146}{0.854^2} \approx 0.80$$
$$\text{Var}[T_{\text{total}}] = 100 \times 0.80 = 80$$
$$\sigma = \sqrt{80} \approx 9$$

**With 99% confidence:** $T_{\text{total}} \leq 234 + 3 \times 9 = 261$

**Compare deterministic:** ~5000-10000 T gates for equivalent precision.

### Example 2: Success Probability Calculation

**Problem:** An RUS circuit has the structure:

$$U = \begin{pmatrix} a & b & c & d \\ e & f & g & h \\ i & j & k & l \\ m & n & o & p \end{pmatrix}$$

where the first qubit is target, second is ancilla. Calculate success probability for target in $|+\rangle$.

**Solution:**

The success operator (project ancilla to $|0\rangle$) gives:
$$V = \begin{pmatrix} a & c \\ i & k \end{pmatrix}$$

Input: $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$

Output (unnormalized): $V|+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} a+c \\ i+k \end{pmatrix}$

Success probability:
$$p = \frac{|a+c|^2 + |i+k|^2}{2}$$

### Example 3: Catalyst Amortization

**Problem:** A catalyst state requires 20 T gates to prepare and enables RUS with success probability 0.9. How many RUS operations needed to amortize the catalyst cost?

**Solution:**

Without catalyst: assume deterministic synthesis at 50 T gates each.

With catalyst:
- Catalyst prep: 20 T gates (one-time)
- Per operation: $2/0.9 \approx 2.22$ T gates

Break-even point:
$$20 + n \cdot 2.22 = n \cdot 50$$
$$20 = n \cdot 47.78$$
$$n \approx 0.42$$

Even **one** RUS operation is worth preparing the catalyst!

For $n$ operations:
- Deterministic: $50n$ T gates
- RUS: $20 + 2.22n$ T gates

**Savings:** $47.78n - 20$ T gates

---

## Practice Problems

### Level 1: Direct Application

**P1.1** An RUS circuit has success probability $p = 0.75$. Calculate:
a) Expected number of attempts
b) Probability of success within 3 attempts
c) Expected T-count if each attempt costs 3 T gates

**P1.2** Verify that $\cos^2(\pi/8) \approx 0.854$ using the identity $\cos(\pi/8) = \sqrt{(1 + \cos(\pi/4))/2}$.

**P1.3** For a geometric distribution with success probability $p$, derive the formula for expected value $\mathbb{E}[X] = 1/p$.

### Level 2: Intermediate

**P2.1** Design an RUS circuit for $R_z(\pi/3)$ using:
- One ancilla qubit
- CNOT, H, and T gates
Calculate the success probability.

**P2.2** A circuit requires 50 $R_z(\theta)$ gates where $\theta$ has RUS success probability 0.8 and deterministic cost 40 T gates. RUS costs 3 T gates per attempt. Calculate:
a) Expected T-count using all RUS
b) Expected T-count using all deterministic
c) Hybrid strategy: use RUS for first 40, deterministic for rest

**P2.3** Prove that for an RUS circuit, if the failure operation is Clifford, then correction and retry preserves the original input state.

### Level 3: Challenging

**P3.1** Derive the optimal catalyst state $|C\rangle = \alpha|0\rangle + \beta|1\rangle$ that maximizes success probability for a given RUS unitary.

**P3.2** Analyze the tail bound: prove that for RUS with success probability $p$, the probability of requiring more than $k$ attempts is $(1-p)^k$.

**P3.3** Design a nested RUS scheme for implementing $R_z(\theta)$ where $\theta$ is close to $\pi$ (where simple RUS fails). Analyze the expected T-count.

---

## Computational Lab

```python
"""
Day 885: Repeat-Until-Success Circuits
======================================

Implementing RUS protocols and analysis tools.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# Basic Gates
# =============================================================================

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    """Z-rotation by angle theta."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


def CNOT() -> np.ndarray:
    """CNOT gate (control first qubit, target second)."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)


def kron(*args) -> np.ndarray:
    """Kronecker product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


# =============================================================================
# RUS Circuit Framework
# =============================================================================

@dataclass
class RUSResult:
    """Result of RUS simulation."""
    success: bool
    attempts: int
    final_state: np.ndarray
    total_t_count: int


class RUSCircuit:
    """
    Base class for Repeat-Until-Success circuits.

    Subclasses implement specific RUS protocols.
    """

    def __init__(self, t_per_attempt: int = 2):
        self.t_per_attempt = t_per_attempt
        self._build_circuit()

    def _build_circuit(self):
        """Build the RUS unitary. Override in subclasses."""
        raise NotImplementedError

    def get_success_operator(self) -> np.ndarray:
        """Return the operator applied on success (ancilla in |0⟩)."""
        raise NotImplementedError

    def get_failure_operator(self) -> np.ndarray:
        """Return the operator applied on failure (ancilla in |1⟩)."""
        raise NotImplementedError

    def get_correction(self) -> np.ndarray:
        """Return correction to apply after failure."""
        return I  # Default: no correction needed

    def success_probability(self, input_state: np.ndarray) -> float:
        """Calculate success probability for given input state."""
        V = self.get_success_operator()
        output = V @ input_state
        return float(np.real(np.vdot(output, output)))

    def average_success_probability(self) -> float:
        """Calculate average success probability over Haar-random inputs."""
        V = self.get_success_operator()
        return 0.5 * np.real(np.trace(V.conj().T @ V))

    def expected_t_count(self) -> float:
        """Calculate expected T-count."""
        p = self.average_success_probability()
        if p <= 0:
            return float('inf')
        return self.t_per_attempt / p

    def simulate(self, input_state: np.ndarray,
                 max_attempts: int = 1000) -> RUSResult:
        """
        Simulate RUS protocol on given input state.

        Returns RUSResult with success status, attempts, and final state.
        """
        current_state = input_state.copy()
        V = self.get_success_operator()
        W = self.get_failure_operator()
        correction = self.get_correction()

        for attempt in range(1, max_attempts + 1):
            # Calculate success probability for current state
            p_success = self.success_probability(current_state)

            # Simulate measurement
            if np.random.random() < p_success:
                # Success!
                final_state = V @ current_state
                final_state = final_state / np.linalg.norm(final_state)
                return RUSResult(
                    success=True,
                    attempts=attempt,
                    final_state=final_state,
                    total_t_count=attempt * self.t_per_attempt
                )
            else:
                # Failure - apply correction and continue
                current_state = correction @ W @ current_state
                current_state = current_state / np.linalg.norm(current_state)

        # Max attempts reached
        return RUSResult(
            success=False,
            attempts=max_attempts,
            final_state=current_state,
            total_t_count=max_attempts * self.t_per_attempt
        )


class V3RUS(RUSCircuit):
    """
    RUS circuit for the V3 gate.

    V3 = Rz(theta) where cos(theta/2) = cos(pi/8)cos(pi/5)
    """

    def __init__(self):
        super().__init__(t_per_attempt=2)

    def _build_circuit(self):
        """
        Circuit:
        |ψ⟩ ──●──────H──●──────S†──
              │         │
        |0⟩ ──X──T†──H──X──T──────M
        """
        # Build full unitary on 2 qubits
        # This is a simplified construction for demonstration
        self._success_op = None
        self._failure_op = None
        self._compute_operators()

    def _compute_operators(self):
        """Compute success and failure operators."""
        # For V3, the success operator implements the desired rotation
        theta_v3 = 2 * np.arccos(np.cos(np.pi/8) * np.cos(np.pi/5))
        self._success_op = Rz(theta_v3)
        # Failure gives a Clifford (here we approximate as S)
        self._failure_op = S

    def get_success_operator(self) -> np.ndarray:
        return self._success_op

    def get_failure_operator(self) -> np.ndarray:
        return self._failure_op

    def get_correction(self) -> np.ndarray:
        return Sdg  # Undo the S from failure


class GenericRotationRUS(RUSCircuit):
    """
    Generic RUS circuit for Rz(theta).

    Success probability depends on theta.
    """

    def __init__(self, theta: float, t_per_attempt: int = 4):
        self.theta = theta
        super().__init__(t_per_attempt=t_per_attempt)

    def _build_circuit(self):
        """Compute operators for generic rotation."""
        # Success probability: (1 + cos(theta))/2
        self._success_prob = (1 + np.cos(self.theta)) / 2
        self._success_op = Rz(self.theta)
        self._failure_op = Z @ Rz(self.theta)  # Simplified

    def get_success_operator(self) -> np.ndarray:
        # Scale to get correct success probability
        scale = np.sqrt(self._success_prob) if self._success_prob > 0 else 0
        return scale * self._success_op

    def get_failure_operator(self) -> np.ndarray:
        scale = np.sqrt(1 - self._success_prob) if self._success_prob < 1 else 0
        return scale * self._failure_op

    def success_probability(self, input_state: np.ndarray) -> float:
        return self._success_prob

    def average_success_probability(self) -> float:
        return self._success_prob


# =============================================================================
# Analysis Tools
# =============================================================================

def analyze_rus_statistics(rus: RUSCircuit,
                           n_trials: int = 10000) -> dict:
    """
    Run Monte Carlo analysis of RUS protocol.

    Returns statistics on attempts and T-count.
    """
    input_state = np.array([1, 0], dtype=complex)  # |0⟩

    attempts_list = []
    t_counts = []
    successes = 0

    for _ in range(n_trials):
        result = rus.simulate(input_state)
        if result.success:
            successes += 1
            attempts_list.append(result.attempts)
            t_counts.append(result.total_t_count)

    if not attempts_list:
        return {'success_rate': 0}

    return {
        'success_rate': successes / n_trials,
        'mean_attempts': np.mean(attempts_list),
        'std_attempts': np.std(attempts_list),
        'mean_t_count': np.mean(t_counts),
        'std_t_count': np.std(t_counts),
        'max_attempts': max(attempts_list),
        'theoretical_expected': rus.expected_t_count()
    }


def compare_rus_vs_deterministic(theta: float,
                                 epsilon: float = 1e-8) -> dict:
    """
    Compare RUS and deterministic synthesis for Rz(theta).
    """
    # RUS analysis
    rus = GenericRotationRUS(theta)
    rus_stats = analyze_rus_statistics(rus, n_trials=1000)

    # Deterministic (Ross-Selinger bound)
    det_t_count = 3 * np.log2(1/epsilon) + 10  # Approximate

    return {
        'theta': theta,
        'epsilon': epsilon,
        'rus_expected_t': rus_stats.get('mean_t_count', float('inf')),
        'rus_success_prob': rus.average_success_probability(),
        'det_t_count': det_t_count,
        'winner': 'RUS' if rus_stats.get('mean_t_count', float('inf')) < det_t_count else 'Deterministic'
    }


def catalyst_benefit_analysis(base_success_prob: float,
                              catalyst_prep_cost: int,
                              enhanced_success_prob: float,
                              n_operations: int,
                              base_t_per_attempt: int = 2) -> dict:
    """
    Analyze benefit of using catalyst states.
    """
    # Without catalyst
    without_cost = n_operations * (base_t_per_attempt / base_success_prob)

    # With catalyst
    with_cost = catalyst_prep_cost + n_operations * (base_t_per_attempt / enhanced_success_prob)

    return {
        'without_catalyst_t': without_cost,
        'with_catalyst_t': with_cost,
        'savings': without_cost - with_cost,
        'break_even_ops': catalyst_prep_cost / (
            base_t_per_attempt/base_success_prob - base_t_per_attempt/enhanced_success_prob
        )
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_rus_distribution(rus: RUSCircuit, n_trials: int = 5000):
    """Plot distribution of attempts and T-count for RUS."""
    input_state = np.array([1, 0], dtype=complex)

    attempts = []
    for _ in range(n_trials):
        result = rus.simulate(input_state)
        if result.success:
            attempts.append(result.attempts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of attempts
    axes[0].hist(attempts, bins=range(1, max(attempts)+2), density=True,
                 alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Number of Attempts')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Distribution of RUS Attempts')

    # Theoretical geometric distribution
    p = rus.average_success_probability()
    x = np.arange(1, max(attempts)+1)
    axes[0].plot(x, stats.geom.pmf(x, p), 'r-', lw=2,
                 label=f'Geometric(p={p:.3f})')
    axes[0].legend()

    # CDF
    axes[1].hist(attempts, bins=range(1, max(attempts)+2), density=True,
                 cumulative=True, alpha=0.7, edgecolor='black')
    axes[1].plot(x, stats.geom.cdf(x, p), 'r-', lw=2,
                 label='Theoretical CDF')
    axes[1].set_xlabel('Number of Attempts')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('CDF of RUS Attempts')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('rus_distribution.png', dpi=150)
    plt.close()
    print("Saved: rus_distribution.png")


def plot_theta_comparison():
    """Plot RUS vs deterministic across different rotation angles."""
    thetas = np.linspace(0.1, np.pi - 0.1, 50)
    epsilon = 1e-8

    rus_costs = []
    det_cost = 3 * np.log2(1/epsilon) + 10

    for theta in thetas:
        rus = GenericRotationRUS(theta)
        p = rus.average_success_probability()
        if p > 0:
            rus_costs.append(rus.t_per_attempt / p)
        else:
            rus_costs.append(float('inf'))

    plt.figure(figsize=(10, 6))
    plt.plot(thetas/np.pi, rus_costs, 'b-', lw=2, label='RUS Expected T-count')
    plt.axhline(y=det_cost, color='r', linestyle='--', lw=2,
                label=f'Deterministic (ε={epsilon:.0e})')
    plt.xlabel('θ/π')
    plt.ylabel('T-count')
    plt.title('RUS vs Deterministic Synthesis')
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.savefig('rus_vs_det.png', dpi=150)
    plt.close()
    print("Saved: rus_vs_det.png")


# =============================================================================
# Demo
# =============================================================================

def demo_rus():
    """Demonstrate RUS protocols."""

    print("=" * 70)
    print("Day 885: Repeat-Until-Success Circuits - Demonstration")
    print("=" * 70)

    # Example 1: V3 gate
    print("\n1. V3 Gate RUS Analysis")
    print("-" * 40)

    v3 = V3RUS()
    print(f"Success probability: {v3.average_success_probability():.4f}")
    print(f"T gates per attempt: {v3.t_per_attempt}")
    print(f"Expected T-count: {v3.expected_t_count():.2f}")

    # Simulate
    stats = analyze_rus_statistics(v3, n_trials=10000)
    print(f"\nMonte Carlo simulation (10000 trials):")
    print(f"  Mean attempts: {stats['mean_attempts']:.2f}")
    print(f"  Std attempts: {stats['std_attempts']:.2f}")
    print(f"  Mean T-count: {stats['mean_t_count']:.2f}")
    print(f"  Max attempts seen: {stats['max_attempts']}")

    # Example 2: Compare for different angles
    print("\n2. Angle-Dependent Analysis")
    print("-" * 40)

    for theta in [np.pi/8, np.pi/4, np.pi/2, 3*np.pi/4]:
        rus = GenericRotationRUS(theta)
        p = rus.average_success_probability()
        expected_t = rus.t_per_attempt / p if p > 0 else float('inf')
        print(f"  θ = {theta/np.pi:.3f}π: p = {p:.3f}, E[T] = {expected_t:.1f}")

    # Example 3: RUS vs Deterministic comparison
    print("\n3. RUS vs Deterministic Comparison")
    print("-" * 40)

    for theta in [np.pi/8, np.pi/3, np.pi/2]:
        result = compare_rus_vs_deterministic(theta)
        print(f"  θ = {theta/np.pi:.3f}π:")
        print(f"    RUS E[T]: {result['rus_expected_t']:.1f}")
        print(f"    Det T: {result['det_t_count']:.1f}")
        print(f"    Winner: {result['winner']}")

    # Example 4: Catalyst benefit
    print("\n4. Catalyst State Analysis")
    print("-" * 40)

    result = catalyst_benefit_analysis(
        base_success_prob=0.5,
        catalyst_prep_cost=20,
        enhanced_success_prob=0.9,
        n_operations=100
    )
    print(f"  100 operations:")
    print(f"    Without catalyst: {result['without_catalyst_t']:.0f} T gates")
    print(f"    With catalyst: {result['with_catalyst_t']:.0f} T gates")
    print(f"    Savings: {result['savings']:.0f} T gates")
    print(f"    Break-even: {result['break_even_ops']:.1f} operations")

    # Example 5: Tail bounds
    print("\n5. High-Probability Bounds")
    print("-" * 40)

    p = 0.854  # V3 success probability
    for confidence in [0.9, 0.95, 0.99, 0.999]:
        # P(X > k) = (1-p)^k < 1-confidence
        # k > log(1-confidence) / log(1-p)
        k = int(np.ceil(np.log(1-confidence) / np.log(1-p)))
        t_bound = k * 2  # 2 T gates per attempt
        print(f"  {confidence*100:.1f}% confidence: T ≤ {t_bound}")

    # Generate plots
    print("\n6. Generating Plots...")
    print("-" * 40)
    try:
        plot_rus_distribution(v3)
        plot_theta_comparison()
    except Exception as e:
        print(f"  Plotting skipped: {e}")


if __name__ == "__main__":
    demo_rus()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Expected iterations | $\mathbb{E}[k] = 1/p$ |
| Expected T-count | $\mathbb{E}[T] = T_0/p$ |
| Success probability (generic) | $p(\theta) = (1 + \cos\theta)/2$ |
| Variance of iterations | $\text{Var}[k] = (1-p)/p^2$ |
| Tail bound | $P(k > n) = (1-p)^n$ |
| Catalyst break-even | $n_{\text{break}} = C_{\text{prep}}/(T_{\text{old}} - T_{\text{new}})$ |

### Main Takeaways

1. **RUS circuits** offer probabilistic gate synthesis with potentially constant expected T-count
2. **Success probability** determines expected cost and varies with target rotation
3. **Catalyst states** can dramatically improve success probability
4. **Trade-offs** exist between worst-case guarantees and expected performance
5. **Hybrid strategies** combine RUS for favorable gates with deterministic for others
6. **Geometric distribution** governs attempt statistics, enabling precise analysis

---

## Daily Checklist

- [ ] I can explain the RUS paradigm and its advantages
- [ ] I can calculate expected T-count for an RUS protocol
- [ ] I understand how success probability affects performance
- [ ] I know what catalyst states are and how they help
- [ ] I can compare RUS vs. deterministic synthesis
- [ ] I understand when to use each approach

---

## Preview: Day 886

Tomorrow we explore **Lattice Surgery Scheduling**:

- Translating logical gates to surgery primitives
- Constructing instruction dependency graphs
- Scheduling algorithms for surgery operations
- Managing ancilla patches and routing
- Optimizing merge/split operation timing

Lattice surgery is how we actually execute logical gates on the surface code.
