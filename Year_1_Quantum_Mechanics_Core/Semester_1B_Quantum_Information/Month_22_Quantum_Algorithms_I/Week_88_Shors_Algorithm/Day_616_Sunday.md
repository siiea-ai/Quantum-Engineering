# Day 616: Month 22 Review - Quantum Algorithms I

## Overview

**Day 616** | Week 88, Day 7 | Month 22 | Quantum Algorithms I

Today we consolidate our understanding of all quantum algorithms covered this month: the Deutsch-Jozsa family, Quantum Fourier Transform, Phase Estimation, and Shor's Algorithm.

---

## Learning Objectives

1. Synthesize all Month 22 concepts
2. Compare algorithm structures and speedups
3. Solve comprehensive problems
4. Connect algorithms to applications
5. Preview Month 23: Quantum Algorithms II

---

## Month Summary

### Week 85: Deutsch-Jozsa Family

| Day | Topic | Key Ideas |
|-----|-------|-----------|
| 589 | Quantum Query Model | Oracle access, query complexity |
| 590 | Deutsch's Algorithm | 1 query vs 2 classical, phase kickback |
| 591 | Deutsch-Jozsa | n qubits, exponential speedup for promise |
| 592 | Bernstein-Vazirani | Hidden string, 1 query vs n |
| 593 | Simon's Algorithm | Hidden period, exponential speedup |
| 594 | Oracle Construction | Building quantum oracles |
| 595 | Week Review | Synthesis and comparisons |

**Key Theme:** Quantum parallelism + interference = query advantage

### Week 86: Quantum Fourier Transform

| Day | Topic | Key Ideas |
|-----|-------|-----------|
| 596 | Discrete Fourier Transform | Classical DFT, $O(N^2)$ |
| 597 | QFT Definition | Unitary on amplitudes, $O(n^2)$ gates |
| 598 | QFT Circuit | H + controlled-R_k pattern |
| 599 | Phase Kickback | Eigenvalues of controlled gates |
| 600 | Inverse QFT | Reversing circuit, measurement patterns |
| 601 | Approximate QFT | Truncating small phases |
| 602 | Week Review | QFT applications |

**Key Theme:** Fourier analysis on quantum states enables phase detection

### Week 87: Phase Estimation

| Day | Topic | Key Ideas |
|-----|-------|-----------|
| 603 | Eigenvalue Problem | Unitary eigenvalues are phases |
| 604 | QPE Circuit | H + controlled-U + QFT$^{-1}$ |
| 605 | Precision Analysis | n bits, $2^n$ amplification |
| 606 | Success Probability | $\geq 4/\pi^2$, boosting methods |
| 607 | Iterative QPE | Sequential bit extraction |
| 608 | Kitaev's Algorithm | Statistical, fault-tolerant |
| 609 | Week Review | QPE variants comparison |

**Key Theme:** Extract eigenvalue phases with polynomial resources

### Week 88: Shor's Algorithm

| Day | Topic | Key Ideas |
|-----|-------|-----------|
| 610 | Factoring to Order-Finding | Classical reduction |
| 611 | Number Theory | Euler, CRT, group structure |
| 612 | Quantum Period-Finding | QPE on $U_a$ |
| 613 | Continued Fractions | Classical post-processing |
| 614 | Full Shor's Algorithm | Complete integration |
| 615 | Complexity Analysis | $O(n^3)$ vs sub-exponential |
| 616 | Month Review | This day! |

**Key Theme:** Period-finding solves factoring in polynomial time

---

## Master Algorithm Comparison

### Speedup Summary

| Algorithm | Classical | Quantum | Type | Advantage |
|-----------|-----------|---------|------|-----------|
| Deutsch | 2 queries | 1 query | Bounded error | 2× |
| Deutsch-Jozsa | $2^{n-1}+1$ | 1 query | Exact | Exponential |
| Bernstein-Vazirani | $n$ | 1 query | Exact | $n×$ |
| Simon's | $O(2^{n/2})$ | $O(n)$ | Bounded error | Exponential |
| QFT | $O(N^2)$ or $O(N\log N)$ | $O(n^2)$ | Unitary | Exponential |
| QPE | Exponential | $O(n)$ | Bounded error | Exponential |
| Shor | Sub-exponential | $O(n^3)$ | Bounded error | Exponential |

### Structural Patterns

**Pattern 1: Superposition → Oracle → Interference → Measure**
```
|0⟩^n ──[H^⊗n]──[Oracle]──[Interference]──[Measure]
```
Used in: Deutsch, Deutsch-Jozsa, Bernstein-Vazirani, Simon's

**Pattern 2: Superposition → Controlled-U → QFT$^{-1}$ → Measure**
```
|0⟩^n ──[H^⊗n]──[Controlled-U^{2^k}]──[QFT^{-1}]──[Measure]
```
Used in: QPE, Shor's algorithm

### Quantum Techniques Used

| Technique | Where Used | Purpose |
|-----------|------------|---------|
| Superposition | All algorithms | Parallel evaluation |
| Phase kickback | DJ, BV, QPE, Shor | Encode info in phases |
| Interference | All algorithms | Amplify correct answers |
| QFT | QPE, Shor | Phase → computational basis |
| Entanglement | Simon, QPE | Correlate registers |

---

## Comprehensive Problems

### Problem 1: Algorithm Selection

For each task, identify the best quantum algorithm:

(a) Determine if $f:\{0,1\}^{10} \to \{0,1\}$ is balanced or constant
(b) Find hidden string $s$ where $f(x) = s \cdot x$
(c) Find period $r$ where $f(x) = f(x \oplus r)$
(d) Factor $N = 1147$
(e) Find eigenvalue of unitary $U$

**Solutions:**
(a) Deutsch-Jozsa: 1 query vs 513 classical
(b) Bernstein-Vazirani: 1 query vs 10 classical
(c) Simon's algorithm: $O(10)$ queries vs $O(2^5)$ classical
(d) Shor's algorithm: $O(n^3)$ vs sub-exponential classical
(e) Quantum Phase Estimation: $O(n)$ controlled-U operations

### Problem 2: Circuit Analysis

Draw the complete circuit for Shor's algorithm on $N = 15$ with $a = 7$ using 4 ancilla qubits.

**Solution:**
```
|0⟩ ─[H]─────●──────────────────────────────[QFT^{-1}]─ m₀
             │
|0⟩ ─[H]─────┼────●─────────────────────────[QFT^{-1}]─ m₁
             │    │
|0⟩ ─[H]─────┼────┼────●────────────────────[QFT^{-1}]─ m₂
             │    │    │
|0⟩ ─[H]─────┼────┼────┼────●───────────────[QFT^{-1}]─ m₃
             │    │    │    │
|1⟩ ────────[×7⁸]─[×7⁴]─[×7²]─[×7]──────────────────────
             mod 15

where ×7^k represents multiplication by 7^k mod 15:
- 7¹ ≡ 7
- 7² ≡ 4
- 7⁴ ≡ 1
- 7⁸ ≡ 1
```

### Problem 3: Success Probability Chain

Calculate the overall success probability for factoring $N = 21$ with Shor's algorithm, given:
- QPE success: 80%
- Even period: 75%
- Non-trivial factor: 90%

**Solution:**
$$P(\text{success}) = 0.80 \times 0.75 \times 0.90 = 0.54 = 54\%$$

Expected attempts to success: $1/0.54 \approx 1.85$

### Problem 4: Resource Comparison

Compare resources for:
(a) Bernstein-Vazirani on $n = 100$ bits
(b) Shor on $N$ with 100 bits

**Solution:**

(a) **Bernstein-Vazirani:**
- Qubits: $n + 1 = 101$
- Oracle calls: 1
- Hadamards: $2n + 2 = 202$
- Total gates: $O(n) = O(100)$

(b) **Shor:**
- Ancilla qubits: $2n = 200$
- Work qubits: $n + O(n) = O(100)$
- Total qubits: $O(300)$
- Gates: $O(n^3) = O(10^6)$

---

## Key Formulas Reference

### Query Complexity

| Algorithm | Queries |
|-----------|---------|
| Deutsch | 1 |
| Deutsch-Jozsa | 1 |
| Bernstein-Vazirani | 1 |
| Simon | $O(n)$ |

### QFT

$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi ijk/N} |k\rangle$$

$$\text{QFT} = \frac{1}{\sqrt{N}} \begin{pmatrix} 1 & 1 & 1 & \cdots \\ 1 & \omega & \omega^2 & \cdots \\ 1 & \omega^2 & \omega^4 & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

### Phase Estimation

$$U|u\rangle = e^{2\pi i\phi}|u\rangle \xrightarrow{\text{QPE}} |\tilde{\phi}\rangle$$

Precision: $|\phi - \tilde{\phi}| < 1/2^n$ with probability $\geq 4/\pi^2$

### Shor's Algorithm

$$a^r \equiv 1 \pmod{N} \implies \gcd(a^{r/2} \pm 1, N) = \text{factor}$$

Complexity: $O(n^3)$ gates, $O(n)$ qubits

---

## Computational Lab

```python
"""
Day 616: Month 22 Comprehensive Review
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from fractions import Fraction

# ============================================================
# PART 1: Unified Algorithm Framework
# ============================================================

class QuantumAlgorithm:
    """Base class for quantum algorithms."""

    def __init__(self, name: str, n_qubits: int):
        self.name = name
        self.n_qubits = n_qubits
        self.state = None
        self.measurements = []

    def hadamard_layer(self, state: np.ndarray) -> np.ndarray:
        """Apply H^⊗n."""
        n = int(np.log2(len(state)))
        H_n = np.ones((2**n, 2**n)) / np.sqrt(2**n)
        for i in range(2**n):
            for j in range(2**n):
                parity = bin(i & j).count('1')
                H_n[i, j] *= (-1) ** parity
        return H_n @ state

    def qft(self, state: np.ndarray) -> np.ndarray:
        """Apply QFT."""
        N = len(state)
        omega = np.exp(2j * np.pi / N)
        F = np.array([[omega ** (j * k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
        return F @ state

    def inverse_qft(self, state: np.ndarray) -> np.ndarray:
        """Apply QFT^{-1}."""
        return np.conj(self.qft(np.conj(state)))

class DeutschJozsa(QuantumAlgorithm):
    """Deutsch-Jozsa algorithm."""

    def __init__(self, n: int, oracle_type: str = 'balanced'):
        super().__init__('Deutsch-Jozsa', n)
        self.oracle_type = oracle_type

    def run(self) -> str:
        N = 2 ** self.n_qubits

        # Initial state |0...0⟩
        state = np.zeros(N, dtype=complex)
        state[0] = 1.0

        # Apply H^⊗n
        state = self.hadamard_layer(state)

        # Apply oracle (phase oracle)
        if self.oracle_type == 'constant':
            # f(x) = 0 for all x: do nothing
            pass
        else:
            # Balanced: flip phase for first half
            for i in range(N // 2):
                state[i] *= -1

        # Apply H^⊗n again
        state = self.hadamard_layer(state)

        # Measure probability of |0...0⟩
        prob_zero = abs(state[0]) ** 2

        if prob_zero > 0.5:
            return 'constant'
        else:
            return 'balanced'

class QuantumPhaseEstimation(QuantumAlgorithm):
    """Quantum Phase Estimation."""

    def __init__(self, n_ancilla: int, true_phase: float):
        super().__init__('QPE', n_ancilla)
        self.true_phase = true_phase

    def run(self) -> Tuple[float, int]:
        N = 2 ** self.n_qubits

        # Initial superposition
        state = np.ones(N, dtype=complex) / np.sqrt(N)

        # Apply controlled-U operations (encode phase)
        for j in range(N):
            phase = 2 * np.pi * j * self.true_phase
            state[j] *= np.exp(1j * phase)

        # Apply inverse QFT
        state = self.inverse_qft(state)

        # Measure (find peak)
        probs = np.abs(state) ** 2
        measured = np.argmax(probs)

        estimated_phase = measured / N

        return estimated_phase, measured

class SimplifiedShor(QuantumAlgorithm):
    """Simplified Shor's algorithm demonstration."""

    def __init__(self, N: int, a: int):
        super().__init__('Shor', int(2 * np.ceil(np.log2(N))))
        self.N = N
        self.a = a
        self.true_order = self._compute_order()

    def _compute_order(self) -> int:
        """Compute order of a mod N classically."""
        if np.gcd(self.a, self.N) != 1:
            return -1
        order = 1
        current = self.a % self.N
        while current != 1 and order < self.N:
            current = (current * self.a) % self.N
            order += 1
        return order

    def run(self) -> Tuple[int, bool]:
        """Run simplified Shor's algorithm."""
        r = self.true_order

        if r <= 0:
            return -1, False

        if r % 2 == 1:
            return r, False

        half_power = pow(self.a, r // 2, self.N)
        if half_power == self.N - 1:
            return r, False

        g1 = np.gcd(half_power - 1, self.N)
        g2 = np.gcd(half_power + 1, self.N)

        if 1 < g1 < self.N:
            return g1, True
        if 1 < g2 < self.N:
            return g2, True

        return r, False

# ============================================================
# PART 2: Algorithm Demonstrations
# ============================================================

print("MONTH 22 REVIEW: QUANTUM ALGORITHMS I")
print("="*60)

# Demo 1: Deutsch-Jozsa
print("\n1. DEUTSCH-JOZSA ALGORITHM")
print("-"*40)

for n in [3, 5, 8]:
    for oracle in ['constant', 'balanced']:
        dj = DeutschJozsa(n, oracle)
        result = dj.run()
        status = "✓" if result == oracle else "✗"
        print(f"  n={n}, oracle={oracle}: detected={result} {status}")

# Demo 2: QPE
print("\n2. QUANTUM PHASE ESTIMATION")
print("-"*40)

test_phases = [0.25, 0.375, 0.6, 1/3, 1/7]
n_bits = 8

print(f"Using {n_bits} ancilla bits:")
for phi in test_phases:
    qpe = QuantumPhaseEstimation(n_bits, phi)
    estimated, m = qpe.run()
    error = abs(phi - estimated)
    print(f"  True φ={phi:.4f}, Estimated={estimated:.4f}, Error={error:.2e}")

# Demo 3: Shor
print("\n3. SHOR'S ALGORITHM")
print("-"*40)

test_cases = [(15, 7), (21, 2), (33, 2), (35, 3), (55, 2), (77, 3)]

for N, a in test_cases:
    shor = SimplifiedShor(N, a)
    result, success = shor.run()
    if success:
        print(f"  N={N}, a={a}: Factor found = {result}, N = {result} × {N//result} ✓")
    else:
        print(f"  N={N}, a={a}: Order = {result}, no factor (need retry)")

# ============================================================
# PART 3: Complexity Comparison Visualization
# ============================================================

print("\n4. COMPLEXITY COMPARISON")
print("-"*40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Query complexity
ax1 = axes[0, 0]
n_vals = np.arange(1, 20)

classical_dj = 2**(n_vals - 1) + 1
quantum_dj = np.ones_like(n_vals)

classical_bv = n_vals
quantum_bv = np.ones_like(n_vals)

classical_simon = 2**(n_vals / 2)
quantum_simon = n_vals

ax1.semilogy(n_vals, classical_dj, 'b-', label='Classical DJ', linewidth=2)
ax1.semilogy(n_vals, quantum_dj, 'b--', label='Quantum DJ', linewidth=2)
ax1.semilogy(n_vals, classical_simon, 'r-', label='Classical Simon', linewidth=2)
ax1.semilogy(n_vals, quantum_simon, 'r--', label='Quantum Simon', linewidth=2)
ax1.set_xlabel('Number of bits n')
ax1.set_ylabel('Query complexity')
ax1.set_title('Query Complexity Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Shor vs Classical factoring
ax2 = axes[0, 1]
n_bits = np.array([64, 128, 256, 512, 1024, 2048])

shor_ops = n_bits ** 3
gnfs_log = 1.9 * (n_bits ** (1/3)) * (np.log(n_bits) ** (2/3))

ax2.semilogy(n_bits, shor_ops, 'g-o', label="Shor's Algorithm", linewidth=2)
ax2_twin = ax2.twinx()
ax2_twin.plot(n_bits, gnfs_log, 'r-s', label='log(GNFS)', linewidth=2)

ax2.set_xlabel('Bit size')
ax2.set_ylabel('Shor operations', color='green')
ax2_twin.set_ylabel('log(Classical ops)', color='red')
ax2.set_title("Factoring: Quantum vs Classical")
ax2.grid(True, alpha=0.3)

# Plot 3: QPE precision
ax3 = axes[1, 0]
n_ancilla = np.arange(2, 16)
precision = 1 / (2 ** n_ancilla)

ax3.semilogy(n_ancilla, precision, 'purple', linewidth=2, marker='o')
ax3.set_xlabel('Number of ancilla qubits')
ax3.set_ylabel('Phase precision')
ax3.set_title('QPE Precision vs Ancilla Qubits')
ax3.grid(True, alpha=0.3)

# Plot 4: Algorithm resource comparison
ax4 = axes[1, 1]
algorithms = ['Deutsch', 'D-J', 'B-V', 'Simon', 'QPE', 'Shor']
queries = [1, 1, 1, 10, 10, 10]  # For n=10
gates = [2, 21, 21, 210, 210, 10000]  # Approximate
qubits = [2, 11, 11, 20, 20, 30]  # Approximate for n=10

x = np.arange(len(algorithms))
width = 0.25

bars1 = ax4.bar(x - width, queries, width, label='Queries', color='blue', alpha=0.7)
bars2 = ax4.bar(x, [g/100 for g in gates], width, label='Gates/100', color='green', alpha=0.7)
bars3 = ax4.bar(x + width, qubits, width, label='Qubits', color='red', alpha=0.7)

ax4.set_ylabel('Resource count')
ax4.set_title('Resource Comparison (n=10)')
ax4.set_xticks(x)
ax4.set_xticklabels(algorithms)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('month22_review.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to 'month22_review.png'")

# ============================================================
# PART 4: Summary Statistics
# ============================================================

print("\n" + "="*60)
print("MONTH 22 SUMMARY")
print("="*60)

print("""
ALGORITHMS MASTERED:
--------------------
1. Deutsch-Jozsa Family
   - Deutsch: First quantum algorithm, 2× speedup
   - Deutsch-Jozsa: Exponential speedup (promise problem)
   - Bernstein-Vazirani: Linear speedup
   - Simon's: Exponential speedup (with classical post-processing)

2. Quantum Fourier Transform
   - Definition and circuit implementation
   - O(n²) gates vs O(N log N) classical FFT
   - Foundation for phase estimation

3. Quantum Phase Estimation
   - Extract eigenvalue phases
   - Multiple variants: standard, iterative, Kitaev
   - Core subroutine for many algorithms

4. Shor's Algorithm
   - Polynomial-time factoring
   - Threatens RSA cryptography
   - Combines QPE with number theory

KEY CONCEPTS:
-------------
- Quantum parallelism through superposition
- Phase kickback mechanism
- Constructive/destructive interference
- QFT as phase-to-computational basis converter
- Classical post-processing (continued fractions)

LOOKING AHEAD TO MONTH 23:
--------------------------
- Grover's search algorithm
- Amplitude amplification
- Quantum walks
- Variational quantum algorithms (VQE, QAOA)
- Quantum machine learning basics
""")
```

---

## Summary

### Algorithm Mastery

| Week | Algorithm Family | Key Technique | Speedup Type |
|------|------------------|---------------|--------------|
| 85 | Deutsch-Jozsa | Phase kickback | Query |
| 86 | QFT | Fourier analysis | Computational |
| 87 | QPE | Eigenvalue extraction | Exponential |
| 88 | Shor | Period-finding | Exponential |

### Key Takeaways

1. **Quantum algorithms** exploit superposition and interference
2. **Query complexity** can be exponentially better
3. **QFT** is the foundation of many algorithms
4. **QPE** extracts eigenvalue information efficiently
5. **Shor's algorithm** is the most famous practical application

---

## Preview: Month 23

**Quantum Algorithms II** will cover:
- Week 89: Grover's Search Algorithm
- Week 90: Amplitude Amplification
- Week 91: Quantum Walks
- Week 92: Variational Algorithms

---

## Daily Checklist

- [ ] I can compare all Month 22 algorithms
- [ ] I understand when to use each algorithm
- [ ] I can estimate resources for each
- [ ] I see the connections between algorithms
- [ ] I'm ready for Month 23

---

*End of Month 22 | Next: Month 23 - Quantum Algorithms II*
