# Day 623: Week 89 Review - Grover's Search Algorithm

## Overview
**Day 623** | Week 89, Day 7 | Year 1, Month 23 | Week Review and Synthesis

Today we consolidate our understanding of Grover's search algorithm through comprehensive review, challenging problems, and connections to the broader quantum computing landscape.

---

## Week Summary

### Topics Covered

| Day | Topic | Key Insight |
|-----|-------|-------------|
| 617 | Unstructured Search | Classical O(N) vs Quantum O(√N) |
| 618 | Grover Oracle | Phase kickback marks solutions |
| 619 | Diffusion Operator | Inversion about the mean |
| 620 | Geometric Interpretation | Rotation in 2D subspace |
| 621 | Optimal Iterations | k ≈ (π/4)√N is optimal and tight |
| 622 | Multiple Solutions | M solutions: k ≈ (π/4)√(N/M) |

### The Complete Grover Algorithm

```
Algorithm: Grover's Search
Input: Oracle O_f marking M solutions in N items
Output: A marked item with high probability

1. Initialize: |ψ⟩ = H^⊗n |0⟩^⊗n
2. Repeat k = ⌊(π/4)√(N/M)⌋ times:
   a. Apply Oracle: |ψ⟩ ← O_f |ψ⟩
   b. Apply Diffusion: |ψ⟩ ← D |ψ⟩
3. Measure in computational basis
4. Verify result (if needed)
```

---

## Key Formulas Summary

### Core Operators

| Operator | Definition | Action |
|----------|------------|--------|
| Oracle | $O_f = I - 2\sum_{w}\|w\rangle\langle w\|$ | Flips phase of solutions |
| Diffusion | $D = 2\|\psi_0\rangle\langle\psi_0\| - I$ | Inverts about mean |
| Grover | $G = D \cdot O_f$ | Rotates by 2θ |

### Complexity Results

$$\boxed{\text{Classical: } \Omega(N) \quad \text{Quantum: } O(\sqrt{N}) \quad \text{Speedup: } \sqrt{N}}$$

### Success Probability

$$P_{success}(k) = \sin^2((2k+1)\theta), \quad \sin\theta = \sqrt{M/N}$$

$$P_{success}(k_{opt}) \geq 1 - \frac{M}{N}$$

---

## Comprehensive Practice Problems

### Problem Set A: Fundamentals

**A1. Oracle Construction**
Construct the phase oracle matrix for marking states $|010\rangle$ and $|101\rangle$ in a 3-qubit system. Verify it satisfies $O^2 = I$.

**A2. Diffusion Calculation**
For $n = 3$ qubits, compute the diffusion operator explicitly and verify it equals $H^{\otimes 3}(2|0\rangle\langle 0| - I)H^{\otimes 3}$.

**A3. Iteration Trace**
For $N = 8$ with marked state $|111\rangle$, trace through 2 complete Grover iterations, showing the state vector after each operation.

### Problem Set B: Analysis

**B1. Optimality Analysis**
Prove that for $N = 4$ with one marked state, exactly one Grover iteration gives success probability 1.

**B2. Suboptimal Iterations**
For $N = 256$ with $M = 1$:
a) Find $k_{opt}$ and $P_{success}(k_{opt})$
b) Calculate $P_{success}$ for $k = k_{opt} \pm 3$
c) How many iterations until $P_{success}$ returns to its initial value?

**B3. Crossover Point**
Find the smallest $M$ (as a function of $N$) such that classical search outperforms Grover.

### Problem Set C: Applications

**C1. Database Search**
A password database has $N = 2^{40}$ entries. A password hash matches $M = 100$ entries (collision attack).
a) How many classical queries needed (expected)?
b) How many Grover iterations?
c) What is the speedup factor?

**C2. SAT Solving**
A 3-SAT formula has $n = 20$ variables and is known to have exactly $M = 2^{10}$ satisfying assignments.
a) How many Grover iterations to find a solution?
b) What is the success probability?
c) Compare to random classical search.

**C3. Cryptographic Key Search**
To break a symmetric cipher with $k$-bit key:
a) Classical brute force: how many operations?
b) Grover search: how many operations?
c) What effective key length provides equivalent security against Grover?

---

## Solutions to Selected Problems

### Solution A1: Oracle Construction

For marked states $|010\rangle$ (index 2) and $|101\rangle$ (index 5):

$$O_f = I_8 - 2(|010\rangle\langle 010| + |101\rangle\langle 101|)$$

Matrix form (8×8, showing diagonal):
$$\text{diag}(O_f) = (1, 1, -1, 1, 1, -1, 1, 1)$$

Verification: $O_f^2 = I$ since each diagonal entry squared equals 1.

### Solution B1: N=4 Optimality

For $N = 4$, $M = 1$: $\theta = \arcsin(1/2) = \pi/6$

After 1 iteration: $(2 \cdot 1 + 1) \cdot \pi/6 = \pi/2$

$P_{success} = \sin^2(\pi/2) = 1$

The state has rotated exactly to $|w\rangle$.

### Solution C3: Cryptographic Key Search

a) Classical: $2^k$ operations (exhaustive search)

b) Grover: $\frac{\pi}{4}\sqrt{2^k} = \frac{\pi}{4} \cdot 2^{k/2}$ operations

c) To match $2^{128}$ security:
   - Grover gives $2^{k/2}$ effective operations
   - Need $k/2 = 128$, so $k = 256$ bits

**Key insight:** Grover effectively halves symmetric key lengths!

---

## Computational Lab: Complete Implementation

```python
"""Day 623: Complete Grover's Algorithm Implementation"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class GroverSearch:
    """Complete Grover's search algorithm implementation."""

    def __init__(self, n_qubits: int, marked_states: List[int]):
        """
        Initialize Grover's algorithm.

        Args:
            n_qubits: Number of qubits
            marked_states: List of marked state indices
        """
        self.n = n_qubits
        self.N = 2**n_qubits
        self.marked = marked_states
        self.M = len(marked_states)

        # Precompute operators
        self._build_operators()
        self._compute_parameters()

    def _build_operators(self):
        """Build oracle and diffusion operators."""
        # Oracle
        self.oracle = np.eye(self.N)
        for m in self.marked:
            self.oracle[m, m] = -1

        # Diffusion
        psi_0 = np.ones(self.N) / np.sqrt(self.N)
        self.diffusion = 2 * np.outer(psi_0, psi_0) - np.eye(self.N)

        # Grover operator
        self.G = self.diffusion @ self.oracle

    def _compute_parameters(self):
        """Compute algorithm parameters."""
        self.theta = np.arcsin(np.sqrt(self.M / self.N))
        self.k_opt = int(np.floor(np.pi / (4 * self.theta))) if self.theta < np.pi/4 else 0

    def initial_state(self) -> np.ndarray:
        """Return the initial uniform superposition."""
        return np.ones(self.N) / np.sqrt(self.N)

    def run(self, iterations: int = None) -> Tuple[np.ndarray, float]:
        """
        Run Grover's algorithm.

        Args:
            iterations: Number of iterations (default: optimal)

        Returns:
            Final state vector and success probability
        """
        if iterations is None:
            iterations = self.k_opt

        state = self.initial_state()

        for _ in range(iterations):
            state = self.G @ state

        # Success probability
        prob = sum(abs(state[m])**2 for m in self.marked)

        return state, prob

    def theoretical_success_prob(self, k: int) -> float:
        """Theoretical success probability after k iterations."""
        return np.sin((2*k + 1) * self.theta)**2

    def measure(self, state: np.ndarray) -> int:
        """Simulate measurement on the state."""
        probs = np.abs(state)**2
        return np.random.choice(self.N, p=probs)

    def search_with_verification(self, max_attempts: int = 10) -> Tuple[int, int]:
        """
        Run Grover with verification.

        Returns:
            (result, total_iterations) or (-1, iterations) if failed
        """
        total_iterations = 0

        for _ in range(max_attempts):
            state, _ = self.run()
            total_iterations += self.k_opt

            result = self.measure(state)
            if result in self.marked:
                return result, total_iterations

        return -1, total_iterations

    def amplitude_history(self, max_k: int = None) -> dict:
        """Track amplitude evolution over iterations."""
        if max_k is None:
            max_k = 3 * self.k_opt if self.k_opt > 0 else 10

        history = {
            'k': list(range(max_k + 1)),
            'marked_amp': [],
            'marked_prob': [],
            'other_prob': []
        }

        state = self.initial_state()

        for k in range(max_k + 1):
            if k > 0:
                state = self.G @ state

            marked_prob = sum(abs(state[m])**2 for m in self.marked)
            other_prob = 1 - marked_prob

            # Average marked amplitude (for visualization)
            marked_amp = np.mean([state[m].real for m in self.marked])

            history['marked_amp'].append(marked_amp)
            history['marked_prob'].append(marked_prob)
            history['other_prob'].append(other_prob)

        return history

    def visualize(self):
        """Create comprehensive visualization."""
        history = self.amplitude_history()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Probability evolution
        ax1 = axes[0, 0]
        ax1.plot(history['k'], history['marked_prob'], 'ro-',
                 label='P(marked)', linewidth=2, markersize=6)
        ax1.axvline(x=self.k_opt, color='green', linestyle='--',
                    label=f'Optimal k={self.k_opt}')
        ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Success Probability (N={self.N}, M={self.M})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. State vector after optimal iterations
        ax2 = axes[0, 1]
        state, _ = self.run()
        colors = ['red' if i in self.marked else 'blue' for i in range(self.N)]
        ax2.bar(range(self.N), np.abs(state)**2, color=colors, alpha=0.7)
        ax2.set_xlabel('State Index')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'State Distribution (k={self.k_opt})')
        if self.N <= 16:
            ax2.set_xticks(range(self.N))
            ax2.set_xticklabels([f'{i:0{self.n}b}' for i in range(self.N)], rotation=45)

        # 3. Amplitude evolution
        ax3 = axes[1, 0]
        ax3.plot(history['k'], history['marked_amp'], 'r-',
                 label='Marked amplitude', linewidth=2)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax3.axhline(y=1/np.sqrt(self.N), color='blue', linestyle=':',
                    label='Initial amplitude')
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Amplitude Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Theoretical vs actual
        ax4 = axes[1, 1]
        k_range = history['k']
        theoretical = [self.theoretical_success_prob(k) for k in k_range]
        ax4.plot(k_range, history['marked_prob'], 'bo-', label='Simulated',
                 markersize=4)
        ax4.plot(k_range, theoretical, 'r--', label='Theoretical', linewidth=2)
        ax4.set_xlabel('Iterations')
        ax4.set_ylabel('Success Probability')
        ax4.set_title('Theory vs Simulation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('grover_complete.png', dpi=150, bbox_inches='tight')
        plt.show()

    def summary(self):
        """Print algorithm summary."""
        print(f"\n{'='*50}")
        print(f"Grover's Algorithm Summary")
        print(f"{'='*50}")
        print(f"Number of qubits: {self.n}")
        print(f"Search space size: N = {self.N}")
        print(f"Number of solutions: M = {self.M}")
        print(f"Marked states: {self.marked}")
        print(f"\nParameters:")
        print(f"  θ = {np.degrees(self.theta):.4f}°")
        print(f"  Optimal iterations: {self.k_opt}")
        print(f"  Theoretical P_success: {self.theoretical_success_prob(self.k_opt):.6f}")
        print(f"\nComplexity:")
        print(f"  Classical (expected): {self.N/(2*self.M):.1f} queries")
        print(f"  Quantum: {self.k_opt} iterations")
        print(f"  Speedup: {self.N/(2*self.M)/max(1,self.k_opt):.1f}x")


def run_complete_demo():
    """Run complete demonstration of Grover's algorithm."""

    # Demo 1: Basic search
    print("\n" + "="*60)
    print("DEMO 1: Basic Grover Search")
    print("="*60)

    grover = GroverSearch(n_qubits=4, marked_states=[7])
    grover.summary()

    state, prob = grover.run()
    print(f"\nAfter {grover.k_opt} iterations:")
    print(f"  Actual success probability: {prob:.6f}")

    # Verify with multiple measurements
    successes = sum(1 for _ in range(1000) if grover.measure(state) in grover.marked)
    print(f"  Empirical success rate: {successes/1000:.3f}")

    grover.visualize()

    # Demo 2: Multiple solutions
    print("\n" + "="*60)
    print("DEMO 2: Multiple Solutions")
    print("="*60)

    grover_multi = GroverSearch(n_qubits=5, marked_states=[3, 7, 15, 31])
    grover_multi.summary()
    grover_multi.visualize()

    # Demo 3: Search with verification
    print("\n" + "="*60)
    print("DEMO 3: Search with Verification")
    print("="*60)

    grover_verify = GroverSearch(n_qubits=6, marked_states=[42])

    results = []
    for trial in range(10):
        result, total_k = grover_verify.search_with_verification()
        results.append((result, total_k, result == 42))
        print(f"  Trial {trial+1}: Found {result}, iterations = {total_k}, "
              f"Correct: {result == 42}")

    success_rate = sum(1 for r in results if r[2]) / len(results)
    avg_iterations = np.mean([r[1] for r in results])
    print(f"\n  Success rate: {success_rate:.1%}")
    print(f"  Average iterations: {avg_iterations:.1f}")

    # Demo 4: Complexity comparison
    print("\n" + "="*60)
    print("DEMO 4: Complexity Scaling")
    print("="*60)

    print(f"\n{'N':>10} | {'M':>5} | {'k_opt':>6} | {'Classical':>12} | {'Speedup':>10}")
    print("-"*55)

    for n in [4, 6, 8, 10, 12]:
        N = 2**n
        for M in [1, N//16]:
            if M < 1:
                continue
            grover_test = GroverSearch(n_qubits=n, marked_states=list(range(M)))
            classical = N / (2*M)
            speedup = classical / max(1, grover_test.k_opt)
            print(f"{N:>10} | {M:>5} | {grover_test.k_opt:>6} | "
                  f"{classical:>12.1f} | {speedup:>10.1f}x")


if __name__ == "__main__":
    run_complete_demo()
```

---

## Connections and Extensions

### Relation to Other Algorithms

1. **Quantum Counting (Week 90):** Uses phase estimation on G to estimate M
2. **Amplitude Amplification (Week 90):** Generalizes Grover to arbitrary initial states
3. **Quantum Walks (Week 91):** Alternative framework achieving similar speedups

### Applications in Quantum Computing

- **Cryptanalysis:** Breaking symmetric ciphers (halves effective key length)
- **SAT Solving:** Heuristic speedup for constraint satisfaction
- **Database Search:** Genuine quadratic speedup
- **Optimization:** Subroutine in variational algorithms

### Limitations

1. **Structured problems:** Better algorithms exist (e.g., Simon's, Shor's)
2. **Oracle construction:** May dominate complexity
3. **QRAM assumption:** Database must be queryable quantumly
4. **No exponential speedup:** Only quadratic improvement

---

## Week 89 Assessment Checklist

### Conceptual Understanding
- [ ] I can explain why classical search requires O(N) queries
- [ ] I understand the phase oracle and diffusion operator
- [ ] I can derive the geometric rotation interpretation
- [ ] I know why Grover's algorithm is optimal (BBBV bound)
- [ ] I can analyze the multiple solutions case

### Technical Skills
- [ ] I can construct oracle circuits for specific problems
- [ ] I can calculate optimal iteration counts
- [ ] I can trace through Grover iterations step by step
- [ ] I can implement Grover's algorithm in code

### Problem Solving
- [ ] I can apply Grover to database search problems
- [ ] I can analyze speedups for various M/N ratios
- [ ] I can handle unknown M using appropriate strategies
- [ ] I can connect Grover to cryptographic applications

---

## Preview: Week 90

Next week we generalize Grover's algorithm to **Amplitude Amplification**, covering:
- Arbitrary initial state preparation
- Amplitude estimation via phase estimation
- Fixed-point amplification (avoiding overshooting)
- Quantum counting

---

*End of Week 89 — Grover's Search Algorithm*
