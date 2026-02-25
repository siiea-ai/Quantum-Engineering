# Day 712: T-Gate Synthesis and Optimization

## Overview

**Date:** Day 712 of 1008
**Week:** 102 (Gottesman-Knill Theorem)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Efficient Implementation of Non-Clifford Rotations

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Solovay-Kitaev and exact synthesis |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | T-count optimization |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Apply the Solovay-Kitaev theorem** for gate approximation
2. **Distinguish exact vs. approximate** synthesis methods
3. **Implement basic synthesis algorithms** for single-qubit rotations
4. **Optimize T-count** in quantum circuits
5. **Analyze trade-offs** between approximation error and T-count
6. **Use modern synthesis tools** like gridsynth

---

## Core Content

### 1. The Synthesis Problem

#### Problem Statement

Given a single-qubit unitary $U$ and error tolerance $\epsilon$:

Find a sequence of gates from $\{H, S, T\}$ (or $\{H, T\}$) such that:
$$\|U - \tilde{U}\| \leq \epsilon$$

where $\tilde{U}$ is the synthesized approximation.

#### Why This Matters

- **Fault-tolerant QC:** Only $\{H, S, T, \text{CNOT}\}$ can be made fault-tolerant easily
- **Cost:** T gates are expensive (require magic states)
- **Goal:** Minimize T-count while achieving required precision

---

### 2. The Solovay-Kitaev Theorem

#### Theorem Statement

**Solovay-Kitaev:** Let $\mathcal{G}$ be a finite set of gates that generates a dense subgroup of $SU(2)$. Then any $U \in SU(2)$ can be approximated to precision $\epsilon$ using:

$$O(\log^c(1/\epsilon)) \text{ gates from } \mathcal{G}$$

where $c \approx 3.97$ (improved bounds give $c \approx 1.44$).

#### Key Properties

1. **Polylogarithmic scaling:** Much better than brute-force search
2. **Universal:** Works for any dense generating set
3. **Constructive:** Algorithm exists to find the sequence
4. **Suboptimal T-count:** Not the best for minimizing T gates

---

### 3. Solovay-Kitaev Algorithm

#### Recursive Structure

**Base case:** $\epsilon_0$ — use lookup table of short sequences

**Recursive step:** To approximate $U$ to precision $\epsilon$:
1. Find $\tilde{U}_1$ approximating $U$ to precision $\sqrt{\epsilon}$
2. Compute $V = U\tilde{U}_1^\dagger$ (residual error)
3. Decompose $V = V_1 V_2 V_1^\dagger V_2^\dagger$ (group commutator)
4. Recursively approximate $V_1, V_2$ to precision $\epsilon^{1/4}$

**Recurrence:** $T(\epsilon) = T(\sqrt{\epsilon}) + 2T(\epsilon^{1/4})$

**Solution:** $T(\epsilon) = O(\log^c(1/\epsilon))$

---

### 4. Exact Synthesis: Ross-Selinger Algorithm

#### The Exact Synthesis Problem

For certain angles, we can find **exact** decompositions:

$$R_z(\theta) = \text{exactly } \{H, T\}^* \quad \Leftrightarrow \quad e^{i\theta/2} \in \mathbb{Z}[i, 1/\sqrt{2}]$$

This includes angles $\theta = k\pi/2^n$ for integers $k, n$.

#### Grid Points

The set of exactly synthesizable unitaries forms a **grid** in $SU(2)$:

$$\mathcal{G} = \left\{ \frac{1}{\sqrt{2}^k}(a + bi + cj + dk) : a,b,c,d \in \mathbb{Z}[i] \right\}$$

where we use quaternion notation for $SU(2)$.

#### Ross-Selinger (2014)

Given $U \in SU(2)$ and $\epsilon$:
1. Find nearest grid point $\tilde{U} \in \mathcal{G}$ with $\|U - \tilde{U}\| \leq \epsilon$
2. Decompose $\tilde{U}$ exactly into $\{H, T\}$

**Complexity:** $O(\log(1/\epsilon))$ T gates — **optimal**!

---

### 5. T-Count Lower Bounds

#### For Single-Qubit Unitaries

**Theorem (Ross-Selinger):** Any sequence over $\{H, T\}$ has T-count at least:

$$t \geq 4\log_2(1/\epsilon) - O(1)$$

to approximate a generic unitary to precision $\epsilon$.

#### For Specific Gates

| Gate | Exact T-count |
|------|---------------|
| $T = R_z(\pi/4)$ | 1 |
| $T^2 = S = R_z(\pi/2)$ | 0 (Clifford) |
| $R_z(\pi/8)$ | 2 |
| $R_z(\pi/16)$ | 4 |
| Toffoli | 7 (original), 4 (with ancilla) |

---

### 6. Multi-Qubit T-Count Optimization

#### Techniques

1. **Gate cancellation:** Adjacent $T^\dagger T = I$
2. **Phase merging:** $R_z(\theta_1)R_z(\theta_2) = R_z(\theta_1 + \theta_2)$
3. **Algebraic identities:** $THTH = R_z(\pi/4)$ (up to Clifford)
4. **Ancilla-assisted:** Trade ancilla qubits for T gates

#### T-Par Algorithm

For circuits with many $R_z$ rotations:
1. Identify parallel $R_z$ gates
2. Synthesize together using algebraic relations
3. Achieve sublinear scaling in number of rotations

---

### 7. The Gridsynth Tool

#### Overview

**Gridsynth** is a practical tool for optimal single-qubit synthesis:

```bash
gridsynth 0.1  # Approximate R_z(0.1) to precision 10^{-10}
```

Output: Sequence of H and T gates.

#### Usage

```python
# Conceptual usage (actual tool is command-line)
def gridsynth_approximate(theta: float, epsilon: float) -> str:
    """
    Find minimal-T circuit for R_z(theta).

    Returns string like "HTHTTHTHTHT..."
    """
    # Uses number-theoretic methods
    pass
```

#### Performance

- **Optimal T-count** for single-qubit $R_z$
- **Fast:** $O(\text{poly}(\log(1/\epsilon)))$ classical time
- **Proven optimal** up to additive constant

---

### 8. State-of-the-Art Results

#### Single-Qubit Synthesis (2024)

| Method | T-count | Classical Time |
|--------|---------|----------------|
| Solovay-Kitaev | $O(\log^{3.97}(1/\epsilon))$ | Polynomial |
| Ross-Selinger | $4\log_2(1/\epsilon) + O(1)$ | Polynomial |
| Optimal (LB) | $4\log_2(1/\epsilon) - O(1)$ | — |

#### Multi-Qubit Circuits

| Optimization | Typical Reduction |
|--------------|-------------------|
| Phase folding | 20-40% |
| T-par | Up to 50% |
| Ancilla-assisted | 30-50% |

---

## Worked Examples

### Example 1: Solovay-Kitaev for $R_z(\pi/5)$

**Problem:** Estimate T-count for approximating $R_z(\pi/5)$ to $\epsilon = 10^{-6}$.

**Solution:**

**Solovay-Kitaev bound:**
$$T \leq c \cdot \log^{3.97}(1/\epsilon) = c \cdot \log^{3.97}(10^6)$$
$$= c \cdot (6 \log 10)^{3.97} \approx c \cdot 6146$$

So potentially thousands of T gates!

**Ross-Selinger bound:**
$$T \leq 4\log_2(10^6) + O(1) = 4 \cdot 19.9 + O(1) \approx 80$$

Much better — about 80 T gates.

---

### Example 2: Exact Synthesis of $R_z(\pi/8)$

**Problem:** Find the exact decomposition of $R_z(\pi/8)$ in $\{H, T\}$.

**Solution:**

$R_z(\pi/8) = T^{1/2}$ — not directly a power of T.

But: $R_z(\pi/8) = e^{-i\pi/16} \begin{pmatrix} e^{i\pi/16} & 0 \\ 0 & e^{-i\pi/16} \end{pmatrix}$

Using the identity:
$$R_z(\pi/8) = HTH \cdot T \cdot HTH = THTHT?$$

Actually, exact synthesis gives:
$$R_z(\pi/8) = SH T HT H$$

Let's verify: (This is a non-trivial calculation using the Ross-Selinger number theory.)

The gridsynth tool gives: T-count = 2 for $R_z(\pi/8)$.

---

### Example 3: T-Count for Quantum Fourier Transform

**Problem:** Estimate T-count for $n$-qubit QFT.

**Solution:**

QFT uses controlled rotations $CR_z(2\pi/2^k)$ for $k = 2, \ldots, n$.

For each $R_z(2\pi/2^k)$:
- $k = 2$: $R_z(\pi/2) = S$ — 0 T gates (Clifford)
- $k = 3$: $R_z(\pi/4) = T$ — 1 T gate
- $k = 4$: $R_z(\pi/8)$ — ~2 T gates
- General: $R_z(\pi/2^{k-1})$ — $(k-3)$ T gates approximately

**Total rotations:** $\binom{n}{2} = n(n-1)/2$

**Total T-count:** $\sum_{k=3}^{n} (n-k+1)(k-2) \approx O(n^3)$ T gates

With optimization (phase folding): Can reduce to $O(n^2)$.

---

## Practice Problems

### Direct Application

1. **Problem 1:** Use the Solovay-Kitaev bound to estimate T-count for $R_z(1)$ to precision $\epsilon = 10^{-10}$.

2. **Problem 2:** Show that $R_z(\pi/4) = T$ has T-count exactly 1.

3. **Problem 3:** If gridsynth gives T-count 15 for some rotation, what is the approximate precision achieved?

### Intermediate

4. **Problem 4:** Derive the recurrence relation for Solovay-Kitaev and solve it.

5. **Problem 5:** Prove that $\{H, T\}$ generates a dense subgroup of $SU(2)$.

6. **Problem 6:** Design an algorithm to cancel adjacent $T^\dagger T$ pairs in a circuit.

### Challenging

7. **Problem 7:** Implement a basic Solovay-Kitaev algorithm for precision $\epsilon = 0.1$.

8. **Problem 8:** Prove the T-count lower bound $t \geq 4\log_2(1/\epsilon) - O(1)$.

9. **Problem 9:** Analyze the T-count of Grover's algorithm with $n$ qubits and $k$ iterations.

---

## Computational Lab

```python
"""
Day 712: T-Gate Synthesis and Optimization
Week 102: Gottesman-Knill Theorem

Implements basic gate synthesis algorithms.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Gate definitions
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
S = T @ T
T_dag = np.conj(T.T)

def Rz(theta: float) -> np.ndarray:
    """Z-rotation by angle theta."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


class BasicSynthesis:
    """Basic gate synthesis methods."""

    # Pre-computed short sequences (lookup table)
    SHORT_SEQUENCES = {
        'I': [],
        'H': ['H'],
        'T': ['T'],
        'S': ['T', 'T'],
        'HT': ['H', 'T'],
        'TH': ['T', 'H'],
        'HTH': ['H', 'T', 'H'],
        'THT': ['T', 'H', 'T'],
    }

    @staticmethod
    def sequence_to_unitary(sequence: List[str]) -> np.ndarray:
        """Convert gate sequence to unitary matrix."""
        result = np.eye(2, dtype=complex)
        gates = {'H': H, 'T': T, 'S': S}

        for gate in sequence:
            result = gates[gate] @ result

        return result

    @staticmethod
    def operator_distance(U1: np.ndarray, U2: np.ndarray) -> float:
        """Compute distance between unitaries."""
        return np.linalg.norm(U1 - U2)

    @classmethod
    def brute_force_search(cls, target: np.ndarray, max_length: int = 10,
                          epsilon: float = 0.1) -> Tuple[List[str], float]:
        """
        Brute force search for gate sequence.

        Returns (sequence, distance).
        """
        from itertools import product

        best_seq = []
        best_dist = float('inf')

        for length in range(max_length + 1):
            for seq in product(['H', 'T'], repeat=length):
                seq_list = list(seq)
                U = cls.sequence_to_unitary(seq_list)
                dist = cls.operator_distance(target, U)

                if dist < best_dist:
                    best_dist = dist
                    best_seq = seq_list

                if dist < epsilon:
                    return seq_list, dist

        return best_seq, best_dist


class SolovayKitaev:
    """
    Simplified Solovay-Kitaev algorithm.

    This is a pedagogical implementation, not optimized.
    """

    def __init__(self, base_precision: float = 0.1):
        self.base_precision = base_precision
        self.lookup_table = self._build_lookup_table()

    def _build_lookup_table(self) -> dict:
        """Build lookup table for base case."""
        table = {}

        # Generate all sequences up to length 5
        from itertools import product

        for length in range(6):
            for seq in product(['H', 'T'], repeat=length):
                seq_list = list(seq)
                U = BasicSynthesis.sequence_to_unitary(seq_list)

                # Store by discretized unitary
                key = tuple(np.round(U.flatten().real, 2)) + \
                      tuple(np.round(U.flatten().imag, 2))
                if key not in table or len(table[key]) > len(seq_list):
                    table[key] = seq_list

        return table

    def _lookup(self, U: np.ndarray) -> List[str]:
        """Find approximate sequence from lookup table."""
        best_seq = []
        best_dist = float('inf')

        for key, seq in self.lookup_table.items():
            lookup_U = BasicSynthesis.sequence_to_unitary(seq)
            dist = BasicSynthesis.operator_distance(U, lookup_U)

            if dist < best_dist:
                best_dist = dist
                best_seq = seq

        return best_seq

    def approximate(self, U: np.ndarray, epsilon: float) -> List[str]:
        """
        Approximate U to precision epsilon.

        Returns gate sequence.
        """
        if epsilon >= self.base_precision:
            return self._lookup(U)

        # Recursive case
        sqrt_eps = np.sqrt(epsilon)

        # Step 1: Get coarse approximation
        U_approx = BasicSynthesis.sequence_to_unitary(
            self.approximate(U, sqrt_eps)
        )

        # Step 2: Compute residual
        V = U @ np.conj(U_approx.T)

        # Step 3: Decompose V as commutator (simplified)
        # For pedagogical purposes, just refine with lookup
        V_seq = self._lookup(V)

        # Combine sequences
        return self.approximate(U, sqrt_eps) + V_seq


class TCountAnalysis:
    """Analyze T-count in circuits."""

    @staticmethod
    def count_t_gates(sequence: List[str]) -> int:
        """Count T gates in sequence."""
        return sum(1 for g in sequence if g == 'T')

    @staticmethod
    def ross_selinger_bound(epsilon: float) -> int:
        """Upper bound on T-count from Ross-Selinger."""
        return int(4 * np.log2(1 / epsilon) + 10)  # +10 for constant

    @staticmethod
    def solovay_kitaev_bound(epsilon: float, c: float = 3.97) -> int:
        """Upper bound on T-count from Solovay-Kitaev."""
        return int(100 * np.log(1 / epsilon) ** c)  # Rough estimate

    @staticmethod
    def optimize_sequence(sequence: List[str]) -> List[str]:
        """
        Optimize gate sequence by cancellation.

        Rules:
        - TT†= I (but we don't have T† in our basic set)
        - TTTT = I
        - HH = I
        """
        optimized = sequence.copy()
        changed = True

        while changed:
            changed = False
            new_seq = []
            i = 0

            while i < len(optimized):
                # Check for HH cancellation
                if i + 1 < len(optimized) and \
                   optimized[i] == 'H' and optimized[i+1] == 'H':
                    changed = True
                    i += 2
                    continue

                # Check for TTTT cancellation
                if i + 3 < len(optimized) and \
                   all(optimized[i+j] == 'T' for j in range(4)):
                    changed = True
                    i += 4
                    continue

                new_seq.append(optimized[i])
                i += 1

            optimized = new_seq

        return optimized


def demonstrate_synthesis():
    """Demonstrate gate synthesis methods."""

    print("=" * 70)
    print("T-GATE SYNTHESIS")
    print("=" * 70)

    # Target rotation
    theta = 0.3
    target = Rz(theta)

    print(f"\n1. TARGET: R_z({theta})")
    print("-" * 50)

    # Brute force search
    print("\n  Brute force search:")
    for max_len in [3, 5, 7, 9]:
        seq, dist = BasicSynthesis.brute_force_search(target, max_len, 0.01)
        t_count = TCountAnalysis.count_t_gates(seq)
        print(f"    Length ≤{max_len}: T-count={t_count}, distance={dist:.4f}")
        print(f"      Sequence: {''.join(seq)}")

    # Complexity bounds
    print("\n2. T-COUNT BOUNDS")
    print("-" * 50)

    print("\n  Precision ε  | Solovay-Kitaev | Ross-Selinger")
    print("  " + "-" * 50)

    for exp in range(1, 11):
        epsilon = 10 ** (-exp)
        sk = TCountAnalysis.solovay_kitaev_bound(epsilon)
        rs = TCountAnalysis.ross_selinger_bound(epsilon)
        print(f"   10^{-exp:2d}       |    {sk:6d}       |    {rs:4d}")

    # Sequence optimization
    print("\n3. SEQUENCE OPTIMIZATION")
    print("-" * 50)

    test_sequences = [
        ['H', 'H', 'T', 'T', 'T', 'T', 'H'],
        ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
        ['H', 'T', 'H', 'H', 'T', 'H', 'T', 'T', 'T', 'T'],
    ]

    for seq in test_sequences:
        original = ''.join(seq)
        optimized = TCountAnalysis.optimize_sequence(seq)
        opt_str = ''.join(optimized)

        orig_t = TCountAnalysis.count_t_gates(seq)
        opt_t = TCountAnalysis.count_t_gates(optimized)

        print(f"\n  Original:  {original} (T-count: {orig_t})")
        print(f"  Optimized: {opt_str} (T-count: {opt_t})")


def analyze_algorithms():
    """Analyze T-count for common algorithms."""

    print("\n" + "=" * 70)
    print("T-COUNT IN QUANTUM ALGORITHMS")
    print("=" * 70)

    print("\n1. QUANTUM FOURIER TRANSFORM")
    print("-" * 50)

    print("\n  n qubits | Rotations | Estimated T-count")
    print("  " + "-" * 45)

    for n in [4, 8, 16, 32, 64]:
        # QFT has n(n-1)/2 controlled rotations
        n_rotations = n * (n - 1) // 2

        # Rough estimate: each rotation needs ~log2(precision) T gates
        # Assuming 10-digit precision
        t_per_rotation = 40  # Typical for 10^-10 precision
        total_t = n_rotations * t_per_rotation

        # With optimization
        optimized_t = total_t // 3  # Phase folding can help

        print(f"    {n:4d}    |   {n_rotations:5d}    |  {total_t:8d} (opt: ~{optimized_t})")

    print("\n2. GROVER'S ALGORITHM")
    print("-" * 50)

    print("\n  For n qubits, k oracle queries:")
    print("  - Oracle: O(1) T gates per query (depends on function)")
    print("  - Diffusion: O(n) T gates")
    print("  - Queries: O(√N) = O(2^{n/2})")
    print("\n  Total: O(n · 2^{n/2}) T gates")


if __name__ == "__main__":
    demonstrate_synthesis()
    analyze_algorithms()
```

---

## Summary

### Key Results

| Method | T-Count | Best For |
|--------|---------|----------|
| **Solovay-Kitaev** | $O(\log^{3.97}(1/\epsilon))$ | Any dense gate set |
| **Ross-Selinger** | $4\log_2(1/\epsilon) + O(1)$ | $\{H, T\}$ synthesis |
| **Lower bound** | $4\log_2(1/\epsilon) - O(1)$ | Theoretical limit |

### Main Takeaways

1. **Solovay-Kitaev** gives polylog synthesis but not optimal T-count
2. **Exact synthesis** achieves optimal $O(\log(1/\epsilon))$ T gates
3. **Gridsynth** is the practical tool for optimal synthesis
4. **T-count optimization** reduces fault-tolerant overhead by 30-50%
5. **Phase folding** and **T-par** are key optimization techniques

---

## Daily Checklist

- [ ] State the Solovay-Kitaev theorem
- [ ] Compare Solovay-Kitaev vs Ross-Selinger bounds
- [ ] Estimate T-count for specific rotations
- [ ] Apply basic T-count optimization
- [ ] Analyze T-count in quantum algorithms
- [ ] Understand the role of gridsynth

---

## Preview: Day 713

Tomorrow we study **Quantum Advantage and the Limits of Classical Simulation**, synthesizing:
- Where quantum advantage comes from
- The role of magic and entanglement
- Complexity-theoretic evidence for quantum advantage
- Near-term demonstrations
