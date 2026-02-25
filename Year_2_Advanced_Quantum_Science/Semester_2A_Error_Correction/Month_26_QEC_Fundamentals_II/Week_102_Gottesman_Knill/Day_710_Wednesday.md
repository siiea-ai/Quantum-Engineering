# Day 710: Boundaries of Classical Simulation

## Overview

**Date:** Day 710 of 1008
**Week:** 102 (Gottesman-Knill Theorem)
**Month:** 26 (QEC Fundamentals II)
**Topic:** The Frontier Between Classical and Quantum Computation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Stabilizer rank and non-stabilizerness |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Simulation complexity hierarchies |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Computational exploration |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define and compute** the stabilizer rank of quantum states
2. **Quantify "non-stabilizerness"** using magic measures
3. **Explain the complexity hierarchy** from Clifford to universal
4. **Analyze circuits** with bounded non-Clifford gates
5. **Connect** simulation complexity to quantum advantage
6. **Identify** the precise boundary of efficient classical simulation

---

## Core Content

### 1. The Classical-Quantum Boundary

#### Where Gottesman-Knill Ends

The theorem shows Clifford circuits are classically simulable. But:

$$\text{Clifford} + \text{one T gate} = \text{???}$$

**Key Question:** How many T gates before classical simulation fails?

#### The Surprising Answer

**Theorem (Bravyi-Gosset 2016):** A circuit with $n$ qubits and $t$ T gates can be classically simulated in time:

$$O(\text{poly}(n) \cdot 2^{ct})$$

where $c \approx 0.396$.

**Implication:** For $t = O(\log n)$ T gates, simulation is still polynomial!

---

### 2. Stabilizer Rank

#### Definition

The **stabilizer rank** of a state $|\psi\rangle$, denoted $\chi(|\psi\rangle)$, is the minimum number of stabilizer states needed to express $|\psi\rangle$:

$$|\psi\rangle = \sum_{j=1}^{\chi} c_j |\phi_j\rangle$$

where each $|\phi_j\rangle$ is a stabilizer state.

#### Properties

1. **Stabilizer states:** $\chi(|\psi\rangle) = 1$
2. **Submultiplicativity:** $\chi(|\psi\rangle \otimes |\phi\rangle) \leq \chi(|\psi\rangle) \cdot \chi(|\phi\rangle)$
3. **Clifford invariance:** $\chi(C|\psi\rangle) = \chi(|\psi\rangle)$ for Clifford $C$

#### Key Examples

| State | Stabilizer Rank $\chi$ |
|-------|----------------------|
| $\|0\rangle$ | 1 |
| $\|+\rangle$ | 1 |
| Bell state | 1 |
| $\|T\rangle = T\|+\rangle$ | 2 |
| $\|T\rangle^{\otimes t}$ | $2^{t/2}$ (approx) |

---

### 3. Magic States and Non-Stabilizerness

#### The T-State (Magic State)

$$|T\rangle = T|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

**Stabilizer decomposition:**

$$|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle) = c_1|+\rangle + c_2|-\rangle + \ldots$$

Actually:
$$|T\rangle = \frac{1+e^{i\pi/4}}{2}|+\rangle + \frac{1-e^{i\pi/4}}{2}|-\rangle$$

Let $\omega = e^{i\pi/4}$:
$$|T\rangle = \frac{1+\omega}{2}|+\rangle + \frac{1-\omega}{2}|-\rangle$$

This requires 2 stabilizer states, so $\chi(|T\rangle) = 2$.

#### Magic Monotones

**Robustness of Magic:**
$$\mathcal{R}(|\psi\rangle) = \min \left\{ \sum_j |c_j| : |\psi\rangle = \sum_j c_j |\phi_j\rangle, \; |\phi_j\rangle \text{ stabilizer} \right\}$$

For stabilizer states: $\mathcal{R} = 1$
For $|T\rangle$: $\mathcal{R} = 1 + \sqrt{2}/2 \approx 1.71$

---

### 4. Simulation with Stabilizer Decomposition

#### The Algorithm

Given state $|\psi\rangle = \sum_{j=1}^{\chi} c_j |\phi_j\rangle$:

1. For each stabilizer term $|\phi_j\rangle$:
   - Simulate Clifford circuit on $|\phi_j\rangle$ efficiently
   - Weight outcomes by $|c_j|^2$

2. Combine results:
   - Total amplitude = $\sum_j c_j \langle x | U | \phi_j \rangle$

**Complexity:** $O(\chi \cdot \text{poly}(n))$ per amplitude.

#### Growth of Stabilizer Rank

For $t$ T gates applied to a stabilizer state:

$$\chi \leq 2^{t/2} \cdot \text{poly}(n, t)$$

This gives simulation complexity $O(2^{t/2} \cdot \text{poly}(n))$.

---

### 5. The Complexity Hierarchy

#### Levels of Simulation Hardness

| Circuit Class | Classical Complexity | Notes |
|--------------|---------------------|-------|
| Clifford only | $O(\text{poly}(n))$ | Gottesman-Knill |
| Clifford + $O(\log n)$ T | $O(\text{poly}(n))$ | Still efficient |
| Clifford + $O(n)$ T | $O(2^{n/2})$ | Borderline |
| Clifford + $O(n^2)$ T | $O(2^n)$ | Exponential |
| Universal | $O(2^n)$ | State vector simulation |

#### The Phase Transition

**Conjecture:** There exists a threshold $t^*$ such that:
- $t < t^*$: Classical simulation in $O(\text{poly}(n))$
- $t > t^*$: Classical simulation requires $\Omega(2^{n^\epsilon})$

Current evidence suggests $t^* = \Theta(n)$.

---

### 6. Other Non-Clifford Resources

#### Toffoli Gate

The Toffoli (CCNOT) gate is non-Clifford:

$$\text{CCNOT}|a, b, c\rangle = |a, b, c \oplus ab\rangle$$

**Stabilizer rank of Toffoli output:**

Starting from $|+++\rangle$ (stabilizer) and applying Toffoli:
$$\text{CCNOT}|+++\rangle = ?$$

This state has $\chi > 1$, making it non-stabilizer.

#### Relation to T Gates

Toffoli can be decomposed into Clifford + T:
- Requires 7 T gates (originally)
- Optimal: 4 T gates (with measurement)

---

### 7. Approximate Simulation

#### Relaxing Exactness

**Approximate simulation:** Compute probabilities to within additive error $\epsilon$.

**Theorem:** A circuit with $t$ T gates can be approximately simulated in time:

$$O\left(\frac{2^{ct}}{\epsilon^2}\right)$$

where $c < 1$.

#### Sampling vs. Computing Probabilities

**Strong simulation:** Compute $\Pr(\text{outcome})$ exactly
**Weak simulation:** Sample from output distribution

Sampling can be easier than computing probabilities for some circuits.

---

### 8. Quantum Advantage from Non-Stabilizerness

#### The IQP Circuits

**Instantaneous Quantum Polynomial** circuits:
- All gates diagonal in X basis
- Includes non-Clifford rotations

**Hardness result:** Sampling from IQP output is hard for classical computers (under complexity assumptions).

#### Random Circuit Sampling

**Google's quantum supremacy experiment (2019):**
- Random circuits with Clifford + T-like gates
- Demonstrated sampling that classical computers cannot match

#### The Magic Threshold

Quantum advantage requires sufficient "magic":
- Too little magic → classically simulable
- Sufficient magic → potential quantum advantage
- The boundary depends on circuit structure

---

## Worked Examples

### Example 1: Compute Stabilizer Rank

**Problem:** Find the stabilizer rank of $|T\rangle = T|+\rangle$.

**Solution:**

$$|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

Express in $|+\rangle, |-\rangle$ basis:
- $|0\rangle = \frac{1}{\sqrt{2}}(|+\rangle + |-\rangle)$
- $|1\rangle = \frac{1}{\sqrt{2}}(|+\rangle - |-\rangle)$

$$|T\rangle = \frac{1}{2}[(1+e^{i\pi/4})|+\rangle + (1-e^{i\pi/4})|-\rangle]$$

Let $\alpha = \frac{1+e^{i\pi/4}}{2}$ and $\beta = \frac{1-e^{i\pi/4}}{2}$.

$$|T\rangle = \alpha|+\rangle + \beta|-\rangle$$

Both $|+\rangle$ and $|-\rangle$ are stabilizer states, and $\alpha, \beta \neq 0$.

**Can we do with 1 stabilizer state?**

If $|T\rangle = c|\phi\rangle$ for stabilizer $|\phi\rangle$, then $|\phi\rangle$ must be an eigenstate of some Pauli.

But $|T\rangle$ is not an eigenstate of any Pauli (computed in Day 708).

Therefore, $\chi(|T\rangle) = 2$.

---

### Example 2: Simulate Circuit with One T Gate

**Problem:** Classically simulate $H \cdot T \cdot H$ on $|0\rangle$.

**Solution:**

**Step 1:** Track through circuit:
- $|0\rangle$: stabilizer state (stabilizer = $Z$)
- $H|0\rangle = |+\rangle$: stabilizer state (stabilizer = $X$)
- $T|+\rangle = |T\rangle$: non-stabilizer, $\chi = 2$
- $H|T\rangle$: still $\chi = 2$ (Clifford preserves rank)

**Step 2:** Decompose $|T\rangle$:
$$|T\rangle = \alpha|+\rangle + \beta|-\rangle$$

**Step 3:** Apply final $H$:
$$H|T\rangle = \alpha H|+\rangle + \beta H|-\rangle = \alpha|0\rangle + \beta|1\rangle$$

**Step 4:** Compute measurement probability:
$$\Pr(0) = |\alpha|^2 = \left|\frac{1+e^{i\pi/4}}{2}\right|^2$$

$$= \frac{1}{4}|1+e^{i\pi/4}|^2 = \frac{1}{4}(1 + \cos(\pi/4))^2 + \frac{1}{4}\sin^2(\pi/4)$$

$$= \frac{1}{4}(2 + 2\cos(\pi/4)) = \frac{1}{2}(1 + \frac{\sqrt{2}}{2}) \approx 0.854$$

**Complexity:** $O(\chi) = O(2)$ — still efficient!

---

### Example 3: Exponential Growth with T Gates

**Problem:** Estimate the stabilizer rank for $n$ T gates applied to $|+\rangle^{\otimes n}$.

**Solution:**

Apply $T^{\otimes n}$ to $|+\rangle^{\otimes n}$:

$$T^{\otimes n}|+\rangle^{\otimes n} = |T\rangle^{\otimes n}$$

Using submultiplicativity:
$$\chi(|T\rangle^{\otimes n}) \leq \chi(|T\rangle)^n = 2^n$$

**Better bound (Bravyi et al.):**
$$\chi(|T\rangle^{\otimes n}) \approx 2^{n/2}$$

This exponential growth is why circuits with many T gates become hard to simulate.

---

## Practice Problems

### Direct Application

1. **Problem 1:** Prove that $\chi(|\psi\rangle) = 1$ if and only if $|\psi\rangle$ is a stabilizer state.

2. **Problem 2:** Compute the stabilizer decomposition of $|T\rangle \otimes |T\rangle$.

3. **Problem 3:** Show that the stabilizer rank is invariant under Clifford unitaries.

### Intermediate

4. **Problem 4:** A circuit has $n=100$ qubits and $t=10$ T gates. Estimate the classical simulation time using stabilizer rank methods.

5. **Problem 5:** Prove that $\chi(|T\rangle^{\otimes 2}) \leq 4$ and find the explicit decomposition.

6. **Problem 6:** Explain why Toffoli is non-Clifford by showing its effect on a stabilizer state.

### Challenging

7. **Problem 7:** Prove that any single-qubit unitary $U$ that is not Clifford produces a state $U|+\rangle$ with $\chi > 1$.

8. **Problem 8:** Derive the complexity bound $O(2^{0.396t})$ for simulating $t$ T gates (sketch the main ideas).

9. **Problem 9:** Design a family of circuits that have $t$ T gates but are still classically simulable in poly-time (hint: use structure).

---

## Computational Lab

```python
"""
Day 710: Boundaries of Classical Simulation
Week 102: Gottesman-Knill Theorem

Explores stabilizer rank and the boundary of classical simulation.
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import product
import matplotlib.pyplot as plt

class StabilizerRankAnalysis:
    """Analyze stabilizer rank and simulation complexity."""

    # Single-qubit stabilizer states
    STABILIZER_STATES = {
        '|0⟩': np.array([1, 0], dtype=complex),
        '|1⟩': np.array([0, 1], dtype=complex),
        '|+⟩': np.array([1, 1], dtype=complex) / np.sqrt(2),
        '|-⟩': np.array([1, -1], dtype=complex) / np.sqrt(2),
        '|+i⟩': np.array([1, 1j], dtype=complex) / np.sqrt(2),
        '|-i⟩': np.array([1, -1j], dtype=complex) / np.sqrt(2),
    }

    @classmethod
    def is_stabilizer_state(cls, state: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a single-qubit state is a stabilizer state."""
        state = state / np.linalg.norm(state)

        for name, stab in cls.STABILIZER_STATES.items():
            # Check if parallel (up to global phase)
            overlap = np.abs(np.vdot(stab, state))
            if np.abs(overlap - 1.0) < tol:
                return True
        return False

    @classmethod
    def stabilizer_decomposition(cls, state: np.ndarray) -> Tuple[List[complex], List[str]]:
        """
        Find stabilizer decomposition of single-qubit state.

        Returns coefficients and state names.
        """
        state = state / np.linalg.norm(state)

        # Use |+⟩ and |-⟩ basis
        plus = cls.STABILIZER_STATES['|+⟩']
        minus = cls.STABILIZER_STATES['|-⟩']

        c_plus = np.vdot(plus, state)
        c_minus = np.vdot(minus, state)

        # Verify reconstruction
        reconstructed = c_plus * plus + c_minus * minus
        assert np.allclose(reconstructed, state), "Decomposition failed"

        coeffs = []
        names = []

        if np.abs(c_plus) > 1e-10:
            coeffs.append(c_plus)
            names.append('|+⟩')
        if np.abs(c_minus) > 1e-10:
            coeffs.append(c_minus)
            names.append('|-⟩')

        return coeffs, names

    @classmethod
    def compute_stabilizer_rank(cls, state: np.ndarray) -> int:
        """Compute stabilizer rank of single-qubit state."""
        if cls.is_stabilizer_state(state):
            return 1

        coeffs, _ = cls.stabilizer_decomposition(state)
        return len([c for c in coeffs if np.abs(c) > 1e-10])


def analyze_t_state():
    """Analyze the T state in detail."""

    print("=" * 70)
    print("ANALYSIS OF THE T STATE")
    print("=" * 70)

    # Create T state
    T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    T_state = T_gate @ plus

    print("\n1. T STATE DEFINITION")
    print("-" * 50)
    print(f"  |T⟩ = T|+⟩ = {T_state[0]:.4f}|0⟩ + {T_state[1]:.4f}|1⟩")

    # Check if stabilizer
    analyzer = StabilizerRankAnalysis()
    is_stab = analyzer.is_stabilizer_state(T_state)
    print(f"\n  Is stabilizer state? {is_stab}")

    # Decomposition
    print("\n2. STABILIZER DECOMPOSITION")
    print("-" * 50)

    coeffs, names = analyzer.stabilizer_decomposition(T_state)
    print(f"  |T⟩ = ", end="")
    terms = [f"({c:.4f}){name}" for c, name in zip(coeffs, names)]
    print(" + ".join(terms))

    rank = analyzer.compute_stabilizer_rank(T_state)
    print(f"\n  Stabilizer rank χ(|T⟩) = {rank}")

    # Robustness of magic
    print("\n3. ROBUSTNESS OF MAGIC")
    print("-" * 50)

    robustness = sum(np.abs(c) for c in coeffs)
    print(f"  R(|T⟩) = Σ|cⱼ| = {robustness:.4f}")
    print(f"  For stabilizer states, R = 1")
    print(f"  |T⟩ has 'magic' quantified by R - 1 = {robustness - 1:.4f}")


def analyze_multi_t_states():
    """Analyze tensor products of T states."""

    print("\n" + "=" * 70)
    print("STABILIZER RANK GROWTH")
    print("=" * 70)

    T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    T_state = T_gate @ plus

    print("\n1. TENSOR PRODUCTS |T⟩^⊗n")
    print("-" * 50)

    # Theoretical bounds
    print("\n  n  | χ upper bound | χ approx")
    print("  ---|---------------|----------")

    for n in range(1, 8):
        # Upper bound from submultiplicativity
        upper_bound = 2 ** n

        # Better bound from literature: ~2^(n/2)
        approx = int(np.ceil(2 ** (n * 0.5)))

        print(f"  {n}  |     {upper_bound:4d}      |   {approx:4d}")

    print("\n2. SIMULATION COMPLEXITY")
    print("-" * 50)

    print("""
    For circuit with n qubits and t T gates:

    Method                  | Complexity
    ------------------------|-----------------
    State vector            | O(2^n)
    Stabilizer rank         | O(2^{t/2} · poly(n))
    Bravyi-Gosset bound     | O(2^{0.396t} · poly(n))
    """)

    # Plot complexity comparison
    plt.figure(figsize=(10, 6))

    n = 100  # qubits
    t_values = np.arange(0, 200, 5)

    # State vector (assuming we can't actually do 2^100)
    state_vector = np.ones_like(t_values, dtype=float) * 2**30  # cap at 2^30

    # Stabilizer rank method
    stab_rank = 2 ** (t_values / 2)

    # Bravyi-Gosset
    bravyi_gosset = 2 ** (0.396 * t_values)

    plt.semilogy(t_values, stab_rank, 'b-', label='Stabilizer rank: $2^{t/2}$')
    plt.semilogy(t_values, bravyi_gosset, 'g-', label='Bravyi-Gosset: $2^{0.396t}$')
    plt.axhline(y=2**30, color='r', linestyle='--', label='State vector (capped)')
    plt.axhline(y=n**3, color='k', linestyle=':', label=f'Poly({n}) reference')

    plt.xlabel('Number of T gates (t)')
    plt.ylabel('Simulation complexity')
    plt.title('Classical Simulation Complexity vs T-Gate Count (n=100 qubits)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(1, 1e40)

    plt.tight_layout()
    plt.savefig('t_gate_complexity.png', dpi=150)
    plt.close()

    print("\n  [Plot saved to t_gate_complexity.png]")


def demonstrate_boundary():
    """Demonstrate the simulation boundary."""

    print("\n" + "=" * 70)
    print("THE CLASSICAL-QUANTUM BOUNDARY")
    print("=" * 70)

    print("\n1. CIRCUIT CLASSIFICATION")
    print("-" * 50)

    circuits = [
        ("100 qubits, 0 T gates", "Clifford", "poly(n)", "✓ Efficient"),
        ("100 qubits, 10 T gates", "Near-Clifford", "~2^5", "✓ Efficient"),
        ("100 qubits, 50 T gates", "Moderate T", "~2^25", "? Borderline"),
        ("100 qubits, 200 T gates", "Many T", "~2^80", "✗ Hard"),
        ("100 qubits, universal", "Universal", "2^100", "✗ Very Hard"),
    ]

    print("\n  Circuit                    | Type          | Complexity  | Status")
    print("  " + "-" * 75)
    for circuit, typ, complexity, status in circuits:
        print(f"  {circuit:27s} | {typ:13s} | {complexity:11s} | {status}")

    print("\n2. THE PHASE TRANSITION")
    print("-" * 50)

    print("""
    As we add T gates to a Clifford circuit:

         Clifford ──→ ──→ ──→ ──→ ──→ Universal
              ↑                        ↑
         Efficient                Exponential

    The transition is gradual, not sharp:
    - O(log n) T gates: still polynomial
    - O(n) T gates: borderline
    - O(n²) T gates: definitely exponential
    """)

    print("\n3. WHAT MAKES T SPECIAL?")
    print("-" * 50)

    print("""
    T gate rotates the Bloch sphere by π/4 around Z:

    T = diag(1, e^{iπ/4})

    This is NOT a stabilizer operation because:
    - e^{iπ/4} is an 8th root of unity
    - Stabilizer phases are only 1, -1, i, -i (4th roots)
    - T|+⟩ cannot be written as a single stabilizer state

    The "magic" of T gates comes from this irrational phase!
    """)


def simulate_with_stabilizer_rank():
    """Demonstrate simulation using stabilizer decomposition."""

    print("\n" + "=" * 70)
    print("SIMULATION VIA STABILIZER DECOMPOSITION")
    print("=" * 70)

    print("\n1. CIRCUIT: H - T - H")
    print("-" * 50)

    # Build circuit
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    # Apply to |0⟩
    state = np.array([1, 0], dtype=complex)
    state = H @ state  # |+⟩
    state = T @ state  # |T⟩
    state = H @ state  # H|T⟩

    print(f"  Final state: {state[0]:.4f}|0⟩ + {state[1]:.4f}|1⟩")

    # Direct probability calculation
    prob_0 = np.abs(state[0])**2
    prob_1 = np.abs(state[1])**2
    print(f"\n  Direct: P(0) = {prob_0:.4f}, P(1) = {prob_1:.4f}")

    # Via stabilizer decomposition
    print("\n2. VIA STABILIZER DECOMPOSITION")
    print("-" * 50)

    # |T⟩ = α|+⟩ + β|-⟩
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

    T_state = T @ plus
    alpha = np.vdot(plus, T_state)
    beta = np.vdot(minus, T_state)

    print(f"  |T⟩ = ({alpha:.4f})|+⟩ + ({beta:.4f})|−⟩")

    # Apply H to each term
    H_plus = H @ plus   # |0⟩
    H_minus = H @ minus # |1⟩

    print(f"  H|+⟩ = |0⟩")
    print(f"  H|−⟩ = |1⟩")

    # Combine
    final_from_decomp = alpha * H_plus + beta * H_minus
    print(f"\n  H|T⟩ = ({alpha:.4f})|0⟩ + ({beta:.4f})|1⟩")

    # Verify
    print(f"\n  Verification: states match? {np.allclose(state, final_from_decomp)}")

    # Probabilities
    prob_0_decomp = np.abs(alpha)**2
    prob_1_decomp = np.abs(beta)**2
    print(f"  Via decomposition: P(0) = {prob_0_decomp:.4f}, P(1) = {prob_1_decomp:.4f}")


if __name__ == "__main__":
    analyze_t_state()
    analyze_multi_t_states()
    demonstrate_boundary()
    simulate_with_stabilizer_rank()
```

---

## Summary

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Stabilizer rank** | Min stabilizer states to express $\|\psi\rangle$ |
| **Magic** | Quantifies non-stabilizerness |
| **Robustness** | $\mathcal{R} = \sum_j \|c_j\|$ in decomposition |
| **Simulation bound** | $O(2^{0.396t})$ for $t$ T gates |
| **Phase transition** | Around $t = O(n)$ T gates |

### Main Takeaways

1. **Stabilizer rank** quantifies how "non-stabilizer" a state is
2. **T gates add magic** — each T roughly doubles stabilizer rank
3. **Logarithmic T gates** still allow polynomial simulation
4. **Linear T gates** mark the boundary of efficient simulation
5. **Quantum advantage** requires sufficient magic content

---

## Daily Checklist

- [ ] Define and compute stabilizer rank
- [ ] Find stabilizer decomposition of $|T\rangle$
- [ ] Explain simulation complexity hierarchy
- [ ] Identify boundary between classical and quantum
- [ ] Connect magic to quantum advantage
- [ ] Analyze circuits with bounded T gates

---

## Preview: Day 711

Tomorrow we dive into **Magic States and Non-Clifford Resources**, exploring:
- Magic state injection protocol
- T gate implementation via state injection
- Magic state distillation basics
- Resource theory of magic
