# Day 345: State Collapse — The Post-Measurement State

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: State Collapse & Projection Operators |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 345, you will be able to:

1. State the collapse postulate and explain its physical meaning
2. Construct projection operators for non-degenerate and degenerate eigenspaces
3. Calculate post-measurement states for arbitrary observables
4. Predict outcomes of sequential measurements
5. Handle degenerate eigenvalues correctly
6. Connect collapse to the quantum Zeno effect

---

## Core Content

### 1. The Collapse Postulate: What Happens After Measurement?

Yesterday we learned *what* we can measure (eigenvalues) and *how likely* each outcome is (Born rule). Today we ask: *what is the state after measurement?*

**Postulate 5 (State Collapse / Projection Postulate):**

> *If a measurement of observable A on state |ψ⟩ yields result a, then immediately after the measurement, the system is in eigenstate |a⟩.*

$$\boxed{|ψ⟩ \xrightarrow{\text{measure } A, \text{ get } a} |a⟩}$$

This is often called "wave function collapse" or the "reduction of the state vector."

**The mystery:** This transition is:
- **Instantaneous** — occurs at the moment of measurement
- **Discontinuous** — |ψ⟩ jumps to |a⟩, not a smooth evolution
- **Irreversible** — cannot undo without additional operations
- **Non-unitary** — does not preserve superposition

---

### 2. The Projection Operator Formulation

For non-degenerate eigenvalue a with eigenstate |a⟩, define the **projection operator**:

$$\boxed{\hat{P}_a = |a⟩⟨a|}$$

**Properties of projection operators:**

1. **Idempotent:** $\hat{P}_a^2 = \hat{P}_a$
   $$\hat{P}_a^2 = |a⟩⟨a|a⟩⟨a| = |a⟩(1)⟨a| = \hat{P}_a$$

2. **Hermitian:** $\hat{P}_a^\dagger = \hat{P}_a$
   $$\hat{P}_a^\dagger = (|a⟩⟨a|)^\dagger = |a⟩⟨a| = \hat{P}_a$$

3. **Orthogonality:** For distinct eigenvalues a ≠ a':
   $$\hat{P}_a \hat{P}_{a'} = |a⟩⟨a|a'⟩⟨a'| = 0$$

4. **Completeness:** $\sum_a \hat{P}_a = \hat{I}$

**The collapse rule using projectors:**

$$\boxed{|ψ⟩ \xrightarrow{\text{measure } a} |ψ'\rangle = \frac{\hat{P}_a|ψ⟩}{\sqrt{⟨ψ|\hat{P}_a|ψ⟩}}}$$

The denominator normalizes the post-measurement state.

---

### 3. Why Projection?

The projection operator formulation reveals the geometric meaning of collapse:

**Before measurement:** |ψ⟩ is a vector in full Hilbert space

**After measurement:** The state is *projected* onto the eigenspace of the measured eigenvalue

Consider |ψ⟩ = α|a₁⟩ + β|a₂⟩ + γ|a₃⟩:

- Measure A, get a₁ → State becomes |a₁⟩
- The components β|a₂⟩ + γ|a₃⟩ are "collapsed away"
- Information about those components is lost

**Visualization:**

```
Before:        |ψ⟩ = α|a₁⟩ + β|a₂⟩
                    ↗
                  /
                /
              /
            /
          ○ ────────→ |a₂⟩
                |a₁⟩

After measuring a₁:  State = |a₁⟩ (projected onto |a₁⟩ axis)
```

---

### 4. Repeated Measurements

A key consequence of collapse: **repeated measurements give the same result**.

**Theorem:** If we measure observable A twice in succession (with no evolution between), the second measurement gives the same result as the first with probability 1.

**Proof:**

1. First measurement: get a with probability |⟨a|ψ⟩|²
2. State collapses to |a⟩
3. Second measurement: P(a) = |⟨a|a⟩|² = 1 ✓

This is sometimes called the "eigenstate stability postulate" — eigenstates are stable under repeated measurement of the same observable.

**Example: Stern-Gerlach**

1. Measure Sz → get +ℏ/2, state becomes |↑⟩
2. Immediately measure Sz again → always get +ℏ/2

But if we measure a *different* observable:

1. Measure Sz → get +ℏ/2, state becomes |↑⟩
2. Measure Sx → get ±ℏ/2 with 50% each

The Sx measurement destroys the Sz eigenstate.

---

### 5. Degenerate Eigenvalues

Many physical systems have **degenerate** eigenvalues — multiple linearly independent eigenstates with the same eigenvalue.

**Example:** Hydrogen atom energy levels have degeneracy:
- n = 2: four states (2s, 2p₀, 2p₊₁, 2p₋₁) all with same E₂

**Collapse for degenerate eigenvalue a:**

If a has g-fold degeneracy with orthonormal eigenstates {|a,1⟩, |a,2⟩, ..., |a,g⟩}, the projector is:

$$\boxed{\hat{P}_a = \sum_{i=1}^{g} |a,i⟩⟨a,i|}$$

Post-measurement state:

$$|ψ'⟩ = \frac{\hat{P}_a|ψ⟩}{\|\hat{P}_a|ψ⟩\|}$$

**Key point:** The state collapses to the **subspace** spanned by degenerate eigenstates, not to a specific eigenstate within that subspace.

---

### 6. Example: Spin-1 System

A spin-1 particle has three Sz eigenstates: |+1⟩, |0⟩, |-1⟩ with eigenvalues +ℏ, 0, -ℏ.

Consider state:
$$|ψ⟩ = \frac{1}{2}|+1⟩ + \frac{1}{\sqrt{2}}|0⟩ + \frac{1}{2}|-1⟩$$

**Measure Sz:**

| Outcome | Probability | Post-measurement state |
|---------|-------------|----------------------|
| +ℏ | 1/4 | \|+1⟩ |
| 0 | 1/2 | \|0⟩ |
| -ℏ | 1/4 | \|-1⟩ |

Now consider measuring Sz² (which has eigenvalue 0 for |0⟩ and ℏ² for |±1⟩):

**Measure Sz²:**

| Outcome | Probability | Post-measurement state |
|---------|-------------|----------------------|
| ℏ² | 1/2 | (1/√2)(\|+1⟩ + \|-1⟩) |
| 0 | 1/2 | \|0⟩ |

The ℏ² eigenspace is 2D (degenerate), so collapse preserves the superposition within that subspace!

---

### 7. Sequential Measurements: Non-Commuting Observables

The most striking feature of quantum mechanics emerges when measuring non-commuting observables in sequence.

**Setup:** Start with |ψ₀⟩, measure A, then B, then A again.

**Example: Spin-1/2**

Let |ψ₀⟩ = |↑⟩ (eigenstate of Sz)

**Sequence:**
1. Measure Sz → get +ℏ/2 (certainty), state = |↑⟩
2. Measure Sx → get +ℏ/2 (50%) or -ℏ/2 (50%), state = |+x⟩ or |-x⟩
3. Measure Sz again → get +ℏ/2 (50%) or -ℏ/2 (50%)

**Remarkable:** The intermediate Sx measurement **destroys** the Sz = +ℏ/2 we started with!

This is because:
$$|+x⟩ = \frac{1}{\sqrt{2}}(|↑⟩ + |↓⟩), \quad |-x⟩ = \frac{1}{\sqrt{2}}(|↑⟩ - |↓⟩)$$

Each contains both Sz eigenstates.

**General principle:** Measuring B disturbs the state if [Â, B̂] ≠ 0.

---

### 8. The Quantum Zeno Effect

Frequent measurement can "freeze" quantum evolution!

**Setup:** System evolves under Hamiltonian Ĥ, starts in |ψ₀⟩ (eigenstate of observable A).

**Without measurement:** State evolves: |ψ(t)⟩ = e^{-iĤt/ℏ}|ψ₀⟩

For small t: $|ψ(t)⟩ ≈ |ψ₀⟩ - \frac{it}{\hbar}\hat{H}|ψ₀⟩$

Probability of remaining in |ψ₀⟩:
$$P(t) = |⟨ψ₀|ψ(t)⟩|^2 ≈ 1 - \frac{t^2}{\hbar^2}⟨ψ₀|\hat{H}^2|ψ₀⟩ = 1 - \left(\frac{t}{τ}\right)^2$$

**With frequent measurements:** Measure A every Δt:

After each measurement (if we get the initial eigenvalue):
- State resets to |ψ₀⟩
- Next interval starts fresh

Probability of staying in |ψ₀⟩ through N measurements (total time T = NΔt):
$$P_N ≈ \left(1 - \frac{(Δt)^2}{τ^2}\right)^N ≈ 1 - \frac{T \cdot Δt}{τ^2}$$

As Δt → 0 (continuous measurement): P → 1

**Conclusion:** Continuous measurement freezes the system in its initial state!

$$\boxed{\lim_{N→∞, Δt→0} P_N = 1 \quad \text{(Quantum Zeno Effect)}}$$

Named after Zeno's paradox: "A watched pot never boils."

---

## Quantum Computing Connection

### Measurement-Based Reset

In quantum computing, measurement can reset qubits:

```python
from qiskit import QuantumCircuit

# Reset qubit to |0⟩ using measurement
qc = QuantumCircuit(1, 1)
qc.measure(0, 0)      # Measure qubit
# If result is 1, apply X gate to flip to |0⟩
# In Qiskit: qc.reset(0) does this automatically
```

### Mid-Circuit Measurement

Modern quantum computers support measurement mid-circuit:

```python
qc = QuantumCircuit(2, 2)
qc.h(0)                # Create superposition
qc.cx(0, 1)           # Entangle
qc.measure(0, 0)      # Measure first qubit (collapses both!)
qc.h(1)               # Apply gate to second qubit
qc.measure(1, 1)      # Final measurement
```

After measuring qubit 0, qubit 1 also collapses due to entanglement!

### Quantum Error Correction

Syndrome measurement detects errors without collapsing the logical qubit:

1. Encode logical qubit in multiple physical qubits
2. Measure **parity operators** (e.g., ZZ, XX)
3. Parity measurement projects onto error subspaces
4. Apply correction based on syndrome

The projection formalism is essential for understanding QEC.

### Measurement-Based Quantum Computing

An alternative model where computation proceeds by:
1. Prepare highly entangled "cluster state"
2. Perform sequence of single-qubit measurements
3. Measurement angles determine computation
4. Each measurement collapses part of the state, driving computation forward

---

## Worked Examples

### Example 1: Sequential Spin Measurements

**Problem:** A spin-1/2 particle starts in state |+⟩ = (|↑⟩ + |↓⟩)/√2. We measure Sz, then measure Sz again. Calculate:
(a) Probability of getting +ℏ/2, then +ℏ/2
(b) Probability of getting +ℏ/2, then -ℏ/2
(c) Probability of getting -ℏ/2 on the second measurement (regardless of first)

**Solution:**

(a) First measurement Sz on |+⟩:
- P(+ℏ/2) = |⟨↑|+⟩|² = |1/√2|² = 1/2
- After: state = |↑⟩

Second measurement Sz on |↑⟩:
- P(+ℏ/2) = |⟨↑|↑⟩|² = 1

Joint probability:
$$P(+,+) = \frac{1}{2} × 1 = \boxed{\frac{1}{2}}$$

(b) If first measurement gives +ℏ/2, state = |↑⟩.
Second measurement: P(-ℏ/2) = |⟨↓|↑⟩|² = 0

$$P(+,-) = \frac{1}{2} × 0 = \boxed{0}$$

(c) Second measurement gives -ℏ/2 in two scenarios:
- First gives +ℏ/2, second gives -ℏ/2: P = 1/2 × 0 = 0
- First gives -ℏ/2, second gives -ℏ/2: P = 1/2 × 1 = 1/2

Total:
$$P(\text{second} = -ℏ/2) = 0 + \frac{1}{2} = \boxed{\frac{1}{2}}$$

---

### Example 2: Degenerate Eigenvalue Projection

**Problem:** A spin-1 system is in state:
$$|ψ⟩ = \frac{1}{2}|+1⟩ + \frac{1}{2}|0⟩ + \frac{1}{\sqrt{2}}|-1⟩$$

We measure $\hat{S}_z^2$, which has eigenvalue ℏ² for |±1⟩ and 0 for |0⟩.

(a) What is the projector onto the ℏ² eigenspace?
(b) What is P(ℏ²)?
(c) What is the post-measurement state if we get ℏ²?

**Solution:**

(a) The ℏ² eigenspace is spanned by |+1⟩ and |-1⟩:
$$\boxed{\hat{P}_{\hbar^2} = |+1⟩⟨+1| + |-1⟩⟨-1|}$$

(b) Apply projector to state:
$$\hat{P}_{\hbar^2}|ψ⟩ = \frac{1}{2}|+1⟩ + \frac{1}{\sqrt{2}}|-1⟩$$

Probability:
$$P(ℏ²) = \|\hat{P}_{\hbar^2}|ψ⟩\|^2 = \left|\frac{1}{2}\right|^2 + \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{4} + \frac{1}{2} = \boxed{\frac{3}{4}}$$

(c) Normalize the projected state:
$$|ψ'⟩ = \frac{\hat{P}_{\hbar^2}|ψ⟩}{\|\hat{P}_{\hbar^2}|ψ⟩\|} = \frac{1}{\sqrt{3/4}}\left(\frac{1}{2}|+1⟩ + \frac{1}{\sqrt{2}}|-1⟩\right)$$

$$\boxed{|ψ'⟩ = \frac{1}{\sqrt{3}}|+1⟩ + \sqrt{\frac{2}{3}}|-1⟩}$$

---

### Example 3: Three Sequential Measurements

**Problem:** Start with |↑⟩. Measure Sx, then Sy, then Sz. If we get +ℏ/2 each time, what is the probability?

Given:
- |+x⟩ = (|↑⟩ + |↓⟩)/√2
- |+y⟩ = (|↑⟩ + i|↓⟩)/√2

**Solution:**

**Step 1:** Measure Sx on |↑⟩
$$|↑⟩ = \frac{1}{\sqrt{2}}(|+x⟩ + |-x⟩)$$
$$P(S_x = +ℏ/2) = \frac{1}{2}$$
After: state = |+x⟩

**Step 2:** Measure Sy on |+x⟩
Express |+x⟩ in Sy basis:
$$|+x⟩ = \frac{1}{\sqrt{2}}(|↑⟩ + |↓⟩)$$
$$|+y⟩ = \frac{1}{\sqrt{2}}(|↑⟩ + i|↓⟩), \quad |-y⟩ = \frac{1}{\sqrt{2}}(|↑⟩ - i|↓⟩)$$

$$⟨+y|+x⟩ = \frac{1}{2}(⟨↑| - i⟨↓|)(|↑⟩ + |↓⟩) = \frac{1}{2}(1 - i)$$
$$P(S_y = +ℏ/2) = |⟨+y|+x⟩|^2 = \frac{|1-i|^2}{4} = \frac{2}{4} = \frac{1}{2}$$

After: state = |+y⟩

**Step 3:** Measure Sz on |+y⟩
$$|+y⟩ = \frac{1}{\sqrt{2}}(|↑⟩ + i|↓⟩)$$
$$P(S_z = +ℏ/2) = |⟨↑|+y⟩|^2 = \frac{1}{2}$$

**Total probability:**
$$P = \frac{1}{2} × \frac{1}{2} × \frac{1}{2} = \boxed{\frac{1}{8}}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Basic collapse:** A qubit in state |ψ⟩ = (|0⟩ + |1⟩)/√2 is measured in the computational basis.
   (a) What are the possible outcomes?
   (b) What is the state after measuring 0?
   (c) What is the state after measuring 1?

2. **Projection operator:** Write down the projection operator for eigenstate |+⟩ = (|0⟩ + |1⟩)/√2.
   Verify that P̂₊² = P̂₊.

3. **Repeated measurement:** Starting from |ψ⟩ = (3|0⟩ + 4i|1⟩)/5, measure Z, then measure Z again.
   What is the probability of getting +1 both times?

### Level 2: Intermediate

4. **Degenerate projection:** A particle in 3D can have orbital angular momentum states |l,m⟩. Consider L² measurement on:
   $$|ψ⟩ = \frac{1}{\sqrt{3}}|1,1⟩ + \frac{1}{\sqrt{3}}|1,0⟩ + \frac{1}{\sqrt{3}}|2,0⟩$$
   (a) What is P(L² = 2ℏ²)?
   (b) What is the post-measurement state if we get L² = 2ℏ²?

5. **Three observables:** Starting from |↑⟩, calculate the probability of the sequence:
   Sz → +ℏ/2, then Sx → +ℏ/2, then Sz → -ℏ/2

6. **Selective measurement:** A spin-1 system is in state |ψ⟩ = (|+1⟩ + |0⟩ + |-1⟩)/√3.
   We measure Sz but only record whether the result is 0 or non-zero (we don't distinguish +ℏ from -ℏ).
   What is the post-measurement state if we record "non-zero"?

### Level 3: Challenging

7. **Quantum Zeno:** A two-level system evolves under H = ℏω σx/2, starting in |0⟩.
   (a) Find the probability of remaining in |0⟩ after time t (no measurement).
   (b) If we measure Z every Δt = T/N, find P(all measurements give +1) for large N.
   (c) Show P → 1 as N → ∞.

8. **Non-selective measurement:** Consider a measurement where we don't record the outcome. Show that the post-measurement density matrix is:
   $$ρ' = \sum_a \hat{P}_a ρ \hat{P}_a$$
   Apply this to ρ = |+⟩⟨+| with Z measurement.

9. **Research:** The measurement postulate is controversial. Research the "measurement problem" and explain at least two proposed resolutions (e.g., Copenhagen interpretation, Many-Worlds, decoherence).

---

## Computational Lab

### Objective
Implement state collapse and explore sequential measurements.

```python
"""
Day 345 Computational Lab: State Collapse
Quantum Mechanics Core - Year 1, Week 50
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# =============================================================================
# Part 1: Projection Operators
# =============================================================================

print("=" * 70)
print("Part 1: Projection Operators")
print("=" * 70)

def projector(eigenstate: np.ndarray) -> np.ndarray:
    """
    Construct projection operator P = |a⟩⟨a| for a non-degenerate eigenstate.

    Parameters:
        eigenstate: Column vector |a⟩

    Returns:
        Projection matrix P = |a⟩⟨a|
    """
    return eigenstate @ eigenstate.conj().T

def multi_projector(eigenstates: List[np.ndarray]) -> np.ndarray:
    """
    Construct projection operator for degenerate eigenspace.
    P = Σᵢ |aᵢ⟩⟨aᵢ|

    Parameters:
        eigenstates: List of orthonormal eigenstates spanning the eigenspace

    Returns:
        Projection matrix onto the eigenspace
    """
    P = np.zeros((eigenstates[0].shape[0], eigenstates[0].shape[0]), dtype=complex)
    for state in eigenstates:
        P += projector(state)
    return P

# Define qubit basis
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Projection operators
P_0 = projector(ket_0)
P_1 = projector(ket_1)

print("\nProjection operators for computational basis:")
print(f"\nP̂₀ = |0⟩⟨0| =\n{P_0}")
print(f"\nP̂₁ = |1⟩⟨1| =\n{P_1}")

# Verify properties
print("\nVerifying projector properties:")
print(f"P̂₀² = P̂₀? {np.allclose(P_0 @ P_0, P_0)}")
print(f"P̂₀† = P̂₀? {np.allclose(P_0.conj().T, P_0)}")
print(f"P̂₀ P̂₁ = 0? {np.allclose(P_0 @ P_1, 0)}")
print(f"P̂₀ + P̂₁ = I? {np.allclose(P_0 + P_1, np.eye(2))}")

# =============================================================================
# Part 2: State Collapse Implementation
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: State Collapse")
print("=" * 70)

def collapse(state: np.ndarray, projector: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply measurement collapse using projection operator.

    Parameters:
        state: Pre-measurement state |ψ⟩
        projector: Projection operator P̂ₐ for the measured eigenvalue

    Returns:
        collapsed_state: Normalized post-measurement state
        probability: Probability of this outcome
    """
    # Apply projector
    projected = projector @ state

    # Calculate probability
    probability = np.vdot(projected, projected).real

    if probability < 1e-10:
        # This outcome has zero probability
        return None, 0.0

    # Normalize
    collapsed_state = projected / np.sqrt(probability)

    return collapsed_state, probability

def measure(state: np.ndarray, basis: List[np.ndarray],
            eigenvalues: Optional[List[float]] = None) -> Tuple[int, np.ndarray, float]:
    """
    Simulate a quantum measurement.

    Parameters:
        state: State to measure
        basis: Measurement basis (list of orthonormal eigenstates)
        eigenvalues: Optional list of eigenvalues (for labeling)

    Returns:
        outcome_index: Index of the measured eigenvalue
        collapsed_state: Post-measurement state
        eigenvalue: The measured eigenvalue (or index if eigenvalues not given)
    """
    # Calculate probabilities
    probs = []
    for b in basis:
        P = projector(b)
        _, p = collapse(state, P)
        probs.append(p)

    probs = np.array(probs)

    # Sample outcome
    outcome_index = np.random.choice(len(basis), p=probs)

    # Collapse state
    P = projector(basis[outcome_index])
    collapsed_state, _ = collapse(state, P)

    eigenvalue = eigenvalues[outcome_index] if eigenvalues else outcome_index

    return outcome_index, collapsed_state, eigenvalue

# Test collapse
psi = (ket_0 + ket_1) / np.sqrt(2)  # |+⟩ state
print(f"\nInitial state: |+⟩ = (|0⟩ + |1⟩)/√2 = {psi.flatten()}")

# Collapse to |0⟩
collapsed_0, prob_0 = collapse(psi, P_0)
print(f"\nAfter measuring 0:")
print(f"Probability: {prob_0:.4f}")
print(f"Collapsed state: {collapsed_0.flatten()}")

# Collapse to |1⟩
collapsed_1, prob_1 = collapse(psi, P_1)
print(f"\nAfter measuring 1:")
print(f"Probability: {prob_1:.4f}")
print(f"Collapsed state: {collapsed_1.flatten()}")

# =============================================================================
# Part 3: Sequential Measurements
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Sequential Measurements")
print("=" * 70)

# Define spin-1/2 bases
ket_up = ket_0
ket_down = ket_1
z_basis = [ket_up, ket_down]

ket_plus_x = (ket_up + ket_down) / np.sqrt(2)
ket_minus_x = (ket_up - ket_down) / np.sqrt(2)
x_basis = [ket_plus_x, ket_minus_x]

ket_plus_y = (ket_up + 1j*ket_down) / np.sqrt(2)
ket_minus_y = (ket_up - 1j*ket_down) / np.sqrt(2)
y_basis = [ket_plus_y, ket_minus_y]

def sequential_measurement_experiment(initial_state: np.ndarray,
                                        measurements: List[Tuple[str, List[np.ndarray]]],
                                        n_trials: int = 10000) -> dict:
    """
    Run sequential measurement experiment many times and collect statistics.

    Parameters:
        initial_state: Starting state
        measurements: List of (name, basis) pairs for sequential measurements
        n_trials: Number of experimental runs

    Returns:
        Dictionary with outcome frequencies
    """
    outcomes_record = []

    for _ in range(n_trials):
        state = initial_state.copy()
        trial_outcomes = []

        for name, basis in measurements:
            idx, state, _ = measure(state, basis)
            trial_outcomes.append(idx)

        outcomes_record.append(tuple(trial_outcomes))

    # Count frequencies
    from collections import Counter
    counts = Counter(outcomes_record)

    return {k: v/n_trials for k, v in counts.items()}

# Experiment 1: Z, then Z again (should always match)
print("\nExperiment 1: Measure Z, then Z (starting from |+⟩)")
result1 = sequential_measurement_experiment(
    (ket_up + ket_down)/np.sqrt(2),
    [('Z', z_basis), ('Z', z_basis)],
    n_trials=5000
)
print("Outcome frequencies (Z₁, Z₂):")
for outcome, freq in sorted(result1.items()):
    z1 = '+' if outcome[0] == 0 else '-'
    z2 = '+' if outcome[1] == 0 else '-'
    print(f"  ({z1}, {z2}): {freq:.4f}")

# Experiment 2: Z, then X, then Z
print("\nExperiment 2: Measure Z, then X, then Z (starting from |↑⟩)")
result2 = sequential_measurement_experiment(
    ket_up,
    [('Z', z_basis), ('X', x_basis), ('Z', z_basis)],
    n_trials=5000
)
print("Outcome frequencies (Z₁, X, Z₂):")
for outcome, freq in sorted(result2.items()):
    z1 = '+' if outcome[0] == 0 else '-'
    x = '+' if outcome[1] == 0 else '-'
    z2 = '+' if outcome[2] == 0 else '-'
    print(f"  ({z1}, {x}, {z2}): {freq:.4f}")

# Verify: P(Z₁=+, X=+, Z₂=-) should be 1/8
expected = 1.0 * 0.5 * 0.5  # P(Z=+|↑)=1, P(X=+|↑)=0.5, P(Z=-|+x)=0.5
print(f"\nTheoretical P(+,+,-) = {expected:.4f}")

# =============================================================================
# Part 4: Degenerate Eigenvalue Collapse
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Degenerate Eigenvalue Collapse (Spin-1)")
print("=" * 70)

# Spin-1 basis: |+1⟩, |0⟩, |-1⟩
ket_p1 = np.array([[1], [0], [0]], dtype=complex)
ket_0_spin1 = np.array([[0], [1], [0]], dtype=complex)
ket_m1 = np.array([[0], [0], [1]], dtype=complex)

# Sz² has eigenvalue ℏ² for |±1⟩ and 0 for |0⟩
P_nonzero = multi_projector([ket_p1, ket_m1])  # Degenerate eigenspace
P_zero = projector(ket_0_spin1)

print("Spin-1: Sz² measurement")
print(f"\nP̂(Sz²=ℏ²) (projects onto degenerate subspace):")
print(P_nonzero)

# Initial state
psi_spin1 = (ket_p1 + ket_0_spin1 + np.sqrt(2)*ket_m1) / 2
print(f"\nInitial state: |ψ⟩ = (|+1⟩ + |0⟩ + √2|-1⟩)/2")
print(f"|ψ⟩ = {psi_spin1.flatten()}")

# Collapse to Sz² = ℏ² eigenspace
collapsed_nonzero, prob_nonzero = collapse(psi_spin1, P_nonzero)
print(f"\nMeasure Sz², get ℏ²:")
print(f"Probability: {prob_nonzero:.4f}")
print(f"Post-measurement state: {collapsed_nonzero.flatten()}")
print("Note: Still a superposition of |+1⟩ and |-1⟩!")

# Collapse to Sz² = 0
collapsed_zero, prob_zero = collapse(psi_spin1, P_zero)
print(f"\nMeasure Sz², get 0:")
print(f"Probability: {prob_zero:.4f}")
print(f"Post-measurement state: {collapsed_zero.flatten()}")

# =============================================================================
# Part 5: Quantum Zeno Effect Simulation
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Quantum Zeno Effect")
print("=" * 70)

def evolve(state: np.ndarray, H: np.ndarray, dt: float) -> np.ndarray:
    """
    Evolve state under Hamiltonian for time dt.
    |ψ(t+dt)⟩ = exp(-iHdt/ℏ)|ψ(t)⟩

    Using Taylor expansion for small dt.
    """
    # For simplicity, use matrix exponential
    from scipy.linalg import expm
    U = expm(-1j * H * dt)
    return U @ state

def zeno_experiment(initial_state: np.ndarray, H: np.ndarray,
                     total_time: float, n_measurements: int,
                     basis: List[np.ndarray], n_trials: int = 1000) -> float:
    """
    Simulate quantum Zeno effect.

    Returns probability of staying in initial state throughout.
    """
    dt = total_time / n_measurements
    P_initial = projector(initial_state)
    survival_count = 0

    for _ in range(n_trials):
        state = initial_state.copy()
        survived = True

        for _ in range(n_measurements):
            # Evolve
            state = evolve(state, H, dt)

            # Measure
            idx, state, _ = measure(state, basis)

            # Check if still in initial state
            if idx != 0:  # Assuming initial state is first in basis
                survived = False
                break

        if survived:
            survival_count += 1

    return survival_count / n_trials

# Hamiltonian: H = ω σx (induces Rabi oscillations)
omega = 1.0
H = omega * np.array([[0, 1], [1, 0]], dtype=complex)

# Total evolution time (half Rabi period for maximum contrast)
T = np.pi / (2 * omega)

# Compare different measurement frequencies
n_meas_values = [1, 2, 5, 10, 20, 50, 100]
survival_probs = []

print(f"\nHamiltonian: H = ω σx, Total time T = π/(2ω)")
print(f"Initial state: |0⟩")
print(f"\nWithout measurement: P(stay in |0⟩) = cos²(ωT) = 0")
print("\nWith measurements:")

for n in n_meas_values:
    p_survive = zeno_experiment(ket_0, H, T, n, z_basis, n_trials=2000)
    survival_probs.append(p_survive)
    print(f"  N = {n:3d} measurements: P(survive) = {p_survive:.4f}")

# Theoretical prediction for Zeno effect
# P ≈ 1 - (ωT)²/N for large N
T_val = T
n_theory = np.linspace(1, 100, 100)
p_theory = 1 - (omega * T_val)**2 / n_theory
p_theory = np.clip(p_theory, 0, 1)

# =============================================================================
# Part 6: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Sequential measurements statistics
ax1 = axes[0, 0]
outcomes = [(0,0,0), (0,0,1), (0,1,0), (0,1,1)]
freqs = [result2.get(o, 0) for o in outcomes]
labels = [f'(+,{["+" if o[1]==0 else "-" for o in [o]][0]},{["+" if o[2]==0 else "-" for o in [o]][0]})'
          for o in outcomes]
# Simpler labels
labels = ['(+,+,+)', '(+,+,-)', '(+,-,+)', '(+,-,-)']
ax1.bar(range(4), freqs, color='steelblue', alpha=0.7)
ax1.axhline(y=0.125, color='red', linestyle='--', label='Theory: 1/8')
ax1.set_xticks(range(4))
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel('Frequency')
ax1.set_title('Sequential Z-X-Z Measurement (start: |↑⟩)')
ax1.legend()

# Panel 2: Quantum Zeno effect
ax2 = axes[0, 1]
ax2.plot(n_meas_values, survival_probs, 'bo-', markersize=8, linewidth=2,
         label='Simulation')
ax2.plot(n_theory, p_theory, 'r-', linewidth=2, alpha=0.5,
         label='Theory: 1 - (ωT)²/N')
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Number of measurements N')
ax2.set_ylabel('Survival probability')
ax2.set_title('Quantum Zeno Effect')
ax2.legend()
ax2.set_xlim(0, 105)
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True, alpha=0.3)

# Panel 3: Projection visualization (Bloch sphere projection)
ax3 = axes[1, 0]
theta = np.linspace(0, 2*np.pi, 100)
ax3.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)

# Initial state on Bloch sphere (|+⟩ is on equator)
ax3.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.05, fc='blue', ec='blue')
ax3.text(1.1, 0.1, '|+⟩', fontsize=12, color='blue')

# Z projection (onto poles)
ax3.plot([0], [1], 'ro', markersize=15, label='|↑⟩ (Z=+1)')
ax3.plot([0], [-1], 'go', markersize=15, label='|↓⟩ (Z=-1)')
ax3.arrow(0.7, 0, 0, 0.3, head_width=0.08, head_length=0.05, fc='red', ec='red',
          linestyle='--', alpha=0.7)
ax3.arrow(0.7, 0, 0, -0.3, head_width=0.08, head_length=0.05, fc='green', ec='green',
          linestyle='--', alpha=0.7)

ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.set_xlabel('x')
ax3.set_ylabel('z')
ax3.set_title('Z-Measurement Collapse (Bloch sphere side view)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Panel 4: Repeated vs. interrupted measurements
ax4 = axes[1, 1]

# Simulate: start in |+⟩, measure Z repeatedly
n_steps = 20
state = (ket_up + ket_down) / np.sqrt(2)

# Track Z expectation after each step
exp_z_repeated = []
exp_z_with_x = []

# Path 1: Z measurements only
for i in range(n_steps):
    exp_z_repeated.append(np.real(state.conj().T @ np.array([[1,0],[0,-1]]) @ state)[0,0])
    idx, state, _ = measure(state, z_basis)

# Path 2: Alternating Z and X
state = (ket_up + ket_down) / np.sqrt(2)
for i in range(n_steps):
    exp_z_with_x.append(np.real(state.conj().T @ np.array([[1,0],[0,-1]]) @ state)[0,0])
    if i % 2 == 0:
        idx, state, _ = measure(state, z_basis)
    else:
        idx, state, _ = measure(state, x_basis)

ax4.plot(range(n_steps), exp_z_repeated, 'b-o', label='Z only', markersize=4)
ax4.plot(range(n_steps), exp_z_with_x, 'r-s', label='Alternating Z/X', markersize=4)
ax4.set_xlabel('Measurement number')
ax4.set_ylabel('⟨σz⟩')
ax4.set_title('Repeated vs. Interrupted Measurements')
ax4.legend()
ax4.set_ylim(-1.2, 1.2)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_345_state_collapse.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_345_state_collapse.png'")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Collapse (non-degenerate) | \|ψ⟩ → \|a⟩ after measuring a |
| Projection operator | P̂ₐ = \|a⟩⟨a\| |
| Collapse (general) | \|ψ'⟩ = P̂ₐ\|ψ⟩ / \|\|P̂ₐ\|ψ⟩\|\| |
| Degenerate projector | P̂ₐ = Σᵢ \|a,i⟩⟨a,i\| |
| Repeated measurement | P(same result) = 1 |
| Projector properties | P̂² = P̂, P̂† = P̂ |

### Main Takeaways

1. **Measurement changes the state** — The system collapses to an eigenstate of the measured observable
2. **Projection operators encode collapse** — P̂ₐ = |a⟩⟨a| projects onto the eigenspace
3. **Repeated measurements are stable** — Measuring the same observable twice gives the same result
4. **Degenerate eigenvalues preserve superposition** — Collapse is to the eigenspace, not a specific eigenstate
5. **Non-commuting observables disturb each other** — Measuring B destroys information about A if [A,B] ≠ 0
6. **Quantum Zeno effect** — Frequent measurement can freeze evolution

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.3
- [ ] Read Sakurai Chapter 1.5
- [ ] Derive the collapse formula using projection operators
- [ ] Work through all three examples in detail
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Explain in your own words why measurement is irreversible

---

## Preview: Day 346

Tomorrow we study **expectation values** — how to extract average results from quantum states. The formula ⟨Â⟩ = ⟨ψ|Â|ψ⟩ connects the abstract operator to measurable averages, and the variance (ΔA)² quantifies quantum uncertainty.

---

*"The act of measurement causes the system to jump into an eigenstate of the dynamical variable that is being measured."* — Paul Dirac

---

**Next:** [Day_346_Wednesday.md](Day_346_Wednesday.md) — Expectation Values
