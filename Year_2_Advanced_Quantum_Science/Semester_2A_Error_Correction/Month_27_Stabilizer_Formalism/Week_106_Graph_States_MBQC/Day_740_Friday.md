# Day 740: Measurement-Based QC Foundations

## Overview

**Day:** 740 of 1008
**Week:** 106 (Graph States & MBQC)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Foundations of Measurement-Based Quantum Computation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | MBQC fundamentals |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Gate implementation |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational examples |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Explain** the MBQC computational model
2. **Describe** how measurements drive computation
3. **Understand** byproduct operators and corrections
4. **Implement** single-qubit gates via measurement
5. **Connect** teleportation to MBQC
6. **Analyze** measurement patterns for simple circuits

---

## Core Content

### The MBQC Paradigm

**Traditional Circuit Model:**
- Prepare input state
- Apply sequence of gates
- Measure output

**Measurement-Based Model:**
- Prepare entangled resource state (graph state)
- Perform adaptive single-qubit measurements
- Classical processing determines corrections
- Output appears on unmeasured qubits

**Key Insight:** Entanglement is prepared first; computation is driven by measurements.

### The Basic MBQC Process

**1. Prepare resource state:**
$$|\psi_{\text{resource}}\rangle = |G\rangle \text{ (graph state)}$$

**2. Perform measurements:**
- Measure qubits in bases parameterized by angles
- Results are random (0 or 1)
- Feed-forward: later measurements depend on earlier results

**3. Apply corrections:**
- Pauli corrections based on measurement outcomes
- Deterministic computation despite random outcomes

**4. Read output:**
- Unmeasured qubits hold the result

### Single-Qubit Measurement Bases

**Pauli-X eigenbasis:**
$$|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \quad |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

**General XY-plane measurement:**
$$|\pm_\theta\rangle = \frac{|0\rangle \pm e^{i\theta}|1\rangle}{\sqrt{2}}$$

Measurement operator:
$$M_\theta = |+_\theta\rangle\langle+_\theta| - |-_\theta\rangle\langle-_\theta|$$

### Teleportation as MBQC Primitive

**Setup:**
- Input qubit 1 in state |ψ⟩
- Ancilla qubit 2 in |+⟩
- Apply CZ to create entanglement

**State after CZ:**
$$CZ_{12}(|\psi\rangle \otimes |+\rangle)$$

**Measure qubit 1 in X-basis:**
- Outcome s ∈ {0, 1}
- Qubit 2 becomes $Z^s |\psi\rangle$ (up to phase)

**This teleports |ψ⟩ to qubit 2 with a Z^s byproduct!**

### Byproduct Operators

**Definition:**
Byproduct operators are Pauli corrections that accumulate from measurement randomness.

**Propagation:**
Byproducts propagate through the computation:
$$X^s |\psi\rangle \xrightarrow{U} U X^s U^\dagger \cdot U|\psi\rangle$$

For Clifford U, this gives another Pauli!

### Implementing Single-Qubit Rotations

**Z-rotation by angle θ:**

**Setup:** Linear graph state: input—ancilla

**Measurement:** Measure input in basis $|\pm_\theta\rangle$

**Result:**
$$|\psi\rangle \xrightarrow{\text{measure}} X^s R_Z(\theta) |\psi\rangle$$

The Z-rotation is implemented, with byproduct X^s.

**X-rotation:**
Use Hadamard conjugation: $R_X(\theta) = H R_Z(\theta) H$

### The Wire (Identity Operation)

**Simplest MBQC:**
- Two-qubit graph: 1—2
- Input on qubit 1, output on qubit 2
- Measure qubit 1 in X-basis

**Effect:**
$$|+\rangle_1 \xrightarrow{CZ} \text{entangled} \xrightarrow{M_X(1)} Z^{s_1}|\psi\rangle_2$$

The identity is implemented with a Z byproduct.

### Implementing Arbitrary Single-Qubit Gates

**Decomposition:**
Any single-qubit unitary:
$$U = e^{i\alpha} R_Z(\beta) R_X(\gamma) R_Z(\delta)$$

**MBQC implementation:**
- 4-qubit linear graph
- Measurements at angles α, β, γ, δ
- Adaptivity handles byproducts

### The J-Basis Measurement

**Definition:**
$$|J(\alpha)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\alpha}|1\rangle)$$

Measuring in this basis implements:
$$J(\alpha) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & e^{i\alpha} \\ 1 & -e^{i\alpha} \end{pmatrix} = H R_Z(\alpha)$$

### Measurement Pattern Notation

**Pattern notation:**
- $M_a^{s,t}(\alpha)$: Measure qubit a in angle α
- Dependent angle: $\alpha' = (-1)^s \alpha + t\pi$
- s, t are previous outcomes

**Example pattern for R_Z(θ):**
$$M_1^0(θ)$$
Output: $X^{s_1} R_Z(θ) |ψ⟩$

### Feed-Forward and Adaptivity

**Why adaptive?**
Byproducts must be corrected or tracked.

**Correction strategies:**
1. **Active correction:** Apply Pauli gates after measurements
2. **Tracking:** Propagate Paulis through remaining circuit
3. **Adaptive angles:** Modify future measurement angles

**Adaptive angle formula:**
If byproduct $X^s$ precedes measurement at angle α:
$$\text{New angle} = (-1)^s \alpha$$

---

## Worked Examples

### Example 1: Teleportation with Corrections

**State:** $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ on qubit 1

**Graph:** 1—2 (edge between qubits 1 and 2)

**Initial state:**
$$|\psi\rangle_1 |+\rangle_2 \xrightarrow{CZ} \alpha|0+\rangle + \beta|1-\rangle$$

**Measure qubit 1 in X-basis:**
- Project onto $|+\rangle_1$ (outcome 0):
  $$\langle +|_1 (\alpha|0+\rangle + \beta|1-\rangle) = \frac{1}{\sqrt{2}}(\alpha|+\rangle_2 + \beta|-\rangle_2) = \frac{1}{\sqrt{2}}|\psi\rangle_2$$

- Project onto $|-\rangle_1$ (outcome 1):
  $$\langle -|_1 (\alpha|0+\rangle + \beta|1-\rangle) = \frac{1}{\sqrt{2}}(\alpha|+\rangle_2 - \beta|-\rangle_2) = \frac{1}{\sqrt{2}}Z|\psi\rangle_2$$

**Result:** Qubit 2 has $Z^{s_1}|\psi\rangle$ (up to normalization).

### Example 2: Z-Rotation

**Implement $R_Z(\theta)$ via MBQC:**

**Graph:** 1—2

**Measure qubit 1 in basis:**
$$|\pm_\theta\rangle = \frac{1}{\sqrt{2}}(|0\rangle \pm e^{i\theta}|1\rangle)$$

**Analysis:**
Starting from $|\psi\rangle_1|+\rangle_2$ after CZ:
$$(\alpha|0\rangle + \beta|1\rangle)|+\rangle \xrightarrow{CZ} \alpha|0+\rangle + \beta|1-\rangle$$

Projecting onto $|+_\theta\rangle$:
$$\langle+_\theta|_1(\alpha|0+\rangle + \beta|1-\rangle)$$
$$= \frac{1}{\sqrt{2}}(\alpha|+\rangle + \beta e^{i\theta}|-\rangle)$$
$$= \frac{1}{\sqrt{2}}R_Z(\theta)|\psi\rangle$$

Projecting onto $|-_\theta\rangle$:
$$= \frac{1}{\sqrt{2}}X R_Z(\theta)|\psi\rangle$$

**Output:** $X^{s_1} R_Z(\theta) |\psi\rangle$

### Example 3: Hadamard Gate

**Implement H via MBQC:**

$$H = R_Z(\pi/2) R_X(\pi/2) R_Z(\pi/2)$$

But simpler: $H$ is just measurement in the computational basis followed by X correction!

**Alternative:** Use the fact that measuring in Z-basis after X-preparation implements H.

**Graph:** 1—2—3

**Measurements:**
- Qubit 1: angle 0 (X-basis)
- Qubit 2: angle π/2

This produces $X^{s_1} Z^{s_2} H |\psi\rangle$

### Example 4: CNOT Implementation

**CNOT requires a 2D cluster!**

**Minimal pattern:**
```
    1—3
    |
    2—4
```

**Measurement pattern:**
- Measure 1 in X-basis
- Measure 2 in X-basis
- Output on 3 (control) and 4 (target)

**Byproduct:** $X_3^{s_1} Z_4^{s_1} X_4^{s_2} Z_3^{s_2}$

---

## Practice Problems

### Level 1: Direct Application

1. **Wire Verification:** For a 2-qubit graph state 1—2:
   a) Write the state after CZ
   b) Compute the post-measurement state when qubit 1 gives outcome 0
   c) Verify the Z byproduct for outcome 1

2. **Byproduct Tracking:** If the current byproduct is $X^a Z^b$, what is it after:
   a) Applying H?
   b) Applying $R_Z(\theta)$?
   c) Applying CNOT (as control)?

3. **Measurement Basis:** Write $|\pm_\theta\rangle$ for:
   a) $\theta = 0$
   b) $\theta = \pi/2$
   c) $\theta = \pi$

### Level 2: Intermediate

4. **T-Gate Implementation:** The T gate is $R_Z(\pi/4)$.
   a) Describe the MBQC implementation
   b) What is the byproduct?
   c) How does the byproduct affect subsequent operations?

5. **Two-Rotation Sequence:** Implement $R_Z(\alpha) R_Z(\beta)$ via MBQC.
   a) What graph do you need?
   b) What are the measurement angles?
   c) What is the final byproduct?

6. **Adaptive Angles:** If outcome $s_1 = 1$, how do you modify the next measurement angle to compensate?

### Level 3: Challenging

7. **Universality:** Prove that measurements at angles $0, \pi/4, \pi/2$ are sufficient for universal single-qubit computation.

8. **Determinism:** Explain why MBQC is deterministic despite random measurement outcomes.

9. **Circuit Translation:** Convert the circuit $H \cdot T \cdot H$ to an MBQC pattern.

---

## Computational Lab

```python
"""
Day 740: Measurement-Based QC Foundations
=========================================
Simulating MBQC operations.
"""

import numpy as np
from typing import Tuple, List

# Basic states and gates
def ket(bits: str) -> np.ndarray:
    """Create computational basis state."""
    n = len(bits)
    idx = int(bits, 2)
    state = np.zeros(2**n, dtype=complex)
    state[idx] = 1
    return state

def plus_state() -> np.ndarray:
    return np.array([1, 1], dtype=complex) / np.sqrt(2)

def minus_state() -> np.ndarray:
    return np.array([1, -1], dtype=complex) / np.sqrt(2)

def plus_theta(theta: float) -> np.ndarray:
    """Measurement basis state |+_θ⟩."""
    return np.array([1, np.exp(1j * theta)], dtype=complex) / np.sqrt(2)

def minus_theta(theta: float) -> np.ndarray:
    """Measurement basis state |-_θ⟩."""
    return np.array([1, -np.exp(1j * theta)], dtype=complex) / np.sqrt(2)

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def R_Z(theta: float) -> np.ndarray:
    """Z-rotation by angle theta."""
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

def R_X(theta: float) -> np.ndarray:
    """X-rotation by angle theta."""
    return H @ R_Z(theta) @ H

def CZ_gate() -> np.ndarray:
    """Controlled-Z gate."""
    return np.diag([1, 1, 1, -1]).astype(complex)

def tensor(*args) -> np.ndarray:
    """Tensor product of matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result

def partial_trace_first(state: np.ndarray) -> np.ndarray:
    """Trace out first qubit, return reduced state of second."""
    # state is 4-dim (2 qubits)
    rho = np.outer(state, state.conj())
    # Partial trace over first qubit
    rho_reduced = np.zeros((2, 2), dtype=complex)
    rho_reduced[0, 0] = rho[0, 0] + rho[2, 2]
    rho_reduced[0, 1] = rho[0, 1] + rho[2, 3]
    rho_reduced[1, 0] = rho[1, 0] + rho[3, 2]
    rho_reduced[1, 1] = rho[1, 1] + rho[3, 3]
    return rho_reduced

def measure_qubit(state: np.ndarray, qubit: int, basis_0: np.ndarray,
                  basis_1: np.ndarray, n_qubits: int) -> Tuple[int, np.ndarray]:
    """
    Measure a qubit in specified basis.

    Returns (outcome, post_measurement_state).
    """
    # Build projectors
    proj_0 = np.outer(basis_0, basis_0.conj())
    proj_1 = np.outer(basis_1, basis_1.conj())

    # Extend to full system
    def extend_operator(op, q, n):
        parts = [I] * n
        parts[q] = op
        return tensor(*parts)

    P_0 = extend_operator(proj_0, qubit, n_qubits)
    P_1 = extend_operator(proj_1, qubit, n_qubits)

    # Probabilities
    p_0 = np.real(state.conj() @ P_0 @ state)
    p_1 = np.real(state.conj() @ P_1 @ state)

    # Simulate measurement
    if np.random.random() < p_0:
        outcome = 0
        post_state = P_0 @ state / np.sqrt(p_0)
    else:
        outcome = 1
        post_state = P_1 @ state / np.sqrt(p_1)

    return outcome, post_state

def mbqc_wire(psi: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    MBQC wire: teleport state through one edge.

    Input: |ψ⟩
    Graph: 1—2
    Output: Z^s |ψ⟩ on qubit 2
    """
    # Prepare |ψ⟩|+⟩
    state = tensor(psi, plus_state())

    # Apply CZ
    state = CZ_gate() @ state

    # Measure qubit 0 in X-basis
    outcome, post_state = measure_qubit(state, 0, plus_state(), minus_state(), 2)

    # Extract qubit 1 state (trace out measured qubit conceptually)
    # After projection, state is product: |±⟩|output⟩
    # Extract the second qubit
    if outcome == 0:
        output = post_state[:2] * np.sqrt(2)  # Unnormalize projection
    else:
        output = post_state[2:] * np.sqrt(2)

    # Normalize
    output = output / np.linalg.norm(output)

    return output, outcome

def mbqc_rz(psi: np.ndarray, theta: float) -> Tuple[np.ndarray, int]:
    """
    MBQC Z-rotation.

    Input: |ψ⟩
    Output: X^s R_Z(θ) |ψ⟩
    """
    # Prepare |ψ⟩|+⟩
    state = tensor(psi, plus_state())

    # Apply CZ
    state = CZ_gate() @ state

    # Measure qubit 0 in θ-basis
    outcome, post_state = measure_qubit(state, 0, plus_theta(theta),
                                         minus_theta(theta), 2)

    # Extract output qubit
    if outcome == 0:
        output = post_state[:2] * np.sqrt(2)
    else:
        output = post_state[2:] * np.sqrt(2)

    output = output / np.linalg.norm(output)

    return output, outcome

def verify_rz_implementation(theta: float, n_trials: int = 100) -> float:
    """Verify R_Z implementation by comparing to direct gate."""
    successes = 0

    for _ in range(n_trials):
        # Random input state
        alpha = np.random.randn() + 1j * np.random.randn()
        beta = np.random.randn() + 1j * np.random.randn()
        psi = np.array([alpha, beta], dtype=complex)
        psi = psi / np.linalg.norm(psi)

        # Expected output (before byproduct)
        expected = R_Z(theta) @ psi

        # MBQC output
        output, s = mbqc_rz(psi, theta)

        # Apply byproduct correction
        if s == 1:
            output = X @ output

        # Check equivalence (up to global phase)
        overlap = np.abs(np.dot(output.conj(), expected))
        if overlap > 0.99:
            successes += 1

    return successes / n_trials

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 740: Measurement-Based QC Foundations")
    print("=" * 60)

    np.random.seed(42)

    # Example 1: Wire
    print("\n1. MBQC Wire (Teleportation)")
    print("-" * 40)

    psi = np.array([0.6, 0.8], dtype=complex)
    print(f"Input: |ψ⟩ = {psi}")

    for trial in range(3):
        output, s = mbqc_wire(psi.copy())
        # Apply correction
        corrected = (np.linalg.matrix_power(Z, s) @ output)
        print(f"  Trial {trial+1}: s={s}, output after Z^s = {corrected}")

    # Example 2: Z-rotation
    print("\n2. MBQC Z-Rotation (θ = π/4)")
    print("-" * 40)

    theta = np.pi/4
    psi = np.array([1, 0], dtype=complex)  # |0⟩
    expected = R_Z(theta) @ psi

    print(f"Input: |0⟩")
    print(f"Expected R_Z(π/4)|0⟩ = {expected}")

    for trial in range(3):
        output, s = mbqc_rz(psi.copy(), theta)
        # Apply X^s correction
        corrected = np.linalg.matrix_power(X, s) @ output
        print(f"  Trial {trial+1}: s={s}, X^s·output = {corrected}")

    # Example 3: Verification
    print("\n3. Verification of R_Z Implementation")
    print("-" * 40)

    for theta in [0, np.pi/4, np.pi/2, np.pi]:
        success_rate = verify_rz_implementation(theta, n_trials=100)
        print(f"  θ = {theta:.4f}: success rate = {success_rate:.2%}")

    # Example 4: Byproduct propagation
    print("\n4. Byproduct Propagation Rules")
    print("-" * 40)

    print("X through H: H X H† =", "Z" if np.allclose(H @ X @ H.conj().T, Z) else "?")
    print("Z through H: H Z H† =", "X" if np.allclose(H @ Z @ H.conj().T, X) else "?")
    print("X through R_Z(θ): R_Z X R_Z† = X (up to phase)" if
          np.allclose(R_Z(0.5) @ X @ R_Z(0.5).conj().T, X) else "different")

    # Actually: R_Z(θ) X R_Z(-θ) = X
    print("Verify: R_Z(θ) X R_Z(-θ) =", "X ✓" if
          np.allclose(R_Z(0.5) @ X @ R_Z(-0.5), X) else "?")

    # Example 5: Full sequence
    print("\n5. Implementing H via MBQC")
    print("-" * 40)

    # H = R_Z(π/2) R_X(π/2) R_Z(π/2) up to global phase
    # Simpler: H is its own inverse, can use single measurement at π/2

    psi = np.array([1, 0], dtype=complex)  # |0⟩
    expected_H = H @ psi  # Should be |+⟩

    # MBQC: measure at θ = π/2 implements H R_Z(π/2) ~ H
    output, s = mbqc_rz(psi, np.pi/2)

    # The output should be related to H|ψ⟩
    print(f"Input: |0⟩")
    print(f"Expected H|0⟩ = |+⟩ = {expected_H}")
    print(f"MBQC output: {output}")
    print(f"Measurement outcome: s = {s}")

    print("\n" + "=" * 60)
    print("End of Day 740 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Wire | $\|ψ\rangle \to Z^s\|ψ\rangle$ |
| Z-rotation | $\|ψ\rangle \xrightarrow{M_θ} X^s R_Z(θ)\|ψ\rangle$ |
| Adaptive angle | $θ' = (-1)^s θ$ (for X byproduct) |
| Byproduct through H | $H X = Z H$, $H Z = X H$ |

### Main Takeaways

1. **MBQC** uses measurements to drive computation
2. **Byproduct operators** arise from measurement randomness
3. **Adaptive angles** or corrections maintain determinism
4. **Single-qubit gates** use linear graph states
5. **Two-qubit gates** require 2D cluster states
6. **Teleportation** is the fundamental MBQC primitive

---

## Daily Checklist

- [ ] I understand the MBQC paradigm
- [ ] I can implement the wire (identity) via MBQC
- [ ] I know how to implement Z-rotations
- [ ] I understand byproduct operators
- [ ] I can track byproducts through gates
- [ ] I know how adaptive measurements work

---

## Preview: Day 741

Tomorrow we study **Cluster States and Universality**:
- 2D cluster state structure
- Universal computation with measurements
- Computational depth in MBQC
- Error propagation in cluster states
