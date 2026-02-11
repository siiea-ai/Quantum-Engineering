# Day 823: Surface Code CNOT via Lattice Surgery

## Week 118: Lattice Surgery & Logical Gates | Month 30: Surface Codes

### Semester 2A: Error Correction | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | CNOT protocol theory, measurement-based implementation |
| **Afternoon** | 2.5 hours | Protocol verification, error analysis, practice problems |
| **Evening** | 1.5 hours | Computational lab: CNOT simulation and verification |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 823, you will be able to:

1. **Derive the lattice surgery CNOT** from measurement-based quantum computation principles
2. **Execute the ZZ-merge/XX-merge protocol** step-by-step with outcome tracking
3. **Calculate Pauli frame corrections** for all measurement outcome combinations
4. **Analyze the fault-tolerance** of the lattice surgery CNOT
5. **Compare resource costs** with alternative CNOT implementations
6. **Simulate the complete CNOT protocol** and verify correctness

---

## 1. Introduction: The CNOT Challenge

### Why CNOT Matters

The **Controlled-NOT (CNOT)** gate is fundamental to quantum computation:
- Creates entanglement from product states
- Together with single-qubit gates, forms a universal gate set
- Essential for syndrome extraction, state preparation, and most quantum algorithms

$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

### The Surface Code CNOT Problem

**Challenge:** Surface code patches don't naturally support transversal CNOT.

For two patches A (control) and B (target):
- Physical CNOT on corresponding qubits doesn't give logical CNOT
- Boundary types and stabilizer structure prevent simple transversal approach

**Solution:** Use **lattice surgery** - a sequence of merge and split operations that collectively implement CNOT.

$$\boxed{\text{Lattice Surgery CNOT} = \text{ZZ Merge} + \text{XX Merge} + \text{Pauli Corrections}}$$

---

## 2. CNOT from Joint Measurements

### Measurement-Based Decomposition

The CNOT gate can be decomposed using only joint Pauli measurements:

$$\text{CNOT}_{C \to T} = H_T \cdot CZ_{CT} \cdot H_T$$

where CZ (controlled-Z) is symmetric and can be implemented via ZZ measurement.

**Alternative decomposition using measurements:**

Consider the circuit identity:
$$\text{CNOT}_{C \to T} \equiv (\text{Measure } Z_C Z_A) \cdot (\text{Measure } X_A X_T) \cdot (\text{Corrections})$$

where A is an ancilla qubit initialized in $|+\rangle$.

### Proof of Equivalence

**Setup:**
- Control qubit C in state $|\psi_C\rangle = \alpha|0\rangle + \beta|1\rangle$
- Target qubit T in state $|\phi_T\rangle = \gamma|0\rangle + \delta|1\rangle$
- Ancilla A initialized in $|+\rangle$

**Initial state:**
$$|\Psi_0\rangle = |\psi_C\rangle \otimes |+\rangle_A \otimes |\phi_T\rangle$$

**Step 1: Measure $Z_C Z_A$**

Outcome $m_1 \in \{+1, -1\}$ projects:
$$|\Psi_1\rangle \propto \begin{cases}
\alpha|0,+\rangle + \beta|1,-\rangle & m_1 = +1 \\
\alpha|0,-\rangle + \beta|1,+\rangle & m_1 = -1
\end{cases} \otimes |\phi_T\rangle$$

**Step 2: Measure $X_A X_T$**

This correlates the ancilla's X value with target's X value.

**Step 3: Apply corrections**

Based on outcomes $(m_1, m_2)$, apply Pauli corrections to recover CNOT.

$$\boxed{\text{CNOT} = X_T^{(1-m_2)/2} \cdot Z_C^{(1-m_1)/2} \cdot Z_T^{(1-m_1)(1-m_2)/4} \cdot (M_{XX}) \cdot (M_{ZZ}) \cdot |+\rangle_A}$$

---

## 3. The Lattice Surgery CNOT Protocol

### Complete Protocol Specification

**Prerequisites:**
- Control patch C with logical qubit $|\psi_C\rangle$
- Target patch T with logical qubit $|\phi_T\rangle$
- Ancilla patch A initialized in $|+_L\rangle$

**Layout:**

```
    Control (C)          Ancilla (A)          Target (T)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│              │    │              │    │              │
│   |ψ_C⟩      │~~~~│   |+⟩        │~~~~│   |φ_T⟩      │
│              │ ZZ │              │ XX │              │
└──────────────┘    └──────────────┘    └──────────────┘
   Rough edge        Both edges        Smooth edge
```

### Step-by-Step Execution

**Step 1: Initialize ancilla**
$$|A\rangle = |+_L\rangle$$

Time: $d$ cycles

**Step 2: ZZ merge between C and A**

Connect rough boundaries of C and A:
- Add merge-zone qubits in $|0\rangle$
- Measure joint Z stabilizers for $d$ rounds
- Record outcome $m_1 = Z_C Z_A$

Time: $d$ cycles

**Step 3: XX merge between A and T**

Connect smooth boundaries of A and T:
- Add merge-zone qubits in $|+\rangle$
- Measure joint X stabilizers for $d$ rounds
- Record outcome $m_2 = X_A X_T$

Time: $d$ cycles

**Step 4: Apply Pauli corrections**

Based on $(m_1, m_2)$:

| $m_1$ | $m_2$ | Correction |
|-------|-------|------------|
| +1 | +1 | None |
| +1 | -1 | $X_T$ |
| -1 | +1 | $Z_C$ |
| -1 | -1 | $Z_C X_T$ |

Time: 0 cycles (Pauli frame tracking)

$$\boxed{t_{\text{CNOT}} = 3d \times \tau_{\text{cycle}}}$$

---

## 4. Detailed State Evolution

### Tracking the Quantum State

Let's trace through the full state evolution for arbitrary input.

**Initial state:**
$$|\Psi_0\rangle = (\alpha|0\rangle_C + \beta|1\rangle_C) \otimes |+\rangle_A \otimes (\gamma|0\rangle_T + \delta|1\rangle_T)$$

Expanding:
$$|\Psi_0\rangle = \frac{1}{\sqrt{2}} \sum_{c,a,t} \alpha^{1-c}\beta^c \gamma^{1-t}\delta^t |c,a,t\rangle$$

**After ZZ measurement on C,A:**

The $Z_C Z_A$ measurement projects onto states with definite $z_C \oplus z_A$:

For $m_1 = +1$ (meaning $z_C = z_A$):
$$|\Psi_1^{(+)}\rangle = \frac{1}{\sqrt{2}}(\alpha|0,0\rangle + \beta|1,1\rangle)_{CA} \otimes |\phi\rangle_T$$

For $m_1 = -1$ (meaning $z_C \neq z_A$):
$$|\Psi_1^{(-)}\rangle = \frac{1}{\sqrt{2}}(\alpha|0,1\rangle + \beta|1,0\rangle)_{CA} \otimes |\phi\rangle_T$$

**After XX measurement on A,T:**

Convert to X basis for analysis. For $m_1 = +1$:
$$|0,0\rangle_{CA} = \frac{1}{2}(|+,+\rangle + |+,-\rangle + |-,+\rangle + |-,-\rangle)$$
$$|1,1\rangle_{CA} = \frac{1}{2}(|+,+\rangle - |+,-\rangle - |-,+\rangle + |-,-\rangle)$$

So:
$$|\Psi_1^{(+)}\rangle_{CA} = \frac{1}{\sqrt{2}}[(\alpha+\beta)|+,+\rangle + (\alpha-\beta)|-,-\rangle]_{CA}$$

After $X_A X_T$ measurement with outcome $m_2$, the target qubit gets correlated with the ancilla's X value.

**Final state (before correction):**

Through careful tracking:
$$|\Psi_{\text{final}}\rangle = Z_C^{(1-m_1)/2} X_T^{(1-m_2)/2} \cdot \text{CNOT}|\psi_C\rangle|\phi_T\rangle$$

$$\boxed{\text{CNOT}|\psi_C\rangle|\phi_T\rangle = Z_C^{(1-m_1)/2} X_T^{(1-m_2)/2} |\Psi_{\text{measured}}\rangle}$$

---

## 5. Pauli Frame Tracking

### Classical Tracking vs. Physical Correction

In practice, we don't physically apply Pauli corrections after each gate. Instead:

**Pauli Frame:** Track accumulated Pauli operations classically
- Maintain a "frame" $(P_C, P_T)$ for each qubit
- Update frame based on measurement outcomes
- Apply physical corrections only at final readout

### Frame Update Rules

For CNOT with outcomes $(m_1, m_2)$:
$$P_C \leftarrow P_C \cdot Z^{(1-m_1)/2}$$
$$P_T \leftarrow P_T \cdot X^{(1-m_2)/2}$$

**Note:** CNOT also propagates existing Pauli frame:
- $X_C$ before CNOT → $X_C X_T$ after
- $Z_T$ before CNOT → $Z_C Z_T$ after

$$\boxed{\text{CNOT}: X_C \mapsto X_C X_T, \quad Z_T \mapsto Z_C Z_T}$$

### Complete Frame Tracking Example

**Initial frame:** $(I, I)$ (no corrections needed)

**After CNOT with $(m_1=-1, m_2=+1)$:**
- Frame: $(Z, I)$

**After second CNOT with $(m_1=+1, m_2=-1)$:**
- $Z$ on first qubit propagates: becomes $Z \otimes Z$
- New correction: $X$ on second qubit
- Frame: $(Z, ZX) = (Z, -iY)$

---

## 6. Fault-Tolerance Analysis

### Error Sources During CNOT

**1. Errors on data qubits**
- Handled by ongoing syndrome measurement
- Distance-$d$ code corrects up to $\lfloor(d-1)/2\rfloor$ errors

**2. Errors on merge-zone qubits**
- These qubits exist only during merge
- Errors detected by merge stabilizers
- Time-limited exposure reduces error accumulation

**3. Measurement errors**
- $d$ rounds of measurement provide redundancy
- Minimum-weight perfect matching decodes across space-time

### Logical Error Rate

The logical error rate for a single CNOT scales as:

$$P_L^{\text{CNOT}} \sim 3 \times \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

The factor of 3 comes from three stages (init, ZZ merge, XX merge).

$$\boxed{P_L^{\text{CNOT}} = O(d) \times P_L^{\text{memory}}}$$

### Distance Requirements

For target logical error rate $\epsilon$:

$$d \geq 2 \log\left(\frac{3}{\epsilon}\right) / \log\left(\frac{p_{\text{th}}}{p}\right) + 1$$

**Example:** For $p = 10^{-3}$, $p_{\text{th}} = 10^{-2}$, $\epsilon = 10^{-12}$:
$$d \geq 2 \times 12 \times \ln(10) / \ln(10) + 1 = 25$$

---

## 7. Worked Examples

### Example 1: CNOT on Computational Basis States

**Problem:** Apply the lattice surgery CNOT to $|1\rangle_C |0\rangle_T$. Trace through all steps.

**Solution:**

**Initial:**
$$|\Psi_0\rangle = |1\rangle_C |+\rangle_A |0\rangle_T$$

**ZZ merge (C-A):**

$Z_C|1\rangle = -1$, $Z_A|+\rangle = ?$ (undefined, so random)

For $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:

$$|1\rangle_C |+\rangle_A = \frac{1}{\sqrt{2}}(|1,0\rangle + |1,1\rangle)$$

$Z_CZ_A$ eigenvalues:
- $|1,0\rangle$: $(-1)(+1) = -1$
- $|1,1\rangle$: $(-1)(-1) = +1$

Outcomes:
- $m_1 = +1$ (prob 1/2): State → $|1,1\rangle$
- $m_1 = -1$ (prob 1/2): State → $|1,0\rangle$

**XX merge (A-T):**

For $m_1 = +1$, state is $|1,1\rangle_C,A |0\rangle_T$

Convert to X basis:
- $|1\rangle = \frac{1}{\sqrt{2}}(|+\rangle - |-\rangle)$
- $|0\rangle = \frac{1}{\sqrt{2}}(|+\rangle + |-\rangle)$

$X_A X_T$ measurement:
- $m_2 = +1$: projects to same X parity
- $m_2 = -1$: projects to opposite X parity

**Corrections and final state:**

For $(m_1=+1, m_2=+1)$: No correction, final = $|1\rangle_C|1\rangle_T$ ✓
For $(m_1=+1, m_2=-1)$: Apply $X_T$, final = $|1\rangle_C|1\rangle_T$ ✓
(Similar analysis for $m_1=-1$ cases)

$$\boxed{|1,0\rangle \xrightarrow{\text{CNOT}} |1,1\rangle \text{ (verified)}}$$

---

### Example 2: Creating Bell State with CNOT

**Problem:** Use the lattice surgery CNOT on $|+\rangle_C |0\rangle_T$ to create a Bell state.

**Solution:**

**Initial:**
$$|\Psi_0\rangle = |+\rangle_C |+\rangle_A |0\rangle_T = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)_C |+\rangle_A |0\rangle_T$$

**ZZ merge:**

Full state: $\frac{1}{2}(|0,0\rangle + |0,1\rangle + |1,0\rangle + |1,1\rangle)_{C,A} |0\rangle_T$

$Z_CZ_A$ values:
- $|0,0\rangle$: +1
- $|0,1\rangle$: -1
- $|1,0\rangle$: -1
- $|1,1\rangle$: +1

For $m_1 = +1$: $\frac{1}{\sqrt{2}}(|0,0\rangle + |1,1\rangle)_{C,A} |0\rangle_T$

**XX merge:**

This creates entanglement between control and target via the ancilla.

After proper tracking through XX measurement:

For $(m_1=+1, m_2=+1)$:
$$|\Psi_{\text{final}}\rangle = \frac{1}{\sqrt{2}}(|0,0\rangle + |1,1\rangle)_{C,T} = |\Phi^+\rangle$$

$$\boxed{|+\rangle|0\rangle \xrightarrow{\text{CNOT}} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle}$$

---

### Example 3: CNOT Resource Calculation

**Problem:** Calculate the total qubit-cycles for implementing 10 CNOTs in parallel on a distance-7 surface code.

**Solution:**

**Per CNOT:**
- Control patch: $2d^2 = 2(49) = 98$ qubits
- Target patch: 98 qubits
- Ancilla patch: 98 qubits
- Merge zones: $2d = 14$ qubits
- Total qubits: $98 \times 3 + 14 = 308$ qubits

**Time:**
- Ancilla prep: $d = 7$ cycles
- ZZ merge: $d = 7$ cycles
- XX merge: $d = 7$ cycles
- Total: $21$ cycles

**Per CNOT volume:**
$$V_{\text{CNOT}} = 308 \times 21 = 6,468 \text{ qubit-cycles}$$

**10 parallel CNOTs:**
$$V_{\text{total}} = 308 \times 10 \times 21 = 64,680 \text{ qubit-cycles}$$

$$\boxed{V_{10 \text{ CNOTs}} \approx 65,000 \text{ qubit-cycles (parallel)}}$$

Note: Serial execution would take $10 \times 21 = 210$ cycles.

---

## 8. Practice Problems

### Problem Set A: Direct Application

**A1.** What is the CNOT output for input $|0\rangle_C|1\rangle_T$? Verify using the lattice surgery protocol.

**A2.** If the ZZ merge gives $m_1 = -1$ and XX merge gives $m_2 = -1$, what Pauli correction is needed?

**A3.** How many measurement rounds are needed for a distance-11 CNOT?

---

### Problem Set B: Intermediate

**B1.** Derive the full state after ZZ merge for input state $|-\rangle_C|+\rangle_T|+\rangle_A$.

**B2.** A circuit has three consecutive CNOTs: $C_1 \to T_1$, $T_1 \to T_2$, $T_2 \to T_3$. Track the Pauli frame if all ZZ outcomes are $-1$ and all XX outcomes are $+1$.

**B3.** Calculate the logical error rate for a CNOT with $d=9$, physical error rate $p=5 \times 10^{-4}$, and threshold $p_{\text{th}} = 1\%$.

---

### Problem Set C: Challenging

**C1.** Design a lattice surgery protocol for the Toffoli gate (CCNOT) using only CNOT and T gates. Calculate the total space-time volume.

**C2.** Prove that the lattice surgery CNOT commutes with the standard CNOT up to Pauli corrections, regardless of measurement outcomes.

**C3.** Analyze how correlated errors across the merge zone affect the effective code distance during the CNOT operation.

---

## 9. Computational Lab: CNOT Protocol Simulation

```python
"""
Day 823 Computational Lab: Surface Code CNOT via Lattice Surgery
Complete simulation of the merge-based CNOT protocol

This lab implements the full lattice surgery CNOT and verifies
correctness through extensive testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)


def tensor(*args):
    """Compute tensor product of multiple matrices/vectors."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def measure_pauli(state, pauli_op):
    """
    Measure a Pauli observable on a state.

    Parameters:
    -----------
    state : ndarray
        State vector
    pauli_op : ndarray
        Pauli operator (eigenvalues +1, -1)

    Returns:
    --------
    outcome : int
        +1 or -1
    post_state : ndarray
        Post-measurement state
    """
    # Projectors
    P_plus = (np.eye(len(pauli_op)) + pauli_op) / 2
    P_minus = (np.eye(len(pauli_op)) - pauli_op) / 2

    # Probabilities
    p_plus = np.real(state.conj().T @ P_plus @ state)[0, 0]
    p_minus = np.real(state.conj().T @ P_minus @ state)[0, 0]

    # Sample
    outcome = np.random.choice([1, -1], p=[p_plus, p_minus])

    # Project
    if outcome == 1:
        post_state = P_plus @ state
    else:
        post_state = P_minus @ state

    post_state = post_state / np.linalg.norm(post_state)

    return outcome, post_state


def standard_cnot():
    """Return the standard CNOT matrix."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)


def lattice_surgery_cnot(control_state, target_state, verbose=True):
    """
    Implement CNOT via lattice surgery protocol.

    Protocol:
    1. Initialize ancilla in |+>
    2. ZZ merge between control and ancilla
    3. XX merge between ancilla and target
    4. Apply Pauli corrections

    Parameters:
    -----------
    control_state : ndarray
        Control qubit state
    target_state : ndarray
        Target qubit state
    verbose : bool
        Print detailed steps

    Returns:
    --------
    final_state : ndarray
        Two-qubit output state (control tensor target)
    outcomes : tuple
        (m1, m2) measurement outcomes
    """
    if verbose:
        print("\n" + "="*60)
        print("LATTICE SURGERY CNOT PROTOCOL")
        print("="*60)

    # Step 0: Initialize
    ancilla_state = ket_plus.copy()

    if verbose:
        print("\nStep 0: Initialize")
        print(f"  Control: {control_state.flatten()}")
        print(f"  Ancilla: {ancilla_state.flatten()} = |+⟩")
        print(f"  Target:  {target_state.flatten()}")

    # Full initial state: Control ⊗ Ancilla ⊗ Target
    psi = tensor(control_state, ancilla_state, target_state)

    if verbose:
        print(f"\nInitial 3-qubit state dimension: {psi.shape}")

    # Step 1: ZZ merge between Control (qubit 0) and Ancilla (qubit 1)
    # Measure Z_C ⊗ Z_A ⊗ I_T
    ZZ_CA = tensor(Z, Z, I)
    m1, psi = measure_pauli(psi, ZZ_CA)

    if verbose:
        print(f"\nStep 1: ZZ Merge (Control-Ancilla)")
        print(f"  Measure Z_C Z_A")
        print(f"  Outcome m1 = {'+1' if m1 == 1 else '-1'}")

    # Step 2: XX merge between Ancilla (qubit 1) and Target (qubit 2)
    # Measure I_C ⊗ X_A ⊗ X_T
    XX_AT = tensor(I, X, X)
    m2, psi = measure_pauli(psi, XX_AT)

    if verbose:
        print(f"\nStep 2: XX Merge (Ancilla-Target)")
        print(f"  Measure X_A X_T")
        print(f"  Outcome m2 = {'+1' if m2 == 1 else '-1'}")

    # Step 3: Trace out ancilla (qubit 1)
    # Reshape to trace out middle qubit
    psi_matrix = psi.reshape(2, 2, 2)  # C, A, T
    # Trace over ancilla
    final_CT = np.zeros((4, 1), dtype=complex)
    for a in range(2):
        component = psi_matrix[:, a, :].reshape(4, 1)
        final_CT += component

    # Renormalize
    final_CT = final_CT / np.linalg.norm(final_CT)

    if verbose:
        print(f"\nStep 3: Trace out ancilla")
        print(f"  State (before correction): {final_CT.flatten()}")

    # Step 4: Apply Pauli corrections
    correction_C = I
    correction_T = I

    if m1 == -1:
        correction_C = Z
    if m2 == -1:
        correction_T = X

    correction = tensor(correction_C, correction_T)
    final_state = correction @ final_CT

    if verbose:
        print(f"\nStep 4: Apply corrections")
        print(f"  m1={'+1' if m1==1 else '-1'} → Z_C correction: {'Yes' if m1==-1 else 'No'}")
        print(f"  m2={'+1' if m2==1 else '-1'} → X_T correction: {'Yes' if m2==-1 else 'No'}")
        print(f"\nFinal state: {final_state.flatten()}")

    return final_state, (m1, m2)


def verify_cnot_correctness(num_tests=100):
    """
    Verify lattice surgery CNOT against standard CNOT for random inputs.

    Parameters:
    -----------
    num_tests : int
        Number of random states to test

    Returns:
    --------
    success_rate : float
        Fraction of tests that passed
    """
    print("\n" + "="*60)
    print("CNOT VERIFICATION")
    print("="*60)

    cnot_matrix = standard_cnot()
    successes = 0
    max_error = 0

    for i in range(num_tests):
        # Random control state
        theta_c = np.random.uniform(0, np.pi)
        phi_c = np.random.uniform(0, 2*np.pi)
        control = np.cos(theta_c/2) * ket_0 + np.exp(1j*phi_c) * np.sin(theta_c/2) * ket_1

        # Random target state
        theta_t = np.random.uniform(0, np.pi)
        phi_t = np.random.uniform(0, 2*np.pi)
        target = np.cos(theta_t/2) * ket_0 + np.exp(1j*phi_t) * np.sin(theta_t/2) * ket_1

        # Expected output
        input_state = tensor(control, target)
        expected = cnot_matrix @ input_state

        # Lattice surgery output
        actual, _ = lattice_surgery_cnot(control, target, verbose=False)

        # Compare (up to global phase)
        # Find phase that minimizes difference
        phases = np.linspace(0, 2*np.pi, 100)
        min_diff = float('inf')
        for phase in phases:
            diff = np.linalg.norm(actual - np.exp(1j*phase) * expected)
            min_diff = min(min_diff, diff)

        max_error = max(max_error, min_diff)

        if min_diff < 1e-10:
            successes += 1

    success_rate = successes / num_tests
    print(f"\nResults: {successes}/{num_tests} tests passed")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Maximum error: {max_error:.2e}")

    return success_rate


def test_computational_basis():
    """Test CNOT on all computational basis states."""
    print("\n" + "="*60)
    print("COMPUTATIONAL BASIS TESTS")
    print("="*60)

    basis_states = [ket_0, ket_1]
    labels = ['|0⟩', '|1⟩']

    cnot_matrix = standard_cnot()

    print("\nInput → Expected → Actual (with outcomes)")
    print("-" * 50)

    all_correct = True

    for i, (c, lc) in enumerate(zip(basis_states, labels)):
        for j, (t, lt) in enumerate(zip(basis_states, labels)):
            # Expected
            input_state = tensor(c, t)
            expected = cnot_matrix @ input_state

            # Actual (run multiple times to see different outcomes)
            actual, (m1, m2) = lattice_surgery_cnot(c, t, verbose=False)

            # Check correctness
            phases = np.linspace(0, 2*np.pi, 100)
            min_diff = float('inf')
            for phase in phases:
                diff = np.linalg.norm(actual - np.exp(1j*phase) * expected)
                min_diff = min(min_diff, diff)

            correct = min_diff < 1e-10
            all_correct = all_correct and correct

            # Determine output state label
            if np.abs(actual[0]) > 0.9:
                out_label = "|00⟩"
            elif np.abs(actual[1]) > 0.9:
                out_label = "|01⟩"
            elif np.abs(actual[2]) > 0.9:
                out_label = "|10⟩"
            else:
                out_label = "|11⟩"

            status = "✓" if correct else "✗"
            print(f"  {lc}⊗{lt} → {out_label}  (m1={'+1' if m1==1 else '-1'}, m2={'+1' if m2==1 else '-1'}) {status}")

    print("-" * 50)
    print(f"All computational basis tests: {'PASSED' if all_correct else 'FAILED'}")


def analyze_outcome_distribution():
    """Analyze the distribution of measurement outcomes."""
    print("\n" + "="*60)
    print("MEASUREMENT OUTCOME DISTRIBUTION")
    print("="*60)

    # Test state: |+⟩ ⊗ |0⟩ (creates Bell state)
    control = ket_plus
    target = ket_0

    n_trials = 1000
    outcomes = {'(+1,+1)': 0, '(+1,-1)': 0, '(-1,+1)': 0, '(-1,-1)': 0}

    for _ in range(n_trials):
        _, (m1, m2) = lattice_surgery_cnot(control, target, verbose=False)
        key = f"({'+1' if m1==1 else '-1'},{'+1' if m2==1 else '-1'})"
        outcomes[key] += 1

    print(f"\nInput: |+⟩ ⊗ |0⟩")
    print(f"Trials: {n_trials}")
    print("\nOutcome distribution:")
    for key, count in outcomes.items():
        print(f"  {key}: {count/n_trials*100:.1f}%")

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(outcomes.keys(), [v/n_trials for v in outcomes.values()],
                  color=['blue', 'lightblue', 'red', 'salmon'])
    ax.set_ylabel('Probability')
    ax.set_xlabel('Outcome (m1, m2)')
    ax.set_title('Lattice Surgery CNOT Measurement Outcomes\nInput: |+⟩⊗|0⟩')
    ax.set_ylim(0, 0.5)
    ax.axhline(y=0.25, color='gray', linestyle='--', label='Uniform (0.25)')
    ax.legend()

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('cnot_outcomes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nOutcome distribution saved to 'cnot_outcomes.png'")


def visualize_cnot_protocol():
    """Create a visual representation of the CNOT protocol."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    steps = [
        ("Initial State", "C: |ψ⟩\nA: |+⟩\nT: |φ⟩"),
        ("ZZ Merge (C-A)", "Measure Z_C Z_A\nOutcome: m₁"),
        ("XX Merge (A-T)", "Measure X_A X_T\nOutcome: m₂"),
        ("Final + Correction", "Apply Z_C^(m₁) X_T^(m₂)\nResult: CNOT|ψφ⟩")
    ]

    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    for idx, (ax, (title, desc)) in enumerate(zip(axes, steps)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        # Background
        ax.add_patch(plt.Rectangle((0.5, 2), 9, 6,
                                    facecolor=colors[idx], edgecolor='black', lw=2))

        # Title
        ax.text(5, 9, f"Step {idx+1}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(5, 8.5, title, ha='center', va='bottom', fontsize=11)

        # Draw patches
        if idx == 0:  # Initial
            ax.add_patch(plt.Rectangle((1, 5), 2, 2, facecolor='blue', alpha=0.5))
            ax.add_patch(plt.Rectangle((4, 5), 2, 2, facecolor='green', alpha=0.5))
            ax.add_patch(plt.Rectangle((7, 5), 2, 2, facecolor='red', alpha=0.5))
            ax.text(2, 6, 'C', ha='center', va='center', fontsize=14, color='white')
            ax.text(5, 6, 'A', ha='center', va='center', fontsize=14, color='white')
            ax.text(8, 6, 'T', ha='center', va='center', fontsize=14, color='white')
        elif idx == 1:  # ZZ merge
            ax.add_patch(plt.Rectangle((1, 5), 5, 2, facecolor='purple', alpha=0.5))
            ax.add_patch(plt.Rectangle((7, 5), 2, 2, facecolor='red', alpha=0.5))
            ax.text(3.5, 6, 'C-A', ha='center', va='center', fontsize=14, color='white')
            ax.text(8, 6, 'T', ha='center', va='center', fontsize=14, color='white')
            ax.text(3.5, 4.5, 'ZZ', ha='center', fontsize=12, color='purple')
        elif idx == 2:  # XX merge
            ax.add_patch(plt.Rectangle((1, 5), 2, 2, facecolor='blue', alpha=0.5))
            ax.add_patch(plt.Rectangle((4, 5), 5, 2, facecolor='orange', alpha=0.5))
            ax.text(2, 6, 'C', ha='center', va='center', fontsize=14, color='white')
            ax.text(6.5, 6, 'A-T', ha='center', va='center', fontsize=14, color='white')
            ax.text(6.5, 4.5, 'XX', ha='center', fontsize=12, color='orange')
        else:  # Final
            ax.add_patch(plt.Rectangle((1, 5), 2, 2, facecolor='blue', alpha=0.5))
            ax.add_patch(plt.Rectangle((7, 5), 2, 2, facecolor='red', alpha=0.5))
            ax.text(2, 6, 'C\'', ha='center', va='center', fontsize=14, color='white')
            ax.text(8, 6, 'T\'', ha='center', va='center', fontsize=14, color='white')
            ax.annotate('', xy=(6.8, 6), xytext=(3.2, 6),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=2))

        # Description
        ax.text(5, 3, desc, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Lattice Surgery CNOT Protocol', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cnot_protocol.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nProtocol visualization saved to 'cnot_protocol.png'")


def resource_analysis():
    """Analyze resources for CNOT as a function of code distance."""
    print("\n" + "="*60)
    print("CNOT RESOURCE ANALYSIS")
    print("="*60)

    distances = np.arange(3, 21, 2)

    # Per CNOT resources
    qubits_per_patch = 2 * distances**2  # Data + ancilla qubits
    total_qubits = 3 * qubits_per_patch + 2 * distances  # 3 patches + merge zones
    time_cycles = 3 * distances  # Init + ZZ + XX
    volume = total_qubits * time_cycles

    # Logical error rate (simplified model)
    p_phys = 1e-3
    p_th = 1e-2
    p_logical = 3 * (p_phys / p_th) ** ((distances + 1) / 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Qubits
    axes[0, 0].plot(distances, total_qubits, 'bo-', markersize=8)
    axes[0, 0].set_xlabel('Code Distance d')
    axes[0, 0].set_ylabel('Total Physical Qubits')
    axes[0, 0].set_title('Qubit Count per CNOT')
    axes[0, 0].grid(True, alpha=0.3)

    # Time
    axes[0, 1].plot(distances, time_cycles, 'rs-', markersize=8)
    axes[0, 1].set_xlabel('Code Distance d')
    axes[0, 1].set_ylabel('Syndrome Cycles')
    axes[0, 1].set_title('Time per CNOT')
    axes[0, 1].grid(True, alpha=0.3)

    # Volume
    axes[1, 0].semilogy(distances, volume, 'g^-', markersize=8)
    axes[1, 0].set_xlabel('Code Distance d')
    axes[1, 0].set_ylabel('Qubit-Cycles')
    axes[1, 0].set_title('Space-Time Volume per CNOT')
    axes[1, 0].grid(True, alpha=0.3)

    # Error rate
    axes[1, 1].semilogy(distances, p_logical, 'mp-', markersize=8)
    axes[1, 1].set_xlabel('Code Distance d')
    axes[1, 1].set_ylabel('Logical Error Rate')
    axes[1, 1].set_title(f'Logical Error Rate (p_phys={p_phys})')
    axes[1, 1].axhline(y=1e-12, color='red', linestyle='--', label='Target: 10⁻¹²')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Lattice Surgery CNOT Resource Scaling', fontsize=14)
    plt.tight_layout()
    plt.savefig('cnot_resources.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print table
    print("\nResource Table:")
    print("-" * 60)
    print(f"{'d':>3} | {'Qubits':>8} | {'Cycles':>7} | {'Volume':>10} | {'P_L':>12}")
    print("-" * 60)
    for d, q, t, v, p in zip(distances, total_qubits, time_cycles, volume, p_logical):
        print(f"{d:3d} | {q:8d} | {t:7d} | {v:10d} | {p:12.2e}")
    print("-" * 60)

    print("\nResource analysis saved to 'cnot_resources.png'")


def main():
    """Run all Day 823 demonstrations."""
    print("Day 823: Surface Code CNOT via Lattice Surgery")
    print("="*60)

    # Detailed example
    print("\n--- Example: CNOT on |1⟩⊗|0⟩ ---")
    lattice_surgery_cnot(ket_1, ket_0, verbose=True)

    # Computational basis tests
    test_computational_basis()

    # Verification
    verify_cnot_correctness(num_tests=100)

    # Outcome distribution
    analyze_outcome_distribution()

    # Protocol visualization
    visualize_cnot_protocol()

    # Resource analysis
    resource_analysis()

    print("\n" + "="*60)
    print("Day 823 Computational Lab Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 10. Summary

### Key Formulas Table

| Concept | Formula/Expression |
|---------|-------------------|
| CNOT decomposition | $\text{CNOT} = M_{XX} \cdot M_{ZZ} \cdot \text{corrections}$ |
| ZZ merge outcome | $m_1 = Z_C Z_A \in \{+1, -1\}$ |
| XX merge outcome | $m_2 = X_A X_T \in \{+1, -1\}$ |
| Z correction | $Z_C^{(1-m_1)/2}$ |
| X correction | $X_T^{(1-m_2)/2}$ |
| CNOT time | $t = 3d \cdot \tau_{\text{cycle}}$ |
| Qubits per CNOT | $N \approx 6d^2 + 2d$ |
| Logical error rate | $P_L \sim 3(p/p_{\text{th}})^{(d+1)/2}$ |
| Pauli propagation | $X_C \mapsto X_C X_T$, $Z_T \mapsto Z_C Z_T$ |

### Key Takeaways

1. **Lattice surgery CNOT** uses ZZ and XX merge operations with an ancilla patch
2. **Measurement outcomes** are random but tracked classically for Pauli corrections
3. **Pauli frame tracking** avoids physical correction gates until final readout
4. **Fault-tolerance** is maintained through repeated syndrome measurement during merges
5. **Resource scaling** is $O(d^2)$ qubits and $O(d)$ time cycles per CNOT
6. **Logical error rate** scales exponentially with distance, enabling arbitrarily high fidelity

---

## 11. Daily Checklist

- [ ] I can derive CNOT from joint ZZ and XX measurements
- [ ] I understand the role of the ancilla patch in the protocol
- [ ] I can track Pauli corrections through multiple gates
- [ ] I can calculate the resource overhead for lattice surgery CNOT
- [ ] I understand how errors are handled during the protocol
- [ ] I completed the computational lab and verified CNOT correctness

---

## 12. Preview: Day 824

Tomorrow we explore **Multi-Patch Architectures**:

- Organizing patches for parallel gate execution
- Routing and scheduling in 2D patch arrays
- Defect-based operations for more compact designs
- Trade-offs between parallelism and qubit overhead

Multi-patch layouts are essential for scaling lattice surgery to useful quantum algorithms.

---

*"The lattice surgery CNOT demonstrates that quantum computation doesn't require long-range interactions - local operations, carefully orchestrated, can implement any quantum gate."*
