# Day 825: T-Gate Injection & Magic State Integration

## Week 118: Lattice Surgery & Logical Gates | Month 30: Surface Codes

### Semester 2A: Error Correction | Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | T gate theory, magic states, distillation protocols |
| **Afternoon** | 2.5 hours | Factory design, resource estimation, practice problems |
| **Evening** | 1.5 hours | Computational lab: T-gate injection simulation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 825, you will be able to:

1. **Explain why T gates are special** in fault-tolerant quantum computing
2. **Define magic states** and their role in implementing non-Clifford gates
3. **Describe the gate teleportation protocol** for T-gate injection
4. **Analyze magic state distillation** protocols and their overhead
5. **Design T-gate factories** integrated with surface code architectures
6. **Estimate T-gate resources** for quantum algorithms

---

## 1. Introduction: The T Gate Challenge

### The Clifford Hierarchy

Quantum gates are organized into a hierarchy based on their relationship to Pauli operators:

**Level 1 (Pauli):** $\{I, X, Y, Z\}$
**Level 2 (Clifford):** Gates that map Paulis to Paulis: $\{H, S, \text{CNOT}, ...\}$
**Level 3:** Gates that map Cliffords to Cliffords: $\{T, ...\}$

The **T gate** (also called $\pi/8$ gate):

$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = e^{i\pi/8} R_z(\pi/4)$$

$$\boxed{T|0\rangle = |0\rangle, \quad T|1\rangle = e^{i\pi/4}|1\rangle}$$

### Why T is Hard

**Easmon-Knill Theorem:** Clifford circuits can be efficiently simulated classically.

**Implication:** We need non-Clifford gates (like T) for quantum advantage.

**Problem:** The surface code admits:
- Transversal Clifford gates: $H$, $S$, CNOT
- NO transversal T gate

**Why no transversal T?** The T gate doesn't preserve the code space structure in a way that allows qubit-by-qubit application.

$$\boxed{\text{T gate requires special injection protocol}}$$

---

## 2. Magic States

### Definition

A **magic state** is a quantum state that, when combined with Clifford operations, enables non-Clifford gates.

**T magic state:**
$$|T\rangle = T|+\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + e^{i\pi/4}|1\rangle\right)$$

**Alternative form:**
$$|T\rangle = \cos\frac{\pi}{8}|0\rangle + e^{i\pi/4}\sin\frac{\pi}{8}|1\rangle$$

On the Bloch sphere: $|T\rangle$ is at $(\theta, \phi) = (\pi/4, \pi/4)$

### Magic State Properties

**Key property:** Given $|T\rangle$ and Clifford gates, we can implement T on any state.

$$\boxed{T|\psi\rangle = (\text{Clifford operations}) + |T\rangle \text{ (consumed)}}$$

### Other Magic States

**$|H\rangle$ state** (for Hadamard-type gates):
$$|H\rangle = \cos\frac{\pi}{8}|0\rangle + \sin\frac{\pi}{8}|1\rangle$$

**$|A\rangle$ state** (for arbitrary angle rotations):
Various states enabling $R_z(\theta)$ for specific $\theta$

---

## 3. T-Gate Teleportation Protocol

### The Protocol

**Goal:** Apply T gate to $|\psi\rangle$ using $|T\rangle$ and Clifford operations.

**Circuit:**

```
|ψ⟩ ─────●───── M_X ═══╗
         │             ║
|T⟩ ─────X───────────S^m──→ T|ψ⟩
```

**Steps:**

1. **Prepare:** $|T\rangle$ magic state (in surface code patch)
2. **CNOT:** Control = $|\psi\rangle$, Target = $|T\rangle$
3. **Measure:** $|\psi\rangle$ in X basis, outcome $m \in \{0, 1\}$
4. **Correct:** Apply $S^m$ to output if $m = 1$

### Mathematical Verification

**Initial state:**
$$|\Psi_0\rangle = |\psi\rangle \otimes |T\rangle = (\alpha|0\rangle + \beta|1\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

**After CNOT:**
$$|\Psi_1\rangle = \frac{1}{\sqrt{2}}\left[\alpha|0\rangle(|0\rangle + e^{i\pi/4}|1\rangle) + \beta|1\rangle(|1\rangle + e^{i\pi/4}|0\rangle)\right]$$

Rewrite in X basis for first qubit ($|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$):

$$|\Psi_1\rangle = \frac{1}{2}|+\rangle(\alpha + \beta e^{i\pi/4})|0\rangle + ... (\text{terms for } |−\rangle)$$

**After X measurement:**

- Outcome 0 (measured $|+\rangle$): Output is $\alpha|0\rangle + \beta e^{i\pi/4}|1\rangle = T|\psi\rangle$ ✓
- Outcome 1 (measured $|-\rangle$): Output is $\alpha|0\rangle - \beta e^{i\pi/4}|1\rangle = ZT|\psi\rangle$

**Correction:** $S \cdot ZT = S Z T = e^{-i\pi/4} T$ (up to global phase)

Actually: $SZT|\psi\rangle = T|\psi\rangle$ (the S gate compensates for the Z error).

$$\boxed{T|\psi\rangle = S^m \cdot (\text{CNOT} \cdot M_X) \cdot |T\rangle}$$

---

## 4. Lattice Surgery Implementation

### T-Gate via Merge Operations

In surface code language:

**Step 1:** Prepare magic state patch in $|T_L\rangle$

**Step 2:** ZZ merge between data patch (containing $|\psi_L\rangle$) and magic patch

This is equivalent to the CNOT in the teleportation protocol.

**Step 3:** XX measurement on data patch (logical X measurement)

**Step 4:** Apply $S_L$ correction if measurement outcome is $-1$

### Protocol Timing

| Step | Operation | Time |
|------|-----------|------|
| 1 | Magic state prep | Factory time |
| 2 | ZZ merge | $d$ cycles |
| 3 | XX measurement | $d$ cycles |
| 4 | S correction | 0 (Pauli frame) |

**Total T gate time:** $2d$ cycles + magic state availability

$$\boxed{t_T = 2d + t_{\text{wait for magic state}}}$$

---

## 5. Magic State Distillation

### The Problem: Noisy Magic States

Physical preparation of $|T\rangle$ has errors:
$$\rho = (1-p)|T\rangle\langle T| + p \cdot \text{noise}$$

For fault-tolerant computation, we need $p_{\text{out}} \ll p_{\text{code}}$.

### Distillation Concept

Take many noisy magic states and produce fewer, higher-fidelity states:

$$n_{\text{in}} \text{ states at error } p_{\text{in}} \rightarrow n_{\text{out}} \text{ states at error } p_{\text{out}}$$

where $p_{\text{out}} \ll p_{\text{in}}$.

### The 15-to-1 Protocol

**Classic distillation protocol:**
- Input: 15 noisy $|T\rangle$ states with error $p$
- Output: 1 clean $|T\rangle$ state with error $\sim 35p^3$
- Success probability: $1 - O(p)$

**Error reduction:**
$$p_{\text{out}} \approx 35 p_{\text{in}}^3$$

**Example:** $p_{\text{in}} = 10^{-3}$ → $p_{\text{out}} \approx 3.5 \times 10^{-8}$

$$\boxed{\text{15-to-1: } p \rightarrow 35p^3}$$

### Multi-Level Distillation

For very low target error, cascade distillation levels:

**Level 1:** 15 physical → 1 at error $p_1 = 35p^3$
**Level 2:** 15 level-1 → 1 at error $p_2 = 35p_1^3$

After $k$ levels:
$$p_k \approx 35^{(3^k-1)/2} p^{3^k}$$

**Example:** Two levels from $p = 10^{-3}$:
- Level 1: $p_1 \approx 3.5 \times 10^{-8}$
- Level 2: $p_2 \approx 1.5 \times 10^{-21}$

### Resource Overhead

**States consumed per output:**
$$n_{\text{total}} = 15^k \text{ for } k \text{ levels}$$

**Qubits per distillation:**
- Each input state: $2d^2$ qubits
- 15 inputs: $30d^2$ qubits
- Distillation circuit: additional ancillas

**Time per distillation:**
$$t_{\text{distill}} \approx 10d \text{ cycles}$$

$$\boxed{\text{Cost per T: } \sim 15^k \times O(d^2) \text{ qubit-cycles}}$$

---

## 6. Magic State Factories

### Factory Architecture

A **magic state factory** continuously produces high-fidelity $|T\rangle$ states:

```
┌─────────────────────────────────────────────────┐
│              MAGIC STATE FACTORY                 │
│                                                  │
│  ┌─────┐  ┌─────┐  ┌─────┐       ┌─────────┐   │
│  │Raw T│──│Raw T│──│Raw T│──...──│ 15-to-1 │   │
│  └─────┘  └─────┘  └─────┘       │ Distill │   │
│      ×15 inputs                   └────┬────┘   │
│                                        │        │
│                                        ▼        │
│                             ┌─────────────────┐ │
│                             │  Output Queue   │ │
│                             │  (clean |T⟩s)   │ │
│                             └────────┬────────┘ │
└──────────────────────────────────────┼──────────┘
                                       │
                              To computation zone
```

### Factory Sizing

**Production rate:** $r_T$ magic states per time unit

**Demand rate:** $d_T$ = T gates per time unit in algorithm

**Requirement:** $r_T \geq d_T$ to avoid stalling

**Factory footprint:**
$$A_{\text{factory}} \approx 15 \times 2d^2 = 30d^2 \text{ qubits (per distillation unit)}$$

### Integration with Computation

**Layout option 1: Dedicated factory zone**

```
┌──────────────────────────────────────┐
│         COMPUTATION ZONE             │
│   ┌───┐ ┌───┐ ┌───┐ ┌───┐          │
│   │ Q │ │ Q │ │ Q │ │ Q │ ...      │
│   └───┘ └───┘ └───┘ └───┘          │
└───────────────┬──────────────────────┘
                │ Magic state channel
┌───────────────┴──────────────────────┐
│          FACTORY ZONE                │
│   ┌─────────┐ ┌─────────┐           │
│   │Factory 1│ │Factory 2│ ...       │
│   └─────────┘ └─────────┘           │
└──────────────────────────────────────┘
```

**Layout option 2: Distributed factories**

Factories interspersed with computation patches

---

## 7. Worked Examples

### Example 1: T-Gate Teleportation Verification

**Problem:** Verify that the T-gate teleportation protocol correctly implements T on $|+\rangle$.

**Solution:**

**Input:** $|\psi\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$

**Expected output:** $T|+\rangle = |T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$

**Protocol execution:**

1. Initial state: $|+\rangle \otimes |T\rangle$
   $$= \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle + e^{i\pi/4}|1\rangle)$$

2. After CNOT:
   $$= \frac{1}{2}(|00\rangle + e^{i\pi/4}|01\rangle + |11\rangle + e^{i\pi/4}|10\rangle)$$
   $$= \frac{1}{2}(|0\rangle(|0\rangle + e^{i\pi/4}|1\rangle) + |1\rangle(|1\rangle + e^{i\pi/4}|0\rangle))$$

3. Rewrite in X basis:
   $$= \frac{1}{2\sqrt{2}}|+\rangle(|0\rangle + e^{i\pi/4}|1\rangle + |1\rangle + e^{i\pi/4}|0\rangle)$$
   $$+ \frac{1}{2\sqrt{2}}|-\rangle(|0\rangle + e^{i\pi/4}|1\rangle - |1\rangle - e^{i\pi/4}|0\rangle)$$

4. X measurement outcome 0 ($|+\rangle$):
   $$|\text{out}\rangle \propto (1 + e^{i\pi/4})|0\rangle + (1 + e^{i\pi/4})|1\rangle \cdot e^{i\pi/4}$$

   Wait, let me redo this more carefully:

   After CNOT: $(|00\rangle + e^{i\pi/4}|01\rangle + |11\rangle + e^{i\pi/4}|10\rangle)/2$

   Group by second qubit:
   - Coefficient of $|x0\rangle$: $|0\rangle + e^{i\pi/4}|1\rangle$ (first qubit)
   - Coefficient of $|x1\rangle$: $e^{i\pi/4}|0\rangle + |1\rangle$ (first qubit)

   Measure first qubit in X basis:
   - $|+\rangle$ outcome: second qubit → $(|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2} = |T\rangle$ ✓
   - $|-\rangle$ outcome: second qubit → $S \cdot (...)= |T\rangle$ after correction ✓

$$\boxed{T|+\rangle = |T\rangle \text{ (verified via teleportation)}}$$

---

### Example 2: Distillation Resource Calculation

**Problem:** How many raw magic states are needed to produce one $|T\rangle$ with error $< 10^{-15}$, starting from raw error $p = 10^{-3}$?

**Solution:**

Using 15-to-1 distillation: $p_{\text{out}} = 35p_{\text{in}}^3$

**Level 1:**
$$p_1 = 35 \times (10^{-3})^3 = 3.5 \times 10^{-8}$$

**Level 2:**
$$p_2 = 35 \times (3.5 \times 10^{-8})^3 = 35 \times 4.3 \times 10^{-23} = 1.5 \times 10^{-21}$$

$p_2 < 10^{-15}$ ✓

**Resources:**
- Level 2 needs: 15 level-1 states
- Level 1 needs: 15 raw states each
- Total: $15 \times 15 = 225$ raw magic states

$$\boxed{225 \text{ raw states} \rightarrow 1 \text{ state at } p < 10^{-15}}$$

---

### Example 3: Factory Throughput

**Problem:** A quantum algorithm requires $10^6$ T gates. Design a factory that completes in under 1 hour, assuming 1 MHz cycle rate and $d = 7$.

**Solution:**

**Time budget:** 1 hour = $3600$ s = $3.6 \times 10^9$ cycles

**T gates needed:** $10^6$

**Minimum T rate:** $10^6 / (3.6 \times 10^9) = 2.8 \times 10^{-4}$ T/cycle

**Per distillation unit:**
- Time: $\sim 10d = 70$ cycles per output
- Rate: $1/70 = 0.014$ T/cycle per unit

**Number of units needed:**
$$n_{\text{units}} = \frac{2.8 \times 10^{-4}}{0.014} \approx 0.02$$

So just 1 unit is sufficient!

**But with parallelism in algorithm:**
If algorithm can use 100 T gates in parallel:
- Demand: $100 / 70 = 1.4$ T/cycle
- Units needed: $1.4 / 0.014 = 100$ distillation units

**Factory size:**
$$A = 100 \times 30d^2 = 100 \times 30 \times 49 = 147,000 \text{ qubits}$$

$$\boxed{\text{Factory: 100 units, ~150k qubits for 10}^6 \text{ T gates in 1 hour}}$$

---

## 8. Practice Problems

### Problem Set A: Direct Application

**A1.** Calculate $T|1\rangle$. Is it a magic state?

**A2.** In the T-gate teleportation protocol, what correction is needed if the X measurement gives $|-\rangle$?

**A3.** How many raw magic states are needed for 3-level 15-to-1 distillation?

---

### Problem Set B: Intermediate

**B1.** Derive the error rate formula $p_{\text{out}} = 35p_{\text{in}}^3$ for 15-to-1 distillation by counting failure modes.

**B2.** Compare the space-time volume of implementing $T^8 = Z$ directly versus via T-gate teleportation.

**B3.** Design a factory layout that can sustain 1000 T gates per second with $d = 11$ and 1 MHz cycles.

---

### Problem Set C: Challenging

**C1.** Analyze the "Litinski factory" design that achieves better than 15-to-1 overhead. What is the key insight?

**C2.** For an algorithm with non-uniform T-gate density (bursts of T gates followed by Clifford-only sections), design an adaptive factory that minimizes average qubit count.

**C3.** Prove that no Clifford circuit can transform $|0\rangle$ into $|T\rangle$, demonstrating the necessity of non-Clifford resources.

---

## 9. Computational Lab: T-Gate Injection Simulation

```python
"""
Day 825 Computational Lab: T-Gate Injection & Magic State Simulation
Simulating T-gate teleportation and analyzing distillation overhead

This lab implements the T-gate teleportation protocol and models
magic state distillation for fault-tolerant quantum computing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Quantum gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# States
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# Magic state
ket_T = T @ ket_plus


def tensor(*args):
    """Compute tensor product."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def cnot():
    """CNOT gate matrix."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)


def measure_x(state):
    """
    Measure single qubit in X basis.

    Returns outcome (0 or 1) and post-measurement state.
    """
    # Project onto |+⟩ or |−⟩
    proj_plus = ket_plus @ ket_plus.conj().T
    proj_minus = ket_minus @ ket_minus.conj().T

    p_plus = np.real((state.conj().T @ proj_plus @ state)[0, 0])
    p_minus = np.real((state.conj().T @ proj_minus @ state)[0, 0])

    outcome = np.random.choice([0, 1], p=[p_plus, p_minus])

    return outcome


def t_gate_teleportation(psi, verbose=True):
    """
    Implement T gate via teleportation protocol.

    Parameters:
    -----------
    psi : ndarray
        Input single-qubit state
    verbose : bool
        Print detailed steps

    Returns:
    --------
    output : ndarray
        T|psi⟩
    outcome : int
        Measurement outcome
    """
    if verbose:
        print("\n" + "="*50)
        print("T-GATE TELEPORTATION")
        print("="*50)
        print(f"\nInput state |ψ⟩: {psi.flatten()}")

    # Step 1: Prepare |T⟩
    magic = ket_T.copy()
    if verbose:
        print(f"|T⟩ magic state: {magic.flatten()}")

    # Step 2: Form joint state |ψ⟩ ⊗ |T⟩
    joint = tensor(psi, magic)
    if verbose:
        print(f"\nJoint state |ψ⟩⊗|T⟩ dimension: {joint.shape}")

    # Step 3: Apply CNOT (control = first qubit, target = second)
    CNOT = cnot()
    after_cnot = CNOT @ joint
    if verbose:
        print(f"After CNOT: {after_cnot.flatten()}")

    # Step 4: Measure first qubit in X basis
    # Reshape to separate qubits
    after_cnot_2q = after_cnot.reshape(2, 2)

    # Probability of measuring |+⟩ on first qubit
    # |+⟩ = (|0⟩ + |1⟩)/√2, so coefficient is (row0 + row1)/√2
    coeff_plus = (after_cnot_2q[0, :] + after_cnot_2q[1, :]) / np.sqrt(2)
    coeff_minus = (after_cnot_2q[0, :] - after_cnot_2q[1, :]) / np.sqrt(2)

    p_plus = np.real(np.sum(np.abs(coeff_plus)**2))
    p_minus = np.real(np.sum(np.abs(coeff_minus)**2))

    outcome = np.random.choice([0, 1], p=[p_plus, p_minus])

    if verbose:
        print(f"\nX measurement on first qubit:")
        print(f"  P(+) = {p_plus:.4f}, P(-) = {p_minus:.4f}")
        print(f"  Outcome: {'|+⟩' if outcome == 0 else '|−⟩'}")

    # Step 5: Get second qubit state and apply correction
    if outcome == 0:
        output = coeff_plus.reshape(2, 1)
    else:
        output = coeff_minus.reshape(2, 1)

    output = output / np.linalg.norm(output)

    # Step 6: Apply S correction if outcome was |−⟩
    if outcome == 1:
        output = S @ output
        if verbose:
            print(f"\nApplying S correction (outcome was |−⟩)")

    if verbose:
        print(f"\nOutput state: {output.flatten()}")

        # Verify against direct T application
        expected = T @ psi
        expected = expected / np.linalg.norm(expected)

        # Check equivalence up to global phase
        phase = (expected.conj().T @ output)[0, 0]
        phase = phase / np.abs(phase)  # Normalize to unit phase
        fidelity = np.abs((expected.conj().T @ output)[0, 0])**2

        print(f"\nVerification:")
        print(f"  Expected T|ψ⟩: {expected.flatten()}")
        print(f"  Fidelity: {fidelity:.6f}")
        print(f"  Global phase: {np.angle(phase):.4f} rad")

    return output, outcome


def simulate_noisy_magic_state(target_error):
    """
    Simulate a noisy magic state.

    With probability (1-p), return |T⟩.
    With probability p, return a random error state.
    """
    if np.random.random() > target_error:
        return ket_T.copy()
    else:
        # Depolarizing error: random Pauli
        pauli = np.random.choice([I, X, Y, Z])
        return pauli @ ket_T


def distillation_15_to_1(input_error, verbose=False):
    """
    Simulate 15-to-1 magic state distillation.

    Parameters:
    -----------
    input_error : float
        Error rate of input states
    verbose : bool
        Print details

    Returns:
    --------
    output_error : float
        Error rate of output state
    success : bool
        Whether distillation succeeded
    """
    # 15-to-1 distillation succeeds if at most 1 input is faulty
    # (simplified model)

    n_faulty = np.random.binomial(15, input_error)

    if n_faulty == 0:
        output_error = 35 * input_error**3
        success = True
    elif n_faulty == 1:
        # Detected and corrected
        output_error = 35 * input_error**3
        success = True
    else:
        # Distillation fails (too many errors)
        success = False
        output_error = 1.0  # Undefined/bad state

    if verbose:
        print(f"Distillation: {n_faulty}/15 faulty inputs")
        print(f"  Success: {success}, Output error: {output_error:.2e}")

    return output_error, success


def analyze_distillation_overhead():
    """Analyze distillation overhead for various target error rates."""
    print("\n" + "="*60)
    print("DISTILLATION OVERHEAD ANALYSIS")
    print("="*60)

    raw_errors = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    target_errors = [1e-6, 1e-9, 1e-12, 1e-15]

    results = {}

    print("\nRaw Magic States Required:")
    print("-" * 60)
    header = "Raw Error   | " + " | ".join([f"Target {t:.0e}" for t in target_errors])
    print(header)
    print("-" * 60)

    for raw_p in raw_errors:
        row = f"{raw_p:.0e}       |"
        for target in target_errors:
            # Calculate levels needed
            p = raw_p
            levels = 0
            while p > target and levels < 10:
                p = 35 * p**3
                levels += 1

            if levels < 10:
                n_raw = 15**levels
                row += f" {n_raw:>12} |"
                results[(raw_p, target)] = (levels, n_raw)
            else:
                row += f" {'N/A':>12} |"

        print(row)

    print("-" * 60)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error reduction per level
    ax = axes[0]
    levels = range(0, 5)
    for raw_p in [1e-2, 1e-3, 1e-4]:
        errors = [raw_p]
        for _ in range(4):
            errors.append(35 * errors[-1]**3)
        ax.semilogy(levels, errors, 'o-', label=f'Raw p = {raw_p:.0e}')

    ax.set_xlabel('Distillation Level')
    ax.set_ylabel('Output Error Rate')
    ax.set_title('Error Reduction per Distillation Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(levels)

    # Plot 2: Raw states required
    ax = axes[1]
    for raw_p in [1e-2, 1e-3, 1e-4]:
        targets = []
        n_raws = []
        for target in np.logspace(-15, -6, 20):
            p = raw_p
            levels = 0
            while p > target and levels < 10:
                p = 35 * p**3
                levels += 1
            if levels < 10:
                targets.append(target)
                n_raws.append(15**levels)

        ax.loglog(targets, n_raws, 'o-', label=f'Raw p = {raw_p:.0e}')

    ax.set_xlabel('Target Error Rate')
    ax.set_ylabel('Raw Magic States Required')
    ax.set_title('Distillation Overhead')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig('distillation_overhead.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nDistillation analysis saved to 'distillation_overhead.png'")


def factory_simulation(n_T_gates, factory_rate, algorithm_rate):
    """
    Simulate magic state factory feeding an algorithm.

    Parameters:
    -----------
    n_T_gates : int
        Total T gates in algorithm
    factory_rate : float
        Magic states produced per cycle
    algorithm_rate : float
        T gates consumed per cycle (when not stalled)
    """
    print("\n" + "="*60)
    print("FACTORY SIMULATION")
    print("="*60)

    buffer = []  # Queue of available magic states
    buffer_size = 10  # Maximum buffer size

    t_gates_done = 0
    cycles = 0
    stall_cycles = 0
    buffer_history = []

    while t_gates_done < n_T_gates:
        cycles += 1

        # Factory produces states
        if np.random.random() < factory_rate:
            if len(buffer) < buffer_size:
                buffer.append(1)

        # Algorithm consumes states
        if np.random.random() < algorithm_rate:
            if buffer:
                buffer.pop(0)
                t_gates_done += 1
            else:
                stall_cycles += 1

        buffer_history.append(len(buffer))

    print(f"\nSimulation Results:")
    print(f"  Total T gates: {n_T_gates}")
    print(f"  Total cycles: {cycles}")
    print(f"  Stall cycles: {stall_cycles} ({100*stall_cycles/cycles:.1f}%)")
    print(f"  Effective rate: {n_T_gates/cycles:.4f} T/cycle")
    print(f"  Factory rate: {factory_rate:.4f} T/cycle")
    print(f"  Demand rate: {algorithm_rate:.4f} T/cycle")

    # Plot buffer history
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(buffer_history, 'b-', alpha=0.7)
    ax.axhline(y=buffer_size, color='r', linestyle='--', label='Buffer max')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Buffer Size')
    ax.set_title(f'Magic State Buffer (Factory={factory_rate:.2f}/cycle, Demand={algorithm_rate:.2f}/cycle)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('factory_buffer.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Factory simulation saved to 'factory_buffer.png'")


def t_gate_verification():
    """Verify T-gate teleportation on various input states."""
    print("\n" + "="*60)
    print("T-GATE TELEPORTATION VERIFICATION")
    print("="*60)

    test_states = [
        (ket_0, "|0⟩"),
        (ket_1, "|1⟩"),
        (ket_plus, "|+⟩"),
        (ket_minus, "|−⟩"),
        ((ket_0 + 1j * ket_1) / np.sqrt(2), "|i⟩")
    ]

    print("\nTesting T-gate on various states:")
    print("-" * 50)

    all_passed = True

    for psi, name in test_states:
        # Run teleportation
        output, outcome = t_gate_teleportation(psi, verbose=False)

        # Compare to direct T application
        expected = T @ psi
        expected = expected / np.linalg.norm(expected)

        # Fidelity
        fidelity = np.abs((expected.conj().T @ output)[0, 0])**2

        passed = fidelity > 0.9999
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(f"  T{name}: Fidelity = {fidelity:.6f} [{status}]")

    print("-" * 50)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")


def resource_estimation_example():
    """Example resource estimation for a realistic algorithm."""
    print("\n" + "="*60)
    print("RESOURCE ESTIMATION: SHOR'S ALGORITHM (2048-bit)")
    print("="*60)

    # Parameters
    n_bits = 2048
    n_logical_qubits = 4 * n_bits  # Approximate
    n_T_gates = 8e10  # Approximate for 2048-bit factoring

    # Surface code parameters
    p_phys = 1e-3
    p_target = 1e-15

    # Calculate code distance
    # p_L ~ (p/p_th)^((d+1)/2), solve for d
    p_th = 0.01
    d = 2 * np.log(1/p_target) / np.log(p_th/p_phys) + 1
    d = int(np.ceil(d)) | 1  # Round up to odd

    # Distillation levels needed
    p = p_phys
    levels = 0
    while p > p_target / 100:  # Extra margin
        p = 35 * p**3
        levels += 1

    raw_per_output = 15**levels

    print(f"\nAlgorithm Parameters:")
    print(f"  Number to factor: 2^{n_bits}")
    print(f"  Logical qubits: ~{n_logical_qubits}")
    print(f"  T gates: ~{n_T_gates:.0e}")

    print(f"\nSurface Code Parameters:")
    print(f"  Physical error rate: {p_phys}")
    print(f"  Target logical error: {p_target}")
    print(f"  Code distance: d = {d}")

    print(f"\nDistillation:")
    print(f"  Levels: {levels}")
    print(f"  Raw states per output: {raw_per_output}")

    # Physical qubits
    qubits_per_logical = 2 * d**2
    data_qubits = n_logical_qubits * qubits_per_logical

    # Factory (assume needs to sustain 1e6 T/s at 1 MHz)
    factory_units = 100
    factory_qubits = factory_units * 30 * d**2

    total_qubits = data_qubits + factory_qubits

    print(f"\nPhysical Resources:")
    print(f"  Data qubits: {data_qubits:,.0f}")
    print(f"  Factory qubits: {factory_qubits:,.0f}")
    print(f"  Total physical qubits: {total_qubits:,.0f}")

    # Time estimate
    cycle_rate = 1e6  # 1 MHz
    t_per_T = 2 * d  # Cycles per T gate (excluding factory)
    parallelism = 100  # Assume 100 T gates in parallel

    total_cycles = n_T_gates / parallelism * t_per_T
    total_time = total_cycles / cycle_rate

    print(f"\nTime Estimate:")
    print(f"  Cycle rate: {cycle_rate/1e6:.1f} MHz")
    print(f"  T parallelism: {parallelism}")
    print(f"  Total cycles: {total_cycles:.2e}")
    print(f"  Total time: {total_time/3600:.1f} hours = {total_time/86400:.1f} days")


def main():
    """Run all Day 825 demonstrations."""
    print("Day 825: T-Gate Injection & Magic State Integration")
    print("="*60)

    # T-gate teleportation demo
    print("\n--- T-Gate Teleportation Demo ---")
    t_gate_teleportation(ket_plus, verbose=True)

    # Verification
    t_gate_verification()

    # Distillation analysis
    analyze_distillation_overhead()

    # Factory simulation
    factory_simulation(n_T_gates=1000, factory_rate=0.15, algorithm_rate=0.1)

    # Resource estimation
    resource_estimation_example()

    print("\n" + "="*60)
    print("Day 825 Computational Lab Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 10. Summary

### Key Formulas Table

| Concept | Formula/Expression |
|---------|-------------------|
| T gate | $T = \text{diag}(1, e^{i\pi/4})$ |
| Magic state | $\|T\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$ |
| T teleportation correction | $S^m$ where $m$ is X measurement outcome |
| 15-to-1 error reduction | $p_{\text{out}} = 35 p_{\text{in}}^3$ |
| Multi-level states | $n_{\text{raw}} = 15^k$ for $k$ levels |
| T gate time (surgery) | $t_T = 2d$ cycles + factory delay |
| Factory footprint | $A \approx 30d^2$ qubits per unit |
| Factory rate | $r \approx 1/(10d)$ states/cycle/unit |

### Key Takeaways

1. **T gates are the bottleneck** in fault-tolerant quantum computing
2. **Magic states** enable non-Clifford gates via teleportation
3. **Distillation** reduces magic state errors exponentially (15-to-1: $p \to 35p^3$)
4. **Factory design** balances space (qubits) vs. throughput (T gates/time)
5. **Multi-level distillation** achieves arbitrarily low error at polynomial cost
6. **Architecture integration** requires careful placement of factories near computation

---

## 11. Daily Checklist

- [ ] I understand why T gates cannot be transversal on surface codes
- [ ] I can execute the T-gate teleportation protocol step-by-step
- [ ] I know how 15-to-1 distillation reduces magic state error
- [ ] I can calculate distillation overhead for a target error rate
- [ ] I can design a factory to meet T-gate throughput requirements
- [ ] I completed the computational lab and verified T-gate correctness

---

## 12. Preview: Day 826

Tomorrow is **Week 118 Synthesis** where we bring together all lattice surgery concepts:

- Complete lattice surgery compilation pipeline
- End-to-end algorithm execution
- Comparison with other fault-tolerant approaches
- Research frontiers and recent advances

We will compile a small quantum algorithm to lattice surgery primitives and estimate its full resource requirements.

---

*"The T gate is the quantum computer's most precious resource - every algorithm's feasibility ultimately comes down to how efficiently we can produce magic states."*
