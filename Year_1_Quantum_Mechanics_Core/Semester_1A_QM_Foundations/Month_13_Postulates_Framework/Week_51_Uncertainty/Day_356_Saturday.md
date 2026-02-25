# Day 356: Incompatible Observables — Complementarity and Beyond

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Complementarity & Measurement |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 356, you will be able to:

1. Explain Bohr's complementarity principle and its modern formulation
2. Analyze sequential measurements of non-commuting observables
3. Connect measurement disturbance to non-commutativity
4. Apply uncertainty relations to quantum cryptography (BB84)
5. Understand the no-cloning theorem as a consequence of incompatibility
6. Distinguish between preparation uncertainty and measurement disturbance

---

## Core Content

### 1. Bohr's Complementarity Principle

In 1927, Niels Bohr articulated the **complementarity principle**:

> *Certain pairs of physical properties are mutually exclusive: precise knowledge of one precludes precise knowledge of the other. Together, they provide a complete description.*

**Examples of complementary descriptions:**

| Property 1 | Property 2 | System |
|------------|------------|--------|
| Position | Momentum | Particle |
| Wave behavior | Particle behavior | Photon |
| Sx | Sy | Spin-1/2 |
| Path | Interference | Double-slit |
| Energy | Time of emission | Photon |

**Modern mathematical formulation:**

Two observables Â and B̂ are complementary (incompatible) if:
$$[\hat{A}, \hat{B}] \neq 0$$

They cannot simultaneously have definite values in any quantum state.

---

### 2. Sequential Measurements

What happens when we measure non-commuting observables in sequence?

**Scenario:** Start with state |ψ⟩, measure A, then measure B.

**Result:** The outcomes depend on the order!

**Example: Spin-1/2**

1. Prepare |↑z⟩ (eigenstate of Ŝz with eigenvalue +ℏ/2)
2. Measure Ŝx: get ±ℏ/2 with probability 1/2 each
3. State collapses to |±x⟩
4. Measure Ŝz: get ±ℏ/2 with probability 1/2 each!

The second Ŝx measurement **destroyed** the Ŝz information.

**Contrast with commuting observables:**

For L̂² and L̂z (which commute):
1. Measure L̂² → get l(l+1)ℏ², state in eigenspace of L̂²
2. Measure L̂z → get mℏ, state is now |l,m⟩
3. Measure L̂² again → get same l(l+1)ℏ² with certainty!

Commuting observables can be measured without disturbing each other.

---

### 3. Measurement Disturbance

**Heisenberg's microscope (1927):**

To measure position precisely:
- Use short wavelength photon (high momentum)
- Photon scatters, transferring uncertain momentum
- Position-momentum disturbance relation

**Modern understanding:**

The disturbance is not merely practical—it's fundamental. For non-commuting observables:

$$\epsilon_A \cdot \eta_B + \eta_A \cdot \epsilon_B + \epsilon_A \cdot \epsilon_B \geq \frac{1}{2}|⟨[\hat{A}, \hat{B}]⟩|$$

where:
- εₐ = error in measuring A
- ηₐ = disturbance to A caused by measuring B

**Ozawa's inequality (2003):**

$$\epsilon_A \cdot \eta_B + \epsilon_A \cdot \sigma_B + \sigma_A \cdot \eta_B \geq \frac{1}{2}|⟨[\hat{A}, \hat{B}]⟩|$$

This separates measurement error from preparation uncertainty.

---

### 4. The No-Cloning Theorem

**Theorem (Wootters-Zurek, Dieks, 1982):**

> *It is impossible to create an exact copy of an arbitrary unknown quantum state.*

**Proof:**

Suppose a cloning machine U exists:
$$U|ψ⟩|0⟩ = |ψ⟩|ψ⟩$$

for all |ψ⟩. Consider two states |ψ⟩ and |φ⟩:

$$U|ψ⟩|0⟩ = |ψ⟩|ψ⟩, \quad U|φ⟩|0⟩ = |φ⟩|φ⟩$$

By linearity:
$$U(|ψ⟩ + |φ⟩)|0⟩ = |ψ⟩|ψ⟩ + |φ⟩|φ⟩$$

But cloning would require:
$$U(|ψ⟩ + |φ⟩)|0⟩ = (|ψ⟩ + |φ⟩)(|ψ⟩ + |φ⟩)$$

These are not equal → contradiction! **Q.E.D.**

**Connection to uncertainty:**

If cloning were possible, we could:
1. Clone a state many times
2. Measure x on some copies, p on others
3. Beat the uncertainty principle

No-cloning protects the uncertainty principle!

---

### 5. Quantum Cryptography: BB84 Protocol

The incompatibility of different bases enables secure key distribution.

**BB84 Protocol (Bennett-Brassard, 1984):**

1. **Alice** randomly chooses bits (0 or 1) and bases (Z or X)
2. **Alice** sends qubits:
   - Z-basis: |0⟩ for 0, |1⟩ for 1
   - X-basis: |+⟩ for 0, |-⟩ for 1
3. **Bob** randomly chooses measurement basis (Z or X)
4. **Public discussion:** Alice and Bob reveal bases (not bits)
5. **Key sifting:** Keep only bits where bases matched

**Security from uncertainty:**

An eavesdropper (Eve) cannot measure both Z and X bases without disturbance!

- If Eve measures in Z when Alice sent X-basis → 50% error introduced
- If Eve measures in X when Alice sent Z-basis → 50% error introduced

Any eavesdropping introduces detectable errors (~25% for simple attacks).

**Mathematical basis:**

$$[\sigma_z, \sigma_x] = 2i\sigma_y \neq 0$$

Z and X measurements are incompatible → security!

---

### 6. Which-Path Information and Interference

The **double-slit experiment** beautifully illustrates complementarity.

**Setup:**
- Particle passes through double slit
- Detected on screen

**Without path detection:**
- Wave function passes through both slits
- Interference pattern observed
- No which-path information

**With path detection:**
- Measurement determines which slit
- State collapses to one slit
- No interference pattern
- Full which-path information

**The complementarity relation:**

$$\mathcal{D}^2 + \mathcal{V}^2 \leq 1$$

where:
- D = distinguishability (which-path knowledge)
- V = visibility (interference contrast)

**Full path information (D = 1) → No interference (V = 0)**

**Full interference (V = 1) → No path information (D = 0)**

---

### 7. Complete Sets of Commuting Observables (CSCO)

A **Complete Set of Commuting Observables (CSCO)** is a maximal set of mutually commuting operators.

**Definition:** {Â₁, Â₂, ..., Âₙ} is a CSCO if:
1. [Âᵢ, Âⱼ] = 0 for all i, j
2. Their simultaneous eigenstates are non-degenerate

**Examples:**

| System | CSCO | Quantum numbers |
|--------|------|-----------------|
| 1D particle | {Ĥ} | n |
| Free particle 3D | {Ĥ, p̂x, p̂y, p̂z} | E, px, py, pz |
| Hydrogen atom | {Ĥ, L̂², L̂z, Ŝ²,  Ŝz} | n, l, m, s, ms |
| 3D HO | {Ĥ, L̂², L̂z} or {n̂x, n̂y, n̂z} | Different labelings |

**Physical meaning:**

- A CSCO fully specifies a quantum state (up to phase)
- Different CSCOs give different "coordinate systems" for Hilbert space
- Adding a non-commuting observable would over-specify (impossible)

---

### 8. State Preparation vs. Measurement Uncertainty

Two distinct sources of uncertainty:

**Preparation uncertainty (Kennard-Robertson):**
$$\sigma_A \sigma_B \geq \frac{1}{2}|⟨[\hat{A}, \hat{B}]⟩|$$

This limits how sharp we can **prepare** a state in both A and B.

**Measurement-disturbance (Ozawa):**
$$\epsilon_A \eta_B + \epsilon_A \sigma_B + \sigma_A \eta_B \geq \frac{1}{2}|⟨[\hat{A}, \hat{B}]⟩|$$

This limits how precisely we can **measure** A without disturbing B.

**Key distinction:**

- Preparation uncertainty: intrinsic to quantum states
- Measurement-disturbance: involves interaction with apparatus

Both arise from non-commutativity, but they are conceptually different.

---

## Physical Interpretation

### The Deep Meaning of Incompatibility

**What it says:**

1. Nature has fundamentally exclusive properties
2. Knowing one precludes knowing another
3. This is not a limitation of our knowledge—it's reality

**Philosophical implications:**

- **Realism challenged:** Physical properties may not exist before measurement
- **Contextuality:** Measurement results can depend on what else is measured
- **Information bounds:** Nature limits accessible information

**Einstein's objection (EPR, 1935):**

"If, without in any way disturbing a system, we can predict with certainty the value of a physical quantity, then there exists an element of physical reality corresponding to this quantity."

**Bohr's response:** The experimental context matters. Different measurement setups probe different realities.

---

## Worked Examples

### Example 1: Stern-Gerlach Sequential Measurement

**Problem:** Particles in state |↑z⟩ pass through three Stern-Gerlach apparatuses:
1. SGz (blocks -z)
2. SGx (blocks -x)
3. SGz (blocks -z)

What fraction of original particles emerge?

**Solution:**

**After SG1 (SGz, block -z):**
- Input: |↑z⟩
- All pass (100%), state remains |↑z⟩

**After SG2 (SGx, block -x):**
- Input: |↑z⟩ = (|+x⟩ + |-x⟩)/√2
- Probability to pass (+x): |⟨+x|↑z⟩|² = 1/2
- State becomes: |+x⟩ = (|↑z⟩ + |↓z⟩)/√2

**After SG3 (SGz, block -z):**
- Input: |+x⟩ = (|↑z⟩ + |↓z⟩)/√2
- Probability to pass (↑z): |⟨↑z|+x⟩|² = 1/2

**Total fraction:**
$$P = 1 \times \frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$$

$$\boxed{\text{25% of particles emerge}}$$

**Note:** If SG2 were removed, 100% would emerge! The intermediate measurement destroys Sz information.

---

### Example 2: BB84 Security Analysis

**Problem:** In BB84, Alice sends 1000 qubits. Eve intercepts and measures in a random basis. Calculate the expected error rate.

**Solution:**

**Eve's attack:**
- Eve randomly chooses Z or X basis (50% each)
- Eve resends based on her measurement result

**When Eve's basis matches Alice's:**
- No error introduced (50% of qubits)
- Eve gets correct bit value

**When Eve's basis differs from Alice's:**
- 50% of qubits (Eve chose wrong)
- Eve's measurement disturbs the state
- When Bob measures in Alice's basis: 50% error

**Total error rate:**
$$P_{error} = P(\text{Eve wrong}) \times P(\text{error | Eve wrong})$$
$$= 0.5 \times 0.5 = 0.25$$

$$\boxed{\text{Error rate} = 25\%}$$

If Alice and Bob see ~25% errors in their test subset, they know an eavesdropper is present.

---

### Example 3: CSCO for Hydrogen Atom

**Problem:** Explain why {Ĥ, L̂², L̂z} is a CSCO for hydrogen but {Ĥ, L̂x, L̂z} is not.

**Solution:**

**Checking {Ĥ, L̂², L̂z}:**

1. [Ĥ, L̂²] = 0 ✓ (for central potential)
2. [Ĥ, L̂z] = 0 ✓ (rotation symmetry)
3. [L̂², L̂z] = 0 ✓

All pairs commute → can have simultaneous eigenstates |n, l, m⟩.

**Checking {Ĥ, L̂x, L̂z}:**

1. [Ĥ, L̂x] = 0 ✓
2. [Ĥ, L̂z] = 0 ✓
3. [L̂x, L̂z] = -iℏL̂y ≠ 0 ✗

L̂x and L̂z don't commute → not a valid CSCO!

Cannot label states by definite Lx AND Lz simultaneously.

$$\boxed{\{Ĥ, L̂², L̂z\} \text{ is a valid CSCO; } \{Ĥ, L̂_x, L̂_z\} \text{ is not}}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Sequential measurements:** A spin-1/2 particle is in state |+x⟩. Calculate the probability of getting +ℏ/2 if we first measure Sz, then Sx.

2. **CSCO identification:** Which of the following are valid CSCOs for a 2D harmonic oscillator?
   (a) {Ĥx, Ĥy}
   (b) {Ĥ, L̂z}
   (c) {Ĥ, p̂x}
   (d) {n̂x, n̂y}

3. **Path-interference trade-off:** In a Mach-Zehnder interferometer with 80% distinguishability, what is the maximum visibility?

### Level 2: Intermediate

4. **Eavesdropping detection:** In BB84, Alice and Bob publicly compare 200 test bits. If Eve measures all qubits in a fixed basis (Z), what is the expected number of errors? What if Eve uses the optimal random strategy?

5. **Spin complementarity:** For a spin-1 particle, find the minimum value of σ²_{Sx} + σ²_{Sy} + σ²_{Sz} over all states. Which states achieve this minimum?

6. **No-cloning consequence:** Alice has one copy of an unknown qubit state |ψ⟩ = α|0⟩ + β|1⟩. She wants to learn both |α|² and the relative phase arg(β/α). Explain why this is impossible.

### Level 3: Challenging

7. **Ozawa's inequality:** For a spin-1/2 particle in state |↑z⟩, a measurement of Sx is performed with error εₓ = 0.1ℏ. What is the minimum disturbance ηz to Sz?

8. **Weak measurements:** In weak measurement, the pointer variable is only weakly coupled to the observable. Show that weak measurements can violate the error-disturbance trade-off while still satisfying the uncertainty principle.

9. **Kochen-Specker theorem:** Explain how contextuality (the dependence of measurement outcomes on the measurement context) is related to non-commutativity. Give an example using spin-1 measurements.

---

## Computational Lab

### Objective
Simulate sequential measurements, explore complementarity, and implement BB84.

```python
"""
Day 356 Computational Lab: Incompatible Observables
Quantum Mechanics Core - Year 1, Week 51
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

# =============================================================================
# Part 1: Sequential Measurements
# =============================================================================

print("=" * 70)
print("Part 1: Sequential Measurements of Incompatible Observables")
print("=" * 70)

# Pauli matrices (with hbar = 1)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# Spin operators (hbar = 1)
Sx = 0.5 * sigma_x
Sy = 0.5 * sigma_y
Sz = 0.5 * sigma_z

# Basis states
up_z = np.array([[1], [0]], dtype=complex)
down_z = np.array([[0], [1]], dtype=complex)
up_x = (up_z + down_z) / np.sqrt(2)
down_x = (up_z - down_z) / np.sqrt(2)
up_y = (up_z + 1j*down_z) / np.sqrt(2)
down_y = (up_z - 1j*down_z) / np.sqrt(2)

def measure(state: np.ndarray, operator: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Simulate a quantum measurement.

    Returns:
        (outcome, post_measurement_state)
    """
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(operator)

    # Calculate probabilities
    probabilities = [np.abs(np.vdot(eigenvectors[:, i], state))**2
                     for i in range(len(eigenvalues))]

    # Randomly select outcome
    outcome_idx = np.random.choice(len(eigenvalues), p=probabilities)
    outcome = eigenvalues[outcome_idx]
    new_state = eigenvectors[:, outcome_idx:outcome_idx+1]

    return outcome, new_state

def sequential_measurement_experiment(initial_state: np.ndarray,
                                       operators: List[np.ndarray],
                                       n_trials: int = 1000) -> np.ndarray:
    """
    Perform sequential measurements and record all outcomes.

    Returns:
        Array of shape (n_trials, n_measurements) with outcomes
    """
    n_ops = len(operators)
    results = np.zeros((n_trials, n_ops))

    for trial in range(n_trials):
        state = initial_state.copy()
        for i, op in enumerate(operators):
            outcome, state = measure(state, op)
            results[trial, i] = outcome

    return results

# Experiment: Sz -> Sx -> Sz on |↑z⟩
print("\nExperiment: Sz → Sx → Sz starting from |↑z⟩")
print("-" * 50)

initial = up_z
operators = [Sz, Sx, Sz]
n_trials = 10000

results = sequential_measurement_experiment(initial, operators, n_trials)

print(f"Number of trials: {n_trials}")
print(f"\nFirst measurement (Sz):")
print(f"  P(+1/2) = {np.mean(results[:, 0] > 0):.3f}")
print(f"  P(-1/2) = {np.mean(results[:, 0] < 0):.3f}")

print(f"\nSecond measurement (Sx):")
print(f"  P(+1/2) = {np.mean(results[:, 1] > 0):.3f}")
print(f"  P(-1/2) = {np.mean(results[:, 1] < 0):.3f}")

print(f"\nThird measurement (Sz):")
print(f"  P(+1/2) = {np.mean(results[:, 2] > 0):.3f}")
print(f"  P(-1/2) = {np.mean(results[:, 2] < 0):.3f}")

print("\nNote: Initial state was |↑z⟩ but final Sz is random!")
print("The Sx measurement destroyed the Sz information.")

# =============================================================================
# Part 2: Commuting vs Non-Commuting Observables
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Commuting vs Non-Commuting Observables")
print("=" * 70)

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

# Check commutators
print("\nCommutators (magnitude):")
print(f"|[Sz, Sx]| = {np.linalg.norm(commutator(Sz, Sx)):.4f}")
print(f"|[Sz, Sy]| = {np.linalg.norm(commutator(Sz, Sy)):.4f}")
print(f"|[Sx, Sy]| = {np.linalg.norm(commutator(Sx, Sy)):.4f}")

# For commuting observables, create a 3-level system
# L^2 and Lz commute
L = 1  # l = 1 (spin-1)

# L^2 in |l,m⟩ basis: always l(l+1)ℏ² = 2
L_squared = 2 * np.eye(3, dtype=complex)

# Lz in |1,1⟩, |1,0⟩, |1,-1⟩ basis
Lz_3 = np.diag([1, 0, -1]).astype(complex)

print(f"\n|[L², Lz]| = {np.linalg.norm(commutator(L_squared, Lz_3)):.4f}")
print("L² and Lz commute → can measure both without disturbance")

# =============================================================================
# Part 3: BB84 Quantum Key Distribution
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: BB84 Quantum Key Distribution Simulation")
print("=" * 70)

def bb84_simulation(n_bits: int, eve_present: bool = False,
                    eve_strategy: str = 'random') -> dict:
    """
    Simulate BB84 quantum key distribution.

    Parameters:
        n_bits: Number of qubits sent
        eve_present: Whether an eavesdropper is present
        eve_strategy: 'random' or 'fixed_z'

    Returns:
        Dictionary with results
    """
    # Alice's random bits and bases
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0 = Z, 1 = X

    # Prepare qubits
    qubits = []
    for bit, basis in zip(alice_bits, alice_bases):
        if basis == 0:  # Z basis
            qubit = up_z if bit == 0 else down_z
        else:  # X basis
            qubit = up_x if bit == 0 else down_x
        qubits.append(qubit)

    # Eve's interception
    if eve_present:
        if eve_strategy == 'random':
            eve_bases = np.random.randint(0, 2, n_bits)
        else:  # fixed_z
            eve_bases = np.zeros(n_bits, dtype=int)

        eve_bits = []
        new_qubits = []
        for i, (qubit, eve_basis) in enumerate(zip(qubits, eve_bases)):
            # Eve measures
            if eve_basis == 0:
                op = Sz
                states = [up_z, down_z]
            else:
                op = Sx
                states = [up_x, down_x]

            # Get measurement outcome
            probs = [np.abs(np.vdot(s, qubit))**2 for s in states]
            eve_outcome = np.random.choice([0, 1], p=probs)
            eve_bits.append(eve_outcome)

            # Eve resends
            new_qubits.append(states[eve_outcome])

        qubits = new_qubits
        eve_bits = np.array(eve_bits)

    # Bob's measurement
    bob_bases = np.random.randint(0, 2, n_bits)
    bob_bits = []

    for qubit, bob_basis in zip(qubits, bob_bases):
        if bob_basis == 0:
            op = Sz
            states = [up_z, down_z]
        else:
            op = Sx
            states = [up_x, down_x]

        probs = [np.abs(np.vdot(s, qubit))**2 for s in states]
        bob_outcome = np.random.choice([0, 1], p=probs)
        bob_bits.append(bob_outcome)

    bob_bits = np.array(bob_bits)

    # Sifting: keep only matching bases
    matching = alice_bases == bob_bases
    sifted_alice = alice_bits[matching]
    sifted_bob = bob_bits[matching]

    # Calculate error rate
    errors = sifted_alice != sifted_bob
    error_rate = np.mean(errors) if len(errors) > 0 else 0

    return {
        'n_bits': n_bits,
        'n_sifted': len(sifted_alice),
        'error_rate': error_rate,
        'eve_present': eve_present
    }

# Run simulations
print("\n--- No Eavesdropper ---")
result_no_eve = bb84_simulation(10000, eve_present=False)
print(f"Bits sent: {result_no_eve['n_bits']}")
print(f"Sifted key length: {result_no_eve['n_sifted']}")
print(f"Error rate: {result_no_eve['error_rate']:.4f}")

print("\n--- With Eavesdropper (Random Basis) ---")
result_eve_random = bb84_simulation(10000, eve_present=True, eve_strategy='random')
print(f"Bits sent: {result_eve_random['n_bits']}")
print(f"Sifted key length: {result_eve_random['n_sifted']}")
print(f"Error rate: {result_eve_random['error_rate']:.4f}")
print(f"Expected error rate: 0.25")

print("\n--- With Eavesdropper (Fixed Z Basis) ---")
result_eve_fixed = bb84_simulation(10000, eve_present=True, eve_strategy='fixed_z')
print(f"Error rate: {result_eve_fixed['error_rate']:.4f}")
print(f"Expected error rate: 0.25")

# =============================================================================
# Part 4: Path-Interference Complementarity
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Path-Interference Complementarity")
print("=" * 70)

def interference_visibility(distinguishability: float) -> float:
    """
    Calculate maximum visibility given distinguishability.
    D² + V² ≤ 1
    """
    return np.sqrt(1 - distinguishability**2)

distinguishabilities = np.linspace(0, 1, 100)
visibilities = interference_visibility(distinguishabilities)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: D-V trade-off
ax1 = axes[0]
ax1.plot(distinguishabilities, visibilities, 'b-', linewidth=2)
ax1.fill_between(distinguishabilities, visibilities, 1, alpha=0.3, color='red',
                  label='Forbidden region')
ax1.fill_between(distinguishabilities, 0, visibilities, alpha=0.3, color='green',
                  label='Allowed region')
ax1.set_xlabel('Distinguishability D (which-path knowledge)', fontsize=12)
ax1.set_ylabel('Visibility V (interference contrast)', fontsize=12)
ax1.set_title('Complementarity: D² + V² ≤ 1', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Panel 2: Double-slit patterns
ax2 = axes[1]
x = np.linspace(-10, 10, 500)

# Different distinguishabilities
D_values = [0, 0.5, 0.8, 1.0]
for D in D_values:
    V = interference_visibility(D)
    # Interference pattern: 1 + V*cos(kx)
    pattern = 1 + V * np.cos(2 * x)
    ax2.plot(x, pattern, linewidth=2, label=f'D = {D}, V = {V:.2f}')

ax2.set_xlabel('Position', fontsize=12)
ax2.set_ylabel('Intensity', fontsize=12)
ax2.set_title('Interference Patterns for Different Path Knowledge', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_356_complementarity.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_356_complementarity.png'")

# =============================================================================
# Part 5: Stern-Gerlach Cascade
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Stern-Gerlach Cascade Simulation")
print("=" * 70)

def stern_gerlach_cascade(initial_state: np.ndarray,
                          measurements: List[str],
                          n_particles: int = 10000) -> np.ndarray:
    """
    Simulate particles passing through Stern-Gerlach apparatuses.

    Parameters:
        initial_state: Initial spin state
        measurements: List of axes ('z', 'x', 'y') and selections ('+' or None for both)
        n_particles: Number of particles to simulate

    Returns:
        Fraction of particles passing each stage
    """
    operators = {'z': Sz, 'x': Sx, 'y': Sy}
    up_states = {'z': up_z, 'x': up_x, 'y': up_y}
    down_states = {'z': down_z, 'x': down_x, 'y': down_y}

    fractions = [1.0]
    state = initial_state.copy()

    for meas in measurements:
        axis = meas[0]
        select = meas[1] if len(meas) > 1 else None

        op = operators[axis]
        up = up_states[axis]
        down = down_states[axis]

        # Probabilities
        p_up = np.abs(np.vdot(up, state))**2
        p_down = np.abs(np.vdot(down, state))**2

        if select == '+':
            fractions.append(fractions[-1] * p_up)
            state = up
        elif select == '-':
            fractions.append(fractions[-1] * p_down)
            state = down
        else:
            fractions.append(fractions[-1])

    return np.array(fractions[1:])

# SGz+ -> SGx+ -> SGz+ cascade
cascade1 = stern_gerlach_cascade(up_z, ['z+', 'x+', 'z+'])
print("\nCascade: SGz(+) → SGx(+) → SGz(+)")
print(f"After SGz(+): {cascade1[0]:.3f}")
print(f"After SGx(+): {cascade1[1]:.3f}")
print(f"After SGz(+): {cascade1[2]:.3f}")

# SGz+ -> SGz+ cascade (no intermediate measurement)
cascade2 = stern_gerlach_cascade(up_z, ['z+', 'z+'])
print(f"\nCascade: SGz(+) → SGz(+) (no intermediate)")
print(f"After SGz(+): {cascade2[0]:.3f}")
print(f"After SGz(+): {cascade2[1]:.3f}")

# =============================================================================
# Part 6: Visualization Summary
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Summary Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Sequential measurement outcomes
ax1 = axes[0, 0]
labels = ['Sz₁', 'Sx', 'Sz₂']
means = [np.mean(results[:, i]) for i in range(3)]
stds = [np.std(results[:, i]) for i in range(3)]
ax1.bar(labels, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
ax1.set_ylabel('Mean Outcome (ℏ units)', fontsize=12)
ax1.set_title('Sequential Measurements Sz → Sx → Sz\n(Starting from |↑z⟩)', fontsize=14)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: BB84 error rates
ax2 = axes[0, 1]
scenarios = ['No Eve', 'Eve (random)', 'Eve (fixed Z)']
error_rates = [result_no_eve['error_rate'],
               result_eve_random['error_rate'],
               result_eve_fixed['error_rate']]
colors = ['green', 'red', 'orange']
bars = ax2.bar(scenarios, error_rates, color=colors, alpha=0.7)
ax2.axhline(y=0.25, color='black', linestyle='--', label='Expected with Eve')
ax2.set_ylabel('Error Rate', fontsize=12)
ax2.set_title('BB84 Protocol: Eavesdropping Detection', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 0.35)

# Panel 3: Commutation table
ax3 = axes[1, 0]
operators_list = [Sx, Sy, Sz]
op_names = ['Sx', 'Sy', 'Sz']
comm_matrix = np.zeros((3, 3))
for i, A in enumerate(operators_list):
    for j, B in enumerate(operators_list):
        comm_matrix[i, j] = np.linalg.norm(commutator(A, B))

im = ax3.imshow(comm_matrix, cmap='Reds')
ax3.set_xticks(range(3))
ax3.set_yticks(range(3))
ax3.set_xticklabels(op_names)
ax3.set_yticklabels(op_names)
ax3.set_title('Commutator Magnitudes |[A, B]|', fontsize=14)
for i in range(3):
    for j in range(3):
        ax3.text(j, i, f'{comm_matrix[i,j]:.2f}', ha='center', va='center')
plt.colorbar(im, ax=ax3)

# Panel 4: Stern-Gerlach cascade comparison
ax4 = axes[1, 1]
x_pos = [1, 2, 3]
ax4.bar([x-0.15 for x in x_pos], cascade1, width=0.3, label='With Sx', color='blue', alpha=0.7)
ax4.bar([x+0.15 for x in [1, 2]], cascade2, width=0.3, label='Without Sx', color='green', alpha=0.7)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['After SG1', 'After SG2', 'After SG3'])
ax4.set_ylabel('Fraction Passing', fontsize=12)
ax4.set_title('Stern-Gerlach Cascade: Effect of Intermediate Measurement', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('day_356_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_356_summary.png'")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Incompatibility condition | [Â, B̂] ≠ 0 |
| Path-interference | D² + V² ≤ 1 |
| Ozawa's inequality | ε_A η_B + ε_A σ_B + σ_A η_B ≥ ½\|⟨[Â, B̂]⟩\| |
| No-cloning | Cannot create \|ψ⟩\|ψ⟩ from \|ψ⟩\|0⟩ for arbitrary \|ψ⟩ |
| BB84 error (Eve) | ~25% for intercept-resend attack |
| CSCO condition | [Âᵢ, Âⱼ] = 0 for all i, j |

### Main Takeaways

1. **Non-commuting observables are fundamentally incompatible** — Cannot have definite values simultaneously
2. **Sequential measurements disturb the state** — Order matters for non-commuting observables
3. **No-cloning protects uncertainty** — Cannot circumvent by copying states
4. **BB84 exploits incompatibility** — Eavesdropping introduces detectable errors
5. **Complementarity is fundamental** — Wave-particle, path-interference, conjugate variables

---

## Daily Checklist

- [ ] Read Sakurai Chapter 1.4 (Measurements, Observables, Uncertainty)
- [ ] Read about BB84 in Nielsen & Chuang Chapter 12
- [ ] Work through the sequential measurement examples
- [ ] Prove the no-cloning theorem
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Explain why measuring Sx destroys Sz information

---

## Preview: Day 357

Tomorrow is our **Week Review** session. We'll consolidate all the concepts from Week 51: commutators, canonical commutation relations, the generalized uncertainty principle, position-momentum and energy-time uncertainty, and incompatible observables. There will be a practice exam and comprehensive computational exercises.

---

*"Anyone who is not shocked by quantum theory has not understood it."* — Niels Bohr

---

**Next:** [Day_357_Sunday.md](Day_357_Sunday.md) — Week Review
