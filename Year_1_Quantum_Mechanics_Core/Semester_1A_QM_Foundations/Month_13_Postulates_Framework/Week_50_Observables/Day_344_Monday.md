# Day 344: The Measurement Postulate — Eigenvalues as Physical Outcomes

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Measurement Postulate & Born Rule |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 344, you will be able to:

1. State the measurement postulate and explain its physical significance
2. Apply the Born rule to calculate measurement probabilities
3. Distinguish between discrete and continuous spectra in measurements
4. Connect abstract eigenvalue problems to laboratory measurements
5. Calculate probabilities for real quantum systems (spin, energy, position)
6. Explain the probabilistic nature of quantum mechanics

---

## Core Content

### 1. The Central Problem: What Happens When We Measure?

In classical physics, measurement reveals pre-existing properties. A baseball has a definite position and velocity before, during, and after we look at it. Quantum mechanics fundamentally changes this picture.

**The quantum measurement question:**
> If a particle is in state |ψ⟩ = α|↑⟩ + β|↓⟩, and we measure its spin, what result do we get?

The answer is **not** "some combination of up and down." The measurement always yields a **definite** result: either ↑ or ↓. But which one?

---

### 2. The Measurement Postulate

**Postulate 3 (Measurement Outcomes):**

> *The only possible results of measuring an observable A are the eigenvalues {a} of the corresponding Hermitian operator Â.*

This connects the mathematical eigenvalue problem to laboratory measurements:

$$\boxed{\hat{A}|a⟩ = a|a⟩ \quad \Longleftrightarrow \quad \text{Measuring } A \text{ can yield result } a}$$

**Key implications:**
1. Eigenvalues are real (because Â is Hermitian) — measured values are real numbers
2. The spectrum of Â determines all possible measurement outcomes
3. If Â has eigenvalues {a₁, a₂, ...}, these are the **only** results you can ever observe

**Example: Spin-1/2**

The operator Ŝz has eigenvalues ±ℏ/2:

$$\hat{S}_z|↑⟩ = +\frac{\hbar}{2}|↑⟩, \quad \hat{S}_z|↓⟩ = -\frac{\hbar}{2}|↓⟩$$

Therefore, measuring Sz always yields exactly +ℏ/2 or -ℏ/2. Never 0, never ℏ/4, never any other value.

---

### 3. The Born Rule: Probability of Outcomes

**Postulate 4 (Born Rule for Discrete Spectra):**

> *If the system is in state |ψ⟩ and observable A is measured, the probability of obtaining eigenvalue a is:*

$$\boxed{P(a) = |⟨a|ψ⟩|^2}$$

This is the **Born rule**, named after Max Born (1926), for which he received the Nobel Prize. It provides the critical link between quantum states and experimental observations.

**Understanding the formula:**

- |ψ⟩ is the state before measurement
- |a⟩ is the eigenstate corresponding to eigenvalue a
- ⟨a|ψ⟩ is the probability **amplitude** (complex number)
- |⟨a|ψ⟩|² is the probability (real, non-negative)

**Normalization guarantees total probability = 1:**

$$\sum_a P(a) = \sum_a |⟨a|ψ⟩|^2 = ⟨ψ|ψ⟩ = 1$$

using completeness: $\sum_a |a⟩⟨a| = \hat{I}$.

---

### 4. Equivalent Formulations of the Born Rule

The Born rule can be written in several equivalent ways:

**Form 1: Inner product squared**
$$P(a) = |⟨a|ψ⟩|^2$$

**Form 2: Projection operator**
$$P(a) = ⟨ψ|\hat{P}_a|ψ⟩ \quad \text{where} \quad \hat{P}_a = |a⟩⟨a|$$

**Form 3: Coefficients in eigenstate expansion**

If $|ψ⟩ = \sum_n c_n|a_n⟩$, then:
$$P(a_n) = |c_n|^2$$

**Proof of equivalence (Form 1 = Form 2):**

$$⟨ψ|\hat{P}_a|ψ⟩ = ⟨ψ|a⟩⟨a|ψ⟩ = ⟨a|ψ⟩^* ⟨a|ψ⟩ = |⟨a|ψ⟩|^2 \quad ✓$$

---

### 5. Discrete vs. Continuous Spectra

#### Discrete Spectrum

For observables with discrete eigenvalues (spin, energy of bound states, angular momentum quantum numbers):

$$P(a) = |⟨a|ψ⟩|^2$$

Probabilities are well-defined numbers that sum to 1.

#### Continuous Spectrum

For observables with continuous eigenvalues (position x, momentum p):

$$\boxed{P(x ∈ [a,b]) = \int_a^b |ψ(x)|^2 dx}$$

The probability **density** is:

$$ρ(x) = |ψ(x)|^2 = |⟨x|ψ⟩|^2$$

**Key distinction:**
- Discrete: P(measuring exactly a) is finite
- Continuous: P(measuring exactly x₀) = 0; only probability densities are meaningful

**Example: Position measurement**

For a particle in state ψ(x):
- P(particle between x = 0 and x = 1) = ∫₀¹ |ψ(x)|² dx
- P(particle at exactly x = 0.5) = 0 (measure-zero set)

---

### 6. Physical Interpretation: What Does Probability Mean?

The Born rule has profound implications:

**1. Individual outcomes are fundamentally unpredictable**

Even with complete knowledge of |ψ⟩, we cannot predict which eigenvalue will be observed. This is not ignorance—it is intrinsic randomness.

**2. Probabilities require ensembles**

The Born rule predicts frequencies in repeated experiments on identically prepared systems:

$$\frac{N_a}{N_{total}} \xrightarrow{N→∞} P(a) = |⟨a|ψ⟩|^2$$

**3. Amplitudes, not probabilities, are fundamental**

Quantum mechanics works with probability amplitudes ⟨a|ψ⟩. Probabilities emerge only at the moment of measurement. This explains interference:

$$|⟨a|ψ_1 + ψ_2⟩|^2 ≠ |⟨a|ψ_1⟩|^2 + |⟨a|ψ_2⟩|^2$$

---

### 7. The Stern-Gerlach Experiment: Measurement in Action

The 1922 Stern-Gerlach experiment provides the paradigmatic example of quantum measurement.

**Setup:**
- Silver atoms pass through inhomogeneous magnetic field
- Field gradient couples to magnetic moment (spin)
- Atoms deflect based on spin component along field direction

**Classical prediction:** Continuous distribution of deflections

**Quantum result:** Only **two** discrete spots on detector (spin ±ℏ/2)

**Mathematical description:**

Initial state: $|ψ⟩ = α|↑⟩ + β|↓⟩$

After measurement:
- Result +ℏ/2 with probability |α|²
- Result -ℏ/2 with probability |β|²

**Sequential measurements:**

1. Measure Sz → get +ℏ/2 (state becomes |↑⟩)
2. Measure Sx → get ±ℏ/2 with 50% each (because ⟨±x|↑⟩ = 1/√2)
3. Measure Sz again → get ±ℏ/2 with 50% each!

The intermediate Sx measurement **destroys** the Sz information.

---

### 8. Connection to Experiments

**Photon polarization:**

$$|ψ⟩ = α|H⟩ + β|V⟩$$

When passed through a polarizer aligned with H:
- P(transmitted) = |α|² = |⟨H|ψ⟩|²
- P(absorbed) = |β|² = |⟨V|ψ⟩|²

**Energy measurements:**

For hydrogen atom in state $|ψ⟩ = \sum_n c_n|n⟩$:
- P(measuring En) = |cn|²
- Experimentally: spectral line intensities

**Position measurements:**

For particle with wave function ψ(x):
- P(detecting particle in [x, x+dx]) = |ψ(x)|² dx
- Experimentally: particle detector counts

---

## Quantum Computing Connection

### Measurement in the Computational Basis

In quantum computing, the standard measurement is in the **computational basis** {|0⟩, |1⟩}:

For qubit state $|ψ⟩ = α|0⟩ + β|1⟩$:

$$P(0) = |α|^2, \quad P(1) = |β|^2$$

**Qiskit implementation:**

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Create circuit with measurement
qc = QuantumCircuit(1, 1)
qc.h(0)  # Create |+⟩ = (|0⟩ + |1⟩)/√2
qc.measure(0, 0)

# Run on simulator
simulator = AerSimulator()
compiled = transpile(qc, simulator)
result = simulator.run(compiled, shots=1000).result()
counts = result.get_counts()
print(counts)  # {'0': ~500, '1': ~500}
```

### Measuring in Different Bases

To measure in the X-basis {|+⟩, |-⟩}:

1. Apply Hadamard: H|+⟩ = |0⟩, H|-⟩ = |1⟩
2. Measure in Z-basis
3. Map results: 0 → +, 1 → -

This is equivalent to measuring the Pauli-X observable σx.

### Measurement as a Resource

In quantum algorithms:
- **Grover's algorithm:** Measurement extracts the marked item
- **Shor's algorithm:** Measurement collapses to a periodic state
- **Quantum error correction:** Syndrome measurements detect errors without collapsing logical qubits

---

## Worked Examples

### Example 1: Spin Measurement Probabilities

**Problem:** A spin-1/2 particle is in state:
$$|ψ⟩ = \frac{1}{\sqrt{3}}|↑⟩ + \sqrt{\frac{2}{3}}|↓⟩$$

(a) What are the possible outcomes of measuring Sz?
(b) What is the probability of each outcome?
(c) Verify that probabilities sum to 1.

**Solution:**

(a) The possible outcomes are the eigenvalues of Ŝz:
$$\boxed{+\frac{\hbar}{2} \text{ and } -\frac{\hbar}{2}}$$

(b) Using the Born rule:

$$P\left(+\frac{\hbar}{2}\right) = |⟨↑|ψ⟩|^2 = \left|\frac{1}{\sqrt{3}}\right|^2 = \boxed{\frac{1}{3}}$$

$$P\left(-\frac{\hbar}{2}\right) = |⟨↓|ψ⟩|^2 = \left|\sqrt{\frac{2}{3}}\right|^2 = \boxed{\frac{2}{3}}$$

(c) Verification:
$$P(↑) + P(↓) = \frac{1}{3} + \frac{2}{3} = 1 \quad ✓$$

---

### Example 2: Qubit in Superposition

**Problem:** A qubit is prepared in state:
$$|ψ⟩ = \frac{1}{2}|0⟩ + \frac{\sqrt{3}}{2}|1⟩$$

(a) What is P(measuring 0)?
(b) What is P(measuring 1)?
(c) If we measure in the X-basis {|+⟩, |-⟩}, what is P(+)?

**Solution:**

(a) $P(0) = |⟨0|ψ⟩|^2 = |1/2|^2 = \boxed{1/4}$

(b) $P(1) = |⟨1|ψ⟩|^2 = |\sqrt{3}/2|^2 = \boxed{3/4}$

(c) For X-basis measurement, we need ⟨+|ψ⟩:

$$|+⟩ = \frac{1}{\sqrt{2}}(|0⟩ + |1⟩)$$

$$⟨+|ψ⟩ = \frac{1}{\sqrt{2}}\left(\frac{1}{2} + \frac{\sqrt{3}}{2}\right) = \frac{1 + \sqrt{3}}{2\sqrt{2}}$$

$$P(+) = |⟨+|ψ⟩|^2 = \frac{(1 + \sqrt{3})^2}{8} = \frac{1 + 2\sqrt{3} + 3}{8} = \frac{4 + 2\sqrt{3}}{8} = \boxed{\frac{2 + \sqrt{3}}{4} ≈ 0.933}$$

---

### Example 3: Harmonic Oscillator Energy Measurement

**Problem:** A quantum harmonic oscillator is in state:
$$|ψ⟩ = \frac{1}{\sqrt{2}}|0⟩ + \frac{1}{2}|1⟩ + \frac{1}{2}|2⟩$$

where |n⟩ are energy eigenstates with En = ℏω(n + 1/2).

(a) Verify normalization.
(b) What energies can be measured and with what probabilities?
(c) What is the probability of measuring E > ℏω?

**Solution:**

(a) Normalization check:
$$⟨ψ|ψ⟩ = \left|\frac{1}{\sqrt{2}}\right|^2 + \left|\frac{1}{2}\right|^2 + \left|\frac{1}{2}\right|^2 = \frac{1}{2} + \frac{1}{4} + \frac{1}{4} = 1 \quad ✓$$

(b) Possible energies and probabilities:

| n | En | P(En) = |cn|² |
|---|-----|---------------|
| 0 | ℏω/2 | 1/2 |
| 1 | 3ℏω/2 | 1/4 |
| 2 | 5ℏω/2 | 1/4 |

(c) Energy E > ℏω means E₁ or E₂:
$$P(E > ℏω) = P(E_1) + P(E_2) = \frac{1}{4} + \frac{1}{4} = \boxed{\frac{1}{2}}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Spin state analysis:** A spin-1/2 system is in state |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5.
   (a) Find P(Sz = +ℏ/2) and P(Sz = -ℏ/2).
   (b) Verify normalization.

2. **Three-level system:** A qutrit is in state |ψ⟩ = (|0⟩ + |1⟩ + |2⟩)/√3.
   Calculate the probability of measuring each basis state.

3. **Position probability:** A particle has wave function ψ(x) = √2 sin(πx) for 0 ≤ x ≤ 1 (zero elsewhere).
   (a) Verify normalization.
   (b) Find P(0 ≤ x ≤ 1/2).

### Level 2: Intermediate

4. **Basis change:** For |ψ⟩ = |↑⟩ (eigenstate of Sz):
   (a) Express |ψ⟩ in the Sx eigenbasis {|+x⟩, |-x⟩}.
   (b) Calculate P(Sx = +ℏ/2) and P(Sx = -ℏ/2).
   (Hint: |±x⟩ = (|↑⟩ ± |↓⟩)/√2)

5. **Sequential probabilities:** Starting from |ψ⟩ = |+⟩ = (|↑⟩ + |↓⟩)/√2:
   (a) What is P(Sz = +ℏ/2)?
   (b) After measuring Sz and getting +ℏ/2, what is P(Sx = +ℏ/2)?
   (c) Compare with the probability if we had measured Sx directly on |+⟩.

6. **Gaussian wave packet:** A particle has wave function:
   $$ψ(x) = \left(\frac{1}{πσ^2}\right)^{1/4} e^{-x^2/(2σ^2)}$$
   (a) Verify normalization.
   (b) Find P(|x| < σ).

### Level 3: Challenging

7. **Projection operator formalism:** Using P̂a = |a⟩⟨a|:
   (a) Show that P̂a² = P̂a (idempotent).
   (b) Show that P̂a† = P̂a (Hermitian).
   (c) Show that if |a⟩ and |a'⟩ are orthogonal eigenstates, then P̂a P̂a' = 0.

8. **Continuous measurement:** For a free particle with momentum-space wave function:
   $$φ(p) = \frac{1}{\sqrt{2\pi\hbar}} \cdot \frac{1}{\sqrt{Δp}} \quad \text{for } |p| < Δp$$
   (a) Find ψ(x) via Fourier transform.
   (b) Calculate P(|x| < ℏ/(2Δp)).

9. **Research problem:** The Born rule P(a) = |⟨a|ψ⟩|² is often taken as a postulate. Research "Gleason's theorem" and explain how it derives the Born rule from more basic axioms about probability and Hilbert spaces.

---

## Computational Lab

### Objective
Implement the Born rule and verify measurement probabilities through simulation.

```python
"""
Day 344 Computational Lab: The Measurement Postulate
Quantum Mechanics Core - Year 1, Week 50
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# =============================================================================
# Part 1: Born Rule Implementation
# =============================================================================

print("=" * 70)
print("Part 1: Born Rule Implementation")
print("=" * 70)

def born_probability(state: np.ndarray, eigenstate: np.ndarray) -> float:
    """
    Calculate measurement probability using the Born rule.
    P(a) = |⟨a|ψ⟩|²

    Parameters:
        state: The quantum state |ψ⟩ (column vector)
        eigenstate: The eigenstate |a⟩ (column vector)

    Returns:
        Probability of measuring the eigenvalue corresponding to |a⟩
    """
    amplitude = np.vdot(eigenstate, state)  # ⟨a|ψ⟩
    probability = np.abs(amplitude)**2
    return probability

def measurement_probabilities(state: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
    """
    Calculate all measurement probabilities in a given basis.

    Parameters:
        state: The quantum state |ψ⟩
        basis: List of orthonormal basis vectors

    Returns:
        Array of probabilities for each basis state
    """
    probs = np.array([born_probability(state, b) for b in basis])
    return probs

# Define computational basis for qubit
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
computational_basis = [ket_0, ket_1]

# Define Hadamard basis
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)
hadamard_basis = [ket_plus, ket_minus]

# Test state: |ψ⟩ = (1|0⟩ + √3|1⟩)/2
psi = (ket_0 + np.sqrt(3) * ket_1) / 2

print("\nTest state: |ψ⟩ = (|0⟩ + √3|1⟩)/2")
print(f"|ψ⟩ = {psi.flatten()}")

# Verify normalization
norm = np.vdot(psi, psi)
print(f"\nNormalization: ⟨ψ|ψ⟩ = {norm.real:.6f}")

# Calculate probabilities in computational basis
probs_z = measurement_probabilities(psi, computational_basis)
print(f"\nZ-basis (computational) measurement:")
print(f"P(|0⟩) = {probs_z[0]:.6f} (expected: 0.25)")
print(f"P(|1⟩) = {probs_z[1]:.6f} (expected: 0.75)")
print(f"Sum = {np.sum(probs_z):.6f}")

# Calculate probabilities in Hadamard basis
probs_x = measurement_probabilities(psi, hadamard_basis)
print(f"\nX-basis (Hadamard) measurement:")
print(f"P(|+⟩) = {probs_x[0]:.6f}")
print(f"P(|-⟩) = {probs_x[1]:.6f}")
print(f"Sum = {np.sum(probs_x):.6f}")

# =============================================================================
# Part 2: Simulating Measurements
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Simulating Quantum Measurements")
print("=" * 70)

def simulate_measurement(state: np.ndarray, basis: List[np.ndarray],
                         n_shots: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate quantum measurements and return outcome frequencies.

    Parameters:
        state: The quantum state to measure
        basis: Measurement basis (list of orthonormal vectors)
        n_shots: Number of measurement repetitions

    Returns:
        outcomes: Array of measurement outcomes (indices)
        frequencies: Normalized frequency of each outcome
    """
    # Calculate theoretical probabilities
    probs = measurement_probabilities(state, basis)

    # Simulate measurements by sampling from probability distribution
    outcomes = np.random.choice(len(basis), size=n_shots, p=probs)

    # Count frequencies
    counts = np.bincount(outcomes, minlength=len(basis))
    frequencies = counts / n_shots

    return outcomes, frequencies

# Simulate measurements
n_shots = 10000

print(f"\nSimulating {n_shots} measurements on |ψ⟩ = (|0⟩ + √3|1⟩)/2")

outcomes_z, freqs_z = simulate_measurement(psi, computational_basis, n_shots)
print(f"\nZ-basis results:")
print(f"Frequency(|0⟩) = {freqs_z[0]:.4f} (theory: 0.2500)")
print(f"Frequency(|1⟩) = {freqs_z[1]:.4f} (theory: 0.7500)")

outcomes_x, freqs_x = simulate_measurement(psi, hadamard_basis, n_shots)
print(f"\nX-basis results:")
print(f"Frequency(|+⟩) = {freqs_x[0]:.4f} (theory: {probs_x[0]:.4f})")
print(f"Frequency(|-⟩) = {freqs_x[1]:.4f} (theory: {probs_x[1]:.4f})")

# =============================================================================
# Part 3: Statistical Convergence
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Statistical Convergence of Measurement Frequencies")
print("=" * 70)

def convergence_analysis(state: np.ndarray, basis: List[np.ndarray],
                         max_shots: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze how measurement frequencies converge to theoretical probabilities.
    """
    shot_counts = np.logspace(1, np.log10(max_shots), 50, dtype=int)
    shot_counts = np.unique(shot_counts)

    theoretical = measurement_probabilities(state, basis)
    errors = []

    for n in shot_counts:
        _, freqs = simulate_measurement(state, basis, n)
        error = np.mean(np.abs(freqs - theoretical))
        errors.append(error)

    return shot_counts, np.array(errors)

shots, errors = convergence_analysis(psi, computational_basis)

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Convergence plot
ax1 = axes[0]
ax1.loglog(shots, errors, 'b-', linewidth=2, label='Observed error')
ax1.loglog(shots, 0.5/np.sqrt(shots), 'r--', linewidth=2, label='1/√N (theory)')
ax1.set_xlabel('Number of measurements', fontsize=12)
ax1.set_ylabel('Mean absolute error', fontsize=12)
ax1.set_title('Convergence to Born Rule Probabilities', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Histogram of outcomes
ax2 = axes[1]
x = np.arange(2)
width = 0.35

bars1 = ax2.bar(x - width/2, probs_z, width, label='Theory (Born rule)',
                color='steelblue', alpha=0.7)
bars2 = ax2.bar(x + width/2, freqs_z, width, label=f'Simulation ({n_shots} shots)',
                color='coral', alpha=0.7)

ax2.set_xlabel('Measurement outcome', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Z-basis Measurement Statistics', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(['|0⟩', '|1⟩'])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('day_344_born_rule.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_344_born_rule.png'")

# =============================================================================
# Part 4: Three-Level System (Qutrit)
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Qutrit Measurement")
print("=" * 70)

# Define qutrit basis
ket_qutrit_0 = np.array([[1], [0], [0]], dtype=complex)
ket_qutrit_1 = np.array([[0], [1], [0]], dtype=complex)
ket_qutrit_2 = np.array([[0], [0], [1]], dtype=complex)
qutrit_basis = [ket_qutrit_0, ket_qutrit_1, ket_qutrit_2]

# Qutrit state: |ψ⟩ = (2|0⟩ + i|1⟩ + |2⟩)/√6
psi_qutrit = (2*ket_qutrit_0 + 1j*ket_qutrit_1 + ket_qutrit_2) / np.sqrt(6)

print(f"\nQutrit state: |ψ⟩ = (2|0⟩ + i|1⟩ + |2⟩)/√6")
print(f"|ψ⟩ = {psi_qutrit.flatten()}")

# Calculate and verify probabilities
probs_qutrit = measurement_probabilities(psi_qutrit, qutrit_basis)
print(f"\nTheoretical probabilities:")
for i, p in enumerate(probs_qutrit):
    print(f"P(|{i}⟩) = {p:.6f}")
print(f"Sum = {np.sum(probs_qutrit):.6f}")

# Simulate measurements
_, freqs_qutrit = simulate_measurement(psi_qutrit, qutrit_basis, 10000)
print(f"\nSimulated frequencies (10000 shots):")
for i, f in enumerate(freqs_qutrit):
    print(f"Frequency(|{i}⟩) = {f:.4f}")

# =============================================================================
# Part 5: Spin-1/2 in Arbitrary Direction
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Spin Measurement in Arbitrary Direction")
print("=" * 70)

def spin_state(theta: float, phi: float) -> np.ndarray:
    """
    Create spin-1/2 state pointing in direction (θ, φ) on Bloch sphere.
    |ψ⟩ = cos(θ/2)|↑⟩ + e^(iφ)sin(θ/2)|↓⟩
    """
    return np.array([[np.cos(theta/2)],
                     [np.exp(1j*phi) * np.sin(theta/2)]], dtype=complex)

def measurement_basis(theta: float, phi: float) -> List[np.ndarray]:
    """
    Create measurement basis for spin component in direction (θ, φ).
    """
    up = spin_state(theta, phi)
    down = spin_state(np.pi - theta, phi + np.pi)
    return [up, down]

# State: spin along +x (θ=π/2, φ=0)
psi_x = spin_state(np.pi/2, 0)
print(f"\nSpin state along +x: |+x⟩ = (|↑⟩ + |↓⟩)/√2")
print(f"|ψ⟩ = {psi_x.flatten()}")

# Measure along z
z_basis = [ket_0, ket_1]  # |↑⟩ = |0⟩, |↓⟩ = |1⟩
probs_along_z = measurement_probabilities(psi_x, z_basis)
print(f"\nMeasure Sz on |+x⟩:")
print(f"P(↑) = {probs_along_z[0]:.4f} (expected: 0.5)")
print(f"P(↓) = {probs_along_z[1]:.4f} (expected: 0.5)")

# Measure along arbitrary direction (θ=π/3, φ=0)
theta_meas = np.pi/3
x_tilted_basis = measurement_basis(theta_meas, 0)
probs_tilted = measurement_probabilities(psi_x, x_tilted_basis)
print(f"\nMeasure spin along θ={np.degrees(theta_meas):.0f}° (tilted from z):")
print(f"P(+) = {probs_tilted[0]:.4f}")
print(f"P(-) = {probs_tilted[1]:.4f}")

# =============================================================================
# Part 6: Visualization - Measurement Outcomes
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualization of Measurement Statistics")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Qubit measurement
ax1 = axes[0, 0]
_, freqs_test = simulate_measurement(psi, computational_basis, 5000)
x = np.arange(2)
ax1.bar(x, probs_z, width=0.4, label='Theory', color='steelblue', alpha=0.7)
ax1.bar(x + 0.4, freqs_test, width=0.4, label='Simulation', color='coral', alpha=0.7)
ax1.set_xticks(x + 0.2)
ax1.set_xticklabels(['|0⟩', '|1⟩'])
ax1.set_ylabel('Probability')
ax1.set_title('Qubit: |ψ⟩ = (|0⟩ + √3|1⟩)/2')
ax1.legend()
ax1.set_ylim(0, 1)

# Panel 2: Qutrit measurement
ax2 = axes[0, 1]
_, freqs_qutrit_test = simulate_measurement(psi_qutrit, qutrit_basis, 5000)
x = np.arange(3)
ax2.bar(x, probs_qutrit, width=0.4, label='Theory', color='steelblue', alpha=0.7)
ax2.bar(x + 0.4, freqs_qutrit_test, width=0.4, label='Simulation', color='coral', alpha=0.7)
ax2.set_xticks(x + 0.2)
ax2.set_xticklabels(['|0⟩', '|1⟩', '|2⟩'])
ax2.set_ylabel('Probability')
ax2.set_title('Qutrit: |ψ⟩ = (2|0⟩ + i|1⟩ + |2⟩)/√6')
ax2.legend()
ax2.set_ylim(0, 1)

# Panel 3: Different basis measurements on same state
ax3 = axes[1, 0]
test_state = (ket_0 + 1j*ket_1) / np.sqrt(2)
probs_comp = measurement_probabilities(test_state, computational_basis)
probs_had = measurement_probabilities(test_state, hadamard_basis)

x = np.arange(2)
ax3.bar(x - 0.2, probs_comp, width=0.4, label='Z-basis', color='purple', alpha=0.7)
ax3.bar(x + 0.2, probs_had, width=0.4, label='X-basis', color='green', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(['Outcome 0/+', 'Outcome 1/-'])
ax3.set_ylabel('Probability')
ax3.set_title('|ψ⟩ = (|0⟩ + i|1⟩)/√2 in different bases')
ax3.legend()
ax3.set_ylim(0, 1)

# Panel 4: Spin measurement angle dependence
ax4 = axes[1, 1]
angles = np.linspace(0, np.pi, 50)
probs_up = []

# Start in |↑⟩, measure along direction θ from z-axis
psi_up = ket_0
for theta in angles:
    basis_theta = measurement_basis(theta, 0)
    p = measurement_probabilities(psi_up, basis_theta)
    probs_up.append(p[0])

ax4.plot(np.degrees(angles), probs_up, 'b-', linewidth=2, label='P(+)')
ax4.plot(np.degrees(angles), 1 - np.array(probs_up), 'r-', linewidth=2, label='P(-)')
ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Measurement angle θ (degrees from z)')
ax4.set_ylabel('Probability')
ax4.set_title('Measuring |↑⟩ along direction θ')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Theoretical curve: P(+) = cos²(θ/2)
theta_theory = np.linspace(0, np.pi, 100)
ax4.plot(np.degrees(theta_theory), np.cos(theta_theory/2)**2, 'k--',
         alpha=0.5, label='cos²(θ/2)')
ax4.legend()

plt.tight_layout()
plt.savefig('day_344_measurement_statistics.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_344_measurement_statistics.png'")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Measurement outcomes | Eigenvalues of Â |
| Born rule (discrete) | P(a) = \|⟨a\|ψ⟩\|² |
| Born rule (projection) | P(a) = ⟨ψ\|P̂ₐ\|ψ⟩ |
| Projection operator | P̂ₐ = \|a⟩⟨a\| |
| Born rule (continuous) | P(x ∈ [a,b]) = ∫ₐᵇ \|ψ(x)\|² dx |
| Probability density | ρ(x) = \|ψ(x)\|² |
| Normalization | Σₐ P(a) = 1 or ∫ ρ(x) dx = 1 |

### Main Takeaways

1. **Measurement outcomes are eigenvalues** — The only values you can observe are the eigenvalues of the observable's operator
2. **Probability = amplitude squared** — The Born rule |⟨a|ψ⟩|² bridges formalism and experiment
3. **Discrete vs. continuous** — Point probabilities for discrete spectra, probability densities for continuous
4. **Amplitudes enable interference** — Working with amplitudes (not probabilities) explains quantum phenomena
5. **Individual outcomes are random** — Quantum mechanics predicts only statistical distributions

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.1-4.2
- [ ] Read Sakurai Chapter 1.4
- [ ] Derive the Born rule from the projection operator formulation
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Write a paragraph explaining why |⟨a|ψ⟩|² (not just ⟨a|ψ⟩) gives probabilities

---

## Preview: Day 345

Tomorrow we examine **state collapse** — what happens to the quantum state *after* measurement. If we measure observable A and get result a, the state instantaneously becomes |a⟩. This is the most mysterious aspect of quantum mechanics: the measurement-induced "jump" from superposition to eigenstate.

---

*"God does not play dice with the universe."* — Albert Einstein

*"Stop telling God what to do."* — Niels Bohr

---

**Next:** [Day_345_Tuesday.md](Day_345_Tuesday.md) — State Collapse
