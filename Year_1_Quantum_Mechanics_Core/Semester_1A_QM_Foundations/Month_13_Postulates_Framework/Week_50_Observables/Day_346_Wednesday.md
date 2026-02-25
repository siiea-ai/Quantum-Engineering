# Day 346: Expectation Values — The Quantum Average

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Expectation Values & Variance |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 346, you will be able to:

1. Calculate expectation values using ⟨Â⟩ = ⟨ψ|Â|ψ⟩
2. Compute variance and standard deviation of observables
3. Interpret expectation values as experimental averages
4. Relate quantum expectation values to classical limits
5. Calculate expectation values in different representations
6. Apply uncertainty concepts to prepare for the uncertainty principle

---

## Core Content

### 1. What Do We Actually Measure?

Yesterday we learned that individual measurements yield random eigenvalues. But physics is about **reproducible, predictable** quantities. How do we extract meaningful information?

**The answer:** Measure the same observable on many identically prepared systems and take the average.

**Definition (Expectation Value):**

> The **expectation value** of observable A in state |ψ⟩ is:

$$\boxed{⟨\hat{A}⟩ = ⟨ψ|\hat{A}|ψ⟩}$$

This is the average result of measuring A on an ensemble of systems all prepared in state |ψ⟩.

---

### 2. Derivation from the Born Rule

The expectation value formula emerges naturally from probability theory.

**For discrete spectrum:**

If Â has eigenvalues {aₙ} with eigenstates {|aₙ⟩}, and we expand |ψ⟩ = Σₙ cₙ|aₙ⟩:

$$⟨\hat{A}⟩ = \sum_n a_n \cdot P(a_n) = \sum_n a_n |c_n|^2$$

**Proof that this equals ⟨ψ|Â|ψ⟩:**

$$⟨ψ|\hat{A}|ψ⟩ = \left(\sum_m c_m^* ⟨a_m|\right) \hat{A} \left(\sum_n c_n |a_n⟩\right)$$

$$= \sum_{m,n} c_m^* c_n ⟨a_m|\hat{A}|a_n⟩ = \sum_{m,n} c_m^* c_n \cdot a_n \cdot δ_{mn}$$

$$= \sum_n |c_n|^2 a_n = \sum_n a_n P(a_n) \quad ✓$$

**For continuous spectrum (position):**

$$⟨\hat{x}⟩ = \int_{-∞}^{∞} x |\psi(x)|^2 dx = \int_{-∞}^{∞} \psi^*(x) \cdot x \cdot \psi(x) dx$$

This is ⟨ψ|x̂|ψ⟩ in position representation.

---

### 3. Properties of Expectation Values

**Linearity:**

$$⟨α\hat{A} + β\hat{B}⟩ = α⟨\hat{A}⟩ + β⟨\hat{B}⟩$$

**Real values:** For Hermitian operators (observables):

$$⟨\hat{A}⟩^* = ⟨ψ|\hat{A}|ψ⟩^* = ⟨ψ|\hat{A}^†|ψ⟩ = ⟨ψ|\hat{A}|ψ⟩ = ⟨\hat{A}⟩$$

So ⟨Â⟩ is always real for observables. ✓

**Bounded by eigenvalue range:**

$$a_{\min} ≤ ⟨\hat{A}⟩ ≤ a_{\max}$$

The expectation value lies within the range of eigenvalues.

**Eigenstate expectation:**

If |ψ⟩ = |a⟩ is an eigenstate:

$$⟨\hat{A}⟩ = ⟨a|\hat{A}|a⟩ = a⟨a|a⟩ = a$$

The expectation value equals the eigenvalue with certainty.

---

### 4. Variance and Standard Deviation

The expectation value tells us the average, but how spread out are the measurement results?

**Definition (Variance):**

$$\boxed{(\Delta A)^2 = ⟨(\hat{A} - ⟨\hat{A}⟩)^2⟩ = ⟨\hat{A}^2⟩ - ⟨\hat{A}⟩^2}$$

**Definition (Standard Deviation / Uncertainty):**

$$\boxed{\Delta A = \sqrt{⟨\hat{A}^2⟩ - ⟨\hat{A}⟩^2}}$$

**Derivation of the variance formula:**

$$(\Delta A)^2 = ⟨(\hat{A} - ⟨\hat{A}⟩)^2⟩ = ⟨\hat{A}^2 - 2\hat{A}⟨\hat{A}⟩ + ⟨\hat{A}⟩^2⟩$$

$$= ⟨\hat{A}^2⟩ - 2⟨\hat{A}⟩⟨\hat{A}⟩ + ⟨\hat{A}⟩^2 = ⟨\hat{A}^2⟩ - ⟨\hat{A}⟩^2$$

**Physical interpretation:**
- ΔA = 0: All measurements give the same result (eigenstate)
- ΔA > 0: Results are spread around the mean

---

### 5. Eigenstates Have Zero Uncertainty

**Theorem:** If |ψ⟩ is an eigenstate of Â with eigenvalue a, then ΔA = 0.

**Proof:**

$$⟨\hat{A}⟩ = a$$

$$⟨\hat{A}^2⟩ = ⟨a|\hat{A}^2|a⟩ = ⟨a|\hat{A} \cdot \hat{A}|a⟩ = ⟨a|a \cdot \hat{A}|a⟩ = a⟨a|\hat{A}|a⟩ = a^2$$

$$(\Delta A)^2 = ⟨\hat{A}^2⟩ - ⟨\hat{A}⟩^2 = a^2 - a^2 = 0 \quad ✓$$

**Physical meaning:** If you're in an eigenstate of A, measuring A always gives the same value — no uncertainty!

**Converse:** If ΔA = 0, then |ψ⟩ must be an eigenstate of Â.

---

### 6. Expectation Values in Different Representations

The expectation value formula adapts to different representations:

**Position representation:**

$$⟨\hat{A}⟩ = \int_{-∞}^{∞} ψ^*(x) \hat{A} \psi(x) dx$$

where Â acts on ψ(x) in its position-representation form.

**Examples:**

$$⟨\hat{x}⟩ = \int_{-∞}^{∞} \psi^*(x) \cdot x \cdot \psi(x) dx$$

$$⟨\hat{p}⟩ = \int_{-∞}^{∞} \psi^*(x) \left(-i\hbar\frac{d}{dx}\right) \psi(x) dx$$

$$⟨\hat{T}⟩ = ⟨\frac{\hat{p}^2}{2m}⟩ = \int_{-∞}^{∞} \psi^*(x) \left(-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}\right) \psi(x) dx$$

**Momentum representation:**

$$⟨\hat{p}⟩ = \int_{-∞}^{∞} φ^*(p) \cdot p \cdot φ(p) dp$$

$$⟨\hat{x}⟩ = \int_{-∞}^{∞} φ^*(p) \left(i\hbar\frac{d}{dp}\right) φ(p) dp$$

---

### 7. Connection to Classical Mechanics

Expectation values provide the bridge between quantum and classical physics.

**Ehrenfest's Theorem (preview):**

$$\frac{d⟨\hat{x}⟩}{dt} = \frac{⟨\hat{p}⟩}{m}, \quad \frac{d⟨\hat{p}⟩}{dt} = -⟨\frac{dV}{dx}⟩$$

These look like Newton's equations! Classical mechanics emerges when:
1. Wave packets are narrow (small Δx, Δp)
2. The potential varies slowly over the wave packet width
3. ⟨dV/dx⟩ ≈ dV/d⟨x⟩

**Correspondence principle:** In the classical limit, quantum expectation values follow classical trajectories.

---

### 8. Important Examples

#### Example: Spin-1/2 Expectation Values

For qubit state |ψ⟩ = α|↑⟩ + β|↓⟩:

$$⟨\hat{S}_z⟩ = ⟨ψ|\hat{S}_z|ψ⟩ = (α^*⟨↑| + β^*⟨↓|)\hat{S}_z(α|↑⟩ + β|↓⟩)$$

Using Ŝz|↑⟩ = (ℏ/2)|↑⟩ and Ŝz|↓⟩ = -(ℏ/2)|↓⟩:

$$⟨\hat{S}_z⟩ = |α|^2 \frac{\hbar}{2} + |β|^2 \left(-\frac{\hbar}{2}\right) = \frac{\hbar}{2}(|α|^2 - |β|^2)$$

**Special cases:**
- |ψ⟩ = |↑⟩: ⟨Sz⟩ = +ℏ/2
- |ψ⟩ = |↓⟩: ⟨Sz⟩ = -ℏ/2
- |ψ⟩ = |+⟩ = (|↑⟩+|↓⟩)/√2: ⟨Sz⟩ = 0

#### Example: Harmonic Oscillator Ground State

The ground state is:

$$ψ_0(x) = \left(\frac{mω}{πℏ}\right)^{1/4} e^{-mωx^2/(2ℏ)}$$

**Position expectation:**

$$⟨\hat{x}⟩ = \int_{-∞}^{∞} x |ψ_0(x)|^2 dx = 0$$

(The integrand is odd.)

**Position variance:**

$$⟨\hat{x}^2⟩ = \int_{-∞}^{∞} x^2 |ψ_0(x)|^2 dx = \frac{\hbar}{2mω}$$

$$\Delta x = \sqrt{⟨\hat{x}^2⟩ - ⟨\hat{x}⟩^2} = \sqrt{\frac{\hbar}{2mω}}$$

Similarly, Δp = √(ℏmω/2), giving:

$$\Delta x \cdot \Delta p = \frac{\hbar}{2}$$

This is the minimum uncertainty state!

---

## Quantum Computing Connection

### Expectation Value Estimation

Quantum computers cannot directly output ⟨Â⟩. Instead, they estimate it statistically:

1. Prepare state |ψ⟩
2. Measure in eigenbasis of Â
3. Record outcome aₖ
4. Repeat N times
5. Compute: ⟨Â⟩ ≈ (1/N) Σₖ aₖ

**Statistical error:** σ/√N where σ = ΔA

### Measuring Pauli Expectation Values

For a qubit, the Pauli expectation values characterize the state completely:

$$⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ \quad \leftrightarrow \quad \text{Bloch vector}$$

**Measuring ⟨σz⟩:**
- Measure in computational basis
- ⟨σz⟩ = P(0) - P(1)

**Measuring ⟨σx⟩:**
- Apply Hadamard H
- Measure in computational basis
- ⟨σx⟩ = P(0) - P(1)

**Measuring ⟨σy⟩:**
- Apply S†H (rotate Y to Z)
- Measure in computational basis
- ⟨σy⟩ = P(0) - P(1)

### Variational Quantum Eigensolver (VQE)

VQE finds ground state energy by minimizing:

$$E(θ) = ⟨ψ(θ)|\hat{H}|ψ(θ)⟩$$

**Algorithm:**
1. Prepare parameterized state |ψ(θ)⟩
2. Estimate ⟨H⟩ via measurements
3. Use classical optimizer to minimize E(θ)
4. The minimum is the ground state energy estimate

This is a hybrid quantum-classical algorithm, central to NISQ-era quantum computing.

### Quantum Phase Estimation

QPE estimates eigenvalues by encoding them in expectation values:

If Û|u⟩ = e^{2πiφ}|u⟩, QPE outputs an estimate of φ.

The precision scales as:

$$\Delta φ ∝ \frac{1}{2^n}$$

where n is the number of ancilla qubits.

---

## Worked Examples

### Example 1: Spin Expectation Values and Variance

**Problem:** For |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5, calculate:
(a) ⟨Sz⟩
(b) ⟨Sz²⟩
(c) ΔSz

**Solution:**

First, identify: α = 3/5, β = 4i/5

(a) **Expectation value:**

$$⟨\hat{S}_z⟩ = |α|^2 \cdot \frac{ℏ}{2} + |β|^2 \cdot \left(-\frac{ℏ}{2}\right)$$

$$= \frac{9}{25} \cdot \frac{ℏ}{2} - \frac{16}{25} \cdot \frac{ℏ}{2} = -\frac{7ℏ}{50} = \boxed{-0.14ℏ}$$

(b) **Expectation of Sz²:**

Note: Ŝz²|↑⟩ = (ℏ/2)²|↑⟩ and Ŝz²|↓⟩ = (ℏ/2)²|↓⟩

So Ŝz² = (ℏ²/4)Î for spin-1/2.

$$⟨\hat{S}_z^2⟩ = \frac{ℏ^2}{4} \cdot 1 = \boxed{\frac{ℏ^2}{4}}$$

(c) **Standard deviation:**

$$(\Delta S_z)^2 = ⟨\hat{S}_z^2⟩ - ⟨\hat{S}_z⟩^2 = \frac{ℏ^2}{4} - \frac{49ℏ^2}{2500}$$

$$= ℏ^2\left(\frac{625 - 49}{2500}\right) = \frac{576ℏ^2}{2500} = \frac{144ℏ^2}{625}$$

$$\Delta S_z = \frac{12ℏ}{25} = \boxed{0.48ℏ}$$

---

### Example 2: Position Expectation for Particle in Box

**Problem:** For a particle in infinite square well (0 ≤ x ≤ L) in the ground state:
$$ψ_1(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{πx}{L}\right)$$

Calculate ⟨x⟩ and ⟨x²⟩.

**Solution:**

**⟨x⟩:**

$$⟨\hat{x}⟩ = \frac{2}{L}\int_0^L x \sin^2\left(\frac{πx}{L}\right) dx$$

Using $\sin^2θ = (1 - \cos 2θ)/2$:

$$⟨\hat{x}⟩ = \frac{1}{L}\int_0^L x dx - \frac{1}{L}\int_0^L x \cos\left(\frac{2πx}{L}\right) dx$$

The first integral: $\frac{1}{L} \cdot \frac{L^2}{2} = \frac{L}{2}$

The second integral: Use integration by parts, which gives 0 (periodic function).

$$\boxed{⟨\hat{x}⟩ = \frac{L}{2}}$$

(The particle is centered in the well — expected by symmetry!)

**⟨x²⟩:**

$$⟨\hat{x}^2⟩ = \frac{2}{L}\int_0^L x^2 \sin^2\left(\frac{πx}{L}\right) dx$$

This evaluates to:

$$\boxed{⟨\hat{x}^2⟩ = \frac{L^2}{3} - \frac{L^2}{2π^2} = L^2\left(\frac{1}{3} - \frac{1}{2π^2}\right)}$$

**Variance:**

$$(\Delta x)^2 = ⟨\hat{x}^2⟩ - ⟨\hat{x}⟩^2 = L^2\left(\frac{1}{3} - \frac{1}{2π^2}\right) - \frac{L^2}{4}$$

$$= L^2\left(\frac{1}{12} - \frac{1}{2π^2}\right) ≈ 0.0326 L^2$$

$$\boxed{\Delta x ≈ 0.181 L}$$

---

### Example 3: Gaussian Wave Packet

**Problem:** A free particle has wave function:
$$ψ(x) = \left(\frac{1}{2πσ^2}\right)^{1/4} e^{-(x-x_0)^2/(4σ^2)} e^{ip_0 x/ℏ}$$

Calculate ⟨x⟩, ⟨p⟩, Δx, and Δp.

**Solution:**

**⟨x⟩:**

The probability density |ψ(x)|² is a Gaussian centered at x₀:

$$|ψ(x)|^2 = \sqrt{\frac{1}{2πσ^2}} e^{-(x-x_0)^2/(2σ^2)}$$

$$\boxed{⟨\hat{x}⟩ = x_0}$$

**Δx:**

For a Gaussian with width parameter σ:

$$\boxed{\Delta x = σ}$$

**⟨p⟩:**

$$⟨\hat{p}⟩ = \int_{-∞}^{∞} ψ^*(x) \left(-iℏ\frac{d}{dx}\right) ψ(x) dx$$

The factor e^{ip₀x/ℏ} contributes momentum p₀:

$$\boxed{⟨\hat{p}⟩ = p_0}$$

**Δp:**

For a Gaussian wave packet:

$$\boxed{\Delta p = \frac{ℏ}{2σ}}$$

**Uncertainty product:**

$$\Delta x \cdot \Delta p = σ \cdot \frac{ℏ}{2σ} = \frac{ℏ}{2}$$

The Gaussian wave packet is a **minimum uncertainty state**!

---

## Practice Problems

### Level 1: Direct Application

1. **Basic expectation:** For |ψ⟩ = (|0⟩ + |1⟩)/√2, calculate ⟨σz⟩ and ⟨σx⟩.
   (Pauli matrices: σz = diag(1,-1), σx = off-diagonal 1s)

2. **Spin variance:** For |ψ⟩ = |↑⟩, calculate ΔSx.
   (Hint: Express |↑⟩ in Sx eigenbasis)

3. **Harmonic oscillator:** For the first excited state |1⟩ of the harmonic oscillator, what are ⟨x⟩ and ⟨p⟩?
   (Hint: Use symmetry arguments)

### Level 2: Intermediate

4. **General spin-1/2:** For |ψ⟩ = cos(θ/2)|↑⟩ + e^{iφ}sin(θ/2)|↓⟩:
   (a) Calculate ⟨σx⟩, ⟨σy⟩, ⟨σz⟩
   (b) Show that ⟨σx⟩² + ⟨σy⟩² + ⟨σz⟩² = 1
   (c) What is the geometric interpretation?

5. **Position-momentum relation:** Show that for any state:
   $$⟨\hat{p}⟩ = m\frac{d⟨\hat{x}⟩}{dt}$$
   when there are no explicit time-dependence in the Hamiltonian.

6. **Superposition energy:** A harmonic oscillator is in state |ψ⟩ = (|0⟩ + |1⟩)/√2.
   (a) Calculate ⟨H⟩.
   (b) Calculate ΔE.
   (c) Compare to the energy eigenvalues.

### Level 3: Challenging

7. **Virial theorem:** For a particle in potential V(x) ∝ x^n, prove:
   $$2⟨\hat{T}⟩ = n⟨\hat{V}⟩$$
   where T is kinetic energy.

8. **Momentum expectation:** Show that:
   $$⟨\hat{p}⟩ = m\frac{d⟨\hat{x}⟩}{dt} = \frac{im}{ℏ}⟨[\hat{H}, \hat{x}]⟩$$

9. **Research:** The uncertainty ΔA depends on the state |ψ⟩. For a given observable A, what state minimizes ΔA? What state maximizes it (if bounded)? Relate this to coherent states.

---

## Computational Lab

### Objective
Calculate expectation values and variances numerically, and visualize quantum uncertainties.

```python
"""
Day 346 Computational Lab: Expectation Values
Quantum Mechanics Core - Year 1, Week 50
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import hermite
from typing import Callable, Tuple

# =============================================================================
# Part 1: Matrix Expectation Values (Finite Dimensions)
# =============================================================================

print("=" * 70)
print("Part 1: Matrix Expectation Values")
print("=" * 70)

def expectation_value(state: np.ndarray, operator: np.ndarray) -> complex:
    """
    Calculate expectation value ⟨ψ|Â|ψ⟩.

    Parameters:
        state: Column vector |ψ⟩
        operator: Matrix representation of Â

    Returns:
        Expectation value (complex, but real for Hermitian operators)
    """
    return (state.conj().T @ operator @ state)[0, 0]

def variance(state: np.ndarray, operator: np.ndarray) -> float:
    """
    Calculate variance (ΔA)² = ⟨Â²⟩ - ⟨Â⟩².
    """
    exp_A = expectation_value(state, operator)
    exp_A2 = expectation_value(state, operator @ operator)
    return (exp_A2 - exp_A**2).real

def std_dev(state: np.ndarray, operator: np.ndarray) -> float:
    """Calculate standard deviation ΔA."""
    return np.sqrt(variance(state, operator))

# Define Pauli matrices
I = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Test state: |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5
psi = np.array([[3], [4j]], dtype=complex) / 5

print(f"\nState: |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5")
print(f"|ψ⟩ = {psi.flatten()}")

# Verify normalization
norm = np.vdot(psi, psi)
print(f"\nNormalization: ⟨ψ|ψ⟩ = {norm.real:.6f}")

# Calculate expectation values
print("\nExpectation values (in units of ℏ/2):")
print(f"⟨σx⟩ = {expectation_value(psi, sigma_x).real:.6f}")
print(f"⟨σy⟩ = {expectation_value(psi, sigma_y).real:.6f}")
print(f"⟨σz⟩ = {expectation_value(psi, sigma_z).real:.6f}")

# Calculate standard deviations
print("\nStandard deviations (in units of ℏ/2):")
print(f"Δσx = {std_dev(psi, sigma_x):.6f}")
print(f"Δσy = {std_dev(psi, sigma_y):.6f}")
print(f"Δσz = {std_dev(psi, sigma_z):.6f}")

# Verify: For spin-1/2, σᵢ² = I, so ⟨σᵢ²⟩ = 1 for all i
print("\nVerification: ⟨σᵢ²⟩ = 1 for all Pauli matrices")
for name, op in [('x', sigma_x), ('y', sigma_y), ('z', sigma_z)]:
    exp_sq = expectation_value(psi, op @ op)
    print(f"⟨σ{name}²⟩ = {exp_sq.real:.6f}")

# =============================================================================
# Part 2: Bloch Vector from Expectation Values
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Bloch Vector Representation")
print("=" * 70)

def bloch_vector(state: np.ndarray) -> np.ndarray:
    """
    Calculate Bloch vector (⟨σx⟩, ⟨σy⟩, ⟨σz⟩) from state.
    """
    return np.array([
        expectation_value(state, sigma_x).real,
        expectation_value(state, sigma_y).real,
        expectation_value(state, sigma_z).real
    ])

# Our test state
r = bloch_vector(psi)
print(f"\nBloch vector for |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5:")
print(f"r = ({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})")
print(f"|r| = {np.linalg.norm(r):.6f} (should be 1 for pure state)")

# Compare with various states
states = {
    "|↑⟩": np.array([[1], [0]], dtype=complex),
    "|↓⟩": np.array([[0], [1]], dtype=complex),
    "|+⟩": np.array([[1], [1]], dtype=complex) / np.sqrt(2),
    "|-⟩": np.array([[1], [-1]], dtype=complex) / np.sqrt(2),
    "|+y⟩": np.array([[1], [1j]], dtype=complex) / np.sqrt(2),
    "|-y⟩": np.array([[1], [-1j]], dtype=complex) / np.sqrt(2),
}

print("\nBloch vectors for various states:")
for name, state in states.items():
    r = bloch_vector(state)
    print(f"{name:6s}: ({r[0]:6.3f}, {r[1]:6.3f}, {r[2]:6.3f})")

# =============================================================================
# Part 3: Position Space Expectation Values
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Position Space Expectation Values")
print("=" * 70)

def position_expectation(psi_func: Callable, x_min: float, x_max: float,
                          power: int = 1) -> float:
    """
    Calculate ⟨x^n⟩ = ∫ ψ*(x) x^n ψ(x) dx
    """
    def integrand(x):
        psi_val = psi_func(x)
        return np.abs(psi_val)**2 * x**power

    result, _ = integrate.quad(integrand, x_min, x_max)
    return result

def momentum_expectation(psi_func: Callable, dpsi_func: Callable,
                          x_min: float, x_max: float) -> float:
    """
    Calculate ⟨p⟩ = -iℏ ∫ ψ*(x) dψ/dx dx (returns in units of ℏ)
    """
    def integrand(x):
        return np.real(np.conj(psi_func(x)) * (-1j) * dpsi_func(x))

    result, _ = integrate.quad(integrand, x_min, x_max)
    return result

# Particle in a box ground state
L = 1.0  # Box length

def psi_box(x):
    if 0 < x < L:
        return np.sqrt(2/L) * np.sin(np.pi * x / L)
    return 0.0

# Position expectation
exp_x = position_expectation(psi_box, 0, L, power=1)
exp_x2 = position_expectation(psi_box, 0, L, power=2)
delta_x = np.sqrt(exp_x2 - exp_x**2)

print(f"\nParticle in box (L={L}):")
print(f"⟨x⟩ = {exp_x:.6f} (expected: {L/2:.6f})")
print(f"⟨x²⟩ = {exp_x2:.6f} (expected: {L**2*(1/3 - 1/(2*np.pi**2)):.6f})")
print(f"Δx = {delta_x:.6f}")

# =============================================================================
# Part 4: Harmonic Oscillator Expectation Values
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Harmonic Oscillator Expectation Values")
print("=" * 70)

def harmonic_oscillator_wf(x: np.ndarray, n: int, omega: float = 1.0,
                            m: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """
    Calculate harmonic oscillator wave function ψ_n(x).
    """
    xi = np.sqrt(m * omega / hbar) * x
    norm = (m * omega / (np.pi * hbar))**0.25 / np.sqrt(2**n * np.math.factorial(n))
    H_n = hermite(n)
    return norm * H_n(xi) * np.exp(-xi**2 / 2)

# Parameters (natural units: m=ω=ℏ=1)
m, omega, hbar = 1.0, 1.0, 1.0

# Ground state (n=0)
x = np.linspace(-5, 5, 1000)
psi_0 = harmonic_oscillator_wf(x, 0, omega, m, hbar)
psi_1 = harmonic_oscillator_wf(x, 1, omega, m, hbar)

# Superposition state: (|0⟩ + |1⟩)/√2
psi_super = (psi_0 + psi_1) / np.sqrt(2)

# Numerical integration for expectation values
dx = x[1] - x[0]

# Ground state
exp_x_0 = np.sum(np.abs(psi_0)**2 * x) * dx
exp_x2_0 = np.sum(np.abs(psi_0)**2 * x**2) * dx
delta_x_0 = np.sqrt(exp_x2_0 - exp_x_0**2)

print(f"\nGround state |0⟩:")
print(f"⟨x⟩ = {exp_x_0:.6f} (expected: 0)")
print(f"⟨x²⟩ = {exp_x2_0:.6f} (expected: {hbar/(2*m*omega):.6f})")
print(f"Δx = {delta_x_0:.6f} (expected: {np.sqrt(hbar/(2*m*omega)):.6f})")

# First excited state
exp_x_1 = np.sum(np.abs(psi_1)**2 * x) * dx
exp_x2_1 = np.sum(np.abs(psi_1)**2 * x**2) * dx
delta_x_1 = np.sqrt(exp_x2_1 - exp_x_1**2)

print(f"\nFirst excited state |1⟩:")
print(f"⟨x⟩ = {exp_x_1:.6f} (expected: 0)")
print(f"⟨x²⟩ = {exp_x2_1:.6f} (expected: {3*hbar/(2*m*omega):.6f})")
print(f"Δx = {delta_x_1:.6f}")

# Superposition state
exp_x_s = np.sum(np.abs(psi_super)**2 * x) * dx
exp_x2_s = np.sum(np.abs(psi_super)**2 * x**2) * dx
delta_x_s = np.sqrt(exp_x2_s - exp_x_s**2)

print(f"\nSuperposition (|0⟩+|1⟩)/√2:")
print(f"⟨x⟩ = {exp_x_s:.6f}")
print(f"Δx = {delta_x_s:.6f}")

# =============================================================================
# Part 5: Energy Expectation Values
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Energy Expectation Values (Harmonic Oscillator)")
print("=" * 70)

# Energy eigenvalues: E_n = ℏω(n + 1/2)
E_0 = hbar * omega * 0.5
E_1 = hbar * omega * 1.5

# For superposition |ψ⟩ = (|0⟩ + |1⟩)/√2
# ⟨H⟩ = |c_0|² E_0 + |c_1|² E_1 = 0.5 * E_0 + 0.5 * E_1

c0 = 1/np.sqrt(2)
c1 = 1/np.sqrt(2)

exp_E = np.abs(c0)**2 * E_0 + np.abs(c1)**2 * E_1
exp_E2 = np.abs(c0)**2 * E_0**2 + np.abs(c1)**2 * E_1**2
delta_E = np.sqrt(exp_E2 - exp_E**2)

print(f"\nSuperposition (|0⟩+|1⟩)/√2:")
print(f"E_0 = {E_0:.4f}, E_1 = {E_1:.4f}")
print(f"⟨H⟩ = {exp_E:.4f} (average of E_0 and E_1)")
print(f"⟨H²⟩ = {exp_E2:.4f}")
print(f"ΔE = {delta_E:.4f}")
print(f"(E_1 - E_0)/2 = {(E_1 - E_0)/2:.4f} (matches ΔE for equal superposition)")

# =============================================================================
# Part 6: Statistical Verification
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Statistical Verification of Expectation Values")
print("=" * 70)

def measure_and_average(state: np.ndarray, operator: np.ndarray,
                         n_measurements: int = 10000) -> Tuple[float, float, float]:
    """
    Simulate measurements and compute average, comparing to expectation value.

    Returns:
        sample_mean: Average of measurement outcomes
        sample_std: Sample standard deviation
        theoretical_exp: ⟨Â⟩ from formula
    """
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(operator)

    # Calculate probabilities
    probs = []
    for i in range(len(eigenvalues)):
        eigvec = eigenvectors[:, i:i+1]
        prob = np.abs(np.vdot(eigvec, state))**2
        probs.append(prob)

    # Simulate measurements
    outcomes = np.random.choice(eigenvalues, size=n_measurements, p=probs)

    sample_mean = np.mean(outcomes)
    sample_std = np.std(outcomes) / np.sqrt(n_measurements)  # Standard error

    theoretical_exp = expectation_value(state, operator).real

    return sample_mean, sample_std, theoretical_exp

# Test on our state with σz
n_meas = 10000
sample_avg, sample_err, theory = measure_and_average(psi, sigma_z, n_meas)

print(f"\nMeasuring σz on |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5:")
print(f"Theoretical ⟨σz⟩ = {theory:.6f}")
print(f"Sample average ({n_meas} measurements) = {sample_avg:.6f} ± {sample_err:.6f}")

# Test with more measurements to see convergence
print("\nConvergence with number of measurements:")
for n in [100, 1000, 10000, 100000]:
    avg, err, _ = measure_and_average(psi, sigma_z, n)
    print(f"N = {n:6d}: ⟨σz⟩ = {avg:8.5f} ± {err:.5f}")

# =============================================================================
# Part 7: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Harmonic oscillator wave functions and expectation values
ax1 = axes[0, 0]
ax1.fill_between(x, 0, np.abs(psi_0)**2, alpha=0.3, label='|ψ₀|²')
ax1.fill_between(x, 0, np.abs(psi_1)**2, alpha=0.3, label='|ψ₁|²')
ax1.axvline(x=exp_x_0, color='C0', linestyle='--', label=f'⟨x⟩₀ = {exp_x_0:.2f}')
ax1.axvline(x=exp_x_1, color='C1', linestyle='--', label=f'⟨x⟩₁ = {exp_x_1:.2f}')
# Mark ±Δx regions
ax1.axvspan(exp_x_0 - delta_x_0, exp_x_0 + delta_x_0, alpha=0.1, color='C0')
ax1.set_xlabel('x')
ax1.set_ylabel('Probability density')
ax1.set_title('Harmonic Oscillator Probability Densities')
ax1.legend()
ax1.set_xlim(-4, 4)

# Panel 2: Bloch sphere (2D projection)
ax2 = axes[0, 1]
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)

# Plot Bloch vectors for various states
for name, state in states.items():
    r = bloch_vector(state)
    ax2.arrow(0, 0, r[0]*0.9, r[2]*0.9, head_width=0.08, head_length=0.05,
              fc='C0', ec='C0', alpha=0.7)
    ax2.annotate(name, (r[0], r[2]), fontsize=9)

# Our test state
r_test = bloch_vector(psi)
ax2.arrow(0, 0, r_test[0]*0.9, r_test[2]*0.9, head_width=0.08, head_length=0.05,
          fc='red', ec='red', linewidth=2)
ax2.annotate('|ψ⟩', (r_test[0], r_test[2]), fontsize=10, color='red')

ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)
ax2.set_aspect('equal')
ax2.set_xlabel('⟨σx⟩')
ax2.set_ylabel('⟨σz⟩')
ax2.set_title('Bloch Vectors (x-z plane)')
ax2.grid(True, alpha=0.3)

# Panel 3: Convergence of sample mean to expectation value
ax3 = axes[1, 0]
n_values = np.logspace(1, 5, 30, dtype=int)
sample_means = []
errors = []

for n in n_values:
    avg, err, _ = measure_and_average(psi, sigma_z, int(n))
    sample_means.append(avg)
    errors.append(np.abs(avg - theory))

ax3.loglog(n_values, errors, 'bo-', markersize=4, linewidth=1,
           label='|Sample mean - ⟨σz⟩|')
ax3.loglog(n_values, 1/np.sqrt(n_values), 'r--', linewidth=2,
           label='1/√N (theory)')
ax3.set_xlabel('Number of measurements N')
ax3.set_ylabel('Absolute error')
ax3.set_title('Convergence to Expectation Value')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Uncertainty (ΔA) vs state
ax4 = axes[1, 1]

# Vary θ in |ψ(θ)⟩ = cos(θ/2)|↑⟩ + sin(θ/2)|↓⟩
thetas = np.linspace(0, 2*np.pi, 100)
delta_z = []
delta_x = []
delta_y = []

for th in thetas:
    state = np.array([[np.cos(th/2)], [np.sin(th/2)]], dtype=complex)
    delta_z.append(std_dev(state, sigma_z))
    delta_x.append(std_dev(state, sigma_x))
    delta_y.append(std_dev(state, sigma_y))

ax4.plot(np.degrees(thetas), delta_z, 'b-', linewidth=2, label='Δσz')
ax4.plot(np.degrees(thetas), delta_x, 'r-', linewidth=2, label='Δσx')
ax4.plot(np.degrees(thetas), delta_y, 'g-', linewidth=2, label='Δσy')
ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('θ (degrees)')
ax4.set_ylabel('Standard deviation')
ax4.set_title('Spin Uncertainties vs. State Parameter θ')
ax4.legend()
ax4.set_xlim(0, 360)
ax4.set_ylim(-0.1, 1.2)
ax4.grid(True, alpha=0.3)
ax4.set_xticks([0, 90, 180, 270, 360])

plt.tight_layout()
plt.savefig('day_346_expectation_values.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_346_expectation_values.png'")

# =============================================================================
# Part 8: Uncertainty Product
# =============================================================================

print("\n" + "=" * 70)
print("Part 8: Uncertainty Product Preview")
print("=" * 70)

# For spin-1/2, check uncertainty relation: ΔSx · ΔSy ≥ |⟨Sz⟩|/2

print("\nUncertainty product for |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5:")
delta_sx = std_dev(psi, sigma_x)
delta_sy = std_dev(psi, sigma_y)
exp_sz = expectation_value(psi, sigma_z).real

print(f"Δσx = {delta_sx:.6f}")
print(f"Δσy = {delta_sy:.6f}")
print(f"Δσx · Δσy = {delta_sx * delta_sy:.6f}")
print(f"|⟨σz⟩|/2 = {np.abs(exp_sz)/2:.6f}")
print(f"Uncertainty relation satisfied: {delta_sx * delta_sy >= np.abs(exp_sz)/2 - 1e-10}")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Expectation value | ⟨Â⟩ = ⟨ψ\|Â\|ψ⟩ |
| Variance | (ΔA)² = ⟨Â²⟩ - ⟨Â⟩² |
| Standard deviation | ΔA = √(⟨Â²⟩ - ⟨Â⟩²) |
| Eigenstate expectation | ⟨a\|Â\|a⟩ = a |
| Eigenstate uncertainty | ΔA = 0 for eigenstate |
| Position expectation | ⟨x⟩ = ∫ψ*(x) x ψ(x) dx |
| Momentum expectation | ⟨p⟩ = ∫ψ*(x)(-iℏ d/dx)ψ(x) dx |

### Main Takeaways

1. **Expectation value is the average** — ⟨Â⟩ = ⟨ψ|Â|ψ⟩ predicts the mean of many measurements
2. **Variance measures spread** — (ΔA)² = ⟨Â²⟩ - ⟨Â⟩² quantifies quantum uncertainty
3. **Eigenstates have zero uncertainty** — ΔA = 0 if and only if |ψ⟩ is an eigenstate of Â
4. **Expectation values are real** — For Hermitian operators (observables)
5. **Statistical interpretation** — Sample averages converge to ⟨Â⟩ as N → ∞
6. **Classical correspondence** — Expectation values satisfy classical-like equations (Ehrenfest)

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.4
- [ ] Read Sakurai Chapter 1.4-1.5
- [ ] Derive ⟨Â⟩ = ⟨ψ|Â|ψ⟩ from the Born rule
- [ ] Prove that (ΔA)² = ⟨Â²⟩ - ⟨Â⟩²
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Calculate ⟨x⟩, ⟨p⟩, Δx, Δp for a Gaussian wave packet by hand

---

## Preview: Day 347

Tomorrow we explore **compatible observables** — when can two observables be measured simultaneously without disturbance? The key is the commutator [Â, B̂]. When it vanishes, A and B share eigenstates and can both have definite values. When it doesn't, the uncertainty principle applies!

---

*"In quantum mechanics, the observable is not the property itself, but the average of what we measure."* — Werner Heisenberg

---

**Next:** [Day_347_Thursday.md](Day_347_Thursday.md) — Compatible Observables
