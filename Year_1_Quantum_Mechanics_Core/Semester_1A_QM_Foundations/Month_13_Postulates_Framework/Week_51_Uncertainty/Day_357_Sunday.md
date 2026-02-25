# Day 357: Week Review — Uncertainty and Commutators Synthesis

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Review & Practice Exam |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Comprehensive Problem Session |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Synthesis Computational Lab |

**Total Study Time:** 7 hours

---

## Week 51 Summary

This week we explored the **uncertainty principle** — one of the most profound features of quantum mechanics. We learned that certain pairs of observables cannot be simultaneously known with arbitrary precision, and this limitation is not technological but fundamental.

### Key Concepts Covered

| Day | Topic | Core Content |
|-----|-------|--------------|
| 351 | The Commutator | [Â, B̂] = ÂB̂ - B̂Â, properties, Jacobi identity |
| 352 | Canonical Commutation | [x̂, p̂] = iℏ, Poisson brackets, translations |
| 353 | Generalized Uncertainty | σₐσᵦ ≥ ½\|⟨[Â, B̂]⟩\|, Robertson proof |
| 354 | Position-Momentum | Δx·Δp ≥ ℏ/2, Gaussians, spreading |
| 355 | Energy-Time | ΔE·Δt ≥ ℏ/2, lifetimes, speed limits |
| 356 | Incompatible Observables | Complementarity, no-cloning, BB84 |

---

## Complete Formula Reference

### Commutator Algebra

| Formula | Description |
|---------|-------------|
| [Â, B̂] = ÂB̂ - B̂Â | Definition |
| [Â, B̂] = -[B̂, Â] | Antisymmetry |
| [Â, αB̂ + βĈ] = α[Â, B̂] + β[Â, Ĉ] | Linearity |
| [Â, B̂Ĉ] = [Â, B̂]Ĉ + B̂[Â, Ĉ] | Product rule |
| [Â, [B̂, Ĉ]] + [B̂, [Ĉ, Â]] + [Ĉ, [Â, B̂]] = 0 | Jacobi identity |
| {Â, B̂} = ÂB̂ + B̂Â | Anticommutator |

### Canonical Commutation Relations

| Commutator | Value |
|------------|-------|
| [x̂, p̂] | iℏ |
| [x̂ᵢ, p̂ⱼ] | iℏδᵢⱼ |
| [x̂ⁿ, p̂] | iℏnx̂ⁿ⁻¹ |
| [x̂, p̂ⁿ] | iℏnp̂ⁿ⁻¹ |
| [x̂, f(p̂)] | iℏ ∂f/∂p̂ |
| [L̂ᵢ, L̂ⱼ] | iℏεᵢⱼₖL̂ₖ |

### Uncertainty Relations

| Relation | Formula |
|----------|---------|
| Generalized (Robertson) | σₐσᵦ ≥ ½\|⟨[Â, B̂]⟩\| |
| Schrödinger | σₐ²σᵦ² ≥ ¼\|⟨[Â, B̂]⟩\|² + ¼⟨{Â', B̂'}⟩² |
| Position-momentum | Δx·Δp ≥ ℏ/2 |
| Energy-time | ΔE·Δt ≥ ℏ/2 |
| Angular momentum | σ_{Lx}σ_{Ly} ≥ (ℏ/2)\|⟨L̂z⟩\| |
| Spin | σ_{Sx}σ_{Sy} ≥ (ℏ/2)\|⟨Ŝz⟩\| |

### Special States and Operators

| Concept | Formula |
|---------|---------|
| Gaussian wave packet | ψ(x) = (2πσ²)⁻¹/⁴ exp(-(x-x₀)²/4σ²) exp(ip₀x/ℏ) |
| Translation operator | T̂(a) = exp(-ip̂a/ℏ) |
| Wave packet spreading | Δx(t) = σ₀√(1 + (ℏt/2mσ₀²)²) |
| Lifetime-linewidth | Γτ = ℏ |
| Quantum speed limit | τ_⊥ ≥ πℏ/(2ΔE) |

---

## Practice Exam

### Part A: Conceptual Questions (30 minutes)

**A1.** (5 points) Explain why time is not an operator in quantum mechanics, and how this affects the interpretation of the energy-time uncertainty relation.

**A2.** (5 points) State the no-cloning theorem and explain how it is connected to the uncertainty principle.

**A3.** (5 points) What is a Complete Set of Commuting Observables (CSCO)? Give an example for the hydrogen atom.

**A4.** (5 points) Explain the difference between preparation uncertainty and measurement disturbance.

**A5.** (5 points) Why are Gaussian wave packets called "minimum uncertainty states"?

---

### Part B: Computational Problems (60 minutes)

**B1.** (15 points) **Commutator Calculation**

Calculate the following commutators:
(a) [x̂², p̂²]
(b) [L̂x, L̂y²]
(c) [Ĥ, x̂] where Ĥ = p̂²/(2m) + V(x̂)

**B2.** (15 points) **Uncertainty Products**

For a particle in the ground state of an infinite square well (0 < x < L):
(a) Calculate Δx
(b) Calculate Δp
(c) Show that Δx·Δp > ℏ/2 and compute the ratio to the minimum

**B3.** (15 points) **Spin Measurements**

A spin-1/2 particle is in state |ψ⟩ = (|↑⟩ + 2|↓⟩)/√5.
(a) Calculate ⟨Ŝz⟩
(b) Calculate σ_{Sz}
(c) Calculate σ_{Sx}σ_{Sy} and verify the uncertainty relation

**B4.** (15 points) **Energy-Time Application**

An excited atomic state decays with half-life t₁/₂ = 10 ns.
(a) Calculate the mean lifetime τ
(b) Calculate the natural linewidth Γ in eV
(c) Calculate the frequency width Δν in Hz

---

### Part C: Derivations (30 minutes)

**C1.** (15 points) Starting from the Cauchy-Schwarz inequality:
$$|\langle f|g\rangle|^2 \leq \langle f|f\rangle\langle g|g\rangle$$
Prove the Robertson uncertainty relation:
$$\sigma_A \sigma_B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$$

**C2.** (15 points) Prove that if [Â, B̂] = c (a constant), then:
$$[\hat{A}, \hat{B}^n] = nc\hat{B}^{n-1}$$
using mathematical induction.

---

## Solutions to Practice Exam

### Part A Solutions

**A1.** Time is a parameter that labels the evolution of states (|ψ(t)⟩), not an operator on Hilbert space. This is because:
- Pauli's theorem shows a self-adjoint time operator would imply an energy spectrum unbounded below
- There are no "time eigenstates" |t⟩
- The energy-time uncertainty relates energy spread to evolution rate (Mandelstam-Tamm), not to "simultaneous measurement" like x-p

**A2.** The no-cloning theorem states that no quantum operation can create an exact copy of an arbitrary unknown quantum state: U|ψ⟩|0⟩ ≠ |ψ⟩|ψ⟩ for all |ψ⟩.

Connection to uncertainty: If cloning were possible, we could clone a state multiple times, measure x on some copies and p on others, thereby extracting more information than the uncertainty principle allows.

**A3.** A CSCO is a maximal set of mutually commuting observables whose simultaneous eigenstates are non-degenerate. For hydrogen: {Ĥ, L̂², L̂z, Ŝ², Ŝz} labels states by |n, l, m, s, mₛ⟩.

**A4.**
- **Preparation uncertainty** (Robertson): The inherent spread in observables for any quantum state, σₐσᵦ ≥ ½|⟨[Â, B̂]⟩|
- **Measurement disturbance** (Ozawa): The trade-off between measurement precision and disturbance to conjugate observables, εₐηᵦ + εₐσᵦ + σₐηᵦ ≥ ½|⟨[Â, B̂]⟩|

**A5.** Gaussian wave packets satisfy Δx·Δp = ℏ/2 (equality, not just inequality). They are the unique solutions to the minimum uncertainty condition: (x̂ - ⟨x̂⟩)|ψ⟩ = iα(p̂ - ⟨p̂⟩)|ψ⟩ with α real and positive.

---

### Part B Solutions

**B1.**
(a) Using the product rule:
$$[x̂², p̂²] = [x̂², p̂]p̂ + p̂[x̂², p̂] = 2iℏx̂p̂ + 2iℏp̂x̂ = 2iℏ(x̂p̂ + p̂x̂) = 2iℏ\{x̂, p̂\}$$

(b) $$[L̂_x, L̂_y^2] = [L̂_x, L̂_y]L̂_y + L̂_y[L̂_x, L̂_y] = iℏL̂_zL̂_y + iℏL̂_yL̂_z = iℏ\{L̂_z, L̂_y\}$$

(c) $$[Ĥ, x̂] = \frac{1}{2m}[p̂², x̂] = \frac{1}{2m}(-2iℏp̂) = -\frac{iℏp̂}{m}$$

**B2.** Ground state: ψ₁(x) = √(2/L)sin(πx/L)

(a) ⟨x⟩ = L/2, ⟨x²⟩ = L²/3 - L²/(2π²)
$$Δx = \sqrt{\frac{L²}{3} - \frac{L²}{2π²} - \frac{L²}{4}} = L\sqrt{\frac{1}{12} - \frac{1}{2π²}} ≈ 0.181L$$

(b) ⟨p⟩ = 0, ⟨p²⟩ = π²ℏ²/L²
$$Δp = \frac{πℏ}{L}$$

(c) $$Δx·Δp = 0.181L × \frac{πℏ}{L} = 0.568ℏ > \frac{ℏ}{2}$$
Ratio: 0.568/(0.5) = 1.14

**B3.**
|ψ⟩ = (|↑⟩ + 2|↓⟩)/√5

(a) ⟨Ŝz⟩ = (1/5)(ℏ/2) + (4/5)(-ℏ/2) = -3ℏ/10

(b) ⟨Ŝz²⟩ = (1/5 + 4/5)(ℏ/2)² = ℏ²/4
σ²_{Sz} = ℏ²/4 - 9ℏ²/100 = 16ℏ²/100
σ_{Sz} = 2ℏ/5

(c) For any spin-1/2 state: ⟨Ŝx²⟩ = ⟨Ŝy²⟩ = ℏ²/4
By calculation: ⟨Ŝx⟩ = 2ℏ/5, so σ²_{Sx} = ℏ²/4 - 4ℏ²/25 = 9ℏ²/100
⟨Ŝy⟩ = 0, so σ_{Sy} = ℏ/2
σ_{Sx}σ_{Sy} = (3ℏ/10)(ℏ/2) = 3ℏ²/20

Bound: (ℏ/2)|⟨Ŝz⟩| = (ℏ/2)(3ℏ/10) = 3ℏ²/20 ✓ Equality!

**B4.**
(a) τ = t₁/₂/ln(2) = 10 ns / 0.693 = 14.4 ns

(b) Γ = ℏ/τ = (6.58×10⁻¹⁶ eV·s)/(14.4×10⁻⁹ s) = 4.6×10⁻⁸ eV = 46 neV

(c) Δν = Γ/h = (4.6×10⁻⁸ eV)/(4.14×10⁻¹⁵ eV·s) = 11 MHz

---

### Part C Solutions

**C1.**
Define Â' = Â - ⟨Â⟩, B̂' = B̂ - ⟨B̂⟩. Let |f⟩ = Â'|ψ⟩, |g⟩ = B̂'|ψ⟩.

Then ⟨f|f⟩ = σₐ², ⟨g|g⟩ = σᵦ².

Cauchy-Schwarz: |⟨f|g⟩|² ≤ σₐ²σᵦ²

Now ⟨f|g⟩ = ⟨Â'B̂'⟩ = ½⟨[Â', B̂']⟩ + ½⟨{Â', B̂'}⟩

Since [Â', B̂'] = [Â, B̂], and the commutator expectation is purely imaginary while the anticommutator is real:

|⟨Â'B̂'⟩|² = ¼|⟨[Â, B̂]⟩|² + ¼⟨{Â', B̂'}⟩² ≥ ¼|⟨[Â, B̂]⟩|²

Therefore: σₐ²σᵦ² ≥ ¼|⟨[Â, B̂]⟩|²

Taking square root: σₐσᵦ ≥ ½|⟨[Â, B̂]⟩| ∎

**C2.**
Base case (n=1): [Â, B̂¹] = [Â, B̂] = c = 1·c·B̂⁰ ✓

Inductive step: Assume [Â, B̂ⁿ] = ncB̂ⁿ⁻¹

[Â, B̂ⁿ⁺¹] = [Â, B̂ⁿ·B̂] = [Â, B̂ⁿ]B̂ + B̂ⁿ[Â, B̂]
= ncB̂ⁿ⁻¹B̂ + B̂ⁿc = ncB̂ⁿ + cB̂ⁿ = (n+1)cB̂ⁿ ✓

By induction: [Â, B̂ⁿ] = ncB̂ⁿ⁻¹ for all n ≥ 1 ∎

---

## Comprehensive Computational Lab

```python
"""
Day 357 Computational Lab: Week 51 Synthesis
Quantum Mechanics Core - Year 1, Week 51
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from typing import List, Tuple, Callable

# Physical constants (natural units: hbar = 1)
hbar = 1.0

# =============================================================================
# Part 1: Complete Commutator Toolkit
# =============================================================================

print("=" * 70)
print("Part 1: Complete Commutator Toolkit")
print("=" * 70)

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculate [A, B] = AB - BA."""
    return A @ B - B @ A

def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculate {A, B} = AB + BA."""
    return A @ B + B @ A

def jacobi_sum(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Calculate [A,[B,C]] + [B,[C,A]] + [C,[A,B]]."""
    return (commutator(A, commutator(B, C)) +
            commutator(B, commutator(C, A)) +
            commutator(C, commutator(A, B)))

def verify_ccr(X: np.ndarray, P: np.ndarray, hbar: float = 1.0) -> float:
    """Verify canonical commutation relation, return relative error."""
    comm = commutator(X, P)
    expected = 1j * hbar * np.eye(X.shape[0])
    return np.linalg.norm(comm - expected) / np.linalg.norm(expected)

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Spin operators
Sx = 0.5 * sigma_x
Sy = 0.5 * sigma_y
Sz = 0.5 * sigma_z

print("\nSpin commutation relations:")
print(f"[Sx, Sy] = i*hbar*Sz: {np.allclose(commutator(Sx, Sy), 1j*hbar*Sz)}")
print(f"[Sy, Sz] = i*hbar*Sx: {np.allclose(commutator(Sy, Sz), 1j*hbar*Sx)}")
print(f"[Sz, Sx] = i*hbar*Sy: {np.allclose(commutator(Sz, Sx), 1j*hbar*Sy)}")

print(f"\nJacobi identity satisfied: {np.allclose(jacobi_sum(Sx, Sy, Sz), 0)}")

# =============================================================================
# Part 2: Uncertainty Calculation Engine
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Uncertainty Calculation Engine")
print("=" * 70)

def expectation_value(op: np.ndarray, state: np.ndarray) -> complex:
    """Calculate ⟨ψ|O|ψ⟩."""
    return np.vdot(state, op @ state)

def variance(op: np.ndarray, state: np.ndarray) -> float:
    """Calculate σ² = ⟨O²⟩ - ⟨O⟩²."""
    exp_O = expectation_value(op, state)
    exp_O2 = expectation_value(op @ op, state)
    return np.real(exp_O2 - exp_O**2)

def std_dev(op: np.ndarray, state: np.ndarray) -> float:
    """Calculate σ = √(⟨O²⟩ - ⟨O⟩²)."""
    return np.sqrt(max(0, variance(op, state)))

def uncertainty_product(A: np.ndarray, B: np.ndarray,
                        state: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate uncertainty product σ_A * σ_B and compare to bound.

    Returns:
        (product, bound, ratio)
    """
    sigma_A = std_dev(A, state)
    sigma_B = std_dev(B, state)
    product = sigma_A * sigma_B

    comm_exp = expectation_value(commutator(A, B), state)
    bound = 0.5 * np.abs(comm_exp)

    ratio = product / bound if bound > 1e-10 else float('inf')

    return product, bound, ratio

# Test on spin states
up = np.array([[1], [0]], dtype=complex)
down = np.array([[0], [1]], dtype=complex)
plus_x = (up + down) / np.sqrt(2)
plus_y = (up + 1j*down) / np.sqrt(2)

states = [
    (up, "|↑z⟩"),
    (down, "|↓z⟩"),
    (plus_x, "|+x⟩"),
    (plus_y, "|+y⟩"),
]

print("\nUncertainty products for Sx and Sy:")
print("-" * 60)
print(f"{'State':<12} {'σ_Sx·σ_Sy':<15} {'Bound':<15} {'Ratio':<10}")
print("-" * 60)

for state, name in states:
    prod, bound, ratio = uncertainty_product(Sx, Sy, state)
    print(f"{name:<12} {prod:.6f}       {bound:.6f}       {ratio:.4f}")

# =============================================================================
# Part 3: Position-Momentum Simulation
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Position-Momentum Uncertainty Verification")
print("=" * 70)

def create_gaussian(x: np.ndarray, x0: float, sigma: float,
                    p0: float = 0) -> np.ndarray:
    """Create normalized Gaussian wave packet."""
    psi = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * p0 * x)
    dx = x[1] - x[0]
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

def calculate_uncertainties(x: np.ndarray, psi: np.ndarray) -> Tuple[float, float]:
    """Calculate position and momentum uncertainties."""
    dx = x[1] - x[0]

    # Position
    prob_x = np.abs(psi)**2
    exp_x = np.sum(x * prob_x) * dx
    exp_x2 = np.sum(x**2 * prob_x) * dx
    sigma_x = np.sqrt(exp_x2 - exp_x**2)

    # Momentum via Fourier transform
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2*np.pi)
    k = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * np.pi
    p = hbar * k
    dp = p[1] - p[0] if len(p) > 1 else 1

    prob_p = np.abs(psi_p)**2
    norm_p = np.sum(prob_p) * dp
    exp_p = np.sum(p * prob_p) * dp / norm_p
    exp_p2 = np.sum(p**2 * prob_p) * dp / norm_p
    sigma_p = np.sqrt(np.abs(exp_p2 - exp_p**2))

    return sigma_x, sigma_p

# Test Gaussian wave packets
N = 2048
L = 40.0
x = np.linspace(-L/2, L/2, N)

sigmas = [0.5, 1.0, 2.0, 3.0, 4.0]
print("\nGaussian wave packet uncertainties:")
print("-" * 55)
print(f"{'σ_wf':<10} {'Δx':<12} {'Δp':<12} {'Δx·Δp':<12} {'Ratio':<10}")
print("-" * 55)

for sigma_wf in sigmas:
    psi = create_gaussian(x, 0, sigma_wf)
    sigma_x, sigma_p = calculate_uncertainties(x, psi)
    product = sigma_x * sigma_p
    ratio = product / (hbar / 2)
    print(f"{sigma_wf:<10} {sigma_x:<12.4f} {sigma_p:<12.4f} {product:<12.4f} {ratio:<10.4f}")

# =============================================================================
# Part 4: Energy-Time Uncertainty Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Energy-Time Uncertainty Visualization")
print("=" * 70)

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))

# Panel 1: Commutator structure
ax1 = fig.add_subplot(2, 3, 1)
ops = [Sx, Sy, Sz]
labels = ['Sx', 'Sy', 'Sz']
comm_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        comm_matrix[i, j] = np.linalg.norm(commutator(ops[i], ops[j]))

im1 = ax1.imshow(comm_matrix, cmap='Reds')
ax1.set_xticks(range(3))
ax1.set_yticks(range(3))
ax1.set_xticklabels(labels)
ax1.set_yticklabels(labels)
ax1.set_title('|[Sᵢ, Sⱼ]| (Spin-1/2)', fontsize=12)
plt.colorbar(im1, ax=ax1)

# Panel 2: Uncertainty product vs angle
ax2 = fig.add_subplot(2, 3, 2)
theta_vals = np.linspace(0.01, np.pi-0.01, 100)
products = []
bounds = []

for theta in theta_vals:
    state = np.array([[np.cos(theta/2)], [np.sin(theta/2)]], dtype=complex)
    sigma_Sx = std_dev(Sx, state)
    sigma_Sy = std_dev(Sy, state)
    exp_Sz = np.abs(np.real(expectation_value(Sz, state)))

    products.append(sigma_Sx * sigma_Sy)
    bounds.append(0.5 * exp_Sz)

ax2.plot(np.degrees(theta_vals), products, 'b-', linewidth=2, label='σ_Sx·σ_Sy')
ax2.plot(np.degrees(theta_vals), bounds, 'r--', linewidth=2, label='½|⟨Sz⟩|')
ax2.fill_between(np.degrees(theta_vals), 0, bounds, alpha=0.3, color='red')
ax2.set_xlabel('θ (degrees)', fontsize=12)
ax2.set_ylabel('Uncertainty Product', fontsize=12)
ax2.set_title('Spin Uncertainty vs. Bloch Angle', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Gaussian wave functions
ax3 = fig.add_subplot(2, 3, 3)
sigma_vals = [0.5, 1.0, 2.0, 4.0]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(sigma_vals)))

for sigma, color in zip(sigma_vals, colors):
    psi = create_gaussian(x, 0, sigma)
    ax3.plot(x, np.abs(psi)**2, color=color, linewidth=2, label=f'σ = {sigma}')

ax3.set_xlim(-10, 10)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('|ψ(x)|²', fontsize=12)
ax3.set_title('Gaussian Wave Packets', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Uncertainty trade-off
ax4 = fig.add_subplot(2, 3, 4)
sigma_x_range = np.linspace(0.3, 3, 100)
sigma_p_min = hbar / (2 * sigma_x_range)

ax4.fill_between(sigma_x_range, 0, sigma_p_min, alpha=0.3, color='red',
                 label='Forbidden')
ax4.plot(sigma_x_range, sigma_p_min, 'r-', linewidth=2, label='Δx·Δp = ℏ/2')
ax4.scatter([1/np.sqrt(2)], [1/np.sqrt(2)], s=100, c='blue', zorder=5,
            label='Coherent state')
ax4.set_xlabel('Δx', fontsize=12)
ax4.set_ylabel('Δp', fontsize=12)
ax4.set_title('Position-Momentum Trade-off', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Panel 5: Lifetime-linewidth
ax5 = fig.add_subplot(2, 3, 5)
hbar_SI = 1.055e-34
eV = 1.602e-19
tau_range = np.logspace(-25, -6, 100)
gamma_range = hbar_SI / tau_range / eV

ax5.loglog(tau_range, gamma_range, 'b-', linewidth=2)
ax5.set_xlabel('Lifetime τ (s)', fontsize=12)
ax5.set_ylabel('Width Γ (eV)', fontsize=12)
ax5.set_title('Γτ = ℏ', fontsize=12)
ax5.grid(True, alpha=0.3, which='both')

# Mark some particles
particles = [
    (1.6e-9, 'H 2p→1s'),
    (2.2e-6, 'Muon'),
    (8.4e-17, 'π⁰'),
]
for tau, name in particles:
    gamma = hbar_SI / tau / eV
    ax5.scatter([tau], [gamma], s=80, zorder=5)
    ax5.annotate(name, (tau, gamma), xytext=(5, 5), textcoords='offset points')

# Panel 6: Wave packet spreading
ax6 = fig.add_subplot(2, 3, 6)

def spreading_width(t, sigma0, hbar=1, m=1):
    return sigma0 * np.sqrt(1 + (hbar * t / (2 * m * sigma0**2))**2)

t_range = np.linspace(0, 10, 100)
for sigma0 in [0.5, 1.0, 2.0]:
    widths = [spreading_width(t, sigma0) for t in t_range]
    tau = 2 * 1 * sigma0**2 / 1  # m=1, hbar=1
    ax6.plot(t_range, widths, linewidth=2, label=f'σ₀ = {sigma0}, τ = {tau:.1f}')

ax6.set_xlabel('Time t', fontsize=12)
ax6.set_ylabel('Δx(t)', fontsize=12)
ax6.set_title('Wave Packet Spreading', fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_357_week_review.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_357_week_review.png'")

# =============================================================================
# Part 5: Complete Example: Hydrogen-like System
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Complete Example - Angular Momentum Uncertainty")
print("=" * 70)

# Spin-1 matrices (l=1)
Lx_1 = (1/np.sqrt(2)) * np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=complex)

Ly_1 = (1/np.sqrt(2)) * np.array([
    [0, -1j, 0],
    [1j, 0, -1j],
    [0, 1j, 0]
], dtype=complex)

Lz_1 = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1]
], dtype=complex)

L2 = Lx_1 @ Lx_1 + Ly_1 @ Ly_1 + Lz_1 @ Lz_1

print("\nSpin-1 Angular Momentum:")
print(f"[Lx, Ly] = i*Lz: {np.allclose(commutator(Lx_1, Ly_1), 1j*Lz_1)}")
print(f"[L², Lz] = 0: {np.allclose(commutator(L2, Lz_1), 0)}")
print(f"L² eigenvalue: {np.real(L2[0,0])} = l(l+1) = 2 for l=1 ✓")

# States |1,m⟩
m1 = np.array([[1], [0], [0]], dtype=complex)   # m = +1
m0 = np.array([[0], [1], [0]], dtype=complex)   # m = 0
mm1 = np.array([[0], [0], [1]], dtype=complex)  # m = -1

print("\nUncertainty relations for angular momentum states:")
print("-" * 60)

for state, name in [(m1, "|1,+1⟩"), (m0, "|1,0⟩"), (mm1, "|1,-1⟩")]:
    sigma_Lx = std_dev(Lx_1, state)
    sigma_Ly = std_dev(Ly_1, state)
    exp_Lz = np.real(expectation_value(Lz_1, state))

    product = sigma_Lx * sigma_Ly
    bound = 0.5 * np.abs(exp_Lz)

    print(f"\n{name}:")
    print(f"  ⟨Lz⟩ = {exp_Lz:.2f}")
    print(f"  σ_Lx·σ_Ly = {product:.4f}")
    print(f"  Bound = {bound:.4f}")
    print(f"  Satisfied: {product >= bound - 1e-10}")

# =============================================================================
# Part 6: Summary Statistics
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Week 51 Summary Statistics")
print("=" * 70)

print("""
WEEK 51: UNCERTAINTY AND COMMUTATORS

Key Results Verified:
--------------------
1. Spin commutation relations: [Si, Sj] = iℏεijk Sk
2. Jacobi identity: [A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0
3. Robertson uncertainty: σA·σB ≥ ½|⟨[A,B]⟩|
4. Position-momentum: Δx·Δp ≥ ℏ/2 (Gaussians achieve equality)
5. Energy-time: Γτ = ℏ (lifetime-linewidth)
6. Angular momentum: [L², Lz] = 0 (compatible), [Lx, Ly] ≠ 0 (incompatible)

Physical Insights:
-----------------
• Non-commutativity is the mathematical origin of uncertainty
• Uncertainty is intrinsic to quantum states, not measurement limitations
• Minimum uncertainty states (Gaussians) are "most classical"
• Complementarity: precise knowledge of one variable precludes another
• No-cloning protects the uncertainty principle

Applications:
-------------
• Quantum cryptography (BB84)
• Spectral line widths
• Particle lifetimes
• Quantum speed limits
• Wave packet dynamics
""")

print("\n" + "=" * 70)
print("Week 51 Review Complete!")
print("=" * 70)
```

---

## Daily Checklist

- [ ] Complete the practice exam (allow 2 hours)
- [ ] Review solutions and identify weak areas
- [ ] Run the comprehensive computational lab
- [ ] Create personal summary notes for the week
- [ ] Attempt additional problems in weak areas
- [ ] Review key derivations (Robertson, CCR consequences)
- [ ] Prepare questions for Week 52

---

## Key Takeaways from Week 51

1. **Commutators encode physics:** [Â, B̂] = 0 means simultaneous knowledge is possible; [Â, B̂] ≠ 0 means a fundamental trade-off exists.

2. **[x̂, p̂] = iℏ is foundational:** This single equation implies the uncertainty principle, wave-particle duality, and much of quantum mechanics.

3. **Uncertainty is preparation limitation:** The Robertson relation bounds how precisely we can prepare states, not how precisely we can measure.

4. **Gaussians are special:** Only Gaussian wave packets saturate Δx·Δp = ℏ/2.

5. **Energy-time is different:** Time is a parameter, so ΔE·Δt relates to evolution rate, not simultaneous measurement.

6. **Incompatibility enables security:** Quantum cryptography exploits the impossibility of measuring conjugate variables.

---

## Preview: Week 52

Next week we begin **Time Evolution** — how quantum states change according to the Schrödinger equation. We'll see that the Hamiltonian Ĥ generates time translations, just as momentum p̂ generates spatial translations. This will connect everything we've learned about operators to the dynamics of quantum systems.

**Topics:**
- The Schrödinger equation
- Time evolution operator U(t)
- Stationary states and time dependence
- Ehrenfest's theorem
- Heisenberg vs. Schrödinger pictures

---

*"The uncertainty principle has been the central point of debate about the interpretation of quantum mechanics since its formulation in 1927."* — John Bell

---

**Week 51 Complete!**

**Next:** [Week_52_Time_Evolution](../Week_52_Time_Evolution/README.md)
