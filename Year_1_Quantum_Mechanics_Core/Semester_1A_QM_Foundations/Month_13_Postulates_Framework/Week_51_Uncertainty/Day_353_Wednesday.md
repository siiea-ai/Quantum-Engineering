# Day 353: Generalized Uncertainty Principle — Proof and Applications

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Robertson Uncertainty Relation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 353, you will be able to:

1. State and prove the Robertson uncertainty relation
2. Compute uncertainty products for arbitrary observable pairs
3. Apply the generalized uncertainty principle to specific systems
4. Explain the role of expectation values in the uncertainty bound
5. Derive specific uncertainty relations from the general formula
6. Understand when equality holds (minimum uncertainty states)

---

## Core Content

### 1. The Robertson Uncertainty Relation

The **generalized uncertainty principle** (Robertson, 1929) states:

$$\boxed{\sigma_A \sigma_B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|}$$

where:
- σₐ = √(⟨Â²⟩ - ⟨Â⟩²) is the standard deviation of A
- σᵦ = √(⟨B̂²⟩ - ⟨B̂⟩²) is the standard deviation of B
- [Â, B̂] = ÂB̂ - B̂Â is the commutator
- ⟨·⟩ denotes the expectation value in state |ψ⟩

**Key insight:** The lower bound depends on the **state** through ⟨[Â, B̂]⟩.

---

### 2. Proof of the Generalized Uncertainty Principle

This proof is a beautiful application of the Cauchy-Schwarz inequality.

#### Step 1: Define Shifted Operators

Let:
$$\hat{A}' = \hat{A} - \langle\hat{A}\rangle\hat{I}, \quad \hat{B}' = \hat{B} - \langle\hat{B}\rangle\hat{I}$$

These have zero expectation: ⟨Â'⟩ = ⟨B̂'⟩ = 0.

Note that:
$$[\hat{A}', \hat{B}'] = [\hat{A}, \hat{B}]$$

(constants commute with everything).

The variances are:
$$\sigma_A^2 = \langle\hat{A}'^2\rangle, \quad \sigma_B^2 = \langle\hat{B}'^2\rangle$$

#### Step 2: Create States

Define:
$$|f\rangle = \hat{A}'|\psi\rangle, \quad |g\rangle = \hat{B}'|\psi\rangle$$

Then:
$$\langle f|f\rangle = \langle\psi|\hat{A}'^\dagger\hat{A}'|\psi\rangle = \langle\psi|\hat{A}'^2|\psi\rangle = \sigma_A^2$$

(since Â' is Hermitian: Â'† = Â').

Similarly: ⟨g|g⟩ = σᵦ².

#### Step 3: Apply Cauchy-Schwarz

The Cauchy-Schwarz inequality states:
$$|\langle f|g\rangle|^2 \leq \langle f|f\rangle\langle g|g\rangle$$

Therefore:
$$|\langle\psi|\hat{A}'\hat{B}'|\psi\rangle|^2 \leq \sigma_A^2 \sigma_B^2$$

#### Step 4: Decompose the Inner Product

Write Â'B̂' in terms of commutator and anticommutator:
$$\hat{A}'\hat{B}' = \frac{1}{2}[\hat{A}', \hat{B}'] + \frac{1}{2}\{\hat{A}', \hat{B}'\}$$

where {Â', B̂'} = Â'B̂' + B̂'Â' is the anticommutator.

Taking expectation values:
$$\langle\hat{A}'\hat{B}'\rangle = \frac{1}{2}\langle[\hat{A}', \hat{B}']\rangle + \frac{1}{2}\langle\{\hat{A}', \hat{B}'\}\rangle$$

Note:
- [Â', B̂'] is anti-Hermitian (since [Â, B̂]† = -[Â, B̂]), so ⟨[Â', B̂']⟩ is purely imaginary
- {Â', B̂'} is Hermitian, so ⟨{Â', B̂'}⟩ is purely real

Therefore:
$$|\langle\hat{A}'\hat{B}'\rangle|^2 = \frac{1}{4}|\langle[\hat{A}', \hat{B}']\rangle|^2 + \frac{1}{4}|\langle\{\hat{A}', \hat{B}'\}\rangle|^2$$

#### Step 5: Apply the Bound

Since both terms are non-negative:
$$|\langle\hat{A}'\hat{B}'\rangle|^2 \geq \frac{1}{4}|\langle[\hat{A}', \hat{B}']\rangle|^2 = \frac{1}{4}|\langle[\hat{A}, \hat{B}]\rangle|^2$$

Combined with Step 3:
$$\sigma_A^2 \sigma_B^2 \geq \frac{1}{4}|\langle[\hat{A}, \hat{B}]\rangle|^2$$

Taking the square root:

$$\boxed{\sigma_A \sigma_B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|}$$

**Q.E.D.**

---

### 3. The Schrödinger Uncertainty Relation

A stronger form includes both the commutator and anticommutator:

$$\boxed{\sigma_A^2 \sigma_B^2 \geq \frac{1}{4}|\langle[\hat{A}, \hat{B}]\rangle|^2 + \frac{1}{4}\left(\langle\{\hat{A}', \hat{B}'\}\rangle\right)^2}$$

where Â' = Â - ⟨Â⟩ and B̂' = B̂ - ⟨B̂⟩.

This is the **Schrödinger uncertainty relation** (1930), which is tighter than Robertson's.

---

### 4. When Does Equality Hold?

Equality in Robertson's relation occurs when:

1. **Cauchy-Schwarz saturated:** |f⟩ = λ|g⟩ for some complex λ
   $$\hat{A}'|\psi\rangle = \lambda\hat{B}'|\psi\rangle$$

2. **Anticommutator term vanishes:** ⟨{Â', B̂'}⟩ = 0

For position-momentum: λ = iα with α real, giving Gaussian wave packets.

**Minimum uncertainty states** satisfy:
$$(\hat{x} - \langle\hat{x}\rangle)|\psi\rangle = i\alpha(\hat{p} - \langle\hat{p}\rangle)|\psi\rangle$$

with α > 0 for normalizable solutions.

---

### 5. Applications to Specific Pairs

#### Position and Momentum

$$[\hat{x}, \hat{p}] = i\hbar$$

Since iℏ is a constant (not an operator):
$$|\langle[\hat{x}, \hat{p}]\rangle| = |i\hbar| = \hbar$$

Therefore:
$$\boxed{\sigma_x \sigma_p \geq \frac{\hbar}{2}}$$

This is Heisenberg's famous relation. The bound is **state-independent**.

#### Angular Momentum Components

$$[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$$

The uncertainty relation is:
$$\sigma_{L_x} \sigma_{L_y} \geq \frac{\hbar}{2}|\langle\hat{L}_z\rangle|$$

The bound is **state-dependent**! If ⟨L̂z⟩ = 0, there's no lower bound.

#### Number and Phase

For a quantum harmonic oscillator, define the number operator N̂ = â†â and phase operator φ̂ (carefully defined).

$$[\hat{N}, \hat{\phi}] \approx i$$

Leading to:
$$\sigma_N \sigma_\phi \geq \frac{1}{2}$$

This is crucial for quantum optics and coherent states.

---

### 6. Interpretation: What the Uncertainty Principle Really Says

**What it says:**

1. For any quantum state, there is a fundamental limit to how precisely we can know both A and B simultaneously
2. The limit depends on the non-commutativity of the operators
3. For conjugate variables like x and p, this limit is ℏ/2

**What it does NOT say:**

1. It does NOT say measurement disturbs the system (that's a separate phenomenon)
2. It does NOT say we can't measure A precisely (we can, but then B spreads)
3. It does NOT limit a single measurement's precision

**The uncertainty is intrinsic to quantum states, not to measurement apparatus.**

---

### 7. Connection to Experimental Observations

**Stern-Gerlach experiment:**

For a spin-1/2 particle in state |↑z⟩:
- σ_{Sz} = 0 (eigenstate)
- ⟨L̂z⟩ = ℏ/2

The uncertainty relation [Ŝx, Ŝy] = iℏŜz gives:
$$\sigma_{S_x}\sigma_{S_y} \geq \frac{\hbar}{2}|\langle\hat{S}_z\rangle| = \frac{\hbar^2}{4}$$

For |↑z⟩: σ_{Sx} = σ_{Sy} = ℏ/2, so:
$$\sigma_{S_x}\sigma_{S_y} = \frac{\hbar^2}{4}$$

Equality! The spin eigenstate is a minimum uncertainty state for Sx and Sy.

---

## Physical Interpretation

### The Deep Meaning of σₐσᵦ ≥ ½|⟨[Â, B̂]⟩|

1. **Non-commutativity quantifies incompatibility:** The larger |[Â, B̂]|, the more incompatible the observables.

2. **Expectation value matters:** The bound depends on the state. For some states, incompatible observables can both have small uncertainties.

3. **Trade-off is fundamental:** You cannot prepare a state with arbitrarily small uncertainties in both A and B if they don't commute.

4. **Quantum information perspective:** Uncertainty limits the information that can be extracted about conjugate observables.

### Robertson vs. Heisenberg

| Aspect | Robertson (1929) | Heisenberg (1927) |
|--------|------------------|-------------------|
| Form | σₐσᵦ ≥ ½\|⟨[Â, B̂]⟩\| | ΔxΔp ~ ℏ |
| Rigor | Mathematical theorem | Heuristic argument |
| Generality | Any operators | Position-momentum |
| Bound | State-dependent | Order of magnitude |

---

## Worked Examples

### Example 1: Spin-1/2 Uncertainty

**Problem:** A spin-1/2 particle is in state |ψ⟩ = cos(θ/2)|↑⟩ + sin(θ/2)|↓⟩.
Calculate σ_{Sx}σ_{Sy} and compare to the uncertainty bound.

**Solution:**

First, calculate ⟨Ŝz⟩:
$$\langle\hat{S}_z\rangle = \cos^2(\theta/2)\frac{\hbar}{2} + \sin^2(\theta/2)\left(-\frac{\hbar}{2}\right) = \frac{\hbar}{2}\cos\theta$$

The uncertainty bound is:
$$\sigma_{S_x}\sigma_{S_y} \geq \frac{\hbar}{2}|\langle\hat{S}_z\rangle| = \frac{\hbar^2}{4}|\cos\theta|$$

Now calculate the actual uncertainties. For this state:
$$\langle\hat{S}_x\rangle = \frac{\hbar}{2}\sin\theta, \quad \langle\hat{S}_y\rangle = 0$$

$$\langle\hat{S}_x^2\rangle = \langle\hat{S}_y^2\rangle = \frac{\hbar^2}{4}$$ (always for spin-1/2)

Therefore:
$$\sigma_{S_x}^2 = \frac{\hbar^2}{4} - \frac{\hbar^2}{4}\sin^2\theta = \frac{\hbar^2}{4}\cos^2\theta$$
$$\sigma_{S_y}^2 = \frac{\hbar^2}{4}$$

$$\sigma_{S_x}\sigma_{S_y} = \frac{\hbar^2}{4}|\cos\theta|$$

$$\boxed{\sigma_{S_x}\sigma_{S_y} = \frac{\hbar^2}{4}|\cos\theta| = \frac{\hbar}{2}|\langle\hat{S}_z\rangle|}$$

Equality holds! All spin-1/2 coherent states are minimum uncertainty states.

---

### Example 2: Harmonic Oscillator Ground State

**Problem:** Verify that the harmonic oscillator ground state is a minimum uncertainty state.

**Solution:**

The ground state wave function is:
$$\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} e^{-m\omega x^2/(2\hbar)}$$

**Calculate ⟨x⟩ and ⟨x²⟩:**

By symmetry: ⟨x⟩ = 0

$$\langle x^2\rangle = \int_{-\infty}^{\infty} x^2|\psi_0(x)|^2 dx = \frac{\hbar}{2m\omega}$$

Therefore: σₓ² = ⟨x²⟩ - ⟨x⟩² = ℏ/(2mω)

$$\sigma_x = \sqrt{\frac{\hbar}{2m\omega}}$$

**Calculate ⟨p⟩ and ⟨p²⟩:**

By symmetry: ⟨p⟩ = 0

For the ground state: ⟨p²⟩ = mℏω/2 (from virial theorem or direct calculation)

$$\sigma_p = \sqrt{\frac{m\hbar\omega}{2}}$$

**Check the product:**

$$\sigma_x\sigma_p = \sqrt{\frac{\hbar}{2m\omega}} \cdot \sqrt{\frac{m\hbar\omega}{2}} = \sqrt{\frac{\hbar^2}{4}} = \frac{\hbar}{2}$$

$$\boxed{\sigma_x\sigma_p = \frac{\hbar}{2}}$$

Equality holds! The ground state achieves the minimum possible uncertainty.

---

### Example 3: Angular Momentum Uncertainty

**Problem:** For a particle in state |l=1, m=0⟩, calculate σ_{Lx}σ_{Ly}.

**Solution:**

For |l=1, m=0⟩:
$$\langle\hat{L}_z\rangle = m\hbar = 0$$

The uncertainty bound is:
$$\sigma_{L_x}\sigma_{L_y} \geq \frac{\hbar}{2}|\langle\hat{L}_z\rangle| = 0$$

So there's no constraint from the uncertainty principle!

Let's calculate the actual uncertainties.

$$\langle\hat{L}_x\rangle = \langle\hat{L}_y\rangle = 0$$ (by symmetry)

$$\langle\hat{L}_x^2\rangle = \langle\hat{L}_y^2\rangle = \frac{1}{2}\left(\langle\hat{L}^2\rangle - \langle\hat{L}_z^2\rangle\right)$$

For |l=1, m=0⟩:
- ⟨L̂²⟩ = l(l+1)ℏ² = 2ℏ²
- ⟨L̂z²⟩ = m²ℏ² = 0

$$\langle\hat{L}_x^2\rangle = \langle\hat{L}_y^2\rangle = \frac{1}{2}(2\hbar^2 - 0) = \hbar^2$$

Therefore:
$$\sigma_{L_x} = \sigma_{L_y} = \hbar$$

$$\boxed{\sigma_{L_x}\sigma_{L_y} = \hbar^2 \geq 0}$$

The uncertainty is large despite the bound being zero. The bound is not tight here.

---

## Practice Problems

### Level 1: Direct Application

1. **Position-momentum:** For a state with ⟨x⟩ = 0, ⟨x²⟩ = 4 (in appropriate units) and ⟨p⟩ = 0, ⟨p²⟩ = 1, calculate σₓσₚ and compare to ℏ/2.

2. **Spin verification:** Show that for any spin-1/2 state: ⟨Ŝₓ²⟩ = ⟨Ŝᵧ²⟩ = ⟨Ŝᵤ²⟩ = ℏ²/4.

3. **State-dependent bound:** For the state |+x⟩ = (|↑⟩ + |↓⟩)/√2, calculate:
   (a) ⟨Ŝz⟩
   (b) The bound on σ_{Sx}σ_{Sy}
   (c) The actual product σ_{Sx}σ_{Sy}

### Level 2: Intermediate

4. **Squeezed state:** A state has σₓ = ℏ/(4mω) and σₚ = 2mωℏ for a harmonic oscillator.
   (a) Verify the uncertainty principle is satisfied.
   (b) Compare to the ground state uncertainties.
   (c) Describe why this is called "squeezed."

5. **Uncertainty sum rule:** Prove that for spin-1/2:
   $$\sigma_{S_x}^2 + \sigma_{S_y}^2 + \sigma_{S_z}^2 = \frac{\hbar^2}{2}$$
   for any pure state.

6. **Mixed angular momentum:** For a state that is an equal superposition |ψ⟩ = (|l,m⟩ + |l,m'⟩)/√2, derive the uncertainty relation for Lₓ and Lᵧ.

### Level 3: Challenging

7. **Minimum uncertainty states:** Find the wave function ψ(x) that satisfies:
   $$(\hat{x} - x_0)|\psi\rangle = i\alpha(\hat{p} - p_0)|\psi\rangle$$
   with α > 0, x₀, p₀ real. This is the most general minimum uncertainty state.

8. **Entropic uncertainty:** The entropic uncertainty relation is:
   $$H(X) + H(P) \geq \ln(\pi e\hbar)$$
   where H(X) = -∫|ψ(x)|²ln|ψ(x)|² dx is the Shannon entropy.
   Verify this for a Gaussian wave packet.

9. **Operator bound:** Prove the operator inequality:
   $$\hat{A}^2 \hat{B}^2 \geq \frac{1}{4}[\hat{A}, \hat{B}]^2$$
   is NOT generally true by finding a counterexample. Why does the uncertainty relation still hold?

---

## Computational Lab

### Objective
Verify the generalized uncertainty principle numerically and explore minimum uncertainty states.

```python
"""
Day 353 Computational Lab: Generalized Uncertainty Principle
Quantum Mechanics Core - Year 1, Week 51
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from typing import Tuple, List

# =============================================================================
# Part 1: Spin-1/2 Uncertainty Relations
# =============================================================================

print("=" * 70)
print("Part 1: Spin-1/2 Uncertainty Relations")
print("=" * 70)

# Pauli matrices (in units of hbar/2)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

hbar = 1  # Natural units

# Spin operators (in units of hbar)
Sx = hbar/2 * sigma_x
Sy = hbar/2 * sigma_y
Sz = hbar/2 * sigma_z

def expectation_value(op: np.ndarray, state: np.ndarray) -> complex:
    """Calculate ⟨ψ|O|ψ⟩."""
    return np.vdot(state, op @ state)

def variance(op: np.ndarray, state: np.ndarray) -> float:
    """Calculate σ² = ⟨O²⟩ - ⟨O⟩²."""
    exp_O = expectation_value(op, state)
    exp_O2 = expectation_value(op @ op, state)
    return np.real(exp_O2 - exp_O**2)

def std_dev(op: np.ndarray, state: np.ndarray) -> float:
    """Calculate standard deviation σ."""
    return np.sqrt(variance(op, state))

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculate [A, B] = AB - BA."""
    return A @ B - B @ A

# Test states
up = np.array([[1], [0]], dtype=complex)
down = np.array([[0], [1]], dtype=complex)
plus_x = (up + down) / np.sqrt(2)
plus_y = (up + 1j*down) / np.sqrt(2)

print("\n--- Test: |↑z⟩ state ---")
sigma_Sx = std_dev(Sx, up)
sigma_Sy = std_dev(Sy, up)
exp_Sz = np.real(expectation_value(Sz, up))

product = sigma_Sx * sigma_Sy
bound = 0.5 * np.abs(exp_Sz)

print(f"σ(Sx) = {sigma_Sx:.4f}")
print(f"σ(Sy) = {sigma_Sy:.4f}")
print(f"⟨Sz⟩ = {exp_Sz:.4f}")
print(f"σ(Sx)·σ(Sy) = {product:.4f}")
print(f"Bound = (1/2)|⟨Sz⟩| = {bound:.4f}")
print(f"Satisfies uncertainty: {product >= bound - 1e-10}")

# Scan over all states on Bloch sphere
print("\n--- Scanning Bloch sphere ---")
n_theta = 50
n_phi = 50
thetas = np.linspace(0, np.pi, n_theta)
phis = np.linspace(0, 2*np.pi, n_phi)

products = []
bounds = []

for theta in thetas:
    for phi in phis:
        state = np.array([[np.cos(theta/2)],
                          [np.exp(1j*phi)*np.sin(theta/2)]], dtype=complex)

        sigma_Sx = std_dev(Sx, state)
        sigma_Sy = std_dev(Sy, state)
        exp_Sz = np.real(expectation_value(Sz, state))

        products.append(sigma_Sx * sigma_Sy)
        bounds.append(0.5 * np.abs(exp_Sz))

products = np.array(products)
bounds = np.array(bounds)

print(f"All states satisfy uncertainty principle: {np.all(products >= bounds - 1e-10)}")
print(f"Minimum ratio (product/bound): {np.min(products[bounds > 1e-10] / bounds[bounds > 1e-10]):.4f}")
print(f"(1.0 means equality is achieved)")

# =============================================================================
# Part 2: Position-Momentum Uncertainty
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Position-Momentum Uncertainty")
print("=" * 70)

def gaussian_wavefunction(x: np.ndarray, x0: float, sigma: float,
                          p0: float = 0) -> np.ndarray:
    """
    Gaussian wave packet centered at x0 with width sigma and momentum p0.
    ψ(x) = (1/(2πσ²))^(1/4) exp(-(x-x0)²/(4σ²)) exp(ip0x/ℏ)
    """
    norm = (1 / (2 * np.pi * sigma**2))**(0.25)
    psi = norm * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * p0 * x / hbar)
    return psi

def calculate_xp_uncertainties(x: np.ndarray, psi: np.ndarray) -> Tuple[float, float]:
    """Calculate σx and σp from wave function."""
    dx = x[1] - x[0]

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    psi = psi / norm

    # Position uncertainty
    prob = np.abs(psi)**2
    exp_x = np.sum(x * prob) * dx
    exp_x2 = np.sum(x**2 * prob) * dx
    sigma_x = np.sqrt(exp_x2 - exp_x**2)

    # Momentum uncertainty (via Fourier transform)
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2*np.pi)
    k = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * np.pi
    p = hbar * k

    prob_p = np.abs(psi_p)**2
    dp = p[1] - p[0] if len(p) > 1 else 1
    norm_p = np.sum(prob_p) * dp

    exp_p = np.sum(p * prob_p) * dp / norm_p
    exp_p2 = np.sum(p**2 * prob_p) * dp / norm_p
    sigma_p = np.sqrt(np.abs(exp_p2 - exp_p**2))

    return sigma_x, sigma_p

# Create grid
N = 2048
L = 40.0
x = np.linspace(-L/2, L/2, N)

# Test different Gaussian widths
sigmas = [0.5, 1.0, 2.0, 4.0]
print("\nGaussian wave packet uncertainties:")
print("  σ_wf     σ_x       σ_p       σ_x·σ_p   Bound (ℏ/2)")
print("-" * 55)

for sigma_wf in sigmas:
    psi = gaussian_wavefunction(x, 0, sigma_wf, 0)
    sigma_x, sigma_p = calculate_xp_uncertainties(x, psi)
    product = sigma_x * sigma_p
    bound = hbar / 2

    print(f"  {sigma_wf:.1f}      {sigma_x:.4f}    {sigma_p:.4f}    {product:.4f}     {bound:.4f}")

print("\nGaussian wave packets achieve the minimum uncertainty (equality)!")

# =============================================================================
# Part 3: Squeezed States
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Squeezed States")
print("=" * 70)

def squeezed_wavefunction(x: np.ndarray, squeeze_factor: float) -> np.ndarray:
    """
    Squeezed vacuum state with squeeze factor r.
    σ_x = exp(-r)/√2, σ_p = exp(r)/√2 (in natural units where ℏ=m=ω=1)
    """
    sigma_x = np.exp(-squeeze_factor) / np.sqrt(2)
    norm = (1 / (2 * np.pi * sigma_x**2))**(0.25)
    psi = norm * np.exp(-x**2 / (4 * sigma_x**2))
    return psi

squeeze_factors = [-1.0, -0.5, 0, 0.5, 1.0]
print("\nSqueezed state uncertainties (r = squeeze factor):")
print("  r       σ_x       σ_p       σ_x·σ_p")
print("-" * 45)

fig, ax = plt.subplots(figsize=(10, 6))

for r in squeeze_factors:
    psi = squeezed_wavefunction(x, r)
    sigma_x, sigma_p = calculate_xp_uncertainties(x, psi)
    product = sigma_x * sigma_p

    print(f"  {r:+.1f}     {sigma_x:.4f}    {sigma_p:.4f}    {product:.4f}")

    # Plot probability density
    prob = np.abs(psi)**2
    ax.plot(x, prob, label=f'r = {r:+.1f}', linewidth=2)

ax.set_xlim(-5, 5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('|ψ(x)|²', fontsize=12)
ax.set_title('Squeezed States: Position Probability Density', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_353_squeezed_states.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_353_squeezed_states.png'")

# =============================================================================
# Part 4: Violation Detection (should never happen!)
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Random State Testing")
print("=" * 70)

def random_spin_state() -> np.ndarray:
    """Generate random normalized spin-1/2 state."""
    state = np.random.randn(2) + 1j * np.random.randn(2)
    state = state.reshape(2, 1)
    return state / np.linalg.norm(state)

n_tests = 10000
violations = 0
min_ratio = float('inf')

for _ in range(n_tests):
    state = random_spin_state()

    sigma_Sx = std_dev(Sx, state)
    sigma_Sy = std_dev(Sy, state)
    exp_Sz = np.abs(np.real(expectation_value(Sz, state)))

    product = sigma_Sx * sigma_Sy
    bound = 0.5 * exp_Sz

    if bound > 1e-10:
        ratio = product / bound
        min_ratio = min(min_ratio, ratio)
        if product < bound - 1e-10:
            violations += 1

print(f"Tested {n_tests} random spin-1/2 states")
print(f"Violations found: {violations}")
print(f"Minimum ratio (product/bound): {min_ratio:.6f}")
print("The uncertainty principle is never violated!")

# =============================================================================
# Part 5: Visualization of Uncertainty Regions
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Uncertainty Region Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Spin uncertainty
ax1 = axes[0]
theta_vals = np.linspace(0.01, np.pi - 0.01, 100)

sigma_Sx_vals = []
sigma_Sy_vals = []
bound_vals = []

for theta in theta_vals:
    state = np.array([[np.cos(theta/2)], [np.sin(theta/2)]], dtype=complex)

    sigma_Sx = std_dev(Sx, state)
    sigma_Sy = std_dev(Sy, state)
    exp_Sz = np.abs(np.real(expectation_value(Sz, state)))

    sigma_Sx_vals.append(sigma_Sx)
    sigma_Sy_vals.append(sigma_Sy)
    bound_vals.append(0.5 * exp_Sz)

sigma_Sx_vals = np.array(sigma_Sx_vals)
sigma_Sy_vals = np.array(sigma_Sy_vals)
bound_vals = np.array(bound_vals)

ax1.plot(np.degrees(theta_vals), sigma_Sx_vals * sigma_Sy_vals, 'b-',
         linewidth=2, label='σ(Sx)·σ(Sy)')
ax1.plot(np.degrees(theta_vals), bound_vals, 'r--',
         linewidth=2, label='(1/2)|⟨Sz⟩|')
ax1.fill_between(np.degrees(theta_vals), 0, bound_vals, alpha=0.3, color='red',
                 label='Forbidden region')
ax1.set_xlabel('θ (polar angle on Bloch sphere)', fontsize=12)
ax1.set_ylabel('Uncertainty product', fontsize=12)
ax1.set_title('Spin-1/2 Uncertainty: σ(Sx)·σ(Sy) vs Bound', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Position-momentum uncertainty
ax2 = axes[1]

# Create states with different σ_x
sigma_x_range = np.linspace(0.2, 3, 100)
sigma_p_theory = hbar / (2 * sigma_x_range)  # Minimum uncertainty
sigma_p_gaussian = 1 / (2 * sigma_x_range)  # Actual Gaussian (with ℏ=1)

ax2.fill_between(sigma_x_range, 0, sigma_p_theory, alpha=0.3, color='red',
                 label='Forbidden: σxσp < ℏ/2')
ax2.plot(sigma_x_range, sigma_p_theory, 'r-', linewidth=2,
         label='Minimum: σxσp = ℏ/2')
ax2.scatter([1.0], [0.5], s=100, c='blue', zorder=5,
            label='Coherent state (σx=σp=1/√2)')
ax2.scatter([0.5], [1.0], s=100, c='green', marker='^', zorder=5,
            label='Squeezed in x')
ax2.scatter([2.0], [0.25], s=100, c='purple', marker='v', zorder=5,
            label='Squeezed in p')

ax2.set_xlabel('σx', fontsize=12)
ax2.set_ylabel('σp', fontsize=12)
ax2.set_title('Position-Momentum Uncertainty Region', fontsize=14)
ax2.set_xlim(0, 3.5)
ax2.set_ylim(0, 2)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_353_uncertainty_regions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_353_uncertainty_regions.png'")

# =============================================================================
# Part 6: Proof Verification
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Cauchy-Schwarz in the Proof")
print("=" * 70)

def verify_cs_inequality(A: np.ndarray, B: np.ndarray,
                         state: np.ndarray) -> Tuple[float, float]:
    """
    Verify Cauchy-Schwarz: |⟨f|g⟩|² ≤ ⟨f|f⟩⟨g|g⟩
    where |f⟩ = A'|ψ⟩, |g⟩ = B'|ψ⟩
    """
    exp_A = expectation_value(A, state)
    exp_B = expectation_value(B, state)

    A_prime = A - exp_A * np.eye(A.shape[0])
    B_prime = B - exp_B * np.eye(B.shape[0])

    f = A_prime @ state
    g = B_prime @ state

    lhs = np.abs(np.vdot(f, g))**2
    rhs = np.real(np.vdot(f, f) * np.vdot(g, g))

    return lhs, rhs

print("\nCauchy-Schwarz verification for spin states:")
print("State          |⟨f|g⟩|²      ⟨f|f⟩⟨g|g⟩    Satisfied?")
print("-" * 55)

test_states = [
    (up, "|↑⟩"),
    (plus_x, "|+x⟩"),
    (plus_y, "|+y⟩"),
    ((up + 1j*np.sqrt(3)*down)/2, "|custom⟩")
]

for state, name in test_states:
    lhs, rhs = verify_cs_inequality(Sx, Sy, state)
    satisfied = lhs <= rhs + 1e-10
    print(f"{name:12}  {lhs:.6f}     {rhs:.6f}      {satisfied}")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Robertson uncertainty | σₐσᵦ ≥ ½\|⟨[Â, B̂]⟩\| |
| Schrödinger uncertainty | σₐ²σᵦ² ≥ ¼\|⟨[Â, B̂]⟩\|² + ¼⟨{Â', B̂'}⟩² |
| Position-momentum | σₓσₚ ≥ ℏ/2 |
| Angular momentum | σ_{Lx}σ_{Ly} ≥ (ℏ/2)\|⟨L̂z⟩\| |
| Minimum uncertainty condition | Â'\|ψ⟩ = λB̂'\|ψ⟩ |
| Variance | σₐ² = ⟨Â²⟩ - ⟨Â⟩² |

### Main Takeaways

1. **The generalized uncertainty principle** applies to any pair of observables
2. **The bound is state-dependent** through ⟨[Â, B̂]⟩
3. **Proof uses Cauchy-Schwarz** inequality on carefully constructed states
4. **Equality requires** minimum uncertainty states (Gaussians for x-p)
5. **Uncertainty is intrinsic** to quantum states, not measurement limitations

---

## Daily Checklist

- [ ] Read Sakurai Chapter 1.4 (Uncertainty Relations)
- [ ] Read Shankar Chapter 9.2 (General Uncertainty Relations)
- [ ] Work through the Robertson proof step by step
- [ ] Verify the spin-1/2 uncertainty calculation
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Explain why the bound is state-dependent for angular momentum but not for x-p

---

## Preview: Day 354

Tomorrow we focus specifically on the **position-momentum uncertainty**: Δx·Δp ≥ ℏ/2. We'll explore Gaussian minimum uncertainty states, wave packet spreading, and the physical meaning of "complementary" descriptions. The position-momentum pair is the prototype for all conjugate variables.

---

*"The more precisely the position is determined, the less precisely the momentum is known in this instant, and vice versa."* — Werner Heisenberg (1927)

---

**Next:** [Day_354_Thursday.md](Day_354_Thursday.md) — Position-Momentum Uncertainty
