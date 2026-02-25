# Day 354: Position-Momentum Uncertainty — Δx·Δp ≥ ℏ/2

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Position-Momentum Conjugacy |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 354, you will be able to:

1. Derive Δx·Δp ≥ ℏ/2 from the generalized uncertainty principle
2. Identify and construct Gaussian minimum uncertainty wave packets
3. Analyze wave packet spreading and its connection to uncertainty
4. Calculate uncertainties for various quantum states
5. Explain the complementary nature of position and momentum
6. Connect the uncertainty relation to diffraction and interference

---

## Core Content

### 1. The Canonical Uncertainty Relation

From the generalized uncertainty principle σₐσᵦ ≥ ½|⟨[Â, B̂]⟩| and [x̂, p̂] = iℏ:

$$\boxed{\Delta x \cdot \Delta p \geq \frac{\hbar}{2}}$$

where we use Δx = σₓ and Δp = σₚ (standard deviations).

**Key features:**

1. **Universal:** Applies to any quantum state
2. **State-independent bound:** The right side is always ℏ/2
3. **Fundamental:** Cannot be circumvented by any measurement scheme
4. **Scale:** ℏ/2 ≈ 5.27 × 10⁻³⁵ J·s

---

### 2. Minimum Uncertainty States: Gaussians

States that saturate the inequality (equality holds) are **minimum uncertainty states**.

**Condition for equality:**

From the proof of the generalized uncertainty principle, equality requires:
$$(\hat{x} - \langle\hat{x}\rangle)|\psi\rangle = \gamma(\hat{p} - \langle\hat{p}\rangle)|\psi\rangle$$

with γ = iα (purely imaginary) and ⟨{Â', B̂'}⟩ = 0.

**Solution:** The unique normalizable solutions are Gaussian wave packets:

$$\boxed{\psi(x) = \left(\frac{1}{2\pi\sigma^2}\right)^{1/4} e^{-(x-x_0)^2/(4\sigma^2)} e^{ip_0 x/\hbar}}$$

with:
- x₀ = ⟨x̂⟩ (center position)
- p₀ = ⟨p̂⟩ (center momentum)
- σ = Δx (position uncertainty)
- Δp = ℏ/(2σ) (momentum uncertainty)

**Product:** Δx·Δp = σ·(ℏ/2σ) = ℏ/2 ✓

---

### 3. Position and Momentum Representations

**Position space:** The wave function ψ(x) = ⟨x|ψ⟩ gives:
$$|\psi(x)|^2 = \text{probability density for position}$$
$$\Delta x = \sqrt{\langle x^2\rangle - \langle x\rangle^2}$$

**Momentum space:** The wave function φ(p) = ⟨p|ψ⟩ is related by Fourier transform:
$$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty} \psi(x)e^{-ipx/\hbar}dx$$

$$|\phi(p)|^2 = \text{probability density for momentum}$$
$$\Delta p = \sqrt{\langle p^2\rangle - \langle p\rangle^2}$$

**The uncertainty principle as Fourier duality:**

The Fourier transform relates widths inversely:
$$\Delta x \cdot \Delta k \geq \frac{1}{2}$$

With p = ℏk:
$$\Delta x \cdot \frac{\Delta p}{\hbar} \geq \frac{1}{2} \quad \Rightarrow \quad \Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

---

### 4. Gaussian Wave Packet: Detailed Analysis

Consider the Gaussian centered at origin with zero momentum:

$$\psi(x) = \left(\frac{1}{2\pi\sigma^2}\right)^{1/4} e^{-x^2/(4\sigma^2)}$$

**Position statistics:**

$$\langle x\rangle = 0, \quad \langle x^2\rangle = \sigma^2$$
$$\Delta x = \sigma$$

**Momentum space wave function:**

$$\phi(p) = \left(\frac{2\sigma^2}{\pi\hbar^2}\right)^{1/4} e^{-\sigma^2 p^2/\hbar^2}$$

This is also Gaussian with width parameter ℏ/(2σ).

**Momentum statistics:**

$$\langle p\rangle = 0, \quad \langle p^2\rangle = \frac{\hbar^2}{4\sigma^2}$$
$$\Delta p = \frac{\hbar}{2\sigma}$$

**Uncertainty product:**

$$\boxed{\Delta x \cdot \Delta p = \sigma \cdot \frac{\hbar}{2\sigma} = \frac{\hbar}{2}}$$

The Gaussian achieves the minimum!

---

### 5. Wave Packet Spreading

A localized wave packet **spreads** over time due to the uncertainty principle.

**Physical picture:**

1. Initial Gaussian with width σ₀
2. Momentum uncertainty Δp = ℏ/(2σ₀)
3. Velocity uncertainty Δv = Δp/m = ℏ/(2mσ₀)
4. After time t, width increases by ~(Δv)t

**Exact formula for free particle:**

Starting from Gaussian with Δx(0) = σ₀:

$$\boxed{\Delta x(t) = \sigma_0\sqrt{1 + \left(\frac{\hbar t}{2m\sigma_0^2}\right)^2}}$$

**Spreading time scale:**

$$\tau = \frac{2m\sigma_0^2}{\hbar}$$

For t >> τ: Δx(t) ≈ (ℏt)/(2mσ₀) ∝ t

**Examples:**

| Object | σ₀ | m | τ |
|--------|-----|----|----|
| Electron | 1 nm | 9.1×10⁻³¹ kg | 10⁻¹⁴ s |
| Proton | 1 nm | 1.7×10⁻²⁷ kg | 2×10⁻¹¹ s |
| Atom | 1 μm | 10⁻²⁵ kg | 2×10⁻⁴ s |
| Dust | 1 μm | 10⁻¹² kg | 2×10¹⁰ s |

Macroscopic objects don't spread because τ >> age of universe!

---

### 6. The Uncertainty Principle and Diffraction

Single-slit diffraction demonstrates the uncertainty principle directly.

**Setup:**
- Particle (photon, electron) passes through slit of width Δx = a
- After the slit: position uncertainty ~a

**Consequence:**
- Momentum uncertainty: Δp ≥ ℏ/(2a)
- Transverse velocity uncertainty: Δv ≥ ℏ/(2ma)
- Angular spread: θ ~ Δv/v ~ ℏ/(2ma·v) ~ λ/(2a)

This matches the classical diffraction formula θ ~ λ/a!

**The uncertainty principle and wave diffraction are the same phenomenon.**

---

### 7. Other Conjugate Pairs

The position-momentum pair is the prototype for all conjugate variables:

| Conjugate pair | Commutator | Uncertainty relation |
|----------------|------------|---------------------|
| x, pₓ | [x̂, p̂ₓ] = iℏ | Δx·Δpₓ ≥ ℏ/2 |
| φ, Lz | [φ̂, L̂z] = iℏ | Δφ·ΔLz ≥ ℏ/2 |
| θ, pθ | [θ̂, p̂θ] = iℏ | Δθ·Δpθ ≥ ℏ/2 |
| N, φ | [N̂, φ̂] ≈ i | ΔN·Δφ ≥ 1/2 |

**Note:** The number-phase uncertainty for harmonic oscillators/photons is important for quantum optics.

---

## Physical Interpretation

### What Δx·Δp ≥ ℏ/2 Really Means

**What it says:**

1. A quantum particle cannot simultaneously have a definite position AND definite momentum
2. There is a trade-off: sharper position → broader momentum distribution
3. This is encoded in the wave function itself

**What it does NOT say:**

1. A single measurement cannot be precise
2. Measurement necessarily disturbs the system
3. We can only know probabilities (this is true, but not what the uncertainty principle says)

### The Wave-Particle Duality Connection

- **Particle picture:** Definite position
- **Wave picture:** Definite wavelength (momentum)

The uncertainty principle quantifies the incompatibility:

$$\Delta x \rightarrow 0 \Rightarrow \Delta p \rightarrow \infty \Rightarrow \text{no definite wavelength}$$
$$\Delta p \rightarrow 0 \Rightarrow \Delta x \rightarrow \infty \Rightarrow \text{delocalized}$$

A wave packet is a **compromise** between these extremes.

---

## Worked Examples

### Example 1: Ground State of Infinite Square Well

**Problem:** Calculate Δx and Δp for the ground state of an infinite square well (0 < x < L).

**Solution:**

The ground state wave function is:
$$\psi_1(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{\pi x}{L}\right)$$

**Position expectation values:**

$$\langle x\rangle = \frac{2}{L}\int_0^L x\sin^2\left(\frac{\pi x}{L}\right)dx = \frac{L}{2}$$

$$\langle x^2\rangle = \frac{2}{L}\int_0^L x^2\sin^2\left(\frac{\pi x}{L}\right)dx = \frac{L^2}{3} - \frac{L^2}{2\pi^2}$$

$$\Delta x = \sqrt{\langle x^2\rangle - \langle x\rangle^2} = L\sqrt{\frac{1}{12} - \frac{1}{2\pi^2}} \approx 0.181L$$

**Momentum expectation values:**

By symmetry: ⟨p⟩ = 0

$$\langle p^2\rangle = \frac{\pi^2\hbar^2}{L^2}$$ (from the eigenvalue equation)

$$\Delta p = \frac{\pi\hbar}{L}$$

**Uncertainty product:**

$$\Delta x \cdot \Delta p = 0.181L \cdot \frac{\pi\hbar}{L} = 0.568\hbar > \frac{\hbar}{2}$$

$$\boxed{\Delta x \cdot \Delta p = \frac{\pi\hbar}{L}\sqrt{\frac{L^2}{12} - \frac{L^2}{2\pi^2}} = \hbar\sqrt{\frac{\pi^2}{12} - \frac{1}{2}} \approx 0.568\hbar}$$

The ground state is NOT a minimum uncertainty state (only Gaussians are).

---

### Example 2: Coherent States

**Problem:** Show that coherent states of the harmonic oscillator are minimum uncertainty states.

**Solution:**

A coherent state |α⟩ is an eigenstate of the annihilation operator:
$$\hat{a}|α\rangle = α|α\rangle$$

where:
$$\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\hat{x} + i\frac{\hat{p}}{\sqrt{2m\omega\hbar}}$$

For coherent states:
$$\langle\hat{x}\rangle = \sqrt{\frac{2\hbar}{m\omega}}\text{Re}(α), \quad \langle\hat{p}\rangle = \sqrt{2m\omega\hbar}\text{Im}(α)$$

$$\langle\hat{x}^2\rangle - \langle\hat{x}\rangle^2 = \frac{\hbar}{2m\omega}$$
$$\langle\hat{p}^2\rangle - \langle\hat{p}\rangle^2 = \frac{m\omega\hbar}{2}$$

Therefore:
$$\Delta x = \sqrt{\frac{\hbar}{2m\omega}}, \quad \Delta p = \sqrt{\frac{m\omega\hbar}{2}}$$

$$\boxed{\Delta x \cdot \Delta p = \sqrt{\frac{\hbar}{2m\omega}} \cdot \sqrt{\frac{m\omega\hbar}{2}} = \frac{\hbar}{2}}$$

Coherent states are minimum uncertainty states! This is why they are "most classical."

---

### Example 3: Superposition State

**Problem:** A particle is in state ψ(x) = N[ψ₁(x) + ψ₂(x)] where ψₙ are harmonic oscillator eigenstates. Find Δx·Δp.

**Solution:**

Normalization: N = 1/√2

Position expectation values:
$$\langle x\rangle = \frac{1}{2}[\langle 1|x|1\rangle + \langle 2|x|2\rangle + 2\text{Re}\langle 1|x|2\rangle]$$

For harmonic oscillator:
$$\langle n|x|m\rangle = \sqrt{\frac{\hbar}{2m\omega}}\left[\sqrt{m}\delta_{n,m-1} + \sqrt{m+1}\delta_{n,m+1}\right]$$

So ⟨1|x̂|1⟩ = ⟨2|x̂|2⟩ = 0 and ⟨1|x̂|2⟩ = √(ℏ/mω)·√2/√2 = √(ℏ/mω)

$$\langle x\rangle = \sqrt{\frac{\hbar}{m\omega}}$$

Similarly:
$$\langle x^2\rangle = \frac{\hbar}{2m\omega}(2 \cdot 1 + 1 + 2 \cdot 2 + 1) + \frac{\hbar}{m\omega} = \frac{7\hbar}{2m\omega}$$

$$\Delta x^2 = \frac{7\hbar}{2m\omega} - \frac{\hbar}{m\omega} = \frac{5\hbar}{2m\omega}$$

For momentum (similar calculation):
$$\langle p\rangle = 0, \quad \Delta p^2 = \frac{5m\omega\hbar}{2}$$

$$\Delta x \cdot \Delta p = \sqrt{\frac{5\hbar}{2m\omega}} \cdot \sqrt{\frac{5m\omega\hbar}{2}} = \frac{5\hbar}{2}$$

$$\boxed{\Delta x \cdot \Delta p = \frac{5\hbar}{2} > \frac{\hbar}{2}}$$

Superpositions of energy eigenstates are NOT minimum uncertainty states.

---

## Practice Problems

### Level 1: Direct Application

1. **Gaussian width:** A Gaussian wave packet has Δx = 2 nm. Calculate Δp and the velocity uncertainty for an electron.

2. **Momentum space:** Given ψ(x) = Ne^{-x²/a²}, find φ(p) and verify Δx·Δp = ℏ/2.

3. **Harmonic oscillator:** For the first excited state of HO, calculate Δx and Δp.

### Level 2: Intermediate

4. **Wave packet spreading:** An electron is localized to σ₀ = 10 nm.
   (a) Calculate the spreading time τ.
   (b) What is Δx after t = 10⁻¹⁵ s?
   (c) When does Δx double?

5. **Non-Gaussian state:** For ψ(x) ∝ sech(x/a), calculate Δx, Δp, and the product.

6. **Squeezed states:** A squeezed vacuum has Δx = ℏ/(4mω) and Δp = mωℏ.
   (a) Verify Δx·Δp ≥ ℏ/2.
   (b) Compare to the coherent state uncertainties.
   (c) What physical advantage does squeezing provide?

### Level 3: Challenging

7. **Optimal localization:** You want to localize a particle to Δx = 1 fm. What minimum kinetic energy does this require? Compare to rest mass energy for an electron and proton.

8. **Atom confinement:** An atom in an optical trap has T = 1 μK.
   (a) Estimate the momentum spread from temperature.
   (b) What is the minimum position uncertainty?
   (c) Compare to the de Broglie wavelength.

9. **Entropic formulation:** Prove that for Gaussian states:
   $$H(X) + H(P) = 1 + \ln(\pi) + \ln(\Delta x) + \ln(\Delta p) \geq 1 + \ln(\pi\hbar)$$
   where H is the differential entropy.

---

## Computational Lab

### Objective
Visualize position-momentum uncertainty, Gaussian wave packets, and wave packet spreading.

```python
"""
Day 354 Computational Lab: Position-Momentum Uncertainty
Quantum Mechanics Core - Year 1, Week 51
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from typing import Tuple

# Physical constants (use natural units: hbar = m = 1)
hbar = 1.0
m = 1.0

# =============================================================================
# Part 1: Gaussian Wave Packet - Position and Momentum Space
# =============================================================================

print("=" * 70)
print("Part 1: Gaussian Wave Packet Analysis")
print("=" * 70)

def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float,
                        p0: float = 0) -> np.ndarray:
    """Gaussian wave packet in position space."""
    norm = (1 / (2 * np.pi * sigma**2))**0.25
    psi = norm * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * p0 * x / hbar)
    return psi

def fourier_transform(x: np.ndarray, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute momentum space wave function via FFT."""
    dx = x[1] - x[0]
    N = len(x)

    # Fourier transform
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2 * np.pi * hbar)

    # Momentum grid
    dk = 2 * np.pi / (N * dx)
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    p = hbar * k

    return p, psi_p

def calculate_uncertainty(x: np.ndarray, psi: np.ndarray) -> Tuple[float, float]:
    """Calculate position uncertainty."""
    dx = x[1] - x[0]
    prob = np.abs(psi)**2
    norm = np.sum(prob) * dx

    exp_x = np.sum(x * prob) * dx / norm
    exp_x2 = np.sum(x**2 * prob) * dx / norm

    return np.sqrt(exp_x2 - exp_x**2)

# Create position grid
N = 2048
L = 50.0
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

# Parameters for Gaussian
sigma = 2.0  # Position uncertainty
x0 = 0.0
p0 = 0.0

# Create wave packet
psi_x = gaussian_wavepacket(x, x0, sigma, p0)

# Transform to momentum space
p, psi_p = fourier_transform(x, psi_x)

# Calculate uncertainties
delta_x = calculate_uncertainty(x, psi_x)

# For momentum
dp = p[1] - p[0]
prob_p = np.abs(psi_p)**2
norm_p = np.sum(prob_p) * dp
exp_p = np.sum(p * prob_p) * dp / norm_p
exp_p2 = np.sum(p**2 * prob_p) * dp / norm_p
delta_p = np.sqrt(exp_p2 - exp_p**2)

print(f"\nGaussian wave packet with σ = {sigma}")
print(f"Calculated Δx = {delta_x:.4f} (expected: {sigma:.4f})")
print(f"Calculated Δp = {delta_p:.4f} (expected: {hbar/(2*sigma):.4f})")
print(f"Product Δx·Δp = {delta_x * delta_p:.4f}")
print(f"Lower bound ℏ/2 = {hbar/2:.4f}")
print(f"Ratio to bound: {delta_x * delta_p / (hbar/2):.4f}")

# =============================================================================
# Part 2: Visualization of Position and Momentum Distributions
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Position and Momentum Space Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Different sigmas
sigmas = [0.5, 1.0, 2.0, 4.0]
colors = ['blue', 'green', 'red', 'purple']

ax1 = axes[0, 0]
ax2 = axes[0, 1]

for sigma_val, color in zip(sigmas, colors):
    psi = gaussian_wavepacket(x, 0, sigma_val, 0)
    p_arr, psi_p = fourier_transform(x, psi)

    ax1.plot(x, np.abs(psi)**2, color=color, linewidth=2,
             label=f'Δx = {sigma_val}')
    ax2.plot(p_arr, np.abs(psi_p)**2, color=color, linewidth=2,
             label=f'Δp = {hbar/(2*sigma_val):.2f}')

ax1.set_xlim(-10, 10)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('|ψ(x)|²', fontsize=12)
ax1.set_title('Position Space Probability Density', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlim(-3, 3)
ax2.set_xlabel('p', fontsize=12)
ax2.set_ylabel('|φ(p)|²', fontsize=12)
ax2.set_title('Momentum Space Probability Density', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Lower panels: uncertainty product
ax3 = axes[1, 0]
sigma_range = np.linspace(0.5, 5, 50)
delta_x_arr = sigma_range
delta_p_arr = hbar / (2 * sigma_range)
product_arr = delta_x_arr * delta_p_arr

ax3.plot(sigma_range, delta_x_arr, 'b-', linewidth=2, label='Δx')
ax3.plot(sigma_range, delta_p_arr, 'r-', linewidth=2, label='Δp')
ax3.axhline(y=hbar/2, color='green', linestyle='--', label='ℏ/2')
ax3.set_xlabel('σ (Gaussian width parameter)', fontsize=12)
ax3.set_ylabel('Uncertainty', fontsize=12)
ax3.set_title('Trade-off: Δx and Δp vs Width Parameter', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.axhline(y=hbar/2, color='red', linestyle='--', linewidth=2, label='Bound = ℏ/2')
ax4.plot(sigma_range, product_arr, 'b-', linewidth=2, label='Δx·Δp (Gaussian)')
ax4.set_xlabel('σ (Gaussian width parameter)', fontsize=12)
ax4.set_ylabel('Δx · Δp', fontsize=12)
ax4.set_title('Uncertainty Product (Minimum for Gaussians)', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('day_354_gaussian_uncertainty.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_354_gaussian_uncertainty.png'")

# =============================================================================
# Part 3: Wave Packet Spreading
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Wave Packet Spreading in Time")
print("=" * 70)

def spreading_width(t: float, sigma0: float, m: float = 1.0,
                    hbar: float = 1.0) -> float:
    """Calculate the width of a spreading Gaussian wave packet."""
    return sigma0 * np.sqrt(1 + (hbar * t / (2 * m * sigma0**2))**2)

sigma0 = 1.0
spreading_time = 2 * m * sigma0**2 / hbar
print(f"\nInitial width: σ₀ = {sigma0}")
print(f"Spreading time: τ = 2mσ₀²/ℏ = {spreading_time:.2f}")

times = np.linspace(0, 5 * spreading_time, 100)
widths = [spreading_width(t, sigma0) for t in times]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Width vs time
ax1 = axes[0]
ax1.plot(times, widths, 'b-', linewidth=2)
ax1.axvline(x=spreading_time, color='red', linestyle='--',
            label=f'τ = {spreading_time:.2f}')
ax1.axhline(y=sigma0, color='green', linestyle=':', label=f'σ₀ = {sigma0}')
ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel('Width Δx(t)', fontsize=12)
ax1.set_title('Wave Packet Spreading', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Snapshots of wave packet
ax2 = axes[1]
time_snapshots = [0, spreading_time/2, spreading_time, 2*spreading_time]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(time_snapshots)))

for t, color in zip(time_snapshots, colors):
    sigma_t = spreading_width(t, sigma0)
    psi_t = gaussian_wavepacket(x, 0, sigma_t, 0)
    ax2.plot(x, np.abs(psi_t)**2, color=color, linewidth=2,
             label=f't = {t:.1f}, Δx = {sigma_t:.2f}')

ax2.set_xlim(-10, 10)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('|ψ(x,t)|²', fontsize=12)
ax2.set_title('Wave Packet at Different Times', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_354_spreading.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_354_spreading.png'")

# =============================================================================
# Part 4: Non-Gaussian States - Comparison
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Comparison with Non-Gaussian States")
print("=" * 70)

def box_wavefunction(x: np.ndarray, L: float) -> np.ndarray:
    """Normalized box (rectangular) wave function."""
    psi = np.where(np.abs(x) < L/2, 1/np.sqrt(L), 0.0)
    return psi.astype(complex)

def sech_wavefunction(x: np.ndarray, a: float) -> np.ndarray:
    """Normalized sech wave function."""
    psi = np.sqrt(1/(2*a)) / np.cosh(x/a)
    return psi.astype(complex)

# Compare different wave functions
wavefunctions = [
    ('Gaussian', lambda x: gaussian_wavepacket(x, 0, 2, 0)),
    ('Box', lambda x: box_wavefunction(x, 4)),
    ('Sech', lambda x: sech_wavefunction(x, 1.5))
]

print("\nUncertainty products for different wave functions:")
print("-" * 50)
print(f"{'Wave function':<15} {'Δx':<10} {'Δp':<10} {'Δx·Δp':<10} {'Ratio':<10}")
print("-" * 50)

for name, psi_func in wavefunctions:
    psi = psi_func(x)
    p_arr, psi_p = fourier_transform(x, psi)

    # Position uncertainty
    prob_x = np.abs(psi)**2
    norm_x = np.sum(prob_x) * dx
    exp_x = np.sum(x * prob_x) * dx / norm_x
    exp_x2 = np.sum(x**2 * prob_x) * dx / norm_x
    delta_x = np.sqrt(np.abs(exp_x2 - exp_x**2))

    # Momentum uncertainty
    dp = p_arr[1] - p_arr[0]
    prob_p = np.abs(psi_p)**2
    norm_p = np.sum(prob_p) * dp
    exp_p = np.sum(p_arr * prob_p) * dp / norm_p
    exp_p2 = np.sum(p_arr**2 * prob_p) * dp / norm_p
    delta_p = np.sqrt(np.abs(exp_p2 - exp_p**2))

    product = delta_x * delta_p
    ratio = product / (hbar/2)

    print(f"{name:<15} {delta_x:<10.4f} {delta_p:<10.4f} {product:<10.4f} {ratio:<10.4f}")

print("-" * 50)
print(f"Minimum (ℏ/2) = {hbar/2:.4f}")

# =============================================================================
# Part 5: Harmonic Oscillator States
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Harmonic Oscillator States")
print("=" * 70)

omega = 1.0

def hermite_coeff(n: int, x: np.ndarray) -> np.ndarray:
    """Hermite polynomial H_n(x)."""
    h = hermite(n)
    return h(x)

def ho_eigenstate(x: np.ndarray, n: int, m: float = 1.0,
                  omega: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """Harmonic oscillator eigenstate."""
    xi = np.sqrt(m * omega / hbar) * x
    norm = (m * omega / (np.pi * hbar))**0.25 / np.sqrt(2**n * np.math.factorial(n))
    psi = norm * hermite_coeff(n, xi) * np.exp(-xi**2 / 2)
    return psi.astype(complex)

print("\nHarmonic oscillator eigenstate uncertainties:")
print("-" * 50)
print(f"{'n':<5} {'Δx':<12} {'Δp':<12} {'Δx·Δp':<12} {'Theory (n+1/2)ℏ':<15}")
print("-" * 50)

for n in range(5):
    psi = ho_eigenstate(x, n)
    p_arr, psi_p = fourier_transform(x, psi)

    # Position uncertainty
    prob_x = np.abs(psi)**2
    norm_x = np.sum(prob_x) * dx
    exp_x2 = np.sum(x**2 * prob_x) * dx / norm_x
    delta_x = np.sqrt(exp_x2)  # <x> = 0 by symmetry

    # Momentum uncertainty
    dp = p_arr[1] - p_arr[0]
    prob_p = np.abs(psi_p)**2
    norm_p = np.sum(prob_p) * dp
    exp_p2 = np.sum(p_arr**2 * prob_p) * dp / norm_p
    delta_p = np.sqrt(exp_p2)  # <p> = 0 by symmetry

    product = delta_x * delta_p
    theory = (n + 0.5) * hbar

    print(f"{n:<5} {delta_x:<12.4f} {delta_p:<12.4f} {product:<12.4f} {theory:<15.4f}")

# =============================================================================
# Part 6: Visualization Summary
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Summary Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Different wave functions
ax1 = axes[0, 0]
for name, psi_func in wavefunctions:
    psi = psi_func(x)
    ax1.plot(x, np.abs(psi)**2, linewidth=2, label=name)
ax1.set_xlim(-8, 8)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('|ψ(x)|²', fontsize=12)
ax1.set_title('Different Wave Functions', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: HO eigenstates
ax2 = axes[0, 1]
for n in range(4):
    psi = ho_eigenstate(x, n)
    ax2.plot(x, np.abs(psi)**2, linewidth=2, label=f'n = {n}')
ax2.set_xlim(-5, 5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('|ψ(x)|²', fontsize=12)
ax2.set_title('Harmonic Oscillator Eigenstates', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Uncertainty trade-off
ax3 = axes[1, 0]
n_states = 20
delta_x_vals = np.linspace(0.3, 3, n_states)
delta_p_vals = hbar / (2 * delta_x_vals)  # Minimum uncertainty

ax3.scatter(delta_x_vals, delta_p_vals, c=delta_x_vals, cmap='viridis',
            s=100, label='Minimum uncertainty states')
ax3.fill_between(delta_x_vals, 0, delta_p_vals, alpha=0.3, color='red',
                 label='Forbidden region')
ax3.set_xlabel('Δx', fontsize=12)
ax3.set_ylabel('Δp', fontsize=12)
ax3.set_title('Uncertainty Trade-off Curve', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Spreading comparison
ax4 = axes[1, 1]
sigma_initials = [0.5, 1.0, 2.0]
times_plot = np.linspace(0, 10, 100)

for s0 in sigma_initials:
    widths = [spreading_width(t, s0) for t in times_plot]
    ax4.plot(times_plot, widths, linewidth=2, label=f'σ₀ = {s0}')

ax4.set_xlabel('Time t', fontsize=12)
ax4.set_ylabel('Δx(t)', fontsize=12)
ax4.set_title('Wave Packet Spreading for Different Initial Widths', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_354_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_354_summary.png'")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Position-momentum uncertainty | Δx·Δp ≥ ℏ/2 |
| Gaussian minimum uncertainty | Δx·Δp = ℏ/2 |
| Gaussian wave packet | ψ(x) = (2πσ²)^(-1/4) exp(-(x-x₀)²/4σ²) exp(ip₀x/ℏ) |
| Position uncertainty | Δx = σ |
| Momentum uncertainty | Δp = ℏ/(2σ) |
| Fourier relation | φ(p) = (2π)^(-1/2) ∫ψ(x)e^(-ipx/ℏ)dx |
| Wave packet spreading | Δx(t) = σ₀√(1 + (ℏt/2mσ₀²)²) |
| Spreading time | τ = 2mσ₀²/ℏ |

### Main Takeaways

1. **Δx·Δp ≥ ℏ/2 is universal** — Applies to all quantum states
2. **Only Gaussians saturate the bound** — Coherent states are "most classical"
3. **Fourier duality** — Position and momentum are Fourier conjugates
4. **Wave packets spread** — Uncertainty in momentum causes spatial spreading
5. **Macroscopic objects don't spread** — τ >> age of universe for everyday objects

---

## Daily Checklist

- [ ] Read Griffiths Section 3.5.2 (Position-Momentum Uncertainty)
- [ ] Read Shankar Section 9.4 (The Minimum Uncertainty Wave Packet)
- [ ] Derive the spreading formula Δx(t)
- [ ] Calculate Δx·Δp for the infinite square well ground state
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Explain why coherent states are called "most classical"

---

## Preview: Day 355

Tomorrow we explore the **energy-time uncertainty relation**: ΔE·Δt ≥ ℏ/2. Unlike position-momentum, time is NOT an operator in quantum mechanics, giving this relation a fundamentally different character. We'll see how it relates to particle lifetimes, spectral line widths, and quantum tunneling.

---

*"The uncertainty principle 'protects' quantum mechanics. Heisenberg recognized that if it were possible to measure the momentum and position simultaneously with greater accuracy, quantum mechanics would collapse."* — Richard Feynman

---

**Next:** [Day_355_Friday.md](Day_355_Friday.md) — Energy-Time Uncertainty
