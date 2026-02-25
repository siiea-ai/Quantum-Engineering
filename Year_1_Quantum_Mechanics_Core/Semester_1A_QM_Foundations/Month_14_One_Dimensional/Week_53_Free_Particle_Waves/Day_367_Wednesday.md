# Day 367: Wave Packets I - Superposition and Localization

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Superposition principle, wave packet construction |
| Afternoon | 2.5 hours | Problem solving: Fourier transforms, localization |
| Evening | 2 hours | Computational lab: Building wave packets |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Apply the superposition principle** to construct localized wave packets
2. **Interpret wave packets** as physically realizable quantum states
3. **Use Fourier analysis** to relate position and momentum distributions
4. **Quantify the trade-off** between position and momentum localization
5. **Construct wave packets** from momentum distributions φ(k)
6. **Visualize wave packet structure** computationally

---

## Core Content

### 1. The Need for Wave Packets

Plane waves ψ_k(x) = Ae^{ikx} have definite momentum but are:
- Not normalizable (extend to infinity)
- Completely delocalized in position
- Unphysical as representations of real particles

**Wave packets** solve these problems by superposing plane waves:

$$\boxed{\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\phi(k)e^{ikx}dk}$$

This is the **inverse Fourier transform** of the momentum distribution φ(k).

### 2. The Superposition Principle

The Schrödinger equation is **linear**: if ψ₁ and ψ₂ are solutions, so is any linear combination:

$$\psi = c_1\psi_1 + c_2\psi_2$$

For the free particle, each plane wave e^{ikx} is a solution. Therefore, any superposition:

$$\psi(x) = \int_{-\infty}^{\infty}\phi(k)e^{ikx}dk$$

is also a solution, where φ(k) is an arbitrary complex-valued function giving the "weight" of each k-component.

### 3. Fourier Transform Pairs

The position and momentum representations are related by Fourier transforms:

**Position → Momentum (Fourier Transform):**

$$\boxed{\phi(k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\psi(x)e^{-ikx}dx}$$

**Momentum → Position (Inverse Fourier Transform):**

$$\boxed{\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\phi(k)e^{ikx}dk}$$

These form a **Fourier transform pair** with the symmetric normalization convention.

### 4. Physical Interpretation of φ(k)

The function φ(k) is the **momentum-space wave function** (in units where ℏ = 1):

- |φ(k)|² dk = probability of finding momentum in [ℏk, ℏ(k + dk)]
- The wave packet contains a **range** of momenta, not just one
- Narrow φ(k) → well-defined momentum → spread out in position
- Wide φ(k) → uncertain momentum → localized in position

### 5. Localization vs. Momentum Spread

**Key insight:** You cannot have both sharp position AND sharp momentum.

Consider a wave packet centered at k₀ with width Δk:

$$\phi(k) \approx 0 \quad \text{for} \quad |k - k_0| > \Delta k$$

The resulting position-space wave packet has width Δx where:

$$\boxed{\Delta x \cdot \Delta k \gtrsim 1}$$

In terms of momentum (p = ℏk):

$$\boxed{\Delta x \cdot \Delta p \gtrsim \hbar}$$

This is the **uncertainty principle** emerging from Fourier analysis!

### 6. Types of Wave Packets

#### Rectangular (Top-Hat) Distribution

$$\phi(k) = \begin{cases} 1/\sqrt{2k_0} & |k - k_c| < k_0 \\ 0 & \text{otherwise} \end{cases}$$

Position space:

$$\psi(x) \propto e^{ik_c x}\frac{\sin(k_0 x)}{x} = e^{ik_c x} \cdot k_0 \text{sinc}(k_0 x)$$

Features:
- Oscillating tails (sinc function)
- First zeros at x = ±π/k₀
- Width Δx ~ π/k₀, so Δx·Δk ~ π

#### Gaussian Distribution

$$\phi(k) = \left(\frac{2\sigma_k^2}{\pi}\right)^{1/4}e^{-(k-k_0)^2\sigma_k^2}$$

Position space:

$$\psi(x) \propto e^{ik_0 x}e^{-x^2/4\sigma_x^2}$$

where σ_x = 1/(2σ_k).

Features:
- No oscillating tails
- Minimum uncertainty: Δx·Δk = 1/2
- Gaussian in both spaces

#### Lorentzian Distribution

$$\phi(k) = \frac{A}{\sqrt{\pi\gamma}}\frac{1}{1 + (k-k_0)^2/\gamma^2}$$

Position space:

$$\psi(x) \propto e^{ik_0 x}e^{-\gamma|x|}$$

Features:
- Exponential decay in position
- Heavy tails in momentum (slow decay)
- Δx·Δk larger than Gaussian

### 7. Constructing Wave Packets

**Recipe for building a wave packet:**

1. Choose a center wave number k₀ (determines average momentum)
2. Choose a distribution shape φ(k - k₀)
3. Choose distribution width Δk (determines position spread)
4. Compute inverse Fourier transform to get ψ(x)

**Key relationships:**

| Momentum space | Position space |
|---------------|----------------|
| Center k₀ | Oscillation frequency e^{ik₀x} |
| Width Δk | Envelope width Δx ~ 1/Δk |
| Shape of φ(k) | Shape of envelope |

### 8. Normalization of Wave Packets

If φ(k) is normalized:

$$\int_{-\infty}^{\infty}|\phi(k)|^2 dk = 1$$

Then by **Parseval's theorem**, ψ(x) is automatically normalized:

$$\int_{-\infty}^{\infty}|\psi(x)|^2 dx = \int_{-\infty}^{\infty}|\phi(k)|^2 dk = 1$$

This is a fundamental property of Fourier transforms.

---

## Quantum Computing Connection

### Wave Packet Control in Quantum Systems

**Ultrafast laser pulses** are essentially wave packets of photons used to:
- Control molecular dynamics (femtochemistry)
- Manipulate qubit states in trapped ions
- Generate entanglement in photonic systems

**Pulse shaping** technology precisely controls φ(k) to engineer desired wave packet profiles.

### Quantum Communication

In quantum key distribution (QKD):
- Single photons are sent as wave packets
- Narrow bandwidth (small Δk) improves signal
- But requires long temporal duration (large Δt)
- Trade-off determines channel capacity

---

## Worked Examples

### Example 1: Sinc Wave Packet

**Problem:** Find the position-space wave packet for a rectangular momentum distribution:
$$\phi(k) = \begin{cases} 1/\sqrt{2a} & |k| < a \\ 0 & \text{otherwise} \end{cases}$$

**Solution:**

Apply the inverse Fourier transform:

$$\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-a}^{a}\frac{1}{\sqrt{2a}}e^{ikx}dk$$

$$= \frac{1}{\sqrt{4\pi a}}\left[\frac{e^{ikx}}{ix}\right]_{-a}^{a}$$

$$= \frac{1}{\sqrt{4\pi a}}\cdot\frac{e^{iax} - e^{-iax}}{ix}$$

$$= \frac{1}{\sqrt{4\pi a}}\cdot\frac{2i\sin(ax)}{ix}$$

$$= \frac{1}{\sqrt{4\pi a}}\cdot\frac{2\sin(ax)}{x}$$

Using sinc(u) = sin(u)/u:

$$\boxed{\psi(x) = \sqrt{\frac{a}{\pi}}\text{sinc}(ax) = \sqrt{\frac{a}{\pi}}\frac{\sin(ax)}{ax}}$$

**Position spread:** First zero at x = π/a, so Δx ~ π/a.
**Momentum spread:** Δk = 2a.
**Product:** Δx·Δk ~ 2π > 1/2 (not minimum uncertainty).

### Example 2: Shifted Wave Packet

**Problem:** If φ(k) is centered at k₀, show that ψ(x) = e^{ik₀x}ψ₀(x) where ψ₀ is the wave packet for φ centered at k = 0.

**Solution:**

Write φ(k) = g(k - k₀) where g is centered at zero.

$$\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}g(k - k_0)e^{ikx}dk$$

Substitute u = k - k₀, so k = u + k₀, dk = du:

$$\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}g(u)e^{i(u + k_0)x}du$$

$$= e^{ik_0 x}\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}g(u)e^{iux}du$$

$$\boxed{\psi(x) = e^{ik_0 x}\psi_0(x)}$$

The carrier wave e^{ik₀x} oscillates at the central frequency, modulated by the envelope ψ₀(x).

### Example 3: Sum of Two Wave Numbers

**Problem:** A wave packet contains only two plane waves with equal amplitude:
$$\phi(k) = \frac{1}{\sqrt{2}}[\delta(k - k_1) + \delta(k - k_2)]$$
Find ψ(x) and |ψ(x)|².

**Solution:**

$$\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\phi(k)e^{ikx}dk$$

$$= \frac{1}{\sqrt{4\pi}}[e^{ik_1 x} + e^{ik_2 x}]$$

For the probability density:

$$|\psi(x)|^2 = \frac{1}{4\pi}|e^{ik_1 x} + e^{ik_2 x}|^2$$

$$= \frac{1}{4\pi}(e^{-ik_1 x} + e^{-ik_2 x})(e^{ik_1 x} + e^{ik_2 x})$$

$$= \frac{1}{4\pi}[2 + e^{i(k_2-k_1)x} + e^{-i(k_2-k_1)x}]$$

$$\boxed{|\psi(x)|^2 = \frac{1}{2\pi}[1 + \cos((k_2 - k_1)x)]}$$

This is a **beat pattern** with spatial period 2π/(k₂ - k₁).

---

## Practice Problems

### Level 1: Direct Application

1. **Simple superposition:** If ψ₁(x) = e^{ikx} and ψ₂(x) = e^{-ikx}, find |ψ₁ + ψ₂|² and sketch it.

2. **Normalization check:** Verify that the sinc wave packet ψ(x) = √(a/π) sinc(ax) is normalized.

3. **Central momentum:** For a wave packet ψ(x) = Ae^{ik₀x}e^{-x²/4σ²}, what is the expectation value ⟨p⟩?

### Level 2: Intermediate

4. **Triangular distribution:** Find the position-space wave packet for:
   $$\phi(k) = \begin{cases} A(1 - |k|/a) & |k| < a \\ 0 & \text{otherwise} \end{cases}$$
   Normalize φ(k) first.

5. **Uncertainty product:** For the rectangular φ(k) in Example 1, compute ⟨x²⟩ and verify the uncertainty relation.

6. **Complex amplitude:** What happens to ψ(x) if φ(k) → φ(k)e^{iδ} (constant phase shift)?

### Level 3: Challenging

7. **Asymmetric distribution:** For φ(k) = Ae^{-(k-k₀)²/σ²}e^{ik₀a} (Gaussian with linear phase), show that the wave packet is shifted to x = a.

8. **Moments and derivatives:** Prove that:
   $$\langle x \rangle = i\left[\frac{d}{dk}\phi(k)\right]_{k=0} / \phi(0)$$
   if φ(k) is real and symmetric.

9. **Energy spread:** For a wave packet φ(k), express ⟨H⟩ and ⟨H²⟩ in terms of moments of |φ(k)|² and find ΔE.

---

## Computational Lab: Building Wave Packets

```python
"""
Day 367 Computational Lab: Wave Packet Construction
====================================================
Building and analyzing wave packets from superpositions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.integrate import simps

# =============================================================================
# Part 1: Wave Packet from Superposition
# =============================================================================

def plane_wave(x, k):
    """Single plane wave e^{ikx}"""
    return np.exp(1j * k * x)

# Create position grid
x = np.linspace(-20, 20, 2000)
dx = x[1] - x[0]

# Superposition of discrete plane waves
k_center = 5.0
k_spread = 1.0
n_waves = 21
k_values = np.linspace(k_center - k_spread, k_center + k_spread, n_waves)

# Gaussian weights for each k
weights = np.exp(-(k_values - k_center)**2 / (2 * (k_spread/3)**2))
weights /= np.sqrt(np.sum(weights**2))  # Normalize

# Build wave packet
psi = np.zeros_like(x, dtype=complex)
for k, w in zip(k_values, weights):
    psi += w * plane_wave(x, k)
psi /= np.sqrt(n_waves)  # Approximate normalization

# Visualize construction
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Individual plane waves
ax = axes[0, 0]
for i, (k, w) in enumerate(zip(k_values[::5], weights[::5])):  # Every 5th
    wave = w * plane_wave(x, k)
    ax.plot(x, np.real(wave), alpha=0.5, label=f'k={k:.1f}')
ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('Re(ψ_k)')
ax.set_title(f'Individual Plane Waves (showing {len(k_values[::5])} of {n_waves})')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Superposition result
ax = axes[0, 1]
ax.plot(x, np.real(psi), 'b-', linewidth=1.5, label='Re(ψ)')
ax.plot(x, np.imag(psi), 'r-', linewidth=1.5, label='Im(ψ)')
ax.plot(x, np.abs(psi), 'k--', linewidth=2, label='|ψ|')
ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('ψ(x)')
ax.set_title('Superposition: Wave Packet')
ax.legend()
ax.grid(True, alpha=0.3)

# Momentum distribution
ax = axes[1, 0]
ax.bar(k_values, weights**2, width=0.08, alpha=0.7)
ax.set_xlabel('k')
ax.set_ylabel('|φ(k)|²')
ax.set_title('Momentum Distribution (Discrete Weights)')
ax.grid(True, alpha=0.3)

# Probability density
ax = axes[1, 1]
ax.fill_between(x, 0, np.abs(psi)**2, alpha=0.5)
ax.plot(x, np.abs(psi)**2, 'b-', linewidth=2)
ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x)|²')
ax.set_title('Position Probability Density')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wave_packet_construction.png', dpi=150)
plt.show()

# =============================================================================
# Part 2: Different Momentum Distributions
# =============================================================================

def build_wave_packet(x, phi_func, k_range=(-10, 10), n_k=500):
    """
    Build wave packet from continuous momentum distribution.
    Uses numerical integration (midpoint rule).
    """
    k = np.linspace(k_range[0], k_range[1], n_k)
    dk = k[1] - k[0]

    phi = phi_func(k)
    psi = np.zeros_like(x, dtype=complex)

    for ki, phi_i in zip(k, phi):
        psi += phi_i * np.exp(1j * ki * x) * dk

    psi /= np.sqrt(2 * np.pi)
    return psi, k, phi

# Define different momentum distributions
def rectangular(k, k_center=5, width=2):
    phi = np.where(np.abs(k - k_center) < width/2, 1.0, 0.0)
    return phi / np.sqrt(width)  # Normalized

def gaussian(k, k_center=5, sigma=0.5):
    phi = np.exp(-(k - k_center)**2 / (2 * sigma**2))
    return phi / (sigma * np.sqrt(2 * np.pi))**0.5

def lorentzian(k, k_center=5, gamma=0.5):
    phi = 1 / (1 + ((k - k_center) / gamma)**2)
    norm = np.sqrt(np.pi * gamma / 2)
    return phi / norm

distributions = [
    ('Rectangular', rectangular),
    ('Gaussian', gaussian),
    ('Lorentzian', lorentzian)
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for i, (name, phi_func) in enumerate(distributions):
    psi, k, phi = build_wave_packet(x, phi_func)

    # Momentum space
    axes[i, 0].plot(k, np.abs(phi)**2, 'b-', linewidth=2)
    axes[i, 0].fill_between(k, 0, np.abs(phi)**2, alpha=0.3)
    axes[i, 0].set_xlim(0, 10)
    axes[i, 0].set_xlabel('k')
    axes[i, 0].set_ylabel('|φ(k)|²')
    axes[i, 0].set_title(f'{name}: Momentum Distribution')
    axes[i, 0].grid(True, alpha=0.3)

    # Position space
    axes[i, 1].plot(x, np.real(psi), 'b-', linewidth=1, label='Re(ψ)', alpha=0.7)
    axes[i, 1].plot(x, np.abs(psi), 'k-', linewidth=2, label='|ψ|')
    axes[i, 1].set_xlim(-10, 10)
    axes[i, 1].set_xlabel('x')
    axes[i, 1].set_ylabel('ψ(x)')
    axes[i, 1].set_title(f'{name}: Position Space Wave Packet')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('different_distributions.png', dpi=150)
plt.show()

# =============================================================================
# Part 3: Width Relationship (Uncertainty Principle)
# =============================================================================

sigma_k_values = [0.25, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(2, len(sigma_k_values), figsize=(16, 8))

delta_x_list = []
delta_k_list = []

for i, sigma_k in enumerate(sigma_k_values):
    phi_func = lambda k, sk=sigma_k: gaussian(k, k_center=5, sigma=sk)
    psi, k, phi = build_wave_packet(x, phi_func, k_range=(0, 10))

    # Normalize properly
    norm = np.sqrt(simps(np.abs(psi)**2, x))
    psi = psi / norm

    # Compute uncertainties
    prob_k = np.abs(phi)**2
    prob_k /= simps(prob_k, k)

    k_mean = simps(k * prob_k, k)
    k2_mean = simps(k**2 * prob_k, k)
    delta_k = np.sqrt(k2_mean - k_mean**2)

    prob_x = np.abs(psi)**2
    x_mean = simps(x * prob_x, x)
    x2_mean = simps(x**2 * prob_x, x)
    delta_x = np.sqrt(x2_mean - x_mean**2)

    delta_x_list.append(delta_x)
    delta_k_list.append(delta_k)

    # Momentum space
    axes[0, i].plot(k, np.abs(phi)**2 / np.max(np.abs(phi)**2), 'b-', linewidth=2)
    axes[0, i].fill_between(k, 0, np.abs(phi)**2 / np.max(np.abs(phi)**2), alpha=0.3)
    axes[0, i].axvline(k_mean, color='r', linestyle='--')
    axes[0, i].set_xlim(2, 8)
    axes[0, i].set_xlabel('k')
    axes[0, i].set_title(f'σ_k = {sigma_k}, Δk = {delta_k:.2f}')
    if i == 0:
        axes[0, i].set_ylabel('|φ(k)|² (normalized)')

    # Position space
    axes[1, i].plot(x, np.abs(psi)**2 / np.max(np.abs(psi)**2), 'r-', linewidth=2)
    axes[1, i].fill_between(x, 0, np.abs(psi)**2 / np.max(np.abs(psi)**2),
                            alpha=0.3, color='red')
    axes[1, i].axvline(x_mean, color='b', linestyle='--')
    axes[1, i].set_xlim(-15, 15)
    axes[1, i].set_xlabel('x')
    axes[1, i].set_title(f'Δx = {delta_x:.2f}')
    if i == 0:
        axes[1, i].set_ylabel('|ψ(x)|² (normalized)')

plt.suptitle('Uncertainty Principle: Narrow Δk ↔ Wide Δx', fontsize=14)
plt.tight_layout()
plt.savefig('uncertainty_widths.png', dpi=150)
plt.show()

# Print uncertainty products
print("\n" + "="*60)
print("UNCERTAINTY PRODUCTS")
print("="*60)
print(f"{'σ_k':>8} {'Δk':>10} {'Δx':>10} {'Δx·Δk':>12} {'≥ 0.5?':>10}")
print("-"*52)
for sigma_k, dk, dx in zip(sigma_k_values, delta_k_list, delta_x_list):
    product = dx * dk
    check = "✓" if product >= 0.5 else "✗"
    print(f"{sigma_k:>8.2f} {dk:>10.3f} {dx:>10.3f} {product:>12.4f} {check:>10}")

# =============================================================================
# Part 4: Beat Patterns from Two Frequencies
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Two waves with different k
k1, k2 = 4.0, 5.0
psi1 = plane_wave(x, k1)
psi2 = plane_wave(x, k2)
psi_sum = (psi1 + psi2) / np.sqrt(2)

# Individual waves
ax = axes[0, 0]
ax.plot(x, np.real(psi1), 'b-', alpha=0.7, label=f'k₁ = {k1}')
ax.plot(x, np.real(psi2), 'r-', alpha=0.7, label=f'k₂ = {k2}')
ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('Re(ψ)')
ax.set_title('Two Individual Plane Waves')
ax.legend()
ax.grid(True, alpha=0.3)

# Superposition
ax = axes[0, 1]
ax.plot(x, np.real(psi_sum), 'g-', linewidth=1.5)
ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('Re(ψ₁ + ψ₂)')
ax.set_title('Superposition: Beat Pattern')
ax.grid(True, alpha=0.3)

# Probability density with beat pattern
ax = axes[1, 0]
prob = np.abs(psi_sum)**2
ax.fill_between(x, 0, prob, alpha=0.5, color='purple')
ax.plot(x, prob, 'purple', linewidth=2)
beat_wavelength = 2 * np.pi / np.abs(k2 - k1)
ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('|ψ|²')
ax.set_title(f'Probability Density (Beat wavelength = {beat_wavelength:.2f})')
ax.grid(True, alpha=0.3)

# Three waves
k_vals = [4.0, 5.0, 6.0]
psi_3 = sum(plane_wave(x, k) for k in k_vals) / np.sqrt(3)

ax = axes[1, 1]
ax.fill_between(x, 0, np.abs(psi_3)**2, alpha=0.5, color='orange')
ax.plot(x, np.abs(psi_3)**2, 'orange', linewidth=2)
ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('|ψ|²')
ax.set_title('Three Waves: More Localization')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('beat_patterns.png', dpi=150)
plt.show()

# =============================================================================
# Part 5: FFT-Based Wave Packet Analysis
# =============================================================================

# Create a Gaussian wave packet directly
sigma_x = 2.0
k0 = 3.0
psi_gaussian = np.exp(1j * k0 * x) * np.exp(-x**2 / (4 * sigma_x**2))
psi_gaussian /= (2 * np.pi * sigma_x**2)**0.25  # Normalize

# Use FFT to get momentum space
N = len(x)
k_fft = fftfreq(N, dx) * 2 * np.pi  # Angular frequency
phi_fft = fftshift(fft(ifftshift(psi_gaussian))) * dx / np.sqrt(2 * np.pi)
k_fft = fftshift(k_fft)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Position space
ax = axes[0]
ax.plot(x, np.abs(psi_gaussian)**2, 'b-', linewidth=2)
ax.fill_between(x, 0, np.abs(psi_gaussian)**2, alpha=0.3)
ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x)|²')
ax.set_title(f'Position Space (σ_x = {sigma_x})')
ax.grid(True, alpha=0.3)

# Momentum space (from FFT)
ax = axes[1]
ax.plot(k_fft, np.abs(phi_fft)**2, 'r-', linewidth=2)
ax.fill_between(k_fft, 0, np.abs(phi_fft)**2, alpha=0.3, color='red')
ax.axvline(k0, color='k', linestyle='--', label=f'k₀ = {k0}')
ax.set_xlim(-2, 8)
ax.set_xlabel('k')
ax.set_ylabel('|φ(k)|²')
sigma_k_expected = 1 / (2 * sigma_x)
ax.set_title(f'Momentum Space via FFT (expected σ_k = {sigma_k_expected:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fft_analysis.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS FROM THIS LAB")
print("="*60)
print("""
1. Wave packets are superpositions of plane waves with different k.

2. The momentum distribution φ(k) determines the position envelope.

3. Narrow φ(k) → wide ψ(x) and vice versa (uncertainty principle).

4. Gaussian distributions minimize uncertainty (Δx·Δk = 1/2).

5. Beat patterns emerge from discrete superpositions.

6. FFT provides efficient numerical Fourier transforms.
""")
```

---

## Summary

### Key Formulas Table

| Quantity | Formula | Notes |
|----------|---------|-------|
| Wave packet | $$\psi(x) = \frac{1}{\sqrt{2\pi}}\int\phi(k)e^{ikx}dk$$ | Inverse FT |
| Momentum dist. | $$\phi(k) = \frac{1}{\sqrt{2\pi}}\int\psi(x)e^{-ikx}dx$$ | Forward FT |
| Uncertainty | $$\Delta x \cdot \Delta k \geq \frac{1}{2}$$ | From Fourier |
| Carrier wave | $$\psi(x) = e^{ik_0 x}\psi_0(x)$$ | Envelope × oscillation |
| Parseval | $$\int|\psi|^2 dx = \int|\phi|^2 dk$$ | Norm preserved |

### Main Takeaways

1. **Wave packets** = superpositions of plane waves → localized, normalizable states
2. **Fourier transform** connects position and momentum representations
3. **Localization trade-off:** Cannot have sharp position AND sharp momentum
4. **Uncertainty principle** emerges from Fourier analysis: Δx·Δk ≥ 1/2
5. **Different distributions** (rectangular, Gaussian, Lorentzian) give different envelope shapes
6. **Gaussian** achieves minimum uncertainty Δx·Δk = 1/2

---

## Daily Checklist

- [ ] I can construct wave packets from plane wave superpositions
- [ ] I understand the Fourier transform relationship
- [ ] I can explain localization vs. momentum spread trade-off
- [ ] I understand how φ(k) determines the envelope
- [ ] I can use FFT for numerical Fourier transforms
- [ ] I successfully ran the computational lab
- [ ] I completed at least 4 practice problems

---

## Preview: Day 368

Tomorrow we focus on the special case of **Gaussian wave packets**:
- Why Gaussians are unique (minimum uncertainty)
- Complete analytical treatment
- Connection to coherent states in quantum optics
- Preparation for time evolution (Day 369)

---

*"The wave packet is quantum mechanics' answer to the question: how can something be both a wave and a particle? It's a compromise—localized enough to act like a particle, spread out enough to be a wave."*
