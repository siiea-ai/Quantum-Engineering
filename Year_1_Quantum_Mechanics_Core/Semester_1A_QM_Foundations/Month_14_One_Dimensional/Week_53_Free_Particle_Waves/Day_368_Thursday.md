# Day 368: Gaussian Wave Packets

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Gaussian structure, minimum uncertainty |
| Afternoon | 2.5 hours | Problem solving: Analytical calculations |
| Evening | 2 hours | Computational lab: Gaussian packet analysis |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Write the normalized Gaussian wave packet** in position and momentum space
2. **Prove the minimum uncertainty relation** Δx·Δp = ℏ/2 for Gaussians
3. **Compute all expectation values** ⟨x⟩, ⟨p⟩, ⟨x²⟩, ⟨p²⟩ analytically
4. **Explain why Gaussians are special** from multiple perspectives
5. **Connect Gaussians to coherent states** in quantum optics
6. **Construct Gaussian wave packets** with specified properties

---

## Core Content

### 1. The Standard Gaussian Wave Packet

The normalized Gaussian wave packet centered at x = 0 with no average momentum:

$$\boxed{\psi(x) = \left(\frac{1}{2\pi\sigma^2}\right)^{1/4}e^{-x^2/4\sigma^2}}$$

where σ is the **position width parameter**.

**Verification of normalization:**

$$\int_{-\infty}^{\infty}|\psi(x)|^2 dx = \sqrt{\frac{1}{2\pi\sigma^2}}\int_{-\infty}^{\infty}e^{-x^2/2\sigma^2}dx$$

Using ∫e^{-ax²}dx = √(π/a):

$$= \sqrt{\frac{1}{2\pi\sigma^2}} \cdot \sqrt{2\pi\sigma^2} = 1 \quad \checkmark$$

### 2. General Gaussian Wave Packet

Including center position x₀ and center momentum p₀:

$$\boxed{\psi(x) = \left(\frac{1}{2\pi\sigma^2}\right)^{1/4}\exp\left[-\frac{(x-x_0)^2}{4\sigma^2} + \frac{ip_0 x}{\hbar}\right]}$$

The components:
- **Envelope:** exp[-(x-x₀)²/4σ²] — localization around x₀
- **Carrier wave:** exp(ip₀x/ℏ) — oscillation with momentum p₀

### 3. Fourier Transform: Gaussian to Gaussian

The momentum-space wave function is obtained via Fourier transform:

$$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}\psi(x)dx$$

For the standard Gaussian (x₀ = 0, p₀ = 0):

$$\boxed{\phi(p) = \left(\frac{2\sigma^2}{\pi\hbar^2}\right)^{1/4}\exp\left[-\frac{p^2\sigma^2}{\hbar^2}\right]}$$

**Key result:** The Fourier transform of a Gaussian is also a Gaussian!

**Momentum width:**

$$\sigma_p = \frac{\hbar}{2\sigma}$$

Note the inverse relationship: σ↑ → σ_p↓ (wider in position → narrower in momentum).

### 4. Expectation Values

For the general Gaussian centered at (x₀, p₀):

**Position moments:**

$$\boxed{\langle x \rangle = x_0}$$

$$\boxed{\langle x^2 \rangle = x_0^2 + \sigma^2}$$

**Momentum moments:**

$$\boxed{\langle p \rangle = p_0}$$

$$\boxed{\langle p^2 \rangle = p_0^2 + \frac{\hbar^2}{4\sigma^2}}$$

### 5. Uncertainties

**Position uncertainty:**

$$\Delta x = \sqrt{\langle x^2 \rangle - \langle x \rangle^2} = \sqrt{\sigma^2} = \sigma$$

$$\boxed{\Delta x = \sigma}$$

**Momentum uncertainty:**

$$\Delta p = \sqrt{\langle p^2 \rangle - \langle p \rangle^2} = \sqrt{\frac{\hbar^2}{4\sigma^2}} = \frac{\hbar}{2\sigma}$$

$$\boxed{\Delta p = \frac{\hbar}{2\sigma}}$$

### 6. Minimum Uncertainty

The product of uncertainties:

$$\Delta x \cdot \Delta p = \sigma \cdot \frac{\hbar}{2\sigma} = \frac{\hbar}{2}$$

$$\boxed{\Delta x \cdot \Delta p = \frac{\hbar}{2}}$$

This is the **minimum possible** value allowed by the uncertainty principle:

$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

**Gaussians saturate the uncertainty bound!**

### 7. Why Gaussians Are Special

Gaussians are unique for several reasons:

#### Mathematical Properties

1. **Self-conjugate under Fourier transform:** Gaussian → Gaussian
2. **Minimum uncertainty:** Saturate the Heisenberg bound
3. **Maximum entropy:** Among distributions with fixed variance, Gaussian has maximum entropy
4. **Central limit theorem:** Sum of many random variables → Gaussian

#### Physical Properties

1. **Ground state of harmonic oscillator** is Gaussian
2. **Coherent states** (closest to classical) are Gaussian wave packets
3. **Stable under free evolution:** Remain Gaussian (though spread)
4. **Optimal for measurement:** Minimize information-disturbance trade-off

### 8. Coherent States Connection

In quantum optics, **coherent states** |α⟩ are:
- Eigenstates of the annihilation operator: â|α⟩ = α|α⟩
- Minimum uncertainty states
- Gaussian wave packets in the harmonic oscillator basis

The ground state of the harmonic oscillator:

$$\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}\exp\left(-\frac{m\omega x^2}{2\hbar}\right)$$

is a Gaussian with σ = √(ℏ/2mω).

### 9. Probability Distributions

**Position probability density:**

$$\boxed{|\psi(x)|^2 = \frac{1}{\sqrt{2\pi}\sigma}\exp\left[-\frac{(x-x_0)^2}{2\sigma^2}\right]}$$

This is a normal distribution N(x₀, σ²).

**Momentum probability density:**

$$\boxed{|\phi(p)|^2 = \frac{1}{\sqrt{2\pi}\sigma_p}\exp\left[-\frac{(p-p_0)^2}{2\sigma_p^2}\right]}$$

This is a normal distribution N(p₀, σ_p²) with σ_p = ℏ/2σ.

### 10. Wigner Function

The Wigner quasi-probability distribution combines position and momentum information:

$$W(x,p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty}\psi^*(x+y)\psi(x-y)e^{2ipy/\hbar}dy$$

For a Gaussian:

$$\boxed{W(x,p) = \frac{1}{\pi\hbar}\exp\left[-\frac{(x-x_0)^2}{2\sigma^2} - \frac{(p-p_0)^2}{2\sigma_p^2}\right]}$$

This is a 2D Gaussian centered at (x₀, p₀) — a "blob" in phase space!

---

## Quantum Computing Connection

### Gaussian States in Continuous-Variable QC

**Gaussian states** form the foundation of continuous-variable quantum computing:

1. **Definition:** States with Gaussian Wigner function
2. **Operations:** Displacement, squeezing, rotation (all preserve Gaussianity)
3. **Measurement:** Homodyne detection measures quadratures
4. **Entanglement:** Two-mode squeezed states are Gaussian

**Important theorem:** Gaussian operations on Gaussian states can be efficiently simulated classically. Non-Gaussian resources (like photon subtraction) are needed for quantum advantage.

### Quantum Error Correction

Gaussian wave packets appear in:
- **Bosonic codes** (cat codes, GKP codes)
- **Photonic quantum computing**
- **Quantum sensing** (gravitational wave detection uses squeezed states)

---

## Worked Examples

### Example 1: Computing Position Uncertainty

**Problem:** Verify that ⟨x²⟩ = σ² for the standard Gaussian ψ(x) = (2πσ²)^{-1/4} exp(-x²/4σ²).

**Solution:**

$$\langle x^2 \rangle = \int_{-\infty}^{\infty}x^2|\psi(x)|^2 dx = \sqrt{\frac{1}{2\pi\sigma^2}}\int_{-\infty}^{\infty}x^2 e^{-x^2/2\sigma^2}dx$$

Use the Gaussian integral formula:
$$\int_{-\infty}^{\infty}x^2 e^{-ax^2}dx = \frac{1}{2}\sqrt{\frac{\pi}{a^3}}$$

With a = 1/(2σ²):

$$\int_{-\infty}^{\infty}x^2 e^{-x^2/2\sigma^2}dx = \frac{1}{2}\sqrt{\frac{\pi}{(2\sigma^2)^{-3}}} = \frac{1}{2}\sqrt{8\pi\sigma^6} = \sigma^3\sqrt{2\pi}$$

Therefore:

$$\langle x^2 \rangle = \sqrt{\frac{1}{2\pi\sigma^2}} \cdot \sigma^3\sqrt{2\pi} = \sigma^2$$

$$\boxed{\langle x^2 \rangle = \sigma^2}$$

Since ⟨x⟩ = 0 for the centered Gaussian:

$$\Delta x = \sqrt{\langle x^2 \rangle - \langle x \rangle^2} = \sqrt{\sigma^2 - 0} = \sigma$$

### Example 2: Momentum Expectation Value

**Problem:** For ψ(x) = (2πσ²)^{-1/4} exp(-x²/4σ² + ip₀x/ℏ), show that ⟨p⟩ = p₀.

**Solution:**

$$\langle p \rangle = \int_{-\infty}^{\infty}\psi^*(x)\left(-i\hbar\frac{d}{dx}\right)\psi(x)dx$$

First compute the derivative:
$$\frac{d\psi}{dx} = \left(-\frac{x}{2\sigma^2} + \frac{ip_0}{\hbar}\right)\psi(x)$$

So:
$$\hat{p}\psi = -i\hbar\left(-\frac{x}{2\sigma^2} + \frac{ip_0}{\hbar}\right)\psi = \left(\frac{i\hbar x}{2\sigma^2} + p_0\right)\psi$$

Now:
$$\langle p \rangle = \int_{-\infty}^{\infty}|\psi|^2\left(\frac{i\hbar x}{2\sigma^2} + p_0\right)dx$$

The first term vanishes (odd integrand):
$$\frac{i\hbar}{2\sigma^2}\int_{-\infty}^{\infty}x|\psi|^2 dx = \frac{i\hbar}{2\sigma^2}\langle x \rangle = 0$$

The second term gives:
$$p_0\int_{-\infty}^{\infty}|\psi|^2 dx = p_0 \cdot 1 = p_0$$

$$\boxed{\langle p \rangle = p_0}$$

### Example 3: Designing a Gaussian Wave Packet

**Problem:** Construct a Gaussian wave packet with Δx = 1 nm, centered at x₀ = 5 nm, with average momentum corresponding to kinetic energy 0.1 eV for an electron.

**Solution:**

Step 1: Position parameters
- Δx = σ = 1 nm = 10⁻⁹ m
- x₀ = 5 nm = 5 × 10⁻⁹ m

Step 2: Momentum from energy
$$E = \frac{p_0^2}{2m} \Rightarrow p_0 = \sqrt{2mE}$$

$$p_0 = \sqrt{2 \times 9.11 \times 10^{-31} \times 0.1 \times 1.6 \times 10^{-19}}$$
$$p_0 = \sqrt{2.916 \times 10^{-50}} = 1.71 \times 10^{-25} \text{ kg·m/s}$$

Step 3: Momentum uncertainty
$$\Delta p = \frac{\hbar}{2\sigma} = \frac{1.055 \times 10^{-34}}{2 \times 10^{-9}} = 5.28 \times 10^{-26} \text{ kg·m/s}$$

Step 4: Wave function (in SI units)
$$\psi(x) = \left(\frac{1}{2\pi(10^{-9})^2}\right)^{1/4}\exp\left[-\frac{(x - 5\times 10^{-9})^2}{4 \times 10^{-18}}\right]\exp\left[\frac{1.71 \times 10^{-25} x}{1.055 \times 10^{-34}}i\right]$$

Step 5: Verify minimum uncertainty
$$\Delta x \cdot \Delta p = 10^{-9} \times 5.28 \times 10^{-26} = 5.28 \times 10^{-35} \approx \frac{\hbar}{2}$$

$$\boxed{\psi(x) = (2\pi\sigma^2)^{-1/4}e^{-(x-x_0)^2/4\sigma^2}e^{ip_0 x/\hbar}}$$

with σ = 1 nm, x₀ = 5 nm, p₀ = 1.71 × 10⁻²⁵ kg·m/s.

---

## Practice Problems

### Level 1: Direct Application

1. **Normalization:** Show that ψ(x) = A exp(-x²/4σ²) is normalized when A = (2πσ²)^{-1/4}.

2. **Simple uncertainty:** A Gaussian has σ = 2 nm. What is Δp in eV/c?

3. **Phase space area:** Show that Δx·Δp = ℏ/2 corresponds to a phase space area of ℏ/2.

### Level 2: Intermediate

4. **Momentum space calculation:** Verify that the Fourier transform of ψ(x) = (2πσ²)^{-1/4} exp(-x²/4σ²) is φ(p) = (2σ²/πℏ²)^{1/4} exp(-p²σ²/ℏ²).

5. **Harmonic oscillator ground state:** The ground state of a harmonic oscillator is ψ₀(x) = (mω/πℏ)^{1/4} exp(-mωx²/2ℏ). Find Δx and Δp, and verify Δx·Δp = ℏ/2.

6. **Shifted Gaussian:** For ψ(x) = N exp[-(x-a)²/4σ² + ikx], compute ⟨x⟩, ⟨p⟩, and show the uncertainties are unchanged from the centered case.

### Level 3: Challenging

7. **Energy expectation:** For the free particle Gaussian with momentum p₀, calculate ⟨H⟩ = ⟨p²⟩/2m and ΔE = √(⟨H²⟩ - ⟨H⟩²).

8. **Squeezed states:** A "squeezed" Gaussian has the form ψ(x) = N exp(-x²/4σ² + iγx²) where γ is real. Calculate Δx and Δp and show Δx·Δp > ℏ/2.

9. **Wigner function:** Calculate the Wigner function for ψ(x) = (2πσ²)^{-1/4} exp(-x²/4σ²) and verify it integrates to 1 over phase space.

---

## Computational Lab: Gaussian Wave Packet Analysis

```python
"""
Day 368 Computational Lab: Gaussian Wave Packets
=================================================
Complete analysis of Gaussian wave packet properties
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.fft import fft, fftfreq, fftshift, ifftshift
from mpl_toolkits.mplot3d import Axes3D

# Physical constants (using ℏ = 1 for simplicity)
hbar = 1.0

# =============================================================================
# Part 1: Standard Gaussian Wave Packet
# =============================================================================

def gaussian_wavepacket(x, sigma, x0=0, p0=0, hbar=1.0):
    """
    Normalized Gaussian wave packet.

    Parameters:
    -----------
    x : array-like
        Position coordinates
    sigma : float
        Position width parameter (Δx = sigma)
    x0 : float
        Center position
    p0 : float
        Center momentum

    Returns:
    --------
    psi : complex array
        Wave function values
    """
    norm = (2 * np.pi * sigma**2)**(-0.25)
    envelope = np.exp(-(x - x0)**2 / (4 * sigma**2))
    carrier = np.exp(1j * p0 * x / hbar)
    return norm * envelope * carrier

def gaussian_momentum(p, sigma, x0=0, p0=0, hbar=1.0):
    """Analytical momentum-space Gaussian."""
    sigma_p = hbar / (2 * sigma)
    norm = (2 * sigma**2 / (np.pi * hbar**2))**0.25
    return norm * np.exp(-(p - p0)**2 / (4 * sigma_p**2)) * np.exp(-1j * (p - p0) * x0 / hbar)

# Create grids
x = np.linspace(-15, 15, 2000)
dx = x[1] - x[0]
p = np.linspace(-10, 10, 1000)

# Parameters
sigma = 1.5
x0 = 2.0
p0 = 3.0

# Generate wave packets
psi = gaussian_wavepacket(x, sigma, x0, p0)
phi_analytical = gaussian_momentum(p, sigma, x0, p0)

# Numerical Fourier transform for comparison
N = len(x)
k_fft = fftfreq(N, dx) * 2 * np.pi * hbar  # Convert to momentum
phi_numerical = fftshift(fft(ifftshift(psi))) * dx / np.sqrt(2 * np.pi * hbar)
k_fft = fftshift(k_fft)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Position space
ax = axes[0, 0]
ax.plot(x, np.real(psi), 'b-', linewidth=1.5, label='Re(ψ)')
ax.plot(x, np.imag(psi), 'r-', linewidth=1.5, label='Im(ψ)')
ax.plot(x, np.abs(psi), 'k--', linewidth=2, label='|ψ|')
ax.axvline(x0, color='g', linestyle=':', label=f'x₀ = {x0}')
ax.set_xlabel('x')
ax.set_ylabel('ψ(x)')
ax.set_title('Position Space Wave Function')
ax.legend()
ax.grid(True, alpha=0.3)

# Position probability
ax = axes[0, 1]
prob_x = np.abs(psi)**2
ax.fill_between(x, 0, prob_x, alpha=0.4)
ax.plot(x, prob_x, 'b-', linewidth=2)
ax.axvline(x0, color='r', linestyle='--', label=f'⟨x⟩ = {x0}')
ax.axvline(x0 - sigma, color='g', linestyle=':', alpha=0.7)
ax.axvline(x0 + sigma, color='g', linestyle=':', alpha=0.7, label=f'±σ = ±{sigma}')
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x)|²')
ax.set_title('Position Probability Density')
ax.legend()
ax.grid(True, alpha=0.3)

# Momentum space comparison
ax = axes[1, 0]
ax.plot(p, np.abs(phi_analytical)**2, 'b-', linewidth=2, label='Analytical')
mask = (k_fft > p.min()) & (k_fft < p.max())
ax.plot(k_fft[mask], np.abs(phi_numerical[mask])**2, 'ro', markersize=2,
        alpha=0.5, label='Numerical (FFT)')
sigma_p = hbar / (2 * sigma)
ax.axvline(p0, color='r', linestyle='--', label=f'⟨p⟩ = {p0}')
ax.axvline(p0 - sigma_p, color='g', linestyle=':', alpha=0.7)
ax.axvline(p0 + sigma_p, color='g', linestyle=':', alpha=0.7, label=f'±σ_p = ±{sigma_p:.2f}')
ax.set_xlabel('p')
ax.set_ylabel('|φ(p)|²')
ax.set_title('Momentum Probability Density')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 8)

# Verify normalization and uncertainty
ax = axes[1, 1]
ax.axis('off')

# Calculate numerical values
norm_x = simps(np.abs(psi)**2, x)
norm_p = simps(np.abs(phi_analytical)**2, p)

x_mean = simps(x * np.abs(psi)**2, x)
x2_mean = simps(x**2 * np.abs(psi)**2, x)
delta_x = np.sqrt(x2_mean - x_mean**2)

p_mean = simps(p * np.abs(phi_analytical)**2, p)
p2_mean = simps(p**2 * np.abs(phi_analytical)**2, p)
delta_p = np.sqrt(p2_mean - p_mean**2)

text = f"""
GAUSSIAN WAVE PACKET ANALYSIS
{'='*40}

Parameters:
  σ = {sigma:.3f}
  x₀ = {x0:.3f}
  p₀ = {p0:.3f}

Normalization:
  ∫|ψ(x)|² dx = {norm_x:.6f} (should be 1)
  ∫|φ(p)|² dp = {norm_p:.6f} (should be 1)

Expectation Values:
  ⟨x⟩ = {x_mean:.4f} (expected: {x0})
  ⟨p⟩ = {p_mean:.4f} (expected: {p0})

Uncertainties:
  Δx = {delta_x:.4f} (expected: {sigma})
  Δp = {delta_p:.4f} (expected: {hbar/(2*sigma):.4f})

UNCERTAINTY PRODUCT:
  Δx · Δp = {delta_x * delta_p:.6f}
  ℏ/2     = {hbar/2:.6f}

MINIMUM UNCERTAINTY ACHIEVED!
"""
ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('gaussian_analysis.png', dpi=150)
plt.show()

# =============================================================================
# Part 2: Varying Width - Uncertainty Trade-off
# =============================================================================

sigma_values = [0.5, 1.0, 2.0, 4.0]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, sigma in enumerate(sigma_values):
    psi = gaussian_wavepacket(x, sigma, x0=0, p0=0)
    phi = gaussian_momentum(p, sigma, x0=0, p0=0)
    sigma_p = hbar / (2 * sigma)

    # Position space
    axes[0, i].fill_between(x, 0, np.abs(psi)**2, alpha=0.4, color='blue')
    axes[0, i].plot(x, np.abs(psi)**2, 'b-', linewidth=2)
    axes[0, i].set_xlim(-10, 10)
    axes[0, i].set_xlabel('x')
    axes[0, i].set_title(f'σ = {sigma}\nΔx = {sigma:.2f}')
    if i == 0:
        axes[0, i].set_ylabel('|ψ(x)|²')

    # Momentum space
    axes[1, i].fill_between(p, 0, np.abs(phi)**2, alpha=0.4, color='red')
    axes[1, i].plot(p, np.abs(phi)**2, 'r-', linewidth=2)
    axes[1, i].set_xlim(-5, 5)
    axes[1, i].set_xlabel('p')
    axes[1, i].set_title(f'σ_p = {sigma_p:.2f}\nΔx·Δp = {sigma*sigma_p:.2f} = ℏ/2')
    if i == 0:
        axes[1, i].set_ylabel('|φ(p)|²')

plt.suptitle('Uncertainty Principle: Trade-off Between Position and Momentum Spread',
             fontsize=14)
plt.tight_layout()
plt.savefig('uncertainty_tradeoff.png', dpi=150)
plt.show()

# =============================================================================
# Part 3: Wigner Function
# =============================================================================

def wigner_function(x_grid, p_grid, psi_func, sigma, x0=0, p0=0):
    """
    Calculate the Wigner quasi-probability distribution.

    For a Gaussian, we use the analytical formula.
    """
    sigma_p = hbar / (2 * sigma)
    X, P = np.meshgrid(x_grid, p_grid)

    W = (1 / (np.pi * hbar)) * np.exp(-(X - x0)**2 / (2 * sigma**2)) * \
        np.exp(-(P - p0)**2 / (2 * sigma_p**2))

    return X, P, W

# Calculate Wigner function
x_wigner = np.linspace(-8, 8, 200)
p_wigner = np.linspace(-8, 8, 200)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Different (x0, p0) centers
centers = [(0, 0), (3, 2), (-2, 4)]

for ax, (x0_w, p0_w) in zip(axes, centers):
    sigma_w = 1.5
    X, P, W = wigner_function(x_wigner, p_wigner, gaussian_wavepacket,
                              sigma_w, x0_w, p0_w)

    contour = ax.contourf(X, P, W, levels=50, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, label='W(x,p)')
    ax.plot(x0_w, p0_w, 'ko', markersize=10, label=f'Center ({x0_w}, {p0_w})')

    # Draw uncertainty ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    sigma_p_w = hbar / (2 * sigma_w)
    ellipse_x = x0_w + sigma_w * np.cos(theta)
    ellipse_p = p0_w + sigma_p_w * np.sin(theta)
    ax.plot(ellipse_x, ellipse_p, 'k--', linewidth=2, label='1σ contour')

    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title(f'Wigner Function: (x₀, p₀) = ({x0_w}, {p0_w})')
    ax.legend()
    ax.set_aspect('equal')

plt.suptitle('Wigner Function for Gaussian Wave Packets (Phase Space)', fontsize=14)
plt.tight_layout()
plt.savefig('wigner_function.png', dpi=150)
plt.show()

# =============================================================================
# Part 4: 3D Wigner Visualization
# =============================================================================

fig = plt.figure(figsize=(12, 10))

sigma_3d = 1.0
x0_3d, p0_3d = 0, 0
X, P, W = wigner_function(x_wigner, p_wigner, gaussian_wavepacket,
                          sigma_3d, x0_3d, p0_3d)

ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, P, W, cmap='viridis', alpha=0.8,
                       linewidth=0, antialiased=True)

ax.set_xlabel('x')
ax.set_ylabel('p')
ax.set_zlabel('W(x,p)')
ax.set_title('Wigner Function: 3D View\nGaussian Wave Packet')
plt.colorbar(surf, ax=ax, shrink=0.5, label='W(x,p)')

plt.tight_layout()
plt.savefig('wigner_3d.png', dpi=150)
plt.show()

# =============================================================================
# Part 5: Comparison with Non-Gaussian Packets
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Gaussian
psi_gaussian = gaussian_wavepacket(x, sigma=1.5, x0=0, p0=3)

# Rectangular (sinc in momentum)
def rectangular_wavepacket(x, width, p0=0):
    # Sinc envelope with carrier
    psi = np.sinc(x / width) * np.exp(1j * p0 * x)
    norm = np.sqrt(simps(np.abs(psi)**2, x))
    return psi / norm

psi_rect = rectangular_wavepacket(x, width=1.5, p0=3)

# Double Gaussian
def double_gaussian(x, sigma, d, p0=0):
    psi = (gaussian_wavepacket(x, sigma, x0=-d, p0=p0) +
           gaussian_wavepacket(x, sigma, x0=d, p0=p0)) / np.sqrt(2)
    return psi

psi_double = double_gaussian(x, sigma=1.0, d=3, p0=3)

packets = [
    ('Gaussian', psi_gaussian),
    ('Rectangular (sinc)', psi_rect),
    ('Double Gaussian', psi_double)
]

for i, (name, psi) in enumerate(packets):
    # Compute FFT for momentum space
    N = len(x)
    k_fft = fftfreq(N, dx) * 2 * np.pi
    phi_fft = fftshift(fft(ifftshift(psi))) * dx / np.sqrt(2 * np.pi)
    k_fft = fftshift(k_fft)

    # Position space
    axes[0, i].fill_between(x, 0, np.abs(psi)**2, alpha=0.4)
    axes[0, i].plot(x, np.abs(psi)**2, 'b-', linewidth=2)
    axes[0, i].set_xlim(-10, 10)
    axes[0, i].set_xlabel('x')
    axes[0, i].set_title(f'{name}\nPosition Space')
    if i == 0:
        axes[0, i].set_ylabel('|ψ(x)|²')

    # Momentum space
    axes[1, i].fill_between(k_fft, 0, np.abs(phi_fft)**2, alpha=0.4, color='red')
    axes[1, i].plot(k_fft, np.abs(phi_fft)**2, 'r-', linewidth=2)
    axes[1, i].set_xlim(-2, 8)
    axes[1, i].set_xlabel('k')
    axes[1, i].set_title('Momentum Space')
    if i == 0:
        axes[1, i].set_ylabel('|φ(k)|²')

    # Compute uncertainty product
    prob_x = np.abs(psi)**2
    x_mean = simps(x * prob_x, x)
    x2_mean = simps(x**2 * prob_x, x)
    delta_x = np.sqrt(x2_mean - x_mean**2)

    prob_k = np.abs(phi_fft)**2
    dk = k_fft[1] - k_fft[0]
    norm_k = simps(prob_k, k_fft)
    prob_k /= norm_k
    k_mean = simps(k_fft * prob_k, k_fft)
    k2_mean = simps(k_fft**2 * prob_k, k_fft)
    delta_k = np.sqrt(k2_mean - k_mean**2)

    print(f"{name}: Δx = {delta_x:.3f}, Δk = {delta_k:.3f}, Δx·Δk = {delta_x*delta_k:.3f}")

plt.suptitle('Comparison: Gaussian vs Non-Gaussian Wave Packets', fontsize=14)
plt.tight_layout()
plt.savefig('packet_comparison.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("""
1. Gaussian wave packets have minimum uncertainty Δx·Δp = ℏ/2.

2. The Fourier transform of a Gaussian is also a Gaussian.

3. The Wigner function of a Gaussian is a 2D Gaussian in phase space.

4. Position and momentum widths are inversely related: σ_p = ℏ/(2σ).

5. Non-Gaussian packets (sinc, double Gaussian) have larger Δx·Δp.

6. Gaussians are special: minimum uncertainty, maximum entropy,
   self-Fourier, ground state of harmonic oscillator.
""")
```

---

## Summary

### Key Formulas Table

| Quantity | Formula | Notes |
|----------|---------|-------|
| Position-space Gaussian | $$\psi(x) = (2\pi\sigma^2)^{-1/4}e^{-x^2/4\sigma^2}$$ | Centered, no momentum |
| Momentum-space Gaussian | $$\phi(p) = (2\sigma^2/\pi\hbar^2)^{1/4}e^{-p^2\sigma^2/\hbar^2}$$ | FT of position Gaussian |
| Position uncertainty | $$\Delta x = \sigma$$ | Width parameter |
| Momentum uncertainty | $$\Delta p = \frac{\hbar}{2\sigma}$$ | Inverse of position width |
| Minimum uncertainty | $$\Delta x \cdot \Delta p = \frac{\hbar}{2}$$ | Gaussians are unique! |
| Wigner function | $$W(x,p) = \frac{1}{\pi\hbar}e^{-x^2/2\sigma^2}e^{-p^2/2\sigma_p^2}$$ | 2D Gaussian in phase space |

### Main Takeaways

1. **Gaussians are minimum uncertainty states:** Δx·Δp = ℏ/2 (exact equality)
2. **Fourier self-conjugacy:** FT of Gaussian = Gaussian
3. **Inverse width relationship:** σ_p = ℏ/2σ
4. **Phase space picture:** Wigner function is 2D Gaussian "blob"
5. **Physical importance:** Ground state of harmonic oscillator, coherent states
6. **Quantum computing relevance:** Foundation of CV quantum computing

---

## Daily Checklist

- [ ] I can write normalized Gaussian wave packets
- [ ] I understand why Δx·Δp = ℏ/2 for Gaussians
- [ ] I can perform the Fourier transform analytically
- [ ] I understand the inverse width relationship
- [ ] I know why Gaussians are special
- [ ] I successfully ran the computational lab
- [ ] I completed at least 4 practice problems

---

## Preview: Day 369

Tomorrow we study **wave packet dynamics** — how Gaussian packets evolve in time under free particle propagation:
- Time-dependent Schrödinger equation
- Phase velocity vs. group velocity
- Wave packet spreading (dispersion)
- Classical correspondence through group velocity

---

*"The Gaussian wave packet is nature's sweet spot — the perfect balance between position and momentum knowledge, achieving the minimum uncertainty allowed by quantum mechanics."*
