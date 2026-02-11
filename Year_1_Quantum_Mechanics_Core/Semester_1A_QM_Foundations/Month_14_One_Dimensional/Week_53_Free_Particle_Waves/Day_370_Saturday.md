# Day 370: Position and Momentum Space

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Dual representations, operator forms |
| Afternoon | 2.5 hours | Problem solving: Calculations in both bases |
| Evening | 2 hours | Computational lab: FFT and momentum space |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Transform between position and momentum representations** using Fourier transforms
2. **Express operators in both representations** (x̂ in p-space, p̂ in x-space)
3. **Apply completeness relations** to expand states
4. **Calculate expectation values** in either representation
5. **Use the FFT** for practical momentum-space calculations
6. **Understand the deep duality** between position and momentum

---

## Core Content

### 1. The Two Representations

A quantum state |ψ⟩ can be expressed in different bases:

**Position representation:**
$$\psi(x) = \langle x|\psi\rangle$$

**Momentum representation:**
$$\phi(p) = \langle p|\psi\rangle$$

These contain the **same information** — they are related by Fourier transform.

### 2. Fourier Transform Relations

**Position → Momentum:**

$$\boxed{\phi(p) = \langle p|\psi\rangle = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}\psi(x)dx}$$

**Momentum → Position:**

$$\boxed{\psi(x) = \langle x|\psi\rangle = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{ipx/\hbar}\phi(p)dp}$$

The kernel ⟨x|p⟩ connects the bases:

$$\langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$

### 3. Completeness Relations

**Position basis:**

$$\boxed{\int_{-\infty}^{\infty}|x\rangle\langle x|dx = \hat{1}}$$

**Momentum basis:**

$$\boxed{\int_{-\infty}^{\infty}|p\rangle\langle p|dp = \hat{1}}$$

These allow us to expand any state and insert identity operators.

### 4. Operators in Position Space

In position representation, the fundamental operators are:

**Position operator:**
$$\boxed{\hat{x}\psi(x) = x\psi(x)}$$

Just multiplication by x — position is "diagonal" in position space.

**Momentum operator:**
$$\boxed{\hat{p}\psi(x) = -i\hbar\frac{d}{dx}\psi(x)}$$

A differential operator — momentum is "off-diagonal" in position space.

### 5. Operators in Momentum Space

In momentum representation, the roles reverse:

**Momentum operator:**
$$\boxed{\hat{p}\phi(p) = p\phi(p)}$$

Just multiplication — momentum is diagonal in momentum space.

**Position operator:**
$$\boxed{\hat{x}\phi(p) = i\hbar\frac{d}{dp}\phi(p)}$$

A differential operator — position is off-diagonal in momentum space.

### 6. Derivation of Position Operator in Momentum Space

Starting from the position-space definition and Fourier transforming:

$$\langle p|\hat{x}|\psi\rangle = \int_{-\infty}^{\infty}\langle p|x\rangle x\langle x|\psi\rangle dx$$

$$= \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}x\psi(x)dx$$

Using the identity x e^{-ipx/ℏ} = iℏ (d/dp) e^{-ipx/ℏ}:

$$= i\hbar\frac{d}{dp}\left[\frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}\psi(x)dx\right]$$

$$= i\hbar\frac{d}{dp}\phi(p)$$

Therefore: **x̂ = iℏ d/dp in momentum representation**.

### 7. Symmetry of the Representations

| Quantity | Position Space | Momentum Space |
|----------|----------------|----------------|
| Wave function | ψ(x) | φ(p) |
| Position operator | x (multiply) | iℏ d/dp |
| Momentum operator | -iℏ d/dx | p (multiply) |
| Kinetic energy | -(ℏ²/2m) d²/dx² | p²/2m |
| Potential V(x) | V(x) | V(iℏ d/dp) |

The representations are **symmetric under exchange** with appropriate sign changes.

### 8. Expectation Values

**Position expectation value:**

Position space: $\langle x \rangle = \int \psi^*(x) x \psi(x) dx$

Momentum space: $\langle x \rangle = \int \phi^*(p) \left(i\hbar\frac{d}{dp}\right) \phi(p) dp$

**Momentum expectation value:**

Position space: $\langle p \rangle = \int \psi^*(x) \left(-i\hbar\frac{d}{dx}\right) \psi(x) dx$

Momentum space: $\langle p \rangle = \int \phi^*(p) p \phi(p) dp$

### 9. Kinetic Energy

The kinetic energy operator T̂ = p̂²/2m:

**Position space:**
$$\langle T \rangle = -\frac{\hbar^2}{2m}\int \psi^*(x)\frac{d^2\psi}{dx^2}dx$$

**Momentum space:**
$$\langle T \rangle = \frac{1}{2m}\int |{\phi(p)}|^2 p^2 dp$$

The momentum-space form is often simpler!

### 10. When to Use Each Representation

**Use position space when:**
- The potential V(x) is simple (e.g., harmonic oscillator)
- Boundary conditions are in position
- Visualizing localization

**Use momentum space when:**
- The kinetic term dominates (free particle)
- Computing momentum distributions
- Scattering problems (asymptotic states are plane waves)

---

## Quantum Computing Connection

### Quantum Fourier Transform

The mathematical relationship between position and momentum representations mirrors the **Quantum Fourier Transform (QFT)**:

$$|j\rangle \to \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}e^{2\pi ijk/N}|k\rangle$$

Key applications:
- **Shor's algorithm:** Period finding via QFT
- **Phase estimation:** Extracting eigenvalues
- **Quantum simulation:** Switching between position and momentum

### Continuous-Variable Systems

In CV quantum computing:
- Position quadrature x̂ and momentum quadrature p̂ are complementary observables
- Homodyne detection measures one quadrature
- Fourier transform gate: $F|x\rangle = |p\rangle$

### Digital-Analog Quantum Computing

Understanding both representations is essential for:
- Mapping problems between bases
- Choosing efficient representations for different Hamiltonians
- Optimizing quantum circuits

---

## Worked Examples

### Example 1: Momentum Operator Acting on Gaussian

**Problem:** For ψ(x) = (2πσ²)^{-1/4} exp(-x²/4σ²), calculate p̂ψ(x) and verify ⟨p⟩ = 0.

**Solution:**

Apply the momentum operator:
$$\hat{p}\psi(x) = -i\hbar\frac{d}{dx}\left[(2\pi\sigma^2)^{-1/4}e^{-x^2/4\sigma^2}\right]$$

Compute the derivative:
$$\frac{d\psi}{dx} = \psi(x) \cdot \left(-\frac{x}{2\sigma^2}\right)$$

Therefore:
$$\hat{p}\psi(x) = -i\hbar \cdot \left(-\frac{x}{2\sigma^2}\right)\psi(x) = \frac{i\hbar x}{2\sigma^2}\psi(x)$$

Now compute ⟨p⟩:
$$\langle p \rangle = \int_{-\infty}^{\infty}\psi^*(x)\hat{p}\psi(x)dx = \frac{i\hbar}{2\sigma^2}\int_{-\infty}^{\infty}x|\psi(x)|^2 dx$$

The integrand is odd (x · even function), so:

$$\boxed{\langle p \rangle = 0}$$

This confirms the centered Gaussian has zero average momentum.

### Example 2: Position Operator in Momentum Space

**Problem:** For φ(p) = (2σ²/πℏ²)^{1/4} exp(-p²σ²/ℏ²), calculate x̂φ(p) and find ⟨x⟩.

**Solution:**

Apply x̂ in momentum space:
$$\hat{x}\phi(p) = i\hbar\frac{d}{dp}\phi(p)$$

Compute the derivative:
$$\frac{d\phi}{dp} = \phi(p) \cdot \left(-\frac{2p\sigma^2}{\hbar^2}\right)$$

Therefore:
$$\hat{x}\phi(p) = i\hbar \cdot \left(-\frac{2p\sigma^2}{\hbar^2}\right)\phi(p) = -\frac{2ip\sigma^2}{\hbar}\phi(p)$$

For ⟨x⟩:
$$\langle x \rangle = \int_{-\infty}^{\infty}\phi^*(p)\hat{x}\phi(p)dp = -\frac{2i\sigma^2}{\hbar}\int_{-\infty}^{\infty}p|\phi(p)|^2 dp$$

Again, odd integrand:

$$\boxed{\langle x \rangle = 0}$$

### Example 3: Kinetic Energy via Momentum Space

**Problem:** For a Gaussian wave packet with width σ, compute ⟨T⟩ = ⟨p²⟩/2m using momentum space.

**Solution:**

The momentum-space wave function is:
$$\phi(p) = \left(\frac{2\sigma^2}{\pi\hbar^2}\right)^{1/4}e^{-p^2\sigma^2/\hbar^2}$$

Probability density:
$$|\phi(p)|^2 = \sqrt{\frac{2\sigma^2}{\pi\hbar^2}}e^{-2p^2\sigma^2/\hbar^2}$$

Compute ⟨p²⟩:
$$\langle p^2 \rangle = \int_{-\infty}^{\infty}p^2|\phi(p)|^2 dp = \sqrt{\frac{2\sigma^2}{\pi\hbar^2}}\int_{-\infty}^{\infty}p^2 e^{-2p^2\sigma^2/\hbar^2}dp$$

Using ∫x²e^{-ax²}dx = (1/2)√(π/a³):
$$\langle p^2 \rangle = \sqrt{\frac{2\sigma^2}{\pi\hbar^2}} \cdot \frac{1}{2}\sqrt{\frac{\pi\hbar^6}{8\sigma^6}} = \frac{\hbar^2}{4\sigma^2}$$

Therefore:
$$\langle T \rangle = \frac{\langle p^2 \rangle}{2m} = \frac{\hbar^2}{8m\sigma^2}$$

$$\boxed{\langle T \rangle = \frac{\hbar^2}{8m\sigma^2}}$$

Note: This equals (Δp)²/2m since Δp = ℏ/2σ for the Gaussian.

---

## Practice Problems

### Level 1: Direct Application

1. **Operator verification:** Show that [x̂, p̂] = iℏ using the position-space representations x̂ = x and p̂ = -iℏ d/dx.

2. **Transform check:** Verify that the Fourier transform of ψ(x) = A e^{-|x|/a} is φ(p) = B/(1 + (pa/ℏ)²), and find the constants A and B.

3. **Kinetic energy:** For φ(p) = √(a/π) (1/(p² + a²)) (Lorentzian), compute ⟨T⟩.

### Level 2: Intermediate

4. **Position uncertainty in p-space:** Show that Δx can be calculated from φ(p) using:
   $$\Delta x = \sqrt{\langle x^2 \rangle} = \sqrt{-\hbar^2 \int \phi^* \frac{d^2\phi}{dp^2}dp}$$
   (assuming ⟨x⟩ = 0).

5. **Hamiltonian in momentum space:** For a particle in a linear potential V(x) = Fx, write the Schrödinger equation entirely in momentum space.

6. **Plancherel theorem:** Prove that ∫|ψ(x)|²dx = ∫|φ(p)|²dp using properties of Fourier transforms.

### Level 3: Challenging

7. **Operator identity:** Prove that in momentum space, e^{iax̂/ℏ} acts as a translation: e^{iax̂/ℏ}φ(p) = φ(p + a).

8. **Mixed expectation value:** For general ψ(x), express ⟨xp + px⟩ in terms of ψ and show it equals 2⟨xp⟩ - iℏ.

9. **Momentum space potential:** For the harmonic oscillator V = mω²x²/2, write the TISE in momentum space and show it has the same form as the position-space equation (with x ↔ p).

---

## Computational Lab: FFT for Momentum Space

```python
"""
Day 370 Computational Lab: Position and Momentum Space
======================================================
Using FFT to work in momentum space and transform between representations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.integrate import simps

# Physical constants (ℏ = 1)
hbar = 1.0

# =============================================================================
# Part 1: Basic Fourier Transform
# =============================================================================

# Create position grid
N = 2048
x_max = 20.0
x = np.linspace(-x_max, x_max, N)
dx = x[1] - x[0]

# Momentum grid (from FFT frequencies)
k = fftfreq(N, dx) * 2 * np.pi  # Wave number k
p = hbar * k  # Momentum p = ℏk

# Shift to center the zero frequency
k_shifted = fftshift(k)
p_shifted = fftshift(p)

# Create a Gaussian wave function
sigma = 1.5
x0 = 0.0
p0 = 3.0

psi_x = (2 * np.pi * sigma**2)**(-0.25) * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * p0 * x / hbar)

# Fourier transform to momentum space
# Note: Need proper normalization for physical momentum space
psi_k = fftshift(fft(ifftshift(psi_x))) * dx / np.sqrt(2 * np.pi)

# For momentum normalization (p = ℏk)
phi_p = psi_k / np.sqrt(hbar)

# Analytical momentum-space wave function for comparison
sigma_p = hbar / (2 * sigma)
phi_p_analytical = (2 * sigma**2 / (np.pi * hbar**2))**0.25 * \
                   np.exp(-(p_shifted - p0)**2 / (4 * sigma_p**2)) * \
                   np.exp(-1j * (p_shifted - p0) * x0 / hbar)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Position space
ax = axes[0, 0]
ax.plot(x, np.real(psi_x), 'b-', linewidth=1.5, label='Re(ψ)')
ax.plot(x, np.imag(psi_x), 'r-', linewidth=1.5, label='Im(ψ)')
ax.plot(x, np.abs(psi_x), 'k--', linewidth=2, label='|ψ|')
ax.set_xlabel('x')
ax.set_ylabel('ψ(x)')
ax.set_title('Position Space Wave Function')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 10)

# Position probability
ax = axes[0, 1]
ax.fill_between(x, 0, np.abs(psi_x)**2, alpha=0.4)
ax.plot(x, np.abs(psi_x)**2, 'b-', linewidth=2)
ax.axvline(x0, color='r', linestyle='--', label=f'⟨x⟩ = {x0}')
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x)|²')
ax.set_title('Position Probability Density')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 10)

# Momentum space (numerical vs analytical)
ax = axes[1, 0]
ax.plot(p_shifted, np.abs(phi_p)**2, 'b-', linewidth=2, label='Numerical (FFT)')
ax.plot(p_shifted, np.abs(phi_p_analytical)**2, 'r--', linewidth=2, label='Analytical')
ax.axvline(p0, color='g', linestyle='--', alpha=0.7, label=f'⟨p⟩ = {p0}')
ax.set_xlabel('p')
ax.set_ylabel('|φ(p)|²')
ax.set_title('Momentum Probability Density')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 10)

# Phase comparison
ax = axes[1, 1]
phase_numerical = np.angle(phi_p)
phase_analytical = np.angle(phi_p_analytical)
mask = np.abs(phi_p) > 0.01 * np.max(np.abs(phi_p))
ax.plot(p_shifted[mask], phase_numerical[mask], 'b.', markersize=3, label='Numerical')
ax.plot(p_shifted[mask], phase_analytical[mask], 'r-', linewidth=1, label='Analytical')
ax.set_xlabel('p')
ax.set_ylabel('arg(φ(p))')
ax.set_title('Phase of Momentum Wave Function')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 10)

plt.tight_layout()
plt.savefig('position_momentum_transform.png', dpi=150)
plt.show()

# Verify normalization
norm_x = simps(np.abs(psi_x)**2, x)
norm_p = simps(np.abs(phi_p)**2, p_shifted)
print(f"Position space normalization: {norm_x:.6f}")
print(f"Momentum space normalization: {norm_p:.6f}")

# =============================================================================
# Part 2: Operators in Both Spaces
# =============================================================================

def position_operator_x_space(psi, x):
    """Apply x̂ in position space: x̂ψ = xψ"""
    return x * psi

def momentum_operator_x_space(psi, x, hbar=1.0):
    """Apply p̂ in position space: p̂ψ = -iℏ dψ/dx"""
    dx = x[1] - x[0]
    # Central difference for derivative
    dpsi = np.gradient(psi, dx)
    return -1j * hbar * dpsi

def position_operator_p_space(phi, p, hbar=1.0):
    """Apply x̂ in momentum space: x̂φ = iℏ dφ/dp"""
    dp = p[1] - p[0]
    dphi = np.gradient(phi, dp)
    return 1j * hbar * dphi

def momentum_operator_p_space(phi, p):
    """Apply p̂ in momentum space: p̂φ = pφ"""
    return p * phi

# Compute expectation values both ways
# ⟨p⟩ in position space
p_psi = momentum_operator_x_space(psi_x, x)
p_expect_x = np.real(simps(np.conj(psi_x) * p_psi, x))

# ⟨p⟩ in momentum space
p_expect_p = np.real(simps(np.conj(phi_p) * p_shifted * phi_p, p_shifted))

print(f"\n⟨p⟩ calculated in position space: {p_expect_x:.4f}")
print(f"⟨p⟩ calculated in momentum space: {p_expect_p:.4f}")
print(f"Expected value: {p0:.4f}")

# ⟨x⟩ in both spaces
x_expect_x = np.real(simps(np.conj(psi_x) * x * psi_x, x))
x_phi = position_operator_p_space(phi_p, p_shifted)
x_expect_p = np.real(simps(np.conj(phi_p) * x_phi, p_shifted))

print(f"\n⟨x⟩ calculated in position space: {x_expect_x:.4f}")
print(f"⟨x⟩ calculated in momentum space: {x_expect_p:.4f}")
print(f"Expected value: {x0:.4f}")

# =============================================================================
# Part 3: Kinetic Energy Calculation
# =============================================================================

# Method 1: Position space (second derivative)
d2psi = np.gradient(np.gradient(psi_x, dx), dx)
T_x_space = -hbar**2 / 2 * np.real(simps(np.conj(psi_x) * d2psi, x))

# Method 2: Momentum space (p²/2m, with m=1)
m = 1.0
T_p_space = np.real(simps(np.abs(phi_p)**2 * p_shifted**2, p_shifted)) / (2 * m)

# Analytical: ⟨T⟩ = ⟨p²⟩/2m = (p₀² + Δp²)/2m
delta_p = hbar / (2 * sigma)
T_analytical = (p0**2 + delta_p**2) / (2 * m)

print(f"\n⟨T⟩ calculated in position space: {T_x_space:.4f}")
print(f"⟨T⟩ calculated in momentum space: {T_p_space:.4f}")
print(f"Analytical value: {T_analytical:.4f}")

# =============================================================================
# Part 4: Different Wave Functions
# =============================================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Test functions
test_functions = [
    ("Gaussian", (2 * np.pi * sigma**2)**(-0.25) * np.exp(-x**2 / (4 * sigma**2))),
    ("Double Gaussian", 0.5 * ((2 * np.pi * sigma**2)**(-0.25) *
                               (np.exp(-(x-3)**2 / (4 * sigma**2)) +
                                np.exp(-(x+3)**2 / (4 * sigma**2))))),
    ("Sinc (rectangular in k)", np.sinc(x) / np.sqrt(np.pi)),
]

for i, (name, psi) in enumerate(test_functions):
    # Normalize
    norm = np.sqrt(simps(np.abs(psi)**2, x))
    psi = psi / norm

    # FFT to momentum space
    phi = fftshift(fft(ifftshift(psi))) * dx / np.sqrt(2 * np.pi * hbar)

    # Position space
    axes[i, 0].fill_between(x, 0, np.abs(psi)**2, alpha=0.4)
    axes[i, 0].plot(x, np.abs(psi)**2, 'b-', linewidth=2)
    axes[i, 0].set_xlabel('x')
    axes[i, 0].set_ylabel('|ψ(x)|²')
    axes[i, 0].set_title(f'{name}: Position Space')
    axes[i, 0].set_xlim(-10, 10)
    axes[i, 0].grid(True, alpha=0.3)

    # Momentum space
    axes[i, 1].fill_between(p_shifted, 0, np.abs(phi)**2, alpha=0.4, color='red')
    axes[i, 1].plot(p_shifted, np.abs(phi)**2, 'r-', linewidth=2)
    axes[i, 1].set_xlabel('p')
    axes[i, 1].set_ylabel('|φ(p)|²')
    axes[i, 1].set_title(f'{name}: Momentum Space')
    axes[i, 1].set_xlim(-10, 10)
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('different_wavefunctions.png', dpi=150)
plt.show()

# =============================================================================
# Part 5: Commutator Verification
# =============================================================================

print("\n" + "="*60)
print("COMMUTATOR [x̂, p̂] = iℏ VERIFICATION")
print("="*60)

# For a test function, compute [x̂, p̂]ψ = x̂p̂ψ - p̂x̂ψ
psi_test = psi_x  # Use the Gaussian

# x̂ p̂ ψ
p_psi = momentum_operator_x_space(psi_test, x)
xp_psi = position_operator_x_space(p_psi, x)

# p̂ x̂ ψ
x_psi = position_operator_x_space(psi_test, x)
px_psi = momentum_operator_x_space(x_psi, x)

# Commutator
commutator = xp_psi - px_psi

# Should equal iℏ ψ
expected = 1j * hbar * psi_test

# Compare at center (where numerical errors are smallest)
center_idx = N // 2
print(f"At x = 0:")
print(f"  [x̂, p̂]ψ = {commutator[center_idx]:.6f}")
print(f"  iℏ ψ    = {expected[center_idx]:.6f}")
print(f"  Ratio   = {commutator[center_idx] / expected[center_idx]:.6f}")

# =============================================================================
# Part 6: Time Evolution Comparison
# =============================================================================

def evolve_position_space(psi_x, x, t, m=1.0, hbar=1.0):
    """Evolve in position space using FFT method."""
    dx = x[1] - x[0]
    k = fftfreq(len(x), dx) * 2 * np.pi

    # Transform to k-space
    psi_k = fft(psi_x)

    # Apply time evolution (kinetic energy only for free particle)
    omega = hbar * k**2 / (2 * m)
    psi_k *= np.exp(-1j * omega * t)

    # Transform back
    return ifft(psi_k)

def evolve_momentum_space(phi_p, p, t, m=1.0, hbar=1.0):
    """Evolve directly in momentum space."""
    omega = p**2 / (2 * m * hbar)
    return phi_p * np.exp(-1j * omega * t)

# Initial state
psi_0 = (2 * np.pi * sigma**2)**(-0.25) * np.exp(-x**2 / (4 * sigma**2)) * np.exp(1j * p0 * x / hbar)
phi_0 = fftshift(fft(ifftshift(psi_0))) * dx / np.sqrt(2 * np.pi * hbar)

# Evolve
t = 2.0

psi_t_x = evolve_position_space(psi_0, x, t)
phi_t_p = evolve_momentum_space(phi_0, p_shifted, t)

# Transform the momentum-evolved state back to position space for comparison
phi_t_unshifted = ifftshift(phi_t_p)
psi_from_p = ifftshift(ifft(fftshift(phi_t_unshifted))) * np.sqrt(2 * np.pi * hbar) / dx

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Compare position space results
ax = axes[0]
ax.plot(x, np.abs(psi_t_x)**2, 'b-', linewidth=2, label='Evolved in x-space')
ax.plot(x, np.abs(psi_from_p)**2, 'r--', linewidth=2, label='Evolved in p-space → x')
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x,t)|²')
ax.set_title(f'Time Evolution at t = {t}\n(Both methods should agree)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 20)

# Momentum distribution (should be unchanged except for phase)
ax = axes[1]
ax.plot(p_shifted, np.abs(phi_0)**2, 'b-', linewidth=2, label='t = 0')
ax.plot(p_shifted, np.abs(phi_t_p)**2, 'r--', linewidth=2, label=f't = {t}')
ax.set_xlabel('p')
ax.set_ylabel('|φ(p,t)|²')
ax.set_title('Momentum Distribution\n(Magnitude unchanged under free evolution)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 10)

plt.tight_layout()
plt.savefig('evolution_comparison.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("""
1. Position and momentum representations contain the SAME information,
   related by Fourier transform.

2. Position operator in p-space: x̂ = iℏ d/dp (derivative)
   Momentum operator in x-space: p̂ = -iℏ d/dx (derivative)

3. Each operator is "diagonal" (multiplication) in its own representation.

4. FFT provides efficient numerical Fourier transforms: O(N log N).

5. For free particle, momentum distribution |φ(p)|² is PRESERVED
   (only phases change during evolution).

6. Kinetic energy ⟨T⟩ = ⟨p²⟩/2m is simpler in momentum space.

7. The commutator [x̂, p̂] = iℏ holds in both representations.
""")
```

---

## Summary

### Key Formulas Table

| Quantity | Position Space | Momentum Space |
|----------|----------------|----------------|
| Wave function | ψ(x) = ⟨x\|ψ⟩ | φ(p) = ⟨p\|ψ⟩ |
| Transform | $$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x)dx$$ | $$\psi(x) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{ipx/\hbar}\phi(p)dp$$ |
| x̂ operator | x (multiply) | iℏ d/dp |
| p̂ operator | -iℏ d/dx | p (multiply) |
| T̂ = p̂²/2m | -(ℏ²/2m) d²/dx² | p²/2m |

### Main Takeaways

1. **Dual representations:** Position and momentum contain equivalent information
2. **Fourier transform:** Connects ψ(x) and φ(p)
3. **Operator duality:** x̂ ↔ iℏ d/dp and p̂ ↔ -iℏ d/dx
4. **Diagonal in own basis:** x̂ multiplies in x-space, p̂ multiplies in p-space
5. **Computational:** FFT gives efficient numerical transforms
6. **Free particle:** |φ(p)|² preserved in time evolution

---

## Daily Checklist

- [ ] I can perform Fourier transforms between x and p representations
- [ ] I understand operator forms in both spaces
- [ ] I can calculate expectation values using either representation
- [ ] I understand when each representation is preferable
- [ ] I can use FFT for momentum-space calculations
- [ ] I successfully ran the computational lab
- [ ] I completed at least 4 practice problems

---

## Preview: Day 371

Tomorrow is our **Week Review and Comprehensive Lab**:
- Synthesis of all free particle and wave packet concepts
- Extended computational project: Full wave packet simulator
- Comprehensive practice problems
- Preparation for Week 54: Bound states and the infinite square well

---

*"Position and momentum are two sides of the same quantum coin — completely complementary descriptions of the same physical reality, forever linked by the Fourier transform."*
