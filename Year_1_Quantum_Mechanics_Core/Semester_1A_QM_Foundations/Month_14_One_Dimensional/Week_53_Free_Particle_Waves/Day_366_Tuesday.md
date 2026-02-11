# Day 366: Plane Waves & Normalization

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Normalization methods for continuous spectra |
| Afternoon | 2.5 hours | Problem solving: Momentum eigenstates, orthogonality |
| Evening | 2 hours | Computational lab: Normalization integrals |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain why plane waves cannot be normalized** in the conventional sense
2. **Apply box normalization** and understand the limit L → ∞
3. **Use delta-function normalization** for momentum eigenstates
4. **Prove orthogonality relations** for both normalization conventions
5. **Connect momentum and position eigenstates** through Fourier transforms
6. **Compute normalization integrals** numerically and verify theoretical predictions

---

## Core Content

### 1. The Normalization Problem

Yesterday we found plane wave solutions:

$$\psi_k(x) = Ae^{ikx}$$

Attempting standard normalization:

$$\int_{-\infty}^{\infty}|\psi_k(x)|^2 dx = |A|^2 \int_{-\infty}^{\infty}1\, dx = \infty$$

**The integral diverges!** Plane waves are **not square-integrable** and cannot be normalized in the usual way.

This is a general feature of **continuous spectra**: eigenfunctions of operators with continuous eigenvalues are not normalizable.

### 2. Physical Interpretation

Why does this happen? A plane wave represents a particle with:
- **Definite momentum** p = ℏk (Δp = 0)
- **Completely uncertain position** (Δx = ∞)

By the uncertainty principle:

$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

If Δp = 0, then Δx = ∞, meaning the particle is spread over all space. The probability of finding it in any finite region is infinitesimal—hence the non-normalizability.

### 3. Box Normalization

**Strategy:** Confine the particle to a large box of length L, normalize there, then take L → ∞.

#### Step 1: Impose Periodic Boundary Conditions

Require ψ(x + L) = ψ(x):

$$e^{ik(x+L)} = e^{ikx} \quad \Rightarrow \quad e^{ikL} = 1$$

This quantizes the wave number:

$$\boxed{k_n = \frac{2\pi n}{L}, \quad n = 0, \pm 1, \pm 2, \ldots}$$

#### Step 2: Normalize in the Box

$$\int_0^L |\psi_k(x)|^2 dx = |A|^2 L = 1 \quad \Rightarrow \quad A = \frac{1}{\sqrt{L}}$$

The normalized plane wave in a box:

$$\boxed{\psi_k(x) = \frac{1}{\sqrt{L}}e^{ikx}}$$

#### Step 3: Orthonormality

For k_n ≠ k_m:

$$\int_0^L \psi_{k_n}^*(x)\psi_{k_m}(x)dx = \frac{1}{L}\int_0^L e^{i(k_m - k_n)x}dx$$

$$= \frac{1}{L} \cdot \frac{e^{i(k_m-k_n)L} - 1}{i(k_m - k_n)} = \frac{1}{L} \cdot \frac{1 - 1}{i(k_m - k_n)} = 0$$

The box-normalized plane waves satisfy:

$$\boxed{\int_0^L \psi_{k_n}^*(x)\psi_{k_m}(x)dx = \delta_{nm}}$$

This is Kronecker delta (discrete), not Dirac delta!

#### Step 4: The Limit L → ∞

As L → ∞:
- k becomes continuous (k_n → k)
- The spacing Δk = 2π/L → 0
- Kronecker δ_nm → (2π/L)δ(k - k')

The sum over states becomes an integral:

$$\sum_n \to \frac{L}{2\pi}\int dk$$

### 4. Delta-Function Normalization

The more elegant approach: normalize to a **Dirac delta function**.

Define momentum eigenstates:

$$\boxed{|p\rangle: \quad \psi_p(x) = \langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}}$$

These satisfy:

$$\boxed{\langle p|p'\rangle = \int_{-\infty}^{\infty}\psi_p^*(x)\psi_{p'}(x)dx = \delta(p - p')}$$

**Proof:**

$$\langle p|p'\rangle = \frac{1}{2\pi\hbar}\int_{-\infty}^{\infty}e^{-ipx/\hbar}e^{ip'x/\hbar}dx = \frac{1}{2\pi\hbar}\int_{-\infty}^{\infty}e^{i(p'-p)x/\hbar}dx$$

Using the integral representation of the delta function:

$$\frac{1}{2\pi}\int_{-\infty}^{\infty}e^{i\alpha x}dx = \delta(\alpha)$$

We get:

$$\langle p|p'\rangle = \frac{1}{\hbar}\delta\left(\frac{p'-p}{\hbar}\right) = \delta(p'-p) \quad \checkmark$$

### 5. Wave Number vs. Momentum Conventions

**Wave number normalization:** Use k instead of p

$$\psi_k(x) = \frac{1}{\sqrt{2\pi}}e^{ikx}$$

$$\langle k|k'\rangle = \delta(k - k')$$

**Momentum normalization:** Use p = ℏk

$$\psi_p(x) = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$

$$\langle p|p'\rangle = \delta(p - p')$$

The factor of ℏ ensures proper delta function normalization.

### 6. Completeness Relations

The momentum eigenstates form a **complete basis**:

$$\boxed{\int_{-\infty}^{\infty}|p\rangle\langle p|dp = \hat{1}}$$

In position representation:

$$\int_{-\infty}^{\infty}\psi_p(x)\psi_p^*(x')dp = \delta(x - x')$$

This is the **resolution of identity** — any state can be expanded in momentum eigenstates.

### 7. Position Eigenstates

For comparison, position eigenstates are:

$$\langle x'|x\rangle = \delta(x - x')$$

The connection between position and momentum bases:

$$\boxed{\langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}}$$

$$\boxed{\langle p|x\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{-ipx/\hbar}}$$

These are **Fourier transform kernels**!

### 8. Expansion in Momentum Eigenstates

Any state |ψ⟩ can be written:

$$|\psi\rangle = \int_{-\infty}^{\infty}|p\rangle\langle p|\psi\rangle dp = \int_{-\infty}^{\infty}\phi(p)|p\rangle dp$$

where the **momentum-space wave function** is:

$$\boxed{\phi(p) = \langle p|\psi\rangle = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}\psi(x)dx}$$

This is the **Fourier transform** of the position-space wave function!

---

## Quantum Computing Connection

### Quantum Fourier Transform

The connection between position and momentum bases underlies the **Quantum Fourier Transform (QFT)**, one of the most important quantum algorithms:

1. **Classical DFT:** O(N²) operations
2. **FFT:** O(N log N) operations
3. **QFT:** O(log² N) quantum gates

The QFT is crucial for:
- **Shor's algorithm** for factoring
- **Phase estimation** for quantum chemistry
- **Quantum simulation** of dynamics

### Continuous-Variable Systems

In CV quantum computing:
- Position quadrature: |x⟩ states
- Momentum quadrature: |p⟩ states
- Optical phase shifter implements Fourier transform between them

---

## Worked Examples

### Example 1: Box Normalization Orthogonality

**Problem:** Verify orthogonality for box-normalized plane waves with k₁ = 2π/L and k₂ = 4π/L.

**Solution:**

Wave functions:
$$\psi_1(x) = \frac{1}{\sqrt{L}}e^{2\pi i x/L}, \quad \psi_2(x) = \frac{1}{\sqrt{L}}e^{4\pi i x/L}$$

Orthogonality integral:
$$\langle\psi_1|\psi_2\rangle = \frac{1}{L}\int_0^L e^{-2\pi i x/L}e^{4\pi i x/L}dx = \frac{1}{L}\int_0^L e^{2\pi i x/L}dx$$

$$= \frac{1}{L}\left[\frac{L}{2\pi i}e^{2\pi i x/L}\right]_0^L = \frac{1}{2\pi i}(e^{2\pi i} - 1) = \frac{1}{2\pi i}(1 - 1) = 0$$

$$\boxed{\langle\psi_1|\psi_2\rangle = 0 \quad \checkmark}$$

### Example 2: Delta-Function Normalization Verification

**Problem:** Verify that ψ_p(x) = (2πℏ)^{-1/2} e^{ipx/ℏ} gives ⟨p|p'⟩ = δ(p-p').

**Solution:**

$$\langle p|p'\rangle = \int_{-\infty}^{\infty}\psi_p^*(x)\psi_{p'}(x)dx$$

$$= \frac{1}{2\pi\hbar}\int_{-\infty}^{\infty}e^{-ipx/\hbar}e^{ip'x/\hbar}dx$$

$$= \frac{1}{2\pi\hbar}\int_{-\infty}^{\infty}e^{i(p'-p)x/\hbar}dx$$

Let u = x/ℏ, so du = dx/ℏ:

$$= \frac{1}{2\pi}\int_{-\infty}^{\infty}e^{i(p'-p)u}du = \delta(p'-p)$$

$$\boxed{\langle p|p'\rangle = \delta(p-p') \quad \checkmark}$$

### Example 3: Momentum Wave Function of a Gaussian

**Problem:** Find the momentum-space wave function for ψ(x) = (2πσ²)^{-1/4} e^{-x²/4σ²}.

**Solution:**

$$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}\psi(x)dx$$

$$= \frac{1}{\sqrt{2\pi\hbar}}\cdot\frac{1}{(2\pi\sigma^2)^{1/4}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}e^{-x^2/4\sigma^2}dx$$

Use the Gaussian integral formula:
$$\int_{-\infty}^{\infty}e^{-ax^2 + bx}dx = \sqrt{\frac{\pi}{a}}e^{b^2/4a}$$

With a = 1/(4σ²) and b = -ip/ℏ:

$$\int_{-\infty}^{\infty}e^{-x^2/4\sigma^2}e^{-ipx/\hbar}dx = \sqrt{4\pi\sigma^2}\exp\left(-\frac{p^2\sigma^2}{\hbar^2}\right)$$

Therefore:

$$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\cdot\frac{1}{(2\pi\sigma^2)^{1/4}}\cdot\sqrt{4\pi\sigma^2}\exp\left(-\frac{p^2\sigma^2}{\hbar^2}\right)$$

$$= \left(\frac{2\sigma^2}{\pi\hbar^2}\right)^{1/4}\exp\left(-\frac{p^2\sigma^2}{\hbar^2}\right)$$

$$\boxed{\phi(p) = \left(\frac{2\sigma^2}{\pi\hbar^2}\right)^{1/4}e^{-p^2\sigma^2/\hbar^2}}$$

The Fourier transform of a Gaussian is also a Gaussian! Note:
- Wide in position (large σ) → Narrow in momentum
- Narrow in position (small σ) → Wide in momentum

---

## Practice Problems

### Level 1: Direct Application

1. **Box normalization constant:** Find the normalization constant A for ψ(x) = A sin(nπx/L) in a box of length L.

2. **Wave number quantization:** In a box of L = 1 nm, what are the first three allowed wave numbers for a free particle (positive k only)?

3. **Delta function integral:** Evaluate ∫₀^∞ δ(p - p₀)f(p)dp where f(p) = p² and p₀ > 0.

### Level 2: Intermediate

4. **Orthogonality proof:** Show that box-normalized states ψₙ(x) = (2/L)^{1/2} sin(nπx/L) are orthonormal using the product-to-sum identity.

5. **Completeness check:** Verify the completeness relation by showing that
   $$\int_{-\infty}^{\infty}|\psi_p(x)|^2 dp$$
   diverges (as expected, since δ(0) = ∞).

6. **Mixed normalization:** A plane wave in a box of length L has the form ψ_n(x) = L^{-1/2} e^{ik_n x}. As L → ∞, show that the Kronecker delta in ⟨ψₙ|ψₘ⟩ = δₙₘ becomes (L/2π)δ(k - k').

### Level 3: Challenging

7. **Momentum of a finite wave:** Consider ψ(x) = A e^{ik₀x} for |x| < L/2 and ψ(x) = 0 otherwise.
   (a) Normalize the wave function.
   (b) Find the momentum-space wave function φ(p).
   (c) Show that Δp ∝ 1/L.

8. **Parseval's theorem:** Prove that ∫|ψ(x)|²dx = ∫|φ(p)|²dp for properly normalized wave functions.

9. **Operator matrix elements:** For momentum eigenstates, calculate ⟨p|x̂|p'⟩ and show that x̂ = iℏ∂/∂p in momentum representation.

---

## Computational Lab: Normalization Integrals

```python
"""
Day 366 Computational Lab: Normalization of Plane Waves
========================================================
Exploring box and delta-function normalization numerically
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps
from scipy.special import sinc

# Physical constants
hbar = 1.0

# =============================================================================
# Part 1: Box Normalization
# =============================================================================

def box_normalized_wave(x, k, L):
    """Box-normalized plane wave in [0, L]"""
    return np.exp(1j * k * x) / np.sqrt(L)

def allowed_k(n, L):
    """Allowed wave numbers for periodic boundary conditions"""
    return 2 * np.pi * n / L

# Demonstrate normalization in box
L = 10.0  # Box length
n = 3     # Quantum number
k = allowed_k(n, L)

x = np.linspace(0, L, 1000)
psi = box_normalized_wave(x, k, L)

# Verify normalization
norm_squared = simps(np.abs(psi)**2, x)
print(f"Box normalization check: ∫|ψ|²dx = {norm_squared:.6f} (should be 1)")

# Orthogonality check
n1, n2 = 2, 5
k1, k2 = allowed_k(n1, L), allowed_k(n2, L)
psi1 = box_normalized_wave(x, k1, L)
psi2 = box_normalized_wave(x, k2, L)

overlap = simps(np.conj(psi1) * psi2, x)
print(f"Orthogonality check: ⟨ψ₂|ψ₅⟩ = {np.abs(overlap):.6f} (should be 0)")

# Visualize quantized states
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, n in zip(axes.flatten(), [1, 2, 3, 4]):
    k = allowed_k(n, L)
    psi = box_normalized_wave(x, k, L)

    ax.plot(x, np.real(psi), 'b-', label='Re(ψ)', linewidth=1.5)
    ax.plot(x, np.imag(psi), 'r-', label='Im(ψ)', linewidth=1.5)
    ax.axhline(y=1/np.sqrt(L), color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=-1/np.sqrt(L), color='k', linestyle='--', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('ψ(x)')
    ax.set_title(f'n = {n}, k = 2π·{n}/L = {k:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Box-Normalized Plane Waves (L = {L})', fontsize=14)
plt.tight_layout()
plt.savefig('box_normalization.png', dpi=150)
plt.show()

# =============================================================================
# Part 2: Delta-Function Normalization
# =============================================================================

def delta_normalized_wave(x, p, hbar=1.0):
    """Delta-function normalized momentum eigenstate"""
    return np.exp(1j * p * x / hbar) / np.sqrt(2 * np.pi * hbar)

# Approximate delta function via finite integral
def approximate_delta(p, p_prime, x_max=100, n_points=10000):
    """
    Compute ⟨p|p'⟩ by integrating over finite range.
    Should approach δ(p - p') as x_max → ∞.
    """
    x = np.linspace(-x_max, x_max, n_points)
    psi_p = delta_normalized_wave(x, p)
    psi_p_prime = delta_normalized_wave(x, p_prime)

    integrand = np.conj(psi_p) * psi_p_prime
    return simps(integrand, x)

# Test orthogonality approximation
p1 = 1.0
p_range = np.linspace(-3, 5, 200)

overlaps = []
for p2 in p_range:
    overlap = approximate_delta(p1, p2, x_max=50)
    overlaps.append(np.real(overlap))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(p_range, overlaps, 'b-', linewidth=1.5)
ax.axvline(x=p1, color='r', linestyle='--', label=f'p₁ = {p1}')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

ax.set_xlabel("p'", fontsize=12)
ax.set_ylabel("⟨p|p'⟩ (approximate)", fontsize=12)
ax.set_title('Approximate Delta Function Normalization\n(Integration over finite range)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('delta_normalization.png', dpi=150)
plt.show()

# Show convergence as x_max increases
x_max_values = [10, 25, 50, 100, 200]
peak_values = []

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for x_max in x_max_values:
    overlaps = []
    for p2 in p_range:
        overlap = approximate_delta(p1, p2, x_max=x_max)
        overlaps.append(np.real(overlap))

    axes[0].plot(p_range, overlaps, label=f'x_max = {x_max}')
    peak_values.append(max(overlaps))

axes[0].set_xlabel("p'")
axes[0].set_ylabel("⟨p|p'⟩")
axes[0].set_title('Delta Function Approximation: Increasing Integration Range')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Peak height grows as x_max
axes[1].plot(x_max_values, peak_values, 'bo-', linewidth=2, markersize=8)
axes[1].plot(x_max_values, [x/np.pi for x in x_max_values], 'r--',
             label='x_max/π (theoretical)')
axes[1].set_xlabel('x_max')
axes[1].set_ylabel('Peak height')
axes[1].set_title('Peak Height vs Integration Range\n(δ(0) → ∞ as range → ∞)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('delta_convergence.png', dpi=150)
plt.show()

# =============================================================================
# Part 3: Gaussian Fourier Transform
# =============================================================================

def gaussian_position(x, sigma):
    """Normalized Gaussian in position space"""
    return (2 * np.pi * sigma**2)**(-0.25) * np.exp(-x**2 / (4 * sigma**2))

def gaussian_momentum(p, sigma, hbar=1.0):
    """Fourier transform of Gaussian (also Gaussian)"""
    sigma_p = hbar / (2 * sigma)
    return (2 * np.pi * sigma_p**2)**(-0.25) * np.exp(-p**2 / (4 * sigma_p**2))

def numerical_fourier_transform(x, psi, p_values, hbar=1.0):
    """Compute momentum-space wave function numerically"""
    phi = np.zeros_like(p_values, dtype=complex)
    dx = x[1] - x[0]

    for i, p in enumerate(p_values):
        integrand = np.exp(-1j * p * x / hbar) * psi / np.sqrt(2 * np.pi * hbar)
        phi[i] = simps(integrand, x)

    return phi

# Compare analytical and numerical Fourier transforms
sigma = 1.0
x = np.linspace(-10, 10, 1000)
p = np.linspace(-5, 5, 200)

psi_x = gaussian_position(x, sigma)
phi_p_analytical = gaussian_momentum(p, sigma)
phi_p_numerical = numerical_fourier_transform(x, psi_x, p)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Position space
axes[0].plot(x, psi_x, 'b-', linewidth=2)
axes[0].fill_between(x, 0, psi_x, alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('ψ(x)')
axes[0].set_title(f'Position Space Gaussian (σ = {sigma})')
axes[0].grid(True, alpha=0.3)

# Momentum space - analytical
axes[1].plot(p, np.real(phi_p_analytical), 'b-', linewidth=2, label='Analytical')
axes[1].plot(p, np.real(phi_p_numerical), 'ro', markersize=3, label='Numerical')
axes[1].set_xlabel('p')
axes[1].set_ylabel('φ(p)')
axes[1].set_title('Momentum Space Gaussian')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Probability densities
axes[2].plot(x, np.abs(psi_x)**2, 'b-', linewidth=2, label='|ψ(x)|²')
axes[2].plot(p, np.abs(phi_p_numerical)**2, 'r-', linewidth=2, label='|φ(p)|²')
axes[2].set_xlabel('x or p')
axes[2].set_ylabel('Probability density')
axes[2].set_title('Probability Densities')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gaussian_fourier_transform.png', dpi=150)
plt.show()

# Verify Parseval's theorem
norm_x = simps(np.abs(psi_x)**2, x)
norm_p = simps(np.abs(phi_p_numerical)**2, p)

print("\n" + "="*60)
print("PARSEVAL'S THEOREM VERIFICATION")
print("="*60)
print(f"∫|ψ(x)|² dx = {norm_x:.6f}")
print(f"∫|φ(p)|² dp = {norm_p:.6f}")
print(f"Ratio: {norm_p/norm_x:.6f} (should be 1)")

# =============================================================================
# Part 4: Width Relationship
# =============================================================================

sigma_values = [0.5, 1.0, 2.0, 4.0]

fig, axes = plt.subplots(2, len(sigma_values), figsize=(16, 8))

for i, sigma in enumerate(sigma_values):
    psi_x = gaussian_position(x, sigma)
    phi_p = numerical_fourier_transform(x, psi_x, p)

    # Position space
    axes[0, i].plot(x, np.abs(psi_x)**2, 'b-', linewidth=2)
    axes[0, i].fill_between(x, 0, np.abs(psi_x)**2, alpha=0.3)
    axes[0, i].set_xlabel('x')
    axes[0, i].set_title(f'σ_x = {sigma}')
    axes[0, i].set_xlim(-8, 8)
    if i == 0:
        axes[0, i].set_ylabel('|ψ(x)|²')

    # Momentum space
    sigma_p = hbar / (2 * sigma)
    axes[1, i].plot(p, np.abs(phi_p)**2, 'r-', linewidth=2)
    axes[1, i].fill_between(p, 0, np.abs(phi_p)**2, alpha=0.3, color='red')
    axes[1, i].set_xlabel('p')
    axes[1, i].set_title(f'σ_p = {sigma_p:.2f}')
    axes[1, i].set_xlim(-5, 5)
    if i == 0:
        axes[1, i].set_ylabel('|φ(p)|²')

plt.suptitle('Uncertainty Principle: Wide in x ↔ Narrow in p', fontsize=14)
plt.tight_layout()
plt.savefig('uncertainty_visualization.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("UNCERTAINTY PRODUCT")
print("="*60)
print(f"{'σ_x':>8} {'σ_p':>8} {'σ_x · σ_p':>12} {'ℏ/2':>8}")
print("-"*40)
for sigma in sigma_values:
    sigma_p = hbar / (2 * sigma)
    product = sigma * sigma_p
    print(f"{sigma:>8.2f} {sigma_p:>8.2f} {product:>12.4f} {hbar/2:>8.2f}")

print("\nAll products equal ℏ/2 — Gaussian achieves minimum uncertainty!")
```

---

## Summary

### Key Formulas Table

| Quantity | Formula | Notes |
|----------|---------|-------|
| Box normalization | $$\psi_k(x) = \frac{1}{\sqrt{L}}e^{ikx}$$ | k = 2πn/L quantized |
| Box orthonormality | $$\langle k_n|k_m\rangle = \delta_{nm}$$ | Kronecker delta |
| Delta normalization | $$\psi_p(x) = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$ | Continuous p |
| Delta orthonormality | $$\langle p|p'\rangle = \delta(p-p')$$ | Dirac delta |
| Completeness | $$\int|p\rangle\langle p|dp = \hat{1}$$ | Resolution of identity |
| Position-momentum | $$\langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$ | Fourier kernel |

### Main Takeaways

1. **Plane waves cannot be normalized** conventionally (integral diverges)
2. **Box normalization:** Confine to box, quantize k, normalize to 1
3. **Delta-function normalization:** ⟨p|p'⟩ = δ(p - p') for continuous spectra
4. **Momentum eigenstates** form a complete orthonormal (in delta sense) basis
5. **Fourier transform** connects position and momentum representations
6. **Gaussian is special:** Its Fourier transform is also Gaussian

---

## Daily Checklist

- [ ] I understand why plane waves are not square-integrable
- [ ] I can apply box normalization and take L → ∞
- [ ] I can verify delta-function orthonormality
- [ ] I understand the completeness relation
- [ ] I can relate position and momentum representations
- [ ] I successfully ran the computational lab
- [ ] I completed at least 4 practice problems

---

## Preview: Day 367

Tomorrow we learn how to build **wave packets** — physically meaningful states from superpositions of plane waves:

$$\psi(x) = \int_{-\infty}^{\infty}\phi(k)e^{ikx}dk$$

Wave packets are localized in both position and momentum, satisfying the uncertainty principle while describing realistic quantum particles.

---

*"The delta function is a mathematical idealization, but it captures the essential physics: a state with definite momentum is maximally uncertain in position."*
