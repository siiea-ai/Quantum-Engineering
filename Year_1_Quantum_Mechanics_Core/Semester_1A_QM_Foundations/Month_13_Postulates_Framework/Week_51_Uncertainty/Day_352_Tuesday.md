# Day 352: Canonical Commutation Relation — [x, p] = iℏ

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Canonical Commutation & Poisson Brackets |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 352, you will be able to:

1. State and derive the canonical commutation relation [x̂, p̂] = iℏ
2. Connect quantum commutators to classical Poisson brackets
3. Explain why position and momentum are "canonically conjugate"
4. Apply the canonical commutation relation in calculations
5. Derive commutators involving functions of x̂ and p̂
6. Understand the Stone-von Neumann theorem and its implications

---

## Core Content

### 1. The Most Important Equation in Quantum Mechanics

The **canonical commutation relation** (CCR) is:

$$\boxed{[\hat{x}, \hat{p}] = i\hbar}$$

This single equation encapsulates the core of quantum mechanics:
- It implies the uncertainty principle
- It determines the structure of quantum phase space
- It distinguishes quantum from classical physics

**Historical note:** Heisenberg discovered this relation in 1925, though he initially wrote it as pq - qp = ℏ/i. Born recognized this as the key to the new mechanics.

---

### 2. Derivation from the Momentum Operator

In the position representation, the momentum operator is:

$$\hat{p} = -i\hbar\frac{d}{dx}$$

Let's verify the CCR by acting on an arbitrary wave function ψ(x):

$$[\hat{x}, \hat{p}]\psi(x) = \hat{x}\hat{p}\psi(x) - \hat{p}\hat{x}\psi(x)$$

**First term:**
$$\hat{x}\hat{p}\psi(x) = x \cdot \left(-i\hbar\frac{d\psi}{dx}\right) = -i\hbar x\frac{d\psi}{dx}$$

**Second term:**
$$\hat{p}\hat{x}\psi(x) = -i\hbar\frac{d}{dx}(x\psi(x)) = -i\hbar\left(\psi(x) + x\frac{d\psi}{dx}\right)$$

**Difference:**
$$[\hat{x}, \hat{p}]\psi(x) = -i\hbar x\frac{d\psi}{dx} + i\hbar\psi(x) + i\hbar x\frac{d\psi}{dx} = i\hbar\psi(x)$$

Since this holds for all ψ(x):

$$\boxed{[\hat{x}, \hat{p}] = i\hbar \cdot \hat{I}}$$

where Î is the identity operator (often suppressed).

---

### 3. Connection to Classical Mechanics: Poisson Brackets

In Hamiltonian mechanics, the **Poisson bracket** of two observables is:

$$\{A, B\}_{PB} = \frac{\partial A}{\partial q}\frac{\partial B}{\partial p} - \frac{\partial A}{\partial p}\frac{\partial B}{\partial q}$$

For the canonical variables q (position) and p (momentum):

$$\{q, p\}_{PB} = \frac{\partial q}{\partial q}\frac{\partial p}{\partial p} - \frac{\partial q}{\partial p}\frac{\partial p}{\partial q} = 1 \cdot 1 - 0 \cdot 0 = 1$$

**Dirac's quantization rule:**

$$\boxed{[\hat{A}, \hat{B}] = i\hbar\{A, B\}_{PB}}$$

This "canonical quantization" prescription connects classical and quantum mechanics:

$$\{q, p\}_{PB} = 1 \quad \Longrightarrow \quad [\hat{x}, \hat{p}] = i\hbar$$

**Other Poisson bracket relations:**

| Classical | Quantum |
|-----------|---------|
| {q, q} = 0 | [x̂, x̂] = 0 |
| {p, p} = 0 | [p̂, p̂] = 0 |
| {q, p} = 1 | [x̂, p̂] = iℏ |
| {f, H} = df/dt | [f̂, Ĥ]/iℏ = d⟨f̂⟩/dt |

---

### 4. The Fundamental Canonical Commutation Relations

For a particle in 3D:

$$\boxed{[\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}}$$

Explicitly:
- [x̂, p̂x] = iℏ, [ŷ, p̂y] = iℏ, [ẑ, p̂z] = iℏ
- [x̂, p̂y] = [x̂, p̂z] = [ŷ, p̂x] = ... = 0
- [x̂, ŷ] = [x̂, ẑ] = ... = 0 (positions commute)
- [p̂x, p̂y] = [p̂x, p̂z] = ... = 0 (momenta commute)

**Matrix notation:**
$$[\hat{x}_i, \hat{x}_j] = 0, \quad [\hat{p}_i, \hat{p}_j] = 0, \quad [\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}$$

---

### 5. Commutators with Functions of x̂ and p̂

Using the CCR, we can derive commutators involving any function.

#### For [x̂, f(p̂)]:

If f has a Taylor expansion:
$$[\hat{x}, f(\hat{p})] = i\hbar\frac{df}{d\hat{p}}$$

**Proof sketch:** For f(p̂) = p̂ⁿ:
$$[\hat{x}, \hat{p}^n] = i\hbar n\hat{p}^{n-1}$$

By the power rule (proven yesterday), and extending to Taylor series.

**Example:**
$$[\hat{x}, e^{i\hat{p}a/\hbar}] = i\hbar \cdot \frac{ia}{\hbar}e^{i\hat{p}a/\hbar} = -a \cdot e^{i\hat{p}a/\hbar}$$

#### For [g(x̂), p̂]:

$$[\hat{g(\hat{x})}, \hat{p}] = i\hbar\frac{dg}{d\hat{x}}$$

**Example:**
$$[\hat{x}^2, \hat{p}] = i\hbar \cdot 2\hat{x} = 2i\hbar\hat{x}$$

---

### 6. The Translation Operator

The operator that translates the wave function by distance a is:

$$\hat{T}(a) = e^{-i\hat{p}a/\hbar}$$

**Action on position eigenstates:**
$$\hat{T}(a)|x\rangle = |x + a\rangle$$

**Proof using the CCR:**

Consider:
$$[\hat{x}, \hat{T}(a)] = [\hat{x}, e^{-i\hat{p}a/\hbar}]$$

Using our formula:
$$= -(-a)e^{-i\hat{p}a/\hbar} = a\hat{T}(a)$$

Now act on |x⟩:
$$\hat{x}\hat{T}(a)|x\rangle - \hat{T}(a)\hat{x}|x\rangle = a\hat{T}(a)|x\rangle$$

$$\hat{x}\hat{T}(a)|x\rangle - x\hat{T}(a)|x\rangle = a\hat{T}(a)|x\rangle$$

$$\hat{x}(\hat{T}(a)|x\rangle) = (x + a)(\hat{T}(a)|x\rangle)$$

So T̂(a)|x⟩ is an eigenstate of x̂ with eigenvalue x + a, meaning:

$$\boxed{\hat{T}(a)|x\rangle = |x + a\rangle}$$

**Physical interpretation:** Momentum generates spatial translations. This is the quantum version of the classical relation {x, p} = 1.

---

### 7. Stone-von Neumann Theorem

A remarkable theorem states that the CCR essentially **uniquely determines** quantum mechanics.

**Stone-von Neumann Theorem (1930):**

> *Any irreducible representation of the canonical commutation relations [x̂, p̂] = iℏ on a Hilbert space is unitarily equivalent to the standard representation on L²(ℝ).*

**Implications:**

1. There is essentially one way to do quantum mechanics for a single particle
2. Position and momentum representations are equivalent (related by Fourier transform)
3. The specific form of x̂ and p̂ is fixed up to unitary equivalence

**Caveat:** This fails for infinite degrees of freedom (quantum field theory), where infinitely many inequivalent representations exist.

---

### 8. Generalization: Many Particles

For N particles with positions x̂₁, ..., x̂ₙ and momenta p̂₁, ..., p̂ₙ:

$$\boxed{[\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}}$$

$$[\hat{x}_i, \hat{x}_j] = 0, \quad [\hat{p}_i, \hat{p}_j] = 0$$

For distinguishable particles, each particle's operators commute with the others. For identical particles (bosons/fermions), additional statistics must be imposed.

---

## Physical Interpretation

### Why [x̂, p̂] = iℏ is the Heart of Quantum Mechanics

1. **Non-commutativity implies uncertainty:** If x̂ and p̂ commuted, we could measure both simultaneously with arbitrary precision. The iℏ prevents this.

2. **Wave-particle duality:** The de Broglie relation p = ℏk connects momentum to wavelength. A sharp position requires many k-values (Fourier), spreading momentum.

3. **The action quantum:** ℏ has units of action (energy × time = momentum × length). It sets the scale where quantum effects dominate.

4. **Classical limit:** When xp >> ℏ, the uncertainty product becomes negligible, and we recover classical determinism.

### Dimensional Analysis

$$[\hat{x}, \hat{p}] = i\hbar$$

Dimensions: [length][momentum] = [action] = Js = kg·m²/s

The reduced Planck constant ℏ = h/(2π) ≈ 1.055 × 10⁻³⁴ J·s sets the scale of quantum effects.

---

## Worked Examples

### Example 1: Commutator [x², p²]

**Problem:** Calculate [x̂², p̂²].

**Solution:**

Use the product rule twice. First:
$$[\hat{x}^2, \hat{p}^2] = [\hat{x}^2, \hat{p}]\hat{p} + \hat{p}[\hat{x}^2, \hat{p}]$$

We need [x̂², p̂]:
$$[\hat{x}^2, \hat{p}] = 2i\hbar\hat{x}$$

So:
$$[\hat{x}^2, \hat{p}^2] = 2i\hbar\hat{x}\hat{p} + \hat{p}(2i\hbar\hat{x})$$
$$= 2i\hbar\hat{x}\hat{p} + 2i\hbar\hat{p}\hat{x}$$
$$= 2i\hbar(\hat{x}\hat{p} + \hat{p}\hat{x})$$

Using {x̂, p̂} = x̂p̂ + p̂x̂ = 2x̂p̂ + [p̂, x̂] = 2x̂p̂ - iℏ:

$$\boxed{[\hat{x}^2, \hat{p}^2] = 2i\hbar(2\hat{x}\hat{p} - i\hbar) = 4i\hbar\hat{x}\hat{p} + 2\hbar^2}$$

Or equivalently: 2iℏ{x̂, p̂}

---

### Example 2: Hamiltonian Commutators

**Problem:** For a particle with Ĥ = p̂²/(2m) + V(x̂), calculate [Ĥ, x̂] and [Ĥ, p̂].

**Solution:**

**[Ĥ, x̂]:**
$$[\hat{H}, \hat{x}] = \left[\frac{\hat{p}^2}{2m} + V(\hat{x}), \hat{x}\right]$$

Since [V(x̂), x̂] = 0:
$$= \frac{1}{2m}[\hat{p}^2, \hat{x}] = \frac{1}{2m}(-2i\hbar\hat{p}) = -\frac{i\hbar\hat{p}}{m}$$

$$\boxed{[\hat{H}, \hat{x}] = -\frac{i\hbar\hat{p}}{m}}$$

**[Ĥ, p̂]:**
$$[\hat{H}, \hat{p}] = \left[\frac{\hat{p}^2}{2m} + V(\hat{x}), \hat{p}\right]$$

Since [p̂², p̂] = 0:
$$= [V(\hat{x}), \hat{p}] = -i\hbar\frac{dV}{d\hat{x}}$$

$$\boxed{[\hat{H}, \hat{p}] = i\hbar\frac{dV}{dx}}$$

**Ehrenfest's theorem connection:**
$$\frac{d\langle\hat{x}\rangle}{dt} = \frac{1}{i\hbar}\langle[\hat{x}, \hat{H}]\rangle = \frac{\langle\hat{p}\rangle}{m}$$

$$\frac{d\langle\hat{p}\rangle}{dt} = \frac{1}{i\hbar}\langle[\hat{p}, \hat{H}]\rangle = -\left\langle\frac{dV}{dx}\right\rangle$$

These are the quantum analogs of Hamilton's equations!

---

### Example 3: Translation Operator Verification

**Problem:** Verify that T̂(a) = exp(-ip̂a/ℏ) shifts position by a using the wave function.

**Solution:**

In position representation, p̂ = -iℏ(d/dx):

$$\hat{T}(a)\psi(x) = e^{-i\hat{p}a/\hbar}\psi(x) = e^{-a\frac{d}{dx}}\psi(x)$$

Expand the exponential:
$$= \sum_{n=0}^{\infty} \frac{1}{n!}\left(-a\frac{d}{dx}\right)^n \psi(x)$$

$$= \psi(x) - a\psi'(x) + \frac{a^2}{2!}\psi''(x) - \frac{a^3}{3!}\psi'''(x) + ...$$

This is exactly the Taylor series of ψ(x - a) about x:

$$\boxed{\hat{T}(a)\psi(x) = \psi(x - a)}$$

The wave function is shifted to the right by a, which corresponds to the particle being displaced by -a (or equivalently, the origin being shifted by +a).

---

## Practice Problems

### Level 1: Direct Application

1. **Basic CCR calculations:** Calculate:
   (a) [x̂³, p̂]
   (b) [x̂, p̂⁴]
   (c) [x̂², p̂³]

2. **Multi-dimensional:** For a 3D particle, which of these commutators are zero?
   (a) [x̂, p̂y]
   (b) [ŷ, p̂z]
   (c) [p̂x, ẑ]
   (d) [x̂, ŷ]

3. **Hamiltonian commutators:** For Ĥ = p̂²/(2m) + mω²x̂²/2 (harmonic oscillator), verify:
   (a) [Ĥ, x̂] = -iℏp̂/m
   (b) [Ĥ, p̂] = iℏmω²x̂

### Level 2: Intermediate

4. **General function commutators:** Prove that [f(x̂), p̂] = iℏf'(x̂) for any differentiable function f.

5. **Commutator of kinetic and potential:** For Ĥ = T̂ + V̂ with T̂ = p̂²/(2m) and V̂ = V(x̂), calculate [T̂, V̂].

6. **Momentum space representation:** In momentum space, x̂ = iℏ(d/dp). Verify that [x̂, p̂] = iℏ in this representation.

### Level 3: Challenging

7. **Bopp operators:** Define Â = x̂ + ip̂/(mω) and B̂ = x̂ - ip̂/(mω).
   (a) Calculate [Â, B̂].
   (b) Relate Â and B̂ to the creation/annihilation operators.

8. **Magnetic field:** In a magnetic field, the canonical momentum is p̂ - qÂ where Â is the vector potential.
   Show that [x̂i, p̂j - qÂj] = iℏδij still holds (position is unchanged).

9. **Weyl form:** The Weyl form of the CCR is:
   $$e^{i\hat{x}a/\hbar}e^{i\hat{p}b/\hbar} = e^{-iab/\hbar}e^{i\hat{p}b/\hbar}e^{i\hat{x}a/\hbar}$$
   Prove this using [x̂, p̂] = iℏ and the BCH formula.

---

## Computational Lab

### Objective
Explore the canonical commutation relation through numerical representations and verify translation operator properties.

```python
"""
Day 352 Computational Lab: Canonical Commutation Relation
Quantum Mechanics Core - Year 1, Week 51
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from typing import Tuple

# =============================================================================
# Part 1: Finite-Dimensional Approximation
# =============================================================================

print("=" * 70)
print("Part 1: Finite-Dimensional CCR Approximation")
print("=" * 70)

def create_xp_operators(N: int, L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create position and momentum operators on a finite grid.

    Parameters:
        N: Number of grid points
        L: Total length of the domain [-L/2, L/2]

    Returns:
        X: Position operator (diagonal matrix)
        P: Momentum operator (via FFT prescription)
    """
    dx = L / N
    x_values = np.linspace(-L/2 + dx/2, L/2 - dx/2, N)

    # Position operator: diagonal
    X = np.diag(x_values).astype(complex)

    # Momentum operator via spectral method
    # In momentum space, p is diagonal with values hbar * k
    # k values for FFT
    dk = 2 * np.pi / L
    if N % 2 == 0:
        k_values = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    else:
        k_values = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    # DFT matrix
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(n, n) / N) / np.sqrt(N)
    F_inv = np.exp(2j * np.pi * np.outer(n, n) / N) / np.sqrt(N)

    # P in position basis: P = F^(-1) * diag(hbar*k) * F
    # Using hbar = 1 for simplicity
    P_diag = np.diag(k_values)
    P = F_inv @ P_diag @ F

    return X, P

# Create operators
N = 64
L = 10.0
hbar = 1.0

X, P = create_xp_operators(N, L)

# Calculate commutator
comm_XP = X @ P - P @ X

print(f"\nGrid parameters: N = {N}, L = {L}")
print(f"\nCommutator [X, P]:")
print(f"Should be approximately i*hbar*I = i*{hbar:.1f}*I")

# Check diagonal elements
diag_comm = np.diag(comm_XP)
print(f"\nDiagonal elements (first 5): {diag_comm[:5]}")
print(f"Expected: {1j * hbar}")
print(f"Mean diagonal: {np.mean(diag_comm):.6f}")
print(f"Std of diagonal: {np.std(diag_comm):.2e}")

# Check off-diagonal elements
off_diag_norm = np.linalg.norm(comm_XP - np.diag(diag_comm))
print(f"\nNorm of off-diagonal elements: {off_diag_norm:.2e}")
print(f"Relative error: {off_diag_norm / np.linalg.norm(comm_XP):.2e}")

# =============================================================================
# Part 2: Verification via Direct Differentiation
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: CCR via Finite Differences")
print("=" * 70)

def create_xp_finite_diff(N: int, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create X and P using finite difference for momentum.
    """
    x_values = (np.arange(N) - N//2) * dx
    X = np.diag(x_values).astype(complex)

    # Central difference: P = -i*hbar * (psi(x+dx) - psi(x-dx)) / (2*dx)
    P = np.zeros((N, N), dtype=complex)
    for i in range(N):
        if i > 0:
            P[i, i-1] = 1
        if i < N-1:
            P[i, i+1] = -1
    P = -1j * hbar / (2 * dx) * P

    return X, P

N_fd = 50
dx_fd = 0.2
X_fd, P_fd = create_xp_finite_diff(N_fd, dx_fd)

# Calculate commutator
comm_fd = X_fd @ P_fd - P_fd @ X_fd

print(f"\nFinite difference grid: N = {N_fd}, dx = {dx_fd}")
print(f"\nDiagonal of [X, P] (middle 5 elements):")
mid = N_fd // 2
diag_fd = np.diag(comm_fd)
print(f"Values: {diag_fd[mid-2:mid+3]}")
print(f"Expected: {1j * hbar}")

# Interior elements should be close to i*hbar
interior = diag_fd[5:-5]
print(f"\nInterior mean: {np.mean(interior):.6f}")
print(f"Relative error: {np.abs(np.mean(interior) - 1j*hbar) / hbar:.2%}")

# =============================================================================
# Part 3: Translation Operator
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Translation Operator T(a) = exp(-i*P*a/hbar)")
print("=" * 70)

def gaussian_state(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    """Create a normalized Gaussian wave packet."""
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2))
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
    return psi.astype(complex)

# Create Gaussian centered at x0 = 0
N_t = 128
L_t = 20.0
dx_t = L_t / N_t
x_values = np.linspace(-L_t/2, L_t/2, N_t, endpoint=False)

X_t, P_t = create_xp_operators(N_t, L_t)

sigma = 1.0
x0 = 0.0
psi_0 = gaussian_state(x_values, x0, sigma)

# Apply translation by a = 3
a = 3.0
T_a = expm(-1j * P_t * a / hbar)
psi_translated = T_a @ psi_0

# Expected result: Gaussian centered at x0 + a = 3
psi_expected = gaussian_state(x_values, x0 + a, sigma)

print(f"Initial state: Gaussian at x0 = {x0}, sigma = {sigma}")
print(f"Translation amount: a = {a}")
print(f"\nOverlap with expected translated state: {np.abs(np.vdot(psi_translated, psi_expected)):.6f}")
print(f"(Should be close to 1.0)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(x_values, np.abs(psi_0)**2, 'b-', linewidth=2, label='Original |ψ(x)|²')
ax1.plot(x_values, np.abs(psi_translated)**2, 'r--', linewidth=2, label=f'T({a})|ψ⟩')
ax1.plot(x_values, np.abs(psi_expected)**2, 'g:', linewidth=2, label='Expected')
ax1.axvline(x=x0, color='blue', linestyle=':', alpha=0.5)
ax1.axvline(x=x0+a, color='red', linestyle=':', alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('|ψ(x)|²')
ax1.set_title(f'Translation Operator: T({a}) shifts by {a}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Test multiple translations
ax2 = axes[1]
translations = [-4, -2, 0, 2, 4]
colors = plt.cm.viridis(np.linspace(0, 1, len(translations)))

for a_test, color in zip(translations, colors):
    T_test = expm(-1j * P_t * a_test / hbar)
    psi_test = T_test @ psi_0
    ax2.plot(x_values, np.abs(psi_test)**2, color=color, linewidth=1.5,
             label=f'a = {a_test}')

ax2.set_xlabel('x')
ax2.set_ylabel('|ψ(x)|²')
ax2.set_title('Multiple Translations of Gaussian Wave Packet')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_352_translation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_352_translation.png'")

# =============================================================================
# Part 4: Commutator Identities [x^n, p] and [x, p^n]
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Power Commutator Identities")
print("=" * 70)

def verify_power_commutator(X: np.ndarray, P: np.ndarray, n: int) -> Tuple[float, float]:
    """
    Verify [X^n, P] = i*hbar*n*X^(n-1).

    Returns:
        error_Xn_P: Relative error for [X^n, P]
        error_X_Pn: Relative error for [X, P^n]
    """
    # [X^n, P] should equal i*hbar*n*X^(n-1)
    Xn = np.linalg.matrix_power(X, n)
    comm_Xn_P = Xn @ P - P @ Xn
    if n > 1:
        expected_Xn_P = 1j * hbar * n * np.linalg.matrix_power(X, n-1)
    else:
        expected_Xn_P = 1j * hbar * np.eye(X.shape[0])

    error_Xn_P = np.linalg.norm(comm_Xn_P - expected_Xn_P) / np.linalg.norm(expected_Xn_P)

    # [X, P^n] should equal i*hbar*n*P^(n-1)
    Pn = np.linalg.matrix_power(P, n)
    comm_X_Pn = X @ Pn - Pn @ X
    if n > 1:
        expected_X_Pn = 1j * hbar * n * np.linalg.matrix_power(P, n-1)
    else:
        expected_X_Pn = 1j * hbar * np.eye(P.shape[0])

    error_X_Pn = np.linalg.norm(comm_X_Pn - expected_X_Pn) / np.linalg.norm(expected_X_Pn)

    return error_Xn_P, error_X_Pn

print("\nVerifying [X^n, P] = i*hbar*n*X^(n-1) and [X, P^n] = i*hbar*n*P^(n-1)")
print("\n  n    Error [X^n, P]    Error [X, P^n]")
print("-" * 45)

for n in range(1, 6):
    err1, err2 = verify_power_commutator(X, P, n)
    print(f"  {n}    {err1:.2e}         {err2:.2e}")

# =============================================================================
# Part 5: Ehrenfest's Theorem
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Ehrenfest's Theorem: d<x>/dt = <p>/m")
print("=" * 70)

def harmonic_oscillator_hamiltonian(X: np.ndarray, P: np.ndarray,
                                    m: float, omega: float) -> np.ndarray:
    """Create harmonic oscillator Hamiltonian."""
    return P @ P / (2 * m) + m * omega**2 / 2 * X @ X

m = 1.0
omega = 1.0
H = harmonic_oscillator_hamiltonian(X, P, m, omega)

# Verify [H, X] = -i*hbar*P/m
comm_H_X = H @ X - X @ H
expected_H_X = -1j * hbar * P / m

print(f"\nHarmonic oscillator: m = {m}, omega = {omega}")
print(f"\nVerifying [H, X] = -i*hbar*P/m:")
error_H_X = np.linalg.norm(comm_H_X - expected_H_X) / np.linalg.norm(expected_H_X)
print(f"Relative error: {error_H_X:.2e}")

# Verify [H, P] = i*hbar*m*omega^2*X
comm_H_P = H @ P - P @ H
expected_H_P = 1j * hbar * m * omega**2 * X

print(f"\nVerifying [H, P] = i*hbar*m*omega^2*X:")
error_H_P = np.linalg.norm(comm_H_P - expected_H_P) / np.linalg.norm(expected_H_P)
print(f"Relative error: {error_H_P:.2e}")

# =============================================================================
# Part 6: Visualization of Commutator Structure
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualizing the Commutator [X, P]")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Real part of [X, P]
ax1 = axes[0]
im1 = ax1.imshow(np.real(comm_XP), cmap='RdBu', aspect='auto')
ax1.set_title('Re([X, P])')
ax1.set_xlabel('j')
ax1.set_ylabel('i')
plt.colorbar(im1, ax=ax1)

# Imaginary part of [X, P]
ax2 = axes[1]
im2 = ax2.imshow(np.imag(comm_XP), cmap='RdBu', aspect='auto')
ax2.set_title('Im([X, P]) - should be hbar*I')
ax2.set_xlabel('j')
ax2.set_ylabel('i')
plt.colorbar(im2, ax=ax2)

# Diagonal elements vs expected
ax3 = axes[2]
indices = np.arange(N)
ax3.plot(indices, np.imag(np.diag(comm_XP)), 'b-', linewidth=1.5, label='Computed')
ax3.axhline(y=hbar, color='r', linestyle='--', label=f'Expected (hbar = {hbar})')
ax3.set_xlabel('Grid point')
ax3.set_ylabel('Im([X, P])_ii')
ax3.set_title('Diagonal of Im([X, P])')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_352_ccr_structure.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_352_ccr_structure.png'")

# =============================================================================
# Part 7: Poisson Bracket Analogy
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Classical Poisson Bracket Correspondence")
print("=" * 70)

print("\nClassical Poisson brackets vs Quantum commutators:")
print("\n{q, p}_PB = 1  <-->  [x, p] = i*hbar")
print("{q, q}_PB = 0  <-->  [x, x] = 0")
print("{p, p}_PB = 0  <-->  [p, p] = 0")

print("\nFor H = p²/2m + V(x):")
print("{x, H}_PB = p/m  <-->  [x, H]/(i*hbar) = p/m")
print("{p, H}_PB = -dV/dx  <-->  [p, H]/(i*hbar) = -dV/dx")

print("\nDirac's quantization rule: [A, B] = i*hbar * {A, B}_PB")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Canonical commutation relation | [x̂, p̂] = iℏ |
| 3D generalization | [x̂ᵢ, p̂ⱼ] = iℏδᵢⱼ |
| Function commutator | [x̂, f(p̂)] = iℏ df/dp̂ |
| Function commutator | [g(x̂), p̂] = iℏ dg/dx̂ |
| Translation operator | T̂(a) = exp(-ip̂a/ℏ) |
| Translation action | T̂(a)\|x⟩ = \|x + a⟩ |
| Dirac's rule | [Â, B̂] = iℏ{A, B}_{PB} |

### Main Takeaways

1. **[x̂, p̂] = iℏ is the foundation** — This single equation implies the uncertainty principle and much of quantum mechanics

2. **Momentum generates translations** — The translation operator exp(-ip̂a/ℏ) shifts position by a

3. **Classical-quantum correspondence** — Commutators are i times ℏ times the corresponding Poisson brackets

4. **Uniqueness** — The Stone-von Neumann theorem shows the CCR essentially determines quantum mechanics

5. **Ehrenfest connection** — The CCR leads to quantum equations of motion matching classical form

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.2 (Canonical Quantization)
- [ ] Read Sakurai Chapter 1.6 (Position, Momentum, and Translation)
- [ ] Derive [x̂, f(p̂)] = iℏ df/dp̂ from the CCR
- [ ] Verify [Ĥ, x̂] and [Ĥ, p̂] for the harmonic oscillator
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Write a paragraph explaining how [x̂, p̂] = iℏ leads to uncertainty

---

## Preview: Day 353

Tomorrow we prove the **Generalized Uncertainty Principle**: σₐσᵦ ≥ ½|⟨[Â, B̂]⟩|. This beautiful mathematical result connects the commutator (algebraic structure) to the uncertainty product (physical observables). For position and momentum, we'll recover Heisenberg's famous Δx·Δp ≥ ℏ/2.

---

*"The product of the uncertainties of two canonically conjugate variables is at least of the order of Planck's constant."* — Werner Heisenberg (1927)

---

**Next:** [Day_353_Wednesday.md](Day_353_Wednesday.md) — Generalized Uncertainty Principle
