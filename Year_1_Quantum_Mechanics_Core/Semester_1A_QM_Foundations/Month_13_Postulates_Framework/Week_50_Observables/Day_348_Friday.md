# Day 348: Position and Momentum Operators

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Position and Momentum in Quantum Mechanics |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving and Representations |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Wave Functions and Operators |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Define the position operator $\hat{x}$ and understand its action on wave functions
2. Derive the momentum operator $\hat{p} = -i\hbar\frac{d}{dx}$ in position representation
3. Solve eigenvalue equations for position and momentum operators
4. Understand the relationship $\langle x|\psi\rangle = \psi(x)$
5. Work with both position and momentum representations
6. Verify the canonical commutation relation $[\hat{x}, \hat{p}] = i\hbar$

---

## Required Reading

### Primary Texts
- **Shankar, Chapter 4.3**: Position and Momentum Operators (pp. 171-185)
- **Sakurai, Chapter 1.6**: Position, Momentum, and Translation (pp. 41-55)
- **Griffiths, Chapter 3.3**: Continuous Spectra (pp. 100-105)

### Supplementary Reading
- **Cohen-Tannoudji, Chapter II.D**: Position and Momentum Representations
- **Dirac, Chapter III**: Continuous Spectra

---

## Core Content: Theory and Concepts

### 1. The Position Observable

In classical mechanics, position $x$ is a continuous observable. In quantum mechanics, position becomes an operator $\hat{x}$.

**Key Question:** What are the eigenstates and eigenvalues of $\hat{x}$?

**Postulate:** The position operator $\hat{x}$ has a continuous spectrum of eigenvalues $x' \in \mathbb{R}$ with eigenstates $|x'\rangle$:

$$\boxed{\hat{x}|x'\rangle = x'|x'\rangle}$$

### 2. Properties of Position Eigenstates

**Orthonormality (Dirac delta):**

$$\langle x'|x''\rangle = \delta(x' - x'')$$

**Completeness:**

$$\int_{-\infty}^{\infty} |x'\rangle\langle x'| \, dx' = \hat{I}$$

**Physical interpretation:** $|x'\rangle$ represents a particle perfectly localized at position $x'$.

**Important:** Position eigenstates are **not normalizable** in the usual sense:

$$\langle x'|x'\rangle = \delta(0) = \infty$$

They are "improper" or "generalized" eigenstates, useful for expanding proper states.

### 3. The Wave Function

**Definition:** For any state $|\psi\rangle$, the **wave function** is:

$$\boxed{\psi(x) \equiv \langle x|\psi\rangle}$$

This is the probability amplitude for finding the particle at position $x$.

**Probability density:**

$$|\psi(x)|^2 \, dx = \text{probability of finding particle between } x \text{ and } x + dx$$

**Normalization:**

$$\int_{-\infty}^{\infty} |\psi(x)|^2 \, dx = \langle\psi|\psi\rangle = 1$$

### 4. Position Operator in Position Representation

How does $\hat{x}$ act on wave functions?

For any state $|\psi\rangle$:

$$\langle x|\hat{x}|\psi\rangle = x\langle x|\psi\rangle = x\psi(x)$$

Therefore, in position representation:

$$\boxed{\hat{x}\psi(x) = x\psi(x)}$$

The position operator simply multiplies by $x$.

### 5. The Momentum Observable

**Classical momentum:** $p = mv$ is also continuous.

**Quantum momentum operator:** $\hat{p}$ with eigenvalue equation:

$$\hat{p}|p'\rangle = p'|p'\rangle$$

**Momentum eigenstates** have properties:

$$\langle p'|p''\rangle = \delta(p' - p'')$$

$$\int_{-\infty}^{\infty} |p'\rangle\langle p'| \, dp' = \hat{I}$$

### 6. Deriving the Momentum Operator

**Key insight:** The momentum operator is the generator of spatial translations.

**Translation operator:**

$$\hat{T}(a)|x\rangle = |x + a\rangle$$

For infinitesimal translation $dx$:

$$\hat{T}(dx) = \hat{I} - \frac{i}{\hbar}\hat{p} \, dx$$

**Derivation of $\hat{p}$ in position representation:**

Consider $\langle x|\hat{T}(dx)|\psi\rangle$:

$$\langle x|\hat{T}(dx)|\psi\rangle = \langle x - dx|\psi\rangle = \psi(x - dx)$$

Taylor expanding:

$$\psi(x - dx) = \psi(x) - dx\frac{d\psi}{dx}$$

From $\hat{T}(dx) = \hat{I} - \frac{i}{\hbar}\hat{p} \, dx$:

$$\langle x|\hat{T}(dx)|\psi\rangle = \psi(x) - \frac{i}{\hbar}dx\langle x|\hat{p}|\psi\rangle$$

Comparing:

$$\langle x|\hat{p}|\psi\rangle = -i\hbar\frac{d\psi}{dx}$$

Therefore:

$$\boxed{\hat{p} = -i\hbar\frac{d}{dx} \quad \text{(position representation)}}$$

### 7. Momentum Eigenfunctions in Position Space

Solve the eigenvalue equation:

$$\hat{p}\psi_p(x) = p\psi_p(x)$$

$$-i\hbar\frac{d\psi_p}{dx} = p\psi_p(x)$$

This is a first-order ODE with solution:

$$\psi_p(x) = Ae^{ipx/\hbar}$$

**Normalization:** Using $\delta$-function normalization:

$$\langle p'|p\rangle = \int_{-\infty}^{\infty} \psi_{p'}^*(x)\psi_p(x) \, dx = |A|^2 \cdot 2\pi\hbar\delta(p - p')$$

Choosing $A = \frac{1}{\sqrt{2\pi\hbar}}$:

$$\boxed{\psi_p(x) = \langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}}$$

**Physical interpretation:** Momentum eigenstates are **plane waves** with wavelength $\lambda = 2\pi\hbar/p = h/p$ (de Broglie relation).

### 8. The Canonical Commutation Relation

**Theorem:** The position and momentum operators satisfy:

$$\boxed{[\hat{x}, \hat{p}] = i\hbar}$$

**Proof (in position representation):**

For any wave function $\psi(x)$:

$$[\hat{x}, \hat{p}]\psi(x) = \hat{x}(\hat{p}\psi) - \hat{p}(\hat{x}\psi)$$

$$= x\left(-i\hbar\frac{d\psi}{dx}\right) - \left(-i\hbar\frac{d}{dx}\right)(x\psi)$$

$$= -i\hbar x\frac{d\psi}{dx} + i\hbar\left(\psi + x\frac{d\psi}{dx}\right)$$

$$= i\hbar\psi$$

Therefore $[\hat{x}, \hat{p}] = i\hbar \cdot \hat{I}$. $\blacksquare$

**Consequence:** Position and momentum are **incompatible observables**. They cannot be simultaneously measured with arbitrary precision.

### 9. Position and Momentum Representations

We can describe quantum mechanics in different "representations":

**Position representation:**
- Basis: $\{|x\rangle\}$
- Wave function: $\psi(x) = \langle x|\psi\rangle$
- Operators: $\hat{x} \to x$, $\hat{p} \to -i\hbar\frac{d}{dx}$

**Momentum representation:**
- Basis: $\{|p\rangle\}$
- Wave function: $\phi(p) = \langle p|\psi\rangle$
- Operators: $\hat{p} \to p$, $\hat{x} \to i\hbar\frac{d}{dp}$

The two representations are related by **Fourier transform** (tomorrow's topic).

### 10. Inner Product in Position Representation

For states $|\psi\rangle$ and $|\phi\rangle$:

$$\langle\phi|\psi\rangle = \int_{-\infty}^{\infty} \langle\phi|x\rangle\langle x|\psi\rangle \, dx = \int_{-\infty}^{\infty} \phi^*(x)\psi(x) \, dx$$

**Expectation values:**

$$\langle\hat{x}\rangle = \int_{-\infty}^{\infty} |\psi(x)|^2 x \, dx$$

$$\langle\hat{p}\rangle = \int_{-\infty}^{\infty} \psi^*(x)\left(-i\hbar\frac{d\psi}{dx}\right) dx$$

### 11. Hermiticity of Position and Momentum

**Position operator is Hermitian:**

$$\langle\phi|\hat{x}|\psi\rangle = \int \phi^*(x) \cdot x \cdot \psi(x) \, dx = \int (x\phi(x))^* \psi(x) \, dx = \langle\hat{x}\phi|\psi\rangle$$

**Momentum operator is Hermitian** (for proper boundary conditions):

$$\langle\phi|\hat{p}|\psi\rangle = \int \phi^*(x)\left(-i\hbar\frac{d\psi}{dx}\right) dx$$

Using integration by parts (with boundary terms vanishing):

$$= \int \left(i\hbar\frac{d\phi^*}{dx}\right) \psi(x) \, dx = \int \left(-i\hbar\frac{d\phi}{dx}\right)^* \psi(x) \, dx = \langle\hat{p}\phi|\psi\rangle$$

### 12. Extension to Three Dimensions

In 3D, we have three position operators $\hat{x}, \hat{y}, \hat{z}$ and three momentum operators $\hat{p}_x, \hat{p}_y, \hat{p}_z$.

**Commutation relations:**

$$[\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}$$

$$[\hat{x}_i, \hat{x}_j] = 0, \quad [\hat{p}_i, \hat{p}_j] = 0$$

The position components are compatible with each other, as are the momentum components.

Position and momentum in the **same direction** are incompatible.

---

## Quantum Computing Connection

### Position and Momentum in Discrete Systems

Quantum computers work with discrete systems, but the position-momentum structure has analogues:

**Qudit systems:** For a $d$-dimensional Hilbert space:

$$|j\rangle, \quad j = 0, 1, ..., d-1$$

Define "position" and "momentum" operators:

$$\hat{X}|j\rangle = \omega^j|j\rangle, \quad \omega = e^{2\pi i/d}$$

$$\hat{Z}|j\rangle = |j+1 \mod d\rangle$$

These satisfy $\hat{X}\hat{Z} = \omega\hat{Z}\hat{X}$ (discrete analogue of $[\hat{x}, \hat{p}] = i\hbar$).

### Continuous-Variable Quantum Computing

Some quantum computing architectures use **continuous variables**:

- **Optical quantum computing:** Uses quadrature operators $\hat{X}$ and $\hat{P}$ of electromagnetic field
- **Bosonic codes:** Encode qubits in harmonic oscillator states (e.g., GKP codes)

These directly use the position-momentum structure:

$$[\hat{X}, \hat{P}] = i$$

**Gaussian operations** (squeezing, displacement) are implemented through linear optics.

### The Phase Space Picture

In classical mechanics, phase space is $(x, p)$ space. In quantum mechanics:

- Cannot have definite $(x, p)$ simultaneously
- **Wigner function** provides quasi-probability distribution on phase space
- Gaussian states are fully characterized by mean and covariance in phase space

---

## Worked Examples

### Example 1: Position Measurement Probability

**Problem:** A particle has wave function $\psi(x) = Ae^{-x^2/2a^2}$. Find:
(a) The normalization constant $A$
(b) The probability of finding the particle between $x = 0$ and $x = a$
(c) $\langle x\rangle$ and $\langle x^2\rangle$

**Solution:**

**(a) Normalization:**

$$\int_{-\infty}^{\infty} |A|^2 e^{-x^2/a^2} dx = 1$$

Using $\int_{-\infty}^{\infty} e^{-\alpha x^2} dx = \sqrt{\frac{\pi}{\alpha}}$:

$$|A|^2 \sqrt{\pi a^2} = 1 \implies A = \left(\frac{1}{\pi a^2}\right)^{1/4}$$

**(b) Probability:**

$$P(0 \leq x \leq a) = \int_0^a |A|^2 e^{-x^2/a^2} dx = \frac{1}{\sqrt{\pi}} \int_0^1 e^{-u^2} du = \frac{1}{\sqrt{\pi}} \cdot \frac{\sqrt{\pi}}{2} \text{erf}(1)$$

$$P = \frac{1}{2}\text{erf}(1) \approx 0.421$$

**(c) Expectation values:**

By symmetry (even function): $\langle x\rangle = 0$

$$\langle x^2\rangle = \int_{-\infty}^{\infty} x^2 |A|^2 e^{-x^2/a^2} dx = \frac{1}{\sqrt{\pi a^2}} \cdot \frac{\sqrt{\pi}a^3}{2} = \frac{a^2}{2}$$

### Example 2: Momentum Expectation Value

**Problem:** For the Gaussian wave function $\psi(x) = \left(\frac{1}{\pi a^2}\right)^{1/4} e^{-x^2/2a^2}$, calculate $\langle p\rangle$ and $\langle p^2\rangle$.

**Solution:**

**For $\langle p\rangle$:**

$$\langle p\rangle = \int_{-\infty}^{\infty} \psi^*(x)\left(-i\hbar\frac{d\psi}{dx}\right) dx$$

$$\frac{d\psi}{dx} = -\frac{x}{a^2}\psi(x)$$

$$\langle p\rangle = \int_{-\infty}^{\infty} \psi^*(x)\left(-i\hbar\right)\left(-\frac{x}{a^2}\right)\psi(x) dx = \frac{i\hbar}{a^2}\int_{-\infty}^{\infty} x|\psi(x)|^2 dx$$

The integrand is odd, so $\langle p\rangle = 0$.

**For $\langle p^2\rangle$:**

$$\langle p^2\rangle = \int_{-\infty}^{\infty} \psi^*\left(-\hbar^2\frac{d^2\psi}{dx^2}\right) dx$$

$$\frac{d^2\psi}{dx^2} = \frac{d}{dx}\left(-\frac{x}{a^2}\psi\right) = -\frac{1}{a^2}\psi + \frac{x^2}{a^4}\psi = \left(\frac{x^2}{a^4} - \frac{1}{a^2}\right)\psi$$

$$\langle p^2\rangle = -\hbar^2\left(\frac{\langle x^2\rangle}{a^4} - \frac{1}{a^2}\right) = -\hbar^2\left(\frac{a^2/2}{a^4} - \frac{1}{a^2}\right) = \frac{\hbar^2}{2a^2}$$

**Uncertainty check:**

$$\Delta x = \sqrt{\langle x^2\rangle - \langle x\rangle^2} = \frac{a}{\sqrt{2}}$$

$$\Delta p = \sqrt{\langle p^2\rangle - \langle p\rangle^2} = \frac{\hbar}{a\sqrt{2}}$$

$$\Delta x \cdot \Delta p = \frac{a}{\sqrt{2}} \cdot \frac{\hbar}{a\sqrt{2}} = \frac{\hbar}{2}$$

This is a **minimum uncertainty state** (saturates the Heisenberg bound).

### Example 3: Verifying Momentum Eigenfunction

**Problem:** Verify that $\psi_p(x) = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$ is an eigenfunction of $\hat{p}$ with eigenvalue $p$.

**Solution:**

Apply $\hat{p}$:

$$\hat{p}\psi_p(x) = -i\hbar\frac{d}{dx}\left(\frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}\right)$$

$$= -i\hbar \cdot \frac{1}{\sqrt{2\pi\hbar}} \cdot \frac{ip}{\hbar}e^{ipx/\hbar}$$

$$= \frac{-i \cdot i \cdot p}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$

$$= p \cdot \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$

$$= p\psi_p(x)$$

Yes, $\psi_p$ is an eigenfunction of $\hat{p}$ with eigenvalue $p$. $\checkmark$

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate $[\hat{x}^2, \hat{p}]$.

*Hint: Use $[\hat{A}\hat{B}, \hat{C}] = \hat{A}[\hat{B}, \hat{C}] + [\hat{A}, \hat{C}]\hat{B}$*

**Answer:** $[\hat{x}^2, \hat{p}] = 2i\hbar\hat{x}$

---

**Problem 1.2:** Find $[\hat{x}, \hat{p}^2]$.

**Answer:** $[\hat{x}, \hat{p}^2] = 2i\hbar\hat{p}$

---

**Problem 1.3:** For wave function $\psi(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{\pi x}{L}\right)$ on $[0, L]$, calculate $\langle x\rangle$.

**Answer:** $\langle x\rangle = \frac{L}{2}$ (by symmetry about $x = L/2$)

---

### Level 2: Intermediate

**Problem 2.1:** Show that for a real wave function $\psi(x)$, $\langle p\rangle = 0$.

**Solution:**
$$\langle p\rangle = \int \psi^*(x)\left(-i\hbar\frac{d\psi}{dx}\right)dx = -i\hbar\int \psi\frac{d\psi}{dx}dx$$

For real $\psi$: $\int \psi\frac{d\psi}{dx}dx = \frac{1}{2}\int \frac{d(\psi^2)}{dx}dx = \frac{1}{2}[\psi^2]_{-\infty}^{\infty} = 0$

So $\langle p\rangle = 0$. $\checkmark$

---

**Problem 2.2:** The wave function is $\psi(x) = Ne^{-|x|/a}$. Find $N$, $\langle x\rangle$, and $\langle x^2\rangle$.

**Answer:**
- $N = 1/\sqrt{a}$
- $\langle x\rangle = 0$ (symmetric about origin)
- $\langle x^2\rangle = a^2$

---

**Problem 2.3:** A particle has wave function $\psi(x) = Ne^{ip_0x/\hbar}e^{-x^2/2a^2}$. Calculate $\langle p\rangle$.

**Answer:** $\langle p\rangle = p_0$ (the exponential factor shifts momentum by $p_0$)

---

### Level 3: Challenging

**Problem 3.1:** Prove that $[\hat{x}^n, \hat{p}] = i\hbar n\hat{x}^{n-1}$ for any positive integer $n$.

**Solution:** Use induction.

Base case: $[\hat{x}, \hat{p}] = i\hbar$ $\checkmark$

Inductive step: Assume $[\hat{x}^k, \hat{p}] = i\hbar k\hat{x}^{k-1}$.

$$[\hat{x}^{k+1}, \hat{p}] = [\hat{x}\hat{x}^k, \hat{p}] = \hat{x}[\hat{x}^k, \hat{p}] + [\hat{x}, \hat{p}]\hat{x}^k$$
$$= \hat{x}(i\hbar k\hat{x}^{k-1}) + i\hbar\hat{x}^k = i\hbar(k+1)\hat{x}^k$$ $\checkmark$

---

**Problem 3.2:** Show that in momentum representation, $\hat{x} = i\hbar\frac{d}{dp}$.

**Solution:** Consider $\langle p|\hat{x}|\psi\rangle$. Using completeness:

$$\langle p|\hat{x}|\psi\rangle = \int dx \langle p|x\rangle x \langle x|\psi\rangle$$

With $\langle p|x\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{-ipx/\hbar}$:

$$= \frac{1}{\sqrt{2\pi\hbar}}\int dx \, e^{-ipx/\hbar} x \psi(x)$$

Note that $x e^{-ipx/\hbar} = i\hbar\frac{d}{dp}e^{-ipx/\hbar}$.

$$= i\hbar\frac{d}{dp}\left[\frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x)dx\right] = i\hbar\frac{d\phi(p)}{dp}$$

where $\phi(p) = \langle p|\psi\rangle$. Therefore $\hat{x} = i\hbar\frac{d}{dp}$ in momentum representation.

---

**Problem 3.3:** The commutator $[\hat{x}, f(\hat{p})] = ?$ for a function $f(\hat{p})$ that can be Taylor expanded.

**Solution:** Using $[\hat{x}, \hat{p}^n] = i\hbar n\hat{p}^{n-1}$:

$$[\hat{x}, f(\hat{p})] = \sum_n \frac{f^{(n)}(0)}{n!}[\hat{x}, \hat{p}^n] = i\hbar\sum_n \frac{f^{(n)}(0)}{(n-1)!}\hat{p}^{n-1} = i\hbar f'(\hat{p})$$

$$\boxed{[\hat{x}, f(\hat{p})] = i\hbar\frac{df}{d\hat{p}}}$$

---

## Computational Lab: Wave Functions and Operators

```python
"""
Day 348 Computational Lab: Position and Momentum
Topics: Wave functions, operators, expectation values
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps
from scipy.special import erf
from numpy.fft import fft, fftfreq, fftshift

# Physical constants (using hbar = 1 for simplicity)
hbar = 1.0

# ============================================
# Part 1: Wave Function Basics
# ============================================

print("=" * 50)
print("Part 1: Gaussian Wave Function")
print("=" * 50)

def gaussian_wavefunction(x, a):
    """Normalized Gaussian wave function"""
    norm = (1 / (np.pi * a**2))**0.25
    return norm * np.exp(-x**2 / (2 * a**2))

# Create position grid
a = 1.0  # Width parameter
x = np.linspace(-5*a, 5*a, 1000)
dx = x[1] - x[0]

psi = gaussian_wavefunction(x, a)

# Verify normalization
norm_check = simps(np.abs(psi)**2, x)
print(f"Normalization check: {norm_check:.6f} (should be 1.0)")

# Plot wave function and probability density
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, psi.real, 'b-', linewidth=2, label=r'Re($\psi$)')
ax1.plot(x, psi.imag, 'r--', linewidth=2, label=r'Im($\psi$)')
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\psi(x)$')
ax1.set_title('Gaussian Wave Function')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.fill_between(x, 0, np.abs(psi)**2, alpha=0.5, color='blue')
ax2.plot(x, np.abs(psi)**2, 'b-', linewidth=2)
ax2.set_xlabel('x')
ax2.set_ylabel(r'$|\psi(x)|^2$')
ax2.set_title('Probability Density')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gaussian_wavefunction.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 2: Expectation Values
# ============================================

print("\n" + "=" * 50)
print("Part 2: Expectation Values")
print("=" * 50)

# <x>
x_mean = simps(np.abs(psi)**2 * x, x)
print(f"<x> = {x_mean:.6f} (expected: 0)")

# <x^2>
x2_mean = simps(np.abs(psi)**2 * x**2, x)
print(f"<x^2> = {x2_mean:.6f} (expected: {a**2/2:.6f})")

# Delta x
delta_x = np.sqrt(x2_mean - x_mean**2)
print(f"Delta x = {delta_x:.6f} (expected: {a/np.sqrt(2):.6f})")

# <p> using derivative
dpsi_dx = np.gradient(psi, dx)
p_mean = simps(np.conj(psi) * (-1j * hbar * dpsi_dx), x).real
print(f"<p> = {p_mean:.6f} (expected: 0)")

# <p^2>
d2psi_dx2 = np.gradient(dpsi_dx, dx)
p2_mean = simps(np.conj(psi) * (-hbar**2 * d2psi_dx2), x).real
print(f"<p^2> = {p2_mean:.6f} (expected: {hbar**2/(2*a**2):.6f})")

# Delta p
delta_p = np.sqrt(p2_mean - p_mean**2)
print(f"Delta p = {delta_p:.6f} (expected: {hbar/(a*np.sqrt(2)):.6f})")

# Uncertainty product
uncertainty_product = delta_x * delta_p
print(f"\nDelta x * Delta p = {uncertainty_product:.6f}")
print(f"Heisenberg limit (hbar/2) = {hbar/2:.6f}")
print(f"Ratio to minimum: {uncertainty_product/(hbar/2):.4f}")

# ============================================
# Part 3: Position Operator Action
# ============================================

print("\n" + "=" * 50)
print("Part 3: Position Operator x*psi(x)")
print("=" * 50)

# x|psi> = x * psi(x)
x_psi = x * psi

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, psi.real, 'b-', linewidth=2, label=r'$\psi(x)$')
ax.plot(x, x_psi.real, 'r--', linewidth=2, label=r'$\hat{x}\psi(x) = x\psi(x)$')
ax.set_xlabel('x')
ax.set_ylabel('Function value')
ax.set_title('Action of Position Operator')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('position_operator.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 4: Momentum Operator Action
# ============================================

print("\n" + "=" * 50)
print("Part 4: Momentum Operator -ihbar * d/dx")
print("=" * 50)

# p|psi> = -i*hbar * d(psi)/dx
p_psi = -1j * hbar * dpsi_dx

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, psi.real, 'b-', linewidth=2, label=r'$\psi(x)$')
ax1.plot(x, p_psi.real, 'r-', linewidth=2, label=r'Re($\hat{p}\psi$)')
ax1.plot(x, p_psi.imag, 'g--', linewidth=2, label=r'Im($\hat{p}\psi$)')
ax1.set_xlabel('x')
ax1.set_title('Action of Momentum Operator on Gaussian')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Now for a plane wave (momentum eigenstate)
p0 = 2.0  # momentum value
plane_wave = np.exp(1j * p0 * x / hbar) / np.sqrt(2 * np.pi * hbar)
p_plane_wave = -1j * hbar * np.gradient(plane_wave, dx)

ax2.plot(x, plane_wave.real, 'b-', linewidth=1, alpha=0.7, label=r'Re(plane wave)')
ax2.plot(x, (p0 * plane_wave).real, 'r--', linewidth=2, label=r'Re($p_0 \cdot$ plane wave)')
ax2.plot(x, p_plane_wave.real, 'g:', linewidth=3, label=r'Re($\hat{p}$ plane wave)')
ax2.set_xlabel('x')
ax2.set_title(f'Momentum Eigenstate ($p_0$ = {p0})')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-3, 3)

plt.tight_layout()
plt.savefig('momentum_operator.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 5: Verifying Commutation Relation
# ============================================

print("\n" + "=" * 50)
print("Part 5: Verifying [x, p] = i*hbar")
print("=" * 50)

def apply_xp(psi, x, dx):
    """Apply x then p"""
    dpsi = np.gradient(x * psi, dx)
    return -1j * hbar * dpsi

def apply_px(psi, x, dx):
    """Apply p then x"""
    dpsi = np.gradient(psi, dx)
    return x * (-1j * hbar * dpsi)

def apply_commutator_xp(psi, x, dx):
    """Apply [x, p] = xp - px"""
    return apply_xp(psi, x, dx) - apply_px(psi, x, dx)

# Test on Gaussian
comm_psi = apply_commutator_xp(psi, x, dx)
expected = 1j * hbar * psi

# Compare
error = np.max(np.abs(comm_psi - expected))
print(f"Max error in [x,p]|psi> - i*hbar|psi>: {error:.2e}")
print("Commutation relation verified!" if error < 0.01 else "Error too large")

# Visualize
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, (1j * hbar * psi).real, 'b-', linewidth=2, label=r'$i\hbar\psi$ (expected)')
ax.plot(x, comm_psi.real, 'r--', linewidth=2, label=r'$[\hat{x},\hat{p}]\psi$ (computed)')
ax.set_xlabel('x')
ax.set_title('Verification of $[\hat{x}, \hat{p}] = i\hbar$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('commutator_verification.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 6: Wave Packet with Nonzero Momentum
# ============================================

print("\n" + "=" * 50)
print("Part 6: Wave Packet with Initial Momentum")
print("=" * 50)

# Gaussian wave packet with momentum p0
p0 = 3.0
def moving_gaussian(x, a, p0):
    """Gaussian wave packet with average momentum p0"""
    norm = (1 / (np.pi * a**2))**0.25
    return norm * np.exp(-x**2 / (2 * a**2)) * np.exp(1j * p0 * x / hbar)

psi_moving = moving_gaussian(x, a, p0)

# Calculate <p>
dpsi_moving = np.gradient(psi_moving, dx)
p_mean_moving = simps(np.conj(psi_moving) * (-1j * hbar * dpsi_moving), x).real
print(f"<p> for moving wave packet = {p_mean_moving:.4f} (expected: {p0})")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, psi_moving.real, 'b-', linewidth=2, label=r'Re($\psi$)')
ax1.plot(x, psi_moving.imag, 'r--', linewidth=2, label=r'Im($\psi$)')
ax1.plot(x, np.abs(psi_moving), 'k-', linewidth=1, label=r'$|\psi|$')
ax1.set_xlabel('x')
ax1.set_title(f'Wave Packet with $p_0$ = {p0}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Probability density (same as stationary)
ax2.fill_between(x, 0, np.abs(psi_moving)**2, alpha=0.5, color='blue')
ax2.set_xlabel('x')
ax2.set_ylabel(r'$|\psi(x)|^2$')
ax2.set_title('Probability Density (unchanged by momentum)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('moving_wavepacket.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 7: Multiple Momentum Eigenstates
# ============================================

print("\n" + "=" * 50)
print("Part 7: Momentum Eigenstates Visualization")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

p_values = [0, 1, 2, 5]
for ax, p_val in zip(axes.flat, p_values):
    psi_p = np.exp(1j * p_val * x / hbar)

    ax.plot(x, psi_p.real, 'b-', linewidth=2, label=r'Re($\psi_p$)')
    ax.plot(x, psi_p.imag, 'r--', linewidth=2, label=r'Im($\psi_p$)')
    ax.set_xlabel('x')
    ax.set_title(f'Momentum Eigenstate $p = {p_val}$, $\\lambda = {2*np.pi*hbar/p_val if p_val != 0 else "\\infty"}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)

plt.tight_layout()
plt.savefig('momentum_eigenstates.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 8: Probability Current
# ============================================

print("\n" + "=" * 50)
print("Part 8: Probability Current")
print("=" * 50)

def probability_current(psi, x, dx, m=1):
    """
    Calculate probability current density:
    j = (hbar/2mi) * (psi* dpsi/dx - psi dpsi*/dx)
    """
    dpsi = np.gradient(psi, dx)
    dpsi_conj = np.gradient(np.conj(psi), dx)
    j = (hbar / (2 * m * 1j)) * (np.conj(psi) * dpsi - psi * dpsi_conj)
    return j.real

# For stationary Gaussian (p0 = 0)
j_stationary = probability_current(psi, x, dx)
print(f"Max current for stationary Gaussian: {np.max(np.abs(j_stationary)):.2e}")

# For moving Gaussian (p0 != 0)
j_moving = probability_current(psi_moving, x, dx)
print(f"Max current for moving Gaussian: {np.max(np.abs(j_moving)):.4f}")

# Classical current would be p0/m * |psi|^2
j_classical = (p0 / 1.0) * np.abs(psi_moving)**2
print(f"Expected (p0/m * |psi|^2) max: {np.max(j_classical):.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, np.abs(psi_moving)**2, 'b-', linewidth=2, label=r'$|\psi|^2$')
ax.plot(x, j_moving, 'r-', linewidth=2, label='Probability current $j$')
ax.set_xlabel('x')
ax.set_title('Probability Density and Current')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('probability_current.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("Lab Complete!")
print("=" * 50)
```

---

## Summary

### Key Formulas

| Concept | Position Representation | Momentum Representation |
|---------|------------------------|------------------------|
| Wave function | $\psi(x) = \langle x|\psi\rangle$ | $\phi(p) = \langle p|\psi\rangle$ |
| Position operator | $\hat{x} \to x$ | $\hat{x} \to i\hbar\frac{d}{dp}$ |
| Momentum operator | $\hat{p} \to -i\hbar\frac{d}{dx}$ | $\hat{p} \to p$ |
| Eigenfunctions | $\langle x|x'\rangle = \delta(x-x')$ | $\langle x|p\rangle = \frac{e^{ipx/\hbar}}{\sqrt{2\pi\hbar}}$ |

### Key Equations

| Formula | Expression |
|---------|------------|
| Canonical commutation | $[\hat{x}, \hat{p}] = i\hbar$ |
| Position eigenvalue | $\hat{x}|x'\rangle = x'|x'\rangle$ |
| Momentum eigenvalue | $\hat{p}|p'\rangle = p'|p'\rangle$ |
| Probability density | $P(x) = |\psi(x)|^2$ |
| Expectation value | $\langle\hat{A}\rangle = \int \psi^*\hat{A}\psi \, dx$ |

### Key Takeaways

1. **Position operator** multiplies by $x$ in position representation.

2. **Momentum operator** is $-i\hbar\frac{d}{dx}$, generating translations.

3. **Momentum eigenstates** are plane waves $e^{ipx/\hbar}$ (de Broglie waves).

4. **The canonical commutation relation** $[\hat{x}, \hat{p}] = i\hbar$ is fundamental.

5. **Position and momentum are incompatible** - the Heisenberg uncertainty principle follows.

---

## Daily Checklist

- [ ] Read Shankar 4.3 and Sakurai 1.6 on position and momentum
- [ ] Derive momentum operator from translation generator
- [ ] Verify canonical commutation relation $[\hat{x}, \hat{p}] = i\hbar$
- [ ] Work through all three examples
- [ ] Complete Level 1 and 2 practice problems
- [ ] Attempt at least one Level 3 problem
- [ ] Run computational lab and visualize results
- [ ] Calculate $\langle x\rangle$, $\langle x^2\rangle$, $\langle p\rangle$, $\langle p^2\rangle$ for Gaussian
- [ ] Write derivation of momentum eigenfunction in study journal

---

## Preview: Tomorrow's Topics

**Day 349: Fourier Transform Connection**

Tomorrow we discover the beautiful relationship:

- Position and momentum wave functions are **Fourier transforms** of each other
- $\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x)dx$
- Parseval's theorem and probability conservation
- Physical meaning of momentum space wave function
- Uncertainty principle from Fourier analysis

**Preparation:** Review Fourier transforms from your mathematical foundations course.

---

**References:**
- Shankar, R. (1994). Principles of Quantum Mechanics, Chapter 4
- Sakurai, J.J. (2017). Modern Quantum Mechanics, Chapter 1.6
- Griffiths, D.J. (2018). Introduction to Quantum Mechanics, Chapter 3.3
- Cohen-Tannoudji, C. (1977). Quantum Mechanics, Chapter II
