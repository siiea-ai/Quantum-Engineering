# Day 382: QHO Wave Functions — Hermite Polynomials

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Wave Function Derivation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 382, you will be able to:

1. Derive the ground state wave function from $\hat{a}|0\rangle = 0$
2. Understand the Hermite polynomial solutions and their properties
3. Write explicit wave functions $\psi_n(x)$ for arbitrary $n$
4. Prove orthonormality using wave function integrals
5. Visualize probability densities and classical turning points
6. Connect wave function nodes to quantum numbers

---

## Core Content

### 1. The Ground State Wave Function

We know $\hat{a}|0\rangle = 0$. Let's find $\psi_0(x) = \langle x|0\rangle$.

#### Setting Up the Equation

In position representation:
$$\hat{x} \to x, \quad \hat{p} \to -i\hbar\frac{d}{dx}$$

The annihilation operator becomes:
$$\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\left(x + \frac{i(-i\hbar\frac{d}{dx})}{m\omega}\right) = \sqrt{\frac{m\omega}{2\hbar}}\left(x + \frac{\hbar}{m\omega}\frac{d}{dx}\right)$$

#### The Ground State Equation

Applying $\hat{a}|0\rangle = 0$ in position representation:
$$\sqrt{\frac{m\omega}{2\hbar}}\left(x + \frac{\hbar}{m\omega}\frac{d}{dx}\right)\psi_0(x) = 0$$

This simplifies to:
$$\frac{d\psi_0}{dx} = -\frac{m\omega}{\hbar}x\psi_0$$

#### Solving the First-Order ODE

This is separable:
$$\frac{d\psi_0}{\psi_0} = -\frac{m\omega}{\hbar}x\,dx$$

Integrating:
$$\ln\psi_0 = -\frac{m\omega}{2\hbar}x^2 + C$$

$$\psi_0(x) = A\exp\left(-\frac{m\omega x^2}{2\hbar}\right)$$

#### Normalization

Requiring $\int_{-\infty}^{\infty}|\psi_0|^2 dx = 1$:

$$|A|^2 \int_{-\infty}^{\infty}e^{-m\omega x^2/\hbar}dx = 1$$

Using the Gaussian integral $\int_{-\infty}^{\infty}e^{-ax^2}dx = \sqrt{\pi/a}$:

$$|A|^2 \sqrt{\frac{\pi\hbar}{m\omega}} = 1 \implies A = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}$$

$$\boxed{\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} e^{-m\omega x^2/2\hbar}}$$

---

### 2. Dimensionless Form

Define the dimensionless position:
$$\xi = x\sqrt{\frac{m\omega}{\hbar}} = \frac{x}{x_0}$$

Then:
$$\boxed{\psi_0(\xi) = \frac{1}{\pi^{1/4}}e^{-\xi^2/2}}$$

This is a **Gaussian** centered at the origin with width $\sigma = 1/\sqrt{2}$ in dimensionless units.

---

### 3. Excited State Wave Functions

#### Method 1: Apply Creation Operators

Since $|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$, in position representation:

$$\psi_n(x) = \frac{1}{\sqrt{n!}}\left(\sqrt{\frac{m\omega}{2\hbar}}\left(x - \frac{\hbar}{m\omega}\frac{d}{dx}\right)\right)^n \psi_0(x)$$

In dimensionless variables:
$$\hat{a}^\dagger = \frac{1}{\sqrt{2}}\left(\xi - \frac{d}{d\xi}\right)$$

So:
$$\psi_n(\xi) = \frac{1}{\sqrt{n!}}\left(\frac{1}{\sqrt{2}}\left(\xi - \frac{d}{d\xi}\right)\right)^n \frac{e^{-\xi^2/2}}{\pi^{1/4}}$$

#### Method 2: Hermite Polynomials

The result is elegantly expressed using **Hermite polynomials** $H_n(\xi)$:

$$\boxed{\psi_n(\xi) = \frac{1}{\sqrt{2^n n!}}\frac{1}{\pi^{1/4}}H_n(\xi)e^{-\xi^2/2}}$$

Or in dimensional form:
$$\boxed{\psi_n(x) = \frac{1}{\sqrt{2^n n!}}\left(\frac{m\omega}{\pi\hbar}\right)^{1/4}H_n\left(x\sqrt{\frac{m\omega}{\hbar}}\right)\exp\left(-\frac{m\omega x^2}{2\hbar}\right)}$$

---

### 4. Hermite Polynomials

#### Definition via Rodrigues Formula

$$\boxed{H_n(\xi) = (-1)^n e^{\xi^2}\frac{d^n}{d\xi^n}e^{-\xi^2}}$$

#### First Few Hermite Polynomials

| n | $H_n(\xi)$ | $\psi_n(\xi)$ (unnormalized) |
|---|------------|------------------------------|
| 0 | 1 | $e^{-\xi^2/2}$ |
| 1 | $2\xi$ | $2\xi e^{-\xi^2/2}$ |
| 2 | $4\xi^2 - 2$ | $(4\xi^2 - 2)e^{-\xi^2/2}$ |
| 3 | $8\xi^3 - 12\xi$ | $(8\xi^3 - 12\xi)e^{-\xi^2/2}$ |
| 4 | $16\xi^4 - 48\xi^2 + 12$ | $(16\xi^4 - 48\xi^2 + 12)e^{-\xi^2/2}$ |

#### Recurrence Relations

**Three-term recurrence:**
$$\boxed{H_{n+1}(\xi) = 2\xi H_n(\xi) - 2n H_{n-1}(\xi)}$$

**Derivative:**
$$\boxed{\frac{d}{d\xi}H_n(\xi) = 2n H_{n-1}(\xi)}$$

#### Generating Function

$$e^{2\xi t - t^2} = \sum_{n=0}^{\infty}\frac{t^n}{n!}H_n(\xi)$$

---

### 5. Orthonormality

The wave functions satisfy:
$$\int_{-\infty}^{\infty}\psi_m^*(x)\psi_n(x)dx = \delta_{mn}$$

This follows from:
1. Hermiticity of $\hat{H}$ (different eigenvalues → orthogonal)
2. Explicit integral using Hermite polynomial orthogonality:

$$\int_{-\infty}^{\infty}H_m(\xi)H_n(\xi)e^{-\xi^2}d\xi = \sqrt{\pi}2^n n!\,\delta_{mn}$$

---

### 6. Completeness

The set $\{\psi_n(x)\}_{n=0}^{\infty}$ forms a complete orthonormal basis for $L^2(\mathbb{R})$:

$$\sum_{n=0}^{\infty}\psi_n^*(x')\psi_n(x) = \delta(x - x')$$

Any square-integrable function can be expanded:
$$f(x) = \sum_{n=0}^{\infty}c_n\psi_n(x), \quad c_n = \int_{-\infty}^{\infty}\psi_n^*(x)f(x)dx$$

---

### 7. Properties of Wave Functions

#### Parity

$$\psi_n(-x) = (-1)^n\psi_n(x)$$

- Even $n$: even functions (symmetric about origin)
- Odd $n$: odd functions (antisymmetric)

This follows from $H_n(-\xi) = (-1)^n H_n(\xi)$.

#### Number of Nodes

$\psi_n(x)$ has exactly **$n$ nodes** (zeros) in the interior of the classically allowed region.

| n | Nodes | Parity |
|---|-------|--------|
| 0 | 0 | Even |
| 1 | 1 | Odd |
| 2 | 2 | Even |
| 3 | 3 | Odd |
| n | n | $(-1)^n$ |

**Physical interpretation:** More nodes = higher kinetic energy = higher total energy.

#### Classical Turning Points

At energy $E_n$, the classical turning points are where $E_n = V(x)$:
$$\hbar\omega\left(n + \frac{1}{2}\right) = \frac{1}{2}m\omega^2 x_{turn}^2$$

$$x_{turn} = \pm\sqrt{\frac{2E_n}{m\omega^2}} = \pm\sqrt{\frac{(2n+1)\hbar}{m\omega}}$$

In dimensionless units:
$$\boxed{\xi_{turn} = \pm\sqrt{2n+1}}$$

---

### 8. Probability Density

The probability density is:
$$\boxed{|\psi_n(x)|^2 = \frac{1}{2^n n!\sqrt{\pi}}\frac{1}{x_0}H_n^2(\xi)e^{-\xi^2}}$$

#### Classical Limit (Large n)

For large $n$:
- Wave function oscillates rapidly
- When averaged, probability approaches classical result
- Classical probability: $P_{classical}(x) \propto 1/v(x) \propto 1/\sqrt{x_{turn}^2 - x^2}$

This is the **correspondence principle** at work!

---

### 9. Quantum Computing Connection: Bosonic Encodings

#### GKP (Gottesman-Kitaev-Preskill) Codes

The wave function approach is essential for **GKP qubits**:
- Logical $|0_L\rangle$: Comb of Gaussians at $x = 2\sqrt{\pi}n$ (even integers)
- Logical $|1_L\rangle$: Comb at $x = 2\sqrt{\pi}(n + 1/2)$ (odd half-integers)

These are special superpositions of position eigenstates, designed for error correction.

#### Wave Function Engineering

Modern experiments create arbitrary wave functions:
- **Superconducting circuits:** Microwave pulses shape states
- **Trapped ions:** Laser interactions engineer motional states
- **Optical cavities:** Nonlinear optics creates non-Gaussian states

Understanding $\psi_n(x)$ is the foundation for these technologies!

---

## Worked Examples

### Example 1: Deriving $\psi_1(x)$

**Problem:** Use $\hat{a}^\dagger$ to derive $\psi_1(x)$ from $\psi_0(x)$.

**Solution:**

$$\psi_1(x) = \frac{\hat{a}^\dagger\psi_0(x)}{\sqrt{1!}} = \hat{a}^\dagger\psi_0(x)$$

In position representation:
$$\hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\left(x - \frac{\hbar}{m\omega}\frac{d}{dx}\right)$$

Let $\alpha = m\omega/\hbar$. Then $\psi_0(x) = (\alpha/\pi)^{1/4}e^{-\alpha x^2/2}$.

$$\frac{d\psi_0}{dx} = -\alpha x \psi_0$$

So:
$$\hat{a}^\dagger\psi_0 = \sqrt{\frac{\alpha}{2}}\left(x - \frac{1}{\alpha}(-\alpha x)\right)\psi_0 = \sqrt{\frac{\alpha}{2}} \cdot 2x \cdot \psi_0$$

$$= \sqrt{2\alpha}\,x\,\psi_0 = \sqrt{2\alpha}\left(\frac{\alpha}{\pi}\right)^{1/4}x\,e^{-\alpha x^2/2}$$

In dimensionless form with $\xi = \sqrt{\alpha}x$:
$$\boxed{\psi_1(\xi) = \frac{\sqrt{2}}{\pi^{1/4}}\xi\,e^{-\xi^2/2} = \frac{1}{\sqrt{2\cdot 1!}\pi^{1/4}}H_1(\xi)e^{-\xi^2/2}}$$

since $H_1(\xi) = 2\xi$. $\blacksquare$

---

### Example 2: Verifying Orthogonality

**Problem:** Verify $\langle\psi_0|\psi_2\rangle = 0$ by direct integration.

**Solution:**

In dimensionless units:
$$\psi_0(\xi) = \frac{1}{\pi^{1/4}}e^{-\xi^2/2}$$
$$\psi_2(\xi) = \frac{1}{\sqrt{8}\pi^{1/4}}(4\xi^2 - 2)e^{-\xi^2/2}$$

The integral:
$$\langle\psi_0|\psi_2\rangle = \frac{1}{\sqrt{8}\sqrt{\pi}}\int_{-\infty}^{\infty}(4\xi^2 - 2)e^{-\xi^2}d\xi$$

Using Gaussian integrals:
$$\int_{-\infty}^{\infty}e^{-\xi^2}d\xi = \sqrt{\pi}$$
$$\int_{-\infty}^{\infty}\xi^2 e^{-\xi^2}d\xi = \frac{\sqrt{\pi}}{2}$$

Therefore:
$$\langle\psi_0|\psi_2\rangle = \frac{1}{\sqrt{8}\sqrt{\pi}}\left(4 \cdot \frac{\sqrt{\pi}}{2} - 2\sqrt{\pi}\right) = \frac{1}{\sqrt{8}\sqrt{\pi}}(2\sqrt{\pi} - 2\sqrt{\pi}) = 0$$

$$\boxed{\langle\psi_0|\psi_2\rangle = 0}$$ ✓ $\blacksquare$

---

### Example 3: Classical Turning Points

**Problem:** For the state $|5\rangle$, find:
(a) The classical turning points
(b) The probability of finding the particle in the classically forbidden region

**Solution:**

(a) Energy: $E_5 = \hbar\omega(5 + \frac{1}{2}) = \frac{11}{2}\hbar\omega$

Classical turning points (dimensionless):
$$\xi_{turn} = \pm\sqrt{2 \cdot 5 + 1} = \pm\sqrt{11} \approx \pm 3.32$$

In dimensional units with $x_0 = \sqrt{\hbar/m\omega}$:
$$x_{turn} = \pm\sqrt{11}x_0$$

(b) Probability in classically forbidden region:

$$P_{forbidden} = \int_{-\infty}^{-\xi_{turn}}|\psi_5|^2 d\xi + \int_{\xi_{turn}}^{\infty}|\psi_5|^2 d\xi$$

This requires numerical integration. For large $n$, the probability approaches:
$$P_{forbidden} \approx \frac{1}{\pi n}$$

For $n = 5$: $P_{forbidden} \approx 0.06$ (about 6%) $\blacksquare$

---

## Practice Problems

### Level 1: Direct Application

1. **Hermite Recursion:** Use the recurrence relation to find $H_3(\xi)$ from $H_1(\xi)$ and $H_2(\xi)$.

2. **Normalization Check:** Verify that $\int_{-\infty}^{\infty}|\psi_1(x)|^2 dx = 1$ using Gaussian integrals.

3. **Parity:** Show explicitly that $\psi_2(-x) = \psi_2(x)$ (even function).

### Level 2: Intermediate

4. **Wave Function by Hand:** Derive $\psi_2(x)$ by applying $\hat{a}^\dagger$ twice to $\psi_0(x)$.

5. **Node Locations:** Find the positions of the nodes for $\psi_3(x)$.
   *Hint:* Solve $H_3(\xi) = 8\xi^3 - 12\xi = 0$.

6. **Expectation Value:** Compute $\langle\psi_1|x^2|\psi_1\rangle$ using the explicit wave function.

### Level 3: Challenging

7. **Generating Function:** Use the generating function to derive the orthogonality relation:
   $$\int_{-\infty}^{\infty}H_m(\xi)H_n(\xi)e^{-\xi^2}d\xi = \sqrt{\pi}2^n n!\delta_{mn}$$

8. **Classical Limit:** For large $n$, show that the time-averaged probability density $|\psi_n(x)|^2$ approaches the classical result:
   $$P_{classical}(x) = \frac{1}{\pi\sqrt{x_{turn}^2 - x^2}}$$
   for $|x| < x_{turn}$.

9. **Momentum Space:** The momentum-space wave function is $\tilde{\psi}_n(p) = \langle p|n\rangle$. Show that $\tilde{\psi}_n(p)$ has the same functional form as $\psi_n(x)$ (up to scaling).

---

## Computational Lab

### Objective
Generate and visualize QHO wave functions, verify orthonormality, and explore the classical correspondence.

```python
"""
Day 382 Computational Lab: QHO Wave Functions and Hermite Polynomials
Quantum Harmonic Oscillator - Week 55
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial
from scipy.integrate import quad

# =============================================================================
# Part 1: Hermite Polynomials
# =============================================================================

print("=" * 70)
print("Part 1: Hermite Polynomials")
print("=" * 70)

def H_n(n, xi):
    """
    Compute Hermite polynomial H_n(xi) using scipy's hermite function.
    """
    H = hermite(n)  # Returns polynomial coefficients
    return H(xi)

# Display first few Hermite polynomials
xi_plot = np.linspace(-3, 3, 200)

fig, ax = plt.subplots(figsize=(10, 6))
for n in range(5):
    ax.plot(xi_plot, H_n(n, xi_plot), label=f'H_{n}(ξ)', linewidth=2)

ax.set_xlabel('ξ', fontsize=12)
ax.set_ylabel('H_n(ξ)', fontsize=12)
ax.set_title('Hermite Polynomials H_n(ξ)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 30)
ax.axhline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('day_382_hermite_polynomials.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFirst few Hermite polynomials:")
print("H_0(ξ) = 1")
print("H_1(ξ) = 2ξ")
print("H_2(ξ) = 4ξ² - 2")
print("H_3(ξ) = 8ξ³ - 12ξ")
print("H_4(ξ) = 16ξ⁴ - 48ξ² + 12")

# =============================================================================
# Part 2: QHO Wave Functions
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: QHO Wave Functions ψ_n(x)")
print("=" * 70)

def psi_n(n, xi):
    """
    Normalized QHO wave function in dimensionless units.

    ψ_n(ξ) = (1/√(2^n n! √π)) H_n(ξ) exp(-ξ²/2)
    """
    normalization = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return normalization * H_n(n, xi) * np.exp(-xi**2 / 2)

# Plot wave functions
xi = np.linspace(-6, 6, 500)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for n in range(6):
    ax = axes[n // 3, n % 3]

    # Wave function
    psi = psi_n(n, xi)
    ax.plot(xi, psi, 'b-', linewidth=2, label=f'ψ_{n}(ξ)')

    # Probability density
    prob = np.abs(psi)**2
    ax.fill_between(xi, 0, prob, alpha=0.3, color='blue', label='|ψ|²')

    # Classical turning points
    xi_turn = np.sqrt(2*n + 1)
    ax.axvline(xi_turn, color='red', linestyle='--', alpha=0.7)
    ax.axvline(-xi_turn, color='red', linestyle='--', alpha=0.7, label='Classical turning')

    # Energy level (scaled for display)
    E_n = n + 0.5
    V = 0.5 * xi**2
    V_display = V / 10  # Scale for visibility
    ax.plot(xi, V_display - 0.5, 'k-', alpha=0.3, linewidth=1)

    ax.set_xlabel('ξ', fontsize=10)
    ax.set_ylabel('ψ, |ψ|²', fontsize=10)
    ax.set_title(f'n = {n}, E_{n} = {n + 0.5}ℏω', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)

plt.tight_layout()
plt.savefig('day_382_wave_functions.png', dpi=150, bbox_inches='tight')
plt.show()

print("Wave function plots saved.")

# =============================================================================
# Part 3: Energy Level Diagram with Wave Functions
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Energy Level Diagram with Wave Functions")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 10))

# Potential
xi = np.linspace(-6, 6, 200)
V = 0.5 * xi**2
ax.plot(xi, V, 'k-', linewidth=2, label='V(x) = ½mω²x²')

# Plot wave functions at their energy levels
colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
scale = 2.5  # Scale factor for wave function display

for n in range(6):
    E_n = n + 0.5
    psi = psi_n(n, xi)

    # Offset and scale wave function for display
    psi_display = E_n + scale * psi

    ax.plot(xi, psi_display, color=colors[n], linewidth=2, label=f'n={n}: E={E_n}ℏω')
    ax.fill_between(xi, E_n, psi_display, alpha=0.3, color=colors[n])

    # Energy level line
    xi_turn = np.sqrt(2*n + 1)
    ax.hlines(E_n, -xi_turn, xi_turn, colors='gray', linewidth=1, linestyles='dashed')

ax.set_xlabel('Dimensionless position ξ = x/x₀', fontsize=12)
ax.set_ylabel('Energy / ℏω', fontsize=12)
ax.set_title('QHO: Wave Functions at Energy Levels', fontsize=14)
ax.legend(loc='upper right')
ax.set_xlim(-6, 6)
ax.set_ylim(0, 8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_382_energy_wavefunctions.png', dpi=150, bbox_inches='tight')
plt.show()

print("Energy level diagram saved.")

# =============================================================================
# Part 4: Verify Orthonormality
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Numerical Verification of Orthonormality")
print("=" * 70)

def inner_product(m, n, xi_max=10, num_points=1000):
    """
    Compute ⟨ψ_m|ψ_n⟩ = ∫ ψ_m*(x) ψ_n(x) dx
    """
    result, _ = quad(lambda xi: psi_n(m, xi) * psi_n(n, xi), -xi_max, xi_max)
    return result

print("\nInner product matrix ⟨m|n⟩:")
N_max = 6
overlap_matrix = np.zeros((N_max, N_max))

for m in range(N_max):
    for n in range(N_max):
        overlap_matrix[m, n] = inner_product(m, n)

print(np.round(overlap_matrix, 4))
print(f"\nIs orthonormal? Max deviation from identity: {np.max(np.abs(overlap_matrix - np.eye(N_max))):.6f}")

# =============================================================================
# Part 5: Node Counting
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Node Locations")
print("=" * 70)

def find_nodes(n, num_points=10000):
    """Find zeros of ψ_n(x) by looking for sign changes"""
    xi = np.linspace(-5, 5, num_points)
    psi = psi_n(n, xi)

    nodes = []
    for i in range(len(psi) - 1):
        if psi[i] * psi[i+1] < 0:  # Sign change
            # Linear interpolation for better accuracy
            xi_node = xi[i] - psi[i] * (xi[i+1] - xi[i]) / (psi[i+1] - psi[i])
            nodes.append(xi_node)

    return np.array(nodes)

print("\nNodes (zeros) of wave functions:")
for n in range(6):
    nodes = find_nodes(n)
    print(f"ψ_{n}: {len(nodes)} nodes at ξ = {np.round(nodes, 3)}")

# =============================================================================
# Part 6: Classical Correspondence
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Classical Correspondence for Large n")
print("=" * 70)

def classical_probability(xi, xi_turn):
    """
    Classical probability density: P(x) ∝ 1/v(x) ∝ 1/√(x_turn² - x²)
    """
    if np.isscalar(xi):
        if abs(xi) >= xi_turn:
            return 0
        return 1.0 / (np.pi * np.sqrt(xi_turn**2 - xi**2))
    else:
        result = np.zeros_like(xi)
        mask = np.abs(xi) < xi_turn
        result[mask] = 1.0 / (np.pi * np.sqrt(xi_turn**2 - xi[mask]**2))
        return result

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, n in enumerate([5, 20, 50]):
    ax = axes[idx]
    xi = np.linspace(-np.sqrt(2*n+1) - 1, np.sqrt(2*n+1) + 1, 1000)
    xi_turn = np.sqrt(2*n + 1)

    # Quantum probability (smoothed for large n)
    prob_quantum = np.abs(psi_n(n, xi))**2

    # Classical probability
    prob_classical = classical_probability(xi, xi_turn)

    ax.plot(xi, prob_quantum, 'b-', linewidth=1, alpha=0.7, label='|ψ_n|² (quantum)')
    ax.plot(xi, prob_classical, 'r--', linewidth=2, label='Classical')
    ax.axvline(xi_turn, color='green', linestyle=':', label='Turning point')
    ax.axvline(-xi_turn, color='green', linestyle=':')

    ax.set_xlabel('ξ', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title(f'n = {n}: Classical Correspondence', fontsize=12)
    ax.legend()
    ax.set_xlim(-xi_turn - 1, xi_turn + 1)
    ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('day_382_classical_limit.png', dpi=150, bbox_inches='tight')
plt.show()

print("Classical correspondence plots saved.")

# =============================================================================
# Part 7: Probability in Classically Forbidden Region
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Tunneling Probability (Classically Forbidden Region)")
print("=" * 70)

def forbidden_probability(n):
    """
    Probability of finding particle beyond classical turning points.
    """
    xi_turn = np.sqrt(2*n + 1)

    # Integrate from turning point to infinity (both sides)
    prob_right, _ = quad(lambda xi: np.abs(psi_n(n, xi))**2, xi_turn, 10)
    prob_left, _ = quad(lambda xi: np.abs(psi_n(n, xi))**2, -10, -xi_turn)

    return prob_left + prob_right

print("\nProbability in classically forbidden region:")
print("-" * 40)
n_values = [0, 1, 2, 5, 10, 20]

for n in n_values:
    p_forbidden = forbidden_probability(n)
    approx = 1 / (np.pi * n) if n > 0 else 0.157  # Approximation for large n
    print(f"n = {n:2d}: P_forbidden = {p_forbidden:.4f} (approx: {approx:.4f})")

# =============================================================================
# Part 8: Momentum Space Wave Functions
# =============================================================================

print("\n" + "=" * 70)
print("Part 8: Momentum Space Wave Functions")
print("=" * 70)

def psi_n_momentum(n, eta):
    """
    Momentum space wave function.
    For QHO: φ_n(p) has the same form as ψ_n(x) up to phase.

    η = p/p_0 where p_0 = √(mωℏ)
    """
    # Same functional form!
    normalization = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return (-1j)**n * normalization * H_n(n, eta) * np.exp(-eta**2 / 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Position space
ax = axes[0]
xi = np.linspace(-5, 5, 200)
for n in range(4):
    psi = psi_n(n, xi)
    ax.plot(xi, psi, label=f'ψ_{n}(ξ)', linewidth=2)

ax.set_xlabel('Dimensionless position ξ', fontsize=12)
ax.set_ylabel('ψ_n(ξ)', fontsize=12)
ax.set_title('Position Space Wave Functions', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Momentum space
ax = axes[1]
eta = np.linspace(-5, 5, 200)
for n in range(4):
    phi = psi_n_momentum(n, eta)
    ax.plot(eta, np.real(phi), label=f'Re[φ_{n}(η)]', linewidth=2)

ax.set_xlabel('Dimensionless momentum η', fontsize=12)
ax.set_ylabel('φ_n(η)', fontsize=12)
ax.set_title('Momentum Space Wave Functions', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_382_momentum_space.png', dpi=150, bbox_inches='tight')
plt.show()

print("Momentum space comparison saved.")

# =============================================================================
# Part 9: 3D Visualization of Probability Densities
# =============================================================================

print("\n" + "=" * 70)
print("Part 9: Time Evolution Preview (Superposition)")
print("=" * 70)

# Superposition of ground and first excited states
# |ψ(t)⟩ = (|0⟩ + |1⟩)/√2 → ψ(x,t) = (ψ_0 e^{-iE_0t/ℏ} + ψ_1 e^{-iE_1t/ℏ})/√2

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

xi = np.linspace(-5, 5, 200)
times = np.linspace(0, 2*np.pi, 8)  # One period: T = 2π/ω

for idx, t in enumerate(times):
    ax = axes[idx // 4, idx % 4]

    # Time-dependent wave function (ℏω = 1)
    psi_0_t = psi_n(0, xi) * np.exp(-1j * 0.5 * t)
    psi_1_t = psi_n(1, xi) * np.exp(-1j * 1.5 * t)
    psi_super = (psi_0_t + psi_1_t) / np.sqrt(2)

    prob = np.abs(psi_super)**2

    ax.fill_between(xi, 0, prob, alpha=0.7, color='purple')
    ax.plot(xi, np.real(psi_super), 'b-', linewidth=1, alpha=0.5, label='Re(ψ)')
    ax.set_xlabel('ξ')
    ax.set_title(f't = {t/(2*np.pi):.2f} T')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.8, 0.8)

fig.suptitle('Time Evolution: |ψ⟩ = (|0⟩ + |1⟩)/√2 oscillates like a classical packet', fontsize=14)
plt.tight_layout()
plt.savefig('day_382_time_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Time evolution visualization saved.")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Ground state | $\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}e^{-m\omega x^2/2\hbar}$ |
| Excited states | $\psi_n(\xi) = \frac{1}{\sqrt{2^n n!\sqrt{\pi}}}H_n(\xi)e^{-\xi^2/2}$ |
| Hermite recurrence | $H_{n+1} = 2\xi H_n - 2nH_{n-1}$ |
| Orthonormality | $\int \psi_m^*\psi_n dx = \delta_{mn}$ |
| Classical turning point | $\xi_{turn} = \pm\sqrt{2n+1}$ |
| Parity | $\psi_n(-x) = (-1)^n\psi_n(x)$ |
| Number of nodes | n interior nodes |

### Main Takeaways

1. **Ground state from algebra:** Solving $\hat{a}\psi_0 = 0$ gives a Gaussian
2. **Hermite polynomials emerge:** They encode the excited state structure
3. **Nodes count energy:** State $|n\rangle$ has exactly $n$ nodes
4. **Parity alternates:** Even n = even function, odd n = odd function
5. **Classical limit:** For large $n$, quantum probability matches classical average
6. **Position-momentum symmetry:** Same functional form in both representations

---

## Daily Checklist

- [ ] Read Shankar Section 7.4 on wave functions
- [ ] Derive $\psi_0(x)$ from $\hat{a}|0\rangle = 0$ independently
- [ ] Verify $\psi_1(x)$ and $\psi_2(x)$ using the recurrence relation
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt the classical limit problem (Level 3, #8)
- [ ] Run and understand the computational lab
- [ ] Sketch the first four wave functions by hand (without looking!)

---

## Preview: Day 383

Tomorrow we introduce **coherent states** — eigenstates of the annihilation operator. These are the "most classical" quantum states, exhibiting minimum uncertainty and following classical trajectories.

---

*"The wave functions of the harmonic oscillator, built from Hermite polynomials, are among the most important in all of physics."*
— J.J. Sakurai

---

**Next:** [Day_383_Friday.md](Day_383_Friday.md) — Coherent States
