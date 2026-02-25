# Day 429: Radial Solution for Hydrogen

## Overview
**Day 429** | Year 1, Month 16, Week 62 | Solving the Radial Equation

Today we solve the hydrogen radial equation, introducing Laguerre polynomials and deriving the famous quantization condition.

---

## Learning Objectives

By the end of today, you will be able to:
1. Transform the radial equation to standard form
2. Identify the asymptotic behavior at large r
3. Derive the power series solution
4. Understand how quantization emerges from normalizability
5. Connect to associated Laguerre polynomials
6. Write the general radial wavefunction

---

## Core Content

### The Radial Equation

In atomic units (ℏ = m_e = e = 1):
$$\frac{d^2u}{dr^2} + \left[\frac{2}{r} + 2E - \frac{l(l+1)}{r^2}\right]u = 0$$

where u = rR and E < 0 for bound states.

### Asymptotic Analysis

**Large r:** The 1/r and 1/r² terms become negligible:
$$\frac{d^2u}{dr^2} + 2Eu \approx 0$$

For E < 0, define κ² = -2E (so κ > 0):
$$u \sim e^{-\kappa r} \quad \text{(normalizable)}$$

**Small r:**
$$u \sim r^{l+1}$$

### Change of Variables

Define dimensionless variable:
$$\rho = 2\kappa r = 2r\sqrt{-2E}$$

And write:
$$u(\rho) = \rho^{l+1} e^{-\rho/2} w(\rho)$$

### The Equation for w(ρ)

$$\rho \frac{d^2w}{d\rho^2} + (2l + 2 - \rho)\frac{dw}{d\rho} + \left[\frac{1}{\kappa} - l - 1\right]w = 0$$

This is the **associated Laguerre equation** if:
$$\frac{1}{\kappa} - l - 1 = n_r \quad \text{(non-negative integer)}$$

### The Quantization Condition

Define the principal quantum number:
$$n = n_r + l + 1$$

where n_r = 0, 1, 2, ... is the radial quantum number.

Then:
$$\kappa = \frac{1}{n}$$

### Energy Eigenvalues

From κ² = -2E and κ = 1/n:
$$\boxed{E_n = -\frac{1}{2n^2}} \quad \text{(atomic units)}$$

In SI units:
$$\boxed{E_n = -\frac{13.6 \text{ eV}}{n^2}}$$

### Associated Laguerre Polynomials

$$L_p^q(x) = \frac{d^q}{dx^q}L_p(x)$$

where L_p(x) are Laguerre polynomials:
$$L_p(x) = e^x \frac{d^p}{dx^p}(x^p e^{-x})$$

First few:
- L₀^q(x) = 1
- L₁^q(x) = q + 1 - x
- L₂^q(x) = ½[(q+1)(q+2) - 2(q+2)x + x²]

### The Radial Wavefunction

$$\boxed{R_{nl}(r) = N_{nl} \left(\frac{2r}{na_0}\right)^l e^{-r/(na_0)} L_{n-l-1}^{2l+1}\left(\frac{2r}{na_0}\right)}$$

### Normalization

$$N_{nl} = -\sqrt{\left(\frac{2}{na_0}\right)^3 \frac{(n-l-1)!}{2n[(n+l)!]^3}}$$

---

## Quantum Computing Connection

### Quantum Phase Estimation

The hydrogen energy spectrum:
- Demonstrates eigenvalue structure
- Test case for QPE algorithms
- Energy differences give transition frequencies

### Variational Methods

Trial wavefunctions use:
- Exponential decay (correct asymptotic)
- Polynomial corrections (radial nodes)

---

## Worked Examples

### Example 1: Ground State (n=1, l=0)

**Problem:** Derive R₁₀(r).

**Solution:**
n_r = n - l - 1 = 1 - 0 - 1 = 0

L₀^1(x) = 1

$$R_{10}(r) = N_{10} e^{-r/a_0}$$

Normalizing: ∫₀^∞ |R|² r² dr = 1
$$N_{10} = 2/a_0^{3/2}$$

$$\boxed{R_{10}(r) = \frac{2}{a_0^{3/2}} e^{-r/a_0}}$$

### Example 2: 2s State (n=2, l=0)

**Problem:** Find R₂₀(r).

**Solution:**
n_r = 2 - 0 - 1 = 1

L₁^1(x) = 2 - x

$$R_{20}(r) = N_{20}\left(1 - \frac{r}{2a_0}\right)e^{-r/(2a_0)}$$

After normalization:
$$\boxed{R_{20}(r) = \frac{1}{2\sqrt{2}a_0^{3/2}}\left(2 - \frac{r}{a_0}\right)e^{-r/(2a_0)}}$$

Note the node at r = 2a₀.

### Example 3: 2p State (n=2, l=1)

**Solution:**
n_r = 2 - 1 - 1 = 0

L₀^3(x) = 1

$$\boxed{R_{21}(r) = \frac{1}{2\sqrt{6}a_0^{3/2}}\frac{r}{a_0}e^{-r/(2a_0)}}$$

---

## Practice Problems

### Direct Application
1. How many radial nodes does R₃₁ have?
2. What is the asymptotic form of R_{nl}(r) as r → ∞?
3. Write L₂^1(x) explicitly.

### Intermediate
4. Verify that R₁₀(r) is normalized.
5. Find the position of the node in R₂₀(r).
6. Show that R_{nl}(0) = 0 for l > 0.

### Challenging
7. Derive the Laguerre differential equation from the radial Schrödinger equation.
8. Prove the orthogonality ∫₀^∞ R_{nl}R_{n'l}r²dr = δ_{nn'}.

---

## Computational Lab

```python
"""
Day 429: Hydrogen Radial Wavefunctions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, factorial

def hydrogen_radial(n, l, r, a0=1):
    """
    Normalized hydrogen radial wavefunction R_nl(r)
    """
    rho = 2 * r / (n * a0)

    # Normalization
    norm = np.sqrt((2/(n*a0))**3 * factorial(n-l-1) / (2*n*factorial(n+l)**3))

    # Associated Laguerre polynomial
    L = genlaguerre(n-l-1, 2*l+1)(rho)

    R = norm * rho**l * np.exp(-rho/2) * L

    return R

# Plot radial wavefunctions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

r = np.linspace(0, 25, 500)

# Panel 1: n=1,2,3 s-orbitals
ax = axes[0, 0]
for n in [1, 2, 3]:
    R = hydrogen_radial(n, 0, r)
    ax.plot(r, R, linewidth=2, label=f'{n}s')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('R_{n0}(r)', fontsize=12)
ax.set_title('s-orbital Wavefunctions', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Radial probability for n=1,2,3
ax = axes[0, 1]
for n in [1, 2, 3]:
    R = hydrogen_radial(n, 0, r)
    P = np.abs(R)**2 * r**2
    ax.plot(r, P, linewidth=2, label=f'{n}s')
ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('P(r) = |R|²r²', fontsize=12)
ax.set_title('Radial Probability Density', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: n=2 shell
ax = axes[1, 0]
R_2s = hydrogen_radial(2, 0, r)
R_2p = hydrogen_radial(2, 1, r)
ax.plot(r, R_2s, 'b-', linewidth=2, label='2s')
ax.plot(r, R_2p, 'r-', linewidth=2, label='2p')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('R_{2l}(r)', fontsize=12)
ax.set_title('n=2 Shell (2s and 2p)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: n=3 shell
ax = axes[1, 1]
for l, name in [(0, '3s'), (1, '3p'), (2, '3d')]:
    R = hydrogen_radial(3, l, r)
    ax.plot(r, R, linewidth=2, label=name)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('R_{3l}(r)', fontsize=12)
ax.set_title('n=3 Shell', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day429_radial_wavefunctions.png', dpi=150)
plt.show()

# Verify normalization
print("=== Normalization Verification ===")
r_fine = np.linspace(0.001, 100, 10000)
dr = r_fine[1] - r_fine[0]

for n in range(1, 5):
    for l in range(n):
        R = hydrogen_radial(n, l, r_fine)
        norm = np.sum(np.abs(R)**2 * r_fine**2) * dr
        print(f"R_{n}{l}: ∫|R|²r²dr = {norm:.6f}")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Energy | E_n = -13.6 eV/n² |
| Principal QN | n = n_r + l + 1 |
| Radial nodes | n_r = n - l - 1 |
| Radial WF | R_{nl} ∝ ρ^l e^{-ρ/2} L_{n-l-1}^{2l+1}(ρ) |
| ρ definition | ρ = 2r/(na₀) |

### Key Insights

1. **Quantization** emerges from requiring normalizable solutions
2. **Laguerre polynomials** are the radial eigenfunctions
3. **Radial nodes** = n - l - 1
4. **Energy depends only on n** (accidental degeneracy)

---

## Daily Checklist

- [ ] I can solve the radial equation asymptotically
- [ ] I understand how quantization emerges
- [ ] I can write the energy formula
- [ ] I know the structure of Laguerre polynomials
- [ ] I can construct specific R_{nl}(r)
- [ ] I understand the quantum number relationships

---

**Next:** [Day_430_Wednesday.md](Day_430_Wednesday.md) — Energy Spectrum
