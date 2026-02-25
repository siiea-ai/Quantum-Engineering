# Day 190: Green's Functions and Propagators

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Green's Functions |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Propagator Calculations |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 190, you will be able to:

1. Define Green's functions as inverses of differential operators
2. Compute Green's functions using contour integration
3. Understand the resolvent and its analytic properties
4. Relate poles to bound states and branch cuts to continua
5. Calculate propagators for free particles
6. Apply the $+i\varepsilon$ prescription for causality

---

## Core Content

### 1. Green's Functions: Definition

**Physical Motivation:** Solve $\mathcal{L}\psi = f$ where $\mathcal{L}$ is a differential operator.

**Definition:** The Green's function $G(x, x')$ satisfies:
$$\mathcal{L}_x G(x, x') = \delta(x - x')$$

Then the solution to $\mathcal{L}\psi = f$ is:
$$\psi(x) = \int G(x, x') f(x') dx'$$

### 2. The Resolvent Operator

**Definition:** For Hamiltonian $H$, the resolvent is:
$$\boxed{G(E) = (E - H)^{-1}}$$

**Properties:**
- Analytic in $E$ except at spectrum of $H$
- Poles at discrete eigenvalues
- Branch cuts along continuous spectrum

**Spectral representation:**
$$G(E) = \sum_n \frac{|n\rangle\langle n|}{E - E_n} + \int \frac{|\alpha\rangle\langle\alpha|}{E - E_\alpha} d\alpha$$

### 3. Free Particle Propagator

**Time-independent:** The free particle Green's function in energy representation:

$$G_0(\mathbf{r}, \mathbf{r}'; E) = \langle \mathbf{r} | \frac{1}{E - H_0 + i\varepsilon} | \mathbf{r}' \rangle$$

**1D calculation:**
$$G_0(x, x'; E) = -\frac{im}{\hbar^2 k} e^{ik|x-x'|}$$

where $k = \sqrt{2mE}/\hbar$.

**3D calculation:**
$$G_0(\mathbf{r}, \mathbf{r}'; E) = -\frac{m}{2\pi\hbar^2} \frac{e^{ik|\mathbf{r}-\mathbf{r}'|}}{|\mathbf{r}-\mathbf{r}'|}$$

### 4. The $+i\varepsilon$ Prescription

**Problem:** $E$ may be real and hit the continuous spectrum.

**Solution:** Add infinitesimal imaginary part:
$$G^{\pm}(E) = \frac{1}{E - H \pm i\varepsilon}$$

- $G^+$ (retarded): analytic in upper half-plane → causal (effect after cause)
- $G^-$ (advanced): analytic in lower half-plane → anti-causal

**Physical meaning:** $+i\varepsilon$ selects outgoing waves at infinity.

### 5. Contour Integral Computation

**Example:** Compute 1D free particle Green's function.

$$G_0(x, x'; E) = \int_{-\infty}^{\infty} \frac{dp}{2\pi\hbar} \frac{e^{ip(x-x')/\hbar}}{E - p^2/2m + i\varepsilon}$$

**Poles:** $p = \pm\sqrt{2mE + i\varepsilon} \approx \pm\hbar k \mp i\varepsilon'$

For $x > x'$, close in upper half-plane:
- Pick up pole at $p = +\hbar k - i\varepsilon'$

$$G_0 = \frac{1}{2\pi\hbar} \cdot 2\pi i \cdot \frac{e^{ik(x-x')}}{-\hbar k/m} = -\frac{im}{\hbar^2 k}e^{ik(x-x')}$$

### 6. Time-Dependent Propagator

**Definition:**
$$K(x, t; x', 0) = \langle x | e^{-iHt/\hbar} | x' \rangle$$

**Relation to Green's function:**
$$K(x, t; x', 0) = \int_{-\infty}^{\infty} \frac{dE}{2\pi\hbar} e^{-iEt/\hbar} G(x, x'; E)$$

**Free particle result:**
$$K_0(x, t; x', 0) = \sqrt{\frac{m}{2\pi i\hbar t}} \exp\left(\frac{im(x-x')^2}{2\hbar t}\right)$$

---

## Worked Examples

### Example 1: Harmonic Oscillator Green's Function

**Problem:** Find poles of $G(E)$ for harmonic oscillator.

**Solution:**
Eigenvalues: $E_n = \hbar\omega(n + 1/2)$

$$G(E) = \sum_{n=0}^{\infty} \frac{|n\rangle\langle n|}{E - \hbar\omega(n + 1/2)}$$

**Poles:** At $E = \hbar\omega(n + 1/2)$ for $n = 0, 1, 2, \ldots$

**Residue at $E_n$:** $|n\rangle\langle n|$ (projection onto eigenstate)

### Example 2: Attractive Delta Function Potential

**Problem:** Find bound state from pole of $G(E)$.

For $V(x) = -\alpha\delta(x)$:

**T-matrix equation:**
$$T(E) = V + VG_0V + VG_0VG_0V + \cdots = \frac{V}{1 - VG_0}$$

With $G_0(0,0;E) = -im/(\hbar^2 k)$ and $V = -\alpha$:
$$T(E) = \frac{-\alpha}{1 + \alpha m/(i\hbar^2 k)}$$

**Pole condition:** $1 + \alpha m/(i\hbar^2 k) = 0$
$$k = \frac{im\alpha}{\hbar^2}$$

Since $k = \sqrt{2mE}/\hbar$, this gives:
$$E = -\frac{m\alpha^2}{2\hbar^2}$$

This is the bound state energy!

---

## Practice Problems

**P1.** Derive the 3D free particle Green's function using contour integration.

**P2.** Show that $G^+(E)$ satisfies outgoing wave boundary conditions.

**P3.** Compute the density of states from $\rho(E) = -\frac{1}{\pi}\text{Im Tr } G^+(E)$.

**P4.** Find the propagator for a particle in a box using the spectral representation.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

def free_particle_greens_1d(x, xp, E, m=1, hbar=1, epsilon=1e-6):
    """1D free particle retarded Green's function."""
    E_complex = E + 1j * epsilon
    k = np.sqrt(2 * m * E_complex) / hbar
    return -1j * m / (hbar**2 * k) * np.exp(1j * k * np.abs(x - xp))

def propagator_from_greens(x, xp, t, E_range, m=1, hbar=1):
    """Compute time-dependent propagator via Fourier transform."""
    dE = E_range[1] - E_range[0]
    integrand = np.array([free_particle_greens_1d(x, xp, E, m, hbar) *
                         np.exp(-1j * E * t / hbar) for E in E_range])
    return np.trapz(integrand, E_range) / (2 * np.pi * hbar)

# Visualize Green's function
E_values = np.linspace(0.1, 5, 100)
x = np.linspace(-5, 5, 200)
xp = 0

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# G(x, 0; E) for fixed E
for E in [0.5, 1, 2]:
    G = free_particle_greens_1d(x, xp, E)
    axes[0, 0].plot(x, G.real, label=f'Re G, E={E}')
    axes[0, 1].plot(x, G.imag, label=f'Im G, E={E}')

axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Re G(x, 0; E)')
axes[0, 0].set_title('Real Part of Green\'s Function')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Im G(x, 0; E)')
axes[0, 1].set_title('Imaginary Part (Propagating Waves)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Spectral function A(E) = -Im G / π
A = -np.array([free_particle_greens_1d(0, 0, E).imag for E in E_values]) / np.pi
axes[1, 0].plot(E_values, A)
axes[1, 0].set_xlabel('E')
axes[1, 0].set_ylabel('A(E)')
axes[1, 0].set_title('Spectral Function (Density of States)')
axes[1, 0].grid(True, alpha=0.3)

# Analytic structure in complex E plane
E_real = np.linspace(-1, 3, 100)
E_imag = np.linspace(-1, 1, 100)
E_R, E_I = np.meshgrid(E_real, E_imag)
E_complex = E_R + 1j * E_I

G_complex = free_particle_greens_1d(0, 0, E_complex)
axes[1, 1].contourf(E_R, E_I, np.log10(np.abs(G_complex) + 1e-10),
                   levels=50, cmap='viridis')
axes[1, 1].axhline(y=0, color='r', linewidth=2, label='Branch cut')
axes[1, 1].set_xlabel('Re(E)')
axes[1, 1].set_ylabel('Im(E)')
axes[1, 1].set_title('|G(E)| in Complex E Plane')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('greens_functions.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $G(E) = (E - H)^{-1}$ | Resolvent definition |
| $G^+(E) = (E - H + i\varepsilon)^{-1}$ | Retarded Green's function |
| $G_0 = -\frac{im}{\hbar^2 k}e^{ik\|x-x'\|}$ | 1D free particle |
| $\rho(E) = -\frac{1}{\pi}\text{Im Tr } G^+$ | Density of states |

### Main Takeaways

1. Green's functions invert differential operators
2. Poles correspond to bound states, residues give projectors
3. The $+i\varepsilon$ prescription ensures causality
4. Contour integration computes Green's functions efficiently
5. Branch cuts encode continuous spectra

---

## Preview: Day 191

Tomorrow: **Dispersion Relations and Causality**
- Kramers-Kronig relations from analyticity
- Sum rules in quantum mechanics
- Optical theorem

---

*"The Green's function contains all the physics."*
— Common saying in theoretical physics
