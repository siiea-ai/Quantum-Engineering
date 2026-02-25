# Day 325: Capstone Project — Core Implementation II

## Overview

**Month 12, Week 47, Day 3 — Wednesday**

Today you complete the remaining core functions and add advanced features: time evolution, phase space representations, and special states.

## Learning Objectives

1. Implement advanced algorithms
2. Add coherent state functionality
3. Create Wigner function computation
4. Integrate all components

---

## Advanced Implementation: Coherent States and Wigner Functions

```python
"""
Day 325: Advanced Implementation - Coherent States and Phase Space
"""

import numpy as np
from scipy.special import factorial

class QuantumHarmonicOscillatorAdvanced:
    """Extended QHO with coherent states and phase space."""

    def __init__(self, mass=1, omega=1, hbar=1):
        self.m = mass
        self.omega = omega
        self.hbar = hbar
        self.x0 = np.sqrt(hbar / (mass * omega))

    def coherent_state(self, alpha, x, n_max=30):
        """
        Construct coherent state |α⟩.

        |α⟩ = exp(-|α|²/2) Σ (α^n / √n!) |n⟩

        Parameters
        ----------
        alpha : complex
            Coherent state parameter
        x : array
            Position grid
        n_max : int
            Maximum n for summation
        """
        psi = np.zeros_like(x, dtype=complex)

        prefactor = np.exp(-np.abs(alpha)**2 / 2)

        for n in range(n_max):
            coeff = alpha**n / np.sqrt(factorial(n))
            psi_n = self.eigenfunction(n, x)
            psi += prefactor * coeff * psi_n

        return psi

    def coherent_state_evolution(self, alpha_0, x, t):
        """
        Time-evolve coherent state.

        α(t) = α_0 exp(-iωt)

        The coherent state remains coherent during evolution!
        """
        alpha_t = alpha_0 * np.exp(-1j * self.omega * t)
        return self.coherent_state(alpha_t, x)

    def wigner_function(self, psi, x_grid, p_grid):
        """
        Compute Wigner quasi-probability distribution.

        W(x,p) = (1/πℏ) ∫ ψ*(x+y) ψ(x-y) exp(2ipy/ℏ) dy

        Parameters
        ----------
        psi : array
            Wavefunction on position grid
        x_grid : array
            Position values for W(x,p)
        p_grid : array
            Momentum values for W(x,p)

        Returns
        -------
        W : 2D array
            Wigner function W(x,p)
        """
        nx = len(x_grid)
        np_ = len(p_grid)
        dx = x_grid[1] - x_grid[0]

        W = np.zeros((np_, nx))

        # Use a finer y-grid for integration
        y_max = 5 * self.x0
        ny = 200
        y = np.linspace(-y_max, y_max, ny)
        dy = y[1] - y[0]

        for i, x in enumerate(x_grid):
            # Interpolate psi at x+y and x-y
            psi_plus = np.interp(x + y, x_grid, psi, left=0, right=0)
            psi_minus = np.interp(x - y, x_grid, psi, left=0, right=0)

            for j, p in enumerate(p_grid):
                integrand = np.conj(psi_plus) * psi_minus * np.exp(2j * p * y / self.hbar)
                W[j, i] = np.real(np.trapz(integrand, y)) / (np.pi * self.hbar)

        return W

    def eigenfunction(self, n, x):
        """Compute nth eigenfunction."""
        from scipy.special import hermite

        xi = x / self.x0
        norm = 1 / np.sqrt(self.x0 * np.sqrt(np.pi) * 2**n * factorial(n))
        Hn = hermite(n)
        return norm * Hn(xi) * np.exp(-xi**2 / 2)


def visualize_wigner():
    """Create Wigner function visualization."""
    import matplotlib.pyplot as plt

    qho = QuantumHarmonicOscillatorAdvanced()

    x_grid = np.linspace(-5, 5, 100)
    p_grid = np.linspace(-3, 3, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Ground state
    psi_0 = qho.eigenfunction(0, x_grid)
    W_0 = qho.wigner_function(psi_0, x_grid, p_grid)
    axes[0, 0].contourf(x_grid, p_grid, W_0, levels=50, cmap='RdBu')
    axes[0, 0].set_title('Wigner: Ground State |0⟩')

    # First excited state
    psi_1 = qho.eigenfunction(1, x_grid)
    W_1 = qho.wigner_function(psi_1, x_grid, p_grid)
    axes[0, 1].contourf(x_grid, p_grid, W_1, levels=50, cmap='RdBu')
    axes[0, 1].set_title('Wigner: First Excited |1⟩')

    # Coherent state
    alpha = 2
    psi_coh = qho.coherent_state(alpha, x_grid)
    W_coh = qho.wigner_function(psi_coh, x_grid, p_grid)
    axes[1, 0].contourf(x_grid, p_grid, W_coh, levels=50, cmap='RdBu')
    axes[1, 0].set_title(f'Wigner: Coherent |α={alpha}⟩')

    # Superposition
    psi_sup = (psi_0 + psi_1) / np.sqrt(2)
    W_sup = qho.wigner_function(psi_sup, x_grid, p_grid)
    axes[1, 1].contourf(x_grid, p_grid, W_sup, levels=50, cmap='RdBu')
    axes[1, 1].set_title('Wigner: Superposition (|0⟩+|1⟩)/√2')

    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('p')

    plt.tight_layout()
    plt.savefig('wigner_functions.png', dpi=150)
    plt.close()
    print("Saved: wigner_functions.png")


if __name__ == "__main__":
    visualize_wigner()
```

---

## Today's Checklist

- [ ] Advanced functions implemented
- [ ] Coherent states working
- [ ] Wigner function computed
- [ ] All components integrated
- [ ] Code tested

---

## Preview: Day 326

Tomorrow: **Visualization and Analysis** — create publication-quality figures.
