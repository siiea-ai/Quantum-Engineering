# Day 324: Capstone Project — Core Implementation I

## Overview

**Month 12, Week 47, Day 2 — Tuesday**

Today you implement the core mathematical functions for your capstone project. Focus on correctness and clarity over optimization.

## Learning Objectives

1. Implement core algorithms
2. Verify mathematical correctness
3. Write clean, documented code
4. Handle edge cases

---

## Implementation Guidelines

### Code Quality Standards

```python
"""
Example: Well-documented function

Includes:
- Docstring with description, parameters, returns
- Type hints (optional but recommended)
- Input validation
- Clear variable names
"""

import numpy as np
from scipy.special import hermite
from typing import Union, Callable

def eigenfunction(n: int, x: Union[float, np.ndarray],
                  mass: float = 1, omega: float = 1, hbar: float = 1) -> np.ndarray:
    """
    Compute the nth eigenfunction of the quantum harmonic oscillator.

    Parameters
    ----------
    n : int
        Quantum number (n >= 0)
    x : float or array
        Position(s) at which to evaluate
    mass : float
        Particle mass (default: 1)
    omega : float
        Angular frequency (default: 1)
    hbar : float
        Reduced Planck constant (default: 1)

    Returns
    -------
    psi : ndarray
        Wavefunction values at positions x

    Notes
    -----
    ψ_n(x) = (mω/πℏ)^{1/4} * (1/√(2^n n!)) * H_n(ξ) * exp(-ξ²/2)
    where ξ = √(mω/ℏ) * x

    Examples
    --------
    >>> psi = eigenfunction(0, np.linspace(-5, 5, 100))
    >>> np.allclose(np.trapz(psi**2, np.linspace(-5, 5, 100)), 1, atol=0.01)
    True
    """
    if n < 0 or not isinstance(n, int):
        raise ValueError(f"Quantum number n must be non-negative integer, got {n}")

    x = np.asarray(x)

    # Dimensionless coordinate
    xi = np.sqrt(mass * omega / hbar) * x

    # Normalization constant
    norm = (mass * omega / (np.pi * hbar))**0.25
    norm *= 1 / np.sqrt(2**n * np.math.factorial(n))

    # Hermite polynomial
    Hn = hermite(n)

    # Wavefunction
    psi = norm * Hn(xi) * np.exp(-xi**2 / 2)

    return psi
```

---

## Example Implementation: Harmonic Oscillator Core

```python
"""
Day 324: Core Implementation - Quantum Harmonic Oscillator
"""

import numpy as np
from scipy.special import hermite
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt


class QuantumHarmonicOscillator:
    """
    Complete quantum harmonic oscillator implementation.

    Provides analytical and numerical solutions, time evolution,
    and phase space representations.
    """

    def __init__(self, mass=1, omega=1, hbar=1):
        """Initialize oscillator parameters."""
        self.m = mass
        self.omega = omega
        self.hbar = hbar

        # Characteristic length scale
        self.x0 = np.sqrt(hbar / (mass * omega))

    # =========== ANALYTICAL SOLUTIONS ===========

    def energy(self, n):
        """Return the nth energy eigenvalue E_n = ℏω(n + 1/2)."""
        return self.hbar * self.omega * (n + 0.5)

    def eigenfunction(self, n, x):
        """
        Compute the nth eigenfunction ψ_n(x).

        Uses Hermite polynomials from scipy.
        """
        xi = x / self.x0  # Dimensionless coordinate

        # Normalization
        norm = 1 / np.sqrt(self.x0 * np.sqrt(np.pi) * 2**n * np.math.factorial(n))

        # Hermite polynomial
        Hn = hermite(n)

        return norm * Hn(xi) * np.exp(-xi**2 / 2)

    # =========== NUMERICAL SOLUTIONS ===========

    def solve_numerically(self, x_grid, n_states=10):
        """
        Solve Schrödinger equation numerically using finite differences.

        Returns eigenvalues and eigenvectors on the grid.
        """
        N = len(x_grid)
        dx = x_grid[1] - x_grid[0]

        # Kinetic energy (finite difference)
        kinetic_coef = -self.hbar**2 / (2 * self.m * dx**2)

        # Potential energy
        V = 0.5 * self.m * self.omega**2 * x_grid**2

        # Tridiagonal Hamiltonian
        diagonal = -2 * kinetic_coef + V
        off_diagonal = kinetic_coef * np.ones(N - 1)

        # Solve eigenvalue problem
        eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off_diagonal)

        # Normalize eigenvectors
        for i in range(eigenvectors.shape[1]):
            eigenvectors[:, i] /= np.sqrt(np.trapz(eigenvectors[:, i]**2, x_grid))

        return eigenvalues[:n_states], eigenvectors[:, :n_states]

    # =========== TIME EVOLUTION ===========

    def time_evolve(self, psi_0, x, t, n_max=20):
        """
        Time-evolve initial state using spectral decomposition.

        psi(x, t) = Σ c_n * ψ_n(x) * exp(-i E_n t / ℏ)
        """
        # Compute expansion coefficients
        coeffs = []
        for n in range(n_max):
            psi_n = self.eigenfunction(n, x)
            c_n = np.trapz(np.conj(psi_n) * psi_0, x)
            coeffs.append(c_n)

        # Time evolution
        psi_t = np.zeros_like(psi_0, dtype=complex)
        for n, c_n in enumerate(coeffs):
            phase = np.exp(-1j * self.energy(n) * t / self.hbar)
            psi_t += c_n * phase * self.eigenfunction(n, x)

        return psi_t

    # =========== VALIDATION ===========

    def validate(self, n_max=5, x_range=(-10, 10), n_points=500):
        """Validate implementation against known results."""
        x = np.linspace(*x_range, n_points)

        print("Validation Results")
        print("=" * 50)

        # Check energies
        print("\n1. Energy eigenvalues:")
        for n in range(n_max):
            E_analytical = self.energy(n)
            print(f"   E_{n} = {E_analytical:.4f} ℏω")

        # Check normalization
        print("\n2. Wavefunction normalization:")
        for n in range(n_max):
            psi = self.eigenfunction(n, x)
            norm = np.trapz(np.abs(psi)**2, x)
            status = "✓" if np.abs(norm - 1) < 0.01 else "✗"
            print(f"   ∫|ψ_{n}|² dx = {norm:.6f} {status}")

        # Check orthogonality
        print("\n3. Orthogonality:")
        for n in range(min(3, n_max)):
            for m in range(n+1, min(4, n_max)):
                psi_n = self.eigenfunction(n, x)
                psi_m = self.eigenfunction(m, x)
                overlap = np.trapz(psi_n * psi_m, x)
                status = "✓" if np.abs(overlap) < 0.01 else "✗"
                print(f"   ⟨ψ_{n}|ψ_{m}⟩ = {overlap:.6f} {status}")

        # Compare analytical and numerical
        print("\n4. Analytical vs Numerical:")
        eigenvalues, _ = self.solve_numerically(x)
        for n in range(min(5, len(eigenvalues))):
            E_anal = self.energy(n)
            E_num = eigenvalues[n]
            error = abs(E_num - E_anal) / E_anal * 100
            status = "✓" if error < 1 else "✗"
            print(f"   n={n}: analytical={E_anal:.4f}, "
                  f"numerical={E_num:.4f}, error={error:.2f}% {status}")


# Test the implementation
if __name__ == "__main__":
    qho = QuantumHarmonicOscillator()
    qho.validate()
```

---

## Today's Checklist

- [ ] Core class/module structure created
- [ ] Main functions implemented
- [ ] Basic validation tests pass
- [ ] Code is well-documented
- [ ] Edge cases handled

---

## Preview: Day 325

Tomorrow: **Core Implementation II** — complete remaining functions and add advanced features.
