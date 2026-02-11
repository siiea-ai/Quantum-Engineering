# Day 487: Field Operators

## Overview

**Day 487 of 2520 | Week 70, Day 4 | Month 18: Identical Particles & Many-Body Physics**

Today we introduce field operators, which create or annihilate particles at specific positions in space. Field operators bridge the gap between the abstract occupation number formalism and the familiar wave function picture of quantum mechanics. They are the fundamental objects in quantum field theory and provide powerful tools for calculating physical observables like particle density, current, and correlation functions.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | From Discrete Modes to Continuous Position | 60 min |
| 10:00 AM | Field Operator Definitions | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Commutation/Anticommutation in Position Space | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Connection to Wave Functions | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Particle Density and Current Operators | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define** field operators $\hat{\psi}(\mathbf{r})$ and $\hat{\psi}^\dagger(\mathbf{r})$ in position space
2. **Derive** commutation/anticommutation relations for field operators
3. **Connect** field operators to single-particle wave functions
4. **Express** the particle density operator $\hat{\rho}(\mathbf{r})$ using field operators
5. **Calculate** correlation functions using field operators
6. **Transform** between momentum-space and position-space representations

---

## 1. From Discrete Modes to Continuous Position

### Review: Discrete Mode Operators

For discrete single-particle states $\{|\phi_\alpha\rangle\}$:

$$\hat{a}_\alpha \text{ (bosons)}, \quad \hat{c}_\alpha \text{ (fermions)}$$

These create/annihilate particles in state $|\phi_\alpha\rangle$.

### The Position Basis

Position eigenstates $|\mathbf{r}\rangle$ form a continuous basis:

$$\langle \mathbf{r} | \mathbf{r}' \rangle = \delta^3(\mathbf{r} - \mathbf{r}')$$

$$\int d^3r \, |\mathbf{r}\rangle\langle\mathbf{r}| = \hat{I}$$

### Motivation for Field Operators

We want operators that create/annihilate particles at **specific positions**:

$$\hat{\psi}^\dagger(\mathbf{r}) |0\rangle = |\text{particle at } \mathbf{r}\rangle$$

This requires extending our discrete formalism to continuous variables.

---

## 2. Field Operator Definitions

### Expansion in Terms of Mode Operators

Let $\{\phi_\alpha(\mathbf{r})\}$ be a complete orthonormal set of single-particle wave functions:

$$\int d^3r \, \phi_\alpha^*(\mathbf{r}) \phi_\beta(\mathbf{r}) = \delta_{\alpha\beta}$$

$$\sum_\alpha \phi_\alpha^*(\mathbf{r}) \phi_\alpha(\mathbf{r}') = \delta^3(\mathbf{r} - \mathbf{r}')$$

**Definition of field operators:**

$$\boxed{\hat{\psi}(\mathbf{r}) = \sum_\alpha \phi_\alpha(\mathbf{r}) \hat{a}_\alpha}$$

$$\boxed{\hat{\psi}^\dagger(\mathbf{r}) = \sum_\alpha \phi_\alpha^*(\mathbf{r}) \hat{a}_\alpha^\dagger}$$

(Replace $\hat{a}_\alpha$ with $\hat{c}_\alpha$ for fermions)

### Physical Interpretation

- $\hat{\psi}^\dagger(\mathbf{r})$: Creates a particle at position $\mathbf{r}$
- $\hat{\psi}(\mathbf{r})$: Annihilates a particle at position $\mathbf{r}$

### Inverse Relation

We can express mode operators in terms of field operators:

$$\hat{a}_\alpha = \int d^3r \, \phi_\alpha^*(\mathbf{r}) \hat{\psi}(\mathbf{r})$$

$$\hat{a}_\alpha^\dagger = \int d^3r \, \phi_\alpha(\mathbf{r}) \hat{\psi}^\dagger(\mathbf{r})$$

---

## 3. Commutation and Anticommutation Relations

### Bosonic Field Operators

From $[\hat{a}_\alpha, \hat{a}_\beta^\dagger] = \delta_{\alpha\beta}$:

$$[\hat{\psi}(\mathbf{r}), \hat{\psi}^\dagger(\mathbf{r}')] = \sum_{\alpha,\beta} \phi_\alpha(\mathbf{r}) \phi_\beta^*(\mathbf{r}') [\hat{a}_\alpha, \hat{a}_\beta^\dagger]$$

$$= \sum_\alpha \phi_\alpha(\mathbf{r}) \phi_\alpha^*(\mathbf{r}') = \delta^3(\mathbf{r} - \mathbf{r}')$$

**Bosonic commutation relations:**

$$\boxed{[\hat{\psi}(\mathbf{r}), \hat{\psi}^\dagger(\mathbf{r}')] = \delta^3(\mathbf{r} - \mathbf{r}')}$$

$$[\hat{\psi}(\mathbf{r}), \hat{\psi}(\mathbf{r}')] = 0$$

$$[\hat{\psi}^\dagger(\mathbf{r}), \hat{\psi}^\dagger(\mathbf{r}')] = 0$$

### Fermionic Field Operators

From $\{\hat{c}_\alpha, \hat{c}_\beta^\dagger\} = \delta_{\alpha\beta}$:

**Fermionic anticommutation relations:**

$$\boxed{\{\hat{\psi}(\mathbf{r}), \hat{\psi}^\dagger(\mathbf{r}')\} = \delta^3(\mathbf{r} - \mathbf{r}')}$$

$$\{\hat{\psi}(\mathbf{r}), \hat{\psi}(\mathbf{r}')\} = 0$$

$$\{\hat{\psi}^\dagger(\mathbf{r}), \hat{\psi}^\dagger(\mathbf{r}')\} = 0$$

### Equal-Time Relations

The above are **equal-time** relations. In relativistic quantum field theory, these become:

$$[\hat{\psi}(\mathbf{r}, t), \hat{\psi}^\dagger(\mathbf{r}', t)] = \delta^3(\mathbf{r} - \mathbf{r}')$$

with different commutators at unequal times.

---

## 4. Connection to Wave Functions

### Creating a Single-Particle State

The state with one particle in wave function $\phi(\mathbf{r})$:

$$|\phi\rangle = \int d^3r \, \phi(\mathbf{r}) \hat{\psi}^\dagger(\mathbf{r}) |0\rangle$$

**Verification:** Using $\hat{\psi}^\dagger(\mathbf{r}) = \sum_\alpha \phi_\alpha^*(\mathbf{r}) \hat{a}_\alpha^\dagger$:

$$|\phi\rangle = \int d^3r \, \phi(\mathbf{r}) \sum_\alpha \phi_\alpha^*(\mathbf{r}) \hat{a}_\alpha^\dagger |0\rangle$$

$$= \sum_\alpha \left(\int d^3r \, \phi(\mathbf{r}) \phi_\alpha^*(\mathbf{r})\right) \hat{a}_\alpha^\dagger |0\rangle$$

$$= \sum_\alpha c_\alpha \hat{a}_\alpha^\dagger |0\rangle$$

where $c_\alpha = \int d^3r \, \phi(\mathbf{r}) \phi_\alpha^*(\mathbf{r}) = \langle \phi_\alpha | \phi \rangle$ are the expansion coefficients.

### The Wave Function as a Field Expectation Value

For a single-particle state $|\Psi\rangle$ in first quantization, the wave function is:

$$\psi(\mathbf{r}) = \langle \mathbf{r} | \Psi \rangle$$

In second quantization:

$$\boxed{\psi(\mathbf{r}) = \langle 0 | \hat{\psi}(\mathbf{r}) | \Psi \rangle}$$

**Proof:** Let $|\Psi\rangle = \int d^3r' \, \psi(\mathbf{r}') \hat{\psi}^\dagger(\mathbf{r}') |0\rangle$. Then:

$$\langle 0 | \hat{\psi}(\mathbf{r}) | \Psi \rangle = \int d^3r' \, \psi(\mathbf{r}') \langle 0 | \hat{\psi}(\mathbf{r}) \hat{\psi}^\dagger(\mathbf{r}') | 0 \rangle$$

Using $[\hat{\psi}(\mathbf{r}), \hat{\psi}^\dagger(\mathbf{r}')] = \delta^3(\mathbf{r} - \mathbf{r}')$ and $\hat{\psi}|0\rangle = 0$:

$$= \int d^3r' \, \psi(\mathbf{r}') \delta^3(\mathbf{r} - \mathbf{r}') = \psi(\mathbf{r})$$ ✓

### N-Particle Wave Function

For an N-particle state $|\Psi_N\rangle$:

$$\Psi_N(\mathbf{r}_1, \ldots, \mathbf{r}_N) = \frac{1}{\sqrt{N!}} \langle 0 | \hat{\psi}(\mathbf{r}_1) \cdots \hat{\psi}(\mathbf{r}_N) | \Psi_N \rangle$$

This automatically has the correct symmetry (symmetric for bosons, antisymmetric for fermions).

---

## 5. Particle Density Operator

### Definition

The **particle density operator** at position $\mathbf{r}$:

$$\boxed{\hat{\rho}(\mathbf{r}) = \hat{\psi}^\dagger(\mathbf{r}) \hat{\psi}(\mathbf{r})}$$

### Physical Meaning

$$\langle \hat{\rho}(\mathbf{r}) \rangle = \langle \Psi | \hat{\psi}^\dagger(\mathbf{r}) \hat{\psi}(\mathbf{r}) | \Psi \rangle$$

is the **expected number density** of particles at $\mathbf{r}$.

### Total Number Operator

$$\hat{N} = \int d^3r \, \hat{\rho}(\mathbf{r}) = \int d^3r \, \hat{\psi}^\dagger(\mathbf{r}) \hat{\psi}(\mathbf{r})$$

**Verification:**
$$\hat{N} = \int d^3r \sum_{\alpha,\beta} \phi_\alpha^*(\mathbf{r}) \phi_\beta(\mathbf{r}) \hat{a}_\alpha^\dagger \hat{a}_\beta = \sum_\alpha \hat{a}_\alpha^\dagger \hat{a}_\alpha = \sum_\alpha \hat{n}_\alpha$$ ✓

### Single-Particle State Density

For $|\phi\rangle = \hat{a}^\dagger |0\rangle$ where $\hat{a}^\dagger = \int d^3r \, \phi(\mathbf{r}) \hat{\psi}^\dagger(\mathbf{r})$:

$$\langle \phi | \hat{\rho}(\mathbf{r}) | \phi \rangle = |\phi(\mathbf{r})|^2$$

The density is just $|\psi|^2$ as expected from first quantization!

---

## 6. Momentum-Space Field Operators

### Definition

Using plane-wave basis $\phi_\mathbf{k}(\mathbf{r}) = \frac{1}{\sqrt{V}} e^{i\mathbf{k} \cdot \mathbf{r}}$:

$$\boxed{\hat{\psi}(\mathbf{r}) = \frac{1}{\sqrt{V}} \sum_\mathbf{k} e^{i\mathbf{k} \cdot \mathbf{r}} \hat{a}_\mathbf{k}}$$

$$\boxed{\hat{\psi}^\dagger(\mathbf{r}) = \frac{1}{\sqrt{V}} \sum_\mathbf{k} e^{-i\mathbf{k} \cdot \mathbf{r}} \hat{a}_\mathbf{k}^\dagger}$$

### Continuum Limit

For infinite volume, $\sum_\mathbf{k} \to \frac{V}{(2\pi)^3} \int d^3k$:

$$\hat{\psi}(\mathbf{r}) = \int \frac{d^3k}{(2\pi)^{3/2}} e^{i\mathbf{k} \cdot \mathbf{r}} \hat{a}_\mathbf{k}$$

### Momentum-Space Operators

The field operator in momentum space:

$$\hat{\tilde{\psi}}(\mathbf{k}) = \int d^3r \, e^{-i\mathbf{k} \cdot \mathbf{r}} \hat{\psi}(\mathbf{r}) = \sqrt{V} \hat{a}_\mathbf{k}$$

Fourier transform relation:

$$\hat{\psi}(\mathbf{r}) = \int \frac{d^3k}{(2\pi)^{3/2}} e^{i\mathbf{k} \cdot \mathbf{r}} \hat{\tilde{\psi}}(\mathbf{k})$$

---

## 7. Correlation Functions

### One-Body Density Matrix

The **one-body density matrix** (or first-order correlation function):

$$\boxed{G^{(1)}(\mathbf{r}, \mathbf{r}') = \langle \hat{\psi}^\dagger(\mathbf{r}') \hat{\psi}(\mathbf{r}) \rangle}$$

**Properties:**
- Diagonal: $G^{(1)}(\mathbf{r}, \mathbf{r}) = \langle \hat{\rho}(\mathbf{r}) \rangle = n(\mathbf{r})$
- Off-diagonal: measures quantum coherence

### Two-Body Correlation Function

$$\boxed{G^{(2)}(\mathbf{r}_1, \mathbf{r}_2) = \langle \hat{\psi}^\dagger(\mathbf{r}_1) \hat{\psi}^\dagger(\mathbf{r}_2) \hat{\psi}(\mathbf{r}_2) \hat{\psi}(\mathbf{r}_1) \rangle}$$

**Physical meaning:** Joint probability density of finding particles at $\mathbf{r}_1$ and $\mathbf{r}_2$.

### Pair Correlation Function

$$g^{(2)}(\mathbf{r}_1, \mathbf{r}_2) = \frac{G^{(2)}(\mathbf{r}_1, \mathbf{r}_2)}{n(\mathbf{r}_1) n(\mathbf{r}_2)}$$

For a homogeneous system: $g^{(2)}(\mathbf{r}) = g^{(2)}(|\mathbf{r}_1 - \mathbf{r}_2|)$

**Key values:**
- $g^{(2)}(0) = 0$ for fermions (Pauli exclusion)
- $g^{(2)}(0) = 1$ for uncorrelated bosons
- $g^{(2)}(0) = 2$ for thermal bosons (bunching)

---

## 8. Worked Examples

### Example 1: Single-Particle Density

**Problem:** Calculate $\langle \hat{\rho}(\mathbf{r}) \rangle$ for a single particle in state $\phi(\mathbf{r}) = \sqrt{\frac{2}{L}}\sin\left(\frac{\pi x}{L}\right)$ (infinite square well ground state, 1D).

**Solution:**

The state is:
$$|\phi\rangle = \int dx \, \phi(x) \hat{\psi}^\dagger(x) |0\rangle$$

The density:
$$\langle \phi | \hat{\rho}(x) | \phi \rangle = \langle \phi | \hat{\psi}^\dagger(x) \hat{\psi}(x) | \phi \rangle$$

For a single-particle state, this equals:
$$\langle \hat{\rho}(x) \rangle = |\phi(x)|^2 = \frac{2}{L}\sin^2\left(\frac{\pi x}{L}\right)$$

$$\boxed{\langle \hat{\rho}(x) \rangle = \frac{2}{L}\sin^2\left(\frac{\pi x}{L}\right)}$$

### Example 2: Two-Fermion Correlation

**Problem:** Two non-interacting fermions occupy states $\phi_1(\mathbf{r})$ and $\phi_2(\mathbf{r})$. Calculate $G^{(1)}(\mathbf{r}, \mathbf{r}')$.

**Solution:**

The two-fermion state:
$$|\Psi\rangle = \hat{c}_1^\dagger \hat{c}_2^\dagger |0\rangle$$

Using $\hat{\psi}(\mathbf{r}) = \phi_1(\mathbf{r}) \hat{c}_1 + \phi_2(\mathbf{r}) \hat{c}_2 + \ldots$:

$$G^{(1)}(\mathbf{r}, \mathbf{r}') = \langle \Psi | \hat{\psi}^\dagger(\mathbf{r}') \hat{\psi}(\mathbf{r}) | \Psi \rangle$$

Only terms with $\hat{c}_1^\dagger \hat{c}_1$ or $\hat{c}_2^\dagger \hat{c}_2$ survive:

$$G^{(1)}(\mathbf{r}, \mathbf{r}') = \phi_1^*(\mathbf{r}') \phi_1(\mathbf{r}) \langle \hat{n}_1 \rangle + \phi_2^*(\mathbf{r}') \phi_2(\mathbf{r}) \langle \hat{n}_2 \rangle$$

$$= \phi_1^*(\mathbf{r}') \phi_1(\mathbf{r}) + \phi_2^*(\mathbf{r}') \phi_2(\mathbf{r})$$

$$\boxed{G^{(1)}(\mathbf{r}, \mathbf{r}') = \sum_{i=1}^2 \phi_i^*(\mathbf{r}') \phi_i(\mathbf{r})}$$

This is the sum of projectors onto occupied states.

### Example 3: Fermionic $g^{(2)}(0)$

**Problem:** Show that $g^{(2)}(0) = 0$ for fermions.

**Solution:**

$$G^{(2)}(\mathbf{r}, \mathbf{r}) = \langle \hat{\psi}^\dagger(\mathbf{r}) \hat{\psi}^\dagger(\mathbf{r}) \hat{\psi}(\mathbf{r}) \hat{\psi}(\mathbf{r}) \rangle$$

For fermions: $\{\hat{\psi}(\mathbf{r}), \hat{\psi}(\mathbf{r})\} = 0 \Rightarrow \hat{\psi}(\mathbf{r})^2 = 0$

Therefore:
$$G^{(2)}(\mathbf{r}, \mathbf{r}) = 0$$

$$\boxed{g^{(2)}(0) = 0}$$

**Physical interpretation:** Two fermions cannot be at the same position (Pauli exclusion).

---

## 9. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Verify that $[\hat{\psi}(x), \hat{\psi}^\dagger(x')] = \delta(x - x')$ using the plane wave expansion.

**Problem 1.2:** Calculate $\langle 0 | \hat{\psi}(x) \hat{\psi}^\dagger(x') | 0 \rangle$ and interpret the result.

**Problem 1.3:** For a single particle in state $\psi(x) = (2/a)^{1/4} e^{-x^2/a}$, calculate the expected particle number.

### Level 2: Intermediate

**Problem 2.1:** Derive the expression for the particle current operator:
$$\hat{\mathbf{j}}(\mathbf{r}) = \frac{\hbar}{2mi}[\hat{\psi}^\dagger(\mathbf{r}) \nabla \hat{\psi}(\mathbf{r}) - (\nabla \hat{\psi}^\dagger(\mathbf{r})) \hat{\psi}(\mathbf{r})]$$

**Problem 2.2:** For two bosons in the same state $\phi(\mathbf{r})$, show that:
$$G^{(2)}(\mathbf{r}, \mathbf{r}') = 2|\phi(\mathbf{r})|^2 |\phi(\mathbf{r}')|^2 + 2|\phi(\mathbf{r})|^2 |\phi(\mathbf{r}')|^2$$
What does this imply about $g^{(2)}$?

**Problem 2.3:** Express the kinetic energy operator $\hat{T} = \sum_i \frac{\hat{p}_i^2}{2m}$ in terms of field operators.

### Level 3: Challenging

**Problem 3.1:** For a uniform Fermi gas at zero temperature, calculate $G^{(1)}(\mathbf{r}, \mathbf{r}')$ and show it depends only on $|\mathbf{r} - \mathbf{r}'|$.

**Problem 3.2:** Derive the equation of motion for the field operator: $i\hbar \frac{\partial}{\partial t} \hat{\psi}(\mathbf{r}, t) = [\hat{\psi}, \hat{H}]$ for the free-particle Hamiltonian.

**Problem 3.3:** Show that for a Bose-Einstein condensate with condensate wave function $\psi_0(\mathbf{r})$, the field operator can be written as $\hat{\psi}(\mathbf{r}) = \psi_0(\mathbf{r}) + \hat{\phi}(\mathbf{r})$ where $\hat{\phi}$ describes fluctuations.

---

## 10. Computational Lab: Field Operators

```python
"""
Day 487 Computational Lab: Field Operators
Implementing field operators and calculating correlation functions.
"""

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial

class FieldOperatorSystem:
    """
    System for working with field operators in 1D.
    Uses discretized position space.
    """

    def __init__(self, x_min=-10, x_max=10, num_points=100, num_modes=10):
        """
        Initialize system.

        Parameters:
        -----------
        x_min, x_max : float
            Position space boundaries
        num_points : int
            Number of grid points
        num_modes : int
            Number of single-particle modes to include
        """
        self.x = np.linspace(x_min, x_max, num_points)
        self.dx = self.x[1] - self.x[0]
        self.num_points = num_points
        self.num_modes = num_modes

        # Generate basis functions (harmonic oscillator eigenstates)
        self.modes = self._generate_modes()

    def _generate_modes(self):
        """Generate harmonic oscillator eigenfunctions."""
        modes = []
        for n in range(self.num_modes):
            psi = self._harmonic_oscillator_wavefunction(n, self.x)
            modes.append(psi)
        return np.array(modes)

    def _harmonic_oscillator_wavefunction(self, n, x, omega=1.0, m=1.0, hbar=1.0):
        """
        Harmonic oscillator eigenfunction.
        """
        xi = np.sqrt(m * omega / hbar) * x
        norm = (m * omega / (np.pi * hbar))**0.25 / np.sqrt(2**n * factorial(n))
        H_n = hermite(n)
        return norm * H_n(xi) * np.exp(-xi**2 / 2)

    def create_single_particle_state(self, wave_function):
        """
        Create single-particle state from wave function.
        Returns expansion coefficients in mode basis.
        """
        coefficients = np.zeros(self.num_modes, dtype=complex)
        for n in range(self.num_modes):
            # c_n = ∫ φ_n*(x) ψ(x) dx
            coefficients[n] = simps(np.conj(self.modes[n]) * wave_function, self.x)
        return coefficients

    def density_from_coefficients(self, coefficients):
        """
        Calculate density ρ(x) = |ψ(x)|² from mode coefficients.
        """
        psi = np.zeros(self.num_points, dtype=complex)
        for n in range(self.num_modes):
            psi += coefficients[n] * self.modes[n]
        return np.abs(psi)**2

    def one_body_density_matrix(self, occupied_modes):
        """
        Calculate G^(1)(x, x') = Σ_n φ_n*(x') φ_n(x) for occupied modes.

        Parameters:
        -----------
        occupied_modes : list
            Indices of occupied single-particle states
        """
        G1 = np.zeros((self.num_points, self.num_points), dtype=complex)
        for n in occupied_modes:
            G1 += np.outer(self.modes[n], np.conj(self.modes[n]))
        return G1


def demonstrate_field_operators():
    """Demonstrate basic field operator concepts."""

    print("=" * 60)
    print("FIELD OPERATOR DEMONSTRATION")
    print("=" * 60)

    system = FieldOperatorSystem(x_min=-8, x_max=8, num_points=200, num_modes=15)

    # Single particle in ground state
    print("\n1. Single particle in harmonic oscillator ground state")

    # Wave function
    psi_0 = system.modes[0]

    # Density
    rho = np.abs(psi_0)**2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(system.x, psi_0, 'b-', linewidth=2, label=r'$\psi_0(x)$')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$\psi(x)$', fontsize=12)
    ax.set_title('Ground State Wave Function', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(system.x, rho, alpha=0.5, color='blue', label=r'$|\psi_0(x)|^2$')
    ax.plot(system.x, rho, 'b-', linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$\rho(x) = \langle\hat{\psi}^\dagger\hat{\psi}\rangle$', fontsize=12)
    ax.set_title('Particle Density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Verify normalization
    norm = simps(rho, system.x)
    print(f"   Normalization: ∫ρ(x)dx = {norm:.6f} (should be 1)")

    plt.tight_layout()
    plt.savefig('field_operator_basics.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_density_matrix():
    """Demonstrate one-body density matrix."""

    print("\n" + "=" * 60)
    print("ONE-BODY DENSITY MATRIX")
    print("=" * 60)

    system = FieldOperatorSystem(x_min=-8, x_max=8, num_points=100, num_modes=10)

    # Two fermions in ground and first excited states
    occupied = [0, 1]

    G1 = system.one_body_density_matrix(occupied)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Real part
    ax = axes[0]
    im = ax.imshow(np.real(G1), extent=[system.x[0], system.x[-1],
                                         system.x[0], system.x[-1]],
                   origin='lower', cmap='RdBu', aspect='auto')
    ax.set_xlabel("x'", fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title(r"Re[$G^{(1)}(x, x')$]", fontsize=12)
    plt.colorbar(im, ax=ax)

    # Diagonal (density)
    ax = axes[1]
    diagonal = np.real(np.diag(G1))
    ax.fill_between(system.x, diagonal, alpha=0.5, color='green')
    ax.plot(system.x, diagonal, 'g-', linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$G^{(1)}(x, x) = n(x)$', fontsize=12)
    ax.set_title('Density (diagonal)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Off-diagonal at x=0
    ax = axes[2]
    middle_idx = len(system.x) // 2
    slice_data = np.real(G1[middle_idx, :])
    ax.plot(system.x, slice_data, 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("x'", fontsize=12)
    ax.set_ylabel(r"$G^{(1)}(0, x')$", fontsize=12)
    ax.set_title("Off-diagonal coherence (x=0)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('density_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Calculate total particle number
    N = simps(diagonal, system.x)
    print(f"Total particle number N = ∫n(x)dx = {N:.4f} (should be 2)")


def momentum_space_representation():
    """Demonstrate momentum space field operators."""

    print("\n" + "=" * 60)
    print("MOMENTUM SPACE REPRESENTATION")
    print("=" * 60)

    # Position grid
    L = 20  # Box size
    N = 256  # Grid points
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]

    # Momentum grid
    dk = 2*np.pi / L
    k = np.fft.fftfreq(N, dx) * 2 * np.pi
    k = np.fft.fftshift(k)

    # Gaussian wave packet in position space
    sigma = 1.0
    k0 = 2.0  # Central momentum
    psi_x = (1/(2*np.pi*sigma**2))**0.25 * np.exp(-x**2/(4*sigma**2)) * np.exp(1j*k0*x)

    # Fourier transform to momentum space
    psi_k = np.fft.fftshift(np.fft.fft(psi_x)) * dx / np.sqrt(2*np.pi)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position space |ψ(x)|²
    ax = axes[0, 0]
    ax.fill_between(x, np.abs(psi_x)**2, alpha=0.5, color='blue')
    ax.plot(x, np.abs(psi_x)**2, 'b-', linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$|\psi(x)|^2$', fontsize=12)
    ax.set_title('Position-space density', fontsize=12)
    ax.set_xlim(-10, 10)
    ax.grid(True, alpha=0.3)

    # Position space real/imag
    ax = axes[0, 1]
    ax.plot(x, np.real(psi_x), 'b-', linewidth=2, label='Re')
    ax.plot(x, np.imag(psi_x), 'r--', linewidth=2, label='Im')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$\psi(x)$', fontsize=12)
    ax.set_title('Position-space wave function', fontsize=12)
    ax.set_xlim(-10, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Momentum space |ψ(k)|²
    ax = axes[1, 0]
    ax.fill_between(k, np.abs(psi_k)**2, alpha=0.5, color='green')
    ax.plot(k, np.abs(psi_k)**2, 'g-', linewidth=2)
    ax.axvline(x=k0, color='red', linestyle='--', label=f'$k_0 = {k0}$')
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel(r'$|\tilde{\psi}(k)|^2$', fontsize=12)
    ax.set_title('Momentum-space density', fontsize=12)
    ax.set_xlim(-6, 8)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Verify Parseval
    ax = axes[1, 1]
    norm_x = simps(np.abs(psi_x)**2, x)
    norm_k = simps(np.abs(psi_k)**2, k)

    text = f"""Normalization Check (Parseval's Theorem):

∫|ψ(x)|²dx = {norm_x:.6f}
∫|ψ̃(k)|²dk = {norm_k:.6f}

Uncertainty relation:
Δx = {sigma:.2f}
Δk ≈ {1/(2*sigma):.2f}
ΔxΔk ≈ {sigma * 1/(2*sigma):.2f} ≥ 1/2
"""
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', family='monospace')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('momentum_space.png', dpi=150, bbox_inches='tight')
    plt.show()


def fermi_gas_correlations():
    """Calculate correlations in a 1D Fermi gas."""

    print("\n" + "=" * 60)
    print("FERMI GAS CORRELATIONS")
    print("=" * 60)

    # 1D box with N fermions
    L = 10.0  # Box length
    N_fermions = 5  # Number of fermions

    # Plane wave modes in a box
    def plane_wave(n, x, L):
        """Normalized plane wave: φ_n(x) = (1/√L) e^(ikn*x)"""
        k_n = 2 * np.pi * n / L
        return np.exp(1j * k_n * x) / np.sqrt(L)

    x = np.linspace(0, L, 200)

    # Fill lowest N modes (Fermi sea)
    # For N odd: n = -(N-1)/2, ..., 0, ..., (N-1)/2
    modes = list(range(-(N_fermions-1)//2, (N_fermions)//2 + 1))
    print(f"Occupied modes: {modes}")

    # Calculate density n(x) = Σ |φ_n(x)|²
    density = np.zeros_like(x)
    for n in modes:
        psi_n = plane_wave(n, x, L)
        density += np.abs(psi_n)**2

    # Calculate G^(1)(x, x')
    G1 = np.zeros((len(x), len(x)), dtype=complex)
    for n in modes:
        psi_n = plane_wave(n, x, L)
        G1 += np.outer(psi_n, np.conj(psi_n))

    # G^(1) for homogeneous system depends only on x-x'
    # Plot G^(1)(0, x')
    G1_slice = G1[0, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(x, np.real(density), 'b-', linewidth=2)
    ax.axhline(y=N_fermions/L, color='red', linestyle='--',
               label=f'n = N/L = {N_fermions/L:.2f}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('n(x)', fontsize=12)
    ax.set_title('Fermi Gas Density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    im = ax.imshow(np.abs(G1)**2, extent=[0, L, 0, L],
                   origin='lower', cmap='hot', aspect='auto')
    ax.set_xlabel("x'", fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title(r"$|G^{(1)}(x, x')|^2$", fontsize=12)
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    # Theoretical for homogeneous Fermi gas: sin(k_F r) / (k_F r)
    k_F = np.pi * N_fermions / L
    r = x - x[0]
    r[0] = 1e-10  # Avoid division by zero
    theory = N_fermions/L * np.sin(k_F * r) / (k_F * r)
    theory[0] = N_fermions/L

    ax.plot(x, np.real(G1_slice), 'b-', linewidth=2, label='Numerical')
    ax.plot(x, np.real(theory), 'r--', linewidth=2, label='Theory')
    ax.set_xlabel("x' (x = 0)", fontsize=12)
    ax.set_ylabel(r"$G^{(1)}(0, x')$", fontsize=12)
    ax.set_title('Off-diagonal coherence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fermi_gas_correlations.png', dpi=150, bbox_inches='tight')
    plt.show()


def quantum_computing_connection():
    """Discuss quantum computing applications."""

    print("\n" + "=" * 60)
    print("QUANTUM COMPUTING CONNECTION")
    print("=" * 60)

    print("""
    FIELD OPERATORS IN QUANTUM COMPUTING
    ====================================

    1. QUANTUM SIMULATION OF FIELD THEORIES:
       - Lattice discretization: ψ(x) → ψ_j at discrete sites
       - Field operators become ordinary mode operators
       - Simulation of QFT on quantum computers

    2. ANALOG QUANTUM SIMULATORS:
       - Cold atoms in optical lattices
       - Superconducting circuits (circuit QED)
       - Ion traps with motional modes
       - These naturally implement field operators!

    3. DIGITAL QUANTUM SIMULATION:
       - Jordan-Wigner or Bravyi-Kitaev mapping
       - Trotterized time evolution
       - VQE for ground state energies

    4. CORRELATION FUNCTION MEASUREMENT:
       - G^(1): single-particle coherence
       - G^(2): density-density correlations
       - Higher-order correlations accessible

    5. APPLICATIONS:
       - Quantum chemistry (molecular orbital theory)
       - Nuclear physics (shell model)
       - High-energy physics (lattice QCD)
       - Condensed matter (strongly correlated systems)

    Key Insight:
    ------------
    Field operators are the bridge between:
    - Abstract quantum information (qubits, gates)
    - Physical systems (atoms, electrons, photons)

    Every quantum simulation ultimately computes expectation
    values of operators built from field operators!
    """)


# Main execution
if __name__ == "__main__":
    print("Day 487: Field Operators")
    print("=" * 60)

    demonstrate_field_operators()
    demonstrate_density_matrix()
    momentum_space_representation()
    fermi_gas_correlations()
    quantum_computing_connection()
```

---

## 11. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Field operators | $\hat{\psi}(\mathbf{r})$, $\hat{\psi}^\dagger(\mathbf{r})$ create/annihilate at position |
| Mode expansion | $\hat{\psi}(\mathbf{r}) = \sum_\alpha \phi_\alpha(\mathbf{r}) \hat{a}_\alpha$ |
| Density operator | $\hat{\rho}(\mathbf{r}) = \hat{\psi}^\dagger(\mathbf{r})\hat{\psi}(\mathbf{r})$ |
| One-body density matrix | $G^{(1)}(\mathbf{r}, \mathbf{r}') = \langle \hat{\psi}^\dagger(\mathbf{r}')\hat{\psi}(\mathbf{r}) \rangle$ |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$[\hat{\psi}(\mathbf{r}), \hat{\psi}^\dagger(\mathbf{r}')] = \delta^3(\mathbf{r} - \mathbf{r}')$$ | Bosonic commutation |
| $$\{\hat{\psi}(\mathbf{r}), \hat{\psi}^\dagger(\mathbf{r}')\} = \delta^3(\mathbf{r} - \mathbf{r}')$$ | Fermionic anticommutation |
| $$\psi(\mathbf{r}) = \langle 0 | \hat{\psi}(\mathbf{r}) | \Psi \rangle$$ | Wave function from field operator |
| $$\hat{N} = \int d^3r \, \hat{\psi}^\dagger(\mathbf{r}) \hat{\psi}(\mathbf{r})$$ | Total number operator |

---

## 12. Daily Checklist

### Conceptual Understanding
- [ ] I can define field operators and explain their physical meaning
- [ ] I understand commutation/anticommutation in position space
- [ ] I can connect field operators to wave functions
- [ ] I know how to calculate density and correlation functions

### Mathematical Skills
- [ ] I can derive field operator commutation relations
- [ ] I can transform between position and momentum representations
- [ ] I can calculate one-body density matrix elements
- [ ] I can verify normalization using field operators

### Computational Skills
- [ ] I implemented field operators on a discretized grid
- [ ] I calculated and visualized the density matrix
- [ ] I studied Fermi gas correlations numerically

### Quantum Computing Connection
- [ ] I understand how field operators relate to lattice models
- [ ] I see the connection to quantum simulation
- [ ] I know correlation functions are key observables

---

## 13. Preview: Day 488

Tomorrow we study **many-body Hamiltonians** in second quantization:

- One-body operators: $\hat{T} = \int d^3r \, \hat{\psi}^\dagger(\mathbf{r}) h(\mathbf{r}) \hat{\psi}(\mathbf{r})$
- Two-body interactions: $\hat{V} = \frac{1}{2} \int d^3r d^3r' \, \hat{\psi}^\dagger(\mathbf{r}) \hat{\psi}^\dagger(\mathbf{r}') V(\mathbf{r}, \mathbf{r}') \hat{\psi}(\mathbf{r}') \hat{\psi}(\mathbf{r})$
- Coulomb interaction in second quantization
- Normal ordering and Wick's theorem

This will complete our toolkit for describing interacting many-body systems.

---

## References

1. Fetter, A.L. & Walecka, J.D. (2003). *Quantum Theory of Many-Particle Systems*. Dover, Ch. 1-2.

2. Negele, J.W. & Orland, H. (1998). *Quantum Many-Particle Systems*. Westview Press, Ch. 1.

3. Mahan, G.D. (2000). *Many-Particle Physics*, 3rd ed. Kluwer, Ch. 1.

4. Altland, A. & Simons, B. (2010). *Condensed Matter Field Theory*, 2nd ed. Cambridge, Ch. 2.

---

*"The field operator is the fundamental building block of quantum field theory, encoding both particle creation and wave-like propagation."*
— Julian Schwinger

---

**Day 487 Complete.** Tomorrow: Many-Body Hamiltonians.
