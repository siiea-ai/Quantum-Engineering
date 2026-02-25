# Day 278: Quantum Eigenvalue Problems

## Schedule Overview
**Date**: Week 40, Day 5 (Friday)
**Duration**: 7 hours
**Theme**: Numerical Methods for Finding Quantum Energy Levels and Eigenstates

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Shooting method, matrix diagonalization |
| Afternoon | 2.5 hours | Variational methods, anharmonic oscillators |
| Evening | 1.5 hours | Computational lab: Universal eigenvalue solver |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Implement the shooting method for bound states
2. Construct Hamiltonian matrices for eigenvalue problems
3. Apply variational methods to estimate ground state energies
4. Solve eigenvalue problems for arbitrary potentials
5. Compare numerical results with analytical solutions

---

## Core Content

### 1. The Eigenvalue Problem

The time-independent Schrödinger equation:
$$\hat{H}\psi_n = E_n\psi_n$$

In position representation:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

### 2. Matrix Diagonalization Method

Discretize on a grid and represent derivatives as finite differences:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

class QuantumEigenSolver:
    """
    Solve time-independent Schrödinger equation using matrix methods.
    """

    def __init__(self, x_min=-10, x_max=10, nx=500, hbar=1, m=1):
        self.hbar = hbar
        self.m = m
        self.nx = nx

        self.x = np.linspace(x_min, x_max, nx)
        self.dx = self.x[1] - self.x[0]

        # Kinetic energy matrix (second derivative)
        self.T = self._build_kinetic_matrix()

    def _build_kinetic_matrix(self):
        """
        Build kinetic energy matrix using finite differences.

        d²ψ/dx² ≈ (ψ_{i+1} - 2ψ_i + ψ_{i-1}) / dx²
        """
        coeff = -self.hbar**2 / (2 * self.m * self.dx**2)
        diagonal = -2 * coeff * np.ones(self.nx)
        off_diagonal = coeff * np.ones(self.nx - 1)

        T = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
        return T

    def set_potential(self, V_func):
        """Set potential and build full Hamiltonian."""
        self.V = V_func(self.x)
        self.H = self.T + np.diag(self.V)

    def solve(self, n_states=10):
        """
        Solve eigenvalue problem.

        Returns energies and normalized wave functions.
        """
        eigenvalues, eigenvectors = eigh(self.H)

        # Select first n_states
        E = eigenvalues[:n_states]
        psi = eigenvectors[:, :n_states].T

        # Normalize
        for i in range(n_states):
            norm = np.sqrt(np.trapz(np.abs(psi[i])**2, self.x))
            psi[i] /= norm
            # Fix sign convention (positive at some point)
            if psi[i, len(self.x)//4] < 0:
                psi[i] *= -1

        return E, psi

    def solve_sparse(self, n_states=10):
        """Solve using sparse eigenvalue solver (efficient for large matrices)."""
        H_sparse = diags(
            [self.V - 2*self.T[0,0]/self.dx**2*np.ones(self.nx),
             self.T[0,1]/self.dx**2*np.ones(self.nx-1),
             self.T[0,1]/self.dx**2*np.ones(self.nx-1)],
            [0, 1, -1]
        )
        E, psi = eigsh(H_sparse, k=n_states, which='SA')

        # Sort by energy
        idx = np.argsort(E)
        E = E[idx]
        psi = psi[:, idx].T

        # Normalize
        for i in range(n_states):
            norm = np.sqrt(np.trapz(np.abs(psi[i])**2, self.x))
            psi[i] /= norm

        return E, psi


# Example: Harmonic Oscillator
solver = QuantumEigenSolver(x_min=-8, x_max=8, nx=500)
omega = 1.0
solver.set_potential(lambda x: 0.5 * omega**2 * x**2)

E, psi = solver.solve(n_states=6)

# Compare with analytical
E_exact = np.array([n + 0.5 for n in range(6)])
print("Harmonic Oscillator Eigenvalues:")
print("n    Numerical    Exact       Error")
for n, (E_num, E_ex) in enumerate(zip(E, E_exact)):
    print(f"{n}    {E_num:.6f}    {E_ex:.6f}    {abs(E_num-E_ex):.2e}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Eigenstates in potential
ax = axes[0, 0]
ax.plot(solver.x, solver.V, 'k-', linewidth=2, label='V(x)')
for n in range(5):
    scale = 0.3
    ax.fill_between(solver.x, E[n], E[n] + scale * psi[n]**2 / np.max(np.abs(psi[n])**2),
                   alpha=0.5)
    ax.axhline(E[n], color=f'C{n}', linestyle='--', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('Energy / Wave function')
ax.set_title('Harmonic Oscillator Eigenstates')
ax.set_xlim(-5, 5)
ax.set_ylim(-0.5, 6)

# Wave functions
ax = axes[0, 1]
for n in range(5):
    ax.plot(solver.x, psi[n] + n, label=f'n={n}')
ax.set_xlabel('x')
ax.set_ylabel('ψ_n(x) (offset)')
ax.set_title('Wave Functions')
ax.set_xlim(-5, 5)
ax.legend()

# Energy level comparison
ax = axes[1, 0]
ax.plot(range(6), E, 'bo', markersize=10, label='Numerical')
ax.plot(range(6), E_exact, 'r+', markersize=15, mew=3, label='Exact')
ax.set_xlabel('n')
ax.set_ylabel('Energy')
ax.set_title('Energy Eigenvalues')
ax.legend()

# Probability densities
ax = axes[1, 1]
for n in range(3):
    ax.plot(solver.x, np.abs(psi[n])**2, label=f'|ψ_{n}|²')
ax.set_xlabel('x')
ax.set_ylabel('Probability density')
ax.set_title('Probability Densities')
ax.set_xlim(-5, 5)
ax.legend()

plt.tight_layout()
plt.savefig('harmonic_oscillator_eigenstates.png', dpi=150)
plt.show()
```

### 3. Shooting Method

For highly accurate eigenvalues, use the shooting method:

```python
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

class ShootingMethod:
    """
    Find eigenvalues using the shooting method.

    Integrates from both boundaries and matches at midpoint.
    """

    def __init__(self, x_min=-10, x_max=10, V_func=None, hbar=1, m=1):
        self.x_min = x_min
        self.x_max = x_max
        self.V = V_func
        self.hbar = hbar
        self.m = m

    def schrodinger_ode(self, x, y, E):
        """
        Schrödinger equation as first-order system.

        y = [ψ, dψ/dx]
        dy/dx = [dψ/dx, (2m/ℏ²)(V-E)ψ]
        """
        psi, dpsi = y
        d2psi = (2 * self.m / self.hbar**2) * (self.V(x) - E) * psi
        return [dpsi, d2psi]

    def shoot(self, E, n_nodes_target):
        """
        Shoot from left boundary.

        Returns final ψ value (should be 0 for eigenvalue).
        """
        # Initial conditions at left boundary
        # For even states: ψ(0) ≠ 0, ψ'(0) = 0
        # For odd states: ψ(0) = 0, ψ'(0) ≠ 0
        y0 = [0, 1e-10] if n_nodes_target % 2 == 1 else [1, 0]

        x_span = (self.x_min, self.x_max)
        sol = solve_ivp(lambda x, y: self.schrodinger_ode(x, y, E),
                       x_span, y0, max_step=0.01)

        # Count nodes
        psi = sol.y[0]
        n_nodes = np.sum(np.diff(np.sign(psi)) != 0)

        return sol.y[0, -1], n_nodes

    def find_eigenvalue(self, n, E_min, E_max, tol=1e-10):
        """Find n-th eigenvalue using bisection."""

        def objective(E):
            psi_end, n_nodes = self.shoot(E, n)
            return psi_end

        # Binary search
        try:
            E_n = brentq(objective, E_min, E_max, xtol=tol)
            return E_n
        except ValueError:
            return None

    def find_eigenvalues(self, n_states, E_range):
        """Find first n_states eigenvalues."""
        eigenvalues = []
        E_min, E_max = E_range

        dE = (E_max - E_min) / 100
        E = E_min

        while len(eigenvalues) < n_states and E < E_max:
            try:
                E_n = self.find_eigenvalue(len(eigenvalues), E, E + 2*dE)
                if E_n is not None:
                    eigenvalues.append(E_n)
                    E = E_n + dE
            except:
                pass
            E += dE

        return np.array(eigenvalues)


# Test on harmonic oscillator
shooter = ShootingMethod(x_min=-8, x_max=8,
                        V_func=lambda x: 0.5 * x**2)

print("\nShooting Method Results:")
print("n    E_shooting    E_exact")
for n in range(5):
    E_min = n
    E_max = n + 2
    E_n = shooter.find_eigenvalue(n, E_min, E_max)
    if E_n:
        print(f"{n}    {E_n:.8f}    {n + 0.5:.8f}")
```

### 4. Variational Method

The variational principle: for any trial wave function $$\phi$$,
$$E_0 \leq \frac{\langle\phi|\hat{H}|\phi\rangle}{\langle\phi|\phi\rangle}$$

```python
from scipy.optimize import minimize

def variational_method(V_func, trial_func, x, params_init):
    """
    Apply variational method with parametrized trial function.

    Parameters
    ----------
    V_func : callable
        Potential energy function
    trial_func : callable
        Trial wave function ψ(x, params)
    x : array
        Position grid
    params_init : array
        Initial parameter guess
    """
    dx = x[1] - x[0]

    def energy(params):
        psi = trial_func(x, params)
        psi_norm = psi / np.sqrt(np.trapz(np.abs(psi)**2, x))

        # Kinetic energy
        d2psi = np.gradient(np.gradient(psi_norm, dx), dx)
        T = -0.5 * np.trapz(np.conj(psi_norm) * d2psi, x)

        # Potential energy
        V = np.trapz(V_func(x) * np.abs(psi_norm)**2, x)

        return (T + V).real

    result = minimize(energy, params_init, method='Nelder-Mead')
    return result.x, result.fun


# Example: Quartic oscillator V = x⁴
x = np.linspace(-5, 5, 1000)

def quartic_potential(x):
    return x**4

# Trial function: Gaussian with width parameter
def gaussian_trial(x, params):
    alpha = params[0]
    return np.exp(-alpha * x**2)

params_opt, E_variational = variational_method(
    quartic_potential, gaussian_trial, x, [0.5]
)

print(f"\nVariational Method for V = x⁴:")
print(f"Optimal α = {params_opt[0]:.4f}")
print(f"Variational energy = {E_variational:.6f}")

# Compare with matrix method
solver = QuantumEigenSolver(x_min=-5, x_max=5, nx=500)
solver.set_potential(quartic_potential)
E_exact, _ = solver.solve(n_states=1)
print(f"Matrix method energy = {E_exact[0]:.6f}")
print(f"Variational error = {(E_variational - E_exact[0])/E_exact[0]*100:.2f}%")
```

### 5. Anharmonic Potentials

```python
def anharmonic_oscillator_study():
    """Study eigenstates of various anharmonic potentials."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    x = np.linspace(-5, 5, 500)

    potentials = [
        ('Harmonic: x²/2', lambda x: 0.5 * x**2),
        ('Quartic: x⁴', lambda x: x**4),
        ('Sextic: x⁶', lambda x: x**6),
        ('Double Well: (x²-1)²', lambda x: (x**2 - 1)**2),
        ('Morse: (1-e^{-x})²', lambda x: (1 - np.exp(-x))**2 * 5),
        ('Asymmetric: x²/2 + x³/10', lambda x: 0.5*x**2 + 0.1*x**3),
    ]

    for ax, (name, V_func) in zip(axes.flatten(), potentials):
        solver = QuantumEigenSolver(x_min=-5, x_max=5, nx=500)
        solver.set_potential(V_func)
        E, psi = solver.solve(n_states=5)

        # Plot potential and eigenstates
        V = V_func(x)
        ax.plot(x, V, 'k-', linewidth=2)

        for n in range(min(4, len(E))):
            if E[n] < V.max():
                scale = 0.3 * (E[1] - E[0]) if len(E) > 1 else 1
                ax.fill_between(x, E[n], E[n] + scale * psi[n]**2 / np.max(psi[n]**2),
                               alpha=0.5, label=f'E_{n}={E[n]:.2f}')
                ax.axhline(E[n], color=f'C{n}', linestyle='--', alpha=0.5)

        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('E')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.5, min(10, V.max()*1.2))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('anharmonic_oscillators.png', dpi=150)
    plt.show()

anharmonic_oscillator_study()
```

---

## Quantum Mechanics Connection

### Spectroscopic Applications

The energy level spacing determines absorption/emission spectra:
- Harmonic oscillator: Equal spacing $$\Delta E = \hbar\omega$$
- Anharmonic: Non-equal spacing reveals molecular structure
- Double well: Tunneling splitting reveals barrier properties

---

## Summary

### Key Methods

| Method | Pros | Cons |
|--------|------|------|
| Matrix diagonalization | Simple, all states at once | Memory scales as N² |
| Shooting | Very accurate | One state at a time |
| Variational | Physical insight | Only upper bound |
| Sparse methods | Large systems | Technical complexity |

### Key Equations

$$\boxed{\hat{H}\psi_n = E_n\psi_n}$$

$$\boxed{E_0 \leq \frac{\langle\phi|\hat{H}|\phi\rangle}{\langle\phi|\phi\rangle}}$$

---

## Daily Checklist

- [ ] Implemented matrix diagonalization method
- [ ] Compared with exact harmonic oscillator results
- [ ] Applied shooting method
- [ ] Used variational principle
- [ ] Studied anharmonic potentials
- [ ] Completed practice problems
