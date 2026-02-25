# Day 256: Linear Algebra with NumPy

## Overview

**Day 256** | **Week 37** | **Month 10: Scientific Computing**

Today we wield NumPy's linear algebra capabilities to solve quantum mechanical problems. The eigenvalue problem $H\psi = E\psi$ is the central equation of quantum mechanics, and today we solve it numerically. We'll find energy levels, compute stationary states, and verify the mathematical structure of quantum mechanics through computation.

**Prerequisites:** Days 253-255 (Python, NumPy basics), Linear algebra (Months 4-5)
**Outcome:** Solve eigenvalue problems and linear systems for quantum systems

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Eigenvalue solvers, matrix decompositions |
| Afternoon | 3 hours | Practice: Quantum eigenvalue problems |
| Evening | 2 hours | Lab: Particle in box and harmonic oscillator |

---

## Learning Objectives

By the end of Day 256, you will be able to:

1. **Solve eigenvalue problems** using `np.linalg.eig` and `np.linalg.eigh`
2. **Solve linear systems** with `np.linalg.solve` and understand when to use it
3. **Compute matrix decompositions** (SVD, QR, Cholesky, LU)
4. **Calculate matrix properties** (determinant, trace, rank, condition number)
5. **Find energy levels** of quantum systems numerically
6. **Verify orthonormality** and completeness of eigenstates
7. **Choose appropriate algorithms** for specific problem types

---

## Core Content

### 1. The Eigenvalue Problem

The time-independent Schrödinger equation is an eigenvalue problem:
$$\hat{H}\psi = E\psi$$

When discretized, this becomes:
$$\mathbf{H}\vec{\psi} = E\vec{\psi}$$

where $\mathbf{H}$ is an $N \times N$ matrix and $\vec{\psi}$ is an $N$-component vector.

```python
import numpy as np

# General eigenvalue solver
eigenvalues, eigenvectors = np.linalg.eig(A)

# For Hermitian matrices (quantum Hamiltonians)
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Key differences:
# - eig: general matrices, eigenvalues may be complex
# - eigh: Hermitian (H = H†), eigenvalues guaranteed real, faster
```

### 2. Eigenvalue Solvers

```python
import numpy as np

# Create a Hermitian matrix (Hamiltonian)
H = np.array([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
], dtype=float)

# Solve eigenvalue problem
energies, states = np.linalg.eigh(H)

print("Eigenvalues (energies):", energies)
print("\nEigenvectors (states):")
print(states)

# Properties of eigh:
# - Eigenvalues sorted in ascending order
# - Eigenvectors are columns: states[:, n] is the nth eigenstate
# - Eigenvectors are orthonormal: states.T @ states = I
```

#### Verification of Results

```python
# Verify eigenvalue equation: H @ ψ = E @ ψ
for n in range(len(energies)):
    E = energies[n]
    psi = states[:, n]

    # Hψ
    H_psi = H @ psi

    # Eψ
    E_psi = E * psi

    # Should be equal
    diff = np.linalg.norm(H_psi - E_psi)
    print(f"State {n}: ||Hψ - Eψ|| = {diff:.2e}")

# Verify orthonormality
overlap = states.T @ states
print("\nOverlap matrix (should be identity):")
print(np.round(overlap, 10))
```

### 3. Building Quantum Hamiltonians

```python
def particle_in_box_hamiltonian(N, L, hbar=1.0, m=1.0):
    """
    Construct Hamiltonian for particle in 1D box.

    Uses finite difference approximation for kinetic energy.
    """
    dx = L / (N + 1)  # Grid spacing (N interior points)

    # Kinetic energy: T = -ℏ²/(2m) d²/dx²
    # Finite difference: d²ψ/dx² ≈ (ψᵢ₊₁ - 2ψᵢ + ψᵢ₋₁)/dx²
    coeff = hbar**2 / (2 * m * dx**2)

    # Tridiagonal matrix
    diag = np.full(N, 2 * coeff)
    off_diag = np.full(N - 1, -coeff)

    H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    return H, dx

def harmonic_oscillator_hamiltonian(N, x_max, omega=1.0, hbar=1.0, m=1.0):
    """
    Construct Hamiltonian for 1D harmonic oscillator.

    H = -ℏ²/(2m) d²/dx² + ½mω²x²
    """
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]

    # Kinetic energy (tridiagonal)
    T_coeff = hbar**2 / (2 * m * dx**2)
    T = np.diag(np.full(N, 2*T_coeff)) + \
        np.diag(np.full(N-1, -T_coeff), 1) + \
        np.diag(np.full(N-1, -T_coeff), -1)

    # Potential energy (diagonal)
    V = np.diag(0.5 * m * omega**2 * x**2)

    return T + V, x
```

### 4. Solving Linear Systems

```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = np.linalg.solve(A, b)
print(f"Solution: x = {x}")  # [2, 3]

# Verify: A @ x = b
print(f"Verification: A @ x = {A @ x}")  # [9, 8]

# For multiple right-hand sides
B = np.array([[9, 8], [8, 7]]).T  # Columns are different b vectors
X = np.linalg.solve(A, B)
print(f"Multiple solutions:\n{X}")
```

#### When to Use solve vs inv

```python
# PREFER: np.linalg.solve(A, b)
# - More numerically stable
# - Faster (doesn't compute full inverse)
# - Uses LU decomposition internally

# AVOID: np.linalg.inv(A) @ b
# - Less stable
# - Slower
# - Only use if you need the inverse for multiple operations

# Example of instability
ill_conditioned = np.array([[1, 1], [1, 1.0001]])
b = np.array([2, 2.0001])

# solve handles this better than inv
x_solve = np.linalg.solve(ill_conditioned, b)
x_inv = np.linalg.inv(ill_conditioned) @ b

print(f"solve: {x_solve}")
print(f"inv:   {x_inv}")
print(f"Condition number: {np.linalg.cond(ill_conditioned):.0f}")
```

### 5. Matrix Decompositions

#### Singular Value Decomposition (SVD)

```python
A = np.random.randn(4, 3)

# Full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)
print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

# Reduced SVD (more efficient)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Reconstruct: A = U @ diag(S) @ Vt
A_reconstructed = U @ np.diag(S) @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Applications:
# - Low-rank approximations (truncate small singular values)
# - Pseudoinverse: A⁺ = V @ diag(1/S) @ U†
# - Condition number: cond(A) = S_max / S_min
```

#### QR Decomposition

```python
A = np.random.randn(4, 3)

Q, R = np.linalg.qr(A)
print(f"Q orthogonal: {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"R upper triangular: {np.allclose(R, np.triu(R))}")

# Application: Gram-Schmidt orthogonalization
# Q contains orthonormalized columns of A
```

#### Cholesky Decomposition

```python
# For positive definite matrices (like covariance matrices)
A = np.array([[4, 2], [2, 3]])

L = np.linalg.cholesky(A)  # A = L @ L†
print(f"L:\n{L}")
print(f"Reconstruction: {np.allclose(A, L @ L.T)}")

# Faster than LU for positive definite systems
```

### 6. Matrix Properties

```python
A = np.array([[1, 2], [3, 4]])

# Determinant
det = np.linalg.det(A)
print(f"det(A) = {det}")

# Trace
tr = np.trace(A)
print(f"tr(A) = {tr}")

# Rank
rank = np.linalg.matrix_rank(A)
print(f"rank(A) = {rank}")

# Norm
norm_fro = np.linalg.norm(A, 'fro')  # Frobenius
norm_2 = np.linalg.norm(A, 2)        # Spectral (largest singular value)
print(f"||A||_F = {norm_fro}, ||A||_2 = {norm_2}")

# Condition number
cond = np.linalg.cond(A)
print(f"cond(A) = {cond}")
```

---

## Quantum Mechanics Connection

### The Central Role of Eigenvalue Problems

In quantum mechanics:

| Mathematical Object | Physical Meaning |
|---------------------|------------------|
| Eigenvalues $E_n$ | Allowed energy levels |
| Eigenvectors $\psi_n$ | Stationary states |
| Degeneracy | Multiple states with same energy |
| Ground state | Smallest eigenvalue |
| Orthogonality | States are distinguishable |
| Completeness | Any state expandable in eigenbasis |

### Solving the Schrödinger Equation Numerically

```python
def solve_schrodinger(V, x, hbar=1.0, m=1.0):
    """
    Solve 1D time-independent Schrödinger equation.

    Returns energy levels and normalized wave functions.
    """
    N = len(x)
    dx = x[1] - x[0]

    # Build Hamiltonian
    T_coeff = hbar**2 / (2 * m * dx**2)
    H = np.diag(np.full(N, 2*T_coeff) + V) + \
        np.diag(np.full(N-1, -T_coeff), 1) + \
        np.diag(np.full(N-1, -T_coeff), -1)

    # Solve eigenvalue problem
    energies, states = np.linalg.eigh(H)

    # Normalize wave functions (including dx factor)
    for n in range(N):
        norm = np.sqrt(np.sum(np.abs(states[:, n])**2) * dx)
        states[:, n] /= norm

    return energies, states

# Example: Harmonic oscillator
x = np.linspace(-10, 10, 500)
V = 0.5 * x**2  # ω = 1

energies, states = solve_schrodinger(V, x)

print("First 5 energy levels:")
for n in range(5):
    print(f"  E_{n} = {energies[n]:.6f} (exact: {n + 0.5})")
```

### Verifying Quantum Properties

```python
def verify_orthonormality(states, dx, n_check=5):
    """Verify ⟨ψₘ|ψₙ⟩ = δₘₙ for first n_check states."""
    print("Orthonormality check:")
    for m in range(n_check):
        for n in range(m, n_check):
            overlap = np.sum(np.conj(states[:, m]) * states[:, n]) * dx
            expected = 1.0 if m == n else 0.0
            status = "✓" if abs(overlap - expected) < 1e-6 else "✗"
            print(f"  ⟨ψ_{m}|ψ_{n}⟩ = {overlap.real:+.6f} {status}")

def verify_completeness(states, dx, n_states=20):
    """Verify Σₙ |ψₙ⟩⟨ψₙ| ≈ I (resolution of identity)."""
    N = states.shape[0]
    identity_approx = np.zeros((N, N))

    for n in range(n_states):
        psi_n = states[:, n:n+1]  # Column vector
        identity_approx += psi_n @ psi_n.T * dx

    # Check diagonal elements (should be 1/dx for continuous normalization)
    print(f"\nCompleteness (using {n_states} states):")
    print(f"  Trace of Σ|ψₙ⟩⟨ψₙ| = {np.trace(identity_approx):.2f}")
    print(f"  Expected trace = {N * dx:.2f}")
```

---

## Worked Examples

### Example 1: Particle in a Box Energy Levels

**Problem:** Find the first 10 energy levels of a particle in a 1D box of length L=1.

```python
# Analytic formula: Eₙ = n²π²ℏ²/(2mL²), n = 1, 2, 3, ...

L, N = 1.0, 200
H, dx = particle_in_box_hamiltonian(N, L)

# Solve
energies, states = np.linalg.eigh(H)

# Compare with analytic
print("Particle in Box (L=1, ℏ=m=1):")
print("-" * 40)
print(f"{'n':>3} {'Numerical':>12} {'Analytic':>12} {'Error':>12}")
print("-" * 40)
for n in range(1, 11):
    E_numeric = energies[n-1]
    E_analytic = (n * np.pi)**2 / 2
    error = abs(E_numeric - E_analytic) / E_analytic
    print(f"{n:>3} {E_numeric:>12.6f} {E_analytic:>12.6f} {error:>12.2e}")
```

### Example 2: Finite Square Well

**Problem:** Find bound states of a finite square well.

```python
def finite_well_hamiltonian(N, x_max, V0, L):
    """
    Finite square well: V(x) = -V0 for |x| < L/2, else 0
    """
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]

    # Potential
    V = np.where(np.abs(x) < L/2, -V0, 0.0)

    # Kinetic + Potential
    T_coeff = 1 / (2 * dx**2)  # ℏ=m=1
    H = np.diag(2*T_coeff + V) + \
        np.diag(np.full(N-1, -T_coeff), 1) + \
        np.diag(np.full(N-1, -T_coeff), -1)

    return H, x, V

# Parameters
V0 = 10.0  # Well depth
L = 2.0   # Well width

H, x, V = finite_well_hamiltonian(500, 15.0, V0, L)
energies, states = np.linalg.eigh(H)

# Find bound states (E < 0)
bound_mask = energies < 0
bound_energies = energies[bound_mask]
bound_states = states[:, bound_mask]

print(f"Finite Well: V0={V0}, L={L}")
print(f"Number of bound states: {len(bound_energies)}")
for n, E in enumerate(bound_energies):
    print(f"  E_{n} = {E:.6f}")
```

### Example 3: Double Well and Tunneling

**Problem:** Analyze energy splitting in a double well potential.

```python
def double_well_hamiltonian(N, x_max, V0, a, barrier_width):
    """
    Double well: V(x) = V0 for |x| < barrier_width/2, else 0 in wells
    """
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]

    # Create double well
    V = np.zeros(N)
    # Central barrier
    V[np.abs(x) < barrier_width/2] = V0
    # Walls
    V[(x < -a) | (x > a)] = 100 * V0

    # Hamiltonian
    T_coeff = 1 / (2 * dx**2)
    H = np.diag(2*T_coeff + V) + \
        np.diag(np.full(N-1, -T_coeff), 1) + \
        np.diag(np.full(N-1, -T_coeff), -1)

    return H, x, V

# Analyze tunnel splitting vs barrier width
print("Tunnel splitting vs barrier width:")
print("-" * 40)
for barrier in [0.5, 1.0, 1.5, 2.0]:
    H, x, V = double_well_hamiltonian(500, 10.0, 5.0, 4.0, barrier)
    E, psi = np.linalg.eigh(H)

    # Ground and first excited (tunnel-split pair)
    E0, E1 = E[0], E[1]
    splitting = E1 - E0
    print(f"  Barrier width {barrier:.1f}: ΔE = {splitting:.6f}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Solve the eigenvalue problem for the matrix representing spin-1/2 in x-direction: $S_x = \frac{\hbar}{2}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.

**Problem 2:** Compute the determinant, trace, and eigenvalues of a 4×4 Hamiltonian matrix of your choice. Verify that det = product of eigenvalues and trace = sum of eigenvalues.

**Problem 3:** Create a 50×50 tridiagonal matrix with 2 on the diagonal and -1 on the off-diagonals. Find its 5 smallest eigenvalues.

### Intermediate

**Problem 4:** Implement a function that computes the energy levels of an anharmonic oscillator with $V(x) = \frac{1}{2}x^2 + \lambda x^4$ and study how they depend on $\lambda$.

**Problem 5:** Create the Hamiltonian for a 1D tight-binding model: $H_{ij} = -t$ for $|i-j|=1$, $H_{ii} = \epsilon_i$ (site energies). Find the band structure.

**Problem 6:** Implement the Gram-Schmidt process using QR decomposition to orthonormalize a set of wave functions.

### Challenging

**Problem 7:** Compute the lowest 5 eigenstates of the hydrogen atom radial equation (using effective potential $V(r) = -1/r + l(l+1)/(2r^2)$ for different $l$).

**Problem 8:** Implement the Lanczos algorithm to find the ground state energy of a large sparse Hamiltonian without fully diagonalizing it.

**Problem 9:** Solve the generalized eigenvalue problem $H\psi = E S \psi$ that arises when using non-orthogonal basis functions.

---

## Computational Lab

### Quantum Eigenvalue Problems

```python
"""
Day 256 Computational Lab: Linear Algebra for Quantum Mechanics
===============================================================

This lab solves eigenvalue problems for quantum systems.
"""

import numpy as np
from typing import Tuple, List, Callable
import time

# ============================================================
# HAMILTONIAN CONSTRUCTION
# ============================================================

def construct_hamiltonian(x: np.ndarray, V: np.ndarray,
                          hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    """
    Construct Hamiltonian matrix using finite difference.

    H = T + V = -ℏ²/(2m)d²/dx² + V(x)
    """
    N = len(x)
    dx = x[1] - x[0]

    # Kinetic energy coefficient
    T_coeff = hbar**2 / (2 * m * dx**2)

    # Build tridiagonal kinetic + diagonal potential
    H = np.diag(2*T_coeff + V) + \
        np.diag(np.full(N-1, -T_coeff), k=1) + \
        np.diag(np.full(N-1, -T_coeff), k=-1)

    return H

def solve_1d_schrodinger(x: np.ndarray, V: np.ndarray,
                         n_states: int = None,
                         hbar: float = 1.0, m: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve 1D time-independent Schrödinger equation.

    Returns
    -------
    energies : ndarray
        Energy eigenvalues (sorted)
    states : ndarray
        Normalized eigenstates (columns)
    """
    H = construct_hamiltonian(x, V, hbar, m)
    dx = x[1] - x[0]

    # Solve
    energies, states = np.linalg.eigh(H)

    # Normalize (including grid spacing)
    for n in range(states.shape[1]):
        norm = np.sqrt(np.sum(np.abs(states[:, n])**2) * dx)
        if norm > 0:
            states[:, n] /= norm

    if n_states is not None:
        return energies[:n_states], states[:, :n_states]
    return energies, states

# ============================================================
# STANDARD POTENTIALS
# ============================================================

def harmonic_potential(x: np.ndarray, omega: float = 1.0,
                       m: float = 1.0) -> np.ndarray:
    """V(x) = ½mω²x²"""
    return 0.5 * m * omega**2 * x**2

def infinite_well_potential(x: np.ndarray, L: float = 1.0,
                            V_wall: float = 1e10) -> np.ndarray:
    """Approximate infinite well with large finite walls."""
    V = np.zeros_like(x)
    V[(x < 0) | (x > L)] = V_wall
    return V

def finite_well_potential(x: np.ndarray, V0: float = 10.0,
                          L: float = 2.0) -> np.ndarray:
    """V(x) = -V0 for |x| < L/2, else 0"""
    return np.where(np.abs(x) < L/2, -V0, 0.0)

def morse_potential(x: np.ndarray, D: float = 10.0,
                    a: float = 1.0, x_e: float = 0.0) -> np.ndarray:
    """V(x) = D(1 - e^(-a(x-x_e)))² - D"""
    return D * (1 - np.exp(-a*(x - x_e)))**2 - D

def double_well_potential(x: np.ndarray, V0: float = 5.0,
                          barrier_width: float = 1.0) -> np.ndarray:
    """Symmetric double well with central barrier."""
    return V0 * np.exp(-x**2/0.1) - V0 * np.exp(-(x-2)**2) - V0 * np.exp(-(x+2)**2)

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def check_orthonormality(states: np.ndarray, dx: float,
                         n_check: int = 5) -> np.ndarray:
    """Compute overlap matrix ⟨ψ_m|ψ_n⟩."""
    n_check = min(n_check, states.shape[1])
    overlap = np.zeros((n_check, n_check))
    for m in range(n_check):
        for n in range(n_check):
            overlap[m, n] = np.sum(np.conj(states[:, m]) * states[:, n]) * dx
    return overlap

def expectation_value(psi: np.ndarray, operator_psi: np.ndarray,
                      dx: float) -> float:
    """Compute ⟨ψ|Ô|ψ⟩."""
    return np.real(np.sum(np.conj(psi) * operator_psi) * dx)

def energy_uncertainty(psi: np.ndarray, H: np.ndarray,
                       dx: float) -> Tuple[float, float]:
    """Compute ⟨H⟩ and ΔH for a state."""
    H_psi = H @ psi
    H2_psi = H @ H_psi

    E_avg = expectation_value(psi, H_psi, dx)
    E2_avg = expectation_value(psi, H2_psi, dx)

    delta_E = np.sqrt(max(0, E2_avg - E_avg**2))
    return E_avg, delta_E

def count_nodes(psi: np.ndarray) -> int:
    """Count number of nodes (sign changes) in wave function."""
    real_psi = np.real(psi)
    signs = np.sign(real_psi)
    signs[signs == 0] = 1  # Treat zeros as positive
    return np.sum(np.abs(np.diff(signs)) > 1)

# ============================================================
# MATRIX DECOMPOSITION UTILITIES
# ============================================================

def svd_analysis(A: np.ndarray) -> dict:
    """Analyze matrix using SVD."""
    U, S, Vt = np.linalg.svd(A)
    return {
        'singular_values': S,
        'rank': np.sum(S > 1e-10),
        'condition_number': S[0] / S[-1] if S[-1] > 0 else np.inf,
        'frobenius_norm': np.sqrt(np.sum(S**2)),
        'spectral_norm': S[0]
    }

def matrix_exponential(A: np.ndarray, method: str = 'eigendecomposition') -> np.ndarray:
    """
    Compute matrix exponential e^A.

    For Hermitian A (like -iHt/ℏ), use eigendecomposition.
    """
    if method == 'eigendecomposition':
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T.conj()
    else:
        from scipy.linalg import expm
        return expm(A)

# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Day 256: Linear Algebra for Quantum Mechanics")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Harmonic Oscillator
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. HARMONIC OSCILLATOR")
    print("=" * 70)

    x = np.linspace(-10, 10, 500)
    dx = x[1] - x[0]
    V_ho = harmonic_potential(x, omega=1.0)

    print("\nSolving Schrödinger equation...")
    start = time.perf_counter()
    E_ho, psi_ho = solve_1d_schrodinger(x, V_ho, n_states=10)
    elapsed = time.perf_counter() - start
    print(f"Solved in {elapsed*1000:.2f} ms")

    print("\nEnergy levels (ω = 1, ℏ = 1):")
    print("-" * 45)
    print(f"{'n':>3} {'Numerical':>12} {'Exact (n+½)':>12} {'Rel. Error':>12}")
    print("-" * 45)
    for n in range(8):
        E_exact = n + 0.5
        rel_err = abs(E_ho[n] - E_exact) / E_exact
        print(f"{n:>3} {E_ho[n]:>12.6f} {E_exact:>12.6f} {rel_err:>12.2e}")

    # Orthonormality
    print("\nOrthonormality check (first 5 states):")
    overlap = check_orthonormality(psi_ho, dx, 5)
    print(np.round(overlap, 6))

    # Node count
    print("\nNode count verification:")
    for n in range(5):
        nodes = count_nodes(psi_ho[:, n])
        print(f"  ψ_{n}: {nodes} nodes (expected: {n})")

    # --------------------------------------------------------
    # 2. Particle in Finite Well
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. FINITE SQUARE WELL")
    print("=" * 70)

    V0, L = 20.0, 2.0
    x_well = np.linspace(-5, 5, 500)
    V_well = finite_well_potential(x_well, V0, L)

    E_well, psi_well = solve_1d_schrodinger(x_well, V_well, n_states=20)

    # Count bound states (E < 0)
    bound_mask = E_well < 0
    n_bound = np.sum(bound_mask)
    print(f"\nWell depth V0 = {V0}, width L = {L}")
    print(f"Number of bound states: {n_bound}")

    print("\nBound state energies:")
    for n, E in enumerate(E_well[bound_mask]):
        print(f"  E_{n} = {E:.6f}")

    # --------------------------------------------------------
    # 3. Morse Potential (Molecular Vibrations)
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. MORSE POTENTIAL (Molecular Model)")
    print("=" * 70)

    D, a = 10.0, 1.0  # Dissociation energy, width
    x_morse = np.linspace(-2, 10, 600)
    V_morse = morse_potential(x_morse, D, a)

    E_morse, psi_morse = solve_1d_schrodinger(x_morse, V_morse, n_states=30)

    # Bound states (E < 0, since we shifted V)
    bound_morse = E_morse[E_morse < 0]
    print(f"\nMorse potential: D = {D}, a = {a}")
    print(f"Number of bound states: {len(bound_morse)}")

    # Analytic: E_n = ℏω(n+½) - (ℏω)²(n+½)²/(4D) where ω = a√(2D/m)
    omega = a * np.sqrt(2*D)
    print(f"\nComparison with analytic formula (ω = {omega:.4f}):")
    print("-" * 50)
    print(f"{'n':>3} {'Numerical':>12} {'Analytic':>12} {'Error':>12}")
    print("-" * 50)
    for n in range(min(5, len(bound_morse))):
        E_analytic = omega*(n+0.5) - (omega**2)*(n+0.5)**2/(4*D) - D
        error = bound_morse[n] - E_analytic
        print(f"{n:>3} {bound_morse[n]:>12.6f} {E_analytic:>12.6f} {error:>12.4f}")

    # --------------------------------------------------------
    # 4. Matrix Decomposition Demo
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. MATRIX DECOMPOSITION ANALYSIS")
    print("=" * 70)

    # Build a small Hamiltonian for analysis
    x_small = np.linspace(-5, 5, 50)
    V_small = harmonic_potential(x_small)
    H = construct_hamiltonian(x_small, V_small)

    print(f"\nHamiltonian size: {H.shape}")

    # Basic properties
    print(f"Trace(H) = {np.trace(H):.4f}")
    print(f"det(H) = {np.linalg.det(H):.4e}")
    print(f"Condition number = {np.linalg.cond(H):.4e}")

    # SVD analysis
    svd_info = svd_analysis(H)
    print(f"\nSVD Analysis:")
    print(f"  Rank = {svd_info['rank']}")
    print(f"  Largest singular value = {svd_info['singular_values'][0]:.4f}")
    print(f"  Smallest singular value = {svd_info['singular_values'][-1]:.4f}")
    print(f"  ||H||_F = {svd_info['frobenius_norm']:.4f}")

    # Verify trace = sum of eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"\nVerification:")
    print(f"  Trace(H) = {np.trace(H):.6f}")
    print(f"  Sum(eigenvalues) = {np.sum(eigenvalues):.6f}")

    # --------------------------------------------------------
    # 5. Time Evolution Preview
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. TIME EVOLUTION (Preview)")
    print("=" * 70)

    # Evolve a superposition state
    psi_0 = (psi_ho[:, 0] + psi_ho[:, 1]) / np.sqrt(2)  # |0⟩ + |1⟩

    print("\nInitial state: (|0⟩ + |1⟩)/√2")
    H_full = construct_hamiltonian(x, V_ho)
    E_init, dE_init = energy_uncertainty(psi_0, H_full, dx)
    print(f"  ⟨H⟩ = {E_init:.6f}")
    print(f"  ΔH = {dE_init:.6f}")

    # Time evolution: ψ(t) = e^(-iHt/ℏ)ψ(0)
    # In energy eigenbasis: ψ(t) = Σ c_n e^(-iE_n t/ℏ) |n⟩
    t = np.pi  # Half period for ω=1 oscillation

    # Expand in energy eigenbasis
    c_n = np.array([np.sum(np.conj(psi_ho[:, n]) * psi_0) * dx for n in range(10)])
    print(f"\nExpansion coefficients |c_n|²:")
    for n in range(5):
        print(f"  |c_{n}|² = {np.abs(c_n[n])**2:.6f}")

    # Evolve
    psi_t = np.zeros_like(psi_0)
    for n in range(10):
        psi_t += c_n[n] * np.exp(-1j * E_ho[n] * t) * psi_ho[:, n]

    print(f"\nAfter t = π (half oscillation period):")
    print(f"  Norm preserved: {np.sum(np.abs(psi_t)**2) * dx:.6f}")

    # Overlap with initial state
    overlap_t = np.abs(np.sum(np.conj(psi_0) * psi_t) * dx)**2
    print(f"  |⟨ψ(0)|ψ(t)⟩|² = {overlap_t:.6f}")

    print("\n" + "=" * 70)
    print("Lab complete! Random numbers and Monte Carlo on Day 257.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `np.linalg.eigh(H)` | Hermitian eigenvalue problem | (eigenvalues, eigenvectors) |
| `np.linalg.eig(A)` | General eigenvalue problem | (eigenvalues, eigenvectors) |
| `np.linalg.solve(A, b)` | Solve Ax = b | Solution x |
| `np.linalg.svd(A)` | Singular value decomposition | (U, S, Vt) |
| `np.linalg.qr(A)` | QR decomposition | (Q, R) |
| `np.linalg.det(A)` | Determinant | scalar |
| `np.linalg.cond(A)` | Condition number | scalar |

### Main Takeaways

1. **Use `eigh` for Hamiltonians** — guarantees real eigenvalues, faster
2. **Use `solve`, not `inv`** — more stable and efficient
3. **Eigenvalues from `eigh` are sorted** — ground state is first
4. **Eigenvectors are columns** — `states[:, n]` is the nth state
5. **Verify results** — check orthonormality, eigenvalue equation
6. **Condition number indicates stability** — large = ill-conditioned

---

## Daily Checklist

- [ ] Can solve eigenvalue problems with `eigh` and `eig`
- [ ] Know when to use `solve` vs `inv`
- [ ] Understand SVD and QR decompositions
- [ ] Can build quantum Hamiltonians numerically
- [ ] Verified orthonormality of computed eigenstates
- [ ] Computed energy levels matching analytic results
- [ ] Completed all practice problems
- [ ] Ran lab successfully

---

## Preview: Day 257

Tomorrow we explore **Random Numbers and Statistics** in NumPy. We'll learn:
- Generating random samples from various distributions
- Setting seeds for reproducibility
- Computing statistical quantities
- Introduction to Monte Carlo methods for quantum mechanics

This prepares us for variational Monte Carlo and path integral methods.

---

*"The eigenvalue problem is the Rosetta Stone of quantum mechanics—solve it and the physics reveals itself."*
