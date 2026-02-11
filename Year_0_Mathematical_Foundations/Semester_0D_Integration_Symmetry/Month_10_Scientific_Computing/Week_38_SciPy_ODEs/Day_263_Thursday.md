# Day 263: Advanced Linear Algebra with SciPy

## Overview

**Day 263** | **Week 38** | **Month 10: Scientific Computing**

Today we unlock `scipy.linalg`'s advanced capabilities beyond NumPy. Matrix exponentials enable quantum time evolution, specialized decompositions solve structured problems efficiently, and matrix functions generalize scalar operations to operators. These tools are essential for propagators, Lie theory applications, and quantum simulation.

**Prerequisites:** Days 260-262, linear algebra (Months 4-5)
**Outcome:** Compute matrix functions and specialized decompositions for quantum physics

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Matrix exponentials, functions, decompositions |
| Afternoon | 3 hours | Practice: Time evolution, Lie algebra |
| Evening | 2 hours | Lab: Quantum propagators and unitary dynamics |

---

## Learning Objectives

By the end of Day 263, you will be able to:

1. **Compute matrix exponentials** with `expm` for time evolution
2. **Apply matrix functions** (sqrt, log, sin, cos) to operators
3. **Use specialized decompositions** (Schur, polar, QZ)
4. **Solve structured linear systems** efficiently
5. **Implement quantum propagators** numerically
6. **Understand numerical stability** in matrix computations
7. **Connect to Lie theory** (exponential map, BCH formula)

---

## Core Content

### 1. Matrix Exponential

The matrix exponential is defined by the power series:
$$e^A = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$

```python
import numpy as np
from scipy import linalg

# Simple example: diagonal matrix
D = np.diag([1, 2, 3])
expD = linalg.expm(D)
print("exp(D) for diagonal:")
print(expD)
print(f"Diagonal elements: {np.diag(expD)}")
print(f"Expected: {np.exp([1, 2, 3])}")

# Non-diagonal matrix
A = np.array([[0, 1],
              [-1, 0]])  # Rotation generator
expA = linalg.expm(A)
print(f"\nexp([[0,1],[-1,0]]):")
print(expA)
print(f"This is rotation by 1 radian")
print(f"cos(1) = {np.cos(1):.6f}, sin(1) = {np.sin(1):.6f}")
```

### 2. Time Evolution Operator

The quantum propagator $U(t) = e^{-iHt/\hbar}$ evolves states:

```python
def time_evolution_operator(H, t, hbar=1.0):
    """
    Compute U(t) = exp(-iHt/ℏ).

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix (Hermitian)
    t : float
        Time
    hbar : float
        Reduced Planck constant

    Returns
    -------
    U : ndarray
        Unitary time evolution operator
    """
    return linalg.expm(-1j * H * t / hbar)

# Two-level system
H = np.array([[1, 0.5],
              [0.5, -1]])  # σz + 0.5σx

# Verify unitarity
t = 1.0
U = time_evolution_operator(H, t)
print("Time evolution operator U(t=1):")
print(U)
print(f"\nU†U = I: {np.allclose(U.conj().T @ U, np.eye(2))}")
print(f"det(U) = {np.linalg.det(U):.6f} (should be e^{-i*trace(H)*t})")

# Evolve state
psi_0 = np.array([1, 0], dtype=complex)  # |↑⟩
psi_t = U @ psi_0
print(f"\n|ψ(0)⟩ = |↑⟩")
print(f"|ψ(t)⟩ = {psi_t}")
print(f"Norm preserved: {np.abs(np.vdot(psi_t, psi_t)):.10f}")
```

### 3. Matrix Functions

```python
# Matrix square root
A = np.array([[4, 0],
              [0, 9]])
sqrtA = linalg.sqrtm(A)
print("Matrix square root:")
print(sqrtA)
print(f"sqrtA @ sqrtA:\n{sqrtA @ sqrtA}")

# Matrix logarithm (inverse of exponential)
B = np.array([[np.e, 0],
              [0, np.e**2]])
logB = linalg.logm(B)
print(f"\nMatrix logarithm:")
print(logB)

# Trigonometric functions
theta = np.pi/4
J = np.array([[0, -1],
              [1, 0]])  # Generator of rotations

cosJ = linalg.cosm(theta * J)
sinJ = linalg.sinm(theta * J)
print(f"\ncos(θJ) for θ=π/4:")
print(cosJ)
print(f"sin(θJ):")
print(sinJ)

# General matrix function
def f(x):
    return np.sqrt(x)

A = np.array([[4, 1],
              [0, 9]])
fA = linalg.funm(A, f)
print(f"\nGeneral function sqrt via funm:")
print(fA)
print(f"Verification (fA @ fA):\n{fA @ fA}")
```

### 4. Schur Decomposition

Every matrix can be written as $A = QTQ^H$ where $Q$ is unitary and $T$ is upper triangular:

```python
A = np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]])

T, Q = linalg.schur(A)
print("Schur decomposition A = QTQ†")
print(f"T (upper triangular):\n{T}")
print(f"Eigenvalues on diagonal: {np.diag(T)}")
print(f"Reconstruction error: {np.linalg.norm(A - Q @ T @ Q.conj().T):.2e}")

# For Hermitian matrices, Schur = eigendecomposition
H = np.array([[2, 1, 0],
              [1, 2, 1],
              [0, 1, 2]])

T_h, Q_h = linalg.schur(H)
print(f"\nHermitian Schur (diagonal T):\n{T_h}")
```

### 5. Polar Decomposition

Any matrix can be written as $A = UP$ where $U$ is unitary and $P$ is positive semidefinite:

```python
A = np.array([[1, 2],
              [3, 4]])

U, P = linalg.polar(A)
print("Polar decomposition A = UP")
print(f"U (unitary):\n{U}")
print(f"P (positive semidefinite):\n{P}")
print(f"U†U = I: {np.allclose(U.conj().T @ U, np.eye(2))}")
print(f"Reconstruction: {np.allclose(A, U @ P)}")

# For quantum channels: Kraus operators
```

### 6. Solving Linear Systems

```python
# Standard solve
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = linalg.solve(A, b)
print(f"Solve Ax = b: x = {x}")

# Hermitian positive definite (faster, more stable)
H = np.array([[4, 2], [2, 5]])
b = np.array([1, 2])
x = linalg.solve(H, b, assume_a='pos')
print(f"Positive definite solve: x = {x}")

# Banded matrices (tridiagonal Hamiltonians)
# Ab format: diagonals stored in rows
N = 5
ab = np.array([
    [0, -1, -1, -1, -1],  # Upper diagonal
    [2, 2, 2, 2, 2],       # Main diagonal
    [-1, -1, -1, -1, 0]    # Lower diagonal
])
b = np.ones(N)
x = linalg.solve_banded((1, 1), ab, b)
print(f"Banded solve: x = {x}")

# Triangular (from decompositions)
L = np.array([[1, 0, 0],
              [2, 1, 0],
              [3, 4, 1]])
b = np.array([1, 2, 3])
x = linalg.solve_triangular(L, b, lower=True)
print(f"Triangular solve: x = {x}")
```

### 7. Matrix Decompositions for Physics

```python
# LU decomposition
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]])
P, L, U = linalg.lu(A)
print("LU decomposition:")
print(f"L:\n{L}")
print(f"U:\n{U}")
print(f"PA = LU: {np.allclose(P @ A, L @ U)}")

# Cholesky (for positive definite, like covariance matrices)
C = np.array([[4, 2, 2],
              [2, 5, 1],
              [2, 1, 6]])
L_chol = linalg.cholesky(C, lower=True)
print(f"\nCholesky: C = LL†")
print(f"L:\n{L_chol}")
print(f"Reconstruction: {np.allclose(C, L_chol @ L_chol.T)}")

# SVD with full matrices
A = np.random.randn(4, 3)
U, s, Vh = linalg.svd(A)
print(f"\nSVD shapes: U={U.shape}, s={s.shape}, Vh={Vh.shape}")
print(f"Singular values: {s}")
```

---

## Quantum Mechanics Connection

### Quantum Propagation

```python
def propagate_state(H, psi_0, times, hbar=1.0):
    """
    Propagate quantum state through a sequence of times.

    Uses eigendecomposition for efficiency when evaluating at many times.
    """
    # Eigendecomposition: H = V @ diag(E) @ V†
    E, V = linalg.eigh(H)

    # Transform initial state to energy basis
    c_0 = V.conj().T @ psi_0

    states = []
    for t in times:
        # Evolution in energy basis: c_n(t) = c_n(0) exp(-iE_n t/ℏ)
        c_t = c_0 * np.exp(-1j * E * t / hbar)
        # Transform back
        psi_t = V @ c_t
        states.append(psi_t)

    return np.array(states).T  # Shape: (dim, n_times)

# Example: 3-level system
H = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]], dtype=float)

psi_0 = np.array([1, 0, 0], dtype=complex)
times = np.linspace(0, 10, 100)

states = propagate_state(H, psi_0, times)
populations = np.abs(states)**2

print("Population dynamics:")
print(f"  P_0(t=0) = {populations[0, 0]:.4f}")
print(f"  P_0(t=5) = {populations[0, 50]:.4f}")
print(f"  Total probability preserved: {np.allclose(np.sum(populations, axis=0), 1)}")
```

### Baker-Campbell-Hausdorff Formula

For non-commuting operators: $e^A e^B = e^{A+B+\frac{1}{2}[A,B]+...}$

```python
def bch_approximation(A, B, order=2):
    """
    Approximate log(exp(A) exp(B)) using BCH formula.

    To second order: A + B + [A,B]/2 + ...
    """
    commutator = A @ B - B @ A

    if order == 1:
        return A + B
    elif order == 2:
        return A + B + commutator / 2
    else:
        raise ValueError("Higher orders not implemented")

# Test with Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

A = 0.1 * sigma_x
B = 0.1 * sigma_z

# Exact
exp_A_exp_B = linalg.expm(A) @ linalg.expm(B)

# BCH approximations
C_1 = bch_approximation(A, B, order=1)
C_2 = bch_approximation(A, B, order=2)

print("BCH formula test:")
print(f"  ||exp(A)exp(B) - exp(A+B)||     = {np.linalg.norm(exp_A_exp_B - linalg.expm(C_1)):.6f}")
print(f"  ||exp(A)exp(B) - exp(BCH₂)||    = {np.linalg.norm(exp_A_exp_B - linalg.expm(C_2)):.6f}")
```

### Lie Algebra Generators

```python
def su2_generators():
    """SU(2) Lie algebra generators (Pauli matrices / 2)."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Generators: t_i = σ_i / 2
    return sigma_x/2, sigma_y/2, sigma_z/2

def rotation_su2(theta, axis):
    """
    SU(2) rotation by angle theta around axis.

    R = exp(-i θ n·σ/2)
    """
    t_x, t_y, t_z = su2_generators()

    # Generator for rotation around axis
    n = axis / np.linalg.norm(axis)
    generator = n[0]*t_x + n[1]*t_y + n[2]*t_z

    return linalg.expm(-1j * theta * generator)

# Rotation around z-axis
theta = np.pi / 2
R_z = rotation_su2(theta, [0, 0, 1])
print("Rotation by π/2 around z:")
print(R_z)
print(f"This maps |↑⟩ → e^(-iπ/4)|↑⟩, |↓⟩ → e^(iπ/4)|↓⟩")

# Verify: rotate |+x⟩ to |+y⟩
plus_x = np.array([1, 1]) / np.sqrt(2)
plus_y = np.array([1, 1j]) / np.sqrt(2)
rotated = R_z @ plus_x
print(f"\nRotated |+x⟩: {rotated}")
print(f"|+y⟩:         {plus_y}")
print(f"Match (up to phase): {np.allclose(np.abs(rotated), np.abs(plus_y))}")
```

### Density Matrix Evolution

```python
def liouville_superoperator(H, gamma=0.0, L=None):
    """
    Build Liouville superoperator for density matrix evolution.

    dρ/dt = -i[H, ρ] + γ(LρL† - ½{L†L, ρ})

    Vectorizes: dρ_vec/dt = L_super @ ρ_vec
    """
    dim = H.shape[0]
    dim_sq = dim**2

    # Commutator part: -i(H⊗I - I⊗H*)
    L_comm = -1j * (np.kron(H, np.eye(dim)) - np.kron(np.eye(dim), H.conj()))

    if gamma > 0 and L is not None:
        # Lindblad terms
        L_dag_L = L.conj().T @ L
        L_lindblad = gamma * (
            np.kron(L, L.conj()) -
            0.5 * np.kron(L_dag_L, np.eye(dim)) -
            0.5 * np.kron(np.eye(dim), L_dag_L.T)
        )
        return L_comm + L_lindblad

    return L_comm

# Example: decaying qubit
H = np.array([[1, 0], [0, -1]], dtype=complex)  # Energy splitting
L = np.array([[0, 1], [0, 0]], dtype=complex)   # Lowering operator

L_super = liouville_superoperator(H, gamma=0.1, L=L)

# Propagate density matrix
rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |↑⟩⟨↑|
rho_vec_0 = rho_0.flatten()

# Propagator
t = 5.0
propagator = linalg.expm(L_super * t)
rho_vec_t = propagator @ rho_vec_0
rho_t = rho_vec_t.reshape(2, 2)

print("Density matrix evolution with decay:")
print(f"ρ(0) = |↑⟩⟨↑|:")
print(rho_0)
print(f"\nρ(t=5) with γ=0.1:")
print(rho_t)
print(f"Trace: {np.trace(rho_t):.6f}")
print(f"P(↑) = {rho_t[0,0].real:.4f}, P(↓) = {rho_t[1,1].real:.4f}")
```

---

## Worked Examples

### Example 1: Trotter-Suzuki Decomposition

```python
def trotter_evolution(H1, H2, t, n_steps):
    """
    Approximate exp(-i(H1+H2)t) using Trotter formula.

    exp(-i(H1+H2)t) ≈ (exp(-iH1 dt) exp(-iH2 dt))^n
    """
    dt = t / n_steps
    U1 = linalg.expm(-1j * H1 * dt)
    U2 = linalg.expm(-1j * H2 * dt)

    U = np.eye(H1.shape[0], dtype=complex)
    for _ in range(n_steps):
        U = U1 @ U2 @ U

    return U

def trotter_suzuki_2nd(H1, H2, t, n_steps):
    """
    Second-order Trotter-Suzuki: exp(-iH1 dt/2) exp(-iH2 dt) exp(-iH1 dt/2)
    """
    dt = t / n_steps
    U1_half = linalg.expm(-1j * H1 * dt / 2)
    U2 = linalg.expm(-1j * H2 * dt)

    U = np.eye(H1.shape[0], dtype=complex)
    for _ in range(n_steps):
        U = U1_half @ U2 @ U1_half @ U

    return U

# Test: H = H1 + H2 where [H1, H2] ≠ 0
H1 = np.array([[1, 0], [0, -1]], dtype=float)  # σz
H2 = np.array([[0, 1], [1, 0]], dtype=float)   # σx

t = 2.0
U_exact = linalg.expm(-1j * (H1 + H2) * t)

print("Trotter-Suzuki convergence:")
print(f"{'n_steps':>8} {'1st order':>15} {'2nd order':>15}")
for n in [1, 10, 100, 1000]:
    U_1st = trotter_evolution(H1, H2, t, n)
    U_2nd = trotter_suzuki_2nd(H1, H2, t, n)

    error_1st = np.linalg.norm(U_1st - U_exact)
    error_2nd = np.linalg.norm(U_2nd - U_exact)

    print(f"{n:>8} {error_1st:>15.2e} {error_2nd:>15.2e}")
```

### Example 2: Matrix Function via Eigendecomposition

```python
def matrix_function_eig(A, f):
    """
    Compute f(A) using eigendecomposition.

    f(A) = V @ diag(f(λ)) @ V^(-1)
    """
    eigenvalues, V = np.linalg.eig(A)
    f_eigenvalues = f(eigenvalues)
    return V @ np.diag(f_eigenvalues) @ np.linalg.inv(V)

# Test: compute A^(1/2)
A = np.array([[4, 2], [1, 3]])

sqrt_A_eig = matrix_function_eig(A, np.sqrt)
sqrt_A_scipy = linalg.sqrtm(A)

print("Matrix square root comparison:")
print(f"Eigendecomposition:\n{sqrt_A_eig}")
print(f"SciPy sqrtm:\n{sqrt_A_scipy}")
print(f"Match: {np.allclose(sqrt_A_eig, sqrt_A_scipy)}")
print(f"Verification (sqrt(A))²:\n{sqrt_A_scipy @ sqrt_A_scipy}")
```

### Example 3: Quantum Gate Decomposition

```python
def decompose_single_qubit_gate(U):
    """
    Decompose single-qubit unitary into Euler angles.

    U = e^(iα) Rz(β) Ry(γ) Rz(δ)
    """
    # Extract global phase
    det = np.linalg.det(U)
    alpha = np.angle(det) / 2
    U_special = U * np.exp(-1j * alpha)  # Make det = 1

    # Extract angles from matrix elements
    # U = [[cos(γ/2)e^(-i(β+δ)/2), -sin(γ/2)e^(-i(β-δ)/2)],
    #      [sin(γ/2)e^(i(β-δ)/2),   cos(γ/2)e^(i(β+δ)/2)]]

    gamma = 2 * np.arccos(np.clip(np.abs(U_special[0, 0]), 0, 1))

    if np.abs(np.sin(gamma/2)) > 1e-10:
        beta_plus_delta = -np.angle(U_special[0, 0]) * 2
        beta_minus_delta = -np.angle(-U_special[0, 1]) * 2
        beta = (beta_plus_delta + beta_minus_delta) / 2
        delta = (beta_plus_delta - beta_minus_delta) / 2
    else:
        beta = -np.angle(U_special[0, 0]) * 2
        delta = 0

    return alpha, beta, gamma, delta

# Test with Hadamard gate
H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
alpha, beta, gamma, delta = decompose_single_qubit_gate(H_gate)

print("Hadamard gate decomposition:")
print(f"  α = {alpha:.4f}")
print(f"  β = {beta:.4f}")
print(f"  γ = {gamma:.4f}")
print(f"  δ = {delta:.4f}")

# Verify by reconstruction
def Rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]])

def Ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])

H_reconstructed = np.exp(1j*alpha) * Rz(beta) @ Ry(gamma) @ Rz(delta)
print(f"Reconstruction matches: {np.allclose(H_gate, H_reconstructed)}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Compute $e^{A}$ where $A = \begin{pmatrix} 0 & \theta \\ -\theta & 0 \end{pmatrix}$ and verify it equals a rotation matrix.

**Problem 2:** Find $\sqrt{A}$ for $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ and verify that $(\sqrt{A})^2 = A$.

**Problem 3:** Solve the tridiagonal system arising from discretizing $-d^2\psi/dx^2 = \lambda\psi$ with boundary conditions.

### Intermediate

**Problem 4:** Implement time evolution using eigendecomposition and compare performance with `expm` for various matrix sizes.

**Problem 5:** Compute the geometric phase acquired by a qubit undergoing cyclic evolution on the Bloch sphere.

**Problem 6:** Implement the Cayley transform $U = (I - iH)(I + iH)^{-1}$ for generating unitary matrices from Hermitian ones.

### Challenging

**Problem 7:** Implement the Suzuki fractal decomposition for higher-order Trotter formulas.

**Problem 8:** Compute the matrix logarithm of a unitary matrix and verify it gives a Hermitian generator.

**Problem 9:** Implement the KAK decomposition for two-qubit gates using Cartan's decomposition of SU(4).

---

## Computational Lab

```python
"""
Day 263 Lab: Advanced Linear Algebra for Quantum Physics
========================================================
"""

import numpy as np
from scipy import linalg
from typing import Callable

# [Full lab implementation with all demonstrations]

if __name__ == "__main__":
    print("=" * 70)
    print("Day 263: Advanced Linear Algebra")
    print("=" * 70)

    # Demonstrations...

    print("\n" + "=" * 70)
    print("Lab complete! Special functions continue on Day 264.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `linalg.expm(A)` | Matrix exponential | `U = linalg.expm(-1j*H*t)` |
| `linalg.sqrtm(A)` | Matrix square root | `sqrtA = linalg.sqrtm(A)` |
| `linalg.logm(A)` | Matrix logarithm | `logA = linalg.logm(A)` |
| `linalg.funm(A, f)` | General function | `fA = linalg.funm(A, np.sin)` |
| `linalg.schur(A)` | Schur decomposition | `T, Q = linalg.schur(A)` |
| `linalg.polar(A)` | Polar decomposition | `U, P = linalg.polar(A)` |

### Quantum Applications

| Operation | SciPy Function |
|-----------|---------------|
| Time evolution $U(t) = e^{-iHt}$ | `expm(-1j*H*t)` |
| Density matrix propagator | `expm(L_super*t)` |
| Unitary from Hermitian | `expm(1j*H)` |
| Gate decomposition | `schur`, eigendecomposition |

---

## Daily Checklist

- [ ] Can compute matrix exponentials with `expm`
- [ ] Understand matrix functions (sqrt, log, sin, cos)
- [ ] Know Schur and polar decompositions
- [ ] Can solve structured linear systems efficiently
- [ ] Implemented quantum propagators
- [ ] Understand Trotter-Suzuki decomposition
- [ ] Completed practice problems
- [ ] Ran lab successfully

---

## Preview: Day 264

Tomorrow we explore **Special Functions and FFT** with `scipy.special` and `scipy.fft`. We'll learn:
- Hermite polynomials for harmonic oscillator
- Spherical harmonics for angular momentum
- Bessel functions for cylindrical/spherical problems
- FFT for momentum space

Essential for analytical quantum mechanics!

---

*"The exponential map connects Lie algebras to Lie groups, infinitesimal to finite transformations."*
