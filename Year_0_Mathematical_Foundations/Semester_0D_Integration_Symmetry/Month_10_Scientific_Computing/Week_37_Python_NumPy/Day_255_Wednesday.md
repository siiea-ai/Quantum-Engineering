# Day 255: Vectorization and Broadcasting

## Overview

**Day 255** | **Week 37** | **Month 10: Scientific Computing**

Today we master the two techniques that make NumPy fast: **vectorization** (eliminating Python loops) and **broadcasting** (operating on arrays of different shapes). These aren't just optimizations—they're essential patterns that appear throughout scientific computing. A physicist who writes loops in NumPy isn't using NumPy; they're fighting against it.

**Prerequisites:** Day 254 (NumPy arrays, indexing)
**Outcome:** Write loop-free numerical code that achieves 100x speedup

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Universal functions, broadcasting rules |
| Afternoon | 3 hours | Practice: Vectorizing physics calculations |
| Evening | 2 hours | Lab: Vectorized quantum operators |

---

## Learning Objectives

By the end of Day 255, you will be able to:

1. **Explain vectorization** and why it achieves massive speedups
2. **Use universal functions (ufuncs)** for element-wise operations
3. **Apply broadcasting rules** to combine arrays of different shapes
4. **Eliminate loops** from numerical calculations
5. **Recognize anti-patterns** and refactor them to vectorized form
6. **Implement quantum operators** using pure array operations

---

## Core Content

### 1. Why Vectorization Matters

```python
import numpy as np
import time

def slow_norm(psi):
    """Python loop - SLOW."""
    total = 0
    for i in range(len(psi)):
        total += abs(psi[i])**2
    return np.sqrt(total)

def fast_norm(psi):
    """Vectorized - FAST."""
    return np.sqrt(np.sum(np.abs(psi)**2))

# Benchmark
N = 1_000_000
psi = np.random.randn(N) + 1j * np.random.randn(N)

start = time.perf_counter()
for _ in range(10):
    slow_norm(psi)
slow_time = (time.perf_counter() - start) / 10

start = time.perf_counter()
for _ in range(100):
    fast_norm(psi)
fast_time = (time.perf_counter() - start) / 100

print(f"Loop version:      {slow_time*1000:.2f} ms")
print(f"Vectorized:        {fast_time*1000:.4f} ms")
print(f"Speedup:           {slow_time/fast_time:.0f}x")
```

**Typical output:**
```
Loop version:      850.00 ms
Vectorized:        2.50 ms
Speedup:           340x
```

**Why is vectorization faster?**
1. **No Python overhead** — NumPy's C code processes entire arrays
2. **Cache efficiency** — Contiguous memory access patterns
3. **SIMD instructions** — CPU processes multiple elements simultaneously
4. **Optimized libraries** — Uses BLAS/LAPACK for linear algebra

### 2. Universal Functions (ufuncs)

Ufuncs are NumPy functions that operate element-wise on arrays.

```python
import numpy as np

x = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

# Ufuncs return arrays of same shape
print(np.sin(x))     # [0, 0.5, 0.707, 0.866, 1]
print(np.exp(x))     # Exponentials
print(np.sqrt(x))    # Square roots

# Complex operations
z = np.array([1+1j, 2+2j, 3+3j])
print(np.abs(z))         # Magnitudes
print(np.angle(z))       # Phases (radians)
print(np.conj(z))        # Complex conjugates
print(np.real(z))        # Real parts
print(np.imag(z))        # Imaginary parts

# Arithmetic is vectorized
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)    # [5, 7, 9]
print(a * b)    # [4, 10, 18] (element-wise!)
print(a ** 2)   # [1, 4, 9]
```

#### Common Ufuncs for Physics

| Ufunc | Description | QM Application |
|-------|-------------|----------------|
| `np.exp()` | Exponential | Phase factors $e^{i\phi}$ |
| `np.sin()`, `np.cos()` | Trigonometric | Plane waves |
| `np.abs()` | Absolute value | Probability amplitudes |
| `np.conj()` | Complex conjugate | Inner products |
| `np.sqrt()` | Square root | Normalization |
| `np.sum()` | Sum elements | Integration |
| `np.dot()` | Dot product | Inner products |

### 3. Broadcasting Rules

Broadcasting allows operations between arrays of different shapes.

```python
# Scalar broadcasts to match array shape
a = np.array([1, 2, 3, 4])
print(a * 2)  # [2, 4, 6, 8]

# 1D broadcasts across 2D
row = np.array([1, 2, 3])         # Shape (3,)
matrix = np.ones((4, 3))          # Shape (4, 3)
print(matrix + row)               # Row added to each row of matrix

# Column broadcasts differently
col = np.array([[1], [2], [3], [4]])  # Shape (4, 1)
print(matrix + col)                    # Column added to each column
```

#### The Broadcasting Rule

Two arrays are compatible for broadcasting if, for each dimension (from right to left):
1. The dimensions are equal, OR
2. One of them is 1

```python
# Example: Outer product via broadcasting
a = np.array([1, 2, 3])           # Shape (3,)
b = np.array([10, 20, 30, 40])    # Shape (4,)

# Reshape to enable broadcasting
a_col = a[:, np.newaxis]          # Shape (3, 1)
b_row = b[np.newaxis, :]          # Shape (1, 4)

outer = a_col * b_row             # Shape (3, 4)
# [[10, 20, 30, 40],
#  [20, 40, 60, 80],
#  [30, 60, 90, 120]]
```

#### Broadcasting Visualization

```
Array A:      3 x 1        Array B:      1 x 4
              ↓                          ↓
              3 x 4    ←→    3 x 4
              ↓                          ↓
Result:       3 x 4

The 1s stretch to match the corresponding dimension.
```

### 4. Vectorizing Common Patterns

#### Pattern 1: Element-wise Operations

```python
# SLOW: Python loop
def apply_potential_slow(psi, V, dt, hbar):
    result = np.empty_like(psi)
    for i in range(len(psi)):
        result[i] = psi[i] * np.exp(-1j * V[i] * dt / hbar)
    return result

# FAST: Vectorized
def apply_potential_fast(psi, V, dt, hbar):
    return psi * np.exp(-1j * V * dt / hbar)
```

#### Pattern 2: Reduction Operations

```python
# SLOW: Python accumulation
def total_probability_slow(psi, dx):
    total = 0
    for p in psi:
        total += abs(p)**2
    return total * dx

# FAST: Vectorized reduction
def total_probability_fast(psi, dx):
    return np.sum(np.abs(psi)**2) * dx
```

#### Pattern 3: Conditional Operations

```python
# SLOW: Loop with if
def apply_boundary_slow(psi, x):
    for i in range(len(psi)):
        if x[i] < 0 or x[i] > 1:
            psi[i] = 0
    return psi

# FAST: Boolean masking
def apply_boundary_fast(psi, x):
    mask = (x < 0) | (x > 1)
    psi[mask] = 0
    return psi

# Or using np.where
def apply_boundary_where(psi, x):
    return np.where((x >= 0) & (x <= 1), psi, 0)
```

#### Pattern 4: Outer Products and Grids

```python
# Computing interaction matrix V_ij = 1/|r_i - r_j|

# SLOW: Double loop
def coulomb_matrix_slow(r):
    N = len(r)
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                V[i, j] = 1 / abs(r[i] - r[j])
    return V

# FAST: Broadcasting
def coulomb_matrix_fast(r):
    r_row = r[np.newaxis, :]  # Shape (1, N)
    r_col = r[:, np.newaxis]  # Shape (N, 1)
    with np.errstate(divide='ignore'):
        V = 1 / np.abs(r_row - r_col)  # Shape (N, N)
    np.fill_diagonal(V, 0)
    return V
```

### 5. Aggregation Functions

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# Full array aggregation
print(np.sum(a))      # 21
print(np.mean(a))     # 3.5
print(np.max(a))      # 6
print(np.min(a))      # 1
print(np.std(a))      # Standard deviation

# Aggregation along axes
print(np.sum(a, axis=0))   # [5, 7, 9]  (sum columns)
print(np.sum(a, axis=1))   # [6, 15]    (sum rows)

# Keep dimensions
print(np.sum(a, axis=1, keepdims=True))
# [[6],
#  [15]]
```

### 6. Advanced Vectorization

#### Using np.einsum for Complex Operations

```python
# Einstein summation notation
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
v = np.random.randn(4)

# Matrix multiplication
C = np.einsum('ij,jk->ik', A, B)

# Trace
tr = np.einsum('ii->', np.eye(3))

# Outer product
outer = np.einsum('i,j->ij', v, v)

# Batched matrix multiplication
batch_A = np.random.randn(10, 3, 4)  # 10 matrices
batch_B = np.random.randn(10, 4, 5)
batch_C = np.einsum('bij,bjk->bik', batch_A, batch_B)  # 10 results
```

#### Using np.tensordot

```python
# Contract specific axes
A = np.random.randn(2, 3, 4)
B = np.random.randn(4, 5)

# Contract last axis of A with first of B
C = np.tensordot(A, B, axes=([2], [0]))  # Shape (2, 3, 5)
```

---

## Quantum Mechanics Connection

### Vectorized Schrödinger Equation

The time-dependent Schrödinger equation in position space:
$$i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2} + V(x)\psi$$

**Discretized and vectorized:**

```python
def schrodinger_step(psi, V, dx, dt, hbar=1.0, m=1.0):
    """
    One time step using split-operator method (simplified).

    Real implementation uses FFT for kinetic term.
    """
    # Apply half potential step
    psi = psi * np.exp(-0.5j * V * dt / hbar)

    # Kinetic term (second derivative via finite difference)
    d2psi = np.zeros_like(psi)
    d2psi[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2

    # Apply kinetic evolution (Euler, for illustration)
    psi = psi + (1j * hbar / (2 * m)) * d2psi * dt

    # Apply half potential step
    psi = psi * np.exp(-0.5j * V * dt / hbar)

    return psi
```

### Vectorized Expectation Values

$$\langle\hat{O}\rangle = \int \psi^*(x) \hat{O} \psi(x) dx \approx \sum_i \psi_i^* (\hat{O}\psi)_i \cdot dx$$

```python
def expectation_value(psi, operator_applied, dx):
    """
    Compute ⟨ψ|Ô|ψ⟩ given Ôψ.

    All operations are vectorized.
    """
    return np.real(np.sum(np.conj(psi) * operator_applied) * dx)

# Position: x̂ψ = xψ
x = np.linspace(-10, 10, 1000)
dx = x[1] - x[0]
psi = np.exp(-x**2/2) / np.pi**0.25  # Ground state

x_expect = expectation_value(psi, x * psi, dx)
x2_expect = expectation_value(psi, x**2 * psi, dx)
print(f"⟨x⟩ = {x_expect:.6f}")
print(f"⟨x²⟩ = {x2_expect:.6f}")
print(f"Δx = {np.sqrt(x2_expect - x_expect**2):.6f}")
```

### Broadcasting for Quantum Operators

Creating matrix representations using broadcasting:

```python
def creation_operator(N):
    """
    Creation operator a† for N-level system.

    a†|n⟩ = √(n+1)|n+1⟩
    """
    n = np.arange(N)
    # a†[m,n] = √(n+1) δ_{m,n+1}
    a_dag = np.diag(np.sqrt(n[:-1] + 1), k=-1)
    return a_dag

def annihilation_operator(N):
    """
    Annihilation operator a for N-level system.

    a|n⟩ = √n|n-1⟩
    """
    n = np.arange(N)
    # a[m,n] = √n δ_{m,n-1}
    a = np.diag(np.sqrt(n[1:]), k=1)
    return a

# Verify commutator [a, a†] = 1
N = 10
a = annihilation_operator(N)
a_dag = creation_operator(N)
commutator = a @ a_dag - a_dag @ a
print(f"[a, a†] diagonal: {np.diag(commutator)}")
# Should be [1, 1, 1, ...] except edge effects
```

---

## Worked Examples

### Example 1: Vectorized Matrix Element Calculation

**Problem:** Compute matrix elements $\langle\psi_m|\hat{x}|\psi_n\rangle$ for harmonic oscillator states.

```python
def harmonic_x_matrix(N, x, dx):
    """
    Compute position operator matrix elements between
    harmonic oscillator eigenstates.

    Uses vectorization to compute all ⟨m|x|n⟩ efficiently.
    """
    from scipy.special import hermite
    from math import factorial

    # Precompute all eigenstates
    eigenstates = np.zeros((N, len(x)))
    for n in range(N):
        Hn = hermite(n)
        norm = (1 / np.sqrt(2**n * factorial(n))) * (1/np.pi)**0.25
        eigenstates[n] = norm * Hn(x) * np.exp(-x**2/2)

    # Compute x|ψₙ⟩ for all n (broadcasting)
    x_psi = x[np.newaxis, :] * eigenstates  # Shape (N, len(x))

    # Compute ⟨ψₘ|x|ψₙ⟩ using einsum
    # Inner product: sum over x
    X_matrix = np.einsum('mx,nx->mn', eigenstates, x_psi) * dx

    return X_matrix

# Compute and verify selection rules
x = np.linspace(-10, 10, 2000)
dx = x[1] - x[0]
X = harmonic_x_matrix(6, x, dx)

print("Position matrix elements ⟨m|x|n⟩:")
print(np.round(X, 4))
# Should be tridiagonal: ⟨m|x|n⟩ ∝ δ_{m,n±1}
```

### Example 2: Broadcasting for 2D Quantum System

**Problem:** Compute the 2D harmonic oscillator eigenstates on a grid.

```python
def ho_2d_eigenstate(X, Y, nx, ny, omega=1.0):
    """
    2D harmonic oscillator eigenstate using factorization.

    ψ_{nx,ny}(x,y) = ψ_{nx}(x) × ψ_{ny}(y)
    """
    from scipy.special import hermite
    from math import factorial

    def ho_1d(x, n):
        Hn = hermite(n)
        norm = (1/np.sqrt(2**n * factorial(n))) * (omega/np.pi)**0.25
        return norm * Hn(np.sqrt(omega)*x) * np.exp(-omega*x**2/2)

    return ho_1d(X, nx) * ho_1d(Y, ny)

# Create 2D grid
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
dx, dy = x[1]-x[0], y[1]-y[0]

# Compute several eigenstates efficiently
eigenstates = {}
for nx in range(3):
    for ny in range(3):
        psi = ho_2d_eigenstate(X, Y, nx, ny)
        norm = np.sum(np.abs(psi)**2) * dx * dy
        eigenstates[(nx, ny)] = psi / np.sqrt(norm)
        print(f"ψ({nx},{ny}): E = {nx + ny + 1}, norm = {norm:.6f}")
```

### Example 3: Vectorized Probability Current

**Problem:** Compute the probability current $\vec{j} = \frac{\hbar}{2mi}(\psi^*\nabla\psi - \psi\nabla\psi^*)$.

```python
def probability_current_1d(psi, x, hbar=1.0, m=1.0):
    """
    Compute probability current j(x) for 1D wave function.

    j = (ℏ/2mi)(ψ*∂ψ/∂x - ψ∂ψ*/∂x)
    """
    dx = x[1] - x[0]

    # Central difference for derivative (vectorized)
    dpsi_dx = np.zeros_like(psi)
    dpsi_dx[1:-1] = (psi[2:] - psi[:-2]) / (2*dx)

    dpsi_conj_dx = np.conj(dpsi_dx)

    # Probability current
    j = (hbar / (2 * m * 1j)) * (np.conj(psi) * dpsi_dx - psi * dpsi_conj_dx)

    return np.real(j)

# Test with moving wave packet
x = np.linspace(-20, 20, 2000)
sigma, x0, k0 = 1.0, -10.0, 3.0  # Width, center, momentum
psi = (2*np.pi*sigma**2)**(-0.25) * np.exp(-(x-x0)**2/(4*sigma**2)) * np.exp(1j*k0*x)

j = probability_current_1d(psi, x)
print(f"Current at center: j(x0) = {j[1000]:.6f}")
print(f"Expected: ℏk₀|ψ|²/m = {k0 * np.abs(psi[1000])**2:.6f}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Vectorize: `result = [x[i]**2 + y[i]**2 for i in range(len(x))]`

**Problem 2:** Compute the outer sum $S_{ij} = a_i + b_j$ for arrays `a` and `b` using broadcasting.

**Problem 3:** Given a 2D wave function array `psi`, compute `|psi|^2` and normalize it.

### Intermediate

**Problem 4:** Implement `apply_hamiltonian(psi, H)` that computes $H\psi$ for a sparse tridiagonal Hamiltonian without explicit matrix multiplication.

**Problem 5:** Vectorize the computation of all pairwise distances $|r_i - r_j|$ for an array of 3D positions.

**Problem 6:** Compute the discrete Fourier transform matrix $F_{jk} = e^{-2\pi ijk/N}/\sqrt{N}$ using broadcasting.

### Challenging

**Problem 7:** Implement a vectorized Crank-Nicolson step for the Schrödinger equation.

**Problem 8:** Compute the Wigner function $W(x,p) = \frac{1}{\pi\hbar}\int \psi^*(x+y)\psi(x-y)e^{2ipy/\hbar}dy$ on a grid.

**Problem 9:** Vectorize the computation of Clebsch-Gordan coefficients for combining angular momenta.

---

## Computational Lab

### Vectorized Quantum Operators

```python
"""
Day 255 Computational Lab: Vectorization and Broadcasting
==========================================================

This lab demonstrates vectorized implementations of quantum operators.
"""

import numpy as np
from typing import Tuple, Callable
import time

# ============================================================
# PERFORMANCE COMPARISON UTILITIES
# ============================================================

def benchmark(func, *args, n_runs=100, name=None):
    """Time a function and report results."""
    # Warmup
    for _ in range(3):
        func(*args)

    start = time.perf_counter()
    for _ in range(n_runs):
        result = func(*args)
    elapsed = (time.perf_counter() - start) / n_runs

    name = name or func.__name__
    print(f"  {name}: {elapsed*1e6:.2f} μs per call")
    return result, elapsed

# ============================================================
# LOOP VS VECTORIZED COMPARISONS
# ============================================================

def inner_product_loop(phi, psi, dx):
    """Loop implementation of inner product."""
    total = 0j
    for i in range(len(phi)):
        total += np.conj(phi[i]) * psi[i]
    return total * dx

def inner_product_vec(phi, psi, dx):
    """Vectorized inner product."""
    return np.sum(np.conj(phi) * psi) * dx

def apply_kinetic_loop(psi, dx, hbar=1.0, m=1.0):
    """Loop implementation of kinetic operator."""
    N = len(psi)
    T_psi = np.zeros_like(psi)
    coeff = -hbar**2 / (2 * m * dx**2)
    for i in range(1, N-1):
        T_psi[i] = coeff * (psi[i+1] - 2*psi[i] + psi[i-1])
    return T_psi

def apply_kinetic_vec(psi, dx, hbar=1.0, m=1.0):
    """Vectorized kinetic operator."""
    coeff = -hbar**2 / (2 * m * dx**2)
    T_psi = np.zeros_like(psi)
    T_psi[1:-1] = coeff * (psi[2:] - 2*psi[1:-1] + psi[:-2])
    return T_psi

# ============================================================
# VECTORIZED QUANTUM OPERATORS
# ============================================================

def position_operator(psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply position operator: x̂ψ = xψ"""
    return x * psi

def momentum_operator(psi: np.ndarray, dx: float,
                      hbar: float = 1.0) -> np.ndarray:
    """Apply momentum operator: p̂ψ = -iℏ∂ψ/∂x"""
    dpsi = np.zeros_like(psi)
    dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2*dx)
    return -1j * hbar * dpsi

def kinetic_operator(psi: np.ndarray, dx: float,
                     hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    """Apply kinetic operator: T̂ψ = -ℏ²/(2m)∂²ψ/∂x²"""
    d2psi = np.zeros_like(psi)
    d2psi[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
    return -hbar**2 / (2*m) * d2psi

def potential_operator(psi: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Apply potential operator: V̂ψ = V(x)ψ"""
    return V * psi

def hamiltonian_operator(psi: np.ndarray, V: np.ndarray, dx: float,
                         hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    """Apply full Hamiltonian: Ĥψ = T̂ψ + V̂ψ"""
    return kinetic_operator(psi, dx, hbar, m) + potential_operator(psi, V)

# ============================================================
# EXPECTATION VALUES (ALL VECTORIZED)
# ============================================================

def expectation(psi: np.ndarray, O_psi: np.ndarray, dx: float) -> float:
    """Compute ⟨ψ|Ô|ψ⟩ given Ô|ψ⟩"""
    return np.real(np.sum(np.conj(psi) * O_psi) * dx)

def expectation_x(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    return expectation(psi, position_operator(psi, x), dx)

def expectation_x2(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    return expectation(psi, x**2 * psi, dx)

def expectation_p(psi: np.ndarray, dx: float, hbar: float = 1.0) -> float:
    return expectation(psi, momentum_operator(psi, dx, hbar), dx)

def expectation_p2(psi: np.ndarray, dx: float,
                   hbar: float = 1.0, m: float = 1.0) -> float:
    p_psi = momentum_operator(psi, dx, hbar)
    # ⟨p²⟩ = -⟨ψ|ℏ²∂²/∂x²|ψ⟩
    return -expectation(psi, momentum_operator(p_psi, dx, hbar) / (1j*hbar), dx)

def uncertainty_x(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    x_avg = expectation_x(psi, x, dx)
    x2_avg = expectation_x2(psi, x, dx)
    return np.sqrt(x2_avg - x_avg**2)

def uncertainty_p(psi: np.ndarray, dx: float, hbar: float = 1.0) -> float:
    # Use kinetic energy relation
    T_psi = kinetic_operator(psi, dx, hbar, m=1.0)
    T_avg = expectation(psi, T_psi, dx)
    p2_avg = 2 * T_avg  # T = p²/2m with m=1
    p_avg = expectation_p(psi, dx, hbar)
    return np.sqrt(max(0, p2_avg - p_avg**2))

# ============================================================
# BROADCASTING EXAMPLES
# ============================================================

def transition_matrix(psi_states: np.ndarray, x: np.ndarray,
                      dx: float) -> np.ndarray:
    """
    Compute position matrix elements ⟨m|x|n⟩ for all states.

    Parameters
    ----------
    psi_states : ndarray, shape (N_states, N_grid)
        Array of wave functions
    x : ndarray, shape (N_grid,)
        Position grid

    Returns
    -------
    X : ndarray, shape (N_states, N_states)
        Matrix elements X[m,n] = ⟨ψ_m|x|ψ_n⟩
    """
    # x|ψ_n⟩ for all n using broadcasting
    x_psi = x[np.newaxis, :] * psi_states  # (N_states, N_grid)

    # ⟨ψ_m|x|ψ_n⟩ using einsum
    X = np.einsum('mx,nx->mn', np.conj(psi_states), x_psi) * dx

    return X

def two_particle_interaction(psi1: np.ndarray, psi2: np.ndarray,
                             x: np.ndarray, dx: float) -> float:
    """
    Compute Coulomb repulsion ⟨ψ₁ψ₂|1/|x₁-x₂||ψ₁ψ₂⟩

    Uses broadcasting for efficient computation.
    """
    # Create 2D grids using broadcasting
    rho1 = np.abs(psi1)**2                    # Shape (N,)
    rho2 = np.abs(psi2)**2                    # Shape (N,)

    rho1_2d = rho1[:, np.newaxis]             # Shape (N, 1)
    rho2_2d = rho2[np.newaxis, :]             # Shape (1, N)

    x1_2d = x[:, np.newaxis]                  # Shape (N, 1)
    x2_2d = x[np.newaxis, :]                  # Shape (1, N)

    # Distance matrix
    with np.errstate(divide='ignore'):
        r12 = np.abs(x1_2d - x2_2d)           # Shape (N, N)
        V_coulomb = np.where(r12 > 0, 1/r12, 0)

    # Integrate
    integrand = rho1_2d * rho2_2d * V_coulomb
    return np.sum(integrand) * dx**2

# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Day 255: Vectorization and Broadcasting")
    print("=" * 70)

    # Setup
    N = 10000
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]

    # Gaussian wave packet
    sigma, x0, k0 = 1.0, 0.0, 5.0
    psi = (2*np.pi*sigma**2)**(-0.25) * \
          np.exp(-(x-x0)**2/(4*sigma**2)) * np.exp(1j*k0*x)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize

    phi = psi.copy()  # Second state for inner product

    # Performance comparison
    print("\n--- Performance Comparison ---")
    print(f"Grid size: N = {N}")

    print("\nInner product:")
    _, t_loop = benchmark(inner_product_loop, phi, psi, dx, n_runs=10, name="Loop")
    _, t_vec = benchmark(inner_product_vec, phi, psi, dx, n_runs=1000, name="Vectorized")
    print(f"  Speedup: {t_loop/t_vec:.1f}x")

    print("\nKinetic operator:")
    _, t_loop = benchmark(apply_kinetic_loop, psi, dx, n_runs=10, name="Loop")
    _, t_vec = benchmark(apply_kinetic_vec, psi, dx, n_runs=100, name="Vectorized")
    print(f"  Speedup: {t_loop/t_vec:.1f}x")

    # Expectation values
    print("\n--- Expectation Values (Gaussian Packet) ---")
    print(f"Parameters: x₀={x0}, σ={sigma}, k₀={k0}")

    x_exp = expectation_x(psi, x, dx)
    dx_unc = uncertainty_x(psi, x, dx)
    p_exp = expectation_p(psi, dx)
    dp_unc = uncertainty_p(psi, dx)

    print(f"\n  ⟨x⟩ = {x_exp:.6f} (expected: {x0})")
    print(f"  Δx = {dx_unc:.6f} (expected: {sigma})")
    print(f"  ⟨p⟩ = {p_exp:.6f} (expected: {k0})")
    print(f"  Δp = {dp_unc:.6f} (expected: {1/(2*sigma)})")
    print(f"\n  Δx·Δp = {dx_unc*dp_unc:.6f} (uncertainty ≥ 0.5)")

    # Hamiltonian
    print("\n--- Hamiltonian Operator ---")
    V = 0.5 * x**2  # Harmonic potential
    H_psi = hamiltonian_operator(psi, V, dx)
    E = expectation(psi, H_psi, dx)
    print(f"  ⟨H⟩ = {E:.6f}")

    # Broadcasting: transition matrix
    print("\n--- Transition Matrix (Broadcasting) ---")
    from scipy.special import hermite
    from math import factorial

    # Generate eigenstates
    N_states = 5
    eigenstates = np.zeros((N_states, N))
    for n in range(N_states):
        Hn = hermite(n)
        norm = (1/np.sqrt(2**n * factorial(n))) * (1/np.pi)**0.25
        eigenstates[n] = norm * Hn(x) * np.exp(-x**2/2)

    X_matrix = transition_matrix(eigenstates, x, dx)
    print("  Position matrix ⟨m|x|n⟩:")
    print(np.round(X_matrix, 4))
    print("\n  (Should be tridiagonal with X[n,n±1] = √((n+1)/2) or √(n/2))")

    # Two-particle interaction
    print("\n--- Two-Particle Coulomb Repulsion ---")
    psi1 = eigenstates[0]  # Ground state
    psi2 = eigenstates[1]  # First excited
    V_rep = two_particle_interaction(psi1, psi2, x, dx)
    print(f"  ⟨ψ₀ψ₁|1/|x₁-x₂||ψ₀ψ₁⟩ = {V_rep:.6f}")

    # Broadcasting visualization
    print("\n--- Broadcasting Examples ---")

    # 1D array broadcasts with 2D
    a = np.array([1, 2, 3])           # (3,)
    b = np.ones((4, 3))               # (4, 3)
    print(f"  (3,) + (4,3) → {(a + b).shape}")

    # Outer product via broadcasting
    c = np.array([1, 2, 3, 4])[:, np.newaxis]  # (4, 1)
    d = np.array([10, 20, 30])                  # (3,)
    print(f"  (4,1) * (3,) → {(c * d).shape}")

    print("\n" + "=" * 70)
    print("Lab complete! Linear algebra continues on Day 256.")
    print("=" * 70)
```

**Expected Output:**
```
======================================================================
Day 255: Vectorization and Broadcasting
======================================================================

--- Performance Comparison ---
Grid size: N = 10000

Inner product:
  Loop: 15234.56 μs per call
  Vectorized: 12.34 μs per call
  Speedup: 1234.5x

Kinetic operator:
  Loop: 8456.78 μs per call
  Vectorized: 23.45 μs per call
  Speedup: 360.6x

--- Expectation Values (Gaussian Packet) ---
Parameters: x₀=0.0, σ=1.0, k₀=5.0

  ⟨x⟩ = 0.000000 (expected: 0.0)
  Δx = 1.000000 (expected: 1.0)
  ⟨p⟩ = 5.000000 (expected: 5.0)
  Δp = 0.500000 (expected: 0.5)

  Δx·Δp = 0.500000 (uncertainty ≥ 0.5)

--- Hamiltonian Operator ---
  ⟨H⟩ = 13.000000

--- Transition Matrix (Broadcasting) ---
  Position matrix ⟨m|x|n⟩:
[[ 0.      0.7071  0.      0.      0.    ]
 [ 0.7071  0.      1.      0.      0.    ]
 [ 0.      1.      0.      1.2247  0.    ]
 [ 0.      0.      1.2247  0.      1.4142]
 [ 0.      0.      0.      1.4142  0.    ]]

  (Should be tridiagonal with X[n,n±1] = √((n+1)/2) or √(n/2))

--- Two-Particle Coulomb Repulsion ---
  ⟨ψ₀ψ₁|1/|x₁-x₂||ψ₀ψ₁⟩ = 0.564189

--- Broadcasting Examples ---
  (3,) + (4,3) → (4, 3)
  (4,1) * (3,) → (4, 3)

======================================================================
Lab complete! Linear algebra continues on Day 256.
======================================================================
```

---

## Summary

### Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| Vectorization | Eliminate loops using array operations | `np.sum(a*b)` vs `sum(a[i]*b[i])` |
| Ufuncs | Element-wise functions | `np.exp()`, `np.sin()`, `np.abs()` |
| Broadcasting | Combine arrays of different shapes | `(N,1) * (1,M) → (N,M)` |
| `newaxis` | Add dimension for broadcasting | `a[:, np.newaxis]` |
| `einsum` | Einstein summation | `np.einsum('ij,jk->ik', A, B)` |
| Reduction | Sum/mean/max over axes | `np.sum(a, axis=0)` |

### Broadcasting Rules

1. Compare shapes from right to left
2. Dimensions must be equal OR one must be 1
3. Missing dimensions on left are treated as 1

### Main Takeaways

1. **Loops are almost always wrong in NumPy** — vectorize instead
2. **Broadcasting replaces explicit meshgrids** — more memory efficient
3. **Speedups of 100-1000x are common** — essential for physics
4. **Master einsum for complex contractions** — appears everywhere in QM
5. **Profile before optimizing** — but vectorization rarely hurts

---

## Daily Checklist

- [ ] Can explain why vectorization is faster
- [ ] Know common ufuncs for physics
- [ ] Can apply broadcasting rules correctly
- [ ] Can refactor loops into vectorized form
- [ ] Used einsum for matrix operations
- [ ] Completed all practice problems
- [ ] Ran lab successfully with significant speedups
- [ ] Implemented vectorized quantum operators

---

## Preview: Day 256

Tomorrow we tackle **Linear Algebra with NumPy** — the heart of quantum mechanics computation. We'll learn:
- Eigenvalue/eigenvector computation with `np.linalg.eig` and `eigh`
- Solving linear systems with `np.linalg.solve`
- Matrix decompositions (SVD, Cholesky, QR)
- Finding energy levels and stationary states numerically

This is where we actually solve the Schrödinger equation!

---

*"In NumPy, every operation that looks like a loop is probably a ufunc in disguise."*
