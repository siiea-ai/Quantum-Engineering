# Day 254: NumPy Arrays — Creation and Indexing

## Overview

**Day 254** | **Week 37** | **Month 10: Scientific Computing**

Today we master NumPy arrays—the fundamental data structure for all scientific Python. NumPy's `ndarray` provides contiguous memory storage, element-wise operations, and sophisticated indexing that makes numerical computation fast and expressive. By day's end, you'll manipulate multi-dimensional arrays with the fluency required for quantum physics simulations.

**Prerequisites:** Day 253 (Python fundamentals), basic linear algebra
**Outcome:** Create, reshape, and index NumPy arrays efficiently

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Array creation, dtype, memory layout |
| Afternoon | 3 hours | Practice: Indexing, slicing, fancy indexing |
| Evening | 2 hours | Lab: Discretizing wave functions on grids |

---

## Learning Objectives

By the end of Day 254, you will be able to:

1. **Create arrays** using `np.array`, `zeros`, `ones`, `linspace`, `arange`, `meshgrid`
2. **Understand dtypes** and choose appropriate precision for physics applications
3. **Reshape and transpose** arrays without copying data
4. **Apply slicing syntax** including multi-dimensional and negative indices
5. **Use fancy indexing** with boolean masks and integer arrays
6. **Recognize views vs copies** to avoid subtle bugs
7. **Discretize continuous functions** onto numerical grids

---

## Core Content

### 1. NumPy Array Fundamentals

```python
import numpy as np

# The fundamental object: ndarray
# - Homogeneous (all elements same type)
# - Fixed size at creation
# - Contiguous memory (usually)
# - Fast: operations implemented in C

# Creating from Python lists
a = np.array([1, 2, 3, 4, 5])
print(f"Array: {a}")
print(f"Shape: {a.shape}")      # (5,)
print(f"Dtype: {a.dtype}")      # int64
print(f"Ndim: {a.ndim}")        # 1
print(f"Size: {a.size}")        # 5
print(f"Itemsize: {a.itemsize}") # 8 bytes
```

#### Multi-dimensional Arrays

```python
# 2D array (matrix)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(f"Shape: {matrix.shape}")  # (2, 3) = 2 rows, 3 columns

# 3D array (e.g., for time-dependent wave function)
tensor = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(f"Shape: {tensor.shape}")  # (2, 2, 2)
```

### 2. Array Creation Functions

```python
# Zeros and ones
zeros = np.zeros((3, 4))           # 3×4 matrix of zeros
ones = np.ones((2, 3, 4))          # 2×3×4 tensor of ones
empty = np.empty((5, 5))           # Uninitialized (fast but dangerous)

# Identity and diagonal
I = np.eye(3)                      # 3×3 identity matrix
diag = np.diag([1, 2, 3])          # Diagonal matrix
off_diag = np.diag([1, 1], k=1)    # Off-diagonal

# Ranges
a = np.arange(0, 10, 2)            # [0, 2, 4, 6, 8] - like range()
b = np.linspace(0, 1, 5)           # [0, 0.25, 0.5, 0.75, 1] - n evenly spaced

# For physics: spatial grids
x = np.linspace(-10, 10, 1000)     # 1000 points from -10 to 10
dx = x[1] - x[0]                   # Grid spacing

# Logarithmic spacing (for energies, frequencies)
energies = np.logspace(-2, 2, 50)  # 50 points from 0.01 to 100

# Grid creation for 2D/3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)           # 2D coordinate grids
# X[i,j] = x[j], Y[i,j] = y[i]
```

### 3. Data Types for Physics

```python
# Floating point precision
x_32 = np.array([1.0, 2.0], dtype=np.float32)   # Single precision (4 bytes)
x_64 = np.array([1.0, 2.0], dtype=np.float64)   # Double precision (8 bytes)

# Complex numbers (essential for quantum mechanics!)
psi = np.array([1+0j, 0+1j], dtype=np.complex128)  # Double complex
print(psi.real)   # Real parts
print(psi.imag)   # Imaginary parts
print(np.conj(psi))  # Complex conjugate

# Type conversion
a = np.array([1, 2, 3])
b = a.astype(np.float64)   # Convert int to float

# Common dtypes for physics:
# - np.float64: default for calculations (15-16 sig figs)
# - np.complex128: wave functions, amplitudes
# - np.float32: when memory is limited (GPU computing)
# - np.int64: quantum numbers, indices
```

### 4. Memory Layout and Strides

```python
# NumPy arrays are stored in contiguous memory
a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float64)

print(f"Data pointer: {a.ctypes.data}")
print(f"Strides: {a.strides}")  # (24, 8) = bytes to next row, next column

# C order (row-major, default) vs Fortran order (column-major)
c_order = np.array([[1, 2], [3, 4]], order='C')
f_order = np.array([[1, 2], [3, 4]], order='F')
print(f"C strides: {c_order.strides}")    # (16, 8)
print(f"F strides: {f_order.strides}")    # (8, 16)

# Why this matters: cache efficiency
# Access along contiguous memory is much faster
# For row-major: iterate over columns in inner loop
```

### 5. Indexing and Slicing

```python
a = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic indexing (returns scalar or view)
print(a[0])      # 0 (first element)
print(a[-1])     # 9 (last element)
print(a[2:5])    # [2, 3, 4] (slice)
print(a[::2])    # [0, 2, 4, 6, 8] (every 2nd)
print(a[::-1])   # [9, 8, 7, ...] (reversed)

# 2D indexing
m = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print(m[0, 0])      # 0 (element at row 0, col 0)
print(m[1, :])      # [4, 5, 6, 7] (row 1)
print(m[:, 2])      # [2, 6, 10] (column 2)
print(m[0:2, 1:3])  # [[1, 2], [5, 6]] (submatrix)
print(m[-1, -1])    # 11 (last element)

# Crucial: slices return VIEWS, not copies!
row = m[1, :]
row[0] = 100
print(m[1, 0])  # 100! Original modified!
```

### 6. Fancy Indexing

```python
a = np.array([10, 20, 30, 40, 50])

# Integer array indexing (returns COPY)
indices = np.array([0, 2, 4])
print(a[indices])  # [10, 30, 50]

# Boolean indexing (masking)
mask = a > 25
print(mask)        # [False, False, True, True, True]
print(a[mask])     # [30, 40, 50]

# Combine conditions
mask = (a > 15) & (a < 45)  # Note: use & not 'and'
print(a[mask])     # [20, 30, 40]

# Practical example: selecting energies in range
E = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
thermal = E[(E > 0.3) & (E < 3.0)]  # Thermal energies
```

### 7. Reshaping and Transposing

```python
a = np.arange(12)

# Reshape (returns view if possible)
b = a.reshape(3, 4)
c = a.reshape(2, 2, 3)  # 3D
d = a.reshape(12, 1)    # Column vector
e = a.reshape(-1, 3)    # -1 means "infer this dimension"

# Transpose
print(b.T)              # 4×3 transpose
print(b.T.shape)        # (4, 3)

# For higher dimensions
t = np.arange(24).reshape(2, 3, 4)
print(t.transpose(1, 0, 2).shape)  # (3, 2, 4)

# Flatten (returns copy)
flat = b.flatten()      # Always copy
ravel = b.ravel()       # View if possible

# Newaxis for adding dimensions (broadcasting)
row = np.array([1, 2, 3])          # Shape (3,)
col = row[:, np.newaxis]           # Shape (3, 1)
outer = row[np.newaxis, :] * col   # Outer product (3, 3)
```

### 8. Views vs Copies

```python
# CRITICAL: Understanding when data is shared

a = np.arange(10)

# VIEWS (shared data)
b = a[2:5]        # Slice = view
c = a.reshape(2, 5)  # Reshape = view (usually)
d = a.T           # Transpose = view
e = a.ravel()     # Ravel = view (if contiguous)

# COPIES (independent data)
f = a.copy()      # Explicit copy
g = a[[1, 3, 5]]  # Fancy indexing = copy
h = a.flatten()   # Flatten = always copy
i = np.array(a)   # np.array() = copy

# Check if view
print(np.shares_memory(a, b))  # True (view)
print(np.shares_memory(a, f))  # False (copy)

# The base attribute
print(b.base is a)  # True (b is view of a)
print(f.base is a)  # False (f is independent)
```

---

## Quantum Mechanics Connection

### Discretizing Wave Functions

A wave function $\psi(x)$ is continuous, but we represent it on a discrete grid:

$$\psi(x) \rightarrow (\psi_0, \psi_1, \ldots, \psi_{N-1}) = (\psi(x_0), \psi(x_1), \ldots, \psi(x_{N-1}))$$

```python
# Grid setup for 1D quantum mechanics
x_min, x_max = -10.0, 10.0
N = 1000
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# Discretized wave function (harmonic oscillator ground state)
def psi_harmonic(x, n=0):
    """Harmonic oscillator eigenstate (atomic units)."""
    from scipy.special import hermite
    from math import factorial, pi
    Hn = hermite(n)
    norm = 1 / np.sqrt(2**n * factorial(n)) * (1/np.pi)**0.25
    return norm * Hn(x) * np.exp(-x**2 / 2)

psi = psi_harmonic(x, n=0)  # Ground state as array
```

### The Discretized Inner Product

The inner product becomes a sum:
$$\langle\phi|\psi\rangle = \int_{-\infty}^{\infty} \phi^*(x)\psi(x)dx \approx \sum_i \phi_i^* \psi_i \cdot dx$$

```python
def inner_product(phi, psi, dx):
    """Discrete inner product ⟨φ|ψ⟩."""
    return np.sum(np.conj(phi) * psi) * dx

# Normalization check
norm_sq = inner_product(psi, psi, dx)
print(f"⟨ψ|ψ⟩ = {norm_sq:.6f}")  # Should be ≈ 1
```

### Operators as Matrix Operations

In the discretized basis, operators become matrices:

```python
# Position operator: x̂ψ = xψ
def apply_x(psi, x):
    """Apply position operator."""
    return x * psi  # Element-wise multiplication

# Kinetic energy operator: T̂ = -ℏ²/(2m) d²/dx²
def apply_kinetic(psi, dx, hbar=1.0, m=1.0):
    """Apply kinetic energy operator using finite difference."""
    d2_psi = np.zeros_like(psi)
    d2_psi[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
    return -hbar**2 / (2*m) * d2_psi

# Expectation value ⟨x⟩
x_expect = inner_product(psi, apply_x(psi, x), dx)
print(f"⟨x⟩ = {x_expect:.6f}")  # Should be 0 for ground state
```

---

## Worked Examples

### Example 1: Creating a 2D Probability Density

**Problem:** Create a 2D Gaussian probability distribution centered at origin.

```python
# Create coordinate grids
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Parameters
sigma_x, sigma_y = 1.0, 2.0
x0, y0 = 0.0, 0.0

# 2D Gaussian
rho = (1 / (2 * np.pi * sigma_x * sigma_y)) * \
      np.exp(-0.5 * ((X - x0)**2 / sigma_x**2 + (Y - y0)**2 / sigma_y**2))

# Check normalization
dx, dy = x[1] - x[0], y[1] - y[0]
total = np.sum(rho) * dx * dy
print(f"Total probability: {total:.6f}")  # ≈ 1.0
```

### Example 2: Extracting Bound State Region

**Problem:** Given a potential and energy, extract the classically allowed region.

```python
# Harmonic oscillator potential
x = np.linspace(-5, 5, 500)
V = 0.5 * x**2  # V(x) = ½x²

# Energy level
E = 2.5  # n=2 energy

# Classical turning points: V(x) = E → x = ±√(2E)
classically_allowed = V <= E
x_allowed = x[classically_allowed]

print(f"Classically allowed: x ∈ [{x_allowed[0]:.3f}, {x_allowed[-1]:.3f}]")
print(f"Theoretical: x ∈ [{-np.sqrt(2*E):.3f}, {np.sqrt(2*E):.3f}]")
```

### Example 3: Building a Hamiltonian Matrix

**Problem:** Construct the kinetic energy matrix using finite differences.

```python
def kinetic_matrix(N, dx, hbar=1.0, m=1.0):
    """
    Build kinetic energy matrix for N grid points.

    T = -ℏ²/(2m) d²/dx² ≈ -ℏ²/(2m dx²) * tridiagonal(-1, 2, -1)
    """
    coeff = -hbar**2 / (2 * m * dx**2)

    # Create tridiagonal matrix
    diag = np.full(N, -2.0 * coeff)
    off_diag = np.full(N-1, 1.0 * coeff)

    T = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    return T

# Test
N, L = 100, 10.0
dx = L / (N - 1)
T = kinetic_matrix(N, dx)

print(f"T shape: {T.shape}")
print(f"T[50,50] (diagonal): {T[50,50]:.4f}")
print(f"T[50,51] (off-diagonal): {T[50,51]:.4f}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Create a 3D array of shape (10, 20, 30) filled with random numbers from a normal distribution. Calculate its mean and standard deviation.

**Problem 2:** Given `x = np.linspace(-np.pi, np.pi, 100)`, extract all x values where `sin(x) > 0.5` using boolean indexing.

**Problem 3:** Create a 5×5 matrix where element [i,j] equals i² + j. Do this without loops using broadcasting.

### Intermediate

**Problem 4:** Implement a function `checkerboard(n)` that creates an n×n checkerboard pattern of 0s and 1s using slicing, not loops.

**Problem 5:** Given a wave function array `psi`, compute the probability current $j = \frac{\hbar}{2mi}(\psi^* \nabla\psi - \psi\nabla\psi^*)$ using finite differences.

**Problem 6:** Create the matrices for raising and lowering operators $a^\dagger$ and $a$ in the harmonic oscillator (truncated to N levels).

### Challenging

**Problem 7:** Implement a function that takes a 2D wave function ψ(x,y) and computes the reduced density in x by integrating over y.

**Problem 8:** Create a 3D array representing $|\psi(x,y,z)|^2$ for the hydrogen 2p orbital (use spherical to Cartesian conversion).

**Problem 9:** Implement periodic boundary conditions for a 1D lattice: create a circulant matrix for the kinetic operator.

---

## Computational Lab

### Discretizing Quantum Wave Functions

```python
"""
Day 254 Computational Lab: NumPy Array Fundamentals
====================================================

This lab builds the numerical infrastructure for quantum simulations.
"""

import numpy as np
from typing import Tuple, Callable

# ============================================================
# GRID CREATION UTILITIES
# ============================================================

def create_grid_1d(x_min: float, x_max: float, N: int) -> Tuple[np.ndarray, float]:
    """
    Create 1D spatial grid.

    Returns
    -------
    x : ndarray
        Grid points
    dx : float
        Grid spacing
    """
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    return x, dx

def create_grid_2d(x_range: Tuple[float, float],
                   y_range: Tuple[float, float],
                   Nx: int, Ny: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Create 2D spatial grid.

    Returns
    -------
    X, Y : ndarray
        2D coordinate meshes
    dx, dy : float
        Grid spacings
    """
    x = np.linspace(*x_range, Nx)
    y = np.linspace(*y_range, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y, dx, dy

# ============================================================
# WAVE FUNCTION DISCRETIZATION
# ============================================================

def discretize_function(f: Callable, x: np.ndarray) -> np.ndarray:
    """
    Discretize a continuous function onto a grid.

    Parameters
    ----------
    f : callable
        Function f(x) to discretize (can be vectorized)
    x : ndarray
        Grid points

    Returns
    -------
    ndarray
        Function values at grid points
    """
    return np.asarray(f(x))

def normalize_wavefunction(psi: np.ndarray, dx: float) -> np.ndarray:
    """
    Normalize wave function to unit probability.

    Parameters
    ----------
    psi : ndarray
        Wave function (complex)
    dx : float
        Grid spacing

    Returns
    -------
    ndarray
        Normalized wave function
    """
    norm_sq = np.sum(np.abs(psi)**2) * dx
    return psi / np.sqrt(norm_sq)

# ============================================================
# STANDARD WAVE FUNCTIONS
# ============================================================

def gaussian_wavepacket(x: np.ndarray, x0: float = 0.0,
                         sigma: float = 1.0, k0: float = 0.0) -> np.ndarray:
    """
    Gaussian wave packet.

    ψ(x) = (2πσ²)^(-1/4) exp(-(x-x₀)²/4σ²) exp(ik₀x)

    Parameters
    ----------
    x : ndarray
        Position grid
    x0 : float
        Center position
    sigma : float
        Width parameter
    k0 : float
        Central wave number (momentum ℏk₀)

    Returns
    -------
    ndarray
        Complex wave function values
    """
    norm = (2 * np.pi * sigma**2)**(-0.25)
    envelope = np.exp(-(x - x0)**2 / (4 * sigma**2))
    phase = np.exp(1j * k0 * x)
    return norm * envelope * phase

def harmonic_eigenstate(x: np.ndarray, n: int,
                        m: float = 1.0, omega: float = 1.0,
                        hbar: float = 1.0) -> np.ndarray:
    """
    Harmonic oscillator eigenstate.

    ψₙ(x) = (mω/πℏ)^(1/4) / √(2ⁿn!) Hₙ(ξ) exp(-ξ²/2)

    where ξ = x√(mω/ℏ)
    """
    from scipy.special import hermite
    from math import factorial

    xi = x * np.sqrt(m * omega / hbar)
    Hn = hermite(n)

    prefactor = (m * omega / (np.pi * hbar))**0.25
    prefactor /= np.sqrt(2**n * factorial(n))

    return prefactor * Hn(xi) * np.exp(-xi**2 / 2)

def particle_in_box_eigenstate(x: np.ndarray, n: int, L: float) -> np.ndarray:
    """
    Particle-in-a-box eigenstate.

    ψₙ(x) = √(2/L) sin(nπx/L) for 0 < x < L

    Parameters
    ----------
    x : ndarray
        Position grid (should be within [0, L])
    n : int
        Quantum number (n = 1, 2, 3, ...)
    L : float
        Box length
    """
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    # Set to zero outside box
    psi = np.where((x >= 0) & (x <= L), psi, 0.0)
    return psi

# ============================================================
# OBSERVABLES AND EXPECTATION VALUES
# ============================================================

def inner_product(phi: np.ndarray, psi: np.ndarray, dx: float) -> complex:
    """Compute ⟨φ|ψ⟩."""
    return np.sum(np.conj(phi) * psi) * dx

def expectation_position(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    """Compute ⟨x⟩ = ⟨ψ|x̂|ψ⟩."""
    return np.real(np.sum(np.conj(psi) * x * psi) * dx)

def expectation_position_squared(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    """Compute ⟨x²⟩."""
    return np.real(np.sum(np.conj(psi) * x**2 * psi) * dx)

def uncertainty_position(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    """Compute position uncertainty Δx = √(⟨x²⟩ - ⟨x⟩²)."""
    x_avg = expectation_position(psi, x, dx)
    x2_avg = expectation_position_squared(psi, x, dx)
    return np.sqrt(x2_avg - x_avg**2)

# ============================================================
# OPERATORS AS MATRIX OPERATIONS
# ============================================================

def kinetic_operator_matrix(N: int, dx: float,
                            hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    """
    Create kinetic energy operator matrix using finite difference.

    T = -ℏ²/(2m) d²/dx²

    Uses central difference: d²ψ/dx² ≈ (ψᵢ₊₁ - 2ψᵢ + ψᵢ₋₁)/dx²
    """
    coeff = hbar**2 / (2 * m * dx**2)

    # Tridiagonal: (-1, 2, -1) pattern
    diag = np.full(N, 2.0 * coeff)
    off = np.full(N - 1, -coeff)

    T = np.diag(diag) + np.diag(off, k=1) + np.diag(off, k=-1)
    return T

def potential_operator_matrix(V_values: np.ndarray) -> np.ndarray:
    """
    Create potential energy operator matrix.

    V is diagonal in position basis.
    """
    return np.diag(V_values)

def hamiltonian_matrix(x: np.ndarray, V: Callable,
                       hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    """
    Build full Hamiltonian matrix H = T + V.

    Parameters
    ----------
    x : ndarray
        Position grid
    V : callable
        Potential function V(x)
    hbar, m : float
        Physical constants

    Returns
    -------
    ndarray
        Hamiltonian matrix (N × N)
    """
    N = len(x)
    dx = x[1] - x[0]

    T = kinetic_operator_matrix(N, dx, hbar, m)
    V_diag = potential_operator_matrix(V(x))

    return T + V_diag

# ============================================================
# ANALYSIS TOOLS
# ============================================================

def probability_density(psi: np.ndarray) -> np.ndarray:
    """Compute |ψ(x)|²."""
    return np.abs(psi)**2

def probability_in_region(psi: np.ndarray, x: np.ndarray,
                          dx: float, x_min: float, x_max: float) -> float:
    """Compute probability to find particle in [x_min, x_max]."""
    mask = (x >= x_min) & (x <= x_max)
    return np.sum(np.abs(psi[mask])**2) * dx

def nodes(psi: np.ndarray, threshold: float = 1e-10) -> int:
    """Count number of nodes (zero crossings) in wave function."""
    real_psi = np.real(psi)
    signs = np.sign(real_psi)
    # Remove near-zero values
    signs[np.abs(real_psi) < threshold] = 0
    # Count sign changes
    sign_changes = np.diff(signs)
    return np.sum(sign_changes != 0)

# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Day 254: NumPy Arrays - Quantum Wave Function Discretization")
    print("=" * 70)

    # Create grid
    x, dx = create_grid_1d(-10.0, 10.0, 1000)
    print(f"\nGrid: {len(x)} points, dx = {dx:.4f}")

    # Harmonic oscillator eigenstates
    print("\n--- Harmonic Oscillator Eigenstates ---")
    for n in range(4):
        psi_n = harmonic_eigenstate(x, n)
        norm = inner_product(psi_n, psi_n, dx)
        x_exp = expectation_position(psi_n, x, dx)
        delta_x = uncertainty_position(psi_n, x, dx)
        n_nodes = nodes(psi_n)

        print(f"  n={n}: ⟨ψ|ψ⟩={norm:.6f}, ⟨x⟩={x_exp:.4f}, "
              f"Δx={delta_x:.4f}, nodes={n_nodes}")

    # Orthogonality check
    print("\n--- Orthogonality ---")
    psi_0 = harmonic_eigenstate(x, 0)
    psi_1 = harmonic_eigenstate(x, 1)
    psi_2 = harmonic_eigenstate(x, 2)

    print(f"  ⟨ψ₀|ψ₁⟩ = {inner_product(psi_0, psi_1, dx):.2e}")
    print(f"  ⟨ψ₀|ψ₂⟩ = {inner_product(psi_0, psi_2, dx):.2e}")
    print(f"  ⟨ψ₁|ψ₂⟩ = {inner_product(psi_1, psi_2, dx):.2e}")

    # Gaussian wave packet
    print("\n--- Gaussian Wave Packet ---")
    psi_packet = gaussian_wavepacket(x, x0=2.0, sigma=0.5, k0=5.0)
    psi_packet = normalize_wavefunction(psi_packet, dx)
    print(f"  Center: x₀ = 2.0, momentum: k₀ = 5.0")
    print(f"  ⟨x⟩ = {expectation_position(psi_packet, x, dx):.4f}")
    print(f"  Δx = {uncertainty_position(psi_packet, x, dx):.4f}")

    # Particle in box
    print("\n--- Particle in Box (L=5) ---")
    L = 5.0
    x_box, dx_box = create_grid_1d(0, L, 500)
    for n in [1, 2, 3]:
        psi_box = particle_in_box_eigenstate(x_box, n, L)
        norm = inner_product(psi_box, psi_box, dx_box)
        n_nodes = nodes(psi_box)
        print(f"  n={n}: norm={norm:.6f}, nodes={n_nodes}")

    # Hamiltonian matrix
    print("\n--- Hamiltonian Matrix (Harmonic Oscillator) ---")
    x_small, dx_small = create_grid_1d(-5, 5, 50)
    V = lambda x: 0.5 * x**2
    H = hamiltonian_matrix(x_small, V)
    print(f"  H shape: {H.shape}")
    print(f"  H is Hermitian: {np.allclose(H, H.T.conj())}")

    # Eigenvalues (preview of Day 256)
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"  First 5 eigenvalues: {eigenvalues[:5]}")
    print(f"  Expected (n+0.5):    {[n + 0.5 for n in range(5)]}")

    # 2D grid demo
    print("\n--- 2D Grid Creation ---")
    X, Y, dx2, dy2 = create_grid_2d((-5, 5), (-5, 5), 100, 100)
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"  dx = {dx2:.4f}, dy = {dy2:.4f}")

    # 2D Gaussian
    sigma = 1.0
    psi_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psi_2d /= np.sqrt(np.sum(np.abs(psi_2d)**2) * dx2 * dy2)
    print(f"  2D Gaussian normalized: {np.sum(np.abs(psi_2d)**2) * dx2 * dy2:.6f}")

    print("\n" + "=" * 70)
    print("Lab complete! Vectorization and broadcasting continue on Day 255.")
    print("=" * 70)
```

**Expected Output:**
```
======================================================================
Day 254: NumPy Arrays - Quantum Wave Function Discretization
======================================================================

Grid: 1000 points, dx = 0.0200

--- Harmonic Oscillator Eigenstates ---
  n=0: ⟨ψ|ψ⟩=1.000000, ⟨x⟩=0.0000, Δx=0.7071, nodes=0
  n=1: ⟨ψ|ψ⟩=1.000000, ⟨x⟩=-0.0000, Δx=1.2247, nodes=1
  n=2: ⟨ψ|ψ⟩=1.000000, ⟨x⟩=0.0000, Δx=1.5811, nodes=2
  n=3: ⟨ψ|ψ⟩=1.000000, ⟨x⟩=-0.0000, Δx=1.8708, nodes=3

--- Orthogonality ---
  ⟨ψ₀|ψ₁⟩ = 8.84e-17
  ⟨ψ₀|ψ₂⟩ = -1.69e-16
  ⟨ψ₁|ψ₂⟩ = 3.09e-17

--- Gaussian Wave Packet ---
  Center: x₀ = 2.0, momentum: k₀ = 5.0
  ⟨x⟩ = 2.0000
  Δx = 0.5000

--- Particle in Box (L=5) ---
  n=1: norm=1.000000, nodes=0
  n=2: norm=1.000000, nodes=1
  n=3: norm=1.000000, nodes=2

--- Hamiltonian Matrix (Harmonic Oscillator) ---
  H shape: (50, 50)
  H is Hermitian: True
  First 5 eigenvalues: [0.5000, 1.4999, 2.4997, 3.4991, 4.4981]
  Expected (n+0.5):    [0.5, 1.5, 2.5, 3.5, 4.5]

--- 2D Grid Creation ---
  X shape: (100, 100), Y shape: (100, 100)
  dx = 0.1010, dy = 0.1010
  2D Gaussian normalized: 1.000000

======================================================================
Lab complete! Vectorization and broadcasting continue on Day 255.
======================================================================
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `np.linspace(a, b, n)` | n evenly spaced points | `x = np.linspace(-10, 10, 1000)` |
| `np.arange(a, b, step)` | Points with fixed step | `t = np.arange(0, 1, 0.01)` |
| `np.meshgrid(x, y)` | 2D coordinate grids | `X, Y = np.meshgrid(x, y)` |
| `np.zeros((n, m))` | Zero matrix | `H = np.zeros((100, 100))` |
| `np.diag(v, k)` | Diagonal matrix | `T = np.diag(d) + np.diag(off, 1)` |
| `a.reshape(shape)` | Change shape | `H = a.reshape(10, 10)` |
| `a[mask]` | Boolean indexing | `x_pos = x[x > 0]` |

### Main Takeaways

1. **NumPy arrays are homogeneous and contiguous** — enables fast operations
2. **Slicing returns views** — modifications affect the original
3. **Fancy indexing returns copies** — safe but slower
4. **Use appropriate dtypes** — `complex128` for wave functions
5. **Discretization converts continuous physics to arrays** — foundation for numerics
6. **Matrix operators in position basis** — kinetic energy is tridiagonal

---

## Daily Checklist

- [ ] Can create arrays with `zeros`, `linspace`, `arange`, `meshgrid`
- [ ] Understand dtypes and can choose appropriate precision
- [ ] Can reshape arrays and use transpose
- [ ] Master slicing syntax including negative indices
- [ ] Can use boolean and integer array indexing
- [ ] Distinguish views from copies
- [ ] Completed discretization of wave functions
- [ ] Ran computational lab successfully

---

## Preview: Day 255

Tomorrow we unlock NumPy's full power with **vectorization and broadcasting**. We'll learn:
- Universal functions (ufuncs) that eliminate loops
- Broadcasting rules for combining arrays of different shapes
- Why `np.sum(psi * x * psi.conj())` is 100x faster than a Python loop
- Vectorized implementations of all our operators

This is where NumPy becomes truly powerful for physics.

---

*"Every element of an array operation happens simultaneously in the mind of NumPy."*
