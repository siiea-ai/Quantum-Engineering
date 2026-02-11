# Day 265: Sparse Matrices with SciPy

## Overview

**Day 265** | **Week 38** | **Month 10: Scientific Computing**

Today we master `scipy.sparse` for handling large quantum systems. Real quantum simulations involve Hilbert spaces of millions or billions of dimensions, but the Hamiltonians are typically sparse—most matrix elements are zero. Sparse matrix techniques reduce memory from O(N²) to O(N) and enable eigenvalue computations that would be impossible otherwise.

**Prerequisites:** Days 260-264, linear algebra
**Outcome:** Build and solve sparse quantum Hamiltonians at scale

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Sparse formats, construction, operations |
| Afternoon | 3 hours | Practice: Large-scale eigenvalue problems |
| Evening | 2 hours | Lab: Tight-binding and Hubbard models |

---

## Learning Objectives

By the end of Day 265, you will be able to:

1. **Choose appropriate sparse formats** (CSR, CSC, COO, LIL, DOK)
2. **Construct sparse matrices** efficiently for physics problems
3. **Perform sparse matrix operations** (arithmetic, products)
4. **Solve sparse eigenvalue problems** with `eigsh` and `eigs`
5. **Implement large-scale Hamiltonians** (tight-binding, Hubbard)
6. **Understand memory and performance tradeoffs**
7. **Convert between sparse and dense** as needed

---

## Core Content

### 1. Why Sparse Matrices?

```python
import numpy as np
from scipy import sparse
import sys

# Dense matrix memory usage
N = 10000
dense_size = N * N * 8  # bytes (float64)
print(f"Dense {N}×{N}: {dense_size / 1e9:.2f} GB")

# Sparse matrix: only store non-zeros
# Tridiagonal has 3N non-zeros
sparse_size = 3 * N * 8 + 3 * N * 4  # values + indices (approximate)
print(f"Sparse tridiagonal: {sparse_size / 1e6:.2f} MB")
print(f"Compression ratio: {dense_size / sparse_size:.0f}x")

# For N = 1,000,000 (typical tight-binding)
N_large = 1_000_000
print(f"\nFor N = {N_large:,}:")
print(f"  Dense: {N_large**2 * 8 / 1e15:.0f} PB (impossible)")
print(f"  Sparse: {3 * N_large * 12 / 1e9:.2f} GB (manageable)")
```

### 2. Sparse Matrix Formats

```python
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix, dok_matrix

# Different formats for different use cases

# COO (Coordinate): Easy to construct
row = [0, 1, 2, 0]
col = [0, 1, 2, 2]
data = [1, 2, 3, 4]
A_coo = coo_matrix((data, (row, col)), shape=(3, 3))
print("COO format:")
print(A_coo.toarray())

# CSR (Compressed Sparse Row): Fast row slicing, matrix-vector products
A_csr = A_coo.tocsr()
print(f"\nCSR: indptr={A_csr.indptr}, indices={A_csr.indices}, data={A_csr.data}")

# CSC (Compressed Sparse Column): Fast column slicing
A_csc = A_coo.tocsc()

# LIL (List of Lists): Efficient for incremental construction
A_lil = lil_matrix((3, 3))
A_lil[0, 0] = 1
A_lil[1, 1] = 2
A_lil[2, 2] = 3
A_lil[0, 2] = 4

# DOK (Dictionary of Keys): Fast random access during construction
A_dok = dok_matrix((3, 3))
A_dok[0, 0] = 1
A_dok[1, 1] = 2

# Convert to CSR for computation
A_final = A_lil.tocsr()
```

### 3. Building Sparse Matrices

```python
from scipy.sparse import diags, eye, kron, block_diag

# Diagonal matrices (most common in physics)
N = 100
diagonal = np.arange(N)
off_diagonal = np.ones(N - 1)

# Tridiagonal matrix
H = diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1], format='csr')
print(f"Tridiagonal: shape={H.shape}, nnz={H.nnz}")

# Identity
I = eye(N, format='csr')

# Kronecker product (for tensor products)
A = diags([1, 2, 1], [-1, 0, 1], shape=(3, 3))
B = diags([0, 1, 0], [-1, 0, 1], shape=(3, 3))
AB = kron(A, B, format='csr')  # A ⊗ B
print(f"Kronecker product: {A.shape} ⊗ {B.shape} = {AB.shape}")

# Block diagonal
blocks = [diags([1, 2, 1], [-1, 0, 1], shape=(3, 3)) for _ in range(4)]
BD = block_diag(blocks, format='csr')
print(f"Block diagonal: shape={BD.shape}")
```

### 4. Sparse Matrix Operations

```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

N = 1000
A = diags([np.ones(N-1), 2*np.ones(N), np.ones(N-1)], [-1, 0, 1], format='csr')
B = diags([np.ones(N-1), -1*np.ones(N), np.ones(N-1)], [-1, 0, 1], format='csr')

# Arithmetic
C = A + B
D = 2 * A
E = A - B

# Matrix-matrix product (result may be dense!)
AB = A @ B  # or A.dot(B)
print(f"A @ B: nnz = {AB.nnz} / {N*N} = {AB.nnz/(N*N)*100:.1f}%")

# Matrix-vector product (fast!)
x = np.random.randn(N)
y = A @ x

# Power (be careful - can densify)
# A_squared = A @ A  # OK for sparse result
# A_power_10 = A ** 10  # May be very slow/dense

# Transpose
At = A.T  # or A.transpose()

# Norm
frobenius = norm(A, 'fro')
print(f"||A||_F = {frobenius:.4f}")
```

### 5. Sparse Eigenvalue Problems

```python
from scipy.sparse.linalg import eigsh, eigs

# Build Hamiltonian
N = 5000
x = np.linspace(-10, 10, N)
dx = x[1] - x[0]

# Kinetic + harmonic potential
T_coeff = 1 / (2 * dx**2)
V = 0.5 * x**2

H = diags(
    [-T_coeff * np.ones(N-1), 2*T_coeff + V, -T_coeff * np.ones(N-1)],
    [-1, 0, 1],
    format='csr'
)

# Find k smallest eigenvalues (Hermitian)
k = 10
eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')  # Smallest Algebraic

print(f"First {k} eigenvalues of harmonic oscillator (N={N}):")
for n, E in enumerate(eigenvalues):
    exact = n + 0.5
    print(f"  E_{n} = {E:.6f} (exact: {exact})")

# Verify eigenvector
psi_0 = eigenvectors[:, 0]
norm = np.sqrt(np.sum(psi_0**2) * dx)
print(f"\nGround state normalization: {norm:.6f}")
```

### 6. Which Eigenvalues to Find

```python
from scipy.sparse.linalg import eigsh, eigs

# For Hermitian matrices (Hamiltonians): eigsh
# which='SA': Smallest Algebraic (ground state)
# which='LA': Largest Algebraic
# which='SM': Smallest Magnitude
# which='LM': Largest Magnitude (default)

# Example: Find eigenvalues near a target
sigma = 5.0  # Target energy
eigenvalues, eigenvectors = eigsh(H, k=5, sigma=sigma, which='LM')
print(f"Eigenvalues near E={sigma}:")
print(eigenvalues)

# For non-Hermitian: eigs
# (e.g., non-Hermitian effective Hamiltonians)

# Shift-invert mode (much faster for interior eigenvalues)
# Solves (H - σI)^(-1) x = μx, then E = σ + 1/μ
```

### 7. Sparse Linear Systems

```python
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab

# Direct solve (for moderately sized systems)
b = np.random.randn(N)
x = spsolve(H, b)

# Iterative solvers (for very large systems)

# Conjugate Gradient (for positive definite)
H_pd = H + 10 * eye(N, format='csr')  # Make positive definite
x_cg, info = cg(H_pd, b, tol=1e-10)
print(f"CG converged: {info == 0}")

# GMRES (general, non-symmetric)
x_gmres, info = gmres(H, b, tol=1e-10)
print(f"GMRES converged: {info == 0}")

# Check residual
residual = np.linalg.norm(H @ x - b)
print(f"Residual: {residual:.2e}")
```

---

## Quantum Mechanics Connection

### Tight-Binding Model

```python
def tight_binding_1d(N, t=1.0, periodic=False):
    """
    1D tight-binding Hamiltonian.

    H = -t Σ (c†_i c_{i+1} + h.c.)
    """
    diag = np.zeros(N)
    off_diag = -t * np.ones(N - 1)

    diagonals = [off_diag, diag, off_diag]
    offsets = [-1, 0, 1]

    H = diags(diagonals, offsets, shape=(N, N), format='csr')

    if periodic:
        # Add periodic boundary conditions
        H = H + csr_matrix(([[-t], [-t]], [[0, N-1], [N-1, 0]]), shape=(N, N))

    return H

# 1D chain
N = 100
H_tb = tight_binding_1d(N, t=1.0, periodic=True)

# Band structure (all eigenvalues for small system)
eigenvalues = eigsh(H_tb, k=N-2, which='SA', return_eigenvectors=False)
eigenvalues = np.sort(eigenvalues)

# Analytical: E_k = -2t cos(ka), k = 2πn/N
k_analytical = 2 * np.pi * np.arange(N) / N
E_analytical = -2 * np.cos(k_analytical)

print("Tight-binding band structure:")
print(f"  Bandwidth: {np.max(eigenvalues) - np.min(eigenvalues):.4f} (theory: 4t = 4)")
```

### 2D Tight-Binding (Square Lattice)

```python
def tight_binding_2d(Nx, Ny, t=1.0, periodic=False):
    """
    2D tight-binding on Nx × Ny square lattice.

    Site (i, j) → index i + j*Nx
    """
    N = Nx * Ny

    # Hopping in x-direction
    diag_x = []
    for j in range(Ny):
        diag_x.extend([-t] * (Nx - 1))
        if periodic:
            pass  # Handle wrap-around separately

    H_x = diags([diag_x, diag_x], [-1, 1], shape=(N, N), format='lil')

    # Remove incorrect hoppings at row boundaries
    for j in range(1, Ny):
        idx = j * Nx
        H_x[idx, idx-1] = 0
        H_x[idx-1, idx] = 0

    # Hopping in y-direction
    diag_y = -t * np.ones(N - Nx)
    H_y = diags([diag_y, diag_y], [-Nx, Nx], shape=(N, N), format='csr')

    H = (H_x + H_y).tocsr()

    return H

# 10x10 lattice
Nx, Ny = 10, 10
H_2d = tight_binding_2d(Nx, Ny)

eigenvalues_2d, eigenvectors_2d = eigsh(H_2d, k=20, which='SA')
print(f"\n2D tight-binding ({Nx}×{Ny} lattice):")
print(f"  Ground state energy: {eigenvalues_2d[0]:.6f}")
print(f"  Bandwidth: {np.max(eigenvalues_2d) - np.min(eigenvalues_2d):.4f}")
```

### Hubbard Model

```python
def hubbard_2site(t=1.0, U=4.0):
    """
    Two-site Hubbard model.

    H = -t(c†_1↑c_2↑ + c†_2↑c_1↑ + c†_1↓c_2↓ + c†_2↓c_1↓) + U(n_1↑n_1↓ + n_2↑n_2↓)

    Basis: |n_1↑, n_1↓, n_2↑, n_2↓⟩
    16 states for 2 sites with spin
    """
    # For simplicity, work in half-filling sector (2 electrons)
    # Basis: |↑,0⟩, |0,↑⟩, |↓,0⟩, |0,↓⟩, |↑↓,0⟩, |0,↑↓⟩, |↑,↓⟩, |↓,↑⟩

    # At half-filling with Sz=0: |↑,↓⟩, |↓,↑⟩, |↑↓,0⟩, |0,↑↓⟩
    # 4-dimensional subspace

    H = np.array([
        [0, -t, -t, 0],
        [-t, 0, 0, -t],
        [-t, 0, U, 0],
        [0, -t, 0, U]
    ])

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    print(f"2-site Hubbard (t={t}, U={U}):")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:.6f}")

    return eigenvalues, eigenvectors

hubbard_2site(t=1.0, U=4.0)
hubbard_2site(t=1.0, U=0.0)  # Non-interacting limit
```

### Large-Scale Example

```python
def large_scale_demo():
    """
    Demonstrate sparse methods for large system.
    """
    N = 100000  # 100,000 site chain
    t = 1.0

    print(f"\nLarge-scale demo: N = {N:,} sites")

    # Build Hamiltonian
    import time
    start = time.perf_counter()

    H = tight_binding_1d(N, t, periodic=True)

    build_time = time.perf_counter() - start
    print(f"  Build time: {build_time:.3f} s")
    print(f"  Memory: {(H.data.nbytes + H.indices.nbytes + H.indptr.nbytes) / 1e6:.2f} MB")

    # Find ground state
    start = time.perf_counter()
    E0, psi0 = eigsh(H, k=1, which='SA')
    solve_time = time.perf_counter() - start

    print(f"  Solve time: {solve_time:.3f} s")
    print(f"  Ground state energy: {E0[0]:.10f}")
    print(f"  Exact (periodic): {-2*t:.10f}")

large_scale_demo()
```

---

## Worked Examples

### Example 1: Disorder in Tight-Binding

```python
def anderson_model(N, t=1.0, W=0.0, seed=42):
    """
    Anderson model with on-site disorder.

    H = -t Σ(c†_i c_{i+1} + h.c.) + Σ ε_i n_i

    where ε_i ∈ [-W/2, W/2] uniformly distributed.
    """
    np.random.seed(seed)
    disorder = np.random.uniform(-W/2, W/2, N)

    H = tight_binding_1d(N, t)
    H = H + diags([disorder], [0], format='csr')

    return H

# Compare clean vs disordered
N = 1000
H_clean = anderson_model(N, t=1.0, W=0.0)
H_weak = anderson_model(N, t=1.0, W=1.0)
H_strong = anderson_model(N, t=1.0, W=5.0)

print("Anderson localization:")
for name, H in [("Clean", H_clean), ("W=1", H_weak), ("W=5", H_strong)]:
    E, psi = eigsh(H, k=5, which='SA')
    # Inverse participation ratio (localization measure)
    ipr = np.sum(np.abs(psi)**4, axis=0)
    print(f"  {name}: E_0 = {E[0]:.4f}, IPR = {ipr[0]:.4f}")
```

### Example 2: SSH Model (Topological)

```python
def ssh_model(N, v=1.0, w=0.5):
    """
    Su-Schrieffer-Heeger model.

    H = Σ (v c†_{2i} c_{2i+1} + w c†_{2i+1} c_{2i+2} + h.c.)

    Dimerized chain with alternating hoppings.
    """
    # N unit cells → 2N sites
    sites = 2 * N

    # Intracell hopping (v)
    v_hops = np.array([v if i % 2 == 0 else 0 for i in range(sites - 1)])

    # Intercell hopping (w)
    w_hops = np.array([w if i % 2 == 1 else 0 for i in range(sites - 1)])

    hops = v_hops + w_hops

    H = diags([hops, hops], [-1, 1], shape=(sites, sites), format='csr')

    return H

# Trivial vs topological phase
N = 50
H_trivial = ssh_model(N, v=1.0, w=0.5)  # v > w: trivial
H_topo = ssh_model(N, v=0.5, w=1.0)     # v < w: topological

E_trivial, _ = eigsh(H_trivial, k=10, sigma=0, which='LM')
E_topo, _ = eigsh(H_topo, k=10, sigma=0, which='LM')

print("SSH model (50 unit cells, open BC):")
print(f"  Trivial (v>w): E near 0: {np.sort(E_trivial)[:4]}")
print(f"  Topological (v<w): E near 0: {np.sort(E_topo)[:4]}")
print("  (Topological phase has zero-energy edge states)")
```

### Example 3: Lanczos Algorithm

```python
def lanczos(H, v0, k):
    """
    Lanczos algorithm for finding extremal eigenvalues.

    Builds k-dimensional Krylov subspace.
    """
    n = len(v0)
    V = np.zeros((n, k+1))
    T = np.zeros((k, k))

    # Normalize initial vector
    V[:, 0] = v0 / np.linalg.norm(v0)

    for j in range(k):
        w = H @ V[:, j]

        if j > 0:
            w = w - T[j-1, j] * V[:, j-1]

        T[j, j] = np.dot(V[:, j], w)
        w = w - T[j, j] * V[:, j]

        if j < k - 1:
            T[j, j+1] = np.linalg.norm(w)
            T[j+1, j] = T[j, j+1]

            if T[j, j+1] > 1e-10:
                V[:, j+1] = w / T[j, j+1]
            else:
                break  # Invariant subspace found

    # Eigenvalues of tridiagonal T approximate extremal eigenvalues of H
    eigenvalues_T = np.linalg.eigvalsh(T)
    return eigenvalues_T, T

# Test
N = 1000
H = tight_binding_1d(N, t=1.0)
v0 = np.random.randn(N)

E_lanczos, T = lanczos(H, v0, k=50)
E_exact, _ = eigsh(H, k=5, which='SA')

print("Lanczos vs exact (ground state):")
print(f"  Lanczos (k=50): {E_lanczos[0]:.10f}")
print(f"  Exact (eigsh):  {E_exact[0]:.10f}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Build a sparse tridiagonal matrix with 2 on the diagonal and -1 on off-diagonals for N=10000. Find its 5 smallest eigenvalues.

**Problem 2:** Implement a 1D tight-binding chain with a single impurity (different on-site energy at one site) and find bound states.

**Problem 3:** Compare the memory usage and eigenvalue computation time for dense vs sparse representations of the same 2000×2000 tridiagonal matrix.

### Intermediate

**Problem 4:** Implement the 2D Hubbard model for a 4×4 lattice at half-filling in the Sz=0 sector using exact diagonalization.

**Problem 5:** Build a sparse Hamiltonian for a particle in a 2D box with a circular potential well and find the lowest energy states.

**Problem 6:** Implement periodic boundary conditions for the 2D tight-binding model and compute the density of states.

### Challenging

**Problem 7:** Implement the DMRG (Density Matrix Renormalization Group) algorithm for the 1D Heisenberg model.

**Problem 8:** Build the Hamiltonian for a quantum spin chain with long-range interactions (power-law decay) using sparse matrices.

**Problem 9:** Implement time evolution using Krylov subspace methods for a large sparse Hamiltonian.

---

## Computational Lab

```python
"""
Day 265 Lab: Sparse Matrices for Quantum Systems
================================================
"""

import numpy as np
from scipy import sparse
from scipy.sparse import diags, eye, kron, csr_matrix
from scipy.sparse.linalg import eigsh, eigs
import time

# [Full lab implementation]

if __name__ == "__main__":
    print("=" * 70)
    print("Day 265: Sparse Matrices for Quantum Systems")
    print("=" * 70)

    # Demonstrations...

    print("\n" + "=" * 70)
    print("Lab complete! Week review on Day 266.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `diags(d, offsets)` | Diagonal sparse matrix | `diags([a,b,c], [-1,0,1])` |
| `eye(N)` | Sparse identity | `sparse.eye(1000)` |
| `kron(A, B)` | Kronecker product | `kron(sigma_z, I)` |
| `eigsh(A, k)` | Hermitian eigenvalues | `eigsh(H, k=10, which='SA')` |
| `eigs(A, k)` | General eigenvalues | `eigs(A, k=10)` |
| `spsolve(A, b)` | Sparse linear solve | `x = spsolve(H, b)` |

### Sparse Formats

| Format | Best For |
|--------|----------|
| COO | Construction from lists |
| LIL | Incremental construction |
| DOK | Random access during build |
| CSR | Row slicing, matrix-vector |
| CSC | Column slicing |

---

## Daily Checklist

- [ ] Understand different sparse formats
- [ ] Can construct sparse matrices efficiently
- [ ] Know sparse matrix arithmetic
- [ ] Can solve sparse eigenvalue problems
- [ ] Implemented tight-binding Hamiltonian
- [ ] Understand memory/performance tradeoffs
- [ ] Completed practice problems
- [ ] Ran lab successfully

---

## Preview: Day 266

Tomorrow is the **Week 38 Review** where we integrate everything:
- Build a complete quantum simulation package
- Combine integration, ODEs, optimization, linear algebra, special functions, and sparse methods
- Capstone project: Time evolution of a wave packet in a complex potential

---

*"Sparsity is not just an optimization—it's what makes quantum simulation possible."*
