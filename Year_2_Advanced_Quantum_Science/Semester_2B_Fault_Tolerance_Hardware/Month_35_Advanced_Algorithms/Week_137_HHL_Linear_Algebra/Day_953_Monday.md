# Day 953: Linear Systems & Classical Complexity

## Week 137, Day 1 | Month 35: Advanced Quantum Algorithms

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Linear systems and classical algorithms |
| Afternoon | 2.5 hours | Problem solving: Complexity analysis |
| Evening | 2 hours | Computational lab: Classical solvers implementation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Formulate linear systems** in matrix form and understand their ubiquity in science
2. **Analyze classical direct methods** including Gaussian elimination and LU decomposition
3. **Explain iterative methods** such as conjugate gradient and their convergence properties
4. **Define the condition number** and its role in numerical stability and complexity
5. **Characterize sparse matrices** and their exploitation for efficient algorithms
6. **Establish the classical baseline** against which quantum speedups are measured

---

## Core Content

### 1. The Linear System Problem

Linear systems of equations are among the most fundamental problems in computational science. Nearly every domain—from physics simulations to machine learning to finance—reduces critical calculations to solving:

$$\boxed{Ax = b}$$

where:
- $A \in \mathbb{C}^{N \times N}$ is a known matrix
- $b \in \mathbb{C}^N$ is a known vector
- $x \in \mathbb{C}^N$ is the unknown solution vector

#### Applications Across Sciences

| Domain | Linear System Application |
|--------|---------------------------|
| Physics | Finite element analysis, electromagnetic simulation |
| Engineering | Structural analysis, circuit simulation |
| Machine Learning | Linear regression, kernel methods |
| Finance | Portfolio optimization, risk analysis |
| Computer Graphics | Rendering, fluid simulation |
| Chemistry | Electronic structure calculations |

#### Why Linear Systems Matter

The importance of linear systems stems from:
1. **Direct applications**: Many problems are inherently linear
2. **Linearization**: Nonlinear problems are often solved by iterating linear approximations
3. **Optimization**: Quadratic optimization reduces to linear systems
4. **Discretization**: PDEs become linear systems after discretization

### 2. Classical Direct Methods

#### Gaussian Elimination

The oldest and most intuitive method transforms $Ax = b$ into upper triangular form $Ux = c$, then solves by back-substitution.

**Algorithm Complexity:**
$$\boxed{T_{Gauss} = O(N^3)}$$

**Steps:**
1. Forward elimination: $O(N^3)$ operations
2. Back substitution: $O(N^2)$ operations
3. Total: $\frac{2N^3}{3} + O(N^2)$

**Numerical Stability:**
- Partial pivoting: Choose largest element in column
- Complete pivoting: Choose largest element in remaining submatrix
- Adds $O(N^2)$ comparisons but crucial for stability

#### LU Decomposition

Factor $A = LU$ where $L$ is lower triangular and $U$ is upper triangular.

**Advantage:** Once factored, solving for different $b$ vectors costs only $O(N^2)$.

$$\boxed{A = LU \implies Ly = b, \quad Ux = y}$$

**Complexity:**
- Factorization: $O(N^3)$
- Each solve: $O(N^2)$

#### Cholesky Decomposition

For symmetric positive definite (SPD) matrices:
$$A = LL^T$$

**Complexity:** $O(N^3/3)$ — half the cost of LU.

**When applicable:** Covariance matrices, Gram matrices, discretized Laplacians.

### 3. Iterative Methods

For large sparse systems, direct methods are impractical. Iterative methods construct a sequence $x_0, x_1, x_2, \ldots$ converging to the solution.

#### Jacobi Method

Split $A = D + R$ where $D$ is diagonal:
$$x^{(k+1)} = D^{-1}(b - Rx^{(k)})$$

**Complexity per iteration:** $O(N \cdot s)$ where $s$ = non-zeros per row

**Convergence:** Requires $\rho(D^{-1}R) < 1$ (spectral radius condition)

#### Gauss-Seidel Method

Use updated values immediately:
$$x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j<i} a_{ij}x_j^{(k+1)} - \sum_{j>i} a_{ij}x_j^{(k)}\right)$$

**Convergence:** Often faster than Jacobi, guaranteed for SPD matrices.

#### Conjugate Gradient Method

The gold standard for SPD systems. Minimizes $\phi(x) = \frac{1}{2}x^TAx - b^Tx$ over Krylov subspaces.

**Algorithm:**
```
r_0 = b - Ax_0
p_0 = r_0
for k = 0, 1, 2, ...
    α_k = (r_k^T r_k) / (p_k^T A p_k)
    x_{k+1} = x_k + α_k p_k
    r_{k+1} = r_k - α_k A p_k
    β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
    p_{k+1} = r_{k+1} + β_k p_k
```

**Complexity:**
$$\boxed{T_{CG} = O(N \cdot s \cdot \sqrt{\kappa} \cdot \log(1/\epsilon))}$$

where:
- $s$ = sparsity (non-zeros per row)
- $\kappa$ = condition number
- $\epsilon$ = target accuracy

**Convergence guarantee:** For SPD matrices, terminates in at most $N$ iterations (exact arithmetic).

### 4. The Condition Number

The condition number is the single most important parameter characterizing linear system difficulty.

#### Definition

For a non-singular matrix $A$:
$$\boxed{\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\lambda_{max}}{\lambda_{min}}}$$

(using 2-norm, equals ratio of largest to smallest singular values)

#### Interpretation

| $\kappa$ | Matrix Type | Numerical Behavior |
|----------|-------------|-------------------|
| $\kappa \approx 1$ | Well-conditioned | Stable, fast convergence |
| $\kappa \sim 10^3$ | Moderate | Some precision loss |
| $\kappa \sim 10^6$ | Ill-conditioned | Significant errors |
| $\kappa \to \infty$ | Singular | Unsolvable |

#### Error Amplification

For perturbation $\delta b$ in the right-hand side:
$$\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \cdot \frac{\|\delta b\|}{\|b\|}$$

**Critical insight:** High condition numbers amplify errors proportionally.

#### Condition Number Examples

| Matrix Type | Typical $\kappa$ |
|-------------|------------------|
| Orthogonal | 1 |
| Diagonal (varied) | $\max|d_i|/\min|d_i|$ |
| Hilbert matrix | $O(e^{3.5N})$ |
| Discretized Laplacian | $O(N^2)$ |
| Random dense | $O(N)$ |

### 5. Sparse Matrix Structure

Real-world matrices are often sparse—most entries are zero. Exploiting sparsity is essential for large-scale computation.

#### Sparsity Definitions

**Sparsity parameter $s$:** Maximum non-zeros per row (or column).

$$\text{Dense: } s = N \qquad \text{Sparse: } s \ll N$$

#### Common Sparse Structures

| Structure | Description | Example |
|-----------|-------------|---------|
| Diagonal | Only main diagonal | Mass matrices |
| Tridiagonal | 3 diagonals | 1D finite differences |
| Banded | $2k+1$ diagonals | Higher-order FD |
| Block diagonal | Diagonal blocks | Independent subsystems |
| Sparse random | Random non-zero pattern | Network graphs |

#### Sparse Storage Formats

**Compressed Sparse Row (CSR):**
- `values`: Non-zero values
- `col_indices`: Column indices
- `row_pointers`: Start of each row

**Storage:** $O(N \cdot s)$ instead of $O(N^2)$

**Matrix-vector multiply:** $O(N \cdot s)$ instead of $O(N^2)$

### 6. Complexity Summary

The classical landscape for solving $Ax = b$:

| Method | Time Complexity | Space | Requirements |
|--------|-----------------|-------|--------------|
| Gaussian Elimination | $O(N^3)$ | $O(N^2)$ | General |
| LU Decomposition | $O(N^3)$ | $O(N^2)$ | General |
| Cholesky | $O(N^3/3)$ | $O(N^2)$ | SPD |
| Jacobi/Gauss-Seidel | $O(N \cdot s \cdot k)$ | $O(N \cdot s)$ | Convergent |
| Conjugate Gradient | $O(N \cdot s \cdot \sqrt{\kappa})$ | $O(N \cdot s)$ | SPD |
| GMRES | $O(N \cdot s \cdot k^2)$ | $O(N \cdot k)$ | General |

**Key observations:**
1. Direct methods: $O(N^3)$ unavoidable for dense matrices
2. Iterative methods: $O(N \cdot s \cdot f(\kappa))$ exploiting sparsity
3. Condition number appears in all iterative complexity bounds

### 7. The Quantum Opportunity

Given the classical complexity landscape, where can quantum computing help?

**HHL Promise:**
$$T_{HHL} = O(\log(N) \cdot s^2 \cdot \kappa^2 / \epsilon)$$

**Comparison to Conjugate Gradient:**
$$\frac{T_{CG}}{T_{HHL}} = \frac{O(N \cdot s \cdot \sqrt{\kappa})}{O(\log N \cdot s^2 \cdot \kappa^2)} = O\left(\frac{N}{\log N} \cdot \frac{1}{s \cdot \kappa^{3/2}}\right)$$

**Exponential speedup in $N$** — but:
- Worse dependence on $\kappa$
- Requires quantum state input/output
- Sparsity factor different

This week, we'll explore when this quantum advantage materializes.

---

## Worked Examples

### Example 1: Condition Number Calculation

**Problem:** Find the condition number of:
$$A = \begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix}$$

**Solution:**

Step 1: Find eigenvalues (A is symmetric, so $\sigma_i = |\lambda_i|$)

Characteristic polynomial:
$$\det(A - \lambda I) = (4-\lambda)(5-\lambda) - 4 = \lambda^2 - 9\lambda + 16$$

$$\lambda = \frac{9 \pm \sqrt{81-64}}{2} = \frac{9 \pm \sqrt{17}}{2}$$

$$\lambda_1 = \frac{9 + \sqrt{17}}{2} \approx 6.56, \quad \lambda_2 = \frac{9 - \sqrt{17}}{2} \approx 2.44$$

Step 2: Condition number
$$\kappa(A) = \frac{\lambda_{max}}{\lambda_{min}} = \frac{6.56}{2.44} \approx \boxed{2.69}$$

**Interpretation:** Well-conditioned matrix; errors amplified by factor ~2.7.

---

### Example 2: Conjugate Gradient Iteration Count

**Problem:** Estimate iterations for CG on a 10,000 × 10,000 discretized 2D Laplacian to achieve $10^{-8}$ relative residual.

**Solution:**

Step 1: Condition number of 2D Laplacian on $n \times n$ grid:
$$\kappa \approx \frac{4n^2/\pi^2}{4/\pi^2} = n^2$$

For $N = n^2 = 10,000$, we have $n = 100$, so $\kappa \approx 10,000$.

Step 2: CG convergence bound:
$$\|e_k\| \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|e_0\|$$

We need:
$$\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \leq 10^{-8}$$

Step 3: Solve for $k$:
$$\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} = \frac{100-1}{100+1} = \frac{99}{101} \approx 0.98$$

$$k \cdot \log(0.98) \leq -8\log(10)$$
$$k \geq \frac{8 \times 2.303}{0.0202} \approx 912$$

$$\boxed{k \approx 900-1000 \text{ iterations}}$$

**Note:** With preconditioning, this can be reduced to $O(\sqrt[4]{\kappa}) \approx 10$ iterations.

---

### Example 3: Sparsity and Complexity

**Problem:** Compare the time to solve a banded system (bandwidth 5) vs. dense system, both $N = 10^6$.

**Solution:**

**Dense system (LU):**
$$T_{dense} = O(N^3) = O(10^{18}) \text{ operations}$$

At $10^{12}$ ops/sec: $\sim 10^6$ seconds $\approx 11.5$ days

**Banded system (band LU):**
$$T_{banded} = O(N \cdot w^2) = O(10^6 \times 25) = O(2.5 \times 10^7)$$

At $10^{12}$ ops/sec: $\sim 25$ microseconds

**Speedup:** $\boxed{4 \times 10^{10}}$ (40 billion times faster!)

**Lesson:** Structure exploitation is the key to practical linear algebra.

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the condition number of:
$$A = \begin{pmatrix} 10 & 1 \\ 1 & 10 \end{pmatrix}$$

**Problem 1.2:** How many operations for LU decomposition of a 5000 × 5000 matrix?

**Problem 1.3:** A tridiagonal matrix has sparsity $s = 3$. How many non-zeros in an $N \times N$ tridiagonal matrix?

### Level 2: Intermediate Analysis

**Problem 2.1:** The 5×5 Hilbert matrix has entries $H_{ij} = 1/(i+j-1)$.
- Compute its condition number numerically
- Explain why this makes it difficult to solve $Hx = b$

**Problem 2.2:** Prove that conjugate gradient converges in at most $N$ steps for an $N \times N$ SPD matrix (assuming exact arithmetic).

**Problem 2.3:** Derive the operation count for tridiagonal LU decomposition (Thomas algorithm) and show it's $O(N)$.

### Level 3: Challenging Problems

**Problem 3.1:** **Condition Number Sensitivity**

For the matrix family:
$$A(\epsilon) = \begin{pmatrix} 1 & 1 \\ 1 & 1+\epsilon \end{pmatrix}$$

Find $\kappa(A(\epsilon))$ as a function of $\epsilon$ and analyze the limit as $\epsilon \to 0$.

**Problem 3.2:** **Optimal Preconditioning**

If $M$ is a preconditioner for $A$, the preconditioned system is $M^{-1}Ax = M^{-1}b$.
- Show that $\kappa(M^{-1}A) = 1$ when $M = A$
- Explain why this isn't practical
- What properties should a good preconditioner have?

**Problem 3.3:** **Quantum vs Classical Crossover**

Given:
- Classical CG: $T_{CG} = c_1 \cdot N \cdot s \cdot \sqrt{\kappa}$
- Quantum HHL: $T_{HHL} = c_2 \cdot \log(N) \cdot s^2 \cdot \kappa^2 / \epsilon$

For $s = 10$, $\kappa = 100$, $\epsilon = 0.01$:
- At what $N$ does HHL become faster (assuming $c_1 = c_2$)?
- How does this crossover point change with $\kappa$?

---

## Computational Lab

### Classical Linear System Solvers

```python
"""
Day 953: Classical Linear System Solvers
Comprehensive implementation and comparison of classical methods.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Callable
import matplotlib.pyplot as plt
import time

class LinearSystemAnalyzer:
    """Analyze and solve linear systems with various methods."""

    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        Initialize with system Ax = b.

        Parameters:
        -----------
        A : ndarray
            System matrix (N x N)
        b : ndarray
            Right-hand side vector (N,)
        """
        self.A = A
        self.b = b
        self.N = A.shape[0]

        # Compute matrix properties
        self.eigenvalues = None
        self.condition_number = None
        self.sparsity = None

    def analyze_matrix(self) -> dict:
        """Compute key matrix properties."""
        # Eigenvalues (for moderate-sized matrices)
        if self.N <= 1000:
            self.eigenvalues = np.linalg.eigvalsh(self.A) if self._is_symmetric() else np.linalg.eigvals(self.A)
            lambda_max = np.max(np.abs(self.eigenvalues))
            lambda_min = np.min(np.abs(self.eigenvalues))
            self.condition_number = lambda_max / lambda_min if lambda_min > 0 else np.inf
        else:
            # Use SVD for condition number
            self.condition_number = np.linalg.cond(self.A)

        # Sparsity
        self.sparsity = np.count_nonzero(self.A) / (self.N ** 2)

        return {
            'dimension': self.N,
            'condition_number': self.condition_number,
            'sparsity': self.sparsity,
            'is_symmetric': self._is_symmetric(),
            'is_positive_definite': self._is_positive_definite(),
            'max_eigenvalue': np.max(np.abs(self.eigenvalues)) if self.eigenvalues is not None else None,
            'min_eigenvalue': np.min(np.abs(self.eigenvalues)) if self.eigenvalues is not None else None
        }

    def _is_symmetric(self) -> bool:
        """Check if matrix is symmetric."""
        return np.allclose(self.A, self.A.T)

    def _is_positive_definite(self) -> bool:
        """Check if matrix is positive definite."""
        if not self._is_symmetric():
            return False
        try:
            np.linalg.cholesky(self.A)
            return True
        except np.linalg.LinAlgError:
            return False

    def solve_direct(self, method: str = 'lu') -> Tuple[np.ndarray, float]:
        """
        Solve using direct method.

        Parameters:
        -----------
        method : str
            'lu' for LU decomposition, 'cholesky' for Cholesky (SPD only)

        Returns:
        --------
        x : ndarray
            Solution vector
        time_elapsed : float
            Computation time in seconds
        """
        start = time.time()

        if method == 'lu':
            x = la.solve(self.A, self.b)
        elif method == 'cholesky':
            if not self._is_positive_definite():
                raise ValueError("Cholesky requires positive definite matrix")
            L = la.cholesky(self.A, lower=True)
            y = la.solve_triangular(L, self.b, lower=True)
            x = la.solve_triangular(L.T, y, lower=False)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start
        return x, elapsed

    def solve_iterative(self, method: str = 'cg', tol: float = 1e-10,
                        maxiter: int = None) -> Tuple[np.ndarray, float, int]:
        """
        Solve using iterative method.

        Parameters:
        -----------
        method : str
            'cg' (conjugate gradient), 'gmres', 'jacobi', 'gauss_seidel'
        tol : float
            Convergence tolerance
        maxiter : int
            Maximum iterations

        Returns:
        --------
        x : ndarray
            Solution vector
        time_elapsed : float
            Computation time
        iterations : int
            Number of iterations
        """
        maxiter = maxiter or self.N
        start = time.time()
        iterations = [0]  # Use list to allow modification in callback

        def count_iter(xk):
            iterations[0] += 1

        if method == 'cg':
            if not self._is_positive_definite():
                print("Warning: CG requires SPD matrix")
            A_sparse = sp.csr_matrix(self.A)
            x, info = spla.cg(A_sparse, self.b, tol=tol, maxiter=maxiter, callback=count_iter)

        elif method == 'gmres':
            A_sparse = sp.csr_matrix(self.A)
            x, info = spla.gmres(A_sparse, self.b, tol=tol, maxiter=maxiter, callback=count_iter)

        elif method == 'jacobi':
            x, iterations[0] = self._jacobi(tol, maxiter)

        elif method == 'gauss_seidel':
            x, iterations[0] = self._gauss_seidel(tol, maxiter)

        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start
        return x, elapsed, iterations[0]

    def _jacobi(self, tol: float, maxiter: int) -> Tuple[np.ndarray, int]:
        """Jacobi iteration."""
        D_inv = 1.0 / np.diag(self.A)
        R = self.A - np.diag(np.diag(self.A))
        x = np.zeros(self.N)

        for k in range(maxiter):
            x_new = D_inv * (self.b - R @ x)
            if np.linalg.norm(x_new - x) < tol:
                return x_new, k + 1
            x = x_new

        return x, maxiter

    def _gauss_seidel(self, tol: float, maxiter: int) -> Tuple[np.ndarray, int]:
        """Gauss-Seidel iteration."""
        x = np.zeros(self.N)

        for k in range(maxiter):
            x_old = x.copy()
            for i in range(self.N):
                sigma = np.dot(self.A[i, :i], x[:i]) + np.dot(self.A[i, i+1:], x[i+1:])
                x[i] = (self.b[i] - sigma) / self.A[i, i]

            if np.linalg.norm(x - x_old) < tol:
                return x, k + 1

        return x, maxiter

    def compare_methods(self) -> dict:
        """Compare all applicable methods."""
        results = {}

        # Direct methods
        x_lu, t_lu = self.solve_direct('lu')
        results['lu'] = {'time': t_lu, 'residual': np.linalg.norm(self.A @ x_lu - self.b)}

        if self._is_positive_definite():
            x_chol, t_chol = self.solve_direct('cholesky')
            results['cholesky'] = {'time': t_chol, 'residual': np.linalg.norm(self.A @ x_chol - self.b)}

        # Iterative methods
        try:
            x_cg, t_cg, it_cg = self.solve_iterative('cg')
            results['cg'] = {'time': t_cg, 'iterations': it_cg,
                           'residual': np.linalg.norm(self.A @ x_cg - self.b)}
        except:
            pass

        x_gmres, t_gmres, it_gmres = self.solve_iterative('gmres')
        results['gmres'] = {'time': t_gmres, 'iterations': it_gmres,
                          'residual': np.linalg.norm(self.A @ x_gmres - self.b)}

        return results


def create_test_matrices(N: int) -> dict:
    """Create various test matrices for benchmarking."""
    matrices = {}

    # Random SPD matrix
    A_rand = np.random.randn(N, N)
    matrices['random_spd'] = A_rand @ A_rand.T + N * np.eye(N)

    # Hilbert matrix (ill-conditioned)
    matrices['hilbert'] = np.array([[1/(i+j+1) for j in range(N)] for i in range(N)])

    # Tridiagonal (well-conditioned)
    matrices['tridiagonal'] = (np.diag(4*np.ones(N))
                               - np.diag(np.ones(N-1), 1)
                               - np.diag(np.ones(N-1), -1))

    # 2D Laplacian discretization
    n = int(np.sqrt(N))
    if n*n == N:
        matrices['laplacian_2d'] = create_2d_laplacian(n)

    return matrices


def create_2d_laplacian(n: int) -> np.ndarray:
    """Create 2D Laplacian matrix for n x n grid."""
    N = n * n
    A = np.zeros((N, N))

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = 4
            if i > 0:
                A[idx, idx - n] = -1
            if i < n - 1:
                A[idx, idx + n] = -1
            if j > 0:
                A[idx, idx - 1] = -1
            if j < n - 1:
                A[idx, idx + 1] = -1

    return A


def condition_number_scaling():
    """Analyze condition number scaling for different matrix types."""
    sizes = [10, 20, 50, 100, 200]

    results = {
        'hilbert': [],
        'laplacian': [],
        'random': []
    }

    for N in sizes:
        # Hilbert
        H = np.array([[1/(i+j+1) for j in range(N)] for i in range(N)])
        results['hilbert'].append(np.linalg.cond(H))

        # 2D Laplacian (N must be perfect square)
        n = int(np.sqrt(N))
        L = create_2d_laplacian(n)
        results['laplacian'].append(np.linalg.cond(L))

        # Random SPD
        R = np.random.randn(N, N)
        R = R @ R.T + N * np.eye(N)
        results['random'].append(np.linalg.cond(R))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(sizes, results['hilbert'], 'ro-', label='Hilbert (ill-conditioned)', linewidth=2)
    ax.semilogy(sizes, results['laplacian'], 'bs-', label='2D Laplacian', linewidth=2)
    ax.semilogy(sizes, results['random'], 'g^-', label='Random SPD', linewidth=2)

    ax.set_xlabel('Matrix Size N', fontsize=12)
    ax.set_ylabel('Condition Number κ', fontsize=12)
    ax.set_title('Condition Number Scaling by Matrix Type', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('condition_number_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


def benchmark_solvers():
    """Benchmark solver performance across matrix sizes."""
    sizes = [100, 200, 500, 1000]

    times = {
        'lu': [],
        'cg': [],
        'gmres': []
    }

    for N in sizes:
        print(f"\nBenchmarking N = {N}...")

        # Create SPD test matrix
        A = np.random.randn(N, N)
        A = A @ A.T + N * np.eye(N)
        b = np.random.randn(N)

        analyzer = LinearSystemAnalyzer(A, b)

        # LU
        _, t_lu = analyzer.solve_direct('lu')
        times['lu'].append(t_lu)

        # CG
        _, t_cg, _ = analyzer.solve_iterative('cg')
        times['cg'].append(t_cg)

        # GMRES
        _, t_gmres, _ = analyzer.solve_iterative('gmres')
        times['gmres'].append(t_gmres)

        print(f"  LU: {t_lu:.4f}s, CG: {t_cg:.4f}s, GMRES: {t_gmres:.4f}s")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(sizes, times['lu'], 'ro-', label='LU (direct)', linewidth=2, markersize=8)
    ax.loglog(sizes, times['cg'], 'bs-', label='Conjugate Gradient', linewidth=2, markersize=8)
    ax.loglog(sizes, times['gmres'], 'g^-', label='GMRES', linewidth=2, markersize=8)

    # Add O(N^3) reference line
    N_ref = np.array(sizes)
    ax.loglog(N_ref, (N_ref/100)**3 * times['lu'][0], 'k--', alpha=0.5, label='O(N³) reference')

    ax.set_xlabel('Matrix Size N', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Solver Performance Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('solver_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()

    return times


def cg_convergence_demo():
    """Demonstrate CG convergence for different condition numbers."""
    N = 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for kappa, color in [(10, 'blue'), (100, 'green'), (1000, 'red')]:
        # Create matrix with specified condition number
        eigenvalues = np.linspace(1, kappa, N)
        Q, _ = np.linalg.qr(np.random.randn(N, N))
        A = Q @ np.diag(eigenvalues) @ Q.T

        b = np.random.randn(N)
        x_exact = np.linalg.solve(A, b)

        # Run CG with tracking
        x = np.zeros(N)
        r = b - A @ x
        p = r.copy()
        residuals = [np.linalg.norm(r)]
        errors = [np.linalg.norm(x - x_exact)]

        for k in range(min(200, N)):
            Ap = A @ p
            alpha = np.dot(r, r) / np.dot(p, Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap

            residuals.append(np.linalg.norm(r_new))
            errors.append(np.linalg.norm(x - x_exact))

            if np.linalg.norm(r_new) < 1e-12:
                break

            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new

        axes[0].semilogy(residuals, color=color, label=f'κ = {kappa}', linewidth=2)
        axes[1].semilogy(errors, color=color, label=f'κ = {kappa}', linewidth=2)

    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Residual ||r||', fontsize=12)
    axes[0].set_title('CG Residual Convergence', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Error ||x - x*||', fontsize=12)
    axes[1].set_title('CG Error Convergence', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cg_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main demonstration
if __name__ == "__main__":
    print("Day 953: Classical Linear System Solvers")
    print("=" * 50)

    # Example analysis
    print("\n--- Example: Analyze 100x100 SPD Matrix ---")
    np.random.seed(42)
    N = 100
    A = np.random.randn(N, N)
    A = A @ A.T + N * np.eye(N)
    b = np.random.randn(N)

    analyzer = LinearSystemAnalyzer(A, b)
    props = analyzer.analyze_matrix()

    print(f"Dimension: {props['dimension']}")
    print(f"Condition number: {props['condition_number']:.2e}")
    print(f"Sparsity: {props['sparsity']:.2%}")
    print(f"Is symmetric: {props['is_symmetric']}")
    print(f"Is positive definite: {props['is_positive_definite']}")

    # Compare methods
    print("\n--- Method Comparison ---")
    results = analyzer.compare_methods()
    for method, data in results.items():
        print(f"{method:12s}: time = {data['time']:.4f}s, residual = {data['residual']:.2e}")

    # Run visualizations
    print("\n--- Generating Visualizations ---")
    condition_number_scaling()
    benchmark_solvers()
    cg_convergence_demo()
```

---

## Summary

### Key Formulas

| Formula | Expression | Context |
|---------|------------|---------|
| System equation | $Ax = b$ | Fundamental problem |
| Direct complexity | $O(N^3)$ | Gaussian elimination/LU |
| CG complexity | $O(N \cdot s \cdot \sqrt{\kappa})$ | Sparse SPD systems |
| Condition number | $\kappa = \|A\| \cdot \|A^{-1}\|$ | Numerical stability |
| Error bound | $\frac{\|\delta x\|}{\|x\|} \leq \kappa \frac{\|\delta b\|}{\|b\|}$ | Error amplification |

### Complexity Hierarchy

$$O(N^3) \xrightarrow{\text{sparsity}} O(N \cdot s \cdot k) \xrightarrow{\text{structure}} O(N) \xrightarrow{\text{quantum}} O(\log N)$$

### Key Insights

1. **Classical direct methods scale cubically** — impractical for $N > 10^5$
2. **Iterative methods exploit sparsity** — essential for large-scale problems
3. **Condition number determines everything** — high $\kappa$ means slow convergence and errors
4. **Preconditioning transforms the problem** — reduces effective condition number
5. **HHL promises exponential speedup in N** — but with important caveats

---

## Daily Checklist

- [ ] I can formulate problems as linear systems
- [ ] I understand Gaussian elimination and LU decomposition
- [ ] I can explain conjugate gradient and its convergence
- [ ] I can calculate and interpret condition numbers
- [ ] I understand sparse matrix storage and operations
- [ ] I can compare classical algorithm complexities
- [ ] I recognize the quantum opportunity and its requirements

---

## Preview: Day 954

Tomorrow we review **Quantum Phase Estimation (QPE)**, the key subroutine at the heart of HHL. We'll cover:

- QPE circuit architecture and ancilla qubit requirements
- Precision analysis: how $n$ ancilla qubits give $2^{-n}$ eigenvalue precision
- Success probability and the role of eigenvalue gaps
- Controlled unitary operations and Hamiltonian simulation connection
- Setting up QPE for eigenvalue inversion in HHL

Understanding QPE deeply is essential—it's the engine that extracts eigenvalues, enabling the controlled rotation that inverts them.

---

*Day 953 of 2184 | Week 137 of 312 | Month 35 of 72*

*"All of computational science is, at its core, about solving Ax = b efficiently."*
