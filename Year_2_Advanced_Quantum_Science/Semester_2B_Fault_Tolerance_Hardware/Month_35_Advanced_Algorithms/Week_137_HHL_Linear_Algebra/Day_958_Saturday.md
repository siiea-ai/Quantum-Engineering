# Day 958: Dequantization & Classical Competition

## Week 137, Day 6 | Month 35: Advanced Quantum Algorithms

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Tang's algorithm and dequantization |
| Afternoon | 2.5 hours | Problem solving: When classical suffices |
| Evening | 2 hours | Computational lab: Quantum-inspired algorithms |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Explain Tang's 2018 dequantization breakthrough** and its implications
2. **Understand sampling and query access** models for classical algorithms
3. **Identify when HHL loses its advantage** to classical methods
4. **Analyze the structure required for quantum speedup**
5. **Compare quantum-inspired vs genuine quantum algorithms**
6. **Navigate the refined landscape** of quantum linear algebra advantage

---

## Core Content

### 1. The Dequantization Revolution

In 2018, Ewin Tang (then an undergraduate) proved a shocking result: for many machine learning applications, classical algorithms can match HHL's performance.

#### The Context

Prior belief: HHL provides exponential speedup for linear algebra, enabling fast quantum machine learning.

**Kerenidis-Prakash (2017):** Quantum recommendation algorithm with exponential speedup over classical.

**Tang (2018):** Classical algorithm matching the quantum speedup!

This sparked a wave of "dequantization" results, fundamentally reshaping our understanding of quantum advantage.

### 2. Sampling Access Model

The key insight: HHL's advantage relies on specific input/output assumptions. When those assumptions hold classically, classical algorithms can be fast too.

#### SQ (Sample and Query) Access

A data structure providing **sample and query access** to a vector $v \in \mathbb{R}^n$:

**Query:** Given index $i$, return $v_i$ in $O(1)$ time.

**Sample:** Return index $i$ with probability $|v_i|^2/\|v\|^2$ in $O(1)$ time.

**Norm query:** Return $\|v\|$ in $O(1)$ time.

#### SQ Access to Matrices

For matrix $A \in \mathbb{R}^{m \times n}$:

**Row sampling:** Sample row $i$ proportional to $\|A_i\|^2$

**Entry sampling:** Within row $i$, sample column $j$ proportional to $|A_{ij}|^2$

**Frobenius norm:** Return $\|A\|_F$ in $O(1)$ time

**Key observation:** This is precisely what qRAM would provide!

### 3. Tang's Algorithm

#### Problem: Matrix Inversion with Sampling Output

**Input:** SQ access to $A$ and $b$
**Output:** Sample from $x = A^{-1}b$

**Tang's result:** Classical algorithm achieving:
$$\boxed{T = O\left(\text{poly}\left(\|A\|_F^6, \kappa^6, 1/\epsilon\right)\right)}$$

independent of matrix dimension $n$!

#### Key Technique: Randomized Linear Algebra

**Leverage score sampling:** Sample rows/columns based on their "importance"

**Low-rank approximation:** If $A \approx A_k$ (rank-$k$ approximation), work with $A_k$

**Monte Carlo estimation:** Estimate inner products via sampling

### 4. When Does Dequantization Apply?

#### Conditions for Classical Matching

Dequantization works when:

1. **SQ access available:** Data structure supports fast sampling
2. **Low effective rank:** Matrix has good low-rank approximation
3. **Sampling output sufficient:** Don't need full solution vector

#### Formal Statement

For a matrix $A$ with Frobenius norm $\|A\|_F$ and condition number $\kappa$:

If we have SQ access and want to sample from $A^{-1}b$:

| Algorithm | Complexity |
|-----------|------------|
| HHL | $O(\text{poly}(\log n, \kappa, 1/\epsilon))$ |
| Tang | $O(\text{poly}(\|A\|_F, \kappa, 1/\epsilon))$ |

**No dependence on $n$** in both cases!

### 5. The Structural Requirements

#### HHL Still Wins When:

1. **Full state output needed:** Quantum parallelism outputs all components
2. **No low-rank structure:** Matrix is genuinely high-rank
3. **Quantum input already available:** No classical data loading
4. **Specific problem structure:** Some problems have inherent quantum speedups

#### Genuine Quantum Advantage Scenarios

| Scenario | Why Quantum Wins |
|----------|------------------|
| Quantum simulation | Input is quantum state |
| Cryptographic | Structured number theory |
| Unstructured search | Grover's optimality |
| Certain optimization | QAOA, phase transitions |

### 6. Quantum-Inspired Algorithms

The dequantization work spawned a new field: **quantum-inspired classical algorithms**.

#### Key Examples

| Algorithm | Quantum Original | Classical Matching |
|-----------|------------------|-------------------|
| Recommendation | Kerenidis-Prakash | Tang (2018) |
| Matrix inversion | HHL | Gilyén et al. (2018) |
| PCA | Lloyd et al. | Tang (2018) |
| SVM | Rebentrost et al. | Chia et al. (2019) |
| Regression | Harrow-Montanaro | Multiple |

#### Common Techniques

1. **Randomized sampling:** Replace quantum superposition with random samples
2. **Sketching:** Compress high-dimensional data
3. **Approximate matrix operations:** Work with low-rank approximations
4. **Importance sampling:** Focus computation on significant components

### 7. The Refined Landscape

#### Updated Quantum Advantage Assessment

| Application | Pre-2018 Belief | Post-2018 Reality |
|-------------|-----------------|-------------------|
| Recommendation | Exponential speedup | None (matched) |
| ML classification | Exponential speedup | Conditional |
| Solving PDEs | Exponential speedup | Depends on structure |
| Optimization | Polynomial speedup | Active research |
| Chemistry simulation | Exponential speedup | **Likely genuine** |
| Cryptography | Exponential speedup | **Genuine** |

### 8. What Remains for HHL?

Despite dequantization, HHL remains valuable:

#### Genuine Applications

1. **Quantum chemistry/materials:** Input from quantum simulation
2. **Quantum-to-quantum computation:** HHL as subroutine
3. **Specific structured problems:** Where sampling access is hard classically
4. **End-to-end quantum pipelines:** No classical data conversion

#### The "Right" Problem for HHL

The ideal HHL application:
- Matrix derived from quantum process
- Output feeds into another quantum algorithm
- High-rank, no obvious low-rank structure
- Expectation values needed, not full solution

### 9. Complexity Landscape Summary

$$\boxed{\text{Classical (SQ)} \approx \text{Quantum (qRAM)} \text{ for many problems}}$$

But:
$$\text{Classical (general)} \ll \text{Quantum} \text{ for structured problems}$$

The lesson: **quantum advantage requires careful analysis of problem structure**.

---

## Worked Examples

### Example 1: Low-Rank Approximation

**Problem:** Matrix $A \in \mathbb{R}^{1000 \times 1000}$ has singular values $\sigma_i = 1/(i+1)$ for $i = 0, \ldots, 999$. How well can a rank-10 approximation capture $A$?

**Solution:**

Step 1: Frobenius norm
$$\|A\|_F^2 = \sum_{i=0}^{999} \sigma_i^2 = \sum_{i=1}^{1000} \frac{1}{i^2} \approx \frac{\pi^2}{6} \approx 1.645$$

Step 2: Rank-10 approximation captures
$$\|A_{10}\|_F^2 = \sum_{i=0}^{9} \frac{1}{(i+1)^2} = 1 + 0.25 + 0.111 + \cdots \approx 1.549$$

Step 3: Fraction captured
$$\frac{\|A_{10}\|_F^2}{\|A\|_F^2} \approx \frac{1.549}{1.645} \approx 94.2\%$$

$$\boxed{\text{Rank-10 captures 94\% of Frobenius norm}}$$

**Implication:** Tang's algorithm works well—the effective dimension is ~10, not 1000!

---

### Example 2: When Tang Fails

**Problem:** Matrix $A = I_{1000}$ (identity). Can Tang's algorithm solve $Ax = b$ efficiently?

**Solution:**

Step 1: Analyze structure
- $\|A\|_F = \sqrt{1000} \approx 31.6$
- Condition number $\kappa = 1$
- Rank = 1000 (full rank)

Step 2: Tang complexity
$$T_{Tang} = O(\|A\|_F^6 \cdot \kappa^6) = O(31.6^6) = O(10^9)$$

Step 3: Direct solution
$$x = A^{-1}b = b$$

Cost: $O(1)$ since $A = I$!

**Lesson:** Dequantization's $\|A\|_F$ dependence can be worse than dimension dependence for "flat" matrices.

$$\boxed{\text{Tang: } O(10^9) \text{ vs Direct: } O(1)}$$

---

### Example 3: Comparing Regimes

**Problem:** Compare HHL, Tang, and Conjugate Gradient for:
- $A$: $10^6 \times 10^6$ sparse matrix
- Effective rank: 100
- $\kappa = 50$
- Sparsity $s = 5$

**Solution:**

Step 1: HHL complexity
$$T_{HHL} = O(\log(N) \cdot s^2 \cdot \kappa^2 / \epsilon) = O(20 \cdot 25 \cdot 2500 / 0.01) = O(1.25 \times 10^8)$$

Step 2: Tang complexity (with $\|A\|_F \approx \sqrt{\text{rank}} \cdot \|A\|_2$)
Assuming $\|A\|_F \approx 100$:
$$T_{Tang} = O(\|A\|_F^6 \cdot \kappa^6 / \epsilon) = O(10^{12} \cdot 10^{10} / 0.01) = O(10^{24})$$

Wait—this is huge! But if effective rank is 100, we should use that:
$$T_{Tang} \approx O(\text{rank}^3 \cdot \kappa^6 / \epsilon) = O(10^6 \cdot 10^{10} / 0.01) = O(10^{18})$$

Step 3: Conjugate Gradient
$$T_{CG} = O(N \cdot s \cdot \sqrt{\kappa} \cdot \log(1/\epsilon)) = O(10^6 \cdot 5 \cdot 7 \cdot 7) = O(2.5 \times 10^8)$$

$$\boxed{\text{CG: } 2.5 \times 10^8 < \text{HHL: } 1.25 \times 10^8 < \text{Tang: } 10^{18}}$$

**Conclusion:** For this problem, CG and HHL are comparable; Tang's general bound is loose but applies without sparsity assumption.

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** A matrix has singular values uniformly distributed from 1 to 100. What is its condition number and approximate Frobenius norm?

**Problem 1.2:** Define SQ (sample and query) access. What classical data structure provides this?

**Problem 1.3:** If Tang's algorithm has complexity $O(\|A\|_F^6)$, when is this better than $O(N)$?

### Level 2: Intermediate Analysis

**Problem 2.1:** Prove that for a rank-$r$ matrix $A \in \mathbb{R}^{n \times n}$:
$$\|A\|_F \leq \sqrt{r} \cdot \|A\|_2$$

**Problem 2.2:** Design a data structure that provides SQ access to a sparse matrix stored in CSR format. What is the preprocessing cost?

**Problem 2.3:** Compare the output of HHL (quantum state $|x\rangle$) vs Tang (sample from $x$). In what scenarios are these equivalent? Different?

### Level 3: Challenging Problems

**Problem 3.1:** **Dequantization Limits**

Prove that if SQ access requires $\Omega(N)$ preprocessing, then Tang's algorithm doesn't beat classical $O(N)$ methods.

**Problem 3.2:** **Hybrid Strategy**

Design an algorithm that uses Tang's method when the matrix is low-rank and HHL when it's not. What's the crossover point?

**Problem 3.3:** **Lower Bounds**

Consider a random $N \times N$ matrix. What is the expected rank needed to capture 99% of the Frobenius norm? How does this scale with $N$?

---

## Computational Lab

### Quantum-Inspired Algorithms

```python
"""
Day 958: Quantum-Inspired Classical Algorithms
Implementing dequantization techniques.
"""

import numpy as np
from typing import Tuple, Optional, Callable
import matplotlib.pyplot as plt
from scipy.linalg import svd
import time


class SampleQueryAccess:
    """
    Implement Sample and Query (SQ) access to a matrix.

    This is the classical analog of qRAM.
    """

    def __init__(self, matrix: np.ndarray):
        """
        Initialize with matrix A.

        Preprocessing: O(nnz) where nnz = number of non-zeros.
        """
        self.A = matrix
        self.m, self.n = matrix.shape

        # Precompute row norms
        self.row_norms = np.linalg.norm(matrix, axis=1)

        # Precompute Frobenius norm
        self.frobenius_norm = np.linalg.norm(matrix, 'fro')

        # Build sampling data structures
        self._build_samplers()

    def _build_samplers(self):
        """Build data structures for efficient sampling."""
        # Row sampling probabilities
        self.row_probs = self.row_norms**2 / self.frobenius_norm**2

        # Within-row sampling (cumulative for each row)
        self.col_samplers = []
        for i in range(self.m):
            if self.row_norms[i] > 0:
                probs = np.abs(self.A[i, :])**2 / self.row_norms[i]**2
            else:
                probs = np.ones(self.n) / self.n
            self.col_samplers.append(probs)

    def query(self, i: int, j: int) -> float:
        """Query entry A[i,j]. O(1) time."""
        return self.A[i, j]

    def sample_row(self) -> int:
        """Sample row index proportional to ||A_i||^2. O(1) time."""
        return np.random.choice(self.m, p=self.row_probs)

    def sample_entry(self, row: Optional[int] = None) -> Tuple[int, int]:
        """Sample entry (i,j) proportional to |A_ij|^2. O(1) time."""
        if row is None:
            row = self.sample_row()
        col = np.random.choice(self.n, p=self.col_samplers[row])
        return row, col

    def get_frobenius_norm(self) -> float:
        """Return Frobenius norm. O(1) time."""
        return self.frobenius_norm

    def get_row_norm(self, i: int) -> float:
        """Return ||A_i||. O(1) time."""
        return self.row_norms[i]


class QuantumInspiredSolver:
    """
    Quantum-inspired classical solver for linear systems.

    Based on Tang's dequantization techniques.
    """

    def __init__(self, A: np.ndarray, rank_threshold: int = 50):
        """
        Initialize solver with matrix A.

        Parameters:
        -----------
        A : ndarray
            System matrix
        rank_threshold : int
            Maximum rank for low-rank approximation
        """
        self.A = A
        self.m, self.n = A.shape
        self.rank_threshold = rank_threshold

        # Build SQ access
        self.sq_access = SampleQueryAccess(A)

        # Compute low-rank approximation
        self._compute_low_rank()

    def _compute_low_rank(self):
        """Compute truncated SVD for low-rank approximation."""
        # For demonstration, use full SVD then truncate
        # In practice, use randomized SVD
        U, s, Vh = svd(self.A, full_matrices=False)

        # Determine effective rank
        total_energy = np.sum(s**2)
        cumulative = np.cumsum(s**2) / total_energy

        # Find rank capturing 99% of energy
        self.effective_rank = np.searchsorted(cumulative, 0.99) + 1
        self.effective_rank = min(self.effective_rank, self.rank_threshold)

        # Store truncated factors
        self.U_k = U[:, :self.effective_rank]
        self.s_k = s[:self.effective_rank]
        self.Vh_k = Vh[:self.effective_rank, :]

        # Approximation error
        self.approx_error = np.sqrt(total_energy * (1 - cumulative[self.effective_rank-1])) if self.effective_rank < len(s) else 0

    def solve_low_rank(self, b: np.ndarray) -> np.ndarray:
        """
        Solve Ax = b using low-rank approximation.

        Complexity: O(effective_rank^2 * n)
        """
        # A_k = U_k * S_k * Vh_k
        # A_k^{-1} = Vh_k^T * S_k^{-1} * U_k^T

        # Step 1: y = U_k^T @ b
        y = self.U_k.T @ b

        # Step 2: z = S_k^{-1} @ y
        z = y / self.s_k

        # Step 3: x = Vh_k^T @ z
        x = self.Vh_k.T @ z

        return x

    def sample_solution(self, b: np.ndarray, num_samples: int = 100) -> np.ndarray:
        """
        Sample from solution distribution.

        This is what Tang's algorithm computes.
        """
        x = self.solve_low_rank(b)

        # Normalize to probability distribution
        probs = np.abs(x)**2
        probs /= np.sum(probs)

        # Sample
        samples = np.random.choice(len(x), size=num_samples, p=probs)
        return samples

    def estimate_inner_product(self, b: np.ndarray, v: np.ndarray,
                                num_samples: int = 1000) -> Tuple[float, float]:
        """
        Estimate <x, v> where x = A^{-1}b.

        Uses Monte Carlo estimation.
        """
        x = self.solve_low_rank(b)
        x_normalized = x / np.linalg.norm(x)

        # Direct computation for comparison
        true_value = np.dot(x_normalized, v)

        # Monte Carlo estimation
        x_probs = np.abs(x)**2 / np.sum(np.abs(x)**2)

        estimates = []
        for _ in range(num_samples // 10):
            idx = np.random.choice(len(x), p=x_probs)
            # Unbiased estimator
            sign = np.sign(x[idx]) if x[idx] != 0 else 1
            estimate = sign * v[idx] / np.sqrt(x_probs[idx])
            estimates.append(estimate)

        mean_estimate = np.mean(estimates)
        std_error = np.std(estimates) / np.sqrt(len(estimates))

        return mean_estimate, std_error


def compare_methods():
    """Compare quantum-inspired, direct, and iterative solvers."""
    sizes = [100, 200, 500, 1000]
    ranks = [10, 10, 10, 10]  # Effective rank stays constant

    results = {
        'direct': [],
        'low_rank': [],
        'cg': []
    }

    for n, r in zip(sizes, ranks):
        print(f"\nMatrix size: {n}x{n}, effective rank: {r}")

        # Create low-rank matrix
        U = np.random.randn(n, r)
        V = np.random.randn(r, n)
        A = U @ V
        # Add small full-rank component for non-singularity
        A += 0.01 * np.eye(n)

        b = np.random.randn(n)

        # Direct solve
        start = time.time()
        x_direct = np.linalg.solve(A, b)
        t_direct = time.time() - start
        results['direct'].append(t_direct)

        # Quantum-inspired (low-rank)
        start = time.time()
        solver = QuantumInspiredSolver(A, rank_threshold=r)
        x_lr = solver.solve_low_rank(b)
        t_lr = time.time() - start
        results['low_rank'].append(t_lr)

        # Conjugate gradient
        from scipy.sparse.linalg import cg
        start = time.time()
        x_cg, _ = cg(A, b, tol=1e-6)
        t_cg = time.time() - start
        results['cg'].append(t_cg)

        # Accuracy
        err_lr = np.linalg.norm(x_lr - x_direct) / np.linalg.norm(x_direct)
        err_cg = np.linalg.norm(x_cg - x_direct) / np.linalg.norm(x_direct)

        print(f"  Direct: {t_direct:.4f}s")
        print(f"  Low-rank: {t_lr:.4f}s (error: {err_lr:.2e})")
        print(f"  CG: {t_cg:.4f}s (error: {err_cg:.2e})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(sizes, results['direct'], 'ro-', label='Direct (O(N³))', linewidth=2)
    ax.loglog(sizes, results['low_rank'], 'bs-', label='Low-rank (Tang-style)', linewidth=2)
    ax.loglog(sizes, results['cg'], 'g^-', label='Conjugate Gradient', linewidth=2)

    ax.set_xlabel('Matrix Size N', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Solver Comparison (Fixed Effective Rank = 10)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('dequantization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def effective_rank_analysis():
    """Analyze how effective rank affects Tang's algorithm."""
    n = 500

    # Different rank structures
    rank_fractions = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    results = []

    for frac in rank_fractions:
        r = max(1, int(n * frac))

        # Create rank-r matrix
        U = np.random.randn(n, r)
        V = np.random.randn(r, n)
        A = U @ V + 0.01 * np.eye(n)

        b = np.random.randn(n)

        # Solve
        solver = QuantumInspiredSolver(A, rank_threshold=r)

        # Time and accuracy
        start = time.time()
        x_lr = solver.solve_low_rank(b)
        t_lr = time.time() - start

        x_direct = np.linalg.solve(A, b)
        error = np.linalg.norm(x_lr - x_direct) / np.linalg.norm(x_direct)

        results.append({
            'rank_fraction': frac,
            'rank': r,
            'time': t_lr,
            'error': error
        })

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fracs = [r['rank_fraction'] for r in results]
    times = [r['time'] for r in results]
    errors = [r['error'] for r in results]

    axes[0].plot(fracs, times, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Rank / N', fontsize=12)
    axes[0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0].set_title('Low-Rank Solver Time vs Rank', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(fracs, errors, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Rank / N', fontsize=12)
    axes[1].set_ylabel('Relative Error', fontsize=12)
    axes[1].set_title('Low-Rank Approximation Error', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('effective_rank_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


def sampling_access_demo():
    """Demonstrate SQ access operations."""
    print("Sample and Query Access Demonstration")
    print("=" * 50)

    # Create test matrix
    n = 100
    A = np.random.randn(n, n)

    # Build SQ access
    sq = SampleQueryAccess(A)

    print(f"\nMatrix: {n}x{n}")
    print(f"Frobenius norm: {sq.get_frobenius_norm():.4f}")

    # Test sampling
    print("\nRow sampling (10 samples):")
    row_counts = np.zeros(n)
    for _ in range(10000):
        row_counts[sq.sample_row()] += 1
    row_counts /= 10000

    # Compare to true distribution
    true_probs = sq.row_probs
    correlation = np.corrcoef(row_counts, true_probs)[0, 1]
    print(f"  Correlation with true distribution: {correlation:.4f}")

    # Entry sampling
    print("\nEntry sampling (showing top entries):")
    entry_counts = np.zeros((n, n))
    for _ in range(10000):
        i, j = sq.sample_entry()
        entry_counts[i, j] += 1
    entry_counts /= 10000

    flat_counts = entry_counts.flatten()
    flat_true = (np.abs(A)**2 / sq.frobenius_norm**2).flatten()

    top_idx = np.argsort(flat_true)[-5:]
    for idx in top_idx:
        i, j = idx // n, idx % n
        print(f"  ({i},{j}): sampled {flat_counts[idx]:.4f}, true {flat_true[idx]:.4f}")


def dequantization_boundaries():
    """Explore the boundaries of dequantization advantage."""
    print("\nDequantization Advantage Boundaries")
    print("=" * 50)

    n_values = [100, 500, 1000, 2000]
    rank_values = [5, 20, 50]

    print("\nTime comparison: Low-rank solver vs Direct")
    print("-" * 60)
    print(f"{'N':>6} {'Rank':>6} {'Low-rank (s)':>12} {'Direct (s)':>12} {'Speedup':>8}")
    print("-" * 60)

    for n in n_values:
        for r in rank_values:
            if r > n // 2:
                continue

            # Create matrix
            U = np.random.randn(n, r)
            V = np.random.randn(r, n)
            A = U @ V + 0.01 * np.eye(n)
            b = np.random.randn(n)

            # Time low-rank
            start = time.time()
            solver = QuantumInspiredSolver(A, rank_threshold=r)
            _ = solver.solve_low_rank(b)
            t_lr = time.time() - start

            # Time direct
            start = time.time()
            _ = np.linalg.solve(A, b)
            t_direct = time.time() - start

            speedup = t_direct / t_lr

            print(f"{n:>6} {r:>6} {t_lr:>12.4f} {t_direct:>12.4f} {speedup:>8.2f}x")


# Main execution
if __name__ == "__main__":
    print("Day 958: Dequantization & Classical Competition")
    print("=" * 60)

    # SQ access demo
    sampling_access_demo()

    # Method comparison
    print("\n" + "=" * 60)
    compare_methods()

    # Rank analysis
    print("\n" + "=" * 60)
    effective_rank_analysis()

    # Boundaries
    print("\n" + "=" * 60)
    dequantization_boundaries()
```

---

## Summary

### Key Formulas

| Formula | Expression | Context |
|---------|------------|---------|
| Tang complexity | $O(\|A\|_F^6 \kappa^6 / \epsilon)$ | Sampling output |
| HHL complexity | $O(\log N \cdot s^2 \kappa^2 / \epsilon)$ | Quantum state output |
| Effective rank | $r: \sum_{i=1}^r \sigma_i^2 \geq 0.99\|A\|_F^2$ | Low-rank threshold |

### Dequantization Conditions

Quantum advantage is **lost** when:
1. SQ access is available classically
2. Matrix has low effective rank
3. Only sampling/expectation outputs needed

### Key Insights

1. **Tang's breakthrough:** Classical algorithms can match HHL for sampling output
2. **SQ access is key:** Mimics qRAM capabilities classically
3. **Low-rank structure:** Enables classical randomized algorithms
4. **Genuine advantage remains:** For quantum inputs, full-rank matrices, specific problems
5. **The landscape has refined:** Not all "quantum speedups" survive scrutiny

---

## Daily Checklist

- [ ] I understand Tang's dequantization result
- [ ] I can define SQ (sample and query) access
- [ ] I know when classical algorithms match HHL
- [ ] I recognize genuine quantum advantage scenarios
- [ ] I understand quantum-inspired algorithm techniques
- [ ] I can analyze effective rank impact

---

## Preview: Day 959

Tomorrow we synthesize the week with **Applications and Week Review**:

- End-to-end HHL applications
- Quantum machine learning context
- Solving differential equations
- Complete complexity comparison table
- Month 36 preview: Quantum simulation algorithms

We'll consolidate everything learned this week into a practical decision framework.

---

*Day 958 of 2184 | Week 137 of 312 | Month 35 of 72*

*"Dequantization teaches us that quantum advantage is earned, not assumed."*
