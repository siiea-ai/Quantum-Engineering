# Day 121: SVD Applications — From Data Science to Quantum Mechanics

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: SVD Applications |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute the Moore-Penrose pseudoinverse using SVD
2. Solve least squares problems via SVD
3. Apply Principal Component Analysis (PCA)
4. Understand the polar decomposition
5. Connect SVD to the Schmidt decomposition in quantum mechanics
6. Apply SVD to quantum state analysis

---

## 📚 Required Reading

### Primary Text: Strang, "Introduction to Linear Algebra"
- **Section 7.3**: Principal Component Analysis (pp. 391-400)
- **Section 7.4**: The Geometry of the SVD (pp. 401-410)

### Physics Connection
- **Nielsen & Chuang, Section 2.5**: Schmidt decomposition

---

## 📖 Core Content: Theory and Concepts

### 1. The Moore-Penrose Pseudoinverse

**Problem:** For non-square or singular matrices, A⁻¹ doesn't exist. How do we solve Ax = b?

**Definition:** The **pseudoinverse** A⁺ is defined via SVD:
$$A = U\Sigma V^* \implies \boxed{A^+ = V\Sigma^+ U^*}$$

where Σ⁺ is formed by taking reciprocals of nonzero singular values:
$$\Sigma = \text{diag}(\sigma_1, ..., \sigma_r, 0, ..., 0) \implies \Sigma^+ = \text{diag}(1/\sigma_1, ..., 1/\sigma_r, 0, ..., 0)$$

**Properties of A⁺:**
1. AA⁺A = A (generalized inverse)
2. A⁺AA⁺ = A⁺
3. (AA⁺)* = AA⁺ (Hermitian)
4. (A⁺A)* = A⁺A (Hermitian)

### 2. Least Squares via SVD

**Problem:** Solve Ax = b when no exact solution exists (overdetermined system).

**Solution:** x = A⁺b minimizes ‖Ax - b‖₂

**Why?**
- A⁺b projects b onto col(A)
- Then solves for x in that subspace
- Among all minimizers, chooses the one with smallest ‖x‖

**Explicit formula:**
$$x = A^+b = V\Sigma^+U^*b = \sum_{i=1}^{r} \frac{u_i^* b}{\sigma_i} v_i$$

### 3. Principal Component Analysis (PCA)

**Goal:** Find the directions of maximum variance in data.

**Setup:**
- Data matrix X: n samples × p features
- Center the data: X̃ = X - X̄

**Covariance matrix:** C = X̃ᵀX̃/(n-1)

**Key insight:** Eigenvectors of C = right singular vectors of X̃!

**PCA Algorithm:**
1. Center data: X̃ᵢⱼ = Xᵢⱼ - μⱼ
2. Compute SVD: X̃ = UΣVᵀ
3. Principal components: columns of V
4. Variance explained by PC k: σₖ²/(Σσᵢ²)
5. Project data: Z = X̃V (new coordinates)

**Dimensionality reduction:**
Keep top k components: Z_k = X̃V_k where V_k is first k columns of V

### 4. Polar Decomposition via SVD

**Theorem:** Every matrix A can be written as:
$$A = UP \quad \text{(right polar)}$$
$$A = QU \quad \text{(left polar)}$$

where U is unitary and P, Q are positive semidefinite Hermitian.

**Construction from SVD:**
Given A = UₛΣVₛ*:
- **Right polar:** A = (UₛVₛ*)(VₛΣVₛ*) = U·P
  - U = UₛVₛ* (unitary, combines both rotations)
  - P = VₛΣVₛ* = √(A*A) (positive)
  
- **Left polar:** A = (UₛΣUₛ*)(UₛVₛ*) = Q·U
  - Q = UₛΣUₛ* = √(AA*) (positive)

**Geometric meaning:** Any linear map = (stretch along some axes) × (rotation)

### 5. Matrix Norms and Condition Number

**2-norm (spectral norm):**
$$\|A\|_2 = \sigma_1 = \max_{\|x\|=1} \|Ax\|$$

**Frobenius norm:**
$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\sum_i \sigma_i^2}$$

**Condition number:**
$$\kappa(A) = \frac{\sigma_1}{\sigma_r} = \|A\|_2 \|A^{-1}\|_2$$

**Interpretation:**
- κ ≈ 1: Well-conditioned (stable computations)
- κ >> 1: Ill-conditioned (sensitive to errors)
- κ = ∞: Singular matrix

---

## 🔬 Quantum Mechanics Connection

### The Schmidt Decomposition

**Theorem (Schmidt Decomposition):**
Any pure state |ψ⟩ in a bipartite system ℋ_A ⊗ ℋ_B can be written as:

$$\boxed{|\psi\rangle = \sum_{i=1}^{r} \lambda_i |a_i\rangle \otimes |b_i\rangle}$$

where:
- {|aᵢ⟩} orthonormal in ℋ_A
- {|bᵢ⟩} orthonormal in ℋ_B  
- λᵢ > 0 are **Schmidt coefficients** with Σλᵢ² = 1
- r is the **Schmidt rank**

### Connection to SVD

**Key insight:** The Schmidt decomposition IS the SVD!

Given |ψ⟩ = Σᵢⱼ Cᵢⱼ |i⟩_A |j⟩_B, the coefficient matrix C has SVD:
$$C = U\Sigma V^\dagger$$

Then:
- |aᵢ⟩ = Σⱼ Uⱼᵢ |j⟩_A (columns of U)
- |bᵢ⟩ = Σₖ Vₖᵢ* |k⟩_B (columns of V*)
- λᵢ = σᵢ (singular values)

### Entanglement and Schmidt Rank

**Schmidt rank = 1:** Product state (no entanglement)
$$|\psi\rangle = |a\rangle \otimes |b\rangle$$

**Schmidt rank > 1:** Entangled state!

**Entanglement measure (entropy):**
$$S = -\sum_i \lambda_i^2 \log_2(\lambda_i^2)$$

This is the **entanglement entropy** — von Neumann entropy of reduced density matrix.

### Example: Bell States

**Bell state:** |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

Coefficient matrix:
$$C = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

SVD: C = I · (1/√2)I · I (trivial since C is diagonal)

Schmidt coefficients: λ₁ = λ₂ = 1/√2
Schmidt rank: 2 (maximally entangled!)

Entanglement entropy: S = -2 × (1/2)log₂(1/2) = 1 ebit (maximum for 2 qubits)

### Quantum State Tomography

**Problem:** Determine unknown quantum state from measurements.

**Connection to SVD:**
- Density matrix ρ can be reconstructed from expectation values
- SVD helps with:
  - Noise reduction in experimental data
  - Enforcing positivity constraints
  - Finding optimal low-rank approximations

---

## ✏️ Worked Examples

### Example 1: Least Squares via SVD

Solve the least squares problem:
$$\begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix}$$

**Step 1:** Compute SVD of A
$$A^TA = \begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix}$$

Eigenvalues: λ = 16.16, 0.84 → σ₁ = 4.02, σ₂ = 0.92

(For full solution, compute eigenvectors for V, then U = AV/Σ)

**Step 2:** Compute pseudoinverse A⁺ = VΣ⁺Uᵀ

**Step 3:** x = A⁺b

Using numerical computation: x ≈ (0.5, 0.5)

**Verification:** Ax = (1, 1.5, 2) minimizes distance to b = (1, 2, 2)

### Example 2: PCA on 2D Data

Data points: (1,2), (2,4), (3,5), (4,7), (5,8)

**Step 1:** Center data
Mean: (3, 5.2)
X̃ = [(-2,-3.2), (-1,-1.2), (0,-0.2), (1,1.8), (2,2.8)]

**Step 2:** Compute SVD of X̃
First principal component ≈ direction of maximum variance

**Result:** PC1 captures ~98% of variance (data lies nearly on a line)

### Example 3: Schmidt Decomposition

Find Schmidt decomposition of |ψ⟩ = (|00⟩ + |01⟩ + |10⟩)/√3

**Step 1:** Write coefficient matrix
$$C = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$$

**Step 2:** Compute SVD
$$C^*C = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$

Eigenvalues: λ₁ = (3+√5)/6, λ₂ = (3-√5)/6

σ₁ = √λ₁ ≈ 0.934, σ₂ = √λ₂ ≈ 0.357

**Step 3:** Find Schmidt vectors
The right singular vectors give |bᵢ⟩ (for system B)
The left singular vectors give |aᵢ⟩ (for system A)

**Result:**
$$|\psi\rangle = 0.934|a_1\rangle|b_1\rangle + 0.357|a_2\rangle|b_2\rangle$$

**Entanglement entropy:**
S = -0.934²log₂(0.934²) - 0.357²log₂(0.357²) ≈ 0.55 ebits

### Example 4: Polar Decomposition

Find the polar decomposition of:
$$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

**Step 1:** Compute A*A
$$A^*A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}$$

**Step 2:** Find P = √(A*A)
Eigenvalues of A*A: λ = (3±√5)/2
P = V diag(√λ₁, √λ₂) V*

**Step 3:** Find U = AP⁻¹

**Result:** A = UP where U is unitary (rotation) and P is positive (stretch)

---

## 📝 Practice Problems

### Level 1: Pseudoinverse
1. Compute A⁺ for A = [[1, 0], [0, 0], [0, 1]] using SVD.

2. Show that (A*)⁺ = (A⁺)*.

3. Verify AA⁺A = A for the matrix in problem 1.

### Level 2: Least Squares and PCA
4. Use SVD to solve the least squares problem: fit y = ax + b to points (0,1), (1,3), (2,4).

5. Given data X = [[1,2], [2,3], [3,5], [4,6]], find the first principal component.

6. How much variance is explained by the first PC in problem 5?

### Level 3: Quantum Applications
7. Find the Schmidt decomposition of |ψ⟩ = (2|00⟩ + |11⟩)/√5.

8. Compute the entanglement entropy for the state in problem 7.

9. Show that a product state |ψ⟩ = |a⟩|b⟩ has Schmidt rank 1.

### Level 4: Theory
10. Prove: The pseudoinverse of a unitary matrix equals its adjoint.

11. Show that PCA is equivalent to finding directions that minimize reconstruction error.

12. Prove: For Hermitian A, the polar decomposition is A = U|A| where U is also Hermitian.

---

## 💻 Evening Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm, pinv, eigh

np.set_printoptions(precision=4, suppress=True)

# ============================================
# Least Squares via SVD
# ============================================

def least_squares_svd(A, b, tol=1e-10):
    """Solve Ax ≈ b using SVD pseudoinverse"""
    U, s, Vh = svd(A, full_matrices=False)
    
    # x = V Σ⁺ U* b
    s_inv = np.array([1/si if si > tol else 0 for si in s])
    x = Vh.T @ np.diag(s_inv) @ U.T @ b
    
    residual = norm(A @ x - b)
    return x, residual

# Test: Linear regression
A = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
b = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

x_svd, res = least_squares_svd(A, b)
x_np = pinv(A) @ b

print("=== Least Squares via SVD ===")
print(f"SVD solution: {x_svd} (intercept, slope)")
print(f"NumPy pinv: {x_np}")
print(f"Residual: {res:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(A[:, 1], b, s=100, label='Data')
x_line = np.linspace(0, 6, 100)
y_line = x_svd[0] + x_svd[1] * x_line
plt.plot(x_line, y_line, 'r-', label=f'Fit: y = {x_svd[0]:.2f} + {x_svd[1]:.2f}x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Least Squares Fit via SVD')
plt.savefig('least_squares_svd.png', dpi=150)
plt.show()

# ============================================
# Principal Component Analysis
# ============================================

def pca_svd(X, n_components=None):
    """PCA via SVD"""
    # Center data
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    
    # SVD
    U, s, Vh = svd(X_centered, full_matrices=False)
    
    # Variance explained
    variance = s**2 / (X.shape[0] - 1)
    variance_ratio = variance / variance.sum()
    
    # Principal components (rows of Vh = columns of V)
    components = Vh
    
    if n_components is not None:
        components = components[:n_components]
        X_transformed = X_centered @ components.T
    else:
        X_transformed = X_centered @ components.T
    
    return X_transformed, components, variance_ratio, X_mean

# Generate correlated 2D data
np.random.seed(42)
n_samples = 200
t = np.random.randn(n_samples)
X = np.column_stack([
    t + 0.2*np.random.randn(n_samples),
    2*t + 0.3*np.random.randn(n_samples)
])

# Apply PCA
X_pca, components, var_ratio, X_mean = pca_svd(X, n_components=2)

print("\n=== Principal Component Analysis ===")
print(f"PC1 direction: {components[0]}")
print(f"PC2 direction: {components[1]}")
print(f"Variance explained: {var_ratio * 100}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original data with PC directions
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5)
# Plot PC directions
for i, (pc, var) in enumerate(zip(components, var_ratio)):
    axes[0].arrow(X_mean[0], X_mean[1], 2*pc[0], 2*pc[1], 
                 head_width=0.1, head_length=0.1, fc=f'C{i+1}', ec=f'C{i+1}')
    axes[0].text(X_mean[0] + 2.2*pc[0], X_mean[1] + 2.2*pc[1], 
                f'PC{i+1} ({var*100:.1f}%)', fontsize=10)
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')
axes[0].set_title('Original Data with Principal Components')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Transformed data
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Data in PC Coordinates')
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=150)
plt.show()

# ============================================
# Schmidt Decomposition for Quantum States
# ============================================

def schmidt_decomposition(psi, dim_A, dim_B):
    """
    Compute Schmidt decomposition of bipartite state.
    
    psi: state vector of length dim_A * dim_B
    Returns: Schmidt coefficients, Schmidt vectors for A and B
    """
    # Reshape to coefficient matrix
    C = psi.reshape(dim_A, dim_B)
    
    # SVD
    U, s, Vh = svd(C, full_matrices=False)
    
    # Schmidt vectors
    schmidt_A = U  # columns are |a_i⟩
    schmidt_B = Vh.conj().T  # columns are |b_i⟩
    
    # Schmidt coefficients (normalized singular values)
    schmidt_coeffs = s
    
    return schmidt_coeffs, schmidt_A, schmidt_B

def entanglement_entropy(schmidt_coeffs):
    """Compute von Neumann entropy of entanglement"""
    # λ_i^2 are the eigenvalues of reduced density matrix
    probs = schmidt_coeffs**2
    probs = probs[probs > 1e-15]  # Remove zeros
    return -np.sum(probs * np.log2(probs))

# Example 1: Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
print("\n=== Schmidt Decomposition ===")
bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
coeffs, A_vecs, B_vecs = schmidt_decomposition(bell_state, 2, 2)

print("Bell state |Φ+⟩:")
print(f"  Schmidt coefficients: {coeffs}")
print(f"  Schmidt rank: {np.sum(coeffs > 1e-10)}")
print(f"  Entanglement entropy: {entanglement_entropy(coeffs):.4f} ebits")

# Example 2: Product state |0⟩|+⟩
product_state = np.array([1, 1, 0, 0]) / np.sqrt(2)
coeffs2, _, _ = schmidt_decomposition(product_state, 2, 2)
print("\nProduct state |0⟩|+⟩:")
print(f"  Schmidt coefficients: {coeffs2}")
print(f"  Schmidt rank: {np.sum(coeffs2 > 1e-10)}")
print(f"  Entanglement entropy: {entanglement_entropy(coeffs2):.4f} ebits")

# Example 3: Partially entangled state
partial = np.array([np.sqrt(0.8), 0, 0, np.sqrt(0.2)])
coeffs3, _, _ = schmidt_decomposition(partial, 2, 2)
print("\nPartially entangled state:")
print(f"  Schmidt coefficients: {coeffs3}")
print(f"  Entanglement entropy: {entanglement_entropy(coeffs3):.4f} ebits")

# ============================================
# Polar Decomposition
# ============================================

def polar_decomposition(A):
    """Compute A = U P (right polar decomposition)"""
    U_svd, s, Vh = svd(A)
    
    # Unitary part: U = U_svd @ Vh
    U = U_svd @ Vh
    
    # Positive part: P = V Σ V* = Vh* @ diag(s) @ Vh
    P = Vh.conj().T @ np.diag(s) @ Vh
    
    return U, P

# Test
A = np.array([[1, 1], [0, 1]], dtype=complex)
U, P = polar_decomposition(A)

print("\n=== Polar Decomposition ===")
print(f"A =\n{A}")
print(f"\nU (unitary) =\n{U}")
print(f"\nP (positive) =\n{P}")
print(f"\nU @ P =\n{U @ P}")
print(f"\nU unitary check (U*U): \n{U.conj().T @ U}")
print(f"\nP positive check (eigenvalues): {np.linalg.eigvalsh(P)}")

# ============================================
# Condition Number Analysis
# ============================================

def analyze_condition(A):
    """Analyze matrix conditioning via SVD"""
    U, s, Vh = svd(A)
    
    print(f"Singular values: {s}")
    print(f"Condition number: {s[0]/s[-1]:.2f}")
    print(f"2-norm: {s[0]:.4f}")
    print(f"Frobenius norm: {np.sqrt(np.sum(s**2)):.4f}")
    
    return s[0]/s[-1]

print("\n=== Condition Number Analysis ===")
# Well-conditioned
A_good = np.array([[1, 0], [0, 1]])
print("Identity matrix:")
analyze_condition(A_good)

# Ill-conditioned
A_bad = np.array([[1, 1], [1, 1.0001]])
print("\nNearly singular matrix:")
analyze_condition(A_bad)
```

---

## ✅ Daily Checklist

- [ ] Understand pseudoinverse via SVD
- [ ] Solve least squares problems
- [ ] Apply PCA to data
- [ ] Compute polar decomposition
- [ ] Derive Schmidt decomposition for quantum states
- [ ] Calculate entanglement entropy
- [ ] Complete computational lab
- [ ] Solve at least 6 practice problems

---

## 🔜 Preview: Tomorrow

**Day 122: Tensor Products — Building Composite Systems**
- Definition of tensor product spaces
- Tensor product of vectors and operators
- Computational basis for multi-qubit systems
- Kronecker product
- QM Connection: Multi-particle quantum states

---

*"The Schmidt decomposition is the SVD with a quantum mechanical hat on."*
— Quantum Information Saying
