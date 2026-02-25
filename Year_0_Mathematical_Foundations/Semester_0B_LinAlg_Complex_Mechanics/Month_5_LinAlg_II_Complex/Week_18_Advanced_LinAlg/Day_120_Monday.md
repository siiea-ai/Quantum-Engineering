# Day 120: Singular Value Decomposition — The Ultimate Matrix Factorization

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: SVD Foundations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State and prove the Singular Value Decomposition theorem
2. Compute SVD for small matrices by hand
3. Understand the geometric interpretation of SVD
4. Relate SVD to eigendecomposition
5. Apply SVD to matrix approximation and rank
6. Connect SVD to quantum state analysis

---

## 📚 Required Reading

### Primary Text: Strang, "Introduction to Linear Algebra"
- **Section 7.1-7.2**: Singular Value Decomposition (pp. 371-390)

### Secondary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 7.E**: Singular Value Decomposition (pp. 249-260)

### Supplementary
- Trefethen & Bau, "Numerical Linear Algebra," Lectures 4-5

---

## 🎬 Video Resources

### 3Blue1Brown
- Singular Value Decomposition visualization (if available)

### MIT OCW 18.06
- **Lecture 29**: Singular Value Decomposition
- Gilbert Strang's geometric explanation

### Steve Brunton (YouTube)
- SVD series: Excellent applied perspective

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: Why SVD?

**Problem:** Eigendecomposition A = PDP⁻¹ only works for:
- Square matrices
- Diagonalizable matrices

**SVD solves this:** Works for ANY m×n matrix!

**Applications:**
- Data compression (images, signals)
- Noise reduction
- Recommender systems
- Principal Component Analysis (PCA)
- Quantum state analysis
- Low-rank approximation

### 2. The SVD Theorem

**Theorem (Singular Value Decomposition):**
Every m×n matrix A (real or complex) can be written as:

$$\boxed{A = U \Sigma V^*}$$

where:
- **U** is m×m unitary (orthogonal if real): columns are **left singular vectors**
- **Σ** is m×n diagonal with non-negative entries: **singular values** σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0
- **V** is n×n unitary: columns are **right singular vectors**

**Diagram:**
```
       n              n           n              n
    ┌─────┐       ┌─────┐     ┌─────┐       ┌─────┐
  m │  A  │  =  m │  U  │  ×  m │  Σ  │  ×  n │ V* │
    └─────┘       └─────┘     └─────┘       └─────┘
     m×n           m×m         m×n           n×n
```

### 3. Singular Values and Vectors

**Definitions:**
- **Singular values:** σᵢ = √(λᵢ(A*A)) = √(λᵢ(AA*)) (square roots of eigenvalues)
- **Right singular vectors:** Eigenvectors of A*A (columns of V)
- **Left singular vectors:** Eigenvectors of AA* (columns of U)

**Key relations:**
$$A^*A = V\Sigma^*\Sigma V^* \quad \text{(eigendecomposition of } A^*A\text{)}$$
$$AA^* = U\Sigma\Sigma^* U^* \quad \text{(eigendecomposition of } AA^*\text{)}$$

$$Av_i = \sigma_i u_i \quad \text{(singular vectors are related!)}$$
$$A^*u_i = \sigma_i v_i$$

### 4. Proof Outline

**Step 1:** A*A is Hermitian and positive semidefinite
- Eigenvalues are real and ≥ 0
- Has orthonormal eigenbasis {v₁, ..., vₙ}

**Step 2:** Define singular values σᵢ = √λᵢ where A*Avᵢ = λᵢvᵢ

**Step 3:** For σᵢ > 0, define uᵢ = Avᵢ/σᵢ
- These are orthonormal: ⟨uᵢ, uⱼ⟩ = ⟨Avᵢ/σᵢ, Avⱼ/σⱼ⟩ = vᵢ*A*Avⱼ/(σᵢσⱼ) = δᵢⱼ

**Step 4:** Extend {u₁, ..., uᵣ} to orthonormal basis of ℂᵐ

**Step 5:** Verify A = UΣV* by checking Avⱼ = σⱼuⱼ for all j ∎

### 5. Reduced SVD (Compact Form)

For rank-r matrix A:

**Full SVD:** A = U_{m×m} Σ_{m×n} V*_{n×n}

**Reduced SVD:** A = U_{m×r} Σ_{r×r} V*_{r×n}

Only keep the r non-zero singular values and corresponding vectors.

**Outer product form:**
$$A = \sum_{i=1}^{r} \sigma_i u_i v_i^*$$

### 6. Geometric Interpretation

**SVD reveals the geometry of linear maps:**

1. **V*** rotates/reflects the input space (align with principal axes)
2. **Σ** scales along each axis (by singular values)
3. **U** rotates/reflects the output space

**Any linear map = rotation × scaling × rotation**

**Image of unit sphere:**
- Input: Unit sphere in ℝⁿ
- After V*: Still unit sphere (rotation)
- After Σ: Ellipsoid with semi-axes σ₁, σ₂, ...
- After U: Rotated ellipsoid in ℝᵐ

### 7. Properties of Singular Values

| Property | Formula |
|----------|---------|
| Rank | r = number of nonzero σᵢ |
| 2-norm | ‖A‖₂ = σ₁ (largest singular value) |
| Frobenius norm | ‖A‖_F = √(Σσᵢ²) |
| Condition number | κ(A) = σ₁/σᵣ (for invertible A) |
| Determinant | \|det(A)\| = ∏σᵢ (for square A) |

### 8. Low-Rank Approximation

**Eckart-Young-Mirsky Theorem:**
The best rank-k approximation to A (in 2-norm or Frobenius norm) is:

$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^*$$

**Error:** ‖A - Aₖ‖₂ = σₖ₊₁

**Application:** Data compression, noise reduction, dimensionality reduction

---

## 🔬 Quantum Mechanics Connection

### SVD and Quantum States

**Schmidt Decomposition (Preview):**
For a bipartite pure state |ψ⟩ ∈ ℋ_A ⊗ ℋ_B, there exist orthonormal bases such that:

$$|\psi\rangle = \sum_{i=1}^{r} \lambda_i |a_i\rangle \otimes |b_i\rangle$$

where λᵢ > 0 are **Schmidt coefficients** (related to singular values!).

**Connection:**
If we write |ψ⟩ with coefficient matrix C_{ij} = ⟨i,j|ψ⟩, then:
- SVD of C gives Schmidt decomposition
- Schmidt rank = rank of C = number of nonzero singular values
- Entanglement measures relate to singular values

### Quantum Channels and SVD

**Kraus representation:** ℰ(ρ) = Σₖ KₖρKₖ†

The singular values of the process matrix characterize the channel.

### Fidelity and Singular Values

**Fidelity between states:**
$$F(\rho, \sigma) = \left(\text{tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

involves singular values of √ρ √σ.

---

## ✏️ Worked Examples

### Example 1: Computing SVD (2×2)

Find the SVD of:
$$A = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}$$

**Step 1:** Compute A^TA
$$A^TA = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}\begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix} = \begin{pmatrix} 13 & 12 \\ 12 & 13 \end{pmatrix}$$

**Step 2:** Find eigenvalues of A^TA
$$\det(A^TA - \lambda I) = (13-\lambda)^2 - 144 = \lambda^2 - 26\lambda + 25 = (\lambda-25)(\lambda-1)$$
λ₁ = 25, λ₂ = 1

**Step 3:** Singular values
σ₁ = √25 = 5, σ₂ = √1 = 1

**Step 4:** Right singular vectors (eigenvectors of A^TA)
λ = 25: v₁ = (1, 1)ᵀ/√2
λ = 1: v₂ = (1, -1)ᵀ/√2

$$V = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Step 5:** Left singular vectors (u = Av/σ)
$$u_1 = \frac{1}{5}A v_1 = \frac{1}{5}\begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

$$u_2 = \frac{1}{1}A v_2 = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

$$U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Result:**
$$A = U\Sigma V^T = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 5 & 0 \\ 0 & 1 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Verification:** Note that for this symmetric matrix, U = V!

### Example 2: SVD of Non-Square Matrix

Find the SVD of:
$$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**Step 1:** A^TA (2×2)
$$A^TA = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

Eigenvalues: λ = 3, 1 → σ₁ = √3, σ₂ = 1

Eigenvectors: v₁ = (1,1)ᵀ/√2, v₂ = (1,-1)ᵀ/√2

**Step 2:** Left singular vectors
$$u_1 = \frac{Av_1}{\sigma_1} = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{6}}\begin{pmatrix} 2 \\ 1 \\ 1 \end{pmatrix}$$

$$u_2 = \frac{Av_2}{\sigma_2} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 \\ -1 \\ 1 \end{pmatrix}$$

**Step 3:** Extend to full U (need u₃ orthogonal to u₁, u₂)
Using Gram-Schmidt or cross product insight:
$$u_3 = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 \\ -1 \\ -1 \end{pmatrix}$$

**Result:**
$$\Sigma = \begin{pmatrix} \sqrt{3} & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}$$

### Example 3: Rank-1 Approximation

For A from Example 1, find the best rank-1 approximation.

$$A_1 = \sigma_1 u_1 v_1^T = 5 \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \end{pmatrix} = \frac{5}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**Error:** ‖A - A₁‖₂ = σ₂ = 1

---

## 📝 Practice Problems

### Level 1: Computation
1. Compute the SVD of A = [[4, 0], [3, -5]].

2. Find singular values of A = [[1, 2], [0, 0], [0, 0]].

3. For A = [[1, 0], [0, 2], [0, 0]], write the full and reduced SVD.

### Level 2: Properties
4. Prove that singular values of A and A* are the same.

5. Show that ‖A‖_F² = Σσᵢ² = tr(A*A).

6. Prove: rank(A) = number of nonzero singular values.

7. If A is Hermitian, how do singular values relate to eigenvalues?

### Level 3: Applications
8. For A = [[3, 2, 2], [2, 3, -2]], find the best rank-1 approximation.

9. Compute the condition number of A = [[1, 2], [3, 4]].

10. Use SVD to find the pseudoinverse of A = [[1, 2], [2, 4]].

### Level 4: Theory
11. Prove the Eckart-Young theorem: Aₖ minimizes ‖A - B‖_F over rank-k matrices B.

12. Show that for unitary U: σᵢ(UA) = σᵢ(A) = σᵢ(AV) for unitary V.

13. Prove: ‖A‖₂ = σ₁ = max_{‖x‖=1} ‖Ax‖.

---

## 💻 Evening Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm, matrix_rank

np.set_printoptions(precision=4, suppress=True)

# ============================================
# Basic SVD Computation
# ============================================

def demonstrate_svd(A, name="A"):
    """Compute and visualize SVD of matrix A"""
    print(f"\n{'='*50}")
    print(f"SVD of {name}")
    print(f"{'='*50}")
    print(f"Matrix {name}:\n{A}")
    
    U, s, Vh = svd(A)
    
    print(f"\nU (left singular vectors):\n{U}")
    print(f"\nSingular values: {s}")
    print(f"\nV* (right singular vectors):\n{Vh}")
    
    # Reconstruct
    m, n = A.shape
    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = s[i]
    
    A_reconstructed = U @ Sigma @ Vh
    print(f"\nReconstruction error: {norm(A - A_reconstructed):.2e}")
    
    return U, s, Vh

# Example 1: Symmetric matrix
A1 = np.array([[3, 2], [2, 3]])
U1, s1, Vh1 = demonstrate_svd(A1, "Symmetric")

# Example 2: Non-square matrix
A2 = np.array([[1, 1], [0, 1], [1, 0]])
U2, s2, Vh2 = demonstrate_svd(A2, "Non-square")

# Example 3: Rank-deficient matrix
A3 = np.array([[1, 2, 3], [2, 4, 6]])
U3, s3, Vh3 = demonstrate_svd(A3, "Rank-deficient")

# ============================================
# Geometric Visualization
# ============================================

def plot_svd_geometry(A):
    """Visualize SVD as rotation-scaling-rotation"""
    U, s, Vh = svd(A)
    
    # Generate unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original unit circle
    axes[0].plot(circle[0], circle[1], 'b-', linewidth=2)
    axes[0].set_title('Unit Circle (Input)')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # After V* (rotation)
    rotated = Vh @ circle
    axes[1].plot(rotated[0], rotated[1], 'g-', linewidth=2)
    axes[1].set_title('After V* (Rotation)')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    # After Σ (scaling)
    Sigma = np.diag(s)
    scaled = Sigma @ rotated
    axes[2].plot(scaled[0], scaled[1], 'r-', linewidth=2)
    axes[2].set_title(f'After Σ (Scale by {s[0]:.2f}, {s[1]:.2f})')
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    
    # After U (final rotation) = A @ circle
    final = A @ circle
    axes[3].plot(final[0], final[1], 'm-', linewidth=2)
    axes[3].set_title('After U = Final (A × circle)')
    axes[3].set_xlim(-3, 3)
    axes[3].set_ylim(-3, 3)
    axes[3].set_aspect('equal')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svd_geometry.png', dpi=150)
    plt.show()

A_geo = np.array([[2, 1], [1, 2]])
plot_svd_geometry(A_geo)

# ============================================
# Low-Rank Approximation
# ============================================

def low_rank_approximation(A, k):
    """Compute best rank-k approximation"""
    U, s, Vh = svd(A, full_matrices=False)
    
    # Keep only top k components
    U_k = U[:, :k]
    s_k = s[:k]
    Vh_k = Vh[:k, :]
    
    A_k = U_k @ np.diag(s_k) @ Vh_k
    
    error = norm(A - A_k, 'fro')
    
    return A_k, error

# Create a test matrix
np.random.seed(42)
A_test = np.random.randn(10, 10)

print("\n" + "="*50)
print("Low-Rank Approximation")
print("="*50)

U, s, Vh = svd(A_test)
print(f"Singular values: {s}")

for k in [1, 2, 3, 5, 10]:
    A_k, error = low_rank_approximation(A_test, k)
    print(f"Rank-{k}: Error = {error:.4f}, Expected = {np.sqrt(np.sum(s[k:]**2)):.4f}")

# ============================================
# Image Compression Demo
# ============================================

def compress_matrix(A, k):
    """Compress matrix using rank-k SVD approximation"""
    U, s, Vh = svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]

# Create a simple "image" pattern
n = 50
image = np.zeros((n, n))
# Add some structure
for i in range(n):
    for j in range(n):
        image[i, j] = np.sin(i/5) * np.cos(j/5) + 0.5*np.sin((i+j)/3)

# Compress at different ranks
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

U, s, Vh = svd(image)
axes[0, 0].semilogy(s, 'b.-')
axes[0, 0].set_title('Singular Value Spectrum')
axes[0, 0].set_xlabel('Index')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].imshow(image, cmap='viridis')
axes[0, 1].set_title(f'Original (rank {matrix_rank(image)})')
axes[0, 1].axis('off')

for i, k in enumerate([1, 2, 5, 10, 20]):
    ax = axes[(i+2)//4, (i+2)%4]
    compressed = compress_matrix(image, k)
    ax.imshow(compressed, cmap='viridis')
    error = norm(image - compressed, 'fro') / norm(image, 'fro')
    ax.set_title(f'Rank {k} ({error*100:.1f}% error)')
    ax.axis('off')

plt.tight_layout()
plt.savefig('svd_compression.png', dpi=150)
plt.show()

# ============================================
# Pseudoinverse via SVD
# ============================================

def pseudoinverse_svd(A, tol=1e-10):
    """Compute Moore-Penrose pseudoinverse via SVD"""
    U, s, Vh = svd(A, full_matrices=False)
    
    # Invert non-zero singular values
    s_inv = np.array([1/si if si > tol else 0 for si in s])
    
    # A⁺ = V Σ⁺ U*
    A_pinv = Vh.T @ np.diag(s_inv) @ U.T
    
    return A_pinv

# Test on rank-deficient matrix
A_rank_def = np.array([[1, 2], [2, 4], [3, 6]])
A_pinv = pseudoinverse_svd(A_rank_def)
A_pinv_np = np.linalg.pinv(A_rank_def)

print("\n" + "="*50)
print("Pseudoinverse via SVD")
print("="*50)
print(f"A =\n{A_rank_def}")
print(f"\nA⁺ (our method) =\n{A_pinv}")
print(f"\nA⁺ (numpy) =\n{A_pinv_np}")
print(f"\nA A⁺ A =\n{A_rank_def @ A_pinv @ A_rank_def}")
print("(Should equal A)")
```

---

## ✅ Daily Checklist

- [ ] Read Strang Chapter 7.1-7.2 on SVD
- [ ] Understand the three components: U, Σ, V*
- [ ] Compute SVD by hand for 2×2 matrix
- [ ] Understand geometric interpretation
- [ ] Connect to eigendecomposition of A*A and AA*
- [ ] Understand low-rank approximation
- [ ] Complete computational lab
- [ ] Solve at least 6 practice problems

---

## 📓 Reflection Questions

1. Why does SVD work for any matrix while eigendecomposition requires special conditions?

2. How does the condition number σ₁/σᵣ relate to numerical stability?

3. What's the physical meaning of singular values in data analysis?

---

## 🔜 Preview: Tomorrow

**Day 121: SVD Applications and Connections**
- Pseudoinverse and least squares
- Principal Component Analysis (PCA)
- Data compression and denoising
- Connection to Schmidt decomposition in QM
- Polar decomposition via SVD

---

*"The SVD is arguably the most important matrix factorization."*
— Gilbert Strang
