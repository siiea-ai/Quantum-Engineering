# Day 109: Gram-Schmidt Orthogonalization

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Gram-Schmidt Algorithm |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Execute the Gram-Schmidt process step-by-step
2. Convert any linearly independent set to an orthonormal set
3. Understand and compute QR decomposition
4. Apply Gram-Schmidt to function spaces
5. Recognize numerical stability issues
6. Connect to quantum state orthogonalization

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 6.B**: Orthonormal Bases (pp. 193-204)
- Focus on: Gram-Schmidt Process (6.31)

### Secondary Reading
- **Strang, Chapter 4.4**: Orthogonal Bases and Gram-Schmidt
- **MIT 18.06 Lecture Notes**: Lecture 17

---

## 📖 Core Content: Theory and Concepts

### 1. The Problem

**Given:** Linearly independent vectors {v₁, v₂, ..., vₖ}
**Want:** Orthonormal vectors {e₁, e₂, ..., eₖ} spanning the same space

**Why?** Orthonormal bases make everything easier:
- Coordinates: cᵢ = ⟨eᵢ|v⟩ (no matrix inversion!)
- Projections: proj_W(v) = Σᵢ ⟨eᵢ|v⟩ eᵢ
- Eigenvalue problems
- Quantum states

### 2. The Key Idea

**Orthogonal projection removes the "parallel" part:**

If we want e₂ ⊥ e₁, start with v₂ and subtract its projection onto e₁:
$$w_2 = v_2 - \langle e_1 | v_2 \rangle e_1$$

Now w₂ ⊥ e₁ by construction!

### 3. The Gram-Schmidt Algorithm

**Input:** Linearly independent {v₁, v₂, ..., vₖ}
**Output:** Orthonormal {e₁, e₂, ..., eₖ}

**Step 1:** Normalize first vector
$$e_1 = \frac{v_1}{\|v_1\|}$$

**Step 2:** Make v₂ orthogonal to e₁, then normalize
$$w_2 = v_2 - \langle e_1 | v_2 \rangle e_1$$
$$e_2 = \frac{w_2}{\|w_2\|}$$

**Step 3:** Make v₃ orthogonal to e₁ AND e₂, then normalize
$$w_3 = v_3 - \langle e_1 | v_3 \rangle e_1 - \langle e_2 | v_3 \rangle e_2$$
$$e_3 = \frac{w_3}{\|w_3\|}$$

**General Step j:** 
$$w_j = v_j - \sum_{i=1}^{j-1} \langle e_i | v_j \rangle e_i$$
$$e_j = \frac{w_j}{\|w_j\|}$$

### 4. Why It Works

**Theorem:** At each step:
1. wⱼ ⊥ eᵢ for all i < j
2. span{e₁,...,eⱼ} = span{v₁,...,vⱼ}
3. wⱼ ≠ 0 (since vⱼ is independent of v₁,...,vⱼ₋₁)

**Proof of orthogonality:** For i < j:
$$\langle e_i | w_j \rangle = \langle e_i | v_j \rangle - \sum_{k=1}^{j-1} \langle e_k | v_j \rangle \langle e_i | e_k \rangle$$
$$= \langle e_i | v_j \rangle - \langle e_i | v_j \rangle \cdot 1 = 0 \quad \blacksquare$$

### 5. Example: Full Walkthrough

**Given:** v₁ = (1, 1, 1), v₂ = (1, 0, 1), v₃ = (1, 1, 0)

**Step 1:**
$$\|v_1\| = \sqrt{3}$$
$$e_1 = \frac{1}{\sqrt{3}}(1, 1, 1)$$

**Step 2:**
$$\langle e_1 | v_2 \rangle = \frac{1}{\sqrt{3}}(1 + 0 + 1) = \frac{2}{\sqrt{3}}$$
$$w_2 = v_2 - \frac{2}{\sqrt{3}} \cdot \frac{1}{\sqrt{3}}(1,1,1) = (1,0,1) - \frac{2}{3}(1,1,1) = \frac{1}{3}(1,-2,1)$$
$$\|w_2\| = \frac{1}{3}\sqrt{1+4+1} = \frac{\sqrt{6}}{3}$$
$$e_2 = \frac{1}{\sqrt{6}}(1, -2, 1)$$

**Step 3:**
$$\langle e_1 | v_3 \rangle = \frac{1}{\sqrt{3}}(1+1+0) = \frac{2}{\sqrt{3}}$$
$$\langle e_2 | v_3 \rangle = \frac{1}{\sqrt{6}}(1-2+0) = \frac{-1}{\sqrt{6}}$$
$$w_3 = (1,1,0) - \frac{2}{\sqrt{3}} \cdot \frac{1}{\sqrt{3}}(1,1,1) - \frac{-1}{\sqrt{6}} \cdot \frac{1}{\sqrt{6}}(1,-2,1)$$
$$= (1,1,0) - \frac{2}{3}(1,1,1) + \frac{1}{6}(1,-2,1)$$
$$= (1-\frac{2}{3}+\frac{1}{6}, 1-\frac{2}{3}-\frac{2}{6}, 0-\frac{2}{3}+\frac{1}{6}) = (\frac{1}{2}, 0, -\frac{1}{2})$$
$$\|w_3\| = \frac{1}{\sqrt{2}}$$
$$e_3 = \frac{1}{\sqrt{2}}(1, 0, -1)$$

**Result:**
$$e_1 = \frac{1}{\sqrt{3}}(1,1,1), \quad e_2 = \frac{1}{\sqrt{6}}(1,-2,1), \quad e_3 = \frac{1}{\sqrt{2}}(1,0,-1)$$

### 6. QR Decomposition

Gram-Schmidt gives us the **QR decomposition**:
$$A = QR$$

Where:
- **A** has columns v₁, v₂, ..., vₙ
- **Q** has columns e₁, e₂, ..., eₙ (orthonormal)
- **R** is upper triangular with rᵢⱼ = ⟨eᵢ|vⱼ⟩

**Explicit form:**
$$v_j = \sum_{i=1}^j r_{ij} e_i = r_{1j}e_1 + r_{2j}e_2 + \cdots + r_{jj}e_j$$

So:
- rᵢⱼ = ⟨eᵢ|vⱼ⟩ for i < j
- rⱼⱼ = ‖wⱼ‖

### 7. Gram-Schmidt on Function Spaces

**Example:** Orthogonalize {1, x, x²} on L²[-1, 1]

Inner product: ⟨f, g⟩ = ∫₋₁¹ f(x)g(x) dx

**Step 1:** v₁ = 1
$$\|1\|^2 = \int_{-1}^1 1 \, dx = 2$$
$$e_1 = \frac{1}{\sqrt{2}}$$

**Step 2:** v₂ = x
$$\langle e_1, x \rangle = \frac{1}{\sqrt{2}} \int_{-1}^1 x \, dx = 0$$ (odd function!)
$$w_2 = x - 0 = x$$
$$\|x\|^2 = \int_{-1}^1 x^2 \, dx = \frac{2}{3}$$
$$e_2 = \sqrt{\frac{3}{2}} x$$

**Step 3:** v₃ = x²
$$\langle e_1, x^2 \rangle = \frac{1}{\sqrt{2}} \cdot \frac{2}{3} = \frac{\sqrt{2}}{3}$$
$$\langle e_2, x^2 \rangle = \sqrt{\frac{3}{2}} \int_{-1}^1 x^3 \, dx = 0$$ (odd function!)
$$w_3 = x^2 - \frac{\sqrt{2}}{3} \cdot \frac{1}{\sqrt{2}} = x^2 - \frac{1}{3}$$

This gives the **Legendre polynomials** (up to normalization):
- P₀(x) = 1
- P₁(x) = x  
- P₂(x) = (3x² - 1)/2

### 8. Numerical Stability: Modified Gram-Schmidt

**Problem:** Classical Gram-Schmidt can lose orthogonality due to rounding errors.

**Solution:** Modified Gram-Schmidt (MGS)

Instead of computing all projections then subtracting, subtract each projection immediately:

```
for j = 1 to n:
    for i = 1 to j-1:
        v_j = v_j - ⟨e_i|v_j⟩ e_i  # Subtract immediately
    e_j = v_j / ||v_j||
```

MGS is mathematically equivalent but numerically more stable.

---

## 🔬 Quantum Mechanics Connection

### Constructing Orthonormal Measurement Bases

Given non-orthogonal states, Gram-Schmidt creates a valid measurement basis:

**Example:** Alice sends states |ψ₁⟩ = |0⟩ and |ψ₂⟩ = (|0⟩ + |1⟩)/√2.

To distinguish them optimally, Bob constructs an orthonormal basis:
- e₁ = |0⟩ (normalize |ψ₁⟩)
- w₂ = |ψ₂⟩ - ⟨e₁|ψ₂⟩ e₁ = (|0⟩+|1⟩)/√2 - (1/√2)|0⟩ = |1⟩/√2
- e₂ = |1⟩

Result: Bob measures in computational basis!

### Schmidt Decomposition Connection

For bipartite states |ψ⟩ ∈ ℋ_A ⊗ ℋ_B, the Schmidt decomposition:
$$|\psi\rangle = \sum_i \lambda_i |a_i\rangle |b_i\rangle$$

uses orthonormal bases {|aᵢ⟩} and {|bᵢ⟩}, found via Gram-Schmidt-like procedures.

### Quantum Error Correction

Constructing orthogonal code words uses Gram-Schmidt:
- Start with logical states
- Orthogonalize while preserving error-correction properties

---

## ✏️ Worked Examples

### Example 1: ℝ² Gram-Schmidt

Orthogonalize {(3, 4), (1, 0)} in ℝ².

**Step 1:**
$$e_1 = \frac{(3,4)}{5} = (0.6, 0.8)$$

**Step 2:**
$$\langle e_1, (1,0) \rangle = 0.6$$
$$w_2 = (1,0) - 0.6(0.6, 0.8) = (1-0.36, -0.48) = (0.64, -0.48)$$
$$\|w_2\| = \sqrt{0.64^2 + 0.48^2} = 0.8$$
$$e_2 = \frac{(0.64, -0.48)}{0.8} = (0.8, -0.6)$$

**Verify:** ⟨e₁, e₂⟩ = 0.6(0.8) + 0.8(-0.6) = 0 ✓

### Example 2: Complex Vectors

Orthogonalize {(1, i), (1, 1)} in ℂ².

**Step 1:**
$$\|v_1\|^2 = |1|^2 + |i|^2 = 2$$
$$e_1 = \frac{1}{\sqrt{2}}(1, i)$$

**Step 2:**
$$\langle e_1 | v_2 \rangle = \frac{1}{\sqrt{2}}(1^*(1) + (-i)^*(1)) = \frac{1}{\sqrt{2}}(1 - i)$$
$$w_2 = (1,1) - \frac{1-i}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1,i) = (1,1) - \frac{1-i}{2}(1,i)$$
$$= (1,1) - (\frac{1-i}{2}, \frac{i-i^2}{2}) = (1,1) - (\frac{1-i}{2}, \frac{1+i}{2})$$
$$= (\frac{1+i}{2}, \frac{1-i}{2})$$
$$\|w_2\|^2 = |\frac{1+i}{2}|^2 + |\frac{1-i}{2}|^2 = \frac{2}{4} + \frac{2}{4} = 1$$
$$e_2 = (\frac{1+i}{2}, \frac{1-i}{2})$$

### Example 3: QR Decomposition

Find QR decomposition of A = $\begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 1 & 1 \end{pmatrix}$

Columns: v₁ = (1,1,1), v₂ = (1,0,1)

**Gram-Schmidt:** (from earlier example)
- e₁ = (1,1,1)/√3
- For v₂: ⟨e₁|v₂⟩ = 2/√3
- w₂ = (1,0,1) - (2/3)(1,1,1) = (1/3, -2/3, 1/3)
- e₂ = (1,-2,1)/√6

**Q and R:**
$$Q = \begin{pmatrix} 1/\sqrt{3} & 1/\sqrt{6} \\ 1/\sqrt{3} & -2/\sqrt{6} \\ 1/\sqrt{3} & 1/\sqrt{6} \end{pmatrix}$$

$$R = \begin{pmatrix} \|v_1\| & \langle e_1|v_2\rangle \\ 0 & \|w_2\| \end{pmatrix} = \begin{pmatrix} \sqrt{3} & 2/\sqrt{3} \\ 0 & \sqrt{6}/3 \end{pmatrix}$$

Verify: QR = A ✓

---

## 📝 Practice Problems

### Level 1: Basic Algorithm

1. Apply Gram-Schmidt to {(1, 1), (2, 0)} in ℝ².

2. Apply Gram-Schmidt to {(1, 0, 0), (1, 1, 0), (1, 1, 1)} in ℝ³.

3. Verify your answers to #1 and #2 are orthonormal.

4. Apply Gram-Schmidt to {|0⟩, |+⟩} in ℂ² where |+⟩ = (|0⟩+|1⟩)/√2.

### Level 2: QR Decomposition

5. Find the QR decomposition of A = $\begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}$

6. Show that R is upper triangular by construction of Gram-Schmidt.

7. If A = QR and Q is orthogonal, what is A^T A in terms of R?

8. Use QR decomposition to solve the least squares problem Ax ≈ b.

### Level 3: Function Spaces

9. Orthogonalize {1, x, x²} on L²[0, 1] (interval [0,1], not [-1,1]!).

10. Show that the resulting polynomials are shifted Legendre polynomials.

11. Orthogonalize {eˣ, e⁻ˣ} on L²[-1, 1].

12. Orthogonalize {sin(x), cos(x), sin(2x)} on L²[0, 2π].

### Level 4: Theory and Applications

13. Prove that if {v₁,...,vₖ} is linearly dependent, Gram-Schmidt will produce wⱼ = 0 for some j.

14. Show that QR decomposition is unique if we require rᵢᵢ > 0.

15. Implement Modified Gram-Schmidt and compare numerical stability.

16. Apply Gram-Schmidt to the rows of a matrix. What does this give?

---

## 📊 Answers and Hints

1. e₁ = (1,1)/√2, e₂ = (1,-1)/√2
2. e₁ = (1,0,0), e₂ = (0,1,0), e₃ = (0,0,1) (already orthogonal!)
3. Check all inner products
4. e₁ = |0⟩, e₂ = |1⟩
5. Work through step by step
6. vⱼ = Σᵢ≤ⱼ rᵢⱼeᵢ, so rᵢⱼ = 0 for i > j
7. AᵀA = RᵀQᵀQR = RᵀR
8. QRx = b → Rx = Qᵀb (easy to solve!)
9. Different from [-1,1] case
10. Look up shifted Legendre polynomials
11. Result involves cosh and sinh
12. sin(x) and cos(x) already orthogonal; compute projections for sin(2x)
13. wⱼ = vⱼ - proj = 0 means vⱼ ∈ span{v₁,...,vⱼ₋₁}
14. Uniqueness from algorithm structure
15. Code exercise
16. Row space orthonormalization

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Lab 1: Classical Gram-Schmidt
# ============================================

def gram_schmidt_classical(V):
    """
    Classical Gram-Schmidt orthogonalization
    V: matrix with vectors as columns
    Returns: Q (orthonormal columns), R (upper triangular)
    """
    n, k = V.shape
    Q = np.zeros((n, k), dtype=complex)
    R = np.zeros((k, k), dtype=complex)
    
    for j in range(k):
        # Start with original vector
        w = V[:, j].copy()
        
        # Subtract projections onto previous vectors
        for i in range(j):
            R[i, j] = np.vdot(Q[:, i], V[:, j])
            w = w - R[i, j] * Q[:, i]
        
        # Normalize
        R[j, j] = np.linalg.norm(w)
        if R[j, j] > 1e-10:
            Q[:, j] = w / R[j, j]
        else:
            print(f"Warning: Vector {j} is linearly dependent")
            Q[:, j] = 0
    
    return Q, R

# Test
V = np.array([[1, 1, 1],
              [1, 0, 1],
              [1, 1, 0]], dtype=float).T  # Columns

Q, R = gram_schmidt_classical(V)
print("=== Classical Gram-Schmidt ===")
print(f"Q = \n{Q}")
print(f"\nR = \n{R}")
print(f"\nQ @ R = \n{Q @ R}")
print(f"\nOriginal V = \n{V}")
print(f"\nQ^T @ Q = \n{Q.T @ Q}")  # Should be identity

# ============================================
# Lab 2: Modified Gram-Schmidt
# ============================================

def gram_schmidt_modified(V):
    """
    Modified Gram-Schmidt (numerically stable)
    """
    n, k = V.shape
    Q = V.astype(complex).copy()
    R = np.zeros((k, k), dtype=complex)
    
    for i in range(k):
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]
        
        for j in range(i+1, k):
            R[i, j] = np.vdot(Q[:, i], Q[:, j])
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
    
    return Q, R

# Compare stability
np.random.seed(42)
n = 20
# Nearly dependent vectors (ill-conditioned)
V_ill = np.random.randn(n, n)
V_ill[:, 1] = V_ill[:, 0] + 1e-8 * np.random.randn(n)

Q_classical, R_classical = gram_schmidt_classical(V_ill)
Q_modified, R_modified = gram_schmidt_modified(V_ill)
Q_numpy, R_numpy = np.linalg.qr(V_ill)

print("\n=== Numerical Stability Comparison ===")
print(f"Classical: ||Q^T Q - I|| = {np.linalg.norm(Q_classical.T @ Q_classical - np.eye(n)):.2e}")
print(f"Modified:  ||Q^T Q - I|| = {np.linalg.norm(Q_modified.T @ Q_modified - np.eye(n)):.2e}")
print(f"NumPy QR:  ||Q^T Q - I|| = {np.linalg.norm(Q_numpy.T @ Q_numpy - np.eye(n)):.2e}")

# ============================================
# Lab 3: Complex Vectors
# ============================================

# Quantum state orthogonalization
psi1 = np.array([1, 1j], dtype=complex)
psi2 = np.array([1, 1], dtype=complex)

V_complex = np.column_stack([psi1, psi2])
Q_complex, R_complex = gram_schmidt_modified(V_complex)

print("\n=== Complex Gram-Schmidt ===")
print(f"Original states:\n  |ψ₁⟩ = {psi1}\n  |ψ₂⟩ = {psi2}")
print(f"\nOrthonormal states:\n  |e₁⟩ = {Q_complex[:, 0]}\n  |e₂⟩ = {Q_complex[:, 1]}")
print(f"\nVerify orthonormality:")
print(f"  ⟨e₁|e₁⟩ = {np.vdot(Q_complex[:,0], Q_complex[:,0]):.4f}")
print(f"  ⟨e₂|e₂⟩ = {np.vdot(Q_complex[:,1], Q_complex[:,1]):.4f}")
print(f"  ⟨e₁|e₂⟩ = {np.vdot(Q_complex[:,0], Q_complex[:,1]):.4f}")

# ============================================
# Lab 4: Function Space (Polynomials)
# ============================================

from scipy.integrate import quad

def inner_product_L2(f, g, a=-1, b=1):
    """Compute L² inner product on [a, b]"""
    real_part, _ = quad(lambda x: np.real(np.conj(f(x)) * g(x)), a, b)
    imag_part, _ = quad(lambda x: np.imag(np.conj(f(x)) * g(x)), a, b)
    return real_part + 1j * imag_part

def gram_schmidt_functions(functions, a=-1, b=1):
    """Gram-Schmidt for functions on [a,b]"""
    orthonormal = []
    
    for f in functions:
        # Subtract projections
        w = f
        for e in orthonormal:
            coeff = inner_product_L2(e, f, a, b)
            w = lambda x, w=w, e=e, c=coeff: w(x) - c * e(x)
        
        # Normalize
        norm_sq = inner_product_L2(w, w, a, b)
        norm = np.sqrt(np.real(norm_sq))
        e_new = lambda x, w=w, n=norm: w(x) / n
        orthonormal.append(e_new)
    
    return orthonormal

# Orthogonalize monomials
f0 = lambda x: 1 + 0*x
f1 = lambda x: x
f2 = lambda x: x**2

# Use numpy for polynomial orthogonalization
# Gram matrix for {1, x, x²} on [-1, 1]
G = np.zeros((3, 3))
polys = [f0, f1, f2]
for i in range(3):
    for j in range(3):
        G[i, j] = np.real(inner_product_L2(polys[i], polys[j]))

print("\n=== Polynomial Gram-Schmidt ===")
print(f"Gram matrix for {{1, x, x²}}:\n{G}")

# The Legendre polynomials (properly normalized)
x_plot = np.linspace(-1, 1, 100)

# Legendre polynomials
P0 = np.ones_like(x_plot)
P1 = x_plot
P2 = (3*x_plot**2 - 1) / 2

plt.figure(figsize=(10, 6))
plt.plot(x_plot, P0, label='P₀(x) = 1')
plt.plot(x_plot, P1, label='P₁(x) = x')
plt.plot(x_plot, P2, label='P₂(x) = (3x²-1)/2')
plt.xlabel('x')
plt.ylabel('Pₙ(x)')
plt.title('Legendre Polynomials (from Gram-Schmidt on {1, x, x²})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('legendre_polynomials.png', dpi=150)
plt.show()

# ============================================
# Lab 5: Visualization of Gram-Schmidt
# ============================================

fig = plt.figure(figsize=(15, 5))

# 2D case
ax1 = fig.add_subplot(131)
v1 = np.array([2, 1])
v2 = np.array([1, 2])

e1 = v1 / np.linalg.norm(v1)
proj = np.dot(e1, v2) * e1
w2 = v2 - proj
e2 = w2 / np.linalg.norm(w2)

ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.03, label='v₁', alpha=0.5)
ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.03, label='v₂', alpha=0.5)
ax1.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.05, label='e₁')
ax1.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, 
           color='green', width=0.03, label='proj')
ax1.quiver(proj[0], proj[1], w2[0], w2[1], angles='xy', scale_units='xy', scale=1, 
           color='orange', width=0.03, label='w₂')
ax1.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.05, label='e₂')

ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 2.5)
ax1.set_aspect('equal')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_title('Gram-Schmidt in ℝ²')

# 3D case
ax2 = fig.add_subplot(132, projection='3d')
v1 = np.array([1, 1, 1])
v2 = np.array([1, 0, 1])
v3 = np.array([1, 1, 0])

V = np.column_stack([v1, v2, v3])
Q, R = gram_schmidt_modified(V)

# Plot original and orthonormal
colors = ['blue', 'red', 'green']
for i, (v, e, c) in enumerate(zip([v1, v2, v3], Q.T, colors)):
    ax2.quiver(0, 0, 0, v[0], v[1], v[2], color=c, alpha=0.3, arrow_length_ratio=0.1)
    ax2.quiver(0, 0, 0, np.real(e[0]), np.real(e[1]), np.real(e[2]), 
               color=c, arrow_length_ratio=0.1, linewidth=2)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Gram-Schmidt in ℝ³\n(faded: original, solid: orthonormal)')

# Orthogonality quality vs condition number
ax3 = fig.add_subplot(133)
cond_numbers = np.logspace(0, 8, 20)
classical_errors = []
modified_errors = []

for cond in cond_numbers:
    # Create matrix with specific condition number
    U, _, Vh = np.linalg.svd(np.random.randn(10, 10))
    S = np.diag(np.logspace(0, -np.log10(cond), 10))
    A = U @ S @ Vh
    
    Q_c, _ = gram_schmidt_classical(A)
    Q_m, _ = gram_schmidt_modified(A)
    
    classical_errors.append(np.linalg.norm(Q_c.T @ Q_c - np.eye(10)))
    modified_errors.append(np.linalg.norm(Q_m.T @ Q_m - np.eye(10)))

ax3.loglog(cond_numbers, classical_errors, 'b-o', label='Classical GS')
ax3.loglog(cond_numbers, modified_errors, 'r-s', label='Modified GS')
ax3.set_xlabel('Condition Number')
ax3.set_ylabel('||Q^T Q - I||')
ax3.set_title('Numerical Stability')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gram_schmidt_visualization.png', dpi=150)
plt.show()

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Read Axler 6.B on Gram-Schmidt
- [ ] Execute algorithm by hand on 3 examples
- [ ] Understand why the algorithm works
- [ ] Learn QR decomposition connection
- [ ] Complete Level 1-2 practice problems
- [ ] Complete computational lab
- [ ] Understand numerical stability issues

---

## 📓 Reflection Questions

1. Why does Gram-Schmidt always succeed for linearly independent vectors?

2. What happens if you apply Gram-Schmidt to a linearly dependent set?

3. Why is Modified Gram-Schmidt more numerically stable?

4. How does QR decomposition help solve least squares problems?

---

## 🔜 Preview: Tomorrow's Topics

**Day 110: Orthonormal Bases and Projections**
- Properties of orthonormal bases
- Parseval's identity
- Bessel's inequality
- Least squares approximation

---

*"Gram-Schmidt is the assembly line of linear algebra."*
— Gilbert Strang
