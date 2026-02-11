# Day 94: Matrix Operations and Composition

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Matrix Operations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Perform matrix addition and scalar multiplication
2. Understand and compute matrix multiplication
3. Connect composition of maps to matrix multiplication
4. Work with matrix powers and polynomial functions
5. Understand when matrices commute (and why it matters for QM)
6. Begin thinking about invertibility

---

## 📚 Required Reading

**Before starting:**
- Axler, Chapter 3.C continued (Matrix multiplication)
- Strang, Chapter 2.4-2.5: Matrix Operations

---

## 🌅 Morning Session: Theory (3.5 hours)

### Part 1: Elementary Matrix Operations (45 min)

#### Matrix Addition

For A, B ∈ M_{m×n}(F):
$$(A + B)_{ij} = A_{ij} + B_{ij}$$

**Properties:**
- Commutative: A + B = B + A
- Associative: (A + B) + C = A + (B + C)
- Zero element: A + 0 = A
- Additive inverse: A + (-A) = 0

#### Scalar Multiplication

For c ∈ F and A ∈ M_{m×n}(F):
$$(cA)_{ij} = c \cdot A_{ij}$$

**Properties:**
- Distributive: c(A + B) = cA + cB
- Distributive: (c + d)A = cA + dA
- Associative: c(dA) = (cd)A
- Identity: 1 · A = A

#### Connection to Linear Maps

These operations correspond to:
- (S + T)(v) = S(v) + T(v) ↔ M(S + T) = M(S) + M(T)
- (cT)(v) = cT(v) ↔ M(cT) = cM(T)

### Part 2: Matrix Multiplication — The Big One (90 min)

#### Definition

For A ∈ M_{m×n}(F) and B ∈ M_{n×p}(F), the product AB ∈ M_{m×p}(F) is:
$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

**Mnemonic:** Entry (i,j) of AB = (row i of A) · (column j of B)

#### Size Compatibility

A is m×n, B is n×p → AB is m×p

The inner dimensions must match: (m × **n**) × (**n** × p)

#### Fundamental Connection: Composition

**Theorem:** If S: U → V and T: V → W are linear maps, then:
$$M(T \circ S) = M(T) \cdot M(S)$$

This is **why** matrix multiplication is defined this way!

**Proof sketch:**
For basis vector uⱼ of U:
- (T ∘ S)(uⱼ) = T(S(uⱼ))
- S(uⱼ) = Σₖ (M(S))ₖⱼ vₖ (expressing in V's basis)
- T(S(uⱼ)) = T(Σₖ (M(S))ₖⱼ vₖ) = Σₖ (M(S))ₖⱼ T(vₖ)
- = Σₖ (M(S))ₖⱼ Σᵢ (M(T))ᵢₖ wᵢ
- = Σᵢ (Σₖ (M(T))ᵢₖ (M(S))ₖⱼ) wᵢ

The coefficient of wᵢ is exactly (M(T)·M(S))ᵢⱼ.

#### Four Ways to Think About Matrix Multiplication

**1. Entry-by-Entry (Definition)**
$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

**2. Row-Column Dot Products**
Entry (i,j) = (row i of A) · (column j of B)

**3. Column Picture**
Column j of AB = A × (column j of B)

Each column of AB is a linear combination of columns of A.

**4. Outer Product Sum**
$$AB = \sum_{k=1}^{n} (\text{column } k \text{ of } A) \otimes (\text{row } k \text{ of } B)$$

where ⊗ is the outer product.

#### Properties of Matrix Multiplication

**Associative:** (AB)C = A(BC)

**Distributive:** 
- A(B + C) = AB + AC
- (A + B)C = AC + BC

**NOT Commutative in general:** AB ≠ BA

**Identity:** AI = IA = A (for I = identity matrix of appropriate size)

#### Non-Commutativity is Fundamental!

**Example:**
$$A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$$

$$AB = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad BA = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

AB ≠ BA, and even AB ≠ 0 while BA ≠ 0 with different results!

### Part 3: Special Matrices (30 min)

#### Identity Matrix

$$I_n = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}$$

Represents the identity map: I(v) = v

#### Diagonal Matrices

$$D = \begin{pmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{pmatrix}$$

Diagonal matrices commute with each other!

#### The Transpose

$$A^T = (A^T)_{ij} = A_{ji}$$

**Properties:**
- (A + B)ᵀ = Aᵀ + Bᵀ
- (cA)ᵀ = cAᵀ
- (AB)ᵀ = BᵀAᵀ (order reverses!)
- (Aᵀ)ᵀ = A

### Part 4: Matrix Powers and Polynomials (30 min)

#### Powers of Square Matrices

For square matrix A:
- A⁰ = I
- A¹ = A
- Aⁿ = A · A · ... · A (n times)

#### Polynomial Functions of Matrices

If p(x) = aₙxⁿ + ... + a₁x + a₀, then:
$$p(A) = a_n A^n + \cdots + a_1 A + a_0 I$$

This is well-defined because matrix multiplication is associative.

#### Example: Nilpotent Matrix

$$N = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

$$N^2 = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad N^3 = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} = 0$$

N is **nilpotent**: some power equals zero.

### Part 5: Quantum Connection — Commutators (30 min)

#### Why Non-Commutativity Matters

In quantum mechanics, observables are represented by operators.

If two operators **commute** (AB = BA), they can be measured simultaneously.

If they **don't commute**, the uncertainty principle applies!

#### The Commutator

$$[A, B] = AB - BA$$

**Properties:**
- [A, A] = 0
- [A, B] = -[B, A]
- [A, B + C] = [A, B] + [A, C]
- [A, BC] = [A, B]C + B[A, C]

#### Canonical Commutation Relation

In quantum mechanics:
$$[\hat{x}, \hat{p}] = i\hbar$$

This is the mathematical content of the Heisenberg uncertainty principle!

#### Pauli Matrix Commutators

$$[\sigma_x, \sigma_y] = \sigma_x \sigma_y - \sigma_y \sigma_x$$

$$\sigma_x \sigma_y = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} = \begin{pmatrix} i & 0 \\ 0 & -i \end{pmatrix} = i\sigma_z$$

$$\sigma_y \sigma_x = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} -i & 0 \\ 0 & i \end{pmatrix} = -i\sigma_z$$

$$[\sigma_x, \sigma_y] = i\sigma_z - (-i\sigma_z) = 2i\sigma_z$$

**General Pauli relations:**
$$[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k$$

where ε is the Levi-Civita symbol.

---

## 🌆 Afternoon Session: Problem Solving (2.5 hours)

### Problem Set A: Matrix Multiplication (50 min)

**Problem 1.** Compute AB and BA for:
$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**Solution:**
$$AB = \begin{pmatrix} 1·0+2·1 & 1·1+2·0 \\ 3·0+4·1 & 3·1+4·0 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 4 & 3 \end{pmatrix}$$

$$BA = \begin{pmatrix} 0·1+1·3 & 0·2+1·4 \\ 1·1+0·3 & 1·2+0·4 \end{pmatrix} = \begin{pmatrix} 3 & 4 \\ 1 & 2 \end{pmatrix}$$

Note: AB ≠ BA.

**Problem 2.** Let A be m×n and B be n×p. How many scalar multiplications are needed for AB?

**Solution:**
Each entry of AB requires n multiplications (dot product of row and column).
AB has m×p entries.
Total: mnp multiplications.

**Problem 3.** Prove that (AB)C = A(BC) for all compatible matrices.

**Proof:**
$$((AB)C)_{ij} = \sum_k (AB)_{ik} C_{kj} = \sum_k \left(\sum_l A_{il} B_{lk}\right) C_{kj}$$
$$= \sum_l A_{il} \sum_k B_{lk} C_{kj} = \sum_l A_{il} (BC)_{lj} = (A(BC))_{ij}$$

**Problem 4.** Show that the product of two upper triangular matrices is upper triangular.

**Proof:**
Let A, B be upper triangular (A_{ij} = 0 for i > j, same for B).
$$(AB)_{ij} = \sum_k A_{ik} B_{kj}$$

For i > j, we need to show this sum is zero.
- If k < i, then A_{ik} = 0 (since k < i means i > k)
- If k ≥ i > j, then k > j, so B_{kj} = 0

In all cases, each term has a zero factor. Therefore (AB)_{ij} = 0 for i > j. ∎

### Problem Set B: Composition and Powers (50 min)

**Problem 5.** Let T: ℝ² → ℝ² be rotation by 30° and S: ℝ² → ℝ² be rotation by 60°. 
Find the matrices of T, S, TS, and ST.

**Solution:**
$$M(T) = \begin{pmatrix} \cos 30° & -\sin 30° \\ \sin 30° & \cos 30° \end{pmatrix} = \begin{pmatrix} \sqrt{3}/2 & -1/2 \\ 1/2 & \sqrt{3}/2 \end{pmatrix}$$

$$M(S) = \begin{pmatrix} \cos 60° & -\sin 60° \\ \sin 60° & \cos 60° \end{pmatrix} = \begin{pmatrix} 1/2 & -\sqrt{3}/2 \\ \sqrt{3}/2 & 1/2 \end{pmatrix}$$

M(TS) = M(T)M(S) = M(S∘T) should be rotation by 90°:
$$M(TS) = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

Since rotations commute: M(ST) = M(TS) = rotation by 90°.

**Problem 6.** For what matrices A does A² = I?

**Solution:**
These are called **involutions**. Examples:
- I (trivially)
- -I
- Any reflection: [[cos2θ, sin2θ], [sin2θ, -cos2θ]]
- Permutation matrices of order 2
- Any diagonal matrix with ±1 on diagonal

General form: eigenvalues must be ±1.

**Problem 7.** Compute A^n for:
$$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

**Solution:**
$$A^2 = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}, \quad A^3 = \begin{pmatrix} 1 & 3 \\ 0 & 1 \end{pmatrix}$$

Pattern: $A^n = \begin{pmatrix} 1 & n \\ 0 & 1 \end{pmatrix}$

**Proof by induction:**
$$A^{n+1} = A^n \cdot A = \begin{pmatrix} 1 & n \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & n+1 \\ 0 & 1 \end{pmatrix}$$ ✓

### Problem Set C: Commutators (50 min)

**Problem 8.** Verify that [σx, σz] = -2iσy.

**Solution:**
$$\sigma_x \sigma_z = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

$$\sigma_z \sigma_x = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$$

$$[\sigma_x, \sigma_z] = \begin{pmatrix} 0 & -2 \\ 2 & 0 \end{pmatrix} = -2i\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} = -2i\sigma_y$$ ✓

**Problem 9.** Prove that [A, A^n] = 0 for all n ≥ 0.

**Proof:**
A commutes with itself, so A commutes with any power of A:
AA^n = A^{n+1} = A^nA

Therefore [A, A^n] = AA^n - A^nA = 0. ∎

**Problem 10.** Show that [A, B] = 0 implies [A^m, B^n] = 0 for all m, n ≥ 1.

**Proof:**
First show [A, B^n] = 0 by induction on n:
- Base: [A, B] = 0 given
- Step: [A, B^{n+1}] = [A, B^nB] = [A, B^n]B + B^n[A, B] = 0·B + B^n·0 = 0

Then [A^m, B^n] = 0 follows similarly by induction on m. ∎

---

## 🌙 Evening Session: Computational Lab (1 hour)

```python
"""
Day 94: Matrix Operations and Composition
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# Part 1: Matrix Multiplication Methods
# =============================================================

print("="*60)
print("Matrix Multiplication: Different Perspectives")
print("="*60)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"\nA = \n{A}")
print(f"\nB = \n{B}")

# Method 1: Direct computation
print(f"\nAB (direct) = \n{A @ B}")

# Method 2: Row-column dot products
print("\nMethod 2: Row-Column Dot Products")
result = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        result[i, j] = np.dot(A[i, :], B[:, j])
        print(f"  (AB)[{i},{j}] = A[{i},:] · B[:,{j}] = {A[i,:]} · {B[:,j]} = {result[i,j]}")

# Method 3: Column picture
print("\nMethod 3: Column Picture")
print("Each column of AB is A times corresponding column of B:")
for j in range(2):
    print(f"  (AB)[:,{j}] = A @ B[:,{j}] = {A @ B[:,j]}")

# Method 4: Outer product sum
print("\nMethod 4: Outer Product Sum")
outer_sum = np.zeros((2, 2))
for k in range(2):
    outer = np.outer(A[:, k], B[k, :])
    print(f"  A[:,{k}] ⊗ B[{k},:] = {A[:,k]} ⊗ {B[k,:]} = \n{outer}")
    outer_sum += outer
print(f"\nSum of outer products = \n{outer_sum}")

# =============================================================
# Part 2: Non-Commutativity
# =============================================================

print("\n" + "="*60)
print("Non-Commutativity of Matrix Multiplication")
print("="*60)

print(f"\nAB = \n{A @ B}")
print(f"\nBA = \n{B @ A}")
print(f"\nAB = BA? {np.allclose(A @ B, B @ A)}")
print(f"AB - BA = \n{A @ B - B @ A}")

# Pauli matrices (important for QM)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

print("\n--- Pauli Matrices ---")
print(f"σx·σy = \n{sigma_x @ sigma_y}")
print(f"σy·σx = \n{sigma_y @ sigma_x}")
print(f"[σx, σy] = σxσy - σyσx = \n{sigma_x @ sigma_y - sigma_y @ sigma_x}")
print(f"2iσz = \n{2j * sigma_z}")
print(f"[σx, σy] = 2iσz? {np.allclose(sigma_x @ sigma_y - sigma_y @ sigma_x, 2j * sigma_z)}")

# =============================================================
# Part 3: Matrix Powers
# =============================================================

print("\n" + "="*60)
print("Matrix Powers")
print("="*60)

# Nilpotent matrix
N = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
print(f"Nilpotent N = \n{N}")
print(f"N² = \n{np.linalg.matrix_power(N, 2)}")
print(f"N³ = \n{np.linalg.matrix_power(N, 3)}")

# Rotation matrix powers
theta = np.pi / 6  # 30°
R = np.array([[np.cos(theta), -np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]])

print(f"\nRotation by 30°:")
print(f"R = \n{R.round(4)}")
print(f"R⁶ (rotation by 180°) = \n{np.linalg.matrix_power(R, 6).round(4)}")
print(f"R¹² (rotation by 360°) = \n{np.linalg.matrix_power(R, 12).round(4)}")
print(f"R¹² ≈ I? {np.allclose(np.linalg.matrix_power(R, 12), np.eye(2))}")

# Upper triangular pattern
U = np.array([[1, 1], [0, 1]])
print(f"\nUpper triangular U = [[1,1],[0,1]]")
for n in range(1, 6):
    print(f"U^{n} = \n{np.linalg.matrix_power(U, n)}")

# =============================================================
# Part 4: Composition of Transformations
# =============================================================

print("\n" + "="*60)
print("Composition = Matrix Multiplication")
print("="*60)

def plot_composition():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original shape: unit square
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])
    
    # Transformations
    scale = np.array([[2, 0], [0, 1]])  # Scale x by 2
    rotate = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                       [np.sin(np.pi/4), np.cos(np.pi/4)]])  # Rotate 45°
    
    # Plot original
    axes[0].plot(square[0], square[1], 'b-', linewidth=2)
    axes[0].fill(square[0], square[1], alpha=0.3)
    axes[0].set_title('Original')
    axes[0].set_xlim(-2, 3)
    axes[0].set_ylim(-2, 3)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot after scaling
    scaled = scale @ square
    axes[1].plot(scaled[0], scaled[1], 'r-', linewidth=2)
    axes[1].fill(scaled[0], scaled[1], alpha=0.3, color='red')
    axes[1].set_title('After Scale (A)')
    axes[1].set_xlim(-2, 3)
    axes[1].set_ylim(-2, 3)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    # Plot after rotation
    rotated = rotate @ square
    axes[2].plot(rotated[0], rotated[1], 'g-', linewidth=2)
    axes[2].fill(rotated[0], rotated[1], alpha=0.3, color='green')
    axes[2].set_title('After Rotate (B)')
    axes[2].set_xlim(-2, 3)
    axes[2].set_ylim(-2, 3)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    
    # Plot composition: rotate then scale (scale @ rotate @ square)
    composed = scale @ (rotate @ square)  # or (scale @ rotate) @ square
    axes[3].plot(composed[0], composed[1], 'm-', linewidth=2)
    axes[3].fill(composed[0], composed[1], alpha=0.3, color='magenta')
    axes[3].set_title('Rotate then Scale (AB)')
    axes[3].set_xlim(-2, 3)
    axes[3].set_ylim(-2, 3)
    axes[3].set_aspect('equal')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day94_composition.png', dpi=150)
    plt.show()
    
    # Show that order matters
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    
    # Scale then rotate
    sr = rotate @ scale @ square
    axes2[0].plot(sr[0], sr[1], 'b-', linewidth=2)
    axes2[0].fill(sr[0], sr[1], alpha=0.3)
    axes2[0].set_title('Scale THEN Rotate: R(Sx)')
    axes2[0].set_xlim(-3, 3)
    axes2[0].set_ylim(-3, 3)
    axes2[0].set_aspect('equal')
    axes2[0].grid(True, alpha=0.3)
    
    # Rotate then scale
    rs = scale @ rotate @ square
    axes2[1].plot(rs[0], rs[1], 'r-', linewidth=2)
    axes2[1].fill(rs[0], rs[1], alpha=0.3, color='red')
    axes2[1].set_title('Rotate THEN Scale: S(Rx)')
    axes2[1].set_xlim(-3, 3)
    axes2[1].set_ylim(-3, 3)
    axes2[1].set_aspect('equal')
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day94_order_matters.png', dpi=150)
    plt.show()


plot_composition()

# =============================================================
# Part 5: Quantum Gate Composition
# =============================================================

print("\n" + "="*60)
print("Quantum Gate Composition")
print("="*60)

# Define gates
I = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = sigma_x
Z = sigma_z

# Basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

print("Hadamard gate H:")
print(f"H|0⟩ = {H @ ket_0} = |+⟩")
print(f"H|1⟩ = {H @ ket_1} = |-⟩")

print("\nH² = I?")
print(f"H² = \n{(H @ H).round(4)}")
print(f"H² = I: {np.allclose(H @ H, I)}")

print("\nXZ vs ZX:")
print(f"XZ = \n{X @ Z}")
print(f"ZX = \n{Z @ X}")
print(f"XZ = -ZX? {np.allclose(X @ Z, -Z @ X)}")

# HXH = Z (Hadamard transforms between X and Z bases)
print("\nH·X·H = Z? (Basis transformation)")
print(f"HXH = \n{(H @ X @ H).round(4)}")
print(f"Z = \n{Z}")
print(f"HXH = Z: {np.allclose(H @ X @ H, Z)}")

# =============================================================
# Part 6: Commutator Analysis
# =============================================================

print("\n" + "="*60)
print("Commutator Analysis")
print("="*60)

def commutator(A, B):
    """Compute [A, B] = AB - BA"""
    return A @ B - B @ A

def anticommutator(A, B):
    """Compute {A, B} = AB + BA"""
    return A @ B + B @ A

# Pauli algebra
print("Pauli Commutators [σi, σj] = 2iεijk σk:")
print(f"[σx, σy] = 2iσz: {np.allclose(commutator(sigma_x, sigma_y), 2j*sigma_z)}")
print(f"[σy, σz] = 2iσx: {np.allclose(commutator(sigma_y, sigma_z), 2j*sigma_x)}")
print(f"[σz, σx] = 2iσy: {np.allclose(commutator(sigma_z, sigma_x), 2j*sigma_y)}")

print("\nPauli Anticommutators {σi, σj} = 2δij I:")
print(f"{{σx, σx}} = 2I: {np.allclose(anticommutator(sigma_x, sigma_x), 2*I)}")
print(f"{{σx, σy}} = 0: {np.allclose(anticommutator(sigma_x, sigma_y), 0)}")

# Which matrices commute?
print("\n--- Commutation Analysis ---")
matrices = {'I': I, 'X': X, 'Z': Z, 'H': H}
for name1, M1 in matrices.items():
    for name2, M2 in matrices.items():
        if name1 < name2:  # Avoid duplicates
            comm = np.allclose(commutator(M1, M2), 0)
            print(f"[{name1}, {name2}] = 0: {comm}")
```

---

## 📝 Homework

### Written Problems

1. Compute A³ where A = [[0, 1, 0], [0, 0, 1], [1, 0, 0]].

2. Let A be an n×n matrix. Prove that if A² = 0 (nilpotent of index 2), then (I - A) is invertible with inverse (I + A).

3. Show that for any matrices A, B: (AB)ᵀ = BᵀAᵀ.

4. Prove that if A and B commute, then (A + B)² = A² + 2AB + B².

5. Find all 2×2 matrices that commute with [[1, 0], [0, 2]].

6. In quantum computing, the CNOT gate (on 2 qubits) is:
   $$CNOT = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$
   Verify that CNOT² = I.

---

## ✅ Daily Checklist

- [ ] Understand matrix multiplication definition
- [ ] Can multiply matrices by hand
- [ ] Understand composition ↔ multiplication
- [ ] Know why non-commutativity matters
- [ ] Understand commutators for QM
- [ ] Completed all problem sets
- [ ] Ran computational lab

---

## 🔮 Preview: Tomorrow

**Day 95: Kernel and Range**
- Null space (kernel) of a linear map
- Image (range) of a linear map  
- Both are subspaces
- Connection to solving Ax = b

---

*"Do not worry about your difficulties in Mathematics. I can assure you mine are still greater."*
— Albert Einstein
