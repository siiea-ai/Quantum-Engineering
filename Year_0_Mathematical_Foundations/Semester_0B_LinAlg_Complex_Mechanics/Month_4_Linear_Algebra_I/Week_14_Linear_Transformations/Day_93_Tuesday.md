# Day 93: Matrix Representation of Linear Maps

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Matrices and Linear Maps |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Construct the matrix of a linear transformation
2. Understand how the matrix depends on choice of bases
3. Apply matrix multiplication to compute T(v)
4. Convert between linear maps and matrices fluently
5. Represent quantum operators as matrices in different bases

---

## 📚 Required Reading

**Before starting:**
- Axler, Chapter 3.C: "Matrices" (pp. 82-94)
- Strang, Chapter 2.4: "Matrix Operations"

---

## 🌅 Morning Session: Theory (3.5 hours)

### Part 1: The Matrix of a Linear Map (75 min)

#### The Fundamental Construction

Let T: V → W be a linear map where:
- V has basis B = {v₁, v₂, ..., vₙ}
- W has basis B' = {w₁, w₂, ..., wₘ}

**Definition:** The **matrix of T with respect to B and B'** is the m×n matrix M(T) where:

**Column j of M(T) contains the coordinates of T(vⱼ) in basis B'**

In symbols: If T(vⱼ) = a₁ⱼw₁ + a₂ⱼw₂ + ... + aₘⱼwₘ, then column j is:
$$[M(T)]_j = \begin{pmatrix} a_{1j} \\ a_{2j} \\ \vdots \\ a_{mj} \end{pmatrix}$$

#### Why This Definition?

**Key insight:** We want M(T) to satisfy:
$$[T(v)]_{B'} = M(T) \cdot [v]_B$$

That is: "coordinates of T(v) in W" = "matrix" × "coordinates of v in V"

**Verification:**
Let v = c₁v₁ + ... + cₙvₙ, so [v]_B = (c₁, ..., cₙ)ᵀ.

Then:
$$T(v) = T(\sum_{j=1}^n c_j v_j) = \sum_{j=1}^n c_j T(v_j)$$

If column j of M(T) gives coordinates of T(vⱼ), then M(T)·[v]_B gives:
$$\sum_{j=1}^n c_j \cdot (\text{column } j \text{ of } M(T)) = \sum_{j=1}^n c_j [T(v_j)]_{B'} = [T(v)]_{B'}$$

This is exactly what we wanted!

#### Notation

We write:
- M(T) or [T] for the matrix (when bases are understood)
- M(T)_{B',B} or [T]_{B',B} to emphasize the bases
- The subscript order is: output basis first, input basis second

#### Example 1: Rotation in ℝ²

Let R_θ: ℝ² → ℝ² be rotation by angle θ.
Use standard basis B = B' = {e₁, e₂} = {(1,0), (0,1)}.

Compute T(e₁) and T(e₂):
- R_θ(e₁) = R_θ(1, 0) = (cos θ, sin θ) = (cos θ)e₁ + (sin θ)e₂
- R_θ(e₂) = R_θ(0, 1) = (-sin θ, cos θ) = (-sin θ)e₁ + (cos θ)e₂

Therefore:
$$M(R_\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Check:** For θ = 90° = π/2:
$$M(R_{90°}) = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

R_{90°}(1, 2) should be (-2, 1). Verify:
$$\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} -2 \\ 1 \end{pmatrix} \checkmark$$

#### Example 2: Derivative as a Linear Map

Let D: P₃ → P₂ be differentiation: D(p) = p'.

Use bases:
- B = {1, x, x², x³} for P₃
- B' = {1, x, x²} for P₂

Compute D on each basis element:
- D(1) = 0 = 0·1 + 0·x + 0·x²
- D(x) = 1 = 1·1 + 0·x + 0·x²
- D(x²) = 2x = 0·1 + 2·x + 0·x²
- D(x³) = 3x² = 0·1 + 0·x + 3·x²

Therefore:
$$M(D) = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \end{pmatrix}$$

**Check:** D(2 + 3x - x² + 4x³) = 3 - 2x + 12x²

$$\begin{pmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \end{pmatrix} \begin{pmatrix} 2 \\ 3 \\ -1 \\ 4 \end{pmatrix} = \begin{pmatrix} 3 \\ -2 \\ 12 \end{pmatrix} \checkmark$$

### Part 2: Matrices Depend on Bases (45 min)

#### A Critical Point

The same linear transformation can have different matrices in different bases!

#### Example: Reflection Across y = x

T: ℝ² → ℝ² defined by T(x, y) = (y, x).

**Standard basis B = {e₁, e₂}:**
- T(e₁) = T(1, 0) = (0, 1) = e₂
- T(e₂) = T(0, 1) = (1, 0) = e₁

$$M(T)_B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**Diagonal basis B' = {(1,1), (1,-1)}:**
- T(1, 1) = (1, 1)  ← This is 1·(1,1) + 0·(1,-1)
- T(1, -1) = (-1, 1)  ← This is 0·(1,1) + (-1)·(1,-1)

$$M(T)_{B'} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Observation:** In the B' basis, the reflection is diagonal! This is because (1,1) and (1,-1) are **eigenvectors** of the reflection. (More on this in Week 15.)

#### Change of Basis Matrix

If we change bases, how do matrices transform?

Let B and B' be two bases for V.
The **change of basis matrix** P from B to B' satisfies:
$$[v]_{B'} = P \cdot [v]_B$$

Columns of P are: coordinates of B-vectors expressed in B'.

If T: V → V has matrix M_B in basis B and M_{B'} in basis B':
$$M_{B'} = P^{-1} M_B P$$

This is called a **similarity transformation**.

### Part 3: Matrix-Vector Multiplication (30 min)

#### Definition Revisited

For A an m×n matrix and x an n×1 vector:
$$(Ax)_i = \sum_{j=1}^n A_{ij} x_j$$

#### Column Picture

Ax = x₁(column 1) + x₂(column 2) + ... + xₙ(column n)

This is a **linear combination of the columns of A**!

#### Row Picture

Row i of Ax = (row i of A) · x = dot product

### Part 4: The Vector Space of Matrices (30 min)

#### M_{m×n}(F) is a Vector Space

The set of all m×n matrices with entries from field F forms a vector space:
- Zero element: Zero matrix (all entries 0)
- Addition: (A + B)_{ij} = A_{ij} + B_{ij}
- Scalar multiplication: (cA)_{ij} = c · A_{ij}

**Dimension:** dim(M_{m×n}(F)) = mn

**Standard basis:** E_{ij} = matrix with 1 in position (i,j) and 0 elsewhere
There are mn such matrices.

#### Connection to Linear Maps

**Theorem:** The vector space L(V, W) of linear maps from V to W is isomorphic to M_{m×n}(F), where n = dim(V) and m = dim(W).

In particular: dim(L(V, W)) = dim(V) · dim(W)

### Part 5: Quantum Connection — Operators as Matrices (30 min)

#### Quantum Operators in Computational Basis

In the computational basis {|0⟩, |1⟩}, a qubit operator Ô is a 2×2 matrix:
$$Ô = \begin{pmatrix} \langle 0|\hat{O}|0\rangle & \langle 0|\hat{O}|1\rangle \\ \langle 1|\hat{O}|0\rangle & \langle 1|\hat{O}|1\rangle \end{pmatrix}$$

#### Pauli Matrices

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

#### Hadamard Gate

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Column 1 = H|0⟩ = (|0⟩ + |1⟩)/√2
Column 2 = H|1⟩ = (|0⟩ - |1⟩)/√2

#### Operator in Different Bases

In the Hadamard basis {|+⟩, |-⟩}:
$$[\sigma_z]_{Had} = H \sigma_z H^{-1} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \sigma_x$$

In the computational basis, σ_z is diagonal.
In the Hadamard basis, σ_z becomes σ_x!

---

## 🌆 Afternoon Session: Problem Solving (2.5 hours)

### Problem Set A: Matrix Construction (50 min)

**Problem 1.** Find the matrix of T: ℝ³ → ℝ² defined by T(x, y, z) = (x + y, y - z) with respect to standard bases.

**Solution:**
- T(e₁) = T(1,0,0) = (1, 0)
- T(e₂) = T(0,1,0) = (1, 1)
- T(e₃) = T(0,0,1) = (0, -1)

$$M(T) = \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & -1 \end{pmatrix}$$

**Problem 2.** Find the matrix of the projection P: ℝ³ → ℝ³ onto the xy-plane.

**Solution:**
P(x, y, z) = (x, y, 0)
- P(e₁) = (1, 0, 0)
- P(e₂) = (0, 1, 0)
- P(e₃) = (0, 0, 0)

$$M(P) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

**Problem 3.** Let T: P₂ → P₂ be defined by T(p)(x) = p(x + 1). Find the matrix with respect to {1, x, x²}.

**Solution:**
- T(1) = 1 = 1·1 + 0·x + 0·x²
- T(x) = x + 1 = 1·1 + 1·x + 0·x²
- T(x²) = (x+1)² = x² + 2x + 1 = 1·1 + 2·x + 1·x²

$$M(T) = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \\ 0 & 0 & 1 \end{pmatrix}$$

**Problem 4.** Find the matrix of T: ℂ² → ℂ² defined by T(z₁, z₂) = (z₁ + iz₂, z₁ - iz₂).

**Solution:**
- T(e₁) = T(1, 0) = (1, 1)
- T(e₂) = T(0, 1) = (i, -i)

$$M(T) = \begin{pmatrix} 1 & i \\ 1 & -i \end{pmatrix}$$

### Problem Set B: Change of Basis (50 min)

**Problem 5.** Let B = {(1,1), (1,-1)} be a basis for ℝ².
a) Find the change of basis matrix P from standard to B.
b) If T: ℝ² → ℝ² has [T]_{std} = [2, 0; 0, 1], find [T]_B.

**Solution:**
a) We need P such that [v]_B = P·[v]_{std}.
   (1,0) = ½(1,1) + ½(1,-1), so [(1,0)]_B = (½, ½)
   (0,1) = ½(1,1) - ½(1,-1), so [(0,1)]_B = (½, -½)
   
   $$P = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}$$

b) [T]_B = P·[T]_{std}·P^{-1}

   $$P^{-1} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$
   
   $$[T]_B = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \begin{pmatrix} 3/2 & 1/2 \\ 1/2 & 3/2 \end{pmatrix}$$

**Problem 6.** Verify that rotation by θ in ℝ² has the same matrix in any orthonormal basis.

**Solution sketch:**
For any orthonormal basis, the matrix of rotation by θ must satisfy:
- Preserves lengths (orthogonal matrix)
- Determinant = 1 (not a reflection)
- Rotates counterclockwise by θ

In any orthonormal basis, this gives the same standard rotation matrix.

### Problem Set C: Proofs (50 min)

**Problem 7.** Prove that M(S + T) = M(S) + M(T) for linear maps S, T: V → W.

**Proof:**
Column j of M(S + T) = [(S + T)(vⱼ)]_{B'} = [S(vⱼ) + T(vⱼ)]_{B'}
= [S(vⱼ)]_{B'} + [T(vⱼ)]_{B'} = (column j of M(S)) + (column j of M(T))

Therefore M(S + T) = M(S) + M(T). ∎

**Problem 8.** Prove that M(cT) = cM(T) for scalar c and linear T.

**Proof:**
Column j of M(cT) = [(cT)(vⱼ)]_{B'} = [cT(vⱼ)]_{B'} = c[T(vⱼ)]_{B'} = c(column j of M(T))

Therefore M(cT) = cM(T). ∎

**Problem 9.** Prove: If A is the matrix of T: V → W and B is the matrix of S: W → X, then BA is the matrix of S∘T.

**Proof:**
We need to show [S(T(v))]_{B_X} = (BA)[v]_{B_V} for all v.

We have:
- [T(v)]_{B_W} = A[v]_{B_V}
- [S(w)]_{B_X} = B[w]_{B_W}

Therefore:
[S(T(v))]_{B_X} = B[T(v)]_{B_W} = B(A[v]_{B_V}) = (BA)[v]_{B_V} ∎

---

## 🌙 Evening Session: Computational Lab (1 hour)

```python
"""
Day 93: Matrix Representation of Linear Maps
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# Part 1: Constructing Matrices from Linear Maps
# =============================================================

def matrix_of_transformation(T, input_basis, output_basis):
    """
    Construct the matrix of a linear transformation T.
    
    Parameters:
    -----------
    T : function that applies the linear map
    input_basis : list of input basis vectors
    output_basis : list of output basis vectors
    
    Returns:
    --------
    matrix : the matrix representation
    """
    n = len(input_basis)  # input dimension
    m = len(output_basis)  # output dimension
    
    # Build output basis matrix for solving coordinate systems
    B_out = np.column_stack(output_basis)
    
    # Matrix columns are T(v_j) expressed in output basis
    M = np.zeros((m, n), dtype=complex)
    
    for j, v_j in enumerate(input_basis):
        T_v_j = T(v_j)
        # Solve: B_out @ coords = T_v_j
        coords = np.linalg.solve(B_out, T_v_j)
        M[:, j] = coords
    
    return M


print("="*60)
print("Constructing Matrices of Linear Transformations")
print("="*60)

# Example 1: Rotation by 30°
theta = np.pi / 6
R = lambda v: np.array([v[0]*np.cos(theta) - v[1]*np.sin(theta),
                        v[0]*np.sin(theta) + v[1]*np.cos(theta)])

std_basis_2d = [np.array([1, 0]), np.array([0, 1])]

M_rotation = matrix_of_transformation(R, std_basis_2d, std_basis_2d)
print(f"\nRotation by 30°:")
print(f"Matrix:\n{M_rotation.real.round(4)}")
print(f"Expected: [[cos(30°), -sin(30°)], [sin(30°), cos(30°)]]")
print(f"         [[{np.cos(theta):.4f}, {-np.sin(theta):.4f}],")
print(f"          [{np.sin(theta):.4f}, {np.cos(theta):.4f}]]")

# Example 2: Derivative on polynomials
# Represent as coefficient vectors: p(x) = a + bx + cx² → [a, b, c]
D = lambda p: np.array([p[1], 2*p[2], 0])  # derivative

poly_basis_3 = [np.array([1, 0, 0]),   # 1
                np.array([0, 1, 0]),   # x
                np.array([0, 0, 1])]   # x²

M_deriv = matrix_of_transformation(D, poly_basis_3, poly_basis_3)
print(f"\nDerivative D: P₂ → P₂:")
print(f"Matrix:\n{M_deriv.real.astype(int)}")

# Test: D(2 + 3x + x²) = 3 + 2x
p = np.array([2, 3, 1])
Dp = M_deriv.real @ p
print(f"D(2 + 3x + x²) = {Dp[0]:.0f} + {Dp[1]:.0f}x + {Dp[2]:.0f}x²")

# =============================================================
# Part 2: Change of Basis
# =============================================================

def change_of_basis_matrix(old_basis, new_basis):
    """
    Compute change of basis matrix P such that [v]_new = P @ [v]_old.
    """
    # P's columns are the old basis vectors expressed in new basis
    B_new = np.column_stack(new_basis)
    n = len(old_basis)
    P = np.zeros((n, n), dtype=complex)
    
    for j, v_old in enumerate(old_basis):
        P[:, j] = np.linalg.solve(B_new, v_old)
    
    return P


print("\n" + "="*60)
print("Change of Basis")
print("="*60)

# Standard basis
std = [np.array([1, 0]), np.array([0, 1])]

# New basis: 45° rotated
new_basis = [np.array([1, 1])/np.sqrt(2), np.array([-1, 1])/np.sqrt(2)]

P = change_of_basis_matrix(std, new_basis)
print(f"\nChange of basis matrix (std → rotated 45°):")
print(f"P = \n{P.real.round(4)}")

# Verify: (1, 0) in rotated basis should be (1/√2, -1/√2)
v = np.array([1, 0])
v_new = P @ v
print(f"\n(1, 0) in rotated basis: {v_new.real.round(4)}")
print(f"Expected: [{1/np.sqrt(2):.4f}, {-1/np.sqrt(2):.4f}]")

# Transform a matrix under change of basis
print("\n--- Matrix Similarity Transform ---")

# Diagonal matrix in standard basis
M_std = np.array([[2, 0], [0, 1]])
print(f"M in standard basis:\n{M_std}")

# Same transformation in rotated basis
M_new = P @ M_std @ np.linalg.inv(P)
print(f"\nM in rotated basis:\n{M_new.real.round(4)}")

# =============================================================
# Part 3: Quantum Operators
# =============================================================

print("\n" + "="*60)
print("Quantum Operators as Matrices")
print("="*60)

# Computational basis
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Hadamard basis
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# Pauli Z in computational basis
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
print(f"\nσz in computational basis:")
print(sigma_z)

# Pauli Z in Hadamard basis
comp_basis = [ket_0, ket_1]
had_basis = [ket_plus, ket_minus]

P_comp_to_had = change_of_basis_matrix(comp_basis, had_basis)
sigma_z_had = P_comp_to_had @ sigma_z @ np.linalg.inv(P_comp_to_had)
print(f"\nσz in Hadamard basis:")
print(sigma_z_had.real.round(4))
print("(This is σx!)")

# Verify: σz|+⟩ = |-⟩ in computational basis
result = sigma_z @ ket_plus
print(f"\nσz|+⟩ = {result} = |-⟩? {np.allclose(result, ket_minus)}")

# Hadamard gate construction
print("\n--- Constructing Hadamard Gate ---")

# H is defined by H|0⟩ = |+⟩, H|1⟩ = |-⟩
def Hadamard(v):
    # Express H in terms of its action
    return np.array([v[0]/np.sqrt(2) + v[1]/np.sqrt(2),
                     v[0]/np.sqrt(2) - v[1]/np.sqrt(2)], dtype=complex)

H = matrix_of_transformation(Hadamard, comp_basis, comp_basis)
print(f"Hadamard gate matrix:")
print(H.real.round(4))

# Verify: H² = I
print(f"\nH² = I? {np.allclose(H @ H, np.eye(2))}")

# =============================================================
# Part 4: Visualization
# =============================================================

def visualize_matrix_action(A, title, ax):
    """Visualize how a 2x2 matrix transforms the unit circle and basis vectors."""
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform
    transformed = A @ circle
    
    # Plot
    ax.plot(circle[0], circle[1], 'b--', alpha=0.5, label='Original')
    ax.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
    
    # Basis vectors
    e1, e2 = np.array([1, 0]), np.array([0, 1])
    Te1, Te2 = A @ e1, A @ e2
    
    ax.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', alpha=0.5, width=0.02)
    ax.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', alpha=0.5, width=0.02)
    ax.quiver(0, 0, Te1[0], Te1[1], angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.02)
    ax.quiver(0, 0, Te2[0], Te2[1], angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.02)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Various matrices
matrices = [
    (np.array([[2, 0], [0, 1]]), 'Scaling (2,1)'),
    (np.array([[1, 1], [0, 1]]), 'Shear'),
    (np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], 
               [np.sin(np.pi/4), np.cos(np.pi/4)]]), 'Rotation 45°'),
    (np.array([[0, 1], [1, 0]]), 'Reflection y=x'),
    (np.array([[1, 0], [0, 0]]), 'Projection x-axis'),
    (np.array([[2, 1], [1, 2]]), 'General (eigenvalues 1, 3)')
]

for ax, (A, title) in zip(axes.flat, matrices):
    visualize_matrix_action(A, title, ax)

plt.tight_layout()
plt.savefig('day93_matrices.png', dpi=150)
plt.show()

# =============================================================
# Part 5: Matrix-Vector Multiplication Interpretations
# =============================================================

print("\n" + "="*60)
print("Matrix-Vector Multiplication: Two Views")
print("="*60)

A = np.array([[1, 2, 3],
              [4, 5, 6]])
x = np.array([1, -1, 2])

print(f"A = \n{A}")
print(f"x = {x}")
print(f"\nAx = {A @ x}")

print("\n--- Row Picture ---")
for i in range(A.shape[0]):
    print(f"Row {i+1}: {A[i]} · {x} = {np.dot(A[i], x)}")

print("\n--- Column Picture ---")
col_combo = " + ".join([f"{x[j]}*col{j+1}" for j in range(len(x))])
print(f"Ax = {col_combo}")
result = sum(x[j] * A[:, j] for j in range(len(x)))
print(f"   = {result}")
```

---

## 📝 Homework

### Written Problems

1. Find the matrix of T: ℝ⁴ → ℝ³ defined by T(x₁, x₂, x₃, x₄) = (x₁ - x₂, x₂ - x₃, x₃ - x₄).

2. Let T: M₂ₓ₂ → M₂ₓ₂ be defined by T(A) = Aᵀ (transpose). Find the matrix of T with respect to the basis {E₁₁, E₁₂, E₂₁, E₂₂}.

3. Let T: ℝ² → ℝ² be rotation by 90°.
   a) Find [T] in standard basis.
   b) Find [T] in basis {(1,1), (1,-1)}.
   c) Are these matrices similar?

4. Prove: If A and B are similar matrices (B = P⁻¹AP for some invertible P), then they have the same determinant.

5. In quantum mechanics, the Pauli Y gate is:
   $$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$
   
   Show that Y = iσxσz and verify Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩.

---

## ✅ Daily Checklist

- [ ] Read Axler Chapter 3.C
- [ ] Understand matrix construction procedure
- [ ] Can work with change of basis
- [ ] Completed Problem Sets A, B, C
- [ ] Ran computational lab
- [ ] Understand quantum operator matrices
- [ ] Started homework

---

## 🔮 Preview: Tomorrow

**Day 94: Matrix Operations and Composition**
- Matrix addition and scalar multiplication
- Matrix multiplication
- Composition corresponds to multiplication
- Powers of matrices
- The inverse of a matrix

---

*"God used beautiful mathematics in creating the world."*
— Paul Dirac
