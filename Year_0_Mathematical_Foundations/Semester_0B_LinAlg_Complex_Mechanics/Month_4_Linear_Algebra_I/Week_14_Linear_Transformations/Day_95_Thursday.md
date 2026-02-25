# Day 95: Kernel and Range

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Kernel and Range |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define and compute the kernel (null space) of a linear map
2. Define and compute the range (image) of a linear map
3. Prove both are subspaces
4. Find bases for kernel and range
5. Understand the connection to solving linear systems
6. Connect to quantum measurement (projection operators)

---

## 📚 Required Reading

**Before starting:**
- Axler, Chapter 3.B: "Null Spaces and Ranges" (pp. 74-82)
- Strang, Chapter 3.1-3.2: Column Space and Nullspace

---

## 🌅 Morning Session: Theory (3.5 hours)

### Part 1: The Kernel (Null Space) (60 min)

#### Definition

Let T: V → W be a linear transformation.

The **kernel** (or **null space**) of T is:
$$\ker(T) = \{v \in V : T(v) = 0_W\}$$

In matrix language, if A is the matrix of T:
$$\text{null}(A) = \{x \in \mathbb{R}^n : Ax = 0\}$$

#### Theorem: The Kernel is a Subspace

**Proof:**
1. **Contains zero:** T(0_V) = 0_W, so 0_V ∈ ker(T). ✓

2. **Closed under addition:** Let u, v ∈ ker(T).
   T(u + v) = T(u) + T(v) = 0 + 0 = 0
   So u + v ∈ ker(T). ✓

3. **Closed under scalar multiplication:** Let v ∈ ker(T), c ∈ F.
   T(cv) = cT(v) = c · 0 = 0
   So cv ∈ ker(T). ✓

Therefore ker(T) is a subspace of V. ∎

#### Nullity

The **nullity** of T is the dimension of the kernel:
$$\text{nullity}(T) = \dim(\ker(T))$$

#### Finding the Kernel

For a matrix A, solve Ax = 0:
1. Row reduce A to echelon form
2. Identify free variables
3. Express basic variables in terms of free variables
4. Write the general solution as linear combination of basis vectors

#### Example 1: Kernel of a 3×4 Matrix

$$A = \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 1 & 2 & 1 & 3 \end{pmatrix}$$

Row reduce:
$$\to \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Pivot columns: 1, 3. Free variables: x₂, x₄.

From row 1: x₁ = -2x₂ - x₄
From row 2: x₃ = -2x₄

General solution:
$$x = x_2\begin{pmatrix} -2 \\ 1 \\ 0 \\ 0 \end{pmatrix} + x_4\begin{pmatrix} -1 \\ 0 \\ -2 \\ 1 \end{pmatrix}$$

Basis for ker(A): {(-2, 1, 0, 0), (-1, 0, -2, 1)}
nullity(A) = 2

### Part 2: The Range (Image) (60 min)

#### Definition

The **range** (or **image**) of T: V → W is:
$$\text{range}(T) = \{T(v) : v \in V\} = \{w \in W : \exists v \in V, T(v) = w\}$$

In matrix language:
$$\text{col}(A) = \{Ax : x \in \mathbb{R}^n\} = \text{span of columns of } A$$

This is why it's also called the **column space**.

#### Theorem: The Range is a Subspace

**Proof:**
1. **Contains zero:** T(0_V) = 0_W ∈ range(T). ✓

2. **Closed under addition:** Let w₁, w₂ ∈ range(T).
   Then w₁ = T(v₁), w₂ = T(v₂) for some v₁, v₂ ∈ V.
   w₁ + w₂ = T(v₁) + T(v₂) = T(v₁ + v₂) ∈ range(T). ✓

3. **Closed under scalar multiplication:** Let w ∈ range(T), c ∈ F.
   w = T(v) for some v ∈ V.
   cw = cT(v) = T(cv) ∈ range(T). ✓

Therefore range(T) is a subspace of W. ∎

#### Rank

The **rank** of T is the dimension of the range:
$$\text{rank}(T) = \dim(\text{range}(T))$$

For matrices: rank(A) = number of pivot columns = number of linearly independent columns.

#### Finding the Range

1. The range is spanned by the columns of A
2. To find a basis: row reduce and keep original columns corresponding to pivots
3. Or: the column space of A = column space of RREF(A)

#### Example 2: Range of the Same Matrix

$$A = \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 1 & 2 & 1 & 3 \end{pmatrix}$$

Pivot columns are 1 and 3.

Basis for range(A) = {column 1, column 3} of original A:
$$\left\{\begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}, \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}\right\}$$

rank(A) = 2

### Part 3: Injectivity and Surjectivity (45 min)

#### Injective (One-to-One)

T is **injective** if T(u) = T(v) implies u = v.

**Theorem:** T is injective ⟺ ker(T) = {0}

**Proof:**
(⇒) Suppose T is injective. If v ∈ ker(T), then T(v) = 0 = T(0).
By injectivity, v = 0. So ker(T) = {0}.

(⇐) Suppose ker(T) = {0}. If T(u) = T(v), then T(u-v) = T(u) - T(v) = 0.
So u - v ∈ ker(T) = {0}, meaning u - v = 0, i.e., u = v.
Therefore T is injective. ∎

**Corollary:** T is injective ⟺ nullity(T) = 0

#### Surjective (Onto)

T: V → W is **surjective** if range(T) = W.

**Corollary:** T is surjective ⟺ rank(T) = dim(W)

#### Bijective (Invertible)

T is **bijective** if it's both injective and surjective.
This requires:
- ker(T) = {0}
- range(T) = W
- When V = W (square matrix case): dim(V) = dim(W)

### Part 4: Connection to Linear Systems (30 min)

#### Solving Ax = b

The system Ax = b has a solution ⟺ b ∈ range(A) = col(A)

**Characterization of solutions:**
- If x₀ is a particular solution (Ax₀ = b), then:
- All solutions are: {x₀ + n : n ∈ ker(A)}

**Geometric picture:**
- range(A) tells us which right-hand sides are achievable
- ker(A) tells us the "freedom" in solutions

#### Summary Table

| Property | Linear algebra term | Condition |
|----------|---------------------|-----------|
| Ax = b has solution | b ∈ range(A) | b in column space |
| Solution is unique | ker(A) = {0} | A is injective |
| Solution exists ∀b | range(A) = ℝᵐ | A is surjective |
| Unique solution ∀b | A invertible | A is bijective |

### Part 5: Quantum Connection — Projections (30 min)

#### Projection Operators

In quantum mechanics, measurements are described by projection operators.

A linear operator P: V → V is a **projection** if P² = P.

#### Key Properties of Projections

1. **Eigenvalues:** If P² = P, eigenvalues satisfy λ² = λ, so λ ∈ {0, 1}
2. **Decomposition:** V = ker(P) ⊕ range(P)
3. **Complement:** I - P is also a projection onto ker(P)

#### Quantum Measurement

For a quantum state |ψ⟩ and projection P = |φ⟩⟨φ|:
- **Probability of outcome:** p = ⟨ψ|P|ψ⟩ = |⟨φ|ψ⟩|²
- **Post-measurement state:** P|ψ⟩/||P|ψ⟩|| (if measured)

#### Example: Spin Measurement

For spin-1/2:
$$P_{up} = |0\rangle\langle 0| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

$$P_{down} = |1\rangle\langle 1| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

For state |ψ⟩ = α|0⟩ + β|1⟩:
- P(spin up) = |α|²
- P(spin down) = |β|²
- P_{up}|ψ⟩ = α|0⟩ (unnormalized post-measurement state)

---

## 🌆 Afternoon Session: Problem Solving (2.5 hours)

### Problem Set A: Kernel Computation (50 min)

**Problem 1.** Find the kernel and nullity of:
$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$

**Solution:**
Row reduce:
$$\to \begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & -6 & -12 \end{pmatrix} \to \begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & 2 \\ 0 & 0 & 0 \end{pmatrix} \to \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & 2 \\ 0 & 0 & 0 \end{pmatrix}$$

Free variable: x₃
x₁ = x₃, x₂ = -2x₃

ker(A) = span{(1, -2, 1)}
nullity(A) = 1

**Problem 2.** Find the kernel of T: P₂ → P₁ defined by T(p) = p'(1) + p'(0).

**Solution:**
Let p(x) = a + bx + cx².
p'(x) = b + 2cx
p'(1) = b + 2c
p'(0) = b
T(p) = (b + 2c) + b = 2b + 2c

ker(T) = {a + bx + cx² : 2b + 2c = 0} = {a + bx + cx² : c = -b}
       = {a + bx - bx² : a, b ∈ ℝ}
       = span{1, x - x²}

nullity(T) = 2

**Problem 3.** Let T: ℝ³ → ℝ³ be defined by T(x, y, z) = (x - y, y - z, z - x).
Find ker(T).

**Solution:**
T(x, y, z) = (0, 0, 0) requires:
x - y = 0 → x = y
y - z = 0 → y = z
z - x = 0 → z = x

So x = y = z.
ker(T) = {(t, t, t) : t ∈ ℝ} = span{(1, 1, 1)}
nullity(T) = 1

### Problem Set B: Range Computation (50 min)

**Problem 4.** Find the range and rank of the matrix from Problem 1.

**Solution:**
From the RREF, pivot columns are 1 and 2.
Basis for range(A) = {original columns 1, 2} = {(1, 4, 7), (2, 5, 8)}
rank(A) = 2

Note: nullity(A) + rank(A) = 1 + 2 = 3 = number of columns.

**Problem 5.** Find the range of T from Problem 3.

**Solution:**
For T(x, y, z) = (x-y, y-z, z-x), note that:
(x-y) + (y-z) + (z-x) = 0

So range(T) ⊆ {(a, b, c) : a + b + c = 0}.

To show equality, take any (a, b, c) with a + b + c = 0.
We need to find (x, y, z) such that:
x - y = a
y - z = b
z - x = c = -(a+b)  (this is automatic!)

Choose z = 0, then y = b, x = a + b.
Check: T(a+b, b, 0) = ((a+b)-b, b-0, 0-(a+b)) = (a, b, -(a+b)) = (a, b, c). ✓

range(T) = {(a, b, c) : a + b + c = 0}
rank(T) = 2

**Problem 6.** Let A be an m×n matrix.
a) What's the maximum possible rank(A)?
b) When is A surjective?
c) When is A injective?

**Solution:**
a) rank(A) ≤ min(m, n) (limited by both dimensions)

b) A is surjective when range(A) = ℝᵐ, i.e., rank(A) = m.
   This requires n ≥ m.

c) A is injective when ker(A) = {0}, i.e., nullity(A) = 0.
   Since rank + nullity = n, this means rank(A) = n.
   This requires m ≥ n.

### Problem Set C: Proofs (50 min)

**Problem 7.** Prove that if T: V → W is linear and W is finite-dimensional, then range(T) is finite-dimensional with dim(range(T)) ≤ dim(W).

**Proof:**
range(T) is a subspace of W (proved earlier).
For any subspace U of a finite-dimensional space W:
dim(U) ≤ dim(W)

Therefore dim(range(T)) ≤ dim(W). ∎

**Problem 8.** Prove: T is injective ⟺ T maps linearly independent sets to linearly independent sets.

**Proof:**
(⇒) Suppose T is injective and {v₁, ..., vₖ} is linearly independent.
If c₁T(v₁) + ... + cₖT(vₖ) = 0, then T(c₁v₁ + ... + cₖvₖ) = 0.
So c₁v₁ + ... + cₖvₖ ∈ ker(T) = {0}.
Thus c₁v₁ + ... + cₖvₖ = 0, which implies all cᵢ = 0 by independence.
Therefore {T(v₁), ..., T(vₖ)} is independent.

(⇐) Suppose T maps independent sets to independent sets.
Let v ∈ ker(T) with v ≠ 0. Then {v} is independent but T({v}) = {0} is dependent.
Contradiction! So ker(T) = {0}, meaning T is injective. ∎

**Problem 9.** Let P: V → V be a projection (P² = P). Prove:
a) V = ker(P) + range(P)
b) ker(P) ∩ range(P) = {0}

**Proof:**
a) For any v ∈ V: v = (v - Pv) + Pv.
   - Pv ∈ range(P) clearly.
   - P(v - Pv) = Pv - P²v = Pv - Pv = 0, so v - Pv ∈ ker(P).
   Therefore V = ker(P) + range(P).

b) Suppose w ∈ ker(P) ∩ range(P).
   - w ∈ ker(P) means Pw = 0.
   - w ∈ range(P) means w = Pv for some v.
   Then w = Pv, so Pw = P²v = Pv = w.
   But Pw = 0, so w = 0.
   Therefore ker(P) ∩ range(P) = {0}. ∎

---

## 🌙 Evening Session: Computational Lab (1 hour)

```python
"""
Day 95: Kernel and Range
"""

import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================
# Part 1: Computing Kernel (Null Space)
# =============================================================

print("="*60)
print("Computing Kernel (Null Space)")
print("="*60)

def find_kernel_basis(A, tol=1e-10):
    """Find basis for kernel of A using SVD."""
    ns = null_space(A)
    return ns

def verify_kernel(A, kernel_basis):
    """Verify that vectors are in kernel."""
    if kernel_basis.size == 0:
        print("Kernel is trivial (only contains zero vector)")
        return True
    for i in range(kernel_basis.shape[1]):
        v = kernel_basis[:, i]
        Av = A @ v
        print(f"  A @ v{i+1} = {Av.round(10)}")
    return np.allclose(A @ kernel_basis, 0, atol=tol)


# Example 1: 3x4 matrix
A1 = np.array([[1, 2, 0, 1],
               [0, 0, 1, 2],
               [1, 2, 1, 3]])

print(f"\nA1 = \n{A1}")
ker_A1 = find_kernel_basis(A1)
print(f"\nKernel basis (columns):\n{ker_A1.round(4)}")
print(f"Nullity = {ker_A1.shape[1]}")
print(f"Verification:")
verify_kernel(A1, ker_A1)

# Example 2: Singular 3x3 matrix  
A2 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

print(f"\n\nA2 = \n{A2}")
ker_A2 = find_kernel_basis(A2)
print(f"\nKernel basis:\n{ker_A2.round(4)}")
print(f"Nullity = {ker_A2.shape[1]}")
print(f"Verification:")
verify_kernel(A2, ker_A2)

# Example 3: Invertible matrix (trivial kernel)
A3 = np.array([[1, 2], [3, 4]])
print(f"\n\nA3 = \n{A3}")
ker_A3 = find_kernel_basis(A3)
print(f"Kernel dimension: {ker_A3.shape[1] if ker_A3.size > 0 else 0}")
print(f"A3 is injective (trivial kernel)")

# =============================================================
# Part 2: Computing Range (Column Space)
# =============================================================

print("\n" + "="*60)
print("Computing Range (Column Space)")
print("="*60)

def find_range_basis(A, tol=1e-10):
    """Find basis for range of A using SVD."""
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(S > tol)
    return U[:, :rank], rank


# For A1
print(f"\nFor A1:")
range_A1, rank_A1 = find_range_basis(A1)
print(f"Range basis (columns):\n{range_A1.round(4)}")
print(f"Rank = {rank_A1}")

# Verify rank-nullity theorem
nullity_A1 = ker_A1.shape[1]
print(f"\nRank-nullity check: rank + nullity = {rank_A1} + {nullity_A1} = {rank_A1 + nullity_A1}")
print(f"Number of columns of A1 = {A1.shape[1]}")
print(f"Theorem verified: {rank_A1 + nullity_A1 == A1.shape[1]}")

# For A2
print(f"\n\nFor A2:")
range_A2, rank_A2 = find_range_basis(A2)
print(f"Range basis (columns):\n{range_A2.round(4)}")
print(f"Rank = {rank_A2}")

nullity_A2 = ker_A2.shape[1]
print(f"Rank + nullity = {rank_A2} + {nullity_A2} = {rank_A2 + nullity_A2} = {A2.shape[1]} ✓")

# =============================================================
# Part 3: Solving Ax = b
# =============================================================

print("\n" + "="*60)
print("Solving Linear Systems")
print("="*60)

def analyze_system(A, b):
    """Analyze the linear system Ax = b."""
    print(f"\nSystem: Ax = b where A is {A.shape[0]}×{A.shape[1]}")
    print(f"b = {b}")
    
    # Check if b is in range(A)
    range_basis, rank = find_range_basis(A)
    
    # Project b onto range(A)
    b_in_range = range_basis @ (range_basis.T @ b)
    
    if np.allclose(b, b_in_range, atol=1e-10):
        print("✓ b is in range(A) - solution exists")
        
        # Find a particular solution
        x_particular, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        print(f"Particular solution x₀: {x_particular.round(4)}")
        print(f"Verification: Ax₀ = {(A @ x_particular).round(4)}")
        
        # General solution = x₀ + ker(A)
        ker = find_kernel_basis(A)
        if ker.size > 0:
            print(f"\nGeneral solution: x = x₀ + span of kernel")
            print(f"Kernel basis:\n{ker.round(4)}")
        else:
            print("Unique solution (trivial kernel)")
    else:
        print("✗ b is not in range(A) - no solution exists")
        print(f"Closest point in range: {b_in_range.round(4)}")


# System with unique solution
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 11])
analyze_system(A, b)

# System with infinitely many solutions
A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([6, 15])
analyze_system(A, b)

# System with no solution
A = np.array([[1, 2], [2, 4]])
b = np.array([3, 5])
analyze_system(A, b)

# =============================================================
# Part 4: Projection Operators (QM Application)
# =============================================================

print("\n" + "="*60)
print("Projection Operators (Quantum Mechanics)")
print("="*60)

def is_projection(P, tol=1e-10):
    """Check if P is a projection (P² = P)."""
    return np.allclose(P @ P, P, atol=tol)

def analyze_projection(P):
    """Analyze a projection operator."""
    print(f"\nP = \n{P.round(4)}")
    print(f"Is projection (P² = P): {is_projection(P)}")
    
    # Find eigenvalues
    eigvals = np.linalg.eigvals(P)
    print(f"Eigenvalues: {eigvals.round(4)}")
    
    # Find range and kernel
    range_basis, rank = find_range_basis(P)
    ker_basis = find_kernel_basis(P)
    
    print(f"Rank (dimension of range): {rank}")
    print(f"Nullity (dimension of kernel): {ker_basis.shape[1] if ker_basis.size else 0}")


# Projection onto x-axis in R²
P_x = np.array([[1, 0], [0, 0]])
print("\n--- Projection onto x-axis ---")
analyze_projection(P_x)

# Projection onto line y = x
P_diag = np.array([[0.5, 0.5], [0.5, 0.5]])
print("\n--- Projection onto y = x ---")
analyze_projection(P_diag)

# Quantum: projection onto |0⟩
ket_0 = np.array([1, 0], dtype=complex)
P_0 = np.outer(ket_0, np.conj(ket_0))
print("\n--- Quantum: Projection onto |0⟩ ---")
analyze_projection(P_0.real)

# Quantum measurement simulation
print("\n--- Quantum Measurement Simulation ---")
psi = np.array([0.6, 0.8], dtype=complex)  # |ψ⟩ = 0.6|0⟩ + 0.8|1⟩
print(f"|ψ⟩ = {psi}")

P_up = np.array([[1, 0], [0, 0]], dtype=complex)   # |0⟩⟨0|
P_down = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|

prob_up = np.abs(np.vdot(psi, P_up @ psi))
prob_down = np.abs(np.vdot(psi, P_down @ psi))

print(f"P(measure |0⟩) = |⟨0|ψ⟩|² = {prob_up:.4f}")
print(f"P(measure |1⟩) = |⟨1|ψ⟩|² = {prob_down:.4f}")
print(f"Total probability = {prob_up + prob_down:.4f}")

# Post-measurement state
post_up = P_up @ psi
post_up_normalized = post_up / np.linalg.norm(post_up)
print(f"\nIf we measure |0⟩, post-measurement state: {post_up_normalized}")

# =============================================================
# Part 5: Visualization
# =============================================================

def visualize_kernel_range():
    """Visualize kernel and range in 3D."""
    fig = plt.figure(figsize=(15, 5))
    
    # Example: T(x,y,z) = (x-y, y-z, z-x)
    # Kernel: span{(1,1,1)}
    # Range: plane a+b+c = 0
    
    # 1. Kernel visualization
    ax1 = fig.add_subplot(131, projection='3d')
    t = np.linspace(-2, 2, 100)
    ax1.plot(t, t, t, 'b-', linewidth=3, label='ker(T) = span{(1,1,1)}')
    ax1.scatter([0], [0], [0], color='red', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Kernel (Null Space)')
    ax1.legend()
    
    # 2. Range visualization
    ax2 = fig.add_subplot(132, projection='3d')
    xx, yy = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    zz = -xx - yy  # Plane a + b + c = 0
    ax2.plot_surface(xx, yy, zz, alpha=0.5, color='green')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Range: plane x+y+z=0')
    
    # 3. Input space decomposition
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Show a general vector decomposed into ker and range components
    v = np.array([2, 1, 0])  # Example vector
    
    # Project onto kernel direction
    ker_dir = np.array([1, 1, 1]) / np.sqrt(3)
    v_ker = np.dot(v, ker_dir) * ker_dir
    v_range = v - v_ker
    
    ax3.quiver(0, 0, 0, v[0], v[1], v[2], color='black', arrow_length_ratio=0.1, label='v')
    ax3.quiver(0, 0, 0, v_ker[0], v_ker[1], v_ker[2], color='blue', arrow_length_ratio=0.1, label='v_ker')
    ax3.quiver(v_ker[0], v_ker[1], v_ker[2], v_range[0], v_range[1], v_range[2], 
               color='red', arrow_length_ratio=0.1, label='v_range')
    
    ax3.set_xlim(-2, 3)
    ax3.set_ylim(-2, 3)
    ax3.set_zlim(-2, 3)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Decomposition v = v_ker + v_range')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('day95_kernel_range.png', dpi=150)
    plt.show()


visualize_kernel_range()
```

---

## 📝 Homework

### Written Problems

1. Find the kernel and range of T: ℝ⁴ → ℝ³ defined by T(x₁, x₂, x₃, x₄) = (x₁ + x₂, x₂ + x₃, x₃ + x₄).

2. Prove: If T: V → V satisfies T² = 0 (nilpotent of index 2), then range(T) ⊆ ker(T).

3. Let P be a projection. Prove that I - P is also a projection and find its range and kernel.

4. Show that for any linear T: V → W, ker(T) = {0} if and only if dim(range(T)) = dim(V).

5. In quantum mechanics, for a two-qubit system with basis {|00⟩, |01⟩, |10⟩, |11⟩}, let P be the projection onto the subspace span{|00⟩, |11⟩}.
   a) Write P as a 4×4 matrix.
   b) Find ker(P) and range(P).
   c) What happens when we apply P to the state |ψ⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2?

---

## ✅ Daily Checklist

- [ ] Can define kernel and range
- [ ] Can compute kernel from row reduction
- [ ] Can find basis for range
- [ ] Understand injectivity ↔ trivial kernel
- [ ] Understand surjectivity ↔ full range
- [ ] Completed all problem sets
- [ ] Ran computational lab
- [ ] Understand quantum projection operators

---

## 🔮 Preview: Tomorrow

**Day 96: Rank-Nullity Theorem**
- The fundamental theorem: dim(V) = rank(T) + nullity(T)
- Applications to linear systems
- Dimension of solution spaces

---

*"A mathematician is a blind man in a dark room looking for a black cat which isn't there."*
— Charles Darwin (possibly apocryphal)
