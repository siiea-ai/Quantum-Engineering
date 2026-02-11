# Day 96: The Rank-Nullity Theorem

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Rank-Nullity |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State and prove the rank-nullity theorem
2. Apply it to determine dimensions of kernel and range
3. Use it to analyze solvability of linear systems
4. Understand its geometric meaning
5. Connect to quantum state space dimensions

---

## 📚 Required Reading

**Before starting:**
- Axler, Chapter 3.B: Theorem on null space and range (pp. 77-80)
- Strang, Chapter 3.3: Dimension Formula

---

## 🌅 Morning Session: Theory (3.5 hours)

### Part 1: Statement of the Theorem (30 min)

#### The Fundamental Theorem of Linear Maps

**Theorem (Rank-Nullity):** Let T: V → W be a linear map, where V is finite-dimensional. Then:

$$\dim(V) = \dim(\ker(T)) + \dim(\text{range}(T))$$

Or equivalently:
$$\dim(V) = \text{nullity}(T) + \text{rank}(T)$$

#### Why This is Profound

This theorem says: **"Everything in V either dies (goes to 0) or survives (reaches range)."**

No information is lost without a trace—the dimensions account for everything.

#### Matrix Form

For an m×n matrix A:
$$n = \text{nullity}(A) + \text{rank}(A)$$

where n = number of columns.

### Part 2: The Proof (60 min)

#### Setup

Let T: V → W with dim(V) = n.
Let nullity(T) = k, so dim(ker(T)) = k.

Let {u₁, ..., uₖ} be a basis for ker(T).

Extend this to a basis {u₁, ..., uₖ, v₁, ..., vₘ} for V.

So n = k + m, and we want to show rank(T) = m.

#### Claim: {T(v₁), ..., T(vₘ)} is a basis for range(T)

**Part 1: These vectors span range(T)**

Let w ∈ range(T). Then w = T(v) for some v ∈ V.

Write v = a₁u₁ + ... + aₖuₖ + b₁v₁ + ... + bₘvₘ

Then:
$$T(v) = T(a_1u_1 + \cdots + a_ku_k + b_1v_1 + \cdots + b_mv_m)$$
$$= a_1T(u_1) + \cdots + a_kT(u_k) + b_1T(v_1) + \cdots + b_mT(v_m)$$
$$= a_1 \cdot 0 + \cdots + a_k \cdot 0 + b_1T(v_1) + \cdots + b_mT(v_m)$$
$$= b_1T(v_1) + \cdots + b_mT(v_m)$$

So w is in span{T(v₁), ..., T(vₘ)}. ✓

**Part 2: These vectors are linearly independent**

Suppose c₁T(v₁) + ... + cₘT(vₘ) = 0.

By linearity: T(c₁v₁ + ... + cₘvₘ) = 0.

So c₁v₁ + ... + cₘvₘ ∈ ker(T).

Since {u₁, ..., uₖ} is a basis for ker(T):
$$c_1v_1 + \cdots + c_mv_m = d_1u_1 + \cdots + d_ku_k$$

for some scalars d₁, ..., dₖ.

Rearranging:
$$c_1v_1 + \cdots + c_mv_m - d_1u_1 - \cdots - d_ku_k = 0$$

But {u₁, ..., uₖ, v₁, ..., vₘ} is a basis for V (linearly independent).

So all coefficients must be zero: c₁ = ... = cₘ = d₁ = ... = dₖ = 0.

Therefore {T(v₁), ..., T(vₘ)} is linearly independent. ✓

#### Conclusion

{T(v₁), ..., T(vₘ)} is a basis for range(T).

Therefore rank(T) = dim(range(T)) = m = n - k = dim(V) - nullity(T).

So: **dim(V) = nullity(T) + rank(T)**. ∎

### Part 3: Applications (60 min)

#### Application 1: Injectivity and Dimension

**Corollary:** T: V → W is injective ⟺ rank(T) = dim(V)

**Proof:**
T injective ⟺ ker(T) = {0} ⟺ nullity(T) = 0 ⟺ rank(T) = dim(V) - 0 = dim(V). ∎

#### Application 2: Surjectivity and Dimension

**Corollary:** T: V → W is surjective ⟺ rank(T) = dim(W)

#### Application 3: Isomorphism Criterion

**Corollary:** If dim(V) = dim(W), then T: V → W is injective ⟺ T is surjective ⟺ T is bijective.

**Proof:**
If dim(V) = dim(W) = n:
- T injective ⟺ rank(T) = n ⟺ range(T) = W ⟺ T surjective ∎

#### Application 4: Number of Solutions to Ax = b

For Ax = b with A being m×n:
- If Ax = b has a solution x₀, all solutions are: {x₀ + n : n ∈ ker(A)}
- The "dimension of the solution space" is nullity(A)
- Number of free variables = n - rank(A) = nullity(A)

#### Application 5: Counting Free Variables

**Theorem:** For an m×n matrix A:
- Number of pivot variables = rank(A)
- Number of free variables = n - rank(A) = nullity(A)

### Part 4: Geometric Interpretation (30 min)

#### Dimension as "Degrees of Freedom"

Think of dim(V) as the total degrees of freedom in V.

When we apply T:
- nullity(T) degrees of freedom are "collapsed" (sent to 0)
- rank(T) degrees of freedom "survive" (span the range)

**Example:** 
T: ℝ³ → ℝ² squashes 3D space onto a 2D plane.
If rank(T) = 2, then nullity(T) = 1.
One dimension (a line) is collapsed to a point (the origin).

#### Visual Picture

```
V (3D)                    W (2D)
   ↑                        ↑
   |                        |
 [dim=3]      T          [range has dim=2]
   |    ----------→          |
   |                        |
 kernel                     ↓
[dim=1]                (range fills W)
(collapses
 to 0)
```

### Part 5: Quantum Connection (30 min)

#### Dimension Conservation in QM

The rank-nullity theorem underlies many quantum facts:

1. **Unitary operators:** U: ℋ → ℋ unitary ⟹ rank(U) = dim(ℋ), nullity = 0
   - Unitary maps are bijections (no information loss)
   - This is the mathematical content of "quantum evolution is reversible"

2. **Projections:** P² = P with P: ℋ → ℋ
   - ℋ = ker(P) ⊕ range(P)
   - dim(ℋ) = dim(ker(P)) + rank(P)
   - This is the QM measurement decomposition!

3. **Decoherence:** When a quantum system loses coherence, effective dimension of accessible state space decreases (rank decreases, nullity increases).

#### Example: Spin-1/2 System

V = ℂ² (2-dimensional)

Projection onto |0⟩: P = |0⟩⟨0| = [[1,0],[0,0]]
- rank(P) = 1 (projects onto 1D subspace)
- nullity(P) = 1 (|1⟩ is sent to 0)
- Check: 2 = 1 + 1 ✓

---

## 🌆 Afternoon Session: Problem Solving (2.5 hours)

### Problem Set A: Basic Applications (50 min)

**Problem 1.** Let T: ℝ⁴ → ℝ³ have rank 2. What is nullity(T)?

**Solution:**
By rank-nullity: 4 = nullity(T) + 2
Therefore nullity(T) = 2.

**Problem 2.** Can a linear map T: ℝ⁵ → ℝ³ be injective?

**Solution:**
If T is injective, rank(T) = dim(ℝ⁵) = 5.
But rank(T) ≤ dim(ℝ³) = 3.
5 ≤ 3 is false, so T cannot be injective.

**Problem 3.** Can a linear map T: ℝ³ → ℝ⁵ be surjective?

**Solution:**
If T is surjective, rank(T) = dim(ℝ⁵) = 5.
But rank(T) ≤ dim(ℝ³) = 3.
5 ≤ 3 is false, so T cannot be surjective.

**Problem 4.** Let A be a 4×6 matrix with rank 3. Find:
a) nullity(A)
b) Dimension of solution space of Ax = 0
c) If Ax = b is consistent, how many parameters describe all solutions?

**Solution:**
a) nullity(A) = 6 - 3 = 3

b) The solution space of Ax = 0 is ker(A), which has dimension = nullity(A) = 3.

c) Solutions are x₀ + ker(A), a 3-dimensional affine subspace.
   So 3 free parameters describe all solutions.

**Problem 5.** Prove: If T: V → V with dim(V) < ∞, then T injective ⟺ T surjective.

**Proof:**
Let dim(V) = n.

T injective ⟺ nullity(T) = 0 ⟺ rank(T) = n
                        ⟺ dim(range(T)) = n = dim(V)
                        ⟺ range(T) = V
                        ⟺ T surjective ∎

### Problem Set B: Matrix Problems (50 min)

**Problem 6.** For which values of a does the following system have:
(i) no solution, (ii) a unique solution, (iii) infinitely many solutions?

$$\begin{pmatrix} 1 & 2 & 3 \\ 0 & a-4 & 0 \\ 0 & 0 & a-5 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$$

**Solution:**
The matrix is upper triangular. Pivots exist when a ≠ 4 and a ≠ 5.

Case 1: a ≠ 4 and a ≠ 5
- Rank = 3, unique solution.

Case 2: a = 4
- Row 2 becomes [0, 0, 0 | 2], so no solution.

Case 3: a = 5
- Row 3 becomes [0, 0, 0 | 3], so no solution.

Therefore:
(i) No solution: a = 4 or a = 5
(ii) Unique solution: a ≠ 4 and a ≠ 5
(iii) Infinitely many: never (for any a, either no solution or unique)

**Problem 7.** Let A be an n×n matrix. Prove: Ax = b has a unique solution for all b ⟺ A is invertible.

**Proof:**
(⟹) If unique solution exists for all b:
- ∀b has a solution ⟹ A is surjective ⟹ rank(A) = n
- Solution is unique ⟹ ker(A) = {0} ⟹ A is injective
- For square A: surjective + injective ⟹ bijective ⟹ A invertible.

(⟸) If A is invertible:
- x = A⁻¹b is the unique solution. ∎

**Problem 8.** Let A be a 5×7 matrix with nullity 3. What is rank(A)?

**Solution:**
rank(A) = 7 - nullity(A) = 7 - 3 = 4.

### Problem Set C: Proofs (50 min)

**Problem 9.** Prove: For T: V → W, rank(T) ≤ min(dim(V), dim(W)).

**Proof:**
- rank(T) = dim(range(T)) ≤ dim(W) (range is subspace of W)
- rank(T) = dim(V) - nullity(T) ≤ dim(V) (since nullity ≥ 0)

Therefore rank(T) ≤ min(dim(V), dim(W)). ∎

**Problem 10.** Let S: U → V and T: V → W be linear. Prove:
a) rank(T ∘ S) ≤ min(rank(T), rank(S))
b) nullity(T ∘ S) ≥ nullity(S)

**Proof:**
a) range(T ∘ S) = T(S(U)) = T(range(S)) ⊆ range(T)
   So rank(T ∘ S) ≤ rank(T).
   
   Also, range(T ∘ S) = T(range(S)).
   Since T restricted to range(S) has rank at most dim(range(S)) = rank(S):
   rank(T ∘ S) ≤ rank(S).
   
   Therefore rank(T ∘ S) ≤ min(rank(T), rank(S)). ∎

b) ker(S) ⊆ ker(T ∘ S) (if S(u) = 0, then T(S(u)) = T(0) = 0)
   So nullity(S) ≤ nullity(T ∘ S). ∎

**Problem 11.** Let T: V → V with T² = T (T is a projection). Use rank-nullity to show V = ker(T) ⊕ range(T).

**Proof:**
We proved yesterday that V = ker(T) + range(T) and ker(T) ∩ range(T) = {0}.

By rank-nullity: dim(V) = nullity(T) + rank(T) = dim(ker(T)) + dim(range(T)).

For a direct sum V = U ⊕ W, we need:
1. V = U + W ✓
2. U ∩ W = {0} ✓
3. dim(V) = dim(U) + dim(W) ✓

All conditions satisfied, so V = ker(T) ⊕ range(T). ∎

---

## 🌙 Evening Session: Computational Lab (1 hour)

```python
"""
Day 96: Rank-Nullity Theorem
"""

import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

# =============================================================
# Part 1: Verifying Rank-Nullity
# =============================================================

print("="*60)
print("Verifying the Rank-Nullity Theorem")
print("="*60)

def rank_nullity_analysis(A, name="A"):
    """Complete analysis of a matrix using rank-nullity."""
    m, n = A.shape
    
    # Compute rank (number of linearly independent rows/columns)
    rank = np.linalg.matrix_rank(A)
    
    # Compute nullity (dimension of null space)
    ns = null_space(A)
    nullity = ns.shape[1] if ns.size > 0 else 0
    
    print(f"\n{name} is {m}×{n}:")
    print(f"Matrix:\n{A}")
    print(f"\nRank = {rank}")
    print(f"Nullity = {nullity}")
    print(f"rank + nullity = {rank} + {nullity} = {rank + nullity}")
    print(f"Number of columns = {n}")
    print(f"Rank-Nullity verified: {rank + nullity == n}")
    
    return rank, nullity


# Example 1: Full rank square matrix
A1 = np.array([[1, 2], [3, 4]])
rank_nullity_analysis(A1, "A1 (2×2 invertible)")

# Example 2: Rank-deficient square matrix
A2 = np.array([[1, 2], [2, 4]])
rank_nullity_analysis(A2, "A2 (2×2 singular)")

# Example 3: Wide matrix (more columns)
A3 = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8]])
rank_nullity_analysis(A3, "A3 (2×4 wide)")

# Example 4: Tall matrix (more rows)
A4 = np.array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
rank_nullity_analysis(A4, "A4 (4×2 tall)")

# Example 5: Special pattern
A5 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
rank_nullity_analysis(A5, "A5 (3×3 rank-deficient)")

# =============================================================
# Part 2: Applications to Linear Systems
# =============================================================

print("\n" + "="*60)
print("Applications to Linear Systems Ax = b")
print("="*60)

def analyze_linear_system(A, b):
    """Analyze solution structure using rank-nullity."""
    m, n = A.shape
    rank_A = np.linalg.matrix_rank(A)
    
    # Augmented matrix
    Ab = np.column_stack([A, b])
    rank_Ab = np.linalg.matrix_rank(Ab)
    
    print(f"\nA is {m}×{n}, b is {m}×1")
    print(f"rank(A) = {rank_A}")
    print(f"rank([A|b]) = {rank_Ab}")
    
    if rank_A < rank_Ab:
        print("No solution exists (b not in column space of A)")
        return None
    else:
        nullity = n - rank_A
        print(f"Solutions exist!")
        print(f"Nullity of A = {nullity} = number of free parameters")
        
        if nullity == 0:
            print("Unique solution")
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            print(f"x = {x.round(4)}")
        else:
            print(f"Infinitely many solutions (affine subspace of dimension {nullity})")
            x_particular = np.linalg.lstsq(A, b, rcond=None)[0]
            ker = null_space(A)
            print(f"Particular solution x₀ = {x_particular.round(4)}")
            print(f"Kernel basis (columns):\n{ker.round(4)}")
        return True


# System with unique solution
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 11])
print("\n--- System 1 ---")
analyze_linear_system(A, b)

# System with infinitely many solutions
A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([6, 15])
print("\n--- System 2 ---")
analyze_linear_system(A, b)

# System with no solution
A = np.array([[1, 2], [2, 4]])
b = np.array([3, 5])
print("\n--- System 3 ---")
analyze_linear_system(A, b)

# =============================================================
# Part 3: Rank-Nullity for Compositions
# =============================================================

print("\n" + "="*60)
print("Rank-Nullity for Compositions")
print("="*60)

def analyze_composition(A, B):
    """Analyze rank-nullity for composition AB."""
    # A is m×n, B is n×p, so AB is m×p
    AB = A @ B
    
    rank_A = np.linalg.matrix_rank(A)
    rank_B = np.linalg.matrix_rank(B)
    rank_AB = np.linalg.matrix_rank(AB)
    
    print(f"\nA is {A.shape}, rank(A) = {rank_A}")
    print(f"B is {B.shape}, rank(B) = {rank_B}")
    print(f"AB is {AB.shape}, rank(AB) = {rank_AB}")
    print(f"\nVerify: rank(AB) ≤ min(rank(A), rank(B)) = {min(rank_A, rank_B)}")
    print(f"Satisfied: {rank_AB <= min(rank_A, rank_B)}")


# Example compositions
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2, rank 2
B = np.array([[1, 0, 0], [0, 1, 0]])    # 2×3, rank 2
analyze_composition(A, B)

# Another example with rank drop
A = np.array([[1, 1], [1, 1]])  # rank 1
B = np.array([[1, 0], [0, 1]])  # rank 2
analyze_composition(A, B)

# =============================================================
# Part 4: Quantum Application - Projections
# =============================================================

print("\n" + "="*60)
print("Quantum Projections and Rank-Nullity")
print("="*60)

def analyze_projection(P, name="P"):
    """Analyze a projection using rank-nullity."""
    n = P.shape[0]
    
    # Verify it's a projection
    is_proj = np.allclose(P @ P, P)
    
    # Rank and nullity
    rank = np.linalg.matrix_rank(P)
    nullity = n - rank
    
    # Eigenvalues (should be 0 and 1)
    eigvals = np.linalg.eigvals(P)
    
    print(f"\n{name} ({n}×{n}):")
    print(f"Matrix:\n{P.round(4)}")
    print(f"Is projection (P² = P): {is_proj}")
    print(f"Rank (dim of range) = {rank}")
    print(f"Nullity (dim of kernel) = {nullity}")
    print(f"rank + nullity = {rank + nullity} = {n} ✓")
    print(f"Eigenvalues: {np.sort(eigvals.real).round(4)}")
    
    return rank, nullity


# Projection onto |0⟩ in 2D
P_0 = np.array([[1, 0], [0, 0]])
analyze_projection(P_0, "|0⟩⟨0|")

# Projection onto |+⟩ = (|0⟩ + |1⟩)/√2
ket_plus = np.array([1, 1]) / np.sqrt(2)
P_plus = np.outer(ket_plus, ket_plus)
analyze_projection(P_plus, "|+⟩⟨+|")

# 2-qubit: Projection onto span{|00⟩, |11⟩}
# Basis: |00⟩ = [1,0,0,0], |11⟩ = [0,0,0,1]
P_bell = np.diag([1, 0, 0, 1])
analyze_projection(P_bell, "P(span{|00⟩,|11⟩})")

# =============================================================
# Part 5: Visualization
# =============================================================

def visualize_rank_nullity():
    """Visualize rank-nullity theorem geometrically."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Example 1: Injective map (trivial kernel)
    ax1 = axes[0]
    ax1.set_title('Injective: T: ℝ² → ℝ³\nker(T) = {0}, rank = 2')
    
    # Draw domain R²
    square = plt.Polygon([[0, 0], [1, 0], [1, 1], [0, 1]], 
                         fill=True, alpha=0.3, color='blue', label='Domain ℝ²')
    ax1.add_patch(square)
    
    # Arrow to codomain
    ax1.annotate('T (no collapse)', xy=(2, 0.5), xytext=(1.2, 0.5),
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    # Embedded plane in R³ (shown as parallelogram)
    para = plt.Polygon([[2, 0], [3.5, 0.3], [3.5, 1.3], [2, 1]], 
                       fill=True, alpha=0.3, color='green', label='Range (2D in ℝ³)')
    ax1.add_patch(para)
    
    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(-0.5, 2)
    ax1.legend()
    ax1.axis('equal')
    ax1.set_aspect('equal')
    
    # Example 2: Surjective map (nontrivial kernel)
    ax2 = axes[1]
    ax2.set_title('Surjective: T: ℝ³ → ℝ²\nker(T) = line, rank = 2')
    
    # Domain R³ (represented as cube front face)
    cube = plt.Polygon([[0, 0], [1.2, 0], [1.2, 1.2], [0, 1.2]], 
                       fill=True, alpha=0.3, color='blue', label='Domain ℝ³')
    ax2.add_patch(cube)
    
    # Kernel line
    ax2.plot([0.6, 0.6], [0, 1.2], 'r-', linewidth=3, label='ker(T) (collapsed)')
    
    ax2.annotate('T', xy=(2, 0.6), xytext=(1.4, 0.6),
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    # Range = all of R²
    rect = plt.Polygon([[2, 0], [3.2, 0], [3.2, 1.2], [2, 1.2]], 
                       fill=True, alpha=0.3, color='green', label='Range = ℝ²')
    ax2.add_patch(rect)
    
    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(-0.5, 2)
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Example 3: Projection
    ax3 = axes[2]
    ax3.set_title('Projection: P: ℝ² → ℝ²\nker(P) ⊕ range(P) = ℝ²')
    
    # Draw plane
    ax3.axhline(y=0, color='gray', linewidth=0.5)
    ax3.axvline(x=0, color='gray', linewidth=0.5)
    
    # Range (x-axis)
    ax3.plot([-2, 2], [0, 0], 'g-', linewidth=3, label='range(P)')
    
    # Kernel (y-axis)
    ax3.plot([0, 0], [-2, 2], 'r-', linewidth=3, label='ker(P)')
    
    # Show a vector and its projection
    v = np.array([1.5, 1])
    Pv = np.array([1.5, 0])  # Projection onto x-axis
    
    ax3.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.05, fc='blue', ec='blue')
    ax3.arrow(0, 0, Pv[0], Pv[1], head_width=0.1, head_length=0.05, fc='green', ec='green')
    ax3.plot([v[0], Pv[0]], [v[1], Pv[1]], 'k--', alpha=0.5)
    
    ax3.text(v[0]+0.1, v[1]+0.1, 'v', fontsize=12)
    ax3.text(Pv[0]+0.1, Pv[1]-0.3, 'P(v)', fontsize=12)
    
    ax3.set_xlim(-2, 2.5)
    ax3.set_ylim(-2, 2)
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day96_rank_nullity.png', dpi=150)
    plt.show()


visualize_rank_nullity()

# =============================================================
# Part 6: Interactive Exploration
# =============================================================

print("\n" + "="*60)
print("Parameter Space of Matrices")
print("="*60)

def explore_rank_distribution(m, n, num_samples=1000):
    """Explore distribution of ranks for random matrices."""
    ranks = []
    for _ in range(num_samples):
        A = np.random.randn(m, n)
        ranks.append(np.linalg.matrix_rank(A))
    
    unique, counts = np.unique(ranks, return_counts=True)
    print(f"\nFor random {m}×{n} matrices:")
    print(f"Max possible rank = {min(m, n)}")
    for r, c in zip(unique, counts):
        print(f"  rank = {r}: {c/num_samples*100:.1f}%")


explore_rank_distribution(3, 3)
explore_rank_distribution(3, 5)
explore_rank_distribution(5, 3)
```

---

## 📝 Homework

### Written Problems

1. An 8×5 matrix A has rank 4. Find:
   a) nullity(A)
   b) dimension of solution space of Ax = 0
   c) Is A injective? Surjective?

2. Prove: For any A, rank(A) = rank(Aᵀ).

3. Let T: V → V where dim(V) = n. Prove:
   - If T^k = 0 for some k, then rank(T) ≤ n/2 when k = 2.
   
4. Let A be n×n. Prove: ker(A) = ker(A²) implies range(A) = range(A²).

5. In quantum mechanics, let P₁, P₂ be orthogonal projections (P₁P₂ = P₂P₁ = 0) on ℂⁿ. Prove:
   rank(P₁ + P₂) = rank(P₁) + rank(P₂)

---

## ✅ Daily Checklist

- [ ] Can state rank-nullity theorem
- [ ] Understand the proof
- [ ] Can apply to find dimensions
- [ ] Understand connection to linear systems
- [ ] Completed all problem sets
- [ ] Ran computational lab
- [ ] Understand geometric interpretation

---

## 🔮 Preview: Tomorrow

**Day 97: Computational Lab**
- NumPy implementations
- Solving systems numerically
- SVD and rank computation
- Applications to data analysis

---

*"The essence of mathematics is not to make simple things complicated, but to make complicated things simple."*
— Stan Gudder
