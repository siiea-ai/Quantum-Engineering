# Day 89: Bases and Dimension

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Bases and Dimension |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define basis and state its two essential properties
2. Prove that every basis of a finite-dimensional space has the same size
3. Calculate the dimension of vector spaces and subspaces
4. Find bases for various vector spaces
5. Understand coordinate representations
6. Connect bases to complete sets of states in quantum mechanics

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 2.B**: Bases (pp. 43-52)
- **Section 2.C**: Dimension (pp. 53-60)

### Secondary Text: Strang, "Introduction to Linear Algebra"
- **Section 3.4**: Independence, Basis, and Dimension (pp. 162-175)

---

## 🎬 Video Resources

### 3Blue1Brown: Essence of Linear Algebra
- **Chapter 2**: Linear combinations, span, and basis vectors
- Excellent geometric intuition for bases

### MIT OCW 18.06
- **Lecture 9**: Independence, Basis, and Dimension
- Gilbert Strang's masterful explanation

---

## 📖 Core Content: Theory and Concepts

### 1. Definition of Basis

**Definition:** A **basis** of a vector space $V$ is a list of vectors in $V$ that is:
1. **Linearly independent**
2. **Spans** $V$

A basis is the "Goldilocks" set of vectors: not too many (dependent), not too few (doesn't span).

### 2. Equivalent Characterization

**Theorem:** A list $v_1, \ldots, v_n$ is a basis of $V$ if and only if every vector in $V$ can be written **uniquely** as a linear combination of $v_1, \ldots, v_n$.

**Proof:**
$(\Rightarrow)$ If basis, then spans $V$ (every vector is a linear combination) and independent (unique representation, proved yesterday).

$(\Leftarrow)$ Unique representation means it spans (every vector is some combination) and is independent (if $\sum a_iv_i = 0 = \sum 0v_i$, then $a_i = 0$ by uniqueness).

### 3. Standard Bases

#### Standard Basis of ℝⁿ (and ℂⁿ)
$$e_1 = (1, 0, 0, \ldots, 0), \quad e_2 = (0, 1, 0, \ldots, 0), \quad \ldots, \quad e_n = (0, \ldots, 0, 1)$$

Any $(x_1, \ldots, x_n) = x_1e_1 + \cdots + x_ne_n$

#### Standard Basis of $\mathcal{P}_n(\mathbb{F})$
$$1, x, x^2, \ldots, x^n$$

Any polynomial $a_0 + a_1x + \cdots + a_nx^n$ is a unique linear combination.

#### Standard Basis of $M_{m \times n}(\mathbb{F})$
The matrices $E_{ij}$ with 1 in position $(i,j)$ and 0 elsewhere.

For $M_{2 \times 2}(\mathbb{R})$: $\begin{pmatrix}1&0\\0&0\end{pmatrix}, \begin{pmatrix}0&1\\0&0\end{pmatrix}, \begin{pmatrix}0&0\\1&0\end{pmatrix}, \begin{pmatrix}0&0\\0&1\end{pmatrix}$

### 4. Finite-Dimensional Vector Spaces

**Definition:** A vector space $V$ is **finite-dimensional** if it has a finite spanning list (some finite list spans $V$).

**Observation:** $\mathcal{P}(\mathbb{F})$ (all polynomials, no degree bound) is NOT finite-dimensional.

### 5. The Dimension Theorem

**Theorem (Fundamental):** Any two bases of a finite-dimensional vector space have the same length.

**Definition:** The **dimension** of a finite-dimensional vector space $V$ is the length of any basis of $V$.

**Notation:** dim $V$ or dim($V$)

**Examples:**
- dim ℝⁿ = $n$
- dim ℂⁿ = $n$ (over ℂ)
- dim $\mathcal{P}_n(\mathbb{F})$ = $n + 1$
- dim $M_{m \times n}(\mathbb{F})$ = $mn$

### 6. Key Theorems About Dimension

**Theorem 1:** Every spanning list can be reduced to a basis.
(Remove vectors that are in the span of preceding ones.)

**Theorem 2:** Every linearly independent list can be extended to a basis.
(Add vectors not in the span until you span the whole space.)

**Theorem 3:** If $V$ is finite-dimensional and $U$ is a subspace of $V$, then
$$\dim U \leq \dim V$$
with equality iff $U = V$.

**Theorem 4:** In an $n$-dimensional space:
- Any $n$ independent vectors form a basis
- Any $n$ vectors that span form a basis

### 7. Finding Bases

#### For subspaces defined by equations:
1. Write general element using free parameters
2. Express as linear combination of fixed vectors
3. Those vectors form a basis

**Example:** $U = \{(x, y, z) : x + y + z = 0\}$ in ℝ³

General element: $(x, y, -x-y) = x(1, 0, -1) + y(0, 1, -1)$

Basis: $\{(1, 0, -1), (0, 1, -1)\}$, dim $U$ = 2

#### For spans:
1. Form matrix with given vectors as columns
2. Row reduce
3. Pivot columns in original matrix form a basis

### 8. Coordinate Vectors

**Definition:** If $B = (v_1, \ldots, v_n)$ is an ordered basis of $V$, and
$$v = a_1v_1 + \cdots + a_nv_n$$
then the **coordinate vector** of $v$ with respect to $B$ is:
$$[v]_B = \begin{pmatrix} a_1 \\ \vdots \\ a_n \end{pmatrix}$$

**Key Property:** The map $v \mapsto [v]_B$ is a bijection (isomorphism) from $V$ to $\mathbb{F}^n$.

---

## 🔬 Quantum Mechanics Connection

### Complete Orthonormal Bases

In quantum mechanics, a **complete orthonormal basis** (CONB) is the fundamental object:

**Definition:** A CONB for a Hilbert space $\mathcal{H}$ is a basis $\{|n\rangle\}$ such that:
1. $\langle m|n \rangle = \delta_{mn}$ (orthonormal)
2. Every state can be expanded: $|\psi\rangle = \sum_n c_n |n\rangle$ (complete)

### Dimension and Quantum Systems

| Physical System | Hilbert Space | Dimension |
|-----------------|---------------|-----------|
| Qubit (spin-1/2) | ℂ² | 2 |
| Qutrit (spin-1) | ℂ³ | 3 |
| Spin-$s$ particle | ℂ^{2s+1} | $2s + 1$ |
| $n$ qubits | (ℂ²)^⊗n = ℂ^{2ⁿ} | $2^n$ |
| Harmonic oscillator | ℓ² | ∞ |

### The Computational Basis

For an $n$-qubit system, the **computational basis** is:
$$|00\ldots0\rangle, |00\ldots1\rangle, \ldots, |11\ldots1\rangle$$

These $2^n$ vectors form a basis for ℂ^{2ⁿ}.

Any $n$-qubit state:
$$|\psi\rangle = \sum_{i=0}^{2^n-1} c_i |i\rangle$$

where $|i\rangle$ is the binary representation.

### Change of Basis in QM

Different physical observables suggest different bases:

- **Position basis:** $\{|x\rangle\}$ — eigenkets of position operator
- **Momentum basis:** $\{|p\rangle\}$ — eigenkets of momentum operator
- **Energy basis:** $\{|E_n\rangle\}$ — eigenkets of Hamiltonian

The wave function $\psi(x) = \langle x|\psi\rangle$ is the coordinate representation in the position basis!

### Dimension Determines Information Content

A quantum system of dimension $d$ can encode at most $\log_2(d)$ bits of classical information.

- 1 qubit: $\log_2(2) = 1$ bit
- $n$ qubits: $\log_2(2^n) = n$ bits

But quantum mechanically, we need $2^n$ complex amplitudes to describe the state!

---

## ✏️ Worked Examples

### Example 1: Finding a Basis for a Subspace

**Question:** Find a basis for $U = \{(a, b, c, d) \in \mathbb{R}^4 : a = b + c, \, d = 0\}$.

**Solution:**
Using $a = b + c$ and $d = 0$:
$$(a, b, c, d) = (b + c, b, c, 0) = b(1, 1, 0, 0) + c(1, 0, 1, 0)$$

Basis: $\{(1, 1, 0, 0), (1, 0, 1, 0)\}$

dim $U$ = 2

### Example 2: Dimension of Matrix Subspace

**Question:** What is the dimension of the space of symmetric $2 \times 2$ matrices?

**Solution:**
A symmetric matrix has form: $\begin{pmatrix} a & b \\ b & c \end{pmatrix}$

Basis: $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$

Dimension = 3

(General formula: $n \times n$ symmetric matrices have dimension $\frac{n(n+1)}{2}$.)

### Example 3: Extending to a Basis

**Question:** Extend $\{(1, 1, 0), (1, 0, 1)\}$ to a basis of ℝ³.

**Solution:**
We have 2 independent vectors; need 3 for ℝ³.

Try adding standard basis vectors:
- $(1, 0, 0)$: Check independence with $(1, 1, 0), (1, 0, 1)$

$$\det\begin{pmatrix} 1 & 1 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 1 \end{pmatrix} = 1(0-1) - 1(1-0) + 1(1-0) = -1 - 1 + 1 = -1 \neq 0$$

Independent! Basis: $\{(1, 1, 0), (1, 0, 1), (1, 0, 0)\}$

### Example 4: Coordinate Vector

**Question:** Let $B = \{1 + x, 1 - x\}$ be a basis for $\mathcal{P}_1(\mathbb{R})$. Find $[3 + 5x]_B$.

**Solution:**
Need: $a(1 + x) + b(1 - x) = 3 + 5x$

$(a + b) + (a - b)x = 3 + 5x$

System: $a + b = 3$ and $a - b = 5$

Adding: $2a = 8 \Rightarrow a = 4$
Subtracting: $2b = -2 \Rightarrow b = -1$

$$[3 + 5x]_B = \begin{pmatrix} 4 \\ -1 \end{pmatrix}$$

Verify: $4(1 + x) + (-1)(1 - x) = 4 + 4x - 1 + x = 3 + 5x$ ✓

### Example 5: Basis from Span

**Question:** Find a basis for span$\{(1, 2, 3), (4, 5, 6), (7, 8, 9), (2, 4, 6)\}$.

**Solution:**
Form matrix and row reduce:
$$\begin{pmatrix} 1 & 4 & 7 & 2 \\ 2 & 5 & 8 & 4 \\ 3 & 6 & 9 & 6 \end{pmatrix} \xrightarrow{RREF} \begin{pmatrix} 1 & 0 & -1 & 2 \\ 0 & 1 & 2 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Pivot columns are 1 and 2. Original vectors at positions 1 and 2 form a basis:

Basis: $\{(1, 2, 3), (4, 5, 6)\}$

Dimension of span = 2

---

## 📝 Practice Problems

### Level 1: Basic Calculations
1. Find the dimension of $\{(x, y, z) : x + y = 0\}$ in ℝ³.

2. Is $\{(1, 0, 1), (0, 1, 1), (1, 1, 2)\}$ a basis for ℝ³?

3. Find a basis for the space of $2 \times 2$ matrices with trace zero.

### Level 2: Finding Bases
4. Find a basis for $U = \{(a, b, c, d) : a + b + c + d = 0\}$ in ℝ⁴.

5. Extend $\{1 + x^2, x\}$ to a basis of $\mathcal{P}_2(\mathbb{R})$.

6. Find a basis for the column space of $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 3 & 4 \end{pmatrix}$.

### Level 3: Proofs
7. Prove: If dim $V$ = $n$ and $v_1, \ldots, v_n$ span $V$, then they are linearly independent.

8. Prove: If $U$ and $W$ are subspaces of $V$ with $U \subseteq W$ and dim $U$ = dim $W$, then $U = W$.

9. Prove: dim($U + W$) = dim $U$ + dim $W$ - dim($U \cap W$) for subspaces $U, W$.

### Level 4: Challenge
10. Find the dimension of the space of $n \times n$ skew-symmetric matrices.

11. Let $T: V \to W$ be a linear map. Prove: dim $V$ = dim(ker $T$) + dim(range $T$).

12. A basis $B$ of ℂ² over ℂ has 2 elements. How many elements does a basis of ℂ² over ℝ have?

---

## 📊 Answers and Hints

1. dim = 2; basis: $\{(1, -1, 0), (0, 0, 1)\}$
2. No — compute determinant (= 0), or note $v_3 = v_1 + v_2$
3. Trace zero: $\begin{pmatrix} a & b \\ c & -a \end{pmatrix}$; dim = 3
4. Basis: $\{(1, -1, 0, 0), (1, 0, -1, 0), (1, 0, 0, -1)\}$; dim = 3
5. Add $1$ or $x^2$: Basis = $\{1 + x^2, x, 1\}$ or $\{1 + x^2, x, x^2\}$
6. Row reduce, identify pivot columns
7. If dependent, some $v_j$ is in span of others; removing it still spans, giving $n-1$ vectors spanning $n$-dim space — contradiction
8. $U \subseteq W$ means $U$ is subspace of $W$; dim $U$ = dim $W$ implies $U = W$
9. Dimension formula for sum of subspaces
10. $\frac{n(n-1)}{2}$
11. Rank-nullity theorem
12. 4 elements (ℂ² as real vector space is ℝ⁴)

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

# ============================================
# Lab 1: Computing Dimension via Rank
# ============================================

def find_dimension(vectors):
    """Compute dimension of span of vectors."""
    if len(vectors) == 0:
        return 0
    A = np.column_stack(vectors)
    return np.linalg.matrix_rank(A)

def find_basis(vectors, tol=1e-10):
    """Find a basis for the span of vectors (subset of input vectors)."""
    if len(vectors) == 0:
        return []
    
    A = np.column_stack(vectors)
    m, n = A.shape
    
    # Row reduce to find pivot columns
    # Using QR decomposition for numerical stability
    Q, R = np.linalg.qr(A)
    
    # Find columns with non-negligible diagonal entries in R
    diag = np.abs(np.diag(R[:min(m,n), :min(m,n)]))
    pivot_cols = np.where(diag > tol)[0]
    
    return [vectors[i] for i in pivot_cols]

# Test
vectors = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9]),
    np.array([2, 4, 6])
]

print("Original vectors:")
for v in vectors:
    print(f"  {v}")

dim = find_dimension(vectors)
basis = find_basis(vectors)

print(f"\nDimension: {dim}")
print(f"Basis:")
for v in basis:
    print(f"  {v}")

# ============================================
# Lab 2: Basis for Subspace Defined by Equations
# ============================================

def basis_from_equations(A, b=None):
    """
    Find basis for solution space of Ax = b.
    If b is None, find basis for null space (Ax = 0).
    """
    if b is None:
        # Null space
        ns = null_space(A)
        return [ns[:, i] for i in range(ns.shape[1])]
    else:
        # Particular solution + null space
        # (More complex, handle separately)
        pass

# Example: x + y + z = 0 in ℝ³
A = np.array([[1, 1, 1]])
basis_null = basis_from_equations(A)

print("\nSubspace {x + y + z = 0}:")
print(f"Dimension: {len(basis_null)}")
print("Basis vectors:")
for v in basis_null:
    print(f"  {v}")

# Verify they satisfy equation
for v in basis_null:
    print(f"  A @ v = {A @ v}")

# ============================================
# Lab 3: Change of Basis
# ============================================

def coordinate_vector(v, basis):
    """Find coordinates of v with respect to given basis."""
    B = np.column_stack(basis)
    return np.linalg.solve(B, v)

def change_of_basis_matrix(old_basis, new_basis):
    """Compute matrix that converts from old coordinates to new coordinates."""
    # P[new <- old] such that [v]_new = P @ [v]_old
    n = len(old_basis)
    P = np.zeros((n, n))
    for i, v in enumerate(old_basis):
        P[:, i] = coordinate_vector(v, new_basis)
    return P

# Example in ℝ²
standard_basis = [np.array([1, 0]), np.array([0, 1])]
new_basis = [np.array([1, 1]), np.array([1, -1])]

# Find coordinates of (3, 5) in both bases
v = np.array([3, 5])

coords_std = coordinate_vector(v, standard_basis)
coords_new = coordinate_vector(v, new_basis)

print("\nChange of Basis in ℝ²:")
print(f"Vector v = {v}")
print(f"Coordinates in standard basis: {coords_std}")
print(f"Coordinates in new basis {{(1,1), (1,-1)}}: {coords_new}")
print(f"Verify: {coords_new[0]}*(1,1) + {coords_new[1]}*(1,-1) = {coords_new[0]*new_basis[0] + coords_new[1]*new_basis[1]}")

# Change of basis matrix
P = change_of_basis_matrix(standard_basis, new_basis)
print(f"\nChange of basis matrix (std -> new):\n{P}")
print(f"P @ [v]_std = {P @ coords_std}")

# ============================================
# Lab 4: Visualizing Different Bases in ℝ²
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def plot_basis_grid(ax, basis, color='blue', title=''):
    """Plot grid lines defined by a basis."""
    v1, v2 = basis
    
    # Plot basis vectors
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.02, label='e1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.02, label='e2')
    
    # Plot grid
    for i in range(-3, 4):
        for j in range(-3, 4):
            point = i * v1 + j * v2
            ax.plot(point[0], point[1], 'ko', markersize=3, alpha=0.5)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

# Standard basis
plot_basis_grid(axes[0], [np.array([1, 0]), np.array([0, 1])], title='Standard Basis')

# Rotated basis
theta = np.pi/6
plot_basis_grid(axes[1], [np.array([np.cos(theta), np.sin(theta)]), 
                          np.array([-np.sin(theta), np.cos(theta)])], title='Rotated Basis')

# Skewed basis
plot_basis_grid(axes[2], [np.array([1, 0.5]), np.array([0.5, 1])], title='Skewed Basis')

plt.tight_layout()
plt.savefig('bases_visualization.png', dpi=150)
plt.show()

# ============================================
# Lab 5: Dimension of Matrix Subspaces
# ============================================

def dim_symmetric(n):
    """Dimension of n×n symmetric matrices."""
    return n * (n + 1) // 2

def dim_skew_symmetric(n):
    """Dimension of n×n skew-symmetric matrices."""
    return n * (n - 1) // 2

def dim_trace_zero(n):
    """Dimension of n×n matrices with trace zero."""
    return n * n - 1

def dim_diagonal(n):
    """Dimension of n×n diagonal matrices."""
    return n

def dim_upper_triangular(n):
    """Dimension of n×n upper triangular matrices."""
    return n * (n + 1) // 2

print("\nDimensions of Matrix Subspaces (n = 3):")
n = 3
print(f"All 3×3 matrices: {n*n}")
print(f"Symmetric: {dim_symmetric(n)}")
print(f"Skew-symmetric: {dim_skew_symmetric(n)}")
print(f"Trace zero: {dim_trace_zero(n)}")
print(f"Diagonal: {dim_diagonal(n)}")
print(f"Upper triangular: {dim_upper_triangular(n)}")
print(f"\nNote: Symmetric + Skew-symmetric = {dim_symmetric(n)} + {dim_skew_symmetric(n)} = {n*n}")

# ============================================
# Lab 6: QM - Qubit State Space Bases
# ============================================

# Standard (computational) basis
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Hadamard (superposition) basis
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# Circular basis
ket_R = (ket_0 + 1j*ket_1) / np.sqrt(2)
ket_L = (ket_0 - 1j*ket_1) / np.sqrt(2)

print("\nQubit Bases (dim = 2):")
print("\nComputational basis:")
print(f"  |0⟩ = {ket_0}")
print(f"  |1⟩ = {ket_1}")

print("\nHadamard basis:")
print(f"  |+⟩ = {ket_plus}")
print(f"  |-⟩ = {ket_minus}")

print("\nCircular basis:")
print(f"  |R⟩ = {ket_R}")
print(f"  |L⟩ = {ket_L}")

# Express a state in different bases
psi = np.array([0.6, 0.8], dtype=complex)

coord_comp = coordinate_vector(psi, [ket_0, ket_1])
coord_had = coordinate_vector(psi, [ket_plus, ket_minus])

print(f"\nState |ψ⟩ = {psi}")
print(f"In computational basis: {coord_comp}")
print(f"In Hadamard basis: {coord_had}")

# Verify
print(f"\nVerification:")
print(f"  {coord_comp[0]}|0⟩ + {coord_comp[1]}|1⟩ = {coord_comp[0]*ket_0 + coord_comp[1]*ket_1}")
print(f"  {coord_had[0]}|+⟩ + {coord_had[1]}|-⟩ = {coord_had[0]*ket_plus + coord_had[1]*ket_minus}")
```

---

## ✅ Daily Checklist

- [ ] Read Axler Sections 2.B and 2.C
- [ ] Understand and memorize basis definition
- [ ] Learn dimension theorem
- [ ] Complete Level 1-2 problems
- [ ] Attempt Level 3 proofs
- [ ] Complete computational lab
- [ ] Create flashcards for:
  - Basis definition (two properties)
  - Dimension definition
  - Standard bases for ℝⁿ, 𝒫ₙ, matrices
  - Key dimension formulas
- [ ] Write QM connection notes

---

## 📓 Reflection Questions

1. Why is the dimension theorem surprising? (All bases same size)

2. What does it mean physically that a qubit Hilbert space has dimension 2?

3. How does changing basis affect the representation but not the vector itself?

4. Why do we often prefer orthonormal bases?

---

## 🔜 Preview: Tomorrow's Topics

**Day 90: Computational Lab — Putting It All Together**

Tomorrow we'll implement everything in Python:
- Vector space operations
- Subspace verification
- Independence testing
- Basis finding
- Dimension computation
- QM state manipulations

**Preparation:** Review all computational labs from this week.

---

*"The dimension of a vector space is its intrinsic size — the number of degrees of freedom."*
