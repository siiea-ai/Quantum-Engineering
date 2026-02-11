# Day 88: Linear Independence

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Linear Independence |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define linear independence and dependence precisely
2. Test sets of vectors for linear independence
3. Understand the geometric meaning of independence
4. Use the matrix method to test independence
5. Recognize that independent sets have unique representations
6. Connect linear independence to orthogonal states in QM

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 2.A**: Span and Linear Independence (pp. 33-42, independence portion)
- Focus on: Definition 2.17, theorems on independence

### Secondary Text: Strang, "Introduction to Linear Algebra"
- **Section 3.4**: Independence, Basis, and Dimension (pp. 162-175)

---

## 🎬 Video Resources

### 3Blue1Brown: Essence of Linear Algebra
- **Chapter 2**: Linear combinations, span, and basis vectors (review)
- The independence discussion is woven throughout

### MIT OCW 18.06
- **Lecture 7**: Solving Ax = 0: Pivot Variables, Special Solutions
- **Lecture 9**: Independence, Basis, and Dimension

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: When Are Vectors "Redundant"?

**Question:** Given vectors $v_1, v_2, v_3$, when does one of them "add nothing new" to the span?

**Answer:** When one vector can be written as a linear combination of the others.

This leads to the definition of linear independence.

### 2. Definition of Linear Independence

**Definition:** A list of vectors $v_1, \ldots, v_n$ in a vector space $V$ is **linearly independent** if the only way to write

$$a_1 v_1 + a_2 v_2 + \cdots + a_n v_n = 0$$

is with $a_1 = a_2 = \cdots = a_n = 0$.

**Equivalently:** No non-trivial linear combination equals zero.

**Definition (Dependent):** A list of vectors is **linearly dependent** if it is not linearly independent.

### 3. Equivalent Characterizations

**Theorem:** The following are equivalent for vectors $v_1, \ldots, v_n$:

1. $v_1, \ldots, v_n$ are linearly dependent
2. There exist scalars $a_1, \ldots, a_n$, not all zero, such that $a_1v_1 + \cdots + a_nv_n = 0$
3. Some $v_j$ can be written as a linear combination of the others
4. Removing some $v_j$ doesn't change the span: $\text{span}(v_1, \ldots, v_n) = \text{span}(v_1, \ldots, v_{j-1}, v_{j+1}, \ldots, v_n)$

### 4. Key Properties

**Property 1:** Any list containing the zero vector is linearly dependent.
*Proof:* $1 \cdot 0 + 0 \cdot v_2 + \cdots + 0 \cdot v_n = 0$ is a non-trivial combination.

**Property 2:** A list of one nonzero vector is linearly independent.
*Proof:* $av = 0$ with $v \neq 0$ implies $a = 0$.

**Property 3:** Two vectors are linearly dependent iff one is a scalar multiple of the other.
*Proof:* $av_1 + bv_2 = 0$ with $a \neq 0$ gives $v_1 = (-b/a)v_2$.

### 5. The Linear Dependence Lemma

**Lemma:** Suppose $v_1, \ldots, v_n$ is linearly dependent and $v_1 \neq 0$. Then there exists $j \in \{2, \ldots, n\}$ such that:
1. $v_j \in \text{span}(v_1, \ldots, v_{j-1})$
2. Removing $v_j$ doesn't change the span

**Importance:** This lemma is key for proving dimension theorems.

### 6. Testing Linear Independence

**Method 1: Definition (Direct)**
Set up $a_1v_1 + \cdots + a_nv_n = 0$ and solve. If only solution is all zeros, independent.

**Method 2: Matrix Method**
Form matrix $A = [v_1 | v_2 | \cdots | v_n]$ (vectors as columns).
The vectors are linearly independent iff $Ax = 0$ has only the trivial solution.
This happens iff $A$ has a pivot in every column (full column rank).

**Method 3: Determinant (Square Case)**
If we have $n$ vectors in ℝⁿ (or ℂⁿ), they are linearly independent iff $\det(A) \neq 0$.

### 7. Independence in Different Spaces

#### In ℝⁿ and ℂⁿ:
- At most $n$ vectors can be linearly independent
- $n$ independent vectors automatically span the space

#### In Polynomial Spaces:
$1, x, x^2, \ldots, x^n$ are linearly independent in $\mathcal{P}(\mathbb{F})$.
*Proof:* If $a_0 + a_1x + \cdots + a_nx^n = 0$ for all $x$, then all $a_i = 0$ (polynomial identity).

#### In Function Spaces:
$e^x, e^{2x}, e^{3x}$ are linearly independent in $C(\mathbb{R})$.
*Proof:* Use the Wronskian or evaluate at specific points.

### 8. Unique Representation Theorem

**Theorem:** If $v_1, \ldots, v_n$ are linearly independent and $v \in \text{span}(v_1, \ldots, v_n)$, then there is a **unique** way to write $v$ as a linear combination of $v_1, \ldots, v_n$.

**Proof:** Suppose $v = a_1v_1 + \cdots + a_nv_n = b_1v_1 + \cdots + b_nv_n$.
Then $(a_1 - b_1)v_1 + \cdots + (a_n - b_n)v_n = 0$.
By independence, $a_i - b_i = 0$ for all $i$, so $a_i = b_i$. ∎

**Importance:** This is why we want independent spanning sets (bases)!

---

## 🔬 Quantum Mechanics Connection

### Orthogonal States are Independent

In quantum mechanics, **orthogonal states** are automatically linearly independent:

**Theorem:** If $\langle\psi_i|\psi_j\rangle = 0$ for $i \neq j$ (orthogonal states), and each $|\psi_i\rangle \neq 0$, then $|\psi_1\rangle, \ldots, |\psi_n\rangle$ are linearly independent.

**Proof:** Suppose $\sum_i c_i|\psi_i\rangle = 0$.
Take inner product with $|\psi_k\rangle$:
$$\langle\psi_k|\sum_i c_i|\psi_i\rangle = \sum_i c_i\langle\psi_k|\psi_i\rangle = c_k\langle\psi_k|\psi_k\rangle = c_k||\psi_k||^2 = 0$$
Since $|\psi_k\rangle \neq 0$, we have $c_k = 0$. ∎

### Distinguishable Quantum States

Linearly independent states correspond to **distinguishable** (or at least partially distinguishable) quantum states:

- If $|\psi\rangle = c|\phi\rangle$ (linearly dependent), they represent the same physical state (up to phase)
- Linearly independent states can potentially be distinguished by some measurement

### Energy Eigenstates

Energy eigenstates of a Hamiltonian with distinct eigenvalues are automatically linearly independent:

If $\hat{H}|E_n\rangle = E_n|E_n\rangle$ with all $E_n$ distinct, then $|E_1\rangle, |E_2\rangle, \ldots$ are linearly independent.

**Why?** Eigenvectors of distinct eigenvalues are always linearly independent (general theorem from linear algebra).

### Measurement Outcomes

When we measure an observable, we get one of its eigenvalues, and the state collapses to the corresponding eigenspace. Linearly independent eigenvectors for distinct eigenvalues ensure:
- Different measurement outcomes correspond to distinguishable states
- The state after measurement is uniquely determined

---

## ✏️ Worked Examples

### Example 1: Testing Independence in ℝ³

**Question:** Are $(1, 0, 1)$, $(2, 1, 1)$, $(1, 1, 0)$ linearly independent?

**Solution (Matrix Method):**
$$A = \begin{pmatrix} 1 & 2 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 0 \end{pmatrix}$$

Row reduce:
$$\begin{pmatrix} 1 & 2 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 0 \end{pmatrix} \xrightarrow{R_3 - R_1} \begin{pmatrix} 1 & 2 & 1 \\ 0 & 1 & 1 \\ 0 & -1 & -1 \end{pmatrix} \xrightarrow{R_3 + R_2} \begin{pmatrix} 1 & 2 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

Only 2 pivots for 3 vectors. **Linearly dependent!**

**Finding the dependence relation:**
From RREF, we can find that $(1, 0, 1) - (2, 1, 1) + (1, 1, 0) = (0, 0, 0)$.

### Example 2: Testing Independence in ℝ² 

**Question:** Are $(1, 2)$ and $(3, 4)$ linearly independent?

**Solution (Determinant):**
$$\det\begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} = 1(4) - 3(2) = 4 - 6 = -2 \neq 0$$

**Linearly independent!**

### Example 3: Polynomials

**Question:** Are $p_1(x) = 1 + x$, $p_2(x) = 1 - x$, $p_3(x) = x$ linearly independent in $\mathcal{P}_1(\mathbb{R})$?

**Solution:**
$\mathcal{P}_1(\mathbb{R})$ has dimension 2 (basis: $\{1, x\}$).
We have 3 polynomials in a 2-dimensional space.

**At most 2 can be independent!** So they must be dependent.

Finding relation: $\frac{1}{2}p_1 + \frac{1}{2}p_2 - p_3 = \frac{1}{2}(1+x) + \frac{1}{2}(1-x) - x = 1 - x \neq 0$

Let's try again: $p_1 - p_2 = (1+x) - (1-x) = 2x = 2p_3$

So $p_1 - p_2 - 2p_3 = 0$. **Dependent!**

### Example 4: Functions

**Question:** Are $f_1(x) = e^x$, $f_2(x) = e^{2x}$ linearly independent?

**Solution (Wronskian):**
The Wronskian is:
$$W(f_1, f_2) = \det\begin{pmatrix} e^x & e^{2x} \\ e^x & 2e^{2x} \end{pmatrix} = 2e^{3x} - e^{3x} = e^{3x} \neq 0$$

Since Wronskian is nonzero, **linearly independent!**

**Alternative (evaluation):**
If $ae^x + be^{2x} = 0$ for all $x$:
- At $x = 0$: $a + b = 0$
- At $x = 1$: $ae + be^2 = 0$, so $a = -b$ gives $-be + be^2 = b(e^2 - e) = 0$

Since $e^2 - e \neq 0$, we need $b = 0$, hence $a = 0$. Independent!

### Example 5: Complex Vectors

**Question:** In ℂ², are $(1, i)$ and $(i, -1)$ linearly independent over ℂ?

**Solution:**
Note: $(i, -1) = i(1, i)$ (check: $i \cdot 1 = i$, $i \cdot i = -1$ ✓)

So $(i, -1)$ is a scalar multiple of $(1, i)$.

**Linearly dependent!** (only span a 1-dimensional subspace of ℂ²)

---

## 📝 Practice Problems

### Level 1: Basic Tests
1. Are $(1, 2)$ and $(2, 4)$ linearly independent in ℝ²?

2. Are $(1, 0, 0)$, $(0, 1, 0)$, $(0, 0, 1)$ linearly independent in ℝ³?

3. Are $(1, 1, 1)$, $(1, 2, 3)$, $(2, 3, 4)$ linearly independent in ℝ³?

### Level 2: Calculations
4. Find all values of $c$ for which $(1, c, 0)$, $(0, 1, c)$, $(c, 0, 1)$ are linearly dependent.

5. Show that $\sin x$ and $\cos x$ are linearly independent in $C(\mathbb{R})$.

6. For what values of $k$ are $(1, k, 3)$, $(2, 1, 1)$, $(3, 4, k)$ linearly dependent?

### Level 3: Proofs
7. Prove: If $v_1, \ldots, v_n$ are linearly independent and $v_{n+1} \notin \text{span}(v_1, \ldots, v_n)$, then $v_1, \ldots, v_{n+1}$ are linearly independent.

8. Prove: Distinct eigenvectors of a linear operator corresponding to distinct eigenvalues are linearly independent.

9. Prove: If $v_1, \ldots, v_n$ are linearly independent in $V$, and $T: V \to W$ is injective (one-to-one), then $Tv_1, \ldots, Tv_n$ are linearly independent in $W$.

### Level 4: Challenge
10. Let $f_1, f_2, f_3$ be three times differentiable functions on ℝ. Prove that if the Wronskian
$$W(x) = \det\begin{pmatrix} f_1 & f_2 & f_3 \\ f_1' & f_2' & f_3' \\ f_1'' & f_2'' & f_3'' \end{pmatrix}$$
is nonzero for some $x$, then $f_1, f_2, f_3$ are linearly independent.

11. In ℂ², find two linearly independent vectors that are not orthogonal (w.r.t. standard inner product).

12. Prove: The maximum size of a linearly independent set in ℝⁿ is $n$.

---

## 📊 Answers and Hints

1. **Dependent** — $(2, 4) = 2(1, 2)$
2. **Independent** — standard basis
3. Compute determinant or row reduce
4. Set up determinant = 0, solve for $c$. Answer: $c = -1$ or $c = 1$
5. If $a\sin x + b\cos x = 0$ for all $x$: at $x = 0$: $b = 0$; at $x = \pi/2$: $a = 0$
6. Set determinant = 0, solve for $k$
7. If $a_1v_1 + \cdots + a_nv_n + a_{n+1}v_{n+1} = 0$, show $a_{n+1} = 0$ (else $v_{n+1}$ in span), then use independence of $v_1, \ldots, v_n$
8. If $\lambda_i \neq \lambda_j$ and $Tv_i = \lambda_iv_i$, use eigenvalue equation
9. If $a_1Tv_1 + \cdots + a_nTv_n = 0$, then $T(a_1v_1 + \cdots + a_nv_n) = 0$. Injectivity gives sum = 0.
10. If dependent, Wronskian identically zero (contrapositive)
11. Example: $(1, 0)$ and $(1, 1)$ — inner product is $1 \neq 0$
12. Use the fact that ℝⁿ is spanned by $n$ vectors

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det

# ============================================
# Lab 1: Testing Linear Independence
# ============================================

def test_independence(vectors, tol=1e-10):
    """
    Test if vectors are linearly independent.
    Returns (independent: bool, rank: int, null_space: array)
    """
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A, tol=tol)
    
    # Find null space (dependencies)
    U, S, Vt = np.linalg.svd(A)
    null_space = Vt[rank:].T  # Columns form basis of null space
    
    independent = (rank == len(vectors))
    
    return independent, rank, null_space

# Test examples
print("Testing Linear Independence:")
print("=" * 50)

# Example 1: Independent vectors in ℝ³
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
ind, rank, null = test_independence([v1, v2, v3])
print(f"\n{v1}, {v2}, {v3}")
print(f"Independent: {ind}, Rank: {rank}")

# Example 2: Dependent vectors
v1 = np.array([1, 0, 1])
v2 = np.array([2, 1, 1])
v3 = np.array([1, 1, 0])
ind, rank, null = test_independence([v1, v2, v3])
print(f"\n{v1}, {v2}, {v3}")
print(f"Independent: {ind}, Rank: {rank}")
if not ind:
    print(f"Null space dimension: {null.shape[1]}")
    # Verify dependency
    if null.shape[1] > 0:
        c = null[:, 0]
        result = c[0]*v1 + c[1]*v2 + c[2]*v3
        print(f"Dependency: {c[0]:.4f}*v1 + {c[1]:.4f}*v2 + {c[2]:.4f}*v3 = {result}")

# Example 3: More vectors than dimension
v1 = np.array([1, 2])
v2 = np.array([3, 4])
v3 = np.array([5, 6])
ind, rank, null = test_independence([v1, v2, v3])
print(f"\n{v1}, {v2}, {v3}")
print(f"Independent: {ind}, Rank: {rank}")

# ============================================
# Lab 2: Visualizing Independence in ℝ² and ℝ³
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ℝ²: Independent vectors
ax1 = axes[0]
v1 = np.array([1, 0])
v2 = np.array([0.5, 1])
ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', label='v1')
ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v2')
ax1.set_xlim(-0.5, 2)
ax1.set_ylim(-0.5, 1.5)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.legend()
ax1.set_title('Independent: Span = ℝ²')
ax1.grid(True, alpha=0.3)

# ℝ²: Dependent vectors (collinear)
ax2 = axes[1]
v1 = np.array([1, 2])
v2 = np.array([2, 4])
ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', label='v1')
ax2.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v2=2v1')
ax2.set_xlim(-0.5, 3)
ax2.set_ylim(-0.5, 5)
ax2.set_aspect('equal')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.legend()
ax2.set_title('Dependent: Span = line')
ax2.grid(True, alpha=0.3)

# Determinant visualization
ax3 = axes[2]
thetas = np.linspace(0, 2*np.pi, 100)
v1 = np.array([1, 0])
for theta in np.linspace(0, np.pi, 6):
    v2 = np.array([np.cos(theta), np.sin(theta)])
    A = np.column_stack([v1, v2])
    d = det(A)
    ax3.scatter(theta, d, color='blue', s=50)
ax3.axhline(y=0, color='r', linestyle='--', label='det=0 → dependent')
ax3.set_xlabel('Angle between v1 and v2')
ax3.set_ylabel('Determinant')
ax3.set_title('Det vs Angle (det=0 means dependent)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('independence_visualization.png', dpi=150)
plt.show()

# ============================================
# Lab 3: Parameter Values for Dependence
# ============================================

def find_dependence_values(A_func, param_range=(-5, 5), num_points=1000):
    """
    Find parameter values that make vectors dependent.
    A_func(c) returns matrix whose columns are the vectors.
    """
    params = np.linspace(param_range[0], param_range[1], num_points)
    dependent_params = []
    
    for c in params:
        A = A_func(c)
        rank = np.linalg.matrix_rank(A, tol=1e-10)
        if rank < A.shape[1]:
            dependent_params.append(c)
    
    return dependent_params

# Example: (1, c, 0), (0, 1, c), (c, 0, 1)
def A_func(c):
    return np.array([
        [1, 0, c],
        [c, 1, 0],
        [0, c, 1]
    ])

params = np.linspace(-3, 3, 1000)
dets = [det(A_func(c)) for c in params]

plt.figure(figsize=(10, 5))
plt.plot(params, dets, 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Parameter c')
plt.ylabel('Determinant')
plt.title('Finding c where vectors are dependent (det = 0)')
plt.grid(True, alpha=0.3)

# Find zeros numerically
from scipy.optimize import brentq
zeros = []
for i in range(len(params)-1):
    if dets[i] * dets[i+1] < 0:
        zero = brentq(lambda c: det(A_func(c)), params[i], params[i+1])
        zeros.append(zero)
        plt.scatter(zero, 0, color='red', s=100, zorder=5)
        plt.annotate(f'c = {zero:.3f}', (zero, 0.5), fontsize=10)

print(f"\nVectors dependent when c ≈ {zeros}")
plt.savefig('dependence_parameter.png', dpi=150)
plt.show()

# ============================================
# Lab 4: Wronskian for Functions
# ============================================

def wronskian(funcs, x):
    """
    Compute Wronskian of functions at point x.
    funcs: list of (function, derivative, ...) tuples
    """
    n = len(funcs)
    W = np.zeros((n, n))
    for i, f in enumerate(funcs):
        for j in range(n):
            W[j, i] = f[j](x)  # j-th derivative of i-th function
    return det(W)

# e^x, e^2x: functions and their derivatives
f1 = [lambda x: np.exp(x), lambda x: np.exp(x)]  # e^x and its derivative
f2 = [lambda x: np.exp(2*x), lambda x: 2*np.exp(2*x)]  # e^2x and its derivative

x_vals = np.linspace(-1, 2, 100)
wron_vals = [wronskian([f1, f2], x) for x in x_vals]

plt.figure(figsize=(10, 5))
plt.plot(x_vals, wron_vals, 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', label='W=0 would mean dependent')
plt.xlabel('x')
plt.ylabel('Wronskian W(e^x, e^{2x})')
plt.title('Wronskian is always nonzero → Functions are independent')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wronskian.png', dpi=150)
plt.show()

# ============================================
# Lab 5: QM - Orthogonal States are Independent
# ============================================

# Create orthogonal states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Verify orthogonality
print("\nQuantum States:")
print(f"|0⟩ = {ket_0}")
print(f"|1⟩ = {ket_1}")
print(f"⟨0|1⟩ = {np.vdot(ket_0, ket_1)}")  # Should be 0

# Check independence
ind, rank, _ = test_independence([ket_0, ket_1])
print(f"Linearly independent: {ind}")

# Non-orthogonal but still independent
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_right = (ket_0 + 1j*ket_1) / np.sqrt(2)

print(f"\n|+⟩ = {ket_plus}")
print(f"|R⟩ = {ket_right}")
print(f"⟨+|R⟩ = {np.vdot(ket_plus, ket_right)}")  # Not zero!

ind, rank, _ = test_independence([ket_plus.real, ket_right.real])
print(f"Independent (real parts): {ind}")

# ============================================
# Lab 6: Maximum Independent Set Size
# ============================================

def max_independent_set_size(n, num_vectors=10, num_trials=100):
    """
    Experimentally verify max independent set size in ℝⁿ.
    """
    max_sizes = []
    
    for _ in range(num_trials):
        vectors = [np.random.randn(n) for _ in range(num_vectors)]
        
        # Find largest independent subset (greedy)
        independent = [vectors[0]]
        for v in vectors[1:]:
            test_set = independent + [v]
            ind, rank, _ = test_independence(test_set)
            if ind:
                independent.append(v)
        
        max_sizes.append(len(independent))
    
    return max_sizes

for n in [2, 3, 4, 5]:
    sizes = max_independent_set_size(n, num_vectors=n+3, num_trials=50)
    print(f"ℝ^{n}: Max independent set size always = {max(sizes)} (theory: {n})")
```

---

## ✅ Daily Checklist

- [ ] Read Axler Section 2.A (independence portion)
- [ ] Memorize: Independence definition, equivalent characterizations
- [ ] Learn matrix method for testing independence
- [ ] Complete Level 1-2 problems
- [ ] Attempt Level 3 proofs
- [ ] Complete computational lab
- [ ] Create flashcards for:
  - Linear independence definition
  - Linear dependence lemma
  - Connection to unique representation
- [ ] Write QM connection in study journal

---

## 📓 Reflection Questions

1. Why is "no nontrivial linear combination equals zero" equivalent to "no vector is in the span of the others"?

2. Why can't you have more than $n$ linearly independent vectors in ℝⁿ?

3. How does linear independence relate to distinguishability in quantum mechanics?

4. What's the geometric meaning of linear independence?

---

## 🔜 Preview: Tomorrow's Topics

**Day 89: Bases and Dimension**

Tomorrow we unite span and independence:
- Definition of basis
- Every basis has the same size (dimension)
- Standard bases
- How to find bases for subspaces

**Preparation:** A basis should span the space AND be linearly independent. Think about why both conditions are needed.

---

*"Linear independence is the mathematician's guarantee that every vector carries unique information."*
