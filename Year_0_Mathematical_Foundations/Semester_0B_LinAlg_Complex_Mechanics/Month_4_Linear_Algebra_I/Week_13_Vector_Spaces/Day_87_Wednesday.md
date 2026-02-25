# Day 87: Span and Spanning Sets

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Span and Spanning Sets |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define the span of a set of vectors
2. Determine if a set of vectors spans a given space
3. Find the span of specific vector sets in ℝⁿ and ℂⁿ
4. Understand span as the "smallest subspace containing" a set
5. Connect span to systems of linear equations
6. Relate span to superposition in quantum mechanics

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 2.A**: Span and Linear Independence (pp. 27-33, span portion)
- Focus on: Definition of span, examples, connection to subspaces

### Secondary Text: Strang, "Introduction to Linear Algebra"
- **Section 3.2**: The Nullspace of A: Solving Ax = 0
- **Section 3.3**: The Complete Solution to Ax = b

---

## 🎬 Video Resources

### 3Blue1Brown: Essence of Linear Algebra
- **Chapter 2**: Linear combinations, span, and basis vectors
- URL: https://www.youtube.com/watch?v=k7RM-ot2NWY
- Duration: 10 minutes
- **Essential viewing** — beautiful visualization of span

### MIT OCW 18.06
- **Lecture 6**: Column Space and Nullspace
- Focus on column space = span of column vectors

---

## 📖 Core Content: Theory and Concepts

### 1. Definition of Span

**Definition:** The **span** of a list of vectors $v_1, \ldots, v_n$ in a vector space $V$ is the set of all linear combinations of these vectors:

$$\text{span}(v_1, \ldots, v_n) = \{a_1 v_1 + a_2 v_2 + \cdots + a_n v_n : a_1, \ldots, a_n \in \mathbb{F}\}$$

**Notation variants:**
- span{v₁, ..., vₙ}
- span(v₁, ..., vₙ)
- ⟨v₁, ..., vₙ⟩

**Convention:** span(∅) = span of empty list = {0}

### 2. Span is a Subspace

**Theorem:** For any vectors $v_1, \ldots, v_n \in V$, the span $\text{span}(v_1, \ldots, v_n)$ is a subspace of $V$.

**Proof:**
1. **Zero vector:** $0 = 0v_1 + 0v_2 + \cdots + 0v_n \in \text{span}(v_1, \ldots, v_n)$ ✓

2. **Closed under addition:** Let $u = a_1v_1 + \cdots + a_nv_n$ and $w = b_1v_1 + \cdots + b_nv_n$.
   Then $u + w = (a_1 + b_1)v_1 + \cdots + (a_n + b_n)v_n \in \text{span}$ ✓

3. **Closed under scalar multiplication:** Let $u = a_1v_1 + \cdots + a_nv_n$ and $c \in \mathbb{F}$.
   Then $cu = (ca_1)v_1 + \cdots + (ca_n)v_n \in \text{span}$ ✓

**Corollary:** Span is the **smallest subspace** containing $\{v_1, \ldots, v_n\}$.

### 3. Fundamental Examples

#### Example 1: Span in ℝ²

Let $v_1 = (1, 0)$ and $v_2 = (0, 1)$.

$\text{span}(v_1, v_2) = \{a(1,0) + b(0,1) : a, b \in \mathbb{R}\} = \{(a, b) : a, b \in \mathbb{R}\} = \mathbb{R}^2$

**Conclusion:** These two vectors span all of ℝ².

#### Example 2: Span of a single vector

Let $v = (1, 2, 3) \in \mathbb{R}^3$.

$\text{span}(v) = \{a(1, 2, 3) : a \in \mathbb{R}\} = \{(a, 2a, 3a) : a \in \mathbb{R}\}$

**Geometrically:** This is a line through the origin in ℝ³.

#### Example 3: Span might be smaller than expected

Let $v_1 = (1, 2)$ and $v_2 = (2, 4)$.

Note: $v_2 = 2v_1$!

$\text{span}(v_1, v_2) = \text{span}(v_1) = \{(a, 2a) : a \in \mathbb{R}\}$

**Geometrically:** Just a line, not all of ℝ².

#### Example 4: Span in ℝ³

Let $v_1 = (1, 0, 1)$, $v_2 = (0, 1, 1)$, $v_3 = (1, 1, 2)$.

Note: $v_3 = v_1 + v_2$!

$\text{span}(v_1, v_2, v_3) = \text{span}(v_1, v_2)$

This is a plane through the origin (not all of ℝ³).

### 4. Spanning Sets

**Definition:** A list of vectors $v_1, \ldots, v_n$ **spans** $V$ if $\text{span}(v_1, \ldots, v_n) = V$.

Equivalently: Every vector in $V$ can be written as a linear combination of $v_1, \ldots, v_n$.

**Example:** The standard basis vectors $e_1 = (1, 0, \ldots, 0)$, ..., $e_n = (0, \ldots, 0, 1)$ span ℝⁿ.

### 5. Testing if a Vector is in a Span

**Problem:** Given vectors $v_1, \ldots, v_k$ and a target vector $w$, is $w \in \text{span}(v_1, \ldots, v_k)$?

**Method:** $w \in \text{span}(v_1, \ldots, v_k)$ if and only if the system

$$a_1 v_1 + a_2 v_2 + \cdots + a_k v_k = w$$

has a solution (find coefficients $a_1, \ldots, a_k$).

**In matrix form:** Let $A = [v_1 | v_2 | \cdots | v_k]$ (vectors as columns).
Then $w \in \text{span}$ iff $Ax = w$ has a solution.

### 6. The Column Space

**Definition:** The **column space** of a matrix $A$ is the span of its column vectors:
$$\text{col}(A) = \text{span}(\text{columns of } A)$$

**Key Insight:** The equation $Ax = b$ has a solution iff $b \in \text{col}(A)$.

### 7. When Does a Set Span ℝⁿ?

**Theorem:** Vectors $v_1, \ldots, v_k$ span ℝⁿ if and only if the matrix $A = [v_1 | \cdots | v_k]$ has a pivot in every row (when in reduced row echelon form).

**Consequence:** You need at least $n$ vectors to span ℝⁿ.

### 8. Polynomial Spaces

**Example:** In $\mathcal{P}_2(\mathbb{R})$ (polynomials of degree ≤ 2):

$\text{span}(1, x, x^2) = \mathcal{P}_2(\mathbb{R})$ (spans the whole space)

$\text{span}(1, x) = \mathcal{P}_1(\mathbb{R})$ (only degree ≤ 1 polynomials)

$\text{span}(1 + x, 1 - x) = \text{span}(1, x) = \mathcal{P}_1(\mathbb{R})$

(The last equality: $(1+x) + (1-x) = 2$, so we can make constant; $(1+x) - (1-x) = 2x$, so we can make $x$.)

---

## 🔬 Quantum Mechanics Connection

### Superposition and Span

**The superposition principle says quantum states form a vector space.** But physically, we often work with specific spanning sets:

**1. Complete Sets of States**

A **complete set of states** in quantum mechanics is a spanning set for the Hilbert space.

For a qubit: $\{|0\rangle, |1\rangle\}$ spans ℂ²
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

Any qubit state is in span($|0\rangle, |1\rangle$).

**2. Energy Eigenstates**

For a quantum system with Hamiltonian $\hat{H}$, the energy eigenstates often span the Hilbert space:
$$|\psi\rangle = \sum_n c_n |E_n\rangle$$

**3. Position and Momentum Bases**

In continuous systems:
- Position eigenstates $|x\rangle$ span the Hilbert space
- Momentum eigenstates $|p\rangle$ also span the Hilbert space

Any state: $|\psi\rangle = \int \psi(x) |x\rangle \, dx$

**4. Reachable States**

Given a set of quantum gates, the **reachable states** from $|0\rangle$ are:
$$\text{span of all } U_k \cdots U_1 |0\rangle$$

This is fundamentally about span!

### Example: Spin-1/2 Particle

A spin-1/2 particle (like an electron) has a 2-dimensional Hilbert space.

The states $|\uparrow\rangle$ and $|\downarrow\rangle$ (spin up and spin down along z-axis) span ℂ².

Any spin state:
$$|\psi\rangle = \alpha|\uparrow\rangle + \beta|\downarrow\rangle$$

But we could also use $|+\rangle = \frac{1}{\sqrt{2}}(|\uparrow\rangle + |\downarrow\rangle)$ and $|-\rangle = \frac{1}{\sqrt{2}}(|\uparrow\rangle - |\downarrow\rangle)$:

These also span ℂ²! (Different basis, same span)

---

## ✏️ Worked Examples

### Example 1: Computing Span in ℝ³

**Question:** Describe span$((1, 0, 1), (0, 1, 1))$ geometrically.

**Solution:**
General element: $a(1, 0, 1) + b(0, 1, 1) = (a, b, a + b)$

Let $x = a$, $y = b$, $z = a + b$. Then $z = x + y$.

**Span is the plane $z = x + y$** (or equivalently, $x + y - z = 0$).

**Verification:** This plane passes through origin ✓ and is 2-dimensional.

### Example 2: Is a vector in the span?

**Question:** Is $(3, 7, 2)$ in span$((1, 2, 1), (1, 1, 0))$?

**Solution:**
Need to solve: $a(1, 2, 1) + b(1, 1, 0) = (3, 7, 2)$

System:
$$a + b = 3$$
$$2a + b = 7$$
$$a = 2$$

From equation 3: $a = 2$
From equation 1: $2 + b = 3 \Rightarrow b = 1$
Check equation 2: $2(2) + 1 = 5 \neq 7$ ✗

**No solution exists!** $(3, 7, 2) \notin \text{span}((1, 2, 1), (1, 1, 0))$.

### Example 3: Finding a spanning set

**Question:** Find a spanning set for $U = \{(x, y, z) \in \mathbb{R}^3 : x + y + z = 0\}$.

**Solution:**
From $x + y + z = 0$: $z = -x - y$

General element: $(x, y, -x-y) = x(1, 0, -1) + y(0, 1, -1)$

**Spanning set:** $\{(1, 0, -1), (0, 1, -1)\}$

Check: span$((1, 0, -1), (0, 1, -1)) = U$ ✓

### Example 4: Do vectors span ℝ³?

**Question:** Do $(1, 2, 3), (4, 5, 6), (7, 8, 9)$ span ℝ³?

**Solution:**
Form matrix and row reduce:
$$A = \begin{pmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{pmatrix}$$

Row reduce:
$$\begin{pmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{pmatrix} \xrightarrow{R_2 - 2R_1, R_3 - 3R_1} \begin{pmatrix} 1 & 4 & 7 \\ 0 & -3 & -6 \\ 0 & -6 & -12 \end{pmatrix} \xrightarrow{R_3 - 2R_2} \begin{pmatrix} 1 & 4 & 7 \\ 0 & -3 & -6 \\ 0 & 0 & 0 \end{pmatrix}$$

Only 2 pivots! Need 3 pivots to span ℝ³.

**No, they don't span ℝ³.** They span a plane.

### Example 5: Polynomial span

**Question:** Does $\{1 + x, x + x^2, 1 + x^2\}$ span $\mathcal{P}_2(\mathbb{R})$?

**Solution:**
Represent polynomials as vectors: $ax^2 + bx + c \leftrightarrow (a, b, c)$

- $1 + x \leftrightarrow (0, 1, 1)$
- $x + x^2 \leftrightarrow (1, 1, 0)$
- $1 + x^2 \leftrightarrow (1, 0, 1)$

Check if these span ℝ³:
$$\begin{pmatrix} 0 & 1 & 1 \\ 1 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix}$$

Determinant = $0(1-0) - 1(1-0) + 1(0-1) = 0 - 1 - 1 = -2 \neq 0$

Three pivots! **Yes, they span $\mathcal{P}_2(\mathbb{R})$.**

---

## 📝 Practice Problems

### Level 1: Basic Span
1. Describe span$((2, 1))$ in ℝ² geometrically.

2. Find span$((1, 0, 0), (0, 1, 0))$ in ℝ³.

3. Is $(4, 6)$ in span$((1, 2), (3, 4))$?

### Level 2: Span Calculations
4. Is $(1, 1, 1)$ in span$((1, 2, 3), (4, 5, 6))$? If yes, find the coefficients.

5. Find a spanning set for $\{(x, y, z, w) : x + y = 0 \text{ and } z + w = 0\}$.

6. Do $(1, 1, 0), (0, 1, 1), (1, 0, 1)$ span ℝ³?

### Level 3: Conceptual
7. Prove: If $v \in \text{span}(v_1, \ldots, v_k)$, then $\text{span}(v_1, \ldots, v_k, v) = \text{span}(v_1, \ldots, v_k)$.

8. Prove: span$(v_1, \ldots, v_n)$ is the intersection of all subspaces containing $\{v_1, \ldots, v_n\}$.

9. If $U$ and $W$ are subspaces, prove span$(U \cup W) = U + W$ where $U + W = \{u + w : u \in U, w \in W\}$.

### Level 4: Challenge
10. Let $V = C^1(\mathbb{R})$ (continuously differentiable functions). Consider:
    $f_1(x) = e^x$, $f_2(x) = e^{2x}$, $f_3(x) = e^{3x}$
    
    Can $e^{4x}$ be written as a linear combination of $f_1, f_2, f_3$? (Hint: differentiate)

11. In ℂ², consider $v_1 = (1, i)$ and $v_2 = (i, -1)$. Do they span ℂ² over ℂ?

12. Find all values of $c$ such that $(1, c, c^2)$ is in span$((1, 1, 1), (1, 2, 4))$.

---

## 📊 Answers and Hints

1. Line through origin with slope 1/2
2. The xy-plane: $\{(x, y, 0) : x, y \in \mathbb{R}\}$
3. Yes: $-\frac{1}{2}(1,2) + \frac{3}{2}(3,4) = (4, 6)$
4. No — solve system and show inconsistency
5. $\{(1, -1, 0, 0), (0, 0, 1, -1)\}$
6. Yes — check determinant or row reduce
7. If $v = a_1v_1 + \cdots + a_kv_k$, any linear combination including $v$ can be rewritten without $v$
8. Show span is contained in every such subspace, and is itself a subspace containing the vectors
9. Show inclusion both ways
10. No — if $e^{4x} = ae^x + be^{2x} + ce^{3x}$, differentiate and evaluate at $x=0$ for contradictions
11. Check: Is $(1, i)$ a scalar multiple of $(i, -1)$? $\alpha(1, i) = (i, -1)$ means $\alpha = i$ and $\alpha i = -1$ ✓. So $v_2 = iv_1$! They span only a 1D subspace over ℂ.
12. Solve: $(1, c, c^2) = a(1, 1, 1) + b(1, 2, 4)$. Get $c = 1, 2$.

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# Lab 1: Visualizing Span in ℝ² and ℝ³
# ============================================

def visualize_span_2d(vectors, ax=None):
    """Visualize span of vectors in ℝ²."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot vectors
    origin = np.zeros(2)
    colors = ['red', 'blue', 'green', 'purple']
    
    for i, v in enumerate(vectors):
        ax.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, 
                  color=colors[i % len(colors)], width=0.02, label=f'v{i+1}={v}')
    
    # If 2 vectors, show span (parallelogram region)
    if len(vectors) == 2:
        v1, v2 = vectors[0], vectors[1]
        # Generate points in the span
        t = np.linspace(-2, 2, 100)
        s = np.linspace(-2, 2, 100)
        T, S = np.meshgrid(t, s)
        X = T * v1[0] + S * v2[0]
        Y = T * v1[1] + S * v2[1]
        ax.scatter(X.flatten(), Y.flatten(), alpha=0.1, s=1, color='gray')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Span Visualization in ℝ²')
    return ax

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Case 1: Two linearly independent vectors span ℝ²
visualize_span_2d([np.array([1, 0]), np.array([0, 1])], axes[0])
axes[0].set_title('span{(1,0), (0,1)} = ℝ²')

# Case 2: Two dependent vectors span a line
visualize_span_2d([np.array([1, 2]), np.array([2, 4])], axes[1])
axes[1].set_title('span{(1,2), (2,4)} = line')

# Case 3: Two general vectors
visualize_span_2d([np.array([1, 1]), np.array([1, -1])], axes[2])
axes[2].set_title('span{(1,1), (1,-1)} = ℝ²')

plt.tight_layout()
plt.savefig('span_2d_visualization.png', dpi=150)
plt.show()

# ============================================
# Lab 2: Span in ℝ³
# ============================================

fig = plt.figure(figsize=(15, 5))

# Span of 1 vector = line
ax1 = fig.add_subplot(131, projection='3d')
v = np.array([1, 2, 1])
t = np.linspace(-2, 2, 100)
line = np.outer(t, v)
ax1.plot(line[:, 0], line[:, 1], line[:, 2], 'b-', linewidth=2)
ax1.quiver(0, 0, 0, v[0], v[1], v[2], color='red', arrow_length_ratio=0.1)
ax1.set_title('span{(1,2,1)} = line')

# Span of 2 independent vectors = plane
ax2 = fig.add_subplot(132, projection='3d')
v1 = np.array([1, 0, 1])
v2 = np.array([0, 1, 1])
t = np.linspace(-2, 2, 20)
s = np.linspace(-2, 2, 20)
T, S = np.meshgrid(t, s)
X = T * v1[0] + S * v2[0]
Y = T * v1[1] + S * v2[1]
Z = T * v1[2] + S * v2[2]
ax2.plot_surface(X, Y, Z, alpha=0.5, color='cyan')
ax2.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='red', arrow_length_ratio=0.1)
ax2.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='blue', arrow_length_ratio=0.1)
ax2.set_title('span{(1,0,1), (0,1,1)} = plane')

# Span of 3 independent vectors = ℝ³
ax3 = fig.add_subplot(133, projection='3d')
ax3.text(0, 0, 0, 'ℝ³', fontsize=20, ha='center')
ax3.quiver(0, 0, 0, 1, 0, 0, color='red', arrow_length_ratio=0.1)
ax3.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1)
ax3.quiver(0, 0, 0, 0, 0, 1, color='blue', arrow_length_ratio=0.1)
ax3.set_title('span{e₁, e₂, e₃} = ℝ³')

for ax in [ax1, ax2, ax3]:
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig('span_3d_visualization.png', dpi=150)
plt.show()

# ============================================
# Lab 3: Checking if Vector is in Span
# ============================================

def is_in_span(target, vectors):
    """
    Check if target is in span of vectors.
    Returns (True, coefficients) or (False, None).
    """
    A = np.column_stack(vectors)
    
    try:
        # Use least squares (works for overdetermined systems too)
        coeffs, residuals, rank, s = np.linalg.lstsq(A, target, rcond=None)
        
        # Check if solution is exact
        if np.allclose(A @ coeffs, target):
            return True, coeffs
        else:
            return False, None
    except:
        return False, None

# Test cases
v1 = np.array([1, 2, 1])
v2 = np.array([1, 1, 0])

target1 = np.array([3, 7, 2])  # NOT in span
target2 = np.array([2, 3, 1])  # IN span: v1 + v2

print("Span membership tests:")
result, coeffs = is_in_span(target1, [v1, v2])
print(f"(3,7,2) in span: {result}")

result, coeffs = is_in_span(target2, [v1, v2])
print(f"(2,3,1) in span: {result}, coeffs: {coeffs}")
if result:
    print(f"Verification: {coeffs[0]}*{v1} + {coeffs[1]}*{v2} = {coeffs[0]*v1 + coeffs[1]*v2}")

# ============================================
# Lab 4: Span Dimension via Rank
# ============================================

def span_dimension(vectors):
    """Compute dimension of span using matrix rank."""
    A = np.column_stack(vectors)
    return np.linalg.matrix_rank(A)

# Examples
print("\nSpan dimensions:")

vectors1 = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
print(f"span{{(1,2,3), (4,5,6), (7,8,9)}}: dim = {span_dimension(vectors1)}")

vectors2 = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
print(f"span{{e1, e2, e3}}: dim = {span_dimension(vectors2)}")

vectors3 = [np.array([1, 1, 0]), np.array([0, 1, 1]), np.array([1, 0, 1])]
print(f"span{{(1,1,0), (0,1,1), (1,0,1)}}: dim = {span_dimension(vectors3)}")

# ============================================
# Lab 5: QM Application - Qubit States
# ============================================

# Standard basis for qubit
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# span{|0⟩, |1⟩} = ℂ²
print("\nQuantum Computing: Qubit State Space")
print("Standard basis spans ℂ²:")
print(f"  |0⟩ = {ket_0}")
print(f"  |1⟩ = {ket_1}")

# Any qubit state
alpha = (1 + 1j) / np.sqrt(3)
beta = 1 / np.sqrt(3)
psi = alpha * ket_0 + beta * ket_1
print(f"\nGeneral state |ψ⟩ = {alpha:.3f}|0⟩ + {beta:.3f}|1⟩")
print(f"  = {psi}")
print(f"Norm: {np.linalg.norm(psi):.4f}")

# Hadamard basis also spans ℂ²
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

print("\nHadamard basis also spans ℂ²:")
print(f"  |+⟩ = {ket_plus}")
print(f"  |-⟩ = {ket_minus}")

# Express |0⟩ in Hadamard basis
# |0⟩ = (|+⟩ + |-⟩)/√2
recon_0 = (ket_plus + ket_minus) / np.sqrt(2)
print(f"\n|0⟩ in Hadamard basis: (|+⟩ + |-⟩)/√2 = {recon_0}")
print(f"Matches |0⟩: {np.allclose(ket_0, recon_0)}")

# ============================================
# Lab 6: Column Space = Span of Columns
# ============================================

A = np.array([
    [1, 4, 7],
    [2, 5, 8],
    [3, 6, 9]
])

print("\nColumn Space Analysis")
print("Matrix A:")
print(A)

print(f"\nRank of A: {np.linalg.matrix_rank(A)}")
print("This means col(A) is a 2-dimensional subspace of ℝ³")

# Check: Is b = (1, 1, 1) in col(A)?
b = np.array([1, 1, 1])
result, coeffs = is_in_span(b, [A[:, 0], A[:, 1], A[:, 2]])
print(f"\nIs (1,1,1) in col(A)? {result}")

# Check: Is b = (1, 2, 3) in col(A)?
b = np.array([1, 2, 3])
result, coeffs = is_in_span(b, [A[:, 0], A[:, 1], A[:, 2]])
print(f"Is (1,2,3) in col(A)? {result}")
if result:
    print(f"  (1,2,3) = {coeffs[0]:.2f}*col1 + {coeffs[1]:.2f}*col2 + {coeffs[2]:.2f}*col3")
```

---

## ✅ Daily Checklist

- [ ] Read Axler Section 2.A (span portion)
- [ ] Watch 3Blue1Brown Chapter 2 on span
- [ ] Memorize: span definition, span is a subspace
- [ ] Complete Level 1-2 problems
- [ ] Attempt at least one Level 3 problem
- [ ] Complete computational lab
- [ ] Create flashcards for:
  - Span definition
  - Spanning set definition
  - Connection to linear systems
- [ ] Write QM connection in study journal

---

## 📓 Reflection Questions

1. How is span related to "what vectors can I reach"?

2. Why is span(v₁, ..., vₙ) always a subspace?

3. In quantum mechanics, what does it mean for a set of states to span the Hilbert space?

4. How does adding more vectors affect the span?

---

## 🔜 Preview: Tomorrow's Topics

**Day 88: Linear Independence**

Tomorrow we'll answer: Given vectors, are any of them "redundant"?
- Definition of linear independence
- Testing for linear independence
- Connection to uniqueness of representations
- Why independence matters for bases

**Preparation:** Think about when a vector "adds nothing new" to a span.

---

*"The span of a set of vectors is the smallest stage on which they can all perform."*
