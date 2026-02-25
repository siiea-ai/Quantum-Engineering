# Day 86: Subspaces and Linear Combinations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Subspaces |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Linear Combinations & Problems |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Exploration |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define subspace and state the subspace test
2. Apply the subspace test to verify subspaces
3. Identify common subspaces in ℝⁿ, ℂⁿ, and function spaces
4. Compute linear combinations of vectors
5. Understand the connection between subspaces and solution sets
6. Connect subspaces to quantum mechanical constraints

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 1.C**: Subspaces (pp. 15-22)
- Focus on: Definition 1.32, Theorem on subspace conditions

### Secondary Text: Strang, "Introduction to Linear Algebra"
- **Section 3.1**: Spaces of Vectors (pp. 123-134)
- Focus on: Column space, null space previews

---

## 🎬 Video Resources

### MIT OCW 18.06
- **Lecture 5**: Transposes, Permutations, Vector Spaces (parts on subspaces)
- **Lecture 6**: Column Space and Nullspace

### 3Blue1Brown
- Review: Chapter 2 on span (prepares for tomorrow)

---

## 📖 Core Content: Theory and Concepts

### 1. Subspace Definition

**Definition:** A subset U of a vector space V is called a **subspace** of V if U is itself a vector space under the same operations (using the same addition and scalar multiplication from V).

**Intuition:** A subspace is a "vector space living inside another vector space."

### 2. The Subspace Test (Crucial!)

**Theorem:** A subset U ⊆ V is a subspace if and only if:
1. **0 ∈ U** (contains the zero vector)
2. **u, w ∈ U ⟹ u + w ∈ U** (closed under addition)
3. **c ∈ 𝔽, u ∈ U ⟹ cu ∈ U** (closed under scalar multiplication)

**Why this works:** The remaining axioms (commutativity, associativity, etc.) are inherited from V.

**Equivalently (combined test):**
U is a subspace ⟺ U is nonempty and closed under linear combinations:
$$u, w \in U, \, a, b \in \mathbb{F} \implies au + bw \in U$$

### 3. Fundamental Examples of Subspaces

#### Example 1: Lines through the origin in ℝ²

$U = \{(x, y) \in \mathbb{R}^2 : y = mx\}$ for some fixed slope $m$.

**Verification:**
- **Zero?** $(0, 0)$ satisfies $0 = m \cdot 0$ ✓
- **Addition?** If $(x_1, mx_1)$ and $(x_2, mx_2)$ are in U, then
  $(x_1 + x_2, mx_1 + mx_2) = (x_1 + x_2, m(x_1 + x_2)) \in U$ ✓
- **Scalar mult?** $c(x, mx) = (cx, cmx) = (cx, m(cx)) \in U$ ✓

**Geometrically:** Any line through the origin in ℝ² is a 1-dimensional subspace.

#### Example 2: Planes through the origin in ℝ³

$U = \{(x, y, z) \in \mathbb{R}^3 : ax + by + cz = 0\}$

This is a 2-dimensional subspace (as we'll quantify later with "dimension").

#### Example 3: NOT a subspace — Line not through origin

$W = \{(x, y) : y = 2x + 1\}$

**Fails:** $(0, 1) \in W$ but the zero vector $(0, 0)$ is not in $W$.

Also fails closure: $(1, 3), (2, 5) \in W$, but $(1, 3) + (2, 5) = (3, 8) \notin W$ since $8 \neq 2(3) + 1 = 7$.

#### Example 4: Solution set of homogeneous system

$U = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$ (null space of matrix A)

**Verification:**
- **Zero?** $A\mathbf{0} = \mathbf{0}$ ✓
- **Addition?** If $A\mathbf{x} = \mathbf{0}$ and $A\mathbf{y} = \mathbf{0}$, then
  $A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y} = \mathbf{0} + \mathbf{0} = \mathbf{0}$ ✓
- **Scalar?** $A(c\mathbf{x}) = cA\mathbf{x} = c\mathbf{0} = \mathbf{0}$ ✓

**Critical Insight:** Solution sets of homogeneous linear equations are always subspaces!

#### Example 5: NOT a subspace — Non-homogeneous solutions

$W = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{b}\}$ where $\mathbf{b} \neq \mathbf{0}$

**Fails:** Zero vector not in $W$ (since $A\mathbf{0} = \mathbf{0} \neq \mathbf{b}$).

#### Example 6: Polynomial subspace

$\mathcal{P}_n(\mathbb{F}) = \{\text{polynomials of degree} \leq n\}$ is a subspace of $\mathcal{P}(\mathbb{F})$.

#### Example 7: Continuous functions as subspace

$C(\mathbb{R}) = \{\text{continuous functions } f: \mathbb{R} \to \mathbb{R}\}$

is a subspace of $\mathcal{F}(\mathbb{R}, \mathbb{R}) = \{$all functions $\mathbb{R} \to \mathbb{R}\}$.

### 4. Linear Combinations

**Definition:** A **linear combination** of vectors $v_1, \ldots, v_n$ in a vector space V is any expression of the form:

$$a_1 v_1 + a_2 v_2 + \cdots + a_n v_n$$

where $a_1, \ldots, a_n \in \mathbb{F}$ are scalars (called **coefficients**).

### 5. The Trivial and Non-Trivial Linear Combinations

- **Trivial linear combination:** All coefficients are zero
  $$0 \cdot v_1 + 0 \cdot v_2 + \cdots + 0 \cdot v_n = \mathbf{0}$$

- **Non-trivial linear combination:** At least one coefficient is nonzero

### 6. Important Observations

**Observation 1:** Every subspace contains the zero vector.
(The trivial linear combination of any vectors gives 0.)

**Observation 2:** The intersection of subspaces is a subspace.

**Proof:** Let U, W be subspaces of V. Let I = U ∩ W.
- **Zero?** 0 ∈ U and 0 ∈ W, so 0 ∈ I ✓
- **Addition?** If u, v ∈ I, then u, v ∈ U ⟹ u + v ∈ U (U is subspace)
  and u, v ∈ W ⟹ u + v ∈ W (W is subspace).
  Thus u + v ∈ U ∩ W = I ✓
- **Scalar?** Similar argument ✓

**Observation 3:** The union of two subspaces is generally NOT a subspace!

**Example:** In ℝ², let U = {(x, 0)} and W = {(0, y)}.
Both are subspaces (the x and y axes).
But U ∪ W is not a subspace: (1, 0) ∈ U, (0, 1) ∈ W, but (1, 0) + (0, 1) = (1, 1) ∉ U ∪ W.

### 7. The Smallest Subspace Containing a Set

Given any subset S of V, there is a **smallest subspace** containing S:
- It's the intersection of all subspaces containing S
- It equals the set of all linear combinations of elements of S

This motivates tomorrow's topic: **span**.

---

## 🔬 Quantum Mechanics Connection

### Constraint Subspaces in Quantum Mechanics

In quantum mechanics, physical constraints often define subspaces:

**1. Normalization Constraint?**
The set of normalized states $\{|\psi\rangle : \langle\psi|\psi\rangle = 1\}$ is NOT a subspace!
- Not closed under addition: If $\langle\psi|\psi\rangle = 1$ and $\langle\phi|\phi\rangle = 1$,
  then $|\psi\rangle + |\phi\rangle$ typically does NOT have norm 1.
- The zero vector is not in this set.

**This is why we work with rays (equivalence classes) in projective Hilbert space!**

**2. Symmetry Constraints ARE Subspaces**
States satisfying a symmetry condition often form subspaces:
- States with definite parity (even or odd functions)
- States with definite angular momentum projection $m$

**3. Energy Eigenspaces**
The set of all states with energy E:
$$\mathcal{H}_E = \{|\psi\rangle : \hat{H}|\psi\rangle = E|\psi\rangle\}$$
is a subspace (possibly 1-dimensional or higher for degenerate energies).

**Why?** If $\hat{H}|\psi_1\rangle = E|\psi_1\rangle$ and $\hat{H}|\psi_2\rangle = E|\psi_2\rangle$, then:
$$\hat{H}(\alpha|\psi_1\rangle + \beta|\psi_2\rangle) = \alpha E|\psi_1\rangle + \beta E|\psi_2\rangle = E(\alpha|\psi_1\rangle + \beta|\psi_2\rangle)$$

Linear combinations of eigenstates with the same eigenvalue are eigenstates!

### Superposition Within a Subspace

When working in a subspace (say, the spin states of an electron), superposition still works:
$$|\psi\rangle = \alpha|\uparrow\rangle + \beta|\downarrow\rangle$$

The subspace structure tells us which superpositions are "allowed" by the physics.

---

## ✏️ Worked Examples

### Example 1: Subspace Test in ℝ³

**Question:** Is $U = \{(x, y, z) \in \mathbb{R}^3 : x = 2y\}$ a subspace of ℝ³?

**Solution:**
Let's apply the subspace test:

1. **Zero vector?**
   Is $(0, 0, 0)$ in $U$? Check: $0 = 2(0)$ ✓

2. **Closed under addition?**
   Take $(2a, a, b)$ and $(2c, c, d)$ in $U$.
   Sum: $(2a + 2c, a + c, b + d) = (2(a+c), a+c, b+d)$
   Is $2(a+c) = 2(a+c)$? Yes ✓

3. **Closed under scalar multiplication?**
   Take $k \in \mathbb{R}$ and $(2a, a, b) \in U$.
   $k(2a, a, b) = (2ka, ka, kb)$
   Is $2ka = 2(ka)$? Yes ✓

**Conclusion:** $U$ is a subspace of ℝ³.

**Geometric interpretation:** This is a plane through the origin defined by $x - 2y = 0$.

### Example 2: Not a Subspace

**Question:** Is $W = \{(x, y) \in \mathbb{R}^2 : xy \geq 0\}$ a subspace of ℝ²?

**Solution:**
This is the union of the first and third quadrants (including axes).

**Test zero:** $(0, 0)$ has $0 \cdot 0 = 0 \geq 0$ ✓

**Test addition:**
$(1, 1) \in W$ (both positive)
$(-2, 0) \in W$ (on axis)
$(1, 1) + (-2, 0) = (-1, 1)$

But $(-1)(1) = -1 < 0$, so $(-1, 1) \notin W$ ✗

**Conclusion:** $W$ is NOT a subspace.

### Example 3: Linear Combination Calculation

**Question:** Express $\mathbf{v} = (7, 4)$ as a linear combination of $\mathbf{u}_1 = (1, 2)$ and $\mathbf{u}_2 = (3, -1)$.

**Solution:**
We need to find $a, b$ such that $a(1, 2) + b(3, -1) = (7, 4)$.

System of equations:
$$a + 3b = 7$$
$$2a - b = 4$$

From equation 2: $b = 2a - 4$

Substitute into equation 1: $a + 3(2a - 4) = 7$
$a + 6a - 12 = 7$
$7a = 19$
$a = 19/7$

Then $b = 2(19/7) - 4 = 38/7 - 28/7 = 10/7$

**Verify:** $(19/7)(1, 2) + (10/7)(3, -1) = (19/7 + 30/7, 38/7 - 10/7) = (49/7, 28/7) = (7, 4)$ ✓

### Example 4: Function Space Subspace

**Question:** Is $U = \{f \in C(\mathbb{R}) : f(0) = 0\}$ a subspace of $C(\mathbb{R})$?

**Solution:**
1. **Zero?** The zero function has $f(0) = 0$ ✓
2. **Addition?** If $f(0) = 0$ and $g(0) = 0$, then $(f+g)(0) = f(0) + g(0) = 0$ ✓
3. **Scalar?** If $f(0) = 0$ and $c \in \mathbb{R}$, then $(cf)(0) = c \cdot f(0) = c \cdot 0 = 0$ ✓

**Conclusion:** $U$ is a subspace.

### Example 5: NOT a subspace — Condition at a point

**Question:** Is $W = \{f \in C(\mathbb{R}) : f(0) = 1\}$ a subspace of $C(\mathbb{R})$?

**Solution:**
The zero function has $f(0) = 0 \neq 1$, so $\mathbf{0} \notin W$.

**Conclusion:** $W$ is NOT a subspace (fails condition 1 immediately).

---

## 📝 Practice Problems

### Level 1: Subspace Verification
1. Is $\{(a, b, a+b) : a, b \in \mathbb{R}\}$ a subspace of ℝ³?

2. Is $\{(a, b, c) \in \mathbb{R}^3 : a^2 + b^2 + c^2 \leq 1\}$ a subspace of ℝ³?

3. Is the set of symmetric $n \times n$ matrices a subspace of $M_{n \times n}(\mathbb{R})$?

### Level 2: Linear Combinations
4. Express $(1, 5, 3)$ as a linear combination of $(1, 1, 0)$, $(0, 1, 1)$, $(1, 0, 1)$.

5. Can $(1, 2, 3)$ be written as a linear combination of $(1, 0, 0)$ and $(0, 1, 0)$?

6. Find all ways to write $(6, 3, 9)$ as a linear combination of $(1, 0, 2)$ and $(2, 1, 1)$.

### Level 3: Proofs
7. Prove: The set of differentiable functions on ℝ is a subspace of all functions on ℝ.

8. Prove: The intersection of any collection of subspaces is a subspace.

9. Let $U = \{(x, y, z) : x + y = 0\}$ and $W = \{(x, y, z) : y + z = 0\}$ in ℝ³.
   Find $U \cap W$ and verify it's a subspace.

### Level 4: Challenge
10. Prove or disprove: The set of invertible $n \times n$ matrices is a subspace of $M_{n \times n}(\mathbb{R})$.

11. Let $V$ be a vector space and $U_1, U_2$ be subspaces. Prove that $U_1 \cup U_2$ is a subspace if and only if $U_1 \subseteq U_2$ or $U_2 \subseteq U_1$.

12. Define the **sum** of two subspaces: $U + W = \{u + w : u \in U, w \in W\}$.
    Prove that $U + W$ is a subspace.

---

## 📊 Answers and Hints

1. **Yes** — verify all three conditions
2. **No** — not closed under scalar multiplication (scale by 2)
3. **Yes** — sum of symmetric is symmetric, scalar multiple of symmetric is symmetric
4. $2(1,1,0) + 3(0,1,1) + (-1)(1,0,1) = (1, 5, 3)$
5. **No** — z-component would always be 0
6. Solve system; there may be no solution, one solution, or infinitely many
7. Sum and scalar multiples of differentiable functions are differentiable
8. Generalize the proof for two subspaces
9. $U \cap W = \{(x, -x, x) : x \in \mathbb{R}\}$ — a line
10. **No** — zero matrix not invertible, also not closed under addition
11. Hint: If $U_1 \not\subseteq U_2$ and $U_2 \not\subseteq U_1$, find vectors whose sum isn't in union
12. Show 0 ∈ U + W, closure under addition and scalar multiplication

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Lab 1: Visualizing Subspaces in ℝ³
# ============================================

def is_in_subspace_plane(point, normal):
    """Check if point is in plane through origin with given normal."""
    return np.isclose(np.dot(point, normal), 0)

def is_in_subspace_line(point, direction):
    """Check if point is on line through origin with given direction."""
    # Check if point is scalar multiple of direction
    if np.allclose(direction, 0):
        return np.allclose(point, 0)
    # Find non-zero component of direction
    for i in range(len(direction)):
        if not np.isclose(direction[i], 0):
            t = point[i] / direction[i]
            return np.allclose(point, t * direction)
    return False

# Test subspace membership
plane_normal = np.array([1, 1, 1])  # Plane x + y + z = 0
line_direction = np.array([1, 2, 1])

test_points = [
    np.array([1, -1, 0]),
    np.array([1, 0, -1]),
    np.array([1, 2, 1]),
    np.array([2, 4, 2]),
    np.array([1, 1, 1]),
]

print("Testing subspace membership:")
print("\nPlane x + y + z = 0:")
for p in test_points:
    in_plane = is_in_subspace_plane(p, plane_normal)
    print(f"  {p} in plane: {in_plane}")

print("\nLine through (1, 2, 1):")
for p in test_points:
    on_line = is_in_subspace_line(p, line_direction)
    print(f"  {p} on line: {on_line}")

# ============================================
# Lab 2: Linear Combination Solver
# ============================================

def solve_linear_combination(target, vectors):
    """
    Find coefficients such that sum(c_i * v_i) = target.
    Returns None if no solution exists.
    """
    # Stack vectors as columns of a matrix
    A = np.column_stack(vectors)
    
    try:
        # Solve Ax = target
        coeffs, residuals, rank, s = np.linalg.lstsq(A, target, rcond=None)
        
        # Verify solution
        if np.allclose(A @ coeffs, target):
            return coeffs
        else:
            return None
    except:
        return None

# Example: Express (7, 4) as linear combination of (1, 2) and (3, -1)
v1 = np.array([1, 2])
v2 = np.array([3, -1])
target = np.array([7, 4])

coeffs = solve_linear_combination(target, [v1, v2])
print(f"\n(7, 4) = {coeffs[0]:.4f} * (1, 2) + {coeffs[1]:.4f} * (3, -1)")

# Verify
result = coeffs[0] * v1 + coeffs[1] * v2
print(f"Verification: {result}")

# ============================================
# Lab 3: Visualize Subspaces in ℝ²
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subspace: Line y = 2x
ax1 = axes[0]
t = np.linspace(-2, 2, 100)
x = t
y = 2 * t
ax1.plot(x, y, 'b-', linewidth=2, label='Subspace: y = 2x')
ax1.scatter([0], [0], color='red', s=100, zorder=5, label='Zero vector')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-4, 4)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_title('Subspace: Line through Origin')
ax1.legend()
ax1.grid(True, alpha=0.3)

# NOT a subspace: Line y = x + 1
ax2 = axes[1]
x = t
y = t + 1
ax2.plot(x, y, 'r-', linewidth=2, label='NOT subspace: y = x + 1')
ax2.scatter([0], [0], color='blue', s=100, zorder=5, label='Zero vector (not on line!)')
ax2.scatter([0], [1], color='green', s=100, zorder=5, label='(0, 1) on line')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 4)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_title('NOT a Subspace: Line Not Through Origin')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subspace: Entire plane ℝ²
ax3 = axes[2]
ax3.fill([-2, 2, 2, -2], [-4, -4, 4, 4], alpha=0.3, color='green', label='Subspace: ℝ²')
ax3.scatter([0], [0], color='red', s=100, zorder=5, label='Zero vector')
ax3.set_xlim(-2, 2)
ax3.set_ylim(-4, 4)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_title('Subspace: The Entire Space')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('subspaces_visualization.png', dpi=150)
plt.show()

# ============================================
# Lab 4: Subspace Test Function
# ============================================

def test_subspace_numerically(constraint_func, space_dim, num_tests=1000):
    """
    Test if a constraint defines a subspace by random sampling.
    
    constraint_func: returns True if vector is in the subset
    space_dim: dimension of ambient space
    num_tests: number of random tests
    """
    # Test 1: Zero vector
    zero = np.zeros(space_dim)
    if not constraint_func(zero):
        return False, "Zero vector not in set"
    
    # Test 2 & 3: Closure under linear combinations
    for _ in range(num_tests):
        # Generate random vectors that satisfy constraint
        while True:
            v = np.random.randn(space_dim)
            if constraint_func(v):
                break
        
        while True:
            w = np.random.randn(space_dim)
            if constraint_func(w):
                break
        
        # Test closure under addition
        if not constraint_func(v + w):
            return False, f"Not closed under addition: {v} + {w}"
        
        # Test closure under scalar multiplication
        c = np.random.randn()
        if not constraint_func(c * v):
            return False, f"Not closed under scalar mult: {c} * {v}"
    
    return True, "Passed all tests (numerical evidence)"

# Test: Is {(x, y, z) : x + y + z = 0} a subspace?
def plane_constraint(v):
    return np.isclose(v[0] + v[1] + v[2], 0, atol=1e-10)

result, msg = test_subspace_numerically(plane_constraint, 3)
print(f"\nPlane x+y+z=0: {result} - {msg}")

# Test: Is {(x, y, z) : x² + y² ≤ 1} a subspace?
def ball_constraint(v):
    return v[0]**2 + v[1]**2 <= 1

result, msg = test_subspace_numerically(ball_constraint, 3)
print(f"Unit ball: {result} - {msg}")

# ============================================
# Lab 5: QM Preview - Eigenspace as Subspace
# ============================================

# The eigenspace of a matrix is a subspace!
# Let's verify this numerically

# Define a symmetric matrix (will have real eigenvalues)
A = np.array([[4, 2], [2, 1]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nMatrix A eigenvalues: {eigenvalues}")

# For eigenvalue λ, the eigenspace is {v : Av = λv}
# This is the null space of (A - λI), which is a subspace!

for i, lam in enumerate(eigenvalues):
    print(f"\nEigenvalue λ = {lam:.4f}")
    print(f"Eigenvector: {eigenvectors[:, i]}")
    
    # Verify Av = λv
    v = eigenvectors[:, i]
    print(f"Av = {A @ v}")
    print(f"λv = {lam * v}")
    print(f"Check: {np.allclose(A @ v, lam * v)}")
```

---

## ✅ Daily Checklist

- [ ] Read Axler Section 1.C completely
- [ ] Memorize the subspace test (3 conditions)
- [ ] Work through all examples in the text
- [ ] Complete problems 1-6 from practice set
- [ ] Attempt at least one Level 3 proof
- [ ] Complete computational lab
- [ ] Create flashcards for:
  - Subspace definition
  - Subspace test (3 conditions)
  - Linear combination definition
  - Key examples and non-examples
- [ ] Write journal entry on QM eigenspaces

---

## 📓 Reflection Questions

1. Why does a subspace need to contain the zero vector?

2. Why is the union of two subspaces usually NOT a subspace?

3. How does the concept of subspace relate to constraints in physics?

4. What's the geometric interpretation of a subspace in ℝ³?

---

## 🔜 Preview: Tomorrow's Topics

**Day 87: Span and Spanning Sets**

Tomorrow we'll learn:
- The span of a set of vectors
- Spanning sets: when do vectors "fill up" a space?
- Connection to systems of linear equations
- How span relates to reachable states in quantum mechanics

**Preparation:** Think about what vectors you can reach by taking linear combinations of (1, 0) and (0, 1) in ℝ².

---

*"The subspace is to the vector space what the subgroup is to the group: the natural internal structure."*
