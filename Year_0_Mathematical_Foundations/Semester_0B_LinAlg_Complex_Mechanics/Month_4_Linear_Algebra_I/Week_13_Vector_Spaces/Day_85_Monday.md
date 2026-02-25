# Day 85: Definition of Vector Spaces — The Foundation

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Vector Space Axioms |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Examples and Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Exploration |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State all eight vector space axioms precisely
2. Verify whether a given set with operations forms a vector space
3. Work with vector spaces over ℝ (real) and ℂ (complex)
4. Identify common vector spaces: ℝⁿ, ℂⁿ, function spaces, polynomial spaces
5. Understand why complex vector spaces are essential for quantum mechanics
6. Recognize non-examples and why they fail

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 1.A**: ℝⁿ and ℂⁿ (pp. 1-6)
- **Section 1.B**: Definition of Vector Space (pp. 7-14)

### Reading Strategy
1. First pass (30 min): Skim for structure, note definitions
2. Second pass (90 min): Careful reading, write out all axioms
3. Work through every example in the text

### Supplementary Reading
- **Strang, Chapter 3.1**: Vector Spaces (pp. 123-130)
- **Shankar, Chapter 1.1**: Linear Vector Spaces (preview the QM perspective)

---

## 🎬 Video Resources

### 3Blue1Brown: Essence of Linear Algebra
- **Chapter 1**: Vectors, what even are they?
- URL: https://www.youtube.com/watch?v=fNk_zzaMoSs
- Duration: 10 minutes
- **Key Insight**: Multiple perspectives on vectors

### MIT OCW 18.06 (Gilbert Strang)
- **Lecture 5**: Vector Spaces and Subspaces
- Focus on the first 20 minutes for abstract definition
- URL: MIT OpenCourseWare 18.06

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: Why Abstract Vector Spaces?

Before defining vector spaces abstractly, consider where "vectors" appear:

| Context | "Vectors" | Operations |
|---------|-----------|------------|
| Physics | Arrows with magnitude and direction | Addition, scaling |
| ℝⁿ | Ordered n-tuples of real numbers | Component-wise ops |
| Polynomials | Expressions like 3x² + 2x - 1 | Polynomial addition, scaling |
| Functions | f: ℝ → ℝ | Function addition, scaling |
| Quantum states | State vectors \|ψ⟩ | Superposition |

The abstract definition captures what all these have in common.

### 2. Fields: ℝ vs ℂ

A **field** is a set with addition and multiplication satisfying familiar properties (associativity, commutativity, distributivity, existence of 0, 1, and inverses).

**Key Fields:**
- **ℝ** (real numbers): Familiar from calculus
- **ℂ** (complex numbers): Essential for quantum mechanics

**Notation:**
- **𝔽** denotes either ℝ or ℂ (generic field)
- Vector spaces over ℝ are called **real vector spaces**
- Vector spaces over ℂ are called **complex vector spaces**

### 3. The Definition of a Vector Space

**Definition:** A **vector space** over a field 𝔽 is a set V together with:
- An **addition** operation: V × V → V written (u, v) ↦ u + v
- A **scalar multiplication** operation: 𝔽 × V → V written (c, v) ↦ cv

satisfying the following **eight axioms** for all u, v, w ∈ V and all a, b ∈ 𝔽:

#### Addition Axioms (4)
| # | Name | Statement |
|---|------|-----------|
| 1 | Commutativity | u + v = v + u |
| 2 | Associativity | (u + v) + w = u + (v + w) |
| 3 | Additive Identity | ∃ 0 ∈ V: v + 0 = v for all v |
| 4 | Additive Inverse | ∀ v ∈ V, ∃ (-v) ∈ V: v + (-v) = 0 |

#### Scalar Multiplication Axioms (4)
| # | Name | Statement |
|---|------|-----------|
| 5 | Multiplicative Identity | 1v = v |
| 6 | Compatibility | a(bv) = (ab)v |
| 7 | Distributivity over vector addition | a(u + v) = au + av |
| 8 | Distributivity over scalar addition | (a + b)v = av + bv |

### 4. Terminology

**Elements of V** are called **vectors** (even if they're polynomials, functions, or matrices).

**Elements of 𝔽** are called **scalars**.

**The zero vector** (additive identity) is written **0** (bold) or sometimes **0⃗**.

### 5. Fundamental Examples

#### Example 1: ℝⁿ (n-tuples of real numbers)

$$\mathbb{R}^n = \{(x_1, x_2, \ldots, x_n) : x_i \in \mathbb{R}\}$$

Operations:
- $(x_1, \ldots, x_n) + (y_1, \ldots, y_n) = (x_1 + y_1, \ldots, x_n + y_n)$
- $c(x_1, \ldots, x_n) = (cx_1, \ldots, cx_n)$

Zero vector: $(0, 0, \ldots, 0)$

**Verification:** All 8 axioms are inherited from properties of real numbers.

#### Example 2: ℂⁿ (n-tuples of complex numbers)

$$\mathbb{C}^n = \{(z_1, z_2, \ldots, z_n) : z_i \in \mathbb{C}\}$$

Same operations as ℝⁿ, but with complex scalars.

**Critical for QM:** Quantum states live in complex vector spaces!

#### Example 3: 𝒫(𝔽) — Polynomials with coefficients in 𝔽

$$\mathcal{P}(\mathbb{F}) = \{a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n : a_i \in \mathbb{F}, n \geq 0\}$$

Operations:
- Add polynomials by adding coefficients
- Multiply by scalar: multiply all coefficients

Zero vector: The zero polynomial (all coefficients 0)

#### Example 4: 𝒫ₙ(𝔽) — Polynomials of degree at most n

$$\mathcal{P}_n(\mathbb{F}) = \{a_0 + a_1 x + \cdots + a_n x^n : a_i \in \mathbb{F}\}$$

This is a "finite-dimensional" polynomial space.

#### Example 5: Function Spaces

$$\mathcal{F}(\mathbb{R}, \mathbb{R}) = \{f : \mathbb{R} \to \mathbb{R}\}$$

Operations:
- $(f + g)(x) = f(x) + g(x)$
- $(cf)(x) = c \cdot f(x)$

Zero vector: The zero function, $f(x) = 0$ for all $x$.

#### Example 6: Matrix Spaces M_{m×n}(𝔽)

$$M_{m \times n}(\mathbb{F}) = \{m \times n \text{ matrices with entries in } \mathbb{F}\}$$

Operations: Entry-wise addition and scalar multiplication.

### 6. Non-Examples (Crucial for Understanding)

Understanding what **fails** to be a vector space deepens understanding.

#### Non-Example 1: ℝ² with modified addition

Define: $(x_1, x_2) \oplus (y_1, y_2) = (x_1 + y_1 + 1, x_2 + y_2)$

**Why it fails:** No additive identity exists.
If $(e_1, e_2)$ were identity: $(x_1, x_2) \oplus (e_1, e_2) = (x_1 + e_1 + 1, x_2 + e_2) = (x_1, x_2)$
This requires $e_1 + 1 = 0$ and $e_2 = 0$, so $e_1 = -1$.
But then $(0, 0) \oplus (-1, 0) = (0, 0)$, while
$(-1, 0) \oplus (0, 0) = (0, 0)$.
Check: Does $(a, b) \oplus (-1, 0) = (a, b)$?
$(a + (-1) + 1, b + 0) = (a, b)$ ✓
Actually this works! Let me reconsider...

Better Non-Example: $(x_1, x_2) \oplus (y_1, y_2) = (x_1 y_1, x_2 + y_2)$

**Why it fails:** Addition is not commutative in general if we include zero elements, and there's no identity that works for all vectors.

#### Non-Example 2: The positive real numbers with usual multiplication

$V = \mathbb{R}^+ = \{x \in \mathbb{R} : x > 0\}$ with "addition" = multiplication, "scalar multiplication" = exponentiation.

Actually, this IS a vector space (isomorphic to ℝ)! This shows that "vectors" can be strange objects.

#### Non-Example 3: Integers ℤ as a "vector space" over ℝ

**Why it fails:** Closure under scalar multiplication fails.
$\pi \cdot 1 = \pi \notin \mathbb{Z}$

### 7. Properties Derived from Axioms

**Theorem:** In any vector space:
1. The zero vector is unique
2. Additive inverses are unique
3. $0v = 0$ for all $v \in V$
4. $a0 = 0$ for all $a \in \mathbb{F}$
5. $(-1)v = -v$ for all $v \in V$

**Proof of (3):** $0v = (0 + 0)v = 0v + 0v$
Adding $-(0v)$ to both sides: $0 = 0v$ ∎

**Proof of (5):** $v + (-1)v = 1v + (-1)v = (1 + (-1))v = 0v = 0$
So $(-1)v$ is the additive inverse of $v$, which is $-v$ by definition. ∎

---

## 🔬 Quantum Mechanics Connection

### State Spaces are Complex Vector Spaces

In quantum mechanics, the state of a physical system is described by a **state vector** |ψ⟩ living in a **Hilbert space** (a special type of complex vector space with additional structure).

**Key properties from today's material:**

1. **Superposition Principle:** If |ψ₁⟩ and |ψ₂⟩ are valid quantum states, then
   $$|\psi\rangle = \alpha|\psi_1\rangle + \beta|\psi_2\rangle$$
   is also a valid quantum state (for any α, β ∈ ℂ).

   This is exactly the closure axiom for complex vector spaces!

2. **Complex Scalars:** The coefficients α, β are complex numbers.
   - Their magnitudes relate to probabilities
   - Their phases lead to interference effects

3. **Zero State:** The zero vector |0⟩ exists but represents a non-physical state
   (a state with zero probability everywhere).

### Example: Qubit State Space

A **qubit** (quantum bit) lives in ℂ²:
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad \alpha, \beta \in \mathbb{C}$$

where |0⟩ = (1, 0) and |1⟩ = (0, 1) are basis states.

The normalization condition |α|² + |β|² = 1 is additional structure (from the inner product), but the underlying linear algebra is what we're learning today.

### Why Complex and Not Real?

Quantum interference requires complex amplitudes:
- Amplitudes can cancel (destructive interference)
- Phases carry information
- Real vector spaces cannot capture this

This is why we emphasize ℂⁿ alongside ℝⁿ throughout linear algebra.

---

## ✏️ Worked Examples

### Example 1: Verifying ℂ² is a vector space

Let $V = \mathbb{C}^2$ with standard operations.

**Axiom 1 (Commutativity):**
$(z_1, z_2) + (w_1, w_2) = (z_1 + w_1, z_2 + w_2) = (w_1 + z_1, w_2 + z_2) = (w_1, w_2) + (z_1, z_2)$ ✓

**Axiom 3 (Zero vector):**
Zero vector is $(0, 0)$.
$(z_1, z_2) + (0, 0) = (z_1 + 0, z_2 + 0) = (z_1, z_2)$ ✓

**Axiom 7 (Distributivity):**
$c((z_1, z_2) + (w_1, w_2)) = c(z_1 + w_1, z_2 + w_2) = (c(z_1 + w_1), c(z_2 + w_2))$
$= (cz_1 + cw_1, cz_2 + cw_2) = (cz_1, cz_2) + (cw_1, cw_2)$
$= c(z_1, z_2) + c(w_1, w_2)$ ✓

(Continue for all 8 axioms...)

### Example 2: The space of continuous functions

Let $V = C([0,1]) = \{f : [0,1] \to \mathbb{R} : f \text{ is continuous}\}$

**Question:** Is this a vector space over ℝ?

**Verification:**
- **Closure under addition:** Sum of continuous functions is continuous ✓
- **Closure under scalar multiplication:** Scalar multiple of continuous function is continuous ✓
- **Zero vector:** The zero function $f(x) = 0$ is continuous ✓
- **Additive inverse:** If $f$ is continuous, so is $-f$ ✓

All axioms follow from properties of real numbers. Yes, it's a vector space.

### Example 3: Non-example — Polynomials of degree exactly n

Let $V = \{p(x) = a_n x^n + \ldots + a_1 x + a_0 : a_n \neq 0\}$ (polynomials of degree exactly $n$).

**Question:** Is this a vector space?

**Answer:** No! Consider $p(x) = x^2 + 1$ and $q(x) = -x^2 + x$.
Both have degree 2.
$p + q = x + 1$, which has degree 1, not 2.

Closure under addition fails!

### Example 4: Sequences

Let $V = \{(a_1, a_2, a_3, \ldots) : a_i \in \mathbb{R}\}$ (infinite real sequences).

With component-wise operations, this is a vector space (infinite-dimensional).

**QM Connection:** Quantum states of a harmonic oscillator live in such a space!

---

## 📝 Practice Problems

### Level 1: Basic Verification
1. Verify that ℝ³ satisfies the distributivity axiom (Axiom 7).

2. Show that $M_{2 \times 2}(\mathbb{R})$ (2×2 real matrices) is a vector space.

3. Prove that in any vector space, if $v + w = v$, then $w = 0$.

### Level 2: Examples and Non-Examples
4. Is $\{(x, y, z) \in \mathbb{R}^3 : x + y + z = 0\}$ a vector space? (With standard operations)

5. Is $\{(x, y, z) \in \mathbb{R}^3 : x + y + z = 1\}$ a vector space? Why or why not?

6. Consider $V = \mathbb{R}^2$ with operations:
   - $(x_1, y_1) + (x_2, y_2) = (x_1 + x_2, y_1 + y_2)$
   - $c(x, y) = (cx, 0)$
   
   Which axiom(s) fail?

### Level 3: Proofs
7. Prove: In any vector space, if $av = 0$, then $a = 0$ or $v = 0$.

8. Prove: The set of solutions to a homogeneous linear ODE $y'' + p(x)y' + q(x)y = 0$ forms a vector space.

9. Let $V$ be a vector space. Prove that if $u + v = u + w$, then $v = w$ (cancellation law).

### Level 4: Conceptual
10. Can a vector space have exactly three elements? Explain.

11. Is the set of all 2×2 invertible matrices a vector space? Why or why not?

12. Consider the set of real-valued functions on [0,1] that are zero at x = 0. Is this a vector space?

---

## 📊 Answers and Hints

1. Direct computation with components
2. Verify all 8 axioms for matrices
3. Add $(-v)$ to both sides
4. **Yes** — check all axioms (it passes through origin)
5. **No** — zero vector $(0,0,0)$ is not in the set
6. Axiom 5 fails: $1(x,y) = (x,0) \neq (x,y)$ in general
7. Assume $a \neq 0$. Then $a^{-1}$ exists. $a^{-1}(av) = v$, so $v = a^{-1} \cdot 0 = 0$.
8. Show closure under addition and scalar multiplication
9. Add $(-u)$ to both sides
10. Consider the zero vector and closure requirements
11. No — not closed under addition (sum of invertible matrices may not be invertible)
12. **Yes** — verify it's closed under operations and contains zero function

---

## 💻 Evening Computational Lab (1 hour)

### Python/NumPy Exploration

```python
import numpy as np

# ============================================
# Lab 1: Vector Space Operations in NumPy
# ============================================

# 1.1 Working with ℝ³
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

# Vector addition
print("v + w =", v + w)

# Scalar multiplication
c = 3.5
print("c * v =", c * v)

# Zero vector
zero = np.zeros(3)
print("v + 0 =", v + zero)

# Additive inverse
print("v + (-v) =", v + (-v))

# 1.2 Working with ℂ²
z1 = np.array([1+2j, 3-1j])
z2 = np.array([2-1j, 1+4j])

print("\nComplex vectors:")
print("z1 =", z1)
print("z2 =", z2)
print("z1 + z2 =", z1 + z2)

# Scalar multiplication with complex scalar
alpha = 2 + 3j
print("α * z1 =", alpha * z1)

# ============================================
# Lab 2: Verifying Axioms Numerically
# ============================================

def verify_commutativity(v, w):
    """Verify v + w = w + v"""
    return np.allclose(v + w, w + v)

def verify_associativity(u, v, w):
    """Verify (u + v) + w = u + (v + w)"""
    return np.allclose((u + v) + w, u + (v + w))

def verify_distributivity(a, v, w):
    """Verify a(v + w) = av + aw"""
    return np.allclose(a * (v + w), a*v + a*w)

# Test with random vectors
u = np.random.rand(5)
v = np.random.rand(5)
w = np.random.rand(5)
a = np.random.rand()

print("\nAxiom verification:")
print("Commutativity:", verify_commutativity(v, w))
print("Associativity:", verify_associativity(u, v, w))
print("Distributivity:", verify_distributivity(a, v, w))

# ============================================
# Lab 3: Polynomial Space
# ============================================

# Represent polynomials as coefficient arrays
# p(x) = 3x² + 2x + 1 → [1, 2, 3] (lowest degree first)

p = np.array([1, 2, 3])  # 1 + 2x + 3x²
q = np.array([2, -1, 1])  # 2 - x + x²

# Addition (same as vector addition)
print("\nPolynomials:")
print("p + q =", p + q)  # 3 + x + 4x²

# Scalar multiplication
print("2 * p =", 2 * p)

# Evaluate polynomial at a point
def eval_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))

print("p(2) =", eval_poly(p, 2))  # Should be 1 + 4 + 12 = 17

# ============================================
# Lab 4: Matrix Space
# ============================================

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("\nMatrices as vectors:")
print("A + B =")
print(A + B)

print("\n2 * A =")
print(2 * A)

# Zero matrix
print("\nZero matrix:")
print(np.zeros((2, 2)))

# ============================================
# Lab 5: Quantum State Preview
# ============================================

# Qubit state: |ψ⟩ = α|0⟩ + β|1⟩
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Superposition state
alpha = 1/np.sqrt(2)
beta = 1/np.sqrt(2)
psi = alpha * ket_0 + beta * ket_1

print("\nQubit superposition:")
print("|ψ⟩ =", psi)
print("Normalization check: |α|² + |β|² =", np.abs(alpha)**2 + np.abs(beta)**2)

# Another superposition with phase
alpha = 1/np.sqrt(2)
beta = 1j/np.sqrt(2)  # Pure imaginary!
psi_phase = alpha * ket_0 + beta * ket_1

print("\n|ψ⟩ with phase =", psi_phase)
print("Still normalized:", np.abs(alpha)**2 + np.abs(beta)**2)

# This phase difference will matter for interference!
```

### Lab Exercises

1. Create a function that tests all 8 vector space axioms for ℝⁿ numerically.

2. Implement a polynomial class that supports addition and scalar multiplication.

3. Create random complex vectors and verify the axioms.

4. Explore what happens when you try to add vectors of different dimensions.

---

## ✅ Daily Checklist

- [ ] Read Axler 1.A and 1.B completely (two passes)
- [ ] Write out all 8 axioms from memory
- [ ] Watch 3Blue1Brown video on vectors
- [ ] Complete worked examples independently
- [ ] Solve problems 1-6 from practice set
- [ ] Attempt at least one Level 3 proof
- [ ] Complete computational lab
- [ ] Create flashcards for:
  - Vector space definition
  - Each of the 8 axioms
  - Key examples (ℝⁿ, ℂⁿ, 𝒫(𝔽))
- [ ] Write QM connection paragraph in study journal

---

## 📓 Reflection Questions

Before ending today's session, write answers to:

1. In your own words, why do we need an abstract definition of vector space?

2. Why is ℂⁿ more relevant to quantum mechanics than ℝⁿ?

3. What's the most surprising example of a vector space you encountered today?

4. Which axiom do you find most important? Why?

---

## 🔜 Preview: Tomorrow's Topics

**Day 86: Subspaces and Linear Combinations**

Tomorrow we'll explore:
- Subspaces: Vector spaces inside vector spaces
- Linear combinations: Building new vectors from old
- The subspace test (a shortcut for verification)
- Important examples: solution sets, spans

**Preparation:** Think about which subsets of ℝ² might also be vector spaces.

---

*"The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise."*
— Edsger W. Dijkstra
