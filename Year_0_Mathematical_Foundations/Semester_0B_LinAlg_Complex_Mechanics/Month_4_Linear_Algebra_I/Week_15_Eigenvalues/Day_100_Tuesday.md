# Day 100: The Characteristic Polynomial and Determinants

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Determinants & Characteristic Polynomial |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute determinants of 2×2 and 3×3 matrices efficiently
2. Understand key determinant properties
3. Find characteristic polynomials for n×n matrices
4. Distinguish algebraic from geometric multiplicity
5. State and understand the Cayley-Hamilton theorem
6. Connect determinants to volume and invertibility in QM contexts

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 5.B**: Characteristic polynomial (skip determinant derivation)
- **Section 10.A**: Determinants (reference only — we'll take a computational approach)

### Secondary Text: Strang, "Introduction to Linear Algebra"
- **Section 5.1-5.2**: Properties of Determinants (pp. 247-265)
- Strang's approach is more computational and practical

---

## 📖 Core Content: Theory and Concepts

### 1. Determinants: Computational Approach

We take determinants as a computational tool today, not deriving them axiomatically.

#### 2×2 Determinant

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**Geometric meaning:** Signed area of parallelogram formed by column vectors.

#### 3×3 Determinant: Expansion by First Row

$$\det\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = a\det\begin{pmatrix} e & f \\ h & i \end{pmatrix} - b\det\begin{pmatrix} d & f \\ g & i \end{pmatrix} + c\det\begin{pmatrix} d & e \\ g & h \end{pmatrix}$$

$$= a(ei - fh) - b(di - fg) + c(dh - eg)$$

**Geometric meaning:** Signed volume of parallelepiped formed by column vectors.

#### Cofactor Expansion (General)

For any row i or column j:
$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}$$

where $M_{ij}$ is the (i,j)-minor: determinant of matrix with row i and column j deleted.

### 2. Key Determinant Properties

| Property | Statement |
|----------|-----------|
| Identity | det(I) = 1 |
| Transpose | det(Aᵀ) = det(A) |
| Product | det(AB) = det(A)det(B) |
| Inverse | det(A⁻¹) = 1/det(A) |
| Scalar | det(cA) = cⁿdet(A) for n×n A |
| Triangular | det = product of diagonal entries |
| Row swap | Changes sign of det |
| Row scaling | Scales det by same factor |
| Row addition | Doesn't change det |
| Singular | det(A) = 0 ⟺ A not invertible |

### 3. The Characteristic Polynomial

**Definition:** For an n×n matrix A, the **characteristic polynomial** is:
$$p_A(\lambda) = \det(A - \lambda I)$$

This is a polynomial of degree n in λ.

#### General Form

$$p_A(\lambda) = (-1)^n \lambda^n + (-1)^{n-1}\text{tr}(A)\lambda^{n-1} + \cdots + \det(A)$$

**Key coefficients:**
- Leading coefficient: (-1)ⁿ
- Coefficient of λⁿ⁻¹: (-1)ⁿ⁻¹ tr(A)
- Constant term: det(A)

#### Relationships (Vieta's Formulas)

If eigenvalues are λ₁, λ₂, ..., λₙ (counted with multiplicity):

$$\text{tr}(A) = \lambda_1 + \lambda_2 + \cdots + \lambda_n$$
$$\det(A) = \lambda_1 \cdot \lambda_2 \cdots \lambda_n$$

### 4. Computing Characteristic Polynomials

#### Example 1: 2×2 Case

$$A = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$$

$$A - \lambda I = \begin{pmatrix} 3-\lambda & 1 \\ 2 & 4-\lambda \end{pmatrix}$$

$$p_A(\lambda) = (3-\lambda)(4-\lambda) - 2 = \lambda^2 - 7\lambda + 10$$

Check: tr(A) = 7 ✓, det(A) = 10 ✓

Roots: λ = (7 ± √9)/2 = 5, 2

#### Example 2: 3×3 Case

$$B = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 2 \end{pmatrix}$$

Since B is upper triangular:
$$p_B(\lambda) = (1-\lambda)(3-\lambda)(2-\lambda)$$

Eigenvalues: λ = 1, 2, 3

### 5. Algebraic vs Geometric Multiplicity

**Algebraic multiplicity** of λ: Number of times (λ - λᵢ) appears in characteristic polynomial (multiplicity as a root).

**Geometric multiplicity** of λ: dim(Eλ) = dim(ker(A - λI)) = number of linearly independent eigenvectors.

**Key theorem:** 
$$1 \leq \text{geometric mult.} \leq \text{algebraic mult.}$$

#### Example: Defective Matrix

$$A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$$

Characteristic polynomial: (2-λ)² = 0
- λ = 2 with algebraic multiplicity 2

Eigenspace:
$$A - 2I = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$$
Kernel: span{(1,0)}
- Geometric multiplicity = 1

Since geometric < algebraic, this matrix is **defective** (not diagonalizable).

### 6. The Cayley-Hamilton Theorem

**Theorem:** Every square matrix satisfies its own characteristic equation.

If p_A(λ) = λⁿ + cₙ₋₁λⁿ⁻¹ + ⋯ + c₁λ + c₀, then:
$$p_A(A) = Aⁿ + c_{n-1}A^{n-1} + \cdots + c_1 A + c_0 I = 0$$

#### Example

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$

p_A(λ) = λ² - 5λ - 2

Cayley-Hamilton predicts: A² - 5A - 2I = 0

Verify:
$$A^2 = \begin{pmatrix} 7 & 10 \\ 15 & 22 \end{pmatrix}$$

$$A^2 - 5A - 2I = \begin{pmatrix} 7 & 10 \\ 15 & 22 \end{pmatrix} - \begin{pmatrix} 5 & 10 \\ 15 & 20 \end{pmatrix} - \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$$ ✓

#### Application: Computing Matrix Inverses

From Cayley-Hamilton: A² - 5A - 2I = 0
Rearrange: A² - 5A = 2I
Factor: A(A - 5I) = 2I
Therefore: A⁻¹ = (A - 5I)/2

### 7. Fundamental Theorem of Algebra Connection

**Theorem:** Every polynomial of degree n has exactly n roots in ℂ (counted with multiplicity).

**Consequence:** Every n×n complex matrix has exactly n eigenvalues (with multiplicity).

Real matrices may have complex eigenvalues, but they come in conjugate pairs.

---

## 🔬 Quantum Mechanics Connection

### Determinants and Unitarity

For a unitary matrix U (quantum gate):
- det(U†U) = det(I) = 1
- det(U†)det(U) = 1
- |det(U)|² = 1
- **|det(U)| = 1** (determinant is on unit circle)

This is a necessary condition for unitarity.

### Characteristic Polynomial and Energy Levels

For a quantum Hamiltonian H:
$$\det(H - EI) = 0$$

The roots E are energy eigenvalues. Solving this equation gives the **energy spectrum** of the system.

#### Example: Two-Level System

$$H = \begin{pmatrix} E_0 & V \\ V & E_0 \end{pmatrix}$$

where E₀ is the base energy and V is the coupling.

$$\det(H - EI) = (E_0 - E)^2 - V^2 = 0$$
$$E = E_0 \pm V$$

**Level splitting:** The coupling V splits the degenerate levels!

### Multiplicity and Degeneracy

In quantum mechanics:
- **Algebraic multiplicity** = how many times an energy appears
- **Degenerate eigenvalue** = algebraic multiplicity > 1
- **Degeneracy** is crucial for atomic physics (shell structure)

### The Trace in QM

The trace has special importance:
- tr(ρ) = 1 for density matrices (probability conservation)
- tr(AB) = tr(BA) (cyclic property)
- tr(A) = Σᵢ λᵢ connects to partition function in statistical mechanics

---

## ✏️ Worked Examples

### Example 1: Complete 3×3 Analysis

Find all eigenvalues and eigenvectors of:
$$A = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{pmatrix}$$

**Solution:**

This is upper triangular, so eigenvalues = diagonal entries.
- λ₁ = 2 (algebraic multiplicity 2)
- λ₂ = 3 (algebraic multiplicity 1)

**For λ = 2:**
$$A - 2I = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

Row reduce: x₂ = 0, x₃ = 0, x₁ free.
E₂ = span{(1, 0, 0)}
Geometric multiplicity = 1 < 2 (algebraic)
**A is defective!**

**For λ = 3:**
$$A - 3I = \begin{pmatrix} -1 & 1 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

Row reduce: x₂ = 0, x₁ = 0, x₃ free.
E₃ = span{(0, 0, 1)}

### Example 2: Complex Eigenvalues

$$A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

This is 90° rotation.

**Characteristic polynomial:**
$$p(\lambda) = \lambda^2 + 1 = 0$$
$$\lambda = \pm i$$

**For λ = i:**
$$A - iI = \begin{pmatrix} -i & -1 \\ 1 & -i \end{pmatrix}$$

First row: -ix - y = 0, so y = -ix.
Eigenvector: (1, -i)

**For λ = -i:**
Eigenvector: (1, i)

**Note:** Complex conjugate eigenvalues have complex conjugate eigenvectors.

### Example 3: Cayley-Hamilton Application

Use Cayley-Hamilton to find A⁻¹ for:
$$A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$$

**Step 1:** Find characteristic polynomial.
p(λ) = λ² - 5λ + 5

**Step 2:** Apply Cayley-Hamilton.
A² - 5A + 5I = 0

**Step 3:** Solve for A⁻¹.
5I = 5A - A²
5I = A(5I - A)
I = A · (5I - A)/5
A⁻¹ = (5I - A)/5 = (1/5)(5I - A)

$$A^{-1} = \frac{1}{5}\begin{pmatrix} 3 & -1 \\ -1 & 2 \end{pmatrix}$$

**Verify:** AA⁻¹ = I ✓

---

## 📝 Practice Problems

### Level 1: Determinant Computation
1. Compute det([[3,1],[4,2]]).

2. Compute det([[1,2,3],[0,4,5],[0,0,6]]).

3. If det(A) = 5, what is det(3A) for 3×3 matrix A?

4. If det(A) = 2 and det(B) = 3, what is det(AB)?

### Level 2: Characteristic Polynomials
5. Find the characteristic polynomial of [[4,-2],[1,1]].

6. Find eigenvalues of [[0,1,0],[0,0,1],[1,0,0]]. (Hint: it's a permutation matrix)

7. A 3×3 matrix has eigenvalues 1, 2, 3. What is its determinant? Trace?

8. Find all eigenvalues and their multiplicities for [[5,0,0],[0,5,0],[1,2,5]].

### Level 3: Theory
9. Prove: det(A) = 0 ⟺ 0 is an eigenvalue of A.

10. Prove: If A is nilpotent (Aᵏ = 0 for some k), all eigenvalues are 0.

11. Use Cayley-Hamilton to express A⁴ in terms of A, I for 2×2 matrix A.

12. Prove: Eigenvalues of Aᵀ equal eigenvalues of A.

### Level 4: Quantum Applications
13. For H = [[E₁, V],[V, E₂]], find the eigenvalues. When are they degenerate?

14. Show that for Hermitian H, the characteristic polynomial has real coefficients.

15. A quantum system has Hamiltonian H with eigenvalues E₁ < E₂ < E₃. Write tr(e^{-βH}) in terms of these eigenvalues. (This is the partition function!)

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

# ============================================
# Lab 1: Determinant Properties
# ============================================

def verify_determinant_properties():
    """Verify key determinant properties numerically"""
    print("=== Determinant Properties ===\n")
    
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    c = 2.5
    
    print("Testing with random 3×3 matrices...\n")
    
    # Property 1: det(AB) = det(A)det(B)
    lhs = np.linalg.det(A @ B)
    rhs = np.linalg.det(A) * np.linalg.det(B)
    print(f"det(AB) = {lhs:.6f}")
    print(f"det(A)det(B) = {rhs:.6f}")
    print(f"Equal: {np.isclose(lhs, rhs)}\n")
    
    # Property 2: det(A^T) = det(A)
    print(f"det(A) = {np.linalg.det(A):.6f}")
    print(f"det(A^T) = {np.linalg.det(A.T):.6f}")
    print(f"Equal: {np.isclose(np.linalg.det(A), np.linalg.det(A.T))}\n")
    
    # Property 3: det(cA) = c^n det(A)
    n = 3
    print(f"det({c}A) = {np.linalg.det(c * A):.6f}")
    print(f"{c}^{n} det(A) = {c**n * np.linalg.det(A):.6f}")
    print(f"Equal: {np.isclose(np.linalg.det(c*A), c**n * np.linalg.det(A))}\n")
    
    # Property 4: det(A^{-1}) = 1/det(A) (if invertible)
    if np.abs(np.linalg.det(A)) > 1e-10:
        A_inv = np.linalg.inv(A)
        print(f"det(A^(-1)) = {np.linalg.det(A_inv):.6f}")
        print(f"1/det(A) = {1/np.linalg.det(A):.6f}")
        print(f"Equal: {np.isclose(np.linalg.det(A_inv), 1/np.linalg.det(A))}")

verify_determinant_properties()

# ============================================
# Lab 2: Characteristic Polynomial
# ============================================

def characteristic_polynomial(A):
    """
    Compute characteristic polynomial coefficients.
    Returns coefficients [c_0, c_1, ..., c_n] for c_0 + c_1*λ + ... + c_n*λ^n
    """
    n = A.shape[0]
    # Use numpy's poly function on eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    # Characteristic polynomial has roots at eigenvalues
    # p(λ) = (λ - λ_1)(λ - λ_2)...(λ - λ_n)
    coeffs = np.poly(eigenvalues)  # highest degree first
    return coeffs[::-1]  # reverse to lowest degree first

def verify_char_poly(A):
    """Verify characteristic polynomial properties"""
    print("\n=== Characteristic Polynomial Analysis ===\n")
    print(f"Matrix A:\n{A}\n")
    
    n = A.shape[0]
    eigenvalues = np.linalg.eigvals(A)
    trace = np.trace(A)
    det = np.linalg.det(A)
    
    print(f"Eigenvalues: {eigenvalues}")
    print(f"\nSum of eigenvalues: {np.sum(eigenvalues):.6f}")
    print(f"Trace of A: {trace:.6f}")
    print(f"Equal: {np.isclose(np.sum(eigenvalues), trace)}")
    
    print(f"\nProduct of eigenvalues: {np.prod(eigenvalues):.6f}")
    print(f"Determinant of A: {det:.6f}")
    print(f"Equal: {np.isclose(np.prod(eigenvalues), det)}")
    
    # Characteristic polynomial
    coeffs = characteristic_polynomial(A)
    print(f"\nCharacteristic polynomial coefficients (low to high degree):")
    print(f"  {coeffs}")
    
    # Verify constant term = det(A) (up to sign)
    print(f"\nConstant term: {coeffs[0]:.6f}")
    print(f"det(A): {det:.6f}")

# Test with specific matrices
A1 = np.array([[4, 2],
               [1, 3]])
verify_char_poly(A1)

A2 = np.array([[1, 2, 0],
               [0, 3, 1],
               [0, 0, 2]])
verify_char_poly(A2)

# ============================================
# Lab 3: Cayley-Hamilton Theorem
# ============================================

def verify_cayley_hamilton(A):
    """Verify that A satisfies its characteristic equation"""
    print("\n=== Cayley-Hamilton Verification ===\n")
    print(f"Matrix A:\n{A}\n")
    
    n = A.shape[0]
    
    # Get characteristic polynomial coefficients (numpy convention: highest first)
    eigenvalues = np.linalg.eigvals(A)
    coeffs = np.poly(eigenvalues)  # [1, -tr(A), ..., det(A)]
    
    print(f"Characteristic polynomial p(λ) with coefficients: {coeffs}")
    print(f"  (format: λ^n + c_{n-1}λ^{n-1} + ... + c_0)")
    
    # Compute p(A) = A^n + c_{n-1}A^{n-1} + ... + c_0 I
    result = np.zeros_like(A, dtype=complex)
    A_power = np.eye(n)  # Start with I = A^0
    
    for i, c in enumerate(coeffs[::-1]):  # Go from c_0 to c_n
        result += c * A_power
        A_power = A_power @ A
    
    print(f"\np(A) = A^{n} + ... =\n{result}")
    print(f"\nMax absolute value in p(A): {np.max(np.abs(result)):.2e}")
    print(f"Cayley-Hamilton satisfied: {np.allclose(result, 0)}")

verify_cayley_hamilton(np.array([[1, 2], [3, 4]]))
verify_cayley_hamilton(np.array([[2, 1, 0], [0, 2, 1], [0, 0, 3]]))

# ============================================
# Lab 4: Algebraic vs Geometric Multiplicity
# ============================================

def analyze_multiplicity(A):
    """Analyze algebraic and geometric multiplicities"""
    print("\n=== Multiplicity Analysis ===\n")
    print(f"Matrix A:\n{A}\n")
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Count algebraic multiplicities
    unique_evals, counts = np.unique(np.round(eigenvalues, 10), return_counts=True)
    
    print("Eigenvalue analysis:")
    for eval_val, alg_mult in zip(unique_evals, counts):
        # Find geometric multiplicity by computing rank of (A - λI)
        A_minus_lambdaI = A - eval_val * np.eye(A.shape[0])
        nullity = A.shape[0] - np.linalg.matrix_rank(A_minus_lambdaI, tol=1e-10)
        
        print(f"\n  λ = {eval_val:.4f}")
        print(f"    Algebraic multiplicity: {alg_mult}")
        print(f"    Geometric multiplicity: {nullity}")
        
        if nullity < alg_mult:
            print(f"    ⚠️ DEFECTIVE: geometric < algebraic!")
        else:
            print(f"    ✓ Non-defective for this eigenvalue")
    
    # Check if matrix is diagonalizable
    total_geom = sum(A.shape[0] - np.linalg.matrix_rank(A - ev * np.eye(A.shape[0]), tol=1e-10) 
                     for ev in unique_evals)
    
    print(f"\nTotal geometric multiplicities: {total_geom}")
    print(f"Matrix dimension: {A.shape[0]}")
    print(f"Matrix is diagonalizable: {total_geom == A.shape[0]}")

# Non-defective matrix
analyze_multiplicity(np.array([[1, 0], [0, 1]]))

# Defective matrix
analyze_multiplicity(np.array([[2, 1], [0, 2]]))

# 3x3 defective
analyze_multiplicity(np.array([[5, 1, 0], [0, 5, 1], [0, 0, 5]]))

# ============================================
# Lab 5: Quantum Level Splitting
# ============================================

def analyze_two_level_system(E0, V):
    """Analyze a two-level quantum system with coupling"""
    print(f"\n=== Two-Level System (E0={E0}, V={V}) ===\n")
    
    H = np.array([[E0, V], [V, E0]])
    print(f"Hamiltonian H:\n{H}\n")
    
    eigenvalues, eigenvectors = np.linalg.eig(H)
    
    print(f"Energy eigenvalues: {eigenvalues}")
    print(f"Expected: E0 ± V = {E0 + V}, {E0 - V}")
    
    print(f"\nEnergy splitting: ΔE = {np.abs(eigenvalues[0] - eigenvalues[1]):.4f}")
    print(f"Expected: 2|V| = {2*abs(V):.4f}")
    
    print("\nEnergy eigenstates:")
    for i, (E, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"  |E_{i}⟩ = {v} with E = {E:.4f}")

# No coupling (degenerate)
analyze_two_level_system(1.0, 0.0)

# With coupling (non-degenerate)
analyze_two_level_system(1.0, 0.5)

# Plot level splitting vs coupling
V_values = np.linspace(0, 2, 100)
E_plus = 1 + V_values
E_minus = 1 - V_values

plt.figure(figsize=(10, 6))
plt.plot(V_values, E_plus, 'b-', label='E₊ = E₀ + V')
plt.plot(V_values, E_minus, 'r-', label='E₋ = E₀ - V')
plt.axhline(y=1, color='gray', linestyle='--', label='E₀ (uncoupled)')
plt.xlabel('Coupling strength V')
plt.ylabel('Energy')
plt.title('Quantum Level Splitting: Avoided Crossing')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('level_splitting.png', dpi=150)
plt.show()

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Read Axler 5.B and Strang 5.1-5.2
- [ ] Practice 3×3 determinant computation by hand
- [ ] Derive characteristic polynomial for 2×2 general matrix
- [ ] Solve problems 1-8 from practice set
- [ ] Work through Cayley-Hamilton examples
- [ ] Complete computational lab
- [ ] Understand algebraic vs geometric multiplicity
- [ ] Write QM connection notes on level splitting

---

## 🔜 Preview: Tomorrow's Topics

**Day 101: Diagonalization**

Tomorrow we'll explore:
- When is a matrix diagonalizable?
- Finding the diagonalizing matrix P
- D = P⁻¹AP and its uses
- Computing matrix powers via diagonalization
- QM: Diagonal Hamiltonians and stationary states

---

*"Eigenvalues are like the secret code of a matrix—once you crack it, everything becomes clearer."*
— Unknown
