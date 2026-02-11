# Day 113: The Adjoint Operator — Gateway to Quantum Observables

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Adjoint Operators |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define the adjoint (Hermitian conjugate) of a linear operator
2. Compute adjoints of matrices (conjugate transpose)
3. Understand the relationship between adjoint and inner product
4. Prove key properties of the adjoint operation
5. Recognize the adjoint's role in quantum mechanics (bra-ket duality)
6. Work with adjoint operators computationally

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 7.A**: Adjoints (pp. 209-218)
- Focus on: Definition, properties, matrix representation

### Secondary Reading
- **Shankar, "Principles of QM"**: Chapter 1.6 (Operators and their properties)
- **Strang**: Chapter 6.3 (Complex matrices, Hermitian)

### QM Perspective
- **Griffiths, "Introduction to QM"**: Section 3.2 (Observables)
- Pay attention to how physicists use † notation

---

## 🎬 Video Resources

### MIT OCW 18.06
- **Lecture 28**: Similar Matrices, Jordan Form
- **Lecture 33**: Left and Right Inverses; Pseudoinverse

### Physics Videos
- **Professor M does Science**: "Hermitian Operators in QM"
- **Physics Explained**: "What is the adjoint?"

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: From Inner Products to Operators

Recall from Week 16: An inner product ⟨·,·⟩ on V allows us to:
- Measure lengths and angles
- Define orthogonality
- Project vectors

**Key Question:** How do operators interact with inner products?

For a linear operator T: V → V, we want to understand:
$$\langle Tu, v \rangle \quad \text{vs} \quad \langle u, Tv \rangle$$

In general, these are NOT equal. But there's a natural partner for T...

### 2. Definition of the Adjoint

**Definition:** Let V be an inner product space over 𝔽 (ℝ or ℂ), and let T: V → V be a linear operator. The **adjoint** of T, denoted T* (or T†), is the unique linear operator satisfying:

$$\boxed{\langle Tv, w \rangle = \langle v, T^* w \rangle \quad \text{for all } v, w \in V}$$

**Existence & Uniqueness:** The Riesz representation theorem guarantees T* exists and is unique.

**Alternative notation:**
- Mathematics: T* (star)
- Physics: T† (dagger, "T-dagger")
- Engineering: Tᴴ (superscript H)

### 3. Matrix Representation: Conjugate Transpose

For finite-dimensional spaces with orthonormal basis, the adjoint has a simple matrix form.

**Theorem:** If A is the matrix of T with respect to an orthonormal basis, then the matrix of T* is A* = Ā^T (conjugate transpose).

$$(A^*)_{ij} = \overline{A_{ji}}$$

**Example:**
$$A = \begin{pmatrix} 1+i & 2 \\ 3i & 4-2i \end{pmatrix}$$

$$A^* = \overline{A}^T = \begin{pmatrix} \overline{1+i} & \overline{3i} \\ \overline{2} & \overline{4-2i} \end{pmatrix} = \begin{pmatrix} 1-i & -3i \\ 2 & 4+2i \end{pmatrix}$$

**For real matrices:** A* = Aᵀ (just transpose, no conjugation needed)

### 4. Properties of the Adjoint

**Theorem:** For operators S, T on inner product space V and scalar c:

| Property | Statement |
|----------|-----------|
| (i) Involution | (T*)* = T |
| (ii) Linearity in first slot | (S + T)* = S* + T* |
| (iii) Conjugate homogeneity | (cT)* = c̄ T* |
| (iv) Anti-multiplicativity | (ST)* = T* S* |
| (v) Identity | I* = I |

**Proof of (i):** 
⟨v, (T*)*w⟩ = ⟨T*v, w⟩ = ⟨w, T*v⟩̄ = ⟨Tw, v⟩̄ = ⟨v, Tw⟩

Since this holds for all v, w: (T*)* = T ∎

**Proof of (iv):**
⟨(ST)v, w⟩ = ⟨S(Tv), w⟩ = ⟨Tv, S*w⟩ = ⟨v, T*(S*w)⟩ = ⟨v, (T*S*)w⟩

Therefore (ST)* = T*S* ∎

### 5. The Adjoint and the Inner Product

**Key Insight:** The adjoint "transfers" an operator across the inner product:
$$\langle Tv, w \rangle = \langle v, T^*w \rangle$$

This is analogous to:
- Integration by parts: ∫f'g = -∫fg' + boundary terms
- Matrix transpose: (Ax)·y = x·(Aᵀy)

### 6. Computing Adjoints: Examples

#### Example 1: Diagonal Matrix
$$D = \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix} \implies D^* = \begin{pmatrix} \bar{\lambda}_1 & 0 \\ 0 & \bar{\lambda}_2 \end{pmatrix}$$

#### Example 2: Rotation Matrix (Real)
$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \implies R_\theta^* = R_\theta^T = R_{-\theta}$$

#### Example 3: Differentiation Operator

On L²[0,1] with boundary conditions f(0) = f(1) = 0:

$$T = \frac{d}{dx}$$

What is T*? Use integration by parts:
$$\langle Tf, g \rangle = \int_0^1 f'(x)\overline{g(x)}dx = -\int_0^1 f(x)\overline{g'(x)}dx = -\langle f, g' \rangle$$

So T* = -d/dx (the negative of differentiation!)

#### Example 4: Shift Operators

Right shift on ℓ²: S(a₁, a₂, a₃, ...) = (0, a₁, a₂, ...)
Left shift: S*(a₁, a₂, a₃, ...) = (a₂, a₃, a₄, ...)

Note: S*S = I but SS* ≠ I (not unitary!)

### 7. Kernel and Range Relations

**Theorem:** For any linear operator T:
1. ker(T*) = (range(T))⊥
2. range(T*) = (ker(T))⊥
3. ker(T) = (range(T*))⊥
4. range(T) = (ker(T*))⊥

**Proof of (1):**
w ∈ ker(T*) ⟺ T*w = 0
⟺ ⟨v, T*w⟩ = 0 for all v
⟺ ⟨Tv, w⟩ = 0 for all v
⟺ w ⊥ range(T)
⟺ w ∈ (range(T))⊥ ∎

**Geometric meaning:** The null space of T* is orthogonal to where T maps.

---

## 🔬 Quantum Mechanics Connection

### Bra-Ket Duality

In Dirac notation, the adjoint creates the correspondence:

$$|ψ⟩ \xrightarrow{\text{adjoint}} ⟨ψ|$$

**Ket** |ψ⟩: Column vector (element of V)
**Bra** ⟨ψ|: Row vector (element of V*, the dual space)

The inner product becomes:
$$⟨φ|ψ⟩ = ⟨φ, ψ⟩$$

### Operator Adjoints in QM

For an operator A:
$$⟨φ|A|ψ⟩ = ⟨A^†φ|ψ⟩$$

**Matrix element:** ⟨φ|A|ψ⟩ is called a "matrix element" of A

### Why Adjoints Matter for Observables

**Physical observables** (position, momentum, energy) must give real measurement results.

For operator A representing an observable:
- Eigenvalues = measurement outcomes
- Real eigenvalues require special structure → Hermitian operators!

**Preview:** Tomorrow we'll see that A = A† (self-adjoint/Hermitian) guarantees real eigenvalues.

### Example: Creation and Annihilation Operators

In quantum harmonic oscillator:
- **Annihilation operator:** a
- **Creation operator:** a†

Properties:
- (a)† = a† (the adjoint of annihilation is creation!)
- [a, a†] = 1 (fundamental commutator)
- Number operator: N = a†a

---

## ✏️ Worked Examples

### Example 1: Computing A*

Given:
$$A = \begin{pmatrix} 2+i & 3 & 1-i \\ 0 & i & 4 \end{pmatrix}$$

Find A*:

**Step 1:** Transpose
$$A^T = \begin{pmatrix} 2+i & 0 \\ 3 & i \\ 1-i & 4 \end{pmatrix}$$

**Step 2:** Conjugate each entry
$$A^* = \begin{pmatrix} 2-i & 0 \\ 3 & -i \\ 1+i & 4 \end{pmatrix}$$

### Example 2: Verifying Adjoint Property

Verify ⟨Av, w⟩ = ⟨v, A*w⟩ for:
$$A = \begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix}, \quad v = \begin{pmatrix} 1 \\ i \end{pmatrix}, \quad w = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

**Left side:**
$$Av = \begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix}\begin{pmatrix} 1 \\ i \end{pmatrix} = \begin{pmatrix} 1 + i^2 \\ -i + 2i \end{pmatrix} = \begin{pmatrix} 0 \\ i \end{pmatrix}$$

$$⟨Av, w⟩ = 0 \cdot \bar{1} + i \cdot \bar{1} = i$$

**Right side:**
$$A^* = \begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix}$$ (this matrix is self-adjoint!)

$$A^*w = \begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1+i \\ -i+2 \end{pmatrix}$$

$$⟨v, A^*w⟩ = 1 \cdot \overline{(1+i)} + i \cdot \overline{(2-i)} = (1-i) + i(2+i) = 1-i+2i+i^2 = 1-i+2i-1 = i$$ ✓

### Example 3: Adjoint of a Product

If A* = B and B* = A, find (AB)*.

**Solution:**
$$(AB)^* = B^*A^* = AB$$

So AB is self-adjoint!

### Example 4: Trace and Adjoint

Prove: tr(A*) = tr(A)̄

**Proof:**
$$\text{tr}(A^*) = \sum_i (A^*)_{ii} = \sum_i \overline{A_{ii}} = \overline{\sum_i A_{ii}} = \overline{\text{tr}(A)}$$ ∎

---

## 📝 Practice Problems

### Level 1: Computation
1. Find A* for:
   $$A = \begin{pmatrix} 1+2i & 3-i \\ 4i & 5 \end{pmatrix}$$

2. Compute (AB)* and verify it equals B*A* for:
   $$A = \begin{pmatrix} 1 & i \\ 0 & 1 \end{pmatrix}, \quad B = \begin{pmatrix} 2 & 0 \\ i & 3 \end{pmatrix}$$

3. Find the adjoint of the 3×3 identity matrix.

### Level 2: Verification
4. Show that ⟨Av, w⟩ = ⟨v, A*w⟩ for A = [[0,1],[1,0]] with arbitrary v, w ∈ ℂ².

5. Prove: (A*)⁻¹ = (A⁻¹)* for invertible A.

6. Show that ker(A*A) = ker(A).

### Level 3: Theory
7. Prove: rank(A) = rank(A*) = rank(A*A) = rank(AA*).

8. Show that eigenvalues of A* are complex conjugates of eigenvalues of A.

9. Prove: If A is invertible, then (A*)⁻¹ = (A⁻¹)*.

### Level 4: Quantum Applications
10. The momentum operator in 1D is p = -iℏ(d/dx). Show that p† = p (self-adjoint) on appropriate function space.

11. For the Pauli matrices σₓ, σᵧ, σᵤ, verify each is self-adjoint.

12. If |ψ⟩ = α|0⟩ + β|1⟩, express ⟨ψ| in terms of α*, β*.

---

## 📊 Answers and Hints

1. A* = [[1-2i, -4i], [3+i, 5]]

2. AB = [[2+i², i], [i, 3]], (AB)* = [[2-i², -i], [-i, 3]]
   B*A* = [[2, -i], [0, 3]][[1, 0], [-i, 1]] = ... verify equality

3. I* = I (identity is self-adjoint)

4. Direct computation with components

5. (A*)⁻¹A* = I ⟹ take adjoint of both sides

6. A*Ax = 0 ⟹ ⟨Ax, Ax⟩ = ⟨x, A*Ax⟩ = 0 ⟹ Ax = 0

7. Use ker/range relationships

8. If Av = λv, take adjoint: v*A* = λ̄v*, so λ̄ is eigenvalue of A*

10. Use integration by parts with appropriate boundary conditions

11. Direct computation: σₓ* = σₓ, etc.

12. ⟨ψ| = α*⟨0| + β*⟨1|

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ============================================
# Lab 1: Computing Adjoints
# ============================================

def adjoint(A):
    """Compute the adjoint (conjugate transpose) of A"""
    return A.conj().T

# Test matrices
A = np.array([[1+2j, 3-1j],
              [4j, 5]], dtype=complex)

print("A =")
print(A)
print("\nA† =")
print(adjoint(A))

# Verify: (A†)† = A
print("\n(A†)† =")
print(adjoint(adjoint(A)))
print("Equals A:", np.allclose(adjoint(adjoint(A)), A))

# ============================================
# Lab 2: Verifying Adjoint Property
# ============================================

def inner_product(u, v):
    """Standard complex inner product: ⟨u,v⟩ = u†v"""
    return np.vdot(u, v)  # np.vdot handles conjugation correctly

A = np.array([[1, 1j], [-1j, 2]], dtype=complex)
v = np.array([1, 1j], dtype=complex)
w = np.array([1, 1], dtype=complex)

# Compute ⟨Av, w⟩
Av = A @ v
lhs = inner_product(Av, w)

# Compute ⟨v, A†w⟩
A_adj = adjoint(A)
A_adj_w = A_adj @ w
rhs = inner_product(v, A_adj_w)

print("\n=== Verifying ⟨Av, w⟩ = ⟨v, A†w⟩ ===")
print(f"⟨Av, w⟩ = {lhs}")
print(f"⟨v, A†w⟩ = {rhs}")
print(f"Equal: {np.isclose(lhs, rhs)}")

# ============================================
# Lab 3: Product Rule (AB)† = B†A†
# ============================================

A = np.array([[1, 1j], [0, 1]], dtype=complex)
B = np.array([[2, 0], [1j, 3]], dtype=complex)

AB = A @ B
AB_adj = adjoint(AB)
B_adj_A_adj = adjoint(B) @ adjoint(A)

print("\n=== Verifying (AB)† = B†A† ===")
print("(AB)† =")
print(AB_adj)
print("\nB†A† =")
print(B_adj_A_adj)
print(f"Equal: {np.allclose(AB_adj, B_adj_A_adj)}")

# ============================================
# Lab 4: Kernel/Range Relations
# ============================================

# Random matrix
np.random.seed(42)
A = np.random.randn(4, 3) + 1j * np.random.randn(4, 3)

# Compute null spaces and ranges
# ker(A†) should equal range(A)⊥

from scipy.linalg import null_space

ker_A_adj = null_space(adjoint(A))
print("\n=== Kernel/Range Relations ===")
print(f"dim(ker(A†)) = {ker_A_adj.shape[1]}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}")
print(f"dim(A) columns = {A.shape[1]}")
# dim(ker(A†)) + rank(A) should equal number of rows of A

# ============================================
# Lab 5: Pauli Matrices - Self-Adjoint Check
# ============================================

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

print("\n=== Pauli Matrices Self-Adjoint Check ===")
print(f"σx = σx†: {np.allclose(sigma_x, adjoint(sigma_x))}")
print(f"σy = σy†: {np.allclose(sigma_y, adjoint(sigma_y))}")
print(f"σz = σz†: {np.allclose(sigma_z, adjoint(sigma_z))}")

# ============================================
# Lab 6: Eigenvalue Conjugation Property
# ============================================

A = np.array([[2, 1+1j], [1-1j, 3]], dtype=complex)
eigs_A = np.linalg.eigvals(A)
eigs_A_adj = np.linalg.eigvals(adjoint(A))

print("\n=== Eigenvalue Conjugation ===")
print(f"Eigenvalues of A: {eigs_A}")
print(f"Eigenvalues of A†: {eigs_A_adj}")
print(f"Conjugates of A eigenvalues: {np.conj(eigs_A)}")
# Note: eigenvalues of A† are conjugates of eigenvalues of A

# ============================================
# Lab 7: Bra-Ket Notation Simulation
# ============================================

class Ket:
    """Represents a quantum ket |ψ⟩"""
    def __init__(self, components):
        self.vec = np.array(components, dtype=complex).reshape(-1, 1)
    
    def __repr__(self):
        return f"|ψ⟩ = {self.vec.flatten()}"
    
    def bra(self):
        """Return the corresponding bra ⟨ψ|"""
        return Bra(self.vec.conj().flatten())
    
    def norm(self):
        return np.sqrt(np.vdot(self.vec, self.vec).real)

class Bra:
    """Represents a quantum bra ⟨ψ|"""
    def __init__(self, components):
        self.vec = np.array(components, dtype=complex).reshape(1, -1)
    
    def __repr__(self):
        return f"⟨ψ| = {self.vec.flatten()}"
    
    def __matmul__(self, other):
        if isinstance(other, Ket):
            # ⟨φ|ψ⟩ inner product
            return (self.vec @ other.vec)[0, 0]
        elif isinstance(other, np.ndarray):
            # ⟨φ|A|ψ⟩ through ⟨φ|(A|ψ⟩)
            return Bra((self.vec @ other).flatten())

# Example
psi = Ket([1/np.sqrt(2), 1j/np.sqrt(2)])
print("\n=== Bra-Ket Simulation ===")
print(psi)
print(psi.bra())
print(f"⟨ψ|ψ⟩ = {psi.bra() @ psi}")  # Should be 1

phi = Ket([1, 0])
print(f"\n⟨φ|ψ⟩ = {phi.bra() @ psi}")

# Operator action
print(f"\n⟨φ|σx|ψ⟩ = {phi.bra() @ sigma_x @ psi.vec}")

print("\n=== Lab Complete ===")
```

### Lab Exercises

1. Implement a function that checks if a matrix is self-adjoint (A = A†).

2. Verify the kernel/range relations numerically for random matrices.

3. Create a class that represents quantum operators with automatic adjoint computation.

4. Explore what happens to eigenvalues when you compute (A + A†)/2 for non-Hermitian A.

---

## ✅ Daily Checklist

- [ ] Read Axler 7.A on adjoints
- [ ] Understand the defining property ⟨Tv, w⟩ = ⟨v, T*w⟩
- [ ] Practice computing A* for complex matrices
- [ ] Prove at least two properties of adjoints
- [ ] Understand kernel/range relations
- [ ] Complete computational lab
- [ ] Connect to bra-ket notation in QM
- [ ] Create flashcards for key concepts

---

## 📓 Reflection Questions

1. Why is the adjoint defined the way it is (transferring across inner product)?

2. Why does (AB)* = B*A* (reverse order) make sense geometrically?

3. How does the adjoint relate kets to bras in quantum mechanics?

4. What special property would an operator have if T* = T?

---

## 🔜 Preview: Tomorrow's Topics

**Day 114: Hermitian (Self-Adjoint) Operators**

Tomorrow we'll explore operators satisfying A = A†:
- Definition and examples
- Real eigenvalues theorem
- Orthogonal eigenvectors
- Spectral theorem for Hermitian operators
- **QM Connection:** Why all observables must be Hermitian

---

*"The adjoint is the 'transpose' of quantum mechanics — it turns kets into bras and creation into annihilation."*
— Common physics saying
