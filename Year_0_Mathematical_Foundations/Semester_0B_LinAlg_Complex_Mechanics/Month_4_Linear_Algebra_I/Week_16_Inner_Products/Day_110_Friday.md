# Day 110: Orthonormal Bases and Best Approximation

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Orthonormal Bases |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Work fluently with orthonormal bases
2. Apply Parseval's identity and Bessel's inequality
3. Understand least squares as orthogonal projection
4. Solve best approximation problems
5. Connect to Fourier series and quantum completeness

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 6.B**: Orthonormal Bases (complete, pp. 193-210)
- **Section 6.C**: Orthogonal Complements and Minimization Problems (pp. 211-220)

### Secondary Reading
- **Strang, Chapter 4.3**: Least Squares Approximations
- **Shankar, Chapter 1.4**: Completeness Relations

---

## 📖 Core Content: Theory and Concepts

### 1. Orthonormal Bases: The Gold Standard

**Definition:** An **orthonormal basis** for V is a basis {e₁, ..., eₙ} with:
$$\langle e_i | e_j \rangle = \delta_{ij}$$

**Why orthonormal bases are superior:**

| Operation | General Basis | Orthonormal Basis |
|-----------|---------------|-------------------|
| Find coordinates | Solve linear system | Inner products: cᵢ = ⟨eᵢ\|v⟩ |
| Compute norm | Matrix multiplication | Sum: ‖v‖² = Σ\|cᵢ\|² |
| Inner product | Bilinear form | Simple: ⟨u\|v⟩ = Σūᵢvᵢ |
| Projection | Matrix inverse | Direct formula |

### 2. Coordinate Formula for Orthonormal Bases

**Theorem:** If {e₁, ..., eₙ} is an orthonormal basis, then every v ∈ V can be written as:
$$v = \sum_{i=1}^n \langle e_i | v \rangle e_i$$

**Proof:** Write v = Σcᵢeᵢ. Take inner product with eⱼ:
$$\langle e_j | v \rangle = \sum_i c_i \langle e_j | e_i \rangle = \sum_i c_i \delta_{ji} = c_j \quad \blacksquare$$

**In Dirac notation:**
$$|v\rangle = \sum_i |e_i\rangle \langle e_i | v \rangle = \sum_i \langle e_i | v \rangle |e_i\rangle$$

### 3. The Completeness Relation

**For orthonormal basis {|eᵢ⟩}:**
$$\sum_i |e_i\rangle \langle e_i| = I$$

This is the **completeness relation** or **resolution of identity**.

**Why it works:** Apply both sides to arbitrary |v⟩:
$$\left(\sum_i |e_i\rangle \langle e_i|\right)|v\rangle = \sum_i |e_i\rangle \langle e_i|v\rangle = |v\rangle \quad \checkmark$$

### 4. Parseval's Identity

**Theorem (Parseval):** If {e₁, ..., eₙ} is orthonormal and v = Σcᵢeᵢ, then:
$$\|v\|^2 = \sum_{i=1}^n |c_i|^2 = \sum_{i=1}^n |\langle e_i | v \rangle|^2$$

**Proof:**
$$\|v\|^2 = \langle v | v \rangle = \left\langle \sum_i c_i e_i \,\middle|\, \sum_j c_j e_j \right\rangle = \sum_i \sum_j \bar{c}_i c_j \langle e_i | e_j \rangle = \sum_i |c_i|^2 \quad \blacksquare$$

**Physical meaning:** Total probability = sum of squared amplitudes = 1

### 5. Bessel's Inequality

**Theorem (Bessel):** For any orthonormal set {e₁, ..., eₖ} (not necessarily a basis) and any v:
$$\sum_{i=1}^k |\langle e_i | v \rangle|^2 \leq \|v\|^2$$

with equality if and only if v ∈ span{e₁, ..., eₖ}.

**Proof:** Let w = Σⱼ ⟨eⱼ|v⟩ eⱼ be the projection of v onto span{e₁,...,eₖ}.
Then v - w ⊥ eᵢ for all i, and:
$$\|v\|^2 = \|w\|^2 + \|v-w\|^2 \geq \|w\|^2 = \sum_i |\langle e_i | v \rangle|^2 \quad \blacksquare$$

### 6. Orthogonal Projection onto Subspace

**Theorem:** Let W be a finite-dimensional subspace with orthonormal basis {e₁,...,eₖ}. The orthogonal projection of v onto W is:
$$P_W(v) = \sum_{i=1}^k \langle e_i | v \rangle e_i$$

**Properties:**
- P_W(v) ∈ W
- v - P_W(v) ⊥ W
- P_W² = P_W (idempotent)
- P_W† = P_W (Hermitian)

### 7. Best Approximation Theorem

**Theorem:** Let W be a finite-dimensional subspace of V. For any v ∈ V:
$$P_W(v) = \arg\min_{w \in W} \|v - w\|$$

The orthogonal projection is the **closest point** in W to v!

**Proof:** For any w ∈ W:
$$\|v - w\|^2 = \|(v - P_W(v)) + (P_W(v) - w)\|^2$$

Since (v - P_W(v)) ⊥ W and (P_W(v) - w) ∈ W:
$$= \|v - P_W(v)\|^2 + \|P_W(v) - w\|^2 \geq \|v - P_W(v)\|^2$$

Equality holds iff w = P_W(v). ∎

### 8. Least Squares Problem

**Problem:** Find x that minimizes ‖Ax - b‖ where A is m×n with m > n.

**Solution:** The minimizer satisfies the **normal equations**:
$$A^\dagger A x = A^\dagger b$$

**Why?** We want Ax = P_{col(A)}(b), the projection of b onto the column space of A.

**QR approach:** If A = QR, then:
$$x = R^{-1} Q^\dagger b$$

(Much more numerically stable than normal equations!)

---

## 🔬 Quantum Mechanics Connection

### Completeness and Measurement

The completeness relation Σᵢ |i⟩⟨i| = I is fundamental to QM:

**Probability conservation:**
$$1 = \langle\psi|\psi\rangle = \langle\psi| \left(\sum_i |i\rangle\langle i|\right) |\psi\rangle = \sum_i |\langle i|\psi\rangle|^2 = \sum_i P(i)$$

### Parseval = Probability = 1

For normalized state |ψ⟩ = Σᵢ cᵢ|i⟩:
$$\sum_i |c_i|^2 = 1$$

This is just Parseval's identity for ‖ψ‖ = 1!

### Incomplete Measurement

If we only measure some outcomes {|1⟩, ..., |k⟩}, Bessel's inequality says:
$$\sum_{i=1}^k P(i) \leq 1$$

The "missing probability" goes to unmeasured outcomes.

### State Estimation and Tomography

**Quantum state tomography** reconstructs |ψ⟩ from measurements:
$$c_i = \langle e_i | \psi \rangle$$

Measure in enough bases to determine all coefficients.

### Approximating States

If we can only prepare states in subspace W, the best approximation to target |ψ⟩ is:
$$|\psi_{\text{best}}\rangle = P_W |\psi\rangle$$

This minimizes the trace distance (approximately).

---

## ✏️ Worked Examples

### Example 1: Coordinates in Orthonormal Basis

Find coordinates of v = (1, 2, 3) in the orthonormal basis:
- e₁ = (1, 0, 0)
- e₂ = (0, 1, 0)  
- e₃ = (0, 0, 1)

**Solution:** 
$$c_1 = \langle e_1 | v \rangle = 1, \quad c_2 = \langle e_2 | v \rangle = 2, \quad c_3 = \langle e_3 | v \rangle = 3$$

So [v]_B = (1, 2, 3). (For standard basis, coordinates = components!)

### Example 2: Non-Standard Orthonormal Basis

Find coordinates of v = (3, 4) in orthonormal basis:
- e₁ = (3/5, 4/5)
- e₂ = (-4/5, 3/5)

**Solution:**
$$c_1 = \langle e_1 | v \rangle = \frac{3}{5}(3) + \frac{4}{5}(4) = \frac{9 + 16}{5} = 5$$
$$c_2 = \langle e_2 | v \rangle = -\frac{4}{5}(3) + \frac{3}{5}(4) = \frac{-12 + 12}{5} = 0$$

So v = 5e₁ + 0e₂, or [v]_B = (5, 0).

**Verify Parseval:** ‖v‖² = 9 + 16 = 25 = 5² + 0² ✓

### Example 3: Projection via Orthonormal Basis

Project v = (1, 2, 3) onto W = span{(1,1,0)/√2, (1,-1,0)/√2}.

Let e₁ = (1,1,0)/√2, e₂ = (1,-1,0)/√2.

$$\langle e_1 | v \rangle = \frac{1}{\sqrt{2}}(1 + 2) = \frac{3}{\sqrt{2}}$$
$$\langle e_2 | v \rangle = \frac{1}{\sqrt{2}}(1 - 2) = \frac{-1}{\sqrt{2}}$$

$$P_W(v) = \frac{3}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1,1,0) + \frac{-1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1,-1,0)$$
$$= \frac{3}{2}(1,1,0) - \frac{1}{2}(1,-1,0) = (1, 2, 0)$$

**Verify:** v - P_W(v) = (0, 0, 3) ⊥ W ✓

### Example 4: Bessel's Inequality

Let v = (1, 2, 3, 4) and take orthonormal set {e₁, e₂} where:
- e₁ = (1, 0, 0, 0)
- e₂ = (0, 1, 0, 0)

$$|\langle e_1 | v \rangle|^2 + |\langle e_2 | v \rangle|^2 = 1 + 4 = 5$$
$$\|v\|^2 = 1 + 4 + 9 + 16 = 30$$

**Check Bessel:** 5 ≤ 30 ✓ (strict inequality since {e₁, e₂} doesn't span ℝ⁴)

### Example 5: Least Squares

Find the best-fit line y = ax + b to points (0, 1), (1, 2), (2, 4).

**Setup:** We want:
$$\begin{pmatrix} 0 & 1 \\ 1 & 1 \\ 2 & 1 \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} \approx \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}$$

**Normal equations:** AᵀAx = Aᵀb

$$A^T A = \begin{pmatrix} 5 & 3 \\ 3 & 3 \end{pmatrix}, \quad A^T b = \begin{pmatrix} 10 \\ 7 \end{pmatrix}$$

Solving: a = 3/2, b = 2/3.

Best-fit line: y = (3/2)x + 2/3.

### Example 6: Quantum Measurement Completeness

Verify completeness for qubit computational basis:

$$|0\rangle\langle 0| + |1\rangle\langle 1| = \begin{pmatrix} 1 \\ 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \end{pmatrix}\begin{pmatrix} 0 & 1 \end{pmatrix}$$
$$= \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I \quad \checkmark$$

---

## 📝 Practice Problems

### Level 1: Orthonormal Basis Computations

1. Find coordinates of (2, -1, 3) in the standard orthonormal basis of ℝ³.

2. Verify Parseval's identity for v = (3, 4) in the basis {(3/5, 4/5), (-4/5, 3/5)}.

3. Show that {|+⟩, |-⟩} satisfies the completeness relation.

4. Find the orthogonal projection of (1, 1, 1, 1) onto span{(1,0,0,0), (0,1,0,0)}.

### Level 2: Bessel and Parseval

5. Verify Bessel's inequality for v = (1, 2, 3) and orthonormal set {(1,0,0), (0,1,0)}.

6. When does Bessel's inequality become equality?

7. Use Parseval to compute ‖(1+i, 2-i, 3)‖ via coordinates.

8. Prove: If {eᵢ} is orthonormal, then ‖Σcᵢeᵢ‖² = Σ|cᵢ|².

### Level 3: Projections and Best Approximation

9. Find the closest point in the plane x + y + z = 0 to the point (1, 2, 3).

10. Project the polynomial p(x) = x³ onto span{1, x, x²} using L²[-1,1] inner product.

11. Find the least squares solution to:
    $$\begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} x = \begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix}$$

12. Show that the error vector b - Ax is orthogonal to the column space of A.

### Level 4: Theory and Quantum Applications

13. Prove that P² = P for orthogonal projection P.

14. Prove that ‖Pv‖ ≤ ‖v‖ with equality iff v ∈ range(P).

15. For qubit state |ψ⟩ = α|0⟩ + β|1⟩, verify Σᵢ|⟨i|ψ⟩|² = 1 in both Z and X bases.

16. Show that changing measurement basis doesn't change total probability.

---

## 📊 Answers and Hints

1. Coordinates are (2, -1, 3) for standard basis
2. ‖v‖² = 25 = 5² + 0² ✓
3. Compute |+⟩⟨+| + |-⟩⟨-| = I
4. Projection = (1, 1, 0, 0)
5. 1 + 4 = 5 ≤ 14 ✓
6. When v ∈ span{eᵢ}
7. ‖v‖² = |1+i|² + |2-i|² + 9 = 2 + 5 + 9 = 16
8. Direct expansion of inner product
9. Project onto normal direction, subtract
10. Use ⟨xⁿ, xᵐ⟩ = 2/(n+m+1) for n+m even, 0 for odd
11. Use normal equations
12. Geometry of projection
13. P²v = P(Pv) = Pv since Pv ∈ W
14. ‖v‖² = ‖Pv‖² + ‖v-Pv‖²
15. Direct calculation in each basis
16. Uses completeness relation

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

# ============================================
# Lab 1: Orthonormal Basis Coordinates
# ============================================

def coords_in_orthonormal_basis(v, basis):
    """Find coordinates of v in orthonormal basis"""
    return np.array([np.vdot(e, v) for e in basis])

def reconstruct_from_coords(coords, basis):
    """Reconstruct vector from coordinates"""
    return sum(c * e for c, e in zip(coords, basis))

# Example: non-standard basis for R²
e1 = np.array([3/5, 4/5])
e2 = np.array([-4/5, 3/5])
basis = [e1, e2]

v = np.array([3.0, 4.0])
coords = coords_in_orthonormal_basis(v, basis)
v_reconstructed = reconstruct_from_coords(coords, basis)

print("=== Orthonormal Basis Coordinates ===")
print(f"v = {v}")
print(f"Coordinates in {{e₁, e₂}}: {coords}")
print(f"Reconstructed: {v_reconstructed}")

# Verify Parseval
print(f"\nParseval's Identity:")
print(f"||v||² = {np.linalg.norm(v)**2}")
print(f"Σ|cᵢ|² = {np.sum(np.abs(coords)**2)}")

# ============================================
# Lab 2: Completeness Relation
# ============================================

def completeness_check(basis, dim):
    """Check if basis satisfies completeness relation"""
    identity = np.zeros((dim, dim), dtype=complex)
    for e in basis:
        e = np.array(e).reshape(-1, 1)
        identity += e @ e.conj().T
    return identity

# Computational basis
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

I_comp = completeness_check([ket_0, ket_1], 2)
print("\n=== Completeness Relation ===")
print(f"Computational basis:")
print(f"|0⟩⟨0| + |1⟩⟨1| = \n{I_comp}")

# Hadamard basis
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

I_had = completeness_check([ket_plus, ket_minus], 2)
print(f"\nHadamard basis:")
print(f"|+⟩⟨+| + |-⟩⟨-| = \n{I_had}")

# ============================================
# Lab 3: Bessel's Inequality
# ============================================

def bessel_check(v, orthonormal_set):
    """Verify Bessel's inequality"""
    sum_sq = sum(np.abs(np.vdot(e, v))**2 for e in orthonormal_set)
    norm_sq = np.linalg.norm(v)**2
    print(f"Σ|⟨eᵢ|v⟩|² = {sum_sq:.6f}")
    print(f"||v||² = {norm_sq:.6f}")
    print(f"Bessel satisfied: {sum_sq <= norm_sq + 1e-10}")
    return sum_sq, norm_sq

v = np.array([1, 2, 3, 4])
partial_basis = [np.array([1,0,0,0]), np.array([0,1,0,0])]

print("\n=== Bessel's Inequality ===")
bessel_check(v, partial_basis)

# ============================================
# Lab 4: Orthogonal Projection
# ============================================

def project_onto_subspace(v, orthonormal_basis):
    """Project v onto span of orthonormal basis"""
    proj = np.zeros_like(v, dtype=complex)
    for e in orthonormal_basis:
        proj += np.vdot(e, v) * e
    return proj

# Project (1,2,3) onto xy-plane
v = np.array([1.0, 2.0, 3.0])
e1 = np.array([1.0, 0.0, 0.0])
e2 = np.array([0.0, 1.0, 0.0])

proj = project_onto_subspace(v, [e1, e2])
residual = v - proj

print("\n=== Orthogonal Projection ===")
print(f"v = {v}")
print(f"Projection onto xy-plane: {proj}")
print(f"Residual: {residual}")
print(f"Residual ⊥ plane: {np.allclose([np.dot(residual, e1), np.dot(residual, e2)], 0)}")

# ============================================
# Lab 5: Least Squares
# ============================================

# Fit line y = ax + b to data
x_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([1.1, 2.0, 2.9, 4.2, 4.8])

# Design matrix
A = np.column_stack([x_data, np.ones_like(x_data)])
b = y_data

# Method 1: Normal equations
x_normal = np.linalg.solve(A.T @ A, A.T @ b)

# Method 2: QR decomposition
Q, R = qr(A, mode='economic')
x_qr = np.linalg.solve(R, Q.T @ b)

# Method 3: NumPy lstsq
x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print("\n=== Least Squares ===")
print(f"Normal equations: a = {x_normal[0]:.4f}, b = {x_normal[1]:.4f}")
print(f"QR decomposition: a = {x_qr[0]:.4f}, b = {x_qr[1]:.4f}")
print(f"NumPy lstsq:      a = {x_lstsq[0]:.4f}, b = {x_lstsq[1]:.4f}")

# Verify error orthogonality
error = b - A @ x_lstsq
print(f"\nError orthogonal to columns of A:")
print(f"A.T @ error = {A.T @ error}")

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_data, y_data, s=100, label='Data')
x_line = np.linspace(-0.5, 4.5, 100)
plt.plot(x_line, x_lstsq[0]*x_line + x_lstsq[1], 'r-', label=f'Fit: y = {x_lstsq[0]:.2f}x + {x_lstsq[1]:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Least Squares Line Fit')
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(1, 2, 2)
plt.stem(x_data, error)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('x')
plt.ylabel('Residual')
plt.title('Residuals (should be random around 0)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('least_squares.png', dpi=150)
plt.show()

# ============================================
# Lab 6: Quantum Measurement Probabilities
# ============================================

def measurement_probs(psi, basis, labels=None):
    """Compute measurement probabilities in given basis"""
    probs = [np.abs(np.vdot(e, psi))**2 for e in basis]
    if labels is None:
        labels = [f"|{i}⟩" for i in range(len(basis))]
    
    print("Measurement probabilities:")
    for label, prob in zip(labels, probs):
        print(f"  P({label}) = {prob:.6f}")
    print(f"  Total: {sum(probs):.6f}")
    return probs

# Create a state
psi = np.array([1, 1j], dtype=complex) / np.sqrt(2)

print("\n=== Quantum Measurement (Same State, Different Bases) ===")
print(f"|ψ⟩ = {psi}")

print("\nZ-basis (computational):")
measurement_probs(psi, [ket_0, ket_1], ["|0⟩", "|1⟩"])

print("\nX-basis (Hadamard):")
measurement_probs(psi, [ket_plus, ket_minus], ["|+⟩", "|-⟩"])

# Y-basis
ket_plus_i = (ket_0 + 1j*ket_1) / np.sqrt(2)
ket_minus_i = (ket_0 - 1j*ket_1) / np.sqrt(2)
print("\nY-basis:")
measurement_probs(psi, [ket_plus_i, ket_minus_i], ["|+i⟩", "|-i⟩"])

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Read Axler 6.B-C on orthonormal bases
- [ ] Master coordinate formula for orthonormal bases
- [ ] Understand and apply Parseval's identity
- [ ] Know Bessel's inequality and when equality holds
- [ ] Solve best approximation problems
- [ ] Complete least squares examples
- [ ] Complete computational lab

---

## 📓 Reflection Questions

1. Why are orthonormal bases so much easier to work with?

2. How does Parseval's identity relate to probability conservation?

3. What is the geometric meaning of the least squares solution?

4. Why does completeness of measurement basis guarantee probability = 1?

---

## 🔜 Preview: Tomorrow's Topics

**Day 111: Computational Lab — Inner Products & QM**
- Extensive Python implementations
- Quantum state manipulation
- Fourier analysis connection
- Visualization projects

---

*"The orthonormal basis is to linear algebra what the standard unit is to measurement."*
— Anonymous mathematician
