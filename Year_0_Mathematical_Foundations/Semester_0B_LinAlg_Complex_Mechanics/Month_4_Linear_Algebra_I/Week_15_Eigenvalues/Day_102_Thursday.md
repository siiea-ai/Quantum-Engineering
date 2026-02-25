# Day 102: The Spectral Theorem and Applications

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Spectral Theorem |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State the spectral theorem for Hermitian/symmetric matrices
2. Understand orthogonal/unitary diagonalization
3. Apply spectral decomposition to compute matrix functions
4. Classify quadratic forms using eigenvalues
5. Understand the spectral decomposition of quantum observables
6. Apply the spectral theorem to measurement theory

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 7.A**: Self-Adjoint and Normal Operators (pp. 209-220)
- **Section 7.B**: Spectral Theorem (pp. 221-228)

### Secondary
- **Shankar, Chapter 1.8-1.9**: Eigenvalue problem and spectral decomposition

---

## 📖 Core Content: Theory and Concepts

### 1. Special Classes of Matrices

| Type | Definition | Properties |
|------|------------|------------|
| Symmetric | A = Aᵀ (real) | Real eigenvalues, orthogonal eigenvectors |
| Hermitian | A = A† (complex) | Real eigenvalues, orthonormal eigenvectors |
| Orthogonal | QᵀQ = I (real) | \|det\| = 1, preserves lengths |
| Unitary | U†U = I (complex) | \|det\| = 1, preserves inner products |
| Normal | AA† = A†A | Unitarily diagonalizable |

### 2. The Spectral Theorem

**Theorem (Real Spectral Theorem):**
A real symmetric matrix A is orthogonally diagonalizable. That is:
$$A = QDQ^T$$
where Q is orthogonal (Q⁻¹ = Qᵀ) and D is diagonal with real entries.

**Theorem (Complex Spectral Theorem):**
A Hermitian matrix A is unitarily diagonalizable. That is:
$$A = UDU^\dagger$$
where U is unitary (U⁻¹ = U†) and D is diagonal with real entries.

**Why eigenvalues are real:** For Hermitian A and eigenpair (λ, v):
$$\lambda \langle v, v \rangle = \langle v, \lambda v \rangle = \langle v, Av \rangle = \langle A^\dagger v, v \rangle = \langle Av, v \rangle = \langle \lambda v, v \rangle = \bar{\lambda} \langle v, v \rangle$$

Since ⟨v, v⟩ ≠ 0 (v nonzero), we have λ = λ̄, so λ ∈ ℝ.

**Why eigenvectors are orthogonal:** For distinct eigenvalues λ₁ ≠ λ₂:
$$\lambda_1 \langle v_1, v_2 \rangle = \langle Av_1, v_2 \rangle = \langle v_1, A^\dagger v_2 \rangle = \langle v_1, Av_2 \rangle = \lambda_2 \langle v_1, v_2 \rangle$$

Since λ₁ ≠ λ₂ (and both real): ⟨v₁, v₂⟩ = 0.

### 3. Spectral Decomposition

**Form:** For Hermitian A with eigenvalues λ₁, ..., λₙ and orthonormal eigenvectors |v₁⟩, ..., |vₙ⟩:

$$A = \sum_{i=1}^{n} \lambda_i |v_i\rangle\langle v_i|$$

Each term $P_i = |v_i\rangle\langle v_i|$ is a **projection operator** onto the eigenspace.

**Properties of projection operators:**
- $P_i^2 = P_i$ (idempotent)
- $P_i^\dagger = P_i$ (Hermitian)
- $P_i P_j = 0$ for $i \neq j$ (orthogonal)
- $\sum_i P_i = I$ (resolution of identity)

### 4. Applications of Spectral Decomposition

#### Computing Matrix Functions

If $A = \sum_i \lambda_i P_i$, then for any function f:

$$f(A) = \sum_i f(\lambda_i) P_i$$

**Examples:**
$$A^n = \sum_i \lambda_i^n P_i$$
$$e^A = \sum_i e^{\lambda_i} P_i$$
$$A^{-1} = \sum_i \lambda_i^{-1} P_i \quad (\text{if all } \lambda_i \neq 0)$$
$$\sqrt{A} = \sum_i \sqrt{\lambda_i} P_i \quad (\text{if all } \lambda_i \geq 0)$$

### 5. Quadratic Forms

A **quadratic form** is a function Q: ℝⁿ → ℝ given by:
$$Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} a_{ij} x_i x_j$$

where A can be taken to be symmetric.

**Classification by eigenvalues:**
| Eigenvalues | Classification | Shape |
|-------------|----------------|-------|
| All λᵢ > 0 | Positive definite | Ellipsoid |
| All λᵢ ≥ 0 | Positive semidefinite | Ellipsoid (possibly degenerate) |
| All λᵢ < 0 | Negative definite | Inverted ellipsoid |
| Mixed signs | Indefinite | Hyperboloid/saddle |

**Principal Axis Theorem:** There exists an orthogonal change of variables y = Qᵀx such that:
$$Q(\mathbf{x}) = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2$$

This diagonalizes the quadratic form!

---

## 🔬 Quantum Mechanics Connection

### Observables and the Spectral Theorem

In quantum mechanics, **observables** are Hermitian operators.

The spectral theorem guarantees:
1. **Eigenvalues are real** → Measurement outcomes are real numbers
2. **Eigenvectors are orthonormal** → Distinct outcomes are distinguishable
3. **Spectral decomposition exists** → Complete set of measurement outcomes

### Measurement Postulate (Spectral Form)

For observable A with spectral decomposition:
$$\hat{A} = \sum_i \lambda_i |i\rangle\langle i|$$

When measuring  on state |ψ⟩:
1. **Possible outcomes:** λᵢ
2. **Probability of λᵢ:** P(λᵢ) = |⟨i|ψ⟩|²
3. **Post-measurement state:** |i⟩ (if outcome is λᵢ)

**Expectation value:**
$$\langle \hat{A} \rangle = \langle \psi | \hat{A} | \psi \rangle = \sum_i \lambda_i |\langle i | \psi \rangle|^2 = \sum_i \lambda_i P(\lambda_i)$$

### Example: Spin Measurement

The spin-x operator:
$$S_x = \frac{\hbar}{2}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \frac{\hbar}{2}\sigma_x$$

**Spectral decomposition:**
- Eigenvalues: ±ℏ/2
- Eigenvectors: |+x⟩ = (|↑⟩ + |↓⟩)/√2, |-x⟩ = (|↑⟩ - |↓⟩)/√2

$$S_x = \frac{\hbar}{2}|+x\rangle\langle +x| - \frac{\hbar}{2}|-x\rangle\langle -x|$$

If spin is in state |↑⟩:
- P(+ℏ/2) = |⟨+x|↑⟩|² = 1/2
- P(-ℏ/2) = |⟨-x|↑⟩|² = 1/2

### Commuting Observables

**Theorem:** Two Hermitian operators A and B can be simultaneously diagonalized ⟺ [A, B] = AB - BA = 0.

**Physical meaning:** Commuting observables have a common eigenbasis — they can be measured simultaneously with definite values.

**Example:** Position and momentum don't commute: [x̂, p̂] = iℏ
→ Cannot know both precisely (Heisenberg uncertainty)

### Complete Sets of Commuting Observables (CSCO)

A **CSCO** is a maximal set of commuting observables that uniquely labels basis states.

**Example:** For hydrogen atom:
- Energy Ĥ
- Angular momentum L̂²
- Angular momentum z-component L̂z
- Spin Ŝz

Eigenstates: |n, l, mₗ, mₛ⟩

---

## ✏️ Worked Examples

### Example 1: Spectral Decomposition

Find the spectral decomposition of:
$$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

**Step 1: Find eigenvalues**
det(A - λI) = (2-λ)² - 1 = λ² - 4λ + 3 = (λ-1)(λ-3) = 0
λ₁ = 1, λ₂ = 3

**Step 2: Find normalized eigenvectors**

For λ = 1:
(A - I)v = 0 → v = (1, -1)/√2

For λ = 3:
(A - 3I)v = 0 → v = (1, 1)/√2

**Step 3: Form projection operators**
$$P_1 = |v_1\rangle\langle v_1| = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$

$$P_2 = |v_2\rangle\langle v_2| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**Step 4: Spectral decomposition**
$$A = 1 \cdot P_1 + 3 \cdot P_2 = P_1 + 3P_2$$

**Verify:** P₁ + 3P₂ = (1/2)[[1,-1],[-1,1]] + (3/2)[[1,1],[1,1]] = [[2,1],[1,2]] ✓

### Example 2: Matrix Square Root

Find √A for A = [[5, 4], [4, 5]].

**Step 1: Spectral decomposition**
Eigenvalues: λ = 1, 9
Eigenvectors: v₁ = (1,-1)/√2, v₂ = (1,1)/√2

$$A = 1 \cdot P_1 + 9 \cdot P_2$$

**Step 2: Apply square root**
$$\sqrt{A} = \sqrt{1} \cdot P_1 + \sqrt{9} \cdot P_2 = P_1 + 3P_2$$

$$\sqrt{A} = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} + \frac{3}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

**Verify:** (√A)² = [[2,1],[1,2]]² = [[5,4],[4,5]] = A ✓

### Example 3: Quadratic Form Classification

Classify the quadratic form:
$$Q(x, y) = 2x^2 + 4xy + 5y^2$$

**Step 1: Write as matrix form**
$$Q = \mathbf{x}^T A \mathbf{x}, \quad A = \begin{pmatrix} 2 & 2 \\ 2 & 5 \end{pmatrix}$$

(Note: off-diagonal entries are half the coefficient of xy)

**Step 2: Find eigenvalues**
det(A - λI) = (2-λ)(5-λ) - 4 = λ² - 7λ + 6 = 0
λ = 1, 6

**Step 3: Classify**
Both eigenvalues positive → **Positive definite**

The quadratic form is always positive (except at origin).

### Example 4: Quantum Measurement

A qubit is in state |ψ⟩ = (3|0⟩ + 4i|1⟩)/5.

Compute measurement statistics for observable σ_z.

**Spectral decomposition:**
$$\sigma_z = (+1)|0\rangle\langle 0| + (-1)|1\rangle\langle 1|$$

**Probabilities:**
P(+1) = |⟨0|ψ⟩|² = |3/5|² = 9/25
P(-1) = |⟨1|ψ⟩|² = |4i/5|² = 16/25

**Expectation value:**
⟨σ_z⟩ = (+1)(9/25) + (-1)(16/25) = -7/25

**Verify:** ⟨ψ|σ_z|ψ⟩ = (1/25)[3·3 - 4i·(-4i)] = (9-16)/25 = -7/25 ✓

---

## 📝 Practice Problems

### Level 1: Spectral Decomposition
1. Find the spectral decomposition of [[3,1],[1,3]].

2. Verify that P₁ + P₂ = I for the projectors in problem 1.

3. Is [[1,2],[0,1]] Hermitian? Can you find its spectral decomposition?

4. Find eigenvalues of a 2×2 projection matrix (hint: P² = P).

### Level 2: Applications
5. Use spectral decomposition to compute [[2,1],[1,2]]¹⁰⁰.

6. Find all matrices B such that B² = [[5,4],[4,5]].

7. Classify: Q(x,y) = x² - 2xy + y². (Is it positive definite?)

8. For A = diag(2, -1, 3), compute e^A using spectral decomposition.

### Level 3: Theory
9. Prove: If A is Hermitian and A² = A, then A is a projection (eigenvalues 0 or 1).

10. Prove: Unitary matrices are normal (UU† = U†U).

11. Prove: The product of two commuting Hermitian matrices is Hermitian.

12. Show that tr(A) = Σᵢ λᵢ using spectral decomposition.

### Level 4: Quantum Applications
13. For the Hadamard gate H, find the spectral decomposition and verify H² = I.

14. A qubit state is |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩. Find ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩.

15. Show that [σ_x, σ_y] = 2iσ_z. What does this imply physically?

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# Lab 1: Spectral Decomposition
# ============================================

def spectral_decomposition(A, verbose=True):
    """
    Compute spectral decomposition A = Σ λᵢ Pᵢ
    Returns eigenvalues and projection operators
    """
    if verbose:
        print("=== Spectral Decomposition ===\n")
        print(f"Matrix A:\n{A}\n")
    
    # Check if Hermitian
    is_hermitian = np.allclose(A, A.conj().T)
    if verbose:
        print(f"Is Hermitian: {is_hermitian}")
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)  # Use eigh for Hermitian
    
    if verbose:
        print(f"\nEigenvalues: {eigenvalues}")
        print(f"\nEigenvectors (columns):\n{eigenvectors}\n")
    
    # Compute projection operators
    projectors = []
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i:i+1]  # Column vector
        P = v @ v.conj().T  # Outer product
        projectors.append(P)
        
        if verbose:
            print(f"P_{i+1} = |v_{i+1}⟩⟨v_{i+1}| (λ = {eigenvalues[i]:.4f}):")
            print(P)
            print()
    
    # Verify decomposition
    A_reconstructed = sum(lam * P for lam, P in zip(eigenvalues, projectors))
    if verbose:
        print(f"Reconstruction error: {np.max(np.abs(A - A_reconstructed)):.2e}")
        
        # Verify projector properties
        print("\nProjector properties:")
        for i, P in enumerate(projectors):
            print(f"  P_{i+1}² = P_{i+1}: {np.allclose(P @ P, P)}")
            print(f"  P_{i+1}† = P_{i+1}: {np.allclose(P, P.conj().T)}")
        
        print(f"\n  ΣPᵢ = I: {np.allclose(sum(projectors), np.eye(A.shape[0]))}")
    
    return eigenvalues, projectors

# Test
A = np.array([[2, 1], [1, 2]], dtype=complex)
eigenvalues, projectors = spectral_decomposition(A)

# ============================================
# Lab 2: Matrix Functions via Spectral Theorem
# ============================================

def matrix_function_spectral(A, f, f_name="f"):
    """Compute f(A) using spectral decomposition"""
    print(f"\n=== Computing {f_name}(A) ===\n")
    
    eigenvalues, projectors = spectral_decomposition(A, verbose=False)
    
    # Apply function to eigenvalues
    f_eigenvalues = f(eigenvalues)
    
    print(f"Eigenvalues: {eigenvalues}")
    print(f"{f_name}(eigenvalues): {f_eigenvalues}")
    
    # Reconstruct
    result = sum(f_lam * P for f_lam, P in zip(f_eigenvalues, projectors))
    
    print(f"\n{f_name}(A) =")
    print(result)
    
    return result

# Example: Square root
A = np.array([[5, 4], [4, 5]], dtype=float)
sqrt_A = matrix_function_spectral(A, np.sqrt, "√")

# Verify
print("\nVerification: (√A)² =")
print(sqrt_A @ sqrt_A)
print("Original A =")
print(A)

# Example: Exponential
exp_A = matrix_function_spectral(A, np.exp, "exp")

# Compare with scipy
from scipy.linalg import expm
print("\nComparison with scipy.linalg.expm:")
print(expm(A))

# ============================================
# Lab 3: Quadratic Forms Visualization
# ============================================

def visualize_quadratic_form(A, title="Quadratic Form"):
    """Visualize a 2D quadratic form and its principal axes"""
    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Contour plot
    ax1 = axes[0]
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute Q(x,y) = [x,y] A [x,y]^T
    Z = A[0,0]*X**2 + (A[0,1]+A[1,0])*X*Y + A[1,1]*Y**2
    
    # Contour plot
    levels = np.linspace(Z.min(), Z.max(), 20)
    contour = ax1.contour(X, Y, Z, levels=levels, cmap='coolwarm')
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Draw eigenvectors (principal axes)
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        color = 'green' if lam > 0 else 'red'
        ax1.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.05, 
                  fc=color, ec=color, linewidth=2)
        ax1.text(v[0]*1.2, v[1]*1.2, f'λ={lam:.2f}', fontsize=10)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_title(f'{title}\nContours of Q(x,y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    
    # Right: 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Q(x,y)')
    ax2.set_title(f'Surface plot\nλ = {eigenvalues}')
    
    # Classification
    if all(eigenvalues > 0):
        classification = "Positive Definite"
    elif all(eigenvalues >= 0):
        classification = "Positive Semidefinite"
    elif all(eigenvalues < 0):
        classification = "Negative Definite"
    elif all(eigenvalues <= 0):
        classification = "Negative Semidefinite"
    else:
        classification = "Indefinite"
    
    fig.suptitle(f'{title}: {classification}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'quadratic_{title.replace(" ", "_").lower()}.png', dpi=150)
    plt.show()
    
    return classification

# Various quadratic forms
forms = [
    (np.array([[2, 1], [1, 2]]), "Positive Definite"),
    (np.array([[1, 2], [2, 1]]), "Indefinite"),
    (np.array([[-2, 1], [1, -2]]), "Negative Definite"),
    (np.array([[1, 1], [1, 1]]), "Positive Semidefinite"),
]

for A, name in forms:
    classification = visualize_quadratic_form(A, name)
    print(f"{name}: eigenvalues = {np.linalg.eigvalsh(A)}, classification = {classification}\n")

# ============================================
# Lab 4: Quantum Measurement Statistics
# ============================================

print("\n=== Quantum Measurement Statistics ===\n")

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def measure_observable(psi, A, name="A"):
    """Compute measurement statistics for observable A on state psi"""
    # Normalize state
    psi = psi / np.linalg.norm(psi)
    
    print(f"Observable: {name}")
    print(f"State |ψ⟩: {psi}")
    
    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    print(f"\nPossible outcomes (eigenvalues): {eigenvalues}")
    
    # Compute probabilities
    probabilities = []
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        prob = np.abs(np.vdot(v, psi))**2
        probabilities.append(prob)
        print(f"  P({lam:.4f}) = |⟨v_{i+1}|ψ⟩|² = {prob:.4f}")
    
    # Expectation value
    expectation = np.real(np.vdot(psi, A @ psi))
    print(f"\nExpectation ⟨{name}⟩ = {expectation:.4f}")
    print(f"Check: Σ λᵢ P(λᵢ) = {sum(lam*p for lam, p in zip(eigenvalues, probabilities)):.4f}")
    
    # Variance
    variance = np.real(np.vdot(psi, A @ A @ psi)) - expectation**2
    print(f"Variance Δ{name}² = {variance:.4f}")
    print(f"Standard deviation Δ{name} = {np.sqrt(variance):.4f}")
    
    return eigenvalues, probabilities, expectation

# Test state: |ψ⟩ = (3|0⟩ + 4i|1⟩)/5
psi = np.array([3, 4j], dtype=complex) / 5

for sigma, name in [(sigma_x, "σ_x"), (sigma_y, "σ_y"), (sigma_z, "σ_z")]:
    print("\n" + "="*50)
    measure_observable(psi, sigma, name)

# ============================================
# Lab 5: Commuting Observables
# ============================================

print("\n\n=== Commuting Observables ===\n")

def commutator(A, B):
    """Compute [A, B] = AB - BA"""
    return A @ B - B @ A

# Pauli commutators
print("[σ_x, σ_y] =")
print(commutator(sigma_x, sigma_y))
print(f"= 2i σ_z? {np.allclose(commutator(sigma_x, sigma_y), 2j * sigma_z)}")

print("\n[σ_y, σ_z] =")
print(commutator(sigma_y, sigma_z))
print(f"= 2i σ_x? {np.allclose(commutator(sigma_y, sigma_z), 2j * sigma_x)}")

print("\n[σ_z, σ_x] =")
print(commutator(sigma_z, sigma_x))
print(f"= 2i σ_y? {np.allclose(commutator(sigma_z, sigma_x), 2j * sigma_y)}")

# Commuting example: σ_z and projection |0⟩⟨0|
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
print("\n[σ_z, |0⟩⟨0|] =")
print(commutator(sigma_z, P0))
print("Commute? They share eigenbasis!")

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Read Axler 7.A-7.B on spectral theorem
- [ ] Understand why Hermitian matrices have real eigenvalues
- [ ] Compute spectral decomposition for 2×2 matrices
- [ ] Use spectral decomposition for matrix functions
- [ ] Classify quadratic forms
- [ ] Complete quantum measurement examples
- [ ] Understand commuting observables

---

## 🔜 Preview: Tomorrow's Topics

**Day 103: Applications to Differential Equations and QM**

Tomorrow we'll explore:
- Solving systems of ODEs using eigenvalues
- Stability analysis
- Time evolution in quantum mechanics
- Schrödinger equation solutions

---

*"The spectral theorem is to linear algebra what the fundamental theorem of calculus is to analysis."*
— Peter Lax
