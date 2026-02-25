# Day 116: The Spectral Theorem — Foundation of Quantum Measurement

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

1. State and prove the spectral theorem for Hermitian operators
2. Understand spectral decomposition: A = Σλᵢ Pᵢ
3. Apply spectral theorem to compute matrix functions
4. Connect spectral decomposition to quantum measurement postulates
5. Work with projection operators and resolutions of identity
6. Compute functions of Hermitian matrices (e.g., e^A, √A)

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 7.A**: Self-Adjoint and Normal Operators (pp. 209-220)
- **Section 7.B**: Spectral Theorem (pp. 221-232)

### Physics Connection
- **Shankar, Chapter 1.8**: Functions of Operators
- **Sakurai, Chapter 1.3**: Measurements, Observables

---

## 🎬 Video Resources

### MIT OCW 18.06
- **Lecture 28**: Spectral Theorem
- **Lecture 29**: Positive Definite Matrices

### Physics Perspective
- Professor M does Science: "Spectral Theorem in Quantum Mechanics"

---

## 📖 Core Content: Theory and Concepts

### 1. Statement of the Spectral Theorem

**Theorem (Spectral Theorem for Hermitian Operators):**
Let A be a Hermitian operator on a finite-dimensional complex inner product space V. Then:

1. All eigenvalues of A are real
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal
3. V has an orthonormal basis consisting of eigenvectors of A
4. A can be written as:
$$\boxed{A = \sum_{i=1}^{n} \lambda_i |e_i\rangle\langle e_i| = \sum_{i=1}^{n} \lambda_i P_i}$$

where {|eᵢ⟩} is an orthonormal eigenbasis and Pᵢ = |eᵢ⟩⟨eᵢ| are projection operators.

### 2. Proof Outline

**Step 1: Real Eigenvalues**
Let A|λ⟩ = λ|λ⟩. Then:
$$\langle\lambda|A|\lambda\rangle = \lambda\langle\lambda|\lambda\rangle$$
$$\langle\lambda|A^\dagger|\lambda\rangle = \lambda^*\langle\lambda|\lambda\rangle$$

Since A = A†: λ = λ*, so λ ∈ ℝ ∎

**Step 2: Orthogonality of Eigenvectors**
Let A|λ⟩ = λ|λ⟩ and A|μ⟩ = μ|μ⟩ with λ ≠ μ.
$$\langle\mu|A|\lambda\rangle = \lambda\langle\mu|\lambda\rangle$$
$$\langle\mu|A^\dagger|\lambda\rangle = \langle A\mu|\lambda\rangle = \mu\langle\mu|\lambda\rangle$$

Since A = A†: λ⟨μ|λ⟩ = μ⟨μ|λ⟩
$$(λ - μ)\langle\mu|\lambda\rangle = 0$$
Since λ ≠ μ: ⟨μ|λ⟩ = 0 ∎

**Step 3: Existence of Orthonormal Eigenbasis**
Uses induction on dimension. Key: A maps the orthogonal complement of any eigenspace to itself.

### 3. Projection Operators

**Definition:** A projection operator P satisfies:
$$P^2 = P \quad \text{(idempotent)}$$
$$P^\dagger = P \quad \text{(Hermitian)}$$

**Properties:**
- Eigenvalues of P are 0 and 1 only
- P projects onto its range (eigenspace for λ=1)
- I - P projects onto the orthogonal complement

**Rank-1 Projection:**
For normalized |ψ⟩, the operator P = |ψ⟩⟨ψ| is a projection onto span{|ψ⟩}:
$$P|v\rangle = |\psi\rangle\langle\psi|v\rangle = \langle\psi|v\rangle|\psi\rangle$$

### 4. Spectral Decomposition

**Resolution of the Identity:**
$$I = \sum_{i=1}^{n} |e_i\rangle\langle e_i| = \sum_{i=1}^{n} P_i$$

**Spectral Decomposition of A:**
$$A = \sum_{i=1}^{n} \lambda_i P_i$$

where Pᵢ = |eᵢ⟩⟨eᵢ| projects onto the eigenspace of λᵢ.

**Verification:**
$$A|e_j\rangle = \sum_{i} \lambda_i P_i|e_j\rangle = \sum_{i} \lambda_i \langle e_i|e_j\rangle|e_i\rangle = \lambda_j|e_j\rangle \checkmark$$

### 5. Functions of Hermitian Operators

**Key Insight:** If A = ΣλᵢPᵢ, then for any function f:
$$\boxed{f(A) = \sum_{i} f(\lambda_i) P_i}$$

**Examples:**

**Matrix Square Root:**
If A is positive semidefinite (all λᵢ ≥ 0):
$$\sqrt{A} = \sum_{i} \sqrt{\lambda_i} P_i$$

**Matrix Exponential:**
$$e^A = \sum_{i} e^{\lambda_i} P_i$$

**Matrix Inverse (if all λᵢ ≠ 0):**
$$A^{-1} = \sum_{i} \frac{1}{\lambda_i} P_i$$

**Matrix Powers:**
$$A^n = \sum_{i} \lambda_i^n P_i$$

### 6. Computing Spectral Decomposition

**Algorithm:**
1. Find eigenvalues by solving det(A - λI) = 0
2. For each eigenvalue, find orthonormal eigenvectors
3. Form projection operators Pᵢ = |eᵢ⟩⟨eᵢ|
4. Verify: A = ΣλᵢPᵢ

**Example:** Decompose the Pauli-Z matrix
$$\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

Eigenvalues: λ₁ = +1, λ₂ = -1
Eigenvectors: |+⟩ = (1,0)ᵀ, |-⟩ = (0,1)ᵀ

Projections:
$$P_+ = |+\rangle\langle+| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$
$$P_- = |-\rangle\langle-| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

Verification:
$$\sigma_z = (+1)P_+ + (-1)P_- = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} - \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \checkmark$$

---

## 🔬 Quantum Mechanics Connection

### The Measurement Postulate

**Postulate:** When measuring observable A on state |ψ⟩:

1. **Possible outcomes:** Eigenvalues {λᵢ} of A
2. **Probability:** P(λᵢ) = |⟨eᵢ|ψ⟩|² = ⟨ψ|Pᵢ|ψ⟩
3. **Post-measurement state:** |eᵢ⟩ (collapse to eigenstate)

**Expectation Value via Spectral Decomposition:**
$$\langle A \rangle = \langle\psi|A|\psi\rangle = \sum_i \lambda_i \langle\psi|P_i|\psi\rangle = \sum_i \lambda_i P(\lambda_i)$$

This is exactly the statistical expectation!

### Example: Measuring σ_z on |+x⟩

State: |+x⟩ = (|+z⟩ + |-z⟩)/√2

Using spectral decomposition of σ_z:
- P(+1) = |⟨+z|+x⟩|² = |1/√2|² = 1/2
- P(-1) = |⟨-z|+x⟩|² = |1/√2|² = 1/2

Expected value:
$$\langle\sigma_z\rangle = (+1)(1/2) + (-1)(1/2) = 0$$

### Uncertainty Principle from Spectral Theory

For non-commuting observables A and B:
$$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[A,B]\rangle|$$

The spectral theorem shows that [A,B] ≠ 0 means A and B cannot share a complete set of eigenvectors, hence cannot both have definite values simultaneously.

---

## ✏️ Worked Examples

### Example 1: Full Spectral Decomposition

Decompose:
$$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

**Step 1: Eigenvalues**
$$\det(A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda-1)(\lambda-3)$$
Eigenvalues: λ₁ = 1, λ₂ = 3

**Step 2: Eigenvectors**
λ = 1: (A - I)v = 0 → v₁ = (1, -1)ᵀ/√2
λ = 3: (A - 3I)v = 0 → v₂ = (1, 1)ᵀ/√2

**Step 3: Projections**
$$P_1 = |v_1\rangle\langle v_1| = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$
$$P_2 = |v_2\rangle\langle v_2| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**Step 4: Verify**
$$A = 1 \cdot P_1 + 3 \cdot P_2 = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} + \frac{3}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} \checkmark$$

### Example 2: Computing e^A

Using the spectral decomposition from Example 1:
$$e^A = e^1 P_1 + e^3 P_2$$
$$= \frac{e}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} + \frac{e^3}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$
$$= \frac{1}{2}\begin{pmatrix} e + e^3 & -e + e^3 \\ -e + e^3 & e + e^3 \end{pmatrix}$$

### Example 3: Quantum Measurement

Given Hamiltonian:
$$H = \hbar\omega\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \hbar\omega\sigma_z$$

and initial state |ψ⟩ = (3|0⟩ + 4i|1⟩)/5.

Find energy measurement probabilities and post-measurement states.

**Solution:**
Eigenvalues: E₊ = +ℏω, E₋ = -ℏω
Eigenstates: |0⟩, |1⟩

P(E₊) = |⟨0|ψ⟩|² = |3/5|² = 9/25
P(E₋) = |⟨1|ψ⟩|² = |4i/5|² = 16/25

Post-measurement:
- If E₊ measured → |0⟩
- If E₋ measured → |1⟩

Expected energy:
⟨H⟩ = (9/25)(+ℏω) + (16/25)(-ℏω) = -7ℏω/25

---

## 📝 Practice Problems

### Level 1: Basic Spectral Decomposition
1. Find the spectral decomposition of σₓ = [[0,1],[1,0]].

2. Show that P₁ + P₂ = I for any spectral decomposition of a 2×2 Hermitian matrix.

3. Compute A⁵ where A = [[3,1],[1,3]] using spectral decomposition.

### Level 2: Matrix Functions
4. Compute √A for A = [[5,4],[4,5]].

5. Find e^(iπσₓ). What quantum gate is this?

6. For A = [[2,1],[1,2]], compute cos(A) using the spectral theorem.

### Level 3: Quantum Applications
7. A spin-½ particle is in state |ψ⟩ = (|↑⟩ + 2|↓⟩)/√5. Find:
   - Probability of measuring Sₓ = +ℏ/2
   - Expected value ⟨Sₓ⟩
   - State after measuring Sₓ = +ℏ/2

8. Show that for any Hermitian A: e^(iA) is unitary.

9. Prove: If [A, B] = 0 and both are Hermitian, they share a common eigenbasis.

### Level 4: Proofs
10. Prove that trace(A) = Σλᵢ using spectral decomposition.

11. Show that det(e^A) = e^(tr(A)).

12. Prove the spectral theorem implies Hermitian matrices are unitarily diagonalizable.

---

## 💻 Evening Computational Lab

```python
import numpy as np
from scipy.linalg import expm, sqrtm
import matplotlib.pyplot as plt

# ============================================
# Spectral Decomposition Implementation
# ============================================

def spectral_decomposition(A):
    """
    Compute spectral decomposition of Hermitian matrix.
    Returns eigenvalues, eigenvectors, and projection operators.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    projections = []
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i:i+1]
        P = v @ v.conj().T
        projections.append(P)
    return eigenvalues, eigenvectors, projections

def verify_spectral_decomposition(A, eigenvalues, projections):
    """Verify A = Σ λᵢ Pᵢ"""
    A_reconstructed = sum(lam * P for lam, P in zip(eigenvalues, projections))
    return np.allclose(A, A_reconstructed)

# Example: Pauli-X matrix
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
eigenvalues, eigenvectors, projections = spectral_decomposition(sigma_x)

print("=== Spectral Decomposition of σₓ ===")
print(f"Eigenvalues: {eigenvalues}")
print(f"\nEigenvectors:\n{eigenvectors}")
print(f"\nProjection P₊:\n{projections[0]}")
print(f"\nProjection P₋:\n{projections[1]}")
print(f"\nVerification: {verify_spectral_decomposition(sigma_x, eigenvalues, projections)}")

# ============================================
# Matrix Functions via Spectral Theorem
# ============================================

def matrix_function(A, f):
    """Compute f(A) using spectral decomposition"""
    eigenvalues, _, projections = spectral_decomposition(A)
    result = sum(f(lam) * P for lam, P in zip(eigenvalues, projections))
    return result

# Example: Matrix exponential
A = np.array([[2, 1], [1, 2]], dtype=complex)
exp_A_spectral = matrix_function(A, np.exp)
exp_A_scipy = expm(A)

print("\n=== Matrix Exponential ===")
print(f"Via spectral theorem:\n{exp_A_spectral}")
print(f"\nVia scipy:\n{exp_A_scipy}")
print(f"\nMatch: {np.allclose(exp_A_spectral, exp_A_scipy)}")

# Matrix square root
sqrt_A_spectral = matrix_function(A, np.sqrt)
sqrt_A_scipy = sqrtm(A)

print("\n=== Matrix Square Root ===")
print(f"√A via spectral:\n{sqrt_A_spectral}")
print(f"(√A)² = A: {np.allclose(sqrt_A_spectral @ sqrt_A_spectral, A)}")

# ============================================
# Quantum Measurement Simulation
# ============================================

def measure_observable(A, psi):
    """
    Simulate measurement of observable A on state |ψ⟩.
    Returns probabilities and expected value.
    """
    eigenvalues, eigenvectors, projections = spectral_decomposition(A)
    
    probabilities = []
    for P in projections:
        prob = np.real(psi.conj().T @ P @ psi)[0, 0]
        probabilities.append(prob)
    
    expectation = np.real(psi.conj().T @ A @ psi)[0, 0]
    
    return eigenvalues, probabilities, expectation

# State |ψ⟩ = (|0⟩ + |1⟩)/√2
psi = np.array([[1], [1]], dtype=complex) / np.sqrt(2)

# Measure σz
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
eigenvals, probs, exp_val = measure_observable(sigma_z, psi)

print("\n=== Quantum Measurement of σz on |+⟩ ===")
print(f"Possible outcomes: {eigenvals}")
print(f"Probabilities: {probs}")
print(f"Expected value: {exp_val}")

# Measure σx
eigenvals_x, probs_x, exp_val_x = measure_observable(sigma_x, psi)
print(f"\n=== Measurement of σx on |+⟩ ===")
print(f"Possible outcomes: {eigenvals_x}")
print(f"Probabilities: {probs_x}")
print(f"Expected value: {exp_val_x}")

# ============================================
# Time Evolution via Spectral Decomposition
# ============================================

def time_evolution_operator(H, t, hbar=1):
    """Compute U(t) = exp(-iHt/ℏ) via spectral decomposition"""
    return matrix_function(H, lambda E: np.exp(-1j * E * t / hbar))

# Two-level system Hamiltonian
omega = 1.0
H = omega * sigma_z

# Evolve |+x⟩ state
psi_0 = np.array([[1], [1]], dtype=complex) / np.sqrt(2)

t_values = np.linspace(0, 4*np.pi, 200)
expectation_z = []
expectation_x = []

for t in t_values:
    U = time_evolution_operator(H, t)
    psi_t = U @ psi_0
    
    exp_z = np.real(psi_t.conj().T @ sigma_z @ psi_t)[0, 0]
    exp_x = np.real(psi_t.conj().T @ sigma_x @ psi_t)[0, 0]
    
    expectation_z.append(exp_z)
    expectation_x.append(exp_x)

plt.figure(figsize=(10, 5))
plt.plot(t_values, expectation_z, 'b-', label='⟨σz⟩')
plt.plot(t_values, expectation_x, 'r-', label='⟨σx⟩')
plt.xlabel('Time (ℏ/ω)')
plt.ylabel('Expectation Value')
plt.title('Larmor Precession: Time Evolution under H = ωσz')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.savefig('larmor_precession.png', dpi=150)
plt.show()

# ============================================
# Resolution of Identity Check
# ============================================

print("\n=== Resolution of Identity ===")
for name, A in [("σx", sigma_x), ("σy", np.array([[0, -1j], [1j, 0]])), ("σz", sigma_z)]:
    _, _, projs = spectral_decomposition(A)
    identity_check = sum(projs)
    print(f"{name}: ΣPᵢ = I? {np.allclose(identity_check, np.eye(2))}")
```

---

## ✅ Daily Checklist

- [ ] Read Axler 7.A-7.B on spectral theorem
- [ ] Understand proof of real eigenvalues for Hermitian
- [ ] Understand orthogonality of eigenvectors
- [ ] Compute spectral decomposition by hand (2×2)
- [ ] Apply to matrix functions (e^A, √A)
- [ ] Connect to quantum measurement postulate
- [ ] Complete computational lab
- [ ] Solve at least 6 practice problems

---

## 🔜 Preview: Tomorrow

**Day 117: Normal Operators and Applications**
- Normal operators: AA* = A*A
- When can matrices be unitarily diagonalized?
- Applications: quantum gates, density matrices
- Simultaneous diagonalization

---

*"The spectral theorem is the cornerstone of quantum mechanics — it tells us that physical measurements yield real numbers and that measuring an observable puts the system in a definite state."*
— A Quantum Physicist
