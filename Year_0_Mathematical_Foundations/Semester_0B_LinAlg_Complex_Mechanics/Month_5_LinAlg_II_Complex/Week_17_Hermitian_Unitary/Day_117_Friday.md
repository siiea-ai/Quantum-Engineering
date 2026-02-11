# Day 117: Normal Operators and Applications

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Normal Operators |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define normal operators and recognize their significance
2. Prove the spectral theorem for normal operators
3. Classify operators: Hermitian, unitary, and normal
4. Understand simultaneous diagonalization
5. Apply to commuting observables in QM
6. Work with positive operators and polar decomposition

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 7.C**: Normal Operators on Complex Spaces (pp. 233-240)
- **Section 7.D**: Positive Operators (pp. 241-248)

### Physics Connection
- **Sakurai, Chapter 1.4**: Compatible Observables
- **Nielsen & Chuang, Section 2.2.6**: Polar and singular value decompositions

---

## 📖 Core Content: Theory and Concepts

### 1. Normal Operators

**Definition:** An operator A on an inner product space is **normal** if:
$$\boxed{AA^* = A^*A}$$

**Key insight:** A commutes with its adjoint.

**Important Examples:**
| Operator Type | Condition | Normal? |
|---------------|-----------|---------|
| Hermitian | A = A* | Yes (trivially) |
| Unitary | A*A = AA* = I | Yes |
| Skew-Hermitian | A* = -A | Yes |
| General | — | Often no |

### 2. The Spectral Theorem for Normal Operators

**Theorem:** On a finite-dimensional complex inner product space, an operator is normal if and only if it is unitarily diagonalizable.

$$A \text{ normal} \iff \exists \text{ unitary } U: U^*AU = D \text{ (diagonal)}$$

**Corollary:** Normal operators have orthonormal eigenbases.

**Key Difference from Hermitian:**
- Hermitian: eigenvalues are real
- Unitary: eigenvalues have |λ| = 1
- Normal: eigenvalues can be any complex numbers

### 3. Characterizations of Normal Operators

**Theorem:** The following are equivalent for A on a complex inner product space:

1. A is normal (AA* = A*A)
2. ||Av|| = ||A*v|| for all v
3. A = B + iC where B, C are Hermitian and [B,C] = 0
4. A is unitarily diagonalizable
5. V has an orthonormal basis of eigenvectors of A

**Proof of (1) ⟺ (2):**
$$||Av||^2 = \langle Av, Av\rangle = \langle v, A^*Av\rangle$$
$$||A^*v||^2 = \langle A^*v, A^*v\rangle = \langle v, AA^*v\rangle$$
These are equal for all v ⟺ A*A = AA* ∎

### 4. Non-Normal Operators

**Example:** The shift operator S on ℂ² 
$$S = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$$

$$S^*S = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

$$SS^* = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

S*S ≠ SS*, so S is not normal. 
Consequence: S is not unitarily diagonalizable (it's a Jordan block).

### 5. Simultaneous Diagonalization

**Theorem:** Two Hermitian operators A and B can be simultaneously diagonalized if and only if [A, B] = 0.

**Meaning:** If AB = BA, there exists an orthonormal basis {|eᵢ⟩} such that:
$$A|e_i\rangle = a_i|e_i\rangle \quad \text{and} \quad B|e_i\rangle = b_i|e_i\rangle$$

**Proof sketch:**
- (⟹) If diagonal in same basis, matrices commute
- (⟸) If [A,B] = 0, B maps eigenspaces of A to themselves
  - Diagonalize A
  - Within each eigenspace of A, diagonalize B
  - Result: common eigenbasis

**Generalization:** Any finite set of pairwise commuting normal operators can be simultaneously diagonalized.

### 6. Positive Operators

**Definition:** A Hermitian operator A is **positive** (or positive semidefinite) if:
$$\langle v, Av\rangle \geq 0 \quad \text{for all } v$$

**Notation:** A ≥ 0

**Equivalent conditions:**
1. ⟨v, Av⟩ ≥ 0 for all v
2. All eigenvalues of A are ≥ 0
3. A = B*B for some operator B
4. A = C² for some Hermitian C

**Strictly positive:** A > 0 means all eigenvalues > 0 (A is positive definite).

### 7. Polar Decomposition

**Theorem:** Every operator A can be written as:
$$A = U|A|$$
where:
- |A| = √(A*A) is positive (the "modulus")
- U is unitary (if A is invertible) or partial isometry (general case)

**Analogy:** Like polar form z = r·e^(iθ) for complex numbers!
- |A| is like r (magnitude)
- U is like e^(iθ) (phase/rotation)

**Computation:**
1. Form A*A (Hermitian, positive)
2. |A| = √(A*A) via spectral decomposition
3. U = A|A|⁻¹ (if invertible)

---

## 🔬 Quantum Mechanics Connection

### Compatible Observables

**Definition:** Observables A and B are **compatible** if [A, B] = 0.

**Physical meaning:**
- Can measure both simultaneously
- Share a common set of eigenstates
- Order of measurement doesn't matter

**Examples of compatible observables:**
- Energy H and angular momentum Lz (for central potentials)
- Position x and momentum pₓ for different particles
- All components of momentum: [pₓ, pᵧ] = 0

**Incompatible observables:**
- Position and momentum: [x, p] = iℏ
- Spin components: [Sₓ, Sᵧ] = iℏSz

### Complete Set of Commuting Observables (CSCO)

**Definition:** A CSCO is a maximal set of mutually commuting observables.

**Significance:** 
- Specifying eigenvalues of all observables in CSCO uniquely determines the state
- The CSCO eigenstates form a basis

**Example:** Hydrogen atom
- CSCO = {H, L², Lz, Sz}
- State specified by |n, ℓ, m, mₛ⟩

### Density Matrices (Preview)

A **density matrix** ρ represents a quantum state (pure or mixed).

**Properties:**
1. ρ is Hermitian (ρ = ρ†)
2. ρ is positive (ρ ≥ 0)
3. tr(ρ) = 1

**Spectral decomposition:**
$$\rho = \sum_i p_i |i\rangle\langle i|$$
where pᵢ ≥ 0 and Σpᵢ = 1 (probabilities!).

**Pure vs Mixed:**
- Pure: ρ = |ψ⟩⟨ψ| (rank 1)
- Mixed: ρ has rank > 1

---

## ✏️ Worked Examples

### Example 1: Verify Normality

Is A = [[1, i], [-i, 2]] normal?

**Solution:**
$$A^* = \begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix}$$

$$A^*A = \begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix}\begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix} = \begin{pmatrix} 2 & i+2i \\ -i-2i & 1+4 \end{pmatrix} = \begin{pmatrix} 2 & 3i \\ -3i & 5 \end{pmatrix}$$

$$AA^* = \begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix}\begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix} = \begin{pmatrix} 2 & 3i \\ -3i & 5 \end{pmatrix}$$

A*A = AA* ✓, so A is normal.

### Example 2: Simultaneous Diagonalization

Find a common eigenbasis for:
$$A = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**First check:** [A, B] = AB - BA
$$AB = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}, \quad BA = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$
$$[A, B] = \begin{pmatrix} 0 & 2 \\ -2 & 0 \end{pmatrix} \neq 0$$

They don't commute! No common eigenbasis exists.

**Physical interpretation:** A and B are incompatible observables (like σz and σx).

### Example 3: Polar Decomposition

Find the polar decomposition of:
$$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

**Step 1:** Compute A*A
$$A^*A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}$$

**Step 2:** Find eigenvalues of A*A
$$\det(A^*A - \lambda I) = (1-\lambda)(2-\lambda) - 1 = \lambda^2 - 3\lambda + 1$$
$$\lambda = \frac{3 \pm \sqrt{5}}{2}$$

Let φ = (1+√5)/2 (golden ratio). Then λ₁ = φ², λ₂ = 1/φ² = φ⁻².

**Step 3:** |A| = √(A*A) via spectral decomposition
$$|A| = \sqrt{\lambda_1}P_1 + \sqrt{\lambda_2}P_2 = \phi P_1 + \phi^{-1}P_2$$

**Step 4:** U = A|A|⁻¹

(Full computation involves finding eigenvectors - typically done numerically)

### Example 4: Density Matrix

Verify ρ = ½[[1, 0], [0, 1]] + ½|+⟩⟨+| where |+⟩ = (|0⟩+|1⟩)/√2 is a valid density matrix.

$$|+\rangle\langle+| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

$$\rho = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} + \frac{1}{2} \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 3/4 & 1/4 \\ 1/4 & 3/4 \end{pmatrix}$$

**Check:**
- Hermitian? ρ = ρ† ✓
- tr(ρ) = 3/4 + 3/4 = 3/2 ≠ 1 ✗

Wait, let me recalculate. If we want a mixture of I/2 and |+⟩⟨+|:
$$\rho = p \frac{I}{2} + (1-p)|+\rangle\langle+|$$

For this to be properly normalized, we need p + (1-p) = 1 with proper interpretation.

Actually: ρ = ½|0⟩⟨0| + ½|+⟩⟨+| makes more sense as a mixture.

---

## 📝 Practice Problems

### Level 1: Normal Operators
1. Prove that every unitary operator is normal.

2. Show that σₓ, σᵧ, σz are all normal.

3. Is A = [[0, 1], [0, 0]] normal? If not, find ||Av|| and ||A*v|| for v = (1, 0).

### Level 2: Simultaneous Diagonalization
4. Show that [σz, σz] = 0 (trivially) and find their common eigenbasis.

5. Prove that if A is Hermitian and [A, B] = 0, then B maps each eigenspace of A to itself.

6. Find matrices A, B with [A,B] = 0 that are not simultaneously diagonalizable. (Hint: non-normal)

### Level 3: Positive Operators
7. Show that A*A is always positive semidefinite.

8. If A is positive definite, show that A⁻¹ is also positive definite.

9. Prove: A ≥ 0 iff all eigenvalues of A are ≥ 0.

### Level 4: Applications
10. Find the polar decomposition of σₓ.

11. Verify that ρ = [[2/3, 1/3], [1/3, 1/3]] is a valid density matrix. Is it pure or mixed?

12. For commuting Hermitians A, B, show that e^(A+B) = e^A e^B.

---

## 💻 Evening Computational Lab

```python
import numpy as np
from scipy.linalg import polar, sqrtm

# ============================================
# Normal Operator Tests
# ============================================

def is_normal(A, tol=1e-10):
    """Check if matrix is normal"""
    return np.allclose(A @ A.conj().T, A.conj().T @ A, atol=tol)

def is_hermitian(A, tol=1e-10):
    """Check if matrix is Hermitian"""
    return np.allclose(A, A.conj().T, atol=tol)

def is_unitary(A, tol=1e-10):
    """Check if matrix is unitary"""
    n = A.shape[0]
    return np.allclose(A @ A.conj().T, np.eye(n), atol=tol)

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

print("=== Operator Classification ===")
for name, A in [("I", I), ("X", X), ("Y", Y), ("Z", Z)]:
    print(f"{name}: Normal={is_normal(A)}, Hermitian={is_hermitian(A)}, Unitary={is_unitary(A)}")

# Non-normal example
S = np.array([[0, 1], [0, 0]], dtype=complex)
print(f"\nShift S: Normal={is_normal(S)}")
print(f"S*S =\n{S.conj().T @ S}")
print(f"SS* =\n{S @ S.conj().T}")

# ============================================
# Simultaneous Diagonalization
# ============================================

def commutator(A, B):
    """Compute [A, B] = AB - BA"""
    return A @ B - B @ A

def check_simultaneous_diag(A, B, tol=1e-10):
    """Check if A and B can be simultaneously diagonalized"""
    comm = commutator(A, B)
    commutes = np.allclose(comm, np.zeros_like(comm), atol=tol)
    return commutes

def find_common_eigenbasis(A, B, tol=1e-10):
    """
    Find common eigenbasis for commuting Hermitian matrices.
    Returns eigenvalues of both and common eigenvectors.
    """
    if not check_simultaneous_diag(A, B, tol):
        raise ValueError("Matrices do not commute!")
    
    # Diagonalize A first
    eig_A, V_A = np.linalg.eigh(A)
    
    # Transform B to A's eigenbasis
    B_transformed = V_A.conj().T @ B @ V_A
    
    # B should be block diagonal in A's eigenspaces
    # For simplicity, assume no degeneracy
    eig_B_in_basis = np.diag(B_transformed)
    
    return eig_A, eig_B_in_basis, V_A

# Test with commuting matrices
A = Z  # σz
B = np.array([[1, 0], [0, 2]], dtype=complex)  # Another diagonal matrix

print("\n=== Simultaneous Diagonalization ===")
print(f"[A, B] commutes: {check_simultaneous_diag(A, B)}")
if check_simultaneous_diag(A, B):
    eig_A, eig_B, V = find_common_eigenbasis(A, B)
    print(f"Eigenvalues of A: {eig_A}")
    print(f"Eigenvalues of B: {eig_B}")
    print(f"Common eigenvectors:\n{V}")

# Non-commuting example
print(f"\n[σx, σz] commutes: {check_simultaneous_diag(X, Z)}")
print(f"[σx, σz] =\n{commutator(X, Z)}")

# ============================================
# Polar Decomposition
# ============================================

def polar_decomposition(A):
    """Compute polar decomposition A = U|A|"""
    # |A| = sqrt(A*A)
    A_dag_A = A.conj().T @ A
    modulus = sqrtm(A_dag_A)
    
    # U = A * |A|^{-1} (if invertible)
    if np.linalg.matrix_rank(A) == A.shape[0]:
        U = A @ np.linalg.inv(modulus)
    else:
        # Use pseudo-inverse for non-invertible
        U, _ = polar(A)
    
    return U, modulus

# Test
A = np.array([[1, 1], [0, 1]], dtype=complex)
U, P = polar_decomposition(A)

print("\n=== Polar Decomposition ===")
print(f"A =\n{A}")
print(f"|A| =\n{P}")
print(f"U =\n{U}")
print(f"U unitary: {is_unitary(U)}")
print(f"UP = A: {np.allclose(U @ P, A)}")

# ============================================
# Positive Operators and Density Matrices
# ============================================

def is_positive_semidefinite(A, tol=1e-10):
    """Check if Hermitian matrix is positive semidefinite"""
    if not is_hermitian(A):
        return False
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues >= -tol)

def is_valid_density_matrix(rho, tol=1e-10):
    """Check if matrix is a valid density matrix"""
    # Must be Hermitian
    if not is_hermitian(rho, tol):
        return False, "Not Hermitian"
    # Must be positive semidefinite
    if not is_positive_semidefinite(rho, tol):
        return False, "Not positive"
    # Must have trace 1
    if not np.isclose(np.trace(rho), 1, atol=tol):
        return False, f"Trace = {np.trace(rho)}, not 1"
    return True, "Valid"

def purity(rho):
    """Compute purity tr(ρ²)"""
    return np.real(np.trace(rho @ rho))

# Example density matrices
rho_pure = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
rho_mixed = np.array([[0.5, 0], [0, 0.5]], dtype=complex)  # maximally mixed

print("\n=== Density Matrices ===")
for name, rho in [("Pure |0⟩⟨0|", rho_pure), ("Maximally mixed I/2", rho_mixed)]:
    valid, msg = is_valid_density_matrix(rho)
    print(f"{name}: {msg}, Purity = {purity(rho):.4f}")

# Create a mixed state
plus = np.array([[1], [1]]) / np.sqrt(2)
rho_plus = plus @ plus.conj().T
rho_mixture = 0.7 * rho_pure + 0.3 * rho_plus

valid, msg = is_valid_density_matrix(rho_mixture)
print(f"\n70% |0⟩ + 30% |+⟩: {msg}")
print(f"Purity = {purity(rho_mixture):.4f}")
print(f"ρ =\n{rho_mixture}")

# ============================================
# Application: CSCO for Spin-1/2
# ============================================

print("\n=== CSCO for Spin-1/2 ===")
# For spin-1/2, Sz alone is a CSCO (non-degenerate)
# Eigenvalues uniquely specify state

Sz = Z / 2  # in units of ℏ
eig_vals, eig_vecs = np.linalg.eigh(Sz)
print(f"Sz eigenvalues: {eig_vals} (in ℏ)")
print(f"Eigenstates form complete basis: {np.allclose(eig_vecs @ eig_vecs.conj().T, I)}")
```

---

## ✅ Daily Checklist

- [ ] Define normal operators and their characterizations
- [ ] Understand spectral theorem for normal operators
- [ ] Master simultaneous diagonalization criterion
- [ ] Work with positive operators
- [ ] Understand polar decomposition
- [ ] Connect to compatible observables in QM
- [ ] Complete computational lab
- [ ] Solve at least 6 practice problems

---

## 🔜 Preview: Tomorrow

**Day 118: Computational Lab — Hermitian & Unitary Operators**
- Implement spectral decomposition algorithms
- Build quantum gate simulator
- Visualize Bloch sphere dynamics
- Explore density matrix evolution

---

*"The question of whether two observables can be measured simultaneously is answered by a simple algebraic test: do their operators commute?"*
— Paul Dirac
