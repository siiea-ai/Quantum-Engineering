# Day 119: Week 17 Review — Hermitian & Unitary Operators Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 12:00 PM | 2 hours | Concept Review & Integration |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Comprehensive Problem Set |

**Total Study Time: 4 hours (Sunday schedule)**

---

## 🎯 Review Objectives

By the end of today, you should:

1. Have mastered all Week 17 concepts deeply
2. Fluently work with Hermitian, unitary, and normal operators
3. Apply spectral theorem to quantum mechanics problems
4. Understand the complete measurement theory foundation
5. Be prepared for Week 18 (Advanced Topics: SVD, Tensor Products, Density Matrices)

---

## 📚 Week 17 Concept Map

```
              OPERATORS ON INNER PRODUCT SPACES
                          │
         ┌────────────────┼────────────────┐
         │                │                │
      ADJOINT          SPECIAL          NORMAL
      A ↔ A*           CLASSES          AA*=A*A
         │                │                │
         │     ┌──────────┼──────────┐     │
         │     │          │          │     │
         │  HERMITIAN  UNITARY   POSITIVE  │
         │   A = A*    U*U = I    A ≥ 0    │
         │     │          │          │     │
         └─────┴──────────┴──────────┴─────┘
                          │
                  SPECTRAL THEOREM
                          │
              ┌───────────┴───────────┐
              │                       │
        DIAGONALIZATION          SPECTRAL
        (orthonormal basis)     DECOMPOSITION
              │                  A = ΣλᵢPᵢ
              │                       │
              └───────────┬───────────┘
                          │
                  QUANTUM MECHANICS
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    OBSERVABLES     TIME EVOLUTION    MEASUREMENTS
     (Hermitian)      (Unitary)      (Projections)
```

---

## 🔄 Morning Session: Concept Review (2 hours)

### Part 1: Core Definitions (30 min)

**Write from memory:**

#### 1. Adjoint Operator
For inner product spaces: ⟨Av, w⟩ = ⟨v, A*w⟩ for all v, w
For matrices: A* = A̅ᵀ (conjugate transpose)

#### 2. Hermitian (Self-Adjoint)
$$A = A^*$$
Properties:
- Real eigenvalues
- Orthogonal eigenvectors
- Unitarily diagonalizable
- ⟨v, Av⟩ ∈ ℝ for all v

#### 3. Unitary
$$U^*U = UU^* = I$$
Properties:
- Preserves inner products: ⟨Uv, Uw⟩ = ⟨v, w⟩
- Preserves norms: ||Uv|| = ||v||
- Eigenvalues on unit circle: |λ| = 1
- Maps ONB to ONB

#### 4. Normal
$$AA^* = A^*A$$
Key insight: Unitarily diagonalizable ⟺ normal

#### 5. Positive (Semidefinite)
$$\langle v, Av \rangle \geq 0 \text{ for all } v$$
Equivalent: All eigenvalues ≥ 0

### Part 2: Key Theorems (30 min)

#### Theorem 1: Spectral Theorem (Hermitian)
Every Hermitian operator A has:
1. Real eigenvalues
2. Orthonormal eigenbasis
3. Spectral decomposition: A = ΣλᵢPᵢ

#### Theorem 2: Spectral Theorem (Normal)
A is normal ⟺ A is unitarily diagonalizable
⟺ V has orthonormal eigenbasis of A

#### Theorem 3: Simultaneous Diagonalization
Hermitian A, B can be simultaneously diagonalized ⟺ [A, B] = 0

#### Theorem 4: Polar Decomposition
Every operator: A = U|A| where |A| = √(A*A) positive, U unitary

#### Theorem 5: Functions of Operators
If A = ΣλᵢPᵢ, then f(A) = Σf(λᵢ)Pᵢ

### Part 3: QM Correspondences (30 min)

| Math Concept | QM Meaning |
|--------------|------------|
| Hermitian operator A | Observable |
| Eigenvalue λ | Measurement outcome |
| Eigenstate \|λ⟩ | State with definite value |
| Unitary U | Time evolution, quantum gate |
| U = e^(-iHt/ℏ) | Schrödinger evolution |
| [A, B] = 0 | Compatible observables |
| [A, B] ≠ 0 | Uncertainty relation |
| Spectral decomposition | Measurement postulate |
| Projection Pᵢ | Post-measurement collapse |
| ⟨ψ\|A\|ψ⟩ | Expectation value |
| Positive ρ, tr(ρ)=1 | Density matrix |

### Part 4: Common Mistakes (30 min)

1. **Confusing Hermitian and unitary:**
   - Hermitian: A = A* (real eigenvalues)
   - Unitary: U*U = I (eigenvalues |λ| = 1)
   - A matrix can be both: only real orthogonal with λ = ±1

2. **Forgetting complex conjugates:**
   - Adjoint: (AB)* = B*A* (order reverses!)
   - Inner product: ⟨αv, w⟩ = α*⟨v, w⟩ (conjugate on first!)

3. **Normal ≠ Hermitian:**
   - All Hermitian matrices are normal
   - Not all normal matrices are Hermitian (e.g., unitary)

4. **Spectral theorem conditions:**
   - Works for normal operators on complex spaces
   - For real spaces: need symmetric (real Hermitian)

5. **Positive definiteness:**
   - Positive semidefinite: λᵢ ≥ 0
   - Positive definite: λᵢ > 0 (strictly)

---

## 📝 Afternoon Session: Comprehensive Problem Set (2 hours)

### Section A: Adjoint Operators (15 min)

**Problem A1.** Find A* for:
$$A = \begin{pmatrix} 1+i & 2 \\ 3i & 4-2i \end{pmatrix}$$

**Problem A2.** Prove: (A + B)* = A* + B*

**Problem A3.** Prove: (AB)* = B*A*

**Problem A4.** Show that ker(A*) = (im(A))⊥

### Section B: Hermitian Operators (20 min)

**Problem B1.** Verify that H = [[2, 1-i], [1+i, 3]] is Hermitian and find its eigenvalues.

**Problem B2.** Prove: If A is Hermitian, then e^(iA) is unitary.

**Problem B3.** Show that ⟨v, Av⟩ is real for all v iff A is Hermitian.

**Problem B4.** For Hermitian A, prove: ||A|| = max|λᵢ| (spectral norm).

**Problem B5.** The Pauli matrices plus identity {I, σₓ, σᵧ, σ_z} form a basis for 2×2 Hermitian matrices. Express [[3, 1-i], [1+i, 1]] in this basis.

### Section C: Unitary Operators (20 min)

**Problem C1.** Verify that U = (1/√2)[[1, 1], [1, -1]] is unitary.

**Problem C2.** Prove: det(U) has absolute value 1 for any unitary U.

**Problem C3.** Show that the product of unitary operators is unitary.

**Problem C4.** Prove: Eigenvalues of unitary operators lie on the unit circle.

**Problem C5.** Find all 2×2 unitary matrices that are also Hermitian.

### Section D: Spectral Theorem (25 min)

**Problem D1.** Find the spectral decomposition of:
$$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$$

**Problem D2.** Use spectral decomposition to compute A¹⁰⁰ for the matrix in D1.

**Problem D3.** Compute e^(iπσₓ/2). What rotation does this represent on the Bloch sphere?

**Problem D4.** For A = [[4, 2], [2, 1]], compute √A using the spectral theorem.

**Problem D5.** Prove: tr(f(A)) = Σf(λᵢ) for Hermitian A.

### Section E: Normal Operators (15 min)

**Problem E1.** Is A = [[1, -1], [1, 1]] normal? If so, diagonalize it.

**Problem E2.** Prove: A is normal iff ||Av|| = ||A*v|| for all v.

**Problem E3.** Show that commuting normal operators can be simultaneously diagonalized.

### Section F: Quantum Applications (25 min)

**Problem F1.** The spin operator in direction n̂ = (nₓ, nᵧ, n_z) is:
$$S_{\hat{n}} = \frac{\hbar}{2}(n_x\sigma_x + n_y\sigma_y + n_z\sigma_z)$$
Find its eigenvalues and eigenstates.

**Problem F2.** A qubit evolves under H = ℏω σ_z/2. If |ψ(0)⟩ = |+⟩, find |ψ(t)⟩.

**Problem F3.** For the Hamiltonian H = [[E₀, V], [V, E₀]]:
- Find energy eigenvalues and eigenstates
- Compute e^(-iHt/ℏ)
- If system starts in |1⟩, find probability of measuring |2⟩ at time t

**Problem F4.** Show that [Sₓ, Sᵧ] = iℏSz and derive the uncertainty relation ΔSₓ ΔSᵧ ≥ ℏ|⟨Sz⟩|/2.

**Problem F5.** A density matrix has the form ρ = (I + r⃗·σ⃗)/2 where σ⃗ = (σₓ, σᵧ, σz).
- Find the condition on |r⃗| for ρ to be a valid density matrix
- When is ρ a pure state?
- Find the von Neumann entropy S = -tr(ρ log ρ) in terms of |r⃗|

---

## ✅ Solutions Outline

### A1 Solution:
$$A^* = \overline{A}^T = \begin{pmatrix} 1-i & -3i \\ 2 & 4+2i \end{pmatrix}$$

### B1 Solution:
H = H† ✓. Characteristic polynomial: (2-λ)(3-λ) - |1-i|² = λ² - 5λ + 4 = (λ-1)(λ-4)
Eigenvalues: λ = 1, 4

### D1 Solution:
Eigenvalues: λ = 2, 4
v₁ = (1,-1)/√2, v₂ = (1,1)/√2
A = 2P₁ + 4P₂ where Pᵢ = |vᵢ⟩⟨vᵢ|

### D3 Solution:
$$e^{i\pi\sigma_x/2} = \cos(\pi/2)I + i\sin(\pi/2)\sigma_x = i\sigma_x$$
This is a 90° rotation around X-axis (up to global phase).

### F3 Solution:
Eigenvalues: E± = E₀ ± V
Eigenstates: |±⟩ = (|1⟩ ± |2⟩)/√2
|ψ(t)⟩ = (e^(-iE₊t/ℏ)|+⟩ - e^(-iE₋t/ℏ)|-⟩)/√2
P₂(t) = |⟨2|ψ(t)⟩|² = sin²(Vt/ℏ) (Rabi oscillations!)

---

## 🎯 Self-Assessment Rubric

### Mastery Indicators

**Level 1 - Recognition:**
- [ ] Know definitions of adjoint, Hermitian, unitary, normal
- [ ] Identify operator types from matrices

**Level 2 - Computation:**
- [ ] Compute adjoints and verify properties
- [ ] Find eigenvalues/eigenvectors of Hermitian matrices
- [ ] Apply spectral decomposition

**Level 3 - Application:**
- [ ] Use spectral theorem for matrix functions
- [ ] Connect to quantum measurement theory
- [ ] Solve time evolution problems

**Level 4 - Analysis:**
- [ ] Prove theorems about operator classes
- [ ] Understand relationships between classes
- [ ] Apply simultaneous diagonalization

**Level 5 - Synthesis:**
- [ ] Design quantum systems using operator theory
- [ ] Connect mathematical structures to physics deeply
- [ ] Derive new results independently

**Your Level:** ___________

---

## 🔄 Spaced Repetition Cards

### Card 1
**Front:** What is the adjoint A* of an operator?
**Back:** Defined by ⟨Av, w⟩ = ⟨v, A*w⟩. For matrices: A* = Ā^T.

### Card 2
**Front:** What are the three equivalent conditions for Hermitian?
**Back:** (1) A = A*, (2) All eigenvalues real, (3) ⟨v, Av⟩ ∈ ℝ for all v

### Card 3
**Front:** What does unitary preserve?
**Back:** Inner products, norms, and orthonormal bases. U*U = I.

### Card 4
**Front:** State the spectral theorem for Hermitian operators.
**Back:** A = ΣλᵢPᵢ where λᵢ are real eigenvalues and Pᵢ are orthogonal projections.

### Card 5
**Front:** When can two Hermitian operators be simultaneously diagonalized?
**Back:** If and only if they commute: [A, B] = 0.

### Card 6
**Front:** What is U(t) for quantum time evolution?
**Back:** U(t) = e^(-iHt/ℏ), unitary, generated by Hermitian Hamiltonian.

---

## 🚀 Preview: Week 18 — Advanced Linear Algebra

**What's coming:**
- Singular Value Decomposition (SVD)
- Tensor products and composite systems
- Density matrices for mixed states
- Trace and partial trace

**QM preview:**
- Multi-qubit systems: |ψ⟩ ∈ ℋ₁ ⊗ ℋ₂
- Entanglement detection via partial trace
- Mixed states and decoherence
- Quantum channels

**Key transition:** From single systems to composite quantum systems!

---

## 📋 Week 17 Completion Checklist

### Concepts Mastered
- [ ] Adjoint operators and properties
- [ ] Hermitian operators and real eigenvalues
- [ ] Unitary operators and norm preservation
- [ ] Spectral theorem for Hermitian/normal operators
- [ ] Functions of operators via spectral decomposition
- [ ] Normal operators and diagonalizability
- [ ] Positive operators
- [ ] Simultaneous diagonalization

### Computational Skills
- [ ] Finding adjoints of matrices
- [ ] Verifying Hermitian/unitary/normal
- [ ] Computing spectral decompositions
- [ ] Matrix exponentials and functions
- [ ] Quantum simulation with operators

### QM Connections
- [ ] Observables as Hermitian operators
- [ ] Time evolution as unitary operators
- [ ] Measurement and spectral decomposition
- [ ] Compatible observables and commutators
- [ ] Density matrices (preview)

### Materials Completed
- [ ] Day 113: Adjoint Operators
- [ ] Day 114: Hermitian Operators
- [ ] Day 115: Unitary Operators  
- [ ] Day 116: Spectral Theorem
- [ ] Day 117: Normal Operators
- [ ] Day 118: Computational Lab
- [ ] Day 119: Review (today)

---

## 📖 Gap-Filling Resources

If you struggled with:

**Adjoint operators:**
- Axler, Section 7.A
- Strang, Section 6.3

**Spectral theorem:**
- Axler, Section 7.B-7.C
- Shankar, Chapter 1.8

**QM applications:**
- Sakurai, Chapter 1.3-1.5
- Griffiths, Chapter 3

---

## 📝 Reflection Questions

1. Why is the spectral theorem considered the cornerstone of quantum mechanics?

2. How does the commutativity of operators relate to the uncertainty principle?

3. What's the physical significance of normal but non-Hermitian operators (like unitary)?

4. How has your understanding of quantum measurement changed this week?

---

**End of Week 17 — Hermitian & Unitary Operators ✓**

*Next: Week 18 — Advanced Topics (SVD, Tensor Products, Density Matrices)*

---

*"God used beautiful mathematics in creating the world."*
— Paul Dirac

*"The underlying physical laws necessary for the mathematical theory of a large part of physics and the whole of chemistry are thus completely known, and the difficulty is only that the exact application of these laws leads to equations much too complicated to be soluble."*
— Paul Dirac
