# Day 105: Week 15 Review — Eigenvalues & Eigenvectors Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 12:00 PM | 2 hours | Concept Review & Integration |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Comprehensive Problem Set |

**Total Study Time: 4 hours (Sunday schedule)**

---

## 🎯 Review Objectives

By the end of today, you should:

1. Have mastered all eigenvalue/eigenvector concepts
2. Solve problems fluently without notes
3. Deeply understand the QM significance (measurements, observables)
4. Be fully prepared for Week 16 (Inner Product Spaces)
5. Identify any gaps requiring additional study

---

## 📚 Week 15 Concept Map

```
                    EIGENVALUES & EIGENVECTORS
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
     FOUNDATIONS          COMPUTATION          APPLICATIONS
          │                   │                   │
    ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
    │           │       │           │       │           │
Definition  Geometric  Char.    Finding   Diagonal-  Spectral
 Av = λv    Meaning   Poly     Eigensp.  ization    Theorem
    │                   │           │           │
    │           det(A-λI)=0    Basis of    P⁻¹AP = D
    └───────────────────┴───────────┴───────────┘
                        │
                QUANTUM CONNECTION
                        │
           ┌────────────┴────────────┐
           │                         │
      Observables              Measurements
      (Hermitian ops)          (Eigenvalues = outcomes)
           │                         │
      A|ψ⟩ = a|ψ⟩              Prob = |⟨a|ψ⟩|²
```

---

## 🔄 Morning Session: Concept Review (2 hours)

### Part 1: Core Definitions Recall (30 min)

**Write from memory:**

#### 1. Eigenvalue/Eigenvector Definition
Let T: V → V be a linear operator (or A an n×n matrix).
- **Eigenvalue:** λ ∈ 𝔽 such that T(v) = λv for some nonzero v
- **Eigenvector:** nonzero v ∈ V such that T(v) = λv

**Key point:** v must be NONZERO, but λ can be zero!

#### 2. Characteristic Polynomial
$$p_A(\lambda) = \det(A - \lambda I)$$

- Degree n polynomial for n×n matrix
- Roots are eigenvalues
- Fundamental theorem of algebra: always n roots in ℂ (counting multiplicity)

#### 3. Eigenspace
$$E_\lambda = \ker(A - \lambda I) = \{v : Av = \lambda v\}$$

- Always a subspace (it's a null space!)
- dim(E_λ) ≥ 1 for each eigenvalue
- dim(E_λ) ≤ algebraic multiplicity of λ

#### 4. Diagonalization Criterion
A is diagonalizable ⟺ A has n linearly independent eigenvectors
⟺ sum of geometric multiplicities = n
⟺ ∃ invertible P such that P⁻¹AP = D (diagonal)

### Part 2: Key Theorems (30 min)

#### Theorem 1: Eigenvalues and Characteristic Polynomial
λ is an eigenvalue of A ⟺ det(A - λI) = 0

*Proof:* λ eigenvalue ⟺ ∃v≠0: Av = λv ⟺ (A-λI)v = 0 has nontrivial solution ⟺ A-λI is singular ⟺ det(A-λI) = 0 ∎

#### Theorem 2: Linear Independence of Eigenvectors
Eigenvectors corresponding to distinct eigenvalues are linearly independent.

*Why it matters:* n distinct eigenvalues ⟹ automatically diagonalizable!

#### Theorem 3: Trace and Determinant
For n×n matrix A with eigenvalues λ₁,...,λₙ:
- tr(A) = λ₁ + λ₂ + ... + λₙ
- det(A) = λ₁ · λ₂ · ... · λₙ

*Application:* Quick check for eigenvalue calculations!

#### Theorem 4: Spectral Theorem (Preview)
Every Hermitian matrix A = A† is diagonalizable by a unitary matrix:
$$U^\dagger A U = D$$
with real eigenvalues and orthonormal eigenvectors.

*QM significance:* This is why observables have real measurement outcomes!

### Part 3: Quantum Mechanics Connections (30 min)

**The Eigenvalue-Measurement Correspondence:**

| Linear Algebra | Quantum Mechanics |
|----------------|-------------------|
| Matrix A | Observable  |
| Eigenvalue λ | Measurement outcome |
| Eigenvector \|λ⟩ | State after measurement |
| Av = λv | Definite value state |
| Spectral decomposition | Measurement postulate |

**Critical QM Results from This Week:**

1. **Observables are Hermitian:** A = A† ensures real eigenvalues (real measurement outcomes)

2. **Measurement collapses to eigenstates:**
   - Before: |ψ⟩ = Σᵢ cᵢ|λᵢ⟩
   - Measure A → get λₖ with prob |cₖ|²
   - After: |ψ⟩ = |λₖ⟩

3. **Compatible observables:** [A,B] = 0 ⟹ simultaneous eigenstates exist

4. **Energy eigenstates:**
   - H|Eₙ⟩ = Eₙ|Eₙ⟩
   - Time evolution: |ψ(t)⟩ = Σₙ cₙ e^(-iEₙt/ℏ)|Eₙ⟩

5. **Spin-½ example:**
   - σ_z eigenvalues: +1, -1 (spin up/down)
   - σ_z eigenvectors: |↑⟩ = (1,0), |↓⟩ = (0,1)

### Part 4: Common Mistakes to Avoid (30 min)

1. **Forgetting v ≠ 0:** The zero vector is NEVER an eigenvector

2. **Confusing multiplicities:**
   - Algebraic: multiplicity as root of char. poly
   - Geometric: dim(eigenspace)
   - Always: geometric ≤ algebraic

3. **Thinking real matrices have real eigenvalues:**
   - False! Example: rotation by 90° has eigenvalues ±i
   - True for: symmetric/Hermitian matrices

4. **Assuming diagonalizability:**
   - Not all matrices are diagonalizable!
   - Example: [[0,1],[0,0]] has only one eigenvector

5. **Order of P columns:**
   - If D = diag(λ₁,λ₂,...), then P = [v₁|v₂|...]
   - Column i of P is eigenvector for λᵢ

---

## 📝 Afternoon Session: Comprehensive Problem Set (2 hours)

### Section A: Fundamentals (20 min)

**Problem A1.** Find all eigenvalues and eigenvectors of:
$$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$

**Problem A2.** Show that λ = 0 is an eigenvalue of A iff A is singular.

**Problem A3.** If Av = λv, show that A²v = λ²v. Generalize to Aⁿ.

**Problem A4.** Prove: If A is invertible and Av = λv, then A⁻¹v = (1/λ)v.

### Section B: Characteristic Polynomials (20 min)

**Problem B1.** Find the characteristic polynomial of:
$$B = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 2 & 1 \\ 0 & 0 & 2 \end{pmatrix}$$
What are its eigenvalues? Is B diagonalizable?

**Problem B2.** A 4×4 matrix has characteristic polynomial:
$$p(\lambda) = (\lambda - 1)^2(\lambda + 2)(\lambda - 3)$$
What are the possible Jordan normal forms?

**Problem B3.** Prove that similar matrices have the same characteristic polynomial.

### Section C: Diagonalization (25 min)

**Problem C1.** Diagonalize (find P and D):
$$C = \begin{pmatrix} 4 & -2 \\ 1 & 1 \end{pmatrix}$$

**Problem C2.** For which values of k is the matrix diagonalizable?
$$M_k = \begin{pmatrix} 2 & k \\ 0 & 2 \end{pmatrix}$$

**Problem C3.** Show that any matrix satisfying A² = A is diagonalizable.

**Problem C4.** If A is diagonalizable, show Aⁿ → 0 as n → ∞ iff all |λᵢ| < 1.

### Section D: Trace and Determinant (15 min)

**Problem D1.** A 3×3 matrix has eigenvalues 2, -1, 3. Find tr(A) and det(A).

**Problem D2.** If tr(A) = 0 and det(A) = 0 for a 2×2 matrix, what are its eigenvalues?

**Problem D3.** Prove: tr(AB) = tr(BA) for any n×n matrices A, B.

### Section E: Quantum Applications (25 min)

**Problem E1.** The Hamiltonian for a two-level atom is:
$$H = \begin{pmatrix} E_0 & V \\ V & E_0 \end{pmatrix}$$
Find the energy eigenvalues and eigenstates.

**Problem E2.** For the spin operator S_x = (ℏ/2)σ_x:
- Find eigenvalues and eigenvectors
- If we measure S_x on state |↑⟩ (spin-up in z), what outcomes are possible and with what probabilities?

**Problem E3.** Show that [σ_x, σ_y] = 2iσ_z (Pauli matrix commutator).
What does this say about simultaneous measurements of S_x and S_y?

**Problem E4.** The time evolution operator is U(t) = e^(-iHt/ℏ).
- Show that if |E⟩ is an energy eigenstate, U(t)|E⟩ = e^(-iEt/ℏ)|E⟩
- Why are energy eigenstates called "stationary states"?

### Section F: Advanced Problems (15 min)

**Problem F1.** Prove that a normal matrix (AA† = A†A) is diagonalizable by a unitary.

**Problem F2.** Show that eigenvalues of a unitary matrix have |λ| = 1.

**Problem F3.** If A and B are simultaneously diagonalizable, prove [A,B] = 0.

---

## ✅ Solutions Outline

### A1 Solution:
char poly: (3-λ)(2-λ) = 0 → λ = 3, 2
λ=3: (A-3I)v = 0 → v = (1,0)
λ=2: (A-2I)v = 0 → v = (1,-1)

### B1 Solution:
char poly: (2-λ)³ = 0 → λ = 2 (multiplicity 3)
E_2 = ker(B-2I): only 1D (check!)
Not diagonalizable (geometric mult < algebraic mult)

### C1 Solution:
char poly: λ² - 5λ + 6 = (λ-2)(λ-3)
λ=2: v₁ = (1,1), λ=3: v₂ = (2,1)
P = [[1,2],[1,1]], D = diag(2,3)
Verify: P⁻¹AP = D

### E1 Solution:
char poly: (E₀-λ)² - V² = 0
λ = E₀ ± V
Eigenstates: |±⟩ = (|1⟩ ± |2⟩)/√2
(This is the avoided crossing / level repulsion!)

---

## 🎯 Self-Assessment Rubric

### Mastery Indicators

**Level 1 - Recognition:**
- [ ] Can identify eigenvalue problems
- [ ] Knows the definitions

**Level 2 - Computation:**
- [ ] Finds eigenvalues via char polynomial
- [ ] Finds eigenvectors via null space
- [ ] Diagonalizes 2×2 and 3×3 matrices

**Level 3 - Application:**
- [ ] Uses eigenvalues to analyze matrix powers
- [ ] Applies to differential equations
- [ ] Connects to quantum measurements

**Level 4 - Analysis:**
- [ ] Proves theorems about eigenvalues
- [ ] Determines diagonalizability criteria
- [ ] Understands spectral theorem deeply

**Level 5 - Synthesis:**
- [ ] Designs quantum systems using eigenvalue analysis
- [ ] Connects to broader mathematical structures
- [ ] Could derive results independently

**Your Level:** ___________

---

## 🔄 Spaced Repetition Cards

### Card 1
**Front:** What is the definition of eigenvalue/eigenvector?
**Back:** λ is eigenvalue, v is eigenvector if Av = λv with v ≠ 0

### Card 2
**Front:** How do you find eigenvalues?
**Back:** Solve det(A - λI) = 0 (characteristic polynomial)

### Card 3
**Front:** When is a matrix diagonalizable?
**Back:** When it has n linearly independent eigenvectors (for n×n matrix)

### Card 4
**Front:** What's the QM interpretation of eigenvalues?
**Back:** Eigenvalues of an observable = possible measurement outcomes

### Card 5
**Front:** Why must quantum observables be Hermitian?
**Back:** Hermitian matrices have real eigenvalues (real measurement outcomes)

### Card 6
**Front:** What happens when you measure an observable on an eigenstate?
**Back:** You get that eigenvalue with probability 1; state unchanged

---

## 🚀 Preview: Week 16 — Inner Product Spaces

**What's coming:**
- Inner products: ⟨u, v⟩ (generalized dot product)
- Norms and orthogonality
- Gram-Schmidt orthogonalization
- Orthonormal bases
- Orthogonal projections

**QM preview:**
- Bra-ket notation: ⟨φ|ψ⟩
- Probability amplitudes
- Born rule: P = |⟨outcome|state⟩|²
- Hilbert spaces

**Key transition:** Inner products + eigenvalues = complete QM measurement theory!

---

## 📋 Week 15 Completion Checklist

### Concepts Mastered
- [ ] Eigenvalue/eigenvector definition
- [ ] Characteristic polynomial computation
- [ ] Finding eigenspaces
- [ ] Diagonalization procedure
- [ ] Trace/determinant relationships
- [ ] Spectral theorem statement
- [ ] QM measurement postulate connection

### Computational Skills
- [ ] Hand calculation for 2×2, 3×3 matrices
- [ ] NumPy eigenvalue functions
- [ ] Power method implementation
- [ ] QR algorithm understanding
- [ ] Quantum simulation basics

### Materials Completed
- [ ] Day 99: Eigenvalue Foundations
- [ ] Day 100: Characteristic Polynomials
- [ ] Day 101: Eigenspaces and Diagonalization
- [ ] Day 102: Spectral Theorem
- [ ] Day 103: Advanced Topics & Applications
- [ ] Day 104: Computational Lab
- [ ] Day 105: Review (today)

---

## 📖 Gap-Filling Resources

If you struggled with:

**Finding eigenvalues:**
- Axler, Chapter 5.A
- 3Blue1Brown, Episode 14

**Diagonalization:**
- Strang, Chapter 6.2
- MIT 18.06 Lecture 21

**QM connections:**
- Shankar, Chapter 1.4-1.5
- Griffiths, Chapter 3.3

---

## 📝 Reflection Questions

1. Why is the spectral theorem so important for quantum mechanics?

2. What's the physical meaning of non-diagonalizable matrices in QM?

3. How does eigenvalue analysis connect to stability of dynamical systems?

4. What made this week's material click for you?

---

**End of Week 15 — Eigenvalues & Eigenvectors ✓**

**End of Month 4 — Linear Algebra I Complete!**

*Next: Week 16 — Inner Product Spaces (begins Month 4 wrap-up, prepares for advanced topics)*

---

*"In mathematics you don't understand things. You just get used to them."*
— John von Neumann

*"The eigenvalue problem is the most important problem in mathematics."*
— Peter Lax
