# Day 126: Week 18 Review — Advanced Linear Algebra Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 12:00 PM | 2 hours | Concept Review & Integration |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Comprehensive Problem Set |

**Total Study Time: 4 hours (Sunday schedule)**

---

## 🎯 Review Objectives

By the end of today, you should:

1. Have mastered SVD and its quantum applications
2. Fluently work with tensor products and composite systems
3. Deeply understand density matrices and quantum channels
4. Apply partial trace and entanglement measures
5. Be prepared for Week 19 (Complex Analysis)

---

## 📚 Week 18 Concept Map

```
              ADVANCED LINEAR ALGEBRA FOR QM
                          │
         ┌────────────────┼────────────────┐
         │                │                │
       SVD           TENSOR            DENSITY
    A = UΣV*         PRODUCTS          MATRICES
         │                │                │
    ┌────┴────┐     ┌────┴────┐      ┌────┴────┐
    │         │     │         │      │         │
  Schmidt   Low-   V⊗W     Partial   Pure    Mixed
  Decomp.  Rank   |a⟩⊗|b⟩   Trace   ρ=|ψ⟩⟨ψ|  ρ=Σpᵢρᵢ
              │         │           │         │
              └────┬────┘           └────┬────┘
                   │                     │
            ENTANGLEMENT            DECOHERENCE
                   │                     │
         ┌─────────┴─────────┐    ┌──────┴──────┐
         │                   │    │             │
    Entropy            Negativity  Channels   Lindblad
   S(ρ_A)             (mixed ent)  Kraus ops  Master Eq
```

---

## 🔄 Morning Session: Concept Review (2 hours)

### Part 1: Singular Value Decomposition (30 min)

**Core Result:**
Every m×n matrix A can be written as:
$$A = U\Sigma V^*$$
- U: m×m unitary (left singular vectors)
- Σ: m×n diagonal (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V: n×n unitary (right singular vectors)

**Key Properties:**
| Property | Formula |
|----------|---------|
| σᵢ | √(eigenvalues of A*A) |
| Rank | # nonzero σᵢ |
| ‖A‖₂ | σ₁ |
| ‖A‖_F | √(Σσᵢ²) |
| Low-rank approx | Aₖ = Σᵢ₌₁ᵏ σᵢuᵢvᵢ* |
| Pseudoinverse | A⁺ = VΣ⁺U* |

**Schmidt Decomposition:**
For bipartite |ψ⟩_AB with coefficient matrix C:
$$|\psi\rangle = \sum_i \lambda_i |a_i\rangle|b_i\rangle$$
where λᵢ are singular values of C.

### Part 2: Tensor Products (30 min)

**Definition:**
$$V \otimes W: \quad (v \otimes w)_{ij} = v_i w_j$$
$$\dim(V \otimes W) = \dim(V) \times \dim(W)$$

**Kronecker Product:**
$$(A \otimes B)_{im+j, kn+l} = A_{ik} B_{jl}$$

**Key Properties:**
- (A⊗B)(C⊗D) = (AC)⊗(BD) (mixed product rule)
- (A⊗B)* = A*⊗B*
- tr(A⊗B) = tr(A)·tr(B)

**Product vs Entangled States:**
- Product: |ψ⟩ = |a⟩⊗|b⟩ (Schmidt rank = 1)
- Entangled: Cannot be written as product (Schmidt rank > 1)

### Part 3: Density Matrices (30 min)

**Pure State:** ρ = |ψ⟩⟨ψ|
**Mixed State:** ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|

**Valid Density Matrix:**
1. Hermitian: ρ = ρ†
2. Positive: ρ ≥ 0
3. Normalized: tr(ρ) = 1

**Pure vs Mixed:**
- Purity: γ = tr(ρ²)
- Pure: γ = 1
- Mixed: γ < 1
- Maximally mixed: γ = 1/d

**Bloch Sphere (Qubits):**
$$\rho = \frac{I + \vec{r}\cdot\vec{\sigma}}{2}$$
- |r⃗| = 1: Pure (surface)
- |r⃗| < 1: Mixed (interior)

### Part 4: Partial Trace and Entanglement (30 min)

**Partial Trace:**
$$\rho_A = \text{tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|) \rho_{AB} (I_A \otimes |j\rangle)$$

**Entanglement ↔ Mixed Reduced State:**
For pure |ψ⟩_AB:
- Product ⟺ ρ_A pure
- Entangled ⟺ ρ_A mixed

**Entanglement Entropy:**
$$E(|\psi\rangle) = S(\rho_A) = -\text{tr}(\rho_A \log_2 \rho_A) = -\sum_i \lambda_i^2 \log_2 \lambda_i^2$$

**Other Measures:**
- Concurrence (pure 2-qubit): C = 2|det(C)|
- Negativity (mixed states): N = (‖ρ^{T_B}‖₁ - 1)/2

---

## 📝 Afternoon Session: Comprehensive Problem Set (2 hours)

### Section A: SVD (20 min)

**Problem A1.** Find the SVD of A = [[3, 0], [0, 2], [0, 0]].

**Problem A2.** Use SVD to find the best rank-1 approximation of:
$$B = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$

**Problem A3.** Find the Schmidt decomposition of |ψ⟩ = (|00⟩ + |01⟩ + |11⟩)/√3.

**Problem A4.** Prove: σᵢ(A*) = σᵢ(A).

### Section B: Tensor Products (20 min)

**Problem B1.** Compute (σₓ ⊗ σ_z)(|0⟩ ⊗ |+⟩).

**Problem B2.** Show that CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X.

**Problem B3.** Prove: If |ψ⟩ = |a⟩⊗|b⟩, then ⟨A⊗B⟩ = ⟨A⟩_a · ⟨B⟩_b.

**Problem B4.** For 3 qubits, construct the operator that applies H to qubit 2 only.

### Section C: Density Matrices (25 min)

**Problem C1.** A qubit is prepared as |0⟩ with probability 2/3 and |+⟩ with probability 1/3. Write ρ and compute tr(ρ²).

**Problem C2.** Find the Bloch vector for ρ = [[0.6, 0.2i], [-0.2i, 0.4]].

**Problem C3.** Compute ⟨σₓ⟩, ⟨σᵧ⟩, ⟨σ_z⟩ for the state in C2.

**Problem C4.** Show that if ρ has Bloch vector r⃗, then tr(ρ²) = (1 + |r⃗|²)/2.

**Problem C5.** Prove: The von Neumann entropy S(ρ) ≥ 0, with equality iff ρ is pure.

### Section D: Partial Trace (25 min)

**Problem D1.** Compute ρ_A for |ψ⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2.

**Problem D2.** For the state in D1, is it entangled? Compute the entanglement entropy.

**Problem D3.** Show that tr_B(A⊗B) = A·tr(B).

**Problem D4.** For the W state |W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3, compute ρ_A (first qubit).

**Problem D5.** Compare the 2-qubit reduced density matrices of GHZ and W states. Which is more entangled?

### Section E: Quantum Channels (15 min)

**Problem E1.** Apply the depolarizing channel with p = 0.2 to |0⟩⟨0|.

**Problem E2.** Show that the composition of two depolarizing channels is depolarizing.

**Problem E3.** Prove that amplitude damping drives any state to |0⟩ as γ → 1.

### Section F: Integration Problems (15 min)

**Problem F1.** A Bell state |Φ⁺⟩ undergoes local depolarizing noise with p = 0.1 on each qubit. Compute the final negativity.

**Problem F2.** Use SVD to show that a bipartite pure state |ψ⟩ is product iff its coefficient matrix has rank 1.

**Problem F3.** Explain why the reduced density matrix of a subsystem of a pure entangled state is always mixed.

---

## ✅ Solutions Outline

### A1 Solution:
A*A = diag(9, 4) → σ₁ = 3, σ₂ = 2
V = I₂, U = [e₁ | e₂ | e₃], Σ = [[3,0],[0,2],[0,0]]

### A3 Solution:
C = (1/√3)[[1,1],[0,1]]
SVD gives Schmidt coefficients ≈ (0.888, 0.460)
State is entangled (two nonzero coefficients)

### C1 Solution:
ρ = (2/3)|0⟩⟨0| + (1/3)|+⟩⟨+| = [[5/6, 1/6], [1/6, 1/6]]
tr(ρ²) = 25/36 + 1/36 + 1/36 + 1/36 = 28/36 = 7/9

### D1 Solution:
|ψ⟩ = (|0⟩+|1⟩)⊗(|0⟩+|1⟩)/2 = |+⟩⊗|+⟩
Product state! ρ_A = |+⟩⟨+| = [[1/2, 1/2], [1/2, 1/2]]

### D4 Solution:
|W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3
ρ_ABC = |W⟩⟨W|
ρ_A = (1/3)|0⟩⟨0| + (2/3)|1⟩⟨1| (asymmetric!)
Wait, let me recalculate: 
ρ_A = tr_BC(|W⟩⟨W|) = (2/3)|0⟩⟨0| + (1/3)|1⟩⟨1|
(Two terms have qubit A = 0, one term has qubit A = 1)

---

## 🎯 Self-Assessment Rubric

### Mastery Indicators

**Level 1 - Recognition:**
- [ ] Know SVD, tensor product, density matrix definitions
- [ ] Identify entangled vs product states

**Level 2 - Computation:**
- [ ] Compute SVD for small matrices
- [ ] Calculate tensor products
- [ ] Find reduced density matrices

**Level 3 - Application:**
- [ ] Use Schmidt decomposition for entanglement
- [ ] Apply quantum channels
- [ ] Compute entanglement entropy

**Level 4 - Analysis:**
- [ ] Prove properties of partial trace
- [ ] Analyze entanglement dynamics
- [ ] Design quantum protocols

**Level 5 - Synthesis:**
- [ ] Derive new entanglement measures
- [ ] Design error correction schemes
- [ ] Connect math to physical experiments

**Your Level:** ___________

---

## 🔄 Spaced Repetition Cards

### Card 1
**Front:** What is the SVD of a matrix A?
**Back:** A = UΣV* where U, V are unitary and Σ is diagonal with non-negative entries (singular values).

### Card 2
**Front:** What is the Schmidt decomposition?
**Back:** Any bipartite pure state: |ψ⟩ = Σᵢ λᵢ|aᵢ⟩|bᵢ⟩ with orthonormal {|aᵢ⟩}, {|bᵢ⟩} and λᵢ = singular values of coefficient matrix.

### Card 3
**Front:** How do you detect entanglement in a pure bipartite state?
**Back:** Schmidt rank > 1, or equivalently ρ_A is mixed (purity < 1), or entanglement entropy > 0.

### Card 4
**Front:** What is the partial trace?
**Back:** tr_B(ρ_AB) = Σⱼ(I⊗⟨j|)ρ(I⊗|j⟩) — traces out system B to get reduced state of A.

### Card 5
**Front:** What's the difference between pure and mixed states?
**Back:** Pure: ρ = |ψ⟩⟨ψ|, tr(ρ²) = 1. Mixed: ρ = Σpᵢ|ψᵢ⟩⟨ψᵢ|, tr(ρ²) < 1.

### Card 6
**Front:** What is a quantum channel?
**Back:** A completely positive trace-preserving map ℰ(ρ) = Σₖ Kₖ ρ Kₖ† where Σₖ Kₖ†Kₖ = I.

---

## 🚀 Preview: Week 19 — Complex Analysis I

**What's coming:**
- Complex numbers and the complex plane
- Analytic functions
- Cauchy-Riemann equations
- Elementary functions (exp, log, trig)

**QM connections:**
- Wave functions are complex-valued
- Complex amplitudes and interference
- Analytic structure of Green's functions

**Key transition:** From discrete (matrices) to continuous (functions)!

---

## 📋 Week 18 Completion Checklist

### Concepts Mastered
- [ ] SVD computation and interpretation
- [ ] Low-rank approximation via SVD
- [ ] Schmidt decomposition
- [ ] Tensor products and Kronecker products
- [ ] Multi-qubit state spaces
- [ ] Pure vs mixed density matrices
- [ ] Partial trace operation
- [ ] Entanglement entropy
- [ ] Quantum channels (Kraus representation)

### Computational Skills
- [ ] NumPy SVD and tensor operations
- [ ] Density matrix simulation
- [ ] Entanglement quantification
- [ ] Channel simulation
- [ ] Lindblad evolution

### Materials Completed
- [ ] Day 120: SVD Foundations
- [ ] Day 121: SVD Applications
- [ ] Day 122: Tensor Products
- [ ] Day 123: Partial Trace
- [ ] Day 124: Density Matrices
- [ ] Day 125: Computational Lab
- [ ] Day 126: Review (today)

---

## 📖 Gap-Filling Resources

**SVD:**
- Strang, Chapter 7
- Trefethen & Bau, Lectures 4-5

**Tensor Products:**
- Nielsen & Chuang, Section 2.1.7
- Preskill Notes, Chapter 2

**Density Matrices:**
- Sakurai, Chapter 3.4
- Nielsen & Chuang, Section 2.4

---

## 📝 Reflection Questions

1. Why is the partial trace the correct way to describe subsystems?

2. How does entanglement relate to the inability to describe subsystems independently?

3. What's the physical meaning of quantum channels having Kraus representations?

4. How has your understanding of quantum correlations evolved this week?

---

**End of Week 18 — Advanced Linear Algebra ✓**

**Month 5 Progress: 2/4 weeks complete**

*Next: Week 19 — Complex Analysis I (Complex Numbers and Analytic Functions)*

---

*"Entanglement is perhaps the most profound difference between quantum and classical physics. It is the characteristic trait of quantum mechanics."*
— Erwin Schrödinger

*"Linear algebra is the mathematics of quantum mechanics. If you understand linear algebra deeply, you understand the mathematical structure of quantum theory."*
— Scott Aaronson
