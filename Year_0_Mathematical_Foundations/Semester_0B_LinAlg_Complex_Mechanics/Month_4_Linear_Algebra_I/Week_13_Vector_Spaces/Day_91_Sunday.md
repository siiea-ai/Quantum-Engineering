# Day 91: Week 13 Review — Vector Spaces Comprehensive Assessment

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 12:00 PM | 2 hours | Concept Review & Integration |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Comprehensive Problem Set |

**Total Study Time: 4 hours (Sunday schedule)**

---

## 🎯 Review Objectives

By the end of today, you should:

1. Have internalized all Week 13 concepts with deep understanding
2. Be able to solve problems without consulting notes
3. Understand the quantum mechanical significance of every concept
4. Be fully prepared for Week 14 (Linear Transformations)
5. Identify any remaining gaps requiring additional study

---

## 📚 Week 13 Concept Map

```
                     VECTOR SPACES
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    STRUCTURE         SUBSTRUCTURE      REPRESENTATION
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │         │      │         │      │         │
  Axioms   Field   Subspaces  Span   Basis   Coordinates
  (8 laws) (ℝ/ℂ)            Linear     │
    │                    Independence  Dimension
    │                         │            │
    └─────────────────────────┴────────────┘
                    │
            QUANTUM CONNECTION
                    │
         ┌──────────┴──────────┐
         │                     │
    State Spaces          Superposition
    (ℂⁿ, ℋ)              (Linear combo
                         of basis states)
```

---

## 🔄 Morning Session: Concept Review (2 hours)

### Part 1: Core Definitions Recall (30 min)

**Without looking at notes**, write down:

#### 1. Vector Space Definition
A vector space V over a field F is a set with two operations:
- **Vector addition:** V × V → V
- **Scalar multiplication:** F × V → V

satisfying eight axioms:
1. **Commutativity:** u + v = v + u
2. **Associativity (addition):** (u + v) + w = u + (v + w)
3. **Zero vector:** ∃ 0 ∈ V such that v + 0 = v
4. **Additive inverse:** ∀v ∈ V, ∃ -v such that v + (-v) = 0
5. **Scalar identity:** 1 · v = v
6. **Scalar compatibility:** a(bv) = (ab)v
7. **Distributivity (vectors):** a(u + v) = au + av
8. **Distributivity (scalars):** (a + b)v = av + bv

**Self-check:** Can you explain why each axiom is necessary?

#### 2. Subspace Definition
A subset W ⊆ V is a subspace if:
1. 0 ∈ W (contains zero)
2. u, v ∈ W ⟹ u + v ∈ W (closed under addition)
3. v ∈ W, c ∈ F ⟹ cv ∈ W (closed under scalar multiplication)

**Self-check:** Why do conditions 2 & 3 imply all linear combinations stay in W?

#### 3. Key Concepts Chain
- **Span:** span{v₁,...,vₖ} = {c₁v₁ + ... + cₖvₖ : cᵢ ∈ F}
- **Linear Independence:** c₁v₁ + ... + cₖvₖ = 0 ⟹ all cᵢ = 0
- **Basis:** A linearly independent spanning set
- **Dimension:** Number of vectors in any basis (unique!)

### Part 2: Theorem Review (30 min)

**Key Theorems to Remember:**

#### Theorem 1: Subspace Equivalence
W is a subspace of V ⟺ W is nonempty and closed under linear combinations

*Why it matters:* Can test with one condition instead of three

#### Theorem 2: Dimension Theorem for Subspaces
If W is a subspace of finite-dimensional V, then:
- dim(W) ≤ dim(V)
- dim(W) = dim(V) ⟹ W = V

*Why it matters:* Dimension constrains subspace structure

#### Theorem 3: Steinitz Exchange Lemma
If {u₁,...,uₘ} is independent and {w₁,...,wₙ} spans V, then m ≤ n

*Why it matters:* Proves all bases have the same size

#### Theorem 4: Extension to Basis
Any linearly independent set can be extended to a basis

*Why it matters:* Can always complete a partial basis

#### Theorem 5: Coordinate Uniqueness
Every vector has unique coordinates in a given basis

*Why it matters:* Basis provides a coordinate system

### Part 3: Quantum Connections Review (30 min)

**Key QM Correspondences:**

| Linear Algebra | Quantum Mechanics |
|----------------|-------------------|
| Vector space V | State space ℋ |
| Vector v ∈ V | State \|ψ⟩ ∈ ℋ |
| Scalar multiplication cv | Phase/amplitude scaling |
| Linear combination | Superposition |
| Basis {e₁,...,eₙ} | Computational basis {\|0⟩,...,\|n-1⟩} |
| Coordinates [v]_B | Probability amplitudes |
| Subspace W ⊂ V | Subspace of states (e.g., spin-up sector) |
| Dimension | Number of distinguishable states |

**Critical QM Facts:**
1. Quantum state spaces are complex vector spaces
2. Superposition = linear combination
3. \|ψ⟩ = α\|0⟩ + β\|1⟩ with \|α\|² + \|β\|² = 1
4. Different bases = different measurement contexts
5. dim(ℋ) for n qubits = 2ⁿ (exponential!)

### Part 4: Integration Problems (30 min)

Solve these connecting all concepts:

**Problem 1:** The states of a spin-1 particle form a 3-dimensional complex vector space.
- Write down a computational basis.
- What is the general form of any state?
- How many real parameters define a normalized state?

**Solution outline:**
- Basis: {|+1⟩, |0⟩, |-1⟩} (spin projections)
- General state: α|+1⟩ + β|0⟩ + γ|-1⟩
- Complex: 6 real parameters, minus 1 for normalization, minus 1 for global phase = 4 real parameters

**Problem 2:** Consider V = span{|0⟩|0⟩, |1⟩|1⟩} in ℂ⁴ (two-qubit space).
- Show V is a subspace.
- Find dim(V).
- Give a physical interpretation.

**Solution outline:**
- V is span of two vectors, so automatically a subspace
- Check independence: |00⟩ and |11⟩ are orthogonal, so independent
- dim(V) = 2
- Physical: V contains all states with perfect classical correlation

---

## 📝 Afternoon Session: Comprehensive Problem Set (2 hours)

### Section A: Vector Space Axioms (20 min)

**Problem A1.** Prove that in any vector space, 0 · v = 0 for all v.
*Hint:* Use 0 · v = (0 + 0) · v and distributivity.

**Problem A2.** Prove that (-1) · v = -v for all v.
*Hint:* Show v + (-1)·v = 0.

**Problem A3.** Let V = {(x, y, z) ∈ ℝ³ : x + 2y - z = 0}. Verify V is a subspace by checking the definition.

### Section B: Span and Linear Combinations (25 min)

**Problem B1.** Express (7, 2, -5) as a linear combination of (1, 0, 1), (0, 1, 1), and (1, 1, 0).

**Problem B2.** Show that {(1, 1, 0), (1, 0, 1), (0, 1, 1)} spans ℝ³.

**Problem B3.** Find span{sin(x), cos(x)} in the vector space of smooth functions. What familiar space is this?

**Problem B4.** Prove: If v ∈ span{u₁,...,uₖ}, then span{u₁,...,uₖ} = span{u₁,...,uₖ, v}.

### Section C: Linear Independence (25 min)

**Problem C1.** Determine if {(1, 2, 3), (4, 5, 6), (7, 8, 9)} is linearly independent.

**Problem C2.** Prove that {1, x, x², x³} is linearly independent in the space of polynomials.

**Problem C3.** For which values of a are {(1, a, 0), (0, 1, a), (a, 0, 1)} linearly independent?

**Problem C4.** Show that any set containing the zero vector is linearly dependent.

### Section D: Bases and Dimension (25 min)

**Problem D1.** Find a basis for the solution space of:
```
x + 2y - z + w = 0
2x + 4y - 2z + 2w = 0
```

**Problem D2.** Extend {(1, 1, 0, 0), (0, 0, 1, 1)} to a basis of ℝ⁴.

**Problem D3.** Find the dimension of the space of 2×2 symmetric matrices.

**Problem D4.** Prove: dim(U + W) = dim(U) + dim(W) - dim(U ∩ W)
*This is the dimension formula for subspaces.*

### Section E: Coordinate Systems (15 min)

**Problem E1.** Let B = {(1, 1), (1, -1)} be a basis for ℝ².
- Find [v]_B where v = (5, 3).
- If [w]_B = (2, -1), find w.

**Problem E2.** Find the change of basis matrix from B = {(1, 0), (0, 1)} to B' = {(1, 1), (1, -1)}.

**Problem E3.** In ℂ², let B = {|+⟩, |-⟩} where |±⟩ = (|0⟩ ± |1⟩)/√2.
Find [|ψ⟩]_B where |ψ⟩ = (3|0⟩ + 4i|1⟩)/5.

### Section F: Quantum Applications (10 min)

**Problem F1.** A quantum system has orthonormal basis {|a⟩, |b⟩, |c⟩}.
The state is |ψ⟩ = (2|a⟩ + i|b⟩ + 2|c⟩)/3.
- Verify this is normalized.
- Find P(measure |a⟩), P(measure |b⟩), P(measure |c⟩).
- If we measure and get outcome b, what is the post-measurement state?

**Problem F2.** Show that the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 cannot be written as |ψ⟩ ⊗ |φ⟩ for any single-qubit states. (This proves entanglement!)

---

## 🎯 Self-Assessment Rubric

### Mastery Indicators

**Level 1 - Recognition (D grade):**
- [ ] Can state definitions when prompted
- [ ] Recognizes examples of vector spaces
- [ ] Follows worked solutions

**Level 2 - Comprehension (C grade):**
- [ ] Understands why axioms are needed
- [ ] Can verify subspace conditions
- [ ] Computes span/independence for simple cases

**Level 3 - Application (B grade):**
- [ ] Solves standard problems without help
- [ ] Finds bases systematically
- [ ] Works with coordinate systems

**Level 4 - Analysis (A grade):**
- [ ] Proves theorems independently
- [ ] Recognizes which theorems apply
- [ ] Connects to quantum mechanics fluently

**Level 5 - Synthesis (A+ grade):**
- [ ] Creates novel problems
- [ ] Sees deep connections
- [ ] Could teach the material

**Your assessment:** ______________________

---

## 🔄 Spaced Repetition Cards

Add these to your Anki deck:

### Card 1
**Front:** What are the eight vector space axioms?
**Back:** 
1. Commutativity of addition
2. Associativity of addition
3. Existence of zero vector
4. Existence of additive inverse
5. Scalar identity (1·v = v)
6. Scalar compatibility (a(bv) = (ab)v)
7. Distributivity over vectors
8. Distributivity over scalars

### Card 2
**Front:** How do you test if a subset is a subspace?
**Back:** Verify: (1) Contains zero, (2) Closed under addition, (3) Closed under scalar multiplication. Or: nonempty and closed under all linear combinations.

### Card 3
**Front:** What does linear independence of {v₁,...,vₖ} mean?
**Back:** c₁v₁ + ... + cₖvₖ = 0 implies all cᵢ = 0. Equivalently: no vector is a linear combination of the others.

### Card 4
**Front:** What is a basis?
**Back:** A linearly independent spanning set. Equivalently: a minimal spanning set or a maximal independent set.

### Card 5
**Front:** In QM, what is superposition mathematically?
**Back:** A linear combination of basis states: |ψ⟩ = Σᵢ cᵢ|i⟩ where Σᵢ|cᵢ|² = 1.

### Card 6
**Front:** Why must quantum state spaces be complex?
**Back:** Real spaces can't represent interference (phase matters). Unitary evolution requires complex exponentials e^(iHt/ℏ).

---

## 🚀 Preview: Week 14 — Linear Transformations

**What's coming:**
- Linear maps: T: V → W with T(av + bw) = aT(v) + bT(w)
- Matrix representation of linear maps
- Kernel (null space) and image (range)
- Rank-nullity theorem
- Composition and inverses

**QM preview:** Operators on state spaces represent:
- Measurements (Hermitian operators)
- Time evolution (unitary operators)
- State preparation and gates

**Key transition:** Vector spaces are the stage; linear transformations are the actors.

---

## 📋 Week 13 Completion Checklist

### Knowledge Checkpoints
- [ ] Can state all 8 vector space axioms from memory
- [ ] Can verify subspace conditions
- [ ] Can compute span and check independence
- [ ] Can find bases and extend to full bases
- [ ] Can work with coordinate vectors
- [ ] Can apply change of basis
- [ ] Understands dimension theorem
- [ ] Connects all concepts to quantum mechanics

### Practical Skills
- [ ] NumPy operations for vector spaces
- [ ] Testing independence computationally
- [ ] Visualizing subspaces
- [ ] Manipulating quantum states in code

### Materials Completed
- [ ] Day 85: Vector Space Axioms and Examples
- [ ] Day 86: Subspaces and Span
- [ ] Day 87: Linear Independence
- [ ] Day 88: Bases and Dimension
- [ ] Day 89: Coordinate Systems and Change of Basis
- [ ] Day 90: Computational Lab
- [ ] Day 91: Review and Assessment (today)
- [ ] Lab Report submitted
- [ ] Anki cards created and reviewed

---

## 📖 Additional Resources for Gaps

If you struggled with any topic:

**Vector space axioms:**
- Axler, Chapter 1.A-1.B
- MIT 18.06 Lecture 1

**Subspaces:**
- Axler, Chapter 1.C
- 3Blue1Brown: Essence of Linear Algebra, Episode 2

**Linear independence:**
- Axler, Chapter 2.A
- Khan Academy: Linear independence

**Bases:**
- Axler, Chapter 2.B-2.C
- MIT 18.06 Lectures 9-10

---

## 📅 Week 14 Preparation

**Read before Monday:**
- Axler, Chapter 3.A (Linear Maps, Definition)
- Preview: Matrix of a linear map

**Mental preparation:**
Think about: "What makes a function 'linear'?" 
Hint: It preserves the operations of the vector space.

---

*"Mathematics is not about numbers, equations, computations, or algorithms: it is about understanding."*
— William Paul Thurston

---

**End of Week 13 — Vector Spaces ✓**

*Next: Week 14 — Linear Transformations and Matrices*
