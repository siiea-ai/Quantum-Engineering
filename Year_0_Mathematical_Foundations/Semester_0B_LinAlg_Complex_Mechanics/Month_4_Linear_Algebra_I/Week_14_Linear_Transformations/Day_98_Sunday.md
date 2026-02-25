# Day 98: Week 14 Review — Linear Transformations Comprehensive Assessment

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 12:00 PM | 2 hours | Concept Review & Integration |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Comprehensive Problem Set |

**Total Study Time: 4 hours (Sunday schedule)**

---

## 🎯 Review Objectives

By the end of today, you should:

1. Have internalized all Week 14 concepts with deep understanding
2. Fluently translate between transformations and matrices
3. Apply rank-nullity theorem to any problem
4. Understand quantum operators as linear transformations
5. Be fully prepared for Week 15 (Eigenvalues and Eigenvectors)

---

## 📚 Week 14 Concept Map

```
                    LINEAR TRANSFORMATIONS
                            │
           ┌────────────────┼────────────────┐
           │                │                │
      DEFINITION       REPRESENTATION      STRUCTURE
           │                │                │
      T: V → W         Matrix [T]_B      Kernel & Range
      T(αu+βv)=        Columns =         ker(T) = {v: Tv=0}
      αT(u)+βT(v)      T(basis vectors)  range(T) = {Tv: v∈V}
           │                │                │
           └────────────────┴────────────────┘
                            │
                    RANK-NULLITY THEOREM
                            │
                dim(ker T) + dim(range T) = dim(V)
                    nullity + rank = n
                            │
                    ┌───────┴───────┐
                    │               │
               INJECTIVE       SURJECTIVE
               ker T = {0}     range T = W
               nullity = 0     rank = dim W
                    │               │
                    └───────┬───────┘
                            │
                       BIJECTIVE
                    (Invertible)
                            │
                    QUANTUM CONNECTION
                            │
              ┌─────────────┼─────────────┐
              │             │             │
         Operators     Unitarity      Gates
         on States     U†U = I      Quantum
                                    Circuits
```

---

## 🔄 Morning Session: Concept Review (2 hours)

### Part 1: Core Definitions Recall (30 min)

**Without looking at notes**, write down:

#### 1. Linear Transformation Definition
A function T: V → W between vector spaces is **linear** if:
- **Additivity:** T(u + v) = T(u) + T(v) for all u, v ∈ V
- **Homogeneity:** T(cv) = cT(v) for all c ∈ F, v ∈ V

Equivalently (combined): T(αu + βv) = αT(u) + βT(v)

**Self-check:** Why are these two conditions equivalent to the combined form?

#### 2. Matrix Representation
Given bases B = {v₁, ..., vₙ} for V and C = {w₁, ..., wₘ} for W:

The **matrix of T** with respect to B and C is the m×n matrix [T]ᶜᴮ where:
- Column j = [T(vⱼ)]ᶜ (coordinates of T(vⱼ) in basis C)

**Key formula:** [T(v)]ᶜ = [T]ᶜᴮ · [v]ᴮ

**Self-check:** Why do the columns come from applying T to basis vectors?

#### 3. Kernel and Range
- **Kernel (null space):** ker(T) = {v ∈ V : T(v) = 0}
- **Range (image):** range(T) = {T(v) : v ∈ V} = {w ∈ W : ∃v, T(v) = w}

Both are subspaces! (Verify this.)

**Self-check:** How do you find kernel and range from a matrix?

#### 4. Rank-Nullity Theorem
For T: V → W with V finite-dimensional:

$$\text{dim}(\ker T) + \text{dim}(\text{range } T) = \text{dim}(V)$$

Equivalently: **nullity(T) + rank(T) = n**

**Self-check:** What does this theorem tell you about injectivity?

### Part 2: Key Theorems and Their Meaning (30 min)

#### Theorem 1: Linearity and Zero
T is linear ⟹ T(0) = 0

*Proof:* T(0) = T(0·v) = 0·T(v) = 0

*Why it matters:* Zero always maps to zero; origin is preserved.

#### Theorem 2: Kernel is a Subspace
ker(T) is a subspace of V.

*Proof:* 
- 0 ∈ ker(T) since T(0) = 0 ✓
- u, v ∈ ker(T) ⟹ T(u+v) = T(u)+T(v) = 0+0 = 0 ⟹ u+v ∈ ker(T) ✓
- v ∈ ker(T), c ∈ F ⟹ T(cv) = cT(v) = c·0 = 0 ⟹ cv ∈ ker(T) ✓

*Why it matters:* Kernel has a dimension (nullity).

#### Theorem 3: Range is a Subspace
range(T) is a subspace of W.

*Why it matters:* We can find a basis for the range (columns of matrix!).

#### Theorem 4: Injectivity Characterization
T is injective ⟺ ker(T) = {0}

*Proof (⟹):* If T injective and T(v) = 0 = T(0), then v = 0.
*Proof (⟸):* If ker(T) = {0} and T(u) = T(v), then T(u-v) = 0, so u-v = 0, so u = v.

*Why it matters:* Check injectivity by computing kernel.

#### Theorem 5: Dimension Constraint
If T: V → W is linear:
- T injective ⟹ dim(V) ≤ dim(W)
- T surjective ⟹ dim(V) ≥ dim(W)
- T bijective ⟹ dim(V) = dim(W)

*Why it matters:* Dimensions constrain what's possible.

#### Theorem 6: Matrix Operations
For linear maps S: U → V and T: V → W:
- [T ∘ S] = [T][S] (composition ↔ matrix multiplication)
- [T⁻¹] = [T]⁻¹ (inverse map ↔ inverse matrix)

*Why it matters:* Matrix algebra = transformation algebra.

### Part 3: Computational Procedures (30 min)

#### Procedure 1: Find Matrix of Transformation

**Given:** T: V → W and bases B for V, C for W

**Steps:**
1. For each basis vector vⱼ ∈ B:
   a. Compute T(vⱼ)
   b. Express T(vⱼ) as linear combination of C
   c. These coordinates form column j of [T]

**Example:** T: ℝ² → ℝ³ given by T(x,y) = (x+y, x-y, 2y)

Standard bases: B = {e₁, e₂}, C = {f₁, f₂, f₃}

T(e₁) = T(1,0) = (1, 1, 0) = 1·f₁ + 1·f₂ + 0·f₃
T(e₂) = T(0,1) = (1, -1, 2) = 1·f₁ - 1·f₂ + 2·f₃

[T] = [1  1]
      [1 -1]
      [0  2]

#### Procedure 2: Find Kernel

**Given:** Matrix A (m×n)

**Steps:**
1. Set up Ax = 0
2. Row reduce A to echelon form
3. Free variables correspond to basis vectors of kernel
4. Express solution in parametric form

**Example:** A = [1 2 1]
                [2 4 2]

Row reduce: [1 2 1] → [1 2 1]
            [2 4 2]   [0 0 0]

Free variables: x₂ = s, x₃ = t
x₁ = -2s - t

Solution: (x₁, x₂, x₃) = s(-2, 1, 0) + t(-1, 0, 1)

ker(A) = span{(-2, 1, 0), (-1, 0, 1)}
nullity = 2

#### Procedure 3: Find Range

**Given:** Matrix A (m×n)

**Steps:**
1. Row reduce A
2. Pivot columns of original A form basis for range
3. Range = column space of A

**Example:** Same A as above

Pivot column: column 1 only
range(A) = span{(1, 2)} (column 1 of original A)
rank = 1

**Check:** 2 + 1 = 3 ✓ (rank-nullity)

### Part 4: Quantum Connections Review (30 min)

#### Linear Operators in QM

| Concept | Linear Algebra | Quantum Mechanics |
|---------|---------------|-------------------|
| State space | Complex vector space V | Hilbert space ℋ |
| State | Vector v ∈ V | Ket \|ψ⟩ ∈ ℋ |
| Operator | Linear map T: V → V | Observable Â, Gate Û |
| Matrix | [T] in some basis | Matrix rep in comp. basis |
| Kernel | ker(T) = {v: Tv = 0} | States annihilated |
| Range | range(T) | Accessible states |

#### Unitary Operators (Quantum Gates)

U is **unitary** if U†U = UU† = I

Properties:
1. ||Uv|| = ||v|| (preserves norms → probability conservation)
2. ⟨Uu, Uv⟩ = ⟨u, v⟩ (preserves inner products)
3. U⁻¹ = U† (inverse = adjoint)
4. |det(U)| = 1
5. Eigenvalues have |λ| = 1

**Why unitarity?** Quantum evolution must be reversible and preserve probability.

#### Hermitian Operators (Observables)

A is **Hermitian** if A† = A

Properties:
1. Eigenvalues are real (measurement outcomes)
2. Eigenvectors for different eigenvalues are orthogonal
3. Spectral theorem: A = Σᵢ λᵢ |i⟩⟨i|

**Why Hermitian?** Measurement outcomes must be real numbers.

#### Key Gate Facts

| Gate | Matrix | Action | Hermitian? | Unitary? |
|------|--------|--------|------------|----------|
| X | [[0,1],[1,0]] | Bit flip | ✓ | ✓ |
| Y | [[0,-i],[i,0]] | Y rotation | ✓ | ✓ |
| Z | [[1,0],[0,-1]] | Phase flip | ✓ | ✓ |
| H | [[1,1],[1,-1]]/√2 | Superposition | ✓ | ✓ |
| S | [[1,0],[0,i]] | √Z | ✗ | ✓ |
| T | [[1,0],[0,e^(iπ/4)]] | π/8 gate | ✗ | ✓ |

**Note:** Pauli gates and Hadamard are both Hermitian AND unitary (they're their own inverses).

---

## 📝 Afternoon Session: Comprehensive Problem Set (2 hours)

### Section A: Definitions and Basic Properties (20 min)

**Problem A1.** Prove that T: ℝ² → ℝ² given by T(x,y) = (x+y, 2x-y) is linear.

**Problem A2.** Show that S: ℝ² → ℝ given by S(x,y) = xy is NOT linear.

**Problem A3.** Let T: V → W be linear. Prove: T is injective ⟺ T maps linearly independent sets to linearly independent sets.

**Problem A4.** If T: ℝ³ → ℝ² is a linear surjection, what is dim(ker T)?

### Section B: Matrix Representation (25 min)

**Problem B1.** Find the matrix of T: ℝ³ → ℝ² given by T(x,y,z) = (x-y+2z, 3x+y-z) with respect to standard bases.

**Problem B2.** Let D: 𝒫₂(ℝ) → 𝒫₁(ℝ) be differentiation: D(p) = p'.
Find [D] with respect to bases {1, x, x²} and {1, x}.

**Problem B3.** T: ℝ² → ℝ² has [T]_S = [[1,2],[3,4]] in standard basis S.
Find [T]_B where B = {(1,1), (1,-1)}.

**Problem B4.** If [T]_B = A and [S]_B = B (same basis), what is [T∘S]_B?

### Section C: Kernel and Range (30 min)

**Problem C1.** Find a basis for the kernel and range of:
A = [1  2  -1  0]
    [2  4   1  3]
    [3  6   0  3]

Verify rank-nullity theorem.

**Problem C2.** Let T: ℝ⁴ → ℝ³ be linear with rank(T) = 2.
(a) What is nullity(T)?
(b) Is T injective? Surjective?
(c) What are the possible dimensions of range(T) ∩ range(S) for another such T = S?

**Problem C3.** Define T: M₂ₓ₂(ℝ) → M₂ₓ₂(ℝ) by T(A) = A - Aᵀ.
(a) Prove T is linear.
(b) Find ker(T). What matrices are in it?
(c) Find range(T). What matrices are in it?

**Problem C4.** Let V = C([0,1], ℝ) (continuous functions) and T: V → V by T(f) = xf(x).
(a) Prove T is linear.
(b) What is ker(T)?
(c) Is T injective? Surjective?

### Section D: Rank-Nullity Applications (20 min)

**Problem D1.** A 5×7 matrix has nullity 3. What is its rank?

**Problem D2.** Can a linear map T: ℝ⁵ → ℝ³ be injective? Surjective? Both?

**Problem D3.** Let T: ℝⁿ → ℝⁿ be linear. Prove: T injective ⟺ T surjective.

**Problem D4.** If A is m×n with m < n, prove Ax = 0 has nontrivial solutions.

### Section E: Composition and Inverses (15 min)

**Problem E1.** Let S: ℝ² → ℝ³ and T: ℝ³ → ℝ² be linear.
(a) Can T ∘ S be invertible?
(b) Can S ∘ T be invertible?

**Problem E2.** If A is 3×3 with rank 2, is A invertible? Why?

**Problem E3.** Prove: (AB)⁻¹ = B⁻¹A⁻¹ when both inverses exist.

### Section F: Quantum Applications (10 min)

**Problem F1.** Verify that the Hadamard gate H is unitary and Hermitian.

**Problem F2.** The CNOT gate acts on ℂ⁴. Find its kernel and range.
CNOT = [[1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0]]

**Problem F3.** Show that composition of unitary operators is unitary.

---

## 📊 Solutions and Key Steps

### Section A Solutions

**A1.** T(α(x₁,y₁) + β(x₂,y₂)) = T(αx₁+βx₂, αy₁+βy₂)
= (αx₁+βx₂+αy₁+βy₂, 2(αx₁+βx₂)-(αy₁+βy₂))
= α(x₁+y₁, 2x₁-y₁) + β(x₂+y₂, 2x₂-y₂)
= αT(x₁,y₁) + βT(x₂,y₂) ✓

**A2.** S(2·(1,1)) = S(2,2) = 4, but 2·S(1,1) = 2·1 = 2. Not equal.

**A3.** (⟹) If T injective and Σcᵢvᵢ = 0 for T(vᵢ) images, then T(Σcᵢvᵢ) = 0 = T(0), so Σcᵢvᵢ = 0, so all cᵢ = 0.
(⟸) If {v} maps to independent set and T(v) = 0 = T(0), then {v,0} → {0,0} dependent, contradiction unless v = 0.

**A4.** rank(T) = dim(ℝ²) = 2 (surjective). By rank-nullity: 3 = 2 + nullity, so nullity = 1.

### Section B Solutions

**B1.** T(1,0,0) = (1,3), T(0,1,0) = (-1,1), T(0,0,1) = (2,-1)
[T] = [[1,-1,2],[3,1,-1]]

**B2.** D(1) = 0 = 0·1 + 0·x
D(x) = 1 = 1·1 + 0·x
D(x²) = 2x = 0·1 + 2·x
[D] = [[0,1,0],[0,0,2]]

**B3.** Change of basis: [T]_B = P⁻¹[T]_S P where P = [[1,1],[1,-1]]

**B4.** [T∘S]_B = [T]_B[S]_B = AB

### Section C Solutions

**C1.** Row reduce to find:
- Pivot positions → rank
- Free variables → nullity
- Pivot columns of original → range basis
- Parametric solution → kernel basis

**C2.** (a) 4 - 2 = 2
(b) Not injective (nullity ≠ 0), not surjective (rank < 3)
(c) 0 ≤ dim(intersection) ≤ 2

**C3.** ker(T) = symmetric matrices, range(T) = skew-symmetric matrices

**C4.** ker(T) = {0} (only zero function is zero at all x), T is injective but not surjective (constant functions not in range).

### Section F Solutions

**F1.** H† = H (symmetric, real), H†H = H² = I ✓

**F2.** ker(CNOT) = {0} (it's invertible), range = ℂ⁴

**F3.** (UV)†(UV) = V†U†UV = V†V = I ✓

---

## 🎯 Self-Assessment Rubric

### Mastery Levels

**Level 1 - Recognition:**
- [ ] State definition of linear transformation
- [ ] Identify matrices as representing transformations
- [ ] Know what kernel and range mean

**Level 2 - Comprehension:**
- [ ] Verify linearity of given functions
- [ ] Understand why matrix columns come from basis vectors
- [ ] Interpret rank-nullity geometrically

**Level 3 - Application:**
- [ ] Find matrices from transformation rules
- [ ] Compute kernel and range from matrices
- [ ] Apply rank-nullity to determine injectivity/surjectivity

**Level 4 - Analysis:**
- [ ] Change basis for matrix representations
- [ ] Prove properties using kernel/range
- [ ] Analyze quantum operators

**Level 5 - Synthesis:**
- [ ] Design transformations with specific properties
- [ ] Connect to advanced quantum concepts
- [ ] Prove new theorems

**Your level:** _______________

---

## 🔄 Spaced Repetition Cards

### Card 1
**Front:** What makes T: V → W a linear transformation?
**Back:** T(αu + βv) = αT(u) + βT(v) for all scalars α, β and vectors u, v.

### Card 2
**Front:** How do you find the matrix of T in bases B, C?
**Back:** Column j = coordinates of T(bⱼ) in basis C, where bⱼ is j-th basis vector of B.

### Card 3
**Front:** State the rank-nullity theorem.
**Back:** dim(ker T) + dim(range T) = dim(V), or nullity + rank = n.

### Card 4
**Front:** When is T: V → W injective?
**Back:** ker(T) = {0}, equivalently nullity(T) = 0.

### Card 5
**Front:** What is a unitary operator?
**Back:** U†U = UU† = I. Preserves norms, inner products. Eigenvalues on unit circle.

### Card 6
**Front:** Why must quantum gates be unitary?
**Back:** To preserve probability (||Uψ|| = ||ψ||) and be reversible (U⁻¹ exists).

---

## 📋 Week 14 Completion Checklist

### Knowledge Checkpoints
- [ ] Define and verify linear transformations
- [ ] Find matrix representations in any basis
- [ ] Compute kernel and range
- [ ] Apply rank-nullity theorem
- [ ] Change bases for matrix representations
- [ ] Understand composition ↔ multiplication
- [ ] Analyze quantum operators as linear maps
- [ ] Verify unitarity and Hermiticity

### Practical Skills
- [ ] Matrix-from-transformation algorithm
- [ ] Kernel computation via row reduction
- [ ] Range as column space
- [ ] Quantum gate implementation in NumPy

### Materials Completed
- [ ] Day 92: Linear Maps — Definition and Examples
- [ ] Day 93: Matrix Representations
- [ ] Day 94: Kernel (Null Space) and Range (Image)
- [ ] Day 95: Matrix Operations and Composition
- [ ] Day 96: Rank-Nullity Theorem
- [ ] Day 97: Computational Lab
- [ ] Day 98: Review and Assessment (today)
- [ ] All practice problems attempted
- [ ] Anki cards created

---

## 🚀 Preview: Week 15 — Eigenvalues and Eigenvectors

**What's coming:**
- Eigenvalue equation: Av = λv
- Characteristic polynomial: det(A - λI) = 0
- Finding eigenvalues and eigenvectors
- Diagonalization: A = PDP⁻¹
- Spectral properties

**QM preview:**
- Eigenvalues = measurement outcomes
- Eigenvectors = definite-value states
- Spectral decomposition = complete set of observables

**Key transition:** This week we asked "what does T do to vectors?" Next week: "which vectors does T only scale?"

**Preparation:** 
- Review determinants (we'll need them for characteristic polynomial)
- Read Axler Chapter 5.A (Eigenvalues and Eigenvectors)

---

## 📓 Reflection Questions

Before ending today, answer:

1. What is the geometric meaning of the kernel of a 3D → 2D transformation?

2. Why is rank-nullity called a "conservation law"?

3. How does changing basis affect the kernel and range? (Hint: it doesn't!)

4. In quantum mechanics, why are projection operators important? (Think about measurement.)

---

## 📊 Week 14 Summary Table

| Day | Topic | Key Concept | QM Connection |
|-----|-------|-------------|---------------|
| 92 | Linear Maps | T(αu+βv) = αT(u)+βT(v) | Operators on states |
| 93 | Matrices | [T] from basis vectors | Gate matrices |
| 94 | Kernel/Range | Subspaces of T | Annihilated/reachable states |
| 95 | Operations | [TS] = [T][S] | Circuit composition |
| 96 | Rank-Nullity | rank + nullity = n | Dimension constraints |
| 97 | Lab | NumPy implementation | Quantum simulator |
| 98 | Review | Integration | Full picture |

---

**End of Week 14 — Linear Transformations ✓**

*Next: Week 15 — Eigenvalues and Eigenvectors*

---

*"In mathematics you don't understand things. You just get used to them."*
— John von Neumann
