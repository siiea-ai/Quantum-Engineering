# Day 231: Week 33 Review — Metric Spaces

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning I** | 2 hours | Concept review and synthesis |
| **Morning II** | 1.5 hours | Key theorems and proofs |
| **Afternoon I** | 2 hours | Comprehensive problem set |
| **Afternoon II** | 1.5 hours | Quantum mechanics applications |
| **Evening** | 1 hour | Self-assessment and preparation for Week 34 |

## Learning Objectives

By the end of today, you will be able to:

1. **Synthesize** all metric space concepts from the week
2. **Prove** key theorems: completeness, Banach fixed-point, Arzelà-Ascoli
3. **Solve** comprehensive problems combining multiple concepts
4. **Apply** metric space theory to quantum mechanics problems
5. **Identify** areas needing additional review
6. **Prepare** for the transition to normed and Banach spaces

---

## 1. Week Summary: Core Concepts

### Day 225: Metric Space Fundamentals

**Definition:** A metric space (X, d) consists of a set X and a distance function d: X × X → [0, ∞) satisfying:

$$\boxed{\begin{aligned}
&\text{(M1) } d(x,y) \geq 0, \quad d(x,y) = 0 \iff x = y \\
&\text{(M2) } d(x,y) = d(y,x) \\
&\text{(M3) } d(x,z) \leq d(x,y) + d(y,z)
\end{aligned}}$$

**Key Spaces:**
| Space | Metric | QM Connection |
|-------|--------|---------------|
| ℝⁿ | $d_p(x,y) = \left(\sum\|x_i-y_i\|^p\right)^{1/p}$ | Position/momentum |
| l² | $d(x,y) = \sqrt{\sum\|x_n-y_n\|^2}$ | Discrete bases |
| L² | $d(f,g) = \sqrt{\int\|f-g\|^2 d\mu}$ | State space |
| C[a,b] | $d_\infty(f,g) = \sup\|f-g\|$ | Bounded operators |

### Day 226: Convergence and Continuity

**Convergence:** xₙ → x ⟺ d(xₙ, x) → 0

**Open ball:** B(x, r) = {y ∈ X : d(x, y) < r}

**Open set:** Contains a ball around each point

**Closed set:** Complement of open set; contains all its limit points

**Continuity (three equivalent definitions):**
1. Sequential: xₙ → x ⟹ f(xₙ) → f(x)
2. ε-δ: ∀ε > 0, ∃δ > 0: d(x, x₀) < δ ⟹ d(f(x), f(x₀)) < ε
3. Topological: Preimages of open sets are open

### Day 227: Completeness

**Cauchy sequence:** ∀ε > 0, ∃N: m, n ≥ N ⟹ d(xₘ, xₙ) < ε

**Complete space:** Every Cauchy sequence converges

**Banach space:** Complete normed vector space

**Key result:** L² is complete (Riesz-Fischer theorem)

### Day 228: Banach Fixed-Point Theorem

**Contraction:** d(Tx, Ty) ≤ k·d(x, y) for some k < 1

**Theorem:** If (X, d) is complete and T: X → X is a contraction, then:
1. T has a unique fixed point x*
2. xₙ₊₁ = T(xₙ) converges to x* from any start
3. d(xₙ, x*) ≤ kⁿ/(1-k) · d(x₀, x₁)

**Applications:** ODEs (Picard), integral equations, SCF methods

### Day 229: Compactness

**Sequential compactness:** Every sequence has a convergent subsequence

**Total boundedness:** Finitely many ε-balls cover X for all ε > 0

**Theorem:** Compact ⟺ Complete + Totally bounded

**Heine-Borel (ℝⁿ):** Compact ⟺ Closed + Bounded

**Arzelà-Ascoli:** F ⊆ C[a,b] is relatively compact ⟺ uniformly bounded + equicontinuous

**Compact operators:** T(B_X) has compact closure

### Day 230: Completion

**Construction:** X̃ = C(X)/∼ where (xₙ) ∼ (yₙ) ⟺ d(xₙ, yₙ) → 0

**Completion metric:** d̃([(xₙ)], [(yₙ)]) = lim d(xₙ, yₙ)

**Properties:**
- X embeds isometrically in X̃
- X is dense in X̃
- X̃ is complete
- Unique up to isometry

---

## 2. Theorem Summary

### Fundamental Results

| Theorem | Hypothesis | Conclusion |
|---------|------------|------------|
| Limits unique | (X, d) metric space | If xₙ → x and xₙ → y, then x = y |
| Closed ⟺ Limits | F ⊆ X | F closed ⟺ F contains all limits of sequences in F |
| Continuous = Bounded | T linear, X, Y normed | T continuous ⟺ T bounded |
| Convergent ⟹ Cauchy | (X, d) metric | xₙ → x ⟹ (xₙ) Cauchy |
| Cauchy ⟹ Convergent | (X, d) **complete** | (xₙ) Cauchy ⟹ ∃x: xₙ → x |

### The Big Three

**1. Banach Fixed-Point Theorem**
- Conditions: Complete + Contraction
- Gives: Existence, uniqueness, algorithm, error bound

**2. Arzelà-Ascoli Theorem**
- Conditions: Uniformly bounded + Equicontinuous
- Gives: Relatively compact in C[a,b]

**3. Completion Theorem**
- Construction: Cauchy sequence equivalence classes
- Gives: Minimal complete extension

---

## 3. Proof Techniques

### Technique 1: The Triangle Inequality Trick

**Pattern:** To show d(x, z) is small, factor through an intermediate point:

$$d(x, z) \leq d(x, y) + d(y, z)$$

**Example:** Proving limits are unique.

### Technique 2: Choosing ε and N

**Pattern:** Given "for all ε" to prove, "there exists N" to use.

**Example:** Proving convergent ⟹ Cauchy.
- We know: xₙ → x means for all ε, ∃N: n ≥ N ⟹ d(xₙ, x) < ε/2
- We want: for all ε, ∃N: m, n ≥ N ⟹ d(xₘ, xₙ) < ε
- Trick: d(xₘ, xₙ) ≤ d(xₘ, x) + d(x, xₙ) < ε/2 + ε/2 = ε

### Technique 3: Diagonal Argument

**Pattern:** Extract a convergent subsequence by iteratively refining.

**Example:** Arzelà-Ascoli proof.
- Sequence in C[a,b], want convergent subsequence
- On a dense countable set {r₁, r₂, ...}, extract subsequences

### Technique 4: Contraction Estimates

**Pattern:** For contractions, errors decay geometrically.

$$d(x_{n+1}, x_n) \leq k^n d(x_1, x_0)$$

Use geometric series: 1 + k + k² + ... = 1/(1-k)

---

## 4. Comprehensive Problem Set

### Problem 1: Metric Verification

**Question:** For X = ℝ, define d(x, y) = |arctan(x) - arctan(y)|. Is this a metric?

**Solution:**

**(M1) Positivity:**
- d(x, y) = |arctan(x) - arctan(y)| ≥ 0 ✓
- d(x, y) = 0 ⟺ arctan(x) = arctan(y) ⟺ x = y (arctan is bijective) ✓

**(M2) Symmetry:**
- d(x, y) = |arctan(x) - arctan(y)| = |arctan(y) - arctan(x)| = d(y, x) ✓

**(M3) Triangle Inequality:**
$$|arctan(x) - arctan(z)| \leq |arctan(x) - arctan(y)| + |arctan(y) - arctan(z)|$$
This holds because | · | satisfies the triangle inequality. ✓

**Conclusion:** Yes, d is a metric. Note: d(x, y) < π for all x, y (ℝ is bounded in this metric!). ∎

### Problem 2: Completeness

**Question:** Is (C[0,1], d₂) complete where d₂(f, g) = (∫₀¹ |f-g|² dx)^{1/2}?

**Solution:**

**Claim:** No, (C[0,1], d₂) is NOT complete.

**Proof:** Construct a Cauchy sequence of continuous functions whose L² limit is discontinuous.

Define:
$$f_n(x) = \begin{cases}
0 & x \leq 1/2 - 1/n \\
\frac{1}{2} + \frac{n(x - 1/2)}{2} & 1/2 - 1/n < x < 1/2 + 1/n \\
1 & x \geq 1/2 + 1/n
\end{cases}$$

Each fₙ is continuous (linear ramp of width 2/n).

**Cauchy check:**
$$d_2(f_m, f_n)^2 = \int_0^1 |f_m - f_n|^2 dx$$

Both fₘ and fₙ differ only on a shrinking interval of width O(1/min(m,n)).
So d₂(fₘ, fₙ) → 0 as m, n → ∞.

**Limit:** The L² limit is the step function
$$f(x) = \begin{cases} 0 & x < 1/2 \\ 1 & x > 1/2 \end{cases}$$

This is NOT continuous, so f ∉ C[0,1]. ∎

### Problem 3: Fixed-Point Application

**Question:** Prove that x = cos(sin(x)) has a unique solution in ℝ.

**Solution:**

Define T(x) = cos(sin(x)).

**Claim:** T is a contraction on ℝ.

**Proof:**
$$|T(x) - T(y)| = |\cos(\sin x) - \cos(\sin y)|$$

By mean value theorem: |cos(a) - cos(b)| ≤ |a - b| (since |sin| ≤ 1).

So: |T(x) - T(y)| ≤ |sin(x) - sin(y)|

Again by MVT: |sin(x) - sin(y)| ≤ |x - y| (since |cos| ≤ 1).

Thus: |T(x) - T(y)| ≤ |x - y|

Hmm, this gives k = 1, not a contraction!

**Better approach:** Restrict to [0, π/2].

On [0, π/2], sin is increasing, cos∘sin maps [0, π/2] into [cos(1), 1] ⊂ [0.54, 1].

Let's compute the derivative:
$$T'(x) = -\sin(\sin x) \cdot \cos x$$

On [0, 1]:
$$|T'(x)| = |\sin(\sin x)| \cdot |\cos x| \leq \sin(1) \cdot 1 \approx 0.84 < 1$$

So T is a contraction on [0, 1] with k ≈ 0.84.

By Banach fixed-point theorem, T has a unique fixed point in [0, 1].

To show uniqueness on all of ℝ: Note T(ℝ) ⊆ [-1, 1] and for |x| > 1:
$$|T(x)| = |\cos(\sin x)| \leq 1 < |x|$$

Any fixed point must satisfy |x| ≤ 1. Combined with uniqueness on [0, 1] (and T even), there's a unique fixed point.

**Numerical:** x* ≈ 0.7682. ∎

### Problem 4: Compactness

**Question:** Let K = {f ∈ C[0,1] : |f(x)| ≤ 1, |f(x) - f(y)| ≤ |x-y| for all x, y}. Is K compact in (C[0,1], ||·||_∞)?

**Solution:**

**Claim:** Yes, K is compact.

**Proof using Arzelà-Ascoli:**

**Uniformly bounded:** By definition, |f(x)| ≤ 1 for all f ∈ K, x ∈ [0,1]. So ||f||_∞ ≤ 1. ✓

**Equicontinuous:** By definition, |f(x) - f(y)| ≤ |x - y| for all f ∈ K.
Given ε > 0, choose δ = ε. Then |x - y| < δ ⟹ |f(x) - f(y)| < ε for ALL f ∈ K. ✓

**K is closed:** Need to show. Let (fₙ) ⊆ K with fₙ → f uniformly.
- |f(x)| = lim |fₙ(x)| ≤ 1 ✓
- |f(x) - f(y)| = lim |fₙ(x) - fₙ(y)| ≤ |x - y| ✓

So f ∈ K, and K is closed.

By Arzelà-Ascoli, K is relatively compact. Since K is closed, K is compact. ∎

### Problem 5: Completion

**Question:** Describe the completion of the space of polynomials P on [-1, 1] with the inner product ⟨p, q⟩ = ∫₋₁¹ p(x)q(x) dx.

**Solution:**

The completion is **L²[-1, 1]**.

**Reason:**
1. P is a pre-Hilbert space with the given inner product
2. The Weierstrass approximation theorem implies P is dense in C[-1, 1] with sup norm
3. C[-1, 1] is dense in L²[-1, 1] with L² norm
4. Therefore P is dense in L²[-1, 1]
5. L²[-1, 1] is complete (Riesz-Fischer)

The completion adds all square-integrable functions, including:
- Discontinuous functions
- Functions only defined almost everywhere
- The famous Legendre polynomials form an orthonormal basis ∎

---

## 5. Quantum Mechanics Integration

### Synthesis: The L² Framework

The metric space theory we've developed provides the mathematical foundation for quantum mechanics:

| QM Concept | Metric Space Concept |
|------------|---------------------|
| Quantum state | Element of L²(ℝ³) |
| Distinguishability | d(ψ, φ) = ||ψ - φ||₂ |
| State preparation | Convergent sequence |
| Physical limits exist | Completeness of L² |
| SCF iteration | Banach fixed-point |
| Compact operators | Discrete spectra |
| Approximation | Dense subspaces |

### Example: Time Evolution as Continuous Map

The time evolution U(t) = e^{-iHt/ℏ} defines a map:

$$U(t): L^2 \to L^2$$

**Properties from metric space theory:**

1. **Isometric:** ||U(t)ψ||₂ = ||ψ||₂ (probability conservation)

2. **Continuous:** If ψₙ → ψ in L², then U(t)ψₙ → U(t)ψ

3. **Fixed points of U(τ):** If U(τ)ψ = ψ for some τ, then ψ is a stationary state (eigenstate of H with eigenvalue Eₖ where Eₖτ/ℏ ∈ 2πℤ)

### Example: Variational Method

The variational principle for ground state energy:

$$E_0 = \inf_{\|\psi\|=1} \langle \psi | H | \psi \rangle$$

**Why does the infimum exist?**

- The set {ψ : ||ψ|| = 1} is the unit sphere in L²
- For Hamiltonians bounded below, the functional is bounded below
- But the infimum might not be attained (no ground state for free particle on ℝ)

**When is it attained?**

- If the Hamiltonian has compact resolvent (e.g., confining potential)
- Then the spectrum is discrete and the ground state exists

---

## 6. Self-Assessment

### Concept Checklist

Rate your understanding (1-5):

| Concept | Self-Rating | Notes |
|---------|-------------|-------|
| Metric axioms | | |
| Open/closed sets | | |
| Convergence | | |
| Continuity (all forms) | | |
| Cauchy sequences | | |
| Completeness | | |
| Banach fixed-point | | |
| Sequential compactness | | |
| Total boundedness | | |
| Arzelà-Ascoli | | |
| Compact operators | | |
| Completion construction | | |

### Key Questions to Answer

1. Can you state and prove the Banach fixed-point theorem?
2. Can you verify metric axioms for a proposed distance function?
3. Can you determine if a specific metric space is complete?
4. Can you apply Arzelà-Ascoli to function families?
5. Can you explain why L² completeness matters for QM?

---

## 7. Looking Ahead: Week 34

Next week we study **Normed Spaces and Banach Spaces** in detail:

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 232 | Normed Spaces | Norm axioms, examples, induced metric |
| 233 | Bounded Linear Operators | Operator norm, B(X,Y), continuity |
| 234 | Finite-Dimensional Spaces | Equivalence of norms, compactness |
| 235 | Dual Spaces | Linear functionals, Hahn-Banach |
| 236 | Baire Category Theorem | Open mapping, closed graph |
| 237 | Applications | Banach algebras, spectral theory |
| 238 | Week Review | Integration and practice |

**Preparation:**
- Review linear algebra (vector spaces, linear maps)
- Understand the relationship between norms and metrics
- Think about how bounded operators generalize matrices

---

## Summary: Week 33 Key Takeaways

### The Big Picture

1. **Metric spaces** generalize distance from ℝⁿ to abstract settings
2. **Convergence, continuity, and open sets** depend only on the metric
3. **Completeness** ensures limits exist — crucial for L²
4. **The Banach fixed-point theorem** provides existence, uniqueness, and algorithms
5. **Compactness** is a finiteness condition enabling powerful theorems
6. **Completion** fills in gaps to make any space complete

### Mathematical Maturity Gained

- **Abstraction:** Working with general metric spaces, not just ℝⁿ
- **Proof techniques:** ε-δ arguments, triangle inequality tricks
- **Construction:** Building the completion from Cauchy sequences
- **Application:** Connecting abstract theory to concrete problems

### Ready for Quantum Mechanics

The metric space framework is essential for:
- Understanding L² as the state space
- Proving existence of solutions to Schrödinger equation
- Self-consistent field calculations
- Spectral theory via compact operators
- Approximation methods (variational, perturbative)

---

## Final Practice: Mixed Problems

1. Prove that if (X, d) is compact and f: X → ℝ is continuous, then f attains its maximum.

2. Let T: C[0,1] → C[0,1] be defined by (Tf)(x) = ∫₀ˣ f(t) dt. Show T is not a contraction, but T² is.

3. Find a bounded sequence in l² with no convergent subsequence.

4. Show that the completion of (ℚ ∩ [0,1], |·|) is [0,1].

5. Prove that a compact metric space is separable (has a countable dense subset).

---

*Congratulations on completing Week 33! The metric space framework you've built will serve as the foundation for all the functional analysis to come.*
