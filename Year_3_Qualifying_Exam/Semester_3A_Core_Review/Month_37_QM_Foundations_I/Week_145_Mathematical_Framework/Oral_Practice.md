# Week 145: Mathematical Framework — Oral Practice Questions

## Introduction

Oral qualifying exams test your ability to explain physics concepts clearly and think on your feet. This document contains 15 oral exam questions with answer frameworks. Practice answering these out loud, ideally at a whiteboard. Time yourself — you typically have 5-10 minutes per question.

---

## Conceptual Questions

### Question 1: What is a Hilbert Space?

**Examiner's Intent:** Test understanding of mathematical foundations.

**Answer Framework:**

1. **Definition:** A Hilbert space is a complete inner product space over the complex numbers.

2. **Key Properties:**
   - Inner product defines geometry (angles, lengths)
   - Completeness: all Cauchy sequences converge within the space
   - Often infinite-dimensional in QM

3. **Physical Significance:**
   - State vectors live in Hilbert space
   - The space of square-integrable functions $$L^2$$ is a Hilbert space
   - Structure enables superposition principle

4. **Examples:**
   - $$\mathbb{C}^n$$ for n-level systems
   - $$L^2(\mathbb{R})$$ for a particle on a line
   - Fock space for many-particle systems

**Follow-up Preparation:** Why must the space be complete? What would go wrong otherwise?

---

### Question 2: Why Must Observables Be Hermitian Operators?

**Examiner's Intent:** Connect mathematical structure to physical requirements.

**Answer Framework:**

1. **Physical Requirement:** Measured values must be real numbers.

2. **Mathematical Guarantee:** Hermitian operators have real eigenvalues.

3. **Proof Sketch:**
   - For $$\hat{A}|a\rangle = a|a\rangle$$, take inner product with $$\langle a|$$
   - $$\langle a|\hat{A}|a\rangle = a\langle a|a\rangle$$
   - Hermiticity: $$\langle a|\hat{A}|a\rangle = \langle a|\hat{A}^\dagger|a\rangle^* = a^*\langle a|a\rangle$$
   - Therefore $$a = a^*$$, so $$a \in \mathbb{R}$$

4. **Additional Benefits:**
   - Eigenvectors form orthonormal basis (spectral theorem)
   - Probabilities are well-defined
   - Expectation values are real

**Follow-up Preparation:** What about non-Hermitian operators like the ladder operators?

---

### Question 3: Explain the Canonical Commutation Relation

**Examiner's Intent:** Test understanding of a fundamental QM postulate.

**Answer Framework:**

1. **Statement:** $$[\hat{x}, \hat{p}] = i\hbar$$

2. **Physical Meaning:**
   - Position and momentum cannot be simultaneously known with arbitrary precision
   - Fundamental incompatibility between these observables
   - Arises from wave nature of matter

3. **Derivation in Position Representation:**
   - $$\hat{x} \to x$$, $$\hat{p} \to -i\hbar\frac{d}{dx}$$
   - Apply to test function: $$[\hat{x}, \hat{p}]\psi = x(-i\hbar\psi') - (-i\hbar)(x\psi)' = i\hbar\psi$$

4. **Consequences:**
   - No simultaneous eigenstates of $$\hat{x}$$ and $$\hat{p}$$
   - Heisenberg uncertainty: $$\Delta x \Delta p \geq \hbar/2$$
   - Basis of all quantum phenomena

**Follow-up Preparation:** What is the classical limit? How does $$[\hat{x}, \hat{p}] = i\hbar$$ relate to Poisson brackets?

---

### Question 4: What is the Physical Meaning of the Uncertainty Principle?

**Examiner's Intent:** Distinguish correct from incorrect interpretations.

**Answer Framework:**

1. **Statement:** $$\Delta A \Delta B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$$

2. **What It IS:**
   - Fundamental limit on simultaneous knowledge
   - Property of quantum states themselves
   - Statistical statement about ensemble of measurements

3. **What It is NOT:**
   - NOT about measurement disturbance (though related)
   - NOT about experimental limitations
   - NOT saying "we just don't know" — the uncertainty is intrinsic

4. **Examples:**
   - $$\Delta x \Delta p \geq \hbar/2$$: A particle cannot have well-defined position AND momentum
   - $$\Delta E \Delta t \geq \hbar/2$$: Requires careful interpretation (time is not an operator)

5. **Minimum Uncertainty States:**
   - Gaussian wave packets for $$x$$-$$p$$
   - Coherent states of harmonic oscillator

**Follow-up Preparation:** Derive the uncertainty relation from the Schwarz inequality.

---

### Question 5: Explain Dirac Notation

**Examiner's Intent:** Test fluency with fundamental formalism.

**Answer Framework:**

1. **Kets and Bras:**
   - Ket $$|\psi\rangle$$: state vector in Hilbert space
   - Bra $$\langle\psi|$$: dual vector (linear functional)
   - Correspondence is antilinear: $$|c\psi\rangle \leftrightarrow c^*\langle\psi|$$

2. **Inner Product (Bracket):**
   - $$\langle\phi|\psi\rangle$$: complex number
   - Satisfies $$\langle\phi|\psi\rangle = \langle\psi|\phi\rangle^*$$
   - Represents probability amplitude

3. **Outer Product:**
   - $$|\phi\rangle\langle\psi|$$: operator
   - Projects onto $$|\phi\rangle$$ with amplitude $$\langle\psi|\cdot\rangle$$

4. **Why Useful:**
   - Representation-independent
   - Compact notation for complex operations
   - Makes completeness relations transparent

**Follow-up Preparation:** Write the resolution of identity in both discrete and continuous cases.

---

## Technical Questions

### Question 6: Prove That Unitary Operators Preserve Inner Products

**Examiner's Intent:** Test mathematical rigor.

**Answer Framework:**

1. **Definition of Unitary:** $$\hat{U}^\dagger\hat{U} = \hat{U}\hat{U}^\dagger = \hat{1}$$

2. **Proof:**
   $$\langle\hat{U}\phi|\hat{U}\psi\rangle = \langle\phi|\hat{U}^\dagger\hat{U}|\psi\rangle = \langle\phi|\hat{1}|\psi\rangle = \langle\phi|\psi\rangle$$

3. **Physical Significance:**
   - Preserves probability: $$|\langle\hat{U}\phi|\hat{U}\psi\rangle|^2 = |\langle\phi|\psi\rangle|^2$$
   - Preserves normalization
   - Represents symmetries and time evolution

4. **Examples:**
   - Time evolution: $$\hat{U}(t) = e^{-i\hat{H}t/\hbar}$$
   - Rotations: $$\hat{R}(\theta) = e^{-i\hat{L}_z\theta/\hbar}$$

**Follow-up Preparation:** Show that eigenvalues of unitary operators have unit modulus.

---

### Question 7: What is the Spectral Theorem?

**Examiner's Intent:** Test knowledge of foundational mathematics.

**Answer Framework:**

1. **Statement:** For a Hermitian operator $$\hat{A}$$, there exists an orthonormal basis of eigenvectors, and:
   $$\hat{A} = \sum_n a_n |a_n\rangle\langle a_n|$$

   For continuous spectra:
   $$\hat{A} = \int a |a\rangle\langle a| \, da$$

2. **Consequences:**
   - Any state can be expanded: $$|\psi\rangle = \sum_n c_n|a_n\rangle$$
   - Functions of operators: $$f(\hat{A}) = \sum_n f(a_n)|a_n\rangle\langle a_n|$$
   - Probability interpretation: $$P(a_n) = |c_n|^2$$

3. **Physical Application:**
   - Measurement yields eigenvalues
   - Eigenstates are stationary states of $$\hat{H}$$
   - Enables solving many QM problems

**Follow-up Preparation:** What if the operator is not Hermitian?

---

### Question 8: Explain Complete Sets of Commuting Observables (CSCO)

**Examiner's Intent:** Test understanding of quantum state specification.

**Answer Framework:**

1. **Definition:** A CSCO is a maximal set of mutually commuting Hermitian operators whose simultaneous eigenstates uniquely specify a state.

2. **Necessity:**
   - In degenerate spaces, one operator isn't enough
   - Need additional quantum numbers to label states uniquely

3. **Example — Hydrogen Atom:**
   - $$\{\hat{H}, \hat{L}^2, \hat{L}_z\}$$ for spinless electron
   - States labeled $$|n, l, m\rangle$$
   - With spin: $$\{\hat{H}, \hat{L}^2, \hat{L}_z, \hat{S}^2, \hat{S}_z\}$$

4. **Construction:**
   - Start with Hamiltonian
   - Add operators that commute with $$\hat{H}$$ and each other
   - Stop when eigenstates are non-degenerate

**Follow-up Preparation:** Why can't we use $$\hat{L}_x$$ along with $$\hat{L}_z$$ in a CSCO?

---

### Question 9: Derive the Position-Momentum Uncertainty Relation

**Examiner's Intent:** Test ability to derive fundamental results.

**Answer Framework:**

1. **Starting Point:** Robertson relation
   $$\Delta A \Delta B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$$

2. **Application:**
   - $$\hat{A} = \hat{x}$$, $$\hat{B} = \hat{p}$$
   - $$[\hat{x}, \hat{p}] = i\hbar$$

3. **Calculation:**
   $$\Delta x \Delta p \geq \frac{1}{2}|i\hbar| = \frac{\hbar}{2}$$

4. **Physical Interpretation:**
   - Sharp position → spread momentum (and vice versa)
   - Minimum uncertainty is $$\hbar/2$$, not zero
   - Achieved by Gaussian wave packets

**Follow-up Preparation:** Prove the Robertson relation from scratch.

---

### Question 10: What is the Difference Between Pure and Mixed States?

**Examiner's Intent:** Test understanding of density matrix formalism.

**Answer Framework:**

1. **Pure State:**
   - Described by single ket $$|\psi\rangle$$
   - Density matrix: $$\hat{\rho} = |\psi\rangle\langle\psi|$$
   - $$\hat{\rho}^2 = \hat{\rho}$$, so $$\text{Tr}(\hat{\rho}^2) = 1$$

2. **Mixed State:**
   - Classical probability distribution over pure states
   - $$\hat{\rho} = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$
   - $$\text{Tr}(\hat{\rho}^2) < 1$$

3. **Physical Difference:**
   - Pure: maximal quantum information
   - Mixed: classical uncertainty OR entanglement with environment
   - Reduced density matrix of entangled system is mixed

4. **Detection:**
   - Compute $$\text{Tr}(\hat{\rho}^2)$$
   - Pure if equals 1, mixed if less

**Follow-up Preparation:** How does decoherence convert pure to mixed states?

---

## "Explain to a Non-Expert" Questions

### Question 11: Explain Quantum Superposition to a First-Year Physics Student

**Answer Framework:**

1. **Classical Analogy:** Waves can add together (water waves, sound waves)

2. **Quantum Reality:**
   - Particles behave like waves
   - A particle can be in multiple states "at once"
   - Mathematically: $$|\psi\rangle = c_1|1\rangle + c_2|2\rangle$$

3. **What It Means:**
   - NOT the particle is secretly in one state
   - The particle genuinely has properties of both
   - Measurement forces a "choice"

4. **Everyday Consequence:**
   - Interference patterns in double-slit experiment
   - Quantum computing uses superposition for parallel processing

5. **What's Weird:**
   - We never see superposition directly
   - Measurement always gives definite result
   - "Collapse" is still philosophically mysterious

---

### Question 12: Why Can't We Know Position and Momentum Exactly at the Same Time?

**Answer Framework:**

1. **Key Insight:** It's not a measurement problem — it's fundamental

2. **Wave Explanation:**
   - Particles have wave properties
   - Well-defined position = localized wave = many wavelengths mixed
   - Well-defined momentum = single wavelength = spread-out wave
   - These are mathematically incompatible

3. **Analogy:**
   - A musical note: short duration means uncertain pitch
   - You can't have both precise time AND precise frequency

4. **Mathematical Reason:**
   - $$[\hat{x}, \hat{p}] = i\hbar \neq 0$$
   - Non-commuting operators can't have simultaneous eigenstates

5. **Importance:**
   - Not just academic — this is why atoms are stable!
   - Explains chemical bonding, semiconductors, lasers

---

### Question 13: What Does "Eigenvalue" Mean in Quantum Mechanics?

**Answer Framework:**

1. **Mathematical Definition:**
   - An eigenvalue $$a$$ satisfies $$\hat{A}|a\rangle = a|a\rangle$$
   - The operator just multiplies by a number

2. **Physical Meaning:**
   - Eigenvalue = possible measurement outcome
   - Eigenstate = state with definite value of that observable
   - You ONLY ever measure eigenvalues

3. **Example — Energy:**
   - $$\hat{H}|n\rangle = E_n|n\rangle$$
   - $$E_n$$ is the energy level
   - $$|n\rangle$$ is the stationary state

4. **Probability Connection:**
   - If $$|\psi\rangle = \sum_n c_n|a_n\rangle$$
   - Probability of measuring $$a_n$$ is $$|c_n|^2$$

---

### Question 14: Why is the Inner Product Important?

**Answer Framework:**

1. **Mathematical Role:**
   - Defines geometry: lengths, angles, orthogonality
   - $$\langle\psi|\psi\rangle = \|\psi\|^2$$ is the norm squared

2. **Physical Meaning:**
   - $$|\langle\phi|\psi\rangle|^2$$ is transition probability
   - Probability of finding state $$|\psi\rangle$$ in state $$|\phi\rangle$$

3. **Normalization:**
   - Total probability must be 1
   - $$\langle\psi|\psi\rangle = 1$$ ensures this

4. **Orthogonality:**
   - $$\langle\phi|\psi\rangle = 0$$ means states are distinguishable
   - Eigenstates of different eigenvalues are orthogonal

---

### Question 15: What is an Operator in Quantum Mechanics?

**Answer Framework:**

1. **Basic Idea:**
   - An operator acts on states to produce new states
   - $$\hat{A}|\psi\rangle = |\phi\rangle$$

2. **Physical Observables:**
   - Every measurable quantity has an operator
   - Position $$\to \hat{x}$$, Momentum $$\to \hat{p}$$, Energy $$\to \hat{H}$$

3. **Rules:**
   - Must be linear: $$\hat{A}(c_1|\psi_1\rangle + c_2|\psi_2\rangle) = c_1\hat{A}|\psi_1\rangle + c_2\hat{A}|\psi_2\rangle$$
   - Observables are Hermitian: $$\hat{A}^\dagger = \hat{A}$$

4. **Key Insight:**
   - Order matters: $$\hat{A}\hat{B} \neq \hat{B}\hat{A}$$ in general
   - The commutator measures this non-commutativity
   - Source of uncertainty principle

---

## Oral Exam Tips

### Presentation Strategy

1. **Start with the big picture** before diving into details
2. **Draw diagrams** whenever possible
3. **State assumptions** clearly
4. **Check limiting cases** to verify results
5. **Acknowledge what you don't know** — honesty is valued

### Common Follow-up Questions

- "Can you give an example?"
- "What happens in the limit of...?"
- "How does this connect to...?"
- "What if the operator were not Hermitian?"
- "Derive that on the board."

### Time Management

- Don't spend more than 1-2 minutes on any single point
- If stuck, say "Let me think about that" and move on
- Circle back if time permits

---

*Oral Practice Questions for Week 145 — Mathematical Framework*
*Practice these out loud, preferably at a whiteboard*
