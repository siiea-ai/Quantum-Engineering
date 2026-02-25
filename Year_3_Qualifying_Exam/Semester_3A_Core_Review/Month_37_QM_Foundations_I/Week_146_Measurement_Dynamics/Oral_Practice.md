# Week 146: Measurement and Dynamics — Oral Practice Questions

## Introduction

This document contains 15 oral exam questions on measurement theory and quantum dynamics. Practice explaining these concepts clearly and completely. Aim for 5-10 minutes per question.

---

## Conceptual Questions

### Question 1: State the Postulates of Quantum Mechanics

**Examiner's Intent:** Test comprehensive knowledge of foundations.

**Answer Framework:**

1. **State Space:** Quantum states are vectors in Hilbert space
2. **Observables:** Physical quantities are Hermitian operators
3. **Measurement (Born Rule):** Probability of outcome $$a$$ is $$|\langle a|\psi\rangle|^2$$
4. **Collapse:** After measurement, state becomes the eigenstate
5. **Evolution:** $$i\hbar\partial_t|\psi\rangle = \hat{H}|\psi\rangle$$

**Key Points:**
- These are AXIOMS — not derived from anything else
- All of quantum mechanics follows from these
- Controversial aspects: collapse, measurement problem

---

### Question 2: What Happens When We Measure a Quantum System?

**Examiner's Intent:** Test understanding of measurement process.

**Answer Framework:**

1. **Before Measurement:**
   - System in state $$|\psi\rangle = \sum_n c_n|a_n\rangle$$
   - Probabilities determined by $$|c_n|^2$$

2. **During Measurement:**
   - One eigenvalue $$a_k$$ is obtained
   - This is probabilistic, not deterministic
   - Probability is $$|c_k|^2$$

3. **After Measurement:**
   - State collapses to $$|a_k\rangle$$
   - Information about other amplitudes is lost
   - Immediate remeasurement gives same result

4. **Unresolved Questions:**
   - What triggers collapse?
   - Is it instantaneous?
   - Copenhagen vs. many-worlds interpretations

---

### Question 3: Explain Time Evolution in Quantum Mechanics

**Examiner's Intent:** Test understanding of dynamics.

**Answer Framework:**

1. **Schrödinger Equation:**
   $$i\hbar\frac{\partial}{\partial t}|\psi\rangle = \hat{H}|\psi\rangle$$

2. **Solution (time-independent H):**
   - Expand in energy eigenstates
   - Each acquires phase $$e^{-iE_nt/\hbar}$$
   - $$|\psi(t)\rangle = \sum_n c_n e^{-iE_nt/\hbar}|n\rangle$$

3. **Properties:**
   - Unitary: preserves normalization
   - Deterministic: future determined by present
   - Linear: superposition preserved

4. **Contrast with Measurement:**
   - Evolution is smooth and reversible
   - Collapse is discontinuous and irreversible
   - This is the "measurement problem"

---

### Question 4: What is the Propagator?

**Examiner's Intent:** Test knowledge of Green's function approach.

**Answer Framework:**

1. **Definition:**
   $$K(x,t;x',0) = \langle x|e^{-i\hat{H}t/\hbar}|x'\rangle$$

2. **Physical Meaning:**
   - Amplitude for particle at $$x'$$ to reach $$x$$ in time $$t$$
   - Contains ALL dynamical information

3. **Usage:**
   $$\psi(x,t) = \int K(x,t;x',0)\psi(x',0)dx'$$

4. **Examples:**
   - Free particle: Gaussian spreading
   - Harmonic oscillator: Periodic
   - Leads to path integrals

---

### Question 5: Compare Schrödinger and Heisenberg Pictures

**Examiner's Intent:** Test understanding of equivalent formulations.

**Answer Framework:**

1. **Schrödinger Picture:**
   - States evolve: $$|\psi_S(t)\rangle = \hat{U}(t)|\psi(0)\rangle$$
   - Operators fixed
   - Most common in textbooks

2. **Heisenberg Picture:**
   - States fixed: $$|\psi_H\rangle = |\psi(0)\rangle$$
   - Operators evolve: $$\hat{A}_H(t) = \hat{U}^\dagger\hat{A}\hat{U}$$
   - Resembles classical mechanics

3. **Equivalence:**
   - Same expectation values
   - Same physics
   - Choice is convenience

4. **When to Use Each:**
   - Schrödinger: wavefunction-based problems
   - Heisenberg: operator algebra, symmetries
   - Interaction picture: perturbation theory

---

## Technical Questions

### Question 6: Derive Ehrenfest's Theorem

**Examiner's Intent:** Test ability to derive results.

**Answer Framework:**

1. **Goal:** Show $$\frac{d\langle\hat{A}\rangle}{dt} = \frac{i}{\hbar}\langle[\hat{H},\hat{A}]\rangle + \langle\partial_t\hat{A}\rangle$$

2. **Derivation:**
   - Start with $$\langle\hat{A}\rangle = \langle\psi|\hat{A}|\psi\rangle$$
   - Take time derivative
   - Use Schrödinger equation for $$\partial_t|\psi\rangle$$
   - Collect terms

3. **Applications:**
   - $$\frac{d\langle x\rangle}{dt} = \frac{\langle p\rangle}{m}$$
   - $$\frac{d\langle p\rangle}{dt} = -\langle V'\rangle$$
   - Resembles Hamilton's equations!

4. **Significance:**
   - Shows classical limit
   - Conserved quantities: $$[\hat{H},\hat{A}] = 0$$

---

### Question 7: Explain the Born Rule

**Examiner's Intent:** Test foundational understanding.

**Answer Framework:**

1. **Statement:**
   $$P(a) = |\langle a|\psi\rangle|^2$$

2. **Why Squared Modulus?**
   - Probabilities must be real and positive
   - Complex amplitudes allow interference
   - Verified by countless experiments

3. **For Continuous Variables:**
   $$P(x \in [a,b]) = \int_a^b |\psi(x)|^2 dx$$

4. **Generalization:**
   - Density matrices: $$P(a) = \text{Tr}(\hat{\rho}|a\rangle\langle a|)$$
   - POVMs: generalized measurements

---

### Question 8: What is State Collapse?

**Examiner's Intent:** Test understanding of controversial concept.

**Answer Framework:**

1. **Description:**
   - Before: superposition $$|\psi\rangle = \sum c_n|a_n\rangle$$
   - After measurement of $$a_k$$: state becomes $$|a_k\rangle$$
   - Instantaneous, discontinuous

2. **Problems:**
   - Not described by Schrödinger equation
   - When exactly does it happen?
   - What constitutes a "measurement"?

3. **Interpretations:**
   - Copenhagen: collapse is real
   - Many-worlds: no collapse, branching
   - Decoherence: apparent collapse from entanglement

4. **Practical Approach:**
   - Use collapse rule, it works
   - Philosophical questions remain

---

### Question 9: How Does the Free Particle Propagator Work?

**Examiner's Intent:** Test ability to derive and interpret.

**Answer Framework:**

1. **Derivation:**
   - Insert completeness: $$K = \int \langle x|p\rangle\langle p|x'\rangle e^{-ip^2t/(2m\hbar)}dp$$
   - Gaussian integral
   - Result: $$K_0 \propto \exp[im(x-x')^2/(2\hbar t)]$$

2. **Features:**
   - Oscillatory, not decaying
   - Spreads as $$\sqrt{t}$$
   - Classical trajectory encoded in stationary phase

3. **Limiting Cases:**
   - $$t \to 0$$: $$K \to \delta(x-x')$$
   - $$\hbar \to 0$$: classical particle

---

### Question 10: Explain Conservation Laws in Quantum Mechanics

**Examiner's Intent:** Test understanding of symmetry-conservation connection.

**Answer Framework:**

1. **Statement:**
   If $$[\hat{H}, \hat{G}] = 0$$, then $$\langle\hat{G}\rangle$$ is constant.

2. **Proof via Ehrenfest:**
   $$\frac{d\langle\hat{G}\rangle}{dt} = \frac{i}{\hbar}\langle[\hat{H},\hat{G}]\rangle = 0$$

3. **Symmetry Connection:**
   - $$\hat{G}$$ generates a symmetry transformation
   - $$[\hat{H},\hat{G}] = 0$$ means $$\hat{H}$$ is invariant
   - Noether's theorem (quantum version)

4. **Examples:**
   | Symmetry | Generator | Conserved |
   |----------|-----------|-----------|
   | Translation | $$\hat{p}$$ | Momentum |
   | Rotation | $$\hat{L}$$ | Angular momentum |
   | Time translation | $$\hat{H}$$ | Energy |

---

## "Explain to Non-Expert" Questions

### Question 11: Why is Quantum Mechanics Probabilistic?

**Answer Framework:**

1. **Not Our Ignorance:**
   - Classical probability reflects incomplete knowledge
   - Quantum probability is fundamental
   - Nature is inherently random at quantum scale

2. **Evidence:**
   - Same preparation gives different results
   - Bell inequalities rule out hidden variables
   - No deeper deterministic theory

3. **Philosophical Impact:**
   - Einstein: "God does not play dice"
   - But experiments confirm probability is real
   - We must accept it

---

### Question 12: What Does the Schrödinger Equation Tell Us?

**Answer Framework:**

1. **Simple Answer:**
   - How quantum states change in time
   - Energy determines rate of change
   - Gives wave-like behavior

2. **Key Features:**
   - Linear: superposition preserved
   - First-order in time: initial state determines future
   - Complex: allows interference

3. **Comparison to Classical:**
   - Newton's law: $$F = ma$$
   - Schrödinger: $$i\hbar\partial_t\psi = H\psi$$
   - Both are deterministic evolution equations

---

### Question 13: What is the Difference Between Classical and Quantum Probability?

**Answer Framework:**

1. **Classical:**
   - Either/or: system IS in one state
   - Probability from ignorance
   - Probabilities add

2. **Quantum:**
   - Both/and: superposition is real
   - Probability is fundamental
   - Amplitudes add, then square

3. **Interference:**
   - Classical: $$P = P_1 + P_2$$
   - Quantum: $$P = |c_1 + c_2|^2 = |c_1|^2 + |c_2|^2 + 2\text{Re}(c_1^*c_2)$$
   - Cross term allows interference!

---

### Question 14: Why Can't We Predict Individual Measurement Outcomes?

**Answer Framework:**

1. **It's Not Technical Limitation:**
   - Not about better instruments
   - Not about hidden information
   - Fundamentally unpredictable

2. **What We CAN Predict:**
   - Probabilities exactly
   - Statistics of many measurements
   - Expectation values

3. **Evidence:**
   - Bell's theorem
   - No local hidden variable theory works
   - Verified experimentally

---

### Question 15: What is the Classical Limit of Quantum Mechanics?

**Answer Framework:**

1. **Ehrenfest's Theorem:**
   - $$\langle x \rangle$$ and $$\langle p \rangle$$ obey Newton's laws
   - ... if $$\langle f(x) \rangle \approx f(\langle x \rangle)$$
   - Valid for narrow wave packets, smooth potentials

2. **Conditions:**
   - Large quantum numbers
   - $$\hbar$$ small compared to action
   - Decoherence from environment

3. **Correspondence Principle:**
   - QM reduces to classical in appropriate limit
   - Quantum effects become negligible at large scales
   - But superposition, entanglement can persist

---

## Oral Exam Tips

### For Measurement Questions:
- Distinguish between evolution and collapse
- Know the Born rule cold
- Be prepared to discuss interpretations

### For Dynamics Questions:
- Master the time evolution operator
- Know how to switch between pictures
- Connect to classical mechanics via Ehrenfest

### For Propagator Questions:
- Derive the free particle case
- Explain physical meaning
- Know how to use it

---

*Oral Practice Questions for Week 146 — Measurement and Dynamics*
