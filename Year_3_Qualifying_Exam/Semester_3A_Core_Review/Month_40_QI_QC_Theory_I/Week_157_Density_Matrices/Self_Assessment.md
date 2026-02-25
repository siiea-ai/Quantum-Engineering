# Week 157: Density Matrices - Self-Assessment

## Pre-Study Diagnostic

Before beginning this week's review, complete this diagnostic to identify areas needing focus.

### Quick Check (2 minutes each)

**Q1:** Write the density matrix for $$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$.

Your answer: _______________________

**Q2:** What is the Bloch vector for the maximally mixed state $$\rho = I/2$$?

Your answer: _______________________

**Q3:** For a pure state, what is $$\text{Tr}(\rho^2)$$?

Your answer: _______________________

**Q4:** Write the formula for von Neumann entropy.

Your answer: _______________________

**Q5:** What is the partial trace $$\text{Tr}_B(|0\rangle\langle 0| \otimes |1\rangle\langle 1|)$$?

Your answer: _______________________

---

### Diagnostic Answers

<details>
<summary>Click to reveal answers</summary>

**A1:** $$\rho = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**A2:** $$\vec{r} = (0, 0, 0)$$ - the origin of the Bloch ball

**A3:** $$\text{Tr}(\rho^2) = 1$$

**A4:** $$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

**A5:** $$|0\rangle\langle 0| \cdot \text{Tr}(|1\rangle\langle 1|) = |0\rangle\langle 0|$$

</details>

**Score:** ___/5

- 5/5: Proceed with confidence
- 3-4/5: Review weak areas before problem set
- 0-2/5: Study Review Guide thoroughly first

---

## Concept Mastery Checklist

Rate your understanding of each concept: 1 (not familiar) to 5 (can teach it)

### Pure States and Density Operators

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Density matrix definition $$\rho = \|\psi\rangle\langle\psi\|$$ | | |
| Three properties: Hermitian, trace 1, positive | | |
| Computing expectation values via trace | | |
| Pure state idempotency $$\rho^2 = \rho$$ | | |

### Mixed States

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Ensemble interpretation | | |
| Non-uniqueness of decomposition | | |
| Purity $$\gamma = \text{Tr}(\rho^2)$$ | | |
| Von Neumann entropy | | |
| Difference between superposition and mixture | | |

### Bloch Sphere

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Bloch representation formula | | |
| Computing Bloch vector from $$\rho$$ | | |
| Constructing $$\rho$$ from Bloch vector | | |
| Pure vs mixed on Bloch sphere | | |
| Effect of unitaries on Bloch vector | | |

### Trace Operations

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Trace properties (cyclic, linear) | | |
| Trace distance | | |
| Fidelity | | |
| Partial trace definition | | |
| Computing reduced density matrices | | |

---

## Skill Verification Problems

Complete these problems to verify your understanding. Time yourself.

### Problem 1: Basic Density Matrix (5 min)

Given the ensemble: $$|0\rangle$$ with probability 0.6 and $$|+\rangle$$ with probability 0.4.

(a) Write the density matrix.
(b) Calculate the purity.
(c) Is this pure or mixed?

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

(a) $$\rho = 0.6|0\rangle\langle 0| + 0.4|+\rangle\langle+| = 0.6\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + 0.4 \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 0.8 & 0.2 \\ 0.2 & 0.2 \end{pmatrix}$$

(b) $$\gamma = \text{Tr}(\rho^2) = 0.8^2 + 2(0.2)^2 + 0.2^2 = 0.64 + 0.08 + 0.04 = 0.76$$

Actually, computing more carefully:
$$\rho^2 = \begin{pmatrix} 0.68 & 0.2 \\ 0.2 & 0.08 \end{pmatrix}$$
$$\gamma = 0.68 + 0.08 = 0.76$$

(c) Mixed (since $$\gamma < 1$$)

</details>

---

### Problem 2: Bloch Sphere (5 min)

A state has Bloch vector $$\vec{r} = (0.5, 0.5, 0)$$.

(a) Write the density matrix.
(b) Find the eigenvalues.
(c) Calculate the von Neumann entropy.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

(a) $$\rho = \frac{1}{2}(I + 0.5\sigma_x + 0.5\sigma_y) = \frac{1}{2}\begin{pmatrix} 1 & 0.5 - 0.5i \\ 0.5 + 0.5i & 1 \end{pmatrix}$$

(b) $$|\vec{r}| = \sqrt{0.25 + 0.25} = 1/\sqrt{2}$$

Eigenvalues: $$\lambda_\pm = \frac{1}{2}(1 \pm 1/\sqrt{2}) \approx 0.854, 0.146$$

(c) $$S = -0.854\log_2(0.854) - 0.146\log_2(0.146) \approx 0.600$$ bits

</details>

---

### Problem 3: Partial Trace (8 min)

For the state $$|\psi\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$:

(a) Write the density matrix $$\rho_{AB}$$.
(b) Compute $$\rho_A = \text{Tr}_B(\rho_{AB})$$.
(c) Is $$|\psi\rangle$$ entangled or separable?

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

(a) First note $$|\psi\rangle = |+\rangle \otimes |+\rangle$$ (product state!)

$$\rho_{AB} = |+\rangle\langle+| \otimes |+\rangle\langle+| = \frac{1}{4}\begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \end{pmatrix}$$

(b) $$\rho_A = |+\rangle\langle+| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

(c) **Separable** - it's a product state. Alternatively, $$\rho_A$$ is pure ($$\text{Tr}(\rho_A^2) = 1$$), confirming separability.

</details>

---

### Problem 4: Distinguishability (5 min)

Compute the trace distance and fidelity between $$|0\rangle$$ and $$|+\rangle$$.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

**Bloch vectors:**
- $$|0\rangle$$: $$\vec{r}_1 = (0, 0, 1)$$
- $$|+\rangle$$: $$\vec{r}_2 = (1, 0, 0)$$

**Trace distance:**
$$D = \frac{1}{2}|\vec{r}_1 - \vec{r}_2| = \frac{1}{2}\sqrt{1 + 1} = \frac{1}{\sqrt{2}} \approx 0.707$$

**Fidelity:**
$$F = |\langle 0|+\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

**Verify Fuchs-van de Graaf:**
$$1 - \sqrt{0.5} = 0.293 \leq 0.707 \leq \sqrt{0.5} = 0.707$$ ✓

</details>

---

## Common Mistakes to Avoid

Check each box once you understand the mistake and how to avoid it:

- [ ] **Confusing pure state projector with superposition coefficients**
  - Wrong: $$\rho$$ for $$|\psi\rangle = c_0|0\rangle + c_1|1\rangle$$ is $$\begin{pmatrix} c_0 & 0 \\ 0 & c_1 \end{pmatrix}$$
  - Right: $$\rho = \begin{pmatrix} |c_0|^2 & c_0c_1^* \\ c_1c_0^* & |c_1|^2 \end{pmatrix}$$

- [ ] **Forgetting to normalize mixed states**
  - Probabilities in an ensemble must sum to 1

- [ ] **Incorrect partial trace computation**
  - Remember: $$\text{Tr}_B$$ traces over the second subsystem in $$|ab\rangle$$ basis

- [ ] **Sign error in Bloch vector**
  - $$r_z = \rho_{00} - \rho_{11}$$, not $$\rho_{11} - \rho_{00}$$

- [ ] **Confusing eigenvalues with ensemble probabilities**
  - Spectral decomposition is unique; ensemble decomposition is not

- [ ] **Forgetting complex conjugate in off-diagonal elements**
  - $$\rho_{01} = c_0 c_1^*$$, not $$c_0 c_1$$

---

## Post-Study Assessment

After completing all Week 157 materials, answer these questions:

### Conceptual Understanding

1. In your own words, explain why density matrices are necessary in quantum mechanics (not just state vectors).

_________________________________________________________

2. Describe the relationship between entanglement and the purity of reduced density matrices.

_________________________________________________________

3. What is the physical meaning of trace distance?

_________________________________________________________

### Computational Fluency

Rate your speed on these calculations (Fast/Medium/Slow):

| Calculation | Speed | Notes |
|------------|-------|-------|
| Construct $$\rho$$ from $$\|\psi\rangle$$ | | |
| Compute Bloch vector | | |
| Calculate purity | | |
| Partial trace of 2-qubit state | | |
| Trace distance between qubits | | |

### Ready for Oral Exam?

Can you explain these to an examiner without notes? (Yes/Mostly/No)

- [ ] Definition and properties of density matrices
- [ ] Pure vs. mixed states
- [ ] Bloch sphere representation
- [ ] Partial trace and reduced density matrices
- [ ] Trace distance and fidelity
- [ ] Von Neumann entropy

---

## Study Plan for Weak Areas

Based on your self-assessment, identify areas needing more work:

**Topics to review:**
1. _______________________
2. _______________________
3. _______________________

**Specific problems to redo:**
1. Problem Set #_____
2. Problem Set #_____
3. Problem Set #_____

**Resources to consult:**
- Nielsen & Chuang Section: _______
- Preskill Notes Chapter: _______
- Other: _______________________

---

## Week 157 Completion Checklist

- [ ] Read Review Guide completely
- [ ] Completed Problem Set (at least 25/30 problems)
- [ ] Reviewed Problem Solutions for missed problems
- [ ] Practiced Oral Practice questions (recorded myself)
- [ ] Completed Self-Assessment
- [ ] Identified and addressed weak areas
- [ ] Can explain all concepts without notes
- [ ] Ready to proceed to Week 158

**Estimated completion date:** _______________

**Actual completion date:** _______________

**Notes for future review:**

_________________________________________________________

_________________________________________________________

---

*This self-assessment helps ensure thorough preparation for the qualifying examination. Be honest in your ratings - it's better to identify gaps now than during the exam.*
