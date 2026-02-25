# Week 158: Composite Systems - Self-Assessment

## Pre-Study Diagnostic

### Quick Check (2 minutes each)

**Q1:** What is the dimension of $$\mathbb{C}^3 \otimes \mathbb{C}^4$$?

Your answer: _______________________

**Q2:** Compute $$\text{Tr}_B(|0\rangle\langle 1| \otimes |1\rangle\langle 1|)$$.

Your answer: _______________________

**Q3:** What is the Schmidt rank of a product state?

Your answer: _______________________

**Q4:** For the Bell state $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$, what are the Schmidt coefficients?

Your answer: _______________________

**Q5:** If $$\rho_A$$ has rank 3, what is the minimum dimension needed for an ancilla to purify it?

Your answer: _______________________

---

### Diagnostic Answers

<details>
<summary>Click to reveal answers</summary>

**A1:** $$3 \times 4 = 12$$

**A2:** $$|0\rangle\langle 1| \cdot \text{Tr}(|1\rangle\langle 1|) = |0\rangle\langle 1| \cdot 1 = |0\rangle\langle 1|$$

**A3:** 1

**A4:** $$\lambda_1 = \lambda_2 = \frac{1}{\sqrt{2}}$$

**A5:** 3 (dimension must be at least rank of $$\rho_A$$)

</details>

**Score:** ___/5

---

## Concept Mastery Checklist

Rate your understanding: 1 (not familiar) to 5 (can teach it)

### Tensor Products

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Tensor product of vectors | | |
| Kronecker product of matrices | | |
| Mixed-product property | | |
| Dimension of composite space | | |

### Partial Trace

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Definition and computation | | |
| Physical interpretation | | |
| $$\text{Tr}_B(A \otimes B) = A \cdot \text{Tr}(B)$$ | | |
| Two-qubit partial trace | | |

### Schmidt Decomposition

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Theorem statement | | |
| Connection to SVD | | |
| Schmidt rank and entanglement | | |
| Reduced states from Schmidt form | | |

### Purification

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Definition and construction | | |
| Non-uniqueness | | |
| Minimum ancilla size | | |
| Applications | | |

---

## Skill Verification Problems

### Problem 1: Tensor Product (5 min)

Compute $$(H \otimes I)|00\rangle$$ where $$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

$$H|0\rangle = |+\rangle$$

$$(H \otimes I)|00\rangle = |+\rangle \otimes |0\rangle = |+0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

</details>

---

### Problem 2: Partial Trace (8 min)

For $$|\psi\rangle = \frac{1}{\sqrt{3}}|00\rangle + \frac{1}{\sqrt{3}}|01\rangle + \frac{1}{\sqrt{3}}|10\rangle$$:

(a) Compute $$\rho_{AB} = |\psi\rangle\langle\psi|$$
(b) Compute $$\rho_A = \text{Tr}_B(\rho_{AB})$$
(c) Is the state entangled?

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

(a) $$\rho_{AB}$$ is a $$4 \times 4$$ matrix with entries from $$|\psi\rangle\langle\psi|$$

(b) $$\rho_A = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$

(c) $$\text{Tr}(\rho_A^2) = \frac{1}{9}(4 + 2 + 1) = \frac{7}{9} < 1$$

Yes, the state is **entangled**.

</details>

---

### Problem 3: Schmidt Decomposition (10 min)

Find the Schmidt decomposition of $$|\psi\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle - |11\rangle)$$.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

Coefficient matrix: $$C = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

This is $$\frac{1}{\sqrt{2}}H$$ where $$H$$ is Hadamard.

$$CC^\dagger = \frac{1}{2}I$$, so eigenvalues are both $$\frac{1}{2}$$.

Schmidt coefficients: $$\lambda_1 = \lambda_2 = \frac{1}{\sqrt{2}}$$

Schmidt decomposition: $$|\psi\rangle = \frac{1}{\sqrt{2}}|a_1\rangle|b_1\rangle + \frac{1}{\sqrt{2}}|a_2\rangle|b_2\rangle$$

where the Schmidt bases come from the SVD.

</details>

---

### Problem 4: Purification (5 min)

Construct a purification of $$\rho = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{I}{2}$$.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

$$|\Psi\rangle = \frac{1}{\sqrt{2}}|0\rangle|0\rangle + \frac{1}{\sqrt{2}}|1\rangle|1\rangle = |\Phi^+\rangle$$

Verification: $$\text{Tr}_B(|\Phi^+\rangle\langle\Phi^+|) = \frac{I}{2}$$ ✓

</details>

---

## Common Mistakes to Avoid

- [ ] **Confusing tensor product order**: $$|a\rangle \otimes |b\rangle \neq |b\rangle \otimes |a\rangle$$

- [ ] **Wrong partial trace formula**: Remember to trace over the correct subsystem

- [ ] **Forgetting normalization**: Schmidt coefficients must satisfy $$\sum_i \lambda_i^2 = 1$$

- [ ] **Confusing Schmidt coefficients with eigenvalues**: $$\lambda_i$$ are coefficients; $$\lambda_i^2$$ are eigenvalues of reduced states

- [ ] **Not checking entanglement**: Product state iff Schmidt rank = 1

---

## Post-Study Assessment

### Conceptual Understanding

1. Explain in your own words why the partial trace gives the correct reduced state.

_________________________________________________________

2. How does the Schmidt decomposition relate to entanglement?

_________________________________________________________

3. Why is purification non-unique?

_________________________________________________________

### Ready for Oral Exam?

Can you explain these without notes? (Yes/Mostly/No)

- [ ] Tensor product construction
- [ ] Partial trace computation
- [ ] Schmidt decomposition theorem
- [ ] Connection to SVD
- [ ] Purification construction
- [ ] Entanglement detection via Schmidt rank

---

## Week 158 Completion Checklist

- [ ] Read Review Guide completely
- [ ] Completed Problem Set (at least 25/30 problems)
- [ ] Reviewed Problem Solutions for missed problems
- [ ] Practiced Oral Practice questions
- [ ] Completed Self-Assessment
- [ ] Identified and addressed weak areas
- [ ] Ready to proceed to Week 159

**Estimated completion date:** _______________

**Actual completion date:** _______________

---

*This self-assessment helps ensure thorough preparation for the qualifying examination.*
