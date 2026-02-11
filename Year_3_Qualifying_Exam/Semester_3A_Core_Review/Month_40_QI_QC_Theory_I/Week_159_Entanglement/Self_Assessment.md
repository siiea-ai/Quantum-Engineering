# Week 159: Entanglement - Self-Assessment

## Pre-Study Diagnostic

### Quick Check (2 minutes each)

**Q1:** What is the entanglement entropy of $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$?

Your answer: _______________________

**Q2:** What is the classical bound in the CHSH inequality?

Your answer: _______________________

**Q3:** What is the maximum quantum value of CHSH (Tsirelson bound)?

Your answer: _______________________

**Q4:** If $$\rho^{T_B}$$ has a negative eigenvalue, what can you conclude about $$\rho$$?

Your answer: _______________________

**Q5:** What is the concurrence of a product state?

Your answer: _______________________

---

### Diagnostic Answers

<details>
<summary>Click to reveal answers</summary>

**A1:** 1 ebit (maximum for two qubits)

**A2:** $$|S| \leq 2$$

**A3:** $$|S| \leq 2\sqrt{2} \approx 2.83$$

**A4:** The state is entangled

**A5:** 0

</details>

**Score:** ___/5

---

## Concept Mastery Checklist

Rate your understanding: 1 (not familiar) to 5 (can teach it)

### Separability

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Definition of entanglement | | |
| Pure state separability | | |
| Mixed state separability | | |
| PPT criterion | | |

### Bell States

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Four Bell states | | |
| Properties (max entangled) | | |
| Bell basis measurement | | |
| Local unitary equivalence | | |

### Bell Inequalities

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| CHSH derivation | | |
| Quantum violation | | |
| Tsirelson bound | | |
| Physical significance | | |

### Entanglement Measures

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Entanglement entropy | | |
| Concurrence | | |
| Negativity | | |
| LOCC monotonicity | | |

---

## Skill Verification Problems

### Problem 1: Bell State Properties (5 min)

Write the four Bell states and compute $$\rho_A$$ for $$|\Psi^-\rangle$$.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

$$|\Phi^\pm\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle)$$
$$|\Psi^\pm\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)$$

For $$|\Psi^-\rangle$$:
$$\rho_A = \text{Tr}_B(|\Psi^-\rangle\langle\Psi^-|) = \frac{I}{2}$$

</details>

---

### Problem 2: CHSH Calculation (8 min)

For the singlet with correlation $$E(a,b) = -\cos\theta_{ab}$$, compute $$S$$ for angles 0°, 45°, 90°, 135°.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

$$S = E(0°,45°) - E(0°,135°) + E(90°,45°) + E(90°,135°)$$
$$= -\cos 45° + \cos 135° - \cos 45° - \cos 45°$$
$$= -\frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} = -2\sqrt{2}$$

</details>

---

### Problem 3: Concurrence (5 min)

Compute the concurrence for $$|\psi\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|11\rangle$$.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

Using $$C = 2|ad - bc|$$ with $$a = d = 1/\sqrt{2}$$, $$b = c = 0$$:
$$C = 2|1/2 - 0| = 1$$

</details>

---

## Common Mistakes to Avoid

- [ ] Confusing separable (product) with classically correlated
- [ ] Wrong sign in CHSH correlation for singlet vs. $$|\Phi^+\rangle$$
- [ ] Forgetting to take absolute value of negative eigenvalues for negativity
- [ ] PPT is necessary but not sufficient for separability (in general)
- [ ] Concurrence formula only works for two qubits

---

## Week 159 Completion Checklist

- [ ] Read Review Guide completely
- [ ] Completed Problem Set (at least 24/28 problems)
- [ ] Reviewed Problem Solutions
- [ ] Practiced Oral Practice questions
- [ ] Completed Self-Assessment
- [ ] Can derive CHSH bound
- [ ] Can compute concurrence and negativity
- [ ] Ready to proceed to Week 160

**Completion date:** _______________

---

*This self-assessment ensures thorough preparation for the qualifying examination.*
