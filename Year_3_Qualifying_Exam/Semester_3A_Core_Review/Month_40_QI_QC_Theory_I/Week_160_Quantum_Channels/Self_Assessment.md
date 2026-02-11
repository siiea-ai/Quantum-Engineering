# Week 160: Quantum Channels - Self-Assessment

## Pre-Study Diagnostic

### Quick Check (2 minutes each)

**Q1:** What does CPTP stand for?

Your answer: _______________________

**Q2:** Write the completeness relation for Kraus operators.

Your answer: _______________________

**Q3:** Write the Kraus operators for the bit-flip channel.

Your answer: _______________________

**Q4:** What is the fixed point of the amplitude damping channel?

Your answer: _______________________

**Q5:** What is the Choi matrix of the identity channel?

Your answer: _______________________

---

### Diagnostic Answers

<details>
<summary>Click to reveal answers</summary>

**A1:** Completely Positive Trace-Preserving

**A2:** $$\sum_k K_k^\dagger K_k = I$$

**A3:** $$K_0 = \sqrt{1-p}I$$, $$K_1 = \sqrt{p}X$$

**A4:** $$|0\rangle\langle 0|$$ (ground state)

**A5:** $$|\Phi^+\rangle\langle\Phi^+|$$ (maximally entangled state)

</details>

**Score:** ___/5

---

## Concept Mastery Checklist

Rate your understanding: 1 (not familiar) to 5 (can teach it)

### CPTP Maps

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Complete positivity | | |
| Trace preservation | | |
| Why both are needed | | |

### Kraus Representation

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Operator-sum form | | |
| Completeness relation | | |
| Physical derivation | | |
| Non-uniqueness | | |

### Standard Channels

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Depolarizing | | |
| Amplitude damping | | |
| Phase damping | | |
| Bloch sphere effects | | |

### Choi-Jamiolkowski

| Concept | Self-Rating (1-5) | Notes |
|---------|-------------------|-------|
| Choi matrix construction | | |
| CP/TP from Choi | | |
| Channel recovery | | |

---

## Skill Verification Problems

### Problem 1: Kraus Operators (5 min)

Verify the completeness relation for the amplitude damping channel.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

$$K_0^\dagger K_0 + K_1^\dagger K_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1-\gamma \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & \gamma \end{pmatrix} = I$$ ✓

</details>

---

### Problem 2: Channel Application (5 min)

Apply the amplitude damping channel ($$\gamma = 0.5$$) to $$|1\rangle\langle 1|$$.

**Your work:**

**Time taken:** _____ minutes

<details>
<summary>Check your answer</summary>

$$\mathcal{E}(|1\rangle\langle 1|) = K_0|1\rangle\langle 1|K_0^\dagger + K_1|1\rangle\langle 1|K_1^\dagger$$
$$= (1-0.5)|1\rangle\langle 1| + 0.5|0\rangle\langle 0|$$
$$= 0.5|0\rangle\langle 0| + 0.5|1\rangle\langle 1| = \frac{I}{2}$$

</details>

---

### Problem 3: Choi Matrix (8 min)

Compute the Choi matrix for the bit-flip channel with $$p = 0.5$$.

**Your work:**

**Time taken:** _____ minutes

---

## Common Mistakes to Avoid

- [ ] Confusing $$\sum_k K_k^\dagger K_k = I$$ (TP) with $$\sum_k K_k K_k^\dagger = I$$ (unital)
- [ ] Forgetting the $$\sqrt{}$$ in Kraus operators for probability $$p$$
- [ ] Mixing up amplitude damping (energy decay) with phase damping (coherence decay)
- [ ] Wrong normalization in Choi matrix

---

## Week 160 Completion Checklist

- [ ] Read Review Guide completely
- [ ] Completed Problem Set (at least 22/26 problems)
- [ ] Reviewed Problem Solutions
- [ ] Practiced Oral Practice questions
- [ ] Completed Self-Assessment
- [ ] Can write Kraus operators for standard channels
- [ ] Understand Choi-Jamiolkowski isomorphism
- [ ] Month 40 complete!

**Completion date:** _______________

---

## Month 40 Summary

You have completed Month 40: QI/QC Theory Review I, covering:

1. **Week 157**: Density Matrices
2. **Week 158**: Composite Systems
3. **Week 159**: Entanglement
4. **Week 160**: Quantum Channels

These foundational topics are essential for the qualifying examination. Continue to Month 41 for error correction and advanced topics.

---

*Congratulations on completing Month 40!*
