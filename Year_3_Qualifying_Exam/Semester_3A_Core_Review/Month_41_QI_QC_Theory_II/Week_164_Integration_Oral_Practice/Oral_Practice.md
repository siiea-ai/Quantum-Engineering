# Week 164: Integration & Oral Practice - Comprehensive Oral Examination Guide

## Complete Preparation for Qualifying Exam Oral Component

---

## Part I: Full Mock Oral Examinations

### Mock Exam 1: Gates and Universality (30 minutes)

**Examiner Script:**

**(0:00-5:00) Opening**
"Let's start with quantum gates. What does it mean for a gate set to be universal?"

Expected: Define universality, distinguish exact vs. approximate, give examples.

**(5:00-12:00) Deep Dive**
"Prove that {H, T, CNOT} is universal."

Expected: Show H+T generates dense subgroup of SU(2), CNOT provides entanglement.

**(12:00-18:00) Technical**
"State the Solovay-Kitaev theorem and explain its significance."

Expected: Precise statement, $O(\log^c(1/\epsilon))$ complexity, practical implications.

**(18:00-24:00) Application**
"How would you implement an arbitrary single-qubit rotation on IBM hardware?"

Expected: Decompose into native gates, discuss compilation.

**(24:00-30:00) Connections**
"Why is the T gate special in fault-tolerant quantum computing?"

Expected: Non-Clifford, magic state distillation, Gottesman-Knill.

---

### Mock Exam 2: Quantum Algorithms (40 minutes)

**Examiner Script:**

**(0:00-8:00) Shor's Algorithm**
"Walk me through Shor's algorithm for factoring."

Expected: Reduction to order-finding, quantum phase estimation, continued fractions.

Follow-up: "What's the complexity?"
Expected: $O(n^3)$ gates, $O(n)$ qubits.

**(8:00-16:00) Grover's Algorithm**
"Derive Grover's algorithm, including the optimal iteration count."

Expected: Geometric interpretation, $k = \pi\sqrt{N}/4$, success probability.

Follow-up: "Prove it's optimal."
Expected: Outline BBBV theorem.

**(16:00-24:00) Comparison**
"Compare the speedups in Shor's vs. Grover's algorithms. Why are they different?"

Expected: Exponential vs. quadratic, structured vs. unstructured, BBBV implications.

**(24:00-32:00) Modern Algorithms**
"What are VQE and QAOA? When would you use each?"

Expected: Variational principle, ansatz design, optimization vs. eigenvalue problems.

**(32:00-40:00) Synthesis**
"If you had to solve a new problem, how would you decide which quantum algorithm to use?"

Expected: Systematic approach - structure, speedup type, resources, near-term feasibility.

---

### Mock Exam 3: Comprehensive Review (45 minutes)

**Examiner Script:**

**(0:00-10:00) Foundational**
"Start from the basics. What is a qubit and how do gates act on it?"

Work up through: Bloch sphere, Pauli matrices, rotations, universality.

**(10:00-20:00) Algorithms**
"Pick your favorite quantum algorithm and explain it completely."

Any algorithm chosen: full derivation, complexity, significance.

**(20:00-30:00) Proofs**
"Let's do some proofs. [Choose based on student's choice above]"

Options: BBBV, universality, phase estimation analysis.

**(30:00-40:00) Applications**
"What can quantum computers actually do that classical computers can't?"

Expected: Shor (crypto), simulation, optimization (uncertain), NISQ applications.

**(40:00-45:00) Research**
"What are the major open problems in quantum algorithms?"

Expected: Non-abelian HSP, quantum advantage for optimization, error correction overhead.

---

## Part II: Question Bank by Topic

### Gates and Circuits

1. "Write down the matrix for the Hadamard gate and explain its action on the Bloch sphere."

2. "What is the Clifford group and why is it important?"

3. "How many CNOT gates are needed to implement an arbitrary two-qubit gate?"

4. "Explain the KAK decomposition."

5. "What is the T-count of a circuit and why does it matter?"

### Foundational Algorithms

6. "Explain the Deutsch-Jozsa algorithm and its speedup."

7. "How does Simon's algorithm work and why is it historically important?"

8. "Derive the quantum Fourier transform circuit."

9. "Explain phase estimation and its applications."

10. "What is the hidden subgroup problem framework?"

### Shor's Algorithm

11. "State the factoring problem and explain why it's important."

12. "How does factoring reduce to order-finding?"

13. "What are the eigenstates of the modular exponentiation unitary?"

14. "How does continued fractions extract the period from phase estimation?"

15. "Give a complete complexity analysis of Shor's algorithm."

### Grover's Algorithm

16. "Explain Grover's algorithm geometrically."

17. "Why does Grover's algorithm oscillate?"

18. "What happens with multiple solutions?"

19. "State and prove the BBBV theorem."

20. "What is amplitude amplification and how does it generalize Grover?"

### Variational Algorithms

21. "Explain the variational principle and how VQE uses it."

22. "What is an ansatz and what makes a good one?"

23. "Describe the QAOA circuit for Max-Cut."

24. "What are barren plateaus and why are they problematic?"

25. "Compare VQE and QPE for finding ground state energies."

---

## Part III: Common Follow-Up Questions

### After Any Explanation
- "Can you write that more precisely?"
- "What's the intuition behind that?"
- "How would you prove that?"
- "What are the assumptions?"
- "What happens if [condition changes]?"

### After Complexity Claims
- "Is that optimal?"
- "What's the classical complexity?"
- "How do you prove the lower bound?"
- "What about space complexity?"

### After Algorithm Descriptions
- "Draw the circuit."
- "Trace through an example."
- "What could go wrong?"
- "How do you handle errors?"

### After Proofs
- "What's the key insight?"
- "Is there an alternative proof?"
- "What does this imply about [related topic]?"
- "Can you generalize this?"

---

## Part IV: Evaluation Rubric

### Scoring (for each question)

**5 - Excellent:**
- Complete, correct answer
- Clear presentation
- Insightful connections
- Handles follow-ups well

**4 - Good:**
- Mostly correct
- Minor gaps or imprecisions
- Some connections made
- Reasonable follow-up responses

**3 - Adequate:**
- Basic understanding shown
- Some errors or omissions
- Limited connections
- Struggles with follow-ups

**2 - Weak:**
- Partial understanding
- Significant errors
- No connections made
- Cannot handle follow-ups

**1 - Poor:**
- Major misunderstandings
- Fundamental errors
- Confused presentation
- No recovery from mistakes

### Overall Assessment

**Pass (Average 3.5+):**
- Demonstrates solid understanding of core material
- Can derive key results
- Makes connections between topics

**Marginal Pass (Average 3.0-3.4):**
- Knows basics but gaps in understanding
- Can follow derivations but not create them
- Limited synthesis

**Fail (Average < 3.0):**
- Fundamental gaps in knowledge
- Cannot perform key derivations
- Major conceptual errors

---

## Part V: Preparation Strategies

### Two Weeks Before

1. **Complete all problem sets** from Weeks 161-164
2. **Review all solutions** and understand gaps
3. **Practice derivations** on whiteboard/paper
4. **Time yourself** on key derivations

### One Week Before

1. **Daily oral practice** with study partner
2. **Record yourself** explaining topics
3. **Identify weak areas** and focus review
4. **Practice handling** follow-up questions

### Day Before

1. **Light review** of key formulas
2. **Practice explaining** 3-5 key topics
3. **Prepare mentally** for the format
4. **Get good sleep**

### Day Of

1. **Arrive early**, be calm
2. **Listen carefully** to questions
3. **Think before speaking**
4. **Draw diagrams** when helpful
5. **Admit uncertainty** gracefully

---

## Part VI: Common Mistakes to Avoid

### Technical Errors

1. **Grover iterations:** $\pi\sqrt{N}/4$, NOT $\sqrt{N}$
2. **Shor complexity:** $O(n^3)$ gates, NOT $O(n^2)$
3. **BBBV:** Oracle lower bound, NOT unconditional
4. **Clifford:** NOT universal (need T gate)
5. **QFT depth:** $O(n^2)$, NOT $O(n)$

### Presentation Errors

1. **Jumping to details** without overview
2. **Not drawing circuits** when helpful
3. **Being imprecise** with theorems
4. **Guessing** instead of admitting uncertainty
5. **Not connecting** to related topics

### Strategic Errors

1. **Rushing** through explanations
2. **Ignoring** follow-up hints
3. **Being defensive** about mistakes
4. **Giving up** too easily
5. **Not asking** for clarification

---

## Part VII: Key Formulas Quick Reference

### For Quick Review Before Exam

**Gates:**
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

**QFT:**
$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_k e^{2\pi ijk/N}|k\rangle$$

**Grover:**
$$k = \frac{\pi}{4}\sqrt{N}, \quad P = \sin^2((2k+1)\theta)$$

**Shor:**
$$\text{Qubits: } O(\log N), \quad \text{Gates: } O((\log N)^3)$$

**VQE:**
$$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle \geq E_0$$

**QAOA:**
$$|\gamma,\beta\rangle = \prod_l e^{-i\beta_l H_M}e^{-i\gamma_l H_C}|+\rangle^n$$

---

## Part VIII: Self-Assessment Checklist

Before taking the oral exam, verify:

### Core Knowledge
- [ ] Can write all standard gate matrices
- [ ] Can prove Clifford+T universality
- [ ] Can derive QFT circuit
- [ ] Can explain phase estimation

### Algorithms
- [ ] Can derive Shor's complete algorithm
- [ ] Can prove Grover's optimality
- [ ] Can explain VQE and QAOA

### Oral Skills
- [ ] Can explain topics at multiple levels
- [ ] Can handle follow-up questions
- [ ] Can recover from mistakes
- [ ] Can draw clear diagrams

### Mental Preparation
- [ ] Confident in core material
- [ ] Comfortable with format
- [ ] Ready to think on feet
- [ ] Prepared for uncertainty

---

*Good luck on your qualifying examination!*
