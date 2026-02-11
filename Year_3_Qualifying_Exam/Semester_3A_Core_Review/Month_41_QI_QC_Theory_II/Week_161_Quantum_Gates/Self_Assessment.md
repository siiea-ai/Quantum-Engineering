# Week 161: Quantum Gates - Self-Assessment

## Mastery Checklists and Diagnostic Tools

---

## Part I: Core Knowledge Checklist

### Single-Qubit Gates

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| Pauli matrices (X, Y, Z) | [ ] | [ ] | [ ] | ___/3 |
| Bloch sphere representation | [ ] | [ ] | [ ] | ___/3 |
| Rotation operators $R_x, R_y, R_z$ | [ ] | [ ] | [ ] | ___/3 |
| General rotation $R_{\hat{n}}(\theta)$ | [ ] | [ ] | [ ] | ___/3 |
| Hadamard gate | [ ] | [ ] | [ ] | ___/3 |
| Phase gate (S) | [ ] | [ ] | [ ] | ___/3 |
| T gate (π/8 gate) | [ ] | [ ] | [ ] | ___/3 |
| ZYZ decomposition | [ ] | [ ] | [ ] | ___/3 |
| Euler angle parametrization | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/27**

### Two-Qubit Gates

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| CNOT gate | [ ] | [ ] | [ ] | ___/3 |
| Controlled-Z (CZ) | [ ] | [ ] | [ ] | ___/3 |
| SWAP gate | [ ] | [ ] | [ ] | ___/3 |
| iSWAP gate | [ ] | [ ] | [ ] | ___/3 |
| Controlled-U construction | [ ] | [ ] | [ ] | ___/3 |
| KAK decomposition | [ ] | [ ] | [ ] | ___/3 |
| Entangling power | [ ] | [ ] | [ ] | ___/3 |
| Local equivalence classes | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/24**

### Universality and Synthesis

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| Definition of universality | [ ] | [ ] | [ ] | ___/3 |
| Clifford group | [ ] | [ ] | [ ] | ___/3 |
| Clifford+T universality | [ ] | [ ] | [ ] | ___/3 |
| Gottesman-Knill theorem | [ ] | [ ] | [ ] | ___/3 |
| Solovay-Kitaev theorem | [ ] | [ ] | [ ] | ___/3 |
| T-count optimization | [ ] | [ ] | [ ] | ___/3 |
| Hardware gate compilation | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/21**

### **Total Knowledge Score: ___/72**

---

## Part II: Quick Recall Test

**Instructions:** Answer each question in 30 seconds or less without looking at notes.

### Set A: Definitions

1. Write the matrix for the Hadamard gate.

   Answer: _________________

2. What is the eigenvalue equation $H|v\rangle = \lambda|v\rangle$ for H?

   Answer: _________________

3. Write CNOT in Dirac notation.

   Answer: _________________

4. What is the T gate in terms of $R_z$?

   Answer: _________________

5. State the Solovay-Kitaev complexity.

   Answer: _________________

### Set B: True/False

| Statement | T/F |
|-----------|-----|
| 1. The Hadamard gate is its own inverse. | ___ |
| 2. $T^8 = I$ | ___ |
| 3. CNOT is in the Clifford group. | ___ |
| 4. T is in the Clifford group. | ___ |
| 5. Any two-qubit gate can be implemented with 3 CNOTs. | ___ |
| 6. Clifford circuits can be simulated in polynomial time. | ___ |
| 7. $\{H, S\}$ generates a dense subgroup of SU(2). | ___ |
| 8. SWAP requires 3 CNOT gates to implement. | ___ |
| 9. CZ is symmetric under qubit exchange. | ___ |
| 10. iSWAP is maximally entangling. | ___ |

**Answers:** 1-T, 2-T, 3-T, 4-F, 5-T, 6-T, 7-F, 8-T, 9-T, 10-T

### Set C: Fill in the Blank

1. The Pauli group on $n$ qubits has _____ elements.

2. The single-qubit Clifford group has _____ elements (up to global phase).

3. The minimum number of CNOT gates needed for arbitrary two-qubit unitary is _____.

4. The Solovay-Kitaev algorithm produces sequences of length $O(\log^c(1/\epsilon))$ where $c \approx$ _____.

5. Ross-Selinger algorithm achieves T-count approximately _____ $\log_2(1/\epsilon)$.

**Answers:** 1-$4^{n+1}$, 2-24, 3-3, 4-4, 5-3

---

## Part III: Computation Skills Assessment

### Test A: Matrix Calculations

**Problem 1:** Compute $HZH$.

Your answer: _________________

Correct answer: $X$

[ ] Correct [ ] Incorrect

**Problem 2:** Compute $SXS^\dagger$.

Your answer: _________________

Correct answer: $Y$

[ ] Correct [ ] Incorrect

**Problem 3:** What is $R_z(\pi/2)$?

Your answer: _________________

Correct answer: $\begin{pmatrix} e^{-i\pi/4} & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = e^{-i\pi/4}\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$

[ ] Correct [ ] Incorrect

**Problem 4:** Express $R_x(\theta)$ using $H$ and $R_z$.

Your answer: _________________

Correct answer: $R_x(\theta) = HR_z(\theta)H$

[ ] Correct [ ] Incorrect

**Computation Score: ___/4**

---

### Test B: Decompositions

**Problem 1:** Find ZYZ decomposition of $X$.

Work space:





Correct answer: $X = R_z(\pi)R_y(\pi)R_z(0)$ or equivalent

[ ] Correct [ ] Partial [ ] Incorrect

**Problem 2:** Express CZ using CNOT and single-qubit gates.

Work space:





Correct answer: $CZ = (I \otimes H) \cdot \text{CNOT} \cdot (I \otimes H)$

[ ] Correct [ ] Partial [ ] Incorrect

**Problem 3:** Decompose SWAP into CNOTs.

Work space:





Correct answer: $\text{SWAP} = \text{CNOT}_{12} \cdot \text{CNOT}_{21} \cdot \text{CNOT}_{12}$

[ ] Correct [ ] Partial [ ] Incorrect

**Decomposition Score: ___/3**

---

## Part IV: Proof Skills Assessment

### Can You Prove These Statements?

Rate your confidence: 1 (cannot prove) to 5 (can prove fluently)

| Statement | Confidence |
|-----------|------------|
| 1. Any SU(2) element can be written in ZYZ form | ___/5 |
| 2. {H, T} generates a dense subgroup of SU(2) | ___/5 |
| 3. {H, T, CNOT} is universal | ___/5 |
| 4. {H, S, CNOT} is NOT universal | ___/5 |
| 5. T is not in the Clifford group | ___/5 |
| 6. Any two-qubit gate needs at most 3 CNOTs | ___/5 |
| 7. Solovay-Kitaev gives polylog gate count | ___/5 |
| 8. Clifford circuits are classically simulable | ___/5 |

**Proof Confidence Score: ___/40**

---

## Part V: Oral Explanation Assessment

### Self-Evaluation Rubric

Rate your ability to explain each topic clearly to an examiner (1-5):

| Topic | Clarity | Completeness | Correctness | Score |
|-------|---------|--------------|-------------|-------|
| What is a quantum gate? | ___/5 | ___/5 | ___/5 | ___/15 |
| How do gates act on Bloch sphere? | ___/5 | ___/5 | ___/5 | ___/15 |
| What is universality? | ___/5 | ___/5 | ___/5 | ___/15 |
| Why is T gate special? | ___/5 | ___/5 | ___/5 | ___/15 |
| State Solovay-Kitaev | ___/5 | ___/5 | ___/5 | ___/15 |

**Oral Explanation Score: ___/75**

---

## Part VI: Gap Analysis

### Identify Your Weak Areas

Based on your scores above, identify your top 3 areas needing improvement:

1. _________________________________

2. _________________________________

3. _________________________________

### Recommended Study Actions

| If weak in... | Study... |
|---------------|----------|
| Single-qubit gates | Review Guide Section 2, Problems 1-10 |
| Two-qubit gates | Review Guide Section 3, Problems 11-18 |
| Universality proofs | Review Guide Section 4, Problems 19-25 |
| Solovay-Kitaev | Review Guide Section 5, Problem 21 |
| Oral explanations | Practice Oral_Practice.md with timer |
| Computation speed | Do 10 matrix calculations daily |

---

## Part VII: Timed Practice Exam

### Instructions
- Set timer for 30 minutes
- Complete all problems without notes
- Grade yourself honestly

### Problems

**1. (5 points)** Write the matrix representations of $H$, $S$, $T$, and $\text{CNOT}$.

**2. (5 points)** Prove that $HZH = X$.

**3. (5 points)** Find the ZYZ decomposition of the gate $U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$.

**4. (10 points)** Prove that $\{H, T\}$ generates a dense subgroup of $SU(2)$.

**5. (10 points)** Given the circuit $H_1 \cdot T_1 \cdot \text{CNOT}_{12} \cdot T_2 \cdot H_2$, compute the final state when starting from $|00\rangle$.

**6. (5 points)** State the Solovay-Kitaev theorem precisely, including the complexity.

### Grading
- Problems 1-3: Basic competency (15 points)
- Problems 4-5: Intermediate mastery (20 points)
- Problem 6: Advanced knowledge (5 points)

**Passing threshold: 25/40**

**Your Score: ___/40**

---

## Part VIII: Progress Tracking

### Daily Study Log

| Day | Topics Studied | Time (hrs) | Problems Done | Confidence (1-10) |
|-----|----------------|------------|---------------|-------------------|
| 1121 | | | | |
| 1122 | | | | |
| 1123 | | | | |
| 1124 | | | | |
| 1125 | | | | |
| 1126 | | | | |
| 1127 | | | | |

### Weekly Summary

**Starting confidence level:** ___/10

**Ending confidence level:** ___/10

**Total study hours:** ___

**Problems completed:** ___/30

**Areas still needing work:**

1. _________________________________

2. _________________________________

---

## Part IX: Final Readiness Checklist

Before moving to Week 162, verify you can:

### Essential Skills
- [ ] Write all standard gate matrices from memory
- [ ] Decompose any single-qubit unitary into ZYZ form
- [ ] Explain CNOT and construct Bell states
- [ ] State and explain the Clifford group
- [ ] Prove {H, T, CNOT} universality
- [ ] State Solovay-Kitaev theorem correctly
- [ ] Explain why T gates are expensive in fault tolerance

### Problem-Solving
- [ ] Complete foundational problems (1-10) with >80% accuracy
- [ ] Complete intermediate problems (11-20) with >70% accuracy
- [ ] Attempt all advanced problems (21-30)

### Oral Preparation
- [ ] Practice explaining each topic for 2-3 minutes
- [ ] Complete mock oral scenarios
- [ ] Can handle follow-up questions

**Ready for Week 162?** [ ] Yes [ ] Need more review

---

*Use this self-assessment throughout the week to track progress and identify areas needing additional study.*
