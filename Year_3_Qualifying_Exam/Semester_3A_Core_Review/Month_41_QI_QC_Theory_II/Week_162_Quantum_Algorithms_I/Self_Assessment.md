# Week 162: Quantum Algorithms I - Self-Assessment

## Mastery Checklists and Diagnostic Tools

---

## Part I: Core Knowledge Checklist

### Deutsch-Jozsa and Bernstein-Vazirani

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| Oracle model of computation | [ ] | [ ] | [ ] | ___/3 |
| Constant vs. balanced functions | [ ] | [ ] | [ ] | ___/3 |
| Phase kickback mechanism | [ ] | [ ] | [ ] | ___/3 |
| Deutsch-Jozsa circuit | [ ] | [ ] | [ ] | ___/3 |
| Deutsch-Jozsa correctness proof | [ ] | [ ] | [ ] | ___/3 |
| Bernstein-Vazirani algorithm | [ ] | [ ] | [ ] | ___/3 |
| Query complexity comparison | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/21**

### Simon's Algorithm

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| Simon's problem statement | [ ] | [ ] | [ ] | ___/3 |
| Algorithm structure | [ ] | [ ] | [ ] | ___/3 |
| Measurement analysis | [ ] | [ ] | [ ] | ___/3 |
| Classical post-processing | [ ] | [ ] | [ ] | ___/3 |
| Query complexity proof | [ ] | [ ] | [ ] | ___/3 |
| Classical lower bound | [ ] | [ ] | [ ] | ___/3 |
| Connection to Shor | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/21**

### Quantum Fourier Transform

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| QFT definition and matrix | [ ] | [ ] | [ ] | ___/3 |
| Product representation | [ ] | [ ] | [ ] | ___/3 |
| Circuit construction | [ ] | [ ] | [ ] | ___/3 |
| Gate complexity analysis | [ ] | [ ] | [ ] | ___/3 |
| Controlled-R_k gates | [ ] | [ ] | [ ] | ___/3 |
| Approximate QFT | [ ] | [ ] | [ ] | ___/3 |
| Comparison to classical FFT | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/21**

### Phase Estimation

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| Problem statement | [ ] | [ ] | [ ] | ___/3 |
| Algorithm structure | [ ] | [ ] | [ ] | ___/3 |
| Controlled-U^{2^j} operations | [ ] | [ ] | [ ] | ___/3 |
| Precision requirements | [ ] | [ ] | [ ] | ___/3 |
| Success probability | [ ] | [ ] | [ ] | ___/3 |
| Non-eigenstate inputs | [ ] | [ ] | [ ] | ___/3 |
| Applications | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/21**

### **Total Knowledge Score: ___/84**

---

## Part II: Quick Recall Test

**Instructions:** Answer each question in 30 seconds or less.

### Set A: Definitions

1. Write the final state in Deutsch-Jozsa after the oracle (before final Hadamard).

   Answer: _________________

2. In Simon's algorithm, what equation does each measured string $y$ satisfy?

   Answer: _________________

3. Write the QFT action on computational basis state $|j\rangle$.

   Answer: _________________

4. How many ancilla qubits are needed for $t$-bit phase estimation?

   Answer: _________________

5. What is phase kickback?

   Answer: _________________

### Set B: True/False

| Statement | T/F |
|-----------|-----|
| 1. Deutsch-Jozsa requires only 1 query to the oracle. | ___ |
| 2. Simon's algorithm has polynomial query complexity. | ___ |
| 3. The QFT requires $O(n)$ gates. | ___ |
| 4. Phase estimation works only for eigenstates. | ___ |
| 5. The approximate QFT can use $O(n\log n)$ gates. | ___ |
| 6. Classical FFT is faster than QFT in terms of input size. | ___ |
| 7. Bernstein-Vazirani finds a secret string with 1 query. | ___ |
| 8. Simon's algorithm finds collisions in the function. | ___ |
| 9. Phase estimation uses inverse QFT. | ___ |
| 10. Hidden Subgroup Problem generalizes Deutsch-Jozsa. | ___ |

**Answers:** 1-T, 2-T, 3-F (O(n^2)), 4-F, 5-T, 6-F, 7-T, 8-F, 9-T, 10-T

### Set C: Fill in the Blank

1. Deutsch-Jozsa distinguishes _______ functions from _______ functions.

2. Simon's algorithm has _______ quantum query complexity vs. _______ classical.

3. The QFT on $n$ qubits requires _______ controlled-R gates.

4. Phase estimation precision of $\epsilon$ requires _______ extra ancilla qubits.

5. The key to quantum speedup in these algorithms is quantum _______.

**Answers:** 1-constant, balanced; 2-O(n), $\Omega(2^{n/2})$; 3-$n(n-1)/2$; 4-$\log(1/\epsilon)$; 5-interference

---

## Part III: Algorithm Comprehension Test

### Test A: Trace Through Algorithms

**Problem 1:** For $n=2$ Deutsch-Jozsa with $f(00)=f(11)=0$, $f(01)=f(10)=1$:

Step 1 (Initial): $|00\rangle|1\rangle$

Step 2 (After H): _________________

Step 3 (After Oracle): _________________

Step 4 (After H on first register): _________________

Measurement outcome: _________________

**Problem 2:** For Simon's with $n=2$, $s=11$, trace one iteration:

Initial state: _________________

After oracle with $f(00)=f(11)=a$, $f(01)=f(10)=b$: _________________

Possible measurement outcomes: _________________

Equation obtained: _________________

---

### Test B: QFT Circuit Drawing

Draw the QFT circuit for $n=3$ qubits, labeling all gates:

```
|j_2⟩:


|j_1⟩:


|j_0⟩:

```

Count: ___ H gates, ___ controlled-R gates, ___ SWAPs

---

### Test C: Phase Estimation Setup

For $U = Z$ (eigenvalue $-1$ for $|1\rangle$):

Eigenstate: _________________

Phase $\theta$: _________________

For 3-bit precision:
- Number of ancillas: _________________
- Controlled-$U^{2^j}$ for $j$ = _________________
- Expected measurement: _________________

---

## Part IV: Proof Skills Assessment

### Can You Prove These Statements?

Rate your confidence: 1 (cannot prove) to 5 (can prove fluently)

| Statement | Confidence |
|-----------|------------|
| 1. Deutsch-Jozsa outputs $|0\rangle^n$ iff $f$ is constant | ___/5 |
| 2. Simon's algorithm produces equations $y \cdot s = 0$ | ___/5 |
| 3. QFT can be written as product of single-qubit states | ___/5 |
| 4. QFT requires $O(n^2)$ gates | ___/5 |
| 5. Approximate QFT with $O(n\log n)$ gates achieves bounded error | ___/5 |
| 6. Phase estimation success probability $\geq 4/\pi^2$ | ___/5 |
| 7. Simon's algorithm classical lower bound $\Omega(2^{n/2})$ | ___/5 |
| 8. Phase estimation samples eigenvalue spectrum | ___/5 |

**Proof Confidence Score: ___/40**

---

## Part V: Oral Explanation Assessment

### Self-Evaluation Rubric

Rate your ability to explain (1-5):

| Topic | Clarity | Depth | Correctness | Score |
|-------|---------|-------|-------------|-------|
| Oracle model and phase kickback | ___/5 | ___/5 | ___/5 | ___/15 |
| Deutsch-Jozsa speedup source | ___/5 | ___/5 | ___/5 | ___/15 |
| Simon's algorithm structure | ___/5 | ___/5 | ___/5 | ___/15 |
| QFT circuit derivation | ___/5 | ___/5 | ___/5 | ___/15 |
| Phase estimation applications | ___/5 | ___/5 | ___/5 | ___/15 |

**Oral Explanation Score: ___/75**

---

## Part VI: Timed Practice Exam

### Instructions
- Set timer for 45 minutes
- Complete all problems without notes
- Grade yourself honestly

### Problems

**1. (10 points)** Explain the Deutsch-Jozsa algorithm:
a) State the problem
b) Draw the circuit
c) Prove it correctly identifies constant functions

**2. (10 points)** Simon's algorithm analysis:
a) Write the state after oracle application and measurement of second register
b) Show that measurement of first register gives $y$ with $y \cdot s = 0$
c) Explain the classical post-processing

**3. (15 points)** QFT derivation:
a) Start from the definition, derive the product representation
b) Draw the circuit for $n=4$
c) Count all gates

**4. (10 points)** Phase estimation:
a) Draw the circuit for 4-bit precision
b) Explain what happens with non-eigenstate input
c) State the precision-ancilla tradeoff

**5. (5 points)** Hidden Subgroup Problem:
a) State the general problem
b) Explain how Deutsch-Jozsa is a special case

### Grading
- Problems 1-2: Algorithm mastery (20 points)
- Problem 3: QFT expertise (15 points)
- Problem 4: Phase estimation (10 points)
- Problem 5: Connections (5 points)

**Passing threshold: 35/50**

**Your Score: ___/50**

---

## Part VII: Gap Analysis

### Identify Your Weak Areas

Based on your scores above, identify your top 3 areas needing improvement:

1. _________________________________

2. _________________________________

3. _________________________________

### Study Plan

| If weak in... | Study... |
|---------------|----------|
| Deutsch-Jozsa | Review Guide Section 2, Problems 1-6 |
| Simon's algorithm | Review Guide Section 4, Problems 9-14 |
| QFT | Review Guide Section 5, Problems 15-20 |
| Phase Estimation | Review Guide Section 6, Problems 21-28 |
| Proofs | Re-derive key results from scratch |
| Oral skills | Practice with Oral_Practice.md |

---

## Part VIII: Progress Tracking

### Daily Study Log

| Day | Topics Studied | Time (hrs) | Problems Done | Confidence (1-10) |
|-----|----------------|------------|---------------|-------------------|
| 1128 | | | | |
| 1129 | | | | |
| 1130 | | | | |
| 1131 | | | | |
| 1132 | | | | |
| 1133 | | | | |
| 1134 | | | | |

### Weekly Summary

**Starting confidence level:** ___/10

**Ending confidence level:** ___/10

**Total study hours:** ___

**Problems completed:** ___/28

---

## Part IX: Final Readiness Checklist

Before moving to Week 163, verify you can:

### Essential Skills
- [ ] Trace through Deutsch-Jozsa for any given function
- [ ] Explain Simon's algorithm query complexity
- [ ] Construct QFT circuit for any $n$
- [ ] Calculate phase estimation requirements
- [ ] Explain phase kickback clearly

### Derivations
- [ ] Prove Deutsch-Jozsa correctness
- [ ] Derive QFT from definition
- [ ] Analyze phase estimation success probability

### Connections
- [ ] Relate algorithms to Hidden Subgroup Problem
- [ ] Explain how Simon's led to Shor's
- [ ] Discuss approximate vs exact QFT

**Ready for Week 163 (Shor's & Grover's)?** [ ] Yes [ ] Need more review

---

## Part X: Key Formulas Quick Reference

### Deutsch-Jozsa
$$\text{Amplitude of } |0\rangle^n = \frac{1}{2^n}\sum_x (-1)^{f(x)}$$

### Simon's
$$y \cdot s = 0 \text{ for all measured } y$$

### QFT
$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}e^{2\pi ijk/N}|k\rangle$$

### Phase Estimation
$$\text{Precision: } t \text{ bits with } t \text{ ancillas}$$
$$P(\text{correct}) \geq \frac{4}{\pi^2} \approx 0.405$$

---

*Use this self-assessment throughout the week to track progress and identify areas needing additional study.*
