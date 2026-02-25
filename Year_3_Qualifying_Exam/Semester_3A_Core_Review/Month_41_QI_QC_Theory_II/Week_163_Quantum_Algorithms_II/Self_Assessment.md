# Week 163: Quantum Algorithms II - Self-Assessment

## Mastery Checklists for Shor's and Grover's Algorithms

---

## Part I: Core Knowledge Checklist

### Shor's Algorithm

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| Factoring to order-finding reduction | [ ] | [ ] | [ ] | ___/3 |
| Order-finding problem statement | [ ] | [ ] | [ ] | ___/3 |
| Modular exponentiation unitary | [ ] | [ ] | [ ] | ___/3 |
| Eigenstates of $U_a$ | [ ] | [ ] | [ ] | ___/3 |
| Phase estimation application | [ ] | [ ] | [ ] | ___/3 |
| Continued fractions algorithm | [ ] | [ ] | [ ] | ___/3 |
| Success probability analysis | [ ] | [ ] | [ ] | ___/3 |
| Complete complexity analysis | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/24**

### Grover's Algorithm

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| Unstructured search problem | [ ] | [ ] | [ ] | ___/3 |
| Oracle operator $O_f$ | [ ] | [ ] | [ ] | ___/3 |
| Diffusion operator $D$ | [ ] | [ ] | [ ] | ___/3 |
| Grover iterator $G = DO_f$ | [ ] | [ ] | [ ] | ___/3 |
| Geometric interpretation | [ ] | [ ] | [ ] | ___/3 |
| Optimal iteration count | [ ] | [ ] | [ ] | ___/3 |
| Multiple solutions analysis | [ ] | [ ] | [ ] | ___/3 |
| Success probability | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/24**

### Optimality and Extensions

| Topic | Can Define | Can Derive | Can Apply | Mastery |
|-------|------------|------------|-----------|---------|
| Query complexity model | [ ] | [ ] | [ ] | ___/3 |
| BBBV theorem statement | [ ] | [ ] | [ ] | ___/3 |
| BBBV proof outline | [ ] | [ ] | [ ] | ___/3 |
| Amplitude amplification | [ ] | [ ] | [ ] | ___/3 |
| Quantum counting | [ ] | [ ] | [ ] | ___/3 |
| Shor vs. Grover comparison | [ ] | [ ] | [ ] | ___/3 |

**Section Score: ___/18**

### **Total Knowledge Score: ___/66**

---

## Part II: Quick Recall Test

### Set A: Fill in the Formula

1. Shor's algorithm complexity: $O(________)$ gates

2. Grover's optimal iterations: $k = ________$

3. BBBV lower bound: $\Omega(________)$ queries

4. Order-finding eigenstates: $|u_s\rangle = ________$

5. Grover state after $k$ iterations: $G^k|s\rangle = ________$

**Answers:** 1-$(\log N)^3$, 2-$\frac{\pi}{4}\sqrt{N}$, 3-$\sqrt{N}$, 4-$\frac{1}{\sqrt{r}}\sum_j e^{-2\pi isj/r}|a^j\rangle$, 5-$\sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|s'\rangle$

### Set B: True/False

| Statement | T/F |
|-----------|-----|
| 1. Shor's algorithm provides exponential speedup. | ___ |
| 2. Grover's algorithm provides exponential speedup. | ___ |
| 3. Grover's algorithm is optimal (proven). | ___ |
| 4. Shor's algorithm is known to be optimal. | ___ |
| 5. BBBV implies NP $\not\subseteq$ BQP. | ___ |
| 6. More Grover iterations always help. | ___ |
| 7. Order-finding reduces to factoring. | ___ |
| 8. Grover works for multiple solutions. | ___ |
| 9. Shor uses the QFT. | ___ |
| 10. BBBV applies to structured search. | ___ |

**Answers:** 1-T, 2-F, 3-T, 4-F, 5-F, 6-F, 7-F (reverse), 8-T, 9-T, 10-F

### Set C: Quick Calculations

1. For $N = 15$, $a = 7$: $\text{ord}_{15}(7) = $ ___

2. For $N = 64$ items: Grover iterations $\approx$ ___

3. For 1024-bit RSA: Approximate qubits needed = ___

4. $\sin\theta = 1/4$: After 3 Grover iterations, amplitude on solution $\approx$ ___

**Answers:** 1-4, 2-$\approx 6$, 3-$\approx 4000$, 4-$\sin(7\theta) \approx \sin(7 \times 0.25) \approx 0.94$

---

## Part III: Algorithm Trace Practice

### Shor's Algorithm Trace

For $N = 21$, $a = 2$:

1. Check $\gcd(2, 21) = $ ___

2. Find order: $2^1 = $ ___, $2^2 = $ ___, $2^3 = $ ___, ..., order $r = $ ___

3. Is $r$ even? ___

4. $a^{r/2} = $ ___ $\equiv$ ___ $\pmod{21}$

5. Is $a^{r/2} \equiv -1$? ___

6. $\gcd(a^{r/2} - 1, 21) = $ ___

7. $\gcd(a^{r/2} + 1, 21) = $ ___

8. Factors: $21 = $ ___ $\times$ ___

### Grover's Algorithm Trace

For $N = 4$, solution at $|11\rangle$:

1. $\theta = \arcsin(1/\sqrt{4}) = $ ___

2. Initial amplitude on $|11\rangle$: ___

3. After 1 iteration, amplitude: $\sin(3\theta) = $ ___

4. Optimal iterations: $k = $ ___

5. Success probability at optimal: ___

---

## Part IV: Proof Skills Assessment

### Can You Prove These?

Rate 1-5 (1 = cannot, 5 = fluently):

| Statement | Confidence |
|-----------|------------|
| 1. Factoring reduces to order-finding | ___/5 |
| 2. $U_a$ has eigenvalues $e^{2\pi ik/r}$ | ___/5 |
| 3. $|1\rangle$ is superposition of eigenstates | ___/5 |
| 4. Grover iterator is rotation by $2\theta$ | ___/5 |
| 5. Optimal Grover iterations is $\pi/(4\theta)$ | ___/5 |
| 6. BBBV lower bound $\Omega(\sqrt{N})$ | ___/5 |
| 7. Amplitude amplification theorem | ___/5 |
| 8. Continued fractions extracts period | ___/5 |

**Proof Confidence Score: ___/40**

---

## Part V: Oral Explanation Self-Rating

Rate your explanation ability (1-5):

| Topic | Clarity | Depth | Speed | Score |
|-------|---------|-------|-------|-------|
| Shor's algorithm overview | ___/5 | ___/5 | ___/5 | ___/15 |
| Order-finding reduction | ___/5 | ___/5 | ___/5 | ___/15 |
| Grover's geometric picture | ___/5 | ___/5 | ___/5 | ___/15 |
| BBBV proof | ___/5 | ___/5 | ___/5 | ___/15 |
| Shor vs. Grover comparison | ___/5 | ___/5 | ___/5 | ___/15 |

**Oral Score: ___/75**

---

## Part VI: Timed Practice Exam

### Instructions
- 45 minutes, no notes
- Show all work

### Problems

**1. (15 points) Shor's Algorithm:**
a) State the reduction from factoring to order-finding (5 pts)
b) Derive the eigenstates of $U_a$ (5 pts)
c) Explain how phase estimation finds the order (5 pts)

**2. (15 points) Grover's Algorithm:**
a) Write the Grover iterator and explain each component (5 pts)
b) Derive the optimal number of iterations (5 pts)
c) Analyze the case of $t$ solutions (5 pts)

**3. (10 points) BBBV Theorem:**
a) State the theorem precisely (3 pts)
b) Outline the proof (5 pts)
c) Explain implications (2 pts)

**4. (10 points) Complexity Analysis:**
a) Calculate Shor's qubit and gate requirements for 512-bit N (5 pts)
b) Calculate Grover's iterations for $N = 10^6$ (5 pts)

### Grading

**Passing: 35/50**

**Your Score: ___/50**

---

## Part VII: Gap Analysis

### Identify Weak Areas

Based on your scores, list top 3 areas for improvement:

1. _________________________________

2. _________________________________

3. _________________________________

### Recommended Actions

| If weak in... | Do this... |
|---------------|------------|
| Shor's reduction | Re-derive from number theory basics |
| Grover geometry | Draw 2D pictures, trace iterations |
| BBBV proof | Study hybrid argument step-by-step |
| Complexity analysis | Practice calculations with specific N |
| Oral explanations | Record yourself, practice with timer |

---

## Part VIII: Progress Tracking

### Daily Log

| Day | Focus | Hours | Problems | Confidence |
|-----|-------|-------|----------|------------|
| 1135 | | | | |
| 1136 | | | | |
| 1137 | | | | |
| 1138 | | | | |
| 1139 | | | | |
| 1140 | | | | |
| 1141 | | | | |

### Weekly Summary

**Starting confidence:** ___/10
**Ending confidence:** ___/10
**Total hours:** ___
**Problems completed:** ___/30

---

## Part IX: Final Readiness Checklist

Before Week 164, verify:

### Shor's Algorithm
- [ ] Can derive factoring-to-order reduction
- [ ] Can explain quantum order finding
- [ ] Can calculate complete complexity
- [ ] Can trace through small example

### Grover's Algorithm
- [ ] Can draw geometric picture
- [ ] Can derive optimal iterations
- [ ] Can handle multiple solutions
- [ ] Can explain why it oscillates

### Optimality
- [ ] Can state BBBV precisely
- [ ] Can outline BBBV proof
- [ ] Understand implications for NP vs BQP

### Comparison
- [ ] Can compare speedup types
- [ ] Can explain why different
- [ ] Can discuss practical implications

**Ready for Week 164?** [ ] Yes [ ] Need review

---

## Key Formulas Reference

### Shor's Algorithm
$$\text{Qubits: } O(\log N), \quad \text{Gates: } O((\log N)^3)$$

$$|u_s\rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1}e^{-2\pi isj/r}|a^j \mod N\rangle$$

### Grover's Algorithm
$$k_{\text{opt}} = \frac{\pi}{4}\sqrt{N/t}$$

$$G^k|s\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|s'\rangle$$

### BBBV
$$T = \Omega(\sqrt{N}) \text{ queries required}$$

---

*Use this self-assessment to track your mastery of Week 163 material.*
