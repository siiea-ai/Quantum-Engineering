# Week 163: Quantum Algorithms II

## Shor's Algorithm, Grover's Search, and Optimality Proofs

**Days:** 1135-1141
**Theme:** Complete analysis of the landmark quantum algorithms

---

## Week Overview

This week provides deep, comprehensive treatment of the two most important quantum algorithms: Shor's factoring algorithm and Grover's search algorithm. For qualifying exams, students must be able to derive these algorithms from first principles, analyze their complexity rigorously, and prove Grover's optimality using the BBBV theorem. This material represents the core of quantum algorithm theory.

### Core Learning Objectives

By the end of this week, you will be able to:

1. Derive Shor's algorithm from the reduction of factoring to order-finding
2. Analyze the complete complexity of Shor's algorithm including success probability
3. Prove Grover's algorithm achieves $O(\sqrt{N})$ query complexity
4. Prove Grover's algorithm is optimal using the BBBV theorem
5. Apply amplitude amplification as a general framework
6. Compare and contrast these algorithms' speedup mechanisms

---

## Daily Schedule

### Day 1135 (Monday): Shor's Algorithm I - Order Finding Reduction
**Focus:** Reducing factoring to period finding

**Key Topics:**
- Factoring problem and its cryptographic importance
- Reduction from factoring to order-finding
- Order-finding problem: find $r$ such that $a^r \equiv 1 \pmod{N}$
- Number-theoretic lemmas
- Overview of full algorithm structure

**Essential Theorem:**
If we can find $r = \text{ord}_N(a)$ for random $a$, then with probability $\geq 1/2$:
$$\gcd(a^{r/2} \pm 1, N)$$
gives a non-trivial factor of $N$.

### Day 1136 (Tuesday): Shor's Algorithm II - Quantum Order Finding
**Focus:** Phase estimation for period finding

**Key Topics:**
- Modular exponentiation as a unitary
- Eigenstates of $U_a|x\rangle = |ax \mod N\rangle$
- Phase estimation to find eigenvalues
- Recovering the period from phase measurements
- Continued fractions algorithm

**Key Insight:**
Eigenstates of $U_a$ have eigenvalues $e^{2\pi ik/r}$ where $r$ is the order.

### Day 1137 (Wednesday): Shor's Algorithm III - Complete Complexity Analysis
**Focus:** Resource requirements and success probability

**Key Topics:**
- Number of qubits needed: $O(\log N)$
- Circuit depth: $O((\log N)^2 \log\log N \log\log\log N)$
- Modular exponentiation circuit
- Success probability analysis
- Repeat bounds for high confidence

**Essential Result:**
$$\boxed{\text{Shor's Algorithm: } O((\log N)^3) \text{ quantum operations}}$$

### Day 1138 (Thursday): Grover's Algorithm I - Oracle and Amplitude Amplification
**Focus:** Search problem, oracle construction, basic algorithm

**Key Topics:**
- Unstructured search problem
- Oracle construction: $O|x\rangle = (-1)^{f(x)}|x\rangle$
- Diffusion operator: $D = 2|s\rangle\langle s| - I$
- Grover iteration: $G = DO$
- Geometric interpretation

**Grover Iterator:**
$$G = (2|s\rangle\langle s| - I)O_f$$

where $|s\rangle = \frac{1}{\sqrt{N}}\sum_x |x\rangle$.

### Day 1139 (Friday): Grover's Algorithm II - Analysis and Extensions
**Focus:** Optimal iterations, multiple solutions, error analysis

**Key Topics:**
- Optimal number of iterations: $\frac{\pi}{4}\sqrt{N}$
- Analysis for $t$ solutions (unknown $t$)
- Amplitude estimation to find $t$
- Quantum counting
- Fixed-point amplitude amplification

**Key Formula:**
For $t$ solutions among $N$ items:
$$k_{\text{opt}} = \frac{\pi}{4}\sqrt{N/t}$$

### Day 1140 (Saturday): Grover Optimality - BBBV Theorem
**Focus:** Proving $\Omega(\sqrt{N})$ lower bound

**Key Topics:**
- Query complexity model
- Polynomial method (brief)
- Adversary method (brief)
- BBBV theorem proof
- Implications for P vs NP

**BBBV Theorem:**
Any quantum algorithm that distinguishes a function with no solutions from one with one solution requires $\Omega(\sqrt{N})$ queries.

### Day 1141 (Sunday): Amplitude Amplification & Week Review
**Focus:** General framework, applications, comprehensive review

**Key Topics:**
- General amplitude amplification theorem
- Applications: element distinctness, collision finding
- Quantum walks connection
- Week review and synthesis
- Comparison: Shor vs. Grover speedup mechanisms

---

## Key Algorithm Details

### Shor's Algorithm - Complete Structure

**Input:** Integer $N$ to factor

**Algorithm:**
1. Choose random $a$ with $1 < a < N$
2. Compute $\gcd(a, N)$. If $> 1$, return factor.
3. Use quantum order-finding to get $r = \text{ord}_N(a)$
4. If $r$ is odd or $a^{r/2} \equiv -1 \pmod{N}$, go to step 1
5. Return $\gcd(a^{r/2} \pm 1, N)$

**Quantum Subroutine (Order Finding):**
1. Prepare $|0\rangle^{\otimes 2n}|1\rangle$
2. Apply Hadamard to first register
3. Apply controlled modular exponentiation: $|j\rangle|1\rangle \to |j\rangle|a^j \mod N\rangle$
4. Apply inverse QFT to first register
5. Measure, get $\tilde{\theta}$ approximating $k/r$
6. Use continued fractions to extract $r$

**Complexity:**
- Qubits: $3n + O(1)$ where $n = \lceil\log_2 N\rceil$
- Gates: $O(n^2(n + \log n)\log n) = O(n^3)$ using fast multiplication
- Success probability per run: $\Omega(1/\log\log N)$

### Grover's Algorithm - Complete Structure

**Input:** Oracle $O_f$ for $f:\{0,1\}^n \to \{0,1\}$, seeking $x$ with $f(x) = 1$

**Algorithm:**
1. Initialize $|s\rangle = H^{\otimes n}|0\rangle^{\otimes n}$
2. Repeat $k = \lfloor\frac{\pi}{4}\sqrt{N}\rfloor$ times:
   - Apply oracle: $O_f|x\rangle = (-1)^{f(x)}|x\rangle$
   - Apply diffusion: $D = 2|s\rangle\langle s| - I = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n}$
3. Measure in computational basis

**Analysis (single solution):**
- Let $|w\rangle$ = solution, $|s'\rangle$ = uniform superposition of non-solutions
- $|s\rangle = \sin\theta|w\rangle + \cos\theta|s'\rangle$ where $\sin\theta = 1/\sqrt{N}$
- After $k$ iterations: $G^k|s\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|s'\rangle$
- Maximum at $(2k+1)\theta \approx \pi/2$, giving $k \approx \frac{\pi}{4}\sqrt{N}$

**Success Probability:** $\sin^2((2k+1)\theta) \geq 1 - O(1/N)$

---

## Qualifying Exam Focus Areas

### Shor's Algorithm Questions

1. "Explain how factoring reduces to order-finding."
2. "Derive the eigenstates of the modular exponentiation unitary."
3. "Why does continued fractions work for extracting the period?"
4. "What is the complete complexity of Shor's algorithm?"
5. "How many qubits are needed to factor a 2048-bit RSA number?"

### Grover's Algorithm Questions

1. "Prove Grover's algorithm finds a solution with high probability."
2. "Why can't we just run more iterations to increase success probability?"
3. "Prove the $\Omega(\sqrt{N})$ lower bound (BBBV theorem)."
4. "How does Grover change when there are multiple solutions?"
5. "What is amplitude amplification and how does it generalize Grover?"

### Comparison Questions

1. "Compare the speedup mechanisms in Shor's vs. Grover's algorithms."
2. "Which provides stronger evidence that quantum computers are powerful?"
3. "Which is more likely to be practical in the near term?"

---

## Study Resources

### Primary References
- Nielsen & Chuang, Chapters 5.3, 6
- Kaye, Laflamme & Mosca, Chapters 8-10
- Vazirani lecture notes (Berkeley)

### Research Papers
- Shor (1994), "Algorithms for quantum computation"
- Grover (1996), "A fast quantum mechanical algorithm for database search"
- Bennett, Bernstein, Brassard, Vazirani (1997), "Strengths and weaknesses of quantum computing"
- Brassard, Hoyer, Mosca, Tapp (2000), "Quantum amplitude amplification"

---

## Week Deliverables

| Component | Description | Location |
|-----------|-------------|----------|
| Review Guide | Comprehensive topic summary | `Review_Guide.md` |
| Problem Set | 30 problems including optimality proof | `Problem_Set.md` |
| Solutions | Complete worked solutions | `Problem_Solutions.md` |
| Oral Practice | Discussion questions, mock scenarios | `Oral_Practice.md` |
| Self-Assessment | Mastery checklists, diagnostics | `Self_Assessment.md` |

---

## Success Criteria

By week's end, you should be able to:

- [ ] Derive Shor's algorithm from scratch
- [ ] Calculate Shor's success probability and complexity
- [ ] Derive optimal Grover iteration count
- [ ] Prove BBBV lower bound
- [ ] Apply amplitude amplification framework
- [ ] Compare Shor's and Grover's speedup sources

---

*This week covers the most important algorithms in quantum computing. Mastery is essential for qualifying exam success.*
