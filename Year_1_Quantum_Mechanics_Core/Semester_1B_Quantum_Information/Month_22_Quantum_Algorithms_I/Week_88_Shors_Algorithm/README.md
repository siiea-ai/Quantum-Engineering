# Week 88: Shor's Algorithm

## Overview

**Days 610-616** | Week 88 | Month 22 | Year 1

This week we study Shor's factoring algorithm, one of the most important quantum algorithms ever discovered. Shor's algorithm factors integers in polynomial time, threatening the security of widely-used cryptographic systems like RSA. We'll understand the complete algorithm from number theory through quantum implementation.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Reduce factoring to order-finding
2. Understand the number theory foundations
3. Implement quantum period-finding with QPE
4. Apply continued fractions for classical post-processing
5. Analyze the complete Shor's algorithm
6. Evaluate complexity and cryptographic implications

---

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 610 | Factoring to Order-Finding | Reduction proof, modular arithmetic |
| 611 | Number Theory Background | CRT, Euler's theorem, group theory |
| 612 | Quantum Period-Finding | QPE for modular exponentiation |
| 613 | Continued Fractions | Classical post-processing |
| 614 | Full Shor's Algorithm | Complete implementation |
| 615 | Complexity Analysis | Gate count, success probability |
| 616 | Month Review | Comprehensive assessment |

---

## Key Concepts

### The Factoring Problem

**Input:** Composite integer $N$
**Output:** Non-trivial factors $p, q$ with $N = pq$

**Classical:** Best known algorithm is $O(\exp(n^{1/3}))$ for $n$-bit $N$

**Quantum (Shor):** $O(n^3)$ operations

### The Key Reduction

Factoring reduces to **order-finding**:
- Pick random $a < N$ with $\gcd(a, N) = 1$
- Find smallest $r$ such that $a^r \equiv 1 \pmod{N}$
- If $r$ is even and $a^{r/2} \not\equiv -1 \pmod{N}$:
  - $\gcd(a^{r/2} - 1, N)$ or $\gcd(a^{r/2} + 1, N)$ is a factor

### Quantum Order-Finding

QPE on $U_a|x\rangle = |ax \mod N\rangle$ extracts eigenvalues $e^{2\pi is/r}$.

From $s/r$, continued fractions recover $r$.

---

## Key Formulas

### Order-Finding Success

$$P(\text{r found}) \geq \frac{1}{2\log_2 N}$$

After $O(\log N)$ repetitions: high probability of success.

### Continued Fractions

$s/r$ from QPE measurement: expand as continued fraction, convergent denominators give $r$ candidates.

---

## Prerequisites

From Weeks 85-87:
- Quantum Fourier Transform
- Quantum Phase Estimation
- Modular arithmetic basics
- Controlled unitary operations

---

## References

- Shor (1994), "Algorithms for quantum computation"
- Nielsen & Chuang, Section 5.3
- Mermin, "Quantum Computer Science" Ch. 3

---

*Week 88 of 88 in Month 22*
