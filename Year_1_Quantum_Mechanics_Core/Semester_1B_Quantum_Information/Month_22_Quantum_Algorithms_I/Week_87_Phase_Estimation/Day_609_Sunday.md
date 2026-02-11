# Day 609: Week 87 Review - Phase Estimation

## Overview

**Day 609** | Week 87, Day 7 | Month 22 | Quantum Algorithms I

Today we consolidate our understanding of Quantum Phase Estimation and prepare for Shor's algorithm next week.

---

## Learning Objectives

1. Synthesize all QPE concepts
2. Compare different QPE variants
3. Solve comprehensive problems
4. Connect to applications
5. Preview Shor's algorithm

---

## Week Summary

### Concepts Covered

| Day | Topic | Key Ideas |
|-----|-------|-----------|
| 603 | Eigenvalue Problem | Unitary eigenvalues, phase representation |
| 604 | QPE Circuit | H + controlled-U + QFT^{-1} |
| 605 | Precision Analysis | n bits, sinc distribution |
| 606 | Success Probability | 40.5% minimum, boosting |
| 607 | Iterative QPE | Single ancilla, sequential bits |
| 608 | Kitaev's Algorithm | Statistical, fault-tolerant |

### The Central Algorithm

$$U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle \xrightarrow{\text{QPE}} |\tilde{\phi}\rangle$$

where $\tilde{\phi}$ approximates $\phi$ to $n$ bits.

### Variants Comparison

| Variant | Qubits | Measurements | Entanglement | Noise Robustness |
|---------|--------|--------------|--------------|------------------|
| Standard | $n$ | 1 | Yes | Low |
| Iterative | 1 | $n$ | No (sequential) | Medium |
| Kitaev | 1 | $O(n\log n)$ | No | High |

---

## Comprehensive Problems

### Problem 1: Full QPE Analysis

For $U = T^5$ (five T gates) with eigenstate $|1\rangle$:

(a) What is the eigenvalue?
(b) What is the phase $\phi$?
(c) With 4 ancillas, what measurement outcomes are possible?

**Solution:**

(a) $T = e^{i\pi/4}|1\rangle\langle 1|$ on $|1\rangle$, so $T^5|1\rangle = e^{5i\pi/4}|1\rangle$

(b) $\phi = 5/8 = 0.101$ in binary

(c) $2^4 \cdot 5/8 = 10$ exactly. Output: $|1010\rangle$ with certainty.

### Problem 2: Resource Estimation

Design QPE for Shor's algorithm on 1024-bit RSA:
- Modulus $N$ has 1024 bits
- Need period $r$ with probability $> 99\%$

**Solution:**

- Need $n \geq 2 \cdot 1024 = 2048$ ancilla qubits for precision
- Add $\log(1/0.01) \approx 7$ extra qubits for confidence
- Total: ~2055 ancilla qubits + 1024 for modular exponentiation
- Controlled-$U$ operations: 2055 modular exponentiations

### Problem 3: Iterative vs Standard

For $\phi = 0.625$ with 3-bit precision:

(a) Standard QPE: What state before measurement?
(b) Iterative: What corrections in each round?

**Solution:**

(a) $\phi = 0.625 = 5/8$, $2^3\phi = 5$
State: $|101\rangle$ (exact)

(b) Iterative:
- Round 1: Extract $b_3 = 1$, $\theta = 0.5$
- Round 2: Extract $b_2 = 0$, $\theta = 0.25$
- Round 3: Extract $b_1 = 1$, final $\phi = 0.101 = 0.625$

---

## Connection to Shor's Algorithm

QPE is the quantum heart of Shor's algorithm:

1. **Setup:** $U_a|x\rangle = |ax \mod N\rangle$
2. **QPE:** Extracts eigenvalues $e^{2\pi is/r}$
3. **Post-process:** Classical continued fractions gives $s/r$
4. **Factor:** $\gcd(a^{r/2} \pm 1, N)$ likely gives factors

Next week: The complete Shor's algorithm!

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| QPE output | $\|\tilde{\phi}\rangle$ where $\tilde{\phi} = m/2^n$ |
| Precision | $\|\phi - \tilde{\phi}\| < 1/2^n$ |
| Success (exact) | $P = 1$ if $2^n\phi \in \mathbb{Z}$ |
| Success (approx) | $P \geq 4/\pi^2 \approx 0.405$ |

### Key Takeaways

1. **QPE extracts phases** from unitary eigenvalues
2. **Standard QPE** uses $n$ ancillas and QFT
3. **Iterative variants** trade qubits for rounds
4. **Kitaev's method** is noise-robust
5. **QPE enables** Shor, HHL, quantum simulation

---

## Daily Checklist

- [ ] I can design QPE circuits
- [ ] I understand precision vs resources trade-off
- [ ] I know different QPE variants
- [ ] I can analyze success probabilities
- [ ] I see QPE's role in quantum algorithms
- [ ] I'm ready for Shor's algorithm

---

*End of Week 87 | Next: Week 88 - Shor's Algorithm*
