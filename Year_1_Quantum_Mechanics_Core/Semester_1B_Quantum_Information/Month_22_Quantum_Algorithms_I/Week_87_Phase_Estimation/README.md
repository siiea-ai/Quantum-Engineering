# Week 87: Quantum Phase Estimation

## Overview

**Days 603-609** | Week 87 | Month 22 | Year 1

This week we study Quantum Phase Estimation (QPE), one of the most important subroutines in quantum computing. QPE extracts eigenvalues from unitary operators and is the key component of Shor's factoring algorithm, quantum simulation, and the HHL algorithm for linear systems.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Understand the eigenvalue problem in quantum computing
2. Design and analyze the standard QPE circuit
3. Calculate precision requirements and success probabilities
4. Implement iterative phase estimation variants
5. Understand Kitaev's algorithm for robust phase estimation
6. Apply QPE to practical problems

---

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 603 | Eigenvalue Problem in QC | Unitary eigenvalues, phase extraction |
| 604 | QPE Circuit Design | Control register, controlled-U operations |
| 605 | QPE Analysis and Precision | Bit precision, register size |
| 606 | Success Probability | Error bounds, failure cases |
| 607 | Iterative Phase Estimation | Single ancilla methods |
| 608 | Kitaev's Algorithm | Robust estimation, fault tolerance |
| 609 | Week Review | Applications, synthesis |

---

## Key Concepts

### The Phase Estimation Problem

Given:
- A unitary operator $U$
- An eigenstate $|\psi\rangle$ with $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$

Find: The phase $\phi \in [0, 1)$

### Standard QPE Circuit

```
|0⟩^⊗n ─[H^⊗n]──[CU^{2^{n-1}}]──...──[CU^1]──[QFT^{-1}]── Measure
                    │                   │
|ψ⟩    ─────────────U──────────────────U──────────────────
```

### Precision and Accuracy

With $n$ ancilla qubits:
- Estimate $\tilde{\phi}$ approximating true $\phi$
- Error: $|\phi - \tilde{\phi}| < 1/2^n$ with high probability

---

## Key Formulas

### QPE State Evolution

After controlled operations:
$$\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1}e^{2\pi ik\phi}|k\rangle|\psi\rangle$$

After inverse QFT (exact case):
$$|2^n\phi\rangle|\psi\rangle$$

### Success Probability

$$P(\text{exact}) = 1 \text{ if } 2^n\phi \in \mathbb{Z}$$

$$P(\text{within } \pm 1) \geq \frac{4}{\pi^2} \approx 0.405 \text{ otherwise}$$

---

## Prerequisites

From Week 86:
- Quantum Fourier Transform
- Inverse QFT circuit
- Phase kickback mechanism
- Controlled unitary operations

---

## References

- Nielsen & Chuang, Section 5.2
- Kitaev (1995), "Quantum measurements and the Abelian Stabilizer Problem"
- Cleve et al. (1998), "Quantum algorithms revisited"

---

*Week 87 of 88 in Month 22*
