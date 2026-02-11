# Week 85: Deutsch-Jozsa Algorithm

## Overview

**Days 589-595** | Week 85 | Month 22 | Year 1

This week introduces the foundational quantum algorithms that demonstrate provable quantum advantage over classical computation. We begin with the oracle model of computation, progress through Deutsch's single-qubit algorithm, and culminate with Simon's algorithm - the direct precursor to Shor's factoring algorithm.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Define the oracle (black-box) model of quantum computation
2. Analyze query complexity and prove quantum advantages
3. Implement Deutsch's algorithm for single-bit functions
4. Generalize to the n-qubit Deutsch-Jozsa algorithm
5. Solve the hidden string problem with Bernstein-Vazirani
6. Understand Simon's algorithm and exponential quantum speedup
7. Construct quantum oracles from classical function descriptions

---

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 589 | Oracle Model and Query Complexity | Black-box functions, query lower bounds |
| 590 | Deutsch's Algorithm | Single-qubit oracle, phase kickback |
| 591 | Deutsch-Jozsa Generalization | n-qubit constant vs balanced |
| 592 | Bernstein-Vazirani Algorithm | Hidden string, dot product |
| 593 | Simon's Algorithm | Hidden subgroup, period finding |
| 594 | Quantum Oracle Construction | Building oracles from truth tables |
| 595 | Week Review | Problem synthesis, comparisons |

---

## Key Concepts

### The Oracle Model

In the oracle model, we treat a function $f: \{0,1\}^n \to \{0,1\}^m$ as a black box:

$$U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$$

Query complexity measures how many times we must call the oracle to solve a problem.

### Deutsch-Jozsa Promise

Given $f: \{0,1\}^n \to \{0,1\}$ that is either:
- **Constant**: $f(x) = c$ for all $x$
- **Balanced**: $f(x) = 0$ for exactly half the inputs

Classical requires $2^{n-1}+1$ queries; quantum requires **1** query.

### Phase Oracle Trick

Using $|y\rangle = |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$:

$$U_f|x\rangle|-\rangle = (-1)^{f(x)}|x\rangle|-\rangle$$

This converts the oracle to a **phase oracle**.

---

## Key Formulas

### Deutsch-Jozsa State Evolution

$$|0\rangle^{\otimes n}|-\rangle \xrightarrow{H^{\otimes n}} \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle|-\rangle$$

$$\xrightarrow{U_f} \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}(-1)^{f(x)}|x\rangle|-\rangle$$

$$\xrightarrow{H^{\otimes n}} \sum_{y=0}^{2^n-1}\left[\frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{f(x)+x\cdot y}\right]|y\rangle|-\rangle$$

### Bernstein-Vazirani

For $f(x) = s \cdot x$ (mod 2):

$$\text{Measure } |s\rangle \text{ with probability } 1$$

### Simon's Algorithm

For $f(x) = f(y) \Leftrightarrow y = x \oplus s$:

After measurement, obtain $y$ such that $y \cdot s = 0$.

Repeat $O(n)$ times to solve for $s$.

---

## Prerequisites

- Quantum gates and circuits (Month 21)
- Tensor products and multi-qubit states
- Hadamard transform properties
- Basic linear algebra over $\mathbb{Z}_2$

---

## References

- Nielsen & Chuang, Section 1.4.3, 6.1
- Kaye, Laflamme, Mosca, Chapter 5
- Deutsch & Jozsa (1992), Proc. R. Soc. Lond. A

---

*Week 85 of 88 in Month 22*
