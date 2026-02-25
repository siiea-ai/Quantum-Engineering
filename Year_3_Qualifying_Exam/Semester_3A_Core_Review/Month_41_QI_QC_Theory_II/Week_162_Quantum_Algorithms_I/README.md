# Week 162: Quantum Algorithms I

## Foundational Algorithms and Fourier-Based Techniques

**Days:** 1128-1134
**Theme:** Deutsch-Jozsa, Simon's algorithm, Quantum Fourier Transform, and Phase Estimation

---

## Week Overview

This week covers the foundational quantum algorithms that demonstrate exponential speedups over classical computation. We begin with the conceptually simpler Deutsch-Jozsa and Simon's algorithms, which established the theoretical basis for quantum advantage, then progress to the Quantum Fourier Transform and Phase Estimation - the workhorses underlying Shor's algorithm and many other quantum speedups.

### Core Learning Objectives

By the end of this week, you will be able to:

1. Analyze and implement the Deutsch-Jozsa algorithm, explaining its exponential advantage
2. Derive Simon's algorithm and understand its role as Shor's precursor
3. Construct the Quantum Fourier Transform circuit and analyze its complexity
4. Apply Phase Estimation to eigenvalue problems with precision analysis
5. Connect these algorithms to the hidden subgroup problem framework

---

## Daily Schedule

### Day 1128 (Monday): Deutsch-Jozsa Algorithm
**Focus:** Oracle model, single-query solution, exponential speedup

**Key Topics:**
- The oracle (black-box) model of computation
- Constant vs. balanced functions
- Deutsch's algorithm (single qubit)
- Deutsch-Jozsa algorithm (n qubits)
- Query complexity analysis

**Essential Result:**
Classical: Requires $2^{n-1} + 1$ queries (worst case)
Quantum: Requires 1 query

$$|\psi_{\text{final}}\rangle = H^{\otimes n}\left[\frac{1}{\sqrt{2^n}}\sum_{x}(-1)^{f(x)}|x\rangle\right]|{-}\rangle$$

### Day 1129 (Tuesday): Bernstein-Vazirani & Hidden Subgroup
**Focus:** Linear function extraction, HSP framework

**Key Topics:**
- Bernstein-Vazirani problem: find $s$ given $f(x) = s \cdot x$
- One-query quantum solution
- Introduction to Hidden Subgroup Problem (HSP)
- Abelian vs. non-Abelian HSP
- Connections to lattice problems

**Key Insight:** Many quantum speedups reduce to finding hidden subgroups.

### Day 1130 (Wednesday): Simon's Algorithm
**Focus:** Period finding precursor, exponential separation

**Key Topics:**
- Simon's problem: find $s$ where $f(x) = f(y) \Leftrightarrow x \oplus y \in \{0, s\}$
- Algorithm structure and analysis
- $O(n)$ quantum queries vs. $O(2^{n/2})$ classical queries
- Connection to Shor's algorithm
- Proof of correctness

**Historical Significance:** First exponential quantum speedup for a computational problem (1994).

### Day 1131 (Thursday): Quantum Fourier Transform I
**Focus:** Definition, circuit construction, complexity analysis

**Key Topics:**
- Definition of QFT on $n$ qubits
- Relationship to classical DFT
- Circuit construction using H and controlled-phase gates
- Gate count: $O(n^2)$
- Product representation of QFT

**Essential Formula:**
$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}\omega^{jk}|k\rangle, \quad \omega = e^{2\pi i/N}$$

### Day 1132 (Friday): Quantum Fourier Transform II
**Focus:** Applications, approximate QFT, error analysis

**Key Topics:**
- Approximate QFT with $O(n\log n)$ gates
- Semiclassical QFT
- QFT for phase kickback
- Applications beyond factoring
- Connection to signal processing

**Key Result:** Approximate QFT maintains fidelity with polynomially fewer gates.

### Day 1133 (Saturday): Phase Estimation I
**Focus:** Algorithm design, precision requirements

**Key Topics:**
- Phase estimation problem statement
- Algorithm structure: controlled-U powers + inverse QFT
- Number of ancilla qubits for precision $\epsilon$
- Success probability analysis
- Handling non-exact phases

**Algorithm:**
1. Prepare $|0\rangle^{\otimes t}|u\rangle$ (eigenstate of $U$)
2. Apply Hadamards to ancilla
3. Apply controlled-$U^{2^j}$ operations
4. Apply inverse QFT to ancilla
5. Measure ancilla to obtain phase estimate

### Day 1134 (Sunday): Phase Estimation II & Week Review
**Focus:** Applications, eigenvalue problems, comprehensive review

**Key Topics:**
- Application to Hamiltonian simulation
- Connection to quantum chemistry (VQE)
- Ground state energy estimation
- Iterative phase estimation (fewer qubits)
- Week review and integration

---

## Key Algorithms Summary

### Deutsch-Jozsa Algorithm

**Problem:** Given $f:\{0,1\}^n \to \{0,1\}$ promised to be constant or balanced, determine which.

**Circuit:**
```
|0⟩^n --H^n--[Oracle f]--H^n--[Measure]
|1⟩   --H---[         ]-------
```

**Result:** Output is $|0\rangle^{\otimes n}$ if and only if $f$ is constant.

### Simon's Algorithm

**Problem:** Given $f:\{0,1\}^n \to \{0,1\}^n$ where $f(x) = f(y) \Leftrightarrow x \oplus y \in \{0^n, s\}$, find $s$.

**Procedure:**
1. Repeat $O(n)$ times:
   - Prepare $H^{\otimes n}|0\rangle^{\otimes n}|0\rangle^{\otimes n}$
   - Apply oracle: $|x\rangle|0\rangle \to |x\rangle|f(x)\rangle$
   - Apply $H^{\otimes n}$ to first register
   - Measure first register, get $y$ where $y \cdot s = 0$
2. Solve linear system to find $s$

### Quantum Fourier Transform

**Circuit Structure (3 qubits):**
```
|j₂⟩--H--R₂--R₃--------×
|j₁⟩-----H--R₂----×----×
|j₀⟩--------H-----×------
```
Where $R_k = \text{diag}(1, e^{2\pi i/2^k})$

**Complexity:** $O(n^2)$ gates, $O(n)$ depth with parallelization.

### Phase Estimation

**If $U|u\rangle = e^{2\pi i\theta}|u\rangle$, estimate $\theta$ to $t$ bits of precision.**

**Resource Requirements:**
- $t$ ancilla qubits for $t$-bit precision
- Controlled-$U^{2^j}$ operations for $j = 0, 1, \ldots, t-1$
- Inverse QFT on ancilla

---

## Qualifying Exam Focus Areas

### Commonly Asked Questions

1. "Explain the Deutsch-Jozsa algorithm and prove it works"
2. "How does Simon's algorithm achieve exponential speedup?"
3. "Derive the QFT circuit for $n$ qubits"
4. "What is the relationship between phase estimation and eigenvalue problems?"
5. "How does approximate QFT maintain accuracy with fewer gates?"

### Key Proofs to Know

1. Deutsch-Jozsa correctness (interference pattern)
2. Simon's algorithm query complexity
3. QFT circuit derivation from definition
4. Phase estimation success probability

---

## Study Resources

### Primary References
- Nielsen & Chuang, Chapter 5 (Quantum algorithms)
- Kaye, Laflamme & Mosca, Chapters 6-7
- Vazirani lecture notes (Berkeley CS 294)

### Research Papers
- Deutsch & Jozsa (1992), "Rapid solution of problems by quantum computation"
- Simon (1994), "On the power of quantum computation"
- Coppersmith (1994), "An approximate Fourier transform useful in quantum factoring"

### Online Resources
- IBM Quantum Learning: Algorithm tutorials
- Qiskit Textbook: Chapter on Quantum Algorithms

---

## Week Deliverables

| Component | Description | Location |
|-----------|-------------|----------|
| Review Guide | Comprehensive topic summary | `Review_Guide.md` |
| Problem Set | 28 problems, all difficulty levels | `Problem_Set.md` |
| Solutions | Complete worked solutions | `Problem_Solutions.md` |
| Oral Practice | Discussion questions, mock scenarios | `Oral_Practice.md` |
| Self-Assessment | Mastery checklists, diagnostics | `Self_Assessment.md` |

---

## Success Criteria

By week's end, you should be able to:

- [ ] Implement Deutsch-Jozsa on paper and explain each step
- [ ] Derive Simon's algorithm complexity
- [ ] Construct QFT circuit for arbitrary $n$
- [ ] Calculate phase estimation precision requirements
- [ ] Connect algorithms to Hidden Subgroup Problem
- [ ] Explain quantum speedup sources (interference, entanglement)

---

*This week establishes the algorithmic foundations for understanding Shor's and Grover's algorithms in Week 163.*
