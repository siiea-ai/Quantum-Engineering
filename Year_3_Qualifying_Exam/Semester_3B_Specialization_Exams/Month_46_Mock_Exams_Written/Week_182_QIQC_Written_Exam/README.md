# Week 182: Quantum Information & Computing Written Exam

## Overview

**Days:** 1268-1274 (7 days)
**Theme:** Full PhD-level quantum information and computing qualifying examination
**Exam Duration:** 3 hours
**Problems:** 8 problems covering core QI/QC topics

---

## Week Objectives

By the end of this week, you will:

1. Complete a full 3-hour quantum information/computing qualifying exam under timed conditions
2. Self-grade your exam using the detailed rubric provided
3. Perform comprehensive error analysis to identify knowledge gaps
4. Complete targeted remediation for weak areas
5. Demonstrate proficiency across density matrices, entanglement, gates, algorithms, and channels
6. Build on lessons learned from Week 181 QM exam

---

## Daily Schedule

### Day 1268 (Pre-Exam Preparation)

| Time Block | Activity | Duration |
|------------|----------|----------|
| Morning | Review density matrices and quantum states | 2 hours |
| Late Morning | Entanglement theory and measures review | 1.5 hours |
| Afternoon | Quantum algorithms review (Shor, Grover, QFT) | 2 hours |
| Evening | Light review, rest for exam | 1.5 hours |

**Key Topics to Review:**
- Partial trace and reduced density matrices
- Bell inequalities and CHSH
- Circuit identities and gate decompositions
- Algorithm complexity and query bounds

### Day 1269 (EXAM DAY)

| Time Block | Activity | Duration |
|------------|----------|----------|
| Morning | Mental preparation, light breakfast | 30 min |
| **9:00 AM - 12:00 PM** | **WRITTEN EXAM** (strict 3-hour limit) | 3 hours |
| Afternoon | Rest and initial reflection | 2 hours |
| Evening | Note areas of difficulty (no solutions yet) | 1 hour |

**EXAM CONDITIONS:**
- No notes, textbooks, or internet
- Only blank paper and writing instruments
- Timer visible
- No breaks longer than 5 minutes

### Day 1270 (Self-Grading)

| Time Block | Activity | Duration |
|------------|----------|----------|
| Morning | Grade Problems 1-4 with rubric | 2 hours |
| Afternoon | Grade Problems 5-8 with rubric | 2 hours |
| Evening | Calculate total score, compare to Week 181 | 1 hour |

### Day 1271 (Error Analysis)

| Time Block | Activity | Duration |
|------------|----------|----------|
| Morning | Classify errors by type | 2 hours |
| Afternoon | Identify root causes | 2 hours |
| Evening | Update remediation priorities | 1 hour |

### Day 1272 (Remediation Day 1)

| Time Block | Activity | Duration |
|------------|----------|----------|
| Morning | Deep study of weakest QI/QC topic | 2.5 hours |
| Afternoon | Practice problems on weak area | 2.5 hours |
| Evening | Review and consolidate | 1 hour |

### Day 1273 (Remediation Day 2)

| Time Block | Activity | Duration |
|------------|----------|----------|
| Morning | Study second weakest topic | 2.5 hours |
| Afternoon | More practice problems | 2.5 hours |
| Evening | Cross-topic integration | 1 hour |

### Day 1274 (Integration)

| Time Block | Activity | Duration |
|------------|----------|----------|
| Morning | Rework missed problems from scratch | 2 hours |
| Afternoon | Strategy refinement | 1.5 hours |
| Evening | Prepare for Week 183 QEC exam | 1.5 hours |

---

## Exam Topics Coverage

### Distribution of Problems

| Topic | Problems | Points |
|-------|----------|--------|
| Density Matrices & Mixed States | 1 | 25 |
| Entanglement & Bell Inequalities | 1 | 25 |
| Quantum Gates & Circuits | 1 | 25 |
| Quantum Algorithms I (Deutsch-Jozsa, Simon) | 1 | 25 |
| Quantum Algorithms II (Shor, Grover) | 1 | 25 |
| Quantum Channels & Noise | 1 | 25 |
| Quantum Complexity | 1 | 25 |
| Quantum Protocols | 1 | 25 |
| **Total** | **8** | **200** |

### Concept Map

```
Density Matrices
├── Pure vs mixed states
├── Partial trace
├── Bloch sphere (pure and mixed)
└── Quantum operations

Entanglement
├── Separable vs entangled
├── Bell states, GHZ, W states
├── CHSH inequality
├── Entanglement measures (entropy, concurrence)
└── Monogamy

Quantum Gates
├── Single-qubit (Pauli, H, phase, rotation)
├── Two-qubit (CNOT, CZ, SWAP)
├── Universal gate sets
├── Gate synthesis
└── Solovay-Kitaev

Quantum Algorithms
├── Oracle model, query complexity
├── Deutsch-Jozsa, Bernstein-Vazirani
├── Simon's algorithm
├── QFT and phase estimation
├── Shor's algorithm
├── Grover's search
└── Amplitude amplification

Quantum Channels
├── CPTP maps
├── Kraus representation
├── Common channels (depolarizing, amplitude damping)
├── Channel capacity
└── Quantum-classical channels

Quantum Complexity
├── BQP, QMA, QCMA
├── Promise problems
├── Quantum advantages
└── Hardness results
```

---

## Key Formulas to Know

### Density Matrices

$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|, \quad \text{Tr}(\rho) = 1, \quad \rho \geq 0$$

$$\rho_A = \text{Tr}_B(\rho_{AB})$$

$$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

### Entanglement

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$\text{CHSH: } |\langle A_1 B_1 \rangle + \langle A_1 B_2 \rangle + \langle A_2 B_1 \rangle - \langle A_2 B_2 \rangle| \leq 2\sqrt{2}$$

$$E(\rho) = S(\rho_A) \text{ for pure states}$$

### Quantum Gates

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$

### Quantum Algorithms

$$|x\rangle \xrightarrow{QFT} \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi i xk/N}|k\rangle$$

$$\text{Grover: } O(\sqrt{N}) \text{ queries for search}$$

$$\text{Shor: } O((\log N)^3) \text{ for factoring}$$

### Quantum Channels

$$\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger, \quad \sum_k K_k^\dagger K_k = I$$

$$\text{Depolarizing: } \mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

---

## Files in This Week

| File | Purpose |
|------|---------|
| `README.md` | This overview document |
| `Written_Exam.md` | The 3-hour exam (8 problems) |
| `Exam_Solutions.md` | Complete detailed solutions |
| `Grading_Rubric.md` | Point-by-point grading guide |
| `Error_Analysis.md` | Template for analyzing mistakes |

---

## Comparison to Week 181

### Building on QM Foundation

| QM Concept | QI/QC Extension |
|------------|-----------------|
| Pure states $|\psi\rangle$ | Density matrices $\rho$ |
| Unitary evolution $\hat{U}$ | Quantum channels $\mathcal{E}$ |
| Observable eigenstates | Computational basis, POVMs |
| Tensor products | Entanglement, composite systems |
| Measurement postulate | Quantum protocols |

### Exam Strategy Adjustments

Based on Week 181 experience:
- Allocate time based on problem complexity
- Prioritize circuit problems (often most straightforward)
- Algorithm problems may require more setup time
- Show all work for partial credit on proofs

---

## References

- Nielsen & Chuang, *Quantum Computation and Quantum Information*
- Preskill, *Ph219 Lecture Notes* (Caltech)
- Wilde, *Quantum Information Theory*
- Watrous, *Theory of Quantum Information*

---

*"Quantum information builds on quantum mechanics but has its own vocabulary and techniques. Master both."*

---

**Status:** NOT STARTED
**Progress:** 0/7 days (0%)
