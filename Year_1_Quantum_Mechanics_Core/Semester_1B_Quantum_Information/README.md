# Semester 1B: Quantum Information Foundations

## Overview

**Semester 1B** | Months 19-24 | Days 505-672 | Quantum Information & Computing Foundations

This semester introduces quantum information theory, connecting traditional quantum mechanics to quantum computing. This corresponds to the core of Harvard's QSE 200 curriculum.

---

## Semester Structure

| Month | Topic | Weeks | Days | Status |
|-------|-------|-------|------|--------|
| **19** | Density Matrices & Mixed States | 73-76 | 505-532 | ✅ Complete |
| **20** | Entanglement Theory | 77-80 | 533-560 | ✅ Complete |
| **21** | Quantum Gates & Circuits | 81-84 | 561-588 | ✅ Complete |
| **22** | Quantum Algorithms I | 85-88 | 589-616 | ✅ Complete |
| **23** | Quantum Algorithms II | 89-92 | 617-644 | ✅ Complete |
| **24** | Quantum Channels & Error Intro | 93-96 | 645-672 | ✅ Complete |

**Total:** 168 days | 24 weeks | 6 months

---

## Primary Texts

| Text | Author(s) | Role |
|------|-----------|------|
| **Quantum Computation and Quantum Information** | Nielsen & Chuang | Primary textbook |
| **Ph219 Lecture Notes** | John Preskill (Caltech) | Theory reference |
| **Quantum Information Theory** | Mark Wilde | Advanced reference |
| **Principles of Quantum Mechanics** | R. Shankar | QM foundation |

---

## Learning Objectives

Upon completing Semester 1B, students will be able to:

### Theoretical Foundations
1. Master density matrix formalism and mixed states
2. Understand partial trace and reduced density matrices
3. Characterize and quantify entanglement
4. Derive and apply Bell inequalities

### Quantum Computing
5. Implement universal single-qubit and two-qubit gates
6. Design quantum circuits and understand universality
7. Analyze quantum algorithms (Deutsch-Jozsa, Simon, Shor, Grover)
8. Understand the quantum Fourier transform and phase estimation

### Quantum Information
9. Describe quantum channels using Kraus operators
10. Understand decoherence and open quantum systems
11. Implement quantum protocols (teleportation, superdense coding)
12. Preview quantum error correction concepts

---

## Month Summaries

### Month 19: Density Matrices & Mixed States (Days 505-532)

**Primary Reference:** Nielsen & Chuang Chapter 2, Preskill Ph219

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 73 | 505-511 | Pure vs Mixed States | Density operator, trace properties, Bloch sphere |
| 74 | 512-518 | Composite Systems | Tensor products, partial trace, Schmidt decomposition |
| 75 | 519-525 | Generalized Measurements | POVMs, Neumark's theorem, measurement implementations |
| 76 | 526-532 | Quantum Dynamics | Unitary evolution, CP maps, Kraus representation intro |

### Month 20: Entanglement Theory (Days 533-560)

**Primary Reference:** Nielsen & Chuang Chapters 2.6, 12

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 77 | 533-539 | Entanglement Basics | Separable vs entangled, Bell states, witnesses |
| 78 | 540-546 | Bell Inequalities | EPR paradox, CHSH inequality, quantum violation |
| 79 | 547-553 | Entanglement Measures | Von Neumann entropy, concurrence, entanglement of formation |
| 80 | 554-560 | Entanglement Applications | Teleportation, superdense coding, entanglement swapping |

### Month 21: Quantum Gates & Circuits (Days 561-588)

**Primary Reference:** Nielsen & Chuang Chapter 4

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 81 | 561-567 | Single-Qubit Gates | Pauli gates, Hadamard, rotations, Bloch sphere |
| 82 | 568-574 | Two-Qubit Gates | CNOT, CZ, SWAP, entangling gates |
| 83 | 575-581 | Universal Gate Sets | Universality proofs, Clifford+T, Solovay-Kitaev |
| 84 | 582-588 | Circuit Model | Circuit diagrams, depth, optimization, MBQC preview |

### Month 22: Quantum Algorithms I (Days 589-616)

**Primary Reference:** Nielsen & Chuang Chapter 5

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 85 | 589-595 | Query Complexity | Oracle model, Deutsch-Jozsa, Bernstein-Vazirani |
| 86 | 596-602 | Simon's Algorithm | Hidden subgroup, exponential speedup |
| 87 | 603-609 | Quantum Fourier Transform | QFT circuit, comparison to FFT |
| 88 | 610-616 | Phase Estimation | QPE algorithm, eigenvalue problems, applications |

### Month 23: Quantum Algorithms II (Days 617-644)

**Primary Reference:** Nielsen & Chuang Chapters 5-6

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 89 | 617-623 | Shor's Algorithm: Number Theory | Modular arithmetic, period finding, factoring reduction |
| 90 | 624-630 | Shor's Algorithm: Implementation | Order finding, modular exponentiation, complete analysis |
| 91 | 631-637 | Grover's Search | Amplitude amplification, optimal iterations, applications |
| 92 | 638-644 | Variational Algorithms | VQE, QAOA, hybrid quantum-classical computing |

### Month 24: Quantum Channels & Error Intro (Days 645-672)

**Primary Reference:** Nielsen & Chuang Chapter 8

| Week | Days | Topic | Key Concepts |
|------|------|-------|--------------|
| 93 | 645-651 | Quantum Channels | CPTP maps, Kraus operators, Choi-Jamiolkowski |
| 94 | 652-658 | Open Quantum Systems | System-environment model, Lindblad equation, decoherence |
| 95 | 659-665 | Year 1 Integration | Comprehensive review, practice qualifying problems |
| 96 | 666-672 | Year 1 Capstone | Complete project: Quantum teleportation with noise analysis |

---

## Computational Tools

### Required Software
- **Python 3.9+** with NumPy, SciPy, Matplotlib
- **Qiskit** (IBM Quantum SDK)
- **Jupyter Notebooks** for interactive development

### Hardware Access
- IBM Quantum Lab (free tier)
- Qiskit Aer simulator

### Key Libraries
```python
# Core quantum computing stack
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# IBM Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
```

---

## Key Formulas Reference

### Density Matrices
$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|, \quad \text{Tr}(\rho) = 1, \quad \rho \geq 0$$

### Partial Trace
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B)\rho_{AB}(I_A \otimes |j\rangle_B)$$

### Bell States
$$|\Phi^\pm\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle), \quad |\Psi^\pm\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)$$

### CHSH Inequality
$$|S| = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| \leq 2 \text{ (classical)}, \quad \leq 2\sqrt{2} \text{ (quantum)}$$

### Quantum Fourier Transform
$$|j\rangle \rightarrow \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle$$

### Grover Iteration
$$G = (2|\psi\rangle\langle\psi| - I)O, \quad \text{iterations} = O(\sqrt{N})$$

---

## Assessment Structure

### Problem Sets (Weekly)
- Theoretical problems from Nielsen & Chuang
- Computational implementations in Qiskit
- Proofs and derivations

### Monthly Assessments
- Comprehensive exams covering month material
- Lab practical: Circuit implementation and analysis
- Oral presentation of key concepts

### Semester Capstone (Week 96)
- Complete quantum protocol implementation
- Noise analysis and error characterization
- Written report and presentation

---

## Prerequisites

Completion of Semester 1A required, specifically:
- Quantum mechanics postulates and Hilbert space formalism
- Angular momentum and spin (Month 15)
- Time-dependent perturbation theory (Month 17)
- Second quantization basics (Month 18)

---

## Connections to Research

### Active Research Areas
- Quantum supremacy demonstrations
- Variational quantum eigensolvers for chemistry
- Quantum machine learning
- Near-term (NISQ) algorithms

### Industry Applications
- Cryptography and quantum key distribution
- Optimization (logistics, finance)
- Drug discovery and materials science
- Quantum sensing

---

## Directory Structure

```
Semester_1B_Quantum_Information/
├── README.md
├── Month_19_Density_Matrices/
│   ├── README.md
│   ├── Week_73_Pure_vs_Mixed/
│   ├── Week_74_Composite_Systems/
│   ├── Week_75_Generalized_Measurements/
│   └── Week_76_Quantum_Dynamics/
├── Month_20_Entanglement_Theory/
│   ├── README.md
│   ├── Week_77_Entanglement_Basics/
│   ├── Week_78_Bell_Inequalities/
│   ├── Week_79_Entanglement_Measures/
│   └── Week_80_Entanglement_Applications/
├── Month_21_Quantum_Gates_Circuits/
│   ├── README.md
│   ├── Week_81_Single_Qubit_Gates/
│   ├── Week_82_Two_Qubit_Gates/
│   ├── Week_83_Universal_Gate_Sets/
│   └── Week_84_Circuit_Model/
├── Month_22_Quantum_Algorithms_I/
│   ├── README.md
│   ├── Week_85_Query_Complexity/
│   ├── Week_86_Simons_Algorithm/
│   ├── Week_87_Quantum_Fourier_Transform/
│   └── Week_88_Phase_Estimation/
├── Month_23_Quantum_Algorithms_II/
│   ├── README.md
│   ├── Week_89_Shor_Number_Theory/
│   ├── Week_90_Shor_Implementation/
│   ├── Week_91_Grover_Search/
│   └── Week_92_Variational_Algorithms/
└── Month_24_Quantum_Channels_Error_Intro/
    ├── README.md
    ├── Week_93_Quantum_Channels/
    ├── Week_94_Open_Quantum_Systems/
    ├── Week_95_Year1_Integration/
    └── Week_96_Year1_Capstone/
```

---

## References

### Primary Texts
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information* (10th Anniversary Edition). Cambridge University Press.
- Preskill, J. (2015). *Lecture Notes for Physics 219/Computer Science 219: Quantum Computation*. Caltech.

### Supplementary
- Wilde, M. M. (2017). *Quantum Information Theory* (2nd Edition). Cambridge University Press.
- Watrous, J. (2018). *The Theory of Quantum Information*. Cambridge University Press.

### Online Resources
- IBM Quantum Learning: https://learning.quantum.ibm.com/
- Qiskit Textbook: https://qiskit.org/textbook
- Preskill's Ph219 notes: http://theory.caltech.edu/~preskill/ph219/

---

**Start Date:** Day 505
**End Date:** Day 672
**Duration:** 168 days (6 months)

---

*Next: Month 19 — Density Matrices & Mixed States*
