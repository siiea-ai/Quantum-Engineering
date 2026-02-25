# Month 22: Quantum Algorithms I

## Overview
**Days 589-616** | Weeks 85-88 | Year 1, Semester 1B

This month covers the foundational quantum algorithms that demonstrate quantum advantage over classical computation. We progress from the pedagogically important Deutsch-Jozsa algorithm through to Shor's groundbreaking factoring algorithm, which threatens modern cryptography.

---

## Learning Objectives

By the end of this month, you will be able to:

1. **Understand quantum oracles** and the query complexity model
2. **Implement Deutsch-Jozsa** and Bernstein-Vazirani algorithms
3. **Construct and analyze QFT circuits** with controlled rotations
4. **Apply quantum phase estimation** to extract eigenvalues
5. **Derive Shor's algorithm** from period-finding to factoring
6. **Analyze algorithm complexity** and success probabilities

---

## Month Structure

### Week 85: Deutsch-Jozsa Algorithm (Days 589-595)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 589 | Oracle Model and Query Complexity | Black-box model, query lower bounds |
| 590 | Deutsch's Algorithm | Single-qubit oracle, quantum parallelism |
| 591 | Deutsch-Jozsa n-Qubit Generalization | Constant vs balanced, exponential speedup |
| 592 | Bernstein-Vazirani Algorithm | Hidden string problem, dot product oracle |
| 593 | Simon's Algorithm Introduction | Period-finding precursor, exponential separation |
| 594 | Quantum Oracle Construction | Implementing oracles from functions |
| 595 | Week Review | Problem solving, algorithm comparison |

### Week 86: Quantum Fourier Transform (Days 596-602)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 596 | Classical DFT Review | Discrete Fourier transform, FFT |
| 597 | QFT Definition and Properties | Quantum analog, computational basis |
| 598 | QFT Circuit Construction | Hadamard, controlled rotations, SWAP |
| 599 | Phase Kickback and QFT | Eigenvalue encoding, phase estimation prep |
| 600 | Inverse QFT | Reversing the transform, applications |
| 601 | QFT Applications | Quantum arithmetic, phase detection |
| 602 | Week Review | Circuit optimization, complexity |

### Week 87: Phase Estimation (Days 603-609)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 603 | Eigenvalue Problem in QC | Unitary eigenvalues, phase extraction |
| 604 | QPE Circuit Design | Control register, controlled-U operations |
| 605 | QPE Analysis and Precision | Bit precision, register size |
| 606 | Success Probability | Error analysis, failure cases |
| 607 | Iterative Phase Estimation | Single ancilla methods, Kitaev approach |
| 608 | Kitaev's Algorithm | Robust phase estimation, fault tolerance |
| 609 | Week Review | QPE variations, applications |

### Week 88: Shor's Algorithm (Days 610-616)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 610 | Factoring to Order-Finding | Reduction proof, modular exponentiation |
| 611 | Number Theory Background | Chinese remainder theorem, Euler's theorem |
| 612 | Quantum Period-Finding | QPE for modular exponentiation |
| 613 | Continued Fractions | Classical post-processing, convergents |
| 614 | Full Shor's Algorithm | Complete circuit, success analysis |
| 615 | Complexity Analysis | Gate count, qubit requirements |
| 616 | Month Review | Comprehensive assessment |

---

## Key Formulas

### Oracle and Query Model
$$U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$$
$$\text{Phase oracle: } O_f|x\rangle = (-1)^{f(x)}|x\rangle$$

### Quantum Fourier Transform
$$QFT|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle$$
$$QFT = \frac{1}{\sqrt{N}}\sum_{j,k=0}^{N-1} e^{2\pi ijk/N}|k\rangle\langle j|$$

### Phase Estimation
$$U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle \Rightarrow QPE \text{ outputs } |\tilde{\phi}\rangle$$
$$\text{Precision: } |\phi - \tilde{\phi}| < \frac{1}{2^n}$$

### Shor's Algorithm
$$\text{Period } r: a^r \equiv 1 \pmod{N}$$
$$\gcd(a^{r/2} \pm 1, N) \rightarrow \text{factors}$$

---

## Quantum Speedups Summary

| Algorithm | Classical | Quantum | Speedup |
|-----------|-----------|---------|---------|
| Deutsch-Jozsa | $O(2^{n-1}+1)$ | $O(1)$ | Exponential |
| Bernstein-Vazirani | $O(n)$ | $O(1)$ | Linear to constant |
| Simon | $O(2^{n/2})$ | $O(n)$ | Exponential |
| Period-finding | $O(\sqrt{N})$ | $O((\log N)^3)$ | Super-polynomial |
| Factoring (Shor) | $O(\exp(n^{1/3}))$ | $O(n^3)$ | Super-polynomial |

---

## Primary References

### Textbooks
- Nielsen & Chuang, Ch. 5-6 (Quantum algorithms)
- Kaye, Laflamme, Mosca, Ch. 5-8 (Detailed derivations)
- Mermin, "Quantum Computer Science" Ch. 3-4

### Key Papers
- Deutsch (1985): Quantum theory of computation
- Deutsch & Jozsa (1992): Rapid solution of problems
- Simon (1994): Exponential separation
- Shor (1994): Algorithms for quantum computation
- Kitaev (1995): Quantum measurements and Abelian stabilizer problem

---

## Computational Tools

```python
# Core imports for Month 22
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Quantum gates
def hadamard():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def controlled_phase(k):
    """Controlled R_k gate for QFT"""
    angle = 2 * np.pi / (2**k)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, np.exp(1j * angle)]])

def qft_matrix(n):
    """QFT matrix for n qubits"""
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for k in range(N)]
                     for j in range(N)]) / np.sqrt(N)

def phase_estimation_state(phi, n_bits):
    """Final state of QPE for phase phi with n ancilla bits"""
    N = 2**n_bits
    state = np.zeros(N, dtype=complex)
    for k in range(N):
        state[k] = np.exp(2j * np.pi * phi * k)
    return state / np.sqrt(N)
```

---

## Progress Tracking

| Week | Status | Days Complete |
|------|--------|---------------|
| Week 85: Deutsch-Jozsa | ✅ Complete | 7/7 |
| Week 86: Quantum Fourier | ✅ Complete | 7/7 |
| Week 87: Phase Estimation | ✅ Complete | 7/7 |
| Week 88: Shor's Algorithm | ✅ Complete | 7/7 |

**Month Progress:** 28/28 days (100%) ✅

---

## Prerequisites

From Months 19-21:
- Density matrices and quantum states
- Entanglement and Bell states
- Quantum gates and circuits
- Tensor products and composite systems
- Unitary evolution

---

*Next: Month 23 - Quantum Algorithms II (Grover, Variational)*
