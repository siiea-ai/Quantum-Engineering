# Week 86: Quantum Fourier Transform

## Overview

**Days 596-602** | Week 86 | Month 22 | Year 1

This week we study the Quantum Fourier Transform (QFT), one of the most important subroutines in quantum computing. The QFT is the quantum analog of the classical Discrete Fourier Transform and provides exponential speedup when implemented on a quantum computer. It forms the foundation for phase estimation and Shor's factoring algorithm.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Understand the classical Discrete Fourier Transform and its properties
2. Define the Quantum Fourier Transform mathematically
3. Construct efficient QFT circuits using controlled rotations
4. Analyze the phase kickback mechanism in QFT applications
5. Implement and verify the inverse QFT
6. Apply QFT to quantum arithmetic and signal processing

---

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 596 | Classical DFT Review | DFT definition, FFT algorithm, complexity |
| 597 | QFT Definition and Properties | Quantum Fourier basis, unitary matrix |
| 598 | QFT Circuit Construction | Hadamard, controlled-R gates, SWAP |
| 599 | Phase Kickback and QFT | Eigenvalue encoding, QPE connection |
| 600 | Inverse QFT | Reversing the transform, circuit structure |
| 601 | QFT Applications | Quantum arithmetic, phase detection |
| 602 | Week Review | Circuit optimization, comparison |

---

## Key Concepts

### Classical vs Quantum Fourier Transform

**Classical DFT** on $N$ points requires $O(N^2)$ operations naive, $O(N \log N)$ with FFT.

**Quantum QFT** on $n$ qubits ($N = 2^n$ points) requires only $O(n^2)$ gates!

This is exponentially faster: $O((\log N)^2)$ vs $O(N \log N)$.

### QFT Definition

The QFT maps computational basis states to Fourier basis states:

$$QFT|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle$$

where $N = 2^n$ and $\omega = e^{2\pi i/N}$ is the primitive $N$th root of unity.

### Product Representation

The QFT output can be written as a tensor product:

$$QFT|j_1 j_2 \cdots j_n\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{l=1}^{n} \left(|0\rangle + e^{2\pi i \cdot 0.j_{n-l+1}\cdots j_n}|1\rangle\right)$$

This structure enables efficient circuit construction.

---

## Key Formulas

### QFT Matrix

$$QFT_N = \frac{1}{\sqrt{N}}\begin{pmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^2 & \cdots & \omega^{N-1} \\
1 & \omega^2 & \omega^4 & \cdots & \omega^{2(N-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{N-1} & \omega^{2(N-1)} & \cdots & \omega^{(N-1)^2}
\end{pmatrix}$$

### Controlled Rotation Gates

$$R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i/2^k} \end{pmatrix}$$

The QFT circuit uses controlled-$R_k$ gates for $k = 2, 3, \ldots, n$.

### Binary Fraction Notation

$$0.j_1 j_2 \cdots j_m = \frac{j_1}{2} + \frac{j_2}{4} + \cdots + \frac{j_m}{2^m}$$

---

## Prerequisites

From Week 85:
- Hadamard transform
- Quantum oracles and phase kickback
- Multi-qubit gates and circuits
- Complex exponentials and roots of unity

---

## References

- Nielsen & Chuang, Section 5.1 (QFT)
- Kaye, Laflamme, Mosca, Chapter 6
- Cleve et al. (1998), "Quantum algorithms revisited"

---

*Week 86 of 88 in Month 22*
