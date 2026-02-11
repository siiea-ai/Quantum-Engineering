# Week 162: Quantum Algorithms I - Problem Set

## 28 Problems on Foundational Quantum Algorithms

---

## Section A: Deutsch-Jozsa and Bernstein-Vazirani (Problems 1-8)

### Problem 1: Deutsch's Algorithm (Single Qubit)
Consider the single-qubit version with $f:\{0,1\} \to \{0,1\}$.

a) List all four possible functions and classify each as constant or balanced.

b) Write the circuit for Deutsch's algorithm.

c) Compute the final state for each of the four functions.

d) Show that measuring gives $|0\rangle$ for constant functions and $|1\rangle$ for balanced functions.

---

### Problem 2: Deutsch-Jozsa State Analysis
For the $n$-qubit Deutsch-Jozsa algorithm:

a) Show that after applying Hadamards to $|0\rangle^{\otimes n}|1\rangle$, the state is:
$$\frac{1}{\sqrt{2^{n+1}}}\sum_{x \in \{0,1\}^n}|x\rangle(|0\rangle - |1\rangle)$$

b) Prove that applying the oracle $O_f$ gives:
$$\frac{1}{\sqrt{2^n}}\sum_x (-1)^{f(x)}|x\rangle|{-}\rangle$$

c) Compute the amplitude of $|0\rangle^{\otimes n}$ after the final Hadamard layer.

---

### Problem 3: Specific Oracle Implementation
For $n = 3$, consider the function:
$$f(x_1, x_2, x_3) = x_1 \oplus x_2$$

a) Is this function constant or balanced? Prove your answer.

b) Design a quantum circuit implementing the oracle $O_f$.

c) Trace through the Deutsch-Jozsa algorithm with this oracle.

d) Verify that the algorithm correctly identifies the function type.

---

### Problem 4: Bernstein-Vazirani with Specific Secret
For the Bernstein-Vazirani problem with $n = 4$ and secret string $s = 1011$:

a) Write out the function $f(x) = s \cdot x$ for all 16 inputs.

b) Implement the oracle circuit.

c) Show that the algorithm outputs $|1011\rangle$ with probability 1.

d) How would a classical algorithm determine $s$?

---

### Problem 5: Oracle Construction
Given the balanced function on 2 qubits:
$$f(00) = 0, \quad f(01) = 1, \quad f(10) = 1, \quad f(11) = 0$$

a) Construct a quantum circuit for $O_f$ using only CNOT and single-qubit gates.

b) Is your circuit unique? Find an alternative implementation.

c) What is the minimum CNOT count for this oracle?

---

### Problem 6: Generalized Deutsch-Jozsa
Consider a function $f:\{0,1\}^n \to \{0,1\}$ that is promised to be either:
- Constant, OR
- "$k$-skewed": outputs 0 for exactly $k$ inputs, where $k \neq 0, 2^n$

a) Can the Deutsch-Jozsa algorithm distinguish these cases in general?

b) For what values of $k$ does the algorithm always succeed?

c) Calculate the probability of measuring $|0\rangle^{\otimes n}$ when $f$ is $k$-skewed.

---

### Problem 7: Error Analysis
Suppose the Hadamard gates in Deutsch-Jozsa have small errors, modeled as:
$$\tilde{H} = H \cdot e^{i\epsilon Z}$$
where $\epsilon \ll 1$.

a) How does this affect the algorithm for constant functions?

b) How does this affect the algorithm for balanced functions?

c) Estimate the maximum tolerable error $\epsilon$ for 99% success rate.

---

### Problem 8: Hidden Subgroup Connection
The Deutsch-Jozsa problem can be viewed as an instance of the Hidden Subgroup Problem.

a) Identify the group $G$ and the possible hidden subgroups.

b) How does the standard HSP algorithm reduce to Deutsch-Jozsa in this case?

c) Generalize: what hidden subgroups correspond to "$k$-skewed" functions?

---

## Section B: Simon's Algorithm (Problems 9-14)

### Problem 9: Simon's Algorithm Basics
For $n = 3$ with hidden period $s = 101$:

a) Give an example function $f:\{0,1\}^3 \to \{0,1\}^3$ satisfying Simon's promise.

b) List all pairs $(x, x \oplus s)$ that map to the same output.

c) What linear equations will the algorithm generate?

d) How many iterations are needed to determine $s$ with high probability?

---

### Problem 10: State Analysis in Simon's Algorithm
In Simon's algorithm, after the oracle query on input state $\frac{1}{\sqrt{2^n}}\sum_x |x\rangle|0\rangle$:

a) Write the state immediately after the oracle.

b) If the second register is measured and collapses to $|f(x_0)\rangle$, what is the state of the first register?

c) Apply Hadamard to the first register and show that measuring gives $y$ with $y \cdot s = 0$.

d) Why is $y$ uniformly distributed over $\{y : y \cdot s = 0\}$?

---

### Problem 11: Linear Algebra in Simon's
After running Simon's algorithm $m$ times, we obtain vectors $y_1, \ldots, y_m$ with $y_i \cdot s = 0$.

a) Prove that with probability at least $1 - 2^{-m+n}$, these vectors span an $(n-1)$-dimensional subspace.

b) How do we find $s$ from this subspace?

c) If we get a redundant measurement (one in the span of previous ones), what should we do?

d) What is the expected number of iterations to find $s$?

---

### Problem 12: Simon vs. Classical
Prove the classical lower bound for Simon's problem:

a) Show that any classical algorithm must query $f$ at least $\Omega(\sqrt{2^n})$ times.

b) Explain the birthday paradox connection.

c) Why doesn't this attack work for the quantum algorithm?

---

### Problem 13: Simon's Algorithm Implementation
For $n = 2$ and $s = 11$:

a) Define a valid function $f:\{0,1\}^2 \to \{0,1\}^2$.

b) Design the oracle circuit.

c) Trace through one complete iteration of the algorithm.

d) Show that with 2 iterations, $s$ can be determined (assuming lucky measurements).

---

### Problem 14: Generalized Simon's Problem
Consider a generalization where $f(x) = f(y)$ iff $x \oplus y \in H$ for some subgroup $H \leq (\mathbb{Z}_2)^n$.

a) How does the algorithm change for $|H| = 4$ (two generators)?

b) What linear system must be solved?

c) Estimate the query complexity for hidden subgroup of size $2^k$.

---

## Section C: Quantum Fourier Transform (Problems 15-20)

### Problem 15: QFT Computation
For $n = 2$ (so $N = 4$):

a) Write out the full QFT matrix.

b) Compute $\text{QFT}|0\rangle$, $\text{QFT}|1\rangle$, $\text{QFT}|2\rangle$, $\text{QFT}|3\rangle$.

c) Verify that QFT is unitary.

d) Find $\text{QFT}^{-1}$.

---

### Problem 16: QFT Circuit Derivation
Derive the QFT circuit for $n = 3$ qubits.

a) Start from the product representation:
$$\text{QFT}|j\rangle = \frac{1}{\sqrt{8}}\bigotimes_{l=1}^{3}(|0\rangle + e^{2\pi i \cdot 0.j_{4-l}\cdots j_3}|1\rangle)$$

b) Identify which gates are needed for each tensor factor.

c) Draw the complete circuit (before bit-reversal).

d) Count the total number of gates.

---

### Problem 17: Controlled Phase Gates
The QFT uses controlled-$R_k$ gates where $R_k = \text{diag}(1, e^{2\pi i/2^k})$.

a) Express $R_2$ in terms of the $T$ gate.

b) Express controlled-$R_3$ in terms of controlled-$T$ and other gates.

c) For fault-tolerant implementation, how many $T$ gates are needed for controlled-$R_k$?

d) Estimate the total $T$-count for exact $n$-qubit QFT.

---

### Problem 18: Approximate QFT
The approximate QFT omits controlled-$R_k$ for $k > m$.

a) For $n = 10$ and $m = 5$, how many gates are saved?

b) Bound the error $\|\text{QFT} - \text{QFT}_{\text{approx}}\|$ in operator norm.

c) For error tolerance $\epsilon = 0.01$, what value of $m$ suffices?

d) Give the gate complexity of approximate QFT as a function of $n$ and $\epsilon$.

---

### Problem 19: Semiclassical QFT
In the semiclassical QFT, measurements are made as we go, and classical feedforward controls subsequent gates.

a) Explain how this reduces the required ancilla qubits.

b) Draw the semiclassical circuit for $n = 3$.

c) What are the advantages for NISQ devices?

d) Are there any disadvantages compared to coherent QFT?

---

### Problem 20: QFT Applications
The QFT is used in many algorithms beyond phase estimation.

a) Explain how QFT enables period finding (without giving full Shor's algorithm).

b) How is QFT used in quantum simulation (brief explanation)?

c) What is the relationship between QFT and the quantum walk?

---

## Section D: Phase Estimation (Problems 21-28)

### Problem 21: Phase Estimation Basics
Consider $U = R_z(\pi/4)$ with eigenstate $|0\rangle$ and eigenvalue $e^{-i\pi/8}$.

a) What is the phase $\theta$?

b) How many ancilla qubits are needed to estimate $\theta$ to 4 bits?

c) Trace through the phase estimation algorithm.

d) What measurement outcomes are possible?

---

### Problem 22: Controlled-U Implementation
For phase estimation with $U = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$:

a) Implement controlled-$U$ in terms of standard gates.

b) Implement controlled-$U^2$ without using two controlled-$U$ gates.

c) Generally, implement controlled-$U^{2^k}$ efficiently.

d) What is the total gate count for $t$-bit phase estimation?

---

### Problem 23: Non-Eigenstate Input
Phase estimation is typically described for eigenstates, but what if we input a superposition?

a) If $|u\rangle = \alpha|u_1\rangle + \beta|u_2\rangle$ where $U|u_j\rangle = e^{2\pi i\theta_j}|u_j\rangle$, what happens?

b) Describe the measurement statistics.

c) How can this be used to sample from the spectrum of $U$?

d) What are the implications for ground state energy estimation?

---

### Problem 24: Precision Analysis
For phase estimation with $t$ ancilla qubits:

a) If $\theta = 0.j_1j_2\ldots j_t$ exactly (binary), show that measurement gives $j_1\ldots j_t$ with probability 1.

b) If $2^t\theta = m + \delta$ with $|\delta| < 1/2$, bound the probability of measuring $m$.

c) Prove that $P(|m - 2^t\theta| \leq 1) \geq 8/\pi^2 \approx 0.81$.

d) How does adding extra ancilla qubits increase success probability?

---

### Problem 25: Iterative Phase Estimation
Iterative (Kitaev) phase estimation uses only one ancilla qubit, reused multiple times.

a) Describe the algorithm for estimating $\theta$ to $t$ bits.

b) What is the circuit depth compared to standard phase estimation?

c) What are the error accumulation concerns?

d) When would you prefer iterative over standard phase estimation?

---

### Problem 26: Hamiltonian Eigenvalue Estimation
For a Hamiltonian $H$ with ground state $|\psi_0\rangle$ and energy $E_0$:

a) How is $U = e^{-iHt}$ used in phase estimation?

b) Relate the measured phase to $E_0$.

c) What time $t$ gives good resolution for estimating $E_0$?

d) Discuss the challenge of preparing $|\psi_0\rangle$.

---

### Problem 27: Complete Algorithm Design
Design a complete algorithm to find the smallest eigenvalue of the $2 \times 2$ matrix:
$$H = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 2 \end{pmatrix}$$

a) Find the eigenvalues and eigenvectors of $H$.

b) Set up the phase estimation circuit for $U = e^{-iH}$.

c) Determine the number of ancilla qubits for 3-bit precision.

d) Describe the expected measurement outcomes.

---

### Problem 28: Research Connection - VQE vs. QPE
Compare Variational Quantum Eigensolver (VQE) with Quantum Phase Estimation for eigenvalue problems:

a) List the key differences in approach.

b) Which requires more qubits? More circuit depth?

c) Which is more suitable for near-term quantum devices? Why?

d) Can results from VQE be used to initialize QPE? Explain.

---

## Problem Difficulty Summary

| Difficulty | Problems | Focus |
|------------|----------|-------|
| Foundational | 1, 2, 9, 15, 21 | Basic algorithm understanding |
| Intermediate | 3-5, 10-11, 16-17, 22-23 | Derivations and implementations |
| Advanced | 6-8, 12-14, 18-20, 24-26 | Deep analysis and proofs |
| Research-Level | 27, 28 | Synthesis and connections |

---

*Complete solutions are available in Problem_Solutions.md*
