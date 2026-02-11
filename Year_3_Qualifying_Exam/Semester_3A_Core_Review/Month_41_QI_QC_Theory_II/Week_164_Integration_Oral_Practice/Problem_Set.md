# Week 164: Integration & Oral Practice - Problem Set

## 25 Synthesis Problems Integrating All Month 41 Material

---

## Section A: Variational Algorithms (Problems 1-8)

### Problem 1: VQE for Simple Hamiltonian
Consider the Hamiltonian $H = Z_1 + Z_2 + 0.5 X_1 X_2$.

a) Decompose $H$ into Pauli strings with coefficients.

b) Design a hardware-efficient ansatz with 2 qubits and 1 layer.

c) How many measurements are needed to estimate $\langle H \rangle$ to precision 0.01?

d) If the ground state energy is approximately $-2.1$, design an experiment to verify this.

---

### Problem 2: QAOA for Max-Cut
Consider the triangle graph (3 vertices, 3 edges).

a) Write the cost Hamiltonian $H_C$ for Max-Cut.

b) Construct the QAOA circuit for $p = 1$.

c) Find the optimal $\gamma, \beta$ analytically or numerically.

d) What is the maximum cut value and the QAOA approximation ratio?

---

### Problem 3: Ansatz Expressibility
For an $n$-qubit system:

a) How many parameters does a hardware-efficient ansatz with $L$ layers have?

b) Compare this to the dimension of the full Hilbert space.

c) Discuss the trade-off between expressibility and trainability.

d) What is the "barren plateau" problem and how does it affect VQE?

---

### Problem 4: Measurement Optimization
For VQE with Hamiltonian $H = \sum_i c_i P_i$ containing $M$ Pauli strings:

a) What is the naive measurement cost (shots) for precision $\epsilon$?

b) Explain how grouping commuting terms reduces measurements.

c) For $H = Z_1 Z_2 + X_1 + X_2 + Z_1 + Z_2$, find optimal grouping.

d) Estimate the measurement reduction factor.

---

### Problem 5: QAOA Depth Analysis
For QAOA with $p$ layers on $n$-qubit Max-Cut:

a) What is the circuit depth in terms of $p$ and graph properties?

b) How does the approximation ratio typically improve with $p$?

c) At what $p$ do barren plateaus become problematic?

d) Discuss the trade-off between circuit depth and solution quality.

---

### Problem 6: Noise Effects on VQE
Consider VQE under depolarizing noise with error rate $\epsilon$ per gate.

a) How does noise affect the measured energy?

b) Explain zero-noise extrapolation (ZNE).

c) When does noise prevent finding the ground state?

d) Design an error mitigation strategy for a simple VQE.

---

### Problem 7: VQE vs. QPE Comparison
Compare VQE and Quantum Phase Estimation for finding ground state energy.

a) List the resource requirements of each.

b) Which is more suitable for near-term devices? Why?

c) Can they be combined? How?

d) What are the precision limits of each approach?

---

### Problem 8: QAOA for Weighted Max-Cut
Extend QAOA to weighted graphs where edge $(i,j)$ has weight $w_{ij}$.

a) Write the modified cost Hamiltonian.

b) How does the circuit change?

c) For a triangle with weights $w_{12} = 1, w_{23} = 2, w_{13} = 3$, find the optimal cut.

d) Design QAOA circuit and estimate required $p$.

---

## Section B: Algorithm Synthesis (Problems 9-16)

### Problem 9: Grover + Phase Estimation
Design an algorithm that finds the minimum value of a function $f:\{0,1\}^n \to \{0,1\}^m$.

a) How can Grover help find the minimum?

b) Design a complete algorithm using Grover's search.

c) Analyze the query complexity.

d) Compare to classical approaches.

---

### Problem 10: Complete Algorithm Analysis
For each algorithm below, fill in the table:

| Algorithm | Speedup | Key Technique | Qubits | Gates | NISQ? |
|-----------|---------|---------------|--------|-------|-------|
| Deutsch-Jozsa | | | | | |
| Simon | | | | | |
| Shor | | | | | |
| Grover | | | | | |
| VQE | | | | | |
| QAOA | | | | | |

---

### Problem 11: Algorithm Design Challenge
**Problem:** Given a black-box function $f:\{0,1\}^n \to \{0,1\}^n$ that is 2-to-1, find any collision (i.e., $x \neq y$ with $f(x) = f(y)$).

a) What is the classical query complexity?

b) Design a quantum algorithm using available techniques.

c) Analyze your algorithm's query complexity.

d) Is your algorithm optimal? Justify.

---

### Problem 12: Circuit Compilation
Given the unitary $U = e^{-i\pi Z_1 Z_2/8}$:

a) Express $U$ in terms of CNOT and single-qubit gates.

b) Compile to the gate set $\{H, T, \text{CNOT}\}$.

c) Estimate the T-count.

d) Can you reduce the T-count using ancilla qubits?

---

### Problem 13: Hybrid Algorithm Design
Design a hybrid quantum-classical algorithm for the following problem:

**Given:** A weighted graph $G$ and integer $k$.
**Find:** A vertex cover of size at most $k$ (if one exists).

a) Encode the problem as a Hamiltonian.

b) Design a VQE/QAOA-based approach.

c) How do you check feasibility (size $\leq k$)?

d) Estimate quantum resources needed for $n = 10$ vertices.

---

### Problem 14: Error Analysis Across Algorithms
For each algorithm, analyze the effect of $\epsilon$-error in each gate:

a) Deutsch-Jozsa: How does error affect the constant/balanced distinction?

b) Grover: How does error accumulate over $O(\sqrt{N})$ iterations?

c) Shor: What precision is needed for reliable factoring?

d) Which algorithm is most/least sensitive to errors?

---

### Problem 15: Quantum Advantage Identification
For each problem, determine if quantum advantage is likely:

a) Sorting a list of $N$ numbers.

b) Finding a satisfying assignment to a 3-SAT formula.

c) Computing the permanent of a matrix.

d) Finding the ground state of a local Hamiltonian.

e) Simulating quantum systems.

Justify each answer with complexity arguments.

---

### Problem 16: Resource Estimation
For a practical application, estimate the quantum resources:

**Application:** Factor a 2048-bit RSA key.

a) Estimate qubits needed (logical and physical with error correction).

b) Estimate gate count and circuit depth.

c) Estimate run time at 1 MHz gate speed.

d) When might this be achievable? (Estimate year based on current trends.)

---

## Section C: Oral Practice Problems (Problems 17-22)

### Problem 17: Explain to Different Audiences
Prepare 3 explanations of Grover's algorithm for:

a) A physics PhD student unfamiliar with quantum computing (2 min).

b) A computer science professor who knows complexity theory (3 min).

c) A qualifying exam committee (5 min with derivations).

---

### Problem 18: Handle Follow-up Questions
For the prompt "Explain the Solovay-Kitaev theorem":

a) Give your main explanation.

b) Prepare answers for these follow-ups:
   - "What's the constant $c$ in the complexity?"
   - "Why can't we do better?"
   - "How is this used in practice?"
   - "What about multi-qubit gates?"

---

### Problem 19: Whiteboard Derivation
Practice deriving on a whiteboard (time yourself):

a) Grover's optimal iteration count (target: 3 minutes).

b) Shor's reduction from factoring to order-finding (target: 5 minutes).

c) Phase estimation success probability (target: 4 minutes).

---

### Problem 20: Compare and Contrast
Prepare answers comparing:

a) Clifford vs. non-Clifford gates (significance).

b) Phase estimation vs. VQE (for eigenvalue problems).

c) Query complexity vs. gate complexity.

d) Exponential vs. quadratic speedup (significance).

---

### Problem 21: Identify the Error
Find the error in each argument:

a) "Grover gives exponential speedup because $\sqrt{N} = 2^{n/2}$ for $N = 2^n$."

b) "CNOT is universal because it can create any entanglement."

c) "VQE finds the ground state because it always converges to the minimum."

d) "BBBV proves quantum computers can't solve NP problems."

---

### Problem 22: Research Directions
Discuss current research on:

a) Quantum advantage for optimization (QAOA limitations and improvements).

b) Error mitigation vs. error correction for VQE.

c) Quantum algorithms for machine learning.

d) Beyond-Shor factoring algorithms.

---

## Section D: Comprehensive Synthesis (Problems 23-25)

### Problem 23: Complete Algorithm Portfolio
You are given a problem: compute the eigenvalues of a $2^n \times 2^n$ Hermitian matrix $H$.

Design a complete solution:

a) For exact eigenvalues: Use QPE. Detail requirements.

b) For ground state only: Use VQE. Design ansatz.

c) For near-term hardware: Hybrid approach.

d) Compare resources and discuss trade-offs.

---

### Problem 24: Mock Qualifying Exam
Complete this 45-minute written exam (simulated):

**Part A (15 min):** Prove that $\{H, T, \text{CNOT}\}$ is universal for quantum computation.

**Part B (15 min):** Derive Grover's algorithm from first principles, including the optimal iteration count.

**Part C (15 min):** Explain Shor's algorithm, including the reduction to order-finding and the role of phase estimation.

---

### Problem 25: Month 41 Capstone
This comprehensive problem tests all month material:

**Scenario:** You are designing a quantum computer to solve optimization problems.

**Part A: Gates**
1. Choose a universal gate set for your hardware.
2. Analyze how many gates to approximate arbitrary rotations to $10^{-6}$ error.
3. Discuss T-gate overhead for fault tolerance.

**Part B: Algorithms**
4. Compare QAOA and Grover-based approaches for combinatorial optimization.
5. Analyze which provides better scaling.
6. Design a hybrid approach combining both.

**Part C: Implementation**
7. Estimate qubits for 100-variable Max-Cut.
8. Estimate circuit depth for $p = 5$ QAOA.
9. Discuss error mitigation strategies.

**Part D: Analysis**
10. Is quantum advantage achievable for this problem?
11. What are the main bottlenecks?
12. Propose a 5-year research roadmap.

---

## Problem Difficulty Summary

| Difficulty | Problems | Focus |
|------------|----------|-------|
| Foundational | 1-2, 10, 17 | Basic application |
| Intermediate | 3-8, 18-20 | Analysis and derivation |
| Advanced | 9, 11-16, 21-22 | Synthesis and proof |
| Qualifying Exam Level | 23-25 | Comprehensive |

---

*Solutions are available in Problem_Solutions.md*
