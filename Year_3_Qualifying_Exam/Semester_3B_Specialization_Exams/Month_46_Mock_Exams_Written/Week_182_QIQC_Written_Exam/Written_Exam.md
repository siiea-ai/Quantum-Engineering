# Quantum Information & Computing Written Qualifying Exam

## Exam Information

**Duration:** 3 hours (180 minutes)
**Total Points:** 200
**Number of Problems:** 8
**Passing Score:** 160 points (80%)

---

## Instructions

1. **Time Management:** Budget approximately 20-22 minutes per problem.

2. **Show All Work:** Partial credit is awarded for correct reasoning and setup.

3. **Notation:** Use standard quantum information notation. $|0\rangle$, $|1\rangle$ for computational basis; $\rho$ for density matrices; $\mathcal{E}$ for channels.

4. **Circuit Diagrams:** Draw circuits clearly with proper gate symbols.

5. **Proofs:** State clearly what you are proving and structure your argument logically.

6. **No External Resources:** Closed-book exam.

---

## Useful Definitions

**Pauli Matrices:**
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Common Gates:**
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \quad T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

**Bell States:**
$$|\Phi^\pm\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle), \quad |\Psi^\pm\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)$$

---

## Problem 1: Density Matrices and Quantum States (25 points)

Consider the three-qubit state:
$$|\psi\rangle = \frac{1}{2}|000\rangle + \frac{1}{2}|011\rangle + \frac{1}{2}|101\rangle + \frac{1}{2}|110\rangle$$

**(a)** (6 points) Compute the reduced density matrix $\rho_{AB}$ by tracing out qubit C.

**(b)** (6 points) Compute the reduced density matrix $\rho_A$ by tracing out qubits B and C. Is qubit A in a pure or mixed state?

**(c)** (7 points) Calculate the von Neumann entropy $S(\rho_A)$. What does this tell you about the entanglement structure of $|\psi\rangle$?

**(d)** (6 points) Is $|\psi\rangle$ a GHZ-type state or a W-type state? Justify your answer by considering what happens when one qubit is traced out or measured.

---

## Problem 2: Entanglement and Bell Inequalities (25 points)

Alice and Bob share the two-qubit state:
$$|\psi\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$$

where $0 < \theta < \pi/2$.

**(a)** (6 points) Calculate the concurrence $C(\psi)$ of this state. For what value of $\theta$ is entanglement maximized?

**(b)** (8 points) Alice measures in the basis $\{|0\rangle, |1\rangle\}$ or $\{|+\rangle, |-\rangle\}$, and Bob measures in the bases rotated by $\pm\pi/8$ from the $Z$-axis. Calculate the CHSH parameter:
$$S = \langle A_1 B_1 \rangle + \langle A_1 B_2 \rangle + \langle A_2 B_1 \rangle - \langle A_2 B_2 \rangle$$
for the case $\theta = \pi/4$. Compare to the classical bound.

**(c)** (5 points) Prove that for separable states $\rho_{sep} = \sum_i p_i \rho_A^{(i)} \otimes \rho_B^{(i)}$, the CHSH inequality $|S| \leq 2$ must hold.

**(d)** (6 points) If $\theta = \pi/6$, is the state still useful for violating the CHSH inequality? Calculate $S_{max}$ and compare to the threshold.

---

## Problem 3: Quantum Gates and Circuits (25 points)

**(a)** (8 points) Prove that CNOT, together with all single-qubit gates, forms a universal gate set. Specifically, show how to construct an arbitrary two-qubit unitary using these gates.

**(b)** (7 points) The SWAP gate exchanges two qubits: SWAP$|ab\rangle = |ba\rangle$. Express SWAP as a product of three CNOT gates. Draw the circuit.

**(c)** (5 points) The $\sqrt{\text{SWAP}}$ gate satisfies $(\sqrt{\text{SWAP}})^2 = \text{SWAP}$. Write the matrix representation of $\sqrt{\text{SWAP}}$ and show it is entangling.

**(d)** (5 points) Design a circuit using only $H$, $T$, and CNOT gates that approximates the rotation $R_z(\pi/8)$ to first order. How many $T$ gates are needed?

---

## Problem 4: Quantum Algorithms - Oracle Problems (25 points)

Consider the following promise problem: You are given oracle access to a function $f: \{0,1\}^n \to \{0,1\}$ with the promise that either:
- $f$ is constant (outputs 0 for all inputs, or 1 for all inputs), OR
- $f$ is balanced (outputs 0 for exactly half of inputs, 1 for the other half)

**(a)** (6 points) How many classical queries are needed to determine which case holds with certainty? Justify your answer.

**(b)** (8 points) Describe the Deutsch-Jozsa algorithm. Draw the circuit and explain each step.

**(c)** (6 points) Prove that after applying the algorithm, measuring all zeros indicates $f$ is constant, while any other outcome indicates $f$ is balanced.

**(d)** (5 points) Simon's algorithm solves a related problem with exponential speedup. State Simon's problem and explain why it cannot be efficiently solved classically (you may assume standard complexity-theoretic conjectures).

---

## Problem 5: Quantum Algorithms - Shor and Grover (25 points)

**(a)** (8 points) Explain the reduction from factoring to order-finding. Given an algorithm that finds the order $r$ of $a \mod N$ (where $a^r \equiv 1 \mod N$), show how to factor $N$ with high probability.

**(b)** (7 points) In Shor's algorithm, the quantum Fourier transform is applied to a state of the form:
$$|\psi\rangle = \frac{1}{\sqrt{m}}\sum_{j=0}^{m-1}|x_0 + jr\rangle$$
where the period is $r$. What is the output state after QFT? How does one extract $r$?

**(c)** (5 points) Grover's algorithm achieves a quadratic speedup for unstructured search. If the database has $N = 2^{20}$ items with one marked item, how many Grover iterations are needed? What is the success probability after this many iterations?

**(d)** (5 points) Prove that Grover's algorithm is optimal: any quantum algorithm for unstructured search requires $\Omega(\sqrt{N})$ queries.

---

## Problem 6: Quantum Channels and Noise (25 points)

**(a)** (7 points) The amplitude damping channel models energy relaxation with Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

Apply this channel to the state $\rho = |\psi\rangle\langle\psi|$ where $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$. Express the output in terms of $\alpha$, $\beta$, and $\gamma$.

**(b)** (6 points) What is the fixed point of repeated applications of the amplitude damping channel? Interpret this physically.

**(c)** (6 points) The dephasing channel has Kraus operators:
$$K_0 = \sqrt{1-p/2}\, I, \quad K_1 = \sqrt{p/2}\, Z$$

Show that this channel can be written as $\mathcal{E}(\rho) = (1-p)\rho + p Z\rho Z$. What is the effect on off-diagonal elements?

**(d)** (6 points) Consider two channels applied in sequence: amplitude damping with $\gamma = 0.1$ followed by depolarizing with $p = 0.1$. Is the combined channel equivalent to a single channel of either type? Explain why or why not.

---

## Problem 7: Quantum Complexity (25 points)

**(a)** (6 points) Define the complexity class BQP (Bounded-error Quantum Polynomial time). State the completeness and soundness conditions.

**(b)** (6 points) Show that BPP $\subseteq$ BQP. That is, any problem solvable by a probabilistic classical computer can be solved by a quantum computer.

**(c)** (7 points) The class QMA (Quantum Merlin-Arthur) is the quantum analog of NP. Define QMA and explain the role of the quantum proof (witness). Give an example of a QMA-complete problem.

**(d)** (6 points) Explain why it is believed that NP $\not\subseteq$ BQP (i.e., quantum computers cannot efficiently solve all NP problems). Reference the oracle separation result.

---

## Problem 8: Quantum Protocols (25 points)

**(a)** (8 points) Describe the quantum teleportation protocol. Alice has an unknown state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$, and she shares a Bell pair $|\Phi^+\rangle$ with Bob.

i. What measurements does Alice perform?
ii. What are the four possible outcomes and corresponding states at Bob's location?
iii. How does Bob recover $|\psi\rangle$?

**(b)** (6 points) Superdense coding allows two classical bits to be transmitted using one qubit. Describe the protocol and prove that two bits are communicated.

**(c)** (6 points) In the BB84 quantum key distribution protocol, Alice and Bob can detect an eavesdropper Eve. If Eve intercepts and measures in the wrong basis 25% of the time, what is the expected error rate that Alice and Bob observe? How do they use this to establish security?

**(d)** (5 points) Explain the no-cloning theorem and why it is essential for quantum cryptography. Provide a brief proof that an arbitrary unknown quantum state cannot be cloned.

---

## End of Exam

**Checklist before submitting:**
- [ ] All problems attempted
- [ ] Circuit diagrams clearly drawn
- [ ] Proofs logically structured
- [ ] Calculations shown

**Good luck!**

---

*This exam covers material from Nielsen & Chuang and Preskill's lecture notes, representative of graduate quantum information coursework.*
