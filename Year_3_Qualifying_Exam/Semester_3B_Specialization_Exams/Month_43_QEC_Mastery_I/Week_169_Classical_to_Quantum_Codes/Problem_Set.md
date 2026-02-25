# Week 169: Classical to Quantum Codes - Problem Set

## Instructions

This problem set contains 30 problems at the PhD qualifying examination level. Problems are organized by topic and difficulty:
- **Level I:** Direct application of definitions and basic calculations
- **Level II:** Intermediate problems requiring multi-step reasoning
- **Level III:** Challenging problems requiring deep understanding or novel synthesis

Time estimate: 15-20 hours total

---

## Part A: Classical Codes Review (Problems 1-5)

### Problem 1 (Level I)
Consider the $$[7, 4, 3]$$ Hamming code with parity-check matrix:
$$H = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$

(a) Find the generator matrix $$G$$ in systematic form.
(b) Encode the message $$m = (1, 0, 1, 1)$$.
(c) If the received word is $$r = (1, 0, 1, 1, 0, 1, 1)$$, find the syndrome and identify the error.

### Problem 2 (Level I)
For a general $$[n, k, d]$$ linear code:

(a) Prove that the minimum distance $$d$$ equals the minimum number of linearly dependent columns in $$H$$.
(b) Use this to verify that the Hamming code has $$d = 3$$.

### Problem 3 (Level II)
The dual code $$C^\perp$$ of a code $$C$$ consists of all vectors orthogonal to every codeword in $$C$$.

(a) Show that if $$C$$ has generator matrix $$G$$, then $$C^\perp$$ has generator matrix $$H^T$$ (the transpose of the parity-check matrix of $$C$$).
(b) The Hamming code is $$[7, 4, 3]$$. What are the parameters of its dual?
(c) Is the dual of the Hamming code useful for error correction? Explain.

### Problem 4 (Level II)
Prove the Singleton bound for classical codes: For an $$[n, k, d]$$ code,
$$k \leq n - d + 1$$

*Hint:* Consider puncturing the code by deleting $$d-1$$ coordinates.

### Problem 5 (Level III)
The **covering radius** $$\rho$$ of a code $$C$$ is the maximum Hamming distance from any vector to the nearest codeword.

(a) Prove that $$\rho \geq d/2$$ for any code with minimum distance $$d$$.
(b) For the Hamming code, show that $$\rho = 1$$ (it is a "perfect" code).
(c) Explain why perfect codes are rare and list all known binary perfect codes.

---

## Part B: Quantum Errors (Problems 6-10)

### Problem 6 (Level I)
(a) Verify the commutation and anticommutation relations for Pauli matrices:
$$[X, Y] = 2iZ, \quad \{X, Y\} = 0$$

(b) Show that $$Y = iXZ = -iZX$$.

(c) Prove that any $$2 \times 2$$ matrix can be written as a linear combination of $$\{I, X, Y, Z\}$$.

### Problem 7 (Level I)
Consider the amplitude damping channel with Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

(a) Verify that $$K_0^\dagger K_0 + K_1^\dagger K_1 = I$$.
(b) Expand $$K_0$$ and $$K_1$$ in the Pauli basis.
(c) Explain why this channel cannot be corrected by a code that only corrects Pauli errors.

### Problem 8 (Level II)
For the depolarizing channel:
$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

(a) Rewrite this in Kraus operator form.
(b) Show that this is equivalent to applying $$I$$, $$X$$, $$Y$$, or $$Z$$ with probabilities $$(1-p)$$, $$p/3$$, $$p/3$$, $$p/3$$.
(c) For an $$n$$-qubit system with independent depolarizing noise on each qubit, what is the probability of having exactly $$k$$ errors?

### Problem 9 (Level II)
The **Pauli twirl** of a channel $$\mathcal{E}$$ is:
$$\mathcal{T}(\mathcal{E})(\rho) = \frac{1}{4}\sum_{P \in \{I,X,Y,Z\}} P \mathcal{E}(P\rho P) P$$

(a) Show that the Pauli twirl of any single-qubit channel is a Pauli channel (probabilistic application of Pauli operators).
(b) Compute the Pauli twirl of the amplitude damping channel.
(c) Explain the significance of this result for quantum error correction.

### Problem 10 (Level III)
Consider a general single-qubit error:
$$E = \alpha I + \beta X + \gamma Y + \delta Z$$

where $$|\alpha|^2 + |\beta|^2 + |\gamma|^2 + |\delta|^2 = 1$$.

(a) Show that if a code can correct the errors $$\{I, X, Y, Z\}$$, it can correct $$E$$.
(b) Generalize to $$n$$ qubits: if a code corrects all weight-$$t$$ Pauli errors, what other errors can it correct?
(c) This is the error discretization theorem. Write a rigorous proof using the Knill-Laflamme conditions.

---

## Part C: Knill-Laflamme Conditions (Problems 11-17)

### Problem 11 (Level I)
State the Knill-Laflamme conditions precisely. For each of the following, determine whether the condition is satisfied:

(a) Code: span$$\{|00\rangle, |11\rangle\}$$. Errors: $$\{I, X_1\}$$.
(b) Code: span$$\{|00\rangle, |11\rangle\}$$. Errors: $$\{I, Z_1\}$$.
(c) Code: span$$\{|00\rangle, |11\rangle\}$$. Errors: $$\{I, X_1, X_2, X_1X_2\}$$.

### Problem 12 (Level II)
Consider the two-qubit code with codewords:
$$|0_L\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle), \quad |1_L\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$

(a) Show this code can detect any single-qubit error.
(b) Can it correct any single-qubit error? Use Knill-Laflamme to justify.
(c) What is the code distance?

### Problem 13 (Level II)
Prove that for a quantum code to correct $$t$$ arbitrary errors, it must have distance $$d \geq 2t + 1$$.

*Hint:* Consider what happens when two different errors of weight $$\leq t$$ have product of weight $$< d$$.

### Problem 14 (Level II)
The **quantum Hamming bound** states that for an $$[[n, k, d]]$$ code with $$d = 2t + 1$$:
$$2^k \sum_{j=0}^{t} \binom{n}{j} 3^j \leq 2^n$$

(a) Derive this bound by counting syndrome states.
(b) Show that the $$[[5, 1, 3]]$$ code (if it exists) saturates this bound.
(c) Does the Shor code saturate the bound?

### Problem 15 (Level III)
Prove the **quantum Singleton bound**: For any $$[[n, k, d]]$$ quantum code,
$$k \leq n - 2(d - 1)$$

*Hint:* Consider tracing out $$d-1$$ qubits and use the no-cloning theorem.

### Problem 16 (Level III)
A code is **degenerate** if some errors $$E_a \neq E_b$$ satisfy $$E_a|\psi\rangle = E_b|\psi\rangle$$ for all codewords $$|\psi\rangle$$.

(a) Show that a degenerate code can violate the quantum Hamming bound.
(b) Give an example of a degenerate code.
(c) Prove that CSS codes (to be defined in Week 170) with $$C_1^\perp \subset C_2$$ can be degenerate.

### Problem 17 (Level III)
The Knill-Laflamme conditions can be written as $$PE_a^\dagger E_b P = C_{ab}P$$.

(a) Show that the matrix $$C = (C_{ab})$$ is positive semidefinite.
(b) Prove that if $$C$$ has rank $$r$$, we can find $$r$$ "canonical" errors $$\{F_i\}$$ with $$PF_i^\dagger F_j P = \delta_{ij} P$$.
(c) What is the physical interpretation of these canonical errors?

---

## Part D: Three-Qubit Codes (Problems 18-22)

### Problem 18 (Level I)
For the three-qubit bit-flip code:

(a) Write down the encoding circuit.
(b) Compute the syndrome measurement circuit using ancilla qubits.
(c) Verify that measuring $$Z_1Z_2$$ and $$Z_2Z_3$$ does not disturb the encoded state.

### Problem 19 (Level I)
For the three-qubit phase-flip code:

(a) Write the codewords explicitly in the computational basis.
(b) Design the syndrome measurement circuit.
(c) Show that this code corrects $$Z$$ errors but not $$X$$ errors.

### Problem 20 (Level II)
Consider applying a phase-flip error $$Z_2$$ to the bit-flip code.

(a) How does $$Z_2$$ affect $$|0_L\rangle = |000\rangle$$?
(b) How does $$Z_2$$ affect $$|1_L\rangle = |111\rangle$$?
(c) Show that for superposition states $$\alpha|0_L\rangle + \beta|1_L\rangle$$, the error corrupts the logical information.
(d) Explain this failure in terms of the Knill-Laflamme conditions.

### Problem 21 (Level II)
The **logical operators** for a code are operators that act nontrivially on the code space while preserving it.

(a) Find $$\overline{X}$$ and $$\overline{Z}$$ for the bit-flip code (operators that act as $$X$$ and $$Z$$ on the logical qubit).
(b) Verify that $$\overline{X}\overline{Z} = -\overline{Z}\overline{X}$$.
(c) How many independent logical operators are there for an $$[[n, 1, d]]$$ code?

### Problem 22 (Level III)
Generalize the bit-flip code to an $$[[n, 1, n]]$$ repetition code with $$n$$ odd.

(a) Write the codewords.
(b) How many bit-flip errors can this code correct?
(c) Design a majority-vote decoding scheme.
(d) Analyze the failure probability when each qubit has independent bit-flip probability $$p$$.
(e) For what values of $$p$$ does encoding improve fidelity?

---

## Part E: The Shor Code (Problems 23-27)

### Problem 23 (Level I)
Write out the full 9-qubit Shor code codewords:

(a) Express $$|0_L\rangle$$ as a superposition of computational basis states.
(b) Express $$|1_L\rangle$$ as a superposition of computational basis states.
(c) Verify orthonormality: $$\langle 0_L | 1_L \rangle = 0$$.

### Problem 24 (Level II)
For the Shor code, identify the syndrome for each of the following errors:

(a) $$X_1$$ (bit flip on qubit 1)
(b) $$Z_4$$ (phase flip on qubit 4)
(c) $$Y_7 = iX_7Z_7$$ (combined error on qubit 7)
(d) $$X_1X_2$$ (two bit flips)

### Problem 25 (Level II)
Prove that the Shor code has distance 3 by:

(a) Showing all weight-1 Pauli errors are detectable.
(b) Showing all weight-2 Pauli errors are detectable.
(c) Finding a weight-3 error that is NOT detectable (a logical operator).

### Problem 26 (Level III)
The Shor code can be viewed as a **concatenated code**.

(a) Define concatenation of quantum codes.
(b) If an inner code has distance $$d_1$$ and outer code has distance $$d_2$$, what is the distance of the concatenated code?
(c) Design a $$[[27, 1, 9]]$$ code by concatenating the Shor code with itself.
(d) Generalize: what are the parameters of $$k$$-level concatenation of the Shor code?

### Problem 27 (Level III)
Verify the Knill-Laflamme conditions for the Shor code.

(a) For the error set $$\{I, X_1, Z_1, Y_1\}$$, compute the matrix $$C_{ab}$$ defined by $$\langle 0_L | E_a^\dagger E_b | 0_L \rangle$$.
(b) Verify that $$\langle 0_L | E_a^\dagger E_b | 0_L \rangle = \langle 1_L | E_a^\dagger E_b | 1_L \rangle$$.
(c) Show that $$\langle 0_L | E_a^\dagger E_b | 1_L \rangle = 0$$ for all single-qubit errors.
(d) Conclude that the Shor code satisfies Knill-Laflamme for all single-qubit errors.

---

## Part F: Advanced Topics (Problems 28-30)

### Problem 28 (Level III)
The **approximate quantum error correction** conditions (Beny-Oreshkov) relax Knill-Laflamme.

(a) State the approximate conditions in terms of the fidelity after recovery.
(b) For a code with Knill-Laflamme error $$\epsilon$$ (meaning $$|PE_a^\dagger E_b P - C_{ab}P| \leq \epsilon$$), bound the fidelity of the recovered state.
(c) Discuss implications for near-term quantum devices.

### Problem 29 (Level III)
**Operator quantum error correction** generalizes the subspace framework.

(a) Define a subsystem code (code with gauge qubits).
(b) State the generalized Knill-Laflamme conditions for subsystem codes.
(c) Give an example where subsystem codes outperform subspace codes.

### Problem 30 (Level III)
Consider the **random coding** argument for quantum codes.

(a) Prove that random codes achieve capacity on the depolarizing channel with high probability.
(b) Specifically, show that for $$p < p_{\text{threshold}}$$, there exist codes with rate $$R > 0$$ and vanishing logical error rate.
(c) What is the hashing bound for the depolarizing channel?
(d) Why are random codes impractical, and what structure do practical codes impose?

---

## Submission Guidelines

- Show all work and justify each step
- For proofs, clearly state what you are proving and use precise mathematical language
- For calculations, include intermediate steps
- Diagrams and circuits should be clearly labeled
- Numerical answers should be simplified

---

**Problem Set Created:** February 9, 2026
**Total Problems:** 30
**Estimated Time:** 15-20 hours
