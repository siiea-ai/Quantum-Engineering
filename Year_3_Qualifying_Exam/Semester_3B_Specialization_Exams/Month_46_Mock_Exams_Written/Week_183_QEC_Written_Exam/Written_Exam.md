# Quantum Error Correction Written Qualifying Exam

## Exam Information

**Duration:** 3 hours (180 minutes)
**Total Points:** 200
**Number of Problems:** 8
**Passing Score:** 160 points (80%)

---

## Instructions

1. **Time Management:** Budget approximately 22 minutes per problem.

2. **Show All Work:** Partial credit awarded for correct methodology.

3. **Notation:** Use standard stabilizer notation. Denote Pauli operators as $X$, $Y$, $Z$, $I$.

4. **Proofs:** Structure proofs clearly with stated assumptions.

5. **Diagrams:** Draw syndrome extraction circuits clearly when asked.

6. **No External Resources:** Closed-book exam.

---

## Useful Definitions

**Pauli Group:** $\mathcal{P}_n = \{\pm 1, \pm i\} \times \{I, X, Y, Z\}^{\otimes n}$

**Stabilizer Code:** $\mathcal{C} = \{|\psi\rangle : S|\psi\rangle = |\psi\rangle \text{ for all } S \in \mathcal{S}\}$

**CSS Code:** Stabilizers split into $X$-type and $Z$-type, derived from classical codes $C_1 \supseteq C_2$

**Distance:** $d = \min\{\text{wt}(E) : E \in N(\mathcal{S}) \setminus \mathcal{S}\}$

---

## Problem 1: Classical Error Correction Foundations (25 points)

The classical Hamming code $[7, 4, 3]$ has parity check matrix:

$$H = \begin{pmatrix} 1 & 1 & 1 & 0 & 1 & 0 & 0 \\ 1 & 1 & 0 & 1 & 0 & 1 & 0 \\ 1 & 0 & 1 & 1 & 0 & 0 & 1 \end{pmatrix}$$

**(a)** (6 points) Determine the generator matrix $G$ for this code. Verify that $HG^T = 0$.

**(b)** (6 points) A codeword is transmitted and received as $\mathbf{r} = (1, 0, 1, 1, 0, 1, 0)$. Calculate the syndrome and identify the error location.

**(c)** (6 points) What is the dual code $C^\perp$ of the Hamming code? What are its parameters $[n, k, d]$?

**(d)** (7 points) Explain why the Hamming code can correct any single-bit error but cannot correct two-bit errors. Relate this to the Hamming bound.

---

## Problem 2: Quantum Error Correction Basics (25 points)

Consider the 3-qubit bit-flip code with codewords:
$$|0_L\rangle = |000\rangle, \quad |1_L\rangle = |111\rangle$$

**(a)** (6 points) Write the stabilizer generators for this code. What is the code distance?

**(b)** (7 points) Suppose the state $|\psi_L\rangle = \alpha|0_L\rangle + \beta|1_L\rangle$ experiences a bit-flip error on the second qubit. Describe the syndrome measurement process and how the error is corrected.

**(c)** (6 points) Explain why this code cannot correct phase-flip ($Z$) errors. What modification leads to Shor's 9-qubit code?

**(d)** (6 points) State the Knill-Laflamme conditions for quantum error correction. Verify that the 3-qubit bit-flip code satisfies these conditions for the error set $\{I, X_1, X_2, X_3\}$.

---

## Problem 3: Stabilizer Formalism (25 points)

Consider a stabilizer code defined by the generators:
$$S_1 = X_1 X_2 X_3 X_4, \quad S_2 = Z_1 Z_2 Z_3 Z_4, \quad S_3 = X_1 X_2 Z_3 Z_4$$

**(a)** (6 points) Verify that $S_1$, $S_2$, $S_3$ mutually commute and are independent. How many logical qubits does this code encode?

**(b)** (7 points) Find a complete set of logical operators $\bar{X}$ and $\bar{Z}$ for this code. Verify that $\bar{X}$ and $\bar{Z}$ commute with all stabilizers but anticommute with each other.

**(c)** (6 points) What is the code distance? List all weight-2 errors that this code can detect but not correct.

**(d)** (6 points) Suppose syndrome measurement yields $(+1, -1, +1)$ for $(S_1, S_2, S_3)$. What error(s) could have occurred? What correction should be applied?

---

## Problem 4: CSS Code Construction (25 points)

The $[[7, 1, 3]]$ Steane code is constructed from the classical $[7, 4, 3]$ Hamming code $C_1$ and its dual $C_2 = C_1^\perp$.

**(a)** (7 points) Write down all six stabilizer generators for the Steane code (three $X$-type, three $Z$-type). Use the parity check matrix from Problem 1.

**(b)** (6 points) Find the logical operators $\bar{X}$ and $\bar{Z}$ for the Steane code. Verify that $\bar{X}\bar{Z} = -\bar{Z}\bar{X}$.

**(c)** (6 points) Show that the transversal $T^{\otimes 7}$ gate (where $T = \text{diag}(1, e^{i\pi/4})$) does NOT implement a valid logical gate on the Steane code. What goes wrong?

**(d)** (6 points) The Steane code supports transversal Hadamard. Show that $H^{\otimes 7}$ implements $\bar{H}$ (the logical Hadamard). What symmetry of the code makes this possible?

---

## Problem 5: Surface Code (25 points)

Consider a distance-3 surface code on a $3 \times 3$ planar lattice with rough and smooth boundaries.

**(a)** (6 points) Draw the lattice showing data qubits, $X$-stabilizers (plaquettes), and $Z$-stabilizers (vertices). How many physical qubits, $X$-stabilizers, and $Z$-stabilizers are there?

**(b)** (7 points) Identify the logical $\bar{X}$ and $\bar{Z}$ operators as strings connecting boundaries. Why is the code distance $d = 3$?

**(c)** (6 points) Suppose a single $Z$ error occurs on a data qubit in the bulk. Which stabilizers detect this error? Draw the syndrome pattern.

**(d)** (6 points) Explain the minimum-weight perfect matching (MWPM) decoder. What is the computational complexity of MWPM, and why is it practical for surface codes?

---

## Problem 6: Fault-Tolerant Operations (25 points)

**(a)** (6 points) Define what it means for an operation to be "fault-tolerant" in the context of a distance-$d$ code. Why is fault tolerance necessary for scalable quantum computation?

**(b)** (7 points) The CNOT gate is transversal for CSS codes. Draw the transversal CNOT circuit between two blocks of a $[[7,1,3]]$ code. Explain why a single fault in this circuit cannot cause a logical error.

**(c)** (6 points) Explain why there is no code for which the full Clifford group plus $T$ gate can all be implemented transversally. Reference the Eastin-Knill theorem.

**(d)** (6 points) Describe magic state distillation. Starting with noisy $|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$ states, how can high-fidelity $|T\rangle$ states be produced? What is the resource overhead?

---

## Problem 7: Threshold Theorem (25 points)

**(a)** (6 points) State the threshold theorem for fault-tolerant quantum computation. What are the key assumptions?

**(b)** (7 points) For concatenated codes, derive the logical error rate after $k$ levels of concatenation. If the physical error rate is $p$ and the pseudo-threshold is $p_{th}$, show that:
$$p_L^{(k)} \sim \left(\frac{p}{p_{th}}\right)^{2^k}$$

**(c)** (6 points) Estimate the number of physical qubits needed to achieve logical error rate $p_L = 10^{-15}$ using a $[[7,1,3]]$ code with $p = 10^{-3}$ and $p_{th} = 10^{-2}$. How many levels of concatenation are required?

**(d)** (6 points) Compare the overhead of concatenated codes to topological codes (surface codes). For a target logical error rate, which approach is more efficient asymptotically? Discuss the practical trade-offs.

---

## Problem 8: Advanced Topics (25 points)

**(a)** (7 points) Define a quantum LDPC (Low-Density Parity Check) code. What are the advantages of QLDPC codes over surface codes in terms of encoding rate and distance? Why are they harder to implement?

**(b)** (6 points) The toric code is defined on a torus. Write the stabilizer generators and identify the logical operators. How many logical qubits does the toric code encode?

**(c)** (6 points) Describe the Union-Find decoder for surface codes. What is its computational complexity, and how does it compare to MWPM?

**(d)** (6 points) Recent breakthrough: codes with constant overhead achieve fault-tolerant computation with $O(k)$ physical qubits to encode $k$ logical qubits. Explain why this is significant and what challenges remain for implementation.

---

## End of Exam

**Checklist before submitting:**
- [ ] All problems attempted
- [ ] Stabilizer generators written correctly
- [ ] Circuit diagrams clearly drawn
- [ ] Proofs logically structured

**Good luck!**

---

*This exam covers material from Gottesman's lectures, Nielsen & Chuang Chapter 10, and Preskill's Ph219 notes.*
