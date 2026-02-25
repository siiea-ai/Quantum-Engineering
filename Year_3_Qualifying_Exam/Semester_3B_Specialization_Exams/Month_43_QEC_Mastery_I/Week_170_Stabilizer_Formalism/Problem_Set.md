# Week 170: Stabilizer Formalism - Problem Set

## Instructions

This problem set contains 28 problems at the PhD qualifying examination level. Problems emphasize stabilizer calculations, CSS code construction, and the Gottesman-Knill theorem.

**Levels:**
- **Level I:** Direct application of definitions
- **Level II:** Multi-step reasoning and proofs
- **Level III:** Challenging synthesis problems

Time estimate: 15-20 hours total

---

## Part A: Pauli Group (Problems 1-5)

### Problem 1 (Level I)
Compute the following products in $$\mathcal{G}_3$$:

(a) $$(X_1Z_2)(Y_1X_2)$$

(b) $$(X_1X_2X_3)(Z_1Z_2Z_3)$$

(c) $$(iY_1Z_2)(−iX_1Y_2)$$

(d) $$(X_1Z_2Y_3)^2$$

### Problem 2 (Level I)
For each pair of Pauli operators, determine whether they commute or anticommute:

(a) $$X_1X_2$$ and $$Z_1Z_2$$

(b) $$X_1Z_2$$ and $$Z_1X_2$$

(c) $$Y_1Y_2Y_3$$ and $$X_1X_2X_3$$

(d) $$X_1X_2Z_3Z_4$$ and $$Z_1Z_2X_3X_4$$

### Problem 3 (Level II)
(a) Prove that the center of $$\mathcal{G}_n$$ is $$Z(\mathcal{G}_n) = \{\pm I, \pm iI\}$$.

(b) What is the size of $$\mathcal{G}_n/Z(\mathcal{G}_n)$$?

(c) Show that $$\mathcal{G}_n/Z(\mathcal{G}_n) \cong \mathbb{Z}_2^{2n}$$ as groups.

### Problem 4 (Level II)
Define the **symplectic form** on $$\mathbb{Z}_2^{2n}$$ by:
$$\langle (a|b), (a'|b') \rangle = a \cdot b' + a' \cdot b \pmod 2$$

where $$(a|b)$$ represents the Pauli $$X^{a_1}Z^{b_1} \otimes \cdots \otimes X^{a_n}Z^{b_n}$$.

(a) Show that two Paulis commute iff their symplectic inner product is 0.

(b) Find all elements of $$\mathbb{Z}_2^4$$ (i.e., 2-qubit Paulis mod phase) that commute with $$(1,0|0,1) = X_1Z_2$$.

(c) Show that a maximal commuting set has size $$2^n$$.

### Problem 5 (Level III)
The **Pauli weight enumerator** of a stabilizer code counts operators by weight.

(a) Define the weight enumerator $$W_\mathcal{S}(x, y) = \sum_{S \in \mathcal{S}} x^{n-\text{wt}(S)} y^{\text{wt}(S)}$$ for stabilizer group $$\mathcal{S}$$.

(b) For the $$[[5,1,3]]$$ code, compute $$W_\mathcal{S}(x, y)$$.

(c) How does the weight enumerator relate to code distance?

---

## Part B: Stabilizer States (Problems 6-10)

### Problem 6 (Level I)
Find the stabilizer group for each state:

(a) $$|1\rangle$$

(b) $$|-\rangle = (|0\rangle - |1\rangle)/\sqrt{2}$$

(c) $$|01\rangle + |10\rangle)/\sqrt{2}$$

(d) $$(|000\rangle + |111\rangle)/\sqrt{2}$$ (GHZ state)

### Problem 7 (Level I)
Given the stabilizer generators, find the stabilized state:

(a) Generators: $$\{X\}$$ (single qubit)

(b) Generators: $$\{X_1X_2, Z_1Z_2\}$$

(c) Generators: $$\{X_1Z_2, Z_1X_2\}$$

### Problem 8 (Level II)
Prove that the stabilizer group of a state $$|\psi\rangle$$ is abelian.

*Hint:* Consider what happens when two stabilizers don't commute.

### Problem 9 (Level II)
(a) Show that $$|+\rangle^{\otimes n}$$ is a stabilizer state with stabilizer $$\langle X_1, X_2, \ldots, X_n \rangle$$.

(b) Apply $$CZ_{12}$$ (controlled-Z between qubits 1 and 2). Find the new stabilizer generators.

(c) Generalize: For a graph $$G$$, the graph state $$|G\rangle$$ has stabilizers $$g_v = X_v \prod_{u \sim v} Z_u$$.

### Problem 10 (Level III)
A **stabilizer state** on $$n$$ qubits is uniquely determined by $$2n^2 + O(n)$$ bits (the stabilizer tableau).

(a) Count the total number of $$n$$-qubit stabilizer states (up to global phase).

(b) Compare to the dimension of the space of all pure states.

(c) For $$n = 10$$, what fraction of pure states are stabilizer states?

---

## Part C: Stabilizer Codes (Problems 11-16)

### Problem 11 (Level I)
For the 3-qubit bit-flip code with stabilizers $$\langle Z_1Z_2, Z_2Z_3 \rangle$$:

(a) Verify the generators commute.

(b) Find the code space (give an orthonormal basis).

(c) Find logical operators $$\overline{X}$$ and $$\overline{Z}$$.

(d) What is the code distance?

### Problem 12 (Level I)
For each error, find the syndrome with respect to the bit-flip code stabilizers $$\{Z_1Z_2, Z_2Z_3\}$$:

(a) $$I$$

(b) $$X_1$$

(c) $$X_2$$

(d) $$X_3$$

(e) $$Z_1$$

(f) $$X_1X_2$$

### Problem 13 (Level II)
The **[[4, 2, 2]]** code has stabilizers:
$$g_1 = X_1X_2X_3X_4, \quad g_2 = Z_1Z_2Z_3Z_4$$

(a) Verify these generators commute.

(b) Find all logical operators. How many independent logical operators are there?

(c) What is the code distance? What errors can be detected/corrected?

(d) This is a degenerate code. Explain why.

### Problem 14 (Level II)
Prove that for an $$[[n, k, d]]$$ stabilizer code, the distance satisfies:
$$d = \min\{\text{wt}(P) : P \in C(\mathcal{S}) \setminus \mathcal{S}\}$$

where $$C(\mathcal{S})$$ is the centralizer of $$\mathcal{S}$$ in $$\mathcal{G}_n$$.

### Problem 15 (Level III)
The **[[5, 1, 3]]** code has stabilizers:
$$g_1 = XZZXI, \quad g_2 = IXZZX, \quad g_3 = XIXZZ, \quad g_4 = ZXIXZ$$

(a) Verify all pairs commute.

(b) Find logical operators $$\overline{X}$$ and $$\overline{Z}$$.

(c) Verify that any weight-1 or weight-2 Pauli anticommutes with at least one stabilizer.

(d) Find a weight-3 logical operator to confirm $$d = 3$$.

### Problem 16 (Level III)
Design an $$[[8, 3, 2]]$$ stabilizer code.

(a) How many stabilizer generators are needed?

(b) Propose 5 commuting generators.

(c) Find logical operators for all 3 logical qubits.

(d) Verify the distance is 2.

---

## Part D: CSS Codes (Problems 17-22)

### Problem 17 (Level I)
The repetition code $$C = \{000, 111\}$$ is a $$[3, 1, 3]$$ code.

(a) Find the parity-check matrix $$H$$.

(b) Find the dual code $$C^\perp$$.

(c) Construct $$CSS(C, C^\perp)$$. What are the parameters?

### Problem 18 (Level II)
Construct the Steane code as $$CSS(C, C)$$ where $$C$$ is the $$[7, 4, 3]$$ Hamming code.

(a) Write the parity-check matrix of the Hamming code.

(b) Find the generator matrix of $$C^\perp$$ (dual Hamming code).

(c) List all X-type and Z-type stabilizer generators.

(d) Verify the code has parameters $$[[7, 1, 3]]$$.

### Problem 19 (Level II)
For a CSS code $$CSS(C_1, C_2)$$ with $$C_2 \subset C_1$$:

(a) Prove that X-stabilizers and Z-stabilizers commute.

(b) Express the logical $$\overline{X}$$ and $$\overline{Z}$$ in terms of the classical codes.

(c) Show that the distance is $$d = \min(d_1, d_2^\perp)$$ where $$d_2^\perp$$ is the distance of $$C_2^\perp$$.

### Problem 20 (Level II)
The CSS codewords are:
$$|x + C_2\rangle = \frac{1}{\sqrt{|C_2|}} \sum_{y \in C_2} |x + y\rangle$$

(a) Show these states are orthonormal for different cosets.

(b) For the Steane code, explicitly write $$|0_L\rangle$$ and $$|1_L\rangle$$ as sums over computational basis states.

(c) Verify $$Z^{\otimes 7}|0_L\rangle = |0_L\rangle$$ and $$Z^{\otimes 7}|1_L\rangle = -|1_L\rangle$$.

### Problem 21 (Level III)
Design a CSS code from the $$[15, 11, 3]$$ Hamming code and its dual.

(a) What are the parameters of the resulting quantum code?

(b) List the stabilizer generators.

(c) How does this compare to the Steane code?

### Problem 22 (Level III)
**Hypergraph product codes** generalize CSS codes.

(a) Given classical codes $$C_1 = [n_1, k_1, d_1]$$ and $$C_2 = [n_2, k_2, d_2]$$, the hypergraph product is:
$$[[n_1 n_2 + (n_1 - k_1)(n_2 - k_2), k_1 k_2, \min(d_1, d_2)]]$$

Verify this formula for $$C_1 = C_2 =$$ repetition code $$[3, 1, 3]$$.

(b) What is the resulting quantum code? Is it familiar?

---

## Part E: Gottesman-Knill Theorem (Problems 23-28)

### Problem 23 (Level I)
Verify the action of Clifford gates on Pauli operators:

(a) Show $$HXH^\dagger = Z$$ and $$HZH^\dagger = X$$.

(b) Show $$SXS^\dagger = Y$$ and $$SZS^\dagger = Z$$.

(c) For CNOT with control on qubit 1, target on qubit 2:
- $$CNOT \cdot X_1 \cdot CNOT^\dagger = ?$$
- $$CNOT \cdot Z_2 \cdot CNOT^\dagger = ?$$

### Problem 24 (Level I)
Start with state $$|00\rangle$$ (stabilizer: $$\langle Z_1, Z_2 \rangle$$).

Track the stabilizer through this circuit:
1. Apply $$H$$ to qubit 1
2. Apply CNOT from qubit 1 to qubit 2
3. Apply $$S$$ to qubit 2

What is the final stabilizer? What state is this?

### Problem 25 (Level II)
A stabilizer tableau for 2 qubits has the form:

$$\begin{pmatrix} x_{11} & x_{12} & | & z_{11} & z_{12} & | & r_1 \\ x_{21} & x_{22} & | & z_{21} & z_{22} & | & r_2 \end{pmatrix}$$

(a) Write the tableau for $$|00\rangle$$.

(b) Apply $$H_1$$ (Hadamard on qubit 1) and update the tableau.

(c) Apply $$CNOT_{12}$$ and update the tableau.

(d) What state does the final tableau represent?

### Problem 26 (Level II)
Prove that Clifford gates map stabilizer states to stabilizer states.

*Hint:* If $$S|\psi\rangle = |\psi\rangle$$, what can you say about $$USU^\dagger$$ and $$U|\psi\rangle$$?

### Problem 27 (Level III)
The Gottesman-Knill theorem states Clifford circuits are classically simulable.

(a) Describe the simulation algorithm.

(b) Analyze the time complexity: $$O(?)$$ for $$n$$ qubits and $$m$$ gates.

(c) How is measurement in the computational basis simulated?

(d) Why does adding T gates break efficient simulation?

### Problem 28 (Level III)
The **Clifford hierarchy** is defined recursively:
- $$\mathcal{C}_1 = \mathcal{G}_n$$ (Pauli group)
- $$\mathcal{C}_{k+1} = \{U : UPU^\dagger \in \mathcal{C}_k \text{ for all } P \in \mathcal{G}_n\}$$

(a) Show $$\mathcal{C}_2$$ is the Clifford group.

(b) Show the T gate is in $$\mathcal{C}_3$$ but not $$\mathcal{C}_2$$.

(c) Prove that $$\mathcal{C}_3$$ gates, combined with Clifford gates and Pauli measurements, are still classically simulable.

(d) At what level of the hierarchy do we get universal quantum computation?

---

## Submission Guidelines

- Show all stabilizer calculations step by step
- For tableau problems, draw the full tableau at each step
- Clearly indicate when operators commute vs anticommute
- For CSS codes, explicitly relate to the classical code structure

---

**Problem Set Created:** February 10, 2026
**Total Problems:** 28
**Estimated Time:** 15-20 hours
