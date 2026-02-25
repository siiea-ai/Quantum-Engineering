# Week 170: Stabilizer Formalism - Review Guide

## Introduction

The stabilizer formalism provides an elegant algebraic framework for quantum error correction that has become indispensable in modern quantum computing. Developed primarily by Daniel Gottesman in his 1997 PhD thesis, this formalism allows us to describe large classes of quantum states and operations using polynomial resources, despite the exponential size of the underlying Hilbert space.

This review covers the complete stabilizer framework, from the Pauli group through CSS codes and the Gottesman-Knill theorem on classical simulability.

---

## 1. The Pauli Group

### 1.1 Single-Qubit Pauli Group

The single-qubit Pauli group $$\mathcal{G}_1$$ consists of the Pauli matrices with phase factors:

$$\mathcal{G}_1 = \{\pm I, \pm iI, \pm X, \pm iX, \pm Y, \pm iY, \pm Z, \pm iZ\}$$

This is a group under matrix multiplication with $$|\mathcal{G}_1| = 16$$.

**Key properties:**
- Paulis square to identity: $$X^2 = Y^2 = Z^2 = I$$
- Anticommutation: $$XY = -YX = iZ$$ (and cyclic permutations)
- Every element has order dividing 4

### 1.2 n-Qubit Pauli Group

The n-qubit Pauli group is:

$$\mathcal{G}_n = \{e^{i\phi} P_1 \otimes P_2 \otimes \cdots \otimes P_n : \phi \in \{0, \pi/2, \pi, 3\pi/2\}, P_j \in \{I, X, Y, Z\}\}$$

**Size:** $$|\mathcal{G}_n| = 4 \cdot 4^n = 4^{n+1}$$

**Notation:** We often write tensor products without the $$\otimes$$ symbol:
$$X_1Z_2Y_3 \equiv X \otimes Z \otimes Y \otimes I \otimes \cdots \otimes I$$

### 1.3 Commutation Relations

For $$P, Q \in \mathcal{G}_n$$, either:
- $$PQ = QP$$ (they commute), or
- $$PQ = -QP$$ (they anticommute)

**Commutation criterion:** Define the **symplectic inner product** on binary vectors. For $$P = i^a X^{x_1}Z^{z_1} \otimes \cdots$$ and $$Q = i^b X^{x_1'}Z^{z_1'} \otimes \cdots$$:

$$[P, Q] = 0 \iff \sum_j (x_j z_j' + x_j' z_j) \equiv 0 \pmod 2$$

### 1.4 Weight and Support

The **weight** of a Pauli operator is the number of qubits on which it acts non-trivially (not as identity).

The **support** is the set of qubits where it acts non-trivially.

**Example:** $$X_1Y_3Z_5$$ on 7 qubits has weight 3 and support $$\{1, 3, 5\}$$.

---

## 2. Stabilizer States

### 2.1 Definition

A pure state $$|\psi\rangle$$ is a **stabilizer state** if there exists an abelian subgroup $$\mathcal{S} \subset \mathcal{G}_n$$ such that:

1. $$-I \notin \mathcal{S}$$ (otherwise no state is stabilized)
2. $$|\mathcal{S}| = 2^n$$ (maximal size for n qubits)
3. $$S|\psi\rangle = |\psi\rangle$$ for all $$S \in \mathcal{S}$$

The group $$\mathcal{S}$$ is called the **stabilizer group** of $$|\psi\rangle$$.

### 2.2 Examples

**Computational basis states:**
- $$|0\rangle$$ is stabilized by $$\{I, Z\}$$
- $$|00\cdots0\rangle$$ is stabilized by $$\langle Z_1, Z_2, \ldots, Z_n \rangle$$

**Plus states:**
- $$|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$$ is stabilized by $$\{I, X\}$$

**Bell state:**
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
Stabilizer: $$\langle X_1X_2, Z_1Z_2 \rangle = \{I, X_1X_2, Z_1Z_2, Y_1Y_2\}$$

**GHZ state:**
$$|GHZ\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$
Stabilizer: $$\langle X_1X_2X_3, Z_1Z_2, Z_2Z_3 \rangle$$

### 2.3 Generators

A stabilizer group $$\mathcal{S}$$ with $$|\mathcal{S}| = 2^n$$ can be specified by $$n$$ independent generators $$g_1, \ldots, g_n$$:

$$\mathcal{S} = \langle g_1, g_2, \ldots, g_n \rangle$$

**Independent:** No generator can be written as a product of others.

**Commuting:** Generators must pairwise commute (since $$\mathcal{S}$$ is abelian).

### 2.4 Uniqueness

**Theorem:** A maximal abelian stabilizer group (with $$-I \notin \mathcal{S}$$) determines a unique state up to global phase.

**Proof sketch:** The projector onto the stabilized state is:
$$P = \frac{1}{|\mathcal{S}|}\sum_{S \in \mathcal{S}} S = \frac{1}{2^n}\sum_{S \in \mathcal{S}} S$$

Since $$|\mathcal{S}| = 2^n$$ and the Hilbert space is $$2^n$$-dimensional, this projector has rank 1.

---

## 3. Stabilizer Codes

### 3.1 Definition

An $$[[n, k]]$$ **stabilizer code** is defined by an abelian subgroup $$\mathcal{S} \subset \mathcal{G}_n$$ with:
- $$-I \notin \mathcal{S}$$
- $$|\mathcal{S}| = 2^{n-k}$$ (equivalently, $$n - k$$ independent generators)

The **code space** is the simultaneous +1 eigenspace:
$$\mathcal{C} = \{|\psi\rangle : S|\psi\rangle = |\psi\rangle \text{ for all } S \in \mathcal{S}\}$$

**Dimension:** $$\dim(\mathcal{C}) = 2^k$$ (encodes $$k$$ logical qubits)

### 3.2 Stabilizer Generators

Specify the code by $$r = n - k$$ independent, commuting generators:
$$\mathcal{S} = \langle g_1, g_2, \ldots, g_r \rangle$$

**Example: Bit-flip code**
$$\mathcal{S} = \langle Z_1Z_2, Z_2Z_3 \rangle$$

This is an $$[[3, 1]]$$ code (3 qubits, 1 logical qubit).

**Example: Shor code**
$$\mathcal{S} = \langle Z_1Z_2, Z_2Z_3, Z_4Z_5, Z_5Z_6, Z_7Z_8, Z_8Z_9, X_1X_2X_3X_4X_5X_6, X_4X_5X_6X_7X_8X_9 \rangle$$

This is $$[[9, 1]]$$.

### 3.3 Logical Operators

**Centralizer:** The centralizer of $$\mathcal{S}$$ in $$\mathcal{G}_n$$ is:
$$C(\mathcal{S}) = \{P \in \mathcal{G}_n : PS = SP \text{ for all } S \in \mathcal{S}\}$$

**Logical operators:** Elements of $$C(\mathcal{S})$$ that are NOT in $$\mathcal{S}$$ act non-trivially on the code space.

The **logical Pauli group** is:
$$\overline{\mathcal{G}}_k = C(\mathcal{S})/\mathcal{S}$$

For an $$[[n, k]]$$ code, this is isomorphic to $$\mathcal{G}_k$$.

**Finding logical operators:** Choose $$2k$$ operators $$\overline{X}_1, \overline{Z}_1, \ldots, \overline{X}_k, \overline{Z}_k$$ in $$C(\mathcal{S}) \setminus \mathcal{S}$$ satisfying:
- $$\overline{X}_i$$ anticommutes with $$\overline{Z}_i$$
- $$\overline{X}_i$$ commutes with $$\overline{X}_j, \overline{Z}_j$$ for $$i \neq j$$

### 3.4 Code Distance

The **distance** $$d$$ of a stabilizer code is the minimum weight of any logical operator:
$$d = \min\{\text{wt}(P) : P \in C(\mathcal{S}) \setminus \mathcal{S}\}$$

Equivalently: the minimum weight of a Pauli that commutes with all stabilizers but is not in the stabilizer.

**Error correction:** An $$[[n, k, d]]$$ code corrects $$t = \lfloor(d-1)/2\rfloor$$ errors.

### 3.5 Syndrome Measurement

For error $$E \in \mathcal{G}_n$$, the **syndrome** is the vector of commutation relations with generators:

$$s_i = \begin{cases} 0 & \text{if } [E, g_i] = 0 \\ 1 & \text{if } \{E, g_i\} = 0 \end{cases}$$

**Key property:** The syndrome depends only on the error, not the encoded state.

**Correction:** Given syndrome $$s$$, find error $$E$$ with that syndrome and apply $$E$$ to correct.

---

## 4. CSS Codes

### 4.1 The CSS Construction

CSS codes are stabilizer codes with a special structure that separates X and Z errors.

**Input:** Two classical linear codes $$C_1, C_2$$ over $$\mathbb{F}_2$$ with:
- $$C_2 \subset C_1$$
- $$C_1$$ is $$[n, k_1, d_1]$$
- $$C_2$$ is $$[n, k_2, d_2]$$

**Output:** Quantum code $$CSS(C_1, C_2)$$ with parameters:
- $$n$$ physical qubits
- $$k = k_1 - k_2$$ logical qubits
- $$d = \min(d_1, d_2^\perp)$$ where $$d_2^\perp$$ is the distance of $$C_2^\perp$$

### 4.2 Stabilizer Generators

**Z-type stabilizers:** For each row $$h$$ of the parity-check matrix $$H_1$$ of $$C_1$$:
$$g_Z = Z^{h_1} \otimes Z^{h_2} \otimes \cdots \otimes Z^{h_n}$$

**X-type stabilizers:** For each codeword $$c \in C_2^\perp$$:
$$g_X = X^{c_1} \otimes X^{c_2} \otimes \cdots \otimes X^{c_n}$$

**Commutation:** Z-stabilizers commute with X-stabilizers because $$C_2^\perp \subset C_1$$ (orthogonality).

### 4.3 Codewords

The logical computational basis states are:
$$|x + C_2\rangle = \frac{1}{\sqrt{|C_2|}}\sum_{y \in C_2} |x + y\rangle$$

where $$x$$ ranges over coset representatives of $$C_2$$ in $$C_1$$.

### 4.4 Error Correction

**X errors** (bit flips): Detected by Z-stabilizers using the parity-check matrix $$H_1$$.

**Z errors** (phase flips): Detected by X-stabilizers using the parity-check matrix of $$C_2^\perp$$.

The classical decoding algorithms for $$C_1$$ and $$C_2^\perp$$ can be used.

### 4.5 Advantages of CSS Codes

1. **Separate X and Z correction:** Simplifies decoding
2. **Transversal CNOT:** CNOT between two CSS codeblocks is transversal
3. **Classical code leverage:** Use existing classical code theory
4. **Fault-tolerant gates:** Often have nice transversal gate sets

---

## 5. The Steane Code

### 5.1 Construction

The Steane code is $$CSS(C, C)$$ where $$C$$ is the $$[7, 4, 3]$$ Hamming code.

Since the Hamming code is self-dual ($$C = C^\perp$$), we have $$C^\perp \subset C$$.

**Parameters:** $$[[7, 1, 3]]$$
- 7 physical qubits
- 1 logical qubit
- Distance 3

### 5.2 Stabilizer Generators

**X-type (from $$C^\perp$$ codewords):**
$$\begin{aligned}
g_1 &= IIIXXXX \\
g_2 &= IXXIIXX \\
g_3 &= XIXIXIX
\end{aligned}$$

**Z-type (from $$H$$ rows):**
$$\begin{aligned}
g_4 &= IIIZZZZ \\
g_5 &= IZZIIZZ \\
g_6 &= ZIZIZIZ
\end{aligned}$$

### 5.3 Logical Operators

$$\overline{X} = X^{\otimes 7} = XXXXXXX$$
$$\overline{Z} = Z^{\otimes 7} = ZZZZZZZ$$

These have weight 7, but equivalent logical operators with weight 3 exist:
$$\overline{X} \sim X_1X_2X_3$$
$$\overline{Z} \sim Z_1Z_2Z_3$$

(equivalent modulo stabilizers)

### 5.4 Transversal Gates

The Steane code supports transversal implementation of:
- **Pauli gates:** $$\overline{X} = X^{\otimes 7}$$, $$\overline{Z} = Z^{\otimes 7}$$
- **Hadamard:** $$\overline{H} = H^{\otimes 7}$$
- **CNOT:** $$\overline{CNOT} = CNOT^{\otimes 7}$$ (between two code blocks)
- **S gate:** $$\overline{S} = S^{\otimes 7}$$ (with care about signs)

**Not transversal:** T gate (requires magic state injection)

### 5.5 Comparison with Shor Code

| Property | Shor [[9,1,3]] | Steane [[7,1,3]] |
|----------|----------------|------------------|
| Qubits | 9 | 7 |
| Generators | 8 | 6 |
| CSS structure | No | Yes |
| Transversal H | No | Yes |
| Transversal CNOT | Yes | Yes |

---

## 6. The Gottesman-Knill Theorem

### 6.1 The Clifford Group

The **Clifford group** $$\mathcal{C}_n$$ consists of unitaries that normalize the Pauli group:

$$\mathcal{C}_n = \{U : UPU^\dagger \in \mathcal{G}_n \text{ for all } P \in \mathcal{G}_n\}$$

**Generators of the Clifford group:**
- Hadamard: $$H$$
- Phase gate: $$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$
- CNOT: $$CNOT_{ij}$$

**Action on Paulis:**

| Gate | $$X \to$$ | $$Z \to$$ |
|------|-----------|-----------|
| $$H$$ | $$Z$$ | $$X$$ |
| $$S$$ | $$Y$$ | $$Z$$ |
| $$CNOT_{12}$$ | $$X_1 \to X_1X_2$$, $$X_2 \to X_2$$ | $$Z_1 \to Z_1$$, $$Z_2 \to Z_1Z_2$$ |

### 6.2 Stabilizer Tableau

An n-qubit stabilizer state can be represented by a **tableau**: a $$(2n) \times (2n+1)$$ binary matrix encoding:
- n stabilizer generators
- n "destabilizer" generators (for efficient updates)

**Tableau format:**
$$\begin{pmatrix} x_{11} & \cdots & x_{1n} & z_{11} & \cdots & z_{1n} & r_1 \\ \vdots & & \vdots & \vdots & & \vdots & \vdots \\ x_{n1} & \cdots & x_{nn} & z_{n1} & \cdots & z_{nn} & r_n \end{pmatrix}$$

Row $$i$$ represents generator $$g_i = i^{r_i} X^{x_{i1}}Z^{z_{i1}} \otimes \cdots \otimes X^{x_{in}}Z^{z_{in}}$$

### 6.3 The Theorem

**Theorem (Gottesman-Knill):** A quantum circuit consisting of:
1. Preparation of qubits in computational basis states
2. Clifford gates (H, S, CNOT)
3. Measurements in the computational basis

can be efficiently simulated on a classical computer in time $$O(n^2 m)$$ where $$n$$ is the number of qubits and $$m$$ is the number of gates.

**Proof idea:**
- Initial state $$|0\rangle^{\otimes n}$$ has stabilizer $$\langle Z_1, \ldots, Z_n \rangle$$
- Each Clifford gate updates the tableau in $$O(n)$$ time
- Measurement outcomes can be computed from the tableau

### 6.4 Implications

**Not universal:** Clifford circuits cannot achieve universal quantum computation (they form a finite subgroup of unitaries up to phase).

**Need non-Clifford gates:** Adding the T gate makes the gate set universal but breaks efficient simulation.

**Magic state distillation:** T gates can be implemented using "magic states" $$|T\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$$ and Clifford operations.

### 6.5 Tableau Update Rules

**Hadamard on qubit $$j$$:** Swap columns $$j$$ and $$n+j$$ in the tableau.

**S gate on qubit $$j$$:** For each row, if $$x_{ij} = 1$$, flip $$z_{ij}$$ and update phase.

**CNOT from $$j$$ to $$k$$:** For each row:
- $$x_{ik} \leftarrow x_{ij} \oplus x_{ik}$$
- $$z_{ij} \leftarrow z_{ij} \oplus z_{ik}$$

---

## 7. Advanced Topics

### 7.1 The [[5, 1, 3]] Perfect Code

The smallest code achieving distance 3:

**Stabilizers:**
$$\begin{aligned}
g_1 &= XZZXI \\
g_2 &= IXZZX \\
g_3 &= XIXZZ \\
g_4 &= ZXIXZ
\end{aligned}$$

**Logical operators:**
$$\overline{X} = XXXXX, \quad \overline{Z} = ZZZZZ$$

This code saturates the quantum Hamming bound.

### 7.2 Graph States

A **graph state** $$|G\rangle$$ is a stabilizer state associated with a graph $$G = (V, E)$$:

1. Start with $$|+\rangle^{\otimes |V|}$$
2. Apply $$CZ_{ij}$$ for each edge $$(i,j) \in E$$

**Stabilizer:** For each vertex $$v$$:
$$g_v = X_v \prod_{u \in N(v)} Z_u$$

where $$N(v)$$ is the neighborhood of $$v$$.

### 7.3 Measurement-Based Quantum Computation

Graph states enable **measurement-based quantum computation** (MBQC):
- Prepare a large entangled graph state (e.g., cluster state)
- Perform adaptive single-qubit measurements
- Measurement outcomes determine the computation

This is equivalent to the circuit model for universal QC.

---

## 8. Summary of Key Results

| Topic | Key Result |
|-------|------------|
| Pauli group | $$\|\mathcal{G}_n\| = 4^{n+1}$$, elements commute or anticommute |
| Stabilizer state | Unique state from $$2^n$$ commuting Paulis (not containing $$-I$$) |
| $$[[n,k,d]]$$ code | $$n-k$$ stabilizer generators, $$k$$ logical qubits |
| CSS construction | $$C_2 \subset C_1$$ gives $$[[n, k_1-k_2, \min(d_1, d_2^\perp)]]$$ |
| Steane code | $$[[7,1,3]]$$ with transversal H and CNOT |
| Gottesman-Knill | Clifford circuits classically simulable in $$O(n^2 m)$$ |

---

## 9. Exam Preparation

### Key Definitions
- [ ] Pauli group (single and n-qubit)
- [ ] Stabilizer state and stabilizer group
- [ ] Stabilizer code and generators
- [ ] CSS code construction
- [ ] Clifford group

### Key Proofs
- [ ] Stabilizer uniquely determines state
- [ ] CSS codes are valid stabilizer codes
- [ ] Gottesman-Knill theorem

### Key Calculations
- [ ] Find stabilizers for given states
- [ ] Construct CSS code from classical codes
- [ ] Update stabilizer tableau under Clifford gates

---

## References

1. Gottesman, D. "Stabilizer Codes and Quantum Error Correction" [arXiv:quant-ph/9705052](https://arxiv.org/abs/quant-ph/9705052)

2. Calderbank, A.R. & Shor, P.W. "Good Quantum Error-Correcting Codes Exist" [arXiv:quant-ph/9512032](https://arxiv.org/abs/quant-ph/9512032)

3. Nielsen & Chuang, Sections 10.4-10.5

4. Aaronson, S. & Gottesman, D. "Improved Simulation of Stabilizer Circuits" [arXiv:quant-ph/0406196](https://arxiv.org/abs/quant-ph/0406196)

---

**Word Count:** ~2600 words
**Review Guide Created:** February 10, 2026
