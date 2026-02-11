# QEC Mastery II: Practice Written Examination

## Examination Information

**Duration:** 3 hours (180 minutes)
**Total Points:** 100
**Passing Score:** 70%

**Instructions:**
1. This is a closed-book examination
2. A formula sheet is provided at the end
3. Show all work for full credit
4. Partial credit is awarded for correct reasoning
5. Manage your time carefully

---

## Section A: Stabilizer Formalism (20 points, 25 minutes)

### Problem A1 (10 points)

Consider the $$[[5,1,3]]$$ perfect code with stabilizer generators:
$$S_1 = XZZXI$$
$$S_2 = IXZZX$$
$$S_3 = XIXZZ$$
$$S_4 = ZXIXZ$$

**(a)** (3 points) Verify that these generators commute with each other.

**(b)** (3 points) Find the logical operators $$\overline{X}$$ and $$\overline{Z}$$.

**(c)** (4 points) An error $$E$$ produces syndrome $$(+1, -1, -1, +1)$$. Identify the most likely single-qubit error.

---

### Problem A2 (10 points)

**(a)** (5 points) Define the Clifford group and state its relationship to the Pauli group.

**(b)** (5 points) Prove that the Hadamard gate $$H$$ and phase gate $$S$$ together with CNOT generate the Clifford group. You may cite relevant theorems.

---

## Section B: Surface Codes (20 points, 25 minutes)

### Problem B1 (10 points)

Consider a distance-5 rotated surface code.

**(a)** (3 points) How many data qubits and how many ancilla qubits are required?

**(b)** (3 points) What are the X-type and Z-type stabilizer weights?

**(c)** (4 points) An X error chain of length 3 occurs in the bulk of the code. Describe the syndrome pattern and explain how the decoder determines the correction.

---

### Problem B2 (10 points)

**(a)** (5 points) Explain the relationship between the surface code and the toric code. What is the key structural difference?

**(b)** (5 points) For the surface code with boundary conditions, prove that $$k = 1$$ (one logical qubit is encoded) using a counting argument involving stabilizers and logical operators.

---

## Section C: Fault Tolerance (25 points, 35 minutes)

### Problem C1 (10 points)

State and outline the proof of the threshold theorem.

**(a)** (3 points) State the theorem precisely, including all assumptions.

**(b)** (4 points) Define "fault-tolerant operation" and explain why this property is essential.

**(c)** (3 points) For a distance-3 code with fault-tolerant gadgets having 100 locations, estimate the threshold $$p_{\text{th}}$$.

---

### Problem C2 (8 points)

**(a)** (4 points) State the Eastin-Knill theorem and explain its significance for universal quantum computation.

**(b)** (4 points) Describe three methods to circumvent the Eastin-Knill restriction and implement universal fault-tolerant computation.

---

### Problem C3 (7 points)

Consider the 15-to-1 magic state distillation protocol.

**(a)** (3 points) If input magic states have error $$\epsilon = 10^{-2}$$, what is the output error after one round?

**(b)** (2 points) How many rounds are needed to achieve output error $$< 10^{-12}$$?

**(c)** (2 points) What is the total number of input magic states consumed?

---

## Section D: Decoding (15 points, 20 minutes)

### Problem D1 (8 points)

**(a)** (4 points) Describe the minimum-weight perfect matching (MWPM) decoder for the surface code. Include the graph construction.

**(b)** (4 points) What is the threshold for MWPM decoding under:
   - Code capacity noise (perfect measurements)?
   - Phenomenological noise (noisy measurements)?

Explain why these differ.

---

### Problem D2 (7 points)

**(a)** (3 points) Describe the union-find decoder and state its complexity.

**(b)** (2 points) Compare the threshold of union-find to MWPM.

**(c)** (2 points) When would you prefer union-find over MWPM in practice?

---

## Section E: QLDPC Codes (20 points, 35 minutes)

### Problem E1 (8 points)

**(a)** (4 points) Define a quantum LDPC code. What constraints must the parity-check matrices satisfy?

**(b)** (4 points) Explain why achieving both constant rate ($$k = \Theta(n)$$) and linear distance ($$d = \Theta(n)$$) was a major open problem. What made it difficult?

---

### Problem E2 (6 points)

Consider the hypergraph product of two classical $$[n, k, d]$$ LDPC codes.

**(a)** (3 points) State the parameters of the resulting quantum code.

**(b)** (3 points) Prove that the distance is $$d_Q = \Theta(\sqrt{n_Q})$$, not linear in $$n_Q$$.

---

### Problem E3 (6 points)

**(a)** (3 points) State the main result of Panteleev-Kalachev (2021-2022).

**(b)** (3 points) Explain how asymptotically good QLDPC codes enable constant-overhead magic state distillation. What is the key insight?

---

## Formula Sheet

### Pauli Matrices
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### Clifford Gates
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \quad T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

### Threshold Theorem
$$p^{(L)} = p_{\text{th}} \left(\frac{p}{p_{\text{th}}}\right)^{2^L}$$

Threshold estimate: $$p_{\text{th}} \approx 1/\binom{n_{\text{loc}}}{2}$$

### Magic State Distillation
$$\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3 \quad \text{(15-to-1 protocol)}$$

Distillation exponent: $$\gamma = \log_3(15) \approx 2.46$$

### Decoder Thresholds (Surface Code)
- MWPM code capacity: $$\sim 10.3\%$$
- MWPM phenomenological: $$\sim 2.9\%$$
- Union-find code capacity: $$\sim 9.9\%$$

### Hypergraph Product
$$[[n_1 m_2 + m_1 n_2, k_1 k_2, \min(d_1, d_2)]]$$

where $$m_i = n_i - k_i$$.

### Panteleev-Kalachev
$$[[n, \Theta(n), \Theta(n)]]$$ with $$O(1)$$ stabilizer weight

---

## End of Examination

**Time Check:** You should have approximately used:
- Section A: 25 min (cumulative: 25 min)
- Section B: 25 min (cumulative: 50 min)
- Section C: 35 min (cumulative: 85 min)
- Section D: 20 min (cumulative: 105 min)
- Section E: 35 min (cumulative: 140 min)
- Review: 40 min (cumulative: 180 min)
