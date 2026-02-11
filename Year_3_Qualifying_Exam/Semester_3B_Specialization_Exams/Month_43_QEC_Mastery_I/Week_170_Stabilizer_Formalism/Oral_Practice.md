# Week 170: Stabilizer Formalism - Oral Practice

## Introduction

This document contains oral examination practice questions for Week 170 material on the stabilizer formalism. Questions progress from foundational concepts to advanced applications including CSS codes and the Gottesman-Knill theorem.

---

## Short-Answer Questions (2-3 minutes each)

### Question 1: What is a Stabilizer State?

**Examiner asks:** "Define a stabilizer state and give an example."

**Key points to cover:**
- State $$|\psi\rangle$$ stabilized by abelian subgroup $$\mathcal{S} \subset \mathcal{G}_n$$
- $$S|\psi\rangle = |\psi\rangle$$ for all $$S \in \mathcal{S}$$
- $$-I \notin \mathcal{S}$$, $$|\mathcal{S}| = 2^n$$ for unique state
- Example: Bell state stabilized by $$\langle X_1X_2, Z_1Z_2 \rangle$$

**Model answer:** "A stabilizer state is a quantum state that is the simultaneous +1 eigenstate of an abelian subgroup of the Pauli group. For an n-qubit stabilizer state, this group has order $$2^n$$ and is specified by n independent commuting generators. For example, the Bell state $$|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$$ is stabilized by $$X_1X_2$$ and $$Z_1Z_2$$. We can verify: $$X_1X_2|\Phi^+\rangle = |\Phi^+\rangle$$ and $$Z_1Z_2|\Phi^+\rangle = |\Phi^+\rangle$$."

---

### Question 2: Stabilizer vs Logical Operators

**Examiner asks:** "What is the difference between a stabilizer and a logical operator for a quantum code?"

**Key points:**
- Stabilizers: fix all codewords (act as identity on code space)
- Logical operators: act non-trivially on code space
- Both commute with stabilizer group
- Logical = centralizer modulo stabilizer

**Model answer:** "Both stabilizers and logical operators commute with the stabilizer group, but they act differently on the code space. Stabilizers are elements of the stabilizer group itself and act as identity on every codeword. Logical operators are in the centralizer of the stabilizer but not in the stabilizer itself—they act non-trivially, implementing logical gates on the encoded information. For example, in the bit-flip code, $$Z_1Z_2$$ is a stabilizer (acts as identity), while $$X_1X_2X_3$$ is a logical $$\overline{X}$$ operator (flips the logical qubit)."

---

### Question 3: Why CSS Codes?

**Examiner asks:** "What advantages do CSS codes provide over general stabilizer codes?"

**Key points:**
- Separate X and Z error correction
- Built from classical codes (leverage existing theory)
- Transversal CNOT between code blocks
- Often have transversal Hadamard
- Easier decoding algorithms

---

### Question 4: The Gottesman-Knill Theorem

**Examiner asks:** "State the Gottesman-Knill theorem and explain its significance."

**Key points:**
- Clifford circuits + stabilizer inputs + Pauli measurements = classically simulable
- Polynomial time simulation: $$O(n^2 m)$$
- Implies Clifford gates alone don't give quantum advantage
- Need non-Clifford gates (like T) for universality

**Model answer:** "The Gottesman-Knill theorem states that quantum circuits consisting of Clifford gates (H, S, CNOT), stabilizer state inputs, and measurements in the computational basis can be efficiently simulated on a classical computer. The simulation runs in time $$O(n^2 m)$$ for $$n$$ qubits and $$m$$ gates, using the stabilizer tableau representation. This is significant because it shows that Clifford gates alone cannot provide quantum computational advantage—we need non-Clifford gates like the T gate. The T gate, when added to Clifford gates, gives a universal gate set but breaks the efficient classical simulation."

---

### Question 5: Syndrome Measurement

**Examiner asks:** "How does syndrome measurement work in a stabilizer code?"

**Key points:**
- Measure eigenvalue of each stabilizer generator
- Syndrome = pattern of $$\pm 1$$ eigenvalues
- Error anticommutes with generator $$\Leftrightarrow$$ eigenvalue flips to $$-1$$
- Syndrome reveals error without revealing logical state

---

## Extended Explanation Questions (5-10 minutes)

### Question 6: The Steane Code

**Examiner asks:** "Describe the Steane code, its construction, and its properties."

**Structure your answer:**

1. **CSS Construction** (2 min):
   - Built from Hamming code $$[7, 4, 3]$$
   - $$CSS(C, C)$$ where $$C$$ is Hamming
   - Self-orthogonality: $$C^\perp \subset C$$

2. **Parameters** (1 min):
   - $$[[7, 1, 3]]$$: 7 qubits, 1 logical qubit, distance 3
   - Corrects any single-qubit error

3. **Stabilizers** (2 min):
   - 3 X-type: IIIXXXX, IXXIIXX, XIXIXIX
   - 3 Z-type: IIIZZZZ, IZZIIZZ, ZIZIZIZ
   - Pattern follows Hamming parity-check

4. **Transversal Gates** (2 min):
   - $$\overline{X} = X^{\otimes 7}$$, $$\overline{Z} = Z^{\otimes 7}$$
   - $$\overline{H} = H^{\otimes 7}$$ (because CSS structure)
   - $$\overline{CNOT}$$ transversal between two blocks

5. **Comparison with Shor** (1 min):
   - More efficient (7 vs 9 qubits)
   - Better gate set (transversal H)

**Follow-up questions:**
- "Why doesn't the Steane code have a transversal T gate?"
- "How would you implement a T gate fault-tolerantly?"

---

### Question 7: Proving a Code Corrects Errors

**Examiner asks:** "How do you verify that a stabilizer code corrects a given set of errors?"

**Two approaches:**

**Approach 1: Direct verification**
1. List all errors $$\{E_a\}$$ to correct
2. For each error, compute syndrome (commutation with generators)
3. Verify distinct correctable errors give distinct syndromes
4. Verify no correctable error is a logical operator

**Approach 2: Distance argument**
1. Find code distance $$d$$ = minimum weight logical operator
2. Any weight $$< d$$ error anticommutes with some stabilizer OR is in stabilizer
3. Code corrects $$\lfloor(d-1)/2\rfloor$$ errors

**Example with [[5,1,3]] code:**
- Check all weight-1 Paulis anticommute with some generator
- Check all weight-2 Paulis anticommute with some generator
- Find weight-3 logical to confirm $$d = 3$$

---

### Question 8: Stabilizer Tableau Operations

**Examiner asks:** "Explain how to track a stabilizer state through a Clifford circuit using the tableau representation."

**Cover these points:**

1. **Tableau Structure:**
   - n rows for n stabilizer generators
   - Each row: X-part (n bits), Z-part (n bits), phase (2 bits)
   - Row encodes $$i^r X^{x_1}Z^{z_1} \otimes \cdots$$

2. **Gate Updates:**

   | Gate | Update Rule |
   |------|-------------|
   | H on qubit j | Swap X and Z columns for j |
   | S on qubit j | $$z_{ij} \leftarrow z_{ij} \oplus x_{ij}$$, update phase |
   | CNOT c→t | $$x_{it} \leftarrow x_{ic} \oplus x_{it}$$, $$z_{ic} \leftarrow z_{ic} \oplus z_{it}$$ |

3. **Measurement:**
   - Check if measured Pauli commutes with all generators
   - If yes: deterministic outcome
   - If no: random outcome, update tableau

**Demonstrate with example:**
$$|00\rangle \xrightarrow{H_1} \xrightarrow{CNOT_{12}}$$ Bell state

---

### Question 9: CSS Code Construction

**Examiner asks:** "Derive the stabilizer generators and code parameters for a CSS code from two classical codes."

**General construction:**

Given $$C_1 = [n, k_1, d_1]$$ and $$C_2 = [n, k_2, d_2]$$ with $$C_2 \subset C_1$$:

1. **Z-stabilizers:** From parity-check matrix $$H_1$$ of $$C_1$$
   - Each row $$h$$ gives $$Z^{h_1} \otimes \cdots \otimes Z^{h_n}$$
   - Number: $$n - k_1$$ generators

2. **X-stabilizers:** From generator matrix $$G_2^\perp$$ of $$C_2^\perp$$
   - Each row $$g$$ gives $$X^{g_1} \otimes \cdots \otimes X^{g_n}$$
   - Number: $$n - k_2$$ generators
   - But need $$C_2^\perp \subset C_1$$, giving $$k_2$$ X-generators

3. **Parameters:**
   - $$n$$ physical qubits
   - $$k = k_1 - k_2$$ logical qubits
   - $$d = \min(d_1, d_2^\perp)$$

4. **Commutation:**
   - X-stabilizers from $$C_2^\perp$$
   - Z-stabilizers from $$H_1$$
   - $$C_2^\perp \subset C_1 \Rightarrow H_1 \cdot c = 0$$ for $$c \in C_2^\perp$$
   - Thus $$X^c$$ and $$Z^h$$ commute

---

## Deep-Dive Questions (15-20 minutes)

### Question 10: Full Stabilizer Code Analysis

**Examiner asks:** "Analyze the [[5,1,3]] perfect code completely."

**Complete analysis:**

1. **Stabilizers:**
   $$g_1 = XZZXI$$
   $$g_2 = IXZZX$$
   $$g_3 = XIXZZ$$
   $$g_4 = ZXIXZ$$

2. **Verify commutation:** Check all 6 pairs
   - Count X-Z overlaps at each position
   - Even number $$\Rightarrow$$ commute

3. **Find logical operators:**
   $$\overline{X} = XXXXX$$ (check: commutes with all $$g_i$$, not in $$\mathcal{S}$$)
   $$\overline{Z} = ZZZZZ$$

4. **Verify distance 3:**
   - Weight-1: $$X_1$$ anticommutes with $$g_4$$ (check)
   - Weight-2: $$X_1X_2$$ - verify anticommutes with some $$g_i$$
   - Weight-3 logical exists: equivalent to $$\overline{X}$$ mod stabilizer

5. **Hamming bound:**
   $$2^1(1 + 5 \cdot 3) = 32 = 2^5$$ $$\checkmark$$ Saturates!

6. **Codewords:**
   Find explicit $$|0_L\rangle$$ and $$|1_L\rangle$$ by solving stabilizer eigenvalue equations.

**Follow-ups:**
- "Is this code CSS?"
- "What gates are transversal?"
- "How does this compare to Steane?"

---

### Question 11: Gottesman-Knill Proof Outline

**Examiner asks:** "Prove the Gottesman-Knill theorem."

**Proof structure:**

1. **Stabilizer representation:**
   - n-qubit stabilizer state $$\equiv$$ n generators
   - Each generator: 2n bits + 2 phase bits
   - Total: $$O(n^2)$$ classical bits

2. **Clifford gates preserve stabilizer structure:**
   - Clifford $$U$$ maps Paulis to Paulis: $$UPU^\dagger \in \mathcal{G}_n$$
   - If $$S|\psi\rangle = |\psi\rangle$$, then $$(USU^\dagger)(U|\psi\rangle) = U|\psi\rangle$$
   - Update each generator in $$O(n)$$ time

3. **Measurement simulation:**
   - Measure qubit $$k$$ in Z basis = project onto $$Z_k = \pm 1$$
   - Case 1: $$Z_k \in \mathcal{S}$$ or $$-Z_k \in \mathcal{S}$$ $$\Rightarrow$$ deterministic
   - Case 2: $$Z_k$$ anticommutes with some $$g_i$$ $$\Rightarrow$$ random, replace $$g_i$$ with $$Z_k$$

4. **Complexity analysis:**
   - Initial state: $$O(n^2)$$ bits
   - Each gate: $$O(n)$$ time
   - Each measurement: $$O(n^2)$$ time (Gaussian elimination)
   - Total: $$O(n^2 m)$$

**Why T gates break this:**
$$TXT^\dagger = e^{i\pi/4}(X + Y)/\sqrt{2}$$ - not a Pauli!

---

### Question 12: Design Exercise

**Examiner asks:** "Design a stabilizer code with specific properties."

**Example task:** Design a [[6, 2, 2]] code.

**Solution approach:**

1. **Parameter constraints:**
   - 6 qubits, 2 logical qubits
   - Need 4 stabilizer generators
   - Distance 2 (detects 1 error)

2. **Propose generators:**
   Start with simple structure:
   $$g_1 = X_1X_2X_3X_4$$
   $$g_2 = X_3X_4X_5X_6$$
   $$g_3 = Z_1Z_2Z_3Z_4$$
   $$g_4 = Z_3Z_4Z_5Z_6$$

3. **Verify commutation:**
   All X-type commute with each other $$\checkmark$$
   All Z-type commute with each other $$\checkmark$$
   $$g_1$$ and $$g_3$$: 4 X-Z overlaps $$\Rightarrow$$ commute $$\checkmark$$

4. **Find logical operators:**
   Must commute with all generators, not in stabilizer.
   $$\overline{X}_1 = X_1X_2$$, $$\overline{Z}_1 = Z_1Z_2$$
   $$\overline{X}_2 = X_5X_6$$, $$\overline{Z}_2 = Z_5Z_6$$

5. **Verify distance:**
   Weight-2 logical exists (e.g., $$X_1X_2$$) $$\Rightarrow d = 2$$

---

## Mock Oral Scenarios

### Scenario A: Fundamentals Focus (20 min)

1. "What is the Pauli group?" (2 min)
2. "Define stabilizer state and give three examples" (3 min)
3. "How do stabilizer codes generalize repetition codes?" (4 min)
4. "Explain syndrome measurement with the bit-flip code" (4 min)
5. "State and explain the Gottesman-Knill theorem" (5 min)
6. "Questions about your answers" (2 min)

### Scenario B: CSS Codes Focus (20 min)

1. "What is a CSS code?" (2 min)
2. "Construct the Steane code from the Hamming code" (6 min)
3. "Prove CSS X and Z stabilizers commute" (4 min)
4. "What transversal gates does Steane support? Why?" (4 min)
5. "Compare CSS and non-CSS codes" (3 min)
6. "Questions" (1 min)

### Scenario C: Computational Focus (20 min)

1. "Explain stabilizer tableaux" (3 min)
2. "Track $$|00\rangle$$ through H-CNOT-S circuit" (5 min)
3. "How is measurement simulated classically?" (4 min)
4. "Why are T gates hard to simulate?" (3 min)
5. "What is the Clifford hierarchy?" (4 min)
6. "Questions" (1 min)

---

## Common Mistakes to Avoid

1. **Forgetting phase factors:** Stabilizers can have $$-1$$ phase (e.g., $$|-\rangle$$ stabilized by $$-X$$)

2. **Confusing centralizer and stabilizer:** Logical operators are in centralizer but NOT stabilizer

3. **CSS misconception:** Not all stabilizer codes are CSS

4. **Commutation errors:** When checking if Paulis commute, count ALL X-Z overlaps

5. **Tableau phase updates:** S gate and measurements affect phases non-trivially

---

## Key Formulas to Know

| Concept | Formula |
|---------|---------|
| Pauli commutation | $$PQ = (-1)^{\text{# anticomm positions}} QP$$ |
| Code dimension | $$k = n - (\text{# generators})$$ |
| Stabilizer group size | $$\|\mathcal{S}\| = 2^{n-k}$$ |
| CSS parameters | $$[[n, k_1 - k_2, \min(d_1, d_2^\perp)]]$$ |
| GK complexity | $$O(n^2 m)$$ |

---

**Oral Practice Document Created:** February 10, 2026
