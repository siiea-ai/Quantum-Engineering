# Week 161: Quantum Gates - Oral Examination Practice

## Discussion Questions and Mock Exam Scenarios

---

## Part I: Conceptual Discussion Questions

### Question 1: Fundamentals of Quantum Gates
**Examiner:** "Explain what makes a valid quantum gate and why unitarity is required."

**Expected Discussion Points:**
- Quantum gates must preserve probability (normalization of state vectors)
- Unitarity ensures reversibility of quantum evolution
- Connection to Schrodinger equation: $U = e^{-iHt/\hbar}$
- Physical gates are approximations to ideal unitaries
- Distinguishing gates from measurements (non-unitary)

**Follow-up:** "Can you give an example of a non-unitary operation in quantum computing?"
- Measurement, reset, amplitude damping
- These require interaction with environment

---

### Question 2: The Bloch Sphere
**Examiner:** "Draw the Bloch sphere and explain how single-qubit gates act on it."

**Expected Response:**
- Draw sphere with $|0\rangle$ at north pole, $|1\rangle$ at south pole
- Mark $|+\rangle, |-\rangle$ on x-axis, $|+i\rangle, |-i\rangle$ on y-axis
- Explain that any single-qubit gate is a rotation
- Pauli gates: $\pi$ rotations about x, y, z axes
- Hadamard: $\pi$ rotation about (x+z)/sqrt(2) axis

**Follow-up:** "Why can mixed states be represented inside the sphere?"
- Mixed states have Bloch vectors with $|\vec{r}| < 1$
- Maximally mixed state at center
- Surface corresponds to pure states only

---

### Question 3: The CNOT Gate
**Examiner:** "Explain the CNOT gate and why it's important for universal quantum computation."

**Expected Discussion Points:**
- Definition: flips target if control is $|1\rangle$
- Matrix representation and action on computational basis
- Creates entanglement: $\text{CNOT}|+\rangle|0\rangle = |\Phi^+\rangle$
- Combined with single-qubit gates, enables universal computation
- Key for error correction (syndrome extraction)
- Every algorithm uses entangling gates like CNOT

**Follow-up:** "How would you implement CNOT on a superconducting quantum computer?"
- Cross-resonance gate
- Echoed cross-resonance for higher fidelity
- Typical gate time: 200-400 ns

---

### Question 4: Universality
**Examiner:** "What does it mean for a gate set to be universal? Give examples of universal and non-universal gate sets."

**Expected Response:**

**Universal sets:**
- $\{H, T, \text{CNOT}\}$ - standard discrete set
- $\{R_x(\theta), R_y(\theta), \text{CNOT}\}$ - continuous rotations
- Any set generating dense subgroup + entangling gate

**Non-universal sets:**
- $\{H, S, \text{CNOT}\}$ - Clifford only (classically simulable)
- Single-qubit gates alone (no entanglement)
- Classical reversible gates (no superposition)

**Key insight:** Need both superposition AND entanglement capabilities

---

### Question 5: Solovay-Kitaev Theorem
**Examiner:** "State the Solovay-Kitaev theorem and explain its significance."

**Expected Response:**

**Statement:** If $\mathcal{G}$ is a finite gate set generating a dense subgroup of $SU(2)$, then any $U \in SU(2)$ can be approximated to error $\epsilon$ using $O(\log^c(1/\epsilon))$ gates, where $c \approx 4$.

**Significance:**
1. Universal gate sets are truly universal (no forbidden unitaries)
2. Gate set choice adds only polylogarithmic overhead
3. Efficient classical algorithm to find approximation
4. Justifies using finite gate sets in theoretical analysis

**Follow-up:** "Why is the Ross-Selinger algorithm better for Clifford+T?"
- Exploits specific structure of Clifford+T
- Achieves $T$-count $\approx 3\log_2(1/\epsilon)$ vs $O(\log^4(1/\epsilon))$
- Nearly optimal (lower bound is $\log_2(1/\epsilon)$)

---

## Part II: Technical Derivation Questions

### Question 6: ZYZ Decomposition
**Examiner:** "Prove that any single-qubit unitary can be written in ZYZ form."

**Expected Derivation:**

1. General $U \in SU(2)$ has form:
$$U = \begin{pmatrix} \alpha & -\beta^* \\ \beta & \alpha^* \end{pmatrix}, \quad |\alpha|^2 + |\beta|^2 = 1$$

2. Write $\alpha = e^{i\phi}\cos(\theta/2)$, $\beta = e^{i\psi}\sin(\theta/2)$

3. The ZYZ form is:
$$R_z(\gamma_1)R_y(\theta)R_z(\gamma_2) = \begin{pmatrix} e^{-i(\gamma_1+\gamma_2)/2}\cos(\theta/2) & -e^{-i(\gamma_1-\gamma_2)/2}\sin(\theta/2) \\ e^{i(\gamma_1-\gamma_2)/2}\sin(\theta/2) & e^{i(\gamma_1+\gamma_2)/2}\cos(\theta/2) \end{pmatrix}$$

4. Match coefficients to determine $\gamma_1, \gamma_2, \theta$

**Follow-up:** "What's the geometric interpretation?"
- Euler angles for 3D rotations
- $SU(2)$ double covers $SO(3)$

---

### Question 7: Proving CNOT + Single-Qubit is Universal
**Examiner:** "Outline how you would prove that CNOT together with arbitrary single-qubit gates is universal for quantum computation."

**Expected Proof Outline:**

**Step 1:** Any two-qubit unitary can be decomposed using the KAK decomposition:
$$U = (A_1 \otimes A_2) \cdot e^{i(c_1 XX + c_2 YY + c_3 ZZ)} \cdot (B_1 \otimes B_2)$$

**Step 2:** The non-local part $e^{i(c_1 XX + c_2 YY + c_3 ZZ)}$ can be implemented with 3 CNOTs plus single-qubit gates.

**Step 3:** Any $n$-qubit unitary can be decomposed into a product of two-qubit unitaries (e.g., Cosine-Sine decomposition or recursive methods).

**Step 4:** Therefore, CNOT + single-qubit gates suffices.

---

### Question 8: Clifford Group Structure
**Examiner:** "What is the Clifford group and why is it important?"

**Expected Discussion:**

**Definition:** $C_n = \{U : UPU^\dagger \in \mathcal{P}_n \text{ for all } P \in \mathcal{P}_n\}$

**Properties:**
- Generated by $\{H, S, \text{CNOT}\}$
- Finite group (up to global phase)
- Single-qubit: 24 elements (octahedral symmetry)
- Stabilizer formalism for efficient classical simulation

**Importance:**
- Gottesman-Knill theorem: Clifford circuits classically simulable
- Foundation for error correction (stabilizer codes)
- T gate is "magic" - breaks classical simulability

---

## Part III: Problem-Solving Under Pressure

### Scenario 1: Unknown Gate Analysis
**Examiner:** "I give you a gate $G$ and tell you it satisfies $G^2 = I$ and $\text{Tr}(G) = 0$. What can you conclude about $G$?"

**Expected Analysis:**
- $G^2 = I$ implies eigenvalues are $\pm 1$
- $\text{Tr}(G) = 0$ implies equal number of $+1$ and $-1$ eigenvalues
- For single qubit: $G$ must be a Pauli matrix (up to global phase)
- Specifically: $G \in \{X, Y, Z\}$ or rotations by $\pi$

**Follow-up:** "What if $\text{Tr}(G) = \sqrt{2}$?"
- Then one eigenvalue is 1 (from $G^2 = I$)
- Trace $= 1 + e^{i\phi} = \sqrt{2}$ implies $\phi = \pi/4$
- But this contradicts $G^2 = I$!
- Therefore, no such gate exists

---

### Scenario 2: Circuit Analysis
**Examiner:** "Consider the circuit $H_1 \cdot \text{CNOT}_{12} \cdot H_1$. What does this implement?"

**Expected Analysis:**
$$H_1 \cdot \text{CNOT}_{12} \cdot H_1 = (H \otimes I)(|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X)(H \otimes I)$$

Using $H|0\rangle = |+\rangle$, $H|1\rangle = |-\rangle$:
$$= |+\rangle\langle +| \otimes I + |-\rangle\langle -| \otimes X$$

Converting: This is a controlled-X from the perspective of the X-basis on the first qubit.

Actually: $= \text{CZ}$ (controlled-Z gate)

**Verification:** Check action on computational basis states.

---

### Scenario 3: Error in Reasoning
**Examiner:** "A student claims that since $H$ and $S$ generate the Clifford group, and Clifford group is classically simulable, therefore $\{H, S, \text{CNOT}\}$ cannot be universal. What's wrong with this argument?"

**Expected Response:**
- The argument is correct! $\{H, S, \text{CNOT}\}$ is NOT universal.
- The student's logic is valid.
- Common misconception: Clifford group is large but finite
- Need non-Clifford gate (like $T$) for universality
- Gottesman-Knill theorem is the key result here

---

### Scenario 4: Hardware Compilation
**Examiner:** "You need to implement a Toffoli gate on a linear chain of qubits where CNOT is only available between neighbors. How many CNOT gates do you need?"

**Expected Analysis:**
- Standard Toffoli: 6 CNOTs (for arbitrary connectivity)
- Linear nearest-neighbor constraint requires SWAP operations
- Each SWAP = 3 CNOTs
- If qubits are 1-2-3 in line, and Toffoli is on (1,3,2):
  - May need to SWAP to get required connectivity
  - Total: approximately 12-15 CNOTs depending on configuration

**Follow-up:** "How would you optimize this?"
- Use relative phase Toffoli (4 CNOTs)
- Circuit scheduling to minimize SWAPs
- Consider alternative decompositions

---

## Part IV: Research-Connected Questions

### Question 9: Fault-Tolerant Gates
**Examiner:** "Why are T gates expensive in fault-tolerant quantum computing?"

**Expected Discussion:**
- Clifford gates can be implemented transversally (bit-wise on code blocks)
- T gate cannot be transversal for most codes (Eastin-Knill theorem)
- Must use magic state distillation
- Each logical T requires many (10-100+) physical operations
- T-count is primary metric for fault-tolerant circuit cost

**Follow-up:** "What is magic state distillation?"
- Prepare many noisy $|T\rangle$ states
- Use Clifford operations + measurement to distill fewer high-fidelity states
- 15-to-1 protocol: 15 noisy → 1 clean (with overhead)

---

### Question 10: Native vs Compiled Gates
**Examiner:** "Compare the native gate sets of IBM, Google, and IonQ. How does this affect algorithm implementation?"

**Expected Discussion:**

**IBM (superconducting):**
- Native: $\{R_z(\theta), \sqrt{X}, \text{CNOT}\}$
- Virtual Z gates (nearly free)
- CNOT fidelity ~99%

**Google (superconducting):**
- Native: $\{\sqrt{X}, \sqrt{Y}, \sqrt{W}, R_z, \sqrt{\text{iSWAP}}, \text{CZ}\}$
- Sycamore uses fSim gate family
- Flexible but complex compilation

**IonQ (trapped ion):**
- Native: Single-qubit rotations + Molmer-Sorensen XX gate
- All-to-all connectivity
- Slower gates but higher fidelity

**Algorithm impact:**
- Circuit depth varies significantly between platforms
- Compilation optimizes for native gates
- Connectivity affects SWAP overhead

---

## Part V: Mock Oral Exam (15-minute simulation)

### Examiner Script

**Opening (2 min):**
"Tell me about quantum gates and what makes them different from classical logic gates."

**Technical dive (5 min):**
"Walk me through how you would prove that the Hadamard and T gates, combined with CNOT, form a universal gate set."

**Follow-up probes:**
- "What happens if we remove the T gate?"
- "How efficient is the approximation?"
- "State the Solovay-Kitaev theorem precisely."

**Application (3 min):**
"If I wanted to implement an arbitrary single-qubit rotation on IBM hardware, what would be the process?"

**Synthesis (3 min):**
"How do these ideas connect to error correction and fault tolerance?"

**Closing (2 min):**
"Any questions about what we discussed today?"

---

## Evaluation Criteria

### Excellent Performance
- Clear, precise definitions
- Correct mathematical statements
- Ability to prove key results
- Connections between topics
- Awareness of practical implications

### Acceptable Performance
- Mostly correct definitions
- Key ideas present (may lack precision)
- Can outline proofs
- Some connections made

### Needs Improvement
- Definitions unclear or incorrect
- Major gaps in understanding
- Cannot outline key proofs
- No connections between concepts

---

## Self-Practice Instructions

1. **Time yourself:** Each question should be answered in 2-3 minutes
2. **Practice at whiteboard:** Draw diagrams, write equations
3. **Record yourself:** Listen for clarity and precision
4. **Study weak areas:** Focus on questions you struggled with
5. **Practice with peers:** Take turns as examiner and examinee

---

*This oral practice guide covers the key discussion topics for qualifying exams on quantum gates. For written problem practice, see Problem_Set.md.*
