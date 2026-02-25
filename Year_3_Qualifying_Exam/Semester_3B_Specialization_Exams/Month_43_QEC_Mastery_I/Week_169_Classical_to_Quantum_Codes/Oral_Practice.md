# Week 169: Classical to Quantum Codes - Oral Practice

## Introduction

This document contains oral examination practice questions for Week 169 material. Questions are organized by format: short-answer (2-3 minutes), extended explanation (5-10 minutes), and deep-dive (15-20 minutes with follow-ups).

Practice these with a study partner or by recording yourself. The examiner's perspective and common follow-up questions are included.

---

## Short-Answer Questions (2-3 minutes each)

### Question 1: Why Can't We Just Copy?

**Examiner asks:** "In classical error correction, we protect information by making copies. Why doesn't this work for quantum information?"

**Key points to cover:**
- No-cloning theorem: Cannot copy an unknown quantum state
- Measurement disturbs the state
- Entanglement-based encoding instead of copying
- The encoded state is spread across multiple qubits without any single qubit containing the full information

**Model answer:** "The no-cloning theorem forbids copying unknown quantum states, so we cannot use classical repetition. Instead, quantum error correction uses entanglement to spread information across multiple qubits. The key insight is that no single physical qubit—or even small subset—contains the logical information. For example, in the Shor code, any single qubit looks maximally mixed regardless of the encoded state. We extract syndrome information through carefully designed measurements that reveal error information without disturbing the encoded data."

---

### Question 2: Error Discretization

**Examiner asks:** "Quantum errors are continuous. How can discrete error correction work?"

**Key points to cover:**
- Syndrome measurement projects onto discrete error subspaces
- Any error operator can be expanded in the Pauli basis
- The measurement discretizes the error
- Knill-Laflamme ensures this projection preserves logical information

**Model answer:** "While physical errors are indeed continuous, the syndrome measurement projects the system onto discrete error subspaces. Any error operator can be written as a linear combination of Pauli operators. When we measure the syndrome, we effectively perform a projective measurement that collapses this superposition onto a specific Pauli error, which we can then correct. The Knill-Laflamme conditions guarantee this projection doesn't reveal or disturb the logical information."

---

### Question 3: What Does Distance Mean?

**Examiner asks:** "What is the distance of a quantum code and why does it matter?"

**Key points to cover:**
- Minimum weight of undetectable error
- Equivalently: minimum weight logical operator
- $$d = 2t + 1$$ corrects $$t$$ errors
- Trade-off with encoding rate

**Model answer:** "The distance $$d$$ of an $$[[n, k, d]]$$ code is the minimum weight of any operator that acts nontrivially on the code space while commuting with all stabilizers—essentially, the smallest logical operator. A distance-$$d$$ code can detect up to $$d-1$$ errors and correct $$\lfloor(d-1)/2\rfloor$$ errors. For the Shor code with $$d = 3$$, we correct any single-qubit error because even if an error occurs, its effect is distinguishable from all other single-qubit errors."

---

### Question 4: Bit-Flip vs Phase-Flip

**Examiner asks:** "Compare the three-qubit bit-flip and phase-flip codes."

**Key points to cover:**
- Bit-flip: $$|0\rangle \to |000\rangle$$, $$|1\rangle \to |111\rangle$$
- Phase-flip: $$|0\rangle \to |{+}{+}{+}\rangle$$, $$|1\rangle \to |{-}{-}{-}\rangle$$
- Related by Hadamard transform
- Each corrects one type of error but not the other
- Need both for full protection (leads to Shor code)

---

### Question 5: Degenerate Codes

**Examiner asks:** "What is a degenerate code? Give an example."

**Key points to cover:**
- Different errors act identically on code space
- $$E_a|\psi\rangle = E_b|\psi\rangle$$ for all codewords
- Can exceed quantum Hamming bound
- Example: Surface code where $$Z$$ errors on boundary act as identity

---

## Extended Explanation Questions (5-10 minutes)

### Question 6: Knill-Laflamme Theorem

**Examiner asks:** "State and prove the Knill-Laflamme quantum error correction conditions."

**Structure your answer:**

1. **Statement** (1-2 min):
   - Define the setup: code space $$C$$ with basis $$\{|\psi_i\rangle\}$$, error set $$\{E_a\}$$
   - State the condition: $$\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle = C_{ab} \delta_{ij}$$
   - Explain: errors preserve orthogonality and act uniformly

2. **Necessity** (2-3 min):
   - If recovery exists, apply it to $$E_a|\psi_i\rangle$$
   - Recovery must distinguish $$i$$ regardless of error $$a$$
   - This requires orthogonality of error subspaces or identical action

3. **Sufficiency** (2-3 min):
   - Construct the recovery operation
   - Define syndrome measurement from $$\{E_a P\}$$
   - Apply correction based on syndrome
   - Show original state is recovered

**Follow-up questions:**
- "What if the matrix $$C$$ is not diagonal?"
- "How does this relate to the error discretization theorem?"
- "Can you give a code that violates one condition?"

---

### Question 7: Constructing the Shor Code

**Examiner asks:** "Walk me through the construction of the Shor code."

**Structure your answer:**

1. **Motivation** (1 min):
   - Need to correct $$X$$, $$Z$$, and $$Y$$ errors
   - Bit-flip code handles $$X$$, phase-flip handles $$Z$$
   - Concatenation combines both

2. **Construction** (3-4 min):
   - Start with phase-flip code: $$|0\rangle \to |{+}{+}{+}\rangle$$
   - Each $$|{+}\rangle$$ expanded: $$|{+}\rangle \to (|000\rangle + |111\rangle)/\sqrt{2}$$
   - Write explicit codewords
   - Count: 9 physical qubits, 1 logical qubit

3. **Error Correction** (2-3 min):
   - $$X$$ errors detected within blocks by $$Z_iZ_j$$
   - $$Z$$ errors detected between blocks by $$X_1X_2X_3 \cdot X_4X_5X_6$$
   - $$Y = iXZ$$ handled by correcting both

4. **Properties** (1 min):
   - Distance 3
   - Does not saturate Hamming bound
   - First quantum code to correct all single-qubit errors

**Follow-up questions:**
- "What are the stabilizer generators?"
- "Find a weight-3 logical operator"
- "How would you generalize this to higher distance?"

---

### Question 8: Classical vs Quantum Error Correction

**Examiner asks:** "Compare and contrast classical and quantum error correction."

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| Information | Bits (0 or 1) | Qubits (superpositions) |
| Errors | Bit flips | Pauli errors (X, Z, Y) |
| Detection | Parity checks | Stabilizer measurements |
| Correction | Flip bits | Apply Pauli corrections |
| Cloning | Copy freely | No-cloning theorem |
| Rate bounds | Singleton: $$k \leq n-d+1$$ | Quantum Singleton: $$k \leq n-2d+2$$ |
| Example | [7,4,3] Hamming | [[9,1,3]] Shor |

**Key differences to emphasize:**
- Quantum must correct phase errors (no classical analog)
- Measurement disturbs quantum states (syndrome must not reveal logical info)
- Quantum codes are more constrained (factor of 2 in Singleton)

---

### Question 9: Verifying Error Correction

**Examiner asks:** "How would you verify that a proposed code corrects a given set of errors?"

**Procedure:**

1. **Identify the code space:**
   - Write down codeword basis $$\{|\psi_i\rangle\}$$
   - Verify orthonormality

2. **List all errors to correct:**
   - Typically all Paulis up to weight $$t$$
   - For $$n$$ qubits, weight $$t$$: $$\sum_{j=0}^{t}\binom{n}{j}3^j$$ operators

3. **Check Knill-Laflamme for each pair:**
   - Compute $$\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle$$
   - Verify $$= 0$$ when $$i \neq j$$
   - Verify diagonal elements equal across $$i$$

4. **Alternative: Stabilizer method (preview):**
   - Check errors anticommute with at least one stabilizer
   - Check products of correctable errors are not in stabilizer group

**Example:** Verify Shor code corrects $$X_1, Z_4$$.

---

## Deep-Dive Questions (15-20 minutes)

### Question 10: From Knill-Laflamme to Recovery

**Examiner asks:** "Given a code satisfying Knill-Laflamme, explicitly construct the recovery operation."

**Full derivation:**

1. **Setup:**
   - Code projector: $$P = \sum_i |\psi_i\rangle\langle\psi_i|$$
   - Error operators: $$\{E_a\}$$ satisfying $$PE_a^\dagger E_b P = C_{ab}P$$

2. **Diagonalize the error matrix:**
   - $$C$$ is Hermitian, so diagonalize: $$C = VDV^\dagger$$
   - Define canonical errors: $$F_i = \sum_a V_{ai}E_a$$
   - Then $$PF_i^\dagger F_j P = d_i \delta_{ij} P$$

3. **Define syndrome subspaces:**
   - $$\Pi_i = F_i P F_i^\dagger / d_i$$ projects onto the image of error $$F_i$$
   - These are orthogonal by Knill-Laflamme

4. **Construct recovery:**
   - Measure $$\{\Pi_i\}$$ to determine which canonical error occurred
   - Apply $$F_i^\dagger$$ (suitably normalized) to reverse the error
   - Result: $$\mathcal{R}(\rho) = \sum_i F_i^\dagger \Pi_i \rho \Pi_i F_i / d_i$$

5. **Verify:**
   - For encoded state with error: $$\mathcal{R}(E_a \rho E_a^\dagger) = \rho$$

**Follow-ups:**
- "What if the error is outside the correctable set?"
- "How does degeneracy change the recovery?"
- "Prove the recovery is CPTP"

---

### Question 11: Quantum Singleton Bound Proof

**Examiner asks:** "Prove the quantum Singleton bound: $$k \leq n - 2d + 2$$."

**Detailed proof:**

1. **Setup:**
   - $$[[n, k, d]]$$ code encoding $$k$$ logical qubits in $$n$$ physical
   - Split qubits into sets $$A$$ (first $$d-1$$) and $$B$$ (remaining $$n-d+1$$)

2. **Key lemma:**
   - Any $$d-1$$ qubits carry no information about the logical state
   - Proof: Errors on the complementary $$n-d+1$$ qubits have weight $$\leq n-d+1 < n - (d-1) = $$ ... (need $$d$$ to affect logical info)

   Actually: errors of weight $$< d$$ are correctable, so they don't affect logical info. Tracing out $$d-1$$ qubits is equivalent to an "error" on those qubits; since we can correct weight $$d-1$$ errors, the remaining qubits contain full info.

3. **Apply no-cloning:**
   - If $$A$$ (size $$d-1$$) had any information, and $$B$$ (size $$n-d+1$$) has full information, we could clone by preparing the state, measuring $$B$$, then measuring $$A$$
   - Contradiction unless $$A$$ has zero logical information

4. **Dimension counting:**
   - $$A$$ has $$2^{d-1}$$ dimensions, all used for "syndrome" (error subspaces)
   - $$B$$ must encode $$2^k$$ dimensional code space
   - But $$B$$ has $$2^{n-d+1}$$ dimensions
   - Error subspaces on $$B$$ also need $$2^{d-1}$$ factor
   - So: $$2^k \cdot 2^{d-1} \leq 2^{n-d+1}$$
   - Thus: $$k + d - 1 \leq n - d + 1$$
   - Therefore: $$k \leq n - 2d + 2$$

---

### Question 12: Design a New Code

**Examiner asks:** "Design a [[5, 1, 3]] code and verify it saturates the quantum Hamming bound."

**This is a research-level question. Approach:**

1. **Hamming bound check:**
   $$2^1 \cdot (1 + 5 \cdot 3) = 2 \cdot 16 = 32 = 2^5$$ ✓

2. **Constraints:**
   - Need 4 independent stabilizers (generators) for $$5 - 1 = 4$$
   - Each stabilizer in 5-qubit Pauli group
   - Stabilizers must commute
   - All weight-1 and weight-2 Paulis must anticommute with some stabilizer

3. **The perfect [[5,1,3]] code:**
   Stabilizers:
   - $$S_1 = XZZXI$$
   - $$S_2 = IXZZX$$
   - $$S_3 = XIXZZ$$
   - $$S_4 = ZXIXZ$$

4. **Verify:**
   - All pairs commute (check)
   - Any weight-1 Pauli anticommutes with at least one $$S_i$$ (check)
   - Any weight-2 Pauli anticommutes with at least one $$S_i$$ (check)

5. **Find codewords:**
   - Solve for simultaneous $$+1$$ eigenspace of all $$S_i$$

---

## Tips for Oral Exams

### General Strategies

1. **Start with definitions.** Even if you think the examiner knows, stating definitions shows you know them precisely.

2. **Draw pictures.** Error correction is geometric—use diagrams for code spaces, error subspaces, syndromes.

3. **Give examples.** Abstract statements become concrete with examples. "For instance, in the Shor code..."

4. **Admit uncertainty.** "I'm not certain about the exact coefficient, but the structure is..." is better than guessing.

5. **Connect to the big picture.** Examiners like seeing that you understand why something matters.

### Common Mistakes to Avoid

1. **Confusing detection with correction.** A distance-$$d$$ code detects $$d-1$$ errors but only corrects $$\lfloor(d-1)/2\rfloor$$.

2. **Forgetting phase errors.** The bit-flip code doesn't work for arbitrary errors!

3. **Misremembering Singleton bounds.** Classical: $$k \leq n - d + 1$$. Quantum: $$k \leq n - 2d + 2$$.

4. **Ignoring degeneracy.** Not all codes are non-degenerate; degeneracy can be advantageous.

5. **Hand-waving error discretization.** Be precise about how measurement projects onto Pauli errors.

### What Examiners Look For

- **Precision:** Can you state definitions and theorems exactly?
- **Understanding:** Can you explain *why* something is true, not just *that* it's true?
- **Connections:** Do you see how concepts relate to each other?
- **Problem-solving:** Can you apply knowledge to novel situations?
- **Communication:** Can you explain complex ideas clearly?

---

## Mock Oral Exam Scenario

**Setup:** 30-minute oral exam on Week 169 material

**First 10 minutes:**
- "State the Knill-Laflamme conditions" (2 min)
- "Why are quantum codes harder to construct than classical?" (3 min)
- "Explain the Shor code construction" (5 min)

**Middle 10 minutes:**
- "Prove the quantum Singleton bound" (5 min)
- "What is error discretization and why does it matter?" (5 min)

**Final 10 minutes:**
- "Design a syndrome measurement circuit for the bit-flip code" (3 min)
- "How would you verify Knill-Laflamme computationally?" (3 min)
- "What questions do you have about error correction?" (4 min)

---

**Oral Practice Document Created:** February 9, 2026
