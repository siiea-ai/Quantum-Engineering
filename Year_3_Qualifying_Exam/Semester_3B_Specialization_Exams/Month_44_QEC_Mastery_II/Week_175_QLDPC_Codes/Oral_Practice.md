# Week 175: QLDPC Codes - Oral Examination Practice

## Introduction

This document provides oral examination practice questions on quantum LDPC codes. Questions range from foundational concepts to the cutting-edge Panteleev-Kalachev breakthrough.

---

## Part 1: Foundations

### Question 1.1: Classical to Quantum

**Examiner:** "What is a classical LDPC code, and why can't we directly use classical LDPC constructions for quantum codes?"

**Follow-up probes:**
- What makes classical LDPC codes "good"?
- What is the commutativity constraint for CSS codes?
- Why is this constraint hard to satisfy with sparse matrices?

**Key points:**
1. Classical LDPC: Sparse parity-check matrix, $$O(1)$$ weight rows/columns
2. Achieve $$[n, \Theta(n), \Theta(n)]$$ with BP decoding
3. CSS constraint: $$H_X H_Z^T = 0$$
4. Sparse matrices meeting this constraint tend to have low dimension $$k$$
5. This is why QLDPC was an open problem for 20+ years

---

### Question 1.2: QLDPC Definition

**Examiner:** "Define a quantum LDPC code precisely. Is the surface code a QLDPC code?"

**Follow-up probes:**
- What parameters characterize a QLDPC code?
- What is the stabilizer weight of the surface code?
- What distinguishes asymptotically good codes?

**Key points:**
1. QLDPC: Stabilizer weight $$w = O(1)$$, qubit degree $$\Delta = O(1)$$
2. Surface code: $$w = 4$$, $$\Delta = 4$$, so yes it's QLDPC
3. But surface code has $$k = O(1)$$, not "good"
4. Asymptotically good: $$k = \Theta(n)$$ AND $$d = \Theta(n)$$
5. The challenge was achieving both with LDPC structure

---

## Part 2: Hypergraph Product

### Question 2.1: Construction

**Examiner:** "Explain the hypergraph product construction and its limitations."

**Follow-up probes:**
- What are the input codes?
- What are the output code parameters?
- Why is the distance only $$O(\sqrt{n})$$?

**Key points:**
1. Input: Two classical codes $$C_1 = [n_1, k_1, d_1]$$, $$C_2 = [n_2, k_2, d_2]$$
2. Output: $$[[n_1 m_2 + m_1 n_2, k_1 k_2, \min(d_1, d_2)]]$$
3. Rate is constant: $$k/n = \Theta(1)$$
4. Distance: $$d = O(\sqrt{n})$$ because $$n \propto n_1 n_2$$ but $$d \propto n_1$$
5. Product structure limits minimum distance

---

### Question 2.2: Significance

**Examiner:** "Why was the hypergraph product an important step, even though it doesn't achieve linear distance?"

**Key points:**
1. First constant-rate QLDPC codes
2. Proved $$k = \Theta(n)$$ is achievable with LDPC structure
3. Foundation for subsequent improvements
4. Still useful: $$d = \Theta(\sqrt{n})$$ is better than many codes
5. Led to understanding of product-based constructions

---

## Part 3: Panteleev-Kalachev

### Question 3.1: The Main Result

**Examiner:** "State the Panteleev-Kalachev result and explain its significance."

**Follow-up probes:**
- How long was the QLDPC conjecture open?
- What parameters do their codes achieve?
- How does this compare to the surface code?

**Key points:**
1. Result: Explicit QLDPC with $$[[n, \Theta(n), \Theta(n)]]$$
2. Conjecture open for 20+ years (since late 1990s)
3. First asymptotically good QLDPC
4. Comparison: Surface code has $$k = O(1)$$, $$d = O(\sqrt{n})$$
5. Enables constant-overhead fault tolerance

---

### Question 3.2: Construction Intuition

**Examiner:** "Give an intuitive explanation of how Panteleev-Kalachev achieve linear distance."

**Follow-up probes:**
- What is the role of expander graphs?
- Why does non-Abelian group structure matter?
- What are Tanner codes?

**Key points:**
1. Expanders: Graphs where small sets have large boundaries
2. Expansion prevents low-weight logical operators
3. Non-Abelian groups avoid "averaging" that limits Abelian constructions
4. Tanner codes: Classical codes on graph edges with vertex constraints
5. Lifted product: Combines Tanner codes while preserving expansion

---

### Question 3.3: Proof Outline

**Examiner:** "Sketch how expansion implies linear distance in these constructions."

**Key points:**
1. Suppose logical operator has weight $$w < cn$$
2. By expansion, its syndrome has weight $$\Omega(w)$$
3. But logical operators have zero syndrome
4. Contradiction unless $$w = \Omega(n)$$
5. Key: Expansion constant must be large enough

---

## Part 4: Constant-Overhead Fault Tolerance

### Question 4.1: Distillation Revolution

**Examiner:** "How do QLDPC codes enable constant-overhead magic state distillation?"

**Follow-up probes:**
- What was the previous overhead scaling?
- What is the distillation exponent?
- Why does one round suffice?

**Key points:**
1. Previous: $$N = O(\log^\gamma(1/\epsilon))$$ with $$\gamma \approx 1-2.5$$
2. QLDPC: $$\gamma = 0$$ (constant overhead)
3. Mechanism: Linear distance gives $$\epsilon \to \epsilon^{\Theta(n)}$$ suppression
4. Constant rate: Only $$O(1)$$ overhead per output
5. One round of large code suffices for any target error

---

### Question 4.2: Resource Comparison

**Examiner:** "Compare resource requirements for surface codes vs QLDPC codes for large-scale computation."

**Key points:**
1. **Qubits per logical qubit:**
   - Surface: $$O(d^2) = O(\log^2(1/\epsilon))$$
   - QLDPC: $$O(1)$$

2. **Magic state cost:**
   - Surface: $$O(\log^\gamma(1/\epsilon))$$
   - QLDPC: $$O(1)$$

3. **Connectivity:**
   - Surface: 2D local
   - QLDPC: Long-range required

4. **Threshold:**
   - Surface: ~1%
   - QLDPC: Lower (exact TBD)

---

### Question 4.3: Practical Crossover

**Examiner:** "When would you choose QLDPC over surface codes in practice?"

**Key points:**
1. QLDPC advantages emerge at large scale (1000+ logical qubits)
2. Need very low target error ($$\epsilon < 10^{-12}$$)
3. Hardware must support non-local connectivity
4. Decoder must be fast enough
5. Current status: Surface codes for near-term, QLDPC for long-term

---

## Part 5: Open Problems

### Question 5.1: Geometric Locality

**Examiner:** "Can we achieve good QLDPC codes with geometric locality?"

**Follow-up probes:**
- What is the BPT bound?
- What's known in 3D?
- What about 4D?

**Key points:**
1. BPT bound: 2D local codes have $$kd^2 \leq O(n)$$
2. 2D: Cannot achieve both $$k = \Theta(n)$$ and $$d = \Theta(n)$$
3. 3D: Recent results achieve almost optimal parameters
4. 4D+: Asymptotically good geometrically local codes exist
5. Open: Best achievable parameters in 3D

---

### Question 5.2: Decoding

**Examiner:** "What are the challenges in decoding QLDPC codes?"

**Key points:**
1. MWPM doesn't apply directly (not for general LDPC)
2. BP struggles with short cycles in quantum codes
3. BP+OSD is current best approach
4. Threshold estimates lower than surface codes
5. Real-time decoding is challenging

---

### Question 5.3: Future Directions

**Examiner:** "What are the most important open problems in QLDPC codes?"

**Key points:**
1. Higher thresholds (matching surface code quality)
2. Efficient decoders with provable performance
3. Geometrically local constructions in low dimensions
4. Practical gate implementations
5. Experimental demonstrations at useful scale

---

## Whiteboard Exercises

### Exercise 1: Hypergraph Product

Compute the hypergraph product parameters for:
$$C_1 = C_2 = [6, 3, 2]$$

Expected answer:
- $$n = 6 \cdot 3 + 3 \cdot 6 = 36$$
- $$k = 3 \cdot 3 = 9$$
- $$d = \min(2, 2) = 2$$
- Result: $$[[36, 9, 2]]$$

---

### Exercise 2: Expansion Argument

Sketch why a $$w$$-weight error on an expander-based code cannot have zero syndrome if $$w$$ is too small.

Expected approach:
1. Error touches $$\leq w$$ qubits
2. Each qubit in $$O(1)$$ checks
3. By expansion, $$|N(S)| \geq \alpha |S|$$ for small $$S$$
4. Syndrome has $$\geq \alpha w$$ nonzero entries
5. But syndrome must be zero for logical operator

---

### Exercise 3: Rate Calculation

For a QLDPC code with:
- $$n$$ qubits
- $$m$$ X-stabilizers and $$m$$ Z-stabilizers
- Stabilizers mutually independent

Calculate the number of logical qubits $$k$$.

Expected answer:
$$k = n - \text{rank}(H_X) - \text{rank}(H_Z) = n - 2m$$ (if independent)

---

## Examination Tips

### Key Concepts to Master

1. Classical LDPC: Sparse matrices, Tanner graphs, BP decoding
2. CSS constraint: $$H_X H_Z^T = 0$$
3. Hypergraph product: Construction and parameter analysis
4. Expansion: Spectral gap, mixing, distance implications
5. Panteleev-Kalachev: Statement, significance, construction ideas
6. Constant overhead: Distillation exponent, resource comparison

### Numbers to Remember

- Surface code: $$w = 4$$, $$d = O(\sqrt{n})$$, threshold ~1%
- Hypergraph product: $$k = \Theta(n)$$, $$d = \Theta(\sqrt{n})$$
- Panteleev-Kalachev: $$k = \Theta(n)$$, $$d = \Theta(n)$$, $$w = O(1)$$
- Distillation exponent: Standard $$\gamma \approx 2.5$$, QLDPC $$\gamma = 0$$
- QLDPC conjecture: Open for ~20 years

### Communication Tips

1. Start with motivation: Why do we care about QLDPC?
2. Build intuition before formalism
3. Use diagrams: Tanner graphs, expansion pictures
4. Acknowledge what you don't know
5. Connect to practical implications
