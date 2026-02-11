# Week 162: Quantum Algorithms I - Oral Examination Practice

## Discussion Questions and Mock Exam Scenarios

---

## Part I: Conceptual Discussion Questions

### Question 1: The Oracle Model
**Examiner:** "What is the oracle model of computation and why do we use it in quantum algorithms?"

**Expected Discussion Points:**
- Oracle (black-box) model: Function access only through queries
- Abstracts away implementation details
- Allows rigorous comparison of quantum vs. classical algorithms
- Query complexity as fundamental measure
- Separates "algorithmic" speedup from "implementation" speedup
- Examples: Deutsch-Jozsa, Grover, Simon

**Follow-up:** "What are the limitations of oracle-based proofs for real-world speedups?"
- Real functions have structure that might be exploited
- Compilation from oracle to gates may lose advantage
- BQP vs. BPP separation requires non-relativizing techniques

---

### Question 2: Deutsch-Jozsa Algorithm
**Examiner:** "Explain the Deutsch-Jozsa algorithm and how it achieves exponential speedup."

**Expected Response:**

**Problem:** Distinguish constant vs. balanced function $f:\{0,1\}^n \to \{0,1\}$

**Algorithm:**
1. Prepare $|0\rangle^n|1\rangle$, apply Hadamards
2. Apply oracle (phase kickback)
3. Apply Hadamards to first register
4. Measure: $|0\rangle^n$ iff constant

**Speedup:** 1 query vs. $2^{n-1}+1$ classically

**Key insight:** Interference causes all amplitudes to cancel for balanced functions when measuring $|0\rangle^n$.

**Follow-up:** "Is this a 'real' speedup? When would you use this algorithm?"
- Limited practical applications (promise problem)
- Demonstrates quantum interference advantage
- Pedagogically important
- Special case of Hidden Subgroup Problem

---

### Question 3: Simon's Algorithm
**Examiner:** "Walk me through Simon's algorithm and explain why it's considered a precursor to Shor's."

**Expected Response:**

**Problem:** Find period $s$ where $f(x) = f(y) \Leftrightarrow x \oplus y \in \{0, s\}$

**Algorithm:**
1. Create superposition, apply oracle
2. Measure second register (collapses to $|x_0\rangle + |x_0 \oplus s\rangle$)
3. Apply Hadamard, measure (get $y$ with $y \cdot s = 0$)
4. Repeat $O(n)$ times, solve linear system

**Complexity:** $O(n)$ quantum vs. $\Omega(2^{n/2})$ classical

**Connection to Shor:**
- Both find hidden periods
- Simon: period in $(\mathbb{Z}_2)^n$
- Shor: period in $\mathbb{Z}_N$
- Shor uses QFT instead of Hadamard

---

### Question 4: Quantum Fourier Transform
**Examiner:** "Derive the circuit for the Quantum Fourier Transform."

**Expected Derivation:**

Start with definition:
$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_k e^{2\pi ijk/N}|k\rangle$$

Product representation:
$$\text{QFT}|j_1...j_n\rangle = \bigotimes_{l=1}^n \frac{|0\rangle + e^{2\pi i(0.j_{n-l+1}...j_n)}|1\rangle}{\sqrt{2}}$$

Circuit construction:
- Each qubit gets Hadamard (creates $|0\rangle + e^{2\pi i(0.j_k)}|1\rangle$)
- Controlled-$R_m$ gates add remaining phase bits
- Total: $O(n^2)$ gates

**Follow-up:** "Why is this exponentially faster than classical FFT?"
- Classical FFT: $O(N\log N) = O(n \cdot 2^n)$
- QFT: $O(n^2)$
- But: can't read out all $N$ amplitudes (measurement collapses)
- QFT is efficient for specific algorithms (phase estimation)

---

### Question 5: Phase Estimation
**Examiner:** "Explain the phase estimation algorithm and its applications."

**Expected Response:**

**Problem:** Given $U|u\rangle = e^{2\pi i\theta}|u\rangle$, estimate $\theta$

**Algorithm:**
1. Initialize $|0\rangle^t|u\rangle$
2. Hadamard on ancilla
3. Controlled-$U^{2^j}$ operations
4. Inverse QFT
5. Measure ancilla (get $\approx 2^t\theta$)

**Precision:** $t$ qubits give $t$ bits of precision

**Applications:**
- Shor's algorithm (period finding)
- Quantum chemistry (energy estimation)
- Hamiltonian simulation
- Quantum machine learning

---

## Part II: Technical Deep Dives

### Question 6: Phase Kickback
**Examiner:** "Explain phase kickback and why it's fundamental to quantum algorithms."

**Expected Response:**

When applying controlled-$U$ with control in superposition:
$$\text{C-}U|\phi\rangle|u\rangle \to (a|0\rangle + be^{i\theta}|1\rangle)|u\rangle$$

The phase "kicks back" to the control qubit.

**In Deutsch-Jozsa:**
$$O_f|x\rangle|{-}\rangle = (-1)^{f(x)}|x\rangle|{-}\rangle$$

The ancilla $|{-}\rangle$ remains unchanged; phase goes to $|x\rangle$.

**Importance:**
- Enables parallel function evaluation
- Converts function values to phases
- Phases interfere in measurement
- Foundation of all query algorithms

---

### Question 7: Classical Post-Processing in Simon's
**Examiner:** "After running Simon's quantum circuit, classical computation is needed. Explain this step."

**Expected Response:**

**Quantum outputs:** Vectors $y_1, ..., y_m$ with $y_i \cdot s = 0$

**Classical problem:** Solve homogeneous linear system over $\mathbb{F}_2$:
$$\begin{pmatrix} y_1 \\ \vdots \\ y_m \end{pmatrix} s = 0$$

**Algorithm:**
1. Gaussian elimination over $\mathbb{F}_2$
2. Reduce to row echelon form
3. Find null space (dimension 1 if $s \neq 0$)
4. Solution is $s$

**Complexity:** $O(n^3)$ for solving, polynomial overall

**Why needed:** Quantum gives random subspace samples; classical finds structure.

---

### Question 8: Approximate QFT
**Examiner:** "How does approximate QFT work and when is it useful?"

**Expected Response:**

**Observation:** Controlled-$R_k$ for large $k$ contributes tiny phases.

**Approximation:** Omit gates with $k > m$.

**Error analysis:**
$$\|QFT - QFT_{approx}\| \leq \frac{n(n-1)}{2 \cdot 2^m}$$

**Trade-off:** For error $\epsilon$, set $m = O(\log(n/\epsilon))$

**Result:** $O(n\log n)$ gates instead of $O(n^2)$

**Usefulness:**
- Shor's algorithm tolerates approximation
- NISQ devices: fewer gates = less error
- Fault-tolerant: fewer T gates needed

---

## Part III: Problem-Solving Under Pressure

### Scenario 1: Algorithm Design
**Examiner:** "Suppose I give you a function $f:\{0,1\}^n \to \{0,1\}$ promised to be either constant or to have exactly one 1 in its truth table. Design a quantum algorithm."

**Expected Analysis:**

This is NOT Deutsch-Jozsa (not balanced vs. constant).

**Modified approach:**
- If constant 0: Deutsch-Jozsa works
- If exactly one 1: This is OR function

For OR: Use Grover's algorithm!
- $O(\sqrt{2^n})$ queries to find the unique 1
- Then function is "exactly one 1"

**Alternative:** Amplitude estimation to count solutions
- Estimate number of 1s
- Distinguish 0 (constant) from 1 (unique)
- $O(\sqrt{2^n})$ queries

---

### Scenario 2: Error Analysis
**Examiner:** "In phase estimation, what happens if the input state is not an eigenstate?"

**Expected Analysis:**

Let $|\psi\rangle = \sum_j c_j |u_j\rangle$ where $U|u_j\rangle = e^{2\pi i\theta_j}|u_j\rangle$.

After phase estimation:
$$|\text{output}\rangle = \sum_j c_j |\tilde{\theta}_j\rangle|u_j\rangle$$

**Measurement:**
- Probability $|c_j|^2$ to measure phase $\tilde{\theta}_j$
- System collapses to $|u_j\rangle$

**Implication:** Phase estimation samples from the spectrum!

**Application:** Ground state energy estimation
- Start with easy-to-prepare state
- If overlap with ground state is $|c_0|^2$
- Probability $|c_0|^2$ to get ground state energy

---

### Scenario 3: Complexity Argument
**Examiner:** "Prove that any quantum algorithm needs $\Omega(n)$ queries to solve Simon's problem."

**Expected Argument:**

**Information-theoretic bound:**
- Must determine $s \in \{0,1\}^n$
- Each measurement gives at most $n$ bits of information
- Need $\Omega(1)$ iterations minimum

**Tighter bound:**
- Each query to $f$ can reveal at most 1 equation $y \cdot s = 0$
- Need $n-1$ independent equations
- Therefore $\Omega(n)$ queries required

Simon's algorithm is optimal (up to constant factors).

---

### Scenario 4: Implementation Challenge
**Examiner:** "How would you implement the QFT for 100 qubits on a near-term device?"

**Expected Response:**

**Challenges:**
- $O(n^2) = 10000$ gates (too many)
- Controlled-$R_k$ for $k > 20$ essentially identity
- Decoherence limits circuit depth

**Solutions:**
1. **Approximate QFT:** Use $m = 10-20$, reduces to $O(nm) = 1000-2000$ gates
2. **Semiclassical QFT:** Measure as you go, reduces qubits needed
3. **Native gates:** Compile to hardware-native gates
4. **Error mitigation:** Zero-noise extrapolation, probabilistic error cancellation

**Trade-offs:**
- Approximation error vs. gate error
- Must balance for optimal fidelity

---

## Part IV: Historical and Research Context

### Question 9: Development of Quantum Algorithms
**Examiner:** "Trace the historical development from Deutsch to Shor."

**Expected Timeline:**

**1985:** Deutsch proposes quantum Turing machine
**1992:** Deutsch-Jozsa algorithm (exponential speedup)
**1993:** Bernstein-Vazirani (quantum vs. classical query separation)
**1994:** Simon's algorithm (exponential speedup for specific problem)
**1994:** Shor's algorithm (factoring and discrete log)
**1996:** Grover's algorithm (quadratic speedup for search)

**Key progression:**
- Each algorithm built on previous insights
- Simon's period-finding inspired Shor
- QFT emerged as common tool
- Phase estimation generalized the approach

---

### Question 10: Current Research Directions
**Examiner:** "What are open problems related to this week's algorithms?"

**Expected Discussion:**

1. **Non-Abelian HSP:** Can we solve graph isomorphism?
2. **Optimal phase estimation:** Heisenberg-limited sensing
3. **Variational phase estimation:** Hybrid algorithms
4. **QFT applications:** Quantum machine learning, optimization
5. **Error-robust algorithms:** Phase estimation with noise

**Key insight:** These foundational algorithms continue to inspire research 30 years later.

---

## Part V: Mock Oral Exam (20-minute simulation)

### Examiner Script

**Opening (3 min):**
"Give me an overview of the quantum algorithms we use the Quantum Fourier Transform in."

**Technical probe (5 min):**
"Let's focus on phase estimation. Walk me through the circuit and explain each step."

**Derivation (5 min):**
"Derive the success probability when the phase is not exactly representable in $t$ bits."

**Application (4 min):**
"How would you use phase estimation to find the ground state energy of a molecule?"

**Conceptual (3 min):**
"What's the relationship between phase estimation and Grover's algorithm? Are they fundamentally different?"

---

## Evaluation Rubric

### Excellent (A)
- Clear explanations with correct technical details
- Can derive results on demand
- Makes connections between algorithms
- Discusses practical considerations

### Good (B)
- Mostly correct explanations
- Minor errors in derivations
- Some connections made
- Basic practical awareness

### Adequate (C)
- General understanding demonstrated
- Cannot derive key results
- Algorithms treated in isolation
- Limited practical insight

### Needs Work (D/F)
- Fundamental misunderstandings
- Cannot explain basic algorithms
- No derivation ability
- No practical context

---

## Self-Practice Checklist

Before the oral exam, verify you can:

- [ ] Explain Deutsch-Jozsa in 3 minutes
- [ ] Derive Simon's algorithm complexity
- [ ] Construct QFT circuit for arbitrary n
- [ ] Explain phase kickback intuitively
- [ ] Calculate phase estimation precision
- [ ] Connect these algorithms to HSP framework
- [ ] Discuss NISQ implementation challenges

---

*This oral practice guide covers Week 162 topics. For written problems, see Problem_Set.md; for self-assessment, see Self_Assessment.md.*
