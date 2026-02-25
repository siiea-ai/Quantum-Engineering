# Week 173: Fault-Tolerant Operations - Problem Set

## Instructions

This problem set contains 28 problems covering the threshold theorem, transversal gates, the Eastin-Knill theorem, and magic state distillation. Problems are organized by topic and difficulty level.

**Difficulty Levels:**
- **(B)** Basic - Direct application of definitions and formulas
- **(I)** Intermediate - Requires synthesis of multiple concepts
- **(A)** Advanced - Research-level or proof-based problems

**Time Estimate:** 8-10 hours total

---

## Section 1: Error Models and Fault Tolerance Definitions (Problems 1-6)

### Problem 1 (B)
Consider the depolarizing channel:
$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

(a) Show that this can be rewritten as:
$$\mathcal{E}_p(\rho) = \left(1-\frac{4p}{3}\right)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z + \rho)$$

(b) What is the probability that a Pauli error (X, Y, or Z) occurs?

(c) For $$p = 0.01$$, what is the probability of no error occurring?

---

### Problem 2 (B)
Define what it means for an operation to be *fault-tolerant* for a distance-3 code. Explain why this definition is crucial for the threshold theorem.

---

### Problem 3 (I)
Consider a fault-tolerant syndrome extraction circuit for the $$[[7,1,3]]$$ Steane code.

(a) Why can't we simply measure each stabilizer generator using a single ancilla qubit with CNOT gates to each data qubit?

(b) Describe the Shor-style syndrome extraction method and explain how it achieves fault tolerance.

(c) How many ancilla qubits are needed for Shor-style extraction of one stabilizer?

---

### Problem 4 (I)
A *malignant set* of faults is a set that causes logical failure despite error correction.

(a) For a distance-3 code, what is the minimum size of a malignant set?

(b) If a fault-tolerant gadget has $$n_{\text{loc}} = 100$$ locations, give an upper bound on the number of malignant pairs.

(c) Using this bound, estimate the level-1 failure probability if $$p = 10^{-4}$$.

---

### Problem 5 (A)
Prove that for a distance-$$d$$ code with fault-tolerant operations:

$$p^{(1)} \leq A_d \cdot p^{\lceil d/2 \rceil}$$

where $$A_d$$ is a constant depending on the code and gadget complexity.

---

### Problem 6 (I)
Compare the following error models in terms of their effect on threshold estimates:

(a) Independent depolarizing noise on each qubit
(b) Circuit-level noise with gate errors
(c) Correlated noise (e.g., crosstalk between adjacent qubits)

Which model gives the most conservative (lowest) threshold estimate? Why?

---

## Section 2: Threshold Theorem (Problems 7-12)

### Problem 7 (B)
For a concatenated code with threshold $$p_{\text{th}} = 10^{-2}$$:

(a) If $$p = 10^{-3}$$, what is the logical error rate at concatenation level $$L = 3$$?

(b) How many levels of concatenation are needed to achieve logical error rate $$\delta = 10^{-15}$$?

(c) If the base code is $$[[7,1,3]]$$, how many physical qubits are needed at this level?

---

### Problem 8 (I)
The threshold theorem states that resources scale as $$O(\text{poly}\log(1/\delta))$$.

(a) Derive the explicit scaling for concatenated distance-3 codes.

(b) Show that the number of physical qubits per logical qubit is $$O(\log^{\alpha}(1/\delta))$$ and find $$\alpha$$.

(c) Compare this to the scaling for surface codes, which is $$O(\log^2(1/\delta))$$.

---

### Problem 9 (A)
**Proof Problem:** Provide a detailed proof of the following lemma:

*Lemma:* If a fault-tolerant gadget for a distance-3 code has $$n_{\text{loc}}$$ locations and each location fails independently with probability $$p$$, then the logical failure probability satisfies:
$$p_{\text{fail}} \leq \binom{n_{\text{loc}}}{2} p^2$$

---

### Problem 10 (I)
Consider two codes:
- Code A: $$[[7,1,3]]$$ with $$n_{\text{loc}} = 50$$ locations per gadget
- Code B: $$[[23,1,7]]$$ with $$n_{\text{loc}} = 500$$ locations per gadget

(a) Calculate the threshold for each code (approximately).

(b) For physical error rate $$p = 10^{-4}$$, which code gives lower logical error rate at level 1?

(c) At what physical error rate do the two codes give equal level-1 logical error rates?

---

### Problem 11 (B)
Explain the relationship between the following quantities:
- Code distance $$d$$
- Number of locations per gadget $$n_{\text{loc}}$$
- Threshold $$p_{\text{th}}$$
- Concatenation level $$L$$ for target error $$\delta$$

---

### Problem 12 (A)
The threshold for the surface code is approximately 1%, while for concatenated Steane code it is approximately $$10^{-5}$$.

(a) Explain why the surface code threshold is so much higher.

(b) Despite the higher threshold, concatenated codes can sometimes be preferable. Under what circumstances?

(c) Derive the crossover point in terms of physical error rate.

---

## Section 3: Transversal Gates (Problems 13-17)

### Problem 13 (B)
For the $$[[7,1,3]]$$ Steane code with logical operators:
$$\overline{X} = X_1 X_2 X_3 X_4 X_5 X_6 X_7$$
$$\overline{Z} = Z_1 Z_2 Z_3 Z_4 Z_5 Z_6 Z_7$$

(a) Verify that $$H^{\otimes 7}$$ implements a logical Hadamard gate.

(b) Show that $$S^{\otimes 7}$$ does NOT implement a logical $$S$$ gate.

(c) What does $$S^{\otimes 7}$$ implement instead?

---

### Problem 14 (I)
For CSS codes constructed from classical codes $$C_1 \supseteq C_2$$:

(a) Prove that $$\text{CNOT}^{\otimes n}$$ between two code blocks implements a logical CNOT.

(b) Under what condition on $$C_1$$ and $$C_2$$ is $$H^{\otimes n}$$ a valid logical operation?

(c) Give an example of a CSS code where $$H^{\otimes n}$$ is NOT a valid logical gate.

---

### Problem 15 (I)
The $$[[15,1,3]]$$ Reed-Muller code has a transversal $$T$$ gate but NOT a transversal Hadamard.

(a) Explain why this is consistent with the Eastin-Knill theorem.

(b) How can this code be combined with the Steane code to achieve universality via code switching?

(c) What is the overhead of code switching compared to magic state injection?

---

### Problem 16 (A)
**Proof Problem:** Prove that for any stabilizer code, the transversal gates form a finite group.

*Hint:* Consider the action on the logical Pauli group.

---

### Problem 17 (I)
The Clifford group is generated by $$\{H, S, \text{CNOT}\}$$.

(a) Show that the Steane code has transversal implementations of all Clifford generators.

(b) Verify that compositions of transversal gates remain transversal.

(c) Conclude that all Clifford gates have transversal implementations on the Steane code.

---

## Section 4: Eastin-Knill Theorem (Problems 18-21)

### Problem 18 (B)
State the Eastin-Knill theorem precisely. What are the key assumptions?

---

### Problem 19 (I)
The proof of Eastin-Knill uses the fact that continuous transversal gates would violate error correction.

(a) Explain why a continuous 1-parameter family $$U(\theta)$$ of transversal gates cannot exist.

(b) What property of quantum error correction is violated?

(c) How does this argument fail for $$d = 1$$ codes?

---

### Problem 20 (A)
**Proof Problem:** Complete the following proof outline for the Eastin-Knill theorem:

1. Assume transversal gates $$\mathcal{T}$$ include a universal gate set
2. Universal gate sets generate dense subgroups of $$SU(2^k)$$
3. A dense subgroup must contain elements arbitrarily close to identity
4. Such elements would form a continuous family
5. [Your contribution: Show this contradicts error correction]

---

### Problem 21 (I)
List three methods for achieving universal fault-tolerant computation despite the Eastin-Knill theorem:

(a) Magic state injection
(b) Code switching
(c) Gauge fixing

For each, explain how it circumvents the theorem's restriction.

---

## Section 5: Magic States and Gate Injection (Problems 22-24)

### Problem 22 (B)
The T-magic state is defined as:
$$|T\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + e^{i\pi/4}|1\rangle\right)$$

(a) Verify that $$|T\rangle = T|+\rangle$$.

(b) Express $$|T\rangle$$ in the $$\{|+\rangle, |-\rangle\}$$ basis.

(c) Calculate $$\langle T|T\rangle$$ and $$\langle T|S|T\rangle$$.

---

### Problem 23 (I)
Describe the gate injection protocol for implementing $$T|\psi\rangle$$ using $$|T\rangle$$ and Clifford operations.

(a) Draw the circuit.

(b) Prove that it correctly implements the T gate (up to known Pauli corrections).

(c) What happens if the magic state has error $$\epsilon$$? How does this affect the output?

---

### Problem 24 (I)
The H-magic state is:
$$|H\rangle = \cos\left(\frac{\pi}{8}\right)|0\rangle + \sin\left(\frac{\pi}{8}\right)|1\rangle$$

(a) What gate can be implemented using $$|H\rangle$$ state injection?

(b) How is the H-state related to the T-state?

(c) Which state is more commonly used in fault-tolerant protocols? Why?

---

## Section 6: Magic State Distillation (Problems 25-28)

### Problem 25 (B)
The 15-to-1 distillation protocol has output error $$\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3$$.

(a) If $$\epsilon_{\text{in}} = 0.01$$, what is $$\epsilon_{\text{out}}$$?

(b) How many rounds of distillation are needed to reach $$\epsilon < 10^{-10}$$?

(c) How many total input magic states are consumed?

---

### Problem 26 (I)
Analyze the 15-to-1 protocol in detail:

(a) What code is used? What are its parameters?

(b) Why does the protocol achieve cubic error suppression?

(c) What is the acceptance probability (probability of not discarding)?

(d) Including rejection, what is the expected number of input states per output state?

---

### Problem 27 (A)
The distillation overhead exponent $$\gamma$$ is defined by:
$$N_{\text{magic}} = O\left(\log^{\gamma}\left(\frac{1}{\epsilon}\right)\right)$$

(a) Derive $$\gamma$$ for the 15-to-1 protocol with concatenated distillation.

(b) How does using a $$k$$-to-1 protocol with error suppression $$\epsilon^m$$ change $$\gamma$$?

(c) What is the theoretical minimum value of $$\gamma$$? Has it been achieved?

---

### Problem 28 (A)
**Research Problem:** Recent work (2025) has achieved constant-overhead magic state distillation using QLDPC codes.

(a) Explain how asymptotically good QLDPC codes enable $$\gamma = 0$$.

(b) What are the practical challenges in implementing these protocols?

(c) Compare the crossover point: at what target error rate do QLDPC-based protocols become advantageous over standard 15-to-1?

---

## Bonus Challenge Problems

### Problem 29 (A)
Design a fault-tolerant protocol for measuring a logical operator that is not a stabilizer. Specifically, for the Steane code, design a protocol to measure $$\overline{X}$$ fault-tolerantly.

---

### Problem 30 (A)
The threshold theorem assumes perfect classical computation for syndrome processing. Analyze what happens if the classical computation also has errors:

(a) Can the threshold theorem still hold?

(b) What additional assumptions are needed?

(c) How does this affect the threshold value?

---

## Submission Guidelines

1. Show all work and justify each step
2. State any additional assumptions clearly
3. For proof problems, structure your argument logically
4. Numerical answers should include appropriate significant figures
5. Diagrams and circuits should be clearly labeled

**Solutions available in:** [Problem_Solutions.md](Problem_Solutions.md)
