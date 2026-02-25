# Week 165: Quantum Complexity Theory - Problem Set

## Instructions

This problem set contains 30 problems spanning BQP, QMA, and query complexity. Problems are organized by difficulty:
- **Level 1:** Direct application of definitions (Problems 1-10)
- **Level 2:** Intermediate analysis and proofs (Problems 11-20)
- **Level 3:** Qualifying exam level challenges (Problems 21-30)

Time estimate: 15-20 hours total
Recommended approach: 5 problems per day

---

## Section A: BQP Fundamentals (Problems 1-10)

### Problem 1: BQP Definition
State the formal definition of BQP using the quantum circuit model. What are the completeness and soundness parameters? What does "bounded error" mean in this context?

### Problem 2: P ⊆ BQP
Prove that P ⊆ BQP. Your proof should explicitly describe how to simulate a classical Turing machine using quantum gates.

### Problem 3: Error Reduction in BQP
Show that BQP with error bounds (1/3, 2/3) equals BQP with error bounds (2^{-n}, 1 - 2^{-n}). Explain why this "error reduction" is important.

### Problem 4: Gate Set Independence
Explain why BQP is independent of the choice of universal gate set. Reference the Solovay-Kitaev theorem in your answer.

### Problem 5: BQP ⊆ PSPACE
Prove that BQP ⊆ PSPACE. Your proof should describe a PSPACE algorithm that simulates a polynomial-size quantum circuit.

*Hint:* Use the path-sum formulation of quantum amplitudes.

### Problem 6: BQP ⊆ PP
Outline the proof that BQP ⊆ PP. Explain the key idea of how probability amplitudes relate to counting problems.

### Problem 7: BQP vs BPP
Explain the relationship between BQP and BPP. What is the current state of knowledge? Is it proven that BQP ⊃ BPP?

### Problem 8: Oracle Separations
What is an oracle separation? Explain why the existence of an oracle O with BQP^O ≠ BPP^O does not prove BQP ≠ BPP in the real world.

### Problem 9: Factoring in BQP
Explain why integer factoring is believed to be in BQP but not in BPP. Reference Shor's algorithm in your answer.

### Problem 10: Quantum Sampling
Define a sampling problem. Explain why quantum sampling problems (like BosonSampling or random circuit sampling) are considered evidence for quantum advantage, even though they don't decide a language.

---

## Section B: QMA and QMA-Completeness (Problems 11-17)

### Problem 11: QMA Definition
State the formal definition of QMA. Compare and contrast it with NP, highlighting the key differences.

### Problem 12: QMA Error Reduction
Prove that QMA with completeness c and soundness s (where c - s ≥ 1/poly(n)) equals QMA with completeness 1 - 2^{-n} and soundness 2^{-n}.

*Hint:* You cannot simply repeat and take majority because the witness might be entangled across runs.

### Problem 13: QMA Containment
Prove that QMA ⊆ PP.

*Hint:* Combine the BQP ⊆ PP proof with the structure of QMA verification.

### Problem 14: Local Hamiltonian Problem
Define the k-local Hamiltonian problem precisely. State the promise (gap condition) and explain why the gap is necessary.

### Problem 15: Local Hamiltonian in QMA
Prove that the k-local Hamiltonian problem is in QMA.

*Hint:* The ground state is the witness. Explain how to verify the energy using phase estimation.

### Problem 16: QMA-Hardness Intuition
Explain the key ideas behind proving that Local Hamiltonian is QMA-hard. What is a "history state" and how does it encode computation?

### Problem 17: 2-Local vs 5-Local
The original proof (Kitaev) showed 5-local Hamiltonian is QMA-complete. Kempe-Kitaev-Regev later proved 2-local suffices. Explain the significance of reducing locality and the general approach using perturbation theory.

---

## Section C: Query Complexity (Problems 18-25)

### Problem 18: Query Model Definition
Define the quantum query model precisely. What is a query oracle? How is query complexity Q(f) defined?

### Problem 19: Polynomial Method Statement
State the polynomial method theorem. Explain what it means for amplitudes to be multilinear polynomials.

### Problem 20: Polynomial Method Proof Sketch
Prove that if a quantum algorithm makes T queries, then every amplitude is a polynomial of degree at most T in the input bits.

*Hint:* Proceed by induction on the number of queries.

### Problem 21: Approximate Degree
Define the approximate degree of a Boolean function. State the relationship between approximate degree and quantum query complexity.

### Problem 22: Grover Lower Bound
Prove that Q(OR_n) = Ω(√n) using the polynomial method.

*Hint:* Use symmetry to reduce to univariate polynomials, then use Chebyshev polynomial bounds.

### Problem 23: AND Lower Bound
Prove that Q(AND_n) = Ω(√n).

*Hint:* AND(x) = NOT(OR(NOT(x₁), ..., NOT(xₙ))).

### Problem 24: Parity Lower Bound
Prove that Q(PARITY_n) = n (exact, not just Ω(n)).

*Hint:* Consider how flipping any single bit changes the parity.

### Problem 25: Element Distinctness
The element distinctness problem asks: given n elements, are any two equal? The quantum query complexity is Θ(n^{2/3}).

a) Explain why the classical query complexity is Θ(n).
b) Describe the quantum algorithm achieving O(n^{2/3}).
c) Why is this problem harder than OR but easier than the full search problem?

---

## Section D: Oracle Problems and Separations (Problems 26-28)

### Problem 26: Deutsch-Jozsa Analysis
Consider the Deutsch-Jozsa problem: given a function f: {0,1}ⁿ → {0,1} promised to be either constant or balanced, determine which.

a) What is the quantum query complexity?
b) What is the deterministic classical query complexity?
c) What is the randomized classical query complexity (with bounded error)?
d) Explain the quantum algorithm.

### Problem 27: Simon's Problem
Consider Simon's problem: given f: {0,1}ⁿ → {0,1}ⁿ with f(x) = f(y) iff x ⊕ y ∈ {0ⁿ, s} for some unknown s, find s.

a) What is the quantum query complexity?
b) What is the classical query complexity (deterministic and randomized)?
c) Why is Simon's problem significant for complexity theory?

### Problem 28: Forrelation
The Forrelation problem (Aaronson-Ambainis) provides the largest known separation between BQP and PH.

a) Define the Forrelation problem.
b) What is its quantum query complexity?
c) What lower bound is known for classical algorithms?
d) What does this say about BQP vs. PH?

---

## Section E: Advanced and Integration Problems (Problems 29-30)

### Problem 29: Complexity Zoo Navigation
For each of the following, state whether the containment is known, believed, or open:

a) P ⊆ NP
b) P = BQP?
c) BQP ⊆ NP?
d) NP ⊆ BQP?
e) BQP = QMA?
f) QMA ⊆ NEXP?
g) BQP ⊆ PH?

### Problem 30: Research Frontier
Consider the following open question: "Is there a problem in BQP that is NP-hard?"

a) What would it mean if such a problem exists?
b) What evidence exists for or against?
c) How does this relate to the quantum PCP conjecture?
d) If factoring were proven NP-hard, what would this imply about BQP vs. NP?

---

## Bonus Problems (Optional)

### Bonus 1: Adversary Method
State the adversary method for query lower bounds. Prove that Q(OR) = Ω(√n) using the adversary method instead of the polynomial method.

### Bonus 2: Quantum Counting
Analyze the quantum counting algorithm (combining Grover with phase estimation).

a) What problem does it solve?
b) What is its query complexity?
c) Prove its correctness.

### Bonus 3: QMA(2)
QMA(2) allows two unentangled quantum witnesses.

a) Define QMA(2) precisely.
b) Is QMA(2) = QMA known?
c) What problems become easier with two witnesses?

---

## Problem Set Summary

| Section | Problems | Topics | Points |
|---------|----------|--------|--------|
| A | 1-10 | BQP fundamentals | 30 |
| B | 11-17 | QMA and completeness | 25 |
| C | 18-25 | Query complexity | 30 |
| D | 26-28 | Oracle separations | 10 |
| E | 29-30 | Advanced/Integration | 5 |
| Bonus | 1-3 | Extra credit | +10 |

**Total: 100 points + 10 bonus**

---

## Submission Guidelines

For qualifying exam preparation:
1. Write complete solutions with full proofs
2. Time yourself (simulate exam conditions)
3. Mark problems you find challenging for extra review
4. Practice explaining solutions orally

---

**Created:** February 9, 2026
**Estimated Time:** 15-20 hours
