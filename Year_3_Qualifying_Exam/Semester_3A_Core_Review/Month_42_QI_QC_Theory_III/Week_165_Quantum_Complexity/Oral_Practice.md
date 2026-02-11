# Week 165: Quantum Complexity Theory - Oral Exam Practice

## Overview

This document contains oral examination practice questions for quantum complexity theory. Each question includes suggested response frameworks, key points to cover, and potential follow-up questions examiners might ask.

**Format:** PhD qualifying exam oral (15-20 minutes per topic)
**Goal:** Demonstrate deep understanding, not just recall

---

## Question 1: Define and Explain BQP

### Main Question
"Define BQP precisely and explain its significance in quantum computing."

### Response Framework

**Opening (30 seconds):**
"BQP, or Bounded-Error Quantum Polynomial Time, is the class of decision problems solvable by a quantum computer in polynomial time with bounded error probability."

**Formal Definition (1-2 minutes):**
"Formally, a language L is in BQP if there exists a uniform family of polynomial-size quantum circuits such that:
- For YES instances, the circuit accepts with probability at least 2/3
- For NO instances, the circuit accepts with probability at most 1/3

The 'bounded error' refers to this constant gap, which can be amplified exponentially by repetition."

**Key Points to Cover:**
1. Uniform circuit families (describable by classical poly-time algorithm)
2. Error amplification via majority voting
3. Gate set independence (Solovay-Kitaev)
4. Equivalence to quantum Turing machine definition

**Significance (1 minute):**
"BQP captures what quantum computers can efficiently solve. Problems like integer factoring are in BQP but not known to be in BPP, suggesting quantum advantage."

### Potential Follow-ups

**Q: "What's the relationship between BQP and classical complexity classes?"**

A: "We know P ⊆ BPP ⊆ BQP ⊆ PP ⊆ PSPACE. The first containment is trivial. BQP ⊆ PSPACE follows from the ability to simulate quantum circuits using polynomial space via path summation. Whether any containment is strict is open."

**Q: "Is there a BQP-complete problem?"**

A: "No natural BQP-complete problem is known. This would require proving BQP ≠ BPP, which is open. Approximating the Jones polynomial at certain roots of unity is BQP-complete under certain reductions."

**Q: "Why can we amplify errors in BQP?"**

A: "We repeat the circuit O(n) times and take majority vote. By Chernoff bounds, the error drops exponentially. This works because each run is independent."

---

## Question 2: QMA and the Local Hamiltonian Problem

### Main Question
"What is QMA and why is the Local Hamiltonian problem important?"

### Response Framework

**QMA Definition (1-2 minutes):**
"QMA is the quantum analog of NP. A language is in QMA if there's a polynomial-time quantum verifier such that:
- YES instances have a quantum witness the verifier accepts with probability ≥ 2/3
- NO instances are rejected with probability ≥ 2/3 for all witnesses

The witness is a quantum state on polynomially many qubits."

**Local Hamiltonian Problem (1-2 minutes):**
"The k-local Hamiltonian problem asks: given a Hamiltonian H = Σᵢ Hᵢ where each term acts on at most k qubits, and thresholds a < b, determine whether the ground state energy is ≤ a or ≥ b.

This is QMA-complete for k ≥ 2, proven by Kempe, Kitaev, and Regev."

**Why Important (1 minute):**
"This connects computational complexity to physics. It shows that determining ground state energies of quantum many-body systems is fundamentally hard - there's no efficient classical or quantum algorithm unless QMA = P."

### Potential Follow-ups

**Q: "Sketch why Local Hamiltonian is in QMA."**

A: "The ground state is the witness. The verifier uses phase estimation to estimate the energy. If energy is low, accept; if high, reject. The gap ensures distinguishability."

**Q: "What's the key idea for QMA-hardness?"**

A: "We encode verification circuits into Hamiltonians using 'history states.' The ground state encodes the computation history. Low energy corresponds to accepting computations."

**Q: "How does QMA compare to NP?"**

A: "QMA uses quantum witnesses and verifiers. Quantum witnesses can encode exponentially many amplitudes, but measurement limits extractable information. NP ⊆ QMA since classical witnesses are special cases."

---

## Question 3: Proving Grover's Algorithm is Optimal

### Main Question
"Prove that Grover's search algorithm is optimal."

### Response Framework

**Setup (30 seconds):**
"We want to show any quantum algorithm for unstructured search needs Ω(√N) queries. Grover achieves O(√N), so this proves optimality."

**Polynomial Method (2 minutes):**
"The key tool is the polynomial method. After T queries, the acceptance probability is a polynomial of degree at most 2T in the input bits.

For the OR function on n bits:
- If all bits are 0, we should reject (accept prob ≤ 1/3)
- If any bit is 1, we should accept (accept prob ≥ 2/3)"

**Degree Lower Bound (2 minutes):**
"By symmetry, we can assume the polynomial is symmetric: p(x) = q(|x|) where |x| is Hamming weight.

The polynomial q must satisfy:
- q(0) ≤ 1/3
- q(k) ≥ 2/3 for k ≥ 1

This means q must 'jump' from low to high at weight 1 and stay high. Using Chebyshev polynomial bounds, deg(q) = Ω(√n).

Therefore T ≥ deg(q)/2 = Ω(√n)."

### Potential Follow-ups

**Q: "What's the intuition behind Chebyshev polynomials here?"**

A: "Chebyshev polynomials achieve the minimal degree for bounded oscillation on an interval. Any polynomial satisfying our constraints must oscillate similarly, requiring comparable degree."

**Q: "Does this rule out quantum algorithms with better scaling for any problem?"**

A: "No, this is specific to unstructured search. Problems with structure (like Simon's or Shor's) can have exponential speedups. The lower bound applies to black-box/oracle settings."

**Q: "What about the adversary method?"**

A: "The adversary method is an alternative technique giving the same Ω(√n) bound. It's based on analyzing how distinguishable inputs change under queries. For some problems, it gives tighter bounds than the polynomial method."

---

## Question 4: Oracle Separations

### Main Question
"Explain what oracle separations tell us about quantum vs. classical computation."

### Response Framework

**Definition (1 minute):**
"An oracle separation between classes A and B means there exists an oracle O such that A^O ≠ B^O. The oracle is a black box that algorithms can query."

**Examples (1-2 minutes):**
"Simon's problem gives an oracle where BQP^O ≠ BPP^O: quantum needs O(n) queries while classical needs exponential.

The unstructured search oracle shows BQP^O ⊄ NP^O in some relativized world (though actually NP^O ⊄ BQP^O is the right statement here - search needs √N quantum queries but has an NP witness of size 1)."

**What They Tell Us (1-2 minutes):**
"Oracle separations show that techniques that work for all oracles cannot resolve the underlying question. Since most proof techniques relativize, they suggest the problems are hard to resolve.

However, they're not definitive. There could be non-relativizing proofs. Example: IP = PSPACE was proven despite oracle separations."

**Recent Results (1 minute):**
"Raz and Tal (2019) showed BQP^O ⊄ PH^O for some oracle, providing the strongest evidence yet that quantum computing transcends the polynomial hierarchy."

### Potential Follow-ups

**Q: "What's the Baker-Gill-Solovay theorem?"**

A: "It shows there exist oracles A and B such that P^A = NP^A and P^B ≠ NP^B. This means oracle techniques alone can't resolve P vs. NP."

**Q: "Are there non-relativizing quantum complexity results?"**

A: "The study of non-relativizing techniques in quantum complexity is less developed. Most known separations are oracle-based. This is an active research area."

---

## Question 5: BQP vs. NP

### Main Question
"What do we know about the relationship between BQP and NP?"

### Response Framework

**Current Knowledge (1-2 minutes):**
"Neither BQP ⊆ NP nor NP ⊆ BQP is known.

Evidence against BQP ⊆ NP: Factoring is in BQP but not known in NP (requires the factor as witness, which isn't obvious from BQP algorithm).

Evidence against NP ⊆ BQP: Grover's lower bound shows unstructured search needs √N queries, but NP problems have polynomial witnesses. No exponential quantum speedup for NP-complete problems is known."

**Oracle Results (1 minute):**
"There exist oracles separating BQP and NP in both directions:
- Oracle where BQP^O ⊄ NP^O (quantum can solve something NP cannot verify)
- Oracle where NP^O ⊄ BQP^O (NP has witnesses quantum can't find)"

**Implications (1 minute):**
"The classes are likely incomparable. Quantum computers probably offer advantages for structured problems (factoring, simulation) but not for unstructured NP-complete problems like 3-SAT."

### Potential Follow-ups

**Q: "Could quantum computers solve NP-complete problems efficiently?"**

A: "Unlikely. The quadratic Grover speedup is believed optimal for unstructured search. NP-complete problems would require exponential speedup. No such speedup is known or expected."

**Q: "What about QAOA and quantum optimization?"**

A: "QAOA and VQE are heuristics that may offer practical speedups on specific instances but don't change worst-case complexity. They're not expected to solve NP-complete problems in polynomial time."

---

## Question 6: Query Complexity vs. Circuit Complexity

### Main Question
"Explain the difference between query complexity and circuit complexity in quantum computing."

### Response Framework

**Query Complexity (1-2 minutes):**
"Query complexity studies how many oracle calls are needed to compute a function. The input is a black box; we count queries, not other operations.

Example: Grover's search has query complexity O(√N).

Advantages: Can prove strong lower bounds using polynomial/adversary methods."

**Circuit Complexity (1-2 minutes):**
"Circuit complexity counts total operations, not just oracle calls. BQP is defined via circuit complexity.

For decision problems without oracles, this is the relevant measure.

Proving circuit lower bounds is much harder - relates to P vs. NP and similar open problems."

**Relationship (1 minute):**
"Query complexity gives upper bounds on circuit complexity (need at least as many operations as queries), but not always matching.

Oracle separations use query complexity because it's tractable. They suggest but don't prove circuit separations."

### Potential Follow-ups

**Q: "Why is circuit complexity harder to analyze?"**

A: "We can't use black-box techniques - the algorithm can 'see' the input structure. Lower bound proofs must consider all possible algorithms, not just those using oracle access."

**Q: "What's the significance of the polynomial method?"**

A: "It converts quantum query algorithms to polynomials, allowing degree analysis. Since polynomial degree is well-understood mathematically, we can prove tight bounds. No analogous technique exists for circuit complexity."

---

## Question 7: Recent Advances in Quantum Complexity

### Main Question
"Describe a recent result in quantum complexity theory and its significance."

### Response Framework (Choose One)

**Option A: Raz-Tal Oracle Separation (2019)**

"Raz and Tal proved there exists an oracle O where BQP^O ⊄ PH^O. This means quantum computers can solve problems outside the entire polynomial hierarchy relative to some oracle.

Significance: Strongest evidence yet that quantum computing offers advantages beyond classical probabilistic computation. Previous separations were against BPP; this extends to PH.

Technical approach: Uses the Forrelation problem, showing it requires many classical queries but only O(1) quantum queries, and proving Forrelation is hard for PH."

**Option B: MIP* = RE (2020)**

"Slofstra, and later the MIP* = RE team, showed that the class of problems verifiable by multiple quantum provers with entanglement equals RE (recursively enumerable) - undecidable problems!

Significance: Entanglement gives provers extraordinary power. This resolved Tsirelson's problem and Connes' embedding conjecture (negative).

This shows quantum entanglement has profound implications for verification."

**Option C: Quantum Supremacy Experiments**

"Google's 2019 experiment demonstrated quantum sampling that would take classical supercomputers thousands of years. While not a complexity theory result per se, it provides experimental evidence for quantum advantage.

Subsequent work has both improved classical simulation and quantum experiments, leading to ongoing 'quantum supremacy' verification debates."

---

## Question 8: Explaining to Non-Experts

### Main Question
"Explain quantum computational advantage to someone with no physics or CS background."

### Response Framework

**Analogy (2 minutes):**
"Imagine solving a maze. A classical computer tries paths one by one. A quantum computer is like exploring many paths simultaneously, then 'amplifying' the right one.

But it's not magic - quantum mechanics has rules. For some mazes (structured problems like factoring), quantum helps enormously. For others (random mazes), quantum only gives modest speedups."

**Real-World Impact (1 minute):**
"Factoring large numbers protects internet security. Quantum computers could break this. That's why we're developing 'post-quantum' cryptography.

Quantum computers could also simulate molecules for drug discovery - a natural application since molecules follow quantum mechanics."

**Limitations (1 minute):**
"Quantum computers won't solve all hard problems. Many puzzles, like scheduling or optimization, probably won't see exponential quantum speedups. Quantum advantage is problem-dependent."

---

## General Oral Exam Tips

### Structure Your Answers
1. **Start with the definition/setup**
2. **State the main result clearly**
3. **Give key ideas of proof/reasoning**
4. **Discuss significance/implications**
5. **Be ready for follow-ups**

### Common Pitfalls
- Don't claim oracle separations prove absolute separations
- Be precise about which results are proven vs. believed
- Know the difference between query and circuit complexity
- Understand why QMA uses quantum witnesses (not just quantum verifiers)

### Confidence Calibration
- Say "I believe" for conjectures
- Say "It's known that" for theorems
- Say "I'm not certain, but..." if unsure
- It's okay to say "I don't know" - then try to reason through it

---

## Self-Assessment Checklist

After practicing, verify you can:

- [ ] State BQP definition without notes
- [ ] Prove P ⊆ BQP ⊆ PSPACE
- [ ] Define QMA and compare to NP
- [ ] Explain Local Hamiltonian problem and its significance
- [ ] Sketch Grover lower bound proof
- [ ] Discuss BQP vs. NP relationship
- [ ] Explain what oracle separations do and don't tell us
- [ ] Describe at least one recent quantum complexity result
- [ ] Explain quantum advantage to a non-expert

---

**Created:** February 9, 2026
**Practice Time:** 2-3 hours recommended
