# Final Mock Qualifying Examination - Oral Component

## Examination Structure

**Total Duration:** 90 minutes
**Format:** Three phases with simulated committee

---

## Phase 1: Research Proposal Presentation (20 minutes)

### Instructions

Present your research proposal as if to your qualifying committee. You should cover:

1. **Introduction** (3-4 minutes)
   - Motivation and importance of the problem
   - Specific research question

2. **Background** (4-5 minutes)
   - Key prior work
   - Current state of the field
   - Gap your research addresses

3. **Proposed Research** (6-8 minutes)
   - Specific aims
   - Methodology
   - Expected outcomes

4. **Timeline and Significance** (3-4 minutes)
   - Project timeline
   - Expected impact
   - Why you are prepared for this work

### Evaluation Criteria

| Criterion | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|
| Clarity of problem statement | | | | | |
| Command of background | | | | | |
| Feasibility of approach | | | | | |
| Significance of outcomes | | | | | |
| Time management | | | | | |
| Confidence and poise | | | | | |

---

## Phase 2: Broad Questioning (40 minutes)

### Instructions

The committee will ask questions across all domains of quantum science and engineering. Questions progress from fundamental to advanced. Respond as you would to real examiners.

### Question Bank - Quantum Mechanics

#### Fundamentals (Choose 2-3)

**QM-F1:** Explain the physical meaning of the wave function. What does $|\psi(x)|^2$ represent and what constraints must $\psi$ satisfy?

*Expected elements: Probability amplitude, normalization, continuity, Born interpretation*

---

**QM-F2:** State the Heisenberg uncertainty principle for position and momentum. Derive it from the commutation relation $[x,p] = i\hbar$.

*Expected elements: State formula, use Schwarz inequality or Robertson relation, show derivation*

---

**QM-F3:** Describe the differences between the Schrodinger and Heisenberg pictures. When might you prefer one over the other?

*Expected elements: State evolution vs operator evolution, equivalence for expectation values, examples of when each is useful*

---

**QM-F4:** What is a stationary state? Why doesn't the probability distribution of a stationary state change with time?

*Expected elements: Eigenstate of H, time-independent probability, phase factor only*

---

#### Intermediate (Choose 2-3)

**QM-I1:** Explain the quantum harmonic oscillator using the algebraic (ladder operator) method. What are the key steps in deriving the energy spectrum?

*Expected elements: Define a, a†, show [a,a†]=1, show H = ℏω(N+1/2), derive spectrum from N eigenvalues*

---

**QM-I2:** A hydrogen atom is placed in a uniform electric field (Stark effect). How do you calculate the energy shifts using perturbation theory?

*Expected elements: Identify perturbation, note degeneracy in n=2, use degenerate PT, selection rules*

---

**QM-I3:** Explain the physical content of the spin-statistics theorem. Why are electrons fermions and photons bosons?

*Expected elements: Connection between spin and statistics, Pauli exclusion, symmetric vs antisymmetric states*

---

**QM-I4:** Describe the WKB approximation. When is it valid and what are its limitations?

*Expected elements: Semiclassical regime, slowly varying potential, connection formulas at turning points*

---

#### Advanced (Choose 1-2)

**QM-A1:** Explain Berry's phase. Give a physical example where it arises and why it matters.

*Expected elements: Geometric phase, adiabatic evolution, Aharonov-Bohm as example*

---

**QM-A2:** In scattering theory, what is the relationship between the S-matrix and the T-matrix? What physical information does each encode?

*Expected elements: S = 1 + 2πiT, unitarity, cross sections from T-matrix*

---

### Question Bank - Quantum Information

#### Fundamentals (Choose 2-3)

**QI-F1:** What is a density matrix? When do we need to use density matrices instead of state vectors?

*Expected elements: Mixed vs pure states, partial trace, ensemble interpretation*

---

**QI-F2:** Define entanglement. How can you determine if a two-qubit state is entangled?

*Expected elements: Non-separability, Schmidt decomposition, PPT criterion for 2x2*

---

**QI-F3:** Explain quantum teleportation step by step. What resources are required and what is transmitted?

*Expected elements: Bell pair, Bell measurement, classical communication, correction*

---

**QI-F4:** What is the no-cloning theorem? Prove it and explain its implications.

*Expected elements: Linearity argument, implications for error correction and cryptography*

---

#### Intermediate (Choose 2-3)

**QI-I1:** Describe the depolarizing channel. Write its Kraus representation and explain its physical meaning.

*Expected elements: ρ → (1-p)ρ + p I/2, Kraus operators, noise model interpretation*

---

**QI-I2:** Explain the quantum Fourier transform. How does it differ from the classical FFT and why is it useful?

*Expected elements: Definition, circuit construction, O(n²) gates, role in algorithms*

---

**QI-I3:** Compare the Deutsch-Jozsa algorithm to its classical counterpart. What is the quantum speedup and why does it occur?

*Expected elements: 1 query vs O(N) classical, superposition, interference*

---

**QI-I4:** What is the Holevo bound? What does it say about the information capacity of quantum systems?

*Expected elements: χ ≤ S(ρ) - Σpᵢ S(ρᵢ), accessible information, implications*

---

#### Advanced (Choose 1-2)

**QI-A1:** Explain the CHSH inequality and quantum violation. What does this tell us about nature?

*Expected elements: Classical bound 2, quantum max 2√2, local hidden variables ruled out*

---

**QI-A2:** Describe the relationship between quantum complexity classes BQP and classical classes P and NP. What do we know and what remains open?

*Expected elements: P ⊆ BQP, BQP vs NP unknown, examples of problems in each*

---

### Question Bank - Quantum Error Correction

#### Fundamentals (Choose 2-3)

**QEC-F1:** Why is quantum error correction harder than classical error correction?

*Expected elements: No-cloning, continuous errors, measurement destroys state*

---

**QEC-F2:** Explain the 3-qubit bit-flip code. How does it encode, detect, and correct errors?

*Expected elements: Encoding, syndrome measurement, correction, limitations*

---

**QEC-F3:** What is a stabilizer code? Give the key definitions and an example.

*Expected elements: Stabilizer group, code space as +1 eigenspace, Steane code example*

---

**QEC-F4:** State the Knill-Laflamme error correction conditions. What do they mean physically?

*Expected elements: PE†EP = αP, errors distinguishable or identical*

---

#### Intermediate (Choose 2-3)

**QEC-I1:** Explain the surface code architecture. What are its advantages and disadvantages?

*Expected elements: 2D layout, local stabilizers, high threshold, overhead*

---

**QEC-I2:** What is fault tolerance and why is it necessary for practical quantum computing?

*Expected elements: Error propagation, threshold theorem, concatenation*

---

**QEC-I3:** Describe magic state distillation. Why is it needed and how does it work?

*Expected elements: Non-Clifford gates, magic states, distillation protocols, overhead*

---

**QEC-I4:** Compare CSS codes with general stabilizer codes. What special properties do CSS codes have?

*Expected elements: X and Z stabilizers separate, transversal H, construction from classical*

---

#### Advanced (Choose 1-2)

**QEC-A1:** What are quantum LDPC codes and why are they receiving attention?

*Expected elements: Sparse parity checks, constant overhead potential, decoding challenges*

---

**QEC-A2:** Explain the Eastin-Knill theorem and its implications for fault-tolerant computing.

*Expected elements: No transversal universal gate set, need for magic states or alternatives*

---

### Probe Questions (For Any Topic)

When a candidate gives a partial answer, use these probes:

- "Can you be more specific about...?"
- "What would happen if we changed...?"
- "How does this connect to...?"
- "Can you derive that result?"
- "What are the limitations of this approach?"
- "Can you give me a concrete example?"
- "Why is that true physically?"
- "What would you do if you didn't know the answer?"

---

## Phase 3: Deep Dive (30 minutes)

### Instructions

Select one of the following topics for intensive questioning. The candidate should demonstrate expert-level understanding.

### Topic A: Quantum Algorithms Deep Dive

**Opening:** "Let's talk in depth about quantum algorithms. Start by explaining Grover's algorithm."

**Follow-up Questions:**
1. Derive the optimal number of iterations.
2. What happens if there are multiple marked items?
3. Can Grover's algorithm be used for optimization? How?
4. What are the limitations of the oracle model?
5. How does amplitude amplification generalize Grover's algorithm?
6. Compare quantum speedup in Grover's vs Shor's algorithm.
7. What is the quantum query complexity of search?
8. Describe a hybrid classical-quantum approach using Grover.

---

### Topic B: Quantum Error Correction Deep Dive

**Opening:** "Let's explore quantum error correction in depth. Explain the stabilizer formalism."

**Follow-up Questions:**
1. How do you find logical operators for a stabilizer code?
2. Prove that the Steane code has distance 3.
3. How does syndrome decoding work? What are the challenges?
4. Explain the threshold theorem in detail.
5. Why can't we have transversal non-Clifford gates for all stabilizer codes?
6. Describe the surface code decoding problem as a statistical physics problem.
7. What is single-shot error correction?
8. Compare overhead of surface code vs concatenated codes.

---

### Topic C: Quantum Information Theory Deep Dive

**Opening:** "Let's discuss quantum information theory deeply. Start with von Neumann entropy."

**Follow-up Questions:**
1. Prove subadditivity of von Neumann entropy.
2. What is strong subadditivity and why is it important?
3. Explain quantum mutual information and its interpretation.
4. What is the Holevo-Schumacher-Westmoreland theorem?
5. Describe quantum data compression.
6. What is the quantum capacity of a channel?
7. Explain the distinction between private and quantum capacity.
8. What are resource theories in quantum information?

---

### Topic D: Research Proposal Deep Dive

**Opening:** "Let's dig deeper into your research proposal."

**Follow-up Questions:**
1. What is the most significant contribution your work could make?
2. What would happen if your main approach doesn't work?
3. Who are the key competitors in this area?
4. What preliminary results do you have, if any?
5. What are the three most likely failure modes?
6. How will you validate your results?
7. What would be a "home run" outcome vs a "solid single"?
8. Where do you see this research in 10 years?

---

## Scoring Rubrics

### Phase 1: Presentation (20 minutes)

| Category | Weight | 1 | 2 | 3 | 4 | 5 |
|----------|--------|---|---|---|---|---|
| Content accuracy | 25% | | | | | |
| Organization | 20% | | | | | |
| Depth of understanding | 25% | | | | | |
| Communication clarity | 15% | | | | | |
| Time management | 15% | | | | | |

**Phase 1 Score:** ___/5

---

### Phase 2: Broad Questions (40 minutes)

| Category | Weight | 1 | 2 | 3 | 4 | 5 |
|----------|--------|---|---|---|---|---|
| QM fundamentals | 20% | | | | | |
| QI fundamentals | 20% | | | | | |
| QEC fundamentals | 20% | | | | | |
| Intermediate topics | 20% | | | | | |
| Advanced topics | 10% | | | | | |
| Handling unknowns | 10% | | | | | |

**Phase 2 Score:** ___/5

---

### Phase 3: Deep Dive (30 minutes)

| Category | Weight | 1 | 2 | 3 | 4 | 5 |
|----------|--------|---|---|---|---|---|
| Core understanding | 30% | | | | | |
| Technical depth | 25% | | | | | |
| Problem solving | 20% | | | | | |
| Connections made | 15% | | | | | |
| Handling pressure | 10% | | | | | |

**Phase 3 Score:** ___/5

---

## Overall Assessment

| Phase | Score | Weight | Weighted |
|-------|-------|--------|----------|
| 1. Presentation | /5 | 25% | |
| 2. Broad Questions | /5 | 40% | |
| 3. Deep Dive | /5 | 35% | |
| **Total** | | 100% | **/5** |

### Passing Criteria

- Overall average: 3.5/5 or higher
- No phase below 3.0/5
- No critical knowledge gaps exposed

### Result

- [ ] **PASS** - Ready for research phase
- [ ] **CONDITIONAL PASS** - Minor remediation needed
- [ ] **NOT PASS** - Significant preparation needed

### Examiner Comments

**Strengths:**

_______________________________________________
_______________________________________________

**Areas for Improvement:**

_______________________________________________
_______________________________________________

**Recommendations:**

_______________________________________________
_______________________________________________

---

## Self-Administration Guide

If taking this oral exam self-administered:

1. **Record yourself** answering questions
2. **Set timer** for each phase
3. **Use random selection** for questions (dice or random number generator)
4. **Be honest** in self-assessment
5. **Review recording** critically
6. **Grade using rubrics** above

For Phase 2, randomly select:
- 2-3 fundamental questions from each domain
- 2-3 intermediate questions from each domain
- 1-2 advanced questions total

For Phase 3, select one topic randomly or choose your weakest area.
