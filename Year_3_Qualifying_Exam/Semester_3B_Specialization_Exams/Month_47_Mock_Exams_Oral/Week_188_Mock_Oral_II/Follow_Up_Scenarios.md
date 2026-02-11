# Follow-Up Scenarios

## Probing Question Chains for Mock Oral Exam II

---

## Introduction

In real oral exams, faculty don't ask isolated questions. They ask follow-ups that probe deeper and deeper until they find the edge of your knowledge. This document provides structured follow-up chains to simulate that experience.

---

## How to Use This Document

1. Answer the initial question
2. Read the first follow-up and answer
3. Continue down the chain until you reach your limit
4. Note where you got stuck
5. Use this information for targeted remediation

---

## Chain Structure

Each chain follows this pattern:

```
Level 1: Foundational (should know well)
    ↓
Level 2: Standard depth (expected knowledge)
    ↓
Level 3: Advanced (good to know)
    ↓
Level 4: Research-level (shows exceptional preparation)
    ↓
Level 5: Frontier (may be beyond preparation)
```

---

## Section 1: Quantum Mechanics Chains

### Chain QM-1: Uncertainty Principle

**Level 1:** State the Heisenberg uncertainty principle.

**Level 2:** Derive it from the commutator $$[x, p] = i\hbar$$ and the Robertson inequality.

**Level 3:** When is the uncertainty relation saturated? What states achieve this?

**Level 4:** What is the energy-time uncertainty relation? Why is it fundamentally different from position-momentum?

**Level 5:** How does the uncertainty principle constrain quantum error correction? What about squeezed states in continuous-variable systems?

---

### Chain QM-2: Harmonic Oscillator

**Level 1:** What are the energy eigenvalues of the quantum harmonic oscillator?

**Level 2:** Derive this spectrum using ladder operators. Show $$[a, a^\dagger] = 1$$.

**Level 3:** What is a coherent state? Prove it's an eigenstate of the annihilation operator.

**Level 4:** Show that the time evolution of a coherent state is another coherent state. What is the classical correspondence?

**Level 5:** How do coherent states relate to the GKP encoding for bosonic quantum computing? What's the error correction capability?

---

### Chain QM-3: Angular Momentum

**Level 1:** What are the eigenvalues of $$J^2$$ and $$J_z$$?

**Level 2:** Derive these using ladder operators $$J_\pm = J_x \pm iJ_y$$.

**Level 3:** Add two spin-1/2 particles. What are the possible total spin states? Write them explicitly.

**Level 4:** Derive the Clebsch-Gordan coefficients for $$j_1 = j_2 = 1/2$$ using the ladder operator method.

**Level 5:** How does angular momentum coupling relate to the representation theory of SU(2)? What about SU(N) for quantum computing with qudits?

---

### Chain QM-4: Perturbation Theory

**Level 1:** State the first-order energy correction in non-degenerate perturbation theory.

**Level 2:** Derive it. Then derive the first-order state correction.

**Level 3:** What happens in degenerate perturbation theory? What's the key additional step?

**Level 4:** Derive Fermi's Golden Rule from time-dependent perturbation theory. What are its limitations?

**Level 5:** When does perturbation theory fail entirely? Give an example and explain what non-perturbative methods might be used instead.

---

### Chain QM-5: Measurement

**Level 1:** What are the postulates of quantum mechanics regarding measurement?

**Level 2:** What is wave function collapse? Is it unitary?

**Level 3:** Explain the measurement problem. Why is collapse problematic?

**Level 4:** How does decoherence provide a partial resolution? What remains unexplained?

**Level 5:** What experimental tests could distinguish between collapse interpretations (Copenhagen, Many-Worlds, GRW, etc.)?

---

## Section 2: Quantum Information Chains

### Chain QI-1: Entanglement

**Level 1:** Define entanglement. Give an example of an entangled state.

**Level 2:** Prove that $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$ is entangled.

**Level 3:** Calculate the entanglement entropy of $$|\Phi^+\rangle$$. What does it mean?

**Level 4:** What is the entanglement of formation? How does it differ from entanglement of distillation?

**Level 5:** What is bound entanglement? Why does it challenge our understanding of entanglement as a resource?

---

### Chain QI-2: Quantum Teleportation

**Level 1:** Describe the quantum teleportation protocol.

**Level 2:** Derive why it works mathematically. Show the state decomposition.

**Level 3:** What is gate teleportation? How does it enable fault-tolerant computation?

**Level 4:** What resources are consumed? Can we do better than one ebit per teleportation?

**Level 5:** How does teleportation work with continuous variables? What are the key differences from discrete systems?

---

### Chain QI-3: Quantum Channels

**Level 1:** What is a quantum channel? What conditions must it satisfy?

**Level 2:** Write the Kraus representation for the depolarizing channel.

**Level 3:** Derive the action of amplitude damping. What physical process does it model?

**Level 4:** What is the quantum capacity of the depolarizing channel? Why is capacity calculation generally hard?

**Level 5:** Explain superadditivity of quantum capacity. What does it tell us about quantum information theory?

---

### Chain QI-4: Grover's Algorithm

**Level 1:** What problem does Grover's algorithm solve? What's the speedup?

**Level 2:** Describe the Grover iteration. Why does it work?

**Level 3:** Prove the optimal number of iterations is $$O(\sqrt{N})$$.

**Level 4:** Prove this is optimal - no quantum algorithm can do better for unstructured search.

**Level 5:** How does Grover's algorithm perform with noise? What's the fault-tolerant resource cost for a practical-sized problem?

---

### Chain QI-5: Complexity

**Level 1:** What is BQP? Give an example of a problem in BQP.

**Level 2:** What is the relationship between BQP and classical complexity classes (P, NP, PSPACE)?

**Level 3:** What does quantum supremacy mean? What evidence do we have for it?

**Level 4:** What is the polynomial hierarchy? Why does BQP $$\not\subset$$ PH (if true) matter?

**Level 5:** What are the strongest complexity-theoretic arguments for or against quantum computational advantage?

---

## Section 3: Quantum Error Correction Chains

### Chain QEC-1: Basic Principles

**Level 1:** Why can't classical error correction be directly applied to quantum systems?

**Level 2:** Explain the 3-qubit bit-flip code. What errors does it correct?

**Level 3:** Explain the 9-qubit Shor code. Why 9 qubits?

**Level 4:** State and derive the Knill-Laflamme conditions for error correction.

**Level 5:** What is the quantum Singleton bound? What does it tell us about code efficiency?

---

### Chain QEC-2: Stabilizer Formalism

**Level 1:** What is the stabilizer formalism? Define a stabilizer group.

**Level 2:** Write the stabilizers for the 5-qubit perfect code.

**Level 3:** How do you find logical operators from stabilizers?

**Level 4:** Prove the Gottesman-Knill theorem (at least outline the key steps).

**Level 5:** What's the connection between stabilizer codes and classical coding theory? Can all good classical codes be made quantum?

---

### Chain QEC-3: Surface Codes

**Level 1:** Describe the surface code layout. What are the stabilizers?

**Level 2:** How does error correction work? Explain syndrome measurement.

**Level 3:** What determines the code distance? What is the threshold?

**Level 4:** Derive (or explain) the threshold using the random bond Ising model mapping.

**Level 5:** What are the leading proposals for implementing logical gates on the surface code? Compare their overheads.

---

### Chain QEC-4: Fault Tolerance

**Level 1:** What does "fault-tolerant" mean? Why is it necessary?

**Level 2:** State the threshold theorem. What does it guarantee?

**Level 3:** Explain magic state distillation. Why is it needed?

**Level 4:** Derive the overhead scaling for magic state distillation (e.g., 15-to-1 protocol).

**Level 5:** What are the prospects for reducing fault-tolerant overhead? Discuss recent advances.

---

### Chain QEC-5: Advanced Codes

**Level 1:** What is a CSS code? Give an example.

**Level 2:** How do you construct a CSS code from classical codes?

**Level 3:** What are the advantages of topological codes over concatenated codes?

**Level 4:** What are QLDPC codes? Why are they potentially important?

**Level 5:** Describe the recent breakthroughs in QLDPC codes. What implications do they have?

---

## Section 4: General/Physical Intuition Chains

### Chain GEN-1: Classical Limit

**Level 1:** How does quantum mechanics reduce to classical mechanics?

**Level 2:** What is Ehrenfest's theorem? Prove it.

**Level 3:** What is decoherence? How does it contribute to classicality?

**Level 4:** What is einselection? Why do pointer states emerge?

**Level 5:** Does decoherence solve the measurement problem? What aspects remain open?

---

### Chain GEN-2: Quantum Computing Advantage

**Level 1:** Why might quantum computers be more powerful than classical ones?

**Level 2:** What resources give quantum computers their power? Is it superposition? Entanglement? Interference?

**Level 3:** The Gottesman-Knill theorem says stabilizer circuits are classically simulable. What additional ingredient enables quantum advantage?

**Level 4:** Explain the role of contextuality or magic states in quantum computational advantage.

**Level 5:** What's the current theoretical understanding of what problems quantum computers can solve faster?

---

### Chain GEN-3: Hardware Comparison

**Level 1:** What are the main qubit platforms being developed?

**Level 2:** Compare superconducting qubits and trapped ions in terms of gate fidelity, coherence, and connectivity.

**Level 3:** What error types dominate in each platform? How does this affect error correction?

**Level 4:** What are the scaling challenges for each platform?

**Level 5:** Which platform do you think is most likely to achieve fault tolerance first, and why?

---

### Chain GEN-4: Information and Physics

**Level 1:** What is Landauer's principle?

**Level 2:** How much energy does it cost to erase a bit? A qubit?

**Level 3:** What is reversible computing? How does it relate to quantum computing?

**Level 4:** What does the Margolus-Levitin theorem say about computation speed limits?

**Level 5:** Is there a fundamental physical limit to quantum error correction performance?

---

### Chain GEN-5: Research Directions

**Level 1:** What are the biggest challenges in building a fault-tolerant quantum computer?

**Level 2:** What problems are quantum computers expected to solve that classical computers cannot?

**Level 3:** What is the current state of quantum advantage demonstrations?

**Level 4:** How might quantum computers impact cryptography in practice?

**Level 5:** What do you think will be the first commercially useful quantum computation, and why?

---

## Section 5: Topic-Specific Follow-Up Chains

*Use these if your presentation topic matches*

### For Surface Code Presentations

**Chain SC-1:**
1. What is the threshold?
2. How is it calculated?
3. What noise model was assumed?
4. How do correlations in noise affect it?
5. What's the current best experimental demonstration?

**Chain SC-2:**
1. How do you do a CNOT gate on surface codes?
2. What about a T gate?
3. Why can't all Clifford+T gates be transversal?
4. What's the overhead for implementing T gates?
5. How might this change with better magic state distillation?

---

### For Quantum Algorithm Presentations

**Chain ALG-1:**
1. What's the speedup of your algorithm?
2. Is it provably optimal?
3. What's the query complexity vs time complexity?
4. How does noise affect the algorithm?
5. What's the full resource estimate for a practical instance?

**Chain ALG-2:**
1. What subroutines does your algorithm use?
2. How are they implemented fault-tolerantly?
3. What's the dominant cost?
4. How does your algorithm compare to classical heuristics in practice?
5. What would it take to demonstrate useful quantum advantage?

---

### For Entanglement Presentations

**Chain ENT-1:**
1. How do you quantify entanglement?
2. Why are there multiple measures?
3. How do you measure entanglement experimentally?
4. What about multipartite entanglement?
5. How does entanglement relate to quantum computational advantage?

**Chain ENT-2:**
1. Can entanglement be created locally?
2. What is LOCC and why is it important?
3. What is entanglement distillation?
4. What is bound entanglement?
5. What open questions remain in entanglement theory?

---

## How to Score Yourself

After each chain:

| Reached Level | Assessment |
|---------------|------------|
| 5 | Exceptional - research-level understanding |
| 4 | Strong - ready for qualifying exam |
| 3 | Adequate - baseline competency |
| 2 | Needs work - significant gaps |
| 1 | Major gap - fundamental review needed |

---

## Tracking Your Progress

| Chain | Level Reached in Mock I | Level Reached in Mock II | Improvement |
|-------|------------------------|--------------------------|-------------|
| QM-1 | | | |
| QM-2 | | | |
| QI-1 | | | |
| QI-2 | | | |
| QEC-1 | | | |
| QEC-2 | | | |
| GEN-1 | | | |
| GEN-2 | | | |

---

*"The purpose of these chains is to find your limits. Going deeper than before shows growth."*

---

**Week 188 | Mock Oral Exam II | Follow-Up Scenarios**
