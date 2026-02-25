# Mock Oral Exam II: Advanced Question Bank

## Harder Questions for the Second Mock Examination

---

## Instructions

These questions are more challenging than Mock Oral I. They require:
- Deeper derivations
- More subtle physical insight
- Stronger connections between topics
- Research-level awareness

Use for Phase 2 and Phase 3 of Mock Oral II.

---

## Section A: Advanced Quantum Mechanics

### A1: Foundations and Formalism

**A1-1.** Derive the generalized uncertainty relation from the Cauchy-Schwarz inequality. Under what conditions is it saturated?

**A1-2.** Explain the difference between projective measurements and POVMs. Give an example where POVMs are necessary and prove they cannot be replaced by projective measurements.

**A1-3.** What is the Wigner function? Derive its properties and explain why it can be negative. What does negativity imply about classicality?

**A1-4.** Prove that the trace distance $$D(\rho,\sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma|$$ satisfies the data processing inequality. What does this mean physically?

**A1-5.** Derive the quantum Zeno effect from first principles. Under what conditions does the "anti-Zeno" effect occur instead?

---

### A2: Advanced Problems in Standard Systems

**A2-1.** For the harmonic oscillator, prove that coherent states minimize the uncertainty product. Then derive the time evolution of a coherent state and show it remains coherent.

**A2-2.** Derive the selection rules for electric dipole transitions using the commutator $$[H, \vec{r}]$$. Why are "forbidden" transitions sometimes observed?

**A2-3.** Calculate the second-order energy correction for the ground state of helium using perturbation theory. Compare to the variational result and explain the discrepancy.

**A2-4.** Derive the Landau levels for a charged particle in a magnetic field. Explain the connection to the integer quantum Hall effect.

**A2-5.** What is the quantum defect theory for alkali atoms? Derive the modified Rydberg formula and explain its physical origin.

---

### A3: Scattering and Many-Body

**A3-1.** Derive the optical theorem relating the total cross section to the forward scattering amplitude. What is its physical meaning?

**A3-2.** Explain the Feshbach resonance. How can it be used to tune interatomic interactions, and why is this important for quantum simulation?

**A3-3.** Derive the BCS gap equation. What determines the critical temperature for superconductivity?

**A3-4.** What is second quantization? Derive the Hamiltonian for interacting fermions and explain why this formalism is essential for many-body physics.

**A3-5.** Explain the Kondo effect. What makes it a paradigmatic example of strongly correlated physics?

---

## Section B: Advanced Quantum Information

### B1: Entanglement Theory

**B1-1.** Prove that entanglement cannot increase under LOCC (local operations and classical communication). What does this imply about entanglement as a resource?

**B1-2.** Define and derive the entanglement of formation. For a general two-qubit state, how is it calculated?

**B1-3.** What is bound entanglement? Give an example and explain why it cannot be distilled.

**B1-4.** Prove Tsirelson's bound $$2\sqrt{2}$$ for the CHSH inequality violation. What physical principle is responsible for this limit?

**B1-5.** Explain the monogamy of entanglement. Derive the Coffman-Kundu-Wootters inequality for three qubits.

---

### B2: Quantum Channels and Information

**B2-1.** Derive the Holevo bound and explain its operational meaning. When is it achieved?

**B2-2.** What is the quantum capacity of a channel? Explain why it's not additive and give an example of a channel with zero quantum capacity but nonzero private capacity.

**B2-3.** Prove that the completely depolarizing channel has zero quantum capacity but nonzero classical capacity.

**B2-4.** What is the entanglement-assisted classical capacity? Derive it from the quantum mutual information.

**B2-5.** Explain quantum error correction from the channel capacity perspective. How does encoding rate relate to channel properties?

---

### B3: Quantum Algorithms - Deep Understanding

**B3-1.** Prove the optimality of Grover's algorithm using the polynomial method or adversary argument.

**B3-2.** Explain why hidden subgroup problems are natural for quantum computers. Which groups have efficient quantum algorithms? Which don't?

**B3-3.** Derive the quantum speedup in the HHL algorithm for solving linear systems. What are the caveats that limit its practical advantage?

**B3-4.** What is the quantum singular value transformation framework? How does it unify quantum algorithms?

**B3-5.** Explain the relationship between quantum query complexity and quantum time complexity. When do they differ?

---

### B4: Complexity and Foundations

**B4-1.** What is the relationship between BQP and the polynomial hierarchy? What would it mean if BQP $$\not\subset$$ PH?

**B4-2.** Explain sampling complexity and its role in quantum supremacy arguments. What makes random circuit sampling hard classically?

**B4-3.** What is the Gottesman-Knill theorem? Prove that stabilizer circuits are classically simulable.

**B4-4.** Explain magic state injection and why it enables universal quantum computation. What is the connection to non-Clifford operations?

**B4-5.** What is quantum contextuality? How does it relate to quantum computational advantage?

---

## Section C: Advanced Quantum Error Correction

### C1: Theoretical Foundations

**C1-1.** Derive the Knill-Laflamme conditions from first principles. Give a physical interpretation of each condition.

**C1-2.** Prove the Eastin-Knill theorem: no code can have a universal set of transversal gates. What are the implications for fault-tolerant computation?

**C1-3.** What is the quantum Singleton bound? Derive it and explain what it tells us about code parameters.

**C1-4.** Explain the connection between quantum error correction and operator algebra. What is the correctable algebra?

**C1-5.** Derive the relationship between code distance and the number of correctable errors. When can this relationship be violated?

---

### C2: Topological Codes Deep Dive

**C2-1.** Derive the ground state degeneracy of the toric code on a torus. How does this relate to topological order?

**C2-2.** Explain the connection between the surface code and the $$\mathbb{Z}_2$$ lattice gauge theory. What physical insights does this provide?

**C2-3.** Derive the threshold of the surface code under independent depolarizing noise using the random bond Ising model mapping.

**C2-4.** What are twist defects in the surface code? How can they be used to implement non-Clifford operations?

**C2-5.** Explain the color code and its transversal gate set. What makes it potentially advantageous over the surface code?

---

### C3: Fault Tolerance Advanced

**C3-1.** Derive the threshold theorem for concatenated codes. How does the logical error rate scale with code level?

**C3-2.** Explain magic state distillation protocols. Derive the overhead scaling for the 15-to-1 protocol.

**C3-3.** What is gauge fixing? How does it enable conversion between different code types?

**C3-4.** Explain the concept of a logical block and how it's used in surface code implementations of Clifford+T.

**C3-5.** What are the current best estimates for fault-tolerant overhead? How might this scale with problem size?

---

### C4: Emerging Directions

**C4-1.** What are QLDPC codes? Explain why constant-rate, constant-distance codes with efficient decoders would be transformative.

**C4-2.** Describe GKP (Gottesman-Kitaev-Preskill) codes. What noise models are they particularly suited for?

**C4-3.** What is the surface code with biased noise? How does bias affect the threshold and code design?

**C4-4.** Explain floquet codes. What advantages might they offer over static stabilizer codes?

**C4-5.** What is the role of quantum error correction in quantum memory? What are the fundamental limits?

---

## Section D: Cross-Cutting and Research-Level

### D1: Deep Physical Questions

**D1-1.** Explain decoherence and einselection. Why do certain pointer states emerge? What determines the decoherence timescale?

**D1-2.** What is the resource theory of asymmetry? How does it relate to quantum reference frames and superselection rules?

**D1-3.** Explain the thermodynamics of quantum computation. What is the minimum energy cost to erase a qubit?

**D1-4.** What is quantum chaos? How is it characterized, and what are its implications for quantum simulation and computing?

**D1-5.** Explain the holographic principle and its potential relevance to quantum error correction (holographic codes).

---

### D2: Experimental Frontiers

**D2-1.** What are the dominant error mechanisms in superconducting qubits? How do modern architectures address each?

**D2-2.** Explain how trapped ion quantum computers achieve high gate fidelities. What limits further improvement?

**D2-3.** What is quantum error mitigation? Compare zero-noise extrapolation, probabilistic error cancellation, and virtual distillation.

**D2-4.** Explain the current state of quantum advantage demonstrations. What has been proven, and what remains controversial?

**D2-5.** What are the engineering challenges in scaling to millions of physical qubits? Address control, connectivity, and calibration.

---

### D3: Research Connections

**D3-1.** How might quantum computers impact drug discovery and materials science? Be specific about which problems and algorithms.

**D3-2.** What is the current understanding of quantum machine learning? Where might quantum advantage be possible?

**D3-3.** Explain the relationship between quantum error correction and topological phases of matter. What experimental signatures connect them?

**D3-4.** What would a practical fault-tolerant quantum computer require in terms of resources? Give specific estimates.

**D3-5.** Where do you see the biggest open questions in quantum computing research today?

---

## Section E: Presentation Follow-Ups (Advanced)

### For Any Topic

**E-ADV-1.** What are the main open problems in [your topic area]?

**E-ADV-2.** How would recent developments in [related area] affect [your topic]?

**E-ADV-3.** If you had unlimited resources to advance [your topic], what experiment would you do?

**E-ADV-4.** What's the most likely way your analysis could be wrong?

**E-ADV-5.** How does [your topic] connect to the bigger picture of building useful quantum computers?

---

### For Specific Common Topics

**Surface Codes:**
- Derive the statistical mechanics mapping for the threshold.
- Explain how you would implement a T gate fault-tolerantly.
- What's the asymptotic overhead for universal computation?

**Quantum Algorithms:**
- Prove your algorithm's optimality (or explain why it might not be optimal).
- How does noise affect the algorithm's performance?
- What's the end-to-end resource estimate for a practical instance?

**Entanglement:**
- Explain multipartite entanglement classification.
- How would you verify entanglement device-independently?
- What's the role of entanglement in your specific application?

---

## Question Selection for Mock II

### Phase 2 (30 min): Topic Deep Dive

Draw 6 questions from the section matching your topic. After each answer, draw from Follow_Up_Scenarios.md for the chain.

### Phase 3 (25 min): Broad Questions

Draw:
- 2 from Section A (Advanced QM)
- 2 from Section B (Advanced QI)
- 2 from Section C (Advanced QEC)
- 2 from Section D (Cross-cutting)

---

## Difficulty Calibration

| Difficulty | Description | Expected Response |
|------------|-------------|-------------------|
| **Standard** | Typical qualifying exam | Complete answer expected |
| **Challenging** | Above typical exam | Good partial answer acceptable |
| **Research** | Open-ended or frontier | Demonstrate relevant knowledge and reasoning |

Most questions in this bank are **Challenging** or **Research** level.

---

*"These questions are designed to push you past your comfort zone. Struggling with them is part of the learning process."*

---

**Week 188 | Mock Oral Exam II | Advanced Question Bank**
