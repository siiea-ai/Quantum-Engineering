# Mock Oral Exam I: Question Bank

## 60+ Questions Organized by Topic for the 90-Minute Examination

---

## Instructions

- Print this document
- Cut questions into strips or draw numbers randomly
- For each phase, draw the specified number of questions
- Answer questions in the order drawn
- Do not look ahead at questions

---

## Section A: Quantum Mechanics Fundamentals

### A1: Postulates and Mathematical Framework

**A1-1.** State the postulates of quantum mechanics and explain the physical meaning of each.

**A1-2.** What is the physical interpretation of the commutator $$[A, B]$$? Derive the generalized uncertainty relation.

**A1-3.** Explain the difference between Schrodinger, Heisenberg, and interaction pictures. When is each most useful?

**A1-4.** What are the requirements for an operator to represent a physical observable? Prove that Hermitian operators have real eigenvalues.

**A1-5.** Derive the time evolution of expectation values: $$\frac{d}{dt}\langle A \rangle = ?$$

---

### A2: One-Dimensional Systems

**A2-1.** Solve the harmonic oscillator using ladder operators. Derive the spectrum and show why energy is quantized.

**A2-2.** What is a coherent state? Derive its properties and explain why it's called "most classical."

**A2-3.** Derive the transmission coefficient for a particle tunneling through a rectangular barrier. What determines the tunneling probability?

**A2-4.** Explain the physical meaning of the wave function spreading for a free particle. How fast does a Gaussian wave packet spread?

**A2-5.** For the particle in a box, derive the energy spectrum. Why is zero-point energy necessary?

---

### A3: Angular Momentum and Spin

**A3-1.** Derive the commutation relations for angular momentum operators. What is the physical meaning of $$[L_x, L_y] = i\hbar L_z$$?

**A3-2.** Explain the Stern-Gerlach experiment. What does it tell us about spin?

**A3-3.** Derive the eigenvalues of $$J^2$$ and $$J_z$$ using the ladder operator method.

**A3-4.** Explain spin-orbit coupling physically. How does it affect the hydrogen spectrum?

**A3-5.** Add two spin-1/2 particles. What are the possible total spin states? Write them explicitly.

---

### A4: Perturbation Theory

**A4-1.** Derive the first and second order energy corrections in time-independent perturbation theory.

**A4-2.** What is the variational principle? Prove that trial wave functions give upper bounds to ground state energy.

**A4-3.** Derive Fermi's golden rule. What are its assumptions and limitations?

**A4-4.** Explain the adiabatic theorem. What is the Berry phase and when does it arise?

**A4-5.** Calculate the Stark effect for the ground state of hydrogen using perturbation theory.

---

### A5: Scattering and Identical Particles

**A5-1.** What is the scattering cross section? Relate it to the scattering amplitude.

**A5-2.** Explain the partial wave expansion. When is it useful?

**A5-3.** Derive the spin-statistics theorem connection: why are electrons fermions and photons bosons?

**A5-4.** Write the wave function for two identical fermions. What is the Pauli exclusion principle?

**A5-5.** Explain exchange interaction and how it affects helium energy levels.

---

## Section B: Quantum Information and Computing

### B1: Density Matrices and Mixed States

**B1-1.** What is the difference between a pure state and a mixed state? How do you distinguish them experimentally?

**B1-2.** Derive the Bloch sphere representation for qubits. What is the Bloch vector for a mixed state?

**B1-3.** Explain the partial trace operation. Derive the reduced density matrix for one qubit of an entangled pair.

**B1-4.** What is the purity $$\text{Tr}(\rho^2)$$? What are its bounds and what do they mean?

**B1-5.** How does the density matrix evolve under unitary operations? Under measurement?

---

### B2: Entanglement

**B2-1.** Define entanglement mathematically. Give examples of entangled and separable two-qubit states.

**B2-2.** Write all four Bell states and prove they form an orthonormal basis.

**B2-3.** What is the Schmidt decomposition? How does Schmidt number relate to entanglement?

**B2-4.** Prove that the CHSH inequality can be violated by quantum mechanics. What does this mean?

**B2-5.** Explain entanglement entropy. For the state $$\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$, calculate $$S(\rho_A)$$.

---

### B3: Quantum Operations and Channels

**B3-1.** What is a CPTP map? Why are these the allowed operations on quantum states?

**B3-2.** Explain the Kraus representation. Derive it for the depolarizing channel.

**B3-3.** What is the Choi-Jamiolkowski isomorphism? Why is it useful?

**B3-4.** Describe the amplitude damping channel. What physical process does it model?

**B3-5.** Prove the no-cloning theorem using linearity of quantum mechanics.

---

### B4: Quantum Algorithms

**B4-1.** Explain Deutsch's algorithm. Why does it demonstrate quantum advantage?

**B4-2.** Walk through Grover's algorithm step by step. Why is $$O(\sqrt{N})$$ optimal?

**B4-3.** What is the quantum Fourier transform? Derive the circuit for $$n$$ qubits.

**B4-4.** Explain the key ideas behind Shor's algorithm. How does period finding factor integers?

**B4-5.** What is the phase estimation algorithm? What is it used for?

---

### B5: Quantum Complexity and Information Theory

**B5-1.** What is the BQP complexity class? How does it relate to P, NP, and PSPACE?

**B5-2.** Define the von Neumann entropy. Prove it's maximized by the maximally mixed state.

**B5-3.** What is the Holevo bound? What does it tell us about quantum communication?

**B5-4.** Explain quantum teleportation. Why does it require classical communication?

**B5-5.** What is superdense coding? How does it achieve 2 classical bits per qubit?

---

## Section C: Quantum Error Correction

### C1: Fundamentals

**C1-1.** Why can't classical error correction be directly applied to quantum systems?

**C1-2.** State and explain the Knill-Laflamme conditions. What do they tell us about correctable errors?

**C1-3.** Explain the 3-qubit bit-flip code. How does syndrome measurement work?

**C1-4.** What is the 9-qubit Shor code? Why does it need 9 qubits?

**C1-5.** What determines the distance of a quantum error correcting code? Why does distance matter?

---

### C2: Stabilizer Formalism

**C2-1.** What is the Pauli group? Why is it important for error correction?

**C2-2.** Explain stabilizer codes. How do stabilizers define a code space?

**C2-3.** For the 7-qubit Steane code, write the stabilizer generators.

**C2-4.** What is the Gottesman-Knill theorem? What are its implications for quantum advantage?

**C2-5.** How do you find the logical operators of a stabilizer code?

---

### C3: Topological Codes

**C3-1.** Explain the toric code. What are the stabilizers and logical operators?

**C3-2.** How does the surface code differ from the toric code? Why is this difference important for experiments?

**C3-3.** What is topological protection? Why are topological codes robust against local errors?

**C3-4.** Explain the connection between the surface code and anyons.

**C3-5.** How does error correction work in the surface code? What is minimum weight perfect matching?

---

### C4: Fault Tolerance

**C4-1.** What is the threshold theorem? State it precisely and explain its significance.

**C4-2.** What makes a gate "transversal"? Why are transversal gates naturally fault-tolerant?

**C4-3.** Explain magic state distillation. Why is it necessary for universal fault-tolerant computation?

**C4-4.** What is the Eastin-Knill theorem? What does it imply for quantum error correction?

**C4-5.** Describe the surface code logical CNOT gate. Why is it challenging to implement?

---

### C5: Advanced Topics

**C5-1.** What are QLDPC codes? Why might they be important for reducing overhead?

**C5-2.** Explain the concatenated code approach to fault tolerance. How does error suppression scale?

**C5-3.** What is a decoder? Why does decoder choice matter for threshold?

**C5-4.** Describe color codes. How do they compare to surface codes?

**C5-5.** What are GKP states? How do they enable error correction for bosonic systems?

---

## Section D: General and Cross-Cutting

### D1: Physical Intuition

**D1-1.** How does the classical limit emerge from quantum mechanics? Be specific.

**D1-2.** What is decoherence? How does it relate to the measurement problem?

**D1-3.** Why is quantum computing potentially more powerful than classical computing? What's the source of the power?

**D1-4.** What is the role of entanglement in quantum computing? Is it necessary? Sufficient?

**D1-5.** Explain the connection between information and physics (Landauer's principle, etc.).

---

### D2: Experimental Considerations

**D2-1.** What are the main qubit platforms being developed? Compare their strengths and weaknesses.

**D2-2.** What limits qubit coherence times? How do different platforms address this?

**D2-3.** How are two-qubit gates implemented in superconducting and trapped ion systems?

**D2-4.** What is quantum error mitigation? How does it differ from error correction?

**D2-5.** What experiments have demonstrated quantum advantage? What did they actually show?

---

### D3: Research Connections

**D3-1.** What are the biggest challenges in building a fault-tolerant quantum computer?

**D3-2.** How might quantum computing impact cryptography? What should we do about it?

**D3-3.** What problems are quantum computers expected to solve efficiently? Give examples.

**D3-4.** What is quantum simulation? Why might it be the first useful application?

**D3-5.** Where do you see the field of quantum computing in 10 years?

---

## Section E: Presentation Follow-Up Questions

*Use these for the follow-up portion after your prepared presentation*

### Surface Code Presentations

**E-SC-1.** You mentioned the threshold is around 1%. What determines this value?

**E-SC-2.** How does the decoder affect the threshold?

**E-SC-3.** What happens to logical error rates above threshold?

**E-SC-4.** Can you implement non-Clifford gates on the surface code? How?

**E-SC-5.** What's the connection between surface codes and topological quantum field theory?

---

### Quantum Algorithm Presentations

**E-QA-1.** What's the relationship between your algorithm's speedup and classical hardness?

**E-QA-2.** How does noise affect the algorithm you presented?

**E-QA-3.** Can you describe the resource requirements more precisely?

**E-QA-4.** What classical subroutines are required?

**E-QA-5.** How might this algorithm be implemented on near-term devices?

---

### Quantum Information Presentations

**E-QI-1.** What experimental tests have verified the phenomena you described?

**E-QI-2.** What are the fundamental limits on [aspect of your topic]?

**E-QI-3.** How does this connect to other areas of quantum information?

**E-QI-4.** What open questions remain in this area?

**E-QI-5.** How might this be useful for quantum computing applications?

---

### General Follow-Up Questions

**E-GEN-1.** What assumption in your derivation is most crucial?

**E-GEN-2.** What happens in the limit where [parameter] becomes very large/small?

**E-GEN-3.** Can you give a physical intuition for [result]?

**E-GEN-4.** How would an experimentalist verify this?

**E-GEN-5.** What's the connection to [related topic]?

---

## Question Selection Guide

### Phase 1 Follow-Up (5-10 min)
Draw 3-5 questions from Section E (matching your topic) or E-GEN

### Phase 2 Topic Deep Dive (30 min)
Draw 8-10 questions from the section matching your specialization

### Phase 3 Broad Questioning (30 min)
Draw:
- 3 from Section A (QM)
- 3 from Section B (QI)
- 3 from Section C (QEC)
- 2 from Section D (General)

---

## Difficulty Coding

Questions can be mentally categorized:

| Level | Description | Expected Response |
|-------|-------------|-------------------|
| Basic | Fundamental concepts | Complete, confident answer |
| Standard | Typical qualifying exam level | Good answer, possible minor gaps |
| Challenging | At edge of preparation | Partial answer with reasoning |
| Research | Open-ended or cutting-edge | Show relevant knowledge, reason productively |

Most questions in this bank are Standard level. A few are Basic (for warm-up) and a few are Challenging (for probing).

---

*"Quality of preparation shows in the range of questions you can answer confidently."*

---

**Week 187 | Mock Oral Exam I | Question Bank**
