# Month 45: Oral Exam Practice Questions

## Instructions

These questions are designed to simulate a 45-minute oral qualifying exam. Practice answering each question aloud, timing yourself. Aim for clear, well-organized responses.

**Format:**
- Opening question: 8-10 minutes
- Follow-up questions: 2-3 minutes each
- Defense questions: 3-5 minutes

---

## Opening Questions (Choose One)

### Opening 1: Superconducting Qubits

**Question:** "Please explain the physics of the transmon qubit, starting from the Josephson junction and including how gates and readout are performed."

**Expected Coverage (8-10 min):**
1. Josephson junction physics (2 min)
   - Josephson relations
   - Cosine potential energy

2. Circuit quantization (2 min)
   - Lagrangian approach
   - Hamiltonian in $$E_J$$, $$E_C$$

3. Transmon regime (2 min)
   - $$E_J/E_C \gg 1$$ design
   - Charge noise suppression
   - Anharmonicity trade-off

4. Gates (2 min)
   - Single-qubit: microwave pulses
   - Two-qubit: cross-resonance or tunable coupler

5. Readout (2 min)
   - Dispersive coupling to resonator
   - Qubit-state-dependent frequency shift

---

### Opening 2: Trapped Ions

**Question:** "Explain trapped ion quantum computing, including how ions are trapped, how qubits are encoded, and how entangling gates work."

**Expected Coverage (8-10 min):**
1. Paul trap physics (2 min)
   - Oscillating fields
   - Pseudopotential
   - Secular motion

2. Qubit encoding (2 min)
   - Hyperfine vs optical qubits
   - Clock states
   - Coherence times

3. Laser-ion interaction (2 min)
   - Lamb-Dicke regime
   - Carrier and sideband transitions

4. Mølmer-Sørensen gate (3 min)
   - Bichromatic field
   - Spin-dependent force
   - Geometric phase
   - Robustness to thermal motion

5. Architectures (1 min)
   - Linear chains vs QCCD

---

### Opening 3: Neutral Atoms

**Question:** "Describe neutral atom quantum computing using Rydberg atoms, including the blockade mechanism and how it enables multi-qubit gates."

**Expected Coverage (8-10 min):**
1. Optical tweezers (2 min)
   - Dipole trapping
   - Array generation

2. Rydberg physics (2 min)
   - Scaling laws ($$n^{11}$$ for $$C_6$$)
   - van der Waals interaction

3. Blockade mechanism (2 min)
   - Blockade condition
   - Blockade radius derivation

4. Two-qubit gates (2 min)
   - CZ gate protocol
   - Pulse sequence

5. Multi-qubit gates and advantages (2 min)
   - Native CCZ
   - Scalability

---

### Opening 4: NISQ Algorithms

**Question:** "Explain the Variational Quantum Eigensolver algorithm, including its motivation, implementation, and the role of the classical optimizer."

**Expected Coverage (8-10 min):**
1. Motivation (2 min)
   - Classical intractability
   - Variational principle

2. Algorithm structure (2 min)
   - Hybrid loop
   - Parameter updates

3. Ansatz design (2 min)
   - Hardware-efficient vs UCCSD
   - Trade-offs

4. Measurement (2 min)
   - Hamiltonian decomposition
   - Grouping strategies

5. Optimization challenges (2 min)
   - Barren plateaus
   - Noise effects

---

## Follow-Up Questions by Topic

### Superconducting Follow-ups

**F1.** "Derive the dispersive shift $$\chi$$ from the Jaynes-Cummings Hamiltonian."

**F2.** "What limits T1 in modern transmon qubits? What about T2?"

**F3.** "Compare the cross-resonance gate and the tunable coupler approach for two-qubit gates."

**F4.** "What is the Purcell effect and how is it mitigated?"

**F5.** "Explain how a SQUID enables flux-tunable qubits."

---

### Trapped Ion Follow-ups

**F6.** "Derive the normal mode frequencies for two ions in a harmonic trap."

**F7.** "Why is the Mølmer-Sørensen gate insensitive to the initial thermal state?"

**F8.** "What is the QCCD architecture and how does it scale?"

**F9.** "Compare hyperfine and optical qubits. Which would you choose for a 50-qubit system?"

**F10.** "What are the main sources of heating in trapped ion systems?"

---

### Neutral Atom Follow-ups

**F11.** "Calculate the blockade radius for given $$C_6$$ and $$\Omega$$."

**F12.** "How is atom rearrangement performed to achieve defect-free arrays?"

**F13.** "Explain the pulse sequence for a Rydberg CZ gate."

**F14.** "What limits the gate fidelity in neutral atom systems?"

**F15.** "Compare neutral atom and trapped ion platforms for a 100-qubit simulation."

---

### Bosonic Code Follow-ups

**F16.** "Explain the GKP code states and how error correction works."

**F17.** "Why do cat qubits have biased noise? How does this help error correction?"

**F18.** "What is the Kerr-cat Hamiltonian and how does it stabilize cat states?"

**F19.** "Compare GKP and cat codes for near-term implementation."

**F20.** "What hardware platforms can implement bosonic codes?"

---

### NISQ Algorithm Follow-ups

**F21.** "Derive the parameter-shift rule for computing VQE gradients."

**F22.** "What causes barren plateaus and how can they be avoided?"

**F23.** "For MaxCut QAOA at p=1, what approximation ratio is guaranteed?"

**F24.** "Compare gradient-free and gradient-based optimizers for VQE."

**F25.** "What is ADAPT-VQE and when would you use it?"

---

### Error Mitigation Follow-ups

**F26.** "Explain zero-noise extrapolation and its limitations."

**F27.** "What is probabilistic error cancellation and what is its overhead?"

**F28.** "How does symmetry verification reduce errors?"

**F29.** "Compare error mitigation to error correction."

**F30.** "Design an error mitigation strategy for a 20-qubit VQE circuit."

---

## Application Problems

### App1: Chemistry Simulation

**Question:** "You want to simulate the ground state of a small molecule (e.g., BeH2) on a quantum computer. Walk me through your approach, including hardware selection, ansatz choice, and expected accuracy."

**Key Points:**
- Active space selection
- Qubit mapping (Jordan-Wigner)
- Ansatz choice (UCCSD or ADAPT-VQE)
- Hardware comparison (connectivity, fidelity)
- Error mitigation strategy
- Comparison to classical methods

---

### App2: Combinatorial Optimization

**Question:** "A company wants to use quantum computing for portfolio optimization with 50 assets. Evaluate the feasibility and recommend an approach."

**Key Points:**
- Problem formulation (QUBO)
- QAOA implementation
- Hardware requirements
- Expected performance vs classical
- Current limitations
- Realistic timeline

---

### App3: Platform Comparison

**Question:** "You have access to a 100-qubit superconducting machine, a 32-qubit trapped ion system, and a 200-qubit neutral atom array. For each of the following tasks, which would you choose and why?"
- (a) Random circuit sampling
- (b) Molecular ground state calculation
- (c) MaxCut optimization on a 50-node graph
- (d) Quantum simulation of a 2D Heisenberg model

---

### App4: Error Budget

**Question:** "Given a circuit with 100 two-qubit gates on a device with 1% two-qubit error rate, analyze the feasibility and propose solutions."

**Key Points:**
- Calculate expected fidelity
- Identify dominant error sources
- Propose error mitigation
- Discuss error correction requirements
- Compare to alternative approaches

---

## Defense Questions

### D1: Limitations

**Question:** "What is the biggest limitation of the quantum computing platform you just described? How might it be overcome?"

---

### D2: Classical Competition

**Question:** "For the problem you described, how do you know a classical computer couldn't do better?"

---

### D3: Timeline

**Question:** "When do you think quantum computers will achieve practical advantage for this application? What milestones need to be reached?"

---

### D4: Alternative Approaches

**Question:** "If the approach you described doesn't work, what would be your backup plan?"

---

### D5: Research Direction

**Question:** "What do you see as the most important open research question in quantum hardware/algorithms?"

---

## Scoring Rubric

### Technical Accuracy (30%)
- Correct physics and mathematics
- Appropriate use of equations
- Accurate numerical estimates

### Clarity (25%)
- Well-organized response
- Clear explanations
- Appropriate level of detail

### Depth (20%)
- Beyond surface-level understanding
- Connections between concepts
- Awareness of subtleties

### Breadth (15%)
- Integration across topics
- Comparison between platforms
- Practical considerations

### Communication (10%)
- Professional presentation
- Responds to follow-ups
- Acknowledges uncertainty appropriately

---

## Practice Tips

1. **Time yourself:** Practice staying within time limits
2. **Use a whiteboard:** Draw diagrams and write equations
3. **Record yourself:** Review for clarity and organization
4. **Practice with a partner:** Get feedback on explanations
5. **Prepare for the unexpected:** Practice answering unfamiliar questions
6. **Stay calm:** It's okay to think before answering
7. **Ask for clarification:** If a question is unclear, ask!
