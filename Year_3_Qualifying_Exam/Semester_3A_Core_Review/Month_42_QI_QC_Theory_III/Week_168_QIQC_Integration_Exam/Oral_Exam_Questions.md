# QI/QC Theory Integration - Oral Exam Questions

## Overview

This document contains comprehensive oral exam questions covering all QI/QC theory topics. Use these for mock oral exams (90-120 minutes).

**Format:** Examiner asks main question, followed by 2-3 follow-ups based on your response.

---

## Topic 1: Density Matrices and Mixed States

### Main Question
"Explain the density matrix formalism and why it's essential for quantum information."

**Expected Coverage:**
- Definition: $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$
- Properties: Hermitian, positive, trace 1
- Pure vs. mixed states
- Bloch sphere representation

**Follow-ups:**
- "How do you determine if a state is pure or mixed from its density matrix?"
- "Calculate the purity $\text{Tr}(\rho^2)$ for the maximally mixed qubit."
- "What's the physical interpretation of off-diagonal elements?"

---

## Topic 2: Entanglement

### Main Question
"Define entanglement and explain how to detect and quantify it."

**Expected Coverage:**
- Separable vs. entangled states
- Bell states as maximally entangled examples
- PPT criterion for detection
- Entanglement entropy for pure states

**Follow-ups:**
- "Is the Werner state entangled for all $p > 0$? What's the threshold?"
- "Explain the CHSH inequality and how it detects entanglement."
- "What is bound entanglement?"

---

## Topic 3: Quantum Channels

### Main Question
"Describe the mathematical framework for quantum channels."

**Expected Coverage:**
- CPTP (completely positive, trace-preserving) maps
- Kraus representation: $\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$
- Choi-Jamiolkowski isomorphism
- Examples: depolarizing, amplitude damping

**Follow-ups:**
- "Prove that Kraus operators satisfy $\sum_k K_k^\dagger K_k = I$."
- "What does 'completely positive' mean? Why is it necessary?"
- "Give an example of an entanglement-breaking channel."

---

## Topic 4: Quantum Gates and Universality

### Main Question
"What does it mean for a gate set to be universal for quantum computation?"

**Expected Coverage:**
- Universal gate sets: {H, T, CNOT}
- Solovay-Kitaev theorem
- Clifford group and why it's not universal
- Magic states for universality

**Follow-ups:**
- "Why can't we compute arbitrary functions with just Clifford gates?"
- "How many T gates are needed to approximate a rotation to precision $\epsilon$?"
- "Explain the role of the T gate in fault-tolerant computing."

---

## Topic 5: Quantum Algorithms

### Main Question
"Explain the quantum Fourier transform and its role in quantum algorithms."

**Expected Coverage:**
- QFT definition and circuit
- Phase estimation algorithm
- Use in Shor's algorithm
- Comparison to classical FFT

**Follow-ups:**
- "What's the gate complexity of QFT?"
- "In Shor's algorithm, what eigenvalue is being estimated?"
- "Can QFT be used for speedup on arbitrary functions?"

### Alternative Main Question
"Describe Grover's algorithm and prove it is optimal."

**Expected Coverage:**
- Problem setup: unstructured search
- Grover iteration: oracle + diffusion
- Geometric interpretation
- Polynomial method for lower bound

**Follow-ups:**
- "What happens if you iterate too many times?"
- "How does Grover extend to multiple marked items?"
- "Why doesn't Grover give exponential speedup for NP problems?"

---

## Topic 6: Quantum Complexity Theory

### Main Question
"Define BQP and explain what we know about its relationship to classical complexity classes."

**Expected Coverage:**
- BQP definition via quantum circuits
- P ⊆ BPP ⊆ BQP ⊆ PP ⊆ PSPACE
- Oracle separations
- Evidence for separation from BPP

**Follow-ups:**
- "Prove that BQP ⊆ PSPACE."
- "Is BQP contained in NP? What evidence exists?"
- "What is the significance of Simon's problem for complexity?"

### Alternative Main Question
"What is QMA and what is its natural complete problem?"

**Expected Coverage:**
- QMA definition: quantum witness, quantum verifier
- Comparison to NP
- Local Hamiltonian problem
- QMA-completeness proof sketch

**Follow-ups:**
- "Why can quantum witnesses be useful?"
- "What does 2-local mean in 2-local Hamiltonian?"
- "Is there a quantum analog of the PCP theorem?"

---

## Topic 7: Quantum Teleportation

### Main Question
"Derive the quantum teleportation protocol from scratch."

**Expected Coverage:**
- Initial state and Bell state resource
- Bell measurement expansion
- Classical communication
- Bob's correction operations

**Follow-ups:**
- "Why is classical communication necessary?"
- "What happens with noisy entanglement?"
- "Explain gate teleportation."

---

## Topic 8: Quantum Key Distribution

### Main Question
"Explain the BB84 protocol and its security."

**Expected Coverage:**
- Protocol steps
- No-cloning as security basis
- Intercept-resend attack analysis
- Error rate and key rate

**Follow-ups:**
- "What's the maximum tolerable error rate?"
- "How does E91 differ from BB84?"
- "What is device-independent QKD?"

---

## Topic 9: Von Neumann Entropy

### Main Question
"Define von Neumann entropy and state its key properties."

**Expected Coverage:**
- Definition: $S(\rho) = -\text{Tr}(\rho \log \rho)$
- Non-negativity, concavity, maximum value
- Subadditivity and strong subadditivity
- Relation to entanglement

**Follow-ups:**
- "Prove that entropy is concave."
- "When is conditional entropy negative? What does it mean?"
- "State strong subadditivity and give an application."

---

## Topic 10: Holevo Bound

### Main Question
"State and explain the Holevo bound."

**Expected Coverage:**
- Holevo quantity formula
- Bound on accessible information
- When the bound is achieved
- Connection to channel capacity

**Follow-ups:**
- "Calculate $\chi$ for two non-orthogonal pure states."
- "How is this related to superdense coding?"
- "What is the HSW theorem?"

---

## Topic 11: Channel Capacity

### Main Question
"Compare classical, quantum, and entanglement-assisted channel capacity."

**Expected Coverage:**
- Classical capacity: Holevo quantity
- Quantum capacity: coherent information
- Entanglement-assisted: mutual information
- Regularization issues

**Follow-ups:**
- "Why is entanglement-assisted capacity additive but classical isn't?"
- "Can a channel have positive classical but zero quantum capacity?"
- "What is the private capacity?"

---

## Topic 12: Quantum Data Compression

### Main Question
"Explain Schumacher compression."

**Expected Coverage:**
- Compression rate equals von Neumann entropy
- Typical subspace concept
- Comparison to Shannon compression
- Fidelity analysis

**Follow-ups:**
- "What is the typical subspace?"
- "When is quantum compression better than classical?"
- "How does this relate to the quantum source coding theorem?"

---

## Integration Questions

### Integration Question 1
"Trace the role of entanglement from fundamental physics through to practical applications."

**Expected Coverage:**
- Entanglement as physical phenomenon (Bell states)
- Detection methods (CHSH, PPT)
- Quantification (entropy, concurrence)
- Applications: teleportation, QKD, superdense coding, MBQC

### Integration Question 2
"Connect von Neumann entropy to three different areas of quantum information."

**Expected Coverage:**
- Entanglement measure (pure bipartite states)
- Holevo bound (communication limits)
- Schumacher compression (data compression)
- Channel capacity (quantum capacity)

### Integration Question 3
"Explain how quantum complexity theory informs our understanding of quantum advantage."

**Expected Coverage:**
- BQP vs. BPP: what can quantum do better?
- Oracle separations (Simon, Forrelation)
- Specific algorithms (Shor, Grover)
- Sampling problems (BosonSampling)

---

## Quick-Fire Questions

For 1-2 minute responses:

1. "What is the trace of a density matrix and why?"
2. "State the no-cloning theorem."
3. "What's the difference between BPP and BQP?"
4. "Define a CPTP map."
5. "What is the Bloch sphere?"
6. "State the Holevo bound in one sentence."
7. "Why can't we use teleportation for FTL communication?"
8. "What is query complexity?"
9. "Define the T gate."
10. "What is strong subadditivity?"

---

## Mock Oral Exam Format

**Duration:** 90 minutes

**Structure:**
1. (20 min) Deep dive on one topic (examiner's choice)
2. (20 min) Deep dive on another topic (your choice)
3. (30 min) Broad questions across all topics
4. (20 min) Integration questions

**Scoring:**
- 5: Exceptional - could teach this material
- 4: Strong - minor gaps only
- 3: Adequate - knows basics but struggles with depth
- 2: Weak - significant gaps
- 1: Insufficient - cannot explain fundamentals

**Passing:** Average score ≥ 4 across all topics

---

## Self-Practice Instructions

1. Have someone ask you random questions from this list
2. Time yourself: 3-5 minutes per main question
3. Practice handling "I don't know" situations gracefully
4. Record yourself and review for clarity
5. Focus on:
   - Clear definitions
   - Logical flow
   - Connecting concepts
   - Admitting uncertainty when appropriate

---

**Created:** February 9, 2026
