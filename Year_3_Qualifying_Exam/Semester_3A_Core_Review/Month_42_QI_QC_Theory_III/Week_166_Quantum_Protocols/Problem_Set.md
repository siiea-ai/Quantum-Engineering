# Week 166: Quantum Protocols - Problem Set

## Instructions

This problem set contains 28 problems on quantum communication and cryptographic protocols. Problems are organized by topic and difficulty:
- **Level 1:** Direct application (Problems 1-7)
- **Level 2:** Intermediate analysis (Problems 8-18)
- **Level 3:** Qualifying exam level (Problems 19-28)

Time estimate: 15-20 hours total

---

## Section A: Quantum Teleportation (Problems 1-9)

### Problem 1: Basic Teleportation Derivation
Derive the quantum teleportation protocol completely. Start with the initial state $|\psi\rangle_A \otimes |\Phi^+\rangle_{A'B}$ and show:
a) The expansion in the Bell basis
b) Bob's state after each possible measurement outcome
c) The required correction operations

### Problem 2: Teleportation with Different Bell States
Suppose Alice and Bob share $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ instead of $|\Phi^+\rangle$.
a) Rederive the teleportation protocol
b) What corrections must Bob apply now?
c) Create a table mapping measurement outcomes to corrections

### Problem 3: Why Classical Communication?
a) Show that without classical communication, Bob's reduced density matrix is $\rho_B = I/2$ regardless of $|\psi\rangle$
b) Explain why this means teleportation cannot be used for faster-than-light communication
c) Calculate the mutual information between Alice's input state and Bob's state before receiving classical bits

### Problem 4: Teleportation Fidelity
a) Define teleportation fidelity $F = \langle\psi|\rho_{\text{out}}|\psi\rangle$
b) If Alice and Bob share the Werner state $\rho_W = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$, calculate the teleportation fidelity as a function of $p$
c) What is the minimum $p$ required to beat the classical teleportation limit $F = 2/3$?

### Problem 5: Non-Maximally Entangled States
Alice and Bob share $|\phi\rangle = \sqrt{\lambda}|00\rangle + \sqrt{1-\lambda}|11\rangle$ with $\lambda \neq 1/2$.
a) Can perfect teleportation be achieved? Explain.
b) Derive the teleportation fidelity as a function of $\lambda$
c) Describe a probabilistic teleportation scheme that achieves perfect fidelity when successful

### Problem 6: Teleportation Circuit
a) Draw the quantum circuit for teleportation using standard gates (CNOT, H, X, Z)
b) Express the Bell measurement using CNOT and Hadamard followed by computational basis measurement
c) Show that the circuit implements the teleportation protocol

### Problem 7: Entanglement Swapping
Charlie holds qubits $C_1$ and $C_2$ from two separate Bell pairs:
- $|\Phi^+\rangle_{AC_1}$ (shared with Alice)
- $|\Phi^+\rangle_{C_2B}$ (shared with Bob)

a) Write the initial four-qubit state
b) Charlie performs a Bell measurement on $C_1C_2$. What is the resulting state of qubits $A$ and $B$?
c) Show that Alice and Bob now share entanglement despite never interacting

### Problem 8: Gate Teleportation
In gate teleportation, we want Bob to receive $U|\psi\rangle$ instead of $|\psi\rangle$.
a) Show how to achieve this by modifying the shared entangled state
b) For the case $U = T$ (T-gate), derive the modified corrections Bob must apply
c) Explain why gate teleportation is important for fault-tolerant quantum computing

### Problem 9: Continuous Variable Teleportation
For infinite-dimensional systems (modes of light), teleportation uses:
- Two-mode squeezed vacuum state
- Homodyne detection (measures quadratures)

a) State the continuous variable teleportation protocol
b) What is the fidelity for a coherent state input with $r$ squeezing?
c) What limit is achieved as $r \to \infty$?

---

## Section B: Superdense Coding (Problems 10-13)

### Problem 10: Superdense Coding Derivation
a) Derive the superdense coding protocol completely
b) Show that Bob's Bell measurement perfectly distinguishes the four encoded states
c) Explain why 2 classical bits can be sent using only 1 qubit

### Problem 11: Higher Dimensional Superdense Coding
For a $d$-dimensional system:
a) What is the maximally entangled state in $d \times d$?
b) How many classical bits can be sent using one qudit and shared entanglement?
c) Prove this achieves the Holevo bound for entanglement-assisted communication

### Problem 12: Duality with Teleportation
a) Explain the resource duality between teleportation and superdense coding
b) Starting from teleportation, derive superdense coding by "running the protocol backward"
c) What does this duality tell us about the relationship between quantum and classical information?

### Problem 13: Noisy Superdense Coding
If Alice and Bob share a Werner state $\rho_W = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$:
a) Calculate Bob's probability of correctly identifying Alice's message
b) For what values of $p$ does superdense coding outperform sending one classical bit directly?
c) How does this relate to the entanglement of formation of $\rho_W$?

---

## Section C: BB84 Quantum Key Distribution (Problems 14-19)

### Problem 14: BB84 Protocol Analysis
a) Describe the BB84 protocol in detail
b) Why are two non-orthogonal bases necessary?
c) What fraction of transmitted bits are kept after basis reconciliation?

### Problem 15: Intercept-Resend Attack
Eve intercepts each qubit, measures in a random basis, and resends based on her outcome.
a) Calculate the probability that Eve introduces an error (when Alice and Bob use the same basis)
b) What is the expected bit error rate (QBER)?
c) How many bits must Alice and Bob compare to detect this attack with 99% confidence?

### Problem 16: Information Gain vs. Disturbance
a) Prove that if Eve gains information about a qubit, she must disturb it
b) For the intercept-resend attack, calculate Eve's information gain $I(A:E)$
c) Show the trade-off: more information for Eve → higher QBER

### Problem 17: Key Rate Calculation
The asymptotic secret key rate for BB84 is:
$$R = 1 - H(e) - H(e)$$
where $e$ is the QBER and $H$ is binary entropy.

a) Explain each term in this formula
b) At what QBER does the key rate become zero?
c) Why is there a factor of 2 (two $H(e)$ terms)?

### Problem 18: Photon Number Splitting Attack
In practice, Alice uses weak coherent pulses with mean photon number $\mu$.
a) What is the probability of sending exactly $n$ photons?
b) Describe the PNS attack
c) How does the decoy state protocol defend against PNS?

### Problem 19: Finite Key Effects
For a finite number of transmitted qubits $N$:
a) How does the key rate formula change?
b) What is the minimum $N$ required to establish a secure key?
c) Discuss the trade-off between key length and security level

---

## Section D: E91 Protocol and Bell Inequalities (Problems 20-23)

### Problem 20: E91 Protocol Analysis
a) Describe the E91 protocol in detail
b) What measurement bases do Alice and Bob use?
c) How is the key extracted from their measurements?

### Problem 21: CHSH Violation and Security
a) Derive the CHSH inequality $S \leq 2$ for local hidden variable theories
b) Calculate the quantum value $S = 2\sqrt{2}$ for the optimal measurement angles
c) Explain how CHSH violation implies security against eavesdroppers

### Problem 22: Device-Independent Security
a) What does "device-independent" security mean?
b) Why does CHSH violation prove security even with untrusted devices?
c) What loopholes must be closed for device-independent QKD?

### Problem 23: BB84 vs E91 Comparison
Create a detailed comparison of BB84 and E91:
a) Resource requirements
b) Security assumptions
c) Key generation rate
d) Implementation complexity
e) Resistance to specific attacks

---

## Section E: Blind Quantum Computation (Problems 24-26)

### Problem 24: UBQC Protocol
a) Describe the Universal Blind Quantum Computation protocol
b) What does the client prepare and send?
c) How does the server perform computation without learning the input?

### Problem 25: Blindness Proof
a) Define what "blindness" means formally
b) Show that from the server's perspective, measurement angles are uniformly random
c) Why does randomization of angles not affect the computation for the client?

### Problem 26: Verification in BQC
a) Describe how trap qubits work for verification
b) What is the probability of catching a cheating server?
c) How can this be amplified?

---

## Section F: Integration Problems (Problems 27-28)

### Problem 27: Protocol Connections
a) Show how teleportation can be used to build a quantum repeater
b) Explain how QKD relates to teleportation (are they related?)
c) Can blind quantum computation be used for secure QKD? Why or why not?

### Problem 28: Research Frontier
Consider a quantum network with multiple parties.
a) How would you establish pairwise secret keys among $n$ parties?
b) What are the resource requirements (entanglement, classical communication)?
c) Discuss the concept of "conference key agreement" where all parties share one key
d) What are the open problems in quantum network protocols?

---

## Problem Summary

| Section | Problems | Topics | Points |
|---------|----------|--------|--------|
| A | 1-9 | Teleportation | 35 |
| B | 10-13 | Superdense Coding | 15 |
| C | 14-19 | BB84 QKD | 25 |
| D | 20-23 | E91 and Bell | 15 |
| E | 24-26 | Blind QC | 10 |
| F | 27-28 | Integration | 0 (bonus) |

**Total: 100 points**

---

## Submission Guidelines

For qualifying exam preparation:
1. Complete derivations with all steps shown
2. Time yourself under exam conditions
3. Practice explaining solutions orally
4. Focus on Problems 1, 4, 6, 14, 15, 17, 20, 21 as core exam topics

---

**Created:** February 9, 2026
**Estimated Time:** 15-20 hours
