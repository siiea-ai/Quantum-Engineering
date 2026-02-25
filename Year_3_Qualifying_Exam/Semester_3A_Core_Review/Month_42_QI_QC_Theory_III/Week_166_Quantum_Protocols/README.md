# Week 166: Quantum Protocols

## Overview

**Days:** 1156-1162
**Theme:** Quantum Communication and Cryptographic Protocols
**Hours:** 45 hours (7.5 hours/day × 6 days)

---

## Learning Objectives

By the end of this week, you should be able to:

1. Derive the complete quantum teleportation protocol and analyze its resource requirements
2. Perform fidelity calculations for noisy teleportation channels
3. Explain superdense coding and its duality with teleportation
4. Derive the BB84 quantum key distribution protocol and its security
5. Analyze the E91 protocol and its connection to Bell inequalities
6. Explain blind quantum computation and its security guarantees
7. Compare and contrast different QKD protocols

---

## Daily Schedule

### Day 1156 (Monday): Quantum Teleportation - Complete Analysis

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Teleportation derivations |
| 2:00-5:00 | 3 hrs | Review: Bell states, entanglement as resource |
| 7:00-8:30 | 1.5 hrs | Computational: Teleportation circuit simulation |

**Topics:**
- Bell basis and Bell measurements
- Teleportation protocol step-by-step derivation
- Resource counting: 1 ebit + 2 cbits → 1 qubit
- Classical communication requirement (no FTL signaling)

### Day 1157 (Tuesday): Teleportation Extensions and Fidelity

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Noisy teleportation analysis |
| 2:00-5:00 | 3 hrs | Review: Non-maximally entangled states |
| 7:00-8:30 | 1.5 hrs | Oral practice: Explain teleportation |

**Topics:**
- Teleportation fidelity definition and calculation
- Effect of noisy entanglement on teleportation
- Teleportation with non-maximally entangled states
- Entanglement swapping

### Day 1158 (Wednesday): Superdense Coding

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Superdense coding variations |
| 2:00-5:00 | 3 hrs | Review: Duality with teleportation |
| 7:00-8:30 | 1.5 hrs | Written practice: Protocol comparisons |

**Topics:**
- Superdense coding protocol derivation
- Resource accounting: 1 qubit + 1 ebit → 2 cbits
- Higher-dimensional superdense coding
- Holevo bound implications

### Day 1159 (Thursday): BB84 Quantum Key Distribution

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: BB84 security analysis |
| 2:00-5:00 | 3 hrs | Review: Eavesdropping detection |
| 7:00-8:30 | 1.5 hrs | Computational: BB84 simulation |

**Topics:**
- BB84 protocol step-by-step
- Key rate and basis reconciliation
- Intercept-resend attacks and detection
- Photon number splitting attacks

### Day 1160 (Friday): E91 Protocol and Bell Inequalities

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: E91 security proofs |
| 2:00-5:00 | 3 hrs | Review: CHSH inequality connection |
| 7:00-8:30 | 1.5 hrs | Lab: Bell inequality verification |

**Topics:**
- E91 protocol using entangled pairs
- CHSH inequality as security test
- Device-independent QKD concepts
- Comparison with BB84

### Day 1161 (Saturday): Blind Quantum Computation

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: BQC protocol analysis |
| 2:00-5:00 | 3 hrs | Review: Security definitions |
| 7:00-8:30 | 1.5 hrs | Written practice: Security arguments |

**Topics:**
- Universal blind quantum computation (UBQC)
- Measurement-based quantum computation background
- Blindness: server learns nothing about computation
- Verifiability: detecting cheating servers

### Day 1162 (Sunday): Integration and Assessment

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Comprehensive problem set completion |
| 2:00-5:00 | 3 hrs | Oral exam practice session |
| 7:00-8:30 | 1.5 hrs | Self-assessment and gap analysis |

---

## Key Protocols Summary

### Quantum Teleportation

**Protocol:**
1. Alice and Bob share Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
2. Alice has unknown state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ to teleport
3. Alice performs Bell measurement on her two qubits
4. Alice sends 2 classical bits (measurement outcome) to Bob
5. Bob applies correction: $I$, $X$, $Z$, or $ZX$ based on outcome
6. Bob has $|\psi\rangle$

**Resource Equation:**
$$1 \text{ ebit} + 2 \text{ cbits} \rightarrow 1 \text{ qubit}$$

### Superdense Coding

**Protocol:**
1. Alice and Bob share Bell state $|\Phi^+\rangle$
2. Alice wants to send 2 classical bits $ab$
3. Alice applies: $I$ (00), $X$ (01), $Z$ (10), $ZX$ (11) to her qubit
4. Alice sends her qubit to Bob
5. Bob performs Bell measurement to recover $ab$

**Resource Equation:**
$$1 \text{ qubit} + 1 \text{ ebit} \rightarrow 2 \text{ cbits}$$

### BB84 Protocol

**Protocol:**
1. Alice randomly chooses bits and bases (Z or X)
2. Alice prepares qubits: $|0\rangle, |1\rangle, |+\rangle, |-\rangle$
3. Bob randomly measures in Z or X basis
4. Public comparison of bases (not bits)
5. Keep bits where bases matched
6. Sacrifice subset for error estimation
7. Privacy amplification

**Security:** Eavesdropping introduces errors (≥25% with intercept-resend)

### E91 Protocol

**Protocol:**
1. Source distributes entangled pairs $|\Phi^+\rangle$ to Alice and Bob
2. Each randomly measures in one of three bases
3. When bases match: use for key
4. When bases differ: use for CHSH test
5. CHSH violation proves quantum correlations (no eavesdropper)

**Security:** Based on monogamy of entanglement

---

## Key Equations

### Teleportation Fidelity
$$F = \langle\psi|\rho_{\text{out}}|\psi\rangle$$

For depolarizing channel with parameter $p$:
$$F = 1 - \frac{3p}{4}$$

### CHSH Inequality
$$|E(a,b) - E(a,b') + E(a',b) + E(a',b')| \leq 2 \text{ (classical)}$$
$$|E(a,b) - E(a,b') + E(a',b) + E(a',b')| \leq 2\sqrt{2} \text{ (quantum)}$$

### Key Rate (BB84, asymptotic)
$$R = 1 - H(e) - H(e)$$
where $e$ is the bit error rate and $H$ is binary entropy.

### Holevo Bound (for superdense coding)
$$I(X:B) \leq S(\rho) - \sum_x p_x S(\rho_x) \leq \log d$$

---

## Resources

### Primary Reading
- Nielsen & Chuang, Chapters 1.3.7, 2.3, 12
- Preskill, Ph219 Chapter 4
- Bennett & Brassard, "Quantum Cryptography" (original BB84)

### Research Papers
- Bennett et al., "Teleporting an Unknown Quantum State" (1993)
- Ekert, "Quantum Cryptography Based on Bell's Theorem" (1991)
- Broadbent, Fitzsimons, Kashefi, "Universal Blind Quantum Computation" (2009)

### Online Resources
- [IBM Quantum Learning: Teleportation](https://learning.quantum.ibm.com/)
- [Qiskit Textbook: QKD](https://qiskit.org/textbook)

---

## Connections

### From Previous Weeks
| Previous Topic | Connection to This Week |
|----------------|------------------------|
| Entanglement (Week 159) | Entanglement is the key resource |
| Quantum channels (Week 160) | Noisy teleportation analysis |
| Bell inequalities | Security of E91 |

### To Future Topics
| This Week's Topic | Future Application |
|-------------------|-------------------|
| Teleportation | Quantum repeaters, networks |
| QKD | Post-quantum cryptography |
| Blind QC | Quantum cloud computing |

---

**Created:** February 9, 2026
**Status:** Not Started
