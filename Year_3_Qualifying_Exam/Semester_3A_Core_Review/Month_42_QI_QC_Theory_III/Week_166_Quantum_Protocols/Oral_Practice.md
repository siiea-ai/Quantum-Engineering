# Week 166: Quantum Protocols - Oral Exam Practice

## Overview

This document contains oral examination practice questions for quantum protocols. Each question includes response frameworks, key derivations, and anticipated follow-ups.

**Format:** PhD qualifying exam oral (15-20 minutes per topic)

---

## Question 1: Derive Quantum Teleportation

### Main Question
"Walk me through the quantum teleportation protocol."

### Response Framework

**Setup (30 seconds):**
"Teleportation transfers an unknown quantum state from Alice to Bob using shared entanglement and classical communication."

**Protocol (2-3 minutes):**

"Let Alice have unknown state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ and share Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ with Bob.

Step 1: The total state is $|\psi\rangle \otimes |\Phi^+\rangle$.

Step 2: Rewrite Alice's qubits in Bell basis..."

*Write out the Bell expansion:*
$$|\Psi\rangle = \frac{1}{2}\sum_{ij} |\Phi_{ij}\rangle_{AA'} \otimes \sigma_{ij}|\psi\rangle_B$$

"Step 3: Alice measures in Bell basis, getting outcome $ij$.

Step 4: She sends 2 classical bits to Bob.

Step 5: Bob applies $\sigma_{ij}^{-1}$ to recover $|\psi\rangle$."

**Key insight (30 seconds):**
"The classical communication is essential—without it, Bob's state is maximally mixed. This prevents faster-than-light signaling."

### Potential Follow-ups

**Q: "Why can't we teleport without classical communication?"**

A: "Before Alice sends her measurement outcome, Bob's reduced state is $\rho_B = I/2$, independent of $|\psi\rangle$. Each Bell measurement outcome is equally likely, and the four possible states average to the maximally mixed state."

**Q: "What resources are consumed?"**

A: "One ebit of entanglement and two classical bits. The resource equation is: 1 ebit + 2 cbits → 1 qubit. This is dual to superdense coding."

**Q: "What happens with noisy entanglement?"**

A: "The teleportation fidelity decreases. For Werner state with parameter $p$, fidelity is $F = (1+p)/2$. To beat classical limit of $F = 2/3$, we need $p > 1/3$."

---

## Question 2: Explain BB84 Security

### Main Question
"How does BB84 ensure security against eavesdroppers?"

### Response Framework

**Protocol overview (1 minute):**
"BB84 uses two conjugate bases—usually Z ($|0\rangle, |1\rangle$) and X ($|+\rangle, |-\rangle$). Alice randomly picks bits and bases, prepares qubits, sends them to Bob. Bob measures in random bases. They publicly compare bases and keep matching ones."

**Security mechanism (2 minutes):**

"Security comes from the no-cloning theorem and information-disturbance principle.

Consider the intercept-resend attack: Eve measures in a random basis, then sends a qubit in the measured state.

When Eve guesses wrong (probability 1/2), she disturbs the state. For example, measuring $|0\rangle$ in X basis gives $|+\rangle$ or $|-\rangle$ with equal probability. Bob then has 50% chance of error when measuring in Z.

Overall error rate introduced: $\frac{1}{2} \times \frac{1}{2} = 25\%$

Alice and Bob compare a subset of their key. Error rate > 11% (security threshold) triggers abort."

**Why it works (30 seconds):**
"Any attack that gains information must disturb the quantum states because non-orthogonal states cannot be perfectly distinguished. High QBER implies eavesdropping; low QBER implies secure key."

### Potential Follow-ups

**Q: "Where does the 11% threshold come from?"**

A: "The asymptotic key rate is $R = 1 - 2H(e)$ where $H$ is binary entropy. At $e \approx 11\%$, $H(0.11) = 0.5$, giving $R = 0$. Above this error rate, no secure key can be extracted."

**Q: "What about practical attacks?"**

A: "Real implementations face photon number splitting attacks (weak coherent pulses emit multiple photons sometimes), detector blinding, and side channels. Countermeasures include decoy states, measurement-device-independent QKD, and careful engineering."

**Q: "Compare to E91."**

A: "E91 uses entangled pairs and Bell inequality violations to prove security. It's device-independent in principle—security relies on observed correlations, not trusting the devices. BB84 is simpler to implement but requires trusted devices."

---

## Question 3: CHSH and E91

### Main Question
"Explain the connection between Bell inequality violation and QKD security."

### Response Framework

**CHSH setup (1 minute):**
"CHSH tests correlations between distant measurements. Alice measures settings $a$ or $a'$, Bob measures $b$ or $b'$, each giving $\pm 1$.

Define $S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$

Classical (local hidden variables): $|S| \leq 2$
Quantum: $|S| \leq 2\sqrt{2}$"

**Connection to security (1-2 minutes):**

"In E91, Alice and Bob share entangled pairs. When they use matching bases, outcomes form the key. Different bases test CHSH.

High CHSH violation ($S \approx 2\sqrt{2}$) implies:
1. Strong quantum correlations
2. Minimal entanglement with any third party (Eve)

This follows from monogamy of entanglement—if Eve were entangled with the key, Alice-Bob correlations would weaken."

**Device independence (30 seconds):**
"CHSH violation proves security even with untrusted devices. If the statistics violate classical bounds, the correlations must be quantum, limiting Eve's information regardless of device internals."

### Potential Follow-ups

**Q: "Derive the classical bound $|S| \leq 2$."**

A: "For fixed hidden variable $\lambda$: outcomes are deterministic, $A, B \in \{\pm 1\}$.
$A(a)[B(b) - B(b')] + A(a')[B(b) + B(b')]$
Since B values are $\pm 1$: either the first bracket is $\pm 2$ and second is $0$, or vice versa. So $|\cdot| \leq 2$ for each $\lambda$, hence for the average."

**Q: "Calculate the quantum maximum."**

A: "For $|\Phi^+\rangle$: $E(\theta_A, \theta_B) = \cos(2\Delta\theta)$.
Optimal angles: $a=0°, a'=45°, b=22.5°, b'=67.5°$.
$S = \cos(45°) + \cos(45°) + \cos(45°) - \cos(135°) = 4 \times \frac{\sqrt{2}}{2} = 2\sqrt{2}$"

---

## Question 4: Superdense Coding

### Main Question
"Explain superdense coding and its relationship to teleportation."

### Response Framework

**Protocol (1-2 minutes):**
"Alice and Bob share $|\Phi^+\rangle$. Alice wants to send 2 classical bits.

She applies one of four Pauli operations to her qubit:
- $I$: $|\Phi^+\rangle$ (encodes 00)
- $X$: $|\Psi^+\rangle$ (encodes 01)
- $Z$: $|\Phi^-\rangle$ (encodes 10)
- $XZ$: $|\Psi^-\rangle$ (encodes 11)

She sends her qubit to Bob. Bob performs Bell measurement, perfectly distinguishing the four orthogonal states."

**Resource equation:**
"1 qubit + 1 ebit → 2 cbits"

**Duality (1 minute):**
"Compare with teleportation: 1 ebit + 2 cbits → 1 qubit.

They're duals—one trades quantum for classical resources, the other trades classical for quantum. Together, they show entanglement enables resource conversion between quantum and classical information."

### Potential Follow-ups

**Q: "Why can't we send more than 2 bits?"**

A: "Holevo bound limits classical information from one qubit to $\log d$ bits without entanglement. With one ebit, the limit doubles to $2\log d$. For qubits ($d=2$), maximum is 2 bits."

**Q: "What about noisy channels?"**

A: "With noisy entanglement, Bell states become less distinguishable. For Werner state with parameter $p$, the effective classical capacity decreases. Below some threshold, superdense coding offers no advantage over direct transmission."

---

## Question 5: Blind Quantum Computation

### Main Question
"What is blind quantum computation and why is it important?"

### Response Framework

**The problem (30 seconds):**
"A client with limited quantum resources wants a powerful server to perform quantum computation, but without revealing:
- The input
- The algorithm
- The output"

**Solution approach (1-2 minutes):**
"Universal Blind Quantum Computation (UBQC) uses measurement-based QC:

1. Client prepares single qubits in states $|+_\theta\rangle$ with random angles $\theta$
2. Client sends these to server
3. Server creates cluster state (entangles them)
4. Client instructs measurement angles: $\phi' = \phi + \theta + r\pi$ where $\phi$ is the actual computation angle and $r$ is random
5. Server measures and returns outcomes
6. Client decodes using knowledge of $\theta$ and $r$"

**Security (30 seconds):**
"From the server's view, all angles appear uniformly random. The randomness perfectly masks the computation. The client can also verify correctness using trap qubits."

**Importance (30 seconds):**
"As quantum cloud services develop, BQC enables secure delegation—users can harness powerful quantum computers without exposing sensitive computations. This is crucial for privacy-critical applications."

### Potential Follow-ups

**Q: "What quantum resources does the client need?"**

A: "Just single-qubit preparation in $|+_\theta\rangle$ states. No entanglement, no multi-qubit operations. Some protocols require even less—just random BB84 states."

**Q: "Can a fully classical client achieve blindness?"**

A: "Not with a single server (under standard assumptions). But with two non-communicating servers, classical-client BQC is possible. Alternatively, computational assumptions like LWE can enable classical-client protocols."

---

## Question 6: Comparing Protocols

### Main Question
"Compare BB84 and E91 protocols."

### Response Framework

Create a mental table:

| Aspect | BB84 | E91 |
|--------|------|-----|
| **Entanglement** | Not required | Required |
| **Prepare/Measure** | Alice prepares, Bob measures | Both measure |
| **Security basis** | No-cloning, disturbance | Bell inequality |
| **Device trust** | Required | Can be device-independent |
| **Key rate** | Higher | Lower |
| **Implementation** | Simpler | More complex |

**Key insight:**
"Both achieve information-theoretic security, but with different assumptions. BB84 is practical today; E91 offers stronger security guarantees in principle."

---

## Question 7: Research Discussion

### Main Question
"What are open problems in quantum communication protocols?"

### Response Points

**Practical challenges:**
- Long-distance QKD (quantum repeaters)
- Higher key rates
- Integration with classical networks

**Theoretical questions:**
- Optimal rates for noisy channels
- Multi-party protocols
- Composable security frameworks

**Emerging directions:**
- Satellite QKD (demonstrated)
- Quantum internet architectures
- Post-quantum cryptography interaction

---

## Tips for Oral Exams

### Do's
- Start with clear definitions
- Draw diagrams when helpful
- State results before proofs
- Connect to physical intuition
- Acknowledge when you're uncertain

### Don'ts
- Don't memorize scripts verbatim
- Don't skip steps in derivations
- Don't guess—reason through it
- Don't forget to answer the actual question

### Common Mistakes
- Confusing teleportation and superdense coding directions
- Forgetting classical communication in teleportation
- Mixing up QBER calculations
- Imprecise statements about security

---

## Self-Assessment Checklist

After practicing, verify you can:

- [ ] Derive teleportation from scratch
- [ ] Calculate QBER for intercept-resend attack
- [ ] Derive CHSH inequality (both bounds)
- [ ] Explain superdense coding
- [ ] Describe UBQC protocol
- [ ] Compare BB84 and E91
- [ ] Discuss practical limitations

---

**Created:** February 9, 2026
**Practice Time:** 2-3 hours recommended
