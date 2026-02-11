# Week 166: Quantum Protocols - Review Guide

## Introduction

Quantum protocols leverage quantum mechanical phenomena—particularly entanglement and the no-cloning theorem—to achieve tasks impossible or impractical classically. This review covers the fundamental quantum communication and cryptographic protocols essential for PhD qualifying examinations.

These protocols form the foundation of quantum networks, quantum cryptography, and distributed quantum computing. Understanding them deeply means grasping both the mathematical formalism and the physical principles that enable them.

---

## 1. Quantum Teleportation

### 1.1 The Teleportation Problem

**Goal:** Transfer an unknown quantum state $|\psi\rangle$ from Alice to Bob using only:
- Pre-shared entanglement
- Classical communication

**Constraints:**
- No-cloning theorem: Cannot copy unknown quantum states
- No faster-than-light signaling: Classical communication required

### 1.2 The Protocol

**Setup:**
- Alice has unknown state: $|\psi\rangle_A = \alpha|0\rangle + \beta|1\rangle$
- Alice and Bob share Bell state: $|\Phi^+\rangle_{A'B} = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$

**Total initial state:**
$$|\Psi\rangle = |\psi\rangle_A \otimes |\Phi^+\rangle_{A'B} = (\alpha|0\rangle + \beta|1\rangle) \otimes \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

**Step 1: Rewrite in Bell basis**

Expand in the Bell basis for qubits $A$ and $A'$:
$$|\Psi\rangle = \frac{1}{2}\Big[|\Phi^+\rangle_{AA'}(\alpha|0\rangle_B + \beta|1\rangle_B) + |\Phi^-\rangle_{AA'}(\alpha|0\rangle_B - \beta|1\rangle_B)$$
$$+ |\Psi^+\rangle_{AA'}(\alpha|1\rangle_B + \beta|0\rangle_B) + |\Psi^-\rangle_{AA'}(\alpha|1\rangle_B - \beta|0\rangle_B)\Big]$$

**Step 2: Alice performs Bell measurement**

Alice measures qubits $A$ and $A'$ in the Bell basis, obtaining one of four outcomes with probability 1/4 each.

**Step 3: Classical communication**

Alice sends 2 classical bits encoding her measurement outcome to Bob.

**Step 4: Bob applies correction**

| Alice's Outcome | Bob's State | Bob's Correction |
|-----------------|-------------|------------------|
| $\|\Phi^+\rangle$ | $\alpha\|0\rangle + \beta\|1\rangle$ | $I$ (nothing) |
| $\|\Phi^-\rangle$ | $\alpha\|0\rangle - \beta\|1\rangle$ | $Z$ |
| $\|\Psi^+\rangle$ | $\alpha\|1\rangle + \beta\|0\rangle$ | $X$ |
| $\|\Psi^-\rangle$ | $\alpha\|1\rangle - \beta\|0\rangle$ | $XZ$ |

**Result:** Bob has $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$

### 1.3 Resource Analysis

**Resources consumed:**
- 1 ebit (maximally entangled pair)
- 2 classical bits (measurement outcome)

**Resource equation:**
$$\boxed{1 \text{ ebit} + 2 \text{ cbits} \rightarrow 1 \text{ qubit}}$$

**Why classical communication is necessary:**
Without it, Bob's state is maximally mixed (density matrix $I/2$). Classical bits provide the "key" to decode the quantum information.

### 1.4 Teleportation Fidelity

**Definition:**
$$F = \langle\psi|\rho_{\text{out}}|\psi\rangle$$

**Perfect teleportation:** $F = 1$

**With noisy entanglement:**

If the shared state is $\rho_{AB}$ instead of pure $|\Phi^+\rangle\langle\Phi^+|$:

$$F = \frac{1}{2} + \frac{1}{2}\text{Tr}[\rho_{AB}(X \otimes X + Y \otimes Y + Z \otimes Z)]$$

**For Werner state** $\rho_W = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$:
$$F = \frac{1 + 3p}{4}$$

Requires $p > 1/2$ to beat classical limit $F_{\text{classical}} = 2/3$.

### 1.5 Extensions

**Teleportation with non-maximally entangled states:**
$$|\phi\rangle = \sqrt{p}|00\rangle + \sqrt{1-p}|11\rangle$$

Success probability is reduced; can use probabilistic teleportation or entanglement distillation.

**Gate teleportation:**
Teleporting through a gate: $U|\psi\rangle$ can be teleported using modified corrections.

**Entanglement swapping:**
Teleporting entanglement: If Charlie has $|\Phi^+\rangle_{AC}$ and $|\Phi^+\rangle_{BD}$, measuring $C$ and $D$ in Bell basis creates entanglement between $A$ and $B$ (who never interacted).

---

## 2. Superdense Coding

### 2.1 The Protocol

**Goal:** Send 2 classical bits using 1 qubit (pre-shared entanglement helps).

**Setup:**
- Alice and Bob share $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
- Alice wants to send 2 bits: $b_1 b_2 \in \{00, 01, 10, 11\}$

**Protocol:**
1. Alice applies one of four unitaries to her qubit:
   - $00 \to I$: $|\Phi^+\rangle$
   - $01 \to X$: $|\Psi^+\rangle$
   - $10 \to Z$: $|\Phi^-\rangle$
   - $11 \to XZ$: $|\Psi^-\rangle$

2. Alice sends her qubit to Bob

3. Bob performs Bell measurement, recovering $b_1 b_2$

**Resource equation:**
$$\boxed{1 \text{ qubit} + 1 \text{ ebit} \rightarrow 2 \text{ cbits}}$$

### 2.2 Duality with Teleportation

| Teleportation | Superdense Coding |
|---------------|-------------------|
| 1 ebit + 2 cbits → 1 qubit | 1 qubit + 1 ebit → 2 cbits |
| Alice measures, Bob corrects | Alice encodes, Bob measures |
| Quantum → Classical → Quantum | Classical → Quantum → Classical |

The two protocols are "duals" in resource trade-offs.

### 2.3 Higher Dimensions

For $d$-dimensional systems (qudits):
- Superdense coding can send $2\log_2 d$ classical bits using one qudit
- Requires maximally entangled state in $d \times d$ dimensions
- Achieves Holevo bound for entanglement-assisted classical capacity

---

## 3. BB84 Quantum Key Distribution

### 3.1 Protocol Description

**Goal:** Establish shared secret key between Alice and Bob, secure against eavesdropper Eve.

**Step 1: Quantum transmission**
1. Alice randomly generates bit string $a = (a_1, a_2, \ldots, a_n)$
2. Alice randomly chooses bases $b = (b_1, b_2, \ldots, b_n)$ where $b_i \in \{Z, X\}$
3. Alice prepares qubits:
   - $a_i = 0, b_i = Z$: $|0\rangle$
   - $a_i = 1, b_i = Z$: $|1\rangle$
   - $a_i = 0, b_i = X$: $|+\rangle$
   - $a_i = 1, b_i = X$: $|-\rangle$
4. Alice sends qubits to Bob

**Step 2: Bob's measurement**
1. Bob randomly chooses bases $b' = (b'_1, \ldots, b'_n)$
2. Bob measures each qubit in his chosen basis
3. Bob records results $a' = (a'_1, \ldots, a'_n)$

**Step 3: Sifting**
1. Alice and Bob publicly announce their bases $b$ and $b'$
2. Discard bits where $b_i \neq b'_i$
3. When $b_i = b'_i$: $a_i = a'_i$ (in ideal case)

**Step 4: Error estimation**
1. Sacrifice subset of sifted key for error checking
2. If error rate too high, abort (eavesdropper detected)

**Step 5: Privacy amplification**
1. Apply hash function to remaining key
2. Output shorter but more secure key

### 3.2 Security Analysis

**Intercept-resend attack:**

Eve intercepts qubit, measures in random basis, resends:
- Probability of correct basis: 1/2
- When Eve's basis is wrong: 50% error introduced
- Overall error rate: $1/2 \times 1/2 = 25\%$

**Detection:** Alice and Bob compare subset of key; error rate reveals eavesdropping.

**Information-theoretic security:**

No-cloning theorem prevents Eve from copying qubits.
Measurement disturbs quantum states, revealing Eve's presence.

**Key rate (asymptotic):**
$$R = 1 - H(e) - H(e)$$

where $e$ is bit error rate, $H$ is binary entropy.

For $e = 0$: $R = 1$ (full key rate)
For $e = 11\%$: $R = 0$ (threshold)

### 3.3 Practical Attacks

**Photon Number Splitting (PNS):**
- Weak laser pulses sometimes emit 2+ photons
- Eve keeps one, forwards the rest
- Countermeasure: Decoy state protocol

**Detector blinding:**
- Eve controls Bob's detectors using bright light
- Bob clicks deterministically based on Eve's choice
- Countermeasure: Detector monitoring, MDI-QKD

**Side-channel attacks:**
- Timing, power consumption, electromagnetic emissions
- Require careful implementation

---

## 4. E91 Protocol

### 4.1 Protocol Description

**Setup:**
- Source distributes entangled pairs $|\Phi^+\rangle$ to Alice and Bob

**Step 1: Measurements**
- Alice measures in one of: $\{0°, 45°, 90°\}$ (Z, diagonal, X)
- Bob measures in one of: $\{45°, 90°, 135°\}$

**Step 2: Sifting**
- When both use 45° or 90°: use for key (perfect anti-correlation or correlation)

**Step 3: CHSH test**
- Other measurement combinations: check CHSH inequality
- Violation confirms quantum correlations, rules out local hidden variables (and eavesdroppers)

### 4.2 CHSH Inequality

**Classical bound:**
$$S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| \leq 2$$

**Quantum bound (Tsirelson):**
$$S \leq 2\sqrt{2} \approx 2.83$$

**Security connection:**
- Eve's information bounded by how much she disturbs correlations
- CHSH violation → limited eavesdropping → secure key

### 4.3 Device-Independent Security

**Concept:** Security relies only on observed statistics, not on trusting devices.

**Advantage:** Even if Eve manufactured Alice and Bob's devices, Bell violation proves security.

**Requirement:** Loophole-free Bell test (locality, detection efficiency).

### 4.4 Comparison: BB84 vs E91

| Aspect | BB84 | E91 |
|--------|------|-----|
| Entanglement | Not required | Required |
| Security basis | No-cloning | Bell inequality |
| Device trust | Required | Can be device-independent |
| Implementation | Simpler | More complex |
| Key rate | Higher | Lower (more measurements wasted) |

---

## 5. Blind Quantum Computation

### 5.1 The Problem

**Scenario:**
- Client (Alice) has limited quantum resources
- Server (Bob) has full quantum computer
- Alice wants Bob to compute $U|\psi\rangle$ without Bob learning $U$ or $|\psi\rangle$

**Classical analogy:** Encrypted computation (homomorphic encryption), but quantum version.

### 5.2 Universal Blind Quantum Computation (UBQC)

**Key insight:** Use measurement-based quantum computation (MBQC).

**MBQC background:**
- Prepare large entangled "cluster state"
- Computation via single-qubit measurements
- Measurement angles determine computation

**UBQC protocol (Broadbent-Fitzsimons-Kashefi):**

1. **Client prepares:** Qubits in states $|+_\theta\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\theta}|1\rangle)$ with random $\theta$

2. **Client sends to server:** These prepared qubits (server doesn't know $\theta$ values)

3. **Server entangles:** Creates cluster state (this step is "blind" to computation)

4. **Client instructs:** Sends measurement angles $\phi' = \phi + \theta + r\pi$ where:
   - $\phi$ is the actual computation angle
   - $\theta$ is the preparation randomness
   - $r \in \{0,1\}$ is random bit for outcome hiding

5. **Server measures:** Returns outcomes to client

6. **Client decodes:** Uses knowledge of $\theta$, $r$ to interpret results

### 5.3 Security Properties

**Blindness:**
Server learns nothing about:
- Input state
- Computation being performed
- Output

*Proof idea:* From server's view, measurement angles are uniformly random (masked by $\theta$).

**Verifiability:**
Can detect if server deviates from protocol.

*Technique:* Insert "trap qubits" with known measurement outcomes. Server's deviation reveals itself.

### 5.4 Resource Requirements

**Client needs:**
- Prepare single qubits in $|+_\theta\rangle$ states
- Classical computation (polynomial)

**Communication:**
- Quantum: $O(n)$ qubits sent to server
- Classical: $O(n)$ bits per measurement round

### 5.5 Variants and Extensions

**Classical client BQC:**
Can client with NO quantum resources achieve blindness? Requires either:
- Multiple non-communicating servers
- Computational assumptions

**Verified BQC:**
Combining blindness with verifiable computation.

---

## 6. Quantum Repeaters and Networks

### 6.1 The Problem

**Challenge:** Photon loss in fiber: ~0.2 dB/km
- 100 km: 99% loss
- 1000 km: 10^{-10} transmission

**Classical solution:** Amplifiers (read, amplify, retransmit)
**Quantum problem:** No-cloning prevents amplification

### 6.2 Quantum Repeater Concept

**Key ideas:**
1. **Segmentation:** Divide channel into shorter segments
2. **Entanglement distribution:** Create entanglement across each segment
3. **Entanglement swapping:** Connect segments via Bell measurements
4. **Entanglement distillation:** Purify noisy entanglement

**Result:** End-to-end entanglement with polynomial (not exponential) overhead.

### 6.3 Connection to Protocols

Teleportation + entanglement swapping = building blocks for quantum networks.

---

## 7. Exam Preparation

### Key Derivations to Master

1. **Teleportation algebra:** Complete derivation showing Bob's state for each outcome
2. **BB84 error rate:** Calculate error introduced by intercept-resend
3. **CHSH value:** Calculate quantum violation for optimal angles
4. **Superdense coding:** Show encoding and decoding

### Common Exam Questions

1. "Derive the teleportation protocol and explain why classical communication is necessary"
2. "Prove that BB84 can detect an eavesdropper"
3. "Explain the connection between CHSH violation and E91 security"
4. "What is blind quantum computation and why is it important?"

### Conceptual Points

1. **Teleportation doesn't violate relativity:** Requires classical communication
2. **QKD is information-theoretically secure:** Not based on computational assumptions
3. **Entanglement is a resource:** Consumed in protocols, must be established first
4. **Device-independent security:** Strongest form of security in cryptography

---

## 8. Summary

### Protocol Comparison

| Protocol | Input Resources | Output | Key Principle |
|----------|-----------------|--------|---------------|
| Teleportation | 1 ebit + 2 cbits | 1 qubit | Bell measurement |
| Superdense Coding | 1 qubit + 1 ebit | 2 cbits | Unitary encoding |
| BB84 | Qubits in 2 bases | Shared key | No-cloning |
| E91 | Entangled pairs | Shared key | Bell inequality |
| Blind QC | Prepared qubits | Computation | MBQC + randomness |

### Resource Trade-offs

$$\text{Teleportation} \leftrightarrow \text{Superdense Coding}$$

$$1 \text{ ebit} + 2 \text{ cbits} \leftrightarrow 1 \text{ qubit} + 1 \text{ ebit} \leftrightarrow 2 \text{ cbits}$$

---

## References

1. Bennett, C.H., et al. "Teleporting an Unknown Quantum State via Dual Classical and Einstein-Podolsky-Rosen Channels." *PRL* 70, 1895 (1993).

2. Bennett, C.H., Brassard, G. "Quantum Cryptography: Public Key Distribution and Coin Tossing." *Proceedings of IEEE ICCSSP* (1984).

3. Ekert, A.K. "Quantum Cryptography Based on Bell's Theorem." *PRL* 67, 661 (1991).

4. Broadbent, A., Fitzsimons, J., Kashefi, E. "Universal Blind Quantum Computation." *FOCS* (2009).

5. Nielsen, M.A., Chuang, I.L. *Quantum Computation and Quantum Information.* Cambridge (2010).

---

**Word Count:** ~2800 words
**Created:** February 9, 2026
