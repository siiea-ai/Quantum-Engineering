# Week 167: Quantum Information Theory - Oral Exam Practice

## Overview

This document contains oral examination practice for quantum information theory, covering entropy, Holevo bound, and channel capacity.

---

## Question 1: Von Neumann Entropy

### Main Question
"Define von Neumann entropy and explain its key properties."

### Response Framework

**Definition (30 seconds):**
"Von Neumann entropy is the quantum analog of Shannon entropy:
$$S(\rho) = -\text{Tr}(\rho \log \rho) = -\sum_i \lambda_i \log \lambda_i$$
where $\lambda_i$ are eigenvalues of the density matrix."

**Properties (2 minutes):**
"Key properties include:

1. **Non-negativity:** $S(\rho) \geq 0$, with equality for pure states.

2. **Maximum value:** $S(\rho) \leq \log d$ for $d$-dimensional systems, achieved by the maximally mixed state.

3. **Concavity:** $S(\sum_i p_i \rho_i) \geq \sum_i p_i S(\rho_i)$. Mixing increases entropy.

4. **Subadditivity:** $S(AB) \leq S(A) + S(B)$, with equality for product states.

5. **Strong subadditivity:** $S(ABC) + S(B) \leq S(AB) + S(BC)$. This is the deepest entropy inequality."

**Physical interpretation:**
"Entropy measures uncertainty about the quantum state, or equivalently, the 'mixedness' of the state. For bipartite pure states, the reduced entropy measures entanglement."

### Follow-ups

**Q: "What's unusual about quantum conditional entropy?"**

A: "Unlike classical conditional entropy, quantum $S(A|B) = S(AB) - S(B)$ can be negative. For maximally entangled states, $S(A|B) = -\log d$. Negative conditional entropy signals entanglement—knowing $B$ tells you more about $A$ than classical correlations could."

**Q: "Prove subadditivity."**

A: "Use strong subadditivity with trivial system $C$. Alternatively, use the relation $I(A:B) = S(A) + S(B) - S(AB) \geq 0$, which follows from relative entropy non-negativity."

---

## Question 2: Holevo Bound

### Main Question
"State and explain the Holevo bound."

### Response Framework

**Statement (1 minute):**
"For an ensemble $\{p_x, \rho_x\}$ where Alice encodes message $x$ into quantum state $\rho_x$, and Bob performs any measurement getting outcome $Y$:

$$I(X:Y) \leq \chi(\{p_x, \rho_x\})$$

where the Holevo quantity is:
$$\chi = S(\rho) - \sum_x p_x S(\rho_x)$$

with $\rho = \sum_x p_x \rho_x$."

**Interpretation (1 minute):**
"This bounds the accessible information—classical information extractable from quantum states.

For orthogonal pure states: $\chi = H(p)$ (full Shannon entropy)
For non-orthogonal states: $\chi < H(p)$ (some information is lost)
For mixed states: further reduction due to intrinsic uncertainty"

**Significance:**
"The bound shows why you can't send more than $\log d$ bits through a $d$-dimensional quantum system. It's the foundation of classical capacity theory."

### Follow-ups

**Q: "When is the bound achievable?"**

A: "For single uses, orthogonal pure states achieve it. For general ensembles, the HSW theorem shows $\chi$ is achievable asymptotically with block coding."

**Q: "Apply this to superdense coding."**

A: "With shared entanglement, Bob receives one of four orthogonal Bell states. The Holevo quantity is $\chi = \log 4 = 2$ bits. Bell measurement achieves this, extracting 2 bits from 1 qubit—the superdense coding advantage."

---

## Question 3: Channel Capacity

### Main Question
"Compare classical and quantum channel capacity."

### Response Framework

**Classical capacity (1 minute):**
"The classical capacity $C(\mathcal{N})$ is the maximum rate for transmitting classical bits through a quantum channel:

$$C(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \chi^*(\mathcal{N}^{\otimes n})$$

The regularization is needed because the Holevo quantity isn't always additive."

**Quantum capacity (1 minute):**
"The quantum capacity $Q(\mathcal{N})$ measures the rate for transmitting qubits:

$$Q(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} I_c^*(\mathcal{N}^{\otimes n})$$

where $I_c$ is the coherent information. Crucially, $Q$ can be zero even when $C > 0$—entanglement-breaking channels have $Q = 0$."

**Entanglement-assisted (30 seconds):**
"With pre-shared entanglement:
$$C_E = \max_\rho I(A:B)$$

This is additive and always exceeds $C$. For noiseless qubits, $C_E = 2$ (superdense coding)."

### Follow-ups

**Q: "Why isn't channel capacity additive?"**

A: "The Holevo quantity can be superadditive: $\chi(\mathcal{N}^{\otimes 2}) > 2\chi(\mathcal{N})$ for some channels. This happens because entangled inputs can extract more information than product inputs."

**Q: "Give an example of zero quantum capacity."**

A: "The 50% depolarizing channel has $Q = 0$ because it's entanglement-breaking—any entanglement with a reference system is destroyed. Yet $C > 0$ because classical information can still pass through."

---

## Question 4: Quantum Data Compression

### Main Question
"Explain Schumacher compression."

### Response Framework

**Classical context (30 seconds):**
"Shannon showed classical sources can be compressed to $H(X)$ bits per symbol—the entropy rate."

**Quantum result (1-2 minutes):**
"Schumacher extended this to quantum:

**Theorem:** A quantum source with density matrix $\rho$ can be compressed to $S(\rho)$ qubits per symbol asymptotically.

The key insight is the typical subspace. For $n$ emissions:
- Total Hilbert space: dimension $d^n$
- Typical subspace: dimension $\approx 2^{nS(\rho)}$

We project onto the typical subspace and encode it efficiently."

**Compression protocol:**
"1. Receive $n$ source states
2. Project onto typical subspace
3. Encode in $nS(\rho)$ qubits
4. Transmit
5. Decode by reversing the encoding

Fidelity approaches 1 as $n \to \infty$."

### Follow-ups

**Q: "What's the typical subspace?"**

A: "It's spanned by eigenstates of $\rho^{\otimes n}$ with eigenvalues close to $2^{-nS(\rho)}$. By the law of large numbers, almost all probability weight falls here for large $n$."

**Q: "When is quantum compression better than classical?"**

A: "When the source has quantum coherence. An ensemble of non-orthogonal pure states has $S(\rho) < H(p)$ (the preparation entropy). Quantum compression exploits this."

---

## Question 5: Strong Subadditivity

### Main Question
"What is strong subadditivity and why is it important?"

### Response Framework

**Statement (30 seconds):**
"For any tripartite state $\rho_{ABC}$:
$$S(ABC) + S(B) \leq S(AB) + S(BC)$$

Equivalently: $I(A:C|B) \geq 0$ (conditional mutual information is non-negative)."

**Importance (1-2 minutes):**
"This is the most important entropy inequality:

1. **Foundation:** Many quantum information results derive from it
2. **Data processing:** Implies information can't increase under local operations
3. **Physics:** Constrains entropy in many-body systems

The proof (Lieb-Ruskai, 1973) is nontrivial—classical proofs don't generalize."

**Applications:**
"- Proves subadditivity as special case
- Bounds on channel capacity
- Area laws in many-body physics
- Quantum Markov chains"

### Follow-ups

**Q: "What's the data processing inequality?"**

A: "If $A \to B \to C$ forms a Markov chain, then $I(A:C) \leq I(A:B)$. Processing $B$ cannot increase information about $A$. This follows from strong subadditivity."

---

## Question 6: Connecting Concepts

### Main Question
"How do entropy, Holevo bound, and channel capacity relate?"

### Response Framework

**Unified picture (2 minutes):**
"These form a hierarchy:

**Entropy:** Fundamental measure of uncertainty
- Quantifies information content of states
- Measures entanglement for pure bipartite states

**Holevo bound:** Uses entropy to bound communication
- $\chi = S(\rho) - \sum_i p_i S(\rho_i)$
- Maximum classical information from quantum encoding

**Channel capacity:** Optimizes over encodings
- Classical capacity: optimize Holevo quantity
- Quantum capacity: preserve coherence
- Both require regularization in general

**Physical insight:** Quantum mechanics limits information extraction. Entropy quantifies this, Holevo bounds it, capacity optimizes it."

---

## Quick Response Practice

For 1-minute answers:

**"What is von Neumann entropy?"**
"$S(\rho) = -\text{Tr}(\rho \log \rho)$. It measures uncertainty about quantum states. Zero for pure states, maximum $\log d$ for maximally mixed. Measures entanglement for bipartite pure states."

**"State the Holevo bound."**
"For ensemble $\{p_x, \rho_x\}$, accessible information $I(X:Y) \leq S(\rho) - \sum_x p_x S(\rho_x)$ where $\rho$ is the average state. Bounds classical information extractable from quantum encoding."

**"What's special about quantum conditional entropy?"**
"It can be negative—$S(A|B)$ ranges from $-\log d$ to $\log d$. Negative values indicate entanglement: knowing $B$ gives more information about $A$ than A's own entropy would suggest."

**"What is strong subadditivity?"**
"$S(ABC) + S(B) \leq S(AB) + S(BC)$. The deepest entropy inequality. Implies data processing inequality and underlies many capacity proofs."

---

## Self-Assessment

After practice, can you:

- [ ] Define von Neumann entropy and list 4 properties
- [ ] State Holevo bound with correct formula
- [ ] Explain negative conditional entropy
- [ ] State strong subadditivity
- [ ] Explain Schumacher compression
- [ ] Compare $C$, $Q$, $C_E$ for channels
- [ ] Give example of capacity calculation

---

**Created:** February 9, 2026
