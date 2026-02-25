# Week 167: Quantum Information Theory - Problem Set

## Instructions

This problem set contains 25 problems on quantum information theory. Problems are organized by topic:
- **Section A:** Von Neumann Entropy (Problems 1-8)
- **Section B:** Holevo Bound (Problems 9-14)
- **Section C:** Channel Capacity (Problems 15-19)
- **Section D:** Data Compression and Advanced Topics (Problems 20-25)

Time estimate: 15-20 hours total

---

## Section A: Von Neumann Entropy (Problems 1-8)

### Problem 1: Basic Entropy Calculation
Calculate the von Neumann entropy for each of the following states:

a) $\rho_1 = |0\rangle\langle 0|$

b) $\rho_2 = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|$

c) $\rho_3 = \frac{3}{4}|0\rangle\langle 0| + \frac{1}{4}|1\rangle\langle 1|$

d) $\rho_4 = \frac{1}{3}|0\rangle\langle 0| + \frac{1}{3}|1\rangle\langle 1| + \frac{1}{3}|2\rangle\langle 2|$

### Problem 2: Entropy of Bloch Sphere States
A qubit state can be written as:
$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$$
where $|\vec{r}| \leq 1$.

a) Find the eigenvalues of $\rho$ in terms of $|\vec{r}|$
b) Derive the entropy as a function of $r = |\vec{r}|$
c) Verify that $S = 0$ for pure states ($r = 1$) and $S = 1$ for maximally mixed ($r = 0$)

### Problem 3: Entropy and Measurement
Let $\rho = \frac{1}{2}|+\rangle\langle +| + \frac{1}{2}|-\rangle\langle -|$.

a) Calculate $S(\rho)$
b) Perform a measurement in the computational basis. What is the post-measurement entropy?
c) Prove that measurement (on average) never decreases entropy

### Problem 4: Concavity of Entropy
Prove that von Neumann entropy is concave:
$$S\left(\sum_i p_i \rho_i\right) \geq \sum_i p_i S(\rho_i)$$

*Hint:* Use Klein's inequality: $S(\rho \| \sigma) \geq 0$.

### Problem 5: Subadditivity
a) State the subadditivity inequality for bipartite systems
b) Prove it using strong subadditivity (take system $C$ to be trivial)
c) For what states does equality hold?

### Problem 6: Entanglement and Conditional Entropy
For the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

a) Calculate $S(AB)$
b) Calculate $S(A)$ and $S(B)$
c) Calculate the conditional entropy $S(A|B) = S(AB) - S(B)$
d) Explain why negative conditional entropy indicates entanglement

### Problem 7: Strong Subadditivity Application
Consider a tripartite pure state $|\psi\rangle_{ABC}$.

a) Use the Araki-Lieb inequality and purity to show $S(A) = S(BC)$
b) Apply strong subadditivity to derive $S(A) + S(C) \leq S(AB) + S(BC)$
c) For what states does equality hold in strong subadditivity?

### Problem 8: Entropy of Werner State
The Werner state is:
$$\rho_W = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I_4}{4}$$

a) Find the eigenvalues of $\rho_W$
b) Calculate $S(\rho_W)$ as a function of $p$
c) Calculate $S(\rho_A)$ where $\rho_A = \text{Tr}_B(\rho_W)$
d) For what range of $p$ is $S(A|B) < 0$?

---

## Section B: Holevo Bound (Problems 9-14)

### Problem 9: Holevo Bound Statement
a) State the Holevo bound precisely
b) Define the Holevo quantity $\chi$
c) Explain why this bounds the accessible information

### Problem 10: Orthogonal State Ensemble
Consider the ensemble $\{1/2, |0\rangle\langle 0|\}$, $\{1/2, |1\rangle\langle 1|\}$.

a) Calculate the Holevo quantity $\chi$
b) What measurement achieves the bound?
c) How much classical information can be extracted?

### Problem 11: Non-Orthogonal State Ensemble
Consider $\{1/2, |0\rangle\langle 0|\}$, $\{1/2, |+\rangle\langle +|\}$.

a) Calculate the average state $\rho$
b) Find the eigenvalues of $\rho$
c) Calculate $\chi$
d) Is there a measurement that achieves this bound?

### Problem 12: Holevo Bound for Qutrit
Alice encodes one of three messages using states:
- $\rho_0 = |0\rangle\langle 0|$ with probability $1/3$
- $\rho_1 = |1\rangle\langle 1|$ with probability $1/3$
- $\rho_2 = |2\rangle\langle 2|$ with probability $1/3$

a) Calculate $\chi$
b) How many bits of information can Bob extract?
c) Compare to the classical capacity of a noiseless qutrit channel

### Problem 13: Mixed State Ensemble
Consider the ensemble with equal probabilities:
- $\rho_0 = \frac{2}{3}|0\rangle\langle 0| + \frac{1}{3}|1\rangle\langle 1|$
- $\rho_1 = \frac{1}{3}|0\rangle\langle 0| + \frac{2}{3}|1\rangle\langle 1|$

a) Calculate the average state
b) Calculate $S(\rho_0)$ and $S(\rho_1)$
c) Calculate $\chi$
d) Is this larger or smaller than for pure state ensembles?

### Problem 14: Holevo Bound and Superdense Coding
In superdense coding with shared entanglement:

a) What is the effective ensemble from Bob's perspective?
b) Calculate the Holevo quantity
c) Verify that superdense coding achieves $\chi = 2$ bits

---

## Section C: Channel Capacity (Problems 15-19)

### Problem 15: Noiseless Channel Capacity
For a noiseless qubit channel $\mathcal{I}(\rho) = \rho$:

a) Calculate the classical capacity $C$
b) Calculate the quantum capacity $Q$
c) Calculate the entanglement-assisted capacity $C_E$

### Problem 16: Depolarizing Channel
The depolarizing channel is:
$$\mathcal{D}_p(\rho) = (1-p)\rho + p\frac{I}{2}$$

a) Calculate the output entropy $S(\mathcal{D}_p(\rho))$ for input $|0\rangle\langle 0|$
b) For small $p$, estimate the classical capacity
c) For what value of $p$ does the quantum capacity become zero?

### Problem 17: Amplitude Damping Channel
The amplitude damping channel has Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

a) Verify these satisfy $\sum_i K_i^\dagger K_i = I$
b) Calculate the output state for input $|1\rangle$
c) Is this channel entanglement-breaking for any $\gamma$?

### Problem 18: Phase Damping Channel
The phase damping channel is:
$$\mathcal{P}_\lambda(\rho) = (1-\lambda)\rho + \lambda Z\rho Z$$

a) Calculate the output for a general qubit state
b) What happens to off-diagonal elements?
c) Calculate the classical capacity

### Problem 19: Entanglement-Assisted Capacity
For a channel $\mathcal{N}$, the entanglement-assisted capacity is:
$$C_E(\mathcal{N}) = \max_\rho I(A:B)$$

a) Explain why this doesn't require regularization
b) For the depolarizing channel with $p = 1/4$, calculate $C_E$
c) Compare $C_E$ to $C$ for this channel

---

## Section D: Data Compression and Advanced Topics (Problems 20-25)

### Problem 20: Schumacher Compression
A quantum source emits states $|0\rangle$ and $|+\rangle$ with equal probability.

a) Calculate the density matrix $\rho$ of the source
b) Find the eigenvalues and eigenvectors of $\rho$
c) Calculate the compression rate (qubits per symbol)
d) Describe the typical subspace for $n$ emissions

### Problem 21: Classical vs. Quantum Compression
Compare Shannon and Schumacher compression:

a) For a classical source with distribution $p(x)$, what is the compression rate?
b) For a quantum source with density matrix $\rho$, what is the compression rate?
c) When does quantum compression offer an advantage?

### Problem 22: Relative Entropy
Calculate the relative entropy $S(\rho \| \sigma)$ for:

a) $\rho = |0\rangle\langle 0|$, $\sigma = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|$
b) $\rho = \frac{3}{4}|0\rangle\langle 0| + \frac{1}{4}|1\rangle\langle 1|$, $\sigma = \frac{1}{2}I$
c) Verify $S(\rho \| \sigma) \geq 0$ in both cases

### Problem 23: Data Processing Inequality
Prove the data processing inequality:
$$I(X:Y) \geq I(X:Z)$$

when $X \to Y \to Z$ forms a Markov chain (processing $Y$ cannot increase information about $X$).

*Hint:* Use strong subadditivity.

### Problem 24: Entanglement of Formation
For a two-qubit state $\rho$, the entanglement of formation is:
$$E_F(\rho) = \min \sum_i p_i S(\text{Tr}_B(|\psi_i\rangle\langle\psi_i|))$$

over all decompositions $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$.

a) Calculate $E_F$ for a pure state $|\psi\rangle$
b) Calculate $E_F$ for the Werner state with $p = 1$
c) Explain why calculating $E_F$ for general mixed states is hard

### Problem 25: Synthesis Problem
A quantum communication system uses:
- A noisy quantum channel $\mathcal{N}$ with classical capacity $C = 0.8$ bits/use
- Pre-shared entanglement available

a) What is the maximum rate for sending classical information without entanglement?
b) How does pre-shared entanglement change this?
c) Can Alice and Bob use this channel for quantum key distribution? Explain.
d) What is the rate for transmitting quantum states through this channel?

---

## Problem Summary

| Section | Problems | Topic | Points |
|---------|----------|-------|--------|
| A | 1-8 | Von Neumann Entropy | 35 |
| B | 9-14 | Holevo Bound | 25 |
| C | 15-19 | Channel Capacity | 25 |
| D | 20-25 | Compression/Advanced | 15 |

**Total: 100 points**

---

## Submission Guidelines

For qualifying exam preparation:
1. Show all steps in entropy calculations
2. State theorems before applying them
3. Verify numerical answers when possible
4. Focus on Problems 2, 6, 11, 16, 20 as core exam topics

---

**Created:** February 9, 2026
**Estimated Time:** 15-20 hours
