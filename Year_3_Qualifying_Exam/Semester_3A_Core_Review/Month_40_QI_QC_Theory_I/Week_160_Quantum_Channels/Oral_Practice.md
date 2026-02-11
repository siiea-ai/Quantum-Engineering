# Week 160: Quantum Channels - Oral Examination Practice

## Conceptual Questions

### Q1: What is a quantum channel and why must it be CPTP?

**Model Answer:**

A quantum channel is the most general physically realizable transformation of quantum states. Mathematically, it's a linear map $$\mathcal{E}$$ that takes density matrices to density matrices.

**Two required properties:**

**1. Trace-Preserving (TP):**
$$\text{Tr}(\mathcal{E}(\rho)) = \text{Tr}(\rho) = 1$$

Physical meaning: Probability is conserved. The output is a valid density matrix.

**2. Completely Positive (CP):**
Not just positive, but $$\mathcal{E} \otimes \mathcal{I}_n$$ is positive for all $$n$$.

Physical meaning: If our system is entangled with another system we don't act on, the joint state must remain valid.

**Why complete positivity?**
The transpose map $$T(\rho) = \rho^T$$ is positive but not CP. If we could physically implement it, applying $$T \otimes \mathcal{I}$$ to a Bell state would produce a matrix with negative eigenvalues - unphysical!

**Examples of CPTP maps:**
- Unitary evolution: $$\mathcal{U}(\rho) = U\rho U^\dagger$$
- Measurement: $$\mathcal{M}(\rho) = \sum_k P_k \rho P_k$$
- Partial trace: $$\mathcal{E}(\rho_{AB}) = \text{Tr}_B(\rho_{AB})$$
- Noise channels: depolarizing, amplitude damping, etc.

---

### Q2: Explain the Kraus representation theorem.

**Model Answer:**

**Theorem:** A linear map $$\mathcal{E}$$ is CPTP if and only if it can be written as:
$$\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$$

where the Kraus operators satisfy the completeness relation:
$$\sum_k K_k^\dagger K_k = I$$

**Physical derivation (Stinespring dilation):**

1. Start with system in state $$\rho_S$$
2. Attach environment in pure state $$|0\rangle_E$$
3. Apply joint unitary $$U_{SE}$$
4. Trace out environment

$$\mathcal{E}(\rho) = \text{Tr}_E[U(\rho \otimes |0\rangle\langle 0|)U^\dagger]$$

The Kraus operators are: $$K_k = \langle k|_E U |0\rangle_E$$

**Non-uniqueness:**
Different Kraus representations are related by unitary mixing:
$$K'_j = \sum_k U_{jk} K_k$$

Different representations correspond to different measurement schemes on the environment.

**Minimum number:**
At most $$d^2$$ Kraus operators needed for a $$d$$-dimensional system.

---

### Q3: Describe the depolarizing channel and its effect on qubits.

**Model Answer:**

**Definition:**
$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

With probability $$(1-p)$$: state unchanged
With probability $$p/3$$ each: X, Y, or Z error applied

**Kraus operators:**
$$K_0 = \sqrt{1-p}I, \quad K_i = \sqrt{p/3}\sigma_i$$ for $$i = 1,2,3$$

**Alternative form:**
$$\mathcal{E}_p(\rho) = (1-\frac{4p}{3})\rho + \frac{p}{3}I$$

**Effect on Bloch sphere:**
The Bloch vector shrinks uniformly: $$\vec{r} \to (1-\frac{4p}{3})\vec{r}$$

- $$p = 0$$: Identity channel
- $$p = 3/4$$: Completely depolarizing (output = $$I/2$$)
- $$p = 1$$: Uniform random Pauli

**Physical interpretation:**
Isotropic noise affecting all three Bloch components equally. Models decoherence in systems with no preferred direction.

**Importance:**
- Standard noise model in quantum error correction
- Threshold for fault-tolerant quantum computing often quoted for depolarizing noise

---

### Q4: Compare amplitude damping and phase damping channels.

**Model Answer:**

**Amplitude Damping (T1):**

Models energy relaxation - excited state decays to ground state.

Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

Effects:
- $$|1\rangle \to |0\rangle$$ with probability $$\gamma$$
- Off-diagonals decay as $$\sqrt{1-\gamma}$$
- Bloch sphere shrinks toward north pole

Physical examples:
- Spontaneous emission
- Relaxation of superconducting qubits

NOT unital: $$\mathcal{E}(I) \neq I$$

**Phase Damping (T2):**

Models loss of coherence without energy exchange.

Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$

Effects:
- Diagonal elements unchanged
- Off-diagonals decay as $$(1-\lambda)$$ or $$\sqrt{1-\lambda}$$
- Bloch sphere shrinks toward z-axis

Physical examples:
- Fluctuating magnetic fields
- Elastic scattering

IS unital: $$\mathcal{E}(I) = I$$

**Relationship:**
$$T_2 \leq 2T_1$$ because amplitude damping also causes dephasing.

---

### Q5: Explain the Choi-Jamiolkowski isomorphism.

**Model Answer:**

**Key idea:** There's a one-to-one correspondence between quantum channels and positive operators.

**Choi matrix construction:**
For channel $$\mathcal{E}$$:
$$J(\mathcal{E}) = (\mathcal{E} \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|)$$

where $$|\Phi^+\rangle = \frac{1}{\sqrt{d}}\sum_i |ii\rangle$$ is maximally entangled.

**Interpretation:** Apply the channel to half of a maximally entangled state. The resulting state encodes all properties of the channel.

**Key theorems:**

1. $$\mathcal{E}$$ is CP $$\Leftrightarrow$$ $$J(\mathcal{E}) \geq 0$$
2. $$\mathcal{E}$$ is TP $$\Leftrightarrow$$ $$\text{Tr}_B(J) = I/d$$

**In terms of Kraus operators:**
$$J = \frac{1}{d}\sum_k K_k \otimes K_k^*$$

**Recovering the channel:**
$$\mathcal{E}(\rho) = d \cdot \text{Tr}_1[(I \otimes \rho^T)J]$$

**Applications:**
- Verify CP/TP properties
- Characterize unknown channels experimentally
- Prove channel inequalities
- Gate teleportation

**Examples:**
- Identity: $$J = |\Phi^+\rangle\langle\Phi^+|$$
- Complete depolarizing: $$J = I/d^2$$

---

## Technical Questions

### Q6: Derive the Kraus operators for amplitude damping from a physical model.

**Model Answer:**

**Physical setup:**
- Two-level atom (system) initially in superposition
- Single-mode electromagnetic field (environment) initially in vacuum $$|0\rangle_E$$
- Jaynes-Cummings interaction

**Hamiltonian:**
$$H = \omega_a \sigma_+\sigma_- + \omega_f a^\dagger a + g(\sigma_+ a + \sigma_- a^\dagger)$$

Simplified: $$H_{int} = g(|e\rangle\langle g| \otimes a + |g\rangle\langle e| \otimes a^\dagger)$$

**Evolution:**
Starting from $$|e,0\rangle$$ (excited atom, vacuum field):
$$e^{-iHt}|e,0\rangle = \cos(gt)|e,0\rangle - i\sin(gt)|g,1\rangle$$

Starting from $$|g,0\rangle$$:
$$e^{-iHt}|g,0\rangle = |g,0\rangle$$

**Kraus operators:**
$$K_k = \langle k|_E U |0\rangle_E$$

$$K_0 = \langle 0|U|0\rangle = |g\rangle\langle g| + \cos(gt)|e\rangle\langle e|$$
$$K_1 = \langle 1|U|0\rangle = -i\sin(gt)|g\rangle\langle e|$$

With $$\gamma = \sin^2(gt)$$, this gives the standard form.

---

### Q7: Show that amplitude damping is not unital.

**Model Answer:**

A channel is unital if $$\mathcal{E}(I) = I$$.

For amplitude damping:
$$\mathcal{E}(I) = K_0 I K_0^\dagger + K_1 I K_1^\dagger$$

$$= \begin{pmatrix} 1 & 0 \\ 0 & 1-\gamma \end{pmatrix} + \begin{pmatrix} \gamma & 0 \\ 0 & 0 \end{pmatrix}$$

$$= \begin{pmatrix} 1+\gamma & 0 \\ 0 & 1-\gamma \end{pmatrix} \neq I$$

**Physical interpretation:**
Amplitude damping has a preferred fixed point ($$|0\rangle$$). Any state evolves toward this ground state. This asymmetry means the identity is not preserved.

**Contrast with phase damping:**
$$\mathcal{E}_{pd}(I) = K_0 K_0^\dagger + K_1 K_1^\dagger = I$$ ✓

Phase damping is unital because it doesn't change populations, only coherences.

---

## Calculation Questions

### Q8: Apply the depolarizing channel to $$|+\rangle\langle +|$$.

**Model Answer:**

$$|+\rangle\langle+| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

Using $$\mathcal{E}_p(\rho) = (1-\frac{4p}{3})\rho + \frac{p}{3}I$$:

$$\mathcal{E}_p(|+\rangle\langle+|) = (1-\frac{4p}{3})\frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} + \frac{p}{3}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 1-\frac{4p}{3}+\frac{2p}{3} & 1-\frac{4p}{3} \\ 1-\frac{4p}{3} & 1-\frac{4p}{3}+\frac{2p}{3} \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 1-\frac{2p}{3} & 1-\frac{4p}{3} \\ 1-\frac{4p}{3} & 1-\frac{2p}{3} \end{pmatrix}$$

At $$p = 3/4$$: output is $$I/2$$ (maximally mixed).

---

## Oral Exam Tips

1. **Know the physical interpretation**: Every channel models a physical process
2. **Remember key examples**: Depolarizing, amplitude damping, phase damping
3. **Connect to experiments**: T1 vs T2 times, error rates
4. **Understand the Choi isomorphism**: It's a powerful tool for verification
5. **Practice Kraus calculations**: Be able to apply channels to specific states

---

*Practice these questions until you can explain them fluently.*
