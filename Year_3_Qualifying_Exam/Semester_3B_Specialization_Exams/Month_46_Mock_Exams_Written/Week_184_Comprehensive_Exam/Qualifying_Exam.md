# Comprehensive Written Qualifying Exam

## Quantum Science and Engineering PhD Program

---

## Exam Information

**Duration:** 4 hours (240 minutes)
**Total Points:** 200
**Number of Problems:** 10
**Passing Score:** 160 points (80%)

---

## Instructions

1. **Time Management:** You have 240 minutes for 10 problems. Average 24 minutes per problem, but allocate based on difficulty.

2. **Problem Selection:** All problems are required. There is no choice.

3. **Show All Work:** Partial credit is awarded for correct reasoning, even if the final answer is wrong.

4. **Notation:** Use standard notation. Define any non-standard symbols.

5. **Physical Reasoning:** Explain your approach, not just calculations.

6. **Integration Problems:** Problem 10 spans multiple topics. Budget extra time.

7. **No External Resources:** Closed-book, no notes, no electronic devices.

---

## Section A: Quantum Mechanics (60 points)

### Problem 1: Quantum Dynamics (20 points)

A particle of mass $m$ is confined to a one-dimensional harmonic oscillator potential $V(x) = \frac{1}{2}m\omega^2 x^2$. At $t = 0$, the particle is in the state:

$$|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |2\rangle)$$

where $|n\rangle$ are the energy eigenstates.

**(a)** (5 points) Find $|\psi(t)\rangle$ for all $t > 0$.

**(b)** (5 points) Calculate $\langle x(t) \rangle$ and $\langle x^2(t) \rangle$. Does $\langle x \rangle$ oscillate?

**(c)** (5 points) Calculate the uncertainty $\Delta x(t)$. Does the wave packet breathe (oscillate in width)?

**(d)** (5 points) This state is sometimes called a "Schrodinger cat state" for the harmonic oscillator. Explain what makes it cat-like, and discuss its decoherence properties when coupled to an environment.

---

### Problem 2: Angular Momentum and Perturbation Theory (20 points)

Consider a spin-1 particle in a magnetic field along the $z$-axis with Hamiltonian:

$$\hat{H}_0 = \omega_0 \hat{S}_z$$

A perturbation is applied:

$$\hat{V} = \lambda(\hat{S}_x^2 - \hat{S}_y^2)$$

**(a)** (5 points) Write $\hat{H}_0$ and $\hat{V}$ as explicit $3 \times 3$ matrices in the $|1,m\rangle$ basis.

**(b)** (5 points) Calculate the first-order energy corrections to all three energy levels.

**(c)** (5 points) For the $m = 0$ state, calculate the second-order energy correction.

**(d)** (5 points) This type of Hamiltonian appears in spin squeezing. Explain qualitatively how the perturbation $\hat{V}$ can be used to generate spin-squeezed states that beat the standard quantum limit.

---

### Problem 3: Scattering and Path Integrals (20 points)

**(a)** (7 points) A particle scatters from a delta function potential $V(x) = \alpha\delta(x)$ in one dimension. Using the transfer matrix method or matching conditions, find the transmission coefficient $T(E)$ for energy $E > 0$.

**(b)** (6 points) In the limit $\alpha \to \infty$ (infinite barrier), what happens to $T$? In the limit $\alpha \to 0$, what is $T$?

**(c)** (7 points) The path integral formulation of quantum mechanics expresses the propagator as:

$$K(x_f, t_f; x_i, t_i) = \int \mathcal{D}[x(t)] e^{iS[x]/\hbar}$$

For a free particle, derive the propagator using the path integral. You may use the result that the Gaussian integral $\int_{-\infty}^{\infty} dx \, e^{-ax^2} = \sqrt{\pi/a}$.

---

## Section B: Quantum Information and Computing (60 points)

### Problem 4: Entanglement and Quantum States (20 points)

Consider the three-qubit W state:

$$|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$$

**(a)** (5 points) Calculate the reduced density matrix $\rho_{AB}$ obtained by tracing out qubit C.

**(b)** (5 points) Calculate the entanglement of formation $E_F(\rho_{AB})$ for this reduced state. (You may use the concurrence formula.)

**(c)** (5 points) Compare the entanglement properties of the W state to the GHZ state $|GHZ\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$. Which is more robust to particle loss?

**(d)** (5 points) Design a quantum circuit using standard gates (H, CNOT, etc.) that prepares the W state from $|000\rangle$. Draw the circuit.

---

### Problem 5: Quantum Algorithms (20 points)

**(a)** (6 points) State the hidden subgroup problem (HSP) for a finite group $G$. Explain why Shor's algorithm and Simon's algorithm are both instances of the HSP.

**(b)** (7 points) Quantum phase estimation (QPE) is a key subroutine. Given a unitary $U$ and an eigenstate $|u\rangle$ with $U|u\rangle = e^{2\pi i\phi}|u\rangle$:

i. Draw the QPE circuit for $n$ bits of precision.
ii. If $\phi = 0.375 = 3/8$, what is the measurement outcome with 3 ancilla qubits?

**(c)** (7 points) The Harrow-Hassidim-Lloyd (HHL) algorithm solves linear systems $A\vec{x} = \vec{b}$ exponentially faster than classical methods under certain conditions.

i. What conditions on $A$ are required?
ii. What is the output of the algorithm (the quantum state, not the full classical vector)?
iii. Why doesn't this give an exponential speedup for all linear systems?

---

### Problem 6: Quantum Channels and Communication (20 points)

**(a)** (6 points) The quantum capacity $Q(\mathcal{E})$ of a channel $\mathcal{E}$ quantifies the rate of reliable quantum information transmission. State the formula for quantum capacity in terms of coherent information.

**(b)** (7 points) Consider the amplitude damping channel with damping probability $\gamma$:

$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

Show that this channel has zero quantum capacity when $\gamma \geq 1/2$. (Hint: consider the anti-degradable property.)

**(c)** (7 points) Entanglement-assisted classical capacity $C_E$ allows pre-shared entanglement. For a depolarizing channel with error probability $p$:

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

Explain why $C_E > C$ (the unassisted classical capacity) for this channel. What is the maximum advantage gained from entanglement assistance?

---

## Section C: Quantum Error Correction (60 points)

### Problem 7: Stabilizer Codes (20 points)

Consider the $[[5, 1, 3]]$ perfect code with stabilizer generators:

$$S_1 = XZZXI, \quad S_2 = IXZZX, \quad S_3 = XIXZZ, \quad S_4 = ZXIXZ$$

**(a)** (5 points) Verify that these generators mutually commute. Find logical operators $\bar{X}$ and $\bar{Z}$.

**(b)** (5 points) Suppose a $Y_3$ error occurs (Pauli $Y$ on qubit 3). Calculate the syndrome.

**(c)** (5 points) Why is this code called "perfect"? What is its relationship to the classical Hamming bound?

**(d)** (5 points) The 5-qubit code is non-CSS. What is the consequence for transversal gates? Can any Clifford gate be implemented transversally?

---

### Problem 8: Surface Codes and Decoding (20 points)

**(a)** (6 points) For a distance-$d$ surface code, express the following in terms of $d$:
- Number of physical qubits $n$
- Number of $X$-stabilizers
- Number of $Z$-stabilizers
- Code rate $k/n$

**(b)** (7 points) The surface code threshold for depolarizing noise is approximately $p_{th} \approx 1\%$. If the physical error rate is $p = 10^{-3}$, what code distance is needed to achieve logical error rate $p_L < 10^{-12}$?

**(c)** (7 points) Describe how lattice surgery implements logical CNOT between two surface code patches. What is the time overhead compared to a transversal CNOT, and why is this acceptable?

---

### Problem 9: Fault-Tolerant Computation (20 points)

**(a)** (6 points) State the Eastin-Knill theorem. What is its implication for universal fault-tolerant quantum computation?

**(b)** (7 points) Magic state distillation is the standard solution to Eastin-Knill. For the 15-to-1 distillation protocol:
- Input: 15 noisy $|T\rangle$ states with error rate $\epsilon$
- Output: 1 cleaner $|T\rangle$ state

What is the output error rate in terms of $\epsilon$? How many levels of distillation are needed to reduce error from $\epsilon = 10^{-1}$ to $\epsilon_{out} < 10^{-12}$?

**(c)** (7 points) An alternative to magic state distillation is using codes with transversal non-Clifford gates. The $[[15, 1, 3]]$ Reed-Muller code has a transversal $T$ gate. Why isn't this code used in practice? What trade-offs does it involve?

---

## Section D: Integration (20 points)

### Problem 10: Cross-Cutting Problem (20 points)

This problem integrates concepts from quantum mechanics, information theory, and error correction.

**Context:** You are designing a quantum processor for simulating molecular ground states using the Variational Quantum Eigensolver (VQE).

**(a)** (5 points) **QM Foundation:** The molecular Hamiltonian can be mapped to a qubit Hamiltonian:
$$H = \sum_i h_i P_i$$
where $P_i$ are Pauli strings. Explain how the variational principle guarantees that $\langle\psi(\theta)|H|\psi(\theta)\rangle \geq E_0$ for any parameterized ansatz $|\psi(\theta)\rangle$.

**(b)** (5 points) **QI/QC Implementation:** The ansatz is typically a hardware-efficient circuit. If the ansatz has $L$ layers of single-qubit rotations followed by entangling gates on $n$ qubits:
- How many parameters $\theta$ are there?
- What is the circuit depth?
- Why might such ansatze suffer from "barren plateaus"?

**(c)** (5 points) **Error Mitigation:** Before full QEC is available, error mitigation techniques are used. Explain zero-noise extrapolation (ZNE) and probabilistic error cancellation (PEC). What are their limitations?

**(d)** (5 points) **Research Frontier:** Looking forward, how might fault-tolerant quantum computing change the approach to quantum chemistry simulation? Discuss:
- The role of quantum phase estimation
- Expected resource requirements (number of logical qubits, T-gates)
- The crossover point where quantum advantage becomes practical

---

## End of Exam

**Before submitting, verify:**
- [ ] All 10 problems attempted
- [ ] Work shown for each part
- [ ] Physical reasoning explained
- [ ] Time check: Did you pace yourself?

---

**Summary:**

| Section | Problems | Points |
|---------|----------|--------|
| A: Quantum Mechanics | 1-3 | 60 |
| B: Quantum Info/Computing | 4-6 | 60 |
| C: Quantum Error Correction | 7-9 | 60 |
| D: Integration | 10 | 20 |
| **Total** | **10** | **200** |

---

*This comprehensive exam is modeled after PhD qualifying examinations in quantum science and engineering programs.*

**Good luck!**
