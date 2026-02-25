# Final Mock Qualifying Examination - Written Component

## Examination Information

**Duration:** 4 hours (240 minutes)
**Total Points:** 100
**Passing Score:** 80 points overall, 70% minimum in each section
**Materials Allowed:** One page formula sheet (both sides), pen/pencil, calculator

---

## Instructions

1. Read all problems before beginning
2. Allocate time based on point values
3. Show all work for full credit
4. State assumptions clearly
5. Box final answers
6. Partial credit is awarded for correct approaches

**Suggested Time Allocation:**
- Reading all problems: 10 minutes
- QM Section (35 points): 85 minutes
- QI/QC Section (35 points): 85 minutes
- QEC Section (30 points): 45 minutes
- Review: 15 minutes

---

## Section A: Quantum Mechanics (35 points)

### Problem 1: Perturbation Theory (12 points)

A particle of mass $m$ is confined to a one-dimensional harmonic oscillator potential $V_0(x) = \frac{1}{2}m\omega^2 x^2$. A perturbation is applied:

$$H' = \lambda x^3$$

where $\lambda$ is a small parameter.

**(a)** [3 points] Calculate the first-order energy correction $E_n^{(1)}$ for the $n$-th energy level. Explain why your answer makes physical sense.

**(b)** [4 points] Calculate the second-order energy correction $E_0^{(2)}$ for the ground state. Express your answer in terms of $\hbar$, $m$, $\omega$, and $\lambda$.

**(c)** [3 points] Find the first-order correction to the ground state wave function $|\psi_0^{(1)}\rangle$ in terms of unperturbed states.

**(d)** [2 points] For what range of $\lambda$ is this perturbation expansion valid? Give your answer in terms of other parameters.

**Useful relations:**
$$x = \sqrt{\frac{\hbar}{2m\omega}}(a + a^\dagger)$$
$$a|n\rangle = \sqrt{n}|n-1\rangle, \quad a^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$$

---

### Problem 2: Angular Momentum (12 points)

Consider an electron with orbital angular momentum $\ell = 1$ and spin $s = 1/2$.

**(a)** [2 points] List all possible values of the total angular momentum quantum number $j$ and the corresponding values of $m_j$.

**(b)** [4 points] The state $|j = 3/2, m_j = 1/2\rangle$ can be written as a superposition of uncoupled states $|\ell, m_\ell\rangle|s, m_s\rangle$. Find the explicit form of this state.

**(c)** [3 points] A measurement of $L_z$ is performed on the state $|j = 3/2, m_j = 1/2\rangle$. What are the possible outcomes and their probabilities?

**(d)** [3 points] The spin-orbit coupling Hamiltonian is $H_{SO} = \alpha \vec{L} \cdot \vec{S}$. Calculate the energy splitting between the $j = 3/2$ and $j = 1/2$ states in terms of $\alpha$.

**Useful formula:**
$$\vec{L} \cdot \vec{S} = \frac{1}{2}(J^2 - L^2 - S^2)$$

---

### Problem 3: Time-Dependent Quantum Mechanics (11 points)

A spin-1/2 particle is initially in the state $|\psi(0)\rangle = |+z\rangle$ (spin up along z-axis). At $t = 0$, a magnetic field $\vec{B} = B_0\hat{x}$ is turned on, creating a Hamiltonian:

$$H = -\gamma B_0 S_x = -\frac{\gamma B_0 \hbar}{2}\sigma_x$$

where $\gamma$ is the gyromagnetic ratio.

**(a)** [3 points] Find the time evolution operator $U(t) = e^{-iHt/\hbar}$ in matrix form.

**(b)** [3 points] Calculate the state $|\psi(t)\rangle$ at time $t$.

**(c)** [3 points] Find the probability of measuring spin up along the z-axis, $P_{+z}(t)$, as a function of time. At what time is this probability minimum?

**(d)** [2 points] Calculate the expectation value $\langle S_z \rangle(t)$ and verify it is consistent with your answer to part (c).

---

## Section B: Quantum Information and Computing (35 points)

### Problem 4: Density Matrices and Entanglement (12 points)

Consider the two-qubit state:

$$|\psi\rangle = \alpha|00\rangle + \beta|01\rangle + \gamma|10\rangle$$

where $|\alpha|^2 + |\beta|^2 + |\gamma|^2 = 1$ and $\alpha, \beta, \gamma$ are real and positive.

**(a)** [3 points] Write the density matrix $\rho = |\psi\rangle\langle\psi|$ in the computational basis.

**(b)** [3 points] Calculate the reduced density matrix $\rho_A = \text{Tr}_B(\rho)$ for qubit A.

**(c)** [3 points] Calculate the von Neumann entropy $S(\rho_A)$. Under what conditions on $\alpha, \beta, \gamma$ is the entanglement maximized?

**(d)** [3 points] Is the state $|\psi\rangle$ separable or entangled? Prove your answer by attempting to write it in product form or showing this is impossible.

---

### Problem 5: Quantum Channels (11 points)

The amplitude damping channel with parameter $\gamma$ has Kraus operators:

$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

**(a)** [2 points] Verify that $\sum_k K_k^\dagger K_k = I$, confirming this is a valid quantum channel.

**(b)** [3 points] A qubit is prepared in the state $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$. Calculate the output density matrix after passing through the amplitude damping channel.

**(c)** [3 points] Starting from the state $|1\rangle$, how many applications of the channel (with $\gamma = 0.1$) are needed for the population of $|0\rangle$ to exceed 0.95?

**(d)** [3 points] This channel is often used to model energy relaxation in a qubit. Explain physically why the Kraus operators have this form. What does $\gamma$ represent physically?

---

### Problem 6: Quantum Algorithms (12 points)

**(a)** [4 points] Consider Grover's algorithm searching for a single marked item among $N = 64$ items.

(i) How many iterations are optimal?
(ii) What is the probability of finding the marked item after the optimal number of iterations?
(iii) What happens to the success probability if you perform twice the optimal number of iterations?

**(b)** [4 points] In the quantum phase estimation algorithm with $t$ qubits for the phase register, the phase $\phi$ of eigenvalue $e^{2\pi i\phi}$ is estimated.

(i) If $\phi = 0.25$, what output will be measured with certainty when using $t = 3$ qubits?
(ii) If $\phi = 0.3$, what is the probability of measuring the output "010" (binary for 2) with $t = 3$ qubits?

**(c)** [4 points] Describe the key insight that makes Shor's algorithm efficient. Specifically:

(i) What classical problem is reduced to period finding?
(ii) Why is period finding efficient on a quantum computer?
(iii) What is the role of the quantum Fourier transform?

---

## Section C: Quantum Error Correction (30 points)

### Problem 7: Stabilizer Codes (10 points)

Consider the $[[5,1,3]]$ five-qubit code with stabilizer generators:

$$\begin{aligned}
g_1 &= XZZXI \\
g_2 &= IXZZX \\
g_3 &= XIXZZ \\
g_4 &= ZXIXZ
\end{aligned}$$

**(a)** [2 points] Verify that this code encodes exactly one logical qubit by checking the number of independent generators.

**(b)** [3 points] Find the syndrome for a $Z$ error on qubit 3. Show your work explicitly.

**(c)** [3 points] Find the logical $\bar{X}$ and $\bar{Z}$ operators. Verify that they anticommute and commute with all stabilizers.

**(d)** [2 points] Explain why this code can correct any single-qubit error. How many distinct syndromes are needed?

---

### Problem 8: Fault Tolerance and Surface Code (10 points)

**(a)** [3 points] Define what it means for an error correction procedure to be "fault-tolerant." Why is this property essential for practical quantum computing?

**(b)** [3 points] The threshold theorem states that arbitrarily accurate quantum computation is possible if the physical error rate $p$ is below a threshold $p_{th}$. For a concatenated code with logical error rate $p_L^{(1)} = cp^2$ after one level:

(i) What is the threshold value $p_{th}$?
(ii) How many levels of concatenation are needed to achieve logical error rate $p_L < 10^{-10}$ starting from $p = p_{th}/10$ with $c = 10$?

**(c)** [4 points] The surface code on an $L \times L$ lattice:

(i) How many physical qubits are required?
(ii) What is the code distance $d$?
(iii) If the physical error rate is $p = 0.1\%$ and the threshold is $p_{th} = 1\%$, estimate the logical error rate using $p_L \approx 0.03(p/p_{th})^{(d+1)/2}$ for $L = 5$.
(iv) Why is the surface code considered more practical than concatenated codes for near-term quantum computing?

---

### Problem 9: Advanced Error Correction (10 points)

**(a)** [4 points] The Steane code is a CSS code based on the classical $[7,4,3]$ Hamming code.

(i) What are the parameters $[[n,k,d]]$ of the Steane code?
(ii) Explain why the Hadamard gate is transversal for CSS codes.
(iii) Why can't all gates be transversal for any single stabilizer code?

**(b)** [3 points] Magic state distillation is used to implement non-Clifford gates fault-tolerantly.

(i) What is a magic state for the $T$ gate?
(ii) Why is magic state distillation necessary?
(iii) If a distillation protocol converts 15 noisy magic states into 1 purer state, reducing error from $\epsilon$ to $35\epsilon^3$, how many levels of distillation are needed to reduce error from $\epsilon = 0.01$ to below $10^{-10}$?

**(c)** [3 points] Quantum LDPC codes have recently attracted attention as alternatives to the surface code.

(i) What does "LDPC" stand for and what does it mean for the stabilizer structure?
(ii) What is the potential advantage of good qLDPC codes over the surface code?
(iii) What is the main practical challenge for implementing qLDPC codes?

---

## End of Examination

**Checklist before submission:**
- [ ] All problems attempted
- [ ] Work shown for each part
- [ ] Final answers boxed
- [ ] Units included where appropriate
- [ ] Name on all pages

**Total time: 4 hours**
**Total points: 100**

---

## Formula Sheet Suggestions

The following formulas may be useful (you may include others on your formula sheet):

**Quantum Mechanics:**
- Ladder operators: $a|n\rangle = \sqrt{n}|n-1\rangle$, $a^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$
- Perturbation theory: $E_n^{(1)} = \langle n^{(0)}|H'|n^{(0)}\rangle$
- $E_n^{(2)} = \sum_{m \neq n} \frac{|\langle m^{(0)}|H'|n^{(0)}\rangle|^2}{E_n^{(0)} - E_m^{(0)}}$
- Clebsch-Gordan coefficients for $\ell=1, s=1/2$

**Quantum Information:**
- Von Neumann entropy: $S(\rho) = -\text{Tr}(\rho \log_2 \rho)$
- Grover iterations: $k_{opt} \approx \frac{\pi}{4}\sqrt{N}$
- Phase estimation error probability

**Quantum Error Correction:**
- Syndrome calculation
- Threshold formula: $p_L \approx (p/p_{th})^{(d+1)/2}$
- Surface code parameters
