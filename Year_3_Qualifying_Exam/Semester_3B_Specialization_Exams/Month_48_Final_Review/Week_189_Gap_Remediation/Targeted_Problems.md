# Targeted Problems for Gap Remediation

## Overview

This document contains problems organized by common gap areas identified in qualifying examinations. Each section targets a specific weakness with problems of increasing difficulty.

---

## Section A: Quantum Mechanics Problems

### A.1 Perturbation Theory

#### Problem A.1.1: Non-Degenerate Second Order (Moderate)

A one-dimensional harmonic oscillator with Hamiltonian $$H_0 = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 x^2$$ is perturbed by $$H' = \lambda x^4$$.

(a) Calculate the first-order energy correction $$E_n^{(1)}$$ for the $$n$$-th energy level.

(b) Calculate the second-order energy correction $$E_0^{(2)}$$ for the ground state.

(c) For what values of $$\lambda$$ is perturbation theory valid?

**Hints:**
- Use $$x = \sqrt{\frac{\hbar}{2m\omega}}(a + a^\dagger)$$
- Selection rules: $$\langle m|x^4|n\rangle \neq 0$$ only for $$m = n, n \pm 2, n \pm 4$$

---

#### Problem A.1.2: Degenerate Perturbation (Challenging)

Consider the hydrogen atom in the $$n=2$$ level (4-fold degenerate ignoring spin). A uniform electric field $$\mathcal{E}$$ is applied along the z-axis, adding the perturbation $$H' = e\mathcal{E}z$$.

(a) Write down the unperturbed states in the $$|n, \ell, m\rangle$$ basis.

(b) Construct the perturbation matrix $$\langle n'\ell'm'|H'|n\ell m\rangle$$ in this subspace.

(c) Find the first-order energy corrections and the "good" linear combinations.

(d) Explain why only certain matrix elements are non-zero (selection rules).

**Required formulas:**
$$\langle 2,0,0|z|2,1,0\rangle = -3a_0$$
where $$a_0$$ is the Bohr radius.

---

#### Problem A.1.3: Time-Dependent Perturbation (Challenging)

A harmonic oscillator initially in its ground state $$|0\rangle$$ is subjected to a time-dependent perturbation:

$$H'(t) = \begin{cases} F_0 x e^{-t/\tau} & t > 0 \\ 0 & t \leq 0 \end{cases}$$

(a) Using first-order time-dependent perturbation theory, find the probability of transition to the first excited state $$|1\rangle$$ at time $$t$$.

(b) Find the transition probability as $$t \to \infty$$.

(c) In what limit does Fermi's golden rule apply?

---

### A.2 Angular Momentum

#### Problem A.2.1: Clebsch-Gordan Coefficients (Moderate)

Two spin-1/2 particles are in the state:

$$|\psi\rangle = \frac{1}{\sqrt{3}}|+\rangle_1|-\rangle_2 + \sqrt{\frac{2}{3}}|-\rangle_1|+\rangle_2$$

(a) Express this state in the coupled basis $$|s, m_s\rangle$$ using Clebsch-Gordan coefficients.

(b) What is the probability of measuring total spin $$s = 1$$?

(c) What is the probability of measuring total spin $$s = 0$$?

(d) Verify your answer by direct calculation of $$\langle S^2 \rangle$$.

---

#### Problem A.2.2: Addition of Angular Momenta (Challenging)

An electron in a hydrogen atom has orbital angular momentum $$\ell = 2$$ and spin $$s = 1/2$$.

(a) What are the possible values of total angular momentum $$j$$?

(b) Find the state $$|j = 5/2, m_j = 1/2\rangle$$ in terms of the uncoupled basis $$|\ell, m_\ell\rangle|s, m_s\rangle$$.

(c) An operator $$\hat{A}$$ acts only on the orbital part and is given by $$\hat{A} = L_z$$. Calculate $$\langle j = 5/2, m_j = 1/2|\hat{A}|j = 5/2, m_j = 1/2\rangle$$.

---

#### Problem A.2.3: Selection Rules (Moderate)

Derive the selection rules for electric dipole transitions:

(a) Show that $$\Delta m = 0, \pm 1$$ using $$[L_z, z] = 0$$ and $$[L_z, x \pm iy] = \pm\hbar(x \pm iy)$$.

(b) Show that $$\Delta \ell = \pm 1$$ using parity arguments.

(c) Apply these rules to determine which transitions are allowed from the state $$|n=3, \ell=2, m=1\rangle$$.

---

### A.3 Scattering Theory

#### Problem A.3.1: Born Approximation (Moderate)

Calculate the differential cross-section in the first Born approximation for scattering from a Yukawa potential:

$$V(r) = V_0 \frac{e^{-\mu r}}{r}$$

(a) Write down the Born approximation formula for $$f(\theta)$$.

(b) Evaluate the integral to find $$f(\theta)$$ in terms of $$q = |\vec{k}' - \vec{k}| = 2k\sin(\theta/2)$$.

(c) Find the total cross-section by integrating $$|f(\theta)|^2$$.

(d) Take the limit $$\mu \to 0$$ to recover the Rutherford cross-section.

---

#### Problem A.3.2: Partial Wave Analysis (Challenging)

For low-energy scattering ($$ka \ll 1$$) from a hard sphere of radius $$a$$:

(a) Write the general partial wave expansion for the scattering amplitude.

(b) Show that only the $$\ell = 0$$ partial wave contributes significantly at low energy.

(c) Calculate the s-wave phase shift $$\delta_0$$ by matching boundary conditions.

(d) Find the total cross-section and show it approaches $$4\pi a^2$$ at low energy.

---

### A.4 Identical Particles

#### Problem A.4.1: Two-Fermion System (Moderate)

Two identical spin-1/2 fermions are in a one-dimensional harmonic oscillator potential.

(a) Write the ground state wave function (spatial and spin parts).

(b) Write the first excited state wave functions. How many are there?

(c) A measurement of total spin $$S^2$$ is performed on the first excited state. What are the possible outcomes and their probabilities?

---

## Section B: Quantum Information Problems

### B.1 Entropy and Information

#### Problem B.1.1: Von Neumann Entropy (Moderate)

Consider the two-qubit state:

$$\rho = \frac{1}{4}(I \otimes I + \alpha \sigma_z \otimes \sigma_z)$$

where $$0 \leq \alpha \leq 1$$.

(a) Show that $$\rho$$ is a valid density matrix for all $$\alpha$$ in this range.

(b) Calculate the reduced density matrix $$\rho_A = \text{Tr}_B(\rho)$$.

(c) Calculate $$S(\rho_A)$$.

(d) Calculate $$S(\rho)$$.

(e) Verify that $$S(\rho) \leq S(\rho_A) + S(\rho_B)$$.

---

#### Problem B.1.2: Mutual Information (Challenging)

For the Werner state:

$$\rho_W = p|\Phi^+\rangle\langle\Phi^+| + \frac{1-p}{4}I \otimes I$$

where $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$.

(a) Find the eigenvalues of $$\rho_W$$.

(b) Calculate $$S(\rho_W)$$.

(c) Calculate the reduced density matrices and their entropies.

(d) Calculate the quantum mutual information $$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$$.

(e) For what values of $$p$$ is the state entangled?

---

### B.2 Quantum Channels

#### Problem B.2.1: Kraus Representation (Moderate)

The amplitude damping channel with parameter $$\gamma$$ has Kraus operators:

$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

(a) Verify that $$\sum_i K_i^\dagger K_i = I$$.

(b) Calculate the output state when the input is $$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$.

(c) Find the fixed point(s) of this channel.

(d) Describe the physical interpretation of this channel.

---

#### Problem B.2.2: Depolarizing Channel Capacity (Challenging)

The depolarizing channel with parameter $$p$$ is:

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

(a) Show this can be written as $$\mathcal{E}(\rho) = (1-\frac{4p}{3})\rho + \frac{4p}{3}\frac{I}{2}$$.

(b) Find the Kraus operators.

(c) The classical capacity is $$C = 1 - H(p)$$ where $$H$$ is the binary entropy. At what value of $$p$$ does the capacity become zero?

---

### B.3 Quantum Algorithms

#### Problem B.3.1: Phase Estimation Analysis (Moderate)

The quantum phase estimation algorithm estimates the eigenvalue $$e^{2\pi i\phi}$$ of a unitary $$U$$.

(a) If $$\phi = 0.375$$ and we use $$t = 3$$ qubits for the estimate, what is the exact output state after the inverse QFT?

(b) What is the probability of measuring the correct answer exactly?

(c) If $$\phi = 0.3$$ (which cannot be represented exactly with 3 bits), what are the probabilities of the two most likely outcomes?

---

#### Problem B.3.2: Grover's Algorithm (Challenging)

In Grover's algorithm searching a database of $$N = 2^n$$ items with $$M$$ marked items:

(a) Derive the optimal number of iterations $$k_{opt} = \lfloor \frac{\pi}{4}\sqrt{N/M} \rfloor$$.

(b) What is the success probability after $$k_{opt}$$ iterations?

(c) Show that if you iterate too many times, the probability of success decreases.

(d) How does the algorithm need to be modified if $$M$$ is unknown?

---

## Section C: Quantum Error Correction Problems

### C.1 Stabilizer Codes

#### Problem C.1.1: Steane Code Analysis (Moderate)

The Steane $$[[7,1,3]]$$ code has stabilizer generators:

$$\begin{aligned}
g_1 &= IIIXXXX \\
g_2 &= IXXIIXX \\
g_3 &= XIXIXIX \\
g_4 &= IIIZZZZ \\
g_5 &= IZZIIZZ \\
g_6 &= ZIZIZIZ
\end{aligned}$$

(a) Verify that all generators commute.

(b) Find the logical operators $$\bar{X}$$ and $$\bar{Z}$$.

(c) If a $$Y$$ error occurs on qubit 5, what is the syndrome?

(d) Explain why this code can correct any single-qubit error.

---

#### Problem C.1.2: Code Distance (Moderate)

For the $$[[5,1,3]]$$ code with stabilizer generators:

$$\begin{aligned}
g_1 &= XZZXI \\
g_2 &= IXZZX \\
g_3 &= XIXZZ \\
g_4 &= ZXIXZ
\end{aligned}$$

(a) Show that this code encodes 1 logical qubit.

(b) Find a weight-3 logical operator.

(c) Prove that no weight-1 or weight-2 operators can be logical operators (i.e., commute with all stabilizers without being in the stabilizer group).

---

### C.2 Fault Tolerance

#### Problem C.2.1: Threshold Calculation (Challenging)

Consider a concatenated code where the physical error rate is $$p$$ and the logical error rate after one level of concatenation is:

$$p_L^{(1)} = cp^2$$

where $$c$$ is a constant depending on the code.

(a) What is the logical error rate after $$k$$ levels of concatenation?

(b) Find the threshold $$p_{th}$$ such that $$p < p_{th}$$ implies the logical error rate decreases with concatenation level.

(c) If $$c = 100$$ and $$p = 0.001$$, how many levels of concatenation are needed to achieve $$p_L < 10^{-15}$$?

---

#### Problem C.2.2: Transversal Gates (Moderate)

(a) Explain why the Hadamard gate is transversal for the Steane code.

(b) Show that the $$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$ gate cannot be transversal for any stabilizer code that can detect arbitrary single-qubit errors.

(c) Describe how magic state distillation can be used to implement the $$T$$ gate fault-tolerantly.

---

### C.3 Surface Code

#### Problem C.3.1: Surface Code Basics (Moderate)

For a distance-3 surface code on a $$3 \times 3$$ lattice:

(a) Draw the lattice showing data qubits, X-stabilizers, and Z-stabilizers.

(b) How many physical qubits are needed?

(c) Write out one X-stabilizer and one Z-stabilizer explicitly.

(d) What are the logical $$\bar{X}$$ and $$\bar{Z}$$ operators?

(e) If the physical error rate is $$p = 0.1\%$$, estimate the logical error rate using $$p_L \approx 0.03(p/p_{th})^2$$ with $$p_{th} \approx 1\%$$.

---

#### Problem C.3.2: Syndrome Decoding (Challenging)

On a distance-5 surface code, the following X-syndrome pattern is observed (1 indicates a triggered stabilizer):

```
  0   1   0
0   0   0   1
  1   0   0
0   0   0   0
  0   0   0
```

(a) What is the minimum-weight error consistent with this syndrome?

(b) Is there a different minimum-weight error that would produce the same syndrome?

(c) How does the decoder choose between them?

(d) What logical error might result if the wrong correction is applied?

---

## Section D: Solutions Guide

### Solution Approach for A.1.1

(a) $$E_n^{(1)} = \lambda\langle n|x^4|n\rangle$$

Using $$x = \sqrt{\frac{\hbar}{2m\omega}}(a + a^\dagger)$$:

$$x^4 = \left(\frac{\hbar}{2m\omega}\right)^2(a + a^\dagger)^4$$

Expanding and keeping only terms with equal creation/annihilation:

$$\langle n|x^4|n\rangle = \left(\frac{\hbar}{2m\omega}\right)^2 \frac{3}{4}(2n^2 + 2n + 1)$$

So: $$\boxed{E_n^{(1)} = \frac{3\lambda\hbar^2}{4m^2\omega^2}(2n^2 + 2n + 1)}$$

(b) For ground state:

$$E_0^{(2)} = \sum_{m \neq 0} \frac{|\langle m|H'|0\rangle|^2}{E_0 - E_m}$$

Only $$m = 2, 4$$ contribute. After calculation:

$$\boxed{E_0^{(2)} = -\frac{21\lambda^2\hbar^3}{8m^4\omega^5}}$$

(c) Valid when $$|E_n^{(1)}| \ll E_n^{(0)}$$, giving $$\lambda \ll \frac{m^2\omega^3}{\hbar}$$.

---

*[Full solutions for all problems would be provided in a companion solutions document]*

---

## Problem Selection Guide

### For Critical Gaps (Level 5)
- Start with the "(Moderate)" problems
- Work through solution completely
- Repeat similar problems until comfortable

### For Severe Gaps (Level 4)
- Begin with "(Moderate)", progress to "(Challenging)"
- Focus on understanding, not speed

### For Moderate Gaps (Level 3)
- Go directly to "(Challenging)" problems
- Time yourself
- Practice exam conditions

### Recommended Daily Problem Count
- Gap Level 5: 3-4 problems with full solutions
- Gap Level 4: 4-5 problems
- Gap Level 3: 5-6 problems under time pressure
