# Week 178: Problem Set - Neutral Atoms & Photonics

## Instructions

This problem set contains 28 problems covering neutral atom and photonic quantum computing platforms. Problems are organized by topic and difficulty level.

---

## Section A: Rydberg Atom Physics (Problems 1-7)

### Problem 1 [Level 1]
A Rubidium-87 atom is excited to the $$|70S_{1/2}\rangle$$ Rydberg state.

(a) Using the scaling law $$\tau \propto n^3$$, estimate the radiative lifetime if $$\tau(n=10) = 1$$ μs.

(b) Calculate the orbital radius using $$\langle r \rangle = n^2 a_0$$.

(c) How many ground-state atomic diameters (~0.5 nm) would fit inside this Rydberg orbit?

---

### Problem 2 [Level 1]
The van der Waals coefficient for Rb $$|50S\rangle$$ is $$C_6/h = 100$$ GHz·μm$$^6$$.

(a) Calculate $$C_6$$ in SI units (J·m$$^6$$).

(b) What is the interaction energy (in MHz) between two atoms separated by 5 μm?

(c) Using the scaling $$C_6 \propto n^{11}$$, estimate $$C_6$$ for the $$|70S\rangle$$ state.

---

### Problem 3 [Level 2]
Calculate the Rydberg blockade radius for the following parameters:
- $$C_6/h = 500$$ GHz·μm$$^6$$
- Rabi frequency $$\Omega/2\pi = 2$$ MHz

(a) Find $$R_b$$.

(b) How many atoms can fit within a blockade sphere if the array spacing is 3 μm?

(c) What happens to $$R_b$$ if we increase the laser power by a factor of 4?

---

### Problem 4 [Level 2]
Consider the Rydberg blockade gate between two atoms at separation $$R = 5$$ μm.

(a) For the blockade to be effective, we need $$V(R) > \hbar\Omega$$. If $$\Omega/2\pi = 1$$ MHz, what is the minimum $$C_6$$ required?

(b) The gate fidelity is limited by $$F \approx 1 - (\hbar\Omega/V)^2$$. Calculate the fidelity for $$C_6/h = 200$$ GHz·μm$$^6$$.

(c) What is the optimal choice of $$\Omega$$ to maximize gate speed while maintaining $$F > 0.99$$?

---

### Problem 5 [Level 2]
The Rydberg state $$|nS\rangle$$ can decay to $$|(n-1)P\rangle$$ via spontaneous emission.

(a) If the lifetime of $$|50S\rangle$$ is 100 μs and a typical gate takes 1 μs, what is the probability of spontaneous decay during one gate?

(b) For a circuit with 100 two-qubit gates, estimate the total decay error.

(c) How does this compare to typical gate errors from other sources (~1%)?

---

### Problem 6 [Level 3]
Derive the blockade shift for two atoms in Rydberg states.

(a) Write the Hamiltonian for two atoms with ground state $$|g\rangle$$ and Rydberg state $$|r\rangle$$, including laser coupling $$\Omega$$ and interaction $$V$$.

(b) Find the eigenstates and eigenenergies in the $$\{|gg\rangle, |gr\rangle, |rg\rangle, |rr\rangle\}$$ basis.

(c) Show that for $$V \gg \hbar\Omega$$, the state $$|rr\rangle$$ is shifted off-resonance by $$\sim V$$.

---

### Problem 7 [Level 3]
The three-atom Rydberg blockade enables a native CCZ (Toffoli) gate.

(a) Describe a pulse sequence to implement CCZ using Rydberg blockade.

(b) What constraints on the atom geometry are required for the blockade to work for all three pairs?

(c) Compare the resource requirements (gate count, circuit depth) for implementing Toffoli using CCZ vs decomposition into two-qubit gates.

---

## Section B: Neutral Atom Arrays (Problems 8-13)

### Problem 8 [Level 1]
An optical tweezer uses a focused laser at wavelength 852 nm with beam waist $$w_0 = 0.7$$ μm.

(a) Calculate the Rayleigh range $$z_R = \pi w_0^2/\lambda$$.

(b) If the trap depth is 1 mK and the atom temperature is 20 μK, what is the ratio $$U_0/k_B T$$?

(c) Estimate the probability that an atom escapes the trap in 1 second if the escape rate is Arrhenius-like: $$\Gamma_{esc} \propto e^{-U_0/k_BT}$$.

---

### Problem 9 [Level 1]
A neutral atom array has 100 tweezer sites with 50% loading probability per site.

(a) What is the expected number of loaded atoms?

(b) What is the probability of loading exactly 100 atoms (a defect-free array) without rearrangement?

(c) After atom sorting with 99% rearrangement fidelity per move, what is the expected final filling fraction?

---

### Problem 10 [Level 2]
The ground-state qubit uses hyperfine states $$|F=1, m_F=0\rangle$$ and $$|F=2, m_F=0\rangle$$ of Rb-87.

(a) The hyperfine splitting is 6.835 GHz. What wavelength photon would drive this transition directly?

(b) Raman transitions use two lasers detuned by $$\Delta$$ from an excited state. If $$\Delta = 100$$ GHz and each beam has Rabi frequency $$\Omega_1 = \Omega_2 = 100$$ MHz, what is the effective two-photon Rabi frequency?

(c) What is the main source of decoherence for these clock-state qubits?

---

### Problem 11 [Level 2]
Mid-circuit measurement in neutral atom arrays requires selectively reading out some qubits without disturbing others.

(a) Describe a protocol for mid-circuit measurement using atom loss detection.

(b) What is the main challenge for non-destructive readout?

(c) How do recent experiments (Harvard/QuEra 2024) achieve mid-circuit measurement?

---

### Problem 12 [Level 3]
Calculate the two-qubit gate fidelity for a Rydberg CZ gate.

(a) List the main error sources: spontaneous emission, blockade imperfection, motional effects.

(b) If $$\tau_{Ryd} = 100$$ μs, gate time = 500 ns, and there are 3 Rydberg pulses, estimate the spontaneous emission error.

(c) For $$V/\hbar\Omega = 20$$, estimate the blockade error using $$\epsilon_b \approx (\hbar\Omega/V)^2$$.

(d) Add errors in quadrature. What total gate fidelity do you predict?

---

### Problem 13 [Level 3]
Design a 4-qubit algorithm implementation on a neutral atom array.

(a) For the array geometry below, identify which qubit pairs can perform direct two-qubit gates (within $$R_b = 8$$ μm):
```
    q0 (0,0)
    q1 (5,0)
    q2 (0,5)
    q3 (5,5)
```
Distances in μm.

(b) To implement a gate between q0 and q3 (diagonal), what options are available?

(c) Compare to a nearest-neighbor superconducting architecture. Which requires fewer operations for a q0-q3 CNOT?

---

## Section C: Bosonic Codes (Problems 14-19)

### Problem 14 [Level 1]
The GKP code encodes a qubit in grid states of a harmonic oscillator.

(a) What is the spacing between peaks in the position-space wavefunction for $$|0_L\rangle$$?

(b) What displacement error (in units of $$\sqrt{\pi}$$) can be corrected?

(c) If position is measured with precision $$\sigma_q = 0.2\sqrt{\pi}$$, can the syndrome be reliably determined?

---

### Problem 15 [Level 1]
A cat qubit uses coherent states $$|\alpha\rangle$$ and $$|-\alpha\rangle$$.

(a) Calculate the overlap $$\langle\alpha|-\alpha\rangle$$ for $$|\alpha|^2 = 4$$.

(b) For the encoding $$|0_L\rangle = (|\alpha\rangle + |-\alpha\rangle)/\mathcal{N}$$, find the normalization $$\mathcal{N}$$.

(c) Why does small overlap lead to bit-flip suppression?

---

### Problem 16 [Level 2]
The bit-flip rate in a cat qubit scales as $$\Gamma_X \propto e^{-2|\alpha|^2}$$.

(a) If $$\Gamma_X/\Gamma_Z = 10^{-4}$$ for $$|\alpha|^2 = 4$$, what is the ratio for $$|\alpha|^2 = 8$$?

(b) What practical limit prevents us from making $$|\alpha|^2$$ arbitrarily large?

(c) For the repetition code with $$n$$ cat qubits, how does the logical bit-flip rate scale with $$n$$?

---

### Problem 17 [Level 2]
GKP error correction uses syndrome measurements.

(a) Explain how measuring $$\hat{q} \mod \sqrt{\pi}$$ reveals position displacement errors.

(b) What ancilla system and interaction are needed for syndrome extraction?

(c) After measuring syndrome $$s_q$$, what displacement should be applied to correct the error?

---

### Problem 18 [Level 3]
Derive the error correction properties of the GKP code.

(a) Show that the logical Pauli operators are $$\hat{X}_L = e^{i\sqrt{\pi}\hat{p}}$$ and $$\hat{Z}_L = e^{-i\sqrt{\pi}\hat{q}}$$.

(b) Verify that $$\hat{X}_L|0_L\rangle = |1_L\rangle$$.

(c) For a displacement error $$\hat{D}(\alpha)$$ with $$|\alpha| < \sqrt{\pi}/4$$, show that the syndrome measurement reveals $$\alpha$$ (modulo ambiguity).

---

### Problem 19 [Level 3]
Compare the resource overhead of bosonic codes vs surface codes.

(a) A GKP qubit requires one oscillator mode. What is the equivalent "physical qubit" count for a surface code logical qubit at code distance $$d=5$$?

(b) GKP requires ~10 dB squeezing for break-even. What physical resources (e.g., pump power, cavity quality) are needed?

(c) Discuss the trade-offs: which approach is more practical for near-term devices?

---

## Section D: Photonic Quantum Computing (Problems 20-25)

### Problem 20 [Level 1]
A photon qubit uses polarization encoding: $$|0\rangle = |H\rangle$$, $$|1\rangle = |V\rangle$$.

(a) What optical element implements the Hadamard gate?

(b) What implements the Pauli-Z gate?

(c) A beam splitter with transmittance $$t$$ and reflectance $$r = \sqrt{1-t^2}$$ mixes two modes. For $$t = 1/\sqrt{2}$$, what is the output state for input $$|1,0\rangle$$?

---

### Problem 21 [Level 1]
Single-photon sources are characterized by their $$g^{(2)}(0)$$ autocorrelation function.

(a) What is $$g^{(2)}(0)$$ for an ideal single-photon source?

(b) What is $$g^{(2)}(0)$$ for a coherent state (laser)?

(c) A source has $$g^{(2)}(0) = 0.05$$. What fraction of emissions are multi-photon?

---

### Problem 22 [Level 2]
The KLM (Knill-Laflamme-Milburn) CZ gate uses ancilla photons and measurement.

(a) The basic nonlinear sign (NS) gate has success probability $$p_{NS} = 1/4$$. How many ancilla photons are consumed per gate attempt?

(b) A CZ gate requires two NS gates. What is the success probability?

(c) Using teleportation-based boosting with $$n$$ Bell pairs, the success probability approaches $$1 - 1/(n+1)$$. For 99% success, how many Bell pairs are needed?

---

### Problem 23 [Level 2]
Measurement-based quantum computing uses cluster states.

(a) A 1D cluster state of 5 qubits can implement a quantum circuit. What is the maximum circuit depth achievable?

(b) Each qubit measurement consumes one cluster qubit. For a circuit with 100 qubits and depth 50, estimate the cluster state size.

(c) Why is measurement-based QC well-suited for photonics?

---

### Problem 24 [Level 3]
Gaussian boson sampling uses squeezed states as input.

(a) A squeezed vacuum state has photon statistics $$p(n) \propto |\tanh r|^n / (n! \cosh r)$$ for even $$n$$. For $$r = 1$$, calculate $$p(0)$$ and $$p(2)$$.

(b) Xanadu's Borealis used 216 squeezed modes. Estimate the Hilbert space dimension if each mode can have 0-10 photons.

(c) Explain why GBS output probabilities are hard to compute classically (involves permanents of matrices).

---

### Problem 25 [Level 3]
Design a photonic CNOT gate using linear optics and measurement.

(a) Sketch a circuit using dual-rail encoding and beam splitters.

(b) Identify the measurement pattern needed for heralding success.

(c) What feed-forward operation is required based on the measurement outcome?

---

## Section E: Platform Comparison (Problems 26-28)

### Problem 26 [Level 2]
Compare the implementation of a 50-qubit random circuit with depth 20 on:
- Neutral atom array (all pairs within blockade can interact)
- Photonic cluster state computer

(a) Estimate the number of physical operations for each platform.

(b) Which platform has higher expected output fidelity given current error rates?

(c) Discuss the time required for each approach.

---

### Problem 27 [Level 3]
For quantum simulation of a 2D Hubbard model on a 5×5 lattice:

(a) Why might neutral atoms have an advantage for this application?

(b) Compare the connectivity requirements to superconducting and trapped ion platforms.

(c) What native interactions in Rydberg atoms could be exploited?

---

### Problem 28 [Level 3]
Design a hybrid quantum network using photonic interconnects between neutral atom processors.

(a) What is the photon collection efficiency needed for entanglement between remote nodes?

(b) How does network latency affect the computation if error correction requires syndrome exchange?

(c) Estimate the total resource overhead compared to a monolithic system.

---

*Solutions are provided in Problem_Solutions.md*
