# Week 177: Problem Set - Superconducting & Trapped Ion Systems

## Instructions

This problem set contains 30 problems covering superconducting and trapped ion quantum computing platforms. Problems are organized by topic and difficulty level:
- **Level 1:** Direct application of concepts (10 problems)
- **Level 2:** Intermediate analysis (12 problems)
- **Level 3:** Challenging synthesis (8 problems)

Complete solutions are provided in `Problem_Solutions.md`.

---

## Section A: Josephson Junction Physics (Problems 1-5)

### Problem 1 [Level 1]
A Josephson junction has critical current $$I_c = 50$$ nA and is shunted by a capacitor with $$C = 80$$ fF.

(a) Calculate the Josephson energy $$E_J$$ in units of GHz (i.e., $$E_J/h$$).

(b) Calculate the charging energy $$E_C$$ in units of GHz.

(c) What is the ratio $$E_J/E_C$$? Is this in the transmon regime?

---

### Problem 2 [Level 1]
The AC Josephson relation states that a constant voltage $$V$$ across a junction causes the phase to evolve at rate $$d\phi/dt = 2eV/\hbar$$.

(a) If $$V = 100$$ μV, what is the oscillation frequency of the supercurrent?

(b) This frequency is used in voltage standards. Explain why.

---

### Problem 3 [Level 2]
Starting from the Josephson junction potential energy $$U(\phi) = -E_J\cos\phi$$:

(a) Expand the potential to fourth order around $$\phi = 0$$.

(b) Identify the harmonic frequency $$\omega_p$$ (plasma frequency) assuming the junction is shunted by capacitance $$C$$.

(c) Calculate the anharmonicity by treating the quartic term as a perturbation.

---

### Problem 4 [Level 2]
A SQUID (Superconducting Quantum Interference Device) consists of two Josephson junctions in parallel, forming a loop with total inductance $$L$$.

(a) Write the total current through the SQUID as a function of the phase differences $$\phi_1$$ and $$\phi_2$$.

(b) Using flux quantization $$\phi_1 - \phi_2 = 2\pi\Phi_{ext}/\Phi_0$$, show that the SQUID acts as a single junction with tunable critical current.

(c) For $$I_{c1} = I_{c2} = I_c$$, derive the expression for the effective critical current.

---

### Problem 5 [Level 3]
Consider a transmon qubit with $$E_J/h = 15$$ GHz and $$E_C/h = 250$$ MHz.

(a) Calculate the transition frequencies $$\omega_{01}$$ and $$\omega_{12}$$.

(b) A microwave drive at frequency $$\omega_d$$ is applied. At what detuning $$\Delta = \omega_d - \omega_{01}$$ will leakage to the $$|2\rangle$$ state become significant?

(c) The DRAG (Derivative Removal by Adiabatic Gate) pulse adds a quadrature component proportional to $$\dot{\Omega}(t)/\alpha$$. Explain how this suppresses leakage.

---

## Section B: Transmon Qubits & Circuit QED (Problems 6-12)

### Problem 6 [Level 1]
A transmon qubit is coupled to a coplanar waveguide resonator with frequency $$\omega_r/2\pi = 7.0$$ GHz. The transmon frequency is $$\omega_q/2\pi = 5.5$$ GHz and the coupling strength is $$g/2\pi = 100$$ MHz.

(a) Calculate the detuning $$\Delta = \omega_q - \omega_r$$.

(b) Is the system in the dispersive regime? Justify your answer.

(c) Calculate the dispersive shift $$\chi$$ assuming transmon anharmonicity $$\alpha/2\pi = -250$$ MHz.

---

### Problem 7 [Level 1]
In dispersive readout, the resonator frequency depends on the qubit state: $$\omega_r^{(0)} = \omega_r + \chi$$ and $$\omega_r^{(1)} = \omega_r - \chi$$.

(a) For $$\chi/2\pi = 1$$ MHz and resonator linewidth $$\kappa/2\pi = 2$$ MHz, what is the ratio $$\chi/\kappa$$?

(b) Calculate the phase difference between the reflected signals for $$|0\rangle$$ and $$|1\rangle$$ states.

(c) Why is $$\chi/\kappa \sim 1$$ typically optimal for readout?

---

### Problem 8 [Level 2]
Derive the Jaynes-Cummings Hamiltonian for a transmon coupled to a resonator:

$$\hat{H} = \hbar\omega_r\hat{a}^\dagger\hat{a} + \frac{\hbar\omega_q}{2}\hat{\sigma}_z + \hbar g(\hat{a}\hat{\sigma}^+ + \hat{a}^\dagger\hat{\sigma}^-)$$

(a) Transform to the interaction picture with respect to $$\hat{H}_0 = \hbar\omega_r\hat{a}^\dagger\hat{a} + \hbar\omega_r\hat{\sigma}_z/2$$.

(b) Apply second-order perturbation theory to derive the dispersive Hamiltonian.

(c) Identify the Lamb shift and AC Stark shift terms.

---

### Problem 9 [Level 2]
The cross-resonance (CR) gate is implemented by driving qubit 1 at the frequency of qubit 2.

(a) Starting from two coupled transmons with Hamiltonian:
$$\hat{H} = \omega_1\hat{b}_1^\dagger\hat{b}_1 + \omega_2\hat{b}_2^\dagger\hat{b}_2 + J(\hat{b}_1^\dagger\hat{b}_2 + \hat{b}_1\hat{b}_2^\dagger)$$
add a drive on qubit 1 at frequency $$\omega_2$$.

(b) Transform to a frame rotating at the drive frequency and identify the effective $$ZX$$ interaction.

(c) What is the CR gate time for achieving a CNOT-equivalent operation?

---

### Problem 10 [Level 2]
A tunable coupler connects two fixed-frequency transmons. The effective coupling is:

$$g_{eff} = g_1g_2\left(\frac{1}{\omega_1 - \omega_c} + \frac{1}{\omega_2 - \omega_c}\right)$$

where $$g_i$$ is the coupling of qubit $$i$$ to the coupler, and $$\omega_c$$ is the coupler frequency.

(a) At what coupler frequency is $$g_{eff} = 0$$?

(b) If $$\omega_1/2\pi = 5.0$$ GHz, $$\omega_2/2\pi = 5.3$$ GHz, and $$g_1 = g_2 = g/2\pi = 50$$ MHz, plot $$g_{eff}$$ as a function of $$\omega_c$$.

(c) Explain why this enables fast two-qubit gates with low idle ZZ error.

---

### Problem 11 [Level 3]
The iSWAP gate is implemented by tuning two transmons into resonance for time $$t$$.

(a) For two resonant qubits with exchange coupling $$J$$, derive the time evolution of the states $$|01\rangle$$ and $$|10\rangle$$.

(b) At what time $$t$$ is a perfect iSWAP achieved? What about $$\sqrt{\text{iSWAP}}$$?

(c) Frequency tuning causes flux noise sensitivity. If the flux noise has spectral density $$S_\Phi(f) = A^2/f$$, estimate the dephasing during the gate.

---

### Problem 12 [Level 3]
T1 relaxation in transmons can be caused by Purcell decay through the readout resonator.

(a) The Purcell limit on T1 is approximately $$T_1^{Purcell} = \Delta^2/(\kappa g^2)$$. For the parameters in Problem 6, calculate this limit.

(b) Modern transmons use Purcell filters. Explain the operating principle of a Purcell filter.

(c) What other mechanisms limit T1 in state-of-the-art transmons?

---

## Section C: Flux Qubits (Problems 13-16)

### Problem 13 [Level 1]
A flux qubit consists of a superconducting loop with inductance $$L = 100$$ pH interrupted by a Josephson junction with $$E_J/h = 500$$ GHz.

(a) Calculate the inductive energy scale $$E_L = \Phi_0^2/(4\pi^2 L)$$ in GHz.

(b) What is the ratio $$E_J/E_L$$?

(c) Sketch the potential $$U(\phi) = E_L(\phi - \phi_{ext})^2/2 - E_J\cos\phi$$ for $$\phi_{ext} = \pi$$.

---

### Problem 14 [Level 2]
In the fluxonium qubit, the junction is shunted by a superinductor (array of large Josephson junctions).

(a) The superinductor is modeled as $$L = N\Phi_0/(2\pi I_c^{array})$$ for $$N$$ junctions with critical current $$I_c^{array}$$. For $$N = 100$$ and $$I_c^{array} = 10$$ nA, calculate $$L$$.

(b) What are the advantages of fluxonium over the transmon?

(c) Fluxonium can operate at "sweet spots" where $$\partial\omega_{01}/\partial\Phi_{ext} = 0$$. Where are these located?

---

### Problem 15 [Level 2]
The three-junction flux qubit (persistent current qubit) has two degenerate states carrying currents $$\pm I_p$$ at half-flux bias.

(a) The qubit Hamiltonian near half-flux is $$\hat{H} = -\epsilon\hat{\sigma}_z/2 + \Delta\hat{\sigma}_x/2$$, where $$\epsilon \propto (\Phi_{ext} - \Phi_0/2)$$. Find the eigenenergies.

(b) At the degeneracy point ($$\epsilon = 0$$), what protects the qubit from flux noise?

(c) The tunnel splitting $$\Delta$$ depends exponentially on $$E_J/E_C$$. Explain why.

---

### Problem 16 [Level 3]
Consider a circuit with a Josephson junction, capacitor, and linear inductor (the RF-SQUID).

(a) Write the classical Lagrangian and derive the equations of motion.

(b) Quantize the circuit and find the spectrum numerically for $$E_J/E_L = 10$$ and $$E_C/E_L = 0.1$$.

(c) At what external flux values do level anticrossings occur?

---

## Section D: Trapped Ion Fundamentals (Problems 17-22)

### Problem 17 [Level 1]
A $$^{171}$$Yb$$^+$$ ion (mass $$m = 2.84 \times 10^{-25}$$ kg) is confined in a Paul trap with secular frequencies $$\omega_x/2\pi = \omega_y/2\pi = 3$$ MHz and $$\omega_z/2\pi = 1$$ MHz.

(a) Calculate the ground state wave function extent $$z_0 = \sqrt{\hbar/(2m\omega_z)}$$ along the axial direction.

(b) Calculate the Lamb-Dicke parameter $$\eta = kz_0$$ for a laser at wavelength $$\lambda = 369$$ nm.

(c) Is the ion in the Lamb-Dicke regime ($$\eta\sqrt{n+1} \ll 1$$) if $$\bar{n} = 0.1$$?

---

### Problem 18 [Level 1]
The $$^{171}$$Yb$$^+$$ hyperfine qubit uses states $$|F=0, m_F=0\rangle$$ and $$|F=1, m_F=0\rangle$$ with splitting 12.6 GHz.

(a) Why are these called "clock states"?

(b) What is the first-order Zeeman sensitivity $$\partial\omega/\partial B$$ for these states?

(c) Estimate T2 for a magnetic field noise of 1 μG at 60 Hz.

---

### Problem 19 [Level 2]
Two ions of mass $$m$$ in a linear Paul trap with axial frequency $$\omega_z$$ are separated by distance $$d$$.

(a) The equilibrium separation is determined by balancing trap force and Coulomb repulsion. Show that:
$$d = \left(\frac{e^2}{4\pi\epsilon_0 m\omega_z^2}\right)^{1/3}$$

(b) For $$\omega_z/2\pi = 1$$ MHz and $$^{171}$$Yb$$^+$$, calculate $$d$$.

(c) The two normal modes have frequencies $$\omega_z$$ (center-of-mass) and $$\sqrt{3}\omega_z$$ (stretch). Derive these from the coupled equations of motion.

---

### Problem 20 [Level 2]
Doppler cooling uses a laser red-detuned from an atomic transition.

(a) The steady-state temperature is $$T_D = \hbar\Gamma/(2k_B)$$ where $$\Gamma$$ is the transition linewidth. For $$\Gamma/2\pi = 20$$ MHz, calculate $$T_D$$.

(b) How many motional quanta $$\bar{n}$$ does this correspond to for $$\omega_z/2\pi = 1$$ MHz?

(c) Explain why sideband cooling is necessary to reach the ground state.

---

### Problem 21 [Level 2]
Sideband cooling uses the red sideband transition $$|g,n\rangle \rightarrow |e,n-1\rangle$$ followed by spontaneous emission $$|e,n-1\rangle \rightarrow |g,n-1\rangle$$.

(a) The red sideband Rabi frequency is $$\Omega_{n,n-1} = \eta\sqrt{n}\Omega_0$$. Why does cooling slow down as $$n \rightarrow 0$$?

(b) The steady-state mean phonon number is $$\bar{n}_{ss} \approx (\Gamma/2\omega_z)^2$$. Calculate this for the parameters in Problem 20.

(c) How long does it take to cool from $$\bar{n} = 10$$ to $$\bar{n} < 0.1$$?

---

### Problem 22 [Level 3]
For $$N$$ ions in a linear chain, the axial normal modes are found by diagonalizing the interaction matrix.

(a) Write the potential energy for $$N$$ ions including trap and Coulomb terms.

(b) Find the normal mode frequencies for $$N = 3$$ ions.

(c) The Mølmer-Sørensen gate couples to all modes. Explain why using a single mode (e.g., COM) simplifies the analysis but limits gate design.

---

## Section E: Mølmer-Sørensen Gate (Problems 23-27)

### Problem 23 [Level 1]
The MS gate uses a bichromatic laser field with frequencies $$\omega_0 \pm (\omega_m + \delta)$$ where $$\omega_0$$ is the qubit frequency, $$\omega_m$$ is a motional mode frequency, and $$\delta$$ is a small detuning.

(a) In the Lamb-Dicke regime, what transitions does each frequency component drive?

(b) Why is the carrier transition suppressed?

(c) For $$\delta/2\pi = 10$$ kHz and Rabi frequency $$\Omega/2\pi = 50$$ kHz, calculate the gate time $$\tau = 2\pi/\delta$$.

---

### Problem 24 [Level 2]
The MS gate creates a spin-dependent force that displaces the motional state in phase space.

(a) For a single ion, the displacement operator is $$\hat{D}(\alpha) = \exp(\alpha\hat{a}^\dagger - \alpha^*\hat{a})$$. Show that $$\langle\hat{x}\rangle$$ and $$\langle\hat{p}\rangle$$ are displaced.

(b) For two ions with $$\sigma_\phi = \pm 1$$, the displacements are $$\alpha_{\pm\pm}$$. Calculate the geometric phase acquired.

(c) Why does the motional state returning to its initial position (closed loop in phase space) ensure no residual ion-motion entanglement?

---

### Problem 25 [Level 2]
The MS gate Hamiltonian in the interaction picture is:

$$\hat{H}_{MS} = \hbar\Omega\sum_{j=1}^{N}\hat{\sigma}_\phi^{(j)}\left(\hat{a}e^{-i\delta t} + \hat{a}^\dagger e^{i\delta t}\right)$$

(a) For two ions, write out the 4×4 block structure of this Hamiltonian in the basis $$\{|++\rangle, |+-\rangle, |-+\rangle, |--\rangle\}\otimes|n\rangle$$.

(b) The collective spin operators are $$\hat{S}_\phi = (\hat{\sigma}_\phi^{(1)} + \hat{\sigma}_\phi^{(2)})/2$$. Show that the Hamiltonian can be written in terms of $$\hat{S}_\phi$$.

(c) Derive the effective spin-spin interaction after time $$\tau = 2\pi/\delta$$.

---

### Problem 26 [Level 3]
The MS gate is robust to thermal motion because the geometric phase is independent of initial phonon number.

(a) Starting from initial state $$|\psi_s\rangle\otimes|n\rangle$$, calculate the final state after the MS gate.

(b) Show that the spin-dependent phase $$\phi_{spin}$$ is independent of $$n$$.

(c) If the ion is in a thermal state $$\rho_{th} = \sum_n p_n|n\rangle\langle n|$$, what is the final spin state?

---

### Problem 27 [Level 3]
In practice, multiple motional modes must be considered for multi-ion MS gates.

(a) The mode coupling matrix $$b_{jm}$$ describes how ion $$j$$ couples to mode $$m$$. Write the generalized MS Hamiltonian.

(b) For the mode to close (no residual entanglement), we need $$\sum_m |A_m|^2 = 0$$ for certain amplitude functions. What constraint does this impose on gate design?

(c) Amplitude-modulated (AM) and frequency-modulated (FM) gates use pulse shaping to close all modes. Compare their advantages.

---

## Section F: Integration & Comparison (Problems 28-30)

### Problem 28 [Level 2]
Compare the implementation of a CNOT gate on superconducting and trapped ion platforms.

(a) On a superconducting device with nearest-neighbor connectivity, how many SWAP gates are needed for a CNOT between non-adjacent qubits?

(b) On a trapped ion device, what determines the two-qubit gate time?

(c) For a circuit requiring 50 CNOTs between random pairs, estimate the total circuit time on each platform.

---

### Problem 29 [Level 3]
You need to run a quantum algorithm with 100 qubits and circuit depth 1000.

(a) Using current error rates (two-qubit gate error $$10^{-3}$$ for superconducting, $$10^{-4}$$ for trapped ion), estimate the output fidelity on each platform.

(b) Which platform would you recommend, and why?

(c) What error correction overhead would be needed to achieve output fidelity > 0.5?

---

### Problem 30 [Level 3]
The choice of qubit platform depends on the specific application.

(a) For quantum chemistry simulations requiring all-to-all connectivity, compare superconducting and trapped ion approaches.

(b) For quantum machine learning with large qubit counts, which platform has advantages?

(c) For a hybrid quantum-classical optimization loop requiring fast iteration, which platform is preferred?

---

## Bonus Problem [Level 3+]

Design a 10-qubit quantum processor using your choice of platform.

(a) Specify the qubit layout, connectivity, and estimated gate fidelities.

(b) Calculate the expected circuit depth achievable before output fidelity drops below 0.1.

(c) Propose an error mitigation strategy appropriate for your platform.

(d) Estimate the classical compute resources needed for readout and control.

---

*Solutions are provided in Problem_Solutions.md*
