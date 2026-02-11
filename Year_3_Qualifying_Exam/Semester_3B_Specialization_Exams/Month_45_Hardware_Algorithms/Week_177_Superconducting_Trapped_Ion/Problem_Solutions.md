# Week 177: Problem Solutions - Superconducting & Trapped Ion Systems

## Section A: Josephson Junction Physics

### Solution 1

**(a) Josephson Energy:**

$$E_J = \frac{I_c \Phi_0}{2\pi} = \frac{(50 \times 10^{-9})(2.07 \times 10^{-15})}{2\pi}$$

$$E_J = 1.65 \times 10^{-23} \text{ J}$$

Converting to frequency: $$E_J/h = 1.65 \times 10^{-23}/(6.626 \times 10^{-34}) = 24.9 \text{ GHz}$$

$$\boxed{E_J/h = 24.9 \text{ GHz}}$$

**(b) Charging Energy:**

$$E_C = \frac{e^2}{2C} = \frac{(1.6 \times 10^{-19})^2}{2(80 \times 10^{-15})}$$

$$E_C = 1.6 \times 10^{-25} \text{ J}$$

Converting: $$E_C/h = 1.6 \times 10^{-25}/(6.626 \times 10^{-34}) = 0.24 \text{ GHz} = 240 \text{ MHz}$$

$$\boxed{E_C/h = 240 \text{ MHz}}$$

**(c) Ratio:**

$$E_J/E_C = 24.9/0.24 = 104$$

$$\boxed{E_J/E_C = 104}$$

Yes, this is firmly in the transmon regime ($$E_J/E_C \gg 1$$, typically 50-100).

---

### Solution 2

**(a) Oscillation Frequency:**

From $$d\phi/dt = 2eV/\hbar$$:

$$f = \frac{1}{2\pi}\frac{d\phi}{dt} = \frac{2eV}{h} = \frac{2(1.6 \times 10^{-19})(100 \times 10^{-6})}{6.626 \times 10^{-34}}$$

$$\boxed{f = 48.4 \text{ GHz}}$$

**(b)** The frequency depends only on fundamental constants and the applied voltage through $$f = 2eV/h$$. This provides an exact relationship between voltage and frequency, enabling voltage standards traceable to the definition of the second (via frequency). The Josephson constant $$K_J = 2e/h$$ is exactly known.

---

### Solution 3

**(a) Taylor Expansion:**

$$-E_J\cos\phi \approx -E_J\left(1 - \frac{\phi^2}{2} + \frac{\phi^4}{24} - ...\right)$$

$$= -E_J + \frac{E_J}{2}\phi^2 - \frac{E_J}{24}\phi^4$$

**(b) Harmonic Frequency:**

The quadratic term gives potential energy $$U = E_J\phi^2/2$$. Combined with kinetic energy $$T = Q^2/(2C) = C\dot{\Phi}^2/2$$, and using $$\phi = 2\pi\Phi/\Phi_0$$:

$$\omega_p = \sqrt{\frac{8E_JE_C}{\hbar^2}} = \sqrt{8E_JE_C}/\hbar$$

$$\boxed{\omega_p = \sqrt{8E_JE_C}/\hbar}$$

**(c) Anharmonicity from Perturbation Theory:**

Treating $$V' = -E_J\phi^4/24$$ as a perturbation:

$$\langle n|V'|n\rangle = -\frac{E_J}{24}\langle n|\phi^4|n\rangle$$

Using $$\phi = \phi_{ZPF}(\hat{a} + \hat{a}^\dagger)$$ with $$\phi_{ZPF} = (2E_C/E_J)^{1/4}$$:

$$\langle n|\phi^4|n\rangle = \phi_{ZPF}^4(6n^2 + 6n + 3)$$

The anharmonicity (difference between adjacent level spacings):

$$\alpha = E_{12} - E_{01} \approx -E_C$$

$$\boxed{\alpha \approx -E_C}$$

---

### Solution 4

**(a) Total Current:**

$$I = I_{c1}\sin\phi_1 + I_{c2}\sin\phi_2$$

**(b) Flux Quantization:**

The phase difference around the loop must satisfy:

$$\phi_1 - \phi_2 = 2\pi\frac{\Phi_{ext}}{\Phi_0}$$

Defining $$\phi_{avg} = (\phi_1 + \phi_2)/2$$:

$$I = I_{c1}\sin\left(\phi_{avg} + \pi\frac{\Phi_{ext}}{\Phi_0}\right) + I_{c2}\sin\left(\phi_{avg} - \pi\frac{\Phi_{ext}}{\Phi_0}\right)$$

**(c) For $$I_{c1} = I_{c2} = I_c$$:**

$$I = 2I_c\cos\left(\pi\frac{\Phi_{ext}}{\Phi_0}\right)\sin\phi_{avg}$$

$$\boxed{I_{c,eff} = 2I_c\left|\cos\left(\pi\frac{\Phi_{ext}}{\Phi_0}\right)\right|}$$

---

### Solution 5

**(a) Transition Frequencies:**

$$\omega_{01} \approx \sqrt{8E_JE_C} - E_C = \sqrt{8(15)(0.25)} - 0.25 = \sqrt{30} - 0.25$$

$$\omega_{01}/2\pi = 5.48 - 0.25 = 5.23 \text{ GHz}$$

$$\omega_{12} = \omega_{01} + \alpha = 5.23 - 0.25 = 4.98 \text{ GHz}$$

$$\boxed{\omega_{01}/2\pi = 5.23 \text{ GHz}, \quad \omega_{12}/2\pi = 4.98 \text{ GHz}}$$

**(b) Leakage Condition:**

Leakage becomes significant when the drive can excite the $$|1\rangle \rightarrow |2\rangle$$ transition. This occurs when:

$$|\Delta| \lesssim |\alpha| = 250 \text{ MHz}$$

Driving near resonance ($$|\Delta| < 50$$ MHz) with strong pulses will cause leakage.

**(c) DRAG Pulse:**

The DRAG correction adds a quadrature component:

$$\Omega_y(t) = -\frac{\dot{\Omega}_x(t)}{\alpha}$$

This counteracts the off-resonant excitation of the $$|1\rangle \rightarrow |2\rangle$$ transition by providing a compensating amplitude that destructively interferes with the leakage pathway. The derivative term anticipates the leakage and pre-corrects for it.

---

## Section B: Transmon Qubits & Circuit QED

### Solution 6

**(a) Detuning:**

$$\Delta = \omega_q - \omega_r = 5.5 - 7.0 = -1.5 \text{ GHz}$$

$$\boxed{\Delta/2\pi = -1.5 \text{ GHz}}$$

**(b) Dispersive Regime Check:**

$$|g/\Delta| = 100/1500 = 0.067 \ll 1$$

Yes, the system is in the dispersive regime since $$g \ll |\Delta|$$.

**(c) Dispersive Shift:**

Including the effect of anharmonicity:

$$\chi = \frac{g^2}{\Delta}\frac{\alpha}{\Delta + \alpha} = \frac{(100)^2}{-1500}\frac{-250}{-1500 + (-250)}$$

$$\chi = \frac{10000}{-1500}\frac{-250}{-1750} = -6.67 \times 0.143 = -0.95 \text{ MHz}$$

$$\boxed{\chi/2\pi \approx -1 \text{ MHz}}$$

---

### Solution 7

**(a) Ratio:**

$$\chi/\kappa = 1/2 = 0.5$$

$$\boxed{\chi/\kappa = 0.5}$$

**(b) Phase Difference:**

The reflected signal phase is $$\phi = \arctan(2\Delta_r/\kappa)$$ where $$\Delta_r$$ is the detuning from resonance.

For $$|0\rangle$$: $$\phi_0 = \arctan(2\chi/\kappa) = \arctan(1) = 45°$$
For $$|1\rangle$$: $$\phi_1 = \arctan(-2\chi/\kappa) = \arctan(-1) = -45°$$

$$\boxed{\Delta\phi = 90°}$$

**(c)** When $$\chi/\kappa \sim 1$$:
- Large enough $$\chi$$ provides good state discrimination
- Not too large to cause excessive Purcell decay
- Resonator linewidth allows fast readout
- Optimal signal-to-noise ratio is achieved

---

### Solution 8

**(a) Interaction Picture:**

$$\hat{H}_I = \hbar g(\hat{a}\hat{\sigma}^+e^{-i\Delta t} + \hat{a}^\dagger\hat{\sigma}^-e^{i\Delta t})$$

where $$\Delta = \omega_q - \omega_r$$.

**(b) Second-Order Perturbation:**

Using the effective Hamiltonian formalism:

$$\hat{H}_{eff} = \frac{1}{\hbar\Delta}[\hat{V}, \hat{V}^\dagger]$$

where $$\hat{V} = \hbar g\hat{a}\hat{\sigma}^+$$.

$$[\hat{a}\hat{\sigma}^+, \hat{a}^\dagger\hat{\sigma}^-] = \hat{a}\hat{a}^\dagger\hat{\sigma}^+\hat{\sigma}^- - \hat{a}^\dagger\hat{a}\hat{\sigma}^-\hat{\sigma}^+$$

$$= (\hat{n} + 1)|e\rangle\langle e| - \hat{n}|g\rangle\langle g|$$

$$\hat{H}_{disp} = \hbar\chi\hat{a}^\dagger\hat{a}\hat{\sigma}_z + \text{const}$$

where $$\chi = g^2/\Delta$$.

**(c) Lamb and AC Stark Shifts:**

- **Lamb shift:** Frequency shift of qubit due to vacuum fluctuations: $$\delta\omega_q = g^2/\Delta$$
- **AC Stark shift:** Qubit frequency shift proportional to photon number: $$\delta\omega_q = 2\chi\langle\hat{n}\rangle$$

---

### Solution 9

**(a) Drive Hamiltonian:**

$$\hat{H} = \omega_1\hat{b}_1^\dagger\hat{b}_1 + \omega_2\hat{b}_2^\dagger\hat{b}_2 + J(\hat{b}_1^\dagger\hat{b}_2 + h.c.) + \Omega\cos(\omega_2 t)(\hat{b}_1 + \hat{b}_1^\dagger)$$

**(b) Rotating Frame Analysis:**

Transform to frame rotating at $$\omega_2$$. Qubit 1 is detuned by $$\Delta = \omega_1 - \omega_2$$.

The coupling $$J$$ hybridizes the qubits. Driving qubit 1 at $$\omega_2$$ creates transitions in qubit 2 that depend on the state of qubit 1.

Effective interaction:

$$\hat{H}_{eff} \propto \hat{Z}_1\hat{X}_2$$

**(c) Gate Time:**

For a $$ZX_{90}$$ gate (CNOT-equivalent):

$$t_{gate} = \frac{\pi}{2\Omega_{eff}}$$

where $$\Omega_{eff} \propto J\Omega/\Delta$$. Typical gate times: 200-500 ns.

---

### Solution 10

**(a) Zero Coupling Condition:**

$$g_{eff} = 0$$ when:

$$\frac{1}{\omega_1 - \omega_c} + \frac{1}{\omega_2 - \omega_c} = 0$$

$$\omega_2 - \omega_c = -(\omega_1 - \omega_c)$$

$$\boxed{\omega_c = \frac{\omega_1 + \omega_2}{2} = 5.15 \text{ GHz}}$$

**(b) Plot Description:**

$$g_{eff}$$ has a pole at $$\omega_c = \omega_1$$ and $$\omega_c = \omega_2$$, crosses zero at $$\omega_c = 5.15$$ GHz, and changes sign. Below 5.0 GHz and above 5.3 GHz, $$g_{eff}$$ has the same sign; between them, opposite sign.

**(c) Advantage:**

At the zero-coupling point, residual ZZ interaction ($$\propto g_{eff}^2$$) is minimized during idle, reducing crosstalk errors. For gates, the coupler is tuned away from this point to enable strong coupling.

---

### Solution 11

**(a) Time Evolution:**

For resonant qubits, the Hamiltonian in the $$\{|01\rangle, |10\rangle\}$$ subspace is:

$$\hat{H} = \hbar J\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

Evolution: $$|01\rangle \rightarrow \cos(Jt)|01\rangle + i\sin(Jt)|10\rangle$$

**(b) Gate Times:**

- iSWAP: $$Jt = \pi/2$$ → $$t = \pi/(2J)$$
- $$\sqrt{\text{iSWAP}}$$: $$Jt = \pi/4$$ → $$t = \pi/(4J)$$

$$\boxed{t_{iSWAP} = \frac{\pi}{2J}, \quad t_{\sqrt{iSWAP}} = \frac{\pi}{4J}}$$

**(c) Flux Noise Dephasing:**

During frequency tuning, the qubit acquires phase noise:

$$\delta\phi = \int_0^t \frac{\partial\omega}{\partial\Phi}\delta\Phi(t')dt'$$

For $$1/f$$ noise with $$S_\Phi(f) = A^2/f$$:

$$\langle\delta\phi^2\rangle \approx \left(\frac{\partial\omega}{\partial\Phi}\right)^2 A^2 t \ln(t/t_{min})$$

This leads to dephasing $$T_\phi \sim 1/(\partial\omega/\partial\Phi \cdot A)$$.

---

### Solution 12

**(a) Purcell Limit:**

$$T_1^{Purcell} = \frac{\Delta^2}{\kappa g^2} = \frac{(1500 \times 10^6)^2}{(2 \times 10^6)(100 \times 10^6)^2}$$

$$T_1^{Purcell} = \frac{2.25 \times 10^{18}}{2 \times 10^{16}} = 112.5 \text{ μs}$$

$$\boxed{T_1^{Purcell} \approx 110 \text{ μs}}$$

**(b) Purcell Filter:**

A Purcell filter is a bandpass filter placed between the qubit and readout line. It:
- Passes signals at the resonator frequency (for readout)
- Blocks signals at the qubit frequency (preventing decay)
- Typically implemented as a notch filter or multi-pole bandpass

**(c) Other T1 Mechanisms:**

- Dielectric loss in substrate and junction oxide
- Quasiparticle tunneling across the junction
- Radiation to spurious modes
- Two-level system (TLS) defects
- Surface losses at metal-substrate interfaces

---

## Section C: Flux Qubits

### Solution 13

**(a) Inductive Energy:**

$$E_L = \frac{\Phi_0^2}{4\pi^2 L} = \frac{(2.07 \times 10^{-15})^2}{4\pi^2 (100 \times 10^{-12})}$$

$$E_L = \frac{4.28 \times 10^{-30}}{3.95 \times 10^{-9}} = 1.08 \times 10^{-21} \text{ J}$$

$$E_L/h = 1.63 \text{ GHz}$$

$$\boxed{E_L/h = 1.63 \text{ GHz}}$$

**(b) Ratio:**

$$E_J/E_L = 500/1.63 = 307$$

$$\boxed{E_J/E_L = 307}$$

**(c) Potential Sketch:**

At $$\phi_{ext} = \pi$$, the potential $$U(\phi) = E_L(\phi - \pi)^2/2 - E_J\cos\phi$$ has two symmetric minima near $$\phi = \pi \pm \delta$$ where $$\delta \sim \sqrt{E_L/E_J}$$. The barrier height at $$\phi = \pi$$ is $$\sim E_J$$.

---

### Solution 14

**(a) Superinductance:**

$$L = \frac{N\Phi_0}{2\pi I_c^{array}} = \frac{100 \times 2.07 \times 10^{-15}}{2\pi \times 10 \times 10^{-9}}$$

$$L = \frac{2.07 \times 10^{-13}}{6.28 \times 10^{-8}} = 3.3 \text{ μH}$$

$$\boxed{L = 3.3 \text{ μH}}$$

**(b) Advantages over Transmon:**

- Large anharmonicity ($$\alpha \sim $$ GHz vs 200 MHz)
- T1 times exceeding 1 ms demonstrated
- Protected sweet spots with reduced flux noise sensitivity
- Lower frequency transitions reduce thermal population

**(c) Sweet Spots:**

- $$\Phi_{ext} = 0$$: Maximum barrier, symmetric potential
- $$\Phi_{ext} = \Phi_0/2$$: Degenerate double well, first-order flux insensitive
- Half-integer flux quanta generally provide sweet spots

---

### Solution 15

**(a) Eigenenergies:**

$$\hat{H} = -\frac{\epsilon}{2}\hat{\sigma}_z + \frac{\Delta}{2}\hat{\sigma}_x$$

Eigenvalues: $$E_\pm = \pm\frac{1}{2}\sqrt{\epsilon^2 + \Delta^2}$$

$$\boxed{E_\pm = \pm\frac{1}{2}\sqrt{\epsilon^2 + \Delta^2}}$$

**(b) Protection at Degeneracy:**

At $$\epsilon = 0$$:
- Energy gap is $$\Delta$$ (independent of small flux fluctuations)
- $$\partial E/\partial\epsilon|_{\epsilon=0} = 0$$
- First-order flux noise sensitivity vanishes
- Qubit is in superposition of current states: delocalized

**(c) Exponential Dependence:**

The tunnel splitting $$\Delta$$ requires tunneling through the potential barrier separating the two wells. The WKB tunneling amplitude:

$$\Delta \propto \exp\left(-\int\sqrt{2m(V-E)}dx/\hbar\right)$$

The barrier height scales with $$E_J$$, and the tunneling mass with $$E_C^{-1}$$, giving $$\Delta \propto e^{-\sqrt{E_J/E_C}}$$.

---

### Solution 16

**(a) Classical Lagrangian:**

$$\mathcal{L} = \frac{C\dot{\Phi}^2}{2} - \frac{(\Phi - \Phi_{ext})^2}{2L} + E_J\cos\left(\frac{2\pi\Phi}{\Phi_0}\right)$$

Equation of motion:

$$C\ddot{\Phi} + \frac{\Phi - \Phi_{ext}}{L} + \frac{2\pi E_J}{\Phi_0}\sin\left(\frac{2\pi\Phi}{\Phi_0}\right) = 0$$

**(b) Numerical Spectrum:**

Quantizing: $$[\hat{\Phi}, \hat{Q}] = i\hbar$$

Matrix elements computed in the charge or flux basis. For $$E_J/E_L = 10$$, $$E_C/E_L = 0.1$$:
- Ground state and first excited state form the qubit
- Anharmonicity depends on external flux
- Energy levels found by numerical diagonalization

**(c) Level Anticrossings:**

Anticrossings occur at half-integer flux values:
$$\Phi_{ext} = (n + 1/2)\Phi_0$$

At these points, the two circulating current states are degenerate without tunneling, and $$\Delta$$ lifts the degeneracy.

---

## Section D: Trapped Ion Fundamentals

### Solution 17

**(a) Ground State Extent:**

$$z_0 = \sqrt{\frac{\hbar}{2m\omega_z}} = \sqrt{\frac{1.055 \times 10^{-34}}{2(2.84 \times 10^{-25})(2\pi \times 10^6)}}$$

$$z_0 = \sqrt{\frac{1.055 \times 10^{-34}}{3.57 \times 10^{-18}}} = \sqrt{2.96 \times 10^{-17}} = 5.4 \text{ nm}$$

$$\boxed{z_0 = 5.4 \text{ nm}}$$

**(b) Lamb-Dicke Parameter:**

$$k = \frac{2\pi}{\lambda} = \frac{2\pi}{369 \times 10^{-9}} = 1.70 \times 10^7 \text{ m}^{-1}$$

$$\eta = kz_0 = (1.70 \times 10^7)(5.4 \times 10^{-9}) = 0.092$$

$$\boxed{\eta = 0.092}$$

**(c) Lamb-Dicke Regime Check:**

$$\eta\sqrt{\bar{n} + 1} = 0.092\sqrt{1.1} = 0.097 \ll 1$$

$$\boxed{\text{Yes, in Lamb-Dicke regime}}$$

---

### Solution 18

**(a) Clock States:**

These states have $$m_F = 0$$ for both levels, making them first-order insensitive to magnetic field fluctuations. The name "clock states" comes from their use in atomic clocks, where stability against environmental perturbations is essential.

**(b) First-Order Zeeman Sensitivity:**

For $$m_F = 0$$ states:

$$\frac{\partial\omega}{\partial B}\bigg|_{B=0} = 0$$

The first-order Zeeman shift vanishes. Second-order sensitivity exists but is small (~310 Hz/G² for Yb⁺).

**(c) T2 Estimate:**

With second-order Zeeman coefficient $$\beta \approx 310$$ Hz/G²:

At 1 μG noise: $$\delta\omega = \beta \cdot 2B_0\delta B \approx 310 \times 2 \times 10^{-6} \times 10^{-6}$$ Hz

This is negligible. T2 is typically limited by other factors (laser phase noise, motional heating) and can exceed seconds.

---

### Solution 19

**(a) Equilibrium Separation:**

Force balance: $$m\omega_z^2 d/2 = \frac{e^2}{4\pi\epsilon_0 d^2}$$

$$d^3 = \frac{e^2}{2\pi\epsilon_0 m\omega_z^2}$$

$$\boxed{d = \left(\frac{e^2}{2\pi\epsilon_0 m\omega_z^2}\right)^{1/3}}$$

**(b) Numerical Value:**

$$d = \left(\frac{(1.6 \times 10^{-19})^2}{2\pi(8.85 \times 10^{-12})(2.84 \times 10^{-25})(2\pi \times 10^6)^2}\right)^{1/3}$$

$$d = \left(\frac{2.56 \times 10^{-38}}{6.25 \times 10^{-23}}\right)^{1/3} = (4.1 \times 10^{-16})^{1/3} = 7.4 \text{ μm}$$

$$\boxed{d = 7.4 \text{ μm}}$$

**(c) Normal Mode Derivation:**

Equations of motion for displacements $$\delta z_1, \delta z_2$$ from equilibrium:

$$m\ddot{\delta z}_1 = -m\omega_z^2\delta z_1 + \frac{e^2}{2\pi\epsilon_0 d^3}(\delta z_2 - \delta z_1)$$

$$m\ddot{\delta z}_2 = -m\omega_z^2\delta z_2 + \frac{e^2}{2\pi\epsilon_0 d^3}(\delta z_1 - \delta z_2)$$

Define $$\omega_C^2 = e^2/(2\pi\epsilon_0 m d^3) = 2\omega_z^2$$.

Center-of-mass mode: $$z_{COM} = (z_1 + z_2)/2$$ → $$\omega_{COM} = \omega_z$$

Stretch mode: $$z_{str} = z_1 - z_2$$ → $$\omega_{str} = \sqrt{\omega_z^2 + 2\omega_C^2} = \sqrt{3}\omega_z$$

---

### Solution 20

**(a) Doppler Temperature:**

$$T_D = \frac{\hbar\Gamma}{2k_B} = \frac{(1.055 \times 10^{-34})(2\pi \times 20 \times 10^6)}{2(1.38 \times 10^{-23})}$$

$$T_D = \frac{1.33 \times 10^{-26}}{2.76 \times 10^{-23}} = 0.48 \text{ mK}$$

$$\boxed{T_D = 0.48 \text{ mK}}$$

**(b) Motional Quanta:**

$$\bar{n} = \frac{k_B T}{\hbar\omega_z} = \frac{(1.38 \times 10^{-23})(4.8 \times 10^{-4})}{(1.055 \times 10^{-34})(2\pi \times 10^6)}$$

$$\bar{n} = \frac{6.6 \times 10^{-27}}{6.6 \times 10^{-28}} = 10$$

$$\boxed{\bar{n} \approx 10}$$

**(c) Need for Sideband Cooling:**

Doppler cooling has a fundamental limit set by the recoil from spontaneous emission ($$T_D = \hbar\Gamma/2k_B$$). To reach the motional ground state ($$\bar{n} < 1$$), resolved sideband cooling uses narrow transitions to selectively remove motional quanta without the recoil heating limit.

---

### Solution 21

**(a) Cooling Slowdown:**

The red sideband Rabi frequency $$\Omega_{rsb} = \eta\sqrt{n}\Omega_0$$ vanishes as $$n \rightarrow 0$$. The cooling rate $$\propto \Omega_{rsb}^2 \propto n$$, so cooling becomes exponentially slower near the ground state.

**(b) Steady-State Phonon Number:**

$$\bar{n}_{ss} \approx \left(\frac{\Gamma}{2\omega_z}\right)^2 = \left(\frac{20 \times 10^6}{2 \times 10^6}\right)^2 = 100$$

This formula applies to broadband cooling. For resolved sideband cooling on a narrow transition:

$$\bar{n}_{ss} \approx \left(\frac{\gamma}{2\omega_z}\right)^2$$

where $$\gamma$$ is the narrow linewidth (~Hz to kHz), giving $$\bar{n}_{ss} \ll 0.01$$.

$$\boxed{\bar{n}_{ss} < 0.01 \text{ for resolved sideband cooling}}$$

**(c) Cooling Time:**

Cooling rate $$\Gamma_c \sim \eta^2\Omega^2/\gamma$$. For $$\eta = 0.1$$, $$\Omega/2\pi = 100$$ kHz, $$\gamma/2\pi = 10$$ Hz:

$$\Gamma_c \sim (0.01)(10^{10})/(60) \approx 10^6 \text{ s}^{-1}$$

Time to cool from $$\bar{n} = 10$$ to $$\bar{n} = 0.1$$: $$t \sim 10/\Gamma_c \sim 10$$ μs

$$\boxed{t_{cool} \sim 10-100 \text{ μs}}$$

---

### Solution 22

**(a) Potential Energy:**

$$V = \sum_{j=1}^{N}\frac{1}{2}m\omega_z^2 z_j^2 + \sum_{j<k}\frac{e^2}{4\pi\epsilon_0|z_j - z_k|}$$

**(b) N = 3 Normal Modes:**

For three ions at positions $$-d, 0, +d$$:

The normal mode frequencies are:
- COM: $$\omega_1 = \omega_z$$
- Tilt: $$\omega_2 = \sqrt{3}\omega_z$$
- Zig-zag: $$\omega_3 = \sqrt{29/5}\omega_z \approx 2.41\omega_z$$

$$\boxed{\omega_1 = \omega_z, \quad \omega_2 = \sqrt{3}\omega_z, \quad \omega_3 = \sqrt{29/5}\omega_z}$$

**(c) Single Mode Limitation:**

Using only the COM mode:
- All ions couple equally ($$b_{j,COM} = 1/\sqrt{N}$$)
- Simplified pulse design
- But: other modes may be excited off-resonantly
- Multi-mode gates use all modes constructively
- Mode crowding limits scalability

---

## Section E: Mølmer-Sørensen Gate

### Solution 23

**(a) Transitions:**

- Frequency $$\omega_0 + \omega_m + \delta$$: Blue sideband (adds phonon)
- Frequency $$\omega_0 - \omega_m - \delta$$: Red sideband (removes phonon)

**(b) Carrier Suppression:**

The two frequency components are symmetrically detuned from the carrier. Their effects on the carrier transition destructively interfere when properly balanced.

**(c) Gate Time:**

$$\tau = \frac{2\pi}{\delta} = \frac{2\pi}{2\pi \times 10 \times 10^3} = 100 \text{ μs}$$

$$\boxed{\tau = 100 \text{ μs}}$$

---

### Solution 24

**(a) Displacement:**

$$\hat{D}(\alpha)|0\rangle = |\alpha\rangle$$ (coherent state)

$$\langle\hat{x}\rangle = \sqrt{\frac{2\hbar}{m\omega}}\text{Re}(\alpha)$$
$$\langle\hat{p}\rangle = \sqrt{2\hbar m\omega}\text{Im}(\alpha)$$

**(b) Geometric Phase:**

For spin states $$|\pm\pm\rangle$$, the displacement is $$\alpha_{\pm\pm} = (\pm 1 \pm 1)\alpha_0$$.

The geometric phase is the area enclosed in phase space:

$$\phi_{geom} = 2\text{Im}(\alpha^*d\alpha) = |\alpha|^2\sin(\theta)$$

For a closed loop, $$\phi = \pi|\alpha_{max}|^2$$.

**(c) Closed Loop Condition:**

When the motional state returns to its initial position, the joint spin-motion state factors:

$$|\psi_f\rangle = e^{i\phi}|\psi_{spin}\rangle \otimes |\psi_{motion,initial}\rangle$$

No residual entanglement because the loop is closed.

---

### Solution 25

**(a) Block Structure:**

In the basis $$\{|++,n\rangle, |+-,n\rangle, |-+,n\rangle, |--,n\rangle\}$$:

The Hamiltonian couples $$|n\rangle$$ to $$|n\pm 1\rangle$$ with spin-dependent amplitudes.

**(b) Collective Operators:**

$$\hat{S}_\phi = \frac{1}{2}(\hat{\sigma}_\phi^{(1)} + \hat{\sigma}_\phi^{(2)})$$

$$\hat{H}_{MS} = 2\hbar\Omega\hat{S}_\phi(\hat{a}e^{-i\delta t} + \hat{a}^\dagger e^{i\delta t})$$

**(c) Effective Interaction:**

After time $$\tau = 2\pi/\delta$$:

$$\hat{U}_{MS} = \exp\left(-i\frac{4\Omega^2}{\delta}\hat{S}_\phi^2\right)$$

For $$4\Omega^2\tau/\delta = \pi/2$$, this gives a maximally entangling gate.

---

### Solution 26

**(a) Evolution:**

Starting from $$|s_1s_2\rangle|n\rangle$$ where $$s_i = \pm 1$$:

The spin-dependent force creates displacement $$\alpha_{s_1s_2}(t)$$.

After time $$\tau$$: $$|s_1s_2\rangle|n\rangle \rightarrow e^{i\phi(s_1,s_2)}|s_1s_2\rangle|n\rangle$$

**(b) Phase Independence:**

The geometric phase $$\phi(s_1,s_2) = (s_1 + s_2)^2 \Omega^2/\delta$$ depends only on spin configuration, not on $$n$$.

$$\boxed{\phi_{spin} \text{ is independent of } n}$$

**(c) Thermal State:**

For initial thermal state $$\rho_{th}$$:

$$\rho_f = \sum_n p_n e^{i\phi(s_1,s_2)}|s_1s_2\rangle\langle s_1s_2|e^{-i\phi(s_1,s_2)} \otimes |n\rangle\langle n|$$

The spin state acquires the geometric phase regardless of initial temperature.

---

### Solution 27

**(a) Generalized Hamiltonian:**

$$\hat{H} = \hbar\sum_{j,m}\Omega_j b_{jm}\hat{\sigma}_\phi^{(j)}(\hat{a}_m e^{-i\delta_m t} + h.c.)$$

where $$b_{jm}$$ is the mode participation of ion $$j$$ in mode $$m$$.

**(b) Mode Closure Constraint:**

For no residual entanglement:

$$\alpha_m(\tau) = \int_0^\tau \Omega(t)e^{i\delta_m t}dt = 0$$ for all modes $$m$$

This requires the pulse envelope $$\Omega(t)$$ to satisfy N constraints for N modes.

**(c) AM vs FM Gates:**

**Amplitude Modulation:**
- Varies $$\Omega(t)$$ while keeping frequencies fixed
- Easier calibration
- May require longer gate times

**Frequency Modulation:**
- Chirps the laser frequency
- Can achieve faster gates
- More complex calibration
- Better for many-ion systems

---

## Section F: Integration & Comparison

### Solution 28

**(a) SWAP Count:**

For qubits separated by distance $$d$$ on a 1D chain, we need $$d-1$$ SWAPs to bring them adjacent, then 1 CNOT, then $$d-1$$ SWAPs to return.

Total: $$2(d-1) + 1$$ operations, each SWAP = 3 CNOTs.

For average separation $$\bar{d} \sim n/3$$ in n-qubit system: ~$$n$$ CNOT-equivalents per logical CNOT.

**(b) Trapped Ion Gate Time:**

Two-qubit gate time determined by:
- Mode frequency (detuning $$\delta$$)
- Rabi frequency $$\Omega$$
- Gate time $$\tau = 2\pi/\delta$$

Typical: 50-200 μs per gate.

**(c) Circuit Time Estimate:**

**Superconducting (100 qubits, NN connectivity):**
- 50 random CNOTs need ~500 SWAP-equivalent operations
- Each CNOT ~100 ns
- Total: ~50 μs

**Trapped Ion (50 qubits, all-to-all):**
- 50 CNOTs directly, no SWAPs
- Each gate ~100 μs
- Total: ~5 ms

---

### Solution 29

**(a) Output Fidelity:**

For circuit with $$G$$ two-qubit gates, fidelity $$F \approx (1-\epsilon)^G$$.

**Superconducting ($$\epsilon = 10^{-3}$$):**

Need to account for SWAP overhead. Assume effective gates = 3000 (with connectivity overhead).

$$F \approx (0.999)^{3000} \approx e^{-3} \approx 0.05$$

**Trapped Ion ($$\epsilon = 10^{-4}$$):**

Direct implementation, 1000 gates.

$$F \approx (0.9999)^{1000} \approx e^{-0.1} \approx 0.90$$

$$\boxed{F_{SC} \approx 0.05, \quad F_{ion} \approx 0.90}$$

**(b) Recommendation:**

Trapped ions are strongly preferred for this application due to:
- Higher gate fidelity
- All-to-all connectivity (no SWAP overhead)
- Resulting output fidelity 18x higher

**(c) Error Correction Overhead:**

For superconducting to achieve $$F > 0.5$$:

$$0.5 = (1 - \epsilon_{eff})^{3000}$$
$$\epsilon_{eff} < 2 \times 10^{-4}$$

Using surface code with physical error $$10^{-3}$$, need code distance $$d \sim 5$$, requiring ~50 physical qubits per logical qubit. Total: ~5000 physical qubits.

---

### Solution 30

**(a) Quantum Chemistry:**

**Trapped Ions:**
- All-to-all connectivity matches molecular orbital interactions
- No SWAP overhead for fermionic simulations
- Preferred for high-accuracy small molecules

**Superconducting:**
- Requires fermion-to-qubit mapping with locality
- More qubits available for larger systems
- Better for approximate methods (VQE)

**(b) Quantum ML:**

**Superconducting advantages:**
- Large qubit counts (100+)
- Fast gates enable more iterations
- Variational circuits tolerate some errors
- Better for NISQ-era QML

**(c) Hybrid Optimization:**

**Superconducting preferred:**
- Fast gate times (ns vs μs)
- Rapid iteration between quantum and classical
- Lower latency in optimization loop
- Better throughput for VQE/QAOA

---

## Bonus Solution

**(a) 10-Qubit Processor Design:**

**Platform:** Superconducting transmon

**Layout:** 2×5 grid with nearest-neighbor coupling plus diagonal couplers

**Specifications:**
- Qubit frequencies: 4.8-5.5 GHz (staggered)
- Anharmonicity: -250 MHz
- T1: 200 μs, T2: 100 μs
- Single-qubit gate fidelity: 99.95%
- Two-qubit gate fidelity: 99.5%

**(b) Achievable Circuit Depth:**

For output fidelity $$F > 0.1$$:

$$0.1 = (0.995)^{G} \times (0.9995)^{G}$$

$$G \approx 300$$ two-qubit gates

With parallelization, depth ~100-150.

**(c) Error Mitigation:**

- Zero-noise extrapolation: Run at 1x, 1.5x, 2x noise, extrapolate
- Probabilistic error cancellation for systematic errors
- Symmetry verification for conserved quantities
- Post-selection on measurement outcomes

**(d) Classical Resources:**

- Control electronics: 20-channel arbitrary waveform generators
- Readout: 10-channel digitizers, ~1 GS/s
- FPGA for real-time feedback: ~$50k
- Cryostat: dilution refrigerator, 10 mK base
- Total classical compute: ~100 TFLOPS for real-time control
