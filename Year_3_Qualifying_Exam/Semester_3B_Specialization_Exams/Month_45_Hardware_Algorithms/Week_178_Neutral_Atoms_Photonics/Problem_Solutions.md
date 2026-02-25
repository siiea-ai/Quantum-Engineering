# Week 178: Problem Solutions - Neutral Atoms & Photonics

## Section A: Rydberg Atom Physics

### Solution 1

**(a) Radiative Lifetime:**

$$\tau(n) = \tau(10) \times \left(\frac{n}{10}\right)^3 = 1 \text{ μs} \times \left(\frac{70}{10}\right)^3$$

$$\tau(70) = 1 \times 343 = 343 \text{ μs}$$

$$\boxed{\tau(70) \approx 340 \text{ μs}}$$

**(b) Orbital Radius:**

$$\langle r \rangle = n^2 a_0 = 70^2 \times 0.053 \text{ nm} = 4900 \times 0.053 \text{ nm}$$

$$\boxed{\langle r \rangle \approx 260 \text{ nm}}$$

**(c) Ground State Atoms:**

$$N = \frac{260 \text{ nm}}{0.5 \text{ nm}} = 520$$

$$\boxed{\text{About 520 ground-state atoms would fit}}$$

---

### Solution 2

**(a) SI Units:**

$$C_6/h = 100 \text{ GHz·μm}^6 = 100 \times 10^9 \times (10^{-6})^6 \text{ Hz·m}^6$$

$$C_6 = h \times 100 \times 10^{-27} \text{ Hz·m}^6 = 6.626 \times 10^{-34} \times 10^{-25}$$

$$\boxed{C_6 \approx 6.6 \times 10^{-59} \text{ J·m}^6}$$

**(b) Interaction Energy at 5 μm:**

$$V = \frac{C_6}{R^6} = \frac{100 \text{ GHz·μm}^6}{(5 \text{ μm})^6} = \frac{100 \times 10^9}{15625} \text{ Hz}$$

$$V = 6.4 \times 10^6 \text{ Hz} = 6.4 \text{ MHz}$$

$$\boxed{V/h = 6.4 \text{ MHz}}$$

**(c) C6 for n=70:**

$$C_6(70) = C_6(50) \times \left(\frac{70}{50}\right)^{11} = 100 \times (1.4)^{11}$$

$$(1.4)^{11} \approx 56$$

$$\boxed{C_6(70)/h \approx 5600 \text{ GHz·μm}^6}$$

---

### Solution 3

**(a) Blockade Radius:**

$$R_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6} = \left(\frac{500 \times 10^9}{2 \times 10^6}\right)^{1/6} \text{ μm}$$

$$R_b = (2.5 \times 10^5)^{1/6} \text{ μm} = 7.9 \text{ μm}$$

$$\boxed{R_b \approx 8 \text{ μm}}$$

**(b) Atoms in Blockade Sphere:**

Volume of blockade sphere: $$V_b = \frac{4}{3}\pi R_b^3 = \frac{4}{3}\pi (8)^3 \approx 2145 \text{ μm}^3$$

Volume per atom (cubic lattice, 3 μm spacing): $$V_{atom} = 27 \text{ μm}^3$$

$$N = V_b/V_{atom} = 2145/27 \approx 80$$

$$\boxed{\text{About 80 atoms in blockade sphere}}$$

**(c) Increasing Laser Power:**

$$\Omega \rightarrow 2\Omega$$ (power ×4 means intensity ×4, $$\Omega \propto \sqrt{I}$$)

$$R_b' = \left(\frac{C_6}{\hbar \cdot 2\Omega}\right)^{1/6} = R_b \times 2^{-1/6} = 8 \times 0.89 = 7.1 \text{ μm}$$

$$\boxed{R_b' \approx 7.1 \text{ μm} \text{ (decreases with higher power)}}$$

---

### Solution 4

**(a) Minimum C6:**

For effective blockade: $$V(R) = C_6/R^6 > \hbar\Omega$$

$$C_6 > \hbar\Omega \times R^6 = h \times 1 \text{ MHz} \times (5)^6 \text{ μm}^6$$

$$C_6/h > 15625 \text{ MHz·μm}^6 = 15.6 \text{ GHz·μm}^6$$

$$\boxed{C_6/h > 15.6 \text{ GHz·μm}^6}$$

**(b) Gate Fidelity:**

$$F \approx 1 - \left(\frac{\hbar\Omega}{V}\right)^2 = 1 - \left(\frac{\hbar\Omega R^6}{C_6}\right)^2$$

$$= 1 - \left(\frac{1 \text{ MHz} \times (5)^6}{200 \times 10^3 \text{ MHz}}\right)^2 = 1 - \left(\frac{15625}{200000}\right)^2$$

$$= 1 - (0.078)^2 = 1 - 0.006 = 0.994$$

$$\boxed{F \approx 99.4\%}$$

**(c) Optimal Rabi Frequency:**

For $$F > 0.99$$: $$(\hbar\Omega R^6/C_6)^2 < 0.01$$

$$\Omega < 0.1 \times C_6/(hR^6) = 0.1 \times 200 \times 10^3/(5)^6 \text{ MHz}$$

$$\Omega < 1.28 \text{ MHz}$$

Maximum speed: $$\boxed{\Omega_{max}/2\pi \approx 1.3 \text{ MHz}}$$

---

### Solution 5

**(a) Decay Probability:**

$$P_{decay} = 1 - e^{-t/\tau} \approx t/\tau = \frac{1 \text{ μs}}{100 \text{ μs}} = 0.01$$

$$\boxed{P_{decay} = 1\%}$$

**(b) Circuit with 100 Gates:**

Each gate has ~3 Rydberg pulses, total Rydberg time: $$100 \times 3 \times t_{pulse}$$

If each pulse is 0.5 μs: Total time = 150 μs

$$P_{total} \approx 150/100 = 1.5 \text{ (oversimplified)}$$

More carefully: $$(1 - 0.01)^{300} \approx e^{-3} \approx 0.05$$

$$\boxed{\text{Error probability} \approx 95\% \text{ (circuit will likely fail)}}$$

**(c) Comparison:**

This is much worse than ~1% gate error. Need:
- Shorter gate times
- Higher Rydberg states (longer lifetime)
- Fewer gates in circuit
- Error correction

---

### Solution 6

**(a) Hamiltonian:**

$$\hat{H} = \sum_{i=1,2} \left[\frac{\hbar\Omega}{2}(|g_i\rangle\langle r_i| + |r_i\rangle\langle g_i|)\right] + V|rr\rangle\langle rr|$$

In matrix form for basis $$\{|gg\rangle, |gr\rangle, |rg\rangle, |rr\rangle\}$$:

$$\hat{H} = \hbar\begin{pmatrix} 0 & \Omega/2 & \Omega/2 & 0 \\ \Omega/2 & 0 & 0 & \Omega/2 \\ \Omega/2 & 0 & 0 & \Omega/2 \\ 0 & \Omega/2 & \Omega/2 & V/\hbar \end{pmatrix}$$

**(b) Eigenstates:**

For $$V \gg \hbar\Omega$$, the symmetric state $$|+\rangle = (|gr\rangle + |rg\rangle)/\sqrt{2}$$ couples to $$|gg\rangle$$ with enhanced Rabi frequency $$\sqrt{2}\Omega$$.

$$|rr\rangle$$ is shifted by $$V$$ and weakly coupled.

**(c) Blockade Shift:**

The state $$|rr\rangle$$ has energy $$E_{rr} = V$$, far detuned from the laser resonance. Effective coupling to $$|rr\rangle$$ is suppressed by $$(\hbar\Omega/V)^2$$.

$$\boxed{|rr\rangle \text{ shifted by } V, \text{ population } \propto (\hbar\Omega/V)^2}$$

---

### Solution 7

**(a) CCZ Pulse Sequence:**

For three atoms A, B, C all within mutual blockade:

1. $$\pi$$ pulse on A: $$|1_A\rangle \rightarrow |r_A\rangle$$
2. $$\pi$$ pulse on B (blocked if A in $$|r\rangle$$): $$|1_B\rangle \rightarrow |r_B\rangle$$
3. $$2\pi$$ pulse on C (blocked if A or B in $$|r\rangle$$)
4. $$\pi$$ pulse on B: $$|r_B\rangle \rightarrow |1_B\rangle$$
5. $$\pi$$ pulse on A: $$|r_A\rangle \rightarrow |1_A\rangle$$

Only $$|111\rangle$$ acquires no phase from step 3 (blocked), giving CCZ.

**(b) Geometry Constraints:**

All three pairwise distances must be less than $$R_b$$:

$$d_{AB}, d_{BC}, d_{AC} < R_b$$

Equilateral triangle with side < $$R_b$$ works.

**(c) Resource Comparison:**

- Native CCZ: 1 gate operation, depth 5 pulses
- Decomposed: 6 CNOT + several single-qubit gates, depth ~10

$$\boxed{\text{Native CCZ: 5 pulses vs decomposed: ~15 gates}}$$

---

## Section B: Neutral Atom Arrays

### Solution 8

**(a) Rayleigh Range:**

$$z_R = \frac{\pi w_0^2}{\lambda} = \frac{\pi (0.7)^2}{0.852} \text{ μm} = \frac{1.54}{0.852} \approx 1.8 \text{ μm}$$

$$\boxed{z_R \approx 1.8 \text{ μm}}$$

**(b) Trap Depth Ratio:**

$$\frac{U_0}{k_B T} = \frac{1 \text{ mK}}{20 \text{ μK}} = 50$$

$$\boxed{U_0/k_BT = 50}$$

**(c) Escape Probability:**

$$\Gamma_{esc} \propto e^{-U_0/k_BT} = e^{-50} \approx 10^{-22}$$

Over 1 second, probability is essentially zero.

$$\boxed{P_{escape} \approx 0}$$

---

### Solution 9

**(a) Expected Loaded Atoms:**

$$\langle N \rangle = 100 \times 0.5 = 50$$

$$\boxed{\langle N \rangle = 50}$$

**(b) Defect-Free Probability:**

$$P = 0.5^{100} \approx 10^{-30}$$

$$\boxed{P \approx 10^{-30} \text{ (essentially zero)}}$$

**(c) After Rearrangement:**

Starting with ~50 atoms, need ~50 moves to fill 100 sites.

With 99% fidelity per move: $$(0.99)^{50} \approx 0.60$$

So ~60% of initially loaded atoms successfully placed.

Final filling: $$\approx 50 \times 0.60 / 100 \times 100 = 30$$ atoms, but we use atoms from reservoir.

With sufficient reservoir and 99% move fidelity: $$\boxed{\text{Final filling} > 98\%}$$

---

### Solution 10

**(a) Microwave Wavelength:**

$$\lambda = c/f = (3 \times 10^8)/(6.835 \times 10^9) = 0.044 \text{ m} = 4.4 \text{ cm}$$

$$\boxed{\lambda = 4.4 \text{ cm}}$$

**(b) Two-Photon Rabi Frequency:**

$$\Omega_{eff} = \frac{\Omega_1\Omega_2}{2\Delta} = \frac{(100)^2}{2 \times 100 \times 10^3} \text{ MHz} = \frac{10^4}{2 \times 10^5} = 0.05 \text{ MHz}$$

$$\boxed{\Omega_{eff}/2\pi = 50 \text{ kHz}}$$

**(c) Decoherence Sources:**

Main sources for clock-state qubits:
- Differential AC Stark shift from trap light
- Magnetic field fluctuations (second-order Zeeman)
- Photon scattering from trap and Raman beams
- Motional decoherence from trap fluctuations

$$\boxed{\text{Differential light shifts are typically dominant}}$$

---

### Solution 11

**(a) Atom Loss Protocol:**

1. Apply resonant push beam on one qubit state (e.g., $$|0\rangle$$)
2. Atoms in $$|0\rangle$$ are expelled from trap
3. Image array to detect presence/absence
4. Presence = $$|1\rangle$$, absence = $$|0\rangle$$

**(b) Main Challenge:**

Non-destructive readout requires detecting atom without expelling it.
- Need shelving to metastable state
- Fluorescence detection heats the atom
- Cross-talk to neighboring qubits

**(c) Recent Experiments:**

Harvard/QuEra (2024):
- Shelving to different hyperfine manifold
- Site-selective imaging with spatial light modulator
- Demonstrated mid-circuit measurement and feed-forward

$$\boxed{\text{Shelving + selective imaging enables non-destructive readout}}$$

---

### Solution 12

**(a) Error Sources:**

1. Spontaneous emission from Rydberg state
2. Blockade imperfection (finite V/Ω ratio)
3. Motional effects (atom position fluctuations)
4. Laser phase/intensity noise
5. State preparation and measurement errors

**(b) Spontaneous Emission Error:**

$$\epsilon_{SE} = N_{pulses} \times t_{pulse}/\tau = 3 \times 500 \text{ ns}/100 \text{ μs}$$

$$\epsilon_{SE} = 1500/(10^5) = 0.015$$

$$\boxed{\epsilon_{SE} = 1.5\%}$$

**(c) Blockade Error:**

$$\epsilon_b = (\hbar\Omega/V)^2 = (1/20)^2 = 0.0025$$

$$\boxed{\epsilon_b = 0.25\%}$$

**(d) Total Fidelity:**

$$\epsilon_{total} = \sqrt{\epsilon_{SE}^2 + \epsilon_b^2 + \epsilon_{other}^2}$$

Assuming $$\epsilon_{other} \approx 0.5\%$$:

$$\epsilon_{total} \approx \sqrt{(1.5)^2 + (0.25)^2 + (0.5)^2}\% \approx 1.6\%$$

$$\boxed{F \approx 98.4\%}$$

---

### Solution 13

**(a) Direct Gate Pairs:**

Distances:
- q0-q1: 5 μm < 8 μm (yes)
- q0-q2: 5 μm < 8 μm (yes)
- q1-q3: 5 μm < 8 μm (yes)
- q2-q3: 5 μm < 8 μm (yes)
- q0-q3: $$\sqrt{50}$$ = 7.07 μm < 8 μm (yes)
- q1-q2: $$\sqrt{50}$$ = 7.07 μm < 8 μm (yes)

$$\boxed{\text{All pairs can perform direct gates}}$$

**(b) q0-q3 Gate:**

Direct gate is possible since 7.07 μm < $$R_b$$ = 8 μm.

**(c) Comparison to Superconducting:**

On a 2×2 NN superconducting grid:
- q0-q3 requires: SWAP(q0,q1) + CNOT(q1,q3) + SWAP(q0,q1)
- Total: 3 SWAP × 3 CNOT each + 1 CNOT = 10 CNOTs

Neutral atom: 1 direct CZ

$$\boxed{\text{Neutral atom: 1 gate vs SC: 10 gates}}$$

---

## Section C: Bosonic Codes

### Solution 14

**(a) Peak Spacing:**

$$|0_L\rangle$$ has peaks at $$q = 2s\sqrt{\pi}$$ for integer $$s$$.

Spacing = $$2\sqrt{\pi} \approx 3.54$$

$$\boxed{\text{Spacing} = 2\sqrt{\pi} \approx 3.54}$$

**(b) Correctable Displacement:**

Errors up to $$\sqrt{\pi}/2$$ in position or momentum are correctable.

$$\boxed{\text{Correctable displacement} < \sqrt{\pi}/2 \approx 0.89}$$

**(c) Syndrome Reliability:**

Need $$\sigma_q < \sqrt{\pi}/4$$ for reliable syndrome determination.

$$\sigma_q = 0.2\sqrt{\pi} \approx 0.35 < \sqrt{\pi}/4 \approx 0.44$$

$$\boxed{\text{Yes, syndrome can be reliably determined}}$$

---

### Solution 15

**(a) Overlap:**

$$\langle\alpha|-\alpha\rangle = e^{-2|\alpha|^2} = e^{-8} \approx 3.4 \times 10^{-4}$$

$$\boxed{\langle\alpha|-\alpha\rangle \approx 3.4 \times 10^{-4}}$$

**(b) Normalization:**

$$\langle 0_L|0_L\rangle = |\mathcal{N}|^2(\langle\alpha|\alpha\rangle + \langle\alpha|-\alpha\rangle + \langle-\alpha|\alpha\rangle + \langle-\alpha|-\alpha\rangle)$$

$$= |\mathcal{N}|^2(1 + e^{-2|\alpha|^2} + e^{-2|\alpha|^2} + 1) = |\mathcal{N}|^2(2 + 2e^{-8})$$

$$\mathcal{N} = \frac{1}{\sqrt{2(1 + e^{-8})}} \approx \frac{1}{\sqrt{2}}$$

$$\boxed{\mathcal{N} \approx 1/\sqrt{2}}$$

**(c) Bit-Flip Suppression:**

A bit-flip requires transitioning between $$|\alpha\rangle$$ and $$|-\alpha\rangle$$. The overlap is exponentially small, so quantum tunneling between these states is suppressed. The rate scales as the overlap squared.

$$\boxed{\Gamma_X \propto |\langle\alpha|-\alpha\rangle|^2 \propto e^{-4|\alpha|^2}}$$

---

### Solution 16

**(a) Scaling with $$|\alpha|^2$$:**

$$\frac{\Gamma_X}{\Gamma_Z} \propto e^{-2|\alpha|^2}$$

For $$|\alpha|^2 = 4$$: ratio = $$10^{-4}$$

For $$|\alpha|^2 = 8$$: ratio = $$10^{-4} \times e^{-2(8-4)} = 10^{-4} \times e^{-8} \approx 10^{-4} \times 3.4 \times 10^{-4}$$

$$\boxed{\Gamma_X/\Gamma_Z \approx 3 \times 10^{-8}}$$

**(b) Practical Limits:**

- Larger $$|\alpha|^2$$ requires more photons
- Higher photon loss rate: $$\Gamma_Z \propto |\alpha|^2$$
- More susceptible to anharmonicity effects
- Harder to prepare and stabilize

$$\boxed{\text{Photon loss and higher-order nonlinearities limit } |\alpha|^2}$$

**(c) Repetition Code Scaling:**

For $$n$$ cat qubits in a repetition code:
- Logical bit-flip requires majority of physical bit-flips
- Rate: $$\Gamma_X^{logical} \propto \binom{n}{(n+1)/2}\Gamma_X^{n/2}$$
- Exponentially suppressed with $$n$$

$$\boxed{\Gamma_X^{logical} \propto \Gamma_X^{n/2} \text{ (exponential suppression)}}$$

---

### Solution 17

**(a) Syndrome Measurement:**

Position displacement $$\delta q$$ shifts all grid peaks. Measuring $$\hat{q} \mod \sqrt{\pi}$$ gives:

$$s_q = \delta q \mod \sqrt{\pi}$$

For $$|\delta q| < \sqrt{\pi}/2$$, this uniquely identifies the error.

**(b) Ancilla System:**

- Use a two-level ancilla (qubit)
- Controlled displacement: $$CX = e^{i\hat{p}_{anc}\otimes\hat{q}_{GKP}}$$
- Measure ancilla in $$\hat{p}$$ basis
- Outcome reveals $$\hat{q}_{GKP} \mod \sqrt{\pi}$$

**(c) Correction:**

Apply displacement $$\hat{D}(-s_q)$$ to shift peaks back to grid:

$$\hat{D}(-s_q) = e^{is_q\hat{p}/\hbar}$$

$$\boxed{\text{Apply } \hat{D}(-s_q) = e^{is_q\hat{p}/\hbar}}$$

---

### Solution 18

**(a) Logical Paulis:**

$$\hat{X}_L = e^{i\sqrt{\pi}\hat{p}}$$: Displaces by $$\sqrt{\pi}$$ in position

On $$|0_L\rangle$$: shifts $$2s\sqrt{\pi} \rightarrow (2s+1)\sqrt{\pi}$$, giving $$|1_L\rangle$$

$$\hat{Z}_L = e^{-i\sqrt{\pi}\hat{q}}$$: Multiplies momentum eigenstate by phase

$$\boxed{\hat{X}_L = e^{i\sqrt{\pi}\hat{p}}, \quad \hat{Z}_L = e^{-i\sqrt{\pi}\hat{q}}}$$

**(b) Verification:**

$$\hat{X}_L|0_L\rangle = e^{i\sqrt{\pi}\hat{p}}\sum_s |2s\sqrt{\pi}\rangle$$

$$= \sum_s |2s\sqrt{\pi} + \sqrt{\pi}\rangle = \sum_s |(2s+1)\sqrt{\pi}\rangle = |1_L\rangle$$

$$\boxed{\hat{X}_L|0_L\rangle = |1_L\rangle \checkmark}$$

**(c) Syndrome for Small Displacement:**

Displacement $$\hat{D}(\alpha)$$ with $$\alpha = (\delta q + i\delta p)/\sqrt{2}$$ shifts the grid.

Measuring $$\hat{q} \mod \sqrt{\pi}$$ gives $$s_q = \delta q$$ (for $$|\delta q| < \sqrt{\pi}/2$$).

Measuring $$\hat{p} \mod \sqrt{\pi}$$ gives $$s_p = \delta p$$.

Together, $$\alpha$$ is determined modulo the code stabilizers.

---

### Solution 19

**(a) Physical Qubit Comparison:**

Surface code at distance $$d=5$$: $$d^2 = 25$$ physical qubits per logical qubit.

GKP: 1 oscillator mode (but with required squeezing).

$$\boxed{\text{Surface code: 25 physical qubits vs GKP: 1 mode}}$$

**(b) GKP Resources:**

For 10 dB squeezing:
- Squeezing parameter $$r = 1.15$$
- Requires high-Q cavity ($$Q > 10^6$$)
- Pump power depends on specific implementation
- For superconducting: parametric amplifier with flux-pumping

$$\boxed{\text{High-Q cavity + parametric pumping}}$$

**(c) Trade-offs:**

- **GKP**: Hardware efficient but requires high-quality oscillators and gates
- **Surface code**: More physical qubits but uses simpler two-level systems
- Near-term: Surface code may be easier (established technology)
- Long-term: GKP could have lower overhead

$$\boxed{\text{GKP: fewer systems but stricter requirements}}$$

---

## Section D: Photonic Quantum Computing

### Solution 20

**(a) Hadamard:**

Half-wave plate at 22.5° to the optical axis:

$$|H\rangle \rightarrow (|H\rangle + |V\rangle)/\sqrt{2}, \quad |V\rangle \rightarrow (|H\rangle - |V\rangle)/\sqrt{2}$$

$$\boxed{\text{Half-wave plate at 22.5°}}$$

**(b) Pauli-Z:**

Half-wave plate at 0° (fast axis along H):

$$|H\rangle \rightarrow |H\rangle, \quad |V\rangle \rightarrow -|V\rangle$$

$$\boxed{\text{Half-wave plate at 0° or quarter-wave plate round trip}}$$

**(c) Beam Splitter Output:**

For 50:50 beam splitter, input $$|1,0\rangle$$:

$$|1,0\rangle \rightarrow \frac{1}{\sqrt{2}}(|1,0\rangle + i|0,1\rangle)$$

$$\boxed{(|1,0\rangle + i|0,1\rangle)/\sqrt{2}}$$

---

### Solution 21

**(a) Single-Photon Source:**

$$g^{(2)}(0) = 0$$ for ideal single-photon source (no two-photon events).

$$\boxed{g^{(2)}(0) = 0}$$

**(b) Coherent State:**

$$g^{(2)}(0) = 1$$ for Poissonian statistics (coherent/laser light).

$$\boxed{g^{(2)}(0) = 1}$$

**(c) Multi-Photon Fraction:**

$$g^{(2)}(0) = 0.05$$ means:

$$g^{(2)}(0) \approx \frac{P(n \geq 2)}{P(n=1)^2/2}$$

For weak source: $$P(n \geq 2) \approx 0.05 \times P(n=1)^2/2$$

Approximately 2.5% of successful detections are multi-photon.

$$\boxed{\text{Multi-photon fraction} \approx 2.5\%}$$

---

### Solution 22

**(a) Ancilla Photons:**

Basic NS gate uses 1 ancilla photon.

$$\boxed{\text{1 ancilla photon per attempt}}$$

**(b) CZ Success Probability:**

$$P_{CZ} = P_{NS}^2 = (1/4)^2 = 1/16$$

$$\boxed{P_{CZ} = 1/16 = 6.25\%}$$

**(c) Bell Pairs for 99%:**

$$P_{success} = 1 - 1/(n+1) = 0.99$$

$$1/(n+1) = 0.01$$

$$n + 1 = 100$$

$$\boxed{n = 99 \text{ Bell pairs}}$$

---

### Solution 23

**(a) Circuit Depth:**

A 1D cluster of 5 qubits implements single-qubit operations with depth 4 (each measurement implements one gate layer, last qubit is output).

$$\boxed{\text{Maximum depth} = 4}$$

**(b) Cluster Size:**

For 100 qubits, depth 50:
- Need $$100 \times 50 = 5000$$ cluster qubits
- Plus overhead for 2D/3D structure for two-qubit gates

Estimate: $$\sim 10^4$$ cluster qubits.

$$\boxed{\sim 10^4 \text{ cluster qubits}}$$

**(c) Suitability for Photonics:**

- Single-qubit measurements are easy (polarizers + detectors)
- Entangling gates only needed for state prep (can be probabilistic)
- Measurement is naturally destructive for photons
- No need to store photons during computation

$$\boxed{\text{Natural match: computation = easy measurements on entangled state}}$$

---

### Solution 24

**(a) Squeezed State Statistics:**

$$p(0) = 1/\cosh r = 1/\cosh(1) = 1/1.543 = 0.65$$

$$p(2) = |\tanh 1|^2/(2!\cosh 1) = (0.762)^2/(2 \times 1.543) = 0.581/3.086 = 0.19$$

$$\boxed{p(0) = 0.65, \quad p(2) = 0.19}$$

**(b) Hilbert Space Dimension:**

$$\dim = 11^{216} \approx 10^{225}$$

$$\boxed{\text{Hilbert space dimension} \approx 10^{225}}$$

**(c) Computational Hardness:**

GBS output probabilities involve the Hafnian of a matrix (related to perfect matchings in graphs). Computing the Hafnian is #P-hard in general, similar to the permanent.

$$\boxed{\text{Hafnian computation is #P-hard}}$$

---

### Solution 25

**(a) Dual-Rail CNOT:**

Control: modes C0, C1
Target: modes T0, T1

Circuit:
1. PBS mixes C1 and T1 conditioned on polarization
2. Detect ancilla modes
3. Specific pattern heralds success

**(b) Heralding Pattern:**

Success is heralded when exactly one photon is detected in each output mode of the gate, confirming no photon loss or bunching.

**(c) Feed-Forward:**

Based on which detector clicks:
- If pattern A: apply identity
- If pattern B: apply X on target
- If pattern C: apply Z on target
- If pattern D: apply XZ on target

$$\boxed{\text{Pauli corrections based on measurement outcome}}$$

---

## Section E: Platform Comparison

### Solution 26

**(a) Operation Count:**

**Neutral Atoms:**
- 50 qubits, depth 20 with all-to-all (within blockade)
- ~50 × 20 = 1000 operations
- Some may need rearrangement: add ~10% overhead
- Total: ~1100 operations

**Photonic Cluster:**
- Need cluster state of ~50 × 20 = 1000 qubits
- Each measurement is one operation
- Plus state preparation overhead

$$\boxed{\text{Neutral atom: } \sim 1100 \text{ gates; Photonic: } \sim 1000 \text{ measurements}}$$

**(b) Output Fidelity:**

Neutral atom gate fidelity ~97.5%:
$$F_{NA} = (0.975)^{1000} \approx 10^{-11}$$

Photonic measurement fidelity ~99%:
$$F_{ph} = (0.99)^{1000} \approx 10^{-5}$$

$$\boxed{\text{Photonic has higher fidelity for this circuit}}$$

**(c) Time:**

Neutral atom: 1000 gates × 1 μs = 1 ms
Photonic: Limited by photon generation and measurement, ~10 μs per operation

$$\boxed{\text{Neutral atom: } \sim 1 \text{ ms; Photonic: } \sim 10 \text{ ms}}$$

---

### Solution 27

**(a) Neutral Atom Advantage:**

- Native 2D geometry matches lattice structure
- Reconfigurable arrays for different lattice shapes
- Long-range Rydberg interactions can simulate various couplings
- Large qubit counts available (100+)

$$\boxed{\text{Native 2D geometry and analog simulation capability}}$$

**(b) Connectivity:**

- **Neutral atoms:** 2D grid, all neighbors within blockade
- **Superconducting:** Typically NN, need SWAPs for non-adjacent
- **Trapped ions:** All-to-all but limited qubit count

$$\boxed{\text{Neutral atoms: natural 2D; SC needs many SWAPs}}$$

**(c) Native Interactions:**

Rydberg atoms provide:
- $$1/R^6$$ (van der Waals)
- $$1/R^3$$ (dipole-dipole at Förster resonance)
- Can be mapped to XXZ or XY models

$$\boxed{\text{Tunable power-law interactions from Rydberg physics}}$$

---

### Solution 28

**(a) Photon Collection Efficiency:**

For high-fidelity entanglement, need collection efficiency > 50% per photon.

With two-photon interference: $$\eta_{entangle} \propto \eta_1 \eta_2$$

For useful entanglement rate: $$\eta > 0.5$$ desirable, currently ~1-10%.

$$\boxed{\eta > 50\% \text{ needed for practical rates}}$$

**(b) Network Latency:**

- Light travel time: 3.3 μs/km
- For error correction with syndrome exchange:
  - Round trip adds 2× latency
  - Error correction cycle must be faster than error rate
  - Limits inter-node distance to ~km scale

$$\boxed{\text{Latency limits practical distance to } \sim \text{km}}$$

**(c) Resource Overhead:**

- Need quantum repeaters for long distance
- Each node needs local error correction
- Entanglement distillation consumes many raw pairs
- Overhead factor: 10-100× compared to monolithic

$$\boxed{\text{10-100× overhead for distributed architecture}}$$
