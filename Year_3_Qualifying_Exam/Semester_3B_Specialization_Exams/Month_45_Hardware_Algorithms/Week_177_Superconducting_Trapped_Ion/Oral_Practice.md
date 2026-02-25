# Week 177: Oral Exam Practice - Superconducting & Trapped Ion Systems

## Overview

This document contains oral examination questions organized by topic and difficulty. Practice answering these questions aloud, timing yourself to 3-5 minutes per response. Focus on clear explanations, logical structure, and connecting concepts to broader themes.

---

## Section A: Conceptual Understanding (15 Questions)

### A1. Transmon Design Philosophy

**Question:** Explain why the transmon operates in the regime $$E_J/E_C \gg 1$$. What trade-off is involved in this design choice?

**Key Points to Address:**
- Charge noise in Cooper pair box ($$E_J/E_C \sim 1$$)
- Exponential suppression of charge sensitivity
- Cost: reduced anharmonicity ($$\alpha \approx -E_C$$)
- Still sufficient anharmonicity for qubit operation (~200-300 MHz)
- Historical development from charge qubit to transmon

**Follow-up:** How does the anharmonicity limit gate speeds?

---

### A2. Circuit Quantization

**Question:** Walk me through the procedure for quantizing a superconducting circuit.

**Key Points to Address:**
1. Identify node fluxes $$\Phi_i$$
2. Write classical Lagrangian (kinetic from capacitors, potential from inductors/junctions)
3. Find conjugate charges $$Q_i = \partial\mathcal{L}/\partial\dot{\Phi}_i$$
4. Construct Hamiltonian via Legendre transform
5. Promote to operators with $$[\hat{\Phi}, \hat{Q}] = i\hbar$$
6. Express in convenient basis (charge or flux)

**Follow-up:** When is the charge basis preferred over the flux basis?

---

### A3. Dispersive Readout

**Question:** Explain how dispersive readout works in circuit QED. Why is it called "dispersive"?

**Key Points to Address:**
- Qubit-resonator detuning $$|\Delta| \gg g$$
- Effective Hamiltonian: $$\chi\hat{a}^\dagger\hat{a}\hat{\sigma}_z$$
- Resonator frequency depends on qubit state
- "Dispersive" = no energy exchange, only phase shift
- Measurement via transmitted/reflected microwave signal
- State discrimination from phase difference

**Follow-up:** What limits the readout fidelity?

---

### A4. Two-Qubit Gate Mechanisms

**Question:** Compare and contrast the cross-resonance gate and tunable coupler approaches.

**Key Points to Address:**

**Cross-Resonance:**
- Fixed-frequency qubits
- Drive qubit 1 at qubit 2's frequency
- Creates effective ZX interaction
- Advantages: no flux lines, reduced noise sensitivity

**Tunable Coupler:**
- Third transmon mediates coupling
- Tune coupler frequency to adjust $$g_{eff}$$
- Can achieve $$g_{eff} = 0$$ for low crosstalk
- Advantages: fast gates, low idle error

**Follow-up:** Which approach does IBM use? Google?

---

### A5. Flux Qubit Operation

**Question:** Describe the physical mechanism behind a flux qubit and explain its sweet spots.

**Key Points to Address:**
- Persistent currents in superconducting loop
- Double-well potential from competing terms
- Qubit states: clockwise/counterclockwise current
- Sweet spots at half-integer flux ($$\Phi_{ext} = n\Phi_0/2$$)
- First-order flux noise immunity at sweet spots
- Fluxonium as modern implementation

**Follow-up:** Why has the transmon become more popular than flux qubits?

---

### A6. Paul Trap Physics

**Question:** Explain how a Paul trap confines ions using oscillating fields.

**Key Points to Address:**
- Static fields cannot create 3D minima (Earnshaw's theorem)
- RF quadrupole field creates time-varying saddle
- Pseudopotential from time-averaged motion
- Mathieu equation description
- Secular frequencies vs micromotion
- Stability diagram (a, q parameters)

**Follow-up:** What is micromotion and why is it problematic?

---

### A7. Ion Qubit Encodings

**Question:** Compare hyperfine and optical qubit encodings in trapped ions.

**Key Points to Address:**

**Hyperfine (e.g., Yb-171):**
- Two ground state levels (F=0, F=1)
- Microwave or Raman transitions
- Clock states: first-order field insensitive
- Very long coherence times

**Optical (e.g., Ca-40):**
- Ground state to metastable state
- Direct optical addressing
- No microwave infrastructure
- Faster gates possible

**Follow-up:** Which encoding does Quantinuum use? IonQ?

---

### A8. Mølmer-Sørensen Gate Mechanism

**Question:** Explain how the Mølmer-Sørensen gate creates entanglement without requiring ground-state cooling.

**Key Points to Address:**
- Bichromatic laser field (blue + red sideband)
- Spin-dependent force creates displacement in phase space
- Geometric phase from enclosed area
- Motional mode returns to initial state
- Phase independent of initial phonon number
- Entangling phase depends only on spin configuration

**Follow-up:** What determines the gate time?

---

### A9. QCCD Architecture

**Question:** Describe the QCCD (Quantum CCD) architecture and its advantages.

**Key Points to Address:**
- Segmented trap electrodes
- Ions shuttled between zones
- Specialized regions: gate, readout, memory
- Scales beyond linear chain limitations
- All-to-all connectivity via shuttling
- Quantinuum H-series implementation
- Junction designs for multi-path routing

**Follow-up:** What are the main error sources in QCCD?

---

### A10. Coherence Time Limits

**Question:** What are the dominant decoherence mechanisms in superconducting qubits versus trapped ions?

**Key Points to Address:**

**Superconducting:**
- T1: dielectric loss, quasiparticles, Purcell decay
- T2: flux noise, photon shot noise, TLS
- Typical times: T1 ~ 100-500 μs, T2 ~ 50-200 μs

**Trapped Ions:**
- T1: spontaneous emission (very slow for ground states)
- T2: magnetic field noise, laser phase noise
- Motional decoherence: heating from electrode noise
- Typical times: T1 ~ minutes, T2 ~ seconds

**Follow-up:** How do modern systems mitigate these?

---

### A11. Gate Fidelity Comparison

**Question:** Why do trapped ions typically achieve higher two-qubit gate fidelities than superconducting qubits?

**Key Points to Address:**
- Ions: identical particles, no fabrication variation
- Ions: longer coherence times relative to gate time
- Ions: all-to-all connectivity (no SWAP errors)
- SC: faster gates but higher error per gate
- SC: crosstalk and frequency crowding
- Recent SC improvements closing the gap

**Follow-up:** Can superconducting systems ever match ion fidelities?

---

### A12. Scalability Challenges

**Question:** Discuss the main scalability challenges for each platform.

**Key Points to Address:**

**Superconducting:**
- Wiring/control scaling (N qubits needs N+ lines)
- Frequency crowding
- Crosstalk management
- Cryogenic power dissipation

**Trapped Ions:**
- Mode crowding in long chains
- Shuttling overhead
- Laser power and complexity
- Photonic interconnect challenges

**Follow-up:** What qubit count do you expect each platform to reach by 2030?

---

### A13. Error Correction Readiness

**Question:** Which platform is better positioned for fault-tolerant quantum computing?

**Key Points to Address:**
- Need physical error rates below threshold (~1%)
- Ions: already below threshold
- SC: approaching threshold with recent advances
- Ions: limited qubit count challenges logical scaling
- SC: faster cycle times for syndrome extraction
- Both demonstrated logical operations in 2024-2025

**Follow-up:** What did Google's Willow demonstrate about error correction?

---

### A14. Recent Milestones

**Question:** Describe the most significant quantum hardware advances in 2024-2025.

**Key Points to Address:**
- Google Willow: below-threshold error correction
- IBM Nighthawk: 120 qubits, 218 couplers
- Quantinuum H2: 56 qubits, 99.9% fidelity
- QuEra: 6000+ atom arrays
- Microsoft/Quantinuum: logical qubit demonstrations

**Follow-up:** Which milestone do you consider most important for the field?

---

### A15. Application Matching

**Question:** For quantum chemistry simulation, which platform would you recommend and why?

**Key Points to Address:**
- Chemistry needs: many-body Hamiltonians, fermionic operators
- Connectivity requirements from molecular structure
- Ions: all-to-all suits arbitrary interactions
- SC: may need fermion-to-local mappings
- Accuracy requirements favor higher fidelity
- Small molecules: ions; larger systems: SC with VQE

**Follow-up:** How would your answer change for optimization problems?

---

## Section B: Derivation Questions (10 Questions)

### B1. Transmon Hamiltonian

**Question:** Starting from a parallel LC circuit with a Josephson junction, derive the transmon Hamiltonian.

**Expected Derivation:**
1. Lagrangian: $$\mathcal{L} = \frac{C\dot{\Phi}^2}{2} + E_J\cos(2\pi\Phi/\Phi_0)$$
2. Conjugate charge: $$Q = C\dot{\Phi}$$
3. Hamiltonian: $$H = Q^2/(2C) - E_J\cos\phi$$
4. Quantize: $$[\hat{\Phi}, \hat{Q}] = i\hbar$$
5. In number basis: $$\hat{H} = 4E_C\hat{n}^2 - E_J\cos\hat{\phi}$$

---

### B2. Dispersive Shift

**Question:** Derive the dispersive shift $$\chi$$ using second-order perturbation theory.

**Expected Derivation:**
1. Start with Jaynes-Cummings: $$H = \omega_r a^\dagger a + \omega_q\sigma_z/2 + g(a\sigma^+ + a^\dagger\sigma^-)$$
2. Treat coupling as perturbation
3. Second-order shift: $$\Delta E = g^2/\Delta$$ for $$|g,n+1\rangle$$ state
4. Include anharmonicity correction
5. Result: $$\chi = g^2\alpha/[\Delta(\Delta + \alpha)]$$

---

### B3. Ion Equilibrium Positions

**Question:** Find the equilibrium separation of two ions in a harmonic trap.

**Expected Derivation:**
1. Total potential: $$V = \frac{1}{2}m\omega^2(z_1^2 + z_2^2) + \frac{e^2}{4\pi\epsilon_0|z_1-z_2|}$$
2. Symmetry: $$z_1 = -z_2 = d/2$$
3. Force balance: $$m\omega^2(d/2) = e^2/(4\pi\epsilon_0 d^2)$$
4. Solve: $$d = (e^2/(2\pi\epsilon_0 m\omega^2))^{1/3}$$

---

### B4. Normal Mode Frequencies

**Question:** Derive the normal mode frequencies for two ions.

**Expected Derivation:**
1. Expand potential around equilibrium
2. Coupled equations: $$m\ddot{z}_i = -\partial V/\partial z_i$$
3. Matrix form: $$\ddot{\vec{z}} = -\mathbf{K}\vec{z}$$
4. Eigenvalues of K give mode frequencies
5. COM: $$\omega_z$$, Stretch: $$\sqrt{3}\omega_z$$

---

### B5. MS Gate Phase

**Question:** Calculate the geometric phase acquired in a Mølmer-Sørensen gate.

**Expected Derivation:**
1. Spin-dependent displacement: $$\alpha(t) = -i\Omega S_\phi\int_0^t e^{i\delta t'}dt'$$
2. Closed loop condition: $$\alpha(2\pi/\delta) = 0$$
3. Enclosed area: $$A = \pi|\alpha_{max}|^2$$
4. Phase: $$\phi = 2A = \pi\Omega^2/\delta^2 \times (2\pi/\delta)$$
5. Result: $$\phi = 2\pi\Omega^2\tau/\delta$$

---

### B6. Lamb-Dicke Parameter

**Question:** Derive the Lamb-Dicke parameter and explain its physical significance.

**Expected Derivation:**
1. Ion position: $$z = z_0(a + a^\dagger)$$
2. Laser phase: $$kz = \eta(a + a^\dagger)$$ where $$\eta = kz_0$$
3. $$z_0 = \sqrt{\hbar/(2m\omega)}$$
4. Physical meaning: ratio of wave packet size to laser wavelength
5. Lamb-Dicke regime: $$\eta\sqrt{n+1} \ll 1$$

---

### B7. Purcell Decay

**Question:** Derive the Purcell limit on qubit T1.

**Expected Derivation:**
1. Qubit decays through resonator
2. Decay rate: $$\Gamma_P = (g/\Delta)^2 \kappa$$
3. T1 limit: $$T_1^P = 1/\Gamma_P = \Delta^2/(g^2\kappa)$$
4. Trade-off with readout (need finite $$g$$)
5. Purcell filter solution

---

### B8. Sideband Transition Rates

**Question:** Derive the Rabi frequencies for carrier, red, and blue sideband transitions.

**Expected Derivation:**
1. Interaction: $$H = \Omega\sigma_x\cos(kz - \omega_L t)$$
2. Expand: $$e^{ikz} = e^{i\eta(a+a^\dagger)} \approx 1 + i\eta(a+a^\dagger)$$
3. Carrier: $$\Omega_c = \Omega$$
4. Red sideband: $$\Omega_{rsb} = \eta\sqrt{n}\Omega$$
5. Blue sideband: $$\Omega_{bsb} = \eta\sqrt{n+1}\Omega$$

---

### B9. Charge Noise Suppression

**Question:** Show that the transmon's charge sensitivity is exponentially suppressed.

**Expected Derivation:**
1. CPB Hamiltonian: $$H = 4E_C(n-n_g)^2 - E_J\cos\phi$$
2. Energy depends on $$n_g$$ (offset charge)
3. In transmon limit, expand in $$\phi$$
4. Harmonic approximation gives $$\omega$$ independent of $$n_g$$
5. Corrections: $$\partial\omega/\partial n_g \propto e^{-\sqrt{8E_J/E_C}}$$

---

### B10. Tunable Coupler

**Question:** Derive the effective coupling through a tunable coupler.

**Expected Derivation:**
1. Three-qubit Hamiltonian with coupler
2. Adiabatically eliminate coupler (large detuning)
3. Effective coupling: $$g_{eff} = g_1g_2(1/\Delta_1 + 1/\Delta_2)$$
4. Zero coupling point: $$\omega_c = (\omega_1 + \omega_2)/2$$
5. Tunability range and speed

---

## Section C: Analysis Questions (10 Questions)

### C1. Platform Selection

**Scenario:** You need to run a 50-qubit quantum simulation with depth 500. Given current hardware capabilities, analyze the feasibility on both platforms.

**Analysis Points:**
- Circuit depth with connectivity overhead
- Expected output fidelity
- Runtime comparison
- Error mitigation requirements

---

### C2. Error Budget

**Scenario:** A transmon qubit has T1 = 200 μs, T2 = 100 μs, and single-qubit gate time of 30 ns. Construct an error budget.

**Analysis Points:**
- Relaxation error per gate
- Dephasing error per gate
- Total coherence-limited error
- Compare to reported gate fidelities
- Identify other error sources

---

### C3. Gate Speed Limits

**Question:** What fundamentally limits gate speeds in each platform?

**Analysis Points:**
- SC: anharmonicity sets maximum Rabi frequency
- Ions: motional mode frequency limits MS gate
- SC: leakage vs speed trade-off
- Ions: heating during slow gates
- Optimal operating points

---

### C4. Connectivity Impact

**Question:** Analyze how connectivity affects algorithm implementation for random circuit sampling.

**Analysis Points:**
- Gate count inflation for limited connectivity
- SWAP overhead calculation
- Impact on circuit fidelity
- Compiler optimization potential
- Architecture-specific advantages

---

### C5. Noise Spectroscopy

**Question:** How would you characterize the noise spectrum of a superconducting qubit?

**Analysis Points:**
- Ramsey experiments for $$S(0)$$
- CPMG for filter function approach
- T1 vs T2 ratio analysis
- Distinguishing $$1/f$$ from white noise
- Identifying noise sources from spectrum

---

### C6. Crosstalk Mitigation

**Question:** Describe strategies for mitigating crosstalk in multi-qubit systems.

**Analysis Points:**
- Frequency detuning between neighbors
- Dynamical decoupling during idle
- Calibrated compensation pulses
- Tunable coupler approach
- Post-processing corrections

---

### C7. Scalability Projection

**Question:** Project the resources needed for a 1000-qubit superconducting processor.

**Analysis Points:**
- Control line count (DC, RF, readout)
- Cryogenic heat load
- Room-temperature electronics
- Calibration time scaling
- Yield and replacement strategy

---

### C8. Hybrid Approaches

**Question:** Discuss the potential for hybrid superconducting-ion systems.

**Analysis Points:**
- Microwave-optical transduction
- Modular architecture benefits
- Technical challenges (temperature, coupling)
- State transfer fidelity requirements
- Timeline for demonstration

---

### C9. Benchmarking

**Question:** Critically evaluate quantum volume as a benchmark. What are its limitations?

**Analysis Points:**
- Definition and measurement protocol
- What it captures: gate fidelity, connectivity
- What it misses: algorithm-specific performance
- Gaming potential
- Alternative metrics (CLOPS, circuit layer ops)

---

### C10. Future Outlook

**Question:** What breakthrough would most accelerate progress in quantum hardware?

**Analysis Points:**
- Error rates below threshold
- Improved coherence times
- Better control electronics
- New qubit modalities
- Integration challenges

---

## Section D: Defense Questions (5 Questions)

These are challenging questions that probe the limits of your understanding.

### D1.
**Question:** The transmon sacrifices anharmonicity for charge noise immunity. Is there a way to have both? Discuss alternatives.

---

### D2.
**Question:** Some argue that trapped ions cannot scale beyond ~1000 qubits. Defend or refute this claim.

---

### D3.
**Question:** Why haven't superconducting qubits achieved the same fidelities as trapped ions despite decades of development?

---

### D4.
**Question:** The Mølmer-Sørensen gate is often called "heating insensitive." Under what conditions does this break down?

---

### D5.
**Question:** If you could redesign quantum computing hardware from scratch, what would you change about current approaches?

---

## Oral Exam Tips

1. **Structure your answer:** Introduction, main points, conclusion
2. **Use the whiteboard:** Draw diagrams, write key equations
3. **Connect to fundamentals:** Link specific hardware to basic QM
4. **Acknowledge limitations:** "This works well for X, but fails when Y"
5. **Stay calm with unknowns:** "I'm not certain, but my reasoning is..."
6. **Ask clarifying questions:** Ensure you understand what's being asked
7. **Time management:** Don't go too deep on early parts
8. **Practice aloud:** Verbal fluency requires practice
