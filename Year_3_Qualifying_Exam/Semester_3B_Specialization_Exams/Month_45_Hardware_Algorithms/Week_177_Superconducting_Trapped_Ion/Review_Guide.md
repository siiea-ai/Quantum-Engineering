# Week 177: Review Guide - Superconducting & Trapped Ion Systems

## Introduction

This comprehensive review guide covers the physics, engineering, and practical aspects of superconducting and trapped ion quantum computing platforms. These two platforms represent the most mature quantum computing technologies, with Google, IBM, and Rigetti leading superconducting development, while Quantinuum, IonQ, and academic groups advance trapped ion systems.

---

## Part I: Superconducting Quantum Computing

### 1.1 Josephson Junction Physics

The Josephson junction is the essential nonlinear, dissipationless element that enables quantum coherence in superconducting circuits. When two superconductors are separated by a thin insulating barrier (typically aluminum oxide, ~1 nm thick), Cooper pairs can tunnel across, establishing a phase relationship.

**The Josephson Relations:**

The DC Josephson effect describes supercurrent flow:

$$I = I_c \sin(\phi)$$

where $$I_c$$ is the critical current and $$\phi = \phi_1 - \phi_2$$ is the gauge-invariant phase difference across the junction.

The AC Josephson effect relates voltage to phase evolution:

$$V = \frac{\Phi_0}{2\pi}\frac{d\phi}{dt}$$

where $$\Phi_0 = h/2e \approx 2.07 \times 10^{-15}$$ Wb is the magnetic flux quantum.

**Junction Energy:**

Integrating the power $$P = IV$$ gives the potential energy:

$$U(\phi) = -E_J\cos(\phi) + \text{const}$$

where $$E_J = I_c\Phi_0/2\pi$$ is the Josephson energy. This cosine potential is the source of anharmonicity enabling qubit operation.

### 1.2 Circuit Quantization

To derive the quantum Hamiltonian, we apply the standard procedure of circuit quantization:

**Step 1: Classical Lagrangian**

For a circuit with capacitance C and Josephson junction:

$$\mathcal{L} = \frac{1}{2}C\dot{\Phi}^2 + E_J\cos\left(\frac{2\pi\Phi}{\Phi_0}\right)$$

where $$\Phi$$ is the node flux variable related to phase by $$\phi = 2\pi\Phi/\Phi_0$$.

**Step 2: Conjugate Momentum**

The charge on the capacitor is the conjugate momentum:

$$Q = \frac{\partial\mathcal{L}}{\partial\dot{\Phi}} = C\dot{\Phi}$$

**Step 3: Hamiltonian**

Performing the Legendre transformation:

$$H = \frac{Q^2}{2C} - E_J\cos\left(\frac{2\pi\Phi}{\Phi_0}\right)$$

**Step 4: Quantization**

Promoting variables to operators with $$[\hat{\Phi}, \hat{Q}] = i\hbar$$:

$$\hat{H} = 4E_C\hat{n}^2 - E_J\cos\hat{\phi}$$

where $$\hat{n} = \hat{Q}/2e$$ is the number operator for Cooper pairs and $$E_C = e^2/2C$$ is the charging energy.

### 1.3 The Transmon Qubit

The transmon (transmission-line shunted plasma oscillation qubit) was developed at Yale in 2007 to address the charge noise sensitivity of the Cooper pair box.

**Design Principle:**

By adding a large shunt capacitance $$C_B$$, the charging energy is reduced:

$$E_C = \frac{e^2}{2(C_J + C_B)} \ll E_J$$

The transmon operates in the regime $$E_J/E_C \sim 50-100$$.

**Charge Noise Suppression:**

The sensitivity to offset charge $$n_g$$ scales as:

$$\frac{\partial\omega_{01}}{\partial n_g} \propto e^{-\sqrt{8E_J/E_C}}$$

For $$E_J/E_C = 50$$, this gives suppression by a factor of ~$$10^{-9}$$ compared to the Cooper pair box.

**Energy Levels:**

In the transmon regime, we can expand the cosine potential:

$$-E_J\cos\hat{\phi} \approx -E_J + \frac{E_J}{2}\hat{\phi}^2 - \frac{E_J}{24}\hat{\phi}^4 + ...$$

The quadratic term gives a harmonic oscillator with frequency:

$$\omega_p = \sqrt{8E_JE_C}/\hbar$$

The quartic term provides anharmonicity:

$$\alpha = E_{12} - E_{01} \approx -E_C$$

Typical values: $$\omega_{01}/2\pi \sim 5$$ GHz, $$\alpha/2\pi \sim -200$$ MHz.

**The Transmon Hamiltonian (Quantized):**

$$\hat{H} = \hbar\omega_{01}\hat{a}^\dagger\hat{a} + \frac{\alpha}{2}\hat{a}^\dagger\hat{a}^\dagger\hat{a}\hat{a}$$

where $$\hat{a}$$ and $$\hat{a}^\dagger$$ are ladder operators for the transmon.

### 1.4 Flux Qubits

Flux qubits encode information in the direction of persistent current flow around a superconducting loop.

**The RF-SQUID:**

A superconducting loop interrupted by one Josephson junction has Hamiltonian:

$$\hat{H} = 4E_C\hat{n}^2 + \frac{(\hat{\Phi} - \Phi_{ext})^2}{2L} - E_J\cos\left(\frac{2\pi\hat{\Phi}}{\Phi_0}\right)$$

where L is the loop inductance and $$\Phi_{ext}$$ is the external flux.

**Double-Well Potential:**

Near half-flux bias ($$\Phi_{ext} = \Phi_0/2$$), the potential has two minima corresponding to clockwise and counterclockwise circulating currents. The qubit states are:

$$|0\rangle = |\circlearrowright\rangle, \quad |1\rangle = |\circlearrowleft\rangle$$

**Fluxonium:**

The fluxonium qubit replaces the loop inductance with a chain of large Josephson junctions (superinductance), achieving:
- $$E_J/E_C \sim 1-10$$
- Very large anharmonicity
- T1 times exceeding 1 ms demonstrated

### 1.5 Two-Qubit Gates

**Cross-Resonance (CR) Gate:**

Used in IBM systems. Driving qubit 1 at the frequency of qubit 2 creates an effective ZX interaction:

$$\hat{H}_{CR} \propto \hat{Z}_1 \otimes \hat{X}_2$$

The gate time is determined by the coupling strength and detuning.

**Tunable Coupler:**

A third transmon between two computational qubits mediates the coupling. By tuning the coupler frequency, the effective $$ZZ$$ interaction can be turned on/off:

$$g_{eff} = g_1g_2\left(\frac{1}{\Delta_1} + \frac{1}{\Delta_2}\right)$$

**iSWAP and CZ Gates:**

Frequency-tunable transmons can implement iSWAP by tuning qubits into resonance:

$$\text{iSWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

CZ gates use the $$|11\rangle \leftrightarrow |02\rangle$$ avoided crossing.

### 1.6 Dispersive Readout

**Circuit QED Hamiltonian:**

A transmon coupled to a microwave resonator:

$$\hat{H} = \hbar\omega_r\hat{a}^\dagger\hat{a} + \hbar\omega_q\hat{b}^\dagger\hat{b} + \frac{\alpha}{2}\hat{b}^\dagger\hat{b}^\dagger\hat{b}\hat{b} + \hbar g(\hat{a}\hat{b}^\dagger + \hat{a}^\dagger\hat{b})$$

In the dispersive regime ($$|\Delta| = |\omega_q - \omega_r| \gg g$$):

$$\hat{H}_{disp} \approx \hbar(\omega_r + \chi\hat{\sigma}_z)\hat{a}^\dagger\hat{a} + \frac{\hbar\omega_q}{2}\hat{\sigma}_z$$

where $$\chi = g^2/\Delta$$ is the dispersive shift.

**Readout Protocol:**

1. Send microwave pulse at resonator frequency
2. Qubit state shifts resonator by $$\pm\chi$$
3. Measure transmitted/reflected signal
4. State-dependent phase shift enables discrimination

---

## Part II: Trapped Ion Quantum Computing

### 2.1 Paul Trap Physics

Trapped ion quantum computers confine charged atoms using oscillating electromagnetic fields in a Paul trap (RF trap).

**Equations of Motion:**

In a linear Paul trap, the potential is:

$$\Phi(x,y,z,t) = \frac{V_{RF}\cos(\Omega_{RF}t)}{2r_0^2}(x^2 - y^2) + \frac{\kappa V_{DC}}{z_0^2}\left(z^2 - \frac{x^2+y^2}{2}\right)$$

This leads to Mathieu equations for ion motion. In the pseudopotential approximation (valid for small oscillation amplitudes):

$$\omega_x = \omega_y = \omega_r = \frac{qV_{RF}}{\sqrt{2}m\Omega_{RF}r_0^2}$$

$$\omega_z = \sqrt{\frac{2\kappa qV_{DC}}{mz_0^2}}$$

Typical parameters: $$\omega_r/2\pi \sim 5$$ MHz radial, $$\omega_z/2\pi \sim 1$$ MHz axial.

### 2.2 Qubit Encoding

**Hyperfine Qubits:**

Using two hyperfine ground states, e.g., in $$^{171}$$Yb$$^+$$:

$$|0\rangle = |F=0, m_F=0\rangle$$
$$|1\rangle = |F=1, m_F=0\rangle$$

Advantages: Long coherence times (minutes), magnetic field insensitivity (clock states).

**Optical Qubits:**

Using ground and metastable states, e.g., in $$^{40}$$Ca$$^+$$:

$$|0\rangle = |4S_{1/2}\rangle$$
$$|1\rangle = |3D_{5/2}\rangle$$

Advantages: Direct optical addressing, no microwave infrastructure.

### 2.3 Laser-Ion Interaction

**The Interaction Hamiltonian:**

For a laser beam interacting with a trapped ion:

$$\hat{H}_{int} = \frac{\hbar\Omega}{2}\left(\hat{\sigma}^+ e^{i(kz - \omega_L t + \phi)} + \text{h.c.}\right)$$

Expanding the position operator $$\hat{z} = z_0(\hat{a} + \hat{a}^\dagger)$$ where $$z_0 = \sqrt{\hbar/2m\omega_z}$$:

$$e^{ik\hat{z}} = e^{i\eta(\hat{a} + \hat{a}^\dagger)}$$

where $$\eta = kz_0$$ is the Lamb-Dicke parameter (typically $$\eta \sim 0.1$$).

**Resolved Sideband Regime:**

When $$\Omega \ll \omega_z$$ (narrow laser linewidth), we can address individual motional transitions:

- **Carrier:** $$\omega_L = \omega_0$$ → $$|g,n\rangle \leftrightarrow |e,n\rangle$$
- **Red sideband:** $$\omega_L = \omega_0 - \omega_z$$ → $$|g,n\rangle \leftrightarrow |e,n-1\rangle$$
- **Blue sideband:** $$\omega_L = \omega_0 + \omega_z$$ → $$|g,n\rangle \leftrightarrow |e,n+1\rangle$$

### 2.4 The Mølmer-Sørensen Gate

The MS gate creates entanglement between ions using their shared motional modes, without requiring ground-state cooling.

**Bichromatic Field:**

Apply two laser frequencies symmetrically detuned from the carrier:

$$\omega_\pm = \omega_0 \pm (\omega_m + \delta)$$

where $$\omega_m$$ is a motional mode frequency and $$\delta$$ is a small detuning.

**Effective Hamiltonian:**

In the Lamb-Dicke regime, the interaction creates spin-dependent forces:

$$\hat{H}_{MS} = \hbar\sum_{j=1}^{N} \Omega_j\hat{\sigma}_{\phi}^{(j)}\left(\hat{a}_m e^{-i\delta t} + \hat{a}_m^\dagger e^{i\delta t}\right)$$

where $$\hat{\sigma}_\phi = \cos\phi\,\hat{\sigma}_x + \sin\phi\,\hat{\sigma}_y$$.

**Time Evolution:**

After time $$\tau = 2\pi/\delta$$, the motional mode returns to its initial state, leaving a pure spin-spin interaction:

$$\hat{U}_{MS} = \exp\left(-i\frac{\Omega^2}{\delta}\hat{S}_\phi^2\tau\right)$$

where $$\hat{S}_\phi = \sum_j \hat{\sigma}_\phi^{(j)}/2$$.

For $$\Omega^2\tau/\delta = \pi/4$$, this produces a maximally entangling gate.

**Robustness:**

The MS gate is robust because:
1. Insensitive to thermal motion (works with hot ions)
2. Insensitive to initial motional state
3. Geometric phase gate (path-independent)

### 2.5 Trapped Ion Architectures

**Linear Chain:**

- All ions in single trap
- All-to-all connectivity via shared modes
- Limited to ~50-100 ions (mode crowding)
- Used by IonQ, early systems

**QCCD (Quantum Charge-Coupled Device):**

- Segmented trap electrodes
- Ions shuttled between zones
- Separate memory, gate, readout regions
- Used by Quantinuum H-series
- Scalable to thousands of qubits

**Photonic Interconnects:**

- Multiple trapping zones connected by photon channels
- Entanglement via photon interference
- Enables distributed quantum computing
- Demonstrated by Oxford, Duke groups

### 2.6 Error Sources and Mitigation

**Superconducting Systems:**
- T1 relaxation: spontaneous emission, quasiparticles
- T2 dephasing: flux noise, charge noise, photon shot noise
- Gate errors: calibration drift, leakage to non-computational states
- Crosstalk: ZZ coupling between neighboring qubits

**Trapped Ion Systems:**
- Motional heating: electric field noise from electrodes
- Laser intensity fluctuations: Rabi frequency instability
- Magnetic field noise: Zeeman shifts (mitigated by clock states)
- Crosstalk: off-resonant coupling to spectator ions

---

## Part III: Platform Comparison

### 3.1 Performance Metrics

| Metric | Superconducting | Trapped Ion |
|--------|-----------------|-------------|
| Single-qubit gate time | 20-50 ns | 1-10 μs |
| Two-qubit gate time | 50-300 ns | 50-200 μs |
| Single-qubit fidelity | 99.9% | 99.99% |
| Two-qubit fidelity | 99-99.9% | 99.5-99.9% |
| T1 time | 100-500 μs | Minutes |
| T2 time | 50-200 μs | Seconds |
| Connectivity | Nearest-neighbor | All-to-all |
| Current scale | 100-1000 qubits | 50-100 qubits |

### 3.2 Application Suitability

**Superconducting advantages:**
- Fast gate speeds enable more operations before decoherence
- Established semiconductor fabrication
- Room for parallelization with many qubits

**Trapped ion advantages:**
- Highest gate fidelities achieved
- All-to-all connectivity reduces circuit depth
- Identical qubits (atoms are fungible)
- Long coherence times

### 3.3 Recent Milestones (2024-2025)

**Google Willow (Dec 2024):**
- 105 qubits
- Below-threshold error correction demonstrated
- Error rate decreases with increasing code distance
- RCS benchmark: 5 minutes vs 10^25 years classical

**IBM Nighthawk (Nov 2025):**
- 120 qubits with 218 tunable couplers
- 5,000+ two-qubit gate depth
- Path to 10,000 gates by 2027

**Quantinuum H2 (2024-2025):**
- 56 qubits demonstrated
- 99.9% two-qubit gate fidelity
- Fault-tolerant logical operations demonstrated

---

## Summary and Exam Preparation

### Key Derivations to Master

1. **Transmon Hamiltonian:** From circuit Lagrangian to quantized form
2. **Dispersive shift:** Second-order perturbation theory in circuit QED
3. **MS gate:** Effective spin-spin interaction from displaced motional states
4. **Pseudopotential:** Time-averaged confining potential in Paul trap

### Common Exam Questions

1. Why does the transmon operate at $$E_J/E_C \gg 1$$? What is sacrificed?
2. Derive the dispersive shift and explain its role in readout
3. Why is the MS gate robust to motional heating?
4. Compare connectivity and its impact on circuit compilation

### Quick Reference Formulas

$$\boxed{E_J = \frac{I_c\Phi_0}{2\pi}, \quad E_C = \frac{e^2}{2C}, \quad \omega_{01} \approx \sqrt{8E_JE_C} - E_C}$$

$$\boxed{\chi = \frac{g^2}{\Delta}\frac{\alpha}{\Delta + \alpha}, \quad \eta = k\sqrt{\frac{\hbar}{2m\omega}}}$$

$$\boxed{\hat{U}_{MS} = \exp\left(-i\theta\hat{S}_\phi^2\right), \quad \theta = \frac{\Omega^2\tau}{\delta}}$$
