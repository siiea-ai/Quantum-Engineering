# Week 178: Review Guide - Neutral Atoms & Photonics

## Introduction

This review guide covers neutral atom and photonic quantum computing platforms, representing two fundamentally different approaches to quantum information processing. Neutral atoms leverage the exquisite control of atomic physics and strong Rydberg interactions, while photonic systems exploit the natural quantum properties of light and enable room-temperature operation.

---

## Part I: Rydberg Atom Physics

### 1.1 Atomic Structure of Rydberg States

Rydberg atoms are atoms with one or more electrons excited to states with very high principal quantum number $$n$$. For alkali atoms (commonly used: Rb-87, Cs-133), the valence electron occupies a hydrogen-like orbit far from the ionic core.

**Energy Levels:**

The binding energy of a Rydberg state follows a modified Rydberg formula:

$$E_n = -\frac{R_\infty hc}{(n - \delta_\ell)^2}$$

where $$R_\infty = 13.6$$ eV is the Rydberg constant and $$\delta_\ell$$ is the quantum defect (depends on orbital angular momentum $$\ell$$). For Rb:
- S states ($$\ell = 0$$): $$\delta_0 \approx 3.13$$
- P states ($$\ell = 1$$): $$\delta_1 \approx 2.65$$
- D states ($$\ell = 2$$): $$\delta_2 \approx 1.35$$

**Scaling Laws:**

The properties of Rydberg atoms scale dramatically with $$n$$:

| Property | Formula | Physical Origin |
|----------|---------|-----------------|
| Orbital radius | $$\langle r \rangle \approx n^2 a_0$$ | Bohr model |
| Binding energy | $$E_n \propto n^{-2}$$ | Coulomb potential |
| Radiative lifetime | $$\tau \propto n^3$$ | Transition dipole moments |
| Polarizability | $$\alpha \propto n^7$$ | Large electron displacement |
| Transition dipole | $$d_{n,n\pm1} \propto n^2$$ | Wavefunction overlap |

For $$n = 50$$:
- $$\langle r \rangle \approx 130$$ nm (huge compared to ground state ~0.5 nm)
- $$\tau \approx 100$$ μs (long radiative lifetime)
- Dipole moment $$d \approx 1000 \, ea_0$$ (enormous electric dipole)

### 1.2 Rydberg-Rydberg Interactions

**van der Waals Interaction:**

Two atoms in Rydberg states $$|rr\rangle$$ experience a strong interaction due to their large electric dipole moments. At long range, the interaction is:

$$V_{vdW}(R) = -\frac{C_6}{R^6}$$

The $$C_6$$ coefficient scales as:

$$C_6 \propto \frac{d^4}{\Delta E} \propto \frac{n^8}{n^{-3}} = n^{11}$$

For Rb $$|70S\rangle$$ state: $$C_6/h \approx 800$$ GHz·μm$$^6$$.

**Resonant Dipole-Dipole (Förster Resonance):**

When two Rydberg states are nearly degenerate with a pair state, the interaction becomes:

$$V_{dd}(R) = \frac{C_3}{R^3}$$

with $$C_3 \propto n^4$$. This occurs at Förster resonances where:

$$|n\ell, n\ell\rangle \leftrightarrow |n'\ell', n''\ell''\rangle$$

have nearly equal energies. These can be tuned with electric fields.

### 1.3 The Rydberg Blockade

**Blockade Mechanism:**

When two atoms separated by distance $$R$$ are illuminated by a laser resonant with the ground-to-Rydberg transition:

$$|gg\rangle \xrightarrow{\Omega} |gr\rangle + |rg\rangle$$

If $$R$$ is small enough that $$V(R) > \hbar\Omega$$ (interaction exceeds laser linewidth), the doubly-excited state $$|rr\rangle$$ is shifted off-resonance and cannot be populated.

**Blockade Radius:**

The characteristic distance at which blockade occurs:

$$R_b = \left(\frac{|C_6|}{\hbar\Omega}\right)^{1/6}$$

For typical parameters ($$C_6/h = 100$$ GHz·μm$$^6$$, $$\Omega/2\pi = 1$$ MHz):

$$R_b = \left(\frac{100 \times 10^9}{1 \times 10^6}\right)^{1/6} \text{ μm} \approx 10 \text{ μm}$$

**Collective Enhancement:**

Within the blockade radius, $$N$$ atoms share a single Rydberg excitation. The collective state:

$$|W\rangle = \frac{1}{\sqrt{N}}\sum_{j=1}^{N} |g...r_j...g\rangle$$

has enhanced coupling: $$\Omega_{coll} = \sqrt{N}\Omega$$.

### 1.4 Rydberg Gates

**Controlled-Z Gate Protocol:**

For two atoms at positions with separation $$< R_b$$:

1. **Pulse 1:** $$\pi$$ pulse on atom 1: $$|1\rangle \rightarrow |r\rangle$$
2. **Pulse 2:** $$2\pi$$ pulse on atom 2 (only works if atom 1 not in $$|r\rangle$$)
3. **Pulse 3:** $$\pi$$ pulse on atom 1: $$|r\rangle \rightarrow |1\rangle$$

Truth table:
- $$|00\rangle \rightarrow |00\rangle$$ (no Rydberg excitation)
- $$|01\rangle \rightarrow -|01\rangle$$ (atom 2 does full $$2\pi$$)
- $$|10\rangle \rightarrow -|10\rangle$$ (atom 1 does two $$\pi$$ pulses)
- $$|11\rangle \rightarrow -|11\rangle$$ (blockade prevents atom 2's $$2\pi$$, phases accumulate)

Wait—this gives CZ up to single-qubit phases, which can be corrected.

**Native Multi-Qubit Gates:**

A major advantage of Rydberg systems is native multi-qubit gates. The CCZ (Toffoli) gate can be implemented directly using three-atom blockade.

---

## Part II: Neutral Atom Arrays

### 2.1 Optical Tweezer Arrays

**Dipole Trap Potential:**

A focused laser beam creates a conservative trapping potential:

$$U(\mathbf{r}) = -\frac{3\pi c^2}{2\omega_0^3}\frac{\Gamma}{\Delta}I(\mathbf{r})$$

where:
- $$\omega_0$$ is the atomic transition frequency
- $$\Gamma$$ is the natural linewidth
- $$\Delta$$ is the detuning from resonance
- $$I(\mathbf{r})$$ is the laser intensity

For red detuning ($$\Delta < 0$$), atoms are attracted to intensity maxima.

**Trap Parameters:**

Typical optical tweezer specifications:
- Wavelength: 850-1000 nm (far from resonance)
- Beam waist: 0.5-1 μm
- Trap depth: 1-10 mK
- Trap frequencies: $$\omega_r/2\pi \sim 100$$ kHz radial, $$\omega_z/2\pi \sim 20$$ kHz axial

**Array Generation:**

Large arrays are created using:
1. **Spatial Light Modulators (SLM):** Holographic beam shaping
2. **Acousto-Optic Deflectors (AOD):** Time-multiplexed positioning
3. **Microlens arrays:** Fixed geometry

Current systems achieve >1000 trap sites with ~50% loading efficiency per site.

### 2.2 Atom Rearrangement

**Stochastic Loading Problem:**

Atoms load stochastically from a MOT with probability ~50% per site. To achieve defect-free arrays:

1. **Image the array:** Fluorescence detection of occupied sites
2. **Identify defects:** Determine which sites are empty
3. **Sort atoms:** Move atoms from reservoir to fill gaps

**Rearrangement Techniques:**

- **Static rearrangement:** Move tweezers with AODs to new positions
- **Dynamic rearrangement:** Transport atoms between fixed sites
- **Assembly time:** Scales as $$O(N)$$ for $$N$$ atoms

**Recent Advances:**

- >98% filling efficiency demonstrated
- 6000+ atom arrays (QuEra, 2024)
- 3D arrays for increased connectivity

### 2.3 Qubit Encoding and Operations

**Ground-State Qubits:**

Hyperfine states of alkali atoms:
- $$|0\rangle = |F=1, m_F=0\rangle$$
- $$|1\rangle = |F=2, m_F=0\rangle$$

These clock states have:
- T2*: ~1 ms (limited by trap inhomogeneity)
- T2 (with echo): ~100 ms
- Long-term storage fidelity: >99.9%

**Single-Qubit Gates:**

Implemented via:
1. **Microwave pulses:** Global addressing, high fidelity
2. **Raman transitions:** Individual addressing with focused beams
3. **Gate times:** ~1-10 μs

**Two-Qubit Gates:**

The Rydberg blockade enables CZ gates:
1. Transfer $$|1\rangle \rightarrow |r\rangle$$ on both atoms
2. Blockade shift creates conditional phase
3. Return to ground state

Gate fidelities: 97.5% (2024), improving rapidly.

### 2.4 Readout and Mid-Circuit Measurement

**Fluorescence Detection:**

1. Push out atoms in $$|1\rangle$$ (or $$|0\rangle$$) with resonant light
2. Image remaining atoms
3. Presence/absence indicates qubit state

**Mid-Circuit Measurement:**

Recent demonstrations show:
- Non-destructive readout of subset of atoms
- Conditional operations based on measurement
- Enables error correction protocols

---

## Part III: Bosonic Quantum Error Correction

### 3.1 Gottesman-Kitaev-Preskill (GKP) Codes

**Motivation:**

A single bosonic mode (harmonic oscillator) has infinite-dimensional Hilbert space. By carefully encoding a qubit, we can exploit redundancy for error correction without needing many physical systems.

**GKP Code States:**

The ideal GKP code states are:

$$|0_L\rangle = \sum_{s=-\infty}^{\infty} |q = 2s\sqrt{\pi}\rangle$$
$$|1_L\rangle = \sum_{s=-\infty}^{\infty} |q = (2s+1)\sqrt{\pi}\rangle$$

In position space, these are combs of delta functions separated by $$2\sqrt{\pi}$$.

**Realistic GKP States:**

Physical states have finite energy, requiring Gaussian envelope:

$$|\tilde{0}_L\rangle \propto \sum_s e^{-\Delta^2 s^2\pi} |q = 2s\sqrt{\pi}\rangle$$

The squeezing parameter $$\Delta$$ determines the code quality.

**Error Correction:**

Small displacements in position or momentum are correctable:

1. **Syndrome measurement:** Measure $$\hat{q} \mod \sqrt{\pi}$$ and $$\hat{p} \mod \sqrt{\pi}$$
2. **Error identification:** Displacement $$< \sqrt{\pi}/2$$ is correctable
3. **Correction:** Apply compensating displacement

**Logical Operations:**

- Pauli X: Displacement by $$\sqrt{\pi}$$ in position
- Pauli Z: Displacement by $$\sqrt{\pi}$$ in momentum
- Hadamard: Rotation by $$\pi/2$$ in phase space
- CNOT: SUM gate between oscillators

### 3.2 Cat Codes

**Two-Component Cat States:**

$$|\mathcal{C}_\alpha^\pm\rangle = \mathcal{N}_\pm(|\alpha\rangle \pm |-\alpha\rangle)$$

where $$|\alpha\rangle$$ is a coherent state and $$\mathcal{N}_\pm$$ is normalization.

**Encoding:**

$$|0_L\rangle = |\mathcal{C}_\alpha^+\rangle, \quad |1_L\rangle = |\mathcal{C}_\alpha^-\rangle$$

**Biased Noise:**

The key feature of cat codes is biased noise:
- **Bit-flip (X error):** Requires transition between $$|\alpha\rangle$$ and $$|-\alpha\rangle$$ — suppressed exponentially in $$|\alpha|^2$$
- **Phase-flip (Z error):** Affects the relative phase — occurs at normal rate

$$\frac{\Gamma_X}{\Gamma_Z} \propto e^{-2|\alpha|^2}$$

For $$|\alpha|^2 = 8$$: bit-flip suppression by factor ~$$10^6$$.

### 3.3 Kerr-Cat Qubits

**Stabilization Hamiltonian:**

$$\hat{H} = -K\hat{a}^{\dagger 2}\hat{a}^2 + \epsilon_2(\hat{a}^{\dagger 2} + \hat{a}^2)$$

- $$K$$: Kerr nonlinearity (self-interaction)
- $$\epsilon_2$$: Two-photon drive strength

**Phase Space Picture:**

The Hamiltonian creates a double-well potential in phase space with minima at $$\pm\alpha$$. The cat states are the ground state manifold.

**Autonomous Stabilization:**

Adding two-photon dissipation:

$$\mathcal{L}[\rho] = \kappa_2\mathcal{D}[\hat{a}^2 - \alpha^2]\rho$$

continuously projects the state back to the code space.

**Gates:**

- Z rotation: Change relative phase of $$|\alpha\rangle$$ and $$|-\alpha\rangle$$
- X gate: Requires breaking the bit-flip protection (challenging)
- Bias-preserving CNOT: Demonstrated (Alice&Bob, 2024)

### 3.4 Comparison of Bosonic Codes

| Property | GKP | Cat (Kerr) |
|----------|-----|------------|
| Encoding | Position grid states | Coherent state superposition |
| Error model | Corrects small displacements | Biased noise (suppressed X) |
| State preparation | Challenging (requires squeezing) | Easier (two-photon drive) |
| Logical gates | Gaussian operations | Bias-preserving subset |
| Hardware | Superconducting cavities, trapped ions | Superconducting circuits |
| Status | Demonstrated break-even | Demonstrated bit-flip suppression |

---

## Part IV: Photonic Quantum Computing

### 4.1 Photonic Qubit Encodings

**Polarization Encoding:**

$$|0\rangle = |H\rangle, \quad |1\rangle = |V\rangle$$

- Natural for free-space and fiber transmission
- Easy single-qubit gates (waveplates)
- Challenging two-qubit gates

**Dual-Rail (Path) Encoding:**

$$|0\rangle = |1,0\rangle, \quad |1\rangle = |0,1\rangle$$

One photon in one of two spatial modes.
- Natural for integrated photonics
- Beam splitters implement rotations

**Time-Bin Encoding:**

$$|0\rangle = |early\rangle, \quad |1\rangle = |late\rangle$$

Photon arrives in one of two time slots.
- Good for fiber transmission
- Interferometric stability required

### 4.2 Linear Optical Quantum Computing

**The Challenge:**

Photons do not interact directly. Beam splitters and phase shifters only implement single-qubit gates. How do we create entanglement?

**KLM Protocol (Knill-Laflamme-Milburn, 2001):**

Key insight: Measurement-induced nonlinearity.

**Conditional Sign Flip (CZ) Gate:**

1. Interfere two photons on beam splitters
2. Detect ancilla photons
3. Specific detection pattern "heralds" successful gate
4. Success probability: ~1/16 (can be boosted)

**Resource Overhead:**

Each entangling gate requires:
- Multiple ancilla photons
- Multiple detectors
- Classical feed-forward
- Repeat until success (or use teleportation)

**Boosted Gates:**

Using entangled ancilla states, success probability can approach 1, but at cost of more resources.

### 4.3 Measurement-Based Quantum Computing

**Cluster States:**

A 2D array of qubits entangled via CZ gates:

$$|cluster\rangle = \prod_{\langle i,j\rangle} CZ_{ij} |+\rangle^{\otimes N}$$

**Computation by Measurement:**

1. Prepare cluster state (resource state)
2. Measure qubits in sequence with chosen angles
3. Measurement outcomes determine the computation
4. Adaptive angles correct for measurement randomness

**Advantages for Photonics:**

- Entangling gates needed only for state preparation
- Computation is single-qubit measurements (easy for photons)
- Naturally fault-tolerant structure

**Fusion-Based QC:**

Modern approach: Generate small entangled states and "fuse" them:
1. Create Bell pairs or small clusters
2. Fusion operations (partial BSM) connect them
3. Build up large cluster states probabilistically
4. More practical than direct KLM

### 4.4 Continuous-Variable Photonics

**Squeezed States:**

Instead of single photons, use Gaussian states:
- Squeezed vacuum: $$|\xi\rangle = S(\xi)|0\rangle$$
- Reduced noise in one quadrature, increased in conjugate

**Gaussian Boson Sampling (GBS):**

- Input squeezed states into linear optical network
- Measure output photon numbers
- Classically hard to simulate (claimed quantum advantage)
- Xanadu Borealis demonstration (2022)

**CV Cluster States:**

Continuous-variable analog of cluster states:
- Entangled squeezed modes
- Measurement-based CV computation
- Potentially deterministic gate operations

### 4.5 Photonic Hardware

**Integrated Photonics:**

- Silicon photonics: CMOS compatible, scalable
- Silicon nitride: Low loss, good for routing
- Lithium niobate: Fast modulators, nonlinear effects

**Photon Sources:**

- Spontaneous parametric down-conversion (SPDC)
- Four-wave mixing in waveguides
- Quantum dots
- Challenge: High-efficiency, indistinguishable sources

**Detectors:**

- Superconducting nanowire single-photon detectors (SNSPDs)
- >95% efficiency, <50 ps timing jitter
- Require cryogenic cooling (~2K)

**Companies:**

- **Xanadu:** CV photonics, Borealis
- **PsiQuantum:** Fusion-based, silicon photonics
- **Quandela:** Photon sources, quantum dots
- **ORCA Computing:** Memory-based photonic QC

---

## Part V: Platform Comparison

### 5.1 Neutral Atoms vs Other Platforms

| Metric | Neutral Atoms | Superconducting | Trapped Ions |
|--------|---------------|-----------------|--------------|
| Qubit count | 100-6000 | 100-1000 | 10-50 |
| Two-qubit fidelity | 99.5% | 99.5% | 99.9% |
| T1 | Hours (ground state) | 100-500 μs | Minutes |
| T2 | ~100 ms | 50-200 μs | Seconds |
| Connectivity | Reconfigurable | Fixed NN | All-to-all |
| Native gates | CCZ, Toffoli | CZ, iSWAP | MS (XX) |
| Temperature | ~10 μK | ~10 mK | ~1 mK |

### 5.2 Unique Advantages

**Neutral Atoms:**
- Massive parallelism (identical atoms, optical addressing)
- Reconfigurable connectivity
- Native multi-qubit gates
- Long coherence in ground state

**Photonics:**
- Room temperature operation (except detectors)
- Natural for networking/communication
- No decoherence for stored photons
- Modular architecture

**Bosonic Codes:**
- Hardware efficiency (one cavity = one logical qubit)
- Tailored error correction (biased noise)
- Potential path to fault tolerance with fewer physical systems

---

## Summary and Exam Preparation

### Key Derivations to Master

1. **Blockade radius:** From $$V(R_b) = \hbar\Omega$$
2. **C6 scaling:** From perturbation theory of dipole-dipole interaction
3. **GKP error correction:** Displacement detection and correction
4. **Cat code bit-flip suppression:** Overlap of coherent states

### Common Exam Questions

1. Calculate the Rydberg blockade radius for given parameters
2. Explain why neutral atoms can implement native CCZ gates
3. Compare GKP and cat codes for error correction
4. Discuss scalability challenges for photonic QC

### Quick Reference Formulas

$$\boxed{R_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6}, \quad C_6 \propto n^{11}}$$

$$\boxed{|0_L\rangle_{GKP} \propto \sum_s |2s\sqrt{\pi}\rangle, \quad |1_L\rangle_{GKP} \propto \sum_s |(2s+1)\sqrt{\pi}\rangle}$$

$$\boxed{|\mathcal{C}_\alpha^\pm\rangle \propto |\alpha\rangle \pm |-\alpha\rangle, \quad \Gamma_X/\Gamma_Z \propto e^{-2|\alpha|^2}}$$

$$\boxed{\text{KLM CZ success} \approx 1/16 \text{ (basic), } \rightarrow 1 \text{ (with boosting)}}$$
