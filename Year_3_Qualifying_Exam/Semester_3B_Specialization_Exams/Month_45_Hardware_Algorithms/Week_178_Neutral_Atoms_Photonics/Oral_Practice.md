# Week 178: Oral Exam Practice - Neutral Atoms & Photonics

## Overview

This document contains oral examination questions for neutral atom and photonic quantum computing platforms. Practice answering these questions aloud, timing yourself to 3-5 minutes per response.

---

## Section A: Rydberg Physics and Neutral Atoms (15 Questions)

### A1. Rydberg State Properties

**Question:** What makes Rydberg atoms special for quantum computing? Discuss their key properties and scaling laws.

**Key Points:**
- High principal quantum number ($$n \sim 50-100$$)
- Large orbital radius ($$\propto n^2$$)
- Long radiative lifetime ($$\propto n^3$$)
- Enormous polarizability ($$\propto n^7$$)
- Strong interactions ($$C_6 \propto n^{11}$$)
- Exquisite sensitivity to electric fields

**Follow-up:** How does the $$n^{11}$$ scaling of $$C_6$$ arise?

---

### A2. Rydberg Blockade Mechanism

**Question:** Explain the Rydberg blockade and how it enables two-qubit gates.

**Key Points:**
- Strong Rydberg-Rydberg interaction $$V(R) = C_6/R^6$$
- When $$V > \hbar\Omega$$, double excitation is off-resonant
- Blockade radius: $$R_b = (C_6/\hbar\Omega)^{1/6}$$
- Within blockade, only one excitation shared collectively
- Conditional dynamics enable CZ gates

**Follow-up:** What determines the optimal Rabi frequency for a gate?

---

### A3. CZ Gate Protocol

**Question:** Walk through the pulse sequence for a Rydberg CZ gate.

**Key Points:**
1. $$\pi$$ pulse on control: $$|1\rangle \rightarrow |r\rangle$$
2. $$2\pi$$ pulse on target (blocked if control in $$|r\rangle$$)
3. $$\pi$$ pulse on control: $$|r\rangle \rightarrow |1\rangle$$
- Phase accumulation gives CZ
- Global addressing possible with appropriate geometry

**Follow-up:** How is the CCZ (Toffoli) gate implemented natively?

---

### A4. Optical Tweezer Arrays

**Question:** How do optical tweezers trap and arrange neutral atoms?

**Key Points:**
- Dipole force from focused laser
- Red-detuned light creates attractive potential
- Intensity maximum = potential minimum
- SLMs and AODs create programmable arrays
- 1000+ sites demonstrated

**Follow-up:** What limits the trap depth and lifetime?

---

### A5. Atom Rearrangement

**Question:** Neutral atom arrays load stochastically. How do we achieve defect-free filling?

**Key Points:**
- ~50% loading probability per site
- Image array with fluorescence
- Identify empty sites
- Move atoms from reservoir using dynamic tweezers
- >98% filling achieved with feedback

**Follow-up:** What is the time cost of rearrangement?

---

### A6. Qubit Coherence

**Question:** Compare coherence times in neutral atom arrays to other platforms.

**Key Points:**
- Ground-state qubits: T1 ~ hours (no spontaneous emission)
- T2*: ~1 ms (limited by differential light shift)
- T2 (with echo): ~100 ms
- Clock states reduce magnetic sensitivity
- Rydberg states: T1 ~ 100 μs (radiative decay)

**Follow-up:** What limits T2 in neutral atom systems?

---

### A7. Scalability

**Question:** What are the scalability advantages and challenges for neutral atoms?

**Key Points:**

**Advantages:**
- Identical atoms (no fabrication variation)
- Optical addressing scales well
- 1000+ qubits demonstrated
- Reconfigurable connectivity

**Challenges:**
- Gate fidelity still improving
- Rydberg lifetime limits circuit depth
- Crosstalk at high density
- Laser power requirements

**Follow-up:** What qubit count do you expect by 2030?

---

### A8. Native Multi-Qubit Gates

**Question:** Explain why neutral atoms can implement native CCZ gates.

**Key Points:**
- Three atoms within mutual blockade radius
- All pairs interact
- Global pulse sequence creates three-body phase
- No decomposition needed
- Significant circuit depth reduction

**Follow-up:** What geometry constraints exist for this?

---

### A9. Error Sources

**Question:** What are the main error sources in Rydberg-based quantum computing?

**Key Points:**
- Spontaneous emission from Rydberg states
- Blockade imperfection (finite $$V/\Omega$$)
- Motional effects (position fluctuations)
- Laser intensity and phase noise
- State preparation errors
- Detection errors

**Follow-up:** Which error is currently dominant?

---

### A10. QuEra Architecture

**Question:** Describe the QuEra neutral atom quantum computer architecture.

**Key Points:**
- Rubidium atoms in optical tweezer array
- 256 programmable qubits (2024)
- Both analog and digital modes
- Reconfigurable geometry
- Native CCZ gates
- Cloud access available

**Follow-up:** What distinguishes QuEra from Pasqal's approach?

---

### A11. Analog vs Digital Mode

**Question:** Neutral atoms can operate in analog and digital modes. Explain the difference.

**Key Points:**

**Analog:**
- Global Rydberg drive
- Atoms evolve under Ising-like Hamiltonian
- Natural for optimization problems
- Continuous time evolution

**Digital:**
- Site-selective addressing
- Discrete gate operations
- Universal quantum computing
- Higher control overhead

**Follow-up:** When is analog mode preferred?

---

### A12. Recent Milestones

**Question:** Describe recent neutral atom breakthroughs (2024-2025).

**Key Points:**
- Harvard/QuEra: Logical quantum processor (2024)
- 48 logical qubits demonstrated
- Mid-circuit measurement and feed-forward
- Error-corrected operations
- 6000+ atom arrays

**Follow-up:** What error correction code was used?

---

### A13. Comparison to Trapped Ions

**Question:** Compare neutral atom and trapped ion platforms.

**Key Points:**

| Aspect | Neutral Atoms | Trapped Ions |
|--------|---------------|--------------|
| Qubit count | 100-1000+ | 10-50 |
| Gate fidelity | 97-99% | 99.9% |
| Connectivity | Reconfigurable | All-to-all |
| Native gates | CCZ | MS (XX) |
| T1 | Hours | Minutes |
| Operation temperature | ~10 μK | ~1 mK |

**Follow-up:** Which platform would you choose for a 100-qubit simulation?

---

### A14. Future Directions

**Question:** What advances are needed for neutral atoms to achieve fault tolerance?

**Key Points:**
- Higher two-qubit gate fidelity (need >99.5%)
- Better Rydberg state coherence
- Faster gate operations
- Improved atom loss rates
- Scalable mid-circuit measurement

**Follow-up:** What is the current bottleneck?

---

### A15. Application Suitability

**Question:** What applications are neutral atoms particularly suited for?

**Key Points:**
- Combinatorial optimization (native Ising model)
- Quantum simulation (2D materials, lattice models)
- Drug discovery and chemistry
- Machine learning applications
- Large-scale problems needing many qubits

**Follow-up:** Why are neutral atoms good for optimization?

---

## Section B: Bosonic Codes (10 Questions)

### B1. GKP Code Basics

**Question:** Explain the GKP (Gottesman-Kitaev-Preskill) code.

**Key Points:**
- Encodes qubit in harmonic oscillator
- Logical states are grid states in phase space
- $$|0_L\rangle$$: peaks at $$2n\sqrt{\pi}$$
- $$|1_L\rangle$$: peaks at $$(2n+1)\sqrt{\pi}$$
- Small displacements are correctable

**Follow-up:** What level of squeezing is needed for break-even?

---

### B2. GKP Error Correction

**Question:** How does error correction work in the GKP code?

**Key Points:**
- Errors = small displacements in position/momentum
- Syndrome measurement: $$\hat{q} \mod \sqrt{\pi}$$
- Reveals displacement (modulo grid spacing)
- Correction: apply inverse displacement
- Works for errors up to $$\sqrt{\pi}/2$$

**Follow-up:** How is syndrome measurement implemented?

---

### B3. Cat Qubit Encoding

**Question:** Explain cat qubit encoding and its biased noise property.

**Key Points:**
- $$|0_L\rangle \propto |\alpha\rangle + |-\alpha\rangle$$
- $$|1_L\rangle \propto |\alpha\rangle - |-\alpha\rangle$$
- Bit-flip requires $$|\alpha\rangle \leftrightarrow |-\alpha\rangle$$ tunneling
- Suppressed as $$e^{-2|\alpha|^2}$$
- Phase flips occur at normal rate
- Biased noise: $$\Gamma_X \ll \Gamma_Z$$

**Follow-up:** How is bias-preserving error correction designed?

---

### B4. Kerr-Cat Stabilization

**Question:** How is a Kerr-cat qubit stabilized?

**Key Points:**
- Hamiltonian: $$H = -K\hat{a}^{\dagger 2}\hat{a}^2 + \epsilon_2(\hat{a}^{\dagger 2} + \hat{a}^2)$$
- Kerr term creates self-interaction
- Two-photon drive creates double well
- Minima at $$\pm\alpha$$ in phase space
- Autonomous stabilization via engineered dissipation

**Follow-up:** What limits the achievable $$|\alpha|^2$$?

---

### B5. Comparison of Bosonic Codes

**Question:** Compare GKP and cat codes for quantum error correction.

**Key Points:**

| Aspect | GKP | Cat |
|--------|-----|-----|
| Error model | Small displacements | Biased (X suppressed) |
| Encoding | Grid states | Coherent superpositions |
| Preparation | Challenging | Easier |
| Gates | Gaussian operations | Bias-preserving |
| Hardware | Cavities, trapped ions | Superconducting circuits |

**Follow-up:** Which code is more promising for near-term devices?

---

### B6. Hardware for Bosonic Codes

**Question:** What physical systems implement bosonic codes?

**Key Points:**
- Superconducting microwave cavities (3D, coaxial)
- Coupled to transmon for control
- Trapped ion motional states
- Optical modes (challenging)
- Recent: GKP in trapped ions demonstrated

**Follow-up:** Why are superconducting cavities preferred?

---

### B7. Break-Even Results

**Question:** What is the "break-even" point for bosonic error correction, and has it been achieved?

**Key Points:**
- Break-even: logical qubit outlives physical components
- For GKP: logical lifetime > cavity lifetime
- Achieved in 2023-2024 with ~10 dB squeezing
- Cat codes: bit-flip times > 10 seconds
- Both approaching practical thresholds

**Follow-up:** What are the next milestones?

---

### B8. Logical Gates

**Question:** How are logical gates implemented in bosonic codes?

**Key Points:**

**GKP:**
- Gaussian operations (displacements, squeezing)
- Hadamard = rotation by π/2 in phase space
- Non-Clifford requires ancilla

**Cat:**
- Z rotation: phase on cavity
- X gate: requires breaking bias protection
- CNOT: demonstrated with bias preservation

**Follow-up:** What makes non-Clifford gates challenging?

---

### B9. Concatenation

**Question:** How can bosonic codes be concatenated with standard codes?

**Key Points:**
- Use bosonic codes as "physical qubits" for surface code
- Cat codes provide biased noise
- Repetition code handles remaining Z errors
- Significant reduction in overhead
- Nature 2025: demonstrated concatenated scheme

**Follow-up:** How does this compare to bare surface code?

---

### B10. Alice&Bob Approach

**Question:** Describe the Alice&Bob company's approach to cat qubits.

**Key Points:**
- Focus on Kerr-cat qubits
- Exploit biased noise structure
- Repetition code for Z errors (1D)
- Simpler than surface code (2D)
- Path to 10-100 logical qubits
- Demonstrated bias-preserving CNOT

**Follow-up:** What is their timeline for fault tolerance?

---

## Section C: Photonic Quantum Computing (10 Questions)

### C1. Photonic Qubit Encodings

**Question:** Describe the main ways to encode qubits in photons.

**Key Points:**
- Polarization: $$|H\rangle, |V\rangle$$
- Dual-rail (path): $$|1,0\rangle, |0,1\rangle$$
- Time-bin: $$|early\rangle, |late\rangle$$
- Frequency-bin
- Each has trade-offs for gates and transmission

**Follow-up:** Which encoding is best for integrated photonics?

---

### C2. Linear Optical Gates

**Question:** What gates are easy/hard with linear optics?

**Key Points:**
- Easy: All single-qubit gates (waveplates, phase shifters)
- Hard: Two-qubit entangling gates (need nonlinearity)
- Beam splitters only mix modes
- No direct photon-photon interaction

**Follow-up:** How does KLM solve this problem?

---

### C3. KLM Protocol

**Question:** Explain the KLM (Knill-Laflamme-Milburn) protocol.

**Key Points:**
- Use measurement to create effective nonlinearity
- Ancilla photons + detectors
- Specific detection patterns "herald" successful gate
- Basic success probability: ~1/16
- Can be boosted with more resources
- Teleportation-based improvement

**Follow-up:** Why is this called "measurement-induced nonlinearity"?

---

### C4. Measurement-Based QC

**Question:** Explain measurement-based quantum computing and its relevance to photonics.

**Key Points:**
- Prepare entangled cluster state
- Computation = sequence of single-qubit measurements
- Measurement angles encode the algorithm
- Adaptive: later angles depend on earlier outcomes
- Natural for photonics: measurements are easy

**Follow-up:** How is a cluster state prepared?

---

### C5. Fusion-Based QC

**Question:** What is fusion-based quantum computing?

**Key Points:**
- Modern approach to photonic QC
- Generate small entangled resource states
- "Fuse" them together with partial Bell measurements
- Build up large entangled states probabilistically
- More practical than direct KLM
- PsiQuantum's approach

**Follow-up:** What is the advantage over cluster state preparation?

---

### C6. Gaussian Boson Sampling

**Question:** What is Gaussian boson sampling and why is it significant?

**Key Points:**
- Input squeezed states into linear optical network
- Measure output photon numbers
- Output probability involves Hafnians (hard classically)
- Computational advantage claimed (Xanadu, 2022)
- Not universal QC, but demonstrates quantum advantage

**Follow-up:** What is the connection to combinatorial problems?

---

### C7. Photon Sources

**Question:** Discuss the challenges and solutions for single-photon sources.

**Key Points:**
- Need: high efficiency, indistinguishability, on-demand
- SPDC: probabilistic, good indistinguishability
- Quantum dots: near-deterministic, improving
- Multiplexing increases efficiency
- Current: ~50-90% efficiency, 99%+ indistinguishability

**Follow-up:** What is the state of the art for source efficiency?

---

### C8. Integrated Photonics

**Question:** Describe the role of integrated photonics in quantum computing.

**Key Points:**
- Waveguide-based circuits on chips
- Silicon photonics: CMOS compatible
- Silicon nitride: low loss
- Lithium niobate: fast modulators
- Enables scalable, stable interferometers

**Follow-up:** What are the main loss sources?

---

### C9. Scalability Challenges

**Question:** What are the main scalability challenges for photonic QC?

**Key Points:**
- Probabilistic gates require many attempts
- Photon loss accumulates
- Detector efficiency and timing
- Source multiplexing complexity
- Clock rate limitations
- Classical control overhead

**Follow-up:** How do companies like PsiQuantum plan to address these?

---

### C10. Room Temperature Operation

**Question:** Photonics is often cited as "room temperature" QC. Is this accurate?

**Key Points:**
- Photons themselves don't need cooling
- Sources can be room temperature (some)
- Detectors often need cryogenics (SNSPDs at ~2K)
- Electronics can be room temperature
- Hybrid: cold detectors, warm everything else
- Still simpler than millikelvin systems

**Follow-up:** What is the power consumption of photonic QC?

---

## Section D: Cross-Platform Questions (5 Questions)

### D1. Platform Selection

**Question:** You need to simulate a 2D lattice model with 100 sites. Which platform would you choose?

Consider:
- Neutral atoms: natural 2D geometry, analog mode
- Superconducting: many qubits but limited connectivity
- Trapped ions: high fidelity but fewer qubits
- Photonics: room temperature but complex

---

### D2. Networking

**Question:** Which platform is best suited for quantum networking?

Consider:
- Photons are natural carriers
- Matter qubits need transduction
- Neutral atoms and ions can emit photons
- Superconducting requires microwave-optical conversion

---

### D3. Error Correction

**Question:** Compare error correction strategies across platforms.

Consider:
- Surface code on superconducting
- Bosonic codes (cat, GKP)
- Color codes on neutral atoms
- Fusion-based fault tolerance for photonics

---

### D4. Timeline to Fault Tolerance

**Question:** Which platform will achieve fault tolerance first?

Consider:
- Current fidelities
- Qubit count requirements
- Error correction overhead
- Industry investment

---

### D5. Hybrid Approaches

**Question:** Discuss the potential for hybrid quantum systems.

Consider:
- Combine strengths of different platforms
- Photonic interconnects between matter qubits
- Bosonic codes on superconducting hardware
- Neutral atom processors with photonic links

---

## Oral Exam Tips for These Topics

1. **Know the numbers:** Blockade radii, coherence times, gate fidelities
2. **Draw phase space:** Essential for bosonic codes
3. **Compare honestly:** Acknowledge limitations of each platform
4. **Cite recent work:** Shows awareness of field progress
5. **Connect to applications:** Why does this platform matter?
