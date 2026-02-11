# Week 178: Neutral Atoms & Photonics

## Overview

**Days:** 1240-1246
**Theme:** Emerging and alternative quantum computing platforms

This week covers neutral atom and photonic quantum computing, two platforms with unique advantages for scalability and specific applications. Neutral atoms offer massive parallelism through optical tweezer arrays, while photonic systems enable room-temperature operation and natural networking capabilities.

## Daily Schedule

| Day | Date (Day #) | Topic | Focus |
|-----|--------------|-------|-------|
| Monday | 1240 | Rydberg Physics | Atomic structure, Rydberg states, van der Waals interactions |
| Tuesday | 1241 | Rydberg Gates | Blockade mechanism, CZ gates, pulse sequences |
| Wednesday | 1242 | Neutral Atom Arrays | Optical tweezers, atom sorting, QuEra architecture |
| Thursday | 1243 | Bosonic Codes | GKP qubits, grid states, error correction |
| Friday | 1244 | Cat Qubits | Kerr-cat encoding, bias-preserving gates |
| Saturday | 1245 | Photonic QC | Linear optical QC, KLM protocol, measurement-based QC |
| Sunday | 1246 | Review & Integration | Cross-platform comparison, oral practice |

## Learning Objectives

By the end of this week, you will be able to:

1. **Calculate** Rydberg interaction strengths and blockade radii
2. **Design** two-qubit gate sequences using Rydberg blockade
3. **Explain** optical tweezer array operation and atom rearrangement
4. **Describe** GKP qubit encoding and its error correction properties
5. **Analyze** cat qubit biased noise and gate implementations
6. **Understand** linear optical quantum computing and the KLM protocol
7. **Compare** these platforms with superconducting and trapped ion systems

## Key Concepts

### Rydberg Atoms

**Rydberg States:**
Highly excited atomic states with principal quantum number $$n \gg 1$$. Key scalings:

| Property | Scaling | n=50 value |
|----------|---------|------------|
| Orbital radius | $$\propto n^2$$ | ~130 nm |
| Binding energy | $$\propto n^{-2}$$ | ~5 meV |
| Radiative lifetime | $$\propto n^3$$ | ~100 μs |
| Polarizability | $$\propto n^7$$ | ~GHz/(V/cm)² |
| van der Waals C6 | $$\propto n^{11}$$ | ~THz·μm⁶ |

**van der Waals Interaction:**

$$V(R) = -\frac{C_6}{R^6}$$

where $$C_6$$ depends strongly on the Rydberg state and can be attractive or repulsive.

**Rydberg Blockade:**
When two atoms are within the blockade radius:

$$R_b = \left(\frac{|C_6|}{\hbar\Omega}\right)^{1/6}$$

the interaction energy exceeds the laser linewidth, preventing simultaneous excitation.

### Neutral Atom Arrays

**Optical Tweezers:**
Tightly focused laser beams create dipole traps for individual atoms:

$$U_{trap} = -\frac{1}{2}\alpha|\mathbf{E}|^2$$

where $$\alpha$$ is the atomic polarizability.

**Qubit Encoding:**
- Hyperfine ground states: $$|0\rangle = |F=1, m_F=0\rangle$$, $$|1\rangle = |F=2, m_F=0\rangle$$
- Clock states provide first-order magnetic field insensitivity

**Two-Qubit Gates:**
1. Excite control atom to Rydberg state
2. Blockade prevents target excitation if control is in $$|1\rangle$$
3. Conditional phase accumulation creates CZ gate

### Bosonic Codes

**Gottesman-Kitaev-Preskill (GKP) Code:**
Encodes a logical qubit in an oscillator using grid states:

$$|0_L\rangle \propto \sum_{s=-\infty}^{\infty} |2s\sqrt{\pi}\rangle_q$$
$$|1_L\rangle \propto \sum_{s=-\infty}^{\infty} |(2s+1)\sqrt{\pi}\rangle_q$$

where $$|x\rangle_q$$ are position eigenstates.

**Error Correction:**
- Small position/momentum displacements are correctable
- Measure displacement via ancilla coupling
- Correct with displacement operator

**Cat Codes:**
Encode in superpositions of coherent states:

$$|0_L\rangle \propto |\alpha\rangle + |-\alpha\rangle$$
$$|1_L\rangle \propto |\alpha\rangle - |-\alpha\rangle$$

**Kerr-Cat Qubit:**
Stabilized by two-photon drive and Kerr nonlinearity:

$$\hat{H} = -K\hat{a}^{\dagger 2}\hat{a}^2 + \epsilon_2(\hat{a}^{\dagger 2} + \hat{a}^2)$$

Creates a double-well potential in phase space with bit-flip suppression.

### Photonic Quantum Computing

**Linear Optical QC (KLM Protocol):**
- Single photons as qubits (polarization or path encoding)
- Beam splitters and phase shifters for single-qubit gates
- Entangling gates via measurement and feed-forward
- Probabilistic but heralded

**Measurement-Based QC:**
- Create large entangled cluster state
- Computation via single-qubit measurements
- Measurement angles determine the computation
- Fusion operations for scalability

## Hardware Specifications

### QuEra Aquila (2024-2025)
- 256 programmable qubits (atoms)
- Reconfigurable atom arrays
- Rydberg blockade radius: ~10 μm
- Single-shot readout fidelity: >99%
- Native multi-qubit gates (CCZ, Toffoli)

### Pasqal Systems
- 100+ qubit neutral atom processors
- 3D array capability
- Analog and digital modes
- T2*: ~1 ms for ground-state qubits

### Xanadu Borealis (Photonic)
- 216 squeezed-state sources
- Time-multiplexed architecture
- Gaussian boson sampling demonstrated
- Room temperature operation

### Alice&Bob (Cat Qubits)
- Kerr-cat qubit demonstrations
- Bit-flip times: ~10 seconds
- Phase-flip suppression via encoding
- Path to repetition code QEC

## Study Materials

### Required Reading
1. Browaeys & Lahaye, "Many-body physics with individually controlled Rydberg atoms" Nature Physics (2020)
2. Gottesman, Kitaev, Preskill, "Encoding a qubit in an oscillator" PRA (2001)
3. Kok et al., "Linear optical quantum computing with photonic qubits" Rev. Mod. Phys. (2007)

### Recent Papers
- Bluvstein et al., "Logical quantum processor based on reconfigurable atom arrays" Nature (2024)
- "Hardware-efficient quantum error correction via concatenated bosonic qubits" Nature (2025)

### Problem Set Focus
- Rydberg blockade calculations
- GKP state properties
- Photonic gate success probabilities

### Oral Exam Topics
- Explain the Rydberg blockade mechanism
- Derive the blockade radius
- Compare GKP and cat qubit error correction
- Discuss photonic QC scalability challenges

## Deliverables

1. **Review Guide** - Comprehensive theory summary (2000+ words)
2. **Problem Set** - 25-30 problems with solutions
3. **Oral Practice** - Common qualifying exam questions
4. **Self-Assessment** - Conceptual understanding checks

## Key Equations

$$\boxed{R_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6}}$$

$$\boxed{C_6 \propto n^{11}, \quad \tau_{Ryd} \propto n^3}$$

$$\boxed{|0_L\rangle_{GKP} \propto \sum_s |2s\sqrt{\pi}\rangle, \quad |1_L\rangle_{GKP} \propto \sum_s |(2s+1)\sqrt{\pi}\rangle}$$

$$\boxed{|\pm_L\rangle_{cat} \propto |\alpha\rangle \pm |-\alpha\rangle}$$

## Assessment Criteria

| Skill | Novice | Proficient | Expert |
|-------|--------|------------|--------|
| Rydberg Physics | Knows blockade exists | Calculates blockade radius | Designs gate sequences |
| Bosonic Codes | Understands encoding | Analyzes error correction | Compares code performance |
| Photonics | Knows KLM basics | Understands probabilistic gates | Analyzes resource overhead |
