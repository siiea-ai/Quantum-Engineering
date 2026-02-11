# Week 177: Superconducting & Trapped Ion Systems

## Overview

**Days:** 1233-1239
**Theme:** Mastering the two most mature quantum computing platforms

This week provides comprehensive coverage of superconducting and trapped ion quantum computing platforms, focusing on the physics principles, gate implementations, and system architectures that appear on qualifying examinations.

## Daily Schedule

| Day | Date (Day #) | Topic | Focus |
|-----|--------------|-------|-------|
| Monday | 1233 | Transmon Fundamentals | Cooper pair box to transmon, Hamiltonian derivation |
| Tuesday | 1234 | Transmon Gates & Readout | Single-qubit gates, dispersive readout, CR gates |
| Wednesday | 1235 | Flux Qubits | Fluxonium, tunable couplers, frequency tuning |
| Thursday | 1236 | Trapped Ion Basics | Paul traps, motional modes, laser cooling |
| Friday | 1237 | Mølmer-Sørensen Gate | Bichromatic fields, spin-dependent forces |
| Saturday | 1238 | Ion Architectures | Linear chains, QCCD, photonic interconnects |
| Sunday | 1239 | Review & Integration | Cross-platform comparison, oral practice |

## Learning Objectives

By the end of this week, you will be able to:

1. **Derive** the transmon Hamiltonian from circuit quantization principles
2. **Calculate** transmon anharmonicity and explain the $$E_J/E_C$$ ratio design
3. **Explain** dispersive readout and calculate the dispersive shift $$\chi$$
4. **Describe** flux qubit operation including the double-well potential
5. **Derive** the Mølmer-Sørensen gate Hamiltonian and explain its robustness
6. **Compare** QCCD and linear chain architectures for trapped ions
7. **Analyze** trade-offs between superconducting and trapped ion platforms

## Key Concepts

### Superconducting Qubits

**The Josephson Junction:**
The fundamental nonlinear element enabling superconducting qubits. The Josephson relations:

$$I = I_c \sin\phi$$
$$V = \frac{\Phi_0}{2\pi}\frac{d\phi}{dt}$$

where $$\Phi_0 = h/2e$$ is the flux quantum.

**Cooper Pair Box Hamiltonian:**

$$\hat{H}_{CPB} = 4E_C(\hat{n} - n_g)^2 - E_J\cos\hat{\phi}$$

where:
- $$E_C = e^2/2C_\Sigma$$ is the charging energy
- $$E_J = I_c\Phi_0/2\pi$$ is the Josephson energy
- $$n_g = C_gV_g/2e$$ is the gate charge

**Transmon Regime:**
Operating at $$E_J/E_C \gg 1$$ (typically 50-100):
- Exponential suppression of charge noise: $$\propto e^{-\sqrt{8E_J/E_C}}$$
- Reduced anharmonicity: $$\alpha \approx -E_C$$
- Transition frequency: $$\omega_{01} \approx \sqrt{8E_JE_C} - E_C$$

### Trapped Ion Qubits

**Paul Trap Confinement:**
Oscillating RF fields create a pseudopotential:

$$\Phi_{pseudo} = \frac{qV_0^2}{4m\Omega_{RF}^2r_0^4}(x^2 + y^2)$$

Combined with DC endcaps for axial confinement.

**Motional Modes:**
For N ions in a linear chain, there are N axial modes with frequencies:

$$\omega_m = \omega_z\sqrt{\lambda_m}$$

where $$\lambda_m$$ are eigenvalues of the interaction matrix.

**Mølmer-Sørensen Interaction:**
Bichromatic laser fields create an effective spin-spin interaction:

$$\hat{H}_{eff} = \chi\hat{S}_x^2$$

where $$\hat{S}_x = \sum_j \hat{\sigma}_x^{(j)}$$.

## Hardware Specifications

### IBM Quantum Nighthawk (2025)
- 120 qubits
- 218 tunable couplers
- Gate depth: up to 5,000 two-qubit gates
- T1: ~500 μs

### Google Willow (2024)
- 105 qubits
- Average connectivity: 3.47
- T1: 100 μs (5x improvement over Sycamore)
- Below-threshold error correction demonstrated

### Quantinuum H-Series
- H2: 56 qubits
- All-to-all connectivity via QCCD
- Two-qubit gate fidelity: 99.9%
- T1/T2: seconds to minutes

## Study Materials

### Required Reading
1. Koch et al., "Charge-insensitive qubit design" Phys. Rev. A 76, 042319 (2007)
2. Krantz et al., "A quantum engineer's guide to superconducting qubits" (2019)
3. Bruzewicz et al., "Trapped-ion quantum computing" (2019)

### Problem Set Focus
- Circuit quantization and Hamiltonian derivation
- Gate fidelity calculations
- Noise analysis and decoherence

### Oral Exam Topics
- Explain why transmons use $$E_J/E_C \gg 1$$
- Derive the dispersive shift in circuit QED
- Explain MS gate robustness to motional heating

## Deliverables

1. **Review Guide** - Comprehensive theory summary (2000+ words)
2. **Problem Set** - 25-30 problems with solutions
3. **Oral Practice** - Common qualifying exam questions
4. **Self-Assessment** - Conceptual understanding checks

## Assessment Criteria

| Skill | Novice | Proficient | Expert |
|-------|--------|------------|--------|
| Hamiltonian Derivation | Can write down | Can derive from first principles | Can modify for novel circuits |
| Gate Analysis | Understands operation | Can calculate fidelity | Can design error mitigation |
| Platform Comparison | Lists differences | Analyzes trade-offs | Recommends for applications |
