# Week 133: Photonic Quantum Computing

## Overview

This week explores photonic approaches to quantum computing, where quantum information is encoded in the quantum states of light. Photonic systems offer unique advantages including room-temperature operation, natural connectivity through optical fibers, and inherent resistance to certain types of decoherence. We cover both discrete-variable (single-photon) and continuous-variable (field quadrature) approaches.

## Week Structure

| Day | Topic | Focus Areas |
|-----|-------|-------------|
| 925 | Linear Optical QC Fundamentals | Beam splitters, phase shifters, Hong-Ou-Mandel effect |
| 926 | KLM Protocol | Non-deterministic CNOT, measurement-induced nonlinearity |
| 927 | Boson Sampling | Computational complexity, Gaussian boson sampling |
| 928 | Continuous Variable QC | Quadrature operators, Gaussian states, squeezing |
| 929 | GKP Encoding | Grid states, phase-space error correction |
| 930 | Cat States and Bosonic Codes | Coherent state superpositions, Kerr nonlinearity |
| 931 | Integrated Photonics | Silicon photonics, chip-scale implementations |

## Learning Objectives

By the end of this week, you will be able to:

1. Derive transformation matrices for linear optical elements and multi-mode interferometers
2. Explain the KLM protocol and calculate success probabilities for non-deterministic gates
3. Analyze the computational complexity of boson sampling and its implications
4. Work with continuous-variable quantum states using quadrature operators
5. Understand GKP encoding and bosonic error correction in phase space
6. Evaluate current photonic quantum computing platforms and their trade-offs

## Key Concepts

### Discrete-Variable Photonic QC
- **Dual-rail encoding**: $|0_L\rangle = |1,0\rangle$, $|1_L\rangle = |0,1\rangle$
- **Linear optical elements**: Beam splitters, phase shifters, polarization rotators
- **Photon detection**: Single-photon avalanche diodes, transition-edge sensors

### Continuous-Variable QC
- **Quadrature operators**: $\hat{q} = (\hat{a} + \hat{a}^\dagger)/\sqrt{2}$, $\hat{p} = i(\hat{a}^\dagger - \hat{a})/\sqrt{2}$
- **Gaussian states**: Coherent, squeezed, thermal states
- **Non-Gaussian resources**: Cat states, GKP states, photon-added states

### Key Transformations

**Beam Splitter Unitary:**
$$\hat{U}_{BS}(\theta, \phi) = \exp\left[i\theta(e^{i\phi}\hat{a}^\dagger\hat{b} + e^{-i\phi}\hat{a}\hat{b}^\dagger)\right]$$

**Squeezing Operator:**
$$\hat{S}(r) = \exp\left[\frac{r}{2}(\hat{a}^2 - \hat{a}^{\dagger 2})\right]$$

**GKP Logical States:**
$$|0_L\rangle = \sum_{n=-\infty}^{\infty} |2n\sqrt{\pi}\rangle_q, \quad |1_L\rangle = \sum_{n=-\infty}^{\infty} |(2n+1)\sqrt{\pi}\rangle_q$$

## Prerequisites

- Week 130-132: Quantum error correction fundamentals
- Quantum optics basics (coherent states, Fock states)
- Gaussian integrals and phase space methods
- Python: numpy, scipy, matplotlib

## Computational Tools

This week uses Python extensively for:
- Fock state calculations and transformations
- Wigner function visualization
- Beam splitter and interferometer simulation
- Boson sampling simulation (small instances)

## References

1. Knill, Laflamme, Milburn - "A scheme for efficient quantum computation with linear optics" (2001)
2. Aaronson, Arkhipov - "The Computational Complexity of Linear Optics" (2011)
3. Gottesman, Kitaev, Preskill - "Encoding a qubit in an oscillator" (2001)
4. O'Brien et al. - "Photonic quantum technologies" (Nature Photonics, 2009)
5. Bourassa et al. - "Blueprint for a Scalable Photonic Fault-Tolerant Quantum Computer" (2021)

## Industry Connections

- **Xanadu**: Gaussian boson sampling with Borealis (2022)
- **PsiQuantum**: Fault-tolerant photonic quantum computing
- **Quandela**: Single-photon sources and linear optical QC
- **QuiX Quantum**: Universal photonic processors

## Assessment

- Problem sets on linear optical transformations
- Computational lab: Wigner function visualization
- Analysis project: Compare photonic vs. matter-based qubits
