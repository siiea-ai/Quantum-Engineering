# Week 160: Quantum Channels

## Overview

**Days:** 1114-1120
**Theme:** CPTP Maps, Kraus Operators, Standard Channels, Choi-Jamiolkowski Isomorphism

This week covers quantum channels - the most general description of quantum dynamics, including noise, decoherence, and measurement. Understanding channels is essential for quantum error correction, open quantum systems, and the qualifying examination.

## Daily Breakdown

### Day 1114 (Monday): CPTP Maps
- Definition of quantum channels
- Complete positivity: why it's required
- Trace preservation: probability conservation
- Examples: unitary evolution, measurement, partial trace

### Day 1115 (Tuesday): Kraus Operator Representation
- Operator-sum representation
- Kraus operators and completeness relation
- Derivation from system-environment model
- Non-uniqueness of Kraus representation

### Day 1116 (Wednesday): Standard Channels - Depolarizing
- Depolarizing channel definition
- Kraus operators for depolarizing noise
- Effect on Bloch sphere
- Channel capacity

### Day 1117 (Thursday): Standard Channels - Amplitude Damping
- Amplitude damping: energy relaxation (T1)
- Kraus operators derivation
- Physical interpretation: spontaneous emission
- Generalized amplitude damping

### Day 1118 (Friday): Phase Damping and Dephasing
- Phase damping (T2 processes)
- Dephasing channel
- Relationship to decoherence
- Comparison of noise models

### Day 1119 (Saturday): Choi-Jamiolkowski Isomorphism
- Channel-state duality
- Choi matrix construction
- Complete positivity from Choi matrix
- Applications and examples

### Day 1120 (Sunday): Review and Problem Session
- Comprehensive problem solving
- Oral exam practice
- Self-assessment

## Key Formulas

### CPTP Map Definition

A quantum channel $$\mathcal{E}$$ is a linear map that is:
- **Completely Positive (CP):** $$(\mathcal{E} \otimes \mathcal{I})(\rho) \geq 0$$ for all $$\rho \geq 0$$
- **Trace-Preserving (TP):** $$\text{Tr}(\mathcal{E}(\rho)) = \text{Tr}(\rho)$$

### Kraus Representation

$$\boxed{\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger}$$

with completeness: $$\boxed{\sum_k K_k^\dagger K_k = I}$$

### Depolarizing Channel

$$\boxed{\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)}$$

Or equivalently: $$\mathcal{E}_p(\rho) = (1-\frac{4p}{3})\rho + \frac{p}{3}I$$

Kraus operators:
$$K_0 = \sqrt{1-p}I, \quad K_1 = \sqrt{p/3}X, \quad K_2 = \sqrt{p/3}Y, \quad K_3 = \sqrt{p/3}Z$$

### Amplitude Damping Channel

$$\boxed{K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}}$$

Effect: $$|1\rangle \to |0\rangle$$ with probability $$\gamma$$

### Phase Damping Channel

$$\boxed{K_0 = \sqrt{1-\lambda}I, \quad K_1 = \sqrt{\lambda}|0\rangle\langle 0|, \quad K_2 = \sqrt{\lambda}|1\rangle\langle 1|}$$

Or equivalently:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$

### Choi Matrix

$$\boxed{J(\mathcal{E}) = (\mathcal{E} \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|)}$$

where $$|\Phi^+\rangle = \frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}|ii\rangle$$

**Theorem:** $$\mathcal{E}$$ is CP $$\Leftrightarrow$$ $$J(\mathcal{E}) \geq 0$$

### Channel from Choi Matrix

$$\boxed{\mathcal{E}(\rho) = d \cdot \text{Tr}_1[(I \otimes \rho^T)J(\mathcal{E})]}$$

## Learning Objectives

By the end of this week, you should be able to:

1. Define CPTP maps and explain why both properties are physical requirements
2. Construct Kraus operator representations for given channels
3. Derive Kraus operators from system-environment models
4. Compute the effect of depolarizing, amplitude damping, and phase damping channels
5. Construct the Choi matrix for a channel and use it to verify complete positivity
6. Transform between Kraus and Choi representations

## Files in This Week

| File | Description |
|------|-------------|
| `README.md` | This overview document |
| `Review_Guide.md` | Comprehensive theoretical review |
| `Problem_Set.md` | 26 qualifying exam-style problems |
| `Problem_Solutions.md` | Detailed worked solutions |
| `Oral_Practice.md` | Oral exam questions and answers |
| `Self_Assessment.md` | Diagnostic checklist and self-test |

## Prerequisites

Before starting this week, ensure mastery of:
- Density matrices (Week 157)
- Composite systems, partial trace (Week 158)
- Basic linear algebra: positive operators, tensor products

## References

1. Nielsen & Chuang, Chapter 8: Quantum noise and quantum operations
2. Preskill Notes, Chapter 3: Quantum dynamics
3. [CMU Quantum Channels Notes](https://quantum.phys.cmu.edu/QCQI/qitd412.pdf)
4. [ETH Zurich QIT Solutions](https://edu.itp.phys.ethz.ch/hs13/qit/sol08.pdf)
5. [Felix Leditzky - Quantum Channels](https://felixleditzky.info/teaching/ST23/Felix%20Leditzky%20-%20Math%20595%20Quantum%20channels.pdf)
